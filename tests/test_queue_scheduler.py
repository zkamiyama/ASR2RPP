import hashlib
import io
import json
from pathlib import Path
import threading

from asr2rpp import catalog
from asr2rpp.catalog import Model, resolve_model
from asr2rpp.pipeline import Stage
from asr2rpp.queue_runner import _chunks_by_size, _chunks_for_command, _audio_batch, QueueJob


class FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.headers = {'Content-Length': str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_converted_source_checkpoint_is_removed_but_install_stays_valid(tmp_path, monkeypatch):
    checkpoint = b'checkpoint' * 256
    config = b'config-data' * 128
    expected = hashlib.sha256(checkpoint).hexdigest()

    def fake_urlopen(request, timeout=30):
        url = request.full_url
        if '/api/models/' in url:
            return FakeResponse(json.dumps({'sha': 'resolved-revision'}).encode())
        if url.endswith('/big.ckpt'):
            return FakeResponse(checkpoint)
        if url.endswith('/big.yaml'):
            return FakeResponse(config)
        raise AssertionError(url)

    monkeypatch.setattr(catalog, 'urlopen', fake_urlopen)
    monkeypatch.setattr(catalog, 'data_root', lambda: tmp_path / 'home')

    from asr2rpp import model_conversion

    def fake_convert(model, directory, assets_root, temp_root, cancel, progress):
        output = directory / 'converted.gguf'
        output.write_bytes(b'GGUF' + b'0' * 2048)
        return output, {'precision': 'f16'}

    monkeypatch.setattr(model_conversion, 'convert_model', fake_convert)

    model = Model(
        id='converted-test',
        runtime='audio_cpp',
        task='sep',
        family='mel_band_roformer',
        source={
            'repo': 'owner/model',
            'revision': 'pinned',
            'files': ['big.ckpt', 'big.yaml'],
            'entry': 'converted.gguf',
            'sha256': {'big.ckpt': expected},
            'convert': {
                'kind': 'mel_band_roformer_ckpt_to_gguf',
                'checkpoint': 'big.ckpt',
                'config': 'big.yaml',
                'output': 'converted.gguf',
                'precision': 'f16',
            },
        },
    )
    model.validate()
    path, state = resolve_model(model, threading.Event(), lambda _text: None, download=True)
    directory = path.parent
    assert path.is_file()
    assert not (directory / 'big.ckpt').exists()
    assert (directory / 'big.yaml').exists()
    assert state['files']['big.ckpt']['retained'] is False
    assert state['conversion']['source_checkpoint_retained'] is False

    # Missing intentionally-removed source files must not make the installed
    # converted model appear broken on the next normal run.
    second, second_state = resolve_model(model, threading.Event(), lambda _text: None)
    assert second == path
    assert second_state['files']['big.ckpt']['retained'] is False

    # Enabling the advanced retention setting later explicitly restores the
    # source checkpoint instead of silently treating the old state as enough.
    third, third_state = resolve_model(
        model, threading.Event(), lambda _text: None, download=True, keep_source=True)
    assert third == path
    assert (directory / 'big.ckpt').is_file()
    assert third_state['conversion']['source_checkpoint_retained'] is True


def test_keep_source_retains_checkpoint(tmp_path, monkeypatch):
    checkpoint = b'checkpoint' * 256
    config = b'config-data' * 128
    expected = hashlib.sha256(checkpoint).hexdigest()

    def fake_urlopen(request, timeout=30):
        if '/api/models/' in request.full_url:
            return FakeResponse(json.dumps({'sha': 'resolved-revision'}).encode())
        return FakeResponse(checkpoint if request.full_url.endswith('/big.ckpt') else config)

    monkeypatch.setattr(catalog, 'urlopen', fake_urlopen)
    monkeypatch.setattr(catalog, 'data_root', lambda: tmp_path / 'home')
    from asr2rpp import model_conversion

    def fake_convert(model, directory, assets_root, temp_root, cancel, progress):
        output = directory / 'converted.gguf'
        output.write_bytes(b'GGUF' + b'0' * 2048)
        return output, {}

    monkeypatch.setattr(model_conversion, 'convert_model', fake_convert)
    model = Model(
        id='keep-test', runtime='audio_cpp', task='sep', family='mel_band_roformer',
        source={
            'repo': 'owner/model', 'files': ['big.ckpt', 'big.yaml'], 'entry': 'converted.gguf',
            'sha256': {'big.ckpt': expected},
            'convert': {'kind': 'mel_band_roformer_ckpt_to_gguf', 'checkpoint': 'big.ckpt',
                        'config': 'big.yaml', 'output': 'converted.gguf', 'precision': 'f16'},
        })
    path, state = resolve_model(model, threading.Event(), lambda _text: None,
                                download=True, keep_source=True)
    assert (path.parent / 'big.ckpt').is_file()
    assert state['conversion']['source_checkpoint_retained'] is True


def test_audio_batch_chunking_respects_size_budget(tmp_path):
    class Item:
        def __init__(self, name, size):
            self.key = name
            self.path = tmp_path / name
            self.path.write_bytes(b'x' * size)

    items = [Item('a.wav', 60), Item('b.wav', 60), Item('c.wav', 20)]
    chunks = list(_chunks_by_size(items, lambda x: x.path, 100))
    assert [[x.key for x in chunk] for chunk in chunks] == [
        ['a.wav'], ['b.wav', 'c.wav']
    ]


def test_whisper_command_chunking_caps_item_count(tmp_path):
    class Item:
        def __init__(self, n):
            self.key = str(n)
            self.path = tmp_path / f'{n}.wav'

    items = [Item(i) for i in range(7)]
    chunks = list(_chunks_for_command(items, lambda x: x.path, max_chars=99999, max_items=3))
    assert [len(chunk) for chunk in chunks] == [3, 3, 1]



def test_nemotron_stage_major_asr_uses_streaming_per_file(tmp_path, monkeypatch):
    from asr2rpp import queue_runner

    model = Model(
        id='nemotron-asr', runtime='audio_cpp', task='asr',
        family='nemotron_asr', source={'path': 'unused.gguf'},
        defaults={'language': 'ja-JP'}, sample_rate=16000,
    )
    weights = tmp_path / 'model.gguf'
    weights.write_bytes(b'GGUF')
    audio_paths = {}
    jobs = []
    for index in range(2):
        audio = tmp_path / f'a{index}.wav'
        audio.write_bytes(b'WAV' + bytes([index]))
        audio_paths[f'q{index}'] = audio
        report = tmp_path / f'report-{index}'
        report.mkdir()
        jobs.append(QueueJob(
            index=index, key=f'q{index}', source=tmp_path / f'source-{index}.wav',
            output=tmp_path / f'out-{index}.rpp', report=report,
            source_sha256='x', manifest={},
        ))

    commands = []
    monkeypatch.setattr(queue_runner, 'executable', lambda *_args, **_kwargs: tmp_path / 'audiocpp_cli.exe')

    def fake_run(argv, _cancel, _progress, log, timeout=7200):
        commands.append(list(argv))
        out = Path(argv[argv.index('--words-out') + 1])
        out.write_text(json.dumps({
            'words': [{'start': 0.0, 'end': 0.32, 'word': 'x'}]
        }), encoding='utf-8')
        Path(log).write_text('streaming ok', encoding='utf-8')

    monkeypatch.setattr(queue_runner, 'run_process', fake_run)
    statuses = []
    result = _audio_batch(
        model, weights, Stage('nemotron-asr', device='vulkan', language='ja-JP', threads=4),
        jobs, audio_paths, tmp_path / 'batch', threading.Event(), lambda _text: None,
        lambda i, status, detail: statuses.append((i, status, detail)),
        task='asr', max_bytes=1024,
    )

    assert set(result) == {'q0', 'q1'}
    assert len(commands) == 2
    assert all(command[command.index('--mode') + 1] == 'streaming' for command in commands)
    assert all('--batch-audio-dir' not in command for command in commands)
    assert all('--words-out' in command for command in commands)
    assert [status for _, status, _ in statuses] == ['ASR', 'ASR']
