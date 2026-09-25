from pathlib import Path
import base64
import struct
import threading
import zipfile
import pytest

from asr2rpp.model_conversion import (
    checkpoint_to_safetensors, parse_big_beta7_yaml, convert_model, ConversionError,
)
from asr2rpp.catalog import Model


PICKLE_B64 = "gAJ9cQAoWCAAAABiYW5kX3NwbGl0LnRvX2ZlYXR1cmVzLjAuMC5nYW1tYXEBY3RvcmNoLl91dGlscwpfcmVidWlsZF90ZW5zb3JfdjIKcQIoKFgHAAAAc3RvcmFnZXEDY3RvcmNoCkZsb2F0U3RvcmFnZQpxBFgBAAAAMHEFWAMAAABjcHVxBksCdHEHUUsASwKFcQhLAYVxCYljY29sbGVjdGlvbnMKT3JkZXJlZERpY3QKcQopUnELdHEMUnENWCcAAABtYXNrX2VzdGltYXRvcnMuMC50b19mcmVxcy4wLjAuMC53ZWlnaHRxDmgCKChoA2gEWAEAAAAxcQ9oBksEdHEQUUsASwJLAoZxEUsCSwGGcRKJaAopUnETdHEUUnEVWCMAAABsYXllcnMuMC4wLmxheWVycy4wLjAudG9fcWt2LndlaWdodHEWaAIoKGgDaARYAQAAADJxF2gGSxJ0cRhRSwBLBksDhnEZSwNLAYZxGoloCilScRt0cRxScR11Lg=="


def make_torch_zip(path):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as zf:
        zf.writestr('tiny/data.pkl', base64.b64decode(PICKLE_B64))
        zf.writestr('tiny/data/0', struct.pack('<2f', 1.0, 2.0))
        zf.writestr('tiny/data/1', struct.pack('<4f', 0.0, 1.0, 2.0, 3.0))
        zf.writestr('tiny/data/2', struct.pack('<18f', *map(float, range(18))))

YAML = """audio:
  chunk_size: 676935
  n_fft: 2048
  hop_length: 441
  num_channels: 2
  sample_rate: 44100
model:
  dim: 384
  depth: 8
  stereo: true
  num_stems: 1
  time_transformer_depth: 1
  freq_transformer_depth: 1
  num_bands: 60
  dim_head: 64
  heads: 8
  dim_freqs_in: 1025
  sample_rate: 44100
  stft_n_fft: 2048
  stft_hop_length: 441
  stft_win_length: 2048
  stft_normalized: false
  mask_estimator_depth: 2
  multi_stft_resolutions_window_sizes: !!python/tuple
  - 4096
inference:
  batch_size: 2
  dim_t: 1201
  num_overlap: 2
"""


def test_big_beta7_yaml_mapping(tmp_path):
    config = tmp_path / 'big_beta7.yaml'
    config.write_text(YAML, encoding='utf-8')
    data = parse_big_beta7_yaml(config)
    assert data['chunk_size'] == 676935
    assert data['sample_rate'] == 44100
    assert data['num_overlap'] == 2
    assert data['n_fft'] == 2048
    assert data['stereo'] is True



def test_torch_free_checkpoint_reader_and_qkv_split(tmp_path):
    st = pytest.importorskip('safetensors.numpy')
    checkpoint = tmp_path / 'tiny.ckpt'
    make_torch_zip(checkpoint)
    config = tmp_path / 'big_beta7.yaml'
    config.write_text(YAML, encoding='utf-8')
    output = tmp_path / 'out'
    info = checkpoint_to_safetensors(checkpoint, config, output, threading.Event(), lambda _: None)
    tensors = st.load_file(str(output / 'model.safetensors'))
    assert info['qkv_split_count'] == 1
    assert 'layers.0.0.layers.0.0.to_qkv.weight' not in tensors
    assert tensors['layers.0.0.layers.0.0.to_q.weight'].shape == (2, 3)
    assert tensors['layers.0.0.layers.0.0.to_k.weight'].shape == (2, 3)
    assert tensors['layers.0.0.layers.0.0.to_v.weight'].shape == (2, 3)


def test_invalid_checkpoint_is_rejected_without_pickle_execution(tmp_path):
    checkpoint = tmp_path / 'bad.ckpt'
    checkpoint.write_bytes(b'not a zip')
    config = tmp_path / 'big_beta7.yaml'
    config.write_text(YAML, encoding='utf-8')
    with pytest.raises(ConversionError, match='ZIP'):
        checkpoint_to_safetensors(checkpoint, config, tmp_path / 'out', threading.Event(), lambda _: None)



def test_convert_model_keeps_safetensors_alive_until_native_converter(tmp_path, monkeypatch):
    from asr2rpp import model_conversion as conversion

    directory = tmp_path / 'weights'
    temp_root = tmp_path / 'cache'
    directory.mkdir()
    checkpoint = directory / 'big_beta7.ckpt'
    checkpoint.write_bytes(b'checkpoint')
    (directory / 'big_beta7.yaml').write_text(YAML, encoding='utf-8')

    model = Model(
        id='mel-big-beta7',
        runtime='audio_cpp',
        task='sep',
        family='mel_band_roformer',
        source={
            'files': ['big_beta7.ckpt', 'big_beta7.yaml'],
            'entry': 'big_beta7-f16.gguf',
            'convert': {
                'kind': 'mel_band_roformer_ckpt_to_gguf',
                'checkpoint': 'big_beta7.ckpt',
                'config': 'big_beta7.yaml',
                'output': 'big_beta7-f16.gguf',
                'precision': 'f16',
                'keep_intermediate': False,
            },
        },
    )

    observed = {}

    def fake_checkpoint_to_safetensors(_checkpoint, _config, work, _cancel, _progress):
        work.mkdir(parents=True, exist_ok=True)
        intermediate = work / 'model.safetensors'
        intermediate.write_bytes(b'safetensors')
        (work / 'config.json').write_text('{}', encoding='utf-8')
        observed['work'] = work
        return {'tensor_count_source': 732, 'tensor_count_output': 748}

    monkeypatch.setattr(conversion, 'checkpoint_to_safetensors', fake_checkpoint_to_safetensors)
    monkeypatch.setattr(
        conversion, 'find_audio_cpp_tool',
        lambda name, _assets: tmp_path / (name + ('.exe' if conversion.sys.platform == 'win32' else '')),
    )

    def fake_run(argv, _cancel, _progress, timeout=7200):
        if '--output' in argv:
            weights = Path(next(value.split('=', 1)[1] for value in argv if value.startswith('weights=')))
            # Regression: assigning the .part GGUF Path to the TemporaryDirectory
            # variable used to destroy this directory before the converter opened it.
            assert weights.is_file()
            assert weights.parent == observed['work']
            partial = Path(argv[argv.index('--output') + 1])
            partial.write_bytes(b'GGUF')
            observed['partial'] = partial
            return 'converted'
        if Path(argv[0]).stem == 'audiocpp_gguf':
            return 'family: mel_band_roformer'
        return 'runtime inspect ok'

    monkeypatch.setattr(conversion, '_run', fake_run)

    output, info = convert_model(
        model, directory, tmp_path, temp_root, threading.Event(), lambda _: None)

    assert output == directory / 'big_beta7-f16.gguf'
    assert output.read_bytes() == b'GGUF'
    assert info['output_size'] == 4
    assert not observed['work'].exists()
    assert not observed['partial'].exists()
