"""Bounded forced-alignment requests shared by single-file and queue execution."""
from dataclasses import dataclass, replace
from pathlib import Path
import json
import os
import shutil
from .adapters import (executable, split_engine_parameters, validate_model_parameter_constraints,
                       audio_session_args, run_process, parse_audio)
from .catalog import checkpoint
from .media import slice_pcm
from .transcript import has_alignable_text, clean_bounds


DEFAULT_ALIGNMENT_BATCH_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class AlignmentInput:
    key: str
    audio: Path
    begin: float
    length: float
    text: str


def chunks_by_size(items, path_of, max_bytes):
    if max_bytes <= 0:
        raise ValueError('Batch byte budget must be positive')
    chunk, used = [], 0
    for item in items:
        size = max(1, path_of(item).stat().st_size)
        if chunk and used + size > max_bytes:
            yield chunk
            chunk, used = [], 0
        chunk.append(item)
        used += size
    if chunk:
        yield chunk


def infer_requests(model, weights, stage, requests, root, cancel, progress,
                   max_bytes=DEFAULT_ALIGNMENT_BATCH_BYTES):
    """One model session per bounded batch; IDs preserve source/segment mapping.

    Empty timestamp output for speech still raises. Punctuation-only requests are
    filtered by the caller, never silently treated as successfully aligned words.
    """
    if not requests:
        return {}
    binary = executable(model.runtime, stage.device, stage.executable)
    parameters, session = split_engine_parameters(model, stage.parameters or {})
    validate_model_parameter_constraints(model, parameters, session)
    results = {}
    for number, chunk in enumerate(chunks_by_size(requests, lambda r: r.audio, max_bytes), 1):
        checkpoint(cancel)
        folder = Path(root) / f'chunk-{number}'
        inputs, out = folder / 'inputs', folder / 'out'
        inputs.mkdir(parents=True, exist_ok=True)
        out.mkdir(exist_ok=True)
        payload = {'requests': []}
        seen = set()
        for req in chunk:
            if not req.key.isascii() or not req.key.replace('_', '').isalnum() or req.key in seen:
                raise ValueError('Alignment request IDs must be unique ASCII identifiers')
            seen.add(req.key)
            target = inputs / f'{req.key}.wav'
            try:
                os.link(req.audio, target)
            except OSError:
                shutil.copyfile(req.audio, target)
            payload['requests'].append({
                'id': req.key, 'audio': f'inputs/{req.key}.wav', 'text': req.text,
                'language': stage.language or model.defaults.get('language', 'ja'),
                'options': parameters,
            })
        request_file = folder / 'requests.json'
        request_file.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        argv = [str(binary), '--task', 'align', '--family', model.family,
                '--model', str(weights), '--backend', 'best' if stage.device == 'auto' else stage.device,
                '--mode', 'offline', '--request-sequence', str(request_file),
                '--threads', str(stage.threads), '--words-out', str(out / 'words.json')]
        argv += audio_session_args(model, session)
        progress(f'Forced alignment — batch {number}, {len(chunk)} segments')
        run_process(argv, cancel, progress, folder / 'engine.log')
        for req in chunk:
            path = out / f'words_{req.key}.json'
            if not path.is_file():
                raise ValueError(f'Aligner did not produce timestamps for request {req.key}')
            raw = json.loads(path.read_text(encoding='utf-8-sig'))
            results[req.key] = parse_audio(raw, 'align', model.family, model.sample_rate)
        # Only the temporary input copies, not raw outputs/provenance, can go now.
        shutil.rmtree(inputs)
    return results


def align_segments(model, weights, stage, units, pcm, duration, root, cancel, progress, warnings):
    """Slice cached PCM once per segment, then align in reusable native sessions."""
    aligned, requests = [], []
    for index, segment in enumerate(units):
        checkpoint(cancel)
        if not has_alignable_text(segment.text):
            warnings.append(f'{segment.start:.3f}: forced alignment skipped punctuation-only text; ASR interval retained')
            aligned.append(segment)
            continue
        if segment.end - segment.start > 55:
            raise ValueError('Alignment needs <=55-second matched transcript/audio segments')
        begin = max(0.0, segment.start - 0.15)
        length = min(duration, segment.end + 0.25) - begin
        key = f's{index:06d}'
        file = Path(root) / 'slices' / f'{key}.wav'
        length = slice_pcm(pcm, file, begin, length, cancel)
        requests.append(AlignmentInput(key, file, begin, length, segment.text))
    results = infer_requests(model, weights, stage, requests, root, cancel, progress)
    for req in requests:
        local = clean_bounds(results[req.key].units, req.length, warnings)
        if not local:
            raise ValueError(f'Aligner returned no valid intervals for request {req.key}')
        aligned.extend(replace(u, start=u.start + req.begin, end=u.end + req.begin) for u in local)
    return clean_bounds(aligned, duration, warnings)
