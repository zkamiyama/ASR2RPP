"""Command-line entry point; intentionally does not import Qt."""
import argparse
import json
from pathlib import Path
import sys
import threading
from .catalog import load_catalog, resolve_model
from .pipeline import Stage
from .timing import TimingSettings
from .preprocessing import Settings, run_job
from .queue_runner import run_queue


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    from .platforms import backends
    parser = argparse.ArgumentParser(prog='asr2rpp')
    sub = parser.add_subparsers(dest='command', required=True)
    models = sub.add_parser('models', help='list/install TOML model definitions')
    models.add_argument('action', choices=['list', 'install', 'verify'])
    models.add_argument('ids', nargs='*')
    models.add_argument('--json', dest='as_json', action='store_true', help='include the authoritative TOML path and hash')
    models.add_argument('--keep-source', action='store_true', help='keep original source checkpoints after a successful conversion')
    run = sub.add_parser('run', help='convert files through the shared stage scheduler')
    run.add_argument('files', nargs='+', type=Path)
    run.add_argument('--format', dest='formats', action='append', metavar='RPP,OTIO,JSON',
                     help='Output formats, comma-separated or repeatable; default: rpp')
    run.add_argument('--asr', default='whisper-base')
    run.add_argument('--diar', help='omit to disable diarization')
    run.add_argument('--align', help='forced-alignment model; select --timing auto or alignment')
    run.add_argument('--preprocess', help='optional audio.cpp vocal/background separation model')
    run.add_argument('--reference-audio', '--rpp-audio', dest='rpp_audio', choices=['original', 'processed'], default='original',
                     help='processed keeps a persistent WAV next to the exported files; requires --preprocess')
    run.add_argument('--output-dir', default='')
    run.add_argument('--same-directory', action='store_true')
    run.add_argument('--ffmpeg', default='')
    run.add_argument('--start', type=float, default=0)
    run.add_argument('--duration', type=float, default=0)
    run.add_argument('--threads', type=int, default=4)
    run.add_argument('--timing', choices=['auto', 'native', 'vad', 'alignment'], default='auto')
    run.add_argument('--vad-max-seconds', type=float, default=25)
    run.add_argument('--vad-threshold', type=float, default=0.5)
    run.add_argument('--vad-min-silence-ms', type=int, default=250)
    run.add_argument('--vad-before-alignment', action='store_true')
    run.add_argument('--speaker-source', choices=['auto','native','diarizer','none'],default='auto')
    run.add_argument('--queue-window-items',type=int,default=16)
    for stage in ['asr', 'diar', 'align', 'preprocess']:
        run.add_argument('--' + stage + '-device', default='auto', choices=backends())
        run.add_argument('--' + stage + '-exe', default='')
        run.add_argument('--' + stage + '-language', default=None)
        run.add_argument('--' + stage + '-params', default='{}', help='JSON object of scalar request parameters')
    runtimes = sub.add_parser('runtimes',help='inspect or explicitly register trusted runtime binaries')
    runtimes.add_argument('action',choices=['list','probe','register','rollback'])
    runtimes.add_argument('--provider',default='whisper_cpp')
    runtimes.add_argument('--device',default='cpu',choices=backends())
    runtimes.add_argument('--exe',default='')
    runtimes.add_argument('--trust',action='store_true',help='explicitly trust the selected executable code')
    convert = sub.add_parser('convert', help='Re-export an ASR2RPP JSON without running inference')
    convert.add_argument('input', type=Path)
    convert.add_argument('--format', dest='formats', action='append', metavar='RPP,OTIO,JSON')
    convert.add_argument('--output-dir', type=Path)
    sub.add_parser('doctor', help='verify packaged runtimes and model-converter dependencies')
    sub.add_parser('gui')
    args = parser.parse_args(argv)
    if args.command == 'convert':
        from .outputs import parse_formats, validate_document, write_document, FORMATS
        try:
            formats = parse_formats(args.formats)
            if args.input.stat().st_size > 1024**3:
                raise ValueError('Result JSON exceeds the 1 GiB input limit')
            document = json.loads(args.input.read_text(encoding='utf-8-sig'))
            validate_document(document)
            parent = (args.output_dir or args.input.parent).resolve()
            parent.mkdir(parents=True, exist_ok=True)
            for number in range(1, 100000):
                stem = args.input.stem + '_export' + ('' if number == 1 else f'_{number}')
                if not any((parent/(stem+'.'+key)).exists() for key in FORMATS):
                    output = parent/(stem+'.'+formats[0])
                    break
            else:
                raise ValueError('Too many output name collisions')
            for path in write_document(document, output, formats).values():
                print(path)
            return 0
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args.command == 'gui':
        from .gui_dcc import main as gui_main
        return gui_main()
    if args.command == 'runtimes':
        from . import runtime_registry as registry
        try:
            if args.action == 'list':
                value = registry.entries()
            elif args.action == 'probe':
                value = registry.probe(args.provider,args.device,args.exe)
            elif args.action == 'register':
                value = registry.register(args.provider,args.device,args.exe,trust=args.trust)
            else:
                registry.rollback(args.provider,args.device)
                value = registry.entries()
            print(json.dumps(value,ensure_ascii=False,indent=2))
            return 0
        except Exception as exc:
            print(str(exc),file=sys.stderr)
            return 2
    if args.command == 'doctor':
        failures = []
        try:
            import numpy
            print('numpy', numpy.__version__, 'OK')
        except Exception as exc:
            failures.append('numpy: ' + str(exc))
        try:
            import safetensors
            import safetensors.numpy as _safetensors_numpy
            print('safetensors', getattr(safetensors, '__version__', 'unknown'), 'OK')
        except Exception as exc:
            failures.append('safetensors: ' + str(exc))
        try:
            from .catalog import assets_root
            from .model_conversion import find_audio_cpp_tool
            converter = find_audio_cpp_tool('audiocpp_gguf', assets_root())
            import subprocess
            probe = subprocess.run([str(converter), '--help'], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=30)
            if probe.returncode:
                raise RuntimeError(f'converter exited {probe.returncode}')
            print('audiocpp_gguf', converter, 'OK')
        except Exception as exc:
            failures.append('audiocpp_gguf: ' + str(exc))
        try:
            from .adapters import executable, ffmpeg_path
            print('ffmpeg', ffmpeg_path(progress=lambda text: print(text, file=sys.stderr, flush=True)), 'OK')
            for runtime in ('whisper_cpp', 'audio_cpp'):
                path = executable(runtime, 'cpu')
                print(runtime + ':cpu', path, 'OK')
            from .vad import vad_executable
            print('whisper-vad-speech-segments', vad_executable(), 'OK')
        except Exception as exc:
            failures.append('native runtime: ' + str(exc))
        for failure in failures:
            print('FAIL', failure, file=sys.stderr)
        return 1 if failures else 0
    catalog, errors = load_catalog()
    for error in errors:
        print('Catalog warning:', error, file=sys.stderr)
    cancel = threading.Event()
    progress = lambda text: print(text, file=sys.stderr, flush=True)
    try:
        if args.command == 'models':
            if args.action == 'list':
                if args.as_json:
                    print(json.dumps([{'id': m.id, 'runtime': m.runtime, 'task': m.task,
                        'definition': str(m.definition), 'sha256': m.definition_sha256}
                        for m in catalog.values()], ensure_ascii=False))
                    return 0
                for model in catalog.values():
                    print(f'{model.id}\t{model.task}\t{model.runtime}')
            else:
                for model_id in args.ids:
                    if model_id not in catalog:
                        raise ValueError(f'Unknown model: {model_id}')
                    model = catalog[model_id]
                    resolve_model(model, cancel, progress, download=args.action == 'install', keep_source=args.keep_source)
                    from .inference_policy import policy_for
                    from .timing import native_timestamps
                    if model.task == 'asr' and not native_timestamps(model):
                        from .vad import vad_model
                        from .adapters import split_engine_parameters
                        parameters, _ = split_engine_parameters(model, None)
                        vad_model(model, parameters, cancel, progress)
            return 0
        def stage(task):
            model_id = getattr(args, task)
            if not model_id:
                return None
            if model_id not in catalog:
                raise ValueError(f'Unknown model: {model_id}')
            model = catalog[model_id]
            parameters = json.loads(getattr(args, task + '_params'))
            if not isinstance(parameters, dict):
                raise ValueError('Parameters must be a JSON object')
            language = getattr(args, task + '_language') or model.defaults.get('language', 'ja')
            return Stage(model_id, getattr(args, task + '_device'), getattr(args, task + '_exe'), language, args.threads, parameters)
        if args.same_directory and args.output_dir:
            raise ValueError('Choose --same-directory OR --output-dir, not both')
        from .outputs import parse_formats, output_paths
        formats = parse_formats(args.formats)
        settings = Settings(stage('asr'), stage('diar'), stage('align'), not bool(args.output_dir), args.output_dir,
                            args.ffmpeg, args.start, args.duration, stage('preprocess'), args.rpp_audio,
                            timing=TimingSettings(args.timing, args.vad_max_seconds, args.vad_threshold,
                                min_silence_ms=args.vad_min_silence_ms, segment_before_alignment=args.vad_before_alignment,
                                speaker_source=args.speaker_source),queue_window_items=args.queue_window_items,
                            output_formats=formats)
        settings.validate(catalog)
        from .providers import preflight
        from .timing import plan_for
        timing_plan = plan_for(catalog[settings.asr.model_id],settings.timing,settings.align is not None)
        for selected in (settings.preprocess,settings.asr,settings.align,settings.diar):
            if selected is not None:
                info = preflight(catalog[selected.model_id],selected,cancel,
                          segmented=selected is settings.asr and timing_plan.segmented)
                settings.runtime_provenance[catalog[selected.model_id].task] = {k:v for k,v in info.items() if k != 'capabilities'}
        events = []
        outputs = run_queue(list(enumerate(args.files)),settings,catalog,cancel,progress,
            lambda index,status,detail: events.append((index,status,detail)))
        for output in outputs.values():
            for path in output_paths(output, settings.output_formats).values():
                print(path)
        for index,status,detail in events:
            if status == '失敗':
                progress(f'{args.files[index]}: {detail}')
        return 0 if len(outputs) == len(args.files) else 1
    except KeyboardInterrupt:
        cancel.set()
        return 130
    except Exception as error:
        progress(str(error))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
