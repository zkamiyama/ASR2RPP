"""Command-line entry point; intentionally does not import Qt."""
import argparse
import json
from pathlib import Path
import sys
import threading
from .catalog import load_catalog, resolve_model
from .pipeline import Stage
from .preprocessing import Settings, run_job


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(prog='asr2rpp')
    sub = parser.add_subparsers(dest='command', required=True)
    models = sub.add_parser('models', help='list/install TOML model definitions')
    models.add_argument('action', choices=['list', 'install'])
    models.add_argument('ids', nargs='*')
    run = sub.add_parser('run', help='convert files sequentially to non-destructive RPP')
    run.add_argument('files', nargs='+', type=Path)
    run.add_argument('--asr', default='whisper-base')
    run.add_argument('--diar', help='omit to disable diarization')
    run.add_argument('--align', help='omit to disable forced alignment')
    run.add_argument('--preprocess', help='optional audio.cpp vocal/background separation model')
    run.add_argument('--rpp-audio', choices=['original', 'processed'], default='original',
                     help='processed keeps a persistent WAV next to the RPP; requires --preprocess')
    run.add_argument('--output-dir', default='')
    run.add_argument('--same-directory', action='store_true')
    run.add_argument('--ffmpeg', default='')
    run.add_argument('--start', type=float, default=0)
    run.add_argument('--duration', type=float, default=0)
    run.add_argument('--threads', type=int, default=4)
    for stage in ['asr', 'diar', 'align', 'preprocess']:
        run.add_argument('--' + stage + '-device', default='cpu', choices=['cpu', 'vulkan', 'metal', 'cuda', 'auto'])
        run.add_argument('--' + stage + '-exe', default='')
        run.add_argument('--' + stage + '-language', default=None)
        run.add_argument('--' + stage + '-params', default='{}', help='JSON object of scalar request parameters')
    sub.add_parser('doctor', help='verify packaged runtimes and model-converter dependencies')
    sub.add_parser('gui')
    args = parser.parse_args(argv)
    if args.command == 'gui':
        from .gui_preprocessing import main as gui_main
        return gui_main()
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
            from .adapters import executable
            for runtime in ('whisper_cpp', 'audio_cpp'):
                path = executable(runtime, 'cpu')
                print(runtime + ':cpu', path, 'OK')
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
                for model in catalog.values():
                    print(f'{model.id}\t{model.task}\t{model.runtime}\t{model.label}')
            else:
                for model_id in args.ids:
                    if model_id not in catalog:
                        raise ValueError(f'Unknown model: {model_id}')
                    resolve_model(catalog[model_id], cancel, progress, download=True)
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
        settings = Settings(stage('asr'), stage('diar'), stage('align'), not bool(args.output_dir), args.output_dir,
                            args.ffmpeg, args.start, args.duration, stage('preprocess'), args.rpp_audio)
        settings.validate(catalog)
        failed = 0
        for source in args.files:
            try:
                print(run_job(source, settings, catalog, cancel, progress))
            except Exception as error:
                failed += 1
                progress(f'{source}: {error}')
        return 1 if failed else 0
    except KeyboardInterrupt:
        cancel.set()
        return 130
    except Exception as error:
        progress(str(error))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
