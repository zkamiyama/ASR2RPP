"""ASR2RPP CLI. Run without arguments for the optional PySide6 interface."""
import argparse
import json
import sys
from pathlib import Path
from common import VERSION, read_json, write_json
from registry import Registry
from pipeline import run_job, export_project


def json_object(text):
    value = json.loads(text)
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError('Parameters must be a JSON object')
    return value


def parser():
    p = argparse.ArgumentParser(description='ASR2RPP: native ASR + independent diarization -> non-destructive REAPER project')
    p.add_argument('--version', action='version', version=VERSION)
    p.add_argument('--registry', help='External models.json path; loads optional models.user.json alongside it')
    sub = p.add_subparsers(dest='command')
    sub.add_parser('models', help='List models and local availability')
    install = sub.add_parser('install', help='Download one declared model; verifies size/hash and records immutable revision')
    install.add_argument('model')
    sub.add_parser('doctor', help='Check configured executables; this is not a GPU inference test')
    for name in ('run', 'compare'):
        run = sub.add_parser(name)
        run.add_argument('input', type=Path)
        run.add_argument('--out', type=Path, required=True)
        run.add_argument('--asr', required=True, nargs='+' if name == 'compare' else None)
        run.add_argument('--diarization', default='none')
        run.add_argument('--asr-device', choices=['cpu','cuda','vulkan','metal'], default='cpu')
        run.add_argument('--diarization-device', choices=['cpu','cuda','vulkan','metal'], default='cpu')
        run.add_argument('--asr-params', type=json_object, default={})
        run.add_argument('--diarization-params', type=json_object, default={})
        run.add_argument('--start', type=float, default=0, help='Start in seconds relative to decoded audio stream')
        run.add_argument('--duration', type=float, default=55, help='Inference clip duration, default 55 seconds')
        run.add_argument('--timeout', type=int, default=1800, help='Per-model process limit in seconds')
        run.add_argument('--cancel-file', type=Path)
    exp = sub.add_parser('export', help='Re-export edited transcript JSON without running a model')
    exp.add_argument('transcript', type=Path)
    exp.add_argument('--out', type=Path, required=True)
    gui = sub.add_parser('gui')
    gui.add_argument('--screenshot', type=Path, help=argparse.SUPPRESS)
    return p


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    args = parser().parse_args(argv)
    try:
        if args.command in (None, 'gui'):
            from gui import launch
            return launch(args.registry, getattr(args, 'screenshot', None))
        if args.command == 'export':
            if args.out.exists():
                raise FileExistsError('Export target already exists; use a new filename')
            print(export_project(read_json(args.transcript), args.out))
            return 0
        registry = Registry(args.registry)
        if args.command == 'models':
            for m in registry.models.values():
                print(json.dumps(dict(id=m['id'], type=m['type'], engine=m['engine'],
                                      installed=registry.model_path(m).is_file(),
                                      warning=m.get('warning'), status=m.get('status', 'unverified')),
                                 ensure_ascii=False))
        elif args.command == 'install':
            registry.install(args.model, lambda text: print(text, flush=True))
        elif args.command == 'doctor':
            report = {}
            for engine, devices in registry.data.get('engines', {}).items():
                for device in devices:
                    try:
                        report[f'{engine}/{device}'] = dict(executable=registry.executable(engine, device), execution='not_tested')
                    except (ValueError, OSError) as exc:
                        report[f'{engine}/{device}'] = dict(error=str(exc))
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            if args.timeout < 1:
                raise ValueError('timeout must be positive')
            models = args.asr if args.command == 'compare' else [args.asr]
            if args.command == 'compare':
                if len(set(models)) != len(models):
                    raise ValueError('Duplicate ASR model IDs in comparison')
                if args.out.exists() and any(args.out.iterdir()):
                    raise FileExistsError('Comparison output directory must be empty')
            reports = []
            for mid in models:
                out = args.out / mid if args.command == 'compare' else args.out
                try:
                    data = run_job(registry, args.input, out, mid, args.diarization,
                                   args.start, args.duration, args.asr_device, args.diarization_device,
                                   args.asr_params, args.diarization_params, args.timeout, args.cancel_file,
                                   log=lambda text: print(text, flush=True))
                    reports.append(dict(model=mid, status='completed', segments=len(data['segments']),
                                        asr_wall_seconds=data['runs']['asr']['process']['wall_seconds'],
                                        accuracy='not_evaluated_without_human_reference'))
                except Exception as exc:
                    if args.command != 'compare':
                        raise
                    reports.append(dict(model=mid, status='failed', error=str(exc)))
            if args.command == 'compare':
                write_json(args.out / 'comparison.json', reports)
                return int(any(r['status'] != 'completed' for r in reports))
        return 0
    except Exception as exc:
        if sys.stderr:
            print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
