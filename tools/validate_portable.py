"""Validate the extracted distribution, not Python source modules.

Use a public or synthetic fixture. The summary contains hashes and measurements,
not transcript text. Individual diagnostics remain in the chosen report directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--fixture',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--weights-dir',type=Path)
    parser.add_argument('--ffmpeg',default='')
    parser.add_argument('--gpu',action='store_true')
    parser.add_argument('--install',action='store_true')
    args=parser.parse_args()
    package=args.package.resolve();fixture=args.fixture.resolve();report=args.report.resolve()
    cli=package/'asr2rpp-cli.exe'
    if not cli.is_file() or not fixture.is_file():
        raise ValueError('A complete extracted package and speech fixture are required')
    report.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy()
    env.update(ASR2RPP_HOME=str(report/'home'),ASR2RPP_CACHE_DIR=str(report/'cache'),
               PYTHONUTF8='1',ASR2RPP_DISABLE_RUNTIME_BOOTSTRAP='1')
    # The package must not accidentally use an installed Python/CUDA toolkit.
    for key in list(env):
        if key.startswith(('CUDA_PATH','CUDNN','PYTHONPATH','VULKAN_SDK')):
            env.pop(key,None)
    env['PATH']=str(Path(env.get('SystemRoot','C:/Windows'))/'System32')
    if args.weights_dir:
        env['ASR2RPP_WEIGHTS_DIR']=str(args.weights_dir.resolve())
    from portable_runtime import audit_lightweight
    audit_lightweight(package)
    original=digest(fixture)
    results={'fixture_sha256':original,'package_commit':json.loads((package/'version.json').read_text())['commit'],
             'gpu_requested':args.gpu,'installed_python_or_toolkit_path_used':False,'cases':{}}
    def invoke(name,argv,timeout=600):
        start=time.monotonic()
        with (report/(name+'.log')).open('w',encoding='utf-8') as log:
            process=subprocess.Popen([str(cli),*map(str,argv)],stdout=log,stderr=subprocess.STDOUT,env=env)
            try:
                code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill','/F','/T','/PID',str(process.pid)],stdout=log,stderr=subprocess.STDOUT)
                process.wait(timeout=15)
                code=-1
        return {'returncode':code,'elapsed_seconds':round(time.monotonic()-start,4)}
    models=['whisper-base','reazonspeech-k2','qwen3-asr-06b','qwen-forced-aligner']
    if args.install:
        acquisition=invoke('install',['models','install',*models],timeout=1200)
        results['acquisition']=acquisition
        if acquisition['returncode']:
            raise RuntimeError('Model acquisition failed; see install.log')
    cases=[('whisper-cpu-native','whisper-base','cpu','native',[]),
           ('whisper-cpu-vad','whisper-base','cpu','vad',[]),
           ('reazon-cpu-vad','reazonspeech-k2','cpu','vad',[])]
    if args.gpu:
        cases += [('whisper-vulkan-vad','whisper-base','vulkan','vad',[]),
                  ('qwen-vulkan-vad','qwen3-asr-06b','vulkan','vad',[]),
                  ('qwen-vulkan-alignment','qwen3-asr-06b','vulkan','alignment',[
                      '--align','qwen-forced-aligner','--align-device','vulkan','--align-language','Japanese'])]
    for name,model,device,timing,extra in cases:
        output=report/name
        command=['run',fixture,'--asr',model,'--asr-device',device,'--timing',timing,
                 '--asr-language','Japanese' if model.startswith('qwen') else 'ja',
                 '--threads','4','--output-dir',output,*extra]
        if args.ffmpeg:
            command += ['--ffmpeg',args.ffmpeg]
        record=invoke(name,command)
        manifests=list(output.glob('*.asr2rpp/manifest.json'))
        projects=list(output.glob('*.rpp'))
        record['passed']=record['returncode']==0 and len(manifests)==len(projects)==1
        if record['passed']:
            manifest=json.loads(manifests[0].read_text(encoding='utf-8'))
            transcript=json.loads((manifests[0].parent/'transcript.json').read_text(encoding='utf-8'))
            units=transcript['units']
            record.update(source_unchanged=manifest.get('source_unchanged'),timestamp_source=manifest.get('timestamp_source'),
                          unit_count=len(units),methods=sorted({u['method'] for u in units}),
                          text_sha256=hashlib.sha256(''.join(u['text'] for u in units).encode()).hexdigest())
            record['passed'] &= bool(units) and record['source_unchanged'] is True
            if timing=='vad':record['passed'] &= all(u['method']=='vad_segment' for u in units)
            if timing=='alignment':record['passed'] &= all(u['method']=='forced_alignment' for u in units)
            runtime_devices=[]
            for path in manifests[0].parent.rglob('raw.json'):
                raw=json.loads(path.read_text(encoding='utf-8'))
                if isinstance(raw,dict) and isinstance(raw.get('runtime'),dict):
                    runtime_devices.append(raw['runtime'].get('device'))
            record['worker_devices']=sorted({v for v in runtime_devices if v})
            gpu_lines=[]
            for path in manifests[0].parent.rglob('*.log'):
                for line in path.read_text(encoding='utf-8',errors='replace').splitlines():
                    if any(token in line for token in ('ggml_vulkan:', 'Vulkan0')):
                        gpu_lines.append(line[:250])
            record['gpu_evidence']=list(dict.fromkeys(gpu_lines))[:8]
            if device=='vulkan':
                record['passed'] &= bool(record['gpu_evidence'])
        results['cases'][name]=record
        (report/'summary.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
        print(name,json.dumps(record,ensure_ascii=False),flush=True)
    results['original_unchanged']=digest(fixture)==original
    results['passed']=results['original_unchanged'] and all(x['passed'] for x in results['cases'].values())
    (report/'summary.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
    return 0 if results['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
