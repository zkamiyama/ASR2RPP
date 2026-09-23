"""Small shared utilities. No inference framework dependency."""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

VERSION = '0.1.0-preview'


def app_dir() -> Path:
    return Path(sys.executable if getattr(sys, 'frozen', False) else __file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def run_process(args, log_prefix, timeout=1800, cancel_file=None):
    """Run trusted installed executables without shell expansion or global DLL changes."""
    args = [str(arg) for arg in args]
    prefix = Path(log_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if sys.platform != 'win32':
        original = env.get('LD_LIBRARY_PATH_ORIG')
        if original is None:
            env.pop('LD_LIBRARY_PATH', None)
        else:
            env['LD_LIBRARY_PATH'] = original
    env['PYTHONUTF8'] = '1'
    flags = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == 'nt' else 0
    start = time.monotonic()
    stdout_path, stderr_path = str(prefix) + '.stdout.log', str(prefix) + '.stderr.log'
    with open(stdout_path, 'wb') as out, open(stderr_path, 'wb') as err:
        if os.name == 'nt' and getattr(sys, 'frozen', False):
            import ctypes
            ctypes.windll.kernel32.SetDllDirectoryW(None)
        try:
            p = subprocess.Popen(args, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                 env=env, shell=False, creationflags=flags,
                                 start_new_session=(os.name != 'nt'))
        finally:
            if os.name == 'nt' and getattr(sys, 'frozen', False):
                import ctypes
                ctypes.windll.kernel32.SetDllDirectoryW(sys._MEIPASS)
        try:
            while p.poll() is None:
                if cancel_file and Path(cancel_file).exists():
                    raise InterruptedError('Cancelled')
                if time.monotonic() - start > timeout:
                    raise TimeoutError(f'Process exceeded {timeout} seconds')
                time.sleep(0.1)
        except BaseException:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(p.pid), '/T', '/F'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW, check=False)
            else:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            p.wait(timeout=10)
            raise
    report = dict(argv=args, returncode=p.returncode, wall_seconds=time.monotonic()-start,
                  stdout=stdout_path, stderr=stderr_path)
    write_json(str(prefix) + '.process.json', report)
    if p.returncode:
        tail = Path(stderr_path).read_text(encoding='utf-8', errors='replace')[-3000:]
        raise RuntimeError(f'{Path(args[0]).name} failed ({p.returncode}). {tail}')
    return report
