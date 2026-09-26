"""Versioned native PCM plans and response files, with legacy runtime fallback."""
from pathlib import Path
import json
import os
import threading
import wave

_CACHE = {}
_LOCK = threading.Lock()


def capabilities(binary, work, cancel, progress):
    """Probe only explicit feature markers; never assume a replacement supports IPC."""
    if os.getenv('ASR2RPP_WHISPER_IO') == 'legacy':
        return frozenset()
    path = Path(binary)
    if not path.is_file():
        return frozenset()
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    with _LOCK:
        cached = _CACHE.get(key)
    if cached is not None:
        return cached
    from .adapters import run_process
    log = Path(work)/'whisper-capabilities.log'
    run_process([str(path), '--help'], cancel, lambda _: None, log, timeout=30)
    text = log.read_text(encoding='utf-8')
    found = frozenset(x for x in ('ASR2RPP_PCM_PLAN_V1','ASR2RPP_RESPONSE_V1','ASR2RPP_PCM_RESULT_V1') if x in text)
    # The unmodified pinned runtime has documented-in-source @response support.
    manifest = path.parent/'build-manifest.json'
    if not found and manifest.is_file():
        meta = json.loads(manifest.read_text(encoding='utf-8'))
        if meta.get('repository') == 'ggml-org/whisper.cpp' and meta.get('commit') == 'a664346ea5c6dddff3e61a2b7b32dd4514613f50':
            found = frozenset({'ASR2RPP_RESPONSE_V1'})
    with _LOCK:
        if len(_CACHE)>32:
            _CACHE.clear()
        _CACHE[key] = found
    return found


def field(value):
    text = os.fspath(value)
    if not text or any(c in text for c in ('\x00','\r','\n','\t')):
        raise ValueError('Unsupported control character in native input path/argument')
    return text


def response_command(argv, destination):
    # Each line is one argv element; no shell quotes or concatenation.
    values = [str(v) for v in argv[1:]]
    if any('\r' in v or '\n' in v or '\0' in v for v in values):
        raise ValueError('Newlines/NUL cannot be represented in native response arguments')
    Path(destination).write_text('\n'.join(values)+'\n',encoding='utf-8',newline='\n')
    return [str(argv[0]), '@'+str(Path(destination).resolve())]


def write_plan(audio, windows, prefixes, path, cancel):
    """Pass sample indices, not rounded milliseconds. No segment WAVs are written."""
    from .catalog import checkpoint
    if len(windows) != len(prefixes) or not 0<len(windows)<=100000:
        raise ValueError('Invalid PCM plan request count')
    with wave.open(str(audio),'rb') as pcm:
        if (pcm.getframerate(),pcm.getnchannels(),pcm.getsampwidth()) != (16000,1,2):
            raise ValueError('PCM plan requires mono PCM16 at 16000 Hz')
        frames=pcm.getnframes()
    source=field(Path(audio).resolve())
    lines=['ASR2RPP_PCM_PLAN_V1',str(len(windows))]
    intervals=[]
    for w,prefix in zip(windows,prefixes):
        checkpoint(cancel)
        first=round(w.start*16000)
        count=min(frames-first,round((w.end-first/16000)*16000))
        if first<0 or count<=0 or count>28*16000 or first+count>frames:
            raise ValueError('Invalid bounded PCM interval')
        lines.extend([source,field(Path(prefix).resolve()),str(first),str(count)])
        intervals.append((first/16000,count/16000))
    text='\n'.join(lines)+'\n'
    if len(text.encode('utf-8'))>64*1024**2:
        raise ValueError('PCM plan exceeds native size limit')
    Path(path).write_text(text,encoding='utf-8',newline='\n')
    return intervals


def read_bundle(path, count):
    path = Path(path)
    if path.stat().st_size > 64*1024**2:
        raise ValueError('Native result bundle exceeds safety bound')
    try:
        payload = json.loads(path.read_text(encoding='utf-8-sig'))
    except (UnicodeError, ValueError) as exc:
        raise ValueError('Invalid UTF-8/JSON in native result bundle; raw output retained') from exc
    if not isinstance(payload,dict) or type(payload.get('schema')) is not int or payload.get('schema')!=1 or not isinstance(payload.get('results'),list) or len(payload['results'])!=count:
        raise ValueError('Invalid/incomplete native result bundle')
    return payload['results']
