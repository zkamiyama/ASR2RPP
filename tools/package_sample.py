"""A small, pinned upstream speech fixture for trying the unpacked app."""
from pathlib import Path
import hashlib
from urllib.request import urlopen

URL = 'https://raw.githubusercontent.com/ggml-org/whisper.cpp/a664346ea5c6dddff3e61a2b7b32dd4514613f50/samples/jfk.wav'
SHA256 = '59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e'


def make_sample(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root/'sample.wav'
    if not path.exists():
        with urlopen(URL, timeout=60) as response:
            data = response.read(4*1024**2)
        if hashlib.sha256(data).hexdigest() != SHA256:
            raise ValueError('Sample audio checksum mismatch')
        path.write_bytes(data)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError('Sample audio checksum mismatch')
    (root/'README.txt').write_text(
        'Drag sample.wav into ASR2RPP. Choose Whisper Base, language en, then GO.\n'
        'First use downloads the selected model and any missing FFmpeg.\n'
        'Source: ' + URL + '\nSHA-256: ' + SHA256 + '\n', encoding='utf-8')
    return path
