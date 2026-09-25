import sys
for stream in (sys.stdout, sys.stderr):
    if stream is not None and hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')
from asr2rpp.cli import main
if __name__ == '__main__':
    raise SystemExit(main())
