"""Frozen provider entry point; no GUI dependency or model-download side effects."""
from asr2rpp.provider_worker import main

if __name__ == '__main__':
    raise SystemExit(main())
