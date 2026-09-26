"""Retain native diagnostics without duplicating completed cache trees."""
from pathlib import Path
import shutil
from .performance import timed


@timed('diagnostics')
def persist_tree(source, destination):
    source, destination = Path(source), Path(destination)
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Move only completed, media-free private engine directories. No source audio
    # or shared PCM lives here. Cross-device/collision cases keep copy semantics.
    if not destination.exists() and not any(source.rglob('*.wav')):
        try:
            source.rename(destination)
            return
        except OSError:
            pass
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('*.wav'))
