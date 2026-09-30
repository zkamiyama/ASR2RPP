"""Legacy defaults and declared native I/O contracts, isolated from the scheduler."""
from dataclasses import dataclass


@dataclass(frozen=True)
class NativeProfile:
    mode: str
    output: str
    max_batch_items: int
    max_audio_seconds: float


def profile(model):
    output = ('turns' if model.task == 'diar' else 'words' if model.task == 'align' else
              'text' if model.capabilities.get('timestamps') == 'none' else
              'segments' if model.family == 'vibevoice_asr' else 'words')
    mode = 'streaming' if model.task == 'asr' and model.family == 'nemotron_asr' else 'offline'
    return NativeProfile(model.execution.get('mode', mode), model.execution.get('output', output),
        model.execution.get('max_batch_items', 128), model.execution.get('max_audio_seconds', 0))
