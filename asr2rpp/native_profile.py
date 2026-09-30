"""Legacy defaults and declared native I/O contracts, isolated from the scheduler."""
from dataclasses import dataclass


@dataclass(frozen=True)
class NativeProfile:
    mode: str
    output: str
    max_batch_items: int
    max_audio_seconds: float
    pass_language: bool = True


def profile(model):
    output = ('turns' if model.task == 'diar' else 'words' if model.task == 'align' else
              'text' if model.capabilities.get('timestamps') == 'none' else
              'segments' if model.family == 'vibevoice_asr' else 'words')
    mode = 'streaming' if model.task == 'asr' and model.family == 'nemotron_asr' else 'offline'
    return NativeProfile(model.execution.get('mode', mode), model.execution.get('output', output),
        model.execution.get('max_batch_items', 128), model.execution.get('max_audio_seconds', 0),
        model.execution.get('pass_language', True))


def request_fields(model, options, text=''):
    """Some fixed-language models reject even a default language argument."""
    fields = {}
    if model.task in {'asr', 'align'} and profile(model).pass_language:
        fields['language'] = options.get('language') or model.defaults.get('language', 'ja')
    if model.task == 'align' or text:
        fields['text'] = text
    return fields
