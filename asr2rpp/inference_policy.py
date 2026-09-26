"""Model-defined inference invariants, independent of model IDs and UI state."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class InferencePolicy:
    segmentation: str = 'none'
    timestamps: bool = True
    history: bool = True
    max_segment_seconds: float = 25.0
    timestamp_source: str = 'native'

    @property
    def uses_vad_timing(self):
        return self.timestamp_source == 'vad'

    @property
    def requires_alignment(self):
        return self.timestamp_source == 'alignment'

    @property
    def forced_parameters(self):
        result = {}
        if self.segmentation == 'vad':
            result['vad'] = True
        if not self.timestamps:
            result['no_timestamps'] = True
        if not self.history:
            result['max_context'] = 0
        return result


def policy_for(model):
    """Reject unknown/unsupported policies instead of silently ignoring them."""
    data = model.constraints.get('inference', {})
    if not isinstance(data, dict):
        raise ValueError('constraints.inference must be a TOML table')
    unknown = set(data) - {'segmentation', 'timestamps', 'history', 'max_segment_seconds', 'timestamp_source'}
    if unknown:
        raise ValueError(f'Unknown constraints.inference fields: {sorted(unknown)}')
    if data and (model.runtime != 'whisper_cpp' or model.task != 'asr' or model.sample_rate != 16000):
        raise ValueError('constraints.inference currently requires whisper_cpp ASR at 16000 Hz')
    for name in ('timestamps', 'history'):
        if name in data and type(data[name]) is not bool:
            raise ValueError(f'constraints.inference.{name} must be boolean')
        if data.get(name) is True:
            raise ValueError(f'constraints.inference.{name}: omit to allow the normal default; use false to prohibit')
    segmentation = data.get('segmentation', 'none')
    if segmentation not in ('none', 'vad'):
        raise ValueError('constraints.inference.segmentation must be none or vad')
    maximum = data.get('max_segment_seconds', 25.0)
    if (type(maximum) not in (int, float) or not math.isfinite(maximum)
            or not 2 <= maximum <= 28):
        raise ValueError('constraints.inference.max_segment_seconds must be finite and between 2 and 28')
    source = data.get('timestamp_source', 'vad' if segmentation == 'vad' else 'native')
    if source not in ('native', 'vad', 'alignment'):
        raise ValueError('constraints.inference.timestamp_source must be native, vad or alignment')
    policy = InferencePolicy(segmentation, data.get('timestamps', True),
                             data.get('history', True), float(maximum), source)
    if segmentation == 'vad' and (policy.timestamps or policy.history):
        raise ValueError('VAD segmentation requires timestamps=false and history=false in constraints.inference')
    if not policy.timestamps and segmentation != 'vad':
        raise ValueError('timestamps=false requires VAD segmentation with a preserved source timeline')
    if 'max_segment_seconds' in data and segmentation != 'vad':
        raise ValueError('max_segment_seconds requires segmentation=vad')
    if (segmentation == 'vad') != (source in ('vad', 'alignment')):
        raise ValueError('VAD segmentation requires vad/alignment timestamp_source; other models use native')
    return policy


def constrain_parameters(model, parameters):
    """Fill mandatory values; explicit conflicting settings fail before launch."""
    policy = policy_for(model)
    result = dict(parameters)
    for name, required in policy.forced_parameters.items():
        if name in result:
            compatible = result[name] == required and type(result[name]) is type(required)
            if type(required) is float:
                compatible = type(result[name]) in (int, float) and result[name] == required
            if not compatible:
                raise ValueError(f'{model.id}: TOML inference constraint requires {name}={required!r}')
        result[name] = required
    if result.get('no_timestamps') and policy.segmentation != 'vad':
        raise ValueError('no_timestamps requires a TOML VAD policy for timestamp provenance')
    if policy.segmentation == 'vad':
        if result.get('processors', 1) != 1:
            raise ValueError('VAD inference requires processors=1')
        if result.get('translate', False):
            raise ValueError('VAD source-timeline transcription does not support translation')
        maximum = result.get('vad_max_speech_duration_s', 0)
        if (type(maximum) not in (float, int) or not math.isfinite(maximum)
                or maximum < 0 or maximum > policy.max_segment_seconds):
            raise ValueError('vad_max_speech_duration_s exceeds the TOML segment limit')
    return result


def ui_parameters(model, parameters):
    """Remove obsolete saved GUI overrides, but do not hide CLI conflicts."""
    policy = policy_for(model)
    locked = set(policy.forced_parameters) | model.disabled_parameters
    if policy.segmentation == 'vad':
        locked |= {'processors', 'translate'}
    return {key: value for key, value in (parameters or {}).items() if key not in locked}


def policy_notice(model, language='en'):
    policy = policy_for(model)
    if policy.segmentation != 'vad':
        return ''
    if language == 'ja':
        timing = ('強制アライメント必須。' if policy.requires_alignment else
                  'VADの発話区間時刻を使用（単語時刻ではありません）。強制アライメントは任意。')
        return (f'TOML制約: 発話検出で最大{policy.max_segment_seconds:g}秒に分割 / '
                '時刻生成OFF / 履歴OFF。' + timing)
    timing = ('Forced alignment required.' if policy.requires_alignment else
              'VAD region timestamps (not word timing); alignment is optional.')
    return (f'TOML policy: VAD segments <= {policy.max_segment_seconds:g}s / '
            'timestamps OFF / history OFF. ' + timing)
