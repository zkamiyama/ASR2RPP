"""User-owned timing policy, independent of ASR family and native CLI flags."""
from dataclasses import dataclass, asdict, replace
import copy
import math
from .inference_policy import policy_for

MODES = ('auto', 'native', 'vad', 'alignment')


@dataclass(frozen=True)
class TimingSettings:
    mode: str = 'auto'
    max_seconds: float = 25.0
    threshold: float = 0.5
    min_speech_ms: int = 100
    min_silence_ms: int = 250
    pad_ms: int = 200
    overlap: float = 0.2
    segment_before_alignment: bool = False
    speaker_source: str = 'auto'

    def parameters(self):
        return dict(vad_threshold=self.threshold,
                    vad_min_speech_duration_ms=self.min_speech_ms,
                    vad_min_silence_duration_ms=self.min_silence_ms,
                    vad_speech_pad_ms=self.pad_ms, vad_samples_overlap=self.overlap,
                    vad_max_speech_duration_s=self.max_seconds)

    def validate(self):
        if self.speaker_source not in ('auto','native','diarizer','none'):
            raise ValueError('Speaker source must be auto, native, diarizer or none')
        if self.mode not in MODES:
            raise ValueError('Timing must be auto, native, vad or alignment')
        if (type(self.max_seconds) not in (int, float) or
                not math.isfinite(self.max_seconds) or not 2 <= self.max_seconds <= 28):
            raise ValueError('VAD window must be 2..28 seconds')
        if type(self.segment_before_alignment) is not bool:
            raise ValueError('segment_before_alignment must be boolean')
        from .vad import VadOptions
        VadOptions.from_parameters(self.parameters(), self.max_seconds)


@dataclass(frozen=True)
class TimingPlan:
    mode: str
    segmented: bool
    align: bool
    timestamp_source: str


def native_timestamps(model):
    declared = model.capabilities.get('timestamps')
    if declared is not None:
        return declared != 'none'
    return policy_for(model).timestamps


def plan_for(model, timing, alignment_enabled=False):
    timing.validate()
    native = native_timestamps(model)
    mode = timing.mode
    if mode == 'auto':
        # Legacy TOMLs retain their automatic default, but an explicit user
        # choice of VAD supersedes a legacy mandatory alignment workflow.
        if policy_for(model).requires_alignment and not alignment_enabled:
            raise ValueError('Automatic timing for this legacy TOML requires forced alignment: enable it or choose VAD timing in Settings')
        mode = 'alignment' if alignment_enabled else ('native' if native else 'vad')
    if mode == 'native' and not native:
        raise ValueError('This model has no native timestamps; choose VAD or forced alignment')
    if mode == 'alignment' and not alignment_enabled:
        raise ValueError('Forced-alignment timing requires an enabled alignment model')
    if mode in ('native', 'vad') and alignment_enabled:
        raise ValueError('Disable the alignment stage or choose automatic / forced-alignment timing')
    segmented = mode == 'vad' or (mode == 'alignment' and (not native or timing.segment_before_alignment))
    if segmented and timing.speaker_source == 'native':
        raise ValueError('Native speaker labels require native ASR intervals; choose native timing or an external diarizer')
    return TimingPlan(mode, segmented, mode == 'alignment',
                      'forced_alignment' if mode == 'alignment' else 'vad' if segmented else 'native_asr')


def prepare_run(settings, catalog):
    """Make an invocation snapshot; never mutate user TOMLs or shared GUI state."""
    settings = copy.deepcopy(settings)
    model = catalog[settings.asr.model_id]
    plan = plan_for(model, settings.timing, settings.align is not None)
    if model.runtime == 'whisper_cpp' and plan.segmented:
        constraints = copy.deepcopy(model.constraints)
        constraints['inference'] = dict(segmentation='vad', timestamps=False,
            history=False, max_segment_seconds=settings.timing.max_seconds,
            timestamp_source='vad')
        model = replace(model, constraints=constraints)
        params = dict(settings.asr.parameters or {})
        params.update(settings.timing.parameters())
        settings.asr = replace(settings.asr, parameters=params)
    catalog = dict(catalog)
    catalog[model.id] = model
    return settings, catalog, plan


def stage_options(settings):
    return dict(settings.asr.options(), timing=asdict(settings.timing),
                alignment_requested=settings.align is not None, ffmpeg=settings.ffmpeg)
