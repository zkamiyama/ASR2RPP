"""Merge native parameter metadata with declarative model-specific controls."""
from .parameter_specs_native import WHISPER, AUDIO, WEIGHT_TYPES


def specs_for(model) -> list[dict]:
    if model.runtime == "whisper_cpp":
        specs = WHISPER
    else:
        specs = AUDIO.get(model.family, [])
    by_key = {spec['key']:dict(spec) for spec in specs}
    for key, spec in model.parameters.items():
        by_key[key] = dict(spec, key=key)
    specs = list(by_key.values())
    result = []
    disabled = model.disabled_parameters
    for original in specs:
        if original["key"] in disabled:
            continue
        spec = dict(original)
        key = spec["key"]
        if key.startswith("session."):
            plain = key[len("session."):]
            default = model.defaults.get("session", {}).get(plain, spec.get("default"))
        else:
            default = model.defaults.get("request", {}).get(key, spec.get("default"))
        spec["default"] = default
        result.append(spec)
    from .inference_policy import policy_for
    policy = policy_for(model)
    locked = policy.forced_parameters
    if policy.segmentation == 'vad':
        locked = dict(locked, processors=1, translate=False)
        defaults = {'vad_min_speech_duration_ms': 100, 'vad_min_silence_duration_ms': 250,
                    'vad_speech_pad_ms': 200, 'vad_samples_overlap': 0.2}
        limits = {'vad_threshold': (0.01, 1.0), 'vad_min_speech_duration_ms': (0, 1000),
                  'vad_min_silence_duration_ms': (50, 2000), 'vad_speech_pad_ms': (0, 1000)}
        for spec in result:
            key = spec['key']
            if key in defaults and key not in model.defaults.get('request', {}):
                spec['default'] = defaults[key]
            if key in limits:
                spec['min'], spec['max'] = limits[key]
            if key == 'vad_max_speech_duration_s':
                spec['max'] = policy.max_segment_seconds
                spec['tip_en'] = '0 uses the TOML limit. Otherwise choose 2 seconds or more, up to that limit.'
                spec['tip_ja'] = '0はTOMLの上限。変更時は2秒以上、TOMLの上限以下を指定。'
        result.append(dict(key='no_timestamps', type='bool', default=True,
                           en='Disable timestamp generation', ja='時刻生成を無効化',
                           tip_en='Required by TOML. Uses VAD region timestamps or optional alignment.',
                           tip_ja='TOML制約で固定。VADの区間時刻、または強制アライメントの時刻を使います。'))
    for spec in result:
        if spec['key'] in locked:
            spec['default'] = locked[spec['key']]
            spec['locked'] = True
    return result
