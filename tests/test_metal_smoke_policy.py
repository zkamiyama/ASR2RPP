import pytest
from tools.metal_smoke_policy import select_metal_cases


def test_missing_device_is_not_gpu_success():
    runnable, skipped = select_metal_cases(dict(available=False, device='', simdgroup_matrix=False))
    assert not runnable and set(skipped) == {'whisper-base', 'qwen3-asr-06b'}


def test_limited_device_does_not_claim_qwen_metal_verification():
    runnable, skipped = select_metal_cases(dict(available=True, device='Any name', simdgroup_matrix=False))
    assert runnable == ['whisper-base'] and set(skipped) == {'qwen3-asr-06b'}


def test_matrix_capable_device_must_run_both_models():
    runnable, skipped = select_metal_cases(dict(available=True, device='Any name', simdgroup_matrix=True))
    assert runnable == ['whisper-base', 'qwen3-asr-06b'] and not skipped


@pytest.mark.parametrize('value', [{}, None, {'available': 'true', 'device': 'name', 'simdgroup_matrix': True}])
def test_broken_probe_cannot_silently_skip_gpu_tests(value):
    with pytest.raises(ValueError):
        select_metal_cases(value)
