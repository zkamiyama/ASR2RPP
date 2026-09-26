"""A restored CI samples directory must not block a fresh Git checkout."""
from pathlib import Path
import pytest
from tools.build_native import preserve_restored_samples


@pytest.mark.parametrize('ci,reuse,git_exists,expect_move', [
    ('true', '1', False, True),
    ('true', '1', True, False),
    ('false', '1', False, False),
    ('true', '0', False, False),
])
def test_ci_samples_preserved_without_force_or_deletion(tmp_path, monkeypatch, ci, reuse, git_exists, expect_move):
    source = tmp_path / 'native-source'
    samples = source / 'samples'
    samples.mkdir(parents=True)
    (samples / 'jfk.mp3').write_bytes(b'cached public fixture')
    if git_exists:
        (source / '.git').mkdir()
    monkeypatch.setenv('GITHUB_ACTIONS', ci)
    monkeypatch.setenv('ASR2RPP_REUSE_VERIFIED_NATIVE', reuse)
    backup = preserve_restored_samples(source)
    assert (backup is not None) == expect_move
    target = backup / 'samples' if expect_move else samples
    assert (target / 'jfk.mp3').read_bytes() == b'cached public fixture'
    assert samples.exists() != expect_move
    if expect_move:
        assert preserve_restored_samples(source) is None
