"""Release publication cannot be enabled by a merely successful build."""
from copy import deepcopy
import pytest
from tools.publish_release import verify_approval


def fixtures():
    package=dict(artifact_id=12,run_id=34,sha256='a'*64)
    approval=dict(schema_version=1,tag='v0.3.0',source_commit='b'*40,
                  packages=dict(windows=package,macos=deepcopy(package)))
    validation=dict(passed=True,validation_mode='frozen-package',package_commit='b'*40,
                    package_sha256='a'*64,source_unchanged=True,
                    cases={str(i):dict(passed=True,exports=dict(passed=True)) for i in range(20)})
    return approval,validation


def test_exact_frozen_matrix_required():
    a,v=fixtures();assert verify_approval(a,v)==a['source_commit']


@pytest.mark.parametrize('key,value', [('passed',False),('validation_mode','source-with-native-binaries'),
    ('source_unchanged',False),('package_sha256','c'*64),('package_commit','d'*40),('cases',{})])
def test_incomplete_or_wrong_package_cannot_publish(key,value):
    a,v=fixtures();v[key]=value
    with pytest.raises(ValueError):verify_approval(a,v)


def test_boolean_schema_is_not_a_version():
    a,v=fixtures();a['schema_version']=True
    with pytest.raises(ValueError):verify_approval(a,v)


def versioned_fixtures():
    a, v = fixtures()
    ids = ['model-' + str(i) for i in range(21)]
    a.update(schema_version=2, tag='v0.4.0', expected_model_ids=ids)
    v.update(package_version='0.4.0', platform='windows', physical_hardware=True,
             hardware='Physical Windows Vulkan acceptance computer',
             cases={model_id: dict(passed=True, exports=dict(passed=True)) for model_id in ids},
             definition_hashes={model_id: 'e' * 64 for model_id in ids})
    return a, v


def test_versioned_catalog_is_not_fixed_to_twenty_models():
    a, v = versioned_fixtures()
    assert verify_approval(a, v) == a['source_commit']


@pytest.mark.parametrize('key,value', [
    ('package_version', '0.3.0'), ('package_version', None),
    ('platform', 'linux'), ('physical_hardware', False), ('physical_hardware', 1),
    ('hardware', ''), ('hardware', None), ('definition_hashes', {}),
    ('package_commit', 'd' * 40), ('package_sha256', 'd' * 64),
    ('passed', False), ('validation_mode', 'source-with-native-binaries'),
    ('source_unchanged', False),
])
def test_versioned_evidence_rejects_stale_or_unverified_metadata(key, value):
    a, v = versioned_fixtures()
    v[key] = value
    with pytest.raises(ValueError):
        verify_approval(a, v)


@pytest.mark.parametrize('field', ['cases', 'definition_hashes'])
@pytest.mark.parametrize('change', ['missing', 'extra', 'replacement'])
def test_versioned_evidence_requires_exact_catalog_membership(field, change):
    a, v = versioned_fixtures()
    original = v[field]['model-0']
    if change in ('missing', 'replacement'):
        del v[field]['model-0']
    if change in ('extra', 'replacement'):
        v[field]['unexpected-model'] = original
    with pytest.raises(ValueError):
        verify_approval(a, v)


@pytest.mark.parametrize('ids', [[], None, ['model-0', 'model-0'], ['../escape'], [True]])
def test_invalid_approved_catalog_is_rejected(ids):
    a, v = versioned_fixtures()
    a['expected_model_ids'] = ids
    with pytest.raises(ValueError):
        verify_approval(a, v)


@pytest.mark.parametrize('change', ['failed', 'export-failed', 'skip', 'invalid-hash'])
def test_incomplete_versioned_case_is_rejected(change):
    a, v = versioned_fixtures()
    if change == 'failed':
        v['cases']['model-0']['passed'] = False
    elif change == 'export-failed':
        v['cases']['model-0']['exports']['passed'] = False
    elif change == 'skip':
        v['cases']['model-0']['skipped'] = True
    else:
        v['definition_hashes']['model-0'] = 'not-a-sha256'
    with pytest.raises(ValueError):
        verify_approval(a, v)


def test_new_releases_cannot_downgrade_to_legacy_approval():
    a, v = fixtures()
    a['tag'] = 'v0.4.0'
    with pytest.raises(ValueError):
        verify_approval(a, v)


def test_existing_published_approval_and_evidence_remain_compatible():
    import json
    from pathlib import Path
    from tools.publish_release import load_validation
    root = Path(__file__).resolve().parents[1]
    # Historical approval remains independent of any future release-ready.json.
    a = json.loads('''
{
  "schema_version": 1,
  "tag": "v0.3.0",
  "source_commit": "af5dca873ab5d44c779f2c3e079ed81b132cefdb",
  "validation_sha256": "f25f54e5f8ac8e0e38b4e3bf010d7e5c465ef97ff73c2c544d40520e3584ebb2",
  "packages": {
    "windows": {
      "artifact_id": 11125020774,
      "run_id": 36773381932,
      "sha256": "80fc00d07aa0867103881f0317ed5b4a5572119bb5b9048e2892e78546b44219"
    },
    "macos": {
      "artifact_id": 11124753686,
      "run_id": 36773381932,
      "sha256": "8d58b7e41bf7a1b0706d88f036a5f2086a240c9bfe5702b5c46b1e27b22a8285"
    }
  }
}
''')
    path, v = load_validation(a, root)
    assert path.name == 'windows-0.3.0.json'
    assert verify_approval(a, v) == a['source_commit']


def test_versioned_file_is_loaded_and_evidence_hash_is_enforced(tmp_path):
    import hashlib
    import json
    from tools.publish_release import load_validation
    a, v = versioned_fixtures()
    directory = tmp_path / 'docs' / 'validation'
    directory.mkdir(parents=True)
    target = directory / 'windows-0.4.0.json'
    target.write_text(json.dumps(v), encoding='utf-8')
    (directory / 'windows-0.3.0.json').write_text('{}', encoding='utf-8')
    a['validation_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
    path, loaded = load_validation(a, tmp_path)
    assert path == target and loaded == v
    target.write_text(json.dumps(dict(v, package_version='0.3.0')), encoding='utf-8')
    with pytest.raises(ValueError, match='changed after approval'):
        load_validation(a, tmp_path)


@pytest.mark.parametrize('tag', ['v0.4.0/../0.3.0', '../0.3.0', '0.4.0', None])
def test_evidence_path_cannot_be_overridden_with_invalid_tag(tag, tmp_path):
    from tools.publish_release import load_validation
    a, _ = versioned_fixtures()
    a['tag'] = tag
    with pytest.raises(ValueError):
        load_validation(a, tmp_path)
