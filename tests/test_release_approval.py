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
