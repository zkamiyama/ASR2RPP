"""The user guide follows the shipped catalog and examples stay parseable."""
from pathlib import Path
import re
import tomllib
from asr2rpp.catalog import load_catalog

ROOT = Path(__file__).resolve().parents[1]


def test_model_guide_covers_every_shipped_definition():
    guide = (ROOT / 'docs/models.md').read_text(encoding='utf-8')
    ids = {p.stem for p in (ROOT / 'models').glob('*.toml')}
    assert len(ids) == 20
    assert all(f'`{name}`' in guide for name in ids)
    assert '19' in guide and 'CPU' in guide and 'Vulkan' in guide


def test_user_guides_use_existing_relative_links():
    for name in ('README.md', 'README.ja.md', 'docs/models.md', 'docs/provider-models.md',
                 'docs/custom-models.ja.md', 'docs/development.md', 'docs/inference-policy.md'):
        path = ROOT / name
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if '://' not in target and not target.startswith('#'):
                assert (path.parent / target.split('#')[0]).is_file(), (name, target)


def test_customization_model_examples_load_without_gui(tmp_path):
    count = 0
    for name in ('docs/provider-models.md', 'docs/custom-models.ja.md'):
        text = (ROOT / name).read_text(encoding='utf-8')
        for block in re.findall(r'```toml\n(.*?)```', text, flags=re.S):
            data = tomllib.loads(block)
            if 'schema_version' in data:
                file = tmp_path / f'example-{count}.toml'
                file.write_text(block, encoding='utf-8')
                count += 1
    models, errors = load_catalog(tmp_path)
    assert not errors
    assert len(models) == count == 4


def test_distribution_version_tracks_project_version():
    version = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']['version']
    for name in ('tools/package_windows.py', 'tools/package_macos.py'):
        assert version + '-preview' in (ROOT / name).read_text(encoding='utf-8')


def test_published_catalog_evidence_matches_shipped_tomls():
    import hashlib, json
    data = json.loads((ROOT / 'docs/validation/native-catalog-20260930.json').read_text())
    actual = {p.stem: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'models').glob('*.toml')}
    assert data['definition_hashes'] == actual
    assert data['catalog_sha256'] == hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()
    assert set(data['cases']) == set(actual)
    assert data['all_passed'] and all(v['passed'] for v in data['cases'].values())
