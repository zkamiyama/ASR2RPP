"""The user guide follows the shipped catalog and examples stay parseable."""
from pathlib import Path
import re
import tomllib
from asr2rpp.catalog import load_catalog

ROOT = Path(__file__).resolve().parents[1]


def test_model_guide_covers_every_shipped_definition():
    guide = (ROOT / 'docs/models.md').read_text(encoding='utf-8')
    ids = {p.stem for p in (ROOT / 'models').glob('*.toml')}
    assert len(ids) == 21
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
        assert 'from asr2rpp import __version__' in (ROOT / name).read_text(encoding='utf-8')
    from asr2rpp import __version__
    assert version == __version__


def test_published_catalog_evidence_matches_shipped_tomls():
    import hashlib, json
    data = json.loads((ROOT / 'docs/validation/native-catalog-20260930.json').read_text())
    actual = {p.stem: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'models').glob('*.toml')}
    # Historical hardware evidence is immutable, not a claim that new entries
    # or a changed runtime were tested. Keep verifying every recorded definition.
    recorded = data['definition_hashes']
    assert set(actual) - set(recorded) == {'nemotron-transcribe-iq4-nl'}
    for name, original_hash in recorded.items():
        content = (ROOT / 'models' / (name + '.toml')).read_bytes()
        if name.startswith('whisper-'):
            # The old hardware evidence predates glossary metadata. Confirm the
            # underlying historical definition, not that new hints were tested.
            content = content.split(b'\n[capabilities.glossary]', 1)[0]
        assert hashlib.sha256(content).hexdigest() == original_hash
    assert data['catalog_sha256'] == hashlib.sha256(json.dumps(recorded, sort_keys=True).encode()).hexdigest()
    assert set(data['cases']) == set(recorded)
    assert data['all_passed'] and all(v['passed'] for v in data['cases'].values())
