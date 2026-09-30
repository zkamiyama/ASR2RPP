"""Publish only exact CI ZIPs approved by physical-Windows validation.

This workflow is activated by release-ready.json, never by a successful build
alone. Different existing assets and failed or source-only matrices are rejected.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
import zipfile

REPOSITORY = 'zkamiyama/ASR2RPP'
API = 'https://api.github.com/repos/' + REPOSITORY


class RedirectWithoutCredentials(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, url):
        redirected = super().redirect_request(request, fp, code, msg, headers, url)
        if redirected and urllib.parse.urlparse(request.full_url).netloc != urllib.parse.urlparse(url).netloc:
            redirected.remove_header('Authorization')
        return redirected


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_approval(approval, validation):
    if type(approval.get('schema_version')) is not int or approval['schema_version'] != 1 or not re.fullmatch(r'v\d+\.\d+\.\d+', approval.get('tag', '')):
        raise ValueError('Invalid release approval')
    commit = approval.get('source_commit', '')
    if not re.fullmatch(r'[a-f0-9]{40}', commit):
        raise ValueError('Invalid source commit')
    if set(approval.get('packages', {})) != {'windows', 'macos'}:
        raise ValueError('Both platform packages are required')
    if validation.get('passed') is not True or validation.get('validation_mode') != 'frozen-package':
        raise ValueError('Physical Windows frozen-package validation is required')
    if validation.get('package_commit') != commit or validation.get('package_sha256') != approval['packages']['windows']['sha256']:
        raise ValueError('Windows validation does not describe this exact package')
    cases = validation.get('cases', {})
    if len(cases) != 20 or not all(v.get('passed') is True and v.get('exports', {}).get('passed') is True for v in cases.values()):
        raise ValueError('All 20 catalog cases and exporters must pass')
    if validation.get('source_unchanged') is not True:
        raise ValueError('Source integrity was not verified')
    for key, package in approval['packages'].items():
        if type(package.get('artifact_id')) is not int or type(package.get('run_id')) is not int:
            raise ValueError('Invalid CI artifact/run ID')
        if not re.fullmatch(r'[a-f0-9]{64}', package.get('sha256', '')):
            raise ValueError('Missing archive digest')
    return commit


def main():
    if os.environ.get('GITHUB_REPOSITORY') != REPOSITORY:
        raise ValueError('Publication is restricted to the project repository')
    token = os.environ['GITHUB_TOKEN']
    opener = urllib.request.build_opener(RedirectWithoutCredentials())

    def request(url, *, method='GET', data=None, content_type='application/json'):
        payload = json.dumps(data).encode() if data is not None and content_type == 'application/json' else data
        req = urllib.request.Request(url, data=payload, method=method, headers={
            'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28', 'Content-Type': content_type,
            'User-Agent': 'ASR2RPP-release-verifier'})
        with opener.open(req, timeout=600) as response:
            return json.load(response)

    approval = json.loads(Path('release-ready.json').read_text(encoding='utf-8'))
    evidence_path = Path('docs/validation/windows-0.3.0.json')
    if sha(evidence_path) != approval.get('validation_sha256'):
        raise ValueError('Validation record changed after approval')
    validation = json.loads(evidence_path.read_text(encoding='utf-8'))
    commit = verify_approval(approval, validation)
    output = Path('release-files')
    output.mkdir(exist_ok=False)
    files = []
    for key, record in approval['packages'].items():
        run = request(f'{API}/actions/runs/{record["run_id"]}')
        if run.get('head_sha') != commit:
            raise ValueError('CI source commit mismatch')
        jobs = request(f'{API}/actions/runs/{record["run_id"]}/jobs?per_page=100')['jobs']
        required_job = 'windows' if key == 'windows' else 'macos-arm64'
        if not any(j['name'] == required_job and j['conclusion'] == 'success' for j in jobs):
            raise ValueError('Platform CI job did not pass')
        artifact = request(f'{API}/actions/artifacts/{record["artifact_id"]}')
        if artifact.get('expired') or artifact['workflow_run']['head_sha'] != commit:
            raise ValueError('Expired or mismatched artifact')
        name = 'ASR2RPP-Windows-x64.zip' if key == 'windows' else 'ASR2RPP-macOS-arm64.zip'
        wrapper = output/(key+'-artifact.zip')
        req = urllib.request.Request(artifact['archive_download_url'], headers={'Authorization': 'Bearer '+token})
        with opener.open(req, timeout=300) as response, wrapper.open('xb') as target:
            shutil.copyfileobj(response, target, 1024**2)
        if artifact.get('digest') and artifact['digest'] != 'sha256:' + sha(wrapper):
            raise ValueError('GitHub artifact digest mismatch')
        archive = output/name
        with zipfile.ZipFile(wrapper) as z:
            matches = [n for n in z.namelist() if PurePosixPath(n).name == name and '..' not in PurePosixPath(n).parts]
            if len(matches) != 1:
                raise ValueError('Missing or ambiguous release archive')
            with z.open(matches[0]) as source, archive.open('xb') as target:
                shutil.copyfileobj(source, target, 1024**2)
        wrapper.unlink()
        if sha(archive) != record['sha256']:
            raise ValueError('Archive differs from approved, tested bytes')
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:
                raise ValueError('Corrupt release ZIP')
            prefix = 'ASR2RPP' if key == 'windows' else 'ASR2RPP.app/Contents/Resources'
            metadata = json.loads(z.read(prefix+'/version.json'))
            if metadata['commit'] != commit or metadata['version'] != approval['tag'][1:]:
                raise ValueError('Packaged application provenance mismatch')
            if metadata.get('output_formats') != ['rpp','otio','json']:
                raise ValueError('Output formats missing')
            for name_in_zip in z.namelist():
                lower = name_in_zip.lower()
                if Path(lower).suffix in {'.ttf','.otf','.ttc','.woff','.woff2'}:
                    raise ValueError('Unexpected font in package')
                if any(p in lower for p in ('cublas','cudnn','cudart','ggml-cuda','ctranslate2.dll')):
                    raise ValueError('Forbidden GPU dependency in package')
            if key == 'windows':
                hashes = {Path(n).stem: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()
                          if n.startswith(prefix+'/models/') and n.endswith('.toml')}
                if hashes != validation['definition_hashes']:
                    raise ValueError('Shipped TOMLs differ from the verified catalog')
        files.append(archive)
    checksum = output/'SHA256SUMS.txt'
    checksum.write_text(''.join(sha(f)+'  '+f.name+'\n' for f in files), encoding='utf-8')
    evidence = output/'VALIDATION-Windows.json'
    shutil.copy2(evidence_path, evidence)
    files.extend([checksum, evidence])
    try:
        release = request(API+'/releases/tags/'+approval['tag'])
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        release = request(API+'/releases', method='POST', data=dict(
            tag_name=approval['tag'], target_commitish=commit,
            name='ASR2RPP '+approval['tag'][1:], draft=True, prerelease=False,
            body=Path('docs/release-notes.md').read_text(encoding='utf-8')))
    assets = {v['name']: v for v in release.get('assets', [])}
    for file in files:
        expected = 'sha256:' + sha(file)
        if file.name in assets:
            if assets[file.name].get('digest') != expected:
                raise ValueError('Refusing to replace a different existing release asset: '+file.name)
            continue
        if not release['draft']:
            raise ValueError('Refusing to change a published release')
        url = release['upload_url'].split('{')[0]+'?'+urllib.parse.urlencode({'name':file.name})
        asset = request(url, method='POST', data=file.read_bytes(), content_type='application/octet-stream')
        if asset.get('digest') != expected or asset['size'] != file.stat().st_size:
            raise ValueError('Uploaded release asset failed verification')
    if release['draft']:
        release = request(release['url'], method='PATCH', data=dict(draft=False, prerelease=False, make_latest='true'))
    print(release['html_url'])


if __name__ == '__main__':
    main()
