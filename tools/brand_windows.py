"""Apply/verify icons only on the staged unsigned Windows distribution.

Launchers receive --icon before PyInstaller appends their archives. Resource-only
updates below apply to copied native EXEs, never the source build cache or an
installed FFmpeg. Manifest hashes are refreshed after the resource edit.
"""
from pathlib import Path
import hashlib
import json
import os
import struct
try:
    from .build_icons import ico_frames, SIZES
except ImportError:
    from build_icons import ico_frames, SIZES

ROLE_BY_NAME = {
    'asr2rpp.exe': 'app', 'asr2rpp-cli.exe': 'cli',
    'whisper-cli.exe': 'whisper', 'whisper-vad-speech-segments.exe': 'vad',
    'audiocpp_cli.exe': 'audio', 'audiocpp_gguf.exe': 'convert',
}


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def executable_sections(path):
    import pefile
    with pefile.PE(str(path)) as pe:
        return {s.Name.hex(): hashlib.sha256(s.get_data()).hexdigest()
                for s in pe.sections if s.Characteristics & 0x20000000}


def verify_icon(executable, icon):
    import pefile
    expected = ico_frames(icon)
    with pefile.PE(str(executable)) as pe:
        resources = {e.id: e for e in pe.DIRECTORY_ENTRY_RESOURCE.entries}
        groups, images = resources.get(14), resources.get(3)
        if groups is None or images is None:
            raise AssertionError(f'Missing executable icon: {executable}')
        group = next((e for e in groups.directory.entries if e.id == 1), None)
        if group is None:
            raise AssertionError(f'Missing primary icon group: {executable}')
        images_by_id = {e.id: e for e in images.directory.entries}
        for language in group.directory.entries:
            entry = language.data.struct
            raw = pe.get_data(entry.OffsetToData, entry.Size)
            assert struct.unpack_from('<HHH', raw) == (0, 1, len(SIZES))
            assert len(raw) == 6 + 14*len(SIZES)
            for i, (w, h, png) in enumerate(expected):
                rw, rh, _, _, planes, bits, size, image_id = struct.unpack_from('<BBBBHHIH', raw, 6+14*i)
                assert (rw or 256, rh or 256, planes, bits, size) == (w, h, 1, 32, len(png))
                candidates = images_by_id[image_id].directory.entries
                lang = next((x for x in candidates if x.id == language.id), candidates[0])
                resource = lang.data.struct
                assert pe.get_data(resource.OffsetToData, resource.Size) == png
    return list(SIZES)


def apply_icons(package, icon_directory, report):
    if os.name != 'nt':
        raise RuntimeError('Windows resource editing must run on Windows')
    import pefile
    from PyInstaller.utils.win32.icon import CopyIcons_FromIco
    package, icon_directory = Path(package).resolve(), Path(icon_directory).resolve()
    records = []
    exes = sorted(p for p in package.rglob('*') if p.suffix.lower() == '.exe')
    if not exes:
        raise AssertionError('Empty Windows package')
    for exe in exes:
        relative = exe.relative_to(package).as_posix()
        role = ROLE_BY_NAME.get(exe.name.casefold(), 'cli')
        icon = icon_directory/f'{role}.ico'
        before = sha256(exe)
        code = executable_sections(exe)
        if exe.parent == package and exe.name.casefold() in {'asr2rpp.exe', 'asr2rpp-cli.exe'}:
            # Never edit a onefile archive after PyInstaller has created it.
            verify_icon(exe, icon)
        else:
            with pefile.PE(str(exe)) as pe:
                if pe.OPTIONAL_HEADER.DATA_DIRECTORY[4].Size:
                    raise ValueError(f'Refusing to invalidate a signed third-party binary: {relative}')
            CopyIcons_FromIco(str(exe), [str(icon)])
            assert executable_sections(exe) == code, f'Code section changed: {relative}'
            verify_icon(exe, icon)
        records.append({'file': relative, 'role': role, 'sizes': list(SIZES),
                        'before_sha256': before, 'sha256': sha256(exe),
                        'executable_sections_unchanged': True})
    for manifest in (package/'engines').rglob('build-manifest.json'):
        meta = json.loads(manifest.read_text(encoding='utf-8'))
        updates = []
        for name in meta.get('files', {}):
            path = manifest.parent/Path(name.replace('\\', '/'))
            if path.suffix.lower() == '.exe' and path.is_file():
                updates.append({'file': name, 'before_sha256': meta['files'][name], 'sha256': sha256(path)})
                meta['files'][name] = sha256(path)
        meta['packaged_icons'] = {'source': 'ASR2RPP Material SVGs', 'executables': updates}
        manifest.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text(json.dumps({'all_executables_verified': True, 'executables': records}, indent=2), encoding='utf-8')
    return records
