"""Render the original checked-in SVG brand marks to multi-resolution Windows icons.

No downloaded fonts or image libraries are needed. Source SVGs and license are
version controlled; ICO generation is part of the release, never first startup.
"""
from pathlib import Path
import json
import struct

ROOT = Path(__file__).resolve().parents[1]
SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
ROLES = ('app', 'cli', 'whisper', 'vad', 'audio', 'convert')


def ico_frames(path):
    data = Path(path).read_bytes()
    if len(data) < 6 or struct.unpack_from('<HH', data) != (0, 1):
        raise ValueError('Invalid ICO header')
    count = struct.unpack_from('<H', data, 4)[0]
    if not 1 <= count <= 32 or len(data) < 6 + 16 * count:
        raise ValueError('Invalid ICO directory')
    frames = []
    for i in range(count):
        w, h, colors, reserved, planes, bits, length, offset = struct.unpack_from('<BBBBHHII', data, 6 + 16*i)
        if offset < 6 + 16*count or offset + length > len(data) or not length:
            raise ValueError('ICO frame outside file')
        frames.append((w or 256, h or 256, data[offset:offset+length]))
    return frames


def build_icons(destination, preview=None):
    from PySide6.QtCore import Qt, QByteArray, QBuffer, QIODevice, QRectF
    from PySide6.QtGui import QImage, QPainter, QColor
    from PySide6.QtSvg import QSvgRenderer
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    result = {}
    sheet = QImage(640, 6*80, QImage.Format.Format_ARGB32)
    sheet.fill(QColor('#eef1f4'))
    painter = QPainter(sheet)
    for row, role in enumerate(ROLES):
        renderer = QSvgRenderer(str(ROOT/'assets'/'branding'/f'{role}.svg'))
        if not renderer.isValid():
            raise ValueError(f'Invalid SVG: {role}')
        blobs = []
        for n in SIZES:
            image = QImage(n, n, QImage.Format.Format_ARGB32)
            image.fill(Qt.GlobalColor.transparent)
            p = QPainter(image)
            renderer.render(p, QRectF(0, 0, n, n))
            p.end()
            buf = QBuffer()
            if not buf.open(QIODevice.OpenModeFlag.WriteOnly) or not image.save(buf, 'PNG'):
                raise OSError('Could not render icon')
            blobs.append(bytes(buf.data()))
        offset = 6 + 16*len(SIZES)
        data = bytearray(struct.pack('<HHH', 0, 1, len(SIZES)))
        for n, blob in zip(SIZES, blobs):
            data.extend(struct.pack('<BBBBHHII', n % 256, n % 256, 0, 0, 1, 32, len(blob), offset))
            offset += len(blob)
        data.extend(b''.join(blobs))
        icon = destination/f'{role}.ico'
        icon.write_bytes(data)
        assert [w for w, _, _ in ico_frames(icon)] == list(SIZES)
        result[role] = icon
        # Actual-size samples on light and dark backgrounds, suitable for review.
        for col, n in enumerate((16, 24, 32, 48, 64)):
            img = QImage.fromData(blobs[SIZES.index(n)], 'PNG')
            x, y = 16 + col*62, row*80 + (80-n)//2
            painter.drawImage(x, y, img)
            painter.fillRect(328, row*80, 312, 80, QColor('#1b1f24'))
        for col, n in enumerate((16, 24, 32, 48, 64)):
            painter.drawImage(336 + col*62, row*80 + (80-n)//2,
                              QImage.fromData(blobs[SIZES.index(n)], 'PNG'))
    painter.end()
    if preview:
        preview = Path(preview)
        preview.parent.mkdir(parents=True, exist_ok=True)
        if not sheet.save(str(preview), 'PNG'):
            raise OSError('Could not write icon preview')
    return result


if __name__ == '__main__':
    import sys
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/'build'/'icons'
    build_icons(folder, ROOT/'reports'/'executable-icons.png')
