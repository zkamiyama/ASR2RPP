# Launcher names must differ beyond case on Windows.
a = Analysis(['main.py'], pathex=[], binaries=[], datas=[],
             hiddenimports=['gui', 'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets'],
             hookspath=[], hooksconfig={}, runtime_hooks=[], excludes=['tkinter'], noarchive=False)
pyz = PYZ(a.pure)
cli = EXE(pyz, a.scripts, [], exclude_binaries=True, name='asr2rpp-cli', console=True, upx=False)
gui = EXE(pyz, a.scripts, [], exclude_binaries=True, name='ASR2RPP', console=False, upx=False)
collection = COLLECT(cli, gui, a.binaries, a.datas, strip=False, upx=False, name='ASR2RPP')
