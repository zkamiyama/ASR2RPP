"""Application identity shared by source and frozen GUI entry points."""
import sys
from .catalog import assets_root


def configure_window(window):
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication
    icon = QIcon(str(assets_root() / 'assets' / 'branding' / 'app.svg'))
    if icon.isNull():
        raise RuntimeError('Application icon is missing; extract the complete ZIP')
    window.setWindowIcon(icon)
    app = QApplication.instance()
    if app is not None:
        app.setWindowIcon(icon)
        app.setApplicationDisplayName('ASR2RPP')
    if sys.platform == 'win32':
        # No registry changes. Stable identity for Windows taskbar grouping.
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('ASR2RPP.Desktop')
