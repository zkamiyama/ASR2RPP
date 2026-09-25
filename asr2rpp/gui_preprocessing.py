"""Compatibility entry point; the DCC GUI is the sole maintained implementation."""
from .gui_dcc import MainWindow, PreferencesDialog, STYLE, default_storage_hint, main

__all__ = ['MainWindow', 'PreferencesDialog', 'STYLE', 'default_storage_hint', 'main']

if __name__ == '__main__':
    raise SystemExit(main())
