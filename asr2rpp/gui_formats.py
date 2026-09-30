"""Compact output-format chips; selection is data, never a plugin/import name."""
from PySide6.QtCore import QSize, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QWidget, QHBoxLayout, QFrame, QLabel, QToolButton, QMenu
from .catalog import assets_root
from .outputs import FORMATS, validate_formats


class FormatSelector(QWidget):
    changed = Signal()

    def __init__(self, formats=('rpp',), language='ja', parent=None):
        super().__init__(parent)
        self.language = language
        self._formats = ()
        self.buttons = {}
        self.layout_ = QHBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.layout_.setSpacing(4)
        self.chips = QHBoxLayout()
        self.chips.setSpacing(4)
        self.layout_.addLayout(self.chips)
        self.add_button = QToolButton(self)
        self.add_button.setObjectName('formatAdd')
        self.add_button.setIcon(self._icon('add'))
        self.add_button.setIconSize(QSize(14, 14))
        self.add_button.setFixedSize(20, 20)
        self.add_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.menu = QMenu(self.add_button)
        self.add_button.setMenu(self.menu)
        self.menu.aboutToShow.connect(self.refresh_menu)
        self.layout_.addWidget(self.add_button)
        self.layout_.addStretch()
        self.set_formats(formats)
        self.set_language(language)

    @staticmethod
    def _icon(name):
        return QIcon(str(assets_root()/'assets/icons'/f'{name}.svg'))

    def formats(self):
        return self._formats

    def set_formats(self, formats):
        if formats:
            formats = validate_formats(formats)
        elif not isinstance(formats, (tuple, list)):
            raise ValueError('Invalid format selection')
        self._formats = tuple(formats)
        while self.chips.count():
            widget = self.chips.takeAt(0).widget()
            widget.setParent(None)
            widget.deleteLater()
        self.buttons.clear()
        for key in self._formats:
            chip = QFrame(self)
            chip.setObjectName('formatChip')
            row = QHBoxLayout(chip)
            row.setContentsMargins(6, 0, 2, 0)
            row.setSpacing(2)
            row.addWidget(QLabel(key.upper(), chip))
            button = QToolButton(chip)
            button.setObjectName('formatRemove')
            button.setIcon(self._icon('close'))
            button.setIconSize(QSize(12, 12))
            button.setFixedSize(16, 18)
            button.clicked.connect(lambda _checked=False, key=key: self.remove_format(key))
            row.addWidget(button)
            self.buttons[key] = button
            self.chips.addWidget(chip)
        self.add_button.setEnabled(len(self._formats) < len(FORMATS))
        self.set_language(self.language)
        self.refresh_menu()
        self.changed.emit()

    def add_format(self, key):
        if key not in self._formats:
            self.set_formats((*self._formats, key))

    def remove_format(self, key):
        self.set_formats(tuple(value for value in self._formats if value != key))

    def refresh_menu(self):
        self.menu.clear()
        for key in FORMATS:
            if key not in self._formats:
                label = 'JSON · 全結果 / Raw results' if key == 'json' else key.upper()
                action = self.menu.addAction(label)
                action.setData(key)
                action.triggered.connect(lambda _checked=False, key=key: self.add_format(key))

    def set_language(self, language):
        self.language = language
        self.add_button.setToolTip('出力形式を追加' if language == 'ja' else 'Add output format')
        self.add_button.setAccessibleName(self.add_button.toolTip())
        for key, button in self.buttons.items():
            text = f'{key.upper()}を削除' if language == 'ja' else f'Remove {key.upper()}'
            button.setToolTip(text)
            button.setAccessibleName(text)
