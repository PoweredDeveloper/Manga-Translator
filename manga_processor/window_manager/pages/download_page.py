from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *

class DownloadPageWidget(QWidget):
    def __init__(self, parent = None) -> None:
        super(DownloadPageWidget, self).__init__(parent)
        self.base_window = self.parent()
        self.setup_layout()

    def setup_layout(self) -> None:
        main_layout = QGridLayout()

        main_layout.addWidget(QLabel('Download'), 0, 0, alignment = Qt.AlignmentFlag.AlignCenter)

        self.setLayout(main_layout)
