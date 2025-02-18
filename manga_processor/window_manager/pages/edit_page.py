from pathlib import Path
from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *

from .structures import WindowPage

class EditPageWidget(QWidget):
    def __init__(self, parent = None) -> None:
        super(EditPageWidget, self).__init__(parent)
        self.base_window = self.parent()
        self.setup_layout()
    
    def setup_layout(self) -> None:
        main_layout = QGridLayout()

        main_layout.addWidget(QLabel('Edit'), 0, 0, alignment = Qt.AlignmentFlag.AlignCenter)

        self.setLayout(main_layout)

    def load_project(self, path: Path) -> None:
        project_file = path.joinpath(path.stem + '.prj')
        if not project_file.exists():
            print('Yo')
            self.base_window.switch_widget(WindowPage.START)