import os

from pathlib import Path

from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *

from ...page_handler.image_processor import MT_ImageProcessor
from ...mt_config import CACHE_PATH

class StartPageWidget(QWidget):
    def __init__(self, parent = None):
        super(StartPageWidget, self).__init__(parent)
        self.base_window = self.parent()

        model_path = Path(os.path.join(os.getcwd(), 'models', 'comictextdetector.pt'))
        self.image_processor = MT_ImageProcessor(CACHE_PATH, model_path)

        self.setup_layout()

    def setup_layout(self) -> None:
        main_layout = QGridLayout()

        project_buttons_layout = QHBoxLayout(self)
        new_project_button = QPushButton('New project')
        new_project_button.clicked.connect(self.base_window.new_project)
        open_project_button = QPushButton('Open Project')
        open_project_button.clicked.connect(self.base_window.open_project)
        project_buttons_layout.addStretch(1)
        project_buttons_layout.addWidget(new_project_button)
        project_buttons_layout.addWidget(open_project_button)    
        project_buttons_layout.addStretch(1)

        main_layout.addLayout(project_buttons_layout, 0, 0, Qt.AlignmentFlag.AlignCenter)

        self.setLayout(main_layout)

    def upload_file(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, 'Выберите изображения...', os.path.join(os.environ['USERPROFILE'], 'Desktop'))
        if len(paths) == 0: return
        self.image_paths = [Path(path) for path in paths]

    def process_image(self) -> None:
        if self.image_paths == '': return
        self.image_processor.clean(self.image_paths, Path(''), True)
