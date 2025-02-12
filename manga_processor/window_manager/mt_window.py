import os
import sys
import typing

from pathlib import Path
from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *

from ..utils.io import empty_directoty
from ..page_handler.image_processor import MT_ImageProcessor

class MT_Window():
    def __init__(
        self,
        argv: typing.Sequence[str],
        size: tuple[int, int] = (720, 480),
        caption: str = "Manga Translator"
    ) -> None:

        self._app = QApplication(argv)
        self._pyside_window = MT_PySide_Window()
        self.set_size(size[0], size[1])
        self.set_caption(caption)
        # TODO: Window Icon

    def run(self) -> None:
        self._pyside_window.show()
        sys.exit(self._app.exec())

    def set_size(self, w: int = 720, h: int = 480) -> None:
        self._pyside_window.setFixedSize(QSize(w, h))

    def set_caption(self, caption: str = "Manga Translator") -> None:
        self._pyside_window.setWindowTitle(caption)

class MT_PySide_Window(QMainWindow):
    def __init__(self) -> None:
        super(MT_PySide_Window, self).__init__()
        # Setting main PySide layout
        self.init_layout()

        # FIXME: Paths to cache
        cache_path = Path(os.path.join(os.getcwd(), 'cache'))
        model_path = Path(os.path.join(os.getcwd(), 'models', 'comictextdetector.pt'))
        self.image_processor = MT_ImageProcessor(cache_path, model_path)
        empty_directoty(Path(os.path.join(os.getcwd(), 'cache')))

    def init_layout(self) -> None:
        self.setCentralWidget(QWidget())

        main_layout = QGridLayout()
        # TODO: Layout
        upload_button = QPushButton('Upload Image')
        upload_button.clicked.connect(self.upload_file)

        process_button = QPushButton('Process Image')
        process_button.clicked.connect(self.process_image)

        main_layout.addWidget(upload_button, 0, 0, alignment = Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(process_button, 1, 0, alignment = Qt.AlignmentFlag.AlignCenter)

        self.centralWidget().setLayout(main_layout)

    def upload_file(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, 'Выберите изображения...', os.path.join(os.environ['USERPROFILE'], 'Desktop'))
        if len(paths) == 0: return
        self.image_paths = [Path(path) for path in paths]

    def process_image(self) -> None:
        if self.image_paths == '': return
        self.image_processor.clean(self.image_paths, Path(''), True)
