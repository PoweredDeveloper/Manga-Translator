import os
import sys
import typing

from pathlib import Path
from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *
from PySide6.QtUiTools import *

from .pages import *
from ..utils.io import suggest_project_name
from ..mt_config import PROJECTS_PATH

class MT_Window():
    def __init__(
        self,
        argv: typing.Sequence[str],
        size: tuple[int, int] = (1280, 720),
        caption: str = "Manga Translator"
    ) -> None:

        self._app = QApplication(argv)
        self._app.setStyle('windowsvista')
        self._pyside_window = QUiLoader().load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gui', 'start_screen.ui'))
        self._pyside_window.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.set_size(size[0], size[1])
        self.set_caption(caption)
        # TODO: Window Icon

    def run(self) -> None:
        self._pyside_window.show()
        sys.exit(self._app.exec())

    def set_size(self, w: int = 1280, h: int = 720) -> None:
        self._pyside_window.setFixedSize(QSize(w, h))

    def set_caption(self, caption: str = "Manga Translator") -> None:
        self._pyside_window.setWindowTitle(caption)

class MT_PySide_Window(QMainWindow):
    def __init__(self) -> None:
        super(MT_PySide_Window, self).__init__()
        self._setup_window()
        self.switch_widget(WindowPage.START)

    def _setup_window(self) -> None:
        menubar = QMenuBar(self)
        file_menu = menubar.addMenu('&File')
        new_project_action = QAction('New Project', self)
        new_project_action.triggered.connect(self.new_project)
        file_menu.addAction(new_project_action)
        menubar.addMenu('&Settings')
        menubar.addMenu('&Credits')
        self.setMenuBar(menubar)

        self._stacked_layout = QStackedLayout()
        
        self.pages: list[QWidget] = [
            StartPageWidget(self),
            EditPageWidget(self),
            DownloadPageWidget(self),
            SettingsPageWidget(self)
        ]

        for page in self.pages:
            self._stacked_layout.addWidget(page)

        central_widget = QWidget()
        central_widget.setLayout(self._stacked_layout)
        self.setCentralWidget(central_widget)

    def switch_widget(self, page: WindowPage) -> None:
        self._stacked_layout.setCurrentIndex(page.value)

    def new_project(self) -> None:
        dialog = NewProjectDialog(self)
        path = dialog.get_path()
        if path.exists(): self.open_project(path)

    def open_project(self, project_path: Path = None) -> None:
        if project_path:
            self.pages[WindowPage.EDIT.value].load_project(project_path)
        else:
            self.pages[WindowPage.EDIT.value].load_project(Path(QFileDialog.getExistingDirectory(None, 'Select a project folder:', os.path.expanduser('~'))))

        self.switch_widget(WindowPage.EDIT)

class NewProjectDialog(QDialog):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.setWindowTitle('New Project')
        self.setFixedSize(600, 400)
        self.project_path = PROJECTS_PATH
        self.project_name = suggest_project_name(Path(self.project_path))

        self.setup_layout()

    def setup_layout(self) -> None:
        self.main_layout = QVBoxLayout()

        proj_name_label = QLabel('Project name:', self)
        self.proj_name_input = QLineEdit(self)
        self.path_preview = QLabel('', self)
        self.proj_name_input.textChanged.connect(self.update_path_preview)
        self.proj_name_input.setText(self.project_name)
        self.main_layout.addWidget(proj_name_label)
        self.main_layout.addWidget(self.proj_name_input)

        proj_path_label = QLabel('Project path:', self)
        select_layout = QHBoxLayout()
        self.proj_path_select = QComboBox()
        self.proj_path_select.currentTextChanged.connect(self.update_path)
        self.proj_path_select.addItem(str(self.project_path.absolute()))
        proj_path_push = QPushButton('...')
        proj_path_push.setMaximumWidth(25)
        proj_path_push.clicked.connect(self.choose_directory)

        select_layout.addWidget(self.proj_path_select)
        select_layout.addWidget(proj_path_push)
        self.main_layout.addWidget(proj_path_label)
        self.main_layout.addLayout(select_layout)

        path_preview_label = QLabel('Project will be created at:', self)
        self.update_path_preview()

        self.main_layout.addStretch(1)
        
        footer_layout = QVBoxLayout()
        self.button_layout = QHBoxLayout()
        create_button = QPushButton('Create')
        create_button.clicked.connect(self.create_project)
        cancel_button = QPushButton('Cancel')
        cancel_button.clicked.connect(lambda: self.close())

        self.button_layout.addStretch(1)

        self.button_layout.addWidget(cancel_button)
        self.button_layout.addWidget(create_button)

        footer_layout.addWidget(path_preview_label)
        footer_layout.addWidget(self.path_preview)
        footer_layout.addLayout(self.button_layout)

        self.main_layout.addLayout(footer_layout)

        self.setLayout(self.main_layout)

    def choose_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(None, 'Select a folder, where project will be created:', os.path.expanduser('~'))
        self.proj_path_select.addItem(str(Path(path).absolute()))
        self.proj_path_select.setCurrentIndex(self.proj_path_select.findText(str(Path(path).absolute())))
        self.update_path()

    def update_path_preview(self) -> None:
        self.project_name = self.proj_name_input.text()
        self.path_preview.setText(str(Path(self.project_path).joinpath(str(self.project_name))))

    def update_path(self) -> None:
        self.project_path = self.proj_path_select.currentText()
        self.proj_name_input.setText(suggest_project_name(Path(self.project_path)))
        self.update_path_preview()

    def create_project(self) -> None:
        project_folder = os.path.join(self.project_path, self.project_name)
        os.mkdir(project_folder)
        os.mkdir(os.path.join(project_folder, 'cache'))
        Path(os.path.join(project_folder, self.project_name + '.prj')).touch()
        self.close()

    def get_path(self) -> Path:
        self.show()
        self.exec()
        return Path(self.project_path).joinpath(self.project_name)
