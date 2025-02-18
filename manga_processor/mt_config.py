import os
from pathlib import Path

ASSETS_PATH = Path(os.path.join(os.getcwd(), 'assets'))
CACHE_PATH = Path(os.path.join(os.getcwd(), 'cache'))
PROJECTS_PATH = Path(os.path.join(os.path.join(os.environ['USERPROFILE'], 'Documents'), 'Manga Translator Projects'))

FONT_ROBOTO_PATH = Path(os.path.join(ASSETS_PATH, 'Roboto.ttf'))