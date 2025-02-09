import typing

from enum import Enum

from .mt_logger import logger
from .window_manager import MT_Window

class RunMode(Enum):
    MT_MODE_WINDOW = 0
    MT_MODE_CONSOLE = 1

class MT_Processor:
    def __init__(self, argv: typing.Sequence[str]) -> None:
        self._window = MT_Window(argv)
    
    def run(self, width: int, height: int, mode = RunMode.MT_MODE_WINDOW) -> None:
        logger.debug("======= STARTING =======")
        if mode == RunMode.MT_MODE_WINDOW:
            self._window.set_size(width, height)
            self._window.run()