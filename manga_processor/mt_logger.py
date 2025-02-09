import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

console_handler = logging.StreamHandler()
log_filepath = os.path.join(os.getcwd(), 'logs', f'translator-{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}.log')
file_handler = logging.FileHandler(log_filepath, encoding = 'utf-8', mode = 'a')
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt = "{asctime}:{msecs} [{levelname}]: {message}. ({module}) Line: {lineno}",
    datefmt = "%d-%m-%Y %H:%M:%S",
    style = "{"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)