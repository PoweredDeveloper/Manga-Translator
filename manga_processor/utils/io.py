import os
import json
import numpy
import shutil

from ..mt_logger import logger
from pathlib import Path

NP_BOOL_TYPES = (numpy.bool_, numpy.bool8)
NP_FLOAT_TYPES = (numpy.float_, numpy.float16, numpy.float32, numpy.float64)
NP_INT_TYPES = (
    numpy.int_,
    numpy.int8,
    numpy.int16,
    numpy.int32,
    numpy.int64,
    numpy.uint,
    numpy.uint8,
    numpy.uint16,
    numpy.uint32,
    numpy.uint64,
)

# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, numpy.ScalarType):
            if isinstance(obj, NP_BOOL_TYPES):
                return bool(obj)
            elif isinstance(obj, NP_FLOAT_TYPES):
                return float(obj)
            elif isinstance(obj, NP_INT_TYPES):
                return int(obj)
        return json.JSONEncoder.default(self, obj)
    
def empty_directoty(directory_path: Path) -> None:
    for filename in os.listdir(str(directory_path)):
        file_path = os.path.join(str(directory_path), filename)

        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as exception:
            logger.exception(f'Failed to delete {file_path}. Reason: {exception}')