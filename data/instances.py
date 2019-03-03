"""instance types are defined here"""

from typing import NamedTuple

import numpy as np
from PIL.JpegImagePlugin import JpegImageFile


class FrameInstance(NamedTuple):
    """human readable frame instance"""
    im: JpegImageFile
    classes: np.ndarray
    bboxes: np.ndarray
