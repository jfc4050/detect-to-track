"""instance types are defined here"""

from typing import Tuple, NamedTuple

from PIL.JpegImagePlugin import JpegImageFile


class ObjectLabel(NamedTuple):
    """object label"""
    track_id: int
    class_id: int
    class_name: str
    bbox: Tuple[float]


class FrameInstance(NamedTuple):
    """human readable frame instance"""
    im: JpegImageFile
    object_labels: Tuple[ObjectLabel, ...]
