"""instance types are defined here"""

from pathlib import Path
from typing import Tuple, NamedTuple, Optional

from PIL.JpegImagePlugin import JpegImageFile


class ObjectLabel(NamedTuple):
    """object label"""
    class_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]
    track_id: Optional[int] = None


class RawImageInstance(NamedTuple):
    """unprocessed, immutable image instance for storage"""
    impath: Path
    object_labels: Tuple[ObjectLabel, ...]


class ImageInstance(NamedTuple):
    """human readable frame instance"""
    im: JpegImageFile
    object_labels: Tuple[ObjectLabel, ...]
