"""instance types are defined here"""

from pathlib import Path
from typing import Tuple, NamedTuple, Optional

from PIL import Image


class ObjectLabel(NamedTuple):
    """object label"""
    class_id: int
    class_name: str
    box: Tuple[float, float, float, float]
    track_id: Optional[int] = None


class RawImageInstance(NamedTuple):
    """unprocessed, immutable image instance for storage
    images are loaded lazily, labels are loaded eagerly"""
    impath: Path
    labels: Tuple[ObjectLabel, ...]


class ImageInstance(NamedTuple):
    """human readable frame instance
    images are loaded lazily, labels are loaded eagerly"""
    im: Image.Image
    labels: Tuple[ObjectLabel, ...]
