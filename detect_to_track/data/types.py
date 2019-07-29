"""data types."""

import abc
from pathlib import Path
from typing import Tuple, NamedTuple, Optional

from PIL import Image


class DataSampler(abc.ABC):
    """general data sampler object for non-deterministic data sampling.
    handles data i/o and conversion to common format."""

    @abc.abstractmethod
    def sample(self):
        raise NotImplementedError


class DataManager(abc.ABC):
    """general data manager object for iterating through entire dataset.
    handles data i/o and conversion to common format."""

    @abc.abstractmethod
    def __getitem__(self, i):
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError


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
