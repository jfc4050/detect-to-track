"""data types."""

import abc
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
    images and labels loaded lazily."""

    impath: Path
    labelpath: Path


class ImageInstance(NamedTuple):
    """human readable frame instance
    images and labels are loaded lazily."""

    im: Image.Image
    labels: Tuple[ObjectLabel, ...]


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


class DataManagerWrapper(DataManager):
    """wraps a `DataSampler` object so that it behaves like a `DataManager`."""

    def __init__(self, sampler: DataSampler, nominal_len: int) -> None:
        self.sampler = sampler
        self.nominal_len = nominal_len

    def __getitem__(self, i: int) -> Tuple[ImageInstance, ImageInstance]:
        return self.sampler.sample()

    def __len__(self) -> int:
        return self.nominal_len
