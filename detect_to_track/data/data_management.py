"""data samplers and managers"""

import abc
from typing import Tuple

from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor

from .encoding import FRCNNLabelEncoder


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


class ImageDataset(Dataset):
    """dataset for training single stage object detectors and RPNs on images.
    bridges datamanagers and encoders, returns network-readable inputs and
    outputs.

    Args:
        data_manager: data manager
        encoder: data encoder. takes ObjectLabels and returns network-readable
            labels.
        net_input_size: height and width of resized network input
    """

    def __init__(
        self, data_manager: DataManager, encoder: FRCNNLabelEncoder, net_input_size: int
    ) -> None:
        self.data_manager = data_manager
        self._encoder = encoder

        self.im_to_x = Compose([Resize((net_input_size, net_input_size)), ToTensor()])

    def __getitem__(self, i: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """gets image and labels from data manager, then converts to network
        readable inputs and outputs.

        Args:
            i: used to index into data manager.

        Returns:
            loss_weights: (|A|,) anchorwise loss weights.
            x: (C, H, W) resized image tensor
            c_star: (|A|,) ground-truth anchorwise class assignments.
            b_star: (|A|, 4) anchor offsets from closest ground-truth
                bounding box.
        """
        instance = self.data_manager[i]
        x = self.im_to_x(instance.im)
        loss_weights, c_star, b_star = self._encoder(instance.labels)

        return loss_weights, x, c_star, b_star

    def __len__(self) -> int:
        return len(self.data_manager)
