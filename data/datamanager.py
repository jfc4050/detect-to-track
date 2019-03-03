"""abstract datamanager class"""

import abc

class DataManager(abc.ABC):
    """handles dataloading"""
    @abc.abstractmethod
    def __getitem__(self, i):
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError
