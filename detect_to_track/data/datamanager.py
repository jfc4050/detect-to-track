"""abstract datamanager class"""

import abc

class DataManager(abc.ABC):
    """general data manaager object. handles data i/o and conversion to
    common format."""
    @abc.abstractmethod
    def __getitem__(self, i):
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError
