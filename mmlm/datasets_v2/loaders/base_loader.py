"""
Abstract base class for data loaders.
"""

from abc import ABC, abstractmethod
from torch.utils.data import Dataset

from mmlm.datasets_v2.core.molecule import Molecule


class BaseLoader(Dataset, ABC):
    """
    Abstract base class for all data loaders in the datasets_v2 architecture.

    Loaders are responsible for reading data from a source (like LMDB, xyz, etc.)
    and converting it into a standardized `Molecule` object. They must implement
    the `__len__` and `__getitem__` methods of the PyTorch Dataset class.
    """

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Molecule:
        """
        Fetch a single molecule data point by index.

        Args:
            idx: The index of the data point to retrieve.

        Returns:
            A `Molecule` object representing the requested data point.
        """
        pass
