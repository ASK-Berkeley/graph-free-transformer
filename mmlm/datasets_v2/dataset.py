"""
Main TextDataset class for datasets v2.

This module provides the refactored TextDataset, which follows a component-based
architecture consisting of a loader, a formatter, and optional transforms.
"""

from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional, Tuple

from .core.formatter import Formatter
from .loaders.base_loader import BaseLoader
from .core.binner import BinningSpec
from .core.transforms import Transform


class TextDataset(Dataset):
    """
    A PyTorch Dataset that combines a data loader and a formatter.

    This dataset class is the core of the v2 architecture. It retrieves a
    standardized `Molecule` object from a specified loader and then uses a
    formatter to convert that molecule into the text and tensor formats
    required for training a language model.
    """

    def __init__(
        self,
        loader: BaseLoader,
        formatter: Formatter,
        bin_spec: BinningSpec,
        transforms: Optional[List[Transform]] = None,
        first_force_only: bool = False,
    ):
        """
        Initialize the TextDataset.

        Args:
            loader: A data loader instance that inherits from `BaseLoader`.
            formatter: A formatter instance that inherits from `Formatter`.
            bin_spec: A `BinningSpec` instance for discretizing continuous values.
            transforms: An optional list of transforms to apply to the molecule.
        """
        self.loader = loader
        self.formatter = formatter
        self.bin_spec = bin_spec
        self.transforms = transforms or []

        self.first_force_only = first_force_only

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.loader)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Fetch a single formatted data point.

        Args:
            idx: The index of the data point to retrieve.

        Returns:
            A tuple containing the formatted output from the formatter.
            This typically includes (text, cont) or (text, cont, atom_ids).
        """
        # 1. Retrieve the standardized Molecule object from the loader
        molecule = self.loader[idx]

        # 2. Apply transforms if any
        for transform in self.transforms:
            molecule = transform(molecule)

        if self.first_force_only:
            molecule.F = molecule.F[:1]

        # 3. Format the molecule using the specified formatter
        formatted_output = self.formatter(molecule, self.bin_spec)

        return formatted_output
