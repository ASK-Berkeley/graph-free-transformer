"""
This package contains the refactored datasets_v2 implementation.

The main components are:
- `Molecule`: A standardized dataclass for molecular data.
- `BaseLoader`: The abstract base class for data loaders.
- `BinningSpec`: The class for handling discretization of continuous values.
- `Formatter`: The base class for formatting molecules into text.
- `TextDataset`: The main PyTorch Dataset class.
"""

from .core.binner import BinningSpec, SklearnBinner
from .core.formatter import AtomFormatter, Formatter, StandardFormatter
from .core.molecule import Molecule
from .core.transforms import PermutationTransform, RotationTransform, Transform
from .dataset import TextDataset
from .loaders.base_loader import BaseLoader
from .loaders.omol_loader import OmolLoader
from .builder import build_dataset

__all__ = [
    # Core
    "BinningSpec",
    "SklearnBinner",
    "AtomFormatter",
    "Formatter",
    "StandardFormatter",
    "Molecule",
    "PermutationTransform",
    "RotationTransform",
    "Transform",
    # Dataset
    "TextDataset",
    # Builder
    "build_dataset",
    # Loaders
    "BaseLoader",
    "OmolLoader",
]
