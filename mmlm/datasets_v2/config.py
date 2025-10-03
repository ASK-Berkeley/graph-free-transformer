"""
This module defines the structured dataclasses for configuring the datasets_v2
pipeline using Hydra. These configs are used to instantiate the necessary
components like loaders, formatters, and transforms.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

# --------------------- Root Dataset Config ---------------------

@dataclass
class DatasetV2Cfg:
    """
    Root configuration for building a dataset using the v2 pipeline.
    This dataclass is intended to be the entry point for Hydra instantiation.
    """

    loader: Any
    formatter: Any
    bin_spec_path: str
    transforms: List[Any] = field(default_factory=list)
    first_force_only: bool = False
