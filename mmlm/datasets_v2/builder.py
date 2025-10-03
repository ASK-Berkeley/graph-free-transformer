"""
This module contains the builder function for constructing datasets
based on the v2 pipeline and a Hydra config.
"""

import logging
import numpy as np

from hydra.utils import instantiate
from omegaconf import DictConfig

from .core.binner import BinningSpec
from .dataset import TextDataset
from .config import DatasetV2Cfg


def build_dataset(cfg: DatasetV2Cfg):
    """
    Builds a dataset from the given configuration.
    This function orchestrates the instantiation of the loader, transforms,
    formatter.
    """
    logging.info("Building dataset from v2 config...")

    # 1. Instantiate the core components from the config
    loader = cfg.loader
    transforms = cfg.transforms if cfg.transforms else None

    bin_spec = BinningSpec.load(cfg.bin_spec_path)
    first_force_only = cfg.first_force_only
    formatter = cfg.formatter

    # 2. Assemble the base TextDataset
    base_ds = TextDataset(
        loader=loader, formatter=formatter, bin_spec=bin_spec, transforms=transforms, first_force_only=first_force_only
    )
    logging.info(f"Base dataset created with {len(base_ds)} samples.")

    
    final_ds = base_ds

    logging.info("Dataset construction complete.")
    return final_ds
