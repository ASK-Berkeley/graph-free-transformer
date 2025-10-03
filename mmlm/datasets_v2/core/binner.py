"""
Binning specification for datasets v2.

This module provides the BinningSpec class that handles discretization of continuous values
into tokens for molecular machine learning models.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Iterator
from pathlib import Path
import numpy as np
import logging
import re
from abc import ABC, abstractmethod
from tqdm import tqdm
from sklearn.preprocessing import KBinsDiscretizer as SklearnKBinsDiscretizer


from .molecule import Molecule


class KBinsDiscretizer:
    def __init__(self, bin_edges):
        self.bin_edges = [bin_edges]

    def transform(self, x):
        bin_edges = self.bin_edges

        x = np.array(x)
        is_num = False
        if len(x.shape) == 0:
            x = np.array([[x]])
            is_num = True
        if len(x.shape) == 1:
            x = np.expand_dims(x, 1)

        for jj in range(x.shape[1]):
            x[:, jj] = np.searchsorted(bin_edges[jj][1:-1], x[:, jj], side="right")

        return int(x[0][0]) if is_num else x.astype(int)

    def inverse_transform(self, x):
        bin_centers = (self.bin_edges[0][1:] + self.bin_edges[0][:-1]) / 2
        return bin_centers[x]

    # Should be of shape (B, n_bins)
    def weighted_inverse_transform(self, x):
        bin_centers = (self.bin_edges[0][1:] + self.bin_edges[0][:-1]) / 2
        return np.sum(bin_centers * x, axis=-1)


@dataclass
class BinningSpec:
    """
    Frozen binning specification for transforming continuous values to discrete tokens.

    This class contains pre-computed bin edges and metadata for discretizing continuous
    molecular properties into tokens. It serves as the "Phase 2" component that applies
    binning efficiently at runtime.

    Attributes:
        field_bins: Mapping from field names to bin edges arrays
        n_bins: Number of bins per field
        method: Binning method used per field ("quantile", "uniform", etc.)
        created_at: Optional timestamp of creation
        dataset_uuid: Optional dataset identifier for provenance

    Examples:
        >>> # Load from existing format
        >>> spec = BinningSpec.from_current_format("/path/to/bins/folder")
        >>>
        >>> # Transform values
        >>> positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> pos_bins = spec.transform("pos", positions)
        >>>
        >>> # Save/load
        >>> spec.save("binning_spec.npz")
        >>> loaded_spec = BinningSpec.load("binning_spec.npz")
    """

    field_bins: Dict[str, np.ndarray]  # field_name -> bin_edges
    n_bins: Dict[str, int]  # field_name -> n_bins
    method: Dict[str, str]  # field_name -> "quantile"/"uniform"

    # Optional metadata
    created_at: Optional[str] = None  # ISO timestamp
    dataset_uuid: Optional[str] = None  # Dataset provenance

    def transform(self, field: str, values: np.ndarray) -> np.ndarray:
        """
        Transform continuous values to bin indices.

        Args:
            field: Name of the field to transform ("pos", "force", "target", etc.)
            values: Continuous values to discretize

        Returns:
            Array of bin indices

        Raises:
            KeyError: If field is not in the binning specification
        """
        if field not in self.field_bins:
            raise KeyError(
                f"Field '{field}' not found in binning specification. "
                f"Available fields: {list(self.field_bins.keys())}"
            )

        discretizer = KBinsDiscretizer(bin_edges=self.field_bins[field])
        return discretizer.transform(values)

    def inverse_transform(self, field: str, bin_indices: np.ndarray) -> np.ndarray:
        """
        Transform bin indices back to continuous values using bin centers.

        Args:
            field: Name of the field to inverse transform
            bin_indices: Bin indices to convert back

        Returns:
            Array of continuous values (bin centers)
        """
        if field not in self.field_bins:
            raise KeyError(f"Field '{field}' not found in binning specification")

        discretizer = KBinsDiscretizer(bin_edges=self.field_bins[field])
        return discretizer.inverse_transform(bin_indices)

    def save(self, path: Path):
        """
        Save binning specification to disk.

        Args:
            path: Path to save the specification (.npz file)
        """
        path = Path(path)
        data = asdict(self)
        np.savez(path, **data)
        logging.info(f"Saved BinningSpec to {path}")

    @classmethod
    def load(cls, path: Path) -> "BinningSpec":
        """
        Load binning specification from disk.

        Args:
            path: Path to load from (.npz file)

        Returns:
            BinningSpec instance
        """
        path = Path(path)
        data = np.load(path, allow_pickle=True)

        # Convert numpy arrays back to proper format
        spec_data = {}
        for key, value in data.items():
            if key.endswith(".npy"):
                key = key[:-4]  # Remove .npy extension
            spec_data[key] = value.item() if value.ndim == 0 else value

        logging.info(f"Loaded BinningSpec from {path}")
        return cls(**spec_data)

    @classmethod
    def from_current_format(cls, folder_path: Path) -> "BinningSpec":
        """
        Adapter method to load from existing .npy bin files for migration.

        This method scans a folder for files matching the pattern:
        bins_new_{n_bins}{method_name}.npy

        Where method_name is empty for quantile binning, or something like "_uniform"
        for other methods.

        Args:
            folder_path: Path to folder containing binning files

        Returns:
            BinningSpec instance containing all field binning information

        Examples:
            >>> spec = BinningSpec.from_current_format("/path/to/omol/train_4M/")
            >>> print(spec.field_bins.keys())  # ['pos', 'force', 'target', ...]
        """
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            raise ValueError(f"Folder path does not exist: {folder_path}")

        # Pattern to match: bins_new_{n_bins}{method_name}.npy
        pattern = re.compile(r"^bins_new_(\d+)(_\w+)?\.npy$")

        field_bins = {}
        n_bins = {}
        method = {}

        # Scan folder for matching files
        binning_files = []
        for file_path in folder_path.glob("bins_new_*.npy"):
            match = pattern.match(file_path.name)
            if match:
                n_bins_str = match.group(1)
                method_suffix = match.group(2) or ""  # Empty string for quantile
                binning_files.append((file_path, n_bins_str, method_suffix))

        if not binning_files:
            raise ValueError(
                f"No binning files found in {folder_path}. "
                f"Expected files matching pattern: bins_new_{{n_bins}}{{method_name}}.npy"
            )

        logging.info(f"Found {len(binning_files)} binning files in {folder_path}")

        # Process each binning file
        for file_path, _, method_suffix in binning_files:
            try:
                logging.info(f"Loading binning file: {file_path}")
                loaded_data = np.load(file_path, allow_pickle=True).item()

                # Determine method name
                if method_suffix == "":
                    method_name = "quantile"
                else:
                    method_name = method_suffix[1:]  # Remove leading underscore

                # Extract field information from each loaded file
                for field_name, field_data in loaded_data.items():
                    if not isinstance(field_data, dict) or "bins" not in field_data:
                        logging.warning(
                            f"Skipping invalid field data for {field_name} in {file_path}"
                        )
                        continue

                    bins_array = field_data["bins"]
                    actual_n_bins = (
                        len(bins_array) - 1
                    )  # bin_edges has n_bins + 1 elements

                    # Store field binning information
                    field_bins[field_name] = bins_array
                    n_bins[field_name] = actual_n_bins
                    method[field_name] = method_name

                    logging.info(
                        f"  Field '{field_name}': {actual_n_bins} bins, method '{method_name}'"
                    )

            except Exception as e:
                logging.error(f"Failed to load binning file {file_path}: {e}")
                raise

        if not field_bins:
            raise ValueError(f"No valid field binning data found in {folder_path}")

        logging.info(
            f"Successfully loaded binning spec with fields: {list(field_bins.keys())}"
        )

        return cls(
            field_bins=field_bins,
            n_bins=n_bins,
            method=method,
            created_at=None,  # Could add timestamp if needed
            dataset_uuid=None,  # Could add if folder structure provides this
        )

    def __repr__(self) -> str:
        """String representation of the binning specification."""
        fields = list(self.field_bins.keys())
        return f"BinningSpec(fields={fields}, n_bins={self.n_bins})"


class BaseBinner(ABC):
    """
    Base class for computing bin edges from data (Phase 1).

    This is the "fit" phase that scans datasets to compute optimal bin edges.
    Subclasses implement different binning strategies (quantile, uniform, etc.).
    """

    FIELD_TO_ATTR = {
        "pos": "R",
        "force": "F",
        "target": "E",
        "spin": "spin",
        "charge": "charge",
        "cell": "cell",
        "stress": "stress",
    }

    def _collect_data(
        self, dataset_iter: Iterator[Molecule], fields: list[str]
    ) -> Dict[str, list]:
        """Collect data from the dataset iterator."""
        values = {field: [] for field in fields}
        for mol in tqdm(dataset_iter, desc="Collecting values for binning"):
            for field in fields:
                attr = self.FIELD_TO_ATTR.get(field)
                if not attr:
                    logging.warning(f"Unknown field '{field}', skipping.")
                    continue

                val = getattr(mol, attr, None)
                if val is not None:
                    values[field].append(np.array(val).flatten())
        return values

    @abstractmethod
    def fit(self, dataset_iter: Iterator[Molecule]) -> BinningSpec:
        """
        Fit binning specification to dataset.

        Args:
            dataset_iter: Iterator over Molecule objects

        Returns:
            Fitted BinningSpec ready for use
        """
        pass


class SklearnBinner(BaseBinner):
    """
    A wrapper for sklearn's KBinsDiscretizer.

    Computes bin edges using strategies like 'quantile', 'kmeans', or 'uniform'.
    """

    def __init__(
        self, n_bins: Dict[str, int], fields: list[str], strategy: str = "quantile"
    ):
        """
        Initialize the sklearn binner.

        Args:
            n_bins: Number of bins per field.
            fields: List of field names to bin.
            strategy: The binning strategy to use ('uniform', 'quantile', 'kmeans').
        """
        if strategy not in ["uniform", "quantile", "kmeans"]:
            raise ValueError(
                f"Strategy '{strategy}' is not supported. Choose from 'uniform', 'quantile', 'kmeans'."
            )
        self.n_bins = n_bins
        self.fields = fields
        self.strategy = strategy

    def fit(self, dataset_iter: Iterator[Molecule]) -> BinningSpec:
        """
        Fit binning to dataset using the specified strategy.

        Args:
            dataset_iter: Iterator over molecules

        Returns:
            BinningSpec with computed bin edges.
        """
        # 1. Collect values
        values = self._collect_data(dataset_iter, self.fields)

        # 2. Compute bins
        field_bins = {}
        method_dict = {}

        for field in self.fields:
            if not values[field]:
                logging.warning(
                    f"No values found for field '{field}', cannot compute bins."
                )
                continue

            all_values = np.concatenate(values[field])

            # Reshape for sklearn
            data = all_values.reshape(-1, 1)

            n_bins = self.n_bins[field]
            discretizer = SklearnKBinsDiscretizer(
                n_bins=n_bins,
                strategy=self.strategy,
                encode="ordinal",
            )
            discretizer.fit(data)

            field_bins[field] = discretizer.bin_edges_[0]
            method_dict[field] = self.strategy
            logging.info(f"Computed {n_bins} {self.strategy} bins for field '{field}'.")

        return BinningSpec(
            field_bins=field_bins, n_bins=self.n_bins, method=method_dict
        )
