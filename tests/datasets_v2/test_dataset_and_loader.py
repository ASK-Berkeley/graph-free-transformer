"""
Tests for TextDataset and data loaders.

This suite validates the integration of the data loading and formatting pipeline.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from typing import List, Dict
from omegaconf import DictConfig
from hydra.utils import instantiate

from mmlm.datasets_v2.loaders.omol_loader import OmolLoader
from mmlm.datasets_v2.dataset import TextDataset
from mmlm.datasets_v2.core.formatter import StandardFormatter
from mmlm.datasets_v2.core.molecule import Molecule
from mmlm.datasets_v2.core.binner import BinningSpec
from mmlm.datasets_v2.core.transforms import (
    RotationTransform,
    PermutationTransform,
    Transform,
)
from mmlm.datasets_v2.loaders.base_loader import BaseLoader
from mmlm.datasets_v2.builder import build_dataset
from mmlm.datasets_v2.core.formatter import AtomFormatter


@pytest.fixture
def mock_pasedb_dataset():
    """Create a mock PAseDBDataset for testing the loader."""
    mock_data = MagicMock()
    mock_data.atomic_numbers = torch.tensor([6, 1, 1], dtype=torch.long)
    mock_data.pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [-1.0, -1.0, 1.0]], dtype=torch.float32
    )
    mock_data.energy = torch.tensor(-100.0)
    mock_data.forces = torch.ones(3, 3, dtype=torch.float32) * 0.1
    mock_data.spin = torch.tensor(1)
    mock_data.charge = torch.tensor(0)
    mock_data.cell = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3)
    mock_data.stress = torch.ones(6, dtype=torch.float32) * 0.01

    mock_dataset = MagicMock()
    mock_dataset.__getitem__.return_value = mock_data
    mock_dataset.__len__.return_value = 1
    return mock_dataset


class TestOmolLoader:
    """Tests for the OmolLoader."""

    @patch("mmlm.datasets_v2.loaders.omol_loader.PAseDBDataset")
    def test_loader_initialization(self, mock_pasedb_class, mock_pasedb_dataset):
        """Test that the loader initializes correctly and can be instantiated."""
        mock_pasedb_class.return_value = mock_pasedb_dataset
        loader = OmolLoader(path="/fake/path", center=True)
        assert len(loader) == 1
        assert loader.center is True

    @patch("mmlm.datasets_v2.loaders.omol_loader.PAseDBDataset")
    def test_loader_getitem(self, mock_pasedb_class, mock_pasedb_dataset):
        """Test that __getitem__ returns a valid Molecule object with correct data."""
        mock_pasedb_class.return_value = mock_pasedb_dataset
        loader = OmolLoader(path="/fake/path", center=True)
        molecule = loader[0]

        assert isinstance(molecule, Molecule)
        assert molecule.n_atoms == 3
        assert molecule.has_forces
        assert molecule.has_energy

        # Check that centering was applied
        np.testing.assert_almost_equal(molecule.R.mean(axis=0), np.zeros(3))

        # Check that reference energy was subtracted
        # Z = [6, 1, 1] -> C, H, H
        # The elemental references are 1-indexed by atomic number, so we use Z-1
        ref_C = OmolLoader.OMOL_ELEMENTAL_REFERENCES[6]
        ref_H = OmolLoader.OMOL_ELEMENTAL_REFERENCES[1]
        expected_energy = -100.0 - (ref_C + 2 * ref_H)
        assert pytest.approx(molecule.E, 1e-4) == expected_energy

    @patch("mmlm.datasets_v2.loaders.omol_loader.PAseDBDataset")
    @patch("mmlm.datasets_v2.loaders.omol_loader.np.load")
    def test_loader_with_force_prior(
        self, mock_np_load, mock_pasedb_class, mock_pasedb_dataset
    ):
        """Test that the force prior is correctly subtracted."""
        mock_pasedb_class.return_value = mock_pasedb_dataset

        # Mock the force prior data
        force_prior_data = {"forces": np.ones((1, 3, 3), dtype=np.float32) * 0.05}
        mock_np_load.return_value.item.return_value = force_prior_data

        loader = OmolLoader(path="/fake/path", force_prior_path="/fake/prior.npy")
        molecule = loader[0]

        # Original forces are 0.1, prior is 0.05. Expected is 0.05.
        expected_forces = np.ones((3, 3), dtype=np.float32) * 0.05
        np.testing.assert_almost_equal(molecule.F, expected_forces)


class TestTextDataset:
    """Tests for the main TextDataset class."""

    @patch("mmlm.datasets_v2.loaders.omol_loader.PAseDBDataset")
    def test_dataset_integration(self, mock_pasedb_class, mock_pasedb_dataset):
        """Test the full pipeline: Loader -> TextDataset -> Formatter."""
        mock_pasedb_class.return_value = mock_pasedb_dataset

        # 1. Initialize components
        loader = OmolLoader(path="/fake/path")
        formatter = StandardFormatter()
        bin_spec = BinningSpec(
            field_bins={
                "pos": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
                "force": np.array([-20.0, -10.0, 0.0, 10.0, 20.0]),
                "target": np.array([-100.0, -50.0, 0.0, 50.0, 100.0]),
            },
            n_bins={"pos": 4, "force": 4, "target": 4},
            method={
                "pos": "quantile",
                "force": "quantile",
                "target": "uniform",
            },
        )

        # 2. Initialize TextDataset
        text_dataset = TextDataset(
            loader=loader, formatter=formatter, bin_spec=bin_spec
        )

        assert len(text_dataset) == 1

        # 3. Get an item
        text, cont_tensor = text_dataset[0]

        # 4. Validate output
        assert isinstance(text, str)
        assert text.startswith("<BOS>")
        assert isinstance(cont_tensor, torch.Tensor)

        # Check for expected content from the formatter
        assert text.startswith("<BOS>")
        assert "[POS]" in text
        assert "a_6:" in text


@pytest.fixture
def mock_loader():
    """Mock loader for TextDataset tests."""
    return MagicMock(spec=BaseLoader)


@pytest.fixture
def mock_formatter():
    """Mock formatter for TextDataset tests."""
    # Mock an instance of the class, not the class itself
    return MagicMock(spec=StandardFormatter())


@pytest.fixture
def mock_binning_spec():
    """Mock binning spec for testing."""
    return MagicMock(spec=BinningSpec)


def test_text_dataset_initialization(mock_loader, mock_formatter, mock_binning_spec):
    """Test TextDataset initialization."""
    dataset = TextDataset(
        loader=mock_loader, formatter=mock_formatter, bin_spec=mock_binning_spec
    )
    assert dataset.loader is mock_loader
    assert dataset.formatter is mock_formatter
    assert dataset.bin_spec is mock_binning_spec
    assert dataset.transforms == []


def test_text_dataset_len(mock_loader, mock_formatter, mock_binning_spec):
    """Test that TextDataset length is correct."""
    mock_loader.__len__.return_value = 100
    dataset = TextDataset(
        loader=mock_loader, formatter=mock_formatter, bin_spec=mock_binning_spec
    )
    assert len(dataset) == 100


def test_text_dataset_getitem(mock_loader, mock_formatter, mock_binning_spec):
    """Test that __getitem__ retrieves and formats a molecule."""
    # Setup mock loader and formatter
    mock_molecule = MagicMock(spec=Molecule)
    mock_loader.__getitem__.return_value = mock_molecule
    mock_formatter.return_value = {"text": "formatted_text"}

    dataset = TextDataset(
        loader=mock_loader, formatter=mock_formatter, bin_spec=mock_binning_spec
    )

    # Get item
    result = dataset[42]

    # Assertions
    mock_loader.__getitem__.assert_called_once_with(42)
    mock_formatter.assert_called_once_with(mock_molecule, mock_binning_spec)
    assert result == {"text": "formatted_text"}


def test_text_dataset_with_transforms(mock_loader, mock_formatter, mock_binning_spec):
    """Test that transforms are applied correctly in __getitem__."""
    mock_molecule = MagicMock(spec=Molecule)
    mock_loader.__getitem__.return_value = mock_molecule

    # Create mock transforms
    mock_transform_1 = MagicMock(spec=Transform)
    mock_transform_2 = MagicMock(spec=Transform)

    # The first transform returns another mock molecule
    transformed_molecule_1 = MagicMock(spec=Molecule)
    mock_transform_1.return_value = transformed_molecule_1

    # The second transform returns yet another mock molecole
    transformed_molecule_2 = MagicMock(spec=Molecule)
    mock_transform_2.return_value = transformed_molecule_2

    transforms = [mock_transform_1, mock_transform_2]
    dataset = TextDataset(
        loader=mock_loader,
        formatter=mock_formatter,
        bin_spec=mock_binning_spec,
        transforms=transforms,
    )

    # Get item
    dataset[0]

    # Assertions
    mock_loader.__getitem__.assert_called_once_with(0)
    mock_transform_1.assert_called_once_with(mock_molecule)
    mock_transform_2.assert_called_once_with(transformed_molecule_1)
    mock_formatter.assert_called_once_with(transformed_molecule_2, mock_binning_spec)


class MockLoader(BaseLoader):
    """A mock loader for integration tests that instantiates from dicts."""

    def __init__(self, molecules: List[Dict]):
        self.molecules = [Molecule.from_dict(m) for m in molecules]

    def __len__(self):
        return len(self.molecules)

    def __getitem__(self, idx):
        return self.molecules[idx].clone()


def test_dataset_integration_with_real_components(
    simple_molecule, comprehensive_binning_spec
):
    """An integration test using real, simple components."""
    # Setup
    loader = MockLoader([simple_molecule.to_dict()])
    formatter = StandardFormatter()
    transforms = [RotationTransform(), PermutationTransform()]

    dataset = TextDataset(
        loader=loader,
        formatter=formatter,
        bin_spec=comprehensive_binning_spec,
        transforms=transforms,
    )

    # Get item
    text, cont_tensor = dataset[0]

    # Assertions
    assert text.startswith("<BOS>")
    assert text.endswith("<EOS>")
    assert "[POS]" in text
    assert "[TARGET]" in text

    # Check that the output is not empty
    assert cont_tensor.shape[0] > 0


def test_build_dataset_from_config(simple_molecule, comprehensive_binning_spec):
    """Test that the build_dataset function works correctly with a Hydra config."""
    # Create a mock bin spec file to be loaded by the builder
    with patch("mmlm.datasets_v2.core.binner.BinningSpec.load") as mock_load:
        mock_load.return_value = comprehensive_binning_spec

        # 1. Create a Hydra-like config object (DictConfig) that matches what
        #    would be used in a real training script.
        cfg = DictConfig(
            {
                "_target_": "mmlm.datasets_v2.builder.build_dataset",
                "cfg": {
                    "_target_": "mmlm.datasets_v2.config.DatasetV2Cfg",
                    "loader": {
                        "_target_": "tests.datasets_v2.test_dataset_and_loader.MockLoader",
                        "molecules": [simple_molecule.to_dict()],
                    },
                    "formatter": {
                        "_target_": "mmlm.datasets_v2.core.formatter.AtomFormatter",
                        "first_force_only": True,
                    },
                    "bin_spec_path": "/fake/path/to/bins.npz",
                    "transforms": [
                        {
                            "_target_": "mmlm.datasets_v2.core.transforms.PermutationTransform",
                        }
                    ],
                    "icl": None,
                },
            }
        )

        # 2. Build the dataset using hydra's instantiate function, which correctly
        #    resolves the _target_ fields.
        dataset = instantiate(cfg)

        # 3. Validate the output
        assert isinstance(dataset, TextDataset)
        assert len(dataset.transforms) == 1
        assert isinstance(dataset.transforms[0], PermutationTransform)
        assert isinstance(dataset.formatter, AtomFormatter)
        assert dataset.formatter.first_force_only is True

        # 4. Check that we can get an item
        text, cont_tensor, atom_ids = dataset[0]
        assert isinstance(text, str)
        assert cont_tensor.shape[0] > 0
        assert atom_ids.shape[0] > 0
