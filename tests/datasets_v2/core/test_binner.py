"""
Tests for the BinningSpec and binner classes.

These tests validate the binning functionality that discretizes continuous values
into tokens for molecular machine learning models.
"""

import pytest
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

from pathlib import Path
from unittest.mock import patch

from mmlm.datasets_v2.core.binner import BinningSpec, BaseBinner, SklearnBinner


# Test fixtures for creating mock binning data
@pytest.fixture
def mock_binning_data():
    """Create mock binning data in the expected format"""
    return {
        "pos": {
            "bins": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),  # 4 bins
            "mae": 0.12,
            "counts": np.array([100, 100, 100, 100]),
        },
        "force": {
            "bins": np.array([-20.0, -10.0, 0.0, 10.0, 20.0]),  # 4 bins
            "mae": 0.08,
            "counts": np.array([50, 50, 50, 50]),
        },
    }


@pytest.fixture
def mock_target_binning_data():
    """Create mock target binning data"""
    return {
        "target": {
            "bins": np.array([-100.0, -50.0, 0.0, 50.0, 100.0]),  # 4 bins
            "mae": 0.5,
            "counts": np.array([25, 25, 25, 25]),
        }
    }


@pytest.fixture
def temp_binning_folder(tmp_path, mock_binning_data, mock_target_binning_data):
    """Create temporary folder with mock binning files"""
    folder = tmp_path / "binning_test"
    folder.mkdir()

    # Create bins_new_4.npy (quantile method, pos and force)
    np.save(folder / "bins_new_4.npy", mock_binning_data)

    # Create bins_new_4_uniform.npy (uniform method, target)
    np.save(folder / "bins_new_4_uniform.npy", mock_target_binning_data)

    return folder


@pytest.fixture
def temp_single_file_folder(tmp_path, mock_binning_data):
    """Create temporary folder with single binning file"""
    folder = tmp_path / "single_binning"
    folder.mkdir()

    # Create single file with multiple fields
    np.save(folder / "bins_new_1000.npy", mock_binning_data)

    return folder


class TestBinningSpecBasic:
    """Test basic BinningSpec functionality"""

    def test_initialization(self, simple_binning_spec):
        """Test basic initialization"""
        spec = simple_binning_spec
        assert len(spec.field_bins) == 2
        assert spec.n_bins["pos"] == 2
        assert spec.n_bins["force"] == 2
        assert spec.method["pos"] == "quantile"
        assert spec.method["force"] == "quantile"
        assert spec.created_at is None
        assert spec.dataset_uuid is None

    def test_initialization_with_metadata(self):
        """Test initialization with metadata"""
        spec = BinningSpec(
            field_bins={"pos": np.array([-1.0, 0.0, 1.0])},
            n_bins={"pos": 2},
            method={"pos": "quantile"},
            created_at="2024-01-01T00:00:00Z",
            dataset_uuid="test-uuid",
        )
        assert spec.created_at == "2024-01-01T00:00:00Z"
        assert spec.dataset_uuid == "test-uuid"

    def test_string_representation(self, simple_binning_spec):
        """Test string representation"""
        spec_str = str(simple_binning_spec)
        assert "BinningSpec" in spec_str
        assert "pos" in spec_str
        assert "force" in spec_str
        assert "{'pos': 2, 'force': 2}" in spec_str


class TestBinningSpecTransform:
    """Test BinningSpec transform functionality"""

    def test_transform_single_values(self, simple_binning_spec):
        """Test transform with single values"""
        spec = simple_binning_spec

        # Test position transform
        pos_bins = spec.transform("pos", np.array([0.5]))
        assert pos_bins == 1  # Should be in second bin

        # Test force transform
        force_bins = spec.transform("force", np.array([-1.0]))
        assert force_bins == 0  # Should be in first bin

    def test_transform_multiple_values(self, simple_binning_spec):
        """Test transform with multiple values"""
        spec = simple_binning_spec

        values = np.array([-0.5, 0.0, 0.5])
        bins = spec.transform("pos", values)
        np.testing.assert_array_equal(bins, [[0], [1], [1]])

    def test_transform_2d_values(self, simple_binning_spec):
        """Test transform with 2D values (like positions)"""
        spec = simple_binning_spec

        # KBinsDiscretizer expects 1D arrays - flatten 2D positions first
        positions = np.array([[-0.5, 0.0, 0.5], [0.2, -0.3, 0.8]])
        positions_flat = positions.flatten()
        bins = spec.transform("pos", positions_flat)
        expected = np.array([[0], [1], [1], [1], [0], [1]])  # Flattened expected values
        np.testing.assert_array_equal(bins, expected)

    def test_transform_invalid_field(self, simple_binning_spec):
        """Test transform with invalid field name"""
        spec = simple_binning_spec

        with pytest.raises(KeyError, match="Field 'invalid' not found"):
            spec.transform("invalid", np.array([0.0]))

    def test_inverse_transform_single_values(self, simple_binning_spec):
        """Test inverse transform with single values"""
        spec = simple_binning_spec

        # Test position inverse transform
        continuous = spec.inverse_transform("pos", np.array([0]))
        assert continuous == -0.5  # Bin center of first bin

        continuous = spec.inverse_transform("pos", np.array([1]))
        assert continuous == 0.5  # Bin center of second bin

    def test_inverse_transform_multiple_values(self, simple_binning_spec):
        """Test inverse transform with multiple values"""
        spec = simple_binning_spec

        bins = np.array([0, 1, 1])
        continuous = spec.inverse_transform("pos", bins)
        expected = np.array([-0.5, 0.5, 0.5])
        np.testing.assert_array_equal(continuous, expected)

    def test_inverse_transform_invalid_field(self, simple_binning_spec):
        """Test inverse transform with invalid field name"""
        spec = simple_binning_spec

        with pytest.raises(KeyError, match="Field 'invalid' not found"):
            spec.inverse_transform("invalid", np.array([0]))


class TestBinningSpecSaveLoad:
    """Test BinningSpec save and load functionality"""

    def test_save_load_roundtrip(self, simple_binning_spec, tmp_path):
        """Test save and load round-trip"""
        spec = simple_binning_spec
        save_path = tmp_path / "test_spec.npz"

        # Save
        spec.save(save_path)
        assert save_path.exists()

        # Load
        loaded_spec = BinningSpec.load(save_path)

        # Verify equality
        assert loaded_spec.field_bins.keys() == spec.field_bins.keys()
        assert loaded_spec.n_bins == spec.n_bins
        assert loaded_spec.method == spec.method

        for field in spec.field_bins:
            np.testing.assert_array_equal(
                loaded_spec.field_bins[field], spec.field_bins[field]
            )

    def test_save_load_with_metadata(self, tmp_path):
        """Test save and load with metadata"""
        spec = BinningSpec(
            field_bins={"pos": np.array([-1.0, 0.0, 1.0])},
            n_bins={"pos": 2},
            method={"pos": "quantile"},
            created_at="2024-01-01T00:00:00Z",
            dataset_uuid="test-uuid",
        )

        save_path = tmp_path / "test_spec_meta.npz"
        spec.save(save_path)

        loaded_spec = BinningSpec.load(save_path)
        assert loaded_spec.created_at == "2024-01-01T00:00:00Z"
        assert loaded_spec.dataset_uuid == "test-uuid"

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading from non-existent file"""
        nonexistent_path = tmp_path / "does_not_exist.npz"

        with pytest.raises(FileNotFoundError):
            BinningSpec.load(nonexistent_path)

    @patch("mmlm.datasets_v2.core.binner.logging")
    def test_save_load_logging(self, mock_logging, simple_binning_spec, tmp_path):
        """Test that save and load operations log appropriately"""
        spec = simple_binning_spec
        save_path = tmp_path / "test_logging.npz"

        # Test save logging
        spec.save(save_path)
        mock_logging.info.assert_called_with(f"Saved BinningSpec to {save_path}")

        # Test load logging
        BinningSpec.load(save_path)
        mock_logging.info.assert_called_with(f"Loaded BinningSpec from {save_path}")


class TestBinningSpecFromCurrentFormat:
    """Test BinningSpec loading from current format"""

    def test_from_current_format_basic(self, temp_binning_folder):
        """Test loading from current format with multiple files"""
        spec = BinningSpec.from_current_format(temp_binning_folder)

        # Should have all fields from both files
        assert set(spec.field_bins.keys()) == {"pos", "force", "target"}
        assert spec.n_bins["pos"] == 4
        assert spec.n_bins["force"] == 4
        assert spec.n_bins["target"] == 4
        assert spec.method["pos"] == "quantile"
        assert spec.method["force"] == "quantile"
        assert spec.method["target"] == "uniform"

    def test_from_current_format_single_file(self, temp_single_file_folder):
        """Test loading from current format with single file"""
        spec = BinningSpec.from_current_format(temp_single_file_folder)

        assert set(spec.field_bins.keys()) == {"pos", "force"}
        assert spec.n_bins["pos"] == 4
        assert spec.n_bins["force"] == 4
        assert spec.method["pos"] == "quantile"
        assert spec.method["force"] == "quantile"

    def test_from_current_format_nonexistent_folder(self, tmp_path):
        """Test loading from non-existent folder"""
        nonexistent_folder = tmp_path / "does_not_exist"

        with pytest.raises(ValueError, match="Folder path does not exist"):
            BinningSpec.from_current_format(nonexistent_folder)

    def test_from_current_format_no_binning_files(self, tmp_path):
        """Test loading from folder with no binning files"""
        empty_folder = tmp_path / "empty"
        empty_folder.mkdir()

        with pytest.raises(ValueError, match="No binning files found"):
            BinningSpec.from_current_format(empty_folder)

    def test_from_current_format_invalid_binning_data(self, tmp_path):
        """Test loading with invalid binning data"""
        folder = tmp_path / "invalid_data"
        folder.mkdir()

        # Create file with invalid data structure
        invalid_data = {"pos": {"invalid": "data"}}  # Missing "bins" key
        np.save(folder / "bins_new_10.npy", invalid_data)

        with pytest.raises(ValueError, match="No valid field binning data found"):
            BinningSpec.from_current_format(folder)

    def test_from_current_format_partial_invalid_data(self, tmp_path):
        """Test loading with partially invalid data (some fields valid, some invalid)"""
        folder = tmp_path / "partial_invalid"
        folder.mkdir()

        # Create file with mixed valid and invalid data
        mixed_data = {
            "pos": {"bins": np.array([-1.0, 0.0, 1.0])},  # Valid
            "force": {"invalid": "data"},  # Invalid - missing "bins"
            "target": {"bins": np.array([-2.0, 0.0, 2.0])},  # Valid
        }
        np.save(folder / "bins_new_2.npy", mixed_data)

        # Should still work with valid fields
        spec = BinningSpec.from_current_format(folder)
        assert set(spec.field_bins.keys()) == {"pos", "target"}
        assert spec.n_bins["pos"] == 2
        assert spec.n_bins["target"] == 2

    def test_from_current_format_different_methods(self, tmp_path):
        """Test loading files with different binning methods"""
        folder = tmp_path / "methods_test"
        folder.mkdir()

        # Create files with different method suffixes
        quantile_data = {"pos": {"bins": np.array([-1.0, 0.0, 1.0])}}
        uniform_data = {"force": {"bins": np.array([-2.0, 0.0, 2.0])}}
        custom_data = {"target": {"bins": np.array([-3.0, 0.0, 3.0])}}

        np.save(folder / "bins_new_2.npy", quantile_data)  # Quantile (no suffix)
        np.save(folder / "bins_new_2_uniform.npy", uniform_data)  # Uniform
        np.save(folder / "bins_new_2_custom.npy", custom_data)  # Custom method

        spec = BinningSpec.from_current_format(folder)
        assert spec.method["pos"] == "quantile"
        assert spec.method["force"] == "uniform"
        assert spec.method["target"] == "custom"

    @patch("mmlm.datasets_v2.core.binner.logging")
    def test_from_current_format_logging(self, mock_logging, temp_binning_folder):
        """Test that loading from current format logs appropriately"""
        BinningSpec.from_current_format(temp_binning_folder)

        # Should log file discovery and loading
        mock_logging.info.assert_any_call(
            "Found 2 binning files in " + str(temp_binning_folder)
        )
        # mock_logging.info.assert_any_call(
        # "Successfully loaded binning spec with fields: ['pos', 'force', 'target']"
        # )

    def test_from_current_format_bin_edges_integrity(self, temp_binning_folder):
        """Test that bin edges are correctly preserved"""
        spec = BinningSpec.from_current_format(temp_binning_folder)

        # Check that bin edges match the original mock data
        expected_pos_bins = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        expected_force_bins = np.array([-20.0, -10.0, 0.0, 10.0, 20.0])

        np.testing.assert_array_equal(spec.field_bins["pos"], expected_pos_bins)
        np.testing.assert_array_equal(spec.field_bins["force"], expected_force_bins)


class TestBinningSpecEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_field_bins(self):
        """Test with empty field_bins"""
        spec = BinningSpec(field_bins={}, n_bins={}, method={})
        assert len(spec.field_bins) == 0
        assert len(spec.n_bins) == 0
        assert len(spec.method) == 0

    def test_single_bin_edge_case(self):
        """Test with single bin (2 edges)"""
        spec = BinningSpec(
            field_bins={"pos": np.array([-1.0, 1.0])},
            n_bins={"pos": 1},
            method={"pos": "quantile"},
        )

        # Should work with single bin
        bins = spec.transform("pos", np.array([0.0]))
        assert bins == 0

        continuous = spec.inverse_transform("pos", np.array([0]))
        assert continuous == 0.0  # Center of single bin

    def test_extreme_values(self):
        """Test with extreme values"""
        spec = BinningSpec(
            field_bins={"pos": np.array([-1000.0, 0.0, 1000.0])},
            n_bins={"pos": 2},
            method={"pos": "quantile"},
        )

        # Test values outside bin range
        extreme_values = np.array([-9999.0, 9999.0])
        bins = spec.transform("pos", extreme_values)
        # Should be clamped to valid bin indices
        assert all(0 <= b <= 1 for b in bins.flatten())


class TestBinningSpecRoundTrip:
    """Test round-trip conversions and consistency"""

    def test_transform_inverse_roundtrip(self, simple_binning_spec):
        """Test that transform -> inverse_transform preserves bin centers"""
        spec = simple_binning_spec

        # Test with bin centers
        bin_centers = np.array([-0.5, 0.5])  # Centers of the two bins
        bins = spec.transform("pos", bin_centers)
        recovered = spec.inverse_transform("pos", bins)

        # Handle shape - inverse_transform may return 2D, flatten for comparison
        recovered_flat = recovered.flatten()
        np.testing.assert_array_almost_equal(recovered_flat, bin_centers)

    def test_save_load_transform_consistency(self, simple_binning_spec, tmp_path):
        """Test that save/load doesn't affect transform results"""
        spec = simple_binning_spec

        # Transform with original spec
        test_values = np.array([-0.3, 0.2, 0.8])
        original_bins = spec.transform("pos", test_values)

        # Save and load
        save_path = tmp_path / "consistency_test.npz"
        spec.save(save_path)
        loaded_spec = BinningSpec.load(save_path)

        # Transform with loaded spec
        loaded_bins = loaded_spec.transform("pos", test_values)

        # Should be identical
        np.testing.assert_array_equal(original_bins, loaded_bins)

    def test_from_current_format_save_load_roundtrip(
        self, temp_binning_folder, tmp_path
    ):
        """Test round-trip: current format -> BinningSpec -> save -> load"""
        # Load from current format
        spec = BinningSpec.from_current_format(temp_binning_folder)

        # Save in new format
        save_path = tmp_path / "roundtrip_test.npz"
        spec.save(save_path)

        # Load from new format
        loaded_spec = BinningSpec.load(save_path)

        # Should be equivalent
        assert loaded_spec.field_bins.keys() == spec.field_bins.keys()
        assert loaded_spec.n_bins == spec.n_bins
        assert loaded_spec.method == spec.method

        for field in spec.field_bins:
            np.testing.assert_array_equal(
                loaded_spec.field_bins[field], spec.field_bins[field]
            )


class TestSklearnBinner:
    """Tests for the SklearnBinner class."""

    @pytest.mark.parametrize("strategy", ["quantile", "kmeans", "uniform"])
    def test_fit_with_strategies(self, molecule_list, strategy):
        """Test the fit method of SklearnBinner with various strategies."""
        n_bins = {"pos": 10, "target": 5}
        fields = ["pos", "target"]
        binner = SklearnBinner(n_bins=n_bins, fields=fields, strategy=strategy)

        bin_spec = binner.fit(iter(molecule_list))

        assert "pos" in bin_spec.field_bins
        assert "target" in bin_spec.field_bins
        assert len(bin_spec.field_bins["pos"]) == n_bins["pos"] + 1
        assert len(bin_spec.field_bins["target"]) == n_bins["target"] + 1
        assert bin_spec.method["pos"] == strategy
        assert bin_spec.method["target"] == strategy

    def test_init_invalid_strategy(self):
        """Test that SklearnBinner raises an error for an invalid strategy."""
        with pytest.raises(
            ValueError, match="Strategy 'invalid_strategy' is not supported."
        ):
            SklearnBinner(n_bins={}, fields=[], strategy="invalid_strategy")

    def test_fit_with_missing_field_in_molecule(
        self, simple_molecule, molecule_with_forces
    ):
        """Test fit when a field is missing from some molecules."""
        mixed_list = [simple_molecule, molecule_with_forces]

        n_bins = {"pos": 10, "force": 5, "target": 5}
        fields = ["pos", "force", "target"]
        binner = SklearnBinner(n_bins=n_bins, fields=fields, strategy="quantile")

        with pytest.warns(UserWarning, match="Bins whose width are too small"):
            bin_spec = binner.fit(iter(mixed_list))

        assert "pos" in bin_spec.field_bins
        assert "force" in bin_spec.field_bins
        assert "target" in bin_spec.field_bins
        assert len(bin_spec.field_bins["pos"]) <= n_bins["pos"] + 1
        assert len(bin_spec.field_bins["force"]) <= n_bins["force"] + 1
        assert len(bin_spec.field_bins["target"]) <= n_bins["target"] + 1

    @patch("mmlm.datasets_v2.core.binner.logging")
    def test_fit_with_no_values_for_field(self, mock_logging, simple_molecule):
        """Test fit when no values are found for a requested field."""
        n_bins = {"force": 5}
        fields = ["force"]
        # simple_molecule does not have forces
        binner = SklearnBinner(n_bins=n_bins, fields=fields, strategy="quantile")

        bin_spec = binner.fit(iter([simple_molecule]))

        assert "force" not in bin_spec.field_bins
        mock_logging.warning.assert_called_with(
            "No values found for field 'force', cannot compute bins."
        )
