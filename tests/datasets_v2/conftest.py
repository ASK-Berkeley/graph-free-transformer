"""
Pytest configuration for datasets_v2 tests.

Provides fixtures and setup for testing the new modular dataset implementation.
"""

import pytest
import torch
import numpy as np
from mmlm.datasets_v2.core.molecule import Molecule
from mmlm.datasets_v2.core.binner import BinningSpec


@pytest.fixture(autouse=True)
def setup_random_state():
    """Setup deterministic behavior for all tests"""
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    yield


@pytest.fixture
def simple_molecule():
    """Create a simple test molecule (methane CH4)"""
    return Molecule(
        Z=np.array([6, 1, 1, 1, 1], dtype=np.int8),  # C, H, H, H, H
        R=np.array(
            [
                [0.0, 0.0, 0.0],  # C at origin
                [1.0, 1.0, 1.0],  # H
                [-1.0, -1.0, 1.0],  # H
                [-1.0, 1.0, -1.0],  # H
                [1.0, -1.0, -1.0],  # H
            ],
            dtype=np.float32,
        ),
        E=-40.5,
    )


@pytest.fixture
def molecule_with_forces():
    """Create a test molecule with forces"""
    return Molecule(
        Z=np.array([1, 1], dtype=np.int8),  # H2
        R=np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], dtype=np.float32),
        F=np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]], dtype=np.float32),
        E=-1.17,
    )


@pytest.fixture
def periodic_molecule():
    """Create a test molecule with periodic boundary conditions"""
    return Molecule(
        Z=np.array([11, 17], dtype=np.int8),  # NaCl
        R=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=np.float32),
        E=-8.2,
        cell=np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
        ),
        stress=np.array(
            [0.1, 0.1, 0.1, 0.0, 0.0, 0.0], dtype=np.float32
        ),  # Voigt notation
    )


@pytest.fixture
def molecule_dict():
    """Dictionary representation of a molecule for testing from_dict"""
    return {
        "Z": [6, 8],  # CO
        "R": [[0.0, 0.0, 0.0], [1.13, 0.0, 0.0]],
        "E": -113.3,
        "F": [[0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]],
    }


@pytest.fixture
def simple_binning_spec():
    """Create a simple BinningSpec for basic testing"""
    return BinningSpec(
        field_bins={
            "pos": np.array([-1.0, 0.0, 1.0]),  # 2 bins
            "force": np.array([-2.0, 0.0, 2.0]),  # 2 bins
        },
        n_bins={"pos": 2, "force": 2},
        method={"pos": "quantile", "force": "quantile"},
    )


@pytest.fixture
def comprehensive_binning_spec():
    """Create a comprehensive BinningSpec for formatter testing."""
    return BinningSpec(
        field_bins={
            "pos": np.array([-1.0, 0.5, 1.0, 1.5]),
            "force": np.array([-5.0, 0.0, 5.0]),
            "target": np.array([-100.0, 0.0, 100.0]),
            "spin": np.array([1.0, 2.0, 3.0]),
            "charge": np.array([-1.0, 0.0, 1.0]),
        },
        n_bins={
            "pos": 3,
            "force": 2,
            "target": 2,
            "spin": 2,
            "charge": 2,
        },
        method={
            "pos": "quantile",
            "force": "quantile",
            "target": "uniform",
            "spin": "uniform",
            "charge": "uniform",
        },
    )


@pytest.fixture
def molecule_list():
    """Create a list of molecules for binner testing."""
    molecules = []
    for i in range(10):
        # Create deterministic, unique coordinates for each molecule
        r_val = float(i + 1)
        molecules.append(
            Molecule(
                Z=np.array([i % 5 + 1]),  # Different atoms
                R=np.array([[r_val, r_val, r_val]], dtype=np.float32),
                E=float(-i - 1),
            )
        )
    return molecules
