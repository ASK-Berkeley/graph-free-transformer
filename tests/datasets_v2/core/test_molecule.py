"""
Tests for the Molecule dataclass.

These tests validate the core Molecule dataclass that serves as the foundation
for all downstream operations in the datasets v2 architecture.
"""

import pytest
import numpy as np
import torch
from mmlm.datasets_v2.core.molecule import Molecule


class TestMoleculeBasic:
    """Test basic Molecule functionality"""
    
    def test_initialization_minimal(self):
        """Test minimal initialization with only Z and R"""
        mol = Molecule(
            Z=np.array([1, 1], dtype=np.int8),
            R=np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], dtype=np.float32)
        )
        assert mol.n_atoms == 2
        assert not mol.has_forces
        assert not mol.has_energy
        assert not mol.has_cell
        assert not mol.has_stress

    def test_initialization_complete(self, periodic_molecule):
        """Test initialization with all fields"""
        mol = periodic_molecule
        assert mol.n_atoms == 2
        assert mol.has_energy
        assert mol.has_cell
        assert mol.has_stress
        np.testing.assert_almost_equal(mol.E, -8.2)

    def test_properties(self, simple_molecule, molecule_with_forces):
        """Test molecular properties"""
        # Simple molecule without forces
        assert simple_molecule.n_atoms == 5
        assert simple_molecule.has_energy
        assert not simple_molecule.has_forces
        
        # Molecule with forces
        assert molecule_with_forces.has_forces
        assert molecule_with_forces.has_energy

    def test_string_representation(self, simple_molecule, molecule_with_forces):
        """Test string representation"""
        simple_repr = str(simple_molecule)
        assert "n_atoms=5" in simple_repr
        assert "E=-40.500" in simple_repr
        assert "forces=True" not in simple_repr
        
        forces_repr = str(molecule_with_forces)
        assert "forces=True" in forces_repr


class TestMoleculeValidation:
    """Test Molecule validation logic"""
    
    def test_invalid_z_shape(self):
        """Test validation of Z array shape"""
        with pytest.raises(ValueError, match="Z must be 1D array"):
            Molecule(
                Z=np.array([[1, 1]], dtype=np.int8),  # 2D instead of 1D
                R=np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], dtype=np.float32)
            )
    
    def test_invalid_r_shape(self):
        """Test validation of R array shape"""
        with pytest.raises(ValueError, match="R must be \\[n_atoms, 3\\] array"):
            Molecule(
                Z=np.array([1, 1], dtype=np.int8),
                R=np.array([[0.0, 0.0], [0.74, 0.0]], dtype=np.float32)  # Missing z coordinate
            )
    
    def test_mismatched_z_r_lengths(self):
        """Test validation of Z and R length mismatch"""
        with pytest.raises(ValueError, match="Z and R must have same length"):
            Molecule(
                Z=np.array([1, 1, 1], dtype=np.int8),  # 3 atoms
                R=np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], dtype=np.float32)  # 2 positions
            )
    
    def test_invalid_forces_shape(self):
        """Test validation of forces array shape"""
        with pytest.raises(ValueError, match="F must be \\[n_atoms, 3\\] array"):
            Molecule(
                Z=np.array([1, 1], dtype=np.int8),
                R=np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], dtype=np.float32),
                F=np.array([[0.1, 0.0], [-0.1, 0.0]], dtype=np.float32)  # Missing z component
            )
    
    def test_mismatched_forces_length(self):
        """Test validation of forces length mismatch"""
        with pytest.raises(ValueError, match="F and Z must have same length"):
            Molecule(
                Z=np.array([1, 1], dtype=np.int8),
                R=np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], dtype=np.float32),
                F=np.array([[0.1, 0.0, 0.0]], dtype=np.float32)  # Only 1 force for 2 atoms
            )
    
    def test_invalid_cell_shape(self):
        """Test validation of cell array shape"""
        with pytest.raises(ValueError, match="cell must be \\[3, 3\\] array"):
            Molecule(
                Z=np.array([1], dtype=np.int8),
                R=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                cell=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)  # 2x2 instead of 3x3
            )
    
    def test_invalid_stress_shape(self):
        """Test validation of stress array shape"""
        with pytest.raises(ValueError, match="stress must be \\[3, 3\\] or \\[6\\] array"):
            Molecule(
                Z=np.array([1], dtype=np.int8),
                R=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                stress=np.array([0.1, 0.2, 0.3], dtype=np.float32)  # Invalid shape
            )


class TestMoleculeFromDict:
    """Test Molecule creation from dictionary"""
    
    def test_from_dict_minimal(self):
        """Test from_dict with minimal data"""
        data = {
            "Z": [1, 1],
            "R": [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]
        }
        mol = Molecule.from_dict(data)
        assert mol.n_atoms == 2
        assert mol.Z.dtype == np.int8
        assert mol.R.dtype == np.float32
        assert not mol.has_forces
        assert not mol.has_energy
    
    def test_from_dict_complete(self, molecule_dict):
        """Test from_dict with all data"""
        mol = Molecule.from_dict(molecule_dict)
        assert mol.n_atoms == 2
        assert mol.has_forces
        assert mol.has_energy
        assert mol.E == -113.3
        np.testing.assert_array_equal(mol.Z, [6, 8])
    
    def test_from_dict_with_none_values(self):
        """Test from_dict with explicit None values"""
        data = {
            "Z": [1, 1],
            "R": [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],
            "F": None,
            "E": None
        }
        mol = Molecule.from_dict(data)
        assert not mol.has_forces
        assert not mol.has_energy


class TestMoleculeToDict:
    """Test Molecule conversion to dictionary"""
    
    def test_to_dict_minimal(self):
        """Test to_dict with minimal molecule"""
        mol = Molecule(
            Z=np.array([1, 1], dtype=np.int8),
            R=np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], dtype=np.float32)
        )
        data = mol.to_dict()
        assert set(data.keys()) == {"Z", "R"}
        np.testing.assert_array_equal(data["Z"], mol.Z)
        np.testing.assert_array_equal(data["R"], mol.R)
    
    def test_to_dict_complete(self, periodic_molecule):
        """Test to_dict with complete molecule"""
        data = periodic_molecule.to_dict()
        expected_keys = {"Z", "R", "E", "cell", "stress"}
        assert set(data.keys()) == expected_keys
        assert data["E"] == periodic_molecule.E


class TestMoleculeOperations:
    """Test Molecule operations like cloning and conversion"""
    
    def test_clone(self, simple_molecule):
        """Test molecule cloning"""
        cloned = simple_molecule.clone()
        
        # Should be equal but not the same object
        assert cloned is not simple_molecule
        assert cloned.n_atoms == simple_molecule.n_atoms
        assert cloned.E == simple_molecule.E
        np.testing.assert_array_equal(cloned.Z, simple_molecule.Z)
        np.testing.assert_array_equal(cloned.R, simple_molecule.R)
        
        # Arrays should be independent copies
        assert cloned.Z is not simple_molecule.Z
        assert cloned.R is not simple_molecule.R
    
    def test_clone_with_forces(self, molecule_with_forces):
        """Test cloning molecule with forces"""
        cloned = molecule_with_forces.clone()
        assert cloned.has_forces
        np.testing.assert_array_equal(cloned.F, molecule_with_forces.F)
        assert cloned.F is not molecule_with_forces.F
    
    def test_to_torch(self, simple_molecule):
        """Test conversion to torch tensors"""
        torch_mol = simple_molecule.to_torch()
        
        # Should have torch tensors instead of numpy arrays
        assert isinstance(torch_mol.Z, torch.Tensor)
        assert isinstance(torch_mol.R, torch.Tensor)
        assert torch_mol.E == simple_molecule.E  # Scalar should remain the same
        
        # Values should be preserved
        torch.testing.assert_close(torch_mol.Z.float(), torch.from_numpy(simple_molecule.Z.astype(np.float32)))
        torch.testing.assert_close(torch_mol.R, torch.from_numpy(simple_molecule.R))
    
    def test_to_torch_with_forces(self, molecule_with_forces):
        """Test torch conversion with forces"""
        torch_mol = molecule_with_forces.to_torch()
        assert isinstance(torch_mol.F, torch.Tensor)
        torch.testing.assert_close(torch_mol.F, torch.from_numpy(molecule_with_forces.F))


class TestMoleculeRoundTrip:
    """Test round-trip conversions"""
    
    def test_dict_roundtrip(self, molecule_dict):
        """Test dict -> Molecule -> dict roundtrip"""
        mol = Molecule.from_dict(molecule_dict)
        data_back = mol.to_dict()
        
        # Convert back to original types for comparison
        for key in molecule_dict:
            if key in data_back:
                np.testing.assert_array_almost_equal(
                    np.array(molecule_dict[key]), 
                    data_back[key]
                )
    
    def test_clone_roundtrip(self, periodic_molecule):
        """Test original -> clone -> modifications independence"""
        cloned = periodic_molecule.clone()
        
        # Modify clone
        cloned.R[0, 0] = 999.0
        cloned.E = -999.0
        
        # Original should be unchanged
        assert periodic_molecule.R[0, 0] != 999.0
        assert periodic_molecule.E != -999.0 