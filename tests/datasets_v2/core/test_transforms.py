"""
Tests for molecule transforms.
"""
import numpy as np
import pytest

from mmlm.datasets_v2.core.molecule import Molecule
from mmlm.datasets_v2.core.transforms import RotationTransform, PermutationTransform


def test_rotation_transform_simple_molecule(simple_molecule: Molecule):
    """Test rotation transform on a molecule without forces or cell."""
    transform = RotationTransform()
    
    original_R = simple_molecule.R.copy()
    transformed_mol = transform(simple_molecule)

    # Check that original molecule is unchanged
    assert np.array_equal(simple_molecule.R, original_R)

    # Check that positions are rotated
    assert not np.array_equal(transformed_mol.R, original_R)
    
    # Check that distances are preserved
    original_dist = np.linalg.norm(original_R[0] - original_R[1])
    transformed_dist = np.linalg.norm(transformed_mol.R[0] - transformed_mol.R[1])
    assert np.isclose(original_dist, transformed_dist)

    # Check other properties are unchanged
    assert np.array_equal(transformed_mol.Z, simple_molecule.Z)
    assert transformed_mol.E == simple_molecule.E


def test_rotation_transform_with_forces(molecule_with_forces: Molecule):
    """Test rotation transform on a molecule with forces."""
    transform = RotationTransform()
    
    original_R = molecule_with_forces.R.copy()
    original_F = molecule_with_forces.F.copy()
    
    transformed_mol = transform(molecule_with_forces)

    assert not np.array_equal(transformed_mol.R, original_R)
    assert not np.array_equal(transformed_mol.F, original_F)

    # Check that the angle between position and force vectors is preserved
    original_dot = np.dot(original_R[1] - original_R[0], original_F[0])
    transformed_dot = np.dot(transformed_mol.R[1] - transformed_mol.R[0], transformed_mol.F[0])
    assert np.isclose(original_dot, transformed_dot)


def test_rotation_transform_periodic(periodic_molecule: Molecule):
    """Test rotation transform on a periodic molecule with cell and stress."""
    transform = RotationTransform()
    
    # Make stress a 3x3 matrix to test rotation
    periodic_molecule.stress = np.array([
        [0.1, 0.0, 0.0],
        [0.0, 0.2, 0.0],
        [0.0, 0.0, 0.3]
    ])

    original_cell = periodic_molecule.cell.copy()
    original_stress = periodic_molecule.stress.copy()

    transformed_mol = transform(periodic_molecule)

    assert not np.array_equal(transformed_mol.cell, original_cell)
    assert not np.array_equal(transformed_mol.stress, original_stress)
    
    # Determinant of the cell matrix should be preserved
    assert np.isclose(np.linalg.det(original_cell), np.linalg.det(transformed_mol.cell))


def test_permutation_transform_simple_molecule(simple_molecule: Molecule):
    """Test permutation transform on a simple molecule."""
    transform = PermutationTransform()

    original_Z = simple_molecule.Z.copy()
    original_R = simple_molecule.R.copy()

    transformed_mol = transform(simple_molecule)

    # Check that original molecule is unchanged
    assert np.array_equal(simple_molecule.Z, original_Z)
    assert np.array_equal(simple_molecule.R, original_R)
    
    # Check that atoms are permuted
    assert not np.array_equal(transformed_mol.Z, original_Z)
    assert not np.array_equal(transformed_mol.R, original_R)
    
    # Check that the number of atoms of each type is the same
    unique_original, counts_original = np.unique(original_Z, return_counts=True)
    unique_transformed, counts_transformed = np.unique(transformed_mol.Z, return_counts=True)
    assert np.array_equal(unique_original, unique_transformed)
    assert np.array_equal(counts_original, counts_transformed)

    # Other properties should be unchanged
    assert transformed_mol.E == simple_molecule.E


def test_permutation_transform_with_forces(molecule_with_forces: Molecule):
    """Test permutation transform on a molecule with forces."""
    transform = PermutationTransform()
    
    original_Z = molecule_with_forces.Z.copy()
    original_R = molecule_with_forces.R.copy()
    original_F = molecule_with_forces.F.copy()

    transformed_mol = transform(molecule_with_forces)

    # For a molecule with identical atoms (H2), Z will not change.
    # The main check is that the set of (R, F, Z) tuples is preserved.
    assert np.array_equal(transformed_mol.Z, original_Z)

    # Since there are two identical atoms, we can't be sure which one it is.
    # Instead, we check if the set of (Z, R, F) tuples is the same.
    original_set = set(zip(map(tuple, original_R), map(tuple, original_F), original_Z))
    transformed_set = set(zip(map(tuple, transformed_mol.R), map(tuple, transformed_mol.F), transformed_mol.Z))
    assert original_set == transformed_set


def test_permutation_does_not_change_non_atomic_properties(periodic_molecule: Molecule):
    """Test that permutation does not affect non-atomic properties."""
    transform = PermutationTransform()

    original_cell = periodic_molecule.cell.copy()
    original_stress = periodic_molecule.stress.copy()
    original_E = periodic_molecule.E

    transformed_mol = transform(periodic_molecule)

    assert np.array_equal(transformed_mol.cell, original_cell)
    assert np.array_equal(transformed_mol.stress, original_stress)
    assert transformed_mol.E == original_E 