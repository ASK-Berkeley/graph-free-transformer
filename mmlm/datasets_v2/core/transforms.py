"""
Transforms for augmenting Molecule objects.

This module provides composable transform classes that apply augmentations
like rotation and permutation to Molecule objects. These are designed to be
used within the datasets_v2 framework.
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

from .molecule import Molecule


class Transform(ABC):
    """
    Abstract base class for all molecule transforms.

    Subclasses must implement the `__call__` method to apply a specific
    transformation to a Molecule object.
    """

    @abstractmethod
    def __call__(self, molecule: Molecule) -> Molecule:
        """
        Apply the transform to a molecule.

        Args:
            molecule: The input Molecule object.

        Returns:
            The transformed Molecule object.
        """
        pass


class RotationTransform(Transform):
    """
    Applies a random rotation to the molecule.

    This transform rotates atomic positions, forces, and cell vectors.
    """

    def __call__(self, molecule: Molecule) -> Molecule:
        """
        Apply a random rotation to the molecule.

        Args:
            molecule: The input Molecule object.

        Returns:
            A new Molecule object with rotated properties.
        """
        # Generate a random rotation matrix
        rot_vec = np.random.rand(3)
        angle = np.random.rand() * 2 * np.pi
        rot_vec = (rot_vec / np.linalg.norm(rot_vec)) * angle
        rotation_matrix = ScipyRotation.from_rotvec(rot_vec).as_matrix()

        # Clone molecule to avoid modifying the original
        mol = molecule.clone()

        # Apply rotation
        mol.R = mol.R @ rotation_matrix.T
        if mol.has_forces:
            mol.F = mol.F @ rotation_matrix.T
        if mol.has_cell:
            mol.cell = mol.cell @ rotation_matrix
        if mol.has_stress:
            if mol.stress.shape == (3, 3):
                mol.stress = rotation_matrix @ mol.stress @ rotation_matrix.T
            else:
                raise ValueError(
                    "Stress tensor must be in 3x3 matrix form, not Voigt notation. Found shape: {mol.stress.shape}"
                )

        return mol


class PermutationTransform(Transform):
    """
    Applies a random permutation to the atoms in a molecule.

    This transform permutes atomic numbers, positions, and forces consistently.
    """

    def __call__(self, molecule: Molecule) -> Molecule:
        """
        Apply a random atomic permutation to the molecule.

        Args:
            molecule: The input Molecule object.

        Returns:
            A new Molecule object with permuted atoms.
        """
        # Generate a random permutation
        perm = np.random.permutation(molecule.n_atoms)

        # Clone molecule to avoid modifying the original
        mol = molecule.clone()

        # Apply permutation
        mol.Z = mol.Z[perm]
        mol.R = mol.R[perm]
        if mol.has_forces:
            mol.F = mol.F[perm]

        return mol
