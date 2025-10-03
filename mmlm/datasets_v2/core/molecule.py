"""
Molecule dataclass for standardized molecular data representation.

This module provides the core Molecule dataclass that serves as the foundation
for all downstream operations in the datasets v2 architecture.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from numpy.typing import NDArray


@dataclass
class Molecule:
    """
    Standardized representation of a molecule with atomic structure and properties.

    This dataclass serves as the foundation for all downstream operations including
    transforms, binning, and formatting. All molecular data loaders should convert
    their raw data into this standardized format.

    Attributes:
        Z: Atomic numbers (shape: [n_atoms])
        R: Atomic positions in Cartesian coordinates (shape: [n_atoms, 3])
        F: Forces on atoms (shape: [n_atoms, 3], optional)
        E: Total energy (scalar, optional)
        cell: Lattice vectors for periodic systems (shape: [3, 3], optional)
        stress: Stress tensor (shape: [3, 3] or [6], optional)
        spin: Spin multiplicity (scalar, optional)
        charge: Net charge (scalar, optional)

    Examples:
        >>> mol = Molecule(
        ...     Z=np.array([1, 6, 1], dtype=np.int8),
        ...     R=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        ...     E=42.0
        ... )
        >>> mol.n_atoms
        3
        >>> mol.has_forces
        False
    """

    Z: NDArray[np.int8]  # atomic numbers
    R: NDArray[np.float32]  # positions [n_atoms, 3]
    F: Optional[NDArray[np.float32]] = None  # forces [n_atoms, 3]
    E: Optional[float] = None  # energy (scalar)
    cell: Optional[NDArray[np.float32]] = None  # lattice vectors [3, 3]
    stress: Optional[NDArray[np.float32]] = None  # stress tensor
    spin: Optional[int] = None  # spin multiplicity
    charge: Optional[int] = None  # net charge

    def __post_init__(self):
        """Validate molecular data after initialization."""
        self._validate()

    def _validate(self):
        """Validate the consistency of molecular data."""
        if self.Z.ndim != 1:
            raise ValueError(f"Z must be 1D array, got shape {self.Z.shape}")

        if self.R.ndim != 2 or self.R.shape[1] != 3:
            raise ValueError(f"R must be [n_atoms, 3] array, got shape {self.R.shape}")

        if len(self.Z) != len(self.R):
            raise ValueError(
                f"Z and R must have same length: {len(self.Z)} vs {len(self.R)}"
            )

        if self.F is not None:
            if self.F.ndim != 2 or self.F.shape[1] != 3:
                raise ValueError(
                    f"F must be [n_atoms, 3] array, got shape {self.F.shape}"
                )
            if len(self.F) != len(self.Z):
                raise ValueError(
                    f"F and Z must have same length: {len(self.F)} vs {len(self.Z)}"
                )

        if self.cell is not None:
            if self.cell.shape != (3, 3):
                raise ValueError(
                    f"cell must be [3, 3] array, got shape {self.cell.shape}"
                )

        if self.stress is not None:
            if self.stress.shape not in [(3, 3), (6,)]:
                raise ValueError(
                    f"stress must be [3, 3] or [6] array, got shape {self.stress.shape}"
                )

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the molecule."""
        return len(self.Z)

    @property
    def has_forces(self) -> bool:
        """Whether the molecule has force information."""
        return self.F is not None

    @property
    def has_energy(self) -> bool:
        """Whether the molecule has energy information."""
        return self.E is not None

    @property
    def has_cell(self) -> bool:
        """Whether the molecule has periodic cell information."""
        return self.cell is not None

    @property
    def has_stress(self) -> bool:
        """Whether the molecule has stress information."""
        return self.stress is not None

    @property
    def has_spin(self) -> bool:
        """Whether the molecule has spin information."""
        return self.spin is not None

    @property
    def has_charge(self) -> bool:
        """Whether the molecule has charge information."""
        return self.charge is not None

    @classmethod
    def from_dict(cls, data: dict) -> "Molecule":
        """
        Create a Molecule from a dictionary.

        Args:
            data: Dictionary containing molecular data with keys:
                  'Z', 'R', and optionally 'F', 'E', 'cell', 'stress', 'spin', 'charge'

        Returns:
            Molecule instance
        """
        return cls(
            Z=np.asarray(data["Z"], dtype=np.int8),
            R=np.asarray(data["R"], dtype=np.float32),
            F=(
                np.asarray(data["F"], dtype=np.float32)
                if "F" in data and data["F"] is not None
                else None
            ),
            E=float(data["E"]) if "E" in data and data["E"] is not None else None,
            cell=(
                np.asarray(data["cell"], dtype=np.float32)
                if "cell" in data and data["cell"] is not None
                else None
            ),
            stress=(
                np.asarray(data["stress"], dtype=np.float32)
                if "stress" in data and data["stress"] is not None
                else None
            ),
            spin=(
                int(data["spin"])
                if "spin" in data and data["spin"] is not None
                else None
            ),
            charge=(
                int(data["charge"])
                if "charge" in data and data["charge"] is not None
                else None
            ),
        )

    def to_dict(self) -> dict:
        """
        Convert the Molecule to a dictionary.

        Returns:
            Dictionary representation of the molecule
        """
        result = {
            "Z": self.Z.tolist(),
            "R": self.R.tolist(),
        }
        if self.F is not None:
            result["F"] = self.F.tolist()
        if self.E is not None:
            result["E"] = self.E
        if self.cell is not None:
            result["cell"] = self.cell.tolist()
        if self.stress is not None:
            result["stress"] = self.stress.tolist()
        if self.spin is not None:
            result["spin"] = self.spin
        if self.charge is not None:
            result["charge"] = self.charge
        return result

    def clone(self) -> "Molecule":
        """
        Create a deep copy of the molecule.

        Returns:
            New Molecule instance with copied data
        """
        return Molecule(
            Z=self.Z.copy(),
            R=self.R.copy(),
            F=self.F.copy() if self.F is not None else None,
            E=self.E,
            cell=self.cell.copy() if self.cell is not None else None,
            stress=self.stress.copy() if self.stress is not None else None,
            spin=self.spin,
            charge=self.charge,
        )

    def to_torch(self) -> "Molecule":
        """
        Convert numpy arrays to torch tensors.

        Returns:
            New Molecule instance with torch tensors
        """
        return Molecule(
            Z=torch.from_numpy(self.Z),
            R=torch.from_numpy(self.R),
            F=torch.from_numpy(self.F) if self.F is not None else None,
            E=self.E,
            cell=torch.from_numpy(self.cell) if self.cell is not None else None,
            stress=torch.from_numpy(self.stress) if self.stress is not None else None,
            spin=self.spin,
            charge=self.charge,
        )

    def __repr__(self) -> str:
        """String representation of the molecule."""
        attrs = [f"n_atoms={self.n_atoms}"]
        if self.has_energy:
            attrs.append(f"E={self.E:.3f}")
        if self.has_forces:
            attrs.append("forces=True")
        if self.has_cell:
            attrs.append("periodic=True")
        if self.has_stress:
            attrs.append("stress=True")
        if self.has_spin:
            attrs.append(f"spin={self.spin}")
        if self.has_charge:
            attrs.append(f"charge={self.charge}")
        return f"Molecule({', '.join(attrs)})"
