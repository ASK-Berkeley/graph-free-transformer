"""
Formatter classes for converting Molecule objects to text and tensor outputs.

This module provides the core formatting functionality for datasets v2 architecture,
implementing the Strategy Pattern with different formatting strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import torch
import numpy as np
from dataclasses import dataclass

from .molecule import Molecule
from .binner import BinningSpec
from e3nn.o3._spherical_harmonics import _spherical_harmonics

class Formatter(ABC):
    """
    Abstract base class for formatting Molecule objects.

    Formatters convert Molecule objects into text and tensor outputs suitable
    for training language models on molecular data.
    """

    def __init__(
        self,
        bos_token: str = "<BOS>",
        eos_token: str = "<EOS>",
        first_force_only: bool = False,
        joint_force_embedding: bool = True,
        finetune: bool = False,
    ):
        """
        Initialize formatter.

        Args:
            bos_token: Begin-of-sequence token.
            eos_token: End-of-sequence token.
            first_force_only: If True, only use the first force vector.
            joint_force_embedding: If True, embed force vectors as single tokens.
                Otherwise, embed each coordinate separately.
            finetune: If True, only include charge/spin/position sections and use simplified position tokens.
        """
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.first_force_only = first_force_only
        self.joint_force_embedding = joint_force_embedding
        self.finetune = finetune
    
    def __call__(self, mol: Molecule, bin_spec: BinningSpec) -> Tuple:
        """
        Format a molecule into text and tensor outputs.

        This is the base implementation that can be overridden by subclasses.

        Args:
            mol: Molecule object to format
            bin_spec: Binning specification for discretizing continuous values

        Returns:
            A tuple containing the formatted outputs.
            - Normal mode: (text, continuous_tensor)
            - Finetune mode: (text, continuous_tensor, energy_tensor, forces_tensor)
        """
        # Generate text representation
        text = self._generate_text(mol, bin_spec)

        # Generate continuous tensor
        continuous_tensor = self._generate_continuous_tensor(mol)

        if self.finetune:
            # Generate additional tensors for finetune mode
            energy_tensor = self._generate_energy_tensor(mol)
            forces_tensor = self._generate_forces_tensor(mol)
            return text, continuous_tensor, energy_tensor, forces_tensor
        else:
            return text, continuous_tensor

    def _generate_text(self, mol: Molecule, bin_spec: BinningSpec) -> str:
        """Generate text representation of molecule."""
        sections = [self.bos_token]

        if mol.spin is not None:
            sections.append(self._format_spin(mol, bin_spec))

        if mol.charge is not None:
            sections.append(self._format_charge(mol, bin_spec))

        if mol.R is not None:
            sections.append(self._format_positions(mol, bin_spec))

        # Skip force and target sections in finetune mode
        if not self.finetune:
            if mol.F is not None:
                sections.append(self._format_forces(mol, bin_spec))

            if mol.E is not None:
                sections.append(self._format_target(mol, bin_spec))

        sections.append(self.eos_token)

        return "\n".join(sections)

    def _generate_continuous_tensor(self, mol: Molecule) -> torch.Tensor:
        """Generate continuous tensor with 1-to-1 correspondence to text tokens."""
        continuous_values = [[0.0, 0.0, 0.0]]  # Placeholder for BOS

        if mol.spin is not None:
            continuous_values.extend(self._cont_spin(mol))

        if mol.charge is not None:
            continuous_values.extend(self._cont_charge(mol))

        if mol.R is not None:
            continuous_values.extend(self._cont_positions(mol))

        # Skip force and target sections in finetune mode
        if not self.finetune:
            if mol.F is not None:
                continuous_values.extend(self._cont_forces(mol))

            if mol.E is not None:
                continuous_values.extend(self._cont_target(mol))

        continuous_values.append([0.0, 0.0, 0.0])  # Placeholder for EOS

        return torch.tensor(continuous_values, dtype=torch.float32)

    def _generate_energy_tensor(self, mol: Molecule) -> torch.Tensor:
        """Generate energy tensor for finetune mode."""
        if mol.E is not None:
            return torch.tensor([mol.E], dtype=torch.float32)
        else:
            return torch.tensor([0.0], dtype=torch.float32)

    def _generate_forces_tensor(self, mol: Molecule) -> torch.Tensor:
        """Generate forces tensor for finetune mode."""
        if mol.F is not None:
            # In finetune mode, always return all forces (ignore first_force_only)
            return torch.from_numpy(mol.F).float()
        else:
            # Return zero forces with same shape as positions
            return torch.zeros(mol.n_atoms, 3, dtype=torch.float32)

    @abstractmethod
    def _format_positions(self, mol: Molecule, bin_spec: BinningSpec) -> str:
        """Format positions (differs between formatters)."""
        pass

    @abstractmethod
    def _cont_positions(self, mol: Molecule) -> List[List[float]]:
        """Get continuous values for positions (differs between formatters)."""
        pass

    def _format_forces(self, mol: Molecule, bin_spec: BinningSpec) -> str:
        """Format forces with section tokens."""
        lines = []

        forces = mol.F
        if self.first_force_only:
            forces = forces[:1]

        for i, force in enumerate(forces):
            if self.joint_force_embedding:
                # Use vec_2_string approach for joint embedding
                force_token = self._vec_2_string(force, bin_spec, "force")
                lines.append(force_token)
            else:
                # Use num_2_string for each component
                for component in force:
                    force_token = self._num_2_string(component, bin_spec, "force")
                    lines.append(force_token)

        force_vals = "\n".join(lines)
        return f"[FORCE]\n{force_vals}\n[FORCE_END]"

    def _cont_forces(self, mol: Molecule) -> List[List[float]]:
        """Get continuous values for forces, including section placeholders."""
        values = [[0.0, 0.0, 0.0]]  # [FORCE]

        forces = mol.F
        if self.first_force_only:
            forces = forces[:1]

        if self.joint_force_embedding:
            values.extend([force.tolist() for force in forces])
        else:
            for force in forces:
                for component in force:
                    values.append([float(component), 0.0, 0.0])

        values.append([0.0, 0.0, 0.0])  # [FORCE_END]
        return values

    def _format_target(self, mol: Molecule, bin_spec: BinningSpec) -> str:
        """Format target energy with section tokens."""
        target_val = self._num_2_string(mol.E, bin_spec, "target")
        return f"[TARGET]\n{target_val}\n[TARGET_END]"

    def _cont_target(self, mol: Molecule) -> List[List[float]]:
        """Get continuous values for target, including section placeholders."""
        return [
            [0.0, 0.0, 0.0],  # [TARGET]
            [float(mol.E), 0.0, 0.0],
            [0.0, 0.0, 0.0],  # [TARGET_END]
        ]

    def _format_spin(self, mol: Molecule, bin_spec: BinningSpec) -> str:
        """Format spin multiplicity with section tokens."""
        spin_val = f"<NUM_spin_{mol.spin}>"
        return f"[SPIN]\n{spin_val}\n[SPIN_END]"

    def _cont_spin(self, mol: Molecule) -> List[List[float]]:
        """Get continuous values for spin, including section placeholders."""
        return [
            [0.0, 0.0, 0.0],  # [SPIN]
            [float(mol.spin), 0.0, 0.0],
            [0.0, 0.0, 0.0],  # [SPIN_END]
        ]

    def _format_charge(self, mol: Molecule, bin_spec: BinningSpec) -> str:
        """Format charge with section tokens."""
        charge_val = f"<NUM_charge_{mol.charge}>"
        return f"[CHARGE]\n{charge_val}\n[CHARGE_END]"

    def _cont_charge(self, mol: Molecule) -> List[List[float]]:
        """Get continuous values for charge, including section placeholders."""
        return [
            [0.0, 0.0, 0.0],  # [CHARGE]
            [float(mol.charge), 0.0, 0.0],
            [0.0, 0.0, 0.0],  # [CHARGE_END]
        ]

    def _vec_2_string(self, vec: np.ndarray, bin_spec: BinningSpec, field: str) -> str:
        """
        Convert 3D vector to single token using joint embedding (base**3).

        This mirrors the vec_2_string method from text_dataset.py
        """
        # In finetune mode, use simplified position tokens
        if self.finetune and field == "pos":
            return "<NUM>"
        
        # Transform vector to bin indices by flattening
        idx_vector = bin_spec.transform(field, vec).flatten()

        # Get number of bins
        n_bins = bin_spec.n_bins[field]

        # Convert to single index using base**3 encoding
        idx = idx_vector[0] * n_bins**2 + idx_vector[1] * n_bins + idx_vector[2]

        return f"<NUM_{field}_{idx}>" if field != "pos" else f"<NUM_{idx}>"

    def _num_2_string(self, num: float, bin_spec: BinningSpec, field: str) -> str:
        """Convert scalar number to token."""
        bin_idx = bin_spec.transform(field, np.array([num]))[0].item()
        return f"<NUM_{field}_{bin_idx}>" if field != "pos" else f"<NUM_{bin_idx}>"


class StandardFormatter(Formatter):
    """
    Standard formatting strategy that embeds atomic numbers as text tokens.

    This formatter follows the joint embedding approach where positions and forces
    are represented as single tokens using vec_2_string (base**3 embedding).
    """

    # StandardFormatter uses the base __call__ implementation from Formatter

    def _format_positions(self, mol: Molecule, bin_spec: BinningSpec) -> str:
        """Format positions with atomic numbers and section tokens."""
        lines = []

        for i, (pos, z) in enumerate(zip(mol.R, mol.Z)):
            # Use vec_2_string approach for joint embedding
            pos_token = self._vec_2_string(pos, bin_spec, "pos")
            lines.append(f"a_{z}:")
            lines.append(pos_token)

        pos_vals = "\n".join(lines)
        return f"[POS]\n{pos_vals}\n[POS_END]"

    def _cont_positions(self, mol: Molecule) -> List[List[float]]:
        """Get continuous values for positions, including section placeholders."""
        values = [[0.0, 0.0, 0.0]]  # [POS]
        for i, pos in enumerate(mol.R):
            # Add placeholder for atomic number token
            values.append([0.0, 0.0, 0.0])
            # Add actual position values
            values.append(pos.tolist())
        values.append([0.0, 0.0, 0.0])  # [POS_END]
        return values


class AtomFormatter(Formatter):
    """
    Atom embedding formatter that separates atomic numbers into parallel tensor.

    This formatter removes atomic numbers from the text representation and instead
    returns them as a parallel tensor aligned with position tokens.
    """

    def __call__(
        self, mol: Molecule, bin_spec: BinningSpec
    ) -> Tuple[str, torch.Tensor, torch.Tensor, ...]:
        """
        Format molecule using atom embedding strategy.

        Args:
            mol: Molecule object to format
            bin_spec: Binning specification for discretizing continuous values

        Returns:
            A tuple containing:
            - text: The formatted text string.
            - cont: The continuous tensor.
            - atom_ids: The tensor of atom IDs.
            - energy: Energy tensor (finetune mode only).
            - forces: Forces tensor (finetune mode only).
        """
        # Call parent to get base outputs (and possibly energy/forces in finetune mode)
        parent_result = super().__call__(mol, bin_spec)
        text, continuous_tensor, *args = parent_result

        # Generate atom tensor
        atom_tensor = self._generate_atom_tensor(mol)

        return text, continuous_tensor, atom_tensor, *args

    def _format_positions(self, mol: Molecule, bin_spec: BinningSpec) -> str:
        """Format positions without atomic numbers but with section tokens."""
        lines = []

        for i, pos in enumerate(mol.R):
            # Use vec_2_string approach for joint embedding
            pos_token = self._vec_2_string(pos, bin_spec, "pos")
            lines.append(pos_token)

        pos_vals = "\n".join(lines)
        return f"[POS]\n{pos_vals}\n[POS_END]"

    def _cont_positions(self, mol: Molecule) -> List[List[float]]:
        """Get continuous values for positions, including section placeholders."""
        values = [[0.0, 0.0, 0.0]]  # [POS]
        values.extend([pos.tolist() for pos in mol.R])
        values.append([0.0, 0.0, 0.0])  # [POS_END]
        return values

    def _generate_atom_tensor(self, mol: Molecule) -> torch.Tensor:
        """Generate atom tensor with 1-to-1 correspondence to text tokens."""
        atom_values = [0]  # Placeholder for BOS

        if mol.spin is not None:
            atom_values.extend(self._atom_spin(mol))

        if mol.charge is not None:
            atom_values.extend(self._atom_charge(mol))

        if mol.R is not None:
            atom_values.extend(self._atom_positions(mol))

        # Skip force and target sections in finetune mode
        if not self.finetune:
            if mol.F is not None:
                atom_values.extend(self._atom_forces(mol))

            if mol.E is not None:
                atom_values.extend(self._atom_target(mol))

        atom_values.append(0)  # Placeholder for EOS

        return torch.tensor(atom_values, dtype=torch.long)

    def _atom_positions(self, mol: Molecule) -> List[int]:
        """Get atom values for positions, including section placeholders."""
        values = [0]  # [POS]
        values.extend([int(z) for z in mol.Z])
        values.append(0)  # [POS_END]
        return values

    def _atom_forces(self, mol: Molecule) -> List[int]:
        """Get atom values for forces, including section placeholders."""
        values = [0]  # [FORCE]
        forces = mol.F
        if self.first_force_only:
            forces = forces[:1]

        num_force_tokens = len(forces)
        if not self.joint_force_embedding:
            num_force_tokens *= 3  # Each force vector becomes 3 tokens

        values.extend([0] * num_force_tokens)
        values.append(0)  # [FORCE_END]
        return values

    def _atom_target(self, mol: Molecule) -> List[int]:
        """Get atom values for target, including section placeholders."""
        return [0, 0, 0]  # [TARGET], value, [TARGET_END]

    def _atom_spin(self, mol: Molecule) -> List[int]:
        """Get atom values for spin, including section placeholders."""
        return [0, 0, 0]  # [SPIN], value, [SPIN_END]

    def _atom_charge(self, mol: Molecule) -> List[int]:
        """Get atom values for charge, including section placeholders."""
        return [0, 0, 0]  # [CHARGE], value, [CHARGE_END]



class DirFormatter(AtomFormatter):

    def __init__(self, lmax: int, *args, **kwargs):
        """
        Initialize DirFormatter with spherical harmonics parameters.

        Args:
            lmax: Maximum spherical harmonic degree for directional features.
            *args, **kwargs: Arguments passed to parent AtomFormatter.
        """
        super().__init__(*args, **kwargs)
        self.lmax = lmax

    def __call__(
        self, mol: Molecule, bin_spec: BinningSpec
    ) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, ...]:
        """
        Format molecule using atom embedding strategy with directional BOO features.

        Args:
            mol: Molecule object to format
            bin_spec: Binning specification for discretizing continuous values

        Returns:
            A tuple containing:
            - text: The formatted text string.
            - cont: The continuous tensor.
            - atom_ids: The tensor of atom IDs.
            - node_boo: The directional BOO features tensor.
            - energy: Energy tensor (finetune mode only).
            - forces: Forces tensor (finetune mode only).
        """
        # Call parent to get all outputs (including energy/forces in finetune mode)
        parent_result = super().__call__(mol, bin_spec)
        text, cont, atom_ids, *args = parent_result

        # Generate directional BOO features
        node_boo = self._generate_dir_tensor(mol)
        print(f"node_boo: {node_boo.shape}")

        return text, cont, atom_ids, node_boo, *args

    @staticmethod
    def compilable_scatter(
        src: torch.Tensor,
        index: torch.Tensor,
        dim_size: int,
        dim: int = 0,
        reduce: str = "sum",
    ) -> torch.Tensor:
        """
        torch_scatter scatter function with compile support.
        Modified from torch_geometric.utils.scatter_.
        """

        def broadcast(src: torch.Tensor, ref: torch.Tensor, dim: int) -> torch.Tensor:
            dim = ref.dim() + dim if dim < 0 else dim
            size = ((1,) * dim) + (-1,) + ((1,) * (ref.dim() - dim - 1))
            return src.view(size).expand_as(ref)

        dim = src.dim() + dim if dim < 0 else dim
        size = src.size()[:dim] + (dim_size,) + src.size()[dim + 1 :]

        if reduce == "sum" or reduce == "add":
            index = broadcast(index, src, dim)
            return src.new_zeros(size).scatter_add_(dim, index, src)

        if reduce == "mean":
            count = src.new_zeros(dim_size)
            count.scatter_add_(0, index, src.new_ones(src.size(dim)))
            count = count.clamp(min=1)

            index = broadcast(index, src, dim)
            out = src.new_zeros(size).scatter_add_(dim, index, src)

            return out / broadcast(count, out, dim)

        raise ValueError(f"Invalid reduce option '{reduce}'.")


    def _generate_dir_tensor(self, mol: Molecule) -> torch.Tensor:
        """Generate directional BOO features."""
        # Convert numpy positions to torch tensor
        positions = torch.from_numpy(mol.R).float()
        
        displacements = positions.unsqueeze(0) - positions.unsqueeze(1)
        directions = torch.nn.functional.normalize(displacements, dim=-1)

        edge_sh = _spherical_harmonics(
            lmax=self.lmax,
            x=directions[:, :, 0],
            y=directions[:, :, 1],
            z=directions[:, :, 2],
        )

        # Normalize spherical harmonics by sqrt(2l+1) to improve numerical stability
        sh_index = torch.arange(self.lmax + 1, device=edge_sh.device)
        sh_index = torch.repeat_interleave(sh_index, 2 * sh_index + 1)
        edge_sh = edge_sh / torch.clamp(torch.sqrt(2 * sh_index + 1), min=1e-6).unsqueeze(
            0
        ).unsqueeze(0)

        node_boo = edge_sh.mean(dim=1)

        node_boo_squared = node_boo**2
        node_boo = DirFormatter.compilable_scatter(
            node_boo_squared, sh_index, dim_size=self.lmax + 1, dim=-1, reduce="sum"
        )
        node_boo = torch.clamp(node_boo, min=1e-6).sqrt()

        return node_boo