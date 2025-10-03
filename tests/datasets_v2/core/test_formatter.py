"""
Tests for Formatter classes.

This suite validates the functionality of StandardFormatter and AtomFormatter,
ensuring they correctly convert Molecule objects into text and tensor outputs.
"""

import pytest
import torch
import numpy as np

from mmlm.datasets_v2.core.formatter import StandardFormatter, AtomFormatter, DirFormatter
from mmlm.datasets_v2.core.molecule import Molecule


@pytest.fixture
def formatter_molecule():
    """A molecule with all relevant fields for formatter testing."""
    return Molecule(
        Z=np.array([6, 1], dtype=np.int8),
        R=np.array([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]], dtype=np.float32),
        E=-50.0,
        F=np.array([[-0.1, -0.2, -0.3], [0.1, 0.2, 0.3]], dtype=np.float32),
        spin=1,
        charge=0,
    )


class TestStandardFormatter:
    """Tests for the StandardFormatter."""

    def test_format_simple_molecule(self, simple_molecule, comprehensive_binning_spec):
        """Test formatting of a simple molecule with only Z, R, E."""
        formatter = StandardFormatter()
        text, cont_tensor = formatter(simple_molecule, comprehensive_binning_spec)

        # Validate text output
        assert text.startswith("<BOS>")
        assert text.endswith("<EOS>")
        assert "[POS]" in text
        assert "[TARGET]" in text
        assert "[FORCE]" not in text  # No forces in simple_molecule
        
        # Check for atomic number prefixes
        assert "a_6:" in text
        assert "a_1:" in text
        
        # Validate tensor alignment
        lines = text.split('\n')
        assert len(lines) == cont_tensor.shape[0]

    def test_format_comprehensive_molecule(self, formatter_molecule, comprehensive_binning_spec):
        """Test formatting of a periodic molecule with all fields."""
        formatter = StandardFormatter()
        text, cont_tensor = formatter(formatter_molecule, comprehensive_binning_spec)

        # Validate text output
        assert "[SPIN]" in text
        assert "[CHARGE]" in text
        assert "[CELL]" not in text
        assert "[POS]" in text
        assert "[FORCE]" in text
        assert "[TARGET]" in text
        
        # Check for specific formatted values
        assert "<NUM_spin_1>" in text
        assert "<NUM_charge_0>" in text
        assert "a_6:" in text
        assert "a_1:" in text
        
        # Validate tensor alignment
        lines = text.split('\n')
        assert len(lines) == cont_tensor.shape[0]
        
        # Validate continuous tensor content
        # Example check: BOS token should have zero vector
        torch.testing.assert_close(cont_tensor[0], torch.zeros(3))
        # Example check: EOS token should have zero vector
        torch.testing.assert_close(cont_tensor[-1], torch.zeros(3))
        
        # Check that a_Z: token corresponds to a zero vector
        pos_start_index = lines.index("[POS]")
        atom_label_index = pos_start_index + 1
        assert lines[atom_label_index] == "a_6:"
        torch.testing.assert_close(cont_tensor[atom_label_index], torch.zeros(3))
        
        # Check that position vector is correct
        pos_vector_index = atom_label_index + 1
        torch.testing.assert_close(cont_tensor[pos_vector_index], torch.tensor([0.1, 0.2, 0.3]))

        # Check that different atom positions produce different tokens
        pos_2_start_index = lines.index("a_1:")
        pos_1_token = lines[pos_vector_index]
        pos_2_token = lines[pos_2_start_index + 1]
        assert pos_1_token != pos_2_token

    def test_first_force_only_enabled(
        self, formatter_molecule, comprehensive_binning_spec
    ):
        """Test that `first_force_only=True` formats only the first force."""
        formatter = StandardFormatter(first_force_only=True)

        text, cont_tensor = formatter(formatter_molecule, comprehensive_binning_spec)

        # Check that only one force token is present
        assert text.count("<NUM_force_") == 1

        # Check that the continuous tensor has the correct reduced size
        # BOS(1) + spin(3) + charge(3) + POS(1+2*2+1=6) + FORCE(1+1+1=3) + TARGET(3) + EOS(1)
        # Total = 1 + 3 + 3 + 6 + 3 + 3 + 1 = 20
        assert cont_tensor.shape[0] == 20

    def test_first_force_only_disabled(
        self, formatter_molecule, comprehensive_binning_spec
    ):
        """Test that default behavior includes all forces."""
        formatter = StandardFormatter()
        text, _ = formatter(formatter_molecule, comprehensive_binning_spec)

        # formatter_molecule has 2 atoms, so 2 forces should be present
        assert text.count("<NUM_force_") == 2

    def test_first_force_only_value(
        self, formatter_molecule, comprehensive_binning_spec
    ):
        """Test that the force value with `first_force_only=True` is correct."""
        formatter = StandardFormatter(first_force_only=True)

        text, cont_tensor = formatter(formatter_molecule, comprehensive_binning_spec)

        # Find the force value in the continuous tensor
        lines = text.split("\n")
        force_token_index = lines.index("[FORCE]") + 1
        output_force_vector = cont_tensor[force_token_index]

        # Get the expected force vector (the first one)
        expected_force_vector = torch.from_numpy(formatter_molecule.F[0])

        torch.testing.assert_close(output_force_vector, expected_force_vector)

    def test_separate_force_coords(
        self, formatter_molecule, comprehensive_binning_spec
    ):
        """Test that `joint_force_embedding=False` formats forces separately."""
        formatter = StandardFormatter(joint_force_embedding=False)

        text, cont_tensor = formatter(formatter_molecule, comprehensive_binning_spec)

        # We expect 3 tokens per force vector, and there are 2 forces
        assert text.count("<NUM_force_") == 6

        # The continuous tensor should also be larger
        # BOS(1) + spin(3) + charge(3) + POS(1+2*2+1=6) + FORCE(1+2*3+1=8) + TARGET(3) + EOS(1)
        # Total = 1 + 3 + 3 + 6 + 8 + 3 + 1 = 25
        assert cont_tensor.shape[0] == 25

        # Check the values of the separated force vectors
        lines = text.split("\n")
        force_start_index = lines.index("[FORCE]") + 1
        output_force_1 = cont_tensor[force_start_index]
        expected_force_1 = torch.tensor([formatter_molecule.F[0, 0], 0.0, 0.0])
        torch.testing.assert_close(output_force_1, expected_force_1)


class TestAtomFormatter:
    """Tests for the AtomFormatter."""

    def test_format_simple_molecule(self, simple_molecule, comprehensive_binning_spec):
        """Test formatting of a simple molecule, checking for atom_ids."""
        formatter = AtomFormatter()
        text, cont_tensor, atom_tensor = formatter(simple_molecule, comprehensive_binning_spec)

        # Validate text output
        assert "a_6:" not in text  # Atom numbers should not be in text
        assert "a_1:" not in text

        # Validate tensor existence and alignment
        lines = text.split('\n')
        assert len(lines) == cont_tensor.shape[0]
        assert len(lines) == atom_tensor.shape[0]

    def test_format_periodic_molecule_atom_ids(self, formatter_molecule, comprehensive_binning_spec):
        """Test the content of the atom_ids tensor for a complex molecule."""
        formatter = AtomFormatter()
        text, _, atom_tensor = formatter(formatter_molecule, comprehensive_binning_spec)

        lines = text.split('\n')

        # Validate alignment
        assert len(lines) == atom_tensor.shape[0]

        # Find position tokens and check corresponding atom_ids
        pos_start_index = lines.index("[POS]")
        
        # First atom (Z=6)
        pos_1_index = pos_start_index + 1
        assert lines[pos_1_index].startswith("<NUM_")
        assert atom_tensor[pos_1_index] == 6

        # Second atom (Z=1)
        pos_2_index = pos_start_index + 2
        assert lines[pos_2_index].startswith("<NUM_")
        assert atom_tensor[pos_2_index] == 1
        
        # Check that different atom positions produce different tokens
        assert lines[pos_1_index] != lines[pos_2_index]

        # Check that non-position tokens have atom_id 0
        # Example: [SPIN] token
        spin_start_index = lines.index("[SPIN]")
        assert atom_tensor[spin_start_index] == 0
        # Example: Spin value token
        assert atom_tensor[spin_start_index + 1] == 0
        # Example: BOS token
        assert atom_tensor[0] == 0
        # Example: EOS token
        assert atom_tensor[-1] == 0

    def test_first_force_only_enabled(
        self, formatter_molecule, comprehensive_binning_spec
    ):
        """Test that `first_force_only=True` formats only the first force."""
        formatter = AtomFormatter(first_force_only=True)

        text, cont_tensor, atom_tensor = formatter(
            formatter_molecule, comprehensive_binning_spec
        )

        # Check that only one force token is present
        assert text.count("<NUM_force_") == 1

        # Check that the continuous tensor has the correct reduced size
        # BOS(1) + spin(3) + charge(3) + POS(1+2+1=4) + FORCE(1+1+1=3) + TARGET(3) + EOS(1)
        # Total = 1 + 3 + 3 + 4 + 3 + 3 + 1 = 18
        assert cont_tensor.shape[0] == 18
        assert atom_tensor.shape[0] == 18

    def test_first_force_only_disabled(
        self, formatter_molecule, comprehensive_binning_spec
    ):
        """Test that default behavior includes all forces."""
        formatter = AtomFormatter()
        text, _, _ = formatter(formatter_molecule, comprehensive_binning_spec)

        # formatter_molecule has 2 atoms, so 2 forces should be present
        assert text.count("<NUM_force_") == 2

    def test_first_force_only_value(
        self, formatter_molecule, comprehensive_binning_spec
    ):
        """Test that the force value with `first_force_only=True` is correct."""
        formatter = AtomFormatter(first_force_only=True)

        text, cont_tensor, _ = formatter(
            formatter_molecule, comprehensive_binning_spec
        )

        # Find the force value in the continuous tensor
        lines = text.split("\n")
        force_token_index = lines.index("[FORCE]") + 1
        output_force_vector = cont_tensor[force_token_index]

        # Get the expected force vector (the first one)
        expected_force_vector = torch.from_numpy(formatter_molecule.F[0])

        torch.testing.assert_close(output_force_vector, expected_force_vector)

    def test_separate_force_coords(
        self, formatter_molecule, comprehensive_binning_spec
    ):
        """Test that `joint_force_embedding=False` formats forces separately."""
        formatter = AtomFormatter(joint_force_embedding=False)

        text, cont_tensor, atom_tensor = formatter(
            formatter_molecule, comprehensive_binning_spec
        )

        # We expect 3 tokens per force vector, and there are 2 forces
        assert text.count("<NUM_force_") == 6

        # The continuous tensor should also be larger
        # BOS(1) + spin(3) + charge(3) + POS(1+2+1=4) + FORCE(1+2*3+1=8) + TARGET(3) + EOS(1)
        # Total = 1 + 3 + 3 + 4 + 8 + 3 + 1 = 23
        assert cont_tensor.shape[0] == 23
        assert atom_tensor.shape[0] == 23


class TestFinetuneMode:
    """Tests for finetune mode functionality across all formatters."""

    def test_standard_formatter_finetune_mode(self, formatter_molecule, comprehensive_binning_spec):
        """Test StandardFormatter in finetune mode."""
        # Normal mode
        formatter_normal = StandardFormatter(finetune=False)
        text_normal, cont_normal = formatter_normal(formatter_molecule, comprehensive_binning_spec)
        
        # Finetune mode
        formatter_finetune = StandardFormatter(finetune=True)
        result_finetune = formatter_finetune(formatter_molecule, comprehensive_binning_spec)
        text_finetune, cont_finetune = result_finetune[0], result_finetune[1]
        
        # 1. Verify sections: only spin, charge, and position should remain
        assert "[SPIN]" in text_finetune
        assert "[CHARGE]" in text_finetune
        assert "[POS]" in text_finetune
        assert "[FORCE]" not in text_finetune
        assert "[TARGET]" not in text_finetune
        
        # 2. Verify position tokens are simplified
        pos_start = text_finetune.find('[POS]')
        pos_end = text_finetune.find('[POS_END]', pos_start)
        pos_section = text_finetune[pos_start:pos_end]
        
        # Should contain <NUM> tokens but not <NUM_xxx> tokens in position section
        assert '<NUM>' in pos_section
        assert '<NUM_' not in pos_section  # No indexed position tokens
        
        # 3. Verify continuous tensor is smaller
        assert cont_finetune.shape[0] < cont_normal.shape[0]
        
        # 4. Verify still has BOS/EOS
        assert text_finetune.startswith("<BOS>")
        assert text_finetune.endswith("<EOS>")

    def test_atom_formatter_finetune_mode(self, formatter_molecule, comprehensive_binning_spec):
        """Test AtomFormatter in finetune mode."""
        # Normal mode
        formatter_normal = AtomFormatter(finetune=False)
        text_normal, cont_normal, atom_normal = formatter_normal(formatter_molecule, comprehensive_binning_spec)
        
        # Finetune mode
        formatter_finetune = AtomFormatter(finetune=True)
        result_finetune = formatter_finetune(formatter_molecule, comprehensive_binning_spec)
        text_finetune, cont_finetune, atom_finetune = result_finetune[0], result_finetune[1], result_finetune[2]
        
        # 1. Verify sections: only spin, charge, and position should remain
        assert "[SPIN]" in text_finetune
        assert "[CHARGE]" in text_finetune
        assert "[POS]" in text_finetune
        assert "[FORCE]" not in text_finetune
        assert "[TARGET]" not in text_finetune
        
        # 2. Verify no atomic number prefixes in text (AtomFormatter removes them)
        assert "a_6:" not in text_finetune
        assert "a_1:" not in text_finetune
        
        # 3. Verify position tokens are simplified
        pos_start = text_finetune.find('[POS]')
        pos_end = text_finetune.find('[POS_END]', pos_start)
        pos_section = text_finetune[pos_start:pos_end]
        
        assert '<NUM>' in pos_section
        assert '<NUM_' not in pos_section
        
        # 4. Verify tensors are smaller but still aligned
        assert cont_finetune.shape[0] < cont_normal.shape[0]
        assert atom_finetune.shape[0] < atom_normal.shape[0]
        assert cont_finetune.shape[0] == atom_finetune.shape[0]
        
        # 5. Verify atom tensor has correct values for position tokens
        lines = text_finetune.split('\n')
        pos_start_idx = lines.index("[POS]")
        
        # Check that position tokens have correct atom IDs
        for i, z in enumerate(formatter_molecule.Z):
            pos_token_idx = pos_start_idx + 1 + i
            assert atom_finetune[pos_token_idx] == z

    def test_finetune_mode_token_alignment(self, formatter_molecule, comprehensive_binning_spec):
        """Test that all tensors remain properly aligned in finetune mode."""
        formatter = AtomFormatter(finetune=True)
        result = formatter(formatter_molecule, comprehensive_binning_spec)
        text, cont_tensor, atom_tensor = result[0], result[1], result[2]
        
        lines = text.split('\n')
        
        # All tensors should have same length as text lines
        assert len(lines) == cont_tensor.shape[0]
        assert len(lines) == atom_tensor.shape[0]
        
        # Verify BOS/EOS tokens have expected values
        assert atom_tensor[0] == 0  # BOS
        assert atom_tensor[-1] == 0  # EOS
        torch.testing.assert_close(cont_tensor[0], torch.zeros(3))  # BOS
        torch.testing.assert_close(cont_tensor[-1], torch.zeros(3))  # EOS

    def test_finetune_mode_simplified_tokens_count(self, formatter_molecule, comprehensive_binning_spec):
        """Test that finetune mode produces the expected number of simplified tokens."""
        formatter = StandardFormatter(finetune=True)
        result = formatter(formatter_molecule, comprehensive_binning_spec)
        text = result[0]
        
        # Count simplified position tokens
        pos_start = text.find('[POS]')
        pos_end = text.find('[POS_END]', pos_start)
        pos_section = text[pos_start:pos_end]
        
        # Should have one <NUM> token per atom
        num_simplified_tokens = pos_section.count('<NUM>')
        assert num_simplified_tokens == formatter_molecule.n_atoms

    def test_finetune_mode_preserves_other_tokens(self, formatter_molecule, comprehensive_binning_spec):
        """Test that finetune mode preserves non-position tokens correctly."""
        formatter = StandardFormatter(finetune=True)
        result = formatter(formatter_molecule, comprehensive_binning_spec)
        text = result[0]
        
        # Spin and charge tokens should still be formatted normally
        assert "<NUM_spin_1>" in text
        assert "<NUM_charge_0>" in text
        
        # Should not affect non-position token formatting
        assert "[SPIN_END]" in text
        assert "[CHARGE_END]" in text

    def test_finetune_mode_no_forces_no_targets(self, simple_molecule, comprehensive_binning_spec):
        """Test finetune mode with molecule that has no forces/targets."""
        # simple_molecule only has E (energy/target), no forces
        formatter = StandardFormatter(finetune=True)
        result = formatter(simple_molecule, comprehensive_binning_spec)
        text = result[0]
        
        # Should not have force or target sections
        assert "[FORCE]" not in text
        assert "[TARGET]" not in text
        
        # Should still have other sections
        assert "[POS]" in text
        
        # Should have simplified position tokens
        pos_start = text.find('[POS]')
        pos_end = text.find('[POS_END]', pos_start)
        pos_section = text[pos_start:pos_end]
        assert '<NUM>' in pos_section

    def test_finetune_mode_returns_energy_forces(self, formatter_molecule, comprehensive_binning_spec):
        """Test that finetune mode returns energy and forces tensors."""
        # Test StandardFormatter
        formatter = StandardFormatter(finetune=True)
        result = formatter(formatter_molecule, comprehensive_binning_spec)
        
        # Should return 4 items in finetune mode
        assert len(result) == 4
        text, cont_tensor, energy_tensor, forces_tensor = result
        
        # Verify energy tensor
        assert isinstance(energy_tensor, torch.Tensor)
        assert energy_tensor.shape == (1,)
        assert torch.allclose(energy_tensor, torch.tensor([formatter_molecule.E]))
        
        # Verify forces tensor
        assert isinstance(forces_tensor, torch.Tensor)
        assert forces_tensor.shape == (formatter_molecule.n_atoms, 3)
        expected_forces = torch.from_numpy(formatter_molecule.F).float()
        torch.testing.assert_close(forces_tensor, expected_forces)

    def test_atom_formatter_finetune_mode_returns_energy_forces(self, formatter_molecule, comprehensive_binning_spec):
        """Test that AtomFormatter in finetune mode returns energy and forces tensors."""
        # Test AtomFormatter
        formatter = AtomFormatter(finetune=True)
        result = formatter(formatter_molecule, comprehensive_binning_spec)
        
        # Should return 5 items in finetune mode: text, cont, atom_ids, energy, forces
        assert len(result) == 5
        text, cont_tensor, atom_tensor, energy_tensor, forces_tensor = result
        
        # Verify energy tensor
        assert isinstance(energy_tensor, torch.Tensor)
        assert energy_tensor.shape == (1,)
        assert torch.allclose(energy_tensor, torch.tensor([formatter_molecule.E]))
        
        # Verify forces tensor
        assert isinstance(forces_tensor, torch.Tensor)
        assert forces_tensor.shape == (formatter_molecule.n_atoms, 3)
        expected_forces = torch.from_numpy(formatter_molecule.F).float()
        torch.testing.assert_close(forces_tensor, expected_forces)

    def test_finetune_mode_edge_case_no_forces(self, comprehensive_binning_spec):
        """Test finetune mode with molecule that has no forces."""
        # Create molecule without forces
        mol_no_forces = Molecule(
            Z=np.array([1, 1], dtype=np.int8),
            R=np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], dtype=np.float32),
            E=42.0,
            spin=1,
            charge=0
        )
        
        formatter = StandardFormatter(finetune=True)
        result = formatter(mol_no_forces, comprehensive_binning_spec)
        
        assert len(result) == 4
        text, cont_tensor, energy_tensor, forces_tensor = result
        
        # Energy should be present
        assert torch.allclose(energy_tensor, torch.tensor([42.0]))
        
        # Forces should be zero-filled with correct shape
        assert forces_tensor.shape == (mol_no_forces.n_atoms, 3)
        assert torch.allclose(forces_tensor, torch.zeros(mol_no_forces.n_atoms, 3))

    def test_finetune_mode_ignores_first_force_only(self, formatter_molecule, comprehensive_binning_spec):
        """Test that first_force_only flag is ignored in finetune mode."""
        # Formatter with first_force_only=True in finetune mode
        formatter = StandardFormatter(finetune=True, first_force_only=True)
        result = formatter(formatter_molecule, comprehensive_binning_spec)
        
        assert len(result) == 4
        text, cont_tensor, energy_tensor, forces_tensor = result
        
        # Forces tensor should contain ALL forces, not just the first one
        assert forces_tensor.shape == (formatter_molecule.n_atoms, 3)
        expected_all_forces = torch.from_numpy(formatter_molecule.F).float()
        torch.testing.assert_close(forces_tensor, expected_all_forces)

    def test_normal_mode_return_count(self, formatter_molecule, comprehensive_binning_spec):
        """Test that normal mode returns correct number of items."""
        # StandardFormatter normal mode
        formatter_std = StandardFormatter(finetune=False)
        result_std = formatter_std(formatter_molecule, comprehensive_binning_spec)
        assert len(result_std) == 2  # text, cont_tensor
        
        # AtomFormatter normal mode  
        formatter_atom = AtomFormatter(finetune=False)
        result_atom = formatter_atom(formatter_molecule, comprehensive_binning_spec)
        assert len(result_atom) == 3  # text, cont_tensor, atom_tensor


class TestDirFormatter:
    """Tests for the DirFormatter with directional BOO features."""

    def test_dir_formatter_basic_functionality(self, formatter_molecule, comprehensive_binning_spec):
        """Test DirFormatter basic functionality."""
        # DirFormatter requires lmax parameter
        formatter = DirFormatter(lmax=2)
        result = formatter(formatter_molecule, comprehensive_binning_spec)
        
        # Should return 4 outputs in normal mode: text, cont_tensor, atom_tensor, node_boo
        assert len(result) == 4
        text, cont_tensor, atom_tensor, node_boo = result
        
        # Validate output types
        assert isinstance(text, str)
        assert isinstance(cont_tensor, torch.Tensor)
        assert isinstance(atom_tensor, torch.Tensor)
        assert isinstance(node_boo, torch.Tensor)
        
        # Check directional features shape: [n_atoms, lmax+1]
        assert node_boo.shape == (formatter_molecule.n_atoms, 3)  # lmax=2, so lmax+1=3
        
        # Verify all values are non-negative (after sqrt operation)
        assert torch.all(node_boo >= 0)
        
        # Verify no NaN or Inf values
        assert torch.all(torch.isfinite(node_boo))

    def test_dir_formatter_finetune_mode(self, formatter_molecule, comprehensive_binning_spec):
        """Test DirFormatter in finetune mode."""
        # Normal mode
        formatter_normal = DirFormatter(lmax=2, finetune=False)
        result_normal = formatter_normal(formatter_molecule, comprehensive_binning_spec)
        
        # Finetune mode
        formatter_finetune = DirFormatter(lmax=2, finetune=True)
        result_finetune = formatter_finetune(formatter_molecule, comprehensive_binning_spec)
        
        # Normal mode: 4 outputs
        assert len(result_normal) == 4
        text_normal, cont_normal, atom_normal, node_boo_normal = result_normal
        
        # Finetune mode: 6 outputs (adds energy and forces)
        assert len(result_finetune) == 6
        text_finetune, cont_finetune, atom_finetune, node_boo_finetune, energy_finetune, forces_finetune = result_finetune
        
        # Verify finetune sections behavior (only spin, charge, position)
        assert "[SPIN]" in text_finetune
        assert "[CHARGE]" in text_finetune  
        assert "[POS]" in text_finetune
        assert "[FORCE]" not in text_finetune
        assert "[TARGET]" not in text_finetune
        
        # Directional features should be the same (only depends on positions)
        torch.testing.assert_close(node_boo_normal, node_boo_finetune)
        
        # Verify energy and forces tensors
        assert isinstance(energy_finetune, torch.Tensor)
        assert isinstance(forces_finetune, torch.Tensor)
        assert energy_finetune.shape == (1,)
        assert forces_finetune.shape == (formatter_molecule.n_atoms, 3)

    def test_dir_formatter_different_lmax_values(self, formatter_molecule, comprehensive_binning_spec):
        """Test DirFormatter with different lmax values."""
        for lmax in [0, 1, 2, 3]:
            formatter = DirFormatter(lmax=lmax)
            result = formatter(formatter_molecule, comprehensive_binning_spec)
            
            text, cont_tensor, atom_tensor, node_boo = result
            
            # Check that node_boo has correct shape for this lmax
            expected_shape = (formatter_molecule.n_atoms, lmax + 1)
            assert node_boo.shape == expected_shape
            
            # All values should be finite and non-negative
            assert torch.all(torch.isfinite(node_boo))
            assert torch.all(node_boo >= 0)

    def test_dir_formatter_rotational_invariance(self, comprehensive_binning_spec):
        """Test that DirFormatter features are rotationally invariant."""
        # Create a simple molecule
        mol = Molecule(
            Z=np.array([1, 6, 8], dtype=np.int8),
            R=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            E=42.0,
            spin=1,
            charge=0
        )
        
        formatter = DirFormatter(lmax=2)
        
        # Get features for original molecule
        result_orig = formatter(mol, comprehensive_binning_spec)
        node_boo_orig = result_orig[3]
        
        # Create rotated molecule (90 degree rotation around z-axis)
        rotation_matrix = np.array([
            [0, -1, 0],
            [1,  0, 0], 
            [0,  0, 1]
        ], dtype=np.float32)
        
        mol_rotated = Molecule(
            Z=mol.Z.copy(),
            R=(mol.R @ rotation_matrix.T),  # Apply rotation
            E=mol.E,
            spin=mol.spin,
            charge=mol.charge
        )
        
        # Get features for rotated molecule
        result_rot = formatter(mol_rotated, comprehensive_binning_spec)
        node_boo_rot = result_rot[3]
        
        # Features should be approximately the same (within numerical precision)
        torch.testing.assert_close(node_boo_orig, node_boo_rot, rtol=1e-5, atol=1e-6)

    def test_dir_formatter_single_atom_edge_case(self, comprehensive_binning_spec):
        """Test DirFormatter with single atom (edge case)."""
        # Single atom molecule
        mol_single = Molecule(
            Z=np.array([6], dtype=np.int8),
            R=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            E=0.0,
            spin=1,
            charge=0
        )
        
        formatter = DirFormatter(lmax=2)
        result = formatter(mol_single, comprehensive_binning_spec)
        
        text, cont_tensor, atom_tensor, node_boo = result
        
        # Should handle single atom gracefully
        assert node_boo.shape == (1, 3)  # 1 atom, lmax+1=3
        assert torch.all(torch.isfinite(node_boo))

    def test_dir_formatter_inheritance_pattern(self, formatter_molecule, comprehensive_binning_spec):
        """Test that DirFormatter properly inherits AtomFormatter behavior."""
        atom_formatter = AtomFormatter()
        dir_formatter = DirFormatter(lmax=2)
        
        # Get results from both
        atom_result = atom_formatter(formatter_molecule, comprehensive_binning_spec)
        dir_result = dir_formatter(formatter_molecule, comprehensive_binning_spec)
        
        # First 3 outputs should be identical (text, cont, atom_ids)
        assert atom_result[0] == dir_result[0]  # text
        torch.testing.assert_close(atom_result[1], dir_result[1])  # cont_tensor
        torch.testing.assert_close(atom_result[2], dir_result[2])  # atom_tensor
        
        # DirFormatter should have additional node_boo output
        assert len(dir_result) == len(atom_result) + 1

    def test_dir_formatter_numerical_stability(self, comprehensive_binning_spec):
        """Test DirFormatter numerical stability with edge cases."""
        # Create molecule with very close atoms (potential numerical issues)
        mol_close = Molecule(
            Z=np.array([1, 1], dtype=np.int8),
            R=np.array([[0.0, 0.0, 0.0], [1e-6, 0.0, 0.0]], dtype=np.float32),
            E=0.0,
            spin=1,
            charge=0
        )
        
        formatter = DirFormatter(lmax=2)
        result = formatter(mol_close, comprehensive_binning_spec)
        
        text, cont_tensor, atom_tensor, node_boo = result
        
        # Should handle close atoms without NaN/Inf
        assert torch.all(torch.isfinite(node_boo))
        assert not torch.any(torch.isnan(node_boo))
        assert not torch.any(torch.isinf(node_boo)) 