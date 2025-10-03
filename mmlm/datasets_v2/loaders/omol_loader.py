"""
Loader for the OMOL dataset.
"""

from typing import Optional
import numpy as np
import torch
import logging

from mmlm.datasets_v2.lmdb_dataset import PAseDBDataset
from .base_loader import BaseLoader
from mmlm.datasets_v2.core.molecule import Molecule
from mmlm.datasets_v2.core.binner import BinningSpec


class OmolLoader(BaseLoader):
    """
    Loads data from the OMOL dataset and converts it into Molecule objects.

    This loader handles the OMOL-specific logic, including:
    - Reading from the underlying LMDB dataset.
    - Calculating reference energy based on OMOL's elemental references.
    - Optionally centering the molecule's positions.
    """

    OMOL_ELEMENTAL_REFERENCES = np.array(
        [
            -1.156629225040026e-15,
            -16.001316695716987,
            -82.01087430838089,
            -201.82956857988097,
            -392.37382000025474,
            -678.7577802660586,
            -1037.0951083942327,
            -1490.3891424843594,
            -2047.6980007878883,
            -2717.4000859159146,
            -3511.9177958942937,
            -4412.34353756339,
            -5442.048852490966,
            -6589.238285511481,
            -7880.842380936224,
            -9290.366602061926,
            -10834.813633824462,
            -12522.319387833275,
            -14357.418706857397,
            -16323.294755148576,
            -18435.01990024982,
            -20696.697448280793,
            -23112.143048514245,
            -25682.83219323889,
            -28416.863013932467,
            -31314.828666123234,
            -34381.61625929281,
            -37622.0970301371,
            -41038.77074996302,
            -44633.87178654121,
            -48413.89305602666,
            -52371.64723336532,
            -56519.32612658109,
            -60840.896102839535,
            -65348.22274342182,
            -70042.31301581895,
            -74932.45491967947,
            -653.3100819462285,
            -831.4256441321643,
            -1038.3067074649807,
            -1270.5802453884555,
            -1546.2069619314702,
            -1844.151869397674,
            -2185.315025224292,
            -2572.1188857512716,
            -2998.6928887542217,
            -3471.81081638118,
            -3993.2282048969632,
            -4560.650406219408,
            -5163.202864090282,
            -5822.573487382639,
            -6540.2964462130585,
            -7296.665490756577,
            -8099.4071751019965,
            -8964.710394463762,
            -546.2179067379832,
            -689.6398798417333,
            -850.4805849513325,
            -12917.08295504127,
            -14058.981210064643,
            -15267.41531050162,
            -16544.445729634244,
            -17894.054958218647,
            -19321.404884189622,
            -20822.18831205945,
            -22422.829984851378,
            -24073.462031913972,
            -25787.271913480094,
            -27608.607061532348,
            -29517.834852414315,
            -31519.198453963163,
            -33610.154761607024,
            -1293.3349595217703,
            -1542.107003487292,
            -1813.074616850505,
            -2116.0691118518926,
            -2451.555058191304,
            -2827.017838069772,
            -3237.345589780805,
            -3682.7850194399866,
            -4168.118509910375,
            -4683.015643495902,
            -5238.7226964026495,
            -5832.098572286902,
            -6469.07296,
            -7140.86455,
            -7854.60638,
            -0.0,
            -1.6479188753456857,
            -4.207893480442443,
            -6.239290333069828,
            -5.835083816475604,
            -5.524155726787739,
            -4.817312409713694,
            -3.57567303141045,
            -2.943452582378457,
            -4.000608087317199,
            -3.212923593441274,
            -2.031396852627386,
            -1.378447864282869,
        ]
    )

    def __init__(
        self,
        path: str,
        center: bool = True,
        indices: Optional[np.ndarray] = None,
        force_prior_path: Optional[str] = None,
        per_atom_target: bool = False,
        first_force_only: bool = False,
    ):
        """
        Initialize the OmolLoader.

        Args:
            path: Path to the OMOL LMDB dataset directory.
            center: Whether to center the atomic positions.
            indices: Optional subset of indices to load.
            force_prior_path: Optional path to a .npy file with force priors.
        """
        super().__init__()
        self.dataset = PAseDBDataset(
            {
                "src": path,
                "a2g_args": {
                    "r_energy": True,
                    "r_stress": True,
                    "r_forces": True,
                    "r_data_keys": ["spin", "charge"],
                },
            }
        )
        self.center = center
        self.indices = indices if indices is not None else np.arange(len(self.dataset))
        self.force_prior = None
        self.per_atom_target = per_atom_target
        if force_prior_path:
            logging.info(f"Loading force prior from {force_prior_path}")
            self.force_prior = np.load(force_prior_path, allow_pickle=True).item()[
                "forces"
            ]

        if not self.center:
            logging.warning("Not centering positions.")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Molecule:
        """Fetch a single molecule data point and convert it to a Molecule object."""
        dataset_idx = self.indices[idx]
        data = self.dataset[dataset_idx]

        # Calculate reference energy
        reference_energy = self.OMOL_ELEMENTAL_REFERENCES[data.atomic_numbers].sum()

        if self.per_atom_target:
            reference_energy = reference_energy / data.pos.shape[0]
        
        # Center positions if configured
        positions = data.pos - data.pos.mean(dim=0) if self.center else data.pos

        forces = data.forces.numpy()
        if self.force_prior is not None:
            forces = forces - self.force_prior[dataset_idx]

        return Molecule(
            Z=data.atomic_numbers.numpy().astype(np.int8),
            R=positions.numpy().astype(np.float32),
            F=forces.astype(np.float32),
            E=data.energy.item() - reference_energy,
            spin=data.spin.item(),
            charge=data.charge.item(),
            cell=(
                data.cell.numpy().astype(np.float32)[0]
                if hasattr(data, "cell")
                else None
            ),
            stress=(
                data.stress.numpy().astype(np.float32)
                if hasattr(data, "stress")
                else None
            ),
        )
