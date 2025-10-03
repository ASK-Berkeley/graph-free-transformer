import torch
from mmlm.models.pos_readout_model import PositionReadoutModel


from mmlm.utils.utils import to_number_dtype


def cosine_similarity_loss(x, y):
    return 1 - torch.nn.functional.cosine_similarity(x, y, dim=-1)


class DirReadoutModel(PositionReadoutModel):
    """
    This model calculates forces using a readout head from the positions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lm_head_number = self.get_output_head(
            self.config,
            mlp_output_head=True,
            loss_name=None,
            joint_embedding=None,
            out_size=1,
        )
        self.lm_head_dir = self.get_output_head(
            self.config,
            mlp_output_head=True,
            loss_name=None,
            joint_embedding=None,
            out_size=3,
        )

    def get_e_f_pred(self, hidden_states):
        energy_pred = self.lm_head_energy(hidden_states)
        dir_pred = self.lm_head_dir(hidden_states)
        magnitude_pred = self.lm_head_number(hidden_states)

        dir_pred = torch.nn.functional.normalize(dir_pred, dim=-1)
        force_pred = dir_pred * magnitude_pred

        return energy_pred, force_pred
