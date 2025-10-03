from mmlm.models.pos_readout_model import PositionReadoutModel
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, List
from transformers.modeling_outputs import CausalLMOutputWithPast

# from transformers.utils import TransformersKwargs
from transformers.models.llama.modeling_llama import KwargsForCausalLM
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack


class GradientModel(PositionReadoutModel):
    """
    This model calculates forces using gradients of the energy.
    """

    def get_e_f_pred(self, hidden_states, pos_mask, input_ids):
        raise NotImplementedError("CHeck whether we want sum or not here!")
        energy_pred = self.lm_head_energy(hidden_states)
        energy_pred = (energy_pred.squeeze(-1) * pos_mask).sum(dim=-1, keepdim=True)
        force_pred = -torch.autograd.grad(
            energy_pred.sum(),
            input_ids["numbers"],
            # grad_outputs=torch.ones_like(energy_pred),
            create_graph=True,
        )[0]
        return energy_pred, force_pred

    def forward(self, input_ids=None, labels=None, *args, **kwargs):
        with torch.enable_grad():
            input_ids["numbers"].requires_grad_(True)
            return super().forward(input_ids=input_ids, labels=labels, *args, **kwargs)
