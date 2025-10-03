import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, List
from transformers.modeling_outputs import CausalLMOutputWithPast

# from transformers.utils import TransformersKwargs
from transformers.models.llama.modeling_llama import KwargsForCausalLM
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from mmlm.models.continuous_model import ContinuousModelForCausalLM, L2NormLoss

from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from mmlm.utils.utils import to_number_dtype


def cosine_similarity_loss(x, y):
    return 1 - torch.nn.functional.cosine_similarity(x, y, dim=-1)


class PositionReadoutModel(ContinuousModelForCausalLM):
    """
    This model calculates forces using a readout head from the positions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.loss_name == "cos":
            self.force_loss_fn = cosine_similarity_loss
        else:
            self.force_loss_fn = L2NormLoss()

        self.energy_loss_fn = nn.L1Loss(reduction="none")

        if self.pre_readout_layer_norm:
            self.pre_readout_layer_norm = Qwen3RMSNorm(self.config.hidden_size)

        if not self.regress_forces:
            del self.lm_head_number

    def get_e_f_pred(self, hidden_states, pos_mask, input_ids):
        energy_pred = self.lm_head_energy(hidden_states)
        energy_pred = (energy_pred.squeeze(-1) * pos_mask).sum(dim=-1, keepdim=True)

        force_pred = None
        if self.regress_forces:
            force_pred = self.lm_head_number(hidden_states)

        return energy_pred, force_pred

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        attention_mask = self.get_prefix_causal_mask(
            input_ids["tokens"], attention_mask
        )

        if self.double_precision:
            input_ids = to_number_dtype(input_ids, torch.float64)
            attention_mask = to_number_dtype(attention_mask, torch.float64)

        if self.no_pos_embed:
            # Hacky way to turn off position embeddings
            position_ids = torch.zeros_like(input_ids["tokens"])

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        hs = hidden_states[:, -num_logits_to_keep:, :]

        if self.pre_readout_layer_norm:
            hs = self.pre_readout_layer_norm(hs)

        pos_mask = (input_ids["tokens"] >= self.min_pos_token) & (
            input_ids["tokens"] <= self.max_pos_token
        )
        energy_pred, force_pred = self.get_e_f_pred(
            hs, pos_mask=pos_mask, input_ids=input_ids
        )

        loss = None
        if labels is not None:

            # pos_mask = (labels["labels"] >= self.min_pos_token) & (
            #     labels["labels"] <= self.max_pos_token
            # )
            n_atoms = pos_mask.sum(dim=-1, keepdim=True)

            # Should be of shape (B, 1)
            # energy_pred = (energy_pred.squeeze(-1) * pos_mask).sum(dim=-1, keepdim=True)
            eng_true = labels["target_labels"].unsqueeze(-1)

            force_true = None
            if self.regress_forces:
                force_pred = force_pred[pos_mask]
                force_true = labels["force_labels"]

                if self.max_force_per_batch is not None:
                    force_true = force_true[: force_pred.shape[0]]

            if self.ft_normalize_batch:
                energy_num_items_in_batch, force_num_items_in_batch = kwargs.get(
                    "num_items_in_batch", (None, None)
                )
            else:
                energy_num_items_in_batch, force_num_items_in_batch = None, None

            if self.loss_name == "cos":
                force_pred = torch.nn.functional.normalize(force_pred, dim=-1)

            loss = self.continuous_loss(
                force_pred,
                force_true,
                energy_pred,
                eng_true,
                n_atoms=n_atoms,
                energy_num_items_in_batch=energy_num_items_in_batch,
                force_num_items_in_batch=force_num_items_in_batch,
            )

        if not return_dict:
            raise NotImplementedError(
                "This is no longer implemented for newer versions of code!"
            )

        return CausalLMOutputWithPast(
            loss=loss,
            # logits=torch.cat([eng_pred, force_pred], dim=1),
            logits=self.process_continuous_output(
                energy_pred, force_pred, eng_true, force_true
            ),
            # logits=force_pred.unsqueeze(0),
            past_key_values=None,  # outputs.asdict().get("past_key_values", None),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class PCQPositionReadoutModel(PositionReadoutModel):
    """
    This model calculates forces using a readout head from the positions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.constant_(
            self.lm_head_energy.mlp.down_proj.bias, self.config.energy_mean
        )

    def get_e_f_pred(self, hidden_states, pos_mask, input_ids):

        average_hidden_states = (hidden_states * pos_mask.unsqueeze(-1)).sum(
            dim=1, keepdim=False
        ) / pos_mask.sum(dim=1, keepdim=True)
        energy_pred = self.lm_head_energy(average_hidden_states)

        return energy_pred, None

    def continuous_loss(
        self,
        normalized_force_pred,
        force_true,
        normalized_eng_pred,
        eng_true,
        n_atoms,
        energy_num_items_in_batch=None,
        force_num_items_in_batch=None,
    ):
        normalized_eng_true = (
            eng_true  # (eng_true - self.energy_mean) / self.energy_std
        )

        eng_loss = self.energy_loss_fn(normalized_eng_pred, normalized_eng_true)

        if energy_num_items_in_batch is not None:
            eng_loss = eng_loss.sum() / energy_num_items_in_batch
        else:
            eng_loss = eng_loss.mean()

        return eng_loss * self.loss_weights["target"]

    def process_continuous_output(self, energy_pred, force_pred, eng_true, force_true):
        energy_pred = energy_pred

        # Create padded_forces on the same device as energy_pred to avoid DDP issues
        padded_forces = torch.tensor(
            [0.0], device=energy_pred.device, dtype=energy_pred.dtype
        )

        return energy_pred, padded_forces
