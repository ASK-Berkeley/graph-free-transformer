import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, List
from transformers.modeling_outputs import CausalLMOutputWithPast

# from transformers.utils import TransformersKwargs
from transformers.models.llama.modeling_llama import KwargsForCausalLM
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack

from mmlm.models.continuous_model import ContinuousModelForCausalLM, L2NormLoss


class PrefixReadoutModel(ContinuousModelForCausalLM):
    """
    This model calculates forces using a readout head from the positions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.force_loss_fn = L2NormLoss()
        self.energy_loss_fn = nn.L1Loss()

    def compute_loss(
        self, force_pred, force_true, energy_pred, eng_true, num_items_in_batch=None
    ):
        eng_loss = self.energy_loss_fn(energy_pred, eng_true)
        force_loss = self.force_loss_fn(force_pred, force_true)

        if num_items_in_batch is not None:
            eng_loss = eng_loss.sum() / num_items_in_batch
            force_loss = force_loss.sum() / num_items_in_batch
        else:
            eng_loss = eng_loss.mean()
            force_loss = force_loss.mean()

        return (
            eng_loss * self.loss_weights["target"]
            + force_loss * self.loss_weights["force"]
        )

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

        energy_pred = self.lm_head_energy(hs)
        force_pred = self.lm_head_number(hs)

        loss = None
        if labels is not None:

            shifted_labels = labels["labels"][:, 1:]
            shifted_true_numbers = labels["num_labels"][:, 1:]
            shifted_force_pred = force_pred[:, :-1, :]
            shifted_energy_pred = energy_pred[:, :-1, :]

            target_mask = (shifted_labels >= self.min_target_token) & (
                shifted_labels <= self.max_target_token
            )
            force_mask = (shifted_labels >= self.min_force_token) & (
                shifted_labels <= self.max_force_token
            )

            true_targets = shifted_true_numbers[target_mask][:, :1]
            true_forces = shifted_true_numbers[force_mask]

            pred_targets = shifted_energy_pred[target_mask]
            pred_forces = shifted_force_pred[force_mask]

            num_items_in_batch = kwargs.get("num_items_in_batch", None)

            loss = self.compute_loss(
                force_pred=pred_forces,
                force_true=true_forces,
                energy_pred=pred_targets,
                eng_true=true_targets,
                num_items_in_batch=num_items_in_batch,
            )

        if not return_dict:
            raise NotImplementedError(
                "This is no longer implemented for newer versions of code!"
            )

        return CausalLMOutputWithPast(
            loss=loss,
            # logits=torch.cat([eng_pred, force_pred], dim=1),
            logits=(pred_targets, pred_forces),
            # logits=force_pred.unsqueeze(0),
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
