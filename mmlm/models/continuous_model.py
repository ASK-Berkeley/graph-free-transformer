from transformers import (
    LlamaModel,
    LlamaConfig,
    PreTrainedModel,
    Qwen3Model,
)

from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutput,
)

from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import KwargsForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

# from transformers.utils import TransformersKwargs

from transformers.utils import logging as transformers_logging

from typing import Callable, List, Optional, Tuple, Union, Dict, Any

from torch import nn
import torch
from torch.nn import functional as F
from torch.distributions import Categorical, Normal

import logging

from mmlm.models.continuous_config import ContinuousConfigMixin

logger = transformers_logging.get_logger(__name__)


class L2NormLoss(nn.Module):
    """
    Currently this loss is intened to used with vectors.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert target.dim() == 2
        assert target.shape[1] != 1
        return torch.linalg.vector_norm(pred - target, ord=2, dim=-1)


class EmbedTokenNums(nn.Module):
    def __init__(
        self,
        embed_tokens: nn.Embedding,
        hidden_size: int,
        joint_embedding: bool = False,
        atom_embedding: bool = False,
        lmax: Optional[int] = None,
        num_atom_types: int = 100,
        mlp_embed: bool = False,
        concat_embeddings: bool = False,
        config: Optional[LlamaConfig] = None,
    ):

        super().__init__()
        self.embed_tokens = embed_tokens
        self.atom_embedding = atom_embedding
        self.lmax = lmax
        self.concat_embeddings = concat_embeddings

        (
            self.out_dim_token,
            self.out_dim_nums,
            self.out_dim_lmax,
            self.out_dim_atom,
        ) = self.get_out_dim(hidden_size)

        if self.concat_embeddings:
            self.embed_tokens = nn.Embedding(
                config.vocab_size, self.out_dim_token, padding_idx=config.pad_token_id
            )

        if self.concat_embeddings:
            assert (
                not mlp_embed
            ), "MLP embedding is not supported with concatenated embeddings"

        if atom_embedding:
            if config.multi_atom_embedding_dim is not None:
                self.embed_atoms = nn.Linear(
                    config.multi_atom_embedding_dim, self.out_dim_atom
                )
            else:
                self.embed_atoms = nn.Embedding(num_atom_types, self.out_dim_atom)

        if mlp_embed:
            self.embed_tokens_continuous = OutputMLP(
                input_size=3 if joint_embedding else 1,
                output_size=hidden_size,
                config=config,
            )
            if lmax is not None:
                self.embed_dir_embeddings = OutputMLP(
                    input_size=lmax + 1, output_size=hidden_size, config=config
                )
        else:
            self.embed_tokens_continuous = nn.Linear(
                3 if joint_embedding else 1, self.out_dim_nums, bias=False
            )
            if lmax is not None:
                self.embed_dir_embeddings = nn.Linear(lmax + 1, self.out_dim_lmax)

    def get_out_dim(self, hidden_size):
        if not self.concat_embeddings:
            return hidden_size, hidden_size, hidden_size, hidden_size

        out_types = 2
        if self.lmax is not None:
            out_types += 1
        if self.atom_embedding:
            out_types += 1

        out_dim_token = out_dim_nums = out_dim_lmax = out_dim_atom = (
            hidden_size // out_types
        )
        if out_dim_token * out_types != hidden_size:
            out_dim_nums += hidden_size % out_types

        return out_dim_token, out_dim_nums, out_dim_lmax, out_dim_atom

    def forward(self, input_ids):
        tokens = input_ids["tokens"]
        nums = input_ids["numbers"]

        res = [self.embed_tokens(tokens), self.embed_tokens_continuous(nums)]
        if self.atom_embedding:
            atoms = input_ids["atoms"]
            res.append(self.embed_atoms(atoms))
            if self.lmax is not None:
                dir_embeddings = input_ids["dir_embeddings"]
                res.append(self.embed_dir_embeddings(dir_embeddings))

        if self.concat_embeddings:
            return torch.cat(res, dim=-1)
        else:
            return torch.stack(res, dim=0).sum(dim=0)


class OutputMLP(Qwen3MLP):
    def __init__(self, config, output_size: int = 3, input_size: int = None):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        if input_size is None:
            input_size = self.hidden_size

        self.gate_proj = nn.Linear(
            input_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            input_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, output_size, bias=config.mlp_bias
        )
        self.act_fn = ACT2FN[config.hidden_act]


class MLPOutput(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        out_size: int,
        bias: bool = True,
        activation: str = "silu",
        residual: bool = False,
        llama_mlp: bool = False,
        config: Optional[LlamaConfig] = None,
    ):
        super().__init__()
        assert not (llama_mlp and residual), "Llama MLP cannot be residual"

        self.llama_mlp = llama_mlp

        if llama_mlp:
            self.mlp = OutputMLP(config, out_size)
        else:
            activation_fn = {
                "silu": nn.SiLU(),
                "relu": nn.ReLU(),
                "gelu": nn.GELU(),
                "tanh": nn.Tanh(),
            }[activation]

            self.mlp1 = nn.Linear(hidden_size, hidden_size * 2, bias=bias)
            self.mlp2 = nn.Linear(hidden_size * 2, out_size, bias=False)
            self.activation = activation_fn
            self.residual = residual
            if self.residual:
                self.mlp3 = nn.Linear(hidden_size, out_size, bias=False)

    def forward(self, x):
        if self.llama_mlp:
            out = self.mlp(x)
        else:
            out = self.mlp1(x)
            out = self.activation(out)
            out = self.mlp2(out)
            if self.residual:
                out = out + self.mlp3(x)
        return out


class ContinuousLlamaModel(LlamaModel):
    def __init__(
        self,
        config: ContinuousConfigMixin,
        joint_embedding: bool = False,
        atom_embedding: bool = False,
        lmax: Optional[int] = None,
        num_atom_types: int = 100,
        mlp_embed: bool = False,
        concat_embeddings: bool = False,
    ):
        super().__init__(config)

        self.embed_tokens = EmbedTokenNums(
            self.embed_tokens,
            config.hidden_size,
            joint_embedding=joint_embedding,
            atom_embedding=atom_embedding,
            lmax=lmax,
            num_atom_types=num_atom_types,
            mlp_embed=mlp_embed,
            concat_embeddings=concat_embeddings,
            config=config,
        )

        self.post_init()


class ContinuousQwen3Model(Qwen3Model):
    def __init__(
        self,
        config: ContinuousConfigMixin,
        joint_embedding: bool = False,
        atom_embedding: bool = False,
        lmax: Optional[int] = None,
        num_atom_types: int = 100,
        mlp_embed: bool = False,
        concat_embeddings: bool = False,
    ):
        super().__init__(config)

        self.embed_tokens = EmbedTokenNums(
            self.embed_tokens,
            config.hidden_size,
            joint_embedding=joint_embedding,
            atom_embedding=atom_embedding,
            lmax=lmax,
            num_atom_types=num_atom_types,
            mlp_embed=mlp_embed,
            concat_embeddings=concat_embeddings,
            config=config,
        )

        self.post_init()



class ContinuousModelForCausalLM(PreTrainedModel):
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer", "LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True

    def __init__(
        self,
        config: ContinuousConfigMixin,
        start_end_indices_by_token_type,
        loss_weights,
    ):
        super().__init__(config)

        self.config = config

        self.max_force_per_batch = config.max_force_per_batch
        self.force_pad_value = config.force_pad_value
        self.double_precision = config.double_precision
        self.no_pos_embed = config.no_pos_embed
        self.pre_readout_layer_norm = config.pre_readout_layer_norm
        self.ft_normalize_batch = config.ft_normalize_batch
        self.small_init_head = config.small_init_head
        self.regress_forces = config.regress_forces

        self.register_buffer(
            "energy_mean",
            torch.tensor(config.energy_mean, dtype=torch.float, requires_grad=False),
        )
        self.register_buffer(
            "energy_std",
            torch.tensor(config.energy_std, dtype=torch.float, requires_grad=False),
        )
        if self.regress_forces:
            self.register_buffer(
                "force_mean",
                torch.tensor(config.force_mean, dtype=torch.float, requires_grad=False),
            )
            self.register_buffer(
                "force_std",
                torch.tensor(config.force_std, dtype=torch.float, requires_grad=False),
            )

        self.lm_head_number = self.get_output_head(
            config,
            config.mlp_output_head,
            config.loss_name,
            config.joint_embedding,
            llama_mlp=config.llama_mlp,
            residual=config.residual,
        )

        self.energy_head = config.energy_head
        self.grad_accumulation_steps = config.grad_accumulation_steps
        self.batch_size = config.batch_size
        if config.energy_head:
            self.lm_head_energy = self.get_output_head(
                config,
                mlp_output_head=True,
                loss_name=None,
                joint_embedding=None,
                out_size=1,
                llama_mlp=config.llama_mlp,
                residual=config.residual,
            )

            self.mae_loss = torch.nn.L1Loss(reduction="none")

        self.number_weight = config.number_weight

        model_type_to_cls = {
            "llama": ContinuousLlamaModel,
            "qwen3": ContinuousQwen3Model,
        }

        self.model = model_type_to_cls[config.base_model_type](
            config,
            joint_embedding=config.joint_embedding,
            atom_embedding=config.atom_embedding,
            num_atom_types=config.num_atom_types,
            lmax=config.lmax,
            mlp_embed=config.mlp_embed,
            concat_embeddings=config.concat_embeddings,
        )

        self.joint_embedding = config.joint_embedding

        self.register_buffer(
            "min_std",
            torch.tensor(config.min_std, dtype=torch.float, requires_grad=False),
        )
        self.register_buffer(
            "vocab_weights",
            torch.zeros((config.vocab_size,), dtype=torch.float, requires_grad=False),
        )

        self.register_buffer(
            "label_to_token_type",
            torch.zeros((config.vocab_size,), dtype=torch.long, requires_grad=False),
        )

        self.register_buffer(
            "token_type_to_valid_mask",
            torch.zeros(
                (len(start_end_indices_by_token_type), config.vocab_size),
                dtype=torch.bool,
            ),
        )

        self.min_num_token = 1e6
        self.max_num_token = -1e6
        self.min_target_token, self.max_target_token = start_end_indices_by_token_type[
            "target"
        ]

        if "force" in start_end_indices_by_token_type:
            (
                self.min_force_token,
                self.max_force_token,
            ) = start_end_indices_by_token_type["force"]
        self.min_pos_token, self.max_pos_token = start_end_indices_by_token_type["pos"]

        self.pos_end_token_id = config.pos_end_token_id
        if self.pos_end_token_id is not None:
            logger.warning(
                f"Assuming [POS_END] is last token before forces / targets to form prefix causal mask!"
            )

        self.loss_weights = loss_weights
        self.start_end_indices_by_token_type_var = start_end_indices_by_token_type
        self.pos_token_type = None
        if not config.finetune:
            with torch.no_grad():
                for i, k in enumerate(start_end_indices_by_token_type.keys()):
                    start, end = start_end_indices_by_token_type[k]
                    self.vocab_weights[start : end + 1] = loss_weights[k]
                    self.label_to_token_type[start : end + 1] = i
                    self.token_type_to_valid_mask[i][start : end + 1] = True
                    if k == "pos":
                        self.pos_token_type = i
                    if start < self.min_num_token:
                        self.min_num_token = start
                    if end > self.max_num_token:
                        self.max_num_token = end
        logging.warning(
            "Assuming that all the numbers appear continuously in the vocab"
        )

        self.loss_name = config.loss_name

        if config.loss_name == "mse":
            self.loss_fn = torch.nn.MSELoss(reduction="none")
        elif config.loss_name == "mae":
            self.loss_fn = torch.nn.L1Loss(reduction="none")
        elif config.loss_name == "l2mae":
            self.loss_fn = L2NormLoss()
        elif config.loss_name == "xent":
            self.loss_fn = torch.nn.CrossEntropyLoss(
                reduction="none", ignore_index=-100
            )
        elif config.loss_name == "smooth_xent":
            self.gaussian_label_smoothing_sigma = config.gaussian_label_smoothing_sigma
            assert (
                config.gaussian_label_smoothing_sigma > 0.0
            ), "Gaussian label smoothing sigma must be greater than 0.0"
            self.loss_fn = self.gaussian_label_smoothed_loss
            self.register_buffer(
                "class_range",
                torch.arange(config.vocab_size).float(),
            )
        elif config.loss_name == "cos":
            # handled in subclass
            pass
        else:
            raise ValueError(f"Unknown loss function: {config.loss_name}")

        self.post_init()

        if self.small_init_head:
            for n, p in self.named_parameters():
                if "lm_head" in n and "bias" not in n:
                    logger.warning(
                        f"Initializing output head {n} with small weights {self.config.initializer_range / 5}!"
                    )
                    p.data.normal_(mean=0.0, std=self.config.initializer_range / 5)

    def get_output_head(
        self,
        config,
        mlp_output_head,
        loss_name,
        joint_embedding,
        out_size=None,
        llama_mlp: bool = False,
        residual: bool = False,
    ):
        if out_size is None:
            out_size = 1
            if loss_name in ["xent", "smooth_xent"]:
                out_size = config.vocab_size
            elif joint_embedding:
                out_size = 3

        if mlp_output_head:
            return MLPOutput(
                config.hidden_size,
                out_size,
                llama_mlp=llama_mlp,
                residual=residual,
                config=config,
            )
        else:
            return nn.Linear(config.hidden_size, out_size, bias=False)

    def gaussian_smooth_label(self, labels):
        # Labels: # (B, S, 1)
        gaussians = torch.exp(
            -0.5
            * ((self.class_range - labels) / self.gaussian_label_smoothing_sigma) ** 2
        )
        # Threshold small values to prevent numerical instability
        gaussians = torch.where(gaussians < 1e-3, 0, gaussians)
        return gaussians  # (B, S, V)

    def gaussian_label_smoothed_loss(self, logits, labels):
        token_types = self.label_to_token_type[labels]
        # Mask that tells for each token, which vocab elements are valid
        valid_mask = self.token_type_to_valid_mask[token_types]

        non_pad_mask = (labels != -100).unsqueeze(-1)  # (B, S, 1)
        labels_smoothed = self.gaussian_smooth_label(labels.unsqueeze(-1))

        # Mask logits
        masked_logits = torch.where(
            ~valid_mask,
            torch.finfo(logits.dtype).min,
            logits,
        )

        if self.joint_embedding:
            pos_mask = token_types == self.pos_token_type
            one_hot_labels = F.one_hot(
                labels, num_classes=self.config.vocab_size
            ).float()
            labels_smoothed = torch.where(
                pos_mask.unsqueeze(-1),
                one_hot_labels,
                labels_smoothed,
            )

        # First mask the labels to zero out invalid tokens
        masked_labels = labels_smoothed * valid_mask
        # Then add epsilon only to the valid tokens to prevent division by zero
        masked_labels = masked_labels + (valid_mask * 1e-8)
        masked_labels = masked_labels / masked_labels.sum(dim=-1, keepdim=True)

        masked_labels = torch.where(
            non_pad_mask,
            masked_labels,
            0,
        )
        masked_logits = torch.where(
            non_pad_mask,
            masked_logits,
            0,
        )

        return F.cross_entropy(masked_logits, masked_labels)

    def get_prefix_causal_mask(
        self, input_ids: torch.LongTensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns a causal mask that allows everything before the force / target tokens to all attend to eachother.
        In other words, all the atomic numbers, positions, spins, charges, etc. can attend to eachother.

        Args:
            input_ids: (B, S)
            attention_mask: (B, S)

        Returns:
            causal_mask: (B, H, S, S)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Find the position of [POS_END] token in each sequence
        pos_end_positions = (input_ids == self.pos_end_token_id).int().argmax(dim=1)

        # Create position indices for each sequence
        pos_indices = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        )  # (B, S)

        # Create a mask for positions before and including POS_END
        prefix_mask = pos_indices <= pos_end_positions.unsqueeze(1)  # (B, S)

        # Create the full attention mask
        # First create a causal mask for all positions
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))  # (S, S)

        # For positions before POS_END, allow full attention
        prefix_attention = torch.ones(seq_len, seq_len, device=device)  # (S, S)

        # Combine the masks based on whether positions are in prefix or not
        mask = torch.where(
            prefix_mask.unsqueeze(2) & prefix_mask.unsqueeze(1),  # (B, S, S)
            prefix_attention.unsqueeze(0),  # (1, S, S)
            causal_mask.unsqueeze(0),  # (1, S, S)
        )

        # Expand mask to include heads dimension
        mask = mask.unsqueeze(1).expand(-1, 1, -1, -1)

        # Apply attention mask to zero out padding tokens
        if attention_mask is not None:
            last_non_pad = attention_mask.sum(dim=1, dtype=torch.long) - 1
            # Create a mask for padding positions
            padding_mask = ~attention_mask.bool()  # (B, S)

            # Get the attention pattern for each sequence's last non-padding token
            batch_indices = torch.arange(batch_size, device=device)
            last_token_attention = mask[batch_indices, :, last_non_pad]

            mask = torch.where(
                padding_mask.unsqueeze(1).unsqueeze(2),  # (B, 1, S, 1)
                last_token_attention.unsqueeze(2),
                mask,
            )

        # Convert 0s to -inf and 1s to 0s
        mask = torch.where(mask == 0, torch.finfo(mask.dtype).min, 0)

        return mask

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

        if self.pos_end_token_id is not None:
            attention_mask = self.get_prefix_causal_mask(
                input_ids["tokens"], attention_mask
            )

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
        pred_number = self.lm_head_number(hs)
        if self.energy_head:
            energy_pred = self.lm_head_energy(hs)

        loss = None
        if labels is not None:

            true_numbers = labels["num_labels"]

            # shift labels to the right
            true_numbers = true_numbers[:, 1:]
            pred_number_filtered = pred_number[:, :-1, :]
            shifted_labels = labels["labels"][:, 1:]

            if self.loss_name in ["xent", "smooth_xent"]:
                true_numbers = shifted_labels.flatten()
                pred_number_filtered = pred_number_filtered.flatten(0, 1)
                token_type_weights = self.vocab_weights[shifted_labels].flatten()
            else:
                num_token_mask = (self.min_num_token <= shifted_labels) & (
                    shifted_labels <= self.max_num_token
                )

                # TODO: This part causes a graph break due to dynamic shape of indexing with num_token_mask
                true_numbers = true_numbers[num_token_mask]
                pred_number_filtered = pred_number_filtered[num_token_mask]
                shifted_labels_filtered = shifted_labels[num_token_mask]

                if self.energy_head:
                    eng_mask = (shifted_labels_filtered >= self.min_target_token) & (
                        shifted_labels_filtered <= self.max_target_token
                    )
                    force_mask = (shifted_labels_filtered >= self.min_force_token) & (
                        shifted_labels_filtered <= self.max_force_token
                    )

                    energy_pred = energy_pred[
                        :, :-1, :
                    ]  # Shift energy_pred to match shifted labels
                    energy_pred = energy_pred[num_token_mask]
                    eng_pred = energy_pred[eng_mask]
                    eng_true = true_numbers[eng_mask][:, 0].unsqueeze(-1)
                    force_pred = pred_number_filtered[force_mask][:, 0].reshape(-1, 3)
                    force_true = true_numbers[force_mask][:, 0].reshape(-1, 3)

                    force_loss = self.loss_fn(force_pred, force_true).mean()
                    eng_loss = self.mae_loss(eng_pred, eng_true).mean()
                    loss = (force_loss + eng_loss) / self.grad_accumulation_steps

                else:
                    token_type_weights = self.vocab_weights[shifted_labels_filtered]
                    if self.joint_embedding and self.loss_name != "l2mae":
                        token_type_weights = token_type_weights.unsqueeze(-1)

            if self.energy_head:
                loss_n = loss
            else:
                loss_n = (
                    self.loss_fn(pred_number_filtered, true_numbers).squeeze()
                    * token_type_weights
                )

                if self.loss_name in ["l2mae", "mae", "mse"]:
                    num_items_in_batch = num_token_mask.sum()
                else:
                    num_items_in_batch = kwargs.get("num_items_in_batch", None)
                if num_items_in_batch is not None:
                    loss_n = loss_n.sum() / num_items_in_batch
                else:
                    loss_n = loss_n.mean()
                
            loss = loss_n

        if not return_dict:
            raise NotImplementedError(
                "This is no longer implemented for newer versions of code!"
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=pred_number,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

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

        force_loss = 0
        if self.regress_forces:
            normalized_force_true = (force_true - self.force_mean) / self.force_std
            force_loss = self.force_loss_fn(
                normalized_force_pred, normalized_force_true
            )

        normalized_eng_true = (eng_true - self.energy_mean) / self.energy_std

        eng_loss = self.energy_loss_fn(
            normalized_eng_pred / n_atoms, normalized_eng_true / n_atoms
        )

        if energy_num_items_in_batch is not None:
            force_loss = (
                force_loss.sum() / force_num_items_in_batch
                if self.regress_forces
                else 0
            )
            eng_loss = eng_loss.sum() / energy_num_items_in_batch

            return (
                force_loss * self.loss_weights["force"]
                + eng_loss * self.loss_weights["target"]
            )
        else:
            force_loss = force_loss.mean() if self.regress_forces else 0
            eng_loss = eng_loss.mean()
            return (
                force_loss * self.loss_weights["force"]
                + eng_loss * self.loss_weights["target"]
            ) / self.grad_accumulation_steps

    def process_continuous_output(self, energy_pred, force_pred, eng_true, force_true):
        energy_pred = energy_pred * self.energy_std + self.energy_mean

        # Create padded_forces on the same device as energy_pred to avoid DDP issues
        padded_forces = torch.tensor(
            [0.0], device=energy_pred.device, dtype=energy_pred.dtype
        )
        if self.regress_forces:
            force_pred = force_pred * self.force_std + self.force_mean

            if self.max_force_per_batch is not None:
                padded_forces = torch.nn.ConstantPad2d(
                    (0, 0, 0, self.max_force_per_batch - force_pred.shape[0]),
                    self.force_pad_value,
                )(force_pred)
            else:
                padded_forces = force_pred

        return energy_pred, padded_forces

    def floating_point_ops(
        self,
        input_dict: Dict[str, Union[torch.Tensor, Any]],
        exclude_embeddings: bool = True,
    ) -> int:
        input_dict["input_ids"] = input_dict["input_ids"]["tokens"]
        return (
            6
            * self.estimate_tokens(input_dict)
            * self.num_parameters(exclude_embeddings=exclude_embeddings)
        )
