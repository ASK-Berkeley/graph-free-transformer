from transformers import LlamaConfig, PretrainedConfig, Qwen3Config, DebertaV2Config
from typing import Optional
import logging


class ContinuousConfigMixin:
    """Mixin class that adds continuous-specific configuration parameters."""

    def __init__(
        self,
        loss_name="mse",
        number_weight: float = 1.0,
        min_std=0.002,
        joint_embedding: bool = False,
        pos_end_token_id: Optional[int] = None,
        gaussian_label_smoothing_sigma: float = 0.8,
        mlp_output_head: bool = False,
        energy_head: bool = False,
        grad_accumulation_steps: int = 1,
        batch_size: int = 1,
        energy_mean: float = 0.0,
        energy_std: float = 1.0,
        force_mean: float = 0.0,
        force_std: float = 1.0,
        finetune: bool = False,
        max_force_per_batch: Optional[int] = None,
        force_pad_value: int = -200,
        atom_embedding: bool = False,
        num_atom_types: int = 100,
        double_precision: bool = False,
        no_pos_embed: bool = False,
        ft_normalize_batch: bool = False,
        pre_readout_layer_norm: bool = False,
        llama_mlp: bool = False,
        residual: bool = False,
        small_init_head: bool = False,
        lmax: Optional[int] = None,
        mlp_embed: bool = False,
        concat_embeddings: bool = False,
        base_model_type: str = "llama",
        multi_atom_embedding_dim: Optional[int] = None,
        regress_forces: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Add continuous-specific attributes
        self.loss_name = loss_name
        self.number_weight = number_weight
        self.min_std = min_std
        self.joint_embedding = joint_embedding
        self.pos_end_token_id = pos_end_token_id
        self.gaussian_label_smoothing_sigma = gaussian_label_smoothing_sigma
        self.mlp_output_head = mlp_output_head
        self.energy_head = energy_head
        self.grad_accumulation_steps = grad_accumulation_steps
        self.batch_size = batch_size
        self.energy_mean = energy_mean
        self.energy_std = energy_std
        self.force_mean = force_mean
        self.force_std = force_std
        self.finetune = finetune
        self.max_force_per_batch = max_force_per_batch
        self.force_pad_value = force_pad_value
        self.atom_embedding = atom_embedding
        self.num_atom_types = num_atom_types
        self.double_precision = double_precision
        self.no_pos_embed = no_pos_embed
        self.ft_normalize_batch = ft_normalize_batch
        self.pre_readout_layer_norm = pre_readout_layer_norm
        self.llama_mlp = llama_mlp
        self.residual = residual
        self.small_init_head = small_init_head
        self.lmax = lmax
        self.mlp_embed = mlp_embed
        self.concat_embeddings = concat_embeddings
        self.base_model_type = base_model_type
        self.multi_atom_embedding_dim = multi_atom_embedding_dim
        self.regress_forces = regress_forces


class ContinuousLlamaConfig(ContinuousConfigMixin, LlamaConfig):
    def __init__(self, **kwargs):
        super().__init__(base_model_type="llama", **kwargs)


class ContinuousQwen3Config(ContinuousConfigMixin, Qwen3Config):
    def __init__(self, **kwargs):
        super().__init__(base_model_type="qwen3", **kwargs)
