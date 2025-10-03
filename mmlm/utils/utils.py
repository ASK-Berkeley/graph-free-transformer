import torch

from transformers import TrainerCallback
from mmlm.utils.collators import (
    DataCollatorForContinuousMoleculeLanguageModeling,
)
from mmlm.datasets_v2.core.binner import BinningSpec

import logging
import math
from pathlib import Path

import numpy as np


train_amount_to_valid_layers_1B = {
    "head": ["lm_head"],
    "top": ["lm_head", "model.layers.22", "model.layers.21", "model.layers.20"],
}


def batch_to_device(batch, device):
    for k, v in batch.items():
        if type(v) == torch.Tensor:
            batch[k] = v.to(device)
        elif type(v) == dict:
            batch[k] = batch_to_device(v, device)
    return batch


def batch_to_type(batch, dtype):
    for k, v in batch.items():
        if type(v) == torch.Tensor and v.dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ]:
            batch[k] = v.to(dtype)
        elif type(v) == dict:
            batch[k] = batch_to_type(v, dtype)
    return batch


def train_layer(layer_name, train_layer_subnames):
    for train_layer_subname in train_layer_subnames:
        if train_layer_subname in layer_name:
            return True
    return False


def freeze_model(model, train_amount):
    train_layers = train_amount_to_valid_layers_1B[train_amount]
    total_params = 0
    train_params = 0
    for n, p in model.named_parameters():
        if train_layer(n, train_layers):
            p.requires_grad_(True)
            train_params += p.numel()
        else:
            p.requires_grad_(False)
        total_params += p.numel()
    logging.info(
        f"Training {train_params} params ({train_params / total_params * 100:.2f}%) of {total_params} params"
    )


def to_number_dtype(x, dtype=torch.float64):
    if isinstance(x, torch.Tensor) and x.dtype == torch.float32:
        return x.to(dtype)
    elif isinstance(x, dict):
        return {k: to_number_dtype(v) for k, v in x.items()}
    return x


class KBinsDiscretizer:
    def __init__(self, bin_edges):
        self.bin_edges = [bin_edges]

    def transform(self, x):
        bin_edges = self.bin_edges

        x = np.array(x)
        is_num = False
        if len(x.shape) == 0:
            x = np.array([[x]])
            is_num = True
        if len(x.shape) == 1:
            x = np.expand_dims(x, 1)

        for jj in range(x.shape[1]):
            x[:, jj] = np.searchsorted(bin_edges[jj][1:-1], x[:, jj], side="right")

        return int(x[0][0]) if is_num else x.astype(int)

    def inverse_transform(self, x):
        bin_centers = (self.bin_edges[0][1:] + self.bin_edges[0][:-1]) / 2
        return bin_centers[x]

    # Should be of shape (B, n_bins)
    def weighted_inverse_transform(self, x):
        bin_centers = (self.bin_edges[0][1:] + self.bin_edges[0][:-1]) / 2
        return np.sum(bin_centers * x, axis=-1)


class NaNLossCallback(TrainerCallback):
    def __init__(self, *args, cutoff=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.cutoff = cutoff

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            loss = logs["loss"]
            if math.isnan(loss) or loss > self.cutoff:
                logging.warning("NaN loss detected. Stopping training...")
                control.should_training_stop = True


def get_n_actual_bins(args):
    if args.dataset.pipeline_v2:
        bin_spec = BinningSpec.load(args.dataset.train_dataset.cfg.bin_spec_path)

        n_actual_bins = {}
        for k in args.dataset.n_bins:
            n_actual_bins[k] = len(bin_spec.field_bins[k]) - 1
        return n_actual_bins

    n_actual_bins = {}
    if type(args.dataset.bins_method) == str:
        args.dataset.bins_method = {
            k: args.dataset.bins_method for k in args.dataset.n_bins
        }

    parent_dir = Path(args.dataset.train_path).parent
    bins_path = args.dataset.bins_path if args.dataset.bins_path is not None else ""
    paths_to_try = [
        Path(args.dataset.train_path),
        parent_dir,
        parent_dir.parent,
        Path(bins_path),
    ]
    for k in args.dataset.n_bins:
        k_path = k
        if k == "target" and args.dataset.per_atom_target:
            k_path = "per_atom_target"
        if k == "force" and args.dataset.prior_path_train is not None:
            k_path = "delta_force"
        method_name = (
            ""
            if args.dataset.bins_method[k] == "quantile"
            else f"_{args.dataset.bins_method[k]}"
        )
        bins_path = (
            paths_to_try[0] / f"bins_new_{args.dataset.n_bins[k]}{method_name}.npy"
        )
        for bp in paths_to_try:
            bp = bp / f"bins_new_{args.dataset.n_bins[k]}{method_name}.npy"
            if bp.exists():
                bins_path = bp
                break
        loaded_bins = np.load(bins_path, allow_pickle=True).item()
        if (
            k == "target"
            and args.dataset.get("target_name", None) is not None
            and "target" not in loaded_bins
        ):
            k_path = args.dataset.target_name
        n_actual_bins[k] = len(loaded_bins[k_path]["bins"]) - 1

    return n_actual_bins


def get_special_tokens(vocab_path):
    special_tokens = []
    with open(vocab_path, "r") as f:
        for line in f:
            if line.strip() not in [
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "0",
                ".",
                "-",
                "+",
            ]:
                special_tokens.append(line.strip())
    return special_tokens




def hidden_act_to_str_func(hidden_act):
    if hidden_act == "sin":
        return torch.sin
    elif hidden_act == "cos":
        return torch.cos
    else:
        return hidden_act


def get_start_end_indices_by_token_type(args, tokenizer, n_actual_bins):
    start_end_indices_by_token_type = {
        "atomic_numbers": (
            tokenizer.convert_tokens_to_ids("a_1:"),
            tokenizer.convert_tokens_to_ids(f"a_{args.dataset.n_atom_types}:"),
        ),
        "pos": (
            (
                tokenizer.convert_tokens_to_ids("<NUM_0>")
                if not args.training.finetune
                else tokenizer.convert_tokens_to_ids("<NUM>")
            ),
            (
                tokenizer.convert_tokens_to_ids(f"<NUM_{n_actual_bins['pos'] - 1}>")
                if not args.training.finetune
                else tokenizer.convert_tokens_to_ids(f"<NUM>")
            ),
        ),
    }

    if args.dataset.joint_embed_atoms:
        start_end_indices_by_token_type.pop("atomic_numbers")

    if "target" in args.dataset.n_bins:
        start_end_indices_by_token_type["target"] = (
            tokenizer.convert_tokens_to_ids("<NUM_target_0>"),
            tokenizer.convert_tokens_to_ids(
                f"<NUM_target_{n_actual_bins['target'] - 1}>"
            ),
        )
    if "force" in args.dataset.n_bins:
        start_end_indices_by_token_type["force"] = (
            tokenizer.convert_tokens_to_ids("<NUM_force_0>"),
            tokenizer.convert_tokens_to_ids(
                f"<NUM_force_{n_actual_bins['force'] - 1}>"
            ),
        )

   
    if "cell" in args.dataset.n_bins:
        start_end_indices_by_token_type["cell"] = (
            tokenizer.convert_tokens_to_ids("<NUM_cell_0>"),
            tokenizer.convert_tokens_to_ids(f"<NUM_cell_{n_actual_bins['cell'] - 1}>"),
        )
    if "stress" in args.dataset.n_bins:
        start_end_indices_by_token_type["stress"] = (
            tokenizer.convert_tokens_to_ids("<NUM_stress_0>"),
            tokenizer.convert_tokens_to_ids(
                f"<NUM_stress_{n_actual_bins['stress'] - 1}>"
            ),
        )
    return start_end_indices_by_token_type


def get_collator(cfg, tokenizer):

    
        
    max_force_per_batch = None
    if cfg.training.max_force_per_item is not None:
        max_force_per_batch = cfg.training.max_force_per_item * max(
            cfg.training.batch_size, cfg.training.eval_batch_size
        )

    multi_atom_embedding_dim = cfg.model.get("multi_atom_embedding_dim", None)
    if multi_atom_embedding_dim is not None and cfg.dataset.get(
        "add_edge_features", False
    ):
        multi_atom_embedding_dim = (
            multi_atom_embedding_dim
            + cfg.dataset.max_bonds_per_atom * cfg.dataset.bond_features_dim
        )

    data_collator = DataCollatorForContinuousMoleculeLanguageModeling(
        tokenizer=tokenizer,
        preprocessed=cfg.dataset.get("preprocessed", False),
        mlm=False,
        joint_embedding=cfg.dataset.joint_embedding,
        data_bf16=cfg.training.data_bf16,
        validation_mode=cfg.training.get("get_val_errors", False),
        finetune=cfg.training.get("finetune", False),
        max_force_per_batch=max_force_per_batch,
        force_pad_value=cfg.training.force_pad_value,
        atom_embedding=cfg.dataset.joint_embed_atoms,
        lmax=cfg.dataset.lmax,
        multi_atom_embedding_dim=multi_atom_embedding_dim,
    )

    return data_collator


