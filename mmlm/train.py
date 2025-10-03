from transformers import TrainingArguments, Trainer, AutoModel, AutoTokenizer
from safetensors.torch import load_file

from mmlm.utils.compute_metrics_utils import (
    compute_metrics,
    preprocess_logits_for_metrics,
    compute_metrics_continuous,
)

import torch
from torch.utils.data import Subset

import wandb

from mmlm.utils.utils import (
    hidden_act_to_str_func,
    get_collator,
    NaNLossCallback,
    get_n_actual_bins,
    get_start_end_indices_by_token_type,
    freeze_model,
)

from mmlm.utils.model_utils import get_config_and_class
from mmlm.custom_tokenizer import get_tokenizer
from mmlm.trainers.molecule_trainer import MoleculeTrainer
from mmlm.datasets_v2.core.binner import BinningSpec


import os
import numpy as np
import random
import logging
from pathlib import Path
import glob


import hydra
from omegaconf import DictConfig, OmegaConf


def to_device(x, device):
    for k in x:
        if type(x[k]) == torch.Tensor:
            x[k] = x[k].to(device)
        elif type(x[k]) == dict:
            x[k] = to_device(x[k], device)
    return x


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _validate_v2_config_consistency(cfg: DictConfig):
    """
    Ensures that legacy `dataset` flags are consistent with the new v2 config.
    This acts as a guard rail to prevent silent bugs when using the v2 pipeline.
    """
    logging.info(
        "Validating consistency between legacy config and datasets_v2 config..."
    )

    # Load the binning spec to check against legacy binning flags
    bin_spec = BinningSpec.load(cfg.dataset.train_dataset.cfg.bin_spec_path)

    # --- Assertions based on your rules ---
    assert (
        cfg.dataset.continuous
    ), "v2 pipeline is continuous only. `dataset.continuous` must be True."
    assert (
        cfg.dataset.joint_embedding
    ), "v2 pipeline requires joint embedding. `dataset.joint_embedding` must be True."
    assert (
        cfg.dataset.joint_embedding_force
    ), "v2 pipeline requires joint embedding for force. `dataset.joint_embedding_force` must be True."
    assert not cfg.dataset.get(
        "token_enumeration"
    ), "v2 pipeline does not use token enumeration. `dataset.token_enumeration` must be False."
    assert not cfg.dataset.get(
        "fractional_coords"
    ), "v2 pipeline does not support fractional coords. `dataset.fractional_coords` must be False."

    # Assertions for flags that should not be set
    assert (
        cfg.dataset.get("prior_path_train") is None
    ), "`dataset.prior_path_train` is not supported in v2."
    assert (
        cfg.dataset.get("target_name") is None
    ), "`dataset.target_name` is not supported in v2."

    # Injected flags from model config
    assert cfg.model.causal, "`model.causal` must be True for the v2 pipeline."

    # Check for path consistency
    assert cfg.dataset.train_path == cfg.dataset.train_dataset.cfg.loader.path, (
        f"Path mismatch: `dataset.train_path` ({cfg.dataset.train_path}) != "
        f"v2 loader path ({cfg.dataset.train_dataset.cfg.loader.path})"
    )

    if cfg.dataset.get("val_path"):
        assert cfg.dataset.val_path == cfg.dataset.val_dataset.cfg.loader.path, (
            f"Path mismatch: `dataset.val_path` ({cfg.dataset.val_path}) != "
            f"v2 val loader path ({cfg.dataset.val_dataset.cfg.loader.path})"
        )

    # Check for formatter consistency based on atom embedding
    formatter_target = cfg.dataset.train_dataset.cfg.formatter._target_
    if cfg.dataset.joint_embed_atoms:
        assert (
            "AtomFormatter" in formatter_target
        ), "Mismatch: `dataset.joint_embed_atoms` is True but v2 formatter is not AtomFormatter."
    else:
        assert (
            "StandardFormatter" in formatter_target
        ), "Mismatch: `dataset.joint_embed_atoms` is False but v2 formatter is not StandardFormatter."

    # Check for augmentation consistency
    transforms_list = [t._target_ for t in cfg.dataset.train_dataset.cfg.transforms]
    has_permutation = (
        "mmlm.datasets_v2.core.transforms.PermutationTransform" in transforms_list
    )
    assert cfg.dataset.permutation_augmentation == has_permutation, (
        f"Mismatch: `dataset.permutation_augmentation` is {cfg.dataset.permutation_augmentation} but "
        f"PermutationTransform is {'present' if has_permutation else 'absent'} in v2 config."
    )

    # Check for binning consistency
    for key, value in cfg.dataset.n_bins.items():
        assert (
            bin_spec.n_bins[key] == value
        ), f"Bin mismatch for '{key}': legacy config has {value}, but bin spec has {bin_spec.n_bins[key]}."


def get_model(args, tokenizer, start_end_indices_by_token_type):

    model_config_cls, model_cls = get_config_and_class(args.model.model_type)

    if "grad" in args.model.model_type:
        assert (
            args.training.attn_implementation == "eager"
        ), "Gradient model requires eager attention implementation for double backwards!"

    energy_mean, energy_std, force_mean, force_std = 0, 1, 0, 1
    if args.dataset.get("norm_stats_path", None) is not None:
        stats = np.load(args.dataset.norm_stats_path, allow_pickle=True).item()
        energy_mean = stats["energy_mean"]
        energy_std = stats["energy_std"]
        force_mean = stats["force_mean"]
        force_std = stats["force_std"]

    
    
    max_force_per_batch = None
    if args.training.max_force_per_item is not None:
        max_force_per_batch = args.training.max_force_per_item * max(
            args.training.batch_size, args.training.eval_batch_size
        )

    multi_atom_embedding_dim = args.model.get("multi_atom_embedding_dim", None)
    if multi_atom_embedding_dim is not None and args.dataset.get(
        "add_edge_features", False
    ):
        multi_atom_embedding_dim = (
            multi_atom_embedding_dim
            + args.dataset.max_bonds_per_atom * args.dataset.bond_features_dim
        )

    continuous_config = model_config_cls(
        # base_config=config,
        vocab_size=tokenizer.vocab_size,  # Standard vocabulary size for BERT
        hidden_size=args.model.hidden_size,  # Size of hidden layers
        num_hidden_layers=args.model.num_layers,  # Number of transformer layers
        num_attention_heads=args.model.num_attention_heads,  # Number of attention heads
        num_key_value_heads=args.model.get(
            "num_key_value_heads", args.model.num_attention_heads
        ),  # Number of key-value heads (for GQA)
        intermediate_size=args.model.intermediate_size,  # Size of the intermediate layer
        max_position_embeddings=args.dataset.max_seq_length,  # Max sequence length
        # Qwen3-specific parameters
        head_dim=args.model.get("head_dim", None),  # Head dimension
        rms_norm_eps=args.model.get("rms_norm_eps", 1e-6),  # RMS norm epsilon
        rope_theta=args.model.get("rope_theta", 10000.0),  # RoPE theta
        rope_scaling=args.model.get("rope_scaling", None),  # RoPE scaling
        sliding_window=args.model.get("sliding_window", None),  # Sliding window
        use_sliding_window=args.model.get(
            "use_sliding_window", False
        ),  # Use sliding window
        max_window_layers=args.model.get(
            "max_window_layers", None
        ),  # Max window layers
        tie_word_embeddings=args.model.get(
            "tie_word_embeddings", True
        ),  # Tie word embeddings
        hidden_dropout_prob=args.model.hidden_dropout_prob,
        attention_probs_dropout_prob=args.model.attention_dropout_prob,
        pad_token_id=tokenizer.pad_token_id,
        initializer_range=args.model.init_range,
        hidden_act=hidden_act_to_str_func(args.model.hidden_act),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        attn_implementation=args.training.attn_implementation,
        target_cls_token=tokenizer.convert_tokens_to_ids("[TARGET]"),
        eng_sum=args.model.eng_sum,
        use_cache=False,
        mlp_bias=args.model.mlp_bias,
        attention_bias=args.model.attention_bias,
        loss_name=args.model.get("loss_name", "mse"),
        joint_embedding=args.dataset.joint_embedding,
        gaussian_label_smoothing_sigma=args.model.gaussian_label_smoothing_sigma,
        pos_end_token_id=(
            tokenizer.convert_tokens_to_ids("[POS_END]")
            if args.model.get("prefix_causal_mask", False)
            else None
        ),
        mlp_output_head=args.model.get("mlp_output_head", False),
        energy_head=args.model.get("energy_head", False),
        grad_accumulation_steps=(
            args.training.gradient_accumulation_steps if not args.debug else 1
        ),
        energy_mean=energy_mean,
        energy_std=energy_std,
        force_mean=force_mean,
        force_std=force_std,
        finetune=args.training.get("finetune", False),
        max_force_per_batch=max_force_per_batch,
        force_pad_value=args.training.force_pad_value,
        atom_embedding=args.dataset.joint_embed_atoms,
        num_atom_types=args.dataset.n_atom_types,
        double_precision=args.model.get("double_precision", False),
        no_pos_embed=args.model.get("no_pos_embed", False),
        ft_normalize_batch=args.training.get("ft_normalize_batch", False),
        pre_readout_layer_norm=args.model.get("pre_readout_layer_norm", False),
        llama_mlp=args.model.get("llama_mlp", False),
        residual=args.model.get("residual", False),
        small_init_head=args.model.get("small_init_head", False),
        lmax=args.dataset.get("lmax", None),
        mlp_embed=args.model.get("mlp_embed", False),
        concat_embeddings=args.model.get("concat_embeddings", False),
        multi_atom_embedding_dim=multi_atom_embedding_dim,
        regress_forces=args.model.get("regress_forces", True),
    )
    model = model_cls(
        config=continuous_config,
        start_end_indices_by_token_type=start_end_indices_by_token_type,
        loss_weights=args.model.loss_weights,
    )

    # Calculate and log model size
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total model parameters: {total_params:,}")
    
    if (
        args.training.resume_from_checkpoint == False
        and args.training.checkpoint is not None
    ):
        # Load state dict from checkpoint
        if args.training.fsdp_ckpt:
            state_dict = torch.load(
                f"{args.training.checkpoint}/pytorch_model_fsdp.bin"
            )
        else:
            # Check if we have sharded SafeTensors files
            index_path = f"{args.training.checkpoint}/model.safetensors.index.json"
            if os.path.exists(index_path):
                # Load from sharded SafeTensors files
                import json
                with open(index_path, 'r') as f:
                    index_data = json.load(f)
                
                weight_map = index_data.get("weight_map", {})
                state_dict = {}
                
                # Group weights by file
                file_weights = {}
                for weight_name, file_name in weight_map.items():
                    if file_name not in file_weights:
                        file_weights[file_name] = []
                    file_weights[file_name].append(weight_name)
                
                # Load each shard file
                for file_name, weight_names in file_weights.items():
                    file_path = f"{args.training.checkpoint}/{file_name}"
                    file_state_dict = load_file(file_path)
                    
                    # Add only the weights that belong to this file
                    for weight_name in weight_names:
                        if weight_name in file_state_dict:
                            state_dict[weight_name] = file_state_dict[weight_name]
            else:
                # Fallback to single file
                state_dict = load_file(f"{args.training.checkpoint}/model.safetensors")
        # Filter out lm_head_number if finetuning
        if args.training.get("finetune", False) and args.training.get(
            "replace_lm_head", True
        ):
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if not k.startswith("lm_head_number")
            }
            logging.info(f"Chopping off lm_head!")

        if args.training.get("replace_atom_embedding", False):
            state_dict = {k: v for k, v in state_dict.items() if not "embed_atoms" in k}
            logging.info(f"Chopping off embed_atoms!")

        if args.training.get("replace_norm_stats", False):
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if not "energy_mean" in k
                and not "energy_std" in k
                and not "force_mean" in k
                and not "force_std" in k
            }
            logging.info(f"Chopping off norm stats!")

        if args.training.fsdp_ckpt and args.training.fsdp:
            state_dict = {".".join(k.split(".")[1:]): v for k, v in state_dict.items()}
            logging.info(f"Removing _orig_mod prefix from state dict!")

        strict = not (
            args.training.get("finetune", False)
            or args.model.model_type == "llama_prefix_readout"
        )
        # Load state dict into our model
        res = model.load_state_dict(state_dict, strict=strict)
        logging.info(f"Loaded checkpoint: {args.training.checkpoint}")
        logging.info(f"Missing keys: {res.missing_keys}")
        logging.info(f"Unexpected keys: {res.unexpected_keys}")

    # Convert model to bf16 if using flash_attention_2
    if (args.training.bf16 or args.training.fp16) and (
        args.training.attn_implementation == "flash_attention_2"
        or args.training.model_bf16
    ):
        dtype = torch.bfloat16 if args.training.bf16 else torch.float16
        model = model.to(dtype)
        logging.info(f"Setting model to half precision: {dtype}")

    if args.training.get("train_amount", None) is not None and not args.training.get(
        "override_lr", False
    ):
        freeze_model(model, args.training.train_amount)

    if args.model.get("double_precision", False):
        model = model.to(torch.float64)
        logging.info("Setting model to double precision")

    return model


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args: DictConfig):

    if args.training.use_spawn:
        logging.info("Using spawn start method!")
        torch.multiprocessing.set_start_method("spawn", force=True)
        # Avoid /dev/shm issues; use file-backed sharing
        torch.multiprocessing.set_sharing_strategy("file_system")

    # To prevent run from crashing on NERSC due to slow WANDB service
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    logs_path = args.training.logs_path

    n_actual_bins = get_n_actual_bins(args)

    seed_everything(args.training.seed)

    if args.training.finetune:
        assert (
            not args.dataset.per_atom_target
        ), "Per-atom target is already done in finetuning!"
        if args.dataset.get("norm_stats_path", None) is None:
            logging.warning(
                "No norm stats path provided for finetuning! Using default values (0, 1) for energy and force mean and std."
            )

    if args.dataset.joint_embedding:
        if n_actual_bins["pos"] > 15:
            logging.warning(
                f"The number of bins for position is greater than 15. This may lead to a large number of tokens with joint embedding ({n_bins['pos'] ** 3})."
            )
        n_actual_bins["pos"] = n_actual_bins["pos"] ** 3

    if args.dataset.joint_embedding_force and not args.training.finetune:
        if n_actual_bins["force"] > 15:
            logging.warning(
                f"The number of bins for force is greater than 15. This may lead to a large number of tokens with joint embedding ({n_actual_bins['force'] ** 3})."
            )
        n_actual_bins["force"] = n_actual_bins["force"] ** 3

    if args.dataset.n_bins is not None:
        tokenizer = get_tokenizer(
            n_bins=n_actual_bins,
            n_atom_types=args.dataset.n_atom_types,
            spin_min=args.dataset.spin_min,
            spin_max=args.dataset.spin_max,
            charge_min=args.dataset.charge_min,
            charge_max=args.dataset.charge_max,
            finetune=args.training.finetune,
            joint_embed_atoms=args.dataset.joint_embed_atoms,
        )
        tokenizer.add_special_tokens(
            {
                "pad_token": "[PAD]",
                "mask_token": "[MASK]",
                "bos_token": "<BOS>",
                "eos_token": "<EOS>",
            }
        )
    else:
        raise NotImplementedError("Need to discretize numbers from now on!")

    if not args.dataset.permutation_augmentation and args.dataset.first_force_only and not args.training.finetune:
        logging.warning(
            "First force only is activated but permutation augmentation is disabled."
        )

    if args.training.resume_from_checkpoint and args.training.checkpoint is None:
        checkpoint_paths = glob.glob(
            f"{logs_path}/{args.wandb.group_name}/{args.wandb.run_name}/checkpoint-*"
        )
        if len(checkpoint_paths) > 0:
            args.training.checkpoint = sorted(checkpoint_paths)[-1]
            logging.info(f"Resuming from checkpoint: {args.training.checkpoint}")
        else:
            logging.warning("No checkpoint found, will start from scratch!")

    if args.training.checkpoint is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.training.checkpoint)
        except Exception as e:
            logging.warning(f"Error loading tokenizer from checkpoint: {e}")
            logging.warning("Using tokenizer created from config!")

    if torch.cuda.is_available() and args.training.allow_tf32:
        # Can use is_torch_tf32_available() from transformers, set directly in trainer
        major_cc, _ = torch.cuda.get_device_capability()
        if major_cc >= 8:
            logging.info("Setting float32 matmul precision to high")
            torch.set_float32_matmul_precision("high")

    if args.training.attn_implementation == "sdpa_flash":
        logging.info("Enabling sdpa flash")
        torch.backends.cuda.enable_flash_sdp(True)
        args.training.attn_implementation = "sdpa"
        if not args.training.model_bf16:
            logging.warning(
                "Model is not bf16, but sdpa flash is enabled. This will lead to a slowdown!"
            )

    if args.model.eng_sum:
        assert (
            not args.dataset.per_atom_target
        ), "Per-atom target is not recommended with Energy Sum since the summation takes care of system size."

    dir_name = f"{logs_path}/{args.wandb.group_name}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    fsdp_config = None
    if args.training.fsdp:
        if type(args.training.fsdp) == str:
            assert args.training.fsdp in [
                "full_shard",
                "shard_grad_op",
                "hybrid_shard",
                "hybrid_shard_zero2",
                "offload",
            ], f"Invalid FSDP option: {args.training.fsdp}"
            fsdp = [args.training.fsdp, "auto_wrap"]
        else:
            fsdp = ["full_shard", "auto_wrap"]
        fsdp_config = {
            "backward_prefetch": "backward_pre",
            "forward_prefetch": True,
        }
        logging.info(f"Using FSDP with {fsdp}. And config {fsdp_config}.")
    else:
        fsdp = False

    training_args = TrainingArguments(
        output_dir=f"{dir_name}/{args.wandb.run_name}",
        eval_strategy="steps",
        learning_rate=args.training.lr,
        dataloader_num_workers=args.training.dataloader_num_workers,
        dataloader_prefetch_factor=args.training.dataloader_prefetch_factor,
        num_train_epochs=args.training.num_epochs,
        weight_decay=args.training.weight_decay,
        push_to_hub=False,
        per_device_train_batch_size=args.training.batch_size,
        per_device_eval_batch_size=(
            args.training.batch_size
            if args.training.eval_batch_size is None
            else args.training.eval_batch_size
        ),
        run_name=args.wandb.run_name,
        report_to="none" if args.debug else "wandb",
        logging_steps=args.training.logging_steps,
        eval_steps=args.training.eval_steps,
        save_steps=args.training.save_steps,
        save_total_limit=args.training.save_total_limit,
        load_best_model_at_end=args.training.load_best_model_at_end,
        warmup_ratio=args.training.warmup_ratio,
        gradient_accumulation_steps=(
            args.training.gradient_accumulation_steps if not args.debug else 1
        ),
        torch_compile=(
            args.training.torch_compile
            if (not args.debug or args.training.force_compile)
            else False
        ),
        eval_accumulation_steps=args.training.eval_accumulation_steps,
        fp16=args.training.fp16,
        bf16=args.training.bf16,
        tf32=args.training.allow_tf32,
        batch_eval_metrics=args.training.batch_eval_metrics,
        lr_scheduler_type=args.training.lr_scheduler,
        lr_scheduler_kwargs=(
            args.training.lr_scheduler_kwargs
            if args.training.lr_scheduler_kwargs is not None
            else {}
        ),
        metric_for_best_model=args.training.metric_for_best_model,
        ddp_find_unused_parameters=args.training.ddp_find_unused_parameters,
        remove_unused_columns=not args.dataset.preprocessed,
        fsdp_config=fsdp_config,
        fsdp=fsdp,
        gradient_checkpointing=args.training.gradient_checkpointing,
        optim=args.training.optim_name,
    )

    # Initialize the data collator with your tokenizer
    data_collator = get_collator(args, tokenizer)

    start_end_indices_by_token_type = get_start_end_indices_by_token_type(
        args, tokenizer, n_actual_bins
    )

    dataset_args = dict(args.dataset)
    dataset_args.update({"causal": args.model.causal})
    dataset_args.update({"finetune": args.training.get("finetune", False)})

    logging.info(f"Loading training dataset...")

    # Check if we are using the new datasets_v2 pipeline
    if args.dataset.pipeline_v2:
        # First, validate that the rest of the config is consistent
        _validate_v2_config_consistency(args)

        logging.info("Using datasets_v2 pipeline.")
        # The config passed to instantiate is the one under the `train` or `val` key
        train_dataset = hydra.utils.instantiate(args.dataset.train_dataset)
        val_dataset = hydra.utils.instantiate(args.dataset.val_dataset)
        val_dataset_og = val_dataset
    else:
        raise NotImplementedError("Legacy pipeline is not supported anymore!")

    if args.dataset.joint_embedding and args.model.loss_name == "smooth_xent":
        logging.warning(
            "Joint embedding is set but smooth xent loss is used. This might not work as expected since smoothing will happen over bins that are not physically close."
        )

    if args.dataset.training_percentage is not None:
        if os.path.exists(f"{args.dataset.train_path}/train_random_indices.npy"):
            random_indices = np.load(
                f"{args.dataset.train_path}/train_random_indices.npy"
            )
        else:
            random_indices = np.random.permutation(len(train_dataset))
            np.save(
                f"{args.dataset.train_path}/train_random_indices.npy", random_indices
            )
        n_training_samples = int(args.dataset.training_percentage * len(train_dataset))
        if args.dataset.preprocessed:
            train_dataset = train_dataset.select(random_indices[:n_training_samples])
        else:
            train_dataset = Subset(train_dataset, random_indices[:n_training_samples])

    if not args.debug and not args.training.torch_compile:
        logging.warning("Not compiling the model! This will slow down training!")

    dict_args = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)
    dict_args["size_train_dataset"] = len(train_dataset)
    dict_args["size_val_dataset"] = len(val_dataset)

    # Check if we're in a distributed training setup
    global_rank = int(os.environ.get("RANK", -1))
    # Only initialize wandb on the main process (global_rank == 0)
    if global_rank in [-1, 0]:
        wandb.init(
            project="mmlm",
            name=args.wandb.run_name,
            group=args.wandb.group_name,
            config=dict_args,
            mode="online" if not args.debug else "disabled",
        )

    energy_mean, energy_std, force_mean, force_std = 0, 1, 0, 1
    if args.dataset.get("norm_stats_path", None) is not None:
        stats = np.load(args.dataset.norm_stats_path, allow_pickle=True).item()
        energy_mean = stats["energy_mean"]
        energy_std = stats["energy_std"]
        force_mean = stats["force_mean"]
        force_std = stats["force_std"]

    def compute_metrics_wrapper(p):
        with torch.no_grad():
            if args.dataset.continuous and not args.model.get("loss_name", None) in [
                "xent",
                "smooth_xent",
            ]:
                return compute_metrics_continuous(
                    p,
                    val_dataset_og,
                    args,
                    start_end_indices_by_token_type=start_end_indices_by_token_type,
                    tokenizer=tokenizer,
                    energy_mean=energy_mean,
                    energy_std=energy_std,
                    force_mean=force_mean,
                    force_std=force_std,
                )
            raise NotImplementedError("Must include continuous input from now on!")

    def model_init():
        model = get_model(args, tokenizer, start_end_indices_by_token_type)
        return model

    callbacks = [NaNLossCallback(cutoff=args.training.nan_loss_cutoff)]
  
    trainer = MoleculeTrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_wrapper,
        callbacks=callbacks,
        preprocess_logits_for_metrics=(
            preprocess_logits_for_metrics
            if (
                not args.training.full_eval
                and (
                    not args.dataset.continuous
                    or args.model.loss_name in ["xent", "smooth_xent"]
                )
            )
            else None
        ),
        override_lr=args.training.get("override_lr", False),
        checkpoint_path=(
            args.training.checkpoint
            if args.training.get("override_lr", False)
            else None
        ),
        lr_layer_decay=args.training.get("lr_layer_decay", None),
        model_n_layers=args.model.num_layers,
        scheduler_step_percentage=args.training.get("scheduler_step_percentage", None),
        pos_token_id=(
            start_end_indices_by_token_type["pos"][0]
            if (
                args.training.get("finetune", False)
                and args.training.get("ft_normalize_batch", False)
            )
            else None
        ),
        train_amount=args.training.get("train_amount", None),
        prefix_readout=args.model.model_type == "llama_prefix_readout",
        use_muon=args.training.get("use_muon", False),
        muon_lr=args.training.get("muon_lr", 0.02),
        muon_weight_decay=args.training.get("muon_weight_decay", 0.01),
    )

    
    if args.training.checkpoint is not None and args.training.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.training.checkpoint)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
