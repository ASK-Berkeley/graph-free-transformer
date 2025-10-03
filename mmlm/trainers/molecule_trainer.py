from transformers.trainer import Trainer
from transformers.optimization import get_scheduler
from transformers.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    denumpify_detensorize,
)

import torch
import numpy as np
import time
from typing import Optional, List
from packaging import version

import torch.distributed as dist
import torch.nn as nn
import logging

from transformers.utils import is_sagemaker_mp_enabled

from mmlm.utils.utils import freeze_model


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import (
        smp_forward_backward,
        smp_forward_only,
        smp_gather,
        smp_nested_concat,
    )
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


class MoleculeTrainer(Trainer):
    def __init__(
        self,
        *args,
        override_lr=False,
        checkpoint_path=None,
        lr_layer_decay=None,
        model_n_layers=None,
        scheduler_step_percentage=None,
        pos_token_id=None,
        train_amount=None,
        prefix_readout=False,
        use_muon=False,
        muon_lr=0.02,
        muon_weight_decay=0.01,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.override_lr = override_lr
        self.checkpoint_path = checkpoint_path
        self.lr_layer_decay = lr_layer_decay
        self.model_n_layers = model_n_layers
        self.scheduler_step_percentage = scheduler_step_percentage
        self.prefix_readout = prefix_readout

        self.pos_token_id = pos_token_id
        self.train_amount = train_amount
        self.use_muon = use_muon
        self.muon_lr = muon_lr
        self.muon_weight_decay = muon_weight_decay

    # To properly account for gradient accumulation
    # This only gets called in train
    def get_batch_samples(self, epoch_iterator, num_batches, device="cuda"):
        batch_samples = []
        num_items_in_batch = None
        for _ in range(num_batches):
            try:
                batch_samples += [next(epoch_iterator)]
            except StopIteration:
                break

        if len(batch_samples) > 0 and "labels" in batch_samples[0]:
            # For now we don't support object detection
            if self.prefix_readout:
                num_items_in_batch = (
                    batch_samples[0]["labels"]["labels"].shape[0]
                    * self.args.gradient_accumulation_steps
                )
            else:
                try:
                    if self.pos_token_id is not None:
                        num_items_in_batch = sum(
                            [
                                (batch["labels"]["labels"].eq(self.pos_token_id)).sum()
                                for batch in batch_samples
                            ]
                        )
                        energy_num_items_in_batch = sum(
                            [
                                batch["labels"]["labels"].shape[0]
                                for batch in batch_samples
                            ]
                        )
                        num_items_in_batch = (
                            energy_num_items_in_batch,
                            num_items_in_batch,
                        )
                    else:
                        # MODIFIED: Since the labels contain extra information for compute metrics,
                        # we need to index into the actual label_ids here, hence the extra ["labels"]
                        num_items_in_batch = sum(
                            [
                                (batch["labels"]["labels"].ne(-100)).sum()
                                for batch in batch_samples
                            ]
                        )
                except (TypeError, AttributeError):
                    pass

        if num_items_in_batch is not None:
            if self.args.average_tokens_across_devices:
                num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum()

            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(device)

                if self.args.n_gpu > 1 and num_items_in_batch.dim() == 0:
                    # In the DataParallel case, convert the scalar tensor into a 1-dim tensor
                    num_items_in_batch = num_items_in_batch.unsqueeze(0)

        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.item()

        return batch_samples, num_items_in_batch

    def to_device(self, batch, device, dtype=None):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
                if dtype is not None and v.dtype in [
                    torch.float32,
                    torch.float16,
                    torch.bfloat16,
                ]:
                    batch[k] = batch[k].to(dtype)
            elif isinstance(v, dict):
                batch[k] = self.to_device(v, device, dtype)
        return batch

    def _load_optimizer_and_scheduler(self, checkpoint):
        if self.override_lr:
            checkpoint = self.checkpoint_path
            logging.info(f"Loading optimizer and scheduler from {checkpoint}!")
        super()._load_optimizer_and_scheduler(checkpoint)

        if self.override_lr:

            # Freeze model if train_amount is not None
            # And remove frozen parameters from optimizer
            if self.train_amount is not None:
                freeze_model(self.model, self.train_amount)
                # Remove frozen parameters from optimizer
                for param_group in self.optimizer.param_groups:
                    param_group["params"] = [
                        p for p in param_group["params"] if p.requires_grad
                    ]

                self.optimizer.state = {
                    k: v
                    for k, v in self.optimizer.state.items()
                    if k
                    in set(p for g in self.optimizer.param_groups for p in g["params"])
                }

            # Update learning rate for all parameter groups in the optimizer
            lr_to_override = self.muon_lr if self.use_muon else self.args.learning_rate
            logging.info(
                f"Overriding learning rate to {lr_to_override} and using {self.args.lr_scheduler_type} scheduler with {1000 * self.args.warmup_ratio} warmup steps!"
            )

            for param_group in self.optimizer.param_groups:
                if self.use_muon:
                    # Moonlight kernel only has one lr
                    lr = self.muon_lr
                else:
                    if param_group.get("use_muon", False):
                        lr = self.muon_lr
                    else:
                        lr = self.args.learning_rate
                param_group["lr"] = lr
                param_group["initial_lr"] = lr
            if (
                "constant" in self.args.lr_scheduler_type
                or "plateau" in self.args.lr_scheduler_type
            ):
                self.lr_scheduler = get_scheduler(
                    self.args.lr_scheduler_type,
                    optimizer=self.optimizer,
                    num_warmup_steps=int(1000 * self.args.warmup_ratio),
                )
            else:
                total_train_batch_size = (
                    self._train_batch_size
                    * self.args.gradient_accumulation_steps
                    * self.args.world_size
                )
                (_, _, _, _, _, _, max_steps,) = self.set_initial_training_values(
                    self.args, self.get_train_dataloader(), total_train_batch_size
                )
                # Cosine schedule, requires num_training_steps
                self.lr_scheduler = get_scheduler(
                    self.args.lr_scheduler_type,
                    optimizer=self.optimizer,
                    num_warmup_steps=int(1000 * self.args.warmup_ratio),
                    num_training_steps=max_steps,
                    scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
                )

                if self.scheduler_step_percentage is not None:
                    for _ in range(int(max_steps * self.scheduler_step_percentage)):
                        self.lr_scheduler.step()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        # Handle Muon optimizer
        if self.use_muon:
            return self._create_muon_optimizer()

        if self.lr_layer_decay is None:
            return super().create_optimizer()

        logging.info(f"Setting lr per layer with decay factor {self.lr_layer_decay}!")
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            groups = {
                "wd": {
                    "params": [],
                    "weight_decay": self.args.weight_decay,
                },
                "no_wd": {
                    "params": [],
                    "weight_decay": 0.0,
                },
            }

            for name, param in opt_model.named_parameters():
                if param.requires_grad:
                    if "layers" in name:
                        layer_num = int(name.split(".")[2])
                        weight_decay = (
                            self.args.weight_decay if name in decay_parameters else 0.0
                        )
                        lr_decay_factor = self.lr_layer_decay ** (
                            self.model_n_layers - layer_num - 1
                        )
                        if layer_num < self.model_n_layers:
                            if layer_num not in groups:
                                groups[layer_num] = {
                                    "params": [],
                                    "weight_decay": weight_decay,
                                    "lr": self.args.learning_rate * lr_decay_factor,
                                }
                            groups[layer_num]["params"].append(param)
                    elif name in decay_parameters:
                        groups["wd"]["params"].append(param)
                    else:
                        groups["no_wd"]["params"].append(param)

            optimizer_grouped_parameters = [groups[group_name] for group_name in groups]

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                    self.args, opt_model
                )

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

            if (
                "bitsandbytes" in str(optimizer_cls)
                and optimizer_kwargs.get("optim_bits", None) == 8
            ):
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {
                                p.data_ptr(): p.numel() for p in module.parameters()
                            }.values()
                        )
                        logging.info(f"skipped {module}: {skipped / 2**20}M params")
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32}
                        )
                        logging.debug(f"bitsandbytes: will optimize {module} in fp32")
                logging.info(f"skipped: {skipped / 2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluate to support EMA model switching.

        This method handles switching to the EMA model before evaluation
        and restoring the original model after evaluation.
        """
        # Find the EMA callback if it exists
        ema_callback = None
        for callback in self.callback_handler.callbacks:
            if hasattr(callback, "ema_model") and callback.ema_model is not None:
                ema_callback = callback
                break

        # Switch to EMA model before evaluation if callback exists
        if ema_callback is not None and ema_callback.use_ema_for_validation:
            # Store original model
            ema_callback.original_model = self.model

            # Move EMA model to GPU if it was stored on CPU
            if ema_callback.store_ema_on_cpu:
                ema_callback.ema_model = ema_callback.ema_model.cuda()

            # Replace model with EMA model
            self.model = ema_callback.ema_model

            logging.info("Switched to EMA model for evaluation")

        result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # Always restore the original model, even if evaluation fails
        if ema_callback is not None and hasattr(ema_callback, "original_model"):
            # Restore original model
            self.model = ema_callback.original_model

            # Move EMA model back to CPU if it was stored there
            if ema_callback.store_ema_on_cpu:
                ema_callback.ema_model = ema_callback.ema_model.cpu()

            # Clean up
            delattr(ema_callback, "original_model")

            logging.info("Restored original model after evaluation")

        return result

    def _create_muon_optimizer(self):
        """
        Create Muon optimizer with proper parameter grouping.

        Muon is designed for hidden weights (ndim >= 2), while other parameters
        (embeddings, classifier heads, gains/biases) should use AdamW.
        """
        try:
            # from muon import MuonWithAuxAdam
            from kernels import get_kernel

            optimizer = get_kernel(
                "motif-technologies/optimizer",
                revision="99e7c0ce8a61be42956f43ab434fd880eaec9d4c",
            )
            MuonWithAuxAdam = optimizer.Muon
            # get_default_muon_param_groups = optimizer.muon.get_default_muon_param_groups

            # from muon_fsdp2 import Muon as MuonWithAuxAdam
        except ImportError:
            raise ImportError(
                "Muon optimizer not found. Please install it with: "
                "pip install git+https://github.com/KellerJordan/Muon"
            )

        logging.info("Creating Muon optimizer with mixed parameter groups")
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        # When can this be not None?
        if self.optimizer is None:
            # Get standard AdamW parameters from transformers
            _, adamw_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Remove 'params' if it exists (we'll set our own)
            adamw_kwargs.pop("params", None)

            # Categorize parameters for Muon vs AdamW
            hidden_weights = []
            other_params = []

            for name, param in opt_model.named_parameters():
                if param.requires_grad:
                    # if param.ndim >= 2:
                    # Hidden weights (transformer layers, linear projections, etc.)
                    # These benefit from Muon optimization
                    if any(
                        component in name
                        for component in [
                            "layers.",
                            "attention.",
                            "mlp.",
                            "feed_forward.",
                            "self_attn.",
                            "linear",
                            "projection",
                        ]
                    ):
                        if (
                            "bias" in name
                            or "scale" in name
                            or "layer_norm" in name
                            or "layernorm" in name
                            or "head" in name
                        ):
                            other_params.append(param)
                        else:
                            hidden_weights.append(param)
                    else:
                        # 2D params that aren't core transformer weights (e.g., embeddings)
                        other_params.append(param)
                    # else:
                    # 1D parameters (biases, layer norms, scales, etc.)
                    #    other_params.append(param)

            logging.info(
                f"Muon will optimize {len(hidden_weights)} hidden weight parameters"
            )
            logging.info(f"AdamW will optimize {len(other_params)} other parameters")

            # Filter adamw_kwargs to only include parameters that Muon expects
            muon_expected_keys = {
                "params",
                "lr",
                "betas",
                "eps",
                "weight_decay",
                "use_muon",
            }
            filtered_adamw_kwargs = {
                k: v for k, v in adamw_kwargs.items() if k in muon_expected_keys
            }
            # Create parameter groups
            param_groups = [
                {
                    "params": hidden_weights,
                    "use_muon": True,
                    "lr": self.muon_lr,
                    "weight_decay": self.muon_weight_decay,
                },
                {
                    "params": other_params,
                    "use_muon": False,
                    "weight_decay": self.args.weight_decay,
                    **filtered_adamw_kwargs,  # Use filtered AdamW params
                },
            ]
            # self.optimizer = MuonWithAuxAdam(param_groups)
            # self.optimizer.defaults = {"lr": self.muon_lr}
            # params = get_default_muon_param_groups(opt_model)
            self.optimizer = MuonWithAuxAdam(
                opt_model,
                lr=self.muon_lr,
                weight_decay=self.muon_weight_decay,
                adamw_betas=adamw_kwargs["betas"],
                adamw_eps=adamw_kwargs["eps"],
            )

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
