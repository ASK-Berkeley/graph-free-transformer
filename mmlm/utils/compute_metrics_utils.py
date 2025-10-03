import numpy as np

import torch


def mae(predictions, references):
    return {"mae": np.mean(np.abs(predictions - references))}


def accuracy(predictions, references, prefix=None):
    predictions = np.array(predictions)
    references = np.array(references)

    pref = "" if prefix is None else f"{prefix}_"
    return {f"{pref}accuracy": np.mean((predictions == references))}


def cross_entropy(predictions, references, prefix=None):
    pref = "" if prefix is None else f"{prefix}_"

    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions)
    if not isinstance(references, torch.Tensor):
        references = torch.tensor(references)
    loss = torch.nn.functional.cross_entropy(predictions, references).item()
    return {f"{pref}xent_loss": loss}


def cosine_similarity(predictions, references, prefix=None):
    pref = "" if prefix is None else f"{prefix}_"

    cossims = np.sum(predictions * references, axis=-1) / (
        np.linalg.norm(predictions, axis=-1) * np.linalg.norm(references, axis=-1)
    )
    return {f"{pref}cosine_similarity": np.mean(cossims)}


def abs_mag_diff(predictions, references, prefix=None):
    pref = "" if prefix is None else f"{prefix}_"

    abs_mag_diffs = np.abs(
        np.linalg.norm(predictions, axis=-1) - np.linalg.norm(references, axis=-1)
    )
    return {f"{pref}abs_mag_diff": np.mean(abs_mag_diffs)}


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = logits.argmax(-1)
    return pred_ids


def compute_metrics_continuous_grad(
    p,
    force_pad_value,
    regress_forces=True,
    args=None,
    energy_mean=0,
    energy_std=1,
    force_mean=0,
    force_std=1,
):
    eng_pred = p.predictions[0].squeeze()
    force_pred = p.predictions[1]

    if force_pad_value is not None and regress_forces:
        force_pred = force_pred[force_pred != force_pad_value].reshape(-1, 3)

    eng_true = p.label_ids["target_labels"]
    force_true = p.label_ids["force_labels"]
    if force_pad_value is not None and regress_forces:
        force_true = force_true[force_true != force_pad_value].reshape(-1, 3)

    results = {}
    normalized_eng_pred = eng_pred  # (eng_pred - energy_mean) / energy_std
    normalized_eng_true = eng_true  # (eng_true - energy_mean) / energy_std
    energy_loss = (
        args.model.loss_weights["target"]
        * mae(predictions=normalized_eng_pred, references=normalized_eng_true)["mae"]
    )

    results.update(
        {
            "target_mae": mae(predictions=eng_pred, references=eng_true)["mae"],
            "target_loss": energy_loss,
        }
    )

    if regress_forces:
        results.update(
            {
                "force_mae": mae(predictions=force_pred, references=force_true)["mae"],
                "force_cosine_similarity": cosine_similarity(
                    predictions=force_pred, references=force_true
                )["cosine_similarity"],
                "force_abs_mag_diff": abs_mag_diff(
                    predictions=force_pred, references=force_true
                )["abs_mag_diff"],
            }
        )

    return results


def compute_metrics_prefix_readout(p, start_end_indices_by_token_type, args):
    eng_pred = p.predictions[0].squeeze()
    force_pred = p.predictions[1]

    labels = p.label_ids["labels"]
    true_nums = p.label_ids["num_labels"]
    target_mask = (labels >= start_end_indices_by_token_type["target"][0]) & (
        labels <= start_end_indices_by_token_type["target"][1]
    )
    force_mask = (labels >= start_end_indices_by_token_type["force"][0]) & (
        labels <= start_end_indices_by_token_type["force"][1]
    )

    target_true = true_nums[target_mask][:, 0]
    force_true = true_nums[force_mask]

    if not args.dataset.joint_embedding_force:
        force_true = force_true[:, 0].reshape(-1, 3)
        force_pred = force_pred[:, 0].reshape(-1, 3)

    results = {}

    results.update(
        {
            "target_mae": mae(predictions=eng_pred, references=target_true)["mae"],
            "force_mae": mae(predictions=force_pred, references=force_true)["mae"],
        }
    )

    results.update(
        {
            "force_cosine_similarity": cosine_similarity(
                predictions=force_pred, references=force_true
            )["cosine_similarity"],
            "force_abs_mag_diff": abs_mag_diff(
                predictions=force_pred, references=force_true
            )["abs_mag_diff"],
        }
    )

    return results


def compute_metrics_continuous(
    p,
    val_dataset,
    args,
    start_end_indices_by_token_type,
    tokenizer=None,
    energy_mean=0,
    energy_std=1,
    force_mean=0,
    force_std=1,
):

    if (
        "grad" in args.model.model_type
        or "pos_readout" in args.model.model_type
        or "dir_readout" in args.model.model_type
    ):
        return compute_metrics_continuous_grad(
            p,
            args.training.force_pad_value
            if args.training.max_force_per_item is not None
            else None,
            regress_forces=args.model.regress_forces,
            args=args,
            energy_mean=energy_mean,
            energy_std=energy_std,
            force_mean=force_mean,
            force_std=force_std,
        )
    elif args.model.model_type == "llama_prefix_readout":
        return compute_metrics_prefix_readout(p, start_end_indices_by_token_type, args)

    pred_nums = p.predictions
    label_tokens = p.label_ids["labels"]
    true_nums = p.label_ids["num_labels"]

    if args.dataset.joint_embedding:
        pred_nums = pred_nums[:, :-1].squeeze()
    else:
        pred_nums = pred_nums[..., :-1, 0].squeeze()

    label_tokens = label_tokens[..., 1:]
    true_nums = true_nums[..., 1:, :].squeeze()

    results = {}

    for token_type in start_end_indices_by_token_type:

        start, end = start_end_indices_by_token_type[token_type]

        token_mask = (label_tokens <= end) & (label_tokens >= start)

        reference = true_nums[token_mask]
        pred_num = pred_nums[token_mask]

        if token_type == "target" and args.dataset.joint_embedding:
            reference = reference[:, 0]
            pred_num = pred_num[:, 0]

        results.update(
            {
                f"{token_type}_mae": mae(predictions=pred_num, references=reference)[
                    "mae"
                ]
            }
        )

        if token_type == "force":
            pred_num = pred_num.reshape(-1, 3)
            reference = reference.reshape(-1, 3)
            results.update(
                abs_mag_diff(
                    predictions=pred_num, references=reference, prefix=token_type
                )
            )
            results.update(
                cosine_similarity(
                    predictions=pred_num, references=reference, prefix=token_type
                )
            )

    return results


def compute_metrics(
    p, val_dataset, args, start_end_indices_by_token_type=None, tokenizer=None
):
    results = {}
    
    
    # (Eval batch size, seq_len, vocab_size)
    logits = torch.tensor(p.predictions)
    # (Eval batch size, seq_len)
    label_ids = torch.tensor(p.label_ids["labels"])

    true_exact_targets = p.label_ids.get("true_targets", None)
    true_exact_forces_padded = p.label_ids.get("true_forces", None)

    logits = logits[:, :-1]
    label_ids = label_ids[..., 1:]

    predicted_tokens = (
        logits.argmax(-1).cpu().numpy() if args.training.full_eval else logits
    )

    spherical_pred_nums = []
    spherical_true_nums = []
    spherical_weighted_pred_nums = []

    for token_type in start_end_indices_by_token_type:

        if args.dataset.joint_embedding and token_type == "pos":
            continue

        first_idx, last_idx = start_end_indices_by_token_type[token_type]
        token_mask = (label_ids <= last_idx) & (label_ids >= first_idx)
        true_idx = label_ids[token_mask]
        predicted = predicted_tokens[token_mask]
        predicted_logits = logits[token_mask] if args.training.full_eval else None

        results.update(
            accuracy(predictions=predicted, references=true_idx, prefix=token_type)
        )

        if args.training.full_eval:
            results.update(
                cross_entropy(
                    predictions=predicted_logits,
                    references=true_idx,
                    prefix=token_type,
                )
            )

        if token_type == "atomic_numbers":
            continue

        true_idx = true_idx - first_idx
        predicted_idx = np.clip(predicted - first_idx, 0, last_idx - first_idx)

        predicted_weights = (
            torch.nn.functional.softmax(
                predicted_logits[:, first_idx : last_idx + 1], dim=-1
            )
            if args.training.full_eval
            else None
        )

        (
            pred_num,
            pred_weighted_num,
            true_bin_num,
        ) = val_dataset.causal_bin_tokens_to_nums(
            predicted_weights=predicted_weights,
            pred_idxs=predicted_idx,
            true_idx=true_idx,
            token_type=token_type,
        )
        results.update(
            {
                f"{token_type}_mae_binned": mae(
                    predictions=pred_num, references=true_bin_num
                )["mae"]
            }
        )

        if token_type in ["force_r", "force_direction"]:
            spherical_pred_nums.append(
                pred_num.reshape(-1, 1 if token_type == "force_r" else 2)
            )
            spherical_true_nums.append(
                true_bin_num.reshape(-1, 1 if token_type == "force_r" else 2)
            )
            if args.training.full_eval:
                spherical_weighted_pred_nums.append(
                    pred_weighted_num.reshape(-1, 1)
                )

        if token_type in ["target"]:
            if true_exact_targets is None:
                true_exact_targets = true_bin_num
            results.update(
                {
                    f"{token_type}_mae": mae(
                        predictions=pred_num,
                        references=true_exact_targets.squeeze(),
                    )["mae"]
                }
            )
            if args.training.full_eval:
                results.update(
                    {
                        f"{token_type}_mae_weighted": mae(
                            predictions=pred_weighted_num,
                            references=true_exact_targets.squeeze(),
                        )["mae"]
                    }
                )
        elif token_type in ["force"] and args.training.full_eval:
            
            reference_forces = true_bin_num
            results.update(
                {
                    f"{token_type}_mae": mae(
                        predictions=pred_num, references=reference_forces
                    )["mae"]
                }
            )
            if args.training.full_eval and token_type == "force":
                results.update(
                    {
                        f"{token_type}_mae_weighted": mae(
                            predictions=pred_weighted_num,
                            references=reference_forces,
                        )["mae"]
                    }
                )
            pred_num = pred_num.reshape(-1, 3)
            reference_forces = reference_forces.reshape(-1, 3)
            results.update(
                abs_mag_diff(
                    predictions=pred_num,
                    references=reference_forces,
                    prefix=token_type,
                )
            )
            results.update(
                cosine_similarity(
                    predictions=pred_num,
                    references=reference_forces,
                    prefix=token_type,
                )
            )

    return results
