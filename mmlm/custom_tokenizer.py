from transformers import PreTrainedTokenizerFast
from tokenizers import pre_tokenizers, models, Tokenizer
import logging
import numpy as np


def get_tokenizer(
    n_bins,
    n_atom_types=10,
    only_pos_tokens=False,
    spin_min=None,
    spin_max=None,
    charge_min=None,
    charge_max=None,
    finetune=False,
    joint_embed_atoms=False,
):

    # Define the base tokenizer
    tokenizer = Tokenizer(models.WordLevel())

    special_tokens = [
        "[CONTEXTSTART]",
        "[QUERYSTART]",
        "[POS]",
        "[TARGET]",
        "[MASK]",
        "[CONTEXT_TARGET]",
        "[PAD]",
        "<BOS>",
        "<EOS>",
        "[POS_END]",
        "[CELL]",
        "[CELL_END]",
        "[TARGET_END]",
        "[FORCE]",
        "[FORCE_END]",
        "[RELAX]",
        "[RELAX_END]",
        "[STRESS]",
        "[STRESS_END]",
        "[SPIN]",
        "[SPIN_END]",
        "[CHARGE]",
        "[CHARGE_END]",
        "<NUM>",
        "<NUM_target>",
        "<NUM_cell>",
        "<NUM_stress>",
        "<NUM_force>",
        "<NUM_spin>",
        "<NUM_charge>",
    ]


    number_bin_tokens = [f"<NUM_{i}>" for i in range(n_bins["pos"])]

    if not only_pos_tokens and not finetune:
        for t in n_bins:
            if t != "pos":
                number_bin_tokens += [f"<NUM_{t}_{i}>" for i in range(n_bins[t])]

    if joint_embed_atoms:
        atomic_number_tokens = []
    else:
        atomic_number_tokens = [f"a_{i}:" for i in range(1, n_atom_types + 1)]

    if spin_min is not None:
        spin_tokens = [f"<NUM_spin_{i}>" for i in np.arange(spin_min, spin_max + 1, 1)]
    else:
        spin_tokens = []

    if charge_min is not None:
        charge_tokens = [
            f"<NUM_charge_{i}>" for i in np.arange(charge_min, charge_max + 1, 1)
        ]
    else:
        charge_tokens = []

    vocab = {
        token: i
        for i, token in enumerate(
            special_tokens
            + atomic_number_tokens
            + spin_tokens
            + charge_tokens
            + number_bin_tokens
        )
    }

    # Set vocabulary
    tokenizer.model = models.WordLevel(vocab=vocab)

    # Add a pre-tokenizer to handle whitespace and split on special characters or patterns
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Whitespace(),  # Basic whitespace splitting
            pre_tokenizers.Split(
                r"\s+", behavior="removed"
            ),  # Further whitespace cleanup
            pre_tokenizers.Split(
                r"(<NUM_\d+>)", behavior="isolated"
            ),  # Isolate number bins
            pre_tokenizers.Split(
                r"(\[.*?\])", behavior="isolated"
            ),  # Isolate special tokens like [POS], [TARGET]
        ]
    )

    # Convert to Hugging Face's PreTrainedTokenizerFast
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    hf_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    hf_tokenizer.add_tokens(number_bin_tokens)
    hf_tokenizer.add_tokens(atomic_number_tokens)
    hf_tokenizer.add_tokens(spin_tokens)
    hf_tokenizer.add_tokens(charge_tokens)
    return hf_tokenizer
