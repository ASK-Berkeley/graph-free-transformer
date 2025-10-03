from transformers import (
    DataCollatorForLanguageModeling,
)

import torch


class DataCollatorForContinuousMoleculeLanguageModeling(
    DataCollatorForLanguageModeling
):
    def __init__(
        self,
        *args,
        preprocessed=False,
        joint_embedding=False,
        data_bf16=False,
        validation_mode=False,
        max_force_per_batch=None,
        force_pad_value=-200,
        finetune=False,
        atom_embedding=False,
        lmax=None,
        multi_atom_embedding_dim=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.preprocessed = preprocessed
        self.joint_embedding = joint_embedding
        self.data_bf16 = data_bf16
        self.validation_mode = validation_mode
        self.max_force_per_batch = max_force_per_batch
        self.force_pad_value = force_pad_value
        self.finetune = finetune
        self.atom_embedding = atom_embedding
        self.lmax = lmax
        self.multi_atom_embedding_dim = multi_atom_embedding_dim

    def to_bf16(self, x):
        for k, v in x.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                x[k] = v.to(torch.bfloat16)
            elif isinstance(v, dict):
                x[k] = self.to_bf16(v)
        return x

    def torch_call(self, features):

        # Batch size, text, y, y_mean, y_std
        n_atoms = 1
        if self.preprocessed:
            tokens = [f["tokens"] for f in features]
            true_numbers = [f["numbers"] for f in features]
            if self.finetune:
                force_labels = [f["force_label"] for f in features]
                energy_labels = [f["energy_label"] for f in features]
        else:
            tokens = self.tokenizer([f[0] for f in features])["input_ids"]
            true_numbers = [f[1] for f in features]

            if self.atom_embedding:
                atoms = [f[2] for f in features]
            if self.lmax is not None:
                dir_embeddings = [f[3] for f in features]

            if self.finetune:
                force_labels = [f[-1] for f in features]
                energy_labels = [f[-2] for f in features]

        # Pad the tokens
        super_output = super().torch_call(tokens)
        # Pad the numbers
        max_len = max([len(t) for t in super_output["input_ids"]])
        if self.joint_embedding:
            # numbers of shape n x 3
            padded_numbers = [
                torch.concatenate([x, torch.zeros((max_len - len(x), 3))], dim=0)
                for x in true_numbers
            ]
            padded_numbers = torch.stack(padded_numbers).float()

            if self.atom_embedding:
                if self.multi_atom_embedding_dim is not None:
                    padded_atoms = [
                        torch.concatenate(
                            [
                                x,
                                torch.zeros(
                                    (max_len - len(x), self.multi_atom_embedding_dim)
                                ),
                            ],
                            dim=0,
                        )
                        for x in atoms
                    ]
                    padded_atoms = torch.stack(padded_atoms).float()
                else:
                    padded_atoms = [
                        torch.concatenate([x, torch.zeros((max_len - len(x)))], dim=0)
                        for x in atoms
                    ]
                    padded_atoms = torch.stack(padded_atoms).long()

            if self.lmax is not None:
                padded_dir_embeddings = [
                    torch.concatenate(
                        [x, torch.zeros((max_len - len(x), self.lmax + 1))], dim=0
                    )
                    for x in dir_embeddings
                ]
                padded_dir_embeddings = torch.stack(padded_dir_embeddings)

        else:
            # numbers of 1D
            padded_numbers = (
                torch.stack(
                    [
                        torch.nn.ConstantPad1d((0, max_len - len(x)), -100)(x)
                        for x in true_numbers
                    ]
                )
                .unsqueeze(-1)
                .float()
            )

        super_output["input_ids"] = {
            "tokens": super_output["input_ids"],
            "numbers": padded_numbers,
        }

        if self.atom_embedding:
            super_output["input_ids"]["atoms"] = padded_atoms

        if self.lmax is not None:
            super_output["input_ids"]["dir_embeddings"] = padded_dir_embeddings

        new_labels = {
            "labels": super_output["labels"],
            "num_labels": padded_numbers,
        }

        if self.finetune:

            force_labels = torch.concatenate(force_labels, dim=0)
            if self.max_force_per_batch is not None:
                # Calculate how many rows we need to add to reach max_force_per_batch
                # Pad with rows at the bottom (0 padding on the first dimension)
                force_labels = torch.nn.ConstantPad2d(
                    (0, 0, 0, self.max_force_per_batch - force_labels.shape[0]),
                    self.force_pad_value,
                )(force_labels)

            new_labels.update(
                {
                    "target_labels": torch.concatenate(energy_labels, dim=0) * n_atoms,
                    "force_labels": force_labels,
                }
            )
            if self.validation_mode:
                new_labels.update(
                    {
                        "n_atoms": torch.tensor([len(f[3]) for f in features]),
                    }
                )

        super_output["labels"] = new_labels

        if self.data_bf16:
            super_output = self.to_bf16(super_output)

        return super_output

