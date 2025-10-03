# Transformers Discover Molecular Structure Without Graph Priors
<p align="center">
  <a href="https://arxiv.org/abs/2510.02259">
    <img src="https://img.shields.io/badge/arXiv-b31b1b?style=for-the-badge&logo=arxiv" alt="arXiv"/>
  </a>
  <a href="https://x.com/ask1729/status/1973923449019249092">
    <img src="https://img.shields.io/badge/X%20Thread-1DA1F2?logo=x&logoColor=white&style=for-the-badge" alt="X Thread"/>
  </a>
</p>
This is the implementation for "Transformers Discover Molecular Structure Without Graph Priors". We plan to continue rolling out improvements and updates to the code.


## Environment Setup

```bash
mamba env create -f env_simple.yml
mamba activate graph-free
pip install -e .
```

## Example Train Command

The configs expect data to be in a data folder in the directory of the repo (data/Omol/ for example). The path to data can be modified in the configs. Logs will be written to exp_logs. OMol data can be downloaded from [here](https://huggingface.co/facebook/OMol25).

```bash
python -m mmlm.train +models=llama_57M_ch +omol_scaling_experiments=model_scaling wandb.group_name=omol_model_scaling wandb.run_name=57M training.batch_size=32 training.gradient_accumulation_steps=8
```

Note that metadata files (energy and force mean/std) can be found [here](https://drive.google.com/file/d/17YvBVwQmr-VQFSJgSJbYwq2Z4Unma7Qw/view?usp=sharing).