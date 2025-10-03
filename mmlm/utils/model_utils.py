from mmlm.models.continuous_model import ContinuousModelForCausalLM
from mmlm.models.continuous_config import ContinuousLlamaConfig, ContinuousQwen3Config
from mmlm.models.gradient_model import GradientModel
from mmlm.models.pos_readout_model import PositionReadoutModel, PCQPositionReadoutModel
from mmlm.models.dir_readout_model import DirReadoutModel
from mmlm.models.prefix_model import PrefixReadoutModel

name_to_config_and_class = {
    "llama_continuous": (ContinuousLlamaConfig, ContinuousModelForCausalLM),
    "llama_grad": (ContinuousLlamaConfig, GradientModel),
    "llama_pos_readout": (ContinuousLlamaConfig, PositionReadoutModel),
    "llama_prefix_readout": (ContinuousLlamaConfig, PrefixReadoutModel),
    "llama_dir_readout": (ContinuousLlamaConfig, DirReadoutModel),
    "qwen3_pos_readout": (ContinuousQwen3Config, PositionReadoutModel),
    "qwen3_pcq_pos_readout": (ContinuousQwen3Config, PCQPositionReadoutModel),
    "qwen3_continuous": (ContinuousQwen3Config, ContinuousModelForCausalLM),
    "qwen3_grad": (ContinuousQwen3Config, GradientModel),
}


def get_config_and_class(name):
    return name_to_config_and_class[name]
