"""
spiel.nn.problem_defs

Module for tensor2tensor problem definitions
"""
__all__ = ['proto_algonquian', 'southwestern_ojibwe', 'aztec']

from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
from . import proto_algonquian, southwestern_ojibwe, aztec


@registry.register_hparams
def transformer_spiel():
    """
    Hyperparameters for basic SPieL training
    """
    hparams = transformer.transformer_base()
    hparams.num_hidden_layers = 2
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 4
    hparams.attention_dropout = 0.6
    hparams.layer_prepostprocess_dropout = 0.6
    hparams.learning_rate = 0.05
    return hparams


@registry.register_hparams
def transformer_spiel_single_gpu():
    """
    Hyperparameters for basic SPieL training
    """
    hparams = transformer.transformer_base_single_gpu()
    hparams.num_hidden_layers = 2
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 4
    hparams.attention_dropout = 0.6
    hparams.layer_prepostprocess_dropout = 0.6
    hparams.learning_rate = 0.05
    return hparams


@registry.register_ranged_hparams
def transformer_spiel_range(rhp):
    """
    Hyperparameters that should be tuned
    """
    rhp.set_float('learning_rate', 0.01, 0.2, scale=rhp.LOG_SCALE)
    rhp.set_int('num_hidden_layers', 2, 4)
    rhp.set_discrete('hidden_size', [128, 256, 512])
    rhp.set_float('attention_dropout', 0.4, 0.7)
