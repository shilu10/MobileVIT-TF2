import numpy as np 
from ml_collections import ConfigDict 
from typing import * 



def get_base_config(include_top: bool = True, 
                    hidden_sizes: Union(List, Tuple) = [144, 192, 240],
                    neck_hidden_sizes: Union(List, Tuple) = [16, 32, 64, 96, 128, 160, 640], 
                    expand_ratio: float = 4.0
                ):

    config = ConfigDict()
    config.num_classes = 1000 if include_top else 0 
    config.mlp_ratio = 2.0 
    config.hidden_act = "silu"
    config.init_values = 0.0
    config.layer_norm_eps = 1e-5
    config.hidden_dropout_prob = 0.0
    config.attention_probs_dropout_prob = 0.0
    config.qkv_bias = True
    config.num_heads = 4 
    config.expand_ratio = expand_ratio
    config.output_stride = 32
    config.patch_size = 2 
    config.conv_kernel_size = 3  
    config.hidden_sizes = hidden_sizes
    config.neck_hidden_sizes = neck_hidden_sizes

    return config.lock()