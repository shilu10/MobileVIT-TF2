import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
from typing import * 
from ..layers import MultiHeadSelfAttention, MLP, LayerScale, DropPath
from ..layers import norm_layer_factory, act_layer_factory



class Transformer(tf.keras.layers.Layer):
  def __init__(self, 
              embed_dim, 
              ffn_latent_dim,
              num_heads, 
              head_dims=None, 
              qkv_bias=True, 
              attn_drop=0., 
              proj_drop=0., 
              drop_path=0.0,
              init_values=None, 
              norm_layer="layer_norm", 
              act_layer="gelu",
              **kwargs
          ):
      
    super(Transformer, self).__init__(**kwargs)
    norm_layer = norm_layer_factory(norm_layer)

    self.attn = MultiHeadSelfAttention(
          embed_dim=embed_dim,
          num_heads=num_heads,
          attn_bias=qkv_bias,
          attn_drop=attn_drop,
          proj_drop=proj_drop,
        )
        
    self.mlp = MLP(
          hidden_dim=ffn_latent_dim,
          projection_dim=embed_dim,
          drop_rate=proj_drop,
          act_layer=act_layer,
          mlp_bias=True
        )
        
    self.ls_1 = LayerScale(embed_dim, init_values) if init_values else tf.identity
    self.ls_2 = LayerScale(embed_dim, init_values) if init_values else tf.identity

    self.drop_path_1 = DropPath(drop_path) if drop_path else tf.identity
    self.drop_path_2 = DropPath(drop_path) if drop_path else tf.identity

    self.norm_1 = norm_layer(name='transformer_norm1')
    self.norm_2 = norm_layer(name='transformer_norm2')

    self.embed_dim = embed_dim
    self.ffn_latent_dim = ffn_latent_dim 
    self.proj_drop = proj_drop
    self.attn_drop = attn_drop

  def call(self, x, training=False):
    # res1
    shortcut = x 

    x = self.norm_1(x)
    x = self.attn(x)
    x = self.ls_1(x)
    x = self.drop_path_1(x)

    x = x + shortcut 

    # res2
    shortcut = x

    x = self.norm_2(x)
    x = self.mlp(x)
    x = self.ls_2(x)
    x = self.drop_path_2(x)

    x = x + shortcut 

    return x 

  def __repr__(self):
    return "{}(embed_dim={}, ffn_latent_dim={}, proj_drop={}, attn_drop={}".format(
            self.__class__.__name__,
            self.embed_dim,
            self.ffn_latent_dim,
            self.proj_drop,
            self.attn_drop
      )
      
  def get_config(self):
    config = super(Transformer, self).get_config()
    config["embed_dim"] = self.embed_dim
    config['ffn_latent_dim'] = self.ffn_latent_dim
    config['proj_drop'] = self.proj_drop
    config['attn_drop'] = self.attn_drop

    return config


class LinearTransformerBlock(tf.keras.layers.Layer):
    """
    This class defines the pre-norm transformer encoder with linear self-attention in `MobileViTv2 paper <>`_
    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(B, C_{in}, P, N)`
        mlp_ratio (float): Inner dimension ratio of the FFN relative to embed_dim
        drop (float): Dropout rate. Default: 0.0
        attn_drop (float): Dropout rate for attention in multi-head attention. Default: 0.0
        drop_path (float): Stochastic depth rate Default: 0.0
        norm_layer (Callable): Normalization layer. Default: layer_norm_2d
    Shape:
        - Input: :math:`(B, C_{in}, P, N)` where :math:`B` is batch size, :math:`C_{in}` is input embedding dim,
            :math:`P` is number of pixels in a patch, and :math:`N` is number of patches,
        - Output: same shape as the input
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_latent_dim: int,
        attn_drop: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        norm_layer: Optional[str] = "layer_norm",
        drop_path: Optional[float] = 0.0,
        *args,
        **kwargs,
    ) -> None:

        super(LinearTransformerBlock, self).__init__()
        act_layer = "silu"
        norm_layer = norm_layer_factory(norm_layer)

        self.norm1 = norm_layer()
        self.attn = LinearSelfAttention(embed_dim=embed_dim, attn_drop=attn_drop, proj_drop=dropout)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else tf.identity

        self.norm2 = norm_layer()
        self.mlp = ConvMLP(
            projection_dim=embed_dim,
            hidden_dim=ffn_latent_dim,
            act_layer=act_layer,
            drop_rate=ffn_dropout)

        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else tf.identity

        self.embed_dim = embed_dim
        self.ffn_latent_dim = ffn_latent_dim
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

    def forward(self, x: tf.Tensor, x_prev: Optional[tf.Tensor] = None) -> tf.Tensor:
        if x_prev is None:
            # self-attention
            x = x + self.drop_path1(self.attn(self.norm1(x)))
        else:
            # cross-attention
            res = x
            x = self.norm1(x)  # norm
            x = self.attn(x, x_prev)  # attn
            x = self.drop_path1(x) + res  # residual

        # Feed forward network
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

    def get_config(self):
      config = super(Transformer, self).get_config()
      config["embed_dim"] = self.embed_dim
      config['ffn_latent_dim'] = self.ffn_latent_dim
      config['proj_drop'] = self.proj_drop
      config['attn_drop'] = self.attn_drop

      return config

