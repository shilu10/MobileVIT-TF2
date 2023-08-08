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
