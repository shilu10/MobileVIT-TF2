import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
from typing import * 


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self,
                 embed_dim,
                 output_dim=None,
                 num_heads=8,
                 head_dims=None,
                 attn_drop=0.,
                 proj_drop=0.,  # ffn dropout rate
                 drop_rate=0.,
                 attn_bias=True,
              ):

        super(MultiHeadSelfAttention, self).__init__()
        if output_dim is None:
            output_dim = embed_dim

        assert embed_dim % num_heads == 0, "embed_dim % num_head should be 0"

        # attn layers
        self.qkv = tf.keras.layers.Dense(
            units=embed_dim*3,
            use_bias=attn_bias,
        )

        self.attn_dropout = tf.keras.layers.Dropout(attn_drop)

        # proj layers
        self.proj = tf.keras.layers.Dense(
            units=output_dim,
            use_bias=attn_bias,
        )

        self.proj_dropout = tf.keras.layers.Dropout(proj_drop)

        if head_dims is None:
          self.head_dims = embed_dim // num_heads

        self.scaling = self.head_dims**-0.5
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def __repr__(self):
        return "{}(head_dim={}, num_heads={}, attn_dropout={})".format(
            self.__class__.__name__, self.head_dims, self.num_heads, self.attn_dropout.rate
        )

    def call(self, x, training=False):
        #b, n, c = tf.shape(x)
        b, n, c = x.shape
        qkv = self.qkv(x)

        qkv = tf.reshape(qkv, shape=(-1, n, 3, self.num_heads, self.head_dims))
        qkv = tf.transpose(qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = tf.unstack(qkv)

        q = q * self.scaling

        attn_scores = tf.matmul(q, k, transpose_b=True)
        attn_probs = tf.nn.softmax(attn_scores)
        attn_probs = self.attn_dropout(attn_probs, training=training)

        out = tf.matmul(attn_probs, k)
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, shape=(-1, n, c))

        x = self.proj(x)
        x = self.proj_dropout(x, training=training)
        return x

 	def get_config(self):
 		config = super().get_config()

 		config['scaling'] = self.scaling
 		config['num_heads'] = self.num_heads
 		config['embed_dim'] = self.embed_dim

 		return config