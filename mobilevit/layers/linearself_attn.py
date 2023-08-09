from tensorflow import keras 
import tensorflow as tf 
import numpy as np 
from einops import rearrange 
from typing import * 


class LinearSelfAttention(tf.keras.layers.Layer):
    """
    This layer applies a self-attention with linear complexity, as described in `https://arxiv.org/abs/2206.02680`
    This layer can be used for self- as well as cross-attention.
    Args:
        embed_dim (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        attn_drop (float): Dropout value for context scores. Default: 0.0
        bias (bool): Use bias in learnable layers. Default: True
    Shape:
        - Input: :math:`(N, C, P, N)` where :math:`N` is the batch size, :math:`C` is the input channels,
        :math:`P` is the number of pixels in the patch, and :math:`N` is the number of patches
        - Output: same as the input
    .. note::
        For MobileViTv2, we unfold the feature map [B, C, H, W] into [B, C, P, N] where P is the number of pixels
        in a patch and N is the number of patches. Because channel is the first dimension in this unfolded tensor,
        we use point-wise convolution (instead of a linear layer). This avoids a transpose operation (which may be
        expensive on resource-constrained devices) that may be required to convert the unfolded tensor from
        channel-first to channel-last format in case of a linear layer.
    """

    def __init__(
        self,
        embed_dim: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bias: bool = True,
    ) -> None:

        super(LinearSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.attn_drop = attn_drop 
        self.proj_drop = proj_drop
        self.bias = bias 

        self.qkv = tf.keras.layers.Conv2D(
            filters=1 + (2 * embed_dim),
            use_bias=bias,
            kernel_size=1,
        )

        self.attn_drop = tf.keras.layers.Dropout(rate=attn_drop)
        self.proj = tf.keras.layers.Conv2D(
            filters=embed_dim,
            use_bias=bias,
            kernel_size=1,
        )
        self.proj_dropout = tf.keras.layers.Dropout(rate=proj_drop)

    def _forward_self_attn(self, x: tf.Tensor) -> tf.Tensor:
        b, p, n, c = x.shape
        # [B, P, N, C] --> [B,, P, N,  h + 2d]
        qkv = self.qkv(x)

        # Project x into query, key and value
        # Query --> [B, P, N, 1]
        # value, key --> [B P, N, d]
        query, key, value = tf.split(qkv, [1, self.embed_dim, self.embed_dim], axis=-1)

        # apply softmax along N dimension
        context_scores = tf.nn.softmax(query, axis=-1)
        context_scores = self.attn_drop(context_scores)

        # Compute context vector
        # [B, N, P, d] x [B, N, P, d] -> [B, N, P, d] --> [B, 1, P, d]
        context_vector = key * context_scores
        context_vector = tf.math.reduce_sum(context_vector, axis=2, keepdims=True)

        # combine context vector with values
        # [B, N, P, d] * [B, 1, P, d] --> [B, N, P, d]
        value = tf.nn.relu(value)
        value = value * context_vector
        value = tf.broadcast_to(value, (b, p, n, self.embed_dim))
        out = self.proj(value)
        out = self.proj_dropout(out)
        return out

    def _forward_cross_attn(self, x: tf.Tensor, x_prev: Optional[tf.Tensor] = None) -> tf.Tensor:
        pass 

    def call(self, x: tf.Tensor, x_prev: Optional[tf.Tensor] = None) -> tf.Tensor:

        if x_prev is None:
            return self._forward_self_attn(x)
        else:
            return self._forward_cross_attn(x, x_prev=x_prev)

    def get_config(self):
        config = super().get_config()

        config['embed_dim'] = self.embed_dim
        config['attn_drop'] = self.attn_drop
        config['proj_drop'] = self.proj_drop
        config['bias'] = self.bias 

        return config 
