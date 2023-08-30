import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
from typing import * 
from ml_collections import ConfigDict 
from ..layers import LayerScale, StochasticDepth, MLP, act_layer_factory, norm_layer_factory
import os, sys, math


class SeperableSelfAttention(tf.keras.layers.Layer):
  def __init__(self, config: ConfigDict, embed_dim: int, **kwargs) -> None:
    super(SeperableSelfAttention, self).__init__(**kwargs)

    self.qkv_proj = MobileViTV2ConvLayer(
            config=config,
            in_channels=embed_dim,
            out_channels=1 + (2 * embed_dim),
            bias=True,
            kernel_size=1,
            use_norm=False,
            use_act=False,
        )

    self.attn_dropout = tf.keras.layers.Dropout(config.attn_dropout)

    self.out_proj = MobileViTV2ConvLayer(
            config=config,
            in_channels=embed_dim,
            out_channels=embed_dim,
            bias=True,
            kernel_size=1,
            use_norm=False,
            use_act=False,
        )
    self.embed_dim = embed_dim

  def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
    # (batch_size, , num_pixels_in_patch, num_patches, embed_dim ) --> (batch_size, num_pixels_in_patch num_patches, 1+2*embed_dim)
    qkv = self.qkv_proj(hidden_states)

    # Project hidden_states into query, key and value
    # Query --> [batch_size, num_pixels_in_patch, num_patches, 1]
    # value, key --> [batch_size, num_pixels_in_patch, num_patches, embed_dim]
    query, key, value = tf.split(qkv, [1, self.embed_dim, self.embed_dim], axis=-1)

    # apply softmax along num_patches dimension
    context_scores = tf.nn.softmax(query, axis=2)
    context_scores = self.attn_dropout(context_scores)

    # Compute context vector
    # [batch_size, num_pixels_in_patch, num_patches, embed_dim] x [batch_size, num_pixels_in_patch, num_patches, 1] -> [batch_size, num_pixels_in_patch, num_patches, embed_dim]
    context_vector = key * context_scores
    # [B, P, N, d] --> [B, P, 1, d]
    context_vector = tf.math.reduce_sum(context_vector, axis=2, keepdims=True)

    # [batch_size, num_pixels_in_patch, num_patches, embed_dim] x [batch_size, num_pixels_in_patch, num_patches, 1] -> [batch_size, num_pixels_in_patch, num_patches, embed_dim]
    value = tf.nn.relu(value)
    #out = tf.einsum("...nd, ...kd->...nd", value, context_vector)
    #out = tf.keras.activation.relu(value) * context_vector
    context_vector = tf.broadcast_to(context_vector, value.shape)
    out = context_vector * value
    out = self.out_proj(out)
    return out


class MobileViTV2FFN(tf.keras.layers.Layer):
  def __init__(self,
               config: ConfigDict,
               embed_dim: int,
               ffn_latent_dim: int,
               ffn_dropout: float = 0.0,
               **kwargs):

    super(MobileViTV2FFN, self).__init__(**kwargs)

    self.conv1 = MobileViTV2ConvLayer(
            config=config,
            in_channels=embed_dim,
            out_channels=ffn_latent_dim,
            kernel_size=1,
            stride=1,
            bias=True,
            use_norm=False,
            use_act=True,
        )

    self.dropout1 = tf.keras.layers.Dropout(ffn_dropout)

    self.conv2 = MobileViTV2ConvLayer(
            config=config,
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=1,
            stride=1,
            bias=True,
            use_norm=False,
            use_act=False,
        )
    self.dropout2 = tf.keras.layers.Dropout(ffn_dropout)

  def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
    hidden_states = self.conv1(hidden_states)
    hidden_states = self.dropout1(hidden_states)
    hidden_states = self.conv2(hidden_states)
    hidden_states = self.dropout2(hidden_states)
    return hidden_states


class MobileViTV2TransformerLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        config: ConfigDict,
        embed_dim: int,
        ffn_latent_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.layernorm_before = tf.keras.layers.GroupNormalization(groups=1, epsilon=config.layer_norm_eps)
        self.attention = SeperableSelfAttention(config, embed_dim)
        self.dropout1 = tf.keras.layers.Dropout(rate=dropout)
        self.layernorm_after = tf.keras.layers.GroupNormalization(groups=1, epsilon=config.layer_norm_eps)
        self.ffn = MobileViTV2FFN(config, embed_dim, ffn_latent_dim, config.ffn_dropout)

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        layernorm_1_out = self.layernorm_before(hidden_states)
        attention_output = self.attention(layernorm_1_out)
        hidden_states = attention_output + hidden_states

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.ffn(layer_output)

        layer_output = layer_output + hidden_states
        return layer_output