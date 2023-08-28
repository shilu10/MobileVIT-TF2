import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
from typing import * 
from ml_collections import ConfigDict 
from .vision_transformer import TFVITTransformerLayer


class MobileViTTransformer(tf.keras.Model):
    def __init__(self,
                 config: ConfigDict,
                 hidden_size: int,
                 num_stages: int,
                 **kwargs) -> None:

        super(MobileViTTransformer, self).__init__(**kwargs)

        self._layers = []
        for i in range(num_stages):
            transformer_layer = TFVITTransformerLayer(
                config,
                hidden_size=hidden_size,
                mlp_units=[int(hidden_size * config.mlp_ratio), hidden_size],
                name=f"layer.{i}",
            )
            self._layers.append(transformer_layer)

        self.hidden_size = hidden_size
        self.num_stages = num_stages

    def call(self,
             hidden_states: tf.Tensor,
             training: bool = False) -> tf.Tensor:

        for layer_module in self._layers:
            hidden_states = layer_module(hidden_states, training=training)
        return hidden_statess

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                'hidden_size': self.hidden_size,
                'num_stages': self.num_stages
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)