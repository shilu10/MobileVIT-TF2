import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
from typing import * 
from .factory import act_layer_factory, norm_layer_factory
from ml_collections import ConfigDict


class MLP(tf.keras.layers.Layer):
    def __init__(self,
               config: ConfigDict, 
               hidden_units = [768, 192],
               **kwargs):
        super(MLP, self).__init__(**kwargs)
        act_layer = act_layer_factory(config.hidden_act)

        self.fc_1  = tf.keras.layers.Dense(units=hidden_units[0])
        self.activation = act_layer()
        self.fc_2 = tf.keras.layers.Dense(units=hidden_units[1])
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

        self.hidden_units = hidden_units

    def call(self, x: tf.Tensor, training: bool =False):
        x = self.fc_1(x)
        x = self.activation(x)
        x = self.fc_2(x)
        x = self.dropout(x)
        
        return x

    def get_config(self):
        config = super(MLP, self).get_config()
        config["hidden_units"] = self.hidden_units
        
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)