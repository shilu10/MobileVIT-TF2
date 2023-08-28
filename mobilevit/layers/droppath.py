import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
from typing import * 


class StochasticDepth(tf.keras.layers.Layer):
    def __init__(self,
                 drop_prop: float = 0.0,
                 **kwargs)-> tf.keras.layers.Layer:

        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_prob = drop_prop

    def call(self,
             x: tf.Tensor,
             training: bool = None) -> tf.Tensor:
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

    def get_config(self):
      config = super().get_config()
      #config['drop_prob'] = self.drop_prob

      return config

    @classmethod
    def from_config(cls, config):
      return cls(**config)