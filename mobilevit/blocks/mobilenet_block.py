import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
from collections import * 
from ml_collections import ConfigDict 
from typing import * 



class MobileNetBlock(tf.keras.Model):
  def __init__(self, 
               config: ConfigDict, 
               in_channels: int, 
               out_channels: int, 
               stride: int = 1, 
               num_stages: int = 1, 
               **kwargs) ->None:

    super(MobileNetBlock, self).__init__(**kwargs)
    self.config = config 

    layers = []
    for indx in range(num_stages):
      layers.append(
          InvertedResidualLayer(
              config = config, 
              in_channels = in_channels,
              out_channels = out_channels,
              stride = stride if indx == 0 else 1,
              name = f"mobilenetlayer_{indx}"
          )
        )
      in_channels = out_channels 
  
    self.layers = layers 
    self.out_channels = out_channels 
    self.stride = stride
    self.num_stages = num_stages 

  def call(self, 
           x: tf.Tensor, 
           training: bool = False) -> tf.Tensor:
  
    for layer_module in self.layers:
      x = layer_module(x, training=training)

    return x 

  def get_config(self) -> Dict:
    config = super().get_config()
    config.update(
        {
            'out_channels': self.out_channels,
            'stride': self.stride,
            'num_stages': self.num_stages,
        }
      )
    
    return config 

  @classmethod
  def from_config(cls, config):
    return cls(**config)