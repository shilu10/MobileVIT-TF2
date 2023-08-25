import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
from typing import *


class Conv_1x1_bn(tf.keras.layers.Layer):
  def __init__(self,
              in_channels=32,
              out_channels=64,
              kernel_size=1,
              stride=1,
              use_norm=True,
              use_act=True,
               **kwargs
          ):
    super(Conv_1x1_bn, self).__init__(**kwargs)
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.use_norm = use_norm
    self.use_act = use_act

    self.conv = tf.keras.layers.Conv2D(
        filters = out_channels,
        kernel_size = self.kernel_size,
        strides = self.stride,
        padding = "valid",
        use_bias = False
    )

    if use_norm:
      self.norm = tf.keras.layers.BatchNormalization()

    if use_act:
      self.activation = tf.keras.layers.Activation(tf.nn.silu)

  def call(self, x):
    x = self.conv(x)

    if self.use_norm:
      x = self.norm(x)

    if self.use_act:
      x = self.activation(x)

    return x

  def get_config(self):
    config = super(Conv_1x1_bn, self).get_config()
    config["in_channels"] = self.in_channels
    config["kernel_size"] = self.kernel_size
    config['stride'] = self.stride
    config['use_norm'] = self.use_norm
    config['use_act'] = self.use_act
    config["out_channels"] = self.out_channels

    return config


class Conv_3x3_bn(tf.keras.layers.Layer):
  def __init__(self,
              in_channels=32,
              out_channels=64,
              kernel_size=3,
              stride=1,
              use_norm=True,
              use_act=True,
              groups=None,
              dilation=None,
              **kwargs,
          ):
    super(Conv_3x3_bn, self).__init__(**kwargs)
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.use_norm = use_norm
    self.use_act = use_act

   
    self.conv = tf.keras.layers.Conv2D(
          filters = out_channels,
          kernel_size = self.kernel_size,
          strides = self.stride,
          padding = "same",
          use_bias = False,
          groups = 1 if groups is None else groups,
          dilation_rate = 1 if dilation is None else dilation,
      )



    if use_norm:
      self.norm = tf.keras.layers.BatchNormalization()

    if use_act:
      self.activation = tf.keras.layers.Activation(tf.nn.silu)

  def call(self, x):
    x = self.conv(x)

    if self.use_norm:
      x = self.norm(x)

    if self.use_act:
      x = self.activation(x)

    return x

  def get_config(self):
    config = super(Conv_3x3_bn, self).get_config()
    config["in_channels"] = self.in_channels
    config["kernel_size"] = self.kernel_size
    config['stride'] = self.stride
    config['use_norm'] = self.use_norm
    config['use_act'] = self.use_act
    config["out_channels"] = self.out_channels

    return config


## new 
class MobileViTConvLayer(tf.keras.layers.Layer):
  def __init__(self, 
               out_channels: int, 
               kernel_size: int | Tuple, 
               config: ConfigDict,
               strides: int | Tuple = 1, 
               groups: int = 1, 
               bias: bool = False, 
               dilation: int = 1, 
               use_norm: bool = True, 
               use_act: Union[bool, str] = True, 
               **kwargs)-> None:
    super(MobileViTConvLayer, self).__init__(**kwargs)
    self.config = config 
      
    padding = int((kernel_size - 1) / 2) * dilation
    self.padding = tf.keras.layers.ZeroPadding2D(padding)

    if out_channels % groups != 0:
      raise ValueError(f"Output channels ({out_channels}) are not divisible by {groups} groups.")

    self.conv = tf.keras.layers.Conv2D(
                    kernel_size = kernel_size,
                    filters = out_channels,
                    strides = strides,
                    groups = groups,
                    use_bias = bias,
                    dilation_rate = dilation
              )
    
    self.norm = None
    if use_norm: 
      self.norm = tf.keras.layers.BatchNormalization(epsilon=1e-5, 
                                                     momentum=0.1, 
                                                     name="normalization")

    self.activation = None
    if use_act:
      if isinstance(use_act, str):
        _act = act_layer_factory(use_act)
        self.activation = _act()
      elif isinstance(config.hidden_act, str):
        _act = act_layer_factory(config.hidden_act)
        self.activation = _act()
      else:
        self.activation = config.hidden_act 

    self.use_norm = use_norm
    self.use_act = use_act 

  def call(self, 
           x: tf.Tensor, 
           training: bool = False) -> tf.Tensor:

    padding_x = self.padding(x)
    x = self.conv(padding_x)

    if self.use_norm:
      x = self.norm(x)

    if self.use_act:
      x = self.activation(x)
    
    return x 

  def get_config(self)->Dict:
    config = super().get_config()
    config.update(
        {
            "use_norm": self.use_norm,
            'use_act': self.use_act,
        }
    )

    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)
