import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
from ml_collections import ConfigDict 
from typing import * 
from .conv_layers import MobileViTConvLayer
from .factory import act_layer_factory, norm_layer_factory


class InvertedResidualLayer(tf.keras.layers.Layer):
  def __init__(self,
               config: ConfigDict,
               in_channels: int,
               out_channels: int,
               stride: int,
               dilation: int = 1,
               **kwargs) -> None:
    super(InvertedResidualLayer, self).__init__(**kwargs)

    self.config = config
    expanded_channels = make_divisible(int(round(in_channels * config.expand_ratio)), 8)

    if stride not in [1, 2]:
      raise ValueError(f"Invalid stride {stride}.")

    self.use_residual = (stride == 1) and (in_channels == out_channels)

    # pw
    # down-sample in the first conv
    self.expand_1x1 = MobileViTConvLayer(
            expanded_channels, 1, config
        )

    # DW
    self.conv_3x3 = MobileViTConvLayer(
            config=config,
            out_channels=expanded_channels,
            kernel_size=3,
            stride=stride,
            groups=expanded_channels,
            dilation=dilation,
        )

    # pw
    # up-smaple in last conv
    self.reduce_1x1 = MobileViTConvLayer(
            config=config,
            out_channels=out_channels,
            kernel_size=1,
            use_act=False,
        )

    self.in_channels = in_channels
    self.out_channels = out_channels

  def call(self,
           x: tf.Tensor,
            training: bool = False) -> tf.Tensor:

    # shortcut for the resiudal
    shortcut = x

    x = self.expand_1x1(x, training=training)
    x = self.conv_3x3(x, training=training)
    x = self.reduce_1x1(x, training=training)

    return shortcut + x if self.use_residual else x

  def get_config(self) -> Dict:
    config = super().get_config()
    config.update(
        {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels
        }
      )

    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)


def make_divisible(value: int,
                   divisor: int = 8,
                   min_value: Optional[int] = None) -> int:
    """
    Ensure that all layers have a channel count that is divisible by `divisor`. This function is taken from the
    original TensorFlow repo. It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)
