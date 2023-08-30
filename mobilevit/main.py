import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
from typing import *
from ml_collections import ConfigDict
from .blocks import MobileVITLayer, MobileNetBlock, MobileNetBlockV2, MobileViTV2Layer
from .layers import act_layer_factory, norm_layer_factory, MobileViTConvLayer, MobileViTV2ConvLayer



class MobileVITModel(tf.keras.Model):
  def __init__(self,
               config: ConfigDict,
               expand_output: bool = True,
               **kwargs) -> None:

    super(MobileVITModel, self).__init__(**kwargs)
    self.config = config
    self.expand_output = expand_output
    self.num_classes = config.num_classes

    # segmentation architectures like DeepLab and PSPNet modify the strides
    # of the classification backbones
    dilate_layer_4 = dilate_layer_5 = False
    if config.output_stride == 8:
      dilate_layer_4 = True
      dilate_layer_5 = True
    elif config.output_stride == 16:
      dilate_layer_5 = True

    dilation = 1

    self.conv_stem = MobileViTConvLayer(
          config=config,
          out_channels=config.neck_hidden_sizes[0],
          kernel_size=3,
          stride=2,
          name="stem"
        )

    self._layers = []

    # layer1 -> mobilenet-v2 block(no downsampling)
    self.layer_1 = MobileNetBlock(
        config = config,
        in_channels = config.neck_hidden_sizes[0],
        out_channels = config.neck_hidden_sizes[1],
        stride = 1,
        num_stages = 1,
        name = "layer_1"
      )
    self._layers.append(self.layer_1)

    # layer_2 -> mobilenet-v2 block(downsampling)
    self.layer_2 = MobileNetBlock(
        config = config,
        in_channels = config.neck_hidden_sizes[1],
        out_channels = config.neck_hidden_sizes[2],
        stride = 2,
        num_stages = 3,
        name = "layer_2"
      )
    self._layers.append(self.layer_2)

    # layer_3 -> mobilevit block
    self.layer_3 = MobileVITLayer(
        config = config,
        in_channels = config.neck_hidden_sizes[2],
        out_channels = config.neck_hidden_sizes[3],
        hidden_size = config.hidden_sizes[0],
        stride = 2,
        num_stages = 2,
        name = "layer_3"
      )
    self._layers.append(self.layer_3)

    if dilate_layer_4:
      dilation *= 2

    # layer_4 -> mobilevit block
    self.layer_4 = MobileVITLayer(
        config = config,
        in_channels = config.neck_hidden_sizes[3],
        out_channels = config.neck_hidden_sizes[4],
        hidden_size = config.hidden_sizes[1],
        stride = 2,
        num_stages = 4,
        dilation=dilation,
        name = "layer_4"
      )
    self._layers.append(self.layer_4)

    if dilate_layer_5:
      dilation *= 2

    # layer_5 -> mobilevit block
    self.layer_5 = MobileVITLayer(
        config = config,
        in_channels = config.neck_hidden_sizes[4],
        out_channels = config.neck_hidden_sizes[5],
        hidden_size = config.hidden_sizes[2],
        stride = 2,
        num_stages = 3,
        dilation=dilation,
        name = "layer_5"
      )
    self._layers.append(self.layer_5)

    if self.expand_output:
      self.conv_1x1_exp = MobileViTConvLayer(
            config=config,
            out_channels=config.neck_hidden_sizes[6],
            kernel_size=1,
            name="conv_expand"
         )

    self.gap = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_last", name="pooler")

    if self.num_classes > 0:
      self.head = tf.keras.layers.Dense(self.num_classes, name='classification_head')

  def call(self, features: tf.Tensor, training=False) -> tf.Tensor:

    ## conv_stem
    embedding_outputs = self.conv_stem(features, training=training)

    # layer1 -> layer5 3 encoded output
    for indx, layer_module in enumerate(self._layers):
      embedding_outputs = layer_module(embedding_outputs, training=training)

    if self.expand_output:
      last_hidden_state = self.conv_1x1_exp(embedding_outputs)

    # global average pooling: (batch_size, height, width, channels) -> (batch_size, channels)
    pooled_output = self.gap(last_hidden_state)

    if self.num_classes > 0:
      return self.head(pooled_output)

    return pooled_output


class MobileVITV2Model(tf.keras.models.Model):
  def __init__(self, config: ConfigDict, expand_output: bool = True, **kwargs):
    super(MobileVITV2Model, self).__init__(**kwargs)

    self.config = config
    self._layers = []
    self.num_classes = config.num_classes
    self.expand_output = expand_output

    # segmentation architectures like DeepLab and PSPNet modify the strides
    # of the classification backbones
    dilate_layer_4 = dilate_layer_5 = False
    if config.output_stride == 8:
      dilate_layer_4 = True
      dilate_layer_5 = True
    elif config.output_stride == 16:
      dilate_layer_5 = True

    dilation = 1

    layer_1_dim = make_divisible(64 * config.width_multiplier, divisor=16)
    layer_2_dim = make_divisible(128 * config.width_multiplier, divisor=8)
    layer_3_dim = make_divisible(256 * config.width_multiplier, divisor=8)
    layer_4_dim = make_divisible(384 * config.width_multiplier, divisor=8)
    layer_5_dim = make_divisible(512 * config.width_multiplier, divisor=8)

    print(clip(value=32 * config.width_multiplier, min_val=16, max_val=64))
    layer_0_dim = make_divisible(
            clip(value=32 * config.width_multiplier, min_val=16, max_val=64), divisor=8, min_value=16
        )

    self.conv_stem = MobileViTV2ConvLayer(
          config=config,
          in_channels=config.num_channels,
          out_channels=layer_0_dim,
          kernel_size=3,
          stride=2,
          use_norm=True,
          use_act=True,
          name="stem"
        )

    layer_1 = MobileNetBlockV2(
            config,
            in_channels=layer_0_dim,
            out_channels=layer_1_dim,
            stride=1,
            num_stages=1,
        )
    self._layers.append(layer_1)

    layer_2 = MobileNetBlockV2(
            config,
            in_channels=layer_1_dim,
            out_channels=layer_2_dim,
            stride=2,
            num_stages=2,
        )
    self._layers.append(layer_2)

    layer_3 = MobileViTV2Layer(
            config,
            in_channels=layer_2_dim,
            out_channels=layer_3_dim,
            attn_unit_dim=make_divisible(config.base_attn_unit_dims[0] * config.width_multiplier, divisor=8),
            n_attn_blocks=config.n_attn_blocks[0],
        )
    self._layers.append(layer_3)

    if dilate_layer_4:
      dilation *= 2

    layer_4 = MobileViTV2Layer(
            config,
            in_channels=layer_3_dim,
            out_channels=layer_4_dim,
            attn_unit_dim=make_divisible(config.base_attn_unit_dims[1] * config.width_multiplier, divisor=8),
            n_attn_blocks=config.n_attn_blocks[1],
            dilation=dilation,
        )
    self._layers.append(layer_4)

    if dilate_layer_5:
      dilation *= 2

    layer_5 = MobileViTV2Layer(
        config,
        in_channels=layer_4_dim,
        out_channels=layer_5_dim,
        attn_unit_dim=make_divisible(config.base_attn_unit_dims[2] * config.width_multiplier, divisor=8),
        n_attn_blocks=config.n_attn_blocks[2],
        dilation=dilation,
      )
    self._layers.append(layer_5)
    
    if self.expand_output:
      self.conv_1x1_exp = tf.identity


    self.gap = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_last", name="pooler")

    if self.num_classes > 0:
      self.head = tf.keras.layers.Dense(self.num_classes, name='classification_head')

  def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
    ## conv_stem
    embedding_outputs = self.conv_stem(hidden_states, training=training)

    for layer_module in self._layers:
      embedding_outputs = layer_module(embedding_outputs, training=training)

    if self.expand_output:
      last_hidden_state = self.conv_1x1_exp(embedding_outputs)

    # global average pooling: (batch_size, height, width, channels) -> (batch_size, channels)
    pooled_output = self.gap(last_hidden_state)

    if self.num_classes > 0:
      return self.head(pooled_output)

    return pooled_output


def clip(value: float, min_val: float = float("-inf"), max_val: float = float("inf")) -> float:
    return max(min_val, min(max_val, value))


# Copied from transformers.models.mobilevit.modeling_mobilevit.make_divisible
def make_divisible(value: int, divisor: int = 8, min_value: Optional[int] = None) -> int:
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