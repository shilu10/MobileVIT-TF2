import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
from typing import *
from .blocks import InvertedResidual, MobileViTBlock
from .layers import act_layer_factory, norm_layer_factory


class MobileVIT(tf.keras.models.Model):
  def __init__(self, config, classifier_dropout, num_classes, *args, **kwargs):
    super(MobileVIT, self).__init__()

    image_channels = 3
    out_channels = 16

   # mobilevit_config = get_configuration(model_variant)
    mobilevit_config = config

    self.model_conf_dict = dict()

    self.conv1 = Conv_3x3_bn(
          in_channels=image_channels,
          out_channels=out_channels,
          kernel_size=3,
          stride=2,
          use_act=True,
          use_norm=True,
          name="conv1_in"
    )

    self.model_conf_dict["conv1"] = {"in": image_channels, "out": out_channels}

    # stage1
    in_channels = out_channels
    self.layer_1, out_channels = self._make_layer(
            input_channel=in_channels,
            config=mobilevit_config["layer1"],
            name="stage_1"
        )

    self.model_conf_dict["layer1"] = {"in": in_channels, "out": out_channels}

    # stage2
    in_channels = out_channels
    self.layer_2, out_channels = self._make_layer(
            input_channel=in_channels,
            config=mobilevit_config["layer2"],
            name="stage_2"
        )

    self.model_conf_dict["layer2"] = {"in": in_channels, "out": out_channels}

    # stage3
    in_channels = out_channels
    self.layer_3, out_channels = self._make_layer(
            input_channel=in_channels,
            config=mobilevit_config["layer3"],
            name="stage_3"
        )

    self.model_conf_dict["layer3"] = {"in": in_channels, "out": out_channels}

    # stage4
    in_channels = out_channels
    self.layer_4, out_channels = self._make_layer(
            input_channel=in_channels, config=mobilevit_config["layer4"], name="stage_4"
        )

    self.model_conf_dict["layer4"] = {"in": in_channels, "out": out_channels}

    # stage5
    in_channels = out_channels
    self.layer_5, out_channels = self._make_layer(
            input_channel=in_channels,
            config=mobilevit_config["layer5"],
            name="stage_5"
        )

    self.model_conf_dict["layer5"] = {"in": in_channels, "out": out_channels}

    # other layers
    in_channels = out_channels
    exp_channels = min(mobilevit_config["last_layer_exp_factor"] * in_channels, 960)
    self.conv_1x1_exp = Conv_1x1_bn(
          in_channels=in_channels,
          out_channels=exp_channels,
          kernel_size=1,
          stride=1,
          use_act=True,
          use_norm=True,
          name="conv_exp"
      )

    self.model_conf_dict["exp_before_cls"] = {
          "in": in_channels,
          "out": exp_channels,
      }

    self.classifier = tf.keras.Sequential(name="classifier")
    self.classifier.add(GlobalAveragePooling2D())

    if 0.0 < classifier_dropout < 1.0:
      self.classifier.add(Dropout(rate=classifier_dropout))

    self.classifier.add(Dense(units=num_classes, use_bias=True))

  def _make_layer(self, input_channel, config, name):
    # block_type
    block_type = config.get('block_type')

    if block_type.lower() == "mv2":  # mobilenetv2 block
      return self.build_mv2_block(input_channel, config, name)

    else:
      return self.build_mobilevit_block(input_channel, config, name)

  def build_mobilevit_block(self, input_channels, config, name):
    block = []
    stride = config.get("stride", 1)

    if stride == 2:
      layer = InvertedResidual(
            inp_dims=input_channels,
            out_dims=config.get("out_channels"),
            stride=stride,
            expansion=config.get("mv_expand_ratio", 4),
          )

      block.append(layer)
      input_channels = config.get("out_channels")

    head_dim = config.get("head_dim", 32)
    transformer_dim = config["transformer_channels"]
    ffn_dim = config.get("ffn_dim")
    if head_dim is None:
      num_heads = config.get("num_heads", 4)
      if num_heads is None:
        num_heads = 4
      head_dim = transformer_dim // num_heads

    block.append(
        MobileViTBlock(
            in_channels=input_channels,
            transformer_dim=transformer_dim,
            ffn_dim=ffn_dim,
            n_transformer_blocks=config.get("transformer_blocks", 1),
            patch_h=config.get("patch_h", 2),
            patch_w=config.get("patch_w", 2),
            head_dim=head_dim,)
        )

    return tf.keras.Sequential(block, name=name), input_channels

  def build_mv2_block(self, input_channels, config, name):
    output_channels = config.get("out_channels")
    num_blocks = config.get("num_blocks", 2)
    expand_ratio = config.get("expand_ratio", 4)
    block = []

    for i in range(num_blocks):
      stride = config.get("stride", 1) if i == 0 else 1

      layer = InvertedResidual(
                inp_dims=input_channels,
                out_dims=output_channels,
                stride=stride,
                expansion=expand_ratio,
            )
      block.append(layer)
      input_channels = output_channels

    return tf.keras.Sequential(block, name=name), input_channels

  def forward_features(self, x):
    x = self.conv1(x)
    x = self.layer_1(x)
    x = self.layer_2(x)
    x = self.layer_3(x)
    x = self.layer_4(x)
    x = self.layer_5(x)
    x = self.conv_1x1_exp(x)

    return x

  def forward_head(self, x):
    x = self.classifier(x)

    return x

  def call(self, x, training=False):
    x = self.forward_features(x)
    x = self.forward_head(x)

    return x