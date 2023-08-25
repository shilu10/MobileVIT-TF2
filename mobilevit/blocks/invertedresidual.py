import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
from ..layers import act_layer_factory, norm_layer_factory


class InvertedResidual(tf.keras.layers.Layer):
    def __init__(self,
                 inp_dims, 
                 out_dims, 
                 stride=1, 
                 expansion=4,
                 skip_connection=True,
                 **kwargs
              ):

        super(InvertedResidual, self).__init__(**kwargs)
        hidden_dim = int(inp_dims * expansion)
        self.use_res_connect = self.stride == 1 and inp_dims == out_dims
        self.stride = stride
        self.inp_dims = inp_dims 
        self.out_dims = out_dims 
        self.expansion = expansion

        if expansion == 1:
            self.blocks = tf.keras.models.Sequential([
                # dw
                tf.keras.layers.Conv2D(filters=hidden_dim,
                                                kernel_size=3,
                                                strides=stride,
                                                padding="same",
                                                use_bias=False),

                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(tf.nn.silu),

                # pw-linear
                tf.keras.layers.Conv2D(filters=out_dims,
                          kernel_size=1,
                          strides=1,
                          padding='valid',
                          use_bias=False),

                tf.keras.layers.BatchNormalization(),
           ])

        else:
            self.blocks = tf.keras.models.Sequential([
                # pw
                # down-sample in the first conv
                tf.keras.layers.Conv2D(filters=hidden_dim,
                          kernel_size=1,
                          strides=stride,
                          padding='valid',
                          use_bias=False),

                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(tf.nn.silu),

                # dw
                tf.keras.layers.Conv2D(filters=hidden_dim,
                                                kernel_size=3,
                                                strides=1,
                                                padding="same",
                                                groups=hidden_dim,
                                                use_bias=False),

                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(tf.nn.silu),

                # pw-linear
                tf.keras.layers.Conv2D(filters=out_dims,
                          kernel_size=1,
                          strides=1,
                          padding='valid',
                          use_bias=False),

                tf.keras.layers.BatchNormalization(),
            ])


    def call(self, x):
        shortcut = x

        if self.use_res_connect:
            return shortcut + self.blocks(x)

        else:
            return self.blocks(x)

    def get_config(self):
        config = super().get_config()
        config['stride'] = self.stride
        config['inp_dims'] = self.inp_dims 
        config['out_dims'] = self.out_dims 
        config['expansion'] = self.expansion

        return config


## new 
class InvertedResidualLayer(tf.keras.layers.Layer):
  def __init__(self, 
               config: ConfigDict, 
               in_channels: int, 
               out_channels: int, 
               stride: int, 
               dilation: int = 1, 
               **kwargs) -> None:
    super(InvertedResidualLayer, self).__init__(**kwargs)

    expanded_channels = make_divisible(int(round(in_channels * config.expand_ratio)), 8)

    if stride not in [1, 2]:
      raise ValueError(f"Invalid stride {stride}.")

    self.use_residual = (stride == 1) and (in_channels == out_channels)

    # pw
    # down-sample in the first conv
    self.expand_1x1 = MobileViTConvLayer(
            config, out_channels=expanded_channels, kernel_size=1, name="expand_1x1"
        )
    
    # DW
    self.conv_3x3 = MobileViTConvLayer(
            config,
            out_channels=expanded_channels,
            kernel_size=3,
            stride=stride,
            groups=expanded_channels,
            dilation=dilation,
            name="conv_3x3",
        )
    
    # pw
    # up-smaple in last conv
    self.reduce_1x1 = MobileViTConvLayer(
            config,
            out_channels=out_channels,
            kernel_size=1,
            use_activation=False,
            name="reduce_1x1",
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
 