import tensorflow as tf 
from tensorflow import keras 
import numpy as np 


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


class Conv_1x1_bn(tf.keras.layers.Layer):
  def __init__(self, 
              in_channels=32,
              out_channels=64,
              kernel_size=1,
              stride=1,
              use_norm=True,
              use_act=True,
          ):
    super(Conv_1x1_bn, self).__init__()
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
      self.norm = tf.keras.layers.BatchNormalization(name='1*1_bn')
    
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
          ):
    super(Conv_3x3_bn, self).__init__()
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
        use_bias = False
    )

    if use_norm:
      self.norm = tf.keras.layers.BatchNormalization(name='3*3_bn')
    
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
