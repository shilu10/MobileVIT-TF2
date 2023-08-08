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