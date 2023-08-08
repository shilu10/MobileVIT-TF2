import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
from typing import * 

class LayerScale(tf.keras.layers.Layer):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = tf.Variable(init_values * tf.ones(dim), trainable=True)

    def call(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma