
# Referred from: github.com:rwightman/pytorch-image-models.
class LayerScale(tf.keras.layers.Layer):
    def __init__(self, config: ConfigDict, **kwargs)-> tf.keras.layers.Layer:
        super().__init__(**kwargs)
        self.projection_dim = config.projection_dim
        # self.gamma = tf.Variable(
        #     config.init_values * tf.ones((config.projection_dim,)),
        #     name="layer_scale",
        #  )

        self.config = config

    def build(self, input_shape):
      self.gamma = self.add_weight(
            self.config.init_values * tf.ones((self.config.projection_dim,)),
            name="layer_scale",
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return x * self.gamma

    def get_config(self):
      config = super().get_conig()
      config['projection_dim'] = self.projection_dim

      return config

    @classmethod
    def from_config(cls, config):
      return cls(**config)


# drop_path
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