import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
from typing import * 
from ml_collections import ConfigDict 
from ..layers import LayerScale, StochasticDepth, MLP, act_layer_factory, norm_layer_factory


# self_attention (mult-head self attention)
class TFViTSelfAttention(tf.keras.layers.Layer):
    def __init__(self,
                 config: ConfigDict,
                 hidden_size: int,
                 **kwargs)-> tf.keras.layers.Layer:
        super(TFViTSelfAttention, self).__init__(**kwargs)

        if hidden_size % config.num_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_heads})"
            )

        self.num_attention_heads = config.num_heads
        self.attention_head_size = int(hidden_size / config.num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # scale
        scale = tf.cast(self.attention_head_size, dtype=tf.float32)
        self.scale = tf.math.sqrt(scale)

        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        self.query = keras.layers.Dense(units=self.all_head_size,
                                        name="query",
                                        use_bias=config.qkv_bias)

        self.key = keras.layers.Dense(units=self.all_head_size,
                                      name="key",
                                      use_bias=config.qkv_bias)

        self.value = keras.layers.Dense(units=self.all_head_size,
                                        name="value",
                                        use_bias=config.qkv_bias)

        self.dropout = keras.layers.Dropout(rate=config.attention_probs_dropout_prob)

    def transpose_for_scores(
        self, tensor: tf.Tensor, batch_size: int
    ) -> tf.Tensor:
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(
            tensor=tensor,
            shape=(
                batch_size,
                -1,
                self.num_attention_heads,
                self.attention_head_size,
            ),
        )

        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size] to [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor = None,
        output_attentions: bool = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:

        batch_size = tf.shape(hidden_states)[0]

        # getting query, key, value vectors from qkv matrix
        mixed_query_layer = self.query(inputs=hidden_states)
        mixed_key_layer = self.key(inputs=hidden_states)
        mixed_value_layer = self.value(inputs=hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(logits=attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(
            inputs=attention_probs, training=training
        )

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = tf.multiply(attention_probs, head_mask)

        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, all_head_size)
        attention_output = tf.reshape(
            tensor=attention_output, shape=(batch_size, -1, self.all_head_size)
        )
        outputs = (
            (attention_output, attention_probs)
            if output_attentions
            else (attention_output,)
        )

        return outputs

    def get_config(self):
      config = super().get_config()
      config['num_attention_heads'] = self.num_attention_heads
      config['attention_head_size'] = self.attention_head_size
      #config['all_head_size'] = self.all_head_size
      #config['sqrt_att_head_size'] = self.sqrt_att_head_size

      return config

    @classmethod
    def from_config(cls, config):
      return cls(**config)


# self-attention output(projection dense layer)
class TFViTSelfOutput(tf.keras.layers.Layer):
    """
    The residual connection is defined in TFViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self,
                 config: ConfigDict,
                 hidden_size: int,
                 **kwargs)-> tf.keras.layers.Layer:

        super(TFViTSelfOutput, self).__init__(**kwargs)

        self.dense = keras.layers.Dense(units=hidden_size, name="dense")
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)

        self.projection_dim = hidden_size
        self.dropout_rate = config.hidden_dropout_prob

    def call(
        self,
        hidden_states: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:

        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)

        return hidden_states

    def get_config(self):
      config = super().get_config()
      config['projection_dim'] = self.projection_dim
      config['dropout_rate'] = self.dropout_rate

      return config

    @classmethod
    def from_config(cls, config):
      return cls(**config)


# combine of self_attention and self_output
class TFViTAttention(tf.keras.layers.Layer):
    def __init__(self,
                 config: ConfigDict,
                 hidden_size: int,
                 **kwargs)-> tf.keras.layers.Layer:

        super(TFViTAttention, self).__init__(**kwargs)

        self.self_attention = TFViTSelfAttention(config, hidden_size, name="attention")
        self.dense_output = TFViTSelfOutput(config, hidden_size, name="output")

    def call(
        self,
        input_tensor: tf.Tensor,
        head_mask: tf.Tensor = None,
        output_attentions: bool = True,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:

        self_outputs = self.self_attention(
            hidden_states=input_tensor,
            head_mask=head_mask,
            output_attentions=output_attentions,
            training=training,
        )
        attention_output = self.dense_output(
            hidden_states=self_outputs[0]
            if output_attentions
            else self_outputs,
            training=training,
        )
        if output_attentions:
            outputs = (attention_output,) + self_outputs[
                1:
            ]  # add attentions if we output them

        return outputs

    def get_config(self):
      config = super().get_config()

      return config

    @classmethod
    def from_config(cls, config):
      return cls(**config)


# combining of attention and mlp to create a transformer block
class TFVITTransformerLayer(tf.keras.layers.Layer):
    def __init__(self,
                 config: ConfigDict,
                 hidden_size: int,
                 mlp_units: List,
                 drop_prob: float = 0.0,
                 **kwargs)-> None:

        super(TFVITTransformerLayer, self).__init__(**kwargs)
        self.config = config
        self.attention = TFViTAttention(config, hidden_size)
        #self.mlp = mlp(self.config.dropout_rate, mlp_units)
        self.mlp = MLP(mlp_units, name="mlp_output")

        self.layernorm_before = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps,
            name="layernorm_before"
            )
        self.layernorm_after = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps,
            name="layernorm_after"
            )

        self.drop_prob = drop_prob
        self.layer_norm_eps = config.layer_norm_eps
        self.mlp_units = mlp_units
        
    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor = False,
        output_attentions: bool = False,
      #  drop_prob: float = 0.0,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:

        # first layernormalization
        x1 = self.layernorm_before(hidden_states)
        attention_output, attention_scores = self.attention(x1, output_attentions=True)

        attention_output = (
                        LayerScale(self.config)(attention_output)
                        if self.config.init_values
                        else attention_output
                      )

        attention_output = (
                    StochasticDepth(self.drop_prob)(attention_output)
                    if self.drop_prob
                    else attention_output
                )

        # first residual connection
        x2 = tf.keras.layers.Add()([attention_output, hidden_states])

        # second layernormalization
        x3 = self.layernorm_after(x2)
        x4 = self.mlp(x3)
        x4 = LayerScale(self.config)(x4) if self.config.init_values else x4
        x4 = StochasticDepth(self.drop_prob)(x4) if self.drop_prob else x4

        # second residual connection
        outputs = tf.keras.layers.Add()([x2, x4])

        if output_attentions:
            return outputs, attention_scores

        return outputs

    def get_config(self):
      config = super().get_config()
      #config['drop_prob'] = self.drop_prob
      config['layer_norm_eps'] = self.layer_norm_eps
      config['mlp_units'] = self.mlp_units

      return config

    @classmethod
    def from_config(cls, config):
      return cls(**config)
