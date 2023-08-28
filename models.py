from .mobilevit import MobileVITModel
from .base_config import get_base_config
import yaml, os 
import tensorflow as tf 


def load_mobilevit_v1_small() -> tf.keras.Model:
    config_file_path = f"configs/mobilevit_v1_small.yaml"
    with open(config_file_path, "r") as f:
        data = yaml.safe_load(f)

    print("Instantiating Tensorflow model...")
    config = get_base_config(
            include_top = include_top,
            hidden_sizes = data.get('hidden_sizes'),
            neck_hidden_sizes = data.get('neck_hidden_sizes'),
            expand_ratio = data.get('expand_ratio')
    )

    tf_model = MobileVITModel(config)
    image_dim = 256
    dummy_inputs = tf.ones((1, image_dim, image_dim, 3))
    _ = tf_model(dummy_inputs)[0]

    return tf_model 


def load_mobilevit_v1_xsmall() -> tf.keras.Model:
    config_file_path = f"configs/mobilevit_v1_xsmall.yaml"
    with open(config_file_path, "r") as f:
        data = yaml.safe_load(f)

    print("Instantiating Tensorflow model...")
    config = get_base_config(
            include_top = include_top,
            hidden_sizes = data.get('hidden_sizes'),
            neck_hidden_sizes = data.get('neck_hidden_sizes'),
            expand_ratio = data.get('expand_ratio')
    )

    tf_model = MobileVITModel(config)
    image_dim = 256
    dummy_inputs = tf.ones((1, image_dim, image_dim, 3))
    _ = tf_model(dummy_inputs)[0]

    return tf_model


def load_mobilevit_v1_xxsmall() -> tf.keras.Model:
    config_file_path = f"configs/mobilevit_v1_xxsmall.yaml"
    with open(config_file_path, "r") as f:
        data = yaml.safe_load(f)

    print("Instantiating Tensorflow model...")
    config = get_base_config(
            include_top = include_top,
            hidden_sizes = data.get('hidden_sizes'),
            neck_hidden_sizes = data.get('neck_hidden_sizes'),
            expand_ratio = data.get('expand_ratio')
    )

    tf_model = MobileVITModel(config)
    image_dim = 256
    dummy_inputs = tf.ones((1, image_dim, image_dim, 3))
    _ = tf_model(dummy_inputs)[0]

    return tf_model  
