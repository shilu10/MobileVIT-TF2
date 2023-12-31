import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
from typing import * 
from ml_collections import ConfigDict
from .transformer_block import MobileViTTransformer, MobileViTV2Transformer
from ..layers import InvertedResidualLayer, MobileViTConvLayer, act_layer_factory, norm_layer_factory
from ..layers import MobileViTV2ConvLayer, InvertedResidualLayerV2


class MobileVITLayer(tf.keras.layers.Layer):
    def __init__(self,
              config: ConfigDict,
              in_channels: int,
              out_channels: int,
              stride: int,
              hidden_size: int,
              num_stages: int,
              dilation: int = 1,
              **kwargs) -> None:

        super(MobileVITLayer, self).__init__(**kwargs)
        self.patch_width = config.patch_size
        self.patch_height = config.patch_size
        self.config = config

        self.downsampling_layer = None
        if stride == 2:
            self.downsampling_layer = InvertedResidualLayer(
                            config = config,
                            in_channels = in_channels,
                            out_channels = out_channels,
                            dilation = dilation // 2 if dilation > 1 else 1,
                            stride = stride if dilation == 1 else 1,
                            name = "downsampling_layer"
                        )
            in_channels = out_channels

        self.conv_kxk = MobileViTConvLayer(
                config=config,
                out_channels=in_channels,
                kernel_size=config.conv_kernel_size,
            )

        self.conv_1x1 = MobileViTConvLayer(
                config=config,
                out_channels=hidden_size,
                kernel_size=1,
                use_norm=False,
                use_act=False,
            )

        self.transformer = MobileViTTransformer(
                config, hidden_size=hidden_size, num_stages=num_stages, name="transformer"
            )

        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")

        self.conv_projection = MobileViTConvLayer(
                config=config, out_channels=in_channels, kernel_size=1, name="conv_projection"
            )

        self.fusion = MobileViTConvLayer(
                config=config, out_channels=in_channels, kernel_size=config.conv_kernel_size, name="fusion"
            )

    def unfolding(self, features: tf.Tensor) -> Tuple[tf.Tensor, Dict]:
        patch_width, patch_height = self.patch_width, self.patch_height
        patch_area = tf.cast(patch_width * patch_height, "int32")

        batch_size = tf.shape(features)[0]
        orig_height = tf.shape(features)[1]
        orig_width = tf.shape(features)[2]
        channels = tf.shape(features)[3]

        new_height = tf.cast(tf.math.ceil(orig_height / patch_height) * patch_height, "int32")
        new_width = tf.cast(tf.math.ceil(orig_width / patch_width) * patch_width, "int32")

        interpolate = new_width != orig_width or new_height != orig_height
        if interpolate:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            features = tf.image.resize(features, size=(new_height, new_width), method="bilinear")

        # number of patches along width and height
        num_patch_width = new_width // patch_width
        num_patch_height = new_height // patch_height
        num_patches = num_patch_height * num_patch_width

        # convert from shape (batch_size, orig_height, orig_width, channels)
        # to the shape (batch_size * patch_area, num_patches, channels)
        features = tf.transpose(features, [0, 3, 1, 2])
        patches = tf.reshape(
            features, (batch_size * channels * num_patch_height, patch_height, num_patch_width, patch_width)
        )
        patches = tf.transpose(patches, [0, 2, 1, 3])
        patches = tf.reshape(patches, (batch_size, channels, num_patches, patch_area))
        patches = tf.transpose(patches, [0, 3, 2, 1])
        patches = tf.reshape(patches, (batch_size * patch_area, num_patches, channels))

        info_dict = {
            "orig_size": (orig_height, orig_width),
            "batch_size": batch_size,
            "channels": channels,
            "interpolate": interpolate,
            "num_patches": num_patches,
            "num_patches_width": num_patch_width,
            "num_patches_height": num_patch_height,
        }
        return patches, info_dict

    def folding(self, patches: tf.Tensor, info_dict: Dict) -> tf.Tensor:
        patch_width, patch_height = self.patch_width, self.patch_height
        patch_area = int(patch_width * patch_height)

        batch_size = info_dict["batch_size"]
        channels = info_dict["channels"]
        num_patches = info_dict["num_patches"]
        num_patch_height = info_dict["num_patches_height"]
        num_patch_width = info_dict["num_patches_width"]

        # convert from shape (batch_size * patch_area, num_patches, channels)
        # back to shape (batch_size, channels, orig_height, orig_width)
        features = tf.reshape(patches, (batch_size, patch_area, num_patches, -1))
        features = tf.transpose(features, perm=(0, 3, 2, 1))
        features = tf.reshape(
                features, (batch_size * channels * num_patch_height, num_patch_width, patch_height, patch_width)
        )
        features = tf.transpose(features, perm=(0, 2, 1, 3))
        features = tf.reshape(
                features, (batch_size, channels, num_patch_height * patch_height, num_patch_width * patch_width)
            )
        features = tf.transpose(features, perm=(0, 2, 3, 1))

        if info_dict["interpolate"]:
            features = tf.image.resize(features, size=info_dict["orig_size"], method="bilinear")

        return features

    def call(self, features: tf.Tensor, training: bool = False) -> tf.Tensor:
        # reduce spatial dimensions if needed
        if self.downsampling_layer:
            features = self.downsampling_layer(features, training=training)

        shortcut = features

        # local representation
        features = self.conv_kxk(features, training=training)
        features = self.conv_1x1(features, training=training)

        # convert feature map to patches
        patches, info_dict = self.unfolding(features)

        # learn global representations
        patches = self.transformer(patches, training=training)
        patches = self.layernorm(patches)

        # convert patches back to feature maps
        features = self.folding(patches, info_dict)

        features = self.conv_projection(features, training=training)
        features = self.fusion(tf.concat([shortcut, features], axis=-1), training=training)

        return features

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'patch_width': self.patch_width,
                'patch_height': self.patch_height
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MobileViTV2Layer(tf.keras.layers.Layer):
    """
    MobileViTV2 layer: https://arxiv.org/abs/2206.02680
    """

    def __init__(
        self,
        config: ConfigDict,
        in_channels: int,
        out_channels: int,
        attn_unit_dim: int,
        n_attn_blocks: int = 2,
        dilation: int = 1,
        stride: int = 2,
    ) -> None:

        super().__init__()
        self.patch_width = config.patch_size
        self.patch_height = config.patch_size

        cnn_out_dim = attn_unit_dim

        if stride == 2:
            self.downsampling_layer = InvertedResidualLayerV2(
                config,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride if dilation == 1 else 1,
                dilation=dilation // 2 if dilation > 1 else 1,
            )
            in_channels = out_channels
        else:
            self.downsampling_layer = None

        # Local representations
        self.conv_kxk = MobileViTV2ConvLayer(
            config=config,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=config.conv_kernel_size,
            groups=in_channels,
        )

        self.conv_1x1 = MobileViTV2ConvLayer(
            config=config,
            in_channels=in_channels,
            out_channels=cnn_out_dim,
            kernel_size=1,
            use_norm=False,
            use_act=False,
        )

        # Global representations
        self.transformer = MobileViTV2Transformer(config,
                                                  d_model=attn_unit_dim,
                                                  n_layers=n_attn_blocks)

        # self.layernorm = MobileViTV2LayerNorm2D(attn_unit_dim, eps=config.layer_norm_eps)
        self.layernorm = tf.keras.layers.GroupNormalization(groups=1,
                                                            epsilon=config.layer_norm_eps,
                                                            )

        # Fusion
        self.conv_projection = MobileViTV2ConvLayer(
            config=config,
            in_channels=cnn_out_dim,
            out_channels=in_channels,
            kernel_size=1,
            use_norm=True,
            use_act=False,
        )

    def unfolding(self, feature_map: tf.Tensor) -> Tuple[tf.Tensor, Tuple[int, int]]:
        batch_size, img_height, img_width, in_channels = feature_map.shape
        patches = tf.image.extract_patches(
            feature_map,
            (1, self.patch_height, self.patch_width, 1),
            (1, self.patch_height, self.patch_width, 1),
            [1, 1, 1, 1],
            padding="VALID"
          )
        #transposed_patches = tf.transpose(patches, (0, 3, 1, 2))


        # (batch_size, num_pixels_in_patch, num_patches, channel_dims)
        patches = tf.reshape(patches, (batch_size, self.patch_height * self.patch_width, -1, in_channels))

        return patches, (img_height, img_width)

    def folding(self, patches: tf.Tensor, output_size: Tuple[int, int]) -> tf.Tensor:
        batch_size, patch_size, n_patches, in_dim = patches.shape

        n_patches_h = output_size[0] // self.patch_height
        n_patches_w = output_size[1] // self.patch_width

        feature_map = tf.reshape(patches,
            (batch_size, in_dim * self.patch_height * self.patch_width, n_patches_h, n_patches_w)
        )
        assert (
            self.patch_height == self.patch_width
        ), "For Coreml, we need patch_h and patch_w are the same"
        #feature_map = F.pixel_shuffle(feature_map, upscale_factor=self.patch_h)
        feature_map = tf.nn.depth_to_space(feature_map, self.patch_height)

        return feature_map

    def call(self, features: tf.Tensor) -> tf.Tensor:
        # reduce spatial dimensions if needed
        if self.downsampling_layer:
            features = self.downsampling_layer(features)

        # local representation
        features = self.conv_kxk(features)
        features = self.conv_1x1(features)

        # convert feature map to patches
        patches, output_size = self.unfolding(features)

        # learn global representations
        patches = self.transformer(patches)
        patches = self.layernorm(patches)

        # convert patches back to feature maps
        # [batch_size, patch_height, patch_width, input_dim] --> [batch_size, input_dim, patch_height, patch_width]
        features = self.folding(patches, output_size)

        features = self.conv_projection(features)
        return features