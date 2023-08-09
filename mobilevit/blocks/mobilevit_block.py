import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
from typing import * 
from .transformer_block import Transformer
from ..layers import Conv_3x3_bn, Conv_1x1_bn
from ..layers import act_layer_factory, norm_layer_factory


class MobileViTBlock(tf.keras.layers.Layer):
    """
    This class defines the `MobileViT block <https://arxiv.org/abs/2110.02178?context=cs.LG>`_

    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        transformer_dim (int): Input dimension to the transformer unit
        ffn_dim (int): Dimension of the FFN block
        n_transformer_blocks (Optional[int]): Number of transformer blocks. Default: 2
        head_dim (Optional[int]): Head dimension in the multi-head attention. Default: 32
        attn_dropout (Optional[float]): Dropout in multi-head attention. Default: 0.0
        dropout (Optional[float]): Dropout rate. Default: 0.0
        ffn_dropout (Optional[float]): Dropout between FFN layers in transformer. Default: 0.0
        patch_h (Optional[int]): Patch height for unfolding operation. Default: 8
        patch_w (Optional[int]): Patch width for unfolding operation. Default: 8
        transformer_norm_layer (Optional[str]): Normalization layer in the transformer block. Default: layer_norm
        conv_ksize (Optional[int]): Kernel size to learn local representations in MobileViT block. Default: 3
        dilation (Optional[int]): Dilation rate in convolutions. Default: 1
        no_fusion (Optional[bool]): Do not combine the input and output feature maps. Default: False
    """

    def __init__(
        self,
        in_channels: int,
        transformer_dim: int,
        ffn_dim: int,
        n_transformer_blocks: Optional[int] = 2,
        head_dim: Optional[int] = 32,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[int] = 0.0,
        ffn_dropout: Optional[int] = 0.0,
        patch_h: Optional[int] = 8,
        patch_w: Optional[int] = 8,
        transformer_norm_layer: Optional[str] = "layer_norm",
        conv_ksize: Optional[int] = 3,
        dilation: Optional[int] = 1,
        no_fusion: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:

        # local representation
        conv_3x3_in = Conv_3x3_bn(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1,
            use_norm=True,
            use_act=True,
            name="conv_3x3"
        )

        conv_1x1_in = Conv_1x1_bn(
            in_channels=in_channels,
            out_channels=transformer_dim,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=False,
            name="conv_1x1"
        )

        conv_1x1_out = Conv_1x1_bn(
            in_channels=transformer_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            use_norm=True,
            use_act=True,
            name="conv_1x1"
        )

        conv_3x3_out = None
        if not no_fusion:
            conv_3x3_out = Conv_3x3_bn(
                in_channels=2 * in_channels,
                out_channels=in_channels,
                kernel_size=conv_ksize,
                stride=1,
                use_norm=True,
                use_act=True,
                name="conv_3x3"
            )

        super(MobileViTBlock, self).__init__(**kwargs)
        norm_layer = norm_layer_factory(transformer_norm_layer)
        #transformer_norm_layer = norm_layer_factory("layer_norm")
        #print(transformer_norm_layer)
        # local representation
        self.local_rep = tf.keras.models.Sequential()
        self.local_rep.add(conv_3x3_in)
        self.local_rep.add(conv_1x1_in)

        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        # global representation
        global_rep = [
            Transformer(
                embed_dim=transformer_dim,
                ffn_latent_dim=ffn_dim,
                num_heads=num_heads,
                attn_drop=attn_dropout,
                proj_drop=ffn_dropout,
                norm_layer=transformer_norm_layer,
            )
            for _ in range(n_transformer_blocks)
        ]


        self.global_rep = tf.keras.Sequential(global_rep)
        self.norm = norm_layer(name="block_norm")

        self.conv_proj = conv_1x1_out

        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.dilation = dilation
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize

    def __repr__(self) -> str:
        repr_str = "{}(".format(self.__class__.__name__)

        repr_str += "\n\t Local representations"
        if isinstance(self.local_rep, nn.Sequential):
            for m in self.local_rep:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.local_rep)

        repr_str += "\n\t Global representations with patch size of {}x{}".format(
            self.patch_h, self.patch_w
        )
        if isinstance(self.global_rep, nn.Sequential):
            for m in self.global_rep:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.global_rep)

        if isinstance(self.conv_proj, nn.Sequential):
            for m in self.conv_proj:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.conv_proj)

        if self.fusion is not None:
            repr_str += "\n\t Feature fusion"
            if isinstance(self.fusion, nn.Sequential):
                for m in self.fusion:
                    repr_str += "\n\t\t {}".format(m)
            else:
                repr_str += "\n\t\t {}".format(self.fusion)

        repr_str += "\n)"
        return repr_str

    def unfolding(self, feature_map):
        patch_w, patch_h = self.patch_w, self.patch_h  # pre-defined patch_sizes
        patch_area = int(patch_w * patch_h)
        batch_size, orig_h, orig_w, in_channels = tf.shape(feature_map) # shape of the feature map

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        # transposing x from [B, H, W, C] -> [B, C, H, W]
        feature_map = tf.transpose(tf.transpose(feature_map, (0, 3, 2, 1)), (0, 1, 3, 2))

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            feature_map = tf.transpose(tf.transpose(feature_map, (0, 1, 3, 2)), (0, 3, 2, 1))
            # Note: Padding can be done, but then it needs to be handled in attention function.
            feature_map = tf.image.resize(
                feature_map,
                size=(new_h, new_w),
                method=tf.image.ResizeMethod.BILINEAR,
            )
            interpolate = True
            feature_map = tf.transpose(tf.transpose(feature_map, (0, 3, 2, 1)), (0, 1, 3, 2))

        # number of patches along width and height
        num_patch_w = new_w // patch_w  # n_w
        num_patch_h = new_h // patch_h  # n_h
        num_patches = num_patch_h * num_patch_w  # N

        # [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
        reshaped_fm = tf.reshape(
            feature_map, (batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w
        ))

        # [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
        transposed_fm = tf.transpose(reshaped_fm, perm=(0, 2, 1, 3))
        # [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        reshaped_fm = tf.reshape(
            transposed_fm, (batch_size, in_channels, num_patches, patch_area)
        )
        # [B, C, N, P] --> [B, P, N, C]
        transposed_fm = tf.transpose(reshaped_fm, perm=(0, 3, 2, 1))
        # [B, P, N, C] --> [BP, N, C]
        patches = tf.reshape(transposed_fm, (batch_size * patch_area, num_patches, -1))

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }

        return patches, info_dict

    def folding(self, patches, info_dict) :

        n_dim = len(tf.shape(patches))
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(
            patches.shape
        )
        # [BP, N, C] --> [B, P, N, C]
        patches = tf.reshape(patches, (
            info_dict["batch_size"].numpy(), self.patch_area, info_dict["total_patches"], -1)
        )

        batch_size, pixels, num_patches, channels = tf.shape(patches)
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] --> [B, C, N, P]
        patches = tf.transpose(patches, (0, 3, 2, 1))

        # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        feature_map = tf.reshape(
            patches, (batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        )
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        feature_map = tf.transpose(feature_map, (0, 2, 1, 3))
        # [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        feature_map = tf.reshape(
            feature_map, (batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        )
        if info_dict["interpolate"]:
            # transposing to attend the interpolation
            feature_map = tf.transpose(tf.transpose(feature_map, (0, 1, 3, 2)), (0, 3, 2, 1))
            feature_map = tf.image.resize(
                feature_map,
                size=info_dict["orig_size"],
                method="bilinear",
            )

            return feature_map
        return tf.transpose(tf.transpose(feature_map, (0, 1, 3, 2)), (0, 3, 2, 1))

    def forward_spatial(self, x):
        res = x
        fm = self.local_rep(x)

        # convert feature map to patches
        patches, info_dict = self.unfolding(fm)  # unfolding operation

        # learn global representations
        patches = self.global_rep(patches)
        patches = self.norm(patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        fm = self.folding(patches=patches, info_dict=info_dict)   # folding operation

        fm = self.conv_proj(fm)

        if self.fusion is not None:
            fm = self.fusion(tf.concat((res, fm), axis=-1))
        return fm

    def call(self, x, *args, **kwargs):
        if isinstance(x, tf.Tensor):
            # For image data
            return self.forward_spatial(x)
        else:
            raise NotImplementedError


class MobileVitV2Block(tf.keras.layers.Layer):
    """
    This class defines the `MobileViTv2 block <>`_
    """

    def __init__(
        self,
        in_channels: int,
        attn_unit_dim: int,
        mlp_ratio: Optional[Union[Sequence[Union[int, float]], int, float]] = 2.0,
        n_attn_blocks: Optional[int] = 2,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        patch_h: Optional[int] = 8,
        patch_w: Optional[int] = 8,
        conv_ksize: Optional[int] = 3,
        dilation: Optional[int] = 1,
        attn_norm_layer: Optional[str] = "layer_norm",
        *args,
        **kwargs
    ):

        cnn_out_dim = attn_unit_dim
        # local representation
        conv_3x3_in = Conv_3x3_bn(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1,
            use_norm=True,
            use_act=True,
            dilation=dilation,
            groups=in_channels,
          )

        conv_1x1_in = Conv_1x1_bn(
            in_channels=in_channels,
            out_channels=cnn_out_dim,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=False,
          )
        
        transformer_norm_layer = norm_layer_factory(attn_norm_layer)

        super(MobileVitV2Block, self).__init__()
        self.local_rep = tf.keras.Sequential([conv_3x3_in, conv_1x1_in])

        self.global_rep = tf.keras.Sequential([
            LinearTransformerBlock(
                attn_unit_dim,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_dropout,
                dropout=dropout,
                drop_path=0.0,
                norm_layer=attn_norm_layer
            )
            for _ in range(n_attn_blocks)
        ])
        self.norm = transformer_norm_layer()

        self.conv_proj = Conv_1x1_bn(
                      in_channels=cnn_out_dim,
                      out_channels=in_channels,
                      kernel_size=1,
                      stride=1,
                      use_norm=True,
                      use_act=False,
                    )

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = cnn_out_dim
        self.transformer_in_dim = attn_unit_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_attn_blocks
        self.conv_ksize = conv_ksize

    def call(self, x: tf.Tensor) -> tf.Tensor:
        B, H, W, C = x.shape
        patch_h, patch_w = self.patch_h, self.patch_w
        new_h, new_w = math.ceil(H / patch_h) * patch_h, math.ceil(W / patch_w) * patch_w
        num_patch_h, num_patch_w = new_h // patch_h, new_w // patch_w  # n_h, n_w
        num_patches = num_patch_h * num_patch_w  # N
        if new_h != H or new_w != W:
            print('in')
            x = tf.image.resize(x, size=(new_h, new_w), method="bilinear")

        # Local representation
        x = self.local_rep(x)

        # Unfold (feature map -> patches), [B, H, W, C] -> [B, P, N, C]
        B = 1 if B is None else B
        C = x.shape[-1]
        x = tf.reshape(x, (B, C, num_patch_h, patch_h, num_patch_w, patch_w))
        x = tf.transpose(x, perm=(0, 1, 3, 5, 2, 4))
        x = tf.reshape(x, (B,-1, num_patches, C))

        # Global representations
        x = self.global_rep(x)
        x = self.norm(x)

        # Fold (patches -> feature map), [B, P, N, C] --> [B, H, W, C]
        x = tf.reshape(x, (B, C, patch_h, patch_w, num_patch_h, num_patch_w))
        x = tf.transpose(x, perm=(0, 1, 4, 2, 5, 3))
        x = tf.reshape(x, (B, num_patch_h * patch_h, num_patch_w * patch_w, C))

        x = self.conv_proj(x)
        return x