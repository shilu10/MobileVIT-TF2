import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
from typing import * 


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


class DropPath(tf.keras.layers.Layer):
    """
    Per sample stochastic depth when applied in main path of residual blocks

    This is the same as the DropConnect created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in
    a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956
    Following the usage in `timm` we've changed the name to `drop_path`.
    """

    def __init__(self, drop_prob=None, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob
        self.keep_prob = 1.0 - self.drop_prob

    def call(self, x, training=False):
        if not training or not self.drop_prob > 0.0:
            return x

        # Compute drop_connect tensor
        shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
        random_tensor = self.keep_prob + tf.random.uniform(shape, dtype=x.dtype)
        binary_tensor = tf.floor(random_tensor)

        # Rescale output to preserve batch statistics
        x = tf.math.divide(x, self.keep_prob) * binary_tensor
        return x

    def get_config(self):
        config = super(DropPath, self).get_config()
        config["drop_prob"] = self.drop_prob
        return config


class LayerScale(tf.keras.layers.Layer):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = tf.Variable(init_values * tf.ones(dim), trainable=True)

    def call(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class MLP(tf.keras.layers.Layer):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        hidden_dim: int,
        projection_dim: int,
        drop_rate: float,
        act_layer: str,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        mlp_bias: bool = False,
        **kwargs,
    ):
        super(MLP, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.drop_rate = drop_rate
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        act_layer = act_layer_factory(act_layer)

        self.fc1 = tf.keras.layers.Dense(
            units=hidden_dim,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            use_bias=mlp_bias,
            name="fc1",
        )
        self.act = act_layer()
        self.drop1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.fc2 = tf.keras.layers.Dense(
            units=projection_dim,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            use_bias=mlp_bias,
            name="fc2",
        )
        self.drop2 = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x, training=training)
        x = self.fc2(x)
        x = self.drop2(x, training=training)
        return x

    def get_config(self):
        config = super(MLP, self).get_config()
        config["hidden_dim"] = self.hidden_dim
        config["projection_dim"] = self.projection_dim
        config["drop_rate"] = self.drop_rate
        config["kernel_initializer"] = self.kernel_initializer
        config["bias_initializer"] = self.bias_initializer
        return config


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, 
                 embed_dim, 
                 output_dim=None,
                 num_heads=8, 
                 head_dims=None, 
                 attn_drop=0., 
                 proj_drop=0.,  # ffn dropout rate
                 drop_rate=0.,
                 attn_bias=True,
              ):
      
        super(MultiHeadAttention, self).__init__()
        if output_dim is None:
            output_dim = embed_dim

        assert embed_dim % num_heads == 0, "embed_dim % num_head should be 0"

        # attn layers
        self.qkv = tf.keras.layers.Dense(
            units=embed_dim*3,
            use_bias=attn_bias,
            name="qkv_weight"

        )

        self.attn_dropout = tf.keras.layers.Dropout(attn_drop)
        
        # proj layers
        self.proj = tf.keras.layers.Dense(
            units=output_dim,
            use_bias=attn_bias,
        )

        self.proj_dropout = tf.keras.layers.Dropout(proj_drop)

        if head_dims is None:
          self.head_dims = embed_dim // num_heads

        self.scaling = self.head_dims**-0.5
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def __repr__(self):
        return "{}(head_dim={}, num_heads={}, attn_dropout={})".format(
            self.__class__.__name__, self.head_dims, self.num_heads, self.attn_dropout.rate
        )

    def call(self, x, training=False):
        b, n, c = tf.shape(x)
        qkv = self.qkv(x)

        qkv = tf.reshape(qkv, shape=(-1, n, 3, self.num_heads, self.head_dims))
        qkv = tf.transpose(qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = tf.unstack(qkv)

        q = q * self.scaling

        attn_scores = tf.matmul(q, k, transpose_b=True)
        attn_probs = tf.nn.softmax(attn_scores)
        attn_probs = self.attn_dropout(attn_probs, training=training)

        out = tf.matmul(attn_probs, k)
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, shape=(1, n, c))

        x = self.proj(x)
        x = self.proj_dropout(x, training=training)
        return x


class Transformer(tf.keras.layers.Layer):
  def __init__(self, 
              embed_dim, 
              ffn_latent_dim,
              num_heads, 
              head_dims=None, 
              qkv_bias=True, 
              attn_drop=0., 
              proj_drop=0., 
              drop_path=0.0,
              init_values=None, 
              norm_layer="layer_norm", 
              act_layer="gelu",
              **kwargs
          ):
      
    super(Transformer, self).__init__(**kwargs)
    norm_layer = norm_layer_factory(norm_layer)

    self.attn = MultiHeadAttention(
          embed_dim=embed_dim,
          num_heads=num_heads,
          attn_bias=qkv_bias,
          attn_drop=attn_drop,
          proj_drop=proj_drop,
        )
        
    self.mlp = MLP(
          hidden_dim=ffn_latent_dim,
          projection_dim=embed_dim,
          drop_rate=proj_drop,
          act_layer=act_layer,
          mlp_bias=True
        )
        
    self.ls_1 = LayerScale(embed_dim, init_values) if init_values else tf.identity
    self.ls_2 = LayerScale(embed_dim, init_values) if init_values else tf.identity

    self.drop_path_1 = DropPath(drop_path) if drop_path else tf.identity
    self.drop_path_2 = DropPath(drop_path) if drop_path else tf.identity

    self.norm_1 = norm_layer(name='transformer_norm1')
    self.norm_2 = norm_layer(name='transformer_norm2')

    self.embed_dim = embed_dim
    self.ffn_latent_dim = ffn_latent_dim 
    self.proj_drop = proj_drop
    self.attn_drop = attn_drop

  def call(self, x, training=False):
    # res1
    shortcut = x 

    x = self.norm_1(x)
    x = self.attn(x)
    x = self.ls_1(x)
    x = self.drop_path_1(x)

    x = x + shortcut 

    # res2
    shortcut = x

    x = self.norm_2(x)
    x = self.mlp(x)
    x = self.ls_2(x)
    x = self.drop_path_2(x)

    x = x + shortcut 

    return x 

  def __repr__(self):
    return "{}(embed_dim={}, ffn_latent_dim={}, proj_drop={}, attn_drop={}".format(
            self.__class__.__name__,
            self.embed_dim,
            self.ffn_latent_dim,
            self.proj_drop,
            self.attn_drop
      )
      
  def get_config(self):
    config = super(Transformer, self).get_config()
    config["embed_dim"] = self.embed_dim
    config['ffn_latent_dim'] = self.ffn_latent_dim
    config['proj_drop'] = self.proj_drop
    config['attn_drop'] = self.attn_drop

    return config


from typing import * 

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
            # Note: Padding can be done, but then it needs to be handled in attention function.
            feature_map = tf.image.resize(
                feature_map, 
                size=(new_h, new_w), 
                method=tf.image.ResizeMethod.BILINEAR, 
            )
            interpolate = True

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
            feature_map = F.interpolate(
                feature_map,
                size=info_dict["orig_size"],
                mode="bilinear",
                align_corners=False,
            )
        return tf.transpose(tf.transpose(feature_map, (0, 1, 3, 2)), (0, 3, 2, 1))

    def forward_spatial(self, x):
        res = x

        fm = self.local_rep(x)

        print(fm.shape)

        # convert feature map to patches
        patches, info_dict = self.unfolding(fm)  # unfolding operation

        print(patches.shape, info_dict)

        # learn global representations
        patches = self.global_rep(patches)
        print(patches.shape, "af gr")
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