from .utils import *
from tensorflow import keras 
import tensorflow as tf 
import numpy as np 
from typing import * 
from ml_collections import ConfigDict 
import tqdm, sys, os 
from transformers import MobileViTForImageClassification
from .base_config import get_base_config
from .mobilevit import MobileVITModel


def port(model_type: str = 'mobilevit_small', 
        model_savepath: str =  '.', 
        include_top: bool = True):
    
    model_ckpt = {
        'mobilevit_small': "apple/mobilevit-small",
        'mobilevit_xsmall': "apple/mobilevit-x-small",
        'mobilevit_xxsmall': 'apple/mobilevit-xx-small'
    }

    print("Instantiating PyTorch model...")
    pt_model = MobileViTForImageClassification.from_pretrained(model_ckpt[model_type])

    pt_model.eval()

    # intantiating tensorflow model
    config = get_base_config()
    tf_model = MobileVITModel(config)
    image_dim = 256
    dummy_inputs = tf.ones((1, image_dim, image_dim, 3))
    _ = tf_model(dummy_inputs)[0]

    # conv stem
    # convolution
    tf_model.layers[0].conv = modify_tf_block(
        tf_model.layers[0].conv,
        pt_model_dict['mobilevit.conv_stem.convolution.weight'],

    )

    # normalization
    tf_model.layers[0].norm = modify_tf_block(
        tf_model.layers[0].norm,
        pt_model_dict['mobilevit.conv_stem.normalization.weight'],
        pt_model_dict['mobilevit.conv_stem.normalization.bias'],
    )

    # running mean and variance
    moving_mean = tf.Variable(pt_model_dict['mobilevit.conv_stem.normalization.running_mean'])
    tf_model.layers[0].norm.moving_mean.assign(moving_mean)
    moving_variance = tf.Variable(pt_model_dict['mobilevit.conv_stem.normalization.running_var'])
    tf_model.layers[0].norm.moving_variance.assign(moving_variance)

    # conv_1x1_exp
    tf_model.layers[-3].conv = modify_tf_block(
        tf_model.layers[-3].conv,
        pt_model_dict['mobilevit.conv_1x1_exp.convolution.weight'],

    )

    # normalization
    tf_model.layers[-3].norm = modify_tf_block(
        tf_model.layers[-3].norm,
        pt_model_dict['mobilevit.conv_1x1_exp.normalization.weight'],
        pt_model_dict['mobilevit.conv_1x1_exp.normalization.bias'],
    )

    # running mean and variance
    moving_mean = tf.Variable(pt_model_dict['mobilevit.conv_1x1_exp.normalization.running_mean'])
    tf_model.layers[-3].norm.moving_mean.assign(moving_mean)
    moving_variance = tf.Variable(pt_model_dict['mobilevit.conv_1x1_exp.normalization.running_var'])
    tf_model.layers[-3].norm.moving_variance.assign(moving_variance)

    if include_top:
        # classification head
        tf_model.layers[-1] = modify_tf_block(
            tf_model.layers[-1],
            pt_model_dict['classifier.weight'],
            pt_model_dict['classifier.bias'],
        )

    for indx, block in enumerate(tf_model.layers[1: 1+5]):
        modify_mobilenet_and_mobilevit_blocks(block, indx, pt_model_dict)

    print("Porting successful, serializing TensorFlow model...")


    save_path = os.path.join(model_savepath, model_type) 
    save_path = save_path if include_top else save_path + "_feature_extractor"

    tf_model.save_weights(save_path + ".h5")

    print(f"TensorFlow model weights serialized at: {save_path}")



def modify_invertedres_block(block, pt_model_dict, block_name):
  # expand_1*1
  # normalization
  block.expand_1x1.conv = modify_tf_block(
          block.expand_1x1.conv,
          pt_model_dict[f'{block_name}.expand_1x1.convolution.weight']
      )

  # normalization
  block.expand_1x1.norm = modify_tf_block(
          block.expand_1x1.norm,
          pt_model_dict[f'{block_name}.expand_1x1.normalization.weight'],
          pt_model_dict[f'{block_name}.expand_1x1.normalization.bias'],
      )

      # normalization running_mean and var
  moving_mean = tf.Variable(pt_model_dict[f'{block_name}.expand_1x1.normalization.running_mean'])
  block.expand_1x1.norm.moving_mean.assign(moving_mean)
  moving_var = tf.Variable(pt_model_dict[f'{block_name}.expand_1x1.normalization.running_var'])
  block.expand_1x1.norm.moving_variance.assign(moving_var)

  # conv_3x3
  block.conv_3x3.conv = modify_tf_block(
          block.conv_3x3.conv,
          pt_model_dict[f'{block_name}.conv_3x3.convolution.weight']
      )

  # normalization
  block.conv_3x3.norm = modify_tf_block(
          block.conv_3x3.norm,
          pt_model_dict[f'{block_name}.conv_3x3.normalization.weight'],
          pt_model_dict[f'{block_name}.conv_3x3.normalization.bias'],
      )

  # normalization running_mean and var
  moving_mean = tf.Variable(pt_model_dict[f'{block_name}.conv_3x3.normalization.running_mean'])
  block.conv_3x3.norm.moving_mean.assign(moving_mean)
  moving_var = tf.Variable(pt_model_dict[f'{block_name}.conv_3x3.normalization.running_var'])
  block.conv_3x3.norm.moving_variance.assign(moving_var)

  # reduce_1x1
  block.reduce_1x1.conv = modify_tf_block(
          block.reduce_1x1.conv,
          pt_model_dict[f'{block_name}.reduce_1x1.convolution.weight']
      )

  # normalization
  block.reduce_1x1.norm = modify_tf_block(
          block.reduce_1x1.norm,
          pt_model_dict[f'{block_name}.reduce_1x1.normalization.weight'],
          pt_model_dict[f'{block_name}.reduce_1x1.normalization.bias'],
      )

  # normalization running_mean and var
  moving_mean = tf.Variable(pt_model_dict[f'{block_name}.reduce_1x1.normalization.running_mean'])
  block.reduce_1x1.norm.moving_mean.assign(moving_mean)
  moving_var = tf.Variable(pt_model_dict[f'{block_name}.reduce_1x1.normalization.running_var'])
  block.reduce_1x1.norm.moving_variance.assign(moving_var)


def modify_mobilenet_and_mobilevit_blocks(block, block_indx, pt_model_dict):
  pt_encoder_name = f'mobilevit.encoder.layer.{block_indx}'
  if isinstance(block, MobileNetBlock):
    for block_indx, invertedres_block in enumerate(block.layers):
      pt_block_name = pt_encoder_name + f".layer.{block_indx}"
      modify_invertedres_block(invertedres_block, pt_model_dict, pt_block_name)

  if isinstance(block, MobileVITLayer):
    # downsampling -> inverted_res_block
    modify_invertedres_block(block.downsampling_layer, pt_model_dict, pt_encoder_name + '.downsampling_layer')

    # conv_kxk
    block.conv_kxk.conv = modify_tf_block(
          block.conv_kxk.conv,
          pt_model_dict[f'{pt_encoder_name}.conv_kxk.convolution.weight']
      )
    
    block.conv_kxk.norm = modify_tf_block(
          block.conv_kxk.norm,
          pt_model_dict[f'{pt_encoder_name}.conv_kxk.normalization.weight'],
          pt_model_dict[f'{pt_encoder_name}.conv_kxk.normalization.bias']
      )
    
    moving_mean = tf.Variable(pt_model_dict[f'{pt_encoder_name}.conv_kxk.normalization.running_mean'])
    block.conv_kxk.norm.moving_mean.assign(moving_mean)
    moving_var = tf.Variable(pt_model_dict[f'{pt_encoder_name}.conv_kxk.normalization.running_var'])
    block.conv_kxk.norm.moving_variance.assign(moving_var)

    # conv_1*1
    block.conv_1x1.conv = modify_tf_block(
          block.conv_1x1.conv,
          pt_model_dict[f'{pt_encoder_name}.conv_1x1.convolution.weight']
      )
    
    # transformer layers
    for transformer_indx, transformer_block in enumerate(block.transformer.layers):
      pt_transformer_name = pt_encoder_name + f'.transformer.layer.{transformer_indx}'

      # attention query
      transformer_block.attention.self_attention.query = modify_tf_block(
          transformer_block.attention.self_attention.query,
          pt_model_dict[f'{pt_transformer_name}.attention.attention.query.weight'],
          pt_model_dict[f'{pt_transformer_name}.attention.attention.query.bias']
        )

      # attention value
      transformer_block.attention.self_attention.value = modify_tf_block(
          transformer_block.attention.self_attention.value,
          pt_model_dict[f'{pt_transformer_name}.attention.attention.value.weight'],
          pt_model_dict[f'{pt_transformer_name}.attention.attention.value.bias']
        )

      # attention key
      transformer_block.attention.self_attention.key = modify_tf_block(
          transformer_block.attention.self_attention.key,
          pt_model_dict[f'{pt_transformer_name}.attention.attention.key.weight'],
          pt_model_dict[f'{pt_transformer_name}.attention.attention.key.bias']
        )

      # dense output
      transformer_block.attention.dense_output.dense = modify_tf_block(
          transformer_block.attention.dense_output.dense,
          pt_model_dict[f'{pt_transformer_name}.attention.output.dense.weight'],
          pt_model_dict[f'{pt_transformer_name}.attention.output.dense.bias']
        )
      
      # intermediate and output dense 
      transformer_block.mlp.fc_1 = modify_tf_block(
          transformer_block.mlp.fc_1,
          pt_model_dict[f'{pt_transformer_name}.intermediate.dense.weight'],
          pt_model_dict[f'{pt_transformer_name}.intermediate.dense.bias']
        )
      
      transformer_block.mlp.fc_2 = modify_tf_block(
          transformer_block.mlp.fc_2,
          pt_model_dict[f'{pt_transformer_name}.output.dense.weight'],
          pt_model_dict[f'{pt_transformer_name}.output.dense.bias']
        )
      
      # layer_norm 1 and layer_norm 2 
      transformer_block.layernorm_before = modify_tf_block(
          transformer_block.layernorm_before,
          pt_model_dict[f'{pt_transformer_name}.layernorm_before.weight'],
          pt_model_dict[f'{pt_transformer_name}.layernorm_before.bias']
        )
      
      transformer_block.layernorm_after = modify_tf_block(
          transformer_block.layernorm_after,
          pt_model_dict[f'{pt_transformer_name}.layernorm_after.weight'],
          pt_model_dict[f'{pt_transformer_name}.layernorm_after.bias']
        )

    # layernorm 
    block.layernorm = modify_tf_block(
          block.layernorm,
          pt_model_dict[f'{pt_encoder_name}.layernorm.weight'],
          pt_model_dict[f'{pt_encoder_name}.layernorm.bias']
        )
    
    # conv_projection
    block.conv_projection.conv = modify_tf_block(
          block.conv_projection.conv,
          pt_model_dict[f'{pt_encoder_name}.conv_projection.convolution.weight'],
        )
    
    # norm
    block.conv_projection.norm = modify_tf_block(
          block.conv_projection.norm,
          pt_model_dict[f'{pt_encoder_name}.conv_projection.normalization.weight'],
          pt_model_dict[f'{pt_encoder_name}.conv_projection.normalization.bias']
        )
    moving_mean = tf.Variable(pt_model_dict[f'{pt_encoder_name}.conv_projection.normalization.running_mean'])
    block.conv_projection.norm.moving_mean.assign(moving_mean)
    moving_var = tf.Variable(pt_model_dict[f'{pt_encoder_name}.conv_projection.normalization.running_var'])
    block.conv_projection.norm.moving_variance.assign(moving_var)

    # fusion
    block.fusion.conv = modify_tf_block(
          block.fusion.conv,
          pt_model_dict[f'{pt_encoder_name}.fusion.convolution.weight'],
        )
    
    # norm
    block.fusion.norm = modify_tf_block(
          block.fusion.norm,
          pt_model_dict[f'{pt_encoder_name}.fusion.normalization.weight'],
          pt_model_dict[f'{pt_encoder_name}.fusion.normalization.bias']
        )
    moving_mean = tf.Variable(pt_model_dict[f'{pt_encoder_name}.fusion.normalization.running_mean'])
    block.fusion.norm.moving_mean.assign(moving_mean)
    moving_var = tf.Variable(pt_model_dict[f'{pt_encoder_name}.fusion.normalization.running_var'])
    block.fusion.norm.moving_variance.assign(moving_var)


for indx, block in enumerate(tf_model.layers[1: 1+5]):
  modify_mobilenet_and_mobilevit_blocks(block, indx, pt_model_dict)
  