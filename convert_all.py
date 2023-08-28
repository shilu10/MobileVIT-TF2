import numpy as np 
import os, sys, yaml 
from .convert import port 
import tensorflow as tf 
from tensorflow import keras 
from typing import * 
from ml_collections import ConfigDict 


model_variants = [
    'mobilevit_v1_small',
    'mobilevit_v1_xsmall',
    'mobilevit_v1_xxsmall',
]

def main(model_savepath: str = '.', 
        include_top: bool = True):
    
    for model_variant in model_variants:
        print(f'Converting model variant: {model_variant}')

        port(model_savepath=model_savepath, 
            include_top=include_top, 
            model_type=model_variant)
            