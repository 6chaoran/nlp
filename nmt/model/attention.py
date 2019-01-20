import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.backend import permute_dimensions, sqrt, constant
from util.util_prep import load_object
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import matrix_band_part

def scaled_dot_product_attention(Q,K,V,mask = False):
    
    assert K.shape.as_list() == V.shape.as_list(), 'shape of K and V must same'
    
    ## define layers
    matmul_1 = Dot(axes=-1, name = 'dot_att_matmul1')
    matmul_2 = Dot(axes= 1, name = 'dot_att_matmul2')
    scale = Lambda(lambda x: x / sqrt(constant(k)), name = 'dot_att_scale')
    softmax = Activation(activation='softmax',name = 'dot_att_softmax')
    mask_layer = Lambda(lambda x: matrix_band_part(x, -1 ,0), name = 'dot_att_mask')     # lower tri matrix
    
    y = matmul_1([K,Q])
    y = scale(y)
    y = mask_layer(y)
    y = softmax(y)
    y = matmul_2([y,V])
    return y
