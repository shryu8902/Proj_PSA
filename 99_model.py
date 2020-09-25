#%%
# DATA.X : [None, 9]
# DATA.Y : [None, 2500,5]
import tensorflow as tf
import tensorflow.keras
from utility import *

def MV_PECNN(seqlen):
    inputs = layers.Input(shape=(9))
    inputs_extend = layers.RepeatVector(seqlen)(inputs)
    inputs_ext_PE = inputs_extend + positional_encoding(seqlen,9)
    conv = layers.Conv1D(128,3,padding='same')(inputs_extend_wPE)
    bn = layers.BatchNormalization()(conv)        
    conv = 

    num_rows = tf.shape()        
def MV_TS_model_CNN(num_pads=20,num_layer=3,dr_rates=0.3):
    inputs = layers.Input(shape = 9)
    num_rows = tf.shape(inputs,name='num_rows')[0]
    pos_enc_tile = tf.tile(positional_encoding(500+num_pads,9), [num_rows, 1,1],name='pos_enc_tile')
    inputs_extend = RepeatVector(500+num_pads,name='extend_inputs')(inputs)
    inputs_extend_wPE = inputs_extend+pos_enc_tile
    conv = layers.Conv1D(128,3,padding='same')(inputs_extend_wPE)
    bn = layers.BatchNormalization()(conv)
    relu = layers.LeakyReLU()(bn)
    current_input = relu
    for i in range(num_layer):
        conv = layers.Conv1D(128,3,padding='same')(current_input)
        bn = layers.BatchNormalization()(conv)
        relu = layers.LeakyReLU()(bn)
        current_input = relu + current_input
        current_input = tf.keras.layers.Dropout(dr_rates)(current_input) 

    outputs = layers.Conv1D(5,3,padding='same')(current_input)        
    model = Model(inputs, outputs)
    return model
