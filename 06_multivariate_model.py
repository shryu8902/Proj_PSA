#%%
import numpy as np
import pandas as pd
import os, time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import time
from utils import *
from sequential_utils import *
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import RepeatVector,LSTM, GRU, SimpleRNN, TimeDistributed, Dense, Activation, BatchNormalization, InputLayer,Bidirectional, LayerNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
#%%
# os.environ["CUDA_VISIBLE_DEVICES"]='7'
variable = ['primary_pressure', 'primary_temperature', 'secondary_pressure', 'secondary_temperature', 'PCT']

DATA = DataMerge('./DATA')
DATA_test = DataMerge('./DATA/TestSet',SCALERS=DATA.SCALERS)
SEED = 0
#%%
# DATA PREPARATION

X_train, X_test, Y_train, Y_test = train_test_split(DATA.X, DATA.Y, test_size=0.3, random_state=SEED)
X_val, X_test, Y_val, Y_test = train_test_split(X_test,Y_test, test_size = 0.9, random_state=SEED)

X_train_samp,Y_train_samp = augment_Y_by_resample(X_train,Y_train,5,return_all=True)
X_val_samp, Y_val_samp = augment_Y_by_resample(X_val,Y_val,5, return_all=True)

Y_train_samp_pad = pad_sequence_by_first(Y_train_samp,20)
Y_val_samp_pad = pad_sequence_by_first(Y_val_samp,20)

Test_Results = pd.DataFrame(columns=['config','PP_avg','PP_std','PT_avg','PT_std','SP_avg','SP_std','ST_avg','ST_std','PCT_avg','PCT_std'])
Val_Results = pd.DataFrame(columns=['config','PP_avg','PP_std','PT_avg','PT_std','SP_avg','SP_std','ST_avg','ST_std','PCT_avg','PCT_std'])
Test_Results2 = pd.DataFrame(columns=['config','PP_avg','PP_std','PT_avg','PT_std','SP_avg','SP_std','ST_avg','ST_std','PCT_avg','PCT_std'])

#%%
# MV_gen : Bi-direct, idel , PE, LSTM
# MV_gen2 : Uni-direct, idle, PE, LSTM
# MV_gen3 : Uni-direct, idle, LSTM
# MV_gen4 : Bi-direct, idle, PE, LSTM, NO DR
# MV_gen5 : Bi-direct, idle, PE(add), LSTM
# MV_gen6 : CNN, idle, PE(add) NO DR
# MV_gen7 : CNN, idle, PE(add)
K.clear_session()
file_root = './DATA/Models/MV_gen7.hdf5'


model = MV_TS_model_CNN()
# model = MV_TS_model_wPE_idle2()
# model = MV_TS_model_idle()
model.compile(optimizer='adam',
        loss='mean_squared_error',
        metrics=['mean_absolute_error','mean_squared_error']) 
model_save = ModelCheckpoint(file_root, save_weights_only=True, save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train_samp, Y_train_samp_pad, 
                        validation_data=(X_val_samp,Y_val_samp_pad),
                        batch_size = 512, epochs=100,
                        callbacks=[early_stopping,model_save])
#%% validation set
Y_hat = model.predict(X_val_samp)[:,20:,:]
Y_hat_up = upscaling(Y_hat, sampling_rate=5)
Y_hat_up_ = Y_hat_up.copy()
Y_val_ = Y_val.copy()
for i,v in enumerate(variable):
    print(v)
    Y_hat_up_[...,i] = DATA.SCALERS[v].inverse_transform(Y_hat_up[...,i].reshape(-1,1)).reshape(-1,2500)
    Y_val_[...,i] = DATA.SCALERS[v].inverse_transform(Y_val[...,i].reshape(-1,1)).reshape(-1,2500)
Y_val_ = np.tile(Y_val_,(5,1,1)) 

val_temp = []
for i,v in enumerate(variable):
    val_avg, val_std = mean_absolute_percentage_error(Y_val_[...,i],Y_hat_up_[...,i])
    val_temp.append(val_avg); val_temp.append(val_std);
Val_Results.loc[3]= ['MV_LSTM4']+val_temp
#%% #Test set
Y_hat = model.predict(X_test)[:,20:,:]
Y_hat_up = upscaling(Y_hat, sampling_rate=5)
Y_hat_up_ = Y_hat_up.copy()
Y_test_ = Y_test.copy()
for i,v in enumerate(variable):
    print(v)
    Y_hat_up_[...,i] = DATA.SCALERS[v].inverse_transform(Y_hat_up[...,i].reshape(-1,1)).reshape(-1,2500)
    Y_test_[...,i] = DATA.SCALERS[v].inverse_transform(Y_test[...,i].reshape(-1,1)).reshape(-1,2500)
test_temp = []
for i,v in enumerate(variable):
    test_avg, test_std = mean_absolute_percentage_error(Y_test_[...,i],Y_hat_up_[...,i])
    test_temp.append(test_avg); test_temp.append(test_std);
Test_Results.loc[3] = ['MV_LSTM4']+test_temp

#%% Real Test set// UNKNOWN SBLOCA

Y_hat = model.predict(DATA_test.X)[:,20:,:]
Y_hat_up = upscaling(Y_hat, sampling_rate=5)
Y_hat_up_ = Y_hat_up.copy()
Y_test_ = DATA_test.Y.copy()
for i,v in enumerate(variable):
    print(v)
    Y_hat_up_[...,i] = DATA.SCALERS[v].inverse_transform(Y_hat_up[...,i].reshape(-1,1)).reshape(-1,2500)
    Y_test_[...,i] = DATA.SCALERS[v].inverse_transform(DATA_test.Y[...,i].reshape(-1,1)).reshape(-1,2500)
test_temp = []
for i,v in enumerate(variable):
    test_avg, test_std = mean_absolute_percentage_error(Y_test_[...,i],Y_hat_up_[...,i])
    test_temp.append(test_avg); test_temp.append(test_std);
Test_Results2.loc[0] = ['MV_LSTM4']+test_temp
#%%
i=400
dim=4
plt.plot(Y_hat_up_[i,:,dim],label='Pred')
plt.plot(Y_test_[i,:,dim],label='Real')
plt.legend()
#%%
# Positional encoding
# Idle state
# B
def MV_TS_model_wPE_idle(bidirect = True, RNN = 'LSTM', layer_norm = False, dr_rates = 0.3, num_pads=20):
    inputs = layers.Input(shape =(9) ,name='input')
    num_rows = tf.shape(inputs,name='num_rows')[0]
    pos_enc_tile = tf.tile(positional_encoding(500+num_pads,9), [num_rows, 1,1],name='pos_enc_tile')
    inputs_extend = RepeatVector(500+num_pads,name='extend_inputs')(inputs)
    inputs_extend_wPE = tf.concat([inputs_extend, pos_enc_tile],2,name='input_pos_enc')

    layer_1 = Bidirectional(LSTM(128, return_sequences=True,dropout=dr_rates))(inputs_extend_wPE)
    layer_2 = Bidirectional(LSTM(128, return_sequences=True,dropout=dr_rates))(layer_1)
    # layer_1 = LSTM(128, return_sequences=True,dropout=dr_rates)(inputs_extend_wPE)
    # layer_2 = LSTM(128, return_sequences=True,dropout=dr_rates)(layer_1)
    layer_3 = TimeDistributed(Dense(64,activation='elu'))(layer_2)
    outputs = TimeDistributed(Dense(5))(layer_3)
    model= Model(inputs, outputs)
    return model
def MV_TS_model_wPE_idle2(bidirect = True, RNN = 'LSTM', layer_norm = False, dr_rates = 0.3, num_pads=20):
    inputs = layers.Input(shape =(9) ,name='input')
    num_rows = tf.shape(inputs,name='num_rows')[0]
    pos_enc_tile = tf.tile(positional_encoding(500+num_pads,9), [num_rows, 1,1],name='pos_enc_tile')
    inputs_extend = RepeatVector(500+num_pads,name='extend_inputs')(inputs)
    inputs_extend_wPE = inputs_extend+pos_enc_tile
    # inputs_extend_wPE = tf.concat([inputs_extend, pos_enc_tile],2,name='input_pos_enc')

    layer_1 = Bidirectional(LSTM(256, return_sequences=True,dropout=dr_rates))(inputs_extend_wPE)
    layer_2 = Bidirectional(LSTM(256, return_sequences=True,dropout=dr_rates))(layer_1)
    # layer_1 = LSTM(128, return_sequences=True,dropout=dr_rates)(inputs_extend_wPE)
    # layer_2 = LSTM(128, return_sequences=True,dropout=dr_rates)(layer_1)
    layer_3 = TimeDistributed(Dense(256,activation='elu'))(layer_2)    
    outputs = TimeDistributed(Dense(5))(layer_3)
    model= Model(inputs, outputs)
    return model

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

# Without Positional Encoding
def MV_TS_model_idle(bidirect = True, RNN = 'LSTM', layer_norm = False, dr_rates = 0.3, num_pads=20):
    inputs = layers.Input(shape =(9) ,name='input')
    inputs_extend = RepeatVector(500+num_pads,name='extend_inputs')(inputs)

    # layer_1 = Bidirectional(LSTM(128, return_sequences=True,dropout=dr_rates))(inputs_extend_wPE)
    # layer_2 = Bidirectional(LSTM(128, return_sequences=True,dropout=dr_rates))(layer_1)
    layer_1 = LSTM(128, return_sequences=True,dropout=dr_rates)(inputs_extend)
    layer_2 = LSTM(128, return_sequences=True,dropout=dr_rates)(layer_1)
    layer_3 = TimeDistributed(Dense(64,activation='elu'))(layer_2)
    outputs = TimeDistributed(Dense(5))(layer_3)
    model= Model(inputs, outputs)
    return model

    # model.summary()