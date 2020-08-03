#%%
import numpy as np
import pandas as pd
import os, time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import time
from utils import *

import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import RepeatVector,Lambda,LSTM, GRU, SimpleRNN, TimeDistributed, Dense, Activation, BatchNormalization, InputLayer,Bidirectional, LayerNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

#%%
# Target variable to train & generate
variable = ['primary_pressure', 'primary_temperature', 'secondary_pressure', 'secondary_temperature', 'PCT']
# Create dataframe to store the mean and std of Absolute Percentage Error
Test_Results = pd.DataFrame(columns=['config','PP_avg','PP_std','PT_avg','PT_std','SP_avg','SP_std','ST_avg','ST_std','PCT_avg','PCT_std'])
Val_Results = pd.DataFrame(columns=['config','PP_avg','PP_std','PT_avg','PT_std','SP_avg','SP_std','ST_avg','ST_std','PCT_avg','PCT_std'])
# Create list of dictionary that contains the configurations of NN model
Bidirect = [False]
CellType = ['LSTM']
LayerNorm = [True, False] 
Config_List = [{'bidirect':x,'RNN':y,'layer_norm':z} for x in Bidirect for y in CellType for z in LayerNorm]
#%%
for j, config in enumerate(Config_List):
    #Network Configuration
    print(config)
    #Create list for mape results
    val_temp =  []
    test_temp = []
    for i,v in enumerate(variable):
        ## Set Seed for reproduciblity
        SEED=0
        np.random.seed(SEED)
        tf.random.set_seed(SEED)
        ## Read datasets    
        allpath = './DATA/{}_all.csv'.format(v)
        DATA = np.array(pd.read_csv(allpath, header=None)).T
        DATA_input = DATA[:,:9]
        DATA_output = DATA[:,9:]
        input_scaler_std = StandardScaler()
        output_scaler = StandardScaler()
        DATA_input_std = input_scaler_std.fit_transform(DATA_input)
        DATA_output_n = output_scaler.fit_transform(DATA_output.reshape(-1,1)).reshape(-1,2500)

        ## Train / Validation / Test Split
        #%% train/val/test : 0.7/0.03/0.27
        X_train, X_test, Y_train, Y_test = train_test_split(DATA_input_std, DATA_output_n, test_size=0.3, random_state=SEED)
        X_val, X_test, Y_val, Y_test = train_test_split(X_test,Y_test, test_size = 0.9, random_state=SEED)
        ## Reduce time sequence to 1/5 by sampling
        ## Hence ther will be 5 sequence for 1 input data.
        X_train,Y_train = augment_Y_by_resample(X_train,Y_train,5,return_all=True)
        X_val_base = X_val; Y_val_base = Y_val
        X_val,Y_val = augment_Y_by_resample(X_val,Y_val,5,return_all=True)
        # X_test,Y_test = augment_Y_by_resample(X_test,Y_test,5,return_all=False)

        ## Create model
        K.clear_session()
        
        ## Check model exists
        file_name = name_create(config)
        file_root = './DATA/Models/'+v+file_name+'_wPE_nondrop.hdf5'
        # if model exist, load model
        # if os.path.exists(file_root):
        #     model = load_model(file_root) 
        # else: #else train new model
        model = TS_model_wPE(bidirect=config['bidirect'], RNN = config['RNN'], 
                            layer_norm = config['layer_norm'], dr_rate=0)
        model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['mean_absolute_error','mean_squared_error']) 
        
        ## Model training
        model_save = ModelCheckpoint(file_root, save_weights_only=True, save_best_only=True, monitor='val_loss', mode='min')

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(X_train, Y_train.reshape(-1,500,1), 
                        validation_data=(X_val,Y_val),
                        batch_size = 512, epochs=100qj,
                        callbacks=[early_stopping,model_save])
        #%% Model loading
        # model = TS_model_wPE(bidirect=config['bidirect'], RNN = config['RNN'], layer_norm = config['layer_norm'],dr_rate=0)
        # model.load_weights(file_root)
        #
        #  model_test = load_model(file_root),custom_objects={'positional_encoding':positional_encoding,'get_angles':get_angles,'tile_fun':tile_fun})
        ## Calculate val and test mapes 
        #Validation loss       
        Y_hat = model.predict(X_val_base).reshape(-1,500)
        Y_hat_up = upscaling(Y_hat, sampling_rate=5)
        Y_hat_up_ = output_scaler.inverse_transform(Y_hat_up.reshape(-1,1)).reshape(-1,2500)
        Y_val_ = output_scaler.inverse_transform(Y_val_base.reshape(-1,1)).reshape(-1,2500)

        val_avg, val_std = mean_absolute_percentage_error(Y_val_,Y_hat_up_)
        val_temp.append(val_avg); val_temp.append(val_std);

        #Test loss
        Y_hat = model.predict(X_test).reshape(-1,500)
        Y_hat_up = upscaling(Y_hat, sampling_rate=5)
        Y_hat_up_ = output_scaler.inverse_transform(Y_hat_up.reshape(-1,1)).reshape(-1,2500)
        Y_test_ = output_scaler.inverse_transform(Y_test.reshape(-1,1)).reshape(-1,2500)

        test_avg, test_std = mean_absolute_percentage_error(Y_test_,Y_hat_up_)
        test_temp.append(test_avg); test_temp.append(test_std);

    Val_Results.loc[j] = [file_name[1:]] + val_temp
    Test_Results.loc[j] = [file_name[1:]] + test_temp
Val_Results.to_csv('./DATA/val_LSTM_wPE_nondr.csv')
Test_Results.to_csv('./DATA/test_LSTM_wPE_nondr.csv')
#%%
i=0
plt.plot(Y_hat_up_[i,:])
plt.plot(Y_test_[i,:])



#%% Model

def tile_fun(tensor,batch_size):
    out = tf.tile(tensor, (batch_size, 1,1))
    return out

def TS_model_wPE(bidirect = True, RNN = 'LSTM', layer_norm = False,dr_rate=0.3):
    inputs = layers.Input(shape =(9) ,name='input')
    pos_enc_tile = tf.tile(positional_encoding(500,9), [tf.shape(inputs)[0], 1, 1], name='pos_enc_tile')
    inputs_extend = RepeatVector(500,name='extend_inputs')(inputs)
    inputs_extend_wPE = tf.concat([inputs_extend, pos_enc_tile],2,name = 'input_pos_enc')

    if bidirect ==True:
        if RNN == 'LSTM':
            layer_1 = Bidirectional(LSTM(128, return_sequences=True,dropout=dr_rate))(inputs_extend_wPE)
        elif RNN == 'GRU':
            layer_1 = Bidirectional(GRU(128, return_sequences=True,dropout=dr_rate))(inputs_extend_wPE)
        else :
            layer_1 = Bidirectional(SimpleRNN(128, return_sequences=True,dropout=dr_rate))(inputs_extend_wPE)
    else :
        if RNN == 'LSTM':
            layer_1 = LSTM(128, return_sequences=True,dropout=dr_rate)(inputs_extend_wPE)
        elif RNN == 'GRU':
            layer_1 = GRU(128, return_sequences=True,dropout=dr_rate)(inputs_extend_wPE)
        else :
            layer_1 = SimpleRNN(128, return_sequences=True,dropout=dr_rate)(inputs_extend_wPE)

    if layer_norm==True:
        layer_1 = LayerNormalization()(layer_1)
    
    if bidirect ==True:
        if RNN == 'LSTM':
            layer_2 = Bidirectional(LSTM(128, return_sequences=True,dropout=dr_rate))(layer_1)
        elif RNN == 'GRU':
            layer_2 = Bidirectional(GRU(128, return_sequences=True,dropout=dr_rate))(layer_1)
        else :
            layer_2 = Bidirectional(SimpleRNN(128, return_sequences=True,dropout=dr_rate))(layer_1)
    else :
        if RNN == 'LSTM':
            layer_2 = LSTM(128, return_sequences=True,dropout=dr_rate)(layer_1)
        elif RNN == 'GRU':
            layer_2 = GRU(128, return_sequences=True,dropout=dr_rate)(layer_1)
        else :
            layer_2 = SimpleRNN(128, return_sequences=True,dropout=dr_rate)(layer_1)

    if layer_norm==True:
        layer_2 = LayerNormalization()(layer_2)
    
    layer_3 = TimeDistributed(Dense(64,activation='elu'))(layer_2)
    outputs = TimeDistributed(Dense(1))(layer_3)
    model= Model(inputs, outputs)
    return model
    # model.summary()
#%%


#%%

Y_hat=model.predict(X_test)
Y_hat=Y_hat.reshape(-1,500)
Y_hat_up = upscaling(Y_hat,sampling_rate=5)
Y_hat_dn = output_scaler.inverse_transform(Y_hat_up.reshape(-1,1)).reshape(-1,2500)
Y_test_dn = output_scaler.inverse_transform(Y_test.reshape(-1,1)).reshape(-1,2500)

#%%

#%%
i=1
plt.plot(Y_hat_dn[i,:])
plt.plot(Y_test_dn[i,:])

for i in range(len(Y_hat_dn)):
    plt.plot(Y_hat_dn[i,:])
for i in range(len(Y_hat_dn)):
    plt.plot(Y_test_dn[i,:])

#%%
mape, std = mean_absolute_percentage_error(Y_test_dn,Y_hat_dn)
print(mape)
print(std)
#%%


#%% 

#%%
pos_encoding = positional_encoding(500, 9)
print (pos_encoding.shape)

plt.pcolormesh(pos_encoding[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 9))
plt.ylabel('Position')
plt.colorbar()
plt.show()








