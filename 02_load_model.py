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
from tensorflow.keras.layers import RepeatVector,LSTM, GRU, SimpleRNN, TimeDistributed, Dense, Activation, BatchNormalization, InputLayer,Bidirectional, LayerNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

#%%
## Set Seed for reproduciblity
SEED=0
np.random.seed(SEED)
tf.random.set_seed(SEED)
#%%
# Target variable to train & generate
variable = ['primary_pressure', 'primary_temperature', 'secondary_pressure', 'secondary_temperature', 'PCT']
# Create dataframe to store the mean and std of Absolute Percentage Error
Test_Results = pd.DataFrame(columns=['config','PP_avg','PP_std','PT_avg','PT_std','SP_avg','SP_std','ST_avg','ST_std','PCT_avg','PCT_std'])
Test2_Results = pd.DataFrame(columns=['config','PP_avg','PP_std','PT_avg','PT_std','SP_avg','SP_std','ST_avg','ST_std','PCT_avg','PCT_std'])

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
    test_temp = []
    test2_temp = []

    for v in variable:    
        ## Read datasets    
        allpath = './DATA/{}_all.csv'.format(v)
        DATA = np.array(pd.read_csv(allpath, header=None)).T
        DATA_input = DATA[:,:9]
        DATA_output = DATA[:,9:]
        ## Create and fit sclaer
        input_scaler_std = StandardScaler()
        output_scaler = StandardScaler()
        DATA_input_std = input_scaler_std.fit_transform(DATA_input)
        DATA_output_n = output_scaler.fit_transform(DATA_output.reshape(-1,1)).reshape(-1,2500)
        ## Read Test dataset of SBLOCA with 1.2, 1.4, 1.6, 1.8
        TestPath = './DATA/TestSet/{}_all.csv'.format(v)
        TEST_DATA = np.array(pd.read_csv(TestPath,header=None)).T
        TEST_DATA_input = TEST_DATA[:,:9]
        TEST_DATA_output = TEST_DATA[:,9:]
        TEST_DATA_input_std = input_scaler_std.transform(TEST_DATA_input)
        TEST_DATA_output_n = output_scaler.transform(TEST_DATA_output.reshape(-1,1)).reshape(-1,2500)
        X_test = TEST_DATA_input_std
        Y_test = TEST_DATA_output_n

        # #for wPE models
        file_name = name_create(config)
        file_root = './DATA/Models/'+v+file_name+'_wPE.hdf5'

        model = TS_model_wPE(bidirect=config['bidirect'], RNN = config['RNN'],
                            layer_norm = config['layer_norm'],dr_rate=0.3)
        model.load_weights(file_root)

        #for non-wPE models
        # file_name =name_create(config)
        # file_root = './DATA/Models/'+v+file_name+'.hdf5'
        # model = load_model(file_root)

        Y_hat = model.predict(X_test).reshape(-1,500)
        Y_hat_up = upscaling(Y_hat,sampling_rate=5)
        Y_hat_dn = output_scaler.inverse_transform(Y_hat_up.reshape(-1,1)).reshape(-1,2500)
        Y_test_dn = output_scaler.inverse_transform(Y_test.reshape(-1,1)).reshape(-1,2500)
        
        test_avg, test_std = mean_absolute_percentage_error(Y_test_dn,Y_hat_dn)
        test_avg2, test_std2 = Seq_wise_MAPE(Y_test_dn,Y_hat_dn)
        test_temp.append(test_avg); test_temp.append(test_std);
        test2_temp.append(test_avg2); test2_temp.append(test_std2);
        # Val_Results.loc[j] = [file_name[1:]] + val_temp


        # ## Train / Validation / Test Split
        # ## train/val/test : 0.7/0.03/0.27
        # X_train, X_test, Y_train, Y_test = train_test_split(DATA_input_std, DATA_output_n, test_size=0.3, random_state=SEED)
        # X_val, X_test, Y_val, Y_test = train_test_split(X_test,Y_test, test_size = 0.9, random_state=SEED)
        # ## Reduce time sequence to 1/5 by sampling
        # ## Hence ther will be 5 sequence for 1 input data.
        # #X_train,Y_train = augment_Y_by_resample(X_train,Y_train,5,return_all=True)
        # #X_val_base = X_val; Y_val_base = Y_val
        # #X_val,Y_val = augment_Y_by_resample(X_val,Y_val,5,return_all=True)
        # # # for base model
        # # file_name = name_create(config)
        # # # file_root = './DATA/Models/'+v+file_name+'_wPE.hdf5'
        # # file_root = './DATA/Models/'+v+file_name+'.hdf5'
        # # model = load_model(file_root)

        #%%
        for i in [0,50,100,200,400]:
            plt.plot(Y_hat_dn[i,:],label='Pred')
            plt.plot(Y_test_dn[i,:],label='Real')
            plt.legend()
            img_root = './DATA/Figs/test_'+v+file_name+'_wPE_'+str(i)+'.png'
            plt.savefig(img_root)
            plt.clf()

    Test_Results.loc[j] = [file_name[1:]] + test_temp
    Test2_Results.loc[j] = [file_name[1:]] + test2_temp
Test_Results.to_csv('./DATA/unseen_loca_wPE.csv')
Test2_Results.to_csv('./DATA/unseen_loca_wPE_seq_wise_mape.csv')


# %%
i=95
plt.plot(Y_hat_dn[i,:],label='Pred')
plt.plot(Y_test_dn[i,:],label='Real')