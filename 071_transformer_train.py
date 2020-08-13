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


#%%
SEED=0
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Target variable to train & generate
variable = ['primary_pressure', 'primary_temperature', 'secondary_pressure', 'secondary_temperature', 'PCT']

# Create dataframe to store the mean and std of Absolute Percentage Error
Test_Results = pd.DataFrame(columns=['config','PP_avg','PP_std','PT_avg','PT_std','SP_avg','SP_std','ST_avg','ST_std','PCT_avg','PCT_std'])
Val_Results = pd.DataFrame(columns=['config','PP_avg','PP_std','PT_avg','PT_std','SP_avg','SP_std','ST_avg','ST_std','PCT_avg','PCT_std'])

DATA = DataMerge('./DATA')
DATA_test = DataMerge('./DATA/TestSet',SCALERS=DATA.SCALERS)

DATA_X = np.random.uniform(0,1,(8000,9))
DATA_Y = np.random.uniform(0,1,(8000,2500,5))

saver = train_pickle_saver(window_size=10, sampling_rate = 100, stride= 1000)
A,B,C = saver(DATA_X, DATA_Y)

X_train, X_test, Y_train, Y_test = train_test_split(DATA.Y, test_size=0.3, random_state=SEED)
X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size = 0.9, random_state=SEED)


import tqdm
import pickle
#%%
class train_pickle_saver():
    def __init__(self, window_size=100, sampling_rate = 5, stride = 3):
        self.window_size=window_size
        self.sampling_rate = sampling_rate
        self.stride = stride
    def __call__(self, actions, states, path = './DATA/'): # action : N by 9, state : N by seqlen by 5
        N, SEQLEN,DIM = states.shape
        INPUT = []
        OUTPUT = []
        INPUT_action = []
        idxs = list(range(0,SEQLEN - self.window_size*self.sampling_rate, self.stride))
        for i in tqdm.tqdm(range(N)):
            INPUT = []
            OUTPUT = []
            INPUT_action = []
            for j in idxs:
                state_past = states[i][range(j,j+self.window_size*self.sampling_rate,self.sampling_rate),...]
                state_current = states[i][j+self.window_size*self.sampling_rate,...]
                INPUT.append(state_past)
                OUTPUT.append(state_current)
                # INPUT_action.append(np.tile(actions[i],window_size))
                INPUT_action.append(actions[i])
            INPUT= np.float32(np.array(INPUT))
            OUTPUT = np.float32(np.array(OUTPUT))
            INPUT_action = np.float32(np.array(INPUT_action))
            INP_NAME = '{}_INPUT_win{}_smp{}_str{}.pickle'.format(i,self.window_size,self.sampling_rate,self.stride)
            OUT_NAME = '{}_OUTPUT_win{}_smp{}_str{}.pickle'.format(i,self.window_size,self.sampling_rate,self.stride)
            INPA_NAME = '{}_INPUT_ACTION_win{}_smp{}_str{}.pickle'.format(i,self.window_size,self.sampling_rate,self.stride)
            with open(path+INP_NAME, 'wb') as f:
                pickle.dump(INPUT,f)
            with open(path+OUT_NAME, 'wb') as f:
                pickle.dump(OUTPUT,f)
            with open(path+INPA_NAME, 'wb') as f:
                pickle.dump(INPUT_action,f)
                
#%%


    for s in tqdm.tqdm(idxs):
        seq = new_temp_scaled[range(s,s+SEQLEN*SAMPLINGRATE, SAMPLINGRATE),:]
        X.append(seq)
        X_time.append(new_temp.index[s]
            current_seq = states[i]
            for j in 
            seq_current = []

for i in range(len(DATA_Y)):

    with open('list.txt', 'wb') as f:
...     pickle.dump(list, f)    

