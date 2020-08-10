#%%
import numpy as np
import pandas as pd
import os, time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from matplotlib import pyplot as plt
import time
from utils import *
from sequential_utils import *
import tensorflow as tf
from tensorflow.keras import layers, Input, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import RepeatVector,LSTM, GRU, SimpleRNN, TimeDistributed, Dense, Activation, BatchNormalization, InputLayer,Bidirectional, LayerNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
#%%
variable = ['primary_pressure', 'primary_temperature', 'secondary_pressure', 'secondary_temperature', 'PCT']

DATA = DataMerge('./DATA',SCALE='minmax')
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
