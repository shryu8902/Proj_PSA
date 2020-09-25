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
from tensorflow.keras.layers import RepeatVector,Lambda,LSTM, GRU, SimpleRNN, TimeDistributed, Dense, Activation, BatchNormalization, InputLayer,Bidirectional, LayerNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
Val_Results = pd.DataFrame(columns=['config','PP_avg','PP_std','PT_avg','PT_std','SP_avg','SP_std','ST_avg','ST_std','PCT_avg','PCT_std'])
#%%
DATA = DataMerge('./DATA',SCALE='minmax')
DATA_test = DataMerge('./DATA/TestSet',DATA.SCALERS)
# BG = BatchGenerator(DATA.X, DATA.Y, batch_size=512, window_size = 100,sampling_rate=5,
#                 stride = 5, shuffle= True, seed=SEED)
# TEST_BG = BatchGenerator(DATA_test.X, DATA_test.Y, batch_size = 512,window_size = 100, sampling_rate = 5,
#                 stride = 5, shuffle = True, seed=SEED)
# TEST_DATA = DataMerge('./DATA/TestSet',DATA.SCALERS)
#%%
BATCH = 1024
WINDOW = 100
SAMPLING = 5
STRIDE = 3
LR = 1e-4
EPOCH = 10
NUM_CELL = 512
DR_RATES = 0.3
NUM_CH = 5
#%% Train/test splits
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(DATA.X, DATA.Y, test_size=0.3, random_state=SEED)
X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size = 0.9, random_state=SEED)
#%% Create model
NUM_CELL = 256
NUM_CH = 5
DR_RATES = 0.3
BATCH_SIZE = 100
SEQ_SIZE = 100

#Get input
action = Input(shape=(None,9,),batch_size=BATCH_SIZE)
state = Input(shape=(None,5,),batch_size=BATCH_SIZE)

#Concat action and states
act_state = tf.concat([action, state],2)
#models
layer_1 = LSTM(NUM_CELL, return_sequences=True, stateful=True, dropout=DR_RATES)(act_state)
layer_2 = LSTM(NUM_CELL, return_sequences=True,stateful=True,dropout=DR_RATES)(layer_1)
# layer_3 = LSTM(num_cell, return_sequences=True,dropout=dr_rates)(layer_2)
output = TimeDistributed(Dense(NUM_CH, activation='elu'))(layer_2)

model= Model([action, state], output)
model.compile(optimizer='adam',
            loss='mean_squared_error',
            metrics=['mean_absolute_error','mean_squared_error'])
#%%
EPOCH = 100

for epochs in range(EPOCH):
    len_seq_batch = X_train.shape[0]//100
    for seq_batch in range(len_seq_batch):
        selected_sequence = np.random.choice(X_train.shape[0],100,replace=False)
        seq_batch_x = np.repeat(X_train[selected_sequence,np.newaxis,:],100,axis=1)
        seq_batch_y = Y_train[selected_sequence,::5,:]
        seq_batch_y_0 = seq_batch_y[:,:-1,]
        seq_batch_y_1 = seq_batch_y[:,1:,]
        len_perseq_batch = seq_batch_y.shape[1]//SEQ_SIZE
        for perseq_batch in range(len_perseq_batch):
            print(perseq_batch)
            strt = perseq_batch * SEQ_SIZE
            end = (perseq_batch+1) * SEQ_SIZE
            perseq_batch_input = seq_batch_y_0[:,strt:end,:]           
            perseq_batch_output = seq_batch_y_1[:,strt:end,:]
            
            hist = model.fit([seq_batch_x[:,:perseq_batch_input.shape[1],:], perseq_batch_input],perseq_batch_output,batch_size=BATCH_SIZE, epochs=1)
        model.reset_states()




#%% Training Phase
path= './Models/sequential_lstm/stateful-{epoch:04d}.ckpt'
cp_dir = os.path.dirname(path)
checkpoint = ModelCheckpoint(path,             # file명을 지정합니다
                             monitor = 'val_loss',   # val_loss 값이 개선되었을때 호출됩니다
                             verbose = 1,            # 로그를 출력합니다
                             save_best_only = True,  # 가장 best 값만 저장합니다
                             mode = 'auto',           # auto는 알아서 best를 찾습니다. min/max
                             save_weights_only = True
                            )



#%%
Train_BG = BatchGenerator(X_train,Y_train, batch_size = BATCH, window_size = WINDOW, sampling_rate = SAMPLING, stride = STRIDE, shuffle = True )
Val_BG = BatchGenerator(X_val, Y_val,batch_size = BATCH, window_size = WINDOW, sampling_rate = SAMPLING, stride = STRIDE, shuffle = True )


#%% Training Phase

path = './Models/sequential_lstm/m1-{epoch:04d}.ckpt'
cp_dir = os.path.dirname(path)
checkpoint = ModelCheckpoint(path,             # file명을 지정합니다
                             monitor = 'val_loss',   # val_loss 값이 개선되었을때 호출됩니다
                             verbose = 1,            # 로그를 출력합니다
                             save_best_only = True,  # 가장 best 값만 저장합니다
                             mode = 'auto',           # auto는 알아서 best를 찾습니다. min/max
                             save_weights_only = True
                            )
model.compile(optimizer='adam',
            loss='mean_squared_error',
            metrics=['mean_absolute_error','mean_squared_error'])
history = model.fit(Train_BG,epochs=EPOCH, validation_data= Val_BG, callbacks=[checkpoint])

#%% Sampling Rate = 5 [action_in, state_in]
# action_in : shape  = N X Window X 9
# state_in : shape = N X Window X 5

# Test_X = DATA_test.X
# Test_Y_samp = DATA_test.Y[:,range(0,2500,SAMPLING),...]
Test_X = X_test
Test_Y_samp = Y_test[:,range(0,2500,SAMPLING),...]


TEMP_OUTPUT = np.empty((np.shape(Test_X)[0], int(2500/SAMPLING),5))
TEMP_OUTPUT[:,:100,:] = Test_Y_samp[:,:100,:]
for i in range(100,500):    
    test_action = np.tile(Test_X[:,np.newaxis,:],(1,100,1))
    test_state = TEMP_OUTPUT[:,(i-100):i,:]
    output = model.predict([test_action, test_state])    
    TEMP_OUTPUT[:,i,:] = output[:,99,:]

#%%
i=2000
dim=0
plt.plot(Test_Y_samp[i,:,dim],label='Real')
plt.plot(TEMP_OUTPUT[i,:,dim],label='Pred')
plt.legend()
#%% Test Phase

#%%
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle = True,random_state=SEED)
kf.get_n_splits(DATA.X)

for train_index, val_index in kf.split(DATA.X):

    X_train, X_val = DATA.X[train_index], DATA.X[val_index]
    Y_train, Y_val = DATA.Y[train_index], DATA.Y[val_index]

    Train_BG = BatchGenerator(X_train,Y_train, batch_size= 1024, window_size= 100, sampling_rate = 5,
                    stride = 3, shuffle=True )
    Val_BG = BatchGenerator(X_val,Y_val, batch_size= 1024, window_size= 100, sampling_rate = 5,
                    stride = 3, shuffle=True )
    model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['mean_absolute_error','mean_squared_error'])
    history = model.fit(Train_BG,epochs=1, validation_data= Val_BG)

X_DATA = X_val
Y_DATA = Y_val[:,range(0,2500,5),:]

#%%

model = 

for x in range(time_window):
    action # N,9
    state # N, first few steps, 5


#%%
##Build model

WINDOW = 100
BATCH = 1024
LR = 0.001
EPOCH = 100
NUM_CELL = 512
DR_RATES = 0.3
NUM_CH = 5

#%%
#Get input
action = Input(shape=(None,9,))
state = Input(shape=(None,5,))

#Concat action and states
act_state = tf.concat([action, state],2)
#models
layer_1 = LSTM(NUM_CELL, return_sequences=True, stateful=True, dropout=DR_RATES)(act_state)
layer_2 = LSTM(NUM_CELL, return_sequences=True,stateful=True,dropout=DR_RATES)(layer_1)
# layer_3 = LSTM(num_cell, return_sequences=True,dropout=dr_rates)(layer_2)
output = TimeDistributed(Dense(NUM_CH, activation='elu'))(layer_2)

model= Model([action, state], output)

#%%









tr = []
te = []
for train, test in kf.split(DATA.X):
    X[train]

DataGenerator
X=DATA.X
Y=DATA.Y
len(X)
sequence_pool_current = []
sequence_pool_next = [] 
sequence_pool_action = []

len(Y)
#%%

#%%

## Train / Validation / Test Split
#% train/val/test : 0.7/0.03/0.27
X_train, X_test, Y_train, Y_test = train_test_split(DATA_input_std, DATA_output_n, test_size=0.3, random_state=SEED)
X_val, X_test, Y_val, Y_test = train_test_split(X_test,Y_test, test_size = 0.9, random_state=SEED)
## Reduce time sequence to 1/5 by sampling
## Hence ther will be 5 sequence for 1 input data.
X_train,Y_train = augment_Y_by_resample(X_train,Y_train,5,return_all=True)
X_val_base = X_val; Y_val_base = Y_val
X_val,Y_val = augment_Y_by_resample(X_val,Y_val,5,return_all=True)
# X_test,Y_test = augment_Y_by_resample(X_test,Y_test,5,return_all=False)

## Pad for idling
## Y_train,Y_val
Y_train = pad_sequence_by_first(Y_train,20)        
Y_val = pad_sequence_by_first(Y_val,20)

## Create model
K.clear_session()

## Check model exists
file_name = name_create(config)
file_root = './DATA/Models/IDLE'+v+file_name+'_wPE.hdf5'
# file_root = './DATA/Models/test.h5'
# if model exist, load model
# if os.path.exists(file_root):
#     model = load_model(file_root) 
# else: #else train new model
model = TS_model_wPE_idle(bidirect=config['bidirect'], RNN = config['RNN'], 
                    layer_norm = config['layer_norm'], dr_rates=0.3, num_pads=20)
model.compile(optimizer='adam',
        loss='mean_squared_error',
        metrics=['mean_absolute_error','mean_squared_error']) 
# model2 = TS_model_LN(bidirect=config['bidirect'])
# model2.load_weights(file_root,by_name=True)

## Model training
model_save = ModelCheckpoint(file_root, save_weights_only=True, save_best_only=True, monitor='val_loss', mode='min')

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, Y_train.reshape(-1,520,1), 
                validation_data=(X_val,Y_val),
                batch_size = 512, epochs=100,
                callbacks=[early_stopping,model_save])
