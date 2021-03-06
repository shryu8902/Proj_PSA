import numpy as np
import pandas as pd
import tensorflow as tf

def augment_Y_by_resample(X,Y,sampling_rate = 5, return_all=True):
    temp_y=[]
    for i in range(sampling_rate):
        temp_y.append(Y[:,range(i,2500+i,sampling_rate)])
    if return_all == True:
        if len(Y.shape)==3:
            Y_=np.reshape(temp_y,(-1,np.int(2500/sampling_rate),5))
        else:    
            Y_ = np.reshape(temp_y,(-1, np.int(2500/sampling_rate)))
        X_ = np.reshape([X for i in range(sampling_rate)],(-1,9))
    else :
        Y_ = temp_y[0]
        X_ = X
    return X_, Y_

def upscaling(Y_low, sampling_rate = 5):
    if len(Y_low.shape)==2 :
        N, T = np.shape(Y_low)
        Y_up = np.empty((N, sampling_rate*T))
        Y_up[:] = np.NaN
        Y_up[:,range(0,sampling_rate*T,sampling_rate),...]=Y_low
        df_for_fillna = pd.DataFrame(Y_up.T)
        df_for_fillna = df_for_fillna.interpolate()
        Y_up_ = np.array(df_for_fillna).T
        return Y_up_

    elif len(Y_low.shape)==3 :
        N,T,D = np.shape(Y_low)
        Y_up = np.empty((N, sampling_rate*T, D))
        Y_up[:] = np.NaN
        Y_up[:,range(0,sampling_rate*T,sampling_rate),...] = Y_low
        for i in range(D):
            df_for_fillna = pd.DataFrame(Y_up[...,i].T)
            df_for_fillna = df_for_fillna.interpolate()
            Y_up[...,i] = np.array(df_for_fillna).T
        return Y_up
    else:
        print("wrong dimension")
    
def mean_absolute_percentage_error(y_true, y_pred): 
    ape = np.abs((y_true-y_pred)/y_true)*100
    mape = np.mean(ape)
    std = np.std(ape)
    return mape, std

def Seq_wise_MAPE(y_true,y_pred):
    ape = np.abs((y_true-y_pred)/y_true)*100
    mapes = np.mean(ape,axis=1)
    mean = np.mean(mapes)
    std = np.std(mapes)
    return mean, std

# Positional Encodings
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.convert_to_tensor(pos_encoding, dtype=tf.float32)

#%%
def name_create(config):
    if config['bidirect']==True:
        x = '_bi-'
    else:
        x = '_x-'
    if config['layer_norm']==True:
        y = '-ln'
    else:
        y = '-x'
    name = x + config['RNN'] + y
    return name
#%%
def pad_sequence_by_first(sequences, pad_len=20):
    if len(sequences.shape) == 3:
        tiles = np.tile(sequences[:,0,:].reshape(-1,1,5),(1,pad_len,1))
    else:
        tiles = np.tile(sequences[:,0].reshape(-1,1),(1,pad_len))
    padded_sequence = np.concatenate((tiles,sequences),1)
    return padded_sequence
