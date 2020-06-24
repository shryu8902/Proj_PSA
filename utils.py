import numpy as np
import pandas as pd

def augment_Y_by_resample(X,Y,sampling_rate = 5, return_all=True):
    temp_y=[]
    for i in range(sampling_rate):
        temp_y.append(Y[:,range(i,2500+i,sampling_rate)])
    if return_all == True:
        Y_ = np.reshape(temp_y,(-1, np.int(2500/sampling_rate)))
        X_ = np.reshape([X for i in range(sampling_rate)],(-1,9))
    else :
        Y_ = temp_y[0]
        X_ = X
    return X_, Y_

def upscaling(Y_low, sampling_rate = 5):
    row, col = np.shape(Y_low)
    Y_up = np.empty((row,sampling_rate*col))
    Y_up[:] = np.NaN
    Y_up[:,range(0,2500,sampling_rate)]=Y_low

    df_for_fillna = pd.DataFrame(Y_up.T)
    df_for_fillna = df_for_fillna.interpolate()
    Y_up_ = np.array(df_for_fillna).T

    return Y_up_

def mean_absolute_percentage_error(y_true, y_pred): 
    ape = np.abs((y_true-y_pred)/y_true)*100
    mape = np.mean(ape)
    std = np.std(ape)
    return mape, std

