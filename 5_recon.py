import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from def4rnn import *
import time
import tensorflow as tf
import os
# import matplotlib

np.random.seed(29961)
tf.set_random_seed(29961)
# matplotlib.use('Qt5Agg')

# variable = ['primary_pressure', 'primary_temperature', 'secondary_pressure', 'secondary_temperature', 'PCT']
variable = ['secondary_pressure']

for v in variable:
    allpath = '/media/risk/501AE027635804AB/APR1400_SLOCA_DATA/allinch/{}_all.csv'.format(v)
    indata_df = pd.read_csv(allpath, header=None)
    indata = np.array(indata_df)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_variable = scaler.fit_transform(indata[9:,:].reshape(-1,1))
    scaled_variable = scaled_variable.reshape([2500,-1])                #2500,8004
    scaler1 = MinMaxScaler(feature_range=(0,1))
    scaled_input = scaler1.fit_transform(indata[:9,:].T)

    train_min, train_mout, test_min, test_mout = cea_split(scaled_input.T, scaled_variable, 0.7)
    #min: mars input; mout: mars output

    latent_savepath = os.getcwd() + '/latent_190907_{}.npy'.format(v)
    latent = np.load(latent_savepath)

    # ltx = latent+marsin; marsout

    train_x = train_min; train_y = train_mout
    lenval = int(len(test_min) * 0.1)                             #valdiation length
    val_x = test_min[:lenval,:]; val_y =test_mout[:lenval, :]
    test_x = test_min[lenval:,:]; test_y = test_mout[lenval:,:]

    def makeltX(latent, InX):
        r, _ = InX.shape
        l_ = np.tile(latent, (r,1))
        X = np.concatenate((l_,InX), axis=1)

        return X

    InX = makeltX(latent, train_x)            #merge latent space and input
    Inval = makeltX(latent, val_x)
    Intest = makeltX(latent, test_x)

    codeIN=9
    n_hidden4 = 10
    n_hidden1 = int((n_hidden4 + codeIN) * 5)
    n_hidden2 = int((n_hidden4 + codeIN) * 10)
    n_hidden3 = int((n_hidden4 + codeIN) * 20)
    n_output = train_y.shape[1]

    lr = 0.00005
    epoches = 1
    batchsize = 10

    from tensorflow.contrib import layers
    tf.reset_default_graph()

    ltX = tf.placeholder(tf.float32, [None, (n_hidden4 + codeIN)]) # latent space + code Input
    X = tf.placeholder(tf.float32, [None, n_output])


    def recon(ltX):
        DE_L4 = layers.fully_connected(ltX, n_hidden1, activation_fn=tf.nn.relu, weights_initializer=layers.xavier_initializer())
        DE_L3 = layers.fully_connected(DE_L4, n_hidden2, activation_fn=tf.nn.relu, weights_initializer=layers.xavier_initializer())
        DE_L2 = layers.fully_connected(DE_L3, n_hidden3, activation_fn=tf.nn.relu, weights_initializer=layers.xavier_initializer())
        recon_x = layers.fully_connected(DE_L2, n_output)

        return recon_x

    y_recon = recon(ltX)
    y_true = X

    cost = tf.losses.mean_squared_error(labels=y_true, predictions=y_recon)
    tf.summary.scalar('cost',cost)
    optimizer = tf.train.AdamOptimizer(lr).minimize(cost)

    saver = tf.train.Saver()
    SAVER_DIR = os.getcwd() + '/190916_recon_{}'.format(v)
    checkpoint_path = os.path.join(SAVER_DIR, "parameter_{}".format(v))
    ckpt = tf.train.get_checkpoint_state(SAVER_DIR)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    tensorboard_writer = tf.summary.FileWriter('./tensorboard_log', sess.graph)

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Restored validation cost: %f" %(sess.run(cost, feed_dict={ltX: Inval, X:val_y})))
    else:
        print("TF initializer")
    accepoch = []
    valaccepoch = []
    batch = int(len(train_x)/batchsize)
    startTime = time.time()
    for epoch in range(epoches):
        for i in range(batch):
            start = i * batchsize
            end = start + batchsize
            batchltx = InX[start:end,:]
            batchxs = train_y[start:end,:]
            feed_dict = {ltX: batchltx, X: batchxs}
            _, step_cost = sess.run([optimizer, cost], feed_dict=feed_dict)
            summary = sess.run(merged, feed_dict=feed_dict)
            tensorboard_writer.add_summary(summary, i)


        print("[{} epoch] cost: {}".format(epoch, step_cost))
        print("Validation cost: %f" %(sess.run(cost, feed_dict={ltX: Inval, X:val_y})))

        accepoch.append(step_cost)
        valaccepoch.append(sess.run(cost, feed_dict={ltX: Inval, X: val_y}))

    endTime = (time.time() - startTime) / 60
    print("[epoch {} epoch calculating time : {} min ] ".format(epoch, round(endTime, 2)))
    saver.save(sess, checkpoint_path)

    testPred = []       #prediction of normalizatio y without moving average
    pred_ny = []        #prediction of normalization y with moving average
    pred_y = []         #prediction of y with moving average
    true_ny = []        #true value of normalization y
    # true = []           #true value of y
    acc = []            #accuracy comparing pred_y and true
    inchs = []

    _, _, test_nx, test_rawy = cea_split(scaled_input.T, indata[9:,:], 0.7)
    test_nx = test_nx[lenval:,:]; test_rawy = test_rawy[lenval:,:]

    pred_x = makeltX(latent, test_nx)

    # for i in range(len(pred_x)):
    for i in range(10):
        _testPred = sess.run(y_recon, feed_dict={ltX: pred_x[i].reshape(1,-1)})
        testPred.append(_testPred)

        # normalization y
        _y = test_rawy[i].reshape(1,-1)
        # _scaler = MinMaxScaler(feature_range=(0, 1))
        _s = scaler.transform(_y.T)
        _true_ny = sess.run(y_true, feed_dict={X: _s.T})
        true_ny.append(_true_ny)
######
        # moving average to remove outlier
        _df_pred = pd.DataFrame(_testPred[0, :])
        _r_pred = _df_pred.rolling(window=25, min_periods=1).mean()
        mv_pred = np.array(_r_pred)                         # shape (2500,1)
        pred_ny.append(mv_pred.T)

        # _pred_y = _scaler.inverse_transform(mv_pred.T)
        _pred_y = scaler.inverse_transform(mv_pred.T)
        pred_y.append(_pred_y)
        # test_ny = test_ny.T

        _acc = mean_absolute_percentage_error(_y, _pred_y)
        acc.append(_acc)

        inch = (Intest[i].reshape(1,-1)[0,10]) * 1.5 + 0.5
        inchs.append(inch)

        # plt.figure()            #(2500,1)
        # plt.plot(_true_ny.T, label='True')
        # plt.title(str(inch) + 'inch LOCA')
        # # plt.plot(testPred[0,:])
        # plt.plot(mv_pred, label='Prediction')
        # plt.legend(loc=0)
        # plt.xlabel('Time (sec)')
        # plt.ylabel('Normalized {}'.format(v))
        # plt.show()

        plt.figure()
        plt.plot(_y.T, label='True')
        plt.title(str(inch) + 'inch LOCA')
        plt.plot(_pred_y.T, label='Prediction')
        plt.xlabel('Time (sec)')
        plt.ylabel('{}'.format(v))
        plt.ylim(scaler.data_min_, scaler.data_max_)
        plt.legend(loc=0)
        plt.show()

        # file_path = os.getcwd() + '/recon_{}'.format(v)
        # fig_path = file_path + '/190916_recon_%i.png' %i
        # plt.savefig(fig_path)
        # plt.close()

    # acc = np.array(acc)
    # inch = np.array(inchs)
    # predict = np.array(pred_y)
    # predict = predict.reshape(-1,2500)
    # np.savetxt("acc_190916_{}.csv".format(v), acc, delimiter=",")
    # np.savetxt("inch_190916_{}.csv".format(v), inch, delimiter=",")
    # np.savetxt("predict_190916_{}".format(v), predict, delimiter=",")
    # np.savetxt("target_190916_{}.csv".format(v), test_rawy, delimiter=",")


    ## later, make performance metrics R2score, mse
    ## and pick up most higher error



    #%%
import numpy as np
import pandas as pd
import os, time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import RepeatVector,LSTM, TimeDistributed, Dense, Activation, BatchNormalization, InputLayer,Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

#%%
SEED=0
np.random.seed(SEED)
tf.random.set_seed(SEED)
#%%
variable = ['primary_pressure', 'primary_temperature', 'secondary_pressure', 'secondary_temperature', 'PCT']

#%%

v = variable[0]

allpath = './DATA/{}_all.csv'.format(v)
DATA = np.array(pd.read_csv(allpath, header=None)).T
DATA_input = DATA[:,:9]
DATA_output = DATA[:,9:]
input_scaler_mm = MinMaxScaler()
input_scaler_std = StandardScaler()
# output_scaler = MinMaxScaler()
output_scaler = StandardScaler()
DATA_input_mm = input_scaler_mm.fit_transform(DATA_input)
DATA_input_std = input_scaler_std.fit_transform(DATA_input)
DATA_output_n = output_scaler.fit_transform(DATA_output.reshape(-1,1)).reshape(-1,2500,1)

#%%
for i in range(len(DATA_output)):
    plt.plot(DATA_output[i,:])

#%% train/val/test : 0.7/0.15/0.15
X_train, X_test, Y_train, Y_test = train_test_split(DATA_input_mm, DATA_output_n, test_size=0.3, random_state=SEED)
X_val, X_test, Y_val, Y_test = train_test_split(X_test,Y_test, test_size = 0.5, random_state=SEED)

def augmentation_by_resample(X,Y):
    

#%% Model
K.clear_session()
inputs = layers.Input(shape =(9) ,name='input')
inputs_extend = RepeatVector(500)(inputs)

layer_1 = Bidirectional(LSTM(128, return_sequences=True))(inputs_extend)
layer_2 = Bidirectional(LSTM(64, return_sequences=True))(layer_1)
layer_3 = TimeDistributed(Dense(64,activation='elu'))(layer_2)
outputs = TimeDistributed(Dense(1))(layer_3)
# outputs = LSTM(1, return_sequences=True)(layer_2)

model= Model(inputs, outputs, name='LSTM')
model.summary()
#%%
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_absolute_error','mean_squared_error'])

history = model.fit(X_train, Y_train, validation_data=(X_val,Y_val),batch_size = 512, epochs=200)

#%%

Y_hat=model.predict(X_val)
Y_hat=Y_hat.reshape(-1,500)
#%%
i=50
plt.plot(Y_hat[i,:])
plt.plot(Y_test_[i,:])
#%%
Y_hat_n = model.predict(X_test)
Y_hat = output_scaler.inverse_transform(Y_hat_n.reshape(-1,1)).reshape(-1,500)
Y_test_ = output_scaler.inverse_transform(Y_test.reshape(-1,1)).reshape(-1,500)

#%%


fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

plt.show()
#%%
Y_hat=model.predict(X_val)
Y_hat=Y_hat.reshape(-1,500)
#%%
i=50
plt.plot(Y_hat[i,:])
plt.plot(Y_test_[i,:])
#%%
Y_hat_n = model.predict(X_test)
Y_hat = output_scaler.inverse_transform(Y_hat_n.reshape(-1,1)).reshape(-1,500)
Y_test_ = output_scaler.inverse_transform(Y_test.reshape(-1,1)).reshape(-1,500)



#%%

def mean_absolute_percentage_error(y_true, y_pred): 
    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)
    ape = np.abs((y_true-y_pred)/y_true)*100
    mape = np.mean(ape)
    std = np.std(ape)
    return mape, std
#%%
mape, std = mean_absolute_percentage_error(Y_test_,Y_hat)
