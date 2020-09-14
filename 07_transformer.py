#%% 
import tensorflow as tf
import numpy as np
from utils import *
from sequential_utils import *
from transformer_net import *
import tqdm
import pickle


#%%
N = 100
SEQLEN= 10
IN_DIM = 9
OUT_DIM = 5
BATCH_SIZE = 512
INP_NAME = '{}_INPUT_win{}_smp{}_str{}.pickle'.format(i,self.window_size,self.sampling_rate,self.stride)
#%%
with open('./DATA/pickles/minmax/train/0_INPUT_win50_smp5_str2.pickle', 'rb') as f:
    enc_input = pickle.load(f)
with open('./DATA/pickles/minmax/train/0_OUTPUT_win50_smp5_str2.pickle', 'rb') as f:
    enc_output = pickle.load(f)
with open('./DATA/pickles/minmax/train/0_INPUT_ACTION_win50_smp5_str2.pickle', 'rb') as f:
    enc_act_input = pickle.load(f)
#%%
SEED=0
#%%
from sklearn.model_selection import train_test_split
DATA = DataMerge('./DATA',SCALE='minmax')
DATA_test = DataMerge('./DATA/TestSet',SCALERS = DATA.SCALERS)
train_ind, test_ind = train_test_split(list(range(8004)),test_size=0.3, random_state=SEED)
val_ind, test_ind = train_test_split(test_ind,test_size=0.9, random_state=SEED)
#%%
TRAIN = DataLoader(train_ind)
np.save('./DATA/pickles/minmax/train/STATE',TRAIN.STATE)
np.save('./DATA/pickles/minmax/train/ACTION',TRAIN.ACTION)
np.save('./DATA/pickles/minmax/train/OUT',TRAIN.OUT)

VAL = DataLoader(val_ind)
np.save('./DATA/pickles/minmax/train/val_STATE',VAL.STATE)
np.save('./DATA/pickles/minmax/train/val_ACTION',VAL.ACTION)
np.save('./DATA/pickles/minmax/train/val_OUT',VAL.OUT)

TEST = DataLoader(test_ind)
np.save('./DATA/pickles/minmax/train/test_STATE',TEST.STATE)
np.save('./DATA/pickles/minmax/train/test_ACTION',TEST.ACTION)
np.save('./DATA/pickles/minmax/train/test_OUT',TEST.OUT)
#%%
window_size=50, sampling_rate = 5, stride= 2
#%%
class DataLoader():
  def __init__(self, index):
    self.index=index
    self.start_flag = True

    for i in tqdm.tqdm(self.index):
      with open('./DATA/pickles/minmax/train/{}_INPUT_win50_smp5_str2.pickle'.format(i), 'rb') as f:
        temp_state = pickle.load(f)
      with open('./DATA/pickles/minmax/train/{}_OUTPUT_win50_smp5_str2.pickle'.format(i), 'rb') as f:
        temp_out = pickle.load(f)
      with open('./DATA/pickles/minmax/train/{}_INPUT_ACTION_win50_smp5_str2.pickle'.format(i), 'rb') as f:
        temp_action = pickle.load(f)
      if self.start_flag:
        STATE = temp_state
        ACTION = temp_action
        OUT = temp_out
        self.start_flag = False
      else:
        STATE=np.concatenate([STATE,temp_state])
        ACTION=np.concatenate([ACTION,temp_action])
        OUT=np.concatenate([OUT,temp_out])
    self.STATE =STATE
    self.ACTION = ACTION
    self.OUT = OUT

class FullBatchLoader(tf.keras.utils.Sequence):
  def __init__(self,root_STATE, root_ACTION, root_OUT, BATCH, shuffle = True, seed=0):
    self.STATE = np.load(root_STATE)
    self.ACTION = np.load(root_ACTION)
    self.OUT = np.load(root_OUT)
    self.batch_size = BATCH
    self.seed=seed
    self.shuffle = shuffle
    self.indices = list(range(len(self.STATE)))
  def on_epoch_end(self):
    if self.shuffle:
      np.random.shuffle(self.indices)
  def __len__(self):    
    return len(self.indices) // self.batch_size
  def __getitem__(self,index):
    sample_index = self.indices[index*self.batch_size : (index+1)*self.batch_size]    
    return ([self.STATE[sample_index],self.ACTION[sample_index]],self.OUT[sample_index])

#%% 
# Read full batches for training
TRAIN = FullBatchLoader('./DATA/pickles/minmax/train/STATE.npy',
                        './DATA/pickles/minmax/train/ACTION.npy',
                        './DATA/pickles/minmax/train/OUT.npy',
                        BATCH=BATCH_SIZE, shuffle=False, seed=0)
VAL = FullBatchLoader('./DATA/pickles/minmax/train/val_STATE.npy',
                        './DATA/pickles/minmax/train/val_ACTION.npy',
                        './DATA/pickles/minmax/train/val_OUT.npy',
                        BATCH=BATCH_SIZE, shuffle=False, seed=0)
TEST = FullBatchLoader('./DATA/pickles/minmax/train/test_STATE.npy',
                        './DATA/pickles/minmax/train/test_ACTION.npy',
                        './DATA/pickles/minmax/train/test_OUT.npy',
                        BATCH=BATCH_SIZE, shuffle=False, seed=0)



#%% 
# Training Stage

EPOCH=20 
import time
checkpoint_path = "./Models/transformer"
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

TE_ver2 = Transformer_enc(num_layers=3,d_model = 128,num_heads = 16,dff= 256,rate=0.1)
train_loss = tf.keras.metrics.Mean(name='train_loss')
optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
checkpoint_path = "./Models/transformer"
ckpt = tf.train.Checkpoint(transformer=TE_ver1,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

for epoch in range(10):
  start = time.time()  
  train_loss.reset_states()
  for (batch, (inp, outp)) in enumerate(TRAIN):
    train_step(TE_ver1, inp[0], inp[1], outp, optimizer)
    # print(batch)  
    # if batch % 50 == 0:
    #   print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
    #       epoch + 1, batch, train_loss.result(), train_accuracy.result()))
      
  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))
    
  print ('Epoch {} Loss {:.4f}'.format(epoch + 1, train_loss.result()))
  print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

for i in range(100):
  train_loss.reset_states()
  train_step(TE_ver1,enc_input[:512,...], enc_act_input[:512,...], enc_output[:512,...])
  print(i,train_loss.result())

#%%
# Inference stage
DATA_test.X : N by 9
DATA_test.Y : N by 2500 by 9




SAMPLED_OUTPUT = np.float32(DATA_test.Y[:,range(0,2500,5)])
Test_X = np.float32(DATA_test.X)

TEST_OUTPUT = np.empty((np.shape(SAMPLED_OUTPUT)))
TEST_OUTPUT[:,:50,:]=SAMPLED_OUTPUT[:,:50,:]

for i in range(50,500):
    action = np.float32(DATA_test.X)
    state = TEST_OUTPUT[:,(i-50):i,:] 
    output = TE_ver1(state,action,False,None,None,None)
    TEST_OUTPUT[:,i,:] = output

#%%
index=43
channel = 1
plt.plot(TEST_OUTPUT[index,:,channel],label='transformer')
plt.plot(SAMPLED_OUTPUT[index,:,channel],label='real')
plt.legend()
plt.grid()

#%% training_phase 
# @tf.function(input_signature=train_step_signature)
def train_step(model, input_state, input_action, output_real, optimizer):
  with tf.GradientTape() as tape:
    # model(state, action, training, enc_padding_mask, 
    # look_ahead_mask, dec_padding_mask)
    predictions = model(input_state, input_action, 
                                 True, 
                                 None, 
                                 None, 
                                 None)
    # loss = tf.keras.loss_function(tar_real, predictions)
    loss =tf.keras.losses.MSE(output_real, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)    
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  
  train_loss(loss)
  # train_accuracy(tar_real, predictions)

#%%
class Transformer_enc(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
    super(Transformer_enc, self).__init__()
    self.embedding = tf.keras.layers.Dense(d_model)
    self.encoder = Encoder(num_layers, d_model, num_heads, dff, rate)
    # self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
    #                        target_vocab_size, pe_target, rate)
    self.seq_flat = tf.keras.layers.Flatten()
    self.sub_final = tf.keras.layers.Dense(32)
    self.concat = tf.keras.layers.Concatenate(axis=-1)
    self.final_layer = tf.keras.layers.Dense(5)

  def call(self, state, action, training, enc_padding_mask, 
           look_ahead_mask, dec_padding_mask):
    embed_state = self.embedding(state)
    enc_output = self.encoder(embed_state, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    flat = self.seq_flat(enc_output)
    seq_feat_combine = self.sub_final(flat)
    action_state = self.concat([seq_feat_combine, action])
    final_output = self.final_layer(action_state)
    # (batch_size, 5)
    # # dec_output.shape == (batch_size, tar_seq_len, d_model)
    # dec_output, attention_weights = self.decoder(
    #     tar, enc_output, training, look_ahead_mask, dec_padding_mask)    
    # final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)   
    return final_output

#%%
class Transformer_enc(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1, **kwargs):
    super(Transformer_enc, self).__init__(**kwargs)
    self.embedding = tf.keras.layers.Dense(d_model)
    self.encoder = Encoder(num_layers, d_model, num_heads, dff, rate)
    # self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
    #                        target_vocab_size, pe_target, rate)
    self.seq_flat = tf.keras.layers.Flatten()
    self.sub_final = tf.keras.layers.Dense(32)
    self.concat = tf.keras.layers.Concatenate(axis=-1)
    self.final_layer = tf.keras.layers.Dense(5)
    
  def call(self, state, action, training, enc_padding_mask, 
           look_ahead_mask, dec_padding_mask):
    embed_state = self.embedding(state)
    enc_output = self.encoder(embed_state, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    flat = self.seq_flat(enc_output)
    seq_feat_combine = self.sub_final(flat)
    action_state = self.concat([seq_feat_combine, action])
    final_output = self.final_layer(action_state)
    # (batch_size, 5)
    # # dec_output.shape == (batch_size, tar_seq_len, d_model)
    # dec_output, attention_weights = self.decoder(
    #     tar, enc_output, training, look_ahead_mask, dec_padding_mask)    
    # final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)   
    return final_output  

  def train_step(self, data):
    with tf.GradientTape() as tape:
      state = data[0]
      action= data[1]
      predictions = self.call(state,action,True,None,None,None)
    loss =tf.keras.losses.MSE(output_real, predictions)
    gradients = tape.gradient(loss, self.trainable_variables)    
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    return{"loss": loss}  


#%%
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import tensorflow as tf

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs): 
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def Encoder(SEQLEN,DIM):   
    inputs = keras.Input(shape=(SEQLEN,DIM))
    l1 = layers.Flatten()(inputs)
    l2 = layers.Dense(128)(l1)
    l3 = layers.BatchNormalization()(l2)
    l4 = layers.LeakyReLU()(l3)
    l5 = layers.Dense(64)(l4)
    l6 = layers.BatchNormalization()(l5)
    l7 = layers.LeakyReLU()(l6)
    l8 = layers.Dense(32)(l7)
    z_mean = layers.Dense(32, name="z_mean")(l8)
    z_log_var = layers.Dense(32, name="z_log_var")(l8)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(inputs,[z_mean,z_log_var,z])
    return encoder

def Decoder(SEQLEN,DIM):
    inputs = keras.Input(shape=(32))
    d_l1 = layers.Dense(64)(inputs)
    d_l2 = layers.BatchNormalization()(d_l1)
    d_l3 = layers.LeakyReLU()(d_l2)
    d_l4 = layers.Dense(128)(d_l3)
    d_l5 = layers.BatchNormalization()(d_l4)
    d_l6 = layers.LeakyReLU()(d_l5)
    d_l7 = layers.Dense(SEQLEN*DIM)(d_l6)
    dec_out= layers.Reshape((SEQLEN,DIM))(d_l7)
    decoder = keras.Model(inputs,dec_out)
    return decoder

class VAE(keras.Model):
    def __init__(self, SEQLEN = 30, DIM = 221, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.SEQLEN = SEQLEN
        self.DIM = DIM
        self.encoder = Encoder(self.SEQLEN, self.DIM)
        self.decoder = Decoder(self.SEQLEN, self.DIM)

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.MSE(data, reconstruction)
            )
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            # "batch":batch_n,
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def call(self,inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z_mean)
        return ([z_mean,z_log_var,z], reconstruction)
