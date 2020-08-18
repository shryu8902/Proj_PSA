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
DATA = DataMerge('./DATA')
DATA_test = DataMerge('./DATA/TestSet')
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

class FullBatchLoader():
  def __init__(self,root_STATE, root_ACTION, root_OUT):
    self.STATE = np.load(root_STATE)
    self.ACTION = np.load(root_ACTION)
    self.OUT = np.load(root_OUT)

#%% 
# Read full batches for training
TRAIN = FullBatchLoader('./DATA/pickles/minmax/train/STATE.npy',
                        './DATA/pickles/minmax/train/ACTION.npy',
                        './DATA/pickles/minmax/train/OUT.npy')
VAL = FullBatchLoader('./DATA/pickles/minmax/train/val_STATE.npy',
                        './DATA/pickles/minmax/train/val_ACTION.npy',
                        './DATA/pickles/minmax/train/val_OUT.npy')
TEST = FullBatchLoader('./DATA/pickles/minmax/train/test_STATE.npy',
                        './DATA/pickles/minmax/train/test_ACTION.npy',
                        './DATA/pickles/minmax/train/test_OUT.npy')
#%%
train_loss = tf.keras.metrics.Mean(name='train_loss')
optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)

for epoch in range(EPOCH):
  start = time.time()  
  train_loss.reset_states()
  train_accuracy.reset_states()
  for 
    

  for (batch, (inp, tar)) in enumerate(train_dataset):
    train_step(inp, tar)
    
    if batch % 50 == 0:
      print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
          epoch + 1, batch, train_loss.result(), train_accuracy.result()))
      
  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))
    
  print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                train_loss.result(), 
                                                train_accuracy.result()))

  print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

TE_ver1 = Transformer_enc(num_layers=3,d_model = 64,num_heads = 8,dff= 128,rate=0.1)
for i in range(100):
  train_loss.reset_states()
  train_step(TE_ver1,enc_input[:512,...], enc_act_input[:512,...], enc_output[:512,...])
  print(i,train_loss.result())
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
