#%% 
import tensorflow as tf
import numpy as np
from utils import *
from sequential_utils import *
from transformer_net import *
import tqdm
import pickle
from sklearn.model_selection import train_test_split
import time
import os
os.chdir('/home/ashryu/Proj_PSA')
#%%
os.environ["CUDA_VISIBLE_DEVICES"]='5'
BATCH_SIZE=512
SEED=0
#%%
DATA = DataMerge('./DATA',SCALE='minmax')
DATA_test = DataMerge('./DATA/TestSet',SCALERS = DATA.SCALERS)
train_ind, test_ind = train_test_split(list(range(8004)),test_size=0.3, random_state=SEED)
val_ind, test_ind = train_test_split(test_ind,test_size=0.9, random_state=SEED)
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
def val_step(model, input_state, input_action, output_real):
  predictions = model(input_state, input_action,False,None,None,None)
  loss = tf.keras.losses.MSE(output_real,predictions)
  val_loss(loss)
#%% 
# Read full batches for training
TRAIN = FullBatchLoader('./DATA/pickles/minmax/train/STATE.npy',
                        './DATA/pickles/minmax/train/ACTION.npy',
                        './DATA/pickles/minmax/train/OUT.npy',
                        BATCH=BATCH_SIZE, shuffle=True, seed=0)
# VAL = FullBatchLoader('./DATA/pickles/minmax/train/val_STATE.npy',
#                         './DATA/pickles/minmax/train/val_ACTION.npy',
#                         './DATA/pickles/minmax/train/val_OUT.npy',
#                         BATCH=BATCH_SIZE, shuffle=False, seed=0)
# TEST = FullBatchLoader('./DATA/pickles/minmax/train/test_STATE.npy',
#                         './DATA/pickles/minmax/train/test_ACTION.npy',
#                         './DATA/pickles/minmax/train/test_OUT.npy',
#                         BATCH=BATCH_SIZE, shuffle=False, seed=0)
#%% 
# Training Stage

EPOCH=70
TRANSFORMER = Transformer_enc(num_layers=3,d_model = 128,num_heads = 16,dff= 256,rate=0.1)
train_loss = tf.keras.metrics.Mean(name='train_loss')
optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
checkpoint_path = "./Models/transformer_ver2"
ckpt = tf.train.Checkpoint(transformer=TRANSFORMER,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path,max_to_keep=None)

for epoch in range(EPOCH):
  start = time.time()  
  train_loss.reset_states()
  for (batch, (inp, outp)) in enumerate(TRAIN):
    train_step(TRANSFORMER, inp[0], inp[1], outp, optimizer)

  ckpt_save_path = ckpt_manager.save()
  print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,ckpt_save_path))    
  print ('Epoch {} Loss {:.6f}'.format(epoch + 1, train_loss.result()))
  print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))