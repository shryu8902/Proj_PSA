from tensorflow.keras.utils import Sequence
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataMerge():
    def __init__(self, base_path='./DATA', SCALERS=None):
        self.variable = ['primary_pressure', 'primary_temperature', 'secondary_pressure', 'secondary_temperature', 'PCT']
        self.base_path = base_path
        self.RAW_data = {}
        self.PROC_data = {}
        if SCALERS is None:
            self.SCALERS = {}
        else :
            self.SCALERS = SCALERS

        for i,v in enumerate(self.variable):
            root = self.base_path+'/{}_all.csv'.format(v)
            temp_data = np.array(pd.read_csv(root, header=None)).T
            if i ==0:
                self.RAW_data['action']=temp_data[:,:9]
                if SCALERS is None:
                    self.SCALERS['action']=StandardScaler()
                    self.PROC_data['action']=self.SCALERS['action'].fit_transform(self.RAW_data['action'])
                else:
                    self.PROC_data['action']=self.SCALERS['action'].transform(self.RAW_data['action'])

            self.RAW_data[v]=temp_data[:,9:]
            if SCALERS is None:
                self.SCALERS[v]=StandardScaler()
                self.PROC_data[v]=self.SCALERS[v].fit_transform(self.RAW_data[v].reshape(-1,1)).reshape(-1,2500)
            else:
                self.PROC_data[v]=self.SCALERS[v].transform(self.RAW_data[v].reshape(-1,1)).reshape(-1,2500)

        self.RAW_Y = np.stack([self.RAW_data[v] for v in self.variable], axis = 2)
        self.Y = np.stack([self.PROC_data[v] for v in self.variable], axis=2)
        self.X = self.PROC_data['action']

# class BatchGenerator2(Sequence):
#     def __init__(self, X, Y, batch_size = 512, seed = 0):
#         self.X = X
#         self.Y = Y
#         self.shuffle = shuffle
#         self.batch_size = batch_size
#         self.seed = seed
#         self.len_Y, self.len_seq, _ = self.Y.shape
#         self.indices = list(range(0, self.len_Y))

#     def on_epoch_end(self):
#         if self.shuffle:
#             np.random.shuffle(self.indices)
    
#     def __len__(self):
#         return len(self.indices) // self.batch_size

#     def __getitem__(self,index):
#         sample_index = self.indices[index*self.batch_size : (index+1)*self.batch_size]
#         for i in sample_index:
            
#         # return([action])
#         seq_current = []
#         seq_next = []
#         seq_action = []
#         for i in sample_index:
#             sequence_index = i//self.len_seq_limit
#             time_index = i%self.len_seq_limit
#             seq_current.append(self.Y[sequence_index, range(time_index, time_index+self.window_size*self.sampling_rate,self.sampling_rate),:])
#             seq_next.append(self.Y[sequence_index,range(time_index + self.sampling_rate, time_index + (self.window_size+1)*self.sampling_rate, self.sampling_rate),:])
#             seq_action.append(np.tile(self.X[sequence_index],(self.window_size,1))) 
#         action_in = np.array(seq_action,dtype='f')
#         state_in = np.array(seq_current,dtype='f')        
#         state_out = np.array(seq_next,dtype='f')
#         # return [action_in, state_in], state_out, sample_index
#         return ([action_in, state_in], state_out)

class BatchGenerator(Sequence):
    def __init__(self, X, Y, batch_size=512, window_size=100, sampling_rate = 5, stride = 5, shuffle = True, seed = 0):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.stride = stride
        self.shuffle = shuffle
        self.seed=seed
        self.len_Y, self.len_seq, _ = self.Y.shape
        self.len_seq_limit = self.len_seq - self.window_size*self.sampling_rate
        self.indices = list(range(0,self.len_Y*self.len_seq_limit,self.stride))
        np.random.seed(self.seed)
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self,index):
        sample_index = self.indices[index*self.batch_size : (index+1)*self.batch_size]
        seq_current = []
        seq_next = []
        seq_action = []
        for i in sample_index:
            sequence_index = i//self.len_seq_limit
            time_index = i%self.len_seq_limit
            seq_current.append(self.Y[sequence_index, range(time_index, time_index+self.window_size*self.sampling_rate,self.sampling_rate),:])
            seq_next.append(self.Y[sequence_index,range(time_index + self.sampling_rate, time_index + (self.window_size+1)*self.sampling_rate, self.sampling_rate),:])
            seq_action.append(np.tile(self.X[sequence_index],(self.window_size,1))) 
        action_in = np.array(seq_action,dtype='f')
        state_in = np.array(seq_current,dtype='f')        
        state_out = np.array(seq_next,dtype='f')
        # return [action_in, state_in], state_out, sample_index
        return ([action_in, state_in], state_out)
