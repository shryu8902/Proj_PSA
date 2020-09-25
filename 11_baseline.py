#%%
# base library
import numpy as np
import pandas as pd
import os, time
import time

# data handling
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# deep learning
import tensorflow as tf

# utility
from utils import *
from sequential_utils import *

#%%
# os.environ["CUDA_VISIBLE_DEVICES"]='7'
variable = ['primary_pressure', 'primary_temperature', 'secondary_pressure', 'secondary_temperature', 'PCT']

DATA = DataMerge('./DATA')
DATA_test = DataMerge('./DATA/TestSet',SCALERS=DATA.SCALERS)
SEED = 0
