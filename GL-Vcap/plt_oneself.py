import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from lib.Mydarw import *
from model.ASTGCN_r import *
import torch
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import copy
from pylab import rcParams
import math

'''
???????????
'''

data = np.load('result/speed_output_epoch_34_test.npz') # gat-lstm-decomp window = 4,pre-len =1
# data = np.load('experiments/speed_output_epoch_335_test.npz')  # gat-lstm-Vcap window=16,pre_len=1
print(data.files)  # ['input', 'prediction', 'data_target_tensor']cc
input_tensor = data['input']
input_tensor = input_tensor[:,:,0,0]
s1 = input_tensor.shape[0]
s2 = input_tensor.shape[1]
input_tensor = input_tensor.reshape(s1,s2)
data_target_tensor = data['data_target_tensor']
prediction_data=data['prediction']
target_np = data_target_tensor.reshape(s1,s2)
prediction_np = prediction_data.reshape(s1,s2)

# 真实数据 （seq_l,n,f)
origi_data = np.load('/Users/lvzhuanghu/code/data/daily/8dayUnit/case_05_09_49_230_8day_sum.npy')
or_case = origi_data[:, :, 0]  # (seq_l,n,case
or_LST = origi_data[:, :, 1]
or_SR = origi_data[:, :, 2]
x_batch = origi_data[:, :, 1:3]


# plt_2d()

# plt_res(prediction_np,target_np,"GAT-LSTM-decomp",1)
plt_res(prediction_np,target_np,"GAT-LSTM-Vcap",1)

# plt_res_mul(prediction_np,target_np,1)
# ????

# outputs_batch = []
# T = x_batch[:,:,0]
# R = x_batch[:,:,1]
# m = 10*R
# gpc  = 36.5/(T+2.0-9.9) + 0.5
# a = 0.7/gpc
# p = 0.5 ** (1/gpc)
# n = 111.0/((2.0*(26.5/T+2.0-9.9)/gpc)+T-18)
# up = -(m*a*a)*(p**n)
# down = np.log(p)
# vcap = up/down







