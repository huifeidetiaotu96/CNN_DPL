import torch as t
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import data
import warnings
import model
import numpy as np
from scipy.stats import ortho_group
import os

#r
# y = t.load("././Caltech101_features/accordion.pt")
# print(y.size())
# path = r'C:\Users\hfdxt\Desktop\Caltech101\\test'
# sub = os.listdir(r'C:\Users\hfdxt\Desktop\Caltech101\\test')
# sub = os.listdir(r'C:\Users\hfdxt\Desktop\ExtendedYaleB\train').
# for id,su in enumerate(sub):
#     os.rename(path+'\\'+su, path+'\\'+str(id))
# print(sub)
# list = []
# for i in range(101):
#     list.append(str(i))
#
# print(list)
