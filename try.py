import os
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import warnings
import model
import torch.nn as n
import torch as t
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import data


#
model = t.load('/home/kwei/SAR_image_classification/DL_DPL/Caltech101/model_2.pkl')
print(model.self_expression_d.Coefficient,model.self_expression_p.Coefficient)
print(t.pow(t.norm(model.self_expression_d.Coefficient[:,10], p=2), 2))

# data = t.load('/home/kwei/SAR_image_classification/DL_DPL/Caltech101_test_features/0.pt')
# print(data.size())

# features_list.append(t.load('/home/kwei/SAR_image_classification/DL_DPL/Caltech101_test_features/{}.pt'.format(i)))

# a = t.rand(1,3,224,224)
# model = model.VGG16_DPL()
# # print(model)
# res = model(a)
# print(res.size())


