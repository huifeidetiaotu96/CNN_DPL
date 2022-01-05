'''
各类的预训练模型分类精度测试
'''
import torch as t
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import data
import warnings
import scipy.io as io
import model

warnings.filterwarnings('ignore')

device = t.device("cuda:3" if t.cuda.is_available() else "cpu")

batch_size = 128
species = 101
list_x = []
list_label = []

test_dataset = datasets.ImageFolder(root='/data2/ci2p_user_data/kwei/Caltech101//test',
                                    transform=data.transform)

test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size, )


net = model.ResNet50_DPL()
net = net.to(device)



def test():

    with t.no_grad():
        for dataes, labels in test_loader:
            for i in range(len(dataes)):
                input, label = dataes[i], labels[i]
                input = input.unsqueeze(0)
                input = input.to(device)
                label = label.to(device)

                x = net(input)
                x = x.cpu().numpy()
                print(x.shape)
                list_x.append(x)
                list_label.append(label.cpu())


test()
dict = {'label':list_label,'data':list_x,}

# io.savemat('test.mat', dict)

