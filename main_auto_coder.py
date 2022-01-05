import torch as t
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import data
import warnings
import model
import numpy as np
from scipy.stats import ortho_group
import DPL_LOSS

warnings.filterwarnings('ignore')


def init_F(p, batch_size):
    x = t.rand(p, batch_size)
    x = x / t.norm(x, p='fro')
    return x


# 三个超参数
l1 = 100
l2 = 100
l3 = 0.001
# 列数
batch_size = 128
# 行数
# p = 160
p = 42*42*10
# 合成字典
D = init_F(p, batch_size)

# 分析字典
P = init_F(batch_size, p)

D.requires_grad = True
P.requires_grad = True
# train_dataset_2S1 = data.DiabetesDataset(root='../../MSTAR2\\train\\2S1', )
# test_dataset_2S1 = data.DiabetesDataset(root='../../MSTAR2\\test\\2S1', )
train_dataset_2S1 = data.DiabetesDataset(root='/data2/ci2p_user_data/kwei/MSTAR2/train/2S1', )
test_dataset_2S1 = data.DiabetesDataset(root='/data2/ci2p_user_data/kwei/MSTAR2/test/2S1', )

train_loader_2S1 = DataLoader(dataset=train_dataset_2S1,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=1,
                              drop_last=True,
                              )
test_loader_2S1 = DataLoader(dataset=test_dataset_2S1,
                             shuffle=False,
                             )

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
P = P.to(device)
D = D.to(device)
# net = model.AutoEncoder(D, P)
net = model.Test(D,P)
net = net.to(device)

optim = optim.Adam(net.parameters(), lr=0.001)


def train(epoch):
    net.train()
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader_2S1, 0):
        input = data
        input = input.to(device)

        optim.zero_grad()
        encode, decode = net(input)
        encode.to(device)
        decode.to(device)
        loss = DPL_LOSS.DL_DPL(input, encode, decode, D, P, l1, l2, l3)

        loss.backward()
        optim.step()

        running_loss += loss.item()

        print('[{},{}] loss:{}'.format(epoch + 1, batch_idx + 1, running_loss / batch_size))
        running_loss = 0.0


# def test():
#     net.eval()
#     correct = 0
#     total = 0
#     with t.no_grad():
#         for data in test_loader_2S1:
#             images = data
#             images = images.to(device)
#             outputs = net(images)
#             # 每行10个 找（最大值，最大值下标） 沿着第一个维度（行是第0个维度，列是第1个维度）
#             # _为占位符
#             _, pred = t.max(outputs.data, dim=1)
#             # 返回的是第0个维度 多少行 也就是batch_size
#             # print('预测：{}，真实：{}'.format(pred,labels))
#             total += labels.size(0)
#             # 真1假0 统计 返回
#             correct += (pred == labels).sum().item()
#     print('正确率：{}%'.format(100 * correct / total))
#     return 100 * correct / total


if __name__ == '__main__':
    for epoch in range(700):
        train(epoch)
        # acc = test()

    t.save(net, 'model1.pth')
    # writer_acc.add_scalar('ACC',acc,epoch)
