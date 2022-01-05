'''
创建预训练模型
'''
import torch as t
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import data
import warnings
import model
from torch.utils.tensorboard import SummaryWriter
import pylab


warnings.filterwarnings('ignore')
writer = SummaryWriter('./log')

device = t.device("cuda:2" if t.cuda.is_available() else "cpu")

# 列数
batch_size = 256

train_dataset = datasets.ImageFolder(root='/data2/ci2p_user_data/kwei/Caltech101/train',
                                     transform=data.transform)

test_dataset = datasets.ImageFolder(root='/data2/ci2p_user_data/kwei/Caltech101/test',
                                    transform=data.transform)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=1,
                          )

test_loader = DataLoader(dataset=test_dataset,
                         shuffle=True,
                         batch_size=batch_size,
                         )

net = model.ResNet50(101)
net = net.to(device)

criterion = t.nn.CrossEntropyLoss(size_average=True)

optim = optim.SGD(net.parameters(), lr=0.001,momentum=0.9)


def train(epoch):
    net.train()
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        input, target = data
        input = input.to(device)
        target = target.to(device)

        optim.zero_grad()

        loss = criterion(net(input), target)

        loss.backward()
        optim.step()

        running_loss += loss.item()

        print('[{},{}] loss:{}'.format(epoch + 1, batch_idx + 1, running_loss / batch_size))
        running_loss = 0.0


def test():
    net.eval()
    correct = 0
    total = 0
    with t.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)

            _, pred = t.max(outputs.data, dim=1)

            total += labels.size(0)
            correct += (pred == labels).sum().item()
    print('正确率：{}%'.format(100 * correct / total))
    return 100 * correct / total


if __name__ == '__main__':
    acc = []
    for epoch in range(500):
        train(epoch)
        res = test()
        acc.append(res)

        if epoch % 20 == 9:
            # t.save(net.state_dict(), 'model_vgg16_state_dict_ExtendedYaleB_4096_{}_{}_sigmod.pkl'.format(epoch,batch_size))
            t.save(net.state_dict(),
                   '/home/kwei/SAR_image_classification/DL_DPL/Caltech101_pretrained/SGD_model_ResNet50_state_dict_Caltech101_{}.pkl'
                   .format(epoch))

    x = range(500)
    y = acc
    pylab.plot(x, y)
    pylab.show()

