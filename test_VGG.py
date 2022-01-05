'''
测试预训练VGG文件，与CNN_DPL为同一预训练模型
'''
import torch as t
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import data
from torchvision import transforms, datasets
import model
import warnings

warnings.filterwarnings('ignore')
batch_size = 128

test_dataset = datasets.ImageFolder(root='/data2/ci2p_user_data/kwei/Caltech101/test',
                                    transform=data.transform)

test_loader = DataLoader(dataset=test_dataset,
                         shuffle=True,
                         batch_size=batch_size,
                         )
device = t.device("cuda:1" if t.cuda.is_available() else "cpu")

net = model.VGG_Test()
net = net.to(device)


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


if __name__ == '__main__':
    test()
