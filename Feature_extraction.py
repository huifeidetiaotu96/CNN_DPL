import torch as t
from torch.utils.data import DataLoader
import data
import warnings
import model
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

warnings.filterwarnings('ignore')

device = t.device("cuda:2" if t.cuda.is_available() else "cpu")

# 分类种类
species = 101

# batch_size
batch_size = 50


list_species = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35',
                '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52',
                '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69',
                '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86',
                '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100']

list_train_labels = []

# 数据集列表
train_dataset = []
test_dataset = []
for i in list_species:
    train_dataset.append(data.DiabetesDataset(root='/data2/ci2p_user_data/kwei/Caltech101/train/{}'.format(i)))
    test_dataset.append(data.DiabetesDataset(root='/data2/ci2p_user_data/kwei/Caltech101/test/{}'.format(i)))

# loader列表
train_loader = []
test_loader = []
for i in train_dataset:
    train_loader.append(DataLoader(dataset=i, batch_size=batch_size, shuffle=False, num_workers=1))

for i in test_dataset:
    test_loader.append(DataLoader(dataset=i, batch_size=batch_size, shuffle=False, num_workers=1))

# 网络列表运行
net = model.ResNet50_DPL()
net = net.to(device)


def feature_train():
    # 从预加载数据中拿数据
    with t.no_grad():
        for loader in train_loader:
            return_data = []
            for batch_idx, data in enumerate(loader, 0):
                images = data
                images = images.to(device)

                return_data.append(net(images))
            result = t.cat(return_data, dim=1)
            torch.save(result, "/home/kwei/SAR_image_classification/DL_DPL/Caltech101_features/{}.pt"
                       .format(list_species[train_loader.index(loader)]))


def feature_test():
    # 从预加载数据中拿数据
    with t.no_grad():
        for loader in test_loader:
            return_data = []
            for batch_idx, data in enumerate(loader, 0):
                images = data
                images = images.to(device)

                return_data.append(net(images))
            result = t.cat(return_data, dim=1)
            torch.save(result, "/home/kwei/SAR_image_classification/DL_DPL/Caltech101_test_features/{}.pt"
                       .format(list_species[test_loader.index(loader)]))


feature_train()
feature_test()
