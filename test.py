'''
各类的预训练模型分类精度测试
'''
import torch as t
import warnings

warnings.filterwarnings('ignore')

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

species = 101

list_species = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35',
                '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52',
                '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69',
                '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86',
                '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100']

features_list = []

for i in list_species:
    features_list.append(t.load('/home/kwei/SAR_image_classification/DL_DPL/Caltech101_test_features/{}.pt'.format(i)))
    # features_list.append(t.load('D:/Anaconda3-anzhuang/envs/pytorch/aaaaa/Weikang/DL_DPL_gao/yaleB_features/{}.pt'.format(i),map_location=torch.device('cpu')))

list_net = []
for i in range(species):
    list_net.append(
        t.load('/home/kwei/SAR_image_classification/DL_DPL/Caltech101/model_{}.pkl'.format(list_species[i])))


def test():
    for net in list_net:
        net.eval()

    correct_z = 0
    total_z = 0
    with t.no_grad():
        for i in range(len(features_list)):
            test_feature = features_list[i]
            label = i

            correct = 0
            for j in range(test_feature.size(1)):
                input = test_feature[:, j]
                input = input.unsqueeze(1)
                input = input.to(device)
                output_list = []

                for net in list_net:
                    x, y = net(input)
                    output_list.append(t.norm(x - y, p=2).item())
                # 求列表最小值的下标
                # print('list:', output_list)
                pred = output_list.index(min(output_list))
                # print('pred:', pred, 'real', label)
                correct += (pred == label)
                correct_z += (pred == label)

            total = test_feature.size(1)
            total_z += test_feature.size(1)
            print('{}类 正确率：{}%'.format(i, 100 * correct / total))

    print('总正确率：{}%'.format(100 * correct_z / total_z))
    return 100 * correct_z / total_z


if __name__ == '__main__':
    test()
