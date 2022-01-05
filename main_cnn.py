'''
分类训练文件：每一个类构成一个模型、优化器、loss
'''
import torch as t
import torch.optim as optim
import warnings
import model
import torch.multiprocessing
import DPL_LOSS2
import pylab

torch.multiprocessing.set_sharing_strategy('file_system')

warnings.filterwarnings('ignore')

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

# 分类种类
species = 101

# 三个超参数，一个类的；后期一个类分别有三个:参数一，参数二，参数三
l1 = 1
l2 = 1
l3 = 1

# 稀疏 : 行数 提取的特征倒数第二层 样本特征量
p = 2048

# 稀疏：列数 原子数
m = 50

list_species = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35',
                '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52',
                '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69',
                '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86',
                '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100']

# 网络列表运行
net_list = []
for i in list_species:
    # 传的分别是D，P 合成与分析字典
    net_list.append(model.DPL(p, m))
# print(net_list)


features_list = []
for i in list_species:
    features_list.append(t.load('/home/kwei/SAR_image_classification/DL_DPL/Caltech101_features/{}.pt'.format(i)))
# print(features_list[1].shape)

test_list = []
for i in list_species:
    test_list.append(t.load('/home/kwei/SAR_image_classification/DL_DPL/Caltech101_test_features/{}.pt'.format(i)))

optim_list = []
for net in net_list:
    optim_list.append(optim.Adam(net.parameters(), lr=0.0005))


# 训练
def train(epoch):
    # 初始化loss
    running_loss_list = []
    for i in range(len(list_species)):
        running_loss_list.append(0.0)

    # 每类特征分别进入各自网络， 返回过fc层的值
    data_temp = []
    for i in range(len(features_list)):
        net_i1 = net_list[i]
        data_i1 = features_list[i]
        net_i1 = net_i1.to(device)
        data_i1 = data_i1.to(device)

        with torch.no_grad():
            return_data_i1 = net_i1(data_i1)
            # print(return_data_i1)
            data_temp.append(return_data_i1[0])
            # print(return_data_i1[0])

    for net in net_list:
        net.train()

    # 网络输出结果
    input_h_list = []
    return_data_list = []
    loss_list = []
    for i1 in range(len(features_list)):  # 对于第i1类的数据，进行如下操作

        # 生成互补矩阵
        if i1 == 0:
            temp_z = data_temp[1:]
            # print(temp_z[0].size())
            input_h = t.cat(temp_z, dim=1)
            input_h = input_h.to(device)
            # input_h_list.append(input_h)
        elif i1 == (len(data_temp) - 1):
            temp_z = data_temp[0:i1]
            input_h = t.cat(temp_z, dim=1)
            input_h = input_h.to(device)
            # input_h_list.append(input_h)
        else:
            # 分成两段，拼接成互补矩阵
            temp_z1 = data_temp[0:i1]
            temp_z2 = data_temp[i1 + 1:]
            temp_z = temp_z1 + temp_z2
            input_h = t.cat(temp_z, dim=1)
            input_h = input_h.to(device)
            # input_h_list.append(input_h)

        net_i1 = net_list[i1]
        # print("i1",i1,net_i1)
        data_i1 = features_list[i1]
        optim_i1 = optim_list[i1]

        net_i1 = net_i1.to(device)
        data_i1 = data_i1.to(device)

        optim_i1.zero_grad()

        return_data_i1 = net_i1(data_i1)
        # return_data_list.append(return_data_i1)

        # 计算loss ---------------------------------------------------
        D_i1 = net_i1.self_expression_d.Coefficient  # 提取第i个网络中的D
        P_i1 = net_i1.self_expression_p.Coefficient  # 提取第i个网络中的D
        loss_i1 = DPL_LOSS2.DL_DPL_CNN(return_data_i1[0], input_h,
                                       D_i1, P_i1, return_data_i1[1], l1, l2, l3)

        # def DL_DPL_CNN(x_k, x_k_h, D, P, output, l1, l2, l3):

        # loss_i1 = DPL_LOSS2.DL_DPL_CNN(data_i1, input_h, return_data_i1[1], return_data_i1[2], D_i1, P_i1, l1, l2, l3)
        # loss_list.append(loss_i1)
        # end 计算loss -----------------------------------------------

        # 反向+更新
        loss_i1.backward()  # 反向传播计算得到每个参数的梯度值
        optim_i1.step()  # 通过梯度下降执行一步参数更新

        # 统计
        # 计算总loss（平均loss×样本数）
        running_loss_i1 = running_loss_list[i1]
        running_loss_i1 += loss_i1  # 这一句其实没有必要，因为在每个epoch下，每一类的loss不会增加。

        # 打印第i1类的loss
        print('[{}] {}:{}'.format(epoch + 1, list_species[i1], loss_i1))


def test():
    for net in net_list:
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

                for net in net_list:
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


def test_test():
    for net in net_list:
        net.eval()

    correct_z = 0
    total_z = 0
    with t.no_grad():
        for i in range(len(test_list)):
            test_feature = test_list[i]
            label = i

            correct = 0
            for j in range(test_feature.size(1)):
                input = test_feature[:, j]
                input = input.unsqueeze(1)
                input = input.to(device)

                output_list = []

                for net in net_list:
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
    acc = []
    for epoch in range(300):
        train(epoch)
    #     res = test_test()
    #     acc.append(res)
    #
    # x = range(1200)
    # y = acc
    # pylab.plot(x, y)
    # pylab.show()

    for i in range(len(list_species)):
        t.save(net_list[i], '/home/kwei/SAR_image_classification/DL_DPL/Caltech101/model_{}.pkl'
               .format(list_species[i]))
