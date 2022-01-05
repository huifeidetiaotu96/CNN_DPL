'''
分类训练文件：每一个类构成一个模型、优化器、loss
'''
import torch as t
import torch.optim as optim
import warnings
import model
import torch.multiprocessing
import DPL_LOSS2

torch.multiprocessing.set_sharing_strategy('file_system')

warnings.filterwarnings('ignore')

device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(device)

# 分类种类
species = 28

# 三个超参数，一个类的；后期一个类分别有三个:参数一，参数二，参数三
l1 = 2
l2 = 2
l3 = 2

# 稀疏 : 行数 vgg提取的特征倒数第二层 样本特征量
p = 128

# 稀疏：列数 原子数
m = 50

#

# list_species = ['accordion', 'airplanes', 'anchor', 'ant', 'barrel', 'bass', 'beaver', 'binocular', 'bonsai', 'brain',
#                 'brontosaurus', 'buddha', 'butterfly', 'camera', 'cannon', 'car_side', 'ceiling_fan', 'cellphone',
#                 'chair'
#     , 'chandelier', 'cougar_body', 'cougar_face', 'crab', 'crayfish', 'crocodile', 'crocodile_head', 'cup', 'dalmatian',
#                 'dollar_bill', 'dolphin']

# dict = {
#     'accordion': temp[0], 'airplanes': temp[1], 'anchor': temp[2], 'ant': temp[3], 'barrel': temp[4], 'bass': temp[5],
#     'beaver': temp[6], 'binocular': temp[7], 'bonsai': temp[8], 'brain': temp[9],
#     'brontosaurus': temp[10], 'buddha': temp[11], 'butterfly': temp[12], 'camera': temp[13], 'cannon': temp[14],
#     'car_side': temp[15], 'ceiling_fan': temp[16], 'cellphone': temp[17], 'chair': temp[18]
#     , 'chandelier': temp[19], 'cougar_body': temp[20], 'cougar_face': temp[21], 'crab': temp[22], 'crayfish': temp[23],
#     'crocodile': temp[24], 'crocodile_head': temp[25], 'cup': temp[26], 'dalmatian': temp[27],
#     'dollar_bill': temp[28], 'dolphin': temp[29]
# }
#list_species = ['yaleB29', 'yaleB25', 'yaleB37', 'yaleB11', 'yaleB12', 'yaleB22', 'yaleB39', 'yaleB16', 'yaleB28', 'yaleB35', 'yaleB30', 'yaleB24', 'yaleB33', 'yaleB34', 'yaleB31', 'yaleB27', 'yaleB15', 'yaleB17', 'yaleB19', 'yaleB38', 'yaleB18', 'yaleB13', 'yaleB21', 'yaleB36', 'yaleB26', 'yaleB32', 'yaleB20', 'yaleB23']
list_species = ['yaleB11', 'yaleB12', 'yaleB13', 'yaleB15', 'yaleB16', 'yaleB17', 'yaleB18', 'yaleB19', 'yaleB20', 'yaleB21', 'yaleB22', 'yaleB23', 'yaleB24', 'yaleB25', 'yaleB26', 'yaleB27', 'yaleB28', 'yaleB29', 'yaleB30', 'yaleB31', 'yaleB32', 'yaleB33', 'yaleB34', 'yaleB35', 'yaleB36', 'yaleB37', 'yaleB38', 'yaleB39']

# 网络列表运行
net_list = []
for i in list_species:
    # 传的分别是D，P 合成与分析字典
    net_list.append(model.DPL(p, m))
#print(net_list)


features_list = []
for i in list_species:
    # features_list.append(t.load('/home/kwei/SAR_image_classification/DL_DPL/yaleB_features/{}.pt'.format(i)))
    features_list.append(t.load('D:/Anaconda3-anzhuang/envs/pytorch/aaaaa/Weikang/DL_DPL_gao/yaleB_features/{}.pt'.format(i),map_location=torch.device('cpu')))
#print(features_list[1].shape)


optim_list = []
for net in net_list:
    optim_list.append(optim.Adam(net.parameters(), lr=0.01))


# 训练
def train(epoch):

    # 初始化loss
    running_loss_list = []
    for i in range(len(list_species)):
        running_loss_list.append(0.0)

    # 每类特征分别进入各自网络， 返回完整返回值和返回过fc层的值
    data_temp = []  # 用于保存FC层的输出
    for i in range(len(features_list)):
        net_i1 = net_list[i]
        data_i1 = features_list[i]
        net_i1 = net_i1.to(device)
        data_i1 = data_i1.to(device)

        with torch.no_grad():
            return_data_i1 = net_i1(data_i1)
            #print(return_data_i1)
            data_temp.append(return_data_i1[0])
            print(return_data_i1[0])

# 上面不需要梯度？
    for net in net_list:
        net.train()

    # 网络输出结果
    input_h_list = []
    return_data_list = []
    loss_list = []
    for i1 in range(len(features_list)): # 对于第i1类的数据，进行如下操作

        # 生成互补矩阵
        if i1 == 0:
            temp_z = data_temp[1:]
            # print(temp_z[0].size())
            input_h = t.cat(temp_z, dim=1)
            input_h = input_h.to(device)
            #input_h_list.append(input_h)
        elif i1 == (len(data_temp) - 1):
            temp_z = data_temp[0:i1]
            input_h = t.cat(temp_z, dim=1)
            input_h = input_h.to(device)
            #input_h_list.append(input_h)
        else:
            # 分成两段，拼接成互补矩阵
            temp_z1 = data_temp[0:i1]
            temp_z2 = data_temp[i1 + 1:]
            temp_z = temp_z1 + temp_z2
            input_h = t.cat(temp_z, dim=1)
            input_h = input_h.to(device)
            #input_h_list.append(input_h)

        net_i1 = net_list[i1]
        #print("i1",i1,net_i1)
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
        loss_i1 = DPL_LOSS2.DL_DPL_CNN(data_i1, input_h, return_data_i1[1], return_data_i1[2], D_i1, P_i1, l1, l2, l3)
        #loss_list.append(loss_i1)
        # end 计算loss -----------------------------------------------

        # 反向+更新
        loss_i1.backward() #反向传播计算得到每个参数的梯度值
        optim_i1.step()    #通过梯度下降执行一步参数更新

        # 统计
        # 计算总loss（平均loss×样本数）
        running_loss_i1 = running_loss_list[i1]
        running_loss_i1 += loss_i1  # 这一句其实没有必要，因为在每个epoch下，每一类的loss不会增加。

        # 打印第i1类的loss
        print('[{}] {}:{}:{}'.format(epoch + 1, list_species[i1], running_loss_i1, loss_i1))



# def test():
#     for net in net_list:
#         net.eval()
#
#     correct_z = 0
#     total_z = 0
#     with t.no_grad():
#         for i in range(len(features_list)):
#             test_feature = features_list[i]
#             label = i
#
#             correct = 0
#             for j in range(test_feature.size(1)):
#                 input = test_feature[:, j]
#                 input = input.unsqueeze(1)
#                 input = input.to(device)
#
#                 output_list = []
#
#                 for net in net_list:
#                     x, y = net(input)
#                     output_list.append(t.norm(x - y, p=2).item())
#                 # 求列表最小值的下标
#                 # print('list:', output_list)
#                 pred = output_list.index(min(output_list))
#                 # print('pred:', pred, 'real', label)
#                 correct += (pred == label)
#                 correct_z += (pred == label)
#
#             total = test_feature.size(1)
#             total_z += test_feature.size(1)
#             print('{}类 正确率：{}%'.format(i, 100 * correct / total))
#
#     print('总正确率：{}%'.format(100 * correct_z / total_z))
#     return 100 * correct_z / total_z


if __name__ == '__main__':
    for epoch in range(400):
        train(epoch)
        #res = test()

    for i in range(len(list_species)):
        t.save(net_list[i], 'D:/Anaconda3-anzhuang/envs/pytorch/aaaaa/Weikang/DL_DPL_gao/ExtendedYaleB/model_{}.pkl'.format(list_species[i]))
