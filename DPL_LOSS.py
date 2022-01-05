import torch as t
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import model
import numpy as np
import torch.nn.functional as F
import main_cnn


# 自编码器的loss
def DL_DPL(x_k, x_k_encode, out_x_k, D, P, l1, l2, l3):
    x_k_h = x_k_encode[:, 1:]

    input0 = x_k - out_x_k
    # 均方误差
    input_auto_encoder = 1 / 2 * t.pow(t.norm(input0, p='fro'), 2)

    # print(D, P)
    # temp = t.mm(D, P).double()
    temp = t.mm(t.mm(D, P), x_k_encode)
    # print(x_k_encode)
    # temp = t.mm(temp, x_k_encode)
    # print(temp)
    input1 = x_k_encode - temp
    # print(input1)
    part1 = l2 / 2 * t.pow(t.norm(input1, p='fro'), 2)
    # print(part1)

    input2 = t.mm(P, x_k_h)
    part2 = l1 / 2 * t.pow(t.norm(input2, p='fro'), 2)
    # print(part2)

    input3 = 0
    lie = D.size(1)
    for i in range(lie):
        input3 += t.pow(t.norm(D[:, i] - 1, p=2), 2)

    part3 = l3 / 2 * input3
    # print(part3)
    # print("loss11111",input_auto_encoder,part1,part2,part3)
    return input_auto_encoder + part1 + part2 + part3


# CNN预训练+DP的LOSS
def DL_DPL_CNN(x_k, x_k_h, D, P, output, l1, l2, l3):
    # part1 = l2 / 2 * t.pow(t.norm(x_k-output, p='fro'), 2)
    # print(D,P)
    part1 = l2 / 2 * t.nn.functional.mse_loss(x_k, output)
    # print('part1',part1)

    # 生成对应size的全0矩阵，来做MSE的第二项
    input2 = t.mm(P, x_k_h)
    tool = t.zeros(input2.size(0), input2.size(1))
    tool = tool.to(main_cnn.device)
    part2 = l1 / 2 * t.nn.functional.mse_loss(input2, tool)

    # part2 = l1 / 2 * t.pow(t.norm(input2, p='fro'), 2)
    # print('part2',part2)

    # input3 = 0
    # lie = D.size(1)
    # for i in range(lie):
    #     # print(D[:, i])
    #     input3 += t.pow(t.norm(D[:, i] - 1, p=2), 2)

    # 计算一下每一个是否小于等于1 di-1等于0 lamda不等0 反之

    # part3 = l3 / 2 * input3
    # print('part3',part3)

    tool2 = t.ones(D.size(0), D.size(1))
    tool2 = tool2.to(main_cnn.device)
    part3 = l3 / 2 * t.nn.functional.mse_loss(D, tool2)

    return part1 + part2 + part3
