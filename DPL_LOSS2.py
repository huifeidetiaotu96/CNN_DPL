import torch as t
import torch.nn.functional as F
import os

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# CNN预训练+DP的LOSS
def DL_DPL_CNN(x_k, x_k_h, D, P, output, l1, l2, l3):
    # part1 = l2 / 2 * t.pow(t.norm(x_k-output, p='fro'), 2)
    part1 = l2 / 2 * t.nn.functional.mse_loss(x_k, output)
    # print('part1:',part1)
    # print('part1',part1)

    input2 = t.mm(P, x_k_h)
    tool = t.zeros(input2.size(0), input2.size(1))
    tool = tool.to(device)

    part2 = l1 / 2 * t.nn.functional.mse_loss(input2, tool)
    # print('part2:',part2)
    # input2 = t.mm(P, x_k_h)
    # part2 = l1 / 2 * t.pow(t.norm(input2, p='fro'), 2)

    # di要 收敛到1
    input3 = 0
    lie = D.size(1)
    for i in range(lie):
        input3 += t.pow((t.pow(t.norm(D[:, i], p=2), 2) - 1), 2)

    part3 = l3 / 2 * input3
    # print('part3', part3)

    return part1 + part2 + part3
