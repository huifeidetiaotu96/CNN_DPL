import torch as t
import torch.nn as n
import torchvision.models as models


class SelfExpression(n.Module):
    def __init__(self, MyFirstPara, MySecondPara):
        super(SelfExpression, self).__init__()
        self.Coefficient = n.Parameter(1.0e-4 * t.ones(MyFirstPara, MySecondPara, dtype=t.float32), requires_grad=True)

    def forward(self, x):  # shape=[d, n]
        y = t.matmul(self.Coefficient, x)
        return y


class DPL(n.Module):
    def __init__(self, nDim=None, nAtom=None):  # nDim: 输入特征的维度； nAtom：字典的原子维度
        super(DPL, self).__init__()
        self.fc = n.Linear(2048, 2048)
        # self.Sig = n.LeakyReLU()
        # self.Sig = n.Sigmoid()
        self.self_expression_d = SelfExpression(nDim, nAtom)
        self.self_expression_p = SelfExpression(nAtom, nDim)

    def forward(self, x):
        # print(self.self_expression_p.Coefficient)
        # print(self.self_expression_p.Coefficient)
        x = t.transpose(x, 0, 1)
        # print(x.size())
        x = self.fc(x)
        # print(x)
        # x = self.Sig(x)
        # print("11111",x)
        x = t.transpose(x, 0, 1)
        # print(x.size())
        x1 = self.self_expression_p(x)
        # print(x1.shape)
        x2 = self.self_expression_d(x1)
        # print(x2.size())
        # print(x2.shape)
        # return x1, x2
        return x, x2


# a = DPL(128,50)
# print("a", a)
class ResNext50(n.Module):
    def __init__(self, num_class=101):
        super(ResNext50, self).__init__()
        self.resnet50 = models.resnext50_32x4d(pretrained=True)

        for param in self.resnet50.parameters():
            param.requires_grad = False

        self.num_class = num_class
        self.conv1 = self.resnet50.conv1
        self.bn1 = self.resnet50.bn1
        self.relu = self.resnet50.relu
        self.layer1 = self.resnet50.layer1
        self.layer2 = self.resnet50.layer2
        self.layer3 = self.resnet50.layer3
        self.layer4 = self.resnet50.layer4
        self.avgpool = self.resnet50.avgpool

        self.fc = self.resnet50.fc
        self.fc = n.Linear(in_features=2048, out_features=num_class, bias=True)

        for p in self.fc.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.size())
        x = self.avgpool(x)
        # print(x.size())
        x = t.flatten(x, 1)
        x = self.fc(x)
        return x

class ResNext50_DPL(n.Module):
    def __init__(self):
        super(ResNext50_DPL, self).__init__()
        resnet50_model = ResNext50()

        resnet50_model.load_state_dict(
            # t.load('model_vgg16_state_dict_ExtendedYaleB_4096_9_64_sigmod.pkl'))
            t.load('/home/kwei/SAR_image_classification/DL_DPL/model_ResNet50_state_dict_ExtendedYaleB29.pkl'))
        self.conv1 = resnet50_model.conv1
        self.bn1 = resnet50_model.bn1
        self.relu = resnet50_model.relu
        self.layer1 = resnet50_model.layer1
        self.layer2 = resnet50_model.layer2
        self.layer3 = resnet50_model.layer3
        self.layer4 = resnet50_model.layer4
        # self.avgpool = resnet50_model.avgpool
        # self.fc = resnet50_model.fc

        for p in resnet50_model.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.avgpool(x)
        x = t.flatten(x, 1)
        # x = self.fc(x)
        x = t.transpose(x, 0, 1)

        return x

class ResNet50(n.Module):
    def __init__(self, num_class=101):
        super(ResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)

        for param in self.resnet50.parameters():
            param.requires_grad = False

        self.num_class = num_class
        self.conv1 = self.resnet50.conv1
        self.bn1 = self.resnet50.bn1
        self.relu = self.resnet50.relu
        self.layer1 = self.resnet50.layer1
        self.layer2 = self.resnet50.layer2
        self.layer3 = self.resnet50.layer3
        self.layer4 = self.resnet50.layer4
        self.avgpool = self.resnet50.avgpool

        self.fc = self.resnet50.fc
        self.fc = n.Linear(in_features=2048, out_features=num_class, bias=True)

        for p in self.fc.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.size())
        x = self.avgpool(x)
        # print(x.size())
        x = t.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet50_DPL(n.Module):
    def __init__(self):
        super(ResNet50_DPL, self).__init__()
        resnet50_model = ResNet50()

        resnet50_model.load_state_dict(
            # t.load('model_vgg16_state_dict_ExtendedYaleB_4096_9_64_sigmod.pkl'))
            t.load('/home/kwei/SAR_image_classification/DL_DPL/Caltech101_pretrained/model_ResNet50_state_dict_Caltech101_309.pkl'))
        self.conv1 = resnet50_model.conv1
        self.bn1 = resnet50_model.bn1
        self.relu = resnet50_model.relu
        self.layer1 = resnet50_model.layer1
        self.layer2 = resnet50_model.layer2
        self.layer3 = resnet50_model.layer3
        self.layer4 = resnet50_model.layer4
        self.avgpool = resnet50_model.avgpool
        # self.fc = resnet50_model.fc

        for p in resnet50_model.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = t.flatten(x, 1)
        print('11',x.size())
        # x = self.fc(x)
        x = t.transpose(x, 0, 1)
        print('1',x.size())
        return x


class VGG16(n.Module):
    def __init__(self, num_class=28):
        super(VGG16, self).__init__()
        self.vgg16_net = models.vgg16_bn(pretrained=True)

        for param in self.vgg16_net.parameters():
            param.requires_grad = False

        self.num_class = num_class
        self.features = self.vgg16_net.features
        self.ave_pooling = self.vgg16_net.avgpool
        self.classifier = self.vgg16_net.classifier

        self.classifier[3] = n.Linear(in_features=4096, out_features=4096, bias=True)
        self.classifier[4] = n.Sigmoid()
        self.classifier[6] = n.Linear(in_features=4096, out_features=num_class, bias=True)

        for p in self.classifier[3:7].parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        x = self.ave_pooling(x)
        x = t.flatten(x, 1)
        x = self.classifier(x)
        return x


# aa = VGG16()
# print(aa)


class VGG16_DPL(n.Module):
    def __init__(self):
        super(VGG16_DPL, self).__init__()
        vgg_model = VGG16()

        # VGG16预训练模型
        vgg_model.load_state_dict(
            t.load('model_vgg16_state_dict_ExtendedYaleB_4096_9_64_sigmod.pkl'))  # map_location='cuda:0'))
        # vgg_model.load_state_dict(t.load('D:/Anaconda3-anzhuang/envs/pytorch/aaaaa/Weikang/DL_DPL_gao/model_vgg16_state_dict_ExtendedYaleB_19_128_sigmod.pkl',map_location='cuda:0'))

        self.features = vgg_model.features
        self.pooling = vgg_model.ave_pooling
        self.classifier = vgg_model.classifier[0:3]

        for p in self.features.parameters():
            p.requires_grad = False

        for p in self.pooling.parameters():
            p.requires_grad = False

        for p in self.classifier.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        # print("feature", x)
        x = self.pooling(x)
        # print("feature-pool", x)
        x = t.flatten(x, 1)

        x = self.classifier(x)

        x = t.transpose(x, 0, 1)
        # print('x',x)

        return x


class VGG_Test(n.Module):
    def __init__(self):
        super(VGG_Test, self).__init__()
        vgg_model = VGG16(28)
        vgg_model.load_state_dict(t.load('model_vgg16_state_dict_ExtendedYaleB_9_64_sigmod.pkl'))
        self.features = vgg_model.features
        self.pooling = vgg_model.ave_pooling
        self.classifier = vgg_model.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x)
        x = t.flatten(x, 1)

        x = self.classifier(x)

        return x
