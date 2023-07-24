import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict
import torch.nn.functional as F


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50, 
"resnet101":models.resnet101, "resnet152":models.resnet152, "resnext50":models.resnext50_32x4d, "resnext101":models.resnext101_32x8d}

class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        x = self.avgpool(out4)
        x = x.view(x.size(0), -1)
        return x, [out1,out2,out3,out4]

class feat_bootleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bootleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x

class feat_classifier_two(nn.Module):
    def __init__(self, class_num, input_dim, bottleneck_dim=256):
        super(feat_classifier_two, self).__init__()
        self.type = type
        self.fc0 = nn.Linear(input_dim, bottleneck_dim)
        self.fc0.apply(init_weights)
        self.fc1 = nn.Linear(bottleneck_dim, class_num)
        self.fc1.apply(init_weights)

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        return x

class Res50(nn.Module):
    def __init__(self):
        super(Res50, self).__init__()
        model_resnet = models.resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features
        self.fc = model_resnet.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        x = self.avgpool(out4)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return x, y, [out1,out2,out3,out4]

# LossNet modified 080621
class LossNet(nn.Module):
    # def __init__(self, feature_sizes=[32, 16, 8, 4], num_channels=[64, 128, 256, 512], interm_dim=128):
    # Original
    def __init__(self, feature_sizes=[56, 28, 14, 7], num_channels=[256, 512, 1024, 2048], interm_dim=512):
    # def __init__(self, feature_sizes = [56, 28, 14, 7], num_channels = [256, 512, 1024, 2048], interm_dim = 1024):

        super(LossNet, self).__init__()

        self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        self.GAP4 = nn.AvgPool2d(feature_sizes[3])

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)
        self.FC4 = nn.Linear(num_channels[3], interm_dim)
        self.linear = nn.Linear(4 * interm_dim, 1)

        self.bn1 = nn.BatchNorm1d(512, affine=True)
        self.bn2 = nn.BatchNorm1d(512, affine=True)
        self.bn3 = nn.BatchNorm1d(512, affine=True)
        self.bn4 = nn.BatchNorm1d(512, affine=True)
        self.bn5 = nn.BatchNorm1d(2048, affine=True)

        # nn.init.kaiming_uniform_(self.FC1.weight)
        # nn.init.kaiming_uniform_(self.FC2.weight)
        # nn.init.kaiming_uniform_(self.FC3.weight)
        # nn.init.kaiming_uniform_(self.FC4.weight)
        # nn.init.xavier_uniform_(self.linear.weight)
        # nn.init.constant_(self.linear.bias,0.01)


    def forward(self, features):
        out1 = self.GAP1(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.GAP3(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out4 = self.GAP4(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))

        out = self.linear(torch.cat((out1, out2, out3, out4), 1))


        # out1 = self.GAP1(features[0])
        # out1 = out1.view(out1.size(0), -1)
        # out1 = F.relu(self.bn1(self.FC1(out1)))

        # out2 = self.GAP2(features[1])
        # out2 = out2.view(out2.size(0), -1)
        # out2 = F.relu(self.bn2(self.FC2(out2)))

        # out3 = self.GAP3(features[2])
        # out3 = out3.view(out3.size(0), -1)
        # out3 = F.relu(self.bn3(self.FC3(out3)))

        # out4 = self.GAP4(features[3])
        # out4 = out4.view(out4.size(0), -1)
        # out4 = F.relu(self.bn4(self.FC4(out4)))

        # out = self.linear(torch.cat((out1, out2, out3, out4), 1))
        return out

class resnet34_LossNet(nn.Module):
    # def __init__(self, feature_sizes=[32, 16, 8, 4], num_channels=[64, 128, 256, 512], interm_dim=128):
    def __init__(self, feature_sizes = [56, 28, 14, 7], num_channels=[64, 128, 256, 512], interm_dim = 128):

        super(resnet34_LossNet, self).__init__()

        self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        self.GAP4 = nn.AvgPool2d(feature_sizes[3])

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)
        self.FC4 = nn.Linear(num_channels[3], interm_dim)
        self.linear = nn.Linear(4 * interm_dim, 1)

        nn.init.kaiming_uniform_(self.FC1.weight)
        nn.init.kaiming_uniform_(self.FC1.weight)
        nn.init.kaiming_uniform_(self.FC2.weight)
        nn.init.kaiming_uniform_(self.FC3.weight)
        nn.init.kaiming_uniform_(self.FC4.weight)
        nn.init.xavier_uniform_(self.linear.weight)
        # nn.init.constant_(self.linear.bias,0.01)

    def forward(self, features):
        out1 = self.GAP1(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.GAP3(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out4 = self.GAP4(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))

        out = self.linear(torch.cat((out1, out2, out3, out4), 1))
        return out

class ProbLossNet(nn.Module):
    def __init__(self, feature_sizes=[56, 28, 14, 7], num_channels=[256, 512, 1024, 2048], interm_dim=512):

        super(ProbLossNet, self).__init__()

        self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        self.GAP4 = nn.AvgPool2d(feature_sizes[3])

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)
        self.FC4 = nn.Linear(num_channels[3], interm_dim)
        self.linear = nn.Linear(4 * interm_dim, 2)
        # self.linear = weightNorm(nn.Linear(4 * interm_dim, 2))
        self.bn1 = nn.BatchNorm1d(512, affine=True)
        self.bn2 = nn.BatchNorm1d(512, affine=True)
        self.bn3 = nn.BatchNorm1d(512, affine=True)
        self.bn4 = nn.BatchNorm1d(512, affine=True)
        self.bn5 = nn.BatchNorm1d(2048, affine=True)
        self.sigmoid = nn.Sigmoid()
        # nn.init.kaiming_uniform_(self.FC1.weight)
        # nn.init.kaiming_uniform_(self.FC3.weight)
        # nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, features):
        out1 = self.GAP1(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.bn1(self.FC1(out1)))

        out2 = self.GAP2(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.bn2(self.FC2(out2)))

        out3 = self.GAP3(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.bn3(self.FC3(out3)))

        out4 = self.GAP4(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.bn4(self.FC4(out4)))

        
        out = torch.cat((out1, out2, out3, out4),1)
        out = self.linear(out)
        # new = self.bn5(out[:,0])
        # new1 = self.bn5(out[:,1])
        # print(new, new1, out[:,0],out[:,1])
        mean, var = out[:,0], out[:,1]
        var = F.softplus(var) + 1e-6
        # var = self.sigmoid(var)
        return mean, var


# LossNet modified 090321
class digit_LossNet(nn.Module):
    def __init__(self, feature_sizes = [14,5], num_channels = [20,50], interm_dim = 50):

        super(digit_LossNet, self).__init__()

        self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        self.GAP2 = nn.AvgPool2d(feature_sizes[1])


        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)

        self.linear = nn.Linear(2 * interm_dim, 1)

        nn.init.kaiming_uniform_(self.FC1.weight)
        nn.init.kaiming_uniform_(self.FC2.weight)
        nn.init.xavier_uniform_(self.linear.weight)
        # nn.init.constant_(self.linear.bias,0.01)

    def forward(self, features):
        out1 = self.GAP1(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out = self.linear(torch.cat((out1, out2), 1))
        return out

class Discriminator(nn.Module):
    def __init__(self, bottleneck_dim=256):
        super(Discriminator, self).__init__()
        # self.fc1 = nn.Linear(bottleneck_dim,bottleneck_dim)
        self.fc = nn.Linear(bottleneck_dim, 1)
        # modified by kthan 011122
        # self.fc = nn.Linear(bottleneck_dim, 128)
        # self.fc1 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()


        nn.init.xavier_uniform_(self.fc.weight)
        # nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        x = self.fc(x)
        # x = self.fc1(x)
        x = self.sigmoid(x)
        return x
