import torch
from torch import nn as nn
from torch.nn import functional as F

from model.dsbn import DomainSpecificBatchNorm2d

def init_weights(obj):
    for m in obj.modules():
        if isinstance(m, nn.Conv2d):
            # init.xavier_normal_(m.weight)
            m.weight.data.normal_(0, 0.01).clamp_(min=-0.02, max=0.02)
            try:
                m.bias.data.zero_()
            except AttributeError:
                # no bias
                pass
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01).clamp_(min=-0.02, max=0.02)
            try:
                m.bias.data.zero_()
            except AttributeError:
                # no bias
                pass
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.reset_parameters()
        elif isinstance(m, nn.Embedding):
            init.normal_(m.weight, 0, 0.01)

def lenet(pretrained=False, **kwargs):
    model = LeNet(num_classes=10, weights_init_path=None, in_features=500, num_domains=2)
    cls_f = classifier(class_num=10, bottleneck_dim=500)
    return model,cls_f


class classifier(nn.Module):
    def __init__(self,class_num=10, bottleneck_dim=500):
        super(classifier,self).__init__()
        self.fc2 = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.fc3 = nn.Linear(bottleneck_dim,class_num)
        init_weights(self)

    def forward(self,x):
        x = self.fc2(F.relu(x))
        x = self.fc3(x)
        return x


class LeNet(nn.Module):
    """"Network used for MNIST or USPS experiments. Conditional Batch Normalization is added."""

    def __init__(self, num_classes=10, weights_init_path=None, in_features=0, num_domains=2):
        super(LeNet, self).__init__()
        self.num_classes = num_classes
        self.in_features = in_features
        self.num_channels = 3
        self.image_size = 32
        self.num_domains = num_domains
        self.name = 'LeNet'
        self.setup_net()

        if weights_init_path is not None:
            init_weights(self)
            self.load(weights_init_path)
        else:
            init_weights(self)

    def setup_net(self):
        self.conv1 = nn.Conv2d(self.num_channels, 20, kernel_size=5)
        self.bn1 = DomainSpecificBatchNorm2d(20, self.num_domains)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.bn2 = DomainSpecificBatchNorm2d(50, self.num_domains)
        self.pool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(1250, 500)

    def forward(self, x, y, with_ft=False):
        x = self.conv1(x)
        x, _ = self.bn1(x, y)
        x = self.pool1(F.relu(x))
        output1 = x
        x = self.conv2(x)
        x, _ = self.bn2(x, y)
        x = self.pool2(F.relu(x))
        output2 = x
        x = x.view(x.size(0), -1) # 64 * 1250
        x = self.fc1(x)
        return x, [output1,output2]

    def load(self, init_path):
        net_init_dict = torch.load(init_path)
        init_weights(self)
        updated_state_dict = self.state_dict()
        print('load {} params.'.format(init_path))
        for k, v in updated_state_dict.items():
            if k in net_init_dict:
                if v.shape == net_init_dict[k].shape:
                    updated_state_dict[k] = net_init_dict[k]
                else:
                    print(
                        "{0} params' shape not the same as pretrained params. Initialize with default settings.".format(
                            k))
            else:
                print("{0} params does not exist. Initialize with default settings.".format(k))
        self.load_state_dict(updated_state_dict)