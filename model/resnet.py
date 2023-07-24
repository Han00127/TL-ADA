import torch.nn as nn
import math
from torchvision import models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from model.dsbn import DomainSpecificBatchNorm2d
from model.dsbn import DomainSpecificBatchNorm1d
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _ntuple
import torch.nn.utils.weight_norm as weightNorm

from collections import OrderedDict
import operator
from itertools import islice

_pair = _ntuple(2)

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

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

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode='zeros')

    def forward(self, input, domain_label):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups), domain_label

class classifier(nn.Module):
    def __init__(self, class_num=31, bottleneck_dim=256):
        super(classifier, self).__init__()
        self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x

class TwoInputSequential(nn.Module):
    r"""A sequential container forward with two inputs.
    """

    def __init__(self, *args):
        super(TwoInputSequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return TwoInputSequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(TwoInputSequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, input1, input2):
        for module in self._modules.values():
            input1, input2 = module(input1, input2)
        return input1, input2

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        updated_state_dict = _update_initial_weights_dsbn(model_zoo.load_url(model_urls['resnet18']),
                                                          num_classes=model.num_classes,
                                                          num_domains=model.num_domains)
        model.load_state_dict(updated_state_dict, strict=False)
    cls_f = classifier(model.num_classes, 256)
    return model,cls_f


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        updated_state_dict = _update_initial_weights_dsbn(model_zoo.load_url(model_urls['resnet34']),
                                                          num_classes=model.num_classes,
                                                          num_domains=model.num_domains)
        model.load_state_dict(updated_state_dict, strict=False)
    cls_f = classifier(model.num_classes, 256)
    return model,cls_f


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    print(model.num_classes)
    if pretrained:

        updated_state_dict = _update_initial_weights_dsbn(model_zoo.load_url(model_urls['resnet50']),
                                                          num_classes=model.num_classes,
                                                          num_domains=model.num_domains)
        model.load_state_dict(updated_state_dict, strict=False)
    cls_f = classifier(model.num_classes, 256)

    return model,cls_f


def p_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Pure_ResBase(**kwargs)
    print(model.num_classes)
    print("Pure resnet-50 model loaded")
    if pretrained:
        pass
        # updated_state_dict = _update_initial_weights_dsbn(model_zoo.load_url(model_urls['resnet50']),
        #                                                   num_classes=model.num_classes,
        #                                                   num_domains=model.num_domains)
        # model.load_state_dict(updated_state_dict, strict=False)
    cls_f = classifier(model.num_classes, 256)

    return model,cls_f


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        updated_state_dict = _update_initial_weights_dsbn(model_zoo.load_url(model_urls['resnet101']),
                                                          num_classes=model.num_classes,
                                                          num_domains=model.num_domains)
        model.load_state_dict(updated_state_dict, strict=False)
    cls_f = classifier(model.num_classes, 256)

    return model,cls_f


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        updated_state_dict = _update_initial_weights_dsbn(model_zoo.load_url(model_urls['resnet152']),
                                                          num_classes=model.num_classes,
                                                          num_domains=model.num_domains)
        model.load_state_dict(updated_state_dict, strict=False)
    cls_f = classifier(model.num_classes, 256)

    return model, cls_f


def _update_initial_weights_dsbn(state_dict, num_classes=1000, num_domains=2, dsbn_type='all'):
    new_state_dict = state_dict.copy()

    for key, val in state_dict.items():
        update_dict = False
        if ((('bn' in key or 'downsample.1' in key) and dsbn_type == 'all') or
                (('bn1' in key) and dsbn_type == 'partial-bn1')):
            update_dict = True

        if (update_dict):
            if 'weight' in key:
                for d in range(num_domains):
                    new_state_dict[key[0:-6] + 'bns.{}.weight'.format(d)] = val.data.clone()

            elif 'bias' in key:
                for d in range(num_domains):
                    new_state_dict[key[0:-4] + 'bns.{}.bias'.format(d)] = val.data.clone()

            if 'running_mean' in key:
                for d in range(num_domains):
                    new_state_dict[key[0:-12] + 'bns.{}.running_mean'.format(d)] = val.data.clone()

            if 'running_var' in key:
                for d in range(num_domains):
                    new_state_dict[key[0:-11] + 'bns.{}.running_var'.format(d)] = val.data.clone()

            if 'num_batches_tracked' in key:
                for d in range(num_domains):
                    new_state_dict[
                        key[0:-len('num_batches_tracked')] + 'bns.{}.num_batches_tracked'.format(d)] = val.data.clone()

    if num_classes != 1000 or len([key for key in new_state_dict.keys() if 'fc' in key]) > 1:
        key_list = list(new_state_dict.keys())
        for key in key_list:
            if 'fc' in key:
                print('pretrained {} are not used as initial params.'.format(key))
                del new_state_dict[key]

    return new_state_dict


class ResNet(nn.Module):
    def __init__(self, block, layers, in_features=256, num_classes=1000, num_domains=2):
        self.inplanes = 64
        self.in_features = in_features
        self.num_domains = num_domains
        self.num_classes = num_classes
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = DomainSpecificBatchNorm2d(64, self.num_domains)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], num_domains=self.num_domains)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, num_domains=self.num_domains)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, num_domains=self.num_domains)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, num_domains=self.num_domains)
        # self.avgpool = nn.AvgPool2d(7, stride=1)

        # Bottleneck
        # This is for bottleneck block
        self.bottleneck = nn.Linear(512 * 4,self.in_features)

        # This is for basic block
        # self.bottleneck = nn.Linear(512, self.in_features)
        self.b_bn1 = DomainSpecificBatchNorm1d(256, self.num_domains)
        # self.b_bn1 = nn.BatchNorm1d(256)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, num_domains=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = TwoInputSequential(
                Conv2d(self.inplanes, planes * block.expansion,
                       kernel_size=1, stride=stride, bias=False),
                DomainSpecificBatchNorm2d(planes * block.expansion, num_domains),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, num_domains=num_domains))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, num_domains=num_domains))

        return TwoInputSequential(*layers)

    def forward(self, x, domain_label, with_ft=False):
        x = self.conv1(x)
        x, _ = self.bn1(x, domain_label)
        x = self.relu(x)
        x = self.maxpool(x)

        out1, _ = self.layer1(x, domain_label) # 128
        # print("out1 size ", out1.size())
        out2, _ = self.layer2(out1, domain_label) # 512
        # print("out2 size ", out2.size())
        out3, _ = self.layer3(out2, domain_label) # 1024
        # print("out3 size ", out3.size())
        out4, _ = self.layer4(out3, domain_label) # 2048
        # print("out4 size ", out4.size())

        x = out4.mean(3).mean(2)  # global average pooling
        x = x.view(x.size(0), -1)

        # BottleNeck
        x = self.bottleneck(x)
        x, _ = self.b_bn1(x,domain_label)
        return x, [out1,out2,out3,out4]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_domains=2):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = DomainSpecificBatchNorm2d(planes, num_domains)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = DomainSpecificBatchNorm2d(planes, num_domains)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, domain_label):
        residual = x

        out = self.conv1(x)
        out, _ = self.bn1(out, domain_label)
        out = self.relu(out)

        out = self.conv2(out)
        out, _ = self.bn2(out, domain_label)

        if self.downsample is not None:
            residual, _ = self.downsample(x, domain_label)

        out += residual
        out = self.relu(out)

        return out, domain_label


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_domains=2):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = DomainSpecificBatchNorm2d(planes, num_domains)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = DomainSpecificBatchNorm2d(planes, num_domains)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = DomainSpecificBatchNorm2d(planes * 4, num_domains)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, domain_label):
        residual = x

        out = self.conv1(x)
        out, _ = self.bn1(out, domain_label)
        out = self.relu(out)

        out = self.conv2(out)
        out, _ = self.bn2(out, domain_label)
        out = self.relu(out)

        out = self.conv3(out)
        out, _ = self.bn3(out, domain_label)

        if self.downsample is not None:
            residual, _ = self.downsample(x, domain_label)

        out += residual
        out = self.relu(out)

        return out, domain_label


res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50, 
"resnet101":models.resnet101, "resnet152":models.resnet152, "resnext50":models.resnext50_32x4d, "resnext101":models.resnext101_32x8d}
# NEW # 
class Pure_ResBase(nn.Module):
    def __init__(self, in_features=256, num_classes=1000, num_domains=2):
        super(Pure_ResBase, self).__init__()
        model_resnet = res_dict['resnet50'](pretrained=True)
        self.num_classes = num_classes
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

        self.bn = nn.BatchNorm1d(256, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(2048, 256)
        self.bottleneck.apply(init_weights)

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
        x = self.bottleneck(x)
        x = self.bn(x)
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


