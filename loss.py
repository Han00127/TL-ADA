import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb
import random

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss


##
# Loss Prediction Loss modified 080621
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert input.shape == input.flip(0).shape
    if len(input) % 2 != 0:
        return torch.tensor(0.05,requires_grad=True).cuda()
    else:
        input = (input - input.flip(0))[
                :len(input) // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
        target = (target - target.flip(0))[:len(target) // 2]
        target = target.detach()
        one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors
        if reduction == 'mean':
            loss = torch.sum(torch.clamp(margin - one * input, min=0))
            loss = loss / input.size(0)  # Note that the size of input is already halved
        elif reduction == 'none':
            loss = torch.clamp(margin - one * input, min=0)
        else:
            NotImplementedError()

        return loss


def LossPredLoss2(input, target, margin=1.0, reduction='mean'):
    assert input.shape == input.flip(0).shape
    if len(input) % 2 != 0:
        index = input[:-1]
        target = input[:-1]
        input = (input - input.flip(0))[
                :len(input) // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
        target = (target - target.flip(0))[:len(target) // 2]
        target = target.detach()
        one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors
        if reduction == 'mean':
            loss = torch.sum(torch.clamp(margin - one * input, min=0))
            loss = loss / input.size(0)  # Note that the size of input is already halved
        elif reduction == 'none':
            loss = torch.clamp(margin - one * input, min=0)
        else:
            NotImplementedError()

        return loss
    else:
        input = (input - input.flip(0))[
                :len(input) // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
        target = (target - target.flip(0))[:len(target) // 2]
        target = target.detach()
        one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors
        if reduction == 'mean':
            loss = torch.sum(torch.clamp(margin - one * input, min=0))
            loss = loss / input.size(0)  # Note that the size of input is already halved
        elif reduction == 'none':
            loss = torch.clamp(margin - one * input, min=0)
        else:
            NotImplementedError()

        return loss

def LossPredLoss3(input, target, margin=1.0, reduction='mean'):
    return torch.tensor(0.05,requires_grad=True).cuda()

# TA-VAAL Ranking Loss - revised by kthan 092921
def TA_VAAL_LossPredLoss(input, target, margin=1.0, reduction='mean'):
    if len(input) % 2 != 0:
        return torch.tensor(0.05, requires_grad=True).cuda()
    assert input.shape == input.flip(0).shape
    criterion = nn.BCELoss()
    input = (input - input.flip(0))[
            :len(input) // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()
    diff = torch.sigmoid(input)
    one = torch.sign(torch.clamp(target, min=0))  # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = criterion(diff, one)
    elif reduction == 'none':
        loss = criterion(diff, one)
    else:
        NotImplementedError()

    return loss




# get uncertainty modified 080621
def get_uncertainty(netF,netC, lossnet,unlabeled_loader):
    netF.eval()
    netC.eval()
    lossnet.eval()
    loss = torch.tensor([]).cuda()
    pred = torch.tensor([]).cuda()
    labels = torch.tensor([]).cuda()
    with torch.no_grad():
        for (inputs, label, idx) in unlabeled_loader:
            inputs = inputs.cuda()
            t_domain = torch.ones(inputs.shape[0], dtype=torch.long).cuda()
            label = label.cuda()
            features_test, mid_blocks = netF(inputs,t_domain)
            outputs_test = netC(features_test)
            logit = nn.Softmax(dim=1)(outputs_test)
            _, predict = torch.max(logit,1)
            pred_loss = lossnet(mid_blocks)
            pred_loss = pred_loss.view(pred_loss.size(0))

            loss = torch.cat((loss, pred_loss), 0)
            pred = torch.cat((pred,predict.float()),0)
            labels = torch.cat((labels,label.float()),0)

    return loss.cpu(),pred.cpu(),labels.cpu()

# get uncertainty modified 080621
def get_uncertainty2(netF,netC, lossnet,unlabeled_loader):
    netF.eval()
    netC.eval()
    lossnet.eval()
    loss = torch.tensor([]).cuda()
    prob = torch.tensor([]).cuda()
    with torch.no_grad():
        for (inputs, labels, idx) in unlabeled_loader:
            inputs = inputs.cuda()
            # labels = labels.cuda()
            features_test, mid_blocks = netF(inputs)
            outputs_test = netC(features_test)
            probability = nn.Softmax()
            pred_loss = lossnet(mid_blocks)
            pred_loss = pred_loss.view(pred_loss.size(0))

            loss = torch.cat((loss, pred_loss), 0)

    return loss.cpu()












