from sklearn.manifold import TSNE
import numpy as np
import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import model.network as network
import loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy, random
from tqdm import tqdm
from loss import CrossEntropyLabelSmooth
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from model.factory import get_model
import seaborn as sns
import matplotlib.pyplot as plt
import math 

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    dset_lst = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()
    txt_src = open(args.s_dset_path).readlines()

    dset_lst['train'] = txt_tar
    dset_lst['pool'] = txt_test
    dset_lst['label'] = []
    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_src = []
        for i in range(len(txt_src)):
            rec = txt_src[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.src_classes:
                line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                new_src.append(line)
        txt_src = new_src.copy()

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["source"] = ImageList(txt_src, transform=image_train())
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True, num_workers=2,
                                        drop_last=False)
    dsets["source_test"] = ImageList(txt_src, transform=image_test())
    dset_loaders["source_test"] = DataLoader(dsets["source_test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker,
                                        drop_last=False)

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker,
                                      drop_last=False)
    dsets["total"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["total"] = DataLoader(dsets["total"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker,
                                       drop_last=False)

    return dset_loaders, dsets, dset_lst


def obtain_label(loader, model, cls, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            t_domain = torch.ones(inputs.shape[0], dtype=torch.long).cuda()
            feas, _ = model(inputs, t_domain)
            outputs = cls(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    log_str = 'Pseudo label Accuracy = {:.2f}% '.format(accuracy * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')
    return predict.cpu().numpy()


def cal_acc(loader, netF, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            domain = torch.ones(inputs.shape[0], dtype=torch.long).cuda()  # target
            feas, _ = netF(inputs, domain)
            outputs = netC(feas)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent


def pure_training(args, dset_loaders, model, src_classifier, lossnet, discrim, optimizer, optimizer2, optimizer3,lr_scheduler, dset_lst):
    max_iter = (args.max_epoch + 1) * max(len(dset_loaders["target"]), len(dset_loaders["source"]))
    interval_iter = max_iter // args.interval
    # interval_iter = 30
    iter_num = 0
    init_acc = 0.0
    criterion = nn.BCELoss()
    while iter_num < max_iter:
        try:
            inputs_tgt, label_tgt, tar_idx = iter_tgt.next()
        except:
            iter_tgt = iter(dset_loaders["target"])
            inputs_tgt, label_tgt, tar_idx = iter_tgt.next()
        try:
            inputs_src, label_src = iter_src.next()
        except:
            iter_src = iter(dset_loaders["source"])
            inputs_src, label_src = iter_src.next()

        if inputs_src.size(0) <= 1 or inputs_tgt.size(0) <= 1:
            continue
        if iter_num % interval_iter == 0:
            model.eval()
            src_classifier.eval()
            mem_label = obtain_label(dset_loaders['test'], model, src_classifier, args)
            mem_label = torch.from_numpy(mem_label).cuda()
            model.train()

        iter_num += 1
        if iter_num <= int(max_iter / 2):
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)  # feature extractor
            lr_scheduler(optimizer2, iter_num=iter_num, max_iter=max_iter)  # lossnet
            lr_scheduler(optimizer3, iter_num=iter_num, max_iter=max_iter)  # discriminator
        else:
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
            lr_scheduler(optimizer3, iter_num=iter_num, max_iter=max_iter)

        s_domain = torch.zeros(inputs_src.shape[0], dtype=torch.long).cuda()
        t_domain = torch.ones(inputs_tgt.shape[0], dtype=torch.long).cuda()
        t_fake = torch.zeros(inputs_tgt.shape[0], dtype=torch.long).cuda()
        inputs_src, inputs_tgt = inputs_src.cuda(), inputs_tgt.cuda()

        s_outputs, _ = model(inputs_src, s_domain)
        t_outputs, t_mid_blocks = model(inputs_tgt, t_domain)

        # pred labels for tgt
        outputs_test = src_classifier(t_outputs)

        if iter_num > int(max_iter / 2):
            t_mid_blocks[0] = t_mid_blocks[0].detach()
            t_mid_blocks[1] = t_mid_blocks[1].detach()
            t_mid_blocks[2] = t_mid_blocks[2].detach()
            t_mid_blocks[3] = t_mid_blocks[3].detach()
        pred_loss = lossnet(t_mid_blocks)
        pred_loss = pred_loss.view(pred_loss.size(0))

        total_feat = torch.cat((s_outputs, t_outputs), dim=0)
        total_domain = torch.cat((s_domain.type(torch.DoubleTensor).cuda(), t_domain.type(torch.DoubleTensor).cuda()),
                                 dim=0)
        pred_domain = discrim(total_feat.detach())
        pred_domain = pred_domain.squeeze().type(torch.DoubleTensor).cuda()
        discrim_loss = criterion(pred_domain, total_domain)

        # tgt only discrim
        pred_tgt_domain = discrim(t_outputs)
        pred_tgt_domain = pred_tgt_domain.squeeze().type(torch.DoubleTensor).cuda()
        encoder_discrim_loss = criterion(pred_tgt_domain, t_fake.type(torch.DoubleTensor).cuda())

        if args.cls_par > 0:
            pred = mem_label[tar_idx]
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            # new one modified by Tak 080721
            target_loss = nn.CrossEntropyLoss(reduction='none')(outputs_test, pred)
            loss_pred_loss = loss.LossPredLoss2(pred_loss, target_loss, margin=1.0)
            # ###################################################
            classifier_loss *= args.cls_par
            if iter_num < interval_iter and args.dset == "visda-2017":
                classifier_loss *= 0
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        classifier_loss += encoder_discrim_loss
        # encoder 없데이트 w/ pseudo label loss
        optimizer.zero_grad()
        classifier_loss.backward(retain_graph=True)
        optimizer.step()

        # Discriminator loss w/ source and target
        optimizer3.zero_grad()
        discrim_loss.backward()
        optimizer3.step()

        loss_pred_loss += 0.8 * encoder_discrim_loss.type(torch.FloatTensor).cuda()

        if iter_num <= int(max_iter / 2):
            optimizer2.zero_grad()
            loss_pred_loss.backward()
            optimizer2.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            model.eval()
            acc, _ = cal_acc(dset_loaders["test"], model, src_classifier, False)
            if init_acc <= acc:
                init_acc = acc
                best_model = model
                best_loss = lossnet
                best_discrim = discrim
                best_optim = optimizer
                best_optim2 = optimizer2
                best_optim3 = optimizer3
                log_str = 'Task {}: Iter:{}/{}; Accuracy = {:.2f}% best model changed'.format(args.name, iter_num,
                                                                                              max_iter, acc)
            else:
                log_str = 'Task {}: Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
        model.train()
    return best_model, src_classifier, best_loss, best_discrim, best_optim, best_optim2, best_optim3


def active_sample_training(args, dset_loaders, model, src_classifier, lossnet, discrim, optimizer, optimizer2,optimizer3, lr_scheduler, dset_lst, cycle, ratio):
    print("len of label dataset ", len(dset_loaders["label"]))
    max_iter = (args.max_epoch + cycle) * (len(dset_loaders['label']) + ratio)
    # max_iter = (args.max_epoch) * (len(dset_loaders['label']))
    interval_iter = max_iter // ratio
    # interval_iter = max_iter // args.interval
    if interval_iter == 0:
        interval_iter = 2
    iter_num = 0
    init_acc = 0.0
    criterion = nn.BCELoss()
    print(max_iter, interval_iter)
    while iter_num < max_iter:
        try:
            inputs_label, label, label_idx = iter_label.next()
        except:
            iter_label = iter(dset_loaders["label"])
            inputs_label, label, label_idx = iter_label.next()

        if inputs_label.size(0) <= 1:
            continue

        iter_num += 1
        if iter_num <= int(max_iter / 2):
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)  # feature extractor
            lr_scheduler(optimizer2, iter_num=iter_num, max_iter=max_iter)  # lossnet
            # lr_scheduler(optimizer3, iter_num=iter_num, max_iter=max_iter)  # discriminator
        else:
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
            lr_scheduler(optimizer2, iter_num=iter_num, max_iter=max_iter)  # lossnet
            # lr_scheduler(optimizer3, iter_num=iter_num, max_iter=max_iter)

        l_domain = torch.ones(inputs_label.shape[0], dtype=torch.long).cuda()
        t_fake = torch.zeros(inputs_label.shape[0], dtype=torch.long).cuda()

        inputs_label, label = inputs_label.cuda(), label.cuda()
        l_outputs, l_mid_blocks = model(inputs_label, l_domain)

        # pred labels for tgt
        outputs_label = src_classifier(l_outputs)

        pred_loss = lossnet(l_mid_blocks)
        pred_loss = pred_loss.view(pred_loss.size(0))

        classifier_label = nn.CrossEntropyLoss()(outputs_label, label)

        # #828 10:45pm
        target_loss = nn.CrossEntropyLoss(reduction='none')(outputs_label, label)
        loss_pred_loss = loss.LossPredLoss2(pred_loss, target_loss, margin=1.0)

        total_loss = classifier_label.type(torch.FloatTensor).cuda()

        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()


        optimizer2.zero_grad()
        loss_pred_loss.backward()
        optimizer2.step()
        
        model.train()
    return model, src_classifier, lossnet, discrim, optimizer, optimizer2, optimizer3


def label_training(args, dset_loaders, model, src_classifier, lossnet, discrim, optimizer, optimizer2, optimizer3,lr_scheduler, dset_lst, cycle):
    # changed 0828 11:21pm
    # max_iter = (args.max_epoch+ cycle) * max(len(dset_loaders["target"]),len(dset_loaders["source"]))
    max_iter = (args.max_epoch + 3) * max(len(dset_loaders["target"]), len(dset_loaders["source"]))
    ##############
    # office DW or WD turn off max_iter = int(max_iter / 2)
    max_iter = int(max_iter / 2)
    interval_iter = max_iter // args.interval
    # interval_iter = 30
    if interval_iter == 0:
        interval_iter = 2
    iter_num = 0
    init_acc = 0.0
    criterion = nn.BCELoss()
    print(max_iter, interval_iter)
    while iter_num < max_iter:
        try:
            inputs_tgt, label_tgt, tar_idx = iter_tgt.next()
        except:
            iter_tgt = iter(dset_loaders["target"])
            inputs_tgt, label_tgt, tar_idx = iter_tgt.next()
        try:
            inputs_src, label_src = iter_src.next()
        except:
            iter_src = iter(dset_loaders["source"])
            inputs_src, label_src = iter_src.next()

        if inputs_src.size(0) <= 1 or inputs_tgt.size(0) <= 1:
            continue

        if iter_num % interval_iter == 0:
            model.eval()
            src_classifier.eval()
            mem_label = obtain_label(dset_loaders["test"], model, src_classifier, args)
            mem_label = torch.from_numpy(mem_label).cuda()
            model.train()
            src_classifier.eval()

        iter_num += 1
        if iter_num <= int(max_iter / 2):
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)  # feature extractor
            lr_scheduler(optimizer2, iter_num=iter_num, max_iter=max_iter)  # lossnet
            lr_scheduler(optimizer3, iter_num=iter_num, max_iter=max_iter)  # discriminator
        else:
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
            lr_scheduler(optimizer3, iter_num=iter_num, max_iter=max_iter)

        s_domain = torch.zeros(inputs_src.shape[0], dtype=torch.long).cuda()
        t_domain = torch.ones(inputs_tgt.shape[0], dtype=torch.long).cuda()
        t_fake = torch.zeros(inputs_tgt.shape[0], dtype=torch.long).cuda()

        inputs_src, inputs_tgt = inputs_src.cuda(), inputs_tgt.cuda()

        s_outputs, _ = model(inputs_src, s_domain)
        t_outputs, t_mid_blocks = model(inputs_tgt, t_domain)

        # pred labels for tgt
        outputs_test = src_classifier(t_outputs)

        if iter_num > int(max_iter / 2):
            t_mid_blocks[0] = t_mid_blocks[0].detach()
            t_mid_blocks[1] = t_mid_blocks[1].detach()
            t_mid_blocks[2] = t_mid_blocks[2].detach()
            t_mid_blocks[3] = t_mid_blocks[3].detach()
        pred_loss = lossnet(t_mid_blocks)
        pred_loss = pred_loss.view(pred_loss.size(0))

        total_feat = torch.cat((s_outputs, t_outputs), dim=0)
        total_domain = torch.cat((s_domain.type(torch.DoubleTensor).cuda(), t_domain.type(torch.DoubleTensor).cuda()),
                                 dim=0)
        pred_domain = discrim(total_feat.detach())
        pred_domain = pred_domain.squeeze().type(torch.DoubleTensor).cuda()
        discrim_loss = criterion(pred_domain, total_domain)

        # tgt only discrim
        pred_tgt_domain = discrim(t_outputs)
        pred_tgt_domain = pred_tgt_domain.squeeze().type(torch.DoubleTensor).cuda()
        encoder_discrim_loss = criterion(pred_tgt_domain, t_fake.type(torch.DoubleTensor).cuda())
        # print(encoder_discrim_loss)

        if args.cls_par > 0:
            pred = mem_label[tar_idx]
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            # new one modified by Tak 080721
            target_loss = nn.CrossEntropyLoss(reduction='none')(outputs_test, pred)
            loss_pred_loss = loss.LossPredLoss2(pred_loss, target_loss, margin=1.0)
            # ###################################################
            classifier_loss *= args.cls_par
            if iter_num < interval_iter and args.dset == "visda-2017":
                classifier_loss *= 0
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        total_loss = classifier_loss.type(torch.FloatTensor).cuda() + encoder_discrim_loss.type(
            torch.FloatTensor).cuda()
        # # encoder 없데이트 w/ pseudo label loss
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()
        
        # Discriminator loss w/ source and target
        optimizer3.zero_grad()
        discrim_loss.backward()
        optimizer3.step()

        loss_pred_loss += 0.7 * encoder_discrim_loss.type(torch.FloatTensor).cuda()
        if iter_num <= int(max_iter / 2):
            # encoder 업데이트 w/ target discrimination loss
            optimizer2.zero_grad()
            loss_pred_loss.backward()
            optimizer2.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            model.eval()
            src_classifier.eval()
            acc, _ = cal_acc(dset_loaders["total"], model, src_classifier, False)
            if init_acc <= acc:
                init_acc = acc
                best_model = model
                best_loss = lossnet
                best_discrim = discrim
                best_optim = optimizer
                best_optim2 = optimizer2
                best_optim3 = optimizer3
                log_str = 'Total Target Sample Task {}: Iter:{}/{}; # testing data: {}; Accuracy = {:.2f}% best model changed'.format(
                    args.name, iter_num, max_iter, len(dset_lst["train"]), acc)
            else:
                log_str = 'Total Target Sample Task {}: Iter:{}/{}; # testing data: {}; Accuracy = {:.2f}%'.format(
                    args.name, iter_num, max_iter, len(dset_lst["train"]), acc)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
        model.train()
    return best_model, src_classifier, best_loss, best_discrim, best_optim, best_optim2, best_optim3


def train_target(args):
    dset_loaders, dsets, dset_lst = data_load(args)
    if args.dset != "visda-2017":
        model, src_classifier = get_model('resnet50', args.class_num, 256, num_domains=2, pretrained=True)
    else:
        model, src_classifier = get_model('resnet50', args.class_num, 256, num_domains=2, pretrained=True)
    model = model.cuda()
    src_classifier = src_classifier.cuda()
    lossnet = network.LossNet().cuda()
    discrim = network.Discriminator().cuda()
    src_pretrained_path_m = args.output_dir_src + '/FE.pt'
    src_pretrained_path_c = args.output_dir_src + "/CLS.pt"
    model.load_state_dict(torch.load(src_pretrained_path_m))
    src_classifier.load_state_dict(torch.load(src_pretrained_path_c))
    # tsne = TSNE(n_components=2)
    # tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne = TSNE(n_components=2)
    for k, v in src_classifier.named_parameters():
        v.requires_grad = False
    param_group_e = []
    param_group_d = []
    param_group_l = []
    for k, v in model.named_parameters():
        param_group_e += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        # param_group_l += [{'params': v, 'lr': args.lr * args.lr_decay2}]
    for k, v in lossnet.named_parameters():
        param_group_l += [{'params': v, 'lr': args.lr * args.lr_decay2}]
    for k, v in discrim.named_parameters():
        param_group_d += [{'params': v, 'lr': args.lr * args.lr_decay2}]

    optimizer = optim.SGD(param_group_e)
    optimizer = op_copy(optimizer)

    optimizer2 = optim.SGD(param_group_l)
    optimizer2 = op_copy(optimizer2)

    optimizer3 = optim.SGD(param_group_d)
    optimizer3 = op_copy(optimizer3)

    src_classifier.eval()
    ratio = math.ceil(len(dset_lst['train']) * 0.01)
    print(ratio)
    for cycle in range(21):
        log_str = "cycle: {}".format(cycle)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()
        print(log_str)
        if cycle == 0:
            model, src_classifier, lossnet, discrim, optimizer, optimizer2, optimizer3 = pure_training(args,dset_loaders,model,src_classifier,lossnet, discrim,optimizer,optimizer2,optimizer3,lr_scheduler,dset_lst)
        else:
            model, src_classifier, lossnet, discrim, optimizer, optimizer2, optimizer3 = active_sample_training(args,dset_loaders,model,src_classifier,lossnet,discrim,optimizer,optimizer2,optimizer3,lr_scheduler,dset_lst,cycle,ratio)
            model, src_classifier, lossnet, discrim, optimizer, optimizer2, optimizer3 = label_training(args,dset_loaders,model,src_classifier,lossnet,discrim,optimizer,optimizer2,optimizer3,lr_scheduler,dset_lst, cycle)
        # print("Calculating highest losses")
        uncertainty,pred, labels = loss.get_uncertainty(model, src_classifier, lossnet, dset_loaders["test"])

        pred_loss, pred, labels = uncertainty.cpu().numpy(), pred.cpu().numpy(), labels.cpu().numpy()
        with open(args.output_dir + '_{}%_predictions.txt'.format(cycle+1), 'w') as f:
            for (losses, prediction, label) in zip(pred_loss,pred,labels):
                f.write("Loss: {:.3f}, Pred: {}, Label: {} \n".format(losses,int(prediction), int(label)))


        # original
        arg = np.argsort(pred_loss) # low 
        top_uncertainties = arg[arg.shape[0] - ratio:]

        print("top uncertainties ", top_uncertainties)
        
        # top_uncertainties = top_uncertainties.cpu().numpy()
        print("before source ", len(dset_lst['label']), "before target ", len(dset_lst['pool']))
        temp_list = []
        for i in (top_uncertainties):
            temp_list.append(dset_lst['pool'][i])
        
        with open(args.output_dir + '_{}%_quires.txt'.format(cycle+1), 'w') as f:
            for item in temp_list:
                f.write(item + '\n')
            f.write('[')
            for index in top_uncertainties:
                f.write(str(index) + ', ')
            f.write(']')
        # print(temp_list)
        f.close()
        # # remove from the target_list set
        target_list = np.delete(dset_lst['pool'], top_uncertainties, axis=0)
        source_list = np.concatenate((dset_lst['label'], temp_list), axis=0)
        dset_lst['label'] = source_list
        dset_lst['pool'] = target_list
        log_str = "after cycle {} training source {} target {}".format(cycle, len(source_list), len(target_list))
        args.out_file.write(log_str + '\n')
        args.out_file.flush()
        print(log_str + '\n')


        dsets["target"] = ImageList_idx(target_list, transform=image_train())
        dset_loaders["target"] = DataLoader(dsets["target"], batch_size=args.batch_size, shuffle=True,
                                            num_workers=args.worker, drop_last=True)

        dsets["test"] = ImageList_idx(target_list, transform=image_test())
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.worker, drop_last=False)

        dsets["total"] = ImageList_idx(dset_lst['train'], transform=image_test())
        dset_loaders["total"] = DataLoader(dsets["total"], batch_size=args.batch_size, shuffle=False,
                                           num_workers=args.worker, drop_last=False)

        dset_loaders["total2"] = DataLoader(dsets["total"], batch_size=args.batch_size, shuffle=False,
                                           num_workers=args.worker, drop_last=False)

        # # changed 0828 11:21pm
        if args.dset == 'visda-2017':
            dsets["label"] = ImageList_idx(source_list, transform=image_train())
            dset_loaders["label"] = DataLoader(dsets["label"], batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.worker, drop_last=False)
        else:
            dsets["label"] = ImageList_idx(source_list, transform=image_train())
            dset_loaders["label"] = DataLoader(dsets["label"], batch_size=ratio, shuffle=True, num_workers=args.worker,
                                               drop_last=False)
        dsets["label2"] = ImageList_idx(source_list, transform=image_test())
        dset_loaders["label2"] = DataLoader(dsets["label2"], batch_size=ratio, shuffle=True, num_workers=args.worker,
                                           drop_last=False)


        temp_list = []

    # if args.issave:
    #     torch.save(model.state_dict(), osp.join(args.output_dir, "FE" + args.savename + ".pt"))
    #     torch.save(src_classifier.state_dict(), osp.join(args.output_dir, "CLS" + args.savename + ".pt"))
    #     torch.save(lossnet.state_dict(), osp.join(args.output_dir, "lossnet" + args.savename + ".pt"))
    #     torch.save(discrim.state_dict(), osp.join(args.output_dir, "discrim" + args.savename + ".pt"))


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='please')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=0, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home',
                        choices=['visda-2017', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")

    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    #### changed #####
    parser.add_argument('--cls_par', type=float, default=0.3)
    ##################
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=True)
    args = parser.parse_args()

    if args.dset == 'office-home':
        # names = ['Art', 'Clipart', 'Product', 'Real_World']
        names = ['Clipart', 'Product']
        args.class_num = 65
    if args.dset == 'office':
        # names = ['amazon', 'dslr', 'webcam']
        names = ['amazon', 'dslr']
        args.class_num = 31
    if args.dset == 'visda-2017':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        if args.dset == 'office-home':
            folder = '/root/workspace/da_dsets/'
            args.s_dset_path = folder + args.dset + '/' + names[args.s] + '.txt'
            args.t_dset_path = folder + args.dset + '/' + names[args.t] + '.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '.txt'
        else:

            folder = '/root/workspace/da_dsets/'
            args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
            args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]

        args.output_dir_src = osp.join(args.output_src, args.dset, names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.dset, names[args.s][0].upper() + names[args.t][0].upper())
        args.name = names[args.s][0].upper() + names[args.t][0].upper()        # encoder 없데이트 w/ pseudo label loss
        

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par)
        if args.da == 'pda':
            args.gent = ''
            args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        train_target(args) 