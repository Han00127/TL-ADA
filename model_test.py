import argparse

import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch.optim as optim
from torchvision import transforms
import model.network as network
from model.factory import get_model
# import ori_network
import loss
from torch.utils.data import DataLoader
from data_list import ImageList
import random, pdb, math, copy
from tqdm import tqdm
from loss import CrossEntropyLabelSmooth
import torch
import torch.nn as nn
def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
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
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

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
def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

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
    for i in range(len(txt_test)):
        rec = txt_test[i]
        reci = rec.strip().split(' ')
        if int(reci[1]) in args.tar_classes:
            if int(reci[1]) in args.src_classes:
                line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                new_tar.append(line)
            else:
                line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                new_tar.append(line)
    txt_test = new_tar.copy()

    if args.trte == "val":
        dsize = len(txt_src)
        tr_size = int(0.9 * dsize)
        tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    else:
        dsize = len(txt_src)
        tr_size = int(0.9 * dsize)
        _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
        tr_txt = txt_src

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 2, shuffle=True, num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def cal_acc(loader, netF, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            domain = torch.zeros(inputs.shape[0], dtype=torch.long).cuda()  # source
            feas, _ = netF(inputs,domain)
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
def train(args):
    dset_loaders = data_load(args)
    if args.dset != "visda-2017":
        # model,classifier = get_model('resnet50', args.class_num, 256, num_domains=2, pretrained=True)
        model, classifier = get_model('resnet50', args.class_num, 256, num_domains=2, pretrained=True)
    else:
        model, classifier = get_model('resnet50', args.class_num, 256, num_domains=2, pretrained=True)

    model = model.cuda()
    classifier = classifier.cuda()
    param_group = []
    for k,v in model.named_parameters():
        param_group += [{'params': v, 'lr': args.lr*0.1}]
    for k,v in classifier.named_parameters():
        param_group += [{'params': v, 'lr': args.lr*0.1}]


    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0
    acc_init = 0
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    model.train()
    classifier.train()

    while iter_num < max_iter:
        try:
            inputs,labels = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs,labels = iter_source.next()

        iter_num += 1
        lr_scheduler(optimizer, iter_num = iter_num, max_iter=max_iter)

        inputs, labels = inputs.cuda(), labels.cuda()
        domain =  torch.zeros(inputs.shape[0], dtype=torch.long).cuda() # source

        outputs,_ = model(inputs,domain)
        outputs_score = classifier(outputs)
        classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_score,labels)

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()
        #
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            model.eval()
            classifier.eval()
            acc,_ = cal_acc(dset_loaders["source_te"],model,classifier,False)
            log_str = ' Iter:{}/{}; Accuracy = {:.2f}%'.format( iter_num, max_iter, acc)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')

            if acc >= acc_init:
                acc_init = acc
                best_model = model.state_dict()
                best_class = classifier.state_dict()
        model.train()
        classifier.train()

    torch.save(best_model, osp.join(args.output_dir_src, "FE.pt"))
    torch.save(best_class, osp.join(args.output_dir_src, "CLS.pt"))
    return model, classifier


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='please')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=20, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office', choices=['visda-2017', 'office', 'office-home', 'office-caltech', 'image-clef'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="resnet50, resnet101,resnet152")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'Real_World']
        args.class_num = 65
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'visda-2017':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'image-clef':
        names = ['b', 'c','i','p']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
    args.src_classes = [i for i in range(args.class_num)]
    args.tar_classes = [i for i in range(args.class_num)]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if args.dset == 'office-home':
        folder = '/root/workspace/da_dsets/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '.txt'
    else:
        folder = '/root/workspace/da_dsets/'
        args.s_dset_path =  folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'


    args.output_dir_src = osp.join(args.output, args.dset, names[args.s][0].upper())
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    model, cls_f = train(args)

