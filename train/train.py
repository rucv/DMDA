# export PYTHONPATH='./..'
import random
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from dataset.data_loader import GetLoader
from dataset.datasets import Dataset
from dataset.usps import load_usps
from dataset.mnist import load_mnist
from torchvision import datasets, models, transforms
from models.mod import Feature_SM, Predictor_SM, Domain_Classifier_SM, Transfrom_SM
from models.mod import Feature_UM, Predictor_UM, Domain_Classifier_UM, Transfrom_UM
from models.mod import Feature_MU, Predictor_MU, Domain_Classifier_MU, Transfrom_MU
from models.mod import Feature_MM, Predictor_MM, Domain_Classifier_MM, Transfrom_MM
from models.mod import Feature_SG, Predictor_SG, Domain_Classifier_SG, Transfrom_SG
from models.mod import Feature_RESNET, Predictor_RESNET, Domain_Classifier_RESNET, Transfrom_RESNET

import numpy as np
from test import test
from lr_schedule import lr_scheduler

import pylab
from tsne import *

import argparse

parser = argparse.ArgumentParser(description='PyTorch DA Implementation')
parser.add_argument('--gpu',    type=int,   default=0,       help='Which GPU you want to use')
parser.add_argument('--alg',    type=int,   default=0,       help='Which Algorithm you want to use')
parser.add_argument('--source', type=str,   default='SVHN',  help='Name of SOURCE Domain')
parser.add_argument('--target', type=str,   default='MNIST', help='Name of TARGET Domain')
parser.add_argument('--lr',     type=float, default=2e-4,    help='Learning rate')
parser.add_argument('--opt',    type=str,   default='Adam',  help='Learning rate')
parser.add_argument('--sch',    type=bool,  default=False,   help='Using lr_scheduler')
args = parser.parse_args()

source_dataset_name = 'MNIST'
target_dataset_name = 'mnist_m'
source_image_root = os.path.join('../', 'dataset', source_dataset_name)
target_image_root = os.path.join('../', 'dataset', target_dataset_name)
office31_image_root  = os.path.join('../', 'dataset', 'office_31')
visda2017_image_root = os.path.join('../', 'dataset', 'VisDA2017')
gtsrb_image_root     = os.path.join('../', 'dataset', 'GTSRB')
synth_image_root     = os.path.join('../', 'dataset', 'SYN')
model_root = os.path.join('../', 'models/checkpoints')

cuda = True
cudnn.benchmark = True
opt = args.opt
lr = args.lr
batch_size = 128
image_size = 0
n_epoch = 100

source = args.source
target = args.target


# Loading Model AND Loading Dataset
if target == 'MNIST' and source == 'SVHN':
    image_size = 32
    G1 = Feature_SM()
    G2 = Feature_SM()
    C1 = Predictor_SM()
    C2 = Predictor_SM()
    C3 = Predictor_SM()
    C4 = Predictor_SM()
    T1 = Transfrom_SM()
    T2 = Transfrom_SM()
    D1 = Domain_Classifier_SM()
    D2 = Domain_Classifier_SM()
    img_transform_source = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])
    img_transform_target = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset_source = datasets.SVHN(
        root='../dataset',
        split='train',
        transform=img_transform_target,
        download=False)
    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
    dataset_target = datasets.MNIST(
        root='../dataset',
        train=True,
        transform=img_transform_source,
        download=True)
    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
elif target == 'MNIST' and source == 'USPS':
    image_size = 28
    G1 = Feature_UM()
    G2 = Feature_UM()
    C1 = Predictor_UM()
    C2 = Predictor_UM()
    C3 = Predictor_UM()
    C4 = Predictor_UM()
    T1 = Transfrom_UM()
    T2 = Transfrom_UM()
    D1 = Domain_Classifier_UM()
    D2 = Domain_Classifier_UM()
    img_transform_source = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])
    img_transform_target = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])
    img_train, label_train, _, _ = load_usps()
    dataset_source = Dataset(img_train,label_train,img_transform_target)
    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
    img_train, label_train, _, _ = load_mnist(False,True,'yes')
    dataset_target = Dataset(img_train,label_train,img_transform_source)
    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
elif target == 'MNIST_M' and source == 'MNIST':
    image_size = 32
    G1 = Feature_MM()
    G2 = Feature_MM()
    C1 = Predictor_MM()
    C2 = Predictor_MM()
    C3 = Predictor_MM()
    C4 = Predictor_MM()
    T1 = Transfrom_MM()
    T2 = Transfrom_MM()
    D1 = Domain_Classifier_MM()
    D2 = Domain_Classifier_MM()
    img_transform_source = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    img_transform_target = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset_source = datasets.MNIST(
        root='../dataset',
        train=True,
        transform=img_transform_source,
        download=True
    )
    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
    train_list = os.path.join(target_image_root, 'mnist_m_train_labels.txt')
    dataset_target = GetLoader(
        data_root=os.path.join(target_image_root, 'mnist_m_train'),
        data_list=train_list,
        transform=img_transform_target
    )
    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
elif target == 'USPS' and source == 'MNIST':
    image_size = 28
    G1 = Feature_MU()
    G2 = Feature_MU()
    C1 = Predictor_MU()
    C2 = Predictor_MU()
    C3 = Predictor_MU()
    C4 = Predictor_MU()
    T1 = Transfrom_MU()
    T2 = Transfrom_MU()
    D1 = Domain_Classifier_MU()
    D2 = Domain_Classifier_MU()
    img_transform_source = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])
    img_transform_target = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    dataset_source = datasets.MNIST(
        root='../dataset',
        train=True,
        transform=img_transform_source,
        download=True)
    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
    dataset_target = datasets.USPS(
        root='../dataset',
        train=True,
        transform=img_transform_target,
        download=False)
    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
elif target == 'GTSRB' and source == 'SYN':
    image_size = 40
    G1 = Feature_SG()
    G2 = Feature_SG()
    C1 = Predictor_SG()
    C2 = Predictor_SG()
    C3 = Predictor_SG()
    C4 = Predictor_SG()
    T1 = Transfrom_SG()
    T2 = Transfrom_SG()
    D1 = Domain_Classifier_SG()
    D2 = Domain_Classifier_SG()
    img_transform_target = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    train_list = os.path.join(synth_image_root, 'train.txt')
    dataset_source = GetLoader(
        data_root=synth_image_root,
        data_list=train_list,
        transform=img_transform_target
    )
    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)

    train_list = os.path.join(gtsrb_image_root, 'train', 'train.txt')
    dataset_target = GetLoader(
        data_root=os.path.join(gtsrb_image_root, 'train'),
        data_list=train_list,
        transform=img_transform_target
    )
    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
elif target == 'W' and source == 'A':
    image_size = 227
    G1 = Feature_RESNET()
    frozen = 0
    for child in G1.features:
        if frozen < 7:
            frozen+=1
            for param in child.parameters():
                param.requires_grad = False
    # frozen = 0
    # frozen_child = 0
    # for child in G1.features:
    #     if frozen <= 7:
    #         frozen+=1
    #         if frozen == 7:
    #             for c in child.children():
    #                 for param in c.parameters():
    #                     frozen_child+=1
    #                     if frozen_child != 2:
    #                         param.requires_grad = False
    #         else:
    #             for param in child.parameters():
    #                 param.requires_grad = False

    G2 = Feature_RESNET()
    frozen = 0
    for child in G2.features:
        if frozen < 7:
            frozen+=1
            for param in child.parameters():
                param.requires_grad = False
    C1 = Predictor_RESNET()
    C2 = Predictor_RESNET()
    T1 = Transfrom_RESNET()
    T2 = Transfrom_RESNET()
    D1 = Domain_Classifier_RESNET()
    D2 = Domain_Classifier_RESNET()
    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    source_list = os.path.join(office31_image_root, 'amazon_list.txt')
    dataset_source = GetLoader(
        data_root=office31_image_root,
        data_list=source_list,
        transform=img_transform)
    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
    target_list = os.path.join(office31_image_root, 'webcam_list.txt')
    dataset_target = GetLoader(
        data_root=office31_image_root,
        data_list=target_list,
        transform=img_transform)
    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
elif target == 'W' and source == 'D':
    image_size = 227
    G1 = Feature_RESNET()
    G2 = Feature_RESNET()
    C1 = Predictor_RESNET()
    C2 = Predictor_RESNET()
    T1 = Transfrom_RESNET()
    T2 = Transfrom_RESNET()
    D1 = Domain_Classifier_RESNET()
    D2 = Domain_Classifier_RESNET()
    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    source_list = os.path.join(office31_image_root, 'dslr_train.txt')
    dataset_source = GetLoader(
        data_root=office31_image_root,
        data_list=source_list,
        transform=img_transform)
    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
    target_list = os.path.join(office31_image_root, 'wabcam_train.txt')
    dataset_target = GetLoader(
        data_root=office31_image_root,
        data_list=target_list,
        transform=img_transform)
    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
elif target == 'D' and source == 'W':
    image_size = 227
    G1 = Feature_RESNET()
    G2 = Feature_RESNET()
    C1 = Predictor_RESNET()
    C2 = Predictor_RESNET()
    T1 = Transfrom_RESNET()
    T2 = Transfrom_RESNET()
    D1 = Domain_Classifier_RESNET()
    D2 = Domain_Classifier_RESNET()
    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    source_list = os.path.join(office31_image_root, 'webcam_train.txt')
    dataset_source = GetLoader(
        data_root=office31_image_root,
        data_list=source_list,
        transform=img_transform)
    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
    target_list = os.path.join(office31_image_root, 'dslr_train.txt')
    dataset_target = GetLoader(
        data_root=office31_image_root,
        data_list=target_list,
        transform=img_transform)
    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
elif target == 'D' and source == 'A':
    image_size = 227
    G1 = Feature_RESNET()
    G2 = Feature_RESNET()
    C1 = Predictor_RESNET()
    C2 = Predictor_RESNET()
    T1 = Transfrom_RESNET()
    T2 = Transfrom_RESNET()
    D1 = Domain_Classifier_RESNET()
    D2 = Domain_Classifier_RESNET()
    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    source_list = os.path.join(office31_image_root, 'amazon_train.txt')
    dataset_source = GetLoader(
        data_root=office31_image_root,
        data_list=source_list,
        transform=img_transform)
    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
    target_list = os.path.join(office31_image_root, 'dslr_train.txt')
    dataset_target = GetLoader(
        data_root=office31_image_root,
        data_list=target_list,
        transform=img_transform)
    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
elif target == 'A' and source == 'D':
    image_size = 227
    G1 = Feature_RESNET()
    G2 = Feature_RESNET()
    C1 = Predictor_RESNET()
    C2 = Predictor_RESNET()
    T1 = Transfrom_RESNET()
    T2 = Transfrom_RESNET()
    D1 = Domain_Classifier_RESNET()
    D2 = Domain_Classifier_RESNET()
    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    source_list = os.path.join(office31_image_root, 'dslr_train.txt')
    dataset_source = GetLoader(
        data_root=office31_image_root,
        data_list=source_list,
        transform=img_transform)
    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
    target_list = os.path.join(office31_image_root, 'amazon_train.txt')
    dataset_target = GetLoader(
        data_root=office31_image_root,
        data_list=target_list,
        transform=img_transform)
    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
elif target == 'A' and source == 'W':
    image_size = 227
    G1 = Feature_RESNET()
    G2 = Feature_RESNET()
    C1 = Predictor_RESNET()
    C2 = Predictor_RESNET()
    T1 = Transfrom_RESNET()
    T2 = Transfrom_RESNET()
    D1 = Domain_Classifier_RESNET()
    D2 = Domain_Classifier_RESNET()
    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    source_list = os.path.join(office31_image_root, 'webcam_list.txt')
    dataset_source = GetLoader(
        data_root=office31_image_root,
        data_list=source_list,
        transform=img_transform)
    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
    target_list = os.path.join(office31_image_root, 'amazon_list.txt')
    dataset_target = GetLoader(
        data_root=office31_image_root,
        data_list=target_list,
        transform=img_transform)
    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
else:
    image_size = 224
    G1 = Feature_RESNET()
    # G1.load_state_dict(torch.load(os.path.join('{0}/VisDA/{1}_{2}_model_epoch_G1_{3}.pth'.format(model_root,source,target,0))))
    frozen = 0
    for child in G1.features:
        if frozen < 7:
            frozen+=1
            for param in child.parameters():
                param.requires_grad = False
    G2 = Feature_RESNET()
    frozen = 0
    for child in G2.features:
        if frozen < 7:
            frozen+=1
            for param in child.parameters():
                param.requires_grad = False
    # G2.load_state_dict(torch.load(os.path.join('{0}/VisDA/{1}_{2}_model_epoch_G1_{3}.pth'.format(model_root,source,target,0))))
    C1 = Predictor_RESNET()
    C2 = Predictor_RESNET()
    C3 = Predictor_RESNET()
    C4 = Predictor_RESNET()
    T1 = Transfrom_RESNET()
    T2 = Transfrom_RESNET()
    D1 = Domain_Classifier_RESNET()
    D2 = Domain_Classifier_RESNET()
    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        # transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
    ])
    source_list = os.path.join(visda2017_image_root, 'train' ,'image_list.txt')
    dataset_source = GetLoader(
        data_root=os.path.join(visda2017_image_root, 'train'),
        data_list=source_list,
        transform=img_transform)
    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
    target_list = os.path.join(visda2017_image_root, 'validation','image_list.txt')
    dataset_target = GetLoader(
        data_root=os.path.join(visda2017_image_root, 'validation'),
        data_list=target_list,
        transform=img_transform)
    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)

GPU = args.gpu
G1.cuda(device = GPU)
G2.cuda(device = GPU)
C1.cuda(device = GPU)
C2.cuda(device = GPU)
C3.cuda(device = GPU)
C4.cuda(device = GPU)
D1.cuda(device = GPU)
D2.cuda(device = GPU)
T1.cuda(device = GPU)
T2.cuda(device = GPU)

# SET LOSS FUNCTION
loss_class = torch.nn.CrossEntropyLoss()
loss_domain = torch.nn.NLLLoss()

if cuda:
    loss_class = loss_class.cuda(device = GPU)
    loss_domain = loss_domain.cuda(device = GPU)

def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))



def get_optimizer(model):
    learning_rate = args.lr
    param_group = []
    param_group += [{'params': model.base_network.parameters(),
                     'lr': learning_rate}]
    optimizer = optim.SGD(param_group, momentum=0.9)
    return optimizer


def adjust_learning_rate(optimizer, global_step, lr_0):
    gamma = 10
    power = 0.75
    lr = lr_0 / (1 + gamma * global_step/n_epoch)**power
    # print(optimizer.param_groups)
    for i in range(len(optimizer.param_groups)):
        optimizer.param_groups[i]['lr'] = lr
    return optimizer

def train(n_epoch=0):
    if opt == 'Adam':
        # Adam
        print("Adam:", lr)
        opt_g1 = optim.Adam(G1.parameters() ,lr=lr,weight_decay=0.0005)
        opt_g2 = optim.Adam(G2.parameters() ,lr=lr,weight_decay=0.0005)
        opt_c1 = optim.Adam(C1.parameters() ,lr=lr,weight_decay=0.0005)
        opt_c2 = optim.Adam(C2.parameters() ,lr=lr,weight_decay=0.0005)
        opt_c3 = optim.Adam(C3.parameters() ,lr=lr,weight_decay=0.0005)
        opt_c4 = optim.Adam(C4.parameters() ,lr=lr,weight_decay=0.0005)
        opt_d1 = optim.Adam(D1.parameters() ,lr=lr,weight_decay=0.0005)
        opt_d2 = optim.Adam(D2.parameters() ,lr=lr,weight_decay=0.0005)
        opt_t1 = optim.Adam(T1.parameters() ,lr=lr,weight_decay=0.0005)
        opt_t2 = optim.Adam(T2.parameters() ,lr=lr,weight_decay=0.0005)
    elif opt == 'SGD':
        # SGD
        print("SGD:", lr)
        opt_g1 = optim.SGD(G1.parameters(), momentum=0.9,lr=lr,weight_decay=0.0005)
        opt_g2 = optim.SGD(G2.parameters(), momentum=0.9,lr=lr,weight_decay=0.0005)
        opt_c1 = optim.SGD(C1.parameters(), momentum=0.9,lr=lr,weight_decay=0.0005)
        opt_c2 = optim.SGD(C2.parameters(), momentum=0.9,lr=lr,weight_decay=0.0005)
        opt_c3 = optim.SGD(C3.parameters(), momentum=0.9,lr=lr,weight_decay=0.0005)
        opt_c4 = optim.SGD(C4.parameters(), momentum=0.9,lr=lr,weight_decay=0.0005)
        opt_d1 = optim.SGD(D1.parameters(), momentum=0.9,lr=lr,weight_decay=0.0005)
        opt_d2 = optim.SGD(D2.parameters(), momentum=0.9,lr=lr,weight_decay=0.0005)
        opt_t1 = optim.SGD(T1.parameters(), momentum=0.9,lr=lr,weight_decay=0.0005)
        opt_t2 = optim.SGD(T2.parameters(), momentum=0.9,lr=lr,weight_decay=0.0005)


    G1.train()
    G2.train()
    C1.train()
    C2.train()
    C3.train()
    C4.train()
    D1.train()
    D2.train()
    T1.train()
    T2.train()
    torch.cuda.manual_seed(1)
    avg_loss = 0
    count = 1

    for epoch in xrange(n_epoch):
        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)
        print(len(dataloader_source), len(dataloader_target))
        i = 0
        while i < len_dataloader:
            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            # training model using source data
            data_source = data_source_iter.next()
            img_s, label_s = data_source
            data_target = data_target_iter.next()
            img_t, label_t = data_target

            opt_g1.zero_grad()
            opt_g2.zero_grad()
            opt_c1.zero_grad()
            opt_c2.zero_grad()
            opt_c3.zero_grad()
            opt_c4.zero_grad()
            opt_d1.zero_grad()
            opt_d2.zero_grad()
            opt_t1.zero_grad()
            opt_t2.zero_grad()

            batch_size = min(len(label_s), len(img_t))
            input_img_s = torch.FloatTensor(batch_size, 3, image_size, image_size)
            class_label_s = torch.LongTensor(batch_size)
            domain_label_s = torch.zeros(batch_size)
            domain_label_s = domain_label_s.long()
            input_img_t = torch.FloatTensor(batch_size, 3, image_size, image_size)
            class_label_t = torch.LongTensor(batch_size)
            domain_label_t = torch.ones(batch_size)
            domain_label_t = domain_label_t.long()

            if cuda:
                img_s = img_s.cuda(device = GPU)
                label_s = label_s.cuda(device = GPU)
                input_img_s = input_img_s.cuda(device = GPU)
                class_label_s = class_label_s.cuda(device = GPU)
                domain_label_s = domain_label_s.cuda(device = GPU)
                img_t = img_t.cuda(device = GPU)
                input_img_t = input_img_t.cuda(device = GPU)
                class_label_t = class_label_t.cuda(device = GPU)
                domain_label_t = domain_label_t.cuda(device = GPU)

            input_img_s.resize_as_(img_s).copy_(img_s)
            class_label_s.resize_as_(label_s).copy_(label_s)
            class_label_s = class_label_s.narrow(0,0,batch_size)

            input_img_t.resize_as_(img_t).copy_(img_t)
            class_label_t.resize_as_(label_t).copy_(label_t)
            class_label_t = class_label_t.narrow(0,0,batch_size)

###############################################################################
            if args.alg == 0:
                opt_g1 = adjust_learning_rate(opt_g1,epoch,lr)
                opt_c1 = adjust_learning_rate(opt_c1,epoch,lr)
                opt_d1 = adjust_learning_rate(opt_d1,epoch,lr)
                opt_t1 = adjust_learning_rate(opt_t1,epoch,lr)
                # MINE with share fc1
                feat_s = G1(img_s)
                feat_t = G1(img_t)
                feat_s = feat_s.narrow(0, 0, batch_size)
                feat_t = feat_t.narrow(0, 0, batch_size)
                fc1_s = T1(feat_s,False)
                fc1_t = T1(feat_t,False)
                output_ls = C1(fc1_s)
                loss_ls = loss_class(output_ls, class_label_s)
                D1.set_lambda(alpha)
                output_ds = D1(fc1_s,True)
                output_dt = D1(fc1_t,True)
                loss_ds = loss_domain(output_ds, domain_label_s)
                loss_dt = loss_domain(output_dt, domain_label_t)
                loss_gif = loss_ds + loss_dt + loss_ls
                loss_gif.backward()
                opt_g1.step()
                opt_d1.step() # Here should be only for predictor
                opt_c1.step()
                opt_t1.step()

                opt_g1.zero_grad()
                opt_c1.zero_grad()
                opt_d1.zero_grad()
                opt_t1.zero_grad()

                feat_s = G2(img_s)
                feat_t = G2(img_t)
                feat_s = feat_s.narrow(0, 0, batch_size)
                feat_t = feat_t.narrow(0, 0, batch_size)
                fc2_s = T2(feat_s,False)
                fc2_t = T2(feat_t,False)
                output_ds = D2(fc2_s, False)
                output_dt = D2(fc2_t, False)
                loss_ds = loss_domain(output_ds, domain_label_s)
                loss_dt = loss_domain(output_dt, domain_label_t)
                output_ls = C2(fc2_s,False)
                # output_lt = C2(feat_t,False)
                loss_ls = loss_class(output_ls, class_label_s)
                loss_ngif = loss_ls + loss_ds + loss_dt
                loss_ngif.backward()
                opt_g2.step()
                opt_c2.step()
                opt_d2.step()
                opt_t2.step()

                opt_g2.zero_grad()
                opt_c2.zero_grad()
                opt_d2.zero_grad()
                opt_t2.zero_grad()

                feat_s1 = G1(img_s)
                feat_t1 = G1(img_t)
                feat_s2 = G2(img_s)
                feat_t2 = G2(img_t)
                feat_s1 = feat_s1.narrow(0, 0, batch_size)
                feat_t1 = feat_t1.narrow(0, 0, batch_size)
                feat_s2 = feat_s2.narrow(0, 0, batch_size)
                feat_t2 = feat_t2.narrow(0, 0, batch_size)
                fc1_s= T1(feat_s1,False)
                fc1_t= T1(feat_t1,False)
                fc2_s= T2(feat_s2,False)
                fc2_t= T2(feat_t2,False)
                output_ls1 = C1(fc1_s,False)
                output_lt1 = C1(fc1_t,False)
                output_ls2 = C2(fc2_s,False)
                output_lt2 = C2(fc2_t,False)
                loss_diss = discrepancy(output_ls1, output_ls2)
                loss_dist = discrepancy(output_lt1, output_lt2)
                T1.set_lambda(alpha)
                T2.set_lambda(alpha)
                fc1_s= T1(feat_s1,True)
                fc1_t= T1(feat_t1,True)
                fc2_s= T2(feat_s2,True)
                fc2_t= T2(feat_t2,True)
                dis_sf = discrepancy(fc1_s, fc2_s)
                dis_tf = discrepancy(fc1_t, fc2_t)
                loss_dis_f = dis_sf + dis_tf + loss_dist + loss_diss
                loss_dis_f.backward()
                opt_g1.step()
                opt_g2.step()
                opt_t1.step()
                opt_t2.step()

                # avg_loss += loss.cpu().data.numpy()
                # print 'dis_s: %f' % dis_sf.cpu().data.numpy(), 'dis_t: %f' % dis_tf.cpu().data.numpy(), 'Lgif: %f' % loss_gif.cpu().data.numpy(), 'Lngif: %f' % loss_ngif.cpu().data.numpy()#, 'Loss: %f' % loss.cpu().data.numpy(),'Avg Loss: %f' % (avg_loss/count)
            elif args.alg == 1:
                opt_g1 = adjust_learning_rate(opt_g1,epoch,lr)
                opt_c1 = adjust_learning_rate(opt_c1,epoch,lr)
                opt_c2 = adjust_learning_rate(opt_c2,epoch,lr)
                opt_d1 = adjust_learning_rate(opt_d1,epoch,lr)
                opt_t1 = adjust_learning_rate(opt_t1,epoch,lr)

                # get features from G1
                feat_s = G1(img_s)
                feat_s = feat_s.narrow(0, 0, batch_size)
                # Do Linear transform
                fc1_s = T1(feat_s,False)
                #Pass features_s to C1 and C2
                output_ls1 = C1(fc1_s)
                output_ls2 = C2(fc1_s)
                loss_ls1 = loss_class(output_ls1, class_label_s)
                loss_ls2 = loss_class(output_ls2, class_label_s)
                loss_ls = loss_ls1 + loss_ls2
                loss_ls.backward()
                opt_g1.step()
                opt_c1.step()
                opt_c2.step()
                opt_t1.step()

                opt_g1.zero_grad()
                opt_c1.zero_grad()
                opt_c2.zero_grad()
                opt_d1.zero_grad()
                opt_t1.zero_grad()

                # get features from G1
                feat_s = G1(img_s)
                feat_t = G1(img_t)
                feat_s = feat_s.narrow(0, 0, batch_size)
                feat_t = feat_t.narrow(0, 0, batch_size)
                # Do Linear transform
                fc1_s = T1(feat_s,False)
                fc1_t = T1(feat_t,False)
                #Pass features_s to C1 and C2
                output_ls1 = C1(fc1_s, False)
                output_ls2 = C2(fc1_s, False)
                loss_ls1 = loss_class(output_ls1, class_label_s)
                loss_ls2 = loss_class(output_ls2, class_label_s)
                loss_ls = loss_ls1 + loss_ls2
                #Pass features_t to C1 and C2
                output_lt1 = C1(fc1_t, False)
                output_lt2 = C2(fc1_t, False)
                loss_dis = discrepancy(output_lt1, output_lt2)
                loss = loss_ls - loss_dis
                loss.backward()
                opt_c1.step() # Here should be only for predictor
                opt_c2.step()

                opt_g1.zero_grad()
                opt_t1.zero_grad()
                opt_c1.zero_grad()
                opt_c2.zero_grad()

                for k in xrange(4):#h-parameters too lager
                    feat_t = G1(img_t)
                    feat_t = feat_t.narrow(0, 0, batch_size)
                    fc1_t = T1(feat_t,False)
                    output_t1 = C1(fc1_t, False)
                    output_t2 = C2(fc1_t, False)
                    loss_dis = discrepancy(output_t1, output_t2)
                    loss_dis.backward()
                    opt_g1.step()
                    opt_t1.step()
                    opt_g1.zero_grad()
                    opt_t1.zero_grad()
                    opt_c1.zero_grad()
                    opt_c2.zero_grad()

                # get features from G1
                feat_s = G1(img_s)
                feat_t = G1(img_t)
                feat_s = feat_s.narrow(0, 0, batch_size)
                feat_t = feat_t.narrow(0, 0, batch_size)
                # Do Linear transform
                fc1_s = T1(feat_s,False)
                fc1_t = T1(feat_t,False)
                #Pass features_s to C1 and C2
                output_ls1 = C1(fc1_s)
                output_ls2 = C2(fc1_s)
                loss_ls1 = loss_class(output_ls1, class_label_s)
                loss_ls2 = loss_class(output_ls2, class_label_s)
                loss_ls = loss_ls1 + loss_ls2
                #Pass features to Domain discriminator and use GRL
                D1.set_lambda(alpha)
                output_ds = D1(fc1_s,True)
                output_dt = D1(fc1_t,True)
                loss_ds = loss_domain(output_ds, domain_label_s)
                loss_dt = loss_domain(output_dt, domain_label_t)
                loss_gif = loss_ds + loss_dt + loss_ls
                loss_gif.backward()
                opt_g1.step()
                opt_c1.step()
                opt_c2.step()
                opt_d1.step()
                opt_t1.step()


                # Second Part for G2 ngif
                feat_s = G2(img_s)
                feat_t = G2(img_t)
                feat_s = feat_s.narrow(0, 0, batch_size)
                feat_t = feat_t.narrow(0, 0, batch_size)
                fc2_s = T2(feat_s,False)
                fc2_t = T2(feat_t,False)
                output_ds = D2(fc2_s, False)
                output_dt = D2(fc2_t, False)
                loss_ds = loss_domain(output_ds, domain_label_s)
                loss_dt = loss_domain(output_dt, domain_label_t)
                output_ls3 = C3(fc2_s,False)
                loss_ls3 = loss_class(output_ls3, class_label_s)
                loss_ls = loss_ls3
                loss_ngif = loss_ls + loss_ds + loss_dt
                loss_ngif.backward()
                opt_g2.step()
                opt_d2.step()
                opt_t2.step()
                opt_c3.step()

                opt_g2.zero_grad()
                opt_d2.zero_grad()
                opt_t2.zero_grad()
                opt_c3.zero_grad()

                feat_s1 = G1(img_s)
                feat_t1 = G1(img_t)
                feat_s2 = G2(img_s)
                feat_t2 = G2(img_t)
                feat_s1 = feat_s1.narrow(0, 0, batch_size)
                feat_t1 = feat_t1.narrow(0, 0, batch_size)
                feat_s2 = feat_s2.narrow(0, 0, batch_size)
                feat_t2 = feat_t2.narrow(0, 0, batch_size)
                fc1_s= T1(feat_s1,False)
                fc1_t= T1(feat_t1,False)
                fc2_s= T2(feat_s2,False)
                fc2_t= T2(feat_t2,False)
                output_ls1 = C1(fc1_s,False)
                output_lt1 = C1(fc1_t,False)
                output_ls2 = C3(fc2_s,False)
                output_lt2 = C3(fc2_t,False)
                loss_diss = discrepancy(output_ls1, output_ls2)
                loss_dist = discrepancy(output_lt1, output_lt2)
                T1.set_lambda(alpha)
                T2.set_lambda(alpha)
                fc1_s= T1(feat_s1,True)
                fc1_t= T1(feat_t1,True)
                fc2_s= T2(feat_s2,True)
                fc2_t= T2(feat_t2,True)
                dis_sf = discrepancy(fc1_s, fc2_s)
                dis_tf = discrepancy(fc1_t, fc2_t)
                loss_dis_f = dis_sf + dis_tf + loss_dist + loss_diss
                loss_dis_f.backward()
                opt_g1.step()
                opt_g2.step()
                opt_t1.step()
                opt_t2.step()

                # avg_loss += loss.cpu().data.numpy()
                # print 'dis_s: %f' % dis_sf.cpu().data.numpy(), 'dis_t: %f' % dis_tf.cpu().data.numpy(), 'Lgif: %f' % loss_gif.cpu().data.numpy(), 'Lngif: %f' % loss_ngif.cpu().data.numpy()#, 'Loss: %f' % loss.cpu().data.numpy(),'Avg Loss: %f' % (avg_loss/count)
            elif args.alg == 2:
                # opt_g1 = adjust_learning_rate(opt_g1,epoch,lr)
                # opt_c1 = adjust_learning_rate(opt_c1,epoch,lr)
                # opt_d1 = adjust_learning_rate(opt_d1,epoch,lr)
                # MCD
                feat_s = G1(img_s)
                feat_s = feat_s.narrow(0, 0, batch_size)
                # fc1_s = T1(feat_s,False)
                output_ls1 = C1(feat_s, False)
                output_ls2 = C2(feat_s, False)
                loss_ls1 = loss_class(output_ls1, class_label_s)
                loss_ls2 = loss_class(output_ls2, class_label_s)
                loss_ls = loss_ls1 + loss_ls2
                loss_ls.backward()
                opt_g1.step()
                opt_t1.step()
                opt_c1.step()
                opt_c2.step()

                opt_g1.zero_grad()
                opt_c1.zero_grad()
                opt_c2.zero_grad()
                opt_t1.zero_grad()

                feat_s = G1(img_s)
                feat_t = G1(img_t)
                feat_s = feat_s.narrow(0, 0, batch_size)
                feat_t = feat_t.narrow(0, 0, batch_size)
                # fc1_s = T1(feat_s,False)
                # fc1_t = T1(feat_t,False)
                output_ls1 = C1(feat_s, False)
                output_ls2 = C2(feat_s, False)
                loss_ls1 = loss_class(output_ls1, class_label_s)
                loss_ls2 = loss_class(output_ls2, class_label_s)
                loss_ls = loss_ls1 + loss_ls2
                output_lt1 = C1(feat_t, False)
                output_lt2 = C2(feat_t, False)
                loss_dis = discrepancy(output_lt1, output_lt2)
                loss = loss_ls - loss_dis
                loss.backward()
                opt_c1.step() # Here should be only for predictor
                opt_c2.step()
                opt_c1.zero_grad()
                opt_c2.zero_grad()
                opt_g1.zero_grad()
                opt_t1.zero_grad()

                for k in xrange(4):
                    feat_t = G1(img_t)
                    feat_t = feat_t.narrow(0, 0, batch_size)
                    # fc1_t = T1(feat_t,False)
                    output_t1 = C1(feat_t, False)
                    output_t2 = C2(feat_t, False)
                    loss_dis = discrepancy(output_t1, output_t2)
                    loss_dis.backward()
                    opt_g1.step()
                    opt_t1.step()
                    opt_g1.zero_grad()
                    opt_t1.zero_grad()
                    opt_c1.zero_grad()
                    opt_c2.zero_grad()


                # DANN
                # feat_s = G1(img_s)
                # feat_t = G1(img_t)
                # feat_s = feat_s.narrow(0, 0, batch_size)
                # feat_t = feat_t.narrow(0, 0, batch_size)
                # output_ls1 = C1(feat_s,False)
                # output_ls2 = C2(feat_s,False)
                # loss_ls1 = loss_class(output_ls1, class_label_s)
                # loss_ls2 = loss_class(output_ls2, class_label_s)
                # D1.set_lambda(alpha)
                # output_s1 = D1(feat_s,True)
                # output_t1 = D1(feat_t,True)
                # loss_ds1 = loss_domain(output_s1, domain_label_s)
                # loss_dt1 = loss_domain(output_t1, domain_label_t)
                # loss = loss_ds1 + loss_dt1 + loss_ls1 + loss_ls2
                # loss.backward()
                # opt_g1.step()
                # opt_d1.step()
                # opt_c1.step()
                # opt_c2.step()

            elif args.alg == 3:
                opt_g1 = adjust_learning_rate(opt_g1,epoch,lr)
                opt_c1 = adjust_learning_rate(opt_c1,epoch,lr)
                opt_c2 = adjust_learning_rate(opt_c2,epoch,lr)
                opt_d1 = adjust_learning_rate(opt_d1,epoch,lr)
                opt_t1 = adjust_learning_rate(opt_t1,epoch,lr)

                # get features from G1
                feat_s = G1(img_s)
                feat_s = feat_s.narrow(0, 0, batch_size)
                # Do Linear transform
                fc1_s = T1(feat_s,False)
                #Pass features_s to C1 and C2
                output_ls1 = C1(fc1_s)
                output_ls2 = C2(fc1_s)
                loss_ls1 = loss_class(output_ls1, class_label_s)
                loss_ls2 = loss_class(output_ls2, class_label_s)
                loss_ls = loss_ls1 + loss_ls2
                loss_ls.backward()
                opt_g1.step()
                opt_c1.step()
                opt_c2.step()
                opt_t1.step()

                opt_g1.zero_grad()
                opt_c1.zero_grad()
                opt_c2.zero_grad()
                opt_d1.zero_grad()
                opt_t1.zero_grad()

                # get features from G1
                feat_s = G1(img_s)
                feat_t = G1(img_t)
                feat_s = feat_s.narrow(0, 0, batch_size)
                feat_t = feat_t.narrow(0, 0, batch_size)
                # Do Linear transform
                fc1_s = T1(feat_s,False)
                fc1_t = T1(feat_t,False)
                #Pass features_s to C1 and C2
                output_ls1 = C1(fc1_s, False)
                output_ls2 = C2(fc1_s, False)
                loss_ls1 = loss_class(output_ls1, class_label_s)
                loss_ls2 = loss_class(output_ls2, class_label_s)
                loss_ls = loss_ls1 + loss_ls2
                #Pass features_t to C1 and C2
                output_lt1 = C1(fc1_t, False)
                output_lt2 = C2(fc1_t, False)
                loss_dis = discrepancy(output_lt1, output_lt2)
                loss = loss_ls - loss_dis
                loss.backward()
                opt_c1.step() # Here should be only for predictor
                opt_c2.step()

                opt_g1.zero_grad()
                opt_t1.zero_grad()
                opt_c1.zero_grad()
                opt_c2.zero_grad()

                for k in xrange(4):#h-parameters too lager
                    feat_t = G1(img_t)
                    feat_t = feat_t.narrow(0, 0, batch_size)
                    fc1_t = T1(feat_t,False)
                    output_t1 = C1(fc1_t, False)
                    output_t2 = C2(fc1_t, False)
                    loss_dis = discrepancy(output_t1, output_t2)
                    loss_dis.backward()
                    opt_g1.step()
                    opt_t1.step()
                    opt_g1.zero_grad()
                    opt_t1.zero_grad()
                    opt_c1.zero_grad()
                    opt_c2.zero_grad()

                # get features from G1
                feat_s = G1(img_s)
                feat_t = G1(img_t)
                feat_s = feat_s.narrow(0, 0, batch_size)
                feat_t = feat_t.narrow(0, 0, batch_size)
                # Do Linear transform
                fc1_s = T1(feat_s,False)
                fc1_t = T1(feat_t,False)
                #Pass features_s to C1 and C2
                output_ls1 = C1(fc1_s)
                output_ls2 = C2(fc1_s)
                loss_ls1 = loss_class(output_ls1, class_label_s)
                loss_ls2 = loss_class(output_ls2, class_label_s)
                loss_ls = loss_ls1 + loss_ls2
                #Pass features to Domain discriminator and use GRL
                D1.set_lambda(alpha)
                output_ds = D1(fc1_s,True)
                output_dt = D1(fc1_t,True)
                loss_ds = loss_domain(output_ds, domain_label_s)
                loss_dt = loss_domain(output_dt, domain_label_t)
                loss_gif = loss_ds + loss_dt + loss_ls
                loss_gif.backward()
                opt_g1.step()
                opt_c1.step()
                opt_c2.step()
                opt_d1.step()
                opt_t1.step()


                # get features from G2
                feat_s = G2(img_s)
                feat_s = feat_s.narrow(0, 0, batch_size)
                # Do Linear transform
                fc2_s = T2(feat_s,False)
                #Pass features_s to C3 and C4
                output_ls1 = C3(fc2_s)
                output_ls2 = C4(fc2_s)
                loss_ls1 = loss_class(output_ls1, class_label_s)
                loss_ls2 = loss_class(output_ls2, class_label_s)
                loss_ls = loss_ls1 + loss_ls2
                loss_ls.backward()
                opt_g2.step()
                opt_c3.step()
                opt_c4.step()
                opt_t2.step()

                opt_g2.zero_grad()
                opt_c3.zero_grad()
                opt_c4.zero_grad()
                opt_d2.zero_grad()
                opt_t2.zero_grad()

                feat_s = G2(img_s)
                feat_t = G2(img_t)
                feat_s = feat_s.narrow(0, 0, batch_size)
                feat_t = feat_t.narrow(0, 0, batch_size)
                fc2_s = T2(feat_s,False)
                fc2_t = T2(feat_t,False)
                output_ls3 = C3(fc2_s, False)
                output_ls4 = C4(fc2_s, False)
                loss_ls3 = loss_class(output_ls3, class_label_s)
                loss_ls4 = loss_class(output_ls4, class_label_s)
                loss_ls = loss_ls3 + loss_ls4
                output_lt1 = C3(fc2_t, False)
                output_lt2 = C4(fc2_t, False)
                loss_dis = discrepancy(output_lt1, output_lt2)
                loss = loss_ls - loss_dis
                loss.backward()
                opt_c3.step() # Here should be only for predictor
                opt_c4.step()

                opt_g2.zero_grad()
                opt_t2.zero_grad()
                opt_c3.zero_grad()
                opt_c4.zero_grad()

                for k in xrange(4):
                    feat_t = G2(img_t)
                    feat_t = feat_t.narrow(0, 0, batch_size)
                    fc2_t = T2(feat_t,False)
                    output_t1 = C3(fc2_t, False)
                    output_t2 = C4(fc2_t, False)
                    loss_dis = discrepancy(output_t1, output_t2)
                    loss_dis.backward()
                    opt_g2.step()
                    opt_t2.step()
                    opt_g2.zero_grad()
                    opt_t2.zero_grad()
                    opt_c3.zero_grad()
                    opt_c4.zero_grad()

                # Second Part for G2 ngif
                feat_s = G2(img_s)
                feat_t = G2(img_t)
                feat_s = feat_s.narrow(0, 0, batch_size)
                feat_t = feat_t.narrow(0, 0, batch_size)
                fc2_s = T2(feat_s,False)
                fc2_t = T2(feat_t,False)
                output_ds = D2(fc2_s, False)
                output_dt = D2(fc2_t, False)
                loss_ds = loss_domain(output_ds, domain_label_s)
                loss_dt = loss_domain(output_dt, domain_label_t)
                output_ls3 = C3(fc2_s,False)
                output_ls4 = C4(fc2_s,False)
                loss_ls3 = loss_class(output_ls3, class_label_s)
                loss_ls4 = loss_class(output_ls4, class_label_s)
                loss_ls = loss_ls3 + loss_ls4
                loss_ngif = loss_ls + loss_ds + loss_dt
                loss_ngif.backward()
                opt_g2.step()
                opt_d2.step()
                opt_t2.step()
                opt_c3.step()
                opt_c4.step()

                opt_g2.zero_grad()
                opt_d2.zero_grad()
                opt_t2.zero_grad()
                opt_c3.zero_grad()
                opt_c4.zero_grad()


                feat_s1 = G1(img_s)
                feat_t1 = G1(img_t)
                feat_s2 = G2(img_s)
                feat_t2 = G2(img_t)
                feat_s1 = feat_s1.narrow(0, 0, batch_size)
                feat_t1 = feat_t1.narrow(0, 0, batch_size)
                feat_s2 = feat_s2.narrow(0, 0, batch_size)
                feat_t2 = feat_t2.narrow(0, 0, batch_size)
                fc1_s= T1(feat_s1,False)
                fc1_t= T1(feat_t1,False)
                fc2_s= T2(feat_s2,False)
                fc2_t= T2(feat_t2,False)
                output_ls1 = C1(fc1_s,False)
                output_lt1 = C1(fc1_t,False)
                output_ls2 = C3(fc2_s,False)
                output_lt2 = C3(fc2_t,False)
                loss_diss = discrepancy(output_ls1, output_ls2)
                loss_dist = discrepancy(output_lt1, output_lt2)
                T1.set_lambda(alpha)
                T2.set_lambda(alpha)
                fc1_s= T1(feat_s1,True)
                fc1_t= T1(feat_t1,True)
                fc2_s= T2(feat_s2,True)
                fc2_t= T2(feat_t2,True)
                dis_sf = discrepancy(fc1_s, fc2_s)
                dis_tf = discrepancy(fc1_t, fc2_t)
                loss_dis_f = dis_sf + dis_tf + loss_dist + loss_diss
                loss_dis_f.backward()
                opt_g1.step()
                opt_g2.step()
                opt_t1.step()
                opt_t2.step()

            elif args.alg == 4:
                # opt_g1 = adjust_learning_rate(opt_g1,count,0.1)
                # opt_c1 = adjust_learning_rate(opt_c1,count,1)
                #SOURCE ONLY
                feat_s = G1(img_s)
                # feat_t = G1(img_t)
                feat_s = feat_s.narrow(0, 0, batch_size)
                # feat_t = feat_t.narrow(0, 0, batch_size)
                output_ls1 = C1(feat_s,False)
                loss_ls1 = loss_class(output_ls1, class_label_s)
                loss_ls1.backward()
                opt_g1.step()
                opt_c1.step()
            else:
                opt_g1 = adjust_learning_rate(opt_g1,epoch,lr)
                opt_c1 = adjust_learning_rate(opt_c1,epoch,lr)
                opt_d1 = adjust_learning_rate(opt_d1,epoch,lr)
                # DANN
                feat_s = G1(img_s)
                feat_t = G1(img_t)
                feat_s = feat_s.narrow(0, 0, batch_size)
                feat_t = feat_t.narrow(0, 0, batch_size)
                output_ls1 = C1(feat_s,False)
                loss_ls1 = loss_class(output_ls1, class_label_s)
                D1.set_lambda(alpha)
                output_s1 = D1(feat_s,True)
                output_t1 = D1(feat_t,True)
                loss_ds1 = loss_domain(output_s1, domain_label_s)
                loss_dt1 = loss_domain(output_t1, domain_label_t)
                loss = loss_ds1 + loss_dt1 + loss_ls1
                loss.backward()
                opt_g1.step()
                opt_d1.step()
                opt_c1.step()
                avg_loss += loss.cpu().data.numpy()
                # print 'Lds1: %f' % loss_ds1.cpu().data.numpy(), 'Ldt1: %f' % loss_dt1.cpu().data.numpy(), 'Loss: %f' % loss.cpu().data.numpy(),'Avg Loss: %f' % (avg_loss/count)
                print 'Loss: %f' % loss.cpu().data.numpy(), 'Avg Loss: %f' % (avg_loss/count)
            i += 1
            count += 1
###############################################################################

        torch.save(G1.state_dict(), '{0}/{1}/{2}_{3}_model_epoch_G1_{4}.pth'.format(model_root,args.alg,source,target,epoch))
        # torch.save(G.state_dict(),'{0}/{1}/state_dict_G1_{2}.pth'.format(model_root,args.alg,epoch))
        # torch.save(G2, '{0}/{1}/{2}_{3}_model_epoch_G2_{4}.pth'.format(model_root,args.alg,source,target,epoch))
        # torch.save(G.state_dict(),'{0}/{1}/state_dict_G2_{2}.pth'.format(model_root,args.alg,epoch))
        torch.save(C1.state_dict(), '{0}/{1}/{2}_{3}_model_epoch_C1_{4}.pth'.format(model_root,args.alg,source,target,epoch))
        # torch.save(C1.state_dict(),'{0}/{1}/state_dict_C1_{2}.pth'.format(model_root,args.alg,epoch))
        # torch.save(C2, '{0}/{1}/{2}_{3}_model_epoch_C2_{4}.pth'.format(model_root,args.alg,source,target,epoch))
        # torch.save(C2.state_dict(),'{0}/{1}/state_dict_C2_{2}.pth'.format(model_root,args.alg,epoch))
        # torch.save(C3, '{0}/{1}/{2}_{3}_model_epoch_C3_{4}.pth'.format(model_root,args.alg,source,target,epoch))
        # torch.save(C3.state_dict(),'{0}/{1}/state_dict_C3_{2}.pth'.format(model_root,args.alg,epoch))
        torch.save(T1.state_dict(), '{0}/{1}/{2}_{3}_model_epoch_T1_{4}.pth'.format(model_root,args.alg,source,target,epoch))
        # torch.save(C2.state_dict(),'{0}/{1}/state_dict_C2_{2}.pth'.format(model_root,args.alg,epoch))
        # torch.save(T2, '{0}/{1}/{2}_{3}_model_epoch_T2_{4}.pth'.format(model_root,args.alg,source,target,epoch))
        # torch.save(C2.state_dict(),'{0}/{1}/state_dict_C2_{2}.pth'.format(model_root,args.alg,epoch))
        test(source, target, epoch, args.alg, GPU)

train(n_epoch)
print('done')
