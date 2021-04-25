import os
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms, models
from dataset.data_loader import GetLoader
from dataset.datasets import Dataset
from dataset.usps import load_usps
from dataset.mnist import load_mnist
from torchvision import datasets
from models.mod import Feature_SM, Predictor_SM, Domain_Classifier_SM, Transfrom_SM
from models.mod import Feature_UM, Predictor_UM, Domain_Classifier_UM, Transfrom_UM
from models.mod import Feature_MM, Predictor_MM, Domain_Classifier_MM, Transfrom_MM
from models.mod import Feature_MU, Predictor_MU, Domain_Classifier_MU, Transfrom_MU
from models.mod import Feature_SG, Predictor_SG, Domain_Classifier_SG, Transfrom_SG
from models.mod import Feature_RESNET, Predictor_RESNET, Domain_Classifier_RESNET, Transfrom_RESNET
from models.usps import *
from numpy import asarray
from numpy import savetxt
import numpy as np

def test(source, target, epoch, alg, GPU):

    model_root = os.path.join('..', 'models/checkpoints')
    image_root = os.path.join('../', 'dataset/mnist_m')
    office31_image_root  = os.path.join('../', 'dataset', 'office_31')
    office31_image_root  = os.path.join('../', 'dataset', 'office_31')
    visda2017_image_root = os.path.join('../', 'dataset', 'VisDA2017')
    gtsrb_image_root     = os.path.join('../', 'dataset', 'GTSRB')
    synth_image_root     = os.path.join('../', 'dataset', 'SYN')

    cuda = True
    cudnn.benchmark = True
    batch_size = 64
    image_size = 0
    alpha = 0
    # GPU = GPU
    """load data"""
    if target == 'MNIST' and source == 'SVHN':
        image_size = 32
        G1 = Feature_SM()
        G2 = Feature_SM()
        T1 = Transfrom_SM()
        T2 = Transfrom_SM()
        C1 = Predictor_SM()
        C2 = Predictor_SM()
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
            split='test',
            transform=img_transform_target,
            download=False)
        dataloader_source = torch.utils.data.DataLoader(
            dataset=dataset_source,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8)
        dataset_target = datasets.MNIST(
            root='../dataset',
            train=False,
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
        _, _, img_test, label_test = load_usps()
        dataset_source = Dataset(img_test,label_test,img_transform_target)
        dataloader_source = torch.utils.data.DataLoader(
            dataset=dataset_source,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8)
        _, _, img_test, label_test = load_mnist(False,True,'yes')
        dataset_target = Dataset(img_test,label_test,img_transform_source)
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
        T1 = Transfrom_MM()
        T2 = Transfrom_MM()
        D1 = Domain_Classifier_MM()
        D2 = Domain_Classifier_MM()
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
        dataset_source = datasets.MNIST(
            root='../dataset',
            train=False,
            transform=img_transform_source,
            download=True
        )
        dataloader_source = torch.utils.data.DataLoader(
            dataset=dataset_source,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8)
        test_list = os.path.join(image_root, 'mnist_m_test_labels.txt')
        dataset_target = GetLoader(
            data_root=os.path.join(image_root, 'mnist_m_test'),
            data_list=test_list,
            transform=img_transform_target)
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
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])
        dataset_source = datasets.MNIST(
            root='../dataset',
            train=False,
            transform=img_transform_source,
            download=True
        )
        dataloader_source = torch.utils.data.DataLoader(
            dataset=dataset_source,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8)
        _, _, img_test, label_test = load_usps()
        dataset_target = Dataset(img_test,label_test,img_transform_target)
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
            shuffle=False,
            num_workers=8)

        train_list = os.path.join(gtsrb_image_root, 'test.txt')
        dataset_target = GetLoader(
            data_root=os.path.join(gtsrb_image_root, 'test'),
            data_list=train_list,
            transform=img_transform_target
        )
        dataloader_target = torch.utils.data.DataLoader(
            dataset=dataset_target,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8)

    elif target == 'W' and source == 'A':
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
            transforms.Resize(227),
            # transforms.RandomResizedCrop(227),
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
            shuffle=False,
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
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        source_list = os.path.join(office31_image_root, 'dslr_test.txt')
        dataset_source = GetLoader(
            data_root=office31_image_root,
            data_list=source_list,
            transform=img_transform)
        dataloader_source = torch.utils.data.DataLoader(
            dataset=dataset_source,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8)
        target_list = os.path.join(office31_image_root, 'wabcam_test.txt')
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
            transforms.RandomResizedCrop(227),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        source_list = os.path.join(office31_image_root, 'webcam_test.txt')
        dataset_source = GetLoader(
            data_root=office31_image_root,
            data_list=source_list,
            transform=img_transform)
        dataloader_source = torch.utils.data.DataLoader(
            dataset=dataset_source,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8)
        target_list = os.path.join(office31_image_root, 'dslr_test.txt')
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
            transforms.RandomResizedCrop(227),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        source_list = os.path.join(office31_image_root, 'amazon_test.txt')
        dataset_source = GetLoader(
            data_root=office31_image_root,
            data_list=source_list,
            transform=img_transform)
        dataloader_source = torch.utils.data.DataLoader(
            dataset=dataset_source,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8)
        target_list = os.path.join(office31_image_root, 'dslr_test.txt')
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
            transforms.RandomResizedCrop(227),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        source_list = os.path.join(office31_image_root, 'dslr_test.txt')
        dataset_source = GetLoader(
            data_root=office31_image_root,
            data_list=source_list,
            transform=img_transform)
        dataloader_source = torch.utils.data.DataLoader(
            dataset=dataset_source,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8)
        target_list = os.path.join(office31_image_root, 'amazon_test.txt')
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
        G2 = Feature_RESNET()
        C1 = Predictor_RESNET()
        C2 = Predictor_RESNET()
        C3 = Predictor_RESNET()
        C4 = Predictor_RESNET()
        T1 = Transfrom_RESNET()
        T2 = Transfrom_RESNET()
        D1 = Domain_Classifier_RESNET()
        D2 = Domain_Classifier_RESNET()
        img_transform = transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.RandomHorizontalFlip(),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            # transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
        ])
        source_list = os.path.join(visda2017_image_root, 'train' ,'image_list.txt')
        dataset_source = GetLoader(
            data_root=os.path.join(visda2017_image_root,'train'),
            data_list=source_list,
            transform=img_transform)
        dataloader_source = torch.utils.data.DataLoader(
            dataset=dataset_source,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8)
        target_list = os.path.join(visda2017_image_root, 'validation','image_list.txt')
        dataset_target = GetLoader(
            data_root=os.path.join(visda2017_image_root,'validation'),
            data_list=target_list,
            transform=img_transform)
        dataloader_target = torch.utils.data.DataLoader(
            dataset=dataset_target,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8)


    """ training """
    if epoch == -1:
        G1.load_state_dict(torch.load(os.path.join('{0}/{1}/eval/{2}_{3}_model_epoch_G1_{4}.pth'.format(model_root,alg,source,target,0))))
        C1.load_state_dict(torch.load(os.path.join('{0}/{1}/eval/{2}_{3}_model_epoch_C1_{4}.pth'.format(model_root,alg,source,target,0))))
        T1.load_state_dict(torch.load(os.path.join('{0}/{1}/eval/{2}_{3}_model_epoch_T1_{4}.pth'.format(model_root,alg,source,target,0))))
        # G2 = torch.load(os.path.join('{0}/{1}/eval/{2}_{3}_model_epoch_G2_{4}.pth'.format(model_root,alg,source,target,0)))
        # C2 = torch.load(os.path.join('{0}/{1}/eval/{2}_{3}_model_epoch_C2_{4}.pth'.format(model_root,alg,source,target,0)))
        # T2 = torch.load(os.path.join('{0}/{1}/eval/{2}_{3}_model_epoch_T2_{4}.pth'.format(model_root,alg,source,target,0)))

        G1.eval()
        C1.eval()
        T1.eval()
        # G2.eval()
        # C2.eval()
        # T2.eval()

        G1.cuda(device = GPU)
        C1.cuda(device = GPU)
        T1.cuda(device = GPU)
        # G2.cuda(device = GPU)
        # C2.cuda(device = GPU)
        # T2.cuda(device = GPU)

        len_dataloader_s = len(dataloader_source)
        data_source_iter = iter(dataloader_source)
        len_dataloader_t = len(dataloader_target)
        data_target_iter = iter(dataloader_target)
        print(len_dataloader_s, len_dataloader_t)
        i = 0
        n_total = 0
        n_correct = 0

        while i < len_dataloader_t/13:
            # test model using target data
            data_target = data_target_iter.next()
            t_img, t_label = data_target

            batch_size = len(t_label)

            input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
            class_label = torch.LongTensor(batch_size)

            if cuda:
                t_img = t_img.cuda(device = GPU)
                t_label = t_label.cuda(device = GPU)
                input_img = input_img.cuda(device = GPU)
                class_label = class_label.cuda(device = GPU)

                input_img.resize_as_(t_img).copy_(t_img)
                class_label.resize_as_(t_label).copy_(t_label)

            t_feat = G1(t_img)
            if alg == 4 or alg == 5 or alg == 2:
                class_output_t = C1(t_feat,False)
            else:
                t = T1(t_feat)
		# print(t.cpu().data.numpy())
		# savetxt('data'+str(i)+'.txt', np.around(t.cpu().data.numpy(), decimals=5), delimiter=' ', newline='\n')
		#f = open('data.txt','a')
		#f.write('\n')
		#f.close()
                class_output_t = C1(t,False)

            pred = class_output_t.data.max(1, keepdim=True)[1]
            n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()

            n_total += batch_size
            i += 1
        accu = n_correct.data.numpy() * 1.0 / n_total
        print 'epoch: %d, %s dataset: %f' % (epoch, target, accu)

    elif epoch == -2:
        batch_size = 1
        G1.load_state_dict(torch.load(os.path.join('{0}/VisDA/{1}_{2}_model_epoch_G1_{3}.pth'.format(model_root,source,target,0))))
        C1.load_state_dict(torch.load(os.path.join('{0}/VisDA/{1}_{2}_model_epoch_C1_{3}.pth'.format(model_root,source,target,0))))
        T1.load_state_dict(torch.load(os.path.join('{0}/VisDA/{1}_{2}_model_epoch_T1_{3}.pth'.format(model_root,source,target,0))))
        # G2 = torch.load(os.path.join('{0}/{1}/eval/{2}_{3}_model_epoch_G2_{4}.pth'.format(model_root,alg,source,target,0)))
        # C2 = torch.load(os.path.join('{0}/{1}/eval/{2}_{3}_model_epoch_C2_{4}.pth'.format(model_root,alg,source,target,0)))
        # T2 = torch.load(os.path.join('{0}/{1}/eval/{2}_{3}_model_epoch_T2_{4}.pth'.format(model_root,alg,source,target,0)))

        G1.eval()
        C1.eval()
        T1.eval()
        # G2.eval()
        # C2.eval()
        # T2.eval()

        G1.cuda(device = GPU)
        C1.cuda(device = GPU)
        T1.cuda(device = GPU)
        # G2.cuda(device = GPU)
        # C2.cuda(device = GPU)
        # T2.cuda(device = GPU)

        len_dataloader_s = len(dataloader_source)
        data_source_iter = iter(dataloader_source)
        len_dataloader_t = len(dataloader_target)
        data_target_iter = iter(dataloader_target)
        print(len_dataloader_s, len_dataloader_t)
        i = 0
        n_total = 0
        n_correct = 0
        arr1 = [0,0,0,0,0,0,0,0,0,0,0,0]
        arr2 = [0,0,0,0,0,0,0,0,0,0,0,0]
        while i < len_dataloader_t:
            # test model using target data
            data_target = data_target_iter.next()
            t_img, t_label = data_target

            batch_size = len(t_label)

            input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
            class_label = torch.LongTensor(batch_size)

            if cuda:
                t_img = t_img.cuda(device = GPU)
                t_label = t_label.cuda(device = GPU)
                input_img = input_img.cuda(device = GPU)
                class_label = class_label.cuda(device = GPU)

                input_img.resize_as_(t_img).copy_(t_img)
                class_label.resize_as_(t_label).copy_(t_label)

            t_feat = G1(t_img)
            if alg == 4 or alg == 5 or alg == 2:
                class_output_t = C1(t_feat,False)
            else:
                t = T1(t_feat)
		# print(t.cpu().data.numpy())
		# savetxt('data'+str(i)+'.txt', np.around(t.cpu().data.numpy(), decimals=5), delimiter=' ', newline='\n')
		#f = open('data.txt','a')
		#f.write('\n')
		#f.close()
                class_output_t = C1(t,False)

            pred = class_output_t.data.max(1, keepdim=True)[1]
            if list(pred.cpu().data.numpy())[0][0] == list(class_label.data.view_as(pred).cpu().data.numpy())[0][0]:
                arr1[list(pred.cpu().data.numpy())[0][0]] += 1
            arr2[list(class_label.data.view_as(pred).cpu().data.numpy())[0][0]] += 1
            n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()

            n_total += batch_size
            i += 1
        accu = n_correct.data.numpy() * 1.0 / n_total
        print 'epoch: %d, %s dataset: %f' % (epoch, target, accu)
        for index in range(12):
            print(float(arr1[index])/float(arr2[index]))
    else:
        G1.load_state_dict(torch.load(os.path.join('{0}/{1}/{2}_{3}_model_epoch_G1_{4}.pth'.format(model_root,alg,source,target,epoch))))
        # G2.load_state_dict(torch.load(os.path.join('{0}/{1}/{2}_{3}_model_epoch_G2_{4}.pth'.format(model_root,alg,source,target,epoch)))
        C1.load_state_dict(torch.load(os.path.join('{0}/{1}/{2}_{3}_model_epoch_C1_{4}.pth'.format(model_root,alg,source,target,epoch))))
        # C2.load_state_dict(torch.load(os.path.join('{0}/{1}/{2}_{3}_model_epoch_C2_{4}.pth'.format(model_root,alg,source,target,epoch)))
        T1.load_state_dict(torch.load(os.path.join('{0}/{1}/{2}_{3}_model_epoch_T1_{4}.pth'.format(model_root,alg,source,target,epoch))))
        # T2.load_state_dict(torch.load(os.path.join('{0}/{1}/{2}_{3}_model_epoch_T2_{4}.pth'.format(model_root,alg,source,target,epoch)))

        G1.eval()
        # G2.eval()
        C1.eval()
        # C2.eval()
        T1.eval()
        # T2.eval()

        G1.cuda(device = GPU)
        # G2.cuda(device = GPU)
        C1.cuda(device = GPU)
        # C2.cuda(device = GPU)
        T1.cuda(device = GPU)
        # T2.cuda(device = GPU)

        len_dataloader_s = len(dataloader_source)
        data_source_iter = iter(dataloader_source)
        len_dataloader_t = len(dataloader_target)
        data_target_iter = iter(dataloader_target)
        print(len_dataloader_s, len_dataloader_t)
        i = 0
        n_total = 0
        n_correct = 0
        while i < len_dataloader_s:
            # test model using target data
            data_target = data_source_iter.next()
            t_img, t_label = data_target

            batch_size = len(t_label)

            input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
            class_label = torch.LongTensor(batch_size)

            if cuda:
                t_img = t_img.cuda(device = GPU)
                t_label = t_label.cuda(device = GPU)
                input_img = input_img.cuda(device = GPU)
                class_label = class_label.cuda(device = GPU)

            input_img.resize_as_(t_img).copy_(t_img)
            class_label.resize_as_(t_label).copy_(t_label)

            t_feat = G1(t_img)
            if alg == 2 or alg == 4 or alg == 5:
                class_output_t = C1(t_feat,False)
            else:
                t = T1(t_feat)
                class_output_t = C1(t,False)

            pred = class_output_t.data.max(1, keepdim=True)[1]
            n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()

            n_total += batch_size
            i += 1
        accu = n_correct.data.numpy() * 1.0 / n_total
        print 'epoch: %d, %s dataset: %f' % (epoch, source, accu)

        i = 0
        n_total = 0
        n_correct = 0
        VisDA2017_correct = [0,0,0,0,0,0,0,0,0,0,0]
        while i < len_dataloader_t:
            # test model using target data
            data_target = data_target_iter.next()
            t_img, t_label = data_target

            batch_size = len(t_label)

            input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
            class_label = torch.LongTensor(batch_size)

            if cuda:
                t_img = t_img.cuda(device = GPU)
                t_label = t_label.cuda(device = GPU)
                input_img = input_img.cuda(device = GPU)
                class_label = class_label.cuda(device = GPU)

            input_img.resize_as_(t_img).copy_(t_img)
            class_label.resize_as_(t_label).copy_(t_label)

            t_feat = G1(t_img)
            if alg == 2 or alg == 4 or alg == 5:
                class_output_t = C1(t_feat,False)
            else:
                t = T1(t_feat)
                class_output_t = C1(t,False)

            pred = class_output_t.data.max(1, keepdim=True)[1]
            n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()

            n_total += batch_size
            i += 1
        accu = n_correct.data.numpy() * 1.0 / n_total
        print 'epoch: %d, %s dataset: %f' % (epoch, target, accu)
                    # total = [3646,3475,4690,10401,4691,2075,5796,4000,4549,2281,4236,5548]
                    # if source == 'A':
                    #     for x in xrange(12):
                    #         accu = VisDA2017_correct[x].data.numpy() * 1.0 / total[x]
                    #         print 'epoch: %d, %f' % (x, target, accu)
