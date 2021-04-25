import torch.nn as nn
from functions import ReverseLayerF
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models

class Domain_Classifier_SM(nn.Module):
    def __init__(self):
        super(Domain_Classifier_SM, self).__init__()
        self.lambd = 0
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(8192, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 100))
        self.domain_classifier.add_module('d_bn2', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu2', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc3', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = ReverseLayerF.apply(x, self.lambd)
        result = self.domain_classifier(x)
        return result

class Feature_SM(nn.Module):
    def __init__(self):
        super(Feature_SM, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)

    def forward(self, x):
        x = x.expand(x.data.shape[0], 3, 32, 32)
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=3, padding=1)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=3, padding=1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), 8192)
        return x

class Predictor_SM(nn.Module):
    def __init__(self, prob=0.5):
        super(Predictor_SM, self).__init__()
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 10)
        self.bn_fc3 = nn.BatchNorm1d(10)
        self.prob = prob
        self.lambd = 0

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = ReverseLayerF.apply(x, self.lambd)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = self.fc3(x)
        return x

class Transfrom_SM(nn.Module):
    def __init__(self, prob=0.5):
        super(Transfrom_SM, self).__init__()
        self.prob = prob
        self.lambd = 0
        self.fc1 = nn.Linear(8192, 8192)
        self.bn1_fc = nn.BatchNorm1d(8192)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = ReverseLayerF.apply(x, self.lambd)
        # x = F.relu(self.bn1_fc(self.fc1(x)))
        x =  self.fc1(x)
        return x

class Domain_Classifier_UM(nn.Module):
    def __init__(self):
        super(Domain_Classifier_UM, self).__init__()
        self.lambd = 0
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(48*4*4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = ReverseLayerF.apply(x, self.lambd)
        result = self.domain_classifier(x)
        return result

class Feature_UM(nn.Module):
    def __init__(self):
        super(Feature_UM, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(48)

    def forward(self, x):
        x = torch.mean(x,1).view(x.size()[0],1,x.size()[2],x.size()[3])
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=2, dilation=(1, 1))
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=2, dilation=(1, 1))
        x = x.view(-1, 48*4*4)
        return x

class Predictor_UM(nn.Module):
    def __init__(self, prob=0.5):
        super(Predictor_UM, self).__init__()
        self.lambd = 0
        self.fc1 = nn.Linear(48*4*4, 100)
        self.bn1_fc = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 100)
        self.bn2_fc = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 10)
        self.bn_fc3 = nn.BatchNorm1d(10)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = ReverseLayerF.apply(x, self.lambd)
        x = F.dropout(x, training=self.training, p=self.prob)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training, p=self.prob)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = F.dropout(x, training=self.training, p=self.prob)
        x = self.fc3(x)
        return x

class Transfrom_UM(nn.Module):
    def __init__(self, prob=0.5):
        super(Transfrom_UM, self).__init__()
        self.prob = prob
        self.lambd = 0
        self.fc1 = nn.Linear(48*4*4, 48*4*4)
        self.bn1_fc = nn.BatchNorm1d(48*4*4)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = ReverseLayerF.apply(x, self.lambd)
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        # x = self.fc1(x)
        return x

class Domain_Classifier_MU(nn.Module):
    def __init__(self):
        super(Domain_Classifier_MU, self).__init__()
        self.lambd = 0
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(48*4*4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = ReverseLayerF.apply(x, self.lambd)
        result = self.domain_classifier(x)
        return result

class Feature_MU(nn.Module):
    def __init__(self):
        super(Feature_MU, self).__init__()
        # self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)
        # self.bn1 = nn.BatchNorm2d(20)
        # self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1)
        # self.bn2 = nn.BatchNorm2d(50)
        self.drop = nn.Dropout2d()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(48)

    def forward(self, x):

        x = torch.mean(x,1).view(x.size()[0],1,x.size()[2],x.size()[3])
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=2, dilation=(1, 1))
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=2, dilation=(1, 1))

        x = x.view(-1, 48*4*4)
        return x

class Predictor_MU(nn.Module):
    def __init__(self, prob=0.5):
        super(Predictor_MU, self).__init__()
        self.lambd = 0
        # self.fc1 = nn.Linear(50*4*4, 500)
        # self.fc2 = nn.Linear(500, 10)
        # self.prob = prob
        self.fc1 = nn.Linear(48*4*4, 100)
        self.bn1_fc = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 100)
        self.bn2_fc = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 10)
        self.bn_fc3 = nn.BatchNorm1d(10)
        self.prob = prob


    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = ReverseLayerF.apply(x, self.lambd)
        x = F.dropout(x, training=self.training, p=self.prob)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training, p=self.prob)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = F.dropout(x, training=self.training, p=self.prob)
        x = self.fc3(x)
        return x

class Transfrom_MU(nn.Module):
    def __init__(self, prob=0.5):
        super(Transfrom_MU, self).__init__()
        self.prob = prob
        self.lambd = 0
        self.fc1 = nn.Linear(48*4*4, 48*4*4)
        self.bn1_fc = nn.BatchNorm1d(48*4*4)
    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = ReverseLayerF.apply(x, self.lambd)
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        # x = self.fc1(x)
        return x

class Domain_Classifier_MM(nn.Module):
    def __init__(self):
        super(Domain_Classifier_MM, self).__init__()
        self.lambd = 0
        self.domain_classifier = nn.Sequential()
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(8192, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 100))
        self.domain_classifier.add_module('d_bn2', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu2', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc3', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = ReverseLayerF.apply(x, self.lambd)
        result = self.domain_classifier(x)
        return result

class Feature_MM(nn.Module):
    def __init__(self):
        super(Feature_MM, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)

    def forward(self, x):
        x = x.expand(x.data.shape[0], 3, 32, 32)
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=3, padding=1)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=3, padding=1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), 8192)
        return x

class Predictor_MM(nn.Module):
    def __init__(self, prob=0.5):
        super(Predictor_MM, self).__init__()
        self.lambd = 0
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 10)
        self.bn_fc3 = nn.BatchNorm1d(10)
        self.drop = nn.Dropout2d()
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = ReverseLayerF.apply(x, self.lambd)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = self.fc3(x)

        return x

class Transfrom_MM(nn.Module):
    def __init__(self, prob=0.5):
        super(Transfrom_MM, self).__init__()
        self.prob = prob
        self.lambd = 0
        self.fc1 = nn.Linear(8192, 8192)
        self.bn1_fc = nn.BatchNorm1d(8192)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = ReverseLayerF.apply(x, self.lambd)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        # x = self.fc1(x)
        return x

class Feature_SG(nn.Module):
    def __init__(self):
        super(Feature_SG, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 144, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(144)
        self.conv3 = nn.Conv2d(144, 256, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = x.expand(x.data.shape[0], 3, 40, 40)
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=2, padding=0)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=2, padding=0)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), stride=2, kernel_size=2, padding=0)
        x = x.view(x.size(0), 6400)
        return x

class Predictor_SG(nn.Module):
    def __init__(self):
        super(Predictor_SG, self).__init__()
        self.fc2 = nn.Linear(6400, 512)
        self.bn2_fc = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 43)
        self.bn_fc3 = nn.BatchNorm1d(43)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = ReverseLayerF.apply(x, self.lambd)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return x

class Transfrom_SG(nn.Module):
    def __init__(self):
        super(Transfrom_SG, self).__init__()
        self.fc1 = nn.Linear(6400, 6400)
        self.bn1_fc = nn.BatchNorm1d(6400)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = ReverseLayerF.apply(x, self.lambd)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        # x = self.fc1(x)
        return x

class Domain_Classifier_SG(nn.Module):
    def __init__(self):
        super(Domain_Classifier_SG, self).__init__()
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(6400, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 100))
        self.domain_classifier.add_module('d_bn2', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu2', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc3', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = ReverseLayerF.apply(x, self.lambd)
        result = self.domain_classifier(x)
        return result

class Domain_Classifier_RESNET(nn.Module):
    def __init__(self):
        super(Domain_Classifier_RESNET, self).__init__()
        self.lambd = 0
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(512, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        # self.domain_classifier.add_module('d_drop1', nn.Dropout2d())
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 100))
        self.domain_classifier.add_module('d_bn2', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu2', nn.ReLU(True))
        # self.domain_classifier.add_module('d_drop2', nn.Dropout2d())
        self.domain_classifier.add_module('d_fc3', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        # x = x.view(x.size(0), -1)
        if reverse:
            x = ReverseLayerF.apply(x, self.lambd)
        result = self.domain_classifier(x)
        return result

class Predictor_RESNET(nn.Module):
    def __init__(self, prob=0.5):
        super(Predictor_RESNET, self).__init__()
        self.fc1 = nn.Linear(512, 12)
        # self.fc1 = nn.Linear(512, 256)
        # self.bn1_fc = nn.BatchNorm1d(256)
        # self.fc2 = nn.Linear(256, 256)
        # self.bn2_fc = nn.BatchNorm1d(256)
        # self.fc3 = nn.Linear(256, 12)
        # self.bn3_fc = nn.BatchNorm1d(31)
        # self.drop1 = nn.Dropout2d()
        # self.drop2 = nn.Dropout2d()
        # self.avgpool = nn.AvgPool2d(7,stride=1)
        self.prob = prob
        self.lambd = 0

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        # x = x.view(x.size(0), -1)
        if reverse:
            x = ReverseLayerF.apply(x, self.lambd)
        # x = F.relu(self.bn1_fc(self.fc1(x)))
        # x = F.dropout(x, training=self.training)
        # x = F.relu(self.bn2_fc(self.fc2(x)))
        # x = self.fc3(x)
        x = self.fc1(x)
        return x

class Feature_RESNET(nn.Module):
    def __init__(self):
        super(Feature_RESNET, self).__init__()
        self.restored = False
        model_resnet101 = models.resnet18(pretrained=True)
        self.features = nn.Sequential(
            model_resnet101.conv1,
            model_resnet101.bn1,
            model_resnet101.relu,
            model_resnet101.maxpool,
            model_resnet101.layer1,
            model_resnet101.layer2,
            model_resnet101.layer3,
            model_resnet101.layer4,
            model_resnet101.avgpool,
        )
        # self.__in_features = model_resnet50.fc.in_features
        # print(self.__in_features)
        # self.fc = nn.Linear(self.__in_features, 31)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x

class Transfrom_RESNET(nn.Module):
    def __init__(self, prob=0.5):
        super(Transfrom_RESNET, self).__init__()
        self.prob = prob
        self.lambd = 0
        self.fc1 = nn.Linear(512, 512)
        self.bn1_fc = nn.BatchNorm1d(512)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        # x = x.view(x.size(0), -1)
        if reverse:
            x = ReverseLayerF.apply(x, self.lambd)
        # x = F.relu(self.bn1_fc(self.fc1(x)))
        x = self.fc1(x)
        return x
