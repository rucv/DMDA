import torch.nn as nn
from functions import ReverseLayerF
import numpy as np


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax())


        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

        # self.feature_normalize = nn.Sequential()
        # self.feature_normalize.add_module('bn1',nn.BatchNorm2d(50 * 4 * 4))

    def forward(self, input_data_s, input_data_t, mix_ratio, alpha):
        clip_thr = 0.3
        length = min(len(input_data_s), len(input_data_t))
        if length != 0:
            input_data_s = input_data_s.expand(input_data_s.data.shape[0], 3, 28, 28)
            feature_s = self.feature(input_data_s)
            feature_s = feature_s.view(-1, 50 * 4 * 4)

            input_data_t = input_data_t.expand(input_data_t.data.shape[0], 3, 28, 28)
            feature_t = self.feature(input_data_t)
            feature_t = feature_t.view(-1, 50 * 4 * 4)

            feature_s = feature_s.narrow(0, 0, length)
            feature_t = feature_t.narrow(0, 0, length)


            reverse_feature_s = ReverseLayerF.apply(feature_s, alpha)
            reverse_feature_t = ReverseLayerF.apply(feature_t, alpha)
            class_output_s = self.class_classifier(feature_s)
            domain_output_s = self.domain_classifier(reverse_feature_s)
            domain_output_t = self.domain_classifier(reverse_feature_t)
            return class_output_s, class_output_t, domain_output_s, domain_output_t, domain_output_mix1, domain_output_mix2
        else:
            input_data_t = input_data_t.expand(input_data_t.data.shape[0], 3, 28, 28)
            feature_t = self.feature(input_data_t)
            feature_t = feature_t.view(-1, 50 * 4 * 4)
            class_output_t = self.class_classifier(feature_t)
            return 1, class_output_t, 2, 3, 4, 5
