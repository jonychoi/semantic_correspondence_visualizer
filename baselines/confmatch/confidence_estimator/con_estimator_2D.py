import os
import sys
from operator import add
from functools import reduce, partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
from timm.models.layers import DropPath, trunc_normal_
import torchvision.models as models

from models.feature_backbones import resnet
from models.mod import FeatureL2Norm, unnormalise_and_convert_mapping_to_flow
import pdb



# class con_estimator(nn.Module):
#     def __init__(self,input_channels=1):
#         super(con_estimator, self).__init__()
#         self.input_channels = input_channels

#         self.cost_conv1 = Conv4d(in_channels=1,out_channels=1,padding =1, kernel_size=(3,3,3,3),
#         bias_initializer=lambda x: torch.nn.init.constant_(x, 0.0),
#         kernel_initializer = lambda x: nn.init.kaiming_normal_(x, mode='fan_out', nonlinearity='relu'))
#         # self.cost_bn1 = nn.BatchNorm2d(64)

#         self.cost_conv2 = Conv4d(in_channels=1,out_channels=1,padding =1,kernel_size=(3, 3, 3, 3),
#         bias_initializer=lambda x: torch.nn.init.constant_(x, 0.0),
#         kernel_initializer = lambda x: nn.init.kaiming_normal_(x, mode='fan_out', nonlinearity='relu'))

#         # self.cost_bn2 = nn.BatchNorm2d(64)
#         self.cost_conv3 = Conv4d(in_channels=1,out_channels=1,padding =1,kernel_size=(3, 3, 3, 3),
#         bias_initializer=lambda x: torch.nn.init.constant_(x, 0.0),
#         kernel_initializer = lambda x: nn.init.kaiming_normal_(x, mode='fan_out', nonlinearity='relu') )

#         # self.cost_bn3 = nn.BatchNorm2d(64)
#         self.cost_conv4 = Conv4d(in_channels=1,out_channels=1,padding =1,kernel_size=(3, 3, 3, 3),
#         bias_initializer=lambda x: torch.nn.init.constant_(x, 0.0),
#         kernel_initializer = lambda x: nn.init.kaiming_normal_(x, mode='fan_out', nonlinearity='relu'))

#         # self.cost_bn4 = nn.BatchNorm2d(64)
#         self.cost_pred = Conv4d(in_channels=1,out_channels=1,padding =1,kernel_size=(18, 18, 3, 3),
#         bias_initializer=lambda x: torch.nn.init.constant_(x, 0.0),
#         kernel_initializer = lambda x: nn.init.kaiming_normal_(x, mode='fan_out', nonlinearity='relu'))

#         self.sigmoid = nn.Sigmoid()

#         # for m in self.modules():
#         #     if isinstance(m, nn.Conv2d):
#         #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#         #     elif isinstance(m, nn.BatchNorm2d):
#         #         nn.init.constant_(m.weight, 1)
#         #         nn.init.constant_(m.bias, 0)

#     def L2normalize(self, x):
#         norm = x ** 2
#         norm = norm.sum(dim=1, keepdim=True) + 1e-6
#         norm = norm ** (0.5)
#         return (x / norm)

#     def forward(self, corr):
#         corr = corr.detach()
#         x = F.relu(self.cost_conv1.forward(corr))
#         x = F.relu(self.cost_conv2.forward(x))
#         x = F.relu(self.cost_conv3.forward(x))
#         x = F.relu(self.cost_conv4.forward(x))
        
#         out = self.sigmoid(self.cost_pred.forward(x))
#         B,_,_,_,H,W = out.size()
#         out = out.view(B,1,H,W)
#         return out


class con_estimator(nn.Module):
    def __init__(self,input_channels=1):
        super(con_estimator, self).__init__()
        self.input_channels = input_channels
        self.cost_conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=9, padding=4)
        self.cost_bn1 = nn.BatchNorm2d(64)
        self.cost_conv2 = nn.Conv2d(64, 64, kernel_size=7, padding=3)
        self.cost_bn2 = nn.BatchNorm2d(64)
        self.cost_conv3 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.cost_bn3 = nn.BatchNorm2d(64)
        self.cost_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.cost_bn4 = nn.BatchNorm2d(64)
        self.cost_pred = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def L2normalize(self, x):
        norm = x ** 2
        norm = norm.sum(dim=1, keepdim=True) + 1e-6
        norm = norm ** (0.5)
        return (x / norm)

    def forward(self, corr):
        corr = corr.detach()
        x = F.relu(self.cost_bn1(self.cost_conv1(corr)))
        x = F.relu(self.cost_bn2(self.cost_conv2(x)))
        x = F.relu(self.cost_bn3(self.cost_conv3(x)))
        x = F.relu(self.cost_bn4(self.cost_conv4(x)))
        out = self.sigmoid(self.cost_pred(x))

        return out















#### 원본
# class con_estimator(nn.Module):
#     def __init__(self,input_channels=1):
#         super(con_estimator, self).__init__()
#         self.input_channels = input_channels
#         self.cost_conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=9, padding=4)
#         self.cost_bn1 = nn.BatchNorm2d(64)
#         self.cost_conv2 = nn.Conv2d(64, 64, kernel_size=7, padding=3)
#         self.cost_bn2 = nn.BatchNorm2d(64)
#         self.cost_conv3 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
#         self.cost_bn3 = nn.BatchNorm2d(64)
#         self.cost_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.cost_bn4 = nn.BatchNorm2d(64)
#         self.cost_pred = nn.Conv2d(64, 1, kernel_size=1, padding=0)

#         self.sigmoid = nn.Sigmoid()

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def L2normalize(self, x):
#         norm = x ** 2
#         norm = norm.sum(dim=1, keepdim=True) + 1e-6
#         norm = norm ** (0.5)
#         return (x / norm)

#     def forward(self, corr):
#         corr = corr.detach()
#         x = F.relu(self.cost_bn1(self.cost_conv1(corr)))
#         x = F.relu(self.cost_bn2(self.cost_conv2(x)))
#         x = F.relu(self.cost_bn3(self.cost_conv3(x)))
#         x = F.relu(self.cost_bn4(self.cost_conv4(x)))
#         out = self.sigmoid(self.cost_pred(x))

#         return out

# class con_estimator(nn.Module):
#     def __init__(self,input_channels=1):
#         super(con_estimator, self).__init__()
#         ###############################original deep layer #############################333
#         self.input_channels = input_channels
#         self.cost_conv1_1 = nn.Conv2d(self.input_channels, 128, kernel_size=3, padding=1)
#         self.cost_bn1_1 = nn.BatchNorm2d(128)
#         self.cost_conv1_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.cost_bn1_2 = nn.BatchNorm2d(128)
#         self.cost_conv1_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
#         self.cost_bn1_3 = nn.BatchNorm2d(64)

#         self.cost_conv2_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.cost_bn2_1 = nn.BatchNorm2d(64)
#         self.cost_conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.cost_bn2_2 = nn.BatchNorm2d(64)
#         self.cost_conv2_3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
#         self.cost_bn2_3 = nn.BatchNorm2d(32)

#         self.cost_conv3_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
#         self.cost_bn3_1 = nn.BatchNorm2d(32)
#         self.cost_conv3_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
#         self.cost_bn3_2 = nn.BatchNorm2d(32)
#         self.cost_conv3_3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
#         self.cost_bn3_3 = nn.BatchNorm2d(16)

#         self.cost_conv4_1 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
#         self.cost_bn4_1 = nn.BatchNorm2d(16)
#         self.cost_conv4_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
#         self.cost_bn4_2 = nn.BatchNorm2d(16)

#         self.cost_pred = nn.Conv2d(16, 1, kernel_size=1, padding=0)

#         self.sigmoid = nn.Sigmoid()
#         #############################################################################################
#         #######################################test ######################################################
#         # self.input_channels = input_channels
#         # self.cost_conv1_1 = nn.Conv2d(self.input_channels, 128, kernel_size=3, padding=1)
#         # self.cost_bn1_1 = nn.BatchNorm2d(128)
#         # self.cost_conv1_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         # self.cost_bn1_2 = nn.BatchNorm2d(128)
#         # self.cost_conv1_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
#         # self.cost_bn1_3 = nn.BatchNorm2d(64)

#         # self.cost_conv2_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         # self.cost_bn2_1 = nn.BatchNorm2d(64)
#         # self.cost_conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         # self.cost_bn2_2 = nn.BatchNorm2d(64)
#         # self.cost_conv2_3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
#         # self.cost_bn2_3 = nn.BatchNorm2d(32)

#         # self.cost_conv3_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
#         # self.cost_bn3_1 = nn.BatchNorm2d(32)
#         # self.cost_conv3_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
#         # self.cost_bn3_2 = nn.BatchNorm2d(32)
#         # self.cost_conv3_3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
#         # self.cost_bn3_3 = nn.BatchNorm2d(16)

#         # self.cost_conv4_1 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
#         # self.cost_bn4_1 = nn.BatchNorm2d(16)
#         # self.cost_conv4_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
#         # self.cost_bn4_2 = nn.BatchNorm2d(16)
#         # self.cost_conv4_3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
#         # self.cost_bn4_3 = nn.BatchNorm2d(8)

#         # self.cost_conv5_1 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
#         # self.cost_bn5_1 = nn.BatchNorm2d(8)
#         # self.cost_conv5_2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
#         # self.cost_bn5_2 = nn.BatchNorm2d(8)

#         # self.cost_pred = nn.Conv2d(8, 1, kernel_size=1, padding=0)

#         # self.sigmoid = nn.Sigmoid()
#         ##################################################################################################
#         # self.cost_conv3_2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
#         # self.cost_bn3_2 = nn.BatchNorm2d(8)

#         # self.cost_conv4_1 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
#         # self.cost_bn4_1 = nn.BatchNorm2d(8)
#         # self.cost_conv4_2 = nn.Conv2d(8, 4, kernel_size=3, padding=1)
#         # self.cost_bn4_2 = nn.BatchNorm2d(4)

#         # # self.cost_conv5_1 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
#         # # self.cost_bn5_1 = nn.BatchNorm2d(4)
#         # # self.cost_conv5_2 = nn.Conv2d(4, 2, kernel_size=3, padding=1)
#         # # self.cost_bn5_2 = nn.BatchNorm2d(2)

#         # self.cost_conv5_1 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
#         # self.cost_bn5_1 = nn.BatchNorm2d(4)
       

        

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def L2normalize(self, x):
#         norm = x ** 2
#         norm = norm.sum(dim=1, keepdim=True) + 1e-6
#         norm = norm ** (0.5)
#         return (x / norm)

#     def forward(self, corr):
#         corr = corr.detach()
#         x = F.relu(self.cost_bn1_1(self.cost_conv1_1(corr)))
#         x = F.relu(self.cost_bn1_2(self.cost_conv1_2(x)))
#         x = F.relu(self.cost_bn1_3(self.cost_conv1_3(x)))

#         x = F.relu(self.cost_bn2_1(self.cost_conv2_1(x)))
#         x = F.relu(self.cost_bn2_2(self.cost_conv2_2(x)))
#         x = F.relu(self.cost_bn2_3(self.cost_conv2_3(x)))

#         x = F.relu(self.cost_bn3_1(self.cost_conv3_1(x)))
#         x = F.relu(self.cost_bn3_2(self.cost_conv3_2(x)))
#         x = F.relu(self.cost_bn3_3(self.cost_conv3_3(x)))

#         x = F.relu(self.cost_bn4_1(self.cost_conv4_1(x)))
#         x = F.relu(self.cost_bn4_2(self.cost_conv4_2(x)))
#         # x = F.relu(self.cost_bn4_3(self.cost_conv4_3(x)))

#         # x = F.relu(self.cost_bn5_1(self.cost_conv5_1(x)))
#         # x = F.relu(self.cost_bn5_2(self.cost_conv5_2(x)))

        
#         # x = F.relu(self.cost_bn4_1(self.cost_conv4_1(x)))
#         # x = F.relu(self.cost_bn4_2(self.cost_conv4_2(x)))
        

#         # x = F.relu(self.cost_bn5_1(self.cost_conv5_1(x)))

        
#         out = self.sigmoid(self.cost_pred(x))

#         return out


#residual적용

# class con_estimator(nn.Module):
#     def __init__(self,input_channels=1):
#         super(con_estimator, self).__init__()
#         self.input_channels = input_channels

#         self.cost_conv1_1 = nn.Conv2d(self.input_channels, 256, kernel_size=3, padding=1)
#         self.cost_bn1_1 = nn.BatchNorm2d(256)
#         self.cost_conv1_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.cost_bn1_2 = nn.BatchNorm2d(256)
        

#         self.cost_conv2_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.cost_bn2_1 = nn.BatchNorm2d(256)
#         self.cost_conv2_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.cost_bn2_2 = nn.BatchNorm2d(256)


#         self.cost_conv3_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.cost_bn3_1 = nn.BatchNorm2d(256)
#         self.cost_conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.cost_bn3_2 = nn.BatchNorm2d(256)


#         self.cost_conv4_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.cost_bn4_1 = nn.BatchNorm2d(256)
#         self.cost_conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.cost_bn4_2 = nn.BatchNorm2d(256)


#         self.cost_conv5_1 = nn.Conv2d(256,256, kernel_size=3, padding=1)
#         self.cost_bn5_1 = nn.BatchNorm2d(256)
#         self.cost_conv5_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.cost_bn5_2 = nn.BatchNorm2d(256)

#         self.cost_pred = nn.Conv2d(256, 1, kernel_size=3, padding=1)

#         # self.cost_conv4_2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
#         # self.cost_bn4_2 = nn.BatchNorm2d(8)
#         # self.cost_conv4_3 = nn.Conv2d(8, 4, kernel_size=3, padding=1)
#         # self.cost_bn4_3 = nn.BatchNorm2d(8)
#         # self.cost_conv5_1 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
#         # self.cost_bn5_1 = nn.BatchNorm2d(8)
#         # self.cost_conv5_2 = nn.Conv2d(4,4, kernel_size=3, padding=1)
#         # self.cost_bn5_2 = nn.BatchNorm2d(8)
#         # self.cost_conv5_3 = nn.Conv2d(4, 2, kernel_size=3, padding=1)
#         # self.cost_bn5_3 = nn.BatchNorm2d(2)
#         # self.cost_conv5_1 = nn.Conv2d(2, 2, kernel_size=3, padding=1)
#         # self.cost_bn5_1 = nn.BatchNorm2d(2)
#         # self.cost_conv5_2 = nn.Conv2d(2,2, kernel_size=3, padding=1)
#         # self.cost_bn5_2 = nn.BatchNorm2d(2)
#         # self.cost_pred = nn.Conv2d(2, 1, kernel_size=3, padding=1)
#         # self.cost_bn5_3 = nn.BatchNorm2d(4)
#         # self.cost_pred = nn.Conv2d(4, 1, kernel_size=1, padding=0)

#         self.sigmoid = nn.Sigmoid()

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def L2normalize(self, x):
#         norm = x ** 2
#         norm = norm.sum(dim=1, keepdim=True) + 1e-6
#         norm = norm ** (0.5)
#         return (x / norm)

#     def forward(self, corr):
#         corr_1 = corr.detach()
#         residual = corr.detach()
#         x = F.relu(self.cost_bn1_1(self.cost_conv1_1(corr_1)))
#         x = F.relu(self.cost_bn1_2(self.cost_conv1_2(x)))
#         residual= x+residual 
#         x = residual
        
#         x = F.relu(self.cost_bn2_1(self.cost_conv2_1(x)))
#         x = F.relu(self.cost_bn2_2(self.cost_conv2_2(x)))
#         # pdb.set_trace()
#         residual= x+residual 
#         x = residual

#         x = F.relu(self.cost_bn3_1(self.cost_conv3_1(x)))
#         x = F.relu(self.cost_bn3_2(self.cost_conv3_2(x)))

#         residual= x+residual 
#         x = residual

#         x = F.relu(self.cost_bn4_1(self.cost_conv4_1(x)))
#         x = F.relu(self.cost_bn4_2(self.cost_conv4_2(x)))

#         residual= x+residual 
#         x = residual

#         x = F.relu(self.cost_bn5_1(self.cost_conv5_1(x)))
#         x = F.relu(self.cost_bn5_2(self.cost_conv5_2(x)))

#         # residual= x+residual 
#         # x = residual

#         out = self.sigmoid(self.cost_pred(x))

#         return out


# class con_estimator(nn.Module):
#     def __init__(self,input_channels=1):
#         super(con_estimator, self).__init__()
#         self.input_channels = input_channels
#         self.cost_conv1 = nn.Conv2d(self.input_channels, self.input_channels, kernel_size=1, padding=0)
#         self.cost_bn1 = nn.BatchNorm2d(self.input_channels)
#         self.cost_conv2 = nn.Conv2d(self.input_channels, self.input_channels, kernel_size=1, padding=0)
#         self.cost_bn2 = nn.BatchNorm2d(self.input_channels)
#         self.cost_conv3 = nn.Conv2d(self.input_channels, self.input_channels, kernel_size=1, padding=0)
#         self.cost_bn3 = nn.BatchNorm2d(self.input_channels)
#         self.cost_conv4 = nn.Conv2d(self.input_channels, self.input_channels, kernel_size=1, padding=0)
#         self.cost_bn4 = nn.BatchNorm2d(self.input_channels)
#         self.cost_pred = nn.Conv2d(self.input_channels, self.input_channels, kernel_size=1, padding=0)

#         self.sigmoid = nn.Sigmoid()

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def L2normalize(self, x):
#         norm = x ** 2
#         norm = norm.sum(dim=1, keepdim=True) + 1e-6
#         norm = norm ** (0.5)
#         return (x / norm)

#     def forward(self, corr):
#         corr = corr.detach()
#         x = F.relu(self.cost_bn1(self.cost_conv1(corr)))
#         x = F.relu(self.cost_bn2(self.cost_conv2(x)))
#         x = F.relu(self.cost_bn3(self.cost_conv3(x)))
#         x = F.relu(self.cost_bn4(self.cost_conv4(x)))
#         out = self.sigmoid(self.cost_pred(x))
#         b,c,h,w = out.size()
#         out = out.view(b,c,h*w)
#         # pdb.set_trace()
#         return out