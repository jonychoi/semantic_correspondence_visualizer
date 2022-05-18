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

import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _quadruple
from torch.autograd import Variable
from torch.nn import Conv2d

from .conv4d import Conv4d

class con_estimator_4D(nn.Module):
    def __init__(self,input_channels=1):
        super(con_estimator_4D, self).__init__()

        #self.input_channels = 256+128 #input_channels
        # (b, 256, 16, 16)
        # (b, src, trg, trg)
        # 256+8*128=1280

        # self.cost_conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=9, padding=4)
        # self.cost_bn1 = nn.BatchNorm2d(64)
        # self.cost_conv2 = nn.Conv2d(64, 64, kernel_size=7, padding=3)
        # self.cost_bn2 = nn.BatchNorm2d(64)
        # self.cost_conv3 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        # self.cost_bn3 = nn.BatchNorm2d(64)
        # self.cost_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.cost_bn4 = nn.BatchNorm2d(64)
        # self.cost_pred = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        self.cost_conv1 = Conv4d(1, 1, kernel_size=(17,17,9,9), stride=1, padding=4, bias=True,)
        self.cost_conv2 = Conv4d(1, 1, kernel_size=(7,7,7,7), stride=1, padding=3, bias=True,)
        self.cost_conv3 = Conv4d(1, 1, kernel_size=(5,5,5,5), stride=1, padding=2, bias=True,)
        self.cost_conv4 = Conv4d(1, 1, kernel_size=(3,3,3,3), stride=1, padding=1, bias=True,)
        self.cost_pred = Conv4d(1, 1, kernel_size=(8,8,1,1), stride=1, padding=0, bias=True,)

        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            #print("module: ", m)
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def L2normalize(self, x):
        norm = x ** 2
        norm = norm.sum(dim=1, keepdim=True) + 1e-6
        norm = norm ** (0.5)
        return (x / norm)

        """ mean 안하는 경우 """
        # tgt_feat = tgt_feat.permute(0,1,3,2).view(b, num_hyp*c, hw).view(b, num_hyp*c, 16, 16 ).detach()
        # src_feat = src_feat.permute(0,1,3,2).view(b, num_hyp*c, hw).view(b, num_hyp*c, 16, 16 ).detach()
        """ end """
        """
        corr.shape = (b, 256, 16, 16)
        tgt_feat.shape = (b, 8, 256, 128) = (b, 8, hw, c)
        src_feat.shape = (b, 8, 256, 128) = (b, 8, hw, c)

        """

    def forward(self, corr):
        b, hw, w, h = corr.shape
        corr = corr.detach()
        corr = corr.view(b,1,w,h,w,h)
        
        #import pdb; pdb.set_trace()
        # tgt_cat = 
        x = F.relu(self.cost_conv1(corr))
        x = F.relu(self.cost_conv2(x))
        x = F.relu(self.cost_conv3(x))
        x = F.relu(self.cost_conv4(x))
        out = self.sigmoid(self.cost_pred(x)) # (b,1,1,1,w,h)
        out = out.view(b,1,w,h)
        return out

    # def forward(self, corr, tgt_feat, src_feat):
    #     b, hw, w, h = corr.shape
    #     b,num_hyp,hw,c = tgt_feat.shape
        
    #     corr = corr.detach()
    #     corr_T = corr.permute(0,2,3,1).view(b,hw, w, h).detach()

    #     """ untranspose idx """
    #     idx = torch.LongTensor( [ [i] for i in range(b) ] ).to("cuda")
    #     wta_idx = torch.max(corr.view(b,hw,hw), dim=1)[1] # shape: (32,256)
    #     """ end """

    #     """ mean 하는 경우 """
    #     tgt_feat = torch.mean(tgt_feat.permute(0,1,3,2), dim=1).view(b, c, hw).view(b, c, 16, 16 ).detach()
    #     src_feat = torch.mean(src_feat.permute(0,1,3,2), dim=1).view(b, c, hw).view(b, c, 16, 16 ).detach()
    #     """ end """

    #     tgt_cat_corr = torch.cat( [corr, tgt_feat], dim=1) # (b, 256+8*128=1280, 16, 16)
    #     src_cat_corr = torch.cat( [corr_T, src_feat], dim=1) # (b, 256+8*128=1280, 16, 16)

    #     # tgt_cat = 
    #     x = F.relu(self.cost_bn1(self.cost_conv1(tgt_cat_corr)))
    #     x = F.relu(self.cost_bn2(self.cost_conv2(x)))
    #     x = F.relu(self.cost_bn3(self.cost_conv3(x)))
    #     x = F.relu(self.cost_bn4(self.cost_conv4(x)))
    #     out = self.sigmoid(self.cost_pred(x))

    #     x_T = F.relu(self.cost_bn1(self.cost_conv1(src_cat_corr)))
    #     x_T = F.relu(self.cost_bn2(self.cost_conv2(x_T)))
    #     x_T = F.relu(self.cost_bn3(self.cost_conv3(x_T)))
    #     x_T = F.relu(self.cost_bn4(self.cost_conv4(x_T)))
    #     out_T = self.sigmoid(self.cost_pred(x_T)) # (32, 1, 16, 16)
    #     out_T = out_T.view(b,hw)[idx, wta_idx] # (32, 256)
    #     out_T = out_T.unsqueeze(dim=1).view(b,1,w,h) # (32, 1, 16, 16)

    #     return (out+out_T)/2