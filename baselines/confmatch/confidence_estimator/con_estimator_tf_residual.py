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
from models import *
from models.mod import FeatureL2Norm, unnormalise_and_convert_mapping_to_flow
import pdb

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
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
        
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MultiscaleBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn_multiscale = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        '''
        Multi-level aggregation
        '''
        B, N, H, W = x.shape
        if N == 1:
            x = x.flatten(0, 1)
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x.view(B, N, H, W)
        x = x.flatten(0, 1)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp2(self.norm4(x)))
        x = x.view(B, N, H, W).transpose(1, 2).flatten(0, 1) 
        x = x + self.drop_path(self.attn_multiscale(self.norm3(x)))
        x = x.view(B, H, N, W).transpose(1, 2).flatten(0, 1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.view(B, N, H, W)
        return x



class con_estimator_tf_residual_shallow(nn.Module):
    def __init__(self, img_size=16, embed_dim=[96,1], proj_feat_input_dim = 1024, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.img_size = img_size
        self.num_features = self.embed_dim = embed_dim[0]  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.pos_embed_x = nn.Parameter(torch.zeros(1, 1, 1, img_size, embed_dim[0]// 2))
        self.pos_embed_y = nn.Parameter(torch.zeros(1, 1, img_size, 1, embed_dim[0]// 2))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            MultiscaleBlock(
                dim=embed_dim[0], num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.proj_conf = nn.Linear(embed_dim[0], embed_dim[1]) 
        # self.proj_src = nn.Linear(8, 1) ####hyperpixel
        self.proj_feat = nn.Linear(proj_feat_input_dim, 32) ####hyperpixel
        # self.proj_S_Tvec = nn.Linear(2,96)  ####S_Tvec projection
        self.proj_map = nn.Linear(2,32)  ####T_Svec projection
        self.proj_corr = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.norm = norm_layer(embed_dim[0])
       
        trunc_normal_(self.pos_embed_x, std=.02)
        trunc_normal_(self.pos_embed_y, std=.02)
        self.apply(self._init_weights)
        self.sigmoid = nn.Sigmoid()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, corr, source, target, T_Svec_map,S_Tvec_map,args):
        # ???????????? corr = 1??? dimension ??? 1 ?????????
        corr = corr.detach()
        source = source.detach()
        target = target.detach()
        #########################################????????? ?????? ####################################
        T_Svec_map = T_Svec_map.detach()
        S_Tvec_map = S_Tvec_map.detach()

        flattened_T_Svec_map = T_Svec_map.flatten(2,3) #Bx2x16x16  ??? Bx2x256 ?????? flattening
        flattened_S_Tvec_map = S_Tvec_map.flatten(2,3) #Bx2x16x16  ??? Bx2x256 ?????? flattening

        transposed_T_Svec_map = flattened_T_Svec_map.transpose(-1,-2) #Bx2x256  ??? Bx256x2 ?????? transpose ???, Bx1x256x2 ??? unsqueeze
        transposed_S_Tvec_map = flattened_S_Tvec_map.transpose(-1,-2) #Bx2x256  ??? Bx256x2 ?????? transpose ???, Bx1x256x2 ??? unsqueeze
        
        if args.con_est_input_map_direction == 'forward':
            transposed_map_1 = transposed_T_Svec_map
            transposed_map_2 = transposed_S_Tvec_map
        elif args.con_est_input_map_direction == 'reverse':
            transposed_map_1 = transposed_S_Tvec_map
            transposed_map_2 = transposed_T_Svec_map

        B = corr.shape[0] # B=32
        x0 = corr.clone() #torch.Size([32, 1, 256, 256])
        pos_embed = torch.cat((self.pos_embed_x.repeat(1, 1, self.img_size, 1, 1), self.pos_embed_y.repeat(1, 1, 1, self.img_size, 1)), dim=4) 
        pos_embed = pos_embed.flatten(2, 3)


        if args.con_est_input_mode == 'tgt':
            x0 = x0.view(B, self.img_size*self.img_size, self.img_size, self.img_size)
            projected_corr= self.relu(self.proj_corr(x0)) #(B, 32, 16, 16)
            projected_corr = projected_corr.flatten(2).transpose(-1, -2).unsqueeze(1) #(B, 1, 256, 32)                    
            projected_map_1 = self.relu(self.proj_map(transposed_map_1))
            projected_target_feat1 = self.relu(self.proj_feat(target.permute(0, 2, 3, 1).flatten(2)))
            concated_src = torch.cat((projected_corr, projected_map_1.unsqueeze(1), projected_target_feat1.unsqueeze(1)), dim=3) + pos_embed #torch.Size([32, 1, 256, 384])
            residual_concated_src = self.blocks(concated_src) + concated_src
            confidence_map = self.relu(self.proj_conf(residual_concated_src))  # swapping the axis for swapping self-attention. # x2 : torch.Size([32, 8, 256, 256])       

        confidence_map = confidence_map.view(B, 1, 256) 
        confidence_map = self.sigmoid(confidence_map)
        
        return confidence_map.view(B, 1, 16, 16)


