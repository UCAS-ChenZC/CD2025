import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
from functools import partial
from models.ChangeFormerBaseNetworks import *
from models.help_funcs import TwoLayerConv2d, save_to_mat
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
from mmengine.model.weight_init import normal_init
from mmcv.cnn import ConvModule
import pdb
from scipy.io import savemat
from models.pixel_shuffel_up import PS_UP
from mmcv.utils import get_logger
import logging
import warnings
from mmcv.runner import load_checkpoint
from models.swin import SwinTransformer
from models.channel_mapper import ChannelMapper
from models.CRA import ImprovedCRA
from models.FreFusion import FreqFusion,LocalSimGuidedSampler
import torch.fft
from functools import partial
from models.Adaptive_Star_Block import Adaptive_Star_Block
from models.SHSA import DAttention,LayerNormProxy,CCAttention,Mix,GlobalAttentionRepViTBlock,GlobalAttention,Conv2d_BN
#from models.SHSA import BN_Linear,Conv2d_BN,GroupNorm
from models.MSDConv_SSFC import MSDConv_SSFC
from Attention import CBAMLayer, SpatialAttention,ChannelAttention,CBAM_Attention,DAFM,DCG_AttnBlock
from Conv import FADConv

import warnings
warnings.filterwarnings('ignore')

def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name='mmdet', log_file=log_file, log_level=log_level)

    return logger
class EncoderTransformer(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, num_classes=2, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch embedding definitions
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # main  encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        # intra-patch encoder
        self.patch_block1 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(1)])
        self.pnorm1 = norm_layer(embed_dims[1])
        # main  encoder
        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        # intra-patch encoder
        self.patch_block2 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(1)])
        self.pnorm2 = norm_layer(embed_dims[2])
        # main  encoder
        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        # intra-patch encoder
        self.patch_block3 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[1], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(1)])
        self.pnorm3 = norm_layer(embed_dims[3])
        # main  encoder
        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        embed_dims=[64, 128, 320, 512]
        # stage 1
        x1, H1, W1 = self.patch_embed1(x)

        for i, blk in enumerate(self.block1):
            x1 = blk(x1, H1, W1)
        x1 = self.norm1(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()

        outs.append(x1)

        # stage 2
        x1, H1, W1 = self.patch_embed2(x1)
        x1 = x1.permute(0,2,1).reshape(B,embed_dims[1],H1,W1)

        x1 = x1.view(x1.shape[0],x1.shape[1],-1).permute(0,2,1)

        for i, blk in enumerate(self.block2):
            x1 = blk(x1, H1, W1)
        x1 = self.norm2(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        # stage 3
        x1, H1, W1 = self.patch_embed3(x1)
        x1 = x1.permute(0,2,1).reshape(B,embed_dims[2],H1,W1)

        x1 = x1.view(x1.shape[0],x1.shape[1],-1).permute(0,2,1)

        for i, blk in enumerate(self.block3):
            x1 = blk(x1, H1, W1)
        x1 = self.norm3(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        # stage 4
        x1, H1, W1 = self.patch_embed4(x1)
        x1 = x1.permute(0,2,1).reshape(B,embed_dims[3],H1,W1) #+x2

        x1 = x1.view(x1.shape[0],x1.shape[1],-1).permute(0,2,1)

        for i, blk in enumerate(self.block4):
            x1 = blk(x1, H1, W1)
        x1 = self.norm4(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        return outs

    def forward(self, x):
        x = self.forward_features(x)

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # pdb.set_trace()
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Attention_dec(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.task_query = nn.Parameter(torch.randn(1,48,dim))
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        
        B, N, C = x.shape
        task_q = self.task_query
        
        # This is because we fix the task parameters to be of a certain dimension, so with varying batch size, we just stack up the same queries to operate on the entire batch
        if B>1:
            
            task_q = task_q.unsqueeze(0).repeat(B,1,1,1)
            task_q = task_q.squeeze(1)

        q = self.q(task_q).reshape(B, task_q.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = torch.nn.functional.interpolate(q,size= (v.shape[2],v.shape[3]))
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block_dec(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_dec(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Tenc(EncoderTransformer):
    def __init__(self, **kwargs):
        super(Tenc, self).__init__(
            patch_size=16, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class convprojection(nn.Module):
    def __init__(self, path=None, **kwargs):
        super(convprojection,self).__init__()

        self.convd32x = UpsampleConvLayer(512, 512, kernel_size=4, stride=2)
        self.convd16x = UpsampleConvLayer(512, 320, kernel_size=4, stride=2)
        self.dense_4 = nn.Sequential(ResidualBlock(320))
        self.convd8x = UpsampleConvLayer(320, 128, kernel_size=4, stride=2)
        self.dense_3 = nn.Sequential(ResidualBlock(128))
        self.convd4x = UpsampleConvLayer(128, 64, kernel_size=4, stride=2)
        self.dense_2 = nn.Sequential(ResidualBlock(64))
        self.convd2x = UpsampleConvLayer(64, 16, kernel_size=4, stride=2)
        self.dense_1 = nn.Sequential( ResidualBlock(16))
        self.convd1x = UpsampleConvLayer(16, 8, kernel_size=4, stride=2)
        self.conv_output = ConvLayer(8, 2, kernel_size=3, stride=1, padding=1)

        self.active = nn.Tanh()        

    def forward(self,x1,x2):

        res32x = self.convd32x(x2[0])

        if x1[3].shape[3] != res32x.shape[3] and x1[3].shape[2] != res32x.shape[2]:
            p2d = (0,-1,0,-1)
            res32x = F.pad(res32x,p2d,"constant",0)
            
        elif x1[3].shape[3] != res32x.shape[3] and x1[3].shape[2] == res32x.shape[2]:
            p2d = (0,-1,0,0)
            res32x = F.pad(res32x,p2d,"constant",0)
        elif x1[3].shape[3] == res32x.shape[3] and x1[3].shape[2] != res32x.shape[2]:
            p2d = (0,0,0,-1)
            res32x = F.pad(res32x,p2d,"constant",0)

        res16x = res32x + x1[3]
        res16x = self.convd16x(res16x) 

        if x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
            p2d = (0,-1,0,-1)
            res16x = F.pad(res16x,p2d,"constant",0)
        elif x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] == res16x.shape[2]:
            p2d = (0,-1,0,0)
            res16x = F.pad(res16x,p2d,"constant",0)
        elif x1[2].shape[3] == res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
            p2d = (0,0,0,-1)
            res16x = F.pad(res16x,p2d,"constant",0)

        res8x = self.dense_4(res16x) + x1[2]
        res8x = self.convd8x(res8x) 
        res4x = self.dense_3(res8x) + x1[1]
        res4x = self.convd4x(res4x)
        res2x = self.dense_2(res4x) + x1[0]
        res2x = self.convd2x(res2x)
        x = res2x
        x = self.dense_1(x)
        x = self.convd1x(x)

        return x

class convprojection_base(nn.Module):
    def __init__(self, path=None, **kwargs):
        super(convprojection_base,self).__init__()

        # self.convd32x = UpsampleConvLayer(512, 512, kernel_size=4, stride=2)
        self.convd16x = UpsampleConvLayer(512, 320, kernel_size=4, stride=2)
        self.dense_4 = nn.Sequential(ResidualBlock(320))
        self.convd8x = UpsampleConvLayer(320, 128, kernel_size=4, stride=2)
        self.dense_3 = nn.Sequential(ResidualBlock(128))
        self.convd4x = UpsampleConvLayer(128, 64, kernel_size=4, stride=2)
        self.dense_2 = nn.Sequential(ResidualBlock(64))
        self.convd2x = UpsampleConvLayer(64, 16, kernel_size=4, stride=2)
        self.dense_1 = nn.Sequential( ResidualBlock(16))
        self.convd1x = UpsampleConvLayer(16, 8, kernel_size=4, stride=2)

    def forward(self,x1):

#         if x1[3].shape[3] != res32x.shape[3] and x1[3].shape[2] != res32x.shape[2]:
#             p2d = (0,-1,0,-1)
#             res32x = F.pad(res32x,p2d,"constant",0)
            
#         elif x1[3].shape[3] != res32x.shape[3] and x1[3].shape[2] == res32x.shape[2]:
#             p2d = (0,-1,0,0)
#             res32x = F.pad(res32x,p2d,"constant",0)
#         elif x1[3].shape[3] == res32x.shape[3] and x1[3].shape[2] != res32x.shape[2]:
#             p2d = (0,0,0,-1)
#             res32x = F.pad(res32x,p2d,"constant",0)

#         res16x = res32x + x1[3]
        res16x = self.convd16x(x1[3]) 

        if x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
            p2d = (0,-1,0,-1)
            res16x = F.pad(res16x,p2d,"constant",0)
        elif x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] == res16x.shape[2]:
            p2d = (0,-1,0,0)
            res16x = F.pad(res16x,p2d,"constant",0)
        elif x1[2].shape[3] == res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
            p2d = (0,0,0,-1)
            res16x = F.pad(res16x,p2d,"constant",0)

        res8x = self.dense_4(res16x) + x1[2]
        res8x = self.convd8x(res8x) 
        res4x = self.dense_3(res8x) + x1[1]
        res4x = self.convd4x(res4x)
        res2x = self.dense_2(res4x) + x1[0]
        res2x = self.convd2x(res2x)
        x = res2x
        x = self.dense_1(x)
        x = self.convd1x(x)
        return x

### This is the basic ChangeFormer module
class ChangeFormerV1(nn.Module):

    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False):
        super(ChangeFormerV1, self).__init__()

        self.Tenc               = Tenc()
        
        self.convproj           = convprojection_base()

        self.change_probability = ConvLayer(8, output_nc, kernel_size=3, stride=1, padding=1)

        self.output_softmax     = decoder_softmax
        self.active             = torch.nn.Softmax(dim=1)

    def forward(self, x1, x2):

        fx1 = self.Tenc(x1)
        fx2 = self.Tenc(x2)

        DI = []
        for i in range(0,4):
            DI.append(torch.abs(fx1[i] - fx2[i]))

        cp = self.convproj(DI)

        cp = self.change_probability(cp)

        if self.output_softmax:
            cp = self.active(cp)

        return cp

# Transformer Decoder
class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x



class TDecV2(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True, 
                    in_channels = [64, 128, 256, 512], embedding_dim= 256, output_nc=2, 
                    decoder_softmax = False, feature_strides=[4, 8, 16, 32]):
        super(TDecV2, self).__init__()
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        #input transforms
        self.input_transform = input_transform
        self.in_index = in_index
        self.align_corners = align_corners

        #MLP
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        #Final prediction
        self.output_nc = output_nc

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        self.linear_fuse = nn.Conv2d(   in_channels=self.embedding_dim*4, out_channels=self.embedding_dim,
                                        kernel_size=1)

        #self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        # self.convd2x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        # self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        # self.convd1x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        # self.dense_1x   = nn.Sequential( ResidualBlock(self.embedding_dim))

        #Pixel Shiffle
        self.pix_shuffle_conv   = nn.Conv2d(in_channels=self.embedding_dim, out_channels=16*output_nc, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pix_shuffle        = nn.PixelShuffle(4)

        #Final prediction
        # self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        
        #Final activation
        self.output_softmax     = decoder_softmax
        self.active             = nn.Softmax(dim=1) 

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs1, inputs2):
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/4,1/8,1/16,1/32
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/4,1/8,1/16,1/32

        c1_1, c2_1, c3_1, c4_1 = x_1
        c1_2, c2_2, c3_2, c4_2 = x_2

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        _c4_1 = self.linear_c4(c4_1).permute(0,2,1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_1 = resize(_c4_1, size=c1_1.size()[2:],mode='bilinear',align_corners=False)
        _c4_2 = self.linear_c4(c4_2).permute(0,2,1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4_2 = resize(_c4_2, size=c1_2.size()[2:],mode='bilinear',align_corners=False)

        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_1 = resize(_c3_1, size=c1_1.size()[2:],mode='bilinear',align_corners=False)
        _c3_2 = self.linear_c3(c3_2).permute(0,2,1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        _c3_2 = resize(_c3_2, size=c1_2.size()[2:],mode='bilinear',align_corners=False)

        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_1 = resize(_c2_1, size=c1_1.size()[2:],mode='bilinear',align_corners=False)
        _c2_2 = self.linear_c2(c2_2).permute(0,2,1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2_2 = resize(_c2_2, size=c1_2.size()[2:],mode='bilinear',align_corners=False)

        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0,2,1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])

        _c = self.linear_fuse(torch.cat([torch.abs(_c4_1-_c4_2), torch.abs(_c3_1-_c3_2), torch.abs(_c2_1-_c2_2), torch.abs(_c1_1-_c1_2)], dim=1))

        # x = self.dense_2x(x)
        # x = self.convd1x(x)
        # x = self.dense_1x(x)

        # cp = self.change_probability(x)

        # cp = F.interpolate(_c, scale_factor=4, mode="nearest")
        x  = self.relu(self.pix_shuffle_conv(_c))
        cp = self.pix_shuffle(x)

        if self.output_softmax:
            cp = self.active(cp)

        return cp
    
# ChangeFormerV3:
# Feature differencing and pass it through Transformer decoder
class ChangeFormerV3(nn.Module):

    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False):
        super(ChangeFormerV3, self).__init__()
        #Transformer Encoder
        self.Tenc   = Tenc( patch_size=16, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 4, 8], 
                            mlp_ratios=[4, 4, 4, 4], qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                            depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)
        
        #Transformer Decoder
        self.TDec   = TDecV2( input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True, 
                            in_channels = [64, 128, 320, 512], embedding_dim= 64, output_nc=output_nc, 
                            decoder_softmax = decoder_softmax, feature_strides=[4, 8, 16, 32])

    def forward(self, x1, x2):

        fx1 = self.Tenc(x1)
        fx2 = self.Tenc(x2)

        cp = self.TDec(fx1, fx2)

        return cp

#Transormer Ecoder with x2, x4, x8, x16 scales
class EncoderTransformer_x2(nn.Module):
    def __init__(self, img_size=256, patch_size=3, in_chans=3, num_classes=2, embed_dims=[32, 64, 128, 256, 512],
                 num_heads=[2, 2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 3, 6, 18, 3], sr_ratios=[8, 4, 2, 1, 1]):
        super().__init__()
        self.num_classes    = num_classes
        self.depths         = depths
        self.embed_dims     = embed_dims

        # patch embedding definitions
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=2, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 2, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        self.patch_embed5 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[3],
                                              embed_dim=embed_dims[4])

        # Stage-1 (x1/2 scale)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        
        # Stage-2 (x1/4 scale)
        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
       
       # Stage-3 (x1/8 scale)
        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        
        # Stage-4 (x1/16 scale)
        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])
        
        # Stage-5 (x1/32 scale)
        cur += depths[3]
        self.block5 = nn.ModuleList([Block(
            dim=embed_dims[4], num_heads=num_heads[4], mlp_ratio=mlp_ratios[4], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[4])
            for i in range(depths[4])])
        self.norm5 = norm_layer(embed_dims[4])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
    
        # stage 1
        x1, H1, W1 = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x1 = blk(x1, H1, W1)
        x1 = self.norm1(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        # stage 2
        x1, H1, W1 = self.patch_embed2(x1)
        for i, blk in enumerate(self.block2):
            x1 = blk(x1, H1, W1)
        x1 = self.norm2(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        # stage 3
        x1, H1, W1 = self.patch_embed3(x1)
        for i, blk in enumerate(self.block3):
            x1 = blk(x1, H1, W1)
        x1 = self.norm3(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        # stage 4
        x1, H1, W1 = self.patch_embed4(x1)
        for i, blk in enumerate(self.block4):
            x1 = blk(x1, H1, W1)
        x1 = self.norm4(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        # stage 5
        x1, H1, W1 = self.patch_embed5(x1)
        for i, blk in enumerate(self.block5):
            x1 = blk(x1, H1, W1)
        x1 = self.norm5(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x


#Difference module
def conv_diff(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )

#Intermediate prediction module
def make_prediction(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    )

class DecoderTransformer_x2(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3, 4], align_corners=True, 
                    in_channels = [32, 64, 128, 256, 512], embedding_dim= 64, output_nc=2, 
                    decoder_softmax = False, feature_strides=[2, 4, 8, 16, 32]):
        super(DecoderTransformer_x2, self).__init__()
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        #input transforms
        self.input_transform = input_transform
        self.in_index = in_index
        self.align_corners = align_corners

        #MLP
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        #Final prediction
        self.output_nc = output_nc

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels, c5_in_channels = self.in_channels

        self.linear_c5 = MLP(input_dim=c5_in_channels, embed_dim=self.embedding_dim)
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        #Convolutional Difference Modules
        self.diff_c5   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c4   = conv_diff(in_channels=3*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3   = conv_diff(in_channels=3*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2   = conv_diff(in_channels=3*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1   = conv_diff(in_channels=3*self.embedding_dim, out_channels=self.embedding_dim)

        #Taking outputs from middle of the encoder
        self.make_pred_c5 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        self.linear_fuse = nn.Conv2d(   in_channels=self.embedding_dim*len(in_channels), out_channels=self.embedding_dim,
                                        kernel_size=1)

        #self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.convd2x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.convd1x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x   = nn.Sequential( ResidualBlock(self.embedding_dim))

        #Final prediction
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        
        #Final activation
        self.output_softmax     = decoder_softmax
        self.active             = nn.Sigmoid() 

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs1, inputs2):
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2,1/4,1/8,1/16,1/32
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2,1/4,1/8,1/16,1/32

        c1_1, c2_1, c3_1, c4_1, c5_1 = x_1
        c1_2, c2_2, c3_2, c4_2, c5_2 = x_2

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c5_1.shape

        outputs = [] #Multi-scale outputs adding here
        
        _c5_1 = self.linear_c5(c5_1).permute(0,2,1).reshape(n, -1, c5_1.shape[2], c5_1.shape[3])
        _c5_2 = self.linear_c5(c5_2).permute(0,2,1).reshape(n, -1, c5_2.shape[2], c5_2.shape[3])
        _c5   = self.diff_c5(torch.cat((_c5_1, _c5_2), dim=1)) #Difference of features at x1/32 scale
        p_c5  = self.make_pred_c5(_c5) #Predicted change map at x1/32 scale
        outputs.append(p_c5) #x1/32 scale
        _c5_up= resize(_c5, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        _c4_1 = self.linear_c4(c4_1).permute(0,2,1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0,2,1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4   = self.diff_c4(torch.cat((F.interpolate(_c5, scale_factor=2, mode="bilinear"), _c4_1, _c4_2), dim=1)) #Difference of features at x1/16 scale
        p_c4  = self.make_pred_c4(_c4) #Predicted change map at x1/16 scale
        outputs.append(p_c4) #x1/16 scale
        _c4_up= resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0,2,1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        _c3   = self.diff_c3(torch.cat((F.interpolate(_c4, scale_factor=2, mode="bilinear"), _c3_1, _c3_2), dim=1)) #Difference of features at x1/8 scale
        p_c3  = self.make_pred_c3(_c3) #Predicted change map at x1/8 scale
        outputs.append(p_c3) #x1/8 scale
        _c3_up= resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0,2,1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2   = self.diff_c2(torch.cat((F.interpolate(_c3, scale_factor=2, mode="bilinear"), _c2_1, _c2_2), dim=1)) #Difference of features at x1/4 scale
        p_c2  = self.make_pred_c2(_c2) #Predicted change map at x1/4 scale
        outputs.append(p_c2) #x1/4 scale
        _c2_up= resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0,2,1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1   = self.diff_c1(torch.cat((F.interpolate(_c2, scale_factor=2, mode="bilinear"), _c1_1, _c1_2), dim=1)) #Difference of features at x1/2 scale
        p_c1  = self.make_pred_c1(_c1) #Predicted change map at x1/2 scale
        outputs.append(p_c1) #x1/2 scale

        _c = self.linear_fuse(torch.cat((_c5_up, _c4_up, _c3_up, _c2_up, _c1), dim=1))

        x = self.convd2x(_c)
        x = self.dense_2x(x)
        cp = self.change_probability(x)
        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs





class DecoderTransformer_v3(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True, 
                    in_channels = [32, 64, 128, 256], embedding_dim= 64, output_nc=2, 
                    decoder_softmax = False, feature_strides=[1,2, 4, 8, 16]):
        super(DecoderTransformer_v3, self).__init__()
        #assert
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        
        #settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index        = in_index
        self.align_corners   = align_corners
        self.in_channels     = in_channels
        self.embedding_dim   = embedding_dim
        self.output_nc       = output_nc
        c0_in_channels,c5_in_channels,c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        #MLP decoder heads
        self.linear_c5 = MLP(input_dim=c5_in_channels, embed_dim=self.embedding_dim)
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)
        self.linear_c0 = MLP(input_dim=c0_in_channels, embed_dim=self.embedding_dim)

        #convolutional Difference Modules
        self.diff_c5   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c4   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c0   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)

        #taking outputs from middle of the encoder
        self.make_pred_c5 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c0 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        #Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(   in_channels=self.embedding_dim*len(in_channels), out_channels=self.embedding_dim,
                                        kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )

        #Final predction head
        self.convd2x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.convd1x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        
        #Final activation
        self.output_softmax     = decoder_softmax
        self.active             = nn.Sigmoid() 

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs1, inputs2):
        '''处理输入数据'''
        #Transforming encoder features (select layers)
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16                 (16 64 64 64) --> (16 64 64 64)
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16

        #img1 and img2 features
        c0_1, c5_1, c1_1, c2_1, c3_1, c4_1= x_1
        c0_2, c5_2, c1_2, c2_2, c3_2, c4_2= x_2
        
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        outputs = []
        # Stage 4: x1/32 scale
        _c4_1 = self.linear_c4(c4_1).permute(0,2,1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0,2,1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4   = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))
        p_c4  = self.make_pred_c4(_c4)
        outputs.append(p_c4)
        _c4_up= resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0,2,1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        _c3   = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1)) + F.interpolate(_c4, scale_factor=2, mode="bilinear")
        p_c3  = self.make_pred_c3(_c3)
        outputs.append(p_c3)
        _c3_up= resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0,2,1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2   = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1)) + F.interpolate(_c3, scale_factor=2, mode="bilinear")
        p_c2  = self.make_pred_c2(_c2)
        outputs.append(p_c2)
        _c2_up= resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0,2,1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1   = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1)) + F.interpolate(_c2, scale_factor=2, mode="bilinear")
        p_c1  = self.make_pred_c1(_c1)
        outputs.append(p_c1)


       # Stage 5: x1 scale
        _c5_1 = self.linear_c5(c5_1).permute(0,2,1).reshape(n, -1, c5_1.shape[2], c5_1.shape[3])
        _c5_2 = self.linear_c5(c5_2).permute(0,2,1).reshape(n, -1, c5_2.shape[2], c5_2.shape[3])
        _c5   = self.diff_c5(torch.cat((_c5_1, _c5_2), dim=1)) + F.interpolate(_c1, scale_factor=2, mode="bilinear")
        p_c5  = self.make_pred_c5(_c5)
        outputs.append(p_c5)
        _c5_up = resize(_c5, size=c1_2.size()[2:], mode='bilinear', align_corners=False)


       # Stage 0: x1 scale
        _c0_1 = self.linear_c0(c0_1).permute(0,2,1).reshape(n, -1, c0_1.shape[2], c0_1.shape[3])
        _c0_2 = self.linear_c0(c0_2).permute(0,2,1).reshape(n, -1, c0_2.shape[2], c0_2.shape[3])
        _c0   = self.diff_c0(torch.cat((_c0_1, _c0_2), dim=1)) + F.interpolate(_c5, scale_factor=2, mode="bilinear")
        p_c0  = self.make_pred_c0(_c0)
        outputs.append(p_c0)
        _c0_up = resize(_c0, size=c1_2.size()[2:], mode='bilinear', align_corners=False)




        #Linear Fusion of difference image from all scales
        _c = self.linear_fuse(torch.cat((_c4_up, _c3_up, _c2_up, _c1,_c5_up,_c0_up), dim=1))

        # #Dropout
        # if dropout_ratio > 0:
        #     self.dropout = nn.Dropout2d(dropout_ratio)
        # else:
        #     self.dropout = None

        #Upsampling x2 (x1/2 scale)
        x = self.convd2x(_c)
        #Residual block
        x = self.dense_2x(x)
        #Upsampling x2 (x1 scale)
        x = self.convd1x(x)
        #Residual block
        x = self.dense_1x(x)

        #Final prediction
        cp = self.change_probability(x)
        
        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs

    
class S_DecoderTransformer_v3(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True, 
                    in_channels = [32, 64, 128, 256], embedding_dim= 64, output_nc=2, 
                    decoder_softmax = False, feature_strides=[1,2, 4, 8, 16]):
        super(S_DecoderTransformer_v3, self).__init__()
        #assert
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        
        #settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index        = in_index
        self.align_corners   = align_corners
        self.in_channels     = in_channels
        self.embedding_dim   = embedding_dim
        self.output_nc       = output_nc
        c0_in_channels,c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        #MLP decoder heads
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)
        self.linear_c0 = MLP(input_dim=c0_in_channels, embed_dim=self.embedding_dim)

        #convolutional Difference Modules
        self.diff_c4   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c0   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)

        #taking outputs from middle of the encoder
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c0 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        #Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(   in_channels=self.embedding_dim*len(in_channels), out_channels=self.embedding_dim,
                                        kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )

        #Final predction head
        self.convd2x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.convd1x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        
        #Final activation
        self.output_softmax     = decoder_softmax
        self.active             = nn.Sigmoid() 
        self.Convd1_4 = nn.Conv2d(768,256,kernel_size=1)
        self.Convd1_3 = nn.Conv2d(768,256,kernel_size=1)
        self.Convd1_2 = nn.Conv2d(768,256,kernel_size=1)
        self.Convd1_1 = nn.Conv2d(768,256,kernel_size=1)
        self.Convd1_0 = nn.Conv2d(768,256,kernel_size=1)
        self.CRA = ImprovedCRA(256,reduction_ratio=4)
    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs1, inputs2):
        '''处理输入数据'''
        #Transforming encoder features (select layers)
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16                 (16 64 64 64) --> (16 64 64 64)
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16

        #img1 and img2 features
        c0_1,c1_1, c2_1, c3_1, c4_1 = x_1
        c0_2,c1_2, c2_2, c3_2, c4_2 = x_2
        
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        outputs = []
        # Stage 4: x1/32 scale
        _c4_1 = self.linear_c4(c4_1).permute(0,2,1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0,2,1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4_1_CRA = self.CRA(_c4_1)
        _c4_2_CRA = self.CRA(_c4_2)
        _c4   = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))
        _c4 = torch.cat((_c4,_c4_1_CRA,_c4_2_CRA),dim=1)
        _c4 = self.Convd1_4(_c4)
        p_c4  = self.make_pred_c4(_c4)
        outputs.append(p_c4)
        _c4_up= resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0,2,1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        _c3_1_CRA = self.CRA(_c3_1)
        _c3_2_CRA = self.CRA(_c3_2)
        _c3 = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1))
        _c3 = torch.cat((_c3,_c3_1_CRA,_c3_2_CRA),dim=1)
        _c3 = self.Convd1_3(_c3)
        #_c3 = self.Convd1_3(_c3)+ F.interpolate(_c4, scale_factor=2, mode="bilinear")
        #_c3   = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1)) + F.interpolate(_c4, scale_factor=2, mode="bilinear")
        p_c3  = self.make_pred_c3(_c3)
        outputs.append(p_c3)
        _c3_up= resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0,2,1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2_1_CRA = self.CRA(_c2_1)
        _c2_2_CRA = self.CRA(_c2_2)
        _c2 = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1))
        _c2 = torch.cat((_c2,_c2_1_CRA,_c2_2_CRA),dim=1)
        _c2 = self.Convd1_2(_c2)
        #_c2 = self.Convd1_2(_c2) + F.interpolate(_c3, scale_factor=2, mode="bilinear")
        #_c2   = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1)) + F.interpolate(_c3, scale_factor=2, mode="bilinear")
        p_c2  = self.make_pred_c2(_c2)
        outputs.append(p_c2)
        _c2_up= resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0,2,1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1_1_CRA = self.CRA(_c1_1)
        _c1_2_CRA = self.CRA(_c1_2)
        _c1 = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1))
        _c1 = torch.cat((_c1,_c1_1_CRA,_c1_2_CRA),dim=1)
        _c1 = self.Convd1_1(_c1)
        #_c1 = self.Convd1_1(_c1) + F.interpolate(_c2, scale_factor=2, mode="bilinear")
        #_c1   = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1)) + F.interpolate(_c2, scale_factor=2, mode="bilinear")
        p_c1  = self.make_pred_c1(_c1)
        outputs.append(p_c1)
       
       # Stage 0: x1 scale
        _c0_1 = self.linear_c0(c0_1).permute(0,2,1).reshape(n, -1, c0_1.shape[2], c0_1.shape[3])
        _c0_2 = self.linear_c0(c0_2).permute(0,2,1).reshape(n, -1, c0_2.shape[2], c0_2.shape[3])
        _c0_1_CRA = self.CRA(_c0_1)
        _c0_2_CRA = self.CRA(_c0_2)
        _c0 = self.diff_c0(torch.cat((_c0_1, _c0_2), dim=1))
        _c0 = torch.cat((_c0,_c0_1_CRA,_c0_2_CRA),dim=1)
        _c0 = self.Convd1_0(_c0)
        #_c0 = self.Convd1_0(_c0) + F.interpolate(_c1, scale_factor=4, mode="bilinear")       
        #_c0   = self.diff_c0(torch.cat((_c0_1, _c0_2), dim=1)) + F.interpolate(_c1, scale_factor=4, mode="bilinear")
        p_c0  = self.make_pred_c0(_c0)
        outputs.append(p_c0)
        _c0_up = resize(_c0, size=c1_2.size()[2:], mode='bilinear', align_corners=False)


        #Linear Fusion of difference image from all scales
        _c = self.linear_fuse(torch.cat((_c4_up, _c3_up, _c2_up, _c1,_c0_up), dim=1))

        # #Dropout
        # if dropout_ratio > 0:
        #     self.dropout = nn.Dropout2d(dropout_ratio)
        # else:
        #     self.dropout = None

        #Upsampling x2 (x1/2 scale)
        x = self.convd2x(_c)
        #Residual block
        x = self.dense_2x(x)
        #Upsampling x2 (x1 scale)
        x = self.convd1x(x)
        #Residual block
        x = self.dense_1x(x)

        #Final prediction
        cp = self.change_probability(x)
        
        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs




class Decoder_v4_SegFusion(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3,4,5], align_corners=True, 
                    in_channels = [32, 64, 128, 256], embedding_dim= 64, output_nc=2, 
                    decoder_softmax = False, feature_strides=[1,2, 4, 8, 16,32]):
        super(Decoder_v4_SegFusion, self).__init__()
        #assert
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        
        #settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index        = in_index
        self.align_corners   = align_corners
        self.in_channels     = in_channels
        self.embedding_dim   = 128                              # *** 使用的是embedim 128 ***
        self.output_nc       = output_nc

        #定义输入数量
        # c0_in_channels,c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        c5_inchannels,c0_in_channels,c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        # c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        #MLP decoder heads
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)
        self.linear_c0 = MLP(input_dim=c0_in_channels, embed_dim=self.embedding_dim)
        self.linear_c5 = MLP(input_dim=c5_inchannels, embed_dim=self.embedding_dim)

        #convolutional Difference Modules
        self.diff_c4   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c0   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c5   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        # self.diff_c4   = conv_diff(in_channels=4*self.embedding_dim, out_channels=self.embedding_dim)
        # self.diff_c3   = conv_diff(in_channels=4*self.embedding_dim, out_channels=self.embedding_dim)
        # self.diff_c2   = conv_diff(in_channels=4*self.embedding_dim, out_channels=self.embedding_dim)
        # self.diff_c1   = conv_diff(in_channels=4*self.embedding_dim, out_channels=self.embedding_dim)
        # self.diff_c0   = conv_diff(in_channels=4*self.embedding_dim, out_channels=self.embedding_dim)
        # self.diff_c5   = conv_diff(in_channels=4*self.embedding_dim, out_channels=self.embedding_dim)

        #taking outputs from middle of the encoder
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c0 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c5 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        #Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(   in_channels=self.embedding_dim*5, out_channels=self.embedding_dim,
                                        kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )

        #Final predction head
        self.convd2x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.convd1x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4,stride=2)
        self.dense_1x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        self.Convd11 = nn.Conv2d(128,128,kernel_size=1)
        
        #Final activation
        self.output_softmax     = decoder_softmax
        self.active             = nn.Sigmoid() 

        #相关尝试：卷积、CRA模块以及 FreqFusion特征融合模块

        self.Convd1_1 = nn.Conv2d(768,128,kernel_size=1)
        self.Convd1_0 = nn.Conv2d(512,128,kernel_size=1)
        self.Convd1_2 = nn.Conv2d(256,128,kernel_size=1)
        self.Convd1_3 = nn.Conv2d(384,128,kernel_size=1)
        self.Convd1_4 = nn.Conv2d(640,128,kernel_size=1)

        self.CRA = ImprovedCRA(256,reduction_ratio=4)

        self.ff1 = FreqFusion(128, 128)
        self.ff2 = FreqFusion(128,256)
        self.ff3 = FreqFusion(128,384)
        self.ff4 = FreqFusion(128,512)
        self.ff5= FreqFusion(128,640)

    def _transform_inputs(self, inputs):                                        #对输入特征进行转换
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]
        return inputs

    def forward(self, inputs1, inputs2):
        '''处理输入数据'''
        #Transforming encoder features (select layers)
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16                 (16 64 64 64) --> (16 64 64 64)
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16

        #img1 and img2 features         定义T1 2中 每个尺度的特征
        c5_1,c0_1,c1_1, c2_1, c3_1, c4_1 = x_1
        c5_2,c0_2,c1_2, c2_2, c3_2, c4_2 = x_2
        # c1_1, c2_1, c3_1, c4_1 = x_1
        # c1_2, c2_2, c3_2, c4_2 = x_2        
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape
        ### ！！！目前技术路线：经全连接层处理后 cat拼接 并形成差异图像，后对每个尺度的差异图像进行FreqFusion特征融合 ！！！###
        outputs = []
        # Stage 4: x1/32 scale    [B  256 8 8]   MLP的全连接层操作
        _c4_1 = self.linear_c4(c4_1).permute(0,2,1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0,2,1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])            # 尺寸不变，通道数变为了 embdims   (2 128 8 8)
        # _c4_1_CRA = self.CRA(_c4_1)
        # _c4_2_CRA = self.CRA(_c4_2)
        #_c4   = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))                                              # (2 128 8 8)
        # _c4 = torch.cat((_c4,_c4_1_CRA,_c4_2_CRA),dim=1)
        # c4 = self.Convd1_4(_c4)
        # p_c4  = self.make_pred_c4(_c4)
        # outputs.append(p_c4)
        # _c4_up= resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])            # (2 32 16 16) -> (2 128 16 16)
        _c3_2 = self.linear_c3(c3_2).permute(0,2,1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])



        # _c3_1_CRA = self.CRA(_c3_1)
        # _c3_2_CRA = self.CRA(_c3_2)
        # _c3 = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1))                                                # (2 128 16 16)
        # _c3 = torch.cat((_c3,_c3_1_CRA,_c3_2_CRA),dim=1)
        # _c3 = self.Convd1_3(_c3)
        #_c3 = self.Convd1_3(_c3)+ F.interpolate(_c4, scale_factor=2, mode="bilinear")
        #_c3   = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1)) + F.interpolate(_c4, scale_factor=2, mode="bilinear")
        # p_c3  = self.make_pred_c3(_c3)
        # outputs.append(p_c3)
        #_c3_up= resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0,2,1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        # _c2_1_CRA = self.CRA(_c2_1)
        # _c2_2_CRA = self.CRA(_c2_2)
        # _c2 = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1))
        # _c2 = torch.cat((_c2,_c2_1_CRA,_c2_2_CRA),dim=1)
        # _c2 = self.Convd1_2(_c2)
        #_c2 = self.Convd1_2(_c2) + F.interpolate(_c3, scale_factor=2, mode="bilinear")
        #_c2   = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1)) + F.interpolate(_c3, scale_factor=2, mode="bilinear")
        # p_c2  = self.make_pred_c2(_c2)
        # outputs.append(p_c2)
        #_c2_up= resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0,2,1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        # _c1_1_CRA = self.CRA(_c1_1)
        # _c1_2_CRA = self.CRA(_c1_2)
        # _c1 = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1))
        # _c1 = torch.cat((_c1,_c1_1_CRA,_c1_2_CRA),dim=1)
        # _c1 = self.Convd1_1(_c1)
        #_c1 = self.Convd1_1(_c1) + F.interpolate(_c2, scale_factor=2, mode="bilinear")
        #_c1   = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1)) + F.interpolate(_c2, scale_factor=2, mode="bilinear")
        # p_c1  = self.make_pred_c1(_c1)
        # outputs.append(p_c1)
       
       # Stage 0: x1 scale
        _c0_1 = self.linear_c0(c0_1).permute(0,2,1).reshape(n, -1, c0_1.shape[2], c0_1.shape[3])
        _c0_2 = self.linear_c0(c0_2).permute(0,2,1).reshape(n, -1, c0_2.shape[2], c0_2.shape[3])
        # # _c0_1_CRA = self.CRA(_c0_1)
        # # _c0_2_CRA = self.CRA(_c0_2)
        # _c0 = self.diff_c0(torch.cat((_c0_1, _c0_2), dim=1))
        # _c0 = torch.cat((_c0,_c0_1_CRA,_c0_2_CRA),dim=1)
        # _c0 = self.Convd1_0(_c0)
        #_c0 = self.Convd1_0(_c0) + F.interpolate(_c1, scale_factor=4, mode="bilinear")       
        #_c0   = self.diff_c0(torch.cat((_c0_1, _c0_2), dim=1)) + F.interpolate(_c1, scale_factor=4, mode="bilinear")
        # p_c0  = self.make_pred_c0(_c0)
        # outputs.append(p_c0)
        #_c0_up = resize(_c0, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        _c5_1 = self.linear_c5(c5_1).permute(0,2,1).reshape(n, -1, c5_1.shape[2], c5_1.shape[3])
        _c5_2 = self.linear_c5(c5_2).permute(0,2,1).reshape(n, -1, c5_2.shape[2], c5_2.shape[3])
        # # _c0_1_CRA = self.CRA(_c0_1)
        # # _c0_2_CRA = self.CRA(_c0_2)
        # _c5 = self.diff_c0(torch.cat((_c5_1, _c5_2), dim=1))

        #FreqFusion模块
        # _, x3, x4_up = self.ff1(hr_feat=_c3, lr_feat=_c4)
        # cc1 = torch.cat([x3, x4_up],dim=1)
        # _, x2, x34_up = self.ff2(hr_feat=_c2, lr_feat= cc1)
        # cc2 = torch.cat([x2, x34_up],dim=1)
        # _, x1, x234_up = self.ff3(hr_feat=_c1, lr_feat=cc2)
        # cc3 = torch.cat([x1, x234_up],dim=1) # channel=4c, 1/4 img size
        # _, x0, x1234_up = self.ff4(hr_feat=_c0, lr_feat=cc3)
        # cc4 = torch.cat([x0, x1234_up],dim=1)
        # _, x5, x12345_up = self.ff5(hr_feat=_c5, lr_feat=cc4)
        # _c = torch.cat([x5, x12345_up],dim=1)
        # _c = self.Convd1_1(_c)


        #FreqFusion模块    2
        _, x3_1,x4_up_1 = self.ff1(hr_feat = _c3_1,lr_feat = _c4_1)
        cc1_1 = torch.cat([x3_1,x4_up_1],dim = 1)
        _, x2_1,x34_up_1 = self.ff2(hr_feat = _c2_1,lr_feat = cc1_1)
        cc1_2 = torch.cat([x2_1,x34_up_1],dim = 1)
        _, x1_1, x234_up_1 = self.ff3(hr_feat = _c1_1,lr_feat = cc1_2)
        cc1_3 = torch.cat([x1_1,x234_up_1],dim = 1)
        _, x0_1, x1234_up_1 = self.ff4(hr_feat = _c0_1,lr_feat = cc1_3)
        cc1_4 = torch.cat([x0_1,x1234_up_1],dim = 1)
        _, x5_1, x12345_up_1 = self.ff5(hr_feat = _c5_1,lr_feat = cc1_4)
        _c_1 = torch.cat([x5_1,x12345_up_1],dim = 1)                                #(2 768 256 256)
        _c_1 = self.Convd1_1(_c_1)
        _c1_1_up= resize(_c_1, size=cc1_3.size()[2:], mode='bilinear', align_corners=False)
        
        cc1_1_up= resize(cc1_1, size=cc1_3.size()[2:], mode='bilinear', align_corners=False) #256
        cc1_1_up= self.Convd1_2(cc1_1_up)
        cc1_2_up= resize(cc1_2, size=cc1_3.size()[2:], mode='bilinear', align_corners=False) #384
        cc1_2_up= self.Convd1_3(cc1_2_up)
        cc1_4_up= resize(cc1_4, size=cc1_3.size()[2:], mode='bilinear', align_corners=False) #640
        cc1_4_up= self.Convd1_4(cc1_4_up)
        cc1_3_up= self.Convd1_0(cc1_3)

        _, x3_2,x4_up_2 = self.ff1(hr_feat = _c3_2,lr_feat = _c4_2)
        cc2_1 = torch.cat([x3_2,x4_up_2],dim = 1)
        _, x2_2,x34_up_2 = self.ff2(hr_feat = _c2_2,lr_feat = cc2_1)
        cc2_2 = torch.cat([x2_2,x34_up_2],dim = 1)
        _, x1_2, x234_up_2 = self.ff3(hr_feat = _c1_2,lr_feat = cc2_2)
        cc2_3 = torch.cat([x1_2,x234_up_2],dim = 1)
        _, x0_2, x1234_up_2 = self.ff4(hr_feat = _c0_2,lr_feat = cc2_3)
        cc2_4 = torch.cat([x0_2,x1234_up_2],dim = 1)
        _, x5_2, x12345_up_2 = self.ff5(hr_feat = _c5_2,lr_feat = cc2_4)
        _c_2 = torch.cat([x5_2,x12345_up_2],dim = 1)
        _c_2 = self.Convd1_1(_c_2)
        _c2_1_up= resize(_c_2, size=cc2_3.size()[2:], mode='bilinear', align_corners=False)

        cc2_1_up= resize(cc2_1, size=cc2_3.size()[2:], mode='bilinear', align_corners=False)
        cc2_1_up= self.Convd1_2(cc2_1_up)
        cc2_2_up= resize(cc2_2, size=cc2_3.size()[2:], mode='bilinear', align_corners=False)
        cc2_2_up= self.Convd1_3(cc2_2_up)
        cc2_4_up= resize(cc2_4, size=cc2_3.size()[2:], mode='bilinear', align_corners=False)
        cc2_4_up= self.Convd1_4(cc2_4_up)
        cc2_3_up= self.Convd1_0(cc2_3)

        _ccc1= self.diff_c1(torch.cat((cc1_1_up,cc2_1_up),dim=1))
        _ccc2= self.diff_c2(torch.cat((cc1_2_up,cc2_2_up),dim=1))
        _ccc3= self.diff_c3(torch.cat((cc1_3_up,cc2_3_up),dim=1))
        _ccc4= self.diff_c4(torch.cat((cc1_4_up,cc2_4_up),dim=1))
        _c = self.diff_c0(torch.cat((_c1_1_up, _c2_1_up), dim=1))




        #Linear Fusion of difference image from all scales
        #_c = self.linear_fuse(torch.cat((_c4_up, _c3_up, _c2_up, _c1,_c0_up), dim=1))
        _f = self.linear_fuse(torch.cat((_ccc1, _ccc2, _ccc3, _ccc4,_c), dim=1))

        # #Dropout
        # if dropout_ratio > 0:
        #     self.dropout = nn.Dropout2d(dropout_ratio)
        # else:
        #     self.dropout = None

        #Upsampling x2 (x1/2 scale)
        x = self.convd2x(_f)
        # #Residual block
        x = self.dense_2x(x)
        # #Upsampling x2 (x1 scale)
        #x = self.convd1x(_c)
        x = self.convd1x(x)
        # # #Residual block
        x = self.dense_1x(x)

        #Final prediction
        cp = self.change_probability(x)
        
        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs



class Decoder_v5_SegFusion(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3,4,5], align_corners=True, 
                    in_channels = [32, 64, 128, 256], embedding_dim= 64, output_nc=2, 
                    decoder_softmax = False, feature_strides=[1,2, 4, 8, 16,32]):
        super(Decoder_v5_SegFusion, self).__init__()
        #assert
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        
        #settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index        = in_index
        self.align_corners   = align_corners
        self.in_channels     = in_channels
        self.embedding_dim   = 128
        self.output_nc       = output_nc
        # c0_in_channels,c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        c5_inchannels,c0_in_channels,c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        # c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        #MLP decoder heads
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)
        self.linear_c0 = MLP(input_dim=c0_in_channels, embed_dim=self.embedding_dim)
        self.linear_c5 = MLP(input_dim=c5_inchannels, embed_dim=self.embedding_dim)

        #convolutional Difference Modules
        self.diff_c4   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c0   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c5   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)

        #taking outputs from middle of the encoder
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c0 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c5 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        #Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(   in_channels=self.embedding_dim*len(in_channels), out_channels=self.embedding_dim,
                                        kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )

        #Final predction head
        self.convd2x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.convd1x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4,stride=2)
        self.dense_1x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        self.Convd11 = nn.Conv2d(128,128,kernel_size=1)
        
        #Final activation
        self.output_softmax     = decoder_softmax
        self.active             = nn.Sigmoid() 
        self.Convd1_4 = nn.Conv2d(768,256,kernel_size=1)
        self.Convd1_3 = nn.Conv2d(768,256,kernel_size=1)
        self.Convd1_2 = nn.Conv2d(768,256,kernel_size=1)
        self.Convd1_1 = nn.Conv2d(768,128,kernel_size=1)
        self.Convd1_0 = nn.Conv2d(768,256,kernel_size=1)
        self.CRA = ImprovedCRA(256,reduction_ratio=4)
        self.ff1 = FreqFusion(128, 128)
        self.ff2 = FreqFusion(128,256)
        self.ff3 = FreqFusion(128,384)
        self.ff4 = FreqFusion(128,512)
        self.ff5= FreqFusion(128,640)

    def _transform_inputs(self, inputs):                                        #对输入特征进行转换
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]
        return inputs

    def forward(self, inputs1, inputs2):
        '''处理输入数据'''
        #Transforming encoder features (select layers)
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16                 (16 64 64 64) --> (16 64 64 64)
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16

        #img1 and img2 features
        c5_1,c0_1,c1_1, c2_1, c3_1, c4_1 = x_1
        c5_2,c0_2,c1_2, c2_2, c3_2, c4_2 = x_2
        # c1_1, c2_1, c3_1, c4_1 = x_1
        # c1_2, c2_2, c3_2, c4_2 = x_2        
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        outputs = []
        # Stage 4: x1/32 scale    [B  256 8 8]
        _c4_1 = self.linear_c4(c4_1).permute(0,2,1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0,2,1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        # _c4_1_CRA = self.CRA(_c4_1)
        # _c4_2_CRA = self.CRA(_c4_2)
        _c4   = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))
        # _c4 = torch.cat((_c4,_c4_1_CRA,_c4_2_CRA),dim=1)
        # c4 = self.Convd1_4(_c4)
        # p_c4  = self.make_pred_c4(_c4)
        # outputs.append(p_c4)
        #_c4_up= resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0,2,1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        # _c3_1_CRA = self.CRA(_c3_1)
        # _c3_2_CRA = self.CRA(_c3_2)
        _c3 = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1))
        # _c3 = torch.cat((_c3,_c3_1_CRA,_c3_2_CRA),dim=1)
        # _c3 = self.Convd1_3(_c3)
        #_c3 = self.Convd1_3(_c3)+ F.interpolate(_c4, scale_factor=2, mode="bilinear")
        #_c3   = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1)) + F.interpolate(_c4, scale_factor=2, mode="bilinear")
        # p_c3  = self.make_pred_c3(_c3)
        # outputs.append(p_c3)
        #_c3_up= resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0,2,1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        # _c2_1_CRA = self.CRA(_c2_1)
        # _c2_2_CRA = self.CRA(_c2_2)
        _c2 = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1))
        # _c2 = torch.cat((_c2,_c2_1_CRA,_c2_2_CRA),dim=1)
        # _c2 = self.Convd1_2(_c2)
        #_c2 = self.Convd1_2(_c2) + F.interpolate(_c3, scale_factor=2, mode="bilinear")
        #_c2   = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1)) + F.interpolate(_c3, scale_factor=2, mode="bilinear")
        # p_c2  = self.make_pred_c2(_c2)
        # outputs.append(p_c2)
        #_c2_up= resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0,2,1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        # _c1_1_CRA = self.CRA(_c1_1)
        # _c1_2_CRA = self.CRA(_c1_2)
        _c1 = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1))
        # _c1 = torch.cat((_c1,_c1_1_CRA,_c1_2_CRA),dim=1)
        # _c1 = self.Convd1_1(_c1)
        #_c1 = self.Convd1_1(_c1) + F.interpolate(_c2, scale_factor=2, mode="bilinear")
        #_c1   = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1)) + F.interpolate(_c2, scale_factor=2, mode="bilinear")
        # p_c1  = self.make_pred_c1(_c1)
        # outputs.append(p_c1)
       
       # Stage 0: x1 scale
        _c0_1 = self.linear_c0(c0_1).permute(0,2,1).reshape(n, -1, c0_1.shape[2], c0_1.shape[3])
        _c0_2 = self.linear_c0(c0_2).permute(0,2,1).reshape(n, -1, c0_2.shape[2], c0_2.shape[3])
        # # _c0_1_CRA = self.CRA(_c0_1)
        # # _c0_2_CRA = self.CRA(_c0_2)
        _c0 = self.diff_c0(torch.cat((_c0_1, _c0_2), dim=1))
        # _c0 = torch.cat((_c0,_c0_1_CRA,_c0_2_CRA),dim=1)
        # _c0 = self.Convd1_0(_c0)
        #_c0 = self.Convd1_0(_c0) + F.interpolate(_c1, scale_factor=4, mode="bilinear")       
        #_c0   = self.diff_c0(torch.cat((_c0_1, _c0_2), dim=1)) + F.interpolate(_c1, scale_factor=4, mode="bilinear")
        # p_c0  = self.make_pred_c0(_c0)
        # outputs.append(p_c0)
        #_c0_up = resize(_c0, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        _c5_1 = self.linear_c5(c5_1).permute(0,2,1).reshape(n, -1, c5_1.shape[2], c5_1.shape[3])
        _c5_2 = self.linear_c5(c5_2).permute(0,2,1).reshape(n, -1, c5_2.shape[2], c5_2.shape[3])
        # # _c0_1_CRA = self.CRA(_c0_1)
        # # _c0_2_CRA = self.CRA(_c0_2)
        _c5 = self.diff_c0(torch.cat((_c5_1, _c5_2), dim=1))

        #FreqFusion模块
        _, x3, x4_up = self.ff1(hr_feat=_c3, lr_feat=_c4)
        cc1 = torch.cat([x3, x4_up],dim=1)
        _, x2, x34_up = self.ff2(hr_feat=_c2, lr_feat= cc1)
        cc2 = torch.cat([x2, x34_up],dim=1)
        _, x1, x234_up = self.ff3(hr_feat=_c1, lr_feat=cc2)
        cc3 = torch.cat([x1, x234_up],dim=1) # channel=4c, 1/4 img size
        _, x0, x1234_up = self.ff4(hr_feat=_c0, lr_feat=cc3)
        cc4 = torch.cat([x0, x1234_up],dim=1)
        _, x5, x12345_up = self.ff5(hr_feat=_c5, lr_feat=cc4)
        _c = torch.cat([x5, x12345_up],dim=1)
        _c = self.Convd1_1(_c)



        #Linear Fusion of difference image from all scales
        #_c = self.linear_fuse(torch.cat((_c4_up, _c3_up, _c2_up, _c1,_c0_up), dim=1))

        # #Dropout
        # if dropout_ratio > 0:
        #     self.dropout = nn.Dropout2d(dropout_ratio)
        # else:
        #     self.dropout = None

        #Upsampling x2 (x1/2 scale)
        # x = self.convd2x(_c)
        # #Residual block
        # x = self.dense_2x(x)
        # #Upsampling x2 (x1 scale)
        #x = self.convd1x(_c)
        x = self.Convd11(_c)
        # # #Residual block
        x = self.dense_1x(x)

        #Final prediction
        cp = self.change_probability(_c)
        
        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs





class Decoder_v4_UNetFusion(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3,4], align_corners=True, 
                    in_channels = [32, 64, 128, 256], embedding_dim= 256, output_nc=2, 
                    decoder_softmax = False, feature_strides=[2, 4, 8, 16,32]):
        super(Decoder_v4_UNetFusion, self).__init__()
        #assert
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        
        #settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index        = in_index
        self.align_corners   = align_corners
        self.in_channels     = in_channels
        self.embedding_dim   = embedding_dim
        self.output_nc       = output_nc
        c0_in_channels,c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        # c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        #MLP decoder heads
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)
        self.linear_c0 = MLP(input_dim=c0_in_channels, embed_dim=self.embedding_dim)

        #convolutional Difference Modules
        self.diff_c4   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c0   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)

        #taking outputs from middle of the encoder
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c0 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        #Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(   in_channels=self.embedding_dim*len(in_channels), out_channels=self.embedding_dim,
                                        kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )

        #Final predction head
        self.convd2x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.convd1x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        
        #Final activation
        self.output_softmax     = decoder_softmax
        self.active             = nn.Sigmoid() 
        self.Convd1_4 = nn.Conv2d(768,256,kernel_size=1)
        self.Convd1_3 = nn.Conv2d(768,256,kernel_size=1)
        self.Convd1_2 = nn.Conv2d(768,256,kernel_size=1)
        self.Convd1_1 = nn.Conv2d(1280,256,kernel_size=1)
        self.Conv = nn.Conv2d(512,256,kernel_size=1)
        self.CRA = ImprovedCRA(256,reduction_ratio=4)
        self.ff1 = FreqFusion(256, 256)
        self.ff2 = FreqFusion(256,256)
        self.ff3 = FreqFusion(256,256)
        self.ff4 = FreqFusion(256,256)

    def _transform_inputs(self, inputs):                                        #对输入特征进行转换
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]
        return inputs

    def forward(self, inputs1, inputs2):
        '''处理输入数据'''
        #Transforming encoder features (select layers)
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16                 (16 64 64 64) --> (16 64 64 64)
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16

        #img1 and img2 features
        c0_1,c1_1, c2_1, c3_1, c4_1 = x_1
        c0_2,c1_2, c2_2, c3_2, c4_2 = x_2
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        outputs = []
        # Stage 4: x1/32 scale
        _c4_1 = self.linear_c4(c4_1).permute(0,2,1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0,2,1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4   = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))

        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0,2,1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        _c3 = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1))

        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0,2,1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2 = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1))

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0,2,1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1 = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1))

       # Stage 0: x1 scale
        _c0_1 = self.linear_c0(c0_1).permute(0,2,1).reshape(n, -1, c0_1.shape[2], c0_1.shape[3])
        _c0_2 = self.linear_c0(c0_2).permute(0,2,1).reshape(n, -1, c0_2.shape[2], c0_2.shape[3])
        _c0 = self.diff_c0(torch.cat((_c0_1, _c0_2), dim=1))

        #FreqFusion模块__UNetFusion
        _, x3, x4_up = self.ff1(hr_feat=_c3, lr_feat=_c4)
        cc1 = self.Conv(torch.cat([x3, x4_up],dim=1))
        _, x2, x34_up = self.ff2(hr_feat=_c2, lr_feat= cc1)
        cc2 = self.Conv(torch.cat([x2, x34_up],dim=1))
        _, x1, x234_up = self.ff3(hr_feat=_c1, lr_feat=cc2)
        cc3 = self.Conv(torch.cat([x1, x234_up],dim=1))
        _, x0, x1234_up = self.ff4(hr_feat=_c0, lr_feat=cc3)                        # channel=c, 1/2 img size
        _c = self.Conv(torch.cat([x0, x1234_up],dim=1))
        #_c = self.Convd1_1(_c)
        



        #Linear Fusion of difference image from all scales
        #_c = self.linear_fuse(torch.cat((_c4_up, _c3_up, _c2_up, _c1,_c0_up), dim=1))

        # #Dropout
        # if dropout_ratio > 0:
        #     self.dropout = nn.Dropout2d(dropout_ratio)
        # else:
        #     self.dropout = None

        #Upsampling x2 (x1/2 scale)
        # x = self.convd2x(_c)
        # #Residual block
        # x = self.dense_2x(x)
        #Upsampling x2 (x1 scale)
        x = self.convd1x(_c)
        # #Residual block
        x = self.dense_1x(x)

        #Final prediction
        cp = self.change_probability(x)
        
        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs



class Decoder_v4_FpnFusion(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3,4], align_corners=True, 
                    in_channels = [32, 64, 128, 256], embedding_dim= 256, output_nc=2, 
                    decoder_softmax = False, feature_strides=[2, 4, 8, 16,32]):
        super(Decoder_v4_FpnFusion, self).__init__()
        #assert
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        
        #settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index        = in_index
        self.align_corners   = align_corners
        self.in_channels     = in_channels
        self.embedding_dim   = embedding_dim
        self.output_nc       = output_nc
        c0_in_channels,c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        # c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        #MLP decoder heads
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)
        self.linear_c0 = MLP(input_dim=c0_in_channels, embed_dim=self.embedding_dim)

        #convolutional Difference Modules
        self.diff_c4   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c0   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)

        #taking outputs from middle of the encoder
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c0 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        #Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(   in_channels=self.embedding_dim*len(in_channels), out_channels=self.embedding_dim,
                                        kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )

        #Final predction head
        self.convd2x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.convd1x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        
        #Final activation
        self.output_softmax     = decoder_softmax
        self.active             = nn.Sigmoid() 
        self.Convd1_4 = nn.Conv2d(768,256,kernel_size=1)
        self.Convd1_3 = nn.Conv2d(768,256,kernel_size=1)
        self.Convd1_2 = nn.Conv2d(768,256,kernel_size=1)
        self.Convd1_1 = nn.Conv2d(1280,256,kernel_size=1)
        self.Conv = nn.Conv2d(512,256,kernel_size=1)
        self.CRA = ImprovedCRA(256,reduction_ratio=4)
        self.ff1 = FreqFusion(256, 256)
        self.ff2 = FreqFusion(256,256)
        self.ff3 = FreqFusion(256,256)
        self.ff4 = FreqFusion(256,256)

    def _transform_inputs(self, inputs):                                        #对输入特征进行转换
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]
        return inputs

    def forward(self, inputs1, inputs2):
        '''处理输入数据'''
        #Transforming encoder features (select layers)
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16                 (16 64 64 64) --> (16 64 64 64)
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16

        #img1 and img2 features
        c0_1,c1_1, c2_1, c3_1, c4_1 = x_1
        c0_2,c1_2, c2_2, c3_2, c4_2 = x_2
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        outputs = []
        # Stage 4: x1/32 scale
        _c4_1 = self.linear_c4(c4_1).permute(0,2,1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0,2,1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4   = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))

        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0,2,1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        _c3 = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1))

        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0,2,1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2 = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1))

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0,2,1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1 = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1))

       # Stage 0: x1 scale
        _c0_1 = self.linear_c0(c0_1).permute(0,2,1).reshape(n, -1, c0_1.shape[2], c0_1.shape[3])
        _c0_2 = self.linear_c0(c0_2).permute(0,2,1).reshape(n, -1, c0_2.shape[2], c0_2.shape[3])
        _c0 = self.diff_c0(torch.cat((_c0_1, _c0_2), dim=1))

        #FreqFusion模块__UNetFusion
        _, x3, x4_up = self.ff1(hr_feat=_c3, lr_feat=_c4)
        cc1 = x3 + x4_up
        _, x2, x34_up = self.ff2(hr_feat=_c2, lr_feat= cc1)
        cc2 = x2 + x34_up
        _, x1, x234_up = self.ff3(hr_feat=_c1, lr_feat=cc2)
        cc3 = x1 + x234_up
        _, x0, x1234_up = self.ff4(hr_feat=_c0, lr_feat=cc3)
        _c = x0 + x1234_up                                                      # channel=c, 1/2 img size
        #_c = self.Convd1_1(_c)


        #Upsampling x2 (x1/2 scale)
        # x = self.convd2x(_c)
        # #Residual block
        # x = self.dense_2x(x)
        #Upsampling x2 (x1 scale)
        x = self.convd1x(_c)
        # #Residual block
        x = self.dense_1x(x)

        #Final prediction
        cp = self.change_probability(x)
        
        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs




class First_DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(First_DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        


    def forward(self, input):
        x = self.conv(input)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.Conv = nn.Sequential(
            MSDConv_SSFC(in_ch, out_ch, dilation=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            MSDConv_SSFC(out_ch, out_ch, dilation=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.Conv(input)

class SwinChangeFormer(nn.Module):
    '''已改成SegFusion的Decoder： Decoder_v4_SegFusion'''
    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=256,ratio=0.5):
        super(SwinChangeFormer, self).__init__()
        #Transformer Encoder
        self.embed_dims = [16,32,64, 128, 320, 512]
        self.depths     = [3, 3, 4, 3] #[3, 3, 6, 18, 3]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1 
        
        self.swin = SwinTransformer(
                        embed_dims=96,
                        depths=[2, 2, 6, 2],
                        num_heads=[3, 6, 12, 24],
                        window_size=7,
                        mlp_ratio=4,
                        qkv_bias=True,
                        qk_scale=None,
                        drop_rate=0.,
                        attn_drop_rate=0.,
                        drop_path_rate=0.2,
                        patch_norm=True,
                        out_indices=(0, 1, 2, 3),
                        with_cp=False,
                        convert_weights=True,
                        init_cfg=dict(type='Pretrained', checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'))
        self.channelmapper = ChannelMapper(
                in_channels=[32,64,96, 192, 384, 768],
                out_channels=[16,32,64,128,320,512],
                # in_channels=[96, 192, 384, 768],
                # out_channels=[64,128,320,512],                
                num_outs=5)
        self.ASB1 = Adaptive_Star_Block(96,192)
        self.ASB2 = Adaptive_Star_Block(192,384)
        self.ASB3 = Adaptive_Star_Block(384,768)
        self.CBAMLayer1 = CBAMLayer(96)
        self.CBAMLayer2 = CBAMLayer(192)
        self.CBAMLayer3 = CBAMLayer(384)
        self.CBAMLayer4 = CBAMLayer(768)

        self.CBAM_Attention0 =CBAM_Attention(32)
        self.CBAM_Attention1 =CBAM_Attention(96)
        self.CBAM_Attention2 =CBAM_Attention(192)
        self.CBAM_Attention3 =CBAM_Attention(384)
        self.CBAM_Attention4 =CBAM_Attention(768)

        # self.CBAMMultiScale1 =CBAMMultiScale(96)
        # self.CBAMMultiScale2 =CBAMMultiScale(192)
        # self.CBAMMultiScale3 =CBAMMultiScale(384)
        # self.CBAMMultiScale4 =CBAMMultiScale(768)
        # self.EMA1 = EMA(96)
        # self.EMA2 = EMA(192)
        # self.EMA3 = EMA(384)
        # self.EMA4 = EMA(768)

        self.DAFM1 =DAFM(96)
        self.DAFM0 =DAFM(64)
        self.DAFM2 =DAFM(192)
        self.DAFM3 =DAFM(384)
        self.DAFM4 =DAFM(768)
        self.DAFM5 =DAFM(32)
        #self.DAFM4 =DAFM(768)
        
        self.Conv1_1 = First_DoubleConv(3, int(64 * ratio))
        self.Conv1_2 = First_DoubleConv(3, int(64 * ratio))
        #2024.0111
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv2_1 = DoubleConv(int(64 * ratio), int(128 * ratio))
        #self.wf = Wf(32,96)
        self.FADConv1 = FADConv(in_channels=32,out_channels=96,kernel_size=3,stride=1) 
        self.FADConv2 = FADConv(in_channels=96,out_channels=192,kernel_size=3,stride=1) 
        self.FADConv3 = FADConv(in_channels=192,out_channels=384,kernel_size=3,stride=1) 
        self.FADConv4 = FADConv(in_channels=384,out_channels=768,kernel_size=3,stride=1)
        self.pooling1 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.pooling2 = nn.AvgPool2d(kernel_size=2, stride=2) 
        self.sigmoid = nn.Sigmoid()
        self.dd = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        #Transformer Decoder
        # self.TDec_x2   = DecoderTransformer_v3(input_transform='multiple_select', in_index=[0, 1, 2, 3, 4, 5], align_corners=False, 
        #             in_channels = self.embed_dims, embedding_dim= self.embedding_dim, output_nc=output_nc, 
        #             decoder_softmax = decoder_softmax, feature_strides=[1,2, 4, 8, 16, 32])
        

        self.TDec_x2_V2   = Decoder_v4_SegFusion(input_transform='multiple_select', in_index=[0, 1, 2, 3,4,5], align_corners=False, 
                    in_channels = self.embed_dims, embedding_dim= self.embedding_dim, output_nc=output_nc, 
                    decoder_softmax = decoder_softmax, feature_strides=[1,2, 4, 8, 16,32])

    def forward(self, x1, x2):
        #x1 = self.FADConv(x1)               ### 尝试先对输入图像加入FADConv提取频域特征 ###
        #x2 = self.FADConv(x2)
        x1 = self.Conv1_1(x1)
        x2 = self.Conv1_2(x2)
        # x1_temp_1 = self.dd(x1)
        x1_temp_1 = x1
        
        # x1_temp_2 = self.Conv2_1(x1_temp_2)             #输出：(B, , , )
        #Wf1_1 = self.FADConv(x1_temp_2)

        # x2_temp_1 = self.dd(x2)
        x2_temp_1 = x2
        # x2_temp_2 = self.Maxpool(x2_temp_1)
        
        #Wf2_1 = self.FADConv(x2_temp_1)


        x1 = self.swin(x1)                          #x1 x2 为列表格式，含四个尺度的元素[96 192 384 768]
        x2 = self.swin(x2)
        x1_temp_2 = self.Maxpool(x1_temp_1)
        x1_temp_2 = self.Conv2_1(x1_temp_2)
        x2_temp_2 = self.Maxpool(x2_temp_1)
        x2_temp_2 = self.Conv2_1(x2_temp_2)
        # # Wf1_2 = self.FADConv(x1[0])
        # # Wf1_3 = self.FADConv(x1[1])
        # # Wf1_4 = self.FADConv(x1[2])
        # # Wf2_2 = self.FADConv(x2[0])
        # # Wf2_3 = self.FADConv(x2[1])
        # # Wf2_4 = self.FADConv(x2[2])
        # wf1_1 = self.FADConv1(x1_temp_1)
        # wf1_1 = self.pooling2(wf1_1)
        # #wf1_1 = self.sigmoid(wf1_1)
        # wf1_2 = self.FADConv2(x1[0])
        # wf1_2 = self.pooling2(wf1_2)
        # #wf1_2 = self.sigmoid(wf1_2)
        # wf1_3 = self.FADConv3(x1[1])
        # wf1_3 = self.pooling2(wf1_3)
        # #wf1_3 = self.sigmoid(wf1_3)        
        # wf1_4 = self.FADConv4(x1[2])
        # wf1_4 = self.pooling2(wf1_4)
        # #wf1_4 = self.sigmoid(wf1_4)

        # wf2_1 = self.FADConv1(x2_temp_1)
        # wf2_1 = self.pooling2(wf2_1)
        # #wf2_1 = self.sigmoid(wf2_1)
        # wf2_2 = self.FADConv2(x2[0])
        # wf2_2 = self.pooling2(wf2_2)
        # #wf2_2 = self.sigmoid(wf2_2)
        # wf2_3 = self.FADConv3(x2[1])
        # wf2_3 = self.pooling2(wf2_3)
        # #wf2_3 = self.sigmoid(wf2_3)        
        # wf2_4 = self.FADConv4(x2[2])
        # wf2_4 = self.pooling2(wf2_4)
        # #wf2_4 = self.sigmoid(wf2_4)
        # x1[0] = x1[0] + x1[0] * wf1_1
        # x2[0] = x2[0] + x2[0] * wf2_1
        # x1[1] = x1[1] + x1[1] * wf1_2
        # x2[1] = x2[1] + x2[1] * wf2_2
        # x1[2] = x1[2] + x1[2] * wf1_3
        # x2[2] = x2[2] + x2[2] * wf2_3
        # x1[3] = x1[3] + x1[3] * wf1_4
        # x2[3] = x2[3] + x2[3] * wf2_4

        x1.insert(0,x1_temp_2)
        x2.insert(0,x2_temp_2)

        # x1.insert(0,x1_temp_1)
        # x2.insert(0,x2_temp_1)
        # x1_temp = [0]*4
        # x2_temp = [0]*4
        # x1[0] = self.ASB1(x1[0]) + x1[0]          #对于自适应星操作：使用残差连接
        # x1[1] = self.ASB2(x1[1]) + x1[1]
        # x1[2] = self.ASB3(x1[2]) + x1[2]
        # x1[3] = self.ASB4(x1[3]) + x1[3]
        # x2[0] = self.ASB1(x2[0]) + x2[0]
        # x2[1] = self.ASB2(x2[1]) + x2[1]
        # x2[2] = self.ASB3(x2[2]) + x2[2]
        # x2[3] = self.ASB4(x2[3]) + x2[3]
        #x1[0] = self.ASB1(x1[0])                    #对于自适应星操作：不使用残差连接
        # x1_temp[1] = self.CBAM_Attention2(x1[1])
        # x1_temp[2] = self.CBAM_Attention3(x1[2])
        # x1_temp[3] = self.CBAM_Attention4(x1[3])
        # x2_temp[1] = self.CBAM_Attention2(x2[1])
        # x2_temp[2] = self.CBAM_Attention3(x2[2])
        # x2_temp[3] = self.CBAM_Attention4(x2[3])
        # x1[1] = self.ASB1(x1[0],x1[1])
        # x1[2] = self.ASB2(x1[1],x1[2])
        # x1[3] = self.ASB3(x1[2],x1[3])
        # x2[1] = self.ASB1(x2[0],x2[1])
        # x2[2] = self.ASB2(x2[1],x2[2])
        # x2[3] = self.ASB3(x2[2],x2[3])
        # x1[0] = self.CBAMLayer1(x1[0])            #CBAM层操作
        # x1[1] = self.CBAMLayer2(x1[1])
        # x1[2] = self.CBAMLayer3(x1[2])
        # x1[3] = self.CBAMLayer4(x1[3])
        # x2[0] = self.CBAMLayer1(x2[0])
        # x2[1] = self.CBAMLayer2(x2[1])
        # x2[2] = self.CBAMLayer3(x2[2])
        # # x2[3] = self.CBAMLayer4(x2[3])
        # x1[0] = self.CBAM_Attention0(x1[0])               #CBAM注意力操作
        # x1[1] = self.CBAM_Attention1(x1[1])
        # x1[2] = self.CBAM_Attention2(x1[2])
        # x1[3] = self.CBAM_Attention3(x1[3])
        # x1[4] = self.CBAM_Attention4(x1[4])
        # x2[0] = self.CBAM_Attention0(x2[0])
        # x2[1] = self.CBAM_Attention1(x2[1])
        # x2[2] = self.CBAM_Attention2(x2[2])
        # x2[3] = self.CBAM_Attention3(x2[3])
        # x2[4] = self.CBAM_Attention4(x2[4])
                
        x1[0] = self.DAFM5(x1[0])                           #CDAFM注意力操作
        x1[1] = self.DAFM0(x1[1])
        x1[2] = self.DAFM1(x1[2])
        x1[3] = self.DAFM2(x1[3])
        x1[4] = self.DAFM3(x1[4])
        # x1[5] = self.DAFM4(x1[5])
        x2[0] = self.DAFM5(x2[0])
        x2[1] = self.DAFM0(x2[1])
        x2[2] = self.DAFM1(x2[2])
        x2[3] = self.DAFM2(x2[3])
        x2[4] = self.DAFM3(x2[4])
        # x2[5] = self.DAFM4(x2[5])

        [fx1, fx2] = [self.channelmapper(x1), self.channelmapper(x2)]





        # x1[0] = self.CBAMMultiScale1(x1[0])            #CBAM层操作
        # x1[1] = self.CBAMMultiScale2(x1[1])
        # x1[2] = self.CBAMMultiScale3(x1[2])
        # x1[3] = self.CBAMMultiScale4(x1[3])
        # x2[0] = self.CBAMMultiScale1(x2[0])
        # x2[1] = self.CBAMMultiScale2(x2[1])
        # x2[2] = self.CBAMMultiScale3(x2[2])
        # x2[3] = self.CBAMMultiScale4(x2[3])
        # x1[0] = self.EMA1(x1[0])                          #EAM层操作
        # x1[1] = self.EMA2(x1[1])
        # x1[2] = self.EMA3(x1[2])
        # x1[3] = self.EMA4(x1[3])
        # x2[0] = self.EMA1(x2[0])
        # x2[1] = self.EMA2(x2[1])
        # x2[2] = self.EMA3(x2[2])
        # x2[3] = self.EMA4(x2[3])


        cp = self.TDec_x2_V2(fx1, fx2)

        # # Save to mat
        # save_to_mat(x1, x2, fx1, fx2, cp, "ChangeFormerV4")

        # exit()
        return cp
    


    

class SwinChangeFormerV2(nn.Module):
    '''设置为Decoder5输入，UNet特征融合'''
    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=256,ratio=0.5):
        super(SwinChangeFormerV2, self).__init__()
        #Transformer Encoder
        self.embed_dims = [32,64, 128, 320, 512]
        self.depths     = [3, 3, 4, 3] #[3, 3, 6, 18, 3]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1 
        
        self.swin = SwinTransformer(
                        embed_dims=96,
                        depths=[2, 2, 6, 2],
                        num_heads=[3, 6, 12, 24],
                        window_size=7,
                        mlp_ratio=4,
                        qkv_bias=True,
                        qk_scale=None,
                        drop_rate=0.,
                        attn_drop_rate=0.,
                        drop_path_rate=0.2,
                        patch_norm=True,
                        out_indices=(0, 1, 2, 3),
                        with_cp=False,
                        convert_weights=True,
                        init_cfg=dict(type='Pretrained', checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'))
        self.channelmapper = ChannelMapper(
                in_channels=[64,96, 192, 384, 768],
                out_channels=[32,64,128,320,512],
                # in_channels=[96, 192, 384, 768],
                # out_channels=[64,128,320,512],                
                num_outs=5)
        self.ASB1 = Adaptive_Star_Block(96,192)
        self.ASB2 = Adaptive_Star_Block(192,384)
        self.ASB3 = Adaptive_Star_Block(384,768)
        self.CBAMLayer1 = CBAMLayer(96)
        self.CBAMLayer2 = CBAMLayer(192)
        self.CBAMLayer3 = CBAMLayer(384)
        self.CBAMLayer4 = CBAMLayer(768)

        self.CBAM_Attention0 =CBAM_Attention(32)
        self.CBAM_Attention1 =CBAM_Attention(96)
        self.CBAM_Attention2 =CBAM_Attention(192)
        self.CBAM_Attention3 =CBAM_Attention(384)
        self.CBAM_Attention4 =CBAM_Attention(768)

        self.DAFM1 =DAFM(96)
        self.DAFM0 =DAFM(64)
        self.DAFM2 =DAFM(192)
        self.DAFM3 =DAFM(384)
        self.DAFM4 =DAFM(768)
        #self.DAFM4 =DAFM(768)
        
        self.Conv1_1 = First_DoubleConv(3, int(64 * ratio))
        self.Conv1_2 = First_DoubleConv(3, int(64 * ratio))
        #2024.0111
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv2_1 = DoubleConv(int(64 * ratio), int(128 * ratio))
        #self.wf = Wf(32,96)
        self.FADConv1 = FADConv(in_channels=32,out_channels=96,kernel_size=3,stride=1) 
        self.FADConv2 = FADConv(in_channels=96,out_channels=192,kernel_size=3,stride=1) 
        self.FADConv3 = FADConv(in_channels=192,out_channels=384,kernel_size=3,stride=1) 
        self.FADConv4 = FADConv(in_channels=384,out_channels=768,kernel_size=3,stride=1)
        self.pooling1 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.pooling2 = nn.AvgPool2d(kernel_size=2, stride=2) 
        self.sigmoid = nn.Sigmoid()
        self.dd = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        #Transformer Decoder
        self.TDec_x2_V2   = Decoder_v4_UNetFusion(input_transform='multiple_select', in_index=[0, 1, 2, 3,4], align_corners=False, 
                    in_channels = self.embed_dims, embedding_dim= self.embedding_dim, output_nc=output_nc, 
                    decoder_softmax = decoder_softmax, feature_strides=[2, 4, 8, 16,32])

    def forward(self, x1, x2):
        x1 = self.Conv1_1(x1)
        x2 = self.Conv1_2(x2)
        #x1_temp_1 = self.dd(x1)
        x1_temp_1 = x1
        
        # x1_temp_2 = self.Conv2_1(x1_temp_2)             #输出：(B, , , )
        #Wf1_1 = self.FADConv(x1_temp_2)

        #x2_temp_1 = self.dd(x2)
        x2_temp_1 = x2
        # x2_temp_2 = self.Maxpool(x2_temp_1)
        
        #Wf2_1 = self.FADConv(x2_temp_1)


        x1 = self.swin(x1)                          #x1 x2 为列表格式，含四个尺度的元素[96 192 384 768]
        x2 = self.swin(x2)
        x1_temp_2 = self.Maxpool(x1_temp_1)
        x1_temp_2 = self.Conv2_1(x1_temp_2)
        x2_temp_2 = self.Maxpool(x2_temp_1)
        x2_temp_2 = self.Conv2_1(x2_temp_2)
        x1.insert(0,x1_temp_2)
        x2.insert(0,x2_temp_2)
                
        x1[0] = self.DAFM0(x1[0])                           #CDAFM注意力操作
        x1[1] = self.DAFM1(x1[1])
        x1[2] = self.DAFM2(x1[2])
        x1[3] = self.DAFM3(x1[3])
        x1[4] = self.DAFM4(x1[4])
        # #x1[5] = self.DAFM5(x1[5])
        x2[0] = self.DAFM0(x2[0])
        x2[1] = self.DAFM1(x2[1])
        x2[2] = self.DAFM2(x2[2])
        x2[3] = self.DAFM3(x2[3])
        x2[4] = self.DAFM4(x2[4])
        #x2[5] = self.DAFM5(x2[5])

        [fx1, fx2] = [self.channelmapper(x1), self.channelmapper(x2)]

        cp = self.TDec_x2_V2(fx1, fx2)

        # # Save to mat
        # save_to_mat(x1, x2, fx1, fx2, cp, "ChangeFormerV4")

        # exit()
        return cp
    



class SwinChangeFormerV3(nn.Module):
    '''设置为Decoder5输入，FPN特征融合'''
    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=256,ratio=0.5):
        super(SwinChangeFormerV3, self).__init__()
        #Transformer Encoder
        self.embed_dims = [32,64, 128, 320, 512]
        self.depths     = [3, 3, 4, 3] #[3, 3, 6, 18, 3]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1 
        
        self.swin = SwinTransformer(
                        embed_dims=96,
                        depths=[2, 2, 6, 2],
                        num_heads=[3, 6, 12, 24],
                        window_size=7,
                        mlp_ratio=4,
                        qkv_bias=True,
                        qk_scale=None,
                        drop_rate=0.,
                        attn_drop_rate=0.,
                        drop_path_rate=0.2,
                        patch_norm=True,
                        out_indices=(0, 1, 2, 3),
                        with_cp=False,
                        convert_weights=True,
                        init_cfg=dict(type='Pretrained', checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'))
        self.channelmapper = ChannelMapper(
                in_channels=[64,96, 192, 384, 768],
                out_channels=[32,64,128,320,512],
                # in_channels=[96, 192, 384, 768],
                # out_channels=[64,128,320,512],                
                num_outs=5)
        self.ASB1 = Adaptive_Star_Block(96,192)
        self.ASB2 = Adaptive_Star_Block(192,384)
        self.ASB3 = Adaptive_Star_Block(384,768)
        self.CBAMLayer1 = CBAMLayer(96)
        self.CBAMLayer2 = CBAMLayer(192)
        self.CBAMLayer3 = CBAMLayer(384)
        self.CBAMLayer4 = CBAMLayer(768)

        self.CBAM_Attention0 =CBAM_Attention(32)
        self.CBAM_Attention1 =CBAM_Attention(96)
        self.CBAM_Attention2 =CBAM_Attention(192)
        self.CBAM_Attention3 =CBAM_Attention(384)
        self.CBAM_Attention4 =CBAM_Attention(768)

        self.DAFM1 =DAFM(96)
        self.DAFM0 =DAFM(64)
        self.DAFM2 =DAFM(192)
        self.DAFM3 =DAFM(384)
        self.DAFM4 =DAFM(768)
        #self.DAFM4 =DAFM(768)
        
        self.Conv1_1 = First_DoubleConv(3, int(64 * ratio))
        self.Conv1_2 = First_DoubleConv(3, int(64 * ratio))
        #2024.0111
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv2_1 = DoubleConv(int(64 * ratio), int(128 * ratio))
        #self.wf = Wf(32,96)
        self.FADConv1 = FADConv(in_channels=32,out_channels=96,kernel_size=3,stride=1) 
        self.FADConv2 = FADConv(in_channels=96,out_channels=192,kernel_size=3,stride=1) 
        self.FADConv3 = FADConv(in_channels=192,out_channels=384,kernel_size=3,stride=1) 
        self.FADConv4 = FADConv(in_channels=384,out_channels=768,kernel_size=3,stride=1)
        self.pooling1 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.pooling2 = nn.AvgPool2d(kernel_size=2, stride=2) 
        self.sigmoid = nn.Sigmoid()
        self.dd = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        #Transformer Decoder
        # self.TDec_x2   = DecoderTransformer_v3(input_transform='multiple_select', in_index=[0, 1, 2, 3, 4, 5], align_corners=False, 
        #             in_channels = self.embed_dims, embedding_dim= self.embedding_dim, output_nc=output_nc, 
        #             decoder_softmax = decoder_softmax, feature_strides=[1,2, 4, 8, 16, 32])
        

        self.TDec_x2_V2   = Decoder_v4_FpnFusion(input_transform='multiple_select', in_index=[0, 1, 2, 3,4], align_corners=False, 
                    in_channels = self.embed_dims, embedding_dim= self.embedding_dim, output_nc=output_nc, 
                    decoder_softmax = decoder_softmax, feature_strides=[2, 4, 8, 16,32])

    def forward(self, x1, x2):
        x1 = self.Conv1_1(x1)
        x2 = self.Conv1_2(x2)
        #x1_temp_1 = self.dd(x1)
        x1_temp_1 = x1
        
        # x1_temp_2 = self.Conv2_1(x1_temp_2)             #输出：(B, , , )
        #Wf1_1 = self.FADConv(x1_temp_2)

        #x2_temp_1 = self.dd(x2)
        x2_temp_1 = x2
        # x2_temp_2 = self.Maxpool(x2_temp_1)
        
        #Wf2_1 = self.FADConv(x2_temp_1)


        x1 = self.swin(x1)                          #x1 x2 为列表格式，含四个尺度的元素[96 192 384 768]
        x2 = self.swin(x2)
        x1_temp_2 = self.Maxpool(x1_temp_1)
        x1_temp_2 = self.Conv2_1(x1_temp_2)
        x2_temp_2 = self.Maxpool(x2_temp_1)
        x2_temp_2 = self.Conv2_1(x2_temp_2)
        x1.insert(0,x1_temp_2)
        x2.insert(0,x2_temp_2)
                
        x1[0] = self.DAFM0(x1[0])                           #CDAFM注意力操作
        x1[1] = self.DAFM1(x1[1])
        x1[2] = self.DAFM2(x1[2])
        x1[3] = self.DAFM3(x1[3])
        x1[4] = self.DAFM4(x1[4])
        # #x1[5] = self.DAFM5(x1[5])
        x2[0] = self.DAFM0(x2[0])
        x2[1] = self.DAFM1(x2[1])
        x2[2] = self.DAFM2(x2[2])
        x2[3] = self.DAFM3(x2[3])
        x2[4] = self.DAFM4(x2[4])
        #x2[5] = self.DAFM5(x2[5])

        [fx1, fx2] = [self.channelmapper(x1), self.channelmapper(x2)]

        cp = self.TDec_x2_V2(fx1, fx2)

        # # Save to mat
        # save_to_mat(x1, x2, fx1, fx2, cp, "ChangeFormerV4")

        # exit()
        return cp
    





class SwinChangeFormerV4(nn.Module):

    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=256,ratio=0.5):
        super(SwinChangeFormerV4, self).__init__()
        #Transformer Encoder
        self.embed_dims = [16,32,64, 128, 320, 512]
        self.depths     = [3, 3, 4, 3] #[3, 3, 6, 18, 3]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1 
        
        self.swin = SwinTransformer(
                        embed_dims=96,
                        depths=[2, 2, 6, 2],
                        num_heads=[3, 6, 12, 24],
                        window_size=7,
                        mlp_ratio=4,
                        qkv_bias=True,
                        qk_scale=None,
                        drop_rate=0.,
                        attn_drop_rate=0.,
                        drop_path_rate=0.2,
                        patch_norm=True,
                        out_indices=(0, 1, 2, 3),
                        with_cp=False,
                        convert_weights=True,
                        init_cfg=dict(type='Pretrained', checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'))
        self.channelmapper = ChannelMapper(
                in_channels=[32,64,96, 192, 384, 768],
                out_channels=[16,32,64,128,320,512],
                # in_channels=[96, 192, 384, 768],
                # out_channels=[64,128,320,512],                
                num_outs=5)
        self.ASB1 = Adaptive_Star_Block(96,192)
        self.ASB2 = Adaptive_Star_Block(192,384)
        self.ASB3 = Adaptive_Star_Block(384,768)
        self.CBAMLayer1 = CBAMLayer(96)
        self.CBAMLayer2 = CBAMLayer(192)
        self.CBAMLayer3 = CBAMLayer(384)
        self.CBAMLayer4 = CBAMLayer(768)

        self.CBAM_Attention0 =CBAM_Attention(32)
        self.CBAM_Attention1 =CBAM_Attention(96)
        self.CBAM_Attention2 =CBAM_Attention(192)
        self.CBAM_Attention3 =CBAM_Attention(384)
        self.CBAM_Attention4 =CBAM_Attention(768)

        self.DAFM1 =DAFM(96)
        self.DAFM0 =DAFM(64)
        self.DAFM2 =DAFM(192)
        self.DAFM3 =DAFM(384)
        self.DAFM4 =DAFM(768)
        self.DAFM5 =DAFM(32)
        #self.DAFM4 =DAFM(768)
        
        self.Conv1_1 = First_DoubleConv(3, int(64 * ratio))
        self.Conv1_2 = First_DoubleConv(3, int(64 * ratio))
        #2024.0111
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv2_1 = DoubleConv(int(64 * ratio), int(128 * ratio))
        #self.wf = Wf(32,96)
        self.FADConv1 = FADConv(in_channels=32,out_channels=96,kernel_size=3,stride=1) 
        self.FADConv2 = FADConv(in_channels=96,out_channels=192,kernel_size=3,stride=1) 
        self.FADConv3 = FADConv(in_channels=192,out_channels=384,kernel_size=3,stride=1) 
        self.FADConv4 = FADConv(in_channels=384,out_channels=768,kernel_size=3,stride=1)
        self.pooling1 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.pooling2 = nn.AvgPool2d(kernel_size=2, stride=2) 
        self.sigmoid = nn.Sigmoid()
        self.dd = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)

        self.TDec_x2_V2   = Decoder_v5_SegFusion(input_transform='multiple_select', in_index=[0, 1, 2, 3,4,5], align_corners=False, 
                    in_channels = self.embed_dims, embedding_dim= self.embedding_dim, output_nc=output_nc, 
                    decoder_softmax = decoder_softmax, feature_strides=[1,2, 4, 8, 16,32])
        self.BNN = nn.BatchNorm2d(32)
        self.BNN2 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.C1 = nn.Conv2d(32,32,kernel_size=1)
        self.C2 = nn.Conv2d(64,64,kernel_size=1)
    def forward(self, x1, x2):
        x1 = self.Conv1_1(x1)
        x2 = self.Conv1_2(x2)
        # x1_temp_1 = self.dd(x1)
        x1_temp_1 = x1
        x1_temp_1 = self.BNN(x1_temp_1)
        x1_temp_1 = self.relu1(x1_temp_1)
        #x1_temp_1 = self.C1(x1_temp_1)
        # x1_temp_2 = self.Conv2_1(x1_temp_2)             #输出：(B, , , )

        # x2_temp_1 = self.dd(x2)
        x2_temp_1 = x2
        x2_temp_1 = self.BNN(x2_temp_1)
        x2_temp_1 = self.relu1(x2_temp_1)
        #x2_temp_1 = self.C1(x2_temp_1)
        # x2_temp_2 = self.Maxpool(x2_temp_1)
        
        x1 = self.swin(x1)                          #x1 x2 为列表格式，含四个尺度的元素[96 192 384 768]
        x2 = self.swin(x2)
        x1_temp_2 = self.Maxpool(x1_temp_1)
        x1_temp_2 = self.Conv2_1(x1_temp_2)
        x1_temp_2 = self.BNN2(x1_temp_2)
        x1_temp_2 = self.relu2(x1_temp_2)
        #x1_temp_2 = self.C2(x1_temp_2)
        x2_temp_2 = self.Maxpool(x2_temp_1)
        x2_temp_2 = self.Conv2_1(x2_temp_2)
        x2_temp_2 = self.BNN2(x2_temp_2)
        x2_temp_2 = self.relu2(x2_temp_2)
        x1.insert(0,x1_temp_2)
        x2.insert(0,x2_temp_2)
        x1.insert(0,x1_temp_1)
        x2.insert(0,x2_temp_1)
                
        # x1[0] = self.DAFM5(x1[0])                           #CDAFM注意力操作
        # x1[1] = self.DAFM0(x1[1])
        x1[2] = self.DAFM1(x1[2])
        x1[3] = self.DAFM2(x1[3])
        x1[4] = self.DAFM3(x1[4])
        x1[5] = self.DAFM4(x1[5])
        # x2[0] = self.DAFM5(x2[0])
        # x2[1] = self.DAFM0(x2[1])
        x2[2] = self.DAFM1(x2[2])
        x2[3] = self.DAFM2(x2[3])
        x2[4] = self.DAFM3(x2[4])
        x2[5] = self.DAFM4(x2[5])

        [fx1, fx2] = [self.channelmapper(x1), self.channelmapper(x2)]
        cp = self.TDec_x2_V2(fx1, fx2)

        # # Save to mat
        # save_to_mat(x1, x2, fx1, fx2, cp, "ChangeFormerV4")

        # exit()
        return cp
    





class SwinChangeFormerV5(nn.Module):

    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=256,ratio=0.5):
        super(SwinChangeFormerV5, self).__init__()
        #Transformer Encoder
        self.embed_dims = [32,64, 128, 320, 512]
        self.depths     = [3, 3, 4, 3] #[3, 3, 6, 18, 3]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1 
        
        self.swin = SwinTransformer(
                        embed_dims=96,
                        depths=[2, 2, 6, 2],
                        num_heads=[3, 6, 12, 24],
                        window_size=7,
                        mlp_ratio=4,
                        qkv_bias=True,
                        qk_scale=None,
                        drop_rate=0.,
                        attn_drop_rate=0.,
                        drop_path_rate=0.2,
                        patch_norm=True,
                        out_indices=(0, 1, 2, 3),
                        with_cp=False,
                        convert_weights=True,
                        init_cfg=dict(type='Pretrained', checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'))
        self.channelmapper = ChannelMapper(
                in_channels=[32,96, 192, 384, 768],
                out_channels=[32,64,128,320,512],
                # in_channels=[96, 192, 384, 768],
                # out_channels=[64,128,320,512],                
                num_outs=5)
        self.ASB1 = Adaptive_Star_Block(96,192)
        self.ASB2 = Adaptive_Star_Block(192,384)
        self.ASB3 = Adaptive_Star_Block(384,768)

        self.DAFM1 =DAFM(96)
        # self.DAFM0 =DAFM(64)
        self.DAFM2 =DAFM(192)
        self.DAFM3 =DAFM(384)
        self.DAFM4 =DAFM(768)
        # self.DAFM5 =DAFM(32)
        #self.DAFM4 =DAFM(768)
        
        self.Conv1_1 = First_DoubleConv(3, int(64 * ratio))
        self.Conv1_2 = First_DoubleConv(3, int(64 * ratio))
        #2024.0111
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv2_1 = DoubleConv(int(64 * ratio), int(128 * ratio))
        #self.wf = Wf(32,96)
        self.pooling1 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.pooling2 = nn.AvgPool2d(kernel_size=2, stride=2) 
        self.sigmoid = nn.Sigmoid()
        self.dd = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        

        self.TDec_x2_V2   = Decoder_v4_SegFusion(input_transform='multiple_select', in_index=[0, 1, 2, 3,4,5], align_corners=False, 
                    in_channels = self.embed_dims, embedding_dim= self.embedding_dim, output_nc=output_nc, 
                    decoder_softmax = decoder_softmax, feature_strides=[1,2, 4, 8, 16,32])

    def forward(self, x1, x2):
        #x1 = self.FADConv(x1)               ### 尝试先对输入图像加入FADConv提取频域特征 ###
        #x2 = self.FADConv(x2)
        x1 = self.Conv1_1(x1)
        x2 = self.Conv1_2(x2)
        # x1_temp_1 = self.dd(x1)
        x1_temp_1 = x1
        
        # x1_temp_2 = self.Conv2_1(x1_temp_2)             #输出：(B, , , )
        #Wf1_1 = self.FADConv(x1_temp_2)

        # x2_temp_1 = self.dd(x2)
        x2_temp_1 = x2
        # x2_temp_2 = self.Maxpool(x2_temp_1)
        
        #Wf2_1 = self.FADConv(x2_temp_1)


        x1 = self.swin(x1)                          #x1 x2 为列表格式，含四个尺度的元素[96 192 384 768]
        x2 = self.swin(x2)
        # x1_temp_2 = self.Maxpool(x1_temp_1)
        # x1_temp_2 = self.Conv2_1(x1_temp_2)
        # x2_temp_2 = self.Maxpool(x2_temp_1)
        # x2_temp_2 = self.Conv2_1(x2_temp_2)
        # x1.insert(0,x1_temp_2)
        # x2.insert(0,x2_temp_2)
        x1.insert(0,x1_temp_1)
        x2.insert(0,x2_temp_1)
                
        # x1[0] = self.DAFM5(x1[0])                           #DAFM注意力操作
        x1[1] = self.DAFM0(x1[1])
        x1[2] = self.DAFM1(x1[2])
        x1[3] = self.DAFM2(x1[3])
        x1[4] = self.DAFM3(x1[4])
        # x1[5] = self.DAFM4(x1[5])
        # x2[0] = self.DAFM5(x2[0])
        x2[1] = self.DAFM0(x2[1])
        x2[2] = self.DAFM1(x2[2])
        x2[3] = self.DAFM2(x2[3])
        x2[4] = self.DAFM3(x2[4])
        # x2[5] = self.DAFM4(x2[5])

        [fx1, fx2] = [self.channelmapper(x1), self.channelmapper(x2)]

        cp = self.TDec_x2_V2(fx1, fx2)
        return cp
    


class Decoder_v4_DCG(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3,4], align_corners=True, 
                    in_channels = [32, 64, 128, 256], embedding_dim= 64, output_nc=2, 
                    decoder_softmax = False, feature_strides=[2, 4, 8, 16,32]):
        super(Decoder_v4_DCG, self).__init__()
        #assert
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        
        #settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index        = in_index
        self.align_corners   = align_corners
        self.in_channels     = in_channels
        self.embedding_dim   = 128
        self.output_nc       = output_nc
        c0_in_channels,c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        # c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        #MLP decoder heads
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)
        self.linear_c0 = MLP(input_dim=c0_in_channels, embed_dim=self.embedding_dim)

        #convolutional Difference Modules
        self.diff_c4   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c0   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)

        #taking outputs from middle of the encoder
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c0 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        #Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(   in_channels=self.embedding_dim*len(in_channels), out_channels=self.embedding_dim,
                                        kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )

        #Final predction head
        self.convd2x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.convd1x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4,stride=2)
        self.dense_1x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        self.Convd11 = nn.Conv2d(128,128,kernel_size=1)
        
        #Final activation
        self.output_softmax     = decoder_softmax
        self.active             = nn.Sigmoid() 
        self.Convd1_4 = nn.Conv2d(768,256,kernel_size=1)
        self.Convd1_3 = nn.Conv2d(768,256,kernel_size=1)
        self.Convd1_2 = nn.Conv2d(768,256,kernel_size=1)
        self.Convd1_1 = nn.Conv2d(768,128,kernel_size=1)
        self.Convd1_0 = nn.Conv2d(768,256,kernel_size=1)
        self.CRA = ImprovedCRA(256,reduction_ratio=4)
        self.ff1 = FreqFusion(128, 128)
        self.ff2 = FreqFusion(128,256)
        self.ff3 = FreqFusion(128,384)
        self.ff4 = FreqFusion(128,512)
        self.ff5= FreqFusion(128,640)

    def _transform_inputs(self, inputs):                                        #对输入特征进行转换
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]
        return inputs

    def forward(self, inputs1, inputs2):
        '''处理输入数据'''
        #Transforming encoder features (select layers)
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16                 (16 64 64 64) --> (16 64 64 64)
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16

        #img1 and img2 features
        c0_1,c1_1, c2_1, c3_1, c4_1 = x_1
        c0_2,c1_2, c2_2, c3_2, c4_2 = x_2
        # c1_1, c2_1, c3_1, c4_1 = x_1
        # c1_2, c2_2, c3_2, c4_2 = x_2        
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        outputs = []
        # Stage 4: x1/32 scale    [B  256 8 8]
        _c4_1 = self.linear_c4(c4_1).permute(0,2,1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0,2,1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4   = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))
        # _c4 = torch.cat((_c4,_c4_1,_c4_2),dim=1)
        # # c4 = self.Convd1_4(_c4)
        # p_c4  = self.make_pred_c4(_c4)
        # outputs.append(p_c4)
        _c4_up= resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0,2,1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        _c3 = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1))
        _c3   = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1)) + F.interpolate(_c4, scale_factor=2, mode="bilinear")
        # p_c3  = self.make_pred_c3(_c3)
        # outputs.append(p_c3)
        _c3_up= resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0,2,1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2 = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1))
        _c2   = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1)) + F.interpolate(_c3, scale_factor=2, mode="bilinear")
        # p_c2  = self.make_pred_c2(_c2)
        # outputs.append(p_c2)
        _c2_up= resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0,2,1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])

        _c1 = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1))
        # _c1 = torch.cat((_c1,_c1_1_CRA,_c1_2_CRA),dim=1)
        # _c1 = self.Convd1_1(_c1)
        #_c1 = self.Convd1_1(_c1) + F.interpolate(_c2, scale_factor=2, mode="bilinear")
        #_c1   = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1)) + F.interpolate(_c2, scale_factor=2, mode="bilinear")
        # p_c1  = self.make_pred_c1(_c1)
        # outputs.append(p_c1)
       
       # Stage 0: x1 scale
        _c0_1 = self.linear_c0(c0_1).permute(0,2,1).reshape(n, -1, c0_1.shape[2], c0_1.shape[3])
        _c0_2 = self.linear_c0(c0_2).permute(0,2,1).reshape(n, -1, c0_2.shape[2], c0_2.shape[3])

        _c0 = self.diff_c0(torch.cat((_c0_1, _c0_2), dim=1))   
        _c0   = self.diff_c0(torch.cat((_c0_1, _c0_2), dim=1)) + F.interpolate(_c1, scale_factor=4, mode="bilinear")
        # p_c0  = self.make_pred_c0(_c0)
        # outputs.append(p_c0)
        _c0_up = resize(_c0, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        #FreqFusion模块
        
        #Linear Fusion of difference image from all scales
        _c = self.linear_fuse(torch.cat((_c4_up, _c3_up, _c2_up, _c1,_c0_up), dim=1))

        # #Dropout
        # if dropout_ratio > 0:
        #     self.dropout = nn.Dropout2d(dropout_ratio)
        # else:
        #     self.dropout = None

        #Upsampling x2 (x1/2 scale)
        x = self.convd2x(_c)
        # #Residual block
        x = self.dense_2x(x)
        # #Upsampling x2 (x1 scale)
        x = self.convd1x(x)
        # x = self.Convd11(_c)
        # # #Residual block
        x = self.dense_1x(x)

        #Final prediction
        cp = self.change_probability(_c)
        
        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs