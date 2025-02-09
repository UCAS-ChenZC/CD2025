# import torch
# from timm.models.vision_transformer import trunc_normal_

# # 论文题目：SHViT：具有内存高效宏设计的单头视觉Transformer
# # 中文题目：SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design
# # 论文链接：https://openaccess.thecvf.com/content/CVPR2024/papers/Yun_SHViT_Single-Head_Vision_Transformer_with_Memory_Efficient_Macro_Design_CVPR_2024_paper.pdf
# # 官方github：https://github.com/ysj9909/SHViT
# # 所属机构：韩国首尔大学机器智能实验室

# class GroupNorm(torch.nn.GroupNorm):
#     """
#     Group Normalization with 1 group.
#     Input: tensor in shape [B, C, H, W]
#     """
#     def __init__(self, num_channels, **kwargs):
#         super().__init__(1, num_channels, **kwargs)
# class Conv2d_BN(torch.nn.Sequential):
#     def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
#                  groups=1, bn_weight_init=1):
#         super().__init__()
#         self.add_module('c', torch.nn.Conv2d(
#             a, b, ks, stride, pad, dilation, groups, bias=False))
#         self.add_module('bn', torch.nn.BatchNorm2d(b))
#         torch.nn.init.constant_(self.bn.weight, bn_weight_init)
#         torch.nn.init.constant_(self.bn.bias, 0)
#     def fuse(self):
#         c, bn = self._modules.values()
#         w = bn.weight / (bn.running_var + bn.eps)**0.5
#         w = c.weight * w[:, None, None, None]
#         b = bn.bias - bn.running_mean * bn.weight / \
#             (bn.running_var + bn.eps)**0.5
#         m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
#             0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
#             device=c.weight.device)
#         m.weight.data.copy_(w)
#         m.bias.data.copy_(b)
#         return m
# class BN_Linear(torch.nn.Sequential):
#     def __init__(self, a, b, bias=True, std=0.02):
#         super().__init__()
#         self.add_module('bn', torch.nn.BatchNorm1d(a))
#         self.add_module('l', torch.nn.Linear(a, b, bias=bias))
#         trunc_normal_(self.l.weight, std=std)
#         if bias:
#             torch.nn.init.constant_(self.l.bias, 0)

#     def fuse(self):
#         bn, l = self._modules.values()
#         w = bn.weight / (bn.running_var + bn.eps)**0.5
#         b = bn.bias - self.bn.running_mean * \
#             self.bn.weight / (bn.running_var + bn.eps)**0.5
#         w = l.weight * w[None, :]
#         if l.bias is None:
#             b = b @ self.l.weight.T
#         else:
#             b = (l.weight @ b[:, None]).view(-1) + self.l.bias
#         m = torch.nn.Linear(w.size(1), w.size(0))
#         m.weight.data.copy_(w)
#         m.bias.data.copy_(b)
#         return m
    
# class SHSA(torch.nn.Module):
#     """Single-Head Self-Attention"""
#     def __init__(self, dim, qk_dim, pdim):
#         super().__init__()
#         self.scale = qk_dim ** -0.5
#         self.qk_dim = qk_dim
#         self.dim = dim
#         self.pdim = pdim
#         self.pre_norm = GroupNorm(pdim)
#         self.qkv = Conv2d_BN(pdim, qk_dim * 2 + pdim)
#         self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
#             dim, dim, bn_weight_init = 0))
        
#     def forward(self, x):
#         B, C, H, W = x.shape
#         x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim = 1)
#         x1 = self.pre_norm(x1)
#         qkv = self.qkv(x1)
#         q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim = 1)
#         q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        
#         attn = (q.transpose(-2, -1) @ k) * self.scale
#         attn = attn.softmax(dim = -1)
#         x1 = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
#         x = self.proj(torch.cat([x1, x2], dim = 1))
#         return x
    
# if __name__ == '__main__':
#     x = torch.randn(1,64,32,32)
#     shsa = SHSA(dim=64, qk_dim=64, pdim=64)
#     print(shsa)
#     output = shsa(x)
#     print(f"Input shape: {x.shape}")
#     print(f"output shape: {output.shape}")



"""
二、多层次特征融合
1. 缺陷：目前SCSA模块主要基于单一层级的空间和通道注意力信息

2. 解决方案：引入多层次特征融合，例如在不同网络层级上对特征图进行聚合，或者在不同分辨率的特征图之间实现交互。
通过融合低层次的细节信息和高层次的语义信息，增强模型对复杂结构和细节的捕捉能力。进一步提升图像分割、目标检测等
任务中细节处理和物体边界识别的效果。提高模块对多语义空间信息的利用效率，使得特征表达更加丰富。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# class DCG_AttnBlock(nn.Module):
#     def __init__(self, in_ch):
#         super().__init__()
#         if in_ch >= 32:
#             self.group_norm = nn.GroupNorm(32, in_ch)
#         else:
#             self.group_norm = nn.Identity()
#         # self.group_norm = nn.GroupNorm(32, in_ch)
#         self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
#         self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
#         self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
#         self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
#         self.initialize()

#     def initialize(self):
#         for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
#             init.xavier_uniform_(module.weight)
#             init.zeros_(module.bias)
#         init.xavier_uniform_(self.proj.weight, gain=1e-5)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         h = self.group_norm(x)
#         q = self.proj_q(h)
#         k = self.proj_k(h)
#         v = self.proj_v(h)

#         q = q.permute(0, 2, 3, 1).view(B, H * W, C)
#         k = k.view(B, C, H * W)
#         w = torch.bmm(q, k) * (int(C) ** (-0.5))
#         w = F.softmax(w, dim=-1)

#         v = v.permute(0, 2, 3, 1).view(B, H * W, C)
#         h = torch.bmm(w, v)
#         h = h.view(B, H, W, C).permute(0, 3, 1, 2)
#         h = self.proj(h)

#         return x + h

# class DCG_AttnBlock(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         if in_ch >= 32:
#             self.group_norm = nn.GroupNorm(32, out_ch)
#         else:
#             self.group_norm = nn.Identity()
#         self.dw_conv = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
#         # self.group_norm = nn.GroupNorm(32, in_ch)
#         self.proj_q = nn.Conv2d(out_ch, out_ch, 1, stride=1, padding=0)
#         self.proj_k = nn.Conv2d(out_ch, out_ch, 1, stride=1, padding=0)
#         self.proj_v = nn.Conv2d(out_ch, out_ch, 1, stride=1, padding=0)
#         self.proj = nn.Conv2d(out_ch, out_ch, 1, stride=1, padding=0)
#         self.initialize()

#     def initialize(self):
#         for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
#             init.xavier_uniform_(module.weight)
#             init.zeros_(module.bias)
#         init.xavier_uniform_(self.proj.weight, gain=1e-5)

#     def forward(self, x):
#         x = self.dw_conv(x)
#         B, C, H, W = x.shape
#         h = self.group_norm(x)
#         q = self.proj_q(h)
#         k = self.proj_k(h)
#         v = self.proj_v(h)

#         q = q.permute(0, 2, 3, 1).view(B, H * W, C)
#         k = k.view(B, C, H * W)
#         w = torch.bmm(q, k) * (int(C) ** (-0.5))
#         w = F.softmax(w, dim=-1)

#         v = v.permute(0, 2, 3, 1).view(B, H * W, C)
#         h = torch.bmm(w, v)
#         h = h.view(B, H, W, C).permute(0, 3, 1, 2)
#         h = self.proj(h)

#         return x + h

import einops
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np


 #########            添加DAttention        #####################
class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
 
    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')
class DAttention(nn.Module):
 
    def __init__(
            self, q_size=(224, 224), kv_size=(224, 224), n_heads=8, n_head_channels=32, n_groups=1,
            attn_drop=0.0, proj_drop=0.0, stride=1,
            offset_range_factor=-1, use_pe=True, dwc_pe=True,
            no_off=False, fixed_pe=False, ksize=9, log_cpb=False
    ):
 
        super().__init__()
        n_head_channels = int(q_size / 8)
        q_size = (q_size, q_size)
 
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        # self.kv_h, self.kv_w = kv_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = ksize
        self.log_cpb = log_cpb
        self.stride = stride
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0
 
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
 
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)
 
        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
 
        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0)
 
        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
 
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
 
        if self.use_pe and not self.no_off:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(
                    self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            elif self.log_cpb:
                # Borrowed from Swin-V2
                self.rpe_table = nn.Sequential(
                    nn.Linear(2, 32, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, self.n_group_heads, bias=False)
                )
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * 2 - 1, self.q_w * 2 - 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None
 
    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
 
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2
 
        return ref
 
    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):
 
        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2
 
        return ref
 
    def forward(self, x):
        x = x
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device
 
        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off).contiguous()  # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk
 
        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)
 
        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)
 
        if self.no_off:
            offset = offset.fill_(0.0)
 
        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).clamp(-1., +1.)
 
        if self.no_off:
            x_sampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
            assert x_sampled.size(2) == Hk and x_sampled.size(3) == Wk, f"Size is {x_sampled.size()}"
        else:
            x_sampled = F.grid_sample(
                input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
                grid=pos[..., (1, 0)],  # y, x -> x, y
                mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg
 
        x_sampled = x_sampled.reshape(B, C, 1, n_sample)
        # self.proj_k.weight = torch.nn.Parameter(self.proj_k.weight.float())
        # self.proj_k.bias = torch.nn.Parameter(self.proj_k.bias.float())
        # self.proj_v.weight = torch.nn.Parameter(self.proj_v.weight.float())
        # self.proj_v.bias = torch.nn.Parameter(self.proj_v.bias.float())
        # 检查权重的数据类型
        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
 
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
 
        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)
 
        if self.use_pe and (not self.no_off):
 
            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels,
                                                                              H * W)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, n_sample)
            elif self.log_cpb:
                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (
                        q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups,
                                                                                               n_sample,
                                                                                               2).unsqueeze(1)).mul(
                    4.0)  # d_y, d_x [-8, +8]
                displacement = torch.sign(displacement) * torch.log2(torch.abs(displacement) + 1.0) / np.log2(8.0)
                attn_bias = self.rpe_table(displacement)  # B * g, H * W, n_sample, h_g
                attn = attn + einops.rearrange(attn_bias, 'b m n h -> (b h) m n', h=self.n_group_heads)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (
                        q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups,
                                                                                               n_sample,
                                                                                               2).unsqueeze(1)).mul(
                    0.5)
                attn_bias = F.grid_sample(
                    input=einops.rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.n_group_heads,
                                           g=self.n_groups),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True)  # B * g, h_g, HW, Ns
 
                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
                attn = attn + attn_bias
 
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
 
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
 
        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)
 
        y = self.proj_drop(self.proj_out(out))
        h, w = pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2)
 
        return y
# if __name__ == '__main__':
#     input = torch.randn(2, 192, 64, 64).to("cuda")
#     model = DAttention(192,96).to("cuda")
#     output = model(input)
#     print('input_size:', input.size())
#     print('output_size:', output.size())



import torch
import torch.nn as nn
import math
# 论文题目：Unsupervised Bidirectional Contrastive Reconstruction and Adaptive Fine-Grained Channel Attention Networks for image dehazing
# 中文题目：无监督双向对比重建和自适应细粒度通道注意力网络用于图像去雾
# 论文链接：https://doi.org/10.1016/j.neunet.2024.106314
# 官方github：https://github.com/Lose-Code/UBRFC-Net
# 所属机构：三峡大学，华中科技大学同济医学院附属同济医院肿瘤科等
# 代码整理：微信公众号《AI缝合术》
class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()
    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out
class CCAttention(nn.Module):
    def __init__(self,channel,b=1, gamma=2):
        super(CCAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#全局平均池化
        #一维卷积
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()
    def forward(self, input):
        x = self.avg_pool(input)
        x1 = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)#(1,64,1)
        x2 = self.fc(x).squeeze(-1).transpose(-1, -2)#(1,1,64)
        out1 = torch.sum(torch.matmul(x1,x2),dim=1).unsqueeze(-1).unsqueeze(-1)#(1,64,1,1)
        #x1 = x1.transpose(-1, -2).unsqueeze(-1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2),x1.transpose(-1, -2)),dim=1).unsqueeze(-1).unsqueeze(-1)
        #out2 = self.fc(x)
        out2 = self.sigmoid(out2)
        out = self.mix(out1,out2)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        return input*out
# if __name__ == '__main__':
#     input = torch.rand(1,64,256,256)
#     A = CCAttention(channel=64)
#     print(A)
#     output = A(input)
#     print("input shape:", input.shape)
#     print("output shape:", output.shape)

import torch
import torch.nn as nn
from timm.models.layers import SqueezeExcite
"""
CV缝合救星魔改1：全局注意力RepVit
缺乏全局特征捕获能力:
当前的 RepViTBlock 模块主要依赖3x3的深度卷积（Depthwise Convolution）来处理空间信息，这种方法在捕获局部特征方面有效，但缺乏对图像整体
特征的感知能力。因为3x3卷积核只能看到局部区域，因此它难以捕获更远的像素之间的关系，从而在全局信息的整合上存在不足。

改进方法:为了提升全局特征捕获能力，在 RepViTBlock 中引入一个全局自注意力模块 GlobalAttention。该模块通过计算特征图中不同位置之间的关系，
生成全局的特征权重，从而提升模型对远距离依赖关系的建模能力。这种增强的全局信息可以帮助模型更好地理解场景中的大尺度特征，提高分类和分割任务
的准确性。
"""
# 定义带Batch Normalization的卷积层
class Conv2d_BN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(out_channels))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

# 全局自注意力模块
class GlobalAttention(nn.Module):
    def __init__(self, channels):
        super(GlobalAttention, self).__init__()
        self.query_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        attention = self.softmax(torch.bmm(query, key))
        value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch_size, C, width, height)
        return out + x  # 残差连接

# 带有全局注意力的RepViT块
class GlobalAttentionRepViTBlock(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=2, hidden_dim=80, use_se=True, use_hs=True):
        super(GlobalAttentionRepViTBlock, self).__init__()
        # Token Mixer: 3x3 DW卷积 + SE层 + 全局注意力
        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride=stride, padding=(kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                GlobalAttention(inp),  # 添加全局注意力
                Conv2d_BN(inp, oup, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride=1, padding=(kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                GlobalAttention(inp),  # 添加全局注意力
                Conv2d_BN(inp, oup, kernel_size=1, stride=1, padding=0)
            )
        # Channel Mixer
        self.channel_mixer = nn.Sequential(
            Conv2d_BN(oup, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GELU() if use_hs else nn.ReLU(),
            Conv2d_BN(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bn_weight_init=0),
        )

    def forward(self, x):
        x = self.token_mixer(x)
        x = self.channel_mixer(x)
        return x

# 测试代码
# if __name__ == '__main__':
#     input_tensor = torch.randn(2, 64, 128, 128)
#     model = GlobalAttentionRepViTBlock(inp=64, oup=64, kernel_size=3, stride=1)
#     output_tensor = model(input_tensor)
#     print('GlobalAttentionRepViTBlock - input size:', input_tensor.size())
#     print('GlobalAttentionRepViTBlock - output size:', output_tensor.size())

import torch
from torch import nn
"""
CV缝合救星魔改： GEMA（Global Enhanced Multi-scale Attention），全局增强多尺度注意力模块。
不足(research gap)：EMA 模块当前主要依赖于不同尺度的卷积（1×1 和 3×3）和全局平均池化来提取多尺度的特征。可以在设计中增加一个
自适应的全局上下文模块，用于捕获特征图中不同位置的长距离依赖关系，以便让模型能够更好地理解全局语义信息。

CV缝合救星魔改：
在 EMA 模块中加入一个自适应的全局上下文分支，利用1×1 卷积来对全局特征进行压缩。然后，采用扩展卷积（Dilated Convolution）
替代 3×3 卷积分支，进一步增强对局部和全局信息的提取能力。这样可以在不显著增加计算量的前提下，捕获更丰富的空间上下文信息。
"""

class EnhancedEMA(nn.Module):
    def __init__(self, channels, factor=8, dilation_rate=2):
        super(EnhancedEMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)

        # 原始 1x1 卷积，用于基本特征抽取
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)

        # 改进：使用扩展卷积代替原始的 3x3 卷积
        self.conv_dilated = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1,
                                      padding=dilation_rate, dilation=dilation_rate)

        # 新增的全局上下文卷积分支
        self.global_context_conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        b, c, h, w = x.size()

        # 1. 生成全局上下文特征
        global_context = self.global_context_conv(self.agp(x))  # b, c, 1, 1

        # 2. 分组特征计算
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g, c//g, h, w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)

        # 使用 1x1 卷积进行基础特征抽取
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        # 基于 1x1 和 Sigmoid 加权的特征
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())

        # 改进：使用扩展卷积替代原始 3x3 卷积
        x2 = self.conv_dilated(group_x)

        # 生成加权注意力
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)

        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)

        # 结合全局上下文特征，进行加权融合
        final_output = (group_x * weights.sigmoid()).reshape(b, c, h, w)
        return final_output + global_context.expand_as(final_output)  # 加入全局上下文特征


# 测试代码
if __name__ == '__main__':
    block = EnhancedEMA(64).cuda()
    input = torch.rand(1, 64, 64, 64).cuda()
    output = block(input)
    print('input_size:', input.size())
    print('output_size:', output.size())