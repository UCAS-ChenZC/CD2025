import torch.nn as nn
import torch
import torch.nn.functional as F
#from models.SHSA import DAttention,LayerNormProxy,CCAttention,Mix,GlobalAttentionRepViTBlock,GlobalAttention,Conv2d_BN
import torch.nn.init as init
import numpy as np
try:
    from torch import irfft
    from torch import rfft
except ImportError:
    def rfft(x, d):
        t = torch.fft.fft(x, dim=(-d))
        r = torch.stack((t.real, t.imag), -1)
        return r
    def irfft(x, d):
        t = torch.fft.ifft(torch.complex(x[:, :, 0], x[:, :, 1]), dim=(-d))
        return t.real
import math
class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
                    # Conv2d比Linear方便操作
                    # nn.Linear(channel, channel // reduction, bias=False)
                    nn.Conv2d(channel, channel // reduction, 1, bias=False),
                    # inplace=True直接替换，节省内存
                    nn.ReLU(inplace=True),
                    # nn.Linear(channel // reduction, channel,bias=False)
                    nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                                padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x
    


class ChannelAttention(nn.Module): #通道注意力机制
    def __init__(self, in_planes, scaling=16):#scaling为缩放比例，
        # 用来控制两个全连接层中间神经网络神经元的个数，一般设置为16，具体可以根据需要微调
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // scaling, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // scaling, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out

class SpatialAttention(nn.Module): #空间注意力机制
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return x

class CBAM_Attention(nn.Module):
    def __init__(self, channel, scaling=16, kernel_size=7):
        super(CBAM_Attention, self).__init__()
        self.channelattention = ChannelAttention(channel, scaling=scaling)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x
    

class DCG_AttnBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        if in_ch >= 32:
            self.group_norm = nn.GroupNorm(32, out_ch)
        else:
            self.group_norm = nn.Identity()
        self.dw_conv = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        self.proj_q = nn.Conv2d(out_ch, out_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(out_ch, out_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(out_ch, out_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(out_ch, out_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        x = self.dw_conv(x)
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h
    
class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class EEMA(nn.Module):
    def __init__(self, channels, factor=8, dilation_rate=2):
        super(EEMA, self).__init__()
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



class DAFM(nn.Module):
    '''Dual Attention Fusion Module'''
    def __init__(self, channel, scaling=16, kernel_size=7):
        super(DAFM, self).__init__()
        self.channelattention = ChannelAttention(channel, scaling=scaling)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)
        self.Conv3_1 = nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_features=channel),
                    nn.ReLU()
        )
        self.Conv3_2 = nn.Sequential(
                    nn.Conv2d(channel * 2, channel * 2, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_features=channel * 2),
                    nn.ReLU()
        )
        self.Conv3_3 = nn.Sequential(
                    nn.Conv2d(channel * 2, channel, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_features=channel),
                    nn.ReLU()
        )

        #2024.0110
        # self.DCG_attn_1 = DCG_AttnBlock(channel,channel)                            ### 较为有效 ###
        # self.DCG_attn_2 = DCG_AttnBlock(channel * 2,channel * 2)
        # self.DCG_attn_3 = DCG_AttnBlock(channel * 2,channel)
        # self.DAttention1 = DAttention(channel,channel)
        # self.DAttention2 = DAttention(channel * 2,channel * 2)
        # self.DAttention3 = DAttention(channel * 2,channel * 2)
        self.Conv1 = nn.Conv2d(channel * 2,channel * 2,kernel_size=1)
        self.conv1_2 = nn.Conv2d(channel * 2, channel, kernel_size=1)               #降低通道数
        # self.CCAttention1 = CCAttention(channel)
        # self.CCAttention2 = CCAttention(channel * 2)
        # self.CCAttention3= CCAttention(channel * 2)

        #2024.0111
        self.EMA = EMA(channel * 2)

        self.EEMA = EEMA(channel * 2)

    def forward(self,x):
        x1 = x
        x1 = x1 * self.channelattention(x1)
        x1 = x1 * self.Conv3_1(x1)
        x1 = x1 * self.spatialattention(x1)
        x1 = torch.concat([x1,x],dim=1)
        #x1 = x1 * self.EEMA(x1)
        w1 = self.Conv1(x1)
        x1 = x1 * w1
        x1 = self.Conv3_2(x1)
        x1 = self.Conv3_3(x1)
        #x1 = self.conv1_2(x1)
        x  = x + x1 
        return x
    




class ChannelAttentionMultiScale(nn.Module):
    def __init__(self, in_planes, scaling=16):
        super(ChannelAttentionMultiScale, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 多尺度特征提取（卷积核不同）
        self.conv1x1 = nn.Conv2d(in_planes, in_planes // scaling, kernel_size=1, bias=False)
        self.conv3x3 = nn.Conv2d(in_planes, in_planes // scaling, kernel_size=3, padding=1, bias=False)
        self.conv5x5 = nn.Conv2d(in_planes, in_planes // scaling, kernel_size=5, padding=2, bias=False)

        self.relu = nn.ReLU()
        self.fc = nn.Conv2d(3 * (in_planes // scaling), in_planes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)

        # 多尺度卷积
        scale1 = self.relu(self.conv1x1(avg_out + max_out))
        scale2 = self.relu(self.conv3x3(avg_out + max_out))
        scale3 = self.relu(self.conv5x5(avg_out + max_out))

        # 特征融合
        multi_scale_out = torch.cat([scale1, scale2, scale3], dim=1)
        attention = self.sigmoid(self.fc(multi_scale_out))

        return x * attention


class SpatialAttentionMultiScale(nn.Module):
    def __init__(self, kernel_sizes=[3, 5, 7]):
        super(SpatialAttentionMultiScale, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(2, 1, kernel_size=k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)

        # 多尺度卷积
        multi_scale_features = [conv(combined) for conv in self.convs]
        fused_features = torch.sum(torch.stack(multi_scale_features), dim=0)

        attention = self.sigmoid(fused_features)
        return x * attention



class CBAMMultiScale(nn.Module):
    def __init__(self, channel, scaling=16, kernel_sizes=[3, 5, 7]):
        super(CBAMMultiScale, self).__init__()
        self.channel_attention = ChannelAttentionMultiScale(channel, scaling)
        self.spatial_attention = SpatialAttentionMultiScale(kernel_sizes)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
    


    def irfft(x, d):
        t = torch.fft.ifft(torch.complex(x[:, :, 0], x[:, :, 1]), dim=(-d))
        return t.real
 
 
def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)
 
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
 
    Vc = rfft(v, 1)
 
    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
 
    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
 
    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2
 
    V = 2 * V.view(*x_shape)
 
    return V
 
 
class dct_channel_block(nn.Module):
    def __init__(self, channel):
        super(dct_channel_block, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel * 2, bias=False),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(channel * 2, channel, bias=False),
            nn.Sigmoid()
        )
 
        self.dct_norm = nn.LayerNorm([96], eps=1e-6) # for lstm on length-wise
 
    def forward(self, x):
        b, c, l = x.size() # (B,C,L) (32,96,512)
        list = []
        for i in range(c):
            freq = dct(x[:, i, :])
            list.append(freq)
 
        stack_dct = torch.stack(list, dim=1)
 
        lr_weight = self.dct_norm(stack_dct)
        lr_weight = self.fc(lr_weight)
        lr_weight = self.dct_norm(lr_weight)
 
        return x * lr_weight # result
 


def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiSpectralDCTLayer(nn.Module):
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        x = x * self.weight
        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)

        return dct_filter


class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction = 16, freq_sel_method = 'top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,h,w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


if __name__ == '__main__':
    x = torch.randn(4, 512, 32, 32)
    model = MultiSpectralAttentionLayer(512,32,32)
    output = model(x)
    print(output.shape)