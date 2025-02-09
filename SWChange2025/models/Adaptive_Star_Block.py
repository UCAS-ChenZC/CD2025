import torch
import torch.nn as nn
from timm.layers import DropPath

'''
Star模块缺陷
星操作模块的缺点在于其特征表达不够灵活，高维特征的权重控制力不足，特别是在低通道数（网络宽度）条件下，可能无法充分发挥其高维特征映射的
优势。此外，由于逐元素乘法在不同硬件上的效率不同，可能导致性能不稳定。

CV缝合就行创新点：
引入自适应星操作（Adaptive Star Operation）模块，通过在星操作之前增加通道注意力机制，动态调整每个通道的权重，提升特征表达
的灵活性。同时，在逐元素乘法之后增加一个可学习的隐式高维权重矩阵，以控制每个隐式维度的特征权重。
'''


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class SE_Block(nn.Module):
    """Squeeze-and-Excitation (SE) block for channel attention."""
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class Adaptive_Star_Block(nn.Module):
    def __init__(self, in_dim1, in_dim2, mlp_ratio=3, drop_path=0., reduction=4):
        super().__init__()
        # Downsample x1 to match x2 dimensions
        self.downsample = nn.Sequential(
            nn.Conv2d(in_dim1, in_dim2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_dim2),
            nn.ReLU(inplace=True)
        )
        
        # Depthwise convolution and SE block for interaction
        self.dwconv = Conv(in_dim2, in_dim2, 7, g=in_dim2, act=False)
        self.se = SE_Block(in_dim2, reduction)
        
        # High-dimensional mapping and fusion
        self.f1 = nn.Conv2d(in_dim2, mlp_ratio * in_dim2, 1)
        self.f2 = nn.Conv2d(in_dim2, mlp_ratio * in_dim2, 1)
        self.g = Conv(mlp_ratio * in_dim2, in_dim2, 1, act=False)
        
        # Depthwise convolution for output
        self.dwconv2 = nn.Conv2d(in_dim2, in_dim2, 7, 1, (7 - 1) // 2, groups=in_dim2)
        
        # Activation, drop path, and high-dimensional weight
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.hidden_weight = nn.Parameter(torch.ones(in_dim2, 1, 1))  # Learnable weight for high-dimensional control


    def forward(self, x1, x2):
        # Downsample x1 to match the dimensions of x2
        input2 = x2
        x1 = self.downsample(x1)
        
        # Apply depthwise convolution and SE block
        x1, x2 = self.dwconv(x1), self.dwconv(x2)
        x1, x2 = self.se(x1), self.se(x2)

        # High-dimensional feature interaction
        x1, x2 = self.f1(x1), self.f2(x2)
        x2 = self.act(x1) * x2

        # Further processing and high-dimensional weight control
        x2 = self.dwconv2(self.g(x2))
        x2 = x2 * self.hidden_weight

        # Residual connection and output
        x2 = input2 + self.drop_path(x2)
        return x2


# 测试代码
if __name__ == '__main__':
    block = Adaptive_Star_Block(in_dim1=96, in_dim2=192, mlp_ratio=3)
    input1 = torch.rand(2, 96, 64, 64)
    input2 = torch.rand(2, 192, 32, 32)
    output = block(input1, input2)
    print("input1.shape:", input1.shape)
    print("input2.shape:", input2.shape)
    print("output.shape:", output.shape)
