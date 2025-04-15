import torch
import torch.nn as nn
import torchvision.ops as ops

class DeformableConv2d(nn.Module):
    """可变形卷积层，根据输入特征动态计算卷积核的偏移量。"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DeformableConv2d, self).__init__()
        # 用于计算偏移量的卷积层，输出通道数为2 * kernel_size * kernel_size
        self.conv_offset = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=3, padding=1)
        # 实际的可变形卷积操作
        self.conv = ops.DeformConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # 计算偏移量
        offset = self.conv_offset(x)
        # 通过可变形卷积进行特征提取
        return self.conv(x, offset)

class ChannelSpatialAttention(nn.Module):
    """通道-空间注意力模块，先计算通道注意力再计算空间注意力。"""
    def __init__(self, in_channels):
        super(ChannelSpatialAttention, self).__init__()
        # 为防止 in_channels // 16 为0，至少输出1个通道
        hidden_channels = max(1, in_channels // 16)
        # 通道注意力模块
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        # 空间注意力模块
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 计算通道注意力权重，并与输入相乘
        channel_weights = self.channel_att(x)
        x = x * channel_weights
        # 计算空间注意力权重，并与输入相乘
        spatial_weights = self.spatial_att(x)
        x = x * spatial_weights
        return x

class DAB(nn.Module):
    """多尺度特征提取块，包含可变形卷积和注意力机制，可选采样操作。"""
    def __init__(self, in_channels, kernel, sample1=None, sample2=None):
        super().__init__()
        self.sample1 = sample1
        self.sample2 = sample2
        # 特征提取层：先使用可变形卷积，再进行批归一化
        self.extract = nn.Sequential(
            DeformableConv2d(in_channels, in_channels, kernel, padding=kernel // 2),
            nn.BatchNorm2d(in_channels)
        )
        # 注意力机制
        self.attention = ChannelSpatialAttention(in_channels)

    def forward(self, x):
        # 如果存在预采样操作，则先对输入进行处理
        if self.sample1 is not None:
            x = self.sample1(x)
        x = self.extract(x)
        x = self.attention(x)
        # 如果存在后采样操作，则再进行处理
        if self.sample2 is not None:
            x = self.sample2(x)
        return x

class msee(nn.Module):
    """多尺度特征融合模块，将不同尺度的特征进行融合。"""
    def __init__(self, in_channels, kernel_list=[3, 9]):
        super().__init__()
        # 构建多个 MSFA_Block，不同块使用不同的卷积核尺寸和采样策略
        self.msfa1 = DAB(in_channels, kernel_list[0])
        self.msfa2 = DAB(in_channels, kernel_list[1])
        self.msfa3 = DAB(in_channels, kernel_list[0],
                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.MaxPool2d(kernel_size=2, stride=2))
        self.msfa4 = DAB(in_channels, kernel_list[1],
                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.MaxPool2d(kernel_size=2, stride=2))
        self.msfa5 = DAB(in_channels, kernel_list[0],
                                 nn.MaxPool2d(kernel_size=2, stride=2),
                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.msfa6 = DAB(in_channels, kernel_list[1],
                                 nn.MaxPool2d(kernel_size=2, stride=2),
                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        
        # 融合所有块的输出，使用深度可分离卷积进行特征融合
        self.extract = nn.Sequential(
            nn.Conv2d(6 * in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1),
        )

    def forward(self, x):
        # 分别通过各个 MSFA 块进行特征提取
        x1 = self.msfa1(x)
        x2 = self.msfa2(x)
        x3 = self.msfa3(x)
        x4 = self.msfa4(x)
        x5 = self.msfa5(x)
        x6 = self.msfa6(x)
        # 将所有尺度的特征在通道维度上拼接
        out = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)
        out = self.extract(out)
        return out

# 使用示例
if __name__ == "__main__":
    # 创建输入张量 (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 10, 64, 64)  # 1个样本，10个通道，64x64的特征图
    model = msee(in_channels=10)  # 初始化 MSFA 模型，输入通道数为10
    output = model(input_tensor)  # 前向传播
    print("Output shape:", output.shape)  # 打印输出形状
