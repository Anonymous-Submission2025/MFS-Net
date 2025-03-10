import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class SobelOffsets(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 可学习的 Sobel 算子
        self.sobel_kernel_x = nn.Parameter(torch.tensor([[1, 0, -1],
                                                         [2, 0, -2],
                                                         [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3), requires_grad=True)
        
        self.sobel_kernel_y = nn.Parameter(torch.tensor([[1, 2, 1],
                                                         [0, 0, 0],
                                                         [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3), requires_grad=True)
        
        # 用于特征融合的全连接层
        self.fc = nn.Linear(in_channels * 2, in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        # 使用 Sobel 算子进行边缘检测
        edge_x = F.conv2d(x, self.sobel_kernel_x.repeat(C, 1, 1, 1), padding=1, groups=C)
        edge_y = F.conv2d(x, self.sobel_kernel_y.repeat(C, 1, 1, 1), padding=1, groups=C)
        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2)  # 计算边缘强度

        # 连接特征并通过全连接层学习加权
        combined_features = torch.cat((x, edges), dim=1).view(B, -1)
        weights = self.fc(combined_features)  # 计算加权系数
        weights = torch.sigmoid(weights).view(B, C, 1, 1)  # 激活并重塑为与特征图相同的形状

        # 特征融合
        enhanced_features = x * (1 + weights) + edges * weights
        return enhanced_features

class CFR(nn.Module):
    def __init__(self, in_channels, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        # 引入 Sobel 边缘检测
        self.sobel_offsets = SobelOffsets(in_channels)

        # 进行可学习的变形卷积
        self.deform = torchvision.ops.DeformConv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            dilation=dilation,
            bias=False
        )
        
        self.balance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        # 使用 Sobel 边缘检测增强特征
        enhanced_features = self.sobel_offsets(x)
        # 进行变形卷积
        out = self.deform(enhanced_features, offsets=None)  # 这里可以选择是否使用偏移量
        out = self.balance(out) * x
        return out

# 使用示例
input_tensor = torch.randn((8, 16, 32, 32))  # 8个样本，16个通道，32x32大小
cfr = CFR(in_channels=16)
output_tensor = cfr(input_tensor)
