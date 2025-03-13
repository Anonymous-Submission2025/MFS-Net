import torch
from torch import nn
from models.EDM import EDM
from models.GCM import GCM


class EPFM(nn.Module):
    def __init__(self, in_channels, out_channels, sample, up=True, kernel_list=[3, 9]):
        super().__init__()
        
        # 定义特征提取模块 EDM，使用指定的输入通道数和卷积核列表
        self.edm = EDM(in_channels, kernel_list=kernel_list)
        
        # 定义全局上下文模块 GCM，使用指定的输入通道数
        self.gcm = GCM(in_channels)
        
        # 定义多层感知机 MLP 结构
        self.mlp = nn.Sequential(
            nn.BatchNorm2d(in_channels * 2),  # 批归一化，处理输入的通道数为 in_channels*2
            nn.Conv2d(in_channels * 2, out_channels, 1),  # 1x1 卷积，将通道数变为 out_channels
            nn.GELU(),  # GELU 激活函数
            nn.Conv2d(out_channels, out_channels, 1),  # 再次使用 1x1 卷积，保持通道数
            nn.BatchNorm2d(out_channels)  # 批归一化
        )
        
        # 根据输入的 sample 参数选择是否进行下采样或上采样
        if sample:
            if up:
                # 上采样，使用双线性插值方法
                self.sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                # 下采样，使用最大池化
                self.sample = nn.MaxPool2d(2, stride=2)
        else:
            # 如果不需要采样，则 sample 为 None
            self.sample = None

    def forward(self, x):
        # 前向传播
        x_edm = self.edm(x)  # 通过 EDM 模块处理输入 x
        x_gcm = self.gcm(x)  # 通过 GCM 模块处理输入 x
        
        # 将两个模块的输出在通道维度上拼接
        x_cat = torch.cat([x_edm, x_gcm], dim=1)
        
        # 通过 MLP 处理拼接后的特征
        x = self.mlp(x_cat)
        
        # 如果 sample 不为 None，则进行上采样或下采样
        if self.sample is not None:
            x = self.sample(x)
        
        return x  # 返回处理后的输出