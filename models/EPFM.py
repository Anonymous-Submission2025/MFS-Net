import torch
from torch import nn
from models.EDM import EDM
from models.CPGM_1 import EEN


class EPFM(nn.Module):
    def __init__(self, in_channels, out_channels, sample, up=True, kernel_list=[3, 9]):
        super().__init__()
        
        # 定义特征提取模块 EDM，使用指定的输入通道数和卷积核列表
        self.edm = EDM(in_channels, kernel_list=kernel_list)
        
        # 定义全局上下文模块 CPGM，使用指定的输入通道数
        self.CPGM = EEN(in_channels)
        
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
        x_CPGM = self.CPGM(x)  # 通过 CPGM 模块处理输入 x
        
        # 将两个模块的输出在通道维度上拼接
        x_cat = torch.cat([x_edm, x_CPGM], dim=1)
        
        # 通过 MLP 处理拼接后的特征 
        '''
        MLP作用:
        1.特征融合:MLP接收来自EDM和CPGM模块的拼接特征(通道维度拼接）,通过1x1卷积和非线性变换,将这些特征融合成更高级的表示。
        2.维度变换:通过一系列的卷积层和激活函数,MLP将输入特征映射到目标输出通道数(out_channels),同时保持特征图的空间维度不变。
        3.非线性映射强:通过GELU激活函数,MLP引入了非线性变换,增强了模型的表达能力,使其能够学习更复杂的特征关系。
        4.特征归一化:通过BatchNorm层,MLP对特征进行归一化处理,有助于稳定训练过程并加速收敛。
        总结来说,MLP在这个模型中的作用是对EDM和CPGM模块提取的特征进行进一步的融合、变换和增强,从而得到更适合后续任务的特征表示。
        '''
        x = self.mlp(x_cat)
        
        # 如果 sample 不为 None，则进行上采样或下采样
        if self.sample is not None:
            x = self.sample(x)
        
        return x  # 返回处理后的输出)