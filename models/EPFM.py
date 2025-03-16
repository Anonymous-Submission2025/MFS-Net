import torch
from torch import nn
from models.EDM import EDM
from models.CPGM_2 import CPGM
import torch.nn.functional as F

class EPFM(nn.Module):
    def __init__(self, in_channels, out_channels, sample, up=True, kernel_list=[3, 9]):
        super().__init__()
        
        # 定义特征提取模块 EDM，使用指定的输入通道数和卷积核列表
        self.edm = EDM(in_channels, kernel_list=kernel_list)
        
        # 定义全局上下文模块 CPGM，使用指定的输入通道数
        self.CPGM = CPGM(in_channels)
        
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
        
        # print("x_edm shape:", x_edm.shape)  # 打印 x_edm 的形状
        # print("x_CPGM shape:", x_CPGM.shape)  # 打印 x_CPGM 的形状


        # 将 x_CPGM 调整为与 x_edm 相同的大小
        # x_CPGM_resized = F.interpolate(x_CPGM, size=(256, 256), mode='bilinear', align_corners=True)

        # 打印每个张量的形状
        # print("Before concatenation:")
        # print("x_edm shape:", x_edm.shape)            # (4, 10, 256, 256)
        # print("x_CPGM_resized shape:", x_CPGM.shape)  # (4, 10, 256, 256)

        # 拼接
        x_cat = torch.cat([x_edm, x_CPGM], dim=1)  # 在通道维度上拼接
        # print("Concatenated shape:", x_cat.shape)
        

        # print(f"x_cat shape: {x_cat.shape}")  # 应该是 [B, in_channels * 2, H, W]
        # 通过 MLP 处理拼接后的特征 
        x = self.mlp(x_cat)
        
        # 如果 sample 不为 None，则进行上采样或下采样
        if self.sample is not None:
            x = self.sample(x)
        
        return x  # 返回处理后的输出)
    

if __name__ == "__main__":
    input_tensor = torch.randn(8, 3, 32, 32)  # 8个样本，3个通道，32x32的输入图像
    edm_model = EDM(input_channels=3)
    cpgm_model = CPGM(input_channels=256)  # 假设 CPGM 输入 256 通道
    x_edm = edm_model(input_tensor)
    x_CPGM = cpgm_model(input_tensor)

    print("x_edm shape:", x_edm.shape)  # 应输出 (8, 256, H, W)
    print("x_CPGM shape:", x_CPGM.shape)  # 应输出 (8, 256, H, W)
    
    # 确保在这里做拼接
    x_cat = torch.cat([x_edm, x_CPGM], dim=1)  # 确保维度匹配
    print("Concatenated shape:", x_cat.shape)
