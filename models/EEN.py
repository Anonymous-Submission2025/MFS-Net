import torch
import torch.nn as nn
import torchvision

class EdgeEnhancementAttention(nn.Module):
    '''
    function:边缘增强注意力机制，通过边缘检测和通道注意力来增强输入特征的边缘信息和通道信息。
    input:
        in_channels:输入特征的通道数。
        kernel_size:边缘检测卷积核的大小,默认为3。
    output:
        out:增强后的特征。
    '''
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        
        # 边缘检测卷积层:提取边缘特征，使用深度可分离卷积提高计算效率
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),  # 深度可分离卷积
            nn.BatchNorm2d(in_channels),  # 批归一化，提高训练稳定性
            nn.ReLU(inplace=True)  # ReLU激活函数，引入非线性
        )
        
        # 通道注意力机制
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化，生成通道描述符
        self.fc1 = nn.Linear(in_channels, in_channels // 16, bias=False)  # 第一层全连接，用于降低维度
        self.fc2 = nn.Linear(in_channels // 16, in_channels, bias=False)  # 第二层全连接，用于恢复维度
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数，用于生成通道权重

        # 融合层，用于将边缘特征和增强特征结合
        self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)  # 1x1卷积，融合通道
        self.batch_norm = nn.BatchNorm2d(in_channels)  # 批归一化，标准化输出

    def forward(self, x):
        # 边缘特征提取
        edge_features = self.edge_conv(x)  # 通过边缘检测卷积层获取边缘特征

        # 通道注意力计算
        b, c, h, w = x.size()  # 获取输入的批量大小、通道数、高度和宽度
        avg_out = self.avg_pool(x).view(b, c)  # 生成通道描述符
        attention = self.fc1(avg_out)  # 通过第一层全连接层生成中间特征
        attention = nn.ReLU(inplace=True)(attention)  # ReLU激活
        attention = self.fc2(attention)  # 通过第二层全连接层生成最终通道权重
        attention = self.sigmoid(attention).view(b, c, 1, 1)  # 应用Sigmoid并调整形状为(B, C, 1, 1)

        # 应用通道注意力
        enhanced_features = x * attention  # 通过通道权重增强输入特征
        
        # 融合边缘特征和增强特征
        fused = torch.cat((enhanced_features, edge_features), dim=1)  # 将增强特征和边缘特征沿通道维度拼接
        out = self.fusion_conv(fused)  # 通过卷积层融合特征
        out = self.batch_norm(out)  # 归一化输出

        return out

class EEA(nn.Module):
    def __init__(self, in_channels, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size  # 保存卷积核大小
        # 使用边缘增强注意力模块
        self.edge_attention = EdgeEnhancementAttention(in_channels)

        # 学习可变形卷积，能够学习到更复杂的特征
        self.deform = torchvision.ops.DeformConv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,  # 使每个通道独立卷积
            dilation=dilation,  # 膨胀卷积，增加感受野
            bias=False  # 不使用偏置
        )

        # 用于特征平衡的卷积层
        self.balance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),  # 1x1卷积
            nn.BatchNorm2d(in_channels)  # 批归一化
        )

    def forward(self, x):
        # 应用边缘增强注意力模块
        enhanced_features = self.edge_attention(x)  # 获取增强特征

        # 确保 enhanced_features 具有有效的形状
        if enhanced_features.numel() == 0:  # 检查特征是否为空
            raise ValueError("Input features are empty. Check your input data.")

        # 学习偏移量，确保偏移量形状为 (B, 2 * kernel_size * kernel_size, H, W)
        B, C, H, W = enhanced_features.size()  # 获取增强特征的形状
        offsets = torch.zeros(B, 2 * self.kernel_size * self.kernel_size, H, W, device=x.device)  # 初始化偏移量

        # 学习可变形卷积
        out = self.deform(enhanced_features, offsets)  # 传递计算得到的偏移量
        out = self.balance(out) * x  # 融合原始特征，通过平衡层调整输出

        return out
