import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicFusion(nn.Module):
    def __init__(self, in_channels):
        super(DynamicFusion, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, feature1, feature2):
        pooled1 = self.avgpool(feature1).view(feature1.size(0), -1)  # (B, C)
        pooled2 = self.avgpool(feature2).view(feature2.size(0), -1)  # (B, C)

        # 计算融合权重
        weights = self.mlp(torch.cat((pooled1, pooled2), dim=1))  # (B, 1)
        weights = weights.view(-1, 1, 1, 1)  # (B, 1, 1, 1) 以便于后续广播

        # 进行加权融合
        fused = weights * feature1 + (1 - weights) * feature2  # 确保 feature1 和 feature2 的维度匹配
        return fused


class CPGM(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
                
        
        self.linear_o = nn.Linear(input_channels, input_channels)
        self.linear_p = nn.Linear(input_channels, input_channels)
        self.scale = input_channels ** -0.5
        self.norm = nn.LayerNorm(input_channels)
        self.soft = nn.Softmax(-1)

        # 深度可分离卷积层定义
        self.dsc = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, padding=1, groups=input_channels),  # 深度可分离卷积，3x3卷积，保持输入通道数
            nn.GELU(),  # 使用GELU激活函数进行非线性变换
            nn.Conv2d(input_channels, input_channels, 1),  # 1x1卷积用于线性组合输出通道
            nn.BatchNorm2d(input_channels)  # 批归一化，标准化卷积输出
        )
        # LPG 部分
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化，将特征图降至1x1
        self.Q_proj = nn.Linear(input_channels, input_channels)  # 线性层，生成查询（Q）向量
        self.K_proj = nn.Linear(input_channels, input_channels)  # 线性层，生成键（K）向量
        self.liner_l = nn.Linear(input_channels, input_channels)  # 线性层，用于低频提示的输出
        self.sig = nn.Sigmoid()  # Sigmoid激活函数，用于限制输出范围

        # HPG 部分
        self.sc_h = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=(5, 1), padding=(2, 0), groups=input_channels),
            nn.GELU(),
            nn.Conv2d(input_channels, input_channels, 1),
            nn.BatchNorm2d(input_channels)
        )
        
        self.sc_v = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=(1, 5), padding=(0, 2), groups=input_channels),
            nn.GELU(),
            nn.Conv2d(input_channels, input_channels, 1),
            nn.BatchNorm2d(input_channels)
        )

        self.conv_h = nn.Conv2d(input_channels, input_channels, 1)
        
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 更新双向动态融合模块，传递输入通道数
        self.ddf = DynamicFusion(input_channels)

    def forward_lpg(self, x):
        # 生成低频提示
        x_gap = self.gap(x) * x
        B, C, H, W = x_gap.shape
        N = H * W
        x_gap = x_gap.view(B, C, N).permute(0, 2, 1).contiguous()
        x_Q = self.Q_proj(x_gap)
        x_K = self.K_proj(x_gap)
        x_V = self.sig(x_Q) * x_K + x_gap
        prompt_l = self.liner_l(x_V)
        prompt_l = prompt_l.permute(0, 2, 1).contiguous().view(B, C, H, W)
        return prompt_l

    def forward_hpg(self, x):
        # 生成高频提示
        x_h = self.sc_h(x)
        x_v = self.sc_v(x)
        prompt_h = self.conv_h(x_h + x_v)
        return prompt_h

    def forward_ddf(self, prompt_l, prompt_h):
        # 使用双向动态融合
        fused = self.ddf(prompt_l, prompt_h)
        return fused

    def forward(self, x: torch.Tensor):
        original_size = x.shape[2:]  # 记录原始输入尺寸
        x = self.pool(x)  # 池化操作会将空间维度减半
        prompt_l = self.forward_lpg(x)  # 生成低频提示
        prompt_h = self.forward_hpg(x)  # 生成高频提示
        
        # 确保 prompt_l 和 prompt_h 的形状一致
        if prompt_l.shape[2:] != prompt_h.shape[2:]:
            prompt_h = F.interpolate(prompt_h, size=prompt_l.shape[2:], mode='bilinear', align_corners=True)

        out = self.forward_ddf(prompt_l, prompt_h)  # 使用双向动态融合
        out = self.dsc(out) + out  # 处理后的输出与原始输出相加
        
        # 上采样回原始输入大小
        out = F.interpolate(out, size=original_size, mode='bilinear', align_corners=True)  # 使用上采样
        return out  # 输出保持 (B, C, H, W) 的形状

# 示例用法
if __name__ == "__main__":
    input_tensor = torch.randn(8, 3, 32, 32)  # 8个样本，3个通道，32x32的输入图像
    model = CPGM(input_channels=3)
    output = model(input_tensor)
    print(output.shape)  # 应该输出 (8, 3, 32, 32)
