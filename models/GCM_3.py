import torch
import torch.nn as nn
import torch.nn.functional as F

# 高通滤波器（HPG）模块
class HighPassGaussian(nn.Module):
    '''
        初始化高通滤波器模块。

        参数
        in_channels (int): 输入特征图的通道数
    '''
    def __init__(self, in_channels):
        
        super(HighPassGaussian, self).__init__()
        # 定义高通滤波的卷积核
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        # 权重初始化为高通滤波器
        hpg_kernel = torch.tensor([[[-1, -1, -1],
                                    [-1,  9, -1],
                                    [-1, -1, -1]]], dtype=torch.float32)  # 简化的高通滤波器
        self.conv.weight.data = hpg_kernel.repeat(in_channels, 1, 1, 1)  # 扩展到 (in_channels, in_channels, 3, 3)
        self.conv.weight.requires_grad = False  # 不更新权重
        
    def forward(self, x):
        return self.conv(x)

# 低通滤波器（LPG）模块
class LowPassGaussian(nn.Module):
    def __init__(self, in_channels):
        super(LowPassGaussian, self).__init__()
        # 定义低通滤波的卷积核
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        # 权重初始化为低通滤波器

        lpg_kernel = torch.tensor([[1,  2,  1],
                                    [2,  4,  2],
                                    [1,  2,  1]], dtype=torch.float32) / 16  # 高斯低通滤波器
        self.conv.weight.data = lpg_kernel.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)  # 扩展到 (in_channels, in_channels, 3, 3)
        self.conv.weight.requires_grad = False  # 不更新权重

    def forward(self, x):
        return self.conv(x)

# 深度可分离卷积模块
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
    
# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        if in_channels < 16:
            raise ValueError("in_channels must be at least 16")
        self.fc1 = nn.Conv2d(in_channels, in_channels // 16, kernel_size=1)  # 将通道数压缩
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Conv2d(in_channels // 16, in_channels, kernel_size=1)  # 恢复通道数
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数用于输出权重

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(F.adaptive_avg_pool2d(x, 1))))
        max_out = self.fc2(self.relu(self.fc1(F.adaptive_max_pool2d(x, 1))))
        
        # 调试输出
        print(f"avg_out shape: {avg_out.shape}")  # 应该是 (B, C, 1, 1)
        print(f"max_out shape: {max_out.shape}")  # 应该是 (B, C, 1, 1)

        return self.sigmoid(avg_out + max_out)


# GCM 模块，结合 HPG 和 LPG
class GCM(nn.Module):
    def __init__(self, input_channels):
        super(GCM, self).__init__()
        self.hpg = HighPassGaussian(input_channels)  # 高通滤波器
        self.lpg = LowPassGaussian(input_channels)    # 低通滤波器
        self.dsc = DepthwiseSeparableConv(input_channels, input_channels)  # 深度可分离卷积
        self.gru = nn.GRU(input_size=input_channels, hidden_size=input_channels, 
                          num_layers=1, bidirectional=False, batch_first=True)  # 确保输出通道数
        self.channel_attention = ChannelAttention(input_channels)  # 通道注意力
        self.fc_dynamic = nn.Linear(input_channels, input_channels)  # 动态权重共享
        self.conv_out = nn.Conv2d(input_channels, input_channels, kernel_size=1)  # 特征聚合

    def forward(self, x):
        # 1. 通过 HPG 和 LPG 提取特征
        hpg_out = self.hpg(x)  # 高通特征
        lpg_out = self.lpg(x)  # 低通特征

        # 2. 合并 HPG 和 LPG 特征
        combined_out = hpg_out + lpg_out  # 简单相加，可以根据需要调整

        # 3. 深度可分离卷积提取特征
        combined_out = self.dsc(combined_out)

        # 4. 处理 GRU 输入
        B, C, H, W = combined_out.shape
        x_flat = combined_out.view(B, C, -1).permute(0, 2, 1)  # 改变维度为 (B, H*W, C)

        # 调试输出
        print(f"x_flat shape: {x_flat.shape}")  # 应该是 (B, H*W, C)

        gru_out, _ = self.gru(x_flat)  # 通过 GRU
        gru_out = gru_out.permute(0, 2, 1).contiguous().view(B, C, H, W)  # 变回 (B, C, H, W)

        # 调试输出
        print(f"gru_out shape: {gru_out.shape}")  # 确保 C > 0

        # 检查 GRU 输出的形状
        if gru_out.shape[1] == 0:
            raise ValueError("GRU output has zero channels")

        # 5. 应用通道注意力
        attention_out = self.channel_attention(gru_out)  # 计算通道注意力权重
        attention_out = attention_out * gru_out  # 加权输入特征

        # 6. 动态权重共享
        dynamic_weights = self.fc_dynamic(attention_out)
        combined_out = dynamic_weights + attention_out

        # 7. 最终特征聚合
        out = self.conv_out(combined_out)

        return out

# 示例使用
if __name__ == "__main__":
    model = GCM(input_channels=64)  # 创建 GCM 模型实例
    input_tensor = torch.randn(8, 64, 128, 128)  # 假设的输入 (batch_size, channels, height, width)
    output = model(input_tensor)  # 前向传播
    print(output.shape)  # 输出维度
