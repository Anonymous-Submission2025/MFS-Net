import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(batch_size, channels, 1, 1)
        return x * y

class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = torch.sigmoid(self.conv(y))
        return x * y

class LocalFeatureExtraction(nn.Module):
    """局部特征提取模块"""
    def __init__(self, in_channels):
        super(LocalFeatureExtraction, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.ca = ChannelAttention(in_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.ca(x)

class GlobalFeatureExtraction(nn.Module):
    """全局特征提取模块"""
    def __init__(self, in_channels):
        super(GlobalFeatureExtraction, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.sa(x)

class RealFFT2d(nn.Module):
    """实值2D快速傅里叶变换模块"""
    def __init__(self):
        super(RealFFT2d, self).__init__()

    def forward(self, x):
        # 对输入进行快速傅里叶变换，返回复数形式
        return torch.fft.rfft2(x, norm='backward')

class InvFFT2d(nn.Module):
    """逆实值2D快速傅里叶变换模块"""
    def __init__(self):
        super(InvFFT2d, self).__init__()

    def forward(self, x):
        # 对输入进行逆快速傅里叶变换
        return torch.fft.irfft2(x, norm='backward')
    
class SpectralModule(nn.Module):
    """Spectral模块,包含卷积-FFT-卷积-逆FFT流程"""
    def __init__(self, in_channels, out_channels):
        super(SpectralModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)  # 保持空间尺寸
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.fft = RealFFT2d()
        self.conv2 = nn.Conv2d(2 * in_channels, in_channels, kernel_size=3, padding=1, stride=1)  # 保持空间尺寸
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.inv_fft = InvFFT2d()
        self.conv_final = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        # 1. 对输入进行卷积和激活
        x = F.relu(self.bn1(self.conv1(x)))
        
        # 2. 使用FFT转换到频域
        x = self.fft(x)
        
        # 3. 获取FFT结果的实部和虚部
        real_part = torch.real(x)
        imag_part = torch.imag(x)
        
        # 4. 合并实部和虚部
        combined = torch.cat([real_part, imag_part], dim=1)  # 合并后通道数变为 2 * in_channels
        
        # 5. 进行卷积
        x = F.relu(self.bn2(self.conv2(combined)))
        
        # 6. 使用逆FFT转换回空间域
        x = self.inv_fft(x)
        
        # 7. 使用最终卷积进行输出
        x = self.conv_final(x)
        return x

class FeatureGuidedAttentionModule(nn.Module):
    """特征引导注意力模块"""
    def __init__(self, in_channels):
        super(FeatureGuidedAttentionModule, self).__init__()
        self.local_extraction = LocalFeatureExtraction(in_channels)
        self.global_extraction = GlobalFeatureExtraction(in_channels)

    def forward(self, x):
        local_features = self.local_extraction(x)
        global_features = self.global_extraction(x)
        return local_features + global_features

class FGA(nn.Module):
    
    def __init__(self, in_channels):
        super(FGA, self).__init__()
        self.spectral_module = SpectralModule(in_channels, in_channels)
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 保持通道数一致

    def forward(self, x):
        spectral_features = self.spectral_module(x)
        return self.final_conv(spectral_features)

# 使用示例
if __name__ == "__main__":
    input_tensor = torch.randn(6, 40, 32, 32)  
    model = FGA(in_channels=40)  # 初始化模型
    output = model(input_tensor)  # 前向传播
    print("Output shape:", output.shape)  # 打印输出形状
