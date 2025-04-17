import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class ChannelAttention(nn.Module):
    
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
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = torch.sigmoid(self.conv(y))
        return x * y

class FeatureExtractionModule(nn.Module):
    
    def __init__(self, in_channels, dropout_rate=0.5):
        super(FeatureExtractionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()
        self.dropout = nn.Dropout(dropout_rate)

        
        init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            init.zeros_(self.conv.bias)

    def forward(self, x):
        residual = x  
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.ca(x)  
        x = self.sa(x)  
        x = self.dropout(x)  
        return x + residual  

class RealFFT2d(nn.Module):

    def __init__(self):
        super(RealFFT2d, self).__init__()

    def forward(self, x):
        
        return torch.fft.rfft2(x, norm='backward')

class InvFFT2d(nn.Module):
    
    def __init__(self):
        super(InvFFT2d, self).__init__()

    def forward(self, x):
        return torch.fft.irfft2(x, norm='backward')

class SpectralModule(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(SpectralModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)  
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.fft = RealFFT2d()
        self.conv2 = nn.Conv2d(2 * in_channels, in_channels, kernel_size=3, padding=1, stride=1)  
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.inv_fft = InvFFT2d()
        self.conv_final = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

        
        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.conv_final.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        residual = x  
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.fft(x)
        real_part = torch.real(x)
        imag_part = torch.imag(x)
        combined = torch.cat([real_part, imag_part], dim=1)  
        x = F.relu(self.bn2(self.conv2(combined)))
        x = self.inv_fft(x)
        x = self.conv_final(x)
        return x + residual  

class FGA(nn.Module):
    
    def __init__(self, in_channels):
        super(FGA, self).__init__()
        self.feature_extraction = FeatureExtractionModule(in_channels)
        self.spectral_module = SpectralModule(in_channels, in_channels)
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1) 

    def forward(self, x):
        x = self.feature_extraction(x)
        spectral_features = self.spectral_module(x)
        return self.final_conv(spectral_features)

if __name__ == "__main__":
    input_tensor = torch.randn(6, 40, 32, 32)  
    model = FGA(in_channels=40) 
    output = model(input_tensor)  
    print("Output shape:", output.shape)  
