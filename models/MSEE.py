import torch
import torch.nn as nn
import torchvision.ops as ops

class DeformableConv2d(nn.Module):
   
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DeformableConv2d, self).__init__()
        self.conv_offset = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=3, padding=1)
        self.conv = ops.DeformConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        offset = self.conv_offset(x)
        return self.conv(x, offset)

class ChannelSpatialAttention(nn.Module):
    
    def __init__(self, in_channels):
        super(ChannelSpatialAttention, self).__init__()
        
        hidden_channels = max(1, in_channels // 16)
        
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        channel_weights = self.channel_att(x)
        x = x * channel_weights
       
        spatial_weights = self.spatial_att(x)
        x = x * spatial_weights
        return x

class DAB(nn.Module):

    def __init__(self, in_channels, kernel, sample1=None, sample2=None):
        super().__init__()
        self.sample1 = sample1
        self.sample2 = sample2
        
        self.extract = nn.Sequential(
            DeformableConv2d(in_channels, in_channels, kernel, padding=kernel // 2),
            nn.BatchNorm2d(in_channels)
        )
       
        

    def forward(self, x):
        
        if self.sample1 is not None:
            x = self.sample1(x)
        x = self.extract(x)
        
       
        if self.sample2 is not None:
            x = self.sample2(x)
        return x

class msee(nn.Module):
    def __init__(self, in_channels, kernel_list=[3, 9]):
        super().__init__()
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
        
        self.extract = nn.Sequential(
            nn.Conv2d(6 * in_channels, in_channels, 3, padding=1, groups=in_channels),     
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1),                                        
        )
        self.attention = ChannelSpatialAttention(6 * in_channels)

    def forward(self, x):
      
        x1 = self.msfa1(x)
        x2 = self.msfa2(x)
        x3 = self.msfa3(x)
        x4 = self.msfa4(x)
        x5 = self.msfa5(x)
        x6 = self.msfa6(x)
      
        out = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)
        x = self.attention(out)
        out = self.extract(x)
        return out


if __name__ == "__main__":
  
    input_tensor = torch.randn(1, 10, 64, 64)  
    model = msee(in_channels=10) 
    output = model(input_tensor)  
    print("Output shape:", output.shape)  
