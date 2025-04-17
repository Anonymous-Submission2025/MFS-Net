import torch
from torch import nn
from models.MSEE import msee
from models.MPNS import FGA




class DFFM(nn.Module):
    def __init__(self, in_channels, out_channels, sample, up=True, kernel_list=[3, 9]):
        super().__init__()

        # Deformable Multi-Scale Attention Fusion
        self.msf = msee(in_channels,kernel_list=kernel_list)
        
        # Dual Fourier-Feature Guided Attention
        self.mpf = FGA(in_channels)
        self.mlp = nn.Sequential(
                nn.BatchNorm2d(in_channels * 2),  
                nn.Conv2d(in_channels * 2, out_channels, 1), 
                
                nn.GELU(),                                       
                nn.Conv2d(out_channels, out_channels, 1), 
                nn.BatchNorm2d(out_channels)
            )
        if sample:
            if up:
                self.sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                self.sample = nn.MaxPool2d(2, stride=2)
        else:
            self.sample = None

        self.reduce_channels = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x_msf = self.msf(x)
        x_mpf = self.mpf(x)
        x_cat = torch.cat([x_msf, x_mpf], dim=1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
        x = self.mlp(x_cat)  
        if self.sample is not None:
            x = self.sample(x)
        
        return x
