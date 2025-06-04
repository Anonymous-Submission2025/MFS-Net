import torch
from torch import nn

import models.hotfeat as hotfeat

import torchvision.ops as ops
from models.MSEE import msee
from models.MPNS import DPFE




class DFFM(nn.Module):
    def __init__(self, in_channels, out_channels, sample, patchsizes,up=True,kernel_list=[3, 9]):
        super().__init__()

        # Deformable Multi-Scale Attention Fusion
        self.msf = msee(in_channels,kernel_list=kernel_list)
        
        # Dual Fourier-Feature Guided Attention
        self.mpf = DPFE(in_channels,ffn_expansion_factor=4,bias=True,patch_sizes=patchsizes)
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
        self.linear = nn.Linear(in_features=10, out_features=5, bias=True)

    def forward(self, x):
        x_msf = self.msf(x)

        x=self.linear(x)
        x_mpf = self.mpf(x)
        x_mpf = x + x_mpf
        
        x_cat = torch.cat([x_msf, x_mpf], dim=1)      
        # hotfeat.feature_vis(x_cat, "x_cat",isMax=False, save_path="/home/wjj/My_model/test_model_main/热力图")                                                                                                                                             
        x = self.mlp(x_cat)  
        
        if self.sample is not None:
            x = self.sample(x)
        
        return x
    
if __name__ == '__main__':
    x = torch.randn(1, 32, 512, 512)
    model = DFFM(32, 32, sample=True, up=True, patchsizes=[32, 128])
    y = model(x)
    print(y.shape)
