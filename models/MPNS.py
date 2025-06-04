import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import models.hotfeat as hotfeat

class DPFE(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias,patch_sizes):    # dim = 32, ffn_expansion_factor = 4, bias = True
        super(EDFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)           # 4 * 32 = 128

        self.ffn_expansion_factor = ffn_expansion_factor
        self.bias = bias
        self.patch_size1 = patch_sizes[0]
        self.patch_size2 = patch_sizes[1]
        self.bias = bias
        self.dim = dim

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft1 = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size1, self.patch_size1 // 2 + 1)))
        self.fft2 = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size2, self.patch_size2 // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        
        x = self.project_in(x)  
       
        x1, x2 = self.dwconv(x).chunk(2, dim=1)        
        
        x = F.gelu(x1) * x2         
       
        x = self.project_out(x)         
        
        
        
        # print("patch_size1",self.patch_size1)
        # print("patch_size2",self.patch_size2) 
        x1_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size1,
                            patch2=self.patch_size1)
        
        x1_patch_fft = torch.fft.rfft2(x1_patch.float())  
        
        x1_patch_fft = x1_patch_fft * self.fft1
       
        x1_patch = torch.fft.irfft2(x1_patch_fft, s=(self.patch_size1, self.patch_size1))
        
        x1 = rearrange(x1_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size1,
                      patch2=self.patch_size1)
        
        x2_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size2,
                            patch2=self.patch_size2)
        
        x2_patch_fft = torch.fft.rfft2(x2_patch.float())  
        
        x2_patch_fft = x2_patch_fft * self.fft2
       
        x2_patch = torch.fft.irfft2(x2_patch_fft, s=(self.patch_size2, self.patch_size2))
        
        x2 = rearrange(x2_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size2,
                      patch2=self.patch_size2)
        

        x = x1 + x2
        
        return x
    