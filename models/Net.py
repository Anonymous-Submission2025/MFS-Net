import torch.nn as nn



from models.EPFM import EPFM



class RoMER_UNet(nn.Module):
    def __init__(self,input_channels=3, out_channels:list=None,kernel_list=None):
        super().__init__()
        ## 编码部分：使用 EPFM 模块进行特征提取和下采样
        self.en1=EPFM(out_channels[0],out_channels[1],sample=True,up=False,kernel_list=kernel_list)
        self.en2=EPFM(out_channels[1],out_channels[2],sample=True,up=False,kernel_list=kernel_list)
        self.en3=EPFM(out_channels[2],out_channels[3],sample=True,up=False,kernel_list=kernel_list)
        self.en4=EPFM(out_channels[3],out_channels[4],sample=True,up=False,kernel_list=kernel_list)

        # 解码部分：同样使用 EPFM 模块进行特征重建
        self.de1=EPFM(out_channels[1],out_channels[0],sample=True,up=True,kernel_list=kernel_list)
        self.de2=EPFM(out_channels[2],out_channels[1],sample=True,up=True,kernel_list=kernel_list)
        self.de3=EPFM(out_channels[3],out_channels[2],sample=True,up=True,kernel_list=kernel_list)
        self.de4=EPFM(out_channels[4],out_channels[3],sample=True,up=True,kernel_list=kernel_list)

        # Patch卷积：用于初步处理输入图像，降低通道数
        self.patch_conv=nn.Sequential(
            nn.Conv2d(input_channels,out_channels[0],3,padding=1),  # 3x3卷积
            nn.BatchNorm2d(out_channels[0])  # 批量归一化
        )

        # 预测部分：使用 PH 模块进行最终预测
        self.ph=PH(out_channels)
        
    def forward(self,x):
        # 处理输入图像，得到初步特征
        x=self.patch_conv(x)

        # 编码过程：逐层提取特征
        e1=self.en1(x)
        e2=self.en2(e1)
        e3=self.en3(e2)
        e4=self.en4(e3)

        # 解码过程：逐层重建特征，跳过连接
        d4=self.de4(e4)
        d3=self.de3(d4+e3)  # 跳过连接
        d2=self.de2(d3+e2)  # 跳过连接
        d1=self.de1(d2+e1)  # 跳过连接
        
        # 通过预测模块生成最终输出
        x_pre=self.ph([d4,d3,d2,d1])
        return x_pre



class PH_Block(nn.Module):
    def __init__(self,in_channels,scale_factor=1):
        super().__init__()
        # 根据 scale_factor 决定是否进行上采样
        if scale_factor>1:
            self.upsample=nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        else:
            self.upsample=None  # 不进行上采样

        # 1x1 卷积，将特征通道数降为1    
        self.pro=nn.Conv2d(in_channels,1,1)
        self.sig=nn.Sigmoid()  # 使用 Sigmoid 激活函数

    def forward(self,x):
        # 上采样（如果需要）
        if self.upsample!=None:
            x=self.upsample(x)

        # 通过卷积处理
        x=self.pro(x)
        # 应用 Sigmoid 激活函数
        x=self.sig(x)
        return x    # 返回处理后的特征

class PH(nn.Module):
    def __init__(self,in_channels=[12,24,36,48],scale_factor=[1,2,4,8]):
        super().__init__()
        # 创建多个 PH_Block，根据不同的输入通道数和上采样比例
        self.ph1=PH_Block(in_channels[0],scale_factor[0])
        self.ph2=PH_Block(in_channels[1],scale_factor[1])
        self.ph3=PH_Block(in_channels[2],scale_factor[2])
        self.ph4=PH_Block(in_channels[3],scale_factor[3])
        
    def forward(self,x):
        # 解包输入特征
        x4,x3,x2,x1=x
        # 逐个处理每个特征图
        x1=self.ph1(x1)
        x2=self.ph2(x2)
        x3=self.ph3(x3)
        x4=self.ph4(x4)
        return [x1,x2,x3,x4] 
