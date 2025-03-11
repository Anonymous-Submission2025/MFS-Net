import torch
import torch.nn as nn

class GCM(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        # PGC 模块
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
            nn.Conv2d(input_channels, input_channels, kernel_size=(5, 1), padding=(2, 0), groups=input_channels),  # 5x1深度可分离卷积
            nn.GELU(),  # 使用GELU激活函数
            nn.Conv2d(input_channels, input_channels, 1),  # 1x1卷积用于线性组合输出通道
            nn.BatchNorm2d(input_channels)  # 批归一化
        )
    
        self.sc_v = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=(1, 5), padding=(0, 2), groups=input_channels),  # 1x5深度可分离卷积
            nn.GELU(),  # 使用GELU激活函数
            nn.Conv2d(input_channels, input_channels, 1),  # 1x1卷积用于线性组合输出通道
            nn.BatchNorm2d(input_channels)  # 批归一化
        )

        self.conv_h = nn.Conv2d(input_channels, input_channels, 1)  # 1x1卷积，用于融合高频特征
        
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)# 平均池化层，用于下采样
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)# 上采样层，使用双线性插值方法

    def forward_lpg(self, x):
        # 生成低频提示
        x_gap = self.gap(x) * x  # 自适应平均池化后与输入x相乘，得到低频特征
        B, C, H, W = x_gap.shape  # 获取批量大小、通道数、高度和宽度
        N = H * W  # 计算特征图的总像素数
        x_gap = x_gap.view(B, C, N).permute(0, 2, 1).contiguous()  # 变形为(B, N, C)格式
        x_Q = self.Q_proj(x_gap)  # 生成查询向量
        x_K = self.K_proj(x_gap)  # 生成键向量
        x_V = self.sig(x_Q) * x_K + x_gap  # 计算值向量，结合查询和键，加入原始低频特征
        prompt_l = self.liner_l(x_V)  # 通过线性层得到低频提示
        prompt_l = prompt_l.permute(0, 2, 1).contiguous().view(B, C, H, W)  # 变形回(B, C, H, W)格式
        return prompt_l  # 返回低频提示

    def forward_hpg(self, x):
        # 生成高频提示
        x_h = self.sc_h(x)  # 通过水平卷积生成高频特征
        x_v = self.sc_v(x)  # 通过垂直卷积生成高频特征
        prompt_h = self.conv_h(x_h + x_v)  # 将水平和垂直特征相加后通过1x1卷积得到高频提示
        return prompt_h  # 返回高频提示

    def forward_pgc(self, prompt_l, prompt_h, x_ori):
        # 嵌入提示到交叉自注意力
        B, C, H, W = x_ori.shape  # 获取原始输入的形状
        N = H * W  # 计算特征图的总像素数
        x_V = self.linear_o(x_ori.view(B, C, N).permute(0, 2, 1).contiguous())  # 变形输入为(B, N, C)并通过线性层得到值向量
        x_K = prompt_l.view(B, C, N).permute(0, 2, 1).contiguous()  # 变形低频提示为(B, N, C)
        x_Q = prompt_h.view(B, C, N).permute(0, 2, 1).contiguous()  # 变形高频提示为(B, N, C)
        x_attn = x_Q @ x_K.transpose(1, 2)  # 计算注意力矩阵(B, N, N)
        prompt = self.soft(x_attn * self.scale) @ x_V  # 应用软max和缩放得到注意力加权的值
        prompt = self.linear_p(prompt) + x_V  # 通过线性层得到最终提示，并与原始值相加
        p_norm = self.norm(prompt)  # 对提示进行归一化处理
        p_norm = p_norm.permute(0, 2, 1).contiguous().view(B, C, H, W)  # 变形回(B, C, H, W)格式
        p_norm = self.up(p_norm)  # 上采样到原始输入尺寸
        out = self.dsc(p_norm) + p_norm  # 通过深度可分离卷积处理并与上采样的提示相加
        return out  # 返回增强后的输出

    def forward(self, x: torch.Tensor):
        x = self.pool(x)
        prompt_l = self.forward_lpg(x)  # 生成低频提示
        prompt_h = self.forward_hpg(x)  # 生成高频提示
        out = self.forward_pgc(prompt_l, prompt_h, x)  # 嵌入提示到交叉自注意力
        return out

# 示例用法
if __name__ == "__main__":
    input_tensor = torch.randn(8, 3, 32, 32)  # 8个样本，3个通道，32x32的输入图像
    model = GCM(input_channels=3)
    output = model(input_tensor)
    print(output.shape)  # 应该输出 (8, 3, 32, 32)，表示8个样本的输出
