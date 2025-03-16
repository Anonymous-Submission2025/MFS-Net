import torch
import torch.nn as nn

class DualDynamicFusion(nn.Module):
    def __init__(self, input_channels):
        super(DualDynamicFusion, self).__init__()
        self.linear_o = nn.Linear(input_channels, input_channels // 2)
        self.linear_p = nn.Linear(input_channels // 2, input_channels // 2)
        self.scale = (input_channels // 2) ** -0.5
        self.norm = nn.LayerNorm(input_channels // 2)
        self.soft = nn.Softmax(-1)

    def forward(self, prompt_l, prompt_h, x_ori):
        B, C, H, W = x_ori.shape
        N = H * W
        x_V = self.linear_o(x_ori.view(B, C, N).permute(0, 2, 1).contiguous())
        x_K = prompt_l.view(B, C, N).permute(0, 2, 1).contiguous()
        x_Q = prompt_h.view(B, C, N).permute(0, 2, 1).contiguous()
        
        x_attn = x_Q @ x_K.transpose(1, 2)
        prompt = self.soft(x_attn * self.scale) @ x_V
        prompt = self.linear_p(prompt) + x_V
        
        p_norm = self.norm(prompt)
        p_norm = p_norm.permute(0, 2, 1).contiguous().view(B, C, H, W)
        return p_norm

class CPGM(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = input_channels // 2  # 减少输出通道数
        self.scale = self.output_channels ** -0.5
        self.norm = nn.LayerNorm(self.output_channels)
        self.soft = nn.Softmax(-1)

        # 深度可分离卷积层
        self.dsc = nn.Sequential(
            nn.Conv2d(input_channels, self.output_channels, 3, padding=1, groups=input_channels),  # 深度可分离卷积
            nn.ReLU(),
            nn.Conv2d(self.output_channels, self.output_channels, 1),  # 1x1卷积
            nn.BatchNorm2d(self.output_channels)
        )

        # LPG 部分
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.Q_proj = nn.Linear(self.output_channels, self.output_channels)
        self.K_proj = nn.Linear(self.output_channels, self.output_channels)
        self.liner_l = nn.Linear(self.output_channels, self.output_channels)
        self.sig = nn.Sigmoid()

        # HPG 部分
        self.sc_h = nn.Sequential(
            nn.Conv2d(self.output_channels, self.output_channels, kernel_size=(3, 1), padding=(1, 0), groups=self.output_channels),
            nn.ReLU(),
            nn.Conv2d(self.output_channels, self.output_channels, 1),
            nn.BatchNorm2d(self.output_channels)
        )
        
        self.sc_v = nn.Sequential(
            nn.Conv2d(self.output_channels, self.output_channels, kernel_size=(1, 3), padding=(0, 1), groups=self.output_channels),
            nn.ReLU(),
            nn.Conv2d(self.output_channels, self.output_channels, 1),
            nn.BatchNorm2d(self.output_channels)
        )

        self.conv_h = nn.Conv2d(self.output_channels, self.output_channels, 1)
        
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 添加 DDF 模块
        self.ddf = DualDynamicFusion(self.output_channels)

    def forward_lpg(self, x):
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
        x_h = self.sc_h(x)
        x_v = self.sc_v(x)
        prompt_h = self.conv_h(x_h + x_v)
        return prompt_h

    def forward(self, x: torch.Tensor):
        original_size = x.shape[2:]
        x = self.pool(x)  # 池化操作会将空间维度减半
        prompt_l = self.forward_lpg(x)  # 生成低频提示
        prompt_h = self.forward_hpg(x)  # 生成高频提示
        
        # 确保 prompt_l 和 prompt_h 的形状一致
        if prompt_l.shape[2:] != prompt_h.shape[2:]:
            prompt_h = nn.functional.interpolate(prompt_h, size=prompt_l.shape[2:], mode='bilinear', align_corners=True)

        out = self.ddf(prompt_l, prompt_h, x)  # 使用 DDF 融合特征
        out = self.dsc(out) + out  # 处理后的输出与原始输出相加
        
        # 上采样回原始输入大小
        out = self.up(out)  # 使用上采样
        return out  # 输出保持 (B, C, H, W) 的形状

# 示例用法
if __name__ == "__main__":
    input_tensor = torch.randn(8, 3, 32, 32)  # 8个样本，3个通道，32x32的输入图像
    model = CPGM(input_channels=3)  # 确保输入通道数一致
    output = model(input_tensor)
    print(output.shape)  # 应该输出 (8, 3, 32, 32)，表示8个样本的输出
