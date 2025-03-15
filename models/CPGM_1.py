import torch
import torch.nn as nn

class DynamicAttention(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.query_proj = nn.Linear(input_channels, input_channels)
        self.key_proj = nn.Linear(input_channels, input_channels)
        self.value_proj = nn.Linear(input_channels, input_channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        # 计算注意力权重
        q = self.query_proj(q)
        k = self.key_proj(k)
        v = self.value_proj(v)

        attn_weights = self.softmax(q @ k.transpose(-2, -1))
        attended_values = attn_weights @ v
        return attended_values

class EEN(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.scale = input_channels ** -0.5
        self.norm = nn.LayerNorm(input_channels)

        # 深度可分离卷积层定义
        self.dsc = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, padding=1, groups=input_channels),  # 深度可分离卷积
            nn.GELU(),
            nn.Conv2d(input_channels, input_channels, 1),
            nn.BatchNorm2d(input_channels)
        )

        # LPG 部分
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.Q_proj = nn.Linear(input_channels, input_channels)
        self.K_proj = nn.Linear(input_channels, input_channels)
        self.liner_l = nn.Linear(input_channels, input_channels)
        self.sig = nn.Sigmoid()

        # HPG 部分
        self.sc_h = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=(5, 1), padding=(2, 0), groups=input_channels),
            nn.GELU(),
            nn.Conv2d(input_channels, input_channels, 1),
            nn.BatchNorm2d(input_channels)
        )
    
        self.sc_v = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=(1, 5), padding=(0, 2), groups=input_channels),
            nn.GELU(),
            nn.Conv2d(input_channels, input_channels, 1),
            nn.BatchNorm2d(input_channels)
        )

        self.conv_h = nn.Conv2d(input_channels, input_channels, 1)
        
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 动态双向注意力模块
        self.dynamic_attention = DynamicAttention(input_channels)

    def forward_lpg(self, x):
        x_gap = self.gap(x) * x
        B, C, H, W = x_gap.shape
        N = H * W
        x_gap = x_gap.view(B, C, N).permute(0, 2, 1).contiguous()
        prompt_l = self.liner_l(x_gap)
        prompt_l = prompt_l.permute(0, 2, 1).contiguous().view(B, C, H, W)
        return prompt_l

    def forward_hpg(self, x):
        x_h = self.sc_h(x)
        x_v = self.sc_v(x)
        prompt_h = self.conv_h(x_h + x_v)
        return prompt_h

    def forward_pgc(self, prompt_l, prompt_h, x_ori):
        B, C, H, W = x_ori.shape
        N = H * W

    # 生成注意力特征
        attended_prompt_l = self.dynamic_attention(prompt_h.view(B, C, N).permute(0, 2, 1).contiguous(),
                                                prompt_l.view(B, C, N).permute(0, 2, 1).contiguous(),
                                                prompt_l.view(B, C, N).permute(0, 2, 1).contiguous())
    
        attended_prompt_h = self.dynamic_attention(prompt_l.view(B, C, N).permute(0, 2, 1).contiguous(),
                                                prompt_h.view(B, C, N).permute(0, 2, 1).contiguous(),
                                                prompt_h.view(B, C, N).permute(0, 2, 1).contiguous())

    # 融合处理
        final_prompt = attended_prompt_l + attended_prompt_h
    
    # 确保 final_prompt 的形状为 (B, N, C)
        final_prompt = final_prompt.permute(0, 2, 1).contiguous()  # 变为 (B, C, N)
        final_prompt = final_prompt.view(B, C, H, W)  # 变为 (B, C, H, W)

    # 归一化处理
        p_norm = self.norm(final_prompt.view(B, -1, C))  # 将形状调整为 (B, H*W, C) 进行归一化
        p_norm = p_norm.view(B, C, H, W)  # 再次调整为 (B, C, H, W)

        p_norm = self.up(p_norm)  # 上采样到原始输入尺寸

        out = self.dsc(p_norm) + p_norm  # 通过深度可分离卷积处理并与上采样的提示相加
        return out

    def forward(self, x: torch.Tensor):
        x = self.pool(x)
        prompt_l = self.forward_lpg(x)
        prompt_h = self.forward_hpg(x)
        out = self.forward_pgc(prompt_l, prompt_h, x)
        return out

# 示例用法
if __name__ == "__main__":
    input_tensor = torch.randn(8, 3, 32, 32)
    model = EEN(input_channels=3)
    output = model(input_tensor)
    print(output.shape)  # 应该输出 (8, 3, 32, 32)，表示8个样本的输出
