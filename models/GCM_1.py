import torch
import torch.nn as nn


class GCM(nn.Module):
    def __init__(self, input_channels=16):  # 减小通道数以降低内存使用
        super().__init__()
        # PGC
        self.linear_o = nn.Linear(input_channels, input_channels)
        self.linear_p = nn.Linear(input_channels, input_channels)
        self.scale = input_channels ** -0.5
        self.norm = nn.LayerNorm(input_channels)
        self.soft = nn.Softmax(-1)
        self.dsc = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, padding=1, groups=input_channels),
            nn.GELU(),
            nn.Conv2d(input_channels, input_channels, 1),
            nn.BatchNorm2d(input_channels)
        )

        # LPG
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.Q_proj = nn.Linear(input_channels, input_channels)
        self.K_proj = nn.Linear(input_channels, input_channels)
        self.liner_l = nn.Linear(input_channels, input_channels)
        self.sig = nn.Sigmoid()

        # HPG
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

    def forward_lpg(self, x):
        x_gap = self.gap(x) * x
        B, C, H, W = x_gap.shape
        N = H * W
        x_gap = x_gap.view(B, C, N).permute(0, 2, 1).contiguous()  # B N C
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

    def forward_pgc(self, prompt_l, prompt_h, x_ori):
        B, C, H, W = x_ori.shape
        N = H * W

        # 获取高频特征对低频特征的注意力
        x_K_l = prompt_l.view(B, C, N).permute(0, 2, 1).contiguous()  # B N C
        x_Q_h = prompt_h.view(B, C, N).permute(0, 2, 1).contiguous()  # B N C
        attn_h_to_l = x_Q_h @ x_K_l.transpose(1, 2)  # B N N
        attn_h_to_l = self.soft(attn_h_to_l * self.scale)

        # 获取低频特征对高频特征的注意力
        x_K_h = prompt_h.view(B, C, N).permute(0, 2, 1).contiguous()  # B N C
        x_Q_l = prompt_l.view(B, C, N).permute(0, 2, 1).contiguous()  # B N C
        attn_l_to_h = x_Q_l @ x_K_h.transpose(1, 2)  # B N N
        attn_l_to_h = self.soft(attn_l_to_h * self.scale)

        # 动态加权
        x_V = self.linear_o(x_ori.view(B, C, N).permute(0, 2, 1).contiguous())  # B N C
        prompt_h_contrib = attn_h_to_l @ x_V  # B N C
        prompt_l_contrib = attn_l_to_h @ x_V  # B N C
        
        # 融合高频和低频贡献
        prompt = prompt_h_contrib + prompt_l_contrib
        prompt = self.linear_p(prompt) + x_V  # 加上原始的 V
        p_norm = self.norm(prompt)
        p_norm = p_norm.permute(0, 2, 1).contiguous().view(B, C, H, W)
        p_norm = self.up(p_norm)
        out = self.dsc(p_norm) + p_norm
        return out

    def forward(self, x: torch.Tensor):
        x = self.pool(x)
        prompt_l = self.forward_lpg(x)  # generate low-frequency prompts
        prompt_h = self.forward_hpg(x)  # generate high-frequency prompts
        out = self.forward_pgc(prompt_l, prompt_h, x)  # embed prompts into cross self-attention
        return out
