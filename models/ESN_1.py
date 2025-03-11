import torch
import torch.nn as nn
import torch.nn.functional as F


#特征增强模块

class LearnableSobel(nn.Module):
    def __init__(self, in_channels):
        super(LearnableSobel, self).__init__()
        # 创建可学习的 Sobel 核心
        self.kernel_h = nn.Parameter(torch.Tensor(in_channels, 1, 3, 3), requires_grad=True)
        self.kernel_v = nn.Parameter(torch.Tensor(in_channels, 1, 3, 3), requires_grad=True)

        self.initialize_kernels(in_channels)

    def initialize_kernels(self, in_channels):
        # 初始化 Sobel 核心
        sobel_h = torch.tensor([[[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]]], dtype=torch.float32).expand(in_channels, -1, -1, -1)
        sobel_v = torch.tensor([[[-1, -2, -1],
                                  [0, 0, 0],
                                  [1, 2, 1]]], dtype=torch.float32).expand(in_channels, -1, -1, -1)

        self.kernel_h.data.copy_(sobel_h)
        self.kernel_v.data.copy_(sobel_v)

    def forward(self, x):
        # 执行可学习的 Sobel 卷积
        edge_h = F.conv2d(x, self.kernel_h, padding=1, groups=x.size(1))
        edge_v = F.conv2d(x, self.kernel_v, padding=1, groups=x.size(1))
        edges = torch.sqrt(edge_h ** 2 + edge_v ** 2)
        return edges


class EEN(nn.Module):
    def __init__(self, in_channels):
        super(EEN, self).__init__()
        self.learnable_sobel = LearnableSobel(in_channels)

        # 这里可以添加后续处理层，例如卷积层
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        edges = self.learnable_sobel(x)  # 获取边缘信息
        enhanced = x + edges  # 边缘增强，您可以选择其他的融合方式

        # 进一步处理
        enhanced = F.relu(self.conv1(enhanced))
        enhanced = self.conv2(enhanced)
        
        
        return enhanced


# 示例训练框架
def train(model, dataloader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        for images, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")


# 主程序
if __name__ == "__main__":
    # 假设 in_channels 是输入图像的通道数
    in_channels = 3  # 例如 RGB 图像
    model = EEN(in_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()  # 根据具体任务选择损失函数

    # 假设 dataloader 是一个 PyTorch DataLoader 实例
    # train(model, dataloader, optimizer, criterion, epochs=10)
