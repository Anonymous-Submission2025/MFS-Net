import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 nn 模块，提供神经网络组件
import torch.nn.functional as F  # 导入功能性操作模块，提供激活函数等

# 特征增强模块
class LearnableSobel(nn.Module):
    def __init__(self, in_channels):
        super(LearnableSobel, self).__init__()  # 初始化父类
        # 创建可学习的 Sobel 核心，分别为水平和垂直方向
        self.kernel_h = nn.Parameter(torch.Tensor(in_channels, 1, 3, 3), requires_grad=True)  # 水平方向的 Sobel 核心
        self.kernel_v = nn.Parameter(torch.Tensor(in_channels, 1, 3, 3), requires_grad=True)  # 垂直方向的 Sobel 核心

        # 初始化 Sobel 核心的值
        self.initialize_kernels(in_channels)

    def initialize_kernels(self, in_channels):
        # 初始化 Sobel 核心的参数
        # 创建水平 Sobel 核心
        sobel_h = torch.tensor([[[-1, 0, 1],   # Sobel 水平核的数值
                                  [-2, 0, 2],
                                  [-1, 0, 1]]], dtype=torch.float32).expand(in_channels, -1, -1, -1)  # 扩展到输入通道数
        
        # 创建垂直 Sobel 核心
        sobel_v = torch.tensor([[[-1, -2, -1],  # Sobel 垂直核的数值
                                  [0, 0, 0],
                                  [1, 2, 1]]], dtype=torch.float32).expand(in_channels, -1, -1, -1)  # 扩展到输入通道数

        # 将初始化的 Sobel 核心赋值给可学习的参数
        self.kernel_h.data.copy_(sobel_h)  # 复制水平 Sobel 核心数据
        self.kernel_v.data.copy_(sobel_v)  # 复制垂直 Sobel 核心数据

    def forward(self, x):
        # 执行可学习的 Sobel 卷积
        edge_h = F.conv2d(x, self.kernel_h, padding=1, groups=x.size(1))  # 水平边缘检测
        edge_v = F.conv2d(x, self.kernel_v, padding=1, groups=x.size(1))  # 垂直边缘检测
        edges = torch.sqrt(edge_h ** 2 + edge_v ** 2)  # 计算边缘强度，使用欧几里得范数
        return edges  # 返回边缘信息


class EEN(nn.Module):
    def __init__(self, in_channels):
        super(EEN, self).__init__()  # 初始化父类
        self.learnable_sobel = LearnableSobel(in_channels)  # 实例化可学习的 Sobel 模块

        # 添加后续处理层，例如卷积层
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)  # 第一卷积层
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)  # 第二卷积层

    def forward(self, x):
        edges = self.learnable_sobel(x)  # 获取边缘信息
        enhanced = x + edges  # 边缘增强，将原始输入与边缘信息相加

        # 进一步处理通过卷积层
        enhanced = F.relu(self.conv1(enhanced))  # 通过第一卷积层并使用 ReLU 激活
        enhanced = self.conv2(enhanced)  # 通过第二卷积层

        return enhanced  # 返回增强后的特征图


# 示例训练框架
def train(model, dataloader, optimizer, criterion, epochs=10):
    model.train()  # 设置模型为训练模式
    for epoch in range(epochs):  # 遍历每个训练周期
        for images, targets in dataloader:  # 遍历数据加载器
            optimizer.zero_grad()  # 清除梯度
            outputs = model(images)  # 前向传播，获取模型输出
            loss = criterion(outputs, targets)  # 计算损失
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新模型参数
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")  # 输出当前周期的损失


# 主程序
if __name__ == "__main__":
    in_channels = 3  # 假设输入图像的通道数，例如 RGB 图像
    model = EEN(in_channels)  # 实例化 EEN 模型
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 使用 Adam 优化器
    criterion = nn.MSELoss()  # 选择均方误差损失函数

    # 假设 dataloader 是一个 PyTorch DataLoader 实例
    # train(model, dataloader, optimizer, criterion, epochs=10)  # 进行训练
