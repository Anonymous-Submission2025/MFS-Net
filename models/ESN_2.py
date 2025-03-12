import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 nn 模块，提供神经网络组件
import torch.nn.functional as F  # 导入功能性操作模块，提供激活函数等操作

# 可学习的 Sobel 特征提取模块
class LearnableSobel(nn.Module):
    def __init__(self, in_channels):
        super(LearnableSobel, self).__init__()  # 初始化父类
        # 创建可学习的 Sobel 核心，分别为水平和垂直方向
        self.kernel_h = nn.Parameter(torch.Tensor(in_channels, 1, 3, 3), requires_grad=True)  # 水平方向的 Sobel 核心
        self.kernel_v = nn.Parameter(torch.Tensor(in_channels, 1, 3, 3), requires_grad=True)  # 垂直方向的 Sobel 核心
        self.initialize_kernels(in_channels)  # 初始化 Sobel 核心的权重

    def initialize_kernels(self, in_channels):
        # 初始化 Sobel 核心的权重
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


# 深度可分离卷积模块
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels):
        super(DepthwiseSeparableConv, self).__init__()  # 初始化父类
        # 创建深度卷积层，使用 groups=in_channels 实现深度可分离卷积
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)  # 深度卷积
        # 创建逐点卷积层，用于进一步压缩特征
        self.bn=nn.BatchNorm2d(in_channels)
        self.act=nn.GELU()
        self.pointwise = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 逐点卷积

    def forward(self, x):
        # 执行深度可分离卷积
        return self.pointwise(self.act(self.bn(self.depthwise(x)) )) # 先深度卷积再逐点卷积


# 特征增强网络
class EEN(nn.Module):
    def __init__(self, in_channels):
        super(EEN, self).__init__()  # 初始化父类
        self.learnable_sobel = LearnableSobel(in_channels)  # 实例化可学习的 Sobel 模块

        # 使用深度可分离卷积
        self.depthwise_separable_conv1 = DepthwiseSeparableConv(in_channels)  # 第一层深度可分离卷积
        self.bn1 = nn.BatchNorm2d(in_channels)  # 第一层 Batch Normalization
        self.depthwise_separable_conv2 = DepthwiseSeparableConv(in_channels)  # 第二层深度可分离卷积
        self.bn2 = nn.BatchNorm2d(in_channels)  # 第二层 Batch Normalization

    def forward(self, x):
        edges = self.learnable_sobel(x)  # 获取边缘信息
        enhanced = x + edges  # 边缘增强，将原始输入与边缘信息相加

        # 经过第一层深度可分离卷积和 Batch Normalization，并使用 ReLU 激活
        enhanced = F.relu(self.bn1(self.depthwise_separable_conv1(enhanced)))
        # 经过第二层深度可分离卷积和 Batch Normalization
        enhanced = self.bn2(self.depthwise_separable_conv2(enhanced))

        # 应用 Sigmoid 激活函数，确保输出在 [0, 1] 范围内
        enhanced = torch.sigmoid(enhanced)

        return enhanced  # 返回增强后的特征图


# 示例训练框架
def train(model, dataloader, optimizer, criterion, epochs=10):
    model.train()  # 设置模型为训练模式
    for epoch in range(epochs):  # 遍历每个训练周期
        for images, targets in dataloader:  # 遍历数据加载器中的图像和目标
            optimizer.zero_grad()  # 清除梯度
            outputs = model(images)  # 前向传播，获取模型输出
            loss = criterion(outputs, targets)  # 计算损失
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新模型参数
        # 输出当前周期的损失
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")  

# 主程序
if __name__ == "__main__":
    in_channels = 3  # 假设输入图像的通道数，例如 RGB 图像
    model = EEN(in_channels)  # 实例化 EEN 模型
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 使用 Adam 优化器
    criterion = nn.MSELoss()  # 选择均方误差损失函数

    # 假设 dataloader 是一个 PyTorch DataLoader 实例
    # train(model, dataloader, optimizer, criterion, epochs=10)  # 进行训练
