# `.\pytorch\test\onnx\model_defs\super_resolution.py`

```
# 导入 PyTorch 神经网络模块和初始化模块
import torch.nn as nn
import torch.nn.init as init

# 定义一个用于超分辨率的神经网络类，继承自 nn.Module
class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()

        # 定义激活函数 ReLU
        self.relu = nn.ReLU()
        
        # 第一层卷积：输入通道 1，输出通道 64，卷积核大小 5x5，步长 1，填充 2
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        
        # 第二层卷积：输入通道 64，输出通道 64，卷积核大小 3x3，步长 1，填充 1
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        
        # 第三层卷积：输入通道 64，输出通道 32，卷积核大小 3x3，步长 1，填充 1
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        
        # 第四层卷积：输入通道 32，输出通道为 upscale_factor^2，卷积核大小 3x3，步长 1，填充 1
        self.conv4 = nn.Conv2d(32, upscale_factor**2, (3, 3), (1, 1), (1, 1))
        
        # 像素重排模块，用于上采样
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        # 调用初始化权重函数
        self._initialize_weights()

    # 前向传播函数
    def forward(self, x):
        x = self.relu(self.conv1(x))  # ReLU 激活函数应用在第一层卷积后
        x = self.relu(self.conv2(x))  # ReLU 激活函数应用在第二层卷积后
        x = self.relu(self.conv3(x))  # ReLU 激活函数应用在第三层卷积后
        x = self.pixel_shuffle(self.conv4(x))  # 像素重排应用在第四层卷积后
        return x

    # 初始化网络权重的函数
    def _initialize_weights(self):
        # 使用正交初始化方法初始化第一层卷积的权重，并按 ReLU 的增益因子调整
        init.orthogonal_(self.conv1.weight, init.calculate_gain("relu"))
        # 使用正交初始化方法初始化第二层卷积的权重，并按 ReLU 的增益因子调整
        init.orthogonal_(self.conv2.weight, init.calculate_gain("relu"))
        # 使用正交初始化方法初始化第三层卷积的权重，并按 ReLU 的增益因子调整
        init.orthogonal_(self.conv3.weight, init.calculate_gain("relu"))
        # 使用正交初始化方法初始化第四层卷积的权重
        init.orthogonal_(self.conv4.weight)
```