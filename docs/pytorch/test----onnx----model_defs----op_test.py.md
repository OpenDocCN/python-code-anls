# `.\pytorch\test\onnx\model_defs\op_test.py`

```py
# Owner(s): ["module: onnx"]

# 导入 PyTorch 库
import torch
# 导入 PyTorch 的神经网络模块
import torch.nn as nn


# 定义一个名为 DummyNet 的神经网络模型，继承自 nn.Module
class DummyNet(nn.Module):
    # 构造函数，初始化网络结构
    def __init__(self, num_classes=1000):
        super().__init__()
        # 定义网络的特征提取层，使用 nn.Sequential 封装多个层
        self.features = nn.Sequential(
            nn.LeakyReLU(0.02),  # LeakyReLU 激活函数，负斜率系数为 0.02
            nn.BatchNorm2d(3),   # 批标准化层，输入通道数为 3
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),  # 平均池化层，3x3 大小，步长 2，填充 1，非上取整模式
        )

    # 前向传播函数，定义数据从输入到输出的流程
    def forward(self, x):
        output = self.features(x)  # 将输入 x 通过特征提取层得到输出
        return output.view(-1, 1).squeeze(1)  # 将输出视图重塑成指定形状并去除维度为 1 的维度


# 定义一个名为 ConcatNet 的神经网络模型，继承自 nn.Module
class ConcatNet(nn.Module):
    # 前向传播函数，定义数据从输入到输出的流程
    def forward(self, inputs):
        return torch.cat(inputs, 1)  # 拼接输入列表中的张量，沿着第二个维度（列维度）


# 定义一个名为 PermuteNet 的神经网络模型，继承自 nn.Module
class PermuteNet(nn.Module):
    # 前向传播函数，定义数据从输入到输出的流程
    def forward(self, input):
        return input.permute(2, 3, 0, 1)  # 对输入张量进行维度置换，调整顺序为 (2, 3, 0, 1)


# 定义一个名为 PReluNet 的神经网络模型，继承自 nn.Module
class PReluNet(nn.Module):
    # 构造函数，初始化网络结构
    def __init__(self):
        super().__init__()
        # 定义网络的特征提取层，使用 nn.Sequential 封装一个 PReLU 激活函数层，输入通道数为 3
        self.features = nn.Sequential(
            nn.PReLU(3),
        )

    # 前向传播函数，定义数据从输入到输出的流程
    def forward(self, x):
        output = self.features(x)  # 将输入 x 通过特征提取层得到输出
        return output  # 返回输出


# 定义一个名为 FakeQuantNet 的神经网络模型，继承自 nn.Module
class FakeQuantNet(nn.Module):
    # 构造函数，初始化网络结构
    def __init__(self):
        super().__init__()
        # 初始化一个 FakeQuantize 对象
        self.fake_quant = torch.ao.quantization.FakeQuantize()
        self.fake_quant.disable_observer()  # 禁用量化观察器

    # 前向传播函数，定义数据从输入到输出的流程
    def forward(self, x):
        output = self.fake_quant(x)  # 对输入 x 进行量化操作
        return output  # 返回输出
```