# `.\pytorch\test\onnx\model_defs\mnist.py`

```py
import torch.nn as nn
import torch.nn.functional as F

# 定义一个名为 MNIST 的神经网络模型，继承自 nn.Module 类
class MNIST(nn.Module):
    # 初始化函数，定义网络结构的各个层
    def __init__(self):
        super().__init__()
        # 第一个卷积层，输入通道数为1（单通道灰度图像），输出通道数为10，卷积核大小为5x5
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # 第二个卷积层，输入通道数为10，输出通道数为20，卷积核大小为5x5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # Dropout2d 层，用于卷积层的二维空间dropout
        self.conv2_drop = nn.Dropout2d()
        # 全连接层，输入特征数为320（计算得来），输出特征数为50
        self.fc1 = nn.Linear(320, 50)
        # 全连接层，输入特征数为50，输出特征数为10（类别数目，对应MNIST的0-9）
        self.fc2 = nn.Linear(50, 10)

    # 前向传播函数，定义数据从输入到输出的流程
    def forward(self, x):
        # 第一层卷积后经过 relu 激活函数和最大池化层
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # 第二层卷积后经过 relu 激活函数、dropout 和最大池化层
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # 将特征张量展平为一维
        x = x.view(-1, 320)
        # 全连接层经过 relu 激活函数
        x = F.relu(self.fc1(x))
        # 使用 dropout 进行正则化
        x = F.dropout(x, training=self.training)
        # 输出层，不经过激活函数，输出log_softmax
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```