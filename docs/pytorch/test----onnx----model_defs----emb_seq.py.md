# `.\pytorch\test\onnx\model_defs\emb_seq.py`

```py
# 导入 PyTorch 的神经网络模块
import torch.nn as nn

# 定义一个名为 EmbeddingNetwork1 的神经网络类，继承自 nn.Module
class EmbeddingNetwork1(nn.Module):
    # 初始化函数，接受一个可选参数 dim，默认为 5
    def __init__(self, dim=5):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个 Embedding 层，输入大小为 10，输出大小为 dim
        self.emb = nn.Embedding(10, dim)
        # 创建一个线性层，输入大小为 dim，输出大小为 1
        self.lin1 = nn.Linear(dim, 1)
        # 创建一个顺序容器 Sequential，依次包括上述的 Embedding 层和线性层
        self.seq = nn.Sequential(
            self.emb,
            self.lin1,
        )

    # 前向传播函数，接受输入 input
    def forward(self, input):
        # 将输入 input 通过顺序容器 seq 进行前向传播并返回结果
        return self.seq(input)


# 定义一个名为 EmbeddingNetwork2 的神经网络类，继承自 nn.Module
class EmbeddingNetwork2(nn.Module):
    # 初始化函数，接受两个可选参数 in_space 和 dim，默认分别为 10 和 3
    def __init__(self, in_space=10, dim=3):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个 Embedding 层，输入大小为 in_space，输出大小为 dim
        self.embedding = nn.Embedding(in_space, dim)
        # 创建一个顺序容器 Sequential，依次包括上述的 Embedding 层、线性层和 Sigmoid 激活函数层
        self.seq = nn.Sequential(self.embedding, nn.Linear(dim, 1), nn.Sigmoid())

    # 前向传播函数，接受输入 indices
    def forward(self, indices):
        # 将输入 indices 通过顺序容器 seq 进行前向传播并返回结果
        return self.seq(indices)
```