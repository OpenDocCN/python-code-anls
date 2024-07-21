# `.\pytorch\torch\testing\_internal\distributed\_shard\test_common.py`

```py
# 忽略 mypy 的错误，mypy 是一个用于检查 Python 类型的工具
# 导入 PyTorch 库
import torch
# 导入 PyTorch 中的神经网络模块
import torch.nn as nn
# 从 PyTorch 分布式包中导入 ShardedTensor 类
from torch.distributed._shard.sharded_tensor import ShardedTensor

# 定义一个简单的 MegatronLM 类，继承自 nn.Module
class SimpleMegatronLM(nn.Module):
    # 初始化方法，接收 linear_size、rank 和 dtype 参数
    def __init__(self, linear_size, rank=None, dtype=torch.float32):
        super().__init__()
        # 创建第一个全连接层，使用 linear_size 的第一个元素作为输入特征数，dtype 作为数据类型
        self.fc1 = nn.Linear(*linear_size[0], dtype=dtype)
        # 创建 GELU 激活函数层
        self.gelu = nn.GELU()
        # 创建第二个全连接层，使用 linear_size 的第二个元素作为输入特征数，dtype 作为数据类型
        self.fc2 = nn.Linear(*linear_size[1], dtype=dtype)
        # 如果 rank 参数不为 None，则将 fc1 和 fc2 移动到指定 GPU 设备上
        if rank is not None:
            self.fc1.cuda(rank)
            self.fc2.cuda(rank)

    # 前向传播方法，接收输入 inp，返回经过全连接层和 GELU 激活函数处理后的结果
    def forward(self, inp):
        return self.fc2(self.gelu(self.fc1(inp)))

    # 获取权重方法，返回 fc1 和 fc2 层的权重，如果是 ShardedTensor，则获取本地的张量
    def get_weights(self):
        if isinstance(self.fc1.weight, ShardedTensor):
            weight1 = self.fc1.weight.local_tensor()
        else:
            weight1 = self.fc1.weight

        if isinstance(self.fc2.weight, ShardedTensor):
            weight2 = self.fc2.weight.local_tensor()
        else:
            weight2 = self.fc2.weight

        return (weight1, weight2)

    # 获取偏置方法，返回 fc1 和 fc2 层的偏置
    def get_biases(self):
        return (self.fc1.bias, self.fc2.bias)

    # 获取权重梯度方法，返回 fc1 和 fc2 层的权重梯度
    def get_weight_grads(self):
        return (self.fc1.weight.grad, self.fc2.weight.grad)

    # 获取偏置梯度方法，返回 fc1 和 fc2 层的偏置梯度
    def get_bias_grads(self):
        return (self.fc1.bias.grad, self.fc2.bias.grad)
```