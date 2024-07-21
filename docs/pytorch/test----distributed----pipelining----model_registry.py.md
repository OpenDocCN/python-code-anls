# `.\pytorch\test\distributed\pipelining\model_registry.py`

```py
# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
# This file is a model zoo for testing torch.distributed.pipelining.
import torch
from torch.distributed.pipelining import pipe_split, SplitPoint

# 定义一个示例模型，继承自 torch.nn.Module
class ExampleCode(torch.nn.Module):
    def __init__(self, d_hid):
        super().__init__()
        # 定义两个模型参数，并将其声明为可学习的参数
        self.mm_param0 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param1 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        # 将一个张量作为缓冲区（不需要梯度），用随机值初始化
        self.register_buffer("cval", torch.randn((d_hid,), requires_grad=False))
        # 定义两个全连接层
        self.lin0 = torch.nn.Linear(d_hid, d_hid)
        self.lin1 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        # 矩阵乘法操作
        x = torch.mm(x, self.mm_param0)
        # 使用 ReLU 激活函数
        x = torch.relu(x)
        # 创建一个不需要梯度传播的常量张量
        a_constant = self.cval.clone()
        # 第一个线性层
        x = self.lin0(x)
        # 分割流水线（pipelining）
        pipe_split()
        # 使用 ReLU 激活函数，并加上之前创建的常量张量
        x = torch.relu(x) + a_constant
        # 矩阵乘法操作
        x = torch.mm(x, self.mm_param1)
        # 第二个线性层
        x = self.lin1(x)
        # 使用 ReLU 激活函数
        x = torch.relu(x)
        # 返回结果张量
        return x


# 定义一个带有默认参数的模型类
class ModelWithKwargs(torch.nn.Module):
    DEFAULT_DHID = 512
    DEFAULT_BATCH_SIZE = 256

    def __init__(self, d_hid: int = DEFAULT_DHID):
        super().__init__()
        # 定义两个模型参数，并将其声明为可学习的参数
        self.mm_param0 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param1 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        # 定义两个全连接层
        self.lin0 = torch.nn.Linear(d_hid, d_hid)
        self.lin1 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x, y=torch.zeros(DEFAULT_BATCH_SIZE, DEFAULT_DHID)):
        # 矩阵乘法操作
        x = torch.mm(x, self.mm_param0)
        # 将输入张量 x 和 y 相加
        x = x + y
        # 第一个线性层
        x = self.lin0(x)
        # 使用 ReLU 激活函数
        x = torch.relu(x)
        # 分割流水线（pipelining）
        pipe_split()
        # 矩阵乘法操作
        x = torch.mm(x, self.mm_param1)
        # 第二个线性层
        x = self.lin1(x)
        # 使用 ReLU 激活函数
        x = torch.relu(x)
        # 返回结果张量
        return x


# 定义一个带有参数别名的模型类
class ModelWithParamAlias(torch.nn.Module):
    default_dhid = 512
    default_batch_size = 256

    def __init__(self, d_hid: int = default_dhid):
        super().__init__()
        # 定义两个模型参数，并使用参数别名
        self.mm_param1 = self.mm_param0 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        # 定义两个全连接层，并使用参数别名
        self.lin1 = self.lin0 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x, y):
        # 矩阵乘法操作
        x = torch.mm(x, self.mm_param0)
        # 将输入张量 x 和 y 相加
        x = x + y
        # 第一个线性层
        x = self.lin0(x)
        # 使用 ReLU 激活函数
        x = torch.relu(x)
        # 分割流水线（pipelining）
        pipe_split()
        # 矩阵乘法操作
        x = torch.mm(x, self.mm_param1)
        # 第二个线性层
        x = self.lin1(x)
        # 使用 ReLU 激活函数
        x = torch.relu(x)
        # 返回结果张量
        return x


# 定义一个多层感知机模型
class MLPModule(torch.nn.Module):
    def __init__(self, d_hid: int):
        super().__init__()
        # 定义两个全连接层和 ReLU 激活函数
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        # 第一个全连接层
        x = self.net1(x)
        # 使用 ReLU 激活函数
        x = self.relu(x)
        # 第二个全连接层
        x = self.net2(x)
        # 返回结果张量
        return x


# 定义一个多层感知机模型的集合
class MultiMLP(torch.nn.Module):
    # 初始化函数，用于初始化神经网络模型对象
    def __init__(self, d_hid: int, n_layers: int = 2):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个由多个 MLPModule 组成的列表，每个 MLPModule 都具有隐藏层维度 d_hid
        self.layers = torch.nn.ModuleList([MLPModule(d_hid) for _ in range(n_layers)])
        # 用于测试目的的分割规范字典，用户应定义其内容
        self.split_spec = {
            f"layers.{i}": SplitPoint.BEGINNING for i in range(1, n_layers)
        }

    # 前向传播函数，执行模型的前向计算
    def forward(self, x):
        # 遍历每一层 MLPModule，并依次对输入 x 进行计算
        for layer in self.layers:
            x = layer(x)
        # 返回最终的计算结果 x
        return x
```