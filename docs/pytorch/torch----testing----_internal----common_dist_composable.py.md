# `.\pytorch\torch\testing\_internal\common_dist_composable.py`

```
# mypy: ignore-errors

# Owner(s): ["oncall: distributed"]

# 导入需要的模块和类
from typing import Tuple  # 导入类型标注模块

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块


class UnitModule(nn.Module):
    # 单元模块类，继承自nn.Module
    def __init__(self, device: torch.device):
        super().__init__()
        # 定义线性层l1，输入和输出维度均为100，使用指定的设备
        self.l1 = nn.Linear(100, 100, device=device)
        # 定义顺序模块seq，包含ReLU激活函数和线性层，使用指定的设备
        self.seq = nn.Sequential(
            nn.ReLU(),
            nn.Linear(100, 100, device=device),
            nn.ReLU(),
        )
        # 定义线性层l2，输入和输出维度均为100，使用指定的设备
        self.l2 = nn.Linear(100, 100, device=device)

    def forward(self, x):
        # 前向传播函数，先通过l1、seq、l2的顺序处理输入x，返回处理后的结果
        return self.l2(self.seq(self.l1(x)))


class CompositeModel(nn.Module):
    # 复合模型类，继承自nn.Module
    def __init__(self, device: torch.device):
        super().__init__()
        # 定义线性层l1，输入和输出维度均为100，使用指定的设备
        self.l1 = nn.Linear(100, 100, device=device)
        # 定义两个单元模块u1和u2，均使用指定的设备
        self.u1 = UnitModule(device)
        self.u2 = UnitModule(device)
        # 定义线性层l2，输入和输出维度均为100，使用指定的设备
        self.l2 = nn.Linear(100, 100, device=device)

    def forward(self, x):
        # 前向传播函数，通过u1、u2、l1、l2的顺序处理输入x，返回处理后的结果
        return self.l2(self.u2(self.u1(self.l1(x))))


class UnitParamModule(nn.Module):
    # 带参数的单元模块类，继承自nn.Module
    def __init__(self, device: torch.device):
        super().__init__()
        # 定义线性层l，输入和输出维度均为100，使用指定的设备
        self.l = nn.Linear(100, 100, device=device)
        # 定义顺序模块seq，包含ReLU激活函数和线性层，使用指定的设备
        self.seq = nn.Sequential(
            nn.ReLU(),
            nn.Linear(100, 100, device=device),
            nn.ReLU(),
        )
        # 定义参数p，初始化为100x100的随机张量，使用指定的设备
        self.p = nn.Parameter(torch.randn((100, 100), device=device))

    def forward(self, x):
        # 前向传播函数，先通过l和seq的顺序处理输入x，然后与参数p进行矩阵乘法，返回处理后的结果
        return torch.mm(self.seq(self.l(x)), self.p)


class CompositeParamModel(nn.Module):
    # 带参数的复合模型类，继承自nn.Module
    def __init__(self, device: torch.device):
        super().__init__()
        # 定义线性层l，输入和输出维度均为100，使用指定的设备
        self.l = nn.Linear(100, 100, device=device)
        # 定义两个单元模块u1和u2，均使用指定的设备
        self.u1 = UnitModule(device)
        self.u2 = UnitModule(device)
        # 定义参数p，初始化为100x100的随机张量，使用指定的设备
        self.p = nn.Parameter(torch.randn((100, 100), device=device))
        # 注册一个持久化的缓冲区buffer，初始化为100x100的随机张量，使用指定的设备
        self.register_buffer(
            "buffer", torch.randn((100, 100), device=device), persistent=True
        )

    def forward(self, x):
        # 前向传播函数，先通过u1、u2、l的顺序处理输入x，然后与参数p进行矩阵乘法，返回处理后的结果
        a = self.u2(self.u1(self.l(x)))
        b = self.p
        return torch.mm(a, b)


class FakeSequential(nn.Module):
    # 伪顺序模块类，继承自nn.Module
    # 用于实现希望的嵌套包装效果，使用模块包装策略和`nn.Sequential`
    def __init__(self, *modules: Tuple[nn.Module, ...]) -> None:
        super().__init__()
        # 将输入的模块序列保存在_module_sequence属性中
        self._module_sequence = list(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向传播函数，依次对输入x应用_module_sequence中的每个模块，返回处理后的结果
        for module in self._module_sequence:
            x = module(x)
        return x


class NestedSequentialModel(nn.Module):
    # 嵌套顺序模型类，继承自nn.Module
    def __init__(self, device: torch.device) -> None:
        # 初始化函数，用于创建对象时初始化其属性
        super().__init__()
        # 创建一个包含线性层和伪序列的序列模块
        # 这个嵌套结构通过遍历顺序来验证不同的遍历方式（例如 BFS 和 DFS 变体）。
        self.seq1 = nn.Sequential(
            nn.Linear(1, 1, device=device),  # 添加一个线性层，输入维度为1，输出维度为1
            FakeSequential(  # 使用伪序列包装下面的模块
                nn.Linear(1, 1, device=device),  # 再次添加一个线性层，输入维度为1，输出维度为1
                nn.ReLU(),  # 添加一个ReLU激活函数层
                FakeSequential(  # 使用伪序列包装下面的模块
                    nn.Linear(1, 1, device=device),  # 添加一个线性层，输入维度为1，输出维度为1
                ),
                nn.ReLU(),  # 添加一个ReLU激活函数层
            ),
            nn.Linear(1, 2, device=device),  # 添加一个线性层，输入维度为1，输出维度为2
        )
        self.lin = nn.Linear(2, 2, device=device)  # 添加一个线性层，输入维度为2，输出维度为2
        self.seq2 = nn.Sequential(
            nn.ReLU(),  # 添加一个ReLU激活函数层
            nn.Linear(2, 3, device=device),  # 添加一个线性层，输入维度为2，输出维度为3
            FakeSequential(  # 使用伪序列包装下面的模块
                nn.Linear(3, 2, bias=False, device=device),  # 添加一个线性层，输入维度为3，输出维度为2，没有偏置
                nn.Linear(2, 4, bias=False, device=device),  # 添加一个线性层，输入维度为2，输出维度为4，没有偏置
            ),
        )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # 前向传播函数，定义模型的计算流程
            return self.seq2(self.lin(self.seq1(x)))
```