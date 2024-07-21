# `.\pytorch\torch\testing\_internal\quantization_torch_package_models.py`

```py
# 忽略类型检查错误，用于类型检查工具 mypy
# 导入 math 库，用于数学计算
import math

# 导入 PyTorch 库
import torch
import torch.nn as nn

# 定义一个继承自 nn.Module 的类 LinearReluFunctionalChild
class LinearReluFunctionalChild(nn.Module):
    def __init__(self, N):
        super().__init__()
        # 定义一个可学习的参数 w1，是一个 N x N 的矩阵
        self.w1 = nn.Parameter(torch.empty(N, N))
        # 定义一个可学习的参数 b1，是一个大小为 N 的零向量
        self.b1 = nn.Parameter(torch.zeros(N))
        # 使用 Kaiming 均匀初始化方法初始化参数 w1
        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

    # 定义前向传播函数
    def forward(self, x):
        # 对输入 x 执行线性变换，参数为 self.w1 和 self.b1
        x = torch.nn.functional.linear(x, self.w1, self.b1)
        # 对线性变换的结果执行 ReLU 激活函数
        x = torch.nn.functional.relu(x)
        # 返回激活后的结果
        return x

# 定义一个继承自 nn.Module 的类 LinearReluFunctional
class LinearReluFunctional(nn.Module):
    def __init__(self, N):
        super().__init__()
        # 创建 LinearReluFunctionalChild 类的一个实例作为子模块 child
        self.child = LinearReluFunctionalChild(N)
        # 定义一个可学习的参数 w1，是一个 N x N 的矩阵
        self.w1 = nn.Parameter(torch.empty(N, N))
        # 定义一个可学习的参数 b1，是一个大小为 N 的零向量
        self.b1 = nn.Parameter(torch.zeros(N))
        # 使用 Kaiming 均匀初始化方法初始化参数 w1
        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

    # 定义前向传播函数
    def forward(self, x):
        # 调用 child 模块的前向传播方法，对输入 x 进行处理
        x = self.child(x)
        # 对处理后的结果执行线性变换，参数为 self.w1 和 self.b1
        x = torch.nn.functional.linear(x, self.w1, self.b1)
        # 对线性变换的结果执行 ReLU 激活函数
        x = torch.nn.functional.relu(x)
        # 返回激活后的结果
        return x
```