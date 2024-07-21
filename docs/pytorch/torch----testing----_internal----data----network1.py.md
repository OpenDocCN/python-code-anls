# `.\pytorch\torch\testing\_internal\data\network1.py`

```py
# 忽略 mypy 的错误，因为在某些情况下，可能会发生类型检查的错误
import torch.nn as nn

# 定义一个名为 Net 的类，继承自 nn.Module 类
class Net(nn.Module):

    # 构造函数，用于初始化对象
    def __init__(self):
        # 调用父类（nn.Module）的构造函数
        super().__init__()
        # 定义一个线性层，输入维度为 10，输出维度为 20
        self.linear = nn.Linear(10, 20)
```