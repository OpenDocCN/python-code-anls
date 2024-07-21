# `.\pytorch\torch\testing\_internal\data\network2.py`

```
# 忽略类型检查错误，针对 mypy 工具
# 导入 PyTorch 的神经网络模块 nn
import torch.nn as nn

# 定义一个名为 Net 的类，继承自 nn.Module 类
class Net(nn.Module):

    # 定义初始化方法
    def __init__(self):
        # 调用父类 nn.Module 的初始化方法
        super().__init__()
        # 定义一个线性层，输入维度为 10，输出维度为 20
        self.linear = nn.Linear(10, 20)
        # 定义一个 ReLU 激活函数层
        self.relu = nn.ReLU()
```