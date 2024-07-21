# `.\pytorch\torch\_export\db\examples\fn_with_kwargs.py`

```
# 引入torch模块，这是PyTorch深度学习框架的核心模块
import torch

# 从torch._export.db.case模块中引入ExportArgs类
from torch._export.db.case import ExportArgs

# 定义一个继承自torch.nn.Module的类FnWithKwargs
class FnWithKwargs(torch.nn.Module):
    """
    目前不支持关键字参数。
    """

    # 实现torch.nn.Module的forward方法
    def forward(self, pos0, tuple0, *myargs, mykw0, **mykwargs):
        # 初始化输出为pos0
        out = pos0
        # 遍历tuple0中的元素，与out相乘更新out
        for arg in tuple0:
            out = out * arg
        # 遍历myargs中的元素，与out相乘更新out
        for arg in myargs:
            out = out * arg
        # 将out与mykw0相乘更新out
        out = out * mykw0
        # 将out与mykwargs中键为"input0"和"input1"的值相乘更新out
        out = out * mykwargs["input0"] * mykwargs["input1"]
        # 返回更新后的out作为输出
        return out

# 创建一个ExportArgs对象example_inputs，包含了多个输入参数
example_inputs = ExportArgs(
    torch.randn(4),  # 第一个参数是一个4维的随机张量
    (torch.randn(4), torch.randn(4)),  # 第二个参数是一个包含两个4维随机张量的元组
    *[torch.randn(4), torch.randn(4)],  # 剩余的参数是两个4维的随机张量（通过*展开）
    mykw0=torch.randn(4),  # 关键字参数mykw0是一个4维的随机张量
    input0=torch.randn(4), input1=torch.randn(4)  # 关键字参数input0和input1分别是两个4维的随机张量
)

# 创建一个包含字符串"python.data-structure"的标签字典tags
tags = {"python.data-structure"}

# 创建一个FnWithKwargs类的实例model
model = FnWithKwargs()
```