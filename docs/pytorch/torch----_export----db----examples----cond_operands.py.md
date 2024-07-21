# `.\pytorch\torch\_export\db\examples\cond_operands.py`

```
# 引入torch模块
import torch

# 从torch中导入Dim类
from torch.export import Dim

# 从functorch.experimental.control_flow中导入cond函数
from functorch.experimental.control_flow import cond

# 创建一个大小为(3, 2)的随机张量x
x = torch.randn(3, 2)

# 创建一个大小为(2,)的随机张量y
y = torch.randn(2)

# 创建一个维度对象dim0_x，用于后续条件操作
dim0_x = Dim("dim0_x")

# 定义一个继承自torch.nn.Module的类CondOperands，用于条件操作
class CondOperands(torch.nn.Module):
    """
    The operands passed to cond() must be:
    - a list of tensors
    - match arguments of `true_fn` and `false_fn`

    NOTE: If the `pred` is test on a dim with batch size < 2, it will be specialized.
    """

    def forward(self, x, y):
        # 定义条件为x的行数是否大于2的函数true_fn，返回x + y
        def true_fn(x, y):
            return x + y

        # 定义条件为x的行数是否大于2的函数false_fn，返回x - y
        def false_fn(x, y):
            return x - y

        # 调用cond函数，根据条件x.shape[0] > 2选择true_fn或false_fn进行计算，输入为[x, y]
        return cond(x.shape[0] > 2, true_fn, false_fn, [x, y])

# 定义一个示例输入example_inputs为(x, y)
example_inputs = (x, y)

# 定义一个标签集合tags
tags = {
    "torch.cond",            # 条件操作的标签
    "torch.dynamic-shape",   # 动态形状的标签
}

# 定义额外输入extra_inputs为(torch.randn(2, 2), torch.randn(2))
extra_inputs = (torch.randn(2, 2), torch.randn(2))

# 定义动态形状的字典dynamic_shapes，指定"x"的第0维度为dim0_x，"y"为None表示没有特定的形状要求
dynamic_shapes = {"x": {0: dim0_x}, "y": None}

# 创建一个CondOperands类的实例model，用于后续的条件操作
model = CondOperands()
```