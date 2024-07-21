# `.\pytorch\torch\_export\db\examples\cond_predicate.py`

```py
# 导入 torch 库
import torch

# 从 functorch.experimental.control_flow 中导入 cond 函数
from functorch.experimental.control_flow import cond

# 定义一个继承自 torch.nn.Module 的类 CondPredicate
class CondPredicate(torch.nn.Module):
    """
    The conditional statement (aka predicate) passed to cond() must be one of the following:
      - torch.Tensor with a single element
      - boolean expression

    NOTE: If the `pred` is test on a dim with batch size < 2, it will be specialized.
    """

    # 定义前向传播方法
    def forward(self, x):
        # 创建一个条件判断 pred，检查 x 的维度是否大于 2，且第三个维度的大小是否大于 10
        pred = x.dim() > 2 and x.shape[2] > 10

        # 调用 cond 函数，根据 pred 的结果选择执行 x.cos() 或者 y.sin()
        return cond(pred, lambda x: x.cos(), lambda y: y.sin(), [x])

# 定义一个示例输入 example_inputs，包含一个 torch.randn 生成的张量
example_inputs = (torch.randn(6, 4, 3),)

# 定义一个标签集合 tags，包含 "torch.cond" 和 "torch.dynamic-shape"
tags = {
    "torch.cond",
    "torch.dynamic-shape",
}

# 创建一个 CondPredicate 类的实例 model
model = CondPredicate()
```