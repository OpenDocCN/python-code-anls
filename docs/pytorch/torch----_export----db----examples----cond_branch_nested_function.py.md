# `.\pytorch\torch\_export\db\examples\cond_branch_nested_function.py`

```py
# mypy: allow-untyped-defs
import torch  # 导入PyTorch库

from functorch.experimental.control_flow import cond  # 从functorch库中导入cond函数

class CondBranchNestedFunction(torch.nn.Module):
    """
    The branch functions (`true_fn` and `false_fn`) passed to cond() must follow these rules:
      - both branches must take the same args, which must also match the branch args passed to cond.
      - both branches must return a single tensor
      - returned tensor must have the same tensor metadata, e.g. shape and dtype
      - branch function can be free function, nested function, lambda, class methods
      - branch function can not have closure variables
      - no inplace mutations on inputs or global variables

    This example demonstrates using nested function in cond().

    NOTE: If the `pred` is test on a dim with batch size < 2, it will be specialized.
    """

    def forward(self, x):
        def true_fn(x):
            def inner_true_fn(y):
                return x + y  # 返回 x 加上 y 的结果

            return inner_true_fn(x)

        def false_fn(x):
            def inner_false_fn(y):
                return x - y  # 返回 x 减去 y 的结果

            return inner_false_fn(x)

        return cond(x.shape[0] < 10, true_fn, false_fn, [x])  # 使用 cond 函数根据条件选择 true_fn 或 false_fn

example_inputs = (torch.randn(3),)  # 示例输入数据，包含一个形状为 (3,) 的随机张量
tags = {
    "torch.cond",  # 标记：torch.cond
    "torch.dynamic-shape",  # 标记：torch.dynamic-shape
}
model = CondBranchNestedFunction()  # 创建 CondBranchNestedFunction 类的实例对象
```