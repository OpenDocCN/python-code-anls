# `.\pytorch\torch\_export\db\examples\cond_branch_nonlocal_variables.py`

```py
# 使用类型检查时允许未类型化的函数定义
mypy: allow-untyped-defs

# 导入 PyTorch 库
import torch

# 从 functorch.experimental.control_flow 模块导入 cond 函数
from functorch.experimental.control_flow import cond

# 定义一个继承自 torch.nn.Module 的类 CondBranchNonlocalVariables
class CondBranchNonlocalVariables(torch.nn.Module):
    """
    The branch functions (`true_fn` and `false_fn`) passed to cond() must follow these rules:
    - both branches must take the same args, which must also match the branch args passed to cond.
    - both branches must return a single tensor
    - returned tensor must have the same tensor metadata, e.g. shape and dtype
    - branch function can be free function, nested function, lambda, class methods
    - branch function can not have closure variables
    - no inplace mutations on inputs or global variables

    This example demonstrates how to rewrite code to avoid capturing closure variables in branch functions.

    The code below will not work because capturing closure variables is not supported.
    ```
    my_tensor_var = x + 100
    my_primitive_var = 3.14

    def true_fn(y):
        nonlocal my_tensor_var, my_primitive_var
        return y + my_tensor_var + my_primitive_var

    def false_fn(y):
        nonlocal my_tensor_var, my_primitive_var
        return y - my_tensor_var - my_primitive_var

    return cond(x.shape[0] > 5, true_fn, false_fn, [x])
    ```py

    NOTE: If the `pred` is test on a dim with batch size < 2, it will be specialized.
    """

    # 定义前向传播函数
    def forward(self, x):
        # 定义一个局部变量 my_tensor_var，其值为 x + 100
        my_tensor_var = x + 100
        # 定义一个局部变量 my_primitive_var，其值为 3.14
        my_primitive_var = 3.14

        # 定义一个 true_fn 函数，接受参数 x, y, z，并返回它们的和
        def true_fn(x, y, z):
            return x + y + z

        # 定义一个 false_fn 函数，接受参数 x, y, z，并返回它们的差
        def false_fn(x, y, z):
            return x - y - z

        # 调用 cond 函数，根据条件 x.shape[0] > 5 选择 true_fn 或 false_fn
        # 并传入参数列表 [x, my_tensor_var, torch.tensor(my_primitive_var)]
        return cond(
            x.shape[0] > 5,
            true_fn,
            false_fn,
            [x, my_tensor_var, torch.tensor(my_primitive_var)],
        )

# 定义一个示例输入 example_inputs，包含一个形状为 (6,) 的随机张量
example_inputs = (torch.randn(6),)

# 定义一个标签集合 tags，包含 "torch.cond" 和 "torch.dynamic-shape"
tags = {
    "torch.cond",
    "torch.dynamic-shape",
}

# 创建一个 CondBranchNonlocalVariables 类的实例 model
model = CondBranchNonlocalVariables()
```