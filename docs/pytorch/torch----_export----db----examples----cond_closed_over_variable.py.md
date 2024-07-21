# `.\pytorch\torch\_export\db\examples\cond_closed_over_variable.py`

```
# 导入 torch 库，这是 PyTorch 深度学习库的入口
import torch

# 从 functorch.experimental.control_flow 中导入 cond 函数
from functorch.experimental.control_flow import cond

# 定义一个继承自 torch.nn.Module 的类 CondClosedOverVariable
class CondClosedOverVariable(torch.nn.Module):
    """
    torch.cond() supports branches closed over arbitrary variables.
    """

    # 定义模型的前向传播函数
    def forward(self, pred, x):
        # 定义条件为真时执行的函数
        def true_fn(val):
            return x * 2

        # 定义条件为假时执行的函数
        def false_fn(val):
            return x - 2

        # 使用 cond 函数根据 pred 的值选择执行 true_fn 或 false_fn，并传入参数 [x + 1]
        return cond(pred, true_fn, false_fn, [x + 1])

# 定义一个示例输入
example_inputs = (torch.tensor(True), torch.randn(3, 2))

# 定义标签集合，包括 "torch.cond" 和 "python.closure"
tags = {"torch.cond", "python.closure"}

# 创建 CondClosedOverVariable 类的实例，即模型
model = CondClosedOverVariable()
```