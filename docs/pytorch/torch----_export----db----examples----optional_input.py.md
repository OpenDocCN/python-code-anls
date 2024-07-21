# `.\pytorch\torch\_export\db\examples\optional_input.py`

```py
# 引入 torch 库
import torch

# 从 torch._export.db.case 模块导入 SupportLevel 类
from torch._export.db.case import SupportLevel

# 定义一个继承自 torch.nn.Module 的类 OptionalInput
class OptionalInput(torch.nn.Module):
    """
    Tracing through optional input is not supported yet
    """

    # 定义前向传播方法
    def forward(self, x, y=torch.randn(2, 3)):
        # 如果 y 不为 None，则返回 x + y 的结果
        if y is not None:
            return x + y
        # 如果 y 为 None，则返回 x 自身
        return x

# 示例输入，包含一个大小为 (2, 3) 的随机张量
example_inputs = (torch.randn(2, 3),)

# 标签定义为 {"python.object-model"}
tags = {"python.object-model"}

# 支持级别定义为 SupportLevel.NOT_SUPPORTED_YET
support_level = SupportLevel.NOT_SUPPORTED_YET

# 创建 OptionalInput 类的实例，赋值给 model 变量
model = OptionalInput()
```