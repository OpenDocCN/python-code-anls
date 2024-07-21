# `.\pytorch\torch\_export\db\examples\null_context_manager.py`

```
# 引入名为 `mypy` 的注释，允许未类型化的定义
# 引入 `contextlib` 模块，用于创建上下文管理器
import contextlib

# 引入 `torch` 模块，用于神经网络相关操作
import torch

# 定义一个名为 `NullContextManager` 的类，继承自 `torch.nn.Module`
class NullContextManager(torch.nn.Module):
    """
    Null context manager in Python will be traced out.
    """

    def forward(self, x):
        """
        Null context manager in Python will be traced out.
        """
        # 创建一个空的上下文管理器对象 `ctx`
        ctx = contextlib.nullcontext()
        # 使用 `ctx` 上下文管理器进行操作
        with ctx:
            # 返回输入张量 `x` 的正弦值加余弦值
            return x.sin() + x.cos()

# 定义一个示例输入 `example_inputs`，包含一个形状为 (3, 2) 的随机张量
example_inputs = (torch.randn(3, 2),)

# 定义一个标签 `tags`，包含字符串 "python.context-manager"
tags = {"python.context-manager"}

# 创建一个 `NullContextManager` 类的实例 `model`
model = NullContextManager()
```