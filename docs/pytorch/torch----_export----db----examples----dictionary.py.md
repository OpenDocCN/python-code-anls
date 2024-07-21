# `.\pytorch\torch\_export\db\examples\dictionary.py`

```py
# 引入 torch 库
import torch

# 定义一个继承自 torch.nn.Module 的类 Dictionary，表示一个字典结构在追踪中被内联和展平
class Dictionary(torch.nn.Module):
    """
    Dictionary structures are inlined and flattened along tracing.
    """

    # 定义类的前向传播方法
    def forward(self, x, y):
        # 初始化一个空字典 elements
        elements = {}
        # 将 x 的平方存入字典中的键 "x2"
        elements["x2"] = x * x
        # 更新 y，使其乘以 elements 字典中的 "x2" 对应的值
        y = y * elements["x2"]
        # 返回一个包含 "y" 键的字典，其值为 y
        return {"y": y}

# 创建一个示例输入 example_inputs，包括一个形状为 (3, 2) 的随机张量和一个标量值为 4 的张量
example_inputs = (torch.randn(3, 2), torch.tensor(4))

# 创建一个标签字典 tags，包含 "python.data-structure" 键
tags = {"python.data-structure"}

# 创建一个 Dictionary 类的实例 model
model = Dictionary()
```