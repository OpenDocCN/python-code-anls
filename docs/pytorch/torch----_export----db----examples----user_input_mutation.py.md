# `.\pytorch\torch\_export\db\examples\user_input_mutation.py`

```py
# 引入 torch 模块，用于神经网络和张量操作
# mypy: allow-untyped-defs 表示允许未标记类型的定义
import torch

# 定义一个继承自 torch.nn.Module 的类 UserInputMutation，用于用户输入的变异操作
class UserInputMutation(torch.nn.Module):
    """
    直接在 forward 方法中对用户输入进行变异
    """

    def forward(self, x):
        # 将输入张量 x 中的每个元素乘以 2，实现就地修改（in-place）
        x.mul_(2)
        # 返回 x 中每个元素的余弦值
        return x.cos()

# 创建一个示例输入 example_inputs，包含一个形状为 (3, 2) 的随机张量
example_inputs = (torch.randn(3, 2),)

# 创建一个标签 tags，包含字符串 "torch.mutation"
tags = {"torch.mutation"}

# 创建一个 UserInputMutation 类的实例 model，用于后续的变异操作
model = UserInputMutation()
```