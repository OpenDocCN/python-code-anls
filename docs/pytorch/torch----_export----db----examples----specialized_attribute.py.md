# `.\pytorch\torch\_export\db\examples\specialized_attribute.py`

```
# 引入枚举类型 `Enum` 用于定义常量
from enum import Enum

# 引入 PyTorch 深度学习框架
import torch

# 定义动物类，继承自 `Enum` 类，包含一个属性 `COW`，对应值为字符串 "moo"
class Animal(Enum):
    COW = "moo"

# 定义一个继承自 `torch.nn.Module` 的特殊属性类 `SpecializedAttribute`
class SpecializedAttribute(torch.nn.Module):
    """
    Model attributes are specialized.
    """

    # 构造函数，初始化模型属性
    def __init__(self):
        super().__init__()
        # 初始化属性 `a` 为字符串 "moo"
        self.a = "moo"
        # 初始化属性 `b` 为整数 4
        self.b = 4

    # 前向传播方法，接收输入 `x`
    def forward(self, x):
        # 如果属性 `a` 的值等于 `Animal.COW` 的值
        if self.a == Animal.COW.value:
            # 返回 `x` 的平方加上属性 `b` 的值
            return x * x + self.b
        else:
            # 否则抛出数值错误异常
            raise ValueError("bad")

# 示例输入数据，一个包含形状为 (3, 2) 的随机张量的元组
example_inputs = (torch.randn(3, 2),)

# 创建 `SpecializedAttribute` 类的实例 `model`
model = SpecializedAttribute()
```