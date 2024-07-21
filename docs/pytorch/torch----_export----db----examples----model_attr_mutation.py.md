# `.\pytorch\torch\_export\db\examples\model_attr_mutation.py`

```py
# mypy: allow-untyped-defs
# 引入 torch 库
import torch

# 从 torch._export.db.case 中导入 SupportLevel 类
from torch._export.db.case import SupportLevel

# 定义一个继承自 torch.nn.Module 的类 ModelAttrMutation
class ModelAttrMutation(torch.nn.Module):
    """
    Attribute mutation is not supported.
    """

    # 初始化方法，调用父类的初始化方法，创建一个包含两个随机张量的列表
    def __init__(self):
        super().__init__()
        self.attr_list = [torch.randn(3, 2), torch.randn(3, 2)]

    # 重新创建列表的方法，返回包含两个全零张量的列表
    def recreate_list(self):
        return [torch.zeros(3, 2), torch.zeros(3, 2)]

    # 前向传播方法，接收输入 x，重新设置 attr_list 为全零张量列表，并返回 x 的和加上 attr_list 第一个张量的和
    def forward(self, x):
        self.attr_list = self.recreate_list()
        return x.sum() + self.attr_list[0].sum()


# 创建一个例子输入元组，包含一个形状为 (3, 2) 的随机张量
example_inputs = (torch.randn(3, 2),)

# 定义一个标签字典，包含 "python.object-model" 标签
tags = {"python.object-model"}

# 定义支持级别为 SupportLevel.NOT_SUPPORTED_YET
support_level = SupportLevel.NOT_SUPPORTED_YET

# 创建 ModelAttrMutation 类的实例 model
model = ModelAttrMutation()
```