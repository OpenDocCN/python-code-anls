# `.\pytorch\torch\masked\maskedtensor\creation.py`

```py
# 指定类型检查选项，允许未标记类型的定义
# Copyright (c) Meta Platforms, Inc. and affiliates

# 从当前目录导入核心模块中的MaskedTensor类
from .core import MaskedTensor

# 导出模块中公开的函数和类列表
__all__ = [
    "as_masked_tensor",  # 将函数as_masked_tensor添加到导出列表中
    "masked_tensor",     # 将函数masked_tensor添加到导出列表中
]

# 这两个工厂函数的目的是模仿以下torch库中的函数：
#     torch.tensor - 保证是一个叶节点（leaf node）
#     torch.as_tensor - 可微的构造函数，保留自动求导历史记录

# 创建一个MaskedTensor对象，用给定的data和mask初始化，并根据需要启用梯度计算
def masked_tensor(data, mask, requires_grad=False):
    return MaskedTensor(data, mask, requires_grad)

# 通过调用MaskedTensor类的_from_values静态方法，创建一个MaskedTensor对象，
# 使用给定的data和mask作为参数传入
def as_masked_tensor(data, mask):
    return MaskedTensor._from_values(data, mask)
```