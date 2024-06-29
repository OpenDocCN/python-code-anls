# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\nested_sequence.pyi`

```py
# 导入Sequence抽象基类，用于声明序列类型变量
from collections.abc import Sequence
# 导入_NestedSequence类型变量，这是一个内部类型，用于声明嵌套序列
from numpy._typing import _NestedSequence

# 声明变量a为Sequence类型，元素为float类型的序列
a: Sequence[float]
# 声明变量b为列表类型，元素为complex复数类型的列表
b: list[complex]
# 声明变量c为元组类型，包含不可变的str类型元素
c: tuple[str, ...]
# 声明变量d为整数类型
d: int
# 声明变量e为字符串类型
e: str

# 定义函数func，接受_NestedSequence类型的整数序列参数，返回空值
def func(a: _NestedSequence[int]) -> None:
    ...

# 使用reveal_type函数分别输出func函数对各变量类型的推断，预期会显示类型不兼容的错误信息
reveal_type(func(a))  # E: incompatible type
reveal_type(func(b))  # E: incompatible type
reveal_type(func(c))  # E: incompatible type
reveal_type(func(d))  # E: incompatible type
reveal_type(func(e))  # E: incompatible type
```