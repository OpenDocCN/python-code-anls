# `.\numpy\numpy\_typing\_shape.py`

```py
# 导入 Sequence 抽象基类，用于定义序列类型的抽象接口
from collections.abc import Sequence
# 导入 Union 和 SupportsIndex 用于类型注解
from typing import Union, SupportsIndex

# 定义 _Shape 类型别名，表示一个由整数组成的元组
_Shape = tuple[int, ...]

# 定义 _ShapeLike 类型别名，表示可以被转换成 _Shape 的对象类型，可以是单个 SupportsIndex 或者 SupportsIndex 组成的序列
# SupportsIndex 表示支持索引操作的类型
_ShapeLike = Union[SupportsIndex, Sequence[SupportsIndex]]
```