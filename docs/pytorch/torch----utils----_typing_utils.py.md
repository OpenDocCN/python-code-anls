# `.\pytorch\torch\utils\_typing_utils.py`

```py
"""Miscellaneous utilities to aid with typing."""

# 导入必要的模块
from typing import Optional, TypeVar

# 定义一个类型变量 T，用于泛型类型提示
T = TypeVar("T")

# 定义函数 not_none，接受一个可选类型参数 obj，并返回其非空值
def not_none(obj: Optional[T]) -> T:
    # 如果 obj 是 None，则抛出类型错误异常，表明出现了预期外的 None 值
    if obj is None:
        raise TypeError("Invariant encountered: value was None when it should not be")
    # 返回 obj 的值，确保它不为空
    return obj
```