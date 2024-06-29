# `D:\src\scipysrc\pandas\pandas\_libs\properties.pyi`

```
from typing import (
    Sequence,
    overload,
)

from pandas._typing import (
    AnyArrayLike,
    DataFrame,
    Index,
    Series,
)

# note: this is a lie to make type checkers happy (they special
# case property). cache_readonly uses attribute names similar to
# property (fget) but it does not provide fset and fdel.
# 将 cache_readonly 定义为一个 property 对象，用于类型检查，虽然它与 property 类似，
# 但不提供 fset 和 fdel 方法。
cache_readonly = property

class AxisProperty:
    axis: int  # 类属性 axis 表示轴的整数值
    def __init__(self, axis: int = ..., doc: str = ...) -> None:
        # 初始化方法，接受一个整数 axis 和一个文档字符串 doc
        ...

    @overload
    def __get__(self, obj: DataFrame | Series, type) -> Index:
        # 这是一个重载方法，根据 obj 参数的类型返回 Index 对象
        ...

    @overload
    def __get__(self, obj: None, type) -> AxisProperty:
        # 这是一个重载方法，当 obj 参数为 None 时返回 AxisProperty 对象自身
        ...

    def __set__(
        self, obj: DataFrame | Series, value: AnyArrayLike | Sequence
    ) -> None:
        # 设置方法，用于设置 obj 对象的属性为 value 的值，类型可以是 AnyArrayLike 或 Sequence
        ...
```