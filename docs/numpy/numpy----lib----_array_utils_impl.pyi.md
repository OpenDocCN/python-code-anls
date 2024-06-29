# `D:\src\scipysrc\numpy\numpy\lib\_array_utils_impl.pyi`

```py
# 导入必要的类型提示
from typing import Any, Iterable, Tuple

# 导入泛型和 NDArray 类型
from numpy import generic
from numpy.typing import NDArray

# 定义 __all__ 变量，用于声明模块中公开的符号列表
__all__: list[str]

# NOTE: 实际使用中，`byte_bounds` 可以接受实现了 `__array_interface__` 协议的任意对象。
# 但是需要注意的是，根据规范中标记为可选的某些键必须在 `byte_bounds` 中存在，包括 `"strides"` 和 `"data"`。

# 定义 `byte_bounds` 函数，接受泛型或任意 NDArray 对象 `a`，返回两个整数元组
def byte_bounds(a: generic | NDArray[Any]) -> tuple[int, int]: ...

# 定义 `normalize_axis_tuple` 函数，用于规范化轴元组参数
def normalize_axis_tuple(
    axis: int | Iterable[int],  # 轴参数可以是整数或整数迭代器
    ndim: int = ...,             # 数组的维度
    argname: None | str = ...,   # 参数名称（可选）
    allow_duplicate: None | bool = ...,  # 是否允许重复（可选）
) -> Tuple[int, int]: ...

# 定义 `normalize_axis_index` 函数，用于规范化轴索引参数
def normalize_axis_index(
    axis: int = ...,             # 轴索引
    ndim: int = ...,             # 数组的维度
    msg_prefix: None | str = ...,  # 错误消息前缀（可选）
) -> int: ...
```