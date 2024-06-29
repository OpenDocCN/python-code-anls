# `D:\src\scipysrc\numpy\numpy\lib\_arrayterator_impl.pyi`

```
# 导入必要的模块和类型
from collections.abc import Generator
from typing import (
    Any,
    TypeVar,
    overload,
)

from numpy import ndarray, dtype, generic
from numpy._typing import DTypeLike, NDArray

# TODO: Set a shape bound once we've got proper shape support
# 定义类型变量，_Shape 用于数组形状，_DType 用于数组元素类型，_ScalarType 用于标量类型
_Shape = TypeVar("_Shape", bound=Any)
_DType = TypeVar("_DType", bound=dtype[Any])
_ScalarType = TypeVar("_ScalarType", bound=generic)

# 定义索引类型，可以是ellipsis（省略号）、整数、切片或元组，元组中包含ellipsis、整数或切片的组合
_Index = (
    ellipsis
    | int
    | slice
    | tuple[ellipsis | int | slice, ...]
)

# 定义导出列表，此处是空列表，准备用于模块导出
__all__: list[str]

# NOTE: In reality `Arrayterator` does not actually inherit from `ndarray`,
# but its ``__getattr__` method does wrap around the former and thus has
# access to all its methods
# 创建一个自定义类 Arrayterator，它伪装成继承自 ndarray，实际上不是真正的继承关系
class Arrayterator(ndarray[_Shape, _DType]):
    var: ndarray[_Shape, _DType]  # 类型注释：var 是一个 ndarray，用于存储数组数据
    buf_size: None | int  # 缓冲区大小，可以为 None 或整数
    start: list[int]  # 列表，存储起始索引的整数值
    stop: list[int]  # 列表，存储终止索引的整数值
    step: list[int]  # 列表，存储步长的整数值

    @property  # 属性方法装饰器：shape 属性，返回元组形式的数组形状
    def shape(self) -> tuple[int, ...]: ...

    @property
    def flat(self: NDArray[_ScalarType]) -> Generator[_ScalarType, None, None]: ...
    # 方法初始化函数，接受一个 ndarray 和一个可选的 buf_size 参数
    def __init__(
        self, var: ndarray[_Shape, _DType], buf_size: None | int = ...
    ) -> None: ...

    @overload  # 方法重载：__array__ 方法，返回特定类型的数组视图
    def __array__(self, dtype: None = ..., copy: None | bool = ...) -> ndarray[Any, _DType]: ...

    @overload
    def __array__(self, dtype: DTypeLike, copy: None | bool = ...) -> NDArray[Any]: ...

    # 索引操作符重载：获取指定索引的 Arrayterator 对象
    def __getitem__(self, index: _Index) -> Arrayterator[Any, _DType]: ...

    # 迭代器方法：返回一个生成器，生成 ndarray 对象
    def __iter__(self) -> Generator[ndarray[Any, _DType], None, None]: ...
```