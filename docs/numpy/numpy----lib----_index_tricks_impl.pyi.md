# `D:\src\scipysrc\numpy\numpy\lib\_index_tricks_impl.pyi`

```
# 导入所需的模块和类型定义
from collections.abc import Sequence
from typing import (
    Any,
    TypeVar,
    Generic,
    overload,
    Literal,
    SupportsIndex,
)

# 导入 NumPy 库及其子模块和类型定义
import numpy as np
from numpy import (
    # 通过重命名避免与 `AxisConcatenator.matrix` 的命名冲突
    matrix as _Matrix,
    ndenumerate as ndenumerate,
    ndindex as ndindex,
    ndarray,
    dtype,
    str_,
    bytes_,
    int_,
    float64,
    complex128,
)
from numpy._typing import (
    # 数组相关类型
    ArrayLike,
    _NestedSequence,
    _FiniteNestedSequence,
    NDArray,

    # 数据类型相关类型
    DTypeLike,
    _SupportsDType,
)

# 导入 NumPy 核心函数
from numpy._core.multiarray import (
    unravel_index as unravel_index,
    ravel_multi_index as ravel_multi_index,
)

# 定义类型变量
_T = TypeVar("_T")
_DType = TypeVar("_DType", bound=dtype[Any])
_BoolType = TypeVar("_BoolType", Literal[True], Literal[False])
_TupType = TypeVar("_TupType", bound=tuple[Any, ...])
_ArrayType = TypeVar("_ArrayType", bound=NDArray[Any])

# 所有公开的名称列表
__all__: list[str]

# ix_ 函数的重载定义
@overload
def ix_(*args: _FiniteNestedSequence[_SupportsDType[_DType]]) -> tuple[ndarray[Any, _DType], ...]: ...
@overload
def ix_(*args: str | _NestedSequence[str]) -> tuple[NDArray[str_], ...]: ...
@overload
def ix_(*args: bytes | _NestedSequence[bytes]) -> tuple[NDArray[bytes_], ...]: ...
@overload
def ix_(*args: bool | _NestedSequence[bool]) -> tuple[NDArray[np.bool], ...]: ...
@overload
def ix_(*args: int | _NestedSequence[int]) -> tuple[NDArray[int_], ...]: ...
@overload
def ix_(*args: float | _NestedSequence[float]) -> tuple[NDArray[float64], ...]: ...
@overload
def ix_(*args: complex | _NestedSequence[complex]) -> tuple[NDArray[complex128], ...]: ...

# nd_grid 类的泛型定义
class nd_grid(Generic[_BoolType]):
    sparse: _BoolType
    def __init__(self, sparse: _BoolType = ...) -> None: ...
    # nd_grid 的 __getitem__ 方法的重载定义
    @overload
    def __getitem__(
        self: nd_grid[Literal[False]],
        key: slice | Sequence[slice],
    ) -> NDArray[Any]: ...
    @overload
    def __getitem__(
        self: nd_grid[Literal[True]],
        key: slice | Sequence[slice],
    ) -> tuple[NDArray[Any], ...]: ...

# MGridClass 类继承自 nd_grid[Literal[False]]
class MGridClass(nd_grid[Literal[False]]):
    def __init__(self) -> None: ...

# mgrid 是 MGridClass 类的实例
mgrid: MGridClass

# OGridClass 类继承自 nd_grid[Literal[True]]
class OGridClass(nd_grid[Literal[True]]):
    def __init__(self) -> None: ...

# ogrid 是 OGridClass 类的实例
ogrid: OGridClass

# AxisConcatenator 类的定义
class AxisConcatenator:
    axis: int
    matrix: bool
    ndmin: int
    trans1d: int
    def __init__(
        self,
        axis: int = ...,
        matrix: bool = ...,
        ndmin: int = ...,
        trans1d: int = ...,
    ) -> None: ...
    
    # 静态方法 concatenate 的重载定义
    @staticmethod
    @overload
    def concatenate(  # type: ignore[misc]
        *a: ArrayLike, axis: SupportsIndex = ..., out: None = ...
    ) -> NDArray[Any]: ...
    @staticmethod
    @overload
    def concatenate(
        *a: ArrayLike, axis: SupportsIndex = ..., out: _ArrayType = ...
    ) -> _ArrayType: ...
    
    # 静态方法 makemat 的定义
    @staticmethod
    def makemat(
        data: ArrayLike, dtype: DTypeLike = ..., copy: bool = ...
    ) -> _Matrix[Any, Any]: ...

    # TODO: Sort out this `__getitem__` method
    # 定义特殊方法 __getitem__，用于实现对象的索引访问功能
    def __getitem__(self, key: Any) -> Any:
        # 在这里使用省略符号 ... 表示该方法当前没有实现具体的逻辑
        # 该方法通常用于支持对象的索引访问，但具体实现需要根据具体情况来编写
        ...
class RClass(AxisConcatenator):
    axis: Literal[0]  # 设置轴的值为0，表示在第一个轴上进行拼接
    matrix: Literal[False]  # 设置矩阵属性为False，表示不要求输入是矩阵
    ndmin: Literal[1]  # 设置最小维度为1
    trans1d: Literal[-1]  # 设置1维转换为-1

    def __init__(self) -> None: ...  # 初始化方法暂不实现，仅声明存在

r_: RClass  # 声明r_为RClass类型的实例

class CClass(AxisConcatenator):
    axis: Literal[-1]  # 设置轴的值为-1，表示在最后一个轴上进行拼接
    matrix: Literal[False]  # 设置矩阵属性为False，表示不要求输入是矩阵
    ndmin: Literal[2]  # 设置最小维度为2
    trans1d: Literal[0]  # 设置1维转换为0

    def __init__(self) -> None: ...  # 初始化方法暂不实现，仅声明存在

c_: CClass  # 声明c_为CClass类型的实例

class IndexExpression(Generic[_BoolType]):
    maketuple: _BoolType  # 定义maketuple属性，类型为_BoolType
    def __init__(self, maketuple: _BoolType) -> None: ...  # 初始化方法暂不实现，仅声明存在

    @overload
    def __getitem__(self, item: _TupType) -> _TupType: ...  # 声明__getitem__方法的一种重载，返回_TupType类型

    @overload
    def __getitem__(self: IndexExpression[Literal[True]], item: _T) -> tuple[_T]: ...
    # 当IndexExpression的类型为Literal[True]时，声明__getitem__方法的重载，返回类型为tuple[_T]

    @overload
    def __getitem__(self: IndexExpression[Literal[False]], item: _T) -> _T: ...
    # 当IndexExpression的类型为Literal[False]时，声明__getitem__方法的重载，返回类型为_T

index_exp: IndexExpression[Literal[True]]  # 声明index_exp为IndexExpression类型的实例，类型为Literal[True]
s_: IndexExpression[Literal[False]]  # 声明s_为IndexExpression类型的实例，类型为Literal[False]

def fill_diagonal(a: NDArray[Any], val: Any, wrap: bool = ...) -> None:
    # 填充数组的对角线元素
    ...

def diag_indices(n: int, ndim: int = ...) -> tuple[NDArray[int_], ...]:
    # 返回用于创建对角线索引的数组下标
    ...

def diag_indices_from(arr: ArrayLike) -> tuple[NDArray[int_], ...]:
    # 返回用于从给定数组创建对角线索引的数组下标
    ...

# NOTE: see `numpy/__init__.pyi` for `ndenumerate` and `ndindex`
# 注意：查看 `numpy/__init__.pyi` 文件以获取 `ndenumerate` 和 `ndindex` 的信息
```