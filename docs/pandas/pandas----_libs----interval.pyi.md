# `D:\src\scipysrc\pandas\pandas\_libs\interval.pyi`

```
from typing import (
    Any,
    Generic,
    TypeVar,
    overload,
)

import numpy as np
import numpy.typing as npt

from pandas._typing import (
    IntervalClosedType,
    Timedelta,
    Timestamp,
)

VALID_CLOSED: frozenset[str]

_OrderableScalarT = TypeVar("_OrderableScalarT", int, float)
_OrderableTimesT = TypeVar("_OrderableTimesT", Timestamp, Timedelta)
_OrderableT = TypeVar("_OrderableT", int, float, Timestamp, Timedelta)

class _LengthDescriptor:
    @overload
    def __get__(
        self, instance: Interval[_OrderableScalarT], owner: Any
    ) -> _OrderableScalarT: ...
    @overload
    def __get__(
        self, instance: Interval[_OrderableTimesT], owner: Any
    ) -> Timedelta: ...
    # _LengthDescriptor 类定义了两个重载的 __get__ 方法，用于获取 Interval 实例的长度信息

class _MidDescriptor:
    @overload
    def __get__(self, instance: Interval[_OrderableScalarT], owner: Any) -> float: ...
    @overload
    def __get__(
        self, instance: Interval[_OrderableTimesT], owner: Any
    ) -> _OrderableTimesT: ...
    # _MidDescriptor 类定义了两个重载的 __get__ 方法，用于获取 Interval 实例的中点信息

class IntervalMixin:
    @property
    def closed_left(self) -> bool: ...
    @property
    def closed_right(self) -> bool: ...
    @property
    def open_left(self) -> bool: ...
    @property
    def open_right(self) -> bool: ...
    @property
    def is_empty(self) -> bool: ...
    def _check_closed_matches(self, other: IntervalMixin, name: str = ...) -> None: ...
    # IntervalMixin 类定义了一些属性和方法，用于处理区间（Interval）的开闭状态和空值判断

class Interval(IntervalMixin, Generic[_OrderableT]):
    @property
    def left(self: Interval[_OrderableT]) -> _OrderableT: ...
    @property
    def right(self: Interval[_OrderableT]) -> _OrderableT: ...
    @property
    def closed(self) -> IntervalClosedType: ...
    mid: _MidDescriptor
    length: _LengthDescriptor
    def __init__(
        self,
        left: _OrderableT,
        right: _OrderableT,
        closed: IntervalClosedType = ...,
    ) -> None: ...
    def __hash__(self) -> int: ...
    @overload
    def __contains__(
        self: Interval[Timedelta], key: Timedelta | Interval[Timedelta]
    ) -> bool: ...
    @overload
    def __contains__(
        self: Interval[Timestamp], key: Timestamp | Interval[Timestamp]
    ) -> bool: ...
    @overload
    def __contains__(
        self: Interval[_OrderableScalarT],
        key: _OrderableScalarT | Interval[_OrderableScalarT],
    ) -> bool: ...
    @overload
    def __add__(
        self: Interval[_OrderableTimesT], y: Timedelta
    ) -> Interval[_OrderableTimesT]: ...
    @overload
    def __add__(
        self: Interval[int], y: _OrderableScalarT
    ) -> Interval[_OrderableScalarT]: ...
    @overload
    def __add__(self: Interval[float], y: float) -> Interval[float]: ...
    @overload
    def __radd__(
        self: Interval[_OrderableTimesT], y: Timedelta
    ) -> Interval[_OrderableTimesT]: ...
    @overload
    def __radd__(
        self: Interval[int], y: _OrderableScalarT
    ) -> Interval[_OrderableScalarT]: ...
    @overload
    def __radd__(self: Interval[float], y: float) -> Interval[float]: ...
    @overload
    # Interval 类定义了各种方法和属性，用于表示和操作不同类型的区间
    # 定义 Interval 类的减法运算符重载方法，接受 Timedelta 类型的参数，并返回一个 Interval 对象
    def __sub__(
        self: Interval[_OrderableTimesT], y: Timedelta
    ) -> Interval[_OrderableTimesT]: ...

    # 重载 __sub__ 方法，接受 int 类型的参数，返回一个 Interval 对象，参数类型为 _OrderableScalarT
    @overload
    def __sub__(
        self: Interval[int], y: _OrderableScalarT
    ) -> Interval[_OrderableScalarT]: ...

    # 重载 __sub__ 方法，接受 float 类型的参数，返回一个 Interval 对象，参数类型为 float
    @overload
    def __sub__(self: Interval[float], y: float) -> Interval[float]: ...

    # 定义 Interval 类的右减法运算符重载方法，接受 Timedelta 类型的参数，并返回一个 Interval 对象
    @overload
    def __rsub__(
        self: Interval[_OrderableTimesT], y: Timedelta
    ) -> Interval[_OrderableTimesT]: ...

    # 重载 __rsub__ 方法，接受 int 类型的参数，返回一个 Interval 对象，参数类型为 _OrderableScalarT
    @overload
    def __rsub__(
        self: Interval[int], y: _OrderableScalarT
    ) -> Interval[_OrderableScalarT]: ...

    # 重载 __rsub__ 方法，接受 float 类型的参数，返回一个 Interval 对象，参数类型为 float
    @overload
    def __rsub__(self: Interval[float], y: float) -> Interval[float]: ...

    # 重载乘法运算符，接受 int 类型的参数，返回一个 Interval 对象，参数类型为 _OrderableScalarT
    @overload
    def __mul__(
        self: Interval[int], y: _OrderableScalarT
    ) -> Interval[_OrderableScalarT]: ...

    # 重载乘法运算符，接受 float 类型的参数，返回一个 Interval 对象，参数类型为 float
    @overload
    def __mul__(self: Interval[float], y: float) -> Interval[float]: ...

    # 重载右乘法运算符，接受 int 类型的参数，返回一个 Interval 对象，参数类型为 _OrderableScalarT
    @overload
    def __rmul__(
        self: Interval[int], y: _OrderableScalarT
    ) -> Interval[_OrderableScalarT]: ...

    # 重载右乘法运算符，接受 float 类型的参数，返回一个 Interval 对象，参数类型为 float
    @overload
    def __rmul__(self: Interval[float], y: float) -> Interval[float]: ...

    # 重载除法运算符，接受 int 类型的参数，返回一个 Interval 对象，参数类型为 _OrderableScalarT
    @overload
    def __truediv__(
        self: Interval[int], y: _OrderableScalarT
    ) -> Interval[_OrderableScalarT]: ...

    # 重载除法运算符，接受 float 类型的参数，返回一个 Interval 对象，参数类型为 float
    @overload
    def __truediv__(self: Interval[float], y: float) -> Interval[float]: ...

    # 重载整除运算符，接受 int 类型的参数，返回一个 Interval 对象，参数类型为 _OrderableScalarT
    @overload
    def __floordiv__(
        self: Interval[int], y: _OrderableScalarT
    ) -> Interval[_OrderableScalarT]: ...

    # 重载整除运算符，接受 float 类型的参数，返回一个 Interval 对象，参数类型为 float
    @overload
    def __floordiv__(self: Interval[float], y: float) -> Interval[float]: ...

    # 定义 overlaps 方法，用于检查当前 Interval 对象与另一个 Interval 对象是否重叠，返回布尔值
    def overlaps(self: Interval[_OrderableT], other: Interval[_OrderableT]) -> bool: ...
# 定义一个函数 intervals_to_interval_bounds，将间隔数组转换为间隔边界数组
# intervals: 包含间隔信息的 numpy 数组
# validate_closed: 是否验证间隔的闭合类型，默认为...
def intervals_to_interval_bounds(
    intervals: np.ndarray, validate_closed: bool = ...
) -> tuple[np.ndarray, np.ndarray, IntervalClosedType]: ...

# 定义一个类 IntervalTree，实现了 IntervalMixin 接口
class IntervalTree(IntervalMixin):
    # 构造方法，初始化 IntervalTree 实例
    # left: 包含左边界信息的 numpy 数组
    # right: 包含右边界信息的 numpy 数组
    # closed: 间隔的闭合类型，默认为...
    # leaf_size: 叶子节点大小，默认为...
    def __init__(
        self,
        left: np.ndarray,
        right: np.ndarray,
        closed: IntervalClosedType = ...,
        leaf_size: int = ...,
    ) -> None: ...

    # 属性方法 mid，返回中间值的 numpy 数组
    @property
    def mid(self) -> np.ndarray: ...

    # 属性方法 length，返回长度的 numpy 数组
    @property
    def length(self) -> np.ndarray: ...

    # 方法 get_indexer，返回目标索引的整数 numpy 数组
    # target: 目标值
    def get_indexer(self, target) -> npt.NDArray[np.intp]: ...

    # 方法 get_indexer_non_unique，返回目标索引的整数 numpy 数组和非唯一目标索引的整数 numpy 数组
    # target: 目标值
    def get_indexer_non_unique(
        self, target
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]: ...

    # 私有属性 _na_count，表示缺失值的数量
    _na_count: int

    # 属性方法 is_overlapping，检查是否存在重叠
    @property
    def is_overlapping(self) -> bool: ...

    # 属性方法 is_monotonic_increasing，检查是否单调递增
    @property
    def is_monotonic_increasing(self) -> bool: ...

    # 方法 clear_mapping，清除映射关系
    def clear_mapping(self) -> None: ...
```