# `D:\src\scipysrc\pandas\pandas\core\indexes\range.py`

```
from __future__ import annotations
# 导入将来版本的语法支持，用于类型注解

from collections.abc import (
    Callable,
    Hashable,
    Iterator,
)
# 导入抽象基类集合，包括可调用对象、可哈希对象和迭代器

from datetime import timedelta
# 导入时间间隔模块

import operator
# 导入操作符模块

from sys import getsizeof
# 导入获取对象占用内存大小的函数

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
    overload,
)
# 导入类型注解相关的模块

import numpy as np
# 导入NumPy库

from pandas._libs import (
    index as libindex,
    lib,
)
# 导入Pandas内部库，包括索引和其他支持库

from pandas._libs.lib import no_default
# 导入Pandas内部库中的特定函数

from pandas.compat.numpy import function as nv
# 导入Pandas与NumPy兼容性相关的函数

from pandas.util._decorators import (
    cache_readonly,
    doc,
)
# 导入Pandas内部实用工具装饰器

from pandas.core.dtypes.base import ExtensionDtype
# 导入Pandas核心数据类型基类的扩展数据类型

from pandas.core.dtypes.common import (
    ensure_platform_int,
    ensure_python_int,
    is_float,
    is_integer,
    is_scalar,
    is_signed_integer_dtype,
)
# 导入Pandas核心数据类型常用函数

from pandas.core.dtypes.generic import ABCTimedeltaIndex
# 导入Pandas核心通用数据类型的时间间隔索引抽象基类

from pandas.core import ops
# 导入Pandas核心操作模块

import pandas.core.common as com
# 导入Pandas核心通用功能模块

from pandas.core.construction import extract_array
# 导入Pandas核心构建模块中的数组提取函数

from pandas.core.indexers import check_array_indexer
# 导入Pandas核心索引器模块中的数组索引检查函数

import pandas.core.indexes.base as ibase
# 导入Pandas核心索引基类

from pandas.core.indexes.base import (
    Index,
    maybe_extract_name,
)
# 导入Pandas核心索引基类中的索引和可能提取名称函数

from pandas.core.ops.common import unpack_zerodim_and_defer
# 导入Pandas核心操作模块中的零维解包和延迟函数

if TYPE_CHECKING:
    from pandas._typing import (
        Axis,
        Dtype,
        JoinHow,
        NaPosition,
        NumpySorter,
        Self,
        npt,
    )
    # 如果处于类型检查模式，导入Pandas类型注解相关的类型

    from pandas import Series
    # 导入Pandas中的Series类型

_empty_range = range(0)
# 创建一个空的范围对象，包含一个元素0

_dtype_int64 = np.dtype(np.int64)
# 创建一个NumPy的int64数据类型对象

def min_fitting_element(start: int, step: int, lower_limit: int) -> int:
    """Returns the smallest element greater than or equal to the limit"""
    # 返回大于等于下限的最小元素
    no_steps = -(-(lower_limit - start) // abs(step))
    # 计算步数，确保超出下限
    return start + abs(step) * no_steps
    # 返回计算得到的元素值

class RangeIndex(Index):
    """
    Immutable Index implementing a monotonic integer range.

    RangeIndex is a memory-saving special case of an Index limited to representing
    monotonic ranges with a 64-bit dtype. Using RangeIndex may in some instances
    improve computing speed.

    This is the default index type used
    by DataFrame and Series when no explicit index is provided by the user.

    Parameters
    ----------
    start : int (default: 0), range, or other RangeIndex instance
        If int and "stop" is not given, interpreted as "stop" instead.
    stop : int (default: 0)
    step : int (default: 1)
    dtype : np.int64
        Unused, accepted for homogeneity with other index types.
    copy : bool, default False
        Unused, accepted for homogeneity with other index types.
    name : object, optional
        Name to be stored in the index.

    Attributes
    ----------
    start
    stop
    step

    Methods
    -------
    from_range

    See Also
    --------
    Index : The base pandas Index type.

    Examples
    --------
    >>> list(pd.RangeIndex(5))
    [0, 1, 2, 3, 4]

    >>> list(pd.RangeIndex(-2, 4))
    [-2, -1, 0, 1, 2, 3]

    >>> list(pd.RangeIndex(0, 10, 2))
    [0, 2, 4, 6, 8]

    >>> list(pd.RangeIndex(2, -10, -3))
    [2, -1, -4, -7]

    >>> list(pd.RangeIndex(0))
    []
    """
    # 定义一个不可变的索引类，实现单调整数范围

    def __init__(
        self,
        start: int = 0,
        stop: int = 0,
        step: int = 1,
        dtype: np.int64 = _dtype_int64,
        copy: bool = False,
        name=None,
    ):
        # 初始化方法，设置范围索引的起始、结束和步长，默认数据类型为int64

        if isinstance(start, (range, RangeIndex)):
            if stop == 0:
                stop = start.stop
            start = start.start
        # 如果start是范围对象或RangeIndex实例，且没有给定stop，则将其作为stop

        super().__init__(
            data=_empty_range,
            dtype=dtype,
            copy=copy,
            name=name,
        )
        # 调用父类(Index)的初始化方法，传入空范围数据、数据类型、复制标志和名称

        self._start = start
        self._stop = stop
        self._step = step
        # 设置实例的起始、结束和步长属性

    @property
    def start(self) -> int:
        return self._start
    # 返回范围索引的起始值

    @property
    def stop(self) -> int:
        return self._stop
    # 返回范围索引的结束值

    @property
    def step(self) -> int:
        return self._step
    # 返回范围索引的步长值

    @classmethod
    def from_range(cls, range_like) -> RangeIndex:
        # 从类方法创建范围索引对象

        start, stop, step = ibase.get_range_indexer(range_like)
        # 调用基类索引模块中的获取范围索引器函数，获取范围索引的起始、结束和步长

        return cls(start=start, stop=stop, step=step)
        # 返回新创建的范围索引对象
    # 创建一个空列表的 RangeIndex 对象，结果为空列表
    >>> list(pd.RangeIndex(1, 0))
    []
    
    
    
    """
    _typ = "rangeindex"
    _dtype_validation_metadata = (is_signed_integer_dtype, "signed integer")
    _range: range
    _values: np.ndarray
    
    
    @property
    def _engine_type(self) -> type[libindex.Int64Engine]:
        # 返回此类对象的引擎类型为 libindex.Int64Engine 类型
        return libindex.Int64Engine
    
    
    
    # --------------------------------------------------------------------
    # Constructors
    
    
    
    def __new__(
        cls,
        start=None,
        stop=None,
        step=None,
        dtype: Dtype | None = None,
        copy: bool = False,
        name: Hashable | None = None,
    ) -> Self:
        # 验证 dtype 是否合法
        cls._validate_dtype(dtype)
        # 提取可能存在的名称
        name = maybe_extract_name(name, start, cls)
    
        # 如果 start 是 RangeIndex 对象，则复制该对象
        if isinstance(start, cls):
            return start.copy(name=name)
        # 如果 start 是 range 对象，则创建新的 RangeIndex 对象
        elif isinstance(start, range):
            return cls._simple_new(start, name=name)
    
        # 验证参数的有效性
        if com.all_none(start, stop, step):
            raise TypeError("RangeIndex(...) must be called with integers")
    
        # 确保 start 是 Python 整数
        start = ensure_python_int(start) if start is not None else 0
    
        if stop is None:
            start, stop = 0, start
        else:
            stop = ensure_python_int(stop)
    
        # 确保 step 是 Python 整数，并且不为零
        step = ensure_python_int(step) if step is not None else 1
        if step == 0:
            raise ValueError("Step must not be zero")
    
        # 创建 range 对象 rng
        rng = range(start, stop, step)
        return cls._simple_new(rng, name=name)
    
    
    
    @classmethod
    def from_range(cls, data: range, name=None, dtype: Dtype | None = None) -> Self:
        """
        从 range 对象创建 :class:`pandas.RangeIndex` 对象。
    
        Returns
        -------
        RangeIndex
    
        Examples
        --------
        >>> pd.RangeIndex.from_range(range(5))
        RangeIndex(start=0, stop=5, step=1)
    
        >>> pd.RangeIndex.from_range(range(2, -10, -3))
        RangeIndex(start=2, stop=-10, step=-3)
        """
        # 如果 data 不是 range 对象，则抛出 TypeError 异常
        if not isinstance(data, range):
            raise TypeError(
                f"{cls.__name__}(...) must be called with object coercible to a "
                f"range, {data!r} was passed"
            )
        # 验证 dtype 是否合法
        cls._validate_dtype(dtype)
        # 创建新的 RangeIndex 对象
        return cls._simple_new(data, name=name)
    
    
    
    #  error: Argument 1 of "_simple_new" is incompatible with supertype "Index";
    #  supertype defines the argument type as
    #  "Union[ExtensionArray, ndarray[Any, Any]]"  [override]
    @classmethod
    def _simple_new(  # type: ignore[override]
        cls, values: range, name: Hashable | None = None
    ) -> Self:
        # 创建一个新的对象
        result = object.__new__(cls)
    
        # 断言 values 是 range 对象
        assert isinstance(values, range)
    
        # 将 values 赋值给 _range 属性
        result._range = values
        result._name = name
        result._cache = {}
        result._reset_identity()
        result._references = None
        return result
    
    
    
    @classmethod
    def _validate_dtype(cls, dtype: Dtype | None) -> None:
        # 如果 dtype 为 None，则直接返回，不进行验证
        if dtype is None:
            return

        # 获取类的 dtype 验证函数和期望的类型
        validation_func, expected = cls._dtype_validation_metadata
        # 如果传入的 dtype 不符合验证函数要求，则抛出 ValueError 异常
        if not validation_func(dtype):
            raise ValueError(
                f"Incorrect `dtype` passed: expected {expected}, received {dtype}"
            )

    # --------------------------------------------------------------------

    # error: Return type "Type[Index]" of "_constructor" incompatible with return
    # type "Type[RangeIndex]" in supertype "Index"
    @cache_readonly
    def _constructor(self) -> type[Index]:  # type: ignore[override]
        """返回用于构造的类"""
        return Index

    # error: Signature of "_data" incompatible with supertype "Index"
    @cache_readonly
    def _data(self) -> np.ndarray:  # type: ignore[override]
        """
        一个整数数组，出于性能原因只在需要时创建。

        构建的数组保存在 ``_cache`` 中。
        """
        return np.arange(self.start, self.stop, self.step, dtype=np.int64)

    def _get_data_as_items(self) -> list[tuple[str, int]]:
        """返回一个元组列表，包含起始、停止和步长"""
        rng = self._range
        return [("start", rng.start), ("stop", rng.stop), ("step", rng.step)]

    def __reduce__(self):
        # 序列化对象以便后续反序列化
        d = {"name": self._name}
        d.update(dict(self._get_data_as_items()))
        return ibase._new_Index, (type(self), d), None

    # --------------------------------------------------------------------
    # Rendering Methods

    def _format_attrs(self):
        """
        返回属性和格式化值的元组列表
        """
        attrs = cast("list[tuple[str, str | int]]", self._get_data_as_items())
        if self._name is not None:
            attrs.append(("name", ibase.default_pprint(self._name)))
        return attrs

    def _format_with_header(self, *, header: list[str], na_rep: str) -> list[str]:
        # 类似于 Index 实现，但更快
        if not len(self._range):
            return header
        first_val_str = str(self._range[0])
        last_val_str = str(self._range[-1])
        max_length = max(len(first_val_str), len(last_val_str))

        return header + [f"{x:<{max_length}}" for x in self._range]

    # --------------------------------------------------------------------

    @property
    def start(self) -> int:
        """
        `start` 参数的值（如果未提供则为 ``0``）。

        示例
        --------
        >>> idx = pd.RangeIndex(5)
        >>> idx.start
        0

        >>> idx = pd.RangeIndex(2, -10, -3)
        >>> idx.start
        2
        """
        # GH 25710
        return self._range.start

    @property
    def stop(self) -> int:
        """
        The value of the `stop` parameter.

        Examples
        --------
        >>> idx = pd.RangeIndex(5)
        >>> idx.stop
        5

        >>> idx = pd.RangeIndex(2, -10, -3)
        >>> idx.stop
        -10
        """
        # 返回索引的停止值
        return self._range.stop

    @property
    def step(self) -> int:
        """
        The value of the `step` parameter (``1`` if this was not supplied).

        Examples
        --------
        >>> idx = pd.RangeIndex(5)
        >>> idx.step
        1

        >>> idx = pd.RangeIndex(2, -10, -3)
        >>> idx.step
        -3

        Even if :class:`pandas.RangeIndex` is empty, ``step`` is still ``1`` if
        not supplied.

        >>> idx = pd.RangeIndex(1, 0)
        >>> idx.step
        1
        """
        # 返回索引的步长值，默认为 1
        # GH 25710
        return self._range.step

    @cache_readonly
    def nbytes(self) -> int:
        """
        Return the number of bytes in the underlying data.
        """
        # 计算索引底层数据占用的字节数
        rng = self._range
        return getsizeof(rng) + sum(
            getsizeof(getattr(rng, attr_name))
            for attr_name in ["start", "stop", "step"]
        )

    def memory_usage(self, deep: bool = False) -> int:
        """
        Memory usage of my values

        Parameters
        ----------
        deep : bool
            Introspect the data deeply, interrogate
            `object` dtypes for system-level memory consumption

        Returns
        -------
        bytes used

        Notes
        -----
        Memory usage does not include memory consumed by elements that
        are not components of the array if deep=False

        See Also
        --------
        numpy.ndarray.nbytes
        """
        # 返回索引值占用的内存字节数
        return self.nbytes

    @property
    def dtype(self) -> np.dtype:
        # 返回索引的数据类型
        return _dtype_int64

    @property
    def is_unique(self) -> bool:
        """return if the index has unique values"""
        # 返回索引是否具有唯一值
        return True

    @cache_readonly
    def is_monotonic_increasing(self) -> bool:
        # 返回索引是否单调递增
        return self._range.step > 0 or len(self) <= 1

    @cache_readonly
    def is_monotonic_decreasing(self) -> bool:
        # 返回索引是否单调递减
        return self._range.step < 0 or len(self) <= 1

    def __contains__(self, key: Any) -> bool:
        # 检查键是否存在于索引范围内
        hash(key)
        try:
            key = ensure_python_int(key)
        except (TypeError, OverflowError):
            return False
        return key in self._range

    @property
    def inferred_type(self) -> str:
        # 推断索引的数据类型
        return "integer"

    # --------------------------------------------------------------------
    # Indexing Methods

    @doc(Index.get_loc)
    # 返回指定键在索引中的位置
    def get_loc(self, key) -> int:
        # 如果键是整数或浮点数且可以转换为整数，则转换为整数类型
        if is_integer(key) or (is_float(key) and key.is_integer()):
            new_key = int(key)
            try:
                # 在索引中查找并返回键的位置
                return self._range.index(new_key)
            except ValueError as err:
                # 如果键不在索引中，则引发 KeyError
                raise KeyError(key) from err
        # 如果键是可哈希的类型，则引发 KeyError
        if isinstance(key, Hashable):
            raise KeyError(key)
        # 否则，检查索引错误并引发 KeyError
        self._check_indexing_error(key)
        raise KeyError(key)

    # 返回将目标转换为索引数组的索引器
    def _get_indexer(
        self,
        target: Index,
        method: str | None = None,
        limit: int | None = None,
        tolerance=None,
    ) -> npt.NDArray[np.intp]:
        # 如果 method、tolerance 或 limit 中任何一个不为 None，则调用父类方法
        if com.any_not_none(method, tolerance, limit):
            return super()._get_indexer(
                target, method=method, tolerance=tolerance, limit=limit
            )

        # 根据步长的正负性确定起始、结束和步长
        if self.step > 0:
            start, stop, step = self.start, self.stop, self.step
        else:
            # GH 28678: 为简单起见，对反向范围进行处理
            reverse = self._range[::-1]
            start, stop, step = reverse.start, reverse.stop, reverse.step

        # 将目标数组转换为 NumPy 数组
        target_array = np.asarray(target)
        # 计算目标数组与起始值的差距
        locs = target_array - start
        # 确定有效的索引位置
        valid = (locs % step == 0) & (locs >= 0) & (target_array < stop)
        # 不符合条件的位置设为 -1
        locs[~valid] = -1
        # 符合条件的位置进行步长调整
        locs[valid] = locs[valid] / step

        # 如果步长不等于 self.step，则需要将索引位置转换回原始值
        if step != self.step:
            # 我们已经反转了这个范围：将位置转换为原始的索引位置
            locs[valid] = len(self) - 1 - locs[valid]
        # 确保返回的索引位置是适合当前平台的整数类型
        return ensure_platform_int(locs)

    @cache_readonly
    # 返回是否应该将整数键视为位置键
    def _should_fallback_to_positional(self) -> bool:
        """
        Should an integer key be treated as positional?
        """
        return False

    # --------------------------------------------------------------------

    # 返回索引的列表表示形式
    def tolist(self) -> list[int]:
        return list(self._range)

    # 返回迭代器，迭代索引中的整数
    @doc(Index.__iter__)
    def __iter__(self) -> Iterator[int]:
        yield from self._range

    # 返回浅拷贝，根据输入的值和名称创建新的索引对象
    @doc(Index._shallow_copy)
    def _shallow_copy(self, values, name: Hashable = no_default):
        name = self._name if name is no_default else name

        # 如果值的数据类型是浮点数，则返回浮点数类型的索引对象
        if values.dtype.kind == "f":
            return Index(values, name=name, dtype=np.float64)
        # 如果值的数据类型是整数且维度为1，则根据特定情况返回 RangeIndex 或 Index 对象
        if values.dtype.kind == "i" and values.ndim == 1:
            # GH 46675 & 43885: 如果值是等间隔的，则返回更节省内存的 RangeIndex
            if len(values) == 1:
                start = values[0]
                new_range = range(start, start + self.step, self.step)
                return type(self)._simple_new(new_range, name=name)
            maybe_range = ibase.maybe_sequence_to_range(values)
            if isinstance(maybe_range, range):
                return type(self)._simple_new(maybe_range, name=name)
        # 否则，返回原始构造函数创建的新索引对象
        return self._constructor._simple_new(values, name=name)

    # 返回视图，即当前索引对象的浅拷贝
    def _view(self) -> Self:
        result = type(self)._simple_new(self._range, name=self._name)
        result._cache = self._cache
        return result
    # 根据目标和索引器对结果进行重新索引封装，保留名称如果指定
    def _wrap_reindex_result(self, target, indexer, preserve_names: bool):
        # 如果目标不是当前对象的实例并且目标的数据类型为整数
        if not isinstance(target, type(self)) and target.dtype.kind == "i":
            # 对目标的值进行浅拷贝，并指定名称为原始名称
            target = self._shallow_copy(target._values, name=target.name)
        # 调用父类的方法，返回重新索引后的结果
        return super()._wrap_reindex_result(target, indexer, preserve_names)

    # 用于复制当前索引对象，可选指定名称和深度复制选项
    @doc(Index.copy)
    def copy(self, name: Hashable | None = None, deep: bool = False) -> Self:
        # 验证名称有效性并返回有效名称
        name = self._validate_names(name=name, deep=deep)[0]
        # 使用新名称重命名当前索引对象，返回新的索引对象
        new_index = self._rename(name=name)
        return new_index

    # 返回 RangeIndex 对象的最小值或最大值，由 meth 参数指定是最小值还是最大值
    def _minmax(self, meth: Literal["min", "max"]) -> int | float:
        # 计算索引的步长
        no_steps = len(self) - 1
        # 如果索引长度为 -1，则返回 NaN
        if no_steps == -1:
            return np.nan
        # 根据 meth 参数和步长方向判断返回最小值或最大值
        elif (meth == "min" and self.step > 0) or (meth == "max" and self.step < 0):
            return self.start
        # 计算并返回根据步长计算的最小值或最大值
        return self.start + self.step * no_steps

    # 返回 RangeIndex 对象的最小值
    def min(self, axis=None, skipna: bool = True, *args, **kwargs) -> int | float:
        """RangeIndex 的最小值"""
        # 验证轴参数的有效性
        nv.validate_minmax_axis(axis)
        # 验证最小值的额外参数
        nv.validate_min(args, kwargs)
        # 调用 _minmax 方法计算并返回最小值
        return self._minmax("min")

    # 返回 RangeIndex 对象的最大值
    def max(self, axis=None, skipna: bool = True, *args, **kwargs) -> int | float:
        """RangeIndex 的最大值"""
        # 验证轴参数的有效性
        nv.validate_minmax_axis(axis)
        # 验证最大值的额外参数
        nv.validate_max(args, kwargs)
        # 调用 _minmax 方法计算并返回最大值
        return self._minmax("max")

    # 返回 RangeIndex 对象的最小值或最大值的位置索引
    def _argminmax(
        self,
        meth: Literal["min", "max"],
        axis=None,
        skipna: bool = True,
    ) -> int:
        # 验证轴参数的有效性
        nv.validate_minmax_axis(axis)
        # 如果索引长度为 0，则调用父类相应方法返回位置索引
        if len(self) == 0:
            return getattr(super(), f"arg{meth}")(
                axis=axis,
                skipna=skipna,
            )
        # 根据 meth 参数和步长方向判断返回最小值或最大值的位置索引
        elif meth == "min":
            if self.step > 0:
                return 0
            else:
                return len(self) - 1
        elif meth == "max":
            if self.step > 0:
                return len(self) - 1
            else:
                return 0
        else:
            # 抛出异常，说明 meth 参数必须是 "max" 或 "min"
            raise ValueError(f"{meth=} must be max or min")

    # 返回 RangeIndex 对象的最小值的位置索引
    def argmin(self, axis=None, skipna: bool = True, *args, **kwargs) -> int:
        # 验证 argmin 方法的额外参数
        nv.validate_argmin(args, kwargs)
        # 调用 _argminmax 方法计算并返回最小值的位置索引
        return self._argminmax("min", axis=axis, skipna=skipna)

    # 返回 RangeIndex 对象的最大值的位置索引
    def argmax(self, axis=None, skipna: bool = True, *args, **kwargs) -> int:
        # 验证 argmax 方法的额外参数
        nv.validate_argmax(args, kwargs)
        # 调用 _argminmax 方法计算并返回最大值的位置索引
        return self._argminmax("max", axis=axis, skipna=skipna)
    def argsort(self, *args, **kwargs) -> npt.NDArray[np.intp]:
        """
        Returns the indices that would sort the index and its
        underlying data.

        Returns
        -------
        np.ndarray[np.intp]

        See Also
        --------
        numpy.ndarray.argsort
        """
        # 从关键字参数中获取排序顺序，默认为升序
        ascending = kwargs.pop("ascending", True)  # EA compat
        # 从关键字参数中移除排序算法类型，例如 "mergesort" 对本方法无关
        kwargs.pop("kind", None)  # e.g. "mergesort" is irrelevant
        # 验证排序参数的有效性
        nv.validate_argsort(args, kwargs)

        start, stop, step = None, None, None
        # 根据步长确定起始、结束和步进值
        if self._range.step > 0:
            if ascending:
                start = len(self)
            else:
                start, stop, step = len(self) - 1, -1, -1
        elif ascending:
            start, stop, step = len(self) - 1, -1, -1
        else:
            start = len(self)

        # 返回一个以指定步进值创建的整数数组，用于排序
        return np.arange(start, stop, step, dtype=np.intp)

    def factorize(
        self,
        sort: bool = False,
        use_na_sentinel: bool = True,
    ) -> tuple[npt.NDArray[np.intp], RangeIndex]:
        # 如果需要排序且步长为负数，返回倒序排列的编码和唯一值
        if sort and self.step < 0:
            codes = np.arange(len(self) - 1, -1, -1, dtype=np.intp)
            uniques = self[::-1]
        else:
            # 否则返回正序排列的编码和唯一值
            codes = np.arange(len(self), dtype=np.intp)
            uniques = self
        # 返回编码数组和唯一值
        return codes, uniques

    def equals(self, other: object) -> bool:
        """
        Determines if two Index objects contain the same elements.
        """
        # 如果另一个对象是 RangeIndex 类型，比较两者的范围是否相等
        if isinstance(other, RangeIndex):
            return self._range == other._range
        # 否则调用父类方法比较两个对象是否相等
        return super().equals(other)

    # error: Signature of "sort_values" incompatible with supertype "Index"
    @overload  # type: ignore[override]
    def sort_values(
        self,
        *,
        return_indexer: Literal[False] = ...,
        ascending: bool = ...,
        na_position: NaPosition = ...,
        key: Callable | None = ...,
    ) -> Self: ...
    # 函数重载：根据指定参数对当前对象进行排序，返回排序后的对象本身

    @overload
    def sort_values(
        self,
        *,
        return_indexer: Literal[True],
        ascending: bool = ...,
        na_position: NaPosition = ...,
        key: Callable | None = ...,
    ) -> tuple[Self, np.ndarray | RangeIndex]: ...
    # 函数重载：根据指定参数对当前对象进行排序，返回排序后的对象和索引器数组的元组

    @overload
    def sort_values(
        self,
        *,
        return_indexer: bool = ...,
        ascending: bool = ...,
        na_position: NaPosition = ...,
        key: Callable | None = ...,
    ) -> Self | tuple[Self, np.ndarray | RangeIndex]: ...
    # 函数重载：根据指定参数对当前对象进行排序，根据返回索引器是否返回元组或对象本身

    def sort_values(
        self,
        *,
        return_indexer: bool = False,
        ascending: bool = True,
        na_position: NaPosition = "last",
        key: Callable | None = None,
    ) -> Self | tuple[Self, np.ndarray | RangeIndex]:
        # 根据指定参数对当前对象进行排序，并根据需求返回排序后的对象或对象及索引器数组的元组
    ) -> Self | tuple[Self, np.ndarray | RangeIndex]:
        # 如果 key 不为 None，则调用父类的 sort_values 方法，返回排序后的结果或者结果与索引器的元组
        if key is not None:
            return super().sort_values(
                return_indexer=return_indexer,
                ascending=ascending,
                na_position=na_position,
                key=key,
            )
        else:
            # 否则，如果 key 为 None，则根据 ascending 参数进行可能的倒序处理
            sorted_index = self
            inverse_indexer = False
            if ascending:
                if self.step < 0:
                    sorted_index = self[::-1]
                    inverse_indexer = True
            else:
                if self.step > 0:
                    sorted_index = self[::-1]
                    inverse_indexer = True

        # 如果需要返回索引器
        if return_indexer:
            # 根据是否倒序，创建对应的索引范围
            if inverse_indexer:
                rng = range(len(self) - 1, -1, -1)
            else:
                rng = range(len(self))
            # 返回排序后的索引以及对应的 RangeIndex
            return sorted_index, RangeIndex(rng)
        else:
            # 否则，只返回排序后的索引
            return sorted_index

    # --------------------------------------------------------------------
    # Set Operations

    def _intersection(self, other: Index, sort: bool = False):
        # 调用方负责确保 self 和 other 都非空

        # 如果 other 不是 RangeIndex 类型，则调用父类的 _intersection 方法
        if not isinstance(other, RangeIndex):
            return super()._intersection(other, sort=sort)

        # 根据步长确定正序或倒序的范围
        first = self._range[::-1] if self.step < 0 else self._range
        second = other._range[::-1] if other.step < 0 else other._range

        # 检查两个范围是否有交集
        int_low = max(first.start, second.start)
        int_high = min(first.stop, second.stop)
        if int_high <= int_low:
            # 如果没有交集，返回一个空的 RangeIndex
            return self._simple_new(_empty_range)

        # 方法提示：线性丢番图方程
        # 解决交集问题
        # 性能提示：对于相同的步长，可以使用更便宜的替代方法
        gcd, s, _ = self._extended_gcd(first.step, second.step)

        # 检查元素集合是否有交集
        if (first.start - second.start) % gcd:
            return self._simple_new(_empty_range)

        # 计算描述交集的 RangeIndex 的参数，忽略下限
        tmp_start = first.start + (second.start - first.start) * first.step // gcd * s
        new_step = first.step * second.step // gcd

        # 调整索引以适应限制的区间
        new_start = min_fitting_element(tmp_start, new_step, int_low)
        new_range = range(new_start, int_high, new_step)

        # 如果 self 和 other 都是倒序，但新范围不是，则反转新范围
        if (self.step < 0 and other.step < 0) is not (new_range.step < 0):
            new_range = new_range[::-1]

        # 返回新的 RangeIndex
        return self._simple_new(new_range)
    def _extended_gcd(self, a: int, b: int) -> tuple[int, int, int]:
        """
        Extended Euclidean algorithms to solve Bezout's identity:
           a*x + b*y = gcd(x, y)
        Finds one particular solution for x, y: s, t
        Returns: gcd, s, t
        """
        # Initialize variables for the algorithm
        s, old_s = 0, 1
        t, old_t = 1, 0
        r, old_r = b, a
        
        # Perform the extended Euclidean algorithm
        while r:
            quotient = old_r // r
            old_r, r = r, old_r - quotient * r
            old_s, s = s, old_s - quotient * s
            old_t, t = t, old_t - quotient * t
        
        # Return gcd and coefficients s, t satisfying a*s + b*t = gcd(a, b)
        return old_r, old_s, old_t

    def _range_in_self(self, other: range) -> bool:
        """Check if other range is contained in self"""
        # Check if the other range is completely within self
        if not other:
            return True  # An empty range is trivially contained
        if not self._range:
            return False  # If self's range is empty, containment is impossible
        if len(other) > 1 and other.step % self._range.step:
            return False  # If steps are incompatible, ranges cannot match
        return other.start in self._range and other[-1] in self._range

    def symmetric_difference(
        self, other, result_name: Hashable | None = None, sort=None
    ) -> Index:
        # Override symmetric_difference to handle RangeIndex specifically
        if not isinstance(other, RangeIndex) or sort is not None:
            return super().symmetric_difference(other, result_name, sort)
        
        # Compute symmetric difference for two RangeIndex objects
        left = self.difference(other)
        right = other.difference(self)
        result = left.union(right)

        # Optionally rename the result index
        if result_name is not None:
            result = result.rename(result_name)
        
        return result

    def _join_empty(
        self, other: Index, how: JoinHow, sort: bool
    ) -> tuple[Index, npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]:
        # Handle the case where 'other' is not a RangeIndex but is integer-based
        if not isinstance(other, RangeIndex) and other.dtype.kind == "i":
            other = self._shallow_copy(other._values, name=other.name)
        
        # Delegate to superclass method to perform the empty join operation
        return super()._join_empty(other, how=how, sort=sort)

    def _join_monotonic(
        self, other: Index, how: JoinHow = "left"
    ):
        # Handle joining two monotonic Index objects
        # This function's body is incomplete and would continue in actual implementation
        pass
    ) -> tuple[Index, npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]:
        # 这个方法目前只处理单调递增的情况
        # 检查参数 `other` 是否为当前对象的相同类型，如果不是则尝试浅复制 `other` 的值
        if not isinstance(other, type(self)):
            maybe_ri = self._shallow_copy(other._values, name=other.name)
            # 如果浅复制后的对象仍然不是当前对象的类型，则调用父类方法进行连接
            if not isinstance(maybe_ri, type(self)):
                return super()._join_monotonic(other, how=how)
            other = maybe_ri

        # 如果两个对象相等，则根据 `how` 参数返回对应的索引对象
        if self.equals(other):
            ret_index = other if how == "right" else self
            return ret_index, None, None

        # 根据 `how` 参数进行不同类型的连接操作
        if how == "left":
            join_index = self
            lidx = None
            ridx = other.get_indexer(join_index)
        elif how == "right":
            join_index = other
            lidx = self.get_indexer(join_index)
            ridx = None
        elif how == "inner":
            join_index = self.intersection(other)
            lidx = self.get_indexer(join_index)
            ridx = other.get_indexer(join_index)
        elif how == "outer":
            join_index = self.union(other)
            lidx = self.get_indexer(join_index)
            ridx = other.get_indexer(join_index)

        # 确保返回的索引类型是平台兼容的整数类型
        lidx = None if lidx is None else ensure_platform_int(lidx)
        ridx = None if ridx is None else ensure_platform_int(ridx)
        return join_index, lidx, ridx

    # --------------------------------------------------------------------

    # error: Return type "Index" of "delete" incompatible with return type
    #  "RangeIndex" in supertype "Index"
    def delete(self, loc) -> Index:  # type: ignore[override]
        # 在某些情况下可以保留 RangeIndex 类型，参见 DatetimeTimedeltaMixin._get_delete_Freq
        if is_integer(loc):
            # 处理整数类型的 loc 参数
            if loc in (0, -len(self)):
                return self[1:]
            if loc in (-1, len(self) - 1):
                return self[:-1]
            if len(self) == 3 and loc in (1, -2):
                return self[::2]

        elif lib.is_list_like(loc):
            # 将 loc 转换为整数数组，然后尝试创建切片对象
            slc = lib.maybe_indices_to_slice(np.asarray(loc, dtype=np.intp), len(self))

            if isinstance(slc, slice):
                # 委托给 RangeIndex._difference 方法，该方法优化以尽可能返回 RangeIndex 类型
                other = self[slc]
                return self.difference(other, sort=False)

        # 默认情况下调用父类的 delete 方法
        return super().delete(loc)
    def insert(self, loc: int, item) -> Index:
        # 检查插入的项是否是整数或浮点数
        if is_integer(item) or is_float(item):
            # 如果插入位置在开头或末尾，或者在中间插入，保留 RangeIndex
            if len(self) == 0 and loc == 0 and is_integer(item):
                # 如果索引长度为0且在开头插入整数，创建新的范围并返回新的 RangeIndex
                new_rng = range(item, item + self.step, self.step)
                return type(self)._simple_new(new_rng, name=self._name)
            elif len(self):
                rng = self._range
                if loc == 0 and item == self[0] - self.step:
                    # 如果在索引开头插入与第一个元素前一个步长相同的数，扩展范围
                    new_rng = range(rng.start - rng.step, rng.stop, rng.step)
                    return type(self)._simple_new(new_rng, name=self._name)

                elif loc == len(self) and item == self[-1] + self.step:
                    # 如果在索引末尾插入与最后一个元素后一个步长相同的数，扩展范围
                    new_rng = range(rng.start, rng.stop + rng.step, rng.step)
                    return type(self)._simple_new(new_rng, name=self._name)

                elif len(self) == 2 and item == self[0] + self.step / 2:
                    # 如果在索引长度为2时，在两个元素之间插入一半步长的数
                    step = int(self.step / 2)
                    new_rng = range(self.start, self.stop, step)
                    return type(self)._simple_new(new_rng, name=self._name)

        # 如果上述条件都不满足，则调用父类的插入方法
        return super().insert(loc, item)

    def __len__(self) -> int:
        """
        返回 RangeIndex 的长度
        """
        return len(self._range)

    @property
    def size(self) -> int:
        # 返回索引的大小，即长度
        return len(self)

    def __getitem__(self, key):
        """
        为标量和切片键保留 RangeIndex 类型
        """
        if key is Ellipsis:
            key = slice(None)
        if isinstance(key, slice):
            # 如果键是切片，则调用 _getitem_slice 方法处理
            return self._getitem_slice(key)
        elif is_integer(key):
            # 如果键是整数，将其转换为整数，并尝试从范围中获取值
            new_key = int(key)
            try:
                return self._range[new_key]
            except IndexError as err:
                raise IndexError(
                    f"index {key} is out of bounds for axis 0 with size {len(self)}"
                ) from err
        elif is_scalar(key):
            # 如果键是标量，则抛出索引错误
            raise IndexError(
                "only integers, slices (`:`), "
                "ellipsis (`...`), numpy.newaxis (`None`) "
                "and integer or boolean "
                "arrays are valid indices"
            )
        elif com.is_bool_indexer(key):
            # 如果键是布尔索引器，则将其转换为布尔数组
            if isinstance(getattr(key, "dtype", None), ExtensionDtype):
                key = key.to_numpy(dtype=bool, na_value=False)
            else:
                key = np.asarray(key, dtype=bool)
            check_array_indexer(self._range, key)  # type: ignore[arg-type]
            key = np.flatnonzero(key)
        try:
            # 尝试获取指定键的值
            return self.take(key)
        except (TypeError, ValueError):
            # 如果出现类型错误或值错误，则调用父类的 __getitem__ 方法处理
            return super().__getitem__(key)
    def _getitem_slice(self, slobj: slice) -> Self:
        """
        Fastpath for __getitem__ when we know we have a slice.
        """
        # Extract the slice from the internal range object
        res = self._range[slobj]
        # Create a new instance of the class with the sliced range and optional name
        return type(self)._simple_new(res, name=self._name)

    @unpack_zerodim_and_defer("__floordiv__")
    def __floordiv__(self, other):
        # Check if 'other' is an integer and not zero
        if is_integer(other) and other != 0:
            # Check if the current range is empty or divisible by 'other' for start and step
            if len(self) == 0 or self.start % other == 0 and self.step % other == 0:
                # Calculate new start, step, and stop for the floordiv operation
                start = self.start // other
                step = self.step // other
                stop = start + len(self) * step
                # Create a new instance with the modified range
                new_range = range(start, stop, step or 1)
                return self._simple_new(new_range, name=self._name)
            # Special case: when the range has only one element
            if len(self) == 1:
                start = self.start // other
                new_range = range(start, start + 1, 1)
                return self._simple_new(new_range, name=self._name)

        # Call the superclass method if conditions are not met
        return super().__floordiv__(other)

    # --------------------------------------------------------------------
    # Reductions

    def all(self, *args, **kwargs) -> bool:
        # Check if zero is not in the range
        return 0 not in self._range

    def any(self, *args, **kwargs) -> bool:
        # Check if any element in the range evaluates to True
        return any(self._range)

    # --------------------------------------------------------------------

    # error: Return type "RangeIndex | Index" of "round" incompatible with
    # return type "RangeIndex" in supertype "Index"
    def round(self, decimals: int = 0) -> Self | Index:  # type: ignore[override]
        """
        Round each value in the Index to the given number of decimals.

        Parameters
        ----------
        decimals : int, optional
            Number of decimal places to round to. If decimals is negative,
            it specifies the number of positions to the left of the decimal point
            e.g. ``round(11.0, -1) == 10.0``.

        Returns
        -------
        Index or RangeIndex
            A new Index with the rounded values.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.RangeIndex(10, 30, 10)
        >>> idx.round(decimals=-1)
        RangeIndex(start=10, stop=30, step=10)
        >>> idx = pd.RangeIndex(10, 15, 1)
        >>> idx.round(decimals=-1)
        Index([10, 10, 10, 10, 10], dtype='int64')
        """
        if decimals >= 0:
            # Return a copy of self if decimals is non-negative
            return self.copy()
        elif self.start % 10**-decimals == 0 and self.step % 10**-decimals == 0:
            # Return a copy of self if no rounding is needed for start and step
            return self.copy()
        else:
            # Call the superclass method for rounding with negative decimals
            return super().round(decimals=decimals)

    def _cmp_method(self, other, op):
        if isinstance(other, RangeIndex) and self._range == other._range:
            # Shortcut comparison if both instances have identical range attributes
            return super()._cmp_method(self, op)
        # Otherwise, fallback to superclass comparison method
        return super()._cmp_method(other, op)
    def _arith_method(self, other, op):
        """
        Parameters
        ----------
        other : Any
            另一个操作数，可以是任意类型
        op : callable that accepts 2 params
            二元操作的函数，接受两个参数

        Performs arithmetic operations based on the provided operator.

        """
        
        if isinstance(other, ABCTimedeltaIndex):
            # 如果 other 是 ABCTimedeltaIndex 类型，则委托给 TimedeltaIndex 的实现
            return NotImplemented
        elif isinstance(other, (timedelta, np.timedelta64)):
            # 对于 timedelta 或 np.timedelta64 类型的 other，需要特殊处理
            # GH#19333 在 timedelta64 上 is_integer 返回 True，需要显式处理
            return super()._arith_method(other, op)
        elif lib.is_np_dtype(getattr(other, "dtype", None), "m"):
            # 如果 other 是 np.ndarray 类型，特别是日期时间类型，委托给父类的实现
            # GH#22390
            return super()._arith_method(other, op)

        if op in [
            operator.pow,
            ops.rpow,
            operator.mod,
            ops.rmod,
            operator.floordiv,
            ops.rfloordiv,
            divmod,
            ops.rdivmod,
        ]:
            # 如果操作符属于特定的二元操作，委托给父类的实现
            return super()._arith_method(other, op)

        step: Callable | None = None
        if op in [operator.mul, ops.rmul, operator.truediv, ops.rtruediv]:
            # 如果操作符是乘法、真除法等，则设置 step 变量为该操作符
            step = op

        # TODO: 如果 other 是 RangeIndex，可能有更高效的选项
        # 提取 other 的数组表示，支持 numpy 和 RangeIndex
        right = extract_array(other, extract_numpy=True, extract_range=True)
        left = self

        try:
            # 尝试应用操作符，如果有自定义的操作符
            if step:
                with np.errstate(all="ignore"):
                    rstep = step(left.step, right)

                # 如果 rstep 无法表示或为零，抛出 ValueError
                if not is_integer(rstep) or not rstep:
                    raise ValueError

            # 处理 GH#53255
            else:
                rstep = -left.step if op == ops.rsub else left.step

            with np.errstate(all="ignore"):
                rstart = op(left.start, right)
                rstop = op(left.stop, right)

            # 获取操作结果的名称
            res_name = ops.get_op_result_name(self, other)
            # 根据计算结果创建新的对象，保留索引名称
            result = type(self)(rstart, rstop, rstep, name=res_name)

            # 如果 rstart、rstop、rstep 中有任何一个不是整数，将结果转换为 float64 类型
            if not all(is_integer(x) for x in [rstart, rstop, rstep]):
                result = result.astype("float64")

            return result

        except (ValueError, TypeError, ZeroDivisionError):
            # 处理异常情况，例如转换失败或零除错误
            # test_arithmetic_explicit_conversions
            return super()._arith_method(other, op)

    def __abs__(self) -> Self | Index:
        # 如果索引长度为零或最小值大于等于零，返回索引的复制
        if len(self) == 0 or self.min() >= 0:
            return self.copy()
        # 如果最大值小于等于零，返回索引的负值
        elif self.max() <= 0:
            return -self
        else:
            # 否则委托给父类的绝对值实现
            return super().__abs__()

    def __neg__(self) -> Self:
        # 创建一个以负步长生成的新索引
        rng = range(-self.start, -self.stop, -self.step)
        return self._simple_new(rng, name=self.name)

    def __pos__(self) -> Self:
        # 返回索引的复制
        return self.copy()
    def __invert__(self) -> Self:
        # 如果对象长度为0，则返回其副本
        if len(self) == 0:
            return self.copy()
        # 计算反向范围，并返回新的对象
        rng = range(~self.start, ~self.stop, -self.step)
        return self._simple_new(rng, name=self.name)

    # error: Return type "Index" of "take" incompatible with return type
    # "RangeIndex" in supertype "Index"
    def take(  # type: ignore[override]
        self,
        indices,
        axis: Axis = 0,
        allow_fill: bool = True,
        fill_value=None,
        **kwargs,
    ) -> Self | Index:
        # 如果有关键字参数，则验证并抛出异常
        if kwargs:
            nv.validate_take((), kwargs)
        # 如果indices是标量，则抛出类型错误异常
        if is_scalar(indices):
            raise TypeError("Expected indices to be array-like")
        # 确保indices是平台整数
        indices = ensure_platform_int(indices)

        # 如果允许填充并且填充值不为None，则抛出异常
        self._maybe_disallow_fill(allow_fill, fill_value, indices)

        # 如果indices长度为0，则返回空范围的类型对象
        if len(indices) == 0:
            return type(self)(_empty_range, name=self.name)
        else:
            # 计算indices的最大值和最小值
            ind_max = indices.max()
            ind_min = indices.min()

            # 如果最大值超出了对象长度，则抛出索引错误异常
            if ind_max >= len(self):
                raise IndexError(
                    f"index {ind_max} is out of bounds for axis 0 with size {len(self)}"
                )
            # 如果最小值小于负对象长度，则抛出索引错误异常
            if ind_min < -len(self):
                raise IndexError(
                    f"index {ind_min} is out of bounds for axis 0 with size {len(self)}"
                )
            # 将indices转换为指定的数据类型，进行安全类型转换
            taken = indices.astype(self.dtype, casting="safe")
            # 如果最小值小于0，则对其取模
            if ind_min < 0:
                taken %= len(self)
            # 如果对象的步长不为1，则乘以步长
            if self.step != 1:
                taken *= self.step
            # 如果对象的起始值不为0，则加上起始值
            if self.start != 0:
                taken += self.start

        # 返回对象的浅复制，用新的indices替换，设置名称为原名称
        return self._shallow_copy(taken, name=self.name)

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        bins=None,
        dropna: bool = True,
    ) -> Series:
        # 导入Series类
        from pandas import Series

        # 如果指定了bins，则调用父类的value_counts方法
        if bins is not None:
            return super().value_counts(
                normalize=normalize,
                sort=sort,
                ascending=ascending,
                bins=bins,
                dropna=dropna,
            )
        # 否则，根据normalize的值选择名称为"proportion"或"count"
        name = "proportion" if normalize else "count"
        # 创建长度为self长度的数据数组，根据normalize的值进行初始化
        data: npt.NDArray[np.floating] | npt.NDArray[np.signedinteger] = np.ones(
            len(self), dtype=np.int64
        )
        if normalize:
            data = data / len(self)
        # 返回一个Series对象，索引为self的副本，数据为data，名称为name
        return Series(data, index=self.copy(), name=name)

    def searchsorted(  # type: ignore[override]
        self,
        value,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        # 定义函数的签名，指定输入参数和返回类型为 numpy 数组或整数

        if side not in {"left", "right"} or sorter is not None:
            # 如果侧边参数不是 "left" 或 "right"，或者排序器不为 None，则执行基类的搜索操作
            return super().searchsorted(value=value, side=side, sorter=sorter)

        was_scalar = False
        # 初始化标志变量，用于标记输入值是否为标量
        if is_scalar(value):
            # 如果输入值是标量
            was_scalar = True
            # 设置标志变量为 True
            array_value = np.array([value])
            # 将标量值转换为包含单个元素的 numpy 数组
        else:
            # 如果输入值不是标量
            array_value = np.asarray(value)
            # 将输入值转换为 numpy 数组
        if array_value.dtype.kind not in "iu":
            # 如果数组的数据类型不是整数（无符号或有符号）
            return super().searchsorted(value=value, side=side, sorter=sorter)
            # 执行基类的搜索操作

        if flip := (self.step < 0):
            # 使用 walrus 操作符检查步长是否为负数，并将结果赋给 flip 变量
            rng = self._range[::-1]
            # 对象的范围反转
            start = rng.start
            # 设置起始位置为反转范围的起始位置
            step = rng.step
            # 设置步长为反转范围的步长
            shift = side == "right"
            # 设置移动标志为 True，如果 side 为 "right"，否则为 False
        else:
            # 如果步长不是负数
            start = self.start
            # 设置起始位置为对象的起始位置
            step = self.step
            # 设置步长为对象的步长
            shift = side == "left"
            # 设置移动标志为 True，如果 side 为 "left"，否则为 False
        result = (array_value - start - int(shift)) // step + 1
        # 计算结果为（数组值 - 起始值 - 移动标志的整数值）整除步长再加 1
        if flip:
            # 如果 flip 为 True（即步长为负数）
            result = len(self) - result
            # 计算结果为对象长度减去 result
        result = np.maximum(np.minimum(result, len(self)), 0)
        # 限制结果的范围在 [0, len(self)] 内
        if was_scalar:
            # 如果输入值是标量
            return np.intp(result.item())
            # 返回结果的整数类型
        return result.astype(np.intp, copy=False)
        # 返回结果的整数类型，确保不复制数据
```