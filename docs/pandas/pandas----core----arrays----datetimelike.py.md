# `D:\src\scipysrc\pandas\pandas\core\arrays\datetimelike.py`

```
from __future__ import annotations
# 导入未来版本兼容的 annotations 功能

from datetime import (
    datetime,           # 导入 datetime 对象
    timedelta,          # 导入 timedelta 对象
)
from functools import wraps  # 导入 wraps 装饰器
import operator  # 导入 operator 操作符模块
from typing import (
    TYPE_CHECKING,      # 导入 TYPE_CHECKING 常量
    Any,                # 导入 Any 类型
    Literal,            # 导入 Literal 泛型
    Union,              # 导入 Union 泛型
    cast,               # 导入 cast 函数
    final,              # 导入 final 装饰器
    overload,           # 导入 overload 装饰器
)
import warnings  # 导入 warnings 模块

import numpy as np  # 导入 NumPy 库

from pandas._config.config import get_option  # 导入 pandas 配置获取函数

from pandas._libs import (
    algos,              # 导入 pandas 内部算法库
    lib,                # 导入 pandas 内部库
)
from pandas._libs.tslibs import (
    BaseOffset,         # 导入 BaseOffset 类
    IncompatibleFrequency,  # 导入 IncompatibleFrequency 异常
    NaT,                # 导入 NaT 常量
    NaTType,            # 导入 NaTType 类型
    Period,             # 导入 Period 类
    Resolution,         # 导入 Resolution 类
    Tick,               # 导入 Tick 类
    Timedelta,          # 导入 Timedelta 类
    Timestamp,          # 导入 Timestamp 类
    add_overflowsafe,   # 导入 add_overflowsafe 函数
    astype_overflowsafe,    # 导入 astype_overflowsafe 函数
    get_unit_from_dtype,    # 导入 get_unit_from_dtype 函数
    iNaT,               # 导入 iNaT 常量
    ints_to_pydatetime, # 导入 ints_to_pydatetime 函数
    ints_to_pytimedelta,    # 导入 ints_to_pytimedelta 函数
    periods_per_day,    # 导入 periods_per_day 常量
    to_offset,          # 导入 to_offset 函数
)
from pandas._libs.tslibs.fields import (
    RoundTo,            # 导入 RoundTo 类
    round_nsint64,      # 导入 round_nsint64 函数
)
from pandas._libs.tslibs.np_datetime import compare_mismatched_resolutions  # 导入 compare_mismatched_resolutions 函数
from pandas._libs.tslibs.timedeltas import get_unit_for_round  # 导入 get_unit_for_round 函数
from pandas._libs.tslibs.timestamps import integer_op_not_supported  # 导入 integer_op_not_supported 异常
from pandas._typing import (
    ArrayLike,          # 导入 ArrayLike 泛型
    AxisInt,            # 导入 AxisInt 类型
    DatetimeLikeScalar, # 导入 DatetimeLikeScalar 泛型
    Dtype,              # 导入 Dtype 类型
    DtypeObj,           # 导入 DtypeObj 类型
    F,                  # 导入 F 泛型
    InterpolateOptions, # 导入 InterpolateOptions 类型
    NpDtype,            # 导入 NpDtype 泛型
    PositionalIndexer2D,    # 导入 PositionalIndexer2D 类型
    PositionalIndexerTuple, # 导入 PositionalIndexerTuple 类型
    ScalarIndexer,      # 导入 ScalarIndexer 类型
    Self,               # 导入 Self 泛型
    SequenceIndexer,    # 导入 SequenceIndexer 泛型
    TimeAmbiguous,      # 导入 TimeAmbiguous 类型
    TimeNonexistent,    # 导入 TimeNonexistent 类型
    npt,                # 导入 npt 常量
)
from pandas.compat.numpy import function as nv  # 导入 NumPy 兼容函数

from pandas.errors import (
    AbstractMethodError,    # 导入 AbstractMethodError 异常
    InvalidComparison,      # 导入 InvalidComparison 异常
    PerformanceWarning,     # 导入 PerformanceWarning 警告
)
from pandas.util._decorators import (
    Appender,           # 导入 Appender 装饰器
    Substitution,       # 导入 Substitution 装饰器
    cache_readonly,     # 导入 cache_readonly 装饰器
)
from pandas.util._exceptions import find_stack_level  # 导入 find_stack_level 函数

from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike  # 导入从类似列表构建 1 维对象数组的函数
from pandas.core.dtypes.common import (
    is_all_strings,     # 导入 is_all_strings 函数
    is_integer_dtype,   # 导入 is_integer_dtype 函数
    is_list_like,       # 导入 is_list_like 函数
    is_object_dtype,    # 导入 is_object_dtype 函数
    is_string_dtype,    # 导入 is_string_dtype 函数
    pandas_dtype,       # 导入 pandas_dtype 函数
)
from pandas.core.dtypes.dtypes import (
    ArrowDtype,         # 导入 ArrowDtype 类
    CategoricalDtype,   # 导入 CategoricalDtype 类
    DatetimeTZDtype,    # 导入 DatetimeTZDtype 类
    ExtensionDtype,     # 导入 ExtensionDtype 类
    PeriodDtype,        # 导入 PeriodDtype 类
)
from pandas.core.dtypes.generic import (
    ABCCategorical,     # 导入 ABCCategorical 抽象基类
    ABCMultiIndex,      # 导入 ABCMultiIndex 抽象基类
)
from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype,  # 导入 is_valid_na_for_dtype 函数
    isna,               # 导入 isna 函数
)

from pandas.core import (
    algorithms,         # 导入 pandas 算法模块
    missing,            # 导入 pandas 缺失数据处理模块
    nanops,             # 导入 pandas 空值操作模块
    ops,                # 导入 pandas 操作模块
)
from pandas.core.algorithms import (
    isin,               # 导入 isin 函数
    map_array,          # 导入 map_array 函数
    unique1d,           # 导入 unique1d 函数
)
from pandas.core.array_algos import datetimelike_accumulations  # 导入 datetimelike_accumulations 函数
from pandas.core.arraylike import OpsMixin  # 导入 OpsMixin 类
from pandas.core.arrays._mixins import (
    NDArrayBackedExtensionArray,    # 导入 NDArrayBackedExtensionArray 类
    ravel_compat,       # 导入 ravel_compat 函数
)
from pandas.core.arrays.arrow.array import ArrowExtensionArray  # 导入 ArrowExtensionArray 类
from pandas.core.arrays.base import ExtensionArray  # 导入 ExtensionArray 类
from pandas.core.arrays.integer import IntegerArray  # 导入 IntegerArray 类
import pandas.core.common as com  # 导入 pandas 公共函数
from pandas.core.construction import (
    array as pd_array,   # 导入 pandas 数组构造函数
    ensure_wrapped_if_datetimelike, # 导入 ensure_wrapped_if_datetimelike 函数
    extract_array,      # 导入 extract_array 函数
)
from pandas.core.indexers import (
    check_array_indexer,    # 导入 check_array_indexer 函数
    check_setitem_lengths,  # 导入 check_setitem_lengths 函数
)
from pandas.core.ops.common import unpack_zerodim_and_defer  # 导入 unpack_zerodim_and_defer 函数
from pandas.core.ops.invalid import (
    # 导入 pandas 操作无效模块中的异常和警告
    invalid_comparison,  # 引入 invalid_comparison 模块
    make_invalid_op,     # 引入 make_invalid_op 模块
# 导入 TYPE_CHECKING，用于在类型检查模式下导入特定模块
if TYPE_CHECKING:
    # 从 collections.abc 导入 Callable、Iterator、Sequence 类型
    from collections.abc import (
        Callable,
        Iterator,
        Sequence,
    )

    # 从 pandas 导入 Index 类型
    from pandas import Index
    # 从 pandas.core.arrays 导入 DatetimeArray、PeriodArray、TimedeltaArray 类型
    from pandas.core.arrays import (
        DatetimeArray,
        PeriodArray,
        TimedeltaArray,
    )

# 定义 DTScalarOrNaT 类型别名，表示 Union[DatetimeLikeScalar, NaTType]
DTScalarOrNaT = Union[DatetimeLikeScalar, NaTType]


# 定义 _make_unpacked_invalid_op 函数，返回一个通过 make_invalid_op 函数生成的操作
def _make_unpacked_invalid_op(op_name: str):
    op = make_invalid_op(op_name)
    return unpack_zerodim_and_defer(op_name)(op)


# 定义 _period_dispatch 函数，用于 PeriodArray 方法的分发，将结果重新包装成 PeriodArray
def _period_dispatch(meth: F) -> F:
    """
    For PeriodArray methods, dispatch to DatetimeArray and re-wrap the results
    in PeriodArray.  We cannot use ._ndarray directly for the affected
    methods because the i8 data has different semantics on NaT values.
    """

    @wraps(meth)
    def new_meth(self, *args, **kwargs):
        if not isinstance(self.dtype, PeriodDtype):
            return meth(self, *args, **kwargs)

        arr = self.view("M8[ns]")
        result = meth(arr, *args, **kwargs)
        if result is NaT:
            return NaT
        elif isinstance(result, Timestamp):
            return self._box_func(result._value)

        res_i8 = result.view("i8")
        return self._from_backing_data(res_i8)

    return cast(F, new_meth)


# 错误: 在基类 "NDArrayBacked" 中的 "_concat_same_type" 定义与在基类 "ExtensionArray" 中的定义不兼容
class DatetimeLikeArrayMixin(  # type: ignore[misc]
    OpsMixin, NDArrayBackedExtensionArray
):
    """
    DatetimeArray、TimedeltaArray、PeriodArray 的共享基类/混合类

    假定 __new__/__init__ 定义了：
        _ndarray

    并且继承的子类实现了：
        freq
    """

    # _infer_matches -> which infer_dtype strings are close enough to our own
    _infer_matches: tuple[str, ...]
    _is_recognized_dtype: Callable[[DtypeObj], bool]
    _recognized_scalars: tuple[type, ...]
    _ndarray: np.ndarray
    freq: BaseOffset | None

    @cache_readonly
    def _can_hold_na(self) -> bool:
        return True

    def __init__(
        self, data, dtype: Dtype | None = None, freq=None, copy: bool = False
    ) -> None:
        raise AbstractMethodError(self)

    @property
    def _scalar_type(self) -> type[DatetimeLikeScalar]:
        """
        与此日期类型关联的标量类型

        * PeriodArray : Period
        * DatetimeArray : Timestamp
        * TimedeltaArray : Timedelta
        """
        raise AbstractMethodError(self)

    def _scalar_from_string(self, value: str) -> DTScalarOrNaT:
        """
        从字符串构造一个标量类型。

        参数
        ----------
        value : str

        返回
        -------
        Period、Timestamp、Timedelta 或 NaT
            取决于 ``self._scalar_type`` 的类型。

        注意
        -----
        在拆箱结果之前，应调用 ``self._check_compatible_with``。
        """
        raise AbstractMethodError(self)
    def _unbox_scalar(
        self, value: DTScalarOrNaT
    ) -> np.int64 | np.datetime64 | np.timedelta64:
        """
        Unbox the integer value of a scalar `value`.

        Parameters
        ----------
        value : Period, Timestamp, Timedelta, or NaT
            Depending on subclass.

        Returns
        -------
        int

        Examples
        --------
        >>> arr = pd.array(np.array(["1970-01-01"], "datetime64[ns]"))
        >>> arr._unbox_scalar(arr[0])
        numpy.datetime64('1970-01-01T00:00:00.000000000')
        """
        raise AbstractMethodError(self)

    def _check_compatible_with(self, other: DTScalarOrNaT) -> None:
        """
        Verify that `self` and `other` are compatible.

        * DatetimeArray verifies that the timezones (if any) match
        * PeriodArray verifies that the freq matches
        * Timedelta has no verification

        In each case, NaT is considered compatible.

        Parameters
        ----------
        other : DTScalarOrNaT
            Another scalar value to check compatibility with.

        Raises
        ------
        Exception
            If compatibility checks fail.
        """
        raise AbstractMethodError(self)

    # ------------------------------------------------------------------

    def _box_func(self, x):
        """
        box function to get object from internal representation

        Parameters
        ----------
        x : object
            Object to box.

        Raises
        ------
        AbstractMethodError
            Always raises this error as this method is abstract.
        """
        raise AbstractMethodError(self)

    def _box_values(self, values) -> np.ndarray:
        """
        apply box func to passed values

        Parameters
        ----------
        values : array-like
            Values to apply the box function to.

        Returns
        -------
        np.ndarray
            Resulting array after applying the box function.
        """
        return lib.map_infer(values, self._box_func, convert=False)

    def __iter__(self) -> Iterator:
        """
        Iterate over elements in the object.

        Returns
        -------
        Iterator
            Iterator over elements, either boxed or as integers.
        """
        if self.ndim > 1:
            return (self[n] for n in range(len(self)))
        else:
            return (self._box_func(v) for v in self.asi8)

    @property
    def asi8(self) -> npt.NDArray[np.int64]:
        """
        Integer representation of the values.

        Returns
        -------
        np.ndarray
            Array with int64 dtype representing the values.
        """
        # do not cache or you'll create a memory leak
        return self._ndarray.view("i8")

    # ----------------------------------------------------------------
    # Rendering Methods

    def _format_native_types(
        self, *, na_rep: str | float = "NaT", date_format=None
    ) -> npt.NDArray[np.object_]:
        """
        Helper method for astype when converting to strings.

        Parameters
        ----------
        na_rep : str or float, optional
            Representation for Not-a-Time values (default is "NaT").
        date_format : str or None, optional
            Format for date representation (default is None).

        Returns
        -------
        np.ndarray
            Array of strings representing formatted values.
        """
        raise AbstractMethodError(self)

    def _formatter(self, boxed: bool = False) -> Callable[[object], str]:
        """
        Get a formatting function for objects.

        Parameters
        ----------
        boxed : bool, optional
            Whether to format boxed objects (default is False).

        Returns
        -------
        Callable[[object], str]
            Function that formats an object to a string.
        """
        # TODO: Remove Datetime & DatetimeTZ formatters.
        return "'{}'".format

    # ----------------------------------------------------------------
    # Array-Like / EA-Interface Methods

    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        """
        Convert to numpy array.

        Parameters
        ----------
        dtype : np.dtype or None, optional
            Desired dtype for the array (default is None).
        copy : bool or None, optional
            Whether to copy the array (default is None).

        Returns
        -------
        np.ndarray
            Numpy array representation of the object.
        """
        # used for Timedelta/DatetimeArray, overwritten by PeriodArray
        if is_object_dtype(dtype):
            return np.array(list(self), dtype=object)
        return self._ndarray
    @overload
    def __getitem__(self, key: ScalarIndexer) -> DTScalarOrNaT: ...

此方法是`__getitem__`的重载版本，用于处理标量索引器，并返回`DTScalarOrNaT`类型的数据或`Not a Time`值。


    @overload
    def __getitem__(
        self,
        key: SequenceIndexer | PositionalIndexerTuple,
    ) -> Self: ...

这是`__getitem__`的另一个重载版本，用于处理序列索引器或位置索引器元组，并返回自身类型的数据。


    def __getitem__(self, key: PositionalIndexer2D) -> Self | DTScalarOrNaT:
        """
        This getitem defers to the underlying array, which by-definition can
        only handle list-likes, slices, and integer scalars
        """

此方法是`__getitem__`的实现，用于处理二维位置索引器，并返回自身类型的数据或`DTScalarOrNaT`类型的数据。


        # Use cast as we know we will get back a DatetimeLikeArray or DTScalar,
        # but skip evaluating the Union at runtime for performance
        # (see https://github.com/pandas-dev/pandas/pull/44624)
        result = cast("Union[Self, DTScalarOrNaT]", super().__getitem__(key))

使用类型转换`cast`来明确表明返回的数据类型是`Self`或`DTScalarOrNaT`，以提高性能而跳过运行时联合类型的评估。


        if lib.is_scalar(result):
            return result
        else:
            # At this point we know the result is an array.
            result = cast(Self, result)

如果返回的结果是标量，则直接返回；否则将其转换为`Self`类型，表明返回的是一个数组。


        result._freq = self._get_getitem_freq(key)

设置结果的`_freq`属性，调用`_get_getitem_freq`方法来获取适当的频率属性。


        return result

返回处理后的结果。


    def _get_getitem_freq(self, key) -> BaseOffset | None:
        """
        Find the `freq` attribute to assign to the result of a __getitem__ lookup.
        """

此方法用于找到并返回用于`__getitem__`查找结果的`freq`属性。


        is_period = isinstance(self.dtype, PeriodDtype)
        if is_period:
            freq = self.freq
        elif self.ndim != 1:
            freq = None
        else:
            key = check_array_indexer(self, key)  # maybe ndarray[bool] -> slice
            freq = None
            if isinstance(key, slice):
                if self.freq is not None and key.step is not None:
                    freq = key.step * self.freq
                else:
                    freq = self.freq
            elif key is Ellipsis:
                # GH#21282 indexing with Ellipsis is similar to a full slice,
                #  should preserve `freq` attribute
                freq = self.freq
            elif com.is_bool_indexer(key):
                new_key = lib.maybe_booleans_to_slice(key.view(np.uint8))
                if isinstance(new_key, slice):
                    return self._get_getitem_freq(new_key)

根据不同的条件判断来确定返回的`freq`属性，具体包括对是否为周期(`PeriodDtype`)、维度不为1、是否为切片、是否为省略号(`Ellipsis`)、是否为布尔索引器等的判断。


        return freq

返回最终确定的`freq`属性。


    # error: Argument 1 of "__setitem__" is incompatible with supertype
    # "ExtensionArray"; supertype defines the argument type as "Union[int,
    # ndarray]"
    def __setitem__(
        self,
        key: int | Sequence[int] | Sequence[bool] | slice,
        value: NaTType | Any | Sequence[Any],

`__setitem__`方法用于设置给定索引处的值，但由于参数类型与超类型`ExtensionArray`定义的类型不兼容而出错。
    ) -> None:
        # 这里对类型进行了一些调整。上面的 "Any" 取决于 type(self)。对于 PeriodArray，它是 Period
        # （或者可以从 from_sequence 强制转换为 Period）。对于 DatetimeArray，它是 Timestamp...
        # 我不确定 mypy 是否可以处理这种情况，可能需要使用泛型。
        # 参考：https://mypy.readthedocs.io/en/latest/generics.html

        # 检查设置项的长度是否符合预期，返回一个布尔值表明是否不执行任何操作
        no_op = check_setitem_lengths(key, value, self)

        # 在检查是否为 no_op 之前调用 super()，这意味着即使是 no_op（即不执行任何操作），如果 'value' 不合法，也会触发异常，例如错误的 dtype 空数组。
        super().__setitem__(key, value)

        # 如果是 no_op，直接返回，不进行后续操作
        if no_op:
            return

        # 可能清除频率信息，具体实现视情况而定
        self._maybe_clear_freq()

    def _maybe_clear_freq(self) -> None:
        # 像 __setitem__ 这样的原地操作可能会使 DatetimeArray 和 TimedeltaArray 的频率无效化
        pass
    def astype(self, dtype, copy: bool = True):
        # Some notes on cases we don't have to handle here in the base class:
        #   1. PeriodArray.astype handles period -> period
        #   2. DatetimeArray.astype handles conversion between tz.
        #   3. DatetimeArray.astype handles datetime -> period
        
        # 将 dtype 转换为 pandas 的数据类型对象
        dtype = pandas_dtype(dtype)

        if dtype == object:
            # 如果目标 dtype 是 object 类型
            if self.dtype.kind == "M":
                # 如果当前数组是日期时间类型
                self = cast("DatetimeArray", self)
                # 使用快速的 ints_to_pydatetime 方法将整数数组转换为 Python datetime 对象
                # 这比 self._box_values 方法快得多，例如在 test_get_loc_tuple_monotonic_above_size_cutoff 中
                i8data = self.asi8
                converted = ints_to_pydatetime(
                    i8data,
                    tz=self.tz,
                    box="timestamp",
                    reso=self._creso,
                )
                return converted

            elif self.dtype.kind == "m":
                # 如果当前数组是时间增量类型
                return ints_to_pytimedelta(self._ndarray, box=True)

            # 如果目标 dtype 是 object，但当前数组不是日期时间或时间增量类型，
            # 则使用 self._box_values 方法将整数数组的值封装成对象数组返回
            return self._box_values(self.asi8.ravel()).reshape(self.shape)

        elif isinstance(dtype, ExtensionDtype):
            # 如果目标 dtype 是扩展类型，则调用父类的 astype 方法进行转换
            return super().astype(dtype, copy=copy)
        elif is_string_dtype(dtype):
            # 如果目标 dtype 是字符串类型，则调用 _format_native_types 方法处理
            return self._format_native_types()
        elif dtype.kind in "iu":
            # 如果目标 dtype 是整数类型（无符号或有符号）
            # 我们有意忽略 int32 和 int64 之间的区别。
            # 参见 https://github.com/pandas-dev/pandas/issues/24381 了解更多信息。
            values = self.asi8
            if dtype != np.int64:
                # 如果目标 dtype 不是 int64，则抛出类型错误
                raise TypeError(
                    f"Converting from {self.dtype} to {dtype} is not supported. "
                    "Do obj.astype('int64').astype(dtype) instead"
                )

            if copy:
                # 如果指定了复制操作，则复制整数数组的值
                values = values.copy()
            return values
        elif (dtype.kind in "mM" and self.dtype != dtype) or dtype.kind == "f":
            # 如果目标 dtype 是日期时间或时间增量类型，并且当前数组类型与目标类型不同
            # 或者目标 dtype 是浮点类型，则抛出类型错误
            msg = f"Cannot cast {type(self).__name__} to dtype {dtype}"
            raise TypeError(msg)
        else:
            # 对于其他类型，使用 np.asarray 将当前对象转换为指定的 dtype
            return np.asarray(self, dtype=dtype)

    @overload
    def view(self) -> Self: ...

    @overload
    def view(self, dtype: Literal["M8[ns]"]) -> DatetimeArray: ...

    @overload
    def view(self, dtype: Literal["m8[ns]"]) -> TimedeltaArray: ...

    @overload
    def view(self, dtype: Dtype | None = ...) -> ArrayLike: ...

    def view(self, dtype: Dtype | None = None) -> ArrayLike:
        # 由于此文件中存在 @overload 注解，需要显式调用 super() 方法。
        # 返回父类中的 view 方法处理后的结果
        return super().view(dtype)

    # ------------------------------------------------------------------
    # Validation Methods
    # TODO: try to de-duplicate these, ensure identical behavior
    # 验证与另一个值的比较是否有效，用于支持数据类型的兼容性检查
    def _validate_comparison_value(self, other):
        # 如果other是字符串类型
        if isinstance(other, str):
            try:
                # 尝试将字符串转换为标量值（Timestamp/Timedelta/Period）
                other = self._scalar_from_string(other)
            except (ValueError, IncompatibleFrequency) as err:
                # 如果解析失败，则抛出InvalidComparison异常
                # 例如，无法解析为Timestamp/Timedelta/Period
                raise InvalidComparison(other) from err

        # 如果other是已识别的标量类型或为NaT（Not a Time）
        if isinstance(other, self._recognized_scalars) or other is NaT:
            # 将other转换为与self相同的标量类型
            other = self._scalar_type(other)
            try:
                # 检查self和other之间的兼容性
                self._check_compatible_with(other)
            except (TypeError, IncompatibleFrequency) as err:
                # 如果类型不兼容，则抛出InvalidComparison异常
                # 例如，时区不匹配导致的TypeError
                raise InvalidComparison(other) from err

        # 如果other不是列表或类似列表的对象
        elif not is_list_like(other):
            # 抛出InvalidComparison异常，表示无效的比较对象
            raise InvalidComparison(other)

        # 如果other是列表或类似列表的对象，并且长度不等于self的长度
        elif len(other) != len(self):
            # 抛出ValueError异常，要求长度必须匹配
            raise ValueError("Lengths must match")

        else:
            try:
                # 验证other是否类似列表，并允许其中包含对象
                other = self._validate_listlike(other, allow_object=True)
                # 检查self和other之间的兼容性
                self._check_compatible_with(other)
            except (TypeError, IncompatibleFrequency) as err:
                if is_object_dtype(getattr(other, "dtype", None)):
                    # 如果other是对象类型的数组，需要逐元素操作
                    # 在这种情况下，没有进一步的异常处理
                    pass
                else:
                    # 如果类型不兼容，则抛出InvalidComparison异常
                    raise InvalidComparison(other) from err

        # 返回经过验证和处理后的other值
        return other

    # 验证标量值的有效性，并进行必要的类型转换和兼容性检查
    def _validate_scalar(
        self,
        value,
        *,
        allow_listlike: bool = False,
        unbox: bool = True,
    ):
        """
        Validate that the input value can be cast to our scalar_type.

        Parameters
        ----------
        value : object
            输入的值，可以是任意 Python 对象
        allow_listlike: bool, default False
            是否允许列表类输入。当引发异常时，消息是否应说明允许列表类输入。
        unbox : bool, default True
            是否在返回之前取消包装结果。注意：unbox=False 跳过 setitem 兼容性检查。

        Returns
        -------
        self._scalar_type or NaT
            返回已验证的值，如果无法转换则返回 NaT（Not a Time）对象
        """
        if isinstance(value, self._scalar_type):
            pass

        elif isinstance(value, str):
            # NB: Careful about tzawareness
            # 注意：关于时区感知性的注意事项
            try:
                value = self._scalar_from_string(value)
            except ValueError as err:
                msg = self._validation_error_message(value, allow_listlike)
                raise TypeError(msg) from err

        elif is_valid_na_for_dtype(value, self.dtype):
            # GH#18295
            # 如果 value 是该 dtype 的有效 'na' 值，则设置为 NaT
            value = NaT

        elif isna(value):
            # if we are dt64tz and value is dt64("NaT"), dont cast to NaT,
            #  or else we'll fail to raise in _unbox_scalar
            # 如果我们是 dt64tz 并且 value 是 dt64("NaT")，则不转换为 NaT，
            # 否则将在 _unbox_scalar 中引发错误
            msg = self._validation_error_message(value, allow_listlike)
            raise TypeError(msg)

        elif isinstance(value, self._recognized_scalars):
            # error: Argument 1 to "Timestamp" has incompatible type "object"; expected
            # "integer[Any] | float | str | date | datetime | datetime64"
            # 如果 value 是已识别的标量类型，则将其转换为 self._scalar_type 类型
            value = self._scalar_type(value)  # type: ignore[arg-type]

        else:
            msg = self._validation_error_message(value, allow_listlike)
            raise TypeError(msg)

        if not unbox:
            # NB: In general NDArrayBackedExtensionArray will unbox here;
            #  this option exists to prevent a performance hit in
            #  TimedeltaIndex.get_loc
            # 如果 unbox=False，则返回未取消包装的值
            return value
        # 否则返回取消包装的标量值
        return self._unbox_scalar(value)

    def _validation_error_message(self, value, allow_listlike: bool = False) -> str:
        """
        Construct an exception message on validation error.

        Some methods allow only scalar inputs, while others allow either scalar
        or listlike.

        Parameters
        ----------
        allow_listlike: bool, default False
            是否允许列表类输入

        Returns
        -------
        str
            返回构建的异常消息字符串
        """
        if hasattr(value, "dtype") and getattr(value, "ndim", 0) > 0:
            msg_got = f"{value.dtype} array"
        else:
            msg_got = f"'{type(value).__name__}'"
        if allow_listlike:
            msg = (
                f"value should be a '{self._scalar_type.__name__}', 'NaT', "
                f"or array of those. Got {msg_got} instead."
            )
        else:
            msg = (
                f"value should be a '{self._scalar_type.__name__}' or 'NaT'. "
                f"Got {msg_got} instead."
            )
        return msg
    # 对给定的值进行列表验证，如果是列表对象类型和当前对象类型相同，则进行进一步处理
    def _validate_listlike(self, value, allow_object: bool = False):
        if isinstance(value, type(self)):
            # 如果数据类型包含时间或日期，并且不允许对象，并且单位与当前值不同，则转换单位
            if self.dtype.kind in "mM" and not allow_object and self.unit != value.unit:  # type: ignore[attr-defined]
                # error: "DatetimeLikeArrayMixin" has no attribute "as_unit"
                value = value.as_unit(self.unit, round_ok=False)  # type: ignore[attr-defined]
            return value

        # 如果值是空列表，则返回空序列对象，使用当前对象的数据类型
        if isinstance(value, list) and len(value) == 0:
            # We treat empty list as our own dtype.
            return type(self)._from_sequence([], dtype=self.dtype)

        # 如果值具有 "dtype" 属性，并且其数据类型为对象类型
        if hasattr(value, "dtype") and value.dtype == object:
            # 如果推断出的数据类型匹配当前对象的类型
            if lib.infer_dtype(value) in self._infer_matches:
                try:
                    # 尝试从对象序列创建新的对象
                    value = type(self)._from_sequence(value)
                except (ValueError, TypeError) as err:
                    if allow_object:
                        return value
                    msg = self._validation_error_message(value, True)
                    raise TypeError(msg) from err

        # 在处理 NumpyExtensionArray 后进行类型推断
        # 例如，传递 PeriodIndex.values 并得到 Periods 的 ndarray
        value = extract_array(value, extract_numpy=True)
        value = pd_array(value)
        value = extract_array(value, extract_numpy=True)

        # 如果值全部是字符串，则创建 StringArray
        if is_all_strings(value):
            try:
                # TODO: 如果实现了 from_sequence_of_strings 可能会更好
                # 注意：为 PeriodArray 测试时需要传递数据类型
                value = type(self)._from_sequence(value, dtype=self.dtype)
            except ValueError:
                pass

        # 如果值的数据类型是分类数据类型
        if isinstance(value.dtype, CategoricalDtype):
            # 如果类别的数据类型与当前对象的类型相同
            if value.categories.dtype == self.dtype:
                # TODO: 需要相同的数据类型还是可比较的数据类型？
                value = value._internal_get_values()
                value = extract_array(value, extract_numpy=True)

        # 如果允许对象类型，并且值的数据类型是对象类型
        if allow_object and is_object_dtype(value.dtype):
            pass

        # 如果当前对象的数据类型不是已识别的数据类型，则引发类型错误
        elif not type(self)._is_recognized_dtype(value.dtype):
            msg = self._validation_error_message(value, True)
            raise TypeError(msg)

        # 如果数据类型包含时间或日期，并且不允许对象
        if self.dtype.kind in "mM" and not allow_object:
            # error: "DatetimeLikeArrayMixin" has no attribute "as_unit"
            value = value.as_unit(self.unit, round_ok=False)  # type: ignore[attr-defined]
        return value

    # 对设置的值进行验证，如果是类似列表的结构，则调用 _validate_listlike 进行验证
    def _validate_setitem_value(self, value):
        if is_list_like(value):
            value = self._validate_listlike(value)
        else:
            return self._validate_scalar(value, allow_listlike=True)

        # 返回解封装后的值
        return self._unbox(value)

    # 标记方法为最终方法，不允许派生类重写
    @final
    # 解包函数，用于将输入参数转换为 np.int64、np.datetime64、np.timedelta64 或 np.ndarray 类型
    def _unbox(self, other) -> np.int64 | np.datetime64 | np.timedelta64 | np.ndarray:
        """
        Unbox either a scalar with _unbox_scalar or an instance of our own type.
        """
        # 如果 other 是标量，则使用 _unbox_scalar 方法进行处理
        if lib.is_scalar(other):
            other = self._unbox_scalar(other)
        else:
            # 如果 other 和 self 类型相同，则获取其内部的 ndarray 数据
            # 注意：此处假设 other._ndarray 存在且有效
            self._check_compatible_with(other)
            other = other._ndarray
        return other

    # ------------------------------------------------------------------
    # Additional array methods
    #  These are not part of the EA API, but we implement them because
    #  pandas assumes they're there.

    # 定义 map 方法，用于应对 pandas 假设存在的额外数组方法
    @ravel_compat
    def map(self, mapper, na_action: Literal["ignore"] | None = None):
        # 从 pandas 导入 Index 类
        from pandas import Index

        # 调用 map_array 函数对当前对象执行映射操作，根据 na_action 处理缺失值
        result = map_array(self, mapper, na_action=na_action)
        # 将结果转换为 Index 对象
        result = Index(result)

        # 如果 result 是 ABCMultiIndex 的实例，则转换为其对应的 numpy 数组返回
        if isinstance(result, ABCMultiIndex):
            return result.to_numpy()
        else:
            # 否则返回 result 的 array 属性
            return result.array
    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]:
        """
        Compute boolean array of whether each value is found in the
        passed set of values.

        Parameters
        ----------
        values : np.ndarray or ExtensionArray

        Returns
        -------
        ndarray[bool]
        """
        # 如果传入的值的数据类型是浮点数、整数、无符号整数或复数
        if values.dtype.kind in "fiuc":
            # TODO: de-duplicate with equals, validate_comparison_value
            # 返回一个全为 False 的布尔数组，形状与 self 相同
            return np.zeros(self.shape, dtype=bool)

        # 确保处理日期时间数据时进行适当的封装处理
        values = ensure_wrapped_if_datetimelike(values)

        # 如果传入的值不是与当前对象相同类型的对象
        if not isinstance(values, type(self)):
            # 如果传入值的数据类型为对象类型
            if values.dtype == object:
                # 尝试将对象转换为非数值类型，并返回结果
                values = lib.maybe_convert_objects(
                    values,  # type: ignore[arg-type]
                    convert_non_numeric=True,
                    dtype_if_all_nat=self.dtype,
                )
                # 如果转换后的数据类型不是对象类型，则调用 self.isin(values) 返回结果
                if values.dtype != object:
                    return self.isin(values)
                else:
                    # TODO: Deprecate this case
                    # https://github.com/pandas-dev/pandas/pull/58645/files#r1604055791
                    # 对于仍然是对象类型的数据，调用 isin(self.astype(object), values)
                    return isin(self.astype(object), values)
            # 返回一个全为 False 的布尔数组，形状与 self 相同
            return np.zeros(self.shape, dtype=bool)

        # 如果当前对象的数据类型是日期时间类型
        if self.dtype.kind in "mM":
            # 强制类型转换为 DatetimeArray 或 TimedeltaArray
            self = cast("DatetimeArray | TimedeltaArray", self)
            # 错误："DatetimeLikeArrayMixin" 没有 "as_unit" 属性
            # 将 values 转换为当前对象的单位
            values = values.as_unit(self.unit)  # type: ignore[attr-defined]

        try:
            # 错误："DatetimeLikeArrayMixin" 的 "_check_compatible_with" 方法的第一个参数
            # 类型为 "ExtensionArray | ndarray[Any, Any]"，预期为 "Period | Timestamp | Timedelta | NaTType"
            # 检查当前对象与传入值是否兼容
            self._check_compatible_with(values)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            # 包括时区不匹配错误和不兼容频率错误
            # 返回一个全为 False 的布尔数组，形状与 self 相同
            return np.zeros(self.shape, dtype=bool)

        # 错误："ExtensionArray | ndarray[Any, Any]" 中的项 "ExtensionArray" 没有 "asi8" 属性
        # 调用 isin(self.asi8, values.asi8) 返回结果
        return isin(self.asi8, values.asi8)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Null Handling

    def isna(self) -> npt.NDArray[np.bool_]:
        """
        Return a boolean array indicating whether each value is NaN.
        """
        return self._isnan

    @property  # NB: override with cache_readonly in immutable subclasses
    def _isnan(self) -> npt.NDArray[np.bool_]:
        """
        Return a boolean array indicating whether each value is NaN.
        """
        # 返回一个布尔数组，指示每个值是否为 NaN
        return self.asi8 == iNaT

    @property  # NB: override with cache_readonly in immutable subclasses
    def _hasna(self) -> bool:
        """
        Return if I have any NaNs; enables various performance speedups.
        """
        # 返回是否存在任何 NaN 值，用于启用各种性能优化
        return bool(self._isnan.any())

    def _maybe_mask_results(
        self, result: np.ndarray, fill_value=iNaT, convert=None
    ) -> np.ndarray:
        """
        Parameters
        ----------
        result : np.ndarray
            输入参数，一个 NumPy 数组

        fill_value : object, default iNaT
            可选参数，填充值，默认为 iNaT（不是一个时间）

        convert : str, dtype or None
            可选参数，要转换的数据类型或 None

        Returns
        -------
        result : ndarray with values replace by the fill_value
            替换后的 NumPy 数组

        Notes
        -----
        mask the result if needed, convert to the provided dtype if its not
        None
            如果需要，掩码结果，并根据提供的数据类型进行转换，如果不为 None

        This is an internal routine.
        """
        if self._hasna:
            if convert:
                result = result.astype(convert)
                # 如果 convert 不为 None，则将 result 转换为指定的数据类型
            if fill_value is None:
                fill_value = np.nan
                # 如果 fill_value 为 None，则设置 fill_value 为 NaN
            np.putmask(result, self._isnan, fill_value)
            # 使用 np.putmask 根据 _isnan 掩码数组将 fill_value 填充到 result 中
        return result
        # 返回填充后的结果数组

    # ------------------------------------------------------------------
    # Frequency Properties/Methods

    @property
    def freqstr(self) -> str | None:
        """
        Return the frequency object as a string if it's set, otherwise None.

        Returns
        -------
        str | None
            返回频率对象的字符串表示，如果未设置则返回 None。

        See Also
        --------
        DatetimeIndex.inferred_freq : Returns a string representing a frequency
            generated by infer_freq.

        Examples
        --------
        For DatetimeIndex:

        >>> idx = pd.DatetimeIndex(["1/1/2020 10:00:00+00:00"], freq="D")
        >>> idx.freqstr
        'D'

        The frequency can be inferred if there are more than 2 points:

        >>> idx = pd.DatetimeIndex(
        ...     ["2018-01-01", "2018-01-03", "2018-01-05"], freq="infer"
        ... )
        >>> idx.freqstr
        '2D'

        For PeriodIndex:

        >>> idx = pd.PeriodIndex(["2023-1", "2023-2", "2023-3"], freq="M")
        >>> idx.freqstr
        'M'
        """
        if self.freq is None:
            return None
            # 如果频率为 None，则返回 None
        return self.freq.freqstr
        # 返回频率对象的字符串表示

    @property  # NB: override with cache_readonly in immutable subclasses
    def inferred_freq(self) -> str | None:
        """
        Tries to return a string representing a frequency generated by infer_freq.

        Returns
        -------
        str | None
            返回由 infer_freq 生成的频率的字符串表示，如果无法自动检测到频率则返回 None。

        See Also
        --------
        DatetimeIndex.freqstr : Return the frequency object as a string if it's set,
            otherwise None.

        Examples
        --------
        For DatetimeIndex:

        >>> idx = pd.DatetimeIndex(["2018-01-01", "2018-01-03", "2018-01-05"])
        >>> idx.inferred_freq
        '2D'

        For TimedeltaIndex:

        >>> tdelta_idx = pd.to_timedelta(["0 days", "10 days", "20 days"])
        >>> tdelta_idx
        TimedeltaIndex(['0 days', '10 days', '20 days'],
                       dtype='timedelta64[ns]', freq=None)
        >>> tdelta_idx.inferred_freq
        '10D'
        """
        if self.ndim != 1:
            return None
            # 如果数组维度不为 1，则返回 None
        try:
            return frequencies.infer_freq(self)
            # 尝试使用 frequencies 模块推断频率并返回结果
        except ValueError:
            return None
            # 如果出现 ValueError 异常，则返回 None
    # 返回分辨率对象或者None，根据当前对象的频率字符串
    def _resolution_obj(self) -> Resolution | None:
        # 获取频率字符串
        freqstr = self.freqstr
        # 如果频率字符串为None，返回None
        if freqstr is None:
            return None
        try:
            # 根据频率字符串获取分辨率对象
            return Resolution.get_reso_from_freqstr(freqstr)
        except KeyError:
            # 如果发生KeyError异常，返回None
            return None

    @property  # NB: override with cache_readonly in immutable subclasses
    # 返回字符串，表示对象的分辨率（天、小时、分钟、秒、毫秒或微秒）
    def resolution(self) -> str:
        """
        Returns day, hour, minute, second, millisecond or microsecond
        """
        # error: Item "None" of "Optional[Any]" has no attribute "attrname"
        # 返回分辨率对象的属性值attrname，类型标注忽略union-attr错误
        return self._resolution_obj.attrname  # type: ignore[union-attr]

    # monotonicity/uniqueness properties are called via frequencies.infer_freq,
    #  see GH#23789

    @property
    # 返回布尔值，指示对象是否单调递增
    def _is_monotonic_increasing(self) -> bool:
        # 使用算法判断对象的asi8属性是否单调递增
        return algos.is_monotonic(self.asi8, timelike=True)[0]

    @property
    # 返回布尔值，指示对象是否单调递减
    def _is_monotonic_decreasing(self) -> bool:
        # 使用算法判断对象的asi8属性是否单调递减
        return algos.is_monotonic(self.asi8, timelike=True)[1]

    @property
    # 返回布尔值，指示对象的值是否唯一
    def _is_unique(self) -> bool:
        # 将asi8属性展平为一维数组，检查其唯一性
        return len(unique1d(self.asi8.ravel("K"))) == self.size

    # ------------------------------------------------------------------
    # Arithmetic Methods
    # 定义私有方法 `_cmp_method`，用于比较操作
    def _cmp_method(self, other, op):
        # 如果 self 的维度大于 1 并且 other 有 `shape` 属性且与 self 的形状相同
        if self.ndim > 1 and getattr(other, "shape", None) == self.shape:
            # TODO: 处理类似于二维的 listlikes
            return op(self.ravel(), other.ravel()).reshape(self.shape)

        try:
            # 验证比较值 other 是否有效
            other = self._validate_comparison_value(other)
        except InvalidComparison:
            # 如果比较无效，则返回无效比较的结果
            return invalid_comparison(self, other, op)

        # 获取 other 的 dtype
        dtype = getattr(other, "dtype", None)
        # 如果 other 的 dtype 是对象类型
        if is_object_dtype(dtype):
            # 使用 comp_method_OBJECT_ARRAY 处理，而不是使用 numpy 的比较，否则在与 None 比较时会引发异常
            result = ops.comp_method_OBJECT_ARRAY(
                op, np.asarray(self.astype(object)), other
            )
            return result
        
        # 如果 other 是 NaT（Not a Time），处理不同的比较操作
        if other is NaT:
            if op is operator.ne:
                result = np.ones(self.shape, dtype=bool)
            else:
                result = np.zeros(self.shape, dtype=bool)
            return result
        
        # 如果 self 的 dtype 不是 PeriodDtype
        if not isinstance(self.dtype, PeriodDtype):
            # 将 self 强制转换为 TimelikeOps 类型
            self = cast(TimelikeOps, self)
            # 如果 self 和 other 的 _creso 属性不同
            if self._creso != other._creso:
                # 如果 other 不是 self 的实例
                if not isinstance(other, type(self)):
                    # 尝试将 other 转换为与 self 单位相同的单位
                    try:
                        other = other.as_unit(self.unit, round_ok=False)
                    except ValueError:
                        # 将 other 转换为 ndarray，并由 compare_mismatched_resolutions 处理
                        other_arr = np.array(other.asm8)
                        return compare_mismatched_resolutions(
                            self._ndarray, other_arr, op
                        )
                else:
                    # 如果 other 是 self 的实例，则使用 other 的 _ndarray 属性
                    other_arr = other._ndarray
                    return compare_mismatched_resolutions(self._ndarray, other_arr, op)

        # 获取 self 与 other 的非装箱值
        other_vals = self._unbox(other)
        # 使用 op 操作符对 self 的 _ndarray.view("i8") 和 other_vals.view("i8") 进行比较
        result = op(self._ndarray.view("i8"), other_vals.view("i8"))

        # 判断 other 是否为缺失值（NaN），生成对应的掩码
        o_mask = isna(other)
        mask = self._isnan | o_mask
        # 如果掩码中有 True 值，则使用 op is operator.ne 的结果替换 result 中的对应位置
        if mask.any():
            nat_result = op is operator.ne
            np.putmask(result, mask, nat_result)

        # 返回比较的结果
        return result

    # 定义魔术方法 __pow__ 到 __rmod__ 的无效操作，所有三个子类的 pow 操作均无效；TimedeltaArray 会覆盖乘法和除法操作
    __pow__ = _make_unpacked_invalid_op("__pow__")
    __rpow__ = _make_unpacked_invalid_op("__rpow__")
    __mul__ = _make_unpacked_invalid_op("__mul__")
    __rmul__ = _make_unpacked_invalid_op("__rmul__")
    __truediv__ = _make_unpacked_invalid_op("__truediv__")
    __rtruediv__ = _make_unpacked_invalid_op("__rtruediv__")
    __floordiv__ = _make_unpacked_invalid_op("__floordiv__")
    __rfloordiv__ = _make_unpacked_invalid_op("__rfloordiv__")
    __mod__ = _make_unpacked_invalid_op("__mod__")
    __rmod__ = _make_unpacked_invalid_op("__rmod__")
    # 设置特殊方法 __divmod__ 为无效操作，不支持解包
    __divmod__ = _make_unpacked_invalid_op("__divmod__")
    # 设置特殊方法 __rdivmod__ 为无效操作，不支持右向解包
    __rdivmod__ = _make_unpacked_invalid_op("__rdivmod__")

    @final
    def _get_i8_values_and_mask(
        self, other
    ) -> tuple[int | npt.NDArray[np.int64], None | npt.NDArray[np.bool_]]:
        """
        Get the int64 values and b_mask to pass to add_overflowsafe.
        """
        if isinstance(other, Period):
            # 如果 other 是 Period 类型，取其 ordinal 属性作为 i8values
            i8values = other.ordinal
            mask = None
        elif isinstance(other, (Timestamp, Timedelta)):
            # 如果 other 是 Timestamp 或 Timedelta 类型，取其 _value 属性作为 i8values
            i8values = other._value
            mask = None
        else:
            # 对于 PeriodArray, DatetimeArray, TimedeltaArray 等类型，使用 _isnan 和 asi8 属性获取 mask 和 i8values
            mask = other._isnan
            i8values = other.asi8
        return i8values, mask

    @final
    def _get_arithmetic_result_freq(self, other) -> BaseOffset | None:
        """
        Check if we can preserve self.freq in addition or subtraction.
        """
        # 如果 self 的 dtype 是 PeriodDtype 类型，返回 self.freq
        if isinstance(self.dtype, PeriodDtype):
            return self.freq
        # 如果 other 不是标量，则返回 None
        elif not lib.is_scalar(other):
            return None
        # 如果 self.freq 是 Tick 类型，则返回 self.freq
        elif isinstance(self.freq, Tick):
            return self.freq
        # 其他情况返回 None
        return None

    @final
    def _add_datetimelike_scalar(self, other) -> DatetimeArray:
        if not lib.is_np_dtype(self.dtype, "m"):
            # 如果 self 的 dtype 不是 'm' 类型（numpy 中的时间类型），抛出 TypeError
            raise TypeError(
                f"cannot add {type(self).__name__} and {type(other).__name__}"
            )

        self = cast("TimedeltaArray", self)

        from pandas.core.arrays import DatetimeArray
        from pandas.core.arrays.datetimes import tz_to_dtype

        assert other is not NaT
        if isna(other):
            # 如果 other 是 NaT，返回相应的 DatetimeArray 类型
            result = self._ndarray + NaT.to_datetime64().astype(f"M8[{self.unit}]")
            # 保留结果的分辨率
            return DatetimeArray._simple_new(result, dtype=result.dtype)

        other = Timestamp(other)
        self, other = self._ensure_matching_resos(other)
        self = cast("TimedeltaArray", self)

        # 获取 self 和 other 的 int64 值和掩码
        other_i8, o_mask = self._get_i8_values_and_mask(other)
        # 使用 add_overflowsafe 函数进行安全加法操作，并获取结果
        result = add_overflowsafe(self.asi8, np.asarray(other_i8, dtype="i8"))
        # 将结果转换为指定单位的时间值
        res_values = result.view(f"M8[{self.unit}]")

        # 根据 other 的时区信息获取 dtype
        dtype = tz_to_dtype(tz=other.tz, unit=self.unit)
        res_values = result.view(f"M8[{self.unit}]")
        # 获取算术结果的频率信息
        new_freq = self._get_arithmetic_result_freq(other)
        # 返回新的 DatetimeArray 对象
        return DatetimeArray._simple_new(res_values, dtype=dtype, freq=new_freq)

    @final
    # 将另一个 DatetimeArray 添加到当前 DatetimeArray，返回结果 DatetimeArray
    def _add_datetime_arraylike(self, other: DatetimeArray) -> DatetimeArray:
        # 检查当前数组的 dtype 是否为日期时间类型，否则抛出类型错误异常
        if not lib.is_np_dtype(self.dtype, "m"):
            raise TypeError(
                f"cannot add {type(self).__name__} and {type(other).__name__}"
            )

        # 委托给 DatetimeArray.__add__ 处理
        return other + self

    @final
    # 从 DatetimeArray 中减去单个日期时间或 np.datetime64 标量，返回 TimedeltaArray
    def _sub_datetimelike_scalar(
        self, other: datetime | np.datetime64
    ) -> TimedeltaArray:
        # 检查当前数组的 dtype 是否为日期时间类型，否则抛出类型错误异常
        if self.dtype.kind != "M":
            raise TypeError(f"cannot subtract a datelike from a {type(self).__name__}")

        self = cast("DatetimeArray", self)
        # 从当前 DatetimeArray 减去一个日期时间，返回一个 ndarray[timedelta64[ns]]

        if isna(other):
            # 例如 np.datetime64("NaT")
            return self - NaT

        ts = Timestamp(other)

        # 确保自身和 ts 具有匹配的分辨率
        self, ts = self._ensure_matching_resos(ts)
        return self._sub_datetimelike(ts)

    @final
    # 从 DatetimeArray 中减去另一个 DatetimeArray，返回 TimedeltaArray
    def _sub_datetime_arraylike(self, other: DatetimeArray) -> TimedeltaArray:
        # 检查当前数组的 dtype 是否为日期时间类型，否则抛出类型错误异常
        if self.dtype.kind != "M":
            raise TypeError(f"cannot subtract a datelike from a {type(self).__name__}")

        # 检查两个数组长度是否相同，若不同则抛出值错误异常
        if len(self) != len(other):
            raise ValueError("cannot add indices of unequal length")

        self = cast("DatetimeArray", self)

        # 确保自身和 other 具有匹配的分辨率
        self, other = self._ensure_matching_resos(other)
        return self._sub_datetimelike(other)

    @final
    # 从 DatetimeArray 中减去 Timestamp 或 DatetimeArray，返回 TimedeltaArray
    def _sub_datetimelike(self, other: Timestamp | DatetimeArray) -> TimedeltaArray:
        self = cast("DatetimeArray", self)

        from pandas.core.arrays import TimedeltaArray

        try:
            # 检查时区兼容性，若不兼容则抛出类型错误异常
            self._assert_tzawareness_compat(other)
        except TypeError as err:
            # 修改异常消息，将 "compare" 替换为 "subtract"
            new_message = str(err).replace("compare", "subtract")
            raise type(err)(new_message) from err

        # 获取 other 的整数值和掩码
        other_i8, o_mask = self._get_i8_values_and_mask(other)
        # 使用 add_overflowsafe 函数处理加法，返回 timedelta64[ns] 类型的 ndarray
        res_values = add_overflowsafe(self.asi8, np.asarray(-other_i8, dtype="i8"))
        res_m8 = res_values.view(f"timedelta64[{self.unit}]")

        # 获取计算后的频率
        new_freq = self._get_arithmetic_result_freq(other)
        new_freq = cast("Tick | None", new_freq)
        # 使用 TimedeltaArray._simple_new 创建新的 TimedeltaArray 对象并返回
        return TimedeltaArray._simple_new(res_m8, dtype=res_m8.dtype, freq=new_freq)

    @final
    # 将 Period 添加到当前 DatetimeArray，返回 PeriodArray
    def _add_period(self, other: Period) -> PeriodArray:
        # 检查当前数组的 dtype 是否为日期时间类型，否则抛出类型错误异常
        if not lib.is_np_dtype(self.dtype, "m"):
            raise TypeError(f"cannot add Period to a {type(self).__name__}")

        # 将其包装在 PeriodArray 中，委托于反向操作
        from pandas.core.arrays.period import PeriodArray

        i8vals = np.broadcast_to(other.ordinal, self.shape)
        dtype = PeriodDtype(other.freq)
        parr = PeriodArray(i8vals, dtype=dtype)
        return parr + self

    # 抽象方法，表示需要在子类中实现
    def _add_offset(self, offset):
        raise AbstractMethodError(self)
    @final
    def _add_timedeltalike_scalar(self, other):
        """
        Add a delta of a timedeltalike
        
        Returns
        -------
        Same type as self
        """
        if isna(other):
            # 如果 other 是 NA（Not Available），即 np.timedelta64("NaT")
            # 创建一个空的数组，填充 iNaT（即不可用的 timedelta 值），并返回相同类型的新对象
            new_values = np.empty(self.shape, dtype="i8").view(self._ndarray.dtype)
            new_values.fill(iNaT)
            return type(self)._simple_new(new_values, dtype=self.dtype)
        
        # PeriodArray 已经覆盖，所以这里只处理 DatetimeArray 或 TimedeltaArray
        self = cast("DatetimeArray | TimedeltaArray", self)
        other = Timedelta(other)
        # 确保 self 和 other 具有相同的分辨率（resolution）
        self, other = self._ensure_matching_resos(other)
        # 调用 _add_timedeltalike 方法执行加法操作
        return self._add_timedeltalike(other)

    def _add_timedelta_arraylike(self, other: TimedeltaArray) -> Self:
        """
        Add a delta of a TimedeltaIndex
        
        Returns
        -------
        Same type as self
        """
        # 被 PeriodArray 覆盖
        
        # 如果 self 和 other 的长度不相等，则抛出 ValueError
        if len(self) != len(other):
            raise ValueError("cannot add indices of unequal length")
        
        # 确保 self 和 other 具有相同的分辨率（resolution）
        self, other = cast(
            "DatetimeArray | TimedeltaArray", self
        )._ensure_matching_resos(other)
        # 调用 _add_timedeltalike 方法执行加法操作
        return self._add_timedeltalike(other)

    @final
    def _add_timedeltalike(self, other: Timedelta | TimedeltaArray) -> Self:
        """
        Add a delta of a Timedelta or TimedeltaArray
        
        Returns
        -------
        Same type as self
        """
        # 获取 other 的 i8 值和掩码
        other_i8, o_mask = self._get_i8_values_and_mask(other)
        # 使用 add_overflowsafe 函数将 self.asi8 和 other_i8 相加，得到新的值数组
        new_values = add_overflowsafe(self.asi8, np.asarray(other_i8, dtype="i8"))
        # 将新的值数组转换为与 self._ndarray 相同的数据类型
        res_values = new_values.view(self._ndarray.dtype)
        
        # 获取计算后的新频率
        new_freq = self._get_arithmetic_result_freq(other)
        
        # 调用 _simple_new 方法创建一个新对象，传入结果值、数据类型和新频率
        # error: "_simple_new" 的 "NDArrayBacked" 上不期望的关键字参数 "freq"
        return type(self)._simple_new(
            res_values,
            dtype=self.dtype,
            freq=new_freq,  # type: ignore[call-arg]
        )

    @final
    def _add_nat(self) -> Self:
        """
        Add pd.NaT to self
        """
        # 如果 self 的 dtype 是 PeriodDtype 类型，则抛出 TypeError
        if isinstance(self.dtype, PeriodDtype):
            raise TypeError(
                f"Cannot add {type(self).__name__} and {type(NaT).__name__}"
            )
        
        # GH#19124 中指出，pd.NaT 在 timedelta 和 datetime 类型中被视为 timedelta
        # 创建一个由 iNaT 填充的 int64 类型的空数组
        result = np.empty(self.shape, dtype=np.int64)
        result.fill(iNaT)
        # 将结果数组转换为与 self._ndarray 相同的数据类型，以保留分辨率
        result = result.view(self._ndarray.dtype)
        # error: "_simple_new" 的 "NDArrayBacked" 上不期望的关键字参数 "freq"
        return type(self)._simple_new(
            result,
            dtype=self.dtype,
            freq=None,  # type: ignore[call-arg]
        )
    def _sub_nat(self) -> np.ndarray:
        """
        Subtract pd.NaT from self
        """
        # GH#19124 Timedelta - datetime is not in general well-defined.
        # We make an exception for pd.NaT, which in this case quacks
        # like a timedelta.
        # For datetime64 dtypes by convention we treat NaT as a datetime, so
        # this subtraction returns a timedelta64 dtype.
        # For period dtype, timedelta64 is a close-enough return dtype.
        
        # 创建一个空的 numpy 数组，用于存储结果，dtype 为 np.int64
        result = np.empty(self.shape, dtype=np.int64)
        result.fill(iNaT)
        
        # 如果 self 的 dtype 是 'm'（datetime64）或 'M'（timedelta64）
        if self.dtype.kind in "mM":
            # 将 self 强制转换为 DatetimeArray 或 TimedeltaArray 类型
            self = cast("DatetimeArray|TimedeltaArray", self)
            # 返回一个视图，其 dtype 为 timedelta64，并保持单位与 self 相同
            return result.view(f"timedelta64[{self.unit}]")
        else:
            # 返回一个视图，其 dtype 固定为 timedelta64[ns]
            return result.view("timedelta64[ns]")

    @final
    def _sub_periodlike(self, other: Period | PeriodArray) -> npt.NDArray[np.object_]:
        # 如果操作定义良好，则返回一个对象类型的 ndarray，其中填充了 DateOffsets。空条目填充为 pd.NaT
        
        # 如果 self 的 dtype 不是 PeriodDtype，则引发 TypeError
        if not isinstance(self.dtype, PeriodDtype):
            raise TypeError(
                f"cannot subtract {type(other).__name__} from {type(self).__name__}"
            )

        # 将 self 强制转换为 PeriodArray 类型
        self = cast("PeriodArray", self)
        # 检查 self 和 other 是否兼容
        self._check_compatible_with(other)

        # 获取 other 的 i8 值和掩码
        other_i8, o_mask = self._get_i8_values_and_mask(other)
        # 计算新的 i8 数据，处理溢出安全性
        new_i8_data = add_overflowsafe(self.asi8, np.asarray(-other_i8, dtype="i8"))
        # 根据 self 的频率计算新的数据
        new_data = np.array([self.freq.base * x for x in new_i8_data])

        # 如果 o_mask 为 None，则说明 other 是 Period 标量，使用 self 的 _isnan 作为掩码
        if o_mask is None:
            mask = self._isnan
        else:
            # 否则，other 是 PeriodArray，使用 self 的 _isnan 或 o_mask 作为掩码
            mask = self._isnan | o_mask
        # 将新数据中的掩码位置填充为 NaT
        new_data[mask] = NaT
        
        # 返回新数据作为对象类型的 ndarray
        return new_data

    @final
    # 定义一个方法，用于处理添加或减去 DateOffset 对象数组
    def _addsub_object_array(self, other: npt.NDArray[np.object_], op) -> np.ndarray:
        """
        Add or subtract array-like of DateOffset objects
        
        Parameters
        ----------
        other : np.ndarray[object]
            另一个包含 DateOffset 对象的数组
        op : {operator.add, operator.sub}
            操作符，可以是加法或减法函数

        Returns
        -------
        np.ndarray[object]
            返回一个 np.ndarray[object]，特殊情况下长度为1时操作单个标量
        """
        # 断言操作符为加法或减法
        assert op in [operator.add, operator.sub]

        # 如果 other 的长度为1且 self 是一维数组
        if len(other) == 1 and self.ndim == 1:
            # 注意：在这种特殊情况下，我们可以标注返回类型为 ndarray[object]
            # 如果两者都是一维数组，则广播是明确的
            return op(self, other[0])

        # 如果启用了性能警告选项
        if get_option("performance_warnings"):
            # 发出警告，说明正在对 object 类型的数组进行非向量化的加法/减法操作
            warnings.warn(
                "Adding/subtracting object-dtype array to "
                f"{type(self).__name__} not vectorized.",
                PerformanceWarning,
                stacklevel=find_stack_level(),
            )

        # 断言调用者负责在必要时进行广播
        assert self.shape == other.shape, (self.shape, other.shape)

        # 执行操作，并返回结果值数组
        res_values = op(self.astype("O"), np.asarray(other))
        return res_values

    # 积累方法，根据给定的名称累积计算
    def _accumulate(self, name: str, *, skipna: bool = True, **kwargs) -> Self:
        """
        Accumulate computation based on the specified name.
        
        Parameters
        ----------
        name : str
            累积计算的名称，支持 "cummin" 或 "cummax"
        skipna : bool, optional, default True
            是否跳过 NaN 值
        **kwargs
            其他传递给累积操作的参数

        Returns
        -------
        Self
            返回累积计算后的结果对象
        """
        # 如果名称不在支持的集合中
        if name not in {"cummin", "cummax"}:
            raise TypeError(f"Accumulation {name} not supported for {type(self)}")

        # 根据名称获取对应的累积操作函数
        op = getattr(datetimelike_accumulations, name)
        
        # 执行累积操作，得到结果
        result = op(self.copy(), skipna=skipna, **kwargs)

        # 返回与 self 相同类型的新对象，使用结果进行简单创建
        return type(self)._simple_new(result, dtype=self.dtype)

    # 解压零维并延迟 "__add__" 操作
    # 实现加法运算符的特殊方法，用于处理对象与另一个对象相加的情况
    def __add__(self, other):
        # 获取对象 other 的 dtype 属性，如果没有则为 None
        other_dtype = getattr(other, "dtype", None)
        # 确保将日期时间类对象包装为适当的类型
        other = ensure_wrapped_if_datetimelike(other)

        # 处理标量情况
        if other is NaT:
            # 如果 other 是 NaT（Not a Time），调用 self._add_nat() 处理
            result: np.ndarray | DatetimeLikeArrayMixin = self._add_nat()
        elif isinstance(other, (Tick, timedelta, np.timedelta64)):
            # 如果 other 是 Tick 对象、timedelta 对象或 np.timedelta64 类型，调用 self._add_timedeltalike_scalar(other) 处理
            result = self._add_timedeltalike_scalar(other)
        elif isinstance(other, BaseOffset):
            # 如果 other 是 BaseOffset 的子类对象，调用 self._add_offset(other) 处理
            result = self._add_offset(other)
        elif isinstance(other, (datetime, np.datetime64)):
            # 如果 other 是 datetime 对象或 np.datetime64 类型，调用 self._add_datetimelike_scalar(other) 处理
            result = self._add_datetimelike_scalar(other)
        elif isinstance(other, Period) and lib.is_np_dtype(self.dtype, "m"):
            # 如果 other 是 Period 对象且 self 的 dtype 是 "m" 类型，调用 self._add_period(other) 处理
            result = self._add_period(other)
        elif lib.is_integer(other):
            # 如果 other 是整数类型，进一步判断 self 的 dtype 是否是 PeriodDtype 类型，
            # 如果不是则抛出 integer_op_not_supported 异常，否则执行 obj._addsub_int_array_or_scalar 的加法操作
            if not isinstance(self.dtype, PeriodDtype):
                raise integer_op_not_supported(self)
            obj = cast("PeriodArray", self)
            result = obj._addsub_int_array_or_scalar(other * obj.dtype._n, operator.add)

        # 处理数组类对象情况
        elif lib.is_np_dtype(other_dtype, "m"):
            # 如果 other 的 dtype 是 "m" 类型（np.datetime64 或 np.timedelta64），调用 self._add_timedelta_arraylike(other) 处理
            result = self._add_timedelta_arraylike(other)
        elif is_object_dtype(other_dtype):
            # 如果 other 的 dtype 是 object 类型，调用 self._addsub_object_array(other, operator.add) 处理
            result = self._addsub_object_array(other, operator.add)
        elif lib.is_np_dtype(other_dtype, "M") or isinstance(other_dtype, DatetimeTZDtype):
            # 如果 other 的 dtype 是 "M" 类型（np.datetime64 或 DatetimeTZDtype 类型），调用 self._add_datetime_arraylike(other) 处理
            return self._add_datetime_arraylike(other)
        elif is_integer_dtype(other_dtype):
            # 如果 other 的 dtype 是整数类型，进一步判断 self 的 dtype 是否是 PeriodDtype 类型，
            # 如果不是则抛出 integer_op_not_supported 异常，否则执行 obj._addsub_int_array_or_scalar 的加法操作
            if not isinstance(self.dtype, PeriodDtype):
                raise integer_op_not_supported(self)
            obj = cast("PeriodArray", self)
            result = obj._addsub_int_array_or_scalar(other * obj.dtype._n, operator.add)
        else:
            # 处理其他类型情况，返回 NotImplemented 表示不支持此操作
            return NotImplemented

        # 如果 result 是 np.ndarray 类型且其 dtype 是 "m" 类型，使用 TimedeltaArray 的 _from_sequence 方法封装成 TimedeltaArray 类型
        if isinstance(result, np.ndarray) and lib.is_np_dtype(result.dtype, "m"):
            from pandas.core.arrays import TimedeltaArray
            return TimedeltaArray._from_sequence(result)
        # 返回计算结果
        return result

    # 右向加法的别名方法，实际调用 __add__ 方法处理
    def __radd__(self, other):
        return self.__add__(other)

    # 装饰器方法，用于延迟调用 __sub__ 方法的解包和零维数组处理
    @unpack_zerodim_and_defer("__sub__")
    def __sub__(self, other):
        # 获取对象 other 的 dtype 属性，如果不存在则为 None
        other_dtype = getattr(other, "dtype", None)
        # 确保将日期时间类对象进行包装处理，以确保操作的一致性
        other = ensure_wrapped_if_datetimelike(other)

        # 处理标量类型的 other
        if other is NaT:
            # 如果 other 是 NaT（Not a Time），执行 self 对象的 _sub_nat 方法
            result: np.ndarray | DatetimeLikeArrayMixin = self._sub_nat()
        elif isinstance(other, (Tick, timedelta, np.timedelta64)):
            # 如果 other 是 Tick 类型、timedelta 类型或 np.timedelta64 类型，
            # 执行 self 对象的 _add_timedeltalike_scalar 方法，传入负数的 other
            result = self._add_timedeltalike_scalar(-other)
        elif isinstance(other, BaseOffset):
            # 如果 other 是 BaseOffset 类型的对象，执行 self 对象的 _add_offset 方法，传入负数的 other
            # 特别指出，这里 other 不是 Tick 类型
            result = self._add_offset(-other)
        elif isinstance(other, (datetime, np.datetime64)):
            # 如果 other 是 datetime 或 np.datetime64 类型，
            # 执行 self 对象的 _sub_datetimelike_scalar 方法，传入 other
            result = self._sub_datetimelike_scalar(other)
        elif lib.is_integer(other):
            # 如果 other 是整数类型
            # 注意：这个检查必须放在 np.timedelta64 检查之后，因为 is_integer 对 np.timedelta64 也返回 True
            if not isinstance(self.dtype, PeriodDtype):
                # 如果 self 的 dtype 不是 PeriodDtype，则抛出 integer_op_not_supported 异常
                raise integer_op_not_supported(self)
            # 将 self 强制转换为 PeriodArray 类型的对象
            obj = cast("PeriodArray", self)
            # 执行 obj 的 _addsub_int_array_or_scalar 方法，传入 other 乘以 obj.dtype._n 的结果和 operator.sub 函数
            result = obj._addsub_int_array_or_scalar(other * obj.dtype._n, operator.sub)

        elif isinstance(other, Period):
            # 如果 other 是 Period 类型的对象，
            # 执行 self 对象的 _sub_periodlike 方法，传入 other
            result = self._sub_periodlike(other)

        # 处理类数组类型的 other
        elif lib.is_np_dtype(other_dtype, "m"):
            # 如果 other 的 dtype 是 timedelta64 类型的，
            # 执行 self 对象的 _add_timedelta_arraylike 方法，传入负数的 other
            result = self._add_timedelta_arraylike(-other)
        elif is_object_dtype(other_dtype):
            # 如果 other 的 dtype 是对象类型（例如 DateOffset 对象的数组或索引），
            # 执行 self 对象的 _addsub_object_array 方法，传入 other 和 operator.sub 函数
            result = self._addsub_object_array(other, operator.sub)
        elif lib.is_np_dtype(other_dtype, "M") or isinstance(
            other_dtype, DatetimeTZDtype
        ):
            # 如果 other 的 dtype 是 datetime64 或 DatetimeTZDtype 类型，
            # 执行 self 对象的 _sub_datetime_arraylike 方法，传入 other
            result = self._sub_datetime_arraylike(other)
        elif isinstance(other_dtype, PeriodDtype):
            # 如果 other 的 dtype 是 PeriodDtype 类型，
            # 执行 self 对象的 _sub_periodlike 方法，传入 other
            result = self._sub_periodlike(other)
        elif is_integer_dtype(other_dtype):
            # 如果 other 的 dtype 是整数类型
            if not isinstance(self.dtype, PeriodDtype):
                # 如果 self 的 dtype 不是 PeriodDtype，则抛出 integer_op_not_supported 异常
                raise integer_op_not_supported(self)
            # 将 self 强制转换为 PeriodArray 类型的对象
            obj = cast("PeriodArray", self)
            # 执行 obj 的 _addsub_int_array_or_scalar 方法，传入 other 和 operator.sub 函数
            result = obj._addsub_int_array_or_scalar(other * obj.dtype._n, operator.sub)
        else:
            # 包括 ExtensionArrays、float 类型的其他情况，返回 NotImplemented
            return NotImplemented

        # 如果结果 result 是 np.ndarray 类型，并且其 dtype 是 timedelta64 类型
        if isinstance(result, np.ndarray) and lib.is_np_dtype(result.dtype, "m"):
            # 导入 TimedeltaArray 类并使用 _from_sequence 方法从 result 创建 TimedeltaArray 对象
            from pandas.core.arrays import TimedeltaArray
            return TimedeltaArray._from_sequence(result)

        # 返回计算结果 result
        return result
    # 实现右侧减法运算符的特殊方法 __rsub__
    def __rsub__(self, other):
        # 获取 `other` 的数据类型
        other_dtype = getattr(other, "dtype", None)
        # 检查 `other` 是否是 datetime64 类型或者 DatetimeTZDtype 类型的实例
        other_is_dt64 = lib.is_np_dtype(other_dtype, "M") or isinstance(
            other_dtype, DatetimeTZDtype
        )

        # 如果 `other` 是 datetime64 类型或 DatetimeTZDtype 类型的实例，而 `self` 是 timedelta64 类型
        if other_is_dt64 and lib.is_np_dtype(self.dtype, "m"):
            # ndarray[datetime64] 不能从 `self` 中减去，因此需要包装成 DatetimeArray/Index 并翻转操作
            if lib.is_scalar(other):
                # 即 np.datetime64 对象
                return Timestamp(other) - self
            if not isinstance(other, DatetimeLikeArrayMixin):
                # 避免将 DatetimeIndex 进行降级
                from pandas.core.arrays import DatetimeArray

                other = DatetimeArray._from_sequence(other)
            return other - self
        # 如果 `self` 是 datetime64 类型，并且 `other` 有 dtype 属性但不是 datetime64 类型
        elif self.dtype.kind == "M" and hasattr(other, "dtype") and not other_is_dt64:
            # GH#19959 datetime - datetime 可以被定义为 timedelta，但其他类型减去 datetime 无法定义
            raise TypeError(
                f"cannot subtract {type(self).__name__} from {type(other).__name__}"
            )
        # 如果 `self` 是 PeriodDtype 类型，并且 `other` 的 dtype 是 timedelta64 类型
        elif isinstance(self.dtype, PeriodDtype) and lib.is_np_dtype(other_dtype, "m"):
            # TODO: 我们能否简化或者泛化这些情况？
            raise TypeError(f"cannot subtract {type(self).__name__} from {other.dtype}")
        # 如果 `self` 是 timedelta64 类型
        elif lib.is_np_dtype(self.dtype, "m"):
            # 将 `self` 强制转换为 TimedeltaArray 类型
            self = cast("TimedeltaArray", self)
            # 返回 (-self) + other 的结果
            return (-self) + other

        # 如果到达这里，说明有例如 datetime 对象
        return -(self - other)

    # 实现增量加法的特殊方法 __iadd__
    def __iadd__(self, other) -> Self:
        # 计算 self + other 的结果
        result = self + other
        # 将结果复制到 self 中
        self[:] = result[:]

        # 如果 self 的数据类型不是 PeriodDtype
        if not isinstance(self.dtype, PeriodDtype):
            # 恢复 freq，因为 setitem 使其失效
            self._freq = result.freq
        return self

    # 实现增量减法的特殊方法 __isub__
    def __isub__(self, other) -> Self:
        # 计算 self - other 的结果
        result = self - other
        # 将结果复制到 self 中
        self[:] = result[:]

        # 如果 self 的数据类型不是 PeriodDtype
        if not isinstance(self.dtype, PeriodDtype):
            # 恢复 freq，因为 setitem 使其失效
            self._freq = result.freq
        return self

    # --------------------------------------------------------------
    # Reductions

    # 使用 _period_dispatch 装饰器，实现 _quantile 方法
    @_period_dispatch
    def _quantile(
        self,
        qs: npt.NDArray[np.float64],
        interpolation: str,
    ) -> Self:
        return super()._quantile(qs=qs, interpolation=interpolation)

    # 使用 _period_dispatch 装饰器，下面可能还有更多的代码
    @_period_dispatch
    # 定义一个方法用于计算数组或者沿着某个轴的最小值
    def min(self, *, axis: AxisInt | None = None, skipna: bool = True, **kwargs):
        """
        Return the minimum value of the Array or minimum along
        an axis.

        See Also
        --------
        numpy.ndarray.min
        Index.min : Return the minimum value in an Index.
        Series.min : Return the minimum value in a Series.
        """
        # 使用验证器确认不含参数，kwargs 应为空
        nv.validate_min((), kwargs)
        # 确保轴的值有效
        nv.validate_minmax_axis(axis, self.ndim)

        # 调用 nanmin 函数计算数组或者指定轴向上的最小值
        result = nanops.nanmin(self._ndarray, axis=axis, skipna=skipna)
        # 封装并返回归约结果
        return self._wrap_reduction_result(axis, result)

    # 应用周期性分派的装饰器定义方法用于计算数组或者沿着某个轴的最大值
    @_period_dispatch
    def max(self, *, axis: AxisInt | None = None, skipna: bool = True, **kwargs):
        """
        Return the maximum value of the Array or maximum along
        an axis.

        See Also
        --------
        numpy.ndarray.max
        Index.max : Return the maximum value in an Index.
        Series.max : Return the maximum value in a Series.
        """
        # 使用验证器确认不含参数，kwargs 应为空
        nv.validate_max((), kwargs)
        # 确保轴的值有效
        nv.validate_minmax_axis(axis, self.ndim)

        # 调用 nanmax 函数计算数组或者指定轴向上的最大值
        result = nanops.nanmax(self._ndarray, axis=axis, skipna=skipna)
        # 封装并返回归约结果
        return self._wrap_reduction_result(axis, result)
    def mean(self, *, skipna: bool = True, axis: AxisInt | None = 0):
        """
        Return the mean value of the Array.

        Parameters
        ----------
        skipna : bool, default True
            Whether to ignore any NaT elements.
        axis : int, optional, default 0
            Axis for the function to be applied on.

        Returns
        -------
        scalar
            Timestamp or Timedelta.

        See Also
        --------
        numpy.ndarray.mean : Returns the average of array elements along a given axis.
        Series.mean : Return the mean value in a Series.

        Notes
        -----
        mean is only defined for Datetime and Timedelta dtypes, not for Period.

        Examples
        --------
        For :class:`pandas.DatetimeIndex`:

        >>> idx = pd.date_range("2001-01-01 00:00", periods=3)
        >>> idx
        DatetimeIndex(['2001-01-01', '2001-01-02', '2001-01-03'],
                      dtype='datetime64[ns]', freq='D')
        >>> idx.mean()
        Timestamp('2001-01-02 00:00:00')

        For :class:`pandas.TimedeltaIndex`:

        >>> tdelta_idx = pd.to_timedelta([1, 2, 3], unit="D")
        >>> tdelta_idx
        TimedeltaIndex(['1 days', '2 days', '3 days'],
                        dtype='timedelta64[ns]', freq=None)
        >>> tdelta_idx.mean()
        Timedelta('2 days 00:00:00')
        """
        # 如果数组的数据类型是 PeriodDtype，则抛出 TypeError
        if isinstance(self.dtype, PeriodDtype):
            # 参见 GH#24757 中的讨论
            raise TypeError(
                f"mean is not implemented for {type(self).__name__} since the "
                "meaning is ambiguous.  An alternative is "
                "obj.to_timestamp(how='start').mean()"
            )

        # 计算数组的均值，使用 nanmean 处理可能的 NaN 值
        result = nanops.nanmean(
            self._ndarray, axis=axis, skipna=skipna, mask=self.isna()
        )
        # 包装并返回归约结果
        return self._wrap_reduction_result(axis, result)

    @_period_dispatch
    def median(self, *, axis: AxisInt | None = None, skipna: bool = True, **kwargs):
        nv.validate_median((), kwargs)

        # 如果指定的 axis 超出了数组维度的范围，则抛出 ValueError
        if axis is not None and abs(axis) >= self.ndim:
            raise ValueError("abs(axis) must be less than ndim")

        # 计算数组的中位数，使用 nanmedian 处理可能的 NaN 值
        result = nanops.nanmedian(self._ndarray, axis=axis, skipna=skipna)
        # 包装并返回归约结果
        return self._wrap_reduction_result(axis, result)

    def _mode(self, dropna: bool = True):
        # 如果 dropna 为 True，则生成用于标记 NaN 值的掩码
        mask = None
        if dropna:
            mask = self.isna()

        # 计算数组的众数
        i8modes = algorithms.mode(self.view("i8"), mask=mask)
        npmodes = i8modes.view(self._ndarray.dtype)
        npmodes = cast(np.ndarray, npmodes)
        # 从底层数据创建新的数组对象并返回
        return self._from_backing_data(npmodes)

    # ------------------------------------------------------------------
    # GroupBy Methods

    def _groupby_op(
        self,
        *,
        how: str,
        has_dropped_na: bool,
        min_count: int,
        ngroups: int,
        ids: npt.NDArray[np.intp],
        **kwargs,
        ):
            # 获取对象的数据类型
            dtype = self.dtype
            # 如果数据类型是日期时间类型
            if dtype.kind == "M":
                # 如果操作是加法或乘法，则不支持日期时间类型
                if how in ["sum", "prod", "cumsum", "cumprod", "var", "skew"]:
                    raise TypeError(f"datetime64 type does not support operation '{how}'")
                # 如果操作是'any'或'all'
                if how in ["any", "all"]:
                    # 抛出类型错误，不再支持使用'any'和'all'操作datetime64类型
                    raise TypeError(
                        f"'{how}' with datetime64 dtypes is no longer supported. "
                        f"Use (obj != pd.Timestamp(0)).{how}() instead."
                    )

            # 如果数据类型是时期类型
            elif isinstance(dtype, PeriodDtype):
                # 如果操作是加法或乘法，则不支持时期类型
                if how in ["sum", "prod", "cumsum", "cumprod", "var", "skew"]:
                    raise TypeError(f"Period type does not support {how} operations")
                # 如果操作是'any'或'all'
                if how in ["any", "all"]:
                    # 抛出类型错误，不再支持使用'any'和'all'操作PeriodDtype类型
                    raise TypeError(
                        f"'{how}' with PeriodDtype is no longer supported. "
                        f"Use (obj != pd.Period(0, freq)).{how}() instead."
                    )

            else:
                # 如果数据类型是时间差类型，我们可以进行加法操作，但不能进行乘法操作等
                if how in ["prod", "cumprod", "skew", "var"]:
                    raise TypeError(f"timedelta64 type does not support {how} operations")

        # 所有实现的函数都是序数的，因此我们可以在时区无关的等效值上操作
        npvalues = self._ndarray.view("M8[ns]")

        from pandas.core.groupby.ops import WrappedCythonOp

        # 从操作中获取操作类型
        kind = WrappedCythonOp.get_kind_from_how(how)
        # 创建WrappedCythonOp对象
        op = WrappedCythonOp(how=how, kind=kind, has_dropped_na=has_dropped_na)

        # 执行Cython操作并兼容多维数据
        res_values = op._cython_op_ndim_compat(
            npvalues,
            min_count=min_count,
            ngroups=ngroups,
            comp_ids=ids,
            mask=None,
            **kwargs,
        )

        # 如果操作在转换黑名单中，例如how in ["rank"]，则返回结果值
        if op.how in op.cast_blocklist:
            return res_values

        # 我们在上面视图为"M8[ns]"，现在进行反向操作
        assert res_values.dtype == "M8[ns]"
        # 如果操作是"std"或"sem"
        if how in ["std", "sem"]:
            from pandas.core.arrays import TimedeltaArray

            # 如果数据类型是时期类型，则不支持"std"和"sem"
            if isinstance(self.dtype, PeriodDtype):
                raise TypeError("'std' and 'sem' are not valid for PeriodDtype")
            # 将self转换为DatetimeArray或TimedeltaArray
            self = cast("DatetimeArray | TimedeltaArray", self)
            new_dtype = f"m8[{self.unit}]"
            res_values = res_values.view(new_dtype)
            return TimedeltaArray._simple_new(res_values, dtype=res_values.dtype)

        # 将结果值视图为原始数据类型
        res_values = res_values.view(self._ndarray.dtype)
        # 从后端数据创建新的对象
        return self._from_backing_data(res_values)
# DatelikeOps 类继承自 DatetimeLikeArrayMixin，提供了 DatetimeIndex/PeriodIndex 的通用操作，不适用于 TimedeltaIndex。
class DatelikeOps(DatetimeLikeArrayMixin):
    """
    Common ops for DatetimeIndex/PeriodIndex, but not TimedeltaIndex.
    """

    @Substitution(
        URL="https://docs.python.org/3/library/datetime.html"
        "#strftime-and-strptime-behavior"
    )
    # 定义 strftime 方法，用于将日期格式化为指定的字符串格式
    def strftime(self, date_format: str) -> npt.NDArray[np.object_]:
        """
        Convert to Index using specified date_format.

        Return an Index of formatted strings specified by date_format, which
        supports the same string format as the python standard library. Details
        of the string format can be found in `python string format
        doc <%(URL)s>`__.

        Formats supported by the C `strftime` API but not by the python string format
        doc (such as `"%%R"`, `"%%r"`) are not officially supported and should be
        preferably replaced with their supported equivalents (such as `"%%H:%%M"`,
        `"%%I:%%M:%%S %%p"`).

        Note that `PeriodIndex` support additional directives, detailed in
        `Period.strftime`.

        Parameters
        ----------
        date_format : str
            Date format string (e.g. "%%Y-%%m-%%d").

        Returns
        -------
        ndarray[object]
            NumPy ndarray of formatted strings.

        See Also
        --------
        to_datetime : Convert the given argument to datetime.
        DatetimeIndex.normalize : Return DatetimeIndex with times to midnight.
        DatetimeIndex.round : Round the DatetimeIndex to the specified freq.
        DatetimeIndex.floor : Floor the DatetimeIndex to the specified freq.
        Timestamp.strftime : Format a single Timestamp.
        Period.strftime : Format a single Period.

        Examples
        --------
        >>> rng = pd.date_range(pd.Timestamp("2018-03-10 09:00"), periods=3, freq="s")
        >>> rng.strftime("%%B %%d, %%Y, %%r")
        Index(['March 10, 2018, 09:00:00 AM', 'March 10, 2018, 09:00:01 AM',
               'March 10, 2018, 09:00:02 AM'],
              dtype='object')
        """
        # 调用 _format_native_types 方法进行本地类型的格式化处理，返回结果为 NumPy ndarray 对象
        result = self._format_native_types(date_format=date_format, na_rep=np.nan)
        # 强制将结果转换为 object 类型的 ndarray，并确保不复制数据
        return result.astype(object, copy=False)


_round_doc = """
    Perform {op} operation on the data to the specified `freq`.

    Parameters
    ----------
    freq : str or Offset
        The frequency level to {op} the index to. Must be a fixed
        frequency like 's' (second) not 'ME' (month end). See
        :ref:`frequency aliases <timeseries.offset_aliases>` for
        a list of possible `freq` values.
"""
    ambiguous : 'infer', bool-ndarray, 'NaT', default 'raise'
        # 处理模糊时间的参数设定，仅对DatetimeIndex有效：

        - 'infer' 尝试根据顺序推断秋季 DST 转换小时数
        - bool-ndarray 中 True 表示 DST 时间，False 表示非 DST 时间（注意，此标志仅适用于模糊时间）
        - 'NaT' 将在存在模糊时间时返回 NaT
        - 'raise' 将在存在模糊时间时引发 AmbiguousTimeError 异常。

    nonexistent : 'shift_forward', 'shift_backward', 'NaT', timedelta, default 'raise'
        # 处理不存在时间的参数设定，即某些时区因夏令时调整导致某段时间不存在的情况：

        - 'shift_forward' 将不存在时间向前移动到最接近的存在时间
        - 'shift_backward' 将不存在时间向后移动到最接近的存在时间
        - 'NaT' 将在存在不存在时间时返回 NaT
        - timedelta 对象将不存在时间按 timedelta 进行偏移
        - 'raise' 将在存在不存在时间时引发 NonExistentTimeError 异常。

    Returns
    -------
    DatetimeIndex, TimedeltaIndex, or Series
        # 返回与原 DatetimeIndex 或 TimedeltaIndex 相同类型的索引，或者对于 Series 返回具有相同索引的 Series。

    Raises
    ------
    ValueError if the `freq` cannot be converted.
        # 如果无法转换 `freq`，将引发 ValueError 异常。

    See Also
    --------
    DatetimeIndex.floor : 对数据执行 floor 操作以指定的 `freq`。
    DatetimeIndex.snap : 将时间戳对齐到最接近的频率。

    Notes
    -----
    If the timestamps have a timezone, {op}ing will take place relative to the
    local ("wall") time and re-localized to the same timezone. When {op}ing
    near daylight savings time, use ``nonexistent`` and ``ambiguous`` to
    control the re-localization behavior.
        # 如果时间戳具有时区信息，{op}ing 将相对于本地（“墙”）时间进行，并重新定位到相同的时区。
        # 在靠近夏令时调整时，使用 ``nonexistent`` 和 ``ambiguous`` 来控制重新定位行为。

    Examples
    --------
    **DatetimeIndex**

    >>> rng = pd.date_range('1/1/2018 11:59:00', periods=3, freq='min')
    >>> rng
    DatetimeIndex(['2018-01-01 11:59:00', '2018-01-01 12:00:00',
                   '2018-01-01 12:01:00'],
                  dtype='datetime64[ns]', freq='min')
# 定义一个示例字符串，展示如何使用 Pandas 的 DatetimeIndex 对象的 round 方法
_round_example = """>>> rng.round('h')
    DatetimeIndex(['2018-01-01 12:00:00', '2018-01-01 12:00:00',
                   '2018-01-01 12:00:00'],
                  dtype='datetime64[ns]', freq=None)

    **Series**

    >>> pd.Series(rng).dt.round("h")
    0   2018-01-01 12:00:00
    1   2018-01-01 12:00:00
    2   2018-01-01 12:00:00
    dtype: datetime64[ns]

    当接近夏令时转换时，使用 ``ambiguous`` 或 ``nonexistent`` 控制时间戳的重新本地化。

    >>> rng_tz = pd.DatetimeIndex(["2021-10-31 03:30:00"], tz="Europe/Amsterdam")

    >>> rng_tz.floor("2h", ambiguous=False)
    DatetimeIndex(['2021-10-31 02:00:00+01:00'],
                  dtype='datetime64[s, Europe/Amsterdam]', freq=None)

    >>> rng_tz.floor("2h", ambiguous=True)
    DatetimeIndex(['2021-10-31 02:00:00+02:00'],
                  dtype='datetime64[s, Europe/Amsterdam]', freq=None)
    """

# 定义一个示例字符串，展示如何使用 Pandas 的 DatetimeIndex 对象的 floor 方法
_floor_example = """>>> rng.floor('h')
    DatetimeIndex(['2018-01-01 11:00:00', '2018-01-01 12:00:00',
                   '2018-01-01 12:00:00'],
                  dtype='datetime64[ns]', freq=None)

    **Series**

    >>> pd.Series(rng).dt.floor("h")
    0   2018-01-01 11:00:00
    1   2018-01-01 12:00:00
    2   2018-01-01 12:00:00
    dtype: datetime64[ns]

    当接近夏令时转换时，使用 ``ambiguous`` 或 ``nonexistent`` 控制时间戳的重新本地化。

    >>> rng_tz = pd.DatetimeIndex(["2021-10-31 03:30:00"], tz="Europe/Amsterdam")

    >>> rng_tz.floor("2h", ambiguous=False)
    DatetimeIndex(['2021-10-31 02:00:00+01:00'],
                 dtype='datetime64[s, Europe/Amsterdam]', freq=None)

    >>> rng_tz.floor("2h", ambiguous=True)
    DatetimeIndex(['2021-10-31 02:00:00+02:00'],
                  dtype='datetime64[s, Europe/Amsterdam]', freq=None)
    """

# 定义一个示例字符串，展示如何使用 Pandas 的 DatetimeIndex 对象的 ceil 方法
_ceil_example = """>>> rng.ceil('h')
    DatetimeIndex(['2018-01-01 12:00:00', '2018-01-01 12:00:00',
                   '2018-01-01 13:00:00'],
                  dtype='datetime64[ns]', freq=None)

    **Series**

    >>> pd.Series(rng).dt.ceil("h")
    0   2018-01-01 12:00:00
    1   2018-01-01 12:00:00
    2   2018-01-01 13:00:00
    dtype: datetime64[ns]

    当接近夏令时转换时，使用 ``ambiguous`` 或 ``nonexistent`` 控制时间戳的重新本地化。

    >>> rng_tz = pd.DatetimeIndex(["2021-10-31 01:30:00"], tz="Europe/Amsterdam")

    >>> rng_tz.ceil("h", ambiguous=False)
    DatetimeIndex(['2021-10-31 02:00:00+01:00'],
                  dtype='datetime64[s, Europe/Amsterdam]', freq=None)

    >>> rng_tz.ceil("h", ambiguous=True)
    DatetimeIndex(['2021-10-31 02:00:00+02:00'],
                  dtype='datetime64[s, Europe/Amsterdam]', freq=None)
    """


class TimelikeOps(DatetimeLikeArrayMixin):
    """
    Common ops for TimedeltaIndex/DatetimeIndex, but not PeriodIndex.
    """

    @classmethod
    # 定义一个类方法 `_validate_dtype`，用于验证数据类型，抽象方法，需要在子类中实现具体逻辑
    def _validate_dtype(cls, values, dtype):
        raise AbstractMethodError(cls)

    @property
    # 定义一个属性 `freq`，返回频率对象（如果已设置），否则返回 None
    def freq(self):
        """
        Return the frequency object if it is set, otherwise None.

        To learn more about the frequency strings, please see
        :ref:`this link<timeseries.offset_aliases>`.

        See Also
        --------
        DatetimeIndex.freq : Return the frequency object if it is set, otherwise None.
        PeriodIndex.freq : Return the frequency object if it is set, otherwise None.

        Examples
        --------
        >>> datetimeindex = pd.date_range(
        ...     "2022-02-22 02:22:22", periods=10, tz="America/Chicago", freq="h"
        ... )
        >>> datetimeindex
        DatetimeIndex(['2022-02-22 02:22:22-06:00', '2022-02-22 03:22:22-06:00',
                       '2022-02-22 04:22:22-06:00', '2022-02-22 05:22:22-06:00',
                       '2022-02-22 06:22:22-06:00', '2022-02-22 07:22:22-06:00',
                       '2022-02-22 08:22:22-06:00', '2022-02-22 09:22:22-06:00',
                       '2022-02-22 10:22:22-06:00', '2022-02-22 11:22:22-06:00'],
                      dtype='datetime64[ns, America/Chicago]', freq='h')
        >>> datetimeindex.freq
        <Hour>
        """
        return self._freq

    @freq.setter
    # 定义 `freq` 属性的设置方法，允许设置频率对象，验证类型并处理相关错误
    def freq(self, value) -> None:
        if value is not None:
            # 将输入值转换为偏移量对象
            value = to_offset(value)
            # 调用 `_validate_frequency` 方法验证频率设置的正确性
            self._validate_frequency(self, value)
            # 如果数据类型的种类是 'm' 且 `value` 不是 `Tick` 类型，则抛出类型错误
            if self.dtype.kind == "m" and not isinstance(value, Tick):
                raise TypeError("TimedeltaArray/Index freq must be a Tick")

            # 如果数组维度大于 1，则抛出值错误
            if self.ndim > 1:
                raise ValueError("Cannot set freq with ndim > 1")

        # 设置频率属性 `_freq` 为输入值 `value`
        self._freq = value

    @final
   `
    # 定义一个构造函数辅助方法，用于设置适当的 `freq` 属性。假设 self._freq 目前设置为在 _from_sequence_not_strict 中推断出的 freq。
    def _maybe_pin_freq(self, freq, validate_kwds: dict) -> None:
        """
        Constructor helper to pin the appropriate `freq` attribute.  Assumes
        that self._freq is currently set to any freq inferred in
        _from_sequence_not_strict.
        """
        # 如果 freq 为 None，用户明确传入 None -> 覆盖任何推断的 freq
        if freq is None:
            self._freq = None
        # 如果 freq 为 "infer"，并且 self._freq 不是 None，说明已经推断出 freq，无需做任何操作
        elif freq == "infer":
            if self._freq is None:
                # 直接将 _freq 设置为推断的 freq，绕过重复的 _validate_frequency 检查
                self._freq = to_offset(self.inferred_freq)
        # 如果 freq 为 lib.no_default，用户没有指定任何内容，保持原数据的推断 freq，如果原始数据有 freq，则保持不变，否则不做任何操作
        elif freq is lib.no_default:
            pass
        # 如果 self._freq 为 None，说明无法从数据中继承 freq，需要验证用户传入的 freq
        elif self._freq is None:
            freq = to_offset(freq)
            # 调用类方法 _validate_frequency 验证用户传入的 freq
            type(self)._validate_frequency(self, freq, **validate_kwds)
            self._freq = freq
        # 否则，检查用户传入的 freq 是否与已有的 freq 冲突
        else:
            freq = to_offset(freq)
            _validate_inferred_freq(freq, self._freq)

    # 使用 @final 装饰器，表示此方法是最终方法，不能被子类重写
    @final
    # 使用 @classmethod 装饰器，表示该方法为类方法
    @classmethod
    def _validate_frequency(cls, index, freq: BaseOffset, **kwargs) -> None:
        """
        Validate that a frequency is compatible with the values of a given
        Datetime Array/Index or Timedelta Array/Index

        Parameters
        ----------
        index : DatetimeIndex or TimedeltaIndex
            The index on which to determine if the given frequency is valid
        freq : DateOffset
            The frequency to validate
        """
        # 获取推断的频率
        inferred = index.inferred_freq
        # 如果索引为空或推断的频率与给定频率相同，则返回
        if index.size == 0 or inferred == freq.freqstr:
            return None

        try:
            # 生成指定范围的时间序列
            on_freq = cls._generate_range(
                start=index[0],
                end=None,
                periods=len(index),
                freq=freq,
                unit=index.unit,
                **kwargs,
            )
            # 如果索引数据与生成的时间序列数据不相等，则抛出值错误异常
            if not np.array_equal(index.asi8, on_freq.asi8):
                raise ValueError
        except ValueError as err:
            if "non-fixed" in str(err):
                # 对于 timedelta64，非固定频率不具有实际意义；
                # 保留此错误消息
                raise err
            # 如果上面的 `np.array_equal` 检查为 False，则可能会到达此处。
            # 或者如果 index[0] 是 `NaT`，则调用 `cls._generate_range` 会引发 ValueError，
            # 我们重新引发带有更具针对性的消息的 ValueError 异常。
            raise ValueError(
                f"Inferred frequency {inferred} from passed values "
                f"does not conform to passed frequency {freq.freqstr}"
            ) from err

    @classmethod
    def _generate_range(
        cls, start, end, periods: int | None, freq, *args, **kwargs
    ) -> Self:
        # 抽象方法错误，由子类实现
        raise AbstractMethodError(cls)

    # --------------------------------------------------------------

    @cache_readonly
    def _creso(self) -> int:
        # 从数据类型获取单位
        return get_unit_from_dtype(self._ndarray.dtype)

    @cache_readonly
    def unit(self) -> str:
        # 返回数据类型对应的时间单位，例如 "ns", "us", "ms"
        # 错误：参数 1 传递给 "dtype_to_unit" 的类型不兼容 "ExtensionDtype"；期望类型为
        # "Union[DatetimeTZDtype, dtype[Any]]"
        return dtype_to_unit(self.dtype)  # type: ignore[arg-type]
    # 将当前对象转换为指定单位的数据类型。

    # 时间戳表示的极限取决于所选的分辨率。
    # 可以通过as_unit方法将不同分辨率转换为彼此。

    def as_unit(self, unit: str, round_ok: bool = True) -> Self:
        """
        Convert to a dtype with the given unit resolution.

        The limits of timestamp representation depend on the chosen resolution.
        Different resolutions can be converted to each other through as_unit.

        Parameters
        ----------
        unit : {'s', 'ms', 'us', 'ns'}
            The unit to convert to.
        round_ok : bool, default True
            If False and rounding is necessary, raise ValueError.

        Returns
        -------
        type(self)
            Converted to the specified unit.

        Raises
        ------
        ValueError
            If the provided unit is not one of {'s', 'ms', 'us', 'ns'}.

        See Also
        --------
        Timestamp.as_unit : Convert to the given unit.

        Examples
        --------
        For :class:`pandas.DatetimeIndex`:

        >>> idx = pd.DatetimeIndex(["2020-01-02 01:02:03.004005006"])
        >>> idx
        DatetimeIndex(['2020-01-02 01:02:03.004005006'],
                      dtype='datetime64[ns]', freq=None)
        >>> idx.as_unit("s")
        DatetimeIndex(['2020-01-02 01:02:03'], dtype='datetime64[s]', freq=None)

        For :class:`pandas.TimedeltaIndex`:

        >>> tdelta_idx = pd.to_timedelta(["1 day 3 min 2 us 42 ns"])
        >>> tdelta_idx
        TimedeltaIndex(['1 days 00:03:00.000002042'],
                        dtype='timedelta64[ns]', freq=None)
        >>> tdelta_idx.as_unit("s")
        TimedeltaIndex(['1 days 00:03:00'], dtype='timedelta64[s]', freq=None)
        """
        # 如果单位不是 {'s', 'ms', 'us', 'ns'} 中的一个，则引发 ValueError 异常
        if unit not in ["s", "ms", "us", "ns"]:
            raise ValueError("Supported units are 's', 'ms', 'us', 'ns'")

        # 创建一个新的数据类型，根据当前对象的dtype和指定的单位
        dtype = np.dtype(f"{self.dtype.kind}8[{unit}]")

        # 使用astype_overflowsafe方法将self._ndarray转换为新的dtype
        new_values = astype_overflowsafe(self._ndarray, dtype, round_ok=round_ok)

        # 确定新的数据类型
        if isinstance(self.dtype, np.dtype):
            new_dtype = new_values.dtype
        else:
            # 如果当前对象不是np.dtype，则创建一个新的DatetimeTZDtype对象
            tz = cast("DatetimeArray", self).tz
            new_dtype = DatetimeTZDtype(tz=tz, unit=unit)

        # 返回一个新的对象，类型与self相同，使用_simple_new方法创建
        return type(self)._simple_new(
            new_values,
            dtype=new_dtype,
            freq=self.freq,  # type: ignore[call-arg]
        )

    # TODO: annotate other as DatetimeArray | TimedeltaArray | Timestamp | Timedelta
    #  with the return type matching input type.  TypeVar?
    def _ensure_matching_resos(self, other):
        """
        Ensure that self and other have matching resolutions.

        If resolutions differ, convert one to match the other's resolution.
        This is similar to handling Timestamp/Timedelta objects.

        Parameters
        ----------
        other : object
            Another object to compare and potentially convert resolution.

        Returns
        -------
        tuple
            A tuple containing self and other, potentially adjusted for matching resolutions.
        """
        # 如果当前对象的分辨率与other对象的分辨率不同
        if self._creso != other._creso:
            # 类似于处理Timestamp/Timedelta对象，将其中一个转换为更高的分辨率
            if self._creso < other._creso:
                self = self.as_unit(other.unit)
            else:
                other = other.as_unit(self.unit)
        # 返回调整后的self和other对象
        return self, other

    # --------------------------------------------------------------
    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        # 如果 ufunc 是 np.isnan、np.isinf 或 np.isfinite，并且输入只有一个且为 self
        if (
            ufunc in [np.isnan, np.isinf, np.isfinite]
            and len(inputs) == 1
            and inputs[0] is self
        ):
            # numpy 1.18 修改了 isinf 和 isnan，在 dt64/td64 上不再抛出异常
            return getattr(ufunc, method)(self._ndarray, **kwargs)

        # 调用父类的 __array_ufunc__ 方法处理其他情况
        return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)

    def _round(self, freq, mode, ambiguous, nonexistent):
        # 对本地时间进行四舍五入处理
        if isinstance(self.dtype, DatetimeTZDtype):
            # 操作在本地时间戳上，然后再转换回有时区意识的时间
            self = cast("DatetimeArray", self)
            naive = self.tz_localize(None)
            result = naive._round(freq, mode, ambiguous, nonexistent)
            return result.tz_localize(
                self.tz, ambiguous=ambiguous, nonexistent=nonexistent
            )

        # 将数组视图转换为 i8 类型
        values = self.view("i8")
        values = cast(np.ndarray, values)
        # 获取用于四舍五入的纳秒单位
        nanos = get_unit_for_round(freq, self._creso)
        if nanos == 0:
            # GH 52761
            # 如果纳秒单位为 0，返回原数组的副本
            return self.copy()
        # 对 i8 类型的数组进行四舍五入处理
        result_i8 = round_nsint64(values, mode, nanos)
        # 将结果转换为可能的掩码结果，使用 iNaT 作为填充值
        result = self._maybe_mask_results(result_i8, fill_value=iNaT)
        # 将结果视图转换为 self 对象的数据类型
        result = result.view(self._ndarray.dtype)
        # 返回一个新的 self 对象，使用处理后的结果和数据类型
        return self._simple_new(result, dtype=self.dtype)

    @Appender((_round_doc + _round_example).format(op="round"))
    def round(
        self,
        freq,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ) -> Self:
        # 调用 _round 方法进行四舍五入操作，使用最接近的偶数模式
        return self._round(freq, RoundTo.NEAREST_HALF_EVEN, ambiguous, nonexistent)

    @Appender((_round_doc + _floor_example).format(op="floor"))
    def floor(
        self,
        freq,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ) -> Self:
        # 调用 _round 方法进行向下取整操作，使用负无穷模式
        return self._round(freq, RoundTo.MINUS_INFTY, ambiguous, nonexistent)

    @Appender((_round_doc + _ceil_example).format(op="ceil"))
    def ceil(
        self,
        freq,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ) -> Self:
        # 调用 _round 方法进行向上取整操作，使用正无穷模式
        return self._round(freq, RoundTo.PLUS_INFTY, ambiguous, nonexistent)

    # --------------------------------------------------------------
    # Reductions

    def any(self, *, axis: AxisInt | None = None, skipna: bool = True) -> bool:
        # GH#34479 nanops 调用将会对非 td64 数据类型引发 TypeError 异常
        return nanops.nanany(self._ndarray, axis=axis, skipna=skipna, mask=self.isna())

    def all(self, *, axis: AxisInt | None = None, skipna: bool = True) -> bool:
        # GH#34479 nanops 调用将会对非 td64 数据类型引发 TypeError 异常
        return nanops.nanall(self._ndarray, axis=axis, skipna=skipna, mask=self.isna())

    # --------------------------------------------------------------
    # Frequency Methods
    # 将频率属性 `_freq` 设置为 `None`
    def _maybe_clear_freq(self) -> None:
        self._freq = None

    # 返回一个带有新频率的数据视图
    def _with_freq(self, freq) -> Self:
        """
        Helper to get a view on the same data, with a new freq.

        Parameters
        ----------
        freq : DateOffset, None, or "infer"

        Returns
        -------
        Same type as self
        """
        # GH#29843: 根据 GitHub issue 编号说明此处的条件判断
        if freq is None:
            # Always valid: 如果频率为 None，则条件始终成立
            pass
        elif len(self) == 0 and isinstance(freq, BaseOffset):
            # Always valid.  In the TimedeltaArray case, we require a Tick offset
            # 如果数据长度为 0 并且 freq 是 BaseOffset 的实例，则条件始终成立
            if self.dtype.kind == "m" and not isinstance(freq, Tick):
                raise TypeError("TimedeltaArray/Index freq must be a Tick")
        else:
            # As an internal method, we can ensure this assertion always holds
            # 作为内部方法，我们可以确保此断言始终成立
            assert freq == "infer"
            freq = to_offset(self.inferred_freq)

        arr = self.view()
        arr._freq = freq  # 设置 arr 对象的频率属性为计算后的频率
        return arr

    # --------------------------------------------------------------
    # ExtensionArray Interface

    # 返回用于 JSON 序列化的值数组
    def _values_for_json(self) -> np.ndarray:
        # Small performance bump vs the base class which calls np.asarray(self)
        # 与基类相比，稍微提升性能，后者调用 np.asarray(self)
        if isinstance(self.dtype, np.dtype):
            return self._ndarray  # 如果数据类型是 numpy 的 dtype，则直接返回内部数组
        return super()._values_for_json()  # 否则调用父类方法处理 JSON 序列化

    # 因子化操作，返回编码和唯一值
    def factorize(
        self,
        use_na_sentinel: bool = True,
        sort: bool = False,
    ):
        if self.freq is not None:
            # We must be unique, so can short-circuit (and retain freq)
            # 我们必须是唯一的，所以可以简化处理（并保留频率）
            if sort and self.freq.n < 0:
                codes = np.arange(len(self) - 1, -1, -1, dtype=np.intp)
                uniques = self[::-1]
            else:
                codes = np.arange(len(self), dtype=np.intp)
                uniques = self.copy()  # TODO: copy or view?
            return codes, uniques

        if sort:
            # algorithms.factorize only passes sort=True here when freq is
            #  not None, so this should not be reached.
            raise NotImplementedError(
                f"The 'sort' keyword in {type(self).__name__}.factorize is "
                "ignored unless arr.freq is not None. To factorize with sort, "
                "call pd.factorize(obj, sort=True) instead."
            )
        return super().factorize(use_na_sentinel=use_na_sentinel)

    @classmethod
    # 合并相同类型的数据
    def _concat_same_type(
        cls,
        to_concat: Sequence[Self],
        axis: AxisInt = 0,
    ) -> Self:
        # 调用父类方法，用于连接相同类型的对象
        new_obj = super()._concat_same_type(to_concat, axis)

        # 获取第一个待连接的对象
        obj = to_concat[0]

        # 如果轴向为0
        if axis == 0:
            # GH 3232: 如果连接结果是均匀间隔的，可以保留原始频率
            # 过滤掉空对象
            to_concat = [x for x in to_concat if len(x)]

            # 如果第一个对象有频率，并且所有对象的频率都与第一个对象相同
            if obj.freq is not None and all(x.freq == obj.freq for x in to_concat):
                # 检查对象之间的间隔是否为原始频率的倍数
                pairs = zip(to_concat[:-1], to_concat[1:])
                if all(pair[0][-1] + obj.freq == pair[1][0] for pair in pairs):
                    # 设置新的频率
                    new_freq = obj.freq
                    new_obj._freq = new_freq
        return new_obj

    def copy(self, order: str = "C") -> Self:
        # 调用父类方法复制对象
        new_obj = super().copy(order=order)
        # 设置新对象的频率与当前对象相同
        new_obj._freq = self.freq
        return new_obj

    def interpolate(
        self,
        *,
        method: InterpolateOptions,
        axis: int,
        index: Index,
        limit,
        limit_direction,
        limit_area,
        copy: bool,
        **kwargs,
    ) -> Self:
        """
        See NDFrame.interpolate.__doc__.
        """
        # 注意：即使 copy=False，我们也返回 type(self)
        if method != "linear":
            # 抛出未实现的方法异常，除非 method 为 "linear"
            raise NotImplementedError

        if not copy:
            # 如果不复制，直接使用现有数据
            out_data = self._ndarray
        else:
            # 复制数据到新数组
            out_data = self._ndarray.copy()

        # 使用缺失值插值方法填充数据
        missing.interpolate_2d_inplace(
            out_data,
            method=method,
            axis=axis,
            index=index,
            limit=limit,
            limit_direction=limit_direction,
            limit_area=limit_area,
            **kwargs,
        )
        if not copy:
            # 如果不复制，返回当前对象
            return self
        # 否则，返回一个新的类型相同的对象
        return type(self)._simple_new(out_data, dtype=self.dtype)

    # --------------------------------------------------------------
    # 未排序的方法

    @property
    def _is_dates_only(self) -> bool:
        """
        检查是否仅包含日期，时间为午夜（没有时区），这种情况下比其他情况具有更紧凑的 __repr__。
        对于 TimedeltaArray，检查是否为 24 小时的整数倍。
        """
        if not lib.is_np_dtype(self.dtype):
            # 如果数据类型不是 NumPy 类型，则说明有时区信息，返回 False
            return False

        # 获取整数形式的值
        values_int = self.asi8
        # 考虑的值不是缺失值
        consider_values = values_int != iNaT
        # 获取时间分辨率单位
        reso = get_unit_from_dtype(self.dtype)
        # 每天的周期数
        ppd = periods_per_day(reso)

        # TODO: 是否可以重用 is_date_array_normalized? 需要一个 skipna 参数
        #  （第一次尝试比这个实现性能差）
        # 检查是否为整数天数
        even_days = np.logical_and(consider_values, values_int % ppd != 0).sum() == 0
        return even_days
# -------------------------------------------------------------------
# Shared Constructor Helpers

# 确保数据能够被处理成日期时间样式的数组或类似数组
def ensure_arraylike_for_datetimelike(
    data, copy: bool, cls_name: str
) -> tuple[ArrayLike, bool]:
    # 如果数据没有 dtype 属性，如列表或元组
    if not hasattr(data, "dtype"):
        # 如果数据是单个元素且不是列表或元组
        if not isinstance(data, (list, tuple)) and np.ndim(data) == 0:
            # 转换成列表
            data = list(data)

        # 从类列表的数据构造一个 1 维对象数组
        data = construct_1d_object_array_from_listlike(data)
        copy = False
    elif isinstance(data, ABCMultiIndex):
        # 如果数据是多索引，抛出类型错误
        raise TypeError(f"Cannot create a {cls_name} from a MultiIndex.")
    else:
        # 从数据中提取数组，使用 numpy
        data = extract_array(data, extract_numpy=True)

    # 如果数据是整数数组或 Arrow 扩展数组并且数据类型是整数
    if isinstance(data, IntegerArray) or (
        isinstance(data, ArrowExtensionArray) and data.dtype.kind in "iu"
    ):
        # 转换成 int64 类型的 numpy 数组，na_value 使用 iNaT
        data = data.to_numpy("int64", na_value=iNaT)
        copy = False
    elif isinstance(data, ArrowExtensionArray):
        # 尝试转换成日期时间样式的数组
        data = data._maybe_convert_datelike_array()
        # 转换成 numpy 数组
        data = data.to_numpy()
        copy = False
    elif not isinstance(data, (np.ndarray, ExtensionArray)):
        # 如果数据不是 numpy 数组或扩展数组，转换成 numpy 数组
        data = np.asarray(data)

    elif isinstance(data, ABCCategorical):
        # 保留时区信息，如果是日期时间索引到分类到日期时间索引的情况
        data = data.categories.take(data.codes, fill_value=NaT)._values
        copy = False

    return data, copy


# 验证 periods 参数的类型，如果不是 None 或 int 则抛出 TypeError
@overload
def validate_periods(periods: None) -> None: ...


@overload
def validate_periods(periods: int) -> int: ...


def validate_periods(periods: int | None) -> int | None:
    """
    If a `periods` argument is passed to the Datetime/Timedelta Array/Index
    constructor, cast it to an integer.

    Parameters
    ----------
    periods : None, int

    Returns
    -------
    periods : None or int

    Raises
    ------
    TypeError
        if periods is not None or int
    """
    if periods is not None and not lib.is_integer(periods):
        raise TypeError(f"periods must be an integer, got {periods}")
    # 返回 periods 参数，用 type: ignore[return-value] 标记类型检查错误
    return periods  # type: ignore[return-value]


# 验证推断的频率是否与用户传递的频率匹配，返回匹配的频率或 None
def _validate_inferred_freq(
    freq: BaseOffset | None, inferred_freq: BaseOffset | None
) -> BaseOffset | None:
    """
    If the user passes a freq and another freq is inferred from passed data,
    require that they match.

    Parameters
    ----------
    freq : DateOffset or None
    inferred_freq : DateOffset or None

    Returns
    -------
    freq : DateOffset or None
    """
    # 如果推断的频率（inferred_freq）不是 None，则执行以下操作
    if inferred_freq is not None:
        # 如果传入的频率（freq）不是 None，并且不等于推断的频率（inferred_freq），则抛出 ValueError 异常
        if freq is not None and freq != inferred_freq:
            raise ValueError(
                f"Inferred frequency {inferred_freq} from passed "
                "values does not conform to passed frequency "
                f"{freq.freqstr}"
            )
        # 如果传入的频率（freq）是 None，则将其设为推断的频率（inferred_freq）
        if freq is None:
            freq = inferred_freq

    # 返回最终确定的频率（freq）
    return freq
def dtype_to_unit(dtype: DatetimeTZDtype | np.dtype | ArrowDtype) -> str:
    """
    Return the unit str corresponding to the dtype's resolution.

    Parameters
    ----------
    dtype : DatetimeTZDtype or np.dtype
        If np.dtype, we assume it is a datetime64 dtype.

    Returns
    -------
    str
    """
    # 检查 dtype 是否为 DatetimeTZDtype 类型
    if isinstance(dtype, DatetimeTZDtype):
        # 如果是 DatetimeTZDtype 类型，直接返回其 unit 属性
        return dtype.unit
    # 检查 dtype 是否为 ArrowDtype 类型
    elif isinstance(dtype, ArrowDtype):
        # 如果是 ArrowDtype 类型，检查其 kind 是否为 'm' 或 'M'
        if dtype.kind not in "mM":
            # 如果 kind 不是 'm' 或 'M'，抛出异常
            raise ValueError(f"{dtype=} does not have a resolution.")
        # 返回 ArrowDtype 的 pyarrow_dtype 的 unit 属性
        return dtype.pyarrow_dtype.unit
    # 如果以上类型判断都不符合，则假定 dtype 是 np.dtype，使用 np.datetime_data 获取其单位信息
    return np.datetime_data(dtype)[0]
```