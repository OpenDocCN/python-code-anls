# `D:\src\scipysrc\pandas\pandas\core\dtypes\common.py`

```
# 引入未来版本的类型注解支持
from __future__ import annotations

# 引入必要的模块和类
from typing import (
    TYPE_CHECKING,  # 用于类型检查
    Any,  # 任意类型
)
import warnings  # 引入警告模块

import numpy as np  # 引入NumPy库

# 从pandas的内部库中导入特定模块和类
from pandas._libs import (
    Interval,  # 区间对象
    Period,  # 时期对象
    algos,  # 算法函数集合
    lib,  # pandas底层库
)
from pandas._libs.tslibs import conversion  # 时间序列相关的转换功能
from pandas.util._exceptions import find_stack_level  # 查找堆栈级别的异常处理函数

# 从pandas核心数据类型中导入注册表和特定数据类型
from pandas.core.dtypes.base import _registry as registry  # 数据类型注册表
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,  # 分类数据类型
    DatetimeTZDtype,  # 带时区的日期时间数据类型
    ExtensionDtype,  # 扩展数据类型
    IntervalDtype,  # 区间数据类型
    PeriodDtype,  # 时期数据类型
    SparseDtype,  # 稀疏数据类型
)
from pandas.core.dtypes.generic import ABCIndex  # 抽象基类索引
from pandas.core.dtypes.inference import (
    is_array_like,  # 是否类数组
    is_bool,  # 是否布尔型
    is_complex,  # 是否复数类型
    is_dataclass,  # 是否数据类
    is_decimal,  # 是否十进制数
    is_dict_like,  # 是否类字典
    is_file_like,  # 是否文件对象
    is_float,  # 是否浮点数
    is_hashable,  # 是否可哈希
    is_integer,  # 是否整数
    is_iterator,  # 是否迭代器
    is_list_like,  # 是否类列表
    is_named_tuple,  # 是否命名元组
    is_nested_list_like,  # 是否嵌套类列表
    is_number,  # 是否数字
    is_re,  # 是否正则表达式
    is_re_compilable,  # 是否可编译为正则表达式
    is_scalar,  # 是否标量
    is_sequence,  # 是否序列
)

if TYPE_CHECKING:
    from collections.abc import Callable  # 引入标准库中的Callable类型

    from pandas._typing import (
        ArrayLike,  # 类数组类型
        DtypeObj,  # 数据类型对象
    )

DT64NS_DTYPE = conversion.DT64NS_DTYPE  # NumPy的日期时间类型
TD64NS_DTYPE = conversion.TD64NS_DTYPE  # NumPy的时间差类型
INT64_DTYPE = np.dtype(np.int64)  # 64位整数类型

# 以下定义了多个确保数据类型的函数，用于将数据类型转换为特定的数值类型
_is_scipy_sparse = None  # scipy稀疏矩阵标志变量

ensure_float64 = algos.ensure_float64  # 确保为64位浮点数
ensure_int64 = algos.ensure_int64  # 确保为64位整数
ensure_int32 = algos.ensure_int32  # 确保为32位整数
ensure_int16 = algos.ensure_int16  # 确保为16位整数
ensure_int8 = algos.ensure_int8  # 确保为8位整数
ensure_platform_int = algos.ensure_platform_int  # 确保为平台默认的整数类型
ensure_object = algos.ensure_object  # 确保为Python对象
ensure_uint64 = algos.ensure_uint64  # 确保为64位无符号整数


def ensure_str(value: bytes | Any) -> str:
    """
    确保将字节和非字符串转换为str对象。
    """
    if isinstance(value, bytes):
        value = value.decode("utf-8")  # 将字节解码为UTF-8编码的字符串
    elif not isinstance(value, str):
        value = str(value)  # 将非字符串类型转换为字符串
    return value


def ensure_python_int(value: int | np.integer) -> int:
    """
    确保值为Python整数类型。

    参数
    ----------
    value: int or numpy.integer
        要转换的值

    返回
    -------
    int
        转换后的整数值

    异常
    ------
    TypeError: 如果值不是整数或无法转换为整数时抛出异常
    """
    if not (is_integer(value) or is_float(value)):
        if not is_scalar(value):
            raise TypeError(
                f"Value needs to be a scalar value, was type {type(value).__name__}"
            )  # 如果值不是标量类型，则抛出类型错误异常
        raise TypeError(f"Wrong type {type(value)} for value {value}")  # 如果值类型错误，则抛出类型错误异常
    try:
        new_value = int(value)  # 尝试将值转换为整数类型
        assert new_value == value  # 确保转换后的值与原始值相等
    except (TypeError, ValueError, AssertionError) as err:
        raise TypeError(f"Wrong type {type(value)} for value {value}") from err  # 捕获并传递类型错误异常
    return new_value


def classes(*klasses) -> Callable:
    """
    评估类型是否是指定类的子类。
    """
    return lambda tipo: issubclass(tipo, klasses)


def _classes_and_not_datetimelike(*klasses) -> Callable:
    """
    评估类型是否是指定类的子类，并且不是日期时间类。
    """
    # 返回一个 lambda 函数，该函数接受一个参数 tipo，并执行以下逻辑：
    # - 检查 tipo 是否是 klasses 中的子类
    # - 确保 tipo 不是 np.datetime64 或 np.timedelta64 的子类
    return lambda tipo: (
        issubclass(tipo, klasses)
        and not issubclass(tipo, (np.datetime64, np.timedelta64))
    )
# 检查一个数组或数据类型是否是对象数据类型
def is_object_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the object dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the object dtype.

    Examples
    --------
    >>> from pandas.api.types import is_object_dtype
    >>> is_object_dtype(object)
    True
    >>> is_object_dtype(int)
    False
    >>> is_object_dtype(np.array([], dtype=object))
    True
    >>> is_object_dtype(np.array([], dtype=int))
    False
    >>> is_object_dtype([1, 2, 3])
    False
    """
    # 调用 _is_dtype_type 函数，检查是否是 np.object_ 类型
    return _is_dtype_type(arr_or_dtype, classes(np.object_))


# 检查一个数组是否是 1-D 的 pandas 稀疏数组
def is_sparse(arr) -> bool:
    """
    Check whether an array-like is a 1-D pandas sparse array.

    .. deprecated:: 2.1.0
        Use isinstance(dtype, pd.SparseDtype) instead.

    Check that the one-dimensional array-like is a pandas sparse array.
    Returns True if it is a pandas sparse array, not another type of
    sparse array.

    Parameters
    ----------
    arr : array-like
        Array-like to check.

    Returns
    -------
    bool
        Whether or not the array-like is a pandas sparse array.

    Examples
    --------
    Returns `True` if the parameter is a 1-D pandas sparse array.

    >>> from pandas.api.types import is_sparse
    >>> is_sparse(pd.arrays.SparseArray([0, 0, 1, 0]))
    True
    >>> is_sparse(pd.Series(pd.arrays.SparseArray([0, 0, 1, 0])))
    True

    Returns `False` if the parameter is not sparse.

    >>> is_sparse(np.array([0, 0, 1, 0]))
    False
    >>> is_sparse(pd.Series([0, 1, 0, 0]))
    False

    Returns `False` if the parameter is not a pandas sparse array.

    >>> from scipy.sparse import bsr_matrix
    >>> is_sparse(bsr_matrix([0, 1, 0, 0]))
    False

    Returns `False` if the parameter has more than one dimension.
    """
    # 发出警告，此函数已弃用，建议使用 isinstance(dtype, pd.SparseDtype)
    warnings.warn(
        "is_sparse is deprecated and will be removed in a future "
        "version. Check `isinstance(dtype, pd.SparseDtype)` instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # 获取参数的 dtype 属性，或者直接获取参数的值
    dtype = getattr(arr, "dtype", arr)
    # 检查 dtype 是否是 SparseDtype 类型
    return isinstance(dtype, SparseDtype)


# 检查一个数组是否是 scipy.sparse.spmatrix 实例
def is_scipy_sparse(arr) -> bool:
    """
    Check whether an array-like is a scipy.sparse.spmatrix instance.

    Parameters
    ----------
    arr : array-like
        The array-like to check.

    Returns
    -------
    boolean
        Whether or not the array-like is a scipy.sparse.spmatrix instance.

    Notes
    -----
    If scipy is not installed, this function will always return False.

    Examples
    --------
    >>> from scipy.sparse import bsr_matrix
    >>> is_scipy_sparse(bsr_matrix([1, 2, 3]))
    True
    >>> is_scipy_sparse(pd.arrays.SparseArray([1, 2, 3]))
    False
    """
    # 声明全局变量 _is_scipy_sparse
    global _is_scipy_sparse
    # 检查是否已经确定了_scipy_sparse的状态，如果尚未确定则执行以下代码
    if _is_scipy_sparse is None:
        # 尝试导入scipy.sparse模块中的issparse函数并赋值给_is_scipy_sparse变量
        try:
            from scipy.sparse import issparse as _is_scipy_sparse
        # 如果导入失败，则定义一个匿名函数使_is_scipy_sparse返回False
        except ImportError:
            _is_scipy_sparse = lambda _: False

    # 确保_is_scipy_sparse已经被赋值，不为None
    assert _is_scipy_sparse is not None
    # 调用_is_scipy_sparse函数，传入arr作为参数，并返回其结果
    return _is_scipy_sparse(arr)
# 检查数组样式或数据类型是否为 datetime64 类型的函数
def is_datetime64_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the datetime64 dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the datetime64 dtype.

    Examples
    --------
    >>> from pandas.api.types import is_datetime64_dtype
    >>> is_datetime64_dtype(object)
    False
    >>> is_datetime64_dtype(np.datetime64)
    True
    >>> is_datetime64_dtype(np.array([], dtype=int))
    False
    >>> is_datetime64_dtype(np.array([], dtype=np.datetime64))
    True
    >>> is_datetime64_dtype([1, 2, 3])
    False
    """
    if isinstance(arr_or_dtype, np.dtype):
        # GH#33400 fastpath for dtype object
        # 检查 dtype 对象是否为 datetime64 类型
        return arr_or_dtype.kind == "M"
    # 否则调用辅助函数 _is_dtype_type，检查 array-like 是否为 datetime64 类型
    return _is_dtype_type(arr_or_dtype, classes(np.datetime64))


# 检查数组样式或数据类型是否为 DatetimeTZDtype 类型的函数
def is_datetime64tz_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of a DatetimeTZDtype dtype.

    .. deprecated:: 2.1.0
        Use isinstance(dtype, pd.DatetimeTZDtype) instead.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of a DatetimeTZDtype dtype.

    Examples
    --------
    >>> from pandas.api.types import is_datetime64tz_dtype
    >>> is_datetime64tz_dtype(object)
    False
    >>> is_datetime64tz_dtype([1, 2, 3])
    False
    >>> is_datetime64tz_dtype(pd.DatetimeIndex([1, 2, 3]))  # tz-naive
    False
    >>> is_datetime64tz_dtype(pd.DatetimeIndex([1, 2, 3], tz="US/Eastern"))
    True

    >>> from pandas.core.dtypes.dtypes import DatetimeTZDtype
    >>> dtype = DatetimeTZDtype("ns", tz="US/Eastern")
    >>> s = pd.Series([], dtype=dtype)
    >>> is_datetime64tz_dtype(dtype)
    True
    >>> is_datetime64tz_dtype(s)
    True
    """
    # GH#52607
    # 发出警告，表明该函数即将被移除，建议使用 isinstance(dtype, pd.DatetimeTZDtype)
    warnings.warn(
        "is_datetime64tz_dtype is deprecated and will be removed in a future "
        "version. Check `isinstance(dtype, pd.DatetimeTZDtype)` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(arr_or_dtype, DatetimeTZDtype):
        # GH#33400 fastpath for dtype object
        # 如果 arr_or_dtype 是 DatetimeTZDtype 对象，则返回 True
        # GH 34986
        return True

    if arr_or_dtype is None:
        return False
    # 否则调用 DatetimeTZDtype 的静态方法 is_dtype，检查 arr_or_dtype 是否为 DatetimeTZDtype 类型
    return DatetimeTZDtype.is_dtype(arr_or_dtype)


# 检查数组样式或数据类型是否为 timedelta64 类型的函数
def is_timedelta64_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the timedelta64 dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the timedelta64 dtype.

    See Also
    --------
    api.types.is_timedelta64_ns_dtype : Check whether the provided array or dtype is
        of the timedelta64[ns] dtype.
    """
    # 检查 arr_or_dtype 是否为 timedelta64 类型
    return _is_dtype_type(arr_or_dtype, classes(np.timedelta64))
    # 检查输入的数组类或数据类型是否为时间增量类型
    api.types.is_period_dtype : Check whether an array-like or dtype is of the
        Period dtype.

    Examples
    --------
    >>> from pandas.core.dtypes.common import is_timedelta64_dtype
    >>> is_timedelta64_dtype(object)
    False
    >>> is_timedelta64_dtype(np.timedelta64)
    True
    >>> is_timedelta64_dtype([1, 2, 3])
    False
    >>> is_timedelta64_dtype(pd.Series([], dtype="timedelta64[ns]"))
    True
    >>> is_timedelta64_dtype("0 days")
    False
    """
    # 如果输入参数 arr_or_dtype 是 numpy 的数据类型对象
    if isinstance(arr_or_dtype, np.dtype):
        # 对于数据类型对象，通过直接比较 kind 属性是否为 'm' 来快速判断是否为 timedelta64 类型
        return arr_or_dtype.kind == "m"

    # 否则，调用 _is_dtype_type 函数进行详细检查，判断是否属于 timedelta64 类型
    return _is_dtype_type(arr_or_dtype, classes(np.timedelta64))
def is_period_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the Period dtype.

    .. deprecated:: 2.2.0
        Use isinstance(dtype, pd.Period) instead.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the Period dtype.

    Examples
    --------
    >>> from pandas.core.dtypes.common import is_period_dtype
    >>> is_period_dtype(object)
    False
    >>> is_period_dtype(pd.PeriodDtype(freq="D"))
    True
    >>> is_period_dtype([1, 2, 3])
    False
    >>> is_period_dtype(pd.Period("2017-01-01"))
    False
    >>> is_period_dtype(pd.PeriodIndex([], freq="Y"))
    True
    """
    # 发出警告，提醒函数已经废弃，建议使用 isinstance(dtype, pd.PeriodDtype) 替代
    warnings.warn(
        "is_period_dtype is deprecated and will be removed in a future version. "
        "Use `isinstance(dtype, pd.PeriodDtype)` instead",
        DeprecationWarning,
        stacklevel=2,
    )
    # 如果 arr_or_dtype 是 ExtensionDtype 类型，则快速返回其类型是否为 Period
    if isinstance(arr_or_dtype, ExtensionDtype):
        return arr_or_dtype.type is Period

    # 如果 arr_or_dtype 为 None，则返回 False
    if arr_or_dtype is None:
        return False
    # 否则调用 PeriodDtype 类的 is_dtype 方法，判断 arr_or_dtype 是否为 PeriodDtype 类型
    return PeriodDtype.is_dtype(arr_or_dtype)


def is_interval_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the Interval dtype.

    .. deprecated:: 2.2.0
        Use isinstance(dtype, pd.IntervalDtype) instead.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the Interval dtype.

    Examples
    --------
    >>> from pandas.core.dtypes.common import is_interval_dtype
    >>> is_interval_dtype(object)
    False
    >>> is_interval_dtype(pd.IntervalDtype())
    True
    >>> is_interval_dtype([1, 2, 3])
    False
    >>> interval = pd.Interval(1, 2, closed="right")
    >>> is_interval_dtype(interval)
    False
    >>> is_interval_dtype(pd.IntervalIndex([interval]))
    True
    """
    # 发出警告，提醒函数已经废弃，建议使用 isinstance(dtype, pd.IntervalDtype) 替代
    warnings.warn(
        "is_interval_dtype is deprecated and will be removed in a future version. "
        "Use `isinstance(dtype, pd.IntervalDtype)` instead",
        DeprecationWarning,
        stacklevel=2,
    )
    # 如果 arr_or_dtype 是 ExtensionDtype 类型，则快速返回其类型是否为 Interval
    if isinstance(arr_or_dtype, ExtensionDtype):
        return arr_or_dtype.type is Interval

    # 如果 arr_or_dtype 为 None，则返回 False
    if arr_or_dtype is None:
        return False
    # 否则调用 IntervalDtype 类的 is_dtype 方法，判断 arr_or_dtype 是否为 IntervalDtype 类型
    return IntervalDtype.is_dtype(arr_or_dtype)
    # GH#52527
    # 引发警告，说明 is_categorical_dtype 函数即将被弃用，并建议使用 isinstance(dtype, pd.CategoricalDtype) 替代
    warnings.warn(
        "is_categorical_dtype is deprecated and will be removed in a future "
        "version. Use isinstance(dtype, pd.CategoricalDtype) instead",
        DeprecationWarning,
        stacklevel=2,
    )
    
    # 检查给定的对象或数据类型是否为 ExtensionDtype 类型
    if isinstance(arr_or_dtype, ExtensionDtype):
        # 对于 ExtensionDtype 类型的快速路径，特别针对 dtype 为对象类型的情况
        return arr_or_dtype.name == "category"
    
    # 若 arr_or_dtype 为 None，则返回 False
    if arr_or_dtype is None:
        return False
    
    # 调用 CategoricalDtype 类的静态方法 is_dtype 判断 arr_or_dtype 是否为分类类型
    return CategoricalDtype.is_dtype(arr_or_dtype)
# 检查给定的 np.dtype 是否为字符串或对象类型
def is_string_or_object_np_dtype(dtype: np.dtype) -> bool:
    """
    Faster alternative to is_string_dtype, assumes we have a np.dtype object.
    """
    # 返回 True 如果 dtype 是对象类型或其种类包含在 "SU" 中
    return dtype == object or dtype.kind in "SU"


# 检查数组或 dtype 是否为字符串 dtype
def is_string_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of the string dtype.

    If an array is passed with an object dtype, the elements must be
    inferred as strings.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of the string dtype.

    Examples
    --------
    >>> from pandas.api.types import is_string_dtype
    >>> is_string_dtype(str)
    True
    >>> is_string_dtype(object)
    True
    >>> is_string_dtype(int)
    False
    >>> is_string_dtype(np.array(["a", "b"]))
    True
    >>> is_string_dtype(pd.Series([1, 2]))
    False
    >>> is_string_dtype(pd.Series([1, 2], dtype=object))
    False
    """
    # 如果 arr_or_dtype 有 dtype 属性且其种类是对象（'O'），则调用 is_all_strings(arr_or_dtype) 进行进一步检查
    if hasattr(arr_or_dtype, "dtype") and _get_dtype(arr_or_dtype).kind == "O":
        return is_all_strings(arr_or_dtype)

    # 定义检查 dtype 的条件函数
    def condition(dtype) -> bool:
        # 调用 is_string_or_object_np_dtype 检查 dtype 是否为字符串或对象类型
        if is_string_or_object_np_dtype(dtype):
            return True
        try:
            # 尝试比较 dtype 是否为 "string"
            return dtype == "string"
        except TypeError:
            return False

    # 调用 _is_dtype 函数，使用定义的条件函数来判断 arr_or_dtype 是否符合条件
    return _is_dtype(arr_or_dtype, condition)


# 检查两个 dtype 是否相等
def is_dtype_equal(source, target) -> bool:
    """
    Check if two dtypes are equal.

    Parameters
    ----------
    source : The first dtype to compare
    target : The second dtype to compare

    Returns
    -------
    boolean
        Whether or not the two dtypes are equal.

    Examples
    --------
    >>> is_dtype_equal(int, float)
    False
    >>> is_dtype_equal("int", int)
    True
    >>> is_dtype_equal(object, "category")
    False
    >>> is_dtype_equal(CategoricalDtype(), "category")
    True
    >>> is_dtype_equal(DatetimeTZDtype(tz="UTC"), "datetime64")
    False
    """
    # 如果 target 是字符串
    if isinstance(target, str):
        # 如果 source 不是字符串，尝试获取 source 的 dtype，并检查是否为 ExtensionDtype 类型并与 target 相等
        if not isinstance(source, str):
            try:
                src = _get_dtype(source)
                if isinstance(src, ExtensionDtype):
                    return src == target
            except (TypeError, AttributeError, ImportError):
                return False
    # 如果 source 是字符串，则反转参数并递归调用 is_dtype_equal 函数
    elif isinstance(source, str):
        return is_dtype_equal(target, source)

    try:
        # 获取 source 和 target 的 dtype，并比较它们是否相等
        source = _get_dtype(source)
        target = _get_dtype(target)
        return source == target
    except (TypeError, AttributeError, ImportError):
        # 出现异常时返回 False，通常在无效比较时发生，比如 object == category
        return False


# 检查数组或 dtype 是否为整数 dtype
def is_integer_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of an integer dtype.

    Unlike in `is_any_int_dtype`, timedelta64 instances will return False.
    """
    # 实现代码省略，因为该函数未完整提供
    """
    The nullable Integer dtypes (e.g. pandas.Int64Dtype) are also considered
    as integer by this function.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of an integer dtype and
        not an instance of timedelta64.

    Examples
    --------
    >>> from pandas.api.types import is_integer_dtype
    >>> is_integer_dtype(str)
    False
    >>> is_integer_dtype(int)
    True
    >>> is_integer_dtype(float)
    False
    >>> is_integer_dtype(np.uint64)
    True
    >>> is_integer_dtype("int8")
    True
    >>> is_integer_dtype("Int8")
    True
    >>> is_integer_dtype(pd.Int8Dtype)
    True
    >>> is_integer_dtype(np.datetime64)
    False
    >>> is_integer_dtype(np.timedelta64)
    False
    >>> is_integer_dtype(np.array(["a", "b"]))
    False
    >>> is_integer_dtype(pd.Series([1, 2]))
    True
    >>> is_integer_dtype(np.array([], dtype=np.timedelta64))
    False
    >>> is_integer_dtype(pd.Index([1, 2.0]))  # float
    False
    """
    # Check if arr_or_dtype is of an integer dtype excluding timedelta64
    return _is_dtype_type(
        arr_or_dtype, _classes_and_not_datetimelike(np.integer)
    ) or _is_dtype(
        arr_or_dtype, lambda typ: isinstance(typ, ExtensionDtype) and typ.kind in "iu"
    )
# 检查给定的数组或数据类型是否为有符号整数数据类型。

# 参数 arr_or_dtype: 要检查的数组或数据类型。
# 返回值: 如果数组或数据类型是有符号整数数据类型且不是 timedelta64 的实例，则返回 True。

def is_signed_integer_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of a signed integer dtype.

    Unlike in `is_any_int_dtype`, timedelta64 instances will return False.

    The nullable Integer dtypes (e.g. pandas.Int64Dtype) are also considered
    as integer by this function.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a signed integer dtype
        and not an instance of timedelta64.

    Examples
    --------
    >>> from pandas.core.dtypes.common import is_signed_integer_dtype
    >>> is_signed_integer_dtype(str)
    False
    >>> is_signed_integer_dtype(int)
    True
    >>> is_signed_integer_dtype(float)
    False
    >>> is_signed_integer_dtype(np.uint64)  # unsigned
    False
    >>> is_signed_integer_dtype("int8")
    True
    >>> is_signed_integer_dtype("Int8")
    True
    >>> is_signed_integer_dtype(pd.Int8Dtype)
    True
    >>> is_signed_integer_dtype(np.datetime64)
    False
    >>> is_signed_integer_dtype(np.timedelta64)
    False
    >>> is_signed_integer_dtype(np.array(["a", "b"]))
    False
    >>> is_signed_integer_dtype(pd.Series([1, 2]))
    True
    >>> is_signed_integer_dtype(np.array([], dtype=np.timedelta64))
    False
    >>> is_signed_integer_dtype(pd.Index([1, 2.0]))  # float
    False
    >>> is_signed_integer_dtype(np.array([1, 2], dtype=np.uint32))  # unsigned
    False
    """
    
    # 调用 _is_dtype_type 函数，检查是否是有符号整数类型
    return _is_dtype_type(
        arr_or_dtype, _classes_and_not_datetimelike(np.signedinteger)
    ) or _is_dtype(
        arr_or_dtype, lambda typ: isinstance(typ, ExtensionDtype) and typ.kind == "i"
    )


# 检查给定的数组或数据类型是否为无符号整数数据类型。

# 参数 arr_or_dtype: 要检查的数组或数据类型。
# 返回值: 如果数组或数据类型是无符号整数数据类型，则返回 True。

def is_unsigned_integer_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of an unsigned integer dtype.

    The nullable Integer dtypes (e.g. pandas.UInt64Dtype) are also
    considered as integer by this function.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of an unsigned integer dtype.

    Examples
    --------
    >>> from pandas.api.types import is_unsigned_integer_dtype
    >>> is_unsigned_integer_dtype(str)
    False
    >>> is_unsigned_integer_dtype(int)  # signed
    False
    >>> is_unsigned_integer_dtype(float)
    False
    >>> is_unsigned_integer_dtype(np.uint64)
    True
    >>> is_unsigned_integer_dtype("uint8")
    True
    >>> is_unsigned_integer_dtype("UInt8")
    True
    >>> is_unsigned_integer_dtype(pd.UInt8Dtype)
    True
    >>> is_unsigned_integer_dtype(np.array(["a", "b"]))
    False
    >>> is_unsigned_integer_dtype(pd.Series([1, 2]))  # signed
    False
    >>> is_unsigned_integer_dtype(pd.Index([1, 2.0]))  # float
    False
    >>> is_unsigned_integer_dtype(np.array([1, 2], dtype=np.uint32))
    True
    """
    # 调用 _is_dtype_type 函数，检查 arr_or_dtype 是否为给定类型或非日期时间类的无符号整数类型
    return _is_dtype_type(
        arr_or_dtype, _classes_and_not_datetimelike(np.unsignedinteger)
    ) or 
    # 调用 _is_dtype 函数，检查 arr_or_dtype 是否为指定的扩展数据类型，并且其类型为无符号整数
    _is_dtype(
        arr_or_dtype, lambda typ: isinstance(typ, ExtensionDtype) and typ.kind == "u"
    )
# 检查提供的数组或数据类型是否为 int64 数据类型。
# 
# .. deprecated:: 2.1.0
# 
#    is_int64_dtype 已弃用，并将在未来版本中移除。请使用 dtype == np.int64 替代。
# 
# Parameters
# ----------
# arr_or_dtype : array-like or dtype
#     要检查的数组或数据类型。
# 
# Returns
# -------
# boolean
#     数组或数据类型是否为 int64 数据类型的布尔值。
# 
# Notes
# -----
# 根据系统架构，`is_int64_dtype(int)` 的返回值如果操作系统使用64位整数则为 True，使用32位整数则为 False。
# 
# Examples
# --------
# >>> from pandas.api.types import is_int64_dtype
# >>> is_int64_dtype(str)  # doctest: +SKIP
# False
# >>> is_int64_dtype(np.int32)  # doctest: +SKIP
# False
# >>> is_int64_dtype(np.int64)  # doctest: +SKIP
# True
# >>> is_int64_dtype("int8")  # doctest: +SKIP
# False
# >>> is_int64_dtype("Int8")  # doctest: +SKIP
# False
# >>> is_int64_dtype(pd.Int64Dtype)  # doctest: +SKIP
# True
# >>> is_int64_dtype(float)  # doctest: +SKIP
# False
# >>> is_int64_dtype(np.uint64)  # unsigned  # doctest: +SKIP
# False
# >>> is_int64_dtype(np.array(["a", "b"]))  # doctest: +SKIP
# False
# >>> is_int64_dtype(np.array([1, 2], dtype=np.int64))  # doctest: +SKIP
# True
# >>> is_int64_dtype(pd.Index([1, 2.0]))  # float  # doctest: +SKIP
# False
# >>> is_int64_dtype(np.array([1, 2], dtype=np.uint32))  # unsigned  # doctest: +SKIP
# False
def is_int64_dtype(arr_or_dtype) -> bool:
    # GH#52564
    warnings.warn(
        "is_int64_dtype 已弃用并将在未来版本中移除。请使用 dtype == np.int64 替代。",
        DeprecationWarning,
        stacklevel=2,
    )
    return _is_dtype_type(arr_or_dtype, classes(np.int64))


# 检查提供的数组或数据类型是否为 datetime64 数据类型。
# 
# Parameters
# ----------
# arr_or_dtype : array-like or dtype
#     要检查的数组或数据类型。
# 
# Returns
# -------
# bool
#     数组或数据类型是否为 datetime64 数据类型的布尔值。
# 
# See Also
# --------
# api.types.is_datetime64_dtype : 检查数组或数据类型是否为 datetime64 数据类型。
# api.is_datetime64_ns_dtype : 检查提供的数组或数据类型是否为 datetime64[ns] 数据类型。
# api.is_datetime64tz_dtype : 检查数组或数据类型是否为 DatetimeTZDtype 数据类型。
# 
# Examples
# --------
# >>> from pandas.api.types import is_datetime64_any_dtype
# >>> from pandas.core.dtypes.dtypes import DatetimeTZDtype
# >>> is_datetime64_any_dtype(str)
# False
# >>> is_datetime64_any_dtype(int)
# False
# >>> is_datetime64_any_dtype(np.datetime64)  # can be tz-naive
# True
# >>> is_datetime64_any_dtype(DatetimeTZDtype("ns", "US/Eastern"))
# True
def is_datetime64_any_dtype(arr_or_dtype) -> bool:
    # 检查给定的 NumPy 数组或数据类型是否为 datetime64 类型的任何一种
    def is_datetime64_any_dtype(arr_or_dtype):
        if isinstance(arr_or_dtype, (np.dtype, ExtensionDtype)):
            # 对于 dtype 或 ExtensionDtype 对象的快速路径
            return arr_or_dtype.kind == "M"
    
        if arr_or_dtype is None:
            return False
    
        try:
            # 尝试获取 arr_or_dtype 的数据类型
            tipo = _get_dtype(arr_or_dtype)
        except TypeError:
            return False
        
        # 检查数据类型是否为 numpy 的 datetime64 类型，或者是否为 DatetimeTZDtype 类型，或者是否为 ExtensionDtype 且其 kind 为 'M'
        return (
            lib.is_np_dtype(tipo, "M")
            or isinstance(tipo, DatetimeTZDtype)
            or (isinstance(tipo, ExtensionDtype) and tipo.kind == "M")
        )
# 检查提供的数组或数据类型是否为 datetime64[ns] 数据类型。
def is_datetime64_ns_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of the datetime64[ns] dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    bool
        Whether or not the array or dtype is of the datetime64[ns] dtype.

    Examples
    --------
    >>> from pandas.api.types import is_datetime64_ns_dtype
    >>> from pandas.core.dtypes.dtypes import DatetimeTZDtype
    >>> is_datetime64_ns_dtype(str)
    False
    >>> is_datetime64_ns_dtype(int)
    False
    >>> is_datetime64_ns_dtype(np.datetime64)  # no unit
    False
    >>> is_datetime64_ns_dtype(DatetimeTZDtype("ns", "US/Eastern"))
    True
    >>> is_datetime64_ns_dtype(np.array(["a", "b"]))
    False
    >>> is_datetime64_ns_dtype(np.array([1, 2]))
    False
    >>> is_datetime64_ns_dtype(np.array([], dtype="datetime64"))  # no unit
    False
    >>> is_datetime64_ns_dtype(np.array([], dtype="datetime64[ps]"))  # wrong unit
    False
    >>> is_datetime64_ns_dtype(pd.DatetimeIndex([1, 2, 3], dtype="datetime64[ns]"))
    True
    """
    # 如果 arr_or_dtype 为 None，则返回 False
    if arr_or_dtype is None:
        return False
    try:
        # 获取 arr_or_dtype 的数据类型
        tipo = _get_dtype(arr_or_dtype)
    except TypeError:
        # 如果获取数据类型时出现 TypeError，则返回 False
        return False
    # 检查数据类型是否为 datetime64[ns] 或者是 DatetimeTZDtype 并且单位为 "ns"
    return tipo == DT64NS_DTYPE or (
        isinstance(tipo, DatetimeTZDtype) and tipo.unit == "ns"
    )


# 检查提供的数组或数据类型是否为 timedelta64[ns] 数据类型。
def is_timedelta64_ns_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of the timedelta64[ns] dtype.

    This is a very specific dtype, so generic ones like `np.timedelta64`
    will return False if passed into this function.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of the timedelta64[ns] dtype.

    Examples
    --------
    >>> from pandas.core.dtypes.common import is_timedelta64_ns_dtype
    >>> is_timedelta64_ns_dtype(np.dtype("m8[ns]"))
    True
    >>> is_timedelta64_ns_dtype(np.dtype("m8[ps]"))  # Wrong frequency
    False
    >>> is_timedelta64_ns_dtype(np.array([1, 2], dtype="m8[ns]"))
    True
    >>> is_timedelta64_ns_dtype(np.array([1, 2], dtype=np.timedelta64))
    False
    """
    # 使用 _is_dtype 函数检查 arr_or_dtype 是否为 timedelta64[ns] 数据类型
    return _is_dtype(arr_or_dtype, lambda dtype: dtype == TD64NS_DTYPE)


# 用于消除 numpy 弃用警告，参见 GH#29553
def is_numeric_v_string_like(a: ArrayLike, b) -> bool:
    """
    Check if we are comparing a string-like object to a numeric ndarray.
    NumPy doesn't like to compare such objects, especially numeric arrays
    and scalar string-likes.

    Parameters
    ----------
    a : array-like, scalar
        The first object to check.
    b : array-like, scalar
        The second object to check.

    Returns
    -------
    boolean
        Whether we return a comparing a string-like object to a numeric array.

    Examples
    --------
    # 检查变量 a 是否为 NumPy 数组
    is_a_array = isinstance(a, np.ndarray)
    
    # 检查变量 b 是否为 NumPy 数组
    is_b_array = isinstance(b, np.ndarray)
    
    # 检查变量 a 是否为数值类型的 NumPy 数组（包括整数、浮点数、复数、布尔型）
    is_a_numeric_array = is_a_array and a.dtype.kind in ("u", "i", "f", "c", "b")
    
    # 检查变量 b 是否为数值类型的 NumPy 数组（包括整数、浮点数、复数、布尔型）
    is_b_numeric_array = is_b_array and b.dtype.kind in ("u", "i", "f", "c", "b")
    
    # 检查变量 a 是否为字符串类型的 NumPy 数组（包括字节字符串和 Unicode 字符串）
    is_a_string_array = is_a_array and a.dtype.kind in ("S", "U")
    
    # 检查变量 b 是否为字符串类型的 NumPy 数组（包括字节字符串和 Unicode 字符串）
    is_b_string_array = is_b_array and b.dtype.kind in ("S", "U")
    
    # 检查变量 b 是否为字符串标量
    is_b_scalar_string_like = not is_b_array and isinstance(b, str)
    
    # 返回根据变量类型判断是否为数值类型与字符串类似的布尔值
    return (
        (is_a_numeric_array and is_b_scalar_string_like)
        or (is_a_numeric_array and is_b_string_array)
        or (is_b_numeric_array and is_a_string_array)
    )
# 检查给定的数据类型是否应该转换为 int64 类型。
def needs_i8_conversion(dtype: DtypeObj | None) -> bool:
    """
    Check whether the dtype should be converted to int64.

    Dtype "needs" such a conversion if the dtype is of a datetime-like dtype

    Parameters
    ----------
    dtype : np.dtype, ExtensionDtype, or None
        The data type to check.

    Returns
    -------
    boolean
        Whether or not the dtype should be converted to int64.

    Examples
    --------
    >>> needs_i8_conversion(str)
    False
    >>> needs_i8_conversion(np.int64)
    False
    >>> needs_i8_conversion(np.datetime64)
    False
    >>> needs_i8_conversion(np.dtype(np.datetime64))
    True
    >>> needs_i8_conversion(np.array(["a", "b"]))
    False
    >>> needs_i8_conversion(pd.Series([1, 2]))
    False
    >>> needs_i8_conversion(pd.Series([], dtype="timedelta64[ns]"))
    False
    >>> needs_i8_conversion(pd.DatetimeIndex([1, 2, 3], tz="US/Eastern"))
    False
    >>> needs_i8_conversion(pd.DatetimeIndex([1, 2, 3], tz="US/Eastern").dtype)
    True
    """
    # 如果 dtype 是 np.dtype 类型，则检查其种类是否为 'm' 或 'M'
    if isinstance(dtype, np.dtype):
        return dtype.kind in "mM"
    # 否则，检查是否是 PeriodDtype 或 DatetimeTZDtype 类型
    return isinstance(dtype, (PeriodDtype, DatetimeTZDtype))


# 检查提供的数组或数据类型是否为数值型数据类型。
def is_numeric_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of a numeric dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a numeric dtype.

    Examples
    --------
    >>> from pandas.api.types import is_numeric_dtype
    >>> is_numeric_dtype(str)
    False
    >>> is_numeric_dtype(int)
    True
    >>> is_numeric_dtype(float)
    True
    >>> is_numeric_dtype(np.uint64)
    True
    >>> is_numeric_dtype(np.datetime64)
    False
    >>> is_numeric_dtype(np.timedelta64)
    False
    >>> is_numeric_dtype(np.array(["a", "b"]))
    False
    >>> is_numeric_dtype(pd.Series([1, 2]))
    True
    >>> is_numeric_dtype(pd.Index([1, 2.0]))
    True
    >>> is_numeric_dtype(np.array([], dtype=np.timedelta64))
    False
    """
    # 调用内部函数 _is_dtype_type，检查是否为数值类型或布尔类型
    return _is_dtype_type(
        arr_or_dtype, _classes_and_not_datetimelike(np.number, np.bool_)
    ) or _is_dtype(
        arr_or_dtype, lambda typ: isinstance(typ, ExtensionDtype) and typ._is_numeric
    )


# 检查提供的数组或数据类型是否为任何实数数值型数据类型。
def is_any_real_numeric_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of a real number dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a real number dtype.

    Examples
    --------
    >>> from pandas.api.types import is_any_real_numeric_dtype
    >>> is_any_real_numeric_dtype(int)
    True
    >>> is_any_real_numeric_dtype(float)
    True
    >>> is_any_real_numeric_dtype(object)
    False
    >>> is_any_real_numeric_dtype(str)
    False
    >>> is_any_real_numeric_dtype(complex(1, 2))
    False
    >>> is_any_real_numeric_dtype(bool)
    False
    """
    # 检查是否为数值类型中的实数类型，使用 _is_dtype_type 函数
    return _is_dtype_type(
        arr_or_dtype,
        _classes_and_not_datetimelike(np.floating, np.integer, np.bool_),
    )
    """
    检查输入的数组或数据类型是否为数值类型，并且不是复数类型和布尔类型。
    返回一个布尔值，表示是否满足条件。
    """
    return (
        is_numeric_dtype(arr_or_dtype)  # 检查是否为数值类型
        and not is_complex_dtype(arr_or_dtype)  # 排除复数类型
        and not is_bool_dtype(arr_or_dtype)  # 排除布尔类型
    )
# 检查提供的数组或数据类型是否为浮点数数据类型

def is_float_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of a float dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a float dtype.

    Examples
    --------
    >>> from pandas.api.types import is_float_dtype
    >>> is_float_dtype(str)
    False
    >>> is_float_dtype(int)
    False
    >>> is_float_dtype(float)
    True
    >>> is_float_dtype(np.array(["a", "b"]))
    False
    >>> is_float_dtype(pd.Series([1, 2]))
    False
    >>> is_float_dtype(pd.Index([1, 2.0]))
    True
    """
    # 调用内部函数 _is_dtype_type 检查是否为浮点类型之一
    return _is_dtype_type(arr_or_dtype, classes(np.floating)) or _is_dtype(
        arr_or_dtype, lambda typ: isinstance(typ, ExtensionDtype) and typ.kind in "f"
    )


# 检查提供的数组或数据类型是否为布尔数据类型

def is_bool_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of a boolean dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a boolean dtype.

    Notes
    -----
    An ExtensionArray is considered boolean when the ``_is_boolean``
    attribute is set to True.

    Examples
    --------
    >>> from pandas.api.types import is_bool_dtype
    >>> is_bool_dtype(str)
    False
    >>> is_bool_dtype(int)
    False
    >>> is_bool_dtype(bool)
    True
    >>> is_bool_dtype(np.bool_)
    True
    >>> is_bool_dtype(np.array(["a", "b"]))
    False
    >>> is_bool_dtype(pd.Series([1, 2]))
    False
    >>> is_bool_dtype(np.array([True, False]))
    True
    >>> is_bool_dtype(pd.Categorical([True, False]))
    True
    >>> is_bool_dtype(pd.arrays.SparseArray([True, False]))
    True
    """
    # 如果参数为 None，则返回 False
    if arr_or_dtype is None:
        return False
    try:
        dtype = _get_dtype(arr_or_dtype)
    except (TypeError, ValueError):
        return False

    # 如果数据类型为 CategoricalDtype，则检查其 categories 属性
    if isinstance(dtype, CategoricalDtype):
        arr_or_dtype = dtype.categories
        # 现在我们使用 Index 的特殊定义

    # 如果 arr_or_dtype 是 ABCIndex 的实例
    if isinstance(arr_or_dtype, ABCIndex):
        # 允许 Index[object]，即全部为布尔值或 Index["boolean"]
        if arr_or_dtype.inferred_type == "boolean":
            if not is_bool_dtype(arr_or_dtype.dtype):
                # GH#52680
                # 发出警告，表明将来版本中此操作将返回 False
                warnings.warn(
                    "The behavior of is_bool_dtype with an object-dtype Index "
                    "of bool objects is deprecated. In a future version, "
                    "this will return False. Cast the Index to a bool dtype instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            return True
        return False
    # 如果数据类型是 ExtensionDtype，则返回其 _is_boolean 属性
    elif isinstance(dtype, ExtensionDtype):
        return getattr(dtype, "_is_boolean", False)

    # 检查 dtype 类型是否为 np.bool_ 的子类
    return issubclass(dtype.type, np.bool_)


def is_1d_only_ea_dtype(dtype: DtypeObj | None) -> bool:
    """
    
    # 检查给定的数据类型是否为扩展数据类型，但排除 DatetimeTZDtype。
    """
    检查变量 dtype 是否属于 ExtensionDtype 类型，并且不是 DatetimeTZDtype 类型。
    返回结果为真（True）或假（False）。
    """
    return isinstance(dtype, ExtensionDtype) and not dtype._supports_2d
def is_extension_array_dtype(arr_or_dtype) -> bool:
    """
    Check if an object is a pandas extension array type.

    See the :ref:`Use Guide <extending.extension-types>` for more.

    Parameters
    ----------
    arr_or_dtype : object
        For array-like input, the ``.dtype`` attribute will
        be extracted.

    Returns
    -------
    bool
        Whether the `arr_or_dtype` is an extension array type.

    Notes
    -----
    This checks whether an object implements the pandas extension
    array interface. In pandas, this includes:

    * Categorical
    * Sparse
    * Interval
    * Period
    * DatetimeArray
    * TimedeltaArray

    Third-party libraries may implement arrays or types satisfying
    this interface as well.

    Examples
    --------
    >>> from pandas.api.types import is_extension_array_dtype
    >>> arr = pd.Categorical(["a", "b"])
    >>> is_extension_array_dtype(arr)
    True
    >>> is_extension_array_dtype(arr.dtype)
    True

    >>> arr = np.array(["a", "b"])
    >>> is_extension_array_dtype(arr.dtype)
    False
    """

    # Extract the dtype from arr_or_dtype, if possible
    dtype = getattr(arr_or_dtype, "dtype", arr_or_dtype)
    
    # Check if the dtype is an instance of ExtensionDtype
    if isinstance(dtype, ExtensionDtype):
        return True
    # If dtype is an instance of np.dtype, it's not an extension array type
    elif isinstance(dtype, np.dtype):
        return False
    # Otherwise, check if dtype is registered as an extension type
    else:
        return registry.find(dtype) is not None


def is_ea_or_datetimelike_dtype(dtype: DtypeObj | None) -> bool:
    """
    Check for ExtensionDtype, datetime64 dtype, or timedelta64 dtype.

    Notes
    -----
    Checks only for dtype objects, not dtype-castable strings or types.
    """
    
    # Check if dtype is an instance of ExtensionDtype or a datetime or timedelta dtype
    return isinstance(dtype, ExtensionDtype) or (lib.is_np_dtype(dtype, "mM"))


def is_complex_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of a complex dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a complex dtype.

    Examples
    --------
    >>> from pandas.api.types import is_complex_dtype
    >>> is_complex_dtype(str)
    False
    >>> is_complex_dtype(int)
    False
    >>> is_complex_dtype(np.complex128)
    True
    >>> is_complex_dtype(np.array(["a", "b"]))
    False
    >>> is_complex_dtype(pd.Series([1, 2]))
    False
    >>> is_complex_dtype(np.array([1 + 1j, 5]))
    True
    """
    
    # Check if arr_or_dtype is a complex dtype using _is_dtype_type function
    return _is_dtype_type(arr_or_dtype, classes(np.complexfloating))


def _is_dtype(arr_or_dtype, condition) -> bool:
    """
    Return true if the condition is satisfied for the arr_or_dtype.

    Parameters
    ----------
    arr_or_dtype : array-like, str, np.dtype, or ExtensionArrayType
        The array-like or dtype object whose dtype we want to extract.
    condition : callable[Union[np.dtype, ExtensionDtype]]

    Returns
    -------
    bool

    """
    
    # If arr_or_dtype is None, return False
    if arr_or_dtype is None:
        return False
    
    try:
        # Get the dtype from arr_or_dtype using _get_dtype function
        dtype = _get_dtype(arr_or_dtype)
    except (TypeError, ValueError):
        # Return False if there's an error getting dtype
        return False
    # 返回 condition 函数的结果，该函数以 dtype 作为参数
    return condition(dtype)
def _get_dtype(arr_or_dtype) -> DtypeObj:
    """
    Get the dtype instance associated with an array
    or dtype object.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array-like or dtype object whose dtype we want to extract.

    Returns
    -------
    obj_dtype : The extract dtype instance from the
                passed in array or dtype object.

    Raises
    ------
    TypeError : The passed in object is None.
    """
    # 检查传入的 arr_or_dtype 是否为 None，如果是则抛出 TypeError 异常
    if arr_or_dtype is None:
        raise TypeError("Cannot deduce dtype from null object")

    # 快速路径：如果 arr_or_dtype 已经是 np.dtype 类型，则直接返回它
    if isinstance(arr_or_dtype, np.dtype):
        return arr_or_dtype
    # 如果 arr_or_dtype 是一个类型对象，则使用 np.dtype() 创建相应的 dtype 对象
    elif isinstance(arr_or_dtype, type):
        return np.dtype(arr_or_dtype)

    # 如果 arr_or_dtype 是一个类似数组的对象（即具有 dtype 属性），则获取其 dtype
    elif hasattr(arr_or_dtype, "dtype"):
        arr_or_dtype = arr_or_dtype.dtype

    # 使用 pandas_dtype() 函数推断 arr_or_dtype 的 dtype 并返回
    return pandas_dtype(arr_or_dtype)


def _is_dtype_type(arr_or_dtype, condition) -> bool:
    """
    Return true if the condition is satisfied for the arr_or_dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array-like or dtype object whose dtype we want to extract.
    condition : callable[Union[np.dtype, ExtensionDtypeType]]

    Returns
    -------
    bool : if the condition is satisfied for the arr_or_dtype
    """
    # 检查传入的 arr_or_dtype 是否为 None，如果是则根据 condition 判断返回结果
    if arr_or_dtype is None:
        return condition(type(None))

    # 快速路径：如果 arr_or_dtype 已经是 np.dtype 类型，则根据 condition 判断返回结果
    if isinstance(arr_or_dtype, np.dtype):
        return condition(arr_or_dtype.type)
    # 如果 arr_or_dtype 是一个类型对象
    elif isinstance(arr_or_dtype, type):
        # 如果 arr_or_dtype 是 ExtensionDtype 的子类，则获取其类型并继续判断
        if issubclass(arr_or_dtype, ExtensionDtype):
            arr_or_dtype = arr_or_dtype.type
        # 使用 np.dtype() 创建相应的 dtype 对象，并根据 condition 判断返回结果
        return condition(np.dtype(arr_or_dtype).type)

    # 如果 arr_or_dtype 是一个类似数组的对象（即具有 dtype 属性），则获取其 dtype
    if hasattr(arr_or_dtype, "dtype"):
        arr_or_dtype = arr_or_dtype.dtype

    # 如果 arr_or_dtype 不可能是 dtype 类型，则根据 condition 判断返回结果
    elif is_list_like(arr_or_dtype):
        return condition(type(None))

    try:
        # 使用 pandas_dtype() 函数推断 arr_or_dtype 的 dtype 并获取其类型
        tipo = pandas_dtype(arr_or_dtype).type
    except (TypeError, ValueError):
        # 如果推断失败且 arr_or_dtype 是标量，则根据 condition 判断返回结果
        if is_scalar(arr_or_dtype):
            return condition(type(None))

        return False

    # 根据 condition 判断推断出的 tipo 类型是否符合条件并返回结果
    return condition(tipo)


def infer_dtype_from_object(dtype) -> type:
    """
    Get a numpy dtype.type-style object for a dtype object.

    This methods also includes handling of the datetime64[ns] and
    datetime64[ns, TZ] objects.

    If no dtype can be found, we return ``object``.

    Parameters
    ----------
    dtype : dtype, type
        The dtype object whose numpy dtype.type-style
        object we want to extract.

    Returns
    -------
    type
    """
    # 如果 dtype 是一个类型对象且是 np.generic 的子类，则直接返回该对象
    if isinstance(dtype, type) and issubclass(dtype, np.generic):
        # 从 dtype 获取类型对象
        return dtype
    # 如果 dtype 是 np.dtype 或 ExtensionDtype 类型的实例
    elif isinstance(dtype, (np.dtype, ExtensionDtype)):
        # dtype 是一个对象
        try:
            # 验证是否为类似日期的 dtype
            _validate_date_like_dtype(dtype)
        except TypeError:
            # 如果不是日期类的 dtype，应该仍然通过验证
            pass
        if hasattr(dtype, "numpy_dtype"):
            # TODO: 应该正确实现这部分逻辑
            # https://github.com/pandas-dev/pandas/issues/52576
            return dtype.numpy_dtype.type
        # 返回 dtype 的类型
        return dtype.type

    try:
        # 尝试将 dtype 转换为 pandas 中的 dtype
        dtype = pandas_dtype(dtype)
    except TypeError:
        # 如果转换失败，什么都不做
        pass

    if isinstance(dtype, ExtensionDtype):
        # 如果 dtype 是 ExtensionDtype 类型的实例，返回其类型
        return dtype.type
    elif isinstance(dtype, str):
        # 如果 dtype 是字符串类型
        # TODO(jreback)
        # 应该废弃这些
        if dtype in ["datetimetz", "datetime64tz"]:
            # 如果 dtype 是 "datetimetz" 或 "datetime64tz"，返回 DatetimeTZDtype 类型
            return DatetimeTZDtype.type
        elif dtype in ["period"]:
            # 抛出未实现错误，暂时不支持 "period" 类型
            raise NotImplementedError

        if dtype in ["datetime", "timedelta"]:
            # 如果 dtype 是 "datetime" 或 "timedelta"，将其转换为 "datetime64" 或 "timedelta64"
            dtype += "64"
        try:
            # 尝试从 np 模块中获取对应 dtype 的推断类型
            return infer_dtype_from_object(getattr(np, dtype))
        except (AttributeError, TypeError):
            # 处理类似 _get_dtype(int) 的情况，即 Python 对象作为有效 dtype 的情况
            # （通常不包括用户定义的类型）
            #
            # TypeError 处理 'e' 这样的 float16 类型码
            # 进一步处理内部类型
            pass

    # 如果以上条件均不满足，返回从 np.dtype(dtype) 推断出的类型
    return infer_dtype_from_object(np.dtype(dtype))
# 检查传入的数据类型是否为类似日期的数据类型，如果不是则抛出错误。
def _validate_date_like_dtype(dtype) -> None:
    try:
        # 获取数据类型的日期信息
        typ = np.datetime_data(dtype)[0]
    except ValueError as e:
        # 如果获取失败，抛出类型错误
        raise TypeError(e) from e
    # 如果日期类型不在支持的范围内，则抛出值错误
    if typ not in ["generic", "ns"]:
        raise ValueError(
            f"{dtype.name!r} is too specific of a frequency, "
            f"try passing {dtype.type.__name__!r}"
        )


# 验证所有传入参数是否可哈希，否则抛出类型错误。
def validate_all_hashable(*args, error_name: str | None = None) -> None:
    if not all(is_hashable(arg) for arg in args):
        # 如果有不可哈希的参数，根据是否提供了错误名称抛出对应的类型错误
        if error_name:
            raise TypeError(f"{error_name} must be a hashable type")
        raise TypeError("All elements must be hashable")


# 将输入转换为 pandas 或 numpy 的数据类型对象。
def pandas_dtype(dtype) -> DtypeObj:
    # 如果输入为 numpy 数组，直接返回其数据类型
    if isinstance(dtype, np.ndarray):
        return dtype.dtype
    # 如果输入为 numpy 数据类型或者扩展数据类型，直接返回
    elif isinstance(dtype, (np.dtype, ExtensionDtype)):
        return dtype

    # 如果输入为注册的扩展类型，返回其对应的类型对象
    result = registry.find(dtype)
    if result is not None:
        if isinstance(result, type):
            # 对于某些情况下的警告，给出相应的警告信息
            warnings.warn(
                f"Instantiating {result.__name__} without any arguments."
                f"Pass a {result.__name__} instance to silence this warning.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
            result = result()
        return result

    # 尝试转换为 numpy 数据类型，如果失败则抛出类型错误
    try:
        with warnings.catch_warnings():
            # 处理 numpy 转换中的警告信息
            warnings.simplefilter("always", DeprecationWarning)
            npdtype = np.dtype(dtype)
    except SyntaxError as err:
        # 如果转换失败，抛出类型错误并指出数据类型不可理解
        raise TypeError(f"data type '{dtype}' not understood") from err

    # 任何无效的数据类型（例如 pd.Timestamp）都应该引发错误。
    # 检查 dtype 是否可哈希，并且是否属于以下几种值之一
    # 这些值包括 object、np.object_、"object"、"O"、"object_"
    # 如果是，则返回 npdtype，这是一个有效的数据类型
    if is_hashable(dtype) and dtype in [
        object,
        np.object_,
        "object",
        "O",
        "object_",
    ]:
        # 在此处检查 dtype 是否可哈希，以避免当 dtype 是数组时出现错误或 DeprecationWarning
        return npdtype
    # 如果 npdtype 的种类是 "O"，即对象类型，则引发 TypeError 异常
    elif npdtype.kind == "O":
        raise TypeError(f"dtype '{dtype}' not understood")

    # 返回 npdtype，即所检查和处理后的数据类型
    return npdtype
def is_all_strings(value: ArrayLike) -> bool:
    """
    Check if this is an array of strings that we should try parsing.

    Includes object-dtype ndarray containing all-strings, StringArray,
    and Categorical with all-string categories.
    Does not include numpy string dtypes.
    """
    # 获取输入值的数据类型
    dtype = value.dtype

    # 如果数据类型是 numpy 的 dtype 对象
    if isinstance(dtype, np.dtype):
        # 如果数组长度为 0，则考虑是否为 object 类型
        if len(value) == 0:
            return dtype == np.dtype("object")
        else:
            # 否则，检查是否是字符串数组，跳过 NaN 值
            return dtype == np.dtype("object") and lib.is_string_array(
                np.asarray(value), skipna=False
            )
    # 如果数据类型是 CategoricalDtype
    elif isinstance(dtype, CategoricalDtype):
        # 检查分类类型的推断类型是否为字符串
        return dtype.categories.inferred_type == "string"
    
    # 其他情况下，检查数据类型是否为字符串
    return dtype == "string"


__all__ = [
    "classes",
    "DT64NS_DTYPE",
    "ensure_float64",
    "ensure_python_int",
    "ensure_str",
    "infer_dtype_from_object",
    "INT64_DTYPE",
    "is_1d_only_ea_dtype",
    "is_all_strings",
    "is_any_real_numeric_dtype",
    "is_array_like",
    "is_bool",
    "is_bool_dtype",
    "is_categorical_dtype",
    "is_complex",
    "is_complex_dtype",
    "is_dataclass",
    "is_datetime64_any_dtype",
    "is_datetime64_dtype",
    "is_datetime64_ns_dtype",
    "is_datetime64tz_dtype",
    "is_decimal",
    "is_dict_like",
    "is_dtype_equal",
    "is_ea_or_datetimelike_dtype",
    "is_extension_array_dtype",
    "is_file_like",
    "is_float_dtype",
    "is_int64_dtype",
    "is_integer_dtype",
    "is_interval_dtype",
    "is_iterator",
    "is_named_tuple",
    "is_nested_list_like",
    "is_number",
    "is_numeric_dtype",
    "is_object_dtype",
    "is_period_dtype",
    "is_re",
    "is_re_compilable",
    "is_scipy_sparse",
    "is_sequence",
    "is_signed_integer_dtype",
    "is_sparse",
    "is_string_dtype",
    "is_string_or_object_np_dtype",
    "is_timedelta64_dtype",
    "is_timedelta64_ns_dtype",
    "is_unsigned_integer_dtype",
    "needs_i8_conversion",
    "pandas_dtype",
    "TD64NS_DTYPE",
    "validate_all_hashable",
]
```