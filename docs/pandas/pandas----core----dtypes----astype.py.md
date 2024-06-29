# `D:\src\scipysrc\pandas\pandas\core\dtypes\astype.py`

```
"""
Functions for implementing 'astype' methods according to pandas conventions,
particularly ones that differ from numpy.
"""

from __future__ import annotations  # 导入未来版本的注解支持

import inspect  # 导入用于检查对象的模块
from typing import (  # 导入类型提示相关模块
    TYPE_CHECKING,
    overload,
)
import warnings  # 导入警告模块

import numpy as np  # 导入 NumPy 库

from pandas._libs import lib  # 导入 Pandas 库的内部 C 扩展
from pandas._libs.tslibs.timedeltas import array_to_timedelta64  # 导入时间增量转换函数
from pandas.errors import IntCastingNaNError  # 导入整数转换 NaN 错误

from pandas.core.dtypes.common import (  # 导入 Pandas 核心模块的常用数据类型函数
    is_object_dtype,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (  # 导入 Pandas 核心模块的数据类型定义
    ExtensionDtype,
    NumpyEADtype,
)

if TYPE_CHECKING:
    from pandas._typing import (  # 根据条件导入 Pandas 的类型提示
        ArrayLike,
        DtypeObj,
        IgnoreRaise,
    )

    from pandas.core.arrays import ExtensionArray  # 导入 Pandas 核心数组的扩展数组


@overload
def _astype_nansafe(
    arr: np.ndarray, dtype: np.dtype, copy: bool = ..., skipna: bool = ...
) -> np.ndarray: ...


@overload
def _astype_nansafe(
    arr: np.ndarray, dtype: ExtensionDtype, copy: bool = ..., skipna: bool = ...
) -> ExtensionArray: ...


def _astype_nansafe(
    arr: np.ndarray, dtype: DtypeObj, copy: bool = True, skipna: bool = False
) -> ArrayLike:
    """
    Cast the elements of an array to a given dtype a nan-safe manner.

    Parameters
    ----------
    arr : ndarray
        输入的数组
    dtype : np.dtype or ExtensionDtype
        欲转换的目标数据类型
    copy : bool, default True
        是否复制数组。如果为 False，尝试创建视图但可能失败，例如当项目大小不对齐时。
    skipna: bool, default False
        在转换为字符串类型时，是否跳过 NaN 值。

    Raises
    ------
    ValueError
        当 dtype 是 datetime64/timedelta64 类型但没有指定单位时。
    """

    # 如果 dtype 是 ExtensionDtype 类型，则调用其构造函数创建数组类型
    if isinstance(dtype, ExtensionDtype):
        return dtype.construct_array_type()._from_sequence(arr, dtype=dtype, copy=copy)

    elif not isinstance(dtype, np.dtype):  # pragma: no cover
        raise ValueError("dtype must be np.dtype or ExtensionDtype")

    # 如果数组 arr 的数据类型是日期或时间增量类型
    if arr.dtype.kind in "mM":
        from pandas.core.construction import ensure_wrapped_if_datetimelike

        # 确保数组是日期或时间增量类型的封装形式
        arr = ensure_wrapped_if_datetimelike(arr)
        # 使用指定的 dtype 进行类型转换，并返回转换后的 NumPy 数组形式
        res = arr.astype(dtype, copy=copy)
        return np.asarray(res)

    # 如果 dtype 类型是字符串
    if issubclass(dtype.type, str):
        shape = arr.shape
        if arr.ndim > 1:
            arr = arr.ravel()
        # 确保创建字符串数组，跳过 NaN 值，不转换 NaN 为字符串
        return lib.ensure_string_array(
            arr, skipna=skipna, convert_na_value=False
        ).reshape(shape)

    # 如果 arr 的数据类型是浮点类型，并且 dtype 的类型是整数类型
    elif np.issubdtype(arr.dtype, np.floating) and dtype.kind in "iu":
        # 安全地将浮点数转换为整数类型，处理 NaN 值
        return _astype_float_to_int_nansafe(arr, dtype, copy)
    # 如果数组的数据类型为 object
    elif arr.dtype == object:
        # 如果数组中包含 datetime/timedelta 对象数组
        # 强制转换为 datetime64[ns] 并使用 DatetimeArray.astype

        if lib.is_np_dtype(dtype, "M"):
            # 导入 DatetimeArray 类
            from pandas.core.arrays import DatetimeArray
            
            # 从数组序列创建 DatetimeArray 对象，并指定 dtype
            dta = DatetimeArray._from_sequence(arr, dtype=dtype)
            # 返回 DatetimeArray 对象的底层 ndarray
            return dta._ndarray

        elif lib.is_np_dtype(dtype, "m"):
            # 导入 ensure_wrapped_if_datetimelike 函数
            from pandas.core.construction import ensure_wrapped_if_datetimelike
            
            # 因为我们知道 arr.dtype == object，等价于 np.asarray(to_timedelta(arr))
            # 使用一个不需要循环导入的低级 API，将数组转换为 timedelta64 类型的底层 ndarray
            tdvals = array_to_timedelta64(arr).view("m8[ns]")
            
            # 确保如果是类似 datetime 类型的数据，进行包装处理
            tda = ensure_wrapped_if_datetimelike(tdvals)
            # 强制转换为指定 dtype，不进行复制，返回底层 ndarray
            return tda.astype(dtype, copy=False)._ndarray

    # 如果 dtype 的名称为 "datetime64" 或 "timedelta64"
    if dtype.name in ("datetime64", "timedelta64"):
        # 抛出值错误，指示指定 dtype 的单位应该是 "ns"
        msg = (
            f"The '{dtype.name}' dtype has no unit. Please pass in "
            f"'{dtype.name}[ns]' instead."
        )
        raise ValueError(msg)

    # 如果需要复制或者数组的 dtype 为 object，或者目标 dtype 为 object
    if copy or arr.dtype == object or dtype == object:
        # 显式复制，或者由于 NumPy 无法从/到 object 进行视图操作
        # 返回进行指定 dtype 类型转换后的数组的副本
        return arr.astype(dtype, copy=True)

    # 否则，直接返回转换后的数组，不进行复制
    return arr.astype(dtype, copy=copy)
def _astype_float_to_int_nansafe(
    values: np.ndarray, dtype: np.dtype, copy: bool
) -> np.ndarray:
    """
    astype with a check preventing converting NaN to an meaningless integer value.
    """
    # 检查是否所有的值都是有限的，即不是 NaN 或 inf
    if not np.isfinite(values).all():
        raise IntCastingNaNError(
            "Cannot convert non-finite values (NA or inf) to integer"
        )
    # 如果 dtype 的类型是无符号整数类型
    if dtype.kind == "u":
        # 如果 values 中有小于 0 的值，无法无损地从 values.dtype 转换为 dtype
        if not (values >= 0).all():
            raise ValueError(f"Cannot losslessly cast from {values.dtype} to {dtype}")
    # 忽略运行时警告，执行类型转换并返回结果
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return values.astype(dtype, copy=copy)


def astype_array(values: ArrayLike, dtype: DtypeObj, copy: bool = False) -> ArrayLike:
    """
    Cast array (ndarray or ExtensionArray) to the new dtype.

    Parameters
    ----------
    values : ndarray or ExtensionArray
    dtype : dtype object
    copy : bool, default False
        copy if indicated

    Returns
    -------
    ndarray or ExtensionArray
    """
    # 如果 values 的 dtype 已经是 dtype，且需要复制，则返回 values 的副本
    if values.dtype == dtype:
        if copy:
            return values.copy()
        return values

    # 如果 values 不是 np.ndarray 类型（即 ExtensionArray 类型）
    if not isinstance(values, np.ndarray):
        # 将 ExtensionArray 转换为指定 dtype，支持复制操作
        values = values.astype(dtype, copy=copy)

    else:
        # 对于 np.ndarray 类型，执行安全的类型转换（避免 NaN 的问题）
        values = _astype_nansafe(values, dtype, copy=copy)

    # 在 pandas 中不保存 numpy 的 str 类型，所以需要转换为 object 类型
    if isinstance(dtype, np.dtype) and issubclass(values.dtype.type, str):
        values = np.array(values, dtype=object)

    return values


def astype_array_safe(
    values: ArrayLike, dtype, copy: bool = False, errors: IgnoreRaise = "raise"
) -> ArrayLike:
    """
    Cast array (ndarray or ExtensionArray) to the new dtype.

    This basically is the implementation for DataFrame/Series.astype and
    includes all custom logic for pandas (NaN-safety, converting str to object,
    not allowing )

    Parameters
    ----------
    values : ndarray or ExtensionArray
    dtype : str, dtype convertible
    copy : bool, default False
        copy if indicated
    errors : str, {'raise', 'ignore'}, default 'raise'
        - ``raise`` : allow exceptions to be raised
        - ``ignore`` : suppress exceptions. On error return original object

    Returns
    -------
    ndarray or ExtensionArray
    """
    # errors 参数只能是 "raise" 或 "ignore" 中的一个，否则抛出 ValueError
    errors_legal_values = ("raise", "ignore")
    if errors not in errors_legal_values:
        invalid_arg = (
            "Expected value of kwarg 'errors' to be one of "
            f"{list(errors_legal_values)}. Supplied value is '{errors}'"
        )
        raise ValueError(invalid_arg)

    # 如果 dtype 是 ExtensionDtype 类的子类，报错，应该传入 dtype 的实例而不是类本身
    if inspect.isclass(dtype) and issubclass(dtype, ExtensionDtype):
        msg = (
            f"Expected an instance of {dtype.__name__}, "
            "but got the class instead. Try instantiating 'dtype'."
        )
        raise TypeError(msg)

    # 将 dtype 转换为 pandas 的数据类型
    dtype = pandas_dtype(dtype)
    # 检查 dtype 是否是 NumpyEADtype 的实例
    if isinstance(dtype, NumpyEADtype):
        # 确保不会得到 NumpyExtensionArray 类型
        dtype = dtype.numpy_dtype

    try:
        # 尝试将 values 转换为指定的 dtype 类型的数组，并返回新的数组
        new_values = astype_array(values, dtype, copy=copy)
    except (ValueError, TypeError):
        # 处理可能的异常，比如在处理对象类型（如字符串）转换为 float 时可能出错
        if errors == "ignore":
            # 如果设置为忽略错误，则返回原始的 values
            new_values = values
        else:
            # 如果不忽略错误，则重新抛出异常
            raise

    # 返回处理后的新数组 new_values
    return new_values
def astype_is_view(dtype: DtypeObj, new_dtype: DtypeObj) -> bool:
    """Checks if astype avoided copying the data.

    Parameters
    ----------
    dtype : Original dtype
        原始数据类型
    new_dtype : target dtype
        目标数据类型

    Returns
    -------
    bool
        如果新数据是视图或不能保证是复制，则为True；否则为False
    """
    if dtype.kind in "iufb" and dtype.kind == new_dtype.kind:
        # fastpath for numeric dtypes
        if hasattr(dtype, "itemsize") and hasattr(new_dtype, "itemsize"):
            return dtype.itemsize == new_dtype.itemsize  # pyright: ignore[reportAttributeAccessIssue]

    if isinstance(dtype, np.dtype) and not isinstance(new_dtype, np.dtype):
        new_dtype, dtype = dtype, new_dtype

    if dtype == new_dtype:
        return True

    elif isinstance(dtype, np.dtype) and isinstance(new_dtype, np.dtype):
        # Only equal numpy dtypes avoid a copy
        return False

    elif is_string_dtype(dtype) and is_string_dtype(new_dtype):
        # Potentially! a view when converting from object to string
        return True

    elif is_object_dtype(dtype) and new_dtype.kind == "O":
        # When the underlying array has dtype object, we don't have to make a copy
        return True

    elif dtype.kind in "mM" and new_dtype.kind in "mM":
        dtype = getattr(dtype, "numpy_dtype", dtype)
        new_dtype = getattr(new_dtype, "numpy_dtype", new_dtype)
        return getattr(dtype, "unit", None) == getattr(new_dtype, "unit", None)

    numpy_dtype = getattr(dtype, "numpy_dtype", None)
    new_numpy_dtype = getattr(new_dtype, "numpy_dtype", None)

    if numpy_dtype is None and isinstance(dtype, np.dtype):
        numpy_dtype = dtype

    if new_numpy_dtype is None and isinstance(new_dtype, np.dtype):
        new_numpy_dtype = new_dtype

    if numpy_dtype is not None and new_numpy_dtype is not None:
        # if both have NumPy dtype or one of them is a numpy dtype
        # they are only a view when the numpy dtypes are equal, e.g.
        # int64 -> Int64 or int64[pyarrow]
        # int64 -> Int32 copies
        return numpy_dtype == new_numpy_dtype

    # Assume this is a view since we don't know for sure if a copy was made
    return True
```