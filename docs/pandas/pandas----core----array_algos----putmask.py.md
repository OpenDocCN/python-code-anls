# `D:\src\scipysrc\pandas\pandas\core\array_algos\putmask.py`

```
"""
EA-compatible analogue to np.putmask
"""

from __future__ import annotations  # 使用未来的类型注解语法（PEP 563）

from typing import (
    TYPE_CHECKING,  # 导入类型检查相关模块
    Any,  # 导入任意类型
)

import numpy as np  # 导入NumPy库

from pandas._libs import lib  # 导入Pandas私有库中的lib模块

from pandas.core.dtypes.cast import infer_dtype_from  # 从Pandas核心模块导入类型推断函数
from pandas.core.dtypes.common import is_list_like  # 导入Pandas核心模块中的列表检查函数

from pandas.core.arrays import ExtensionArray  # 导入Pandas核心数组模块中的扩展数组类型

if TYPE_CHECKING:
    from pandas._typing import (  # 导入Pandas的类型提示
        ArrayLike,  # 数组样式
        npt,  # NumPy类型
    )

    from pandas import MultiIndex  # 导入多级索引类型


def putmask_inplace(values: ArrayLike, mask: npt.NDArray[np.bool_], value: Any) -> None:
    """
    ExtensionArray-compatible implementation of np.putmask.  The main
    difference is we do not handle repeating or truncating like numpy.

    Parameters
    ----------
    values: np.ndarray or ExtensionArray
        The array-like object where the values will be modified.
    mask : np.ndarray[bool]
        Boolean mask array indicating where values should be updated.
        We assume extract_bool_array has already been called.
    value : Any
        The value or array of values to be placed into `values` where `mask` is True.
    """

    if (
        not isinstance(values, np.ndarray)
        or (values.dtype == object and not lib.is_scalar(value))
        # GH#43424: np.putmask raises TypeError if we cannot cast between types with
        # rule = "safe", a stricter guarantee we may not have here
        or (
            isinstance(value, np.ndarray) and not np.can_cast(value.dtype, values.dtype)
        )
    ):
        # GH#19266 using np.putmask gives unexpected results with listlike value
        #  along with object dtype
        if is_list_like(value) and len(value) == len(values):
            values[mask] = value[mask]
        else:
            values[mask] = value
    else:
        # GH#37833 np.putmask is more performant than __setitem__
        np.putmask(values, mask, value)  # 使用NumPy的putmask函数进行值替换操作


def putmask_without_repeat(
    values: np.ndarray, mask: npt.NDArray[np.bool_], new: Any
) -> None:
    """
    np.putmask will truncate or repeat if `new` is a listlike with
    len(new) != len(values).  We require an exact match.

    Parameters
    ----------
    values : np.ndarray
        The array where values will be updated.
    mask : np.ndarray[bool]
        Boolean mask array indicating where values should be updated.
    new : Any
        The value or array of values to be placed into `values` where `mask` is True.
    """
    if getattr(new, "ndim", 0) >= 1:
        new = new.astype(values.dtype, copy=False)

    # TODO: this prob needs some better checking for 2D cases
    nlocs = mask.sum()  # 计算掩码数组中True值的数量
    # 如果 nlocs 大于 0 并且 new 是类列表对象，并且 new 对象没有多维度的属性，即 ndim 等于 1
    if nlocs > 0 and is_list_like(new) and getattr(new, "ndim", 1) == 1:
        # 获取 new 对象的形状
        shape = np.shape(new)
        
        # 如果 nlocs 等于 new 对象最后一个维度的长度
        if nlocs == shape[-1]:
            # GH#30567
            # 如果 new 的长度小于 values 的长度，np.putmask 会先重复 new 数组然后再赋值给掩码值，因此会产生错误的结果。
            # np.place 则直接使用 new 的值来放置在 values 的掩码位置上，避免了这个问题。
            np.place(values, mask, new)
            # 即 values[mask] = new
        # 如果 mask 的最后一个维度长度等于 new 的最后一个维度长度，或者 new 的最后一个维度长度为 1
        elif mask.shape[-1] == shape[-1] or shape[-1] == 1:
            np.putmask(values, mask, new)
        else:
            # 抛出数值错误，无法将长度不匹配的值分配给掩码数组
            raise ValueError("cannot assign mismatch length to masked array")
    else:
        np.putmask(values, mask, new)
# 验证并处理 putmask 操作的掩码，同时检查是否是空操作
def validate_putmask(
    values: ArrayLike | MultiIndex, mask: np.ndarray
) -> tuple[npt.NDArray[np.bool_], bool]:
    # 将掩码转换为布尔数组
    mask = extract_bool_array(mask)
    # 检查掩码与数据的形状是否一致，若不一致则引发 ValueError 异常
    if mask.shape != values.shape:
        raise ValueError("putmask: mask and data must be the same size")

    # 检查掩码是否全为 False，确定是否为无操作
    noop = not mask.any()
    return mask, noop


# 将稀疏数组或布尔数组转换为 ndarray[bool]
def extract_bool_array(mask: ArrayLike) -> npt.NDArray[np.bool_]:
    # 如果掩码是 ExtensionArray 类型（例如 BooleanArray、Sparse[bool] 等），将其转换为 ndarray[bool]
    if isinstance(mask, ExtensionArray):
        # 除了 BooleanArray 外，这相当于 np.asarray(mask, dtype=bool)
        mask = mask.to_numpy(dtype=bool, na_value=False)

    # 将掩码转换为 ndarray[bool]
    mask = np.asarray(mask, dtype=bool)
    return mask


# 对于日期时间兼容性设置项，处理与 putmask 相关的操作
def setitem_datetimelike_compat(values: np.ndarray, num_set: int, other):
    """
    Parameters
    ----------
    values : np.ndarray
    num_set : int
        对于 putmask 操作，此值是掩码的非零元素个数
    other : Any
    """
    # 如果数据类型是 object 类型
    if values.dtype == object:
        # 推断 other 的数据类型
        dtype, _ = infer_dtype_from(other)

        # 如果推断出的类型为 timedelta64，存在以下问题需要处理
        if lib.is_np_dtype(dtype, "mM"):
            # https://github.com/numpy/numpy/issues/12550
            # timedelta64 会错误地转换为 int
            # 如果 other 不是列表形式，将其复制 num_set 次
            if not is_list_like(other):
                other = [other] * num_set
            else:
                other = list(other)

    return other
```