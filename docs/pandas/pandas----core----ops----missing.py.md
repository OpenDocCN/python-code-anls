# `D:\src\scipysrc\pandas\pandas\core\ops\missing.py`

```
"""
Missing data handling for arithmetic operations.

In particular, pandas conventions regarding division by zero differ
from numpy in the following ways:
    1) np.array([-1, 0, 1], dtype=dtype1) // np.array([0, 0, 0], dtype=dtype2)
       gives [nan, nan, nan] for most dtype combinations, and [0, 0, 0] for
       the remaining pairs
       (the remaining being dtype1==dtype2==intN and dtype==dtype2==uintN).

       pandas convention is to return [-inf, nan, inf] for all dtype
       combinations.

       Note: the numpy behavior described here is py3-specific.

    2) np.array([-1, 0, 1], dtype=dtype1) % np.array([0, 0, 0], dtype=dtype2)
       gives precisely the same results as the // operation.

       pandas convention is to return [nan, nan, nan] for all dtype
       combinations.

    3) divmod behavior consistent with 1) and 2).
"""

from __future__ import annotations

import operator

import numpy as np

from pandas.core import roperator


def _fill_zeros(result: np.ndarray, x, y) -> np.ndarray:
    """
    If this is a reversed op, then flip x,y

    If we have an integer value (or array in y)
    and we have 0's, fill them with np.nan,
    return the result.

    Mask the nan's from x.
    """
    if result.dtype.kind == "f":
        # 如果结果的数据类型是浮点型，直接返回结果，不做处理
        return result

    is_variable_type = hasattr(y, "dtype")
    is_scalar_type = not isinstance(y, np.ndarray)

    if not is_variable_type and not is_scalar_type:
        # 如果y不是变量类型也不是标量类型，则直接返回结果，例如在 mod 操作时可能会出现这种情况
        return result

    if is_scalar_type:
        # 如果y是标量类型，则将其转换为numpy数组
        y = np.array(y)

    if y.dtype.kind in "iu":
        # 如果y的数据类型是无符号整数或有符号整数
        ymask = y == 0
        if ymask.any():
            # 创建一个可以广播的 mask，用来屏蔽和 NaN 相关的位置
            mask = ymask & ~np.isnan(result)

            # 提升性能，将结果转换为 float64 类型
            result = result.astype("float64", copy=False)

            # 使用 np.putmask 函数将 mask 位置处的值替换为 NaN
            np.putmask(result, mask, np.nan)

    return result


def mask_zero_div_zero(x, y, result: np.ndarray) -> np.ndarray:
    """
    Set results of  0 // 0 to np.nan, regardless of the dtypes
    of the numerator or the denominator.

    Parameters
    ----------
    x : ndarray
    y : ndarray
    result : ndarray

    Returns
    -------
    ndarray
        The filled result.

    Examples
    --------
    >>> x = np.array([1, 0, -1], dtype=np.int64)
    >>> x
    array([ 1,  0, -1])
    >>> y = 0  # int 0; numpy behavior is different with float
    >>> result = x // y
    >>> result  # raw numpy result does not fill division by zero
    array([0, 0, 0])
    >>> mask_zero_div_zero(x, y, result)
    array([ inf,  nan, -inf])
    """

    if not hasattr(y, "dtype"):
        # 如果y没有 dtype 属性，则将其转换为numpy数组
        y = np.array(y)
    if not hasattr(x, "dtype"):
        # 如果x没有 dtype 属性，则将其转换为numpy数组
        x = np.array(x)

    zmask = y == 0
    if zmask.any():
        # 如果存在任何非零的掩码值

        # 对于 -0.0，如果需要翻转符号
        zneg_mask = zmask & np.signbit(y)
        zpos_mask = zmask & ~zneg_mask

        # 检查 x 的值范围
        x_lt0 = x < 0
        x_gt0 = x > 0
        nan_mask = zmask & (x == 0)
        neginf_mask = (zpos_mask & x_lt0) | (zneg_mask & x_gt0)
        posinf_mask = (zpos_mask & x_gt0) | (zneg_mask & x_lt0)

        # 如果存在 NaN、-∞ 或 +∞ 的情况
        if nan_mask.any() or neginf_mask.any() or posinf_mask.any():
            # 将结果转换为 float64 类型（如果尚未是该类型），在原地修改
            result = result.astype("float64", copy=False)

            # 将 NaN 填充到 nan_mask 对应的位置
            result[nan_mask] = np.nan
            # 将 +∞ 填充到 posinf_mask 对应的位置
            result[posinf_mask] = np.inf
            # 将 -∞ 填充到 neginf_mask 对应的位置
            result[neginf_mask] = -np.inf

    return result
def dispatch_fill_zeros(op, left, right, result):
    """
    根据操作类型调用 _fill_zeros 函数，根据操作的不同选择适当的填充值，
    特别处理 divmod 和 rdivmod 操作。

    Parameters
    ----------
    op : function (operator.add, operator.div, ...)
        操作函数，如加法、除法等
    left : object (np.ndarray for non-reversed ops)
        左操作数，对于非反转操作，通常是 np.ndarray 类型，排除了 ExtensionArrays
    right : object (np.ndarray for reversed ops)
        右操作数，对于反转操作，通常是 np.ndarray 类型，排除了 ExtensionArrays
    result : ndarray
        结果数组，可以是单个 ndarray 或者是 divmod/rdivmod 操作返回的两个 ndarray 的元组

    Returns
    -------
    result : np.ndarray
        处理后的结果 ndarray 或者是 divmod/rdivmod 操作返回的两个 ndarray 的元组

    Notes
    -----
    对于 divmod 和 rdivmod 操作，`result` 参数及返回的 `result` 都是包含两个 ndarray 对象的元组。
    """
    if op is divmod:
        result = (
            mask_zero_div_zero(left, right, result[0]),
            _fill_zeros(result[1], left, right),
        )
    elif op is roperator.rdivmod:
        result = (
            mask_zero_div_zero(right, left, result[0]),
            _fill_zeros(result[1], right, left),
        )
    elif op is operator.floordiv:
        # 注意：对于 truediv 操作不需要这样做；在 Python 3 中，numpy 的行为已经符合我们的预期。
        result = mask_zero_div_zero(left, right, result)
    elif op is roperator.rfloordiv:
        # 注意：对于 rtruediv 操作不需要这样做；在 Python 3 中，numpy 的行为已经符合我们的预期。
        result = mask_zero_div_zero(right, left, result)
    elif op is operator.mod:
        result = _fill_zeros(result, left, right)
    elif op is roperator.rmod:
        result = _fill_zeros(result, right, left)
    return result
```