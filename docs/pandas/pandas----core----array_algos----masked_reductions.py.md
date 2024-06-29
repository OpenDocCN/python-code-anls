# `D:\src\scipysrc\pandas\pandas\core\array_algos\masked_reductions.py`

```
"""
masked_reductions.py is for reduction algorithms using a mask-based approach
for missing values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

import numpy as np

from pandas._libs import missing as libmissing

from pandas.core.nanops import check_below_min_count

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas._typing import (
        AxisInt,
        npt,
    )


def _reductions(
    func: Callable,
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    min_count: int = 0,
    axis: AxisInt | None = None,
    **kwargs,
):
    """
    Sum, mean or product for 1D masked array.

    Parameters
    ----------
    func : np.sum or np.prod
        Function for the reduction operation (e.g., np.sum, np.prod).
    values : np.ndarray
        Numpy array containing the values to operate upon.
    mask : np.ndarray[bool]
        Boolean numpy array where True values indicate missing values.
    skipna : bool, default True
        Whether to skip missing values (True) or treat them as valid (False).
    min_count : int, default 0
        Minimum number of non-missing values required to perform the operation.
        If fewer non-missing values are present, the result will be NA.
    axis : int or None, optional, default None
        Axis or axes along which the operation is performed. By default, operates
        over the entire array.
    """
    if not skipna:
        # If skipna is False, return NA if any missing values (True in mask) or
        # if the number of non-missing values is below the specified min_count.
        if mask.any() or check_below_min_count(values.shape, None, min_count):
            return libmissing.NA
        else:
            return func(values, axis=axis, **kwargs)
    else:
        # If skipna is True, return NA if the number of non-missing values is below
        # min_count and either axis is None or values is 1-dimensional.
        if check_below_min_count(values.shape, mask, min_count) and (
            axis is None or values.ndim == 1
        ):
            return libmissing.NA

        # Otherwise, perform the reduction operation ignoring the missing values
        return func(values, where=~mask, axis=axis, **kwargs)


def sum(
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    min_count: int = 0,
    axis: AxisInt | None = None,
):
    """
    Sum function for masked arrays.

    Parameters
    ----------
    values : np.ndarray
        Numpy array with the values to sum.
    mask : np.ndarray[bool]
        Boolean numpy array indicating missing values.
    skipna : bool, default True
        Whether to skip NA values.
    min_count : int, default 0
        Minimum count of non-NA values required to perform the operation.
    axis : int or None, optional, default None
        Axis along which to operate. By default, operates over the entire array.
    """
    return _reductions(
        np.sum, values=values, mask=mask, skipna=skipna, min_count=min_count, axis=axis
    )


def prod(
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    min_count: int = 0,
    axis: AxisInt | None = None,
):
    """
    Product function for masked arrays.

    Parameters
    ----------
    values : np.ndarray
        Numpy array with the values to multiply.
    mask : np.ndarray[bool]
        Boolean numpy array indicating missing values.
    skipna : bool, default True
        Whether to skip NA values.
    min_count : int, default 0
        Minimum count of non-NA values required to perform the operation.
    axis : int or None, optional, default None
        Axis along which to operate. By default, operates over the entire array.
    """
    return _reductions(
        np.prod, values=values, mask=mask, skipna=skipna, min_count=min_count, axis=axis
    )


def _minmax(
    func: Callable,
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    axis: AxisInt | None = None,
):
    """
    Min or max function for masked arrays.

    Parameters
    ----------
    func : np.min or np.max
        Function for the reduction operation (e.g., np.min, np.max).
    values : np.ndarray
        Numpy array containing the values to operate upon.
    mask : np.ndarray[bool]
        Boolean numpy array where True values indicate missing values.
    skipna : bool, default True
        Whether to skip missing values (True) or treat them as valid (False).
    axis : int or None, optional, default None
        Axis or axes along which the operation is performed. By default, operates
        over the entire array.
    """
    # 如果不跳过缺失值处理（skipna=False）
    if not skipna:
        # 如果掩码中有任何True值或者数值数组为空
        if mask.any() or not values.size:
            # 当使用numpy进行最小/最大值计算时，空数组会引发异常，但pandas会返回NA
            return libmissing.NA
        else:
            # 否则，调用指定的函数计算数值数组在指定轴上的最小/最大值
            return func(values, axis=axis)
    else:
        # 否则，如果需要跳过缺失值处理
        subset = values[~mask]
        # 如果剩余的有效数值数组不为空
        if subset.size:
            # 调用指定的函数计算有效数值数组在指定轴上的最小/最大值
            return func(subset, axis=axis)
        else:
            # 当使用numpy进行最小/最大值计算时，空数组会引发异常，但pandas会返回NA
            return libmissing.NA
# 计算给定数组的最小值
def min(
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,  # 是否跳过 NaN 值，默认为 True
    axis: AxisInt | None = None,  # 计算的轴向，默认为 None，表示整个数组
):
    # 调用 _minmax 函数计算最小值
    return _minmax(np.min, values=values, mask=mask, skipna=skipna, axis=axis)


# 计算给定数组的最大值
def max(
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,  # 是否跳过 NaN 值，默认为 True
    axis: AxisInt | None = None,  # 计算的轴向，默认为 None，表示整个数组
):
    # 调用 _minmax 函数计算最大值
    return _minmax(np.max, values=values, mask=mask, skipna=skipna, axis=axis)


# 计算给定数组的均值
def mean(
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,  # 是否跳过 NaN 值，默认为 True
    axis: AxisInt | None = None,  # 计算的轴向，默认为 None，表示整个数组
):
    # 如果数组为空或者所有值都被屏蔽，则返回缺失值 NA
    if not values.size or mask.all():
        return libmissing.NA
    # 调用 _reductions 函数计算均值
    return _reductions(np.mean, values=values, mask=mask, skipna=skipna, axis=axis)


# 计算给定数组的方差
def var(
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,  # 是否跳过 NaN 值，默认为 True
    axis: AxisInt | None = None,  # 计算的轴向，默认为 None，表示整个数组
    ddof: int = 1,  # 自由度的修正值，默认为 1
):
    # 如果数组为空或者所有值都被屏蔽，则返回缺失值 NA
    if not values.size or mask.all():
        return libmissing.NA

    # 忽略运行时警告，使用 warnings 模块捕获
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # 调用 _reductions 函数计算方差
        return _reductions(
            np.var, values=values, mask=mask, skipna=skipna, axis=axis, ddof=ddof
        )


# 计算给定数组的标准差
def std(
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,  # 是否跳过 NaN 值，默认为 True
    axis: AxisInt | None = None,  # 计算的轴向，默认为 None，表示整个数组
    ddof: int = 1,  # 自由度的修正值，默认为 1
):
    # 如果数组为空或者所有值都被屏蔽，则返回缺失值 NA
    if not values.size or mask.all():
        return libmissing.NA

    # 忽略运行时警告，使用 warnings 模块捕获
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # 调用 _reductions 函数计算标准差
        return _reductions(
            np.std, values=values, mask=mask, skipna=skipna, axis=axis, ddof=ddof
        )
```