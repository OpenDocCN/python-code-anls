# `D:\src\scipysrc\pandas\pandas\core\ops\mask_ops.py`

```
"""
Ops for masked arrays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pandas._libs import (
    lib,
    missing as libmissing,
)

if TYPE_CHECKING:
    from pandas._typing import npt


def kleene_or(
    left: bool | np.ndarray | libmissing.NAType,
    right: bool | np.ndarray | libmissing.NAType,
    left_mask: np.ndarray | None,
    right_mask: np.ndarray | None,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """
    Boolean ``or`` using Kleene logic.

    Values are NA where we have ``NA | NA`` or ``NA | False``.
    ``NA | True`` is considered True.

    Parameters
    ----------
    left, right : ndarray, NA, or bool
        The values of the array.
    left_mask, right_mask : ndarray, optional
        The masks. Only one of these may be None, which implies that
        the associated `left` or `right` value is a scalar.

    Returns
    -------
    result, mask: ndarray[bool]
        The result of the logical or, and the new mask.
    """
    # To reduce the number of cases, we ensure that `left` & `left_mask`
    # always come from an array, not a scalar. This is safe, since
    # A | B == B | A
    if left_mask is None:
        return kleene_or(right, left, right_mask, left_mask)
    # 如果 left_mask 为 None，则交换 left 和 right，重新调用 kleene_or

    if not isinstance(left, np.ndarray):
        raise TypeError("Either `left` or `right` need to be a np.ndarray.")
    # 如果 left 不是 ndarray，则抛出类型错误异常

    raise_for_nan(right, method="or")
    # 检查 right 中是否包含 NA 值，以方法 "or" 进行检查

    if right is libmissing.NA:
        result = left.copy()
    else:
        result = left | right
    # 如果 right 是 NA，则将 result 设置为 left 的副本；否则计算 left 和 right 的逻辑或并赋给 result

    if right_mask is not None:
        # output is unknown where (False & NA), (NA & False), (NA & NA)
        left_false = ~(left | left_mask)
        right_false = ~(right | right_mask)
        mask = (
            (left_false & right_mask)
            | (right_false & left_mask)
            | (left_mask & right_mask)
        )
    else:
        if right is True:
            mask = np.zeros_like(left_mask)
        elif right is libmissing.NA:
            mask = (~left & ~left_mask) | left_mask
        else:
            # False
            mask = left_mask.copy()
    # 根据 right_mask 的情况确定 mask 的值

    return result, mask


def kleene_xor(
    left: bool | np.ndarray | libmissing.NAType,
    right: bool | np.ndarray | libmissing.NAType,
    left_mask: np.ndarray | None,
    right_mask: np.ndarray | None,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """
    Boolean ``xor`` using Kleene logic.

    This is the same as ``or``, with the following adjustments

    * True, True -> False
    * True, NA   -> NA

    Parameters
    ----------
    left, right : ndarray, NA, or bool
        The values of the array.
    left_mask, right_mask : ndarray, optional
        The masks. Only one of these may be None, which implies that
        the associated `left` or `right` value is a scalar.

    Returns
    -------
    result, mask: ndarray[bool]
        The result of the logical xor, and the new mask.
    """
    # 省略了 kleene_xor 函数的注释部分，需要添加相应的注释
    # 如果 `left_mask` 是 `None`，则调用 `kleene_xor` 函数处理 `right`、`left`、`right_mask` 和 `left_mask`
    # 这样处理是安全的，因为异或运算满足交换律：A ^ B == B ^ A
    if left_mask is None:
        return kleene_xor(right, left, right_mask, left_mask)

    # 如果 `left` 不是 `np.ndarray` 类型，则抛出类型错误异常
    if not isinstance(left, np.ndarray):
        raise TypeError("Either `left` or `right` need to be a np.ndarray.")

    # 检查 `right` 是否为缺失值 NA，如果是，根据方法 "xor" 抛出异常
    raise_for_nan(right, method="xor")

    # 如果 `right` 是缺失值 NA，则初始化结果为和 `left` 相同形状的全零数组
    if right is libmissing.NA:
        result = np.zeros_like(left)
    else:
        # 否则，计算 `left` 与 `right` 的按位异或结果
        result = left ^ right

    # 如果 `right_mask` 是 `None`
    if right_mask is None:
        # 如果 `right` 是缺失值 NA，则初始化掩码为和 `left_mask` 相同形状的全一数组
        if right is libmissing.NA:
            mask = np.ones_like(left_mask)
        else:
            # 否则，复制 `left_mask` 作为掩码
            mask = left_mask.copy()
    else:
        # 否则，合并 `left_mask` 和 `right_mask`，按位或操作
        mask = left_mask | right_mask

    # 返回计算得到的结果 `result` 和合并后的掩码 `mask`
    return result, mask
def kleene_and(
    left: bool | libmissing.NAType | np.ndarray,
    right: bool | libmissing.NAType | np.ndarray,
    left_mask: np.ndarray | None,
    right_mask: np.ndarray | None,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """
    Boolean ``and`` using Kleene logic.

    Values are ``NA`` for ``NA & NA`` or ``True & NA``.

    Parameters
    ----------
    left, right : ndarray, NA, or bool
        The values of the array.
    left_mask, right_mask : ndarray, optional
        The masks. Only one of these may be None, which implies that
        the associated `left` or `right` value is a scalar.

    Returns
    -------
    result, mask: ndarray[bool]
        The result of the logical xor, and the new mask.
    """
    # 如果 left_mask 为 None，则交换 left 和 right，并递归调用函数
    if left_mask is None:
        return kleene_and(right, left, right_mask, left_mask)

    # 检查 left 是否为 np.ndarray 类型，若不是则抛出类型错误
    if not isinstance(left, np.ndarray):
        raise TypeError("Either `left` or `right` need to be a np.ndarray.")
    
    # 调用 raise_for_nan 函数，如果 right 为 NA，则抛出异常
    raise_for_nan(right, method="and")

    # 根据 right 的值确定 result 的计算方式
    if right is libmissing.NA:
        result = np.zeros_like(left)
    else:
        result = left & right

    # 如果 right_mask 为 None，则处理标量 right 的情况
    if right_mask is None:
        if right is libmissing.NA:
            # 标量 right 是 NA，则根据 left_mask 计算 mask
            mask = (left & ~left_mask) | left_mask
        else:
            # 标量 right 非 NA，则直接使用 left_mask
            mask = left_mask.copy()
            if right is False:
                # right 为 False，则全部取消 mask
                mask[:] = False
    else:
        # 处理 right_mask 不为 None 的情况，根据 left 和 right 的值计算 mask
        left_false = ~(left | left_mask)
        right_false = ~(right | right_mask)
        mask = (left_mask & ~right_false) | (right_mask & ~left_false)

    # 返回计算结果和 mask
    return result, mask


def raise_for_nan(value: object, method: str) -> None:
    # 如果 value 是浮点数且为 NaN，则抛出值错误异常
    if lib.is_float(value) and np.isnan(value):
        raise ValueError(f"Cannot perform logical '{method}' with floating NaN")
```