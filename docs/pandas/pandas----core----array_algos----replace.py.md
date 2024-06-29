# `D:\src\scipysrc\pandas\pandas\core\array_algos\replace.py`

```
# Methods used by Block.replace and related methods.

from __future__ import annotations

import operator  # 导入 operator 模块，用于操作符功能
import re  # 导入 re 模块，用于正则表达式操作
from re import Pattern  # 导入 Pattern 类型，用于类型提示
from typing import (  # 导入类型提示模块
    TYPE_CHECKING,
    Any,
)

import numpy as np  # 导入 NumPy 库，并使用 np 别名

from pandas.core.dtypes.common import (  # 从 Pandas 库中导入数据类型相关函数
    is_bool,
    is_re,
    is_re_compilable,
)
from pandas.core.dtypes.missing import isna  # 导入 Pandas 中的缺失值检查函数

if TYPE_CHECKING:
    from pandas._typing import (  # 导入 Pandas 的类型提示
        ArrayLike,
        Scalar,
        npt,
    )


def should_use_regex(regex: bool, to_replace: Any) -> bool:
    """
    Decide whether to treat `to_replace` as a regular expression.
    """
    if is_re(to_replace):
        regex = True

    regex = regex and is_re_compilable(to_replace)

    # Don't use regex if the pattern is empty.
    regex = regex and re.compile(to_replace).pattern != ""
    return regex


def compare_or_regex_search(
    a: ArrayLike, b: Scalar | Pattern, regex: bool, mask: npt.NDArray[np.bool_]
) -> ArrayLike:
    """
    Compare two array-like inputs of the same shape or two scalar values

    Calls operator.eq or re.search, depending on regex argument. If regex is
    True, perform an element-wise regex matching.

    Parameters
    ----------
    a : array-like
    b : scalar or regex pattern
    regex : bool
    mask : np.ndarray[bool]

    Returns
    -------
    mask : array-like of bool
    """
    if isna(b):
        return ~mask

    def _check_comparison_types(
        result: ArrayLike | bool, a: ArrayLike, b: Scalar | Pattern
    ) -> None:
        """
        Raises an error if the two arrays (a,b) cannot be compared.
        Otherwise, returns the comparison result as expected.
        """
        if is_bool(result) and isinstance(a, np.ndarray):
            type_names = [type(a).__name__, type(b).__name__]

            type_names[0] = f"ndarray(dtype={a.dtype})"

            raise TypeError(
                f"Cannot compare types {type_names[0]!r} and {type_names[1]!r}"
            )

    if not regex or not should_use_regex(regex, b):
        # TODO: should use missing.mask_missing?
        op = lambda x: operator.eq(x, b)
    else:
        op = np.vectorize(
            lambda x: bool(re.search(b, x))
            if isinstance(x, str) and isinstance(b, (str, Pattern))
            else False
        )

    # GH#32621 use mask to avoid comparing to NAs
    if isinstance(a, np.ndarray) and mask is not None:
        a = a[mask]
        result = op(a)

        if isinstance(result, np.ndarray):
            # The shape of the mask can differ to that of the result
            # since we may compare only a subset of a's or b's elements
            tmp = np.zeros(mask.shape, dtype=np.bool_)
            np.place(tmp, mask, result)
            result = tmp
    else:
        result = op(a)

    _check_comparison_types(result, a, b)
    return result


def replace_regex(
    values: ArrayLike, rx: re.Pattern, value, mask: npt.NDArray[np.bool_] | None
) -> None:
    """
    Parameters
    ----------
    values : array-like
        The array-like input to perform replacement on.
    rx : re.Pattern
        Compiled regex pattern used for matching.
    value
        Value to replace matched occurrences.
    mask : np.ndarray[bool] or None
        Boolean mask indicating where to apply replacements.

    Notes
    -----
    This function performs regex-based replacements on the input values
    according to the given regex pattern.
    """
    # values : ArrayLike
    # rx : re.Pattern
    # value : Any
    # mask : np.ndarray[bool], optional

    Notes
    -----
    Alters values in-place.
    """
    # 处理替换值为与正则表达式模式匹配但替换结果不是字符串的情况（如数值、nan、对象）
    if isna(value) or not isinstance(value, str):

        def re_replacer(s):
            # 如果值是 NaN 或者不是字符串，则直接返回原始值 s
            if is_re(rx) and isinstance(s, str):
                # 如果 value 是 NaN 或者不是字符串，则返回 value 是否与 s 匹配，匹配则返回 value，否则返回 s
                return value if rx.search(s) is not None else s
            else:
                return s

    else:
        # 此处 value 被保证是字符串，s 可能是字符串或者空值，如果是空值则直接返回
        def re_replacer(s):
            # 如果 value 是字符串，则尝试将 s 中匹配 rx 的部分用 value 替换
            if is_re(rx) and isinstance(s, str):
                return rx.sub(value, s)
            else:
                return s

    # 使用 np.vectorize 将 re_replacer 函数向量化，并指定输出类型为 np.object_
    f = np.vectorize(re_replacer, otypes=[np.object_])

    # 如果 mask 为空，则直接对 values 执行 f 函数
    if mask is None:
        values[:] = f(values)
    else:
        # 如果 mask 不为空，则只对 mask 为 True 的位置的 values 执行 f 函数
        values[mask] = f(values[mask])
```