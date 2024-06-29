# `D:\src\scipysrc\pandas\pandas\tests\arithmetic\common.py`

```
"""
Assertion helpers for arithmetic tests.
"""

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
    array,
)
import pandas._testing as tm
from pandas.core.arrays import (
    BooleanArray,
    NumpyExtensionArray,
)


def assert_cannot_add(left, right, msg="cannot add"):
    """
    Helper function to assert that two objects cannot be added.

    Parameters
    ----------
    left : object
        The first operand.
    right : object
        The second operand.
    msg : str, default "cannot add"
        The error message expected in the TypeError.
    """
    with pytest.raises(TypeError, match=msg):
        left + right
    with pytest.raises(TypeError, match=msg):
        right + left


def assert_invalid_addsub_type(left, right, msg=None):
    """
    Helper function to assert that two objects can
    neither be added nor subtracted.

    Parameters
    ----------
    left : object
        The first operand.
    right : object
        The second operand.
    msg : str or None, default None
        The error message expected in the TypeError.
    """
    with pytest.raises(TypeError, match=msg):
        left + right
    with pytest.raises(TypeError, match=msg):
        right + left
    with pytest.raises(TypeError, match=msg):
        left - right
    with pytest.raises(TypeError, match=msg):
        right - left


def get_upcast_box(left, right, is_cmp: bool = False):
    """
    Get the box to use for 'expected' in an arithmetic or comparison operation.

    Parameters
    ----------
    left : Any
    right : Any
    is_cmp : bool, default False
        Whether the operation is a comparison method.
    """

    if isinstance(left, DataFrame) or isinstance(right, DataFrame):
        # 如果左右操作数中有 DataFrame，则返回 DataFrame 类型
        return DataFrame
    if isinstance(left, Series) or isinstance(right, Series):
        if is_cmp and isinstance(left, Index):
            # 如果是比较操作且左操作数是 Index，则返回 np.array 类型
            # Index 在比较时不会延迟
            return np.array
        # 否则返回 Series 类型
        return Series
    if isinstance(left, Index) or isinstance(right, Index):
        if is_cmp:
            # 如果是比较操作，则返回 np.array 类型
            return np.array
        # 否则返回 Index 类型
        return Index
    # 默认返回 tm.to_array 函数
    return tm.to_array


def assert_invalid_comparison(left, right, box):
    """
    Assert that comparison operations with mismatched types behave correctly.

    Parameters
    ----------
    left : np.ndarray, ExtensionArray, Index, or Series
    right : object
    box : {pd.DataFrame, pd.Series, pd.Index, pd.array, tm.to_array}
    """
    # Not for tznaive-tzaware comparison

    # Note: not quite the same as how we do this for tm.box_expected
    xbox = box if box not in [Index, array] else np.array

    def xbox2(x):
        # Eventually we'd like this to be tighter, but for now we'll
        #  just exclude NumpyExtensionArray[bool]
        if isinstance(x, NumpyExtensionArray):
            # 如果 x 是 NumpyExtensionArray 类型，则返回其内部的 _ndarray 属性
            return x._ndarray
        if isinstance(x, BooleanArray):
            # 如果 x 是 BooleanArray 类型，则假设没有 pd.NA，转换为 bool 类型
            return x.astype(bool)
        return x
    # rev_box: 用于进行反向比较的变量
    rev_box = xbox
    # 如果 right 是 Index 类型并且 left 是 Series 类型，则使用 np.array 作为 rev_box
    if isinstance(right, Index) and isinstance(left, Series):
        rev_box = np.array

    # 对 left == right 进行逻辑比较，结果保存在 result 中
    result = xbox2(left == right)
    # 生成一个与 result 形状相同、元素为布尔类型的全零数组，作为期望结果
    expected = xbox(np.zeros(result.shape, dtype=np.bool_))

    # 使用测试工具库 tm 检查 result 是否等于 expected
    tm.assert_equal(result, expected)

    # 对 right == left 进行逻辑比较，结果保存在 result 中，并应用 rev_box 函数进行变换
    result = xbox2(right == left)
    # 使用测试工具库 tm 检查 result 是否等于 rev_box(expected)
    tm.assert_equal(result, rev_box(expected))

    # 对 left != right 进行逻辑比较，结果保存在 result 中，并对 expected 取反
    result = xbox2(left != right)
    # 使用测试工具库 tm 检查 result 是否等于 expected 的取反
    tm.assert_equal(result, ~expected)

    # 对 right != left 进行逻辑比较，结果保存在 result 中，并应用 rev_box 函数进行变换
    result = xbox2(right != left)
    # 使用测试工具库 tm 检查 result 是否等于 rev_box(expected) 的取反
    tm.assert_equal(result, rev_box(~expected))

    # 设置错误消息，包含多种错误比较情况的描述
    msg = "|".join(
        [
            "Invalid comparison between",
            "Cannot compare type",
            "not supported between",
            "invalid type promotion",
            (
                # GH#36706 npdev 1.20.0 2020-09-28
                r"The DTypes <class 'numpy.dtype\[datetime64\]'> and "
                r"<class 'numpy.dtype\[int64\]'> do not have a common DType. "
                "For example they cannot be stored in a single array unless the "
                "dtype is `object`."
            ),
        ]
    )
    # 使用 pytest 模块的 raises 函数检查以下每一种比较是否会抛出 TypeError，并匹配 msg 的错误消息
    with pytest.raises(TypeError, match=msg):
        left < right
    with pytest.raises(TypeError, match=msg):
        left <= right
    with pytest.raises(TypeError, match=msg):
        left > right
    with pytest.raises(TypeError, match=msg):
        left >= right
    with pytest.raises(TypeError, match=msg):
        right < left
    with pytest.raises(TypeError, match=msg):
        right <= left
    with pytest.raises(TypeError, match=msg):
        right > left
    with pytest.raises(TypeError, match=msg):
        right >= left
```