# `D:\src\scipysrc\pandas\pandas\core\ops\invalid.py`

```
"""
Templates for invalid operations.
"""

from __future__ import annotations  # 导入未来版本的注解支持

import operator  # 导入操作符模块
from typing import (  # 导入类型提示模块
    TYPE_CHECKING,
    Any,
    NoReturn,
)

import numpy as np  # 导入 NumPy 库

if TYPE_CHECKING:
    from collections.abc import Callable  # 导入可调用对象抽象基类

    from pandas._typing import (  # 导入 Pandas 类型定义
        ArrayLike,
        Scalar,
        npt,
    )


def invalid_comparison(  # 定义函数：无效比较
    left: ArrayLike,  # 左操作数，类数组类型
    right: ArrayLike | Scalar,  # 右操作数，标量或类数组类型
    op: Callable[[Any, Any], bool],  # 操作符函数类型，接受两个参数，返回布尔值
) -> npt.NDArray[np.bool_]:  # 返回 NumPy 布尔数组
    """
    If a comparison has mismatched types and is not necessarily meaningful,
    follow python3 conventions by:

        - returning all-False for equality
        - returning all-True for inequality
        - raising TypeError otherwise

    Parameters
    ----------
    left : array-like
        Left operand of the comparison.
    right : scalar, array-like
        Right operand of the comparison.
    op : operator.{eq, ne, lt, le, gt}
        Comparison operator function.

    Raises
    ------
    TypeError : on inequality comparisons
        Raised when the comparison is invalid due to type mismatch.
    """
    if op is operator.eq:  # 如果操作符是等于操作
        res_values = np.zeros(left.shape, dtype=bool)  # 创建全为 False 的布尔数组
    elif op is operator.ne:  # 如果操作符是不等于操作
        res_values = np.ones(left.shape, dtype=bool)  # 创建全为 True 的布尔数组
    else:  # 其他情况
        typ = type(right).__name__  # 获取右操作数的类型名
        raise TypeError(f"Invalid comparison between dtype={left.dtype} and {typ}")  # 抛出类型错误异常
    return res_values  # 返回结果数组


def make_invalid_op(name: str) -> Callable[..., NoReturn]:
    """
    Return a binary method that always raises a TypeError.

    Parameters
    ----------
    name : str
        Name of the invalid operation.

    Returns
    -------
    invalid_op : function
        Invalid operation function that raises TypeError.
    """

    def invalid_op(self: object, other: object = None) -> NoReturn:
        typ = type(self).__name__  # 获取调用对象的类型名
        raise TypeError(f"cannot perform {name} with this index type: {typ}")  # 抛出类型错误异常

    invalid_op.__name__ = name  # 设置函数名
    return invalid_op  # 返回无效操作函数
```