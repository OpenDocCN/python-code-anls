# `D:\src\scipysrc\pandas\pandas\core\ops\dispatch.py`

```
"""
Functions for defining unary operations.
"""

# 从未来导入注解，以支持类型提示
from __future__ import annotations

# 导入必要的类型提示模块
from typing import (
    TYPE_CHECKING,
    Any,
)

# 导入 Pandas 中用于定义通用数据类型的扩展数组的抽象类
from pandas.core.dtypes.generic import ABCExtensionArray

# 如果类型检查开启，导入额外的类型提示
if TYPE_CHECKING:
    from pandas._typing import ArrayLike


def should_extension_dispatch(left: ArrayLike, right: Any) -> bool:
    """
    Identify cases where Series operation should dispatch to ExtensionArray method.

    Parameters
    ----------
    left : np.ndarray or ExtensionArray
        左操作数，可以是 NumPy 数组或者扩展数组
    right : object
        右操作数

    Returns
    -------
    bool
        如果左操作数或右操作数是扩展数组，则返回 True；否则返回 False
    """
    # 判断左操作数或右操作数是否为扩展数组，如果是则返回 True，否则返回 False
    return isinstance(left, ABCExtensionArray) or isinstance(right, ABCExtensionArray)
```