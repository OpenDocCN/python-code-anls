# `D:\src\scipysrc\pandas\pandas\core\reshape\util.py`

```
from __future__ import annotations  # 导入用于支持类型注释的未来版本特性

from typing import TYPE_CHECKING  # 导入用于类型检查的模块

import numpy as np  # 导入 NumPy 库

from pandas.core.dtypes.common import is_list_like  # 从 Pandas 中导入判断是否为列表样式的函数

if TYPE_CHECKING:
    from pandas._typing import NumpyIndexT  # 如果是类型检查模式，导入 NumPy 索引类型

def cartesian_product(X) -> list[np.ndarray]:
    """
    Numpy version of itertools.product.
    Sometimes faster (for large inputs)...

    Parameters
    ----------
    X : list-like of list-likes  # 参数 X 应为列表样式的列表

    Returns
    -------
    product : list of ndarrays  # 返回一个 ndarray 列表

    Examples
    --------
    >>> cartesian_product([list("ABC"), [1, 2]])
    [array(['A', 'A', 'B', 'B', 'C', 'C'], dtype='<U1'), array([1, 2, 1, 2, 1, 2])]

    See Also
    --------
    itertools.product : Cartesian product of input iterables.  Equivalent to
        nested for-loops.
    """
    msg = "Input must be a list-like of list-likes"  # 错误消息：输入必须是列表样式的列表
    if not is_list_like(X):  # 如果 X 不是列表样式的
        raise TypeError(msg)  # 抛出类型错误异常

    for x in X:  # 遍历 X 中的每个元素 x
        if not is_list_like(x):  # 如果 x 不是列表样式的
            raise TypeError(msg)  # 抛出类型错误异常

    if len(X) == 0:  # 如果 X 的长度为 0
        return []  # 返回空列表

    lenX = np.fromiter((len(x) for x in X), dtype=np.intp)  # 创建一个数组，其中包含 X 中每个元素的长度
    cumprodX = np.cumprod(lenX)  # 计算长度数组的累积乘积

    if np.any(cumprodX < 0):  # 如果任何累积乘积小于 0
        raise ValueError("Product space too large to allocate arrays!")  # 抛出数值错误异常：无法分配数组的产品空间太大

    a = np.roll(cumprodX, 1)  # 将累积乘积向左循环移位一位
    a[0] = 1  # 将第一个元素设置为 1

    if cumprodX[-1] != 0:  # 如果累积乘积的最后一个元素不为 0
        b = cumprodX[-1] / cumprodX  # 计算累积乘积的最后一个元素除以累积乘积的每个元素
    else:
        # if any factor is empty, the cartesian product is empty
        b = np.zeros_like(cumprodX)  # 如果任何因子为空，则返回与 cumprodX 相同形状的零数组

    # error: Argument of type "int_" cannot be assigned to parameter "num" of
    # type "int" in function "tile_compat"
    return [
        tile_compat(
            np.repeat(x, b[i]),  # 使用 b[i] 重复 x 中的元素
            np.prod(a[i]),  # 计算 a[i] 的乘积
        )
        for i, x in enumerate(X)  # 遍历 X 中的每个元素 x，并返回由 tile_compat 处理的列表
    ]


def tile_compat(arr: NumpyIndexT, num: int) -> NumpyIndexT:
    """
    Index compat for np.tile.

    Notes
    -----
    Does not support multi-dimensional `num`.
    """
    if isinstance(arr, np.ndarray):  # 如果 arr 是 NumPy 数组
        return np.tile(arr, num)  # 使用 np.tile 扩展 arr

    # Otherwise we have an Index
    taker = np.tile(np.arange(len(arr)), num)  # 使用 np.tile 扩展 arr 的索引
    return arr.take(taker)  # 返回扩展后的 arr
```