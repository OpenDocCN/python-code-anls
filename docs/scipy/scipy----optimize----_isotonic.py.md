# `D:\src\scipysrc\scipy\scipy\optimize\_isotonic.py`

```
# 导入必要的模块和类型注解
from __future__ import annotations
from typing import TYPE_CHECKING

# 导入 NumPy 库并命名为 np
import numpy as np

# 从本地模块中导入 OptimizeResult 类
from ._optimize import OptimizeResult
# 从本地模块中导入 C++ 绑定模块 pava
from ._pava_pybind import pava

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 导入 NumPy 类型提示
    import numpy.typing as npt

# 定义模块导出的公共接口
__all__ = ["isotonic_regression"]

# 定义非参数性等距回归函数 isotonic_regression
def isotonic_regression(
    y: npt.ArrayLike,
    *,
    weights: npt.ArrayLike | None = None,
    increasing: bool = True,
) -> OptimizeResult:
    r"""Nonparametric isotonic regression.

    使用邻近污染者池算法（PAVA）计算响应变量 y 的（非严格）单调递增数组 x，详见[1]_。详见注释部分获取更多细节。

    Parameters
    ----------
    y : (N,) array_like
        响应变量。
    weights : (N,) array_like or None
        案例权重。
    increasing : bool
        如果为 True，则拟合单调递增（等距）回归。
        如果为 False，则拟合单调递减（反等距）回归。
        默认为 True。

    Returns
    -------
    res : OptimizeResult
        优化结果表示为 ``OptimizeResult`` 对象。
        重要属性包括：

        - ``x``: 等距回归解，即与 y 长度相同的递增（或递减）数组，其元素范围从 min(y) 到 max(y)。
        - ``weights`` : 每个池（或块）B的案例权重总和的数组。
        - ``blocks``: 长度为 B+1 的数组，其中包含每个池（或块）B的起始位置的索引。
          第 j 个块由 ``x[blocks[j]:blocks[j+1]]`` 给出，其中所有值相同。

    Notes
    -----
    给定数据 :math:`y` 和案例权重 :math:`w`，等距回归解决以下优化问题：

    .. math::

        \operatorname{argmin}_{x_i} \sum_i w_i (y_i - x_i)^2 \quad
        \text{subject to } x_i \leq x_j \text{ whenever } i \leq j \,.

    对于每个输入值 :math:`y_i`，它生成一个值 :math:`x_i`，使得 :math:`x` 是递增的（但不严格），即 :math:`x_i \leq x_{i+1}`。
    这通过 PAVA 实现。解决方案由池或块组成，即 :math:`x` 的相邻元素，例如 :math:`x_i` 和 :math:`x_{i+1}`，其所有值相同。

    最有趣的是，如果用广泛的 Bregman 函数替换平方损失，解决方案保持不变，这些函数是用于均值的唯一严格一致的评分函数类，详见[2]_及其引用。

    根据[1]_中的实现，PAVA的计算复杂度为 O(N)，输入大小为 N。

    References
    ----------
    .. [1] Busing, F. M. T. A. (2022).
           Monotone Regression: A Simple and Fast O(n) PAVA Implementation.
           Journal of Statistical Software, Code Snippets, 102(1), 1-25.
           :doi:`10.18637/jss.v102.c01`
    # 将输入的 y 强制转换为至少一维数组
    yarr = np.atleast_1d(y)  # Check yarr.ndim == 1 is implicit (pybind11) in pava.
    # 根据是否增序选择切片顺序
    order = slice(None) if increasing else slice(None, None, -1)
    # 根据顺序切片获取数组 x，并指定 C 风格的内存顺序和浮点数类型
    x = np.array(yarr[order], order="C", dtype=np.float64, copy=True)
    # 如果权重为 None，则初始化权重数组 wx 为与 yarr 相同形状的全一数组
    if weights is None:
        wx = np.ones_like(yarr, dtype=np.float64)
    else:
        # 将输入的 weights 强制转换为至少一维数组
        warr = np.atleast_1d(weights)

        # 检查 yarr 和 warr 是否都是一维数组且长度相同，否则抛出异常
        if not (yarr.ndim == warr.ndim == 1 and yarr.shape[0] == warr.shape[0]):
            raise ValueError(
                "Input arrays y and w must have one dimension of equal length."
            )
        # 检查权重数组中是否有非正数值，如果有则抛出异常
        if np.any(warr <= 0):
            raise ValueError("Weights w must be strictly positive.")

        # 根据顺序切片获取数组 wx，并指定 C 风格的内存顺序和浮点数类型
        wx = np.array(warr[order], order="C", dtype=np.float64, copy=True)
    # 获取数组 x 的长度
    n = x.shape[0]
    # 创建全为 -1 的整数数组 r，长度比 x 多 1
    r = np.full(shape=n + 1, fill_value=-1, dtype=np.intp)
    # 调用 pava 函数计算并返回结果数组 x、权重数组 wx、分块信息数组 r 和分块数目 b
    x, wx, r, b = pava(x, wx, r)
    # 根据 pava 的结果，截取有效的分块信息数组 r 和权重数组 wx 的部分
    r = r[:b + 1]
    wx = wx[:b]
    # 如果不是按照递增顺序排列，则反转以下变量：x, wx, r
    if not increasing:
        x = x[::-1]     # 反转 x 数组
        wx = wx[::-1]   # 反转 wx 数组
        r = r[-1] - r[::-1]  # 计算 r 数组的差分（最后一个元素减去整体反转后的数组）
    # 返回优化结果对象，包括 x 数组，权重数组 wx 和块数组 r
    return OptimizeResult(
        x=x,
        weights=wx,
        blocks=r,
    )
```