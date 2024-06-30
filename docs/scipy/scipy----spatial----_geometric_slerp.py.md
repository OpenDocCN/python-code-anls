# `D:\src\scipysrc\scipy\scipy\spatial\_geometric_slerp.py`

```
from __future__ import annotations
# 允许使用类型提示中的类型作为函数返回类型的标注

__all__ = ['geometric_slerp']
# 模块导出的公共接口，包括了函数 geometric_slerp

import warnings
# 导入警告模块，用于处理警告信息

from typing import TYPE_CHECKING
# 导入 TYPE_CHECKING，用于在类型提示中进行条件导入

import numpy as np
# 导入 NumPy 库，用于数值计算

from scipy.spatial.distance import euclidean
# 从 SciPy 库中导入欧氏距离计算函数

if TYPE_CHECKING:
    import numpy.typing as npt
# 如果在类型检查模式下，则导入 NumPy 的类型提示

def _geometric_slerp(start, end, t):
    # 使用 QR 分解创建正交基
    basis = np.vstack([start, end])
    Q, R = np.linalg.qr(basis.T)
    signs = 2 * (np.diag(R) >= 0) - 1
    Q = Q.T * signs.T[:, np.newaxis]
    R = R.T * signs.T[:, np.newaxis]

    # 计算 `start` 和 `end` 之间的角度
    c = np.dot(start, end)
    s = np.linalg.det(R)
    omega = np.arctan2(s, c)

    # 插值
    start, end = Q
    s = np.sin(t * omega)
    c = np.cos(t * omega)
    return start * c[:, np.newaxis] + end * s[:, np.newaxis]
# 使用几何球面线性插值（Geometric Slerp）计算两个向量之间的插值路径

def geometric_slerp(
    start: npt.ArrayLike,
    end: npt.ArrayLike,
    t: npt.ArrayLike,
    tol: float = 1e-7,
) -> np.ndarray:
    """
    几何球面线性插值（Geometric Slerp）。

    插值沿着单位半径大圆弧在任意维度空间中进行。

    Parameters
    ----------
    start : (n_dimensions, ) array-like
        单个 n 维输入坐标，作为 1 维数组对象。
        `n` 必须大于 1。
    end : (n_dimensions, ) array-like
        单个 n 维输入坐标，作为 1 维数组对象。
        `n` 必须大于 1。
    t : float or (n_points,) 1D array-like
        一组双精度浮点数的浮点数或 1 维数组，表示插值参数。
        值应在 0 到 1 的闭区间内。常见的方法是使用 ``np.linspace(0, 1, n_pts)`` 
        生成线性间隔的点数组。允许升序、降序和混乱顺序。
    tol : float
        确定起始点和终点坐标是否为对点的绝对容差。

    Returns
    -------
    result : (t.size, D)
        包含插值球面路径的双精度浮点数数组，
        当 t 使用 0 和 1 时包括起始点和终点。
        插值结果应与 t 数组提供的排序顺序相对应。
        如果 ``t`` 是一个浮点数，则结果可能是 1 维的。

    Raises
    ------
    ValueError
        如果 `start` 和 `end` 是对点，不在单位 n 球面上，
        或存在各种退化条件。

    See Also
    --------
    scipy.spatial.transform.Slerp : 使用四元数的 3D Slerp

    Notes
    -----
    该实现基于 [1]_ 中提供的数学公式，
    这种算法的首次已知介绍来自于 4D 几何的研究，
    由 Glenn Davis 在 Ken Shoemake 的原始四元数 Slerp 发表的脚注中推导而来 [2]_。

    .. versionadded:: 1.5.0

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Slerp#Geometric_Slerp
    """
    # 函数声明：几何球面线性插值（Geometric Slerp）
    pass
    # 将输入的起点和终点转换为 NumPy 数组，并确保数据类型为 float64
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    # 将参数 t 转换为 NumPy 数组
    t = np.asarray(t)
    # 检查插值参数的维度是否大于1，如果是则引发异常
    if t.ndim > 1:
        raise ValueError("The interpolation parameter "
                         "value must be one dimensional.")

    # 检查起始和结束坐标是否为一维数组，否则引发异常
    if start.ndim != 1 or end.ndim != 1:
        raise ValueError("Start and end coordinates "
                         "must be one-dimensional")

    # 检查起始和结束坐标的大小是否相同，如果不同则引发异常
    if start.size != end.size:
        raise ValueError("The dimensions of start and "
                         "end must match (have same size)")

    # 检查起始和结束坐标的大小是否至少为2，如果不是则引发异常
    if start.size < 2 or end.size < 2:
        raise ValueError("The start and end coordinates must "
                         "both be in at least two-dimensional "
                         "space")

    # 检查起始和结束坐标是否完全相等，如果是则返回一个包含起始值的数组
    if np.array_equal(start, end):
        return np.linspace(start, start, t.size)

    # 检查起始和结束坐标是否在单位n-球面上，如果不是则引发异常
    # 使用np.allclose检查是否在给定的容差(tol)内
    for coord in [start, end]:
        if not np.allclose(np.linalg.norm(coord), 1.0,
                           rtol=1e-9,
                           atol=0):
            raise ValueError("start and end are not"
                             " on a unit n-sphere")

    # 检查tol是否为float类型，如果不是则引发异常；否则取其绝对值
    if not isinstance(tol, float):
        raise ValueError("tol must be a float")
    else:
        tol = np.fabs(tol)

    # 计算起始点和结束点之间的欧氏距离
    coord_dist = euclidean(start, end)

    # 如果起始点和结束点之间的距离等于2.0在给定容差(tol)内，则发出警告
    if np.allclose(coord_dist, 2.0, rtol=0, atol=tol):
        warnings.warn("start and end are antipodes "
                      "using the specified tolerance; "
                      "this may cause ambiguous slerp paths",
                      stacklevel=2)

    # 将插值参数t转换为numpy数组，数据类型为np.float64
    t = np.asarray(t, dtype=np.float64)

    # 如果插值参数t的大小为0，则返回一个空的数组，其形状与起始点相同
    if t.size == 0:
        return np.empty((0, start.size))

    # 检查插值参数是否在区间[0, 1]内，如果不是则引发异常
    if t.min() < 0 or t.max() > 1:
        raise ValueError("interpolation parameter must be in [0, 1]")

    # 如果插值参数t的维度为0，则调用_geometric_slerp函数进行插值计算并展平结果
    if t.ndim == 0:
        return _geometric_slerp(start,
                                end,
                                np.atleast_1d(t)).ravel()
    else:
        # 否则调用_geometric_slerp函数进行插值计算
        return _geometric_slerp(start,
                                end,
                                t)
```