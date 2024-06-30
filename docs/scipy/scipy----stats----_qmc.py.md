# `D:\src\scipysrc\scipy\scipy\stats\_qmc.py`

```
# 引入未来的注释风格，支持类型提示
from __future__ import annotations

# 引入必要的库和模块
import copy  # 导入 copy 模块，用于复制对象
import math  # 导入 math 模块，提供数学函数
import numbers  # 导入 numbers 模块，用于处理数字类型
import os  # 导入 os 模块，提供与操作系统交互的功能
import warnings  # 导入 warnings 模块，用于管理警告信息
from abc import ABC, abstractmethod  # 导入 ABC 和 abstractmethod 类，用于抽象类和抽象方法的定义
from functools import partial  # 导入 partial 函数，用于部分函数应用
from typing import (  # 导入类型提示所需的各种类型
    Callable,  # 可调用对象类型
    ClassVar,  # 类变量类型
    Literal,  # 字面量类型
    overload,  # 函数重载装饰器
    TYPE_CHECKING,  # 类型检查标记
)

import numpy as np  # 导入 NumPy 库，重命名为 np

if TYPE_CHECKING:
    import numpy.typing as npt  # 如果是类型检查，导入 numpy.typing 模块
    from scipy._lib._util import (  # 导入 scipy._lib._util 模块中的特定类型
        DecimalNumber, GeneratorType, IntNumber, SeedType
    )

import scipy.stats as stats  # 导入 scipy.stats 模块，重命名为 stats
from scipy._lib._util import rng_integers, _rng_spawn  # 从 scipy._lib._util 导入两个函数
from scipy.sparse.csgraph import minimum_spanning_tree  # 从 scipy.sparse.csgraph 导入 minimum_spanning_tree 函数
from scipy.spatial import distance, Voronoi  # 导入 scipy.spatial 中的 distance 和 Voronoi 函数
from scipy.special import gammainc  # 导入 scipy.special 模块中的 gammainc 函数
from ._sobol import (  # 从当前模块的 _sobol 子模块中导入多个函数
    _initialize_v, _cscramble, _fill_p_cumulative, _draw, _fast_forward,
    _categorize, _MAXDIM
)
from ._qmc_cy import (  # 从当前模块的 _qmc_cy 子模块中导入多个函数
    _cy_wrapper_centered_discrepancy,
    _cy_wrapper_wrap_around_discrepancy,
    _cy_wrapper_mixture_discrepancy,
    _cy_wrapper_l2_star_discrepancy,
    _cy_wrapper_update_discrepancy,
    _cy_van_der_corput_scrambled,
    _cy_van_der_corput,
)

# 定义 __all__ 列表，包含模块中公开的所有符号
__all__ = ['scale', 'discrepancy', 'geometric_discrepancy', 'update_discrepancy',
           'QMCEngine', 'Sobol', 'Halton', 'LatinHypercube', 'PoissonDisk',
           'MultinomialQMC', 'MultivariateNormalQMC']


@overload
def check_random_state(seed: IntNumber | None = ...) -> np.random.Generator:
    ...


@overload
def check_random_state(seed: GeneratorType) -> GeneratorType:
    ...


# 基于 scipy._lib._util.check_random_state 函数定义的检查随机数生成器函数
def check_random_state(seed=None):
    """Turn `seed` into a `numpy.random.Generator` instance.

    Parameters
    ----------
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is an int or None, a new `numpy.random.Generator` is
        created using ``np.random.default_rng(seed)``.
        If `seed` is already a ``Generator`` or ``RandomState`` instance, then
        the provided instance is used.

    Returns
    -------
    seed : {`numpy.random.Generator`, `numpy.random.RandomState`}
        Random number generator.

    """
    if seed is None or isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.default_rng(seed)  # 如果 seed 是整数或 None，创建一个新的 Generator
    elif isinstance(seed, (np.random.RandomState, np.random.Generator)):
        return seed  # 如果 seed 已经是 Generator 或 RandomState 实例，则直接返回
    else:
        raise ValueError(f'{seed!r} cannot be used to seed a'
                         ' numpy.random.Generator instance')


# 定义 scale 函数，用于将样本从单位超立方体缩放到不同的边界
def scale(
    sample: npt.ArrayLike,  # 样本数组，可以是 NumPy 数组或类似数组
    l_bounds: npt.ArrayLike,  # 下界数组，可以是 NumPy 数组或类似数组
    u_bounds: npt.ArrayLike,  # 上界数组，可以是 NumPy 数组或类似数组
    *,
    reverse: bool = False  # 可选参数，是否反向缩放，默认为 False
) -> np.ndarray:
    r"""Sample scaling from unit hypercube to different bounds.

    To convert a sample from :math:`[0, 1)` to :math:`[a, b), b>a`,
    with :math:`a` the lower bounds and :math:`b` the upper bounds.
    The following transformation is used:

    .. math::

        (b - a) \cdot \text{sample} + a

    Parameters
    ----------
    sample : array_like (n, d)
        Sample to scale.
    sample = np.asarray(sample)

将输入的样本数据转换为NumPy数组格式，以便后续处理。


    # Checking bounds and sample
    if not sample.ndim == 2:
        raise ValueError('Sample is not a 2D array')

检查样本数据是否为二维数组，如果不是，则抛出数值错误异常。


    lower, upper = _validate_bounds(
        l_bounds=l_bounds, u_bounds=u_bounds, d=sample.shape[1]
    )

调用 `_validate_bounds` 函数，验证并获取转换的下界和上界。


    if not reverse:
        # Checking that sample is within the hypercube
        if (sample.max() > 1.) or (sample.min() < 0.):
            raise ValueError('Sample is not in unit hypercube')

        return sample * (upper - lower) + lower

如果 `reverse` 参数为 `False`，则检查样本数据是否在单位超立方体内，如果不在范围内则抛出异常；然后将样本数据转换为给定的下界和上界之间的范围。


    else:
        # Checking that sample is within the bounds
        if not (np.all(sample >= lower) and np.all(sample <= upper)):
            raise ValueError('Sample is out of bounds')

        return (sample - lower) / (upper - lower)

如果 `reverse` 参数为 `True`，则检查样本数据是否在指定的下界和上界之间，如果不在范围内则抛出异常；然后将样本数据反向转换回单位超立方体内的范围。
# 确保样本是一个二维数组，并且在单位超立方体内
def _ensure_in_unit_hypercube(sample: npt.ArrayLike) -> np.ndarray:
    # 将输入样本转换为二维数组，使用 C 风格的内存布局，数据类型为 np.float64
    sample = np.asarray(sample, dtype=np.float64, order="C")

    # 如果样本不是二维数组，则抛出数值错误异常
    if not sample.ndim == 2:
        raise ValueError("Sample is not a 2D array")

    # 如果样本中存在超出单位超立方体范围的点，则抛出数值错误异常
    if (sample.max() > 1.) or (sample.min() < 0.):
        raise ValueError("Sample is not in unit hypercube")

    # 返回转换后的样本数组
    return sample


# 计算给定样本的离散度
def discrepancy(
        sample: npt.ArrayLike,
        *,
        iterative: bool = False,
        method: Literal["CD", "WD", "MD", "L2-star"] = "CD",
        workers: IntNumber = 1) -> float:
    """Discrepancy of a given sample.

    Parameters
    ----------
    sample : array_like (n, d)
        The sample to compute the discrepancy from.
    iterative : bool, optional
        Must be False if not using it for updating the discrepancy.
        Default is False. Refer to the notes for more details.
    method : str, optional
        Type of discrepancy, can be ``CD``, ``WD``, ``MD`` or ``L2-star``.
        Refer to the notes for more details. Default is ``CD``.
    workers : int, optional
        Number of workers to use for parallel processing. If -1 is given all
        CPU threads are used. Default is 1.

    Returns
    -------
    discrepancy : float
        Discrepancy.

    See Also
    --------
    geometric_discrepancy

    Notes
    -----
    The discrepancy is a uniformity criterion used to assess the space filling
    of a number of samples in a hypercube. A discrepancy quantifies the
    distance between the continuous uniform distribution on a hypercube and the
    discrete uniform distribution on :math:`n` distinct sample points.

    The lower the value is, the better the coverage of the parameter space is.

    For a collection of subsets of the hypercube, the discrepancy is the
    difference between the fraction of sample points in one of those
    subsets and the volume of that subset. There are different definitions of
    discrepancy corresponding to different collections of subsets. Some
    versions take a root mean square difference over subsets instead of
    a maximum.

    A measure of uniformity is reasonable if it satisfies the following
    criteria [1]_:

    1. It is invariant under permuting factors and/or runs.
    2. It is invariant under rotation of the coordinates.
    3. It can measure not only uniformity of the sample over the hypercube,
       but also the projection uniformity of the sample over non-empty
       subset of lower dimension hypercubes.
    4. There is some reasonable geometric meaning.
    5. It is easy to compute.
    """
    pass
    # 确保样本在单位超立方体内，进行归一化处理
    sample = _ensure_in_unit_hypercube(sample)

    # 验证并确定工作线程数量
    workers = _validate_workers(workers)

    # 定义四种不同的方法和它们对应的函数
    methods = {
        "CD": _cy_wrapper_centered_discrepancy,
        "WD": _cy_wrapper_wrap_around_discrepancy,
        "MD": _cy_wrapper_mixture_discrepancy,
        "L2-star": _cy_wrapper_l2_star_discrepancy,
    }

    # 如果指定的方法存在于方法字典中，则调用对应的函数进行计算
    if method in methods:
        return methods[method](sample, iterative, workers=workers)
    else:
        # 如果指定的方法不在方法字典中，则抛出值错误异常
        raise ValueError(f"{method!r} is not a valid method. It must be one of"
                         f" {set(methods)!r}")
def geometric_discrepancy(
        sample: npt.ArrayLike,
        method: Literal["mindist", "mst"] = "mindist",
        metric: str = "euclidean") -> float:
    """计算给定样本的几何差异度。

    Parameters
    ----------
    sample : array_like (n, d)
        要计算差异度的样本数据。
    method : {"mindist", "mst"}, optional
        使用的方法之一。选择 ``mindist`` 表示最小距离（默认），
        或者选择 ``mst`` 表示最小生成树。
    metric : str or callable, optional
        距离度量方式。参见 `scipy.spatial.distance.pdist` 的文档
        获取可用的度量方式及默认设置。

    Returns
    -------
    discrepancy : float
        差异度（值越大表示样本更均匀）。

    See Also
    --------
    discrepancy

    Notes
    -----
    差异度可以作为衡量随机样本质量的简单指标。
    此指标基于样本中点分布的几何特性，例如任意两点间的最小距离，
    或者最小生成树中平均边长。

    值越高，参数空间的覆盖越好。
    注意，这与 `scipy.stats.qmc.discrepancy` 不同，后者较低的值
    表示样本质量较高。

    此外，在比较不同抽样策略时，必须保持样本大小不变。

    可以从最小生成树中计算两种指标：
    平均边长和边长的标准差。同时使用两个指标比单独使用任一指标
    更能反映均匀性，其中平均值较高且标准差较低更为理想（详见 [1]_）。
    本函数目前仅计算平均边长。

    References
    ----------
    .. [1] Franco J. et al. "Minimum Spanning Tree: A new approach to assess the quality
       of the design of computer experiments." Chemometrics and Intelligent Laboratory
       Systems, 97 (2), pp. 164-169, 2009.

    Examples
    --------
    使用最小欧氏距离（默认设置）计算样本质量：

    >>> import numpy as np
    >>> from scipy.stats import qmc
    >>> rng = np.random.default_rng(191468432622931918890291693003068437394)
    >>> sample = qmc.LatinHypercube(d=2, seed=rng).random(50)
    >>> qmc.geometric_discrepancy(sample)
    0.03708161435687876

    使用最小生成树中的平均边长计算样本质量：

    >>> qmc.geometric_discrepancy(sample, method='mst')
    0.1105149978798376

    显示最小生成树及最小距离的点：

    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.lines import Line2D
    >>> from scipy.sparse.csgraph import minimum_spanning_tree
    >>> from scipy.spatial.distance import pdist, squareform
    >>> dist = pdist(sample)  # 计算样本点之间的距离
    >>> mst = minimum_spanning_tree(squareform(dist))  # 构建距离矩阵的最小生成树
    >>> edges = np.where(mst.toarray() > 0)  # 提取最小生成树的边
    >>> edges = np.asarray(edges).T  # 转换边的格式为numpy数组
    >>> min_dist = np.min(dist)  # 计算距离矩阵中的最小距离
    >>> min_idx = np.argwhere(squareform(dist) == min_dist)[0]  # 找到最小距离对应的索引
    >>> fig, ax = plt.subplots(figsize=(10, 5))  # 创建画布和子图
    >>> _ = ax.set(aspect='equal', xlabel=r'$x_1$', ylabel=r'$x_2$',  # 设置坐标轴标签和纵横比
    ...            xlim=[0, 1], ylim=[0, 1])
    >>> for edge in edges:  # 遍历最小生成树的边
    ...     ax.plot(sample[edge, 0], sample[edge, 1], c='k')  # 在子图上绘制边
    >>> ax.scatter(sample[:, 0], sample[:, 1])  # 在子图上绘制样本点的散点图
    >>> ax.add_patch(plt.Circle(sample[min_idx[0]], min_dist, color='red', fill=False))  # 在子图上添加红色圆圈表示最小距离
    >>> markers = [  # 定义图例中的标记
    ...     Line2D([0], [0], marker='o', lw=0, label='Sample points'),  # 样本点标记
    ...     Line2D([0], [0], color='k', label='Minimum spanning tree'),  # 最小生成树标记
    ...     Line2D([0], [0], marker='o', lw=0, markerfacecolor='w', markeredgecolor='r',
    ...            label='Minimum point-to-point distance'),  # 最小点到点距离标记
    ... ]
    >>> ax.legend(handles=markers, loc='center left', bbox_to_anchor=(1, 0.5));  # 添加图例并设置位置

    """
    sample = _ensure_in_unit_hypercube(sample)  # 将样本点映射到单位超立方体内
    if sample.shape[0] < 2:  # 如果样本点数量少于2个，抛出数值错误
        raise ValueError("Sample must contain at least two points")

    distances = distance.pdist(sample, metric=metric)  # 计算样本点之间的距离

    if np.any(distances == 0.0):  # 如果距离中存在为零的值，发出警告
        warnings.warn("Sample contains duplicate points.", stacklevel=2)

    if method == "mindist":  # 如果方法选择最小距离
        return np.min(distances[distances.nonzero()])  # 返回非零距离的最小值
    elif method == "mst":  # 如果方法选择最小生成树
        fully_connected_graph = distance.squareform(distances)
        mst = minimum_spanning_tree(fully_connected_graph)  # 构建最小生成树
        distances = mst[mst.nonzero()]
        # TODO consider returning both the mean and the standard deviation
        # see [1] for a discussion
        return np.mean(distances)  # 返回最小生成树的平均权重
    else:
        raise ValueError(f"{method!r} is not a valid method. "  # 抛出数值错误，方法不合法
                         f"It must be one of {{'mindist', 'mst'}}")
# 更新带有新样本的中心差异度。

Parameters
----------
x_new : array_like (1, d)
    要添加到 `sample` 中的新样本。
sample : array_like (n, d)
    初始样本。
initial_disc : float
    `sample` 的中心差异度。

Returns
-------
discrepancy : float
    组成由 `x_new` 和 `sample` 组成的样本的中心差异度。

Examples
--------
我们还可以通过 ``iterative=True`` 来迭代计算差异度。

>>> import numpy as np
>>> from scipy.stats import qmc
>>> space = np.array([[1, 3], [2, 6], [3, 2], [4, 5], [5, 1], [6, 4]])
>>> l_bounds = [0.5, 0.5]
>>> u_bounds = [6.5, 6.5]
>>> space = qmc.scale(space, l_bounds, u_bounds, reverse=True)
>>> disc_init = qmc.discrepancy(space[:-1], iterative=True)
>>> disc_init
0.04769081147119336
>>> qmc.update_discrepancy(space[-1], space[:-1], disc_init)
0.008142039609053513

"""
# 将样本转换为 np.ndarray 类型，使用 float64 数据类型，以 C 风格存储
sample = np.asarray(sample, dtype=np.float64, order="C")
# 将 x_new 转换为 np.ndarray 类型，使用 float64 数据类型，以 C 风格存储
x_new = np.asarray(x_new, dtype=np.float64, order="C")

# 检查样本是否为二维数组
if not sample.ndim == 2:
    raise ValueError('Sample is not a 2D array')

# 检查样本是否在单位超立方体内，并且取值为二维
if (sample.max() > 1.) or (sample.min() < 0.):
    raise ValueError('Sample is not in unit hypercube')

# 检查 x_new 是否在单位超立方体内，并且取值为一维
if not x_new.ndim == 1:
    raise ValueError('x_new is not a 1D array')

# 检查 x_new 是否在单位超立方体内
if not (np.all(x_new >= 0) and np.all(x_new <= 1)):
    raise ValueError('x_new is not in unit hypercube')

# 检查 x_new 和 sample 的形状是否可以进行广播
if x_new.shape[0] != sample.shape[1]:
    raise ValueError("x_new and sample must be broadcastable")

# 调用底层函数计算更新后的差异度
return _cy_wrapper_update_discrepancy(x_new, sample, initial_disc)
    """
    n = sample.shape[0]  # 获取样本的行数，即样本数量 n

    z_ij = sample - 0.5  # 将样本中每个元素减去0.5，得到新的矩阵 z_ij

    # Eq (19)
    c_i1j = (1. / n ** 2.
             * np.prod(0.5 * (2. + abs(z_ij[i1, :])
                              + abs(z_ij) - abs(z_ij[i1, :] - z_ij)), axis=1))
    c_i2j = (1. / n ** 2.
             * np.prod(0.5 * (2. + abs(z_ij[i2, :])
                              + abs(z_ij) - abs(z_ij[i2, :] - z_ij)), axis=1))
    # 计算 c_i1j 和 c_i2j，这是根据公式 (19) 计算的两个系数

    # Eq (20)
    c_i1i1 = (1. / n ** 2 * np.prod(1 + abs(z_ij[i1, :]))
              - 2. / n * np.prod(1. + 0.5 * abs(z_ij[i1, :])
                                 - 0.5 * z_ij[i1, :] ** 2))
    c_i2i2 = (1. / n ** 2 * np.prod(1 + abs(z_ij[i2, :]))
              - 2. / n * np.prod(1. + 0.5 * abs(z_ij[i2, :])
                                 - 0.5 * z_ij[i2, :] ** 2))
    # 计算 c_i1i1 和 c_i2i2，这是根据公式 (20) 计算的两个系数

    # Eq (22), typo in the article in the denominator i2 -> i1
    num = (2 + abs(z_ij[i2, k]) + abs(z_ij[:, k])
           - abs(z_ij[i2, k] - z_ij[:, k]))
    denum = (2 + abs(z_ij[i1, k]) + abs(z_ij[:, k])
             - abs(z_ij[i1, k] - z_ij[:, k]))
    gamma = num / denum
    # 计算 gamma，这是根据公式 (22) 计算的比率

    # Eq (23)
    c_p_i1j = gamma * c_i1j
    # 计算 c_p_i1j，这是根据公式 (23) 计算的值

    # Eq (24)
    c_p_i2j = c_i2j / gamma
    # 计算 c_p_i2j，这是根据公式 (24) 计算的值

    alpha = (1 + abs(z_ij[i2, k])) / (1 + abs(z_ij[i1, k]))
    beta = (2 - abs(z_ij[i2, k])) / (2 - abs(z_ij[i1, k]))

    g_i1 = np.prod(1. + abs(z_ij[i1, :]))
    g_i2 = np.prod(1. + abs(z_ij[i2, :]))
    h_i1 = np.prod(1. + 0.5 * abs(z_ij[i1, :]) - 0.5 * (z_ij[i1, :] ** 2))
    h_i2 = np.prod(1. + 0.5 * abs(z_ij[i2, :]) - 0.5 * (z_ij[i2, :] ** 2))

    # Eq (25), typo in the article g is missing
    c_p_i1i1 = ((g_i1 * alpha) / (n ** 2) - 2. * alpha * beta * h_i1 / n)
    # 计算 c_p_i1i1，这是根据公式 (25) 计算的值

    # Eq (26), typo in the article n ** 2
    c_p_i2i2 = ((g_i2 / ((n ** 2) * alpha)) - (2. * h_i2 / (n * alpha * beta)))
    # 计算 c_p_i2i2，这是根据公式 (26) 计算的值

    # Eq (26)
    sum_ = c_p_i1j - c_i1j + c_p_i2j - c_i2j
    # 计算 sum_，这是根据公式 (26) 计算的值

    mask = np.ones(n, dtype=bool)
    mask[[i1, i2]] = False
    sum_ = sum(sum_[mask])
    # 使用掩码从 sum_ 中排除索引 i1 和 i2，然后计算剩余元素的和

    disc_ep = (disc + c_p_i1i1 - c_i1i1 + c_p_i2i2 - c_i2i2 + 2 * sum_)
    # 计算最终的 disc_ep，这是根据给定公式计算的最终结果

    return disc_ep
    ```
    # 创建一个布尔数组，用于筛选质数，数组长度为 n // 3 + (n % 6 == 2)
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=bool)
    # 循环直到根号 n 为止
    for i in range(1, int(n ** 0.5) // 3 + 1):
        # 计算当前循环下的 k 值，3 * i + 1 | 1 是数学运算
        k = 3 * i + 1 | 1
        # 排除 k*k 的倍数
        sieve[k * k // 3::2 * k] = False
        # 排除 k*(k-2*(i&1)+4) 的倍数
        sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
    # 返回所有质数，包括小于 n 的第一个两个特例：2 和 3，以及使用筛法得到的其他质数
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]


# 返回前 n 个质数的列表
def n_primes(n: IntNumber) -> list[int]:
    # 直接给出的前 n 个质数
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
              61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127,
              131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
              197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269,
              271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
              353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431,
              433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
              509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599,
              601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673,
              677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761,
              769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857,
              859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947,
              953, 967, 971, 977, 983, 991, 997][:n]

    # 如果提供的质数数量不足 n，则重新计算，直到满足要求为止
    if len(primes) < n:
        big_number = 2000
        # 当不足 n 个质数时循环
        while 'Not enough primes':
            # 使用函数 primes_from_2_to 获取更多的质数，每次增加 1000
            primes = primes_from_2_to(big_number)[:n]  # type: ignore
            # 如果达到了 n 个质数，跳出循环
            if len(primes) == n:
                break
            # 增加检查的大数范围
            big_number += 1000

    # 返回生成的质数列表
    return primes


# 返回一个用于打乱 Van der Corput 序列的排列索引数组
def _van_der_corput_permutations(
    base: IntNumber, *, random_state: SeedType = None
) -> np.ndarray:
    # 生成 Van der Corput 序列的排列索引数组
    permutations : array_like
        Permutation indices.
    # 使用给定的随机数生成器（rng）初始化随机状态
    rng = check_random_state(random_state)
    # 计算需要生成的排列数量，基于浮点运算的条件：1 - base**-k < 1
    count = math.ceil(54 / math.log2(base)) - 1
    # 创建一个包含 count 个相同的初始排列的数组，每个排列都是长度为 base 的整数数组
    permutations = np.repeat(np.arange(base)[None], count, axis=0)
    # 对每个排列进行随机打乱
    for perm in permutations:
        rng.shuffle(perm)

    # 返回所有排列组成的数组
    return permutations
# 定义一个函数，生成 Van der Corput 序列
def van_der_corput(
        n: IntNumber,
        base: IntNumber = 2,
        *,
        start_index: IntNumber = 0,
        scramble: bool = False,
        permutations: npt.ArrayLike | None = None,
        seed: SeedType = None,
        workers: IntNumber = 1) -> np.ndarray:
    """Van der Corput sequence.

    Pseudo-random number generator based on a b-adic expansion.

    Scrambling uses permutations of the remainders (see [1]_). Multiple
    permutations are applied to construct a point. The sequence of
    permutations has to be the same for all points of the sequence.

    Parameters
    ----------
    n : int
        Number of elements in the sequence.
    base : int, optional
        Base of the sequence. Default is 2.
    start_index : int, optional
        Index to start the sequence from. Default is 0.
    scramble : bool, optional
        If True, use Owen scrambling. Otherwise no scrambling is done.
        Default is False.
    permutations : array_like, optional
        Permutations used for scrambling.
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is an int or None, a new `numpy.random.Generator` is
        created using ``np.random.default_rng(seed)``.
        If `seed` is already a ``Generator`` instance, then the provided
        instance is used.
    workers : int, optional
        Number of workers to use for parallel processing. If -1 is
        given all CPU threads are used. Default is 1.

    Returns
    -------
    sequence : np.ndarray (n,)
        Sequence of Van der Corput.

    References
    ----------
    .. [1] A. B. Owen. "A randomized Halton algorithm in R",
       :arxiv:`1706.02808`, 2017.

    """
    # 如果指定的 base 小于 2，抛出 ValueError 异常
    if base < 2:
        raise ValueError("'base' must be at least 2")

    # 如果需要进行 scrambling
    if scramble:
        # 如果未指定 permutations，则调用内部函数生成 permutations
        if permutations is None:
            permutations = _van_der_corput_permutations(
                base=base, random_state=seed
            )
        else:
            # 将 permutations 转换为 numpy 数组
            permutations = np.asarray(permutations)

        # 将 permutations 转换为 int64 类型
        permutations = permutations.astype(np.int64)
        # 调用 Cython 实现的带 scrambling 的 Van der Corput 序列生成函数
        return _cy_van_der_corput_scrambled(n, base, start_index,
                                            permutations, workers)

    else:
        # 调用 Cython 实现的普通 Van der Corput 序列生成函数
        return _cy_van_der_corput(n, base, start_index, workers)


# 定义一个抽象基类，用于构建特定的准蒙特卡洛采样器
class QMCEngine(ABC):
    """A generic Quasi-Monte Carlo sampler class meant for subclassing.

    QMCEngine is a base class to construct a specific Quasi-Monte Carlo
    sampler. It cannot be used directly as a sampler.

    Parameters
    ----------
    d : int
        Dimension of the parameter space.
    """
    # optimization 是一个可选参数，用于在采样后进行优化以提高质量。
    # 这是一个后处理步骤，并不保证样本的所有属性都会被保留。默认为 None。
    # 
    # * "random-cd": 使用随机坐标置换来降低中心偏差（centered discrepancy）。
    #   基于中心偏差的最佳样本会不断更新。相比使用其他偏差度量方法，
    #   基于中心偏差的采样对于 2D 和 3D 子投影显示出更好的空间填充鲁棒性。
    #
    # * "lloyd": 使用修改过的 Lloyd-Max 算法来扰动样本。
    #   该过程会收敛到等间距样本。
    #
    # .. versionadded:: 1.10.0

    # seed 是一个可选参数，用于设置随机数生成的种子。
    # 如果 seed 是一个整数或 None，则会创建一个新的 `numpy.random.Generator`，
    # 使用方法是 ``np.random.default_rng(seed)``。
    # 如果 seed 已经是一个 ``Generator`` 实例，则直接使用提供的实例。

    Notes
    -----
    # 按照惯例，样本分布在半开区间 ``[0, 1)`` 上。
    # 类的实例可以访问以下属性：``d`` 表示维度；``rng`` 表示用于种子的随机数生成器。

    **Subclassing**

    # 当子类化 `QMCEngine` 来创建一个新的采样器时，需要重新定义 ``__init__`` 和 ``random`` 方法。

    * ``__init__(d, seed=None)``: 至少需要指定维度。如果采样器不使用种子（像 Halton 这样的确定性方法），可以省略这个参数。

    * ``_random(n, *, workers=1)``: 从引擎中抽取 ``n`` 个样本。``workers`` 用于并行计算。查看 `Halton` 的示例。

    可选地，子类还可以重写另外两个方法：

    * ``reset``: 将引擎重置为其原始状态。
    * ``fast_forward``: 如果序列是确定性的（例如 Halton 序列），那么 ``fast_forward(n)`` 将跳过前 ``n`` 个抽样。

    Examples
    --------
    # 要基于 ``np.random.random`` 创建一个随机采样器，可以执行以下操作：

    >>> from scipy.stats import qmc
    >>> class RandomEngine(qmc.QMCEngine):
    ...     def __init__(self, d, seed=None):
    ...         super().__init__(d=d, seed=seed)
    ...
    ...
    ...     def _random(self, n=1, *, workers=1):
    ...         return self.rng.random((n, self.d))
    ...
    ...
    ...     def reset(self):
    ...         super().__init__(d=self.d, seed=self.rng_seed)
    ...         return self
    ...
    ...
    ...     def fast_forward(self, n):
    ...         self.random(n)
    ...         return self

    # 在子类化 `QMCEngine` 来定义我们想要使用的采样策略之后，我们可以创建一个实例来进行采样。

    >>> engine = RandomEngine(2)
    >>> engine.random(5)
    array([[0.22733602, 0.31675834],  # 随机数数组
           [0.79736546, 0.67625467],
           [0.39110955, 0.33281393],
           [0.59830875, 0.18673419],
           [0.67275604, 0.94180287]])

    We can also reset the state of the generator and resample again.

    >>> _ = engine.reset()
    >>> engine.random(5)
    array([[0.22733602, 0.31675834],  # 随机数数组（与之前相同）
           [0.79736546, 0.67625467],
           [0.39110955, 0.33281393],
           [0.59830875, 0.18673419],
           [0.67275604, 0.94180287]])

    """

    @abstractmethod
    def __init__(
        self,
        d: IntNumber,
        *,
        optimization: Literal["random-cd", "lloyd"] | None = None,
        seed: SeedType = None
    ) -> None:
        # 检查参数 `d` 是否为非负整数，若不是则引发 ValueError 异常
        if not np.issubdtype(type(d), np.integer) or d < 0:
            raise ValueError('d must be a non-negative integer value')

        self.d = d

        if isinstance(seed, np.random.Generator):
            # 如果 seed 是 numpy 随机数生成器，生成一个我们可以重置的 Generator 实例
            self.rng = _rng_spawn(seed, 1)[0]
        else:
            # 如果 seed 不是 numpy 随机数生成器，创建一个普通的 Generator 实例
            # 同时处理无法被生成的 RandomState 实例
            self.rng = check_random_state(seed)
        self.rng_seed = copy.deepcopy(self.rng)

        self.num_generated = 0

        config = {
            # random-cd 优化方法的配置参数
            "n_nochange": 100,
            "n_iters": 10_000,
            "rng": self.rng,

            # lloyd 优化方法的配置参数
            "tol": 1e-5,
            "maxiter": 10,
            "qhull_options": None,
        }
        # 根据指定的优化方法选择相应的优化器
        self.optimization_method = _select_optimizer(optimization, config)

    @abstractmethod
    def _random(
        self, n: IntNumber = 1, *, workers: IntNumber = 1
    ) -> np.ndarray:
        ...

    def random(
        self, n: IntNumber = 1, *, workers: IntNumber = 1
    ) -> np.ndarray:
        """从半开区间 `[0, 1)` 中抽取 `n` 个样本。

        Parameters
        ----------
        n : int, optional
            在参数空间中生成的样本数量，默认为 1。
        workers : int, optional
            仅适用于 `Halton`。
            用于并行处理的工作线程数。如果给定 -1，则使用所有 CPU 线程。默认为 1。
            对于大于 :math:`10^3` 的 `n` 值，比单个工作线程更快。

        Returns
        -------
        sample : array_like (n, d)
            QMC 样本。

        """
        # 调用内部方法 `_random` 生成随机样本
        sample = self._random(n, workers=workers)
        # 如果定义了优化方法，则对样本应用该优化方法
        if self.optimization_method is not None:
            sample = self.optimization_method(sample)

        self.num_generated += n
        return sample

    def integers(
        self,
        l_bounds: npt.ArrayLike,
        *,
        u_bounds: npt.ArrayLike | None = None,
        n: IntNumber = 1,
        endpoint: bool = False,
        workers: IntNumber = 1
    ) -> np.ndarray:
        r"""
        从 `l_bounds`（包含）到 `u_bounds`（不包含）之间或者如果 `endpoint=True`，
        从 `l_bounds`（包含）到 `u_bounds`（包含）之间抽取 `n` 个整数。

        Parameters
        ----------
        l_bounds : int or array-like of ints
            要抽取的最小整数（有符号），如果 `u_bounds=None`，则为0，并且此值用于 `u_bounds`。
        u_bounds : int or array-like of ints, optional
            如果提供，表示要抽取的最大整数的上限（不包含），如果 `u_bounds=None`，参见上文。
            如果是数组形式，必须包含整数值。
        n : int, optional
            在参数空间中生成的样本数量，默认为1。
        endpoint : bool, optional
            如果为真，则从区间 ``[l_bounds, u_bounds]`` 中抽样，而不是默认的 ``[l_bounds, u_bounds)``。默认为假。
        workers : int, optional
            用于并行处理的工作线程数。如果给定 `-1`，则使用所有CPU线程。仅在使用 `Halton` 时支持。默认为1。

        Returns
        -------
        sample : array_like (n, d)
            QMC（低差异序列）样本。

        Notes
        -----
        使用 QMC 时，可以安全地使用与 MC 相同的 ``[0, 1)`` 到整数的映射。仍然保持无偏性，
        大数定律，渐近无限方差减少和有限样本方差上界。

        要将从 :math:`[0, 1)` 区间的样本转换为 :math:`[a, b)`，其中 :math:`a` 是下限，:math:`b` 是上限，
        使用以下变换：

        .. math::

            \text{floor}((b - a) \cdot \text{sample} + a)

        """
        if u_bounds is None:
            u_bounds = l_bounds
            l_bounds = 0

        u_bounds = np.atleast_1d(u_bounds)
        l_bounds = np.atleast_1d(l_bounds)

        if endpoint:
            u_bounds = u_bounds + 1

        if (not np.issubdtype(l_bounds.dtype, np.integer) or
                not np.issubdtype(u_bounds.dtype, np.integer)):
            message = ("'u_bounds' 和 'l_bounds' 必须是整数或整数数组")
            raise ValueError(message)

        if isinstance(self, Halton):
            sample = self.random(n=n, workers=workers)
        else:
            sample = self.random(n=n)

        sample = scale(sample, l_bounds=l_bounds, u_bounds=u_bounds)
        sample = np.floor(sample).astype(np.int64)

        return sample
    def reset(self) -> QMCEngine:
        """Reset the engine to base state.

        Returns
        -------
        engine : QMCEngine
            Engine reset to its base state.

        """
        # 深拷贝随机数种子，以确保不改变原始种子值
        seed = copy.deepcopy(self.rng_seed)
        # 使用深拷贝后的种子重新初始化随机数生成器
        self.rng = check_random_state(seed)
        # 重置已生成数目为0
        self.num_generated = 0
        # 返回已重置状态的引擎对象
        return self

    def fast_forward(self, n: IntNumber) -> QMCEngine:
        """Fast-forward the sequence by `n` positions.

        Parameters
        ----------
        n : int
            Number of points to skip in the sequence.

        Returns
        -------
        engine : QMCEngine
            Engine reset to its base state.

        """
        # 调用随机数生成方法，跳过指定数量的点
        self.random(n=n)
        # 返回已快进状态的引擎对象
        return self
class Halton(QMCEngine):
    """Halton sequence.

    Pseudo-random number generator that generalize the Van der Corput sequence
    for multiple dimensions. The Halton sequence uses the base-two Van der
    Corput sequence for the first dimension, base-three for its second and
    base-:math:`n` for its n-dimension.

    Parameters
    ----------
    d : int
        Dimension of the parameter space.
    scramble : bool, optional
        If True, use Owen scrambling. Otherwise no scrambling is done.
        Default is True.
    optimization : {None, "random-cd", "lloyd"}, optional
        Whether to use an optimization scheme to improve the quality after
        sampling. Note that this is a post-processing step that does not
        guarantee that all properties of the sample will be conserved.
        Default is None.

        * ``random-cd``: random permutations of coordinates to lower the
          centered discrepancy. The best sample based on the centered
          discrepancy is constantly updated. Centered discrepancy-based
          sampling shows better space-filling robustness toward 2D and 3D
          subprojections compared to using other discrepancy measures.
        * ``lloyd``: Perturb samples using a modified Lloyd-Max algorithm.
          The process converges to equally spaced samples.

        .. versionadded:: 1.10.0
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is an int or None, a new `numpy.random.Generator` is
        created using ``np.random.default_rng(seed)``.
        If `seed` is already a ``Generator`` instance, then the provided
        instance is used.

    Notes
    -----
    The Halton sequence has severe striping artifacts for even modestly
    large dimensions. These can be ameliorated by scrambling. Scrambling
    also supports replication-based error estimates and extends
    applicabiltiy to unbounded integrands.

    References
    ----------
    .. [1] Halton, "On the efficiency of certain quasi-random sequences of
       points in evaluating multi-dimensional integrals", Numerische
       Mathematik, 1960.
    .. [2] A. B. Owen. "A randomized Halton algorithm in R",
       :arxiv:`1706.02808`, 2017.

    Examples
    --------
    Generate samples from a low discrepancy sequence of Halton.

    >>> from scipy.stats import qmc
    >>> sampler = qmc.Halton(d=2, scramble=False)
    >>> sample = sampler.random(n=5)
    >>> sample
    array([[0.        , 0.        ],
           [0.5       , 0.33333333],
           [0.25      , 0.66666667],
           [0.75      , 0.11111111],
           [0.125     , 0.44444444]])

    Compute the quality of the sample using the discrepancy criterion.

    >>> qmc.discrepancy(sample)
    0.088893711419753

    If some wants to continue an existing design, extra points can be obtained
    by calling again `random`. Alternatively, you can skip some points like:

    >>> _ = sampler.fast_forward(5)
    >>> sample_continued = sampler.random(n=5)
    """

    def __init__(self, d, scramble=True, optimization=None, seed=None):
        """
        Initialize Halton sequence generator.

        Parameters
        ----------
        d : int
            Dimension of the parameter space.
        scramble : bool, optional
            If True, use Owen scrambling. Otherwise no scrambling is done.
            Default is True.
        optimization : {None, "random-cd", "lloyd"}, optional
            Optimization scheme to improve sample quality.
        seed : {None, int, `numpy.random.Generator`}, optional
            Seed for random number generation.

        Notes
        -----
        Initializes a Halton sequence generator with specified parameters.
        """
        super().__init__(d=d, scramble=scramble, optimization=optimization, seed=seed)
    """
    Class representing a Quasi-Monte Carlo sampler for integration.

    Parameters:
    ----------
    d : int
        Dimensionality of the integration space.
    scramble : bool, optional
        Whether to scramble the sequences. Default is True.
    optimization : {"random-cd", "lloyd"} or None, optional
        Optimization method for the sampler. Default is None.
    seed : int or None, optional
        Seed for random number generation. Default is None.

    Methods:
    -------
    __init__(self, d: IntNumber, *, scramble: bool = True,
             optimization: Literal["random-cd", "lloyd"] | None = None,
             seed: SeedType = None) -> None:
        Initializes the QMC sampler with given parameters.

    _initialize_permutations(self) -> None:
        Initializes permutations for Van der Corput sequences if scrambling is enabled.

    _random(self, n: IntNumber = 1, *, workers: IntNumber = 1) -> np.ndarray:
        Generates QMC samples using Van der Corput sequences.

    Notes:
    ------
    This class uses Quasi-Monte Carlo methods for more evenly distributed sampling compared to
    traditional Monte Carlo methods.
    """
    
    def __init__(
        self, d: IntNumber, *, scramble: bool = True,
        optimization: Literal["random-cd", "lloyd"] | None = None,
        seed: SeedType = None
    ) -> None:
        # Used in `scipy.integrate.qmc_quad`
        self._init_quad = {'d': d, 'scramble': True,
                           'optimization': optimization}
        super().__init__(d=d, optimization=optimization, seed=seed)
        self.seed = seed

        # important to have ``type(bdim) == int`` for performance reason
        self.base = [int(bdim) for bdim in n_primes(d)]
        self.scramble = scramble

        self._initialize_permutations()

    def _initialize_permutations(self) -> None:
        """Initialize permutations for all Van der Corput sequences.

        Permutations are only needed for scrambling.
        """
        self._permutations: list = [None] * len(self.base)
        if self.scramble:
            for i, bdim in enumerate(self.base):
                permutations = _van_der_corput_permutations(
                    base=bdim, random_state=self.rng
                )

                self._permutations[i] = permutations

    def _random(
        self, n: IntNumber = 1, *, workers: IntNumber = 1
    ) -> np.ndarray:
        """Draw `n` in the half-open interval ``[0, 1)``.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.
        workers : int, optional
            Number of workers to use for parallel processing. If -1 is
            given all CPU threads are used. Default is 1. It becomes faster
            than one worker for `n` greater than :math:`10^3`.

        Returns
        -------
        sample : array_like (n, d)
            QMC sample.

        """
        workers = _validate_workers(workers)
        # Generate a sample using a Van der Corput sequence per dimension.
        sample = [van_der_corput(n, bdim, start_index=self.num_generated,
                                 scramble=self.scramble,
                                 permutations=self._permutations[i],
                                 workers=workers)
                  for i, bdim in enumerate(self.base)]

        return np.array(sample).T.reshape(n, self.d)
class LatinHypercube(QMCEngine):
    r"""Latin hypercube sampling (LHS).

    A Latin hypercube sample [1]_ generates :math:`n` points in
    :math:`[0,1)^{d}`. Each univariate marginal distribution is stratified,
    placing exactly one point in :math:`[j/n, (j+1)/n)` for
    :math:`j=0,1,...,n-1`. They are still applicable when :math:`n << d`.

    Parameters
    ----------
    d : int
        Dimension of the parameter space.
    scramble : bool, optional
        When False, center samples within cells of a multi-dimensional grid.
        Otherwise, samples are randomly placed within cells of the grid.

        .. note::
            Setting ``scramble=False`` does not ensure deterministic output.
            For that, use the `seed` parameter.

        Default is True.

        .. versionadded:: 1.10.0

    optimization : {None, "random-cd", "lloyd"}, optional
        Whether to use an optimization scheme to improve the quality after
        sampling. Note that this is a post-processing step that does not
        guarantee that all properties of the sample will be conserved.
        Default is None.

        * ``random-cd``: random permutations of coordinates to lower the
          centered discrepancy. The best sample based on the centered
          discrepancy is constantly updated. Centered discrepancy-based
          sampling shows better space-filling robustness toward 2D and 3D
          subprojections compared to using other discrepancy measures.
        * ``lloyd``: Perturb samples using a modified Lloyd-Max algorithm.
          The process converges to equally spaced samples.

        .. versionadded:: 1.8.0
        .. versionchanged:: 1.10.0
            Add ``lloyd``.

    strength : {1, 2}, optional
        Strength of the LHS. ``strength=1`` produces a plain LHS while
        ``strength=2`` produces an orthogonal array based LHS of strength 2
        [7]_, [8]_. In that case, only ``n=p**2`` points can be sampled,
        with ``p`` a prime number. It also constrains ``d <= p + 1``.
        Default is 1.

        .. versionadded:: 1.8.0

    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is an int or None, a new `numpy.random.Generator` is
        created using ``np.random.default_rng(seed)``.
        If `seed` is already a ``Generator`` instance, then the provided
        instance is used.

    Notes
    -----

    When LHS is used for integrating a function :math:`f` over :math:`n`,
    LHS is extremely effective on integrands that are nearly additive [2]_.
    With a LHS of :math:`n` points, the variance of the integral is always
    lower than plain MC on :math:`n-1` points [3]_. There is a central limit
    theorem for LHS on the mean and variance of the integral [4]_, but not
    necessarily for optimized LHS due to the randomization.

    :math:`A` is called an orthogonal array of strength :math:`t` if in each
    n-row-by-t-column submatrix of :math:`A`: all :math:`p^t` possible
    """

    def __init__(self, d, scramble=True, optimization=None, strength=1, seed=None):
        """
        Initialize LatinHypercube sampling parameters.

        Parameters
        ----------
        d : int
            Dimension of the parameter space.
        scramble : bool, optional
            Whether to scramble samples within cells or not.
        optimization : {None, "random-cd", "lloyd"}, optional
            Type of post-sampling optimization.
        strength : {1, 2}, optional
            Strength of the LHS sampling.
        seed : {None, int, `numpy.random.Generator`}, optional
            Seed for random number generation.
        """
        super().__init__()
        self.d = d
        self.scramble = scramble
        self.optimization = optimization
        self.strength = strength
        self.seed = seed
    distinct rows occur the same number of times. The elements of :math:`A`
    are in the set :math:`\{0, 1, ..., p-1\}`, also called symbols.
    The constraint that :math:`p` must be a prime number is to allow modular
    arithmetic. Increasing strength adds some symmetry to the sub-projections
    of a sample. With strength 2, samples are symmetric along the diagonals of
    2D sub-projections. This may be undesirable, but on the other hand, the
    sample dispersion is improved.

    Strength 1 (plain LHS) brings an advantage over strength 0 (MC) and
    strength 2 is a useful increment over strength 1. Going to strength 3 is
    a smaller increment and scrambled QMC like Sobol', Halton are more
    performant [7]_.

    To create a LHS of strength 2, the orthogonal array :math:`A` is
    randomized by applying a random, bijective map of the set of symbols onto
    itself. For example, in column 0, all 0s might become 2; in column 1,
    all 0s might become 1, etc.
    Then, for each column :math:`i` and symbol :math:`j`, we add a plain,
    one-dimensional LHS of size :math:`p` to the subarray where
    :math:`A^i = j`. The resulting matrix is finally divided by :math:`p`.

    References
    ----------
    .. [1] Mckay et al., "A Comparison of Three Methods for Selecting Values
       of Input Variables in the Analysis of Output from a Computer Code."
       Technometrics, 1979.
    .. [2] M. Stein, "Large sample properties of simulations using Latin
       hypercube sampling." Technometrics 29, no. 2: 143-151, 1987.
    .. [3] A. B. Owen, "Monte Carlo variance of scrambled net quadrature."
       SIAM Journal on Numerical Analysis 34, no. 5: 1884-1910, 1997
    .. [4]  Loh, W.-L. "On Latin hypercube sampling." The annals of statistics
       24, no. 5: 2058-2080, 1996.
    .. [5] Fang et al. "Design and modeling for computer experiments".
       Computer Science and Data Analysis Series, 2006.
    .. [6] Damblin et al., "Numerical studies of space filling designs:
       optimization of Latin Hypercube Samples and subprojection properties."
       Journal of Simulation, 2013.
    .. [7] A. B. Owen , "Orthogonal arrays for computer experiments,
       integration and visualization." Statistica Sinica, 1992.
    .. [8] B. Tang, "Orthogonal Array-Based Latin Hypercubes."
       Journal of the American Statistical Association, 1993.
    .. [9] Susan K. Seaholm et al. "Latin hypercube sampling and the
       sensitivity analysis of a Monte Carlo epidemic model".
       Int J Biomed Comput, 23(1-2), 97-112,
       :doi:`10.1016/0020-7101(88)90067-0`, 1988.

    Examples
    --------
    In [9]_, a Latin Hypercube sampling strategy was used to sample a
    parameter space to study the importance of each parameter of an epidemic
    model. Such analysis is also called a sensitivity analysis.

    Since the dimensionality of the problem is high (6), it is computationally
    expensive to cover the space. When numerical experiments are costly,
    QMC enables analysis that may not be possible if using a grid.
    量子蒙特卡罗（QMC）使得分析可能性超出了使用网格的范围。

    The six parameters of the model represented the probability of illness,
    the probability of withdrawal, and four contact probabilities,
    The authors assumed uniform distributions for all parameters and generated
    50 samples.
    模型的六个参数代表了生病的概率、退出的概率以及四个接触概率。
    作者假设所有参数均服从均匀分布，并生成了50个样本。

    Using `scipy.stats.qmc.LatinHypercube` to replicate the protocol, the
    first step is to create a sample in the unit hypercube:
    使用 `scipy.stats.qmc.LatinHypercube` 来复制该协议，首先在单位超立方体中创建样本：

    >>> from scipy.stats import qmc
    >>> sampler = qmc.LatinHypercube(d=6)
    >>> sample = sampler.random(n=50)

    Then the sample can be scaled to the appropriate bounds:
    然后可以将样本缩放到适当的边界：

    >>> l_bounds = [0.000125, 0.01, 0.0025, 0.05, 0.47, 0.7]
    >>> u_bounds = [0.000375, 0.03, 0.0075, 0.15, 0.87, 0.9]
    >>> sample_scaled = qmc.scale(sample, l_bounds, u_bounds)

    Such a sample was used to run the model 50 times, and a polynomial
    response surface was constructed. This allowed the authors to study the
    relative importance of each parameter across the range of
    possibilities of every other parameter.
    这样的样本被用来运行模型50次，并构建了一个多项式响应曲面。这使得作者能够研究每个参数在其他所有参数可能性范围内的相对重要性。

    In this computer experiment, they showed a 14-fold reduction in the number
    of samples required to maintain an error below 2% on their response surface
    when compared to a grid sampling.
    在这个计算实验中，他们展示了相比于网格采样，为了在响应曲面上保持低于2%的误差所需样本数量减少了14倍。

    Below are other examples showing alternative ways to construct LHS
    with even better coverage of the space.
    下面是其他示例，展示了构建更好覆盖空间的LHS的替代方法。

    Using a base LHS as a baseline.
    使用基础LHS作为基准。

    >>> sampler = qmc.LatinHypercube(d=2)
    >>> sample = sampler.random(n=5)
    >>> qmc.discrepancy(sample)
    0.0196...  # random

    Use the `optimization` keyword argument to produce a LHS with
    lower discrepancy at higher computational cost.
    使用 `optimization` 关键字参数来生成具有更低差异但计算成本更高的LHS。

    >>> sampler = qmc.LatinHypercube(d=2, optimization="random-cd")
    >>> sample = sampler.random(n=5)
    >>> qmc.discrepancy(sample)
    0.0176...  # random

    Use the `strength` keyword argument to produce an orthogonal array based
    LHS of strength 2. In this case, the number of sample points must be the
    square of a prime number.
    使用 `strength` 关键字参数生成基于正交数组的强度为2的LHS。在这种情况下，样本点的数量必须是一个质数的平方。

    >>> sampler = qmc.LatinHypercube(d=2, strength=2)
    >>> sample = sampler.random(n=9)
    >>> qmc.discrepancy(sample)
    0.00526...  # random

    Options could be combined to produce an optimized centered
    orthogonal array based LHS. After optimization, the result would not
    be guaranteed to be of strength 2.
    可以组合选项生成基于正交数组的优化中心化的LHS。优化后，结果不一定保证是强度为2的。

    """

    def __init__(
        self, d: IntNumber, *,
        scramble: bool = True,
        strength: int = 1,
        optimization: Literal["random-cd", "lloyd"] | None = None,
        seed: SeedType = None
    ):
    ) -> None:
        # 用于 `scipy.integrate.qmc_quad` 中
        self._init_quad = {'d': d, 'scramble': True, 'strength': strength,
                           'optimization': optimization}
        # 调用父类初始化方法，传递参数 d, seed, optimization
        super().__init__(d=d, seed=seed, optimization=optimization)
        # 设置对象属性 scramble
        self.scramble = scramble

        # 定义不同 strength 对应的 LHS 方法字典
        lhs_method_strength = {
            1: self._random_lhs,
            2: self._random_oa_lhs
        }

        try:
            # 根据 strength 选择相应的 LHS 方法
            self.lhs_method: Callable = lhs_method_strength[strength]
        except KeyError as exc:
            # 如果 strength 不在预期范围内，抛出 ValueError 异常
            message = (f"{strength!r} is not a valid strength. It must be one"
                       f" of {set(lhs_method_strength)!r}")
            raise ValueError(message) from exc

    def _random(
        self, n: IntNumber = 1, *, workers: IntNumber = 1
    ) -> np.ndarray:
        # 调用当前对象的 lhs_method 方法生成随机样本
        lhs = self.lhs_method(n)
        return lhs

    def _random_lhs(self, n: IntNumber = 1) -> np.ndarray:
        """Base LHS algorithm."""
        # 如果不需要 scramble，返回固定的 0.5
        if not self.scramble:
            samples: np.ndarray | float = 0.5
        else:
            # 使用 RNG 生成 n 行 self.d 列的均匀分布样本
            samples = self.rng.uniform(size=(n, self.d))

        # 生成一个从 1 到 n 的整数序列，复制 self.d 次，形成二维数组
        perms = np.tile(np.arange(1, n + 1),
                        (self.d, 1))  # type: ignore[arg-type]

        # 对 perms 每行进行随机打乱
        for i in range(self.d):
            self.rng.shuffle(perms[i, :])
        perms = perms.T

        # 计算最终的 LHS 样本
        samples = (perms - samples) / n
        return samples
    def _random_oa_lhs(self, n: IntNumber = 4) -> np.ndarray:
        """Orthogonal array based LHS of strength 2."""
        # 计算行数和列数
        p = np.sqrt(n).astype(int)
        n_row = p**2
        n_col = p + 1

        # 计算小于等于 p+1 的所有质数
        primes = primes_from_2_to(p + 1)
        # 如果 p 不是质数或者 n 不等于 p 的平方，则引发 ValueError
        if p not in primes or n != n_row:
            raise ValueError(
                "n is not the square of a prime number. Close"
                f" values are {primes[-2:]**2}"
            )
        # 如果 d 大于 p+1，则引发 ValueError
        if self.d > p + 1:
            raise ValueError("n is too small for d. Must be n > (d-1)**2")

        # 创建一个全零的二维数组作为 OA 样本
        oa_sample = np.zeros(shape=(n_row, n_col), dtype=int)

        # 构建强度为 2 的 OA
        arrays = np.tile(np.arange(p), (2, 1))
        oa_sample[:, :2] = np.stack(np.meshgrid(*arrays),
                                    axis=-1).reshape(-1, 2)
        for p_ in range(1, p):
            oa_sample[:, 2+p_-1] = np.mod(oa_sample[:, 0]
                                          + p_*oa_sample[:, 1], p)

        # 打乱 OA
        oa_sample_ = np.empty(shape=(n_row, n_col), dtype=int)
        for j in range(n_col):
            perms = self.rng.permutation(p)
            oa_sample_[:, j] = perms[oa_sample[:, j]]
        
        oa_sample = oa_sample_

        # 将打乱后的 OA 转化为 OA-LHS
        oa_lhs_sample = np.zeros(shape=(n_row, n_col))
        # 创建拉丁超立方引擎用于生成 LHS
        lhs_engine = LatinHypercube(d=1, scramble=self.scramble, strength=1,
                                    seed=self.rng)  # type: QMCEngine
        for j in range(n_col):
            for k in range(p):
                idx = oa_sample[:, j] == k
                # 生成大小为 p 的随机 LHS
                lhs = lhs_engine.random(p).flatten()
                oa_lhs_sample[:, j][idx] = lhs + oa_sample[:, j][idx]

        # 对结果进行归一化
        oa_lhs_sample /= p

        return oa_lhs_sample[:, :self.d]
# 定义一个名为 Sobol 的类，继承自 QMCEngine 类，用于生成（混乱的）Sobol序列。
class Sobol(QMCEngine):
    """Engine for generating (scrambled) Sobol' sequences.

    Sobol' sequences are low-discrepancy, quasi-random numbers. Points
    can be drawn using two methods:

    * `random_base2`: safely draw :math:`n=2^m` points. This method
      guarantees the balance properties of the sequence.
    * `random`: draw an arbitrary number of points from the
      sequence. See warning below.

    Parameters
    ----------
    d : int
        Dimensionality of the sequence. Max dimensionality is 21201.
    scramble : bool, optional
        If True, use LMS+shift scrambling. Otherwise, no scrambling is done.
        Default is True.
    bits : int, optional
        Number of bits of the generator. Control the maximum number of points
        that can be generated, which is ``2**bits``. Maximal value is 64.
        It does not correspond to the return type, which is always
        ``np.float64`` to prevent points from repeating themselves.
        Default is None, which for backward compatibility, corresponds to 30.

        .. versionadded:: 1.9.0
    optimization : {None, "random-cd", "lloyd"}, optional
        Whether to use an optimization scheme to improve the quality after
        sampling. Note that this is a post-processing step that does not
        guarantee that all properties of the sample will be conserved.
        Default is None.

        * ``random-cd``: random permutations of coordinates to lower the
          centered discrepancy. The best sample based on the centered
          discrepancy is constantly updated. Centered discrepancy-based
          sampling shows better space-filling robustness toward 2D and 3D
          subprojections compared to using other discrepancy measures.
        * ``lloyd``: Perturb samples using a modified Lloyd-Max algorithm.
          The process converges to equally spaced samples.

        .. versionadded:: 1.10.0
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is an int or None, a new `numpy.random.Generator` is
        created using ``np.random.default_rng(seed)``.
        If `seed` is already a ``Generator`` instance, then the provided
        instance is used.

    Notes
    -----
    Sobol' sequences [1]_ provide :math:`n=2^m` low discrepancy points in
    :math:`[0,1)^{d}`. Scrambling them [3]_ makes them suitable for singular
    integrands, provides a means of error estimation, and can improve their
    rate of convergence. The scrambling strategy which is implemented is a
    (left) linear matrix scramble (LMS) followed by a digital random shift
    (LMS+shift) [2]_.

    There are many versions of Sobol' sequences depending on their
    'direction numbers'. This code uses direction numbers from [4]_. Hence,
    the maximum number of dimension is 21201. The direction numbers have been
    precomputed with search criterion 6 and can be retrieved at
    https://web.maths.unsw.edu.au/~fkuo/sobol/.
    """
    # 定义类属性 MAXDIM，表示 Sobol' 序列的最大维度，从外部导入的 _MAXDIM 值
    MAXDIM: ClassVar[int] = _MAXDIM
    
    # 初始化方法，用于创建 Sobol' 序列生成器的实例
    def __init__(
        self, d: IntNumber, *, scramble: bool = True,
        bits: IntNumber | None = None, seed: SeedType = None,
        optimization: Literal["random-cd", "lloyd"] | None = None
    ):
    ) -> None:
        # 在 `scipy.integrate.qmc_quad` 中使用的初始化参数
        self._init_quad = {'d': d, 'scramble': True, 'bits': bits,
                           'optimization': optimization}

        # 调用父类的初始化方法，传入维度和优化参数
        super().__init__(d=d, optimization=optimization, seed=seed)
        # 如果维度超过了最大支持维度，则引发数值错误
        if d > self.MAXDIM:
            raise ValueError(
                f"Maximum supported dimensionality is {self.MAXDIM}."
            )

        # 设置属性 `bits`，如果未提供，则默认为 30
        self.bits = bits
        # 设置数据类型 `dtype_i`，根据 `bits` 的大小选择 `np.uint32` 或 `np.uint64`
        self.dtype_i: type

        if self.bits is None:
            self.bits = 30

        if self.bits <= 32:
            self.dtype_i = np.uint32
        elif 32 < self.bits <= 64:
            self.dtype_i = np.uint64
        else:
            raise ValueError("Maximum supported 'bits' is 64")

        # 计算最大数值 `maxn`，即 2 的 `bits` 次方
        self.maxn = 2**self.bits

        # 初始化 `self._sv` 为大小为 `d x bits` 的零矩阵，数据类型为 `dtype_i`
        self._sv: np.ndarray = np.zeros((d, self.bits), dtype=self.dtype_i)
        # 调用 `_initialize_v` 函数，初始化 `_sv` 矩阵
        _initialize_v(self._sv, dim=d, bits=self.bits)

        # 如果不进行混淆 (`scramble` 为 False)，则初始化 `_shift` 为零向量
        if not scramble:
            self._shift: np.ndarray = np.zeros(d, dtype=self.dtype_i)
        else:
            # 如果进行混淆，则调用 `_scramble` 方法进行混淆操作
            self._scramble()

        # 初始化 `_quasi` 为混淆后的 `_shift` 的副本
        self._quasi = self._shift.copy()

        # 计算归一化常数 `_scale`，防止在 Python 中计算时溢出 int 型的范围，值为 1 / 2 的 `bits` 次方
        self._scale = 1.0 / 2 ** self.bits

        # 计算第一个点 `_first_point`，将 `_quasi` 乘以 `_scale` 后变形为行向量，并显式转换为 `np.float64` 类型
        self._first_point = (self._quasi * self._scale).reshape(1, -1)
        self._first_point = self._first_point.astype(np.float64)

    def _scramble(self) -> None:
        """使用 LMS+shift 进行序列混淆。"""
        # 生成混淆向量 `_shift`
        self._shift = np.dot(
            rng_integers(self.rng, 2, size=(self.d, self.bits),
                         dtype=self.dtype_i),
            2 ** np.arange(self.bits, dtype=self.dtype_i),
        )
        # 生成下三角矩阵 `_ltm`（在维度间堆叠），用于进一步混淆
        ltm = np.tril(rng_integers(self.rng, 2,
                                   size=(self.d, self.bits, self.bits),
                                   dtype=self.dtype_i))
        # 调用 `_cscramble` 函数进行混淆操作，传入维度、位数、下三角矩阵和 `_sv`
        _cscramble(
            dim=self.d, bits=self.bits,  # type: ignore[arg-type]
            ltm=ltm, sv=self._sv
        )

    def _random(
        self, n: IntNumber = 1, *, workers: IntNumber = 1
    ) -> np.ndarray:
        """Draw next point(s) in the Sobol' sequence.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.

        Returns
        -------
        sample : array_like (n, d)
            Sobol' sample.

        """
        # 初始化一个空的 numpy 数组用于存储 Sobol' 序列的样本点
        sample: np.ndarray = np.empty((n, self.d), dtype=np.float64)

        if n == 0:
            return sample

        # 计算生成样本点后的总数目
        total_n = self.num_generated + n
        # 检查总数是否超过预设的最大值
        if total_n > self.maxn:
            msg = (
                f"At most 2**{self.bits}={self.maxn} distinct points can be "
                f"generated. {self.num_generated} points have been previously "
                f"generated, then: n={self.num_generated}+{n}={total_n}. "
            )
            # 如果位数不是 64 位，建议增加位数以支持更多样本点的生成
            if self.bits != 64:
                msg += "Consider increasing `bits`."
            # 抛出值错误并附上详细信息
            raise ValueError(msg)

        if self.num_generated == 0:
            # 如果当前没有生成过样本点
            # 验证 n 是否为 2 的幂次方，Sobol' 点的均衡性要求 n 必须是 2 的幂次方
            if not (n & (n - 1) == 0):
                warnings.warn("The balance properties of Sobol' points require"
                              " n to be a power of 2.", stacklevel=2)

            if n == 1:
                # 如果只生成一个样本点，直接使用预定义的第一个点
                sample = self._first_point
            else:
                # 生成 n-1 个样本点，并将结果与预定义的第一个点拼接在一起
                _draw(
                    n=n - 1, num_gen=self.num_generated, dim=self.d,
                    scale=self._scale, sv=self._sv, quasi=self._quasi,
                    sample=sample
                )
                sample = np.concatenate(
                    [self._first_point, sample]
                )[:n]
        else:
            # 如果已经生成过样本点，直接生成 n 个样本点
            _draw(
                n=n, num_gen=self.num_generated - 1, dim=self.d,
                scale=self._scale, sv=self._sv, quasi=self._quasi,
                sample=sample
            )

        # 返回生成的 Sobol' 样本点数组
        return sample

    def random_base2(self, m: IntNumber) -> np.ndarray:
        """Draw point(s) from the Sobol' sequence.

        This function draws :math:`n=2^m` points in the parameter space
        ensuring the balance properties of the sequence.

        Parameters
        ----------
        m : int
            Logarithm in base 2 of the number of samples; i.e., n = 2^m.

        Returns
        -------
        sample : array_like (n, d)
            Sobol' sample.

        """
        # 计算样本点的数量，必须是 2 的幂次方
        n = 2 ** m

        # 计算生成样本点后的总数目
        total_n = self.num_generated + n
        # 检查总数是否为 2 的幂次方，因为 Sobol' 点的均衡性要求 n 必须是 2 的幂次方
        if not (total_n & (total_n - 1) == 0):
            raise ValueError('The balance properties of Sobol\' points require '
                             f'n to be a power of 2. {self.num_generated} points '
                             'have been previously generated, then: '
                             f'n={self.num_generated}+2**{m}={total_n}. '
                             'If you still want to do this, the function '
                             '\'Sobol.random()\' can be used.'
                             )

        # 调用 random 方法生成 Sobol' 样本点数组
        return self.random(n)
    def reset(self) -> Sobol:
        """Reset the engine to base state.
        
        Returns
        -------
        engine : Sobol
            Engine reset to its base state.
        """
        # 调用父类的 reset 方法，重置引擎到基本状态
        super().reset()
        # 复制当前的位移数组作为新的准随机数数组
        self._quasi = self._shift.copy()
        # 返回当前对象实例，用于方法链式调用
        return self

    def fast_forward(self, n: IntNumber) -> Sobol:
        """Fast-forward the sequence by `n` positions.
        
        Parameters
        ----------
        n : int
            Number of points to skip in the sequence.
        
        Returns
        -------
        engine : Sobol
            The fast-forwarded engine.
        """
        # 如果当前生成的点数为零，则按照特定的方式快进序列
        if self.num_generated == 0:
            _fast_forward(
                n=n - 1, num_gen=self.num_generated, dim=self.d,
                sv=self._sv, quasi=self._quasi
            )
        else:
            # 否则，按照另一种方式快进序列
            _fast_forward(
                n=n, num_gen=self.num_generated - 1, dim=self.d,
                sv=self._sv, quasi=self._quasi
            )
        # 更新生成的总点数
        self.num_generated += n
        # 返回当前对象实例，用于方法链式调用
        return self
class PoissonDisk(QMCEngine):
    """Poisson disk sampling.

    Parameters
    ----------
    d : int
        Dimension of the parameter space.
    radius : float
        Minimal distance to keep between points when sampling new candidates.
    hypersphere : {"volume", "surface"}, optional
        Sampling strategy to generate potential candidates to be added in the
        final sample. Default is "volume".

        * ``volume``: original Bridson algorithm as described in [1]_.
          New candidates are sampled *within* the hypersphere.
        * ``surface``: only sample the surface of the hypersphere.
    ncandidates : int
        Number of candidates to sample per iteration. More candidates result
        in a denser sampling as more candidates can be accepted per iteration.
    optimization : {None, "random-cd", "lloyd"}, optional
        Whether to use an optimization scheme to improve the quality after
        sampling. Note that this is a post-processing step that does not
        guarantee that all properties of the sample will be conserved.
        Default is None.

        * ``random-cd``: random permutations of coordinates to lower the
          centered discrepancy. The best sample based on the centered
          discrepancy is constantly updated. Centered discrepancy-based
          sampling shows better space-filling robustness toward 2D and 3D
          subprojections compared to using other discrepancy measures.
        * ``lloyd``: Perturb samples using a modified Lloyd-Max algorithm.
          The process converges to equally spaced samples.

        .. versionadded:: 1.10.0
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is an int or None, a new `numpy.random.Generator` is
        created using ``np.random.default_rng(seed)``.
        If `seed` is already a ``Generator`` instance, then the provided
        instance is used.

    l_bounds, u_bounds : array_like (d,)
        Lower and upper bounds of target sample data.

    Notes
    -----
    Poisson disk sampling is an iterative sampling strategy. Starting from
    a seed sample, `ncandidates` are sampled in the hypersphere
    surrounding the seed. Candidates below a certain `radius` or outside the
    domain are rejected. New samples are added in a pool of sample seed. The
    process stops when the pool is empty or when the number of required
    samples is reached.

    The maximum number of point that a sample can contain is directly linked
    to the `radius`. As the dimension of the space increases, a higher radius
    spreads the points further and help overcome the curse of dimensionality.
    See the :ref:`quasi monte carlo tutorial <quasi-monte-carlo>` for more
    details.
    """

    # PoissonDisk 类，继承自 QMCEngine 类，用于实现泊松盘采样算法

    def __init__(self, d, radius, hypersphere="volume", ncandidates=30,
                 optimization=None, seed=None, l_bounds=None, u_bounds=None):
        # 初始化方法，设置泊松盘采样的各种参数和选项

        super().__init__()

        # 调用父类 QMCEngine 的初始化方法

        self.d = d
        # 参数空间的维度

        self.radius = radius
        # 采样新候选点时保持的最小距离

        self.hypersphere = hypersphere
        # 生成潜在候选点以添加到最终样本中的采样策略，默认为 "volume"

        self.ncandidates = ncandidates
        # 每次迭代中要采样的候选点数量，候选点数量越多，每次迭代接受的候选点越多，导致更密集的采样

        self.optimization = optimization
        # 是否使用优化方案来提高采样质量的选项，默认为 None

        self.seed = seed
        # 随机数种子，用于确定随机数生成的起始状态

        self.l_bounds = l_bounds
        self.u_bounds = u_bounds
        # 目标样本数据的下界和上界

        if self.seed is not None:
            self.seed_rng(seed)

        # 如果指定了种子，则使用 seed_rng 方法设置随机数种子
    # 定义一个名为 PoissonDisk 的类，用于生成 Poisson 点阵的采样
    def __init__(
        self,
        d: IntNumber,  # 参数d：表示空间的维度
        *,
        radius: DecimalNumber = 0.05,  # 参数radius：Poisson 点的最小间距，默认为0.05
        hypersphere: Literal["volume", "surface"] = "volume",  # 参数hypersphere：表示超球体的体积或表面
        ncandidates: IntNumber = 30,  # 参数ncandidates：候选点的数量，默认为30
        optimization: Literal["random-cd", "lloyd"] | None = None,  # 参数optimization：优化方法，可以是'random-cd'或'lloyd'，或者None
        seed: SeedType = None,  # 参数seed：随机数种子，用于确定随机初始化
        l_bounds: npt.ArrayLike | None = None,  # 参数l_bounds：每个维度的下界，可以是数组或None
        u_bounds: npt.ArrayLike | None = None  # 参数u_bounds：每个维度的上界，可以是数组或None
    ):
    ) -> None:
        # 在 `scipy.integrate.qmc_quad` 中使用的初始化方法
        self._init_quad = {'d': d, 'radius': radius,
                           'hypersphere': hypersphere,
                           'ncandidates': ncandidates,
                           'optimization': optimization}
        # 调用父类的初始化方法，传递必要的参数
        super().__init__(d=d, optimization=optimization, seed=seed)

        # 根据不同的超球体采样方法选择相应的采样函数
        hypersphere_sample = {
            "volume": self._hypersphere_volume_sample,
            "surface": self._hypersphere_surface_sample
        }

        try:
            self.hypersphere_method = hypersphere_sample[hypersphere]
        except KeyError as exc:
            # 如果选择的超球体采样方法不在预定义的选项中，抛出错误
            message = (
                f"{hypersphere!r} is not a valid hypersphere sampling"
                f" method. It must be one of {set(hypersphere_sample)!r}")
            raise ValueError(message) from exc

        # 根据超球体的采样方法设定半径因子
        # 对于体积采样，半径因子设为2；对于表面采样，半径因子设为1.001
        self.radius_factor = 2 if hypersphere == "volume" else 1.001
        self.radius = radius
        self.radius_squared = self.radius**2

        # 每次迭代中在超球体中生成的样本数
        self.ncandidates = ncandidates
        
        # 如果上界未指定，默认为维度d的单位向量
        if u_bounds is None:
            u_bounds = np.ones(d)
        # 如果下界未指定，默认为维度d的零向量
        if l_bounds is None:
            l_bounds = np.zeros(d)
        # 验证上下界，确保它们是合法的
        self.l_bounds, self.u_bounds = _validate_bounds(
            l_bounds=l_bounds, u_bounds=u_bounds, d=int(d)
        )

        # 忽略除法时的警告，计算网格单元的尺寸和网格大小
        with np.errstate(divide='ignore'):
            self.cell_size = self.radius / np.sqrt(self.d)
            self.grid_size = (
                np.ceil((self.u_bounds - self.l_bounds) / self.cell_size)
            ).astype(int)

        # 初始化网格池
        self._initialize_grid_pool()

    def _initialize_grid_pool(self):
        """初始化采样池和采样网格。"""
        self.sample_pool = []
        # 每个网格单元的位置
        # 每个网格单元的n维值
        self.sample_grid = np.empty(
            np.append(self.grid_size, self.d),
            dtype=np.float32
        )
        # 用NaN填充空单元格
        self.sample_grid.fill(np.nan)

    def _random(
        self, n: IntNumber = 1, *, workers: IntNumber = 1
    def fill_space(self) -> np.ndarray:
        """Draw ``n`` samples in the interval ``[l_bounds, u_bounds]``.

        Unlike `random`, this method will try to add points until
        the space is full. Depending on ``candidates`` (and to a lesser extent
        other parameters), some empty areas can still be present in the sample.

        .. warning::

           This can be extremely slow in high dimensions or if the
           ``radius`` is very small-with respect to the dimensionality.

        Returns
        -------
        sample : array_like (n, d)
            QMC sample.

        """
        # 调用父类的 random 方法，使用无穷大的半径进行随机采样
        return self.random(np.inf)  # type: ignore[arg-type]

    def reset(self) -> PoissonDisk:
        """Reset the engine to base state.

        Returns
        -------
        engine : PoissonDisk
            Engine reset to its base state.

        """
        # 调用父类的 reset 方法，初始化网格池
        super().reset()
        self._initialize_grid_pool()
        # 返回重置后的引擎实例
        return self

    def _hypersphere_volume_sample(
        self, center: np.ndarray, radius: DecimalNumber,
        candidates: IntNumber = 1
    ) -> np.ndarray:
        """Uniform sampling within hypersphere."""
        # 从标准正态分布中生成随机数矩阵 x
        x = self.rng.standard_normal(size=(candidates, self.d))
        # 计算每行向量的平方和
        ssq = np.sum(x**2, axis=1)
        # 计算分布因子 fr，并进行标准化
        fr = radius * gammainc(self.d/2, ssq/2)**(1/self.d) / np.sqrt(ssq)
        # 将 fr 扩展成与 x 相同形状的矩阵
        fr_tiled = np.tile(
            fr.reshape(-1, 1), (1, self.d)  # type: ignore[arg-type]
        )
        # 计算在超球体内均匀抽样的点 p
        p = center + np.multiply(x, fr_tiled)
        return p

    def _hypersphere_surface_sample(
        self, center: np.ndarray, radius: DecimalNumber,
        candidates: IntNumber = 1
    ) -> np.ndarray:
        """Uniform sampling on the hypersphere's surface."""
        # 从标准正态分布中生成随机数矩阵 vec
        vec = self.rng.standard_normal(size=(candidates, self.d))
        # 对 vec 进行归一化处理，使其位于单位球面上
        vec /= np.linalg.norm(vec, axis=1)[:, None]
        # 计算在超球面上均匀抽样的点 p
        p = center + np.multiply(vec, radius)
        return p
class MultivariateNormalQMC:
    r"""QMC sampling from a multivariate Normal :math:`N(\mu, \Sigma)`.

    Parameters
    ----------
    mean : array_like (d,)
        The mean vector. Where ``d`` is the dimension.
    cov : array_like (d, d), optional
        The covariance matrix. If omitted, use `cov_root` instead.
        If both `cov` and `cov_root` are omitted, use the identity matrix.
    cov_root : array_like (d, d'), optional
        A root decomposition of the covariance matrix, where ``d'`` may be less
        than ``d`` if the covariance is not full rank. If omitted, use `cov`.
    inv_transform : bool, optional
        If True, use inverse transform instead of Box-Muller. Default is True.
    engine : QMCEngine, optional
        Quasi-Monte Carlo engine sampler. If None, `Sobol` is used.
    seed : {None, int, `numpy.random.Generator`}, optional
        Used only if `engine` is None.
        If `seed` is an int or None, a new `numpy.random.Generator` is
        created using ``np.random.default_rng(seed)``.
        If `seed` is already a ``Generator`` instance, then the provided
        instance is used.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import qmc
    >>> dist = qmc.MultivariateNormalQMC(mean=[0, 5], cov=[[1, 0], [0, 1]])
    >>> sample = dist.random(512)
    >>> _ = plt.scatter(sample[:, 0], sample[:, 1])
    >>> plt.show()

    """

    def __init__(
            self, mean: npt.ArrayLike, cov: npt.ArrayLike | None = None, *,
            cov_root: npt.ArrayLike | None = None,
            inv_transform: bool = True,
            engine: QMCEngine | None = None,
            seed: SeedType = None
    ):
        # 初始化函数，用于创建 MultivariateNormalQMC 对象
        # 参数:
        # - mean: 均值向量，长度为 d
        # - cov: 协方差矩阵，可选。如果省略，则使用 cov_root；如果 cov_root 也省略，则使用单位矩阵
        # - cov_root: 协方差矩阵的根分解，可选。如果省略，则使用 cov
        # - inv_transform: 布尔值，可选。如果为 True，则使用逆变换而不是 Box-Muller 方法。默认为 True
        # - engine: QMCEngine 类型，可选。准蒙特卡洛引擎采样器。如果为 None，则使用 Sobol 序列
        # - seed: {None, int, `numpy.random.Generator`}，可选。仅当 engine 为 None 时使用。
        #   如果 seed 是整数或 None，则使用 ``np.random.default_rng(seed)`` 创建新的 `numpy.random.Generator` 实例。
        #   如果 seed 已经是 `Generator` 实例，则使用提供的实例。

        # 下面是初始化方法的实现，用于设置对象的初始状态和属性
        pass
    `
        ) -> None:
            # 将 mean 转换为 NumPy 数组，并确保其为一维数组
            mean = np.asarray(np.atleast_1d(mean))
            # 获取 mean 向量的维度 d
            d = mean.shape[0]
            # 如果 cov 不为 None，表示提供了协方差矩阵
            if cov is not None:
                # 将 cov 转换为 NumPy 数组，并确保其为二维数组
                cov = np.asarray(np.atleast_2d(cov))
                # 检查 mean 向量和 cov 矩阵的维度是否匹配
                if not mean.shape[0] == cov.shape[0]:
                    raise ValueError("Dimension mismatch between mean and "
                                     "covariance.")
                # 检查 cov 是否为对称矩阵
                if not np.allclose(cov, cov.transpose()):
                    raise ValueError("Covariance matrix is not symmetric.")
                # 尝试计算 Cholesky 分解，如果失败，则使用特征值分解
                try:
                    cov_root = np.linalg.cholesky(cov).transpose()
                except np.linalg.LinAlgError:
                    eigval, eigvec = np.linalg.eigh(cov)
                    # 检查特征值是否非负
                    if not np.all(eigval >= -1.0e-8):
                        raise ValueError("Covariance matrix not PSD.")
                    # 将特征值裁剪为非负值
                    eigval = np.clip(eigval, 0.0, None)
                    # 计算特征向量和特征值的矩阵乘积并开平方
                    cov_root = (eigvec * np.sqrt(eigval)).transpose()
            elif cov_root is not None:
                # 如果 cov_root 已经提供，确保其为二维数组
                cov_root = np.atleast_2d(cov_root)
                # 检查 mean 向量和 cov_root 的维度是否匹配
                if not mean.shape[0] == cov_root.shape[0]:
                    raise ValueError("Dimension mismatch between mean and "
                                     "covariance.")
            else:
                # 如果 cov 和 cov_root 都没有提供，表示协方差矩阵为单位矩阵
                cov_root = None
    
            # 设置反变换标志
            self._inv_transform = inv_transform
    
            if not inv_transform:
                # 如果不使用反变换，计算需要偶数维度的引擎维度
                engine_dim = 2 * math.ceil(d / 2)
            else:
                engine_dim = d
            # 如果 engine 为 None，使用 Sobol 引擎初始化
            if engine is None:
                self.engine = Sobol(
                    d=engine_dim, scramble=True, bits=30, seed=seed
                )  # type: QMCEngine
            elif isinstance(engine, QMCEngine):
                # 如果 engine 为 QMCEngine 类型，检查其维度是否匹配
                if engine.d != engine_dim:
                    raise ValueError("Dimension of `engine` must be consistent"
                                     " with dimensions of mean and covariance."
                                     " If `inv_transform` is False, it must be"
                                     " an even number.")
                self.engine = engine
            else:
                # 否则，抛出类型错误
                raise ValueError("`engine` must be an instance of "
                                 "`scipy.stats.qmc.QMCEngine` or `None`.")
    
            # 设置均值向量和协方差矩阵根
            self._mean = mean
            self._corr_matrix = cov_root
    
            # 设置维度 d
            self._d = d
    
        def random(self, n: IntNumber = 1) -> np.ndarray:
            """Draw `n` QMC samples from the multivariate Normal.
    
            Parameters
            ----------
            n : int, optional
                Number of samples to generate in the parameter space. Default is 1.
    
            Returns
            -------
            sample : array_like (n, d)
                Sample.
    
            """
            # 从标准正态分布生成 n 个样本
            base_samples = self._standard_normal_samples(n)
            # 将标准正态分布样本与协方差矩阵相关联，返回结果样本
            return self._correlate(base_samples)
    # 对基础样本进行相关操作，返回相关处理后的样本数据
    def _correlate(self, base_samples: np.ndarray) -> np.ndarray:
        if self._corr_matrix is not None:
            # 如果存在相关系数矩阵，则进行相关处理并加上平均值
            return base_samples @ self._corr_matrix + self._mean
        else:
            # 如果没有相关系数矩阵，则直接加上平均值
            # 避免在此处与单位矩阵相乘
            return base_samples + self._mean

    # 从标准正态分布 N(0, I_d) 中抽取 `n` 个 QMC 样本
    def _standard_normal_samples(self, n: IntNumber = 1) -> np.ndarray:
        """Draw `n` QMC samples from the standard Normal :math:`N(0, I_d)`.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.

        Returns
        -------
        sample : array_like (n, d)
            Sample.

        """
        # 获取基础样本
        samples = self.engine.random(n)
        if self._inv_transform:
            # 如果启用逆变换
            # （接近0或1的值会导致得到无穷大的值）
            return stats.norm.ppf(0.5 + (1 - 1e-10) * (samples - 0.5))  # type: ignore[attr-defined]  # noqa: E501
        else:
            # 如果使用 Box-Muller 变换（注意索引从1开始）
            even = np.arange(0, samples.shape[-1], 2)
            Rs = np.sqrt(-2 * np.log(samples[:, even]))
            thetas = 2 * math.pi * samples[:, 1 + even]
            cos = np.cos(thetas)
            sin = np.sin(thetas)
            transf_samples = np.stack([Rs * cos, Rs * sin],
                                      -1).reshape(n, -1)
            # 确保返回请求的维度数量
            return transf_samples[:, : self._d]
class MultinomialQMC:
    r"""QMC sampling from a multinomial distribution.

    Parameters
    ----------
    pvals : array_like (k,)
        Vector of probabilities of size ``k``, where ``k`` is the number
        of categories. Elements must be non-negative and sum to 1.
    n_trials : int
        Number of trials.
    engine : QMCEngine, optional
        Quasi-Monte Carlo engine sampler. If None, `Sobol` is used.
    seed : {None, int, `numpy.random.Generator`}, optional
        Used only if `engine` is None.
        If `seed` is an int or None, a new `numpy.random.Generator` is
        created using ``np.random.default_rng(seed)``.
        If `seed` is already a ``Generator`` instance, then the provided
        instance is used.

    Examples
    --------
    Let's define 3 categories and for a given sample, the sum of the trials
    of each category is 8. The number of trials per category is determined
    by the `pvals` associated to each category.
    Then, we sample this distribution 64 times.

    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import qmc
    >>> dist = qmc.MultinomialQMC(
    ...     pvals=[0.2, 0.4, 0.4], n_trials=10, engine=qmc.Halton(d=1)
    ... )
    >>> sample = dist.random(64)

    We can plot the sample and verify that the median of number of trials
    for each category is following the `pvals`. That would be
    ``pvals * n_trials = [2, 4, 4]``.

    >>> fig, ax = plt.subplots()
    >>> ax.yaxis.get_major_locator().set_params(integer=True)
    >>> _ = ax.boxplot(sample)
    >>> ax.set(xlabel="Categories", ylabel="Trials")
    >>> plt.show()

    """

    def __init__(
        self, pvals: npt.ArrayLike, n_trials: IntNumber,
        *, engine: QMCEngine | None = None,
        seed: SeedType = None
    ) -> None:
        # Initialize MultinomialQMC object with given parameters

        # Ensure pvals is converted to numpy array and at least 1-dimensional
        self.pvals = np.atleast_1d(np.asarray(pvals))
        # Check if minimum element of pvals is non-negative
        if np.min(pvals) < 0:
            raise ValueError('Elements of pvals must be non-negative.')
        # Check if sum of pvals is approximately 1
        if not np.isclose(np.sum(pvals), 1):
            raise ValueError('Elements of pvals must sum to 1.')
        # Assign number of trials
        self.n_trials = n_trials
        # Determine QMC engine; default to Sobol if engine is None
        if engine is None:
            self.engine = Sobol(
                d=1, scramble=True, bits=30, seed=seed
            )  # type: QMCEngine
        # Check if engine is an instance of QMCEngine
        elif isinstance(engine, QMCEngine):
            # Ensure engine dimension is 1
            if engine.d != 1:
                raise ValueError("Dimension of `engine` must be 1.")
            self.engine = engine
        else:
            # Raise error if engine is not None or an instance of QMCEngine
            raise ValueError("`engine` must be an instance of "
                             "`scipy.stats.qmc.QMCEngine` or `None`.")
    # 定义一个方法用于从多项分布中抽取 `n` 个 QMC 样本

    sample = np.empty((n, len(self.pvals)))
    # 创建一个空的 NumPy 数组，用于存储抽样结果，形状为 (n, len(self.pvals))

    for i in range(n):
        # 循环 `n` 次，生成每个样本
        base_draws = self.engine.random(self.n_trials).ravel()
        # 从引擎中生成 `n_trials` 个随机数，并展开为一维数组，作为基础抽样数据

        p_cumulative = np.empty_like(self.pvals, dtype=float)
        # 创建与 self.pvals 形状相同的空 NumPy 数组，用于存储累积概率

        _fill_p_cumulative(np.array(self.pvals, dtype=float), p_cumulative)
        # 调用 _fill_p_cumulative 函数，填充 p_cumulative 数组

        sample_ = np.zeros_like(self.pvals, dtype=np.intp)
        # 创建与 self.pvals 形状相同的零数组，用于存储整数型抽样结果

        _categorize(base_draws, p_cumulative, sample_)
        # 调用 _categorize 函数，根据 base_draws 和 p_cumulative 进行抽样，将结果存入 sample_

        sample[i] = sample_
        # 将当前抽样结果 sample_ 存入 sample 数组的第 i 行

    return sample
    # 返回抽样结果数组
# 创建一个函数，根据优化方法名称和配置返回对应的优化器函数
def _select_optimizer(
    optimization: Literal["random-cd", "lloyd"] | None, config: dict
) -> Callable | None:
    """A factory for optimization methods."""
    # 定义不同优化方法名称与对应函数的映射关系
    optimization_method: dict[str, Callable] = {
        "random-cd": _random_cd,
        "lloyd": _lloyd_centroidal_voronoi_tessellation
    }

    optimizer: partial | None
    if optimization is not None:
        try:
            # 将优化方法名称转换为小写，以便统一处理
            optimization = optimization.lower()  # type: ignore[assignment]
            # 根据优化方法名称获取对应的优化函数
            optimizer_ = optimization_method[optimization]
        except KeyError as exc:
            # 若优化方法名称不在预定义的列表中，则抛出错误
            message = (f"{optimization!r} is not a valid optimization"
                       f" method. It must be one of"
                       f" {set(optimization_method)!r}")
            raise ValueError(message) from exc

        # 根据配置参数创建部分应用优化函数的函数对象
        optimizer = partial(optimizer_, **config)
    else:
        # 若未指定优化方法，则返回空值
        optimizer = None

    return optimizer


def _random_cd(
    best_sample: np.ndarray, n_iters: int, n_nochange: int, rng: GeneratorType,
    **kwargs: dict
) -> np.ndarray:
    """Optimal LHS on CD.

    Create a base LHS and do random permutations of coordinates to
    lower the centered discrepancy.
    Because it starts with a normal LHS, it also works with the
    `scramble` keyword argument.

    Two stopping criterion are used to stop the algorithm: at most,
    `n_iters` iterations are performed; or if there is no improvement
    for `n_nochange` consecutive iterations.
    """
    # 删除 kwargs 变量，因为只使用预定义的关键字，这由工厂函数所需
    del kwargs  # only use keywords which are defined, needed by factory

    # 获取最佳样本的行数和列数
    n, d = best_sample.shape

    # 若列数或行数为0，则返回一个空数组
    if d == 0 or n == 0:
        return np.empty((n, d))

    # 若列数或行数为1，则返回原始最佳样本，因为置换不会改变差异度度量
    if d == 1 or n == 1:
        return best_sample

    # 计算最初的差异度度量
    best_disc = discrepancy(best_sample)

    # 定义置换操作的边界条件
    bounds = ([0, d - 1],
              [0, n - 1],
              [0, n - 1])

    # 初始化无改进计数器和迭代次数计数器
    n_nochange_ = 0
    n_iters_ = 0
    while n_nochange_ < n_nochange and n_iters_ < n_iters:
        n_iters_ += 1

        # 随机选择要置换的列索引和行索引
        col = rng_integers(rng, *bounds[0], endpoint=True)  # type: ignore[misc]
        row_1 = rng_integers(rng, *bounds[1], endpoint=True)  # type: ignore[misc]
        row_2 = rng_integers(rng, *bounds[2], endpoint=True)  # type: ignore[misc]
        
        # 计算置换后的差异度度量
        disc = _perturb_discrepancy(best_sample,
                                    row_1, row_2, col,
                                    best_disc)
        
        # 如果置换后的差异度比当前最佳差异度小，则更新最佳样本和差异度
        if disc < best_disc:
            best_sample[row_1, col], best_sample[row_2, col] = (
                best_sample[row_2, col], best_sample[row_1, col])

            best_disc = disc
            n_nochange_ = 0
        else:
            n_nochange_ += 1

    # 返回经过优化后的最佳样本
    return best_sample


def _l1_norm(sample: np.ndarray) -> float:
    # 计算样本的L1范数（曼哈顿距离）
    return distance.pdist(sample, 'cityblock').min()


def _lloyd_iteration(
    sample: np.ndarray,
    decay: float,
    qhull_options: str
) -> np.ndarray:
    """Lloyd-Max algorithm iteration.

    Based on the implementation of Stéfan van der Walt:
    """
    # 根据Stéfan van der Walt的实现，执行Lloyd-Max算法的迭代步骤
    # 创建一个与输入样本相同形状的空数组，用于存储迭代后的样本
    new_sample = np.empty_like(sample)
    
    # 使用输入样本和指定的 Qhull 选项创建 Voronoi 对象
    voronoi = Voronoi(sample, qhull_options=qhull_options)
    
    # 遍历每个点的 Voronoi 区域索引
    for ii, idx in enumerate(voronoi.point_region):
        # 获取当前区域的顶点索引列表，去除无穷远处的样本（索引为 -1）
        region = [i for i in voronoi.regions[idx] if i != -1]
    
        # 根据顶点索引获取实际的顶点坐标
        verts = voronoi.vertices[region]
    
        # 计算当前区域顶点的中心点（重心）
        centroid = np.mean(verts, axis=0)
    
        # 根据松弛系数 decay 将样本向中心点移动一步
        new_sample[ii] = sample[ii] + (centroid - sample[ii]) * decay
    
    # 检查新样本是否仍然位于 [0, 1] 的范围内，生成一个布尔数组
    is_valid = np.all(np.logical_and(new_sample >= 0, new_sample <= 1), axis=1)
    
    # 更新原始样本，只保留在有效范围内的新样本
    sample[is_valid] = new_sample[is_valid]
    
    # 返回更新后的样本
    return sample
# 定义函数 _lloyd_centroidal_voronoi_tessellation，用于近似计算质心松弛 Voronoi 镶嵌
# 参数说明：
# - sample: array_like (n, d)，表示待迭代的样本数据，其中 n 是样本数，d 是样本的维度
# - tol: float, optional，终止条件的容差。如果 L1 范数的最小值在样本上的变化小于 tol，则停止算法。默认为 1e-5
# - maxiter: int, optional，最大迭代次数。即使 tol 超过阈值，也会停止算法。默认为 10
# - qhull_options: str, optional，传递给 Qhull 的额外选项，详见 Qhull 手册。默认为 None
# - **kwargs: dict，其他关键字参数

def _lloyd_centroidal_voronoi_tessellation(
    sample: npt.ArrayLike,
    *,
    tol: DecimalNumber = 1e-5,
    maxiter: IntNumber = 10,
    qhull_options: str | None = None,
    **kwargs: dict
) -> np.ndarray:
    """Approximate Centroidal Voronoi Tessellation.

    Perturb samples in N-dimensions using Lloyd-Max algorithm.

    Parameters
    ----------
    sample : array_like (n, d)
        The sample to iterate on. With ``n`` the number of samples and ``d``
        the dimension. Samples must be in :math:`[0, 1]^d`, with ``d>=2``.
    tol : float, optional
        Tolerance for termination. If the min of the L1-norm over the samples
        changes less than `tol`, it stops the algorithm. Default is 1e-5.
    maxiter : int, optional
        Maximum number of iterations. It will stop the algorithm even if
        `tol` is above the threshold.
        Too many iterations tend to cluster the samples as a hypersphere.
        Default is 10.
    qhull_options : str, optional
        Additional options to pass to Qhull. See Qhull manual
        for details. (Default: "Qbb Qc Qz Qj Qx" for ndim > 4 and
        "Qbb Qc Qz Qj" otherwise.)

    Returns
    -------
    sample : array_like (n, d)
        The sample after being processed by Lloyd-Max algorithm.

    Notes
    -----
    Lloyd-Max algorithm is an iterative process with the purpose of improving
    the dispersion of samples. For given sample: (i) compute a Voronoi
    Tessellation; (ii) find the centroid of each Voronoi cell; (iii) move the
    samples toward the centroid of their respective cell. See [1]_, [2]_.

    A relaxation factor is used to control how fast samples can move at each
    iteration. This factor is starting at 2 and ending at 1 after `maxiter`
    following an exponential decay.

    The process converges to equally spaced samples. It implies that measures
    like the discrepancy could suffer from too many iterations. On the other
    hand, L1 and L2 distances should improve. This is especially true with
    QMC methods which tend to favor the discrepancy over other criteria.

    .. note::

        The current implementation does not intersect the Voronoi Tessellation
        with the boundaries. This implies that for a low number of samples,
        empirically below 20, no Voronoi cell is touching the boundaries.
        Hence, samples cannot be moved close to the boundaries.

        Further improvements could consider the samples at infinity so that
        all boundaries are segments of some Voronoi cells. This would fix
        the computation of the centroid position.

    .. warning::

       The Voronoi Tessellation step is expensive and quickly becomes
       intractable with dimensions as low as 10 even for a sample
       of size as low as 1000.

    .. versionadded:: 1.9.0

    References
    ----------
    .. [1] Lloyd. "Least Squares Quantization in PCM".
       IEEE Transactions on Information Theory, 1982.
    """
    del kwargs  # 只使用已定义的关键字，工厂需要的

    # 将输入样本转换为 NumPy 数组的副本
    sample = np.asarray(sample).copy()

    # 如果样本不是二维数组，则抛出数值错误异常
    if not sample.ndim == 2:
        raise ValueError('`sample` 不是二维数组')

    # 如果样本的列数小于2，则抛出数值错误异常
    if not sample.shape[1] >= 2:
        raise ValueError('`sample` 的维度不足2')

    # 检查样本是否位于单位超立方体内部
    if (sample.max() > 1.) or (sample.min() < 0.):
        raise ValueError('`sample` 不在单位超立方体内')

    # 如果未指定 qhull_options，则设置默认选项字符串
    if qhull_options is None:
        qhull_options = 'Qbb Qc Qz QJ'

        # 如果样本的列数大于等于5，则追加额外的 qhull 选项
        if sample.shape[1] >= 5:
            qhull_options += ' Qx'

    # 拟合一个指数函数，使其在 0 处为 2，在 `maxiter` 处为 1
    # 这个衰减函数用于松弛处理
    # 解析解为 y=exp(-maxiter/x) - 0.1
    root = -maxiter / np.log(0.1)
    decay = [np.exp(-x / root)+0.9 for x in range(maxiter)]

    # 计算初始样本的 L1 范数
    l1_old = _l1_norm(sample=sample)

    # 迭代 `maxiter` 次 Lloyd 算法
    for i in range(maxiter):
        # 执行一次 Lloyd 迭代
        sample = _lloyd_iteration(
                sample=sample, decay=decay[i],
                qhull_options=qhull_options,
        )

        # 计算迭代后的样本的 L1 范数
        l1_new = _l1_norm(sample=sample)

        # 如果新旧 L1 范数的差值小于容差 `tol`，则停止迭代
        if abs(l1_new - l1_old) < tol:
            break
        else:
            l1_old = l1_new

    # 返回迭代后的样本
    return sample
# 验证并返回有效的工作线程数，根据平台和值进行验证
def _validate_workers(workers: IntNumber = 1) -> IntNumber:
    """Validate `workers` based on platform and value.

    Parameters
    ----------
    workers : int, optional
        Number of workers to use for parallel processing. If -1 is
        given all CPU threads are used. Default is 1.

    Returns
    -------
    Workers : int
        Number of CPU used by the algorithm

    """
    # 将输入的 workers 转换为整数
    workers = int(workers)
    # 如果 workers 为 -1，则使用所有可用的 CPU 线程数
    if workers == -1:
        workers = os.cpu_count()  # type: ignore[assignment]
        # 如果无法确定 CPU 数量，则抛出 NotImplementedError
        if workers is None:
            raise NotImplementedError(
                "Cannot determine the number of cpus using os.cpu_count(), "
                "cannot use -1 for the number of workers"
            )
    # 如果 workers 小于等于 0，则抛出 ValueError
    elif workers <= 0:
        raise ValueError(f"Invalid number of workers: {workers}, must be -1 "
                         "or > 0")

    # 返回验证后的 workers 数量
    return workers


# 验证并返回有效的边界值，用于输入的边界数组
def _validate_bounds(
    l_bounds: npt.ArrayLike, u_bounds: npt.ArrayLike, d: int
) -> tuple[np.ndarray, ...]:
    """Bounds input validation.

    Parameters
    ----------
    l_bounds, u_bounds : array_like (d,)
        Lower and upper bounds.
    d : int
        Dimension to use for broadcasting.

    Returns
    -------
    l_bounds, u_bounds : array_like (d,)
        Lower and upper bounds.

    """
    try:
        # 将 lower bounds 和 upper bounds 广播为指定维度 d 的数组
        lower = np.broadcast_to(l_bounds, d)
        upper = np.broadcast_to(u_bounds, d)
    except ValueError as exc:
        # 如果无法广播，则抛出 ValueError
        msg = ("'l_bounds' and 'u_bounds' must be broadcastable and respect"
               " the sample dimension")
        raise ValueError(msg) from exc

    # 检查 lower bounds 是否小于 upper bounds，若不是则抛出 ValueError
    if not np.all(lower < upper):
        raise ValueError("Bounds are not consistent 'l_bounds' < 'u_bounds'")

    # 返回验证后的 lower bounds 和 upper bounds
    return lower, upper
```