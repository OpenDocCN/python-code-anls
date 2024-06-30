# `D:\src\scipysrc\scipy\scipy\stats\_mgc.py`

```
import warnings  # 导入警告模块
import numpy as np  # 导入NumPy库

from scipy._lib._util import check_random_state, MapWrapper, rng_integers, _contains_nan  # 导入SciPy库中的相关函数和类
from scipy._lib._bunch import _make_tuple_bunch
from scipy.spatial.distance import cdist
from scipy.ndimage import _measurements

from ._stats import _local_correlations  # 导入局部相关性统计模块（如果找不到则忽略）
from . import distributions  # 导入本地的distributions模块

__all__ = ['multiscale_graphcorr']  # 模块的公开接口列表，仅包含'multiscale_graphcorr'

# FROM MGCPY: https://github.com/neurodata/mgcpy
# 从MGCPY项目中引用的说明

class _ParallelP:
    """Helper function to calculate parallel p-value."""
    # 辅助函数，用于计算并行p值的类

    def __init__(self, x, y, random_states):
        self.x = x
        self.y = y
        self.random_states = random_states

    def __call__(self, index):
        order = self.random_states[index].permutation(self.y.shape[0])
        permy = self.y[order][:, order]

        # calculate permuted stats, store in null distribution
        perm_stat = _mgc_stat(self.x, permy)[0]

        return perm_stat
        # 计算排列后的统计数据，并存储在空分布中

def _perm_test(x, y, stat, reps=1000, workers=-1, random_state=None):
    r"""Helper function that calculates the p-value. See below for uses.

    Parameters
    ----------
    x, y : ndarray
        `x` and `y` have shapes ``(n, p)`` and ``(n, q)``.
    stat : float
        The sample test statistic.
    reps : int, optional
        The number of replications used to estimate the null when using the
        permutation test. The default is 1000 replications.
    workers : int or map-like callable, optional
        If `workers` is an int the population is subdivided into `workers`
        sections and evaluated in parallel (uses
        `multiprocessing.Pool <multiprocessing>`). Supply `-1` to use all cores
        available to the Process. Alternatively supply a map-like callable,
        such as `multiprocessing.Pool.map` for evaluating the population in
        parallel. This evaluation is carried out as `workers(func, iterable)`.
        Requires that `func` be pickleable.
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Returns
    -------
    pvalue : float
        The sample test p-value.
    null_dist : list
        The approximated null distribution.

    """
    # generate seeds for each rep (change to new parallel random number
    # capabilities in numpy >= 1.17+)
    random_state = check_random_state(random_state)
    random_states = [np.random.RandomState(rng_integers(random_state, 1 << 32,
                     size=4, dtype=np.uint32)) for _ in range(reps)]
    # 为每个重复生成种子（在numpy >= 1.17+中使用新的并行随机数功能）

    # parallelizes with specified workers over number of reps and set seeds
    parallelp = _ParallelP(x=x, y=y, random_states=random_states)
    # 使用指定的工作线程数量对重复次数进行并行化，并设置种子
    # 使用 MapWrapper 类对 workers 进行包装，确保并行处理
    with MapWrapper(workers) as mapwrapper:
        # 使用 map 函数并行计算 parallelp 函数在 range(reps) 范围内的结果，并转换为列表后转换为 numpy 数组
        null_dist = np.array(list(mapwrapper(parallelp, range(reps))))

    # 计算 p-value 和显著的置换映射列表
    pvalue = (1 + (null_dist >= stat).sum()) / (1 + reps)

    # 返回计算得到的 p-value 和空分布的 numpy 数组
    return pvalue, null_dist
def _euclidean_dist(x):
    # 计算数据集 x 中样本点之间的欧氏距离
    return cdist(x, x)


MGCResult = _make_tuple_bunch('MGCResult',
                              ['statistic', 'pvalue', 'mgc_dict'], [])


def multiscale_graphcorr(x, y, compute_distance=_euclidean_dist, reps=1000,
                         workers=1, is_twosamp=False, random_state=None):
    r"""Computes the Multiscale Graph Correlation (MGC) test statistic.

    Specifically, for each point, MGC finds the :math:`k`-nearest neighbors for
    one property (e.g. cloud density), and the :math:`l`-nearest neighbors for
    the other property (e.g. grass wetness) [1]_. This pair :math:`(k, l)` is
    called the "scale". A priori, however, it is not know which scales will be
    most informative. So, MGC computes all distance pairs, and then efficiently
    computes the distance correlations for all scales. The local correlations
    illustrate which scales are relatively informative about the relationship.
    The key, therefore, to successfully discover and decipher relationships
    between disparate data modalities is to adaptively determine which scales
    are the most informative, and the geometric implication for the most
    informative scales. Doing so not only provides an estimate of whether the
    modalities are related, but also provides insight into how the
    determination was made. This is especially important in high-dimensional
    data, where simple visualizations do not reveal relationships to the
    unaided human eye. Characterizations of this implementation in particular
    have been derived from and benchmarked within in [2]_.

    Parameters
    ----------
    x, y : ndarray
        If ``x`` and ``y`` have shapes ``(n, p)`` and ``(n, q)`` where `n` is
        the number of samples and `p` and `q` are the number of dimensions,
        then the MGC independence test will be run.  Alternatively, ``x`` and
        ``y`` can have shapes ``(n, n)`` if they are distance or similarity
        matrices, and ``compute_distance`` must be set to ``None``. If ``x``
        and ``y`` have shapes ``(n, p)`` and ``(m, p)``, an unpaired
        two-sample MGC test will be run.
    compute_distance : callable, optional
        A function that computes the distance or similarity among the samples
        within each data matrix. Set to ``None`` if ``x`` and ``y`` are
        already distance matrices. The default uses the euclidean norm metric.
        If you are calling a custom function, either create the distance
        matrix beforehand or create a function of the form
        ``compute_distance(x)`` where `x` is the data matrix for which
        pairwise distances are calculated.
    reps : int, optional
        The number of replications used to estimate the null when using the
        permutation test. The default is ``1000``.
    # `workers` 参数控制并行计算的工作进程数量或者映射函数
    workers : int or map-like callable, optional
        # 如果 `workers` 是一个整数，将总体分成 `workers` 个部分并行评估（使用 `multiprocessing.Pool`）。
        # 用 `-1` 来使用所有可用的处理器核心。
        # 或者提供一个类似映射函数的可调用对象，例如 `multiprocessing.Pool.map` 用于并行评估 p 值。
        # 这个评估是通过 `workers(func, iterable)` 实现的。
        # 要求 `func` 是可被 pickle 序列化的。
        # 默认值是 `1`。

    # `is_twosamp` 参数指示是否运行双样本检验
    is_twosamp : bool, optional
        # 如果为 `True`，将运行双样本检验。
        # 如果 `x` 和 `y` 的形状分别为 `(n, p)` 和 `(m, p)`，则该选项将被覆盖并设置为 `True`。
        # 如果 `x` 和 `y` 的形状都是 `(n, p)` 并且需要进行双样本检验，则设置为 `True`。
        # 默认值是 `False`。
        # 注意，如果输入是距离矩阵，则不会运行此选项。

    # `random_state` 参数用于控制随机数生成器的种子和状态
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional
        # 如果 `seed` 是 None（或 `np.random`），则使用 `numpy.random.RandomState` 单例。
        # 如果 `seed` 是一个整数，则使用一个新的 `RandomState` 实例，并用 `seed` 初始化。
        # 如果 `seed` 已经是 `Generator` 或 `RandomState` 实例，则直接使用该实例。

    # 返回结果对象 `res` 类型是 `MGCResult`，包含以下属性：
    Returns
    -------
    res : MGCResult
        # 统计量：样本 MGC 测试统计量，取值范围在 `[-1, 1]`。
        statistic : float
        # p 值：通过置换得到的 p 值。
        pvalue : float
        # mgc_dict 是一个字典，包含额外的有用结果：
        mgc_dict : dict
            # mgc_map：关系的潜在几何的二维表示。
            - mgc_map : ndarray
            # opt_scale：估计的最优尺度，以 `(x, y)` 对的形式给出。
            - opt_scale : (int, int)
            # null_dist：从置换矩阵推导出的空分布。
            - null_dist : list

    # 相关文档和函数
    See Also
    --------
    # pearsonr：Pearson 相关系数和非相关性检验的 p 值计算。
    pearsonr : Pearson correlation coefficient and p-value for testing non-correlation.
    # kendalltau：计算 Kendall's tau 相关系数。
    kendalltau : Calculates Kendall's tau.
    # spearmanr：计算 Spearman 秩相关系数。
    spearmanr : Calculates a Spearman rank-order correlation coefficient.

    # 说明和注记
    Notes
    -----
    # MGC 过程的描述和在神经科学数据上的应用可以在 [1]_ 中找到。
    # 它是通过以下步骤执行的：
    # 1. 计算两个距离矩阵 :math:`D^X` 和 :math:`D^Y`，并修改为每列的均值为零。
    #    这将得到两个 :math:`n \times n` 的距离矩阵 :math:`A` 和 :math:`B`（中心化和无偏修改）[3]_。
    A description of the process of MGC and applications on neuroscience data
    can be found in [1]_. It is performed using the following steps:

    # 2. 计算两个距离矩阵 :math:`D^X` 和 :math:`D^Y`，并修改为每列的均值为零。
    #    这将得到两个 :math:`n \times n` 的距离矩阵 :math:`A` 和 :math:`B`（中心化和无偏修改）[3]_.
    #. For all values :math:`k` and :math:`l` from :math:`1, ..., n`,
    
       * The :math:`k`-nearest neighbor and :math:`l`-nearest neighbor graphs
         are calculated for each property. Here, :math:`G_k (i, j)` indicates
         the :math:`k`-smallest values of the :math:`i`-th row of :math:`A`
         and :math:`H_l (i, j)` indicates the :math:`l` smallest values of
         the :math:`i`-th row of :math:`B`
    
       * Let :math:`\circ` denotes the entry-wise matrix product, then local
         correlations are summed and normalized using the following statistic:
    
    .. math::
    
        c^{kl} = \frac{\sum_{ij} A G_k B H_l}
                      {\sqrt{\sum_{ij} A^2 G_k \times \sum_{ij} B^2 H_l}}
    
    #. The MGC test statistic is the smoothed optimal local correlation of
       :math:`\{ c^{kl} \}`. Denote the smoothing operation as :math:`R(\cdot)`
       (which essentially sets all isolated large correlations to 0 and
       retains connected large correlations as before, see [3]_.). MGC is,
    
    .. math::
    
        MGC_n (x, y) = \max_{(k, l)} R \left(c^{kl} \left( x_n, y_n \right)
                                                    \right)
    
       The test statistic returns a value between :math:`(-1, 1)` since it is
       normalized.
    
       The p-value returned is calculated using a permutation test. This process
       is completed by first randomly permuting :math:`y` to estimate the null
       distribution and then calculating the probability of observing a test
       statistic, under the null, at least as extreme as the observed test
       statistic.
    
       MGC requires at least 5 samples to run with reliable results. It can also
       handle high-dimensional data sets.
       In addition, by manipulating the input data matrices, the two-sample
       testing problem can be reduced to the independence testing problem [4]_.
       Given sample data :math:`U` and :math:`V` of sizes :math:`p \times n`
       :math:`p \times m`, data matrix :math:`X` and :math:`Y` can be created as
       follows:
    
    .. math::
    
        X = [U | V] \in \mathcal{R}^{p \times (n + m)}
        Y = [0_{1 \times n} | 1_{1 \times m}] \in \mathcal{R}^{(n + m)}
    
       Then, the MGC statistic can be calculated as normal. This methodology can
       be extended to similar tests such as distance correlation [4]_.
    
    .. versionadded:: 1.4.0
    
    References
    ----------
    .. [1] Vogelstein, J. T., Bridgeford, E. W., Wang, Q., Priebe, C. E.,
           Maggioni, M., & Shen, C. (2019). Discovering and deciphering
           relationships across disparate data modalities. ELife.
    .. [2] Panda, S., Palaniappan, S., Xiong, J., Swaminathan, A.,
           Ramachandran, S., Bridgeford, E. W., ... Vogelstein, J. T. (2019).
           mgcpy: A Comprehensive High Dimensional Independence Testing Python
           Package. :arXiv:`1907.02088`
    """
    # 检查输入的 x 和 y 是否为 numpy 数组，如果不是则抛出数值错误
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("x and y must be ndarrays")

    # 将一维数组 x 转换为二维数组 (n, 1)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    # 如果 x 不是二维数组，则抛出数值错误
    elif x.ndim != 2:
        raise ValueError(f"Expected a 2-D array `x`, found shape {x.shape}")

    # 将一维数组 y 转换为二维数组 (n, 1)
    if y.ndim == 1:
        y = y[:, np.newaxis]
    # 如果 y 不是二维数组，则抛出数值错误
    elif y.ndim != 2:
        raise ValueError(f"Expected a 2-D array `y`, found shape {y.shape}")

    # 获取 x 和 y 的行数和列数
    nx, px = x.shape
    ny, py = y.shape

    # 检查 x 和 y 是否包含 NaN 值，如果有则抛出数值错误
    _contains_nan(x, nan_policy='raise')
    _contains_nan(y, nan_policy='raise')

    # 检查 x 和 y 是否包含正负无穷大的值，如果有则抛出数值错误
    if np.sum(np.isinf(x)) > 0 or np.sum(np.isinf(y)) > 0:
        raise ValueError("Inputs contain infinities")

    # 如果 x 和 y 的行数不相等
    if nx != ny:
        # 如果 x 和 y 的列数相等，则标记为进行两样本检验
        if px == py:
            is_twosamp = True
        else:
            # 否则抛出数值错误，要求 x 和 y 的形状必须为 [n, p] 和 [n, q] 或者 [n, p] 和 [m, p]
            raise ValueError("Shape mismatch, x and y must have shape [n, p] "
                             "and [n, q] or have shape [n, p] and [m, p].")

    # 如果 x 或 y 的行数小于 5，则抛出数值错误，要求至少有 5 个样本以获得合理的结果
    if nx < 5 or ny < 5:
        raise ValueError("MGC requires at least 5 samples to give reasonable "
                         "results.")

    # 将 x 和 y 的数据类型转换为 float64
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    # 检查 compute_distance 是否为可调用对象（函数）
    if not callable(compute_distance) and compute_distance is not None:
        raise ValueError("Compute_distance must be a function.")

    # 检查 reps 是否为整数且大于 0，如果不是则抛出数值错误
    if not isinstance(reps, int) or reps < 0:
        raise ValueError("Number of reps must be an integer greater than 0.")
    """
    # 如果重复次数小于1000，则发出运行时警告，提醒用户可能会得到不可靠的 p 值结果
    elif reps < 1000:
        msg = ("The number of replications is low (under 1000), and p-value "
               "calculations may be unreliable. Use the p-value result, with "
               "caution!")
        # 发出警告消息，指定警告类型为 RuntimeWarning，并指定堆栈级别为第二级
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    # 如果是双样本检验
    if is_twosamp:
        # 如果 compute_distance 为 None，则抛出 ValueError，指示无法运行距离矩阵输入
        if compute_distance is None:
            raise ValueError("Cannot run if inputs are distance matrices")
        # 对 x 和 y 进行双样本转换处理
        x, y = _two_sample_transform(x, y)

    # 如果 compute_distance 不为 None，则计算 x 和 y 的距离矩阵
    if compute_distance is not None:
        # 对 x 和 y 计算距离矩阵
        x = compute_distance(x)
        y = compute_distance(y)

    # 计算 MGC 统计量
    stat, stat_dict = _mgc_stat(x, y)
    # 从统计字典中获取 MGC 统计图
    stat_mgc_map = stat_dict["stat_mgc_map"]
    # 从统计字典中获取最优尺度
    opt_scale = stat_dict["opt_scale"]

    # 计算置换 MGC 的 p 值
    pvalue, null_dist = _perm_test(x, y, stat, reps=reps, workers=workers,
                                   random_state=random_state)

    # 将除统计量和 p 值之外的所有统计信息保存在字典中
    mgc_dict = {"mgc_map": stat_mgc_map,
                "opt_scale": opt_scale,
                "null_dist": null_dist}

    # 创建结果对象，为了向后兼容性，添加别名
    res = MGCResult(stat, pvalue, mgc_dict)
    res.stat = stat
    # 返回结果对象
    return res
def _mgc_stat(distx, disty):
    r"""Helper function that calculates the MGC stat. See above for use.

    Parameters
    ----------
    distx, disty : ndarray
        `distx` and `disty` have shapes ``(n, p)`` and ``(n, q)`` or
        ``(n, n)`` and ``(n, n)``
        if distance matrices.

    Returns
    -------
    stat : float
        The sample MGC test statistic within ``[-1, 1]``.
    stat_dict : dict
        Contains additional useful additional returns containing the following
        keys:

            - stat_mgc_map : ndarray
                MGC-map of the statistics.
            - opt_scale : (float, float)
                The estimated optimal scale as a ``(x, y)`` pair.

    """
    # calculate MGC map and optimal scale
    stat_mgc_map = _local_correlations(distx, disty, global_corr='mgc')

    n, m = stat_mgc_map.shape
    if m == 1 or n == 1:
        # the global scale at is the statistic calculated at maximial nearest
        # neighbors. There is not enough local scale to search over, so
        # default to global scale
        stat = stat_mgc_map[m - 1][n - 1]
        opt_scale = m * n
    else:
        samp_size = len(distx) - 1

        # threshold to find connected region of significant local correlations
        sig_connect = _threshold_mgc_map(stat_mgc_map, samp_size)

        # maximum within the significant region
        stat, opt_scale = _smooth_mgc_map(sig_connect, stat_mgc_map)

    stat_dict = {"stat_mgc_map": stat_mgc_map,
                 "opt_scale": opt_scale}

    return stat, stat_dict


def _threshold_mgc_map(stat_mgc_map, samp_size):
    r"""
    Finds a connected region of significance in the MGC-map by thresholding.

    Parameters
    ----------
    stat_mgc_map : ndarray
        All local correlations within ``[-1,1]``.
    samp_size : int
        The sample size of original data.

    Returns
    -------
    sig_connect : ndarray
        A binary matrix with 1's indicating the significant region.

    """
    m, n = stat_mgc_map.shape

    # 0.02 is simply an empirical threshold, this can be set to 0.01 or 0.05
    # with varying levels of performance. Threshold is based on a beta
    # approximation.
    per_sig = 1 - (0.02 / samp_size)  # Percentile to consider as significant
    threshold = samp_size * (samp_size - 3)/4 - 1/2  # Beta approximation
    threshold = distributions.beta.ppf(per_sig, threshold, threshold) * 2 - 1

    # the global scale at is the statistic calculated at maximial nearest
    # neighbors. Threshold is the maximum on the global and local scales
    threshold = max(threshold, stat_mgc_map[m - 1][n - 1])

    # find the largest connected component of significant correlations
    sig_connect = stat_mgc_map > threshold

    # Return the binary matrix indicating significant regions
    return sig_connect
    # 检查 sig_connect 数组中是否有正数元素之和大于 0
    if np.sum(sig_connect) > 0:
        # 对 sig_connect 数组进行连通区域标记，并返回标记后的数组和标签数量
        sig_connect, _ = _measurements.label(sig_connect)
        
        # 计算除了第一个元素外，其余元素的频次
        _, label_counts = np.unique(sig_connect, return_counts=True)

        # 找到频次最大的标签，跳过第一个元素（通常是零值）
        max_label = np.argmax(label_counts[1:]) + 1
        
        # 将 sig_connect 数组更新为只包含最大标签对应的区域
        sig_connect = sig_connect == max_label
    else:
        # 如果 sig_connect 中没有正数元素，则将其设为包含单个 False 的二维数组
        sig_connect = np.array([[False]])

    # 返回处理后的 sig_connect 数组
    return sig_connect
def _smooth_mgc_map(sig_connect, stat_mgc_map):
    """Finds the smoothed maximal within the significant region R.

    If area of R is too small it returns the last local correlation. Otherwise,
    returns the maximum within significant_connected_region.

    Parameters
    ----------
    sig_connect : ndarray
        A binary matrix with 1's indicating the significant region.
    stat_mgc_map : ndarray
        All local correlations within ``[-1, 1]``.

    Returns
    -------
    stat : float
        The sample MGC statistic within ``[-1, 1]``.
    opt_scale: (float, float)
        The estimated optimal scale as an ``(x, y)`` pair.

    """
    # 获取输入矩阵的形状
    m, n = stat_mgc_map.shape

    # 初始化统计量为最后一个局部相关性
    stat = stat_mgc_map[m - 1][n - 1]
    # 初始最优尺度设定为整个矩阵的大小
    opt_scale = [m, n]

    # 如果显著区域的范围不为零
    if np.linalg.norm(sig_connect) != 0:
        # 仅当连接区域的面积足够大时继续
        # 0.02 是经验阈值，可以根据性能需求设置为 0.01 或 0.05
        if np.sum(sig_connect) >= np.ceil(0.02 * max(m, n)) * min(m, n):
            # 在显著连接区域内找到最大的局部相关性
            max_corr = max(stat_mgc_map[sig_connect])

            # 找到所有在显著连接区域内最大化局部相关性的尺度
            max_corr_index = np.where((stat_mgc_map >= max_corr) & sig_connect)

            # 如果最大相关性大于当前统计量，更新统计量和最优尺度
            if max_corr >= stat:
                stat = max_corr

                # 将二维索引转换为一维索引
                k, l = max_corr_index
                one_d_indices = k * n + l
                k = np.max(one_d_indices) // n
                l = np.max(one_d_indices) % n
                opt_scale = [k+1, l+1]  # 加1以匹配 R 的索引

    return stat, opt_scale


def _two_sample_transform(u, v):
    """Helper function that concatenates x and y for two sample MGC stat.

    See above for use.

    Parameters
    ----------
    u, v : ndarray
        `u` and `v` have shapes ``(n, p)`` and ``(m, p)``.

    Returns
    -------
    x : ndarray
        Concatenate `u` and `v` along the ``axis = 0``. `x` thus has shape
        ``(2n, p)``.
    y : ndarray
        Label matrix for `x` where 0 refers to samples that comes from `u` and
        1 refers to samples that come from `v`. `y` thus has shape ``(2n, 1)``.

    """
    # 获取输入数组的行数
    nx = u.shape[0]
    ny = v.shape[0]

    # 沿着 axis=0 连接 u 和 v，形成新的数组 x
    x = np.concatenate([u, v], axis=0)
    # 创建标签矩阵 y，其中 0 表示来自 u 的样本，1 表示来自 v 的样本
    y = np.concatenate([np.zeros(nx), np.ones(ny)], axis=0).reshape(-1, 1)
    return x, y
```