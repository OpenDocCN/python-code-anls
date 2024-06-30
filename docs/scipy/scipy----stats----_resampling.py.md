# `D:\src\scipysrc\scipy\scipy\stats\_resampling.py`

```
# 引入未来版本的注释语法支持
from __future__ import annotations

# 导入警告模块，用于处理警告信息
import warnings
# 导入 NumPy 库，用于科学计算
import numpy as np
# 导入组合、排列、笛卡尔积等迭代工具
from itertools import combinations, permutations, product
# 导入抽象基类 Sequence
from collections.abc import Sequence
# 导入数据类装饰器
from dataclasses import dataclass
# 导入检查模块，用于检查对象信息
import inspect

# 导入随机状态检查函数、参数重命名、随机整数生成函数
from scipy._lib._util import check_random_state, _rename_parameter, rng_integers
# 导入数组 API 相关函数，例如数组命名空间、判断是否为 NumPy 数组、最小值函数、裁剪函数、移动轴函数
from scipy._lib._array_api import (array_namespace, is_numpy, xp_minimum,
                                   xp_clip, xp_moveaxis_to_end)
# 导入特殊函数模块，例如正态分布的累积分布函数、累积分布函数的反函数、组合数、阶乘
from scipy.special import ndtr, ndtri, comb, factorial

# 导入公共模块，包括置信区间
from ._common import ConfidenceInterval
# 导入轴 NaN 策略模块，用于广播连接和广播数组
from ._axis_nan_policy import _broadcast_concatenate, _broadcast_arrays
# 导入警告与错误处理模块，例如退化数据警告
from ._warnings_errors import DegenerateDataWarning

# 定义可导出的公共接口
__all__ = ['bootstrap', 'monte_carlo_test', 'permutation_test']


def _vectorize_statistic(statistic):
    """向量化 n 个样本的统计量"""
    # 相较于 np.nditer，这种方式更简洁，但会产生一些数据复制：
    # 将样本连接在一起，然后使用 np.apply_along_axis
    def stat_nd(*data, axis=0):
        # 计算每个样本的长度
        lengths = [sample.shape[axis] for sample in data]
        # 计算累积长度，排除最后一个样本
        split_indices = np.cumsum(lengths)[:-1]
        # 将数据广播连接在一起
        z = _broadcast_concatenate(data, axis)

        # 将工作轴移动到位置 0，以便 `statistic` 的输出中新维度是前置的。
        # ("这个轴被移除，并且用新的维度替换...")
        z = np.moveaxis(z, axis, 0)

        def stat_1d(z):
            # 将 z 拆分成数据数组
            data = np.split(z, split_indices)
            # 返回统计量应用在数据上的结果
            return statistic(*data)

        # 沿着轴应用统计量函数
        return np.apply_along_axis(stat_1d, 0, z)[()]
    return stat_nd


def _jackknife_resample(sample, batch=None):
    """杰克刀法对样本进行重抽样。目前仅支持单样本统计。"""
    n = sample.shape[-1]
    batch_nominal = batch or n

    for k in range(0, n, batch_nominal):
        # col_start:col_end 是要移除的观测值
        batch_actual = min(batch_nominal, n-k)

        # 杰克刀法 - 每行留出一个观测值
        j = np.ones((batch_actual, n), dtype=bool)
        np.fill_diagonal(j[:, k:k+batch_actual], False)
        i = np.arange(n)
        i = np.broadcast_to(i, (batch_actual, n))
        i = i[j].reshape((batch_actual, n-1))

        resamples = sample[..., i]
        yield resamples


def _bootstrap_resample(sample, n_resamples=None, random_state=None):
    """自助法对样本进行重抽样。"""
    n = sample.shape[-1]

    # 自助法 - 每行是原始观测的一个随机重抽样
    i = rng_integers(random_state, 0, n, (n_resamples, n))

    resamples = sample[..., i]
    return resamples


def _percentile_of_score(a, score, axis):
    """向量化、简化的 `scipy.stats.percentileofscore`。
    使用 `percentileofscore` 的 'kind' 参数的 'mean' 值逻辑。

    不同于 `stats.percentileofscore`，返回的百分位数是在 [0, 1] 范围内的分数。
    """
    B = a.shape[axis]
    return ((a < score).sum(axis=axis) + (a <= score).sum(axis=axis)) / (2 * B)


def _percentile_along_axis(theta_hat_b, alpha):
    """`np.percentile` with different percentile for each slice."""
    # 使用 `np.percentile` 对每个切片使用不同的百分位数。
    
    # 获取 theta_hat_b 的形状（除去最后一个维度），这表示切片的形状
    shape = theta_hat_b.shape[:-1]
    
    # 将 alpha 广播到与 shape 相同的形状
    alpha = np.broadcast_to(alpha, shape)
    
    # 创建一个与 alpha 相同形状的零矩阵，用于存储百分位数结果
    percentiles = np.zeros_like(alpha, dtype=np.float64)
    
    # 遍历 alpha 的每个索引和对应的 alpha_i 值
    for indices, alpha_i in np.ndenumerate(alpha):
        # 如果 alpha_i 是 NaN，例如当自举分布只有一个唯一元素时
        if np.isnan(alpha_i):
            # 报出警告，指出 BCa 置信区间无法计算的原因，如分布退化或统计量为 np.min
            msg = (
                "The BCa confidence interval cannot be calculated."
                " This problem is known to occur when the distribution"
                " is degenerate or the statistic is np.min."
            )
            warnings.warn(DegenerateDataWarning(msg), stacklevel=3)
            # 将 percentiles 中对应位置设置为 NaN
            percentiles[indices] = np.nan
        else:
            # 获取 theta_hat_b 的当前切片 theta_hat_b_i
            theta_hat_b_i = theta_hat_b[indices]
            # 计算 theta_hat_b_i 切片的 alpha_i 百分位数，并存储在 percentiles 中
            percentiles[indices] = np.percentile(theta_hat_b_i, alpha_i)
    
    # 返回百分位数结果，将其作为标量而不是零维数组返回
    return percentiles[()]
    # 输入数据的统计量置信区间的计算，基于偏差校正和加速方法
    def _bca_interval(data, statistic, axis, alpha, theta_hat_b, batch):
        """Bias-corrected and accelerated interval."""
        # 根据参考文献 [1] 的第 14.3 和 15.4 节 (Eq. 15.36) 实现

        # 计算 theta_hat
        theta_hat = np.asarray(statistic(*data, axis=axis))[..., None]
        # 计算百分位数
        percentile = _percentile_of_score(theta_hat_b, theta_hat, axis=-1)
        # 计算 z0_hat
        z0_hat = ndtri(percentile)

        # 计算 a_hat
        theta_hat_ji = []  # j 表示数据样本，i 表示 Jackknife 重采样
        for j, sample in enumerate(data):
            # _jackknife_resample 在最后一个轴之前添加一个轴，对应不同的 Jackknife 重采样
            samples = [np.expand_dims(sample, -2) for sample in data]
            theta_hat_i = []
            for jackknife_sample in _jackknife_resample(sample, batch):
                samples[j] = jackknife_sample
                broadcasted = _broadcast_arrays(samples, axis=-1)
                theta_hat_i.append(statistic(*broadcasted, axis=-1))
            theta_hat_ji.append(theta_hat_i)

        theta_hat_ji = [np.concatenate(theta_hat_i, axis=-1)
                        for theta_hat_i in theta_hat_ji]

        n_j = [theta_hat_i.shape[-1] for theta_hat_i in theta_hat_ji]

        theta_hat_j_dot = [theta_hat_i.mean(axis=-1, keepdims=True)
                           for theta_hat_i in theta_hat_ji]

        U_ji = [(n - 1) * (theta_hat_dot - theta_hat_i)
                for theta_hat_dot, theta_hat_i, n
                in zip(theta_hat_j_dot, theta_hat_ji, n_j)]

        nums = [(U_i**3).sum(axis=-1)/n**3 for U_i, n in zip(U_ji, n_j)]
        dens = [(U_i**2).sum(axis=-1)/n**2 for U_i, n in zip(U_ji, n_j)]
        a_hat = 1/6 * sum(nums) / sum(dens)**(3/2)

        # 计算 alpha_1, alpha_2
        z_alpha = ndtri(alpha)
        z_1alpha = -z_alpha
        num1 = z0_hat + z_alpha
        alpha_1 = ndtr(z0_hat + num1/(1 - a_hat*num1))
        num2 = z0_hat + z_1alpha
        alpha_2 = ndtr(z0_hat + num2/(1 - a_hat*num2))
        return alpha_1, alpha_2, a_hat  # 返回 a_hat 用于测试


    # `bootstrap` 函数的输入验证和标准化
    def _bootstrap_iv(data, statistic, vectorized, paired, axis, confidence_level,
                      alternative, n_resamples, batch, method, bootstrap_result,
                      random_state):
        """Input validation and standardization for `bootstrap`."""

        if vectorized not in {True, False, None}:
            raise ValueError("`vectorized` must be `True`, `False`, or `None`.")

        if vectorized is None:
            vectorized = 'axis' in inspect.signature(statistic).parameters

        if not vectorized:
            statistic = _vectorize_statistic(statistic)

        axis_int = int(axis)
        if axis != axis_int:
            raise ValueError("`axis` must be an integer.")

        n_samples = 0
        try:
            n_samples = len(data)
    # 如果捕获到 TypeError 异常，则抛出 ValueError，要求 `data` 必须是样本序列。
    except TypeError:
        raise ValueError("`data` must be a sequence of samples.")

    # 如果样本数为 0，则抛出 ValueError，要求 `data` 至少包含一个样本。
    if n_samples == 0:
        raise ValueError("`data` must contain at least one sample.")

    # 定义警告信息，指示忽略 `axis` 指定的维度，因为 `data` 数组的形状不一致。
    message = ("Ignoring the dimension specified by `axis`, arrays in `data` do not "
               "have the same shape. Beginning in SciPy 1.16.0, `bootstrap` will "
               "explicitly broadcast elements of `data` to the same shape (ignoring "
               "`axis`) before performing the calculation. To avoid this warning in "
               "the meantime, ensure that all samples have the same shape (except "
               "potentially along `axis`).")
    
    # 将每个样本转换为至少为 1 维的数组
    data = [np.atleast_1d(sample) for sample in data]
    
    # 收集每个样本经过减少 `axis` 维度后的形状，并确保形状的唯一性
    reduced_shapes = set()
    for sample in data:
        reduced_shape = list(sample.shape)
        reduced_shape.pop(axis)
        reduced_shapes.add(tuple(reduced_shape))
    
    # 如果形状的种类数不为 1，则发出警告信息
    if len(reduced_shapes) != 1:
        warnings.warn(message, FutureWarning, stacklevel=3)

    # 初始化空列表，用于存储转换后的样本
    data_iv = []
    
    # 遍历每个样本，进行必要的维度转换
    for sample in data:
        # 如果指定 `axis` 维度的长度小于等于 1，则抛出 ValueError
        if sample.shape[axis_int] <= 1:
            raise ValueError("each sample in `data` must contain two or more "
                             "observations along `axis`.")
        # 将 `axis_int` 维度移动到数组的最后一维
        sample = np.moveaxis(sample, axis_int, -1)
        # 将转换后的样本添加到 `data_iv` 列表中
        data_iv.append(sample)

    # 检查 `paired` 是否为 True 或 False，否则抛出 ValueError
    if paired not in {True, False}:
        raise ValueError("`paired` must be `True` or `False`.")

    # 如果 `paired` 为 True，则确保所有样本在指定 `axis` 维度上具有相同的长度
    if paired:
        n = data_iv[0].shape[-1]
        for sample in data_iv[1:]:
            if sample.shape[-1] != n:
                message = ("When `paired is True`, all samples must have the "
                           "same length along `axis`")
                raise ValueError(message)

        # 为配对样本统计生成自举分布，重新采样观察索引
        def statistic(i, axis=-1, data=data_iv, unpaired_statistic=statistic):
            data = [sample[..., i] for sample in data]
            return unpaired_statistic(*data, axis=axis)

        # 将 `data_iv` 替换为包含索引范围的列表
        data_iv = [np.arange(n)]

    # 将 `confidence_level` 转换为浮点数
    confidence_level_float = float(confidence_level)

    # 将 `alternative` 转换为小写，并验证其在预定义的替代假设集合中
    alternative = alternative.lower()
    alternatives = {'two-sided', 'less', 'greater'}
    if alternative not in alternatives:
        raise ValueError(f"`alternative` must be one of {alternatives}")

    # 将 `n_resamples` 转换为整数，并验证其为非负整数
    n_resamples_int = int(n_resamples)
    if n_resamples != n_resamples_int or n_resamples_int < 0:
        raise ValueError("`n_resamples` must be a non-negative integer.")

    # 将 `batch` 转换为整数或 None，并验证其为正整数或 None
    if batch is None:
        batch_iv = batch
    else:
        batch_iv = int(batch)
        if batch != batch_iv or batch_iv <= 0:
            raise ValueError("`batch` must be a positive integer or None.")

    # 将 `method` 转换为小写，并验证其在预定义的统计方法集合中
    methods = {'percentile', 'basic', 'bca'}
    method = method.lower()
    if method not in methods:
        raise ValueError(f"`method` must be in {methods}")

    # 定义要求 `bootstrap_result` 必须具有 `bootstrap_distribution` 属性的消息
    message = "`bootstrap_result` must have attribute `bootstrap_distribution'"
    # 检查 bootstrap_result 是否不为 None，并且不具有 "bootstrap_distribution" 属性时，抛出 ValueError 异常
    if (bootstrap_result is not None
            and not hasattr(bootstrap_result, "bootstrap_distribution")):
        raise ValueError(message)

    # 设置错误消息，指示必须至少有一个正值的 `bootstrap_result.bootstrap_distribution.size` 或 `n_resamples`
    message = ("Either `bootstrap_result.bootstrap_distribution.size` or "
               "`n_resamples` must be positive.")

    # 如果 bootstrap_result 为假值或者 bootstrap_result.bootstrap_distribution.size 为假值，
    # 并且 n_resamples_int 为 0 时，抛出 ValueError 异常
    if ((not bootstrap_result or
         not bootstrap_result.bootstrap_distribution.size)
            and n_resamples_int == 0):
        raise ValueError(message)

    # 检查并确保 random_state 是一个有效的随机数生成器对象
    random_state = check_random_state(random_state)

    # 返回元组包含以下参数：data_iv, statistic, vectorized, paired, axis_int,
    # confidence_level_float, alternative, n_resamples_int, batch_iv, method,
    # bootstrap_result, random_state
    return (data_iv, statistic, vectorized, paired, axis_int,
            confidence_level_float, alternative, n_resamples_int, batch_iv,
            method, bootstrap_result, random_state)
@dataclass
class BootstrapResult:
    """Result object returned by `scipy.stats.bootstrap`.

    Attributes
    ----------
    confidence_interval : ConfidenceInterval
        The bootstrap confidence interval as an instance of
        `collections.namedtuple` with attributes `low` and `high`.
    bootstrap_distribution : ndarray
        The bootstrap distribution, that is, the value of `statistic` for
        each resample. The last dimension corresponds with the resamples
        (e.g. ``res.bootstrap_distribution.shape[-1] == n_resamples``).
    standard_error : float or ndarray
        The bootstrap standard error, that is, the sample standard
        deviation of the bootstrap distribution.

    """
    confidence_interval: ConfidenceInterval
    bootstrap_distribution: np.ndarray
    standard_error: float | np.ndarray


def bootstrap(data, statistic, *, n_resamples=9999, batch=None,
              vectorized=None, paired=False, axis=0, confidence_level=0.95,
              alternative='two-sided', method='BCa', bootstrap_result=None,
              random_state=None):
    r"""
    Compute a two-sided bootstrap confidence interval of a statistic.

    When `method` is ``'percentile'`` and `alternative` is ``'two-sided'``,
    a bootstrap confidence interval is computed according to the following
    procedure.

    1. Resample the data: for each sample in `data` and for each of
       `n_resamples`, take a random sample of the original sample
       (with replacement) of the same size as the original sample.

    2. Compute the bootstrap distribution of the statistic: for each set of
       resamples, compute the test statistic.

    3. Determine the confidence interval: find the interval of the bootstrap
       distribution that is

       - symmetric about the median and
       - contains `confidence_level` of the resampled statistic values.

    While the ``'percentile'`` method is the most intuitive, it is rarely
    used in practice. Two more common methods are available, ``'basic'``
    ('reverse percentile') and ``'BCa'`` ('bias-corrected and accelerated');
    they differ in how step 3 is performed.

    If the samples in `data` are  taken at random from their respective
    distributions :math:`n` times, the confidence interval returned by
    `bootstrap` will contain the true value of the statistic for those
    distributions approximately `confidence_level`:math:`\, \times \, n` times.

    Parameters
    ----------
    data : array_like
        The input data from which the statistic is computed.
    statistic : function
        The function used to compute the statistic from `data`.
    n_resamples : int, optional
        Number of bootstrap resamples (default is 9999).
    batch : int or None, optional
        Number of resamples to compute in each batch, if using batching.
    vectorized : bool or None, optional
        Whether to use vectorized operations for efficiency.
    paired : bool, optional
        Whether the data are paired (default is False).
    axis : int, optional
        The axis along which to resample the data (default is 0).
    confidence_level : float, optional
        Desired confidence level for the interval (default is 0.95).
    alternative : {'two-sided', 'lower', 'upper'}, optional
        Defines the alternative hypothesis (default is 'two-sided').
    method : {'percentile', 'basic', 'BCa'}, optional
        The method to compute the confidence interval (default is 'BCa').
    bootstrap_result : BootstrapResult or None, optional
        Pre-existing result object to store the bootstrap results.
    random_state : int or np.random.RandomState, optional
        Random seed or RandomState instance.

    Returns
    -------
    BootstrapResult
        An object containing the computed bootstrap results.
    """
    data : sequence of array-like
         每个元素都是一个样本，包含来自基础分布的标量观察结果。`data` 的每个元素必须能够广播到相同的形状（除了 `axis` 指定的维度可能有例外）。

         .. versionchanged:: 1.14.0
             如果 `data` 元素的形状不同（除了 `axis` 指定的维度），`bootstrap` 现在会发出 ``FutureWarning`` 。
             从 SciPy 1.16.0 开始，`bootstrap` 将在执行计算之前显式地将元素广播到相同的形状（除了 `axis`）。

    statistic : callable
        用于计算置信区间的统计量。`statistic` 必须是一个可调用对象，接受 ``len(data)`` 个样本作为单独的参数，并返回结果统计量。
        如果 `vectorized` 设置为 ``True``，则 `statistic` 还必须接受关键字参数 `axis`，并且在提供的 `axis` 上进行向量化计算。
    n_resamples : int, default: ``9999``
        形成统计量的自助法分布的重采样次数。
    batch : int, optional
        每个向量化调用 `statistic` 处理的重采样次数。内存使用量为 O( `batch` * ``n`` )，其中 ``n`` 是样本大小。默认为 ``None``，此时 ``batch = n_resamples``（或者 ``batch = max(n_resamples, n)`` 用于 ``method='BCa'``）。
    vectorized : bool, optional
        如果 `vectorized` 设置为 ``False``，则 `statistic` 不会传递关键字参数 `axis`，且预期仅针对 1D 样本计算统计量。如果设置为 ``True``，则在传递 ND 样本数组时，`statistic` 应接受关键字参数 `axis`，并且预期沿 `axis` 计算统计量。如果为 ``None``（默认），并且 `statistic` 的参数中包含 `axis`，则 `vectorized` 将设置为 ``True``。通常使用向量化统计量可以减少计算时间。
    paired : bool, default: ``False``
        统计量是否将 `data` 中的对应元素视为配对处理。
    axis : int, default: ``0``
        `data` 中样本的轴，沿着该轴计算 `statistic`。
    confidence_level : float, default: ``0.95``
        置信区间的置信水平。
    alternative : {'two-sided', 'less', 'greater'}, default: ``'two-sided'``
    # 参数 alternative 可选值为 {'two-sided', 'less', 'greater'}，默认为 'two-sided'
    # 'two-sided' 表示双侧置信区间，'less' 表示单侧置信区间，下界为 -np.inf，'greater' 表示单侧置信区间，上界为 np.inf。
    # 单侧置信区间的另一个边界与双侧置信区间的边界相同，只是置信水平与 1.0 的距离是双侧置信区间的两倍；例如，95% 'less' 置信区间的上界与 90% 'two-sided' 置信区间的上界相同。

    method : {'percentile', 'basic', 'bca'}, default: ``'BCa'``
    # 参数 method 可选值为 {'percentile', 'basic', 'bca'}，默认为 'BCa'
    # 'percentile' 表示百分位数法计算的自助法置信区间
    # 'basic'（也称为 'reverse'）表示基本自助法置信区间
    # 'BCa' 表示修正的加速置信区间（bias-corrected and accelerated）

    bootstrap_result : BootstrapResult, optional
    # 可选参数 bootstrap_result，类型为 BootstrapResult
    # 如果提供了此参数，则使用先前调用 `bootstrap` 返回的结果对象，将其包含在新的自助法分布中
    # 可以用来修改 `confidence_level`，修改 `method`，或查看执行额外重采样的效果，而无需重复计算。

    random_state : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
    # 可选参数 random_state，用于生成重采样的伪随机数生成器状态
    # 如果 `random_state` 是 None（或 `np.random`），则使用 `numpy.random.RandomState` 单例
    # 如果 `random_state` 是一个整数，将使用一个新的 `RandomState` 实例，并以 `random_state` 为种子
    # 如果 `random_state` 已经是 `Generator` 或 `RandomState` 实例，则直接使用该实例

    Returns
    -------
    res : BootstrapResult
    # 返回值为 BootstrapResult 对象
    # 其包含以下属性：

        confidence_interval : ConfidenceInterval
        # 置信区间，是一个 `collections.namedtuple` 实例，具有 `low` 和 `high` 两个属性

        bootstrap_distribution : ndarray
        # 自助法分布，即每个重采样对应的 `statistic` 值。最后一个维度对应重采样次数（例如，`res.bootstrap_distribution.shape[-1] == n_resamples`）

        standard_error : float or ndarray
        # 自助法标准误差，即自助法分布的样本标准差

    Warns
    -----
    `~scipy.stats.DegenerateDataWarning`
    # 如果 `method='BCa'` 并且自助法分布是退化的（例如，所有元素相同），则发出警告。

    Notes
    -----
    # 如果 `method='BCa'`，则置信区间的元素可能为 NaN，这种情况通常发生在自助法分布为退化分布时。
    the bootstrap distribution is degenerate (e.g. all elements are identical).
    In this case, consider using another `method` or inspecting `data` for
    indications that other analysis may be more appropriate (e.g. all
    observations are identical).

    References
    ----------
    .. [1] B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap,
       Chapman & Hall/CRC, Boca Raton, FL, USA (1993)
    .. [2] Nathaniel E. Helwig, "Bootstrap Confidence Intervals",
       http://users.stat.umn.edu/~helwig/notes/bootci-Notes.pdf
    .. [3] Bootstrapping (statistics), Wikipedia,
       https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29

    Examples
    --------
    Suppose we have sampled data from an unknown distribution.

    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> from scipy.stats import norm
    >>> dist = norm(loc=2, scale=4)  # our "unknown" distribution
    >>> data = dist.rvs(size=100, random_state=rng)

    We are interested in the standard deviation of the distribution.

    >>> std_true = dist.std()      # the true value of the statistic
    >>> print(std_true)
    4.0
    >>> std_sample = np.std(data)  # the sample statistic
    >>> print(std_sample)
    3.9460644295563863

    The bootstrap is used to approximate the variability we would expect if we
    were to repeatedly sample from the unknown distribution and calculate the
    statistic of the sample each time. It does this by repeatedly resampling
    values *from the original sample* with replacement and calculating the
    statistic of each resample. This results in a "bootstrap distribution" of
    the statistic.

    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import bootstrap
    >>> data = (data,)  # samples must be in a sequence
    >>> res = bootstrap(data, np.std, confidence_level=0.9,
    ...                 random_state=rng)
    >>> fig, ax = plt.subplots()
    >>> ax.hist(res.bootstrap_distribution, bins=25)
    >>> ax.set_title('Bootstrap Distribution')
    >>> ax.set_xlabel('statistic value')
    >>> ax.set_ylabel('frequency')
    >>> plt.show()

    The standard error quantifies this variability. It is calculated as the
    standard deviation of the bootstrap distribution.

    >>> res.standard_error
    0.24427002125829136
    >>> res.standard_error == np.std(res.bootstrap_distribution, ddof=1)
    True

    The bootstrap distribution of the statistic is often approximately normal
    with scale equal to the standard error.

    >>> x = np.linspace(3, 5)
    >>> pdf = norm.pdf(x, loc=std_sample, scale=res.standard_error)
    >>> fig, ax = plt.subplots()
    >>> ax.hist(res.bootstrap_distribution, bins=25, density=True)
    >>> ax.plot(x, pdf)
    >>> ax.set_title('Normal Approximation of the Bootstrap Distribution')
    >>> ax.set_xlabel('statistic value')
    >>> ax.set_ylabel('pdf')
    >>> plt.show()

    This suggests that we could construct a 90% confidence interval on the
    statistic based on quantiles of this normal distribution.
    # 基于该正态分布的分位数统计量。

    >>> norm.interval(0.9, loc=std_sample, scale=res.standard_error)
    (3.5442759991341726, 4.3478528599786)
    # 使用正态分布的区间方法计算置信区间，0.9 是置信水平，std_sample 是均值，res.standard_error 是标准误差。

    Due to central limit theorem, this normal approximation is accurate for a
    variety of statistics and distributions underlying the samples; however,
    the approximation is not reliable in all cases. Because `bootstrap` is
    designed to work with arbitrary underlying distributions and statistics,
    it uses more advanced techniques to generate an accurate confidence
    interval.
    # 根据中心极限定理，正态近似适用于多种样本的统计量和分布；但并非在所有情况下都可靠。因为 `bootstrap` 被设计用于任意的底层分布和统计量，它使用更高级的技术生成准确的置信区间。

    >>> print(res.confidence_interval)
    ConfidenceInterval(low=3.57655333533867, high=4.382043696342881)
    # 打印出 `bootstrap` 函数计算得到的置信区间对象，包含低和高边界。

    If we sample from the original distribution 100 times and form a bootstrap
    confidence interval for each sample, the confidence interval
    contains the true value of the statistic approximately 90% of the time.

    >>> n_trials = 100
    >>> ci_contains_true_std = 0
    >>> for i in range(n_trials):
    ...    data = (dist.rvs(size=100, random_state=rng),)
    ...    res = bootstrap(data, np.std, confidence_level=0.9,
    ...                    n_resamples=999, random_state=rng)
    ...    ci = res.confidence_interval
    ...    if ci[0] < std_true < ci[1]:
    ...        ci_contains_true_std += 1
    >>> print(ci_contains_true_std)
    88
    # 进行100次抽样，并针对每次样本生成一个 bootstrap 置信区间。统计这些置信区间中大约90%包含真实统计量 `std_true` 的次数。

    Rather than writing a loop, we can also determine the confidence intervals
    for all 100 samples at once.

    >>> data = (dist.rvs(size=(n_trials, 100), random_state=rng),)
    >>> res = bootstrap(data, np.std, axis=-1, confidence_level=0.9,
    ...                 n_resamples=999, random_state=rng)
    >>> ci_l, ci_u = res.confidence_interval
    # 相比使用循环，我们也可以一次性计算所有100个样本的置信区间。

    Here, `ci_l` and `ci_u` contain the confidence interval for each of the
    ``n_trials = 100`` samples.

    >>> print(ci_l[:5])
    [3.86401283 3.33304394 3.52474647 3.54160981 3.80569252]
    >>> print(ci_u[:5])
    [4.80217409 4.18143252 4.39734707 4.37549713 4.72843584]
    # 输出前5个样本的置信区间下限和上限。

    And again, approximately 90% contain the true value, ``std_true = 4``.

    >>> print(np.sum((ci_l < std_true) & (std_true < ci_u)))
    93
    # 统计这些置信区间中大约90%包含真实值 `std_true = 4` 的次数。

    `bootstrap` can also be used to estimate confidence intervals of
    multi-sample statistics. For example, to get a confidence interval
    for the difference between means, we write a function that accepts
    two sample arguments and returns only the statistic. The use of the
    ``axis`` argument ensures that all mean calculations are perform in
    a single vectorized call, which is faster than looping over pairs
    of resamples in Python.
    # `bootstrap` 也可用于估计多样本统计量的置信区间。例如，可以计算两个样本均值之差的置信区间。

    >>> def my_statistic(sample1, sample2, axis=-1):
    ...     mean1 = np.mean(sample1, axis=axis)
    ...     mean2 = np.mean(sample2, axis=axis)
    ...     return mean1 - mean2
    # 定义一个函数计算两个样本均值的差值作为统计量。

    Here, we use the 'percentile' method with the default 95% confidence level.

    >>> sample1 = norm.rvs(scale=1, size=100, random_state=rng)
    >>> sample2 = norm.rvs(scale=2, size=100, random_state=rng)
    >>> data = (sample1, sample2)
    # 创建两个正态分布样本 `sample1` 和 `sample2`，并合并成一个数据元组。
    # 调用 bootstrap 函数并记录结果，使用给定的数据和统计方法
    res = bootstrap(data, my_statistic, method='basic', random_state=rng)
    
    # 打印两个样本的统计量
    print(my_statistic(sample1, sample2))
    
    # 输出统计结果
    0.16661030792089523
    
    # 打印 bootstrap 结果的置信区间
    print(res.confidence_interval)
    
    # 输出置信区间对象，包含上下界
    ConfidenceInterval(low=-0.29087973240818693, high=0.6371338699912273)
    
    # 输出 bootstrap 估计的标准误差
    print(res.standard_error)
    
    # 输出标准误差的值
    0.238323948262459
    
    # 使用配对样本统计，例如计算 Pearson 相关系数
    from scipy.stats import pearsonr
    
    # 设置样本大小
    n = 100
    
    # 生成 x 轴数据
    x = np.linspace(0, 10, n)
    
    # 生成 y 轴数据，带有随机噪声
    y = x + rng.uniform(size=n)
    
    # 打印 Pearson 相关系数的值（第一个元素为统计量）
    print(pearsonr(x, y)[0])
    
    # 输出 Pearson 相关系数的值
    0.9954306665125647
    
    # 自定义统计函数 my_statistic，返回 Pearson 相关系数
    def my_statistic(x, y, axis=-1):
        return pearsonr(x, y, axis=axis)[0]
    
    # 使用 paired=True 参数调用 bootstrap 函数
    res = bootstrap((x, y), my_statistic, paired=True, random_state=rng)
    
    # 打印配对样本的置信区间
    print(res.confidence_interval)
    
    # 输出置信区间对象，包含上下界
    ConfidenceInterval(low=0.9941504301315878, high=0.996377412215445)
    
    # 访问 bootstrap 结果对象的 bootstrap_distribution 属性
    len(res.bootstrap_distribution)
    
    # 输出 bootstrap 分布的长度
    9999
    
    # 重新使用原始 bootstrap 结果对象进行额外的重抽样
    res = bootstrap((x, y), my_statistic, paired=True,
                    n_resamples=1000, random_state=rng,
                    bootstrap_result=res)
    
    # 访问更新后的 bootstrap 结果对象的 bootstrap_distribution 属性
    len(res.bootstrap_distribution)
    
    # 输出更新后的 bootstrap 分布的长度
    10999
    
    # 使用不同的置信区间选项再次调用 bootstrap 函数，通过 res 参数重复使用原始结果
    res2 = bootstrap((x, y), my_statistic, paired=True,
                     n_resamples=0, random_state=rng, bootstrap_result=res,
                     method='percentile', confidence_level=0.9)
    
    # 使用 NumPy 的测试工具确保两个 bootstrap 分布相等
    np.testing.assert_equal(res2.bootstrap_distribution,
                            res.bootstrap_distribution)
    
    # 打印原始结果对象的置信区间
    res.confidence_interval
    
    # 输出置信区间对象，包含上下界
    ConfidenceInterval(low=0.9941574828235082, high=0.9963781698210212)
    
    # 结束代码块
    for k in range(0, n_resamples, batch_nominal):
        # 循环生成每个批次的重抽样数据
        batch_actual = min(batch_nominal, n_resamples-k)
        # 计算当前批次的实际大小，不超过剩余的重抽样次数

        # Generate resamples
        resampled_data = []
        for sample in data:
            # 对每个样本进行自助重抽样
            resample = _bootstrap_resample(sample, n_resamples=batch_actual,
                                           random_state=random_state)
            resampled_data.append(resample)

        # Compute bootstrap distribution of statistic
        # 计算统计量的自助法分布
        theta_hat_b.append(statistic(*resampled_data, axis=-1))
    theta_hat_b = np.concatenate(theta_hat_b, axis=-1)

    # Calculate percentile interval
    # 计算百分位区间
    alpha = ((1 - confidence_level)/2 if alternative == 'two-sided'
             else (1 - confidence_level))

    if method == 'bca':
        interval = _bca_interval(data, statistic, axis=-1, alpha=alpha,
                                 theta_hat_b=theta_hat_b, batch=batch)[:2]
        percentile_fun = _percentile_along_axis
    else:
        # 如果不是BCA方法，则使用简单的alpha百分位作为区间
        interval = alpha, 1-alpha

        def percentile_fun(a, q):
            # 定义百分位函数，计算给定轴上的百分位数
            return np.percentile(a=a, q=q, axis=-1)

    # Calculate confidence interval of statistic
    # 计算统计量的置信区间
    ci_l = percentile_fun(theta_hat_b, interval[0]*100)
    ci_u = percentile_fun(theta_hat_b, interval[1]*100)

    if method == 'basic':  # see [3]
        # 如果方法是基本的，根据[3]调整置信区间
        theta_hat = statistic(*data, axis=-1)
        ci_l, ci_u = 2*theta_hat - ci_u, 2*theta_hat - ci_l

    if alternative == 'less':
        # 如果备择假设为"小于"，下限置信区间设为负无穷
        ci_l = np.full_like(ci_l, -np.inf)
    elif alternative == 'greater':
        # 如果备择假设为"大于"，上限置信区间设为正无穷
        ci_u = np.full_like(ci_u, np.inf)

    # 返回自助法的结果对象，包括置信区间、自助法分布和标准误差
    return BootstrapResult(confidence_interval=ConfidenceInterval(ci_l, ci_u),
                           bootstrap_distribution=theta_hat_b,
                           standard_error=np.std(theta_hat_b, ddof=1, axis=-1))
# 定义函数 `_monte_carlo_test_iv`，用于验证 `monte_carlo_test` 的输入参数
def _monte_carlo_test_iv(data, rvs, statistic, vectorized, n_resamples,
                         batch, alternative, axis):
    """Input validation for `monte_carlo_test`."""

    # 将 `axis` 转换为整数类型
    axis_int = int(axis)
    if axis != axis_int:
        raise ValueError("`axis` must be an integer.")

    # 检查 `vectorized` 是否为 `True`、`False` 或 `None`
    if vectorized not in {True, False, None}:
        raise ValueError("`vectorized` must be `True`, `False`, or `None`.")

    # 如果 `rvs` 不是序列，则转换为单元素元组
    if not isinstance(rvs, Sequence):
        rvs = (rvs,)
        data = (data,)

    # 检查每个 `rvs` 是否可调用
    for rvs_i in rvs:
        if not callable(rvs_i):
            raise TypeError("`rvs` must be callable or sequence of callables.")

    # 如果 `data` 不是序列，则抛出异常
    message = "If `rvs` is a sequence, `len(rvs)` must equal `len(data)`."
    try:
        len(data)
    except TypeError as e:
        raise ValueError(message) from e
    if not len(rvs) == len(data):
        raise ValueError(message)

    # 检查 `statistic` 是否可调用
    if not callable(statistic):
        raise TypeError("`statistic` must be callable.")

    # 如果 `vectorized` 是 `None`，根据 `statistic` 的签名检查是否需要向量化
    if vectorized is None:
        try:
            signature = inspect.signature(statistic).parameters
        except ValueError as e:
            message = (f"Signature inspection of {statistic=} failed; "
                       "pass `vectorize` explicitly.")
            raise ValueError(message) from e
        vectorized = 'axis' in signature

    # 使用 `array_namespace` 函数创建 `xp` 对象
    xp = array_namespace(*data)

    # 如果 `vectorized` 是 `False`，则检查 `statistic` 是否支持向量化
    if not vectorized:
        if is_numpy(xp):
            statistic_vectorized = _vectorize_statistic(statistic)
        else:
            message = ("`statistic` must be vectorized (i.e. support an `axis` "
                       f"argument) when `data` contains {xp.__name__} arrays.")
            raise ValueError(message)
    else:
        statistic_vectorized = statistic

    # 将 `data` 广播到指定的 `axis` 上
    data = _broadcast_arrays(data, axis, xp=xp)
    data_iv = []

    # 将每个样本广播并移到 `axis_int` 位置后存入 `data_iv`
    for sample in data:
        sample = xp.broadcast_to(sample, (1,)) if sample.ndim == 0 else sample
        sample = xp_moveaxis_to_end(sample, axis_int, xp=xp)
        data_iv.append(sample)

    # 将 `n_resamples` 转换为整数类型，并检查是否为正整数
    n_resamples_int = int(n_resamples)
    if n_resamples != n_resamples_int or n_resamples_int <= 0:
        raise ValueError("`n_resamples` must be a positive integer.")

    # 设置 `batch_iv`，如果 `batch` 为 `None`，则保持为 `None`，否则转换为整数并检查是否为正整数
    if batch is None:
        batch_iv = batch
    else:
        batch_iv = int(batch)
        if batch != batch_iv or batch_iv <= 0:
            raise ValueError("`batch` must be a positive integer or None.")

    # 定义允许的 `alternative` 值集合，并将 `alternative` 转换为小写检查其是否在允许的集合中
    alternatives = {'two-sided', 'greater', 'less'}
    alternative = alternative.lower()
    if alternative not in alternatives:
        raise ValueError(f"`alternative` must be in {alternatives}")

    # 推断所需的 p 值数据类型基于输入类型
    min_float = getattr(xp, 'float16', xp.float32)
    dtype = xp.result_type(*data_iv, min_float)

    # 返回输入参数的元组
    return (data_iv, rvs, statistic_vectorized, vectorized, n_resamples_int,
            batch_iv, alternative, axis_int, dtype, xp)
# 定义一个用于存储 Monte Carlo 测试结果的类
class MonteCarloTestResult:
    """Result object returned by `scipy.stats.monte_carlo_test`.

    Attributes
    ----------
    statistic : float or ndarray
        The observed test statistic of the sample.
    pvalue : float or ndarray
        The p-value for the given alternative.
    null_distribution : ndarray
        The values of the test statistic generated under the null
        hypothesis.
    """
    statistic: float | np.ndarray  # 表示观察到的样本测试统计量，可以是单个值或多维数组
    pvalue: float | np.ndarray  # 表示给定备择假设下的 p 值，可以是单个值或多维数组
    null_distribution: np.ndarray  # 表示在零假设下生成的测试统计量的值，是一个多维数组


# 用于重命名参数的装饰器，将 'sample' 重命名为 'data'
@_rename_parameter('sample', 'data')
def monte_carlo_test(data, rvs, statistic, *, vectorized=None,
                     n_resamples=9999, batch=None, alternative="two-sided",
                     axis=0):
    r"""Perform a Monte Carlo hypothesis test.

    `data` contains a sample or a sequence of one or more samples. `rvs`
    specifies the distribution(s) of the sample(s) in `data` under the null
    hypothesis. The value of `statistic` for the given `data` is compared
    against a Monte Carlo null distribution: the value of the statistic for
    each of `n_resamples` sets of samples generated using `rvs`. This gives
    the p-value, the probability of observing such an extreme value of the
    test statistic under the null hypothesis.

    Parameters
    ----------
    data : array-like or sequence of array-like
        An array or sequence of arrays of observations.
    rvs : callable or tuple of callables
        A callable or sequence of callables that generates random variates
        under the null hypothesis. Each element of `rvs` must be a callable
        that accepts keyword argument ``size`` (e.g. ``rvs(size=(m, n))``) and
        returns an N-d array sample of that shape. If `rvs` is a sequence, the
        number of callables in `rvs` must match the number of samples in
        `data`, i.e. ``len(rvs) == len(data)``. If `rvs` is a single callable,
        `data` is treated as a single sample.
    statistic : callable
        Statistic for which the p-value of the hypothesis test is to be
        calculated. `statistic` must be a callable that accepts a sample
        (e.g. ``statistic(sample)``) or ``len(rvs)`` separate samples (e.g.
        ``statistic(samples1, sample2)`` if `rvs` contains two callables and
        `data` contains two samples) and returns the resulting statistic.
        If `vectorized` is set ``True``, `statistic` must also accept a keyword
        argument `axis` and be vectorized to compute the statistic along the
        provided `axis` of the samples in `data`.
    # 如果 `vectorized` 参数设置为 `False`，则 `statistic` 函数不会接收 `axis` 关键字参数，
    # 并且期望只针对1维样本计算统计量。如果设置为 `True`，则 `statistic` 函数将接收 `axis` 关键字参数，
    # 并期望在传递ND样本数组时沿着指定的 `axis` 计算统计量。如果设为 `None`（默认值），
    # 如果 `statistic` 的参数列表中包含 `axis`，则 `vectorized` 将被设置为 `True`。
    # 使用向量化统计通常可以减少计算时间。
    
        vectorized : bool, optional
    
    # `n_resamples` 表示从 `rvs` 中每个可调用对象中抽取的样本数量。
    # 同样，它也是在蒙特卡洛模拟中用作空假设下的统计值数量。
    # 默认值为 9999。
    
        n_resamples : int, default: 9999
    
    # `batch` 参数表示在每次调用 `statistic` 函数时处理的蒙特卡洛样本数。
    # 内存使用量为 O(`batch` * `sample.size[axis]`)。默认为 `None`，此时 `batch` 等于 `n_resamples`。
    
        batch : int, optional
    
    # `alternative` 参数指定用于计算 p 值的备择假设类型。
    # 对于每种备择假设，p 值的定义如下：
    # - `'greater'`：空假设分布中大于等于观察到的检验统计量值的百分比。
    # - `'less'`：空假设分布中小于等于观察到的检验统计量值的百分比。
    # - `'two-sided'`：上述两个 p 值中较小的值的两倍。
    
        alternative : {'two-sided', 'less', 'greater'}
    
    # `axis` 参数表示在 `data` 中（或者在 `data` 中的每个样本中）计算统计量的轴向。
    
        axis : int, default: 0
    
    # 返回一个 `MonteCarloTestResult` 对象，其具有以下属性：
    # - `statistic`：观察到的 `data` 的检验统计量，可以是浮点数或者 ndarray。
    # - `pvalue`：给定备择假设的 p 值，可以是浮点数或者 ndarray。
    # - `null_distribution`：在空假设下生成的检验统计量的值组成的 ndarray。
    
        Returns
        -------
        res : MonteCarloTestResult
    .. warning::
        The p-value is calculated by counting the elements of the null
        distribution that are as extreme or more extreme than the observed
        value of the statistic. Due to the use of finite precision arithmetic,
        some statistic functions return numerically distinct values when the
        theoretical values would be exactly equal. In some cases, this could
        lead to a large error in the calculated p-value. `monte_carlo_test`
        guards against this by considering elements in the null distribution
        that are "close" (within a relative tolerance of 100 times the
        floating point epsilon of inexact dtypes) to the observed
        value of the test statistic as equal to the observed value of the
        test statistic. However, the user is advised to inspect the null
        distribution to assess whether this method of comparison is
        appropriate, and if not, calculate the p-value manually.

    References
    ----------

    .. [1] B. Phipson and G. K. Smyth. "Permutation P-values Should Never Be
       Zero: Calculating Exact P-values When Permutations Are Randomly Drawn."
       Statistical Applications in Genetics and Molecular Biology 9.1 (2010).

    Examples
    --------

    Suppose we wish to test whether a small sample has been drawn from a normal
    distribution. We decide that we will use the skew of the sample as a
    test statistic, and we will consider a p-value of 0.05 to be statistically
    significant.

    >>> import numpy as np
    >>> from scipy import stats
    >>> def statistic(x, axis):
    ...     return stats.skew(x, axis)

    After collecting our data, we calculate the observed value of the test
    statistic.

    >>> rng = np.random.default_rng()
    >>> x = stats.skewnorm.rvs(a=1, size=50, random_state=rng)
    >>> statistic(x, axis=0)
    0.12457412450240658

    To determine the probability of observing such an extreme value of the
    skewness by chance if the sample were drawn from the normal distribution,
    we can perform a Monte Carlo hypothesis test. The test will draw many
    samples at random from their normal distribution, calculate the skewness
    of each sample, and compare our original skewness against this
    distribution to determine an approximate p-value.

    >>> from scipy.stats import monte_carlo_test
    >>> # because our statistic is vectorized, we pass `vectorized=True`
    >>> rvs = lambda size: stats.norm.rvs(size=size, random_state=rng)
    >>> res = monte_carlo_test(x, rvs, statistic, vectorized=True)
    >>> print(res.statistic)
    0.12457412450240658
    >>> print(res.pvalue)
    0.7012

    The probability of obtaining a test statistic less than or equal to the
    observed value under the null hypothesis is ~70%. This is greater than
    our chosen threshold of 5%, so we cannot consider this to be significant
    evidence against the null hypothesis.

    Note that this p-value essentially matches that of
    args = _monte_carlo_test_iv(data, rvs, statistic, vectorized,
                                n_resamples, batch, alternative, axis)
    (data, rvs, statistic, vectorized, n_resamples,
     batch, alternative, axis, dtype, xp) = args


    # 调用 _monte_carlo_test_iv 函数，并传递所需参数
    # 解包返回的元组 args，获取各个参数的值
    (data, rvs, statistic, vectorized, n_resamples,
     batch, alternative, axis, dtype, xp) = args


    # 有些统计函数可能返回普通的浮点数；确保它们至少是 NumPy 浮点数
    # 将统计量 observed 转换为 NumPy 数组，确保其至少是一个浮点数
    observed = xp.asarray(statistic(*data, axis=-1))
    observed = observed[()] if observed.ndim == 0 else observed


    # 获取每个样本的观测数，并存储在列表 n_observations 中
    n_observations = [sample.shape[-1] for sample in data]
    batch_nominal = batch or n_resamples
    null_distribution = []


    # 循环生成空分布的抽样数据
    for k in range(0, n_resamples, batch_nominal):
        batch_actual = min(batch_nominal, n_resamples - k)
        # 对每个样本生成 resamples，以用于生成空分布
        resamples = [rvs_i(size=(batch_actual, n_observations_i))
                     for rvs_i, n_observations_i in zip(rvs, n_observations)]
        null_distribution.append(statistic(*resamples, axis=-1))
    # 合并所有生成的空分布样本
    null_distribution = xp.concat(null_distribution)
    # 调整空分布的形状以匹配 observed 的维度
    null_distribution = xp.reshape(null_distribution, [-1] + [1]*observed.ndim)


    # 相对误差 tolerance，用于检测在空分布中数值上相等但理论上相等的值
    eps = (0 if not xp.isdtype(observed.dtype, ('real floating'))
           else xp.finfo(observed.dtype).eps*100)
    gamma = xp.abs(eps * observed)


    def less(null_distribution, observed):
        # 比较空分布中的值是否小于等于观察到的值加上 gamma
        cmps = null_distribution <= observed + gamma
        cmps = xp.asarray(cmps, dtype=dtype)
        # 计算 p 值
        pvalues = (xp.sum(cmps, axis=0, dtype=dtype) + 1.) / (n_resamples + 1.)
        return pvalues


    def greater(null_distribution, observed):
        # 比较空分布中的值是否大于等于观察到的值减去 gamma
        cmps = null_distribution >= observed - gamma
        cmps = xp.asarray(cmps, dtype=dtype)
        # 计算 p 值
        pvalues = (xp.sum(cmps, axis=0, dtype=dtype) + 1.) / (n_resamples + 1.)
        return pvalues


    def two_sided(null_distribution, observed):
        # 计算双尾检验的 p 值
        pvalues_less = less(null_distribution, observed)
        pvalues_greater = greater(null_distribution, observed)
        pvalues = xp_minimum(pvalues_less, pvalues_greater) * 2
        return pvalues
    # 创建一个字典，用于将字符串类型的 alternative 映射到对应的函数对象
    compare = {"less": less,
               "greater": greater,
               "two-sided": two_sided}

    # 使用 alternative 参数在 compare 字典中查找对应的函数并执行，计算假设检验的 p 值
    pvalues = compare[alternative](null_distribution, observed)
    
    # 对 p 值进行修剪，确保其在 [0, 1] 的范围内，调用 xp_clip 函数进行处理
    pvalues = xp_clip(pvalues, 0., 1., xp=xp)

    # 返回 MonteCarloTestResult 类的实例，封装了观察到的值、修剪后的 p 值和原始的 null_distribution
    return MonteCarloTestResult(observed, pvalues, null_distribution)
@dataclass
class PowerResult:
    """`scipy.stats.power` 返回的结果对象。

    Attributes
    ----------
    power : float or ndarray
        估计的功效值。
    pvalues : float or ndarray
        模拟得到的 p 值。
    """
    power: float | np.ndarray
    pvalues: float | np.ndarray


def _wrap_kwargs(fun):
    """包装可调用对象，使其接受任意关键字参数并忽略未使用的参数。"""

    try:
        keys = set(inspect.signature(fun).parameters.keys())
    except ValueError:
        # NumPy 生成器方法无法被检查
        keys = {'size'}

    # 设置 keys=keys/fun=fun 避免延迟绑定陷阱
    def wrapped_rvs_i(*args, keys=keys, fun=fun, **all_kwargs):
        kwargs = {key: val for key, val in all_kwargs.items()
                  if key in keys}
        return fun(*args, **kwargs)
    return wrapped_rvs_i


def _power_iv(rvs, test, n_observations, significance, vectorized,
              n_resamples, batch, kwargs):
    """对 `monte_carlo_test` 的输入进行验证和处理。"""

    if vectorized not in {True, False, None}:
        raise ValueError("`vectorized` 必须是 `True`、`False` 或 `None`。")

    if not isinstance(rvs, Sequence):
        rvs = (rvs,)
        n_observations = (n_observations,)
    for rvs_i in rvs:
        if not callable(rvs_i):
            raise TypeError("`rvs` 必须是可调用对象或可调用对象序列。")

    if not len(rvs) == len(n_observations):
        message = ("如果 `rvs` 是一个序列，`len(rvs)` "
                   "必须等于 `len(n_observations)`。")
        raise ValueError(message)

    significance = np.asarray(significance)[()]
    if (not np.issubdtype(significance.dtype, np.floating)
            or np.min(significance) < 0 or np.max(significance) > 1):
        raise ValueError("`significance` 必须包含介于 0 和 1 之间的浮点数。")

    kwargs = dict() if kwargs is None else kwargs
    if not isinstance(kwargs, dict):
        raise TypeError("`kwargs` 必须是一个将关键字映射到数组的字典。")

    vals = kwargs.values()
    keys = kwargs.keys()

    # 包装可调用对象，忽略未使用的关键字参数
    wrapped_rvs = [_wrap_kwargs(rvs_i) for rvs_i in rvs]

    # 广播并展平 nobs/kwarg 组合。最终，`nobs` 和 `vals` 的形状为 (# 组合数, 变量数)
    tmp = np.asarray(np.broadcast_arrays(*n_observations, *vals))
    shape = tmp.shape
    if tmp.ndim == 1:
        tmp = tmp[np.newaxis, :]
    else:
        tmp = tmp.reshape((shape[0], -1)).T
    nobs, vals = tmp[:, :len(rvs)], tmp[:, len(rvs):]
    nobs = nobs.astype(int)

    if not callable(test):
        raise TypeError("`test` 必须是可调用对象。")

    if vectorized is None:
        vectorized = 'axis' in inspect.signature(test).parameters

    if not vectorized:
        test_vectorized = _vectorize_statistic(test)
    else:
        test_vectorized = test
    # 包装 `test` 函数，忽略未使用的 kwargs
    test_vectorized = _wrap_kwargs(test_vectorized)
    # 将 `n_resamples` 转换为整数
    n_resamples_int = int(n_resamples)
    # 检查 `n_resamples` 是否与其整数转换后的值相等且大于零，若不是则引发值错误异常
    if n_resamples != n_resamples_int or n_resamples_int <= 0:
        raise ValueError("`n_resamples` must be a positive integer.")
    
    # 如果 `batch` 是 None，则将 `batch_iv` 设为 None
    if batch is None:
        batch_iv = batch
    # 否则，将 `batch` 转换为整数
    else:
        batch_iv = int(batch)
        # 检查 `batch` 是否与其整数转换后的值相等且大于零，若不是则引发值错误异常
        if batch != batch_iv or batch_iv <= 0:
            raise ValueError("`batch` must be a positive integer or None.")
    
    # 返回元组包含多个值：wrapped_rvs, test_vectorized, nobs, significance, vectorized,
    # n_resamples_int, batch_iv, vals, keys, shape[1:]
    return (wrapped_rvs, test_vectorized, nobs, significance, vectorized,
            n_resamples_int, batch_iv, vals, keys, shape[1:])
# 定义函数 power，用于模拟在备择假设下假设检验的功效（power）
def power(test, rvs, n_observations, *, significance=0.01, vectorized=None,
          n_resamples=10000, batch=None, kwargs=None):
    """
    Simulate the power of a hypothesis test under an alternative hypothesis.

    Parameters
    ----------
    test : callable
        假设检验函数，用于模拟其在备择假设下的功效。
        `test` 必须是一个可调用对象，接受一个样本（例如 `test(sample)`）或 `len(rvs)` 个单独的样本
        （例如 `test(samples1, sample2)`，如果 `rvs` 包含两个可调用对象，并且 `n_observations` 包含两个值），
        并返回检验的 p 值。
        如果 `vectorized` 设为 `True`，`test` 还必须接受关键字参数 `axis`，并且在给定的 `axis` 上进行向量化以执行检验。
        任何带有 `axis` 参数并返回带有 `pvalue` 属性的 `scipy.stats` 中的可调用对象也是可接受的。
    rvs : callable or tuple of callables
        一个可调用对象或可调用对象序列，用于在备择假设下生成随机变量。
        `rvs` 的每个元素必须接受关键字参数 `size`（例如 `rvs(size=(m, n))`），并返回相应形状的 N 维数组。
        如果 `rvs` 是一个序列，则其长度必须与 `n_observations` 的元素数相匹配，即 `len(rvs) == len(n_observations)`。
        如果 `rvs` 是单个可调用对象，则 `n_observations` 被视为单个元素。
    n_observations : tuple of ints or tuple of integer arrays
        如果是整数序列，则为要传递给 `test` 的每个样本的大小。
        如果是整数数组序列，则对应于每组相应样本大小进行功效模拟。参见示例。
    significance : float or array_like of floats, default: 0.01
        显著性阈值；即，将检验的 p 值视为针对零假设的证据时的阈值。
        如果是数组，则对每个显著性阈值进行功效模拟。
    kwargs : dict, optional
        传递给 `rvs` 和/或 `test` 可调用对象的关键字参数。
        使用内省确定每个可调用对象可以传递的关键字参数。
        每个关键字对应的值必须是一个数组。
        数组必须能够与彼此及 `n_observations` 中的每个数组进行广播。
        对于每组相应的样本大小和参数进行功效模拟。参见示例。
    """
    # 函数体内的具体实现逻辑未提供，因此此处不添加更多注释
    pass
    # `vectorized` 参数，指定是否向量化执行测试
    # 如果 `vectorized` 设置为 ``False``，`test` 函数将不会接收 `axis` 关键字参数，
    # 预期仅对1维样本执行测试。
    # 如果设置为 ``True``，`test` 将会接收 `axis` 关键字参数，
    # 预期在传入N维样本数组时沿着 `axis` 执行测试。
    # 如果设为 ``None``（默认），如果 `axis` 是 `test` 的一个参数，`vectorized` 将被设为 ``True``。
    # 使用向量化测试通常可以减少计算时间。
    vectorized: bool, optional

    # `n_resamples` 参数，用于模拟执行的重采样次数
    # 对每个 `rvs` 中的可调用对象进行的采样次数。
    # 同时也是在备择假设下执行的测试次数，以近似计算功效。
    n_resamples: int, default: 10000

    # `batch` 参数，每次调用 `test` 函数处理的样本数
    # 内存使用量与 `batch` 和最大样本大小的乘积成正比。
    # 默认为 ``None``，此时 `batch` 等于 `n_resamples`。
    batch: int, optional

    # 返回结果类型为 `PowerResult` 对象
    # 其中包含以下属性：

    # `power` 属性，估计的备择假设下的功效
    power: float or ndarray

    # `pvalues` 属性，备择假设下观察到的 p 值数组
    pvalues: ndarray

    # 笔记
    # 功效的模拟如下所示：

    # - 从 `n_observations` 指定的大小中抽取许多随机样本（或样本集），
    #   根据 `rvs` 指定的备择假设。
    # - 对每个样本（或样本集），根据 `test` 函数计算 p 值。
    #   这些 p 值记录在结果对象的 ``pvalues`` 属性中。
    # - 计算 p 值小于 `significance` 水平的比例。
    #   这个比例记录在结果对象的 ``power`` 属性中。

    # 假设 `significance` 是一个形状为 ``shape1`` 的数组，
    # `kwargs` 和 `n_observations` 的元素可以在 `shape2` 中互相广播，
    # 而 `test` 返回一个形状为 ``shape3`` 的 p 值数组。
    # 那么结果对象的 ``power`` 属性将是形状为 ``shape1 + shape2 + shape3`` 的数组，
    # 而 ``pvalues`` 属性将是形状为 ``shape2 + shape3 + (n_resamples,)`` 的数组。
    # 调用 stats 模块中的 power 函数，计算 t-test 的统计功效
    >>> res = stats.power(test, rvs, n_observations, significance=0.05)
    # 访问结果对象的 power 属性，显示统计功效的数值
    >>> res.power
    # 输出功效值为 0.6116

    # 当样本大小分别为 10 和 12 时，在显著性水平为 0.05 的条件下，t-test 的统计功效约为60%。我们可以通过传递样本大小数组来研究样本大小对统计功效的影响。

    # 导入 matplotlib.pyplot 库
    >>> import matplotlib.pyplot as plt
    # 创建一个包含整数范围的数组 nobs_x，作为样本大小的取值
    >>> nobs_x = np.arange(5, 21)
    # 将 nobs_x 赋值给 nobs_y，以备后续使用
    >>> nobs_y = nobs_x
    # 将样本大小数组 n_observations 设为包含 nobs_x 和 nobs_y 的元组
    >>> n_observations = (nobs_x, nobs_y)
    # 重新计算 t-test 的统计功效，使用更新后的 n_observations 和显著性水平为 0.05
    >>> res = stats.power(test, rvs, n_observations, significance=0.05)
    # 创建一个 subplot 对象 ax
    >>> ax = plt.subplot()
    # 在 ax 上绘制 nobs_x 对应的统计功效曲线
    >>> ax.plot(nobs_x, res.power)
    # 设置 x 轴标签为 'Sample Size'
    >>> ax.set_xlabel('Sample Size')
    # 设置 y 轴标签为 'Simulated Power'
    >>> ax.set_ylabel('Simulated Power')
    # 设置图表标题为 'Simulated Power of `ttest_ind` with Equal Sample Sizes'
    >>> ax.set_title('Simulated Power of `ttest_ind` with Equal Sample Sizes')
    # 显示图表
    >>> plt.show()

    # 或者，我们可以研究效应大小对统计功效的影响。
    # 在这种情况下，效应大小是第二个样本所在分布的位置。

    # 将 n_observations 设为包含样本大小 10 和 12 的元组
    >>> n_observations = (10, 12)
    # 创建一个包含 0 到 1 之间 20 个等间距值的数组 loc
    >>> loc = np.linspace(0, 1, 20)
    # 定义一个 lambda 函数 rvs2，用于生成指定位置 loc 的正态分布随机变量
    >>> rvs2 = lambda size, loc: rng.normal(loc=loc, size=size)
    # 将 rvs1 和 rvs2 组成的元组赋值给 rvs
    >>> rvs = (rvs1, rvs2)
    # 重新计算 t-test 的统计功效，使用更新后的 n_observations、显著性水平为 0.05，和 loc 作为参数
    >>> res = stats.power(test, rvs, n_observations, significance=0.05,
    ...                   kwargs={'loc': loc})
    # 创建一个 subplot 对象 ax
    >>> ax = plt.subplot()
    # 在 ax 上绘制 loc 对应的统计功效曲线
    >>> ax.plot(loc, res.power)
    # 设置 x 轴标签为 'Effect Size'
    >>> ax.set_xlabel('Effect Size')
    # 设置 y 轴标签为 'Simulated Power'
    >>> ax.set_ylabel('Simulated Power')
    # 设置图表标题为 'Simulated Power of `ttest_ind`, Varying Effect Size'
    >>> ax.set_title('Simulated Power of `ttest_ind`, Varying Effect Size')
    # 显示图表
    >>> plt.show()

    # 我们还可以使用 `power` 函数来估计测试的第一类错误率（也称为“大小”），并评估其是否与名义水平匹配。
    # 例如，`jarque_bera` 的零假设是样本来自具有与正态分布相同的偏度和峰度的分布。为了估计第一类错误率，我们可以将零假设视为真实的“替代”假设并计算其功效。

    # 将 test 设为 stats 模块中的 jarque_bera 测试
    >>> test = stats.jarque_bera
    # 将 n_observations 设为样本大小为 10
    >>> n_observations = 10
    # 将 rvs 设为 rng.normal 函数
    >>> rvs = rng.normal
    # 创建一个包含 0.0001 到 0.1 之间 1000 个等间距值的数组 significance
    >>> significance = np.linspace(0.0001, 0.1, 1000)
    # 重新计算 jarque_bera 测试的统计功效，使用更新后的 n_observations 和 significance
    >>> res = stats.power(test, rvs, n_observations, significance=significance)
    # 将功效值赋值给 size
    >>> size = res.power

    # 如下图所示，对于这样一个小样本，测试的第一类错误率远低于名义水平，正如其文档中所述。

    # 创建一个 subplot 对象 ax
    >>> ax = plt.subplot()
    # 在 ax 上绘制 significance 对应的 size 曲线
    >>> ax.plot(significance, size)
    # 绘制一条理想测试线，从 (0, 0) 到 (0.1, 0.1)
    >>> ax.plot([0, 0.1], [0, 0.1], '--')
    # 设置 x 轴标签为 'nominal significance level'
    >>> ax.set_xlabel('nominal significance level')
    # 设置 y 轴标签为 'estimated test size (Type I error rate)'
    >>> ax.set_ylabel('estimated test size (Type I error rate)')
    # 设置图表标题为 'Estimated test size vs nominal significance level'
    >>> ax.set_title('Estimated test size vs nominal significance level')
    # 设置图表纵横比为相等的正方形
    >>> ax.set_aspect('equal', 'box')
    # 设置图例，显示 `ttest_1samp` 和理想测试的说明
    >>> ax.legend(('`ttest_1samp`', 'ideal test'))
    # 显示图表
    >>> plt.show()

    # 由于这样一个保守的测试，与某些替代方案相比，其功效相当低。
    alternative that the sample was drawn from the Laplace distribution may not
    be much greater than the Type I error rate.

    >>> rvs = rng.laplace
    >>> significance = np.linspace(0.0001, 0.1, 1000)
    >>> res = stats.power(test, rvs, n_observations, significance=0.05)
    >>> print(res.power)
    0.0587

    This is not a mistake in SciPy's implementation; it is simply due to the fact
    that the null distribution of the test statistic is derived under the assumption
    that the sample size is large (i.e. approaches infinity), and this asymptotic
    approximation is not accurate for small samples. In such cases, resampling
    and Monte Carlo methods (e.g. `permutation_test`, `goodness_of_fit`,
    `monte_carlo_test`) may be more appropriate.

"""
# 调用内部函数 _power_iv 处理输入的参数和设置，并返回元组
tmp = _power_iv(rvs, test, n_observations, significance,
                vectorized, n_resamples, batch, kwargs)
# 将返回的元组解包并赋值给各个变量
(rvs, test, nobs, significance,
 vectorized, n_resamples, batch, args, kwds, shape)= tmp

# 根据情况确定 batch_nominal 的值
batch_nominal = batch or n_resamples
# 初始化一个空列表，用来存储不同 nobs/kwargs 组合的 p-values 结果
pvalues = []  # results of various nobs/kwargs combinations

# 遍历 nobs 和 args，使用 zip 函数同时迭代它们
for nobs_i, args_i in zip(nobs, args):
    # 根据 kwds 和 args_i 创建一个字典
    kwargs_i = dict(zip(kwds, args_i))
    # 初始化一个空列表，用来存储固定 nobs/kwargs 组合下的 p-values 结果
    pvalues_i = []  # results of batches; fixed nobs/kwargs combination

    # 使用循环遍历 0 到 n_resamples 的范围，步长为 batch_nominal
    for k in range(0, n_resamples, batch_nominal):
        # 计算每个批次的实际大小，即 min(batch_nominal, n_resamples - k)
        batch_actual = min(batch_nominal, n_resamples - k)
        # 使用列表推导式生成一个包含多个样本的 resamples 列表
        resamples = [rvs_j(size=(batch_actual, nobs_ij), **kwargs_i)
                     for rvs_j, nobs_ij in zip(rvs, nobs_i)]
        # 对生成的样本进行测试，得到测试结果 res
        res = test(*resamples, **kwargs_i, axis=-1)
        # 获取 res 的 p-value 属性，如果不存在，则直接使用 res
        p = getattr(res, 'pvalue', res)
        # 将计算得到的 p 值添加到 pvalues_i 列表中
        pvalues_i.append(p)

    # 将各批次的结果连接起来
    pvalues_i = np.concatenate(pvalues_i, axis=-1)
    # 将 pvalues_i 添加到 pvalues 列表中
    pvalues.append(pvalues_i)

# 计算 pvalues 的形状
shape += pvalues_i.shape[:-1]
# 将 pvalues 中的各个元素连接起来，形成最终的 pvalues 数组
pvalues = np.concatenate(pvalues, axis=0)

# 如果 significance 的维度大于 0，则进行扩展操作
if significance.ndim > 0:
    # 生成一个新的维度元组，使得 significance 的维度与 pvalues 的维度相匹配
    newdims = tuple(range(significance.ndim, pvalues.ndim + significance.ndim))
    significance = np.expand_dims(significance, newdims)

# 计算各个 p-value 是否小于 significance 的均值，得到 power
powers = np.mean(pvalues < significance, axis=-1)

# 返回 PowerResult 对象，包括 power 和 pvalues
return PowerResult(power=powers, pvalues=pvalues)
@dataclass
class PermutationTestResult:
    """Result object returned by `scipy.stats.permutation_test`.

    Attributes
    ----------
    statistic : float or ndarray
        The observed test statistic of the data.
    pvalue : float or ndarray
        The p-value for the given alternative.
    null_distribution : ndarray
        The values of the test statistic generated under the null
        hypothesis.
    """
    statistic: float | np.ndarray
    pvalue: float | np.ndarray
    null_distribution: np.ndarray


def _all_partitions_concatenated(ns):
    """
    Generate all partitions of indices of groups of given sizes, concatenated

    `ns` is an iterable of ints.
    """
    # Define a function to generate all partitions of a set `z` into two parts of size `n`
    def all_partitions(z, n):
        for c in combinations(z, n):
            x0 = set(c)
            x1 = z - x0
            yield [x0, x1]

    # Define a recursive function to generate all partitions of set `z` based on sizes in `ns`
    def all_partitions_n(z, ns):
        if len(ns) == 0:
            yield [z]
            return
        for c in all_partitions(z, ns[0]):
            for d in all_partitions_n(c[1], ns[1:]):
                yield c[0:1] + d

    z = set(range(np.sum(ns)))  # Create a set `z` containing indices up to the sum of `ns`
    # Iterate over all possible partitions of `z` according to `ns` and concatenate them into arrays
    for partitioning in all_partitions_n(z, ns[:]):
        x = np.concatenate([list(partition)
                            for partition in partitioning]).astype(int)
        yield x


def _batch_generator(iterable, batch):
    """A generator that yields batches of elements from an iterable"""
    iterator = iter(iterable)
    if batch <= 0:
        raise ValueError("`batch` must be positive.")
    z = [item for i, item in zip(range(batch), iterator)]  # Collect `batch` items from the iterator
    while z:  # Continue yielding batches until the iterator is exhausted
        yield z
        z = [item for i, item in zip(range(batch), iterator)]  # Collect next `batch` items


def _pairings_permutations_gen(n_permutations, n_samples, n_obs_sample, batch,
                               random_state):
    # Returns a generator that yields arrays of size
    # `(batch, n_samples, n_obs_sample)`.
    # Each row is an independent permutation of indices 0 to `n_obs_sample`.
    batch = min(batch, n_permutations)  # Ensure `batch` does not exceed `n_permutations`

    if hasattr(random_state, 'permuted'):  # Check if `random_state` has attribute `permuted`
        # Define generator for permuted indices using `random_state`
        def batched_perm_generator():
            indices = np.arange(n_obs_sample)
            indices = np.tile(indices, (batch, n_samples, 1))  # Tile indices for batches
            for k in range(0, n_permutations, batch):
                batch_actual = min(batch, n_permutations-k)
                # Generate permuted indices without modifying original, using `permuted`
                permuted_indices = random_state.permuted(indices, axis=-1)
                yield permuted_indices[:batch_actual]
    else:  # If `random_state` does not have `permuted` attribute
        # Define generator for permuted indices using `random_state`
        def batched_perm_generator():
            for k in range(0, n_permutations, batch):
                batch_actual = min(batch, n_permutations-k)
                size = (batch_actual, n_samples, n_obs_sample)
                x = random_state.random(size=size)  # Generate random data
                yield np.argsort(x, axis=-1)[:batch_actual]  # Return sorted indices

    return batched_perm_generator()
# 计算独立样本检验的零分布。
def _calculate_null_both(data, statistic, n_permutations, batch,
                         random_state=None):
    """
    Calculate null distribution for independent sample tests.
    """
    # 获取数据中样本的数量
    n_samples = len(data)

    # 计算置换的数量，即数据分割成这些大小的样本的不同分区
    n_obs_i = [sample.shape[-1] for sample in data]  # 每个样本的观测值数量
    n_obs_ic = np.cumsum(n_obs_i)
    n_obs = n_obs_ic[-1]  # 总观测值数量
    n_max = np.prod([comb(n_obs_ic[i], n_obs_ic[i-1])
                     for i in range(n_samples-1, 0, -1)])

    # perm_generator 是一个迭代器，产生从0到n_obs的索引置换
    # 我们将样本连接起来，使用这些索引置换数据，然后再次分割样本
    if n_permutations >= n_max:
        exact_test = True
        n_permutations = n_max
        perm_generator = _all_partitions_concatenated(n_obs_i)
    else:
        exact_test = False
        # 如果将来添加了轴切片独立置换的功能，RandomState.permutation 或 Generator.permutation 都不可以独立地置换轴切片
        # 在这种情况下，应当一次性生成所需大小的批次
        perm_generator = (random_state.permutation(n_obs)
                          for i in range(n_permutations))

    batch = batch or int(n_permutations)
    null_distribution = []

    # 首先，连接所有样本。分批次使用 `perm_generator` 生成的索引置换样本，
    # 将它们分割成原始大小的新样本，计算每个批次的统计量，并将这些统计值添加到零分布中。
    data = np.concatenate(data, axis=-1)
    for indices in _batch_generator(perm_generator, batch=batch):
        indices = np.array(indices)

        # `indices` 是二维的：每一行是索引的一个置换。
        # 我们使用它来沿着数据的最后一个轴进行索引，该轴对应于观测值。
        # 索引完成后，`data_batch`的倒数第二个轴对应于置换，最后一个轴对应于观测值。
        data_batch = data[..., indices]

        # 将置换轴移到前面：我们将沿着这个零轴连接批次统计值的列表，以形成零分布。
        data_batch = np.moveaxis(data_batch, -2, 0)
        data_batch = np.split(data_batch, n_obs_ic[:-1], axis=-1)
        null_distribution.append(statistic(*data_batch, axis=-1))
    null_distribution = np.concatenate(null_distribution, axis=0)

    return null_distribution, n_permutations, exact_test


def _calculate_null_pairings(data, statistic, n_permutations, batch,
                             random_state=None):
    """
    Calculate null distribution for association tests.
    """
    # 获取数据中样本的数量
    n_samples = len(data)
    # 计算排列数（每个样本的观测值的阶乘排列数）
    n_obs_sample = data[0].shape[-1]  # 每个样本的观测数；每个样本相同
    n_max = factorial(n_obs_sample)**n_samples

    # `perm_generator` 是一个迭代器，为每个样本产生从0到n_obs_sample的索引排列列表
    if n_permutations >= n_max:
        exact_test = True
        n_permutations = n_max
        batch = batch or int(n_permutations)
        # 生成所有索引排列集合的笛卡尔积
        perm_generator = product(*(permutations(range(n_obs_sample))
                                   for i in range(n_samples)))
        batched_perm_generator = _batch_generator(perm_generator, batch=batch)
    else:
        exact_test = False
        batch = batch or int(n_permutations)
        # 为每个样本分别生成随机的索引排列
        # 如果RandomState/Generator.permutation可以分别对每个轴切片进行排列，会更好。
        args = n_permutations, n_samples, n_obs_sample, batch, random_state
        batched_perm_generator = _pairings_permutations_gen(*args)

    null_distribution = []

    for indices in batched_perm_generator:
        indices = np.array(indices)

        # `indices` 是三维的：第一个轴是排列，接下来是样本，最后是观测值。交换前两个轴，
        # 使得第一个轴对应样本，就像`data`一样。
        indices = np.swapaxes(indices, 0, 1)

        # 当完成后，`data_batch` 将会是长度为`n_samples`的列表。
        # 每个元素将是一个样本的随机排列的批次。
        # 每个批次的第一个轴对应排列，最后一个轴对应观测值。（这样可以很容易地传递给`statistic`。）
        data_batch = [None]*n_samples
        for i in range(n_samples):
            data_batch[i] = data[i][..., indices[i]]
            data_batch[i] = np.moveaxis(data_batch[i], -2, 0)

        # 计算每个排列的统计量，并将结果附加到null_distribution中。
        null_distribution.append(statistic(*data_batch, axis=-1))
    # 将所有统计量连接起来形成一个单一的null分布数组。
    null_distribution = np.concatenate(null_distribution, axis=0)

    # 返回null分布、排列数和是否为精确检验的标志。
    return null_distribution, n_permutations, exact_test
def _calculate_null_samples(data, statistic, n_permutations, batch,
                            random_state=None):
    """
    Calculate null distribution for paired-sample tests.
    """
    # 计算数据样本数量
    n_samples = len(data)

    # 根据惯例，当数据只有一个样本时，使用"samples"置换类型意味着翻转观察值的符号。
    # 为了实现这一点，添加一个副本样本，即原始样本的负数。
    if n_samples == 1:
        data = [data[0], -data[0]]

    # "samples"置换策略与"pairings"策略相同，只是样本和观察的角色被交换。
    # 因此交换这些轴，然后使用"pairings"策略的函数来完成所有工作！
    data = np.swapaxes(data, 0, -1)

    # （当然，用户的统计量并不知道我们在这里做了什么，所以我们需要传递它期望的内容。）
    def statistic_wrapped(*data, axis):
        data = np.swapaxes(data, 0, -1)
        if n_samples == 1:
            data = data[0:1]
        return statistic(*data, axis=axis)

    # 调用计算"pairings"策略的函数来计算空分布
    return _calculate_null_pairings(data, statistic_wrapped, n_permutations,
                                    batch, random_state)


def _permutation_test_iv(data, statistic, permutation_type, vectorized,
                         n_resamples, batch, alternative, axis, random_state):
    """Input validation for `permutation_test`."""

    # 将轴转换为整数
    axis_int = int(axis)
    if axis != axis_int:
        raise ValueError("`axis` must be an integer.")

    # 验证置换类型是否合法
    permutation_types = {'samples', 'pairings', 'independent'}
    permutation_type = permutation_type.lower()
    if permutation_type not in permutation_types:
        raise ValueError(f"`permutation_type` must be in {permutation_types}.")

    # 验证vectorized参数的合法性
    if vectorized not in {True, False, None}:
        raise ValueError("`vectorized` must be `True`, `False`, or `None`.")

    # 如果vectorized为None，则根据统计函数的参数签名判断是否需要向量化
    if vectorized is None:
        vectorized = 'axis' in inspect.signature(statistic).parameters

    # 如果不需要向量化，则使用内部函数_vectorize_statistic进行向量化处理
    if not vectorized:
        statistic = _vectorize_statistic(statistic)

    # 检查数据是否是包含至少两个样本的元组
    message = "`data` must be a tuple containing at least two samples"
    try:
        if len(data) < 2 and permutation_type == 'independent':
            raise ValueError(message)
    except TypeError:
        raise TypeError(message)

    # 广播数据数组，使其符合指定的轴
    data = _broadcast_arrays(data, axis)
    data_iv = []
    for sample in data:
        # 将每个样本转换为至少一维，并验证其观察值数量是否大于1
        sample = np.atleast_1d(sample)
        if sample.shape[axis] <= 1:
            raise ValueError("each sample in `data` must contain two or more "
                             "observations along `axis`.")
        # 将样本轴移动到指定轴的最后位置
        sample = np.moveaxis(sample, axis_int, -1)
        data_iv.append(sample)

    # 将n_resamples转换为整数，并验证其为正整数
    n_resamples_int = (int(n_resamples) if not np.isinf(n_resamples)
                       else np.inf)
    if n_resamples != n_resamples_int or n_resamples_int <= 0:
        raise ValueError("`n_resamples` must be a positive integer.")

    # 如果batch为None，则使用原始值，否则保持不变
    if batch is None:
        batch_iv = batch
    else:
        # 将批处理大小转换为整数
        batch_iv = int(batch)
        # 检查转换后的整数是否与原始值不同，或者是否小于等于零，若是则引发异常
        if batch != batch_iv or batch_iv <= 0:
            raise ValueError("`batch` must be a positive integer or None.")

    # 定义可接受的备择假设类型
    alternatives = {'two-sided', 'greater', 'less'}
    # 将备择假设类型转换为小写
    alternative = alternative.lower()
    # 检查转换后的备择假设类型是否在允许的备择假设集合中，若不在则引发异常
    if alternative not in alternatives:
        raise ValueError(f"`alternative` must be in {alternatives}")

    # 检查随机数发生器状态，确保其为有效的随机数生成器
    random_state = check_random_state(random_state)

    # 返回包含多个参数的元组
    return (data_iv, statistic, permutation_type, vectorized, n_resamples_int,
            batch_iv, alternative, axis_int, random_state)
# 定义一个函数执行置换检验，用于计算给定数据的统计量的置换检验的 p 值
def permutation_test(data, statistic, *, permutation_type='independent',
                     vectorized=None, n_resamples=9999, batch=None,
                     alternative="two-sided", axis=0, random_state=None):
    r"""
    Performs a permutation test of a given statistic on provided data.

    For independent sample statistics, the null hypothesis is that the data are
    randomly sampled from the same distribution.
    For paired sample statistics, two null hypothesis can be tested:
    that the data are paired at random or that the data are assigned to samples
    at random.

    Parameters
    ----------
    data : iterable of array-like
        Contains the samples, each of which is an array of observations.
        Dimensions of sample arrays must be compatible for broadcasting except
        along `axis`.
    statistic : callable
        Statistic for which the p-value of the hypothesis test is to be
        calculated. `statistic` must be a callable that accepts samples
        as separate arguments (e.g. ``statistic(*data)``) and returns the
        resulting statistic.
        If `vectorized` is set ``True``, `statistic` must also accept a keyword
        argument `axis` and be vectorized to compute the statistic along the
        provided `axis` of the sample arrays.
    permutation_type : {'independent', 'samples', 'pairings'}, optional
        The type of permutations to be performed, in accordance with the
        null hypothesis. The first two permutation types are for paired sample
        statistics, in which all samples contain the same number of
        observations and observations with corresponding indices along `axis`
        are considered to be paired; the third is for independent sample
        statistics.

        - ``'samples'`` : observations are assigned to different samples
          but remain paired with the same observations from other samples.
          This permutation type is appropriate for paired sample hypothesis
          tests such as the Wilcoxon signed-rank test and the paired t-test.
        - ``'pairings'`` : observations are paired with different observations,
          but they remain within the same sample. This permutation type is
          appropriate for association/correlation tests with statistics such
          as Spearman's :math:`\rho`, Kendall's :math:`\tau`, and Pearson's
          :math:`r`.
        - ``'independent'`` (default) : observations are assigned to different
          samples. Samples may contain different numbers of observations. This
          permutation type is appropriate for independent sample hypothesis
          tests such as the Mann-Whitney :math:`U` test and the independent
          sample t-test.

          Please see the Notes section below for more detailed descriptions
          of the permutation types.
    # `vectorized`参数控制是否向量化计算统计量
    vectorized : bool, optional
        # 如果`vectorized`设置为`False`，则`statistic`函数不会传递关键字参数`axis`，
        # 并且预期只计算1D样本的统计量。
        # 如果设置为`True`，则`statistic`函数将传递关键字参数`axis`，
        # 预期在传递ND样本数组时沿着`axis`计算统计量。
        # 如果为`None`（默认），如果`statistic`函数有`axis`参数，则`vectorized`将被设置为`True`。
        # 使用向量化的统计量通常可以减少计算时间。

    # `n_resamples`参数指定用于近似空分布的随机排列（重采样）次数
    n_resamples : int or np.inf, default: 9999
        # 用于近似空分布的随机排列（重采样）的次数。
        # 如果大于或等于不同排列的数量，则将计算精确的空分布。
        # 注意，随着样本大小的增加，不同排列的数量增长非常迅速，
        # 因此仅对非常小的数据集可行的是精确测试。

    # `batch`参数指定每次调用`statistic`函数处理的排列数量
    batch : int, optional
        # 每次调用`statistic`函数处理的排列数量。
        # 内存使用量为O(`batch` * ``n``)，其中``n``是所有样本的总大小，
        # 不考虑`vectorized`的值。默认为``None``，此时``batch``等于排列的数量。

    # `alternative`参数指定计算p值时的备择假设
    alternative : {'two-sided', 'less', 'greater'}, optional
        # 计算p值时的备择假设。
        # 对于每种备择假设，p值的定义如下：
        # - ``'greater'`` : 空分布中大于或等于观察到的统计量值的百分比。
        # - ``'less'`` : 空分布中小于或等于观察到的统计量值的百分比。
        # - ``'two-sided'``（默认）: 上述两个p值的较小者的两倍。
        # 注意，随机化测试的p值根据文献[2]_和[3]_中建议的保守（过度估计的）逼近计算，
        # 而不是文献[4]_中建议的无偏估计器。
        # 即，在计算随机化空分布中与观察到的统计量值一样极端的比例时，
        # 分子和分母的值都增加了一个。这种调整的解释是，观察到的统计量值始终被包括在随机化的空分布中。
        # 用于双侧p值的惯例并非普遍适用；
        # 如果有其他定义更合适，则观察到的测试统计量和空分布将被返回。
    axis : int, default: 0
        # 指定统计计算时样本的轴。如果样本的维数不同，会在较少维度的样本前添加单例维度，然后考虑轴。

    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional
        # 用于生成排列的伪随机数生成器状态。
        
        如果 `random_state` 是 ``None``（默认），则使用 `numpy.random.RandomState` 单例。
        如果 `random_state` 是一个整数，则使用一个新的 ``RandomState`` 实例，并使用 `random_state` 作为种子。
        如果 `random_state` 已经是 ``Generator`` 或 ``RandomState`` 实例，则直接使用该实例。

    Returns
    -------
    res : PermutationTestResult
        # 一个带有以下属性的对象：

        statistic : float or ndarray
            # 数据的观察测试统计量。
        pvalue : float or ndarray
            # 给定备择假设的 p 值。
        null_distribution : ndarray
            # 在零假设下生成的测试统计量值。

    Notes
    -----

    The three types of permutation tests supported by this function are
    described below.

    **Unpaired statistics** (``permutation_type='independent'``):
        # 不配对统计量的置换检验类型描述如下：

        This paragraph describes the null hypothesis for independent permutation tests,
        assuming all observations are from the same underlying distribution and are randomly assigned.

        Suppose ``data`` contains two samples; e.g. ``a, b = data``.
        When ``1 < n_resamples < binom(n, k)``, where

        * ``k`` is the number of observations in ``a``,
        * ``n`` is the total number of observations in ``a`` and ``b``,
        * ``binom(n, k)`` is the binomial coefficient (``n`` choose ``k``),

        the data are pooled (concatenated), randomly assigned to either the first
        or second sample, and the statistic is calculated. This process is
        performed repeatedly, `permutation` times, generating a distribution of the
        statistic under the null hypothesis. The statistic of the original
        data is compared to this distribution to determine the p-value.

        When ``n_resamples >= binom(n, k)``, an exact test is performed: the data
        are *partitioned* between the samples in each distinct way exactly once,
        and the exact null distribution is formed.
        Note that for a given partitioning of the data between the samples,
        only one ordering/permutation of the data *within* each sample is
        considered. For statistics that do not depend on the order of the data
        within samples, this dramatically reduces computational cost without
        affecting the shape of the null distribution (because the frequency/count
        of each value is affected by the same factor).

        For ``a = [a1, a2, a3, a4]`` and ``b = [b1, b2, b3]``, an example of this
    permutation type is ``x = [b3, a1, a2, b2]`` and ``y = [a4, b1, a3]``.
    Because only one ordering/permutation of the data *within* each sample
    is considered in an exact test, a resampling like ``x = [b3, a1, b2, a2]``
    and ``y = [a4, a3, b1]`` would *not* be considered distinct from the
    example above.

    ``permutation_type='independent'`` does not support one-sample statistics,
    but it can be applied to statistics with more than two samples. In this
    case, if ``n`` is an array of the number of observations within each
    sample, the number of distinct partitions is::

        np.prod([binom(sum(n[i:]), sum(n[i+1:])) for i in range(len(n)-1)])

    **Paired statistics, permute pairings** (``permutation_type='pairings'``):

    The null hypothesis associated with this permutation type is that
    observations within each sample are drawn from the same underlying
    distribution and that pairings with elements of other samples are
    assigned at random.

    Suppose ``data`` contains only one sample; e.g. ``a, = data``, and we
    wish to consider all possible pairings of elements of ``a`` with elements
    of a second sample, ``b``. Let ``n`` be the number of observations in
    ``a``, which must also equal the number of observations in ``b``.

    When ``1 < n_resamples < factorial(n)``, the elements of ``a`` are
    randomly permuted. The user-supplied statistic accepts one data argument,
    say ``a_perm``, and calculates the statistic considering ``a_perm`` and
    ``b``. This process is performed repeatedly, `permutation` times,
    generating a distribution of the statistic under the null hypothesis.
    The statistic of the original data is compared to this distribution to
    determine the p-value.

    When ``n_resamples >= factorial(n)``, an exact test is performed:
    ``a`` is permuted in each distinct way exactly once. Therefore, the
    `statistic` is computed for each unique pairing of samples between ``a``
    and ``b`` exactly once.

    For ``a = [a1, a2, a3]`` and ``b = [b1, b2, b3]``, an example of this
    permutation type is ``a_perm = [a3, a1, a2]`` while ``b`` is left
    in its original order.

    ``permutation_type='pairings'`` supports ``data`` containing any number
    of samples, each of which must contain the same number of observations.
    All samples provided in ``data`` are permuted *independently*. Therefore,
    if ``m`` is the number of samples and ``n`` is the number of observations
    within each sample, then the number of permutations in an exact test is::

        factorial(n)**m

    Note that if a two-sample statistic, for example, does not inherently
    depend on the order in which observations are provided - only on the
    *pairings* of observations - then only one of the two samples should be
    provided in ``data``. This dramatically reduces computational cost without
    affecting the shape of the null distribution (because the frequency/count
    # Paired statistics, permute samples (`permutation_type='samples'`):

    # 当`permutation_type='samples'`时，进行样本置换的配对统计。

    # The null hypothesis associated with this permutation type is that
    # observations within each pair are drawn from the same underlying
    # distribution and that the sample to which they are assigned is random.
    
    # 与此置换类型相关的零假设是，每对观测值都来自相同的基础分布，并且它们被分配到的样本是随机的。

    # Suppose `data` contains two samples; e.g. `a, b = data`.
    # Let `n` be the number of observations in `a`, which must also equal
    # the number of observations in `b`.
    
    # 假设`data`包含两个样本；例如`a, b = data`。
    # 让`n`表示`a`中的观测数量，这个数量也必须等于`b`中的观测数量。

    # When `1 < n_resamples < 2**n`, the elements of `a` are `b` are
    # randomly swapped between samples (maintaining their pairings) and the
    # statistic is calculated. This process is performed repeatedly,
    # `permutation` times, generating a distribution of the statistic under the
    # null hypothesis. The statistic of the original data is compared to this
    # distribution to determine the p-value.
    
    # 当`1 < n_resamples < 2**n`时，`a`和`b`中的元素在样本之间随机交换（保持它们的配对关系），然后计算统计量。
    # 此过程重复进行`permutation`次，生成在零假设下统计量的分布。将原始数据的统计量与此分布进行比较，确定p值。

    # When `n_resamples >= 2**n`, an exact test is performed: the observations
    # are assigned to the two samples in each distinct way (while maintaining
    # pairings) exactly once.
    
    # 当`n_resamples >= 2**n`时，执行精确检验：观测值被准确地分配到每种独特的方式中的两个样本中（同时保持配对关系）。

    # For `a = [a1, a2, a3]` and `b = [b1, b2, b3]`, an example of this
    # permutation type is `x = [b1, a2, b3]` and `y = [a1, b2, a3]`.
    
    # 对于`a = [a1, a2, a3]`和`b = [b1, b2, b3]`，此置换类型的示例是`x = [b1, a2, b3]`和`y = [a1, b2, a3]`。

    # `permutation_type='samples'` supports `data` containing any number
    # of samples, each of which must contain the same number of observations.
    # If `data` contains more than one sample, paired observations within
    # `data` are exchanged between samples *independently*. Therefore, if `m`
    # is the number of samples and `n` is the number of observations within
    # each sample, then the number of permutations in an exact test is::
    
    # `permutation_type='samples'`支持包含任意数量样本的`data`，每个样本必须包含相同数量的观测。
    # 如果`data`包含多个样本，则`data`中的配对观测值在样本之间*独立*交换。因此，如果`m`是样本数量，`n`是每个样本中的观测数量，
    # 那么在精确检验中的排列数量是：

        factorial(m)**n
    
    # Several paired-sample statistical tests, such as the Wilcoxon signed rank
    # test and paired-sample t-test, can be performed considering only the
    # *difference* between two paired elements. Accordingly, if `data` contains
    # only one sample, then the null distribution is formed by independently
    # changing the *sign* of each observation.
    
    # 可以执行几种配对样本统计检验，例如Wilcoxon符号秩检验和配对样本t检验，只考虑两个配对元素之间的*差异*。
    # 因此，如果`data`只包含一个样本，则零分布是通过独立改变每个观测值的*符号*形成的。
    # 导入所需库和模块
    >>> import numpy as np
    # 定义一个函数，计算两个样本在指定轴上的均值差作为统计量
    >>> def statistic(x, y, axis):
    ...     return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    # 使用 SciPy 中的正态分布生成随机数据
    >>> from scipy.stats import norm
    # 创建一个随机数生成器对象
    >>> rng = np.random.default_rng()
    # 从标准正态分布中生成一个长度为 5 的随机样本 x
    >>> x = norm.rvs(size=5, random_state=rng)
    # 从均值为 3 的正态分布中生成一个长度为 6 的随机样本 y
    >>> y = norm.rvs(size=6, loc=3, random_state=rng)
    # 计算在第 0 轴上的统计量，即两个样本均值的差值
    >>> statistic(x, y, 0)
    -3.5411688580987266
    a permutation test.

    >>> from scipy.stats import permutation_test
    >>> # 因为我们的统计量是向量化的，所以传递 `vectorized=True`
    >>> # `n_resamples=np.inf` 表示执行精确检验
    >>> res = permutation_test((x, y), statistic, vectorized=True,
    ...                        n_resamples=np.inf, alternative='less')
    >>> print(res.statistic)
    -3.5411688580987266
    >>> print(res.pvalue)
    0.004329004329004329

    在零假设下，获得小于或等于观察值的检验统计量的概率为0.4329%。这小于我们选择的5%阈值，因此我们认为这是反对零假设，支持备择假设的显著证据。

    由于上述样本的大小较小，`permutation_test`可以执行精确检验。对于较大的样本，我们将使用随机排列检验。

    >>> x = norm.rvs(size=100, random_state=rng)
    >>> y = norm.rvs(size=120, loc=0.2, random_state=rng)
    >>> res = permutation_test((x, y), statistic, n_resamples=9999,
    ...                        vectorized=True, alternative='less',
    ...                        random_state=rng)
    >>> print(res.statistic)
    -0.4230459671240913
    >>> print(res.pvalue)
    0.0015

    在零假设下，获得小于或等于观察值的检验统计量的概率约为0.0225%。这同样小于我们选择的5%阈值，因此我们有显著证据拒绝零假设，支持备择假设。

    对于大样本和大量排列情况，结果与相应的渐近检验——独立样本 t 检验相当。

    >>> from scipy.stats import ttest_ind
    >>> res_asymptotic = ttest_ind(x, y, alternative='less')
    >>> print(res_asymptotic.pvalue)
    0.0014669545224902675

    提供测试统计量的排列分布供进一步研究。

    >>> import matplotlib.pyplot as plt
    >>> plt.hist(res.null_distribution, bins=50)
    >>> plt.title("Permutation distribution of test statistic")
    >>> plt.xlabel("Value of Statistic")
    >>> plt.ylabel("Frequency")
    >>> plt.show()

    如果统计量由于有限机器精度而存在不准确性，检查零分布至关重要。考虑以下情况：

    >>> from scipy.stats import pearsonr
    >>> x = [1, 2, 4, 3]
    >>> y = [2, 4, 6, 8]
    >>> def statistic(x, y, axis=-1):
    ...     return pearsonr(x, y, axis=axis).statistic
    >>> res = permutation_test((x, y), statistic, vectorized=True,
    ...                        permutation_type='pairings',
    ...                        alternative='greater')
    >>> r, pvalue, null = res.statistic, res.pvalue, res.null_distribution

    在这种情况下，某些零分布元素与
    observed value of the correlation coefficient ``r`` due to numerical noise.
    We manually inspect the elements of the null distribution that are nearly
    the same as the observed value of the test statistic.

    >>> r
    0.7999999999999999
    >>> unique = np.unique(null)
    >>> unique
    array([-1. , -1. , -0.8, -0.8, -0.8, -0.6, -0.4, -0.4, -0.2, -0.2, -0.2,
        0. ,  0.2,  0.2,  0.2,  0.4,  0.4,  0.6,  0.8,  0.8,  0.8,  1. ,
        1. ])  # may vary
    >>> unique[np.isclose(r, unique)].tolist()
    [0.7999999999999998, 0.7999999999999999, 0.8]  # may vary

    If `permutation_test` were to perform the comparison naively, the
    elements of the null distribution with value ``0.7999999999999998`` would
    not be considered as extreme or more extreme as the observed value of the
    statistic, so the calculated p-value would be too small.

    >>> incorrect_pvalue = np.count_nonzero(null >= r) / len(null)
    >>> incorrect_pvalue
    0.14583333333333334  # may vary

    Instead, `permutation_test` treats elements of the null distribution that
    are within ``max(1e-14, abs(r)*1e-14)`` of the observed value of the
    statistic ``r`` to be equal to ``r``.

    >>> correct_pvalue = np.count_nonzero(null >= r - 1e-14) / len(null)
    >>> correct_pvalue
    0.16666666666666666
    >>> res.pvalue == correct_pvalue
    True

    This method of comparison is expected to be accurate in most practical
    situations, but the user is advised to assess this by inspecting the
    elements of the null distribution that are close to the observed value
    of the statistic. Also, consider the use of statistics that can be
    calculated using exact arithmetic (e.g. integer statistics).

    """

# 获取 `_permutation_test_iv` 函数的返回值，并将其分配给 `args` 变量
args = _permutation_test_iv(data, statistic, permutation_type, vectorized,
                            n_resamples, batch, alternative, axis,
                            random_state)
(data, statistic, permutation_type, vectorized, n_resamples, batch,
 alternative, axis, random_state) = args

# 使用指定的统计函数 `statistic` 对数据 `data` 进行计算，得到观察值 `observed`
observed = statistic(*data, axis=-1)

# 根据 `permutation_type` 选择合适的空分布计算函数，存储在 `null_calculators` 字典中
null_calculators = {"pairings": _calculate_null_pairings,
                    "samples": _calculate_null_samples,
                    "independent": _calculate_null_both}
# 准备传递给空分布计算函数的参数
null_calculator_args = (data, statistic, n_resamples,
                        batch, random_state)
# 根据 `permutation_type` 选择相应的空分布计算函数，并调用它进行计算
calculate_null = null_calculators[permutation_type]
# 调用空分布计算函数，获取空分布及相关信息
null_distribution, n_resamples, exact_test = (
    calculate_null(*null_calculator_args))

# See References [2] and [3]
# 根据是否进行了精确测试，确定调整因子 `adjustment`
adjustment = 0 if exact_test else 1

# 相对容差，用于检测在空分布中数值上接近但在数值上不同的值
eps =  (0 if not np.issubdtype(observed.dtype, np.inexact)
        else np.finfo(observed.dtype).eps*100)
gamma = np.abs(eps * observed)
    # 定义一个函数，用于计算在零分布下小于观察值的比例，返回对应的 p 值
    def less(null_distribution, observed):
        cmps = null_distribution <= observed + gamma
        pvalues = (cmps.sum(axis=0) + adjustment) / (n_resamples + adjustment)
        return pvalues

    # 定义一个函数，用于计算在零分布下大于观察值的比例，返回对应的 p 值
    def greater(null_distribution, observed):
        cmps = null_distribution >= observed - gamma
        pvalues = (cmps.sum(axis=0) + adjustment) / (n_resamples + adjustment)
        return pvalues

    # 定义一个函数，用于计算双侧检验的 p 值，结合 less 和 greater 函数的结果
    def two_sided(null_distribution, observed):
        pvalues_less = less(null_distribution, observed)
        pvalues_greater = greater(null_distribution, observed)
        pvalues = np.minimum(pvalues_less, pvalues_greater) * 2
        return pvalues

    # 创建一个字典，将不同假设检验方式与对应的函数关联起来
    compare = {"less": less,
               "greater": greater,
               "two-sided": two_sided}

    # 根据给定的假设检验方式调用相应的函数，计算 p 值
    pvalues = compare[alternative](null_distribution, observed)

    # 将 p 值限制在 [0, 1] 的范围内
    pvalues = np.clip(pvalues, 0, 1)

    # 返回一个 PermutationTestResult 对象，包含观察值、p 值和零分布
    return PermutationTestResult(observed, pvalues, null_distribution)
@dataclass
class ResamplingMethod:
    """Configuration information for a statistical resampling method.

    Instances of this class can be passed into the `method` parameter of some
    hypothesis test functions to perform a resampling or Monte Carlo version
    of the hypothesis test.

    Attributes
    ----------
    n_resamples : int
        The number of resamples to perform or Monte Carlo samples to draw.
    batch : int, optional
        The number of resamples to process in each vectorized call to
        the statistic. Batch sizes >>1 tend to be faster when the statistic
        is vectorized, but memory usage scales linearly with the batch size.
        Default is ``None``, which processes all resamples in a single batch.
    """
    n_resamples: int = 9999
    batch: int = None  # type: ignore[assignment]


@dataclass
class MonteCarloMethod(ResamplingMethod):
    """Configuration information for a Monte Carlo hypothesis test.

    Instances of this class can be passed into the `method` parameter of some
    hypothesis test functions to perform a Monte Carlo version of the
    hypothesis tests.

    Attributes
    ----------
    n_resamples : int, optional
        The number of Monte Carlo samples to draw. Default is 9999.
    batch : int, optional
        The number of Monte Carlo samples to process in each vectorized call to
        the statistic. Batch sizes >>1 tend to be faster when the statistic
        is vectorized, but memory usage scales linearly with the batch size.
        Default is ``None``, which processes all samples in a single batch.
    rvs : callable or tuple of callables, optional
        A callable or sequence of callables that generates random variates
        under the null hypothesis. Each element of `rvs` must be a callable
        that accepts keyword argument ``size`` (e.g. ``rvs(size=(m, n))``) and
        returns an N-d array sample of that shape. If `rvs` is a sequence, the
        number of callables in `rvs` must match the number of samples passed
        to the hypothesis test in which the `MonteCarloMethod` is used. Default
        is ``None``, in which case the hypothesis test function chooses values
        to match the standard version of the hypothesis test. For example,
        the null hypothesis of `scipy.stats.pearsonr` is typically that the
        samples are drawn from the standard normal distribution, so
        ``rvs = (rng.normal, rng.normal)`` where
        ``rng = np.random.default_rng()``.
    """
    rvs: object = None

    def _asdict(self):
        # `dataclasses.asdict` deepcopies; we don't want that.
        return dict(n_resamples=self.n_resamples, batch=self.batch,
                    rvs=self.rvs)


@dataclass
class PermutationMethod(ResamplingMethod):
    """Configuration information for a permutation hypothesis test.

    Instances of this class can be passed into the `method` parameter of some
    hypothesis test functions to perform a permutation version of the
    hypothesis tests.
    
    Inherits from `ResamplingMethod`, sharing attributes `n_resamples` and `batch`.

    No additional attributes are defined in this class compared to its superclass.
    """
    random_state: object = None



    # 定义一个类属性 `random_state`，用于控制伪随机数生成器的状态。默认为 None。
    random_state: object = None



    def _asdict(self):
        # `dataclasses.asdict` deepcopies; we don't want that.
        return dict(n_resamples=self.n_resamples, batch=self.batch,
                    random_state=self.random_state)



    # 定义一个方法 `_asdict`，用于将当前对象转换为字典形式的表示。
    def _asdict(self):
        # 返回包含 `n_resamples`、`batch` 和 `random_state` 的字典表示。
        # 注意：`dataclasses.asdict` 会深拷贝对象，这里我们不需要这样的行为。
        return dict(n_resamples=self.n_resamples, batch=self.batch,
                    random_state=self.random_state)
# 使用 `dataclass` 装饰器定义一个名为 `BootstrapMethod` 的类，它继承自 `ResamplingMethod` 类。
@dataclass
class BootstrapMethod(ResamplingMethod):
    """Configuration information for a bootstrap confidence interval.

    Instances of this class can be passed into the `method` parameter of some
    confidence interval methods to generate a bootstrap confidence interval.

    Attributes
    ----------
    n_resamples : int, optional
        The number of resamples to perform. Default is 9999.
    batch : int, optional
        The number of resamples to process in each vectorized call to
        the statistic. Batch sizes >>1 tend to be faster when the statistic
        is vectorized, but memory usage scales linearly with the batch size.
        Default is ``None``, which processes all resamples in a single batch.
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional

        Pseudorandom number generator state used to generate resamples.

        If `random_state` is already a ``Generator`` or ``RandomState``
        instance, then that instance is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is ``None`` (default), the
        `numpy.random.RandomState` singleton is used.

    method : {'bca', 'percentile', 'basic'}
        Whether to use the 'percentile' bootstrap ('percentile'), the 'basic'
        (AKA 'reverse') bootstrap ('basic'), or the bias-corrected and
        accelerated bootstrap ('BCa', default).
    """
    # 定义属性 `random_state` 默认为 `None`，`method` 默认为 `'BCa'`
    random_state: object = None
    method: str = 'BCa'

    # 定义 `_asdict` 方法，返回包含当前对象属性的字典，避免使用 `dataclasses.asdict` 的深拷贝
    def _asdict(self):
        # 返回一个包含当前对象属性的字典
        return dict(n_resamples=self.n_resamples, batch=self.batch,
                    random_state=self.random_state, method=self.method)
```