# `D:\src\scipysrc\scipy\scipy\stats\_stats_py.py`

```
"""
A collection of basic statistical functions for Python.

References
----------
.. [CRCProbStat2000] Zwillinger, D. and Kokoska, S. (2000). CRC Standard
   Probability and Statistics Tables and Formulae. Chapman & Hall: New
   York. 2000.

"""
# 引入警告模块，用于处理警告信息
import warnings
# 引入数学模块
import math
# 从数学模块中导入最大公约数函数 gcd
from math import gcd
# 引入命名元组工具
from collections import namedtuple
# 引入序列抽象基类
from collections.abc import Sequence

# 引入 numpy 库，并从中导入 array, asarray, ma 等函数和类
import numpy as np
from numpy import array, asarray, ma

# 引入 sparse 子模块
from scipy import sparse
# 从 scipy.spatial 中导入 distance_matrix 函数
from scipy.spatial import distance_matrix

# 从 scipy.optimize 中导入 milp 和 LinearConstraint 函数
from scipy.optimize import milp, LinearConstraint
# 从 scipy._lib._util 中导入一系列函数和异常类
from scipy._lib._util import (check_random_state, _get_nan,
                              _rename_parameter, _contains_nan,
                              AxisError, _lazywhere)

# 导入 scipy.special 模块，并用 special 作为别名
import scipy.special as special
# 从 scipy.linalg 中导入模块，尽管未在此处使用，但需要在弃用期结束前保留
from scipy import linalg  # noqa: F401

# 导入 distributions 和 _mstats_basic 模块
from . import distributions
from . import _mstats_basic as mstats_basic

# 从 _stats_mstats_common 模块中导入 _find_repeats, theilslopes, siegelslopes 函数
from ._stats_mstats_common import _find_repeats, theilslopes, siegelslopes
# 从 _stats 模块中导入 _kendall_dis, _toint64, _weightedrankedtau 函数
from ._stats import _kendall_dis, _toint64, _weightedrankedtau

# 导入 dataclass 和 field 函数
from dataclasses import dataclass, field
# 从 _hypotests 模块中导入 _all_partitions 函数
from ._hypotests import _all_partitions
# 从 _stats_pythran 模块中导入 _compute_outer_prob_inside_method 函数
from ._stats_pythran import _compute_outer_prob_inside_method
# 从 _resampling 模块中导入一系列函数和类
from ._resampling import (MonteCarloMethod, PermutationMethod, BootstrapMethod,
                          monte_carlo_test, permutation_test, bootstrap,
                          _batch_generator)
# 从 _axis_nan_policy 模块中导入一系列函数和警告类
from ._axis_nan_policy import (_axis_nan_policy_factory, _broadcast_arrays,
                               _broadcast_concatenate, _broadcast_shapes,
                               _broadcast_array_shapes_remove_axis, SmallSampleWarning,
                               too_small_1d_not_omit, too_small_1d_omit,
                               too_small_nd_not_omit, too_small_nd_omit)
# 从 _binomtest 模块中导入 _binary_search_for_binom_tst 函数
from ._binomtest import _binary_search_for_binom_tst as _binary_search
# 从 scipy._lib._bunch 中导入 _make_tuple_bunch 函数
from scipy._lib._bunch import _make_tuple_bunch
# 从 scipy 中导入 stats 模块
from scipy import stats
# 从 scipy.optimize 中导入 root_scalar 函数
from scipy.optimize import root_scalar
# 从 scipy._lib._util 模块中导入 normalize_axis_index 函数
from scipy._lib._util import normalize_axis_index
# 从 scipy._lib._array_api 模块中导入多个函数和类
from scipy._lib._array_api import (array_namespace, is_numpy, atleast_nd,
                                   xp_clip, xp_moveaxis_to_end, xp_sign,
                                   xp_minimum)
# 从 scipy._lib.array_api_compat 模块中导入 xp_size 函数
from scipy._lib.array_api_compat import size as xp_size

# 在 `__init__.py` 中添加而不是这里，将这些函数和类添加到模块的公共接口中
__all__ = ['find_repeats', 'gmean', 'hmean', 'pmean', 'mode', 'tmean', 'tvar',
           'tmin', 'tmax', 'tstd', 'tsem', 'moment',
           'skew', 'kurtosis', 'describe', 'skewtest', 'kurtosistest',
           'normaltest', 'jarque_bera',
           'scoreatpercentile', 'percentileofscore',
           'cumfreq', 'relfreq', 'obrientransform',
           'sem', 'zmap', 'zscore', 'gzscore', 'iqr', 'gstd',
           'median_abs_deviation',
           'sigmaclip', 'trimboth', 'trim1', 'trim_mean',
           'f_oneway', 'pearsonr', 'fisher_exact',
           'spearmanr', 'pointbiserialr',
           'kendalltau', 'weightedtau',
           'linregress', 'siegelslopes', 'theilslopes', 'ttest_1samp',
           'ttest_ind', 'ttest_ind_from_stats', 'ttest_rel',
           'kstest', 'ks_1samp', 'ks_2samp',
           'chisquare', 'power_divergence',
           'tiecorrect', 'ranksums', 'kruskal', 'friedmanchisquare',
           'rankdata', 'combine_pvalues', 'quantile_test',
           'wasserstein_distance', 'wasserstein_distance_nd', 'energy_distance',
           'brunnermunzel', 'alexandergovern',
           'expectile']

# 检查输入数组是否为 numpy 数组，并根据需要使用数组命名空间
def _chk_asarray(a, axis, *, xp=None):
    # 如果未指定 xp 参数，则根据数组 a 的类型选择数组命名空间
    if xp is None:
        xp = array_namespace(a)

    # 如果未指定轴（axis），将数组 a 重塑为一维数组，输出轴设为 0
    if axis is None:
        a = xp.reshape(a, (-1,))
        outaxis = 0
    else:
        # 否则，将数组 a 转换为数组对象
        a = xp.asarray(a)
        outaxis = axis

    # 如果数组 a 是标量，则将其重塑为一维数组
    if a.ndim == 0:
        a = xp.reshape(a, (-1,))

    return a, outaxis

# 检查两个输入数组 a 和 b 是否为 numpy 数组，并根据需要使用数组命名空间
def _chk2_asarray(a, b, axis):
    # 如果未指定轴（axis），则将数组 a 和 b 都展平为一维数组
    if axis is None:
        a = np.ravel(a)
        b = np.ravel(b)
        outaxis = 0
    else:
        # 否则，将数组 a 和 b 转换为数组对象
        a = np.asarray(a)
        b = np.asarray(b)
        outaxis = axis

    # 如果数组 a 是标量，则至少将其转换为一维数组
    if a.ndim == 0:
        a = np.atleast_1d(a)
    # 如果数组 b 是标量，则至少将其转换为一维数组
    if b.ndim == 0:
        b = np.atleast_1d(b)

    return a, b, outaxis

# 创建名为 SignificanceResult 的具名元组，包含 statistic 和 pvalue 两个字段
SignificanceResult = _make_tuple_bunch('SignificanceResult',
                                       ['statistic', 'pvalue'], [])

# 带权重计算几何平均值的函数，注意 `weights` 与 `x` 是成对的
@_axis_nan_policy_factory(
        lambda x: x, n_samples=1, n_outputs=1, too_small=0, paired=True,
        result_to_tuple=lambda x: (x,), kwd_samples=['weights'])
# 下面是函数文档字符串，详细描述了函数的功能、数学定义和参数说明
def gmean(a, axis=0, dtype=None, weights=None):
    r"""Compute the weighted geometric mean along the specified axis.

    The weighted geometric mean of the array :math:`a_i` associated to weights
    :math:`w_i` is:

    .. math::

        \exp \left( \frac{ \sum_{i=1}^n w_i \ln a_i }{ \sum_{i=1}^n w_i }
                   \right) \, ,

    and, with equal weights, it gives:

    .. math::

        \sqrt[n]{ \prod_{i=1}^n a_i } \, .

    Parameters
    ----------
    xp = array_namespace(a, weights)

调用 `array_namespace` 函数，用输入的 `a` 和 `weights` 创建一个数组命名空间对象 `xp`。


    a = xp.asarray(a, dtype=dtype)

使用 `xp` 命名空间对象的 `asarray` 方法将输入的 `a` 转换为数组，并按照指定的 `dtype` 进行类型转换。


    if weights is not None:
        weights = xp.asarray(weights, dtype=dtype)

如果 `weights` 参数不为 `None`，则使用 `xp` 命名空间对象的 `asarray` 方法将 `weights` 转换为数组，并按照指定的 `dtype` 进行类型转换。


    with np.errstate(divide='ignore'):
        log_a = xp.log(a)

设置 NumPy 的错误状态，忽略除法错误。然后使用 `xp` 命名空间对象的 `log` 方法计算数组 `a` 中每个元素的自然对数，并将结果存储在 `log_a` 中。


    return xp.exp(_xp_mean(log_a, axis=axis, weights=weights))

调用 `_xp_mean` 函数，传入 `log_a` 数组、`axis` 参数和 `weights` 参数（如果存在）。然后使用 `xp` 命名空间对象的 `exp` 方法，对 `_xp_mean` 函数的返回值进行指数运算，并将结果作为函数的返回值返回。
# 定义装饰器函数，用于创建特定的 NaN 策略工厂，返回一个处理 NaN 的函数
@_axis_nan_policy_factory(
    # 使用 lambda 函数对参数进行处理，将 x 转为元组，设置参数：样本数为 1，输出数为 1，太小值为 0，成对计算为 True
    lambda x: x, n_samples=1, n_outputs=1, too_small=0, paired=True,
    # 结果转换为元组的函数
    result_to_tuple=lambda x: (x,),
    # 权重样本的关键字参数为 'weights'
    kwd_samples=['weights']
)
# 定义计算加权调和平均值的函数 hmean
def hmean(a, axis=0, dtype=None, *, weights=None):
    r"""Calculate the weighted harmonic mean along the specified axis.

    The weighted harmonic mean of the array :math:`a_i` associated to weights
    :math:`w_i` is:

    .. math::

        \frac{ \sum_{i=1}^n w_i }{ \sum_{i=1}^n \frac{w_i}{a_i} } \, ,

    and, with equal weights, it gives:

    .. math::

        \frac{ n }{ \sum_{i=1}^n \frac{1}{a_i} } \, .

    Parameters
    ----------
    a : array_like
        Input array, masked array or object that can be converted to an array.
    axis : int or None, optional
        Axis along which the harmonic mean is computed. Default is 0.
        If None, compute over the whole array `a`.
    dtype : dtype, optional
        Type of the returned array and of the accumulator in which the
        elements are summed. If `dtype` is not specified, it defaults to the
        dtype of `a`, unless `a` has an integer `dtype` with a precision less
        than that of the default platform integer. In that case, the default
        platform integer is used.
    weights : array_like, optional
        The weights array can either be 1-D (in which case its length must be
        the size of `a` along the given `axis`) or of the same shape as `a`.
        Default is None, which gives each value a weight of 1.0.

        .. versionadded:: 1.9

    Returns
    -------
    hmean : ndarray
        See `dtype` parameter above.

    See Also
    --------
    numpy.mean : Arithmetic average
    numpy.average : Weighted average
    gmean : Geometric mean

    Notes
    -----
    The sample harmonic mean is the reciprocal of the mean of the reciprocals
    of the observations.

    The harmonic mean is computed over a single dimension of the input
    array, axis=0 by default, or all values in the array if axis=None.
    float64 intermediate and return values are used for integer inputs.

    The harmonic mean is only defined if all observations are non-negative;
    otherwise, the result is NaN.

    References
    ----------
    .. [1] "Weighted Harmonic Mean", *Wikipedia*,
           https://en.wikipedia.org/wiki/Harmonic_mean#Weighted_harmonic_mean
    .. [2] Ferger, F., "The nature and use of the harmonic mean", Journal of
           the American Statistical Association, vol. 26, pp. 36-40, 1931

    Examples
    --------
    >>> from scipy.stats import hmean
    >>> hmean([1, 4])
    1.6000000000000001
    >>> hmean([1, 2, 3, 4, 5, 6, 7])
    2.6997245179063363
    >>> hmean([1, 4, 7], weights=[3, 1, 3])
    1.9029126213592233

    """
    # 使用 array_namespace 函数根据输入数组和权重数组创建适当的数组命名空间对象 xp
    xp = array_namespace(a, weights)
    # 将输入数组 a 转换为 xp 数组，指定数据类型为 dtype
    a = xp.asarray(a, dtype=dtype)

    # 如果权重数组不为 None，则将权重数组也转换为 xp 数组，指定数据类型为 dtype
    if weights is not None:
        weights = xp.asarray(weights, dtype=dtype)

    # 创建一个布尔掩码，标识数组 a 中小于 0 的元素
    negative_mask = a < 0
    # 检查是否存在负数掩码
    if xp.any(negative_mask):
        # 使用 `where` 函数避免对数据类型的敏感性，并确保在 JAX 中正常工作。
        # 这是异常情况，允许稍微慢一些。目前对于 array_api_strict 无效，
        # 但请参阅 data-apis/array-api#807。
        
        # 将负数掩码位置上的元素替换为 NaN
        a = xp.where(negative_mask, xp.nan, a)
        
        # 发出运行时警告，说明调用的函数要求所有元素非负
        message = ("The harmonic mean is only defined if all elements are "
                   "non-negative; otherwise, the result is NaN.")
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    # 忽略除法中的错误，以处理除数为零或无穷大的情况
    with np.errstate(divide='ignore'):
        # 返回倒数的平均数（即调和平均数的倒数）
        return 1.0 / _xp_mean(1.0 / a, axis=axis, weights=weights)
@_axis_nan_policy_factory(
        lambda x: x, n_samples=1, n_outputs=1, too_small=0, paired=True,
        result_to_tuple=lambda x: (x,), kwd_samples=['weights'])
# 装饰器函数，返回一个函数，用于处理带有特定轴向的加权幂均值计算
def pmean(a, p, *, axis=0, dtype=None, weights=None):
    r"""Calculate the weighted power mean along the specified axis.

    The weighted power mean of the array :math:`a_i` associated to weights
    :math:`w_i` is:

    .. math::

        \left( \frac{ \sum_{i=1}^n w_i a_i^p }{ \sum_{i=1}^n w_i }
              \right)^{ 1 / p } \, ,

    and, with equal weights, it gives:

    .. math::

        \left( \frac{ 1 }{ n } \sum_{i=1}^n a_i^p \right)^{ 1 / p }  \, .

    When ``p=0``, it returns the geometric mean.

    This mean is also called generalized mean or Hölder mean, and must not be
    confused with the Kolmogorov generalized mean, also called
    quasi-arithmetic mean or generalized f-mean [3]_.

    Parameters
    ----------
    a : array_like
        Input array, masked array or object that can be converted to an array.
    p : int or float
        Exponent.
    axis : int or None, optional
        Axis along which the power mean is computed. Default is 0.
        If None, compute over the whole array `a`.
    dtype : dtype, optional
        Type of the returned array and of the accumulator in which the
        elements are summed. If `dtype` is not specified, it defaults to the
        dtype of `a`, unless `a` has an integer `dtype` with a precision less
        than that of the default platform integer. In that case, the default
        platform integer is used.
    weights : array_like, optional
        The weights array can either be 1-D (in which case its length must be
        the size of `a` along the given `axis`) or of the same shape as `a`.
        Default is None, which gives each value a weight of 1.0.

    Returns
    -------
    pmean : ndarray, see `dtype` parameter above.
        Output array containing the power mean values.

    See Also
    --------
    numpy.average : Weighted average
    gmean : Geometric mean
    hmean : Harmonic mean

    Notes
    -----
    The power mean is computed over a single dimension of the input
    array, ``axis=0`` by default, or all values in the array if ``axis=None``.
    float64 intermediate and return values are used for integer inputs.

    The power mean is only defined if all observations are non-negative;
    otherwise, the result is NaN.

    .. versionadded:: 1.9

    References
    ----------
    .. [1] "Generalized Mean", *Wikipedia*,
           https://en.wikipedia.org/wiki/Generalized_mean
    .. [2] Norris, N., "Convexity properties of generalized mean value
           functions", The Annals of Mathematical Statistics, vol. 8,
           pp. 118-120, 1937
    .. [3] Bullen, P.S., Handbook of Means and Their Inequalities, 2003

    Examples
    --------
    >>> from scipy.stats import pmean, hmean, gmean
    >>> pmean([1, 4], 1.3)
    2.639372938300652
    >>> pmean([1, 2, 3, 4, 5, 6, 7], 1.3)
    """
    如果指数 p 不是 int 或 float 类型，则抛出数值错误异常
    """
    if not isinstance(p, (int, float)):
        raise ValueError("Power mean only defined for exponent of type int or "
                         "float.")

    """
    当 p 等于 0 时，返回几何平均值 gmean 的计算结果
    """
    if p == 0:
        return gmean(a, axis=axis, dtype=dtype, weights=weights)

    """
    使用 array_namespace 函数处理数组 a 和 weights 的命名空间
    """
    xp = array_namespace(a, weights)
    """
    将数组 a 转换为 xp 数组，并指定数据类型为 dtype
    """
    a = xp.asarray(a, dtype=dtype)

    """
    如果 weights 不为 None，则将其转换为 xp 数组，并指定数据类型为 dtype
    """
    if weights is not None:
        weights = xp.asarray(weights, dtype=dtype)

    """
    生成一个布尔掩码，标记数组 a 中小于 0 的元素
    """
    negative_mask = a < 0
    """
    如果掩码中有任何 True 值，则将对应位置的元素设置为 NaN
    """
    if xp.any(negative_mask):
        """
        使用 xp.where 函数替换负数元素为 NaN，避免处理数据类型时的复杂性，适用于 JAX 等环境
        """
        a = xp.where(negative_mask, np.nan, a)
        """
        引发运行时警告，提示计算幂均值时应所有元素均为非负数，否则结果为 NaN
        """
        message = ("The power mean is only defined if all elements are "
                   "non-negative; otherwise, the result is NaN.")
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    """
    设置运行时错误处理，对除以 0 和无效操作进行忽略
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        """
        返回数组 a 的每个元素的 p 次方的均值的 p 次方根的结果
        """
        return _xp_mean(a**float(p), axis=axis, weights=weights)**(1/p)
# 使用 namedtuple 定义一个名为 ModeResult 的数据结构，包含 'mode' 和 'count' 两个字段
ModeResult = namedtuple('ModeResult', ('mode', 'count'))

# 定义一个函数 `_mode_result`，用于处理计算结果中的 mode 和 count
def _mode_result(mode, count):
    # 如果 count 包含 NaN 值，则将其处理为合理的值：mode 可以是 NaN，但 count 应该是 0
    i = np.isnan(count)
    if i.shape == ():
        count = np.asarray(0, dtype=count.dtype)[()] if i else count
    else:
        count[i] = 0
    return ModeResult(mode, count)

# 使用装饰器 `_axis_nan_policy_factory` 包装 `mode` 函数，实现向量化处理，并覆盖默认设置
@_axis_nan_policy_factory(_mode_result, override={'vectorization': True,
                                                  'nan_propagation': False})
# `mode` 函数定义，用于计算数组中的模态（最常见）值
def mode(a, axis=0, nan_policy='propagate', keepdims=False):
    r"""Return an array of the modal (most common) value in the passed array.

    If there is more than one such value, only one is returned.
    The bin-count for the modal bins is also returned.

    Parameters
    ----------
    a : array_like
        Numeric, n-dimensional array of which to find mode(s).
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over
        the whole array `a`.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

          * 'propagate': treats nan as it would treat any other value
          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values
    keepdims : bool, optional
        If set to ``False``, the `axis` over which the statistic is taken
        is consumed (eliminated from the output array). If set to ``True``,
        the `axis` is retained with size one, and the result will broadcast
        correctly against the input array.

    Returns
    -------
    mode : ndarray
        Array of modal values.
    count : ndarray
        Array of counts for each mode.

    Notes
    -----
    The mode  is calculated using `numpy.unique`.
    In NumPy versions 1.21 and after, all NaNs - even those with different
    binary representations - are treated as equivalent and counted as separate
    instances of the same value.

    By convention, the mode of an empty array is NaN, and the associated count
    is zero.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[3, 0, 3, 7],
    ...               [3, 2, 6, 2],
    ...               [1, 7, 2, 8],
    ...               [3, 0, 6, 1],
    ...               [3, 2, 5, 5]])
    >>> from scipy import stats
    >>> stats.mode(a, keepdims=True)
    ModeResult(mode=array([[3, 0, 6, 1]]), count=array([[4, 2, 2, 1]]))

    To get mode of whole array, specify ``axis=None``:

    >>> stats.mode(a, axis=None, keepdims=True)
    ModeResult(mode=[[3]], count=[[5]])
    >>> stats.mode(a, axis=None, keepdims=False)
    ModeResult(mode=3, count=5)

    """
    # `axis`, `nan_policy`, and `keepdims` are handled by `_axis_nan_policy`
    # 检查数组 `a` 的数据类型是否为数值类型，如果不是则抛出类型错误
    if not np.issubdtype(a.dtype, np.number):
        # 准备错误消息，说明参数 `a` 不被识别为数值类型，并提供替代的建议
        message = ("Argument `a` is not recognized as numeric. "
                   "Support for input that cannot be coerced to a numeric "
                   "array was deprecated in SciPy 1.9.0 and removed in SciPy "
                   "1.11.0. Please consider `np.unique`.")
        # 抛出类型错误异常，显示上述错误消息
        raise TypeError(message)

    # 如果数组 `a` 的大小为 0
    if a.size == 0:
        # 获取 NaN 值，这里函数 `_get_nan` 根据参数 `a` 返回 NaN
        NaN = _get_nan(a)
        # 返回一个 ModeResult 对象，包含 NaN 值和计数 0，数据类型与 NaN 相同
        return ModeResult(*np.array([NaN, 0], dtype=NaN.dtype))

    # 计算数组 `a` 中唯一值及其出现次数
    vals, cnts = np.unique(a, return_counts=True)
    # 找出出现次数最多的值及其计数
    modes, counts = vals[cnts.argmax()], cnts.max()
    # 返回一个 ModeResult 对象，包含众数和其计数
    return ModeResult(modes[()], counts[()])
# 将数组中超出限制范围的元素替换为指定的值。
# 这是一个实用函数。

def _put_val_to_limits(a, limits, inclusive, val=np.nan, xp=None):
    """Replace elements outside limits with a value.

    This is primarily a utility function.

    Parameters
    ----------
    a : array
        待处理的数组。
    limits : (float or None, float or None)
        包含 (下限, 上限) 的元组。输入数组中小于下限或大于上限的元素将被替换为 `val`。None 表示没有限制。
    inclusive : (bool, bool)
        包含 (下限标志, 上限标志) 的元组。这些标志确定是否包含等于下限或上限的值。
    val : float, default: NaN
        数组的极端元素将被替换的值，默认为 NaN。
    xp : array module, optional
        用于数组操作的模块，如果为 None，则使用数组 `a` 所在的模块。

    """
    # 如果 xp 为 None，则使用数组 `a` 所在的模块
    xp = array_namespace(a) if xp is None else xp
    # 创建一个与数组 `a` 相同形状的布尔类型的掩码数组
    mask = xp.zeros(a.shape, dtype=xp.bool)
    # 如果 limits 为 None，则返回原始数组 `a` 和空的掩码数组
    if limits is None:
        return a, mask
    # 从 limits 元组中获取下限和上限
    lower_limit, upper_limit = limits
    # 从 inclusive 元组中获取下限和上限的包含标志
    lower_include, upper_include = inclusive
    # 如果 lower_limit 不为 None，则根据 lower_include 标志设置掩码
    if lower_limit is not None:
        mask |= (a < lower_limit) if lower_include else a <= lower_limit
    # 如果 upper_limit 不为 None，则根据 upper_include 标志设置掩码
    if upper_limit is not None:
        mask |= (a > upper_limit) if upper_include else a >= upper_limit
    # 如果掩码全为 True，则抛出 ValueError 异常，表示给定范围内没有数组值
    if xp.all(mask):
        raise ValueError("No array values within given limits")
    # 如果掩码中有任何 True 值，则根据掩码替换数组 `a` 的极端值为 `val`
    if xp.any(mask):
        # 当 data-apis/array-api#807 解决后，希望这种习惯用法（及其许多其他实例）是临时的
        # 根据数组 `a` 的数据类型和 `val` 的类型设置新的数据类型
        dtype = xp.asarray(1.).dtype if xp.isdtype(a.dtype, 'integral') else a.dtype
        # 使用 where 函数替换数组 `a` 的极端值
        a = xp.where(mask, xp.asarray(val, dtype=dtype), a)
    # 返回替换后的数组 `a` 和掩码数组
    return a, mask



# 创建一个修饰器函数，用于生成带有轴上 NaN 策略的修饰器
# 使用 lambda 表达式定义修饰器函数，根据输入的函数返回一个修饰过的函数
# 默认的轴为 None，并将结果转换为元组返回

@_axis_nan_policy_factory(
    lambda x: x, n_outputs=1, default_axis=None,
    result_to_tuple=lambda x: (x,)
)
# 计算修剪均值
# 该函数计算给定值的算术均值，忽略超出给定 `limits` 范围之外的值
def tmean(a, limits=None, inclusive=(True, True), axis=None):
    """Compute the trimmed mean.

    This function finds the arithmetic mean of given values, ignoring values
    outside the given `limits`.

    Parameters
    ----------
    a : array_like
        数组形式的值。
    limits : None or (lower limit, upper limit), optional
        输入数组中小于下限或大于上限的值将被忽略。当 limits 为 None（默认值）时，使用所有值。
        元组中任一限制值也可以为 None，表示半开区间。
    inclusive : (bool, bool), optional
        包含 (下限标志, 上限标志) 的元组。这些标志确定是否包含等于下限或上限的值。默认值为 (True, True)。
    axis : int or None, optional
        计算的轴向。默认为 None。

    Returns
    -------
    tmean : ndarray
        修剪后的均值。

    See Also
    --------
    trim_mean : 返回修剪了两侧比例后的均值。

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> x = np.arange(20)
    >>> stats.tmean(x)
    9.5
    """
    # 使用统计模块 stats 的 tmean 函数计算数组 x 在指定轴向上的均值，轴范围为 (3,17)，返回结果为 10.0
    >>> stats.tmean(x, (3,17))
    10.0

    """
    # 将数组 a 转换为适合的命名空间 xp
    xp = array_namespace(a)
    # 调用 _put_val_to_limits 函数，将数组 a 中的值置于指定的限制范围内，并返回处理后的数组 a 和掩码 mask
    a, mask = _put_val_to_limits(a, limits, inclusive, val=0., xp=xp)
    # 由于数据接口兼容性问题，需要显式指定数据类型 dtype
    # 在指定轴上计算数组 a 的总和，结果存储在变量 sum 中
    sum = xp.sum(a, axis=axis, dtype=a.dtype)
    # 在指定轴上计算掩码 mask 的反向数组的总和，表示有效元素的数量，结果存储在变量 n 中
    n = xp.sum(xp.asarray(~mask, dtype=a.dtype), axis=axis, dtype=a.dtype)
    # 使用 _lazywhere 函数，根据条件 n != 0，计算 sum 和 n 的比值，如果 n 为 0 则返回 NaN
    mean = _lazywhere(n != 0, (sum, n), xp.divide, xp.nan)
    # 如果 mean 的维度为 0，则返回其标量值；否则返回 mean 的数组形式
    return mean[()] if mean.ndim == 0 else mean
# 使用 `_axis_nan_policy_factory` 装饰器创建修饰函数，用于处理NaN值策略和结果转换
@_axis_nan_policy_factory(
    lambda x: x, n_outputs=1, result_to_tuple=lambda x: (x,)
)
# 定义 `tvar` 函数，用于计算修剪后的方差
def tvar(a, limits=None, inclusive=(True, True), axis=0, ddof=1):
    """Compute the trimmed variance.

    This function computes the sample variance of an array of values,
    while ignoring values which are outside of given `limits`.

    Parameters
    ----------
    a : array_like
        Array of values.
    limits : None or (lower limit, upper limit), optional
        Values in the input array less than the lower limit or greater than the
        upper limit will be ignored. When limits is None, then all values are
        used. Either of the limit values in the tuple can also be None
        representing a half-open interval.  The default value is None.
    inclusive : (bool, bool), optional
        A tuple consisting of the (lower flag, upper flag).  These flags
        determine whether values exactly equal to the lower or upper limits
        are included.  The default value is (True, True).
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over the
        whole array `a`.
    ddof : int, optional
        Delta degrees of freedom.  Default is 1.

    Returns
    -------
    tvar : float
        Trimmed variance.

    Notes
    -----
    `tvar` computes the unbiased sample variance, i.e. it uses a correction
    factor ``n / (n - 1)``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> x = np.arange(20)
    >>> stats.tvar(x)
    35.0
    >>> stats.tvar(x, (3,17))
    20.0

    """
    # 将输入数组 `a` 转换为适当的数组命名空间
    xp = array_namespace(a)
    # 调用 `_put_val_to_limits` 函数，根据给定的限制修剪输入数组 `a` 的值
    a, _ = _put_val_to_limits(a, limits, inclusive, xp=xp)
    # 使用 `_xp_var` 函数计算修剪后的方差，忽略NaN值，应用修正因子 `ddof`
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SmallSampleWarning)
        # 当前行为对于替代数组后端类似于 `nan_policy='omit'`，但对于其他后端将处理 `nan_policy='propagate'`
        return _xp_var(a, correction=ddof, axis=axis, nan_policy='omit', xp=xp)

# 使用 `_axis_nan_policy_factory` 装饰器创建修饰函数，用于处理NaN值策略和结果转换
@_axis_nan_policy_factory(
    lambda x: x, n_outputs=1, result_to_tuple=lambda x: (x,)
)
# 定义 `tmin` 函数，用于计算修剪后的最小值
def tmin(a, lowerlimit=None, axis=0, inclusive=True, nan_policy='propagate'):
    """Compute the trimmed minimum.

    This function finds the minimum value of an array `a` along the
    specified axis, but only considering values greater than a specified
    lower limit.

    Parameters
    ----------
    a : array_like
        Array of values.
    lowerlimit : None or float, optional
        Values in the input array less than the given limit will be ignored.
        When lowerlimit is None, then all values are used. The default value
        is None.
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over the
        whole array `a`.

    inclusive : bool, optional
        Determines whether values exactly equal to the lower limit are included.
        The default value is True.
    nan_policy : {'propagate', 'omit'}, optional
        Defines how to handle NaN values. 'propagate' means propagate NaNs,
        while 'omit' means ignore NaNs. The default is 'propagate'.
    """
    xp = array_namespace(a)



    # 使用 array_namespace 函数根据输入数组 a 确定适当的执行命名空间
    xp = array_namespace(a)



    # 记住原始的数据类型；_put_val_to_limits 可能需要修改它
    dtype = a.dtype



    # 调用 _put_val_to_limits 函数，将数组 a 的值限制在指定的下限和上限内，
    # 包括是否包含下限值的判断，同时使用 xp.inf 作为无效值进行标记
    a, mask = _put_val_to_limits(a, (lowerlimit, None), (inclusive, None),
                                 val=xp.inf, xp=xp)



    # 在指定的轴上计算数组 a 的最小值
    min = xp.min(a, axis=axis)



    # 计算非遮罩值的数量，并转换为与 a 相同的数据类型
    n = xp.sum(xp.asarray(~mask, dtype=a.dtype), axis=axis)



    # 使用 xp.where 函数根据条件选择返回最终的结果数组 res，
    # 如果 n 不为零，则返回最小值 min，否则返回 NaN
    res = xp.where(n != 0, min, xp.nan)



    # 如果结果数组 res 中不存在 NaN 值，则根据原始数据类型 dtype 进行类型转换
    if not xp.any(xp.isnan(res)):
        res = xp.astype(res, dtype, copy=False)



    # 如果结果数组 res 是标量（0 维），则返回其单个元素；否则返回整个数组 res
    return res[()] if res.ndim == 0 else res


这段代码是一个数学统计函数，用于计算在给定数组中的修剪最小值。它首先对输入数组进行一系列操作和限制，然后计算数组的最小值，并根据特定的条件返回结果。
@_axis_nan_policy_factory(
    lambda x: x, n_outputs=1, result_to_tuple=lambda x: (x,)
)
# 定义了一个修饰器函数，用于处理轴向操作和NaN策略
def tmax(a, upperlimit=None, axis=0, inclusive=True, nan_policy='propagate'):
    """Compute the trimmed maximum.

    This function computes the maximum value of an array along a given axis,
    while ignoring values larger than a specified upper limit.

    Parameters
    ----------
    a : array_like
        Array of values.
    upperlimit : None or float, optional
        Values in the input array greater than the given limit will be ignored.
        When upperlimit is None, then all values are used. The default value
        is None.
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over the
        whole array `a`.
    inclusive : {True, False}, optional
        This flag determines whether values exactly equal to the upper limit
        are included.  The default value is True.

    Returns
    -------
    tmax : float, int or ndarray
        Trimmed maximum.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> x = np.arange(20)
    >>> stats.tmax(x)
    19

    >>> stats.tmax(x, 13)
    13

    >>> stats.tmax(x, 13, inclusive=False)
    12

    """
    xp = array_namespace(a)

    # 记录原始数据类型；_put_val_to_limits 可能需要改变它
    dtype = a.dtype
    # 将数组 a 处理成限制范围内的值，并返回处理后的数组和掩码
    a, mask = _put_val_to_limits(a, (None, upperlimit), (None, inclusive),
                                 val=-xp.inf, xp=xp)

    # 沿指定轴计算数组 a 的最大值
    max = xp.max(a, axis=axis)
    # 计算非 NaN 值的数量
    n = xp.sum(xp.asarray(~mask, dtype=a.dtype), axis=axis)
    # 如果 n 不为零，返回 max；否则返回 NaN
    res = xp.where(n != 0, max, xp.nan)

    # 如果结果数组没有 NaN 值，则将其转换为原始数据类型
    if not xp.any(xp.isnan(res)):
        res = xp.astype(res, dtype, copy=False)

    # 如果结果数组是标量（零维），则返回其值；否则返回数组本身
    return res[()] if res.ndim == 0 else res


@_axis_nan_policy_factory(
    lambda x: x, n_outputs=1, result_to_tuple=lambda x: (x,)
)
# 定义了一个修饰器函数，用于处理轴向操作和NaN策略
def tstd(a, limits=None, inclusive=(True, True), axis=0, ddof=1):
    """Compute the trimmed sample standard deviation.

    This function finds the sample standard deviation of given values,
    ignoring values outside the given `limits`.

    Parameters
    ----------
    a : array_like
        Array of values.
    limits : None or (lower limit, upper limit), optional
        Values in the input array less than the lower limit or greater than the
        upper limit will be ignored. When limits is None, then all values are
        used. Either of the limit values in the tuple can also be None
        representing a half-open interval.  The default value is None.
    inclusive : (bool, bool), optional
        A tuple consisting of the (lower flag, upper flag).  These flags
        determine whether values exactly equal to the lower or upper limits
        are included.  The default value is (True, True).
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over the
        whole array `a`.

    """
    # 函数用于计算给定值的修剪样本标准差，忽略超出给定限制范围的值
    pass
    # ddof : int, optional
    # 定义自由度修正因子，默认值为1。

    # Returns
    # -------
    # tstd : float
    # 返回修剪后的样本标准差。

    # Notes
    # -----
    # `tstd` 计算无偏样本标准差，即使用校正因子 ``n / (n - 1)``。

    # Examples
    # --------
    # >>> import numpy as np
    # >>> from scipy import stats
    # >>> x = np.arange(20)
    # >>> stats.tstd(x)
    # 5.9160797830996161
    # >>> stats.tstd(x, (3,17))
    # 4.4721359549995796
@_axis_nan_policy_factory(
    lambda x: x, n_outputs=1, result_to_tuple=lambda x: (x,)
)
def tsem(a, limits=None, inclusive=(True, True), axis=0, ddof=1):
    """Compute the trimmed standard error of the mean.

    This function finds the standard error of the mean for given
    values, ignoring values outside the given `limits`.

    Parameters
    ----------
    a : array_like
        Array of values.
    limits : None or (lower limit, upper limit), optional
        Values in the input array less than the lower limit or greater than the
        upper limit will be ignored. When limits is None, then all values are
        used. Either of the limit values in the tuple can also be None
        representing a half-open interval.  The default value is None.
    inclusive : (bool, bool), optional
        A tuple consisting of the (lower flag, upper flag).  These flags
        determine whether values exactly equal to the lower or upper limits
        are included.  The default value is (True, True).
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over the
        whole array `a`.
    ddof : int, optional
        Delta degrees of freedom.  Default is 1.

    Returns
    -------
    tsem : float
        Trimmed standard error of the mean.

    Notes
    -----
    `tsem` uses unbiased sample standard deviation, i.e. it uses a
    correction factor ``n / (n - 1)``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> x = np.arange(20)
    >>> stats.tsem(x)
    1.3228756555322954
    >>> stats.tsem(x, (3,17))
    1.1547005383792515

    """
    xp = array_namespace(a)
    # Adjust `a` to conform to specified `limits` and `inclusive` criteria
    a, _ = _put_val_to_limits(a, limits, inclusive, xp=xp)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SmallSampleWarning)
        # Calculate the standard deviation (`sd`) of `a` with specified `ddof`
        # and handle NaN values with `nan_policy='omit'`
        sd = _xp_var(a, correction=ddof, axis=axis, nan_policy='omit', xp=xp)**0.5

    # Count non-NaN observations along `axis` and compute `tsem` as sd divided by
    # the square root of the number of observations
    n_obs = xp.sum(~xp.isnan(a), axis=axis, dtype=sd.dtype)
    return sd / n_obs**0.5


#####################################
#              MOMENTS              #
#####################################


def _moment_outputs(kwds):
    # Extract and return the number of elements in the 'order' field of `kwds`
    order = np.atleast_1d(kwds.get('order', 1))
    if order.size == 0:
        raise ValueError("'order' must be a scalar or a non-empty 1D "
                         "list/array.")
    return len(order)


def _moment_result_object(*args):
    # Return `args` as a numpy array if there's more than one argument, otherwise
    # return the single argument as-is
    if len(args) == 1:
        return args[0]
    return np.asarray(args)

# `moment` fits into the `_axis_nan_policy` pattern, but it is a bit unusual
# because the number of outputs is variable. Specifically,
# `result_to_tuple=lambda x: (x,)` may be surprising for a function that
# can produce more than one output, but it is intended here.
# When `moment is called to produce the output:
# - `_rename_parameter` decorator renames the `moment` parameter to `order`.
# - `_axis_nan_policy_factory` is a decorator that customizes the behavior of
#   nan handling and the output structure of the function `_moment_result_object`.
#   It specifies:
#   - `result_to_tuple` function that wraps the returned array into a single-element tuple.
#   - `n_outputs` specifies the number of outputs expected from `_moment_result_object`.
# - The function `moment` calculates the nth moment about the mean for a given array `a`.
# - It accepts several optional parameters:
#   - `order`: Specifies the order of the moment to be calculated.
#   - `axis`: Specifies the axis along which the moment is computed.
#   - `nan_policy`: Defines how NaN (Not a Number) values in the input are handled.
#   - `center`: Specifies the point about which moments are calculated; defaults to the sample mean.
# - Returns the nth moment about the `center`, either as a ndarray or float,
#   depending on the input dimensions and axis.
# - This function is useful for calculating coefficients like skewness and kurtosis.
@_rename_parameter('moment', 'order')
@_axis_nan_policy_factory(
    _moment_result_object, n_samples=1, result_to_tuple=lambda x: (x,),
    n_outputs=_moment_outputs
)
def moment(a, order=1, axis=0, nan_policy='propagate', *, center=None):
    r"""Calculate the nth moment about the mean for a sample.

    A moment is a specific quantitative measure of the shape of a set of
    points. It is often used to calculate coefficients of skewness and kurtosis
    due to its close relationship with them.

    Parameters
    ----------
    a : array_like
       Input array.
    order : int or 1-D array_like of ints, optional
       Order of central moment that is returned. Default is 1.
    axis : int or None, optional
       Axis along which the central moment is computed. Default is 0.
       If None, compute over the whole array `a`.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):
          * 'propagate': returns nan
          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values
    center : float or None, optional
       The point about which moments are taken. This can be the sample mean,
       the origin, or any other be point. If `None` (default) compute the
       center as the sample mean.

    Returns
    -------
    n-th moment about the `center` : ndarray or float
       The appropriate moment along the given axis or over all values if axis
       is None. The denominator for the moment calculation is the number of
       observations, no degrees of freedom correction is done.

    See Also
    --------
    kurtosis, skew, describe

    Notes
    -----
    The k-th moment of a data sample is:

    .. math::

        m_k = \frac{1}{n} \sum_{i = 1}^n (x_i - c)^k
    xp = array_namespace(a)
    # 使用 array_namespace 函数创建适当的数组命名空间
    a, axis = _chk_asarray(a, axis, xp=xp)
    # 将输入数组 `a` 和轴参数 `axis` 转换为数组，使用 xp 表示数组命名空间

    if xp.isdtype(a.dtype, 'integral'):
        # 如果 `a` 的数据类型是整数类型，将其转换为 float64 类型
        a = xp.asarray(a, dtype=xp.float64)
    else:
        # 否则，将 `a` 转换为 xp 命名空间中对应的数组类型
        a = xp.asarray(a)

    order = xp.asarray(order, dtype=a.dtype)
    if xp_size(order) == 0:
        # 如果 `order` 的大小为 0，抛出值错误异常，要求 `order` 必须是标量或非空的 1D 列表/数组
        raise ValueError("'order' must be a scalar or a non-empty 1D list/array.")
    if xp.any(order != xp.round(order)):
        # 如果 `order` 中的任何元素不是整数，抛出值错误异常，要求所有 `order` 元素必须是整数
        raise ValueError("All elements of `order` must be integral.")
    order = order[()] if order.ndim == 0 else order
    # 如果 `order` 是标量，则转换为 Python 标量，否则保持原样

    # 对于数组样式的 `order` 输入，为每个返回一个值
    if order.ndim > 0:
        # 计算均值时，仅在需要时计算一次，并且仅当 `center` 未指定且 `order` 中有大于 1 的值时
        calculate_mean = center is None and xp.any(order > 1)
        mean = xp.mean(a, axis=axis, keepdims=True) if calculate_mean else None
        mmnt = []
        for i in order:
            if center is None and i > 1:
                # 如果 `center` 未指定且 `i` 大于 1，则计算带有均值的矩
                mmnt.append(_moment(a, i, axis, mean=mean)[np.newaxis, ...])
            else:
                # 否则，计算以 `center` 为中心的矩
                mmnt.append(_moment(a, i, axis, mean=center)[np.newaxis, ...])
        return xp.concat(mmnt, axis=0)
    else:
        # 否则，计算给定 `order` 的矩，带有指定的中心
        return _moment(a, order, axis, mean=center)
def _moment(a, order, axis, *, mean=None, xp=None):
    """Vectorized calculation of raw moment about specified center

    When `mean` is None, the mean is computed and used as the center;
    otherwise, the provided value is used as the center.

    """
    xp = array_namespace(a) if xp is None else xp  # 确定使用的数组命名空间

    if xp.isdtype(a.dtype, 'integral'):
        a = xp.asarray(a, dtype=xp.float64)  # 将输入数组转换为 float64 类型

    dtype = a.dtype

    # moment of empty array is the same regardless of order
    if xp_size(a) == 0:
        return xp.mean(a, axis=axis)  # 对于空数组，返回沿指定轴的平均值

    if order == 0 or (order == 1 and mean is None):
        # By definition the zeroth moment is always 1, and the first *central*
        # moment is 0.
        shape = list(a.shape)
        del shape[axis]

        temp = (xp.ones(shape, dtype=dtype) if order == 0
                else xp.zeros(shape, dtype=dtype))
        return temp[()] if temp.ndim == 0 else temp  # 返回零阶或中心矩的计算结果

    # Exponentiation by squares: form exponent sequence
    n_list = [order]
    current_n = order
    while current_n > 2:
        if current_n % 2:
            current_n = (current_n - 1) / 2
        else:
            current_n /= 2
        n_list.append(current_n)  # 生成指数序列用于二次方计算

    # Starting point for exponentiation by squares
    mean = (xp.mean(a, axis=axis, keepdims=True) if mean is None
            else xp.asarray(mean, dtype=dtype))
    mean = mean[()] if mean.ndim == 0 else mean  # 计算均值并处理为标量
    a_zero_mean = a - mean  # 将数据减去均值得到零均值数据

    eps = xp.finfo(dtype).eps * 10  # 获取数据类型的机器精度

    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = xp.max(xp.abs(a_zero_mean), axis=axis,
                          keepdims=True) / xp.abs(mean)  # 计算相对差异

    with np.errstate(invalid='ignore'):
        precision_loss = xp.any(rel_diff < eps)  # 检查是否存在精度损失

    n = a.shape[axis] if axis is not None else xp_size(a)
    if precision_loss and n > 1:
        message = ("Precision loss occurred in moment calculation due to "
                   "catastrophic cancellation. This occurs when the data "
                   "are nearly identical. Results may be unreliable.")
        warnings.warn(message, RuntimeWarning, stacklevel=4)  # 如果存在精度损失则发出警告

    if n_list[-1] == 1:
        s = xp.asarray(a_zero_mean, copy=True)  # 如果最后一个指数是1，则直接使用零均值数据
    else:
        s = a_zero_mean**2  # 否则计算零均值数据的平方

    # Perform multiplications
    for n in n_list[-2::-1]:  # 对于剩余的指数，执行平方乘法
        s = s**2
        if n % 2:
            s *= a_zero_mean
    return xp.mean(s, axis=axis)  # 返回沿指定轴的平均值


def _var(x, axis=0, ddof=0, mean=None, xp=None):
    # Calculate variance of sample, warning if precision is lost
    xp = array_namespace(x) if xp is None else xp  # 确定使用的数组命名空间
    var = _moment(x, 2, axis, mean=mean, xp=xp)  # 计算二阶矩作为方差的计算基础
    if ddof != 0:
        n = x.shape[axis] if axis is not None else xp_size(x)
        var *= np.divide(n, n-ddof)  # 调整自由度以避免除零错误
    return var  # 返回方差值


@_axis_nan_policy_factory(
    lambda x: x, result_to_tuple=lambda x: (x,), n_outputs=1
)
# nan_policy handled by `_axis_nan_policy`, but needs to be left
# in signature to preserve use as a positional argument
# 导入所需的库或模块
from scipy.stats import skew

# 计算给定数据集的样本偏度
def skew(a, axis=0, bias=True, nan_policy='propagate'):
    r"""Compute the sample skewness of a data set.

    For normally distributed data, the skewness should be about zero. For
    unimodal continuous distributions, a skewness value greater than zero means
    that there is more weight in the right tail of the distribution. The
    function `skewtest` can be used to determine if the skewness value
    is close enough to zero, statistically speaking.

    Parameters
    ----------
    a : ndarray
        Input array.
    axis : int or None, optional
        Axis along which skewness is calculated. Default is 0.
        If None, compute over the whole array `a`.
    bias : bool, optional
        If False, then the calculations are corrected for statistical bias.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

          * 'propagate': returns nan
          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values

    Returns
    -------
    skewness : ndarray
        The skewness of values along an axis, returning NaN where all values
        are equal.

    Notes
    -----
    The sample skewness is computed as the Fisher-Pearson coefficient
    of skewness, i.e.

    .. math::

        g_1=\frac{m_3}{m_2^{3/2}}

    where

    .. math::

        m_i=\frac{1}{N}\sum_{n=1}^N(x[n]-\bar{x})^i

    is the biased sample :math:`i\texttt{th}` central moment, and
    :math:`\bar{x}` is
    the sample mean.  If ``bias`` is False, the calculations are
    corrected for bias and the value computed is the adjusted
    Fisher-Pearson standardized moment coefficient, i.e.

    .. math::

        G_1=\frac{k_3}{k_2^{3/2}}=
            \frac{\sqrt{N(N-1)}}{N-2}\frac{m_3}{m_2^{3/2}}.

    References
    ----------
    .. [1] Zwillinger, D. and Kokoska, S. (2000). CRC Standard
       Probability and Statistics Tables and Formulae. Chapman & Hall: New
       York. 2000.
       Section 2.2.24.1

    Examples
    --------
    >>> from scipy.stats import skew
    >>> skew([1, 2, 3, 4, 5])
    0.0
    >>> skew([2, 8, 0, 4, 1, 9, 9, 0])
    0.2650554122698573

    """
    # 使用适当的数组命名空间
    xp = array_namespace(a)
    # 调用 _chk_asarray 函数，将输入数组转换为 ndarray 类型并处理轴向
    a, axis = _chk_asarray(a, axis, xp=xp)
    # 获取数组在指定轴向上的维度大小
    n = a.shape[axis]

    # 计算沿指定轴向的均值，保持维度信息
    mean = xp.mean(a, axis=axis, keepdims=True)
    # 在后续计算中可能需要，去除均值后的数组
    mean_reduced = xp.squeeze(mean, axis=axis)
    # 计算二阶中心矩
    m2 = _moment(a, 2, axis, mean=mean, xp=xp)
    # 计算三阶中心矩
    m3 = _moment(a, 3, axis, mean=mean, xp=xp)
    
    # 忽略错误状态下的操作
    with np.errstate(all='ignore'):
        # 获取给定数据类型的 epsilon
        eps = xp.finfo(m2.dtype).eps
        # 判断是否需要处理零值
        zero = m2 <= (eps * mean_reduced)**2
        # 根据条件进行值的替换
        vals = xp.where(zero, xp.asarray(xp.nan), m3 / m2**1.5)
    
    # 如果 bias 参数为 False，对结果进行修正
    if not bias:
        # 判断是否可以修正
        can_correct = ~zero & (n > 2)
        if xp.any(can_correct):
            # 获取需要修正的中心矩
            m2 = m2[can_correct]
            m3 = m3[can_correct]
            # 计算修正后的值
            nval = ((n - 1.0) * n)**0.5 / (n - 2.0) * m3 / m2**1.5
            # 将修正后的值赋给对应位置
            vals[can_correct] = nval
    # 如果 `vals` 的维度是 0，则返回它的零维标量值；否则返回 `vals` 自身。
    return vals[()] if vals.ndim == 0 else vals
@_axis_nan_policy_factory(
    lambda x: x, result_to_tuple=lambda x: (x,), n_outputs=1
)
# 使用 `_axis_nan_policy_factory` 创建一个装饰器，用于处理轴向的 NaN 策略
# `nan_policy` 参数保留在函数签名中，以便作为位置参数使用
def kurtosis(a, axis=0, fisher=True, bias=True, nan_policy='propagate'):
    """Compute the kurtosis (Fisher or Pearson) of a dataset.

    Kurtosis is the fourth central moment divided by the square of the
    variance. If Fisher's definition is used, then 3.0 is subtracted from
    the result to give 0.0 for a normal distribution.

    If bias is False then the kurtosis is calculated using k statistics to
    eliminate bias coming from biased moment estimators

    Use `kurtosistest` to see if result is close enough to normal.

    Parameters
    ----------
    a : array
        Data for which the kurtosis is calculated.
    axis : int or None, optional
        Axis along which the kurtosis is calculated. Default is 0.
        If None, compute over the whole array `a`.
    fisher : bool, optional
        If True, Fisher's definition is used (normal ==> 0.0). If False,
        Pearson's definition is used (normal ==> 3.0).
    bias : bool, optional
        If False, then the calculations are corrected for statistical bias.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan. 'propagate' returns nan,
        'raise' throws an error, 'omit' performs the calculations ignoring nan
        values. Default is 'propagate'.

    Returns
    -------
    kurtosis : array
        The kurtosis of values along an axis, returning NaN where all values
        are equal.

    References
    ----------
    .. [1] Zwillinger, D. and Kokoska, S. (2000). CRC Standard
       Probability and Statistics Tables and Formulae. Chapman & Hall: New
       York. 2000.

    Examples
    --------
    In Fisher's definition, the kurtosis of the normal distribution is zero.
    In the following example, the kurtosis is close to zero, because it was
    calculated from the dataset, not from the continuous distribution.

    >>> import numpy as np
    >>> from scipy.stats import norm, kurtosis
    >>> data = norm.rvs(size=1000, random_state=3)
    >>> kurtosis(data)
    -0.06928694200380558

    The distribution with a higher kurtosis has a heavier tail.
    The zero valued kurtosis of the normal distribution in Fisher's definition
    can serve as a reference point.

    >>> import matplotlib.pyplot as plt
    >>> import scipy.stats as stats
    >>> from scipy.stats import kurtosis

    >>> x = np.linspace(-5, 5, 100)
    >>> ax = plt.subplot()
    >>> distnames = ['laplace', 'norm', 'uniform']

    >>> for distname in distnames:
    ...     if distname == 'uniform':
    ...         dist = getattr(stats, distname)(loc=-2, scale=4)
    ...     else:
    ...         dist = getattr(stats, distname)
    ...     data = dist.rvs(size=1000)
    ...     kur = kurtosis(data, fisher=True)
    ...     y = dist.pdf(x)
    # 绘制图表，使用给定的数据 x 和 y，并添加标签
    ax.plot(x, y, label="{}, {}".format(distname, round(kur, 3)))
    # 在图表上添加图例
    ax.legend()

    # Laplace 分布的尾部比正态分布更重。
    # 均匀分布（具有负峰度）的尾部最薄。

    """
    xp = array_namespace(a)  # 将数组 a 转换为指定命名空间中的数组
    a, axis = _chk_asarray(a, axis, xp=xp)  # 将数组 a 确认为数组，并确定轴

    n = a.shape[axis]  # 获取数组在指定轴上的长度
    mean = xp.mean(a, axis=axis, keepdims=True)  # 计算数组沿指定轴的均值，并保持维度
    mean_reduced = xp.squeeze(mean, axis=axis)  # 将均值数组在指定轴上去除多余的维度，用于后续操作
    m2 = _moment(a, 2, axis, mean=mean, xp=xp)  # 计算数组的二阶矩
    m4 = _moment(a, 4, axis, mean=mean, xp=xp)  # 计算数组的四阶矩
    with np.errstate(all='ignore'):
        zero = m2 <= (xp.finfo(m2.dtype).eps * mean_reduced)**2  # 确定阶矩是否接近于零
        NaN = _get_nan(m4, xp=xp)  # 获取 NaN 值
        vals = xp.where(zero, NaN, m4 / m2**2.0)  # 计算峰度值

    if not bias:
        can_correct = ~zero & (n > 3)  # 确定是否可以进行修正
        if xp.any(can_correct):
            m2 = m2[can_correct]
            m4 = m4[can_correct]
            nval = 1.0/(n-2)/(n-3) * ((n**2-1.0)*m4/m2**2.0 - 3*(n-1)**2.0)  # 计算修正后的峰度值
            vals[can_correct] = nval + 3.0

    vals = vals - 3 if fisher else vals  # 根据 Fisher 要求修正峰度值
    return vals[()] if vals.ndim == 0 else vals  # 返回修正后的峰度值数组或单个值
# 使用 namedtuple 定义 DescribeResult 类型，用于存储描述统计结果
DescribeResult = namedtuple('DescribeResult',
                            ('nobs', 'minmax', 'mean', 'variance', 'skewness',
                             'kurtosis'))


# 定义描述统计函数 describe，计算传入数组的多个描述统计量
def describe(a, axis=0, ddof=1, bias=True, nan_policy='propagate'):
    """Compute several descriptive statistics of the passed array.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : int or None, optional
        Axis along which statistics are calculated. Default is 0.
        If None, compute over the whole array `a`.
    ddof : int, optional
        Delta degrees of freedom (only for variance).  Default is 1.
    bias : bool, optional
        If False, then the skewness and kurtosis calculations are corrected
        for statistical bias.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

        * 'propagate': returns nan
        * 'raise': throws an error
        * 'omit': performs the calculations ignoring nan values

    Returns
    -------
    nobs : int or ndarray of ints
        Number of observations (length of data along `axis`).
        When 'omit' is chosen as nan_policy, the length along each axis
        slice is counted separately.
    minmax: tuple of ndarrays or floats
        Minimum and maximum value of `a` along the given axis.
    mean : ndarray or float
        Arithmetic mean of `a` along the given axis.
    variance : ndarray or float
        Unbiased variance of `a` along the given axis; denominator is number
        of observations minus one.
    skewness : ndarray or float
        Skewness of `a` along the given axis, based on moment calculations
        with denominator equal to the number of observations, i.e. no degrees
        of freedom correction.
    kurtosis : ndarray or float
        Kurtosis (Fisher) of `a` along the given axis.  The kurtosis is
        normalized so that it is zero for the normal distribution.  No
        degrees of freedom are used.

    See Also
    --------
    skew, kurtosis

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> a = np.arange(10)
    >>> stats.describe(a)
    DescribeResult(nobs=10, minmax=(0, 9), mean=4.5,
                   variance=9.166666666666666, skewness=0.0,
                   kurtosis=-1.2242424242424244)
    >>> b = [[1, 2], [3, 4]]
    >>> stats.describe(b)
    DescribeResult(nobs=2, minmax=(array([1, 2]), array([3, 4])),
                   mean=array([2., 3.]), variance=array([2., 2.]),
                   skewness=array([0., 0.]), kurtosis=array([-2., -2.]))

    """
    
    # 导入 array_namespace 函数，并根据传入数组创建 xp 对象
    xp = array_namespace(a)
    # 对传入数组进行处理，返回处理后的数组 a 和处理后的轴 axis
    a, axis = _chk_asarray(a, axis, xp=xp)

    # 检查数组中是否包含 NaN 值，并根据 nan_policy 设置处理策略
    contains_nan, nan_policy = _contains_nan(a, nan_policy)
    # 检查是否存在 NaN 并且处理策略为 'omit'
    if contains_nan and nan_policy == 'omit':
        # 如果包含 NaN 并且策略为 'omit'，则使用 NumPy 创建掩码对象，并返回基本描述统计信息
        a = ma.masked_invalid(a)
        return mstats_basic.describe(a, axis, ddof, bias)

    # 如果输入数组 a 的尺寸为 0，则抛出值错误异常
    if xp_size(a) == 0:
        raise ValueError("The input must not be empty.")

    # 计算数组 a 在指定轴上的形状大小
    n = a.shape[axis]
    # 计算数组 a 在指定轴上的最小值和最大值，返回一个元组
    mm = (xp.min(a, axis=axis), xp.max(a, axis=axis))
    # 计算数组 a 在指定轴上的均值
    m = xp.mean(a, axis=axis)
    # 计算数组 a 在指定轴上的方差
    v = _var(a, axis=axis, ddof=ddof, xp=xp)
    # 计算数组 a 在指定轴上的偏度
    sk = skew(a, axis, bias=bias)
    # 计算数组 a 在指定轴上的峰度
    kurt = kurtosis(a, axis, bias=bias)

    # 返回描述统计信息的 DescribeResult 对象，包括样本数量、最小值和最大值元组、均值、方差、偏度、峰度
    return DescribeResult(n, mm, m, v, sk, kurt)
#####################################
#         NORMALITY TESTS           #
#####################################

# 定义一个命名元组，用于表示偏度测试结果，包含统计量和 p 值
SkewtestResult = namedtuple('SkewtestResult', ('statistic', 'pvalue'))


# 用装饰器修饰的函数，用于测试样本数据的偏度是否与正态分布不同
@_axis_nan_policy_factory(SkewtestResult, n_samples=1, too_small=7)
# nan_policy 参数由 `_axis_nan_policy` 处理，但为了保留作为位置参数的使用，需要保留在签名中
def skewtest(a, axis=0, nan_policy='propagate', alternative='two-sided'):
    r"""Test whether the skew is different from the normal distribution.

    This function tests the null hypothesis that the skewness of
    the population that the sample was drawn from is the same
    as that of a corresponding normal distribution.

    Parameters
    ----------
    a : array
        The data to be tested. Must contain at least eight observations.
    axis : int or None, optional
       Axis along which statistics are calculated. Default is 0.
       If None, compute over the whole array `a`.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

        * 'propagate': returns nan
        * 'raise': throws an error
        * 'omit': performs the calculations ignoring nan values

    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        The following options are available:

        * 'two-sided': the skewness of the distribution underlying the sample
          is different from that of the normal distribution (i.e. 0)
        * 'less': the skewness of the distribution underlying the sample
          is less than that of the normal distribution
        * 'greater': the skewness of the distribution underlying the sample
          is greater than that of the normal distribution

        .. versionadded:: 1.7.0

    Returns
    -------
    statistic : float
        The computed z-score for this test.
    pvalue : float
        The p-value for the hypothesis test.

    Notes
    -----
    The sample size must be at least 8.

    References
    ----------

    """
    # 如果 xp 参数为 None，则使用 statistic 构建 array_namespace
    xp = array_namespace(statistic) if xp is None else xp

    # 根据 alternative 参数计算 p 值
    if alternative == 'less':
        pvalue = distribution.cdf(statistic)
    elif alternative == 'greater':
        pvalue = distribution.sf(statistic)
    elif alternative == 'two-sided':
        pvalue = 2 * (distribution.sf(xp.abs(statistic)) if symmetric
                      else xp_minimum(distribution.cdf(statistic),
                                      distribution.sf(statistic),
                                      xp=xp))
    else:
        # 如果 alternative 参数不是 'less', 'greater', 'two-sided' 中的一个，抛出 ValueError 异常
        message = "`alternative` must be 'less', 'greater', or 'two-sided'."
        raise ValueError(message)

    return pvalue
    # 引用的文献参考，提供了关于正态性检验和统计分布的背景信息
    # [1] R. B. D'Agostino, A. J. Belanger and R. B. D'Agostino Jr.,
    #         "A suggestion for using powerful and informative tests of
    #         normality", American Statistician 44, pp. 316-321, 1990.
    # [2] Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test
    #        for normality (complete samples). Biometrika, 52(3/4), 591-611.
    # [3] B. Phipson and G. K. Smyth. "Permutation P-values Should Never Be
    #        Zero: Calculating Exact P-values When Permutations Are Randomly
    #        Drawn." Statistical Applications in Genetics and Molecular Biology
    #        9.1 (2010).
    
    # 示例
    # --------
    # 假设我们希望从测量中推断医学研究中成年男性的体重是否不服从正态分布 [2]_。
    # 体重（磅）记录在下面的数组 ``x`` 中。
    >>> import numpy as np
    >>> x = np.array([148, 154, 158, 160, 161, 162, 166, 170, 182, 195, 236])
    
    # 从 [1]_ 中的偏度测试开始，计算基于样本偏度的统计量。
    >>> from scipy import stats
    >>> res = stats.skewtest(x)
    >>> res.statistic
    2.7788579769903414
    
    # 因为正态分布的偏度为零，所以对于从正态分布抽取的样本，此统计量的大小趋向于较小。
    
    # 该测试通过比较观察到的统计量值与零假设的空分布进行。
    # 零假设是体重来自正态分布的统计量值分布。
    
    # 对于此测试，统计量在非常大的样本下的空分布是标准正态分布。
    >>> import matplotlib.pyplot as plt
    >>> dist = stats.norm()
    >>> st_val = np.linspace(-5, 5, 100)
    >>> pdf = dist.pdf(st_val)
    >>> fig, ax = plt.subplots(figsize=(8, 5))
    >>> def st_plot(ax):  # we'll reuse this
    ...     ax.plot(st_val, pdf)
    ...     ax.set_title("Skew Test Null Distribution")
    ...     ax.set_xlabel("statistic")
    ...     ax.set_ylabel("probability density")
    >>> st_plot(ax)
    >>> plt.show()
    
    # 比较通过 p 值量化：在一个双侧测试中，空分布中大于观察统计量的值和小于观察统计量的负值都被认为是“更极端的”。
    >>> fig, ax = plt.subplots(figsize=(8, 5))
    >>> st_plot(ax)
    >>> pvalue = dist.cdf(-res.statistic) + dist.sf(res.statistic)
    >>> annotation = (f'p-value={pvalue:.3f}\n(shaded area)')
    >>> props = dict(facecolor='black', width=1, headwidth=5, headlength=8)
    >>> _ = ax.annotate(annotation, (3, 0.005), (3.25, 0.02), arrowprops=props)
    >>> i = st_val >= res.statistic
    >>> ax.fill_between(st_val[i], y1=0, y2=pdf[i], color='C0')
    >>> i = st_val <= -res.statistic
    # 检查 st_val 是否小于等于 res.statistic 的负值，返回布尔数组 i

    >>> ax.fill_between(st_val[i], y1=0, y2=pdf[i], color='C0')
    # 在图形 ax 中，根据条件 i 对 st_val 的部分区域进行填充，y1=0 是填充下限，y2=pdf[i] 是填充上限，使用颜色 'C0'

    >>> ax.set_xlim(-5, 5)
    # 设置图形 ax 的 x 轴显示范围为 [-5, 5]

    >>> ax.set_ylim(0, 0.1)
    # 设置图形 ax 的 y 轴显示范围为 [0, 0.1]

    >>> plt.show()
    # 显示图形 plt，即前面定义的图形

    >>> res.pvalue
    # 返回变量 res 的 p-value 值，表示统计检验的显著性水平

    0.005455036974740185
    # p-value 的具体数值，用于判断是否拒绝原假设

    If the p-value is "small" - that is, if there is a low probability of
    sampling data from a normally distributed population that produces such an
    extreme value of the statistic - this may be taken as evidence against
    the null hypothesis in favor of the alternative: the weights were not
    drawn from a normal distribution. Note that:

    - The inverse is not true; that is, the test is not used to provide
      evidence for the null hypothesis.
    - The threshold for values that will be considered "small" is a choice that
      should be made before the data is analyzed [3]_ with consideration of the
      risks of both false positives (incorrectly rejecting the null hypothesis)
      and false negatives (failure to reject a false null hypothesis).

    Note that the standard normal distribution provides an asymptotic
    approximation of the null distribution; it is only accurate for samples
    with many observations. For small samples like ours,
    `scipy.stats.monte_carlo_test` may provide a more accurate, albeit
    stochastic, approximation of the exact p-value.

    >>> def statistic(x, axis):
    ...     # get just the skewtest statistic; ignore the p-value
    ...     return stats.skewtest(x, axis=axis).statistic
    # 定义函数 statistic，用于计算 skewtest 的统计量，忽略 p-value

    >>> res = stats.monte_carlo_test(x, stats.norm.rvs, statistic)
    # 进行 Monte Carlo 模拟检验，计算与正态分布随机变量相关的统计量 res

    >>> fig, ax = plt.subplots(figsize=(8, 5))
    # 创建新的图形 fig 和坐标轴 ax，设置图形大小为 (8, 5)

    >>> st_plot(ax)
    # 调用 st_plot 函数，将数据绘制在 ax 上

    >>> ax.hist(res.null_distribution, np.linspace(-5, 5, 50),
    ...         density=True)
    # 在图形 ax 上绘制 res.null_distribution 的直方图，bin 设置为 np.linspace(-5, 5, 50)，进行密度归一化

    >>> ax.legend(['aymptotic approximation\n(many observations)',
    ...            'Monte Carlo approximation\n(11 observations)'])
    # 在图形 ax 中添加图例，显示两种近似方法的标签

    >>> plt.show()
    # 显示图形 plt，即前面定义的图形

    >>> res.pvalue
    # 返回变量 res 的 p-value 值，表示统计检验的显著性水平

    0.0062  # may vary
    # p-value 的具体数值，用于判断是否拒绝原假设

    In this case, the asymptotic approximation and Monte Carlo approximation
    agree fairly closely, even for our small sample.

    """
    xp = array_namespace(a)
    # 使用 array_namespace 将 a 转换为 xp

    a, axis = _chk_asarray(a, axis, xp=xp)
    # 检查并将 a 和 axis 转换为数组格式，使用 xp 命名空间

    b2 = skew(a, axis, _no_deco=True)
    # 计算数组 a 在轴 axis 上的偏度，返回 b2

    n = a.shape[axis]
    # 获取数组 a 在轴 axis 上的长度，赋值给 n

    if n < 8:
        message = ("`skewtest` requires at least 8 observations; "
                   f"only {n=} observations were given.")
        raise ValueError(message)
    # 如果 n 小于 8，则抛出 ValueError 异常，提示样本数量不足

    y = b2 * math.sqrt(((n + 1) * (n + 3)) / (6.0 * (n - 2)))
    # 根据偏度 b2 计算 y 值

    beta2 = (3.0 * (n**2 + 27*n - 70) * (n+1) * (n+3) /
             ((n-2.0) * (n+5) * (n+7) * (n+9)))
    # 计算 beta2

    W2 = -1 + math.sqrt(2 * (beta2 - 1))
    # 计算 W2

    delta = 1 / math.sqrt(0.5 * math.log(W2))
    # 计算 delta

    alpha = math.sqrt(2.0 / (W2 - 1))
    # 计算 alpha

    y = xp.where(y == 0, xp.asarray(1, dtype=y.dtype), y)
    # 将 y 中等于 0 的元素替换为 1

    Z = delta * xp.log(y / alpha + xp.sqrt((y / alpha)**2 + 1))
    # 计算 Z 值

    pvalue = _get_pvalue(Z, _SimpleNormal(), alternative, xp=xp)
    # 调用 _get_pvalue 函数计算 p-value

    Z = Z[()] if Z.ndim == 0 else Z
    # 如果 Z 是标量，转换为标量形式

    pvalue = pvalue[()] if pvalue.ndim == 0 else pvalue
    # 如果 pvalue 是标量，转换为标量形式

    return SkewtestResult(Z, pvalue)
    # 返回 SkewtestResult 对象，包含 Z 值和 p-value
# 使用 namedtuple 创建一个命名元组类型 KurtosistestResult，包含 statistic 和 pvalue 两个字段
KurtosistestResult = namedtuple('KurtosistestResult', ('statistic', 'pvalue'))

# 定义修饰符函数 @_axis_nan_policy_factory，返回一个装饰后的 kurtosistest 函数
@_axis_nan_policy_factory(KurtosistestResult, n_samples=1, too_small=4)
# kurtosistest 函数用于测试数据集的峰度是否符合正态分布
def kurtosistest(a, axis=0, nan_policy='propagate', alternative='two-sided'):
    r"""Test whether a dataset has normal kurtosis.

    This function tests the null hypothesis that the kurtosis
    of the population from which the sample was drawn is that
    of the normal distribution.

    Parameters
    ----------
    a : array
        Array of the sample data. Must contain at least five observations.
    axis : int or None, optional
       Axis along which to compute test. Default is 0. If None,
       compute over the whole array `a`.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

        * 'propagate': returns nan
        * 'raise': throws an error
        * 'omit': performs the calculations ignoring nan values
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

        * 'two-sided': the kurtosis of the distribution underlying the sample
          is different from that of the normal distribution
        * 'less': the kurtosis of the distribution underlying the sample
          is less than that of the normal distribution
        * 'greater': the kurtosis of the distribution underlying the sample
          is greater than that of the normal distribution

        .. versionadded:: 1.7.0

    Returns
    -------
    statistic : float
        The computed z-score for this test.
    pvalue : float
        The p-value for the hypothesis test.

    Notes
    -----
    Valid only for n>20. This function uses the method described in [1]_.

    References
    ----------
    .. [1] see e.g. F. J. Anscombe, W. J. Glynn, "Distribution of the kurtosis
       statistic b2 for normal samples", Biometrika, vol. 70, pp. 227-234, 1983.
    .. [2] Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test
           for normality (complete samples). Biometrika, 52(3/4), 591-611.
    .. [3] B. Phipson and G. K. Smyth. "Permutation P-values Should Never Be
           Zero: Calculating Exact P-values When Permutations Are Randomly
           Drawn." Statistical Applications in Genetics and Molecular Biology
           9.1 (2010).
    .. [4] Panagiotakos, D. B. (2008). The value of p-value in biomedical
           research. The open cardiovascular medicine journal, 2, 97.

    Examples
    --------
    Suppose we wish to infer from measurements whether the weights of adult
    human males in a medical study are not normally distributed [2]_.
    The weights (lbs) are recorded in the array ``x`` below.

    >>> import numpy as np
    >>> x = np.array([148, 154, 158, 160, 161, 162, 166, 170, 182, 195, 236])
    # 引入 scipy 库中的统计模块
    from scipy import stats
    
    # 对给定样本 x 进行峰度检验，返回检验结果
    res = stats.kurtosistest(x)
    
    # 输出检验统计量的数值
    res.statistic
    2.3048235214240873
    
    # 由于样本观测值数量较少，检验警告样本数量不足，将在示例末尾返回到这一点
    # 因为正态分布的样本具有零峰度（定义如此），所以统计量的幅度在从正态分布中抽取的样本中通常较低。
    
    # 此检验通过将观察到的统计量值与空假设的空分布进行比较来执行。
    # 对于该检验，对于非常大的样本，统计量的空分布是标准正态分布。
    
    # 引入 matplotlib 库中的 pyplot 模块，并将其重命名为 plt
    import matplotlib.pyplot as plt
    
    # 创建一个标准正态分布对象
    dist = stats.norm()
    
    # 在指定区间内生成一组均匀分布的值
    kt_val = np.linspace(-5, 5, 100)
    
    # 计算标准正态分布在给定值上的概率密度函数
    pdf = dist.pdf(kt_val)
    
    # 创建一个图形和轴对象，设置图形大小
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 定义一个函数 kt_plot，用于绘制峰度检验的空分布图
    def kt_plot(ax):  # 我们将重复使用这个函数
        ax.plot(kt_val, pdf)
        ax.set_title("Kurtosis Test Null Distribution")  # 设置图表标题
        ax.set_xlabel("statistic")  # 设置 x 轴标签
        ax.set_ylabel("probability density")  # 设置 y 轴标签
    
    # 调用 kt_plot 函数，绘制峰度检验的空分布图
    kt_plot(ax)
    
    # 显示图形
    plt.show()
    
    # 比较由 p 值量化的检验结果
    # p 值表示在空分布中比观察到的统计量值更极端或更极端的值的比例。
    # 在双侧检验中，如果统计量为正，则空分布中大于观察统计量的值和空分布中小于观察统计量的负值都被认为是“更极端”的。
    fig, ax = plt.subplots(figsize=(8, 5))
    kt_plot(ax)
    
    # 计算 p 值，并用注释标明阴影区域
    pvalue = dist.cdf(-res.statistic) + dist.sf(res.statistic)
    annotation = (f'p-value={pvalue:.3f}\n(shaded area)')
    props = dict(facecolor='black', width=1, headwidth=5, headlength=8)
    _ = ax.annotate(annotation, (3, 0.005), (3.25, 0.02), arrowprops=props)
    
    # 标记在统计量大于等于观察统计量的区域中填充阴影
    i = kt_val >= res.statistic
    ax.fill_between(kt_val[i], y1=0, y2=pdf[i], color='C0')
    
    # 标记在统计量小于等于观察统计量的区域中填充阴影
    i = kt_val <= -res.statistic
    ax.fill_between(kt_val[i], y1=0, y2=pdf[i], color='C0')
    
    # 设置 x 和 y 轴的限制
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 0.1)
    
    # 显示图形
    plt.show()
    
    # 输出检验结果的 p 值
    res.pvalue
    0.0211764592113868
    
    # 如果 p 值“较小”——即从正态分布人群中抽样得到的可能性很低，这可能被视为反对空假设的证据，即权重不是从正态分布中抽取的。
    # 注意：
    # - 反之不成立；即，该检验不能用来支持空假设的证据。
    xp = array_namespace(a)
    # 使用 array_namespace 函数获取数组 a 所属的命名空间 xp

    a, axis = _chk_asarray(a, axis, xp=xp)
    # 调用 _chk_asarray 函数将 a 转换为数组，并确定轴的方向，同时传递命名空间 xp

    n = a.shape[axis]
    # 获取数组 a 在指定轴上的长度 n

    if n < 5:
        # 如果数组长度 n 小于 5，抛出 ValueError 异常
        message = ("`kurtosistest` requires at least 5 observations; "
                   f"only {n=} observations were given.")
        raise ValueError(message)

    if n < 20:
        # 如果数组长度 n 小于 20，发出警告，提醒用户可能会导致 `kurtosistest` 的 p 值不准确
        message = ("`kurtosistest` p-value may be inaccurate with fewer than 20 "
                   f"observations; only {n=} observations were given.")
        warnings.warn(message, stacklevel=2)

    b2 = kurtosis(a, axis, fisher=False, _no_deco=True)
    # 计算数组 a 在指定轴上的峰度，使用 Fisher 方法计算

    E = 3.0*(n-1) / (n+1)
    # 计算 E 值，用于后续公式的计算

    varb2 = 24.0*n*(n-2)*(n-3) / ((n+1)*(n+1.)*(n+3)*(n+5))  # [1]_ Eq. 1
    # 计算 varb2 的值，使用给定的公式 [1]_ Eq. 1

    x = (b2-E) / varb2**0.5  # [1]_ Eq. 4
    # 计算 x 值，使用给定的公式 [1]_ Eq. 4

    sqrtbeta1 = 6.0*(n*n-5*n+2)/((n+7)*(n+9)) * ((6.0*(n+3)*(n+5))
                                                 / (n*(n-2)*(n-3)))**0.5
    # 计算 sqrtbeta1 的值，使用给定的公式 [1]_ Eq. 2

    A = 6.0 + 8.0/sqrtbeta1 * (2.0/sqrtbeta1 + (1+4.0/(sqrtbeta1**2))**0.5)
    # 计算 A 的值，使用给定的公式 [1]_ Eq. 3

    term1 = 1 - 2/(9.0*A)
    # 计算 term1 的值，使用给定的公式

    denom = 1 + x * (2/(A-4.0))**0.5
    # 计算 denom 的值，使用给定的公式

    NaN = _get_nan(x, xp=xp)
    # 获取 NaN 值，用于后续计算

    term2 = xp_sign(denom) * xp.where(denom == 0.0, NaN,
                                      ((1-2.0/A)/xp.abs(denom))**(1/3))
    # 计算 term2 的值，使用 xp_sign 和 xp.where 函数

    if xp.any(denom == 0):
        # 如果 denom 中存在值为 0 的情况，发出 RuntimeWarning 警告
        msg = ("Test statistic not defined in some cases due to division by "
               "zero. Return nan in that case...")
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    Z = (term1 - term2) / (2/(9.0*A))**0.5  # [1]_ Eq. 5
    # 计算 Z 值，使用给定的公式 [1]_ Eq. 5

    pvalue = _get_pvalue(Z, _SimpleNormal(), alternative, xp=xp)
    # 调用 _get_pvalue 函数计算 p 值，使用 _SimpleNormal() 作为分布模型，传递 alternative 和 xp 参数
    # 如果 Z 是零维数组（标量），则转换成 Python 标量
    Z = Z[()] if Z.ndim == 0 else Z
    # 如果 pvalue 是零维数组（标量），则转换成 Python 标量
    pvalue = pvalue[()] if pvalue.ndim == 0 else pvalue
    # 返回 KurtosistestResult 对象，其中包含 Z 和 pvalue
    return KurtosistestResult(Z, pvalue)
# 使用命名元组定义了一个新的数据类型 NormaltestResult，包含 statistic 和 pvalue 两个字段
NormaltestResult = namedtuple('NormaltestResult', ('statistic', 'pvalue'))

# 使用装饰器函数 @_axis_nan_policy_factory 包装了 normaltest 函数，配置了特定的参数
@_axis_nan_policy_factory(NormaltestResult, n_samples=1, too_small=7)
def normaltest(a, axis=0, nan_policy='propagate'):
    r"""Test whether a sample differs from a normal distribution.

    This function tests the null hypothesis that a sample comes
    from a normal distribution.  It is based on D'Agostino and
    Pearson's [1]_, [2]_ test that combines skew and kurtosis to
    produce an omnibus test of normality.

    Parameters
    ----------
    a : array_like
        The array containing the sample to be tested. Must contain
        at least eight observations.
    axis : int or None, optional
        Axis along which to compute test. Default is 0. If None,
        compute over the whole array `a`.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

          * 'propagate': returns nan
          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values

    Returns
    -------
    statistic : float or array
        ``s^2 + k^2``, where ``s`` is the z-score returned by `skewtest` and
        ``k`` is the z-score returned by `kurtosistest`.
    pvalue : float or array
       A 2-sided chi squared probability for the hypothesis test.

    References
    ----------
    .. [1] D'Agostino, R. B. (1971), "An omnibus test of normality for
           moderate and large sample size", Biometrika, 58, 341-348
    .. [2] D'Agostino, R. and Pearson, E. S. (1973), "Tests for departure from
           normality", Biometrika, 60, 613-622
    .. [3] Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test
           for normality (complete samples). Biometrika, 52(3/4), 591-611.
    .. [4] B. Phipson and G. K. Smyth. "Permutation P-values Should Never Be
           Zero: Calculating Exact P-values When Permutations Are Randomly
           Drawn." Statistical Applications in Genetics and Molecular Biology
           9.1 (2010).
    .. [5] Panagiotakos, D. B. (2008). The value of p-value in biomedical
           research. The open cardiovascular medicine journal, 2, 97.

    Examples
    --------
    Suppose we wish to infer from measurements whether the weights of adult
    human males in a medical study are not normally distributed [3]_.
    The weights (lbs) are recorded in the array ``x`` below.

    >>> import numpy as np
    >>> x = np.array([148, 154, 158, 160, 161, 162, 166, 170, 182, 195, 236])

    The normality test of [1]_ and [2]_ begins by computing a statistic based
    on the sample skewness and kurtosis.

    >>> from scipy import stats
    >>> res = stats.normaltest(x)
    >>> res.statistic
    13.034263121192582

    (The test warns that our sample has too few observations to perform the
    test. We'll return to this at the end of the example.)
    Because the normal distribution has zero skewness and zero
    ("excess" or "Fisher") kurtosis, the value of this statistic tends to be
    low for samples drawn from a normal distribution.

    The test is performed by comparing the observed value of the statistic
    against the null distribution: the distribution of statistic values derived
    under the null hypothesis that the weights were drawn from a normal
    distribution.
    For this normality test, the null distribution for very large samples is
    the chi-squared distribution with two degrees of freedom.

    >>> import matplotlib.pyplot as plt
    >>> dist = stats.chi2(df=2)  # 创建一个自由度为2的卡方分布对象
    >>> stat_vals = np.linspace(0, 16, 100)  # 生成从0到16的100个等间距数值
    >>> pdf = dist.pdf(stat_vals)  # 计算卡方分布在给定数值上的概率密度函数值
    >>> fig, ax = plt.subplots(figsize=(8, 5))  # 创建一个8x5尺寸的图形对象和子图对象
    >>> def plot(ax):  # 定义一个用于绘图的函数，将在后续复用
    ...     ax.plot(stat_vals, pdf)  # 在子图上绘制统计值与概率密度函数的图像
    ...     ax.set_title("Normality Test Null Distribution")  # 设置子图标题
    ...     ax.set_xlabel("statistic")  # 设置x轴标签
    ...     ax.set_ylabel("probability density")  # 设置y轴标签
    >>> plot(ax)  # 在子图上调用绘图函数
    >>> plt.show()  # 显示图形

    The comparison is quantified by the p-value: the proportion of values in
    the null distribution greater than or equal to the observed value of the
    statistic.

    >>> fig, ax = plt.subplots(figsize=(8, 5))  # 创建新的8x5尺寸的图形和子图对象
    >>> plot(ax)  # 在子图上调用绘图函数
    >>> pvalue = dist.sf(res.statistic)  # 计算p-value，即卡方分布中大于等于观测统计值的概率
    >>> annotation = (f'p-value={pvalue:.6f}\n(shaded area)')  # 准备标注字符串
    >>> props = dict(facecolor='black', width=1, headwidth=5, headlength=8)  # 定义箭头属性
    >>> _ = ax.annotate(annotation, (13.5, 5e-4), (14, 5e-3), arrowprops=props)  # 在指定位置添加标注和箭头
    >>> i = stat_vals >= res.statistic  # 找到比观测统计值更极端的统计值的索引
    >>> ax.fill_between(stat_vals[i], y1=0, y2=pdf[i])  # 在子图中填充两个统计值之间的区域
    >>> ax.set_xlim(8, 16)  # 设置x轴的显示范围
    >>> ax.set_ylim(0, 0.01)  # 设置y轴的显示范围
    >>> plt.show()  # 显示图形
    >>> res.pvalue  # 输出结果对象的p-value值
    0.0014779023013100172

    If the p-value is "small" - that is, if there is a low probability of
    sampling data from a normally distributed population that produces such an
    extreme value of the statistic - this may be taken as evidence against
    the null hypothesis in favor of the alternative: the weights were not
    drawn from a normal distribution. Note that:

    - The inverse is not true; that is, the test is not used to provide
      evidence for the null hypothesis.
    - The threshold for values that will be considered "small" is a choice that
      should be made before the data is analyzed [4]_ with consideration of the
      risks of both false positives (incorrectly rejecting the null hypothesis)
      and false negatives (failure to reject a false null hypothesis).

    Note that the chi-squared distribution provides an asymptotic
    approximation of the null distribution; it is only accurate for samples
    with many observations. This is the reason we received a warning at the
    beginning of the example; our sample is quite small. In this case,
    `scipy.stats.monte_carlo_test` may provide a more accurate, albeit
    stochastic, approximation of the exact p-value.
    # 将输入数组 `a` 转换为适当的数组命名空间
    xp = array_namespace(a)
    
    # 计算数据在给定轴向上的偏斜度（skewness）和峰度（kurtosis）测试统计量，不进行装饰
    s, _ = skewtest(a, axis, _no_deco=True)
    k, _ = kurtosistest(a, axis, _no_deco=True)
    
    # 计算统计量，这里将偏斜度和峰度的测试统计量平方后相加
    statistic = s*s + k*k
    
    # 创建一个简单的卡方对象 `_SimpleChi2`，使用参数 `2.0` 并转换为适当的数组命名空间
    chi2 = _SimpleChi2(xp.asarray(2.))
    
    # 使用给定的统计量、卡方值、单边检验的选择、非对称性、以及适当的数组命名空间，计算 p 值
    pvalue = _get_pvalue(statistic, chi2, alternative='greater', symmetric=False, xp=xp)
    
    # 如果统计量的维度是 0，则将其转换为标量
    statistic = statistic[()] if statistic.ndim == 0 else statistic
    # 如果 p 值的维度是 0，则将其转换为标量
    pvalue = pvalue[()] if pvalue.ndim == 0 else pvalue
    
    # 返回正态检验结果对象，其中包含统计量和 p 值
    return NormaltestResult(statistic, pvalue)
# 创建一个装饰器函数，用于设置 NaN 策略为 'SignificanceResult'，默认轴为 None
@_axis_nan_policy_factory(SignificanceResult, default_axis=None)
# 定义 Jarque-Bera 函数，用于执行 Jarque-Bera 正态性检验
def jarque_bera(x, *, axis=None):
    r"""Perform the Jarque-Bera goodness of fit test on sample data.

    The Jarque-Bera test tests whether the sample data has the skewness and
    kurtosis matching a normal distribution.

    Note that this test only works for a large enough number of data samples
    (>2000) as the test statistic asymptotically has a Chi-squared distribution
    with 2 degrees of freedom.

    Parameters
    ----------
    x : array_like
        Observations of a random variable.
    axis : int or None, default: 0
        If an int, the axis of the input along which to compute the statistic.
        The statistic of each axis-slice (e.g. row) of the input will appear in
        a corresponding element of the output.
        If ``None``, the input will be raveled before computing the statistic.

    Returns
    -------
    result : SignificanceResult
        An object with the following attributes:

        statistic : float
            The test statistic.
        pvalue : float
            The p-value for the hypothesis test.

    References
    ----------
    .. [1] Jarque, C. and Bera, A. (1980) "Efficient tests for normality,
           homoscedasticity and serial independence of regression residuals",
           6 Econometric Letters 255-259.
    .. [2] Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test
           for normality (complete samples). Biometrika, 52(3/4), 591-611.
    .. [3] B. Phipson and G. K. Smyth. "Permutation P-values Should Never Be
           Zero: Calculating Exact P-values When Permutations Are Randomly
           Drawn." Statistical Applications in Genetics and Molecular Biology
           9.1 (2010).
    .. [4] Panagiotakos, D. B. (2008). The value of p-value in biomedical
           research. The open cardiovascular medicine journal, 2, 97.

    Examples
    --------
    Suppose we wish to infer from measurements whether the weights of adult
    human males in a medical study are not normally distributed [2]_.
    The weights (lbs) are recorded in the array ``x`` below.

    >>> import numpy as np
    >>> x = np.array([148, 154, 158, 160, 161, 162, 166, 170, 182, 195, 236])

    The Jarque-Bera test begins by computing a statistic based on the sample
    skewness and kurtosis.

    >>> from scipy import stats
    >>> res = stats.jarque_bera(x)
    >>> res.statistic
    6.982848237344646

    Because the normal distribution has zero skewness and zero
    ("excess" or "Fisher") kurtosis, the value of this statistic tends to be
    low for samples drawn from a normal distribution.

    The test is performed by comparing the observed value of the statistic
    against the null distribution: the distribution of statistic values derived
    under the null hypothesis that the weights were drawn from a normal
    distribution.
    For the Jarque-Bera test, the null distribution for very large samples is
    ```
    the chi-squared distribution with two degrees of freedom.

    >>> import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块
    >>> dist = stats.chi2(df=2)  # 创建自由度为 2 的卡方分布对象
    >>> jb_val = np.linspace(0, 11, 100)  # 创建一个从 0 到 11 的间隔为 100 的数列
    >>> pdf = dist.pdf(jb_val)  # 计算卡方分布在 jb_val 数组中的概率密度函数值
    >>> fig, ax = plt.subplots(figsize=(8, 5))  # 创建一个图形窗口和一个坐标轴对象
    >>> def jb_plot(ax):  # 定义一个函数用于绘制 Jarque-Bera 空假设分布图
    ...     ax.plot(jb_val, pdf)  # 在指定坐标轴上绘制 jb_val 和对应的 pdf
    ...     ax.set_title("Jarque-Bera Null Distribution")  # 设置图表标题
    ...     ax.set_xlabel("statistic")  # 设置 x 轴标签
    ...     ax.set_ylabel("probability density")  # 设置 y 轴标签
    >>> jb_plot(ax)  # 调用 jb_plot 函数绘制 Jarque-Bera 空假设分布图至指定坐标轴
    >>> plt.show()  # 显示图形

    The comparison is quantified by the p-value: the proportion of values in
    the null distribution greater than or equal to the observed value of the
    statistic.

    >>> fig, ax = plt.subplots(figsize=(8, 5))  # 创建一个新的图形窗口和坐标轴对象
    >>> jb_plot(ax)  # 调用 jb_plot 函数绘制 Jarque-Bera 空假设分布图至指定坐标轴
    >>> pvalue = dist.sf(res.statistic)  # 计算 Jarque-Bera 统计量的 p 值
    >>> annotation = (f'p-value={pvalue:.6f}\n(shaded area)')  # 设置注释内容
    >>> props = dict(facecolor='black', width=1, headwidth=5, headlength=8)  # 设置箭头属性
    >>> _ = ax.annotate(annotation, (7.5, 0.01), (8, 0.05), arrowprops=props)  # 在指定位置添加注释及箭头
    >>> i = jb_val >= res.statistic  # 找到比 Jarque-Bera 统计量更极端的值的索引
    >>> ax.fill_between(jb_val[i], y1=0, y2=pdf[i])  # 填充指定区域
    >>> ax.set_xlim(0, 11)  # 设置 x 轴的显示范围
    >>> ax.set_ylim(0, 0.3)  # 设置 y 轴的显示范围
    >>> plt.show()  # 显示图形
    >>> res.pvalue  # 打印 Jarque-Bera 检验的 p 值
    0.03045746622458189

    If the p-value is "small" - that is, if there is a low probability of
    sampling data from a normally distributed population that produces such an
    extreme value of the statistic - this may be taken as evidence against
    the null hypothesis in favor of the alternative: the weights were not
    drawn from a normal distribution. Note that:

    - The inverse is not true; that is, the test is not used to provide
      evidence for the null hypothesis.
    - The threshold for values that will be considered "small" is a choice that
      should be made before the data is analyzed [3]_ with consideration of the
      risks of both false positives (incorrectly rejecting the null hypothesis)
      and false negatives (failure to reject a false null hypothesis).

    Note that the chi-squared distribution provides an asymptotic approximation
    of the null distribution; it is only accurate for samples with many
    observations. For small samples like ours, `scipy.stats.monte_carlo_test`
    may provide a more accurate, albeit stochastic, approximation of the
    exact p-value.

    >>> def statistic(x, axis):
    ...     # underlying calculation of the Jarque Bera statistic
    ...     s = stats.skew(x, axis=axis)  # 计算偏度
    ...     k = stats.kurtosis(x, axis=axis)  # 计算峰度
    ...     return x.shape[axis]/6 * (s**2 + k**2/4)  # 返回 Jarque-Bera 统计量
    >>> res = stats.monte_carlo_test(x, stats.norm.rvs, statistic,
    ...                              alternative='greater')  # 进行蒙特卡洛模拟检验
    >>> fig, ax = plt.subplots(figsize=(8, 5))  # 创建一个新的图形窗口和坐标轴对象
    >>> jb_plot(ax)  # 调用 jb_plot 函数绘制 Jarque-Bera 空假设分布图至指定坐标轴
    >>> ax.hist(res.null_distribution, np.linspace(0, 10, 50),  # 绘制空假设分布的直方图
    ...         density=True)
    >>> ax.legend(['aymptotic approximation (many observations)',  # 设置图例说明
    xp = array_namespace(x)
    x = xp.asarray(x)
    # 将输入的 x 转换为与给定的数组命名空间相关的数组类型

    if axis is None:
        # 如果未指定轴向，则将 x 展平为一维数组
        x = xp.reshape(x, (-1,))
        axis = 0

    n = x.shape[axis]
    # 获取数组 x 在指定轴向上的长度

    if n == 0:
        # 如果数组长度为 0，则抛出数值错误异常
        raise ValueError('At least one observation is required.')

    mu = xp.mean(x, axis=axis, keepdims=True)
    # 计算数组 x 在指定轴向上的均值，并保持维度不变

    diffx = x - mu
    # 计算每个观测值与均值之间的差异

    s = skew(diffx, axis=axis, _no_deco=True)
    # 计算数组 x 在指定轴向上的偏度

    k = kurtosis(diffx, axis=axis, _no_deco=True)
    # 计算数组 x 在指定轴向上的峰度

    statistic = n / 6 * (s**2 + k**2 / 4)
    # 计算偏度和峰度的组合统计量

    chi2 = _SimpleChi2(xp.asarray(2.))
    # 创建一个简单的卡方分布对象，自由度为 2

    pvalue = _get_pvalue(statistic, chi2, alternative='greater', symmetric=False, xp=xp)
    # 根据给定的统计量和卡方分布，计算单侧假设的 p 值

    statistic = statistic[()] if statistic.ndim == 0 else statistic
    # 将统计量转换为标量（如果它是零维数组）

    pvalue = pvalue[()] if pvalue.ndim == 0 else pvalue
    # 将 p 值转换为标量（如果它是零维数组）

    return SignificanceResult(statistic, pvalue)
    # 返回一个包含统计量和 p 值的 SignificanceResult 对象
# 定义一个函数用于计算输入序列中给定百分位数的分数
def scoreatpercentile(a, per, limit=(), interpolation_method='fraction',
                      axis=None):
    """Calculate the score at a given percentile of the input sequence.

    For example, the score at ``per=50`` is the median. If the desired quantile
    lies between two data points, we interpolate between them, according to
    the value of `interpolation`. If the parameter `limit` is provided, it
    should be a tuple (lower, upper) of two values.

    Parameters
    ----------
    a : array_like
        A 1-D array of values from which to extract score.
    per : array_like
        Percentile(s) at which to extract score.  Values should be in range
        [0,100].
    limit : tuple, optional
        Tuple of two scalars, the lower and upper limits within which to
        compute the percentile. Values of `a` outside
        this (closed) interval will be ignored.
    interpolation_method : {'fraction', 'lower', 'higher'}, optional
        Specifies the interpolation method to use,
        when the desired quantile lies between two data points `i` and `j`
        The following options are available (default is 'fraction'):

          * 'fraction': ``i + (j - i) * fraction`` where ``fraction`` is the
            fractional part of the index surrounded by ``i`` and ``j``
          * 'lower': ``i``
          * 'higher': ``j``

    axis : int, optional
        Axis along which the percentiles are computed. Default is None. If
        None, compute over the whole array `a`.

    Returns
    -------
    score : float or ndarray
        Score at percentile(s).

    See Also
    --------
    percentileofscore, numpy.percentile

    Notes
    -----
    This function will become obsolete in the future.
    For NumPy 1.9 and higher, `numpy.percentile` provides all the functionality
    that `scoreatpercentile` provides.  And it's significantly faster.
    Therefore it's recommended to use `numpy.percentile` for users that have
    numpy >= 1.9.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> a = np.arange(100)
    >>> stats.scoreatpercentile(a, 50)
    49.5

    """
    # 将输入转换为 NumPy 数组
    a = np.asarray(a)
    # 如果输入数组为空，返回与 `per` 形状匹配的 NaN 值
    if a.size == 0:
        if np.isscalar(per):
            return np.nan
        else:
            return np.full(np.asarray(per).shape, np.nan, dtype=np.float64)

    # 如果提供了限制范围 `limit`，则筛选出在此范围内的数组元素
    if limit:
        a = a[(limit[0] <= a) & (a <= limit[1])]

    # 对数组进行排序
    sorted_ = np.sort(a, axis=axis)
    # 如果未指定轴向，则默认为 0
    if axis is None:
        axis = 0

    # 调用内部函数计算指定百分位数的分数并返回结果
    return _compute_qth_percentile(sorted_, per, interpolation_method, axis)


# 处理一系列百分位数而不必多次调用排序函数
# 计算给定排序数组的第per分位数的值
def _compute_qth_percentile(sorted_, per, interpolation_method, axis):
    # 如果per不是标量，递归地计算每个per对应的分位数值
    if not np.isscalar(per):
        score = [_compute_qth_percentile(sorted_, i,
                                         interpolation_method, axis)
                 for i in per]
        return np.array(score)

    # 确保分位数per在[0, 100]范围内
    if not (0 <= per <= 100):
        raise ValueError("percentile must be in the range [0, 100]")

    # 计算在排序数组中对应于分位数per的索引位置
    indexer = [slice(None)] * sorted_.ndim
    idx = per / 100. * (sorted_.shape[axis] - 1)

    # 如果idx不是整数，根据插值方法对其进行舍入
    if int(idx) != idx:
        if interpolation_method == 'lower':
            idx = int(np.floor(idx))
        elif interpolation_method == 'higher':
            idx = int(np.ceil(idx))
        elif interpolation_method == 'fraction':
            pass  # 保持idx为分数并插值
        else:
            raise ValueError("interpolation_method can only be 'fraction', "
                             "'lower' or 'higher'")

    # 将idx转换为整数
    i = int(idx)
    if i == idx:
        # 如果i等于idx，权重为1，sumval为1.0
        indexer[axis] = slice(i, i + 1)
        weights = array(1)
        sumval = 1.0
    else:
        # 如果i不等于idx，计算两个相邻索引i和i+1之间的权重
        indexer[axis] = slice(i, i + 2)
        j = i + 1
        weights = array([(j - idx), (idx - i)], float)
        wshape = [1] * sorted_.ndim
        wshape[axis] = 2
        weights.shape = wshape
        sumval = weights.sum()

    # 使用np.add.reduce（等同于np.sum但更快）来强制数据类型
    return np.add.reduce(sorted_[tuple(indexer)] * weights, axis=axis) / sumval


# 计算得分相对于得分列表a的百分位排名
def percentileofscore(a, score, kind='rank', nan_policy='propagate'):
    """计算得分相对于得分列表a的百分位排名。

    例如，百分位排名为80%表示a中80%的得分低于给定的得分。对于有间隔或并列的情况，
    精确定义取决于可选关键字kind。

    Parameters
    ----------
    a : array_like
        要比较score的1维数组。
    score : array_like
        要计算百分位的得分。
    kind : {'rank', 'weak', 'strict', 'mean'}, optional
        指定结果得分的解释方式。
        可用选项有（默认为'rank'）：

          * 'rank': 得分的平均百分位排名。如果存在多个匹配项，平均所有匹配得分的百分位排名。
          * 'weak': 对应累积分布函数的定义。百分位排名为80%表示80%的值小于或等于提供的得分。
          * 'strict': 类似于'weak'，但仅计数严格小于给定得分的值。
          * 'mean': 'weak'和'strict'分数的平均值，通常用于测试。参见https://en.wikipedia.org/wiki/Percentile_rank
    """
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        # 定义了如何处理参数 `a` 中的 `nan` 值
        # 可选值有三种（默认为 'propagate'）：
        # - 'propagate': 将每个 `score` 的输出设为 nan
        # - 'raise': 抛出错误
        # - 'omit': 在计算中忽略 nan 值

    Returns
    -------
    pcos : float
        # 返回 `score` 相对于 `a` 的百分位位置 (0-100)。

    See Also
    --------
    numpy.percentile
    scipy.stats.scoreatpercentile, scipy.stats.rankdata

    Examples
    --------
    Three-quarters of the given values lie below a given score:

    >>> import numpy as np
    >>> from scipy import stats
    >>> stats.percentileofscore([1, 2, 3, 4], 3)
    75.0

    With multiple matches, note how the scores of the two matches, 0.6
    and 0.8 respectively, are averaged:

    >>> stats.percentileofscore([1, 2, 3, 3, 4], 3)
    70.0

    Only 2/5 values are strictly less than 3:

    >>> stats.percentileofscore([1, 2, 3, 3, 4], 3, kind='strict')
    40.0

    But 4/5 values are less than or equal to 3:

    >>> stats.percentileofscore([1, 2, 3, 3, 4], 3, kind='weak')
    80.0

    The average between the weak and the strict scores is:

    >>> stats.percentileofscore([1, 2, 3, 3, 4], 3, kind='mean')
    60.0

    Score arrays (of any dimensionality) are supported:

    >>> stats.percentileofscore([1, 2, 3, 3, 4], [2, 3])
    array([40., 70.])

    The inputs can be infinite:

    >>> stats.percentileofscore([-np.inf, 0, 1, np.inf], [1, 2, np.inf])
    array([75., 75., 100.])

    If `a` is empty, then the resulting percentiles are all `nan`:

    >>> stats.percentileofscore([], [1, 2])
    array([nan, nan])
    """

    a = np.asarray(a)
    # 将 `a` 转换为 NumPy 数组

    n = len(a)
    # 获取数组 `a` 的长度

    score = np.asarray(score)
    # 将 `score` 转换为 NumPy 数组

    # Nan treatment
    cna, npa = _contains_nan(a, nan_policy)
    # 检查数组 `a` 中是否包含 `nan` 值，并根据 `nan_policy` 处理
    cns, nps = _contains_nan(score, nan_policy)
    # 检查数组 `score` 中是否包含 `nan` 值，并根据 `nan_policy` 处理

    if (cna or cns) and nan_policy == 'raise':
        # 如果 `a` 或 `score` 中包含 `nan` 值，并且 `nan_policy` 是 'raise'，则抛出错误
        raise ValueError("The input contains nan values")

    if cns:
        # 如果 `score` 中包含 `nan` 值，则将其标记为 `masked`，以便在计算中忽略
        score = ma.masked_where(np.isnan(score), score)

    if cna:
        if nan_policy == "omit":
            # 如果 `a` 中包含 `nan` 值，并且 `nan_policy` 是 'omit'，则在计算中忽略这些值
            a = ma.masked_where(np.isnan(a), a)
            n = a.count()

        if nan_policy == "propagate":
            # 如果 `a` 中包含 `nan` 值，并且 `nan_policy` 是 'propagate'，则所有输出设为 `nan`
            n = 0

    # Cannot compare to empty list ==> nan
    # 如果 `a` 是空的，则百分位位置结果都为 `nan`
    if n == 0:
        perct = np.full_like(score, np.nan, dtype=np.float64)
    else:
        # 准备进行广播操作，将分数数组扩展一个维度
        score = score[..., None]

        # 定义一个函数用于计算数组中非零元素的数量
        def count(x):
            return np.count_nonzero(x, -1)

        # 主要的计算逻辑
        if kind == 'rank':
            # 按照排名方式计算百分比
            left = count(a < score)
            right = count(a <= score)
            plus1 = left < right
            perct = (left + right + plus1) * (50.0 / n)
        elif kind == 'strict':
            # 按照严格方式计算百分比
            perct = count(a < score) * (100.0 / n)
        elif kind == 'weak':
            # 按照宽松方式计算百分比
            perct = count(a <= score) * (100.0 / n)
        elif kind == 'mean':
            # 按照均值方式计算百分比
            left = count(a < score)
            right = count(a <= score)
            perct = (left + right) * (50.0 / n)
        else:
            # 抛出数值错误，kind 参数只能是 'rank', 'strict', 'weak' 或 'mean'
            raise ValueError(
                "kind can only be 'rank', 'strict', 'weak' or 'mean'")

    # 将 perct 中的空值替换为 NaN
    perct = ma.filled(perct, np.nan)

    # 如果 perct 是零维数组，返回其标量值
    if perct.ndim == 0:
        return perct[()]
    # 否则返回 perct 数组
    return perct
# 使用 namedtuple 定义一个名为 HistogramResult 的元组类型，包含 count、lowerlimit、binsize、extrapoints 四个字段
HistogramResult = namedtuple('HistogramResult',
                             ('count', 'lowerlimit', 'binsize', 'extrapoints'))

# 定义一个名为 _histogram 的函数，用于生成直方图
def _histogram(a, numbins=10, defaultlimits=None, weights=None,
               printextras=False):
    """Create a histogram.

    Separate the range into several bins and return the number of instances
    in each bin.

    Parameters
    ----------
    a : array_like
        Array of scores which will be put into bins.
    numbins : int, optional
        The number of bins to use for the histogram. Default is 10.
    defaultlimits : tuple (lower, upper), optional
        The lower and upper values for the range of the histogram.
        If no value is given, a range slightly larger than the range of the
        values in a is used. Specifically ``(a.min() - s, a.max() + s)``,
        where ``s = (1/2)(a.max() - a.min()) / (numbins - 1)``.
    weights : array_like, optional
        The weights for each value in `a`. Default is None, which gives each
        value a weight of 1.0
    printextras : bool, optional
        If True, if there are extra points (i.e. the points that fall outside
        the bin limits) a warning is raised saying how many of those points
        there are.  Default is False.

    Returns
    -------
    count : ndarray
        Number of points (or sum of weights) in each bin.
    lowerlimit : float
        Lowest value of histogram, the lower limit of the first bin.
    binsize : float
        The size of the bins (all bins have the same size).
    extrapoints : int
        The number of points outside the range of the histogram.

    See Also
    --------
    numpy.histogram

    Notes
    -----
    This histogram is based on numpy's histogram but has a larger range by
    default if default limits is not set.

    """
    # 将输入数组 a 扁平化处理
    a = np.ravel(a)
    # 如果未提供默认限制范围
    if defaultlimits is None:
        # 如果数组 a 为空，则设定默认范围为 0 到 1
        if a.size == 0:
            defaultlimits = (0, 1)
        else:
            # 否则使用数组 a 的最小值和最大值来设置默认范围，并在最大值和最小值的基础上稍微扩展
            data_min = a.min()
            data_max = a.max()
            s = (data_max - data_min) / (2. * (numbins - 1.))
            defaultlimits = (data_min - s, data_max + s)

    # 使用 numpy 的直方图方法计算直方图及其分 bin 边界
    hist, bin_edges = np.histogram(a, bins=numbins, range=defaultlimits,
                                   weights=weights)
    # 将 hist 转换为浮点型数组，以保持与旧输出的一致性
    hist = np.array(hist, dtype=float)
    # 假设 numpy 的直方图对于整数 'bins' 给出固定宽度的 bins
    binsize = bin_edges[1] - bin_edges[0]
    # 计算超出范围的点数
    extrapoints = len([v for v in a
                       if defaultlimits[0] > v or v > defaultlimits[1]])
    # 如果 extrapoints 大于 0 并且 printextras 为真，则发出警告
    if extrapoints > 0 and printextras:
        # 使用警告模块发出警告，指出超出给定直方图范围的额外点数
        warnings.warn(f"Points outside given histogram range = {extrapoints}",
                      stacklevel=3,)
    
    # 返回直方图计算结果对象，包括直方图数据(hist)、默认限制的第一个值、箱子大小、额外点数
    return HistogramResult(hist, defaultlimits[0], binsize, extrapoints)
# 创建命名元组 CumfreqResult，用于存储累积频率直方图的结果
CumfreqResult = namedtuple('CumfreqResult',
                           ('cumcount', 'lowerlimit', 'binsize',
                            'extrapoints'))

# 定义函数 cumfreq，返回累积频率直方图，利用内部函数 _histogram 实现
def cumfreq(a, numbins=10, defaultreallimits=None, weights=None):
    """Return a cumulative frequency histogram, using the histogram function.

    A cumulative histogram is a mapping that counts the cumulative number of
    observations in all of the bins up to the specified bin.

    Parameters
    ----------
    a : array_like
        Input array.
    numbins : int, optional
        The number of bins to use for the histogram. Default is 10.
    defaultreallimits : tuple (lower, upper), optional
        The lower and upper values for the range of the histogram.
        If no value is given, a range slightly larger than the range of the
        values in `a` is used. Specifically ``(a.min() - s, a.max() + s)``,
        where ``s = (1/2)(a.max() - a.min()) / (numbins - 1)``.
    weights : array_like, optional
        The weights for each value in `a`. Default is None, which gives each
        value a weight of 1.0

    Returns
    -------
    cumcount : ndarray
        Binned values of cumulative frequency.
    lowerlimit : float
        Lower real limit
    binsize : float
        Width of each bin.
    extrapoints : int
        Extra points.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy import stats
    >>> rng = np.random.default_rng()
    >>> x = [1, 4, 2, 1, 3, 1]
    >>> res = stats.cumfreq(x, numbins=4, defaultreallimits=(1.5, 5))
    >>> res.cumcount
    array([ 1.,  2.,  3.,  3.])
    >>> res.extrapoints
    3

    Create a normal distribution with 1000 random values

    >>> samples = stats.norm.rvs(size=1000, random_state=rng)

    Calculate cumulative frequencies

    >>> res = stats.cumfreq(samples, numbins=25)

    Calculate space of values for x

    >>> x = res.lowerlimit + np.linspace(0, res.binsize*res.cumcount.size,
    ...                                  res.cumcount.size)

    Plot histogram and cumulative histogram

    >>> fig = plt.figure(figsize=(10, 4))
    >>> ax1 = fig.add_subplot(1, 2, 1)
    >>> ax2 = fig.add_subplot(1, 2, 2)
    >>> ax1.hist(samples, bins=25)
    >>> ax1.set_title('Histogram')
    >>> ax2.bar(x, res.cumcount, width=res.binsize)
    >>> ax2.set_title('Cumulative histogram')
    >>> ax2.set_xlim([x.min(), x.max()])

    >>> plt.show()

    """
    # 调用内部函数 _histogram，获取直方图结果 h, l, b, e
    h, l, b, e = _histogram(a, numbins, defaultreallimits, weights=weights)
    # 计算累积直方图，即将每个 bin 的频率累加
    cumhist = np.cumsum(h * 1, axis=0)
    # 返回命名元组 CumfreqResult，包含累积频率直方图的结果
    return CumfreqResult(cumhist, l, b, e)

# 创建命名元组 RelfreqResult，用于存储相对频率直方图的结果
RelfreqResult = namedtuple('RelfreqResult',
                           ('frequency', 'lowerlimit', 'binsize',
                            'extrapoints'))

# 定义函数 relfreq，返回相对频率直方图，与 cumfreq 类似，使用 _histogram 函数
def relfreq(a, numbins=10, defaultreallimits=None, weights=None):
    """Return a relative frequency histogram, using the histogram function.

    A relative frequency histogram is a mapping of the number of
    """
    def relfreq(a, numbins=10, defaultreallimits=None, weights=None):
        """
        Calculate relative frequency histogram.
    
        Parameters
        ----------
        a : array_like
            Input array.
        numbins : int, optional
            The number of bins to use for the histogram. Default is 10.
        defaultreallimits : tuple (lower, upper), optional
            The lower and upper values for the range of the histogram.
            If no value is given, a range slightly larger than the range of the
            values in a is used. Specifically ``(a.min() - s, a.max() + s)``,
            where ``s = (1/2)(a.max() - a.min()) / (numbins - 1)``.
        weights : array_like, optional
            The weights for each value in `a`. Default is None, which gives each
            value a weight of 1.0
    
        Returns
        -------
        frequency : ndarray
            Binned values of relative frequency.
        lowerlimit : float
            Lower real limit.
        binsize : float
            Width of each bin.
        extrapoints : int
            Extra points.
    
        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from scipy import stats
        >>> rng = np.random.default_rng()
        >>> a = np.array([2, 4, 1, 2, 3, 2])
        >>> res = stats.relfreq(a, numbins=4)
        >>> res.frequency
        array([ 0.16666667, 0.5       , 0.16666667,  0.16666667])
        >>> np.sum(res.frequency)  # relative frequencies should add up to 1
        1.0
    
        Create a normal distribution with 1000 random values
    
        >>> samples = stats.norm.rvs(size=1000, random_state=rng)
    
        Calculate relative frequencies
    
        >>> res = stats.relfreq(samples, numbins=25)
    
        Calculate space of values for x
    
        >>> x = res.lowerlimit + np.linspace(0, res.binsize*res.frequency.size,
        ...                                  res.frequency.size)
    
        Plot relative frequency histogram
    
        >>> fig = plt.figure(figsize=(5, 4))
        >>> ax = fig.add_subplot(1, 1, 1)
        >>> ax.bar(x, res.frequency, width=res.binsize)
        >>> ax.set_title('Relative frequency histogram')
        >>> ax.set_xlim([x.min(), x.max()])
    
        >>> plt.show()
    
        """
        # Convert input `a` to a numpy array
        a = np.asanyarray(a)
        # Compute histogram values, lower real limit, bin size, and extra points
        h, l, b, e = _histogram(a, numbins, defaultreallimits, weights=weights)
        # Normalize histogram values to relative frequencies
        h = h / a.shape[0]
    
        # Return the result as an instance of RelfreqResult
        return RelfreqResult(h, l, b, e)
#####################################
#        VARIABILITY FUNCTIONS      #
#####################################

# 计算奥布莱恩变换
def obrientransform(*samples):
    """Compute the O'Brien transform on input data (any number of arrays).

    Used to test for homogeneity of variance prior to running one-way stats.
    Each array in ``*samples`` is one level of a factor.
    If `f_oneway` is run on the transformed data and found significant,
    the variances are unequal.  From Maxwell and Delaney [1]_, p.112.

    Parameters
    ----------
    sample1, sample2, ... : array_like
        Any number of arrays.

    Returns
    -------
    obrientransform : ndarray
        Transformed data for use in an ANOVA.  The first dimension
        of the result corresponds to the sequence of transformed
        arrays.  If the arrays given are all 1-D of the same length,
        the return value is a 2-D array; otherwise it is a 1-D array
        of type object, with each element being an ndarray.

    References
    ----------
    .. [1] S. E. Maxwell and H. D. Delaney, "Designing Experiments and
           Analyzing Data: A Model Comparison Perspective", Wadsworth, 1990.

    Examples
    --------
    We'll test the following data sets for differences in their variance.

    >>> x = [10, 11, 13, 9, 7, 12, 12, 9, 10]
    >>> y = [13, 21, 5, 10, 8, 14, 10, 12, 7, 15]

    Apply the O'Brien transform to the data.

    >>> from scipy.stats import obrientransform
    >>> tx, ty = obrientransform(x, y)

    Use `scipy.stats.f_oneway` to apply a one-way ANOVA test to the
    transformed data.

    >>> from scipy.stats import f_oneway
    >>> F, p = f_oneway(tx, ty)
    >>> p
    0.1314139477040335

    If we require that ``p < 0.05`` for significance, we cannot conclude
    that the variances are different.

    """
    
    TINY = np.sqrt(np.finfo(float).eps)  # 极小值，用于数值比较

    # `arrays` will hold the transformed arguments.
    arrays = []  # 用于存放变换后的数组
    sLast = None  # 上一个数组的形状

    for sample in samples:
        a = np.asarray(sample)  # 将输入样本转换为 ndarray
        n = len(a)  # 数组长度
        mu = np.mean(a)  # 计算均值
        sq = (a - mu)**2  # 平方差
        sumsq = sq.sum()  # 平方差总和

        # The O'Brien transform.
        # 奥布莱恩变换公式
        t = ((n - 1.5) * n * sq - 0.5 * sumsq) / ((n - 1) * (n - 2))

        # Check that the mean of the transformed data is equal to the
        # original variance.
        var = sumsq / (n - 1)
        if abs(var - np.mean(t)) > TINY:
            raise ValueError('Lack of convergence in obrientransform.')

        arrays.append(t)  # 将变换后的数组添加到数组列表中
        sLast = a.shape  # 记录当前数组的形状

    if sLast:
        for arr in arrays[:-1]:
            if sLast != arr.shape:
                return np.array(arrays, dtype=object)  # 返回对象类型的数组
    return np.array(arrays)


@_axis_nan_policy_factory(
    lambda x: x, result_to_tuple=lambda x: (x,), n_outputs=1, too_small=1
)
# 标准误差的计算函数
def sem(a, axis=0, ddof=1, nan_policy='propagate'):
    """Compute standard error of the mean.

    Calculate the standard error of the mean (or standard error of
    measurement) of the values in the input array.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose SEM is desired.
    axis : int or None, optional
        Axis along which the means are computed. Default is 0.
    ddof : int, optional
        Delta degrees of freedom. Default is 1.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan. Default is 'propagate'.
    ----------
    a : array_like
        包含要计算标准误差的值的数组。至少包含两个观测值。
    axis : int or None, optional
        操作的轴向。默认为 0。如果为 None，则在整个数组 `a` 上计算。
    ddof : int, optional
        自由度修正值。用于在有限样本中调整偏差相对于总体方差的自由度。默认为 1。
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        定义处理包含 NaN（Not a Number）时的策略。
        可选项如下（默认为 'propagate'）：

          * 'propagate': 返回 NaN
          * 'raise': 抛出错误
          * 'omit': 在计算中忽略 NaN 值

    Returns
    -------
    s : ndarray or float
        样本（或样本集）中均值的标准误差，沿着输入轴的方向。

    Notes
    -----
    `ddof` 的默认值与其他包含 ddof 的函数默认值（0）不同，如 np.std 和 np.nanstd。

    Examples
    --------
    沿着第一个轴计算标准误差：

    >>> import numpy as np
    >>> from scipy import stats
    >>> a = np.arange(20).reshape(5,4)
    >>> stats.sem(a)
    array([ 2.8284,  2.8284,  2.8284,  2.8284])

    在整个数组上计算标准误差，使用 n 个自由度：

    >>> stats.sem(a, axis=None, ddof=0)
    1.2893796958227628

    """
    xp = array_namespace(a)  # 将数组 a 命名空间化为 xp
    if axis is None:
        a = xp.reshape(a, (-1,))  # 将数组 a 按指定形状重新排列为一维数组
        axis = 0  # 设置操作轴向为 0
    a = atleast_nd(a, ndim=1, xp=xp)  # 将数组 a 至少扩展到指定维度为 1，并使用 xp
    n = a.shape[axis]  # 获取数组 a 沿指定轴向的长度
    s = xp.std(a, axis=axis, correction=ddof) / n**0.5  # 计算标准差的均值的标准误差
    return s  # 返回计算结果
# 检查数组 x 中的所有值是否相同，忽略 NaN 值
def _isconst(x):
    y = x[~np.isnan(x)]  # 去除 x 中的 NaN 值，得到非 NaN 值的数组 y
    if y.size == 0:  # 如果 y 为空数组
        return np.array([True])  # 返回一个包含 True 的数组，表示所有值相同（或者全为 NaN）
    else:
        return (y[0] == y).all(keepdims=True)  # 返回一个布尔值，指示数组 y 中的所有元素是否相同


# 计算数组 x 的均值（mean），如果 x 全为 NaN，则静默返回 NaN
def _quiet_nanmean(x):
    y = x[~np.isnan(x)]  # 去除 x 中的 NaN 值，得到非 NaN 值的数组 y
    if y.size == 0:  # 如果 y 为空数组
        return np.array([np.nan])  # 返回一个包含 NaN 的数组，表示 x 全为 NaN
    else:
        return np.mean(y, keepdims=True)  # 计算 y 的均值，保持维度为 1，返回数组


# 计算数组 x 的标准差（std），如果 x 全为 NaN，则静默返回 NaN
def _quiet_nanstd(x, ddof=0):
    y = x[~np.isnan(x)]  # 去除 x 中的 NaN 值，得到非 NaN 值的数组 y
    if y.size == 0:  # 如果 y 为空数组
        return np.array([np.nan])  # 返回一个包含 NaN 的数组，表示 x 全为 NaN
    else:
        return np.std(y, keepdims=True, ddof=ddof)  # 计算 y 的标准差，保持维度为 1，返回数组


# 计算样本数据的 z 分数（标准化分数）
def zscore(a, axis=0, ddof=0, nan_policy='propagate'):
    """
    计算 z 分数。

    计算每个样本值相对于样本均值和标准差的 z 分数。

    Parameters
    ----------
    a : array_like
        包含样本数据的类数组对象。
    axis : int or None, optional
        操作的轴线。默认为 0。如果为 None，则在整个数组 `a` 上计算。
    ddof : int, optional
        在标准差计算中的自由度校正。默认为 0。
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        定义输入包含 NaN 时的处理方式。'propagate' 返回 NaN，
        'raise' 抛出错误，'omit' 忽略 NaN 值进行计算。默认为 'propagate'。
        注意，当值为 'omit' 时，输入中的 NaN 也会传播到输出中，
        但不影响对非 NaN 值计算的 z 分数。

    Returns
    -------
    zscore : array_like
        标准化后的 z 分数，由输入数组 `a` 的均值和标准差计算得到。

    See Also
    --------
    numpy.mean : 算术平均值
    numpy.std : 算术标准差
    scipy.stats.gzscore : 几何标准分数

    Notes
    -----
    此函数保留 ndarray 的子类，并且还适用于矩阵和掩码数组（它使用 `asanyarray` 而不是 `asarray` 来处理参数）。

    References
    ----------
    .. [1] "Standard score", *Wikipedia*,
           https://en.wikipedia.org/wiki/Standard_score.
    .. [2] Huck, S. W., Cross, T. L., Clark, S. B, "Overcoming misconceptions
           about Z-scores", Teaching Statistics, vol. 8, pp. 38-40, 1986

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([ 0.7972,  0.0767,  0.4383,  0.7866,  0.8091,
    """
    # 导入 scipy 库中的 stats 模块
    >>> from scipy import stats
    # 对数组 a 计算标准化得分
    >>> stats.zscore(a)
    # 输出数组 a 的标准化结果
    array([ 1.1273, -1.247 , -0.0552,  1.0923,  1.1664, -0.8559,  0.5786,
            0.6748, -1.1488, -1.3324])

    # 沿指定轴计算标准化得分，使用 n-1 自由度（ddof=1）来计算标准差
    >>> b = np.array([[ 0.3148,  0.0478,  0.6243,  0.4608],
    ...               [ 0.7149,  0.0775,  0.6072,  0.9656],
    ...               [ 0.6341,  0.1403,  0.9759,  0.4064],
    ...               [ 0.5918,  0.6948,  0.904 ,  0.3721],
    ...               [ 0.0921,  0.2481,  0.1188,  0.1366]])
    # 对数组 b 沿轴 1 计算标准化得分，使用 ddof=1 自由度
    >>> stats.zscore(b, axis=1, ddof=1)
    # 输出数组 b 沿轴 1 的标准化结果
    array([[-0.19264823, -1.28415119,  1.07259584,  0.40420358],
           [ 0.33048416, -1.37380874,  0.04251374,  1.00081084],
           [ 0.26796377, -1.12598418,  1.23283094, -0.37481053],
           [-0.22095197,  0.24468594,  1.19042819, -1.21416216],
           [-0.82780366,  1.4457416 , -0.43867764, -0.1792603 ]])

    # 使用 nan_policy='omit' 的示例
    >>> x = np.array([[25.11, 30.10, np.nan, 32.02, 43.15],
    ...               [14.95, 16.06, 121.25, 94.35, 29.81]])
    # 对数组 x 沿轴 1 计算标准化得分，忽略 NaN 值
    >>> stats.zscore(x, axis=1, nan_policy='omit')
    # 输出数组 x 沿轴 1 的标准化结果，忽略 NaN 值
    array([[-1.13490897, -0.37830299,         nan, -0.08718406,  1.60039602],
           [-0.91611681, -0.89090508,  1.4983032 ,  0.88731639, -0.5785977 ]])
    """
    # 调用 zmap 函数，返回对数组 a 的标准化结果，可指定轴和自由度，以及处理 NaN 的策略
    return zmap(a, a, axis=axis, ddof=ddof, nan_policy=nan_policy)
# 定义函数 gzscore，计算几何标准化分数
def gzscore(a, *, axis=0, ddof=0, nan_policy='propagate'):
    """
    Compute the geometric standard score.

    Compute the geometric z score of each strictly positive value in the
    sample, relative to the geometric mean and standard deviation.
    Mathematically the geometric z score can be evaluated as::

        gzscore = log(a/gmu) / log(gsigma)

    where ``gmu`` (resp. ``gsigma``) is the geometric mean (resp. standard
    deviation).

    Parameters
    ----------
    a : array_like
        Sample data.
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over
        the whole array `a`.
    ddof : int, optional
        Degrees of freedom correction in the calculation of the
        standard deviation. Default is 0.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan. 'propagate' returns nan,
        'raise' throws an error, 'omit' performs the calculations ignoring nan
        values. Default is 'propagate'.  Note that when the value is 'omit',
        nans in the input also propagate to the output, but they do not affect
        the geometric z scores computed for the non-nan values.

    Returns
    -------
    gzscore : array_like
        The geometric z scores, standardized by geometric mean and geometric
        standard deviation of input array `a`.

    See Also
    --------
    gmean : Geometric mean
    gstd : Geometric standard deviation
    zscore : Standard score

    Notes
    -----
    This function preserves ndarray subclasses, and works also with
    matrices and masked arrays (it uses ``asanyarray`` instead of
    ``asarray`` for parameters).

    .. versionadded:: 1.8

    References
    ----------
    .. [1] "Geometric standard score", *Wikipedia*,
           https://en.wikipedia.org/wiki/Geometric_standard_deviation#Geometric_standard_score.

    Examples
    --------
    Draw samples from a log-normal distribution:

    >>> import numpy as np
    >>> from scipy.stats import zscore, gzscore
    >>> import matplotlib.pyplot as plt

    >>> rng = np.random.default_rng()
    >>> mu, sigma = 3., 1.  # mean and standard deviation
    >>> x = rng.lognormal(mu, sigma, size=500)

    Display the histogram of the samples:

    >>> fig, ax = plt.subplots()
    >>> ax.hist(x, 50)
    >>> plt.show()

    Display the histogram of the samples standardized by the classical zscore.
    Distribution is rescaled but its shape is unchanged.

    >>> fig, ax = plt.subplots()
    >>> ax.hist(zscore(x), 50)
    >>> plt.show()

    Demonstrate that the distribution of geometric zscores is rescaled and
    quasinormal:

    >>> fig, ax = plt.subplots()
    >>> ax.hist(gzscore(x), 50)
    >>> plt.show()

    """
    # 将输入数据转换为数组（不管是不是掩码数组）
    a = np.asanyarray(a)
    # 如果输入数据是掩码数组，则使用掩码数组的对数函数；否则使用标准的对数函数
    log = ma.log if isinstance(a, ma.MaskedArray) else np.log
    # 返回经过对数变换后的数据的 z 分数，按指定的轴和参数计算
    return zscore(log(a), axis=axis, ddof=ddof, nan_policy=nan_policy)
# 将 compare 转换为任意 ndarray 子类，确保参数为数组或矩阵，同时处理掩码数组
a = np.asanyarray(compare)

# 如果 a 的大小为 0，返回一个形状与 a 相同的空数组
if a.size == 0:
    return np.empty(a.shape)

# 检查数组 a 是否包含 NaN 值，并根据 nan_policy 处理 NaN
contains_nan, nan_policy = _contains_nan(a, nan_policy)

# 如果数组包含 NaN 且 nan_policy 为 'omit'
if contains_nan and nan_policy == 'omit':
    # 如果 axis 为 None，计算平均值 mn、标准差 std 和是否全为常数 isconst
    if axis is None:
        mn = _quiet_nanmean(a.ravel())  # 平均值，忽略 NaN
        std = _quiet_nanstd(a.ravel(), ddof=ddof)  # 标准差，忽略 NaN
        isconst = _isconst(a.ravel())  # 判断是否全部为常数
    else:
        # 沿着指定 axis 轴计算 mn、std 和 isconst
        mn = np.apply_along_axis(_quiet_nanmean, axis, a)
        std = np.apply_along_axis(_quiet_nanstd, axis, a, ddof=ddof)
        isconst = np.apply_along_axis(_isconst, axis, a)
else:
    # 计算整体平均值 mn 和标准差 std，以及是否全为常数 isconst
    mn = a.mean(axis=axis, keepdims=True)  # 平均值
    std = a.std(axis=axis, ddof=ddof, keepdims=True)  # 标准差
    # 意图是检查数组 a 沿 axis 轴的所有元素是否相同
    # 由于有限精度算术，直接与平均值 mn 比较不适用
    # 此前用的是与 _first 比较，但它无视掩码。为简单起见，与最小值比较。
    a0 = a.min(axis=axis, keepdims=True)  # 最小值
    isconst = (a == a0).all(axis=axis, keepdims=True)  # 判断是否全部相同
    # 将标准差为0的值设为1，以避免除以0的情况。
    std[isconst] = 1.0
    # 计算标准化后的分数，使用平均值和标准差进行标准化。
    z = (scores - mn) / std
    # 将与常量输入相关联的输出设为 NaN（不是一个数字）。
    z[np.broadcast_to(isconst, z.shape)] = np.nan
    # 返回标准化后的分数数组。
    return z
# 定义函数 gstd，计算数组的几何标准差
def gstd(a, axis=0, ddof=1):
    """
    计算数组的几何标准差。

    几何标准差描述了一组数字的分散程度，其中几何平均值优先考虑。它是一个乘法因子，
    因此是一个无量纲的量。

    它定义为观测的自然对数的标准差的指数。

    Parameters
    ----------
    a : array_like
        包含有限、严格正实数的数组。

        .. deprecated:: 1.14.0
            对遮罩数组输入的支持在 SciPy 1.14.0 中已弃用，并将在版本 1.16.0 中删除。

    axis : int, tuple or None, optional
        操作的轴线。默认为 0。如果为 None，则在整个数组 `a` 上计算。
    ddof : int, optional
        在计算几何标准差时的自由度校正。默认为 1。

    Returns
    -------
    gstd : ndarray or float
        几何标准差的数组。如果 `axis` 是 None 或者 `a` 是一维数组，则返回一个浮点数。

    See Also
    --------
    gmean : 几何平均值
    numpy.std : 标准差
    gzscore : 几何标准化得分

    Notes
    -----
    从数学上讲，样本的几何标准差 :math:`s_G` 可以用观测的自然对数 :math:`y_i = \log(x_i)` 来定义：

    .. math::

        s_G = \exp(s), \quad s = \sqrt{\frac{1}{n - d} \sum_{i=1}^n (y_i - \bar y)^2}

    其中 :math:`n` 是观测数量，:math:`d` 是自由度调整 `ddof`，:math:`\bar y` 表示观测的自然对数的平均值。
    注意，默认的 ``ddof=1`` 与类似函数（例如 `numpy.std` 和 `numpy.var`）使用的默认值不同。

    当观测为无穷大时，几何标准差为 NaN（未定义）。非正数观测也会导致输出为 NaN，因为*自然*对数
    （而不是*复*对数）仅对正实数定义和有限。

    几何标准差有时会与标准差的指数函数混淆，即 ``exp(std(a))``。相反，几何标准差是 ``exp(std(log(a)))``。

    References
    ----------
    .. [1] "Geometric standard deviation", *Wikipedia*,
           https://en.wikipedia.org/wiki/Geometric_standard_deviation.
    .. [2] Kirkwood, T. B., "Geometric means and measures of dispersion",
           Biometrics, vol. 35, pp. 908-909, 1979

    Examples
    --------
    找到对数正态分布样本的几何标准差。
    注意，分布的标准差为 1；在对数尺度上，这相当于约为 ``exp(1)``。

    >>> import numpy as np
    >>> from scipy.stats import gstd
    """
    # 使用 NumPy 提供的默认随机数生成器创建一个 RNG 对象
    rng = np.random.default_rng()
    # 从对数正态分布中抽取 1000 个样本
    sample = rng.lognormal(mean=0, sigma=1, size=1000)
    # 调用函数 gstd 计算样本的几何标准差
    gstd(sample)
    # 期望输出结果为 2.810010162475324
    
    Compute the geometric standard deviation of a multidimensional array and
    of a given axis.
    
    # 创建一个形状为 (2, 3, 4) 的 NumPy 数组
    a = np.arange(1, 25).reshape(2, 3, 4)
    # 调用 gstd 函数计算整个数组的几何标准差
    gstd(a, axis=None)
    # 期望输出结果为 2.2944076136018947
    # 调用 gstd 函数计算数组在 axis=2 轴上的几何标准差
    gstd(a, axis=2)
    # 期望输出为一个形状为 (2, 3) 的数组，每个元素表示在对应子数组上的几何标准差
    array([[1.82424757, 1.22436866, 1.13183117],
           [1.09348306, 1.07244798, 1.05914985]])
    # 调用 gstd 函数计算数组在轴 (1,2) 上的几何标准差
    gstd(a, axis=(1,2))
    # 期望输出为一个包含两个元素的数组，分别表示对应子数组的几何标准差
    array([2.12939215, 1.22120169])
    
    """
    # 将输入参数 a 转换为 NumPy 的数组（不管输入是否是数组或者掩码数组）
    a = np.asanyarray(a)
    # 如果输入是掩码数组，则给出警告信息，表明在将来的版本中将移除对掩码数组的支持
    if isinstance(a, ma.MaskedArray):
        message = ("`gstd` support for masked array input was deprecated in "
                   "SciPy 1.14.0 and will be removed in version 1.16.0.")
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        # 对数函数使用掩码数组的 log
        log = ma.log
    else:
        # 对数函数使用 NumPy 的 log
        log = np.log
    
    # 忽略除以零和无效操作的错误
    with np.errstate(invalid='ignore', divide='ignore'):
        # 计算对数后的标准差，并将其指数化以得到几何标准差
        res = np.exp(np.std(log(a), axis=axis, ddof=ddof))
    
    # 如果数组中存在小于等于零的元素，则给出警告，说明几何标准差只在所有元素大于等于零时定义
    if (a <= 0).any():
        message = ("The geometric standard deviation is only defined if all elements "
                   "are greater than or equal to zero; otherwise, the result is NaN.")
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    
    # 返回计算得到的几何标准差
    return res
#`
# Private dictionary initialized only once at module level
# See https://en.wikipedia.org/wiki/Robust_measures_of_scale
_scale_conversions = {'normal': special.erfinv(0.5) * 2.0 * math.sqrt(2.0)}

# Decorator to handle axis and NaN policy for the iqr function
@_axis_nan_policy_factory(
    lambda x: x, result_to_tuple=lambda x: (x,), n_outputs=1,  # Define function to handle axis and return tuple
    default_axis=None, override={'nan_propagation': False}  # Default axis and nan propagation policy
)
def iqr(x, axis=None, rng=(25, 75), scale=1.0, nan_policy='propagate', interpolation='linear', keepdims=False):
    r"""
    Compute the interquartile range of the data along the specified axis.

    The interquartile range (IQR) is the difference between the 75th and
    25th percentile of the data. It is a measure of the dispersion
    similar to standard deviation or variance, but is much more robust
    against outliers [2]_.

    The ``rng`` parameter allows this function to compute other
    percentile ranges than the actual IQR. For example, setting
    ``rng=(0, 100)`` is equivalent to `numpy.ptp`.

    The IQR of an empty array is `np.nan`.

    .. versionadded:: 0.18.0

    Parameters
    ----------
    x : array_like
        Input array or object that can be converted to an array.
    axis : int or sequence of int, optional
        Axis along which the range is computed. The default is to
        compute the IQR for the entire array.
    rng : Two-element sequence containing floats in range of [0,100] optional
        Percentiles over which to compute the range. Each must be
        between 0 and 100, inclusive. The default is the true IQR:
        ``(25, 75)``. The order of the elements is not important.
    scale : scalar or str or array_like of reals, optional
        The numerical value of scale will be divided out of the final
        result. The following string value is also recognized:

          * 'normal' : Scale by
            :math:`2 \sqrt{2} erf^{-1}(\frac{1}{2}) \approx 1.349`.

        The default is 1.0.
        Array-like `scale` of real dtype is also allowed, as long
        as it broadcasts correctly to the output such that
        ``out / scale`` is a valid operation. The output dimensions
        depend on the input array, `x`, the `axis` argument, and the
        `keepdims` flag.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

          * 'propagate': returns nan
          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values
    interpolation : str, optional
        插值方法，指定在百分位数边界介于两个数据点 `i` 和 `j` 之间时使用的插值方法。
        可用选项如下（默认为 'linear'）：

          * 'linear': ``i + (j - i)*fraction``，其中 ``fraction`` 是介于 `i` 和 `j` 之间的索引的小数部分。
          * 'lower': ``i``。
          * 'higher': ``j``。
          * 'nearest': 最接近的 ``i`` 或 ``j``。
          * 'midpoint': ``(i + j)/2``。

        对于 NumPy >= 1.22.0，`numpy.percentile` 的 `method` 关键字提供的额外选项也是有效的。

    keepdims : bool, optional
        如果设置为 True，则将减少的轴保留在结果中作为尺寸为一的维度。
        使用此选项，结果将正确广播到原始数组 `x`。

    Returns
    -------
    iqr : scalar or ndarray
        如果 ``axis=None``，则返回标量。如果输入包含小于 ``np.float64`` 精度的整数或浮点数，则输出数据类型为 ``np.float64``。
        否则，输出数据类型与输入的数据类型相同。

    See Also
    --------
    numpy.std, numpy.var

    References
    ----------
    .. [1] "Interquartile range" https://en.wikipedia.org/wiki/Interquartile_range
    .. [2] "Robust measures of scale" https://en.wikipedia.org/wiki/Robust_measures_of_scale
    .. [3] "Quantile" https://en.wikipedia.org/wiki/Quantile

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import iqr
    >>> x = np.array([[10, 7, 4], [3, 2, 1]])
    >>> x
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> iqr(x)
    4.0
    >>> iqr(x, axis=0)
    array([ 3.5,  2.5,  1.5])
    >>> iqr(x, axis=1)
    array([ 3.,  1.])
    >>> iqr(x, axis=1, keepdims=True)
    array([[ 3.],
           [ 1.]])

    """
    x = asarray(x)  # 将输入参数 x 转换为 NumPy 数组

    # 这个检查防止后续百分位数函数引发错误。这也与 `np.var` 和 `np.std` 的行为保持一致。
    if not x.size:
        return _get_nan(x)  # 如果 x 为空，则返回 NaN 结果

    # 在进行复杂计算之前，这里进行检查，防止 `scale` 在后面被使用时引发错误。
    if isinstance(scale, str):
        scale_key = scale.lower()
        if scale_key not in _scale_conversions:
            raise ValueError(f"{scale} not a valid scale for `iqr`")
        scale = _scale_conversions[scale_key]

    # 根据是否包含 NaN 值和 NaN 策略，选择使用的百分位数函数
    contains_nan, nan_policy = _contains_nan(x, nan_policy)

    if contains_nan and nan_policy == 'omit':
        percentile_func = np.nanpercentile  # 如果包含 NaN 并且策略为忽略，则使用 np.nanpercentile
    else:
        percentile_func = np.percentile  # 否则使用 np.percentile

    if len(rng) != 2:
        raise TypeError("quantile range must be two element sequence")  # 如果 rng 不是长度为两的序列，则引发类型错误

    if np.isnan(rng).any():
        raise ValueError("range must not contain NaNs")  # 如果 rng 包含 NaN 值，则引发值错误

    rng = sorted(rng)  # 对 rng 进行排序
    # 调用自定义的百分位数计算函数，计算数组 x 在指定轴上的百分位数
    pct = percentile_func(x, rng, axis=axis, method=interpolation,
                          keepdims=keepdims)
    
    # 计算百分位数数组中第二个百分位数与第一个百分位数的差值
    out = np.subtract(pct[1], pct[0])
    
    # 如果指定了除数 scale 不为 1.0，则将 out 数组中的每个元素除以 scale
    if scale != 1.0:
        out /= scale
    
    # 返回处理后的结果数组 out
    return out
# Median absolute deviation for 1-d array x.
# This is a helper function for `median_abs_deviation`; it assumes its
# arguments have been validated already.  In particular,  x must be a
# 1-d numpy array, center must be callable, and if nan_policy is not
# 'propagate', it is assumed to be 'omit', because 'raise' is handled
# in `median_abs_deviation`.
# No warning is generated if x is empty or all nan.
def _mad_1d(x, center, nan_policy):
    # Check for NaN values in array x
    isnan = np.isnan(x)
    # Handle NaN values based on nan_policy
    if isnan.any():
        if nan_policy == 'propagate':
            # If nan_policy is 'propagate', return NaN
            return np.nan
        # Otherwise, remove NaN values from array x
        x = x[~isnan]
    # If array x is empty after NaN removal, return NaN
    if x.size == 0:
        return np.nan
    # Compute the median of array x using the specified center function
    med = center(x)
    # Compute the median absolute deviation (MAD) of array x
    mad = np.median(np.abs(x - med))
    return mad


def median_abs_deviation(x, axis=0, center=np.median, scale=1.0,
                         nan_policy='propagate'):
    r"""
    Compute the median absolute deviation of the data along the given axis.

    The median absolute deviation (MAD, [1]_) computes the median over the
    absolute deviations from the median. It is a measure of dispersion
    similar to the standard deviation but more robust to outliers [2]_.

    The MAD of an empty array is ``np.nan``.

    .. versionadded:: 1.5.0

    Parameters
    ----------
    x : array_like
        Input array or object that can be converted to an array.
    axis : int or None, optional
        Axis along which the range is computed. Default is 0. If None, compute
        the MAD over the entire array.
    center : callable, optional
        A function that will return the central value. The default is to use
        np.median. Any user defined function used will need to have the
        function signature ``func(arr, axis)``.
    scale : scalar or str, optional
        The numerical value of scale will be divided out of the final
        result. The default is 1.0. The string "normal" is also accepted,
        and results in `scale` being the inverse of the standard normal
        quantile function at 0.75, which is approximately 0.67449.
        Array-like scale is also allowed, as long as it broadcasts correctly
        to the output such that ``out / scale`` is a valid operation. The
        output dimensions depend on the input array, `x`, and the `axis`
        argument.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

        * 'propagate': returns nan
        * 'raise': throws an error
        * 'omit': performs the calculations ignoring nan values

    Returns
    -------
    ```
    # 如果 `center` 参数不是可调用对象，则抛出类型错误异常，提示必须是可调用的
    if not callable(center):
        raise TypeError("The argument 'center' must be callable. The given "
                        f"value {repr(center)} is not callable.")

    # 在执行较长的计算之前，尽早检测可能出现的错误，即使 `scale` 参数在后续才会使用
    # 如果 `scale` 是字符串类型，则根据其值进行特定处理
    if isinstance(scale, str):
        if scale.lower() == 'normal':
            # 当 `scale` 参数为 'normal' 时，设置特定的标准化值
            scale = 0.6744897501960817  # special.ndtri(0.75)
        else:
            # 如果 `scale` 不是合法的字符串值，则抛出值错误异常
            raise ValueError(f"{scale} is not a valid scale value.")

    # 将输入 `x` 转换为数组格式，以便后续处理
    x = asarray(x)

    # 与 `np.var` 和 `np.std` 的行为保持一致。
    # 如果数组 x 的大小为零（即空数组）
    if not x.size:
        # 如果没有指定 axis（轴），返回 NaN
        if axis is None:
            return np.nan
        # 根据指定的 axis 计算 nan_shape，排除该轴的维度
        nan_shape = tuple(item for i, item in enumerate(x.shape) if i != axis)
        # 如果 nan_shape 是空元组，意味着数组 x 只有一个元素，返回 NaN
        if nan_shape == ():
            # 返回 NaN，而不是 array(nan)
            return np.nan
        # 返回一个全为 NaN 的数组，形状为 nan_shape
        return np.full(nan_shape, np.nan)

    # 检查数组 x 是否包含 NaN，并获取 NaN 策略
    contains_nan, nan_policy = _contains_nan(x, nan_policy)

    # 如果数组包含 NaN
    if contains_nan:
        # 如果没有指定 axis，使用 _mad_1d 计算中心化后的 MAD
        if axis is None:
            mad = _mad_1d(x.ravel(), center, nan_policy)
        else:
            # 在指定的 axis 上应用 _mad_1d 函数，计算中心化后的 MAD
            mad = np.apply_along_axis(_mad_1d, axis, x, center, nan_policy)
    else:
        # 如果数组不包含 NaN
        # 如果没有指定 axis，计算整个数组的中位数并计算 MAD
        if axis is None:
            med = center(x, axis=None)  # 计算整个数组 x 的中心位置
            mad = np.median(np.abs(x - med))  # 计算中心化后的绝对偏差 MAD
        else:
            # 在指定的 axis 上计算中心位置，并使用 expand_dims 包装以使其行为类似于 keepdims=True
            med = np.expand_dims(center(x, axis=axis), axis)
            # 计算在指定 axis 上中心化后的绝对偏差 MAD
            mad = np.median(np.abs(x - med), axis=axis)

    # 返回标准化后的 MAD 值
    return mad / scale
# 定义一个具名元组 SigmaclipResult，用于存储 sigmaclip 函数的返回结果
SigmaclipResult = namedtuple('SigmaclipResult', ('clipped', 'lower', 'upper'))


def sigmaclip(a, low=4., high=4.):
    """Perform iterative sigma-clipping of array elements.

    Starting from the full sample, all elements outside the critical range are
    removed, i.e. all elements of the input array `c` that satisfy either of
    the following conditions::

        c < mean(c) - std(c)*low
        c > mean(c) + std(c)*high

    The iteration continues with the updated sample until no
    elements are outside the (updated) range.

    Parameters
    ----------
    a : array_like
        Data array, will be raveled if not 1-D.
    low : float, optional
        Lower bound factor of sigma clipping. Default is 4.
    high : float, optional
        Upper bound factor of sigma clipping. Default is 4.

    Returns
    -------
    clipped : ndarray
        Input array with clipped elements removed.
    lower : float
        Lower threshold value use for clipping.
    upper : float
        Upper threshold value use for clipping.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import sigmaclip
    >>> a = np.concatenate((np.linspace(9.5, 10.5, 31),
    ...                     np.linspace(0, 20, 5)))
    >>> fact = 1.5
    >>> c, low, upp = sigmaclip(a, fact, fact)
    >>> c
    array([  9.96666667,  10.        ,  10.03333333,  10.        ])
    >>> c.var(), c.std()
    (0.00055555555555555165, 0.023570226039551501)
    >>> low, c.mean() - fact*c.std(), c.min()
    (9.9646446609406727, 9.9646446609406727, 9.9666666666666668)
    >>> upp, c.mean() + fact*c.std(), c.max()
    (10.035355339059327, 10.035355339059327, 10.033333333333333)

    >>> a = np.concatenate((np.linspace(9.5, 10.5, 11),
    ...                     np.linspace(-100, -50, 3)))
    >>> c, low, upp = sigmaclip(a, 1.8, 1.8)
    >>> (c == np.linspace(9.5, 10.5, 11)).all()
    True

    """
    # 将输入数组转换为一维 ndarray
    c = np.asarray(a).ravel()
    # 初始化 delta 为非零以进入循环
    delta = 1
    while delta:
        # 计算当前样本的标准差和均值
        c_std = c.std()
        c_mean = c.mean()
        size = c.size
        # 计算 sigma-clip 的下限和上限
        critlower = c_mean - c_std * low
        critupper = c_mean + c_std * high
        # 仅保留在 critlower 和 critupper 范围内的元素
        c = c[(c >= critlower) & (c <= critupper)]
        # 更新 delta，用于判断是否还有元素被移除
        delta = size - c.size

    # 返回 sigma-clip 处理后的结果，包括剪切后的数组以及下限和上限值
    return SigmaclipResult(c, critlower, critupper)


def trimboth(a, proportiontocut, axis=0):
    """Slice off a proportion of items from both ends of an array.

    Slice off the passed proportion of items from both ends of the passed
    array (i.e., with `proportiontocut` = 0.1, slices leftmost 10% **and**
    rightmost 10% of scores). The trimmed values are the lowest and
    highest ones.
    Slice off less if proportion results in a non-integer slice index (i.e.
    conservatively slices off `proportiontocut`).

    Parameters
    ----------
    a : array_like
        Data to trim.
    # 将输入的数据转换为 NumPy 数组
    a = np.asarray(a)
    
    # 如果数组为空，则直接返回空数组
    if a.size == 0:
        return a
    
    # 如果指定了 axis 为 None，则将数组展平，并设置 axis 为 0
    if axis is None:
        a = a.ravel()
        axis = 0
    
    # 获取沿指定 axis 轴的观测值数量
    nobs = a.shape[axis]
    
    # 根据 proportiontocut 计算要从两端截取的元素数量
    lowercut = int(proportiontocut * nobs)
    uppercut = nobs - lowercut
    
    # 检查 lowercut 和 uppercut 是否合理，即 lowercut 是否小于 uppercut
    if lowercut >= uppercut:
        raise ValueError("Proportion too big.")
    
    # 对数组 a 沿指定 axis 轴进行分区，使得指定区间内的元素被放置在正确的位置
    atmp = np.partition(a, (lowercut, uppercut - 1), axis)
    
    # 构建切片列表，用于从 atmp 中选择指定的切片
    sl = [slice(None)] * atmp.ndim
    sl[axis] = slice(lowercut, uppercut)
    
    # 返回根据计算的切片选择的结果数组
    return atmp[tuple(sl)]
# 定义函数 `trim1`，用于从数组的一个端口切掉指定比例的数据分布。

def trim1(a, proportiontocut, tail='right', axis=0):
    """Slice off a proportion from ONE end of the passed array distribution.

    如果 `proportiontocut` = 0.1，切掉数组分布的 'leftmost' 或 'rightmost' 10% 的分数。
    根据 `tail` 参数决定是切掉最低值还是最高值。
    如果 `proportiontocut` 的结果不是整数，则谨慎地切掉 `proportiontocut`。

    Parameters
    ----------
    a : array_like
        输入的数组。
    proportiontocut : float
        要从分布的 'left' 或 'right' 切掉的比例。
    tail : {'left', 'right'}, optional
        默认为 'right'。
    axis : int or None, optional
        要修剪数据的轴。默认为 0。如果为 None，则计算整个数组 `a`。

    Returns
    -------
    trim1 : ndarray
        数组 `a` 的修剪版本。修剪后内容的顺序未定义。

    Examples
    --------
    创建一个包含 10 个值的数组，并切掉最低值的 20%：

    >>> import numpy as np
    >>> from scipy import stats
    >>> a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> stats.trim1(a, 0.2, 'left')
    array([2, 4, 3, 5, 6, 7, 8, 9])

    注意，输入数组的元素按值修剪，但输出数组不一定按顺序排列。

    要修剪的比例被向下舍入到最接近的整数。例如，从包含 10 个值的数组中修剪 25% 的值将返回包含 8 个值的数组：

    >>> b = np.arange(10)
    >>> stats.trim1(b, 1/4).shape
    (8,)

    多维数组可以沿任何轴修剪或跨整个数组修剪：

    >>> c = [2, 4, 6, 8, 0, 1, 3, 5, 7, 9]
    >>> d = np.array([a, b, c])
    >>> stats.trim1(d, 0.8, axis=0).shape
    (1, 10)
    >>> stats.trim1(d, 0.8, axis=1).shape
    (3, 2)
    >>> stats.trim1(d, 0.8, axis=None).shape
    (6,)

    """
    # 将输入数组 `a` 转换为 NumPy 数组
    a = np.asarray(a)
    # 如果 `axis` 是 None，则将数组展平，并将 `axis` 设为 0
    if axis is None:
        a = a.ravel()
        axis = 0

    # 获取数组 `a` 在指定轴上的元素数量
    nobs = a.shape[axis]

    # 避免可能的边界情况
    if proportiontocut >= 1:
        return []

    # 根据 `tail` 参数设置修剪的下限和上限
    if tail.lower() == 'right':
        lowercut = 0
        uppercut = nobs - int(proportiontocut * nobs)

    elif tail.lower() == 'left':
        lowercut = int(proportiontocut * nobs)
        uppercut = nobs

    # 对数组 `a` 在指定轴上的一部分进行分区，以获取修剪后的数组
    atmp = np.partition(a, (lowercut, uppercut - 1), axis)

    # 创建切片对象，以便从 `atmp` 中获取修剪后的内容
    sl = [slice(None)] * atmp.ndim
    sl[axis] = slice(lowercut, uppercut)
    # 返回修剪后的数组
    return atmp[tuple(sl)]
    a = np.asarray(a)
    # 将输入的数组转换为 NumPy 数组，确保可以进行数组操作

    if a.size == 0:
        return np.nan
    # 如果数组为空，返回 NaN

    if axis is None:
        a = a.ravel()
        axis = 0
    # 如果 axis 为 None，则将数组展平，并设置 axis 为 0

    nobs = a.shape[axis]
    # 获取沿指定轴的数组形状

    lowercut = int(proportiontocut * nobs)
    uppercut = nobs - lowercut
    # 根据 proportiontocut 计算要剪切的元素数目

    if (lowercut > uppercut):
        raise ValueError("Proportion too big.")
    # 如果 lowercut 大于 uppercut，则抛出 ValueError 异常

    atmp = np.partition(a, (lowercut, uppercut - 1), axis)
    # 对数组进行部分排序，以便获取需要保留的元素

    sl = [slice(None)] * atmp.ndim
    sl[axis] = slice(lowercut, uppercut)
    # 创建一个切片对象，用于选择要计算平均值的部分

    return np.mean(atmp[tuple(sl)], axis=axis)
    # 返回沿指定轴计算的平均值
F_onewayResult = namedtuple('F_onewayResult', ('statistic', 'pvalue'))

# Helper function for f_oneway to create nan results in degenerate conditions
def _create_f_oneway_nan_result(shape, axis, samples):
    """
    This is a helper function for f_oneway for creating the return values
    in certain degenerate conditions.  It creates return values that are
    all nan with the appropriate shape for the given `shape` and `axis`.
    """
    axis = normalize_axis_index(axis, len(shape))
    shp = shape[:axis] + shape[axis+1:]
    # Create arrays filled with nan values
    f = np.full(shp, fill_value=_get_nan(*samples))
    prob = f.copy()
    return F_onewayResult(f[()], prob[()])

# Return arr[..., 0:1, ...] where 0:1 is in the `axis` position
def _first(arr, axis):
    """Return arr[..., 0:1, ...] where 0:1 is in the `axis` position."""
    return np.take_along_axis(arr, np.array(0, ndmin=arr.ndim), axis)

# Checks if samples for f_oneway are too small; raises TypeError if < 2 samples
def _f_oneway_is_too_small(samples, kwargs={}, axis=-1):
    message = f"At least two samples are required; got {len(samples)}."
    if len(samples) < 2:
        raise TypeError(message)

    # Check if any sample along the specified axis has length 0
    if any(sample.shape[axis] == 0 for sample in samples):
        return True

    # Warns if all samples along the axis have length 1
    if all(sample.shape[axis] == 1 for sample in samples):
        msg = ('all input arrays have length 1.  f_oneway requires that at '
               'least one input has length greater than 1.')
        warnings.warn(SmallSampleWarning(msg), stacklevel=2)
        return True

    return False

# Decorated function for f_oneway with axis nan policy
@_axis_nan_policy_factory(
    F_onewayResult, n_samples=None, too_small=_f_oneway_is_too_small)
def f_oneway(*samples, axis=0):
    """Perform one-way ANOVA.

    The one-way ANOVA tests the null hypothesis that two or more groups have
    the same population mean.  The test is applied to samples from two or
    more groups, possibly with differing sizes.

    Parameters
    ----------
    sample1, sample2, ... : array_like
        The sample measurements for each group.  There must be at least
        two arguments.  If the arrays are multidimensional, then all the
        dimensions of the array must be the same except for `axis`.
    axis : int, optional
        Axis of the input arrays along which the test is applied.
        Default is 0.

    Returns
    -------
    statistic : float
        The computed F statistic of the test.
    pvalue : float
        The associated p-value from the F distribution.

    Warns
    -----
    `~scipy.stats.ConstantInputWarning`
        Emitted if all values within each of the input arrays are identical.
        In this case the F statistic is either infinite or isn't defined,
        so ``np.inf`` or ``np.nan`` is returned.

    RuntimeWarning
        Emitted if the length of any input array is 0, or if all the input
        arrays have length 1.  ``np.nan`` is returned for the F statistic
        and the p-value in these cases.

    Notes
    -----
    The ANOVA test has important assumptions that must be satisfied in order
    """
    for the associated p-value to be valid.

    1. The samples are independent.
    2. Each sample is from a normally distributed population.
    3. The population standard deviations of the groups are all equal.  This
       property is known as homoscedasticity.

    If these assumptions are not true for a given set of data, it may still
    be possible to use the Kruskal-Wallis H-test (`scipy.stats.kruskal`) or
    the Alexander-Govern test (`scipy.stats.alexandergovern`) although with
    some loss of power.

    The length of each group must be at least one, and there must be at
    least one group with length greater than one.  If these conditions
    are not satisfied, a warning is generated and (``np.nan``, ``np.nan``)
    is returned.

    If all values in each group are identical, and there exist at least two
    groups with different values, the function generates a warning and
    returns (``np.inf``, 0).

    If all values in all groups are the same, function generates a warning
    and returns (``np.nan``, ``np.nan``).

    The algorithm is from Heiman [2]_, pp.394-7.

    References
    ----------
    .. [1] R. Lowry, "Concepts and Applications of Inferential Statistics",
           Chapter 14, 2014, http://vassarstats.net/textbook/

    .. [2] G.W. Heiman, "Understanding research methods and statistics: An
           integrated introduction for psychology", Houghton, Mifflin and
           Company, 2001.

    .. [3] G.H. McDonald, "Handbook of Biological Statistics", One-way ANOVA.
           http://www.biostathandbook.com/onewayanova.html

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import f_oneway

    Here are some data [3]_ on a shell measurement (the length of the anterior
    adductor muscle scar, standardized by dividing by length) in the mussel
    Mytilus trossulus from five locations: Tillamook, Oregon; Newport, Oregon;
    Petersburg, Alaska; Magadan, Russia; and Tvarminne, Finland, taken from a
    much larger data set used in McDonald et al. (1991).

    >>> tillamook = [0.0571, 0.0813, 0.0831, 0.0976, 0.0817, 0.0859, 0.0735,
    ...              0.0659, 0.0923, 0.0836]
    >>> newport = [0.0873, 0.0662, 0.0672, 0.0819, 0.0749, 0.0649, 0.0835,
    ...            0.0725]
    >>> petersburg = [0.0974, 0.1352, 0.0817, 0.1016, 0.0968, 0.1064, 0.105]
    >>> magadan = [0.1033, 0.0915, 0.0781, 0.0685, 0.0677, 0.0697, 0.0764,
    ...            0.0689]
    >>> tvarminne = [0.0703, 0.1026, 0.0956, 0.0973, 0.1039, 0.1045]
    >>> f_oneway(tillamook, newport, petersburg, magadan, tvarminne)
    F_onewayResult(statistic=7.121019471642447, pvalue=0.0002812242314534544)

    `f_oneway` accepts multidimensional input arrays.  When the inputs
    are multidimensional and `axis` is not given, the test is performed
    along the first axis of the input arrays.  For the following data, the
    test is performed three times, once for each column.

    >>> a = np.array([[9.87, 9.03, 6.81],


注释：
    """
    # 如果样本数量小于2，则抛出类型错误，要求至少两个输入样本
    if len(samples) < 2:
        raise TypeError('at least two inputs are required;'
                        f' got {len(samples)}.')

    # 计算样本的数量
    num_groups = len(samples)

    # 检查轴参数的有效性，如果轴参数无效，np.concatenate 会引发 AxisError 错误。
    # 如果除了轴维度以外的所有数组维度不相同，则会引发 ValueError 错误。
    alldata = np.concatenate(samples, axis=axis)
    bign = alldata.shape[axis]

    # 检查输入样本是否过小，若过小则返回 NaN 结果
    if _f_oneway_is_too_small(samples):
        return _create_f_oneway_nan_result(alldata.shape, axis, samples)

    # 检查每个组内的值是否完全相同，以及至少一个组内的公共值是否与另一个组的不同。
    # 参考 https://github.com/scipy/scipy/issues/11669

    # 如果 axis=0，例如，且各组的形状为 (n0, ...)、(n1, ...)，则 is_const 是一个布尔数组，
    # 形状为 (num_groups, ...)。如果沿着轴切片的组内值相同，则为 True。
    is_const = np.concatenate(
        [(_first(sample, axis) == sample).all(axis=axis,
                                              keepdims=True)
         for sample in samples],
        axis=axis
    )

    # all_const 是一个布尔数组，形状为 (...)（参见上一个注释）。
    # 如果沿着轴切片的每个组内的值相同，则为 True。
    all_const = is_const.all(axis=axis)
    if all_const.any():
        # 如果所有组的值都相同，则给出警告信息，F 统计量未定义或无穷大。
        msg = ("Each of the input arrays is constant; "
               "the F statistic is not defined or infinite")
        warnings.warn(stats.ConstantInputWarning(msg), stacklevel=2)

    # 如果所有组沿 axis=0 切片的值都相同，则 all_same_const 为 True。
    # 例如 [[3, 3, 3], [3, 3, 3, 3], [3, 3, 3]]。
    # 检查所有数据是否都是同一个常数
    all_same_const = (_first(alldata, axis) == alldata).all(axis=axis)

    # 计算数据的均值，并将其从所有输入数据中减去，以进行方差计算
    # 通过 sum_of_sq / sq_of_sum 方法。方差对位置的偏移不变，将所有数据
    # 居中到零极大地提高了数值稳定性。
    offset = alldata.mean(axis=axis, keepdims=True)
    alldata = alldata - offset

    # 标准化后的平方和除以样本数量，用于总平方和的计算
    normalized_ss = _square_of_sums(alldata, axis=axis) / bign

    # 计算总平方和减去标准化后的平方和，得到总离差平方和
    sstot = _sum_of_squares(alldata, axis=axis) - normalized_ss

    # 初始化组间平方和
    ssbn = 0
    for sample in samples:
        # 计算每个样本减去均值后的平方和，除以样本的轴数，累加到组间平方和
        smo_ss = _square_of_sums(sample - offset, axis=axis)
        ssbn = ssbn + smo_ss / sample.shape[axis]

    # 命名规则：以 bn/b 结尾的变量用于“组间处理”，以 wn/w 结尾的变量用于“组内处理”
    # 调整组间平方和，减去标准化后的平方和，得到调整后的组间平方和
    ssbn = ssbn - normalized_ss
    sswn = sstot - ssbn

    # 计算自由度
    dfbn = num_groups - 1
    dfwn = bign - num_groups

    # 计算均方差
    msb = ssbn / dfbn
    msw = sswn / dfwn

    # 使用 np.errstate 设置错误处理，忽略除零和无效值错误
    with np.errstate(divide='ignore', invalid='ignore'):
        # 计算 F 统计量
        f = msb / msw

    # 使用 special.fdtrc 函数计算 F 统计量对应的概率，等效于 stats.f.sf
    prob = special.fdtrc(dfbn, dfwn, f)

    # 修正因输入数据常数导致的无穷值或 NaN 值的 F 统计量和概率
    if np.isscalar(f):
        if all_same_const:
            f = np.nan
            prob = np.nan
        elif all_const:
            f = np.inf
            prob = 0.0
    else:
        f[all_const] = np.inf
        prob[all_const] = 0.0
        f[all_same_const] = np.nan
        prob[all_same_const] = np.nan

    # 返回 F 检验的结果对象
    return F_onewayResult(f, prob)
# 使用 dataclass 装饰器定义了一个名为 AlexanderGovernResult 的数据类
@dataclass
class AlexanderGovernResult:
    statistic: float  # 类中的属性，用于存储统计量
    pvalue: float  # 类中的属性，用于存储 p 值


# 定义了一个带有装饰器 @_axis_nan_policy_factory 的函数 alexandergovern
# 该装饰器对 AlexanderGovernResult 类进行了处理，设置了一些参数和行为
@_axis_nan_policy_factory(
    AlexanderGovernResult, n_samples=None,
    result_to_tuple=lambda x: (x.statistic, x.pvalue),
    too_small=1
)
def alexandergovern(*samples, nan_policy='propagate'):
    """Performs the Alexander Govern test.

    The Alexander-Govern approximation tests the equality of k independent
    means in the face of heterogeneity of variance. The test is applied to
    samples from two or more groups, possibly with differing sizes.

    Parameters
    ----------
    sample1, sample2, ... : array_like
        The sample measurements for each group.  There must be at least
        two samples, and each sample must contain at least two observations.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

        * 'propagate': returns nan
        * 'raise': throws an error
        * 'omit': performs the calculations ignoring nan values

    Returns
    -------
    res : AlexanderGovernResult
        An object with attributes:

        statistic : float
            The computed A statistic of the test.
        pvalue : float
            The associated p-value from the chi-squared distribution.

    Warns
    -----
    `~scipy.stats.ConstantInputWarning`
        Raised if an input is a constant array.  The statistic is not defined
        in this case, so ``np.nan`` is returned.

    See Also
    --------
    f_oneway : one-way ANOVA

    Notes
    -----
    The use of this test relies on several assumptions.

    1. The samples are independent.
    2. Each sample is from a normally distributed population.
    3. Unlike `f_oneway`, this test does not assume on homoscedasticity,
       instead relaxing the assumption of equal variances.

    Input samples must be finite, one dimensional, and with size greater than
    one.

    References
    ----------
    .. [1] Alexander, Ralph A., and Diane M. Govern. "A New and Simpler
           Approximation for ANOVA under Variance Heterogeneity." Journal
           of Educational Statistics, vol. 19, no. 2, 1994, pp. 91-101.
           JSTOR, www.jstor.org/stable/1165140. Accessed 12 Sept. 2020.

    Examples
    --------
    >>> from scipy.stats import alexandergovern

    Here are some data on annual percentage rate of interest charged on
    new car loans at nine of the largest banks in four American cities
    taken from the National Institute of Standards and Technology's
    ANOVA dataset.

    We use `alexandergovern` to test the null hypothesis that all cities
    have the same mean APR against the alternative that the cities do not
    all have the same mean APR. We decide that a significance level of 5%
    is required to reject the null hypothesis in favor of the alternative.

    >>> atlanta = [13.75, 13.75, 13.5, 13.5, 13.0, 13.0, 13.0, 12.75, 12.5]
    # 输入的样本数据经过验证和处理，确保符合要求
    samples = _alexandergovern_input_validation(samples, nan_policy)

    # 检查是否有任何一个样本数据是常数，如果是，则无法定义统计量，返回空结果
    if np.any([(sample == sample[0]).all() for sample in samples]):
        msg = "An input array is constant; the statistic is not defined."
        # 发出警告，指出输入数据中存在常数数组
        warnings.warn(stats.ConstantInputWarning(msg), stacklevel=2)
        return AlexanderGovernResult(np.nan, np.nan)

    # 下面的公式编号参考了Alexander, Govern在第92页描述的方程式
    # 方程式5、6和7描述了其他测试，是方程式(8)的基础，但不需要执行测试

    # 预先计算每个样本的平均值和长度
    lengths = np.array([len(sample) for sample in samples])
    means = np.array([np.mean(sample) for sample in samples])

    # (1) 计算每个样本的均值标准误差
    standard_errors = [np.std(sample, ddof=1) / np.sqrt(length)
                       for sample, length in zip(samples, lengths)]

    # (2) 为每个样本定义权重
    inv_sq_se = 1 / np.square(standard_errors)
    weights = inv_sq_se / np.sum(inv_sq_se)

    # (3) 计算方差加权的公共均值估计
    var_w = np.sum(weights * means)

    # (4) 计算每组的单样本 t 统计量
    t_stats = (means - var_w) / standard_errors

    # 计算用于变换的参数
    v = lengths - 1
    a = v - .5
    b = 48 * a**2
    c = (a * np.log(1 + (t_stats ** 2)/v))**.5

    # (8) 对 t 统计量执行归一化变换
    z = (c + ((c**3 + 3*c) / b) -
         ((4*c**7 + 33*c**5 + 240*c**3 + 855*c) /
          (b**2 * 10 + 8*b*c**4 + 1000*b)))

    # (9) 计算统计量 A
    A = np.sum(np.square(z))

    # 根据“中心卡方随机偏差来确定的 p 值，自由度为 k - 1”。引用自Alexander, Govern (94)
    df = len(samples) - 1
    # 创建一个简单的卡方分布对象
    chi2 = _SimpleChi2(df)
    # 获取 p 值
    p = _get_pvalue(A, chi2, alternative='greater', symmetric=False, xp=np)
    return AlexanderGovernResult(A, p)
# 对输入样本和 NaN 处理策略进行验证
def _alexandergovern_input_validation(samples, nan_policy):
    # 如果样本数量少于2个，则抛出类型错误异常
    if len(samples) < 2:
        raise TypeError(f"2 or more inputs required, got {len(samples)}")

    # 遍历每个样本
    for sample in samples:
        # 如果样本大小小于等于1，则抛出数值错误异常
        if np.size(sample) <= 1:
            raise ValueError("Input sample size must be greater than one.")
        # 如果样本中包含无穷大值，则抛出数值错误异常
        if np.isinf(sample).any():
            raise ValueError("Input samples must be finite.")

    # 返回经过验证的样本
    return samples


def _pearsonr_fisher_ci(r, n, confidence_level, alternative):
    """
    Compute the confidence interval for Pearson's R.

    Fisher's transformation is used to compute the confidence interval
    (https://en.wikipedia.org/wiki/Fisher_transformation).
    """
    # 利用数组命名空间处理输入的相关系数 r
    xp = array_namespace(r)

    # 忽略除法产生的错误
    with np.errstate(divide='ignore'):
        # 计算 Fisher 变换后的相关系数 zr
        zr = xp.atanh(r)

    # 创建一个全为1的数组
    ones = xp.ones_like(r)
    # 将 n 和 confidence_level 转换为数组
    n, confidence_level = xp.asarray([n, confidence_level], dtype=r.dtype)
    # 如果样本量大于3
    if n > 3:
        # 计算标准误差
        se = xp.sqrt(1 / (n - 3))
        # 根据 alternative 不同计算不同类型的置信区间
        if alternative == "two-sided":
            # 使用正态分布逆累积分布函数计算 h 值
            h = special.ndtri(0.5 + confidence_level/2)
            # 计算置信区间的上下界
            zlo = zr - h*se
            zhi = zr + h*se
            rlo = xp.tanh(zlo)
            rhi = xp.tanh(zhi)
        elif alternative == "less":
            h = special.ndtri(confidence_level)
            zhi = zr + h*se
            rhi = xp.tanh(zhi)
            rlo = -ones
        else:
            # alternative == "greater":
            h = special.ndtri(confidence_level)
            zlo = zr - h*se
            rlo = xp.tanh(zlo)
            rhi = ones
    else:
        # 如果样本量不大于3，则置信区间为 [-1, 1]
        rlo, rhi = -ones, ones

    # 如果 rlo 或 rhi 是标量，则将其转换为标量
    rlo = rlo[()] if rlo.ndim == 0 else rlo
    rhi = rhi[()] if rhi.ndim == 0 else rhi
    # 返回计算得到的置信区间
    return ConfidenceInterval(low=rlo, high=rhi)


def _pearsonr_bootstrap_ci(confidence_level, method, x, y, alternative, axis):
    """
    Compute the confidence interval for Pearson's R using the bootstrap.
    """
    # 定义用于 bootstrap 的统计量函数
    def statistic(x, y, axis):
        # 计算皮尔逊相关系数和其 p 值
        statistic, _ = pearsonr(x, y, axis=axis)
        return statistic

    # 使用 bootstrap 方法计算相关系数的置信区间
    res = bootstrap((x, y), statistic, confidence_level=confidence_level, axis=axis,
                    paired=True, alternative=alternative, **method._asdict())
    # 对于单侧置信区间，将置信区间限制在 [-1, 1] 范围内
    res.confidence_interval = np.clip(res.confidence_interval, -1, 1)

    # 返回置信区间对象
    return ConfidenceInterval(*res.confidence_interval)


# 定义一个命名元组 ConfidenceInterval，表示相关系数的置信区间
ConfidenceInterval = namedtuple('ConfidenceInterval', ['low', 'high'])

# 创建一个用于包含 PearsonRResult 属性的命名元组基类
PearsonRResultBase = _make_tuple_bunch('PearsonRResultBase',
                                       ['statistic', 'pvalue'], [])


class PearsonRResult(PearsonRResultBase):
    """
    Result of `scipy.stats.pearsonr`

    Attributes
    ----------
    statistic : float
        Pearson product-moment correlation coefficient.
    pvalue : float
        The p-value associated with the chosen alternative.

    Methods
    -------
    confidence_interval
        Computes the confidence interval of the correlation
        coefficient `statistic` for the given confidence level.

    """
    # 统计量、P值、备择假设、样本数、变量x、变量y和轴线参数初始化
    def __init__(self, statistic, pvalue, alternative, n, x, y, axis):
        # 调用父类的初始化方法，传入统计量和P值
        super().__init__(statistic, pvalue)
        # 设置对象的备选假设属性
        self._alternative = alternative
        # 设置对象的样本数属性
        self._n = n
        # 设置对象的变量x属性
        self._x = x
        # 设置对象的变量y属性
        self._y = y
        # 设置对象的轴线参数属性
        self._axis = axis

        # 为了与其他相关函数的命名一致，添加correlation属性并赋值为statistic
        self.correlation = statistic
    def confidence_interval(self, confidence_level=0.95, method=None):
        """
        计算相关系数的置信区间。

        根据给定的置信水平计算相关系数 `statistic` 的置信区间。

        如果未提供 `method`，
        使用 Fisher 变换计算置信区间 F(r) = arctanh(r) [1]_。
        当样本对从双变量正态分布中抽取时，F(r) 近似服从标准误差为 `1/sqrt(n - 3)` 的正态分布，
        其中 `n` 是沿计算轴的原始样本长度。当 `n <= 3` 时，此近似不会产生有限的实数标准误差，
        因此我们将置信区间定义为 -1 到 1。

        如果 `method` 是 `BootstrapMethod` 的实例，
        则使用 `scipy.stats.bootstrap` 根据提供的配置选项和其他适当设置计算置信区间。
        在某些情况下，由于退化重采样，置信限可能为 NaN，这在非常小的样本（~6个观测值）中很常见。

        Parameters
        ----------
        confidence_level : float
            计算相关系数置信区间时的置信水平。默认为 0.95。

        method : BootstrapMethod, optional
            定义用于计算置信区间的方法。详细信息请参见方法描述。

            .. versionadded:: 1.11.0

        Returns
        -------
        ci : namedtuple
            返回一个 `namedtuple`，包含 `low` 和 `high` 两个字段，表示置信区间。

        Raises
        ------
        ValueError
            如果 `method` 不是 `BootstrapMethod` 的实例或为 `None` 时抛出异常。

        References
        ----------
        .. [1] "Pearson correlation coefficient", Wikipedia,
               https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
        """
        if isinstance(method, BootstrapMethod):
            xp = array_namespace(self._x)
            message = ('如果 `pearsonr` 的参数不是 NumPy 数组，则 `method` 必须为 `None`。')
            if not is_numpy(xp):
                raise ValueError(message)

            ci = _pearsonr_bootstrap_ci(confidence_level, method, self._x, self._y,
                                        self._alternative, self._axis)
        elif method is None:
            ci = _pearsonr_fisher_ci(self.statistic, self._n, confidence_level,
                                     self._alternative)
        else:
            message = ('`method` 必须是 `BootstrapMethod` 的实例或为 `None`。')
            raise ValueError(message)
        return ci
# 计算 Pearson 相关系数和检验相关性的 p 值

def pearsonr(x, y, *, alternative='two-sided', method=None, axis=0):
    r"""
    Pearson correlation coefficient and p-value for testing non-correlation.

    The Pearson correlation coefficient [1]_ measures the linear relationship
    between two datasets. Like other correlation
    coefficients, this one varies between -1 and +1 with 0 implying no
    correlation. Correlations of -1 or +1 imply an exact linear relationship.
    Positive correlations imply that as x increases, so does y. Negative
    correlations imply that as x increases, y decreases.

    This function also performs a test of the null hypothesis that the
    distributions underlying the samples are uncorrelated and normally
    distributed. (See Kowalski [3]_
    for a discussion of the effects of non-normality of the input on the
    distribution of the correlation coefficient.)
    The p-value roughly indicates the probability of an uncorrelated system
    producing datasets that have a Pearson correlation at least as extreme
    as the one computed from these datasets.

    Parameters
    ----------
    x : array_like
        Input array.
    y : array_like
        Input array.
    axis : int or None, default
        Axis along which to perform the calculation. Default is 0.
        If None, ravel both arrays before performing the calculation.

        .. versionadded:: 1.13.0
    alternative : {'two-sided', 'greater', 'less'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        The following options are available:

        * 'two-sided': the correlation is nonzero
        * 'less': the correlation is negative (less than zero)
        * 'greater':  the correlation is positive (greater than zero)

        .. versionadded:: 1.9.0
    method : ResamplingMethod, optional
        Defines the method used to compute the p-value. If `method` is an
        instance of `PermutationMethod`/`MonteCarloMethod`, the p-value is
        computed using
        `scipy.stats.permutation_test`/`scipy.stats.monte_carlo_test` with the
        provided configuration options and other appropriate settings.
        Otherwise, the p-value is computed as documented in the notes.

        .. versionadded:: 1.11.0

    Returns
    -------
    # result 是一个 `~scipy.stats._result_classes.PearsonRResult` 类型的对象，
    # 该对象具有以下属性和方法：

    # statistic 是一个浮点数，代表皮尔逊积矩相关系数。
    # pvalue 是一个浮点数，代表与选择的备择假设相关联的 p 值。

    # confidence_interval(confidence_level, method) 方法用于计算给定置信水平下相关系数 `statistic` 的置信区间。
    # 置信区间以一个命名元组的形式返回，包括 `low` 和 `high` 两个字段。
    # 如果未提供 `method` 参数，则使用 Fisher 变换 [1] 计算置信区间。
    # 如果 `method` 是 `BootstrapMethod` 的实例，则使用提供的配置选项和其他适当的设置使用 `scipy.stats.bootstrap` 计算置信区间。
    # 在某些情况下，由于重采样的退化，置信限可能为 NaN，在极小样本（约6个观测）的情况下这是典型的。

    # Warns
    # -----
    # `~scipy.stats.ConstantInputWarning`
    # 如果输入是一个常数数组，则引发此警告。在这种情况下相关系数未定义，因此返回 `np.nan`。
    
    # `~scipy.stats.NearConstantInputWarning`
    # 如果输入是“几乎”常数，则引发此警告。如果 `norm(x - mean(x)) < 1e-13 * abs(mean(x))`，则认为数组 `x` 是几乎常数。
    # 在这种情况下，计算 `x - mean(x)` 的数值误差可能导致 r 的不准确计算。

    # See Also
    # --------
    # spearmanr : 斯皮尔曼等级相关系数。
    # kendalltau : Kendall's tau，用于有序数据的相关度量。

    # Notes
    # -----
    # 相关系数的计算方式如下：

    # .. math::

    #     r = \frac{\sum (x - m_x) (y - m_y)}
    #              {\sqrt{\sum (x - m_x)^2 \sum (y - m_y)^2}}

    # 其中 :math:`m_x` 是向量 x 的均值，:math:`m_y` 是向量 y 的均值。

    # 在假设 x 和 y 是从独立的正态分布中抽取的情况下（因此总体相关系数为0），
    # 样本相关系数 r 的概率密度函数是 ([1]_, [2]_)：

    # .. math::
    #     f(r) = \frac{{(1-r^2)}^{n/2-2}}{\mathrm{B}(\frac{1}{2},\frac{n}{2}-1)}

    # 其中 n 是样本数量，B 是 beta 函数。这有时被称为 r 的精确分布。
    # 这是在 `pearsonr` 中用于计算 p 值时默认情况下（`method` 参数为 None 时）使用的分布。
    # 该分布是在区间 [-1, 1] 上的 beta 分布，具有相等的形状参数 a = b = n/2 - 1。
    # 就 SciPy 的术语而言，这是 `pearsonr` 所使用的分布。
    implementation of the beta distribution, the distribution of r is::

        dist = scipy.stats.beta(n/2 - 1, n/2 - 1, loc=-1, scale=2)

    The default p-value returned by `pearsonr` is a two-sided p-value. For a
    given sample with correlation coefficient r, the p-value is
    the probability that abs(r') of a random sample x' and y' drawn from
    the population with zero correlation would be greater than or equal
    to abs(r). In terms of the object ``dist`` shown above, the p-value
    for a given r and length n can be computed as::

        p = 2*dist.cdf(-abs(r))

    When n is 2, the above continuous distribution is not well-defined.
    One can interpret the limit of the beta distribution as the shape
    parameters a and b approach a = b = 0 as a discrete distribution with
    equal probability masses at r = 1 and r = -1.  More directly, one
    can observe that, given the data x = [x1, x2] and y = [y1, y2], and
    assuming x1 != x2 and y1 != y2, the only possible values for r are 1
    and -1.  Because abs(r') for any sample x' and y' with length 2 will
    be 1, the two-sided p-value for a sample of length 2 is always 1.

    For backwards compatibility, the object that is returned also behaves
    like a tuple of length two that holds the statistic and the p-value.

    References
    ----------
    .. [1] "Pearson correlation coefficient", Wikipedia,
           https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    .. [2] Student, "Probable error of a correlation coefficient",
           Biometrika, Volume 6, Issue 2-3, 1 September 1908, pp. 302-310.
    .. [3] C. J. Kowalski, "On the Effects of Non-Normality on the Distribution
           of the Sample Product-Moment Correlation Coefficient"
           Journal of the Royal Statistical Society. Series C (Applied
           Statistics), Vol. 21, No. 1 (1972), pp. 1-12.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> x, y = [1, 2, 3, 4, 5, 6, 7], [10, 9, 2.5, 6, 4, 3, 2]
    >>> res = stats.pearsonr(x, y)
    >>> res
    PearsonRResult(statistic=-0.828503883588428, pvalue=0.021280260007523286)

    To perform an exact permutation version of the test:

    >>> rng = np.random.default_rng(7796654889291491997)
    >>> method = stats.PermutationMethod(n_resamples=np.inf, random_state=rng)
    >>> stats.pearsonr(x, y, method=method)
    PearsonRResult(statistic=-0.828503883588428, pvalue=0.028174603174603175)

    To perform the test under the null hypothesis that the data were drawn from
    *uniform* distributions:

    >>> method = stats.MonteCarloMethod(rvs=(rng.uniform, rng.uniform))
    >>> stats.pearsonr(x, y, method=method)
    PearsonRResult(statistic=-0.828503883588428, pvalue=0.0188)

    To produce an asymptotic 90% confidence interval:

    >>> res.confidence_interval(confidence_level=0.9)
    ConfidenceInterval(low=-0.9644331982722841, high=-0.3460237473272273)

    And for a bootstrap confidence interval:
    # 创建一个基于BCa方法的BootstrapMethod对象，并指定随机数生成器rng
    >>> method = stats.BootstrapMethod(method='BCa', random_state=rng)
    # 使用指定的置信水平和Bootstrap方法计算置信区间
    >>> res.confidence_interval(confidence_level=0.9, method=method)
    # 返回计算得到的置信区间，其上下限值可能有所变化

    # 如果提供了N维数组，将根据与大多数scipy.stats函数相同的约定进行单次调用中的多个测试：
    >>> rng = np.random.default_rng(2348246935601934321)
    # 生成两个8x15的标准正态分布随机数组x和y
    >>> x = rng.standard_normal((8, 15))
    >>> y = rng.standard_normal((8, 15))
    # 计算沿轴0（列）的Pearson相关系数，并返回统计量的形状
    >>> stats.pearsonr(x, y, axis=0).statistic.shape
    (15,)
    # 计算沿轴1（行）的Pearson相关系数，并返回统计量的形状
    >>> stats.pearsonr(x, y, axis=1).statistic.shape
    (8,)

    # 使用标准的NumPy广播技术执行数组切片的所有配对比较，例如，计算所有行对之间的相关性：
    >>> stats.pearsonr(x[:, np.newaxis, :], y, axis=-1).statistic.shape
    (8, 8)

    # 如果y = a + b*x + e，其中a、b为常数，e为假设独立于x的随机误差项
    # 简化起见，假设x为标准正态分布，a=0，b=1，并且e为均值为零、标准差为s>0的正态分布
    >>> rng = np.random.default_rng()
    >>> s = 0.5
    >>> x = stats.norm.rvs(size=500, random_state=rng)
    >>> e = stats.norm.rvs(scale=s, size=500, random_state=rng)
    >>> y = x + e
    # 计算x和y之间的Pearson相关系数的统计量
    >>> stats.pearsonr(x, y).statistic
    0.9001942438244763

    # 这个值应该接近于以下精确值
    >>> 1/np.sqrt(1 + s**2)
    0.8944271909999159

    # 对于s=0.5，我们观察到很高的相关性。一般而言，噪声方差的增加会降低相关性，
    # 当误差方差趋近于零时，相关性接近于1。

    # 需要记住的是，没有相关性并不意味着独立，除非(x, y)共同正态分布。
    # 即使在非常简单的依赖结构下，相关性也可以为零：
    # 如果X服从标准正态分布，令y = abs(x)。注意x和y之间的相关性为零。
    >>> y = np.abs(x)
    >>> stats.pearsonr(x, y)
    PearsonRResult(statistic=-0.05444919272687482, pvalue=0.22422294836207743)

    # 非零的相关系数可能会误导。例如，如果X服从标准正态分布，定义y为x（如果x < 0），
    # 否则y为0。简单的计算表明corr(x, y) = sqrt(2/Pi) = 0.797...，暗示了高度的相关性：
    >>> y = np.where(x < 0, x, 0)
    >>> stats.pearsonr(x, y)
    PearsonRResult(statistic=0.861985781588, pvalue=4.813432002751103e-149)
    This is unintuitive since there is no dependence of x and y if x is larger
    than zero which happens in about half of the cases if we sample x and y.

    """
    # 将输入的 x 和 y 转换为指定数组命名空间的对象
    xp = array_namespace(x, y)
    # 将 x 和 y 分别转换为 xp 命名空间的数组
    x = xp.asarray(x)
    y = xp.asarray(y)

    # 如果 xp 不是 NumPy 数组，并且指定了 method，则将 method 设置为 'invalid'
    if not is_numpy(xp) and method is not None:
        method = 'invalid'

    # 如果未指定 axis，则将 x 和 y 展平为一维数组
    if axis is None:
        x = xp.reshape(x, (-1,))
        y = xp.reshape(y, (-1,))
        axis = -1

    # 将 axis 转换为整数类型，如果不能转换则抛出 ValueError
    axis_int = int(axis)
    if axis_int != axis:
        raise ValueError('`axis` must be an integer.')
    axis = axis_int

    # 检查 x 和 y 在指定 axis 上的长度是否相等，不相等则抛出 ValueError
    n = x.shape[axis]
    if n != y.shape[axis]:
        raise ValueError('`x` and `y` must have the same length along `axis`.')

    # 检查 x 和 y 在指定 axis 上的长度是否至少为 2，如果不是则抛出 ValueError
    if n < 2:
        raise ValueError('`x` and `y` must have length at least 2.')

    try:
        # 尝试广播 x 和 y，如果不能广播则抛出异常
        x, y = xp.broadcast_arrays(x, y)
    except (ValueError, RuntimeError) as e:
        # 如果广播失败，则抛出详细的 ValueError
        message = '`x` and `y` must be broadcastable.'
        raise ValueError(message) from e

    # 在 array API 严格模式下，`moveaxis` 函数尚未可用，使用 `xp_moveaxis_to_end` 代替
    x = xp_moveaxis_to_end(x, axis, xp=xp)
    y = xp_moveaxis_to_end(y, axis, xp=xp)
    axis = -1

    # 确定 x 和 y 的结果数据类型
    dtype = xp.result_type(x.dtype, y.dtype)
    if xp.isdtype(dtype, "integral"):
        dtype = xp.asarray(1.).dtype

    # 如果数据类型是复数浮点数，则不支持该函数，抛出 ValueError
    if xp.isdtype(dtype, "complex floating"):
        raise ValueError('This function does not support complex data')

    # 将 x 和 y 转换为指定的数据类型，避免复制数据
    x = xp.astype(x, dtype, copy=False)
    y = xp.astype(y, dtype, copy=False)
    # 计算阈值用于判断数据是否为常量
    threshold = xp.finfo(dtype).eps ** 0.75

    # 如果输入数组中有常量，则相关系数不被定义，发出警告
    const_x = xp.all(x == x[..., 0:1], axis=-1)
    const_y = xp.all(y == y[..., 0:1], axis=-1)
    const_xy = const_x | const_y
    if xp.any(const_xy):
        msg = ("An input array is constant; the correlation coefficient "
               "is not defined.")
        warnings.warn(stats.ConstantInputWarning(msg), stacklevel=2)

    # 如果 method 是 PermutationMethod 的实例，则执行置换检验
    if isinstance(method, PermutationMethod):
        # 定义统计函数，计算 Pearson 相关系数和统计量
        def statistic(y, axis):
            statistic, _ = pearsonr(x, y, axis=axis, alternative=alternative)
            return statistic

        # 执行置换检验，返回置换检验的结果
        res = permutation_test((y,), statistic, permutation_type='pairings',
                               axis=axis, alternative=alternative, **method._asdict())

        # 返回 Pearson 相关系数的结果对象
        return PearsonRResult(statistic=res.statistic, pvalue=res.pvalue, n=n,
                              alternative=alternative, x=x, y=y, axis=axis)
    elif isinstance(method, MonteCarloMethod):
        # 如果 `method` 是 `MonteCarloMethod` 类的实例
        def statistic(x, y, axis):
            # 定义统计函数，使用 Pearson 相关系数计算
            statistic, _ = pearsonr(x, y, axis=axis, alternative=alternative)
            return statistic

        if method.rvs is None:
            # 如果 `method` 的 `rvs` 属性为空，则使用默认随机数生成器创建
            rng = np.random.default_rng()
            method.rvs = rng.normal, rng.normal

        # 执行蒙特卡洛方法的假设检验，使用指定的统计函数和参数
        res = monte_carlo_test((x, y,), statistic=statistic, axis=axis,
                               alternative=alternative, **method._asdict())

        # 返回 Pearson 相关系数的结果对象
        return PearsonRResult(statistic=res.statistic, pvalue=res.pvalue, n=n,
                              alternative=alternative, x=x, y=y, axis=axis)
    elif method == 'invalid':
        # 如果 `method` 是字符串 `'invalid'`，抛出值错误，要求 `method` 必须为 `None`
        message = '`method` must be `None` if arguments are not NumPy arrays.'
        raise ValueError(message)
    elif method is not None:
        # 如果 `method` 不是 `None` 也不是有效的假设检验方法，抛出值错误
        message = ('`method` must be an instance of `PermutationMethod`,'
                   '`MonteCarloMethod`, or None.')
        raise ValueError(message)

    if n == 2:
        # 如果样本数 `n` 等于 2，计算简单情况下的 Pearson 相关系数和 p 值
        r = (xp.sign(x[..., 1] - x[..., 0])*xp.sign(y[..., 1] - y[..., 0]))
        r = r[()] if r.ndim == 0 else r
        pvalue = xp.ones_like(r)
        pvalue = pvalue[()] if pvalue.ndim == 0 else pvalue
        # 返回简单情况下的 Pearson 相关系数结果对象
        result = PearsonRResult(statistic=r, pvalue=pvalue, n=n,
                                alternative=alternative, x=x, y=y, axis=axis)
        return result

    xmean = xp.mean(x, axis=axis, keepdims=True)
    ymean = xp.mean(y, axis=axis, keepdims=True)
    xm = x - xmean
    ym = y - ymean

    # 使用 `xp.max` 和 `xp.linalg.vector_norm` 计算归一化因子 `normxm` 和 `normym`
    # 以处理数值计算中的溢出和浮点数误差
    xmax = xp.max(xp.abs(xm), axis=axis, keepdims=True)
    ymax = xp.max(xp.abs(ym), axis=axis, keepdims=True)
    with np.errstate(invalid='ignore'):
        normxm = xmax * xp.linalg.vector_norm(xm/xmax, axis=axis, keepdims=True)
        normym = ymax * xp.linalg.vector_norm(ym/ymax, axis=axis, keepdims=True)

    # 计算 `nconst_x` 和 `nconst_y`，检查是否存在接近常数的输入数组
    nconst_x = xp.any(normxm < threshold*xp.abs(xmean), axis=axis)
    nconst_y = xp.any(normym < threshold*xp.abs(ymean), axis=axis)
    nconst_xy = nconst_x | nconst_y

    if xp.any(nconst_xy & (~const_xy)):
        # 如果输入数组接近常数，则警告用户可能存在精度损失
        msg = ("An input array is nearly constant; the computed "
               "correlation coefficient may be inaccurate.")
        warnings.warn(stats.NearConstantInputWarning(msg), stacklevel=2)

    # 使用 `xp.sum` 计算 Pearson 相关系数 `r`
    with np.errstate(invalid='ignore', divide='ignore'):
        r = xp.sum(xm/normxm * ym/normym, axis=axis)

    # 如果 `r` 的绝对值大于 1，可能是浮点运算的小误差，设置 `one` 为浮点数 `1`
    one = xp.asarray(1, dtype=dtype)
    # 使用 `xp_clip` 函数对数组 `r` 进行裁剪操作，限制在 [-1, 1] 范围内，并将结果保存在 `r` 中
    r = xp.asarray(xp_clip(r, -one, one, xp=xp))
    
    # 将 `r` 中指定索引 `const_xy` 处的元素设置为 NaN，用于符合文档字符串中描述的操作
    r[const_xy] = xp.nan

    # 根据零假设下 `r` 的分布特性，创建一个 Beta 分布对象 `_SimpleBeta`，参数 a = b = n/2 - 1，分布范围为 (-1, 1)
    ab = xp.asarray(n/2 - 1)
    dist = _SimpleBeta(ab, ab, loc=-1, scale=2)
    
    # 根据计算得到的 `r` 和上述 Beta 分布 `dist`，以及备择假设 `alternative`，计算出 `r` 的 p 值
    pvalue = _get_pvalue(r, dist, alternative, xp=xp)

    # 如果 `r` 是零维数组，将其转换为标量；否则保持不变
    r = r[()] if r.ndim == 0 else r
    
    # 如果 `pvalue` 是零维数组，将其转换为标量；否则保持不变
    pvalue = pvalue[()] if pvalue.ndim == 0 else pvalue
    
    # 返回 PearsonRResult 对象，包含计算得到的统计量 `r`、p 值 `pvalue`，以及其他参数和选项
    return PearsonRResult(statistic=r, pvalue=pvalue, n=n,
                          alternative=alternative, x=x, y=y, axis=axis)
# 定义 Fisher 精确检验函数，用于处理一个 2x2 的列联表
def fisher_exact(table, alternative='two-sided'):
    """Perform a Fisher exact test on a 2x2 contingency table.

    The null hypothesis is that the true odds ratio of the populations
    underlying the observations is one, and the observations were sampled
    from these populations under a condition: the marginals of the
    resulting table must equal those of the observed table. The statistic
    returned is the unconditional maximum likelihood estimate of the odds
    ratio, and the p-value is the probability under the null hypothesis of
    obtaining a table at least as extreme as the one that was actually
    observed. There are other possible choices of statistic and two-sided
    p-value definition associated with Fisher's exact test; please see the
    Notes for more information.

    Parameters
    ----------
    table : array_like of ints
        A 2x2 contingency table.  Elements must be non-negative integers.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

        * 'two-sided': the odds ratio of the underlying population is not one
        * 'less': the odds ratio of the underlying population is less than one
        * 'greater': the odds ratio of the underlying population is greater
          than one

        See the Notes for more details.

    Returns
    -------
    res : SignificanceResult
        An object containing attributes:

        statistic : float
            This is the prior odds ratio, not a posterior estimate.
        pvalue : float
            The probability under the null hypothesis of obtaining a
            table at least as extreme as the one that was actually observed.

    See Also
    --------
    chi2_contingency : Chi-square test of independence of variables in a
        contingency table.  This can be used as an alternative to
        `fisher_exact` when the numbers in the table are large.
    contingency.odds_ratio : Compute the odds ratio (sample or conditional
        MLE) for a 2x2 contingency table.
    barnard_exact : Barnard's exact test, which is a more powerful alternative
        than Fisher's exact test for 2x2 contingency tables.
    boschloo_exact : Boschloo's exact test, which is a more powerful
        alternative than Fisher's exact test for 2x2 contingency tables.

    Notes
    -----
    *Null hypothesis and p-values*

    The null hypothesis is that the true odds ratio of the populations
    underlying the observations is one, and the observations were sampled at
    random from these populations under a condition: the marginals of the
    resulting table must equal those of the observed table. Equivalently,
    the null hypothesis is that the input table is from the hypergeometric
    distribution with parameters (as used in `hypergeom`)
    ``M = a + b + c + d``, ``n = a + b`` and ``N = a + c``, where the
    input table is ``[[a, b], [c, d]]``.  This distribution has support
    ``max(0, N + n - M) <= x <= min(N, n)``, or, in terms of the values
    in the input table, ``min(0, a - d) <= x <= a + min(b, c)``.  ``x``
    can be interpreted as the upper-left element of a 2x2 table, so the
    tables in the distribution have form::

        [  x           n - x     ]
        [N - x    M - (n + N) + x]

    For example, if::

        table = [6  2]
                [1  4]

    then the support is ``2 <= x <= 7``, and the tables in the distribution
    are::

        [2 6]   [3 5]   [4 4]   [5 3]   [6 2]  [7 1]
        [5 0]   [4 1]   [3 2]   [2 3]   [1 4]  [0 5]

    The probability of each table is given by the hypergeometric distribution
    ``hypergeom.pmf(x, M, n, N)``.  For this example, these are (rounded to
    three significant digits)::

        x       2      3      4      5       6        7
        p  0.0163  0.163  0.408  0.326  0.0816  0.00466

    These can be computed with::

        >>> import numpy as np
        >>> from scipy.stats import hypergeom
        >>> table = np.array([[6, 2], [1, 4]])
        >>> M = table.sum()  # 计算输入表格所有元素之和
        >>> n = table[0].sum()  # 计算第一行的元素之和
        >>> N = table[:, 0].sum()  # 计算第一列的元素之和
        >>> start, end = hypergeom.support(M, n, N)  # 计算超几何分布的支持范围
        >>> hypergeom.pmf(np.arange(start, end+1), M, n, N)  # 计算超几何分布的概率质量函数
        array([0.01631702, 0.16317016, 0.40792541, 0.32634033, 0.08158508,
               0.004662  ])

    The two-sided p-value is the probability that, under the null hypothesis,
    a random table would have a probability equal to or less than the
    probability of the input table.  For our example, the probability of
    the input table (where ``x = 6``) is 0.0816.  The x values where the
    probability does not exceed this are 2, 6 and 7, so the two-sided p-value
    is ``0.0163 + 0.0816 + 0.00466 ~= 0.10256``::

        >>> from scipy.stats import fisher_exact
        >>> res = fisher_exact(table, alternative='two-sided')
        >>> res.pvalue
        0.10256410256410257

    The one-sided p-value for ``alternative='greater'`` is the probability
    that a random table has ``x >= a``, which in our example is ``x >= 6``,
    or ``0.0816 + 0.00466 ~= 0.08626``::

        >>> res = fisher_exact(table, alternative='greater')
        >>> res.pvalue
        0.08624708624708627

    This is equivalent to computing the survival function of the
    distribution at ``x = 5`` (one less than ``x`` from the input table,
    because we want to include the probability of ``x = 6`` in the sum)::

        >>> hypergeom.sf(5, M, n, N)
        0.08624708624708627

    For ``alternative='less'``, the one-sided p-value is the probability
    that a random table has ``x <= a``, (i.e. ``x <= 6`` in our example),
    or ``0.0163 + 0.163 + 0.408 + 0.326 + 0.0816 ~= 0.9949``::

        >>> res = fisher_exact(table, alternative='less')
        >>> res.pvalue
        0.9953379953379957

    This is equivalent to computing the cumulative distribution function
    of the distribution at ``x = 6``:

        >>> hypergeom.cdf(6, M, n, N)
        0.9953379953379957

    *Odds ratio*

    The calculated odds ratio is different from the value computed by the
    R function ``fisher.test``.  This implementation returns the "sample"
    or "unconditional" maximum likelihood estimate, while ``fisher.test``
    in R uses the conditional maximum likelihood estimate.  To compute the
    conditional maximum likelihood estimate of the odds ratio, use
    `scipy.stats.contingency.odds_ratio`.

    References
    ----------
    .. [1] Fisher, Sir Ronald A, "The Design of Experiments:
           Mathematics of a Lady Tasting Tea." ISBN 978-0-486-41151-4, 1935.
    .. [2] "Fisher's exact test",
           https://en.wikipedia.org/wiki/Fisher's_exact_test
    .. [3] Emma V. Low et al. "Identifying the lowest effective dose of
           acetazolamide for the prophylaxis of acute mountain sickness:
           systematic review and meta-analysis."
           BMJ, 345, :doi:`10.1136/bmj.e6779`, 2012.

    Examples
    --------
    In [3]_, the effective dose of acetazolamide for the prophylaxis of acute
    mountain sickness was investigated. The study notably concluded:

        Acetazolamide 250 mg, 500 mg, and 750 mg daily were all efficacious for
        preventing acute mountain sickness. Acetazolamide 250 mg was the lowest
        effective dose with available evidence for this indication.

    The following table summarizes the results of the experiment in which
    some participants took a daily dose of acetazolamide 250 mg while others
    took a placebo.
    Cases of acute mountain sickness were recorded::

                                    Acetazolamide   Control/Placebo
        Acute mountain sickness            7           17
        No                                15            5


    Is there evidence that the acetazolamide 250 mg reduces the risk of
    acute mountain sickness?
    We begin by formulating a null hypothesis :math:`H_0`:

        The odds of experiencing acute mountain sickness are the same with
        the acetazolamide treatment as they are with placebo.

    Let's assess the plausibility of this hypothesis with
    Fisher's test.

    >>> from scipy.stats import fisher_exact
    载入 `fisher_exact` 函数从 `scipy.stats`
    >>> res = fisher_exact([[7, 17], [15, 5]], alternative='less')
    进行 Fisher 精确检验，传入 2x2 的列联表和选择 `less` 作为备择假设
    >>> res.statistic
    计算并输出 Fisher 检验的统计量
    0.13725490196078433
    >>> res.pvalue
    计算并输出 Fisher 检验的 p 值

    Using a significance level of 5%, we would reject the null hypothesis in
    favor of the alternative hypothesis: "The odds of experiencing acute
    mountain sickness with acetazolamide treatment are less than the odds of
    experiencing acute mountain sickness with placebo."
    hypergeom = distributions.hypergeom
    # 导入超几何分布函数并赋值给变量hypergeom

    # int32 is not enough for the algorithm
    c = np.asarray(table, dtype=np.int64)
    # 将输入的table转换为numpy数组，并指定数据类型为int64，以确保算法正常运行

    if not c.shape == (2, 2):
        raise ValueError("The input `table` must be of shape (2, 2).")
    # 检查数组c的形状是否为(2, 2)，如果不是则抛出数值错误异常

    if np.any(c < 0):
        raise ValueError("All values in `table` must be nonnegative.")
    # 检查数组c中是否有负值，如果有则抛出数值错误异常，要求所有值必须为非负数

    if 0 in c.sum(axis=0) or 0 in c.sum(axis=1):
        # 如果任何行或列的和为0，则返回一个SignificanceResult对象，其中p值为1，odds ratio为NaN
        return SignificanceResult(np.nan, 1.0)

    if c[1, 0] > 0 and c[0, 1] > 0:
        oddsratio = c[0, 0] * c[1, 1] / (c[1, 0] * c[0, 1])
    else:
        oddsratio = np.inf
    # 计算odds ratio，如果条件不满足则设为无穷大

    n1 = c[0, 0] + c[0, 1]
    n2 = c[1, 0] + c[1, 1]
    n = c[0, 0] + c[1, 0]
    # 计算相关的n1、n2和n值

    def pmf(x):
        return hypergeom.pmf(x, n1 + n2, n1, n)
    # 定义一个概率质量函数pmf，调用超几何分布的pmf方法

    if alternative == 'less':
        pvalue = hypergeom.cdf(c[0, 0], n1 + n2, n1, n)
    elif alternative == 'greater':
        # 和'less'情况下的公式相同，但是使用第二列的数据
        pvalue = hypergeom.cdf(c[0, 1], n1 + n2, n1, c[0, 1] + c[1, 1])
    # 根据alternative参数的不同，计算p值
    # 如果 alternative 为 'two-sided'，计算 mode 值
    mode = int((n + 1) * (n1 + 1) / (n1 + n2 + 2))
    
    # 计算精确超几何分布的概率质量函数值
    pexact = hypergeom.pmf(c[0, 0], n1 + n2, n1, n)
    
    # 计算 mode 处的超几何分布的概率质量函数值
    pmode = hypergeom.pmf(mode, n1 + n2, n1, n)
    
    # 设定一个极小的值 epsilon
    epsilon = 1e-14
    
    # 设定 gamma 为 1 + epsilon
    gamma = 1 + epsilon
    
    # 检查精确概率与 mode 处概率的相对误差是否在极小值 epsilon 内
    if np.abs(pexact - pmode) / np.maximum(pexact, pmode) <= epsilon:
        # 如果满足条件，返回结果为一个显著性结果对象，概率为 1
        return SignificanceResult(oddsratio, 1.)
    
    # 如果 c[0, 0] 小于 mode 值
    elif c[0, 0] < mode:
        # 计算小于等于 c[0, 0] 的累积分布函数值
        plower = hypergeom.cdf(c[0, 0], n1 + n2, n1, n)
        
        # 如果超几何分布在全体样本上的概率质量函数值大于 pexact * gamma
        if hypergeom.pmf(n, n1 + n2, n1, n) > pexact * gamma:
            # 返回一个显著性结果对象，概率为 plower
            return SignificanceResult(oddsratio, plower)
        
        # 使用二分搜索函数 _binary_search 寻找合适的猜测值
        guess = _binary_search(lambda x: -pmf(x), -pexact * gamma, mode, n)
        
        # 计算 pvalue 值
        pvalue = plower + hypergeom.sf(guess, n1 + n2, n1, n)
    
    # 如果 c[0, 0] 大于等于 mode 值
    else:
        # 计算大于 c[0, 0] - 1 的生存函数值
        pupper = hypergeom.sf(c[0, 0] - 1, n1 + n2, n1, n)
        
        # 如果超几何分布在样本中无事件的概率质量函数值大于 pexact * gamma
        if hypergeom.pmf(0, n1 + n2, n1, n) > pexact * gamma:
            # 返回一个显著性结果对象，概率为 pupper
            return SignificanceResult(oddsratio, pupper)
        
        # 使用二分搜索函数 _binary_search 寻找合适的猜测值
        guess = _binary_search(pmf, pexact * gamma, 0, mode)
        
        # 计算 pvalue 值
        pvalue = pupper + hypergeom.cdf(guess, n1 + n2, n1, n)
    
    # 如果 alternative 不是 {'two-sided', 'less', 'greater'} 中的一个，抛出异常
    else:
        msg = "`alternative` should be one of {'two-sided', 'less', 'greater'}"
        raise ValueError(msg)
    
    # 确保 pvalue 不超过 1.0
    pvalue = min(pvalue, 1.0)
    
    # 返回一个显著性结果对象，包含 oddsratio 和计算出的 pvalue
    return SignificanceResult(oddsratio, pvalue)
# 计算 Spearman 相关系数及其相关的 p 值
def spearmanr(a, b=None, axis=0, nan_policy='propagate',
              alternative='two-sided'):
    r"""Calculate a Spearman correlation coefficient with associated p-value.

    The Spearman rank-order correlation coefficient is a nonparametric measure
    of the monotonicity of the relationship between two datasets.
    Like other correlation coefficients,
    this one varies between -1 and +1 with 0 implying no correlation.
    Correlations of -1 or +1 imply an exact monotonic relationship. Positive
    correlations imply that as x increases, so does y. Negative correlations
    imply that as x increases, y decreases.

    The p-value roughly indicates the probability of an uncorrelated system
    producing datasets that have a Spearman correlation at least as extreme
    as the one computed from these datasets. Although calculation of the
    p-value does not make strong assumptions about the distributions underlying
    the samples, it is only accurate for very large samples (>500
    observations). For smaller sample sizes, consider a permutation test (see
    Examples section below).

    Parameters
    ----------
    a, b : 1D or 2D array_like, b is optional
        One or two 1-D or 2-D arrays containing multiple variables and
        observations. When these are 1-D, each represents a vector of
        observations of a single variable. For the behavior in the 2-D case,
        see under ``axis``, below.
        Both arrays need to have the same length in the ``axis`` dimension.
    axis : int or None, optional
        If axis=0 (default), then each column represents a variable, with
        observations in the rows. If axis=1, the relationship is transposed:
        each row represents a variable, while the columns contain observations.
        If axis=None, then both arrays will be raveled.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

        * 'propagate': returns nan
        * 'raise': throws an error
        * 'omit': performs the calculations ignoring nan values

    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        The following options are available:

        * 'two-sided': the correlation is nonzero
        * 'less': the correlation is negative (less than zero)
        * 'greater':  the correlation is positive (greater than zero)

        .. versionadded:: 1.7.0

    Returns
    -------
    res : SignificanceResult
        # 定义变量 res，类型为 SignificanceResult，包含以下属性：
        
        statistic : float or ndarray (2-D square)
            # 属性 statistic：float 或者二维方阵 ndarray
            # Spearman 相关系数矩阵或相关系数（如果参数只包含两个变量）。相关系数矩阵是一个方阵，其长度等于参数 a 和 b 中所有变量（列或行）的总数。
        
        pvalue : float
            # 属性 pvalue：float
            # 假设检验的 p 值，其零假设是两个样本没有序数相关性。参见上面的 `alternative` 参数以获取备择假设。`pvalue` 的形状与 `statistic` 相同。
    
    Warns
    -----
    `~scipy.stats.ConstantInputWarning`
        # 引发警告 `~scipy.stats.ConstantInputWarning`
        # 如果输入是一个常数数组。在这种情况下相关系数未定义，因此返回 `np.nan`。
    
    References
    ----------
    .. [1] Zwillinger, D. and Kokoska, S. (2000). CRC Standard
       Probability and Statistics Tables and Formulae. Chapman & Hall: New
       York. 2000.
       Section  14.7
    .. [2] Kendall, M. G. and Stuart, A. (1973).
       The Advanced Theory of Statistics, Volume 2: Inference and Relationship.
       Griffin. 1973.
       Section 31.18
    .. [3] Kershenobich, D., Fierro, F. J., & Rojkind, M. (1970). The
       relationship between the free pool of proline and collagen content in
       human liver cirrhosis. The Journal of Clinical Investigation, 49(12),
       2246-2249.
    .. [4] Hollander, M., Wolfe, D. A., & Chicken, E. (2013). Nonparametric
       statistical methods. John Wiley & Sons.
    .. [5] B. Phipson and G. K. Smyth. "Permutation P-values Should Never Be
       Zero: Calculating Exact P-values When Permutations Are Randomly Drawn."
       Statistical Applications in Genetics and Molecular Biology 9.1 (2010).
    .. [6] Ludbrook, J., & Dudley, H. (1998). Why permutation tests are
       superior to t and F tests in biomedical research. The American
       Statistician, 52(2), 127-132.

    Examples
    --------
    Consider the following data from [3]_, which studied the relationship
    between free proline (an amino acid) and total collagen (a protein often
    found in connective tissue) in unhealthy human livers.

    The ``x`` and ``y`` arrays below record measurements of the two compounds.
    The observations are paired: each free proline measurement was taken from
    the same liver as the total collagen measurement at the same index.

    >>> import numpy as np
    >>> # total collagen (mg/g dry weight of liver)
    >>> x = np.array([7.1, 7.1, 7.2, 8.3, 9.4, 10.5, 11.4])
    >>> # free proline (μ mole/g dry weight of liver)
    >>> y = np.array([2.8, 2.9, 2.8, 2.6, 3.5, 4.6, 5.0])

    These data were analyzed in [4]_ using Spearman's correlation coefficient,
    a statistic sensitive to monotonic correlation between the samples.

    >>> from scipy import stats
    >>> res = stats.spearmanr(x, y)
    >>> res.statistic
    # 统计量，用于衡量样本之间的序数相关性，其值趋向于1表示强正相关，趋向于-1表示强负相关，接近于0表示弱相关。
    # 在这个测试中，通过将观察到的统计量与空假设下的分布进行比较，空假设假定总胶原和游离脯氨酸测量是独立的。
    
    >>> import matplotlib.pyplot as plt
    >>> dof = len(x)-2  # 自由度，与样本数量相关
    >>> dist = stats.t(df=dof)  # 创建 t 分布对象，自由度为 dof
    >>> t_vals = np.linspace(-5, 5, 100)  # 在 -5 到 5 之间生成 100 个等间距的值
    >>> pdf = dist.pdf(t_vals)  # 计算 t 值对应的概率密度函数值
    >>> fig, ax = plt.subplots(figsize=(8, 5))  # 创建图形和坐标轴对象
    
    >>> def plot(ax):  # 定义绘图函数，用于重复使用
    ...     ax.plot(t_vals, pdf)  # 绘制 t 值与概率密度函数的曲线
    ...     ax.set_title("Spearman's Rho Test Null Distribution")  # 设置图表标题
    ...     ax.set_xlabel("statistic")  # 设置 x 轴标签
    ...     ax.set_ylabel("probability density")  # 设置 y 轴标签
    >>> plot(ax)  # 绘制图表
    >>> plt.show()  # 显示图表
    
    # 通过 p 值进行比较，p 值表示在空假设分布中，有多少比例的值比观察到的统计量更极端。
    # 在双侧检验中，如果统计量为正，空假设分布中大于变换后的统计量以及小于观察统计量的负值都被认为是“更极端”的情况。
    >>> fig, ax = plt.subplots(figsize=(8, 5))  # 创建图形和坐标轴对象
    >>> plot(ax)  # 绘制图表
    >>> rs = res.statistic  # 原始统计量
    >>> transformed = rs * np.sqrt(dof / ((rs+1.0)*(1.0-rs)))  # 对统计量进行变换
    >>> pvalue = dist.cdf(-transformed) + dist.sf(transformed)  # 计算 p 值
    >>> annotation = (f'p-value={pvalue:.4f}\n(shaded area)')  # 注释文本内容，显示 p 值和阴影区域说明
    >>> props = dict(facecolor='black', width=1, headwidth=5, headlength=8)  # 定义注释框样式
    >>> _ = ax.annotate(annotation, (2.7, 0.025), (3, 0.03), arrowprops=props)  # 添加注释到图表中
    >>> i = t_vals >= transformed  # 找出大于或等于变换后统计量的 t 值索引
    >>> ax.fill_between(t_vals[i], y1=0, y2=pdf[i], color='C0')  # 填充 t 值区域
    >>> i = t_vals <= -transformed  # 找出小于或等于变换后统计量的负值的 t 值索引
    >>> ax.fill_between(t_vals[i], y1=0, y2=pdf[i], color='C0')  # 填充 t 值区域
    >>> ax.set_xlim(-5, 5)  # 设置 x 轴范围
    >>> ax.set_ylim(0, 0.1)  # 设置 y 轴范围
    >>> plt.show()  # 显示图表
    >>> res.pvalue  # 双侧检验的 p 值
    0.07991669030889909
    
    # 如果 p 值很小，即从独立分布中抽样得到这样一个极端统计量值的概率很低，
    # 这可能被认为是对空假设的反证据，支持备择假设：总胶原和游离脯氨酸的分布 *不* 独立。
    # 注意：
    # - 反之并不成立；也就是说，此测试不能用来支持空假设。
    # 设置判断“小值”的阈值，应在数据分析之前根据研究需要做出选择，考虑到误差的风险，
    # 包括误报（错误地拒绝零假设）和漏报（未能拒绝虚假的零假设）。
    - The threshold for values that will be considered "small" is a choice that
      should be made before the data is analyzed [5]_ with consideration of the
      risks of both false positives (incorrectly rejecting the null hypothesis)
      and false negatives (failure to reject a false null hypothesis).

    # 小的 p 值并不意味着存在“大”效应；它们只能表明存在一个“显著”的效应，
    # 意味着在零假设下这种情况发生的概率很小。
    - Small p-values are not evidence for a *large* effect; rather, they can
      only provide evidence for a "significant" effect, meaning that they are
      unlikely to have occurred under the null hypothesis.

    # 假设在执行实验之前，作者有理由预测总胶原和游离脯氨酸测量之间存在正相关，
    # 并选择根据单边替代来评估零假设的合理性：游离脯氨酸与总胶原有正序相关。
    # 在这种情况下，只有在空分布中与观察统计量一样大或更大的值才被视为更极端的情况。
    >>> res = stats.spearmanr(x, y, alternative='greater')
    >>> res.statistic
    0.7000000000000001  # 相同的统计量
    >>> fig, ax = plt.subplots(figsize=(8, 5))
    >>> plot(ax)
    >>> pvalue = dist.sf(transformed)
    >>> annotation = (f'p-value={pvalue:.6f}\n(shaded area)')
    >>> props = dict(facecolor='black', width=1, headwidth=5, headlength=8)
    >>> _ = ax.annotate(annotation, (3, 0.018), (3.5, 0.03), arrowprops=props)
    >>> i = t_vals >= transformed
    >>> ax.fill_between(t_vals[i], y1=0, y2=pdf[i], color='C0')
    >>> ax.set_xlim(1, 5)
    >>> ax.set_ylim(0, 0.1)
    >>> plt.show()
    >>> res.pvalue
    0.03995834515444954  # 单边 p 值；双边 p 值的一半

    # 注意， t 分布提供零分布的渐近近似；它只对具有许多观测样本的样本准确。
    # 对于小样本，执行排列检验可能更合适：在总胶原和游离脯氨酸独立的零假设下，
    # 每个游离脯氨酸测量与任何总胶原测量同等可能观察到。因此，我们可以通过
    # 在 `x` 和 `y` 之间的每种可能的元素配对下计算统计量来形成*精确*的零分布。
    >>> def statistic(x):  # 探索通过对 `x` 进行排列来形成所有可能的配对
    ...     rs = stats.spearmanr(x, y).statistic  # 忽略 p 值
    ...     transformed = rs * np.sqrt(dof / ((rs+1.0)*(1.0-rs)))
    ...     return transformed
    >>> ref = stats.permutation_test((x,), statistic, alternative='greater',
    ...                              permutation_type='pairings')
    >>> fig, ax = plt.subplots(figsize=(8, 5))
    >>> plot(ax)
    >>> ax.hist(ref.null_distribution, np.linspace(-5, 5, 26),
    ...         density=True)
    >>> ax.legend(['渐近近似\n(大量观测)',  # 图例说明
    """
    如果指定了轴并且轴大于1，则引发值错误
    spearmanr 只处理 1-D 或 2-D 数组，提供的轴参数为 {axis}，请仅使用 0、1 或 None
    """
    # 检查并确保输入数组 `a` 是一个数组，处理轴参数并返回调整后的数组及轴方向
    a, axisout = _chk_asarray(a, axis)
    if a.ndim > 2:
        # 如果数组 `a` 的维度大于2，则引发值错误
        raise ValueError("spearmanr 只处理 1-D 或 2-D 数组")

    if b is None:
        if a.ndim < 2:
            # 如果数组 `a` 的维度小于2，则引发值错误
            raise ValueError("`spearmanr` 需要至少两个变量来比较")
    else:
        # 确保输入的 `b` 是一个数组，处理轴参数并返回调整后的数组及轴方向
        b, _ = _chk_asarray(b, axis)
        if axisout == 0:
            # 如果 `a` 的轴方向是0，则沿列堆叠 `a` 和 `b`
            a = np.column_stack((a, b))
        else:
            # 否则，沿行堆叠 `a` 和 `b`
            a = np.vstack((a, b))

    # 获取数组 `a` 的变量数和观察数
    n_vars = a.shape[1 - axisout]
    n_obs = a.shape[axisout]
    if n_obs <= 1:
        # 处理空数组或单个观察值的情况，返回一个结果对象，相关系数设置为 NaN
        res = SignificanceResult(np.nan, np.nan)
        res.correlation = np.nan
        return res

    # 如果输入数组中存在常量值，则给出警告并返回相关系数为 NaN 的结果对象
    warn_msg = ("An input array is constant; the correlation coefficient "
                "is not defined.")
    if axisout == 0:
        if (a[:, 0][0] == a[:, 0]).all() or (a[:, 1][0] == a[:, 1]).all():
            warnings.warn(stats.ConstantInputWarning(warn_msg), stacklevel=2)
            res = SignificanceResult(np.nan, np.nan)
            res.correlation = np.nan
            return res
    else:
        if (a[0, :][0] == a[0, :]).all() or (a[1, :][0] == a[1, :]).all():
            warnings.warn(stats.ConstantInputWarning(warn_msg), stacklevel=2)
            res = SignificanceResult(np.nan, np.nan)
            res.correlation = np.nan
            return res

    # 检查输入数组 `a` 是否包含 NaN 值，并根据 `nan_policy` 处理
    a_contains_nan, nan_policy = _contains_nan(a, nan_policy)
    variable_has_nan = np.zeros(n_vars, dtype=bool)
    if a_contains_nan:
        if nan_policy == 'omit':
            # 如果 `nan_policy` 设置为 'omit'，则调用相关的 Spearman 相关系数计算函数
            return mstats_basic.spearmanr(a, axis=axis, nan_policy=nan_policy,
                                          alternative=alternative)
        elif nan_policy == 'propagate':
            if a.ndim == 1 or n_vars <= 2:
                # 如果数组 `a` 是一维或变量数小于等于2，则返回相关系数为 NaN 的结果对象
                res = SignificanceResult(np.nan, np.nan)
                res.correlation = np.nan
                return res
            else:
                # 跟踪包含 NaN 值的变量，并将相关系数设置为 NaN
                variable_has_nan = np.isnan(a).any(axis=axisout)

    # 对数组 `a` 应用秩操作，计算秩相关系数
    a_ranked = np.apply_along_axis(rankdata, axisout, a)
    rs = np.corrcoef(a_ranked, rowvar=axisout)
    dof = n_obs - 2  # 自由度
    # 当 rs 可能包含元素等于 1 时，为避免零除警告，设置错误状态以忽略除法操作中的零除警告
    with np.errstate(divide='ignore'):
        # 修剪由于舍入误差可能导致的小负值，然后再进行平方根运算
        t = rs * np.sqrt((dof/((rs+1.0)*(1.0-rs))).clip(0))

    # 使用自由度 dof 构造简单的学生 t 分布对象
    dist = _SimpleStudentT(dof)
    # 调用 _get_pvalue 函数计算 t 值对应的 p 值，根据 alternative 参数选择检验方式，使用 np 的实现
    prob = _get_pvalue(t, dist, alternative, xp=np)

    # 为了向后兼容性，在比较两列时返回标量结果
    if rs.shape == (2, 2):
        # 当 rs 的形状为 (2, 2) 时，创建一个 SignificanceResult 对象，包含特定的相关系数和 p 值
        res = SignificanceResult(rs[1, 0], prob[1, 0])
        # 设置 SignificanceResult 对象的 correlation 属性为相关系数 rs[1, 0]
        res.correlation = rs[1, 0]
        return res
    else:
        # 将 rs 中包含 NaN 值的行和列设为 NaN
        rs[variable_has_nan, :] = np.nan
        rs[:, variable_has_nan] = np.nan
        # 创建一个 SignificanceResult 对象，包含整体的相关系数和 p 值
        res = SignificanceResult(rs[()], prob[()])
        # 设置 SignificanceResult 对象的 correlation 属性为整体的相关系数 rs
        res.correlation = rs
        return res
# 定义函数 `pointbiserialr`，用于计算点二列相关系数及其 p 值
def pointbiserialr(x, y):
    # 文档字符串，解释了点二列相关系数的计算方法及其用途
    r"""Calculate a point biserial correlation coefficient and its p-value.

    The point biserial correlation is used to measure the relationship
    between a binary variable, x, and a continuous variable, y. Like other
    correlation coefficients, this one varies between -1 and +1 with 0
    implying no correlation. Correlations of -1 or +1 imply a determinative
    relationship.

    This function may be computed using a shortcut formula but produces the
    same result as `pearsonr`.

    Parameters
    ----------
    x : array_like of bools
        Input array.
    y : array_like
        Input array.

    Returns
    -------
    res: SignificanceResult
        An object containing attributes:

        statistic : float
            The R value.
        pvalue : float
            The two-sided p-value.

    Notes
    -----
    `pointbiserialr` uses a t-test with ``n-1`` degrees of freedom.
    It is equivalent to `pearsonr`.

    The value of the point-biserial correlation can be calculated from:

    .. math::

        r_{pb} = \frac{\overline{Y_1} - \overline{Y_0}}
                      {s_y}
                 \sqrt{\frac{N_0 N_1}
                            {N (N - 1)}}

    Where :math:`\overline{Y_{0}}` and :math:`\overline{Y_{1}}` are means
    of the metric observations coded 0 and 1 respectively; :math:`N_{0}` and
    :math:`N_{1}` are number of observations coded 0 and 1 respectively;
    :math:`N` is the total number of observations and :math:`s_{y}` is the
    standard deviation of all the metric observations.

    A value of :math:`r_{pb}` that is significantly different from zero is
    completely equivalent to a significant difference in means between the two
    groups. Thus, an independent groups t Test with :math:`N-2` degrees of
    freedom may be used to test whether :math:`r_{pb}` is nonzero. The
    relation between the t-statistic for comparing two independent groups and
    :math:`r_{pb}` is given by:

    .. math::

        t = \sqrt{N - 2}\frac{r_{pb}}{\sqrt{1 - r^{2}_{pb}}}

    References
    ----------
    .. [1] J. Lev, "The Point Biserial Coefficient of Correlation", Ann. Math.
           Statist., Vol. 20, no.1, pp. 125-126, 1949.

    .. [2] R.F. Tate, "Correlation Between a Discrete and a Continuous
           Variable. Point-Biserial Correlation.", Ann. Math. Statist., Vol. 25,
           np. 3, pp. 603-607, 1954.

    .. [3] D. Kornbrot "Point Biserial Correlation", In Wiley StatsRef:
           Statistics Reference Online (eds N. Balakrishnan, et al.), 2014.
           :doi:`10.1002/9781118445112.stat06227`

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> a = np.array([0, 0, 0, 1, 1, 1, 1])
    >>> b = np.arange(7)
    >>> stats.pointbiserialr(a, b)
    (0.8660254037844386, 0.011724811003954652)
    >>> stats.pearsonr(a, b)
    (0.86602540378443871, 0.011724811003954626)
    >>> np.corrcoef(a, b)

    """

    # 实现点二列相关系数的计算
    r_pb = np.corrcoef(x, y)[0, 1] * np.sqrt((np.count_nonzero(x) * np.count_nonzero(~x)) / (len(x) ** 2))
    # 计算 t 统计量
    t = np.sqrt(len(x) - 2) * r_pb / np.sqrt(1 - r_pb ** 2)
    # 计算 p 值
    p = 2 * stats.t.sf(np.abs(t), len(x) - 1)
    # 返回点二列相关系数及其 p 值
    return r_pb, p
    array([[ 1.       ,  0.8660254],
           [ 0.8660254,  1.       ]])

    """
    计算 Pearson 相关系数和其对应的 p 值
    rpb, prob = pearsonr(x, y)
    # 创建一个结果对象，为了向后兼容，给相关性结果设置了一个别名
    res = SignificanceResult(rpb, prob)
    # 将计算得到的相关系数赋值给结果对象的 correlation 属性
    res.correlation = rpb
    # 返回结果对象
    return res
def kendalltau(x, y, *, nan_policy='propagate',
               method='auto', variant='b', alternative='two-sided'):
    r"""Calculate Kendall's tau, a correlation measure for ordinal data.

    Kendall's tau is a measure of the correspondence between two rankings.
    Values close to 1 indicate strong agreement, and values close to -1
    indicate strong disagreement. This implements two variants of Kendall's
    tau: tau-b (the default) and tau-c (also known as Stuart's tau-c). These
    differ only in how they are normalized to lie within the range -1 to 1;
    the hypothesis tests (their p-values) are identical. Kendall's original
    tau-a is not implemented separately because both tau-b and tau-c reduce
    to tau-a in the absence of ties.

    Parameters
    ----------
    x, y : array_like
        Arrays of rankings, of the same shape. If arrays are not 1-D, they
        will be flattened to 1-D.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

          * 'propagate': returns nan
          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values

    method : {'auto', 'asymptotic', 'exact'}, optional
        Defines which method is used to calculate the p-value [5]_.
        The following options are available (default is 'auto'):

          * 'auto': selects the appropriate method based on a trade-off
            between speed and accuracy
          * 'asymptotic': uses a normal approximation valid for large samples
          * 'exact': computes the exact p-value, but can only be used if no ties
            are present. As the sample size increases, the 'exact' computation
            time may grow and the result may lose some precision.
    variant : {'b', 'c'}, optional
        Defines which variant of Kendall's tau is returned. Default is 'b'.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        The following options are available:

        * 'two-sided': the rank correlation is nonzero
        * 'less': the rank correlation is negative (less than zero)
        * 'greater':  the rank correlation is positive (greater than zero)

    Returns
    -------
    res : SignificanceResult
        An object containing attributes:

        statistic : float
           The tau statistic.
        pvalue : float
           The p-value for a hypothesis test whose null hypothesis is
           an absence of association, tau = 0.

    See Also
    --------
    spearmanr : Calculates a Spearman rank-order correlation coefficient.
    theilslopes : Computes the Theil-Sen estimator for a set of points (x, y).
    weightedtau : Computes a weighted version of Kendall's tau.

    Notes
    -----
    """
    # Implementing Kendall's tau correlation coefficient for ordinal data

    # Flatten x and y arrays if they are not already 1-D
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    # Handling NaN values based on nan_policy
    if nan_policy == 'propagate':
        # Return NaN if any input contains NaN
        if np.isnan(x).any() or np.isnan(y).any():
            return SignificanceResult(np.nan, np.nan)
    elif nan_policy == 'raise':
        # Raise an error if NaN values are present
        if np.isnan(x).any() or np.isnan(y).any():
            raise ValueError("Input contains NaN values")
    elif nan_policy == 'omit':
        # Remove NaN values from x and y before computing Kendall's tau
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]

    # Calculate Kendall's tau-b or tau-c based on variant
    if variant == 'b':
        # Calculate tau-b
        tau, p_value = stats.kendalltau(x, y, method=method)
    elif variant == 'c':
        # Calculate tau-c (Stuart's tau-c)
        tau, p_value = stats.kendalltau(x, y, method=method, variant='c')

    # Create a SignificanceResult object to store the result
    res = SignificanceResult(statistic=tau, pvalue=p_value)

    # Return the calculated result
    return res
    The definition of Kendall's tau that is used is [2]_::
    
      tau_b = (P - Q) / sqrt((P + Q + T) * (P + Q + U))
    
      tau_c = 2 (P - Q) / (n**2 * (m - 1) / m)
    
    where P is the number of concordant pairs, Q the number of discordant
    pairs, T the number of ties only in `x`, and U the number of ties only in
    `y`.  If a tie occurs for the same pair in both `x` and `y`, it is not
    added to either T or U. n is the total number of samples, and m is the
    number of unique values in either `x` or `y`, whichever is smaller.
    
    References
    ----------
    .. [1] Maurice G. Kendall, "A New Measure of Rank Correlation", Biometrika
           Vol. 30, No. 1/2, pp. 81-93, 1938.
    .. [2] Maurice G. Kendall, "The treatment of ties in ranking problems",
           Biometrika Vol. 33, No. 3, pp. 239-251. 1945.
    .. [3] Gottfried E. Noether, "Elements of Nonparametric Statistics", John
           Wiley & Sons, 1967.
    .. [4] Peter M. Fenwick, "A new data structure for cumulative frequency
           tables", Software: Practice and Experience, Vol. 24, No. 3,
           pp. 327-336, 1994.
    .. [5] Maurice G. Kendall, "Rank Correlation Methods" (4th Edition),
           Charles Griffin & Co., 1970.
    .. [6] Kershenobich, D., Fierro, F. J., & Rojkind, M. (1970). The
           relationship between the free pool of proline and collagen content
           in human liver cirrhosis. The Journal of Clinical Investigation,
           49(12), 2246-2249.
    .. [7] Hollander, M., Wolfe, D. A., & Chicken, E. (2013). Nonparametric
           statistical methods. John Wiley & Sons.
    .. [8] B. Phipson and G. K. Smyth. "Permutation P-values Should Never Be
           Zero: Calculating Exact P-values When Permutations Are Randomly
           Drawn." Statistical Applications in Genetics and Molecular Biology
           9.1 (2010).
    
    Examples
    --------
    Consider the following data from [6]_, which studied the relationship
    between free proline (an amino acid) and total collagen (a protein often
    found in connective tissue) in unhealthy human livers.
    
    The ``x`` and ``y`` arrays below record measurements of the two compounds.
    The observations are paired: each free proline measurement was taken from
    the same liver as the total collagen measurement at the same index.
    
    >>> import numpy as np
    >>> # total collagen (mg/g dry weight of liver)
    >>> x = np.array([7.1, 7.1, 7.2, 8.3, 9.4, 10.5, 11.4])
    >>> # free proline (μ mole/g dry weight of liver)
    >>> y = np.array([2.8, 2.9, 2.8, 2.6, 3.5, 4.6, 5.0])
    
    These data were analyzed in [7]_ using Spearman's correlation coefficient,
    a statistic similar to Kendall's tau in that it is also sensitive to
    ordinal correlation between the samples. Let's perform an analogous study
    using Kendall's tau.
    
    >>> from scipy import stats
    >>> res = stats.kendalltau(x, y)
    >>> res.statistic
    0.5499999999999999
    The value of this statistic tends to be high (close to 1) for samples with
    a strongly positive ordinal correlation, low (close to -1) for samples with
    a strongly negative ordinal correlation, and small in magnitude (close to
    zero) for samples with weak ordinal correlation.



    The test is performed by comparing the observed value of the
    statistic against the null distribution: the distribution of statistic
    values derived under the null hypothesis that total collagen and free
    proline measurements are independent.



    For this test, the null distribution for large samples without ties is
    approximated as the normal distribution with variance
    ``(2*(2*n + 5))/(9*n*(n - 1))``, where ``n = len(x)``.



    >>> import matplotlib.pyplot as plt
    >>> n = len(x)  # len(x) == len(y)
    初始化 matplotlib 库并计算样本数 n（假设 x 和 y 长度相同）



    >>> var = (2*(2*n + 5))/(9*n*(n - 1))
    计算 null distribution 的方差 var



    >>> dist = stats.norm(scale=np.sqrt(var))
    创建一个以 var 为标准差的正态分布对象 dist



    >>> z_vals = np.linspace(-1.25, 1.25, 100)
    创建一个包含 100 个均匀分布的 z 值数组



    >>> pdf = dist.pdf(z_vals)
    计算 dist 对应 z 值的概率密度函数值数组



    >>> fig, ax = plt.subplots(figsize=(8, 5))
    创建一个 8x5 大小的图形对象 fig 和坐标轴对象 ax



    >>> def plot(ax):  # we'll reuse this
    ...     ax.plot(z_vals, pdf)
    ...     ax.set_title("Kendall Tau Test Null Distribution")
    ...     ax.set_xlabel("statistic")
    ...     ax.set_ylabel("probability density")
    定义一个函数 plot，用于绘制图形和设置标题、坐标轴标签



    >>> plot(ax)
    调用 plot 函数，将图形绘制在 ax 坐标轴上



    >>> plt.show()
    显示绘制好的图形



    The comparison is quantified by the p-value: the proportion of values in
    the null distribution as extreme or more extreme than the observed
    value of the statistic. In a two-sided test in which the statistic is
    positive, elements of the null distribution greater than the transformed
    statistic and elements of the null distribution less than the negative of
    the observed statistic are both considered "more extreme".



    >>> fig, ax = plt.subplots(figsize=(8, 5))
    创建一个 8x5 大小的新图形对象 fig 和坐标轴对象 ax



    >>> plot(ax)
    调用 plot 函数，绘制另一个与之前类似的图形



    >>> pvalue = dist.cdf(-res.statistic) + dist.sf(res.statistic)
    计算 p-value，表示 null distribution 中比观察到的统计量更极端的值的比例



    >>> annotation = (f'p-value={pvalue:.4f}\n(shaded area)')
    创建一个带有 p-value 和 "shaded area" 注释信息的字符串



    >>> props = dict(facecolor='black', width=1, headwidth=5, headlength=8)
    创建一个用于注释箭头属性的字典 props



    >>> _ = ax.annotate(annotation, (0.65, 0.15), (0.8, 0.3), arrowprops=props)
    在 ax 坐标轴上添加注释，指定箭头的位置和属性



    >>> i = z_vals >= res.statistic
    创建一个布尔数组，用于标记 z_vals 中大于或等于 res.statistic 的位置



    >>> ax.fill_between(z_vals[i], y1=0, y2=pdf[i], color='C0')
    使用颜色 'C0' 填充 z_vals 中大于或等于 res.statistic 的区域



    >>> i = z_vals <= -res.statistic
    创建一个布尔数组，用于标记 z_vals 中小于或等于 -res.statistic 的位置



    >>> ax.fill_between(z_vals[i], y1=0, y2=pdf[i], color='C0')
    使用颜色 'C0' 填充 z_vals 中小于或等于 -res.statistic 的区域



    >>> ax.set_xlim(-1.25, 1.25)
    设置 x 轴的显示范围为 [-1.25, 1.25]



    >>> ax.set_ylim(0, 0.5)
    设置 y 轴的显示范围为 [0, 0.5]



    >>> plt.show()
    显示绘制好的图形



    >>> res.pvalue
    输出结果对象 res 的 p-value 值



    0.09108705741631495  # approximate p-value
    近似的 p-value 值



    Note that there is slight disagreement between the shaded area of the curve
    and the p-value returned by `kendalltau`. This is because our data has
    ties, and we have neglected a tie correction to the null distribution
    variance that `kendalltau` performs. For samples without ties, the shaded
    areas of our plot and p-value returned by `kendalltau` would match exactly.



    If the p-value is "small" - that is, if there is a low probability of
    sampling data from independent distributions that produces such an extreme
    ```
    """
    Convert input arrays `x` and `y` to numpy arrays if they are not already,
    and ensure they are flattened to 1D arrays. Check for size compatibility
    and presence of NaN values.

    Parameters:
    ----------
    x : array_like
        Input array.
    y : array_like
        Input array.

    Returns:
    -------
    SignificanceResult
        Object containing NaN if arrays are empty or NaN values are present.
    """

    # Convert input arrays to numpy arrays and flatten to 1D arrays
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    # Check if the sizes of x and y are equal
    if x.size != y.size:
        raise ValueError("All inputs to `kendalltau` must be of the same "
                         f"size, found x-size {x.size} and y-size {y.size}")
    # Check if either x or y is empty; return NaN if true
    elif not x.size or not y.size:
        res = SignificanceResult(np.nan, np.nan)
        res.correlation = np.nan
        return res

    # Check for NaN values in both x and y arrays based on the specified policy
    cnx, npx = _contains_nan(x, nan_policy)
    cny, npy = _contains_nan(y, nan_policy)
    contains_nan = cnx or cny
    # 如果 npx 或 npy 中有任何一个为 'omit'，则设定 nan_policy 为 'omit'
    if npx == 'omit' or npy == 'omit':
        nan_policy = 'omit'

    # 如果数据包含 NaN 并且 nan_policy 为 'propagate'，则返回 NaN 的相关性结果
    if contains_nan and nan_policy == 'propagate':
        res = SignificanceResult(np.nan, np.nan)
        res.correlation = np.nan
        return res

    # 如果数据包含 NaN 并且 nan_policy 为 'omit'，则处理 NaN 值并根据 variant 类型计算 Kendall Tau 相关性
    elif contains_nan and nan_policy == 'omit':
        # 将 x 和 y 中的无效值标记为 masked array
        x = ma.masked_invalid(x)
        y = ma.masked_invalid(y)
        if variant == 'b':
            # 使用 mstats_basic 中的 Kendall Tau 计算方法 'b'，包括处理 ties
            return mstats_basic.kendalltau(x, y, method=method, use_ties=True,
                                           alternative=alternative)
        else:
            # 抛出异常，因为当前只有 variant='b' 支持 nan_policy='omit'
            message = ("nan_policy='omit' is currently compatible only with "
                       "variant='b'.")
            raise ValueError(message)

    # 定义一个函数 count_rank_tie，用于计算秩次中的 ties 数量
    def count_rank_tie(ranks):
        cnt = np.bincount(ranks).astype('int64', copy=False)
        cnt = cnt[cnt > 1]
        # 返回计算的 ties 统计结果，避免后续溢出
        return (int((cnt * (cnt - 1) // 2).sum()),
                int((cnt * (cnt - 1.) * (cnt - 2)).sum()),
                int((cnt * (cnt - 1.) * (2*cnt + 5)).sum()))

    # 计算数据集 x 的大小
    size = x.size
    # 根据 y 的排序顺序重新排列 x，并将 y 转换为 dense ranks
    perm = np.argsort(y)
    x, y = x[perm], y[perm]
    y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)

    # 根据 x 的稳定排序顺序重新排列 y，并将 x 转换为 dense ranks
    perm = np.argsort(x, kind='mergesort')
    x, y = x[perm], y[perm]
    x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype=np.intp)

    # 计算 discordant pairs 数量
    dis = _kendall_dis(x, y)

    # 观察值数组，用于标识排列中的 ties 和 discontinuities
    obs = np.r_[True, (x[1:] != x[:-1]) | (y[1:] != y[:-1]), True]
    cnt = np.diff(np.nonzero(obs)[0]).astype('int64', copy=False)

    # 计算 joint ties 的数量
    ntie = int((cnt * (cnt - 1) // 2).sum())
    # 计算 x 和 y 中的 ties 数量以及相关统计信息
    xtie, x0, x1 = count_rank_tie(x)
    ytie, y0, y1 = count_rank_tie(y)

    # 总的 possible pairs 数量
    tot = (size * (size - 1)) // 2

    # 如果 x 或 y 中的 ties 数量等于总可能的 pairs 数量，则返回 NaN 相关性结果
    if xtie == tot or ytie == tot:
        res = SignificanceResult(np.nan, np.nan)
        res.correlation = np.nan
        return res

    # 计算 con_minus_dis，这是 Kendall Tau 相关性的一部分
    con_minus_dis = tot - xtie - ytie + ntie - 2 * dis

    # 根据选择的 variant 计算 Kendall Tau 相关性
    if variant == 'b':
        tau = con_minus_dis / np.sqrt(tot - xtie) / np.sqrt(tot - ytie)
    elif variant == 'c':
        minclasses = min(len(set(x)), len(set(y)))
        tau = 2 * con_minus_dis / (size**2 * (minclasses - 1) / minclasses)
    else:
        # 抛出异常，因为 variant 必须是 'b' 或 'c'
        raise ValueError(f"Unknown variant of the method chosen: {variant}. "
                         "variant must be 'b' or 'c'.")

    # 限制 tau 的范围，以修正计算误差
    tau = np.minimum(1., max(-1., tau))

    # 所有 variant 下 p-value 计算都依赖于 con_minus_dis
    if method == 'exact' and (xtie != 0 or ytie != 0):
        raise ValueError("Ties found, exact method cannot be used.")
    # 如果方法(method)为 'auto'
    if method == 'auto':
        # 检查以下条件：
        # 1. xtie 和 ytie 均为 0
        # 2. size 小于等于 33 或者 dis 和 tot-dis 中的最小值小于等于 1
        if (xtie == 0 and ytie == 0) and (size <= 33 or
                                          min(dis, tot-dis) <= 1):
            # 若以上条件满足，则设置方法为 'exact'
            method = 'exact'
        else:
            # 否则，设置方法为 'asymptotic'
            method = 'asymptotic'
    
    # 如果 xtie 和 ytie 均为 0，并且方法为 'exact'
    if xtie == 0 and ytie == 0 and method == 'exact':
        # 调用 mstats_basic 模块中的 _kendall_p_exact 函数计算精确 p 值
        pvalue = mstats_basic._kendall_p_exact(size, tot-dis, alternative)
    elif method == 'asymptotic':
        # 如果方法为 'asymptotic'
        # 计算方差 var，con_minus_dis 大致服从正态分布，其方差为 [3] 所指明的形式
        m = size * (size - 1.)
        var = ((m * (2*size + 5) - x1 - y1) / 18 +
               (2 * xtie * ytie) / m + x0 * y0 / (9 * m * (size - 2)))
        z = con_minus_dis / np.sqrt(var)
        # 调用 _get_pvalue 函数获取 p 值，使用 _SimpleNormal() 近似正态分布
        pvalue = _get_pvalue(z, _SimpleNormal(), alternative, xp=np)
    else:
        # 如果方法既不是 'exact' 也不是 'asymptotic'，抛出错误
        raise ValueError(f"Unknown method {method} specified.  Use 'auto', "
                         "'exact' or 'asymptotic'.")
    
    # 创建结果对象 res，并为了向后兼容性设置别名
    res = SignificanceResult(tau[()], pvalue[()])
    res.correlation = tau[()]
    # 返回结果对象 res
    return res
# 计算加权 Kendall's tau 系数的函数
def weightedtau(x, y, rank=True, weigher=None, additive=True):
    r"""Compute a weighted version of Kendall's :math:`\tau`.

    The weighted :math:`\tau` is a weighted version of Kendall's
    :math:`\tau` in which exchanges of high weight are more influential than
    exchanges of low weight. The default parameters compute the additive
    hyperbolic version of the index, :math:`\tau_\mathrm h`, which has
    been shown to provide the best balance between important and
    unimportant elements [1]_.

    The weighting is defined by means of a rank array, which assigns a
    nonnegative rank to each element (higher importance ranks being
    associated with smaller values, e.g., 0 is the highest possible rank),
    and a weigher function, which assigns a weight based on the rank to
    each element. The weight of an exchange is then the sum or the product
    of the weights of the ranks of the exchanged elements. The default
    parameters compute :math:`\tau_\mathrm h`: an exchange between
    elements with rank :math:`r` and :math:`s` (starting from zero) has
    weight :math:`1/(r+1) + 1/(s+1)`.

    Specifying a rank array is meaningful only if you have in mind an
    external criterion of importance. If, as it usually happens, you do
    not have in mind a specific rank, the weighted :math:`\tau` is
    defined by averaging the values obtained using the decreasing
    lexicographical rank by (`x`, `y`) and by (`y`, `x`). This is the
    behavior with default parameters. Note that the convention used
    here for ranking (lower values imply higher importance) is opposite
    to that used by other SciPy statistical functions.

    Parameters
    ----------
    x, y : array_like
        Arrays of scores, of the same shape. If arrays are not 1-D, they will
        be flattened to 1-D.
    rank : array_like of ints or bool, optional
        A nonnegative rank assigned to each element. If it is None, the
        decreasing lexicographical rank by (`x`, `y`) will be used: elements of
        higher rank will be those with larger `x`-values, using `y`-values to
        break ties (in particular, swapping `x` and `y` will give a different
        result). If it is False, the element indices will be used
        directly as ranks. The default is True, in which case this
        function returns the average of the values obtained using the
        decreasing lexicographical rank by (`x`, `y`) and by (`y`, `x`).
    weigher : callable, optional
        The weigher function. Must map nonnegative integers (zero
        representing the most important element) to a nonnegative weight.
        The default, None, provides hyperbolic weighing, that is,
        rank :math:`r` is mapped to weight :math:`1/(r+1)`.
    additive : bool, optional
        If True, the weight of an exchange is computed by adding the
        weights of the ranks of the exchanged elements; otherwise, the weights
        are multiplied. The default is True.

    Returns
    -------
    """
    # 将输入的 x 和 y 转换为一维 NumPy 数组
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    # 如果 x 和 y 的大小不一致，则抛出 ValueError 异常
    if x.size != y.size:
        raise ValueError("All inputs to `weightedtau` must be "
                         "of the same size, "
                         f"found x-size {x.size} and y-size {y.size}")
    # 如果 x 的大小为零，返回 NaN 的显著性结果对象
    if not x.size:
        res = SignificanceResult(np.nan, np.nan)
        res.correlation = np.nan
        return res

    # 如果 x 中包含 NaN 值，应用 _toint64() 函数进行处理
    if np.isnan(np.sum(x)):
        x = _toint64(x)
    # 如果 y 中包含 NaN 值，应用 _toint64() 函数进行处理
    if np.isnan(np.sum(y)):
        y = _toint64(y)

    # 如果 x 和 y 的数据类型不同，将不支持的类型转换为 int64
    if x.dtype != y.dtype:
        if x.dtype != np.int64:
            x = _toint64(x)
        if y.dtype != np.int64:
            y = _toint64(y)
    else:
        # 如果数据类型是相同的，但不在支持的类型范围内，转换为 int64
        if x.dtype not in (np.int32, np.int64, np.float32, np.float64):
            x = _toint64(x)
            y = _toint64(y)

    # 如果 rank 参数为 True，计算加权的排名 tau 相关系数
    if rank is True:
        tau = (
            _weightedrankedtau(x, y, None, weigher, additive) +
            _weightedrankedtau(y, x, None, weigher, additive)
        ) / 2
        res = SignificanceResult(tau, np.nan)
        res.correlation = tau
        return res

    # 如果 rank 参数为 False，创建默认的排名
    if rank is False:
        rank = np.arange(x.size, dtype=np.intp)
    # 如果 rank 参数不为 None，则验证其与 x 的大小是否一致
    elif rank is not None:
        rank = np.asarray(rank).ravel()
        if rank.size != x.size:
            raise ValueError(
                "All inputs to `weightedtau` must be of the same size, "
                f"found x-size {x.size} and rank-size {rank.size}"
            )

    # 计算加权的排名 tau 相关系数
    tau = _weightedrankedtau(x, y, rank, weigher, additive)
    res = SignificanceResult(tau, np.nan)
    res.correlation = tau
    return res
#####################################
#       INFERENTIAL STATISTICS      #
#####################################

# 定义一个包含 t 检验结果的命名元组 TtestResultBase，包括 statistic 和 pvalue 属性，以及 df 属性
TtestResultBase = _make_tuple_bunch('TtestResultBase',
                                    ['statistic', 'pvalue'], ['df'])


class TtestResult(TtestResultBase):
    """
    Result of a t-test.

    See the documentation of the particular t-test function for more
    information about the definition of the statistic and meaning of
    the confidence interval.

    Attributes
    ----------
    statistic : float or array
        The t-statistic of the sample.
    pvalue : float or array
        The p-value associated with the given alternative.
    df : float or array
        The number of degrees of freedom used in calculation of the
        t-statistic; this is one less than the size of the sample
        (``a.shape[axis]-1`` if there are no masked elements or omitted NaNs).

    Methods
    -------
    confidence_interval
        Computes a confidence interval around the population statistic
        for the given confidence level.
        The confidence interval is returned in a ``namedtuple`` with
        fields `low` and `high`.

    """

    def __init__(self, statistic, pvalue, df,  # public
                 alternative, standard_error, estimate,  # private
                 statistic_np=None, xp=None):  # private
        # 调用父类的构造函数，初始化 statistic, pvalue 和 df 属性
        super().__init__(statistic, pvalue, df=df)
        self._alternative = alternative  # 存储备择假设
        self._standard_error = standard_error  # 存储 t 统计量的分母
        self._estimate = estimate  # 存储样本均值的点估计
        self._statistic_np = statistic if statistic_np is None else statistic_np  # 存储 t 统计量的 NumPy 数组表示或标量
        self._dtype = statistic.dtype  # 存储 statistic 的数据类型
        self._xp = array_namespace(statistic, pvalue) if xp is None else xp  # 存储数组命名空间

    def confidence_interval(self, confidence_level=0.95):
        """
        Parameters
        ----------
        confidence_level : float
            The confidence level for the calculation of the population mean
            confidence interval. Default is 0.95.

        Returns
        -------
        ci : namedtuple
            The confidence interval is returned in a ``namedtuple`` with
            fields `low` and `high`.

        """
        # 调用 _t_confidence_interval 函数计算置信区间的下限和上限
        low, high = _t_confidence_interval(self.df, self._statistic_np,
                                           confidence_level, self._alternative,
                                           self._dtype, self._xp)
        # 计算置信区间的实际值范围
        low = low * self._standard_error + self._estimate
        high = high * self._standard_error + self._estimate
        # 返回 ConfidenceInterval 命名元组，包含计算出的置信区间的低和高
        return ConfidenceInterval(low=low, high=high)


def pack_TtestResult(statistic, pvalue, df, alternative, standard_error,
                     estimate):
    # 将 alternative 转换为至少是一维的 NumPy 数组，确保可以索引非零维对象
    alternative = np.atleast_1d(alternative)  # can't index 0D object
    # 从 alternative 中选择有限的值，并将其存储回 alternative
    alternative = alternative[np.isfinite(alternative)]
    # 如果 alternative.size 不为零，则取 alternative 列表的第一个元素，否则设置为 NaN
    alternative = alternative[0] if alternative.size else np.nan
    # 返回 T 检验的结果，包括统计量 statistic、p 值 pvalue、自由度 df、备择假设 alternative、标准误差 standard_error、估计值 estimate
    return TtestResult(statistic, pvalue, df=df, alternative=alternative,
                       standard_error=standard_error, estimate=estimate)
# 将 `unpack_TtestResult` 函数应用于 `_axis_nan_policy_factory` 的结果以解包 `TtestResult` 对象
@_axis_nan_policy_factory(pack_TtestResult, default_axis=0, n_samples=2,
                          result_to_tuple=unpack_TtestResult, n_outputs=6)
# `_axis_nan_policy` 处理 `nan_policy`，但需要保留在签名中作为位置参数使用
# 用于计算一组分数的均值的 T 检验。

def ttest_1samp(a, popmean, axis=0, nan_policy="propagate", alternative="two-sided"):
    """Calculate the T-test for the mean of ONE group of scores.

    This is a test for the null hypothesis that the expected value
    (mean) of a sample of independent observations `a` is equal to the given
    population mean, `popmean`.

    Parameters
    ----------
    a : array_like
        Sample observations.
    popmean : float or array_like
        Expected value in null hypothesis. If array_like, then its length along
        `axis` must equal 1, and it must otherwise be broadcastable with `a`.
    axis : int or None, optional
        Axis along which to compute test; default is 0. If None, compute over
        the whole array `a`.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

          * 'propagate': returns nan
          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values

    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

        * 'two-sided': the mean of the underlying distribution of the sample
          is different than the given population mean (`popmean`)
        * 'less': the mean of the underlying distribution of the sample is
          less than the given population mean (`popmean`)
        * 'greater': the mean of the underlying distribution of the sample is
          greater than the given population mean (`popmean`)

    Returns
    -------
    result : `~scipy.stats._result_classes.TtestResult`
        An object with the following attributes:

        statistic : float or array
            The t-statistic.
        pvalue : float or array
            The p-value associated with the given alternative.
        df : float or array
            The number of degrees of freedom used in calculation of the
            t-statistic; this is one less than the size of the sample
            (``a.shape[axis]``).

            .. versionadded:: 1.10.0

        The object also has the following method:

        confidence_interval(confidence_level=0.95)
            Computes a confidence interval around the population
            mean for the given confidence level.
            The confidence interval is returned in a ``namedtuple`` with
            fields `low` and `high`.

            .. versionadded:: 1.10.0

    Notes
    -----
    """
    The statistic is calculated as ``(np.mean(a) - popmean)/se``, where
    ``se`` is the standard error. Therefore, the statistic will be positive
    when the sample mean is greater than the population mean and negative when
    the sample mean is less than the population mean.

    Examples
    --------
    Suppose we wish to test the null hypothesis that the mean of a population
    is equal to 0.5. We choose a confidence level of 99%; that is, we will
    reject the null hypothesis in favor of the alternative if the p-value is
    less than 0.01.

    When testing random variates from the standard uniform distribution, which
    has a mean of 0.5, we expect the data to be consistent with the null
    hypothesis most of the time.

    >>> import numpy as np
    >>> from scipy import stats
    >>> rng = np.random.default_rng()
    >>> rvs = stats.uniform.rvs(size=50, random_state=rng)
    >>> stats.ttest_1samp(rvs, popmean=0.5)
    TtestResult(statistic=2.456308468440, pvalue=0.017628209047638, df=49)


注释：


    # 导入必要的库和模块
    >>> import numpy as np
    >>> from scipy import stats
    # 使用随机数生成器创建一个随机数种子
    >>> rng = np.random.default_rng()
    # 从标准均匀分布中生成大小为50的随机样本
    >>> rvs = stats.uniform.rvs(size=50, random_state=rng)
    # 对生成的样本进行单样本 t 检验，测试其均值是否等于0.5
    >>> stats.ttest_1samp(rvs, popmean=0.5)
    # 输出 t 检验结果，包括统计量(statistic)、p 值(pvalue)和自由度(df)
    TtestResult(statistic=2.456308468440, pvalue=0.017628209047638, df=49)



    As expected, the p-value of 0.017 is not below our threshold of 0.01, so
    we cannot reject the null hypothesis.

    When testing data from the standard *normal* distribution, which has a mean
    of 0, we would expect the null hypothesis to be rejected.

    >>> rvs = stats.norm.rvs(size=50, random_state=rng)
    >>> stats.ttest_1samp(rvs, popmean=0.5)
    TtestResult(statistic=-7.433605518875, pvalue=1.416760157221e-09, df=49)


注释：


    # 生成一个随机样本，来自标准正态分布，具有均值为0和方差为1
    >>> rvs = stats.norm.rvs(size=50, random_state=rng)
    # 对生成的样本进行单样本 t 检验，测试其均值是否等于0.5
    >>> stats.ttest_1samp(rvs, popmean=0.5)
    # 输出 t 检验结果，包括统计量(statistic)、p 值(pvalue)和自由度(df)
    TtestResult(statistic=-7.433605518875, pvalue=1.416760157221e-09, df=49)



    Indeed, the p-value is lower than our threshold of 0.01, so we reject the
    null hypothesis in favor of the default "two-sided" alternative: the mean
    of the population is *not* equal to 0.5.

    However, suppose we were to test the null hypothesis against the
    one-sided alternative that the mean of the population is *greater* than
    0.5. Since the mean of the standard normal is less than 0.5, we would not
    expect the null hypothesis to be rejected.

    >>> stats.ttest_1samp(rvs, popmean=0.5, alternative='greater')
    TtestResult(statistic=-7.433605518875, pvalue=0.99999999929, df=49)


注释：


    # 对相同的样本进行单样本 t 检验，测试其均值是否大于0.5，使用单侧检验
    >>> stats.ttest_1samp(rvs, popmean=0.5, alternative='greater')
    # 输出 t 检验结果，包括统计量(statistic)、p 值(pvalue)和自由度(df)
    TtestResult(statistic=-7.433605518875, pvalue=0.99999999929, df=49)



    Unsurprisingly, with a p-value greater than our threshold, we would not
    reject the null hypothesis.

    Note that when working with a confidence level of 99%, a true null
    hypothesis will be rejected approximately 1% of the time.

    >>> rvs = stats.uniform.rvs(size=(100, 50), random_state=rng)
    >>> res = stats.ttest_1samp(rvs, popmean=0.5, axis=1)
    >>> np.sum(res.pvalue < 0.01)
    1


注释：


    # 从标准均匀分布中生成大小为(100, 50)的随机样本
    >>> rvs = stats.uniform.rvs(size=(100, 50), random_state=rng)
    # 对每一行（axis=1）的样本进行单样本 t 检验，测试均值是否等于0.5
    >>> res = stats.ttest_1samp(rvs, popmean=0.5, axis=1)
    # 统计 p 值小于0.01的数量
    >>> np.sum(res.pvalue < 0.01)
    # 输出符合条件的数量，预期为1
    1



    Indeed, even though all 100 samples above were drawn from the standard
    uniform distribution, which *does* have a population mean of 0.5, we would
    mistakenly reject the null hypothesis for one of them.

    `ttest_1samp` can also compute a confidence interval around the population
    mean.

    >>> rvs = stats.norm.rvs(size=50, random_state=rng)
    >>> res = stats.ttest_1samp(rvs, popmean=0)
    >>> ci = res.confidence_interval(confidence_level=0.95)
    >>> ci


注释：


    # 生成一个随机样本，来自标准正态分布，具有均值为0和方差为1
    >>> rvs = stats.norm.rvs(size=50, random_state=rng)
    # 对生成的样本进行单样本 t 检验，测试其均值是否等于0
    >>> res = stats.ttest_1samp(rvs, popmean=0)
    # 计算均值的置信区间，置信水平为0.95
    >>> ci = res.confidence_interval(confidence_level=0.95)
    # 输出置信区间
    >>> ci
    ConfidenceInterval(low=-0.3193887540880017, high=0.2898583388980972)

    The bounds of the 95% confidence interval are the
    minimum and maximum values of the parameter `popmean` for which the
    p-value of the test would be 0.05.

    >>> res = stats.ttest_1samp(rvs, popmean=ci.low)
    >>> np.testing.assert_allclose(res.pvalue, 0.05)
    >>> res = stats.ttest_1samp(rvs, popmean=ci.high)
    >>> np.testing.assert_allclose(res.pvalue, 0.05)

    Under certain assumptions about the population from which a sample
    is drawn, the confidence interval with confidence level 95% is expected
    to contain the true population mean in 95% of sample replications.

    >>> rvs = stats.norm.rvs(size=(50, 1000), loc=1, random_state=rng)
    >>> res = stats.ttest_1samp(rvs, popmean=0)
    >>> ci = res.confidence_interval()
    >>> contains_pop_mean = (ci.low < 1) & (ci.high > 1)
    >>> contains_pop_mean.sum()
    953

    """
    xp = array_namespace(a)
    # 将数组 `a` 转换为适当的命名空间 `xp`
    a, axis = _chk_asarray(a, axis, xp=xp)
    # 使用 `_chk_asarray` 函数确保 `a` 是一个数组，并设置正确的轴 `axis`

    n = a.shape[axis]
    # 获取数组 `a` 在指定轴上的长度 `n`
    df = n - 1
    # 计算自由度 `df`

    if n == 0:
        # 如果样本大小为 0
        # 这仅在测试 `_axis_nan_policy` 装饰器时需要
        # 在使用装饰器时不会发生这种情况
        NaN = _get_nan(a)
        # 获取数组 `a` 中的 NaN 值
        return TtestResult(NaN, NaN, df=NaN, alternative=NaN,
                           standard_error=NaN, estimate=NaN)

    mean = xp.mean(a, axis=axis)
    # 计算数组 `a` 在指定轴上的平均值 `mean`
    try:
        popmean = xp.asarray(popmean)
        # 将 `popmean` 转换为 `xp` 的数组形式
        popmean = xp.squeeze(popmean, axis=axis) if popmean.ndim > 0 else popmean
        # 如果 `popmean` 的维度大于 0，则在指定轴上去除多余的维度
    except ValueError as e:
        raise ValueError("`popmean.shape[axis]` must equal 1.") from e
        # 如果出现错误，抛出异常，要求 `popmean.shape[axis]` 必须等于 1
    d = mean - popmean
    # 计算均值与 `popmean` 的差值 `d`
    v = _var(a, axis=axis, ddof=1)
    # 计算数组 `a` 在指定轴上的方差 `v`，自由度为 1
    denom = xp.sqrt(v / n)
    # 计算标准误差 `denom`

    with np.errstate(divide='ignore', invalid='ignore'):
        # 设置 `np.errstate`，在计算中忽略除法和无效值错误
        t = xp.divide(d, denom)
        # 计算 t 统计量
        t = t[()] if t.ndim == 0 else t
        # 将 t 统计量转换为标量，如果其维度为 0

    dist = _SimpleStudentT(xp.asarray(df, dtype=t.dtype))
    # 创建简单的学生 t 分布对象 `dist`
    prob = _get_pvalue(t, dist, alternative, xp=xp)
    # 获取 t 统计量的 p 值 `prob`
    prob = prob[()] if prob.ndim == 0 else prob
    # 将 p 值转换为标量，如果其维度为 0

    # 当 nan_policy='omit' 时，不同轴切片的 `df` 可能不同
    df = xp.broadcast_to(xp.asarray(df), t.shape)
    # 将 `df` 广播到与 t 统计量相同的形状
    df = df[()] if df.ndim == 0 else df
    # 将 df 转换为标量，如果其维度为 0
    # `_axis_nan_policy` 装饰器与字符串不兼容
    alternative_num = {"less": -1, "two-sided": 0, "greater": 1}[alternative]
    # 根据 `alternative` 的值选择对应的数值
    return TtestResult(t, prob, df=df, alternative=alternative_num,
                       standard_error=denom, estimate=mean,
                       statistic_np=xp.asarray(t), xp=xp)
    # 返回 t 检验的结果对象 `TtestResult`
# 计算 t 分布下的置信区间的下限和上限
def _t_confidence_interval(df, t, confidence_level, alternative, dtype=None, xp=None):
    # 输入参数 `alternative` 已经通过验证
    # 如果 `dtype` 未指定，则使用 t 的数据类型
    dtype = t.dtype if dtype is None else dtype
    # 如果 `xp` 未指定，则使用 t 的数组命名空间
    xp = array_namespace(t) if xp is None else xp

    # 将 df 和 t 转换为 NumPy 数组
    df, t = np.asarray(df), np.asarray(t)

    # 检查置信水平是否在 0 到 1 之间
    if confidence_level < 0 or confidence_level > 1:
        message = "`confidence_level` 必须是介于 0 和 1 之间的数值."
        raise ValueError(message)

    # 根据 alternative 的不同取值进行不同的置信区间计算
    if alternative < 0:  # 'less'
        p = confidence_level
        # 计算 t 分布的下限
        low, high = np.broadcast_arrays(-np.inf, special.stdtrit(df, p))
    elif alternative > 0:  # 'greater'
        p = 1 - confidence_level
        # 计算 t 分布的上限
        low, high = np.broadcast_arrays(special.stdtrit(df, p), np.inf)
    elif alternative == 0:  # 'two-sided'
        # 计算双侧置信区间的尾部概率
        tail_probability = (1 - confidence_level) / 2
        p = tail_probability, 1 - tail_probability
        # 将 p 调整为正确的形状
        p = np.reshape(p, [2] + [1] * np.asarray(df).ndim)
        # 计算 t 分布的下限和上限
        low, high = special.stdtrit(df, p)
    else:  # 当输入为空时，alternative 为 NaN（参见 _axis_nan_policy）
        p, nans = np.broadcast_arrays(t, np.nan)
        low, high = nans, nans

    # 将结果转换为指定的数据类型，并返回
    low = xp.asarray(low, dtype=dtype)
    low = low[()] if low.ndim == 0 else low
    high = xp.asarray(high, dtype=dtype)
    high = high[()] if high.ndim == 0 else high
    return low, high


# 从统计数据中计算 t 检验的 t 值和概率值
def _ttest_ind_from_stats(mean1, mean2, denom, df, alternative, xp=None):
    xp = array_namespace(mean1, mean2, denom) if xp is None else xp

    # 计算差异 d 和 t 值
    d = mean1 - mean2
    with np.errstate(divide='ignore', invalid='ignore'):
        t = xp.divide(d, denom)

    # 将 t 转换为 NumPy 数组
    t_np = np.asarray(t)
    df_np = np.asarray(df)
    # 获取 p 值
    prob = _get_pvalue(t_np, distributions.t(df_np), alternative, xp=np)
    prob = xp.asarray(prob, dtype=t.dtype)

    # 将 t 和 prob 转换为标量值，并返回
    t = t[()] if t.ndim == 0 else t
    prob = prob[()] if prob.ndim == 0 else prob
    return t, prob


# 计算两组方差不相等条件下的 t 检验的自由度和标准差
def _unequal_var_ttest_denom(v1, n1, v2, n2, xp=None):
    xp = array_namespace(v1, v2) if xp is None else xp
    # 计算各组的方差除以样本数
    vn1 = v1 / n1
    vn2 = v2 / n2
    with np.errstate(divide='ignore', invalid='ignore'):
        # 计算 t 检验的自由度 df
        df = (vn1 + vn2)**2 / (vn1**2 / (n1 - 1) + vn2**2 / (n2 - 1))

    # 如果 df 未定义，说明方差为零（假设 n1 > 0 & n2 > 0）
    # 因此，df 可以设定为任意非 NaN 值
    df = xp.where(xp.isnan(df), xp.asarray(1.), df)
    # 计算标准差 denom
    denom = xp.sqrt(vn1 + vn2)
    return df, denom


# 计算两组方差相等条件下的 t 检验的标准差
def _equal_var_ttest_denom(v1, n1, v2, n2, xp=None):
    xp = array_namespace(v1, v2) if xp is None else xp

    # 如果一个样本中只有一个观测值，导致汇总方差公式失效，因为该样本的方差未定义。
    # 然而，汇总方差仍然是定义的，因为分子中的 (n-1) 应该与分母中的 (n-1) 相抵消，
    # 仅留下与均值的平方差的和：零。
    # 使用xp.asarray将0转换为数组表示，并赋值给zero
    zero = xp.asarray(0.)
    
    # 使用xp.asarray将条件n1 == 1转换为数组，如果条件成立则用zero替换v1的对应位置值
    v1 = xp.where(xp.asarray(n1 == 1), zero, v1)
    
    # 使用xp.asarray将条件n2 == 1转换为数组，如果条件成立则用zero替换v2的对应位置值
    v2 = xp.where(xp.asarray(n2 == 1), zero, v2)

    # 计算df，即自由度，n1 + n2 - 2.0
    df = n1 + n2 - 2.0
    
    # 计算svar，即方差估计值
    svar = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
    
    # 计算denom，即标准差估计值的平方根
    denom = xp.sqrt(svar * (1.0 / n1 + 1.0 / n2))
    
    # 返回df和denom作为结果
    return df, denom
# 定义了一个命名元组 Ttest_indResult，用于存储独立双样本 T 检验的结果
Ttest_indResult = namedtuple('Ttest_indResult', ('statistic', 'pvalue'))

# 定义函数 ttest_ind_from_stats，执行基于描述统计数据的独立双样本 T 检验
def ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2,
                         equal_var=True, alternative="two-sided"):
    r"""
    T-test for means of two independent samples from descriptive statistics.

    This is a test for the null hypothesis that two independent
    samples have identical average (expected) values.

    Parameters
    ----------
    mean1 : array_like
        第一组样本的均值。
    std1 : array_like
        第一组样本的修正标准差（即 ``ddof=1``）。
    nobs1 : array_like
        第一组样本的观测数。
    mean2 : array_like
        第二组样本的均值。
    std2 : array_like
        第二组样本的修正标准差（即 ``ddof=1``）。
    nobs2 : array_like
        第二组样本的观测数。
    equal_var : bool, optional
        如果为 True（默认），执行假定总体方差相等的独立双样本检验 [1]_。
        如果为 False，执行韦尔奇 t 检验，该检验不假定总体方差相等 [2]_。
    alternative : {'two-sided', 'less', 'greater'}, optional
        定义备择假设。
        可选的选项有（默认为 'two-sided'）：

        * 'two-sided': 分布的均值不相等。
        * 'less': 第一分布的均值小于第二分布的均值。
        * 'greater': 第一分布的均值大于第二分布的均值。

        .. versionadded:: 1.6.0

    Returns
    -------
    statistic : float or array
        计算得到的 t 统计量。
    pvalue : float or array
        双侧 p 值。

    See Also
    --------
    scipy.stats.ttest_ind

    Notes
    -----
    统计量计算为 ``(mean1 - mean2)/se``，其中 ``se`` 是标准误差。因此，
    当 `mean1` 大于 `mean2` 时，统计量为正；当 `mean1` 小于 `mean2` 时，
    统计量为负。

    此方法不检查 `std1` 或 `std2` 中的任何元素是否为负数。如果在调用此方法时，
    `std1` 或 `std2` 的任何元素为负数，则此方法将返回与分别传递
    ``numpy.abs(std1)`` 和 ``numpy.abs(std2)`` 相同的结果；不会引发异常或警告。

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/T-test#Independent_two-sample_t-test

    .. [2] https://en.wikipedia.org/wiki/Welch%27s_t-test

    Examples
    --------
    假设我们有两个样本的汇总数据，如下所示（样本方差为修正样本方差）::

                         样本大小   均值   方差
        样本 1           13      15.0  87.5
        样本 2           11      12.0  39.0
    Apply the t-test to this data (with the assumption that the population
    variances are equal):

    >>> import numpy as np
    >>> from scipy.stats import ttest_ind_from_stats
    >>> ttest_ind_from_stats(mean1=15.0, std1=np.sqrt(87.5), nobs1=13,
    ...                      mean2=12.0, std2=np.sqrt(39.0), nobs2=11)
    Ttest_indResult(statistic=0.9051358093310269, pvalue=0.3751996797581487)

    For comparison, here is the data from which those summary statistics
    were taken.  With this data, we can compute the same result using
    `scipy.stats.ttest_ind`:

    >>> a = np.array([1, 3, 4, 6, 11, 13, 15, 19, 22, 24, 25, 26, 26])
    >>> b = np.array([2, 4, 6, 9, 11, 13, 14, 15, 18, 19, 21])
    >>> from scipy.stats import ttest_ind
    >>> ttest_ind(a, b)
    TtestResult(statistic=0.905135809331027,
                pvalue=0.3751996797581486,
                df=22.0)

    Suppose we instead have binary data and would like to apply a t-test to
    compare the proportion of 1s in two independent groups::

                          Number of    Sample     Sample
                    Size    ones        Mean     Variance
        Sample 1    150      30         0.2        0.161073
        Sample 2    200      45         0.225      0.175251

    The sample mean :math:`\hat{p}` is the proportion of ones in the sample
    and the variance for a binary observation is estimated by
    :math:`\hat{p}(1-\hat{p})`.

    >>> ttest_ind_from_stats(mean1=0.2, std1=np.sqrt(0.161073), nobs1=150,
    ...                      mean2=0.225, std2=np.sqrt(0.175251), nobs2=200)
    Ttest_indResult(statistic=-0.5627187905196761, pvalue=0.5739887114209541)

    For comparison, we could compute the t statistic and p-value using
    arrays of 0s and 1s and `scipy.stat.ttest_ind`, as above.

    >>> group1 = np.array([1]*30 + [0]*(150-30))
    >>> group2 = np.array([1]*45 + [0]*(200-45))
    >>> ttest_ind(group1, group2)
    TtestResult(statistic=-0.5627179589855622,
                pvalue=0.573989277115258,
                df=348.0)

    """
    xp = array_namespace(mean1, std1, mean2, std2)

    mean1 = xp.asarray(mean1)  # 将均值1转换为xp数组格式
    std1 = xp.asarray(std1)    # 将标准差1转换为xp数组格式
    mean2 = xp.asarray(mean2)  # 将均值2转换为xp数组格式
    std2 = xp.asarray(std2)    # 将标准差2转换为xp数组格式

    if equal_var:
        df, denom = _equal_var_ttest_denom(std1**2, nobs1, std2**2, nobs2, xp=xp)
    else:
        df, denom = _unequal_var_ttest_denom(std1**2, nobs1, std2**2, nobs2, xp=xp)

    res = _ttest_ind_from_stats(mean1, mean2, denom, df, alternative)  # 调用内部函数计算 t 检验结果
    return Ttest_indResult(*res)
# 使用装饰器工厂 `_axis_nan_policy_factory` 对函数 `ttest_ind` 进行装饰，增加特定行为
@_axis_nan_policy_factory(pack_TtestResult, default_axis=0, n_samples=2,
                          result_to_tuple=unpack_TtestResult, n_outputs=6)
# 定义函数 ttest_ind，用于计算两个独立样本的 T 检验
def ttest_ind(a, b, axis=0, equal_var=True, nan_policy='propagate',
              permutations=None, random_state=None, alternative="two-sided",
              trim=0):
    """
    Calculate the T-test for the means of *two independent* samples of scores.

    This is a test for the null hypothesis that 2 independent samples
    have identical average (expected) values. This test assumes that the
    populations have identical variances by default.

    Parameters
    ----------
    a, b : array_like
        The arrays must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default).
    axis : int or None, optional
        Axis along which to compute test. If None, compute over the whole
        arrays, `a`, and `b`.
    equal_var : bool, optional
        If True (default), perform a standard independent 2 sample test
        that assumes equal population variances [1]_.
        If False, perform Welch's t-test, which does not assume equal
        population variance [2]_.

        .. versionadded:: 0.11.0

    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

          * 'propagate': returns nan
          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values

        The 'omit' option is not currently available for permutation tests or
        one-sided asympyotic tests.

    permutations : non-negative int, np.inf, or None (default), optional
        If 0 or None (default), use the t-distribution to calculate p-values.
        Otherwise, `permutations` is  the number of random permutations that
        will be used to estimate p-values using a permutation test. If
        `permutations` equals or exceeds the number of distinct partitions of
        the pooled data, an exact test is performed instead (i.e. each
        distinct partition is used exactly once). See Notes for details.

        .. versionadded:: 1.7.0

    random_state : {None, int, `numpy.random.Generator`,
            `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

        Pseudorandom number generator state used to generate permutations
        (used only when `permutations` is not None).

        .. versionadded:: 1.7.0
    """
    alternative : {'two-sided', 'less', 'greater'}, optional
        # 定义备择假设的类型。
        # 可选的类型包括：
        # * 'two-sided': 表示两个样本的分布均值不相等。
        # * 'less': 表示第一个样本的分布均值小于第二个样本的均值。
        # * 'greater': 表示第一个样本的分布均值大于第二个样本的均值。
        # 默认为 'two-sided'。

        .. versionadded:: 1.6.0

    trim : float, optional
        # 如果非零，则执行修剪的 t 检验（Yuen's t-test）。
        # 定义从输入样本的每一端修剪的元素比例。
        # 如果为 0（默认），则不会从任一侧修剪元素。
        # 每个尾部修剪的元素数量是修剪比例乘以元素数量的整数部分。
        # 有效范围为 [0, .5)。

        .. versionadded:: 1.7

    Returns
    -------
    result : `~scipy.stats._result_classes.TtestResult`
        # 返回一个对象，包含以下属性：

        statistic : float or ndarray
            # t 统计量。

        pvalue : float or ndarray
            # 与给定备择假设相关的 p 值。

        df : float or ndarray
            # 在 t 统计量计算中使用的自由度数量。
            # 对于置换 t 检验，始终为 NaN。

            .. versionadded:: 1.11.0

        The object also has the following method:

        confidence_interval(confidence_level=0.95)
            # 计算给定置信水平下，人群均值差异的置信区间。
            # 置信区间以一个带有 'low' 和 'high' 字段的命名元组形式返回。
            # 当执行置换 t 检验时，不计算置信区间，'low' 和 'high' 字段包含 NaN。

            .. versionadded:: 1.11.0

    Notes
    -----
    # 假设我们观察到两个独立样本，例如花瓣长度，
    # 我们要考虑这两个样本是否来自同一总体（例如同一种花或具有类似花瓣特征的两个种类）或不同总体。

    # t 检验量化了两个样本算术平均数之间的差异。
    # p 值量化了在假设零假设成立时观察到等于或更极端值的概率，
    # 即样本来自具有相同总体均值的总体的假设。
    # 如果 p 值大于所选阈值（例如 5% 或 1%），则表明我们的观察结果不太可能是偶然发生的。
    # 因此，我们不拒绝等总体均值的零假设。
    # 如果 p 值小于我们的阈值，则我们有证据
    against the null hypothesis of equal population means.
    # 对等总体均值的零假设进行独立双样本 t 检验。

    By default, the p-value is determined by comparing the t-statistic of the
    observed data against a theoretical t-distribution.
    # 默认情况下，p 值通过将观察数据的 t 统计量与理论 t 分布进行比较来确定。

    When ``1 < permutations < binom(n, k)``, where

    * ``k`` is the number of observations in `a`,
    * ``n`` is the total number of observations in `a` and `b`, and
    * ``binom(n, k)`` is the binomial coefficient (``n`` choose ``k``),

    the data are pooled (concatenated), randomly assigned to either group `a`
    or `b`, and the t-statistic is calculated. This process is performed
    repeatedly (`permutation` times), generating a distribution of the
    t-statistic under the null hypothesis, and the t-statistic of the observed
    data is compared to this distribution to determine the p-value.
    # 当 ``1 < permutations < binom(n, k)`` 时，其中
    # * ``k`` 是 `a` 组中的观察次数，
    # * ``n`` 是 `a` 和 `b` 中的总观察次数，
    # * ``binom(n, k)`` 是二项式系数（``n`` 中选择 ``k``），
    # 数据被汇总（连接），随机分配到 `a` 或 `b` 组，并计算 t 统计量。
    # 这个过程重复进行（`permutation` 次），生成零假设下 t 统计量的分布，
    # 并将观察数据的 t 统计量与该分布进行比较，以确定 p 值。

    Specifically, the p-value reported is the "achieved significance level"
    (ASL) as defined in 4.4 of [3]_. Note that there are other ways of
    estimating p-values using randomized permutation tests; for other
    options, see the more general `permutation_test`.
    # 报告的具体 p 值是在文献 [3]_ 的 4.4 节中定义的“实际显著水平”（ASL）。
    # 请注意，使用随机排列测试估算 p 值还有其他方法；有关其他选项，请参阅更通用的 `permutation_test`。

    When ``permutations >= binom(n, k)``, an exact test is performed: the data
    are partitioned between the groups in each distinct way exactly once.
    # 当 ``permutations >= binom(n, k)`` 时，执行精确检验：
    # 数据在每种不同方式下仅分区组一次。

    The permutation test can be computationally expensive and not necessarily
    more accurate than the analytical test, but it does not make strong
    assumptions about the shape of the underlying distribution.
    # 排列检验可能计算成本高，且不一定比分析检验更准确，
    # 但它不对底层分布的形状做出强烈假设。

    Use of trimming is commonly referred to as the trimmed t-test. At times
    called Yuen's t-test, this is an extension of Welch's t-test, with the
    difference being the use of winsorized means in calculation of the variance
    and the trimmed sample size in calculation of the statistic. Trimming is
    recommended if the underlying distribution is long-tailed or contaminated
    with outliers [4]_.
    # 常用修剪称为修剪 t 检验。有时称为 Yuen 的 t 检验，
    # 这是 Welch 的 t 检验的扩展，其区别在于在方差计算中使用修剪均值，
    # 并在统计量计算中使用修剪样本大小。如果底层分布是长尾分布或受离群值污染，建议使用修剪 [4]_。

    The statistic is calculated as ``(np.mean(a) - np.mean(b))/se``, where
    ``se`` is the standard error. Therefore, the statistic will be positive
    when the sample mean of `a` is greater than the sample mean of `b` and
    negative when the sample mean of `a` is less than the sample mean of
    `b`.
    # 统计量计算为 ``(np.mean(a) - np.mean(b))/se``，其中
    # ``se`` 是标准误差。因此，当 `a` 的样本均值大于 `b` 的样本均值时，
    # 统计量将为正；当 `a` 的样本均值小于 `b` 的样本均值时，统计量为负。

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/T-test#Independent_two-sample_t-test

    .. [2] https://en.wikipedia.org/wiki/Welch%27s_t-test

    .. [3] B. Efron and T. Hastie. Computer Age Statistical Inference. (2016).

    .. [4] Yuen, Karen K. "The Two-Sample Trimmed t for Unequal Population
           Variances." Biometrika, vol. 61, no. 1, 1974, pp. 165-170. JSTOR,
           www.jstor.org/stable/2334299. Accessed 30 Mar. 2021.

    .. [5] Yuen, Karen K., and W. J. Dixon. "The Approximate Behaviour and
           Performance of the Two-Sample Trimmed t." Biometrika, vol. 60,
           no. 2, 1973, pp. 369-374. JSTOR, www.jstor.org/stable/2334550.
           Accessed 30 Mar. 2021.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> rng = np.random.default_rng()

    使用 NumPy 提供的随机数生成器 `default_rng()` 创建一个随机数生成器对象 `rng`

    Test with sample with identical means:

    >>> rvs1 = stats.norm.rvs(loc=5, scale=10, size=500, random_state=rng)
    >>> rvs2 = stats.norm.rvs(loc=5, scale=10, size=500, random_state=rng)
    生成均值相同的正态分布样本 `rvs1` 和 `rvs2`，每个样本包含500个数据点，使用前面创建的随机数生成器 `rng` 确保重现性
    >>> stats.ttest_ind(rvs1, rvs2)
    执行独立双样本 t 检验，比较两个样本 `rvs1` 和 `rvs2` 的统计显著性
    TtestResult(statistic=-0.4390847099199348,
                pvalue=0.6606952038870015,
                df=998.0)
    输出 t 检验结果，包括统计量、 p 值和自由度

    >>> stats.ttest_ind(rvs1, rvs2, equal_var=False)
    执行独立双样本 t 检验，考虑两个样本 `rvs1` 和 `rvs2` 方差不相等的情况
    TtestResult(statistic=-0.4390847099199348,
                pvalue=0.6606952553131064,
                df=997.4602304121448)
    输出 t 检验结果，包括统计量、 p 值和自由度

    `ttest_ind` underestimates p for unequal variances:

    >>> rvs3 = stats.norm.rvs(loc=5, scale=20, size=500, random_state=rng)
    生成均值相同但方差不同的正态分布样本 `rvs3`，每个样本包含500个数据点，使用前面创建的随机数生成器 `rng` 确保重现性
    >>> stats.ttest_ind(rvs1, rvs3)
    执行独立双样本 t 检验，比较两个样本 `rvs1` 和 `rvs3` 的统计显著性
    TtestResult(statistic=-1.6370984482905417,
                pvalue=0.1019251574705033,
                df=998.0)
    输出 t 检验结果，包括统计量、 p 值和自由度

    >>> stats.ttest_ind(rvs1, rvs3, equal_var=False)
    执行独立双样本 t 检验，考虑两个样本 `rvs1` 和 `rvs3` 方差不相等的情况
    TtestResult(statistic=-1.637098448290542,
                pvalue=0.10202110497954867,
                df=765.1098655246868)
    输出 t 检验结果，包括统计量、 p 值和自由度

    When ``n1 != n2``, the equal variance t-statistic is no longer equal to the
    unequal variance t-statistic:

    >>> rvs4 = stats.norm.rvs(loc=5, scale=20, size=100, random_state=rng)
    生成均值相同但方差不同的正态分布样本 `rvs4`，每个样本包含100个数据点，使用前面创建的随机数生成器 `rng` 确保重现性
    >>> stats.ttest_ind(rvs1, rvs4)
    执行独立双样本 t 检验，比较两个样本 `rvs1` 和 `rvs4` 的统计显著性
    TtestResult(statistic=-1.9481646859513422,
                pvalue=0.05186270935842703,
                df=598.0)
    输出 t 检验结果，包括统计量、 p 值和自由度

    >>> stats.ttest_ind(rvs1, rvs4, equal_var=False)
    执行独立双样本 t 检验，考虑两个样本 `rvs1` 和 `rvs4` 方差不相等的情况
    TtestResult(statistic=-1.3146566100751664,
                pvalue=0.1913495266513811,
                df=110.41349083985212)
    输出 t 检验结果，包括统计量、 p 值和自由度

    T-test with different means, variance, and n:

    >>> rvs5 = stats.norm.rvs(loc=8, scale=20, size=100, random_state=rng)
    生成均值和方差都不同的正态分布样本 `rvs5`，每个样本包含100个数据点，使用前面创建的随机数生成器 `rng` 确保重现性
    >>> stats.ttest_ind(rvs1, rvs5)
    执行独立双样本 t 检验，比较两个样本 `rvs1` 和 `rvs5` 的统计显著性
    TtestResult(statistic=-2.8415950600298774,
                pvalue=0.0046418707568707885,
                df=598.0)
    输出 t 检验结果，包括统计量、 p 值和自由度

    >>> stats.ttest_ind(rvs1, rvs5, equal_var=False)
    执行独立双样本 t 检验，考虑两个样本 `rvs1` 和 `rvs5` 方差不相等的情况
    TtestResult(statistic=-1.8686598649188084,
                pvalue=0.06434714193919686,
                df=109.32167496550137)
    输出 t 检验结果，包括统计量、 p 值和自由度

    When performing a permutation test, more permutations typically yields
    more accurate results. Use a ``np.random.Generator`` to ensure
    reproducibility:

    >>> stats.ttest_ind(rvs1, rvs5, permutations=10000,
    ...                 random_state=rng)
    执行独立双样本 t 检验，使用置换检验（permutation test）来计算 p 值，增加置换次数可以提高结果的准确性，使用 `np.random.Generator` 确保重现性
    TtestResult(statistic=-2.8415950600298774,
                pvalue=0.0052994700529947,
                df=nan)
    输出 t 检验结果，包括统计量、 p 值和自由度

    Take these two samples, one of which has an extreme tail.

    >>> a = (56, 128.6, 12, 123.8, 64.34, 78, 763.3)
    >>> b = (1.1, 2.9, 4.2)
    定义两个样本 `a` 和 `b`，`a` 包含7个数据点，`b` 包含3个数据点

    Use the `trim` keyword to perform a trimmed (Yuen) t-test. For example,
    using 20% trimming, ``trim=.2``, the test will reduce the impact of one
    (``np.floor(trim*len(a))``) element from each tail of sample `a`. It will
    have no effect on sample `b` because ``np.floor(trim*len(b))`` is 0.

    >>> stats.ttest_ind(a, b, trim=.2)
    执行修剪 t 检验，使用 `trim` 参数进行修剪（Yuen）t 检验，例如，使用 20% 修剪，即 `trim=.2`，该测试将减少样本 `a` 尾部各自一个元素的影响。样本 `b` 不受影响，因为 `np.floor(trim*len(b))` 为0。
    TtestResult(statistic=3.4463884028073513,
                pvalue=0.01369338726499547,
                df=6.0)
    """
    xp = array_namespace(a, b)

    default_float = xp.asarray(1.).dtype
    if xp.isdtype(a.dtype, 'integral'):
        a = xp.astype(a, default_float)
    if xp.isdtype(b.dtype, 'integral'):
        b = xp.astype(b, default_float)

    if not (0 <= trim < .5):
        raise ValueError("Trimming percentage should be 0 <= `trim` < .5.")

    result_shape = _broadcast_array_shapes_remove_axis((a, b), axis=axis)
    NaN = xp.full(result_shape, _get_nan(a, b, xp=xp))
    NaN = NaN[()] if NaN.ndim == 0 else NaN
    if xp_size(a) == 0 or xp_size(b) == 0:
        return TtestResult(NaN, NaN, df=NaN, alternative=NaN,
                           standard_error=NaN, estimate=NaN)

    alternative_nums = {"less": -1, "two-sided": 0, "greater": 1}

    # This probably should be deprecated and replaced with a `method` argument
    if permutations is not None and permutations != 0:
        message = "Use of `permutations` is compatible only with NumPy arrays."
        if not is_numpy(xp):
            raise NotImplementedError(message)

        message = "Use of `permutations` is incompatible with with use of `trim`."
        if trim != 0:
            raise NotImplementedError(message)

        t, prob = _permutation_ttest(a, b, permutations=permutations,
                                     axis=axis, equal_var=equal_var,
                                     nan_policy=nan_policy,
                                     random_state=random_state,
                                     alternative=alternative)
        df, denom, estimate = NaN, NaN, NaN

        # _axis_nan_policy decorator doesn't play well with strings
        return TtestResult(t, prob, df=df, alternative=alternative_nums[alternative],
                           standard_error=denom, estimate=estimate)

    n1 = xp.asarray(a.shape[axis], dtype=a.dtype)
    n2 = xp.asarray(b.shape[axis], dtype=b.dtype)

    # Compute variance and mean for both arrays `a` and `b`
    if trim == 0:
        with np.errstate(divide='ignore', invalid='ignore'):
            v1 = _var(a, axis, ddof=1, xp=xp)
            v2 = _var(b, axis, ddof=1, xp=xp)

        m1 = xp.mean(a, axis=axis)
        m2 = xp.mean(b, axis=axis)
    else:
        # Handling trim option for trimming the data
        message = "Use of `trim` is compatible only with NumPy arrays."
        if not is_numpy(xp):
            raise NotImplementedError(message)

        v1, m1, n1 = _ttest_trim_var_mean_len(a, trim, axis)
        v2, m2, n2 = _ttest_trim_var_mean_len(b, trim, axis)

    # Determine degrees of freedom and denominator for t-test
    if equal_var:
        df, denom = _equal_var_ttest_denom(v1, n1, v2, n2, xp=xp)
    else:
        df, denom = _unequal_var_ttest_denom(v1, n1, v2, n2, xp=xp)

    # Calculate t-statistic and probability
    t, prob = _ttest_ind_from_stats(m1, m2, denom, df, alternative)

    # when nan_policy='omit', `df` can be different for different axis-slices
    df = xp.broadcast_to(df, t.shape)
    df = df[()] if df.ndim == 0 else df
    estimate = m1 - m2
    # 返回一个 TtestResult 对象，包含统计值 t、概率 prob、自由度 df、备择假设的编号、标准误差 denom 和估计值 estimate
    return TtestResult(t, prob, df=df, alternative=alternative_nums[alternative],
                       standard_error=denom, estimate=estimate)
def _ttest_trim_var_mean_len(a, trim, axis):
    """Variance, mean, and length of winsorized input along specified axis"""
    # 对于 `ttest_ind` 使用时进行修剪。
    # 在这个测试中的进一步计算假设输入已经排序。
    # 参考 [4] 第1节“假设 x_1, ..., x_n 是 n 个有序观察值…”
    
    # 按指定的轴对输入 `a` 进行排序
    a = np.sort(a, axis=axis)

    # `g` 是要在每个尾部替换的元素数量，从修剪的百分比转换而来
    n = a.shape[axis]
    g = int(n * trim)

    # 根据指定的 `g` 计算修剪后样本的 Winsorized 方差
    v = _calculate_winsorized_variance(a, g, axis)

    # 修剪后样本的总元素数量
    n -= 2 * g

    # 计算 g 次修剪后的均值，如 [4] (1-1) 中定义
    m = trim_mean(a, trim, axis=axis)
    return v, m, n


def _calculate_winsorized_variance(a, g, axis):
    """Calculates g-times winsorized variance along specified axis"""
    # 预期输入 `a` 沿正确的轴已排序
    if g == 0:
        return _var(a, ddof=1, axis=axis)

    # 将预定轴移到最后，以便更容易操作
    a_win = np.moveaxis(a, axis, -1)

    # 保存 NaN 的位置以供后续使用
    nans_indices = np.any(np.isnan(a_win), axis=-1)

    # Winsorization 和方差计算在 [4] (1-3) 中一步完成，但这里先进行 Winsorization；
    # 左侧和右侧用重复值替换。这可以在 [4] (1-3) 中看到效果，左侧和右侧的尾部被替换为
    # 左侧为 `(g + 1) * x_{g + 1}`，右侧为 `(g + 1) * x_{n - g}`。从零开始索引将
    # `g + 1` 变为 `g`，`n - g` 变为 `- g - 1` 在数组索引中。
    a_win[..., :g] = a_win[..., [g]]
    a_win[..., -g:] = a_win[..., [-g - 1]]

    # 确定方差。在 [4] 中，自由度表达为 `h - 1`，其中 `h = n - 2g`（第1节未编号方程，
    # 第369页末，第370页开头）。这被转换为 NumPy 格式，`n - ddof` 用于 `np.var` 使用。
    # 将结果转换为数组以适应后续索引。
    var_win = np.asarray(_var(a_win, ddof=(2 * g + 1), axis=-1))

    # 使用 `nan_policy='propagate'`，NaN 可能完全被修剪掉，因为它们被排序到数组的尾部。
    # 在这些情况下，用 `np.nan` 替换计算出的方差。
    var_win[nans_indices] = np.nan
    return var_win


def _permutation_distribution_t(data, permutations, size_a, equal_var,
                                random_state=None):
    """Generation permutation distribution of t statistic"""
    random_state = check_random_state(random_state)

    # 准备排列索引
    size = data.shape[-1]
    # 不同组合的数量
    n_max = special.comb(size, size_a)
    # 如果所需的排列数量小于最大允许值 n_max，则使用随机状态生成器创建排列生成器
    if permutations < n_max:
        perm_generator = (random_state.permutation(size)
                          for i in range(permutations))
    else:
        # 如果 permutations 超过了 n_max，将 permutations 设置为 n_max
        permutations = n_max
        # 使用 _all_partitions 函数生成排列生成器
        perm_generator = (np.concatenate(z)
                          for z in _all_partitions(size_a, size-size_a))

    # 初始化一个空列表来存放每次计算的 t 统计量
    t_stat = []
    # 从 perm_generator 中批量获取每次生成的排列索引列表，每次取 50 个
    for indices in _batch_generator(perm_generator, batch=50):
        # 将索引列表转换为 NumPy 数组
        indices = np.array(indices)
        # 根据索引对数据进行排列
        data_perm = data[..., indices]
        # 将排列后的轴移动到最前面，以便与没有此维度的 t_stat_observed 广播
        data_perm = np.moveaxis(data_perm, -2, 0)

        # 分割数据为两部分 a 和 b，以计算 t 统计量
        a = data_perm[..., :size_a]
        b = data_perm[..., size_a:]
        # 计算 t 统计量，并将结果添加到 t_stat 列表中
        t_stat.append(_calc_t_stat(a, b, equal_var))

    # 将所有批次计算得到的 t 统计量连接成一个 NumPy 数组
    t_stat = np.concatenate(t_stat, axis=0)

    # 返回计算得到的 t 统计量、使用的排列次数和实际使用的最大排列数
    return t_stat, permutations, n_max
    compare = {"less": np.less_equal,
               "greater": np.greater_equal,
               "two-sided": lambda x, y: (x <= -np.abs(y)) | (x >= np.abs(y))}
    # 定义一个比较函数字典，根据不同的 alternative 参数选择不同的比较方式

    # 计算观察到的 t 统计量
    t_stat_observed = _calc_t_stat(a, b, equal_var, axis=axis)

    # 获取数组 a 在指定轴上的长度
    na = a.shape[axis]

    # 将数组 a 和 b 沿指定轴广播连接，并将轴移动到最后一维
    mat = _broadcast_concatenate((a, b), axis=axis)
    mat = np.moveaxis(mat, axis, -1)

    # 计算置换分布的 t 统计量及相关值
    t_stat, permutations, n_max = _permutation_distribution_t(
        mat, permutations, size_a=na, equal_var=equal_var,
        random_state=random_state)

    # 根据 alternative 参数选择合适的比较函数，计算 p 值
    # two-sided：双侧检验，对称分布两侧的比较
    # less：左侧检验，观察值小于或等于置换统计量
    # greater：右侧检验，观察值大于或等于置换统计量
    # 根据给定的备选方案使用统计量和观察到的统计量计算比较结果
    cmps = compare[alternative](t_stat, t_stat_observed)
    
    # 对于随机化测试的 p 值计算，应使用偏估计；参考文献：
    # https://www.degruyter.com/document/doi/10.2202/1544-6115.1585/
    adjustment = 1 if n_max > permutations else 0
    
    # 计算每个变量的 p 值，考虑调整
    pvalues = (cmps.sum(axis=0) + adjustment) / (permutations + adjustment)

    # 在统计量计算中，NaN 值会自然传播，但需要手动传播到 p 值中
    if nan_policy == 'propagate' and np.isnan(t_stat_observed).any():
        # 如果 pvalues 是标量，直接将其设为 NaN
        if np.ndim(pvalues) == 0:
            pvalues = np.float64(np.nan)
        else:
            # 将对应 t_stat_observed 为 NaN 的位置在 pvalues 中也设置为 NaN
            pvalues[np.isnan(t_stat_observed)] = np.nan

    # 返回观察到的统计量和计算得到的 p 值
    return (t_stat_observed, pvalues)
# 定义一个函数 `_get_len`，用于获取数组 `a` 在指定轴 `axis` 上的长度
def _get_len(a, axis, msg):
    # 尝试获取数组 `a` 在指定轴 `axis` 上的长度
    try:
        n = a.shape[axis]
    # 如果索引错误，则抛出一个 `AxisError` 异常，其中包含轴号、数组维度和消息 `msg`
    except IndexError:
        raise AxisError(axis, a.ndim, msg) from None
    # 返回数组在指定轴上的长度
    return n

# 使用装饰器 `_axis_nan_policy_factory` 包装 `ttest_rel` 函数，设定一些默认参数和行为
@_axis_nan_policy_factory(pack_TtestResult, default_axis=0, n_samples=2,
                          result_to_tuple=unpack_TtestResult, n_outputs=6,
                          paired=True)
# 定义函数 `ttest_rel`，用于计算两个相关样本的 t 检验
def ttest_rel(a, b, axis=0, nan_policy='propagate', alternative="two-sided"):
    """Calculate the t-test on TWO RELATED samples of scores, a and b.

    This is a test for the null hypothesis that two related or
    repeated samples have identical average (expected) values.

    Parameters
    ----------
    a, b : array_like
        The arrays must have the same shape.
    axis : int or None, optional
        Axis along which to compute test. If None, compute over the whole
        arrays, `a`, and `b`.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

          * 'propagate': returns nan
          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

        * 'two-sided': the means of the distributions underlying the samples
          are unequal.
        * 'less': the mean of the distribution underlying the first sample
          is less than the mean of the distribution underlying the second
          sample.
        * 'greater': the mean of the distribution underlying the first
          sample is greater than the mean of the distribution underlying
          the second sample.

        .. versionadded:: 1.6.0

    Returns
    -------
    result : `~scipy.stats._result_classes.TtestResult`
        An object with the following attributes:

        statistic : float or array
            The t-statistic.
        pvalue : float or array
            The p-value associated with the given alternative.
        df : float or array
            The number of degrees of freedom used in calculation of the
            t-statistic; this is one less than the size of the sample
            (``a.shape[axis]``).

            .. versionadded:: 1.10.0

        The object also has the following method:

        confidence_interval(confidence_level=0.95)
            Computes a confidence interval around the difference in
            population means for the given confidence level.
            The confidence interval is returned in a ``namedtuple`` with
            fields `low` and `high`.

            .. versionadded:: 1.10.0

    Notes
    -----
    Examples for use are scores of the same set of student in
    different exams, or repeated sampling from the same units. The
    test measures whether the average score differs significantly
    """
    # 对两个相关样本进行配对 t 检验，检验它们的平均值是否有显著差异
    return ttest_1samp(a - b, popmean=0, axis=axis, alternative=alternative,
                       _no_deco=True)
# 定义一个字典，将 power_divergence() 函数中 lambda_ 参数的名称映射到相应的数值
_power_div_lambda_names = {
    "pearson": 1,               # Pearson 指数
    "log-likelihood": 0,        # 对数似然比
    "freeman-tukey": -0.5,      # Freeman-Tukey 指数
    "mod-log-likelihood": -1,   # 修正对数似然比
    "neyman": -2,               # Neyman 指数
    "cressie-read": 2/3,        # Cressie-Read 指数
}


def _m_count(a, *, axis, xp):
    """统计数组中未被屏蔽元素的数量。

    此函数类似于 `np.ma.count`，但对于 ndarrays 来说速度更快。
    """
    if hasattr(a, 'count'):
        num = a.count(axis=axis)
        if isinstance(num, np.ndarray) and num.ndim == 0:
            # 在某些情况下，`count` 方法返回一个标量数组（例如 np.array(3)），
            # 但我们希望得到一个普通的整数。
            num = int(num)
    else:
        if axis is None:
            num = xp_size(a)
        else:
            num = a.shape[axis]
    return num


def _m_broadcast_to(a, shape, *, xp):
    """将数组广播到指定的形状。

    如果输入数组是掩码数组，则返回一个掩码数组，否则返回一个广播后的数组。
    """
    if np.ma.isMaskedArray(a):
        return np.ma.masked_array(np.broadcast_to(a, shape),
                                  mask=np.broadcast_to(a.mask, shape))
    return xp.broadcast_to(a, shape)


def _m_sum(a, *, axis, preserve_mask, xp):
    """对数组进行求和。

    如果输入数组是掩码数组，则返回求和结果（保留掩码）；否则返回正常求和结果。
    """
    if np.ma.isMaskedArray(a):
        sum = a.sum(axis)
        return sum if preserve_mask else np.asarray(sum)
    return xp.sum(a, axis=axis)


def _m_mean(a, *, axis, keepdims, xp):
    """计算数组的均值。

    如果输入数组是掩码数组，则返回计算均值结果（保留掩码）；否则返回正常的均值计算结果。
    """
    if np.ma.isMaskedArray(a):
        return np.asarray(a.mean(axis=axis, keepdims=keepdims))
    return xp.mean(a, axis=axis, keepdims=keepdims)


Power_divergenceResult = namedtuple('Power_divergenceResult',
                                    ('statistic', 'pvalue'))


def power_divergence(f_obs, f_exp=None, ddof=0, axis=0, lambda_=None):
    """Cressie-Read 功率分歧统计及拟合优度检验。

    此函数用于检验假设，即分类数据具有给定频率，使用 Cressie-Read 功率分歧统计量。

    Parameters
    ----------
    f_obs : array_like
        每个类别中的观察频率。

        .. deprecated:: 1.14.0
            SciPy 1.14.0 版本开始不再支持掩码数组输入，并将在 1.16.0 版本中移除。

    f_exp : array_like, optional
        每个类别中的期望频率。默认情况下假设所有类别等可能。

        .. deprecated:: 1.14.0
            SciPy 1.14.0 版本开始不再支持掩码数组输入，并将在 1.16.0 版本中移除。

    ddof : int, optional
        "Delta 自由度": 用于 p 值自由度调整。p 值使用自由度为 `k - 1 - ddof` 的卡方分布计算，
        其中 `k` 是观察频率的数量。默认值为 0。
    """
    axis : int or None, optional
        # 指定 `f_obs` 和 `f_exp` 在哪个轴上进行广播以应用检验。如果 axis 是 None，则将 `f_obs` 中的所有值视为单个数据集。默认为 0.
    lambda_ : float or str, optional
        # Cressie-Read 功率差异统计中的幂。默认为 1. 为方便起见，`lambda_` 可以赋予以下字符串之一，对应的数值将被使用：
        #
        # * ``"pearson"``（值为 1）
        #     Pearson 卡方统计量。在这种情况下，函数等效于 `chisquare`。
        # * ``"log-likelihood"``（值为 0）
        #     对数似然比。也称为 G 检验 [3]_。
        # * ``"freeman-tukey"``（值为 -1/2）
        #     Freeman-Tukey 统计量。
        # * ``"mod-log-likelihood"``（值为 -1）
        #     修改的对数似然比。
        # * ``"neyman"``（值为 -2）
        #     Neyman 统计量。
        # * ``"cressie-read"``（值为 2/3）
        #     推荐的功率值 [5]_。
    Returns
    -------
    res: Power_divergenceResult
        # 包含以下属性的对象：
        #
        # statistic : float or ndarray
        #     Cressie-Read 功率差异检验统计量。如果 `axis` 是 None 或者 `f_obs` 和 `f_exp` 是一维的，则是一个浮点数。
        # pvalue : float or ndarray
        #     检验的 p 值。如果 `ddof` 和返回值 `stat` 是标量，则是一个浮点数。
    See Also
    --------
    chisquare
    Notes
    -----
    # 当每个类别中的观测或期望频率过小时，此检验无效。一个典型的规则是所有观测和期望频率都应至少为 5。
    #
    # 此外，观测和期望频率的总和必须相同才能使检验有效；如果和不在相对容差 `eps**0.5` 内一致，其中 `eps` 是输入数据类型的精度，`power_divergence` 将引发错误。
    #
    # 当 `lambda_` 小于零时，统计量的公式涉及除以 `f_obs`，因此如果 `f_obs` 中的任何值为 0，则可能会生成警告或错误。
    #
    # 类似地，当 `lambda_` >= 0 时，如果 `f_exp` 中的任何值为零，则可能会生成警告或错误。
    #
    # 默认的自由度 k-1 适用于未估计分布的任何参数的情况。如果由高效的最大似然估计估计了 p 个参数，则正确的自由度为 k-1-p。如果以不同方式估计了参数，则自由度可以在 k-1-p 和 k-1 之间。然而，也可能渐近分布不是卡方分布，在这种情况下，此检验不适用。
    References
    ----------
    """
    The following code block provides examples and explanations of using the power_divergence function
    from scipy.stats module to perform statistical tests.
    
    Examples include different scenarios:
    - Computing G-test using log-likelihood ratio statistic.
    - Specifying expected frequencies using f_exp argument.
    - Applying tests to each column when f_obs is 2-D.
    - Applying test to flattened array using axis=None.
    - Adjusting degrees of freedom using ddof parameter.
    - Broadcasting f_obs and f_exp arrays and computing chi-squared statistics.
    
    For detailed usage and parameters, refer to the scipy.stats.power_divergence documentation.
    """
    xp = array_namespace(f_obs)
    default_float = xp.asarray(1.).dtype

    # 将输入参数 `lambda_` 转换为数值类型。
    if isinstance(lambda_, str):
        # 如果 `lambda_` 是字符串，检查其是否为预定义的名称之一
        if lambda_ not in _power_div_lambda_names:
            # 如果不是有效的预定义名称，则引发值错误
            names = repr(list(_power_div_lambda_names.keys()))[1:-1]
            raise ValueError(f"invalid string for lambda_: {lambda_!r}. "
                             f"Valid strings are {names}")
        # 将字符串形式的 `lambda_` 转换为对应的数值
        lambda_ = _power_div_lambda_names[lambda_]
    elif lambda_ is None:
        # 如果 `lambda_` 是 None，则设为默认值 1
        lambda_ = 1

    # 定义一个警告函数，用于处理掩码数组输入的警告信息
    def warn_masked(arg):
        if isinstance(arg, ma.MaskedArray):
            message = (
                "`power_divergence` and `chisquare` support for masked array input was "
                "deprecated in SciPy 1.14.0 and will be removed in version 1.16.0.")
            warnings.warn(message, DeprecationWarning, stacklevel=2)

    # 发出关于输入 `f_obs` 的掩码数组的警告
    warn_masked(f_obs)
    # 如果 `f_obs` 是掩码数组，则将其转换为 `xp` 的数组表示形式
    f_obs = f_obs if np.ma.isMaskedArray(f_obs) else xp.asarray(f_obs)
    # 根据 `f_obs` 的数据类型，确定使用的浮点数精度
    dtype = default_float if xp.isdtype(f_obs.dtype, 'integral') else f_obs.dtype
    # 将 `f_obs` 转换为指定的数据类型
    f_obs = (f_obs.astype(dtype) if np.ma.isMaskedArray(f_obs)
             else xp.asarray(f_obs, dtype=dtype))
    # 如果 `f_obs` 具有掩码，则将其转换为浮点数表示形式
    f_obs_float = (f_obs.astype(np.float64) if hasattr(f_obs, 'mask')
                   else xp.asarray(f_obs, dtype=xp.float64))

    # 如果提供了 `f_exp`，则处理预期频数 `f_exp` 的相关逻辑
    if f_exp is not None:
        # 发出关于输入 `f_exp` 的掩码数组的警告
        warn_masked(f_exp)
        # 如果 `f_exp` 是掩码数组，则将其转换为 `xp` 的数组表示形式
        f_exp = f_exp if np.ma.isMaskedArray(f_obs) else xp.asarray(f_exp)
        # 根据 `f_exp` 的数据类型，确定使用的浮点数精度
        dtype = default_float if xp.isdtype(f_exp.dtype, 'integral') else f_exp.dtype
        # 将 `f_exp` 转换为指定的数据类型
        f_exp = (f_exp.astype(dtype) if np.ma.isMaskedArray(f_exp)
                 else xp.asarray(f_exp, dtype=dtype))

        # 计算广播后的形状，以适应 `f_obs_float` 和 `f_exp`
        bshape = _broadcast_shapes((f_obs_float.shape, f_exp.shape))
        # 将 `f_obs_float` 和 `f_exp` 广播至相同形状
        f_obs_float = _m_broadcast_to(f_obs_float, bshape, xp=xp)
        f_exp = _m_broadcast_to(f_exp, bshape, xp=xp)
        # 确定结果数据类型
        dtype_res = xp.result_type(f_obs.dtype, f_exp.dtype)
        # 计算相对误差的相对容忍度
        rtol = xp.finfo(dtype_res).eps**0.5  # 用于通过现有测试
        # 忽略 'invalid' 错误，计算 `f_obs_float` 和 `f_exp` 的和
        with np.errstate(invalid='ignore'):
            f_obs_sum = _m_sum(f_obs_float, axis=axis, preserve_mask=False, xp=xp)
            f_exp_sum = _m_sum(f_exp, axis=axis, preserve_mask=False, xp=xp)
            # 计算相对差异
            relative_diff = (xp.abs(f_obs_sum - f_exp_sum) /
                             xp_minimum(f_obs_sum, f_exp_sum))
            # 检查是否有任何相对差异超过容忍度 `rtol`
            diff_gt_tol = xp.any(relative_diff > rtol, axis=None)
        if diff_gt_tol:
            # 如果有相对差异超过容忍度，则引发值错误
            msg = (f"For each axis slice, the sum of the observed "
                   f"frequencies must agree with the sum of the "
                   f"expected frequencies to a relative tolerance "
                   f"of {rtol}, but the percent differences are:\n"
                   f"{relative_diff}")
            raise ValueError(msg)

    else:
        # 如果未提供 `f_exp`，则忽略 'invalid' 错误，计算 `f_obs` 的平均值
        with np.errstate(invalid='ignore'):
            f_exp = _m_mean(f_obs, axis=axis, keepdims=True, xp=xp)
    # `terms` 是用来计算检验统计量的项的数组，沿着 `axis` 被求和。我们针对几种特殊情况的 lambda_ 使用了一些专门的代码。
    if lambda_ == 1:
        # 当 lambda_ 等于 1 时，使用 Pearson 卡方统计量
        terms = (f_obs - f_exp)**2 / f_exp
    elif lambda_ == 0:
        # 当 lambda_ 等于 0 时，使用对数似然比（即 G 检验）
        terms = 2.0 * special.xlogy(f_obs, f_obs / f_exp)
    elif lambda_ == -1:
        # 当 lambda_ 等于 -1 时，使用修改的对数似然比
        terms = 2.0 * special.xlogy(f_exp, f_exp / f_obs)
    else:
        # 对于一般的 Cressie-Read 动力分歧。
        terms = f_obs * ((f_obs / f_exp)**lambda_ - 1)
        terms /= 0.5 * lambda_ * (lambda_ + 1)

    # 计算统计量
    stat = _m_sum(terms, axis=axis, preserve_mask=True, xp=xp)

    # 计算观测数
    num_obs = _m_count(terms, axis=axis, xp=xp)

    # 将 ddof 转换为数组
    df = xp.asarray(num_obs - 1 - ddof)

    # 创建简单的 chi-squared 对象
    chi2 = _SimpleChi2(df)

    # 获取 p 值
    pvalue = _get_pvalue(stat, chi2, alternative='greater', symmetric=False, xp=xp)

    # 如果 stat 是零维数组，则转换为标量
    stat = stat[()] if stat.ndim == 0 else stat
    # 如果 pvalue 是零维数组，则转换为标量
    pvalue = pvalue[()] if pvalue.ndim == 0 else pvalue

    # 返回 Power_divergenceResult 对象，包含统计量和 p 值
    return Power_divergenceResult(stat, pvalue)
# 定义一个计算单因素卡方检验的函数
def chisquare(f_obs, f_exp=None, ddof=0, axis=0):
    """Calculate a one-way chi-square test.

    The chi-square test tests the null hypothesis that the categorical data
    has the given frequencies.

    Parameters
    ----------
    f_obs : array_like
        Observed frequencies in each category.
    f_exp : array_like, optional
        Expected frequencies in each category.  By default the categories are
        assumed to be equally likely.
    ddof : int, optional
        "Delta degrees of freedom": adjustment to the degrees of freedom
        for the p-value.  The p-value is computed using a chi-squared
        distribution with ``k - 1 - ddof`` degrees of freedom, where `k`
        is the number of observed frequencies.  The default value of `ddof`
        is 0.
    axis : int or None, optional
        The axis of the broadcast result of `f_obs` and `f_exp` along which to
        apply the test.  If axis is None, all values in `f_obs` are treated
        as a single data set.  Default is 0.

    Returns
    -------
    res: Power_divergenceResult
        An object containing attributes:

        statistic : float or ndarray
            The chi-squared test statistic.  The value is a float if `axis` is
            None or `f_obs` and `f_exp` are 1-D.
        pvalue : float or ndarray
            The p-value of the test.  The value is a float if `ddof` and the
            result attribute `statistic` are scalars.

    See Also
    --------
    scipy.stats.power_divergence
    scipy.stats.fisher_exact : Fisher exact test on a 2x2 contingency table.
    scipy.stats.barnard_exact : An unconditional exact test. An alternative
        to chi-squared test for small sample sizes.
    :ref:`hypothesis_chisquare`

    Notes
    -----
    This test is invalid when the observed or expected frequencies in each
    category are too small.  A typical rule is that all of the observed
    and expected frequencies should be at least 5. According to [3]_, the
    total number of samples is recommended to be greater than 13,
    otherwise exact tests (such as Barnard's Exact test) should be used
    because they do not overreject.

    Also, the sum of the observed and expected frequencies must be the same
    for the test to be valid; `chisquare` raises an error if the sums do not
    agree within a relative tolerance of ``1e-8``.

    The default degrees of freedom, k-1, are for the case when no parameters
    of the distribution are estimated. If p parameters are estimated by
    efficient maximum likelihood then the correct degrees of freedom are
    k-1-p. If the parameters are estimated in a different way, then the
    dof can be between k-1-p and k-1. However, it is also possible that
    the asymptotic distribution is not chi-square, in which case this test
    is not appropriate.

    References
    ----------
    """
    """
    .. [1] Lowry, Richard.  "Concepts and Applications of Inferential
           Statistics". Chapter 8.
           https://web.archive.org/web/20171022032306/http://vassarstats.net:80/textbook/ch8pt1.html
    .. [2] "Chi-squared test", https://en.wikipedia.org/wiki/Chi-squared_test
    .. [3] Pearson, Karl. "On the criterion that a given system of deviations from the probable
           in the case of a correlated system of variables is such that it can be reasonably
           supposed to have arisen from random sampling", Philosophical Magazine. Series 5. 50
           (1900), pp. 157-175.
    
    Examples
    --------
    When only the mandatory `f_obs` argument is given, it is assumed that the
    expected frequencies are uniform and given by the mean of the observed
    frequencies:
    
    >>> import numpy as np
    >>> from scipy.stats import chisquare
    >>> chisquare([16, 18, 16, 14, 12, 12])
    Power_divergenceResult(statistic=2.0, pvalue=0.84914503608460956)
    
    The optional `f_exp` argument gives the expected frequencies.
    
    >>> chisquare([16, 18, 16, 14, 12, 12], f_exp=[16, 16, 16, 16, 16, 8])
    Power_divergenceResult(statistic=3.5, pvalue=0.62338762774958223)
    
    When `f_obs` is 2-D, by default the test is applied to each column.
    
    >>> obs = np.array([[16, 18, 16, 14, 12, 12], [32, 24, 16, 28, 20, 24]]).T
    >>> obs.shape
    (6, 2)
    >>> chisquare(obs)
    Power_divergenceResult(statistic=array([2.        , 6.66666667]), pvalue=array([0.84914504, 0.24663415]))
    
    By setting ``axis=None``, the test is applied to all data in the array,
    which is equivalent to applying the test to the flattened array.
    
    >>> chisquare(obs, axis=None)
    Power_divergenceResult(statistic=23.31034482758621, pvalue=0.015975692534127565)
    >>> chisquare(obs.ravel())
    Power_divergenceResult(statistic=23.310344827586206, pvalue=0.01597569253412758)
    
    `ddof` is the change to make to the default degrees of freedom.
    
    >>> chisquare([16, 18, 16, 14, 12, 12], ddof=1)
    Power_divergenceResult(statistic=2.0, pvalue=0.7357588823428847)
    
    The calculation of the p-values is done by broadcasting the
    chi-squared statistic with `ddof`.
    
    >>> chisquare([16, 18, 16, 14, 12, 12], ddof=[0, 1, 2])
    Power_divergenceResult(statistic=2.0, pvalue=array([0.84914504, 0.73575888, 0.5724067 ]))
    
    `f_obs` and `f_exp` are also broadcast.  In the following, `f_obs` has
    shape (6,) and `f_exp` has shape (2, 6), so the result of broadcasting
    `f_obs` and `f_exp` has shape (2, 6).  To compute the desired chi-squared
    statistics, we use ``axis=1``:
    
    >>> chisquare([16, 18, 16, 14, 12, 12],
    ...           f_exp=[[16, 16, 16, 16, 16, 8], [8, 20, 20, 16, 12, 12]],
    ...           axis=1)
    Power_divergenceResult(statistic=array([3.5 , 9.25]), pvalue=array([0.62338763, 0.09949846]))
    
    For a more detailed example, see :ref:`hypothesis_chisquare`.
    """  # noqa: E501
    # 调用 power_divergence 函数，计算给定观察频数和期望频数的差异度
    # f_obs: 观察频数
    # f_exp: 期望频数
    # ddof: 自由度的修正量
    # axis: 计算的轴
    # lambda_: Pearson 检验所使用的指定方法
    return power_divergence(f_obs, f_exp=f_exp, ddof=ddof, axis=axis,
                            lambda_="pearson")
# 定义了一个名为 KstestResult 的类，用于表示 Kolmogorov-Smirnov 检验的结果。
# 该类包含 statistic 和 pvalue 两个属性，并且额外包含 statistic_location 和 statistic_sign 两个属性。
KstestResult = _make_tuple_bunch('KstestResult', ['statistic', 'pvalue'],
                                 ['statistic_location', 'statistic_sign'])


def _compute_dplus(cdfvals, x):
    """计算 Kolmogorov-Smirnov 检验中的 D+ 值。

    Parameters
    ----------
    cdfvals : array_like
        排序后的累积分布函数（CDF）值数组，取值范围为 [0, 1]
    x: array_like
        排序后的随机变量数组本身

    Returns
    -------
    res: Pair，包含以下元素：
        - CDF 值低于 Uniform(0, 1) 的最大距离
        - 达到最大距离时的位置

    """
    n = len(cdfvals)
    # 计算 D+ 值
    dplus = (np.arange(1.0, n + 1) / n - cdfvals)
    # 找出最大值的索引
    amax = dplus.argmax()
    # 获取最大值所在位置
    loc_max = x[amax]
    return (dplus[amax], loc_max)


def _compute_dminus(cdfvals, x):
    """计算 Kolmogorov-Smirnov 检验中的 D- 值。

    Parameters
    ----------
    cdfvals : array_like
        排序后的累积分布函数（CDF）值数组，取值范围为 [0, 1]
    x: array_like
        排序后的随机变量数组本身

    Returns
    -------
    res: Pair，包含以下元素：
        - CDF 值高于 Uniform(0, 1) 的最大距离
        - 达到最大距离时的位置
    """
    n = len(cdfvals)
    # 计算 D- 值
    dminus = (cdfvals - np.arange(0.0, n) / n)
    # 找出最大值的索引
    amax = dminus.argmax()
    # 获取最大值所在位置
    loc_max = x[amax]
    return (dminus[amax], loc_max)


def _tuple_to_KstestResult(statistic, pvalue,
                           statistic_location, statistic_sign):
    """将元组转换为 KstestResult 类对象。

    Parameters
    ----------
    statistic : float
        统计量
    pvalue : float
        p 值
    statistic_location : float
        统计量位置
    statistic_sign : float
        统计量标志

    Returns
    -------
    KstestResult
        包含给定统计量、p 值、统计量位置和统计量标志的 KstestResult 对象
    """
    return KstestResult(statistic, pvalue,
                        statistic_location=statistic_location,
                        statistic_sign=statistic_sign)


def _KstestResult_to_tuple(res):
    """将 KstestResult 对象转换为元组。

    Parameters
    ----------
    res : KstestResult
        KstestResult 对象

    Returns
    -------
    tuple
        包含 KstestResult 对象中 statistic、pvalue、statistic_location 和 statistic_sign 的元组
    """
    return *res, res.statistic_location, res.statistic_sign


@_axis_nan_policy_factory(_tuple_to_KstestResult, n_samples=1, n_outputs=4,
                          result_to_tuple=_KstestResult_to_tuple)
@_rename_parameter("mode", "method")
def ks_1samp(x, cdf, args=(), alternative='two-sided', method='auto'):
    """
    对单样本进行 Kolmogorov-Smirnov 检验以评估拟合度。

    该检验比较了样本的基础分布 F(x) 与给定连续分布 G(x)。请参阅注释获取有关空假设和备择假设的描述。

    Parameters
    ----------
    x : array_like
        包含 iid 随机变量观测值的 1-D 数组。
    cdf : callable
        用于计算累积分布函数的可调用对象。
    args : tuple, sequence, optional
        与 `cdf` 一起使用的分布参数。
    alternative : {'two-sided', 'less', 'greater'}, optional
        定义空假设和备择假设。默认为 'two-sided'。
        请参阅下面的注释以获取更多解释。

    """
    # 方法参数：定义用于计算 p 值的分布类型，可选值为{'auto', 'exact', 'approx', 'asymp'}
    # 'auto'：自动选择其中一种选项。
    # 'exact'：使用测试统计量的精确分布。
    # 'approx'：使用两侧概率的两倍来近似计算双侧概率。
    # 'asymp'：使用测试统计量的渐近分布。
    method : {'auto', 'exact', 'approx', 'asymp'}, optional
        定义用于计算 p 值的分布类型。以下选项可用（默认为 'auto'）：

          * 'auto'：自动选择其他选项之一。
          * 'exact'：使用测试统计量的精确分布。
          * 'approx'：使用两侧概率的两倍来近似计算双侧概率。
          * 'asymp'：使用测试统计量的渐近分布。

    Returns
    -------
    res: KstestResult
        一个包含以下属性的对象：

        statistic : float
            KS 检验统计量，可以是 D+、D- 或 D（两者中的最大值）。
        pvalue : float
            单侧或双侧 p 值。
        statistic_location : float
            与 KS 统计量相对应的值 `x`；即在此观察下测量经验分布函数与假设累积分布函数之间的距离。
        statistic_sign : int
            如果 KS 统计量是经验分布函数与假设累积分布函数之间的最大正差异（D+），则为 +1；
            如果 KS 统计量是最大负差异（D-），则为 -1。

    See Also
    --------
    ks_2samp, kstest

    Notes
    -----
    使用 `alternative` 参数可以选择三种零假设及相应备择假设的选项。

    - `two-sided`：零假设是两个分布相同，即 F(x)=G(x) 对所有 x；备择假设是它们不同。
    
    - `less`：零假设是对所有 x，F(x) >= G(x)；备择假设是至少有一个 x，F(x) < G(x)。
    
    - `greater`：零假设是对所有 x，F(x) <= G(x)；备择假设是至少有一个 x，F(x) > G(x)。

    注意备择假设描述的是基础分布的 *CDF*，而不是观察值。例如，假设 x1 ~ F 和 x2 ~ G。
    如果对所有 x，F(x) > G(x)，则 x1 中的值倾向于小于 x2 中的值。

    Examples
    --------
    假设我们希望测试样本是否符合标准正态分布的零假设。
    我们选择置信水平为 95%；即，如果 p 值小于 0.05，则我们将拒绝零假设，支持备择假设。

    当测试均匀分布数据时，我们预计会拒绝零假设。

    >>> import numpy as np
    >>> from scipy import stats
    >>> rng = np.random.default_rng()
    >>> stats.ks_1samp(stats.uniform.rvs(size=100, random_state=rng),
    ...                stats.norm.cdf)
    KstestResult(statistic=0.5001899973268688,
                 pvalue=1.1616392184763533e-23,
                 statistic_location=0.00047625268963724654,
                 statistic_sign=-1)

    Indeed, the p-value is lower than our threshold of 0.05, so we reject the
    null hypothesis in favor of the default "two-sided" alternative: the data
    are *not* distributed according to the standard normal.

    When testing random variates from the standard normal distribution, we
    expect the data to be consistent with the null hypothesis most of the time.

    >>> x = stats.norm.rvs(size=100, random_state=rng)
    >>> stats.ks_1samp(x, stats.norm.cdf)
    KstestResult(statistic=0.05345882212970396,
                 pvalue=0.9227159037744717,
                 statistic_location=-1.2451343873745018,
                 statistic_sign=1)

    As expected, the p-value of 0.92 is not below our threshold of 0.05, so
    we cannot reject the null hypothesis.

    Suppose, however, that the random variates are distributed according to
    a normal distribution that is shifted toward greater values. In this case,
    the cumulative density function (CDF) of the underlying distribution tends
    to be *less* than the CDF of the standard normal. Therefore, we would
    expect the null hypothesis to be rejected with ``alternative='less'``:

    >>> x = stats.norm.rvs(size=100, loc=0.5, random_state=rng)
    >>> stats.ks_1samp(x, stats.norm.cdf, alternative='less')
    KstestResult(statistic=0.17482387821055168,
                 pvalue=0.001913921057766743,
                 statistic_location=0.3713830565352756,
                 statistic_sign=-1)

    and indeed, with p-value smaller than our threshold, we reject the null
    hypothesis in favor of the alternative.



    """
    mode = method

    alternative = {'t': 'two-sided', 'g': 'greater', 'l': 'less'}.get(
        alternative.lower()[0], alternative)
    if alternative not in ['two-sided', 'greater', 'less']:
        raise ValueError(f"Unexpected value {alternative=}")

    N = len(x)
    x = np.sort(x)
    cdfvals = cdf(x, *args)
    np_one = np.int8(1)

    if alternative == 'greater':
        Dplus, d_location = _compute_dplus(cdfvals, x)
        return KstestResult(Dplus, distributions.ksone.sf(Dplus, N),
                            statistic_location=d_location,
                            statistic_sign=np_one)

    if alternative == 'less':
        Dminus, d_location = _compute_dminus(cdfvals, x)
        return KstestResult(Dminus, distributions.ksone.sf(Dminus, N),
                            statistic_location=d_location,
                            statistic_sign=-np_one)

    # alternative == 'two-sided':
    Dplus, dplus_location = _compute_dplus(cdfvals, x)
    Dminus, dminus_location = _compute_dminus(cdfvals, x)
    if Dplus > Dminus:
        D = Dplus
        d_location = dplus_location
        d_sign = np_one


注释：

    # 将 method 赋值给 mode，这里未提供 method 的具体定义
    mode = method

    # 根据 alternative 的值选择对应的检验假设
    alternative = {'t': 'two-sided', 'g': 'greater', 'l': 'less'}.get(
        alternative.lower()[0], alternative)
    # 如果 alternative 不在预期的取值范围内，抛出 ValueError 异常
    if alternative not in ['two-sided', 'greater', 'less']:
        raise ValueError(f"Unexpected value {alternative=}")

    # 计算数据集 x 的长度 N
    N = len(x)
    # 对数据集 x 进行排序
    x = np.sort(x)
    # 计算数据集 x 的累积分布函数值 cdfvals
    cdfvals = cdf(x, *args)
    # 定义一个整数型变量 np_one，并赋值为 1
    np_one = np.int8(1)

    # 如果 alternative 为 'greater'，计算 Dplus 和对应的位置信息 d_location
    if alternative == 'greater':
        Dplus, d_location = _compute_dplus(cdfvals, x)
        # 返回 K-S 检验结果，包括 Dplus、p-value 和位置信息，统计标志为 1
        return KstestResult(Dplus, distributions.ksone.sf(Dplus, N),
                            statistic_location=d_location,
                            statistic_sign=np_one)

    # 如果 alternative 为 'less'，计算 Dminus 和对应的位置信息 d_location
    if alternative == 'less':
        Dminus, d_location = _compute_dminus(cdfvals, x)
        # 返回 K-S 检验结果，包括 Dminus、p-value 和位置信息，统计标志为 -1
        return KstestResult(Dminus, distributions.ksone.sf(Dminus, N),
                            statistic_location=d_location,
                            statistic_sign=-np_one)

    # 如果 alternative 为 'two-sided'，同时计算 Dplus 和 Dminus，选择较大的值作为 D
    Dplus, dplus_location = _compute_dplus(cdfvals, x)
    Dminus, dminus_location = _compute_dminus(cdfvals, x)
    if Dplus > Dminus:
        D = Dplus
        d_location = dplus_location
        d_sign = np_one
    # 如果mode为'auto'，始终选择'exact'模式
    if mode == 'auto':  # Always select exact
        mode = 'exact'
    
    # 如果mode为'exact'，使用Kolmogorov-Smirnov分布的累积分布函数计算概率
    if mode == 'exact':
        prob = distributions.kstwo.sf(D, N)
    
    # 如果mode为'asymp'，使用Kolmogorov-Smirnov大样本近似分布的累积分布函数计算概率
    elif mode == 'asymp':
        prob = distributions.kstwobign.sf(D * np.sqrt(N))
    
    # 如果mode既不是'exact'也不是'asymp'，假设mode为'approx'，使用Kolmogorov-Smirnov单样本检验的累积分布函数计算概率
    else:
        prob = 2 * distributions.ksone.sf(D, N)
    
    # 将计算得到的概率值限制在0到1之间
    prob = np.clip(prob, 0, 1)
    
    # 返回Kolmogorov-Smirnov检验的结果对象，包括统计量D和计算得到的概率prob
    return KstestResult(D, prob,
                        statistic_location=d_location,
                        statistic_sign=d_sign)
# 将 KstestResult 赋值给 Ks_2sampResult，即将 KstestResult 的别名设置为 Ks_2sampResult
Ks_2sampResult = KstestResult


def _compute_prob_outside_square(n, h):
    """
    计算路径中超出两条对角线之外的比例。

    Parameters
    ----------
    n : integer
        n > 0
        n 为正整数
    h : integer
        0 <= h <= n
        h 为整数，且在 0 到 n 之间（包含）

    Returns
    -------
    p : float
        超出 x-y = +/-h 对角线的路径比例。

    """
    # 计算 Pr(D_{n,n} >= h/n)
    # Prob = 2 * ( binom(2n, n-h) - binom(2n, n-2a) + binom(2n, n-3a) - ... )
    # / binom(2n, n)
    # 此公式展示了减法取消的现象。
    # 可以通过将每个项除以 binom(2n, n)，然后因式分解公共项，使用类似 Horner 的算法
    # P = 2 * A0 * (1 - A1*(1 - A2*(1 - A3*(1 - A4*(...)))))

    P = 0.0
    k = int(np.floor(n / h))
    while k >= 0:
        p1 = 1.0
        # 每个 Ai 项的分子和分母都有 h 个简单项。
        for j in range(h):
            p1 = (n - k * h - j) * p1 / (n + k * h + j + 1)
        P = p1 * (1.0 - P)
        k -= 1
    return 2 * P


def _count_paths_outside_method(m, n, g, h):
    """统计超出指定对角线的路径数量。

    Parameters
    ----------
    m : integer
        m > 0
        m 为正整数
    n : integer
        n > 0
        n 为正整数
    g : integer
        m 和 n 的最大公约数
    h : integer
        0 <= h <= lcm(m,n)
        h 为整数，且在 0 到 lcm(m,n) 之间（包含）

    Returns
    -------
    p : float
        通过低路径的数量。
        计算可能会溢出 - 检查是否有限答案。

    Notes
    -----
    计算从 (0, 0) 到 (m, n) 的整数格点路径，其中某些路径点 (x, y) 满足：
      m*y <= n*x - h*g
    路径步长为 +1，只能向正 x 或正 y 方向移动。

    我们一般遵循 Hodges 的处理方式，参考 Drion/Gnedenko/Korolyuk 的方法。
    Hodges, J.L. Jr.,
    "The Significance Probability of the Smirnov Two-Sample Test,"
    Arkiv fiur Matematik, 3, No. 43 (1958), 469-86.

    """
    # 计算 #paths，这些路径低于 x/m-y/n = h/lcm(m,n)
    # B(x, y) = #{从 (0,0) 到 (x,y) 路径，但未曾越过边界}
    #         = binom(x, y) - #{已经达到边界的路径}
    # 乘以从 (x, y) 到 (m, n) 的路径扩展数
    # 求和。

    # 概率在 m, n 对称。以下计算假设 m >= n。
    if m < n:
        m, n = n, m
    mg = m // g
    ng = n // g

    # 不需要考虑每个 x。
    # xj 包含要检查的 x 值列表。
    # 在 n*x/m + ng*h 跨过整数的任意位置
    lxj = n + (mg-h)//mg
    xj = [(h + mg * j + ng-1)//ng for j in range(lxj)]
    # B 是一个数组，只存储几个需要的 B(x,y) 值。
    # B[j] == B(x_j, j)
    if lxj == 0:
        return special.binom(m + n, n)
    B = np.zeros(lxj)
    B[0] = 1
    # 计算 B(x, y) 项
    # 对于每个 j 在范围 [1, lxj) 内进行循环
    for j in range(1, lxj):
        # 计算 B[j] 的值，其中 xj[j] 表示路径中第 j 个节点的值
        Bj = special.binom(xj[j] + j, j)
        # 对于每个 i 在范围 [0, j) 内进行循环
        for i in range(j):
            # 计算二项式系数 bin，表示路径中第 j 个节点与第 i 个节点之间的组合数
            bin = special.binom(xj[j] - xj[i] + j - i, j-i)
            # 使用 bin 与 B[i] 计算 Bj 的更新值
            Bj -= bin * B[i]
        # 将计算得到的 Bj 存储到 B 列表中的第 j 个位置
        B[j] = Bj
    
    # 计算路径扩展的数量...
    num_paths = 0
    # 对于每个 j 在范围 [0, lxj) 内进行循环
    for j in range(lxj):
        # 计算二项式系数 bin，表示从当前路径的第 j 个节点到目标点的路径数量
        bin = special.binom((m-xj[j]) + (n - j), n-j)
        # 计算 B[j] 乘以 bin 的结果，并将结果添加到 num_paths 中
        term = B[j] * bin
        num_paths += term
    
    # 返回计算得到的路径扩展数量
    return num_paths
# 尝试计算精确的双样本 Kolmogorov-Smirnov 检验概率。

def _attempt_exact_2kssamp(n1, n2, g, d, alternative):
    """Attempts to compute the exact 2sample probability.

    n1, n2 are the sample sizes  # n1, n2 分别是两个样本的大小
    g is the gcd(n1, n2)         # g 是 n1 和 n2 的最大公约数
    d is the computed max difference in ECDFs  # d 是 ECDF 最大差异的计算值

    Returns (success, d, probability)  # 返回元组 (是否成功, d, 概率)
    """
    lcm = (n1 // g) * n2  # 计算 n1 和 n2 的最小公倍数
    h = int(np.round(d * lcm))  # 根据 d 和 lcm 计算 h，并四舍五入为整数
    d = h * 1.0 / lcm  # 更新 d 为 h 除以 lcm 的浮点数结果
    if h == 0:
        return True, d, 1.0  # 如果 h 为 0，则返回成功、更新后的 d 和概率为 1.0
    saw_fp_error, prob = False, np.nan  # 初始化错误标志为 False，概率为 NaN
    try:
        with np.errstate(invalid="raise", over="raise"):
            if alternative == 'two-sided':
                if n1 == n2:
                    prob = _compute_prob_outside_square(n1, h)  # 计算两样本大小相等情况下的概率
                else:
                    prob = _compute_outer_prob_inside_method(n1, n2, g, h)  # 计算其他情况下的概率
            else:
                if n1 == n2:
                    # 使用二项分布计算概率，以避免特殊 binom 函数的舍入误差
                    jrange = np.arange(h)
                    prob = np.prod((n1 - jrange) / (n1 + jrange + 1.0))
                else:
                    with np.errstate(over='raise'):
                        num_paths = _count_paths_outside_method(n1, n2, g, h)  # 计算路径数
                    bin = special.binom(n1 + n2, n1)  # 计算二项式系数
                    if num_paths > bin or np.isinf(bin):
                        saw_fp_error = True  # 如果计算路径数超过二项式系数或者二项式系数为无穷，则标记错误
                    else:
                        prob = num_paths / bin  # 计算概率

    except (FloatingPointError, OverflowError):
        saw_fp_error = True  # 捕捉浮点运算错误或溢出错误

    if saw_fp_error:
        return False, d, np.nan  # 如果出现错误，则返回失败、更新后的 d 和 NaN
    if not (0 <= prob <= 1):
        return False, d, prob  # 如果概率不在 [0, 1] 范围内，则返回失败、更新后的 d 和概率
    return True, d, prob  # 否则返回成功、更新后的 d 和计算得到的概率


@_axis_nan_policy_factory(_tuple_to_KstestResult, n_samples=2, n_outputs=4,
                          result_to_tuple=_KstestResult_to_tuple)
@_rename_parameter("mode", "method")
def ks_2samp(data1, data2, alternative='two-sided', method='auto'):
    """
    Performs the two-sample Kolmogorov-Smirnov test for goodness of fit.

    This test compares the underlying continuous distributions F(x) and G(x)
    of two independent samples.  See Notes for a description of the available
    null and alternative hypotheses.

    Parameters
    ----------
    data1, data2 : array_like, 1-Dimensional
        Two arrays of sample observations assumed to be drawn from a continuous
        distribution, sample sizes can be different.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the null and alternative hypotheses. Default is 'two-sided'.
        Please see explanations in the Notes below.
    method : {'auto', 'exact', 'asymp'}, optional
        Defines the method used for calculating the p-value.
        The following options are available (default is 'auto'):

          * 'auto' : use 'exact' for small size arrays, 'asymp' for large
          * 'exact' : use exact distribution of test statistic
          * 'asymp' : use asymptotic distribution of test statistic

    Returns
    # 定义一个类 KstestResult，用于存储 Kolmogorov-Smirnov 检验的结果
    res: KstestResult
        # 浮点数，KS 检验的统计量
        statistic : float
        # 浮点数，单尾或双尾 p 值
        pvalue : float
        # 浮点数，与 KS 统计量对应的值，来自于 data1 或 data2；
        # 即，在这个观察点测量的经验分布函数之间的距离
        statistic_location : float
        # 整数，如果 data1 的经验分布函数在 statistic_location 处超过
        # data2 的经验分布函数，则为 +1；否则为 -1
        statistic_sign : int

    # 参见相关函数和文档
    See Also
    --------
    kstest, ks_1samp, epps_singleton_2samp, anderson_ksamp

    # 注意事项，关于零假设和对应备择假设有三个选项，可以通过 alternative 参数选择

    - `less`: 零假设是 F(x) >= G(x) 对于所有 x；备择假设是至少有一个 x 满足 F(x) < G(x)；
      统计量是样本经验分布函数之间最小（最负的）差异的大小

    - `greater`: 零假设是 F(x) <= G(x) 对于所有 x；备择假设是至少有一个 x 满足 F(x) > G(x)；
      统计量是样本经验分布函数之间最大（最正的）差异的大小

    - `two-sided`: 零假设是两个分布完全相同，即 F(x)=G(x) 对于所有 x；备择假设是它们不完全相同；
      统计量是样本经验分布函数之间最大绝对差异

    # 注意，备择假设描述的是潜在分布的累积分布函数（CDFs），而不是数据的观察值。
    # 例如，假设 x1 ~ F 和 x2 ~ G。如果对所有 x 都有 F(x) > G(x)，则 x1 中的值 tend to be less than x2 中的值。

    # 如果 KS 统计量较大，则 p 值较小，这可能被视为反对零假设，支持备择假设的证据。

    # 如果 `method='exact'`，则 `ks_2samp` 尝试计算一个精确的 p 值，
    # 即在零假设下获得与从数据计算得到的测试统计值一样极端的概率。
    # 如果 `method='asymp'`，则使用渐近的 Kolmogorov-Smirnov 分布来计算近似 p 值。
    # 如果 `method='auto'`，则尝试精确的 p 值计算，如果两个样本大小都小于 10000；否则使用渐近方法。
    # 无论哪种情况，如果尝试进行精确的 p 值计算失败，将发出警告，并返回渐近 p 值。

    # 'two-sided' 'exact' 计算会计算补充概率，然后从 1 中减去。
    # 因此，它可以返回的最小概率为
    is about 1e-16.  While the algorithm itself is exact, numerical
    errors may accumulate for large sample sizes.   It is most suited to
    situations in which one of the sample sizes is only a few thousand.

    We generally follow Hodges' treatment of Drion/Gnedenko/Korolyuk [1]_.

    References
    ----------
    .. [1] Hodges, J.L. Jr.,  "The Significance Probability of the Smirnov
           Two-Sample Test," Arkiv fiur Matematik, 3, No. 43 (1958), 469-486.

    Examples
    --------
    Suppose we wish to test the null hypothesis that two samples were drawn
    from the same distribution.
    We choose a confidence level of 95%; that is, we will reject the null
    hypothesis in favor of the alternative if the p-value is less than 0.05.

    If the first sample were drawn from a uniform distribution and the second
    were drawn from the standard normal, we would expect the null hypothesis
    to be rejected.

    >>> import numpy as np
    >>> from scipy import stats
    >>> rng = np.random.default_rng()
    >>> sample1 = stats.uniform.rvs(size=100, random_state=rng)
    >>> sample2 = stats.norm.rvs(size=110, random_state=rng)
    >>> stats.ks_2samp(sample1, sample2)
    KstestResult(statistic=0.5454545454545454,
                 pvalue=7.37417839555191e-15,
                 statistic_location=-0.014071496412861274,
                 statistic_sign=-1)


    Indeed, the p-value is lower than our threshold of 0.05, so we reject the
    null hypothesis in favor of the default "two-sided" alternative: the data
    were *not* drawn from the same distribution.

    When both samples are drawn from the same distribution, we expect the data
    to be consistent with the null hypothesis most of the time.

    >>> sample1 = stats.norm.rvs(size=105, random_state=rng)
    >>> sample2 = stats.norm.rvs(size=95, random_state=rng)
    >>> stats.ks_2samp(sample1, sample2)
    KstestResult(statistic=0.10927318295739348,
                 pvalue=0.5438289009927495,
                 statistic_location=-0.1670157701848795,
                 statistic_sign=-1)

    As expected, the p-value of 0.54 is not below our threshold of 0.05, so
    we cannot reject the null hypothesis.

    Suppose, however, that the first sample were drawn from
    a normal distribution shifted toward greater values. In this case,
    the cumulative density function (CDF) of the underlying distribution tends
    to be *less* than the CDF underlying the second sample. Therefore, we would
    expect the null hypothesis to be rejected with ``alternative='less'``:

    >>> sample1 = stats.norm.rvs(size=105, loc=0.5, random_state=rng)
    >>> stats.ks_2samp(sample1, sample2, alternative='less')
    KstestResult(statistic=0.4055137844611529,
                 pvalue=3.5474563068855554e-08,
                 statistic_location=-0.13249370614972575,
                 statistic_sign=-1)

    and indeed, with p-value smaller than our threshold, we reject the null
    hypothesis in favor of the alternative.

    """
    mode = method  # 将输入的方法参数赋值给 mode 变量

    if mode not in ['auto', 'exact', 'asymp']:
        raise ValueError(f'Invalid value for mode: {mode}')  # 如果 mode 不在预定义的模式列表中，则抛出值错误异常

    alternative = {'t': 'two-sided', 'g': 'greater', 'l': 'less'}.get(
        alternative.lower()[0], alternative)  # 根据 alternative 参数选择对应的备择假设描述

    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError(f'Invalid value for alternative: {alternative}')  # 如果 alternative 不在预定义的备择假设列表中，则抛出值错误异常

    MAX_AUTO_N = 10000  # 'auto' 模式下尝试精确计算时的样本大小阈值

    if np.ma.is_masked(data1):
        data1 = data1.compressed()  # 如果 data1 是掩码数组，则压缩掩码
    if np.ma.is_masked(data2):
        data2 = data2.compressed()  # 如果 data2 是掩码数组，则压缩掩码

    data1 = np.sort(data1)  # 对 data1 数组进行排序
    data2 = np.sort(data2)  # 对 data2 数组进行排序

    n1 = data1.shape[0]  # 获取 data1 数组的长度
    n2 = data2.shape[0]  # 获取 data2 数组的长度

    if min(n1, n2) == 0:
        raise ValueError('Data passed to ks_2samp must not be empty')  # 如果 data1 或 data2 中有空数组，则抛出值错误异常

    data_all = np.concatenate([data1, data2])  # 将 data1 和 data2 数组连接起来

    # 使用 searchsorted 解决相等数据的问题，计算累积分布函数
    cdf1 = np.searchsorted(data1, data_all, side='right') / n1
    cdf2 = np.searchsorted(data2, data_all, side='right') / n2

    cddiffs = cdf1 - cdf2  # 计算累积分布函数的差异

    # 找到统计量的位置
    argminS = np.argmin(cddiffs)
    argmaxS = np.argmax(cddiffs)
    loc_minS = data_all[argminS]
    loc_maxS = data_all[argmaxS]

    # 确保 minS 的符号不为负数
    minS = np.clip(-cddiffs[argminS], 0, 1)
    maxS = cddiffs[argmaxS]

    # 根据备择假设选择统计量的值和位置
    if alternative == 'less' or (alternative == 'two-sided' and minS > maxS):
        d = minS
        d_location = loc_minS
        d_sign = -1
    else:
        d = maxS
        d_location = loc_maxS
        d_sign = 1

    g = gcd(n1, n2)  # 计算 n1 和 n2 的最大公约数
    n1g = n1 // g
    n2g = n2 // g

    prob = -np.inf  # 初始化 prob 变量为负无穷

    if mode == 'auto':
        mode = 'exact' if max(n1, n2) <= MAX_AUTO_N else 'asymp'  # 根据样本大小选择模式

    elif mode == 'exact':
        # 如果 lcm(n1, n2) 太大，则从精确模式切换到渐近模式
        if n1g >= np.iinfo(np.int32).max / n2g:
            mode = 'asymp'
            warnings.warn(
                f"Exact ks_2samp calculation not possible with samples sizes "
                f"{n1} and {n2}. Switching to 'asymp'.", RuntimeWarning,
                stacklevel=3)

    if mode == 'exact':
        # 尝试精确计算 ks_2samp
        success, d, prob = _attempt_exact_2kssamp(n1, n2, g, d, alternative)
        if not success:
            mode = 'asymp'
            warnings.warn(f"ks_2samp: Exact calculation unsuccessful. "
                          f"Switching to method={mode}.", RuntimeWarning,
                          stacklevel=3)
    # 如果模式为'asymp'，则使用 Smirnov 的渐近公式计算 Kolmogorov-Smirnov 检验的概率值。
    # 确保将 n1 和 n2 转换为浮点数，以避免在乘法中溢出
    # 对 n1 和 n2 进行排序，因为单侧公式在 n1 和 n2 不对称时不同
    m, n = sorted([float(n1), float(n2)], reverse=True)
    # 计算期望值
    en = m * n / (m + n)
    if alternative == 'two-sided':
        # 对于双侧检验，使用 Kolmogorov-Smirnov 分布的生存函数计算概率值
        prob = distributions.kstwo.sf(d, np.round(en))
    else:
        # 对于单侧检验，根据 Hodges 建议的近似公式 Eqn 5.3 计算概率值
        # 注意这里要求 m 是 (n1, n2) 中较大的一个
        z = np.sqrt(en) * d
        expt = -2 * z**2 - 2 * z * (m + 2*n) / np.sqrt(m * n * (m + n)) / 3.0
        prob = np.exp(expt)

    # 将概率值限制在 [0, 1] 范围内
    prob = np.clip(prob, 0, 1)
    # 目前，`d` 是 Python 的 float 类型。我们希望它是 NumPy 的类型，因此选择 float64 是合适的。
    # 对于更好的改进，应该考虑让 `d` 保持输入的 dtype 类型。
    # 返回 Kolmogorov-Smirnov 检验的结果，包括统计量 d、概率值 prob，以及统计量的位置和符号信息。
    return KstestResult(np.float64(d), prob, statistic_location=d_location,
                        statistic_sign=np.int8(d_sign))
# 定义一个函数用于解析 kstest 的参数，支持多种不同的参数格式。
# 参数包括两个数据集 data1 和 data2，额外的参数 args，以及样本大小 N。
def _parse_kstest_args(data1, data2, args, N):
    # kstest 支持多种不同的参数格式，这里将参数解析提取到单独的函数中
    # (xvals, yvals, )  # 两个样本比较
    # (xvals, cdf 函数,..)
    # (xvals, 分布名称, ...)
    # (分布名称, 分布名称, ...)
    
    # 初始化随机变量生成函数 rvsfunc 和累积分布函数 cdf
    rvsfunc, cdf = None, None
    
    # 如果 data1 是字符串，则通过 getattr 获取 distributions 模块中对应分布的 rvs 函数
    if isinstance(data1, str):
        rvsfunc = getattr(distributions, data1).rvs
    # 如果 data1 是可调用对象，则直接赋值给 rvsfunc
    elif callable(data1):
        rvsfunc = data1
    
    # 如果 data2 是字符串，则通过 getattr 获取 distributions 模块中对应分布的 cdf 函数，并将 data2 置为 None
    if isinstance(data2, str):
        cdf = getattr(distributions, data2).cdf
        data2 = None
    # 如果 data2 是可调用对象，则直接赋值给 cdf，并将 data2 置为 None
    elif callable(data2):
        cdf = data2
        data2 = None
    
    # 使用 rvsfunc 生成 N 个随机样本，并对其排序，如果 rvsfunc 不存在则保持 data1 不变
    data1 = np.sort(rvsfunc(*args, size=N) if rvsfunc else data1)
    
    # 返回解析后的 data1, data2 和 cdf 函数
    return data1, data2, cdf


# 定义一个函数用于确定 kstest 的样本数量是否为 1 或 2
def _kstest_n_samples(kwargs):
    # 获取参数中的 cdf 函数
    cdf = kwargs['cdf']
    # 如果 cdf 是字符串或可调用对象，则返回 1；否则返回 2
    return 1 if (isinstance(cdf, str) or callable(cdf)) else 2


# 定义 kstest 函数的装饰器，用于处理参数解析和结果转换
@_axis_nan_policy_factory(_tuple_to_KstestResult, n_samples=_kstest_n_samples,
                          n_outputs=4, result_to_tuple=_KstestResult_to_tuple)
# 重命名参数 "mode" 为 "method"
@_rename_parameter("mode", "method")
# 主函数 kstest，执行 Kolmogorov-Smirnov 检验，用于拟合优度检验
def kstest(rvs, cdf, args=(), N=20, alternative='two-sided', method='auto'):
    """
    执行 (单样本或双样本) Kolmogorov-Smirnov 拟合优度检验。

    单样本检验比较样本的基础分布 F(x) 与给定分布 G(x)。
    双样本检验比较两个独立样本的基础分布。这两个检验仅对连续分布有效。

    参数
    ----------
    rvs : str, array_like 或 callable
        如果是数组，应为随机变量观测值的 1-D 数组。
        如果是可调用对象，应为生成随机变量的函数；需要有关键字参数 `size`。
        如果是字符串，应为 `scipy.stats` 中分布的名称，将用于生成随机变量。
    cdf : str, array_like 或 callable
        如果是 array_like，则应为随机变量观测值的 1-D 数组，并执行双样本检验
        (rvs 必须是 array_like)。
        如果是可调用对象，则用于计算 cdf。
        如果是字符串，则应为 `scipy.stats` 中分布的名称，将用作 cdf 函数。
    args : tuple, sequence, 可选
        分布参数，当 `rvs` 或 `cdf` 是字符串或可调用对象时使用。
    N : int, 可选
        如果 `rvs` 是字符串或可调用对象，则为样本大小。默认为 20。
    alternative : {'two-sided', 'less', 'greater'}, 可选
        定义零假设和备择假设。默认为 'two-sided'。
        请参阅下面的注释中的解释。

    """
    method : {'auto', 'exact', 'approx', 'asymp'}, optional
        # 定义用于计算 p 值的分布类型。
        # 可选的选项有（默认为 'auto'）：

          * 'auto' : 自动选择其中一种选项。
          * 'exact' : 使用测试统计量的精确分布。
          * 'approx' : 使用两侧概率的两倍来近似双侧概率。
          * 'asymp': 使用测试统计量的渐近分布。

    Returns
    -------
    res: KstestResult
        # 包含以下属性的对象：

        statistic : float
            # KS 检验统计量，可以是 D+、D- 或 D（两者中的最大值）。
        pvalue : float
            # 单侧或双侧 p 值。
        statistic_location : float
            # 在单样本测试中，这是与 KS 统计量对应的 `rvs` 的值；
            # 即，在这个观察值上测量经验分布函数与假设的累积分布函数之间的距离。

            # 在双样本测试中，这是与 KS 统计量对应的 `rvs` 或 `cdf` 的值；
            # 即，在这个观察值上测量两个经验分布函数之间的距离。
        statistic_sign : int
            # 在单样本测试中，如果 KS 统计量是经验分布函数与假设的累积分布函数之间的最大正差异（D+），则为 +1；
            # 如果 KS 统计量是最大负差异（D-），则为 -1。

            # 在双样本测试中，如果 `rvs` 的经验分布函数超过 `cdf` 的经验分布函数在 `statistic_location` 处，则为 +1，否则为 -1。

    See Also
    --------
    ks_1samp, ks_2samp

    Notes
    -----
    # 可以使用 `alternative` 参数选择空假设和相应备择假设的三个选项。

    - `two-sided`: 空假设是两个分布完全相同，F(x)=G(x) 对所有 x；备择假设是它们不完全相同。

    - `less`: 空假设是对于所有 x，F(x) >= G(x)；备择假设是至少存在一个 x，使得 F(x) < G(x)。

    - `greater`: 空假设是对于所有 x，F(x) <= G(x)；备择假设是至少存在一个 x，使得 F(x) > G(x)。

    注意备择假设描述的是潜在分布的累积分布函数（CDFs），而不是观察值。例如，假设 x1 ~ F 和 x2 ~ G。如果对所有 x，F(x) > G(x)，则 x1 中的值 tend to be less than those in x2。

    Examples
    --------
    # 假设我们希望检验一个样本是否符合标准正态分布的空假设。
    # 我们选择 95% 的置信水平；即，我们将拒绝空假设
    # 导入必要的库
    import numpy as np
    from scipy import stats
    
    # 创建一个随机数生成器对象
    rng = np.random.default_rng()
    
    # 对均匀分布的数据进行 Kolmogorov-Smirnov (KS) 检验
    # 预期结果是拒绝零假设，因为 p 值小于 0.05
    stats.kstest(stats.uniform.rvs(size=100, random_state=rng),
                 stats.norm.cdf)
    
    # 由于 p 值小于 0.05，我们拒绝零假设，接受备择假设：
    # 数据不符合标准正态分布
    
    # 对标准正态分布的随机变量进行 KS 检验
    stats.kstest(x, stats.norm.cdf)
    
    # 由于 p 值大于 0.05，我们不能拒绝零假设
    
    # 假设随机变量来自均值向更大值偏移的正态分布，此时
    # 底层分布的累积密度函数 (CDF) 小于标准正态分布的 CDF
    # 因此我们预期使用 alternative='less' 拒绝零假设
    stats.kstest(x, stats.norm.cdf, alternative='less')
    
    # 由于 p 值小于我们设定的阈值，我们拒绝零假设，接受备择假设
    
    # 方便起见，可以使用分布名称作为第二个参数执行前述测试
    stats.kstest(x, "norm", alternative='less')
    
    # 上面的示例都是单样本测试，与 ks_1samp 执行的相同
    # 注意，kstest 也可以执行与 ks_2samp 执行相同的双样本测试
    # 例如，当两个样本来自相同分布时，我们预期数据大部分时间与零假设一致
    sample1 = stats.laplace.rvs(size=105, random_state=rng)
    # 生成一个服从拉普拉斯分布的样本，共计95个样本点，使用给定的随机数生成器rng
    >>> sample2 = stats.laplace.rvs(size=95, random_state=rng)
    
    # 对两个样本sample1和sample2执行 Kolmogorov-Smirnov 检验
    >>> stats.kstest(sample1, sample2)
    
    # KS检验的结果，包括统计量和p值，以及其他相关统计数据
    KstestResult(statistic=0.11779448621553884,
                 pvalue=0.4494256912629795,
                 statistic_location=0.6138814275424155,
                 statistic_sign=1)

    # 根据预期，p值为0.45不低于显著性水平0.05，因此不能拒绝原假设。
    As expected, the p-value of 0.45 is not below our threshold of 0.05, so
    we cannot reject the null hypothesis.

    """
    # 为了不破坏与现有代码的兼容性，如果alternative参数为'two_sided'，则改为'two-sided'
    if alternative == 'two_sided':
        alternative = 'two-sided'
    # 如果alternative不在预期的选项列表['two-sided', 'greater', 'less']中，则抛出异常
    if alternative not in ['two-sided', 'greater', 'less']:
        raise ValueError(f"Unexpected alternative: {alternative}")
    
    # 解析ks检验的参数，得到xvals, yvals和cdf
    xvals, yvals, cdf = _parse_kstest_args(rvs, cdf, args, N)
    
    # 如果提供了cdf，则执行单样本KS检验，返回结果
    if cdf:
        return ks_1samp(xvals, cdf, args=args, alternative=alternative,
                        method=method, _no_deco=True)
    
    # 如果未提供cdf，则执行双样本KS检验，返回结果
    return ks_2samp(xvals, yvals, alternative=alternative, method=method,
                    _no_deco=True)
# 定义命名元组 RanksumsResult 来存储排名和 p 值结果
RanksumsResult = namedtuple('RanksumsResult', ('statistic', 'pvalue'))


# 使用装饰器函数 @_axis_nan_policy_factory 包装 ranksums 函数，指定 n_samples=2
@_axis_nan_policy_factory(RanksumsResult, n_samples=2)
def ranksums(x, y, alternative='two-sided'):
    """Compute the Wilcoxon rank-sum statistic for two samples.

    The Wilcoxon rank-sum test tests the null hypothesis that two sets
    of measurements are drawn from the same distribution.  The alternative
    hypothesis is that values in one sample are more likely to be
    larger than the values in the other sample.

    This test should be used to compare two samples from continuous
    distributions.  It does not handle ties between measurements
    in x and y.  For tie-handling and an optional continuity correction
    see `scipy.stats.mannwhitneyu`.

    Parameters
    ----------
    x,y : array_like
        The data from the two samples.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        The following options are available:

        * 'two-sided': one of the distributions (underlying `x` or `y`) is
          stochastically greater than the other.
        * 'less': the distribution underlying `x` is stochastically less
          than the distribution underlying `y`.
        * 'greater': the distribution underlying `x` is stochastically greater
          than the distribution underlying `y`.

        .. versionadded:: 1.7.0

    Returns
    -------
    statistic : float
        The test statistic under the large-sample approximation that the
        rank sum statistic is normally distributed.
    pvalue : float
        The p-value of the test.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Wilcoxon_rank-sum_test

    Examples
    --------
    """
    We can test the hypothesis that two independent unequal-sized samples are
    drawn from the same distribution with computing the Wilcoxon rank-sum
    statistic.

    >>> import numpy as np
    >>> from scipy.stats import ranksums
    >>> rng = np.random.default_rng()
    >>> sample1 = rng.uniform(-1, 1, 200)
    >>> sample2 = rng.uniform(-0.5, 1.5, 300) # a shifted distribution
    >>> ranksums(sample1, sample2)
    RanksumsResult(statistic=-7.887059,
                   pvalue=3.09390448e-15) # may vary
    >>> ranksums(sample1, sample2, alternative='less')
    RanksumsResult(statistic=-7.750585297581713,
                   pvalue=4.573497606342543e-15) # may vary
    >>> ranksums(sample1, sample2, alternative='greater')
    RanksumsResult(statistic=-7.750585297581713,
                   pvalue=0.9999999999999954) # may vary

    The p-value of less than ``0.05`` indicates that this test rejects the
    hypothesis at the 5% significance level.

    """
    x, y = map(np.asarray, (x, y))  # 将输入的 x 和 y 转换为 NumPy 数组
    n1 = len(x)  # 计算数组 x 的长度，即样本1的大小
    n2 = len(y)  # 计算数组 y 的长度，即样本2的大小
    alldata = np.concatenate((x, y))  # 将样本 x 和 y 连接成一个数组
    ranked = rankdata(alldata)  # 对连接后的数组 alldata 进行排名
    x = ranked[:n1]  # 取前 n1 个元素作为样本 x 的排名
    s = np.sum(x, axis=0)  # 计算样本 x 的排名总和
    expected = n1 * (n1+n2+1) / 2.0  # 计算预期的排名总和
    z = (s - expected) / np.sqrt(n1*n2*(n1+n2+1)/12.0)  # 计算 Wilcoxon 秩和检验统计量
    pvalue = _get_pvalue(z, _SimpleNormal(), alternative, xp=np)  # 调用函数计算 p 值

    return RanksumsResult(z[()], pvalue[()])  # 返回 Wilcoxon 秩和检验结果
# 基于 namedtuple 创建一个名为 KruskalResult 的数据结构，包含 statistic 和 pvalue 两个字段
KruskalResult = namedtuple('KruskalResult', ('statistic', 'pvalue'))

# 使用装饰器函数 @_axis_nan_policy_factory 包装 kruskal 函数，设定 n_samples=None
@_axis_nan_policy_factory(KruskalResult, n_samples=None)
# 定义函数 kruskal，计算独立样本的 Kruskal-Wallis H 检验
def kruskal(*samples, nan_policy='propagate'):
    """Compute the Kruskal-Wallis H-test for independent samples.

    The Kruskal-Wallis H-test tests the null hypothesis that the population
    median of all of the groups are equal.  It is a non-parametric version of
    ANOVA.  The test works on 2 or more independent samples, which may have
    different sizes.  Note that rejecting the null hypothesis does not
    indicate which of the groups differs.  Post hoc comparisons between
    groups are required to determine which groups are different.

    Parameters
    ----------
    sample1, sample2, ... : array_like
       Two or more arrays with the sample measurements can be given as
       arguments. Samples must be one-dimensional.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

          * 'propagate': returns nan
          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values

    Returns
    -------
    statistic : float
       The Kruskal-Wallis H statistic, corrected for ties.
    pvalue : float
       The p-value for the test using the assumption that H has a chi
       square distribution. The p-value returned is the survival function of
       the chi square distribution evaluated at H.

    See Also
    --------
    f_oneway : 1-way ANOVA.
    mannwhitneyu : Mann-Whitney rank test on two samples.
    friedmanchisquare : Friedman test for repeated measurements.

    Notes
    -----
    Due to the assumption that H has a chi square distribution, the number
    of samples in each group must not be too small.  A typical rule is
    that each sample must have at least 5 measurements.

    References
    ----------
    .. [1] W. H. Kruskal & W. W. Wallis, "Use of Ranks in
       One-Criterion Variance Analysis", Journal of the American Statistical
       Association, Vol. 47, Issue 260, pp. 583-621, 1952.
    .. [2] https://en.wikipedia.org/wiki/Kruskal-Wallis_one-way_analysis_of_variance

    Examples
    --------
    >>> from scipy import stats
    >>> x = [1, 3, 5, 7, 9]
    >>> y = [2, 4, 6, 8, 10]
    >>> stats.kruskal(x, y)
    KruskalResult(statistic=0.2727272727272734, pvalue=0.6015081344405895)

    >>> x = [1, 1, 1]
    >>> y = [2, 2, 2]
    >>> z = [2, 2]
    >>> stats.kruskal(x, y, z)
    KruskalResult(statistic=7.0, pvalue=0.0301973834223185)

    """
    # 将输入的 samples 转换为 ndarray 类型的列表
    samples = list(map(np.asarray, samples))

    # 计算样本的个数
    num_groups = len(samples)
    # 如果样本个数小于 2，则抛出 ValueError 异常
    if num_groups < 2:
        raise ValueError("Need at least two groups in stats.kruskal()")

    # 计算每个样本的长度，并转换为 ndarray 类型
    n = np.asarray(list(map(len, samples)))

    # 将所有样本数据合并成一个 ndarray
    alldata = np.concatenate(samples)
    # 对合并后的数据进行排名操作
    ranked = rankdata(alldata)
    # 修正并返回排名中的并列值
    ties = tiecorrect(ranked)
    # 如果所有样本组中的数据都相同，则抛出值错误异常
    if ties == 0:
        raise ValueError('All numbers are identical in kruskal')

    # 计算每组的排名累积和，并在开头插入一个零，以便于索引
    j = np.insert(np.cumsum(n), 0, 0)
    ssbn = 0
    # 遍历每个组，计算其平方和除以组内样本数的结果的总和
    for i in range(num_groups):
        ssbn += _square_of_sums(ranked[j[i]:j[i+1]]) / n[i]

    # 计算总样本数的浮点型总和
    totaln = np.sum(n, dtype=float)
    # 计算 H 统计量的值
    h = 12.0 / (totaln * (totaln + 1)) * ssbn - 3 * (totaln + 1)
    # 计算自由度，即组数减一
    df = num_groups - 1
    # 根据并列的组进行修正 H 统计量
    h /= ties

    # 创建一个简单的卡方对象
    chi2 = _SimpleChi2(df)
    # 获取 H 统计量的 p 值
    pvalue = _get_pvalue(h, chi2, alternative='greater', symmetric=False, xp=np)
    # 返回 Kruskal-Wallis 检验的结果对象
    return KruskalResult(h, pvalue)
# 命名元组，用于存储 Friedman 检验的结果，包括统计量和 p 值
FriedmanchisquareResult = namedtuple('FriedmanchisquareResult',
                                     ('statistic', 'pvalue'))

# 生成带有特定参数的修饰器，用于处理带有 NaN 策略的坐标轴
@_axis_nan_policy_factory(FriedmanchisquareResult, n_samples=None, paired=True)
# 定义 Friedman 检验函数，用于计算重复样本的 Friedman 检验
def friedmanchisquare(*samples):
    """Compute the Friedman test for repeated samples.

    The Friedman test tests the null hypothesis that repeated samples of
    the same individuals have the same distribution.  It is often used
    to test for consistency among samples obtained in different ways.
    For example, if two sampling techniques are used on the same set of
    individuals, the Friedman test can be used to determine if the two
    sampling techniques are consistent.

    Parameters
    ----------
    sample1, sample2, sample3... : array_like
        Arrays of observations.  All of the arrays must have the same number
        of elements.  At least three samples must be given.

    Returns
    -------
    statistic : float
        The test statistic, correcting for ties.
    pvalue : float
        The associated p-value assuming that the test statistic has a chi
        squared distribution.

    Notes
    -----
    Due to the assumption that the test statistic has a chi squared
    distribution, the p-value is only reliable for n > 10 and more than
    6 repeated samples.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Friedman_test
    .. [2] P. Sprent and N.C. Smeeton, "Applied Nonparametric Statistical
           Methods, Third Edition". Chapter 6, Section 6.3.2.

    Examples
    --------
    In [2]_, the pulse rate (per minute) of a group of seven students was
    measured before exercise, immediately after exercise and 5 minutes
    after exercise. Is there evidence to suggest that the pulse rates on
    these three occasions are similar?

    We begin by formulating a null hypothesis :math:`H_0`:

        The pulse rates are identical on these three occasions.

    Let's assess the plausibility of this hypothesis with a Friedman test.

    >>> from scipy.stats import friedmanchisquare
    >>> before = [72, 96, 88, 92, 74, 76, 82]
    >>> immediately_after = [120, 120, 132, 120, 101, 96, 112]
    >>> five_min_after = [76, 95, 104, 96, 84, 72, 76]
    >>> res = friedmanchisquare(before, immediately_after, five_min_after)
    >>> res.statistic
    10.57142857142857
    >>> res.pvalue
    0.005063414171757498

    Using a significance level of 5%, we would reject the null hypothesis in
    favor of the alternative hypothesis: "the pulse rates are different on
    these three occasions".

    """
    # 获取样本数量 k
    k = len(samples)
    # 如果样本数量小于 3，则抛出 ValueError
    if k < 3:
        raise ValueError('At least 3 sets of samples must be given '
                         f'for Friedman test, got {k}.')
    
    # 获取每个样本的观测值数量 n
    n = len(samples[0])
    # 检查所有样本的观测值数量是否一致
    for i in range(1, k):
        if len(samples[i]) != n:
            raise ValueError('Unequal N in friedmanchisquare.  Aborting.')
    
    # 将样本数据按列堆叠成二维数组，并将数据类型转换为 float
    data = np.vstack(samples).T
    data = data.astype(float)
    # 对输入的数据进行排名处理，替换每个子数组中的元素为其排名
    for i in range(len(data)):
        data[i] = rankdata(data[i])
    
    # 处理并计算所有可能的并列情况
    ties = 0
    for d in data:
        # 找到重复值并返回重复值列表和重复次数
        replist, repnum = find_repeats(array(d))
        # 计算并列情况的秩和
        for t in repnum:
            ties += t * (t*t - 1)
    
    # 计算统计量 c
    c = 1 - ties / (k*(k*k - 1)*n)
    
    # 计算 SSBN 统计量
    ssbn = np.sum(data.sum(axis=0)**2)
    
    # 计算 Friedman 检验的统计量
    statistic = (12.0 / (k*n*(k+1)) * ssbn - 3*n*(k+1)) / c
    
    # 初始化卡方分布对象，用于计算 p 值
    chi2 = _SimpleChi2(k - 1)
    
    # 计算 p 值
    pvalue = _get_pvalue(statistic, chi2, alternative='greater', symmetric=False, xp=np)
    
    # 返回 Friedman 检验的结果对象，包括统计量和 p 值
    return FriedmanchisquareResult(statistic, pvalue)
# 创建一个命名元组 BrunnerMunzelResult，包含两个字段 statistic 和 pvalue
BrunnerMunzelResult = namedtuple('BrunnerMunzelResult',
                                 ('statistic', 'pvalue'))

# 使用装饰器函数 @_axis_nan_policy_factory 对 brunnermunzel 函数进行修饰，
# 该修饰器函数返回一个具有统计量和 p 值字段的命名元组 BrunnerMunzelResult，要求样本数量为 2
@_axis_nan_policy_factory(BrunnerMunzelResult, n_samples=2)
# 定义 brunnermunzel 函数，计算样本 x 和 y 的 Brunner-Munzel 检验
def brunnermunzel(x, y, alternative="two-sided", distribution="t",
                  nan_policy='propagate'):
    """Compute the Brunner-Munzel test on samples x and y.

    The Brunner-Munzel test is a nonparametric test of the null hypothesis that
    when values are taken one by one from each group, the probabilities of
    getting large values in both groups are equal.
    Unlike the Wilcoxon-Mann-Whitney's U test, this does not require the
    assumption of equivariance of two groups. Note that this does not assume
    the distributions are same. This test works on two independent samples,
    which may have different sizes.

    Parameters
    ----------
    x, y : array_like
        Array of samples, should be one-dimensional.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

          * 'two-sided'
          * 'less': one-sided
          * 'greater': one-sided
    distribution : {'t', 'normal'}, optional
        Defines how to get the p-value.
        The following options are available (default is 't'):

          * 't': get the p-value by t-distribution
          * 'normal': get the p-value by standard normal distribution.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

          * 'propagate': returns nan
          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values

    Returns
    -------
    statistic : float
        The Brunner-Munzer W statistic.
    pvalue : float
        p-value assuming an t distribution. One-sided or
        two-sided, depending on the choice of `alternative` and `distribution`.

    See Also
    --------
    mannwhitneyu : Mann-Whitney rank test on two samples.

    Notes
    -----
    Brunner and Munzel recommended to estimate the p-value by t-distribution
    when the size of data is 50 or less. If the size is lower than 10, it would
    be better to use permuted Brunner Munzel test (see [2]_).

    References
    ----------
    .. [1] Brunner, E. and Munzel, U. "The nonparametric Benhrens-Fisher
           problem: Asymptotic theory and a small-sample approximation".
           Biometrical Journal. Vol. 42(2000): 17-25.
    .. [2] Neubert, K. and Brunner, E. "A studentized permutation test for the
           non-parametric Behrens-Fisher problem". Computational Statistics and
           Data Analysis. Vol. 51(2007): 5192-5204.

    Examples
    --------
    >>> from scipy import stats
    >>> x1 = [1,2,1,1,1,1,1,1,1,1,2,4,1,1]
    >>> x2 = [3,3,4,3,1,2,3,1,1,5,4]
    >>> w, p_value = stats.brunnermunzel(x1, x2)
    >>> w
    """
    3.1374674823029505
    >>> p_value
    0.0057862086661515377

    """
    计算样本 x 和 y 的长度
    nx = len(x)
    ny = len(y)

    将 x 和 y 合并后，计算各元素的秩
    rankc = rankdata(np.concatenate((x, y)))
    从合并后的秩数组中提取出 x 的秩
    rankcx = rankc[0:nx]
    从合并后的秩数组中提取出 y 的秩
    rankcy = rankc[nx:nx+ny]
    计算 rankcx 和 rankcy 的均值
    rankcx_mean = np.mean(rankcx)
    rankcy_mean = np.mean(rankcy)
    分别计算 x 和 y 的秩
    rankx = rankdata(x)
    ranky = rankdata(y)
    计算 rankx 和 ranky 的均值
    rankx_mean = np.mean(rankx)
    ranky_mean = np.mean(ranky)

    计算 Sx 和 Sy 统计量
    Sx = np.sum(np.power(rankcx - rankx - rankcx_mean + rankx_mean, 2.0))
    Sx /= nx - 1
    Sy = np.sum(np.power(rankcy - ranky - rankcy_mean + ranky_mean, 2.0))
    Sy /= ny - 1

    计算 wbfn 统计量
    wbfn = nx * ny * (rankcy_mean - rankcx_mean)
    wbfn /= (nx + ny) * np.sqrt(nx * Sx + ny * Sy)

    根据指定的分布类型创建对应的分布对象
    if distribution == "t":
        计算自由度相关参数
        df_numer = np.power(nx * Sx + ny * Sy, 2.0)
        df_denom = np.power(nx * Sx, 2.0) / (nx - 1)
        df_denom += np.power(ny * Sy, 2.0) / (ny - 1)
        df = df_numer / df_denom

        如果自由度参数无法计算，发出警告并尝试使用正态分布
        if (df_numer == 0) and (df_denom == 0):
            message = ("p-value cannot be estimated with `distribution='t' "
                       "because degrees of freedom parameter is undefined "
                       "(0/0). Try using `distribution='normal'")
            warnings.warn(message, RuntimeWarning, stacklevel=2)

        创建 t 分布对象
        distribution = _SimpleStudentT(df)
    elif distribution == "normal":
        创建正态分布对象
        distribution = _SimpleNormal()
    else:
        如果分布类型不是 't' 或 'normal'，则抛出值错误
        raise ValueError(
            "distribution should be 't' or 'normal'")

    根据计算的统计量和分布对象获取 p 值
    p = _get_pvalue(-wbfn, distribution, alternative, xp=np)

    返回 Brunner-Munzel 检验的结果对象
    return BrunnerMunzelResult(wbfn, p)
# 定义装饰器函数，用于生成带有特定轴的 NaN 策略的工厂函数，应用于 SignificanceResult 类
@_axis_nan_policy_factory(SignificanceResult, kwd_samples=['weights'], paired=True)
# 合并多个独立测试的 p 值，这些测试都涉及同一个假设
def combine_pvalues(pvalues, method='fisher', weights=None, *, axis=0):
    """
    Combine p-values from independent tests that bear upon the same hypothesis.

    These methods are intended only for combining p-values from hypothesis
    tests based upon continuous distributions.

    Each method assumes that under the null hypothesis, the p-values are
    sampled independently and uniformly from the interval [0, 1]. A test
    statistic (different for each method) is computed and a combined
    p-value is calculated based upon the distribution of this test statistic
    under the null hypothesis.

    Parameters
    ----------
    pvalues : array_like
        Array of p-values assumed to come from independent tests based on
        continuous distributions.
    method : {'fisher', 'pearson', 'tippett', 'stouffer', 'mudholkar_george'}

        Name of method to use to combine p-values.

        The available methods are (see Notes for details):

        * 'fisher': Fisher's method (Fisher's combined probability test)
        * 'pearson': Pearson's method
        * 'mudholkar_george': Mudholkar's and George's method
        * 'tippett': Tippett's method
        * 'stouffer': Stouffer's Z-score method
    weights : array_like, optional
        Optional array of weights used only for Stouffer's Z-score method.
        Ignored by other methods.

    Returns
    -------
    res : SignificanceResult
        An object containing attributes:

        statistic : float
            The statistic calculated by the specified method.
        pvalue : float
            The combined p-value.
    """
    # 省略函数主体部分，具体实现与各种方法有关，返回一个 SignificanceResult 对象
    # 将输入的 p-values 转换为数组的命名空间
    xp = array_namespace(pvalues)
    # 将 p-values 转换为数组，并使用 xp 来表示
    pvalues = xp.asarray(pvalues)
    if xp_size(pvalues) == 0:
        # 只用于测试 _axis_nan_policy 装饰器
        # 在使用装饰器时不会发生这种情况。
        # 获取 NaN 值
        NaN = _get_nan(pvalues)
        # 返回一个具有 NaN 值的 SignificanceResult 对象
        return SignificanceResult(NaN, NaN)

    n = pvalues.shape[axis]
    # 用于将 Python 标量转换为正确的 dtype
    one = xp.asarray(1, dtype=pvalues.dtype)

    if method == 'fisher':
        # 计算统计量为 -2 * 对数似然比和
        statistic = -2 * xp.sum(xp.log(pvalues), axis=axis)
        # 创建一个简单的卡方分布对象
        chi2 = _SimpleChi2(2*n*one)
        # 获取统计量的 p 值
        pval = _get_pvalue(statistic, chi2, alternative='greater',
                           symmetric=False, xp=xp)
    elif method == 'pearson':
        # 计算统计量为 2 * 对数似然比和
        statistic = 2 * xp.sum(xp.log1p(-pvalues), axis=axis)
        # 创建一个简单的卡方分布对象
        chi2 = _SimpleChi2(2*n*one)
        # 获取统计量的 p 值
        pval = _get_pvalue(-statistic, chi2, alternative='less', symmetric=False, xp=xp)
    elif method == 'mudholkar_george':
        # 计算归一化因子
        normalizing_factor = math.sqrt(3/n)/xp.pi
        # 计算统计量为 -log(p) + log(1-p) 和
        statistic = (-xp.sum(xp.log(pvalues), axis=axis)
                     + xp.sum(xp.log1p(-pvalues), axis=axis))
        # 计算自由度 nu
        nu = 5*n  + 4
        # 计算近似因子
        approx_factor = math.sqrt(nu / (nu - 2))
        # 创建一个简单的学生 t 分布对象
        t = _SimpleStudentT(nu*one)
        # 获取统计量的 p 值
        pval = _get_pvalue(statistic * normalizing_factor * approx_factor, t,
                           alternative="greater", xp=xp)
    elif method == 'tippett':
        # 计算统计量为 p 值的最小值
        statistic = xp.min(pvalues, axis=axis)
        # 创建一个简单的贝塔分布对象
        beta = _SimpleBeta(one, n*one)
        # 获取统计量的 p 值
        pval = _get_pvalue(statistic, beta, alternative='less', symmetric=False, xp=xp)
    elif method == 'stouffer':
        # 如果权重 weights 未提供，则使用全为 1 的权重
        if weights is None:
            weights = xp.ones_like(pvalues, dtype=pvalues.dtype)
        # 检查权重长度与 pvalues 长度是否一致
        elif weights.shape[axis] != n:
            raise ValueError("pvalues and weights must be of the same "
                             "length along `axis`.")
        
        # 创建一个简单的正态分布对象
        norm = _SimpleNormal()
        # 计算 Zi 值
        Zi = norm.isf(pvalues)
        # 计算统计量
        statistic = weights @ Zi / xp.linalg.vector_norm(weights, axis=axis)
        # 获取统计量的 p 值
        pval = _get_pvalue(statistic, norm, alternative="greater", xp=xp)

    else:
        # 抛出异常，说明方法名 method 无效
        raise ValueError(
            f"Invalid method {method!r}. Valid methods are 'fisher', "
            "'pearson', 'mudholkar_george', 'tippett', and 'stouffer'"
        )

    # 返回 SignificanceResult 对象，其中包含统计量和 p 值
    return SignificanceResult(statistic, pval)
# 定义一个数据类 QuantileTestResult，用于表示 scipy.stats.quantile_test 的结果
@dataclass
class QuantileTestResult:
    r"""
    Result of `scipy.stats.quantile_test`.

    Attributes
    ----------
    statistic: float
        The statistic used to calculate the p-value; either ``T1``, the
        number of observations less than or equal to the hypothesized quantile,
        or ``T2``, the number of observations strictly less than the
        hypothesized quantile. Two test statistics are required to handle the
        possibility the data was generated from a discrete or mixed
        distribution.

    statistic_type : int
        ``1`` or ``2`` depending on which of ``T1`` or ``T2`` was used to
        calculate the p-value respectively. ``T1`` corresponds to the
        ``"greater"`` alternative hypothesis and ``T2`` to the ``"less"``.  For
        the ``"two-sided"`` case, the statistic type that leads to smallest
        p-value is used.  For significant tests, ``statistic_type = 1`` means
        there is evidence that the population quantile is significantly greater
        than the hypothesized value and ``statistic_type = 2`` means there is
        evidence that it is significantly less than the hypothesized value.

    pvalue : float
        The p-value of the hypothesis test.

    _alternative : list[str] = field(repr=False)
        Private attribute representing alternative hypotheses considered.

    _x : np.ndarray = field(repr=False)
        Private attribute representing the data sample.

    _p : float = field(repr=False)
        Private attribute representing a specific parameter value.
    """
    statistic: float
    statistic_type: int
    pvalue: float
    _alternative: list[str] = field(repr=False)
    _x : np.ndarray = field(repr=False)
    _p : float = field(repr=False)
    def confidence_interval(self, confidence_level=0.95):
        """
        Compute the confidence interval of the quantile.

        Parameters
        ----------
        confidence_level : float, default: 0.95
            Confidence level for the computed confidence interval
            of the quantile. Default is 0.95.

        Returns
        -------
        ci : ``ConfidenceInterval`` object
            The object has attributes ``low`` and ``high`` that hold the
            lower and upper bounds of the confidence interval.

        Examples
        --------
        >>> import numpy as np
        >>> import scipy.stats as stats
        >>> p = 0.75  # quantile of interest
        >>> q = 0  # hypothesized value of the quantile
        >>> x = np.exp(np.arange(0, 1.01, 0.01))
        >>> res = stats.quantile_test(x, q=q, p=p, alternative='less')
        >>> lb, ub = res.confidence_interval()
        >>> lb, ub
        (-inf, 2.293318740264183)
        >>> res = stats.quantile_test(x, q=q, p=p, alternative='two-sided')
        >>> lb, ub = res.confidence_interval(0.9)
        >>> lb, ub
        (1.9542373206359396, 2.293318740264183)
        """

        # 获取备选假设类型
        alternative = self._alternative
        # 获取所设定的分位数
        p = self._p
        # 对数据进行排序
        x = np.sort(self._x)
        # 样本量
        n = len(x)
        # 使用二项分布进行计算
        bd = stats.binom(n, p)

        # 检查置信水平是否在有效范围内
        if confidence_level <= 0 or confidence_level >= 1:
            message = "`confidence_level` must be a number between 0 and 1."
            raise ValueError(message)

        # 初始化下界和上界的索引
        low_index = np.nan
        high_index = np.nan

        # 根据备选假设类型计算置信区间的下界和上界
        if alternative == 'less':
            # 单尾检验，计算下界
            p = 1 - confidence_level
            low = -np.inf
            high_index = int(bd.isf(p))
            high = x[high_index] if high_index < n else np.nan
        elif alternative == 'greater':
            # 单尾检验，计算上界
            p = 1 - confidence_level
            low_index = int(bd.ppf(p)) - 1
            low = x[low_index] if low_index >= 0 else np.nan
            high = np.inf
        elif alternative == 'two-sided':
            # 双尾检验，计算下界和上界
            p = (1 - confidence_level) / 2
            low_index = int(bd.ppf(p)) - 1
            low = x[low_index] if low_index >= 0 else np.nan
            high_index = int(bd.isf(p))
            high = x[high_index] if high_index < n else np.nan

        # 返回置信区间对象
        return ConfidenceInterval(low, high)
def quantile_test_iv(x, q, p, alternative):
    # 将输入的 `x` 转换为至少是一维的数组
    x = np.atleast_1d(x)
    # 检查 `x` 是否为一维数组且元素为数值类型，否则抛出错误
    message = '`x` must be a one-dimensional array of numbers.'
    if x.ndim != 1 or not np.issubdtype(x.dtype, np.number):
        raise ValueError(message)

    # 将 `q` 转换为标量（scalar）
    q = np.array(q)[()]
    # 检查 `q` 是否为标量，否则抛出错误
    message = "`q` must be a scalar."
    if q.ndim != 0 or not np.issubdtype(q.dtype, np.number):
        raise ValueError(message)

    # 将 `p` 转换为标量（scalar）
    p = np.array(p)[()]
    # 检查 `p` 是否为介于 0 到 1 之间的浮点数，否则抛出错误
    message = "`p` must be a float strictly between 0 and 1."
    if p.ndim != 0 or p >= 1 or p <= 0:
        raise ValueError(message)

    # 可接受的 `alternative` 值集合
    alternatives = {'two-sided', 'less', 'greater'}
    # 检查 `alternative` 是否在可接受的值集合中，否则抛出错误
    message = f"`alternative` must be one of {alternatives}"
    if alternative not in alternatives:
        raise ValueError(message)

    # 返回验证后的参数
    return x, q, p, alternative


def quantile_test(x, *, q=0, p=0.5, alternative='two-sided'):
    r"""
    Perform a quantile test and compute a confidence interval of the quantile.

    This function tests the null hypothesis that `q` is the value of the
    quantile associated with probability `p` of the population underlying
    sample `x`. For example, with default parameters, it tests that the
    median of the population underlying `x` is zero. The function returns an
    object including the test statistic, a p-value, and a method for computing
    the confidence interval around the quantile.

    Parameters
    ----------
    x : array_like
        A one-dimensional sample.
    q : float, default: 0
        The hypothesized value of the quantile.
    p : float, default: 0.5
        The probability associated with the quantile; i.e. the proportion of
        the population less than `q` is `p`. Must be strictly between 0 and
        1.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

        * 'two-sided': the quantile associated with the probability `p`
          is not `q`.
        * 'less': the quantile associated with the probability `p` is less
          than `q`.
        * 'greater': the quantile associated with the probability `p` is
          greater than `q`.

    Returns
    -------
    """
    result : QuantileTestResult
        # 定义一个变量 `result`，类型为 `QuantileTestResult`，表示分位数测试结果对象
    
        An object with the following attributes:
    
        statistic : float
            # 测试统计量之一，用于分位数测试的两个可能统计量之一。
            # 第一个统计量 `T1` 是样本中小于等于假设分位数 `q` 的比例。
            # 第二个统计量 `T2` 是样本中严格小于假设分位数 `q` 的比例。
    
            When ``alternative = 'greater'``, ``T1`` 用于计算 p 值，且 `statistic` 被设置为 `T1`。
    
            When ``alternative = 'less'``, ``T2`` 用于计算 p 值，且 `statistic` 被设置为 `T2`。
    
            When ``alternative = 'two-sided'``, 同时考虑 `T1` 和 `T2`，选择 p 值最小的统计量。
    
        statistic_type : int
            # 根据使用的是 `T1` 还是 `T2` 来设置为 `1` 或 `2`。
    
        pvalue : float
            # 给定备择假设的 p 值。
    
        The object also has the following method:
    
        confidence_interval(confidence_level=0.95)
            # 计算与概率 `p` 关联的总体分位数的置信区间。
            # 置信区间以 `namedtuple` 的形式返回，包含 `low` 和 `high` 两个字段。
            # 当没有足够的观测来计算所需置信度的置信区间时，值为 `nan`。
    
    Notes
    -----
    This test and its method for computing confidence intervals are
    non-parametric. They are valid if and only if the observations are i.i.d.
    
    The implementation of the test follows Conover [1]_. Two test statistics
    are considered.
    
    ``T1``: The number of observations in `x` less than or equal to `q`.
    
        ``T1 = (x <= q).sum()``
    
    ``T2``: The number of observations in `x` strictly less than `q`.
    
        ``T2 = (x < q).sum()``
    
    The use of two test statistics is necessary to handle the possibility that
    `x` was generated from a discrete or mixed distribution.
    
    The null hypothesis for the test is:
    
        H0: The :math:`p^{\mathrm{th}}` population quantile is `q`.
    
    and the null distribution for each test statistic is
    :math:`\mathrm{binom}\left(n, p\right)`. When ``alternative='less'``,
    the alternative hypothesis is:
    
        H1: The :math:`p^{\mathrm{th}}` population quantile is less than `q`.
    
    and the p-value is the probability that the binomial random variable
    
    .. math::
        Y \sim \mathrm{binom}\left(n, p\right)
    
    is greater than or equal to the observed value ``T2``.
    
    When ``alternative='greater'``, the alternative hypothesis is:
    
        H1: The :math:`p^{\mathrm{th}}` population quantile is greater than `q`
    and the p-value is the probability that the binomial random variable Y
    is less than or equal to the observed value ``T1``.

    当 ``alternative='two-sided'`` 时，备择假设为

        H1: `q` 不是 :math:`p^{\mathrm{th}}` 总体分位数。

    and the p-value is twice the smaller of the p-values for the ``'less'``
    and ``'greater'`` cases. Both of these p-values can exceed 0.5 for the same
    data, so the value is clipped into the interval :math:`[0, 1]`.

    当备择假设为 'two-sided' 时，p 值是 ``'less'`` 和 ``'greater'`` 情况下较小的 p 值的两倍。这两个 p 值对于同一数据可以都超过 0.5，因此该值被剪裁到区间 :math:`[0, 1]` 内。

    The approach for confidence intervals is attributed to Thompson [2]_ and
    later proven to be applicable to any set of i.i.d. samples [3]_. The
    computation is based on the observation that the probability of a quantile
    :math:`q` to be larger than any observations :math:`x_m (1\leq m \leq N)`
    can be computed as

    信心区间的方法归因于 Thompson [2]_，后来证明适用于任何一组 i.i.d. 样本 [3]_。计算基于以下观察：分位数 :math:`q` 大于任何观测值 :math:`x_m (1\leq m \leq N)` 的概率可以计算为

    .. math::

        \mathbb{P}(x_m \leq q) = 1 - \sum_{k=0}^{m-1} \binom{N}{k}
        q^k(1-q)^{N-k}

    By default, confidence intervals are computed for a 95% confidence level.
    A common interpretation of a 95% confidence intervals is that if i.i.d.
    samples are drawn repeatedly from the same population and confidence
    intervals are formed each time, the confidence interval will contain the
    true value of the specified quantile in approximately 95% of trials.

    默认情况下，信心区间是基于 95% 置信水平计算的。
    95% 置信区间的一般解释是，如果从同一总体中重复抽取 i.i.d. 样本并且每次形成信心区间，那么在约 95% 的试验中，置信区间将包含指定分位数的真实值。

    A similar function is available in the QuantileNPCI R package [4]_. The
    foundation is the same, but it computes the confidence interval bounds by
    doing interpolations between the sample values, whereas this function uses
    only sample values as bounds. Thus, ``quantile_test.confidence_interval``
    returns more conservative intervals (i.e., larger).

    QuantileNPCI R 包中有一个类似的函数 [4]_。基础相同，但它通过样本值之间的插值计算置信区间界限，而这个函数只使用样本值作为边界。因此，``quantile_test.confidence_interval`` 返回更保守的区间（即更大的区间）。

    The same computation of confidence intervals for quantiles is included in
    the confintr package [5]_.

    quantile 的置信区间的计算也包含在 confintr 包中 [5]_。

    Two-sided confidence intervals are not guaranteed to be optimal; i.e.,
    there may exist a tighter interval that may contain the quantile of
    interest with probability larger than the confidence level.
    Without further assumption on the samples (e.g., the nature of the
    underlying distribution), the one-sided intervals are optimally tight.

    双侧置信区间不能保证是最优的；即可能存在一个更紧的区间，可以以高于置信水平的概率包含感兴趣的分位数。
    在没有对样本进一步假设的情况下（例如，底层分布的性质），单侧区间是最优紧的。

    References
    ----------
    .. [1] W. J. Conover. Practical Nonparametric Statistics, 3rd Ed. 1999.
    .. [2] W. R. Thompson, "On Confidence Ranges for the Median and Other
       Expectation Distributions for Populations of Unknown Distribution
       Form," The Annals of Mathematical Statistics, vol. 7, no. 3,
       pp. 122-128, 1936, Accessed: Sep. 18, 2019. [Online]. Available:
       https://www.jstor.org/stable/2957563.
    .. [3] H. A. David and H. N. Nagaraja, "Order Statistics in Nonparametric
       Inference" in Order Statistics, John Wiley & Sons, Ltd, 2005, pp.
       159-170. Available:
       https://onlinelibrary.wiley.com/doi/10.1002/0471722162.ch7.
    Examples
    --------
    
    # 导入必要的库
    >>> import numpy as np
    >>> from scipy import stats
    # 使用指定种子创建随机数生成器对象
    >>> rng = np.random.default_rng(6981396440634228121)
    # 从标准均匀分布中生成100个随机变量
    >>> rvs = stats.uniform.rvs(size=100, random_state=rng)
    # 进行分位数检验，测试中位数是否等于0.5，置信水平为0.5
    >>> stats.quantile_test(rvs, q=0.5, p=0.5)
    QuantileTestResult(statistic=45, statistic_type=1, pvalue=0.36820161732669576)
    
    # 由于计算得到的 p 值不小于设定的阈值0.01，因此无法拒绝原假设。
    
    # 当从标准正态分布中生成数据时，其中位数为0，我们预计将拒绝原假设。
    >>> rvs = stats.norm.rvs(size=100, random_state=rng)
    # 进行分位数检验，测试中位数是否等于0.5，置信水平为0.5
    >>> stats.quantile_test(rvs, q=0.5, p=0.5)
    QuantileTestResult(statistic=67, statistic_type=2, pvalue=0.0008737198369123724)
    
    # 由于计算得到的 p 值小于设定的阈值0.01，因此我们拒绝原假设，
    # 支持默认的“双侧”备择假设：总体中位数不等于0.5。
    
    # 然而，如果我们将原假设与单侧备择假设比较，即总体中位数大于0.5，
    # 由于标准正态分布的中位数小于0.5，我们不会拒绝原假设。
    >>> stats.quantile_test(rvs, q=0.5, p=0.5, alternative='greater')
    QuantileTestResult(statistic=67, statistic_type=1, pvalue=0.9997956114162866)
    
    # 不出意料，由于计算得到的 p 值大于设定的阈值，我们不拒绝原假设，
    # 支持所选备择假设。
    
    # 分位数检验可用于任何分位数，不仅仅是中位数。例如，我们可以测试样本
    # 所在分布的第三四分位数是否大于0.6。
    >>> rvs = stats.uniform.rvs(size=100, random_state=rng)
    # 进行分位数检验，测试第三四分位数是否大于0.6，置信水平为0.75
    >>> stats.quantile_test(rvs, q=0.6, p=0.75, alternative='greater')
    QuantileTestResult(statistic=64, statistic_type=1, pvalue=0.00940696592998271)
    
    # 计算得到的 p 值小于设定的阈值，因此我们拒绝原假设，
    # 支持备择假设：样本所在分布的第三四分位数大于0.6。
    
    # `quantile_test` 还可以计算任何分位数的置信区间。
    # Implementation carefully follows [1] 3.2
    # "H0: the p*th quantile of X is x*"
    # To facilitate comparison with [1], we'll use variable names that
    # best match Conover's notation
    X, x_star, p_star, H1 = quantile_test_iv(x, q, p, alternative)
    # 调用 quantile_test_iv 函数，返回测试结果的关键变量：样本 X、待测的分位数 x_star、分位数 p_star、备择假设 H1

    # "We will use two test statistics in this test. Let T1 equal "
    # "the number of observations less than or equal to x*, and "
    # "let T2 equal the number of observations less than x*."
    T1 = (X <= x_star).sum()
    # 计算样本 X 中小于等于 x_star 的观测数，作为统计量 T1
    T2 = (X < x_star).sum()
    # 计算样本 X 中严格小于 x_star 的观测数，作为统计量 T2

    # "The null distribution of the test statistics T1 and T2 is "
    # "the binomial distribution, with parameters n = sample size, and "
    # "p = p* as given in the null hypothesis.... Y has the binomial "
    # "distribution with parameters n and p*."
    n = len(X)
    # 样本大小 n
    Y = stats.binom(n=n, p=p_star)
    # 根据零假设，Y 服从参数为 n 和 p* 的二项分布

    # "H1: the p* population quantile is less than x*"
    if H1 == 'less':
        # "The p-value is the probability that a binomial random variable Y "
        # "is greater than *or equal to* the observed value of T2...using p=p*"
        pvalue = Y.sf(T2-1)  # Y.pmf(T2) + Y.sf(T2)
        # 计算 P(Y >= T2) 的概率，即 p-value
        statistic = T2
        # 统计量为 T2
        statistic_type = 2
        # 统计类型为 2，表示 T2 的统计量类型
    # 如果备择假设 H1 表示 "大于 x* 的 p* 分位数"
    elif H1 == 'greater':
        # 计算 p 值，即二项分布随机变量 Y 小于或等于 T1 的概率，其中 p = p*
        pvalue = Y.cdf(T1)
        # 统计量为 T1
        statistic = T1
        # 统计量类型为 1
        statistic_type = 1
    # 如果备择假设 H1 表示 "x* 不是 p* 分位数"
    elif H1 == 'two-sided':
        # 计算两个单侧 p 值：Y 小于或等于 T1 的概率和 Y 大于或等于 T2 的概率，其中 p = p*
        pvalues = [Y.cdf(T1), Y.sf(T2 - 1)]  # 分别对应 [大于, 小于]
        # 按照 p 值大小排序索引
        sorted_idx = np.argsort(pvalues)
        # 计算双侧 p 值，确保在 [0, 1] 范围内
        pvalue = np.clip(2 * pvalues[sorted_idx[0]], 0, 1)
        # 根据较小 p 值的索引确定统计量和统计量类型
        if sorted_idx[0]:
            statistic, statistic_type = T2, 2
        else:
            statistic, statistic_type = T1, 1

    # 返回分位数检验的结果对象
    return QuantileTestResult(
        statistic=statistic,
        statistic_type=statistic_type,
        pvalue=pvalue,
        _alternative=H1,
        _x=X,
        _p=p_star
    )
#####################################
#       STATISTICAL DISTANCES       #
#####################################


# 计算两个 N 维离散分布之间的 Wasserstein-1 距离
def wasserstein_distance_nd(u_values, v_values, u_weights=None, v_weights=None):
    r"""
    Compute the Wasserstein-1 distance between two N-D discrete distributions.

    The Wasserstein distance, also called the Earth mover's distance or the
    optimal transport distance, is a similarity metric between two probability
    distributions [1]_. In the discrete case, the Wasserstein distance can be
    understood as the cost of an optimal transport plan to convert one
    distribution into the other. The cost is calculated as the product of the
    amount of probability mass being moved and the distance it is being moved.
    A brief and intuitive introduction can be found at [2]_.

    .. versionadded:: 1.13.0

    Parameters
    ----------
    u_values : 2d array_like
        A sample from a probability distribution or the support (set of all
        possible values) of a probability distribution. Each element along
        axis 0 is an observation or possible value, and axis 1 represents the
        dimensionality of the distribution; i.e., each row is a vector
        observation or possible value.

    v_values : 2d array_like
        A sample from or the support of a second distribution.

    u_weights, v_weights : 1d array_like, optional
        Weights or counts corresponding with the sample or probability masses
        corresponding with the support values. Sum of elements must be positive
        and finite. If unspecified, each value is assigned the same weight.

    Returns
    -------
    distance : float
        The computed distance between the distributions.

    Notes
    -----
    Given two probability mass functions, :math:`u`
    and :math:`v`, the first Wasserstein distance between the distributions
    using the Euclidean norm is:

    .. math::

        l_1 (u, v) = \inf_{\pi \in \Gamma (u, v)} \int \| x-y \|_2 \mathrm{d} \pi (x, y)

    where :math:`\Gamma (u, v)` is the set of (probability) distributions on
    :math:`\mathbb{R}^n \times \mathbb{R}^n` whose marginals are :math:`u` and
    :math:`v` on the first and second factors respectively. For a given value
    :math:`x`, :math:`u(x)` gives the probabilty of :math:`u` at position
    :math:`x`, and the same for :math:`v(x)`.

    This is also called the optimal transport problem or the Monge problem.
    Let the finite point sets :math:`\{x_i\}` and :math:`\{y_j\}` denote
    the support set of probability mass function :math:`u` and :math:`v`
    respectively. The Monge problem can be expressed as follows,

    Let :math:`\Gamma` denote the transport plan, :math:`D` denote the
    distance matrix and,

    .. math::

        x = \text{vec}(\Gamma)          \\
        c = \text{vec}(D)               \\
        b = \begin{bmatrix}
                u\\
                v\\
            \end{bmatrix}

"""
    # The :math:`\text{vec}()` function denotes the Vectorization function
    # that transforms a matrix into a column vector by vertically stacking
    # the columns of the matrix.
    # The tranport plan :math:`\Gamma` is a matrix :math:`[\gamma_{ij}]` in
    # which :math:`\gamma_{ij}` is a positive value representing the amount of
    # probability mass transported from :math:`u(x_i)` to :math:`v(y_i)`.
    # Summing over the rows of :math:`\Gamma` should give the source distribution
    # :math:`u` : :math:`\sum_j \gamma_{ij} = u(x_i)` holds for all :math:`i`
    # and summing over the columns of :math:`\Gamma` should give the target
    # distribution :math:`v`: :math:`\sum_i \gamma_{ij} = v(y_j)` holds for all
    # :math:`j`.
    # The distance matrix :math:`D` is a matrix :math:`[d_{ij}]`, in which
    # :math:`d_{ij} = d(x_i, y_j)`.
    
    # Given :math:`\Gamma`, :math:`D`, :math:`b`, the Monge problem can be
    # transformed into a linear programming problem by
    # taking :math:`A x = b` as constraints and :math:`z = c^T x` as minimization
    # target (sum of costs), where matrix :math:`A` has the form
    # The matrix A is structured as shown in the LaTeX block, representing
    # constraints for the linear programming problem formulation.
    
    # By solving the dual form of the above linear programming problem (with
    # solution :math:`y^*`), the Wasserstein distance :math:`l_1 (u, v)` can
    # be computed as :math:`b^T y^*`.
    
    # The above solution is inspired by Vincent Herrmann's blog [3]_ . For a
    # more thorough explanation, see [4]_ .
    
    # The input distributions can be empirical, therefore coming from samples
    # whose values are effectively inputs of the function, or they can be seen as
    # generalized functions, in which case they are weighted sums of Dirac delta
    # functions located at the specified values.
    
    # References
    # ----------
    # .. [1] "Wasserstein metric",
    #        https://en.wikipedia.org/wiki/Wasserstein_metric
    # .. [2] Lili Weng, "What is Wasserstein distance?", Lil'log,
    #        https://lilianweng.github.io/posts/2017-08-20-gan/#what-is-wasserstein-distance.
    m, n = len(u_values), len(v_values)
    # 获取输入数组 u_values 和 v_values 的长度，分别赋值给 m 和 n

    u_values = asarray(u_values)
    v_values = asarray(v_values)
    # 将 u_values 和 v_values 转换为数组，确保它们是 numpy 数组

    if u_values.ndim > 2 or v_values.ndim > 2:
        raise ValueError('Invalid input values. The inputs must have either '
                         'one or two dimensions.')
    # 如果 u_values 或 v_values 的维度大于2，则抛出 ValueError 异常，要求输入值必须是一维或二维数组

    if u_values.ndim != v_values.ndim:
        raise ValueError('Invalid input values. Dimensions of inputs must be '
                         'equal.')
    # 如果 u_values 和 v_values 的维度不相等，则抛出 ValueError 异常，要求输入的维度必须相等

    if u_values.ndim == 1 and v_values.ndim == 1:
        return _cdf_distance(1, u_values, v_values, u_weights, v_weights)
    # 如果 u_values 和 v_values 都是一维数组，则调用 _cdf_distance 函数进行计算

    u_values, u_weights = _validate_distribution(u_values, u_weights)
    v_values, v_weights = _validate_distribution(v_values, v_weights)
    # 调用 _validate_distribution 函数验证并返回 u_values、v_values 的分布和权重

    if u_values.shape[1] != v_values.shape[1]:
        raise ValueError('Invalid input values. If two-dimensional, '
                         '`u_values` and `v_values` must have the same '
                         'number of columns.')
    # 如果 u_values 和 v_values 是二维数组且列数不相等，则抛出 ValueError 异常，要求列数必须相等

    if np.any(np.isinf(u_values)) ^ np.any(np.isinf(v_values)):
        return np.inf
    elif np.any(np.isinf(u_values)) and np.any(np.isinf(v_values)):
        return np.nan
    # 如果 u_values 中包含 np.inf 而 v_values 不包含，或者反之，则返回 np.inf；
    # 如果 u_values 和 v_values 同时包含 np.inf，则返回 np.nan

    A_upper_part = sparse.block_diag((np.ones((1, n)), ) * m)
    # 创建稀疏块对角矩阵 A_upper_part，其大小为 m*n，每个块对角元素为1

    A_lower_part = sparse.hstack((sparse.eye(n), ) * m)
    # 创建稀疏水平堆叠矩阵 A_lower_part，其大小为 m*n，每个堆叠部分为 n*n 的单位矩阵

    A = sparse.vstack((A_upper_part, A_lower_part))
    A = sparse.coo_array(A)
    # 垂直堆叠 A_upper_part 和 A_lower_part 形成整体稀疏矩阵 A，并转换为 COO 格式稀疏矩阵 A

    D = distance_matrix(u_values, v_values, p=2)
    cost = D.ravel()
    # 计算 u_values 和 v_values 的距离矩阵 D，并将其展平成一维数组 cost

    p_u = np.full(m, 1/m) if u_weights is None else u_weights/np.sum(u_weights)
    p_v = np.full(n, 1/n) if v_weights is None else v_weights/np.sum(v_weights)
    # 计算概率分布向量 p_u 和 p_v，如果未提供权重则假设均匀分布
    # 将 p_u 和 p_v 沿着 axis=0 连接成一个新的 NumPy 数组 b
    b = np.concatenate((p_u, p_v), axis=0)

    # 构造线性约束条件对象，使用 A 的转置作为约束矩阵，上界为 cost
    constraints = LinearConstraint(A=A.T, ub=cost)
    
    # 调用 MILP 求解器，求解线性规划问题：
    #   目标函数为 -b
    #   约束条件为 constraints
    #   变量的边界为 (-inf, inf)
    opt_res = milp(c=-b, constraints=constraints, bounds=(-np.inf, np.inf))
    
    # 返回优化结果的负值作为函数的返回值
    return -opt_res.fun
# 定义计算两个一维离散分布之间的Wasserstein-1距离的函数
def wasserstein_distance(u_values, v_values, u_weights=None, v_weights=None):
    r"""
    Compute the Wasserstein-1 distance between two 1D discrete distributions.

    The Wasserstein distance, also called the Earth mover's distance or the
    optimal transport distance, is a similarity metric between two probability
    distributions [1]_. In the discrete case, the Wasserstein distance can be
    understood as the cost of an optimal transport plan to convert one
    distribution into the other. The cost is calculated as the product of the
    amount of probability mass being moved and the distance it is being moved.
    A brief and intuitive introduction can be found at [2]_.

    .. versionadded:: 1.0.0

    Parameters
    ----------
    u_values : 1d array_like
        A sample from a probability distribution or the support (set of all
        possible values) of a probability distribution. Each element is an
        observation or possible value.

    v_values : 1d array_like
        A sample from or the support of a second distribution.

    u_weights, v_weights : 1d array_like, optional
        Weights or counts corresponding with the sample or probability masses
        corresponding with the support values. Sum of elements must be positive
        and finite. If unspecified, each value is assigned the same weight.

    Returns
    -------
    distance : float
        The computed distance between the distributions.

    Notes
    -----
    Given two 1D probability mass functions, :math:`u` and :math:`v`, the first
    Wasserstein distance between the distributions is:

    .. math::

        l_1 (u, v) = \inf_{\pi \in \Gamma (u, v)} \int_{\mathbb{R} \times
        \mathbb{R}} |x-y| \mathrm{d} \pi (x, y)

    where :math:`\Gamma (u, v)` is the set of (probability) distributions on
    :math:`\mathbb{R} \times \mathbb{R}` whose marginals are :math:`u` and
    :math:`v` on the first and second factors respectively. For a given value
    :math:`x`, :math:`u(x)` gives the probabilty of :math:`u` at position
    :math:`x`, and the same for :math:`v(x)`.

    If :math:`U` and :math:`V` are the respective CDFs of :math:`u` and
    :math:`v`, this distance also equals to:

    .. math::

        l_1(u, v) = \int_{-\infty}^{+\infty} |U-V|

    See [3]_ for a proof of the equivalence of both definitions.

    The input distributions can be empirical, therefore coming from samples
    whose values are effectively inputs of the function, or they can be seen as
    generalized functions, in which case they are weighted sums of Dirac delta
    functions located at the specified values.

    References
    ----------
    .. [1] "Wasserstein metric", https://en.wikipedia.org/wiki/Wasserstein_metric
    .. [2] Lili Weng, "What is Wasserstein distance?", Lil'log,
           https://lilianweng.github.io/posts/2017-08-20-gan/#what-is-wasserstein-distance.
    .. [3] For a proof of the equivalence of the definitions of Wasserstein distance,
           refer to the referenced paper.
    # 返回使用累积分布函数 (CDF) 距离计算函数 `_cdf_distance` 计算的结果
    return _cdf_distance(1, u_values, v_values, u_weights, v_weights)
# 定义计算两个一维分布之间能量距离的函数
def energy_distance(u_values, v_values, u_weights=None, v_weights=None):
    r"""Compute the energy distance between two 1D distributions.

    .. versionadded:: 1.0.0

    Parameters
    ----------
    u_values, v_values : array_like
        Values observed in the (empirical) distribution.
    u_weights, v_weights : array_like, optional
        Weight for each value. If unspecified, each value is assigned the same
        weight.
        `u_weights` (resp. `v_weights`) must have the same length as
        `u_values` (resp. `v_values`). If the weight sum differs from 1, it
        must still be positive and finite so that the weights can be normalized
        to sum to 1.

    Returns
    -------
    distance : float
        The computed distance between the distributions.

    Notes
    -----
    The energy distance between two distributions :math:`u` and :math:`v`, whose
    respective CDFs are :math:`U` and :math:`V`, equals to:

    .. math::

        D(u, v) = \left( 2\mathbb E|X - Y| - \mathbb E|X - X'| -
        \mathbb E|Y - Y'| \right)^{1/2}

    where :math:`X` and :math:`X'` (resp. :math:`Y` and :math:`Y'`) are
    independent random variables whose probability distribution is :math:`u`
    (resp. :math:`v`).

    Sometimes the square of this quantity is referred to as the "energy
    distance" (e.g. in [2]_, [4]_), but as noted in [1]_ and [3]_, only the
    definition above satisfies the axioms of a distance function (metric).

    As shown in [2]_, for one-dimensional real-valued variables, the energy
    distance is linked to the non-distribution-free version of the Cramér-von
    Mises distance:

    .. math::

        D(u, v) = \sqrt{2} l_2(u, v) = \left( 2 \int_{-\infty}^{+\infty} (U-V)^2
        \right)^{1/2}

    Note that the common Cramér-von Mises criterion uses the distribution-free
    version of the distance. See [2]_ (section 2), for more details about both
    versions of the distance.

    The input distributions can be empirical, therefore coming from samples
    whose values are effectively inputs of the function, or they can be seen as
    generalized functions, in which case they are weighted sums of Dirac delta
    functions located at the specified values.

    References
    ----------
    .. [1] Rizzo, Szekely "Energy distance." Wiley Interdisciplinary Reviews:
           Computational Statistics, 8(1):27-38 (2015).
    .. [2] Szekely "E-statistics: The energy of statistical samples." Bowling
           Green State University, Department of Mathematics and Statistics,
           Technical Report 02-16 (2002).
    .. [3] "Energy distance", https://en.wikipedia.org/wiki/Energy_distance
    .. [4] Bellemare, Danihelka, Dabney, Mohamed, Lakshminarayanan, Hoyer,
           Munos "The Cramer Distance as a Solution to Biased Wasserstein
           Gradients" (2017). :arXiv:`1705.10743`.

    Examples
    --------
    >>> from scipy.stats import energy_distance
    >>> energy_distance([0], [2])
    2.0000000000000004
    # 调用 energy_distance 函数计算两个样本数据集之间的能量距离
    >>> energy_distance([0, 8], [0, 8], [3, 1], [2, 2])
    # 返回值为 1.0000000000000002
    
    # 再次调用 energy_distance 函数计算两个不同的样本数据集之间的能量距离
    >>> energy_distance([0.7, 7.4, 2.4, 6.8], [1.4, 8. ],
    ...                 [2.1, 4.2, 7.4, 8. ], [7.6, 8.8])
    # 返回值为 0.88003340976158217
    
    """
    # 使用数学库 NumPy 中的函数 np.sqrt 计算平方根，乘以常数 np.sqrt(2)，并调用 _cdf_distance 函数
    # 该函数计算两组样本数据 u_values 和 v_values 之间按照权重 u_weights 和 v_weights 进行的累积分布函数距离
    return np.sqrt(2) * _cdf_distance(2, u_values, v_values,
                                      u_weights, v_weights)
    # 定义一个函数来计算两个一维分布 u 和 v 之间的统计距离，该距离被定义为 CDF 的差的 p 次幂的 p 次根
    r"""
    Compute, between two one-dimensional distributions :math:`u` and
    :math:`v`, whose respective CDFs are :math:`U` and :math:`V`, the
    statistical distance that is defined as:

    .. math::

        l_p(u, v) = \left( \int_{-\infty}^{+\infty} |U-V|^p \right)^{1/p}

    p is a positive parameter; p = 1 gives the Wasserstein distance, p = 2
    gives the energy distance.

    Parameters
    ----------
    u_values, v_values : array_like
        Values observed in the (empirical) distribution.
    u_weights, v_weights : array_like, optional
        Weight for each value. If unspecified, each value is assigned the same
        weight.
        `u_weights` (resp. `v_weights`) must have the same length as
        `u_values` (resp. `v_values`). If the weight sum differs from 1, it
        must still be positive and finite so that the weights can be normalized
        to sum to 1.

    Returns
    -------
    distance : float
        The computed distance between the distributions.

    Notes
    -----
    The input distributions can be empirical, therefore coming from samples
    whose values are effectively inputs of the function, or they can be seen as
    generalized functions, in which case they are weighted sums of Dirac delta
    functions located at the specified values.

    References
    ----------
    .. [1] Bellemare, Danihelka, Dabney, Mohamed, Lakshminarayanan, Hoyer,
           Munos "The Cramer Distance as a Solution to Biased Wasserstein
           Gradients" (2017). :arXiv:`1705.10743`.

    """
    # 对输入的 u 和 v 分布进行验证，并确保它们符合条件
    u_values, u_weights = _validate_distribution(u_values, u_weights)
    v_values, v_weights = _validate_distribution(v_values, v_weights)

    # 对 u_values 和 v_values 进行排序，并获取排序后的索引
    u_sorter = np.argsort(u_values)
    v_sorter = np.argsort(v_values)

    # 将 u_values 和 v_values 合并并排序，使用 mergesort 确保稳定排序
    all_values = np.concatenate((u_values, v_values))
    all_values.sort(kind='mergesort')

    # 计算 u 和 v 各对应值的差值
    deltas = np.diff(all_values)

    # 获取 u 和 v 在合并后的值数组中的累积分布函数的索引位置
    u_cdf_indices = u_values[u_sorter].searchsorted(all_values[:-1], 'right')
    v_cdf_indices = v_values[v_sorter].searchsorted(all_values[:-1], 'right')

    # 如果未指定权重，使用默认的权重计算 u 和 v 的累积分布函数
    if u_weights is None:
        u_cdf = u_cdf_indices / u_values.size
    else:
        # 如果指定了权重，计算加权累积分布函数
        u_sorted_cumweights = np.concatenate(([0],
                                              np.cumsum(u_weights[u_sorter])))
        u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

    if v_weights is None:
        v_cdf = v_cdf_indices / v_values.size
    else:
        # 如果指定了权重，计算加权累积分布函数
        v_sorted_cumweights = np.concatenate(([0],
                                              np.cumsum(v_weights[v_sorter])))
        v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]
    # 根据累积分布函数（CDF）计算积分的值。
    # 如果 p = 1 或 p = 2，避免使用 np.power，因为这会引入大约15%的额外开销。
    if p == 1:
        # 当 p = 1 时，计算绝对值差乘以增量的总和
        return np.sum(np.multiply(np.abs(u_cdf - v_cdf), deltas))
    if p == 2:
        # 当 p = 2 时，计算绝对值差平方乘以增量的总和的平方根
        return np.sqrt(np.sum(np.multiply(np.square(u_cdf - v_cdf), deltas)))
    # 对于其他 p 值，计算绝对值差的 p 次幂乘以增量的总和再开 p 次方根
    return np.power(np.sum(np.multiply(np.power(np.abs(u_cdf - v_cdf), p),
                                       deltas)), 1/p)
def _validate_distribution(values, weights):
    """
    Validate the values and weights from a distribution input of `cdf_distance`
    and return them as ndarray objects.

    Parameters
    ----------
    values : array_like
        Values observed in the (empirical) distribution.
    weights : array_like
        Weight for each value.

    Returns
    -------
    values : ndarray
        Values as ndarray.
    weights : ndarray
        Weights as ndarray.

    """
    # 将值数组转换为浮点类型的 ndarray
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("Distribution can't be empty.")

    # 如果有权重数组，进行验证
    if weights is not None:
        # 将权重数组转换为浮点类型的 ndarray
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(values):
            raise ValueError('Value and weight array-likes for the same '
                             'empirical distribution must be of the same size.')
        if np.any(weights < 0):
            raise ValueError('All weights must be non-negative.')
        if not 0 < np.sum(weights) < np.inf:
            raise ValueError('Weight array-like sum must be positive and '
                             'finite. Set as None for an equal distribution of '
                             'weight.')

        return values, weights

    # 如果没有权重数组，返回验证后的值数组和 None
    return values, None


#####################################
#         SUPPORT FUNCTIONS         #
#####################################

# 命名元组，用于存储重复值和计数
RepeatedResults = namedtuple('RepeatedResults', ('values', 'counts'))


def find_repeats(arr):
    """Find repeats and repeat counts.

    Parameters
    ----------
    arr : array_like
        Input array. This is cast to float64.

    Returns
    -------
    values : ndarray
        The unique values from the (flattened) input that are repeated.

    counts : ndarray
        Number of times the corresponding 'value' is repeated.

    Notes
    -----
    In numpy >= 1.9 `numpy.unique` provides similar functionality. The main
    difference is that `find_repeats` only returns repeated values.

    Examples
    --------
    >>> from scipy import stats
    >>> stats.find_repeats([2, 1, 2, 3, 2, 2, 5])
    RepeatedResults(values=array([2.]), counts=array([4]))

    >>> stats.find_repeats([[10, 20, 1, 2], [5, 5, 4, 4]])
    RepeatedResults(values=array([4.,  5.]), counts=array([2, 2]))

    """
    # 注意：始终进行数组的拷贝操作
    return RepeatedResults(*_find_repeats(np.array(arr, dtype=np.float64)))


def _sum_of_squares(a, axis=0):
    """Square each element of the input array, and return the sum(s) of that.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or None, optional
        Axis along which to calculate. Default is 0. If None, compute over
        the whole array `a`.

    Returns
    -------
    sum_of_squares : ndarray
        The sum along the given axis for (a**2).

    See Also
    --------
    ```
    # 将数组 `a` 和轴 `axis` 转换为适合计算的形式
    a, axis = _chk_asarray(a, axis)
    # 计算数组 `a` 沿指定轴的每个元素的平方，并对结果进行求和
    return np.sum(a*a, axis)
# 定义函数 `_square_of_sums`，计算输入数组中指定轴上元素的和的平方
def _square_of_sums(a, axis=0):
    """Sum elements of the input array, and return the square(s) of that sum.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or None, optional
        Axis along which to calculate. Default is 0. If None, compute over
        the whole array `a`.

    Returns
    -------
    square_of_sums : float or ndarray
        The square of the sum over `axis`.

    See Also
    --------
    _sum_of_squares : The sum of squares (the opposite of `square_of_sums`).

    """
    # 将输入数组 `a` 和轴 `axis` 转换为数组对象，确保可以处理多种输入情况
    a, axis = _chk_asarray(a, axis)
    # 计算数组 `a` 指定轴上的元素和
    s = np.sum(a, axis)
    # 如果和 `s` 不是标量，则返回其类型转换为浮点型后的平方和
    if not np.isscalar(s):
        return s.astype(float) * s
    else:
        # 如果 `s` 是标量，则返回其转换为浮点型后的平方
        return float(s) * s


# 定义函数 `rankdata`，给定数据分配排名，并处理并列的情况
def rankdata(a, method='average', *, axis=None, nan_policy='propagate'):
    """Assign ranks to data, dealing with ties appropriately.

    By default (``axis=None``), the data array is first flattened, and a flat
    array of ranks is returned. Separately reshape the rank array to the
    shape of the data array if desired (see Examples).

    Ranks begin at 1.  The `method` argument controls how ranks are assigned
    to equal values.  See [1]_ for further discussion of ranking methods.

    Parameters
    ----------
    a : array_like
        The array of values to be ranked.
    method : {'average', 'min', 'max', 'dense', 'ordinal'}, optional
        The method used to assign ranks to tied elements.
        The following methods are available (default is 'average'):

          * 'average': The average of the ranks that would have been assigned to
            all the tied values is assigned to each value.
          * 'min': The minimum of the ranks that would have been assigned to all
            the tied values is assigned to each value.  (This is also
            referred to as "competition" ranking.)
          * 'max': The maximum of the ranks that would have been assigned to all
            the tied values is assigned to each value.
          * 'dense': Like 'min', but the rank of the next highest element is
            assigned the rank immediately after those assigned to the tied
            elements.
          * 'ordinal': All values are given a distinct rank, corresponding to
            the order that the values occur in `a`.
    axis : {None, int}, optional
        Axis along which to perform the ranking. If ``None``, the data array
        is first flattened.
    # 定义了处理 NaN 值的策略，可以是 'propagate', 'omit', 'raise' 中的一种
    nan_policy : {'propagate', 'omit', 'raise'}, optional
        Defines how to handle when input contains nan.

        # 当 'nan_policy' 是 'propagate' 时，输出的数组中所有的值都是 NaN，
        # 因为相对于输入中的 NaN 的排名是未定义的。
        # 当 'nan_policy' 是 'omit' 时，忽略输入中的 NaN 值进行排名，
        # 输出数组相应位置的值也是 NaN。

        The following options are available (default is 'propagate'):

          * 'propagate': propagates nans through the rank calculation
          * 'omit': performs the calculations ignoring nan values
          * 'raise': raises an error

        .. note::

            When `nan_policy` is 'propagate', the output is an array of *all*
            nans because ranks relative to nans in the input are undefined.
            When `nan_policy` is 'omit', nans in `a` are ignored when ranking
            the other values, and the corresponding locations of the output
            are nan.

        .. versionadded:: 1.10

    # 返回值
    Returns
    -------
    # 返回一个大小与输入数组 `a` 相同的 ndarray，包含排名分数。
    ranks : ndarray
         An array of size equal to the size of `a`, containing rank
         scores.

    # 参考文献
    References
    ----------
    # 关于排名的更多信息可以参考维基百科的 "Ranking" 页面
    .. [1] "Ranking", https://en.wikipedia.org/wiki/Ranking

    # 示例
    Examples
    --------
    # 导入需要的库
    >>> import numpy as np
    >>> from scipy.stats import rankdata
    # 对数组进行排名计算，并输出结果
    >>> rankdata([0, 2, 3, 2])
    array([ 1. ,  2.5,  4. ,  2.5])
    # 使用不同的方法进行排名计算
    >>> rankdata([0, 2, 3, 2], method='min')
    array([ 1,  2,  4,  2])
    >>> rankdata([0, 2, 3, 2], method='max')
    array([ 1,  3,  4,  3])
    >>> rankdata([0, 2, 3, 2], method='dense')
    array([ 1,  2,  3,  2])
    >>> rankdata([0, 2, 3, 2], method='ordinal')
    array([ 1,  2,  4,  3])
    >>> rankdata([[0, 2], [3, 2]]).reshape(2,2)
    array([[1. , 2.5],
          [4. , 2.5]])
    >>> rankdata([[0, 2, 2], [3, 2, 5]], axis=1)
    array([[1. , 2.5, 2.5],
           [2. , 1. , 3. ]])
    # 使用不同的 NaN 策略进行排名计算
    >>> rankdata([0, 2, 3, np.nan, -2, np.nan], nan_policy="propagate")
    array([nan, nan, nan, nan, nan, nan])
    >>> rankdata([0, 2, 3, np.nan, -2, np.nan], nan_policy="omit")
    array([ 2.,  3.,  4., nan,  1., nan])

    """
# 根据给定的排序方法对数据 `x` 进行排名，可选择是否返回关于并列排名的信息
def _rankdata(x, method, return_ties=False):
    # 获取数据 `x` 的形状
    shape = x.shape

    # 根据排序方法选择合适的排序算法
    kind = 'mergesort' if method == 'ordinal' else 'quicksort'
    # 对数据 `x` 进行排序，并返回排序后的索引
    j = np.argsort(x, axis=-1, kind=kind)
    # 创建一个按序数方式排列的排名数组
    ordinal_ranks = np.broadcast_to(np.arange(1, shape[-1]+1, dtype=int), shape)

    # 如果是序数排名方法，直接返回按排序后的顺序重排的排名数组
    if method == 'ordinal':
        return _order_ranks(ordinal_ranks, j)  # 不返回并列排名信息

    # 将数组 `x` 按照排序后的索引 `j` 进行排序
    y = np.take_along_axis(x, j, axis=-1)
    # 计算每个唯一元素的逻辑索引
    i = np.concatenate([np.ones(shape[:-1] + (1,), dtype=np.bool_),
                       y[..., :-1] != y[..., 1:]], axis=-1)

    # 获取唯一元素的整数索引
    indices = np.arange(y.size)[i.ravel()]
    # 计算唯一元素的出现次数
    counts = np.diff(indices, append=y.size)

    # 根据不同的方法计算 `'min'`, `'max'`, `'average'` 和 `'dense'` 排名
    if method == 'min':
        ranks = ordinal_ranks[i]
    elif method == 'max':
        ranks = ordinal_ranks[i] + counts - 1
    elif method == 'average':
        ranks = ordinal_ranks[i] + (counts - 1)/2
    elif method == 'dense':
        ranks = np.cumsum(i, axis=-1)[i]

    # 将排名数组 `ranks` 中的元素根据其出现次数进行重复，以恢复到原始数据的形状
    ranks = np.repeat(ranks, counts).reshape(shape)
    # 根据排序后的索引 `j` 对排名数组 `ranks` 进行重排序
    ranks = _order_ranks(ranks, j)

    # 如果需要返回并列排名信息
    if return_ties:
        # 返回并列排名信息 `t`，格式适用于依赖此信息的函数
        t = np.zeros(shape, dtype=float)
        t[i] = counts
        return ranks, t
    # 否则只返回排名数组 `ranks`
    return ranks
    r"""Compute the expectile at the specified level.

    Expectiles are a generalization of the expectation in the same way as
    quantiles are a generalization of the median. The expectile at level
    `alpha = 0.5` is the mean (average). See Notes for more details.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose expectile is desired.
    alpha : float, default: 0.5
        The level of the expectile; e.g., ``alpha=0.5`` gives the mean.
    weights : array_like, optional
        An array of weights associated with the values in `a`.
        The `weights` must be broadcastable to the same shape as `a`.
        Default is None, which gives each value a weight of 1.0.
        An integer valued weight element acts like repeating the corresponding
        observation in `a` that many times. See Notes for more details.

    Returns
    -------
    expectile : ndarray
        The empirical expectile at level `alpha`.

    See Also
    --------
    numpy.mean : Arithmetic average
    numpy.quantile : Quantile

    Notes
    -----
    In general, the expectile at level :math:`\alpha` of a random variable
    :math:`X` with cumulative distribution function (CDF) :math:`F` is given
    by the unique solution :math:`t` of:

    .. math::

        \alpha E((X - t)_+) = (1 - \alpha) E((t - X)_+) \,.

    Here, :math:`(x)_+ = \max(0, x)` is the positive part of :math:`x`.
    This equation can be equivalently written as:

    .. math::

        \alpha \int_t^\infty (x - t)\mathrm{d}F(x)
        = (1 - \alpha) \int_{-\infty}^t (t - x)\mathrm{d}F(x) \,.

    The empirical expectile at level :math:`\alpha` (`alpha`) of a sample
    :math:`a_i` (the array `a`) is defined by plugging in the empirical CDF of
    `a`. Given sample or case weights :math:`w` (the array `weights`), it
    reads :math:`F_a(x) = \frac{1}{\sum_i w_i} \sum_i w_i 1_{a_i \leq x}`
    with indicator function :math:`1_{A}`. This leads to the definition of the
    empirical expectile at level `alpha` as the unique solution :math:`t` of:

    .. math::

        \alpha \sum_{i=1}^n w_i (a_i - t)_+ =
            (1 - \alpha) \sum_{i=1}^n w_i (t - a_i)_+ \,.

    For :math:`\alpha=0.5`, this simplifies to the weighted average.
    Furthermore, the larger :math:`\alpha`, the larger the value of the
    expectile.

    As a final remark, the expectile at level :math:`\alpha` can also be
    written as a minimization problem. One often used choice is

    .. math::

        \operatorname{argmin}_t
        E(\lvert 1_{t\geq X} - \alpha\rvert(t - X)^2) \,.

    References
    ----------
    .. [1] W. K. Newey and J. L. Powell (1987), "Asymmetric Least Squares
           Estimation and Testing," Econometrica, 55, 819-847.
    .. [2] T. Gneiting (2009). "Making and Evaluating Point Forecasts,"
           Journal of the American Statistical Association, 106, 746 - 762.
           :doi:`10.48550/arXiv.0912.0902`

    Examples
    --------

"""
    >>> import numpy as np
    >>> from scipy.stats import expectile
    >>> a = [1, 4, 2, -1]
    >>> expectile(a, alpha=0.5) == np.mean(a)
    True
    >>> expectile(a, alpha=0.2)
    0.42857142857142855
    >>> expectile(a, alpha=0.8)
    2.5714285714285716
    >>> weights = [1, 3, 1, 1]

    """
    # 检查 alpha 是否在合法范围内，若不在则抛出 ValueError 异常
    if alpha < 0 or alpha > 1:
        raise ValueError(
            "The expectile level alpha must be in the range [0, 1]."
        )
    # 将输入列表 a 转换为 NumPy 数组
    a = np.asarray(a)

    # 如果 weights 参数不为 None，则将其广播到与 a 相同的形状
    if weights is not None:
        weights = np.broadcast_to(weights, a.shape)

    # 定义一阶条件函数 first_order(t)，用于计算 expectile
    # 这是根据文献 [2] 中的表格 9 的等效实验公式（省略了因子 2）
    def first_order(t):
        return np.average(np.abs((a <= t) - alpha) * (t - a), weights=weights)

    # 根据 alpha 的值选择 x0 和 x1 的初始值
    if alpha >= 0.5:
        x0 = np.average(a, weights=weights)
        x1 = np.amax(a)
    else:
        x1 = np.average(a, weights=weights)
        x0 = np.amin(a)

    # 若 x0 和 x1 相等，则说明 a 只有一个唯一的元素，直接返回 x0
    if x0 == x1:
        return x0

    # 使用 root_scalar 函数计算 expectile 的数值解
    # 注意，expectile 是唯一解，因此不会出现找到错误的根的情况
    res = root_scalar(first_order, x0=x0, x1=x1)
    return res.root
# 创建名为 LinregressResult 的命名元组扩展类，包含字段 'slope', 'intercept', 'rvalue', 'pvalue', 'stderr'，
# 并额外添加字段 'intercept_stderr'
LinregressResult = _make_tuple_bunch('LinregressResult',
                                     ['slope', 'intercept', 'rvalue',
                                      'pvalue', 'stderr'],
                                     extra_field_names=['intercept_stderr'])

# 定义函数 linregress，计算两组测量数据的线性最小二乘回归
def linregress(x, y=None, alternative='two-sided'):
    """
    Calculate a linear least-squares regression for two sets of measurements.

    Parameters
    ----------
    x, y : array_like
        Two sets of measurements.  Both arrays should have the same length N.  If
        only `x` is given (and ``y=None``), then it must be a two-dimensional
        array where one dimension has length 2.  The two sets of measurements
        are then found by splitting the array along the length-2 dimension. In
        the case where ``y=None`` and `x` is a 2xN array, ``linregress(x)`` is
        equivalent to ``linregress(x[0], x[1])``.

        .. deprecated:: 1.14.0
            Inference of the two sets of measurements from a single argument `x`
            is deprecated will result in an error in SciPy 1.16.0; the sets
            must be specified separately as `x` and `y`.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        The following options are available:

        * 'two-sided': the slope of the regression line is nonzero
        * 'less': the slope of the regression line is less than zero
        * 'greater':  the slope of the regression line is greater than zero

        .. versionadded:: 1.7.0

    Returns
    -------
    result : ``LinregressResult`` instance
        The return value is an object with the following attributes:

        slope : float
            Slope of the regression line.
        intercept : float
            Intercept of the regression line.
        rvalue : float
            The Pearson correlation coefficient. The square of ``rvalue``
            is equal to the coefficient of determination.
        pvalue : float
            The p-value for a hypothesis test whose null hypothesis is
            that the slope is zero, using Wald Test with t-distribution of
            the test statistic. See `alternative` above for alternative
            hypotheses.
        stderr : float
            Standard error of the estimated slope (gradient), under the
            assumption of residual normality.
        intercept_stderr : float
            Standard error of the estimated intercept, under the assumption
            of residual normality.

    See Also
    --------
    scipy.optimize.curve_fit :
        Use non-linear least squares to fit a function to data.
    scipy.optimize.leastsq :
        Minimize the sum of squares of a set of equations.

    Notes
    -----
    For compatibility with older versions of SciPy, the return value acts
    like a ``namedtuple`` of length 5, with fields ``slope``, ``intercept``,
    ``rvalue``, ``pvalue``, ``stderr``.
    """
    """
    TINY = 1.0e-20
    设置一个极小值，用于数值稳定性

    if y is None:  # x is a (2, N) or (N, 2) shaped array_like
        如果 y 为 None，则表示 x 可能是一个形状为 (2, N) 或 (N, 2) 的数组
        message = ('Inference of the two sets of measurements from a single "'
                   'argument `x` is deprecated will result in an error in "'
                   'SciPy 1.16.0; the sets must be specified separately as "'
                   '`x` and `y`.')
        生成一条警告消息，说明从单个参数 `x` 推断两组测量结果已被弃用，并且在 SciPy 1.16.0 中会导致错误；必须将这两组分别指定为 `x` 和 `y`
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        发出警告，使用 DeprecationWarning 类型，并指定警告的堆栈级别为 2
        x = np.asarray(x)
        将 x 转换为 NumPy 数组
        if x.shape[0] == 2:
            如果 x 的第一个维度为 2，
            x, y = x
            则解包 x 为 x 和 y
        elif x.shape[1] == 2:
            如果 x 的第二个维度为 2，
            x, y = x.T
            则转置 x 并解包为 x 和 y
        else:
            否则，抛出 ValueError 异常，说明如果只提供 `x` 作为输入，则其形状必须为 (2, N) 或 (N, 2)，而提供的形状为 x.shape
                             f"was {x.shape}.")
    else:
        否则，如果 y 不为 None，则分别将 x 和 y 转换为 NumPy 数组
        x = np.asarray(x)
        y = np.asarray(y)

    if x.size == 0 or y.size == 0:
        如果 x 或 y 的大小为 0，则抛出 ValueError 异常，说明输入不能为空

    if np.amax(x) == np.amin(x) and len(x) > 1:
        如果 x 的最大值等于最小值，并且 x 的长度大于 1，则抛出 ValueError 异常，说明如果所有 x 值都相同，则无法计算线性回归

    n = len(x)
    计算 x 的长度，并赋值给 n
    xmean = np.mean(x, None)
    计算 x 的均值，并赋值给 xmean
    ymean = np.mean(y, None)
    计算 y 的均值，并赋值给 ymean

    # Average sums of square differences from the mean
    #   ssxm = mean( (x-mean(x))^2 )
    #   ssxym = mean( (x-mean(x)) * (y-mean(y)) )
    计算从均值差异中的平均平方差
    ssxm = np.mean((x - xmean) ** 2)
    计算 x 与 x 均值之间的平方差的平均值，并赋值给 ssxm
    ssxym = np.mean((x - xmean) * (y - ymean))
    计算 (x - x 均值) 与 (y - y 均值) 的乘积的平均值，并赋值给 ssxym
    ```
    # Calculate covariance values ssxm, ssxym, _, ssym using np.cov function
    ssxm, ssxym, _, ssym = np.cov(x, y, bias=1).flat

    # Calculate Pearson correlation coefficient (r-value)
    # r = ssxym / sqrt( ssxm * ssym )
    if ssxm == 0.0 or ssym == 0.0:
        # Handle the case where the denominator would be zero
        r = 0.0
    else:
        r = ssxym / np.sqrt(ssxm * ssym)
        # Ensure r stays within valid range due to numerical precision issues
        if r > 1.0:
            r = 1.0
        elif r < -1.0:
            r = -1.0

    # Calculate slope of the linear regression line
    slope = ssxym / ssxm

    # Calculate intercept of the linear regression line
    intercept = ymean - slope*xmean

    if n == 2:
        # Special case handling when there are only two data points
        if y[0] == y[1]:
            prob = 1.0
        else:
            prob = 0.0
        # Standard errors are zero in this case
        slope_stderr = 0.0
        intercept_stderr = 0.0
    else:
        df = n - 2  # Calculate degrees of freedom
        # Compute t-statistic for hypothesis testing
        TINY = np.finfo(np.float64).tiny  # Tiny value to prevent division by zero
        t = r * np.sqrt(df / ((1.0 - r + TINY)*(1.0 + r + TINY)))

        # Create a Student's t-distribution object
        dist = _SimpleStudentT(df)
        # Calculate p-value using the t-statistic and distribution
        prob = _get_pvalue(t, dist, alternative, xp=np)
        prob = prob[()] if prob.ndim == 0 else prob  # Ensure prob is scalar

        # Calculate standard error of the slope estimate
        slope_stderr = np.sqrt((1 - r**2) * ssym / ssxm / df)

        # Calculate standard error of the intercept estimate
        # Using the relationship ssxm = mean( (x-mean(x))^2 )
        intercept_stderr = slope_stderr * np.sqrt(ssxm + xmean**2)

    # Return results of linear regression analysis as a LinregressResult object
    return LinregressResult(slope=slope, intercept=intercept, rvalue=r,
                            pvalue=prob, stderr=slope_stderr,
                            intercept_stderr=intercept_stderr)
# 定义函数 _xp_mean，计算沿指定轴的算术平均值
def _xp_mean(x, /, *, axis=None, weights=None, keepdims=False, nan_policy='propagate',
             dtype=None, xp=None):
    r"""Compute the arithmetic mean along the specified axis.

    Parameters
    ----------
    x : real array
        Array containing real numbers whose mean is desired.
    axis : int or tuple of ints, default: None
        If an int or tuple of ints, the axis or axes of the input along which
        to compute the statistic. The statistic of each axis-slice (e.g. row)
        of the input will appear in a corresponding element of the output.
        If ``None``, the input will be raveled before computing the statistic.
    weights : real array, optional
        If specified, an array of weights associated with the values in `x`;
        otherwise ``1``. If `weights` and `x` do not have the same shape, the
        arrays will be broadcasted before performing the calculation. See
        Notes for details.
    keepdims : boolean, optional
        If this is set to ``True``, the axes which are reduced are left
        in the result as dimensions with length one. With this option,
        the result will broadcast correctly against the input array.
    nan_policy : {'propagate', 'omit', 'raise'}, default: 'propagate'
        Defines how to handle input NaNs.

        - ``propagate``: if a NaN is present in the axis slice (e.g. row) along
          which the statistic is computed, the corresponding entry of the output
          will be NaN.
        - ``omit``: NaNs will be omitted when performing the calculation.
          If insufficient data remains in the axis slice along which the
          statistic is computed, the corresponding entry of the output will be
          NaN.
        - ``raise``: if a NaN is present, a ``ValueError`` will be raised.

    dtype : dtype, optional
        Type to use in computing the mean. For integer inputs, the default is
        the default float type of the array library; for floating point inputs,
        the dtype is that of the input.

    Returns
    -------
    out : array
        The mean of each slice

    Notes
    -----
    Let :math:`x_i` represent element :math:`i` of data `x` and let :math:`w_i`
    represent the corresponding element of `weights` after broadcasting. Then the
    (weighted) mean :math:`\bar{x}_w` is given by:

    .. math::

        \bar{x}_w = \frac{ \sum_{i=0}^{n-1} w_i x_i }
                         { \sum_{i=0}^{n-1} w_i }

    where :math:`n` is the number of elements along a slice. Note that this simplifies
    to the familiar :math:`(\sum_i x_i) / n` when the weights are all ``1`` (default).

    The behavior of this function with respect to weights is somewhat different
    from that of `np.average`. For instance,
    `np.average` raises an error when `axis` is not specified and the shapes of `x`
    and the `weights` array are not the same; `xp_mean` simply broadcasts the two.
    """
    Also, `np.average` raises an error when weights sum to zero along a slice;
    `xp_mean` computes the appropriate result. The intent is for this function's
    interface to be consistent with the rest of `scipy.stats`.

    Note that according to the formula, including NaNs with zero weights is not
    the same as *omitting* NaNs with ``nan_policy='omit'``; in the former case,
    the NaNs will continue to propagate through the calculation whereas in the
    latter case, the NaNs are excluded entirely.
    """

    # ensure that `x` and `weights` are array-API compatible arrays of identical shape
    xp = array_namespace(x) if xp is None else xp  # 使用给定的数组命名空间来创建数组，如果未提供则使用默认命名空间
    x = xp.asarray(x, dtype=dtype)  # 将输入数据 `x` 转换为给定的数组库类型和数据类型
    weights = xp.asarray(weights, dtype=dtype) if weights is not None else weights  # 如果提供了权重数据，则将权重数据转换为给定的数组库类型和数据类型；否则保持为 None

    # to ensure that this matches the behavior of decorated functions when one of the
    # arguments has size zero, it's easiest to call a similar decorated function.
    if is_numpy(xp) and (xp_size(x) == 0
                         or (weights is not None and xp_size(weights) == 0)):
        return gmean(x, weights=weights, axis=axis, keepdims=keepdims)  # 如果数组库是 NumPy 并且输入数组 `x` 或权重数组 `weights` 的大小为零，则调用 `gmean` 函数并返回结果

    # handle non-broadcastable inputs
    if weights is not None and x.shape != weights.shape:
        try:
            x, weights = _broadcast_arrays((x, weights), xp=xp)  # 如果权重数组 `weights` 存在且其形状与输入数据 `x` 的形状不匹配，则尝试广播它们使其形状相容
        except (ValueError, RuntimeError) as e:
            message = "Array shapes are incompatible for broadcasting."
            raise ValueError(message) from e  # 如果广播失败，则引发值错误并显示广播不兼容的消息

    # convert integers to the default float of the array library
    if not xp.isdtype(x.dtype, 'real floating'):
        dtype = xp.asarray(1.).dtype
        x = xp.asarray(x, dtype=dtype)  # 如果输入数据 `x` 不是实数浮点类型，则将其转换为数组库的默认浮点数类型
    if weights is not None and not xp.isdtype(weights.dtype, 'real floating'):
        dtype = xp.asarray(1.).dtype
        weights = xp.asarray(weights, dtype=dtype)  # 如果权重数组 `weights` 存在且不是实数浮点类型，则将其转换为数组库的默认浮点数类型

    # handle the special case of zero-sized arrays
    message = (too_small_1d_not_omit if (x.ndim == 1 or axis is None)
               else too_small_nd_not_omit)
    if xp_size(x) == 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = xp.mean(x, axis=axis, keepdims=keepdims)  # 如果输入数据 `x` 是零大小的数组，则计算其均值，并忽略警告
        if xp_size(res) != 0:
            warnings.warn(message, SmallSampleWarning, stacklevel=2)  # 如果结果不是零大小的数组，则发出特定警告消息
        return res  # 返回计算的均值结果

    contains_nan, _ = _contains_nan(x, nan_policy, xp_omit_okay=True, xp=xp)
    if weights is not None:
        contains_nan_w, _ = _contains_nan(weights, nan_policy, xp_omit_okay=True, xp=xp)
        contains_nan = contains_nan | contains_nan_w  # 检查输入数据 `x` 和权重数组 `weights` 中是否包含 NaN，根据 `nan_policy` 处理 NaN 值

    # Handle `nan_policy='omit'` by giving zero weight to NaNs, whether they
    # appear in `x` or `weights`. Emit warning if there is an all-NaN slice.
    message = (too_small_1d_omit if (x.ndim == 1 or axis is None)
               else too_small_nd_omit)  # 如果 `nan_policy='omit'`，则将 NaN 值的权重设置为零，无论它们出现在输入数据 `x` 还是权重数组 `weights` 中，并在整个数组都是 NaN 的情况下发出警告
    # 如果数据中包含 NaN，并且策略是忽略 NaN
    if contains_nan and nan_policy == 'omit':
        # 创建 NaN 掩码
        nan_mask = xp.isnan(x)
        # 如果有权重数据，合并 NaN 掩码
        if weights is not None:
            nan_mask |= xp.isnan(weights)
        # 检查是否存在全为 NaN 的行（轴向）
        if xp.any(xp.all(nan_mask, axis=axis)):
            # 发出警告，指示可能出现样本数量较少的情况
            warnings.warn(message, SmallSampleWarning, stacklevel=2)
        # 如果权重数据为空，则初始化为全1的数组；否则保持原样
        weights = xp.ones_like(x) if weights is None else weights
        # 将 NaN 处的值替换为零，分别对原始数据和权重数据
        x = xp.where(nan_mask, xp.asarray(0, dtype=x.dtype), x)
        weights = xp.where(nan_mask, xp.asarray(0, dtype=x.dtype), weights)

    # 执行均值计算
    if weights is None:
        # 如果没有权重数据，直接计算平均值
        return xp.mean(x, axis=axis, keepdims=keepdims)

    # 计算权重的总和
    norm = xp.sum(weights, axis=axis)
    # 计算加权和
    wsum = xp.sum(x * weights, axis=axis)

    # 在进行除法计算时，忽略除以零的警告和错误
    with np.errstate(divide='ignore', invalid='ignore'):
        res = wsum / norm

    # 根据 keepdims 参数保持结果的维度，并将 NumPy 的零维数组转换为标量
    if keepdims:
        # 如果 axis 是 None，则将结果形状设为全1
        if axis is None:
            final_shape = (1,) * len(x.shape)
        else:
            # 如果 axis 是标量或序列
            axes = (axis,) if not isinstance(axis, Sequence) else axis
            final_shape = list(x.shape)
            for i in axes:
                final_shape[i] = 1

        # 重新调整结果的形状
        res = xp.reshape(res, final_shape)

    # 如果结果是零维数组，则返回标量；否则返回数组本身
    return res[()] if res.ndim == 0 else res
# an array-api compatible function for variance with scipy.stats interface
# and features (e.g. `nan_policy`).
def _xp_var(x, /, *, axis=None, correction=0, keepdims=False, nan_policy='propagate',
            dtype=None, xp=None):
    xp = array_namespace(x) if xp is None else xp  # 如果 xp 为 None，则根据 x 确定数组命名空间
    x = xp.asarray(x)  # 将输入 x 转换为 xp 数组表示

    # use `_xp_mean` instead of `xp.var` for desired warning behavior
    # it would be nice to combine this with `_var`, which uses `_moment`
    # and therefore warns when precision is lost, but that does not support
    # `axis` tuples or keepdims. Eventually, `_axis_nan_policy` will simplify
    # `axis` tuples and implement `keepdims` for non-NumPy arrays; then it will
    # be easy.
    kwargs = dict(axis=axis, nan_policy=nan_policy, dtype=dtype, xp=xp)
    mean = _xp_mean(x, keepdims=True, **kwargs)  # 计算平均值，支持 axis 和 nan_policy
    x = xp.asarray(x, dtype=mean.dtype)
    var = _xp_mean((x - mean)**2, keepdims=keepdims, **kwargs)  # 计算方差

    if correction != 0:
        n = (xp_size(x) if axis is None
             # compact way to deal with axis tuples or ints
             else np.prod(np.asarray(x.shape)[np.asarray(axis)]))
        n = xp.asarray(n, dtype=var.dtype)

        if nan_policy == 'omit':
            nan_mask = xp.astype(xp.isnan(x), var.dtype)
            n = n - xp.sum(nan_mask, axis=axis, keepdims=keepdims)

        # Produce NaNs silently when n - correction <= 0
        factor = _lazywhere(n-correction > 0, (n, n-correction), xp.divide, xp.nan)
        var *= factor  # 应用修正因子

    return var[()] if var.ndim == 0 else var


class _SimpleNormal:
    # A very simple, array-API compatible normal distribution for use in
    # hypothesis tests. May be replaced by new infrastructure Normal
    # distribution in due time.

    def cdf(self, x):
        return special.ndtr(x)  # 标准正态分布的累积分布函数

    def sf(self, x):
        return special.ndtr(-x)  # 标准正态分布的生存函数

    def isf(self, x):
        return -special.ndtri(x)  # 标准正态分布的逆累积分布函数


class _SimpleChi2:
    # A very simple, array-API compatible chi-squared distribution for use in
    # hypothesis tests. May be replaced by new infrastructure chi-squared
    # distribution in due time.
    def __init__(self, df):
        self.df = df

    def cdf(self, x):
        return special.chdtr(self.df, x)  # 自由度为 df 的卡方分布的累积分布函数

    def sf(self, x):
        return special.chdtrc(self.df, x)  # 自由度为 df 的卡方分布的生存函数


class _SimpleBeta:
    # A very simple, array-API compatible beta distribution for use in
    # hypothesis tests. May be replaced by new infrastructure beta
    # distribution in due time.
    def __init__(self, a, b, *, loc=None, scale=None):
        self.a = a
        self.b = b
        self.loc = loc
        self.scale = scale

    def cdf(self, x):
        if self.loc is not None or self.scale is not None:
            loc = 0 if self.loc is None else self.loc
            scale = 1 if self.scale is None else self.scale
            return special.betainc(self.a, self.b, (x - loc)/scale)
        return special.betainc(self.a, self.b, x)  # beta 分布的累积分布函数
    # 定义一个函数 sf，接受参数 self 和 x
    def sf(self, x):
        # 如果 self.loc 不是 None 或者 self.scale 不是 None，则执行以下条件判断
        if self.loc is not None or self.scale is not None:
            # 如果 self.loc 是 None，则将 loc 设置为 0，否则设置为 self.loc 的值
            loc = 0 if self.loc is None else self.loc
            # 如果 self.scale 是 None，则将 scale 设置为 1，否则设置为 self.scale 的值
            scale = 1 if self.scale is None else self.scale
            # 调用 special 模块的 betaincc 函数，计算 (x - loc)/scale 的值作为参数传入
            return special.betaincc(self.a, self.b, (x - loc)/scale)
        # 如果 self.loc 和 self.scale 都是 None，则直接调用 betaincc 函数，将 x 作为参数传入
        return special.betaincc(self.a, self.b, x)
class _SimpleStudentT:
    # 一个非常简单的、与数组API兼容的t分布类，用于假设检验。可能会在适当的时候被新的基础设施t分布替代。
    def __init__(self, df):
        # 初始化方法，接受自由度df作为参数，并将其保存在实例变量中
        self.df = df

    def cdf(self, t):
        # 累积分布函数（Cumulative Distribution Function），计算t分布的累积概率分布
        return special.stdtr(self.df, t)

    def sf(self, t):
        # 生存函数（Survival Function），计算t分布的生存函数
        return special.stdtr(self.df, -t)
```