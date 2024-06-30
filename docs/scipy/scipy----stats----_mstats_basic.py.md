# `D:\src\scipysrc\scipy\scipy\stats\_mstats_basic.py`

```
"""
An extension of scipy.stats._stats_py to support masked arrays

"""
# Original author (2007): Pierre GF Gerard-Marchant

# 导入需要的模块和函数
__all__ = ['argstoarray',
           'count_tied_groups',
           'describe',
           'f_oneway', 'find_repeats','friedmanchisquare',
           'kendalltau','kendalltau_seasonal','kruskal','kruskalwallis',
           'ks_twosamp', 'ks_2samp', 'kurtosis', 'kurtosistest',
           'ks_1samp', 'kstest',
           'linregress',
           'mannwhitneyu', 'meppf','mode','moment','mquantiles','msign',
           'normaltest',
           'obrientransform',
           'pearsonr','plotting_positions','pointbiserialr',
           'rankdata',
           'scoreatpercentile','sem',
           'sen_seasonal_slopes','skew','skewtest','spearmanr',
           'siegelslopes', 'theilslopes',
           'tmax','tmean','tmin','trim','trimboth',
           'trimtail','trima','trimr','trimmed_mean','trimmed_std',
           'trimmed_stde','trimmed_var','tsem','ttest_1samp','ttest_onesamp',
           'ttest_ind','ttest_rel','tvar',
           'variation',
           'winsorize',
           'brunnermunzel',
           ]

import numpy as np
from numpy import ndarray
import numpy.ma as ma
from numpy.ma import masked, nomask
import math

import itertools
import warnings
from collections import namedtuple

from . import distributions
from scipy._lib._util import _rename_parameter, _contains_nan
from scipy._lib._bunch import _make_tuple_bunch
import scipy.special as special
import scipy.stats._stats_py
import scipy.stats._stats_py as _stats_py

from ._stats_mstats_common import (
        _find_repeats,
        theilslopes as stats_theilslopes,
        siegelslopes as stats_siegelslopes
        )


def _chk_asarray(a, axis):
    # Always returns a masked array, raveled for axis=None
    # 将输入数组a转换为masked array，并且如果axis为None，则将其展平为一维数组
    a = ma.asanyarray(a)
    if axis is None:
        a = ma.ravel(a)
        outaxis = 0
    else:
        outaxis = axis
    return a, outaxis


def _chk2_asarray(a, b, axis):
    # Always returns masked arrays a and b, raveled for axis=None
    # 将输入数组a和b都转换为masked array，并且如果axis为None，则将它们展平为一维数组
    a = ma.asanyarray(a)
    b = ma.asanyarray(b)
    if axis is None:
        a = ma.ravel(a)
        b = ma.ravel(b)
        outaxis = 0
    else:
        outaxis = axis
    return a, b, outaxis


def _chk_size(a, b):
    # Check and return masked arrays a and b, ensuring they have the same size
    # 检查并返回masked arrays a和b，并确保它们具有相同的大小
    a = ma.asanyarray(a)
    b = ma.asanyarray(b)
    (na, nb) = (a.size, b.size)
    if na != nb:
        raise ValueError("The size of the input array should match!"
                         f" ({na} <> {nb})")
    return (a, b, na)


def _ttest_finish(df, t, alternative):
    """Common code between all 3 t-test functions."""
    # We use ``stdtr`` directly here to preserve masked arrays
    # 在所有三种t检验函数之间共享的通用代码部分
    # 这里直接使用`stdtr`来保持masked arrays的特性

    if alternative == 'less':
        pval = special._ufuncs.stdtr(df, t)
    elif alternative == 'greater':
        pval = special._ufuncs.stdtr(df, -t)
    elif alternative == 'two-sided':
        pval = special._ufuncs.stdtr(df, -np.abs(t))*2
    else:
        raise ValueError("alternative must be "
                         "'less', 'greater' or 'two-sided'")
    # 如果数组 t 的维度为 0，将其转换为标量
    if t.ndim == 0:
        t = t[()]
    # 如果数组 pval 的维度为 0，将其转换为标量
    if pval.ndim == 0:
        pval = pval[()]
    
    # 返回处理后的数组 t 和 pval
    return t, pval
# 从一组序列构建二维数组。
# 序列会填充缺失值，以匹配最长序列的长度。

def argstoarray(*args):
    """
    Constructs a 2D array from a group of sequences.

    Sequences are filled with missing values to match the length of the longest
    sequence.

    Parameters
    ----------
    *args : sequences
        Group of sequences.

    Returns
    -------
    argstoarray : MaskedArray
        A ( `m` x `n` ) masked array, where `m` is the number of arguments and
        `n` the length of the longest argument.

    Notes
    -----
    `numpy.ma.vstack` has identical behavior, but is called with a sequence
    of sequences.

    Examples
    --------
    A 2D masked array constructed from a group of sequences is returned.

    >>> from scipy.stats.mstats import argstoarray
    >>> argstoarray([1, 2, 3], [4, 5, 6])
    masked_array(
     data=[[1.0, 2.0, 3.0],
           [4.0, 5.0, 6.0]],
     mask=[[False, False, False],
           [False, False, False]],
     fill_value=1e+20)

    The returned masked array filled with missing values when the lengths of
    sequences are different.

    >>> argstoarray([1, 3], [4, 5, 6])
    masked_array(
     data=[[1.0, 3.0, --],
           [4.0, 5.0, 6.0]],
     mask=[[False, False,  True],
           [False, False, False]],
     fill_value=1e+20)

    """

    # 如果只有一个参数且不是 ndarray，则转换成 MaskedArray 对象
    if len(args) == 1 and not isinstance(args[0], ndarray):
        output = ma.asarray(args[0])
        # 如果输出不是二维数组，抛出异常
        if output.ndim != 2:
            raise ValueError("The input should be 2D")
    else:
        # 否则，计算参数个数和最长序列的长度
        n = len(args)
        m = max([len(k) for k in args])
        # 创建一个指定大小的 MaskedArray 对象，类型为 float，数据部分为空，全部为 mask
        output = ma.array(np.empty((n,m), dtype=float), mask=True)
        # 遍历参数列表，将每个序列的值赋给 output 对应位置
        for (k,v) in enumerate(args):
            output[k,:len(v)] = v

    # 将不是有限数的值设为 mask
    output[np.logical_not(np.isfinite(output._data))] = masked
    return output


def find_repeats(arr):
    """
    Find repeats in arr and return a tuple (repeats, repeat_count).

    The input is cast to float64. Masked values are discarded.

    Parameters
    ----------
    arr : sequence
        Input array. The array is flattened if it is not 1D.

    Returns
    -------
    repeats : ndarray
        Array of repeated values.
    counts : ndarray
        Array of counts.

    Examples
    --------
    >>> from scipy.stats import mstats
    >>> mstats.find_repeats([2, 1, 2, 3, 2, 2, 5])
    (array([2.]), array([4]))

    In the above example, 2 repeats 4 times.

    >>> mstats.find_repeats([[10, 20, 1, 2], [5, 5, 4, 4]])
    (array([4., 5.]), array([2, 2]))

    In the above example, both 4 and 5 repeat 2 times.

    """

    # 确保得到的是一个复制。ma.compressed 承诺一个“新数组”，但实际上可能返回一个引用。
    compr = np.asarray(ma.compressed(arr), dtype=np.float64)
    try:
        need_copy = np.may_share_memory(compr, arr)
    except AttributeError:
        # numpy < 1.8.2 的 bug：np.may_share_memory([], []) 会抛出异常，
        # 而在 numpy 1.8.2 及以上版本中，它会返回 False。
        need_copy = False
    if need_copy:
        compr = compr.copy()
    # 调用内部函数 _find_repeats 处理复制后的数组
    return _find_repeats(compr)
def count_tied_groups(x, use_missing=False):
    """
    Counts the number of tied values.

    Parameters
    ----------
    x : sequence
        Sequence of data on which to counts the ties
    use_missing : bool, optional
        Whether to consider missing values as tied.

    Returns
    -------
    count_tied_groups : dict
        Returns a dictionary (nb of ties: nb of groups).

    Examples
    --------
    >>> from scipy.stats import mstats
    >>> import numpy as np
    >>> z = [0, 0, 0, 2, 2, 2, 3, 3, 4, 5, 6]
    >>> mstats.count_tied_groups(z)
    {2: 1, 3: 2}

    In the above example, the ties were 0 (3x), 2 (3x) and 3 (2x).

    >>> z = np.ma.array([0, 0, 1, 2, 2, 2, 3, 3, 4, 5, 6])
    >>> mstats.count_tied_groups(z)
    {2: 2, 3: 1}
    >>> z[[1,-1]] = np.ma.masked
    >>> mstats.count_tied_groups(z, use_missing=True)
    {2: 2, 3: 1}

    """
    # 计算被遮蔽数据的数量
    nmasked = ma.getmask(x).sum()
    # 复制数据，因为 find_repeats 会修改初始数据
    data = ma.compressed(x).copy()
    # 找出数据中的重复值及其计数
    (ties, counts) = find_repeats(data)
    nties = {}
    # 如果存在重复值
    if len(ties):
        # 创建一个字典，键为计数的唯一值，值为1（表示每组中的计数）
        nties = dict(zip(np.unique(counts), itertools.repeat(1)))
        # 更新字典，将重复值计数对应的计数添加进去
        nties.update(dict(zip(*find_repeats(counts))))

    # 如果存在遮蔽值并且 use_missing 为真
    if nmasked and use_missing:
        try:
            nties[nmasked] += 1
        except KeyError:
            nties[nmasked] = 1

    return nties


def rankdata(data, axis=None, use_missing=False):
    """Returns the rank (also known as order statistics) of each data point
    along the given axis.

    If some values are tied, their rank is averaged.
    If some values are masked, their rank is set to 0 if use_missing is False,
    or set to the average rank of the unmasked values if use_missing is True.

    Parameters
    ----------
    data : sequence
        Input data. The data is transformed to a masked array
    axis : {None,int}, optional
        Axis along which to perform the ranking.
        If None, the array is first flattened. An exception is raised if
        the axis is specified for arrays with a dimension larger than 2
    use_missing : bool, optional
        Whether the masked values have a rank of 0 (False) or equal to the
        average rank of the unmasked values (True).

    """
    def _rank1d(data, use_missing=False):
        # 计算未遮蔽数据的数量
        n = data.count()
        # 创建一个浮点数数组来存储排名
        rk = np.empty(data.size, dtype=float)
        # 对数据排序，返回排序后的索引
        idx = data.argsort()
        # 给未遮蔽数据分配排名
        rk[idx[:n]] = np.arange(1,n+1)

        # 如果 use_missing 为真，给遮蔽数据分配平均排名
        if use_missing:
            rk[idx[n:]] = (n+1)/2.
        else:
            rk[idx[n:]] = 0

        # 找出数据中的重复值
        repeats = find_repeats(data.copy())
        # 对每个重复值的组进行处理
        for r in repeats[0]:
            condition = (data == r).filled(False)
            # 将每组中的数据的排名设置为平均值
            rk[condition] = rk[condition].mean()
        return rk

    # 将输入数据转换为遮蔽数组
    data = ma.array(data, copy=False)
    if axis is None:
        # 如果 axis 为 None，对一维或多维数据进行排名
        if data.ndim > 1:
            return _rank1d(data.ravel(), use_missing).reshape(data.shape)
        else:
            return _rank1d(data, use_missing)
    else:
        # 如果不是第一个情况，执行以下代码块
        return ma.apply_along_axis(_rank1d, axis, data, use_missing).view(ndarray)
# 使用 namedtuple 创建一个名为 ModeResult 的命名元组，包含 mode 和 count 两个字段
ModeResult = namedtuple('ModeResult', ('mode', 'count'))


def mode(a, axis=0):
    """
    Returns an array of the modal (most common) value in the passed array.

    Parameters
    ----------
    a : array_like
        n-dimensional array of which to find mode(s).
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over
        the whole array `a`.

    Returns
    -------
    mode : ndarray
        Array of modal values.
    count : ndarray
        Array of counts for each mode.

    Notes
    -----
    For more details, see `scipy.stats.mode`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> from scipy.stats import mstats
    >>> m_arr = np.ma.array([1, 1, 0, 0, 0, 0], mask=[0, 0, 1, 1, 1, 0])
    >>> mstats.mode(m_arr)  # note that most zeros are masked
    ModeResult(mode=array([1.]), count=array([2.]))

    """
    # 调用内部函数 _mode 来计算模式及其出现次数
    return _mode(a, axis=axis, keepdims=True)


def _mode(a, axis=0, keepdims=True):
    # 不希望从公共的 mstats.mode 函数中暴露 `keepdims`
    # 对输入数组 a 进行数组化，并检查轴向
    a, axis = _chk_asarray(a, axis)

    def _mode1D(a):
        # 寻找数组 a 中重复出现的元素及其次数
        (rep, cnt) = find_repeats(a)
        if not cnt.ndim:
            return (0, 0)
        elif cnt.size:
            # 返回出现次数最多的元素及其次数
            return (rep[cnt.argmax()], cnt.max())
        else:
            return (a.min(), 1)

    if axis is None:
        # 对扁平化后的数组进行一维模式计算
        output = _mode1D(ma.ravel(a))
        output = (ma.array(output[0]), ma.array(output[1]))
    else:
        # 对指定轴向应用 _mode1D 函数
        output = ma.apply_along_axis(_mode1D, axis, a)
        if keepdims is None or keepdims:
            # 如果 keepdims 为 None 或者 True，则保持维度信息
            newshape = list(a.shape)
            newshape[axis] = 1
            slices = [slice(None)] * output.ndim
            slices[axis] = 0
            # 提取模式值并重新塑形
            modes = output[tuple(slices)].reshape(newshape)
            slices[axis] = 1
            # 提取计数值并重新塑形
            counts = output[tuple(slices)].reshape(newshape)
            output = (modes, counts)
        else:
            # 否则移动轴以匹配输出
            output = np.moveaxis(output, axis, 0)

    # 返回 ModeResult 命名元组，包含模式和计数
    return ModeResult(*output)


def _betai(a, b, x):
    # 将 x 转换为 ndarray 或 masked array，处理 x 大于 1 的情况
    x = np.asanyarray(x)
    x = ma.where(x < 1.0, x, 1.0)  # 如果 x > 1，则返回 1.0
    return special.betainc(a, b, x)


def msign(x):
    """Returns the sign of x, or 0 if x is masked."""
    # 返回 x 的符号，如果 x 被掩码，则返回 0
    return ma.filled(np.sign(x), 0)


def pearsonr(x, y):
    r"""
    Pearson correlation coefficient and p-value for testing non-correlation.

    The Pearson correlation coefficient [1]_ measures the linear relationship
    between two datasets.  The calculation of the p-value relies on the
    assumption that each dataset is normally distributed.  (See Kowalski [3]_
    for a discussion of the effects of non-normality of the input on the
    distribution of the correlation coefficient.)  Like other correlation
    coefficients, this one varies between -1 and +1 with 0 implying no
    correlation. Correlations of -1 or +1 imply an exact linear relationship.

    Parameters
    ----------
    x : (N,) array_like
        Input array.
    # y : (N,) array_like
    # 输入数组 y，长度为 N

    # Returns
    # -------
    # r : float
    # Pearson 相关系数
    # p-value : float
    # 双尾 p 值

    # Warns
    # -----
    # `~scipy.stats.ConstantInputWarning`
    # 如果输入是常数数组，则引发此警告。在这种情况下，相关系数未定义，返回 `np.nan`。
    
    # `~scipy.stats.NearConstantInputWarning`
    # 如果输入数组“几乎”是常数，则引发此警告。如果 `norm(x - mean(x)) < 1e-13 * abs(mean(x))`，
    # 则认为数组 x 是几乎常数。在这种情况下，计算 `x - mean(x)` 的数值误差可能导致 r 的计算不准确。

    # See Also
    # --------
    # spearmanr : Spearman 等级相关系数。
    # kendalltau : Kendall's tau，一种用于有序数据的相关度量。

    # Notes
    # -----
    # 相关系数的计算如下所示：
    #
    # .. math::
    #
    #     r = \frac{\sum (x - m_x) (y - m_y)}
    #              {\sqrt{\sum (x - m_x)^2 \sum (y - m_y)^2}}
    #
    # 其中 :math:`m_x` 是向量 x 的均值，:math:`m_y` 是向量 y 的均值。
    #
    # 假设 x 和 y 是独立正态分布的样本（因此总体相关系数为 0），样本相关系数 r 的概率密度函数是：
    #
    # .. math::
    #
    #     f(r) = \frac{{(1-r^2)}^{n/2-2}}{\mathrm{B}(\frac{1}{2},\frac{n}{2}-1)}
    #
    # 其中 n 是样本数，B 是 beta 函数。这有时被称为 r 的精确分布。这是 `pearsonr` 中用于计算 p 值的分布。
    # 该分布是区间 [-1, 1] 上的 beta 分布，具有相等的形状参数 a = b = n/2 - 1。在 SciPy 的 beta 分布实现中，
    # r 的分布是：
    #
    #     dist = scipy.stats.beta(n/2 - 1, n/2 - 1, loc=-1, scale=2)
    #
    # `pearsonr` 返回的 p 值是双尾 p 值。p 值大致指示了无相关系统产生的数据集可能产生与从这些数据集计算得出的 Pearson 相关系数至少一样极端的概率。
    # 更确切地说，对于给定的相关系数 r 的样本，p 值是随机样本 x' 和 y'（从零相关的总体中抽取）的 abs(r') 大于或等于 abs(r) 的概率。
    # 根据上面展示的 `dist` 对象，对于给定的 r 和长度 n，p 值可以计算为：
    #
    #     p = 2 * dist.cdf(-abs(r))
    #
    # 当 n 为 2 时，上述连续分布不是很明确。可以将形状参数 a 和 b 接近于 a = b = 0 的 beta 分布的极限解释为具有相等概率质量在 r = 1 和 r = -1 的离散分布。
    # 更直接地，可以观察到，鉴于数据 x = [x1, x2] 和 y = [y1, y2]，以及
    # 假设 x1 != x2 且 y1 != y2，r 的可能值只有 1 和 -1。因为对于任意长度为 2 的样本 x' 和 y'，
    # 其相关系数的绝对值 abs(r') 始终为 1，因此长度为 2 的样本的双侧 p 值始终为 1。

    # 参考文献
    # ----------
    # .. [1] "Pearson correlation coefficient", Wikipedia,
    #        https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    # .. [2] Student, "Probable error of a correlation coefficient",
    #        Biometrika, Volume 6, Issue 2-3, 1 September 1908, pp. 302-310.
    # .. [3] C. J. Kowalski, "On the Effects of Non-Normality on the Distribution
    #        of the Sample Product-Moment Correlation Coefficient"
    #        Journal of the Royal Statistical Society. Series C (Applied
    #        Statistics), Vol. 21, No. 1 (1972), pp. 1-12.

    # 示例
    # --------
    # >>> import numpy as np
    # >>> from scipy import stats
    # >>> from scipy.stats import mstats
    # >>> mstats.pearsonr([1, 2, 3, 4, 5], [10, 9, 2.5, 6, 4])
    # (-0.7426106572325057, 0.1505558088534455)

    # 如果存在线性依赖关系 y = a + b*x + e，其中 a, b 为常数，e 为独立于 x 的随机误差，
    # 并且假设 x 是标准正态分布，a=0，b=1，e 是均值为零、标准差为 s>0 的正态分布随机变量。

    # >>> s = 0.5
    # >>> x = stats.norm.rvs(size=500)
    # >>> e = stats.norm.rvs(scale=s, size=500)
    # >>> y = x + e
    # >>> mstats.pearsonr(x, y)
    # (0.9029601878969703, 8.428978827629898e-185) # 结果可能有所不同

    # 这个值应该接近于以下精确值

    # >>> 1/np.sqrt(1 + s**2)
    # 0.8944271909999159

    # 对于 s=0.5，我们观察到高度的相关性。一般情况下，噪声的方差增大会降低相关性，
    # 而当误差的方差趋向于零时，相关性接近于 1。

    # 需要注意的是，零相关并不意味着独立，除非 (x, y) 共同服从正态分布。即使是在
    # 非常简单的依赖结构下，相关性也可能为零：例如，如果 X 服从标准正态分布，令 y = abs(x)。
    # 注意 x 和 y 之间的相关性为零。因为 x 的期望是零，cov(x, y) = E[x*y]。
    # 根据定义，这等于 E[x*abs(x)]，由于对称性，这个期望值为零。下面的代码行展示了这一观察：

    # >>> y = np.abs(x)
    # >>> mstats.pearsonr(x, y)
    # (-0.016172891856853524, 0.7182823678751942) # 结果可能有所不同

    # 非零的相关系数可能具有误导性。例如，如果 X 是标准正态分布，定义 y = x if x < 0 else 0。
    # 简单的计算表明 corr(x, y) = sqrt(2/Pi) = 0.797...，表明具有较高的相关性：

    # >>> y = np.where(x < 0, x, 0)
    # >>> mstats.pearsonr(x, y)
    # (0.8537091583771509, 3.183461621422181e-143) # 结果可能有所不同

    # 这是令人困惑的，因为实际上 x 和 y 之间并没有依赖关系，如果 x 较大，则 y = 0。
    """
    计算两个数组 x 和 y 之间的 Pearson 相关系数及其 p 值，考虑遮罩数组。
    如果 x 或 y 包含遮罩值，将其视为缺失数据处理。

    Parameters:
    x : array_like
        输入的第一个数组。
    y : array_like
        输入的第二个数组。
    
    Returns:
    tuple
        包含 Pearson 相关系数和 p 值的元组。

    Notes:
    此函数计算两个数组之间的 Pearson 相关系数及其双尾 p 值。它处理遮罩（masked）数组以排除遮罩值所代表的缺失数据。
    如果自由度 df 小于 0，则返回 (masked, masked)。

    """

    # 检查并调整 x 和 y 的尺寸，以及处理可能的遮罩
    (x, y, n) = _chk_size(x, y)
    # 将 x 和 y 展平处理
    (x, y) = (x.ravel(), y.ravel())
    
    # 获取 x 和 y 的共同遮罩和未被遮罩的元素总数
    m = ma.mask_or(ma.getmask(x), ma.getmask(y))
    n -= m.sum()  # 减去遮罩中的数量，得到有效数据点的总数
    df = n - 2

    if df < 0:
        # 如果自由度小于 0，则返回遮罩的结果
        return (masked, masked)

    # 使用 scipy 中的函数计算 Pearson 相关系数及其 p 值，只考虑未被遮罩的数据点
    return scipy.stats._stats_py.pearsonr(
                ma.masked_array(x, mask=m).compressed(),
                ma.masked_array(y, mask=m).compressed())
# 计算 Spearman 等级相关系数和 p 值，用于检验非相关性。

def spearmanr(x, y=None, use_ties=True, axis=None, nan_policy='propagate',
              alternative='two-sided'):
    """
    计算 Spearman 等级相关系数和 p 值，用于检验非相关性。

    Spearman 相关系数是衡量两个数据集之间线性关系的非参数方法。与 Pearson 相关系数不同，
    Spearman 相关系数不假设数据集服从正态分布。与其他相关系数一样，它的取值范围在 -1 到 +1 之间，
    0 表示无相关性。相关系数为 -1 或 +1 表示单调关系。正相关表示随着 `x` 的增加，`y` 也增加。
    负相关表示随着 `x` 的增加，`y` 减少。

    缺失值会被逐对丢弃：如果 `x` 中的值缺失，则相应的 `y` 值会被掩盖。

    p 值大致指示了一个无关系统产生的数据集，其 Spearman 相关性至少与从这些数据集计算得到的相关性一样极端的概率。
    p 值并非完全可靠，但对于大于约 500 的数据集可能是合理的。

    Parameters
    ----------
    x, y : 1D or 2D array_like，y 是可选的
        包含多个变量和观测值的一个或两个 1-D 或 2-D 数组。当它们是 1-D 时，每个表示一个单变量的观测值向量。
        对于 2-D 情况的行为，请参见下面的 ``axis``。
    use_ties : bool，可选
        不要使用。由于向后兼容性原因，此关键字已被保留但不起作用。
    axis : int 或 None，可选
        如果 axis=0（默认），则每列代表一个变量，行中包含观测值。如果 axis=1，则关系转置：
        每行代表一个变量，而列包含观测值。如果 axis=None，则两个数组都会被展平。
    nan_policy : {'propagate', 'raise', 'omit'}，可选
        定义输入包含 NaN 时如何处理。'propagate' 返回 NaN，'raise' 抛出错误，'omit' 忽略 NaN 进行计算。默认为 'propagate'。
    alternative : {'two-sided', 'less', 'greater'}，可选
        定义备择假设。默认为 'two-sided'。
        提供以下选项：

        * 'two-sided': 相关性非零
        * 'less': 相关性为负（小于零）
        * 'greater': 相关性为正（大于零）

        .. versionadded:: 1.7.0

    Returns
    -------
    res : SignificanceResult
        # 定义变量 res，类型为 SignificanceResult，表示显著性测试的结果对象

        statistic : float or ndarray (2-D square)
            # statistic 属性，可以是 float 或者 2-D 方阵 ndarray
            # 表示 Spearman 相关系数矩阵或者相关系数（如果只有两个变量作为参数）
            # 相关系数矩阵是一个方阵，其长度等于 `a` 和 `b` 组合后的总变量数（列数或行数）。

        pvalue : float
            # pvalue 属性，表示假设检验的 p 值
            # 假设检验的零假设是两组数据线性不相关。
            # 详情见上述的 `alternative` 参数说明。
            # `pvalue` 的形状与 `statistic` 相同。

    References
    ----------
    [CRCProbStat2000] section 14.7
    # 参考文献引用，指向 CRCProbStat2000 第 14.7 节
    # 提供了关于 Spearman 相关性的详细信息和背景资料
    """
    if not use_ties:
        # 如果 use_ties 参数为 False，则抛出 ValueError 异常
        raise ValueError("`use_ties=False` is not supported in SciPy >= 1.2.0")

    # Always returns a masked array, raveled if axis=None
    # 总是返回一个掩码数组，如果 axis=None，则展平
    x, axisout = _chk_asarray(x, axis)

    if y is not None:
        # 如果 y 不为 None，则处理 2-D `x` 的情况
        y, _ = _chk_asarray(y, axis)
        if axisout == 0:
            # 如果 axisout 为 0，则将 y 与 x 列堆叠起来
            x = ma.column_stack((x, y))
        else:
            # 否则，将 y 与 x 行堆叠起来
            x = ma.vstack((x, y))

    if axisout == 1:
        # 如果 axisout 为 1，则转置 x
        # 以简化后续代码（始终使用 `n_obs, n_vars` 的形状）
        x = x.T

    if nan_policy == 'omit':
        # 如果 nan_policy 设置为 'omit'，则掩盖无效数据
        x = ma.masked_invalid(x)

    def _spearmanr_2cols(x):
        # 定义内部函数 _spearmanr_2cols，计算两列数据的 Spearman 相关性

        # 掩盖所有变量的相同观测值，并丢弃这些观测值（不能保留掩码，rankdata 函数会出错）
        x = ma.mask_rowcols(x, axis=0)
        x = x[~x.mask.any(axis=1), :]

        # 如果任一列完全为 NaN 或 Inf
        if not np.any(x.data):
            # 返回一个具有 NaN 的 SignificanceResult 对象
            res = scipy.stats._stats_py.SignificanceResult(np.nan, np.nan)
            res.correlation = np.nan
            return res

        m = ma.getmask(x)
        n_obs = x.shape[0]
        dof = n_obs - 2 - int(m.sum(axis=0)[0])
        if dof < 0:
            # 如果自由度小于 0，则抛出 ValueError 异常
            raise ValueError("The input must have at least 3 entries!")

        # 获取排名和排名差异
        x_ranked = rankdata(x, axis=0)
        rs = ma.corrcoef(x_ranked, rowvar=False).data

        # rs 可能有元素等于 1，因此避免零除警告
        with np.errstate(divide='ignore'):
            # 在取平方根之前，修剪可能由于舍入误差导致的小负值
            t = rs * np.sqrt((dof / ((rs+1.0) * (1.0-rs))).clip(0))

        t, prob = _ttest_finish(dof, t, alternative)

        # 为了向后兼容，当比较两列时返回标量
        if rs.shape == (2, 2):
            # 返回一个具有相关系数和 p 值的 SignificanceResult 对象
            res = scipy.stats._stats_py.SignificanceResult(rs[1, 0],
                                                           prob[1, 0])
            res.correlation = rs[1, 0]
            return res
        else:
            # 返回一个具有相关系数矩阵和 p 值的 SignificanceResult 对象
            res = scipy.stats._stats_py.SignificanceResult(rs, prob)
            res.correlation = rs
            return res
    # 针对每一对变量执行以下操作，否则第三列中的被丢弃的观测会影响对一对变量的结果。
    # 获取数据矩阵 x 的列数
    n_vars = x.shape[1]
    
    # 如果变量数为2，直接调用 _spearmanr_2cols 函数处理
    if n_vars == 2:
        return _spearmanr_2cols(x)
    else:
        # 初始化相关系数矩阵和 p 值矩阵
        rs = np.ones((n_vars, n_vars), dtype=float)
        prob = np.zeros((n_vars, n_vars), dtype=float)
        
        # 遍历每对变量的索引
        for var1 in range(n_vars - 1):
            for var2 in range(var1 + 1, n_vars):
                # 调用 _spearmanr_2cols 函数计算两列变量的 Spearman 相关系数和 p 值
                result = _spearmanr_2cols(x[:, [var1, var2]])
                
                # 将计算结果填充到相关系数矩阵和 p 值矩阵中对应位置
                rs[var1, var2] = result.correlation
                rs[var2, var1] = result.correlation
                prob[var1, var2] = result.pvalue
                prob[var2, var1] = result.pvalue

        # 构造一个 SignificanceResult 对象，将相关系数矩阵和 p 值矩阵作为参数
        res = scipy.stats._stats_py.SignificanceResult(rs, prob)
        # 将计算得到的相关系数矩阵保存在结果对象中
        res.correlation = rs
        # 返回结果对象
        return res
def _kendall_p_exact(n, c, alternative='two-sided'):
    # 根据分布的对称性，始终计算左尾部的累积分布函数（CDF）。
    # 如果 `c` 处于备择假设预测的空值分布的一侧，则这将是单侧 p 值。
    # 双侧 p 值将是该值的两倍。
    # 如果 `c` 处于空值分布的另一侧，我们将需要取补集并添加回 `c` 处的概率质量。
    in_right_tail = (c >= (n*(n-1))//2 - c)
    alternative_greater = (alternative == 'greater')
    c = int(min(c, (n*(n-1))//2 - c))  # 将 `c` 限制在有效范围内

    # 精确的 p 值，参考 Maurice G. Kendall 的 "Rank Correlation Methods"
    # (第四版)，Charles Griffin & Co.，1970年。
    if n <= 0:
        raise ValueError(f'n ({n}) must be positive')  # n 必须为正数
    elif c < 0 or 4*c > n*(n-1):
        raise ValueError(f'c ({c}) must satisfy 0 <= 4c <= n(n-1) = {n*(n-1)}.')  # c 必须满足的条件
    elif n == 1:
        prob = 1.0
        p_mass_at_c = 1  # 在 c 处的概率质量
    elif n == 2:
        prob = 1.0
        p_mass_at_c = 0.5  # 在 c 处的概率质量
    elif c == 0:
        prob = 2.0/math.factorial(n) if n < 171 else 0.0
        p_mass_at_c = prob/2  # 在 c 处的概率质量
    elif c == 1:
        prob = 2.0/math.factorial(n-1) if n < 172 else 0.0
        p_mass_at_c = (n-1)/math.factorial(n)  # 在 c 处的概率质量
    elif 4*c == n*(n-1) and alternative == 'two-sided':
        # 如果是双侧检验，这种情况下 p_mass_at_c 有一个简单的公式，但我不知道它。
        # 使用单侧 p 值的通用公式。
        prob = 1.0
    elif n < 171:
        new = np.zeros(c+1)
        new[0:2] = 1.0
        for j in range(3,n+1):
            new = np.cumsum(new)
            if j <= c:
                new[j:] -= new[:c+1-j]
        prob = 2.0*np.sum(new)/math.factorial(n)  # 计算概率
        p_mass_at_c = new[-1]/math.factorial(n)  # 在 c 处的概率质量
    else:
        new = np.zeros(c+1)
        new[0:2] = 1.0
        for j in range(3, n+1):
            new = np.cumsum(new)/j
            if j <= c:
                new[j:] -= new[:c+1-j]
        prob = np.sum(new)  # 计算概率
        p_mass_at_c = new[-1]/2  # 在 c 处的概率质量

    if alternative != 'two-sided':
        # 如果备择假设和 alternative 一致，单侧 p 值是双侧 p 值的一半
        if in_right_tail == alternative_greater:
            prob /= 2
        else:
            prob = 1 - prob/2 + p_mass_at_c  # 调整单侧 p 值

    prob = np.clip(prob, 0, 1)  # 将概率值裁剪在 [0, 1] 范围内

    return prob
    # 定义参数 method 用于指定计算 p 值的方法，可选值为 'auto', 'asymptotic', 'exact'
    # 'asymptotic' 使用大样本情况下的正态近似计算 p 值
    # 'exact' 计算精确的 p 值，但只能在不存在绑定的情况下使用，随着样本大小增加，'exact' 的计算时间可能增长并且结果可能失去一些精度
    # 'auto' 是默认值，根据速度和准确性的权衡选择适当的方法
    alternative : {'two-sided', 'less', 'greater'}, optional
        # 定义备择假设，默认为 'two-sided'
        # 可选项包括:
        # * 'two-sided': 排名相关性非零
        # * 'less': 排名相关性为负（小于零）
        # * 'greater': 排名相关性为正（大于零）

    Returns
    -------
    res : SignificanceResult
        # 返回一个包含以下属性的对象:
        # statistic : float
        #    tau 统计量。
        # pvalue : float
        #    用于假设检验的 p 值，其零假设为不存在关联，即 tau = 0。

    References
    ----------
    .. [1] Maurice G. Kendall, "Rank Correlation Methods" (4th Edition),
           Charles Griffin & Co., 1970.

    """
    # 检查并调整 x, y 的大小
    (x, y, n) = _chk_size(x, y)
    # 将 x, y 展平
    (x, y) = (x.flatten(), y.flatten())
    # 创建 x 和 y 的掩码
    m = ma.mask_or(ma.getmask(x), ma.getmask(y))
    # 如果掩码不为空，则使用掩码创建新的 masked array
    if m is not nomask:
        x = ma.array(x, mask=m, copy=True)
        y = ma.array(y, mask=m, copy=True)
        # 需要在此处使用 int()，否则 numpy 在所有 Windows 架构上默认使用 32 位整数，可能会导致溢出
        # int() 将保持无限精度
        n -= int(m.sum())

    # 如果样本大小小于 2，则返回 NaN 的结果对象
    if n < 2:
        res = scipy.stats._stats_py.SignificanceResult(np.nan, np.nan)
        res.correlation = np.nan
        return res

    # 对 x, y 进行排名，并处理可能存在的缺失值
    rx = ma.masked_equal(rankdata(x, use_missing=use_missing), 0)
    ry = ma.masked_equal(rankdata(y, use_missing=use_missing), 0)
    idx = rx.argsort()
    (rx, ry) = (rx[idx], ry[idx])

    # 计算 Kendall's tau 统计量的组成部分 C 和 D
    C = np.sum([((ry[i+1:] > ry[i]) * (rx[i+1:] > rx[i])).filled(0).sum()
                for i in range(len(ry)-1)], dtype=float)
    D = np.sum([((ry[i+1:] < ry[i])*(rx[i+1:] > rx[i])).filled(0).sum()
                for i in range(len(ry)-1)], dtype=float)

    # 计算 x, y 中的绑定组数
    xties = count_tied_groups(x)
    yties = count_tied_groups(y)

    # 如果 use_ties 为 True，则计算修正项 corr_x 和 corr_y，并计算分母 denom
    if use_ties:
        corr_x = np.sum([v*k*(k-1) for (k,v) in xties.items()], dtype=float)
        corr_y = np.sum([v*k*(k-1) for (k,v) in yties.items()], dtype=float)
        denom = ma.sqrt((n*(n-1)-corr_x)/2. * (n*(n-1)-corr_y)/2.)
    else:
        denom = n*(n-1)/2.

    # 计算 Kendall's tau 统计量
    tau = (C-D) / denom

    # 如果 method 为 'exact' 并且存在绑定，则引发 ValueError
    if method == 'exact' and (xties or yties):
        raise ValueError("Ties found, exact method cannot be used.")
    # 如果方法为 'auto'，根据条件选择合适的计算方法
    if method == 'auto':
        # 检查是否存在重复值并且满足条件，或者样本大小小于等于 33 或 C 值满足特定条件时，选择精确方法
        if (not xties and not yties) and (n <= 33 or min(C, n*(n-1)/2.0-C) <= 1):
            method = 'exact'
        else:
            # 否则选择渐近方法
            method = 'asymptotic'

    # 如果不存在重复值并且方法为 'exact'，则使用精确的肯德尔 tau 计算
    if not xties and not yties and method == 'exact':
        prob = _kendall_p_exact(n, C, alternative)

    # 如果方法为 'asymptotic'，则使用渐近方法计算
    elif method == 'asymptotic':
        # 计算渐近方法中的方差估计
        var_s = n*(n-1)*(2*n+5)
        
        # 如果使用重复值（ties），进行相应的调整
        if use_ties:
            var_s -= np.sum([v*k*(k-1)*(2*k+5)*1. for (k,v) in xties.items()])
            var_s -= np.sum([v*k*(k-1)*(2*k+5)*1. for (k,v) in yties.items()])
            
            v1 = (np.sum([v*k*(k-1) for (k, v) in xties.items()], dtype=float) *
                  np.sum([v*k*(k-1) for (k, v) in yties.items()], dtype=float))
            v1 /= 2.*n*(n-1)
            
            # 如果样本大小大于 2，则计算第二个调整项
            if n > 2:
                v2 = np.sum([v*k*(k-1)*(k-2) for (k,v) in xties.items()],
                            dtype=float) * \
                     np.sum([v*k*(k-1)*(k-2) for (k,v) in yties.items()],
                            dtype=float)
                v2 /= 9.*n*(n-1)*(n-2)
            else:
                v2 = 0
        else:
            v1 = v2 = 0
        
        # 最终计算方差
        var_s /= 18.
        var_s += (v1 + v2)
        
        # 计算 z 值
        z = (C-D)/np.sqrt(var_s)
        
        # 根据 z 值计算 p 值
        prob = scipy.stats._stats_py._get_pvalue(z, distributions.norm, alternative)
    
    # 如果方法未知，则抛出 ValueError 异常
    else:
        raise ValueError("Unknown method "+str(method)+" specified, please "
                         "use auto, exact or asymptotic.")
    
    # 构造结果对象
    res = scipy.stats._stats_py.SignificanceResult(tau[()], prob[()])
    res.correlation = tau
    # 返回计算结果
    return res
def pointbiserialr(x, y):
    """Calculates a point biserial correlation coefficient and its p-value.

    Parameters
    ----------
    x : array_like of bools
        Input array.
    y : array_like
        Input array.

    Returns
    -------
    correlation : float
        R value
    pvalue : float
        2-tailed p-value

    Notes
    -----
    Missing values are considered pair-wise: if a value is missing in x,
    the corresponding value in y is masked.

    For more details on `pointbiserialr`, see `scipy.stats.pointbiserialr`.

    """
    # 将输入数组 x 中的无效值修正为对应的掩码值，并转换为布尔型
    x = ma.fix_invalid(x, copy=True).astype(bool)
    # 修复 y 中的无效数据，并将其转换为浮点型
    y = ma.fix_invalid(y, copy=True).astype(float)
    # 获取 x 的掩码（即标记哪些数据是无效的）
    m = ma.mask_or(ma.getmask(x), ma.getmask(y))
    # 如果掩码 m 不是空掩码（nomask），则执行下面的操作
    if m is not nomask:
        # unmask 是 x 的有效数据的布尔值掩码（反转 m 的结果）
        unmask = np.logical_not(m)
        # 根据 unmask 过滤出有效的 x 和 y 数据
        x = x[unmask]
        y = y[unmask]

    # 计算有效 x 的数量
    n = len(x)
    # phat 是 x 中为 True 的比例
    phat = x.sum() / float(n)
    # y0 包含 x 为 False 时对应的 y 值
    y0 = y[~x]
    # y1 包含 x 为 True 时对应的 y 值
    y1 = y[x]
    # 计算 y0 和 y1 的均值
    y0m = y0.mean()
    y1m = y1.mean()

    # 计算点二列相关系数（Point-biserial correlation coefficient）
    rpb = (y1m - y0m) * np.sqrt(phat * (1 - phat)) / y.std()

    # 计算自由度
    df = n - 2
    # 计算 t 值
    t = rpb * ma.sqrt(df / (1.0 - rpb**2))
    # 计算概率值
    prob = _betai(0.5 * df, 0.5, df / (df + t * t))

    # 返回点二列相关系数分析的结果对象
    return PointbiserialrResult(rpb, prob)
# 定义线性回归函数，用于计算两组测量数据的最小二乘线性回归

Parameters
----------
x, y : array_like
    两组测量数据。这两个数组应具有相同的长度 N。如果只提供 `x`（且 ``y=None``），则 `x` 必须是一个二维数组，其中一维长度为 2。则根据长度为 2 的维度将数组拆分为两组测量数据。
    当 ``y=None`` 且 `x` 是一个 2xN 数组时，``linregress(x)`` 等同于 ``linregress(x[0], x[1])``。

Returns
-------
result : ``LinregressResult`` 实例
    返回值是一个对象，具有以下属性：

    slope : float
        回归线的斜率。
    intercept : float
        回归线的截距。
    rvalue : float
        Pearson 相关系数。``rvalue`` 的平方等于确定系数（coefficient of determination）。
    pvalue : float
        斜率为零的假设检验的 p 值，使用 Wald 检验和 t 分布的检验统计量。参见 `alternative` 上面的备择假设。
    stderr : float
        估计斜率（梯度）的标准误差，假设残差服从正态分布。
    intercept_stderr : float
        估计截距的标准误差，假设残差服从正态分布。

See Also
--------
scipy.optimize.curve_fit :
    使用非线性最小二乘拟合数据到函数。
scipy.optimize.leastsq :
    最小化一组方程的平方和。

Notes
-----
缺失值被视为成对处理：如果 `x` 中的值缺失，则对应的 `y` 值被屏蔽。

为了与旧版 SciPy 兼容，返回值可以像长度为 5 的命名元组一样使用，具有字段 ``slope``, ``intercept``, ``rvalue``, ``pvalue`` 和 ``stderr``，因此可以继续编写::

    slope, intercept, r, p, se = linregress(x, y)

然而，这种风格下截距的标准误差不可用。为了访问所有计算值，包括截距的标准误差，应将返回值作为带有属性的对象使用，例如::

    result = linregress(x, y)
    print(result.intercept, result.intercept_stderr)

Examples
--------
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from scipy import stats
>>> rng = np.random.default_rng()

生成一些数据：

>>> x = rng.random(10)
>>> y = 1.6*x + rng.random(10)

执行线性回归：

>>> res = stats.mstats.linregress(x, y)

确定系数（R-squared）：

>>> print(f"R-squared: {res.rvalue**2:.6f}")
    R-squared: 0.717533

    Plot the data along with the fitted line:

    >>> plt.plot(x, y, 'o', label='original data')  # 绘制原始数据散点图
    >>> plt.plot(x, res.intercept + res.slope*x, 'r', label='fitted line')  # 绘制拟合直线
    >>> plt.legend()  # 显示图例
    >>> plt.show()  # 显示图形

    Calculate 95% confidence interval on slope and intercept:

    >>> # 双侧学生 t 分布的反函数
    >>> # p - 概率, df - 自由度
    >>> from scipy.stats import t
    >>> tinv = lambda p, df: abs(t.ppf(p/2, df))

    >>> ts = tinv(0.05, len(x)-2)  # 计算 t 分布的临界值
    >>> print(f"slope (95%): {res.slope:.6f} +/- {ts*res.stderr:.6f}")  # 输出斜率的95%置信区间
    slope (95%): 1.453392 +/- 0.743465
    >>> print(f"intercept (95%): {res.intercept:.6f}"
    ...       f" +/- {ts*res.intercept_stderr:.6f}")  # 输出截距的95%置信区间
    intercept (95%): 0.616950 +/- 0.544475

    """
    if y is None:
        x = ma.array(x)  # 将 x 转换为掩码数组
        if x.shape[0] == 2:
            x, y = x  # 如果 x 是形状为 (2, N) 的数组，则将其拆分为 x 和 y
        elif x.shape[1] == 2:
            x, y = x.T  # 如果 x 是形状为 (N, 2) 的数组，则将其转置并拆分为 x 和 y
        else:
            raise ValueError("If only `x` is given as input, "
                             "it has to be of shape (2, N) or (N, 2), "
                             f"provided shape was {x.shape}")  # 如果 x 不符合预期形状，则引发 ValueError
    else:
        x = ma.array(x)  # 将 x 转换为掩码数组
        y = ma.array(y)  # 将 y 转换为掩码数组

    x = x.flatten()  # 将 x 展平为一维数组
    y = y.flatten()  # 将 y 展平为一维数组

    if np.amax(x) == np.amin(x) and len(x) > 1:
        raise ValueError("Cannot calculate a linear regression "
                         "if all x values are identical")  # 如果所有 x 值相同且长度大于 1，则无法计算线性回归，引发 ValueError

    m = ma.mask_or(ma.getmask(x), ma.getmask(y), shrink=False)  # 计算掩码的逻辑或
    if m is not nomask:
        x = ma.array(x, mask=m)  # 根据掩码 m 屏蔽 x 数组
        y = ma.array(y, mask=m)  # 根据掩码 m 屏蔽 y 数组
        if np.any(~m):
            result = _stats_py.linregress(x.data[~m], y.data[~m])  # 计算非屏蔽数据的线性回归
        else:
            # All data is masked
            result = _stats_py.LinregressResult(slope=None, intercept=None,
                                                rvalue=None, pvalue=None,
                                                stderr=None,
                                                intercept_stderr=None)  # 如果所有数据均被屏蔽，则返回空的线性回归结果
    else:
        result = _stats_py.linregress(x.data, y.data)  # 计算 x 和 y 的线性回归

    return result  # 返回计算结果
# 计算 Theil-Sen 估计器，用于一组点 (x, y) 的稳健线性回归
def theilslopes(y, x=None, alpha=0.95, method='separate'):
    """
    Computes the Theil-Sen estimator for a set of points (x, y).

    `theilslopes` implements a method for robust linear regression.  It
    computes the slope as the median of all slopes between paired values.

    Parameters
    ----------
    y : array_like
        Dependent variable.
    x : array_like or None, optional
        Independent variable. If None, use ``arange(len(y))`` instead.
    alpha : float, optional
        Confidence degree between 0 and 1. Default is 95% confidence.
        Note that `alpha` is symmetric around 0.5, i.e. both 0.1 and 0.9 are
        interpreted as "find the 90% confidence interval".
    method : {'joint', 'separate'}, optional
        Method to be used for computing estimate for intercept.
        Following methods are supported,

            * 'joint': Uses np.median(y - slope * x) as intercept.
            * 'separate': Uses np.median(y) - slope * np.median(x)
                          as intercept.

        The default is 'separate'.

        .. versionadded:: 1.8.0

    Returns
    -------
    result : ``TheilslopesResult`` instance
        The return value is an object with the following attributes:

        slope : float
            Theil slope.
        intercept : float
            Intercept of the Theil line.
        low_slope : float
            Lower bound of the confidence interval on `slope`.
        high_slope : float
            Upper bound of the confidence interval on `slope`.

    See Also
    --------
    siegelslopes : a similar technique using repeated medians


    Notes
    -----
    For more details on `theilslopes`, see `scipy.stats.theilslopes`.

    """
    # 将 y 转换为 MaskedArray 对象，并展平
    y = ma.asarray(y).flatten()
    # 如果 x 为 None，则使用 arange(len(y)) 作为独立变量
    if x is None:
        x = ma.arange(len(y), dtype=float)
    else:
        # 将 x 转换为 MaskedArray 对象，并展平
        x = ma.asarray(x).flatten()
        # 如果 x 和 y 的长度不匹配，则引发 ValueError 异常
        if len(x) != len(y):
            raise ValueError(f"Incompatible lengths ! ({len(y)}<>{len(x)})")

    # 合并 x 和 y 的掩码
    m = ma.mask_or(ma.getmask(x), ma.getmask(y))
    # 应用掩码到 y 和 x
    y._mask = x._mask = m
    # 去除任何被掩码遮挡的 x 或 y 的元素
    y = y.compressed()
    x = x.compressed().astype(float)
    # 现在我们有未被掩码遮挡的数组，可以使用 `scipy.stats.theilslopes` 函数
    return stats_theilslopes(y, x, alpha=alpha, method=method)
    # 将 y 转换为 MaskedArray 并展开成一维数组
    y = ma.asarray(y).ravel()
    
    # 如果 x 为 None，则创建一个浮点类型的 MaskedArray 作为默认 x
    if x is None:
        x = ma.arange(len(y), dtype=float)
    else:
        # 否则，将 x 转换为 MaskedArray 并展开成一维数组
        x = ma.asarray(x).ravel()
        # 如果 x 和 y 的长度不一致，则抛出 ValueError 异常
        if len(x) != len(y):
            raise ValueError(f"Incompatible lengths ! ({len(y)}<>{len(x)})")
    
    # 合并 x 和 y 的掩码
    m = ma.mask_or(ma.getmask(x), ma.getmask(y))
    # 将 y 和 x 的掩码设置为合并后的掩码
    y._mask = x._mask = m
    
    # 去除 x 和 y 中掩码对应的元素
    y = y.compressed()
    x = x.compressed().astype(float)
    
    # 现在 x 和 y 都是非掩码的数组，可以使用 `scipy.stats.siegelslopes` 函数进行计算
    # 使用 siegelslopes 函数计算斜率和截距，根据指定的方法（hierarchical 或 separate）
    return stats_siegelslopes(y, x, method=method)
# 创建一个名为 SenSeasonalSlopesResult 的命名元组，包含两个字段：'intra_slope' 和 'inter_slope'
SenSeasonalSlopesResult = _make_tuple_bunch('SenSeasonalSlopesResult',
                                            ['intra_slope', 'inter_slope'])


def sen_seasonal_slopes(x):
    r"""
    计算季节性 Theil-Sen 和 Kendall 斜率估计值。

    Sen 斜率的季节性泛化计算二维数组的每个“季节”（列）中所有值对之间的斜率。它返回一个数组，
    其中包含每个“季节”的这些“季节内”斜率的中值（每个季节的 Theil-Sen 斜率估计值），并返回跨
    所有季节的“季节内”斜率的中值（季节 Kendall 斜率估计值）。

    Parameters
    ----------
    x : 2D array_like
        `x` 的每一列包含依赖变量的测量值，假设每个季节的自变量（通常是时间）为 ``np.arange(x.shape[0])``。

    Returns
    -------
    result : ``SenSeasonalSlopesResult`` 实例
        返回值是一个对象，具有以下属性：

        intra_slope : ndarray
            每个季节的 Theil-Sen 斜率估计值：季节内斜率的中值。
        inter_slope : float
            季节 Kendall 斜率估计值：所有季节的季节内斜率的中值。

    See Also
    --------
    theilslopes : 针对非季节性数据的类似函数
    scipy.stats.theilslopes : 针对非遮罩数组的非季节性斜率

    Notes
    -----
    季节内斜率 :math:`d_{ijk}` 在季节 :math:`i` 中的定义为：

    .. math::

        d_{ijk} = \frac{x_{ij} - x_{ik}}
                            {j - k}

    其中 :math:`j, k` 是 :math:`x` 的不同整数索引对。

    返回的 `intra_slope` 数组的第 :math:`i` 元素是所有 :math:`j < k` 的 :math:`d_{ijk}` 的中值，
    这是季节 :math:`i` 的 Theil-Sen 斜率估计值。返回的 `inter_slope` 值，也称为季节 Kendall 斜率估计值，
    是所有 :math:`i, j, k` 的 :math:`d_{ijk}` 的中值。

    References
    ----------
    .. [1] Hirsch, Robert M., James R. Slack, and Richard A. Smith.
           "Techniques of trend analysis for monthly water quality data."
           *Water Resources Research* 18.1 (1982): 107-121.

    Examples
    --------
    假设我们有每个季节 100 个观测值的情况，共四个季节：

    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> x = rng.random(size=(100, 4))

    我们计算季节斜率如下：

    >>> from scipy import stats
    >>> intra_slope, inter_slope = stats.mstats.sen_seasonal_slopes(x)

    如果我们定义一个函数来计算季节内观测值之间的所有斜率：

    >>> def dijk(yi):
    ...     n = len(yi)
    ...     x = np.arange(n)
    ...     dy = yi - yi[:, np.newaxis]
    ...     dx = x - x[:, np.newaxis]
    # 我们只希望得到不同索引对的唯一组合
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    # 创建一个上三角形状的布尔掩码矩阵，用于选择不同索引对
    return dy[mask]/dx[mask]

    then element ``i`` of ``intra_slope`` is the median of ``dijk[x[:, i]]``:

    >>> i = 2
    >>> np.allclose(np.median(dijk(x[:, i])), intra_slope[i])
    True

    and ``inter_slope`` is the median of the values returned by ``dijk`` for
    all seasons:

    >>> all_slopes = np.concatenate([dijk(x[:, i]) for i in range(x.shape[1])])
    >>> np.allclose(np.median(all_slopes), inter_slope)
    True

    Because the data are randomly generated, we would expect the median slopes
    to be nearly zero both within and across all seasons, and indeed they are:

    >>> intra_slope.data
    array([ 0.00124504, -0.00277761, -0.00221245, -0.00036338])
    >>> inter_slope
    -0.0010511779872922058

    """
    # 将数组 x 转换为掩码数组（如果需要），确保其至少为二维
    x = ma.array(x, subok=True, copy=False, ndmin=2)
    # 获取数组 x 的形状信息，n 为行数
    (n,_) = x.shape
    # 获取每个季节斜率的列表
    szn_slopes = ma.vstack([(x[i+1:]-x[i])/np.arange(1,n-i)[:,None]
                            for i in range(n)])
    # 计算每个季节斜率的中位数
    szn_medslopes = ma.median(szn_slopes, axis=0)
    # 计算所有季节斜率的全局中位数
    medslope = ma.median(szn_slopes, axis=None)
    # 返回季节斜率分析的结果对象
    return SenSeasonalSlopesResult(szn_medslopes, medslope)
# 命名元组，用于存储 ttest_1samp 函数的返回结果，包括统计量和 p 值
Ttest_1sampResult = namedtuple('Ttest_1sampResult', ('statistic', 'pvalue'))

# 定义计算单样本 T 检验的函数
def ttest_1samp(a, popmean, axis=0, alternative='two-sided'):
    """
    计算单组分数的 T 检验。

    Parameters
    ----------
    a : array_like
        样本观测值
    popmean : float or array_like
        在零假设中的期望值，如果是 array_like，则必须与 `a` 除了轴维度外具有相同的形状
    axis : int or None, optional
        计算检验的轴。如果为 None，则在整个数组 `a` 上计算。
    alternative : {'two-sided', 'less', 'greater'}, optional
        定义备择假设。
        可用的选项有（默认为 'two-sided'）：

        * 'two-sided': 样本的基础分布的均值与给定的总体均值 (`popmean`) 不同
        * 'less': 样本的基础分布的均值小于给定的总体均值 (`popmean`)
        * 'greater': 样本的基础分布的均值大于给定的总体均值 (`popmean`)

        .. versionadded:: 1.7.0

    Returns
    -------
    statistic : float or array
        t 统计量
    pvalue : float or array
        p 值

    Notes
    -----
    更多关于 `ttest_1samp` 的细节，请参阅 `scipy.stats.ttest_1samp`。

    """
    # 将输入 `a` 转换为数组并检查轴
    a, axis = _chk_asarray(a, axis)
    # 如果数组为空，则返回 (NaN, NaN)
    if a.size == 0:
        return (np.nan, np.nan)

    # 计算样本均值
    x = a.mean(axis=axis)
    # 计算样本方差
    v = a.var(axis=axis, ddof=1)
    # 计算样本大小
    n = a.count(axis=axis)
    # 将 df 强制转换为数组以避免遮蔽的除法引发警告
    df = ma.asanyarray(n - 1.0)
    # 计算 svar，避免除以 0 的警告
    svar = ((n - 1.0) * v) / df
    # 忽略除以 0 和无效值的错误状态
    with np.errstate(divide='ignore', invalid='ignore'):
        # 计算 t 统计量
        t = (x - popmean) / ma.sqrt(svar / n)

    # 调用 _ttest_finish 函数完成 t 检验，并返回统计量和 p 值
    t, prob = _ttest_finish(df, t, alternative)
    return Ttest_1sampResult(t, prob)


# 将 ttest_1samp 函数别名为 ttest_onesamp
ttest_onesamp = ttest_1samp


# 命名元组，用于存储 ttest_ind 函数的返回结果，包括统计量和 p 值
Ttest_indResult = namedtuple('Ttest_indResult', ('statistic', 'pvalue'))

# 定义计算独立样本 T 检验的函数
def ttest_ind(a, b, axis=0, equal_var=True, alternative='two-sided'):
    """
    计算两组独立样本分数的 T 检验。

    Parameters
    ----------
    a, b : array_like
        数组必须具有相同的形状，除了与 `axis` 对应的维度（默认是第一个维度）。
    axis : int or None, optional
        计算检验的轴。如果为 None，则在整个数组 `a` 和 `b` 上计算。
    equal_var : bool, optional
        如果为 True，则执行假设等方差的标准独立两样本检验。
        如果为 False，则执行不假设等方差的 Welch's t 检验。

        .. versionadded:: 0.17.0
    """
    Perform independent two-sample t-test.

    Parameters
    ----------
    a, b : array_like
        The arrays must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default).
    axis : int or None, optional
        Axis along which to compute test; default is 0. If None, compute
        over the whole arrays `a` and `b`.
    equal_var : bool, optional
        If True (default), perform standard independent 2 sample test that
        assumes equal population variances [1]_.
        If False, perform Welch's t-test, which does not assume equal
        population variance [2]_.

    Returns
    -------
    statistic : float or array
        The calculated t-statistic.
    pvalue : float or array
        The p-value.

    Notes
    -----
    For more details on `ttest_ind`, see `scipy.stats.ttest_ind`.
    """
    # 将输入的a和b转换为数组，并检查维度
    a, b, axis = _chk2_asarray(a, b, axis)

    # 如果任一数组为空，则直接返回NaN结果
    if a.size == 0 or b.size == 0:
        return Ttest_indResult(np.nan, np.nan)

    # 分别计算两个样本的均值
    (x1, x2) = (a.mean(axis), b.mean(axis))
    
    # 分别计算两个样本的方差
    (v1, v2) = (a.var(axis=axis, ddof=1), b.var(axis=axis, ddof=1))
    
    # 分别获取两个样本的样本量
    (n1, n2) = (a.count(axis), b.count(axis))

    # 如果使用等方差性假设
    if equal_var:
        # 计算等方差情况下的自由度
        df = ma.asanyarray(n1 + n2 - 2.0)
        # 计算 pooled variance
        svar = ((n1-1)*v1 + (n2-1)*v2) / df
        # 计算 t 统计量的分母
        denom = ma.sqrt(svar * (1.0/n1 + 1.0/n2))  # n-D computation here!
    else:
        # 计算 Welch's t-test 的自由度
        vn1 = v1 / n1
        vn2 = v2 / n2
        with np.errstate(divide='ignore', invalid='ignore'):
            df = (vn1 + vn2)**2 / (vn1**2 / (n1 - 1) + vn2**2 / (n2 - 1))

        # 处理 df 为 undefined 的情况，即方差为零的情况
        df = np.where(np.isnan(df), 1, df)
        # 计算 Welch's t-test 的分母
        denom = ma.sqrt(vn1 + vn2)

    # 计算 t 统计量
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (x1 - x2) / denom

    # 调用辅助函数计算 t 统计量和 p 值，并返回结果
    t, prob = _ttest_finish(df, t, alternative)
    return Ttest_indResult(t, prob)
# 命名元组，用于存储相关样本的 T 检验结果，包括统计量和双尾 p 值
Ttest_relResult = namedtuple('Ttest_relResult', ('statistic', 'pvalue'))

# 计算两个相关样本 a 和 b 的 T 检验
def ttest_rel(a, b, axis=0, alternative='two-sided'):
    """
    计算两个相关样本 a 和 b 的 T 检验。

    Parameters
    ----------
    a, b : array_like
        必须具有相同的形状。
    axis : int or None, optional
        计算测试的轴。如果为 None，则在整个数组 `a` 和 `b` 上计算。
    alternative : {'two-sided', 'less', 'greater'}, optional
        定义备择假设。可用的选项为：

        * 'two-sided': 样本所代表的分布的均值不相等。
        * 'less': 第一个样本所代表的分布的均值小于第二个样本所代表的分布的均值。
        * 'greater': 第一个样本所代表的分布的均值大于第二个样本所代表的分布的均值。

        默认为 'two-sided'。

    Returns
    -------
    statistic : float or array
        t 统计量
    pvalue : float or array
        双尾 p 值

    Notes
    -----
    更多关于 `ttest_rel` 的详细信息，请参阅 `scipy.stats.ttest_rel`。
    """
    # 将 a 和 b 转换为数组，并确保它们具有相同的形状和轴
    a, b, axis = _chk2_asarray(a, b, axis)
    # 如果 a 和 b 的长度不相等，则引发 ValueError
    if len(a) != len(b):
        raise ValueError('unequal length arrays')

    # 如果 a 或 b 的大小为 0，则返回一个包含 NaN 的 Ttest_relResult 对象
    if a.size == 0 or b.size == 0:
        return Ttest_relResult(np.nan, np.nan)

    # 计算样本大小
    n = a.count(axis)
    # 计算自由度
    df = ma.asanyarray(n-1.0)
    # 计算差异并求平均
    d = (a-b).astype('d')
    dm = d.mean(axis)
    # 计算方差
    v = d.var(axis=axis, ddof=1)
    # 计算标准误差
    denom = ma.sqrt(v / n)
    # 忽略除以零或无效值的警告，并计算 t 统计量
    with np.errstate(divide='ignore', invalid='ignore'):
        t = dm / denom

    # 完成 t 检验，得到 t 统计量和 p 值
    t, prob = _ttest_finish(df, t, alternative)
    return Ttest_relResult(t, prob)


# 命名元组，用于存储 Mann-Whitney 检验的结果，包括统计量和双尾 p 值
MannwhitneyuResult = namedtuple('MannwhitneyuResult', ('statistic', 'pvalue'))

# 计算 Mann-Whitney 统计量
def mannwhitneyu(x, y, use_continuity=True):
    """
    计算 Mann-Whitney 统计量。

    x 和/或 y 中的缺失值将被丢弃。

    Parameters
    ----------
    x : sequence
        输入序列
    y : sequence
        输入序列
    use_continuity : {True, False}, optional
        是否考虑连续性修正（1/2.）。

    Returns
    -------
    statistic : float
        Mann-Whitney 统计量的最小值
    pvalue : float
        假设正态分布的双侧 p 值的近似值。
    """
    # 压缩 x 和 y 的缺失值，并将它们转换为 ndarray
    x = ma.asarray(x).compressed().view(ndarray)
    y = ma.asarray(y).compressed().view(ndarray)
    # 对合并后的排名进行排序
    ranks = rankdata(np.concatenate([x,y]))
    # 计算样本大小
    (nx, ny) = (len(x), len(y))
    nt = nx + ny
    # 计算 U 统计量
    U = ranks[:nx].sum() - nx*(nx+1)/2.
    U = max(U, nx*ny - U)
    u = nx*ny - U

    # 计算期望的平均值和方差
    mu = (nx*ny)/2.
    sigsq = (nt**3 - nt)/12.
    # 计算并调整并列组的数量
    ties = count_tied_groups(ranks)
    sigsq -= sum(v*(k**3-k) for (k,v) in ties.items())/12.
    sigsq *= nx*ny/float(nt*(nt-1))
    # 如果使用连续性修正，计算 z 值
    if use_continuity:
        z = (U - 1/2. - mu) / ma.sqrt(sigsq)
    # 否则，按标准方式计算 z 值
    else:
        z = (U - mu) / ma.sqrt(sigsq)

    # 使用正态分布的余补函数计算双样本 Mann-Whitney U 检验的概率值
    prob = special.erfc(abs(z)/np.sqrt(2))
    # 返回 Mann-Whitney U 检验的结果，包括统计量 u 和计算得到的概率值
    return MannwhitneyuResult(u, prob)
# 创建一个命名元组 KruskalResult，包含统计量 statistic 和 p 值 pvalue 两个字段
KruskalResult = namedtuple('KruskalResult', ('statistic', 'pvalue'))

# 定义 Kruskal-Wallis H 检验函数，用于独立样本的假设检验
def kruskal(*args):
    """
    计算独立样本的Kruskal-Wallis H检验

    Parameters
    ----------
    sample1, sample2, ... : array_like
       可以给出两个或更多个样本测量值的数组作为参数。

    Returns
    -------
    statistic : float
       Kruskal-Wallis H 统计量，考虑了并列值的影响
    pvalue : float
       假设检验的 p 值，假设 H 符合卡方分布

    Notes
    -----
    有关 `kruskal` 的更多详情，请参阅 `scipy.stats.kruskal`。

    Examples
    --------
    >>> from scipy.stats.mstats import kruskal

    对三个不同品牌电池的随机样本进行测试，以确定其电荷持续时间：

    >>> a = [6.3, 5.4, 5.7, 5.2, 5.0]
    >>> b = [6.9, 7.0, 6.1, 7.9]
    >>> c = [7.2, 6.9, 6.1, 6.5]

    测试所有品牌电池的分布函数是否相同，使用显著性水平为5%。

    >>> kruskal(a, b, c)
    KruskalResult(statistic=7.113812154696133, pvalue=0.028526948491942164)

    在显著性水平为5%的条件下，由于返回的 p 值小于临界值5%，拒绝原假设。

    """
    # 将输入参数转换为数组输出
    output = argstoarray(*args)
    # 计算秩
    ranks = ma.masked_equal(rankdata(output, use_missing=False), 0)
    sumrk = ranks.sum(-1)
    ngrp = ranks.count(-1)
    ntot = ranks.count()
    # 计算 H 统计量
    H = 12./(ntot*(ntot+1)) * (sumrk**2/ngrp).sum() - 3*(ntot+1)
    # 处理并列值的修正
    ties = count_tied_groups(ranks)
    T = 1. - sum(v*(k**3-k) for (k,v) in ties.items())/float(ntot**3-ntot)
    if T == 0:
        raise ValueError('All numbers are identical in kruskal')

    H /= T
    df = len(output) - 1
    prob = distributions.chi2.sf(H, df)
    return KruskalResult(H, prob)


# 为了向后兼容，ks_1samp 函数与 ks_1samp 函数是相同的
kruskalwallis = kruskal


# 重命名参数的装饰器函数，将 'mode' 参数重命名为 'method'
@_rename_parameter("mode", "method")
def ks_1samp(x, cdf, args=(), alternative="two-sided", method='auto'):
    """
    计算对一个样本的Kolmogorov-Smirnov检验。

    在 `x` 中的缺失值将被丢弃。

    Parameters
    ----------
    x : array_like
        随机变量观测值的1-D数组。
    cdf : str or callable
        如果是字符串，应为 `scipy.stats` 中分布的名称。
        如果是可调用对象，则用于计算 cdf。
    args : tuple, sequence, optional
        分布参数，仅当 `cdf` 是字符串时使用。
    alternative : {'two-sided', 'less', 'greater'}, optional
        表示备择假设。默认为 'two-sided'。

    method : {'auto'}, optional
        用于确定要使用的方法的字符串。

    """
    # 定义参数 method，用于指定计算 p 值的方法，可选值为 {'auto', 'exact', 'asymp'}
    # 默认为 'auto'
    method : {'auto', 'exact', 'asymp'}, optional
        定义用于计算 p 值的方法。
        可用选项如下（默认为 'auto'）：

          * 'auto' : 对于小型数组使用 'exact'，对于大型数组使用 'asymp'
          * 'exact' : 使用测试统计量的精确分布的近似
          * 'asymp' : 使用测试统计量的渐近分布

    Returns
    -------
    d : float
        Kolmogorov-Smirnov 检验的值
    p : float
        相应的 p 值。

    """
    # alternative 变量用于指定检验的备择假设，映射关系为 {'t': 'two-sided', 'g': 'greater', 'l': 'less'}
    # 根据用户提供的 alternative 参数设定，如果未提供则使用默认的 alternative
    alternative = {'t': 'two-sided', 'g': 'greater', 'l': 'less'}.get(
       alternative.lower()[0], alternative)
    # 调用 scipy.stats._stats_py.ks_1samp 执行 Kolmogorov-Smirnov 单样本检验
    return scipy.stats._stats_py.ks_1samp(
        x, cdf, args=args, alternative=alternative, method=method)
# 使用装饰器重命名参数"mode"为"method"
@_rename_parameter("mode", "method")
# 定义 Kolmogorov-Smirnov 两样本检验函数，用于比较两个数据样本的分布
def ks_2samp(data1, data2, alternative="two-sided", method='auto'):
    """
    Computes the Kolmogorov-Smirnov test on two samples.

    Missing values in `x` and/or `y` are discarded.

    Parameters
    ----------
    data1 : array_like
        First data set
    data2 : array_like
        Second data set
    alternative : {'two-sided', 'less', 'greater'}, optional
        Indicates the alternative hypothesis.  Default is 'two-sided'.
    method : {'auto', 'exact', 'asymp'}, optional
        Defines the method used for calculating the p-value.
        The following options are available (default is 'auto'):

          * 'auto' : use 'exact' for small size arrays, 'asymp' for large
          * 'exact' : use approximation to exact distribution of test statistic
          * 'asymp' : use asymptotic distribution of test statistic

    Returns
    -------
    d : float
        Value of the Kolmogorov Smirnov test
    p : float
        Corresponding p-value.

    """
    # 根据 alternative 参数的设置，转换为规范的字符串形式
    alternative = {'t': 'two-sided', 'g': 'greater', 'l': 'less'}.get(
       alternative.lower()[0], alternative)
    # 调用 scipy.stats._stats_py 中的 ks_2samp 函数进行实际计算
    return scipy.stats._stats_py.ks_2samp(data1, data2,
                                          alternative=alternative,
                                          method=method)


# 将 ks_2samp 函数重命名为 ks_twosamp，提供更简洁的别名
ks_twosamp = ks_2samp


# 使用装饰器重命名参数"mode"为"method"
@_rename_parameter("mode", "method")
# 定义 Kolmogorov-Smirnov 单样本检验函数
def kstest(data1, data2, args=(), alternative='two-sided', method='auto'):
    """

    Parameters
    ----------
    data1 : array_like
    data2 : str, callable or array_like
    args : tuple, sequence, optional
        Distribution parameters, used if `data1` or `data2` are strings.
    alternative : str, as documented in stats.kstest
    method : str, as documented in stats.kstest

    Returns
    -------
    tuple of (K-S statistic, probability)

    """
    # 调用 scipy.stats._stats_py 中的 kstest 函数进行 Kolmogorov-Smirnov 测试
    return scipy.stats._stats_py.kstest(data1, data2, args,
                                        alternative=alternative, method=method)


# 定义一个函数，用于对数组进行修剪（trim）
def trima(a, limits=None, inclusive=(True,True)):
    """
    Trims an array by masking the data outside some given limits.

    Returns a masked version of the input array.

    Parameters
    ----------
    a : array_like
        Input array.
    limits : {None, tuple}, optional
        Tuple of (lower limit, upper limit) in absolute values.
        Values of the input array lower (greater) than the lower (upper) limit
        will be masked.  A limit is None indicates an open interval.
    inclusive : (bool, bool) tuple, optional
        Tuple of (lower flag, upper flag), indicating whether values exactly
        equal to the lower (upper) limit are allowed.

    Examples
    --------
    >>> from scipy.stats.mstats import trima
    >>> import numpy as np

    >>> a = np.arange(10)
    # 将输入的数组 `a` 转换为掩码数组，以便在数值操作中可以处理缺失值
    a = ma.asarray(a)
    
    # 取消共享掩码，确保对数组进行修改时不会影响原始数据
    a.unshare_mask()
    
    # 如果限制条件 `limits` 为 `None` 或者是 `(None, None)`，直接返回原始数组 `a`
    if (limits is None) or (limits == (None, None)):
        return a
    
    # 分别获取下限和上限
    (lower_lim, upper_lim) = limits
    
    # 分别获取下限和上限是否包含的布尔值
    (lower_in, upper_in) = inclusive
    
    # 初始化条件为 False
    condition = False
    
    # 如果存在下限，根据是否包含下限设置条件
    if lower_lim is not None:
        if lower_in:
            condition |= (a < lower_lim)  # 包含下限时，选择小于下限的条件
        else:
            condition |= (a <= lower_lim)  # 不包含下限时，选择小于等于下限的条件
    
    # 如果存在上限，根据是否包含上限设置条件
    if upper_lim is not None:
        if upper_in:
            condition |= (a > upper_lim)  # 包含上限时，选择大于上限的条件
        else:
            condition |= (a >= upper_lim)  # 不包含上限时，选择大于等于上限的条件
    
    # 根据条件将满足条件的元素设置为掩码值 `masked`
    a[condition.filled(True)] = masked
    
    # 返回处理后的数组 `a`，其中满足条件的元素已被设置为掩码值
    return a
# 定义一个函数 trimr，用于根据给定的限制修剪数组，并返回修剪后的结果
def trimr(a, limits=None, inclusive=(True, True), axis=None):
    """
    Trims an array by masking some proportion of the data on each end.
    Returns a masked version of the input array.

    Parameters
    ----------
    a : sequence
        Input array.
    limits : {None, tuple}, optional
        Tuple of the percentages to cut on each side of the array, with respect
        to the number of unmasked data, as floats between 0. and 1.
        Noting n the number of unmasked data before trimming, the
        (n*limits[0])th smallest data and the (n*limits[1])th largest data are
        masked, and the total number of unmasked data after trimming is
        n*(1.-sum(limits)).  The value of one limit can be set to None to
        indicate an open interval.
    inclusive : {(True,True) tuple}, optional
        Tuple of flags indicating whether the number of data being masked on
        the left (right) end should be truncated (True) or rounded (False) to
        integers.
    axis : {None,int}, optional
        Axis along which to trim. If None, the whole array is trimmed, but its
        shape is maintained.

    """

    # 定义内部函数 _trimr1D，用于在一维数组上执行修剪操作
    def _trimr1D(a, low_limit, up_limit, low_inclusive, up_inclusive):
        # 计算未屏蔽数据的数量
        n = a.count()
        # 获取数据排序后的索引
        idx = a.argsort()
        
        # 处理低限制
        if low_limit:
            if low_inclusive:
                lowidx = int(low_limit*n)
            else:
                lowidx = int(np.round(low_limit*n))
            # 对索引进行屏蔽处理
            a[idx[:lowidx]] = masked
        
        # 处理高限制
        if up_limit is not None:
            if up_inclusive:
                upidx = n - int(n*up_limit)
            else:
                upidx = n - int(np.round(n*up_limit))
            # 对索引进行屏蔽处理
            a[idx[upidx:]] = masked
        
        return a

    # 将输入数组 a 转换为掩码数组
    a = ma.asarray(a)
    # 取消共享掩码以确保在原始数据上直接操作
    a.unshare_mask()
    
    # 如果 limits 参数为 None，则直接返回原始数组
    if limits is None:
        return a

    # 检查限制参数的合法性
    (lolim, uplim) = limits
    errmsg = "The proportion to cut from the %s should be between 0. and 1."
    if lolim is not None:
        if lolim > 1. or lolim < 0:
            raise ValueError(errmsg % 'beginning' + "(got %s)" % lolim)
    if uplim is not None:
        if uplim > 1. or uplim < 0:
            raise ValueError(errmsg % 'end' + "(got %s)" % uplim)

    # 获取 inclusive 参数的值
    (loinc, upinc) = inclusive

    # 如果 axis 为 None，则对整个数组执行一维修剪操作并保持其形状不变
    if axis is None:
        shp = a.shape
        return _trimr1D(a.ravel(), lolim, uplim, loinc, upinc).reshape(shp)
    else:
        # 否则，沿指定的轴应用 _trimr1D 函数
        return ma.apply_along_axis(_trimr1D, axis, a, lolim, uplim, loinc, upinc)


# 定义 trimdoc 变量，包含函数 trimr 的文档字符串的一部分
trimdoc = """
    Parameters
    ----------
    a : sequence
        Input array

"""
    # limits参数：{None, tuple}, optional
    # 如果`relative`为False，则为绝对值范围的元组（下限，上限）。
    # 输入数组中低于（大于）下限（上限）的值将被掩盖。
    # 
    # 如果`relative`为True，则为每侧要修剪的数组的百分比范围的元组。
    # 与未掩盖数据数量相关，n表示修剪前的未掩盖数据数。
    # 第(n*limits[0])小的数据和第(n*limits[1])大的数据将被掩盖，
    # 修剪后未掩盖数据的总数为n*(1.-sum(limits))
    # 在每种情况下，一个限制的值可以设置为None，表示开放区间。
    # 
    # 如果limits为None，则不执行修剪。
    limits : {None, tuple}, optional

    # inclusive参数：{(bool, bool) tuple}, optional
    # 如果`relative`为False，则为元组，指示是否允许与绝对限制值完全相等的值。
    # 如果`relative`为True，则为元组，指示是否应该四舍五入（True）或截断（False）每侧被掩盖的数据数量。
    inclusive : {(bool, bool) tuple}, optional

    # relative参数：bool, optional
    # 是否将限制视为绝对值（False）或要修剪的比例（True）。
    relative : bool, optional

    # axis参数：int, optional
    # 要进行修剪的轴。
    axis : int, optional
"""
定义一个函数，用于根据给定的限制修剪数组，并返回修剪后的遮罩版本。

%s

示例
--------
>>> from scipy.stats.mstats import trim
>>> z = [ 1, 2, 3, 4, 5, 6, 7, 8, 9,10]
>>> print(trim(z,(3,8)))
[-- -- 3 4 5 6 7 8 -- --]
>>> print(trim(z,(0.1,0.2),relative=True))
[-- 2 3 4 5 6 7 8 -- --]
"""
if relative:
    # 如果相对修剪参数为真，则调用trimr函数进行相对修剪
    return trimr(a, limits=limits, inclusive=inclusive, axis=axis)
else:
    # 否则调用trima函数进行绝对修剪
    return trima(a, limits=limits, inclusive=inclusive)


if trim.__doc__:
    # 如果trim函数有文档字符串，则将其格式化为trimdoc的值
    trim.__doc__ = trim.__doc__ % trimdoc


def trimboth(data, proportiontocut=0.2, inclusive=(True,True), axis=None):
    """
    Trims the smallest and largest data values.

    Trims the `data` by masking the ``int(proportiontocut * n)`` smallest and
    ``int(proportiontocut * n)`` largest values of data along the given axis,
    where n is the number of unmasked values before trimming.

    Parameters
    ----------
    data : ndarray
        Data to trim.
    proportiontocut : float, optional
        Percentage of trimming (as a float between 0 and 1).
        If n is the number of unmasked values before trimming, the number of
        values after trimming is ``(1 - 2*proportiontocut) * n``.
        Default is 0.2.
    inclusive : {(bool, bool) tuple}, optional
        Tuple indicating whether the number of data being masked on each side
        should be rounded (True) or truncated (False).
    axis : int, optional
        Axis along which to perform the trimming.
        If None, the input array is first flattened.

    """
    # 调用trimr函数进行修剪，传入限制为(proportiontocut, proportiontocut)
    return trimr(data, limits=(proportiontocut,proportiontocut),
                 inclusive=inclusive, axis=axis)


def trimtail(data, proportiontocut=0.2, tail='left', inclusive=(True,True),
             axis=None):
    """
    Trims the data by masking values from one tail.

    Parameters
    ----------
    data : array_like
        Data to trim.
    proportiontocut : float, optional
        Percentage of trimming. If n is the number of unmasked values
        before trimming, the number of values after trimming is
        ``(1 - proportiontocut) * n``.  Default is 0.2.
    tail : {'left','right'}, optional
        If 'left' the `proportiontocut` lowest values will be masked.
        If 'right' the `proportiontocut` highest values will be masked.
        Default is 'left'.
    inclusive : {(bool, bool) tuple}, optional
        Tuple indicating whether the number of data being masked on each side
        should be rounded (True) or truncated (False).  Default is
        (True, True).
    axis : int, optional
        Axis along which to perform the trimming.
        If None, the input array is first flattened.  Default is None.

    Returns
    -------
    trimtail : ndarray
        Returned array of same shape as `data` with masked tail values.

    """
    # 将变量 tail 转换为字符串，并转换为小写，然后取其第一个字符
    tail = str(tail).lower()[0]
    
    # 根据 tail 的值来设定 limits 变量的元组值
    if tail == 'l':
        limits = (proportiontocut, None)  # 如果 tail 是 'l'，则限制左侧数据的比例，右侧无限制
    elif tail == 'r':
        limits = (None, proportiontocut)  # 如果 tail 是 'r'，则限制右侧数据的比例，左侧无限制
    else:
        raise TypeError("The tail argument should be in ('left','right')")  # 如果 tail 不是 'l' 或 'r'，抛出类型错误异常
    
    # 调用 trimr 函数进行数据截取操作，根据给定的限制参数
    # trimr 函数的参数包括 data（待处理数据）、limits（截取限制）、axis（轴向，默认为 None）、inclusive（是否包含限制边界，默认为 True）
    return trimr(data, limits=limits, axis=axis, inclusive=inclusive)
trim1 = trimtail

# 定义一个变量 trim1，并将其赋值为 trimtail，这里假设 trimtail 是在其他地方定义或导入的变量

def trimmed_mean(a, limits=(0.1,0.1), inclusive=(1,1), relative=True,
                 axis=None):
    """Returns the trimmed mean of the data along the given axis.

    Parameters
    ----------
    a : array_like
        Input data
    limits : float or tuple of floats, optional
        Limits of trimming as fraction of total data, as a float, or as (lower, upper) tuple
    inclusive : int or tuple of ints, optional
        Whether to include the limit data points. Default is (1,1) meaning include.
    relative : bool, optional
        If True, trim relative to the ends of the array. If False, trim by absolute values.
    axis : int, optional
        Axis along which to compute trimmed mean. If None, compute over the whole array.

    Returns
    -------
    trimmed_mean : ndarray
        Trimmed mean along the specified axis

    """
    if (not isinstance(limits,tuple)) and isinstance(limits,float):
        limits = (limits, limits)
    # 如果 limits 是单个 float，转换为 (limits, limits) 的 tuple 形式

    if relative:
        return trimr(a,limits=limits,inclusive=inclusive,axis=axis).mean(axis=axis)
        # 如果 relative 为 True，则调用 trimr 函数计算修剪后的数据，并计算平均值
    else:
        return trima(a,limits=limits,inclusive=inclusive).mean(axis=axis)
        # 如果 relative 为 False，则调用 trima 函数计算修剪后的数据，并计算平均值

if trimmed_mean.__doc__:
    trimmed_mean.__doc__ = trimmed_mean.__doc__ % trimdoc
    # 如果 trimmed_mean 函数有文档字符串，则将其格式化为 trimdoc 变量中的内容

def trimmed_var(a, limits=(0.1,0.1), inclusive=(1,1), relative=True,
                axis=None, ddof=0):
    """Returns the trimmed variance of the data along the given axis.

    Parameters
    ----------
    a : array_like
        Input data
    limits : float or tuple of floats, optional
        Limits of trimming as fraction of total data, as a float, or as (lower, upper) tuple
    inclusive : int or tuple of ints, optional
        Whether to include the limit data points. Default is (1,1) meaning include.
    relative : bool, optional
        If True, trim relative to the ends of the array. If False, trim by absolute values.
    axis : int, optional
        Axis along which to compute trimmed variance. If None, compute over the whole array.
    ddof : {0,integer}, optional
        Means Delta Degrees of Freedom. The denominator used during computations
        is (n-ddof). DDOF=0 corresponds to a biased estimate, DDOF=1 to an un-
        biased estimate of the variance.

    Returns
    -------
    trimmed_var : ndarray
        Trimmed variance along the specified axis

    """
    if (not isinstance(limits,tuple)) and isinstance(limits,float):
        limits = (limits, limits)
    # 如果 limits 是单个 float，转换为 (limits, limits) 的 tuple 形式

    if relative:
        out = trimr(a,limits=limits, inclusive=inclusive,axis=axis)
    else:
        out = trima(a,limits=limits,inclusive=inclusive)
    # 根据 relative 的值选择使用 trimr 或 trima 函数进行数据修剪

    return out.var(axis=axis, ddof=ddof)
    # 返回修剪后数据的方差，可以指定 ddof 参数来选择使用有偏或无偏估计

if trimmed_var.__doc__:
    trimmed_var.__doc__ = trimmed_var.__doc__ % trimdoc
    # 如果 trimmed_var 函数有文档字符串，则将其格式化为 trimdoc 变量中的内容

def trimmed_std(a, limits=(0.1,0.1), inclusive=(1,1), relative=True,
                axis=None, ddof=0):
    """Returns the trimmed standard deviation of the data along the given axis.

    Parameters
    ----------
    a : array_like
        Input data
    limits : float or tuple of floats, optional
        Limits of trimming as fraction of total data, as a float, or as (lower, upper) tuple
    inclusive : int or tuple of ints, optional
        Whether to include the limit data points. Default is (1,1) meaning include.
    relative : bool, optional
        If True, trim relative to the ends of the array. If False, trim by absolute values.
    axis : int, optional
        Axis along which to compute trimmed standard deviation. If None, compute over the whole array.
    ddof : {0,integer}, optional
        Means Delta Degrees of Freedom. The denominator used during computations
        is (n-ddof). DDOF=0 corresponds to a biased estimate, DDOF=1 to an un-
        biased estimate of the variance.

    Returns
    -------
    trimmed_std : ndarray
        Trimmed standard deviation along the specified axis

    """
    if (not isinstance(limits,tuple)) and isinstance(limits,float):
        limits = (limits, limits)
    # 如果 limits 是单个 float，转换为 (limits, limits) 的 tuple 形式

    if relative:
        out = trimr(a,limits=limits,inclusive=inclusive,axis=axis)
    else:
        out = trima(a,limits=limits,inclusive=inclusive)
    # 根据 relative 的值选择使用 trimr 或 trima 函数进行数据修剪

    return out.std(axis=axis,ddof=ddof)
    # 返回修剪后数据的标准差，可以指定 ddof 参数来选择使用有偏或无偏估计

if trimmed_std.__doc__:
    trimmed_std.__doc__ = trimmed_std.__doc__ % trimdoc
    # 如果 trimmed_std 函数有文档字符串，则将其格式化为 trimdoc 变量中的内容

def trimmed_stde(a, limits=(0.1,0.1), inclusive=(1,1), axis=None):
    """
    Returns the standard error of the trimmed mean along the given axis.

    Parameters
    ----------
    a : sequence
        Input array

    Returns
    -------
    trimmed_stde : ndarray
        Standard error of the trimmed mean along the specified axis

    """
    limits : {(0.1,0.1), tuple of float}, optional
        tuple (lower percentage, upper percentage) to cut  on each side of the
        array, with respect to the number of unmasked data.

        If n is the number of unmasked data before trimming, the values
        smaller than ``n * limits[0]`` and the values larger than
        ``n * `limits[1]`` are masked, and the total number of unmasked
        data after trimming is ``n * (1.-sum(limits))``.  In each case,
        the value of one limit can be set to None to indicate an open interval.
        If `limits` is None, no trimming is performed.
    inclusive : {(bool, bool) tuple} optional
        Tuple indicating whether the number of data being masked on each side
        should be rounded (True) or truncated (False).
    axis : int, optional
        Axis along which to trim.

    Returns
    -------
    trimmed_stde : scalar or ndarray
        The standard error of the trimmed mean after applying trimming.

    """
    def _trimmed_stde_1D(a, low_limit, up_limit, low_inclusive, up_inclusive):
        "Returns the standard error of the trimmed mean for a 1D input data."
        n = a.count()  # Count the number of valid data points
        idx = a.argsort()  # Get indices that would sort the array `a`
        if low_limit:
            if low_inclusive:
                lowidx = int(low_limit*n)  # Calculate the index up to which to trim from the lower end
            else:
                lowidx = np.round(low_limit*n)  # Calculate the rounded index up to which to trim
            a[idx[:lowidx]] = masked  # Mask the values below the lower limit
        if up_limit is not None:
            if up_inclusive:
                upidx = n - int(n*up_limit)  # Calculate the index from which to trim from the upper end
            else:
                upidx = n - np.round(n*up_limit)  # Calculate the rounded index from which to trim
            a[idx[upidx:]] = masked  # Mask the values above the upper limit
        a[idx[:lowidx]] = a[idx[lowidx]]  # Fill the trimmed lower values with the last valid value
        a[idx[upidx:]] = a[idx[upidx-1]]  # Fill the trimmed upper values with the last valid value before upper limit
        winstd = a.std(ddof=1)  # Calculate the standard deviation of the trimmed array
        return winstd / ((1-low_limit-up_limit)*np.sqrt(len(a)))  # Return the standard error of the trimmed mean

    a = ma.array(a, copy=True, subok=True)  # Ensure `a` is a masked array
    a.unshare_mask()  # Unshare the mask with other arrays
    if limits is None:
        return a.std(axis=axis,ddof=1)/ma.sqrt(a.count(axis))  # Return the standard deviation over the square root of the count if no trimming is applied
    if (not isinstance(limits,tuple)) and isinstance(limits,float):
        limits = (limits, limits)  # Convert single float limit into a tuple

    # Check the limits are within the valid range
    (lolim, uplim) = limits
    errmsg = "The proportion to cut from the %s should be between 0. and 1."
    if lolim is not None:
        if lolim > 1. or lolim < 0:
            raise ValueError(errmsg % 'beginning' + "(got %s)" % lolim)
    if uplim is not None:
        if uplim > 1. or uplim < 0:
            raise ValueError(errmsg % 'end' + "(got %s)" % uplim)

    (loinc, upinc) = inclusive  # Retrieve the inclusive flags
    if (axis is None):
        return _trimmed_stde_1D(a.ravel(),lolim,uplim,loinc,upinc)  # Trim and compute for 1D case if axis is None
    else:
        if a.ndim > 2:
            raise ValueError("Array 'a' must be at most two dimensional, "
                             "but got a.ndim = %d" % a.ndim)  # Raise error if array `a` is more than 2D
        return ma.apply_along_axis(_trimmed_stde_1D, axis, a,
                                   lolim,uplim,loinc,upinc)  # Apply trimming along the specified axis
# 定义一个函数，用于对数组进行限制范围内的掩码处理，返回一个 MaskedArray 对象
def _mask_to_limits(a, limits, inclusive):
    """Mask an array for values outside of given limits.

    This is primarily a utility function.

    Parameters
    ----------
    a : array
        输入的数组
    limits : (float or None, float or None)
        包含两个元素的元组，表示（下限，上限）。数组中小于下限或大于上限的值将被掩盖。None 表示没有限制。
    inclusive : (bool, bool)
        包含两个元素的元组，表示（下限包含标志，上限包含标志）。这些标志确定是否包含等于下限或上限的值。

    Returns
    -------
    A MaskedArray.
        返回一个掩码数组对象

    Raises
    ------
    A ValueError if there are no values within the given limits.
        如果没有在给定限制范围内的值，则引发 ValueError 异常。
    """
    lower_limit, upper_limit = limits
    lower_include, upper_include = inclusive
    am = ma.MaskedArray(a)
    if lower_limit is not None:
        if lower_include:
            am = ma.masked_less(am, lower_limit)
        else:
            am = ma.masked_less_equal(am, lower_limit)

    if upper_limit is not None:
        if upper_include:
            am = ma.masked_greater(am, upper_limit)
        else:
            am = ma.masked_greater_equal(am, upper_limit)

    if am.count() == 0:
        raise ValueError("No array values within given limits")

    return am


def tmean(a, limits=None, inclusive=(True, True), axis=None):
    """
    Compute the trimmed mean.

    Parameters
    ----------
    a : array_like
        Array of values.
        值的数组。
    limits : None or (lower limit, upper limit), optional
        Values in the input array less than the lower limit or greater than the
        upper limit will be ignored.  When limits is None (default), then all
        values are used.  Either of the limit values in the tuple can also be
        None representing a half-open interval.
        输入数组中小于下限或大于上限的值将被忽略。当 limits 为 None（默认）时，使用所有值。元组中的任一限制值也可以为 None，表示半开区间。
    inclusive : (bool, bool), optional
        A tuple consisting of the (lower flag, upper flag).  These flags
        determine whether values exactly equal to the lower or upper limits
        are included.  The default value is (True, True).
        包含两个元素的元组，表示（下限包含标志，上限包含标志）。这些标志确定是否包含等于下限或上限的值。默认值为 (True, True)。
    axis : int or None, optional
        Axis along which to operate. If None, compute over the
        whole array. Default is None.
        沿着哪个轴操作。如果为 None，则在整个数组上计算。默认为 None。

    Returns
    -------
    tmean : float
        返回修剪均值。

    Notes
    -----
    For more details on `tmean`, see `scipy.stats.tmean`.
    更多关于 `tmean` 的详细信息，请参阅 `scipy.stats.tmean`。

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import mstats
    >>> a = np.array([[6, 8, 3, 0],
    ...               [3, 9, 1, 2],
    ...               [8, 7, 8, 2],
    ...               [5, 6, 0, 2],
    ...               [4, 5, 5, 2]])
    ...
    ...
    >>> mstats.tmean(a, (2,5))
    3.3
    >>> mstats.tmean(a, (2,5), axis=0)
    masked_array(data=[4.0, 5.0, 4.0, 2.0],
                 mask=[False, False, False, False],
           fill_value=1e+20)

    """
    return trima(a, limits=limits, inclusive=inclusive).mean(axis=axis)


def tvar(a, limits=None, inclusive=(True, True), axis=0, ddof=1):
    """
    ```
    Compute the trimmed variance

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
        Axis along which to operate. If None, compute over the
        whole array. Default is zero.
    ddof : int, optional
        Delta degrees of freedom. Default is 1.

    Returns
    -------
    tvar : float
        Trimmed variance.

    Notes
    -----
    For more details on `tvar`, see `scipy.stats.tvar`.

    """
    # 将输入数组转换为浮点类型，并展平为一维数组
    a = a.astype(float).ravel()
    # 如果 limits 参数为 None，则计算未屏蔽值的数量 n，返回调整后的方差
    if limits is None:
        n = (~a.mask).sum()  # todo: better way to do that?
        return np.ma.var(a) * n/(n-1.)
    # 根据给定的 limits 和 inclusive 参数，将数组 a 进行屏蔽操作，得到屏蔽后的数组 am
    am = _mask_to_limits(a, limits=limits, inclusive=inclusive)

    # 返回在指定轴上的屏蔽后数组 am 的方差，带有指定的自由度修正因子 ddof
    return np.ma.var(am, axis=axis, ddof=ddof)
# 计算修剪后的最小值
def tmin(a, lowerlimit=None, axis=0, inclusive=True):
    """
    Compute the trimmed minimum

    Parameters
    ----------
    a : array_like
        array of values
    lowerlimit : None or float, optional
        Values in the input array less than the given limit will be ignored.
        When lowerlimit is None, then all values are used. The default value
        is None.
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over the
        whole array `a`.
    inclusive : {True, False}, optional
        This flag determines whether values exactly equal to the lower limit
        are included.  The default value is True.

    Returns
    -------
    tmin : float, int or ndarray

    Notes
    -----
    For more details on `tmin`, see `scipy.stats.tmin`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import mstats
    >>> a = np.array([[6, 8, 3, 0],
    ...               [3, 2, 1, 2],
    ...               [8, 1, 8, 2],
    ...               [5, 3, 0, 2],
    ...               [4, 7, 5, 2]])
    ...
    >>> mstats.tmin(a, 5)
    masked_array(data=[5, 7, 5, --],
                 mask=[False, False, False,  True],
           fill_value=999999)

    """
    # 将输入数组转换为掩码数组，并指定操作的轴
    a, axis = _chk_asarray(a, axis)
    # 对数组进行修剪，去除小于给定下限的值，并按照 inclusive 参数确定是否包含等于下限的值
    am = trima(a, (lowerlimit, None), (inclusive, False))
    # 在修剪后的结果中找到每个轴上的最小值
    return ma.minimum.reduce(am, axis)


# 计算修剪后的最大值
def tmax(a, upperlimit=None, axis=0, inclusive=True):
    """
    Compute the trimmed maximum

    This function computes the maximum value of an array along a given axis,
    while ignoring values larger than a specified upper limit.

    Parameters
    ----------
    a : array_like
        array of values
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

    Notes
    -----
    For more details on `tmax`, see `scipy.stats.tmax`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import mstats
    >>> a = np.array([[6, 8, 3, 0],
    ...               [3, 9, 1, 2],
    ...               [8, 7, 8, 2],
    ...               [5, 6, 0, 2],
    ...               [4, 5, 5, 2]])
    ...
    >>> mstats.tmax(a, 4)
    masked_array(data=[4, --, 3, 2],
                 mask=[False,  True, False, False],
           fill_value=999999)

    """
    # 将输入数组转换为掩码数组，并指定操作的轴
    a, axis = _chk_asarray(a, axis)
    # 对数组进行修剪，去除大于给定上限的值，并按照 inclusive 参数确定是否包含等于上限的值
    am = trima(a, (None, upperlimit), (False, inclusive))
    # 在修剪后的结果中找到每个轴上的最大值
    return ma.maximum.reduce(am, axis)
    """
    Compute the trimmed standard error of the mean.

    This function finds the standard error of the mean for given
    values, ignoring values outside the given `limits`.

    Parameters
    ----------
    a : array_like
        array of values
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
        Axis along which to operate. If None, compute over the
        whole array. Default is zero.
    ddof : int, optional
        Delta degrees of freedom. Default is 1.

    Returns
    -------
    tsem : float
        Trimmed standard error of the mean.

    Notes
    -----
    For more details on `tsem`, see `scipy.stats.tsem`.

    """
    # Convert input array `a` to a masked array and flatten it
    a = ma.asarray(a).ravel()
    
    # If `limits` is None, compute standard error of the mean using all values
    if limits is None:
        # Count of non-masked elements in `a`
        n = float(a.count())
        # Compute standard deviation of `a` and return its standard error of the mean
        return a.std(axis=axis, ddof=ddof) / ma.sqrt(n)

    # Trim `a` based on given `limits` and `inclusive` flags
    am = trima(a.ravel(), limits, inclusive)
    # Compute standard deviation of trimmed `am` and return its standard error of the mean
    sd = np.sqrt(am.var(axis=axis, ddof=ddof))
    return sd / np.sqrt(am.count())
# 定义函数 `winsorize`，用于对输入数组进行 Winsorize 处理，即限制数组中的极端值
def winsorize(a, limits=None, inclusive=(True, True), inplace=False,
              axis=None, nan_policy='propagate'):
    """Returns a Winsorized version of the input array.

    The (limits[0])th lowest values are set to the (limits[0])th percentile,
    and the (limits[1])th highest values are set to the (1 - limits[1])th
    percentile.
    Masked values are skipped.

    Parameters
    ----------
    a : sequence
        Input array.
    limits : {None, tuple of float}, optional
        Tuple of the percentages to cut on each side of the array, with respect
        to the number of unmasked data, as floats between 0. and 1.
        Noting n the number of unmasked data before trimming, the
        (n*limits[0])th smallest data and the (n*limits[1])th largest data are
        masked, and the total number of unmasked data after trimming
        is n*(1.-sum(limits)) The value of one limit can be set to None to
        indicate an open interval.
    inclusive : {(True, True) tuple}, optional
        Tuple indicating whether the number of data being masked on each side
        should be truncated (True) or rounded (False).
    inplace : {False, True}, optional
        Whether to winsorize in place (True) or to use a copy (False)
    axis : {None, int}, optional
        Axis along which to trim. If None, the whole array is trimmed, but its
        shape is maintained.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

          * 'propagate': allows nan values and may overwrite or propagate them
          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values

    Notes
    -----
    This function is applied to reduce the effect of possibly spurious outliers
    by limiting the extreme values.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats.mstats import winsorize

    A shuffled array contains integers from 1 to 10.

    >>> a = np.array([10, 4, 9, 8, 5, 3, 7, 2, 1, 6])

    The 10% of the lowest value (i.e., ``1``) and the 20% of the highest
    values (i.e., ``9`` and ``10``) are replaced.

    >>> winsorize(a, limits=[0.1, 0.2])
    masked_array(data=[8, 4, 8, 8, 5, 3, 7, 2, 2, 6],
                 mask=False,
           fill_value=999999)
    """
    def _winsorize1D(a, low_limit, up_limit, low_include, up_include,
                     contains_nan, nan_policy):
        # 计算数组a中的元素个数
        n = a.count()
        # 对数组a的元素索引进行排序
        idx = a.argsort()
        # 如果包含NaN值，统计数组a中的NaN个数
        if contains_nan:
            nan_count = np.count_nonzero(np.isnan(a))
        
        # 如果设置了下限low_limit
        if low_limit:
            # 根据low_include参数确定下限索引
            if low_include:
                lowidx = int(low_limit * n)
            else:
                lowidx = np.round(low_limit * n).astype(int)
            # 如果包含NaN且策略是忽略，则调整低限索引
            if contains_nan and nan_policy == 'omit':
                lowidx = min(lowidx, n - nan_count - 1)
            # 对数组a中低于低限索引的元素进行winsorization（用索引为lowidx的元素替换前lowidx个元素）
            a[idx[:lowidx]] = a[idx[lowidx]]
        
        # 如果设置了上限up_limit
        if up_limit is not None:
            # 根据up_include参数确定上限索引
            if up_include:
                upidx = n - int(n * up_limit)
            else:
                upidx = n - np.round(n * up_limit).astype(int)
            # 如果包含NaN且策略是忽略，则调整上限索引
            if contains_nan and nan_policy == 'omit':
                a[idx[upidx:-nan_count]] = a[idx[upidx - 1]]
            else:
                a[idx[upidx:]] = a[idx[upidx - 1]]
        
        # 返回winsorized后的数组a
        return a

    # 检查数组a是否包含NaN值，并根据nan_policy返回相应的策略
    contains_nan, nan_policy = _contains_nan(a, nan_policy)
    
    # 我们将要修改数组a：最好先复制一份
    a = ma.array(a, copy=np.logical_not(inplace))

    # 如果没有设置limits，则直接返回数组a
    if limits is None:
        return a
    
    # 如果limits不是元组，而是一个浮点数，则转换为(limits, limits)
    if (not isinstance(limits, tuple)) and isinstance(limits, float):
        limits = (limits, limits)

    # 检查限制值lolim和uplim是否在0到1之间
    (lolim, uplim) = limits
    errmsg = "The proportion to cut from the %s should be between 0. and 1."
    if lolim is not None:
        if lolim > 1. or lolim < 0:
            raise ValueError(errmsg % 'beginning' + "(got %s)" % lolim)
    if uplim is not None:
        if uplim > 1. or uplim < 0:
            raise ValueError(errmsg % 'end' + "(got %s)" % uplim)

    # 获取inclusive参数的值
    (loinc, upinc) = inclusive

    # 如果axis为None，对数组a进行一维winsorization并恢复形状
    if axis is None:
        shp = a.shape
        return _winsorize1D(a.ravel(), lolim, uplim, loinc, upinc,
                            contains_nan, nan_policy).reshape(shp)
    else:
        # 否则，沿着指定轴应用一维winsorization函数_winsorize1D
        return ma.apply_along_axis(_winsorize1D, axis, a, lolim, uplim, loinc,
                                   upinc, contains_nan, nan_policy)
# 计算样本的关于均值的第 n 次中心矩

def moment(a, moment=1, axis=0):
    """
    Calculates the nth moment about the mean for a sample.

    Parameters
    ----------
    a : array_like
       data
    moment : int, optional
       order of central moment that is returned
    axis : int or None, optional
       Axis along which the central moment is computed. Default is 0.
       If None, compute over the whole array `a`.

    Returns
    -------
    n-th central moment : ndarray or float
       The appropriate moment along the given axis or over all values if axis
       is None. The denominator for the moment calculation is the number of
       observations, no degrees of freedom correction is done.

    Notes
    -----
    For more details about `moment`, see `scipy.stats.moment`.

    """
    # 将输入数据转换为数组，并检查轴的值
    a, axis = _chk_asarray(a, axis)

    # 如果数组为空
    if a.size == 0:
        moment_shape = list(a.shape)
        del moment_shape[axis]
        dtype = a.dtype.type if a.dtype.kind in 'fc' else np.float64
        # 空数组，返回与 `moment` 形状匹配的 NaN 值数组
        out_shape = (moment_shape if np.isscalar(moment)
                     else [len(moment)] + moment_shape)
        if len(out_shape) == 0:
            return dtype(np.nan)
        else:
            return ma.array(np.full(out_shape, np.nan, dtype=dtype))

    # 如果 `moment` 参数为数组，则返回每个值对应的中心矩
    if not np.isscalar(moment):
        # 计算均值
        mean = a.mean(axis, keepdims=True)
        # 计算每个 `moment` 对应的中心矩
        mmnt = [_moment(a, i, axis, mean=mean) for i in moment]
        return ma.array(mmnt)
    else:
        # 否则返回单个 `moment` 对应的中心矩
        return _moment(a, moment, axis)


# 计算中心矩，可选使用预先计算的均值，等同于 `a.mean(axis, keepdims=True)`
def _moment(a, moment, axis, *, mean=None):
    # 如果 `moment` 不是整数，抛出异常
    if np.abs(moment - np.round(moment)) > 0:
        raise ValueError("All moment parameters must be integers")

    # 根据定义，关于均值的零阶中心矩是1，一阶中心矩是0
    if moment == 0 or moment == 1:
        shape = list(a.shape)
        del shape[axis]
        dtype = a.dtype.type if a.dtype.kind in 'fc' else np.float64

        # 如果形状为空
        if len(shape) == 0:
            return dtype(1.0 if moment == 0 else 0.0)
        else:
            return (ma.ones(shape, dtype=dtype) if moment == 0
                    else ma.zeros(shape, dtype=dtype))
    else:
        # Exponentiation by squares: form exponent sequence
        # 初始化指数序列，起始为 moment
        n_list = [moment]
        current_n = moment
        # 生成指数序列直到 current_n <= 2
        while current_n > 2:
            if current_n % 2:
                current_n = (current_n-1)/2
            else:
                current_n /= 2
            n_list.append(current_n)

        # Starting point for exponentiation by squares
        # 计算均值（如果未提供则计算全局均值）并将其从数组 a 中减去
        mean = a.mean(axis, keepdims=True) if mean is None else mean
        a_zero_mean = a - mean
        # 如果指数序列的最后一个元素为 1，则直接赋值 s 为 a_zero_mean
        if n_list[-1] == 1:
            s = a_zero_mean.copy()
        else:
            # 否则将 s 初始化为 a_zero_mean 的平方
            s = a_zero_mean**2

        # Perform multiplications
        # 执行指数乘法的主循环
        for n in n_list[-2::-1]:
            s = s**2
            if n % 2:
                s *= a_zero_mean
        # 返回沿指定轴的均值结果
        return s.mean(axis)
def kurtosis(a, axis=0, fisher=True, bias=True):
    """
    Computes the kurtosis (Fisher or Pearson) of a dataset.

    Parameters
    ----------
    a : ndarray
        Input data.
    axis : int or None, optional
        Axis along which the kurtosis is calculated. Default is 0.
        If None, compute over the whole array `a`.
    fisher : bool, optional
        If True (default), normalizes by N-1; if False, normalizes by N.
    bias : bool, optional
        If False, calculates the unbiased kurtosis.

    Returns
    -------
    kurtosis : ndarray
        The kurtosis of values along an axis.

    Notes
    -----
    For more details about `kurtosis`, see `scipy.stats.kurtosis`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import kurtosis
    >>> a = np.array([1, 2, 3, 4, 5])
    >>> kurtosis(a)
    -1.3

    In this example, `kurtosis` computes the Fisher kurtosis (default)
    for the input array `a`.

    """
    Computes the kurtosis (Fisher or Pearson) of a dataset.

    Kurtosis is the fourth central moment divided by the square of the
    variance. If Fisher's definition is used, then 3.0 is subtracted from
    the result to give 0.0 for a normal distribution.

    If bias is False then the kurtosis is calculated using k statistics to
    eliminate bias coming from biased moment estimators

    Use `kurtosistest` to see if result is close enough to normal.

    Parameters
    ----------
    a : array
        data for which the kurtosis is calculated
    axis : int or None, optional
        Axis along which the kurtosis is calculated. Default is 0.
        If None, compute over the whole array `a`.
    fisher : bool, optional
        If True, Fisher's definition is used (normal ==> 0.0). If False,
        Pearson's definition is used (normal ==> 3.0).
    bias : bool, optional
        If False, then the calculations are corrected for statistical bias.

    Returns
    -------
    kurtosis : array
        The kurtosis of values along an axis. If all values are equal,
        return -3 for Fisher's definition and 0 for Pearson's definition.

    Notes
    -----
    For more details about `kurtosis`, see `scipy.stats.kurtosis`.

    """
    # 将输入的数据数组转换成 numpy 数组，并处理轴参数
    a, axis = _chk_asarray(a, axis)
    # 计算数据数组的均值，keepdims=True 保持维度
    mean = a.mean(axis, keepdims=True)
    # 计算数据数组的二阶中心距，即二阶矩
    m2 = _moment(a, 2, axis, mean=mean)
    # 计算数据数组的四阶中心距，即四阶矩
    m4 = _moment(a, 4, axis, mean=mean)
    # 计算用于判断是否为零的阈值
    zero = (m2 <= (np.finfo(m2.dtype).resolution * mean.squeeze(axis))**2)
    # 忽略所有的运算警告
    with np.errstate(all='ignore'):
        # 根据条件 zero，使用 ma.where 处理 m4 / m2**2.0，得到 kurtosis 的初始值
        vals = ma.where(zero, 0, m4 / m2**2.0)

    # 如果 bias 为 False，且 zero 不是掩码，且 m2 不是掩码
    if not bias and zero is not ma.masked and m2 is not ma.masked:
        # 计算数据数组的数量
        n = a.count(axis)
        # 判断是否可以进行修正
        can_correct = ~zero & (n > 3)
        if can_correct.any():
            # 根据 can_correct 进行修正的计算
            n = np.extract(can_correct, n)
            m2 = np.extract(can_correct, m2)
            m4 = np.extract(can_correct, m4)
            nval = 1.0/(n-2)/(n-3)*((n*n-1.0)*m4/m2**2.0-3*(n-1)**2.0)
            np.place(vals, can_correct, nval+3.0)
    # 如果 fisher 为 True，则根据 Fisher's 定义返回结果
    if fisher:
        return vals - 3
    else:
        # 否则根据 Pearson's 定义返回结果
        return vals
# 使用 namedtuple 创建一个名为 DescribeResult 的命名元组，用于存储描述统计结果
DescribeResult = namedtuple('DescribeResult', ('nobs', 'minmax', 'mean',
                                               'variance', 'skewness',
                                               'kurtosis'))


def describe(a, axis=0, ddof=0, bias=True):
    """
    Computes several descriptive statistics of the passed array.

    Parameters
    ----------
    a : array_like
        Data array
    axis : int or None, optional
        Axis along which to calculate statistics. Default 0. If None,
        compute over the whole array `a`.
    ddof : int, optional
        degree of freedom (default 0); note that default ddof is different
        from the same routine in stats.describe
    bias : bool, optional
        If False, then the skewness and kurtosis calculations are corrected for
        statistical bias.

    Returns
    -------
    nobs : int
        (size of the data (discarding missing values)

    minmax : (int, int)
        min, max

    mean : float
        arithmetic mean

    variance : float
        unbiased variance

    skewness : float
        biased skewness

    kurtosis : float
        biased kurtosis

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats.mstats import describe
    >>> ma = np.ma.array(range(6), mask=[0, 0, 0, 1, 1, 1])
    >>> describe(ma)
    DescribeResult(nobs=np.int64(3), minmax=(masked_array(data=0,
                 mask=False,
           fill_value=999999), masked_array(data=2,
                 mask=False,
           fill_value=999999)), mean=np.float64(1.0),
           variance=np.float64(0.6666666666666666),
           skewness=masked_array(data=0., mask=False, fill_value=1e+20),
            kurtosis=np.float64(-1.5))

    """
    # 将输入的数据转换为数组，并根据指定轴处理
    a, axis = _chk_asarray(a, axis)
    # 计算数据中非缺失值的数量
    n = a.count(axis)
    # 计算沿指定轴的最小值和最大值
    mm = (ma.minimum.reduce(a, axis=axis), ma.maximum.reduce(a, axis=axis))
    # 计算沿指定轴的均值
    m = a.mean(axis)
    # 计算沿指定轴的无偏方差
    v = a.var(axis, ddof=ddof)
    # 计算沿指定轴的偏度
    sk = skew(a, axis, bias=bias)
    # 计算沿指定轴的峰度
    kurt = kurtosis(a, axis, bias=bias)

    # 返回计算结果的命名元组 DescribeResult
    return DescribeResult(n, mm, m, v, sk, kurt)


def stde_median(data, axis=None):
    """Returns the McKean-Schrader estimate of the standard error of the sample
    median along the given axis. masked values are discarded.

    Parameters
    ----------
    data : ndarray
        Data to trim.
    axis : {None,int}, optional
        Axis along which to perform the trimming.
        If None, the input array is first flattened.

    """
    # 定义用于计算样本中位数标准误差的函数
    def _stdemed_1D(data):
        # 对压缩后的数据进行排序
        data = np.sort(data.compressed())
        # 计算参数值
        n = len(data)
        z = 2.5758293035489004
        k = int(np.round((n+1)/2. - z * np.sqrt(n/4.),0))
        # 返回计算结果
        return ((data[n-k] - data[k-1])/(2.*z))

    # 将输入数据转换为掩码数组
    data = ma.array(data, copy=False, subok=True)
    # 如果未指定轴，则对一维数据进行计算并返回结果
    if (axis is None):
        return _stdemed_1D(data)
    else:
        # 如果数据的维度大于2，则抛出值错误异常
        if data.ndim > 2:
            raise ValueError("Array 'data' must be at most two dimensional, "
                             "but got data.ndim = %d" % data.ndim)
        # 对于数据进行沿指定轴应用 _stdemed_1D 函数
        return ma.apply_along_axis(_stdemed_1D, axis, data)
# 使用 collections 模块中的 namedtuple 创建一个命名元组 SkewtestResult，包含 statistic 和 pvalue 两个字段
SkewtestResult = namedtuple('SkewtestResult', ('statistic', 'pvalue'))

# 定义 skewtest 函数，用于检验数据的偏度是否与正态分布不同
def skewtest(a, axis=0, alternative='two-sided'):
    """
    Tests whether the skew is different from the normal distribution.

    Parameters
    ----------
    a : array_like
        The data to be tested
    axis : int or None, optional
       Axis along which statistics are calculated. Default is 0.
       If None, compute over the whole array `a`.
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
    statistic : array_like
        The computed z-score for this test.
    pvalue : array_like
        A p-value for the hypothesis test

    Notes
    -----
    For more details about `skewtest`, see `scipy.stats.skewtest`.

    """
    # 将输入数据 a 和轴参数 axis 转换为数组，并确保合法性
    a, axis = _chk_asarray(a, axis)
    # 如果 axis 为 None，则将数组展平为一维数组，并重新指定 axis 为 0
    if axis is None:
        a = a.ravel()
        axis = 0
    # 计算数组 a 在指定轴上的偏度
    b2 = skew(a,axis)
    # 计算样本数 n
    n = a.count(axis)
    # 如果最小样本数小于 8，抛出 ValueError 异常
    if np.min(n) < 8:
        raise ValueError(
            "skewtest is not valid with less than 8 samples; %i samples"
            " were given." % np.min(n))

    # 计算 y 值
    y = b2 * ma.sqrt(((n+1)*(n+3)) / (6.0*(n-2)))
    # 计算 beta2 值
    beta2 = (3.0*(n*n+27*n-70)*(n+1)*(n+3)) / ((n-2.0)*(n+5)*(n+7)*(n+9))
    # 计算 W2 值
    W2 = -1 + ma.sqrt(2*(beta2-1))
    # 计算 delta 值
    delta = 1/ma.sqrt(0.5*ma.log(W2))
    # 计算 alpha 值
    alpha = ma.sqrt(2.0/(W2-1))
    # 将 y 中的零值替换为 1
    y = ma.where(y == 0, 1, y)
    # 计算 Z 值
    Z = delta*ma.log(y/alpha + ma.sqrt((y/alpha)**2+1))
    # 计算 p 值
    pvalue = scipy.stats._stats_py._get_pvalue(Z, distributions.norm, alternative)

    # 返回 SkewtestResult 命名元组，包含 Z 值和 p 值
    return SkewtestResult(Z[()], pvalue[()])


# 使用 collections 模块中的 namedtuple 创建一个命名元组 KurtosistestResult，包含 statistic 和 pvalue 两个字段
KurtosistestResult = namedtuple('KurtosistestResult', ('statistic', 'pvalue'))


# 定义 kurtosistest 函数，用于检验数据集的峰度是否服从正态分布
def kurtosistest(a, axis=0, alternative='two-sided'):
    """
    Tests whether a dataset has normal kurtosis

    Parameters
    ----------
    a : array_like
        array of the sample data
    axis : int or None, optional
       Axis along which to compute test. Default is 0. If None,
       compute over the whole array `a`.
    # 检查输入数组 `a`，并将其转换为数组形式，同时确定轴的方向
    a, axis = _chk_asarray(a, axis)
    # 计算每个轴上的观测数 `n`
    n = a.count(axis=axis)
    # 如果最小观测数小于5，则抛出数值错误
    if np.min(n) < 5:
        raise ValueError(
            "kurtosistest 需要至少5个观测值；给出了 %i 个观测值。" % np.min(n))
    # 如果最小观测数小于20，则发出警告，说明结果仅在 n>=20 时有效，但继续计算
    if np.min(n) < 20:
        warnings.warn(
            "kurtosistest 仅在 n>=20 时有效... 继续计算，n=%i" % np.min(n),
            stacklevel=2,
        )

    # 计算非 Fisher 标准化的峰度 `b2`
    b2 = kurtosis(a, axis, fisher=False)
    # 计算期望值 `E`
    E = 3.0*(n-1) / (n+1)
    # 计算 `b2` 的方差 `varb2`
    varb2 = 24.0*n*(n-2.)*(n-3) / ((n+1)*(n+1.)*(n+3)*(n+5))
    # 计算标准化变量 `x`
    x = (b2-E)/ma.sqrt(varb2)
    # 计算 `sqrtbeta1`
    sqrtbeta1 = 6.0*(n*n-5*n+2)/((n+7)*(n+9)) * np.sqrt((6.0*(n+3)*(n+5)) /
                                                        (n*(n-2)*(n-3)))
    # 计算 `A`
    A = 6.0 + 8.0/sqrtbeta1 * (2.0/sqrtbeta1 + np.sqrt(1+4.0/(sqrtbeta1**2)))
    # 计算 `term1`
    term1 = 1 - 2./(9.0*A)
    # 计算 `denom`
    denom = 1 + x*ma.sqrt(2/(A-4.0))
    # 如果 `denom` 是掩码数组，则处理为掩码
    if np.ma.isMaskedArray(denom):
        denom[denom == 0.0] = masked
    elif denom == 0.0:
        denom = masked

    # 计算 `term2`，根据条件选择不同的表达式
    term2 = np.ma.where(denom > 0, ma.power((1-2.0/A)/denom, 1/3.0),
                        -ma.power(-(1-2.0/A)/denom, 1/3.0))
    # 计算 `Z` 值
    Z = (term1 - term2) / np.sqrt(2/(9.0*A))
    # 计算 `pvalue` 值，使用指定的备择假设
    pvalue = scipy.stats._stats_py._get_pvalue(Z, distributions.norm, alternative)

    # 返回计算结果，包括 Z 值和 p 值
    return KurtosistestResult(Z[()], pvalue[()])
# 定义一个命名元组 NormaltestResult，包含 statistic 和 pvalue 两个字段
NormaltestResult = namedtuple('NormaltestResult', ('statistic', 'pvalue'))

# 定义 normaltest 函数，用于检验样本数据是否符合正态分布
def normaltest(a, axis=0):
    """
    Tests whether a sample differs from a normal distribution.

    Parameters
    ----------
    a : array_like
        The array containing the data to be tested.
    axis : int or None, optional
        Axis along which to compute test. Default is 0. If None,
        compute over the whole array `a`.

    Returns
    -------
    statistic : float or array
        ``s^2 + k^2``, where ``s`` is the z-score returned by `skewtest` and
        ``k`` is the z-score returned by `kurtosistest`.
    pvalue : float or array
       A 2-sided chi squared probability for the hypothesis test.

    Notes
    -----
    For more details about `normaltest`, see `scipy.stats.normaltest`.

    """
    # 将输入数据 a 转换为数组，并处理轴向信息
    a, axis = _chk_asarray(a, axis)
    # 计算数据 a 的偏度检验得分 s
    s, _ = skewtest(a, axis)
    # 计算数据 a 的峰度检验得分 k
    k, _ = kurtosistest(a, axis)
    # 计算统计量 k2 = s^2 + k^2
    k2 = s*s + k*k

    # 返回 NormaltestResult 对象，包含统计量 k2 和基于自由度 2 的卡方分布的双侧概率 pvalue
    return NormaltestResult(k2, distributions.chi2.sf(k2, 2))


# 定义 mquantiles 函数，用于计算数据数组的经验分位数
def mquantiles(a, prob=list([.25,.5,.75]), alphap=.4, betap=.4, axis=None,
               limit=()):
    """
    Computes empirical quantiles for a data array.

    Samples quantile are defined by ``Q(p) = (1-gamma)*x[j] + gamma*x[j+1]``,
    where ``x[j]`` is the j-th order statistic, and gamma is a function of
    ``j = floor(n*p + m)``, ``m = alphap + p*(1 - alphap - betap)`` and
    ``g = n*p + m - j``.

    Reinterpreting the above equations to compare to **R** lead to the
    equation: ``p(k) = (k - alphap)/(n + 1 - alphap - betap)``

    Typical values of (alphap,betap) are:
        - (0,1)    : ``p(k) = k/n`` : linear interpolation of cdf
          (**R** type 4)
        - (.5,.5)  : ``p(k) = (k - 1/2.)/n`` : piecewise linear function
          (**R** type 5)
        - (0,0)    : ``p(k) = k/(n+1)`` :
          (**R** type 6)
        - (1,1)    : ``p(k) = (k-1)/(n-1)``: p(k) = mode[F(x[k])].
          (**R** type 7, **R** default)
        - (1/3,1/3): ``p(k) = (k-1/3)/(n+1/3)``: Then p(k) ~ median[F(x[k])].
          The resulting quantile estimates are approximately median-unbiased
          regardless of the distribution of x.
          (**R** type 8)
        - (3/8,3/8): ``p(k) = (k-3/8)/(n+1/4)``: Blom.
          The resulting quantile estimates are approximately unbiased
          if x is normally distributed
          (**R** type 9)
        - (.4,.4)  : approximately quantile unbiased (Cunnane)
        - (.35,.35): APL, used with PWM

    Parameters
    ----------
    a : array_like
        Input data, as a sequence or array of dimension at most 2.
    prob : array_like, optional
        List of quantiles to compute.
    alphap : float, optional
        Plotting positions parameter, default is 0.4.
    betap : float, optional
        Plotting positions parameter, default is 0.4.
    axis : int, optional
        Axis along which to perform the trimming.
        If None (default), the input array is first flattened.
    limit : tuple, optional
        Tuple of (lower, upper) values.
        Values of `a` outside this open interval are ignored.

    Returns
    -------
    mquantiles : MaskedArray
        An array containing the calculated quantiles.

    Notes
    -----
    This formulation is very similar to **R** except the calculation of
    ``m`` from ``alphap`` and ``betap``, where in **R** ``m`` is defined
    with each type.

    References
    ----------
    .. [1] *R* statistical software: https://www.r-project.org/
    .. [2] *R* ``quantile`` function:
            http://stat.ethz.ch/R-manual/R-devel/library/stats/html/quantile.html

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats.mstats import mquantiles
    >>> a = np.array([6., 47., 49., 15., 42., 41., 7., 39., 43., 40., 36.])
    >>> mquantiles(a)
    array([ 19.2,  40. ,  42.8])

    Using a 2D array, specifying axis and limit.

    >>> data = np.array([[   6.,    7.,    1.],
    ...                  [  47.,   15.,    2.],
    ...                  [  49.,   36.,    3.],
    ...                  [  15.,   39.,    4.],
    ...                  [  42.,   40., -999.],
    ...                  [  41.,   41., -999.],
    ...                  [   7., -999., -999.],
    ...                  [  39., -999., -999.],
    ...                  [  43., -999., -999.],
    ...                  [  40., -999., -999.],
    ...                  [  36., -999., -999.]])
    >>> print(mquantiles(data, axis=0, limit=(0, 50)))
    [[19.2  14.6   1.45]
     [40.   37.5   2.5 ]
     [42.8  40.05  3.55]]

    >>> data[:, 2] = -999.
    >>> print(mquantiles(data, axis=0, limit=(0, 50)))
    [[19.200000000000003 14.6 --]
     [40.0 37.5 --]
     [42.800000000000004 40.05 --]]

    """
    # 定义内部函数 _quantiles1D，用于计算一维数据的分位数
    def _quantiles1D(data,m,p):
        # 对数据进行排序并去除掩码
        x = np.sort(data.compressed())
        # 获取数据长度
        n = len(x)
        # 处理特殊情况：当数据长度为0时，返回空的掩码数组
        if n == 0:
            return ma.array(np.empty(len(p), dtype=float), mask=True)
        # 处理特殊情况：当数据长度为1时，返回重复数据构成的数组，不带掩码
        elif n == 1:
            return ma.array(np.resize(x, p.shape), mask=nomask)
        # 计算 aleph，并计算 k 和 gamma
        aleph = (n*p + m)
        k = np.floor(aleph.clip(1, n-1)).astype(int)
        gamma = (aleph-k).clip(0,1)
        # 返回分位数的计算结果
        return (1.-gamma)*x[(k-1).tolist()] + gamma*x[k.tolist()]

    # 将输入数组 a 转换为 MaskedArray 对象
    data = ma.array(a, copy=False)
    # 检查数据维度，不应大于二维
    if data.ndim > 2:
        raise TypeError("Array should be 2D at most !")

    # 如果设置了限制条件 limit，则对数据进行掩码处理
    if limit:
        condition = (limit[0] < data) & (data < limit[1])
        data[~condition.filled(True)] = masked

    # 将概率值 prob 转换为至少一维的数组
    p = np.atleast_1d(np.asarray(prob))
    # 根据 alphap 和 betap 计算 m
    m = alphap + p*(1.-alphap-betap)
    # 如果未指定 axis，则在全局计算分位数
    if (axis is None):
        return _quantiles1D(data, m, p)

    # 在指定的 axis 上应用 _quantiles1D 函数
    return ma.apply_along_axis(_quantiles1D, axis, data, m, p)
# 定义函数 scoreatpercentile，计算给定数据在指定百分位数上的得分
def scoreatpercentile(data, per, limit=(), alphap=.4, betap=.4):
    """Calculate the score at the given 'per' percentile of the
    sequence a.  For example, the score at per=50 is the median.

    This function is a shortcut to mquantile

    """
    # 检查百分位数 per 是否在有效范围内（0 到 100 之间）
    if (per < 0) or (per > 100.):
        raise ValueError("The percentile should be between 0. and 100. !"
                         " (got %s)" % per)

    # 调用 mquantiles 函数计算指定百分位数对应的得分，并返回结果
    return mquantiles(data, prob=[per/100.], alphap=alphap, betap=betap,
                      limit=limit, axis=0).squeeze()


# 定义函数 plotting_positions，计算数据的绘图位置（或经验百分位点）
def plotting_positions(data, alpha=0.4, beta=0.4):
    """
    Returns plotting positions (or empirical percentile points) for the data.

    Plotting positions are defined as ``(i-alpha)/(n+1-alpha-beta)``, where:
        - i is the rank order statistics
        - n is the number of unmasked values along the given axis
        - `alpha` and `beta` are two parameters.

    Typical values for `alpha` and `beta` are:
        - (0,1)    : ``p(k) = k/n``, linear interpolation of cdf (R, type 4)
        - (.5,.5)  : ``p(k) = (k-1/2.)/n``, piecewise linear function
          (R, type 5)
        - (0,0)    : ``p(k) = k/(n+1)``, Weibull (R type 6)
        - (1,1)    : ``p(k) = (k-1)/(n-1)``, in this case,
          ``p(k) = mode[F(x[k])]``. That's R default (R type 7)
        - (1/3,1/3): ``p(k) = (k-1/3)/(n+1/3)``, then
          ``p(k) ~ median[F(x[k])]``.
          The resulting quantile estimates are approximately median-unbiased
          regardless of the distribution of x. (R type 8)
        - (3/8,3/8): ``p(k) = (k-3/8)/(n+1/4)``, Blom.
          The resulting quantile estimates are approximately unbiased
          if x is normally distributed (R type 9)
        - (.4,.4)  : approximately quantile unbiased (Cunnane)
        - (.35,.35): APL, used with PWM
        - (.3175, .3175): used in scipy.stats.probplot

    Parameters
    ----------
    data : array_like
        Input data, as a sequence or array of dimension at most 2.
    alpha : float, optional
        Plotting positions parameter. Default is 0.4.
    beta : float, optional
        Plotting positions parameter. Default is 0.4.

    Returns
    -------
    positions : MaskedArray
        The calculated plotting positions.

    """
    # 将输入数据转换为 MaskedArray，处理缺失值
    data = ma.array(data, copy=False).reshape(1,-1)
    # 计算非遮蔽值的数量
    n = data.count()
    # 创建用于存储绘图位置的数组
    plpos = np.empty(data.size, dtype=float)
    # 将未计算的部分初始化为 0
    plpos[n:] = 0
    # 计算排序后的数据的绘图位置
    plpos[data.argsort(axis=None)[:n]] = ((np.arange(1, n+1) - alpha) /
                                          (n + 1.0 - alpha - beta))
    # 返回计算后的绘图位置，保留数据的遮蔽状态
    return ma.array(plpos, mask=data._mask)


# 创建别名函数 meppf，指向 plotting_positions 函数
meppf = plotting_positions


# 定义函数 obrientransform，执行输入数据的变换，用于一元统计前检验方差的齐性
def obrientransform(*args):
    """
    Computes a transform on input data (any number of columns).  Used to
    test for homogeneity of variance prior to running one-way stats.  Each
    array in ``*args`` is one level of a factor.  If an `f_oneway()` run on
    the transformed data and found significant, variances are unequal.   From
    Maxwell and Delaney, p.112.

    """
    # 将参数转换为数组，并转置以便进一步处理
    data = argstoarray(*args).T
    # 计算每列数据的方差（除以 N-1），axis=0 表示沿着列的方向计算
    v = data.var(axis=0, ddof=1)
    # 计算每列数据的均值
    m = data.mean(0)
    # 计算每列数据的有效数据点数，并转换为浮点数
    n = data.count(0).astype(float)
    # 计算 O'Brien 转换的结果，这里是逐步计算公式中的各项
    data -= m
    data **= 2
    data *= (n - 1.5) * n
    data -= 0.5 * v * (n - 1)
    data /= (n - 1.) * (n - 2.)
    # 检查计算后的方差是否与给定阈值近似相等，否则引发收敛不足的异常
    if not ma.allclose(v, data.mean(0)):
        raise ValueError("Lack of convergence in obrientransform.")

    # 返回经过 O'Brien 转换后的数据
    return data
def sem(a, axis=0, ddof=1):
    """
    Calculates the standard error of the mean of the input array.

    Also sometimes called standard error of measurement.

    Parameters
    ----------
    a : array_like
        An array containing the values for which the standard error is
        returned.
    axis : int or None, optional
        If axis is None, ravel `a` first. If axis is an integer, this will be
        the axis over which to operate. Defaults to 0.
    ddof : int, optional
        Delta degrees-of-freedom. How many degrees of freedom to adjust
        for bias in limited samples relative to the population estimate
        of variance. Defaults to 1.

    Returns
    -------
    s : ndarray or float
        The standard error of the mean in the sample(s), along the input axis.

    Notes
    -----
    The default value for `ddof` changed in scipy 0.15.0 to be consistent with
    `scipy.stats.sem` as well as with the most common definition used (like in
    the R documentation).

    Examples
    --------
    Find standard error along the first axis:

    >>> import numpy as np
    >>> from scipy import stats
    >>> a = np.arange(20).reshape(5,4)
    >>> print(stats.mstats.sem(a))
    [2.8284271247461903 2.8284271247461903 2.8284271247461903
     2.8284271247461903]

    Find standard error across the whole array, using n degrees of freedom:

    >>> print(stats.mstats.sem(a, axis=None, ddof=0))
    1.2893796958227628

    """
    # 将输入数组 `a` 和轴参数 `axis` 转换为数组和轴对象
    a, axis = _chk_asarray(a, axis)
    # 计算沿指定轴的元素个数
    n = a.count(axis=axis)
    # 计算标准差除以平方根后的值，作为样本均值的标准误差
    s = a.std(axis=axis, ddof=ddof) / ma.sqrt(n)
    return s


F_onewayResult = namedtuple('F_onewayResult', ('statistic', 'pvalue'))


def f_oneway(*args):
    """
    Performs a 1-way ANOVA, returning an F-value and probability given
    any number of groups.  From Heiman, pp.394-7.

    Usage: ``f_oneway(*args)``, where ``*args`` is 2 or more arrays,
    one per treatment group.

    Returns
    -------
    statistic : float
        The computed F-value of the test.
    pvalue : float
        The associated p-value from the F-distribution.

    """
    # 将输入的参数转换为数组形式，每行代表一个组
    data = argstoarray(*args)
    # 计算组数
    ngroups = len(data)
    # 计算总体样本数
    ntot = data.count()
    # 计算总平方和
    sstot = (data**2).sum() - (data.sum())**2/float(ntot)
    # 计算组间平方和
    ssbg = (data.count(-1) * (data.mean(-1)-data.mean())**2).sum()
    # 计算组内平方和
    sswg = sstot - ssbg
    # 计算组间自由度
    dfbg = ngroups - 1
    # 计算组内自由度
    dfwg = ntot - ngroups
    # 计算组间均方
    msb = ssbg / float(dfbg)
    # 计算组内均方
    msw = sswg / float(dfwg)
    # 计算 F 统计量
    f = msb / msw
    # 计算 F 分布的概率值
    prob = special.fdtrc(dfbg, dfwg, f)  # 等同于 stats.f.sf

    return F_onewayResult(f, prob)


FriedmanchisquareResult = namedtuple('FriedmanchisquareResult',
                                     ('statistic', 'pvalue'))


def friedmanchisquare(*args):
    """Friedman Chi-Square is a non-parametric, one-way within-subjects ANOVA.
    This function calculates the Friedman Chi-square test for repeated measures
    and returns the result, along with the associated probability value.

    ```
    # 将传入的参数转换为数组，并将
# 命名元组，用于存储 Brunner-Munzel 测试的结果，包括统计量和 p 值
BrunnerMunzelResult = namedtuple('BrunnerMunzelResult', ('statistic', 'pvalue'))


def brunnermunzel(x, y, alternative="two-sided", distribution="t"):
    """
    Compute the Brunner-Munzel test on samples x and y.

    Any missing values in `x` and/or `y` are discarded.

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
    alternative : 'less', 'two-sided', or 'greater', optional
        Whether to get the p-value for the one-sided hypothesis ('less'
        or 'greater') or for the two-sided hypothesis ('two-sided').
        Defaults value is 'two-sided' .
    distribution : 't' or 'normal', optional
        Whether to get the p-value by t-distribution or by standard normal
        distribution.
        Defaults value is 't' .

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
    For more details on `brunnermunzel`, see `scipy.stats.brunnermunzel`.

    Examples
    --------
    >>> from scipy.stats.mstats import brunnermunzel
    >>> import numpy as np
    >>> x1 = [1, 2, np.nan, np.nan, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1]
    >>> x2 = [3, 3, 4, 3, 1, 2, 3, 1, 1, 5, 4]
    >>> brunnermunzel(x1, x2)
    BrunnerMunzelResult(statistic=1.4723186918922935, pvalue=0.15479415300426624)  # may vary

    """  # noqa: E501
    # 压缩处理数组，去除缺失值，并将其视为 ndarray 类型
    x = ma.asarray(x).compressed().view(ndarray)
    y = ma.asarray(y).compressed().view(ndarray)
    # 计算样本 x 和 y 的长度
    nx = len(x)
    ny = len(y)
    # 如果任一样本长度为 0，则返回结果为 NaN
    if nx == 0 or ny == 0:
        return BrunnerMunzelResult(np.nan, np.nan)
    # 将 x 和 y 合并后计算秩次
    rankc = rankdata(np.concatenate((x,y)))
    # 分别提取 x 和 y 的秩次
    rankcx = rankc[0:nx]
    rankcy = rankc[nx:nx+ny]
    # 计算各自的平均秩次
    rankcx_mean = np.mean(rankcx)
    rankcy_mean = np.mean(rankcy)
    # 计算 x 和 y 的秩次
    rankx = rankdata(x)
    ranky = rankdata(y)
    # 计算 x 和 y 的平均秩次
    rankx_mean = np.mean(rankx)
    ranky_mean = np.mean(ranky)

    # 计算 Sx 和 Sy 统计量
    Sx = np.sum(np.power(rankcx - rankx - rankcx_mean + rankx_mean, 2.0))
    Sx /= nx - 1
    Sy = np.sum(np.power(rankcy - ranky - rankcy_mean + ranky_mean, 2.0))
    Sy /= ny - 1

    # 计算 Brunner-Munzel 统计量
    wbfn = nx * ny * (rankcy_mean - rankcx_mean)
    wbfn /= (nx + ny) * np.sqrt(nx * Sx + ny * Sy)
    # 如果分布类型为 t 分布
    if distribution == "t":
        # 计算分子部分的自由度
        df_numer = np.power(nx * Sx + ny * Sy, 2.0)
        # 计算分母部分的自由度
        df_denom = np.power(nx * Sx, 2.0) / (nx - 1)
        df_denom += np.power(ny * Sy, 2.0) / (ny - 1)
        # 计算 t 分布的自由度
        df = df_numer / df_denom
        # 计算累积分布函数的值，使用 t 分布
        p = distributions.t.cdf(wbfn, df)
    
    # 如果分布类型为正态分布
    elif distribution == "normal":
        # 计算累积分布函数的值，使用正态分布
        p = distributions.norm.cdf(wbfn)
    
    # 如果分布类型不是 't' 也不是 'normal'，则抛出错误
    else:
        raise ValueError(
            "distribution should be 't' or 'normal'")
    
    # 根据备择假设类型进行 p 值的调整
    if alternative == "greater":
        # 对于大于备择假设的情况，p 值不需要调整
        pass
    elif alternative == "less":
        # 对于小于备择假设的情况，将 p 值调整为 1 - p
        p = 1 - p
    elif alternative == "two-sided":
        # 对于双侧备择假设，将 p 值调整为 2 * min(p, 1 - p)
        p = 2 * np.min([p, 1-p])
    else:
        # 如果备择假设类型不是 'less', 'greater' 或 'two-sided'，则抛出错误
        raise ValueError(
            "alternative should be 'less', 'greater' or 'two-sided'")
    
    # 返回 BrunnerMunzelResult 对象，包含统计值和调整后的 p 值
    return BrunnerMunzelResult(wbfn, p)
```