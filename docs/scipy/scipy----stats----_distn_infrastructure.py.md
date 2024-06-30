# `D:\src\scipysrc\scipy\scipy\stats\_distn_infrastructure.py`

```
#
# Author:  Travis Oliphant  2002-2011 with contributions from
#          SciPy Developers 2004-2011
#

# 从 scipy._lib._util 模块导入 getfullargspec_no_self 函数并重命名为 _getfullargspec
from scipy._lib._util import getfullargspec_no_self as _getfullargspec

# 导入系统模块
import sys
# 导入关键字模块
import keyword
# 导入正则表达式模块
import re
# 导入类型模块
import types
# 导入警告模块
import warnings
# 从 itertools 模块导入 zip_longest 函数
from itertools import zip_longest

# 从 scipy._lib 模块导入 doccer
from scipy._lib import doccer
# 从 ._distr_params 模块导入 distcont, distdiscrete
from ._distr_params import distcont, distdiscrete
# 从 scipy._lib._util 模块导入 check_random_state
from scipy._lib._util import check_random_state

# 从 scipy.special 模块导入 comb, entr
from scipy.special import comb, entr


# 用于连续分布的根查找（ppf）和最大似然估计
# 导入 scipy.optimize 模块
from scipy import optimize

# 用于连续分布函数（例如矩、熵、cdf）的功能
# 导入 scipy.integrate 模块
from scipy import integrate

# 给定连续分布的cdf，近似其pdf
# 从 scipy._lib._finite_differences 模块导入 _derivative
from scipy._lib._finite_differences import _derivative

# 导入 scipy.stats 模块，用于 scipy.stats.entropy
from scipy import stats

# 导入 numpy 模块的若干函数和常量
from numpy import (arange, putmask, ones, shape, ndarray, zeros, floor,
                   logical_and, log, sqrt, place, argmax, vectorize, asarray,
                   nan, inf, isinf, empty)

# 导入 numpy 并重命名为 np
import numpy as np
# 从 ._constants 模块导入 _XMAX, _LOGXMAX
from ._constants import _XMAX, _LOGXMAX
# 从 ._censored_data 模块导入 CensoredData
from ._censored_data import CensoredData
# 从 scipy.stats._warnings_errors 模块导入 FitError

# 用于在特定分布文档字符串中进行替换的文档字符串部分
# 分布文档字符串的方法部分的占位符
docheaders = {'methods': """\nMethods\n-------\n""",
              # 分布文档字符串的笔记部分的占位符
              'notes': """\nNotes\n-----\n""",
              # 分布文档字符串的示例部分的占位符
              'examples': """\nExamples\n--------\n"""}

# 随机变量（rvs）的文档字符串模板
_doc_rvs = """\
rvs(%(shapes)s, loc=0, scale=1, size=1, random_state=None)
    Random variates.
"""
# 概率密度函数（pdf）的文档字符串模板
_doc_pdf = """\
pdf(x, %(shapes)s, loc=0, scale=1)
    Probability density function.
"""
# 概率密度函数的对数（logpdf）的文档字符串模板
_doc_logpdf = """\
logpdf(x, %(shapes)s, loc=0, scale=1)
    Log of the probability density function.
"""
# 概率质量函数（pmf）的文档字符串模板
_doc_pmf = """\
pmf(k, %(shapes)s, loc=0, scale=1)
    Probability mass function.
"""
# 概率质量函数的对数（logpmf）的文档字符串模板
_doc_logpmf = """\
logpmf(k, %(shapes)s, loc=0, scale=1)
    Log of the probability mass function.
"""
# 累积分布函数（cdf）的文档字符串模板
_doc_cdf = """\
cdf(x, %(shapes)s, loc=0, scale=1)
    Cumulative distribution function.
"""
# 累积分布函数的对数（logcdf）的文档字符串模板
_doc_logcdf = """\
logcdf(x, %(shapes)s, loc=0, scale=1)
    Log of the cumulative distribution function.
"""
# 生存函数（sf）的文档字符串模板
_doc_sf = """\
sf(x, %(shapes)s, loc=0, scale=1)
    Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
"""  # noqa: E501
# 生存函数的对数（logsf）的文档字符串模板
_doc_logsf = """\
logsf(x, %(shapes)s, loc=0, scale=1)
    Log of the survival function.
"""
# 百分点函数（ppf）的文档字符串模板
_doc_ppf = """\
ppf(q, %(shapes)s, loc=0, scale=1)
    Percent point function (inverse of ``cdf`` --- percentiles).
"""
# 逆生存函数（isf）的文档字符串模板
_doc_isf = """\
isf(q, %(shapes)s, loc=0, scale=1)
    Inverse survival function (inverse of ``sf``).
"""
# 指定阶数的非中心矩（moment）的文档字符串模板
_doc_moment = """\
moment(order, %(shapes)s, loc=0, scale=1)
    Non-central moment of the specified order.
"""
# 统计量（stats）的文档字符串模板
_doc_stats = """\
stats(%(shapes)s, loc=0, scale=1, moments='mv')
    Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
"""
# 熵（entropy）的文档字符串模板
_doc_entropy = """\
entropy(%(shapes)s, loc=0, scale=1)
    # Calculate the (differential) entropy of the random variable.
"""
_doc_fit = """\
fit(data)
    Parameter estimates for generic data.
    See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
    keyword arguments.
"""  # noqa: E501
_doc_expect = """\
expect(func, args=(%(shapes_)s), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
    Expected value of a function (of one argument) with respect to the distribution.
"""  # noqa: E501
_doc_expect_discrete = """\
expect(func, args=(%(shapes_)s), loc=0, lb=None, ub=None, conditional=False)
    Expected value of a function (of one argument) with respect to the distribution.
"""
_doc_median = """\
median(%(shapes)s, loc=0, scale=1)
    Median of the distribution.
"""
_doc_mean = """\
mean(%(shapes)s, loc=0, scale=1)
    Mean of the distribution.
"""
_doc_var = """\
var(%(shapes)s, loc=0, scale=1)
    Variance of the distribution.
"""
_doc_std = """\
std(%(shapes)s, loc=0, scale=1)
    Standard deviation of the distribution.
"""
_doc_interval = """\
interval(confidence, %(shapes)s, loc=0, scale=1)
    Confidence interval with equal areas around the median.
"""
_doc_allmethods = ''.join([docheaders['methods'], _doc_rvs, _doc_pdf,
                           _doc_logpdf, _doc_cdf, _doc_logcdf, _doc_sf,
                           _doc_logsf, _doc_ppf, _doc_isf, _doc_moment,
                           _doc_stats, _doc_entropy, _doc_fit,
                           _doc_expect, _doc_median,
                           _doc_mean, _doc_var, _doc_std, _doc_interval])

_doc_default_longsummary = """\
As an instance of the `rv_continuous` class, `%(name)s` object inherits from it
a collection of generic methods (see below for the full list),
and completes them with details specific for this particular distribution.
"""

_doc_default_frozen_note = """
Alternatively, the object may be called (as a function) to fix the shape,
location, and scale parameters returning a "frozen" continuous RV object:

rv = %(name)s(%(shapes)s, loc=0, scale=1)
    - Frozen RV object with the same methods but holding the given shape,
      location, and scale fixed.
"""
_doc_default_example = """\
Examples
--------
>>> import numpy as np
>>> from scipy.stats import %(name)s
>>> import matplotlib.pyplot as plt
>>> fig, ax = plt.subplots(1, 1)

Calculate the first four moments:

%(set_vals_stmt)s
>>> mean, var, skew, kurt = %(name)s.stats(%(shapes)s, moments='mvsk')

Display the probability density function (``pdf``):

>>> x = np.linspace(%(name)s.ppf(0.01, %(shapes)s),
...                 %(name)s.ppf(0.99, %(shapes)s), 100)
>>> ax.plot(x, %(name)s.pdf(x, %(shapes)s),
...        'r-', lw=5, alpha=0.6, label='%(name)s pdf')

Alternatively, the distribution object can be called (as a function)
to fix the shape, location and scale parameters. This returns a "frozen"
RV object holding the given parameters fixed.


"""
# 公式中参数 `%(name)s` 和 `%(shapes)s` 是占位符，代表特定分布的名称和形状参数

_doc_default_locscale = """\
概率密度函数在标准化形式下定义。要改变分布的位置和/或尺度，请使用 `loc` 和 `scale` 参数。
具体来说，`%(name)s.pdf(x, %(shapes)s, loc, scale)` 等价于 `%(name)s.pdf(y, %(shapes)s) / scale`，
其中 `y = (x - loc) / scale`。注意，改变分布的位置并不使其成为“非中心”分布；
某些分布的非中心推广在单独的类中提供。
"""

_doc_default = ''.join([_doc_default_longsummary,
                        _doc_allmethods,
                        '\n',
                        _doc_default_example])

_doc_default_before_notes = ''.join([_doc_default_longsummary,
                                     _doc_allmethods])

# 将文档字符串按照特定方法映射到对应的文档内容
docdict = {
    'rvs': _doc_rvs,
    'pdf': _doc_pdf,
    'logpdf': _doc_logpdf,
    'cdf': _doc_cdf,
    'logcdf': _doc_logcdf,
    'sf': _doc_sf,
    'logsf': _doc_logsf,
    'ppf': _doc_ppf,
    'isf': _doc_isf,
    'stats': _doc_stats,
    'entropy': _doc_entropy,
    'fit': _doc_fit,
    'moment': _doc_moment,
    'expect': _doc_expect,
    'interval': _doc_interval,
    'mean': _doc_mean,
    'std': _doc_std,
    'var': _doc_var,
    'median': _doc_median,
    'allmethods': _doc_allmethods,
    'longsummary': _doc_default_longsummary,
    'frozennote': _doc_default_frozen_note,
    'example': _doc_default_example,
    'default': _doc_default,
    'before_notes': _doc_default_before_notes,
    'after_notes': _doc_default_locscale
}

# 在离散分布的文档字典中复用连续分布文档的共同内容，改变一些细节内容
docdict_discrete = docdict.copy()

# 添加离散分布特有的方法和文档字符串
docdict_discrete['pmf'] = _doc_pmf
docdict_discrete['logpmf'] = _doc_logpmf
docdict_discrete['expect'] = _doc_expect_discrete

# 修改离散分布方法中的文档字符串，移除 `scale=1` 参数
_doc_disc_methods = ['rvs', 'pmf', 'logpmf', 'cdf', 'logcdf', 'sf', 'logsf',
                     'ppf', 'isf', 'stats', 'entropy', 'expect', 'median',
                     'mean', 'var', 'std', 'interval']
for obj in _doc_disc_methods:
    docdict_discrete[obj] = docdict_discrete[obj].replace(', scale=1', '')

# 修改特定离散分布方法的参数变量名
_doc_disc_methods_err_varname = ['cdf', 'logcdf', 'sf', 'logsf']
for obj in _doc_disc_methods_err_varname:
    docdict_discrete[obj] = docdict_discrete[obj].replace('(x, ', '(k, ')

# 移除离散分布不适用的文档字符串
docdict_discrete.pop('pdf')
docdict_discrete.pop('logpdf')
# 拼接所有离散分布方法的文档字符串，并将结果存储在_doc_allmethods变量中
_doc_allmethods = ''.join([docdict_discrete[obj] for obj in _doc_disc_methods])

# 将_doc_allmethods添加到docdict_discrete字典中，键为'allmethods'，表示包含所有方法的文档内容
docdict_discrete['allmethods'] = docheaders['methods'] + _doc_allmethods

# 将默认长摘要中的字符串'rv_continuous'替换为'rv_discrete'，并将结果存储在_doc_default_longsummary变量中
docdict_discrete['longsummary'] = _doc_default_longsummary.replace(
    'rv_continuous', 'rv_discrete')

# 定义_doc_default_frozen_note变量，包含一个多行字符串，描述如何调用对象以获取冻结的离散随机变量
_doc_default_frozen_note = """
Alternatively, the object may be called (as a function) to fix the shape and
location parameters returning a "frozen" discrete RV object:

rv = %(name)s(%(shapes)s, loc=0)
    - Frozen RV object with the same methods but holding the given shape and
      location fixed.
"""

# 将_doc_default_frozen_note添加到docdict_discrete字典中，键为'frozennote'，表示冻结离散随机变量的说明文档
docdict_discrete['frozennote'] = _doc_default_frozen_note

# 定义_doc_default_discrete_example变量，包含一个多行字符串，展示如何使用离散随机变量分布对象的示例
_doc_default_discrete_example = """\
Examples
--------
>>> import numpy as np
>>> from scipy.stats import %(name)s
>>> import matplotlib.pyplot as plt
>>> fig, ax = plt.subplots(1, 1)

Calculate the first four moments:

%(set_vals_stmt)s
>>> mean, var, skew, kurt = %(name)s.stats(%(shapes)s, moments='mvsk')

Display the probability mass function (``pmf``):

>>> x = np.arange(%(name)s.ppf(0.01, %(shapes)s),
...               %(name)s.ppf(0.99, %(shapes)s))
>>> ax.plot(x, %(name)s.pmf(x, %(shapes)s), 'bo', ms=8, label='%(name)s pmf')
>>> ax.vlines(x, 0, %(name)s.pmf(x, %(shapes)s), colors='b', lw=5, alpha=0.5)

Alternatively, the distribution object can be called (as a function)
to fix the shape and location. This returns a "frozen" RV object holding
the given parameters fixed.

Freeze the distribution and display the frozen ``pmf``:

>>> rv = %(name)s(%(shapes)s)
>>> ax.vlines(x, 0, rv.pmf(x), colors='k', linestyles='-', lw=1,
...         label='frozen pmf')
>>> ax.legend(loc='best', frameon=False)
>>> plt.show()

Check accuracy of ``cdf`` and ``ppf``:

>>> prob = %(name)s.cdf(x, %(shapes)s)
>>> np.allclose(x, %(name)s.ppf(prob, %(shapes)s))
True

Generate random numbers:

>>> r = %(name)s.rvs(%(shapes)s, size=1000)
"""

# 将_doc_default_discrete_example添加到docdict_discrete字典中，键为'example'，表示离散分布的示例文档
docdict_discrete['example'] = _doc_default_discrete_example

# 定义_doc_default_discrete_locscale变量，包含一个多行字符串，描述离散分布的loc和scale参数如何影响概率质量函数
_doc_default_discrete_locscale = """\
The probability mass function above is defined in the "standardized" form.
To shift distribution use the ``loc`` parameter.
Specifically, ``%(name)s.pmf(k, %(shapes)s, loc)`` is identically
equivalent to ``%(name)s.pmf(k - loc, %(shapes)s)``.
"""

# 将_doc_default_discrete_locscale添加到docdict_discrete字典中，键为'after_notes'，表示在注释之后的说明文档
docdict_discrete['after_notes'] = _doc_default_discrete_locscale

# 拼接默认长摘要和所有方法文档，并将结果存储在_doc_default_before_notes变量中
_doc_default_before_notes = ''.join([docdict_discrete['longsummary'],
                                     docdict_discrete['allmethods']])

# 将_doc_default_before_notes添加到docdict_discrete字典中，键为'before_notes'，表示在注释之前的说明文档
docdict_discrete['before_notes'] = _doc_default_before_notes

# 拼接默认长摘要、所有方法文档、冻结离散随机变量的注释和示例文档，并将结果存储在_doc_default_disc变量中
_doc_default_disc = ''.join([docdict_discrete['longsummary'],
                             docdict_discrete['allmethods'],
                             docdict_discrete['frozennote'],
                             docdict_discrete['example']])

# 将_doc_default_disc添加到docdict_discrete字典中，键为'default'，表示默认的离散分布说明文档
docdict_discrete['default'] = _doc_default_disc

# 清理所有单独的文档字符串元素，不再需要它们
for obj in [s for s in dir() if s.startswith('_doc_')]:
    exec('del ' + obj)
del obj
    # 如果 mu 为 None，则计算 data 的均值 mu
    if mu is None:
        mu = data.mean()
    # 返回 data 减去 mu 后的 n 次方的均值
    return ((data - mu)**n).mean()
# 根据给定的统计量计算指定的矩(moment)
def _moment_from_stats(n, mu, mu2, g1, g2, moment_func, args):
    if (n == 0):
        return 1.0  # 如果 n 为 0，则返回 1.0
    elif (n == 1):
        if mu is None:
            val = moment_func(1, *args)  # 如果 mu 为 None，则调用指定的 moment_func 计算第一个矩
        else:
            val = mu  # 否则直接使用给定的 mu
    elif (n == 2):
        if mu2 is None or mu is None:
            val = moment_func(2, *args)  # 如果 mu2 或 mu 为 None，则调用指定的 moment_func 计算第二个矩
        else:
            val = mu2 + mu * mu  # 否则计算第二个矩，mu2 + mu^2
    elif (n == 3):
        if g1 is None or mu2 is None or mu is None:
            val = moment_func(3, *args)  # 如果 g1 或 mu2 或 mu 为 None，则调用指定的 moment_func 计算第三个矩
        else:
            mu3 = g1 * np.power(mu2, 1.5)  # 计算第三个中心矩，g1 * mu2^1.5
            val = mu3 + 3 * mu * mu2 + mu * mu * mu  # 计算第三个非中心矩
    elif (n == 4):
        if g1 is None or g2 is None or mu2 is None or mu is None:
            val = moment_func(4, *args)  # 如果 g1 或 g2 或 mu2 或 mu 为 None，则调用指定的 moment_func 计算第四个矩
        else:
            mu4 = (g2 + 3.0) * (mu2 ** 2.0)  # 计算第四个中心矩
            mu3 = g1 * np.power(mu2, 1.5)  # 计算第三个中心矩
            val = mu4 + 4 * mu * mu3 + 6 * mu * mu * mu2 + mu * mu * mu * mu  # 计算第四个非中心矩
    else:
        val = moment_func(n, *args)  # 对于其他 n，调用指定的 moment_func 计算对应的矩

    return val  # 返回计算得到的矩的值


def _skew(data):
    """
    skew is third central moment / variance**(1.5)
    """
    data = np.ravel(data)  # 将输入数据展平成一维数组
    mu = data.mean()  # 计算数据的均值
    m2 = ((data - mu)**2).mean()  # 计算数据的方差
    m3 = ((data - mu)**3).mean()  # 计算数据的第三个中心矩
    return m3 / np.power(m2, 1.5)  # 返回偏度(skew)：第三个中心矩除以方差的1.5次方根


def _kurtosis(data):
    """Fisher's excess kurtosis is fourth central moment / variance**2 - 3."""
    data = np.ravel(data)  # 将输入数据展平成一维数组
    mu = data.mean()  # 计算数据的均值
    m2 = ((data - mu)**2).mean()  # 计算数据的方差
    m4 = ((data - mu)**4).mean()  # 计算数据的第四个中心矩
    return m4 / m2**2 - 3  # 返回峰度(kurtosis)：第四个中心矩除以方差的平方，减去 3


def _vectorize_rvs_over_shapes(_rvs1):
    """Decorator that vectorizes _rvs method to work on ndarray shapes"""
    # _rvs1 must be a _function_ that accepts _scalar_ args as positional
    # arguments, `size` and `random_state` as keyword arguments.
    # _rvs1 must return a random variate array with shape `size`. If `size` is
    # None, _rvs1 must return a scalar.
    # When applied to _rvs1, this decorator broadcasts ndarray args
    # and loops over them, calling _rvs1 for each set of scalar args.
    # For usage example, see _nchypergeom_gen
    # 定义一个函数 _rvs，接受任意数量的位置参数 args，以及 size 和 random_state 两个关键字参数
    def _rvs(*args, size, random_state):
        
        # 使用第一个参数的形状和给定的 size 调用 _check_shape 函数，获取调整后的 size 和索引
        _rvs1_size, _rvs1_indices = _check_shape(args[0].shape, size)
        
        # 将 size 转换为 NumPy 数组
        size = np.array(size)
        # 将 _rvs1_size 和 _rvs1_indices 转换为 NumPy 数组
        _rvs1_size = np.array(_rvs1_size)
        _rvs1_indices = np.array(_rvs1_indices)
        
        # 如果所有的 _rvs1_indices 都为 True，说明所有参数都是标量（scalar）
        if np.all(_rvs1_indices):  # all args are scalars
            # 调用 _rvs1 函数处理所有参数并返回结果
            return _rvs1(*args, size, random_state)
        
        # 创建一个空的 NumPy 数组 out，其形状为 size
        out = np.empty(size)

        # out.shape 可能混合了与 arg_shape 和 _rvs1_size 相关的维度
        # 为了便于索引，将它们按照 arg_shape + _rvs1_size 的顺序排序
        j0 = np.arange(out.ndim)
        j1 = np.hstack((j0[~_rvs1_indices], j0[_rvs1_indices]))
        out = np.moveaxis(out, j1, j0)

        # 使用 np.ndindex(*size[~_rvs1_indices]) 遍历 out 中非标量参数对应的维度
        for i in np.ndindex(*size[~_rvs1_indices]):
            # 将每个参数按需挤压（squeeze），因为单维度将与 _rvs1_size 而不是 arg_shape 关联
            # 调用 _rvs1 函数处理挤压后的参数，并将结果赋值给 out[i]
            out[i] = _rvs1(*[np.squeeze(arg)[i] for arg in args],
                           _rvs1_size, random_state)

        # 返回结果之前，将轴移回原来的顺序 j0 到 j1
        return np.moveaxis(out, j0, j1)  # move axes back before returning

    # 返回 _rvs 函数本身
    return _rvs
# 确定优化器函数的选择，确保其为可调用对象或字符串，并处理字符串优化器的命名规范
def _fit_determine_optimizer(optimizer):
    if not callable(optimizer) and isinstance(optimizer, str):
        # 如果优化器不是以'fmin_'开头，则添加'fmin_'前缀
        if not optimizer.startswith('fmin_'):
            optimizer = "fmin_"+optimizer
        # 处理特殊情况，如果优化器仅为'fmin_'，则修改为'fmin'
        if optimizer == 'fmin_':
            optimizer = 'fmin'
        try:
            # 尝试从optimize模块中获取对应名称的优化函数
            optimizer = getattr(optimize, optimizer)
        except AttributeError as e:
            # 如果获取失败，则抛出异常
            raise ValueError("%s is not a valid optimizer" % optimizer) from e
    # 返回处理后的优化器对象
    return optimizer


# 判断输入值是否为整数
def _isintegral(x):
    return x == np.round(x)


# 计算一维数组x中有限值的总和及非有限值的数量
def _sum_finite(x):
    """
    For a 1D array x, return a tuple containing the sum of the
    finite values of x and the number of nonfinite values.

    This is a utility function used when evaluating the negative
    loglikelihood for a distribution and an array of samples.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats._distn_infrastructure import _sum_finite
    >>> tot, nbad = _sum_finite(np.array([-2, -np.inf, 5, 1]))
    >>> tot
    4.0
    >>> nbad
    1
    """
    # 找出数组x中的有限值
    finite_x = np.isfinite(x)
    # 计算非有限值的数量
    bad_count = finite_x.size - np.count_nonzero(finite_x)
    # 返回有限值的总和及非有限值的数量
    return np.sum(x[finite_x]), bad_count


# Frozen RV类，用于封装冻结的随机变量
class rv_frozen:

    def __init__(self, dist, *args, **kwds):
        # 存储位置参数和关键字参数
        self.args = args
        self.kwds = kwds

        # 创建一个新的分布实例
        self.dist = dist.__class__(**dist._updated_ctor_param())

        # 解析参数，获取分布的支持区间
        shapes, _, _ = self.dist._parse_args(*args, **kwds)
        self.a, self.b = self.dist._get_support(*shapes)

    @property
    def random_state(self):
        # 返回分布的随机数生成器状态
        return self.dist._random_state

    @random_state.setter
    def random_state(self, seed):
        # 设置分布的随机数生成器状态
        self.dist._random_state = check_random_state(seed)

    def cdf(self, x):
        # 累积分布函数（CDF）
        return self.dist.cdf(x, *self.args, **self.kwds)

    def logcdf(self, x):
        # 对数累积分布函数
        return self.dist.logcdf(x, *self.args, **self.kwds)

    def ppf(self, q):
        # 百分点函数（CDF的逆函数）
        return self.dist.ppf(q, *self.args, **self.kwds)

    def isf(self, q):
        # 逆生存函数（1 - CDF的逆函数）
        return self.dist.isf(q, *self.args, **self.kwds)

    def rvs(self, size=None, random_state=None):
        # 生成随机样本
        kwds = self.kwds.copy()
        kwds.update({'size': size, 'random_state': random_state})
        return self.dist.rvs(*self.args, **kwds)

    def sf(self, x):
        # 生存函数（1 - CDF）
        return self.dist.sf(x, *self.args, **self.kwds)

    def logsf(self, x):
        # 对数生存函数
        return self.dist.logsf(x, *self.args, **self.kwds)

    def stats(self, moments='mv'):
        # 统计特性（如均值、方差等）
        kwds = self.kwds.copy()
        kwds.update({'moments': moments})
        return self.dist.stats(*self.args, **kwds)

    def median(self):
        # 中位数
        return self.dist.median(*self.args, **self.kwds)

    def mean(self):
        # 均值
        return self.dist.mean(*self.args, **self.kwds)

    def var(self):
        # 方差
        return self.dist.var(*self.args, **self.kwds)

    def std(self):
        # 标准差
        return self.dist.std(*self.args, **self.kwds)

    def moment(self, order=None):
        # 矩
        return self.dist.moment(order, *self.args, **self.kwds)
    # 计算分布的熵并返回结果
    def entropy(self):
        return self.dist.entropy(*self.args, **self.kwds)

    # 计算置信区间并返回结果
    def interval(self, confidence=None):
        return self.dist.interval(confidence, *self.args, **self.kwds)

    # 计算期望值并返回结果
    def expect(self, func=None, lb=None, ub=None, conditional=False, **kwds):
        # expect方法仅接受形状参数作为位置参数
        # 因此需要将self.args、self.kwds以及loc/scale进行转换
        # 其它参数的含义请参见.expect方法的文档字符串
        a, loc, scale = self.dist._parse_args(*self.args, **self.kwds)
        if isinstance(self.dist, rv_discrete):
            # 如果分布是离散型，则调用rv_discrete类的expect方法
            return self.dist.expect(func, a, loc, lb, ub, conditional, **kwds)
        else:
            # 如果分布是连续型，则调用expect方法
            return self.dist.expect(func, a, loc, scale, lb, ub,
                                    conditional, **kwds)

    # 返回分布的支持区间
    def support(self):
        return self.dist.support(*self.args, **self.kwds)
class rv_discrete_frozen(rv_frozen):
    """Subclass of rv_frozen for discrete random variables.

    Attributes:
        dist: The underlying distribution object.
        args: Positional arguments passed to the distribution methods.
        kwds: Keyword arguments passed to the distribution methods.
    """

    def pmf(self, k):
        """Probability mass function for the discrete distribution.

        Args:
            k: Value at which to evaluate the PMF.

        Returns:
            Probability mass at k.
        """
        return self.dist.pmf(k, *self.args, **self.kwds)

    def logpmf(self, k):
        """Logarithm of the probability mass function for discrete distribution.

        Args:
            k: Value at which to evaluate the log PMF.

        Returns:
            Logarithm of the probability mass at k.
        """
        return self.dist.logpmf(k, *self.args, **self.kwds)


class rv_continuous_frozen(rv_frozen):
    """Subclass of rv_frozen for continuous random variables.

    Attributes:
        dist: The underlying distribution object.
        args: Positional arguments passed to the distribution methods.
        kwds: Keyword arguments passed to the distribution methods.
    """

    def pdf(self, x):
        """Probability density function for the continuous distribution.

        Args:
            x: Value at which to evaluate the PDF.

        Returns:
            Probability density at x.
        """
        return self.dist.pdf(x, *self.args, **self.kwds)

    def logpdf(self, x):
        """Logarithm of the probability density function for continuous distribution.

        Args:
            x: Value at which to evaluate the log PDF.

        Returns:
            Logarithm of the probability density at x.
        """
        return self.dist.logpdf(x, *self.args, **self.kwds)


def argsreduce(cond, *args):
    """Clean arguments to:

    1. Ensure all arguments are iterable (arrays of dimension at least one
    2. If cond != True and size > 1, ravel(args[i]) where ravel(condition) is
       True, in 1D.

    Return list of processed arguments.

    Examples and detailed explanation provided in the docstring.
    """
    # Ensure all arguments are at least 1-dimensional arrays
    newargs = np.atleast_1d(*args)

    if not isinstance(newargs, (list, tuple)):
        newargs = (newargs,)

    if np.all(cond):
        # Broadcast arrays with cond
        *newargs, cond = np.broadcast_arrays(*newargs, cond)
        return [arg.ravel() for arg in newargs]

    s = cond.shape
    # Extract flattened arrays based on cond for broadcasting
    return [(arg if np.size(arg) == 1
            else np.extract(cond, np.broadcast_to(arg, s)))
            for arg in newargs]


parse_arg_template = """
def _parse_args(self, %(shape_arg_str)s %(locscale_in)s):
    return (%(shape_arg_str)s), %(locscale_out)s

def _parse_args_rvs(self, %(shape_arg_str)s %(locscale_in)s, size=None):
    return self._argcheck_rvs(%(shape_arg_str)s %(locscale_out)s, size=size)

def _parse_args_stats(self, %(shape_arg_str)s %(locscale_in)s, moments='mv'):
    return (%(shape_arg_str)s), %(locscale_out)s, moments
"""


class rv_generic:
    """Class which encapsulates common functionality between rv_discrete
    and rv_continuous.

    Attributes:
        _stats_has_moments: Boolean indicating if '_stats' method has 'moments' keyword.
        _random_state: Random number generator state.
    """

    def __init__(self, seed=None):
        """Initialize with optional random seed.

        Args:
            seed: Seed value for random number generation.
        """
        super().__init__()

        # Determine if _stats method supports 'moments' keyword
        sig = _getfullargspec(self._stats)
        self._stats_has_moments = ((sig.varkw is not None) or
                                   ('moments' in sig.args) or
                                   ('moments' in sig.kwonlyargs))
        # Initialize random state
        self._random_state = check_random_state(seed)

    @property
    # 返回当前对象的_random_state属性，用于生成随机变量的生成器对象
    def random_state(self):
        """Get or set the generator object for generating random variates.

        If `random_state` is None (or `np.random`), the
        `numpy.random.RandomState` singleton is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` or ``RandomState``
        instance, that instance is used.

        """
        return self._random_state

    @random_state.setter
    # 设置_random_state属性，用于设定生成随机变量的生成器对象
    def random_state(self, seed):
        self._random_state = check_random_state(seed)

    # 重构对象状态的方法，从给定的状态中更新当前对象
    def __setstate__(self, state):
        try:
            self.__dict__.update(state)
            # 给每个实例动态创建的方法附加上
            # 如果子类重写了rv_generic.__setstate__，或者实现了自己的_attach_methods，
            # 那么必须确保调用_attach_argparser_methods。
            self._attach_methods()
        except ValueError:
            # 重构旧版本pickle（scipy<1.6）中包含的状态(_ctor_param, random_state)
            self._ctor_param = state[0]
            self._random_state = state[1]
            self.__init__()

    # 将动态创建的方法附加到rv_*实例上的方法
    def _attach_methods(self):
        """Attaches dynamically created methods to the rv_* instance.

        This method must be overridden by subclasses, and must itself call
         _attach_argparser_methods. This method is called in __init__ in
         subclasses, and in __setstate__
        """
        raise NotImplementedError

    # 生成动态参数解析函数并将其附加到实例上的方法
    def _attach_argparser_methods(self):
        """
        Generates the argument-parsing functions dynamically and attaches
        them to the instance.

        Should be called from `_attach_methods`, typically in __init__ and
        during unpickling (__setstate__)
        """
        ns = {}
        # 通过执行self._parse_arg_template生成解析参数函数
        exec(self._parse_arg_template, ns)
        # 注意：将生成的方法附加到实例而不是类
        for name in ['_parse_args', '_parse_args_stats', '_parse_args_rvs']:
            setattr(self, name, types.MethodType(ns[name], self))
    def _construct_doc(self, docdict, shapes_vals=None):
        """Construct the instance docstring with string substitutions."""
        # 创建一个临时副本，以防修改原始文档字典
        tempdict = docdict.copy()
        # 将实例的名称或默认值设置为 'distname'
        tempdict['name'] = self.name or 'distname'
        # 将实例的形状参数设置为字符串，如果没有则为空字符串
        tempdict['shapes'] = self.shapes or ''

        # 如果未提供形状参数的具体值，则初始化为空元组
        if shapes_vals is None:
            shapes_vals = ()
        # 格式化形状参数的值，保留小数点后三位
        vals = ', '.join('%.3g' % val for val in shapes_vals)
        tempdict['vals'] = vals

        # 将实例的形状参数再次存储为字符串，如果没有则为空字符串
        tempdict['shapes_'] = self.shapes or ''
        # 如果实例具有形状参数且只有一个参数，则在其后添加逗号
        if self.shapes and self.numargs == 1:
            tempdict['shapes_'] += ','

        # 如果实例具有形状参数，则创建设置形状参数值的语句
        if self.shapes:
            tempdict['set_vals_stmt'] = f'>>> {self.shapes} = {vals}'
        else:
            tempdict['set_vals_stmt'] = ''

        # 如果实例没有形状参数
        if self.shapes is None:
            # 从调用参数中删除形状参数（如果存在）
            for item in ['default', 'before_notes']:
                tempdict[item] = tempdict[item].replace(
                    "\n%(shapes)s : array_like\n    shape parameters", "")

        # 进行两次循环是因为我们在两个形式中使用了 %(shapes)s（有和没有逗号）
        for i in range(2):
            # 如果实例没有形状参数，将其从文档字符串中删除
            if self.shapes is None:
                self.__doc__ = self.__doc__.replace("%(shapes)s, ", "")
            try:
                # 使用文档字典格式化文档字符串
                self.__doc__ = doccer.docformat(self.__doc__, tempdict)
            except TypeError as e:
                # 如果出现类型错误，则抛出异常
                raise Exception("Unable to construct docstring for "
                                f"distribution \"{self.name}\": {repr(e)}") from e

        # 修正空形状参数的情况
        self.__doc__ = self.__doc__.replace('(, ', '(').replace(', )', ')')

    def _construct_default_doc(self, longname=None,
                               docdict=None, discrete='continuous'):
        """Construct instance docstring from the default template."""
        # 如果未提供 longname，默认设为 'A'
        if longname is None:
            longname = 'A'
        # 使用默认模板构建实例的文档字符串
        self.__doc__ = ''.join([f'{longname} {discrete} random variable.',
                                '\n\n%(before_notes)s\n', docheaders['notes'],
                                '\n%(example)s'])
        # 使用给定的文档字典构建文档字符串
        self._construct_doc(docdict)

    def freeze(self, *args, **kwds):
        """Freeze the distribution for the given arguments.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution.  Should include all
            the non-optional arguments, may include ``loc`` and ``scale``.

        Returns
        -------
        rv_frozen : rv_frozen instance
            The frozen distribution.

        """
        # 如果实例是 rv_continuous 的子类，则返回 rv_continuous_frozen 对象
        if isinstance(self, rv_continuous):
            return rv_continuous_frozen(self, *args, **kwds)
        else:
            # 否则返回 rv_discrete_frozen 对象
            return rv_discrete_frozen(self, *args, **kwds)

    def __call__(self, *args, **kwds):
        # 将实例调用转发到 freeze 方法
        return self.freeze(*args, **kwds)
    # 将 __call__ 方法的文档字符串设置为 freeze 方法的文档字符串
    __call__.__doc__ = freeze.__doc__

    # 实际的计算函数（无需进行基本检查）
    # 如果定义了这些函数，则其他函数将不会被查看。
    # 否则，可以定义另一个集合。
    def _stats(self, *args, **kwds):
        return None, None, None, None
    
    # 非中心矩（也称为关于原点的矩）。
    # 用 LaTeX 表示，munp 将是 $\mu'_{n}$，即“mu-sub-n-prime”。
    # 带撇的 mu 是非中心矩的广泛使用符号。
    def _munp(self, n, *args):
        # 抑制积分时的浮点数警告。
        with np.errstate(all='ignore'):
            vals = self.generic_moment(n, *args)
        return vals
    
    # 这些是必须定义的方法（标准形式函数）。
    # 注意：rv_continuous 和 rv_discrete 的通用 _pdf、_logpdf、_cdf 是不同的，因此在这些类中进行了定义。
    def _argcheck(self, *args):
        """检查参数和关键字参数的正确性的默认函数。
    
        返回条件数组，其中参数正确时为1，不正确时为0。
    
        """
        cond = 1
        for arg in args:
            cond = logical_and(cond, (asarray(arg) > 0))
        return cond
    
    def _get_support(self, *args, **kwargs):
        """返回（未缩放、未移位）分布的支持范围。
    
        *必须* 被依赖于分布形状参数的支持范围的分布所覆盖。
        任何这样的覆盖*不得*设置或更改任何类成员，因为这些成员在分布的所有实例之间共享。
    
        Parameters
        ----------
        arg1, arg2, ... : array_like
            分布的形状参数（更多信息请参见实例对象的文档字符串）。
    
        Returns
        -------
        a, b : numeric (float, int 或 +/-np.inf)
            分布在指定形状参数下的端点。
    
        """
        return self.a, self.b
    
    def _support_mask(self, x, *args):
        a, b = self._get_support(*args)
        with np.errstate(invalid='ignore'):
            return (a <= x) & (x <= b)
    
    def _open_support_mask(self, x, *args):
        a, b = self._get_support(*args)
        with np.errstate(invalid='ignore'):
            return (a < x) & (x < b)
    
    def _rvs(self, *args, size=None, random_state=None):
        # 此方法必须处理 size 为元组的情况，必须正确地广播 *args 和 size。
        # size 可能是一个空元组，这意味着要生成一个标量随机变量。
    
        # 使用基本的反函数累积分布函数算法作为默认值进行随机变量生成。
        U = random_state.uniform(size=size)
        Y = self._ppf(U, *args)
        return Y
    
    def _logcdf(self, x, *args):
        with np.errstate(divide='ignore'):
            return log(self._cdf(x, *args))
    
    def _sf(self, x, *args):
        return 1.0-self._cdf(x, *args)
    
    def _logsf(self, x, *args):
        with np.errstate(divide='ignore'):
            return log(self._sf(x, *args))
    # 将 q 参数和其他参数传递给 _ppfvec 方法，并返回结果
    def _ppf(self, q, *args):
        return self._ppfvec(q, *args)

    # 将 1.0-q 参数和其他参数传递给 _ppf 方法，并返回结果
    def _isf(self, q, *args):
        return self._ppf(1.0-q, *args)  # use correct _ppf for subclasses

    # 实际调用这些方法，并且如果想要保留错误检查，则不应该被覆盖
    def rvs(self, *args, **kwds):
        """给定分布类型的随机变量。

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            分布的形状参数（详见实例对象的文档字符串获取更多信息）。
        loc : array_like, optional
            位置参数（默认为0）。
        scale : array_like, optional
            缩放参数（默认为1）。
        size : int or tuple of ints, optional
            随机变量的数量（默认为1）。
        random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional

            如果 `random_state` 为 None（或 `np.random`），则使用
            `numpy.random.RandomState` 单例。
            如果 `random_state` 为整数，则使用一个新的 ``RandomState`` 实例，
            并以 `random_state` 为种子。
            如果 `random_state` 已经是 ``Generator`` 或 ``RandomState``
            实例，则直接使用该实例。

        Returns
        -------
        rvs : ndarray or scalar
            给定 `size` 的随机变量。

        """
        discrete = kwds.pop('discrete', None)
        rndm = kwds.pop('random_state', None)
        args, loc, scale, size = self._parse_args_rvs(*args, **kwds)
        cond = logical_and(self._argcheck(*args), (scale >= 0))
        if not np.all(cond):
            message = ("参数的域错误。对于所有分布，`scale` 参数必须为正数，"
                       "并且许多分布对形状参数有限制。"
                       f"请参阅 `scipy.stats.{self.name}` 的文档获取详细信息。")
            raise ValueError(message)

        if np.all(scale == 0):
            return loc * ones(size, 'd')

        # 针对自定义的 random_state 需要额外的处理
        if rndm is not None:
            random_state_saved = self._random_state
            random_state = check_random_state(rndm)
        else:
            random_state = self._random_state

        vals = self._rvs(*args, size=size, random_state=random_state)

        vals = vals * scale + loc

        # 不要忘记恢复 _random_state
        if rndm is not None:
            self._random_state = random_state_saved

        # 如果是离散分布且不是 rv_sample 的实例，则将 vals 转换为整数
        if discrete and not isinstance(self, rv_sample):
            if size == ():
                vals = int(vals)
            else:
                vals = vals.astype(np.int64)

        return vals
    def entropy(self, *args, **kwds):
        """
        计算随机变量的差分熵。

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            分布的形状参数（详见实例对象的文档字符串获取更多信息）。
        loc : array_like, optional
            位置参数（默认为0）。
        scale : array_like, optional  （仅连续分布）
            尺度参数（默认为1）。

        Notes
        -----
        熵的基数为 `e`：

        >>> import numpy as np
        >>> from scipy.stats._distn_infrastructure import rv_discrete
        >>> drv = rv_discrete(values=((0, 1), (0.5, 0.5)))
        >>> np.allclose(drv.entropy(), np.log(2.0))
        True

        """
        # 解析参数
        args, loc, scale = self._parse_args(*args, **kwds)
        # 对位置和尺度参数应用asarray函数
        loc, scale = map(asarray, (loc, scale))
        args = tuple(map(asarray, args))
        # 检查参数合法性，初始化输出数组
        cond0 = self._argcheck(*args) & (scale > 0) & (loc == loc)
        output = zeros(shape(cond0), 'd')
        # 根据条件设置不合法值
        place(output, (1-cond0), self.badvalue)
        # 重新处理参数获取有效参数及其尺度
        goodargs = argsreduce(cond0, scale, *args)
        goodscale = goodargs[0]
        goodargs = goodargs[1:]
        # 对有效参数应用vecentropy方法，并加上尺度的对数
        place(output, cond0, self.vecentropy(*goodargs) + log(goodscale))
        # 返回结果数组
        return output[()]
    def moment(self, order, *args, **kwds):
        """non-central moment of distribution of specified order.

        Parameters
        ----------
        order : int, order >= 1
            Order of moment.
        arg1, arg2, arg3,... : float
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        """
        # 将传入的参数 order 赋值给局部变量 n
        n = order
        # 解析参数，获取形状参数 shapes、位置参数 loc 和尺度参数 scale
        shapes, loc, scale = self._parse_args(*args, **kwds)
        # 对所有参数进行广播，使它们具有相同的形状
        args = np.broadcast_arrays(*(*shapes, loc, scale))
        *shapes, loc, scale = args  # 重新分配解析后的参数

        # 通过 _argcheck 方法检查参数 shapes 的有效性，并确保 scale 大于 0
        i0 = np.logical_and(self._argcheck(*shapes), scale > 0)
        # 进一步细化有效性检查，确保 loc 等于 0
        i1 = np.logical_and(i0, loc == 0)
        # 进一步细化有效性检查，确保 loc 不等于 0
        i2 = np.logical_and(i0, loc != 0)

        # 调用 argsreduce 函数，对参数进行进一步处理，以确保参数正确性
        args = argsreduce(i0, *shapes, loc, scale)
        *shapes, loc, scale = args  # 重新分配处理后的参数

        # 检查 n 是否为整数，如果不是则引发 ValueError 异常
        if (floor(n) != n):
            raise ValueError("Moment must be an integer.")
        # 检查 n 是否为正数，如果不是则引发 ValueError 异常
        if (n < 0):
            raise ValueError("Moment must be positive.")
        
        mu, mu2, g1, g2 = None, None, None, None
        # 如果 n 大于 0 且小于 5，则执行下列代码
        if (n > 0) and (n < 5):
            # 如果对象具有 _stats_has_moments 属性，则设置 mdict
            if self._stats_has_moments:
                mdict = {'moments': {1: 'm', 2: 'v', 3: 'vs', 4: 'mvsk'}[n]}
            else:
                mdict = {}
            # 调用 _stats 方法获取 mu、mu2、g1、g2 的值
            mu, mu2, g1, g2 = self._stats(*shapes, **mdict)

        # 创建一个空数组 val，其形状与 loc 相同
        val = np.empty(loc.shape)  # val needs to be indexed by loc
        # 从统计量计算矩阵中获取第 n 阶矩
        val[...] = _moment_from_stats(n, mu, mu2, g1, g2, self._munp, shapes)

        # 初始化结果数组 result，其形状与 i0 相同
        result = zeros(i0.shape)
        # 将 result 中 i0 为 False 的位置设为 self.badvalue

        # 如果 i1 中有任意 True 值，则执行以下代码
        if i1.any():
            # 计算 loc 为 0 时的结果 res1
            res1 = scale[loc == 0]**n * val[loc == 0]
            place(result, i1, res1)

        # 如果 i2 中有任意 True 值，则执行以下代码
        if i2.any():
            # 将 mu、mu2、g1、g2 中非 None 的元素存入 mom 列表
            mom = [mu, mu2, g1, g2]
            arrs = [i for i in mom if i is not None]
            idx = [i for i in range(4) if mom[i] is not None]
            # 如果 idx 中有任意值，则执行以下代码
            if any(idx):
                # 调用 argsreduce 函数处理 loc 不等于 0 的情况
                arrs = argsreduce(loc != 0, *arrs)
                j = 0
                # 更新 mom 中对应索引的值
                for i in idx:
                    mom[i] = arrs[j]
                    j += 1
            mu, mu2, g1, g2 = mom
            # 调用 argsreduce 函数处理 loc 不等于 0 的情况
            args = argsreduce(loc != 0, *shapes, loc, scale, val)
            *shapes, loc, scale, val = args  # 重新分配处理后的参数

            # 创建长度为 loc.shape 的 double 类型数组 res2
            res2 = zeros(loc.shape, dtype='d')
            fac = scale / loc
            # 计算第 k 阶矩 res2
            for k in range(n):
                valk = _moment_from_stats(k, mu, mu2, g1, g2, self._munp,
                                          shapes)
                res2 += comb(n, k, exact=True)*fac**k * valk
            res2 += fac**n * val
            res2 *= loc**n
            place(result, i2, res2)

        # 返回结果数组 result
        return result[()]
    def median(self, *args, **kwds):
        """
        Median of the distribution.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            Location parameter, Default is 0.
        scale : array_like, optional
            Scale parameter, Default is 1.

        Returns
        -------
        median : float
            The median of the distribution.

        See Also
        --------
        rv_discrete.ppf
            Inverse of the CDF
        """
        # 调用实例对象中的 ppf 方法，计算分布的中位数
        return self.ppf(0.5, *args, **kwds)

    def mean(self, *args, **kwds):
        """
        Mean of the distribution.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            Location parameter (default=0)
        scale : array_like, optional
            Scale parameter (default=1)

        Returns
        -------
        mean : float
            The mean of the distribution
        """
        # 设置关键字参数 'moments' 为 'm'
        kwds['moments'] = 'm'
        # 调用实例对象中的 stats 方法，获取分布的均值
        res = self.stats(*args, **kwds)
        # 如果结果是 ndarray 类型且维度为 0，则返回数组中的单个元素作为结果
        if isinstance(res, ndarray) and res.ndim == 0:
            return res[()]
        return res

    def var(self, *args, **kwds):
        """
        Variance of the distribution.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            Location parameter (default=0)
        scale : array_like, optional
            Scale parameter (default=1)

        Returns
        -------
        var : float
            The variance of the distribution
        """
        # 设置关键字参数 'moments' 为 'v'
        kwds['moments'] = 'v'
        # 调用实例对象中的 stats 方法，获取分布的方差
        res = self.stats(*args, **kwds)
        # 如果结果是 ndarray 类型且维度为 0，则返回数组中的单个元素作为结果
        if isinstance(res, ndarray) and res.ndim == 0:
            return res[()]
        return res

    def std(self, *args, **kwds):
        """
        Standard deviation of the distribution.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            Location parameter (default=0)
        scale : array_like, optional
            Scale parameter (default=1)

        Returns
        -------
        std : float
            Standard deviation of the distribution
        """
        # 设置关键字参数 'moments' 为 'v'
        kwds['moments'] = 'v'
        # 调用实例对象中的 stats 方法，获取分布的方差，并对结果进行平方根计算
        res = sqrt(self.stats(*args, **kwds))
        return res
    def interval(self, confidence, *args, **kwds):
        """
        Confidence interval with equal areas around the median.

        Parameters
        ----------
        confidence : array_like of float
            Probability that an rv will be drawn from the returned range.
            Each value should be in the range [0, 1].
        arg1, arg2, ... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            location parameter, Default is 0.
        scale : array_like, optional
            scale parameter, Default is 1.

        Returns
        -------
        a, b : ndarray of float
            end-points of range that contain ``100 * alpha %`` of the rv's
            possible values.

        Notes
        -----
        This is implemented as ``ppf([p_tail, 1-p_tail])``, where
        ``ppf`` is the inverse cumulative distribution function and
        ``p_tail = (1-confidence)/2``. Suppose ``[c, d]`` is the support of a
        discrete distribution; then ``ppf([0, 1]) == (c-1, d)``. Therefore,
        when ``confidence=1`` and the distribution is discrete, the left end
        of the interval will be beyond the support of the distribution.
        For discrete distributions, the interval will limit the probability
        in each tail to be less than or equal to ``p_tail`` (usually
        strictly less).
        """

        # 将 confidence 转换为数组形式
        alpha = confidence

        # 检查 alpha 是否在合法范围内
        alpha = asarray(alpha)
        if np.any((alpha > 1) | (alpha < 0)):
            raise ValueError("alpha must be between 0 and 1 inclusive")

        # 计算置信区间的两个端点的位置
        q1 = (1.0-alpha)/2
        q2 = (1.0+alpha)/2

        # 使用累积分布函数的逆函数来计算置信区间的两个端点
        a = self.ppf(q1, *args, **kwds)
        b = self.ppf(q2, *args, **kwds)

        # 返回计算得到的置信区间端点
        return a, b
    def support(self, *args, **kwargs):
        """Support of the distribution.

        Parameters
        ----------
        arg1, arg2, ... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            location parameter, Default is 0.
        scale : array_like, optional
            scale parameter, Default is 1.

        Returns
        -------
        a, b : array_like
            end-points of the distribution's support.

        """
        # 解析参数，获取分布的参数和位置、缩放参数
        args, loc, scale = self._parse_args(*args, **kwargs)
        # 广播所有输入数组，包括参数、位置和缩放，以便匹配它们的形状
        arrs = np.broadcast_arrays(*args, loc, scale)
        args, loc, scale = arrs[:-2], arrs[-2], arrs[-1]
        # 检查参数并确保缩放大于零
        cond = self._argcheck(*args) & (scale > 0)
        # 获取分布支持的端点
        _a, _b = self._get_support(*args)
        if cond.all():
            # 如果所有条件都满足，计算支持端点并返回
            return _a * scale + loc, _b * scale + loc
        elif cond.ndim == 0:
            # 如果条件维度为零，返回错误值
            return self.badvalue, self.badvalue
        # 将端点提升至至少 float 类型，以填充错误值
        _a, _b = np.asarray(_a).astype('d'), np.asarray(_b).astype('d')
        out_a, out_b = _a * scale + loc, _b * scale + loc
        # 如果条件不满足，将错误值填充到端点处
        place(out_a, 1-cond, self.badvalue)
        place(out_b, 1-cond, self.badvalue)
        # 返回计算后的端点
        return out_a, out_b

    def nnlf(self, theta, x):
        """Negative loglikelihood function.
        Notes
        -----
        This is ``-sum(log pdf(x, theta), axis=0)`` where `theta` are the
        parameters (including loc and scale).
        """
        # 解包位置和缩放参数以及其他参数
        loc, scale, args = self._unpack_loc_scale(theta)
        # 检查参数是否有效，以及缩放参数是否大于零
        if not self._argcheck(*args) or scale <= 0:
            return inf
        # 根据位置和缩放参数对输入数据进行归一化处理
        x = (asarray(x)-loc) / scale
        # 计算缩放因子的负对数
        n_log_scale = len(x) * log(scale)
        # 检查输入数据是否在分布的支持范围内
        if np.any(~self._support_mask(x, *args)):
            return inf
        # 返回负对数似然函数的值
        return self._nnlf(x, *args) + n_log_scale

    def _nnlf(self, x, *args):
        # 计算负对数似然函数
        return -np.sum(self._logpxf(x, *args), axis=0)

    def _nlff_and_penalty(self, x, args, log_fitfun):
        # 负对数拟合函数及其罚项
        # 计算不符合支持条件的个数
        cond0 = ~self._support_mask(x, *args)
        n_bad = np.count_nonzero(cond0, axis=0)
        if n_bad > 0:
            # 如果有不符合支持条件的数据，进行参数约简
            x = argsreduce(~cond0, x)[0]
        # 计算拟合函数的对数值
        logff = log_fitfun(x, *args)
        # 找出有限的对数值
        finite_logff = np.isfinite(logff)
        # 统计不符合条件的数据个数
        n_bad += np.sum(~finite_logff, axis=0)
        if n_bad > 0:
            # 如果有不符合条件的数据，施加罚项
            penalty = n_bad * log(_XMAX) * 100
            return -np.sum(logff[finite_logff], axis=0) + penalty
        # 返回负对数拟合函数及其罚项的值
        return -np.sum(logff, axis=0)
    def _penalized_nnlf(self, theta, x):
        """Penalized negative loglikelihood function.
        i.e., - sum (log pdf(x, theta), axis=0) + penalty
        where theta are the parameters (including loc and scale)
        """
        # 解包参数 theta 中的 loc、scale 和 args
        loc, scale, args = self._unpack_loc_scale(theta)
        # 检查参数是否有效并且 scale 大于零
        if not self._argcheck(*args) or scale <= 0:
            return inf  # 如果参数无效或 scale 不合法，则返回无穷大
        # 标准化 x 数据到标准正态分布
        x = asarray((x-loc) / scale)
        # 计算 log(scale) 的数量级乘以数据 x 的长度
        n_log_scale = len(x) * log(scale)
        # 返回调用 _nlff_and_penalty 方法计算的结果加上 n_log_scale
        return self._nlff_and_penalty(x, args, self._logpxf) + n_log_scale

    def _penalized_nlpsf(self, theta, x):
        """Penalized negative log product spacing function.
        i.e., - sum (log (diff (cdf (x, theta))), axis=0) + penalty
        where theta are the parameters (including loc and scale)
        Follows reference [1] of scipy.stats.fit
        """
        # 解包参数 theta 中的 loc、scale 和 args
        loc, scale, args = self._unpack_loc_scale(theta)
        # 检查参数是否有效并且 scale 大于零
        if not self._argcheck(*args) or scale <= 0:
            return inf  # 如果参数无效或 scale 不合法，则返回无穷大
        # 对数据 x 进行排序，并标准化到标准正态分布
        x = (np.sort(x) - loc) / scale

        def log_psf(x, *args):
            # 对排序后的数据 x 去重并计算每个值的出现次数
            x, lj = np.unique(x, return_counts=True)  # fast for sorted x
            # 计算数据 x 的累积分布函数值，如果 x 为空，则返回空列表
            cdf_data = self._cdf(x, *args) if x.size else []
            # 如果累积分布函数最后一个值接近 1，则添加一个额外的点 1
            if not (x.size and 1 - cdf_data[-1] <= 0):
                cdf = np.concatenate(([0], cdf_data, [1]))
                lj = np.concatenate((lj, [1]))
            else:
                cdf = np.concatenate(([0], cdf_data))
            # 计算负 log 产品间距函数的值
            # 这里可以使用 logcdf 结合 logsumexp 技巧来计算差异，但在此方法的上下文中，似乎不太重要
            return lj * np.log(np.diff(cdf) / lj)

        # 返回调用 _nlff_and_penalty 方法计算的结果
        return self._nlff_and_penalty(x, args, log_psf)
class _ShapeInfo:
    # 定义形状信息类，用于存储形状的名称、是否整数、定义域及是否包含边界
    def __init__(self, name, integrality=False, domain=(-np.inf, np.inf),
                 inclusive=(True, True)):
        self.name = name  # 设置形状的名称
        self.integrality = integrality  # 设置是否为整数形状标志

        domain = list(domain)
        # 调整定义域的边界以满足不包含边界的要求
        if np.isfinite(domain[0]) and not inclusive[0]:
            domain[0] = np.nextafter(domain[0], np.inf)
        if np.isfinite(domain[1]) and not inclusive[1]:
            domain[1] = np.nextafter(domain[1], -np.inf)
        self.domain = domain  # 设置形状的定义域


def _get_fixed_fit_value(kwds, names):
    """
    给定名称列表如 ['f0', 'fa', 'fix_a']，检查 `kwds` 中最多只有一个非空值与这些名称相关联。
    返回该值，如果 `kwds` 中没有出现这些名称则返回 None。
    同时副作用是从 `kwds` 中删除所有出现的这些名称。
    """
    vals = [(name, kwds.pop(name)) for name in names if name in kwds]
    if len(vals) > 1:
        repeated = [name for name, val in vals]
        raise ValueError("fit method got multiple keyword arguments to "
                         "specify the same fixed parameter: " +
                         ', '.join(repeated))
    return vals[0][1] if vals else None


# 连续随机变量：以后可能会实现
#
# hf  --- 危险函数（PDF / SF）
# chf  --- 累积危险函数（-log(SF)）
# psf --- 概率稀疏函数（PDF 的倒数），以百分点函数单位（作为 q 的函数）。
#         也是百分点函数的导数。
    shapes : str, optional
        # 分布的形状参数。例如，对于一个需要两个整数作为形状参数的分布，可以设为 "m, n"。
        The shape of the distribution. For example ``"m, n"`` for a
        distribution that takes two integers as the two shape arguments for all
        its methods. If not provided, shape parameters will be inferred from
        the signature of the private methods, ``_pdf`` and ``_cdf`` of the
        instance.
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        # 随机数种子。可以是 None、整数、`numpy.random.Generator` 或 `numpy.random.RandomState` 实例。
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Methods
    -------
    rvs
        # 生成随机变量（Random Variates Sampling）
    pdf
        # 概率密度函数（Probability Density Function）
    logpdf
        # 对数概率密度函数（Log Probability Density Function）
    cdf
        # 累积分布函数（Cumulative Distribution Function）
    logcdf
        # 对数累积分布函数（Log Cumulative Distribution Function）
    sf
        # 生存函数（Survival Function）
    logsf
        # 对数生存函数（Log Survival Function）
    ppf
        # 百分位点函数（Percent Point Function，Inverse of CDF）
    isf
        # 逆生存函数（Inverse Survival Function）
    moment
        # 矩（Moment）
    stats
        # 统计量（Statistics）
    entropy
        # 熵（Entropy）
    expect
        # 期望（Expectation）
    median
        # 中位数（Median）
    mean
        # 均值（Mean）
    std
        # 标准差（Standard Deviation）
    var
        # 方差（Variance）
    interval
        # 置信区间（Confidence Interval）
    __call__
        # 实例的调用方法，可直接调用实例对象来计算概率密度函数的值
    fit
        # 拟合分布参数（Fit Distribution Parameters）
    fit_loc_scale
        # 拟合分布的位置和尺度参数（Fit Location and Scale Parameters）
    nnlf
        # 负对数似然函数（Negative Log-Likelihood Function）
    support
        # 支持域（Support）

    Notes
    -----
    Public methods of an instance of a distribution class (e.g., ``pdf``,
    ``cdf``) check their arguments and pass valid arguments to private,
    computational methods (``_pdf``, ``_cdf``). For ``pdf(x)``, ``x`` is valid
    if it is within the support of the distribution.
    Whether a shape parameter is valid is decided by an ``_argcheck`` method
    (which defaults to checking that its arguments are strictly positive.)

    **Subclassing**

    New random variables can be defined by subclassing the `rv_continuous` class
    and re-defining at least the ``_pdf`` or the ``_cdf`` method (normalized
    to location 0 and scale 1).

    If positive argument checking is not correct for your RV
    then you will also need to re-define the ``_argcheck`` method.

    For most of the scipy.stats distributions, the support interval doesn't
    depend on the shape parameters. ``x`` being in the support interval is
    equivalent to ``self.a <= x <= self.b``.  If either of the endpoints of
    the support do depend on the shape parameters, then
    i) the distribution must implement the ``_get_support`` method; and
    ii) those dependent endpoints must be omitted from the distribution's
    call to the ``rv_continuous`` initializer.

    Correct, but potentially slow defaults exist for the remaining
    methods but for speed and/or accuracy you can over-ride::

      _logpdf, _cdf, _logcdf, _ppf, _rvs, _isf, _sf, _logsf

    The default method ``_rvs`` relies on the inverse of the cdf, ``_ppf``,
    applied to a uniform random variate. In order to generate random variates
    efficiently, either the default ``_ppf`` needs to be overwritten (e.g.
    if the inverse cdf can expressed in an explicit form) or a sampling
    method needs to be implemented in a custom ``_rvs`` method.

    If possible, you should override ``_isf``, ``_sf`` or ``_logsf``.
    The main reason would be to improve numerical accuracy: for example,
    # survival function `_sf` 的计算为 `1 - _cdf`，如果 `_cdf(x)` 接近于一，可能会丢失精度。
    
    
    
    # 子类可以重写的方法
    ::
    
      _rvs
      _pdf
      _cdf
      _sf
      _ppf
      _isf
      _stats
      _munp
      _entropy
      _argcheck
      _get_support
    
    
    
    # 还有一些额外的（内部和私有的）通用方法，用于交叉检查和调试，但在直接调用时可能适用于所有情况。
    
    
    
    # 关于 `shapes` 的说明：子类不需要显式指定它们。在这种情况下，`shapes` 将从重写方法（如 `pdf`、`cdf` 等）的签名中自动推断出来。
    # 如果出于某种原因，您希望避免依赖内省，可以将 `shapes` 明确地指定为实例构造函数的参数。
    
    
    
    # 冻结分布
    # 通常，您必须为每次调用分布方法提供形状参数（以及可选的位置和比例参数）。
    # 或者，可以调用对象（作为函数），以固定形状、位置和比例参数，返回一个“冻结”的连续随机变量对象：
    # 
    # rv = generic(<shape(s)>, loc=0, scale=1)
    # `rv_frozen` 对象，具有相同的方法，但固定了给定的形状、位置和比例
    
    
    
    # 统计
    # 默认情况下，使用数值积分计算统计量。
    # 为了提速，您可以重新定义使用 `_stats`：
    # - 使用形状参数并返回 mu、mu2、g1、g2
    # - 如果无法计算其中之一，请返回 None
    # - 也可以使用关键字参数 `moments` 定义，它是由 "m"、"v"、"s" 和/或 "k" 组成的字符串。
    #   仅计算并按照 "m"、"v"、"s"、"k" 的顺序返回出现在字符串中的分量，缺失的值返回 None。
    # 或者，可以重写 `_munp`，它接受 `n` 和形状参数，并返回分布的第 `n` 个非中心矩。
    
    
    
    # 深拷贝 / Pickling
    # 如果分布或冻结分布被深拷贝（pickled/unpickled 等），任何潜在的随机数生成器也会被深拷贝。
    # 这意味着，如果一个分布在拷贝之前依赖于单例 `RandomState`，那么在拷贝后它将依赖于该随机状态的副本，
    # 而 `np.random.seed` 将不再控制该状态。
    
    
    
    # 示例
    # 创建一个新的高斯分布，我们可以执行以下操作：
    # 
    # >>> from scipy.stats import rv_continuous
    # >>> class gaussian_gen(rv_continuous):
    # ...     "高斯分布"
    # ...     def _pdf(self, x):
    # ...         return np.exp(-x**2 / 2.) / np.sqrt(2.0 * np.pi)
    # >>> gaussian = gaussian_gen(name='gaussian')
    # 
    # `scipy.stats` 分布是 *实例*，因此在这里我们子类化 `rv_continuous` 并创建一个实例。
    # 有了这个实例，现在我们有
    """
    a fully functional distribution with all relevant methods automagically
    generated by the framework.
    """

    def __init__(self, momtype=1, a=None, b=None, xtol=1e-14,
                 badvalue=None, name=None, longname=None,
                 shapes=None, seed=None):
        # 调用父类的初始化方法，传入种子参数
        super().__init__(seed)

        # 保存构造函数的参数到 _ctor_param 字典中
        self._ctor_param = dict(
            momtype=momtype, a=a, b=b, xtol=xtol,
            badvalue=badvalue, name=name, longname=longname,
            shapes=shapes, seed=seed)

        # 如果 badvalue 未指定，设为 NaN
        if badvalue is None:
            badvalue = nan
        # 如果 name 未指定，设为 'Distribution'
        if name is None:
            name = 'Distribution'
        # 初始化对象的属性
        self.badvalue = badvalue
        self.name = name
        self.a = a
        self.b = b
        # 如果 a 未指定，设为负无穷
        if a is None:
            self.a = -inf
        # 如果 b 未指定，设为正无穷
        if b is None:
            self.b = inf
        self.xtol = xtol
        self.moment_type = momtype
        self.shapes = shapes

        # 构建参数解析器，传入需要检查的方法列表和输入输出的 loc, scale 描述
        self._construct_argparser(meths_to_inspect=[self._pdf, self._cdf],
                                  locscale_in='loc=0, scale=1',
                                  locscale_out='loc, scale')
        # 将方法绑定到当前对象上
        self._attach_methods()

        # 如果 longname 未指定，根据 name 自动生成一个长名称
        if longname is None:
            if name[0] in ['aeiouAEIOU']:
                hstr = "An "
            else:
                hstr = "A "
            longname = hstr + name

        # 如果 Python 解释器未使用 -OO 参数运行（即优化级别小于 2），则添加文档字符串
        if sys.flags.optimize < 2:
            # 如果对象未定义文档字符串，则使用默认方式构建文档
            if self.__doc__ is None:
                self._construct_default_doc(longname=longname,
                                            docdict=docdict,
                                            discrete='continuous')
            else:
                dct = dict(distcont)
                # 使用给定的文档字典构建文档
                self._construct_doc(docdict, dct.get(self.name))

    def __getstate__(self):
        # 复制对象的字典属性
        dct = self.__dict__.copy()

        # 这些方法将在 __setstate__ 中重新创建
        # _random_state 属性由 rv_generic 处理
        attrs = ["_parse_args", "_parse_args_stats", "_parse_args_rvs",
                 "_cdfvec", "_ppfvec", "vecentropy", "generic_moment"]
        # 从字典中移除这些方法属性
        [dct.pop(attr, None) for attr in attrs]
        # 返回剩余的字典
        return dct
    # 将动态创建的方法附加到 rv_continuous 实例上
    def _attach_methods(self):
        """
        Attaches dynamically created methods to the rv_continuous instance.
        """
        # 调用 _attach_argparser_methods 方法，负责动态附加方法
        self._attach_argparser_methods()

        # 将 _ppf_single 方法向量化，并设置输入参数个数
        self._ppfvec = vectorize(self._ppf_single, otypes='d')
        self._ppfvec.nin = self.numargs + 1

        # 将 _entropy 方法向量化
        self.vecentropy = vectorize(self._entropy, otypes='d')

        # 将 _cdf_single 方法向量化，并设置输入参数个数
        self._cdfvec = vectorize(self._cdf_single, otypes='d')
        self._cdfvec.nin = self.numargs + 1

        # 根据 moment_type 属性选择不同的方法进行向量化
        if self.moment_type == 0:
            self.generic_moment = vectorize(self._mom0_sc, otypes='d')
        else:
            self.generic_moment = vectorize(self._mom1_sc, otypes='d')

        # 因为 _mom0_sc 方法的 *args 参数，vectorize 无法正确计数参数个数
        # 手动设置 generic_moment 方法的输入参数个数
        self.generic_moment.nin = self.numargs + 1

    # 返回当前 _ctor_param 的版本，可能被用户更新
    # 在 freezing 中使用，保持与 __init__ 方法签名的同步
    def _updated_ctor_param(self):
        """Return the current version of _ctor_param, possibly updated by user.

        Used by freezing.
        Keep this in sync with the signature of __init__.
        """
        dct = self._ctor_param.copy()
        dct['a'] = self.a
        dct['b'] = self.b
        dct['xtol'] = self.xtol
        dct['badvalue'] = self.badvalue
        dct['name'] = self.name
        dct['shapes'] = self.shapes
        return dct

    # 用于求解 _ppf_to_solve 函数，解方程 self.cdf(...) - q = 0
    def _ppf_to_solve(self, x, q, *args):
        return self.cdf(*(x,)+args)-q

    # 单个 q 值的分位数函数 _ppf_single
    def _ppf_single(self, q, *args):
        factor = 10.
        # 获取分布的支持范围
        left, right = self._get_support(*args)

        # 处理左侧支持范围为无穷大的情况
        if np.isinf(left):
            left = min(-factor, right)
            while self._ppf_to_solve(left, q, *args) > 0.:
                left, right = left * factor, left
            # left 被调整使得 cdf(left) <= q
            # 如果 right 发生了变化，那么 cdf(right) > q

        # 处理右侧支持范围为无穷大的情况
        if np.isinf(right):
            right = max(factor, left)
            while self._ppf_to_solve(right, q, *args) < 0.:
                left, right = right, right * factor
            # right 被调整使得 cdf(right) >= q

        # 使用 optimize.brentq 方法求解 _ppf_to_solve 函数的根
        return optimize.brentq(self._ppf_to_solve,
                               left, right, args=(q,)+args, xtol=self.xtol)

    # 使用定义计算的矩函数 _mom_integ0
    def _mom_integ0(self, x, m, *args):
        return x**m * self.pdf(x, *args)

    # 使用积分计算的矩函数 _mom0_sc
    def _mom0_sc(self, m, *args):
        _a, _b = self._get_support(*args)
        return integrate.quad(self._mom_integ0, _a, _b,
                              args=(m,)+args)[0]

    # 使用 ppf 方法计算的矩函数 _mom_integ1
    def _mom_integ1(self, q, m, *args):
        return (self.ppf(q, *args))**m

    # 使用积分计算的矩函数 _mom1_sc
    def _mom1_sc(self, m, *args):
        return integrate.quad(self._mom_integ1, 0, 1, args=(m,)+args)[0]

    # 概率密度函数的导数 _pdf
    def _pdf(self, x, *args):
        return _derivative(self._cdf, x, dx=1e-5, args=args, order=5)

    # 可以定义任何一个或多个这些方法
    # 对数概率密度函数 _logpdf
    def _logpdf(self, x, *args):
        # 计算概率密度函数的对数
        p = self._pdf(x, *args)
        with np.errstate(divide='ignore'):
            return log(p)
    # 对于连续分布，使用概率密度函数（PDF），对于离散分布，使用概率质量函数（PMF），
    # 但有时候这种区别并不重要。这使得我们能够同时为离散和连续分布使用 `_logpxf`。
    def _logpxf(self, x, *args):
        return self._logpdf(x, *args)

    # 计算累积分布函数（CDF）的单个值，根据给定参数计算积分区间。
    def _cdf_single(self, x, *args):
        _a, _b = self._get_support(*args)
        return integrate.quad(self._pdf, _a, x, args=args)[0]

    # 将 `_cdf` 重定向到 `_cdfvec`，用于向量化的累积分布函数计算。
    def _cdf(self, x, *args):
        return self._cdfvec(x, *args)

    # 在 `rv_generic` 中定义了通用的 `_argcheck`, `_logcdf`, `_sf`, `_logsf`, `_ppf`, `_isf`, `_rvs` 函数。

    # 概率密度函数（PDF）的实现，返回给定随机变量 x 处的概率密度值。
    def pdf(self, x, *args, **kwds):
        """Probability density function at x of the given RV.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at x

        """
        # 解析参数并确保它们是数组
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        # 提升 x 的数据类型至 np.float64
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc)/scale, dtype=dtyp)
        # 检查参数是否有效并且 scale 大于 0
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._support_mask(x, *args) & (scale > 0)
        cond = cond0 & cond1
        # 初始化输出数组
        output = zeros(shape(cond), dtyp)
        # 根据条件设置输出数组的值
        putmask(output, (1-cond0)+np.isnan(x), self.badvalue)
        # 如果有满足条件的 x，计算概率密度函数值并将结果存入输出数组
        if np.any(cond):
            goodargs = argsreduce(cond, *((x,)+args+(scale,)))
            scale, goodargs = goodargs[-1], goodargs[:-1]
            place(output, cond, self._pdf(*goodargs) / scale)
        # 如果输出数组是标量，返回其标量值
        if output.ndim == 0:
            return output[()]
        # 否则返回输出数组
        return output
    def logpdf(self, x, *args, **kwds):
        """
        返回给定随机变量 x 的概率密度函数的对数值。

        如果有更精确的计算方法可用，则使用该方法。

        Parameters
        ----------
        x : array_like
            分位数
        arg1, arg2, arg3,... : array_like
            分布的形状参数（详见实例对象的文档字符串）
        loc : array_like, optional
            位置参数（默认为 0）
        scale : array_like, optional
            尺度参数（默认为 1）

        Returns
        -------
        logpdf : array_like
            在 x 处评估的概率密度函数的对数值

        """
        # 解析参数，获取 loc 和 scale
        args, loc, scale = self._parse_args(*args, **kwds)
        # 将输入 x、loc、scale 转换为数组
        x, loc, scale = map(asarray, (x, loc, scale))
        # 将参数 args 转换为数组
        args = tuple(map(asarray, args))
        # 获取数据类型，选择更高精度的 np.float64
        dtyp = np.promote_types(x.dtype, np.float64)
        # 标准化 x 的数组表示，并指定数据类型
        x = np.asarray((x - loc) / scale, dtype=dtyp)
        # 检查参数的有效性，并且确保 scale 大于 0
        cond0 = self._argcheck(*args) & (scale > 0)
        # 检查 x 的支持范围是否有效，并且确保 scale 大于 0
        cond1 = self._support_mask(x, *args) & (scale > 0)
        # 组合两个条件
        cond = cond0 & cond1
        # 创建一个与 cond 形状相同的空数组，填充为负无穷
        output = empty(shape(cond), dtyp)
        output.fill(-inf)
        # 将无效的值替换为指定的 badvalue
        putmask(output, (1 - cond0) + np.isnan(x), self.badvalue)
        # 如果有任何有效的条件满足，则执行下面的操作
        if np.any(cond):
            # 缩减参数，获取有效参数
            goodargs = argsreduce(cond, *((x,) + args + (scale,)))
            scale, goodargs = goodargs[-1], goodargs[:-1]
            # 在满足条件的位置上计算 logpdf，并减去 scale 的对数
            place(output, cond, self._logpdf(*goodargs) - log(scale))
        # 如果输出的维度为 0，则返回单个元素的值
        if output.ndim == 0:
            return output[()]
        # 返回结果数组
        return output
    def cdf(self, x, *args, **kwds):
        """
        Cumulative distribution function of the given RV.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        cdf : ndarray
            Cumulative distribution function evaluated at `x`

        """
        # 解析参数，获取位置参数 loc 和尺度参数 scale
        args, loc, scale = self._parse_args(*args, **kwds)
        
        # 将输入 x, loc, scale 转换为 ndarray 类型
        x, loc, scale = map(asarray, (x, loc, scale))
        
        # 将参数 args 中的每个元素也转换为 ndarray 类型
        args = tuple(map(asarray, args))
        
        # 获取分布的支持区间
        _a, _b = self._get_support(*args)
        
        # 推断出 x 的数据类型，并将 x 转换为 dtype 为 np.float64 的 ndarray
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc)/scale, dtype=dtyp)
        
        # 检查参数条件，确保尺度 scale 大于 0
        cond0 = self._argcheck(*args) & (scale > 0)
        
        # 检查支持区间条件，确保尺度 scale 大于 0
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        
        # 检查 x 是否大于等于支持区间的上限 _b，并且满足 cond0 条件
        cond2 = (x >= np.asarray(_b)) & cond0
        
        # 组合所有条件
        cond = cond0 & cond1
        
        # 创建一个与 cond 形状相同的零数组，数据类型为 dtyp
        output = zeros(shape(cond), dtyp)
        
        # 将 output 中满足 (1-cond0) 或 x 是 NaN 的位置设为 self.badvalue
        place(output, (1-cond0)+np.isnan(x), self.badvalue)
        
        # 将 output 中满足 cond2 的位置设为 1.0
        place(output, cond2, 1.0)
        
        # 如果有任何 cond 条件满足，调用 argsreduce 处理参数，计算累积分布函数
        if np.any(cond):
            goodargs = argsreduce(cond, *((x,)+args))
            place(output, cond, self._cdf(*goodargs))
        
        # 如果 output 的维度为 0，返回 output 的单个元素
        if output.ndim == 0:
            return output[()]
        
        # 否则返回 output 数组
        return output
    # 定义一个方法用于计算给定随机变量在 x 处的累积分布函数的对数值

    def logcdf(self, x, *args, **kwds):
        """Log of the cumulative distribution function at x of the given RV.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        logcdf : array_like
            Log of the cumulative distribution function evaluated at x

        """
        # 解析参数，获取位置参数 loc 和尺度参数 scale
        args, loc, scale = self._parse_args(*args, **kwds)
        # 将 x, loc, scale 转换为数组形式
        x, loc, scale = map(asarray, (x, loc, scale))
        # 将参数 args 转换为数组形式
        args = tuple(map(asarray, args))
        # 获取随机变量的支持范围
        _a, _b = self._get_support(*args)
        # 确定输出数据类型
        dtyp = np.promote_types(x.dtype, np.float64)
        # 标准化 x
        x = np.asarray((x - loc)/scale, dtype=dtyp)
        # 检查参数的有效性
        cond0 = self._argcheck(*args) & (scale > 0)
        # 检查是否超出支持范围
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        cond2 = (x >= _b) & cond0
        cond = cond0 & cond1
        # 创建输出数组并用 -inf 填充
        output = empty(shape(cond), dtyp)
        output.fill(-inf)
        # 将无效值替换为 self.badvalue
        place(output, (1-cond0)*(cond1 == cond1)+np.isnan(x), self.badvalue)
        # 将超出上界的部分设为 0.0
        place(output, cond2, 0.0)
        # 如果存在有效条件，计算对数累积分布函数值
        if np.any(cond):  # 仅在至少存在一个有效条件时调用
            goodargs = argsreduce(cond, *((x,)+args))
            place(output, cond, self._logcdf(*goodargs))
        # 如果输出的维度为 0，则返回其单个元素
        if output.ndim == 0:
            return output[()]
        return output
    # 定义生存函数（1 - `cdf`），用于给定随机变量在 x 处的求值

    def sf(self, x, *args, **kwds):
        """Survival function (1 - `cdf`) at x of the given RV.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        sf : array_like
            Survival function evaluated at x

        """
        # 解析参数，获取 loc 和 scale
        args, loc, scale = self._parse_args(*args, **kwds)
        # 将 x, loc, scale 转换为数组
        x, loc, scale = map(asarray, (x, loc, scale))
        # 将 args 中的每个参数都转换为数组
        args = tuple(map(asarray, args))
        # 获取分布的支持范围 _a, _b
        _a, _b = self._get_support(*args)
        # 选择合适的数据类型，确保 x 的 dtype 是 float64
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc) / scale, dtype=dtyp)
        # 检查参数的有效性，并且确保 scale 大于 0
        cond0 = self._argcheck(*args) & (scale > 0)
        # 检查 x 是否在分布的开放支持范围内，并且确保 scale 大于 0
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        # 检查 x 是否小于等于 _a，并且确保 scale 大于 0
        cond2 = cond0 & (x <= _a)
        # 组合所有条件
        cond = cond0 & cond1
        # 创建一个和 cond 形状相同的零数组，使用 dtyp 数据类型
        output = zeros(shape(cond), dtyp)
        # 将不符合 cond0 的位置置为 badvalue，如果 x 是 NaN，则置为 badvalue
        place(output, (1-cond0)+np.isnan(x), self.badvalue)
        # 将符合 cond2 的位置置为 1.0
        place(output, cond2, 1.0)
        # 如果存在符合 cond 的位置，进行求解生存函数值
        if np.any(cond):
            # 通过 argsreduce 函数简化参数，并调用 _sf 函数求解生存函数值
            goodargs = argsreduce(cond, *((x,)+args))
            place(output, cond, self._sf(*goodargs))
        # 如果 output 的维度是 0，则返回一个标量
        if output.ndim == 0:
            return output[()]
        # 否则返回 output 数组
        return output
    def logsf(self, x, *args, **kwds):
        """
        对给定随机变量的存活函数的对数值进行计算。

        返回在`x`处评估的存活函数的对数值，存活函数定义为`(1 - cdf)`。

        Parameters
        ----------
        x : array_like
            分位数
        arg1, arg2, arg3,... : array_like
            分布的形状参数（更多信息请参阅实例对象的文档字符串）
        loc : array_like, optional
            位置参数（默认为0）
        scale : array_like, optional
            尺度参数（默认为1）

        Returns
        -------
        logsf : ndarray
            在`x`处评估的存活函数的对数值。

        """
        # 解析参数
        args, loc, scale = self._parse_args(*args, **kwds)
        # 将输入参数转换为数组
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        # 获取支持区间
        _a, _b = self._get_support(*args)
        # 确定输出数组的数据类型
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc)/scale, dtype=dtyp)
        # 条件检查
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        cond2 = cond0 & (x <= _a)
        cond = cond0 & cond1
        # 创建输出数组并初始化为负无穷
        output = empty(shape(cond), dtyp)
        output.fill(-inf)
        # 填充特定条件下的值
        place(output, (1-cond0)+np.isnan(x), self.badvalue)
        place(output, cond2, 0.0)
        # 如果有满足条件的情况，则进一步计算存活函数的对数值
        if np.any(cond):
            goodargs = argsreduce(cond, *((x,)+args))
            place(output, cond, self._logsf(*goodargs))
        # 如果输出数组维度为0，则返回其标量值
        if output.ndim == 0:
            return output[()]
        return output
    def ppf(self, q, *args, **kwds):
        """
        Percent point function (inverse of `cdf`) at q of the given RV.

        Parameters
        ----------
        q : array_like
            lower tail probability
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        x : array_like
            quantile corresponding to the lower tail probability q.
        """

        # 解析参数，获取位置参数 loc 和尺度参数 scale
        args, loc, scale = self._parse_args(*args, **kwds)
        
        # 将 q, loc, scale 转换为数组形式
        q, loc, scale = map(asarray, (q, loc, scale))
        
        # 将其他参数也转换为数组形式
        args = tuple(map(asarray, args))
        
        # 获取分布的支持范围 _a, _b
        _a, _b = self._get_support(*args)
        
        # 检查参数条件
        cond0 = self._argcheck(*args) & (scale > 0) & (loc == loc)
        
        # 检查 q 的范围条件
        cond1 = (0 < q) & (q < 1)
        
        # 特殊情况处理：当 q 为 0 或 1 时的条件
        cond2 = cond0 & (q == 0)
        cond3 = cond0 & (q == 1)
        
        # 组合所有条件
        cond = cond0 & cond1
        
        # 初始化输出数组，使用 self.badvalue 填充
        output = np.full(shape(cond), fill_value=self.badvalue)
        
        # 计算下界和上界
        lower_bound = _a * scale + loc
        upper_bound = _b * scale + loc
        
        # 根据条件设置 output 的值
        place(output, cond2, argsreduce(cond2, lower_bound)[0])
        place(output, cond3, argsreduce(cond3, upper_bound)[0])
        
        # 如果有符合条件的值，调用 _ppf 方法计算结果，并根据尺度和位置参数进行调整
        if np.any(cond):
            goodargs = argsreduce(cond, *((q,) + args + (scale, loc)))
            scale, loc, goodargs = goodargs[-2], goodargs[-1], goodargs[:-2]
            place(output, cond, self._ppf(*goodargs) * scale + loc)
        
        # 如果输出数组的维度为 0，返回其唯一元素
        if output.ndim == 0:
            return output[()]
        
        # 否则返回输出数组
        return output
    # 定义一个方法 `isf`，用于计算给定随机变量的上尾概率对应的分位数
    def isf(self, q, *args, **kwds):
        """Inverse survival function (inverse of `sf`) at q of the given RV.

        Parameters
        ----------
        q : array_like
            upper tail probability
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        x : ndarray or scalar
            Quantile corresponding to the upper tail probability q.

        """
        # 解析传入的参数 `args` 和关键字参数 `kwds`
        args, loc, scale = self._parse_args(*args, **kwds)
        # 将 `q`、`loc`、`scale` 转换为数组
        q, loc, scale = map(asarray, (q, loc, scale))
        # 将 `args` 中的每个参数转换为数组
        args = tuple(map(asarray, args))
        # 获取分布的支持区间 `_a` 和 `_b`
        _a, _b = self._get_support(*args)
        # 检查参数的有效性
        cond0 = self._argcheck(*args) & (scale > 0) & (loc == loc)
        # 检查概率 `q` 是否在 (0, 1) 区间内
        cond1 = (0 < q) & (q < 1)
        # 特殊情况：当 `q` 等于 1 时的条件
        cond2 = cond0 & (q == 1)
        # 特殊情况：当 `q` 等于 0 时的条件
        cond3 = cond0 & (q == 0)
        # 组合常规条件 `cond0` 和 `cond1`
        cond = cond0 & cond1
        # 初始化输出数组，用 `self.badvalue` 填充
        output = np.full(shape(cond), fill_value=self.badvalue)

        # 计算下限和上限
        lower_bound = _a * scale + loc
        upper_bound = _b * scale + loc
        # 将下限和上限放置到 `output` 中对应的条件位置
        place(output, cond2, argsreduce(cond2, lower_bound)[0])
        place(output, cond3, argsreduce(cond3, upper_bound)[0])

        # 如果存在满足条件的值，计算相应的分位数
        if np.any(cond):
            # 从满足条件的参数中获取有效参数
            goodargs = argsreduce(cond, *((q,)+args+(scale, loc)))
            # 分别获取 `scale`、`loc` 和除了最后两个元素外的其余参数
            scale, loc, goodargs = goodargs[-2], goodargs[-1], goodargs[:-2]
            # 将计算得到的分位数乘以 `scale` 并加上 `loc`，放置到 `output` 中对应的条件位置
            place(output, cond, self._isf(*goodargs) * scale + loc)
        # 如果 `output` 的维度为 0，则返回其标量值
        if output.ndim == 0:
            return output[()]
        # 否则返回 `output` 数组
        return output

    # 定义一个方法 `_unpack_loc_scale`，用于从给定的参数 `theta` 中解包 `loc`、`scale` 和其余参数
    def _unpack_loc_scale(self, theta):
        try:
            # 尝试从 `theta` 中提取 `loc` 和 `scale`，以及其余参数 `args`
            loc = theta[-2]
            scale = theta[-1]
            args = tuple(theta[:-2])
        except IndexError as e:
            # 如果出现索引错误，抛出 `ValueError` 异常
            raise ValueError("Not enough input arguments.") from e
        # 返回解包后的 `loc`、`scale` 和 `args`
        return loc, scale, args
    def _nnlf_and_penalty(self, x, args):
        """
        计算标准化数据（已经通过 loc 平移并通过 scale 缩放）的惩罚负对数似然，
        用于参数 `args` 中的形状参数。

        `x` 可以是一个1维的numpy数组或者一个CensoredData实例。
        """
        if isinstance(x, CensoredData):
            # 过滤不在支持范围内的数据。
            xs = x._supported(*self._get_support(*args))
            n_bad = len(x) - len(xs)  # 计算不在支持范围内的数据数量。
            i1, i2 = xs._interval.T  # 获取区间数据的左右边界。
            terms = [
                # 非被截断数据的对数概率密度函数。
                self._logpdf(xs._uncensored, *args),
                # 左截断数据的对数累积分布函数。
                self._logcdf(xs._left, *args),
                # 右截断数据的对数生存函数。
                self._logsf(xs._right, *args),
                # 区间截断数据的概率对数。
                np.log(self._delta_cdf(i1, i2, *args)),
            ]
        else:
            cond0 = ~self._support_mask(x, *args)  # 确定数据是否在支持范围内的布尔掩码。
            n_bad = np.count_nonzero(cond0)  # 计算不在支持范围内的数据数量。
            if n_bad > 0:
                x = argsreduce(~cond0, x)[0]  # 从数据中移除不在支持范围内的部分。
            terms = [self._logpdf(x, *args)]  # 计算数据的对数概率密度函数。

        totals, bad_counts = zip(*[_sum_finite(term) for term in terms])  # 计算总和和不良数据的数量。
        total = sum(totals)  # 计算总和。
        n_bad += sum(bad_counts)  # 添加不良数据的数量。

        return -total + n_bad * _LOGXMAX * 100  # 返回惩罚负对数似然。

    def _penalized_nnlf(self, theta, x):
        """带惩罚项的负对数似然函数。

        即，- sum (log pdf(x, theta), axis=0) + penalty
        其中 theta 是参数（包括 loc 和 scale）。
        """
        loc, scale, args = self._unpack_loc_scale(theta)  # 解包参数中的 loc 和 scale。
        if not self._argcheck(*args) or scale <= 0:  # 检查参数是否有效。
            return inf  # 如果参数无效，返回无穷大。
        if isinstance(x, CensoredData):
            x = (x - loc) / scale  # 对数据进行标准化处理。
            n_log_scale = (len(x) - x.num_censored()) * log(scale)  # 计算对数缩放的数量。
        else:
            x = (x - loc) / scale  # 对数据进行标准化处理。
            n_log_scale = len(x) * log(scale)  # 计算对数缩放的数量。

        return self._nnlf_and_penalty(x, args) + n_log_scale  # 返回带惩罚项的负对数似然。

    def _fitstart(self, data, args=None):
        """拟合的起始点（包括形状参数 + loc + scale）。"""
        if args is None:
            args = (1.0,)*self.numargs  # 如果没有给定参数，则使用默认参数。
        loc, scale = self._fit_loc_scale_support(data, *args)  # 根据数据和参数计算 loc 和 scale。
        return args + (loc, scale)  # 返回参数和 loc、scale。
    def _reduce_func(self, args, kwds, data=None):
        """
        Return the (possibly reduced) function to optimize in order to find MLE
        estimates for the .fit method.
        """
        # 将固定形状参数转换为标准数值形式：例如对于 stats.beta，shapes='a, b'。
        # 调用者可以通过给定 `f0`, `fa` 或 'fix_a' 来固定 `a` 的值。以下代码将
        # 后两者转换为第一种（数值）形式。
        shapes = []
        if self.shapes:
            # 将形状字符串按逗号分隔并转换为列表
            shapes = self.shapes.replace(',', ' ').split()
            for j, s in enumerate(shapes):
                key = 'f' + str(j)
                names = [key, 'f' + s, 'fix_' + s]
                # 从关键字参数中获取固定值
                val = _get_fixed_fit_value(kwds, names)
                if val is not None:
                    kwds[key] = val

        # 将位置参数转换为列表形式，并计算参数个数
        args = list(args)
        Nargs = len(args)
        fixedn = []
        # 定义固定参数名称列表
        names = ['f%d' % n for n in range(Nargs - 2)] + ['floc', 'fscale']
        x0 = []
        # 处理每个名称，并将固定的参数值从关键字参数中移除
        for n, key in enumerate(names):
            if key in kwds:
                fixedn.append(n)
                args[n] = kwds.pop(key)
            else:
                x0.append(args[n])

        # 定义方法集合和默认方法，根据关键字参数设置确定最优化方法
        methods = {"mle", "mm"}
        method = kwds.pop('method', "mle").lower()
        if method == "mm":
            # 计算指数并计算数据的矩
            n_params = len(shapes) + 2 - len(fixedn)
            exponents = (np.arange(1, n_params+1))[:, np.newaxis]
            data_moments = np.sum(data[None, :]**exponents/len(data), axis=1)

            def objective(theta, x):
                return self._moment_error(theta, x, data_moments)

        elif method == "mle":
            # 使用带罚项的负对数似然函数作为优化目标
            objective = self._penalized_nnlf
        else:
            # 抛出错误，若方法不可用
            raise ValueError(f"Method '{method}' not available; "
                             f"must be one of {methods}")

        # 若无固定参数，则直接使用目标函数
        if len(fixedn) == 0:
            func = objective
            restore = None
        else:
            # 若有固定参数，定义恢复函数和优化函数
            if len(fixedn) == Nargs:
                raise ValueError(
                    "All parameters fixed. There is nothing to optimize.")

            def restore(args, theta):
                # 替换非固定位置的参数值为 theta
                # 允许非固定的值变化，但仍调用所有参数调用 self.nnlf
                i = 0
                for n in range(Nargs):
                    if n not in fixedn:
                        args[n] = theta[i]
                        i += 1
                return args

            def func(theta, x):
                newtheta = restore(args[:], theta)
                return objective(newtheta, x)

        # 返回初始猜测值 x0、优化函数 func、恢复函数 restore 和参数列表 args
        return x0, func, restore, args
    # 计算给定参数 theta 下的矩误差，用于拟合数据的矩
    def _moment_error(self, theta, x, data_moments):
        # 解包参数 theta，获取 loc（位置参数）、scale（尺度参数）、args（其他参数）
        loc, scale, args = self._unpack_loc_scale(theta)
        # 检查参数的合法性并确保尺度大于零，否则返回无穷大
        if not self._argcheck(*args) or scale <= 0:
            return inf

        # 计算数据矩的对应分布矩，存储在数组 dist_moments 中
        dist_moments = np.array([self.moment(i+1, *args, loc=loc, scale=scale)
                                 for i in range(len(data_moments))])
        # 如果计算得到的分布矩中有 NaN（非数值），则抛出 ValueError
        if np.any(np.isnan(dist_moments)):
            raise ValueError("Method of moments encountered a non-finite "
                             "distribution moment and cannot continue. "
                             "Consider trying method='MLE'.")

        # 计算矩误差的平方和，用来衡量数据矩与分布矩之间的差异
        return (((data_moments - dist_moments) /
                 np.maximum(np.abs(data_moments), 1e-8))**2).sum()
    def _fit_loc_scale_support(self, data, *args):
        """
        Estimate loc and scale parameters from data accounting for support.

        Parameters
        ----------
        data : array_like
            Data to fit.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).

        Returns
        -------
        Lhat : float
            Estimated location parameter for the data.
        Shat : float
            Estimated scale parameter for the data.
        """
        if isinstance(data, CensoredData):
            # If the data is of type CensoredData, convert it to uncensored form
            # using the _uncensor method.
            data = data._uncensor()
        else:
            # Convert data to a NumPy array if it is not already.
            data = np.asarray(data)

        # Estimate location and scale using the fit_loc_scale method with given args.
        loc_hat, scale_hat = self.fit_loc_scale(data, *args)

        # Check and compute the support bounds using _argcheck and _get_support methods.
        self._argcheck(*args)
        _a, _b = self._get_support(*args)
        a, b = _a, _b
        support_width = b - a

        # If the support width is non-positive, return the moment-based estimates.
        if support_width <= 0:
            return loc_hat, scale_hat

        # Compute the proposed support bounds using the estimated loc and scale.
        a_hat = loc_hat + a * scale_hat
        b_hat = loc_hat + b * scale_hat

        # Use moment-based estimates if they fit within the data bounds.
        data_a = np.min(data)
        data_b = np.max(data)
        if a_hat < data_a and data_b < b_hat:
            return loc_hat, scale_hat

        # Otherwise, find new estimates compatible with the data bounds.
        data_width = data_b - data_a
        rel_margin = 0.1
        margin = data_width * rel_margin

        # For a finite interval, adjust loc_hat and scale_hat accordingly.
        if support_width < np.inf:
            loc_hat = (data_a - a) - margin
            scale_hat = (data_width + 2 * margin) / support_width
            return loc_hat, scale_hat

        # For a one-sided interval, adjust loc_hat with a margin.
        if a > -np.inf:
            return (data_a - a) - margin, 1
        elif b < np.inf:
            return (data_b - b) + margin, 1
        else:
            # Raise a RuntimeError if no suitable estimates are found.
            raise RuntimeError
    # 从数据中估计 loc 和 scale 参数，使用数据的第一和第二矩进行估计

    def fit_loc_scale(self, data, *args):
        """
        Estimate loc and scale parameters from data using 1st and 2nd moments.

        Parameters
        ----------
        data : array_like
            Data to fit.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).

        Returns
        -------
        Lhat : float
            Estimated location parameter for the data.
        Shat : float
            Estimated scale parameter for the data.

        """
        # 计算分布的期望和第二中心距（平方的期望）
        mu, mu2 = self.stats(*args, **{'moments': 'mv'})
        # 将数据转换为数组
        tmp = asarray(data)
        # 计算数据的均值估计
        muhat = tmp.mean()
        # 计算数据的方差估计
        mu2hat = tmp.var()
        # 通过数据的第一和第二中心距估计数据的标准差参数 Shat
        Shat = sqrt(mu2hat / mu2)
        # 通过期望和估计的标准差 Shat 估计数据的位置参数 Lhat
        with np.errstate(invalid='ignore'):
            Lhat = muhat - Shat*mu
        # 处理 Lhat 不是有限值的情况
        if not np.isfinite(Lhat):
            Lhat = 0
        # 处理 Shat 不是有限值或小于等于零的情况
        if not (np.isfinite(Shat) and (0 < Shat)):
            Shat = 1
        # 返回估计的位置参数和标准差参数
        return Lhat, Shat

    # 计算分布的熵
    def _entropy(self, *args):
        def integ(x):
            # 计算概率密度函数在给定点 x 处的值
            val = self._pdf(x, *args)
            # 计算熵的贡献，使用 entr 函数
            return entr(val)

        # 使用积分计算熵，忽略上限可能为无穷大时的警告
        _a, _b = self._get_support(*args)
        with np.errstate(over='ignore'):
            h = integrate.quad(integ, _a, _b)[0]

        # 如果计算出的熵不是 NaN，则返回该值
        if not np.isnan(h):
            return h
        else:
            # 如果积分计算出问题，尝试使用不同的积分上下限再次计算
            low, upp = self.ppf([1e-10, 1. - 1e-10], *args)
            if np.isinf(_b):
                upper = upp
            else:
                upper = _b
            if np.isinf(_a):
                lower = low
            else:
                lower = _a
            # 再次尝试积分计算熵
            return integrate.quad(integ, lower, upper)[0]

    # 返回分布的参数信息，包括形状信息、位置信息和标准差信息
    def _param_info(self):
        shape_info = self._shape_info()
        # 创建位置信息对象，表示位置参数的属性
        loc_info = _ShapeInfo("loc", False, (-np.inf, np.inf), (False, False))
        # 创建标准差信息对象，表示标准差参数的属性
        scale_info = _ShapeInfo("scale", False, (0, np.inf), (False, False))
        # 返回形状信息、位置信息和标准差信息的列表
        param_info = shape_info + [loc_info, scale_info]
        return param_info

    # 目前，_delta_cdf 是一个私有方法，暂无注释说明其作用。
    def _delta_cdf(self, x1, x2, *args, loc=0, scale=1):
        """
        Compute CDF(x2) - CDF(x1).

        Where x1 is greater than the median, compute SF(x1) - SF(x2),
        otherwise compute CDF(x2) - CDF(x1).

        This function is only useful if `dist.sf(x, ...)` has an implementation
        that is numerically more accurate than `1 - dist.cdf(x, ...)`.
        """
        # 计算 x1 处的累积分布函数（CDF）
        cdf1 = self.cdf(x1, *args, loc=loc, scale=scale)
        
        # 如果 cdf1 > 0.5，优化计算 SF(x1) - SF(x2)，否则计算 CDF(x2) - CDF(x1)
        result = np.where(cdf1 > 0.5,
                          (self.sf(x1, *args, loc=loc, scale=scale)
                           - self.sf(x2, *args, loc=loc, scale=scale)),
                          self.cdf(x2, *args, loc=loc, scale=scale) - cdf1)
        
        # 如果结果是标量，转换为标量值
        if result.ndim == 0:
            result = result[()]
        
        return result
# Helpers for the discrete distributions
def _drv2_moment(self, n, *args):
    """Non-central moment of discrete distribution."""
    # 定义内部函数，计算离散分布的非中心矩
    def fun(x):
        return np.power(x, n) * self._pmf(x, *args)

    # 获取分布的支持区间
    _a, _b = self._get_support(*args)
    # 使用期望函数计算非中心矩
    return _expect(fun, _a, _b, self.ppf(0.5, *args), self.inc)


def _drv2_ppfsingle(self, q, *args):  # Use basic bisection algorithm
    # 获取分布的支持区间
    _a, _b = self._get_support(*args)
    b = _b
    a = _a
    # 确保结束点大于 q
    if isinf(b):
        b = int(max(100*q, 10))
        while 1:
            if b >= _b:
                qb = 1.0
                break
            qb = self._cdf(b, *args)
            if (qb < q):
                b += 10
            else:
                break
    else:
        qb = 1.0
    # 确保起始点小于 q
    if isinf(a):
        a = int(min(-100*q, -10))
        while 1:
            if a <= _a:
                qb = 0.0
                break
            qa = self._cdf(a, *args)
            if (qa > q):
                a -= 10
            else:
                break
    else:
        qa = self._cdf(a, *args)

    while 1:
        # 如果起始点的累积分布等于 q，返回起始点
        if (qa == q):
            return a
        # 如果结束点的累积分布等于 q，返回结束点
        if (qb == q):
            return b
        # 如果结束点小于等于起始点加一，则根据累积分布决定返回哪个点
        if b <= a+1:
            if qa > q:
                return a
            else:
                return b
        # 使用二分法更新 c 点，并计算其累积分布
        c = int((a+b)/2.0)
        qc = self._cdf(c, *args)
        if (qc < q):
            if a != c:
                a = c
            else:
                raise RuntimeError('updating stopped, endless loop')
            qa = qc
        elif (qc > q):
            if b != c:
                b = c
            else:
                raise RuntimeError('updating stopped, endless loop')
            qb = qc
        else:
            return c


# Must over-ride one of _pmf or _cdf or pass in
#  x_k, p(x_k) lists in initialization


class rv_discrete(rv_generic):
    """A generic discrete random variable class meant for subclassing.

    `rv_discrete` is a base class to construct specific distribution classes
    and instances for discrete random variables. It can also be used
    to construct an arbitrary distribution defined by a list of support
    points and corresponding probabilities.

    Parameters
    ----------
    a : float, optional
        Lower bound of the support of the distribution, default: 0
    b : float, optional
        Upper bound of the support of the distribution, default: plus infinity
    moment_tol : float, optional
        The tolerance for the generic calculation of moments.
    values : tuple of two array_like, optional
        ``(xk, pk)`` where ``xk`` are integers and ``pk`` are the non-zero
        probabilities between 0 and 1 with ``sum(pk) = 1``. ``xk``
        and ``pk`` must have the same shape, and ``xk`` must be unique.
    inc : integer, optional
        Increment for the support of the distribution.
        Default is 1. (other values have not been tested)

    """
    # badvalue：float，可选项
    # 表示结果数组中指示某些参数限制被违反的值，默认为 np.nan。
    name : str, optional
    # 实例的名称。此字符串用于构造分布的默认示例。
    longname : str, optional
    # 此字符串用作没有自己文档字符串的子类返回的文档字符串的第一行的一部分。
    # 注意：`longname` 是为了向后兼容而存在，不要在新的子类中使用。
    shapes : str, optional
    # 分布的形状。例如，对于需要两个整数作为其所有方法的两个形状参数的分布，可以指定 "m, n"。
    # 如果未提供，则将从实例的私有方法 `_pmf` 和 `_cdf` 的签名推断形状参数。
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
    # 如果 `seed` 是 None (或 `np.random`)，则使用 `numpy.random.RandomState` 单例。
    # 如果 `seed` 是整数，则使用一个新的 ``RandomState`` 实例，并使用 `seed` 作为种子。
    # 如果 `seed` 已经是 ``Generator`` 或 ``RandomState`` 实例，则直接使用该实例。

    Methods
    -------
    rvs
    # 随机变量抽样方法
    pmf
    # 概率质量函数
    logpmf
    # 对数概率质量函数
    cdf
    # 累积分布函数
    logcdf
    # 对数累积分布函数
    sf
    # 生存函数
    logsf
    # 对数生存函数
    ppf
    # 百分位点函数（分位数函数）
    isf
    # 逆生存函数（逆生存概率函数）
    moment
    # 矩
    stats
    # 统计量
    entropy
    # 熵
    expect
    # 期望值
    median
    # 中位数
    mean
    # 均值
    std
    # 标准差
    var
    # 方差
    interval
    # 置信区间
    __call__
    # 实例的调用方法
    support
    # 分布的支持集合

    Notes
    -----
    # 此类类似于 `rv_continuous`。参数的有效性由 ``_argcheck`` 方法决定（默认检查参数是否严格为正）。
    # 主要区别如下：

    - 支持分布的是一组整数。
    - 该类定义了*概率质量函数* `pmf`（以及对应的私有 `_pmf`）而不是概率密度函数 `pdf`（以及对应的私有 `_pdf`）。
    - 没有 `scale` 参数。
    - 方法的默认实现（例如 `_cdf`）不适用于下限无界的支持（即 `a=-np.inf`），因此必须进行重写。

    # 要创建一个新的离散分布，可以按以下步骤操作：

    >>> from scipy.stats import rv_discrete
    >>> class poisson_gen(rv_discrete):
    ...     "Poisson distribution"
    ...     def _pmf(self, k, mu):
    ...         return exp(-mu) * mu**k / factorial(k)

    # 然后创建一个实例：

    >>> poisson = poisson_gen(name="poisson")

    # 注意，在上面我们以标准形式定义了泊松分布。
    # 通过在实例的方法中提供 `loc` 参数，可以对分布进行移动。
    # 例如，``poisson.pmf(x, mu, loc)`` 将工作委托给 ``poisson._pmf(x-loc, mu)``。
    """
    Create a new class for a discrete random variable (RV) or subclass based on parameters.

    Parameters:
    - `a`: Lower bound of the distribution (default: 0)
    - `b`: Upper bound of the distribution (default: inf)
    - `name`: Name of the distribution (optional)
    - `badvalue`: Value to indicate invalid results (default: None)
    - `moment_tol`: Tolerance for moments computation (default: 1e-8)
    - `values`: Values of the discrete distribution (default: None)
    - `inc`: Increment between values (default: 1)
    - `longname`: Long name for the distribution (optional)
    - `shapes`: Shape parameters for the distribution (default: None)
    - `seed`: Seed for random number generation (default: None)

    Returns:
    - An instance of the appropriate subclass (`rv_sample` if `values` is not None, else `cls`).

    Notes:
    - If `values` is provided, the constructor delegates to a subclass `rv_sample`.
    - Initializes the superclass `rv_sample` or `cls` depending on `values`.
    """

    def __init__(self, a=0, b=inf, name=None, badvalue=None,
                 moment_tol=1e-8, values=None, inc=1, longname=None,
                 shapes=None, seed=None):
        """
        Initialize the discrete random variable instance.

        Parameters:
        - `a`: Lower bound of the distribution (default: 0)
        - `b`: Upper bound of the distribution (default: inf)
        - `name`: Name of the distribution (optional)
        - `badvalue`: Value to indicate invalid results (default: None)
        - `moment_tol`: Tolerance for moments computation (default: 1e-8)
        - `values`: Values of the discrete distribution (default: None)
        - `inc`: Increment between values (default: 1)
        - `longname`: Long name for the distribution (optional)
        - `shapes`: Shape parameters for the distribution (default: None)
        - `seed`: Seed for random number generation (default: None)

        Notes:
        - Initializes the superclass with the given `seed`.
        - Sets `_ctor_param` with a dictionary of constructor parameters.
        - Sets `badvalue`, `a`, `b`, `moment_tol`, `inc`, and `shapes` attributes.
        - Raises a `ValueError` if `values` is not None, as values initialization is not supported in this constructor.
        - Configures argument parsing for methods `_pmf` and `_cdf`.
        - Attaches methods to the instance.
        - Constructs docstrings based on `name` and `longname`.
        """
        super().__init__(seed)

        # Store constructor parameters for potential future use
        self._ctor_param = dict(
            a=a, b=b, name=name, badvalue=badvalue,
            moment_tol=moment_tol, values=values, inc=inc,
            longname=longname, shapes=shapes, seed=seed)

        # Set default value for `badvalue` if not provided
        if badvalue is None:
            badvalue = nan
        self.badvalue = badvalue
        self.a = a
        self.b = b
        self.moment_tol = moment_tol
        self.inc = inc
        self.shapes = shapes

        # Check and raise an error if `values` is provided, as it's not supported in this context
        if values is not None:
            raise ValueError("rv_discrete.__init__(..., values != None, ...)")

        # Configure argument parsing for methods `_pmf` and `_cdf`
        self._construct_argparser(meths_to_inspect=[self._pmf, self._cdf],
                                  locscale_in='loc=0',
                                  locscale_out='loc, 1')  # scale=1 for discrete RVs

        # Attach methods to the instance
        self._attach_methods()

        # Construct docstrings based on `name` and `longname`
        self._construct_docstrings(name, longname)
    # 返回对象的状态字典，排除特定的方法和属性
    def __getstate__(self):
        dct = self.__dict__.copy()
        # 在 __setstate__ 中会重新创建这些方法
        attrs = ["_parse_args", "_parse_args_stats", "_parse_args_rvs",
                 "_cdfvec", "_ppfvec", "generic_moment"]
        # 从字典中移除指定的属性
        [dct.pop(attr, None) for attr in attrs]
        return dct

    # 为 rv_discrete 实例动态附加创建的方法
    def _attach_methods(self):
        """Attaches dynamically created methods to the rv_discrete instance."""
        # 使用 vectorize 方法将 _cdf_single 方法向量化，并赋值给 self._cdfvec
        self._cdfvec = vectorize(self._cdf_single, otypes='d')
        # 使用 vectorize 方法将 _entropy 方法向量化，并赋值给 self.vecentropy
        self.vecentropy = vectorize(self._entropy)

        # 调用 _attach_argparser_methods 方法
        # _attach_methods 负责调用 _attach_argparser_methods
        self._attach_argparser_methods()

        # 在确定 numargs 后，需要进行 nin 校正
        # 为了向量化通用矩方法，需要校正 nin
        _vec_generic_moment = vectorize(_drv2_moment, otypes='d')
        _vec_generic_moment.nin = self.numargs + 2
        self.generic_moment = types.MethodType(_vec_generic_moment, self)

        # 校正 ppf 向量化的 nin
        _vppf = vectorize(_drv2_ppfsingle, otypes='d')
        _vppf.nin = self.numargs + 2
        self._ppfvec = types.MethodType(_vppf, self)

        # 现在 self.numargs 已定义，我们可以调整 _cdfvec 的 nin
        self._cdfvec.nin = self.numargs + 1

    # 构建文档字符串
    def _construct_docstrings(self, name, longname):
        if name is None:
            name = 'Distribution'
        self.name = name

        # 为子类实例生成文档字符串
        if longname is None:
            if name[0] in ['aeiouAEIOU']:
                hstr = "An "
            else:
                hstr = "A "
            longname = hstr + name

        if sys.flags.optimize < 2:
            # 如果解释器运行时没有使用 -OO，则添加文档字符串
            if self.__doc__ is None:
                self._construct_default_doc(longname=longname,
                                            docdict=docdict_discrete,
                                            discrete='discrete')
            else:
                dct = dict(distdiscrete)
                self._construct_doc(docdict_discrete, dct.get(self.name))

            # 离散型随机变量不包含 scale 参数，移除它
            self.__doc__ = self.__doc__.replace(
                '\n    scale : array_like, '
                'optional\n        scale parameter (default=1)', '')

    # 更新 _ctor_param 方法的当前版本，可能会被用户更新
    # 用于冻结时使用
    def _updated_ctor_param(self):
        """Return the current version of _ctor_param, possibly updated by user.

        Used by freezing.
        Keep this in sync with the signature of __init__.
        """
        dct = self._ctor_param.copy()
        dct['a'] = self.a
        dct['b'] = self.b
        dct['badvalue'] = self.badvalue
        dct['moment_tol'] = self.moment_tol
        dct['inc'] = self.inc
        dct['name'] = self.name
        dct['shapes'] = self.shapes
        return dct

    # 检查 k 是否为非零整数
    def _nonzero(self, k, *args):
        return floor(k) == k

    # 计算离散随机变量的概率质量函数（PMF）
    def _pmf(self, k, *args):
        return self._cdf(k, *args) - self._cdf(k-1, *args)
    # 计算给定分布下 k 的对数概率质量函数（PMF）值
    def _logpmf(self, k, *args):
        return log(self._pmf(k, *args))

    # 计算给定分布下 k 的对数概率密度函数（PDF）或概率质量函数（PMF）值
    # 对于连续分布，使用 PDF；对于离散分布，使用 PMF。有时两者的区别不重要，这使得 `_logpxf` 可以同时用于两种类型的分布。
    def _logpxf(self, k, *args):
        return self._logpmf(k, *args)

    # 解包 theta，获取位置（loc）和缩放（scale）参数
    def _unpack_loc_scale(self, theta):
        try:
            loc = theta[-1]
            scale = 1
            args = tuple(theta[:-1])
        except IndexError as e:
            raise ValueError("Not enough input arguments.") from e
        return loc, scale, args

    # 计算给定分布下小于等于 k 的累积分布函数（CDF）值
    def _cdf_single(self, k, *args):
        # 获取分布的支持范围
        _a, _b = self._get_support(*args)
        # 生成从下限到 k 的整数数组
        m = arange(int(_a), k+1)
        # 计算 m 对应的 PMF 值之和作为累积分布函数值
        return np.sum(self._pmf(m, *args), axis=0)

    # 计算给定分布下小于等于 x 的累积分布函数（CDF）值
    def _cdf(self, x, *args):
        # 取 x 的整数部分作为参数 k
        k = floor(x)
        # 调用 _cdfvec 方法计算累积分布函数值
        return self._cdfvec(k, *args)

    # 在 rv_generic 中定义了通用的 _logcdf, _sf, _logsf, _ppf, _isf, _rvs

    # 生成给定分布类型的随机变量
    def rvs(self, *args, **kwargs):
        """给定类型的随机变量。

        参数
        ----------
        arg1, arg2, arg3,... : array_like
            分布的形状参数（详见实例对象的文档字符串）。
        loc : array_like, optional
            位置参数（默认为0）。
        size : int or tuple of ints, optional
            定义随机变量的数量（默认为1）。注意，`size` 必须作为关键字参数提供，而不是位置参数。
        random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional

            如果 `random_state` 为 None（或 `np.random`），则使用
            `numpy.random.RandomState` 单例。
            如果 `random_state` 是整数，则使用一个新的 ``RandomState`` 实例，
            并使用 `random_state` 作为种子。
            如果 `random_state` 已经是 ``Generator`` 或 ``RandomState``
            实例，则直接使用该实例。

        返回
        -------
        rvs : ndarray 或标量
            给定 `size` 的随机变量。

        """
        # 设置 `discrete` 参数为 True
        kwargs['discrete'] = True
        # 调用父类的 rvs 方法生成随机变量
        return super().rvs(*args, **kwargs)
    def pmf(self, k, *args, **kwds):
        """Probability mass function at k of the given RV.

        Parameters
        ----------
        k : array_like
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        pmf : array_like
            Probability mass function evaluated at k

        """
        # 解析参数并返回修正后的参数列表和位置参数
        args, loc, _ = self._parse_args(*args, **kwds)
        # 将 k 和 loc 转换为数组
        k, loc = map(asarray, (k, loc))
        # 将 args 中的每个参数转换为数组
        args = tuple(map(asarray, args))
        # 获取分布的支持范围
        _a, _b = self._get_support(*args)
        # 调整 k，使其减去 loc
        k = asarray(k-loc)
        # 检查参数是否有效
        cond0 = self._argcheck(*args)
        # 检查 k 是否在支持范围内
        cond1 = (k >= _a) & (k <= _b)
        # 如果不是 rv_sample 的实例，则进一步检查 k 是否非零
        if not isinstance(self, rv_sample):
            cond1 = cond1 & self._nonzero(k, *args)
        # 最终有效条件
        cond = cond0 & cond1
        # 创建输出数组，并填充无效值
        output = zeros(shape(cond), 'd')
        place(output, (1-cond0) + np.isnan(k), self.badvalue)
        # 如果有任何有效条件，则计算 pmf，并在输出数组中进行剪切
        if np.any(cond):
            goodargs = argsreduce(cond, *((k,)+args))
            place(output, cond, np.clip(self._pmf(*goodargs), 0, 1))
        # 如果输出数组的维度为零，则返回其标量值
        if output.ndim == 0:
            return output[()]
        return output

    def logpmf(self, k, *args, **kwds):
        """Log of the probability mass function at k of the given RV.

        Parameters
        ----------
        k : array_like
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter. Default is 0.

        Returns
        -------
        logpmf : array_like
            Log of the probability mass function evaluated at k.

        """
        # 解析参数并返回修正后的参数列表和位置参数
        args, loc, _ = self._parse_args(*args, **kwds)
        # 将 k 和 loc 转换为数组
        k, loc = map(asarray, (k, loc))
        # 将 args 中的每个参数转换为数组
        args = tuple(map(asarray, args))
        # 获取分布的支持范围
        _a, _b = self._get_support(*args)
        # 调整 k，使其减去 loc
        k = asarray(k-loc)
        # 检查参数是否有效
        cond0 = self._argcheck(*args)
        # 检查 k 是否在支持范围内
        cond1 = (k >= _a) & (k <= _b)
        # 如果不是 rv_sample 的实例，则进一步检查 k 是否非零
        if not isinstance(self, rv_sample):
            cond1 = cond1 & self._nonzero(k, *args)
        # 最终有效条件
        cond = cond0 & cond1
        # 创建输出数组，并填充 -inf
        output = empty(shape(cond), 'd')
        output.fill(-inf)
        place(output, (1-cond0) + np.isnan(k), self.badvalue)
        # 如果有任何有效条件，则计算 logpmf，并在输出数组中存储结果
        if np.any(cond):
            goodargs = argsreduce(cond, *((k,)+args))
            place(output, cond, self._logpmf(*goodargs))
        # 如果输出数组的维度为零，则返回其标量值
        if output.ndim == 0:
            return output[()]
        return output
    def cdf(self, k, *args, **kwds):
        """Cumulative distribution function of the given RV.

        Parameters
        ----------
        k : array_like, int
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        cdf : ndarray
            Cumulative distribution function evaluated at `k`.

        """
        # 解析参数，获取参数列表和位置参数
        args, loc, _ = self._parse_args(*args, **kwds)
        # 将 k 和 loc 转换为数组
        k, loc = map(asarray, (k, loc))
        # 将参数 args 转换为数组
        args = tuple(map(asarray, args))
        # 获取分布的支持范围
        _a, _b = self._get_support(*args)
        # 计算偏移后的 k
        k = asarray(k - loc)
        # 检查参数是否有效
        cond0 = self._argcheck(*args)
        # 检查 k 是否在支持范围内的条件
        cond1 = (k >= _a) & (k < _b)
        # k 是否大于等于上界的条件
        cond2 = (k >= _b)
        # k 是否为负无穷的条件
        cond3 = np.isneginf(k)
        # 组合所有条件
        cond = cond0 & cond1 & np.isfinite(k)

        # 创建一个与 cond 形状相同的零数组
        output = zeros(shape(cond), 'd')
        # 将 cond2 为真的位置设为 1.0
        place(output, cond2*(cond0 == cond0), 1.0)
        # 将 cond3 为真的位置设为 0.0
        place(output, cond3*(cond0 == cond0), 0.0)
        # 将不满足 cond0 或 k 为 NaN 的位置设为 self.badvalue
        place(output, (1-cond0) + np.isnan(k), self.badvalue)

        # 如果有任何满足 cond 的情况
        if np.any(cond):
            # 对满足条件的参数进行简化处理
            goodargs = argsreduce(cond, *((k,)+args))
            # 将 cond 为真的位置设为经过裁剪的累积分布函数值
            place(output, cond, np.clip(self._cdf(*goodargs), 0, 1))
        # 如果 output 的维度为 0，则返回 output 的第一个元素
        if output.ndim == 0:
            return output[()]
        # 否则返回 output
        return output

    def logcdf(self, k, *args, **kwds):
        """Log of the cumulative distribution function at k of the given RV.

        Parameters
        ----------
        k : array_like, int
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        logcdf : array_like
            Log of the cumulative distribution function evaluated at k.

        """
        # 解析参数，获取参数列表和位置参数
        args, loc, _ = self._parse_args(*args, **kwds)
        # 将 k 和 loc 转换为数组
        k, loc = map(asarray, (k, loc))
        # 将参数 args 转换为数组
        args = tuple(map(asarray, args))
        # 获取分布的支持范围
        _a, _b = self._get_support(*args)
        # 计算偏移后的 k
        k = asarray(k - loc)
        # 检查参数是否有效
        cond0 = self._argcheck(*args)
        # 检查 k 是否在支持范围内的条件
        cond1 = (k >= _a) & (k < _b)
        # k 是否大于等于上界的条件
        cond2 = (k >= _b)
        # 组合有效的条件
        cond = cond0 & cond1
        # 创建一个空数组，形状与 cond 相同，填充为负无穷
        output = empty(shape(cond), 'd')
        output.fill(-inf)
        # 将不满足 cond0 或 k 为 NaN 的位置设为 self.badvalue
        place(output, (1-cond0) + np.isnan(k), self.badvalue)
        # 将 cond2 为真的位置设为 0.0
        place(output, cond2*(cond0 == cond0), 0.0)

        # 如果有任何满足 cond 的情况
        if np.any(cond):
            # 对满足条件的参数进行简化处理
            goodargs = argsreduce(cond, *((k,)+args))
            # 将 cond 为真的位置设为经过对数累积分布函数的值
            place(output, cond, self._logcdf(*goodargs))
        # 如果 output 的维度为 0，则返回 output 的第一个元素
        if output.ndim == 0:
            return output[()]
        # 否则返回 output
        return output
    def sf(self, k, *args, **kwds):
        """
        Survival function (1 - `cdf`) at k of the given RV.

        Parameters
        ----------
        k : array_like
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        sf : array_like
            Survival function evaluated at k.

        """
        # 解析参数，获取分布的参数和位置参数
        args, loc, _ = self._parse_args(*args, **kwds)
        # 将 k 和 loc 转换为数组形式
        k, loc = map(asarray, (k, loc))
        # 将 args 中的每个参数转换为数组形式
        args = tuple(map(asarray, args))
        # 获取分布的支持范围
        _a, _b = self._get_support(*args)
        # 将 k 转换为相对于 loc 的偏移量
        k = asarray(k - loc)
        # 检查参数是否有效
        cond0 = self._argcheck(*args)
        # 检查 k 是否在支持范围内
        cond1 = (k >= _a) & (k < _b)
        # 处理 k 小于支持范围下限且 cond0 为真的情况
        cond2 = ((k < _a) | np.isneginf(k)) & cond0
        # 组合所有条件
        cond = cond0 & cond1 & np.isfinite(k)
        # 创建与 cond 形状相同的零数组
        output = zeros(shape(cond), 'd')
        # 将 cond0 为假或者 k 是 NaN 的位置设为 self.badvalue
        place(output, (1-cond0) + np.isnan(k), self.badvalue)
        # 将 cond2 为真的位置设为 1.0
        place(output, cond2, 1.0)
        # 如果存在满足条件的项，则计算相应的 sf 值并限制在 [0, 1] 范围内
        if np.any(cond):
            goodargs = argsreduce(cond, *((k,)+args))
            place(output, cond, np.clip(self._sf(*goodargs), 0, 1))
        # 如果输出数组的维度为 0，则返回其第一个元素
        if output.ndim == 0:
            return output[()]
        # 否则返回输出数组
        return output

    def logsf(self, k, *args, **kwds):
        """
        Log of the survival function of the given RV.

        Returns the log of the "survival function," defined as 1 - `cdf`,
        evaluated at `k`.

        Parameters
        ----------
        k : array_like
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        logsf : ndarray
            Log of the survival function evaluated at `k`.

        """
        # 解析参数，获取分布的参数和位置参数
        args, loc, _ = self._parse_args(*args, **kwds)
        # 将 k 和 loc 转换为数组形式
        k, loc = map(asarray, (k, loc))
        # 将 args 中的每个参数转换为数组形式
        args = tuple(map(asarray, args))
        # 获取分布的支持范围
        _a, _b = self._get_support(*args)
        # 将 k 转换为相对于 loc 的偏移量
        k = asarray(k - loc)
        # 检查参数是否有效
        cond0 = self._argcheck(*args)
        # 检查 k 是否在支持范围内
        cond1 = (k >= _a) & (k < _b)
        # 处理 k 小于支持范围下限且 cond0 为真的情况
        cond2 = (k < _a) & cond0
        # 组合所有条件
        cond = cond0 & cond1
        # 创建与 cond 形状相同的空数组，并填充为负无穷
        output = empty(shape(cond), 'd')
        output.fill(-inf)
        # 将 cond0 为假或者 k 是 NaN 的位置设为 self.badvalue
        place(output, (1-cond0) + np.isnan(k), self.badvalue)
        # 将 cond2 为真的位置设为 0.0
        place(output, cond2, 0.0)
        # 如果存在满足条件的项，则计算相应的 logsf 值
        if np.any(cond):
            goodargs = argsreduce(cond, *((k,)+args))
            place(output, cond, self._logsf(*goodargs))
        # 如果输出数组的维度为 0，则返回其第一个元素
        if output.ndim == 0:
            return output[()]
        # 否则返回输出数组
        return output
    # 定义概率分布的百分位点函数，即给定随机变量的累积分布函数 (`cdf`) 的逆函数
    # 在给定概率分布参数和位置参数 (`loc`) 的情况下，计算 q 对应的分位数

    def ppf(self, q, *args, **kwds):
        """Percent point function (inverse of `cdf`) at q of the given RV.

        Parameters
        ----------
        q : array_like
            Lower tail probability.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        k : array_like
            Quantile corresponding to the lower tail probability, q.

        """
        # 解析参数 args 和 loc，并且获取默认值 '_'
        args, loc, _ = self._parse_args(*args, **kwds)
        # 将 q 和 loc 转换为数组
        q, loc = map(asarray, (q, loc))
        # 将 args 元组中的每个元素都转换为数组
        args = tuple(map(asarray, args))
        # 获取分布的支持范围 _a 和 _b
        _a, _b = self._get_support(*args)
        # 检查参数是否有效，同时确保 loc 是有效的数值
        cond0 = self._argcheck(*args) & (loc == loc)
        # 检查 q 是否在 (0, 1) 之间
        cond1 = (q > 0) & (q < 1)
        # 特殊情况：当 q = 1 且 cond0 成立时
        cond2 = (q == 1) & cond0
        # 组合所有条件，确保在有效参数范围内
        cond = cond0 & cond1
        # 创建一个与 cond 形状相同的填充值为 self.badvalue 的数组
        output = np.full(shape(cond), fill_value=self.badvalue, dtype='d')
        # 将 (q == 0) 且 cond 成立时，将 _a-1 + loc 放置到 output 中
        place(output, (q == 0)*(cond == cond), _a-1 + loc)
        # 将 cond2 成立时，将 _b + loc 放置到 output 中
        place(output, cond2, _b + loc)
        
        # 如果存在任何 cond 成立，则进一步处理
        if np.any(cond):
            # 使用 argsreduce 函数减少参数，并获取有效的参数列表 goodargs
            goodargs = argsreduce(cond, *((q,)+args+(loc,)))
            loc, goodargs = goodargs[-1], goodargs[:-1]
            # 将 cond 成立时的结果放置到 output 中
            place(output, cond, self._ppf(*goodargs) + loc)

        # 如果 output 的维度为 0，则返回其标量值
        if output.ndim == 0:
            return output[()]
        # 否则返回 output 数组
        return output
    def isf(self, q, *args, **kwds):
        """Inverse survival function (inverse of `sf`) at q of the given RV.

        Parameters
        ----------
        q : array_like
            Upper tail probability.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        k : ndarray or scalar
            Quantile corresponding to the upper tail probability, q.

        """
        # 解析参数，获取形状参数和位置参数
        args, loc, _ = self._parse_args(*args, **kwds)
        
        # 将 q 和 loc 转换为数组
        q, loc = map(asarray, (q, loc))
        
        # 将 args 中的每个参数都转换为数组
        args = tuple(map(asarray, args))
        
        # 获取分布的支持范围
        _a, _b = self._get_support(*args)
        
        # 检查参数条件
        cond0 = self._argcheck(*args) & (loc == loc)
        
        # 判断 q 是否在 (0, 1) 范围内
        cond1 = (q > 0) & (q < 1)
        
        # 判断 q 是否等于 1，并且满足 cond0 条件
        cond2 = (q == 1) & cond0
        
        # 判断 q 是否等于 0，并且满足 cond0 条件
        cond3 = (q == 0) & cond0
        
        # 组合条件
        cond = cond0 & cond1

        # 创建一个填充了 self.badvalue 的数组，类型为 'd'
        output = np.full(shape(cond), fill_value=self.badvalue, dtype='d')
        
        # 计算下界和上界
        lower_bound = _a - 1 + loc
        upper_bound = _b + loc
        
        # 将 lower_bound 放置到 output 中，如果 cond2 和 cond3 都为真
        place(output, cond2*(cond == cond), lower_bound)
        place(output, cond3*(cond == cond), upper_bound)

        # 如果至少有一个有效的参数，调用 argsreduce 函数
        if np.any(cond):
            goodargs = argsreduce(cond, *((q,)+args+(loc,)))
            loc, goodargs = goodargs[-1], goodargs[:-1]
            # 和 ticket 766 相同的 PB
            place(output, cond, self._isf(*goodargs) + loc)

        # 如果 output 的维度为 0，则返回 output 的单个元素
        if output.ndim == 0:
            return output[()]
        
        # 返回 output 数组
        return output

    def _entropy(self, *args):
        # 如果对象具有 'pk' 属性，则返回 pk 的熵值
        if hasattr(self, 'pk'):
            return stats.entropy(self.pk)
        else:
            # 否则，获取分布的支持范围，并计算其期望熵
            _a, _b = self._get_support(*args)
            return _expect(lambda x: entr(self.pmf(x, *args)),
                           _a, _b, self.ppf(0.5, *args), self.inc)

    def _param_info(self):
        # 获取形状信息
        shape_info = self._shape_info()
        
        # 创建位置信息对象
        loc_info = _ShapeInfo("loc", True, (-np.inf, np.inf), (False, False))
        
        # 将形状信息和位置信息合并成参数信息列表
        param_info = shape_info + [loc_info]
        
        # 返回参数信息列表
        return param_info
# 定义一个辅助函数，用于计算函数 `fun` 的期望值。
def _expect(fun, lb, ub, x0, inc, maxcount=1000, tolerance=1e-10,
            chunksize=32):
    # 如果支持集大小足够小，则提前返回结果
    if (ub - lb) <= chunksize:
        # 生成支持集 supp，包括 lb 到 ub 的所有值，步长为 inc
        supp = np.arange(lb, ub+1, inc)
        # 计算 fun 在支持集上的取值，并求和返回
        vals = fun(supp)
        return np.sum(vals)

    # 否则，从 x0 开始迭代
    if x0 < lb:
        x0 = lb
    if x0 > ub:
        x0 = ub

    count, tot = 0, 0.
    # 在 [x0, ub] 区间上进行迭代，包括边界值 ub
    for x in _iter_chunked(x0, ub+1, chunksize=chunksize, inc=inc):
        count += x.size
        delta = np.sum(fun(x))
        tot += delta
        # 如果 delta 绝对值小于容差乘以区间大小，则认为达到精度要求，提前退出循环
        if abs(delta) < tolerance * x.size:
            break
        # 如果超过最大迭代次数，则发出警告并返回当前累积值
        if count > maxcount:
            warnings.warn('expect(): sum did not converge',
                          RuntimeWarning, stacklevel=3)
            return tot

    # 在 [lb, x0) 区间上进行迭代，不包括边界值 x0
    for x in _iter_chunked(x0-1, lb-1, chunksize=chunksize, inc=-inc):
        count += x.size
        delta = np.sum(fun(x))
        tot += delta
        # 如果 delta 绝对值小于容差乘以区间大小，则认为达到精度要求，提前退出循环
        if abs(delta) < tolerance * x.size:
            break
        # 如果超过最大迭代次数，则发出警告并中断循环
        if count > maxcount:
            warnings.warn('expect(): sum did not converge',
                          RuntimeWarning, stacklevel=3)
            break

    # 返回最终累积的期望值
    return tot


def _iter_chunked(x0, x1, chunksize=4, inc=1):
    """按照给定步长和块大小，在 x0 到 x1 之间进行迭代。

    x0 必须是有限的，x1 可以是无限的。在后一种情况下，迭代器是无限的。
    处理 x0 小于 x1 和 x0 大于 x1 两种情况。在后一种情况下，向下迭代（确保设置 inc < 0）。

    >>> from scipy.stats._distn_infrastructure import _iter_chunked
    >>> [x for x in _iter_chunked(2, 5, inc=2)]
    [array([2, 4])]
    >>> [x for x in _iter_chunked(2, 11, inc=2)]
    [array([2, 4, 6, 8]), array([10])]
    >>> [x for x in _iter_chunked(2, -5, inc=-2)]
    [array([ 2,  0, -2, -4])]
    >>> [x for x in _iter_chunked(2, -9, inc=-2)]
    [array([ 2,  0, -2, -4]), array([-6, -8])]

    """
    if inc == 0:
        raise ValueError('Cannot increment by zero.')
    if chunksize <= 0:
        raise ValueError('Chunk size must be positive; got %s.' % chunksize)

    s = 1 if inc > 0 else -1
    stepsize = abs(chunksize * inc)

    x = x0
    while (x - x1) * inc < 0:
        delta = min(stepsize, abs(x - x1))
        step = delta * s
        # 生成从 x 到 x + step，步长为 inc 的支持集
        supp = np.arange(x, x + step, inc)
        x += step
        yield supp


class rv_sample(rv_discrete):
    """由支持集和值定义的 'sample' 离散分布。

    构造函数忽略大部分参数，只需要 `values` 参数。

    """
    def __init__(self, a=0, b=inf, name=None, badvalue=None,
                 moment_tol=1e-8, values=None, inc=1, longname=None,
                 shapes=None, seed=None):
        # 调用父类 rv_discrete 的构造函数并传入种子参数
        super(rv_discrete, self).__init__(seed)

        if values is None:
            # 如果未提供 values 参数，则引发数值错误异常
            raise ValueError("rv_sample.__init__(..., values=None,...)")

        # 存储构造函数的参数到字典 _ctor_param 中，用于冻结实例
        self._ctor_param = dict(
            a=a, b=b, name=name, badvalue=badvalue,
            moment_tol=moment_tol, values=values, inc=inc,
            longname=longname, shapes=shapes, seed=seed)

        if badvalue is None:
            # 如果 badvalue 未指定，则设为 NaN
            badvalue = nan
        self.badvalue = badvalue
        self.moment_tol = moment_tol
        self.inc = inc
        self.shapes = shapes
        self.vecentropy = self._entropy  # 设置属性 vecentropy 为实例方法 _entropy 的引用

        xk, pk = values  # 将 values 解包为 xk 和 pk

        if np.shape(xk) != np.shape(pk):
            # 如果 xk 和 pk 的形状不一致，则引发数值错误异常
            raise ValueError("xk and pk must have the same shape.")
        if np.less(pk, 0.0).any():
            # 如果 pk 中有任何负值，则引发数值错误异常
            raise ValueError("All elements of pk must be non-negative.")
        if not np.allclose(np.sum(pk), 1):
            # 如果 pk 的总和不接近 1，则引发数值错误异常
            raise ValueError("The sum of provided pk is not 1.")
        if not len(set(np.ravel(xk))) == np.size(xk):
            # 如果 xk 中包含重复值，则引发数值错误异常
            raise ValueError("xk may not contain duplicate values.")

        indx = np.argsort(np.ravel(xk))  # 对 xk 进行排序并返回索引
        self.xk = np.take(np.ravel(xk), indx, 0)  # 按照排序后的索引重新排列 xk
        self.pk = np.take(np.ravel(pk), indx, 0)  # 按照排序后的索引重新排列 pk
        self.a = self.xk[0]  # 设置分布的起始点为 xk 的第一个值
        self.b = self.xk[-1]  # 设置分布的结束点为 xk 的最后一个值

        self.qvals = np.cumsum(self.pk, axis=0)  # 计算累积分布函数的值并存储在 qvals 中

        self.shapes = ' '   # 设置 shapes 属性为一个空格字符串，绕过检查

        # 使用指定的方法构建参数解析器，并设置输入和输出的位置参数
        self._construct_argparser(meths_to_inspect=[self._pmf],
                                  locscale_in='loc=0',
                                  locscale_out='loc, 1')  # 对于离散随机变量，scale 固定为 1

        self._attach_methods()  # 调用 _attach_methods 方法，附加动态创建的方法

        self._construct_docstrings(name, longname)  # 构建文档字符串

    def __getstate__(self):
        dct = self.__dict__.copy()

        # 移除需要在 rv_generic.__setstate__ 中重新生成的方法
        attrs = ["_parse_args", "_parse_args_stats", "_parse_args_rvs"]
        [dct.pop(attr, None) for attr in attrs]

        return dct

    def _attach_methods(self):
        """Attaches dynamically created argparser methods."""
        self._attach_argparser_methods()  # 调用 _attach_argparser_methods 方法，附加动态创建的参数解析器方法

    def _get_support(self, *args):
        """Return the support of the (unscaled, unshifted) distribution.

        Parameters
        ----------
        arg1, arg2, ... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).

        Returns
        -------
        a, b : numeric (float, or int or +/-np.inf)
            end-points of the distribution's support.
        """
        return self.a, self.b  # 返回分布的起始点和结束点

    def _pmf(self, x):
        # 返回离散随机变量的概率质量函数值
        return np.select([x == k for k in self.xk],
                         [np.broadcast_arrays(p, x)[0] for p in self.pk], 0)
    # 根据输入的数组 x，返回累积分布函数的值
    def _cdf(self, x):
        # 使用 numpy 的 broadcast_arrays 函数将 x 扩展为与 self.xk 兼容的数组形状
        xx, xxk = np.broadcast_arrays(x[:, None], self.xk)
        # 找到每个元素在 self.xk 中第一次大于对应元素的位置索引
        indx = np.argmax(xxk > xx, axis=-1) - 1
        # 返回对应位置的累积分布函数值
        return self.qvals[indx]

    # 根据输入的数组 q，返回分位点函数的值
    def _ppf(self, q):
        # 使用 numpy 的 broadcast_arrays 函数将 q 扩展为与 self.qvals 兼容的数组形状
        qq, sqq = np.broadcast_arrays(q[..., None], self.qvals)
        # 找到每个元素在 self.qvals 中第一次大于等于对应元素的位置索引
        indx = np.argmax(sqq >= qq, axis=-1)
        # 返回对应位置的分位点函数值
        return self.xk[indx]

    # 根据 size 和 random_state 参数生成随机变量样本
    def _rvs(self, size=None, random_state=None):
        # 如果 size 为 None，则生成一个随机数 U，并将其转换为数组
        U = random_state.uniform(size=size)
        if size is None:
            U = np.array(U, ndmin=1)
            # 计算并返回单个样本值
            Y = self._ppf(U)[0]
        else:
            # 计算并返回多个样本值
            Y = self._ppf(U)
        return Y

    # 计算分布的熵值
    def _entropy(self):
        return stats.entropy(self.pk)

    # 计算分布的 n 阶矩
    def generic_moment(self, n):
        # 将 n 转换为数组形式
        n = np.asarray(n)
        # 计算并返回分布的 n 阶矩
        return np.sum(self.xk**n[np.newaxis, ...] * self.pk, axis=0)

    # 根据指定的函数 fun，以及上下界 lb 和 ub，计算期望值
    def _expect(self, fun, lb, ub, *args, **kwds):
        # 忽略所有的参数 args 和 kwds，直接通过暴力求和方法计算期望值
        # 获取位于 [lb, ub] 区间内的支持点
        supp = self.xk[(lb <= self.xk) & (self.xk <= ub)]
        # 计算支持点对应的函数值
        vals = fun(supp)
        # 返回支持点函数值的总和作为期望值
        return np.sum(vals)
# 这是一个用于检查形状的实用函数，由 geninvgauss_gen 类中的 _rvs() 函数使用。
# 它比较参数 argshape 和 size 所表示的元组。

def _check_shape(argshape, size):
    # scalar_shape 用于将 _rvs_scalar() 返回的一维随机变量数组转换成的形状，
    # 以便将其复制到 _rvs() 的输出数组中。
    scalar_shape = []
    # bc 是一个布尔元组，与 size 的长度相同。如果与该索引相关联的数据在 _rvs_scalar() 的一次调用中生成，
    # 则 bc[j] 为 True。
    bc = []
    # 使用 zip_longest 函数遍历 argshape 和 size 的逆序，fillvalue 为 1。
    for argdim, sizedim in zip_longest(argshape[::-1], size[::-1], fillvalue=1):
        # 如果 sizedim 大于 argdim，或者 argdim 和 sizedim 都为 1，则将 sizedim 添加到 scalar_shape，
        # 并将 True 添加到 bc。
        if sizedim > argdim or (argdim == sizedim == 1):
            scalar_shape.append(sizedim)
            bc.append(True)
        else:
            # 否则将 False 添加到 bc。
            bc.append(False)
    # 返回 scalar_shape 和 bc 的逆序元组。
    return tuple(scalar_shape[::-1]), tuple(bc[::-1])


# 收集统计分布的名称及其生成器的函数名称。
def get_distribution_names(namespace_pairs, rv_base_class):
    # 初始化空列表，用于存储统计分布的名称和生成器的名称。
    distn_names = []
    distn_gen_names = []
    # 遍历 namespace_pairs 中的每个 (name, value) 对。
    for name, value in namespace_pairs:
        # 如果 name 以 '_' 开头，则跳过当前循环。
        if name.startswith('_'):
            continue
        # 如果 name 以 '_gen' 结尾且 value 是 rv_base_class 的子类，则将 name 添加到 distn_gen_names。
        if name.endswith('_gen') and issubclass(value, rv_base_class):
            distn_gen_names.append(name)
        # 如果 value 是 rv_base_class 的实例，则将 name 添加到 distn_names。
        if isinstance(value, rv_base_class):
            distn_names.append(name)
    # 返回统计分布名称列表 distn_names 和生成器名称列表 distn_gen_names。
    return distn_names, distn_gen_names
```