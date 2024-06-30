# `D:\src\scipysrc\scipy\scipy\stats\_continuous_distns.py`

```
#
# Author:  Travis Oliphant  2002-2011 with contributions from
#          SciPy Developers 2004-2011
#

# 导入警告模块
import warnings
# 导入集合抽象基类中的Iterable类
from collections.abc import Iterable
# 导入装饰器相关模块
from functools import wraps, cached_property
# 导入 ctypes 模块
import ctypes

# 导入 numpy 库，并将其命名为 np
import numpy as np
# 导入 numpy.polynomial 中的 Polynomial 类
from numpy.polynomial import Polynomial
# 导入 scipy.interpolate 中的 BSpline 类
from scipy.interpolate import BSpline
# 导入 scipy._lib.doccer 模块中的函数和类
from scipy._lib.doccer import (extend_notes_in_docstring,
                               replace_notes_in_docstring,
                               inherit_docstring_from)
# 导入 scipy._lib._ccallback 模块中的 LowLevelCallable 类
from scipy._lib._ccallback import LowLevelCallable
# 导入 scipy.optimize 模块
from scipy import optimize
# 导入 scipy.integrate 模块
from scipy import integrate
# 导入 scipy.special 库并将其命名为 sc
import scipy.special as sc

# 导入 scipy.special._ufuncs 模块并将其命名为 scu
import scipy.special._ufuncs as scu
# 导入 scipy._lib._util 模块中的 _lazyselect 和 _lazywhere 函数
from scipy._lib._util import _lazyselect, _lazywhere

# 导入当前目录下的 _stats 模块
from . import _stats
# 导入当前目录下的 _tukeylambda_stats 模块中的函数并重命名为 _tlvar 和 _tlkurt
from ._tukeylambda_stats import (tukeylambda_variance as _tlvar,
                                 tukeylambda_kurtosis as _tlkurt)
# 导入当前目录下的 _distn_infrastructure 模块中的函数和类
from ._distn_infrastructure import (_vectorize_rvs_over_shapes,
    get_distribution_names, _kurtosis, _isintegral,
    rv_continuous, _skew, _get_fixed_fit_value, _check_shape, _ShapeInfo)
# 导入当前目录下的 _ksstats 模块中的函数
from ._ksstats import kolmogn, kolmognp, kolmogni
# 导入当前目录下的 _constants 模块中的常量
from ._constants import (_XMIN, _LOGXMIN, _EULER, _ZETA3, _SQRT_PI,
                         _SQRT_2_OVER_PI, _LOG_SQRT_2_OVER_PI)
# 导入当前目录下的 _censored_data 模块中的 CensoredData 类
from ._censored_data import CensoredData
# 导入 scipy.optimize 模块中的 root_scalar 函数
from scipy.optimize import root_scalar
# 导入 scipy.stats._warnings_errors 模块中的 FitError 类
from scipy.stats._warnings_errors import FitError
# 导入 scipy.stats 库并将其命名为 stats
import scipy.stats as stats


def _remove_optimizer_parameters(kwds):
    """
    Remove the optimizer-related keyword arguments 'loc', 'scale' and
    'optimizer' from `kwds`.  Then check that `kwds` is empty, and
    raise `TypeError("Unknown arguments: %s." % kwds)` if it is not.

    This function is used in the fit method of distributions that override
    the default method and do not use the default optimization code.

    `kwds` is modified in-place.
    """
    # 从 kwds 中移除与优化相关的关键字参数 'loc', 'scale', 'optimizer' 和 'method'
    kwds.pop('loc', None)
    kwds.pop('scale', None)
    kwds.pop('optimizer', None)
    kwds.pop('method', None)
    # 如果 kwds 非空，则抛出 TypeError 异常，说明存在未知的关键字参数
    if kwds:
        raise TypeError("Unknown arguments: %s." % kwds)


def _call_super_mom(fun):
    # 如果 fit 方法被覆盖，且仅用于最大似然估计 'mle'，而不指定 'method == 'mm'' 或存在被屏蔽的数据，则使用此装饰器调用通用实现
    @wraps(fun)
    def wrapper(self, data, *args, **kwds):
        # 获取 method 参数，默认为 'mle'
        method = kwds.get('method', 'mle').lower()
        # 判断数据是否为 CensoredData 类的实例
        censored = isinstance(data, CensoredData)
        # 如果 method 为 'mm' 或者 data 是 CensoredData 实例且有屏蔽数据，则调用父类的 fit 方法
        if method == 'mm' or (censored and data.num_censored() > 0):
            return super(type(self), self).fit(data, *args, **kwds)
        else:
            # 如果 data 是 CensoredData 实例，则将其转换为未屏蔽数据的数组
            if censored:
                data = data._uncensored
            # 否则，调用原始的 fun 函数处理数据
            return fun(self, data, *args, **kwds)

    return wrapper


def _get_left_bracket(fun, rbrack, lbrack=None):
    # 查找 root_scalar 方法的左括号。可以提供 lbrack 的猜测值作为参数。
    # lbrack 是 root_scalar 方法的左边界的猜测值，fun 是待求解的函数，rbrack 是右括号的值
    # 如果 `lbrack` 为假值（如None），则将其设为 `rbrack - 1`；否则保持不变
    lbrack = lbrack or rbrack - 1
    # 计算 `rbrack` 到 `lbrack` 的差值
    diff = rbrack - lbrack

    # 如果在括号内没有 `fun` 函数值的符号变化，扩展 `rbrack - lbrack` 直到出现符号变化
    def interval_contains_root(lbrack, rbrack):
        # 如果 `fun` 在两个边界的符号不同，返回True
        return np.sign(fun(lbrack)) != np.sign(fun(rbrack))

    # 循环直到找到包含根的区间
    while not interval_contains_root(lbrack, rbrack):
        # 增加差值的两倍
        diff *= 2
        # 更新 `lbrack` 为 `rbrack` 减去增大后的差值
        lbrack = rbrack - diff

        # 如果 `lbrack` 是无穷大，抛出错误信息
        msg = ("The solver could not find a bracket containing a "
               "root to an MLE first order condition.")
        if np.isinf(lbrack):
            raise FitSolverError(msg)

    # 返回找到的 `lbrack` 值，这是包含根的区间的左边界
    return lbrack
# 定义 Kolmogorov-Smirnov 单边检验统计分布的类 ksone_gen，继承自 rv_continuous
class ksone_gen(rv_continuous):
    r"""Kolmogorov-Smirnov one-sided test statistic distribution.

    This is the distribution of the one-sided Kolmogorov-Smirnov (KS)
    statistics :math:`D_n^+` and :math:`D_n^-`
    for a finite sample size ``n >= 1`` (the shape parameter).

    %(before_notes)s

    See Also
    --------
    kstwobign, kstwo, kstest

    Notes
    -----
    :math:`D_n^+` and :math:`D_n^-` are given by

    .. math::

        D_n^+ &= \text{sup}_x (F_n(x) - F(x)),\\
        D_n^- &= \text{sup}_x (F(x) - F_n(x)),\\

    where :math:`F` is a continuous CDF and :math:`F_n` is an empirical CDF.
    `ksone` describes the distribution under the null hypothesis of the KS test
    that the empirical CDF corresponds to :math:`n` i.i.d. random variates
    with CDF :math:`F`.

    %(after_notes)s

    References
    ----------
    .. [1] Birnbaum, Z. W. and Tingey, F.H. "One-sided confidence contours
       for probability distribution functions", The Annals of Mathematical
       Statistics, 22(4), pp 592-596 (1951).

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import ksone
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1, 1)

    Display the probability density function (``pdf``):

    >>> n = 1e+03
    >>> x = np.linspace(ksone.ppf(0.01, n),
    ...                 ksone.ppf(0.99, n), 100)
    >>> ax.plot(x, ksone.pdf(x, n),
    ...         'r-', lw=5, alpha=0.6, label='ksone pdf')

    Alternatively, the distribution object can be called (as a function)
    to fix the shape, location and scale parameters. This returns a "frozen"
    RV object holding the given parameters fixed.

    Freeze the distribution and display the frozen ``pdf``:

    >>> rv = ksone(n)
    >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
    >>> ax.legend(loc='best', frameon=False)
    >>> plt.show()

    Check accuracy of ``cdf`` and ``ppf``:

    >>> vals = ksone.ppf([0.001, 0.5, 0.999], n)
    >>> np.allclose([0.001, 0.5, 0.999], ksone.cdf(vals, n))
    True

    """
    
    # 验证参数 `n` 是否符合条件的方法
    def _argcheck(self, n):
        return (n >= 1) & (n == np.round(n))

    # 返回参数形状信息的方法
    def _shape_info(self):
        return [_ShapeInfo("n", True, (1, np.inf), (True, False))]

    # 返回概率密度函数 `_pdf` 的方法
    def _pdf(self, x, n):
        return -scu._smirnovp(n, x)

    # 返回累积分布函数 `_cdf` 的方法
    def _cdf(self, x, n):
        return scu._smirnovc(n, x)

    # 返回生存函数 `_sf` 的方法
    def _sf(self, x, n):
        return sc.smirnov(n, x)

    # 返回百分位点函数 `_ppf` 的方法
    def _ppf(self, q, n):
        return scu._smirnovci(n, q)

    # 返回逆生存函数 `_isf` 的方法
    def _isf(self, q, n):
        return sc.smirnovi(n, q)


# 创建一个 Kolmogorov-Smirnov 单边检验分布的实例 ksone，并固定参数 a=0.0, b=1.0, name='ksone'
ksone = ksone_gen(a=0.0, b=1.0, name='ksone')


# 定义 Kolmogorov-Smirnov 双边检验统计分布的类 kstwo_gen，继承自 rv_continuous
class kstwo_gen(rv_continuous):
    r"""Kolmogorov-Smirnov two-sided test statistic distribution.

    This is the distribution of the two-sided Kolmogorov-Smirnov (KS)
    statistic :math:`D_n` for a finite sample size ``n >= 1``
    (the shape parameter).

    %(before_notes)s

    See Also
    --------
    kstwobign, ksone, kstest

    Notes
    -----
    :math:`D_n` is given by
    """
    Define the Kolmogorov-Smirnov (KS) two-sided distribution for testing goodness-of-fit.

    .. math::

        D_n = \text{sup}_x |F_n(x) - F(x)|

    where :math:`F` is a (continuous) CDF and :math:`F_n` is an empirical CDF.

    `kstwo` represents the distribution under the null hypothesis of the KS test,
    where the empirical CDF corresponds to :math:`n` i.i.d. random variates with CDF :math:`F`.

    %(after_notes)s

    References
    ----------
    .. [1] Simard, R., L'Ecuyer, P. "Computing the Two-Sided
       Kolmogorov-Smirnov Distribution",  Journal of Statistical Software,
       Vol 39, 11, 1-18 (2011).

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import kstwo
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1, 1)

    Display the probability density function (``pdf``):

    >>> n = 10
    >>> x = np.linspace(kstwo.ppf(0.01, n),
    ...                 kstwo.ppf(0.99, n), 100)
    >>> ax.plot(x, kstwo.pdf(x, n),
    ...         'r-', lw=5, alpha=0.6, label='kstwo pdf')

    Alternatively, the distribution object can be called (as a function)
    to fix the shape, location, and scale parameters. This returns a "frozen"
    RV object holding the given parameters fixed.

    Freeze the distribution and display the frozen ``pdf``:

    >>> rv = kstwo(n)
    >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
    >>> ax.legend(loc='best', frameon=False)
    >>> plt.show()

    Check accuracy of ``cdf`` and ``ppf``:

    >>> vals = kstwo.ppf([0.001, 0.5, 0.999], n)
    >>> np.allclose([0.001, 0.5, 0.999], kstwo.cdf(vals, n))
    True

    """

    # Define a method to check validity of argument 'n' (must be integer and >= 1)
    def _argcheck(self, n):
        return (n >= 1) & (n == np.round(n))

    # Define method to provide shape information for the distribution
    def _shape_info(self):
        return [_ShapeInfo("n", True, (1, np.inf), (True, False))]

    # Define method to determine support for the distribution given 'n'
    def _get_support(self, n):
        return (0.5/(n if not isinstance(n, Iterable) else np.asanyarray(n)),
                1.0)

    # Define probability density function (pdf) for the distribution
    def _pdf(self, x, n):
        return kolmognp(n, x)

    # Define cumulative distribution function (cdf) for the distribution
    def _cdf(self, x, n):
        return kolmogn(n, x)

    # Define survival function (sf) for the distribution
    def _sf(self, x, n):
        return kolmogn(n, x, cdf=False)

    # Define percent point function (ppf) for the distribution
    def _ppf(self, q, n):
        return kolmogni(n, q, cdf=True)

    # Define inverse survival function (isf) for the distribution
    def _isf(self, q, n):
        return kolmogni(n, q, cdf=False)
# Use the pdf, (not the ppf) to compute moments
# 定义一个 Kolmogorov-Smirnov 分布的生成器 kstwo，用于计算矩，momtype=0 表示原点矩，a=0.0 和 b=1.0 分别为分布的下限和上限，name='kstwo' 是分布的名称
kstwo = kstwo_gen(momtype=0, a=0.0, b=1.0, name='kstwo')


class kstwobign_gen(rv_continuous):
    r"""Limiting distribution of scaled Kolmogorov-Smirnov two-sided test statistic.

    This is the asymptotic distribution of the two-sided Kolmogorov-Smirnov
    statistic :math:`\sqrt{n} D_n` that measures the maximum absolute
    distance of the theoretical (continuous) CDF from the empirical CDF.
    (see `kstest`).

    %(before_notes)s

    See Also
    --------
    ksone, kstwo, kstest

    Notes
    -----
    :math:`\sqrt{n} D_n` is given by

    .. math::

        D_n = \text{sup}_x |F_n(x) - F(x)|

    where :math:`F` is a continuous CDF and :math:`F_n` is an empirical CDF.
    `kstwobign`  describes the asymptotic distribution (i.e. the limit of
    :math:`\sqrt{n} D_n`) under the null hypothesis of the KS test that the
    empirical CDF corresponds to i.i.d. random variates with CDF :math:`F`.

    %(after_notes)s

    References
    ----------
    .. [1] Feller, W. "On the Kolmogorov-Smirnov Limit Theorems for Empirical
       Distributions",  Ann. Math. Statist. Vol 19, 177-189 (1948).

    %(example)s

    """
    def _shape_info(self):
        return []

    # 定义概率密度函数 _pdf，通过调用 scu._kolmogp(x) 返回 -scu._kolmogp(x) 的值
    def _pdf(self, x):
        return -scu._kolmogp(x)

    # 定义累积分布函数 _cdf，通过调用 scu._kolmogc(x) 返回 scu._kolmogc(x) 的值
    def _cdf(self, x):
        return scu._kolmogc(x)

    # 定义生存函数 _sf，通过调用 sc.kolmogorov(x) 返回 sc.kolmogorov(x) 的值
    def _sf(self, x):
        return sc.kolmogorov(x)

    # 定义累积分布函数的反函数 _ppf，通过调用 scu._kolmogci(q) 返回 scu._kolmogci(q) 的值
    def _ppf(self, q):
        return scu._kolmogci(q)

    # 定义生存函数的反函数 _isf，通过调用 sc.kolmogi(q) 返回 sc.kolmogi(q) 的值
    def _isf(self, q):
        return sc.kolmogi(q)


# 创建 kstwobign_gen 类的实例 kstwobign，a=0.0 是分布的下限，name='kstwobign' 是分布的名称
kstwobign = kstwobign_gen(a=0.0, name='kstwobign')


## Normal distribution

# loc = mu, scale = std
# 将这些实现放在类定义之外，以便其他分布可以重复使用

# 定义概率密度函数 _norm_pdf，计算正态分布的概率密度函数 np.exp(-x**2/2.0) / np.sqrt(2*np.pi)
def _norm_pdf(x):
    return np.exp(-x**2/2.0) / np.sqrt(2*np.pi)

# 定义对数概率密度函数 _norm_logpdf，计算正态分布的对数概率密度函数 -x**2 / 2.0 - _norm_pdf_logC
def _norm_logpdf(x):
    return -x**2 / 2.0 - _norm_pdf_logC

# 定义累积分布函数 _norm_cdf，通过调用 sc.ndtr(x) 返回 sc.ndtr(x) 的值
def _norm_cdf(x):
    return sc.ndtr(x)

# 定义对数累积分布函数 _norm_logcdf，通过调用 sc.log_ndtr(x) 返回 sc.log_ndtr(x) 的值
def _norm_logcdf(x):
    return sc.log_ndtr(x)

# 定义累积分布函数的反函数 _norm_ppf，通过调用 sc.ndtri(q) 返回 sc.ndtri(q) 的值
def _norm_ppf(q):
    return sc.ndtri(q)

# 定义生存函数 _norm_sf，通过调用 _norm_cdf(-x) 返回 _norm_cdf(-x) 的值
def _norm_sf(x):
    return _norm_cdf(-x)

# 定义对数生存函数 _norm_logsf，通过调用 _norm_logcdf(-x) 返回 _norm_logcdf(-x) 的值
def _norm_logsf(x):
    return _norm_logcdf(-x)

# 定义生存函数的反函数 _norm_isf，通过调用 -_norm_ppf(q) 返回 -_norm_ppf(q) 的值
def _norm_isf(q):
    return -_norm_ppf(q)


class norm_gen(rv_continuous):
    r"""A normal continuous random variable.

    The location (``loc``) keyword specifies the mean.
    The scale (``scale``) keyword specifies the standard deviation.

    %(before_notes)s

    Notes
    -----
    The probability density function for `norm` is:

    .. math::

        f(x) = \frac{\exp(-x^2/2)}{\sqrt{2\pi}}

    for a real number :math:`x`.

    %(after_notes)s

    %(example)s

    """
    def _shape_info(self):
        return []

    # 定义随机变量生成函数 _rvs，通过 random_state.standard_normal(size) 返回标准正态分布的随机变量
    def _rvs(self, size=None, random_state=None):
        return random_state.standard_normal(size)

    # 定义概率密度函数 _pdf，调用 _norm_pdf(x) 返回正态分布的概率密度函数值
    def _pdf(self, x):
        return _norm_pdf(x)

    # 定义对数概率密度函数 _logpdf，调用 _norm_logpdf(x) 返回正态分布的对数概率密度函数值
    def _logpdf(self, x):
        return _norm_logpdf(x)

    # 定义累积分布函数 _cdf，调用 _norm_cdf(x) 返回正态分布的累积分布函数值
    def _cdf(self, x):
        return _norm_cdf(x)
    # 调用 _norm_logcdf 函数计算正态分布的累积分布函数的对数值
    def _logcdf(self, x):
        return _norm_logcdf(x)

    # 调用 _norm_sf 函数计算正态分布的生存函数（1 - 累积分布函数）
    def _sf(self, x):
        return _norm_sf(x)

    # 调用 _norm_logsf 函数计算正态分布的生存函数的对数值
    def _logsf(self, x):
        return _norm_logsf(x)

    # 调用 _norm_ppf 函数计算正态分布的百分位点函数（逆累积分布函数）
    def _ppf(self, q):
        return _norm_ppf(q)

    # 调用 _norm_isf 函数计算正态分布的逆生存函数（百分位点函数的对数值）
    def _isf(self, q):
        return _norm_isf(q)

    # 返回标准正态分布的统计量：期望值为 0.0，方差为 1.0，偏度为 0.0，峰度为 0.0
    def _stats(self):
        return 0.0, 1.0, 0.0, 0.0

    # 返回标准正态分布的熵值计算结果
    def _entropy(self):
        return 0.5*(np.log(2*np.pi)+1)

    # 装饰器函数：调用 _call_super_mom 和 replace_notes_in_docstring 装饰器修饰 fit 方法
    @ _call_super_mom
    @ replace_notes_in_docstring(rv_continuous, notes="""\
        For the normal distribution, method of moments and maximum likelihood
        estimation give identical fits, and explicit formulas for the estimates
        are available.
        This function uses these explicit formulas for the maximum likelihood
        estimation of the normal distribution parameters, so the
        `optimizer` and `method` arguments are ignored.\n\n""")
    # 拟合正态分布的参数到给定数据上
    def fit(self, data, **kwds):
        # 从 kwds 中弹出可能存在的 floc 和 fscale 参数
        floc = kwds.pop('floc', None)
        fscale = kwds.pop('fscale', None)

        # 移除优化器参数
        _remove_optimizer_parameters(kwds)

        # 如果同时指定了 floc 和 fscale，则抛出异常
        if floc is not None and fscale is not None:
            raise ValueError("All parameters fixed. There is nothing to "
                             "optimize.")

        # 将数据转换为 numpy 数组
        data = np.asarray(data)

        # 检查数据是否全部为有限值
        if not np.isfinite(data).all():
            raise ValueError("The data contains non-finite values.")

        # 如果 floc 未指定，则使用数据的均值作为估计值
        if floc is None:
            loc = data.mean()
        else:
            loc = floc

        # 如果 fscale 未指定，则使用数据标准差作为估计值
        if fscale is None:
            scale = np.sqrt(((data - loc)**2).mean())
        else:
            scale = fscale

        # 返回估计得到的 loc 和 scale 参数
        return loc, scale

    # 计算标准正态分布的 n 阶非中心矩
    def _munp(self, n):
        """
        @returns Moments of standard normal distribution for integer n >= 0

        See eq. 16 of https://arxiv.org/abs/1209.4340v2
        """
        # 如果 n 为偶数，返回 (n-1)!! 的值
        if n % 2 == 0:
            return sc.factorial2(n - 1)
        else:
            return 0.
# 创建一个名为 norm 的随机变量生成器对象，使用指定的 name 参数
norm = norm_gen(name='norm')


class alpha_gen(rv_continuous):
    r"""An alpha continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `alpha` ([1]_, [2]_) is:

    .. math::

        f(x, a) = \frac{1}{x^2 \Phi(a) \sqrt{2\pi}} *
                  \exp(-\frac{1}{2} (a-1/x)^2)

    where :math:`\Phi` is the normal CDF, :math:`x > 0`, and :math:`a > 0`.

    `alpha` takes ``a`` as a shape parameter.

    %(after_notes)s

    References
    ----------
    .. [1] Johnson, Kotz, and Balakrishnan, "Continuous Univariate
           Distributions, Volume 1", Second Edition, John Wiley and Sons,
           p. 173 (1994).
    .. [2] Anthony A. Salvia, "Reliability applications of the Alpha
           Distribution", IEEE Transactions on Reliability, Vol. R-34,
           No. 3, pp. 251-252 (1985).

    %(example)s

    """
    _support_mask = rv_continuous._open_support_mask

    # 定义一个方法，返回与分布形状相关的信息
    def _shape_info(self):
        return [_ShapeInfo("a", False, (0, np.inf), (False, False))]

    # 定义概率密度函数 (_pdf) 方法，计算 alpha 分布在给定点 x 和参数 a 的密度
    def _pdf(self, x, a):
        # alpha.pdf(x, a) = 1/(x**2*Phi(a)*sqrt(2*pi)) * exp(-1/2 * (a-1/x)**2)
        return 1.0/(x**2)/_norm_cdf(a)*_norm_pdf(a-1.0/x)

    # 定义对数概率密度函数 (_logpdf) 方法，计算 alpha 分布在给定点 x 和参数 a 的对数密度
    def _logpdf(self, x, a):
        return -2*np.log(x) + _norm_logpdf(a-1.0/x) - np.log(_norm_cdf(a))

    # 定义累积分布函数 (_cdf) 方法，计算 alpha 分布在给定点 x 和参数 a 的累积分布
    def _cdf(self, x, a):
        return _norm_cdf(a-1.0/x) / _norm_cdf(a)

    # 定义累积分布函数的逆 (_ppf) 方法，计算 alpha 分布在给定分位数 q 和参数 a 的逆函数
    def _ppf(self, q, a):
        return 1.0/np.asarray(a - _norm_ppf(q*_norm_cdf(a)))

    # 定义统计特性 (_stats) 方法，返回与 alpha 分布相关的统计特性
    def _stats(self, a):
        return [np.inf]*2 + [np.nan]*2


# 创建一个名为 alpha 的 alpha_gen 类型的随机变量对象，使用 a=0.0 和 name='alpha' 参数
alpha = alpha_gen(a=0.0, name='alpha')


class anglit_gen(rv_continuous):
    r"""An anglit continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `anglit` is:

    .. math::

        f(x) = \sin(2x + \pi/2) = \cos(2x)

    for :math:`-\pi/4 \le x \le \pi/4`.

    %(after_notes)s

    %(example)s

    """
    # 定义一个方法，返回与分布形状相关的信息
    def _shape_info(self):
        return []

    # 定义概率密度函数 (_pdf) 方法，计算 anglit 分布在给定点 x 的密度
    def _pdf(self, x):
        # anglit.pdf(x) = sin(2*x + \pi/2) = cos(2*x)
        return np.cos(2*x)

    # 定义累积分布函数 (_cdf) 方法，计算 anglit 分布在给定点 x 的累积分布
    def _cdf(self, x):
        return np.sin(x+np.pi/4)**2.0

    # 定义生存函数 (_sf) 方法，计算 anglit 分布在给定点 x 的生存函数
    def _sf(self, x):
        return np.cos(x + np.pi / 4) ** 2.0

    # 定义累积分布函数的逆 (_ppf) 方法，计算 anglit 分布在给定分位数 q 的逆函数
    def _ppf(self, q):
        return np.arcsin(np.sqrt(q))-np.pi/4

    # 定义统计特性 (_stats) 方法，返回与 anglit 分布相关的统计特性
    def _stats(self):
        return 0.0, np.pi*np.pi/16-0.5, 0.0, -2*(np.pi**4 - 96)/(np.pi*np.pi-8)**2

    # 定义熵 (_entropy) 方法，计算 anglit 分布的熵
    def _entropy(self):
        return 1-np.log(2)


# 创建一个名为 anglit 的 anglit_gen 类型的随机变量对象，使用 a=-np.pi/4, b=np.pi/4 和 name='anglit' 参数
anglit = anglit_gen(a=-np.pi/4, b=np.pi/4, name='anglit')


class arcsine_gen(rv_continuous):
    r"""An arcsine continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `arcsine` is:

    .. math::

        f(x) = \frac{1}{\pi \sqrt{x (1-x)}}

    for :math:`0 < x < 1`.

    %(after_notes)s

    %(example)s

    """
    # 定义一个方法，返回与分布形状相关的信息
    def _shape_info(self):
        return []

    # 定义概率密度函数 (_pdf) 方法，计算 arcsine 分布在给定点 x 的密度
    def _pdf(self, x):
        # arcsine.pdf(x) = 1/(pi*sqrt(x*(1-x)))
        with np.errstate(divide='ignore'):
            return 1.0/np.pi/np.sqrt(x*(1-x))


# 创建一个名为 arcsine 的 arcsine_gen 类型的随机变量对象，使用 name='arcsine' 参数
    # 计算累积分布函数 (CDF)，对给定的 x 值计算其正弦函数的反正弦，乘以 2/pi
    def _cdf(self, x):
        return 2.0/np.pi*np.arcsin(np.sqrt(x))

    # 计算百分点函数 (PPF)，对给定的 q 值计算其正弦的平方乘以 pi/2
    def _ppf(self, q):
        return np.sin(np.pi/2.0*q)**2.0

    # 计算分布的统计特性，包括均值 (mu)，二阶矩 (mu2)，偏度 (g1)，和峰度 (g2)
    def _stats(self):
        mu = 0.5        # 均值
        mu2 = 1.0/8     # 二阶矩
        g1 = 0          # 偏度
        g2 = -3.0/2.0   # 峰度
        return mu, mu2, g1, g2

    # 计算分布的熵值，这里直接返回一个预先计算好的数值
    def _entropy(self):
        return -0.24156447527049044468
arcsine = arcsine_gen(a=0.0, b=1.0, name='arcsine')
# 创建一个名为arcsine的变量，并调用arcsine_gen函数生成一个arcsine分布的实例，设置参数a=0.0, b=1.0, name='arcsine'

class FitDataError(ValueError):
    """Raised when input data is inconsistent with fixed parameters."""
    # 当输入数据与固定参数不一致时引发此异常。

    def __init__(self, distr, lower, upper):
        self.args = (
            "Invalid values in `data`.  Maximum likelihood "
            f"estimation with {distr!r} requires that {lower!r} < "
            f"(x - loc)/scale  < {upper!r} for each x in `data`.",
        )
        # 初始化方法，设置异常消息，要求输入数据中每个x满足最大似然估计的条件。

class FitSolverError(FitError):
    """
    Raised when a solver fails to converge while fitting a distribution.
    """
    # 当拟合分布时求解器未收敛时引发此异常。

    def __init__(self, mesg):
        emsg = "Solver for the MLE equations failed to converge: "
        emsg += mesg.replace('\n', '')
        self.args = (emsg,)
        # 初始化方法，设置求解器未收敛时的异常消息。

def _beta_mle_a(a, b, n, s1):
    # This function calculates the MLE for parameter `a` of beta distribution,
    # given `b`, `n` (number of data points), and `s1` (sum of the logs of data).

    psiab = sc.psi(a + b)
    func = s1 - n * (-psiab + sc.psi(a))
    return func
    # 计算 beta 分布参数 `a` 的最大似然估计（MLE）函数，根据 `b`, `n` 和 `s1` 给出的条件。

def _beta_mle_ab(theta, n, s1, s2):
    # This function calculates the MLE for parameters `a` and `b` of beta distribution,
    # given `n`, `s1` (sum of the logs of data), and `s2` (sum of the logs of 1 - data).

    # Zeros of this function are critical points of
    # the maximum likelihood function.  Solving this system
    # for theta (which contains a and b) gives the MLE for a and b
    # given `n`, `s1` and `s2`.  `s1` is the sum of the logs of the data,
    # and `s2` is the sum of the logs of 1 - data.  `n` is the number
    # of data points.
    a, b = theta
    psiab = sc.psi(a + b)
    func = [s1 - n * (-psiab + sc.psi(a)),
            s2 - n * (-psiab + sc.psi(b))]
    return func
    # 计算 beta 分布参数 `a` 和 `b` 的最大似然估计（MLE）函数，给定 `n`, `s1` 和 `s2` 的条件。

class beta_gen(rv_continuous):
    r"""A beta continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `beta` is:

    .. math::

        f(x, a, b) = \frac{\Gamma(a+b) x^{a-1} (1-x)^{b-1}}
                          {\Gamma(a) \Gamma(b)}

    for :math:`0 <= x <= 1`, :math:`a > 0`, :math:`b > 0`, where
    :math:`\Gamma` is the gamma function (`scipy.special.gamma`).

    `beta` takes :math:`a` and :math:`b` as shape parameters.

    %(after_notes)s

    %(example)s

    """
    # beta 分布的概率密度函数定义和相关说明。

    def _shape_info(self):
        ia = _ShapeInfo("a", False, (0, np.inf), (False, False))
        ib = _ShapeInfo("b", False, (0, np.inf), (False, False))
        return [ia, ib]
        # 返回 beta 分布的形状参数信息。

    def _rvs(self, a, b, size=None, random_state=None):
        return random_state.beta(a, b, size)
        # 生成符合 beta 分布的随机变量。

    def _pdf(self, x, a, b):
        # beta 分布的概率密度函数
        #                     gamma(a+b) * x**(a-1) * (1-x)**(b-1)
        # beta.pdf(x, a, b) = ------------------------------------
        #                              gamma(a)*gamma(b)
        with np.errstate(over='ignore'):
            return scu._beta_pdf(x, a, b)
        # 使用 scipy.stats._beta_pdf 计算 beta 分布的概率密度函数。
    # 计算 Beta 分布的对数概率密度函数 (log-pdf)，用于返回对数概率密度
    def _logpdf(self, x, a, b):
        # 计算 lPx = log(beta(x; a, b))
        lPx = sc.xlog1py(b - 1.0, -x) + sc.xlogy(a - 1.0, x)
        # 减去 Beta 函数的对数，得到最终的对数概率密度
        lPx -= sc.betaln(a, b)
        return lPx

    # 计算 Beta 分布的累积分布函数 (CDF)
    def _cdf(self, x, a, b):
        return sc.betainc(a, b, x)

    # 计算 Beta 分布的生存函数 (1 - CDF)
    def _sf(self, x, a, b):
        return sc.betaincc(a, b, x)

    # 计算 Beta 分布的反函数 (Inverse CDF，即 PPF)
    def _isf(self, x, a, b):
        return sc.betainccinv(a, b, x)

    # 计算 Beta 分布的分位点函数 (PPF)，与 _ppf 区分开
    def _ppf(self, q, a, b):
        return scu._beta_ppf(q, a, b)

    # 计算 Beta 分布的统计特性，包括均值、方差、偏度和峰度
    def _stats(self, a, b):
        a_plus_b = a + b
        _beta_mean = a / a_plus_b
        _beta_variance = a * b / (a_plus_b**2 * (a_plus_b + 1))
        _beta_skewness = ((2 * (b - a) * np.sqrt(a_plus_b + 1)) /
                          ((a_plus_b + 2) * np.sqrt(a * b)))
        _beta_kurtosis_excess_n = 6 * ((a - b)**2 * (a_plus_b + 1) -
                                       a * b * (a_plus_b + 2))
        _beta_kurtosis_excess_d = a * b * (a_plus_b + 2) * (a_plus_b + 3)
        _beta_kurtosis_excess = _beta_kurtosis_excess_n / _beta_kurtosis_excess_d
        # 返回 Beta 分布的统计特性：均值、方差、偏度、峰度
        return (
            _beta_mean,
            _beta_variance,
            _beta_skewness,
            _beta_kurtosis_excess)

    # 根据数据的初步估计值开始拟合 Beta 分布的参数
    def _fitstart(self, data):
        # 如果数据是被审查的数据类型，则将其解审查
        if isinstance(data, CensoredData):
            data = data._uncensor()

        # 计算数据的偏度和峰度
        g1 = _skew(data)
        g2 = _kurtosis(data)

        # 定义用于优化求解的函数
        def func(x):
            a, b = x
            sk = 2 * (b - a) * np.sqrt(a + b + 1) / (a + b + 2) / np.sqrt(a * b)
            ku = a**3 - a**2 * (2 * b - 1) + b**2 * (b + 1) - 2 * a * b * (b + 2)
            ku /= a * b * (a + b + 2) * (a + b + 3)
            ku *= 6
            # 返回偏度和峰度与观察值的差异
            return [sk - g1, ku - g2]

        # 使用优化方法 fsolve 求解参数 a 和 b
        a, b = optimize.fsolve(func, (1.0, 1.0))
        # 调用父类的 _fitstart 方法，并传入计算得到的参数值
        return super()._fitstart(data, args=(a, b))

    # 使用装饰器扩展父类的方法，并添加文档字符串的备注信息
    @_call_super_mom
    @extend_notes_in_docstring(rv_continuous, notes="""\
        在特殊情况下，当 `method="MLE"` 且同时给出 `floc` 和 `fscale` 时，
        如果数据中任何值 `x` 不满足 `floc < x < floc + fscale`，则会引发 `ValueError` 异常。\n\n""")
    # 定义一个私有方法 `_entropy`，计算两个参数 a 和 b 的熵值

        # 定义一个内部函数 `regular(a, b)`，计算正常情况下的熵值
        def regular(a, b):
            return (sc.betaln(a, b) - (a - 1) * sc.psi(a) -
                    (b - 1) * sc.psi(b) + (a + b - 2) * sc.psi(a + b))

        # 定义一个内部函数 `asymptotic_ab_large(a, b)`，在 a 和 b 都较大时，使用渐近公式计算熵值
        def asymptotic_ab_large(a, b):
            sum_ab = a + b
            log_term = 0.5 * (
                np.log(2*np.pi) + np.log(a) + np.log(b) - 3*np.log(sum_ab) + 1
            )
            t1 = 110/sum_ab + 20*sum_ab**-2.0 + sum_ab**-3.0 - 2*sum_ab**-4.0
            t2 = -50/a - 10*a**-2.0 - a**-3.0 + a**-4.0
            t3 = -50/b - 10*b**-2.0 - b**-3.0 + b**-4.0
            return log_term + (t1 + t2 + t3) / 120

        # 定义一个内部函数 `asymptotic_b_large(a, b)`，在 b 较大时使用渐近公式计算熵值
        def asymptotic_b_large(a, b):
            sum_ab = a + b
            t1 = sc.gammaln(a) - (a - 1) * sc.psi(a)
            t2 = (
                - 1/(2*b) + 1/(12*b) - b**-2.0/12 - b**-3.0/120 + b**-4.0/120
                + b**-5.0/252 - b**-6.0/252 + 1/sum_ab - 1/(12*sum_ab)
                + sum_ab**-2.0/6 + sum_ab**-3.0/120 - sum_ab**-4.0/60
                - sum_ab**-5.0/252 + sum_ab**-6.0/126
            )
            log_term = sum_ab*np.log1p(a/b) + np.log(b) - 2*np.log(sum_ab)
            return t1 + t2 + log_term

        # 定义一个内部函数 `threshold_large(v)`，计算阈值大于 1.0 时返回的数值
        def threshold_large(v):
            if v == 1.0:
                return 1000

            j = np.log10(v)
            digits = int(j)
            d = int(v / 10 ** digits) + 2
            return d*10**(7 + j)

        # 根据不同条件返回不同的熵值计算方法
        if a >= 4.96e6 and b >= 4.96e6:
            return asymptotic_ab_large(a, b)
        elif a <= 4.9e6 and b - a >= 1e6 and b >= threshold_large(a):
            return asymptotic_b_large(a, b)
        elif b <= 4.9e6 and a - b >= 1e6 and a >= threshold_large(b):
            return asymptotic_b_large(b, a)
        else:
            return regular(a, b)
beta = beta_gen(a=0.0, b=1.0, name='beta')

class betaprime_gen(rv_continuous):
    r"""A beta prime continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `betaprime` is:

    .. math::

        f(x, a, b) = \frac{x^{a-1} (1+x)^{-a-b}}{\beta(a, b)}

    for :math:`x >= 0`, :math:`a > 0`, :math:`b > 0`, where
    :math:`\beta(a, b)` is the beta function (see `scipy.special.beta`).

    `betaprime` takes ``a`` and ``b`` as shape parameters.

    The distribution is related to the `beta` distribution as follows:
    If :math:`X` follows a beta distribution with parameters :math:`a, b`,
    then :math:`Y = X/(1-X)` has a beta prime distribution with
    parameters :math:`a, b` ([1]_).

    The beta prime distribution is a reparametrized version of the
    F distribution.  The beta prime distribution with shape parameters
    ``a`` and ``b`` and ``scale = s`` is equivalent to the F distribution
    with parameters ``d1 = 2*a``, ``d2 = 2*b`` and ``scale = (a/b)*s``.
    For example,

    >>> from scipy.stats import betaprime, f
    >>> x = [1, 2, 5, 10]
    >>> a = 12
    >>> b = 5
    >>> betaprime.pdf(x, a, b, scale=2)
    array([0.00541179, 0.08331299, 0.14669185, 0.03150079])
    >>> f.pdf(x, 2*a, 2*b, scale=(a/b)*2)
    array([0.00541179, 0.08331299, 0.14669185, 0.03150079])

    %(after_notes)s

    References
    ----------
    .. [1] Beta prime distribution, Wikipedia,
           https://en.wikipedia.org/wiki/Beta_prime_distribution

    %(example)s

    """
    _support_mask = rv_continuous._open_support_mask

    def _shape_info(self):
        # 定义参数 a 和 b 的形状信息
        ia = _ShapeInfo("a", False, (0, np.inf), (False, False))
        ib = _ShapeInfo("b", False, (0, np.inf), (False, False))
        return [ia, ib]

    def _rvs(self, a, b, size=None, random_state=None):
        # 生成 Beta Prime 分布的随机变量
        u1 = gamma.rvs(a, size=size, random_state=random_state)
        u2 = gamma.rvs(b, size=size, random_state=random_state)
        return u1 / u2

    def _pdf(self, x, a, b):
        # 返回概率密度函数的值，即 x**(a-1) * (1+x)**(-a-b) / beta(a, b)
        return np.exp(self._logpdf(x, a, b))

    def _logpdf(self, x, a, b):
        # 返回概率密度函数的对数值
        return sc.xlogy(a - 1.0, x) - sc.xlog1py(a + b, x) - sc.betaln(a, b)

    def _cdf(self, x, a, b):
        # 注意：如果 x 较大，则 x/(1+x) 接近 1，此时使用不完全贝塔函数的关系来计算累积分布函数
        # 如果 x > 1，则使用特定的计算方式；否则使用关系 f2
        return _lazywhere(
            x > 1, [x, a, b],
            lambda x_, a_, b_: beta._sf(1/(1+x_), b_, a_),
            f2=lambda x_, a_, b_: beta._cdf(x_/(1+x_), a_, b_))
    # 定义一个私有方法 `_sf`，用于计算特定条件下的 beta 分布的生存函数值。
    def _sf(self, x, a, b):
        return _lazywhere(
            x > 1, [x, a, b],
            lambda x_, a_, b_: beta._cdf(1/(1+x_), b_, a_),  # 如果 x > 1，使用生存函数计算
            f2=lambda x_, a_, b_: beta._sf(x_/(1+x_), a_, b_)  # 否则，使用补生存函数计算
        )

    # 定义一个私有方法 `_ppf`，用于计算 beta 分布的百分位点。
    def _ppf(self, p, a, b):
        p, a, b = np.broadcast_arrays(p, a, b)
        # 默认情况下，通过求解 p = beta._cdf(x/(1+x), a, b) 来计算百分位点。
        # 这意味着 x = r/(1-r)，其中 r = beta._ppf(p, a, b)。如果 r 接近于 1，可能会出现数值问题。
        # 在这种情况下，使用补生存函数的另一种表达式 p = beta._sf(1/(1+x), b, a) 来求解。
        r = stats.beta._ppf(p, a, b)
        with np.errstate(divide='ignore'):
            out = r / (1 - r)
        i = (r > 0.9999)
        out[i] = 1/stats.beta._isf(p[i], b[i], a[i]) - 1
        return out

    # 定义一个私有方法 `_munp`，用于计算 beta 分布的 n 阶原点矩。
    def _munp(self, n, a, b):
        return _lazywhere(
            b > n, (a, b),
            lambda a, b: np.prod([(a+i-1)/(b-i) for i in range(1, n+1)], axis=0),
            fillvalue=np.inf)
# 创建一个 betaprime_gen 实例，设置参数 a=0.0 和 name='betaprime'
betaprime = betaprime_gen(a=0.0, name='betaprime')

# 定义一个继承自 rv_continuous 的类 bradford_gen，表示 Bradford 连续随机变量
class bradford_gen(rv_continuous):
    r"""A Bradford continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `bradford` is:

    .. math::

        f(x, c) = \frac{c}{\log(1+c) (1+cx)}

    for :math:`0 <= x <= 1` and :math:`c > 0`.

    `bradford` takes ``c`` as a shape parameter for :math:`c`.

    %(after_notes)s

    %(example)s

    """
    
    # 定义一个方法 _shape_info，返回一个 _ShapeInfo 对象，表示参数 'c' 的信息
    def _shape_info(self):
        return [_ShapeInfo("c", False, (0, np.inf), (False, False))]
    
    # 定义一个方法 _pdf，表示 Bradford 分布的概率密度函数
    def _pdf(self, x, c):
        # bradford.pdf(x, c) = c / (k * (1+c*x)), 其中 k = log(1+c)
        return c / (c * x + 1.0) / sc.log1p(c)
    
    # 定义一个方法 _cdf，表示 Bradford 分布的累积分布函数
    def _cdf(self, x, c):
        # 返回 Bradford 分布的累积分布函数
        return sc.log1p(c * x) / sc.log1p(c)
    
    # 定义一个方法 _ppf，表示 Bradford 分布的分位点函数
    def _ppf(self, q, c):
        # 返回 Bradford 分布的分位点函数
        return sc.expm1(q * sc.log1p(c)) / c
    
    # 定义一个方法 _stats，表示 Bradford 分布的统计信息
    def _stats(self, c, moments='mv'):
        # 计算 Bradford 分布的均值、方差、偏度和峰度
        k = np.log(1.0 + c)
        mu = (c - k) / (c * k)
        mu2 = ((c + 2.0) * k - 2.0 * c) / (2 * c * k * k)
        g1 = None
        g2 = None
        if 's' in moments:
            g1 = np.sqrt(2) * (12 * c * c - 9 * c * k * (c + 2) + 2 * k * k * (c * (c + 3) + 3))
            g1 /= np.sqrt(c * (c * (k - 2) + 2 * k) * (3 * c * (k - 2) + 6 * k))
        if 'k' in moments:
            g2 = (c**3 * (k - 3) * (k * (3 * k - 16) + 24) + 12 * k * c * c * (k - 4) * (k - 3) +
                  6 * c * k * k * (3 * k - 14) + 12 * k**3)
            g2 /= 3 * c * (c * (k - 2) + 2 * k)**2
        return mu, mu2, g1, g2
    
    # 定义一个方法 _entropy，表示 Bradford 分布的熵
    def _entropy(self, c):
        # 计算 Bradford 分布的熵
        k = np.log(1 + c)
        return k / 2.0 - np.log(c / k)

# 创建一个 bradford_gen 实例，设置参数 a=0.0, b=1.0 和 name='bradford'
bradford = bradford_gen(a=0.0, b=1.0, name='bradford')

# 定义一个继承自 rv_continuous 的类 burr_gen，表示 Burr (Type III) 连续随机变量
class burr_gen(rv_continuous):
    r"""A Burr (Type III) continuous random variable.

    %(before_notes)s

    See Also
    --------
    fisk : a special case of either `burr` or `burr12` with ``d=1``
    burr12 : Burr Type XII distribution
    mielke : Mielke Beta-Kappa / Dagum distribution

    Notes
    -----
    The probability density function for `burr` is:

    .. math::

        f(x; c, d) = c d \frac{x^{-c - 1}}
                              {{(1 + x^{-c})}^{d + 1}}

    for :math:`x >= 0` and :math:`c, d > 0`.

    `burr` takes ``c`` and ``d`` as shape parameters for :math:`c` and
    :math:`d`.

    This is the PDF corresponding to the third CDF given in Burr's list;
    specifically, it is equation (11) in Burr's paper [1]_. The distribution
    is also commonly referred to as the Dagum distribution [2]_. If the
    parameter :math:`c < 1` then the mean of the distribution does not
    exist and if :math:`c < 2` the variance does not exist [2]_.
    The PDF is finite at the left endpoint :math:`x = 0` if :math:`c * d >= 1`.

    %(after_notes)s

    References
    ----------
    .. [1] Burr, I. W. "Cumulative frequency functions", Annals of
       Mathematical Statistics, 13(2), pp 215-232 (1942).
    .. [2] https://en.wikipedia.org/wiki/Dagum_distribution
    .. [3] Kleiber, Christian. "A guide to the Dagum distributions."
       Modeling Income Distributions and Lorenz Curves  pp 97-117 (2008).

    %(example)s

    """
    # 定义一个私有方法 _shape_info，返回一个包含参数形状信息的列表
    def _shape_info(self):
        # 定义参数 c 的信息，允许的取值范围是 (0, 正无穷)，无需验证，不包括边界
        ic = _ShapeInfo("c", False, (0, np.inf), (False, False))
        # 定义参数 d 的信息，允许的取值范围是 (0, 正无穷)，无需验证，不包括边界
        id = _ShapeInfo("d", False, (0, np.inf), (False, False))
        # 返回参数信息列表
        return [ic, id]

    # 定义概率密度函数 _pdf，计算 Burr 分布的概率密度函数值
    def _pdf(self, x, c, d):
        # 当 x 为 0 时，使用 lambda 函数计算概率密度函数值
        output = _lazywhere(
            x == 0, [x, c, d],
            lambda x_, c_, d_: c_ * d_ * (x_**(c_*d_-1)) / (1 + x_**c_),
            # 当 x 不为 0 时，使用 lambda 函数计算概率密度函数值
            f2=lambda x_, c_, d_: (c_ * d_ * (x_ ** (-c_ - 1.0)) /
                                   ((1 + x_ ** (-c_)) ** (d_ + 1.0))))
        # 如果输出是标量，则返回其值，否则返回输出数组
        if output.ndim == 0:
            return output[()]
        return output

    # 定义对数概率密度函数 _logpdf，计算 Burr 分布的对数概率密度函数值
    def _logpdf(self, x, c, d):
        # 当 x 为 0 时，使用 lambda 函数计算对数概率密度函数值
        output = _lazywhere(
            x == 0, [x, c, d],
            lambda x_, c_, d_: (np.log(c_) + np.log(d_) + sc.xlogy(c_*d_ - 1, x_)
                                - (d_+1) * sc.log1p(x_**(c_))),
            # 当 x 不为 0 时，使用 lambda 函数计算对数概率密度函数值
            f2=lambda x_, c_, d_: (np.log(c_) + np.log(d_)
                                   + sc.xlogy(-c_ - 1, x_)
                                   - sc.xlog1py(d_+1, x_**(-c_))))
        # 如果输出是标量，则返回其值，否则返回输出数组
        if output.ndim == 0:
            return output[()]
        return output

    # 定义累积分布函数 _cdf，计算 Burr 分布的累积分布函数值
    def _cdf(self, x, c, d):
        return (1 + x**(-c))**(-d)

    # 定义对数累积分布函数 _logcdf，计算 Burr 分布的对数累积分布函数值
    def _logcdf(self, x, c, d):
        return sc.log1p(x**(-c)) * (-d)

    # 定义生存函数 _sf，计算 Burr 分布的生存函数值
    def _sf(self, x, c, d):
        return np.exp(self._logsf(x, c, d))

    # 定义对数生存函数 _logsf，计算 Burr 分布的对数生存函数值
    def _logsf(self, x, c, d):
        return np.log1p(- (1 + x**(-c))**(-d))

    # 定义百分点函数 _ppf，计算 Burr 分布的百分点函数值
    def _ppf(self, q, c, d):
        return (q**(-1.0/d) - 1)**(-1.0/c)

    # 定义逆生存函数 _isf，计算 Burr 分布的逆生存函数值
    def _isf(self, q, c, d):
        # 计算参数 q 的变换值
        _q = sc.xlog1py(-1.0 / d, -q)
        # 计算逆生存函数值
        return sc.expm1(_q) ** (-1.0 / c)

    # 定义统计量函数 _stats，计算 Burr 分布的前四个统计量
    def _stats(self, c, d):
        # 计算 nc 数组，包含 1 到 4 的值除以 c
        nc = np.arange(1, 5).reshape(4,1) / c
        # 计算原始矩 e1 到 e4
        e1, e2, e3, e4 = sc.beta(d + nc, 1. - nc) * d
        # 计算均值 mu，当 c 大于 1 时有效，否则为 NaN
        mu = np.where(c > 1.0, e1, np.nan)
        # 计算二阶中心矩 mu2，当 c 大于 2 时有效，否则为 NaN
        mu2_if_c = e2 - mu**2
        mu2 = np.where(c > 2.0, mu2_if_c, np.nan)
        # 计算偏度 g1，当 c 大于 3 时有效，否则为 NaN
        g1 = _lazywhere(
            c > 3.0,
            (c, e1, e2, e3, mu2_if_c),
            lambda c, e1, e2, e3, mu2_if_c: ((e3 - 3*e2*e1 + 2*e1**3)
                                             / np.sqrt((mu2_if_c)**3)),
            fillvalue=np.nan)
        # 计算峰度 g2，当 c 大于 4 时有效，否则为 NaN
        g2 = _lazywhere(
            c > 4.0,
            (c, e1, e2, e3, e4, mu2_if_c),
            lambda c, e1, e2, e3, e4, mu2_if_c: (
                ((e4 - 4*e3*e1 + 6*e2*e1**2 - 3*e1**4) / mu2_if_c**2) - 3),
            fillvalue=np.nan)
        # 如果 c 是标量，则返回统计量的标量值，否则返回数组
        if np.ndim(c) == 0:
            return mu.item(), mu2.item(), g1.item(), g2.item()
        return mu, mu2, g1, g2
    # 定义内部函数 _munp，用于计算特定参数下的数学函数值
    def _munp(self, n, c, d):
        
        # 定义内部函数 __munp，计算给定参数下的特定数学公式
        def __munp(n, c, d):
            # 计算 nc 的值
            nc = 1. * n / c
            # 使用 scipy 的 beta 函数计算并返回结果
            return d * sc.beta(1.0 - nc, d + nc)
        
        # 将输入的 n, c, d 转换为 NumPy 数组
        n, c, d = np.asarray(n), np.asarray(c), np.asarray(d)
        
        # 使用 _lazywhere 函数根据条件执行 __munp 函数，如果条件不满足返回 NaN
        return _lazywhere((c > n) & (n == n) & (d == d), (c, d, n),
                          lambda c, d, n: __munp(n, c, d),
                          np.nan)
burr = burr_gen(a=0.0, name='burr')

class burr12_gen(rv_continuous):
    r"""A Burr (Type XII) continuous random variable.

    %(before_notes)s

    See Also
    --------
    fisk : a special case of either `burr` or `burr12` with ``d=1``
    burr : Burr Type III distribution

    Notes
    -----
    The probability density function for `burr12` is:

    .. math::

        f(x; c, d) = c d \frac{x^{c-1}}
                              {(1 + x^c)^{d + 1}}

    for :math:`x >= 0` and :math:`c, d > 0`.

    `burr12` takes ``c`` and ``d`` as shape parameters for :math:`c`
    and :math:`d`.

    This is the PDF corresponding to the twelfth CDF given in Burr's list;
    specifically, it is equation (20) in Burr's paper [1]_.

    %(after_notes)s

    The Burr type 12 distribution is also sometimes referred to as
    the Singh-Maddala distribution from NIST [2]_.

    References
    ----------
    .. [1] Burr, I. W. "Cumulative frequency functions", Annals of
       Mathematical Statistics, 13(2), pp 215-232 (1942).

    .. [2] https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/b12pdf.htm

    .. [3] "Burr distribution",
       https://en.wikipedia.org/wiki/Burr_distribution

    %(example)s

    """
    # 定义用于描述分布形状参数的方法
    def _shape_info(self):
        ic = _ShapeInfo("c", False, (0, np.inf), (False, False))
        id = _ShapeInfo("d", False, (0, np.inf), (False, False))
        return [ic, id]

    # 定义概率密度函数 PDF
    def _pdf(self, x, c, d):
        # 返回 Burr12 分布的概率密度函数值
        return np.exp(self._logpdf(x, c, d))

    # 定义对数概率密度函数 log(PDF)
    def _logpdf(self, x, c, d):
        return np.log(c) + np.log(d) + sc.xlogy(c - 1, x) + sc.xlog1py(-d-1, x**c)

    # 定义累积分布函数 CDF
    def _cdf(self, x, c, d):
        return -sc.expm1(self._logsf(x, c, d))

    # 定义对数累积分布函数 log(CDF)
    def _logcdf(self, x, c, d):
        return sc.log1p(-(1 + x**c)**(-d))

    # 定义生存函数 SF (1 - CDF)
    def _sf(self, x, c, d):
        return np.exp(self._logsf(x, c, d))

    # 定义对数生存函数 log(SF)
    def _logsf(self, x, c, d):
        return sc.xlog1py(-d, x**c)

    # 定义反函数 PPF (Percent Point Function, 逆 CDF)
    def _ppf(self, q, c, d):
        # 下面的实现更好地处理较小的 q 值
        return sc.expm1(-1/d * sc.log1p(-q))**(1/c)

    # 定义逆生存函数 ISF (Inverse Survival Function)
    def _isf(self, p, c, d):
        return sc.expm1(-1/d * np.log(p))**(1/c)

    # 定义 n 阶原点矩 MPF (Mean of Power Function)
    def _munp(self, n, c, d):
        def moment_if_exists(n, c, d):
            nc = 1. * n / c
            return d * sc.beta(1.0 + nc, d - nc)

        return _lazywhere(c * d > n, (n, c, d), moment_if_exists,
                          fillvalue=np.nan)

burr12 = burr12_gen(a=0.0, name='burr12')


class fisk_gen(burr_gen):
    r"""A Fisk continuous random variable.

    The Fisk distribution is also known as the log-logistic distribution.

    %(before_notes)s

    See Also
    --------
    burr

    Notes
    -----
    The probability density function for `fisk` is:

    .. math::

        f(x, c) = \frac{c x^{c-1}}
                       {(1 + x^c)^2}

    for :math:`x >= 0` and :math:`c > 0`.

    """
    """
    fisk 分布的概率密度函数和相关方法的实现。

    %(after_notes)s

    %(example)s

    """
    # 返回一个 _ShapeInfo 对象的列表，描述了分布的参数形状
    def _shape_info(self):
        return [_ShapeInfo("c", False, (0, np.inf), (False, False))]

    # 计算概率密度函数 fisk.pdf(x, c) = c * x**(-c-1) * (1 + x**(-c))**(-2)
    def _pdf(self, x, c):
        return burr._pdf(x, c, 1.0)

    # 计算累积分布函数
    def _cdf(self, x, c):
        return burr._cdf(x, c, 1.0)

    # 计算生存函数 (1 - CDF)
    def _sf(self, x, c):
        return burr._sf(x, c, 1.0)

    # 计算对数概率密度函数
    def _logpdf(self, x, c):
        # 对数概率密度函数 fisk.pdf(x, c) = c * x**(-c-1) * (1 + x**(-c))**(-2)
        return burr._logpdf(x, c, 1.0)

    # 计算对数累积分布函数
    def _logcdf(self, x, c):
        return burr._logcdf(x, c, 1.0)

    # 计算对数生存函数 (log(1 - CDF))
    def _logsf(self, x, c):
        return burr._logsf(x, c, 1.0)

    # 计算百分位点函数 (CDF 的逆函数)
    def _ppf(self, x, c):
        return burr._ppf(x, c, 1.0)

    # 计算逆生存函数 (SF 的逆函数)
    def _isf(self, q, c):
        return burr._isf(q, c, 1.0)

    # 计算非中心矩
    def _munp(self, n, c):
        return burr._munp(n, c, 1.0)

    # 计算分布的统计信息，包括均值、方差等
    def _stats(self, c):
        return burr._stats(c, 1.0)

    # 计算分布的熵
    def _entropy(self, c):
        return 2 - np.log(c)
# 使用 fisk_gen 函数生成一个分布对象 fisk，参数 a 设置为 0.0，name 设置为 'fisk'
fisk = fisk_gen(a=0.0, name='fisk')

# 定义一个名为 cauchy_gen 的类，继承自 rv_continuous 类，表示一个 Cauchy 分布的连续随机变量
class cauchy_gen(rv_continuous):
    r"""A Cauchy continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `cauchy` is

    .. math::

        f(x) = \frac{1}{\pi (1 + x^2)}

    for a real number :math:`x`.

    %(after_notes)s

    %(example)s

    """
    
    # 返回一个空列表，用于表示没有额外的形状参数信息
    def _shape_info(self):
        return []

    # 定义 Cauchy 分布的概率密度函数 _pdf(x)，公式为 1 / (pi * (1 + x**2))
    def _pdf(self, x):
        return 1.0 / np.pi / (1.0 + x*x)

    # 定义 Cauchy 分布的累积分布函数 _cdf(x)
    def _cdf(self, x):
        return 0.5 + 1.0 / np.pi * np.arctan(x)

    # 定义 Cauchy 分布的百分位点函数 _ppf(q)
    def _ppf(self, q):
        return np.tan(np.pi * q - np.pi / 2.0)

    # 定义 Cauchy 分布的生存函数 _sf(x)
    def _sf(self, x):
        return 0.5 - 1.0 / np.pi * np.arctan(x)

    # 定义 Cauchy 分布的逆百分位点函数 _isf(q)
    def _isf(self, q):
        return np.tan(np.pi / 2.0 - np.pi * q)

    # 定义 Cauchy 分布的统计特性函数 _stats()
    def _stats(self):
        return np.nan, np.nan, np.nan, np.nan

    # 定义 Cauchy 分布的熵函数 _entropy()
    def _entropy(self):
        return np.log(4 * np.pi)

    # 定义 Cauchy 分布的拟合起始值函数 _fitstart(data, args=None)
    def _fitstart(self, data, args=None):
        # 使用四分位数而不是矩来初始化最大似然猜测
        if isinstance(data, CensoredData):  # 如果数据是被截断的数据对象，则解除截断
            data = data._uncensor()
        # 计算数据的第 25、50、75 百分位数
        p25, p50, p75 = np.percentile(data, [25, 50, 75])
        # 返回拟合的起始值，中位数和四分位数差的一半
        return p50, (p75 - p25) / 2

# 创建一个 Cauchy 分布对象 cauchy，使用 cauchy_gen 类生成
cauchy = cauchy_gen(name='cauchy')

# 定义一个名为 chi_gen 的类，继承自 rv_continuous 类，表示一个 chi 分布的连续随机变量
class chi_gen(rv_continuous):
    r"""A chi continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `chi` is:

    .. math::

        f(x, k) = \frac{1}{2^{k/2-1} \Gamma \left( k/2 \right)}
                   x^{k-1} \exp \left( -x^2/2 \right)

    for :math:`x >= 0` and :math:`k > 0` (degrees of freedom, denoted ``df``
    in the implementation). :math:`\Gamma` is the gamma function
    (`scipy.special.gamma`).

    Special cases of `chi` are:

        - ``chi(1, loc, scale)`` is equivalent to `halfnorm`
        - ``chi(2, 0, scale)`` is equivalent to `rayleigh`
        - ``chi(3, 0, scale)`` is equivalent to `maxwell`

    `chi` takes ``df`` as a shape parameter.

    %(after_notes)s

    %(example)s

    """
    
    # 返回一个列表，其中包含形状参数信息 _ShapeInfo("df", False, (0, np.inf), (False, False))
    def _shape_info(self):
        return [_ShapeInfo("df", False, (0, np.inf), (False, False))]

    # 从 chi2 分布中生成随机变量，然后对其取平方根作为结果
    def _rvs(self, df, size=None, random_state=None):
        return np.sqrt(chi2.rvs(df, size=size, random_state=random_state))

    # 定义 chi 分布的概率密度函数 _pdf(x, df)
    def _pdf(self, x, df):
        return np.exp(self._logpdf(x, df))

    # 定义 chi 分布的对数概率密度函数 _logpdf(x, df)
    def _logpdf(self, x, df):
        l = np.log(2) - 0.5 * np.log(2) * df - sc.gammaln(0.5 * df)
        return l + sc.xlogy(df - 1., x) - 0.5 * x**2

    # 定义 chi 分布的累积分布函数 _cdf(x, df)
    def _cdf(self, x, df):
        return sc.gammainc(0.5 * df, 0.5 * x**2)

    # 定义 chi 分布的生存函数 _sf(x, df)
    def _sf(self, x, df):
        return sc.gammaincc(0.5 * df, 0.5 * x**2)

    # 定义 chi 分布的逆累积分布函数 _ppf(q, df)
    def _ppf(self, q, df):
        return np.sqrt(2 * sc.gammaincinv(0.5 * df, q))

    # 定义 chi 分布的逆生存函数 _isf(q, df)
    def _isf(self, q, df):
        return np.sqrt(2 * sc.gammainccinv(0.5 * df, q))
    # 计算给定自由度 df 的统计量：均值 mu、方差 mu2、偏度 g1 和峰度 g2
    def _stats(self, df):
        # 使用半阶乘 poch(df/2, 1/2)，其等于 gamma(df/2 + 1/2) / gamma(df/2)
        mu = np.sqrt(2) * sc.poch(0.5 * df, 0.5)
        # 计算方差 mu2
        mu2 = df - mu*mu
        # 计算偏度 g1
        g1 = (2*mu**3.0 + mu*(1-2*df))/np.asarray(np.power(mu2, 1.5))
        # 计算峰度 g2
        g2 = 2*df*(1.0-df)-6*mu**4 + 4*mu**2 * (2*df-1)
        g2 /= np.asarray(mu2**2.0)
        # 返回均值 mu、方差 mu2、偏度 g1 和峰度 g2
        return mu, mu2, g1, g2

    # 计算给定自由度 df 的熵值
    def _entropy(self, df):

        # 定义常规公式计算熵
        def regular_formula(df):
            return (sc.gammaln(.5 * df)
                    + 0.5 * (df - np.log(2) - (df - 1) * sc.digamma(0.5 * df)))

        # 定义渐近公式计算熵
        def asymptotic_formula(df):
            return (0.5 + np.log(np.pi)/2 - (df**-1)/6 - (df**-2)/6
                    - 4/45*(df**-3) + (df**-4)/15)

        # 根据条件选择使用常规或渐近公式计算熵
        return _lazywhere(df < 3e2, (df, ), regular_formula,
                          f2=asymptotic_formula)
# 创建一个名称为 chi 的连续随机变量生成器对象，并设置参数 a=0.0, name='chi'
chi = chi_gen(a=0.0, name='chi')

# 定义一个继承自 rv_continuous 的类 chi2_gen，表示 chi 平方分布的连续随机变量
class chi2_gen(rv_continuous):
    r"""A chi-squared continuous random variable.

    For the noncentral chi-square distribution, see `ncx2`.

    %(before_notes)s

    See Also
    --------
    ncx2

    Notes
    -----
    The probability density function for `chi2` is:

    .. math::

        f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                   x^{k/2-1} \exp \left( -x/2 \right)

    for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
    in the implementation).

    `chi2` takes ``df`` as a shape parameter.

    The chi-squared distribution is a special case of the gamma
    distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
    ``scale = 2``.

    %(after_notes)s

    %(example)s

    """

    # 返回一个描述形状参数的 _shape_info 对象，这里表示参数 df 为非必需、非负且大于0
    def _shape_info(self):
        return [_ShapeInfo("df", False, (0, np.inf), (False, False))]

    # 返回一个随机变量的方法，使用给定的 df 参数生成卡方分布样本
    def _rvs(self, df, size=None, random_state=None):
        return random_state.chisquare(df, size)

    # 返回概率密度函数的方法，通过对数形式的 PDF 计算来简化计算
    def _pdf(self, x, df):
        # chi2.pdf(x, df) = 1 / (2*gamma(df/2)) * (x/2)**(df/2-1) * exp(-x/2)
        return np.exp(self._logpdf(x, df))

    # 返回对数概率密度函数的方法，用于更精确地计算 PDF
    def _logpdf(self, x, df):
        return sc.xlogy(df/2.-1, x) - x/2. - sc.gammaln(df/2.) - (np.log(2)*df)/2.

    # 返回累积分布函数的方法，使用 SciPy 中的 chdtr 函数计算卡方分布的 CDF
    def _cdf(self, x, df):
        return sc.chdtr(df, x)

    # 返回生存函数的方法，使用 SciPy 中的 chdtrc 函数计算卡方分布的 SF
    def _sf(self, x, df):
        return sc.chdtrc(df, x)

    # 返回逆生存函数的方法，使用 SciPy 中的 chdtri 函数计算卡方分布的 ISF
    def _isf(self, p, df):
        return sc.chdtri(df, p)

    # 返回百分点函数的方法，使用 SciPy 中的 gammaincinv 函数计算卡方分布的 PPF
    def _ppf(self, p, df):
        return 2*sc.gammaincinv(df/2, p)

    # 返回统计信息的方法，计算卡方分布的期望、方差和偏度
    def _stats(self, df):
        mu = df
        mu2 = 2*df
        g1 = 2*np.sqrt(2.0/df)
        g2 = 12.0/df
        return mu, mu2, g1, g2

    # 返回熵的方法，计算卡方分布的信息熵
    def _entropy(self, df):
        half_df = 0.5 * df

        # 定义一个常规的计算公式来计算熵
        def regular_formula(half_df):
            return (half_df + np.log(2) + sc.gammaln(half_df) +
                    (1 - half_df) * sc.psi(half_df))

        # 定义一个渐近计算公式来计算熵
        def asymptotic_formula(half_df):
            # 在上述公式中插入以下渐近展开：
            # ln(gamma(a)) ~ (a - 0.5) * ln(a) - a + 0.5 * ln(2 * pi) +
            #                 1/(12 * a) - 1/(360 * a**3)
            # psi(a) ~ ln(a) - 1/(2 * a) - 1/(3 * a**2) + 1/120 * a**4)
            c = np.log(2) + 0.5*(1 + np.log(2*np.pi))
            h = 0.5/half_df
            return (h*(-2/3 + h*(-1/3 + h*(-4/45 + h/7.5))) +
                    0.5*np.log(half_df) + c)

        return _lazywhere(half_df < 125, (half_df, ),
                          regular_formula,
                          f2=asymptotic_formula)


# 创建一个名称为 chi2 的连续随机变量生成器对象，并设置参数 a=0.0, name='chi2'
chi2 = chi2_gen(a=0.0, name='chi2')

# 定义一个继承自 rv_continuous 的类 cosine_gen，表示余弦分布的连续随机变量
class cosine_gen(rv_continuous):
    r"""A cosine continuous random variable.

    %(before_notes)s

    Notes
    -----
    The cosine distribution is an approximation to the normal distribution.
    The probability density function for `cosine` is:

    .. math::

        f(x) = \frac{1}{2\pi} (1+\cos(x))

    for :math:`-\pi \le x \le \pi`.

    %(after_notes)s

    %(example)s

    """
    # 返回一个空列表，表示没有形状信息
    def _shape_info(self):
        return []

    # 返回 x 的概率密度函数值，cosine.pdf(x) = 1/(2*pi) * (1+cos(x))
    def _pdf(self, x):
        return 1.0/2/np.pi*(1+np.cos(x))

    # 返回 x 的对数概率密度函数值
    def _logpdf(self, x):
        # 计算 x 的余弦值
        c = np.cos(x)
        # 当 c 不等于 -1 时，返回 np.log1p(c) - np.log(2*np.pi)，否则返回 -np.inf
        return _lazywhere(c != -1, (c,),
                          lambda c: np.log1p(c) - np.log(2*np.pi),
                          fillvalue=-np.inf)

    # 返回 x 的累积分布函数值
    def _cdf(self, x):
        return scu._cosine_cdf(x)

    # 返回 x 的生存函数值 (1 - CDF(x))
    def _sf(self, x):
        return scu._cosine_cdf(-x)

    # 返回累积分布函数的逆函数值，即分位数函数值
    def _ppf(self, p):
        return scu._cosine_invcdf(p)

    # 返回生存函数的逆函数值，即逆分位数函数值
    def _isf(self, p):
        return -scu._cosine_invcdf(p)

    # 返回分布的期望值和峰度
    def _stats(self):
        v = (np.pi * np.pi / 3.0) - 2.0
        k = -6.0 * (np.pi**4 - 90) / (5.0 * (np.pi * np.pi - 6)**2)
        return 0.0, v, 0.0, k

    # 返回分布的熵值
    def _entropy(self):
        return np.log(4*np.pi)-1.0
cosine = cosine_gen(a=-np.pi, b=np.pi, name='cosine')

class dgamma_gen(rv_continuous):
    r"""A double gamma continuous random variable.

    The double gamma distribution is also known as the reflected gamma
    distribution [1]_.

    %(before_notes)s

    Notes
    -----
    The probability density function for `dgamma` is:

    .. math::

        f(x, a) = \frac{1}{2\Gamma(a)} |x|^{a-1} \exp(-|x|)

    for a real number :math:`x` and :math:`a > 0`. :math:`\Gamma` is the
    gamma function (`scipy.special.gamma`).

    `dgamma` takes ``a`` as a shape parameter for :math:`a`.

    %(after_notes)s

    References
    ----------
    .. [1] Johnson, Kotz, and Balakrishnan, "Continuous Univariate
           Distributions, Volume 1", Second Edition, John Wiley and Sons
           (1994).

    %(example)s

    """
    # 定义一个方法来描述分布的形状参数信息
    def _shape_info(self):
        return [_ShapeInfo("a", False, (0, np.inf), (False, False))]

    # 定义一个方法来生成服从双Gamma分布的随机变量
    def _rvs(self, a, size=None, random_state=None):
        # 生成均匀分布的随机数
        u = random_state.uniform(size=size)
        # 生成Gamma分布的随机数
        gm = gamma.rvs(a, size=size, random_state=random_state)
        # 返回根据均匀分布随机数选择Gamma分布的结果
        return gm * np.where(u >= 0.5, 1, -1)

    # 定义一个方法来计算概率密度函数（PDF）
    def _pdf(self, x, a):
        # dgamma.pdf(x, a) = 1 / (2*gamma(a)) * abs(x)**(a-1) * exp(-abs(x))
        ax = abs(x)
        return 1.0/(2*sc.gamma(a))*ax**(a-1.0) * np.exp(-ax)

    # 定义一个方法来计算对数概率密度函数（log PDF）
    def _logpdf(self, x, a):
        ax = abs(x)
        return sc.xlogy(a - 1.0, ax) - ax - np.log(2) - sc.gammaln(a)

    # 定义一个方法来计算累积分布函数（CDF）
    def _cdf(self, x, a):
        return np.where(x > 0,
                        0.5 + 0.5*sc.gammainc(a, x),
                        0.5*sc.gammaincc(a, -x))

    # 定义一个方法来计算生存函数（SF）
    def _sf(self, x, a):
        return np.where(x > 0,
                        0.5*sc.gammaincc(a, x),
                        0.5 + 0.5*sc.gammainc(a, -x))

    # 定义一个方法来计算熵
    def _entropy(self, a):
        return stats.gamma._entropy(a) - np.log(0.5)

    # 定义一个方法来计算百分位点函数（PPF）
    def _ppf(self, q, a):
        return np.where(q > 0.5,
                        sc.gammaincinv(a, 2*q - 1),
                        -sc.gammainccinv(a, 2*q))

    # 定义一个方法来计算逆生存函数（ISF）
    def _isf(self, q, a):
        return np.where(q > 0.5,
                        -sc.gammaincinv(a, 2*q - 1),
                        sc.gammainccinv(a, 2*q))

    # 定义一个方法来计算统计特性
    def _stats(self, a):
        mu2 = a*(a+1.0)
        return 0.0, mu2, 0.0, (a+2.0)*(a+3.0)/mu2-3.0


# 创建一个双Gamma分布的实例
dgamma = dgamma_gen(name='dgamma')


class dweibull_gen(rv_continuous):
    r"""A double Weibull continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `dweibull` is given by

    .. math::

        f(x, c) = c / 2 |x|^{c-1} \exp(-|x|^c)

    for a real number :math:`x` and :math:`c > 0`.

    `dweibull` takes ``c`` as a shape parameter for :math:`c`.

    %(after_notes)s

    %(example)s

    """
    # 定义一个方法来描述分布的形状参数信息
    def _shape_info(self):
        return [_ShapeInfo("c", False, (0, np.inf), (False, False))]
    # 生成服从Weibull分布的随机变量，根据参数c和指定大小生成随机数
    def _rvs(self, c, size=None, random_state=None):
        # 生成指定大小的均匀分布随机数
        u = random_state.uniform(size=size)
        # 生成Weibull分布的随机变量
        w = weibull_min.rvs(c, size=size, random_state=random_state)
        return w * (np.where(u >= 0.5, 1, -1))

    # 计算Weibull分布的概率密度函数（PDF）
    def _pdf(self, x, c):
        # 计算Weibull分布的概率密度函数表达式
        ax = abs(x)
        Px = c / 2.0 * ax**(c-1.0) * np.exp(-ax**c)
        return Px

    # 计算Weibull分布的对数概率密度函数（logPDF）
    def _logpdf(self, x, c):
        ax = abs(x)
        # 计算Weibull分布的对数概率密度函数表达式
        return np.log(c) - np.log(2.0) + sc.xlogy(c - 1.0, ax) - ax**c

    # 计算Weibull分布的累积分布函数（CDF）
    def _cdf(self, x, c):
        # 计算Weibull分布的累积分布函数表达式
        Cx1 = 0.5 * np.exp(-abs(x)**c)
        return np.where(x > 0, 1 - Cx1, Cx1)

    # 计算Weibull分布的百分位点函数（PPF）
    def _ppf(self, q, c):
        # 计算Weibull分布的百分位点函数表达式
        fac = 2. * np.where(q <= 0.5, q, 1. - q)
        fac = np.power(-np.log(fac), 1.0 / c)
        return np.where(q > 0.5, fac, -fac)

    # 计算Weibull分布的生存函数（SF）
    def _sf(self, x, c):
        # 计算Weibull分布的生存函数表达式
        half_weibull_min_sf = 0.5 * stats.weibull_min._sf(np.abs(x), c)
        return np.where(x > 0, half_weibull_min_sf, 1 - half_weibull_min_sf)

    # 计算Weibull分布的逆累积分布函数（ISF）
    def _isf(self, q, c):
        # 计算Weibull分布的逆累积分布函数表达式
        double_q = 2. * np.where(q <= 0.5, q, 1. - q)
        weibull_min_isf = stats.weibull_min._isf(double_q, c)
        return np.where(q > 0.5, -weibull_min_isf, weibull_min_isf)

    # 计算Weibull分布的n阶非中心矩（non-central moments）
    def _munp(self, n, c):
        return (1 - (n % 2)) * sc.gamma(1.0 + 1.0 * n / c)

    # 计算Weibull分布的统计特征值（moments）
    # 因为我们知道所有奇数阶矩都为零，一次性返回它们
    # 从_stats函数返回None可以使公共的统计函数调用_munp
    # 因此总体上我们可以节省一两次gamma函数的计算。
    def _stats(self, c):
        return 0, None, 0, None

    # 计算Weibull分布的熵（entropy）
    def _entropy(self, c):
        # 计算Weibull分布的熵表达式
        h = stats.weibull_min._entropy(c) - np.log(0.5)
        return h
# 创建一个 Weibull 分布的生成器，命名为 dweibull
dweibull = dweibull_gen(name='dweibull')


# 定义一个指数分布的生成器，继承于 rv_continuous 类
class expon_gen(rv_continuous):
    r"""An exponential continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `expon` is:

    .. math::

        f(x) = \exp(-x)

    for :math:`x \ge 0`.

    %(after_notes)s

    A common parameterization for `expon` is in terms of the rate parameter
    ``lambda``, such that ``pdf = lambda * exp(-lambda * x)``. This
    parameterization corresponds to using ``scale = 1 / lambda``.

    The exponential distribution is a special case of the gamma
    distributions, with gamma shape parameter ``a = 1``.

    %(example)s

    """

    # 返回一个空列表，用于描述分布的形状信息
    def _shape_info(self):
        return []

    # 生成随机变量的方法，使用标准指数分布生成器
    def _rvs(self, size=None, random_state=None):
        return random_state.standard_exponential(size)

    # 概率密度函数，对应于指数分布的概率密度函数 f(x) = exp(-x)
    def _pdf(self, x):
        return np.exp(-x)

    # 对数概率密度函数，即 -x
    def _logpdf(self, x):
        return -x

    # 累积分布函数，使用 -expm1(-x) 来计算
    def _cdf(self, x):
        return -sc.expm1(-x)

    # 百分点函数的反函数，对应于 -log1p(-q)
    def _ppf(self, q):
        return -sc.log1p(-q)

    # 生存函数，即 exp(-x)
    def _sf(self, x):
        return np.exp(-x)

    # 对数生存函数，即 -x
    def _logsf(self, x):
        return -x

    # 生存函数的反函数，对应于 -log(q)
    def _isf(self, q):
        return -np.log(q)

    # 返回分布的统计特性，期望、方差、偏度和峰度
    def _stats(self):
        return 1.0, 1.0, 2.0, 6.0

    # 返回分布的熵，即 1.0
    def _entropy(self):
        return 1.0

    # 使用父类 rv_continuous 的方法进行调用，并替换文档字符串中的说明内容
    @ _call_super_mom
    @ replace_notes_in_docstring(rv_continuous, notes="""\
        When `method='MLE'`,
        this function uses explicit formulas for the maximum likelihood
        estimation of the exponential distribution parameters, so the
        `optimizer`, `loc` and `scale` keyword arguments are
        ignored.\n\n""")
    def fit(self, data, *args, **kwds):
        # 如果传入的参数个数大于 0，抛出参数过多的异常
        if len(args) > 0:
            raise TypeError("Too many arguments.")

        # 从关键字参数中取出 floc 和 fscale
        floc = kwds.pop('floc', None)
        fscale = kwds.pop('fscale', None)

        # 移除优化器相关的参数
        _remove_optimizer_parameters(kwds)

        # 如果同时指定了 floc 和 fscale，抛出所有参数已固定的异常
        if floc is not None and fscale is not None:
            raise ValueError("All parameters fixed. There is nothing to "
                             "optimize.")

        # 将数据转换为 ndarray 格式
        data = np.asarray(data)

        # 如果数据中包含非有限的值，抛出数据包含无限值的异常
        if not np.isfinite(data).all():
            raise ValueError("The data contains non-finite values.")

        # 计算数据的最小值
        data_min = data.min()

        # 如果未指定 floc，则使用数据的最小值作为位置参数的极大似然估计
        if floc is None:
            loc = data_min
        else:
            loc = floc
            # 如果数据的最小值小于指定的 loc 值，抛出拟合数据错误的异常
            if data_min < loc:
                raise FitDataError("expon", lower=floc, upper=np.inf)

        # 如果未指定 fscale，则使用数据的均值减去 loc 作为尺度参数的极大似然估计
        if fscale is None:
            scale = data.mean() - loc
        else:
            scale = fscale

        # 明确返回浮点数的结果
        return float(loc), float(scale)


# 创建一个指数分布的实例，参数化为 a=0.0，命名为 expon
expon = expon_gen(a=0.0, name='expon')
class exponnorm_gen(rv_continuous):
    r"""An exponentially modified Normal continuous random variable.

    Also known as the exponentially modified Gaussian distribution [1]_.

    %(before_notes)s

    Notes
    -----
    The probability density function for `exponnorm` is:

    .. math::

        f(x, K) = \frac{1}{2K} \exp\left(\frac{1}{2 K^2} - x / K \right)
                  \text{erfc}\left(-\frac{x - 1/K}{\sqrt{2}}\right)

    where :math:`x` is a real number and :math:`K > 0`.

    It can be thought of as the sum of a standard normal random variable
    and an independent exponentially distributed random variable with rate
    ``1/K``.

    %(after_notes)s

    An alternative parameterization of this distribution (for example, in
    the Wikipedia article [1]_) involves three parameters, :math:`\mu`,
    :math:`\lambda` and :math:`\sigma`.

    In the present parameterization this corresponds to having ``loc`` and
    ``scale`` equal to :math:`\mu` and :math:`\sigma`, respectively, and
    shape parameter :math:`K = 1/(\sigma\lambda)`.

    .. versionadded:: 0.16.0

    References
    ----------
    .. [1] Exponentially modified Gaussian distribution, Wikipedia,
           https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution

    %(example)s

    """
    def _shape_info(self):
        # 返回一个描述分布形状的列表，表示参数 K 是非必需的，其范围为大于 0 的实数
        return [_ShapeInfo("K", False, (0, np.inf), (False, False))]

    def _rvs(self, K, size=None, random_state=None):
        # 生成指定大小的服从 exponnorm 分布的随机变量
        expval = random_state.standard_exponential(size) * K
        gval = random_state.standard_normal(size)
        return expval + gval

    def _pdf(self, x, K):
        # 计算概率密度函数值，使用 _logpdf 函数的指数形式
        return np.exp(self._logpdf(x, K))

    def _logpdf(self, x, K):
        # 计算对数概率密度函数值
        invK = 1.0 / K
        exparg = invK * (0.5 * invK - x)
        return exparg + _norm_logcdf(x - invK) - np.log(K)

    def _cdf(self, x, K):
        # 计算累积分布函数值
        invK = 1.0 / K
        expval = invK * (0.5 * invK - x)
        logprod = expval + _norm_logcdf(x - invK)
        return _norm_cdf(x) - np.exp(logprod)

    def _sf(self, x, K):
        # 计算生存函数值 (1 - 累积分布函数)
        invK = 1.0 / K
        expval = invK * (0.5 * invK - x)
        logprod = expval + _norm_logcdf(x - invK)
        return _norm_cdf(-x) + np.exp(logprod)

    def _stats(self, K):
        # 计算分布的统计特性，包括均值、方差、偏度和峰度
        K2 = K * K
        opK2 = 1.0 + K2
        skw = 2 * K**3 * opK2**(-1.5)
        krt = 6.0 * K2 * K2 * opK2**(-2)
        return K, opK2, skw, krt


exponnorm = exponnorm_gen(name='exponnorm')


def _pow1pm1(x, y):
    """
    Compute (1 + x)**y - 1.

    Uses expm1 and xlog1py to avoid loss of precision when
    (1 + x)**y is close to 1.

    Note that the inverse of this function with respect to x is
    ``_pow1pm1(x, 1/y)``.  That is, if

        t = _pow1pm1(x, y)

    then

        x = _pow1pm1(t, 1/y)
    """
    return np.expm1(sc.xlog1py(y, x))


class exponweib_gen(rv_continuous):
    r"""An exponentiated Weibull continuous random variable.

    %(before_notes)s

    See Also
    --------
    weibull_min, numpy.random.Generator.weibull

    Notes
    -----
    """
    The probability density function for `exponweib` is:

    .. math::

        f(x, a, c) = a c [1-\exp(-x^c)]^{a-1} \exp(-x^c) x^{c-1}

    and its cumulative distribution function is:

    .. math::

        F(x, a, c) = [1-\exp(-x^c)]^a

    for :math:`x > 0`, :math:`a > 0`, :math:`c > 0`.

    `exponweib` takes :math:`a` and :math:`c` as shape parameters:

    * :math:`a` is the exponentiation parameter,
      with the special case :math:`a=1` corresponding to the
      (non-exponentiated) Weibull distribution `weibull_min`.
    * :math:`c` is the shape parameter of the non-exponentiated Weibull law.

    %(after_notes)s

    References
    ----------
    https://en.wikipedia.org/wiki/Exponentiated_Weibull_distribution

    %(example)s

    """
    # 定义 `_shape_info` 方法，返回参数 `a` 和 `c` 的信息
    def _shape_info(self):
        ia = _ShapeInfo("a", False, (0, np.inf), (False, False))
        ic = _ShapeInfo("c", False, (0, np.inf), (False, False))
        return [ia, ic]

    # 定义 `_pdf` 方法，计算概率密度函数 (PDF) `exponweib.pdf(x, a, c)`
    def _pdf(self, x, a, c):
        # exponweib.pdf(x, a, c) =
        #     a * c * (1-exp(-x**c))**(a-1) * exp(-x**c)*x**(c-1)
        return np.exp(self._logpdf(x, a, c))

    # 定义 `_logpdf` 方法，计算对数概率密度函数
    def _logpdf(self, x, a, c):
        negxc = -x**c
        exm1c = -sc.expm1(negxc)
        logp = (np.log(a) + np.log(c) + sc.xlogy(a - 1.0, exm1c) +
                negxc + sc.xlogy(c - 1.0, x))
        return logp

    # 定义 `_cdf` 方法，计算累积分布函数 (CDF) `exponweib.cdf(x, a, c)`
    def _cdf(self, x, a, c):
        exm1c = -sc.expm1(-x**c)
        return exm1c**a

    # 定义 `_ppf` 方法，计算百分位点函数 (PPF) `exponweib.ppf(q, a, c)`
    def _ppf(self, q, a, c):
        return (-sc.log1p(-q**(1.0/a)))**np.asarray(1.0/c)

    # 定义 `_sf` 方法，计算生存函数 (SF) `exponweib.sf(x, a, c)`
    def _sf(self, x, a, c):
        return -_pow1pm1(-np.exp(-x**c), a)

    # 定义 `_isf` 方法，计算逆生存函数 (ISF) `exponweib.isf(p, a, c)`
    def _isf(self, p, a, c):
        return (-np.log(-_pow1pm1(-p, 1/a)))**(1/c)
# 定义一个指数韦伯（exponweib）连续随机变量生成器，a=0.0 表示形状参数为 0.0，name='exponweib' 是变量名
exponweib = exponweib_gen(a=0.0, name='exponweib')

# 定义一个疲劳寿命（Birnbaum-Saunders）连续随机变量生成器类
class exponpow_gen(rv_continuous):
    r"""An exponential power continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `exponpow` is:

    .. math::

        f(x, b) = b x^{b-1} \exp(1 + x^b - \exp(x^b))

    for :math:`x \ge 0`, :math:`b > 0`.  Note that this is a different
    distribution from the exponential power distribution that is also known
    under the names "generalized normal" or "generalized Gaussian".

    `exponpow` takes ``b`` as a shape parameter for :math:`b`.

    %(after_notes)s

    References
    ----------
    http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Exponentialpower.pdf

    %(example)s

    """
    
    # 定义一个用于描述参数形状信息的方法
    def _shape_info(self):
        return [_ShapeInfo("b", False, (0, np.inf), (False, False))]
    
    # 定义概率密度函数（PDF），根据指数幂分布的公式计算
    def _pdf(self, x, b):
        # exponpow.pdf(x, b) = b * x**(b-1) * exp(1 + x**b - exp(x**b))
        return np.exp(self._logpdf(x, b))
    
    # 定义概率密度函数的对数（log PDF）计算方法
    def _logpdf(self, x, b):
        xb = x**b
        f = 1 + np.log(b) + sc.xlogy(b - 1.0, x) + xb - np.exp(xb)
        return f
    
    # 定义累积分布函数（CDF）的计算方法
    def _cdf(self, x, b):
        return -sc.expm1(-sc.expm1(x**b))
    
    # 定义生存函数（SF）的计算方法
    def _sf(self, x, b):
        return np.exp(-sc.expm1(x**b))
    
    # 定义逆累积分布函数（ISF）的计算方法
    def _isf(self, x, b):
        return (sc.log1p(-np.log(x)))**(1./b)
    
    # 定义百分位点函数（PPF）的计算方法
    def _ppf(self, q, b):
        return pow(sc.log1p(-sc.log1p(-q)), 1.0/b)

# 创建一个指数幂（exponpow）连续随机变量生成器，a=0.0 表示形状参数为 0.0，name='exponpow' 是变量名
exponpow = exponpow_gen(a=0.0, name='exponpow')

# 定义一个疲劳寿命（Birnbaum-Saunders）连续随机变量生成器类
class fatiguelife_gen(rv_continuous):
    r"""A fatigue-life (Birnbaum-Saunders) continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `fatiguelife` is:

    .. math::

        f(x, c) = \frac{x+1}{2c\sqrt{2\pi x^3}} \exp(-\frac{(x-1)^2}{2x c^2})

    for :math:`x >= 0` and :math:`c > 0`.

    `fatiguelife` takes ``c`` as a shape parameter for :math:`c`.

    %(after_notes)s

    References
    ----------
    .. [1] "Birnbaum-Saunders distribution",
           https://en.wikipedia.org/wiki/Birnbaum-Saunders_distribution

    %(example)s

    """
    # 设置支持掩码（support mask）为开放支持（_open_support_mask）
    _support_mask = rv_continuous._open_support_mask
    
    # 定义一个用于描述参数形状信息的方法
    def _shape_info(self):
        return [_ShapeInfo("c", False, (0, np.inf), (False, False))]
    
    # 定义随机变量样本生成方法，依赖随机数发生器 random_state
    def _rvs(self, c, size=None, random_state=None):
        z = random_state.standard_normal(size)
        x = 0.5*c*z
        x2 = x*x
        t = 1.0 + 2*x2 + 2*x*np.sqrt(1 + x2)
        return t
    
    # 定义概率密度函数（PDF），根据疲劳寿命分布的公式计算
    def _pdf(self, x, c):
        # fatiguelife.pdf(x, c) =
        #     (x+1) / (2*c*sqrt(2*pi*x**3)) * exp(-(x-1)**2/(2*x*c**2))
        return np.exp(self._logpdf(x, c))
    
    # 定义概率密度函数的对数（log PDF）计算方法
    def _logpdf(self, x, c):
        return (np.log(x+1) - (x-1)**2 / (2.0*x*c**2) - np.log(2*c) -
                0.5*(np.log(2*np.pi) + 3*np.log(x)))
    
    # 定义累积分布函数（CDF）的计算方法
    def _cdf(self, x, c):
        return _norm_cdf(1.0 / c * (np.sqrt(x) - 1.0/np.sqrt(x)))
    
    # 定义百分位点函数（PPF）的计算方法
    def _ppf(self, q, c):
        tmp = c * _norm_ppf(q)
        return 0.25 * (tmp + np.sqrt(tmp**2 + 4))**2
    # 定义一个私有方法 `_sf`，计算标准 Fisher-Snedecor 分布的倒数与平方根的乘积
    def _sf(self, x, c):
        return _norm_sf(1.0 / c * (np.sqrt(x) - 1.0/np.sqrt(x)))

    # 定义一个私有方法 `_isf`，计算标准 Fisher-Snedecor 分布的逆累积分布函数
    def _isf(self, q, c):
        # 计算临时变量 tmp，用于反向映射累积分布函数的值
        tmp = -c * _norm_ppf(q)
        # 返回 Fisher-Snedecor 分布的逆累积分布函数的值
        return 0.25 * (tmp + np.sqrt(tmp**2 + 4))**2

    # 定义一个私有方法 `_stats`，计算 Fisher-Snedecor 分布的统计量
    def _stats(self, c):
        # NB: 维基百科上关于峰度的公式可能存在错误：
        # 它是40，而不是41。至少与 Wolfram Alpha 给出的不一致。
        # 下面的公式通过了测试，而维基百科的没有。目前我还没有勇气
        # 确实检查原始矩表达式中的系数。
        
        # 计算 c 的平方
        c2 = c*c
        # 计算均值 mu
        mu = c2 / 2.0 + 1.0
        # 计算第二矩 mu2
        den = 5.0 * c2 + 4.0
        mu2 = c2 * den / 4.0
        # 计算偏度 g1
        g1 = 4 * c * (11 * c2 + 6.0) / np.power(den, 1.5)
        # 计算峰度 g2
        g2 = 6 * c2 * (93 * c2 + 40.0) / den**2.0
        # 返回均值、第二矩、偏度和峰度
        return mu, mu2, g1, g2
fatiguelife = fatiguelife_gen(a=0.0, name='fatiguelife')

# 创建一个名为 fatiguelife 的对象，使用 fatiguelife_gen 类生成，参数 a 设置为 0.0，名称设置为 'fatiguelife'

class foldcauchy_gen(rv_continuous):
    r"""A folded Cauchy continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `foldcauchy` is:

    .. math::

        f(x, c) = \frac{1}{\pi (1+(x-c)^2)} + \frac{1}{\pi (1+(x+c)^2)}

    for :math:`x \ge 0` and :math:`c \ge 0`.

    `foldcauchy` takes ``c`` as a shape parameter for :math:`c`.

    %(example)s

    """
    
    def _argcheck(self, c):
        # 检查参数 c 是否大于等于 0
        return c >= 0

    def _shape_info(self):
        # 返回一个 _ShapeInfo 对象的列表，描述了分布的形状参数 c 的取值范围
        return [_ShapeInfo("c", False, (0, np.inf), (True, False))]

    def _rvs(self, c, size=None, random_state=None):
        # 生成服从折叠柯西分布的随机变量，返回其绝对值
        return abs(cauchy.rvs(loc=c, size=size,
                              random_state=random_state))

    def _pdf(self, x, c):
        # 返回折叠柯西分布的概率密度函数值
        return 1.0/np.pi*(1.0/(1+(x-c)**2) + 1.0/(1+(x+c)**2))

    def _cdf(self, x, c):
        # 返回折叠柯西分布的累积分布函数值
        return 1.0/np.pi*(np.arctan(x-c) + np.arctan(x+c))

    def _sf(self, x, c):
        # 返回折叠柯西分布的生存函数值，即 1 - CDF(x, c)
        return (np.arctan2(1, x - c) + np.arctan2(1, x + c))/np.pi

    def _stats(self, c):
        # 返回折叠柯西分布的统计特性，这里包括无穷大和 NaN
        return np.inf, np.inf, np.nan, np.nan

# 创建一个名为 foldcauchy 的对象，使用 foldcauchy_gen 类生成，参数 a 设置为 0.0，名称设置为 'foldcauchy'
foldcauchy = foldcauchy_gen(a=0.0, name='foldcauchy')

class f_gen(rv_continuous):
    r"""An F continuous random variable.

    For the noncentral F distribution, see `ncf`.

    %(before_notes)s

    See Also
    --------
    ncf

    Notes
    -----
    The F distribution with :math:`df_1 > 0` and :math:`df_2 > 0` degrees of freedom is
    the distribution of the ratio of two independent chi-squared distributions with
    :math:`df_1` and :math:`df_2` degrees of freedom, after rescaling by
    :math:`df_2 / df_1`.

    The probability density function for `f` is:

    .. math::

        f(x, df_1, df_2) = \frac{df_2^{df_2/2} df_1^{df_1/2} x^{df_1 / 2-1}}
                                {(df_2+df_1 x)^{(df_1+df_2)/2}
                                 B(df_1/2, df_2/2)}

    for :math:`x > 0`.

    `f` accepts shape parameters ``dfn`` and ``dfd`` for :math:`df_1`, the degrees of
    freedom of the chi-squared distribution in the numerator, and :math:`df_2`, the
    degrees of freedom of the chi-squared distribution in the denominator, respectively.

    %(after_notes)s

    %(example)s

    """
    def _shape_info(self):
        # 返回一个 _ShapeInfo 对象的列表，描述了分布的形状参数 dfn 和 dfd 的取值范围
        idfn = _ShapeInfo("dfn", False, (0, np.inf), (False, False))
        idfd = _ShapeInfo("dfd", False, (0, np.inf), (False, False))
        return [idfn, idfd]

    def _rvs(self, dfn, dfd, size=None, random_state=None):
        # 生成服从 F 分布的随机变量，使用指定的自由度参数 dfn 和 dfd，返回随机变量数组
        return random_state.f(dfn, dfd, size)

# 创建一个名为 f 的对象，使用 f_gen 类生成，没有设置参数 a，名称设置为 'f'
    def _pdf(self, x, dfn, dfd):
        # 概率密度函数 (PDF) 的计算
        # F.pdf(x, df1, df2) = df2**(df2/2) * df1**(df1/2) * x**(df1/2-1) /
        #                      (df2+df1*x)**((df1+df2)/2) * B(df1/2, df2/2)
        return np.exp(self._logpdf(x, dfn, dfd))

    def _logpdf(self, x, dfn, dfd):
        # 对数概率密度函数 (log PDF) 的计算
        n = 1.0 * dfn
        m = 1.0 * dfd
        lPx = (m/2 * np.log(m) + n/2 * np.log(n) + sc.xlogy(n/2 - 1, x)
               - (((n+m)/2) * np.log(m + n*x) + sc.betaln(n/2, m/2)))
        return lPx

    def _cdf(self, x, dfn, dfd):
        # 累积分布函数 (CDF) 的计算
        return sc.fdtr(dfn, dfd, x)

    def _sf(self, x, dfn, dfd):
        # 生存函数 (Survival function) 的计算
        return sc.fdtrc(dfn, dfd, x)

    def _ppf(self, q, dfn, dfd):
        # 百分点函数 (Percent point function, inverse of CDF) 的计算
        return sc.fdtri(dfn, dfd, q)

    def _stats(self, dfn, dfd):
        # 统计量的计算
        v1, v2 = 1. * dfn, 1. * dfd
        v2_2, v2_4, v2_6, v2_8 = v2 - 2., v2 - 4., v2 - 6., v2 - 8.

        mu = _lazywhere(
            v2 > 2, (v2, v2_2),
            lambda v2, v2_2: v2 / v2_2,
            np.inf)

        mu2 = _lazywhere(
            v2 > 4, (v1, v2, v2_2, v2_4),
            lambda v1, v2, v2_2, v2_4:
            2 * v2 * v2 * (v1 + v2_2) / (v1 * v2_2**2 * v2_4),
            np.inf)

        g1 = _lazywhere(
            v2 > 6, (v1, v2_2, v2_4, v2_6),
            lambda v1, v2_2, v2_4, v2_6:
            (2 * v1 + v2_2) / v2_6 * np.sqrt(v2_4 / (v1 * (v1 + v2_2))),
            np.nan)
        g1 *= np.sqrt(8.)

        g2 = _lazywhere(
            v2 > 8, (g1, v2_6, v2_8),
            lambda g1, v2_6, v2_8: (8 + g1 * g1 * v2_6) / v2_8,
            np.nan)
        g2 *= 3. / 2.

        return mu, mu2, g1, g2

    def _entropy(self, dfn, dfd):
        # 熵的计算
        # 文献中的公式不正确。这个公式与使用通用熵定义进行数值积分得到的结果相同。
        # 在 tests/test_conntinous_basic 中进行了测试。
        half_dfn = 0.5 * dfn
        half_dfd = 0.5 * dfd
        half_sum = 0.5 * (dfn + dfd)

        return (np.log(dfd) - np.log(dfn) + sc.betaln(half_dfn, half_dfd) +
                (1 - half_dfn) * sc.psi(half_dfn) - (1 + half_dfd) *
                sc.psi(half_dfd) + half_sum * sc.psi(half_sum))
# 创建一个以 a=0.0 和 name='f' 为参数的 foldnorm_gen 实例，并赋值给 f
f = foldnorm_gen(a=0.0, name='f')

## Folded Normal
##   abs(Z) where (Z is normal with mu=L and std=S so that c=abs(L)/S)
##
##  note: regress docs have scale parameter correct, but first parameter
##    he gives is a shape parameter A = c * scale

##  Half-normal is folded normal with shape-parameter c=0.

# 定义了一个名为 foldnorm_gen 的类，继承自 rv_continuous 类
class foldnorm_gen(rv_continuous):
    r"""A folded normal continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `foldnorm` is:

    .. math::

        f(x, c) = \sqrt{2/\pi} cosh(c x) \exp(-\frac{x^2+c^2}{2})

    for :math:`x \ge 0` and :math:`c \ge 0`.

    `foldnorm` takes ``c`` as a shape parameter for :math:`c`.

    %(after_notes)s

    %(example)s

    """

    # _argcheck 方法用于检查参数 c 是否满足条件 c >= 0
    def _argcheck(self, c):
        return c >= 0

    # _shape_info 方法返回一个描述参数的 _ShapeInfo 实例列表
    def _shape_info(self):
        return [_ShapeInfo("c", False, (0, np.inf), (True, False))]

    # _rvs 方法生成符合折叠正态分布的随机变量
    def _rvs(self, c, size=None, random_state=None):
        return abs(random_state.standard_normal(size) + c)

    # _pdf 方法计算折叠正态分布的概率密度函数
    def _pdf(self, x, c):
        # foldnormal.pdf(x, c) = sqrt(2/pi) * cosh(c*x) * exp(-(x**2+c**2)/2)
        return _norm_pdf(x + c) + _norm_pdf(x-c)

    # _cdf 方法计算折叠正态分布的累积分布函数
    def _cdf(self, x, c):
        sqrt_two = np.sqrt(2)
        return 0.5 * (sc.erf((x - c)/sqrt_two) + sc.erf((x + c)/sqrt_two))

    # _sf 方法计算折叠正态分布的生存函数（1 - CDF）
    def _sf(self, x, c):
        return _norm_sf(x - c) + _norm_sf(x + c)

    # _stats 方法计算折叠正态分布的统计特性
    def _stats(self, c):
        # Regina C. Elandt, Technometrics 3, 551 (1961)
        # https://www.jstor.org/stable/1266561
        #
        c2 = c*c
        expfac = np.exp(-0.5*c2) / np.sqrt(2.*np.pi)

        mu = 2.*expfac + c * sc.erf(c/np.sqrt(2))
        mu2 = c2 + 1 - mu*mu

        g1 = 2. * (mu*mu*mu - c2*mu - expfac)
        g1 /= np.power(mu2, 1.5)

        g2 = c2 * (c2 + 6.) + 3 + 8.*expfac*mu
        g2 += (2. * (c2 - 3.) - 3. * mu**2) * mu**2
        g2 = g2 / mu2**2.0 - 3.

        return mu, mu2, g1, g2


# 创建一个以 a=0.0 和 name='foldnorm' 为参数的 foldnorm_gen 实例，并赋值给 foldnorm
foldnorm = foldnorm_gen(a=0.0, name='foldnorm')


# 定义了一个名为 weibull_min_gen 的类，继承自 rv_continuous 类
class weibull_min_gen(rv_continuous):
    r"""Weibull minimum continuous random variable.

    The Weibull Minimum Extreme Value distribution, from extreme value theory
    (Fisher-Gnedenko theorem), is also often simply called the Weibull
    distribution. It arises as the limiting distribution of the rescaled
    minimum of iid random variables.

    %(before_notes)s

    See Also
    --------
    weibull_max, numpy.random.Generator.weibull, exponweib

    Notes
    -----
    The probability density function for `weibull_min` is:

    .. math::

        f(x, c) = c x^{c-1} \exp(-x^c)

    for :math:`x > 0`, :math:`c > 0`.

    `weibull_min` takes ``c`` as a shape parameter for :math:`c`.
    (named :math:`k` in Wikipedia article and :math:`a` in
    ``numpy.random.weibull``).  Special shape values are :math:`c=1` and
    :math:`c=2` where Weibull distribution reduces to the `expon` and
    `rayleigh` distributions respectively.

    Suppose ``X`` is an exponentially distributed random variable with
    """

# 注释结束
    def _shape_info(self):
        # 返回一个包含 ShapeInfo 对象的列表，表示分布的参数信息
        return [_ShapeInfo("c", False, (0, np.inf), (False, False))]

    def _pdf(self, x, c):
        # 计算 Weibull 分布的概率密度函数值
        # weibull_min.pdf(x, c) = c * x**(c-1) * exp(-x**c)
        return c * pow(x, c-1) * np.exp(-pow(x, c))

    def _logpdf(self, x, c):
        # 计算 Weibull 分布的对数概率密度函数值
        return np.log(c) + sc.xlogy(c - 1, x) - pow(x, c)

    def _cdf(self, x, c):
        # 计算 Weibull 分布的累积分布函数值
        return -sc.expm1(-pow(x, c))

    def _ppf(self, q, c):
        # 计算 Weibull 分布的分位点函数值（逆累积分布函数）
        return pow(-sc.log1p(-q), 1.0 / c)

    def _sf(self, x, c):
        # 计算 Weibull 分布的生存函数值（1 - CDF）
        return np.exp(self._logsf(x, c))

    def _logsf(self, x, c):
        # 计算 Weibull 分布的对数生存函数值
        return -pow(x, c)

    def _isf(self, q, c):
        # 计算 Weibull 分布的逆生存函数值（分位点函数对应值的逆）
        return (-np.log(q))**(1 / c)

    def _munp(self, n, c):
        # 计算 Weibull 分布的 n 阶原点矩
        return sc.gamma(1.0 + n * 1.0 / c)

    def _entropy(self, c):
        # 计算 Weibull 分布的熵
        return -_EULER / c - np.log(c) + _EULER + 1

    @extend_notes_in_docstring(rv_continuous, notes="""\
        如果 ``method='mm'``，则保留用户固定的参数，并尽可能使用剩余参数匹配分布和样本矩。
        例如，如果用户使用 ``floc`` 固定了位置参数，那么参数只会匹配分布的偏度和方差到样本的偏度和方差；
        不会尝试匹配均值或最小化误差的范数。
        \n\n""")
# 使用 scipy.stats 中的 weibull_min_gen 生成器创建一个 Weibull 分布的随机变量对象 weibull_min
weibull_min = weibull_min_gen(a=0.0, name='weibull_min')

# 定义一个新的类 truncweibull_min_gen，继承自 rv_continuous 类，表示一个双向截断的 Weibull 最小值连续随机变量
class truncweibull_min_gen(rv_continuous):
    r"""A doubly truncated Weibull minimum continuous random variable.

    %(before_notes)s

    See Also
    --------
    weibull_min, truncexpon

    Notes
    -----
    The probability density function for `truncweibull_min` is:

    .. math::

        f(x, a, b, c) = \frac{c x^{c-1} \exp(-x^c)}{\exp(-a^c) - \exp(-b^c)}

    for :math:`a < x <= b`, :math:`0 \le a < b` and :math:`c > 0`.

    `truncweibull_min` takes :math:`a`, :math:`b`, and :math:`c` as shape
    parameters.

    Notice that the truncation values, :math:`a` and :math:`b`, are defined in
    standardized form:

    .. math::

        a = (u_l - loc)/scale
        b = (u_r - loc)/scale

    where :math:`u_l` and :math:`u_r` are the specific left and right
    truncation values, respectively. In other words, the support of the
    distribution becomes :math:`(a*scale + loc) < x <= (b*scale + loc)` when
    :math:`loc` and/or :math:`scale` are provided.

    %(after_notes)s

    References
    ----------

    .. [1] Rinne, H. "The Weibull Distribution: A Handbook". CRC Press (2009).

    %(example)s

    """
    
    # 定义一个方法 _argcheck，用于检查参数的有效性，返回布尔值指示参数是否有效
    def _argcheck(self, c, a, b):
        return (a >= 0.) & (b > a) & (c > 0.)

    # 定义一个方法 _shape_info，返回参数的信息，这里使用了一个自定义的 _ShapeInfo 类
    def _shape_info(self):
        ic = _ShapeInfo("c", False, (0, np.inf), (False, False))
        ia = _ShapeInfo("a", False, (0, np.inf), (True, False))
        ib = _ShapeInfo("b", False, (0, np.inf), (False, False))
        return [ic, ia, ib]

    # 定义一个方法 _fitstart，用于估计分布拟合的起始值
    def _fitstart(self, data):
        # 使用默认参数 (1, 0, 1) 作为初始参数，对数据进行拟合
        return super()._fitstart(data, args=(1, 0, 1))

    # 定义一个方法 _get_support，返回分布的支持区间 (a, b)
    def _get_support(self, c, a, b):
        return a, b

    # 定义一个方法 _pdf，计算概率密度函数 (PDF)
    def _pdf(self, x, c, a, b):
        denum = (np.exp(-pow(a, c)) - np.exp(-pow(b, c)))
        return (c * pow(x, c-1) * np.exp(-pow(x, c))) / denum

    # 定义一个方法 _logpdf，计算对数概率密度函数 (log PDF)
    def _logpdf(self, x, c, a, b):
        logdenum = np.log(np.exp(-pow(a, c)) - np.exp(-pow(b, c)))
        return np.log(c) + sc.xlogy(c - 1, x) - pow(x, c) - logdenum

    # 定义一个方法 _cdf，计算累积分布函数 (CDF)
    def _cdf(self, x, c, a, b):
        num = (np.exp(-pow(a, c)) - np.exp(-pow(x, c)))
        denum = (np.exp(-pow(a, c)) - np.exp(-pow(b, c)))
        return num / denum

    # 定义一个方法 _logcdf，计算对数累积分布函数 (log CDF)
    def _logcdf(self, x, c, a, b):
        lognum = np.log(np.exp(-pow(a, c)) - np.exp(-pow(x, c)))
        logdenum = np.log(np.exp(-pow(a, c)) - np.exp(-pow(b, c)))
        return lognum - logdenum

    # 定义一个方法 _sf，计算生存函数 (SF)
    def _sf(self, x, c, a, b):
        num = (np.exp(-pow(x, c)) - np.exp(-pow(b, c)))
        denum = (np.exp(-pow(a, c)) - np.exp(-pow(b, c)))
        return num / denum

    # 定义一个方法 _logsf，计算对数生存函数 (log SF)
    def _logsf(self, x, c, a, b):
        lognum = np.log(np.exp(-pow(x, c)) - np.exp(-pow(b, c)))
        logdenum = np.log(np.exp(-pow(a, c)) - np.exp(-pow(b, c)))
        return lognum - logdenum

    # 定义一个方法 _isf，计算逆生存函数 (ISF)
    def _isf(self, q, c, a, b):
        return pow(
            -np.log((1 - q) * np.exp(-pow(b, c)) + q * np.exp(-pow(a, c))), 1/c
            )
    # 计算概率分布函数（probability distribution function, pdf）的百分位点的逆函数，使用参数 q, c, a, b
    def _ppf(self, q, c, a, b):
        # 计算指数值，基于参数 a, c 的幂次
        return pow(
            # 计算概率密度函数的对数，通过参数 q, a, b, c 计算
            -np.log((1 - q) * np.exp(-pow(a, c)) + q * np.exp(-pow(b, c))), 1/c
            )

    # 计算矩的 n 次中心矩（moment about the mean），使用参数 n, c, a, b
    def _munp(self, n, c, a, b):
        # 计算 gamma 函数的值，基于参数 n/c + 1 和 pow(a, c), pow(b, c)
        gamma_fun = sc.gamma(n/c + 1.) * (
            # 计算不完全 gamma 函数（regularized lower incomplete gamma function）的差值
            sc.gammainc(n/c + 1., pow(b, c)) - sc.gammainc(n/c + 1., pow(a, c))
            )
        # 计算指数值，基于参数 a, c 的幂次，然后减去相反指数值
        denum = (np.exp(-pow(a, c)) - np.exp(-pow(b, c)))
        # 返回 gamma 函数的值除以 denum 的结果
        return gamma_fun / denum


这段代码定义了两个方法 `_ppf` 和 `_munp`，分别用于计算概率分布函数的百分位点的逆函数和矩的 n 次中心矩。每个方法接受四个参数 q, c, a, b，并根据这些参数执行数学运算来计算特定的数学函数值。
# 使用 truncweibull_min_gen 函数生成一个名为 truncweibull_min 的截断韦伯最小生成器对象
truncweibull_min = truncweibull_min_gen(name='truncweibull_min')

# 定义一个名为 weibull_max_gen 的类，继承于 rv_continuous 类
class weibull_max_gen(rv_continuous):
    r"""Weibull maximum continuous random variable.

    The Weibull Maximum Extreme Value distribution, from extreme value theory
    (Fisher-Gnedenko theorem), is the limiting distribution of rescaled
    maximum of iid random variables. This is the distribution of -X
    if X is from the `weibull_min` function.

    %(before_notes)s

    See Also
    --------
    weibull_min

    Notes
    -----
    The probability density function for `weibull_max` is:

    .. math::

        f(x, c) = c (-x)^{c-1} \exp(-(-x)^c)

    for :math:`x < 0`, :math:`c > 0`.

    `weibull_max` takes ``c`` as a shape parameter for :math:`c`.

    %(after_notes)s

    References
    ----------
    https://en.wikipedia.org/wiki/Weibull_distribution

    https://en.wikipedia.org/wiki/Fisher-Tippett-Gnedenko_theorem

    %(example)s

    """
    
    # 定义一个函数 _shape_info，返回一个描述参数 'c' 的 _ShapeInfo 对象列表
    def _shape_info(self):
        return [_ShapeInfo("c", False, (0, np.inf), (False, False))]

    # 定义一个函数 _pdf，计算韦伯最大分布的概率密度函数值
    def _pdf(self, x, c):
        # 返回概率密度函数的计算结果 c * (-x)**(c-1) * exp(-(-x)**c)
        return c*pow(-x, c-1)*np.exp(-pow(-x, c))

    # 定义一个函数 _logpdf，计算韦伯最大分布的对数概率密度函数值
    def _logpdf(self, x, c):
        # 返回对数概率密度函数的计算结果 np.log(c) + sc.xlogy(c-1, -x) - pow(-x, c)
        return np.log(c) + sc.xlogy(c-1, -x) - pow(-x, c)

    # 定义一个函数 _cdf，计算韦伯最大分布的累积分布函数值
    def _cdf(self, x, c):
        # 返回累积分布函数的计算结果 np.exp(-pow(-x, c))
        return np.exp(-pow(-x, c))

    # 定义一个函数 _logcdf，计算韦伯最大分布的对数累积分布函数值
    def _logcdf(self, x, c):
        # 返回对数累积分布函数的计算结果 -pow(-x, c)
        return -pow(-x, c)

    # 定义一个函数 _sf，计算韦伯最大分布的生存函数值
    def _sf(self, x, c):
        # 返回生存函数的计算结果 -sc.expm1(-pow(-x, c))
        return -sc.expm1(-pow(-x, c))

    # 定义一个函数 _ppf，计算韦伯最大分布的分位点函数值
    def _ppf(self, q, c):
        # 返回分位点函数的计算结果 -pow(-np.log(q), 1.0/c)
        return -pow(-np.log(q), 1.0/c)

    # 定义一个函数 _munp，计算韦伯最大分布的 n 阶矩函数值
    def _munp(self, n, c):
        # 计算 gamma(1.0+n*1.0/c) 的值
        val = sc.gamma(1.0+n*1.0/c)
        # 根据 n 的奇偶性确定符号 sgn
        if int(n) % 2:
            sgn = -1
        else:
            sgn = 1
        # 返回最终的 n 阶矩函数值 sgn * val
        return sgn * val

    # 定义一个函数 _entropy，计算韦伯最大分布的熵值
    def _entropy(self, c):
        # 返回熵值的计算结果 -_EULER / c - np.log(c) + _EULER + 1
        return -_EULER / c - np.log(c) + _EULER + 1


# 使用 weibull_max_gen 类生成一个名为 weibull_max 的对象，参数 b=0.0，名字为 'weibull_max'
weibull_max = weibull_max_gen(b=0.0, name='weibull_max')

# 定义一个名为 genlogistic_gen 的类，继承于 rv_continuous 类
class genlogistic_gen(rv_continuous):
    r"""A generalized logistic continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `genlogistic` is:

    .. math::

        f(x, c) = c \frac{\exp(-x)}
                         {(1 + \exp(-x))^{c+1}}

    for real :math:`x` and :math:`c > 0`. In literature, different
    generalizations of the logistic distribution can be found. This is the type 1
    generalized logistic distribution according to [1]_. It is also referred to
    as the skew-logistic distribution [2]_.

    `genlogistic` takes ``c`` as a shape parameter for :math:`c`.

    %(after_notes)s

    References
    ----------
    .. [1] Johnson et al. "Continuous Univariate Distributions", Volume 2,
           Wiley. 1995.
    .. [2] "Generalized Logistic Distribution", Wikipedia,
           https://en.wikipedia.org/wiki/Generalized_logistic_distribution

    %(example)s

    """
    
    # 定义一个函数 _shape_info，返回一个描述参数 'c' 的 _ShapeInfo 对象列表
    def _shape_info(self):
        return [_ShapeInfo("c", False, (0, np.inf), (False, False))]
    def _pdf(self, x, c):
        # 计算广义 logistic 分布的概率密度函数值
        # genlogistic.pdf(x, c) = c * exp(-x) / (1 + exp(-x))**(c+1)
        return np.exp(self._logpdf(x, c))

    def _logpdf(self, x, c):
        # 计算广义 logistic 分布的对数概率密度函数值
        # 有两种数学上等价的表达式：
        #     log(pdf(x, c)) = log(c) - x - (c + 1)*log(1 + exp(-x))
        #                    = log(c) + c*x - (c + 1)*log(1 + exp(x))
        
        # 根据 x 的正负选择不同的系数
        mult = -(c - 1) * (x < 0) - 1
        absx = np.abs(x)
        return np.log(c) + mult*absx - (c+1) * sc.log1p(np.exp(-absx))

    def _cdf(self, x, c):
        # 计算广义 logistic 分布的累积分布函数值
        Cx = (1+np.exp(-x))**(-c)
        return Cx

    def _logcdf(self, x, c):
        # 计算广义 logistic 分布的对数累积分布函数值
        return -c * np.log1p(np.exp(-x))

    def _ppf(self, q, c):
        # 计算广义 logistic 分布的百分位点函数值（逆累积分布函数）
        return -np.log(sc.powm1(q, -1.0/c))

    def _sf(self, x, c):
        # 计算广义 logistic 分布的生存函数值（1 - CDF）
        return -sc.expm1(self._logcdf(x, c))

    def _isf(self, q, c):
        # 计算广义 logistic 分布的逆生存函数值（逆生存函数的百分位点）
        return self._ppf(1 - q, c)

    def _stats(self, c):
        # 计算广义 logistic 分布的统计特性：均值、方差、偏度、峰度
        mu = _EULER + sc.psi(c)
        mu2 = np.pi*np.pi/6.0 + sc.zeta(2, c)
        g1 = -2*sc.zeta(3, c) + 2*_ZETA3
        g1 /= np.power(mu2, 1.5)
        g2 = np.pi**4/15.0 + 6*sc.zeta(4, c)
        g2 /= mu2**2.0
        return mu, mu2, g1, g2

    def _entropy(self, c):
        # 计算广义 logistic 分布的熵
        return _lazywhere(c < 8e6, (c, ),
                          lambda c: -np.log(c) + sc.psi(c + 1) + _EULER + 1,
                          # 熵的渐近展开：psi(c) ~ log(c) - 1/(2 * c)
                          # a = -log(c) + psi(c + 1)
                          #   = -log(c) + psi(c) + 1/c
                          #   ~ -log(c) + log(c) - 1/(2 * c) + 1/c
                          #   = 1/(2 * c)
                          f2=lambda c: 1/(2 * c) + _EULER + 1)
genlogistic = genlogistic_gen(name='genlogistic')

class genpareto_gen(rv_continuous):
    r"""A generalized Pareto continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `genpareto` is:

    .. math::

        f(x, c) = (1 + c x)^{-1 - 1/c}

    defined for :math:`x \ge 0` if :math:`c \ge 0`, and for
    :math:`0 \le x \le -1/c` if :math:`c < 0`.

    `genpareto` takes ``c`` as a shape parameter for :math:`c`.

    For :math:`c=0`, `genpareto` reduces to the exponential
    distribution, `expon`:

    .. math::

        f(x, 0) = \exp(-x)

    For :math:`c=-1`, `genpareto` is uniform on ``[0, 1]``:

    .. math::

        f(x, -1) = 1

    %(after_notes)s

    %(example)s

    """
    
    def _argcheck(self, c):
        # 检查参数 `c` 是否是有限的
        return np.isfinite(c)

    def _shape_info(self):
        # 返回形状信息列表
        return [_ShapeInfo("c", False, (-np.inf, np.inf), (False, False))]

    def _get_support(self, c):
        # 获取支持区间 `[a, b]`，其中 `a` 和 `b` 根据 `c` 的值动态确定
        c = np.asarray(c)
        b = _lazywhere(c < 0, (c,),
                       lambda c: -1. / c,
                       np.inf)
        a = np.where(c >= 0, self.a, self.a)
        return a, b

    def _pdf(self, x, c):
        # 计算概率密度函数 `pdf(x, c) = (1 + c * x)**(-1 - 1/c)`
        return np.exp(self._logpdf(x, c))

    def _logpdf(self, x, c):
        # 计算对数概率密度函数 `log(pdf(x, c))`
        return _lazywhere((x == x) & (c != 0), (x, c),
                          lambda x, c: -sc.xlog1py(c + 1., c*x) / c,
                          -x)

    def _cdf(self, x, c):
        # 计算累积分布函数 `cdf(x, c)`
        return -sc.inv_boxcox1p(-x, -c)

    def _sf(self, x, c):
        # 计算生存函数 `sf(x, c)`
        return sc.inv_boxcox(-x, -c)

    def _logsf(self, x, c):
        # 计算对数生存函数 `log(sf(x, c))`
        return _lazywhere((x == x) & (c != 0), (x, c),
                          lambda x, c: -sc.log1p(c*x) / c,
                          -x)

    def _ppf(self, q, c):
        # 计算百分位点函数 `ppf(q, c)`
        return -sc.boxcox1p(-q, -c)

    def _isf(self, q, c):
        # 计算逆生存函数 `isf(q, c)`
        return -sc.boxcox(q, -c)

    def _stats(self, c, moments='mv'):
        # 计算统计量 `m, v, s, k`，根据所需的矩 `moments` 返回相应的值
        if 'm' not in moments:
            m = None
        else:
            m = _lazywhere(c < 1, (c,),
                           lambda xi: 1/(1 - xi),
                           np.inf)
        if 'v' not in moments:
            v = None
        else:
            v = _lazywhere(c < 1/2, (c,),
                           lambda xi: 1 / (1 - xi)**2 / (1 - 2*xi),
                           np.nan)
        if 's' not in moments:
            s = None
        else:
            s = _lazywhere(c < 1/3, (c,),
                           lambda xi: (2 * (1 + xi) * np.sqrt(1 - 2*xi) /
                                       (1 - 3*xi)),
                           np.nan)
        if 'k' not in moments:
            k = None
        else:
            k = _lazywhere(c < 1/4, (c,),
                           lambda xi: (3 * (1 - 2*xi) * (2*xi**2 + xi + 3) /
                                       (1 - 3*xi) / (1 - 4*xi) - 3),
                           np.nan)
        return m, v, s, k
    # 定义一个私有方法 `_munp`，接受参数 `n` 和 `c`
    def _munp(self, n, c):
        # 定义内部函数 `__munp`，计算概率质量函数的值
        def __munp(n, c):
            val = 0.0
            # 生成 0 到 n 的整数数组
            k = np.arange(0, n + 1)
            # 遍历 k 和对应的二项式系数，计算概率质量函数的值
            for ki, cnk in zip(k, sc.comb(n, k)):
                val = val + cnk * (-1) ** ki / (1.0 - c * ki)
            # 根据条件返回计算结果或者无穷大
            return np.where(c * n < 1, val * (-1.0 / c) ** n, np.inf)
        # 如果 c 不等于 0，则延迟执行 `_munp` 函数，传入参数 c，并返回结果
        return _lazywhere(c != 0, (c,),
                          lambda c: __munp(n, c),
                          sc.gamma(n + 1))

    # 定义一个私有方法 `_entropy`，接受参数 `c`
    def _entropy(self, c):
        # 返回 1 + c 作为熵的值
        return 1. + c
genpareto = genpareto_gen(a=0.0, name='genpareto')

# 定义了一个 genpareto_gen 实例对象 genpareto，使用了参数 a=0.0 和 name='genpareto'


class genexpon_gen(rv_continuous):
    r"""A generalized exponential continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `genexpon` is:

    .. math::

        f(x, a, b, c) = (a + b (1 - \exp(-c x)))
                        \exp(-a x - b x + \frac{b}{c}  (1-\exp(-c x)))

    for :math:`x \ge 0`, :math:`a, b, c > 0`.

    `genexpon` takes :math:`a`, :math:`b` and :math:`c` as shape parameters.

    %(after_notes)s

    References
    ----------
    H.K. Ryu, "An Extension of Marshall and Olkin's Bivariate Exponential
    Distribution", Journal of the American Statistical Association, 1993.

    N. Balakrishnan, Asit P. Basu (editors), *The Exponential Distribution:
    Theory, Methods and Applications*, Gordon and Breach, 1995.
    ISBN 10: 2884491929

    %(example)s

    """
    
    def _shape_info(self):
        # 定义了一个方法 _shape_info，返回包含参数限制信息的列表
        ia = _ShapeInfo("a", False, (0, np.inf), (False, False))
        ib = _ShapeInfo("b", False, (0, np.inf), (False, False))
        ic = _ShapeInfo("c", False, (0, np.inf), (False, False))
        return [ia, ib, ic]

    def _pdf(self, x, a, b, c):
        # 定义了一个方法 _pdf，计算 generalized exponential 分布的概率密度函数
        # genexpon.pdf(x, a, b, c) = (a + b * (1 - exp(-c*x))) * \
        #                            exp(-a*x - b*x + b/c * (1-exp(-c*x)))
        return (a + b*(-sc.expm1(-c*x)))*np.exp((-a-b)*x +
                                                b*(-sc.expm1(-c*x))/c)

    def _logpdf(self, x, a, b, c):
        # 定义了一个方法 _logpdf，计算 generalized exponential 分布的对数概率密度函数
        return np.log(a+b*(-sc.expm1(-c*x))) + (-a-b)*x+b*(-sc.expm1(-c*x))/c

    def _cdf(self, x, a, b, c):
        # 定义了一个方法 _cdf，计算 generalized exponential 分布的累积分布函数
        return -sc.expm1((-a-b)*x + b*(-sc.expm1(-c*x))/c)

    def _ppf(self, p, a, b, c):
        # 定义了一个方法 _ppf，计算 generalized exponential 分布的分位点函数
        s = a + b
        t = (b - c*np.log1p(-p))/s
        return (t + sc.lambertw(-b/s * np.exp(-t)).real)/c

    def _sf(self, x, a, b, c):
        # 定义了一个方法 _sf，计算 generalized exponential 分布的生存函数
        return np.exp((-a-b)*x + b*(-sc.expm1(-c*x))/c)

    def _isf(self, p, a, b, c):
        # 定义了一个方法 _isf，计算 generalized exponential 分布的逆生存函数
        s = a + b
        t = (b - c*np.log(p))/s
        return (t + sc.lambertw(-b/s * np.exp(-t)).real)/c


genexpon = genexpon_gen(a=0.0, name='genexpon')

# 定义了一个 genexpon_gen 实例对象 genexpon，使用了参数 a=0.0 和 name='genexpon'


class genextreme_gen(rv_continuous):
    r"""A generalized extreme value continuous random variable.

    %(before_notes)s

    See Also
    --------
    gumbel_r

    Notes
    -----
    For :math:`c=0`, `genextreme` is equal to `gumbel_r` with
    probability density function

    .. math::

        f(x) = \exp(-\exp(-x)) \exp(-x),

    where :math:`-\infty < x < \infty`.

    For :math:`c \ne 0`, the probability density function for `genextreme` is:

    .. math::

        f(x, c) = \exp(-(1-c x)^{1/c}) (1-c x)^{1/c-1},

    where :math:`-\infty < x \le 1/c` if :math:`c > 0` and
    :math:`1/c \le x < \infty` if :math:`c < 0`.

    Note that several sources and software packages use the opposite
    convention for the sign of the shape parameter :math:`c`.

    `genextreme` takes ``c`` as a shape parameter for :math:`c`.

    %(after_notes)s

    %(example)s

    """
    def _argcheck(self, c):
        # 检查参数 c 是否是有限的
        return np.isfinite(c)

    def _shape_info(self):
        # 返回一个列表，包含一个 _ShapeInfo 对象，描述参数形状为 "c"，非正态分布，取值范围为 (-∞, ∞)，边界情况为非闭合
        return [_ShapeInfo("c", False, (-np.inf, np.inf), (False, False))]

    def _get_support(self, c):
        # 计算参数 c 对应的支持区间 _a 和 _b
        _b = np.where(c > 0, 1.0 / np.maximum(c, _XMIN), np.inf)  # 当 c > 0 时，_b = 1 / max(c, _XMIN)，否则为无穷大
        _a = np.where(c < 0, 1.0 / np.minimum(c, -_XMIN), -np.inf)  # 当 c < 0 时，_a = 1 / min(c, -_XMIN)，否则为负无穷
        return _a, _b

    def _loglogcdf(self, x, c):
        # 返回 log(-log(cdf(x, c)))，其中 cdf(x, c) 是分布函数在 x 处的累积分布函数
        return _lazywhere((x == x) & (c != 0), (x, c),
                          lambda x, c: sc.log1p(-c*x)/c, -x)

    def _pdf(self, x, c):
        # 返回概率密度函数在 x 处的值，对应分布是广义极值分布（Generalized Extreme Value Distribution）
        # 对应公式为 exp(-exp(-x))*exp(-x)（当 c==0 时）
        # 或者 exp(-(1-c*x)**(1/c))*(1-c*x)**(1/c-1)（当 x <= 1/c 且 c > 0 时）
        return np.exp(self._logpdf(x, c))

    def _logpdf(self, x, c):
        # 返回概率密度函数在 x 处的对数值，对应分布是广义极值分布（Generalized Extreme Value Distribution）
        cx = _lazywhere((x == x) & (c != 0), (x, c), lambda x, c: c*x, 0.0)  # 当 c != 0 时，cx = c * x，否则为 0.0
        logex2 = sc.log1p(-cx)
        logpex2 = self._loglogcdf(x, c)
        pex2 = np.exp(logpex2)
        # 处理特殊情况
        np.putmask(logpex2, (c == 0) & (x == -np.inf), 0.0)  # 当 c == 0 且 x == -∞ 时，logpex2 = 0.0
        logpdf = _lazywhere(~((cx == 1) | (cx == -np.inf)),
                            (pex2, logpex2, logex2),
                            lambda pex2, lpex2, lex2: -pex2 + lpex2 - lex2,
                            fillvalue=-np.inf)  # 根据条件返回对数概率密度函数的值
        np.putmask(logpdf, (c == 1) & (x == 1), 0.0)  # 当 c == 1 且 x == 1 时，logpdf = 0.0
        return logpdf

    def _logcdf(self, x, c):
        # 返回累积分布函数在 x 处的对数值
        return -np.exp(self._loglogcdf(x, c))

    def _cdf(self, x, c):
        # 返回累积分布函数在 x 处的值
        return np.exp(self._logcdf(x, c))

    def _sf(self, x, c):
        # 返回生存函数在 x 处的值
        return -sc.expm1(self._logcdf(x, c))

    def _ppf(self, q, c):
        # 返回百分位点函数在 q 处的值
        x = -np.log(-np.log(q))
        return _lazywhere((x == x) & (c != 0), (x, c),
                          lambda x, c: -sc.expm1(-c * x) / c, x)

    def _isf(self, q, c):
        # 返回逆生存函数在 q 处的值
        x = -np.log(-sc.log1p(-q))
        return _lazywhere((x == x) & (c != 0), (x, c),
                          lambda x, c: -sc.expm1(-c * x) / c, x)
    # 定义一个内部方法 _stats，计算给定形状参数 c 下的一些统计量
    def _stats(self, c):
        # 定义一个局部函数 g(n)，返回 gamma(n * c + 1) 的值
        def g(n):
            return sc.gamma(n * c + 1)
        
        # 计算 g(1)，g(2)，g(3)，g(4) 的值
        g1 = g(1)
        g2 = g(2)
        g3 = g(3)
        g4 = g(4)
        
        # 计算 g2mg12，根据 c 的值选择不同计算公式
        g2mg12 = np.where(abs(c) < 1e-7, (c*np.pi)**2.0/6.0, g2-g1**2.0)
        
        # 计算 gam2k，根据 c 的值选择不同计算公式
        gam2k = np.where(abs(c) < 1e-7, np.pi**2.0/6.0,
                         sc.expm1(sc.gammaln(2.0*c+1.0)-2*sc.gammaln(c + 1.0))/c**2.0)
        
        # 设置一个极小值 eps
        eps = 1e-14
        
        # 计算 gamk，根据 c 的值选择不同计算公式
        gamk = np.where(abs(c) < eps, -_EULER, sc.expm1(sc.gammaln(c + 1))/c)
        
        # 计算 m，根据 c 的值选择不同计算公式
        m = np.where(c < -1.0, np.nan, -gamk)
        
        # 计算 v，根据 c 的值选择不同计算公式
        v = np.where(c < -0.5, np.nan, g1**2.0*gam2k)
        
        # 计算偏度 sk，根据 c 的值选择不同计算公式
        sk1 = _lazywhere(c >= -1./3,
                         (c, g1, g2, g3, g2mg12),
                         lambda c, g1, g2, g3, g2mg12:
                             np.sign(c)*(-g3 + (g2 + 2*g2mg12)*g1)/g2mg12**1.5,
                         fillvalue=np.nan)
        sk = np.where(abs(c) <= eps**0.29, 12*np.sqrt(6)*_ZETA3/np.pi**3, sk1)
        
        # 计算峰度 ku，根据 c 的值选择不同计算公式
        ku1 = _lazywhere(c >= -1./4,
                         (g1, g2, g3, g4, g2mg12),
                         lambda g1, g2, g3, g4, g2mg12:
                             (g4 + (-4*g3 + 3*(g2 + g2mg12)*g1)*g1)/g2mg12**2,
                         fillvalue=np.nan)
        ku = np.where(abs(c) <= (eps)**0.23, 12.0/5.0, ku1-3.0)
        
        # 返回计算结果 m, v, sk, ku
        return m, v, sk, ku

    # 定义内部方法 _fitstart，用于初始化拟合过程的起始参数
    def _fitstart(self, data):
        # 如果 data 是 CensoredData 类型，则将其非屏蔽数据提取出来
        if isinstance(data, CensoredData):
            data = data._uncensor()
        
        # 计算数据的偏度 g
        g = _skew(data)
        
        # 根据偏度 g 的符号选择参数 a 的值
        if g < 0:
            a = 0.5
        else:
            a = -0.5
        
        # 调用父类方法 _fitstart，传入参数 data 和 a，并返回其结果
        return super()._fitstart(data, args=(a,))

    # 定义内部方法 _munp，计算非中心矩的值
    def _munp(self, n, c):
        # 生成一个包含 0 到 n 的整数数组 k
        k = np.arange(0, n+1)
        
        # 计算 vals，表示非中心矩的值
        vals = 1.0/c**n * np.sum(
            sc.comb(n, k) * (-1)**k * sc.gamma(c*k + 1),
            axis=0)
        
        # 根据条件选择计算结果，避免无穷大值
        return np.where(c*n > -1, vals, np.inf)

    # 定义内部方法 _entropy，计算分布的熵
    def _entropy(self, c):
        # 返回分布的熵的计算结果
        return _EULER*(1 - c) + 1
genextreme = genextreme_gen(name='genextreme')


def _digammainv(y):
    """Inverse of the digamma function (real positive arguments only).

    This function is used in the `fit` method of `gamma_gen`.
    The function uses either optimize.fsolve or optimize.newton
    to solve `sc.digamma(x) - y = 0`.  There is probably room for
    improvement, but currently it works over a wide range of y:

    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> y = 64*rng.standard_normal(1000000)
    >>> y.min(), y.max()
    (-311.43592651416662, 351.77388222276869)
    >>> x = [_digammainv(t) for t in y]
    >>> np.abs(sc.digamma(x) - y).max()
    1.1368683772161603e-13

    """
    # Euler-Mascheroni constant
    _em = 0.5772156649015328606065120

    def func(x):
        return sc.digamma(x) - y

    # Depending on the value of y, choose initial guess x0 for the root finding
    if y > -0.125:
        x0 = np.exp(y) + 0.5
        if y < 10:
            # For small y, use Newton's method which converges faster than fsolve
            # Experimentally determined that newton is more reliable in this range
            # For larger y, newton sometimes fails to converge
            value = optimize.newton(func, x0, tol=1e-10)
            return value
    elif y > -3:
        x0 = np.exp(y/2.332) + 0.08661
    else:
        x0 = 1.0 / (-y - _em)

    # Use fsolve for root finding with a tolerance level of xtol
    value, info, ier, mesg = optimize.fsolve(func, x0, xtol=1e-11,
                                             full_output=True)
    if ier != 1:
        raise RuntimeError("_digammainv: fsolve failed, y = %r" % y)

    return value[0]


## Gamma (Use MATLAB and MATHEMATICA (b=theta=scale, a=alpha=shape) definition)

## gamma(a, loc, scale)  with a an integer is the Erlang distribution
## gamma(1, loc, scale)  is the Exponential distribution
## gamma(df/2, 0, 2) is the chi2 distribution with df degrees of freedom.

class gamma_gen(rv_continuous):
    r"""A gamma continuous random variable.

    %(before_notes)s

    See Also
    --------
    erlang, expon

    Notes
    -----
    The probability density function for `gamma` is:

    .. math::

        f(x, a) = \frac{x^{a-1} e^{-x}}{\Gamma(a)}

    for :math:`x \ge 0`, :math:`a > 0`. Here :math:`\Gamma(a)` refers to the
    gamma function.

    `gamma` takes ``a`` as a shape parameter for :math:`a`.

    When :math:`a` is an integer, `gamma` reduces to the Erlang
    distribution, and when :math:`a=1` to the exponential distribution.

    Gamma distributions are sometimes parameterized with two variables,
    with a probability density function of:

    .. math::

        f(x, \alpha, \beta) =
        \frac{\beta^\alpha x^{\alpha - 1} e^{-\beta x }}{\Gamma(\alpha)}

    Note that this parameterization is equivalent to the above, with
    ``scale = 1 / beta``.

    %(after_notes)s

    %(example)s

    """
    def _shape_info(self):
        return [_ShapeInfo("a", False, (0, np.inf), (False, False))]

    def _rvs(self, a, size=None, random_state=None):
        return random_state.standard_gamma(a, size)
    # 计算 Gamma 分布的概率密度函数 (PDF)，使用公式 gamma.pdf(x, a) = x**(a-1) * exp(-x) / gamma(a)
    def _pdf(self, x, a):
        return np.exp(self._logpdf(x, a))

    # 计算 Gamma 分布的对数概率密度函数 (log PDF)
    def _logpdf(self, x, a):
        # 使用 SciPy 的函数计算对数概率密度函数
        return sc.xlogy(a-1.0, x) - x - sc.gammaln(a)

    # 计算 Gamma 分布的累积分布函数 (CDF)
    def _cdf(self, x, a):
        return sc.gammainc(a, x)

    # 计算 Gamma 分布的生存函数 (SF)
    def _sf(self, x, a):
        return sc.gammaincc(a, x)

    # 计算 Gamma 分布的百分点函数 (PPF)
    def _ppf(self, q, a):
        return sc.gammaincinv(a, q)

    # 计算 Gamma 分布的逆生存函数 (ISF)
    def _isf(self, q, a):
        return sc.gammainccinv(a, q)

    # 计算 Gamma 分布的统计特性，包括均值、方差、偏度和峰度
    def _stats(self, a):
        return a, a, 2.0/np.sqrt(a), 6.0/a

    # 计算 Gamma 分布的熵 (entropy)
    def _entropy(self, a):

        # 定义标准公式计算熵
        def regular_formula(a):
            return sc.psi(a) * (1-a) + a + sc.gammaln(a)

        # 定义渐近公式计算熵
        def asymptotic_formula(a):
            # 使用扩展公式:
            # psi(a) ~ ln(a) - 1/2a - 1/12a^2 + 1/120a^4
            # gammaln(a) ~ a * ln(a) - a - 1/2 * ln(a) + 1/2 ln(2 * pi) +
            #              1/12a - 1/360a^3
            return (0.5 * (1. + np.log(2*np.pi) + np.log(a)) - 1/(3 * a)
                    - (a**-2.)/12 - (a**-3.)/90 + (a**-4.)/120)

        # 根据 a 的大小选择使用标准公式或渐近公式计算熵
        return _lazywhere(a < 250, (a, ), regular_formula,
                          f2=asymptotic_formula)

    # 根据数据估计 Gamma 分布的起始参数 a
    def _fitstart(self, data):
        # Gamma 分布的偏度是 `2 / np.sqrt(a)`，利用这个关系估计形状参数 a
        if isinstance(data, CensoredData):
            data = data._uncensor()
        sk = _skew(data)
        a = 4 / (1e-8 + sk**2)  # 使用偏度来估计形状参数 a
        return super()._fitstart(data, args=(a,))

    # 扩展文档字符串中的注意事项，描述当使用固定位置参数 `floc` 和 `method='MLE'` 时的行为
    @extend_notes_in_docstring(rv_continuous, notes="""\
        当位置参数使用 `floc` 并且 `method='MLE'` 时，此函数使用显式公式或解决比完整最大似然优化问题更简单的数值问题。
        因此，在这种情况下，`optimizer`、`loc` 和 `scale` 参数将被忽略。
        \n\n""")
# 创建一个 gamma_gen 的实例，并指定参数 a=0.0 和 name='gamma'
gamma = gamma_gen(a=0.0, name='gamma')

# 定义一个 erlang_gen 类，它继承自 gamma_gen 类
class erlang_gen(gamma_gen):
    """An Erlang continuous random variable.

    %(before_notes)s

    See Also
    --------
    gamma

    Notes
    -----
    The Erlang distribution is a special case of the Gamma distribution, with
    the shape parameter `a` an integer.  Note that this restriction is not
    enforced by `erlang`. It will, however, generate a warning the first time
    a non-integer value is used for the shape parameter.

    Refer to `gamma` for examples.

    """

    # 检查参数 `_argcheck` 方法，用于验证参数 `a` 是否为正整数
    def _argcheck(self, a):
        allint = np.all(np.floor(a) == a)
        if not allint:
            # 如果参数 `a` 不是整数，发出警告信息
            message = ('The shape parameter of the erlang distribution '
                       f'has been given a non-integer value {a!r}.')
            warnings.warn(message, RuntimeWarning, stacklevel=3)
        # 返回验证结果，即参数 `a` 必须大于 0
        return a > 0

    # 返回一个 `_shape_info` 方法，指定分布的参数信息
    def _shape_info(self):
        return [_ShapeInfo("a", True, (1, np.inf), (True, False))]

    # 重写 `_fitstart` 方法，用于确定初始拟合值
    def _fitstart(self, data):
        # 覆盖 `gamma_gen_fitstart` 方法，确保使用整数初始值 `a`
        if isinstance(data, CensoredData):
            data = data._uncensor()
        # 计算初始值 `a`，保证除法的正规化，避免当 `_skew(data)` 为 0 或接近 0 时出现问题
        a = int(4.0 / (1e-8 + _skew(data)**2))
        return super(gamma_gen, self)._fitstart(data, args=(a,))

    # 通过修饰器 `@extend_notes_in_docstring` 来修改 `fit` 方法的文档字符串
    @extend_notes_in_docstring(rv_continuous, notes="""\
        The Erlang distribution is generally defined to have integer values
        for the shape parameter.  This is not enforced by the `erlang` class.
        When fitting the distribution, it will generally return a non-integer
        value for the shape parameter.  By using the keyword argument
        `f0=<integer>`, the fit method can be constrained to fit the data to
        a specific integer shape parameter.""")
    # 重写 `fit` 方法，允许对分布进行拟合
    def fit(self, data, *args, **kwds):
        return super().fit(data, *args, **kwds)


# 创建一个 erlang_gen 的实例，并指定参数 a=0.0 和 name='erlang'
erlang = erlang_gen(a=0.0, name='erlang')

# 定义一个 gengamma_gen 类，它继承自 rv_continuous 类
class gengamma_gen(rv_continuous):
    r"""A generalized gamma continuous random variable.

    %(before_notes)s

    See Also
    --------
    gamma, invgamma, weibull_min

    Notes
    -----
    The probability density function for `gengamma` is ([1]_):

    .. math::

        f(x, a, c) = \frac{|c| x^{c a-1} \exp(-x^c)}{\Gamma(a)}

    for :math:`x \ge 0`, :math:`a > 0`, and :math:`c \ne 0`.
    :math:`\Gamma` is the gamma function (`scipy.special.gamma`).

    `gengamma` takes :math:`a` and :math:`c` as shape parameters.

    %(after_notes)s

    References
    ----------
    .. [1] E.W. Stacy, "A Generalization of the Gamma Distribution",
       Annals of Mathematical Statistics, Vol 33(3), pp. 1187--1192.

    %(example)s

    """
    # 检查参数 `_argcheck` 方法，用于验证参数 `a` 和 `c`
    def _argcheck(self, a, c):
        return (a > 0) & (c != 0)
    # 定义一个方法，返回包含两个 _ShapeInfo 实例的列表，分别表示参数 a 和 c 的形状信息
    def _shape_info(self):
        ia = _ShapeInfo("a", False, (0, np.inf), (False, False))
        ic = _ShapeInfo("c", False, (-np.inf, np.inf), (False, False))
        return [ia, ic]

    # 定义一个方法，计算概率密度函数 PDF，在给定参数 a 和 c 的情况下，返回 x 的指数值
    def _pdf(self, x, a, c):
        return np.exp(self._logpdf(x, a, c))

    # 定义一个方法，计算对数概率密度函数 log PDF，在给定参数 a 和 c 的情况下，返回 x 的对数值
    def _logpdf(self, x, a, c):
        # 使用 lazywhere 函数根据条件 (x != 0) | (c > 0) 来计算对数概率密度函数的值
        return _lazywhere((x != 0) | (c > 0), (x, c),
                          lambda x, c: (np.log(abs(c)) + sc.xlogy(c*a - 1, x)
                                        - x**c - sc.gammaln(a)),
                          fillvalue=-np.inf)

    # 定义一个方法，计算累积分布函数 CDF，在给定参数 a 和 c 的情况下，返回 x 的值
    def _cdf(self, x, a, c):
        # 计算 x 的 c 次方
        xc = x**c
        # 根据 c 的正负值返回不同的 gamma 函数值
        val1 = sc.gammainc(a, xc)
        val2 = sc.gammaincc(a, xc)
        return np.where(c > 0, val1, val2)

    # 定义一个方法，生成随机变量值 RVS，在给定参数 a 和 c 的情况下，返回生成的随机数
    def _rvs(self, a, c, size=None, random_state=None):
        # 使用随机数生成器 random_state 生成 gamma 分布的随机数，并对其进行幂运算以得到 RVS
        r = random_state.standard_gamma(a, size=size)
        return r**(1./c)

    # 定义一个方法，计算生存函数 SF，在给定参数 a 和 c 的情况下，返回 x 的值
    def _sf(self, x, a, c):
        # 计算 x 的 c 次方
        xc = x**c
        # 根据 c 的正负值返回不同的 gamma 函数值
        val1 = sc.gammainc(a, xc)
        val2 = sc.gammaincc(a, xc)
        return np.where(c > 0, val2, val1)

    # 定义一个方法，计算百分位点函数 PPF，在给定参数 q、a 和 c 的情况下，返回百分位点的值
    def _ppf(self, q, a, c):
        # 根据 c 的正负值返回不同的 gamma 逆函数值，并对其进行幂运算以得到 PPF
        val1 = sc.gammaincinv(a, q)
        val2 = sc.gammainccinv(a, q)
        return np.where(c > 0, val1, val2)**(1.0/c)

    # 定义一个方法，计算逆生存函数 ISF，在给定参数 q、a 和 c 的情况下，返回逆生存函数的值
    def _isf(self, q, a, c):
        # 根据 c 的正负值返回不同的 gamma 逆函数值，并对其进行幂运算以得到 ISF
        val1 = sc.gammaincinv(a, q)
        val2 = sc.gammainccinv(a, q)
        return np.where(c > 0, val2, val1)**(1.0/c)

    # 定义一个方法，计算非中心矩 MUNP，在给定参数 n、a 和 c 的情况下，返回非中心矩的值
    def _munp(self, n, a, c):
        # 使用 Pochhammer 符号计算非中心矩
        # Pochhammer 符号：sc.poch(a, n*1.0/c) = gamma(a+n)/gamma(a)
        return sc.poch(a, n*1.0/c)

    # 定义一个方法，计算熵 ENTROPY，在给定参数 a 和 c 的情况下，返回熵的值
    def _entropy(self, a, c):
        # 定义两种不同的熵计算方式：regular 和 asymptotic
        def regular(a, c):
            # 计算正常情况下的熵值
            val = sc.psi(a)
            A = a * (1 - val) + val / c
            B = sc.gammaln(a) - np.log(abs(c))
            h = A + B
            return h

        def asymptotic(a, c):
            # 使用渐近展开计算 gammaln 和 psi 函数的熵值
            return (norm._entropy() - np.log(a)/2
                    - np.log(np.abs(c)) + (a**-1.)/6 - (a**-3.)/90
                    + (np.log(a) - (a**-1.)/2 - (a**-2.)/12 + (a**-4.)/120)/c)

        # 根据条件选择使用 regular 或 asymptotic 函数计算熵值 h
        h = _lazywhere(a >= 2e2, (a, c), f=asymptotic, f2=regular)
        return h
gengamma = gengamma_gen(a=0.0, name='gengamma')

class genhalflogistic_gen(rv_continuous):
    r"""A generalized half-logistic continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `genhalflogistic` is:

    .. math::

        f(x, c) = \frac{2 (1 - c x)^{1/(c-1)}}{[1 + (1 - c x)^{1/c}]^2}

    for :math:`0 \le x \le 1/c`, and :math:`c > 0`.

    `genhalflogistic` takes ``c`` as a shape parameter for :math:`c`.

    %(after_notes)s

    %(example)s

    """
    
    def _shape_info(self):
        # 返回形状参数信息
        return [_ShapeInfo("c", False, (0, np.inf), (False, False))]

    def _get_support(self, c):
        # 返回分布的支持区间
        return self.a, 1.0/c

    def _pdf(self, x, c):
        # 概率密度函数，根据公式计算并返回密度值
        # genhalflogistic.pdf(x, c) = 2 * (1-c*x)**(1/c-1) / (1+(1-c*x)**(1/c))**2
        limit = 1.0/c
        tmp = np.asarray(1-c*x)
        tmp0 = tmp**(limit-1)
        tmp2 = tmp0*tmp
        return 2*tmp0 / (1+tmp2)**2

    def _cdf(self, x, c):
        # 累积分布函数，根据公式计算并返回累积概率值
        limit = 1.0/c
        tmp = np.asarray(1-c*x)
        tmp2 = tmp**(limit)
        return (1.0-tmp2) / (1+tmp2)

    def _ppf(self, q, c):
        # 百分位点函数，根据公式计算并返回给定分位数的值
        return 1.0/c * (1 - ((1.0-q) / (1.0+q))**c)

    def _entropy(self, c):
        # 返回熵的值
        return 2 - (2*c+1) * np.log(2)


genhalflogistic = genhalflogistic_gen(a=0.0, name='genhalflogistic')

class genhyperbolic_gen(rv_continuous):
    r"""A generalized hyperbolic continuous random variable.

    %(before_notes)s

    See Also
    --------
    t, norminvgauss, geninvgauss, laplace, cauchy

    Notes
    -----
    The probability density function for `genhyperbolic` is:

    .. math::

        f(x, p, a, b) =
            \frac{(a^2 - b^2)^{p/2}}
            {\sqrt{2\pi}a^{p-1/2}
            K_p\Big(\sqrt{a^2 - b^2}\Big)}
            e^{bx} \times \frac{K_{p - 1/2}
            (a \sqrt{1 + x^2})}
            {(\sqrt{1 + x^2})^{1/2 - p}}

    for :math:`x, p \in ( - \infty; \infty)`,
    :math:`|b| < a` if :math:`p \ge 0`,
    :math:`|b| \le a` if :math:`p < 0`.
    :math:`K_{p}(.)` denotes the modified Bessel function of the second
    kind and order :math:`p` (`scipy.special.kv`)

    `genhyperbolic` takes ``p`` as a tail parameter,
    ``a`` as a shape parameter,
    ``b`` as a skewness parameter.

    %(after_notes)s

    The original parameterization of the Generalized Hyperbolic Distribution
    is found in [1]_ as follows

    .. math::

        f(x, \lambda, \alpha, \beta, \delta, \mu) =
           \frac{(\gamma/\delta)^\lambda}{\sqrt{2\pi}K_\lambda(\delta \gamma)}
           e^{\beta (x - \mu)} \times \frac{K_{\lambda - 1/2}
           (\alpha \sqrt{\delta^2 + (x - \mu)^2})}
           {(\sqrt{\delta^2 + (x - \mu)^2} / \alpha)^{1/2 - \lambda}}

    for :math:`x \in ( - \infty; \infty)`,
    :math:`\gamma := \sqrt{\alpha^2 - \beta^2}`,
    :math:`\lambda, \mu \in ( - \infty; \infty)`,
    :math:`\delta \ge 0, |\beta| < \alpha` if :math:`\lambda \ge 0`,
    :math:`\delta > 0, |\beta| \le \alpha` if :math:`\lambda < 0`.

    """
    """
    The location-scale-based parameterization implemented in
    SciPy is based on [2]_, where :math:`a = \alpha\delta`,
    :math:`b = \beta\delta`, :math:`p = \lambda`,
    :math:`scale=\delta` and :math:`loc=\mu`

    Moments are implemented based on [3]_ and [4]_.

    For the distributions that are a special case such as Student's t,
    it is not recommended to rely on the implementation of genhyperbolic.
    To avoid potential numerical problems and for performance reasons,
    the methods of the specific distributions should be used.

    References
    ----------
    .. [1] O. Barndorff-Nielsen, "Hyperbolic Distributions and Distributions
       on Hyperbolae", Scandinavian Journal of Statistics, Vol. 5(3),
       pp. 151-157, 1978. https://www.jstor.org/stable/4615705

    .. [2] Eberlein E., Prause K. (2002) The Generalized Hyperbolic Model:
        Financial Derivatives and Risk Measures. In: Geman H., Madan D.,
        Pliska S.R., Vorst T. (eds) Mathematical Finance - Bachelier
        Congress 2000. Springer Finance. Springer, Berlin, Heidelberg.
        :doi:`10.1007/978-3-662-12429-1_12`

    .. [3] Scott, David J, Würtz, Diethelm, Dong, Christine and Tran,
       Thanh Tam, (2009), Moments of the generalized hyperbolic
       distribution, MPRA Paper, University Library of Munich, Germany,
       https://EconPapers.repec.org/RePEc:pra:mprapa:19081.

    .. [4] E. Eberlein and E. A. von Hammerstein. Generalized hyperbolic
       and inverse Gaussian distributions: Limiting cases and approximation
       of processes. FDM Preprint 80, April 2003. University of Freiburg.
       https://freidok.uni-freiburg.de/fedora/objects/freidok:7974/datastreams/FILE1/content

    %(example)s

    """

    # 检查参数的有效性，用于分布的参数设定
    def _argcheck(self, p, a, b):
        return (np.logical_and(np.abs(b) < a, p >= 0)
                | np.logical_and(np.abs(b) <= a, p < 0))

    # 返回分布形状的信息，包括参数的范围和是否有下限或上限
    def _shape_info(self):
        ip = _ShapeInfo("p", False, (-np.inf, np.inf), (False, False))
        ia = _ShapeInfo("a", False, (0, np.inf), (True, False))
        ib = _ShapeInfo("b", False, (-np.inf, np.inf), (False, False))
        return [ip, ia, ib]

    # 初始拟合参数设定，确保默认的参数值不违反分布的条件
    def _fitstart(self, data):
        return super()._fitstart(data, args=(1, 1, 0.5))

    # 计算对数概率密度函数
    def _logpdf(self, x, p, a, b):
        # 使用向量化函数计算对数概率密度
        @np.vectorize
        def _logpdf_single(x, p, a, b):
            return _stats.genhyperbolic_logpdf(x, p, a, b)

        return _logpdf_single(x, p, a, b)

    # 计算概率密度函数
    def _pdf(self, x, p, a, b):
        # 使用向量化函数计算概率密度
        @np.vectorize
        def _pdf_single(x, p, a, b):
            return _stats.genhyperbolic_pdf(x, p, a, b)

        return _pdf_single(x, p, a, b)
    # 使用 lambda 函数来包装 func，以便能够使用 np.vectorize 作为装饰器，并且提供 otypes 参数
    # 第一个参数传递给 vectorize 是 func.__get__(object)，以确保与 Python 3.9 兼容，Python 3.10 可以简化为 func
    @lambda func: np.vectorize(func.__get__(object), otypes=[np.float64])
    # _integrate_pdf 函数用于计算 genhyberbolic 分布的概率密度函数从 x0 到 x1 的积分
    @staticmethod
    def _integrate_pdf(x0, x1, p, a, b):
        """
        Integrate the pdf of the genhyberbolic distribution from x0 to x1.
        This is a private function used by _cdf() and _sf() only; either x0
        will be -inf or x1 will be inf.
        """
        # 将用户数据封装成 numpy 数组，并将其作为 ctypes 的 void 指针传递给 C 函数
        user_data = np.array([p, a, b], float).ctypes.data_as(ctypes.c_void_p)
        # 使用 LowLevelCallable 从 Cython 函数 _stats 中获取名为 '_genhyperbolic_pdf' 的低级调用对象
        llc = LowLevelCallable.from_cython(_stats, '_genhyperbolic_pdf',
                                           user_data)
        # 计算一些分布参数
        d = np.sqrt((a + b)*(a - b))
        mean = b/d * sc.kv(p + 1, d) / sc.kv(p, d)
        epsrel = 1e-10
        epsabs = 0
        if x0 < mean < x1:
            # 如果积分区间包含均值，分别在 [x0, mean] 和 [mean, x1] 两个区间上进行积分，并相加
            # 如果尝试在一个 quad 调用中完成积分，而非无限端点远在尾部，可能会返回不正确的结果，
            # 因为它无法“看到”PDF的峰值。
            intgrl = (integrate.quad(llc, x0, mean,
                                     epsrel=epsrel, epsabs=epsabs)[0]
                      + integrate.quad(llc, mean, x1,
                                       epsrel=epsrel, epsabs=epsabs)[0])
        else:
            # 在单个 quad 调用中计算从 x0 到 x1 的积分
            intgrl = integrate.quad(llc, x0, x1,
                                    epsrel=epsrel, epsabs=epsabs)[0]
        # 如果积分结果为 NaN，则发出警告，并返回一个修正后的结果
        if np.isnan(intgrl):
            msg = ("Infinite values encountered in scipy.special.kve. "
                   "Values replaced by NaN to avoid incorrect results.")
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
        # 确保返回值在 [0, 1] 之间
        return max(0.0, min(1.0, intgrl))

    # 计算 genhyberbolic 分布的累积分布函数
    def _cdf(self, x, p, a, b):
        return self._integrate_pdf(-np.inf, x, p, a, b)

    # 计算 genhyberbolic 分布的生存函数
    def _sf(self, x, p, a, b):
        return self._integrate_pdf(x, np.inf, p, a, b)

    # 生成 genhyberbolic 分布的随机变量
    def _rvs(self, p, a, b, size=None, random_state=None):
        # 注意：X = b * V + sqrt(V) * X 具有广义双曲分布
        # 如果 X 是标准正态分布，V 是 geninvgauss(p=p, b=t2, loc=loc, scale=t3) 的分布
        # 计算 geninvgauss 分布中的参数
        t1 = np.float_power(a, 2) - np.float_power(b, 2)
        t2 = np.float_power(t1, 0.5)  # b in the GIG
        t3 = np.float_power(t1, -0.5)  # scale in the GIG
        # 生成 geninvgauss 分布的随机变量
        gig = geninvgauss.rvs(
            p=p,
            b=t2,
            scale=t3,
            size=size,
            random_state=random_state
            )
        # 生成标准正态分布的随机变量
        normst = norm.rvs(size=size, random_state=random_state)

        # 返回 genhyberbolic 分布的随机变量
        return b * gig + np.sqrt(gig) * normst
    # 定义一个方法 _stats，计算统计量 m, v, s, k
    def _stats(self, p, a, b):
        # 使用广播，使得 p, a, b 有相同的形状
        p, a, b = np.broadcast_arrays(p, a, b)
        # 计算 t1：a^2 - b^2 的平方根
        t1 = np.float_power(a, 2) - np.float_power(b, 2)
        t1 = np.float_power(t1, 0.5)
        # 计算 t2：1 / t1
        t2 = np.float_power(1, 2) * np.float_power(t1, - 1)
        # 创建一个包含整数 0 到 4 的数组
        integers = np.linspace(0, 4, 5)
        # 将整数数组转换为与 p 的维度相同的形状
        integers = integers.reshape(integers.shape + (1,) * p.ndim)
        # 调用 scipy 的 kv 函数，计算 b0 到 b4
        b0, b1, b2, b3, b4 = sc.kv(p + integers, t1)
        # 计算 r1 到 r4
        r1, r2, r3, r4 = (b / b0 for b in (b1, b2, b3, b4))

        # 计算 m: b * t2 * r1
        m = b * t2 * r1
        # 计算 v
        v = (
            t2 * r1 + np.float_power(b, 2) * np.float_power(t2, 2) *
            (r2 - np.float_power(r1, 2))
        )
        # 计算 m3e
        m3e = (
            np.float_power(b, 3) * np.float_power(t2, 3) *
            (r3 - 3 * b2 * b1 * np.float_power(b0, -2) +
             2 * np.float_power(r1, 3)) +
            3 * b * np.float_power(t2, 2) *
            (r2 - np.float_power(r1, 2))
        )
        # 计算 s: m3e * v^(-3/2)
        s = m3e * np.float_power(v, - 3 / 2)
        # 计算 m4e
        m4e = (
            np.float_power(b, 4) * np.float_power(t2, 4) *
            (r4 - 4 * b3 * b1 * np.float_power(b0, - 2) +
             6 * b2 * np.float_power(b1, 2) * np.float_power(b0, - 3) -
             3 * np.float_power(r1, 4)) +
            np.float_power(b, 2) * np.float_power(t2, 3) *
            (6 * r3 - 12 * b2 * b1 * np.float_power(b0, - 2) +
             6 * np.float_power(r1, 3)) +
            3 * np.float_power(t2, 2) * r2
        )
        # 计算 k: m4e * v^(-2) - 3
        k = m4e * np.float_power(v, -2) - 3

        # 返回计算得到的统计量 m, v, s, k
        return m, v, s, k
# 创建一个名为 genhyperbolic 的全局变量，并调用 genhyperbolic_gen 函数初始化
genhyperbolic = genhyperbolic_gen(name='genhyperbolic')


class gompertz_gen(rv_continuous):
    r"""A Gompertz (or truncated Gumbel) continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `gompertz` is:

    .. math::

        f(x, c) = c \exp(x) \exp(-c (e^x-1))

    for :math:`x \ge 0`, :math:`c > 0`.

    `gompertz` takes ``c`` as a shape parameter for :math:`c`.

    %(after_notes)s

    %(example)s

    """
    def _shape_info(self):
        # 返回一个包含一个 `_ShapeInfo` 对象的列表，表示 `c` 参数的形状信息
        return [_ShapeInfo("c", False, (0, np.inf), (False, False))]

    def _pdf(self, x, c):
        # 计算 Gompertz 分布的概率密度函数，即 `gompertz.pdf(x, c) = c * exp(x) * exp(-c*(exp(x)-1))`
        return np.exp(self._logpdf(x, c))

    def _logpdf(self, x, c):
        # 计算 Gompertz 分布的对数概率密度函数
        return np.log(c) + x - c * sc.expm1(x)

    def _cdf(self, x, c):
        # 计算 Gompertz 分布的累积分布函数
        return -sc.expm1(-c * sc.expm1(x))

    def _ppf(self, q, c):
        # 计算 Gompertz 分布的分位点函数
        return sc.log1p(-1.0 / c * sc.log1p(-q))

    def _sf(self, x, c):
        # 计算 Gompertz 分布的生存函数
        return np.exp(-c * sc.expm1(x))

    def _isf(self, p, c):
        # 计算 Gompertz 分布的逆生存函数
        return sc.log1p(-np.log(p)/c)

    def _entropy(self, c):
        # 计算 Gompertz 分布的熵
        return 1.0 - np.log(c) - sc._ufuncs._scaled_exp1(c)/c


# 创建一个名为 gompertz 的全局变量，使用 gompertz_gen 类初始化
gompertz = gompertz_gen(a=0.0, name='gompertz')


def _average_with_log_weights(x, logweights):
    # 计算加权平均值，其中 logweights 是对数权重
    x = np.asarray(x)
    logweights = np.asarray(logweights)
    maxlogw = logweights.max()
    weights = np.exp(logweights - maxlogw)
    return np.average(x, weights=weights)


class gumbel_r_gen(rv_continuous):
    r"""A right-skewed Gumbel continuous random variable.

    %(before_notes)s

    See Also
    --------
    gumbel_l, gompertz, genextreme

    Notes
    -----
    The probability density function for `gumbel_r` is:

    .. math::

        f(x) = \exp(-(x + e^{-x}))

    The Gumbel distribution is sometimes referred to as a type I Fisher-Tippett
    distribution.  It is also related to the extreme value distribution,
    log-Weibull and Gompertz distributions.

    %(after_notes)s

    %(example)s

    """
    def _shape_info(self):
        # 返回一个空列表，表示 `gumbel_r` 没有额外的形状参数
        return []

    def _pdf(self, x):
        # 计算 Gumbel 右偏分布的概率密度函数，即 `gumbel_r.pdf(x) = exp(-(x + exp(-x)))`
        return np.exp(self._logpdf(x))

    def _logpdf(self, x):
        # 计算 Gumbel 右偏分布的对数概率密度函数
        return -x - np.exp(-x)

    def _cdf(self, x):
        # 计算 Gumbel 右偏分布的累积分布函数
        return np.exp(-np.exp(-x))

    def _logcdf(self, x):
        # 计算 Gumbel 右偏分布的对数累积分布函数
        return -np.exp(-x)

    def _ppf(self, q):
        # 计算 Gumbel 右偏分布的分位点函数
        return -np.log(-np.log(q))

    def _sf(self, x):
        # 计算 Gumbel 右偏分布的生存函数
        return -sc.expm1(-np.exp(-x))

    def _isf(self, p):
        # 计算 Gumbel 右偏分布的逆生存函数
        return -np.log(-np.log1p(-p))

    def _stats(self):
        # 返回 Gumbel 右偏分布的统计特性，包括均值、方差、偏度和峰度
        return _EULER, np.pi*np.pi/6.0, 12*np.sqrt(6)/np.pi**3 * _ZETA3, 12.0/5

    def _entropy(self):
        # 返回 Gumbel 右偏分布的熵
        return _EULER + 1.

    @_call_super_mom
    @inherit_docstring_from(rv_continuous)
# 创建一个名为 gumbel_r 的全局变量，使用 gumbel_r_gen 类初始化
gumbel_r = gumbel_r_gen(name='gumbel_r')


class gumbel_l_gen(rv_continuous):
    r"""A left-skewed Gumbel continuous random variable.

    %(before_notes)s

    See Also
    --------
    gumbel_r, gompertz, genextreme

    Notes
    -----
    The probability density function for `gumbel_r` is:

    .. math::

        f(x) = \exp(-(x + e^{-x}))

    The Gumbel distribution is sometimes referred to as a type I Fisher-Tippett
    distribution.  It is also related to the extreme value distribution,
    log-Weibull and Gompertz distributions.

    %(after_notes)s

    %(example)s

    """
    """
    The probability density function for `gumbel_l` is:

    .. math::

        f(x) = \exp(x - e^x)

    The Gumbel distribution is sometimes referred to as a type I Fisher-Tippett
    distribution.  It is also related to the extreme value distribution,
    log-Weibull and Gompertz distributions.

    %(after_notes)s

    %(example)s

    """

    # 定义一个概率密度函数为 Gumbel 分布的左尾版本 (`gumbel_l`)
    def _shape_info(self):
        return []  # 返回一个空列表

    # 定义概率密度函数 (PDF) `_pdf(x)`，其实现为 Gumbel 分布的 PDF: exp(x - exp(x))
    def _pdf(self, x):
        return np.exp(self._logpdf(x))

    # 定义对数概率密度函数 (log PDF) `_logpdf(x)`，其实现为 x - exp(x)
    def _logpdf(self, x):
        return x - np.exp(x)

    # 定义累积分布函数 (CDF) `_cdf(x)`，其实现为 -expm1(-exp(x))
    def _cdf(self, x):
        return -sc.expm1(-np.exp(x))

    # 定义反函数 `_ppf(q)`，其实现为 log(-log1p(-q))
    def _ppf(self, q):
        return np.log(-sc.log1p(-q))

    # 定义对数生存函数 (log survival function) `_logsf(x)`，其实现为 -exp(x)
    def _logsf(self, x):
        return -np.exp(x)

    # 定义生存函数 (SF) `_sf(x)`，其实现为 exp(-exp(x))
    def _sf(self, x):
        return np.exp(-np.exp(x))

    # 定义反生存函数 `_isf(x)`，其实现为 log(-log(x))
    def _isf(self, x):
        return np.log(-np.log(x))

    # 定义统计量函数 `_stats()`，返回 Gumbel 分布的统计量
    def _stats(self):
        return -_EULER, np.pi*np.pi/6.0, \
               -12*np.sqrt(6)/np.pi**3 * _ZETA3, 12.0/5

    # 定义熵函数 `_entropy()`，返回 Gumbel 分布的熵值
    def _entropy(self):
        return _EULER + 1.

    # 通过装饰器 `_call_super_mom` 和 `@inherit_docstring_from(rv_continuous)`
    # 继承 `rv_continuous` 的文档字符串，定义 `fit` 方法
    @inherit_docstring_from(rv_continuous)
    def fit(self, data, *args, **kwds):
        # 对于 Gumbel 分布左尾 (`gumbel_l`) 的拟合方法，可以通过以下步骤实现：
        # 1. 将数据取负值并传递给 `gumbel_r.fit`
        #    - 如果位置参数被固定，也应该取负值。
        # 2. 取结果位置参数的负值，保持尺度参数不变。
        # `gumbel_r.fit` 包含必要的输入检查。

        if kwds.get('floc') is not None:
            kwds['floc'] = -kwds['floc']  # 如果有指定的位置参数，取其负值
        # 调用 `gumbel_r.fit` 进行数据拟合
        loc_r, scale_r, = gumbel_r.fit(-np.asarray(data), *args, **kwds)
        # 返回取负后的位置参数和尺度参数
        return -loc_r, scale_r
# 创建一个 Gumbel 左分布的生成器 gumbel_l
gumbel_l = gumbel_l_gen(name='gumbel_l')

# 定义一个半柯西分布的连续随机变量生成器 halfcauchy_gen，继承自 rv_continuous 类
class halfcauchy_gen(rv_continuous):
    r"""A Half-Cauchy continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `halfcauchy` is:

    .. math::

        f(x) = \frac{2}{\pi (1 + x^2)}

    for :math:`x \ge 0`.

    %(after_notes)s

    %(example)s

    """
    
    # 返回分布形状信息的方法，这里返回一个空列表
    def _shape_info(self):
        return []

    # 概率密度函数（PDF）_pdf(x) = 2 / (pi * (1 + x**2))
    def _pdf(self, x):
        return 2.0 / np.pi / (1.0 + x*x)

    # 对数概率密度函数（log PDF）_logpdf(x) = log(2/pi) - log(1 + x**2)
    def _logpdf(self, x):
        return np.log(2.0 / np.pi) - sc.log1p(x*x)

    # 累积分布函数（CDF）_cdf(x) = 2/pi * arctan(x)
    def _cdf(self, x):
        return 2.0 / np.pi * np.arctan(x)

    # 百分点函数（PPF），即CDF的反函数，_ppf(q) = tan(pi/2 * q)
    def _ppf(self, q):
        return np.tan(np.pi/2 * q)

    # 生存函数（Survival function），1 - CDF的值，_sf(x) = 2/pi * arctan(1/x)
    def _sf(self, x):
        return 2.0 / np.pi * np.arctan2(1, x)

    # 逆生存函数（Inverse survival function），即SF的反函数，_isf(p) = 1 / tan(pi * p / 2)
    def _isf(self, p):
        return 1.0 / np.tan(np.pi * p / 2)

    # 返回分布的统计特性，这里是无穷大和NaN
    def _stats(self):
        return np.inf, np.inf, np.nan, np.nan

    # 返回分布的熵值，_entropy() = log(2*pi)
    def _entropy(self):
        return np.log(2 * np.pi)

    # 重写 fit 方法，根据数据拟合分布参数 loc 和 scale
    @_call_super_mom
    @inherit_docstring_from(rv_continuous)
    def fit(self, data, *args, **kwds):
        # 如果设置了 superfit 参数为 True，则调用父类的 fit 方法
        if kwds.pop('superfit', False):
            return super().fit(data, *args, **kwds)

        # 检查并获取拟合输入的参数
        data, floc, fscale = _check_fit_input_parameters(self, data,
                                                         args, kwds)

        # 计算数据的最小值
        data_min = np.min(data)
        
        # 确定 loc 参数
        if floc is not None:
            if data_min < floc:
                # 如果数据中存在小于指定 loc 的值，则抛出异常
                raise FitDataError("halfcauchy", lower=floc, upper=np.inf)
            loc = floc
        else:
            # 如果未提供 loc 参数，则将 loc 设置为数据的最小值（MLE）
            loc = data_min

        # 寻找合适的 scale 参数
        def find_scale(loc, data):
            shifted_data = data - loc
            n = data.size
            shifted_data_squared = np.square(shifted_data)

            def fun_to_solve(scale):
                denominator = scale**2 + shifted_data_squared
                return 2 * np.sum(shifted_data_squared / denominator) - n

            small = np.finfo(1.0).tiny**0.5  # 避免下溢
            res = root_scalar(fun_to_solve, bracket=(small, np.max(shifted_data)))
            return res.root

        # 确定 scale 参数
        if fscale is not None:
            scale = fscale
        else:
            scale = find_scale(loc, data)

        return loc, scale


# 创建一个 halfcauchy_gen 实例，名为 halfcauchy，参数 a=0.0
halfcauchy = halfcauchy_gen(a=0.0, name='halfcauchy')


# 定义一个半逻辑分布的连续随机变量生成器 halflogistic_gen，继承自 rv_continuous 类
class halflogistic_gen(rv_continuous):
    r"""A half-logistic continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `halflogistic` is:

    .. math::

        f(x) = \frac{ 2 e^{-x} }{ (1+e^{-x})^2 }
             = \frac{1}{2} \text{sech}(x/2)^2

    for :math:`x \ge 0`.

    %(after_notes)s

    References
    ----------
    .. [1] Asgharzadeh et al (2011). "Comparisons of Methods of Estimation for the
           Half-Logistic Distribution". Selcuk J. Appl. Math. 93-108.
    def _shape_info(self):
        # 返回一个空列表，表示概率分布没有特定的形状信息
        return []

    def _pdf(self, x):
        # 返回半逻辑分布在给定点 x 处的概率密度函数值
        # halflogistic.pdf(x) = 2 * exp(-x) / (1+exp(-x))**2
        #                     = 1/2 * sech(x/2)**2
        return np.exp(self._logpdf(x))

    def _logpdf(self, x):
        # 返回半逻辑分布在给定点 x 处的对数概率密度函数值
        return np.log(2) - x - 2. * sc.log1p(np.exp(-x))

    def _cdf(self, x):
        # 返回半逻辑分布在给定点 x 处的累积分布函数值
        return np.tanh(x/2.0)

    def _ppf(self, q):
        # 返回半逻辑分布的分位点函数值，即给定概率值 q 时对应的 x
        return 2*np.arctanh(q)

    def _sf(self, x):
        # 返回半逻辑分布在给定点 x 处的生存函数值
        return 2 * sc.expit(-x)

    def _isf(self, q):
        # 返回半逻辑分布的逆生存函数值，即给定概率值 q 时对应的 x
        return _lazywhere(q < 0.5, (q, ),
                          lambda q: -sc.logit(0.5 * q),
                          f2=lambda q: 2*np.arctanh(1 - q))

    def _munp(self, n):
        # 返回半逻辑分布的 n 阶非中心矩
        if n == 1:
            return 2*np.log(2)
        if n == 2:
            return np.pi*np.pi/3.0
        if n == 3:
            return 9*_ZETA3  # 假设 _ZETA3 是一个预定义的常量
        if n == 4:
            return 7*np.pi**4 / 15.0
        return 2*(1-pow(2.0, 1-n))*sc.gamma(n+1)*sc.zeta(n, 1)

    def _entropy(self):
        # 返回半逻辑分布的熵值
        return 2-np.log(2)

    @_call_super_mom
    @inherit_docstring_from(rv_continuous)
    # 定义一个方法 `fit`，用于拟合数据
    def fit(self, data, *args, **kwds):
        # 如果关键字参数中包含 `superfit` 且其值为 `True`
        if kwds.pop('superfit', False):
            # 调用父类的 `fit` 方法并返回其结果
            return super().fit(data, *args, **kwds)

        # 检查拟合输入参数的有效性，并获取相关参数
        data, floc, fscale = _check_fit_input_parameters(self, data,
                                                         args, kwds)

        # 定义一个内部方法 `find_scale`，用于寻找尺度参数
        def find_scale(data, loc):
            # 尺度是一个固定点问题的解 ([1] 2.6)
            # 使用近似最大似然估计作为起始点 ([1] 3.1)
            n_observations = data.shape[0]
            sorted_data = np.sort(data, axis=0)
            p = np.arange(1, n_observations + 1)/(n_observations + 1)
            q = 1 - p
            pp1 = 1 + p
            alpha = p - 0.5 * q * pp1 * np.log(pp1 / q)
            beta = 0.5 * q * pp1
            sorted_data = sorted_data - loc
            B = 2 * np.sum(alpha[1:] * sorted_data[1:])
            C = 2 * np.sum(beta[1:] * sorted_data[1:]**2)
            # 初始猜测
            scale = ((B + np.sqrt(B**2 + 8 * n_observations * C))
                    /(4 * n_observations))

            # 固定点迭代器的相对容差
            rtol = 1e-8
            relative_residual = 1
            shifted_mean = sorted_data.mean()  # y_mean - y_min

            # 通过反复应用方程 (2.6) 找到固定点
            # 简化为
            # exp(-x) / (1 + exp(-x)) = 1 / (1 + exp(x))
            #                         = expit(-x))
            while relative_residual > rtol:
                sum_term = sorted_data * sc.expit(-sorted_data/scale)
                scale_new = shifted_mean - 2/n_observations * sum_term.sum()
                relative_residual = abs((scale - scale_new)/scale)
                scale = scale_new
            return scale

        # 数据的最小值
        data_min = np.min(data)
        # 如果已提供 `floc`，则使用指定的位置参数
        if floc is not None:
            if data_min < floc:
                # 存在小于指定位置的值
                raise FitDataError("halflogistic", lower=floc, upper=np.inf)
            loc = floc
        else:
            # 如果未提供，位置的最大似然估计是最小数据点
            loc = data_min

        # 尺度取决于位置
        scale = fscale if fscale is not None else find_scale(data, loc)

        # 返回位置和尺度参数
        return loc, scale
# 创建一个半正态分布的生成器 `halflogistic`，参数 `a=0.0` 表示分布的起始点为 0.0
halflogistic = halflogistic_gen(a=0.0, name='halflogistic')

# 定义一个半正态分布的生成器 `halfnorm_gen`，继承自连续随机变量 `rv_continuous`
class halfnorm_gen(rv_continuous):
    r"""A half-normal continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `halfnorm` is:

    .. math::

        f(x) = \sqrt{2/\pi} \exp(-x^2 / 2)

    for :math:`x >= 0`.

    `halfnorm` is a special case of `chi` with ``df=1``.

    %(after_notes)s

    %(example)s

    """
    
    # 定义私有方法 `_shape_info`，返回空列表
    def _shape_info(self):
        return []

    # 定义随机变量生成方法 `_rvs`，返回指定大小的标准正态分布的绝对值
    def _rvs(self, size=None, random_state=None):
        return abs(random_state.standard_normal(size=size))

    # 定义概率密度函数 `_pdf`，返回半正态分布的概率密度函数值
    def _pdf(self, x):
        # halfnorm.pdf(x) = sqrt(2/pi) * exp(-x**2/2)
        return np.sqrt(2.0/np.pi)*np.exp(-x*x/2.0)

    # 定义对数概率密度函数 `_logpdf`，返回半正态分布的对数概率密度函数值
    def _logpdf(self, x):
        return 0.5 * np.log(2.0/np.pi) - x*x/2.0

    # 定义累积分布函数 `_cdf`，返回半正态分布的累积分布函数值
    def _cdf(self, x):
        return sc.erf(x / np.sqrt(2))

    # 定义反函数 `_ppf`，返回给定概率值对应的分位点
    def _ppf(self, q):
        return _norm_ppf((1+q)/2.0)

    # 定义生存函数 `_sf`，返回半正态分布的生存函数值
    def _sf(self, x):
        return 2 * _norm_sf(x)

    # 定义反生存函数 `_isf`，返回给定生存函数概率值对应的分位点
    def _isf(self, p):
        return _norm_isf(p/2)

    # 定义统计量方法 `_stats`，返回半正态分布的统计特性
    def _stats(self):
        return (np.sqrt(2.0/np.pi),
                1-2.0/np.pi,
                np.sqrt(2)*(4-np.pi)/(np.pi-2)**1.5,
                8*(np.pi-3)/(np.pi-2)**2)

    # 定义熵方法 `_entropy`，返回半正态分布的熵值
    def _entropy(self):
        return 0.5*np.log(np.pi/2.0)+0.5

    # 使用装饰器调用超类方法 `_call_super_mom`，并继承文档字符串
    @inherit_docstring_from(rv_continuous)
    def fit(self, data, *args, **kwds):
        if kwds.pop('superfit', False):
            return super().fit(data, *args, **kwds)

        # 检查拟合输入参数，并返回有效的数据、位置和尺度
        data, floc, fscale = _check_fit_input_parameters(self, data,
                                                         args, kwds)

        # 获取数据中的最小值
        data_min = np.min(data)

        # 根据输入参数确定位置参数 loc
        if floc is not None:
            if data_min < floc:
                # 如果数据中有小于指定 loc 的值，抛出拟合数据错误
                raise FitDataError("halfnorm", lower=floc, upper=np.inf)
            loc = floc
        else:
            loc = data_min

        # 根据输入参数确定尺度参数 scale
        if fscale is not None:
            scale = fscale
        else:
            scale = stats.moment(data, order=2, center=loc)**0.5

        # 返回拟合结果的位置和尺度
        return loc, scale

# 创建一个半正态分布的生成器 `halfnorm`，参数 `a=0.0` 表示分布的起始点为 0.0
halfnorm = halfnorm_gen(a=0.0, name='halfnorm')

# 定义一个双曲正切连续随机变量生成器 `hypsecant_gen`，继承自连续随机变量 `rv_continuous`
class hypsecant_gen(rv_continuous):
    r"""A hyperbolic secant continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `hypsecant` is:

    .. math::

        f(x) = \frac{1}{\pi} \text{sech}(x)

    for a real number :math:`x`.

    %(after_notes)s

    %(example)s

    """
    
    # 定义私有方法 `_shape_info`，返回空列表
    def _shape_info(self):
        return []

    # 定义概率密度函数 `_pdf`，返回双曲正切分布的概率密度函数值
    def _pdf(self, x):
        # hypsecant.pdf(x) = 1/pi * sech(x)
        return 1.0/(np.pi*np.cosh(x))

    # 定义累积分布函数 `_cdf`，返回双曲正切分布的累积分布函数值
    def _cdf(self, x):
        return 2.0/np.pi*np.arctan(np.exp(x))

    # 定义反函数 `_ppf`，返回给定概率值对应的分位点
    def _ppf(self, q):
        return np.log(np.tan(np.pi*q/2.0))

    # 定义生存函数 `_sf`，返回双曲正切分布的生存函数值
    def _sf(self, x):
        return 2.0/np.pi*np.arctan(np.exp(-x))

    # 定义反生存函数 `_isf`，返回给定生存函数概率值对应的分位点
    def _isf(self, q):
        return -np.log(np.tan(np.pi*q/2.0))

    # 定义统计量方法 `_stats`，返回双曲正切分布的统计特性
    def _stats(self):
        return 0, np.pi*np.pi/4, 0, 2
    # 定义一个方法 `_entropy`，用于计算某种概率分布的熵
    def _entropy(self):
        # 返回以自然对数为底的 2 乘以圆周率的对数值
        return np.log(2*np.pi)
# 创建名为 hypsecant 的 Gauss 型超几何连续随机变量生成器实例
hypsecant = hypsecant_gen(name='hypsecant')


class gausshyper_gen(rv_continuous):
    r"""A Gauss hypergeometric continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `gausshyper` is:

    .. math::

        f(x, a, b, c, z) = C x^{a-1} (1-x)^{b-1} (1+zx)^{-c}

    for :math:`0 \le x \le 1`, :math:`a,b > 0`, :math:`c` a real number,
    :math:`z > -1`, and :math:`C = \frac{1}{B(a, b) F[2, 1](c, a; a+b; -z)}`.
    :math:`F[2, 1]` is the Gauss hypergeometric function
    `scipy.special.hyp2f1`.

    `gausshyper` takes :math:`a`, :math:`b`, :math:`c` and :math:`z` as shape
    parameters.

    %(after_notes)s

    References
    ----------
    .. [1] Armero, C., and M. J. Bayarri. "Prior Assessments for Prediction in
           Queues." *Journal of the Royal Statistical Society*. Series D (The
           Statistician) 43, no. 1 (1994): 139-53. doi:10.2307/2348939

    %(example)s

    """

    def _argcheck(self, a, b, c, z):
        # 确保参数满足特定条件：a > 0, b > 0, c 是实数, z > -1
        return (a > 0) & (b > 0) & (c == c) & (z > -1)

    def _shape_info(self):
        # 返回参数形状信息列表，包括参数名称、是否是扩展参数、取值范围和是否是半开区间
        ia = _ShapeInfo("a", False, (0, np.inf), (False, False))
        ib = _ShapeInfo("b", False, (0, np.inf), (False, False))
        ic = _ShapeInfo("c", False, (-np.inf, np.inf), (False, False))
        iz = _ShapeInfo("z", False, (-1, np.inf), (False, False))
        return [ia, ib, ic, iz]

    def _pdf(self, x, a, b, c, z):
        # 计算概率密度函数：使用特定的参数计算标准化常数，然后计算概率密度
        normalization_constant = sc.beta(a, b) * sc.hyp2f1(c, a, a + b, -z)
        return (1./normalization_constant * x**(a - 1.) * (1. - x)**(b - 1.0)
                / (1.0 + z*x)**c)

    def _munp(self, n, a, b, c, z):
        # 计算 n 阶原点矩：使用特定的参数计算比率，然后计算 n 阶原点矩
        fac = sc.beta(n+a, b) / sc.beta(a, b)
        num = sc.hyp2f1(c, a+n, a+b+n, -z)
        den = sc.hyp2f1(c, a, a+b, -z)
        return fac*num / den


# 创建名为 gausshyper 的 Gauss 型超几何连续随机变量生成器实例，设置初始参数 a=0.0, b=1.0
gausshyper = gausshyper_gen(a=0.0, b=1.0, name='gausshyper')


class invgamma_gen(rv_continuous):
    r"""An inverted gamma continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `invgamma` is:

    .. math::

        f(x, a) = \frac{x^{-a-1}}{\Gamma(a)} \exp(-\frac{1}{x})

    for :math:`x >= 0`, :math:`a > 0`. :math:`\Gamma` is the gamma function
    (`scipy.special.gamma`).

    `invgamma` takes ``a`` as a shape parameter for :math:`a`.

    `invgamma` is a special case of `gengamma` with ``c=-1``, and it is a
    different parameterization of the scaled inverse chi-squared distribution.
    Specifically, if the scaled inverse chi-squared distribution is
    parameterized with degrees of freedom :math:`\nu` and scaling parameter
    :math:`\tau^2`, then it can be modeled using `invgamma` with
    ``a=`` :math:`\nu/2` and ``scale=`` :math:`\nu \tau^2/2`.

    %(after_notes)s

    %(example)s

    """
    # 支持掩码设定为 rv_continuous 类的开放支持掩码
    _support_mask = rv_continuous._open_support_mask

    def _shape_info(self):
        # 返回参数形状信息列表，包括参数名称、是否是扩展参数、取值范围和是否是半开区间
        return [_ShapeInfo("c", False, (0, np.inf), (False, False))]
    def _pdf(self, x, a):
        # 计算逆伽马分布的概率密度函数值，公式为 invgamma.pdf(x, a) = x**(-a-1) / gamma(a) * exp(-1/x)
        return np.exp(self._logpdf(x, a))

    def _logpdf(self, x, a):
        # 计算逆伽马分布的对数概率密度函数值，公式为 -(a+1) * log(x) - gammaln(a) - 1.0/x
        return -(a+1) * np.log(x) - sc.gammaln(a) - 1.0/x

    def _cdf(self, x, a):
        # 计算逆伽马分布的累积分布函数值，使用 SciPy 中的 gammaincc(a, 1.0 / x)
        return sc.gammaincc(a, 1.0 / x)

    def _ppf(self, q, a):
        # 计算逆伽马分布的分位点函数值，使用 1.0 / gammainccinv(a, q)
        return 1.0 / sc.gammainccinv(a, q)

    def _sf(self, x, a):
        # 计算逆伽马分布的生存函数值，使用 gammainc(a, 1.0 / x)
        return sc.gammainc(a, 1.0 / x)

    def _isf(self, q, a):
        # 计算逆伽马分布的逆生存函数值，使用 1.0 / gammaincinv(a, q)
        return 1.0 / sc.gammaincinv(a, q)

    def _stats(self, a, moments='mvsk'):
        # 计算逆伽马分布的统计量，包括均值 m1、方差 m2、偏度 g1、峰度 g2
        m1 = _lazywhere(a > 1, (a,), lambda x: 1. / (x - 1.), np.inf)
        m2 = _lazywhere(a > 2, (a,), lambda x: 1. / (x - 1.)**2 / (x - 2.), np.inf)

        g1, g2 = None, None
        if 's' in moments:
            g1 = _lazywhere(
                a > 3, (a,),
                lambda x: 4. * np.sqrt(x - 2.) / (x - 3.), np.nan)
        if 'k' in moments:
            g2 = _lazywhere(
                a > 4, (a,),
                lambda x: 6. * (5. * x - 11.) / (x - 3.) / (x - 4.), np.nan)
        return m1, m2, g1, g2

    def _entropy(self, a):
        def regular(a):
            # 计算逆伽马分布的熵，正常情况下的计算方式
            h = a - (a + 1.0) * sc.psi(a) + sc.gammaln(a)
            return h

        def asymptotic(a):
            # 计算逆伽马分布的熵，渐近情况下的计算方式
            # gammaln(a) ~ a * ln(a) - a - 0.5 * ln(a) + 0.5 * ln(2 * pi)
            # psi(a) ~ ln(a) - 1 / (2 * a)
            h = ((1 - 3*np.log(a) + np.log(2) + np.log(np.pi))/2
                 + 2/3*a**-1. + a**-2./12 - a**-3./90 - a**-4./120)
            return h

        # 根据 a 的大小选择使用正常或者渐近计算方式
        h = _lazywhere(a >= 2e2, (a,), f=asymptotic, f2=regular)
        return h
invgamma = invgamma_gen(a=0.0, name='invgamma')

class invgauss_gen(rv_continuous):
    r"""An inverse Gaussian continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `invgauss` is:

    .. math::

        f(x; \mu) = \frac{1}{\sqrt{2 \pi x^3}}
                    \exp\left(-\frac{(x-\mu)^2}{2 \mu^2 x}\right)

    for :math:`x \ge 0` and :math:`\mu > 0`.

    `invgauss` takes ``mu`` as a shape parameter for :math:`\mu`.

    %(after_notes)s

    A common shape-scale parameterization of the inverse Gaussian distribution
    has density

    .. math::

        f(x; \nu, \lambda) = \sqrt{\frac{\lambda}{2 \pi x^3}}
                    \exp\left( -\frac{\lambda(x-\nu)^2}{2 \nu^2 x}\right)

    Using ``nu`` for :math:`\nu` and ``lam`` for :math:`\lambda`, this
    parameterization is equivalent to the one above with ``mu = nu/lam``,
    ``loc = 0``, and ``scale = lam``.

    %(example)s

    """
    _support_mask = rv_continuous._open_support_mask

    def _shape_info(self):
        # 返回描述分布形状的信息，包括参数名、是否是必需的、取值范围及其是否包含边界
        return [_ShapeInfo("mu", False, (0, np.inf), (False, False))]

    def _rvs(self, mu, size=None, random_state=None):
        # 生成随机变量，使用Wald分布进行模拟
        return random_state.wald(mu, 1.0, size=size)

    def _pdf(self, x, mu):
        # 概率密度函数，计算特定参数下的密度值
        return 1.0 / np.sqrt(2 * np.pi * x**3.0) * np.exp(-1.0 / (2 * x) * ((x - mu) / mu)**2)

    def _logpdf(self, x, mu):
        # 对数概率密度函数，计算特定参数下的对数密度值
        return -0.5 * np.log(2 * np.pi) - 1.5 * np.log(x) - ((x - mu) / mu)**2 / (2 * x)

    def _logcdf(self, x, mu):
        # 对数累积分布函数，计算特定参数下的对数累积分布值
        fac = 1 / np.sqrt(x)
        a = _norm_logcdf(fac * ((x / mu) - 1))
        b = 2 / mu + _norm_logcdf(-fac * ((x / mu) + 1))
        return a + np.log1p(np.exp(b - a))

    def _logsf(self, x, mu):
        # 对数生存函数，计算特定参数下的对数生存函数值
        fac = 1 / np.sqrt(x)
        a = _norm_logsf(fac * ((x / mu) - 1))
        b = 2 / mu + _norm_logcdf(-fac * (x + mu) / mu)
        return a + np.log1p(-np.exp(b - a))

    def _sf(self, x, mu):
        # 生存函数，计算特定参数下的生存函数值
        return np.exp(self._logsf(x, mu))

    def _cdf(self, x, mu):
        # 累积分布函数，计算特定参数下的累积分布函数值
        return np.exp(self._logcdf(x, mu))

    def _ppf(self, x, mu):
        # 百分点函数，计算特定参数下的百分点函数值
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            x, mu = np.broadcast_arrays(x, mu)
            ppf = scu._invgauss_ppf(x, mu, 1)
            i_wt = x > 0.5  # "wrong tail" - sometimes too inaccurate
            ppf[i_wt] = scu._invgauss_isf(1 - x[i_wt], mu[i_wt], 1)
            i_nan = np.isnan(ppf)
            ppf[i_nan] = super()._ppf(x[i_nan], mu[i_nan])
        return ppf
    # 定义一个方法 _isf，用于计算逆广义正态分布的逆累积分布函数
    def _isf(self, x, mu):
        # 忽略除法时可能出现的警告
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            # 广播输入的数组，使其具有相同的形状
            x, mu = np.broadcast_arrays(x, mu)
            # 计算逆累积分布函数
            isf = scu._invgauss_isf(x, mu, 1)
            # 标记 "wrong tail" 的情况，有时逆累积分布函数可能不够精确
            i_wt = x > 0.5  # "wrong tail" - sometimes too inaccurate
            # 对于标记为 "wrong tail" 的情况，使用累积分布函数的逆函数修正结果
            isf[i_wt] = scu._invgauss_ppf(1-x[i_wt], mu[i_wt], 1)
            # 标记结果中的 NaN 值，并使用超类的方法替换这些值
            i_nan = np.isnan(isf)
            isf[i_nan] = super()._isf(x[i_nan], mu[i_nan])
        return isf

    # 定义一个方法 _stats，用于计算逆广义正态分布的统计量
    def _stats(self, mu):
        return mu, mu**3.0, 3*np.sqrt(mu), 15*mu

    # 覆盖父类的 fit 方法，用于拟合数据到逆广义正态分布模型
    @inherit_docstring_from(rv_continuous)
    def fit(self, data, *args, **kwds):
        # 获取拟合方法，默认为最大似然估计
        method = kwds.get('method', 'mle')

        # 如果数据类型为 CensoredData 或者是 wald_gen 类型或者指定使用 mm 方法，调用父类的 fit 方法
        if (isinstance(data, CensoredData) or type(self) == wald_gen
                or method.lower() == 'mm'):
            return super().fit(data, *args, **kwds)

        # 检查和处理拟合输入参数
        data, fshape_s, floc, fscale = _check_fit_input_parameters(self, data,
                                                                   args, kwds)
        '''
        Source: Statistical Distributions, 3rd Edition. Evans, Hastings,
        and Peacock (2000), Page 121. Their shape parameter is equivalent to
        SciPy's with the conversion `fshape_s = fshape / scale`.

        MLE formulas are not used in 3 conditions:
        - `loc` is not fixed
        - `mu` is fixed
        These cases fall back on the superclass fit method.
        - `loc` is fixed but translation results in negative data raises
          a `FitDataError`.
        '''
        # 如果位置参数 floc 为 None 或者形状参数 fshape_s 不为 None，则调用父类的 fit 方法
        if floc is None or fshape_s is not None:
            return super().fit(data, *args, **kwds)
        # 如果数据中有任何小于 0 的值，抛出 FitDataError 异常
        elif np.any(data - floc < 0):
            raise FitDataError("invgauss", lower=0, upper=np.inf)
        else:
            # 将数据减去位置参数 floc
            data = data - floc
            # 计算新的形状参数估计值
            fshape_n = np.mean(data)
            # 如果尺度参数为 None，则根据公式计算尺度参数
            if fscale is None:
                fscale = len(data) / (np.sum(data ** -1 - fshape_n ** -1))
            # 根据新的形状参数估计值和尺度参数计算形状参数的标准化值
            fshape_s = fshape_n / fscale
        return fshape_s, floc, fscale

    # 定义一个方法 _entropy，用于计算逆广义正态分布的熵
    def _entropy(self, mu):
        """
        Ref.: https://moser-isi.ethz.ch/docs/papers/smos-2012-10.pdf (eq. 9)
        """
        # 计算熵的第一部分
        a = 1. + np.log(2 * np.pi) + 3 * np.log(mu)
        # 计算熵的第二部分，使用特定的函数 _scaled_exp1
        r = 2/mu
        b = sc._ufuncs._scaled_exp1(r)/r
        return 0.5 * a - 1.5 * b
# 创建一个名为invgauss的随机变量生成器对象，参数a默认为0.0，name为'invgauss'
invgauss = invgauss_gen(a=0.0, name='invgauss')

# 定义一个名为geninvgauss_gen的类，继承于rv_continuous类，表示广义逆高斯连续随机变量
class geninvgauss_gen(rv_continuous):
    r"""A Generalized Inverse Gaussian continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `geninvgauss` is:

    .. math::

        f(x, p, b) = x^{p-1} \exp(-b (x + 1/x) / 2) / (2 K_p(b))

    where ``x > 0``, `p` is a real number and ``b > 0``\([1]_).
    :math:`K_p` is the modified Bessel function of second kind of order `p`
    (`scipy.special.kv`).

    %(after_notes)s

    The inverse Gaussian distribution `stats.invgauss(mu)` is a special case of
    `geninvgauss` with ``p = -1/2``, ``b = 1 / mu`` and ``scale = mu``.

    Generating random variates is challenging for this distribution. The
    implementation is based on [2]_.

    References
    ----------
    .. [1] O. Barndorff-Nielsen, P. Blaesild, C. Halgreen, "First hitting time
       models for the generalized inverse gaussian distribution",
       Stochastic Processes and their Applications 7, pp. 49--54, 1978.

    .. [2] W. Hoermann and J. Leydold, "Generating generalized inverse Gaussian
       random variates", Statistics and Computing, 24(4), p. 547--557, 2014.

    %(example)s

    """
    
    # 定义_argcheck方法，用于验证参数p和b的有效性
    def _argcheck(self, p, b):
        return (p == p) & (b > 0)

    # 定义_shape_info方法，返回参数p和b的描述信息，用于参数验证和文档化
    def _shape_info(self):
        ip = _ShapeInfo("p", False, (-np.inf, np.inf), (False, False))
        ib = _ShapeInfo("b", False, (0, np.inf), (False, False))
        return [ip, ib]

    # 定义_logpdf方法，计算概率密度函数的对数值，避免大数值下的溢出
    def _logpdf(self, x, p, b):
        # 使用np.vectorize向量化logpdf_single函数，提高效率和处理能力
        def logpdf_single(x, p, b):
            return _stats.geninvgauss_logpdf(x, p, b)

        logpdf_single = np.vectorize(logpdf_single, otypes=[np.float64])

        z = logpdf_single(x, p, b)
        if np.isnan(z).any():
            # 若计算结果包含NaN值，发出警告信息并替换为NaN，以避免错误结果
            msg = ("Infinite values encountered in scipy.special.kve(p, b). "
                   "Values replaced by NaN to avoid incorrect results.")
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
        return z

    # 定义_pdf方法，计算概率密度函数的值，基于_logpdf方法以避免溢出
    def _pdf(self, x, p, b):
        return np.exp(self._logpdf(x, p, b))

    # 定义_cdf方法，计算累积分布函数的值
    def _cdf(self, x, *args):
        _a, _b = self._get_support(*args)

        def _cdf_single(x, *args):
            p, b = args
            # 使用LowLevelCallable.from_cython创建低级回调对象，调用Cython实现的_geninvgauss_pdf函数
            user_data = np.array([p, b], float).ctypes.data_as(ctypes.c_void_p)
            llc = LowLevelCallable.from_cython(_stats, '_geninvgauss_pdf',
                                               user_data)

            return integrate.quad(llc, _a, x)[0]

        _cdf_single = np.vectorize(_cdf_single, otypes=[np.float64])

        return _cdf_single(x, *args)
    # 计算给定 x, p, b 下的拟密度函数的对数（未归一化常数），用于 _rvs 方法
    def _logquasipdf(self, x, p, b):
        return _lazywhere(x > 0, (x, p, b),
                          lambda x, p, b: (p - 1)*np.log(x) - b*(x + 1/x)/2,
                          -np.inf)

    # 根据参数 p 和 b 的类型来确定使用单一值还是迭代生成输出
    def _rvs(self, p, b, size=None, random_state=None):
        if np.isscalar(p) and np.isscalar(b):
            # 如果 p 和 b 都是标量，使用 _rvs_scalar 方法生成输出
            out = self._rvs_scalar(p, b, size, random_state)
        elif p.size == 1 and b.size == 1:
            # 如果 p 和 b 都是长度为1的数组，使用 _rvs_scalar 方法生成输出
            out = self._rvs_scalar(p.item(), b.item(), size, random_state)
        else:
            # 当该方法被调用时，size 将是一个可能为空的整数元组。它不会是 None；
            # 如果 rvs() 被传入 size=None，那么 size 将是空元组 ()。

            # 使用广播将 p 和 b 广播为相同的形状。
            p, b = np.broadcast_arrays(p, b)
            # 现在 p 和 b 具有相同的形状。

            # shp 是每个参数组合关联的随机变量块的形状。
            # bc 是一个与 size 长度相同的元组。其中的值是布尔值。
            # 如果 bc[j] 为 True，则表示对于给定的广播参数组合，整个轴都被填充。
            shp, bc = _check_shape(p.shape, size)

            # numsamples 是每个输入参数组合要生成的随机变量总数。
            numsamples = int(np.prod(shp))

            # out 是将要返回的数组。它在下面的循环中被填充。
            out = np.empty(size)

            # 使用 np.nditer 迭代器遍历 p 和 b
            it = np.nditer([p, b],
                           flags=['multi_index'],
                           op_flags=[['readonly'], ['readonly']])
            while not it.finished:
                # 将迭代器的 multi_index 转换为 out 数组中的索引，
                # 在这里调用 _rvs_scalar() 存储结果。
                # 当 bc 为 True 时，使用完整切片；否则使用 it.multi_index 中的索引值。
                # len(it.multi_index) 可能小于 len(bc)，在这种情况下我们要对齐这两个序列以右对齐，
                # 因此循环变量 j 从 -len(size) 到 0。
                # 这不会导致 IndexError，因为在 bc[j] 会使得 it.multi_index[j] 引发 IndexError 的情况下，bc[j] 为 True。
                idx = tuple((it.multi_index[j] if not bc[j] else slice(None))
                            for j in range(-len(size), 0))
                out[idx] = self._rvs_scalar(it[0], it[1], numsamples,
                                            random_state).reshape(shp)
                it.iternext()

        # 如果 size 为 ()，将 out 转换为标量值返回。
        if size == ():
            out = out.item()
        return out
    # 根据 p 的值分情况处理，以避免灾难性的取消效应（参见[2]）
    def _mode(self, p, b):
        # 如果 p 小于 1，执行以下操作
        if p < 1:
            # 计算并返回修正后的模式值
            return b / (np.sqrt((p - 1)**2 + b**2) + 1 - p)
        else:
            # 如果 p 大于等于 1，执行以下操作
            # 计算并返回修正后的模式值
            return (np.sqrt((1 - p)**2 + b**2) - (1 - p)) / b

    # 计算非中心矩的值
    def _munp(self, n, p, b):
        # 计算非中心矩的分子值
        num = sc.kve(p + n, b)
        # 计算非中心矩的分母值
        denom = sc.kve(p, b)
        # 检查是否存在无穷大的值
        inf_vals = np.isinf(num) | np.isinf(denom)
        # 如果存在无穷大的值
        if inf_vals.any():
            # 发出警告，说明在使用 scipy.special.kve 进行计算时遇到了无穷大的值，
            # 将这些值替换为 NaN，以避免错误的结果
            msg = ("Infinite values encountered in the moment calculation "
                   "involving scipy.special.kve. Values replaced by NaN to "
                   "avoid incorrect results.")
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
            # 将非无穷大的值设为 NaN，并计算修正后的非中心矩值
            m = np.full_like(num, np.nan, dtype=np.float64)
            m[~inf_vals] = num[~inf_vals] / denom[~inf_vals]
        else:
            # 如果不存在无穷大的值，直接计算修正后的非中心矩值
            m = num / denom
        # 返回计算得到的非中心矩值
        return m
# 定义一个变量 geninvgauss，它是一个使用默认参数的 geninvgauss_gen 类的实例
geninvgauss = geninvgauss_gen(a=0.0, name="geninvgauss")


# 定义 norminvgauss_gen 类，继承自 rv_continuous 类，表示正态逆高斯连续随机变量
class norminvgauss_gen(rv_continuous):
    r"""A Normal Inverse Gaussian continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `norminvgauss` is:

    .. math::

        f(x, a, b) = \frac{a \, K_1(a \sqrt{1 + x^2})}{\pi \sqrt{1 + x^2}} \,
                     \exp(\sqrt{a^2 - b^2} + b x)

    where :math:`x` is a real number, the parameter :math:`a` is the tail
    heaviness and :math:`b` is the asymmetry parameter satisfying
    :math:`a > 0` and :math:`|b| <= a`.
    :math:`K_1` is the modified Bessel function of second kind
    (`scipy.special.k1`).

    %(after_notes)s

    A normal inverse Gaussian random variable `Y` with parameters `a` and `b`
    can be expressed as a normal mean-variance mixture:
    ``Y = b * V + sqrt(V) * X`` where `X` is ``norm(0,1)`` and `V` is
    ``invgauss(mu=1/sqrt(a**2 - b**2))``. This representation is used
    to generate random variates.

    Another common parametrization of the distribution (see Equation 2.1 in
    [2]_) is given by the following expression of the pdf:

    .. math::

        g(x, \alpha, \beta, \delta, \mu) =
        \frac{\alpha\delta K_1\left(\alpha\sqrt{\delta^2 + (x - \mu)^2}\right)}
        {\pi \sqrt{\delta^2 + (x - \mu)^2}} \,
        e^{\delta \sqrt{\alpha^2 - \beta^2} + \beta (x - \mu)}

    In SciPy, this corresponds to
    `a = alpha * delta, b = beta * delta, loc = mu, scale=delta`.

    References
    ----------
    .. [1] O. Barndorff-Nielsen, "Hyperbolic Distributions and Distributions on
           Hyperbolae", Scandinavian Journal of Statistics, Vol. 5(3),
           pp. 151-157, 1978.

    .. [2] O. Barndorff-Nielsen, "Normal Inverse Gaussian Distributions and
           Stochastic Volatility Modelling", Scandinavian Journal of
           Statistics, Vol. 24, pp. 1-13, 1997.

    %(example)s

    """
    # 定义支持掩码，使用默认的开放支持掩码
    _support_mask = rv_continuous._open_support_mask

    # 参数检查函数，确保参数 a > 0 且 |b| < a
    def _argcheck(self, a, b):
        return (a > 0) & (np.absolute(b) < a)

    # 形状信息函数，返回参数 a 和 b 的形状信息
    def _shape_info(self):
        ia = _ShapeInfo("a", False, (0, np.inf), (False, False))
        ib = _ShapeInfo("b", False, (-np.inf, np.inf), (False, False))
        return [ia, ib]

    # 拟合起始函数，返回适合拟合的起始参数值，默认为 (1, 0.5)
    def _fitstart(self, data):
        # 任意选取，但默认的 a = b = 1 不合适；分布要求 |b| < a
        return super()._fitstart(data, args=(1, 0.5))

    # 概率密度函数，计算正态逆高斯分布的概率密度函数值
    def _pdf(self, x, a, b):
        gamma = np.sqrt(a**2 - b**2)
        fac1 = a / np.pi
        sq = np.hypot(1, x)  # 减少溢出
        return fac1 * sc.k1e(a * sq) * np.exp(b*x - a*sq + gamma) / sq
    # 定义一个方法 _sf，用于计算逆高斯分布的生存函数（Survival Function）
    def _sf(self, x, a, b):
        # 如果 x 是标量，则 a 和 b 也是标量
        if np.isscalar(x):
            # 使用 integrate.quad 对 self._pdf 在 [x, ∞) 上进行数值积分，返回生存函数的值
            return integrate.quad(self._pdf, x, np.inf, args=(a, b))[0]
        else:
            # 将 a 和 b 至少转换为一维数组
            a = np.atleast_1d(a)
            b = np.atleast_1d(b)
            result = []
            # 遍历 x, a, b 的组合，计算每个组合下的生存函数值
            for (x0, a0, b0) in zip(x, a, b):
                result.append(integrate.quad(self._pdf, x0, np.inf,
                                             args=(a0, b0))[0])
            # 返回结果数组
            return np.array(result)

    # 定义一个方法 _isf，用于计算逆高斯分布的逆生存函数（Inverse Survival Function）
    def _isf(self, q, a, b):
        # 定义内部函数 _isf_scalar，用于处理标量输入的逆生存函数计算
        def _isf_scalar(q, a, b):

            # 定义方程 eq(x, a, b, q)，用于求解 isf(x, a, b) = q
            def eq(x, a, b, q):
                # 返回生存函数值与 q 的差值
                return self._sf(x, a, b) - q

            # 计算均值 xm，并计算在该点的生存函数值 em
            xm = self.mean(a, b)
            em = eq(xm, a, b, q)
            # 如果 em 等于 0，则直接返回 xm
            if em == 0:
                return xm
            # 如果 em 大于 0，则向右扩展寻找根的区间
            if em > 0:
                delta = 1
                left = xm
                right = xm + delta
                while eq(right, a, b, q) > 0:
                    delta = 2*delta
                    right = xm + delta
            else:
                # em 小于 0
                delta = 1
                right = xm
                left = xm - delta
                while eq(left, a, b, q) < 0:
                    delta = 2*delta
                    left = xm - delta
            # 使用 optimize.brentq 在 [left, right] 区间内求解方程 eq(x, a, b, q)=0
            result = optimize.brentq(eq, left, right, args=(a, b, q),
                                     xtol=self.xtol)
            return result

        # 如果 q 是标量，则直接调用 _isf_scalar 计算逆生存函数值
        if np.isscalar(q):
            return _isf_scalar(q, a, b)
        else:
            # 如果 q 是数组，则对每个元素调用 _isf_scalar 计算逆生存函数值
            result = []
            for (q0, a0, b0) in zip(q, a, b):
                result.append(_isf_scalar(q0, a0, b0))
            return np.array(result)

    # 定义一个方法 _rvs，用于生成逆高斯分布的随机样本
    def _rvs(self, a, b, size=None, random_state=None):
        # 计算参数 gamma
        gamma = np.sqrt(a**2 - b**2)
        # 生成符合逆高斯分布（invgauss）的随机样本
        ig = invgauss.rvs(mu=1/gamma, size=size, random_state=random_state)
        # 返回经过变换的随机样本
        return b * ig + np.sqrt(ig) * norm.rvs(size=size,
                                               random_state=random_state)

    # 定义一个方法 _stats，用于计算逆高斯分布的统计特性
    def _stats(self, a, b):
        # 计算参数 gamma
        gamma = np.sqrt(a**2 - b**2)
        # 计算均值、方差、偏度和峰度
        mean = b / gamma
        variance = a**2 / gamma**3
        skewness = 3.0 * b / (a * np.sqrt(gamma))
        kurtosis = 3.0 * (1 + 4 * b**2 / a**2) / gamma
        # 返回计算结果
        return mean, variance, skewness, kurtosis
# 定义名为 `norminvgauss` 的变量，其值为调用 `norminvgauss_gen` 函数并传入 `name="norminvgauss"` 参数后的结果
norminvgauss = norminvgauss_gen(name="norminvgauss")


class invweibull_gen(rv_continuous):
    """An inverted Weibull continuous random variable.

    This distribution is also known as the Fréchet distribution or the
    type II extreme value distribution.

    %(before_notes)s

    Notes
    -----
    The probability density function for `invweibull` is:

    .. math::

        f(x, c) = c x^{-c-1} \\exp(-x^{-c})

    for :math:`x > 0`, :math:`c > 0`.

    `invweibull` takes ``c`` as a shape parameter for :math:`c`.

    %(after_notes)s

    References
    ----------
    F.R.S. de Gusmao, E.M.M Ortega and G.M. Cordeiro, "The generalized inverse
    Weibull distribution", Stat. Papers, vol. 52, pp. 591-619, 2011.

    %(example)s

    """
    _support_mask = rv_continuous._open_support_mask

    # 定义 `_shape_info` 方法，返回包含参数信息的列表
    def _shape_info(self):
        return [_ShapeInfo("c", False, (0, np.inf), (False, False))]

    # 定义 `_pdf` 方法，计算概率密度函数 `pdf` 的值
    def _pdf(self, x, c):
        # invweibull.pdf(x, c) = c * x**(-c-1) * exp(-x**(-c))
        xc1 = np.power(x, -c - 1.0)
        xc2 = np.power(x, -c)
        xc2 = np.exp(-xc2)
        return c * xc1 * xc2

    # 定义 `_cdf` 方法，计算累积分布函数 `cdf` 的值
    def _cdf(self, x, c):
        xc1 = np.power(x, -c)
        return np.exp(-xc1)

    # 定义 `_sf` 方法，计算生存函数 `sf` 的值
    def _sf(self, x, c):
        return -np.expm1(-x**-c)

    # 定义 `_ppf` 方法，计算百分位点函数 `ppf` 的值
    def _ppf(self, q, c):
        return np.power(-np.log(q), -1.0/c)

    # 定义 `_isf` 方法，计算逆生存函数 `isf` 的值
    def _isf(self, p, c):
        return (-np.log1p(-p))**(-1/c)

    # 定义 `_munp` 方法，计算原点矩 `munp` 的值
    def _munp(self, n, c):
        return sc.gamma(1 - n / c)

    # 定义 `_entropy` 方法，计算熵 `entropy` 的值
    def _entropy(self, c):
        return 1+_EULER + _EULER / c - np.log(c)

    # 定义 `_fitstart` 方法，为参数估计提供起始值
    def _fitstart(self, data, args=None):
        # invweibull requires c > 1 for the first moment to exist, so use 2.0
        args = (2.0,) if args is None else args
        return super()._fitstart(data, args=args)


# 定义名为 `invweibull` 的变量，其值为调用 `invweibull_gen` 类并传入 `a=0, name='invweibull'` 参数后的结果
invweibull = invweibull_gen(a=0, name='invweibull')


class jf_skew_t_gen(rv_continuous):
    r"""Jones and Faddy skew-t distribution.

    %(before_notes)s

    Notes
    -----
    The probability density function for `jf_skew_t` is:

    .. math::

        f(x; a, b) = C_{a,b}^{-1}
                    \left(1+\frac{x}{\left(a+b+x^2\right)^{1/2}}\right)^{a+1/2}
                    \left(1-\frac{x}{\left(a+b+x^2\right)^{1/2}}\right)^{b+1/2}

    for real numbers :math:`a>0` and :math:`b>0`, where
    :math:`C_{a,b} = 2^{a+b-1}B(a,b)(a+b)^{1/2}`, and :math:`B` denotes the
    beta function (`scipy.special.beta`).

    When :math:`a<b`, the distribution is negatively skewed, and when
    :math:`a>b`, the distribution is positively skewed. If :math:`a=b`, then
    we recover the `t` distribution with :math:`2a` degrees of freedom.

    `jf_skew_t` takes :math:`a` and :math:`b` as shape parameters.

    %(after_notes)s

    References
    ----------
    .. [1] M.C. Jones and M.J. Faddy. "A skew extension of the t distribution,
           with applications" *Journal of the Royal Statistical Society*.
           Series B (Statistical Methodology) 65, no. 1 (2003): 159-174.
           :doi:`10.1111/1467-9868.00378`

    """
    def _shape_info(self):
        # 创建形状信息对象ia，表示参数a，非旋转，长度范围为(0, 无穷)，无边界条件
        ia = _ShapeInfo("a", False, (0, np.inf), (False, False))
        # 创建形状信息对象ib，表示参数b，非旋转，长度范围为(0, 无穷)，无边界条件
        ib = _ShapeInfo("b", False, (0, np.inf), (False, False))
        # 返回包含ia和ib对象的列表
        return [ia, ib]

    def _pdf(self, x, a, b):
        # 计算c值，代表常数系数
        c = 2 ** (a + b - 1) * sc.beta(a, b) * np.sqrt(a + b)
        # 计算d1值，代表PDF的分子部分
        d1 = (1 + x / np.sqrt(a + b + x ** 2)) ** (a + 0.5)
        # 计算d2值，代表PDF的分母部分
        d2 = (1 - x / np.sqrt(a + b + x ** 2)) ** (b + 0.5)
        # 返回概率密度函数的计算结果
        return d1 * d2 / c

    def _rvs(self, a, b, size=None, random_state=None):
        # 从指定的随机状态中生成beta分布的随机变量d1
        d1 = random_state.beta(a, b, size)
        # 计算d2值，代表随机变量的转换结果
        d2 = (2 * d1 - 1) * np.sqrt(a + b)
        # 计算d3值，代表随机变量的缩放因子
        d3 = 2 * np.sqrt(d1 * (1 - d1))
        # 返回生成的随机变量
        return d2 / d3

    def _cdf(self, x, a, b):
        # 计算y值，代表CDF的中间结果
        y = (1 + x / np.sqrt(a + b + x ** 2)) * 0.5
        # 返回累积分布函数的计算结果
        return sc.betainc(a, b, y)

    def _ppf(self, q, a, b):
        # 计算d1值，代表百分点函数的初始结果
        d1 = beta.ppf(q, a, b)
        # 计算d2值，代表百分点函数的转换结果
        d2 = (2 * d1 - 1) * np.sqrt(a + b)
        # 计算d3值，代表百分点函数的缩放因子
        d3 = 2 * np.sqrt(d1 * (1 - d1))
        # 返回百分点函数的计算结果
        return d2 / d3

    def _munp(self, n, a, b):
        """Returns the n-th moment(s) where all the following hold:

        - n >= 0
        - a > n / 2
        - b > n / 2

        The result is np.nan in all other cases.
        """
        def nth_moment(n_k, a_k, b_k):
            """Computes E[T^(n_k)] where T is skew-t distributed with
            parameters a_k and b_k.
            """
            # 计算分子部分的系数num
            num = (a_k + b_k) ** (0.5 * n_k)
            # 计算分母部分的系数denom
            denom = 2 ** n_k * sc.beta(a_k, b_k)

            # 生成指数数组indices
            indices = np.arange(n_k + 1)
            # 根据indices的奇偶性生成符号数组sgn
            sgn = np.where(indices % 2 > 0, -1, 1)
            # 计算系数数组d
            d = sc.beta(a_k + 0.5 * n_k - indices, b_k - 0.5 * n_k + indices)
            # 计算所有项的总和
            sum_terms = sc.comb(n_k, indices) * sgn * d

            # 返回期望的n次方的计算结果
            return num / denom * sum_terms.sum()

        # 检查是否满足计算条件的布尔掩码
        nth_moment_valid = (a > 0.5 * n) & (b > 0.5 * n) & (n >= 0)
        # 使用_lazywhere函数计算期望的n次方，如果条件不满足则返回np.nan
        return _lazywhere(
            nth_moment_valid,
            (n, a, b),
            np.vectorize(nth_moment, otypes=[np.float64]),
            np.nan,
        )
jf_skew_t = jf_skew_t_gen(name='jf_skew_t')

class johnsonsb_gen(rv_continuous):
    r"""A Johnson SB continuous random variable.

    %(before_notes)s

    See Also
    --------
    johnsonsu

    Notes
    -----
    The probability density function for `johnsonsb` is:

    .. math::

        f(x, a, b) = \frac{b}{x(1-x)}  \phi(a + b \log \frac{x}{1-x} )

    where :math:`x`, :math:`a`, and :math:`b` are real scalars; :math:`b > 0`
    and :math:`x \in [0,1]`.  :math:`\phi` is the pdf of the normal
    distribution.

    `johnsonsb` takes :math:`a` and :math:`b` as shape parameters.

    %(after_notes)s

    %(example)s

    """
    _support_mask = rv_continuous._open_support_mask

    def _argcheck(self, a, b):
        # 检查参数a和b是否合法，要求b大于0并且a不是NaN
        return (b > 0) & (a == a)

    def _shape_info(self):
        # 返回参数a和b的信息，定义其取值范围和是否有界
        ia = _ShapeInfo("a", False, (-np.inf, np.inf), (False, False))
        ib = _ShapeInfo("b", False, (0, np.inf), (False, False))
        return [ia, ib]

    def _pdf(self, x, a, b):
        # 计算概率密度函数，对应于Johnson SB分布的公式
        # johnsonsb.pdf(x, a, b) = b / (x*(1-x)) * phi(a + b * log(x/(1-x)))
        trm = _norm_pdf(a + b*sc.logit(x))
        return b*1.0/(x*(1-x))*trm

    def _cdf(self, x, a, b):
        # 计算累积分布函数，使用标准正态分布的累积分布函数
        return _norm_cdf(a + b*sc.logit(x))

    def _ppf(self, q, a, b):
        # 计算百分位点函数，使用标准正态分布的逆累积分布函数
        return sc.expit(1.0 / b * (_norm_ppf(q) - a))

    def _sf(self, x, a, b):
        # 计算生存函数，使用标准正态分布的生存函数
        return _norm_sf(a + b*sc.logit(x))

    def _isf(self, q, a, b):
        # 计算逆生存函数，使用标准正态分布的逆生存函数
        return sc.expit(1.0 / b * (_norm_isf(q) - a))


johnsonsb = johnsonsb_gen(a=0.0, b=1.0, name='johnsonsb')


class johnsonsu_gen(rv_continuous):
    r"""A Johnson SU continuous random variable.

    %(before_notes)s

    See Also
    --------
    johnsonsb

    Notes
    -----
    The probability density function for `johnsonsu` is:

    .. math::

        f(x, a, b) = \frac{b}{\sqrt{x^2 + 1}}
                     \phi(a + b \log(x + \sqrt{x^2 + 1}))

    where :math:`x`, :math:`a`, and :math:`b` are real scalars; :math:`b > 0`.
    :math:`\phi` is the pdf of the normal distribution.

    `johnsonsu` takes :math:`a` and :math:`b` as shape parameters.

    The first four central moments are calculated according to the formulas
    in [1]_.

    %(after_notes)s

    References
    ----------
    .. [1] Taylor Enterprises. "Johnson Family of Distributions".
       https://variation.com/wp-content/distribution_analyzer_help/hs126.htm

    %(example)s

    """
    def _argcheck(self, a, b):
        # 检查参数a和b是否合法，要求b大于0并且a不是NaN
        return (b > 0) & (a == a)

    def _shape_info(self):
        # 返回参数a和b的信息，定义其取值范围和是否有界
        ia = _ShapeInfo("a", False, (-np.inf, np.inf), (False, False))
        ib = _ShapeInfo("b", False, (0, np.inf), (False, False))
        return [ia, ib]

    def _pdf(self, x, a, b):
        # 计算概率密度函数，对应于Johnson SU分布的公式
        # johnsonsu.pdf(x, a, b) = b / sqrt(x**2 + 1) * phi(a + b * log(x + sqrt(x**2 + 1)))
        x2 = x*x
        trm = _norm_pdf(a + b * np.arcsinh(x))
        return b*1.0/np.sqrt(x2+1.0)*trm

    def _cdf(self, x, a, b):
        # 计算累积分布函数，使用标准正态分布的累积分布函数
        return _norm_cdf(a + b * np.arcsinh(x))
    # 使用逆正态分布的分位数来计算正弦超b分布中的分位函数
    def _ppf(self, q, a, b):
        return np.sinh((_norm_ppf(q) - a) / b)

    # 使用正弦超b分布中的累积分布函数来计算生存函数
    def _sf(self, x, a, b):
        return _norm_sf(a + b * np.arcsinh(x))

    # 使用逆正态分布的分位数来计算正弦超b分布中的逆生存函数
    def _isf(self, x, a, b):
        return np.sinh((_norm_isf(x) - a) / b)

    # 计算正弦超b分布的统计量，支持指定的矩（'mv'：期望和方差）
    def _stats(self, a, b, moments='mv'):
        # Naive implementation of first and second moment to address gh-18071.
        # https://variation.com/wp-content/distribution_analyzer_help/hs126.htm
        # Numerical improvements left to future enhancements.
        
        # 初始化期望(mu)，期望的平方(mu2)，偏度(g1)，峰度(g2)
        mu, mu2, g1, g2 = None, None, None, None

        # 计算一些常数和表达式
        bn2 = b**-2.
        expbn2 = np.exp(bn2)
        a_b = a / b

        # 计算期望（'m' in moments表示需要期望）
        if 'm' in moments:
            mu = -expbn2**0.5 * np.sinh(a_b)
        
        # 计算方差（'v' in moments表示需要方差）
        if 'v' in moments:
            mu2 = 0.5 * sc.expm1(bn2) * (expbn2 * np.cosh(2 * a_b) + 1)
        
        # 计算偏度（'s' in moments表示需要偏度）
        if 's' in moments:
            t1 = expbn2**.5 * sc.expm1(bn2)**0.5
            t2 = 3 * np.sinh(a_b)
            t3 = expbn2 * (expbn2 + 2) * np.sinh(3 * a_b)
            denom = np.sqrt(2) * (1 + expbn2 * np.cosh(2 * a_b))**(3 / 2)
            g1 = -t1 * (t2 + t3) / denom
        
        # 计算峰度（'k' in moments表示需要峰度）
        if 'k' in moments:
            t1 = 3 + 6 * expbn2
            t2 = 4 * expbn2**2 * (expbn2 + 2) * np.cosh(2 * a_b)
            t3 = expbn2**2 * np.cosh(4 * a_b)
            t4 = -3 + 3 * expbn2**2 + 2 * expbn2**3 + expbn2**4
            denom = 2 * (1 + expbn2 * np.cosh(2 * a_b))**2
            g2 = (t1 + t2 + t3 * t4) / denom - 3
        
        # 返回计算得到的期望、期望平方、偏度和峰度
        return mu, mu2, g1, g2
# 创建一个 Johnson Su 分布的生成器对象，用于生成符合 Johnson Su 分布的随机变量
johnsonsu = johnsonsu_gen(name='johnsonsu')

# 定义一个 Laplace 分布的生成器类，继承自 rv_continuous 类
class laplace_gen(rv_continuous):
    r"""A Laplace continuous random variable.

    %(before_notes)s

    Notes
    -----
    Laplace 分布的概率密度函数为

    .. math::

        f(x) = \frac{1}{2} \exp(-|x|)

    对于实数 :math:`x`。

    %(after_notes)s

    %(example)s

    """
    # 返回分布的形状信息，这里返回一个空列表
    def _shape_info(self):
        return []

    # 生成 Laplace 分布的随机样本，使用给定的随机数生成器 random_state
    def _rvs(self, size=None, random_state=None):
        return random_state.laplace(0, 1, size=size)

    # Laplace 分布的概率密度函数
    def _pdf(self, x):
        # laplace.pdf(x) = 1/2 * exp(-abs(x))
        return 0.5*np.exp(-abs(x))

    # Laplace 分布的累积分布函数
    def _cdf(self, x):
        with np.errstate(over='ignore'):
            return np.where(x > 0, 1.0 - 0.5*np.exp(-x), 0.5*np.exp(x))

    # Laplace 分布的生存函数（1 - CDF）
    def _sf(self, x):
        # 通过对称性得到生存函数
        return self._cdf(-x)

    # Laplace 分布的百分点函数（CDF 的逆函数）
    def _ppf(self, q):
        return np.where(q > 0.5, -np.log(2*(1-q)), np.log(2*q))

    # Laplace 分布的逆生存函数（1 - SF 的逆函数）
    def _isf(self, q):
        # 通过对称性得到逆生存函数
        return -self._ppf(q)

    # Laplace 分布的统计量，这里返回期望、方差、偏度和峰度
    def _stats(self):
        return 0, 2, 0, 3

    # Laplace 分布的熵
    def _entropy(self):
        return np.log(2)+1

    # 替换父类的文档字符串中的特定部分，并调用其超类的方法
    @_call_super_mom
    @replace_notes_in_docstring(rv_continuous, notes="""\
        This function uses explicit formulas for the maximum likelihood
        estimation of the Laplace distribution parameters, so the keyword
        arguments `loc`, `scale`, and `optimizer` are ignored.\n\n""")
    def fit(self, data, *args, **kwds):
        # 检查并处理拟合函数的输入参数
        data, floc, fscale = _check_fit_input_parameters(self, data,
                                                         args, kwds)

        # 参考文献：Statistical Distributions, 3rd Edition. Evans, Hastings,
        # and Peacock (2000), Page 124

        # 如果未指定位置参数，则取数据的中位数作为位置参数
        if floc is None:
            floc = np.median(data)

        # 如果未指定尺度参数，则计算数据的绝对偏差平均值作为尺度参数
        if fscale is None:
            fscale = (np.sum(np.abs(data - floc))) / len(data)

        # 返回估计的位置和尺度参数
        return floc, fscale


# 创建一个 Laplace 分布的生成器对象
laplace = laplace_gen(name='laplace')


# 定义一个不对称 Laplace 分布的生成器类，继承自 rv_continuous 类
class laplace_asymmetric_gen(rv_continuous):
    r"""An asymmetric Laplace continuous random variable.

    %(before_notes)s

    See Also
    --------
    laplace : Laplace distribution

    Notes
    -----
    不对称 Laplace 分布的概率密度函数为

    .. math::

       f(x, \kappa) &= \frac{1}{\kappa+\kappa^{-1}}\exp(-x\kappa),\quad x\ge0\\
                    &= \frac{1}{\kappa+\kappa^{-1}}\exp(x/\kappa),\quad x<0\\

    对于 :math:`-\infty < x < \infty`, :math:`\kappa > 0`。

    `laplace_asymmetric` 以参数 `kappa` 作为形状参数。

    当 :math:`\kappa = 1` 时，它等同于 Laplace 分布。

    %(after_notes)s

    注意某些文献中的尺度参数与 SciPy 中的 `scale` 的倒数相对应。
    例如，[1]_ 中的参数化 :math:`\lambda = 1/2` 相当于 `scale = 2`。

    References
    ----------
    .. [1] "Asymmetric Laplace distribution", Wikipedia
            https://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution

    """
    """
    多变量和不对称拉普拉斯分布的概率密度函数和统计方法 [2]。

    %(example)s

    """

    # 返回关于形状参数的信息，这里仅返回一个 _ShapeInfo 对象的列表，描述参数 kappa
    def _shape_info(self):
        return [_ShapeInfo("kappa", False, (0, np.inf), (False, False))]

    # 计算拉普拉斯分布的概率密度函数
    def _pdf(self, x, kappa):
        # 调用 _logpdf 方法计算对数概率密度函数并返回其指数
        return np.exp(self._logpdf(x, kappa))

    # 计算拉普拉斯分布的对数概率密度函数
    def _logpdf(self, x, kappa):
        # 计算公式中的 kapinv
        kapinv = 1/kappa
        # 计算对数概率密度函数
        lPx = x * np.where(x >= 0, -kappa, kapinv)
        lPx -= np.log(kappa+kapinv)
        return lPx

    # 计算拉普拉斯分布的累积分布函数
    def _cdf(self, x, kappa):
        # 计算公式中的 kapinv 和 kappkapinv
        kapinv = 1/kappa
        kappkapinv = kappa+kapinv
        return np.where(x >= 0,
                        1 - np.exp(-x*kappa)*(kapinv/kappkapinv),
                        np.exp(x*kapinv)*(kappa/kappkapinv))

    # 计算拉普拉斯分布的生存函数（1 - CDF）
    def _sf(self, x, kappa):
        # 计算公式中的 kapinv 和 kappkapinv
        kapinv = 1/kappa
        kappkapinv = kappa+kapinv
        return np.where(x >= 0,
                        np.exp(-x*kappa)*(kapinv/kappkapinv),
                        1 - np.exp(x*kapinv)*(kappa/kappkapinv))

    # 计算拉普拉斯分布的百分位点函数（CDF 的逆函数）
    def _ppf(self, q, kappa):
        # 计算公式中的 kapinv 和 kappkapinv
        kapinv = 1/kappa
        kappkapinv = kappa+kapinv
        return np.where(q >= kappa/kappkapinv,
                        -np.log((1 - q)*kappkapinv*kappa)*kapinv,
                        np.log(q*kappkapinv/kappa)*kappa)

    # 计算拉普拉斯分布的逆生存函数（SF 的逆函数）
    def _isf(self, q, kappa):
        # 计算公式中的 kapinv 和 kappkapinv
        kapinv = 1/kappa
        kappkapinv = kappa+kapinv
        return np.where(q <= kapinv/kappkapinv,
                        -np.log(q*kappkapinv*kappa)*kapinv,
                        np.log((1 - q)*kappkapinv/kappa)*kappa)

    # 计算拉普拉斯分布的统计量：均值、方差、偏度、峰度
    def _stats(self, kappa):
        # 计算公式中的 kapinv
        kapinv = 1/kappa
        # 计算均值、方差、偏度、峰度
        mn = kapinv - kappa
        var = kapinv*kapinv + kappa*kappa
        g1 = 2.0*(1-np.power(kappa, 6))/np.power(1+np.power(kappa, 4), 1.5)
        g2 = 6.0*(1+np.power(kappa, 8))/np.power(1+np.power(kappa, 4), 2)
        return mn, var, g1, g2

    # 计算拉普拉斯分布的熵
    def _entropy(self, kappa):
        return 1 + np.log(kappa+1/kappa)
laplace_asymmetric = laplace_asymmetric_gen(name='laplace_asymmetric')

# 定义一个拉普拉斯不对称分布生成器的实例

def _check_fit_input_parameters(dist, data, args, kwds):
    if not isinstance(data, CensoredData):
        data = np.asarray(data)

    floc = kwds.get('floc', None)
    fscale = kwds.get('fscale', None)

    num_shapes = len(dist.shapes.split(",")) if dist.shapes else 0
    fshape_keys = []
    fshapes = []

    # 用户有多种选择来固定分布的形状参数，这里将其标准化为 'f' + 形状参数的编号。
    # 改编自 `_reduce_func` 在 `_distn_infrastructure.py` 中的实现:
    if dist.shapes:
        # 将形状参数以逗号分隔的形式提取出来，并标准化为以 'f' 开头的键名
        shapes = dist.shapes.replace(',', ' ').split()
        for j, s in enumerate(shapes):
            key = 'f' + str(j)
            names = [key, 'f' + s, 'fix_' + s]
            # 获取固定的拟合值
            val = _get_fixed_fit_value(kwds, names)
            fshape_keys.append(key)
            fshapes.append(val)
            if val is not None:
                kwds[key] = val

    # 检查是否有未知的关键字参数在 kwds 中
    known_keys = {'loc', 'scale', 'optimizer', 'method',
                  'floc', 'fscale', *fshape_keys}
    unknown_keys = set(kwds).difference(known_keys)
    if unknown_keys:
        # 如果有未知的关键字参数，则抛出 TypeError 异常
        raise TypeError(f"Unknown keyword arguments: {unknown_keys}.")

    if len(args) > num_shapes:
        # 如果位置参数数量超过形状参数的数量，则抛出 TypeError 异常
        raise TypeError("Too many positional arguments.")

    if None not in {floc, fscale, *fshapes}:
        # 如果所有参数（位置参数和固定的形状参数）都不为 None，则抛出 RuntimeError 异常
        raise RuntimeError("All parameters fixed. There is nothing to "
                           "optimize.")

    # 如果数据包含非有限值，则抛出 ValueError 异常
    uncensored = data._uncensor() if isinstance(data, CensoredData) else data
    if not np.isfinite(uncensored).all():
        raise ValueError("The data contains non-finite values.")

    # 返回结果，包括数据本身和所有的固定形状参数
    return (data, *fshapes, floc, fscale)


class levy_gen(rv_continuous):
    r"""A Levy continuous random variable.

    %(before_notes)s

    See Also
    --------
    levy_stable, levy_l

    Notes
    -----
    The probability density function for `levy` is:

    .. math::

        f(x) = \frac{1}{\sqrt{2\pi x^3}} \exp\left(-\frac{1}{2x}\right)

    for :math:`x > 0`.

    This is the same as the Levy-stable distribution with :math:`a=1/2` and
    :math:`b=1`.

    %(after_notes)s

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import levy
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1, 1)

    Calculate the first four moments:

    >>> mean, var, skew, kurt = levy.stats(moments='mvsk')

    Display the probability density function (``pdf``):

    >>> # `levy` is very heavy-tailed.
    >>> # To show a nice plot, let's cut off the upper 40 percent.
    >>> a, b = levy.ppf(0), levy.ppf(0.6)
    >>> x = np.linspace(a, b, 100)
    >>> ax.plot(x, levy.pdf(x),
    ...        'r-', lw=5, alpha=0.6, label='levy pdf')

# 定义一个 Levy 连续随机变量的生成器类，包含概率密度函数和示例用法
    """
    _support_mask = rv_continuous._open_support_mask

    def _shape_info(self):
        # 返回空列表，表示没有额外的形状信息
        return []

    def _pdf(self, x):
        # 返回对应于 x 的概率密度函数值
        # levy.pdf(x) = 1 / (x * sqrt(2*pi*x)) * exp(-1/(2*x))
        return 1 / np.sqrt(2*np.pi*x) / x * np.exp(-1/(2*x))

    def _cdf(self, x):
        # 返回对应于 x 的累积分布函数值
        # 相当于 2*norm.sf(np.sqrt(1/x))
        return sc.erfc(np.sqrt(0.5 / x))

    def _sf(self, x):
        # 返回对应于 x 的生存函数值
        return sc.erf(np.sqrt(0.5 / x))

    def _ppf(self, q):
        # 返回对应于累积分布概率 q 的百分点函数值
        # 相当于 1.0/(norm.isf(q/2)**2) 或者 0.5/(erfcinv(q)**2)
        val = _norm_isf(q/2)  # 内部函数调用，求解正态分布的逆函数值
        return 1.0 / (val * val)

    def _isf(self, p):
        # 返回对应于生存函数概率 p 的百分点函数值的倒数
        return 1/(2*sc.erfinv(p)**2)

    def _stats(self):
        # 返回分布的统计信息，无穷大表示不适用，NaN 表示未定义
        return np.inf, np.inf, np.nan, np.nan
    """
levy = levy_gen(a=0.0, name="levy")

class levy_l_gen(rv_continuous):
    r"""A left-skewed Levy continuous random variable.

    %(before_notes)s

    See Also
    --------
    levy, levy_stable

    Notes
    -----
    The probability density function for `levy_l` is:

    .. math::
        f(x) = \frac{1}{|x| \sqrt{2\pi |x|}} \exp{ \left(-\frac{1}{2|x|} \right)}

    for :math:`x < 0`.

    This is the same as the Levy-stable distribution with :math:`a=1/2` and
    :math:`b=-1`.

    %(after_notes)s

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import levy_l
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1, 1)

    Calculate the first four moments:

    >>> mean, var, skew, kurt = levy_l.stats(moments='mvsk')

    Display the probability density function (``pdf``):

    >>> # `levy_l` is very heavy-tailed.
    >>> # To show a nice plot, let's cut off the lower 40 percent.
    >>> a, b = levy_l.ppf(0.4), levy_l.ppf(1)
    >>> x = np.linspace(a, b, 100)
    >>> ax.plot(x, levy_l.pdf(x),
    ...        'r-', lw=5, alpha=0.6, label='levy_l pdf')

    Alternatively, the distribution object can be called (as a function)
    to fix the shape, location, and scale parameters. This returns a "frozen"
    RV object holding the given parameters fixed.

    Freeze the distribution and display the frozen ``pdf``:

    >>> rv = levy_l()
    >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

    Check accuracy of ``cdf`` and ``ppf``:

    >>> vals = levy_l.ppf([0.001, 0.5, 0.999])
    >>> np.allclose([0.001, 0.5, 0.999], levy_l.cdf(vals))
    True

    Generate random numbers:

    >>> r = levy_l.rvs(size=1000)

    And compare the histogram:

    >>> # manual binning to ignore the tail
    >>> bins = np.concatenate(([np.min(r)], np.linspace(a, b, 20)))
    >>> ax.hist(r, bins=bins, density=True, histtype='stepfilled', alpha=0.2)
    >>> ax.set_xlim([x[0], x[-1]])
    >>> ax.legend(loc='best', frameon=False)
    >>> plt.show()

    """
    _support_mask = rv_continuous._open_support_mask

    def _shape_info(self):
        return []

    def _pdf(self, x):
        # levy_l.pdf(x) = 1 / (abs(x) * sqrt(2*pi*abs(x))) * exp(-1/(2*abs(x)))
        ax = abs(x)
        return 1/np.sqrt(2*np.pi*ax)/ax*np.exp(-1/(2*ax))

    def _cdf(self, x):
        ax = abs(x)
        return 2 * _norm_cdf(1 / np.sqrt(ax)) - 1

    def _sf(self, x):
        ax = abs(x)
        return 2 * _norm_sf(1 / np.sqrt(ax))

    def _ppf(self, q):
        val = _norm_ppf((q + 1.0) / 2)
        return -1.0 / (val * val)

    def _isf(self, p):
        return -1/_norm_isf(p/2)**2

    def _stats(self):
        return np.inf, np.inf, np.nan, np.nan


levy_l = levy_l_gen(b=0.0, name="levy_l")


class logistic_gen(rv_continuous):
    r"""A logistic (or Sech-squared) continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `logistic` is:
    """
    定义 Logistic 分布的概率密度函数、累积分布函数及其相关方法。
    `logistic` 是带有 `c=1` 的 `genlogistic` 的特殊情况。

    注意，生存函数 (`logistic.sf`) 等同于描述费米子统计的 Fermi-Dirac 分布。

    %(after_notes)s

    %(example)s

    """

    # 返回空列表，表示 Logistic 分布没有额外的形状信息
    def _shape_info(self):
        return []

    # 生成 Logistic 分布的随机变量
    def _rvs(self, size=None, random_state=None):
        return random_state.logistic(size=size)

    # Logistic 分布的概率密度函数
    def _pdf(self, x):
        # logistic.pdf(x) = exp(-x) / (1+exp(-x))**2
        return np.exp(self._logpdf(x))

    # Logistic 分布的对数概率密度函数
    def _logpdf(self, x):
        y = -np.abs(x)
        return y - 2. * sc.log1p(np.exp(y))

    # Logistic 分布的累积分布函数
    def _cdf(self, x):
        return sc.expit(x)

    # Logistic 分布的对数累积分布函数
    def _logcdf(self, x):
        return sc.log_expit(x)

    # Logistic 分布的分位点函数
    def _ppf(self, q):
        return sc.logit(q)

    # Logistic 分布的生存函数
    def _sf(self, x):
        return sc.expit(-x)

    # Logistic 分布的对数生存函数
    def _logsf(self, x):
        return sc.log_expit(-x)

    # Logistic 分布的逆生存函数
    def _isf(self, q):
        return -sc.logit(q)

    # 返回 Logistic 分布的统计特征，包括均值、方差、偏度和峰度
    def _stats(self):
        return 0, np.pi*np.pi/3.0, 0, 6.0/5.0

    # 返回 Logistic 分布的熵
    def _entropy(self):
        # 参考维基百科关于 Logistic 分布的熵的定义
        return 2.0

    # 装饰器，调用超类方法 `_mom` 的装饰器
    @_call_super_mom
    @inherit_docstring_from(rv_continuous)
    def fit(self, data, *args, **kwds):
        # 如果参数中有'superfit'，则调用父类的fit方法
        if kwds.pop('superfit', False):
            return super().fit(data, *args, **kwds)

        # 检查并处理fit方法的输入参数
        data, floc, fscale = _check_fit_input_parameters(self, data,
                                                         args, kwds)
        n = len(data)

        # 根据数据提供的初步猜测，确定初始的loc和scale
        loc, scale = self._fitstart(data)
        # 如果用户提供了loc和scale参数，则覆盖初步猜测
        loc, scale = kwds.get('loc', loc), kwds.get('scale', scale)

        # 以下是位置和尺度参数的最大似然估计`a`和`b`，通过`func`中描述的两个方程的根
        # 参考：Statistical Distributions, 3rd Edition. Evans, Hastings, and Peacock (2000), Page 130

        # 定义dl_dloc函数，计算对loc的偏导数
        def dl_dloc(loc, scale=fscale):
            c = (data - loc) / scale
            return np.sum(sc.expit(c)) - n/2

        # 定义dl_dscale函数，计算对scale的偏导数
        def dl_dscale(scale, loc=floc):
            c = (data - loc) / scale
            return np.sum(c*np.tanh(c/2)) - n

        # 定义func函数，返回dl_dloc和dl_dscale的结果
        def func(params):
            loc, scale = params
            return dl_dloc(loc, scale), dl_dscale(scale, loc)

        # 根据不同情况使用optimize.root函数进行参数估计
        if fscale is not None and floc is None:
            res = optimize.root(dl_dloc, (loc,))
            loc = res.x[0]
            scale = fscale
        elif floc is not None and fscale is None:
            res = optimize.root(dl_dscale, (scale,))
            scale = res.x[0]
            loc = floc
        else:
            res = optimize.root(func, (loc, scale))
            loc, scale = res.x

        # 注意：gh-18176报告了一个bug，即报告的MLE具有`scale < 0`。为了修复这个bug，我们返回abs(scale)。
        # 这是安全的，因为`dl_dscale`和`dl_dloc`分别是`scale`的偶函数和奇函数，所以如果`-scale`是一个解，那么`scale`也是一个解。
        scale = abs(scale)
        # 如果参数估计成功，则返回(loc, scale)，否则调用父类的fit方法
        return ((loc, scale) if res.success
                else super().fit(data, *args, **kwds))
# 创建一个名为 logistic 的概率分布生成器对象，使用 logistic_gen 函数进行初始化
logistic = logistic_gen(name='logistic')

# 定义一个名为 loggamma_gen 的类，继承自 rv_continuous 类，表示一个对数伽马分布的连续随机变量
class loggamma_gen(rv_continuous):
    r"""A log gamma continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `loggamma` is:

    .. math::

        f(x, c) = \frac{\exp(c x - \exp(x))}
                       {\Gamma(c)}

    for all :math:`x, c > 0`. Here, :math:`\Gamma` is the
    gamma function (`scipy.special.gamma`).

    `loggamma` takes ``c`` as a shape parameter for :math:`c`.

    %(after_notes)s

    %(example)s

    """

    # 返回一个包含 ShapeInfo 对象的列表，用于描述分布的形状参数
    def _shape_info(self):
        return [_ShapeInfo("c", False, (0, np.inf), (False, False))]

    # 实现随机变量生成方法，基于给定的参数 c 和随机数生成器 random_state
    def _rvs(self, c, size=None, random_state=None):
        # 使用 gamma 分布的性质生成随机样本
        #    Gamma(c) ~ Gamma(c + 1)*U**(1/c),
        # 其中 U 是 [0, 1] 上的均匀分布随机数
        # 因此
        #    log(Gamma(c)) ~ log(Gamma(c + 1)) + log(U)/c
        # 使用这种方法生成样本比直接计算 log(Gamma(c)) 略慢，但当 c << 1 时，避免了精度损失
        return (np.log(random_state.gamma(c + 1, size=size))
                + np.log(random_state.uniform(size=size))/c)

    # 实现概率密度函数
    def _pdf(self, x, c):
        # 对数伽马分布的概率密度函数为 exp(c*x - exp(x) - gammaln(c)) / gamma(c)
        return np.exp(c*x - np.exp(x) - sc.gammaln(c))

    # 实现对数概率密度函数
    def _logpdf(self, x, c):
        return c*x - np.exp(x) - sc.gammaln(c)

    # 实现累积分布函数
    def _cdf(self, x, c):
        # 这个函数是 gammainc(c, exp(x))，其中 gammainc(c, z) 是正则化的不完全 gamma 函数
        # gammainc(c, z) 的级数展开的第一项是 z**c/Gamma(c+1)
        # 参考 Abramowitz & Stegun 的 6.5.29 式子，以及相关的符号和定义
        # 在 x 足够负时，exp(x) 将导致亚正规数，可能会失去精度
        # 我们首先计算表达式的对数，以允许除法中项的可能抵消，然后再求指数
        # 即
        #     exp(x)**c/Gamma(c+1) = exp(log(exp(x)**c/Gamma(c+1)))
        #                          = exp(c*x - gammaln(c+1))
        return _lazywhere(x < _LOGXMIN, (x, c),
                          lambda x, c: np.exp(c*x - sc.gammaln(c+1)),
                          f2=lambda x, c: sc.gammainc(c, np.exp(x)))
    def _ppf(self, q, c):
        # 当 g < _XMIN 时，反转在 _cdf() 的注释中给出的一项展开式。
        g = sc.gammaincinv(c, q)
        return _lazywhere(g < _XMIN, (g, q, c),
                          lambda g, q, c: (np.log(q) + sc.gammaln(c+1))/c,
                          f2=lambda g, q, c: np.log(g))

    def _sf(self, x, c):
        # 参见 _cdf() 中有关处理 x < _LOGXMIN 的注释。
        return _lazywhere(x < _LOGXMIN, (x, c),
                          lambda x, c: -np.expm1(c*x - sc.gammaln(c+1)),
                          f2=lambda x, c: sc.gammaincc(c, np.exp(x)))

    def _isf(self, q, c):
        # 当 g < _XMIN 时，反转在 _cdf() 的注释中给出的一项展开式的补集。
        g = sc.gammainccinv(c, q)
        return _lazywhere(g < _XMIN, (g, q, c),
                          lambda g, q, c: (np.log1p(-q) + sc.gammaln(c+1))/c,
                          f2=lambda g, q, c: np.log(g))

    def _stats(self, c):
        # 参见例如 "A Statistical Study of Log-Gamma Distribution", 由 Ping Shing Chan 撰写（麦克马斯特大学，1993年）。
        mean = sc.digamma(c)
        var = sc.polygamma(1, c)
        skewness = sc.polygamma(2, c) / np.power(var, 1.5)
        excess_kurtosis = sc.polygamma(3, c) / (var*var)
        return mean, var, skewness, excess_kurtosis

    def _entropy(self, c):
        def regular(c):
            # h 是通过 gammaln(c) - c * digamma(c) + c 计算得到的。
            h = sc.gammaln(c) - c * sc.digamma(c) + c
            return h

        def asymptotic(c):
            # 使用 gammaln 和 psi 的渐近展开式（参见 gh-18093）。
            term = -0.5*np.log(c) + c**-1./6 - c**-3./90 + c**-5./210
            h = norm._entropy() + term
            return h

        # 对于大于等于 45 的 c，使用渐近方法；否则使用正常方法。
        h = _lazywhere(c >= 45, (c, ), f=asymptotic, f2=regular)
        return h
loggamma = loggamma_gen(name='loggamma')


class loglaplace_gen(rv_continuous):
    r"""A log-Laplace continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `loglaplace` is:

    .. math::

        f(x, c) = \begin{cases}\frac{c}{2} x^{ c-1}  &\text{for } 0 < x < 1\\
                               \frac{c}{2} x^{-c-1}  &\text{for } x \ge 1
                  \end{cases}

    for :math:`c > 0`.

    `loglaplace` takes ``c`` as a shape parameter for :math:`c`.

    %(after_notes)s

    Suppose a random variable ``X`` follows the Laplace distribution with
    location ``a`` and scale ``b``.  Then ``Y = exp(X)`` follows the
    log-Laplace distribution with ``c = 1 / b`` and ``scale = exp(a)``.

    References
    ----------
    T.J. Kozubowski and K. Podgorski, "A log-Laplace growth rate model",
    The Mathematical Scientist, vol. 28, pp. 49-60, 2003.

    %(example)s

    """
    
    # 定义一个方法，返回参数形状信息的列表
    def _shape_info(self):
        return [_ShapeInfo("c", False, (0, np.inf), (False, False))]

    # 定义概率密度函数（PDF），根据参数 c 和 x 计算概率密度
    def _pdf(self, x, c):
        # loglaplace.pdf(x, c) = c / 2 * x**(c-1),   for 0 < x < 1
        #                      = c / 2 * x**(-c-1),  for x >= 1
        cd2 = c/2.0
        c = np.where(x < 1, c, -c)
        return cd2*x**(c-1)

    # 定义累积分布函数（CDF），根据参数 c 和 x 计算累积分布
    def _cdf(self, x, c):
        return np.where(x < 1, 0.5*x**c, 1-0.5*x**(-c))

    # 定义生存函数（SF），根据参数 c 和 x 计算生存函数
    def _sf(self, x, c):
        return np.where(x < 1, 1 - 0.5*x**c, 0.5*x**(-c))

    # 定义反函数（PPF），根据参数 c 和 q 计算百分点函数的逆
    def _ppf(self, q, c):
        return np.where(q < 0.5, (2.0*q)**(1.0/c), (2*(1.0-q))**(-1.0/c))

    # 定义逆生存函数（ISF），根据参数 c 和 q 计算逆生存函数
    def _isf(self, q, c):
        return np.where(q > 0.5, (2.0*(1.0 - q))**(1.0/c), (2*q)**(-1.0/c))

    # 定义原点矩（Moment），根据参数 c 和 n 计算原点矩
    def _munp(self, n, c):
        with np.errstate(divide='ignore'):
            c2, n2 = c**2, n**2
            return np.where(n2 < c2, c2 / (c2 - n2), np.inf)

    # 定义熵（Entropy），根据参数 c 计算熵
    def _entropy(self, c):
        return np.log(2.0/c) + 1.0

    # 调用父类的超类方法，并从连续随机变量继承文档字符串
    @_call_super_mom
    @inherit_docstring_from(rv_continuous)
    # 定义一个 fit 方法，用于拟合数据到对数拉普拉斯分布的参数
    def fit(self, data, *args, **kwds):
        # 使用 _check_fit_input_parameters 函数验证输入参数并返回修正后的数据及相关参数
        data, fc, floc, fscale = _check_fit_input_parameters(self, data,
                                                             args, kwds)

        # 当已知位置时，特化最大似然估计 (MLE)
        if floc is None:
            # 调用父类的 fit 方法进行拟合
            return super(type(self), self).fit(data, *args, **kwds)

        # 如果任何观测值的似然性为零，则引发 FitDataError 异常
        if np.any(data <= floc):
            raise FitDataError("loglaplace", lower=floc, upper=np.inf)

        # 从数据中移除位置参数
        if floc != 0:
            data = data - floc

        # 当位置参数为零时，对数拉普拉斯分布与拉普拉斯分布相关，
        # 如果 X ~ Laplace(loc=a, scale=b)，那么 Y = exp(X) ~ LogLaplace(c=1/b, loc=0, scale=exp(a))
        # 可以证明 Y 的最大似然估计 (MLE) 与 X = ln(Y) 的 MLE 相同。
        # 因此，我们重用 laplace.fit() 的公式，并将结果转换回对数拉普拉斯的参数空间。
        a, b = laplace.fit(np.log(data),
                           floc=np.log(fscale) if fscale is not None else None,
                           fscale=1/fc if fc is not None else None,
                           method='mle')
        # 将位置参数和尺度参数分配给 loc 和 scale
        loc = floc
        scale = np.exp(a) if fscale is None else fscale
        c = 1 / b if fc is None else fc
        # 返回对数拉普拉斯分布的参数 c, loc, scale
        return c, loc, scale
# 创建一个名为 loglaplace 的概率分布生成器，设定参数 a=0.0，名称为 'loglaplace'
loglaplace = loglaplace_gen(a=0.0, name='loglaplace')

# 定义一个函数 _lognorm_logpdf，用于计算对数正态分布的概率密度函数的对数值
def _lognorm_logpdf(x, s):
    return _lazywhere(x != 0, (x, s),
                      lambda x, s: (-np.log(x)**2 / (2 * s**2)
                                    - np.log(s * x * np.sqrt(2 * np.pi))),
                      -np.inf)

# 定义一个名为 lognorm_gen 的概率分布类，继承自 rv_continuous
class lognorm_gen(rv_continuous):
    r"""A lognormal continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `lognorm` is:

    .. math::

        f(x, s) = \frac{1}{s x \sqrt{2\pi}}
                  \exp\left(-\frac{\log^2(x)}{2s^2}\right)

    for :math:`x > 0`, :math:`s > 0`.

    `lognorm` takes ``s`` as a shape parameter for :math:`s`.

    %(after_notes)s

    Suppose a normally distributed random variable ``X`` has  mean ``mu`` and
    standard deviation ``sigma``. Then ``Y = exp(X)`` is lognormally
    distributed with ``s = sigma`` and ``scale = exp(mu)``.

    %(example)s

    The logarithm of a log-normally distributed random variable is
    normally distributed:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy import stats
    >>> fig, ax = plt.subplots(1, 1)
    >>> mu, sigma = 2, 0.5
    >>> X = stats.norm(loc=mu, scale=sigma)
    >>> Y = stats.lognorm(s=sigma, scale=np.exp(mu))
    >>> x = np.linspace(*X.interval(0.999))
    >>> y = Y.rvs(size=10000)
    >>> ax.plot(x, X.pdf(x), label='X (pdf)')
    >>> ax.hist(np.log(y), density=True, bins=x, label='log(Y) (histogram)')
    >>> ax.legend()
    >>> plt.show()

    """
    _support_mask = rv_continuous._open_support_mask

    # 定义一个函数，返回参数信息，这里是参数 's' 的范围和是否必须为正数
    def _shape_info(self):
        return [_ShapeInfo("s", False, (0, np.inf), (False, False))]

    # 定义随机变量生成函数，返回服从指数正态分布的随机变量
    def _rvs(self, s, size=None, random_state=None):
        return np.exp(s * random_state.standard_normal(size))

    # 定义概率密度函数，返回指数正态分布的概率密度函数值
    def _pdf(self, x, s):
        # lognorm.pdf(x, s) = 1 / (s*x*sqrt(2*pi)) * exp(-1/2*(log(x)/s)**2)
        return np.exp(self._logpdf(x, s))

    # 定义对数概率密度函数，返回指数正态分布的对数概率密度函数值
    def _logpdf(self, x, s):
        return _lognorm_logpdf(x, s)

    # 定义累积分布函数，返回指数正态分布的累积分布函数值
    def _cdf(self, x, s):
        return _norm_cdf(np.log(x) / s)

    # 定义对数累积分布函数，返回指数正态分布的对数累积分布函数值
    def _logcdf(self, x, s):
        return _norm_logcdf(np.log(x) / s)

    # 定义反函数，返回给定概率值对应的指数正态分布的反函数值
    def _ppf(self, q, s):
        return np.exp(s * _norm_ppf(q))

    # 定义生存函数，返回指数正态分布的生存函数值
    def _sf(self, x, s):
        return _norm_sf(np.log(x) / s)

    # 定义对数生存函数，返回指数正态分布的对数生存函数值
    def _logsf(self, x, s):
        return _norm_logsf(np.log(x) / s)

    # 定义逆生存函数，返回给定生存概率对应的指数正态分布的逆生存函数值
    def _isf(self, q, s):
        return np.exp(s * _norm_isf(q))

    # 定义统计量函数，返回指数正态分布的期望、方差、偏度和峰度
    def _stats(self, s):
        p = np.exp(s*s)
        mu = np.sqrt(p)
        mu2 = p*(p-1)
        g1 = np.sqrt(p-1)*(2+p)
        g2 = np.polyval([1, 2, 3, 0, -6.0], p)
        return mu, mu2, g1, g2

    # 定义熵函数，返回指数正态分布的熵值
    def _entropy(self, s):
        return 0.5 * (1 + np.log(2*np.pi) + 2 * np.log(s))

    # 用于调用超类方法的装饰器，用于处理 MOM 相关操作
    @_call_super_mom
    # 使用装饰器 `@extend_notes_in_docstring` 扩展了 `rv_continuous` 对象的文档字符串
    # 当 `method='MLE'` 且通过 `floc` 参数固定了位置参数时，
    # 此函数使用显式公式对对数正态分布的形状和尺度参数进行最大似然估计，
    # 因此忽略了 `optimizer`、`loc` 和 `scale` 关键字参数。
    # 如果位置是自由的，则通过将其对位置的偏导数设置为0，并通过替换形状和尺度的解析表达式（或提供的参数）来找到似然最大值。
    # 参见例如 A. Clifford Cohen & Betty Jones Whitten (1980) 的第3.1方程式
    # "Estimation in the Three-Parameter Lognormal Distribution"
    # Journal of the American Statistical Association, 75:370, 399-404
    # https://doi.org/10.2307/2287466
lognorm = lognorm_gen(a=0.0, name='lognorm')

class gibrat_gen(rv_continuous):
    r"""A Gibrat continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `gibrat` is:

    .. math::

        f(x) = \frac{1}{x \sqrt{2\pi}} \exp(-\frac{1}{2} (\log(x))^2)

    `gibrat` is a special case of `lognorm` with ``s=1``.

    %(after_notes)s

    %(example)s

    """
    _support_mask = rv_continuous._open_support_mask

    def _shape_info(self):
        return []  # 返回空列表，表示没有额外的形状信息

    def _rvs(self, size=None, random_state=None):
        return np.exp(random_state.standard_normal(size))  # 从标准正态分布生成指定大小的随机样本，然后取指数

    def _pdf(self, x):
        # gibrat.pdf(x) = 1/(x*sqrt(2*pi)) * exp(-1/2*(log(x))**2)
        return np.exp(self._logpdf(x))  # 返回概率密度函数值，调用内部的对数概率密度函数方法

    def _logpdf(self, x):
        return _lognorm_logpdf(x, 1.0)  # 调用_lognorm_logpdf函数计算对数概率密度函数值

    def _cdf(self, x):
        return _norm_cdf(np.log(x))  # 返回累积分布函数值，使用对数变换后的正态分布的累积分布函数

    def _ppf(self, q):
        return np.exp(_norm_ppf(q))  # 返回百分位点函数值，使用正态分布的百分位点函数后取指数

    def _sf(self, x):
        return _norm_sf(np.log(x))  # 返回生存函数值，使用对数变换后的正态分布的生存函数

    def _isf(self, p):
        return np.exp(_norm_isf(p))  # 返回逆生存函数值，使用正态分布的逆生存函数后取指数

    def _stats(self):
        p = np.e
        mu = np.sqrt(p)
        mu2 = p * (p - 1)
        g1 = np.sqrt(p - 1) * (2 + p)
        g2 = np.polyval([1, 2, 3, 0, -6.0], p)
        return mu, mu2, g1, g2  # 返回统计量：均值mu，方差mu2，偏度g1，峰度g2

    def _entropy(self):
        return 0.5 * np.log(2 * np.pi) + 0.5  # 返回熵，使用公式0.5 * log(2*pi) + 0.5


gibrat = gibrat_gen(a=0.0, name='gibrat')  # 创建一个名为gibrat的gibrat_gen实例

class maxwell_gen(rv_continuous):
    r"""A Maxwell continuous random variable.

    %(before_notes)s

    Notes
    -----
    A special case of a `chi` distribution,  with ``df=3``, ``loc=0.0``,
    and given ``scale = a``, where ``a`` is the parameter used in the
    Mathworld description [1]_.

    The probability density function for `maxwell` is:

    .. math::

        f(x) = \sqrt{2/\pi}x^2 \exp(-x^2/2)

    for :math:`x >= 0`.

    %(after_notes)s

    References
    ----------
    .. [1] http://mathworld.wolfram.com/MaxwellDistribution.html

    %(example)s
    """
    def _shape_info(self):
        return []  # 返回空列表，表示没有额外的形状信息

    def _rvs(self, size=None, random_state=None):
        return chi.rvs(3.0, size=size, random_state=random_state)  # 使用chi分布生成自由度为3的随机样本

    def _pdf(self, x):
        # maxwell.pdf(x) = sqrt(2/pi)x**2 * exp(-x**2/2)
        return _SQRT_2_OVER_PI*x*x*np.exp(-x*x/2.0)  # 返回概率密度函数值，使用maxwell分布的定义

    def _logpdf(self, x):
        # Allow x=0 without 'divide by zero' warnings
        with np.errstate(divide='ignore'):
            return _LOG_SQRT_2_OVER_PI + 2*np.log(x) - 0.5*x*x  # 返回对数概率密度函数值，处理x=0的情况

    def _cdf(self, x):
        return sc.gammainc(1.5, x*x/2.0)  # 返回累积分布函数值，使用gamma函数的不完全伽玛函数

    def _ppf(self, q):
        return np.sqrt(2*sc.gammaincinv(1.5, q))  # 返回百分位点函数值，使用gamma函数的不完全伽玛函数的逆函数

    def _sf(self, x):
        return sc.gammaincc(1.5, x*x/2.0)  # 返回生存函数值，使用gamma函数的不完全伽玛函数的补函数

    def _isf(self, q):
        return np.sqrt(2*sc.gammainccinv(1.5, q))  # 返回逆生存函数值，使用gamma函数的不完全伽玛函数的补函数的逆函数

    def _stats(self):
        val = 3*np.pi-8
        return (2*np.sqrt(2.0/np.pi),
                3-8/np.pi,
                np.sqrt(2)*(32-10*np.pi)/val**1.5,
                (-12*np.pi*np.pi + 160*np.pi - 384) / val**2.0)  # 返回统计量：均值，方差，偏度，峰度
    # 计算熵值的私有方法
    def _entropy(self):
        # 返回欧拉常数加上 0.5 * ln(2π) 减去 0.5 的结果
        return _EULER + 0.5*np.log(2*np.pi)-0.5
maxwell = maxwell_gen(a=0.0, name='maxwell')



# 创建一个 Maxwell 分布的实例 maxwell，参数 a 设定为 0.0，名称设定为 'maxwell'
maxwell = maxwell_gen(a=0.0, name='maxwell')



class mielke_gen(rv_continuous):
    r"""A Mielke Beta-Kappa / Dagum continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `mielke` is:

    .. math::

        f(x, k, s) = \frac{k x^{k-1}}{(1+x^s)^{1+k/s}}

    for :math:`x > 0` and :math:`k, s > 0`. The distribution is sometimes
    called Dagum distribution ([2]_). It was already defined in [3]_, called
    a Burr Type III distribution (`burr` with parameters ``c=s`` and
    ``d=k/s``).

    `mielke` takes ``k`` and ``s`` as shape parameters.

    %(after_notes)s

    References
    ----------
    .. [1] Mielke, P.W., 1973 "Another Family of Distributions for Describing
           and Analyzing Precipitation Data." J. Appl. Meteor., 12, 275-280
    .. [2] Dagum, C., 1977 "A new model for personal income distribution."
           Economie Appliquee, 33, 327-367.
    .. [3] Burr, I. W. "Cumulative frequency functions", Annals of
           Mathematical Statistics, 13(2), pp 215-232 (1942).

    %(example)s

    """
    
    def _shape_info(self):
        ik = _ShapeInfo("k", False, (0, np.inf), (False, False))
        i_s = _ShapeInfo("s", False, (0, np.inf), (False, False))
        return [ik, i_s]

    def _pdf(self, x, k, s):
        return k*x**(k-1.0) / (1.0+x**s)**(1.0+k*1.0/s)

    def _logpdf(self, x, k, s):
        # Allow x=0 without 'divide by zero' warnings.
        with np.errstate(divide='ignore'):
            return np.log(k) + np.log(x)*(k - 1) - np.log1p(x**s)*(1 + k/s)

    def _cdf(self, x, k, s):
        return x**k / (1.0+x**s)**(k*1.0/s)

    def _ppf(self, q, k, s):
        qsk = pow(q, s*1.0/k)
        return pow(qsk/(1.0-qsk), 1.0/s)

    def _munp(self, n, k, s):
        def nth_moment(n, k, s):
            # n-th moment is defined for -k < n < s
            return sc.gamma((k+n)/s)*sc.gamma(1-n/s)/sc.gamma(k/s)

        return _lazywhere(n < s, (n, k, s), nth_moment, np.inf)



# 创建一个 Mielke Beta-Kappa / Dagum 连续随机变量的类 mielke_gen，继承自 rv_continuous
# 类内部包含了该分布的概率密度函数、累积分布函数、反函数等定义
class mielke_gen(rv_continuous):
    r"""A Mielke Beta-Kappa / Dagum continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `mielke` is:

    .. math::

        f(x, k, s) = \frac{k x^{k-1}}{(1+x^s)^{1+k/s}}

    for :math:`x > 0` and :math:`k, s > 0`. The distribution is sometimes
    called Dagum distribution ([2]_). It was already defined in [3]_, called
    a Burr Type III distribution (`burr` with parameters ``c=s`` and
    ``d=k/s``).

    `mielke` takes ``k`` and ``s`` as shape parameters.

    %(after_notes)s

    References
    ----------
    .. [1] Mielke, P.W., 1973 "Another Family of Distributions for Describing
           and Analyzing Precipitation Data." J. Appl. Meteor., 12, 275-280
    .. [2] Dagum, C., 1977 "A new model for personal income distribution."
           Economie Appliquee, 33, 327-367.
    .. [3] Burr, I. W. "Cumulative frequency functions", Annals of
           Mathematical Statistics, 13(2), pp 215-232 (1942).

    %(example)s

    """

    # 返回分布参数的描述信息，包括 k 和 s 的取值范围
    def _shape_info(self):
        ik = _ShapeInfo("k", False, (0, np.inf), (False, False))
        i_s = _ShapeInfo("s", False, (0, np.inf), (False, False))
        return [ik, i_s]

    # 概率密度函数的定义
    def _pdf(self, x, k, s):
        return k*x**(k-1.0) / (1.0+x**s)**(1.0+k*1.0/s)

    # 对数概率密度函数的定义，处理 x=0 时不产生 'divide by zero' 警告
    def _logpdf(self, x, k, s):
        with np.errstate(divide='ignore'):
            return np.log(k) + np.log(x)*(k - 1) - np.log1p(x**s)*(1 + k/s)

    # 累积分布函数的定义
    def _cdf(self, x, k, s):
        return x**k / (1.0+x**s)**(k*1.0/s)

    # 百分点函数（反函数）的定义
    def _ppf(self, q, k, s):
        qsk = pow(q, s*1.0/k)
        return pow(qsk/(1.0-qsk), 1.0/s)

    # n 阶非中心矩的定义
    def _munp(self, n, k, s):
        # 内部函数定义，计算 n 阶非中心矩
        def nth_moment(n, k, s):
            # 只有在 -k < n < s 时才有定义
            return sc.gamma((k+n)/s)*sc.gamma(1-n/s)/sc.gamma(k/s)

        return _lazywhere(n < s, (n, k, s), nth_moment, np.inf)



mielke = mielke_gen(a=0.0, name='mielke')



# 创建一个 Mielke Beta-Kappa / Dagum 分布的实例 mielke，参数 a 设定为 0.0，名称设定为 'mielke'
mielke = mielke_gen(a=0.0, name='mielke')



class kappa4_gen(rv_continuous):
    r"""Kappa 4 parameter distribution.

    %(before_notes)s

    Notes
    -----
    The probability density function for kappa4 is:

    .. math::

        f(x, h, k) = (1 - k x)^{1/k - 1} (1 - h (1 - k x)^{1/k})^{1/h-1}

    if :math:`h` and :math:`k` are not equal to 0.

    If :math:`h` or :math:`k` are zero then the pdf can be simplified:

    h = 0 and k != 0::

        kappa4.pdf(x, h, k) = (1.0 - k*x)**(1.0/k - 1.0)*
                              exp(-(1.0 - k*x)**(1.0/k))

    h != 0 and k = 0::

        kappa4.pdf(x, h, k) = exp(-x)*(1.0 - h*exp(-x))**(1.0/h - 1.0)

    h = 0 and k = 0::

        kappa4.pdf(x, h, k) = exp(-x)*exp(-exp(-x))

    kappa4 takes :math:`h` and :math:`k` as shape parameters.

    The kappa4 distribution returns other distributions when certain
    :math:`h` and :math:`k` values are used.

    +------+-------------+----------------+------------------+



# 创建一个 Kappa 4 参数分布的类 kappa4_gen，继承自 rv_continuous
# 类内部包含了该分布的概率密度函数的多种情况定义
class kappa4_gen(rv_continuous):
    r"""Kappa 4 parameter distribution
    """
    这部分代码定义了一个类内部的两个函数，用于参数检查和形状信息返回。
    
    """
    def _argcheck(self, h, k):
        # 将输入的 h 和 k 进行广播，获取其形状
        shape = np.broadcast_arrays(h, k)[0].shape
        # 返回一个形状相同、填充值为 True 的布尔数组
        return np.full(shape, fill_value=True)
    
    def _shape_info(self):
        # 定义 h 和 k 的形状信息对象，包括名称、是否必须、范围和是否有界
        ih = _ShapeInfo("h", False, (-np.inf, np.inf), (False, False))
        ik = _ShapeInfo("k", False, (-np.inf, np.inf), (False, False))
        # 返回一个列表，包含 h 和 k 的形状信息对象
        return [ih, ik]
    def _get_support(self, h, k):
        # 定义条件列表，根据不同的 h 和 k 进行逻辑条件判断
        condlist = [np.logical_and(h > 0, k > 0),
                    np.logical_and(h > 0, k == 0),
                    np.logical_and(h > 0, k < 0),
                    np.logical_and(h <= 0, k > 0),
                    np.logical_and(h <= 0, k == 0),
                    np.logical_and(h <= 0, k < 0)]

        # 定义函数 f0 到 f5，根据不同条件返回相应计算结果
        def f0(h, k):
            return (1.0 - np.float_power(h, -k))/k

        def f1(h, k):
            return np.log(h)

        def f3(h, k):
            a = np.empty(np.shape(h))
            a[:] = -np.inf
            return a

        def f5(h, k):
            return 1.0/k

        # 使用 Lazy evaluation 选择合适的函数来计算结果 _a
        _a = _lazyselect(condlist,
                         [f0, f1, f0, f3, f3, f5],
                         [h, k],
                         default=np.nan)

        # 再次定义函数 f0 和 f1，用于不同的条件下返回不同的计算结果
        def f0(h, k):
            return 1.0/k

        def f1(h, k):
            a = np.empty(np.shape(h))
            a[:] = np.inf
            return a

        # 使用 Lazy evaluation 选择合适的函数来计算结果 _b
        _b = _lazyselect(condlist,
                         [f0, f1, f1, f0, f1, f1],
                         [h, k],
                         default=np.nan)
        # 返回结果 _a 和 _b
        return _a, _b

    def _pdf(self, x, h, k):
        # 返回 x、h、k 对应的概率密度函数的指数值
        return np.exp(self._logpdf(x, h, k))

    def _logpdf(self, x, h, k):
        # 定义条件列表，根据不同的 h 和 k 进行逻辑条件判断
        condlist = [np.logical_and(h != 0, k != 0),
                    np.logical_and(h == 0, k != 0),
                    np.logical_and(h != 0, k == 0),
                    np.logical_and(h == 0, k == 0)]

        # 定义函数 f0 到 f3，根据不同条件返回相应计算结果
        def f0(x, h, k):
            '''pdf = (1.0 - k*x)**(1.0/k - 1.0)*(
                      1.0 - h*(1.0 - k*x)**(1.0/k))**(1.0/h-1.0)
               logpdf = ...
            '''
            return (sc.xlog1py(1.0/k - 1.0, -k*x) +
                    sc.xlog1py(1.0/h - 1.0, -h*(1.0 - k*x)**(1.0/k)))

        def f1(x, h, k):
            '''pdf = (1.0 - k*x)**(1.0/k - 1.0)*np.exp(-(
                      1.0 - k*x)**(1.0/k))
               logpdf = ...
            '''
            return sc.xlog1py(1.0/k - 1.0, -k*x) - (1.0 - k*x)**(1.0/k)

        def f2(x, h, k):
            '''pdf = np.exp(-x)*(1.0 - h*np.exp(-x))**(1.0/h - 1.0)
               logpdf = ...
            '''
            return -x + sc.xlog1py(1.0/h - 1.0, -h*np.exp(-x))

        def f3(x, h, k):
            '''pdf = np.exp(-x-np.exp(-x))
               logpdf = ...
            '''
            return -x - np.exp(-x)

        # 使用 Lazy evaluation 选择合适的函数来计算结果
        return _lazyselect(condlist,
                           [f0, f1, f2, f3],
                           [x, h, k],
                           default=np.nan)

    def _cdf(self, x, h, k):
        # 返回 x、h、k 对应的累积分布函数的指数值
        return np.exp(self._logcdf(x, h, k))
    # 根据输入的参数 x, h, k 和条件列表选择合适的函数进行计算并返回结果
    def _logcdf(self, x, h, k):
        condlist = [np.logical_and(h != 0, k != 0),
                    np.logical_and(h == 0, k != 0),
                    np.logical_and(h != 0, k == 0),
                    np.logical_and(h == 0, k == 0)]

        # 第一种条件下的计算函数，计算对数累积分布函数的值
        def f0(x, h, k):
            '''cdf = (1.0 - h*(1.0 - k*x)**(1.0/k))**(1.0/h)
               logcdf = ...
            '''
            return (1.0/h)*sc.log1p(-h*(1.0 - k*x)**(1.0/k))

        # 第二种条件下的计算函数，计算对数累积分布函数的值
        def f1(x, h, k):
            '''cdf = np.exp(-(1.0 - k*x)**(1.0/k))
               logcdf = ...
            '''
            return -(1.0 - k*x)**(1.0/k)

        # 第三种条件下的计算函数，计算对数累积分布函数的值
        def f2(x, h, k):
            '''cdf = (1.0 - h*np.exp(-x))**(1.0/h)
               logcdf = ...
            '''
            return (1.0/h)*sc.log1p(-h*np.exp(-x))

        # 第四种条件下的计算函数，计算对数累积分布函数的值
        def f3(x, h, k):
            '''cdf = np.exp(-np.exp(-x))
               logcdf = ...
            '''
            return -np.exp(-x)

        # 根据条件列表选择相应的函数进行计算，若无匹配则返回默认值 np.nan
        return _lazyselect(condlist,
                           [f0, f1, f2, f3],
                           [x, h, k],
                           default=np.nan)

    # 根据输入的参数 q, h, k 和条件列表选择合适的函数进行计算并返回结果
    def _ppf(self, q, h, k):
        condlist = [np.logical_and(h != 0, k != 0),
                    np.logical_and(h == 0, k != 0),
                    np.logical_and(h != 0, k == 0),
                    np.logical_and(h == 0, k == 0)]

        # 第一种条件下的计算函数，计算反函数的值
        def f0(q, h, k):
            return 1.0/k*(1.0 - ((1.0 - (q**h))/h)**k)

        # 第二种条件下的计算函数，计算反函数的值
        def f1(q, h, k):
            return 1.0/k*(1.0 - (-np.log(q))**k)

        # 第三种条件下的计算函数，计算反函数的值
        def f2(q, h, k):
            '''ppf = -np.log((1.0 - (q**h))/h)
            '''
            return -sc.log1p(-(q**h)) + np.log(h)

        # 第四种条件下的计算函数，计算反函数的值
        def f3(q, h, k):
            return -np.log(-np.log(q))

        # 根据条件列表选择相应的函数进行计算，若无匹配则返回默认值 np.nan
        return _lazyselect(condlist,
                           [f0, f1, f2, f3],
                           [q, h, k],
                           default=np.nan)

    # 根据输入的参数 h 和 k 和条件列表选择合适的函数进行计算并返回结果
    def _get_stats_info(self, h, k):
        condlist = [
            np.logical_and(h < 0, k >= 0),
            k < 0,
        ]

        # 第一种条件下的计算函数，返回统计信息
        def f0(h, k):
            return (-1.0/h*k).astype(int)

        # 第二种条件下的计算函数，返回统计信息
        def f1(h, k):
            return (-1.0/k).astype(int)

        # 根据条件列表选择相应的函数进行计算，若无匹配则返回默认值 5
        return _lazyselect(condlist, [f0, f1], [h, k], default=5)

    # 根据输入的参数 h 和 k 计算统计量，并返回结果列表
    def _stats(self, h, k):
        # 调用 _get_stats_info 获取最大的统计信息
        maxr = self._get_stats_info(h, k)
        # 根据条件判断生成输出列表
        outputs = [None if np.any(r < maxr) else np.nan for r in range(1, 5)]
        return outputs[:]

    # 计算一阶矩，根据输入的参数 m 和 args 判断是否返回统计信息或积分结果
    def _mom1_sc(self, m, *args):
        # 调用 _get_stats_info 获取最大的统计信息
        maxr = self._get_stats_info(args[0], args[1])
        # 如果 m 大于等于最大统计信息，则返回 NaN
        if m >= maxr:
            return np.nan
        # 否则，计算积分结果并返回
        return integrate.quad(self._mom_integ1, 0, 1, args=(m,)+args)[0]
# 创建一个名为 kappa4 的对象，使用 kappa4_gen 生成器函数，传入 name='kappa4' 参数
kappa4 = kappa4_gen(name='kappa4')

# 定义一个名为 kappa3_gen 的类，继承自 rv_continuous
class kappa3_gen(rv_continuous):
    r"""Kappa 3 parameter distribution.

    %(before_notes)s

    Notes
    -----
    The probability density function for `kappa3` is:

    .. math::

        f(x, a) = a (a + x^a)^{-(a + 1)/a}

    for :math:`x > 0` and :math:`a > 0`.

    `kappa3` takes ``a`` as a shape parameter for :math:`a`.

    References
    ----------
    P.W. Mielke and E.S. Johnson, "Three-Parameter Kappa Distribution Maximum
    Likelihood and Likelihood Ratio Tests", Methods in Weather Research,
    701-707, (September, 1973),
    :doi:`10.1175/1520-0493(1973)101<0701:TKDMLE>2.3.CO;2`

    B. Kumphon, "Maximum Entropy and Maximum Likelihood Estimation for the
    Three-Parameter Kappa Distribution", Open Journal of Statistics, vol 2,
    415-419 (2012), :doi:`10.4236/ojs.2012.24050`

    %(after_notes)s

    %(example)s

    """
    
    # 定义 _shape_info 方法，返回一个 _ShapeInfo 对象的列表，描述参数 'a' 的特性
    def _shape_info(self):
        return [_ShapeInfo("a", False, (0, np.inf), (False, False))]

    # 定义 _pdf 方法，实现概率密度函数的计算，返回参数 a 和 x 所对应的概率密度值
    def _pdf(self, x, a):
        # kappa3.pdf(x, a) = a*(a + x**a)**(-(a + 1)/a),     for x > 0
        return a*(a + x**a)**(-1.0/a-1)

    # 定义 _cdf 方法，实现累积分布函数的计算，返回参数 a 和 x 所对应的累积概率值
    def _cdf(self, x, a):
        return x*(a + x**a)**(-1.0/a)

    # 定义 _sf 方法，实现生存函数的计算，返回参数 a 和 x 所对应的生存概率值
    def _sf(self, x, a):
        x, a = np.broadcast_arrays(x, a)  # some code paths pass scalars
        sf = super()._sf(x, a)

        # 当生存函数 sf 较小时，使用另一种形式更加准确，但对于大的 a，该方法可能会发散
        cutoff = 0.01
        i = sf < cutoff
        sf2 = -sc.expm1(sc.xlog1py(-1.0 / a[i], a[i] * x[i]**-a[i]))
        i2 = sf2 > cutoff
        sf2[i2] = sf[i][i2]  # replace bad values with original values

        sf[i] = sf2
        return sf

    # 定义 _ppf 方法，实现分位点函数的计算，返回参数 q 和 a 所对应的分位点值
    def _ppf(self, q, a):
        return (a/(q**-a - 1.0))**(1.0/a)

    # 定义 _isf 方法，实现逆生存函数的计算，返回参数 q 和 a 所对应的逆生存概率值
    def _isf(self, q, a):
        lg = sc.xlog1py(-a, -q)
        denom = sc.expm1(lg)
        return (a / denom)**(1.0 / a)

    # 定义 _stats 方法，实现统计特性的计算，返回与参数 a 相关的一组统计特性
    def _stats(self, a):
        outputs = [None if np.any(i < a) else np.nan for i in range(1, 5)]
        return outputs[:]

    # 定义 _mom1_sc 方法，实现第一矩的计算，返回参数 m 和其它参数的一些积分相关计算结果
    def _mom1_sc(self, m, *args):
        if np.any(m >= args[0]):
            return np.nan
        return integrate.quad(self._mom_integ1, 0, 1, args=(m,)+args)[0]

# 创建一个名为 kappa3 的对象，使用 kappa3_gen 生成器类，传入 a=0.0 和 name='kappa3' 参数
kappa3 = kappa3_gen(a=0.0, name='kappa3')

# 定义一个名为 moyal_gen 的类，表示 Moyal 连续随机变量
class moyal_gen(rv_continuous):
    r"""A Moyal continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `moyal` is:

    .. math::

        f(x) = \exp(-(x + \exp(-x))/2) / \sqrt{2\pi}

    for a real number :math:`x`.

    %(after_notes)s

    This distribution has utility in high-energy physics and radiation
    detection. It describes the energy loss of a charged relativistic
    particle due to ionization of the medium [1]_. It also provides an
    approximation for the Landau distribution. For an in depth description
    see [2]_. For additional description, see [3]_.

    References
    ----------
    """
    # 定义一个概率分布的基类，实现了 Moyal 分布的相关方法

    def _shape_info(self):
        # 返回空列表，表示分布没有特定的形状参数
        return []

    def _rvs(self, size=None, random_state=None):
        # 生成 Moyal 分布的随机变量
        # 使用 gamma 分布生成随机数 u1
        u1 = gamma.rvs(a=0.5, scale=2, size=size, random_state=random_state)
        # 计算 Moyal 分布的随机变量并返回
        return -np.log(u1)

    def _pdf(self, x):
        # 计算 Moyal 分布的概率密度函数
        return np.exp(-0.5 * (x + np.exp(-x))) / np.sqrt(2*np.pi)

    def _cdf(self, x):
        # 计算 Moyal 分布的累积分布函数
        return sc.erfc(np.exp(-0.5 * x) / np.sqrt(2))

    def _sf(self, x):
        # 计算 Moyal 分布的生存函数
        return sc.erf(np.exp(-0.5 * x) / np.sqrt(2))

    def _ppf(self, x):
        # 计算 Moyal 分布的分位点函数（反函数）
        return -np.log(2 * sc.erfcinv(x)**2)

    def _stats(self):
        # 计算 Moyal 分布的统计特性：均值（mu）、方差（mu2）、偏度（g1）、峰度（g2）
        mu = np.log(2) + np.euler_gamma
        mu2 = np.pi**2 / 2
        g1 = 28 * np.sqrt(2) * sc.zeta(3) / np.pi**3
        g2 = 4.
        return mu, mu2, g1, g2

    def _munp(self, n):
        # 计算 Moyal 分布的 n 阶原点矩
        if n == 1.0:
            return np.log(2) + np.euler_gamma
        elif n == 2.0:
            return np.pi**2 / 2 + (np.log(2) + np.euler_gamma)**2
        elif n == 3.0:
            tmp1 = 1.5 * np.pi**2 * (np.log(2)+np.euler_gamma)
            tmp2 = (np.log(2)+np.euler_gamma)**3
            tmp3 = 14 * sc.zeta(3)
            return tmp1 + tmp2 + tmp3
        elif n == 4.0:
            tmp1 = 4 * 14 * sc.zeta(3) * (np.log(2) + np.euler_gamma)
            tmp2 = 3 * np.pi**2 * (np.log(2) + np.euler_gamma)**2
            tmp3 = (np.log(2) + np.euler_gamma)**4
            tmp4 = 7 * np.pi**4 / 4
            return tmp1 + tmp2 + tmp3 + tmp4
        else:
            # 对于更高阶的矩，调用默认的方法进行计算
            return self._mom1_sc(n)
    ```
moyal = moyal_gen(name="moyal")

# 定义一个 Nakagami 分布的生成器类，继承自 rv_continuous
class nakagami_gen(rv_continuous):
    r"""A Nakagami continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `nakagami` is:

    .. math::

        f(x, \nu) = \frac{2 \nu^\nu}{\Gamma(\nu)} x^{2\nu-1} \exp(-\nu x^2)

    for :math:`x >= 0`, :math:`\nu > 0`. The distribution was introduced in
    [2]_, see also [1]_ for further information.

    `nakagami` takes ``nu`` as a shape parameter for :math:`\nu`.

    %(after_notes)s

    References
    ----------
    .. [1] "Nakagami distribution", Wikipedia
           https://en.wikipedia.org/wiki/Nakagami_distribution
    .. [2] M. Nakagami, "The m-distribution - A general formula of intensity
           distribution of rapid fading", Statistical methods in radio wave
           propagation, Pergamon Press, 1960, 3-36.
           :doi:`10.1016/B978-0-08-009306-2.50005-4`

    %(example)s

    """
    # 定义参数检查函数，确保 nu 大于 0
    def _argcheck(self, nu):
        return nu > 0

    # 返回参数的形状信息，这里是一个列表，描述了参数 nu 的性质
    def _shape_info(self):
        return [_ShapeInfo("nu", False, (0, np.inf), (False, False))]

    # 定义概率密度函数（PDF）
    def _pdf(self, x, nu):
        return np.exp(self._logpdf(x, nu))

    # 定义概率密度函数的对数（log PDF）
    def _logpdf(self, x, nu):
        # 根据公式计算 Nakagami 分布的对数概率密度函数
        return (np.log(2) + sc.xlogy(nu, nu) - sc.gammaln(nu) +
                sc.xlogy(2*nu - 1, x) - nu*x**2)

    # 定义累积分布函数（CDF）
    def _cdf(self, x, nu):
        return sc.gammainc(nu, nu*x*x)

    # 定义累积分布函数的反函数（PPF）
    def _ppf(self, q, nu):
        return np.sqrt(1.0/nu*sc.gammaincinv(nu, q))

    # 定义生存函数（Survival function）
    def _sf(self, x, nu):
        return sc.gammaincc(nu, nu*x*x)

    # 定义生存函数的反函数（Inverse survival function）
    def _isf(self, p, nu):
        return np.sqrt(1/nu * sc.gammainccinv(nu, p))

    # 定义统计量，返回均值、方差、偏度和峰度
    def _stats(self, nu):
        mu = sc.poch(nu, 0.5)/np.sqrt(nu)
        mu2 = 1.0-mu*mu
        g1 = mu * (1 - 4*nu*mu2) / 2.0 / nu / np.power(mu2, 1.5)
        g2 = -6*mu**4*nu + (8*nu-2)*mu**2-2*nu + 1
        g2 /= nu*mu2**2.0
        return mu, mu2, g1, g2

    # 定义熵（Entropy）函数
    def _entropy(self, nu):
        shape = np.shape(nu)
        # 因为基础设施中可能未处理好这部分…
        nu = np.atleast_1d(nu)
        A = sc.gammaln(nu)
        B = nu - (nu - 0.5) * sc.digamma(nu)
        C = -0.5 * np.log(nu) - np.log(2)
        h = A + B + C
        # 这是 A 和 B 的渐近和（参见 gh-17868）
        norm_entropy = stats.norm._entropy()
        # 对于大 nu，由于舍入误差，使用渐近和更为准确
        i = nu > 5e4  # 舍入误差 ~ 近似误差
        # -1 / (12 * nu) 是 O(1/nu) 项；参见 gh-17929
        h[i] = C[i] + norm_entropy - 1/(12*nu[i])
        return h.reshape(shape)[()]

    # 定义随机变量生成函数（Random variates sampling）
    def _rvs(self, nu, size=None, random_state=None):
        # 这个关系可以在 [1] 中找到，或通过直接计算得到
        return np.sqrt(random_state.standard_gamma(nu, size=size) / nu)
    # 定义私有方法 _fitstart，用于估计分布参数的起始点
    def _fitstart(self, data, args=None):
        # 如果数据类型为 CensoredData，则解除数据的限制
        if isinstance(data, CensoredData):
            data = data._uncensor()
        
        # 如果未提供参数 args，则使用默认参数（长度为 self.numargs，每个元素为 1.0）
        if args is None:
            args = (1.0,) * self.numargs
        
        # 通过解析验证的估计方法，计算起始点的参数
        # 参考文档：https://docs.scipy.org/doc/scipy/reference/tutorial/stats/continuous_nakagami.html
        loc = np.min(data)  # 确定数据的最小值作为位置参数
        scale = np.sqrt(np.sum((data - loc)**2) / len(data))  # 计算数据的标准差作为尺度参数
        return args + (loc, scale)
# 使用给定的参数 a=0.0 创建一个 Nakagami 分布生成器，并赋值给变量 nakagami
nakagami = nakagami_gen(a=0.0, name="nakagami")

# 函数名 ncx2 是非中心卡方分布的缩写，定义了一个用于计算非中心卡方分布对数概率密度的函数
def _ncx2_log_pdf(x, df, nc):
    # 将 df 除以 2，并减去 1，用于改进数值稳定性，计算 xs 和 ns 分别为 x 和 nc 的平方根
    df2 = df/2.0 - 1.0
    xs, ns = np.sqrt(x), np.sqrt(nc)
    # 计算非中心卡方分布对数概率密度的值，使用 scipy.special.xlogy 和 scipy.special.ive 函数
    res = sc.xlogy(df2/2.0, x/nc) - 0.5*(xs - ns)**2
    corr = sc.ive(df2, xs*ns) / 2.0
    # 返回结果 res + np.log(corr)，避免对 np.log(0) 操作
    return _lazywhere(
        corr > 0,
        (res, corr),
        f=lambda r, c: r + np.log(c),
        fillvalue=-np.inf)

# 定义一个类 ncx2_gen，继承于 rv_continuous，用于生成非中心卡方分布的随机变量
class ncx2_gen(rv_continuous):
    r"""A non-central chi-squared continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `ncx2` is:

    .. math::

        f(x, k, \lambda) = \frac{1}{2} \exp(-(\lambda+x)/2)
            (x/\lambda)^{(k-2)/4}  I_{(k-2)/2}(\sqrt{\lambda x})

    for :math:`x >= 0`, :math:`k > 0` and :math:`\lambda \ge 0`.
    :math:`k` specifies the degrees of freedom (denoted ``df`` in the
    implementation) and :math:`\lambda` is the non-centrality parameter
    (denoted ``nc`` in the implementation). :math:`I_\nu` denotes the
    modified Bessel function of first order of degree :math:`\nu`
    (`scipy.special.iv`).

    `ncx2` takes ``df`` and ``nc`` as shape parameters.

    %(after_notes)s

    %(example)s

    """
    # 检查参数 df 和 nc 的合法性，返回布尔值
    def _argcheck(self, df, nc):
        return (df > 0) & np.isfinite(df) & (nc >= 0)

    # 返回一个描述参数形状信息的列表，包括 df 和 nc 的取值范围
    def _shape_info(self):
        idf = _ShapeInfo("df", False, (0, np.inf), (False, False))
        inc = _ShapeInfo("nc", False, (0, np.inf), (True, False))
        return [idf, inc]

    # 生成随机样本函数，使用随机状态生成非中心卡方分布的样本
    def _rvs(self, df, nc, size=None, random_state=None):
        return random_state.noncentral_chisquare(df, nc, size)

    # 计算对数概率密度函数，根据条件调用 _ncx2_log_pdf 或 chi2._logpdf 函数
    def _logpdf(self, x, df, nc):
        cond = np.ones_like(x, dtype=bool) & (nc != 0)
        return _lazywhere(cond, (x, df, nc), f=_ncx2_log_pdf,
                          f2=lambda x, df, _: chi2._logpdf(x, df))

    # 计算概率密度函数，根据条件调用 scu._ncx2_pdf 或 chi2._pdf 函数
    def _pdf(self, x, df, nc):
        cond = np.ones_like(x, dtype=bool) & (nc != 0)
        with np.errstate(over='ignore'):  # see gh-17432
            return _lazywhere(cond, (x, df, nc), f=scu._ncx2_pdf,
                              f2=lambda x, df, _: chi2._pdf(x, df))

    # 计算累积分布函数，根据条件调用 scu._ncx2_cdf 或 chi2._cdf 函数
    def _cdf(self, x, df, nc):
        cond = np.ones_like(x, dtype=bool) & (nc != 0)
        with np.errstate(over='ignore'):  # see gh-17432
            return _lazywhere(cond, (x, df, nc), f=scu._ncx2_cdf,
                              f2=lambda x, df, _: chi2._cdf(x, df))

    # 计算累积分布函数的反函数，根据条件调用 scu._ncx2_ppf 或 chi2._ppf 函数
    def _ppf(self, q, df, nc):
        cond = np.ones_like(q, dtype=bool) & (nc != 0)
        with np.errstate(over='ignore'):  # see gh-17432
            return _lazywhere(cond, (q, df, nc), f=scu._ncx2_ppf,
                              f2=lambda x, df, _: chi2._ppf(x, df))
    # 定义一个函数 _sf，计算非中心卡方分布的生存函数（SF）
    def _sf(self, x, df, nc):
        # 创建一个布尔类型的条件数组，初始化为 True，并且排除 nc 为 0 的情况
        cond = np.ones_like(x, dtype=bool) & (nc != 0)
        # 在忽略浮点溢出错误的情况下，使用 _lazywhere 函数计算生存函数（SF）
        with np.errstate(over='ignore'):  # 参考 gh-17432
            # 如果条件满足，调用 scu._ncx2_sf 函数；否则调用 chi2._sf 函数
            return _lazywhere(cond, (x, df, nc), f=scu._ncx2_sf,
                              f2=lambda x, df, _: chi2._sf(x, df))

    # 定义一个函数 _isf，计算非中心卡方分布的逆生存函数（ISF）
    def _isf(self, x, df, nc):
        # 创建一个布尔类型的条件数组，初始化为 True，并且排除 nc 为 0 的情况
        cond = np.ones_like(x, dtype=bool) & (nc != 0)
        # 在忽略浮点溢出错误的情况下，使用 _lazywhere 函数计算逆生存函数（ISF）
        with np.errstate(over='ignore'):  # 参考 gh-17432
            # 如果条件满足，调用 scu._ncx2_isf 函数；否则调用 chi2._isf 函数
            return _lazywhere(cond, (x, df, nc), f=scu._ncx2_isf,
                              f2=lambda x, df, _: chi2._isf(x, df))

    # 定义一个函数 _stats，计算非中心卡方分布的统计量（均值、方差、偏度、峰度）
    def _stats(self, df, nc):
        # 计算非中心卡方分布的均值
        _ncx2_mean = df + nc
        # 定义一个内部函数 k_plus_cl，用于计算 k + c*l
        def k_plus_cl(k, l, c):
            return k + c*l
        # 计算非中心卡方分布的方差
        _ncx2_variance =  2.0 * k_plus_cl(df, nc, 2.0)
        # 计算非中心卡方分布的偏度
        _ncx2_skewness = (np.sqrt(8.0) * k_plus_cl(df, nc, 3) /
                          np.sqrt(k_plus_cl(df, nc, 2.0)**3))
        # 计算非中心卡方分布的峰度（超额峰度）
        _ncx2_kurtosis_excess = (12.0 * k_plus_cl(df, nc, 4.0) /
                                 k_plus_cl(df, nc, 2.0)**2)
        # 返回计算结果：均值、方差、偏度、峰度
        return (
            _ncx2_mean,
            _ncx2_variance,
            _ncx2_skewness,
            _ncx2_kurtosis_excess,
        )
# 定义一个非中心 F 分布的生成器类 ncf_gen，继承自 scipy.stats.rv_continuous
class ncf_gen(rv_continuous):
    r"""A non-central F distribution continuous random variable.

    %(before_notes)s

    See Also
    --------
    scipy.stats.f : Fisher distribution

    Notes
    -----
    The probability density function for `ncf` is:

    .. math::

        f(x, n_1, n_2, \lambda) =
            \exp\left(\frac{\lambda}{2} +
                      \lambda n_1 \frac{x}{2(n_1 x + n_2)}
                \right)
            n_1^{n_1/2} n_2^{n_2/2} x^{n_1/2 - 1} \\
            (n_2 + n_1 x)^{-(n_1 + n_2)/2}
            \gamma(n_1/2) \gamma(1 + n_2/2) \\
            \frac{L^{\frac{n_1}{2}-1}_{n_2/2}
                \left(-\lambda n_1 \frac{x}{2(n_1 x + n_2)}\right)}
            {B(n_1/2, n_2/2)
                \gamma\left(\frac{n_1 + n_2}{2}\right)}

    for :math:`n_1, n_2 > 0`, :math:`\lambda \ge 0`.  Here :math:`n_1` is the
    degrees of freedom in the numerator, :math:`n_2` the degrees of freedom in
    the denominator, :math:`\lambda` the non-centrality parameter,
    :math:`\gamma` is the logarithm of the Gamma function, :math:`L_n^k` is a
    generalized Laguerre polynomial and :math:`B` is the beta function.

    `ncf` takes ``df1``, ``df2`` and ``nc`` as shape parameters. If ``nc=0``,
    the distribution becomes equivalent to the Fisher distribution.

    %(after_notes)s

    %(example)s

    """
    
    # 定义参数检查函数，检查 df1, df2, nc 是否满足条件
    def _argcheck(self, df1, df2, nc):
        return (df1 > 0) & (df2 > 0) & (nc >= 0)

    # 返回参数的形状信息，用于生成参数空间
    def _shape_info(self):
        idf1 = _ShapeInfo("df1", False, (0, np.inf), (False, False))
        idf2 = _ShapeInfo("df2", False, (0, np.inf), (False, False))
        inc = _ShapeInfo("nc", False, (0, np.inf), (True, False))
        return [idf1, idf2, inc]

    # 生成随机变量样本的方法，使用给定的随机种子
    def _rvs(self, dfn, dfd, nc, size=None, random_state=None):
        return random_state.noncentral_f(dfn, dfd, nc, size)

    # 概率密度函数，计算非中心 F 分布的概率密度函数值
    def _pdf(self, x, dfn, dfd, nc):
        return scu._ncf_pdf(x, dfn, dfd, nc)

    # 累积分布函数，计算非中心 F 分布的累积分布函数值
    def _cdf(self, x, dfn, dfd, nc):
        return scu._ncf_cdf(x, dfn, dfd, nc)

    # 百分位点函数，计算非中心 F 分布的百分位点函数值
    def _ppf(self, q, dfn, dfd, nc):
        with np.errstate(over='ignore'):  # 处理潜在的溢出情况，参见 gh-17432
            return scu._ncf_ppf(q, dfn, dfd, nc)

    # 生存函数，计算非中心 F 分布的生存函数值
    def _sf(self, x, dfn, dfd, nc):
        return scu._ncf_sf(x, dfn, dfd, nc)

    # 逆生存函数，计算非中心 F 分布的逆生存函数值
    def _isf(self, x, dfn, dfd, nc):
        with np.errstate(over='ignore'):  # 处理潜在的溢出情况，参见 gh-17432
            return scu._ncf_isf(x, dfn, dfd, nc)
    # 计算非中心 F 分布的一个特定值
    def _munp(self, n, dfn, dfd, nc):
        # 计算 (dfn/dfd)^n
        val = (dfn * 1.0/dfd)**n
        # 计算 gamma 函数的对数值
        term = sc.gammaln(n+0.5*dfn) + sc.gammaln(0.5*dfd-n) - sc.gammaln(dfd*0.5)
        # 计算非中心参数影响后的指数项
        val *= np.exp(-nc / 2.0 + term)
        # 计算超几何函数 1F1
        val *= sc.hyp1f1(n+0.5*dfn, 0.5*dfn, 0.5*nc)
        # 返回最终计算结果
        return val
    
    # 计算非中心 F 分布的统计量
    def _stats(self, dfn, dfd, nc, moments='mv'):
        # 计算期望值
        mu = scu._ncf_mean(dfn, dfd, nc)
        # 计算方差
        mu2 = scu._ncf_variance(dfn, dfd, nc)
        # 计算偏度（如果 's' 在 moments 中）
        g1 = scu._ncf_skewness(dfn, dfd, nc) if 's' in moments else None
        # 计算峰度（如果 'k' 在 moments 中）
        g2 = scu._ncf_kurtosis_excess(dfn, dfd, nc) if 'k' in moments else None
        # 返回期望值、方差、偏度和峰度
        return mu, mu2, g1, g2
# 定义一个连续型的学生 t 分布随机变量的类
class t_gen(rv_continuous):
    r"""A Student's t continuous random variable.

    For the noncentral t distribution, see `nct`.

    %(before_notes)s

    See Also
    --------
    nct

    Notes
    -----
    The probability density function for `t` is:

    .. math::

        f(x, \nu) = \frac{\Gamma((\nu+1)/2)}
                        {\sqrt{\pi \nu} \Gamma(\nu/2)}
                    (1+x^2/\nu)^{-(\nu+1)/2}

    where :math:`x` is a real number and the degrees of freedom parameter
    :math:`\nu` (denoted ``df`` in the implementation) satisfies
    :math:`\nu > 0`. :math:`\Gamma` is the gamma function
    (`scipy.special.gamma`).

    %(after_notes)s

    %(example)s

    """

    # 返回参数形状信息的列表
    def _shape_info(self):
        return [_ShapeInfo("df", False, (0, np.inf), (False, False))]

    # 生成随机变量的方法
    def _rvs(self, df, size=None, random_state=None):
        return random_state.standard_t(df, size=size)

    # 概率密度函数的实现
    def _pdf(self, x, df):
        return _lazywhere(
            df == np.inf, (x, df),
            f=lambda x, df: norm._pdf(x),
            f2=lambda x, df: (
                np.exp(self._logpdf(x, df))
            )
        )

    # 对数概率密度函数的实现
    def _logpdf(self, x, df):

        # 学生 t 分布的对数概率密度函数
        def t_logpdf(x, df):
            return (np.log(sc.poch(0.5 * df, 0.5))
                    - 0.5 * (np.log(df) + np.log(np.pi))
                    - (df + 1)/2*np.log1p(x * x/df))

        # 标准正态分布的对数概率密度函数
        def norm_logpdf(x, df):
            return norm._logpdf(x)

        # 根据条件选择对数概率密度函数
        return _lazywhere(df == np.inf, (x, df, ), f=norm_logpdf, f2=t_logpdf)

    # 累积分布函数的实现
    def _cdf(self, x, df):
        return sc.stdtr(df, x)

    # 生存函数的实现
    def _sf(self, x, df):
        return sc.stdtr(df, -x)

    # 百分位点函数的实现
    def _ppf(self, q, df):
        return sc.stdtrit(df, q)

    # 逆生存函数的实现
    def _isf(self, q, df):
        return -sc.stdtrit(df, q)

    # 统计量的计算
    def _stats(self, df):
        # 当 df 为无穷大时，返回正态分布的统计量 (0.0, 1.0, 0.0, 0.0)
        infinite_df = np.isposinf(df)

        # 均值 mu 的计算
        mu = np.where(df > 1, 0.0, np.inf)

        # 根据不同的 df 范围选择不同的计算方式
        condlist = ((df > 1) & (df <= 2),
                    (df > 2) & np.isfinite(df),
                    infinite_df)
        choicelist = (lambda df: np.broadcast_to(np.inf, df.shape),
                      lambda df: df / (df-2.0),
                      lambda df: np.broadcast_to(1, df.shape))
        mu2 = _lazyselect(condlist, choicelist, (df,), np.nan)

        # 偏度 g1 的计算
        g1 = np.where(df > 3, 0.0, np.nan)

        condlist = ((df > 2) & (df <= 4),
                    (df > 4) & np.isfinite(df),
                    infinite_df)
        choicelist = (lambda df: np.broadcast_to(np.inf, df.shape),
                      lambda df: 6.0 / (df-4.0),
                      lambda df: np.broadcast_to(0, df.shape))
        # 峰度 g2 的计算
        g2 = _lazyselect(condlist, choicelist, (df,), np.nan)

        return mu, mu2, g1, g2
    # 计算熵值的私有方法，参数 df 是自由度
    def _entropy(self, df):
        # 如果自由度为无穷大，则调用 norm._entropy() 方法返回结果
        if df == np.inf:
            return norm._entropy()

        # 定义正常情况下的熵计算函数
        def regular(df):
            # 计算自由度一半
            half = df/2
            # 计算 (自由度+1)/2
            half1 = (df + 1)/2
            # 计算熵值公式
            return (half1*(sc.digamma(half1) - sc.digamma(half))
                    + np.log(np.sqrt(df)*sc.beta(half, 0.5)))

        # 定义自由度很大时的渐近熵计算函数
        def asymptotic(df):
            # 使用渐近展开的熵值公式
            # 根据 Wolfram Alpha 提供的公式:
            # "asymptotic expansion (d+1)/2 * (digamma((d+1)/2) - digamma(d/2))
            #  + log(sqrt(d) * beta(d/2, 1/2))"
            h = (norm._entropy() + 1/df + (df**-2.)/4 - (df**-3.)/6
                 - (df**-4.)/8 + 3/10*(df**-5.) + (df**-6.)/4)
            return h

        # 根据自由度的大小选择使用渐近函数或正常函数计算熵值
        h = _lazywhere(df >= 100, (df, ), f=asymptotic, f2=regular)
        return h
# 创建一个名为 t 的非中心学生 t 分布的实例，使用指定的名称 't'
t = t_gen(name='t')

# 定义一个自定义的连续型随机变量 nct_gen，表示非中心学生 t 分布
class nct_gen(rv_continuous):
    r"""A non-central Student's t continuous random variable.

    %(before_notes)s

    Notes
    -----
    If :math:`Y` is a standard normal random variable and :math:`V` is
    an independent chi-square random variable (`chi2`) with :math:`k` degrees
    of freedom, then

    .. math::

        X = \frac{Y + c}{\sqrt{V/k}}

    has a non-central Student's t distribution on the real line.
    The degrees of freedom parameter :math:`k` (denoted ``df`` in the
    implementation) satisfies :math:`k > 0` and the noncentrality parameter
    :math:`c` (denoted ``nc`` in the implementation) is a real number.

    %(after_notes)s

    %(example)s

    """

    # 定义 _argcheck 方法，用于检查参数 df 和 nc 的有效性
    def _argcheck(self, df, nc):
        return (df > 0) & (nc == nc)

    # 定义 _shape_info 方法，返回参数 df 和 nc 的形状信息
    def _shape_info(self):
        idf = _ShapeInfo("df", False, (0, np.inf), (False, False))
        inc = _ShapeInfo("nc", False, (-np.inf, np.inf), (False, False))
        return [idf, inc]

    # 定义 _rvs 方法，生成非中心学生 t 分布的随机变量
    def _rvs(self, df, nc, size=None, random_state=None):
        n = norm.rvs(loc=nc, size=size, random_state=random_state)
        c2 = chi2.rvs(df, size=size, random_state=random_state)
        return n * np.sqrt(df) / np.sqrt(c2)

    # 定义 _pdf 方法，计算非中心学生 t 分布的概率密度函数
    def _pdf(self, x, df, nc):
        # Boost 版本在左尾部分存在精度问题；参见 gh-16591
        n = df*1.0
        nc = nc*1.0
        x2 = x*x
        ncx2 = nc*nc*x2
        fac1 = n + x2
        trm1 = (n/2.*np.log(n) + sc.gammaln(n+1)
                - (n*np.log(2) + nc*nc/2 + (n/2)*np.log(fac1)
                   + sc.gammaln(n/2)))
        Px = np.exp(trm1)
        valF = ncx2 / (2*fac1)
        trm1 = (np.sqrt(2)*nc*x*sc.hyp1f1(n/2+1, 1.5, valF)
                / np.asarray(fac1*sc.gamma((n+1)/2)))
        trm2 = (sc.hyp1f1((n+1)/2, 0.5, valF)
                / np.asarray(np.sqrt(fac1)*sc.gamma(n/2+1)))
        Px *= trm1+trm2
        return np.clip(Px, 0, None)

    # 定义 _cdf 方法，计算非中心学生 t 分布的累积分布函数
    def _cdf(self, x, df, nc):
        with np.errstate(over='ignore'):  # 参见 gh-17432
            return np.clip(scu._nct_cdf(x, df, nc), 0, 1)

    # 定义 _ppf 方法，计算非中心学生 t 分布的百分位点函数
    def _ppf(self, q, df, nc):
        with np.errstate(over='ignore'):  # 参见 gh-17432
            return scu._nct_ppf(q, df, nc)

    # 定义 _sf 方法，计算非中心学生 t 分布的生存函数
    def _sf(self, x, df, nc):
        with np.errstate(over='ignore'):  # 参见 gh-17432
            return np.clip(scu._nct_sf(x, df, nc), 0, 1)

    # 定义 _isf 方法，计算非中心学生 t 分布的逆生存函数
    def _isf(self, x, df, nc):
        with np.errstate(over='ignore'):  # 参见 gh-17432
            return scu._nct_isf(x, df, nc)

    # 定义 _stats 方法，计算非中心学生 t 分布的统计量
    def _stats(self, df, nc, moments='mv'):
        mu = scu._nct_mean(df, nc)
        mu2 = scu._nct_variance(df, nc)
        g1 = scu._nct_skewness(df, nc) if 's' in moments else None
        g2 = scu._nct_kurtosis_excess(df, nc) if 'k' in moments else None
        return mu, mu2, g1, g2

# 创建一个名为 nct 的非中心学生 t 分布的实例，使用指定的名称 "nct"
nct = nct_gen(name="nct")

# 定义一个自定义的 Pareto 连续型随机变量 pareto_gen
class pareto_gen(rv_continuous):
    r"""A Pareto continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `pareto` is:
    """
    定义一个自定义的 Pareto 分布类，继承于 rv_continuous 类。

    方法 _shape_info(self):
        返回一个描述参数形状的 _ShapeInfo 对象列表，这里包含了参数 'b' 的信息。

    方法 _pdf(self, x, b):
        计算 Pareto 分布的概率密度函数，公式为 b / x**(b+1)。

    方法 _cdf(self, x, b):
        计算 Pareto 分布的累积分布函数，公式为 1 - x**(-b)。

    方法 _ppf(self, q, b):
        计算 Pareto 分布的分位函数，公式为 (1-q)**(-1.0/b)。

    方法 _sf(self, x, b):
        计算 Pareto 分布的生存函数，公式为 x**(-b)。

    方法 _isf(self, q, b):
        计算 Pareto 分布的逆生存函数，公式为 q**(-1.0/b)。

    方法 _stats(self, b, moments='mv'):
        计算 Pareto 分布的统计特性，包括期望、方差偏度和峰度。
        mu 是期望，mu2 是方差，g1 是偏度，g2 是峰度。

    方法 _entropy(self, c):
        计算 Pareto 分布的熵，公式为 1 + 1.0/c - np.log(c)。

    装饰器 @_call_super_mom 和 @inherit_docstring_from(rv_continuous) 分别调用继承类的方法和继承的文档字符串。
    """
# 使用给定参数a=1.0和名称"pareto"创建 Pareto 分布对象
pareto = pareto_gen(a=1.0, name="pareto")

# 定义一个 Lomax (Pareto 第二类) 连续随机变量的类，继承自 rv_continuous
class lomax_gen(rv_continuous):
    r"""A Lomax (Pareto of the second kind) continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `lomax` is:

    .. math::

        f(x, c) = \frac{c}{(1+x)^{c+1}}

    for :math:`x \ge 0`, :math:`c > 0`.

    `lomax` takes ``c`` as a shape parameter for :math:`c`.

    `lomax` is a special case of `pareto` with ``loc=-1.0``.

    %(after_notes)s

    %(example)s

    """
    # 定义一个方法，返回形状参数的信息
    def _shape_info(self):
        return [_ShapeInfo("c", False, (0, np.inf), (False, False))]

    # 定义概率密度函数 (PDF) 方法，计算 Lomax 分布的概率密度
    def _pdf(self, x, c):
        # lomax.pdf(x, c) = c / (1+x)**(c+1)
        return c * 1.0 / (1.0 + x) ** (c + 1.0)

    # 定义对数概率密度函数 (logPDF) 方法，计算 Lomax 分布的对数概率密度
    def _logpdf(self, x, c):
        return np.log(c) - (c + 1) * sc.log1p(x)

    # 定义累积分布函数 (CDF) 方法，计算 Lomax 分布的累积分布函数
    def _cdf(self, x, c):
        return -sc.expm1(-c * sc.log1p(x))

    # 定义生存函数 (SF) 方法，计算 Lomax 分布的生存函数
    def _sf(self, x, c):
        return np.exp(-c * sc.log1p(x))

    # 定义对数生存函数 (logSF) 方法，计算 Lomax 分布的对数生存函数
    def _logsf(self, x, c):
        return -c * sc.log1p(x)

    # 定义反函数 (PPF) 方法，计算 Lomax 分布的分位点函数
    def _ppf(self, q, c):
        return sc.expm1(-sc.log1p(-q) / c)

    # 定义逆生存函数 (ISF) 方法，计算 Lomax 分布的逆生存函数
    def _isf(self, q, c):
        return q ** (-1.0 / c) - 1

    # 定义统计量 (期望、方差、偏度和峰度) 方法，利用 Pareto 分布的统计量
    def _stats(self, c):
        mu, mu2, g1, g2 = pareto.stats(c, loc=-1.0, moments='mvsk')
        return mu, mu2, g1, g2

    # 定义熵方法，计算 Lomax 分布的熵
    def _entropy(self, c):
        return 1 + 1.0 / c - np.log(c)

# 使用给定参数a=0.0和名称"lomax"创建 Lomax 分布对象
lomax = lomax_gen(a=0.0, name="lomax")

# 定义一个 Pearson 第三型连续随机变量的类，继承自 rv_continuous
class pearson3_gen(rv_continuous):
    r"""A pearson type III continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `pearson3` is:

    .. math::

        f(x, \kappa) = \frac{|\beta|}{\Gamma(\alpha)}
                       (\beta (x - \zeta))^{\alpha - 1}
                       \exp(-\beta (x - \zeta))

    where:

    .. math::

            \beta = \frac{2}{\kappa}

            \alpha = \beta^2 = \frac{4}{\kappa^2}

            \zeta = -\frac{\alpha}{\beta} = -\beta

    :math:`\Gamma` is the gamma function (`scipy.special.gamma`).
    Pass the skew :math:`\kappa` into `pearson3` as the shape parameter
    ``skew``.

    %(after_notes)s

    %(example)s

    References
    ----------
    R.W. Vogel and D.E. McMartin, "Probability Plot Goodness-of-Fit and
    Skewness Estimation Procedures for the Pearson Type 3 Distribution", Water
    Resources Research, Vol.27, 3149-3158 (1991).

    L.R. Salvosa, "Tables of Pearson's Type III Function", Ann. Math. Statist.,
    Vol.1, 191-198 (1930).

    "Using Modern Computing Tools to Fit the Pearson Type III Distribution to
    Aviation Loads Data", Office of Aviation Research (2003).

    """
    def _preprocess(self, x, skew):
        # 在调用pdf(...)时处理真正的 'loc' 和 'scale'。pearson3._pdf 内部的
        # 'loc' 和 'scale' 变量被设定为默认值，仅作为方程的一部分用于文档说明。
        loc = 0.0  # 设定 'loc' 的默认值为 0.0
        scale = 1.0  # 设定 'scale' 的默认值为 1.0

        # 如果 skew 很小，返回 _norm_pdf。通过 brute force 发现 pearson3 和 norm
        # 之间的分界约为 skew = 0.000016。希望没有人会使用接近这么小的 skew 值。
        norm2pearson_transition = 0.000016

        # 使用 np.broadcast_arrays 以确保 ans、x、skew 有相同的广播形状
        ans, x, skew = np.broadcast_arrays(1.0, x, skew)
        ans = ans.copy()  # 复制 ans 数组，确保不改变原始数据

        # 创建一个掩码，标识那些 skew 足够小以使用正态近似的位置
        mask = np.absolute(skew) < norm2pearson_transition
        invmask = ~mask  # 反转 mask，标识 skew 不足够小以使用正态近似

        # 计算 beta、alpha 和 zeta 参数
        beta = 2.0 / (skew[invmask] * scale)
        alpha = (scale * beta)**2
        zeta = loc - alpha / beta

        # 对于函数调用者返回 ans、x、transx、mask、invmask、beta、alpha 和 zeta
        transx = beta * (x[invmask] - zeta)
        return ans, x, transx, mask, invmask, beta, alpha, zeta

    def _argcheck(self, skew):
        # rv_continuous 中的 _argcheck 函数仅允许正参数。但 pearson3 中的 skew
        # 参数可以是零（我希望在 pearson3._pdf 中处理）或负数。因此，对于所有
        # skew 参数，返回 True。
        return np.isfinite(skew)

    def _shape_info(self):
        # 返回一个描述形状的列表，包含 skew 参数的信息
        return [_ShapeInfo("skew", False, (-np.inf, np.inf), (False, False))]

    def _stats(self, skew):
        # 返回 pearson3 分布的均值、方差、偏度和峰度参数
        m = 0.0
        v = 1.0
        s = skew
        k = 1.5 * skew**2
        return m, v, s, k

    def _pdf(self, x, skew):
        # 计算 pearson3 分布的概率密度函数值。使用 _logpdf 进行计算以限制溢出/
        # 下溢问题。
        ans = np.exp(self._logpdf(x, skew))
        if ans.ndim == 0:
            if np.isnan(ans):
                return 0.0
            return ans
        ans[np.isnan(ans)] = 0.0
        return ans

    def _logpdf(self, x, skew):
        # 计算 pearson3 分布的对数概率密度函数值。
        # 对于 skew 较小的情况，使用 _norm_pdf 替代 _logpdf。
        ans, x, transx, mask, invmask, beta, alpha, _ = (
            self._preprocess(x, skew))

        ans[mask] = np.log(_norm_pdf(x[mask]))  # 使用 _norm_pdf 替代 _logpdf，修复 gh-12640 中提到的问题（对于 alpha = 1，_logpdf 不返回正确的结果）
        ans[invmask] = np.log(abs(beta)) + gamma.logpdf(transx, alpha)
        return ans
    # 定义一个方法 `_cdf`，用于计算累积分布函数 (CDF)
    def _cdf(self, x, skew):
        # 调用 `_preprocess` 方法预处理输入数据，获取必要的变量
        ans, x, transx, mask, invmask, _, alpha, _ = (
            self._preprocess(x, skew))

        # 对于属于支持范围内的数据，使用标准正态分布的累积分布函数
        ans[mask] = _norm_cdf(x[mask])

        # 根据 skew 的正负值，使用不同的方法计算 CDF，以解决特定问题
        skew = np.broadcast_to(skew, invmask.shape)
        invmask1a = np.logical_and(invmask, skew > 0)
        invmask1b = skew[invmask] > 0
        # 使用 gamma 分布的 CDF 替代 _cdf 方法，修复 gh-12640 中提到的问题
        ans[invmask1a] = gamma.cdf(transx[invmask1b], alpha[invmask1b])

        # 处理 skew 为负值时的情况，因为 gamma._cdf 方法不适用于负 skew
        invmask2a = np.logical_and(invmask, skew < 0)
        invmask2b = skew[invmask] < 0
        # 使用 gamma 分布的 SF 方法来计算 CDF，解决 transx < 0 时产生 NaN 的问题
        ans[invmask2a] = gamma.sf(transx[invmask2b], alpha[invmask2b])

        # 返回计算结果
        return ans

    # 定义一个方法 `_sf`，用于计算生存函数 (SF)
    def _sf(self, x, skew):
        # 调用 `_preprocess` 方法预处理输入数据，获取必要的变量
        ans, x, transx, mask, invmask, _, alpha, _ = (
            self._preprocess(x, skew))

        # 对于属于支持范围内的数据，使用标准正态分布的生存函数
        ans[mask] = _norm_sf(x[mask])

        # 根据 skew 的正负值，使用不同的方法计算 SF
        skew = np.broadcast_to(skew, invmask.shape)
        invmask1a = np.logical_and(invmask, skew > 0)
        invmask1b = skew[invmask] > 0
        # 使用 gamma 分布的 SF 方法来计算 SF
        ans[invmask1a] = gamma.sf(transx[invmask1b], alpha[invmask1b])

        invmask2a = np.logical_and(invmask, skew < 0)
        invmask2b = skew[invmask] < 0
        # 使用 gamma 分布的 CDF 方法来计算 SF
        ans[invmask2a] = gamma.cdf(transx[invmask2b], alpha[invmask2b])

        # 返回计算结果
        return ans

    # 定义一个方法 `_rvs`，用于生成随机变量
    def _rvs(self, skew, size=None, random_state=None):
        # 将 skew 广播到指定大小
        skew = np.broadcast_to(skew, size)
        # 调用 `_preprocess` 方法预处理输入数据，获取必要的变量
        ans, _, _, mask, invmask, beta, alpha, zeta = (
            self._preprocess([0], skew))

        # 统计属于两种分布的样本数量
        nsmall = mask.sum()
        nbig = mask.size - nsmall
        # 对于属于标准正态分布的样本，使用标准正态分布生成随机数
        ans[mask] = random_state.standard_normal(nsmall)
        # 对于属于 gamma 分布的样本，使用 gamma 分布生成随机数
        ans[invmask] = random_state.standard_gamma(alpha, nbig)/beta + zeta

        # 如果 size 是 ()，则返回单个值而不是数组
        if size == ():
            ans = ans[0]
        return ans

    # 定义一个方法 `_ppf`，用于计算百分点函数 (PPF)
    def _ppf(self, q, skew):
        # 调用 `_preprocess` 方法预处理输入数据，获取必要的变量
        ans, q, _, mask, invmask, beta, alpha, zeta = (
            self._preprocess(q, skew))
        # 对于属于支持范围内的数据，使用标准正态分布的百分点函数
        ans[mask] = _norm_ppf(q[mask])

        # 处理 skew 为负值时的情况，根据 gh-17050 中的说明进行调整
        q = q[invmask]
        q[beta < 0] = 1 - q[beta < 0]
        # 使用 gamma 分布的反函数来计算百分点函数
        ans[invmask] = sc.gammaincinv(alpha, q)/beta + zeta

        # 返回计算结果
        return ans

    # 使用装饰器扩展备注信息，覆盖继承自父类的 `fit` 方法
    @_call_super_mom
    @extend_notes_in_docstring(rv_continuous, notes="""\
        注意，矩估计法 (`method='MM'`) 在此分布中不可用。\n\n""")
    def fit(self, data, *args, **kwds):
        # 如果参数中包含 `method='MM'`，则抛出 NotImplementedError
        if kwds.get("method", None) == 'MM':
            raise NotImplementedError("Fit `method='MM'` is not available for "
                                      "the Pearson3 distribution. Please try "
                                      "the default `method='MLE'`.")
        else:
            # 否则，调用父类的 `fit` 方法
            return super(type(self), self).fit(data, *args, **kwds)
# 创建一个名为 pearson3 的随机变量生成器，基于 Pearson III 分布
pearson3 = pearson3_gen(name="pearson3")

# 定义一个名为 powerlaw_gen 的类，继承自 rv_continuous 类，表示一个幂函数连续随机变量
class powerlaw_gen(rv_continuous):
    r"""A power-function continuous random variable.

    %(before_notes)s

    See Also
    --------
    pareto

    Notes
    -----
    The probability density function for `powerlaw` is:

    .. math::

        f(x, a) = a x^{a-1}

    for :math:`0 \le x \le 1`, :math:`a > 0`.

    `powerlaw` takes ``a`` as a shape parameter for :math:`a`.

    %(after_notes)s

    For example, the support of `powerlaw` can be adjusted from the default
    interval ``[0, 1]`` to the interval ``[c, c+d]`` by setting ``loc=c`` and
    ``scale=d``. For a power-law distribution with infinite support, see
    `pareto`.

    `powerlaw` is a special case of `beta` with ``b=1``.

    %(example)s

    """
    
    # 返回一个描述形状参数的 _ShapeInfo 对象列表
    def _shape_info(self):
        return [_ShapeInfo("a", False, (0, np.inf), (False, False))]

    # 返回概率密度函数 pdf(x, a) = a * x**(a-1)
    def _pdf(self, x, a):
        return a*x**(a-1.0)

    # 返回对数概率密度函数 logpdf(x, a) = log(a) + xlogy(a - 1, x)
    def _logpdf(self, x, a):
        return np.log(a) + sc.xlogy(a - 1, x)

    # 返回累积分布函数 cdf(x, a) = x**(a*1.0)
    def _cdf(self, x, a):
        return x**(a*1.0)

    # 返回对数累积分布函数 logcdf(x, a) = a * log(x)
    def _logcdf(self, x, a):
        return a*np.log(x)

    # 返回分位点函数 ppf(q, a) = q**(1.0/a)
    def _ppf(self, q, a):
        return pow(q, 1.0/a)

    # 返回生存函数 sf(p, a) = -powm1(p, a)
    def _sf(self, p, a):
        return -sc.powm1(p, a)

    # 返回矩（矩部分） _munp(n, a) = a / (a + n)
    def _munp(self, n, a):
        return a / (a + n)

    # 返回统计量 _stats(a)
    def _stats(self, a):
        return (a / (a + 1.0),
                a / (a + 2.0) / (a + 1.0) ** 2,
                -2.0 * ((a - 1.0) / (a + 3.0)) * np.sqrt((a + 2.0) / a),
                6 * np.polyval([1, -1, -6, 2], a) / (a * (a + 3.0) * (a + 4)))

    # 返回熵 _entropy(a) = 1 - 1.0/a - log(a)
    def _entropy(self, a):
        return 1 - 1.0/a - np.log(a)

    # 返回支持掩码 _support_mask(x, a)
    def _support_mask(self, x, a):
        return (super()._support_mask(x, a)
                & ((x != 0) | (a >= 1)))

    # 装饰符修饰的函数，扩展了父类 rv_continuous 的文档注释中的笔记部分，用于 powerlaw.fit
    @_call_super_mom
    @extend_notes_in_docstring(rv_continuous, notes="""\
        Notes specifically for ``powerlaw.fit``: If the location is a free
        parameter and the value returned for the shape parameter is less than
        one, the true maximum likelihood approaches infinity. This causes
        numerical difficulties, and the resulting estimates are approximate.
        \n\n""")
# 创建一个名为 powerlaw 的幂函数随机变量生成器，a=0.0, b=1.0，表示一个 [0, 1] 区间的幂函数分布
powerlaw = powerlaw_gen(a=0.0, b=1.0, name="powerlaw")

# 定义一个名为 powerlognorm_gen 的类，继承自 rv_continuous 类，表示一个幂函数对数正态连续随机变量
class powerlognorm_gen(rv_continuous):
    r"""A power log-normal continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `powerlognorm` is:

    .. math::

        f(x, c, s) = \frac{c}{x s} \phi(\log(x)/s)
                     (\Phi(-\log(x)/s))^{c-1}

    where :math:`\phi` is the normal pdf, and :math:`\Phi` is the normal cdf,
    and :math:`x > 0`, :math:`s, c > 0`.

    `powerlognorm` takes :math:`c` and :math:`s` as shape parameters.

    %(after_notes)s

    %(example)s

    """
    
    # 支持掩码，从父类 rv_continuous 继承的开放支持掩码
    _support_mask = rv_continuous._open_support_mask
    # 返回两个 _ShapeInfo 对象的列表，表示分布的形状信息
    def _shape_info(self):
        ic = _ShapeInfo("c", False, (0, np.inf), (False, False))
        i_s = _ShapeInfo("s", False, (0, np.inf), (False, False))
        return [ic, i_s]

    # 计算概率密度函数 (PDF)，使用对数形式以防止数值溢出
    def _pdf(self, x, c, s):
        return np.exp(self._logpdf(x, c, s))

    # 计算对数概率密度函数 (logPDF)
    def _logpdf(self, x, c, s):
        return (np.log(c) - np.log(x) - np.log(s) +
                _norm_logpdf(np.log(x) / s) +
                _norm_logcdf(-np.log(x) / s) * (c - 1.))

    # 计算累积分布函数 (CDF)
    def _cdf(self, x, c, s):
        return -sc.expm1(self._logsf(x, c, s))

    # 计算百分点函数 (PPF)，通过逆函数计算给定概率下的对应分位点
    def _ppf(self, q, c, s):
        return self._isf(1 - q, c, s)

    # 计算生存函数 (SF)，即 1 - CDF
    def _sf(self, x, c, s):
        return np.exp(self._logsf(x, c, s))

    # 计算对数生存函数 (logSF)
    def _logsf(self, x, c, s):
        return _norm_logcdf(-np.log(x) / s) * c

    # 计算逆生存函数 (ISF)，通过逆函数计算给定生存概率下的对应分位点
    def _isf(self, q, c, s):
        return np.exp(-_norm_ppf(q**(1/c)) * s)
# 定义一个名为 powerlognorm 的概率分布生成器，对应的分布是 Power Log-Normal 分布
powerlognorm = powerlognorm_gen(a=0.0, name="powerlognorm")

# 定义一个名为 powernorm_gen 的类，继承于 rv_continuous 类，表示一个 Power Normal 分布的连续随机变量
class powernorm_gen(rv_continuous):
    r"""A power normal continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `powernorm` is:

    .. math::

        f(x, c) = c \phi(x) (\Phi(-x))^{c-1}

    where :math:`\phi` is the normal pdf, :math:`\Phi` is the normal cdf,
    :math:`x` is any real, and :math:`c > 0` [1]_.

    `powernorm` takes ``c`` as a shape parameter for :math:`c`.

    %(after_notes)s

    References
    ----------
    .. [1] NIST Engineering Statistics Handbook, Section 1.3.6.6.13,
           https://www.itl.nist.gov/div898/handbook//eda/section3/eda366d.htm

    %(example)s

    """
    # 定义 _shape_info 方法，返回一个描述参数形状的对象列表
    def _shape_info(self):
        return [_ShapeInfo("c", False, (0, np.inf), (False, False))]

    # 定义 _pdf 方法，计算概率密度函数 f(x, c)
    def _pdf(self, x, c):
        # powernorm.pdf(x, c) = c * phi(x) * (Phi(-x))**(c-1)
        return c * _norm_pdf(x) * (_norm_cdf(-x)**(c-1.0))

    # 定义 _logpdf 方法，计算对数概率密度函数的值
    def _logpdf(self, x, c):
        return np.log(c) + _norm_logpdf(x) + (c-1) * _norm_logcdf(-x)

    # 定义 _cdf 方法，计算累积分布函数的值
    def _cdf(self, x, c):
        return -sc.expm1(self._logsf(x, c))

    # 定义 _ppf 方法，计算累积分布函数的反函数
    def _ppf(self, q, c):
        return -_norm_ppf(pow(1.0 - q, 1.0 / c))

    # 定义 _sf 方法，计算生存函数（1 - CDF）的值
    def _sf(self, x, c):
        return np.exp(self._logsf(x, c))

    # 定义 _logsf 方法，计算对数生存函数的值
    def _logsf(self, x, c):
        return c * _norm_logcdf(-x)

    # 定义 _isf 方法，计算生存函数的反函数
    def _isf(self, q, c):
        return -_norm_ppf(np.exp(np.log(q) / c))

# 创建一个名为 powernorm 的 powernorm_gen 类的实例对象，表示一个 Power Normal 分布的连续随机变量
powernorm = powernorm_gen(name='powernorm')

# 定义一个名为 rdist_gen 的类，继承于 rv_continuous 类，表示一个 R-distributed 分布的连续随机变量
class rdist_gen(rv_continuous):
    r"""An R-distributed (symmetric beta) continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `rdist` is:

    .. math::

        f(x, c) = \frac{(1-x^2)^{c/2-1}}{B(1/2, c/2)}

    for :math:`-1 \le x \le 1`, :math:`c > 0`. `rdist` is also called the
    symmetric beta distribution: if B has a `beta` distribution with
    parameters (c/2, c/2), then X = 2*B - 1 follows a R-distribution with
    parameter c.

    `rdist` takes ``c`` as a shape parameter for :math:`c`.

    This distribution includes the following distribution kernels as
    special cases::

        c = 2:  uniform
        c = 3:  `semicircular`
        c = 4:  Epanechnikov (parabolic)
        c = 6:  quartic (biweight)
        c = 8:  triweight

    %(after_notes)s

    %(example)s

    """
    # 定义 _shape_info 方法，返回一个描述参数形状的对象列表
    def _shape_info(self):
        return [_ShapeInfo("c", False, (0, np.inf), (False, False))]

    # 使用与 beta 分布的关系定义 pdf、cdf 等方法
    def _pdf(self, x, c):
        return np.exp(self._logpdf(x, c))

    def _logpdf(self, x, c):
        return -np.log(2) + beta._logpdf((x + 1) / 2, c / 2, c / 2)

    def _cdf(self, x, c):
        return beta._cdf((x + 1) / 2, c / 2, c / 2)

    def _sf(self, x, c):
        return beta._sf((x + 1) / 2, c / 2, c / 2)

    def _ppf(self, q, c):
        return 2 * beta._ppf(q, c / 2, c / 2) - 1

    def _rvs(self, c, size=None, random_state=None):
        return 2 * random_state.beta(c / 2, c / 2, size) - 1

# 创建一个名为 rdist 的 rdist_gen 类的实例对象，表示一个 R-distributed 分布的连续随机变量
rdist = rdist_gen(name='rdist')
    # 定义一个私有方法 `_munp`，接受两个参数 `n` 和 `c`
    def _munp(self, n, c):
        # 计算分子，根据奇偶性确定系数，然后计算 beta 函数的结果
        numerator = (1 - (n % 2)) * sc.beta((n + 1.0) / 2, c / 2.0)
        # 计算分母，beta 函数的特定参数值
        denominator = sc.beta(1. / 2, c / 2.)
        # 返回分子除以分母的结果作为方法的返回值
        return numerator / denominator
# 生成一个服从 Rayleigh 分布的随机数生成器，参数范围为 [-1.0, 1.0]
rdist = rdist_gen(a=-1.0, b=1.0, name="rdist")

# 定义一个 Rayleigh 分布的连续随机变量类
class rayleigh_gen(rv_continuous):
    r"""A Rayleigh continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `rayleigh` is:

    .. math::

        f(x) = x \exp(-x^2/2)

    for :math:`x \ge 0`.

    `rayleigh` is a special case of `chi` with ``df=2``.

    %(after_notes)s

    %(example)s

    """
    # 支持区域掩码
    _support_mask = rv_continuous._open_support_mask

    # 返回形状信息的空列表
    def _shape_info(self):
        return []

    # 生成 Rayleigh 分布的随机变量
    def _rvs(self, size=None, random_state=None):
        return chi.rvs(2, size=size, random_state=random_state)

    # 概率密度函数，对应于 rayleigh.pdf(r) = r * exp(-r**2/2)
    def _pdf(self, r):
        return np.exp(self._logpdf(r))

    # 对数概率密度函数，对应于 np.log(r) - 0.5 * r * r
    def _logpdf(self, r):
        return np.log(r) - 0.5 * r * r

    # 累积分布函数
    def _cdf(self, r):
        return -sc.expm1(-0.5 * r**2)

    # 百分点函数（CDF 的反函数）
    def _ppf(self, q):
        return np.sqrt(-2 * sc.log1p(-q))

    # 生存函数（1 - CDF）
    def _sf(self, r):
        return np.exp(self._logsf(r))

    # 对数生存函数
    def _logsf(self, r):
        return -0.5 * r * r

    # 逆百分点函数（SF 的反函数）
    def _isf(self, q):
        return np.sqrt(-2 * np.log(q))

    # 分布的统计量
    def _stats(self):
        val = 4 - np.pi
        return (np.sqrt(np.pi/2),
                val/2,
                2*(np.pi-3)*np.sqrt(np.pi)/val**1.5,
                6*np.pi/val-16/val**2)

    # 熵值
    def _entropy(self):
        return _EULER/2.0 + 1 - 0.5*np.log(2)

    # 调用父类的方法来扩展文档字符串中的注释
    @_call_super_mom
    @extend_notes_in_docstring(rv_continuous, notes="""\
        Notes specifically for ``rayleigh.fit``: If the location is fixed with
        the `floc` parameter, this method uses an analytical formula to find
        the scale.  Otherwise, this function uses a numerical root finder on
        the first order conditions of the log-likelihood function to find the
        MLE.  Only the (optional) `loc` parameter is used as the initial guess
        for the root finder; the `scale` parameter and any other parameters
        for the optimizer are ignored.\n\n""")
    # 对象方法 `fit`，用于拟合数据，可以接受任意位置参数和关键字参数
    def fit(self, data, *args, **kwds):
        # 如果关键字参数中存在 'superfit'，并且其为 True，则调用父类的 fit 方法
        if kwds.pop('superfit', False):
            return super().fit(data, *args, **kwds)
        
        # 检查并调整输入参数，获取数据、位置和尺度
        data, floc, fscale = _check_fit_input_parameters(self, data,
                                                         args, kwds)

        # 最大似然估计尺度函数
        def scale_mle(loc):
            # 参考：Statistical Distributions, 3rd Edition. Evans, Hastings,
            # and Peacock (2000), Page 175
            return (np.sum((data - loc) ** 2) / (2 * len(data))) ** .5

        # 最大似然估计位置函数，当位置和尺度均为自由变量时使用
        def loc_mle(loc):
            # 这是隐式方程用于估计 `loc`，当 `loc` 和 `scale` 均为自由变量时使用。
            xm = data - loc
            s1 = xm.sum()
            s2 = (xm**2).sum()
            s3 = (1/xm).sum()
            return s1 - s2/(2*len(data))*s3

        # 最大似然估计位置函数，当尺度固定但位置为自由变量时使用
        def loc_mle_scale_fixed(loc, scale=fscale):
            # 这是隐式方程用于估计 `loc`，当 `scale` 固定但 `loc` 为自由变量时使用。
            xm = data - loc
            return xm.sum() - scale**2 * (1/xm).sum()

        # 如果位置 `floc` 已知，则分析确定尺度 `scale`
        if floc is not None:
            # 如果数据中存在小于等于 `floc` 的值，则引发 FitDataError 异常
            if np.any(data - floc <= 0):
                raise FitDataError("rayleigh", lower=1, upper=np.inf)
            else:
                return floc, scale_mle(floc)

        # 考虑用户提供的位置的初始猜测值 `loc0`
        loc0 = kwds.get('loc')
        if loc0 is None:
            # 使用 _fitstart 方法估计 `loc` 的初始值，并忽略返回的尺度
            loc0 = self._fitstart(data)[0]

        # 根据是否固定尺度，选择适当的估计函数
        fun = loc_mle if fscale is None else loc_mle_scale_fixed
        # 确定 `fun` 函数的左侧边界
        rbrack = np.nextafter(np.min(data), -np.inf)
        lbrack = _get_left_bracket(fun, rbrack)
        # 使用优化方法求解方程 `fun`，并确定位置 `loc`
        res = optimize.root_scalar(fun, bracket=(lbrack, rbrack))
        if not res.converged:
            # 如果未收敛，则引发 FitSolverError 异常，传递 res.flag
            raise FitSolverError(res.flag)
        loc = res.root
        # 确定尺度 `scale`，若 `fscale` 未指定则计算最大似然估计尺度
        scale = fscale or scale_mle(loc)
        # 返回最终确定的位置 `loc` 和尺度 `scale`
        return loc, scale
rayleigh = rayleigh_gen(a=0.0, name="rayleigh")

class reciprocal_gen(rv_continuous):
    r"""A loguniform or reciprocal continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for this class is:

    .. math::

        f(x, a, b) = \frac{1}{x \log(b/a)}

    for :math:`a \le x \le b`, :math:`b > a > 0`. This class takes
    :math:`a` and :math:`b` as shape parameters.

    %(after_notes)s

    %(example)s

    This doesn't show the equal probability of ``0.01``, ``0.1`` and
    ``1``. This is best when the x-axis is log-scaled:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1, 1)
    >>> ax.hist(np.log10(r))
    >>> ax.set_ylabel("Frequency")
    >>> ax.set_xlabel("Value of random variable")
    >>> ax.xaxis.set_major_locator(plt.FixedLocator([-2, -1, 0]))
    >>> ticks = ["$10^{{ {} }}$".format(i) for i in [-2, -1, 0]]
    >>> ax.set_xticklabels(ticks)  # doctest: +SKIP
    >>> plt.show()

    This random variable will be log-uniform regardless of the base chosen for
    ``a`` and ``b``. Let's specify with base ``2`` instead:

    >>> rvs = %(name)s(2**-2, 2**0).rvs(size=1000)

    Values of ``1/4``, ``1/2`` and ``1`` are equally likely with this random
    variable.  Here's the histogram:

    >>> fig, ax = plt.subplots(1, 1)
    >>> ax.hist(np.log2(rvs))
    >>> ax.set_ylabel("Frequency")
    >>> ax.set_xlabel("Value of random variable")
    >>> ax.xaxis.set_major_locator(plt.FixedLocator([-2, -1, 0]))
    >>> ticks = ["$2^{{ {} }}$".format(i) for i in [-2, -1, 0]]
    >>> ax.set_xticklabels(ticks)  # doctest: +SKIP
    >>> plt.show()

    """
    
    # 检查参数是否有效，确保 a > 0 且 b > a
    def _argcheck(self, a, b):
        return (a > 0) & (b > a)

    # 返回参数的形状信息
    def _shape_info(self):
        ia = _ShapeInfo("a", False, (0, np.inf), (False, False))
        ib = _ShapeInfo("b", False, (0, np.inf), (False, False))
        return [ia, ib]

    # 对于给定数据集，返回适当的起始值
    def _fitstart(self, data):
        if isinstance(data, CensoredData):
            data = data._uncensor()
        # 因为支持区间是 [a, b]，所以选择最小和最大值作为起始值是合理的
        return super()._fitstart(data, args=(np.min(data), np.max(data)))

    # 返回支持区间 [a, b]
    def _get_support(self, a, b):
        return a, b

    # 概率密度函数的实现，对应于 reciprocal.pdf(x, a, b) = 1 / (x*(log(b) - log(a)))
    def _pdf(self, x, a, b):
        return np.exp(self._logpdf(x, a, b))

    # 对数概率密度函数的实现，对应于 -log(x) - log(log(b) - log(a))
    def _logpdf(self, x, a, b):
        return -np.log(x) - np.log(np.log(b) - np.log(a))

    # 累积分布函数的实现，对应于 (log(x)-log(a)) / (log(b) - log(a))
    def _cdf(self, x, a, b):
        return (np.log(x)-np.log(a)) / (np.log(b) - np.log(a))

    # 百分点函数的实现，对应于 exp(log(a) + q*(log(b) - log(a)))
    def _ppf(self, q, a, b):
        return np.exp(np.log(a) + q*(np.log(b) - np.log(a)))

    # 非中心矩的实现，对应于 1 / (log(b) - log(a)) / n * exp(_log_diff(n * log(b), n*log(a)))
    def _munp(self, n, a, b):
        t1 = 1 / (np.log(b) - np.log(a)) / n
        t2 = np.real(np.exp(_log_diff(n * np.log(b), n*np.log(a))))
        return t1 * t2

    # 熵的实现，对应于 0.5*(log(a) + log(b)) + log(log(b) - log(a))
    def _entropy(self, a, b):
        return 0.5*(np.log(a) + np.log(b)) + np.log(np.log(b) - np.log(a))
    # 定义一段关于 `loguniform`/`reciprocal` 参数化过多的注释内容，
    # `fit` 方法在用户未提供 `fscale` 参数时，会自动将 `scale` 固定为 1。
    fit_note = """\
        `loguniform`/`reciprocal` is over-parameterized. `fit` automatically
         fixes `scale` to 1 unless `fscale` is provided by the user.\n\n"""

    # 在 `rv_continuous` 的文档字符串中扩展注释内容，使用 `fit_note`
    @extend_notes_in_docstring(rv_continuous, notes=fit_note)
    def fit(self, data, *args, **kwds):
        # 从 `kwds` 中弹出 `fscale` 参数，若不存在则默认为 1
        fscale = kwds.pop('fscale', 1)
        # 调用父类的 `fit` 方法，传递 `data` 和其他参数，确保 `fscale` 正确传递
        return super().fit(data, *args, fscale=fscale, **kwds)

    # 关于不为该分布定义生存函数的详细决定细节可以在以下 PR 中找到：
    # https://github.com/scipy/scipy/pull/18614
loguniform = reciprocal_gen(name="loguniform")
reciprocal = reciprocal_gen(name="reciprocal")

class rice_gen(rv_continuous):
    r"""A Rice continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `rice` is:

    .. math::

        f(x, b) = x \exp(- \frac{x^2 + b^2}{2}) I_0(x b)

    for :math:`x >= 0`, :math:`b > 0`. :math:`I_0` is the modified Bessel
    function of order zero (`scipy.special.i0`).

    `rice` takes ``b`` as a shape parameter for :math:`b`.

    %(after_notes)s

    The Rice distribution describes the length, :math:`r`, of a 2-D vector with
    components :math:`(U+u, V+v)`, where :math:`U, V` are constant, :math:`u,
    v` are independent Gaussian random variables with standard deviation
    :math:`s`.  Let :math:`R = \sqrt{U^2 + V^2}`. Then the pdf of :math:`r` is
    ``rice.pdf(x, R/s, scale=s)``.

    %(example)s

    """
    # 检查参数 `b` 是否符合要求（非负）
    def _argcheck(self, b):
        return b >= 0

    # 返回分布的形状信息，这里是参数 `b` 的约束条件
    def _shape_info(self):
        return [_ShapeInfo("b", False, (0, np.inf), (True, False))]

    # 生成服从 Rice 分布的随机变量
    def _rvs(self, b, size=None, random_state=None):
        # 生成符合正态分布的随机变量，并按照 Rice 分布定义返回其长度
        t = b/np.sqrt(2) + random_state.standard_normal(size=(2,) + size)
        return np.sqrt((t*t).sum(axis=0))

    # 累积分布函数 (CDF) 的实现
    def _cdf(self, x, b):
        return sc.chndtr(np.square(x), 2, np.square(b))

    # 百分点函数 (PPF) 的实现
    def _ppf(self, q, b):
        return np.sqrt(sc.chndtrix(q, 2, np.square(b)))

    # 概率密度函数 (PDF) 的实现
    def _pdf(self, x, b):
        # Rice 分布的概率密度函数公式
        return x * np.exp(-(x-b)*(x-b)/2.0) * sc.i0e(x*b)

    # 非中心矩的计算
    def _munp(self, n, b):
        nd2 = n/2.0
        n1 = 1 + nd2
        b2 = b*b/2.0
        return (2.0**(nd2) * np.exp(-b2) * sc.gamma(n1) *
                sc.hyp1f1(n1, 1, b2))

# 创建一个 Rice 分布的实例对象
rice = rice_gen(a=0.0, name="rice")

class irwinhall_gen(rv_continuous):
    r"""An Irwin-Hall (Uniform Sum) continuous random variable.

    An `Irwin-Hall <https://en.wikipedia.org/wiki/Irwin-Hall_distribution/>`_
    continuous random variable is the sum of :math:`n` independent
    standard uniform random variables [1]_ [2]_.

    %(before_notes)s

    Notes
    -----
    Applications include `Rao's Spacing Test
    <https://jammalam.faculty.pstat.ucsb.edu/html/favorite/test.htm>`_,
    a more powerful alternative to the Rayleigh test
    when the data are not unimodal, and radar [3]_.

    Conveniently, the pdf and cdf are the :math:`n`-fold convolution of
    the ones for the standard uniform distribution, which is also the
    definition of the cardinal B-splines of degree :math:`n-1`
    having knots evenly spaced from :math:`1` to :math:`n` [4]_ [5]_.

    The Bates distribution, which represents the *mean* of statistically
    """  # noqa: E501

    @replace_notes_in_docstring(rv_continuous, notes="""\
        Raises a ``NotImplementedError`` for the Irwin-Hall distribution because
        the generic `fit` implementation is unreliable and no custom implementation
        is available. Consider using `scipy.stats.fit`.\n\n""")
    定义一个修饰器函数，用于替换文档字符串中的注释内容，特别是对于 `rv_continuous` 分布，
    提示该分布的 `fit` 方法抛出 `NotImplementedError` 异常，因为通用的 `fit` 实现不可靠，
    且没有自定义实现可用。建议考虑使用 `scipy.stats.fit`。

    def fit(self, data, *args, **kwds):
        定义 `fit` 方法，该方法抛出 `NotImplementedError` 异常，提示信息说明了对于此分布，
        通用的 `fit` 实现不可靠，并且没有自定义实现可用。

        fit_notes = ("The generic `fit` implementation is unreliable for this "
                     "distribution, and no custom implementation is available. "
                     "Consider using `scipy.stats.fit`.")
        抛出 `NotImplementedError` 异常，异常消息包含了上述提示信息
        raise NotImplementedError(fit_notes)

    def _argcheck(self, n):
        检查参数 `n` 是否符合条件：大于零、为整数且为实数对象
        return (n > 0) & _isintegral(n) & np.isrealobj(n)
    
    def _get_support(self, n):
        返回支持范围，对于该分布是从 0 到 `n` 的闭区间
        return 0, n 
    
    def _shape_info(self):
        返回形状信息列表，描述了参数 `n`，为一个正整数并且可以是无穷大，这是该分布的特性
        return [_ShapeInfo("n", True, (1, np.inf), (True, False))]
    # 定义私有方法 _munp，用于计算 Irwin-Hall 分布的 m 阶矩
    def _munp(self, order, n):
        # 参考文献：https://link.springer.com/content/pdf/10.1007/s10959-020-01050-9.pdf
        # 在第 640 页，其中 m=n, j=n+order
        def vmunp(order, n):
            # 使用斯特林第二类数和组合数准确计算 Irwin-Hall 分布的 m 阶矩
            return (sc.stirling2(n+order, n, exact=True) 
                    / sc.comb(n+order, n, exact=True))

        # 返回向量化的 Irwin-Hall 分布的 m 阶矩
        # 虽然计算得到的是精确的有理数，但我们将其转换为浮点数返回
        return np.vectorize(vmunp, otypes=[np.float64])(order, n)

    @staticmethod
    # 静态方法 _cardbspl，返回阶数为 n 的 B 样条基函数的元素
    def _cardbspl(n):
        t = np.arange(n+1) 
        return BSpline.basis_element(t)

    # 定义私有方法 _pdf，用于计算 Irwin-Hall 分布的概率密度函数
    def _pdf(self, x, n):
        def vpdf(x, n):
            # 返回向量化的 Irwin-Hall 分布的概率密度函数
            return self._cardbspl(n)(x)
        return np.vectorize(vpdf, otypes=[np.float64])(x, n)
    
    # 定义私有方法 _cdf，用于计算 Irwin-Hall 分布的累积分布函数
    def _cdf(self, x, n):
        def vcdf(x, n):
            # 返回向量化的 Irwin-Hall 分布的累积分布函数
            return self._cardbspl(n).antiderivative()(x)
        return np.vectorize(vcdf, otypes=[np.float64])(x, n)

    # 定义私有方法 _sf，用于计算 Irwin-Hall 分布的生存函数
    def _sf(self, x, n):
        def vsf(x, n):
            # 返回向量化的 Irwin-Hall 分布的生存函数
            return self._cardbspl(n).antiderivative()(n-x)
        return np.vectorize(vsf, otypes=[np.float64])(x, n)

    # 定义私有方法 _rvs，用于生成符合 Irwin-Hall 分布的随机变量
    def _rvs(self, n, size=None, random_state=None, *args):
        # 内部函数 _rvs1，用于向量化地生成 Irwin-Hall 分布的随机变量
        @_vectorize_rvs_over_shapes
        def _rvs1(n, size=None, random_state=None):
            # 将 n 取整并转换为整数类型
            n = np.floor(n).astype(int)
            # 根据 size 参数确定返回数组的形状
            usize = (n,) if size is None else (n, *size)
            # 使用随机数生成器 random_state 生成均匀分布随机数并按列求和
            return random_state.uniform(size=usize).sum(axis=0)
        return _rvs1(n, size=size, random_state=random_state)
    
    # 定义私有方法 _stats，用于计算 Irwin-Hall 分布的统计量
    def _stats(self, n):
        # 返回 Irwin-Hall 分布的期望、方差、偏度和峰度
        # 这些统计量的计算基于 n 个独立同分布的均匀分布随机变量的性质
        # 详细推导可以参考 https://en.wikipedia.org/wiki/Irwin%E2%80%93Hall_distribution
        return n/2, n/12, 0, -6/(5*n)
irwinhall = irwinhall_gen(name="irwinhall")    
# 创建一个 Irwin-Hall 分布的生成器对象，命名为 irwinhall

class recipinvgauss_gen(rv_continuous):
    r"""A reciprocal inverse Gaussian continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `recipinvgauss` is:

    .. math::

        f(x, \mu) = \frac{1}{\sqrt{2\pi x}}
                    \exp\left(\frac{-(1-\mu x)^2}{2\mu^2x}\right)

    for :math:`x \ge 0`.

    `recipinvgauss` takes ``mu`` as a shape parameter for :math:`\mu`.

    %(after_notes)s

    %(example)s

    """
    def _shape_info(self):
        # 返回一个描述分布形状信息的对象列表，这里描述了参数 "mu" 的范围和特性
        return [_ShapeInfo("mu", False, (0, np.inf), (False, False))]

    def _pdf(self, x, mu):
        # 返回概率密度函数 (PDF) 的值，这里使用了 reciprocal inverse Gaussian 的 PDF 公式
        return np.exp(self._logpdf(x, mu))

    def _logpdf(self, x, mu):
        # 返回概率密度函数的对数值，使用了一个 lambda 函数计算
        return _lazywhere(x > 0, (x, mu),
                          lambda x, mu: (-(1 - mu*x)**2.0 / (2*x*mu**2.0)
                                         - 0.5*np.log(2*np.pi*x)),
                          fillvalue=-np.inf)

    def _cdf(self, x, mu):
        # 返回累积分布函数 (CDF) 的值，使用了正态分布的 CDF 进行计算
        trm1 = 1.0/mu - x
        trm2 = 1.0/mu + x
        isqx = 1.0/np.sqrt(x)
        return _norm_cdf(-isqx*trm1) - np.exp(2.0/mu)*_norm_cdf(-isqx*trm2)

    def _sf(self, x, mu):
        # 返回生存函数 (1 - CDF) 的值，同样使用了正态分布的 CDF 进行计算
        trm1 = 1.0/mu - x
        trm2 = 1.0/mu + x
        isqx = 1.0/np.sqrt(x)
        return _norm_cdf(isqx*trm1) + np.exp(2.0/mu)*_norm_cdf(-isqx*trm2)

    def _rvs(self, mu, size=None, random_state=None):
        # 返回随机变量 (Random Variates, RVs)，使用了 Wald 分布的随机变量生成方法
        return 1.0/random_state.wald(mu, 1.0, size=size)


recipinvgauss = recipinvgauss_gen(a=0.0, name='recipinvgauss')


class semicircular_gen(rv_continuous):
    r"""A semicircular continuous random variable.

    %(before_notes)s

    See Also
    --------
    rdist

    Notes
    -----
    The probability density function for `semicircular` is:

    .. math::

        f(x) = \frac{2}{\pi} \sqrt{1-x^2}

    for :math:`-1 \le x \le 1`.

    The distribution is a special case of `rdist` with ``c = 3``.

    %(after_notes)s

    References
    ----------
    .. [1] "Wigner semicircle distribution",
           https://en.wikipedia.org/wiki/Wigner_semicircle_distribution

    %(example)s

    """
    def _shape_info(self):
        # 返回空列表，说明此分布没有额外的形状信息
        return []

    def _pdf(self, x):
        # 返回概率密度函数 (PDF) 的值，这里是半圆分布的 PDF 公式
        return 2.0/np.pi*np.sqrt(1-x*x)

    def _logpdf(self, x):
        # 返回概率密度函数的对数值，使用了 numpy 中的对数函数和 log1p 函数
        return np.log(2/np.pi) + 0.5*sc.log1p(-x*x)

    def _cdf(self, x):
        # 返回累积分布函数 (CDF) 的值，使用了反正弦函数和平方根函数
        return 0.5+1.0/np.pi*(x*np.sqrt(1-x*x) + np.arcsin(x))

    def _ppf(self, q):
        # 返回百分位点函数 (PPF) 的值，使用了另一个分布 rdist 的 PPF 函数
        return rdist._ppf(q, 3)

    def _rvs(self, size=None, random_state=None):
        # 返回随机变量 (Random Variates, RVs)，使用了随机生成半圆分布的方法
        # 先生成均匀分布的随机数，然后通过变换得到半圆分布的随机数
        r = np.sqrt(random_state.uniform(size=size))
        a = np.cos(np.pi * random_state.uniform(size=size))
        return r * a

    def _stats(self):
        # 返回分布的统计特性，这里是半圆分布的均值、方差、偏度和峰度
        return 0, 0.25, 0, -1.0

    def _entropy(self):
        # 返回分布的信息熵
        return 0.64472988584940017414
# 创建一个名为 semicircular 的变量，调用 semicircular_gen 函数生成一个半圆分布的随机变量对象，参数 a 和 b 控制分布的范围，name 指定名称
semicircular = semicircular_gen(a=-1.0, b=1.0, name="semicircular")

# 定义一个名为 skewcauchy_gen 的类，继承于 rv_continuous 类，表示一个偏斜的柯西分布随机变量
class skewcauchy_gen(rv_continuous):
    r"""A skewed Cauchy random variable.

    %(before_notes)s

    See Also
    --------
    cauchy : Cauchy distribution

    Notes
    -----

    The probability density function for `skewcauchy` is:

    .. math::

        f(x) = \\frac{1}{\\pi \\left(\\frac{x^2}{\\left(a\\, \\text{sign}(x) + 1
                                                   \\right)^2} + 1 \\right)}

    for a real number :math:`x` and skewness parameter :math:`-1 < a < 1`.

    When :math:`a=0`, the distribution reduces to the usual Cauchy
    distribution.

    %(after_notes)s

    References
    ----------
    .. [1] "Skewed generalized *t* distribution", Wikipedia
       https://en.wikipedia.org/wiki/Skewed_generalized_t_distribution#Skewed_Cauchy_distribution

    %(example)s

    """
    
    # 定义 _argcheck 方法，用于检查参数 a 的有效性，要求 a 的绝对值小于 1
    def _argcheck(self, a):
        return np.abs(a) < 1

    # 定义 _shape_info 方法，返回参数信息，此处返回一个 _ShapeInfo 对象列表，描述参数 a 的特征
    def _shape_info(self):
        return [_ShapeInfo("a", False, (-1.0, 1.0), (False, False))]

    # 定义 _pdf 方法，计算概率密度函数，根据参数 a 和变量 x 计算偏斜柯西分布的概率密度值
    def _pdf(self, x, a):
        return 1 / (np.pi * (x**2 / (a * np.sign(x) + 1)**2 + 1))

    # 定义 _cdf 方法，计算累积分布函数，根据参数 a 和变量 x 计算偏斜柯西分布的累积分布值
    def _cdf(self, x, a):
        return np.where(x <= 0,
                        (1 - a) / 2 + (1 - a) / np.pi * np.arctan(x / (1 - a)),
                        (1 - a) / 2 + (1 + a) / np.pi * np.arctan(x / (1 + a)))

    # 定义 _ppf 方法，计算反函数，根据参数 a 和累积概率 x 计算对应的分位点
    def _ppf(self, x, a):
        i = x < self._cdf(0, a)
        return np.where(i,
                        np.tan(np.pi / (1 - a) * (x - (1 - a) / 2)) * (1 - a),
                        np.tan(np.pi / (1 + a) * (x - (1 - a) / 2)) * (1 + a))

    # 定义 _stats 方法，返回统计信息，此处返回空值，表示无法计算平均值、方差等统计量
    def _stats(self, a, moments='mvsk'):
        return np.nan, np.nan, np.nan, np.nan

    # 定义 _fitstart 方法，用于估计参数的起始值，基于输入的数据 data 计算估计值，用于拟合分布
    def _fitstart(self, data):
        # Use 0 as the initial guess of the skewness shape parameter.
        # For the location and scale, estimate using the median and
        # quartiles.
        if isinstance(data, CensoredData):  # 如果输入数据是被屏蔽的数据类型，解除屏蔽
            data = data._uncensor()
        p25, p50, p75 = np.percentile(data, [25, 50, 75])  # 计算数据的第25、50、75百分位数
        return 0.0, p50, (p75 - p25)/2  # 返回估计的参数起始值

# 创建一个名为 skewcauchy 的变量，表示一个偏斜柯西分布的随机变量对象
skewcauchy = skewcauchy_gen(name='skewcauchy')

# 定义一个名为 skewnorm_gen 的类，继承于 rv_continuous 类，表示一个偏斜正态分布随机变量
class skewnorm_gen(rv_continuous):
    r"""A skew-normal random variable.

    %(before_notes)s

    Notes
    -----
    The pdf is::

        skewnorm.pdf(x, a) = 2 * norm.pdf(x) * norm.cdf(a*x)

    `skewnorm` takes a real number :math:`a` as a skewness parameter
    When ``a = 0`` the distribution is identical to a normal distribution
    (`norm`). `rvs` implements the method of [1]_.

    %(after_notes)s

    %(example)s

    References
    ----------
    .. [1] A. Azzalini and A. Capitanio (1999). Statistical applications of
        the multivariate skew-normal distribution. J. Roy. Statist. Soc.,
        B 61, 579-602. :arxiv:`0911.2093`

    """
    
    # 定义 _argcheck 方法，检查参数 a 的有效性，要求 a 是有限的数值
    def _argcheck(self, a):
        return np.isfinite(a)

    # 定义 _shape_info 方法，返回参数信息，此处返回一个 _ShapeInfo 对象列表，描述参数 a 的特征
    def _shape_info(self):
        return [_ShapeInfo("a", False, (-np.inf, np.inf), (False, False))]
    # 计算 skew-normal 分布的概率密度函数 (PDF)。
    def _pdf(self, x, a):
        return _lazywhere(
            a == 0, (x, a), lambda x, a: _norm_pdf(x),
            f2=lambda x, a: 2.*_norm_pdf(x)*_norm_cdf(a*x)
        )

    # 计算 skew-normal 分布的对数概率密度函数 (log PDF)。
    def _logpdf(self, x, a):
        return _lazywhere(
            a == 0, (x, a), lambda x, a: _norm_logpdf(x),
            f2=lambda x, a: np.log(2)+_norm_logpdf(x)+_norm_logcdf(a*x),
        )

    # 计算 skew-normal 分布的累积分布函数 (CDF)。
    def _cdf(self, x, a):
        # 将 a 转换为至少一维数组
        a = np.atleast_1d(a)
        # 调用 scipy.stats 中的 _skewnorm_cdf 计算 CDF
        cdf = scu._skewnorm_cdf(x, 0.0, 1.0, a)
        # 如果 a > 0 且 CDF 很小，则使用超类的 _cdf 方法
        i_small_cdf = (cdf < 1e-6) & (a > 0)
        cdf[i_small_cdf] = super()._cdf(x[i_small_cdf], a[i_small_cdf])
        # 将 CDF 值限制在 [0, 1] 之间并返回
        return np.clip(cdf, 0, 1)

    # 计算 skew-normal 分布的百分点函数 (percent point function, PPF)。
    def _ppf(self, x, a):
        return scu._skewnorm_ppf(x, 0.0, 1.0, a)

    # 计算 skew-normal 分布的生存函数 (survival function, SF)。
    def _sf(self, x, a):
        # 使用 _cdf 方法的定制版本来计算 SF
        return self._cdf(-x, -a)

    # 计算 skew-normal 分布的逆生存函数 (inverse survival function, ISF)。
    def _isf(self, x, a):
        return scu._skewnorm_isf(x, 0.0, 1.0, a)

    # 生成 skew-normal 分布的随机变量 (random variates, RVs)。
    def _rvs(self, a, size=None, random_state=None):
        # 生成标准正态分布的随机变量
        u0 = random_state.normal(size=size)
        v = random_state.normal(size=size)
        # 计算 skew-normal 分布的随机变量
        d = a / np.sqrt(1 + a**2)
        u1 = d * u0 + v * np.sqrt(1 - d**2)
        # 根据 u0 的值确定返回的随机变量
        return np.where(u0 >= 0, u1, -u1)

    # 计算 skew-normal 分布的统计特性。
    def _stats(self, a, moments='mvsk'):
        # 初始化输出结果
        output = [None, None, None, None]
        # 计算常数部分
        const = np.sqrt(2/np.pi) * a / np.sqrt(1 + a**2)

        # 根据 moments 参数计算所需的统计量
        if 'm' in moments:
            output[0] = const
        if 'v' in moments:
            output[1] = 1 - const**2
        if 's' in moments:
            output[2] = ((4 - np.pi)/2) * (const / np.sqrt(1 - const**2))**3
        if 'k' in moments:
            output[3] = (2*(np.pi - 3)) * (const**4 / (1 - const**2)**2)

        # 返回计算结果
        return output
    def _skewnorm_odd_moments(self):
        # 定义 skewnorm 分布的奇数阶矩，用多项式表示
        skewnorm_odd_moments = {
            1: Polynomial([1]),  # 一阶矩为常数多项式 1
            3: Polynomial([3, -1]),  # 三阶矩为多项式 3 - x
            5: Polynomial([15, -10, 3]),  # 五阶矩为多项式 15 - 10x + 3x^2
            7: Polynomial([105, -105, 63, -15]),  # 七阶矩为多项式 105 - 105x + 63x^2 - 15x^3
            9: Polynomial([945, -1260, 1134, -540, 105]),  # 九阶矩为多项式 945 - 1260x + 1134x^2 - 540x^3 + 105x^4
            11: Polynomial([10395, -17325, 20790, -14850, 5775, -945]),  # 十一阶矩为多项式 10395 - 17325x + 20790x^2 - 14850x^3 + 5775x^4 - 945x^5
            13: Polynomial([135135, -270270, 405405, -386100, 225225, -73710,
                            10395]),  # 十三阶矩为多项式 135135 - 270270x + 405405x^2 - 386100x^3 + 225225x^4 - 73710x^5 + 10395x^6
            15: Polynomial([2027025, -4729725, 8513505, -10135125, 7882875,
                            -3869775, 1091475, -135135]),  # 十五阶矩为多项式 2027025 - 4729725x + 8513505x^2 - 10135125x^3 + 7882875x^4 - 3869775x^5 + 1091475x^6 - 135135x^7
            17: Polynomial([34459425, -91891800, 192972780, -275675400,
                            268017750, -175429800, 74220300, -18378360,
                            2027025]),  # 十七阶矩为多项式 34459425 - 91891800x + 192972780x^2 - 275675400x^3 + 268017750x^4 - 175429800x^5 + 74220300x^6 - 18378360x^7 + 2027025x^8
            19: Polynomial([654729075, -1964187225, 4714049340, -7856748900,
                            9166207050, -7499623950, 4230557100, -1571349780,
                            346621275, -34459425]),  # 十九阶矩为多项式 654729075 - 1964187225x + 4714049340x^2 - 7856748900x^3 + 9166207050x^4 - 7499623950x^5 + 4230557100x^6 - 1571349780x^7 + 346621275x^8 - 34459425x^9
        }
        return skewnorm_odd_moments

    def _munp(self, order, a):
        if order & 1:
            if order > 19:
                raise NotImplementedError("skewnorm noncentral moments not "
                                          "implemented for odd orders greater "
                                          "than 19.")
            # 根据预计算的多项式计算 skewnorm 分布的非中心矩，基于矩生成函数推导
            delta = a / np.sqrt(1 + a**2)
            return (delta * self._skewnorm_odd_moments[order](delta**2)
                    * _SQRT_2_OVER_PI)
        else:
            # 对于偶数阶矩，矩为 (order-1)!!，其中 !! 是双阶乘的符号；对于奇数整数 m，m!! 表示 m*(m-2)*...*3*1
            # 可以使用 special.factorial2，但我们知道参数是奇数，因此避免使用该函数的开销，在此直接计算结果
            return sc.gamma((order + 1)/2) * 2**(order/2) / _SQRT_PI

    @extend_notes_in_docstring(rv_continuous, notes="""\
        If ``method='mm'``, parameters fixed by the user are respected, and the
        remaining parameters are used to match distribution and sample moments
        where possible. For example, if the user fixes the location with
        ``floc``, the parameters will only match the distribution skewness and
        variance to the sample skewness and variance; no attempt will be made
        to match the means or minimize a norm of the errors.
        Note that the maximum possible skewness magnitude of a
        `scipy.stats.skewnorm` distribution is approximately 0.9952717; if the
        magnitude of the data's sample skewness exceeds this, the returned
        shape parameter ``a`` will be infinite.
        \n\n""")
    def fit(self, data, *args, **kwds):
        # 如果 kwds 中有 "superfit"，并且其值为 True，则调用父类的 fit 方法
        if kwds.pop("superfit", False):
            return super().fit(data, *args, **kwds)
        
        # 如果 data 是 CensoredData 的实例
        if isinstance(data, CensoredData):
            # 如果数据中没有被屏蔽的数据点
            if data.num_censored() == 0:
                # 解除数据的屏蔽
                data = data._uncensor()
            else:
                # 否则，调用父类的 fit 方法
                return super().fit(data, *args, **kwds)

        # 检查并提取数据、固定形状、位置和尺度参数，将它们留在 kwds 中
        data, fa, floc, fscale = _check_fit_input_parameters(self, data,
                                                             args, kwds)
        
        # 从 kwds 中获取拟合方法，默认为 "mle"，如果指定为 "mm"，则不需要用户的猜测值
        method = kwds.get("method", "mle").lower()

        # 定义一个函数，计算偏度参数对应的 delta 值
        def skew_d(d):  # skewness in terms of delta
            return (4-np.pi)/2 * ((d * np.sqrt(2 / np.pi))**3
                                  / (1 - 2*d**2 / np.pi)**(3/2))
        
        # 定义一个函数，计算 delta 值对应的偏度参数
        def d_skew(skew):  # delta in terms of skewness
            s_23 = np.abs(skew)**(2/3)
            return np.sign(skew) * np.sqrt(
                np.pi/2 * s_23 / (s_23 + ((4 - np.pi)/2)**(2/3))
            )

        # 如果方法是方法 of moments，不需要用户的猜测值；否则，从 args 和 kwds 中提取猜测值
        if method == "mm":
            a, loc, scale = None, None, None
        else:
            a = args[0] if len(args) else None
            loc = kwds.pop('loc', None)
            scale = kwds.pop('scale', None)

        # 如果固定参数 fa 是 None 并且猜测值 a 也是 None，使用矩法（method of moments）
        if fa is None and a is None:
            # 计算样本偏度
            s = stats.skew(data)
            if method == 'mle':
                # 对于最大似然估计，将偏度裁剪到一个大但合理的值范围内
                s = np.clip(s, -0.99, 0.99)
            else:
                # 计算偏度对应的 delta 值
                s_max = skew_d(1)
                s = np.clip(s, -s_max, s_max)
            # 根据样本偏度计算 delta 值
            d = d_skew(s)
            with np.errstate(divide='ignore'):
                # 根据 delta 值计算形状参数 a
                a = np.sqrt(np.divide(d**2, (1-d**2)))*np.sign(s)
        else:
            # 否则，使用固定参数 fa 或者猜测值 a
            a = fa if fa is not None else a
            d = a / np.sqrt(1 + a**2)

        # 如果固定参数 fscale 是 None 并且猜测值 scale 也是 None
        if fscale is None and scale is None:
            # 计算数据的方差
            v = np.var(data)
            # 根据方差计算尺度参数 scale
            scale = np.sqrt(v / (1 - 2*d**2/np.pi))
        elif fscale is not None:
            # 否则，使用固定参数 fscale
            scale = fscale

        # 如果固定参数 floc 是 None 并且猜测值 loc 也是 None
        if floc is None and loc is None:
            # 计算数据的均值
            m = np.mean(data)
            # 根据均值和其他参数计算位置参数 loc
            loc = m - scale*d*np.sqrt(2/np.pi)
        elif floc is not None:
            # 否则，使用固定参数 floc
            loc = floc

        # 如果方法是 method of moments，则返回形状参数 a、位置参数 loc 和尺度参数 scale
        if method == 'mm':
            return a, loc, scale
        else:
            # 否则，将所有参数传递给父类的 fit 方法进行拟合
            return super().fit(data, a, loc=loc, scale=scale, **kwds)
skewnorm = skewnorm_gen(name='skewnorm')

class trapezoid_gen(rv_continuous):
    r"""A trapezoidal continuous random variable.

    %(before_notes)s

    Notes
    -----
    The trapezoidal distribution can be represented with an up-sloping line
    from ``loc`` to ``(loc + c*scale)``, then constant to ``(loc + d*scale)``
    and then downsloping from ``(loc + d*scale)`` to ``(loc+scale)``.  This
    defines the trapezoid base from ``loc`` to ``(loc+scale)`` and the flat
    top from ``c`` to ``d`` proportional to the position along the base
    with ``0 <= c <= d <= 1``.  When ``c=d``, this is equivalent to `triang`
    with the same values for `loc`, `scale` and `c`.
    The method of [1]_ is used for computing moments.

    `trapezoid` takes :math:`c` and :math:`d` as shape parameters.

    %(after_notes)s

    The standard form is in the range [0, 1] with c the mode.
    The location parameter shifts the start to `loc`.
    The scale parameter changes the width from 1 to `scale`.

    %(example)s

    References
    ----------
    .. [1] Kacker, R.N. and Lawrence, J.F. (2007). Trapezoidal and triangular
       distributions for Type B evaluation of standard uncertainty.
       Metrologia 44, 117-127. :doi:`10.1088/0026-1394/44/2/003`


    """
    # 定义_argcheck方法用于检查参数c和d的有效性
    def _argcheck(self, c, d):
        return (c >= 0) & (c <= 1) & (d >= 0) & (d <= 1) & (d >= c)

    # 定义_shape_info方法返回参数c和d的信息
    def _shape_info(self):
        ic = _ShapeInfo("c", False, (0, 1.0), (True, True))
        id = _ShapeInfo("d", False, (0, 1.0), (True, True))
        return [ic, id]

    # 定义_pdf方法计算概率密度函数
    def _pdf(self, x, c, d):
        # 计算斜坡的斜率
        u = 2 / (d-c+1)
        # 使用_lazyselect函数根据x的范围返回不同条件下的概率密度值
        return _lazyselect([x < c,
                            (c <= x) & (x <= d),
                            x > d],
                           [lambda x, c, d, u: u * x / c,
                            lambda x, c, d, u: u,
                            lambda x, c, d, u: u * (1-x) / (1-d)],
                           (x, c, d, u))

    # 定义_cdf方法计算累积分布函数
    def _cdf(self, x, c, d):
        # 使用_lazyselect函数根据x的范围返回不同条件下的累积分布值
        return _lazyselect([x < c,
                            (c <= x) & (x <= d),
                            x > d],
                           [lambda x, c, d: x**2 / c / (d-c+1),
                            lambda x, c, d: (c + 2 * (x-c)) / (d-c+1),
                            lambda x, c, d: 1-((1-x) ** 2
                                               / (d-c+1) / (1-d))],
                           (x, c, d))

    # 定义_ppf方法计算百分位点函数（反函数）
    def _ppf(self, q, c, d):
        # 计算累积分布函数在c和d处的值
        qc, qd = self._cdf(c, c, d), self._cdf(d, c, d)
        # 根据_qc和_qd的范围返回不同条件下的百分位点值
        condlist = [q < qc, q <= qd, q > qd]
        choicelist = [np.sqrt(q * c * (1 + d - c)),
                      0.5 * q * (1 + d - c) + 0.5 * c,
                      1 - np.sqrt((1 - q) * (d - c + 1) * (1 - d))]
        return np.select(condlist, choicelist)
    def _munp(self, n, c, d):
        """
        使用 Kacker (2007) 的参数化方法，其中
        a=左下角, c=左上角, d=右上角, b=右下角，则
        E[X^n] = h/(n+1)/(n+2) [(b^{n+2}-d^{n+2})/(b-d)
                                 - ((c^{n+2} - a^{n+2})/(c-a)]
        其中 h = 2/((b-a) - (d-c))。在 scipy 中对应的参数化方式为
        a'=loc, c'=loc+c*scale, d'=loc+d*scale, b'=loc+scale，
        对于标准形式，简化为 a'=0, b'=1, c'=c, d'=d。
        将其代入 E[X^n] 的公式中，bd' 项为 (1 - d^{n+2})/(1 - d)
        而 ac' 项为 c^{n-1}。在 d 接近 1 时 bd' 项可能出现数值问题，
        可以用 expm1((n+2)*log(d))/(d-1) 替代 (1 - d^{n+2})/(1-d)。
        通过 n=18, c=(1e-30,1-eps) 的测试表明这是稳定的。
        我们仍然需要显式测试 d=1 防止除以零，以及测试 d=0 防止 log(0)。

        参数：
        n: int，次数参数
        c: float，左上角参数
        d: float，右上角参数

        返回：
        float，计算得到的值
        """
        ab_term = c**(n+1)
        dc_term = _lazyselect(
            [d == 0.0, (0.0 < d) & (d < 1.0), d == 1.0],
            [lambda d: 1.0,
             lambda d: np.expm1((n+2) * np.log(d)) / (d-1.0),
             lambda d: n+2],
            [d])
        val = 2.0 / (1.0+d-c) * (dc_term - ab_term) / ((n+1) * (n+2))
        return val

    def _entropy(self, c, d):
        """
        使用 Wikipedia (van Dorp, 2003) 的参数化方法，其中
        a=左下角, c=左上角, d=右上角, b=右下角
        对于 loc=0, scale=1 的情况，得到 a'=0, b'=c, c'=d, d'=1。
        将其代入 Wikipedia 的熵公式得到以下结果。

        参数：
        c: float，左上角参数
        d: float，右上角参数

        返回：
        float，计算得到的熵值
        """
        return 0.5 * (1.0-d+c) / (1.0+d-c) + np.log(0.5 * (1.0+d-c))
# 定义一个字符串，用于提示 `trapz` 已被弃用，推荐使用 `trapezoid`
deprmsg = ("`trapz` is deprecated in favour of `trapezoid` "
           "and will be removed in SciPy 1.16.0.")

# 定义一个新的类 `trapz_gen`，继承自 `trapezoid_gen`
class trapz_gen(trapezoid_gen):
    # 重写了 `__call__` 方法，用于实现对实例化冻结分布的弃用警告
    """

    .. deprecated:: 1.14.0
        `trapz` is deprecated and will be removed in SciPy 1.16.
        Plese use `trapezoid` instead!
    """
    def __call__(self, *args, **kwds):
        # 发出弃用警告信息 `deprmsg`，使用 `DeprecationWarning` 级别为 2
        warnings.warn(deprmsg, DeprecationWarning, stacklevel=2)
        # 调用 `freeze` 方法并返回结果
        return self.freeze(*args, **kwds)

# 创建一个 `trapezoid_gen` 的实例 `trapezoid`
trapezoid = trapezoid_gen(a=0.0, b=1.0, name="trapezoid")
# 创建一个 `trapz_gen` 的实例 `trapz`
trapz = trapz_gen(a=0.0, b=1.0, name="trapz")

# 由于在导入时会实例化被弃用的类，因此在每个类方法中添加弃用警告
_method_names = [
    "cdf", "entropy", "expect", "fit", "interval", "isf", "logcdf", "logpdf",
    "logsf", "mean", "median", "moment", "pdf", "ppf", "rvs", "sf", "stats",
    "std", "var"
]

# 为 `_DeprecationWrapper` 类添加方法，用于发出每个方法的弃用警告
class _DeprecationWrapper:
    def __init__(self, method):
        self.msg = (f"`trapz.{method}` is deprecated in favour of trapezoid.{method}. "
                     "Please replace all uses of the distribution class "
                     "`trapz` with `trapezoid`. `trapz` will be removed in SciPy 1.16.")
        self.method = getattr(trapezoid, method)
    
    def __call__(self, *args, **kwargs):
        # 发出弃用警告 `self.msg`，使用 `DeprecationWarning` 级别为 2
        warnings.warn(self.msg, DeprecationWarning, stacklevel=2)
        # 调用 `trapezoid` 实例的对应方法，并返回其结果
        return self.method(*args, **kwargs)

# 遍历 `_method_names` 中的每个方法名，并将其设置为 `trapz` 类的属性，使用 `_DeprecationWrapper` 作为其值
for m in _method_names:
    setattr(trapz, m, _DeprecationWrapper(m))

# 定义 `triang_gen` 类，继承自 `rv_continuous`
class triang_gen(rv_continuous):
    r"""A triangular continuous random variable.

    %(before_notes)s

    Notes
    -----
    The triangular distribution can be represented with an up-sloping line from
    ``loc`` to ``(loc + c*scale)`` and then downsloping for ``(loc + c*scale)``
    to ``(loc + scale)``.

    `triang` takes ``c`` as a shape parameter for :math:`0 \le c \le 1`.

    %(after_notes)s

    The standard form is in the range [0, 1] with c the mode.
    The location parameter shifts the start to `loc`.
    The scale parameter changes the width from 1 to `scale`.

    %(example)s

    """
    # 定义 `_rvs` 方法，用于生成服从三角分布的随机变量
    def _rvs(self, c, size=None, random_state=None):
        return random_state.triangular(0, c, 1, size)

    # 定义 `_argcheck` 方法，用于检查参数 `c` 是否在 [0, 1] 范围内
    def _argcheck(self, c):
        return (c >= 0) & (c <= 1)

    # 定义 `_shape_info` 方法，返回形状参数信息 `_ShapeInfo` 对象列表
    def _shape_info(self):
        return [_ShapeInfo("c", False, (0, 1.0), (True, True))]
    # 计算特定条件下的概率密度函数 (PDF) 值
    def _pdf(self, x, c):
        # 0: 当 c=0 时的边界情况
        # 1: 对于 x < c 的一般情况，不使用 x <= c，因为它不能处理 c = 0 的情况。
        # 2: 对于 x >= c 的一般情况，但不能处理 c = 1 的情况。
        # 3: 当 c=1 时的边界情况
        r = _lazyselect([c == 0,
                         x < c,
                         (x >= c) & (c != 1),
                         c == 1],
                        [lambda x, c: 2 - 2 * x,
                         lambda x, c: 2 * x / c,
                         lambda x, c: 2 * (1 - x) / (1 - c),
                         lambda x, c: 2 * x],
                        (x, c))
        return r

    # 计算特定条件下的累积分布函数 (CDF) 值
    def _cdf(self, x, c):
        r = _lazyselect([c == 0,
                         x < c,
                         (x >= c) & (c != 1),
                         c == 1],
                        [lambda x, c: 2*x - x*x,
                         lambda x, c: x * x / c,
                         lambda x, c: (x*x - 2*x + c) / (c-1),
                         lambda x, c: x * x],
                        (x, c))
        return r

    # 计算特定条件下的百分位点函数 (PPF) 值
    def _ppf(self, q, c):
        return np.where(q < c, np.sqrt(c * q), 1-np.sqrt((1-c) * (1-q)))

    # 计算特定条件下的统计特性值
    def _stats(self, c):
        return ((c+1.0)/3.0,
                (1.0-c+c*c)/18,
                np.sqrt(2)*(2*c-1)*(c+1)*(c-2) / (5*np.power((1.0-c+c*c), 1.5)),
                -3.0/5.0)

    # 计算特定条件下的熵值
    def _entropy(self, c):
        return 0.5 - np.log(2)
# 创建一个三角分布的生成器对象，范围在 [0.0, 1.0]，名称为 "triang"
triang = triang_gen(a=0.0, b=1.0, name="triang")

# 定义一个截尾指数连续随机变量类 `truncexpon_gen`，继承自 `rv_continuous`
class truncexpon_gen(rv_continuous):
    r"""A truncated exponential continuous random variable.

    %(before_notes)s

    Notes
    -----
    `truncexpon` 的概率密度函数为：

    .. math::

        f(x, b) = \frac{\exp(-x)}{1 - \exp(-b)}

    其中 :math:`0 <= x <= b`。

    `truncexpon` 将 `b` 作为形状参数。

    %(after_notes)s

    %(example)s

    """

    # 返回形状信息 `_shape_info` 方法，表示 `b` 是一个范围在 (0, ∞) 之间的非必须参数
    def _shape_info(self):
        return [_ShapeInfo("b", False, (0, np.inf), (False, False))]

    # 返回支持区间 `_get_support` 方法，支持区间是从 `self.a` 到 `b`
    def _get_support(self, b):
        return self.a, b

    # 概率密度函数 `_pdf` 方法，返回 `truncexpon.pdf(x, b) = exp(-x) / (1-exp(-b))`
    def _pdf(self, x, b):
        return np.exp(-x) / (-sc.expm1(-b))

    # 对数概率密度函数 `_logpdf` 方法，返回对数密度函数 `-x - log(-sc.expm1(-b))`
    def _logpdf(self, x, b):
        return -x - np.log(-sc.expm1(-b))

    # 累积分布函数 `_cdf` 方法，返回 `sc.expm1(-x) / sc.expm1(-b)`
    def _cdf(self, x, b):
        return sc.expm1(-x) / sc.expm1(-b)

    # 百分点函数 `_ppf` 方法，返回 `-sc.log1p(q * sc.expm1(-b))`
    def _ppf(self, q, b):
        return -sc.log1p(q * sc.expm1(-b))

    # 生存函数 `_sf` 方法，返回 `(exp(-b) - exp(-x)) / sc.expm1(-b)`
    def _sf(self, x, b):
        return (np.exp(-b) - np.exp(-x)) / sc.expm1(-b)

    # 逆生存函数 `_isf` 方法，返回 `-log(exp(-b) - q * sc.expm1(-b))`
    def _isf(self, q, b):
        return -np.log(np.exp(-b) - q * sc.expm1(-b))

    # 矩 `_munp` 方法，计算特定阶数 `n` 的矩，对于 `n==1` 和 `n==2` 有特定计算公式，否则调用超类的方法
    def _munp(self, n, b):
        if n == 1:
            return (1 - (b + 1) * np.exp(-b)) / (-sc.expm1(-b))
        elif n == 2:
            return 2 * (1 - 0.5 * (b * b + 2 * b + 2) * np.exp(-b)) / (-sc.expm1(-b))
        else:
            return super()._munp(n, b)

    # 熵 `_entropy` 方法，返回指数分布的熵计算公式
    def _entropy(self, b):
        eB = np.exp(b)
        return np.log(eB - 1) + (1 + eB * (b - 1.0)) / (1.0 - eB)


# 创建一个 `truncexpon_gen` 的实例 `truncexpon`，范围在 [0.0, ∞)，名称为 'truncexpon'
truncexpon = truncexpon_gen(a=0.0, name='truncexpon')

# 对数和指数和的技巧，用于计算 `log(p + q)`，其中 `log_p` 和 `log_q` 是对数值
def _log_sum(log_p, log_q):
    return sc.logsumexp([log_p, log_q], axis=0)

# 与上述相同，但使用 `-exp(x) = exp(x + πi)` 的技巧计算 `log(p - q)`
def _log_diff(log_p, log_q):
    return sc.logsumexp([log_p, log_q + np.pi * 1j], axis=0)

# 计算高斯分布在区间内的对数概率质量的函数 `_log_gauss_mass`
def _log_gauss_mass(a, b):
    """Log of Gaussian probability mass within an interval"""
    a, b = np.broadcast_arrays(a, b)

    # 对称性计算，只在左尾中工作，因为右尾的计算不准确
    case_left = b <= 0
    case_right = a > 0
    case_central = ~(case_left | case_right)

    def mass_case_left(a, b):
        return _log_diff(_norm_logcdf(b), _norm_logcdf(a))

    def mass_case_right(a, b):
        return mass_case_left(-b, -a)
    # 定义一个函数来计算中心情况下的质量
    def mass_case_central(a, b):
        # 原先的实现:
        # left_mass = mass_case_left(a, 0)
        # right_mass = mass_case_right(0, b)
        # return _log_sum(left_mass, right_mass)
        # 由于 np.exp(log_mass) 接近 1，会发生灾难性的取消效应。
        # 用另一种形式来修正这个问题。
        # 我们不担心下溢的问题：如果只有一个项下溢，那它是微不足道的；
        # 如果两个项都下溢，结果在对数空间里也无法准确表示，因为对于小的 x，sc.log1p(x) ~ x。
        return sc.log1p(-_norm_cdf(a) - _norm_cdf(-b))

    # _lazyselect 没有工作；不想去调试它
    # 创建一个和 a 相同形状的数组，填充为 NaN，数据类型为复数
    out = np.full_like(a, fill_value=np.nan, dtype=np.complex128)
    # 如果 case_left 中有数据
    if a[case_left].size:
        # 对应位置的 out 数组赋值为 mass_case_left(a[case_left], b[case_left]) 的结果
        out[case_left] = mass_case_left(a[case_left], b[case_left])
    # 如果 case_right 中有数据
    if a[case_right].size:
        # 对应位置的 out 数组赋值为 mass_case_right(a[case_right], b[case_right]) 的结果
        out[case_right] = mass_case_right(a[case_right], b[case_right])
    # 如果 case_central 中有数据
    if a[case_central].size:
        # 对应位置的 out 数组赋值为 mass_case_central(a[case_central], b[case_central]) 的结果
        out[case_central] = mass_case_central(a[case_central], b[case_central])
    # 返回 out 数组的实部，丢弃虚部（即 ~0j）
    return np.real(out)
class truncnorm_gen(rv_continuous):
    r"""A truncated normal continuous random variable.

    %(before_notes)s

    Notes
    -----
    This distribution is the normal distribution centered on ``loc`` (default
    0), with standard deviation ``scale`` (default 1), and truncated at ``a``
    and ``b`` *standard deviations* from ``loc``. For arbitrary ``loc`` and
    ``scale``, ``a`` and ``b`` are *not* the abscissae at which the shifted
    and scaled distribution is truncated.

    .. note::
        If ``a_trunc`` and ``b_trunc`` are the abscissae at which we wish
        to truncate the distribution (as opposed to the number of standard
        deviations from ``loc``), then we can calculate the distribution
        parameters ``a`` and ``b`` as follows::

            a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale

        This is a common point of confusion. For additional clarification,
        please see the example below.

    %(example)s

    In the examples above, ``loc=0`` and ``scale=1``, so the plot is truncated
    at ``a`` on the left and ``b`` on the right. However, suppose we were to
    produce the same histogram with ``loc = 1`` and ``scale=0.5``.

    >>> loc, scale = 1, 0.5
    >>> rv = truncnorm(a, b, loc=loc, scale=scale)
    >>> x = np.linspace(truncnorm.ppf(0.01, a, b),
    ...                 truncnorm.ppf(0.99, a, b), 100)
    >>> r = rv.rvs(size=1000)

    >>> fig, ax = plt.subplots(1, 1)
    >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
    >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
    >>> ax.set_xlim(a, b)
    >>> ax.legend(loc='best', frameon=False)
    >>> plt.show()

    Note that the distribution is no longer appears to be truncated at
    abscissae ``a`` and ``b``. That is because the *standard* normal
    distribution is first truncated at ``a`` and ``b``, *then* the resulting
    distribution is scaled by ``scale`` and shifted by ``loc``. If we instead
    want the shifted and scaled distribution to be truncated at ``a`` and
    ``b``, we need to transform these values before passing them as the
    distribution parameters.

    >>> a_transformed, b_transformed = (a - loc) / scale, (b - loc) / scale
    >>> rv = truncnorm(a_transformed, b_transformed, loc=loc, scale=scale)
    >>> x = np.linspace(truncnorm.ppf(0.01, a, b),
    ...                 truncnorm.ppf(0.99, a, b), 100)
    >>> r = rv.rvs(size=10000)

    >>> fig, ax = plt.subplots(1, 1)
    >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
    >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
    >>> ax.set_xlim(a-0.1, b+0.1)
    >>> ax.legend(loc='best', frameon=False)
    >>> plt.show()
    """

    def _argcheck(self, a, b):
        # 检查参数 a 和 b 是否满足截断条件，即 a 必须小于 b
        return a < b

    def _shape_info(self):
        # 返回分布的形状信息，包括参数名、是否必须、有效范围和默认设置
        ia = _ShapeInfo("a", False, (-np.inf, np.inf), (True, False))
        ib = _ShapeInfo("b", False, (-np.inf, np.inf), (False, True))
        return [ia, ib]
    # 对数据进行初步适配，确保支持区间在[a, b]内
    def _fitstart(self, data):
        # 如果数据类型为CensoredData，则先解除其被屏蔽的部分
        if isinstance(data, CensoredData):
            data = data._uncensor()
        # 调用父类的_fitstart方法，传入数据以及最小值和最大值作为参数
        return super()._fitstart(data, args=(np.min(data), np.max(data)))

    # 返回支持区间的起始点a和终止点b
    def _get_support(self, a, b):
        return a, b

    # 计算概率密度函数的值，使用指数函数来处理对数概率密度函数的结果
    def _pdf(self, x, a, b):
        return np.exp(self._logpdf(x, a, b))

    # 计算对数概率密度函数的值，减去高斯质量函数的对数值
    def _logpdf(self, x, a, b):
        return _norm_logpdf(x) - _log_gauss_mass(a, b)

    # 计算累积分布函数的值，使用指数函数来处理对数累积分布函数的结果
    def _cdf(self, x, a, b):
        return np.exp(self._logcdf(x, a, b))

    # 计算对数累积分布函数的值，避免灾难性的取消
    def _logcdf(self, x, a, b):
        # 广播x, a, b以确保相同维度计算
        x, a, b = np.broadcast_arrays(x, a, b)
        # 计算对数累积分布函数
        logcdf = np.asarray(_log_gauss_mass(a, x) - _log_gauss_mass(a, b))
        # 找到需要避免灾难性取消的地方
        i = logcdf > -0.1  # 避免灾难性的取消
        if np.any(i):
            logcdf[i] = np.log1p(-np.exp(self._logsf(x[i], a[i], b[i])))
        return logcdf

    # 计算生存函数（1 - CDF）的值，使用指数函数来处理对数生存函数的结果
    def _sf(self, x, a, b):
        return np.exp(self._logsf(x, a, b))

    # 计算对数生存函数的值，避免灾难性的取消
    def _logsf(self, x, a, b):
        # 广播x, a, b以确保相同维度计算
        x, a, b = np.broadcast_arrays(x, a, b)
        # 计算对数生存函数
        logsf = np.asarray(_log_gauss_mass(x, b) - _log_gauss_mass(a, b))
        # 找到需要避免灾难性取消的地方
        i = logsf > -0.1  # 避免灾难性的取消
        if np.any(i):
            logsf[i] = np.log1p(-np.exp(self._logcdf(x[i], a[i], b[i])))
        return logsf

    # 计算熵值，衡量概率分布的不确定性
    def _entropy(self, a, b):
        # 计算累积分布函数在a和b处的值
        A = _norm_cdf(a)
        B = _norm_cdf(b)
        # 计算分布的归一化常数Z
        Z = B - A
        # 计算熵的两部分：常数C和动态的D
        C = np.log(np.sqrt(2 * np.pi * np.e) * Z)
        D = (a * _norm_pdf(a) - b * _norm_pdf(b)) / (2 * Z)
        # 综合得出熵值h
        h = C + D
        return h

    # 计算百分位点函数（逆累积分布函数），处理不同情况的q值
    def _ppf(self, q, a, b):
        # 广播q, a, b以确保相同维度计算
        q, a, b = np.broadcast_arrays(q, a, b)

        # 定义左侧和右侧不同情况下的ppf函数
        def ppf_left(q, a, b):
            # 计算左侧情况下的对数累积分布函数的对数和
            log_Phi_x = _log_sum(_norm_logcdf(a),
                                 np.log(q) + _log_gauss_mass(a, b))
            # 返回逆标准正态分布函数的值
            return sc.ndtri_exp(log_Phi_x)

        def ppf_right(q, a, b):
            # 计算右侧情况下的对数累积分布函数的对数和
            log_Phi_x = _log_sum(_norm_logcdf(-b),
                                 np.log1p(-q) + _log_gauss_mass(a, b))
            # 返回逆标准正态分布函数的值
            return -sc.ndtri_exp(log_Phi_x)

        # 创建与q相同形状的输出数组
        out = np.empty_like(q)

        # 分别处理左侧和右侧情况下的q值
        q_left = q[case_left]
        q_right = q[case_right]

        # 如果存在左侧情况下的q值，计算相应的ppf值
        if q_left.size:
            out[case_left] = ppf_left(q_left, a[case_left], b[case_left])
        # 如果存在右侧情况下的q值，计算相应的ppf值
        if q_right.size:
            out[case_right] = ppf_right(q_right, a[case_right], b[case_right])

        # 返回计算得到的ppf值数组
        return out
    def _isf(self, q, a, b):
        # 大部分是 _ppf 的复制粘贴，但我觉得这比合并更简单
        # 使用 numpy 的 broadcast_arrays 函数，将 q, a, b 进行广播
        q, a, b = np.broadcast_arrays(q, a, b)

        # 判断 b 是否小于 0，生成布尔数组 case_left 和 case_right
        case_left = b < 0
        case_right = ~case_left

        # 定义处理左侧情况的函数 isf_left
        def isf_left(q, a, b):
            # 计算 log(1 - Phi(b)) 和 log(q) + log(gauss_mass(a, b)) 的差
            log_Phi_x = _log_diff(_norm_logcdf(b),
                                  np.log(q) + _log_gauss_mass(a, b))
            # 返回 inverse survival function 的值
            return sc.ndtri_exp(np.real(log_Phi_x))

        # 定义处理右侧情况的函数 isf_right
        def isf_right(q, a, b):
            # 计算 log(Phi(-a)) 和 log(1 - q) + log(gauss_mass(a, b)) 的差
            log_Phi_x = _log_diff(_norm_logcdf(-a),
                                  np.log1p(-q) + _log_gauss_mass(a, b))
            # 返回 inverse survival function 的负值
            return -sc.ndtri_exp(np.real(log_Phi_x))

        # 创建一个与 q 同类型的空数组 out
        out = np.empty_like(q)

        # 分别处理左侧和右侧情况
        q_left = q[case_left]
        q_right = q[case_right]

        # 如果 q_left 或 q_right 非空，分别调用 isf_left 和 isf_right 函数
        if q_left.size:
            out[case_left] = isf_left(q_left, a[case_left], b[case_left])
        if q_right.size:
            out[case_right] = isf_right(q_right, a[case_right], b[case_right])

        # 返回处理结果数组 out
        return out

    def _munp(self, n, a, b):
        # 定义 n_th_moment 函数，计算 n 阶矩，只有当 n >= 0 时才定义
        def n_th_moment(n, a, b):
            """
            返回 n 阶矩。只有当 n >= 0 时定义。
            由于需要对 n 进行循环，该函数无法进行广播。
            """
            # 调用 _pdf 函数，计算给定 a 和 b 的概率密度函数值
            pA, pB = self._pdf(np.asarray([a, b]), a, b)
            # 将概率密度函数值放入列表 probs 中
            probs = [pA, -pB]
            # 初始时刻 0 和 1 的阶矩
            moments = [0, 1]
            # 循环计算直到第 n 阶
            for k in range(1, n+1):
                # 如果 a 或 b 为无穷大，对应的概率密度函数值为 0，但是乘法结果是 nan。
                # 然而，当 b 趋向无穷大时，pdf(b)*b**k 趋向于 0。
                # 因此，可以安全地使用 _lazywhere 避免 nan。
                vals = _lazywhere(probs, [probs, [a, b]],
                                  lambda x, y: x * y**(k-1), fillvalue=0)
                # 计算当前阶矩 mk
                mk = np.sum(vals) + (k-1) * moments[-2]
                # 将计算结果添加到 moments 列表中
                moments.append(mk)
            # 返回最终的阶矩 moments[-1]
            return moments[-1]

        # 使用 _lazywhere 函数检查 n >= 0 且 a == a 且 b == b 的条件，
        # 调用 np.vectorize 对 n_th_moment 函数进行向量化处理，
        # 输出类型为 np.float64，如果条件不满足则返回 np.nan
        return _lazywhere((n >= 0) & (a == a) & (b == b), (n, a, b),
                          np.vectorize(n_th_moment, otypes=[np.float64]),
                          np.nan)
    # 定义内部方法 _stats，计算截断正态分布的统计量
    def _stats(self, a, b, moments='mv'):
        # 计算给定区间 [a, b] 的概率密度函数值 pA 和 pB
        pA, pB = self.pdf(np.array([a, b]), a, b)

        # 定义内部方法 _truncnorm_stats_scalar，计算截断正态分布的各阶矩
        def _truncnorm_stats_scalar(a, b, pA, pB, moments):
            # 计算第一阶矩 m1
            m1 = pA - pB
            mu = m1
            # 使用 _lazywhere 避免 NaN（详见 _munp 中的详细注释）
            probs = [pA, -pB]
            # 计算第二阶矩 m2
            vals = _lazywhere(probs, [probs, [a, b]], lambda x, y: x*y,
                              fillvalue=0)
            m2 = 1 + np.sum(vals)
            # 继续计算第二阶矩的替代计算 mu2，比直接计算更稳定
            vals = _lazywhere(probs, [probs, [a-mu, b-mu]], lambda x, y: x*y,
                              fillvalue=0)
            mu2 = 1 + np.sum(vals)
            # 计算第三阶矩 m3
            vals = _lazywhere(probs, [probs, [a, b]], lambda x, y: x*y**2,
                              fillvalue=0)
            m3 = 2*m1 + np.sum(vals)
            # 计算第四阶矩 m4
            vals = _lazywhere(probs, [probs, [a, b]], lambda x, y: x*y**3,
                              fillvalue=0)
            m4 = 3*m2 + np.sum(vals)

            # 计算 mu3 和 skewness (g1)
            mu3 = m3 + m1 * (-3*m2 + 2*m1**2)
            g1 = mu3 / np.power(mu2, 1.5)
            # 计算 mu4 和 kurtosis (g2)
            mu4 = m4 + m1*(-4*m3 + 3*m1*(2*m2 - m1**2))
            g2 = mu4 / mu2**2 - 3
            return mu, mu2, g1, g2

        # 向量化 _truncnorm_stats_scalar 方法，排除 moments 参数
        _truncnorm_stats = np.vectorize(_truncnorm_stats_scalar,
                                        excluded=('moments',))
        # 返回截断正态分布的统计量
        return _truncnorm_stats(a, b, pA, pB, moments)
# 定义一个被截尾正态分布生成器，名称为'truncnorm'，momtype为1
truncnorm = truncnorm_gen(name='truncnorm', momtype=1)

# 定义一个继承自连续随机变量类(rv_continuous)的类，代表上截尾的Pareto分布随机变量
class truncpareto_gen(rv_continuous):
    r"""An upper truncated Pareto continuous random variable.

    %(before_notes)s

    See Also
    --------
    pareto : Pareto distribution

    Notes
    -----
    The probability density function for `truncpareto` is:

    .. math::

        f(x, b, c) = \\frac{b}{1 - c^{-b}} \\frac{1}{x^{b+1}}

    for :math:`b > 0`, :math:`c > 1` and :math:`1 \le x \le c`.

    `truncpareto` takes `b` and `c` as shape parameters for :math:`b` and
    :math:`c`.

    Notice that the upper truncation value :math:`c` is defined in
    standardized form so that random values of an unscaled, unshifted variable
    are within the range ``[1, c]``.
    If ``u_r`` is the upper bound to a scaled and/or shifted variable,
    then ``c = (u_r - loc) / scale``. In other words, the support of the
    distribution becomes ``(scale + loc) <= x <= (c*scale + loc)`` when
    `scale` and/or `loc` are provided.

    %(after_notes)s

    References
    ----------
    .. [1] Burroughs, S. M., and Tebbens S. F.
        "Upper-truncated power laws in natural systems."
        Pure and Applied Geophysics 158.4 (2001): 741-757.

    %(example)s

    """

    # 返回分布参数'b'和'c'的描述信息
    def _shape_info(self):
        ib = _ShapeInfo("b", False, (0.0, np.inf), (False, False))
        ic = _ShapeInfo("c", False, (1.0, np.inf), (False, False))
        return [ib, ic]

    # 检查参数'b'和'c'是否满足条件(b > 0.) & (c > 1.)
    def _argcheck(self, b, c):
        return (b > 0.) & (c > 1.)

    # 获取分布的支持区间，返回下界self.a和上界c
    def _get_support(self, b, c):
        return self.a, c

    # 概率密度函数PDF，计算上截尾Pareto分布的概率密度值
    def _pdf(self, x, b, c):
        return b * x**-(b+1) / (1 - 1/c**b)

    # 对数概率密度函数logPDF，计算上截尾Pareto分布的对数概率密度值
    def _logpdf(self, x, b, c):
        return np.log(b) - np.log(-np.expm1(-b*np.log(c))) - (b+1)*np.log(x)

    # 累积分布函数CDF，计算上截尾Pareto分布的累积分布函数值
    def _cdf(self, x, b, c):
        return (1 - x**-b) / (1 - 1/c**b)

    # 对数累积分布函数logCDF，计算上截尾Pareto分布的对数累积分布函数值
    def _logcdf(self, x, b, c):
        return np.log1p(-x**-b) - np.log1p(-1/c**b)

    # 百分位点函数PPF，计算上截尾Pareto分布的百分位点函数值
    def _ppf(self, q, b, c):
        return pow(1 - (1 - 1/c**b)*q, -1/b)

    # 生存函数SF，计算上截尾Pareto分布的生存函数值
    def _sf(self, x, b, c):
        return (x**-b - 1/c**b) / (1 - 1/c**b)

    # 对数生存函数logSF，计算上截尾Pareto分布的对数生存函数值
    def _logsf(self, x, b, c):
        return np.log(x**-b - 1/c**b) - np.log1p(-1/c**b)

    # 逆百分位点函数ISF，计算上截尾Pareto分布的逆百分位点函数值
    def _isf(self, q, b, c):
        return pow(1/c**b + (1 - 1/c**b)*q, -1/b)

    # 熵函数Entropy，计算上截尾Pareto分布的熵值
    def _entropy(self, b, c):
        return -(np.log(b/(1 - 1/c**b))
                 + (b+1)*(np.log(c)/(c**b - 1) - 1/b))

    # 非中心矩函数munp，计算上截尾Pareto分布的非中心矩
    def _munp(self, n, b, c):
        if (n == b).all():
            return b*np.log(c) / (1 - 1/c**b)
        else:
            return b / (b-n) * (c**b - c**n) / (c**b - 1)

    # 估计起始值函数fitstart，根据数据估计参数'b'和'c'的起始值
    def _fitstart(self, data):
        if isinstance(data, CensoredData):
            data = data._uncensor()
        b, loc, scale = pareto.fit(data)
        c = (max(data) - loc)/scale
        return b, c, loc, scale

    # 调用超类mom方法
    @_call_super_mom
    # 继承rv_continuous类中的文档字符串
    @inherit_docstring_from(rv_continuous)
# 定义一个被截尾Pareto分布生成器，名称为'truncpareto'，下界为1.0
truncpareto = truncpareto_gen(a=1.0, name='truncpareto')

# 定义一个继承自连续随机变量类(rv_continuous)的类，代表Tukey-Lamdba分布随机变量
class tukeylambda_gen(rv_continuous):
    r"""A Tukey-Lamdba continuous random variable.

    %(before_notes)s

    Notes
    -----
    """
    A flexible distribution, able to represent and interpolate between the
    following distributions:

    - Cauchy                (:math:`lambda = -1`)
    - logistic              (:math:`lambda = 0`)
    - approx Normal         (:math:`lambda = 0.14`)
    - uniform from -1 to 1  (:math:`lambda = 1`)

    `tukeylambda` takes a real number :math:`lambda` (denoted ``lam``
    in the implementation) as a shape parameter.

    %(after_notes)s

    %(example)s

    """
    # 定义一个名为 `_argcheck` 的方法，用于检查形状参数 `lam` 是否是有限数
    def _argcheck(self, lam):
        return np.isfinite(lam)

    # 定义一个名为 `_shape_info` 的方法，返回一个 `_ShapeInfo` 对象列表，
    # 表示形状参数 `lam`，不是必需的，范围在负无穷到正无穷之间，不包括边界值
    # 这里的 `(False, False)` 表示边界不包括
    def _shape_info(self):
        return [_ShapeInfo("lam", False, (-np.inf, np.inf), (False, False))]

    # 定义一个名为 `_pdf` 的方法，计算概率密度函数
    def _pdf(self, x, lam):
        # 计算 `sc.tklmbda(x, lam)` 的结果，并转换为 NumPy 数组
        Fx = np.asarray(sc.tklmbda(x, lam))
        # 计算概率密度函数 Px
        Px = Fx**(lam-1.0) + (np.asarray(1-Fx))**(lam-1.0)
        Px = 1.0/np.asarray(Px)
        # 根据条件返回概率密度函数 Px 或者 0.0
        return np.where((lam <= 0) | (abs(x) < 1.0/np.asarray(lam)), Px, 0.0)

    # 定义一个名为 `_cdf` 的方法，计算累积分布函数
    def _cdf(self, x, lam):
        return sc.tklmbda(x, lam)

    # 定义一个名为 `_ppf` 的方法，计算百分位点函数
    def _ppf(self, q, lam):
        # 使用 `sc.boxcox` 和 `sc.boxcox1p` 计算百分位点函数
        return sc.boxcox(q, lam) - sc.boxcox1p(-q, lam)

    # 定义一个名为 `_stats` 的方法，计算分布的统计特性
    def _stats(self, lam):
        # 返回统计特性，具体为 (0, _tlvar(lam), 0, _tlkurt(lam))
        return 0, _tlvar(lam), 0, _tlkurt(lam)

    # 定义一个名为 `_entropy` 的方法，计算分布的熵
    def _entropy(self, lam):
        # 定义一个内部函数 `integ`，计算积分的被积函数
        def integ(p):
            return np.log(pow(p, lam-1)+pow(1-p, lam-1))
        # 使用 `integrate.quad` 对 `integ` 函数在 [0, 1] 区间进行数值积分
        return integrate.quad(integ, 0, 1)[0]
tukeylambda = tukeylambda_gen(name='tukeylambda')

# 定义一个名为 `tukeylambda` 的变量，并调用 `tukeylambda_gen` 函数来生成一个特定的随机变量


class FitUniformFixedScaleDataError(FitDataError):
    def __init__(self, ptp, fscale):
        self.args = (
            "Invalid values in `data`.  Maximum likelihood estimation with "
            "the uniform distribution and fixed scale requires that "
            f"np.ptp(data) <= fscale, but np.ptp(data) = {ptp} and "
            f"fscale = {fscale}."
        )

# 定义一个名为 `FitUniformFixedScaleDataError` 的异常类，继承自 `FitDataError`，用于处理当数据不符合
# 均匀分布和固定尺度的最大似然估计要求时引发的错误


class uniform_gen(rv_continuous):
    r"""A uniform continuous random variable.

    In the standard form, the distribution is uniform on ``[0, 1]``. Using
    the parameters ``loc`` and ``scale``, one obtains the uniform distribution
    on ``[loc, loc + scale]``.

    %(before_notes)s

    %(example)s

    """
    def _shape_info(self):
        return []
    
    # 返回空列表，表示此均匀分布没有特定的形状参数

    def _rvs(self, size=None, random_state=None):
        return random_state.uniform(0.0, 1.0, size)
    
    # 生成服从均匀分布的随机变量数组，取值范围在 [0.0, 1.0)，大小为 `size`

    def _pdf(self, x):
        return 1.0*(x == x)
    
    # 返回指定点 `x` 的概率密度函数值，对于均匀分布来说，返回值为1.0（x存在）或0.0（x不存在）

    def _cdf(self, x):
        return x
    
    # 返回指定点 `x` 的累积分布函数值，对于均匀分布来说，即返回 `x` 本身

    def _ppf(self, q):
        return q
    
    # 返回指定累积分布概率 `q` 对应的百分位点

    def _stats(self):
        return 0.5, 1.0/12, 0, -1.2
    
    # 返回均匀分布的均值、方差、偏度和峰度

    def _entropy(self):
        return 0.0
    
    # 返回分布的熵值，对于均匀分布，熵值为0.0

    @_call_super_mom
uniform = uniform_gen(a=0.0, b=1.0, name='uniform')

# 创建一个名为 `uniform` 的均匀分布对象，`a` 和 `b` 是分布的上下界，`name` 是分布的名称


class vonmises_gen(rv_continuous):
    r"""A Von Mises continuous random variable.

    %(before_notes)s

    See Also
    --------
    scipy.stats.vonmises_fisher : Von-Mises Fisher distribution on a
                                  hypersphere

    Notes
    -----
    The probability density function for `vonmises` and `vonmises_line` is:

    .. math::

        f(x, \kappa) = \frac{ \exp(\kappa \cos(x)) }{ 2 \pi I_0(\kappa) }

    for :math:`-\pi \le x \le \pi`, :math:`\kappa \ge 0`. :math:`I_0` is the
    modified Bessel function of order zero (`scipy.special.i0`).

    `vonmises` is a circular distribution which does not restrict the
    distribution to a fixed interval. Currently, there is no circular
    distribution framework in SciPy. The ``cdf`` is implemented such that
    ``cdf(x + 2*np.pi) == cdf(x) + 1``.

    `vonmises_line` is the same distribution, defined on :math:`[-\pi, \pi]`
    on the real line. This is a regular (i.e. non-circular) distribution.

    Note about distribution parameters: `vonmises` and `vonmises_line` take
    ``kappa`` as a shape parameter (concentration) and ``loc`` as the location
    (circular mean). A ``scale`` parameter is accepted but does not have any
    effect.

    Examples
    --------
    Import the necessary modules.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import vonmises

    Define distribution parameters.

    >>> loc = 0.5 * np.pi  # circular mean
    >>> kappa = 1  # concentration

    Compute the probability density at ``x=0`` via the ``pdf`` method.

    >>> vonmises.pdf(0, loc=loc, kappa=kappa)
    0.12570826359722018

    Verify that the percentile function ``ppf`` inverts the cumulative
    distribution function ``cdf`` up to floating point accuracy.

    """

# 定义一个名为 `vonmises_gen` 的类，表示 Von Mises 连续随机变量的分布特性，包括概率密度函数、累积分布函数等说明
    >>> x = 1
    # 计算 von Mises 分布在 x 处的累积分布函数值，使用给定的 loc 和 kappa 参数
    >>> cdf_value = vonmises.cdf(x, loc=loc, kappa=kappa)
    # 根据累积分布函数值计算 von Mises 分布的百分位点函数值，使用给定的 loc 和 kappa 参数
    >>> ppf_value = vonmises.ppf(cdf_value, loc=loc, kappa=kappa)
    # 打印 x 值、累积分布函数值和百分位点函数值
    >>> x, cdf_value, ppf_value
    (1, 0.31489339900904967, 1.0000000000000004)

    Draw 1000 random variates by calling the ``rvs`` method.

    >>> sample_size = 1000
    # 使用 von Mises 分布生成指定大小的随机样本
    >>> sample = vonmises(loc=loc, kappa=kappa).rvs(sample_size)

    Plot the von Mises density on a Cartesian and polar grid to emphasize
    that it is a circular distribution.

    >>> fig = plt.figure(figsize=(12, 6))
    # 创建一个大小为 12x6 的图形对象
    >>> left = plt.subplot(121)
    # 在图形中创建一个子图，位置为 1x2 网格中的第一个
    >>> right = plt.subplot(122, projection='polar')
    # 在图形中创建一个极坐标子图，位置为 1x2 网格中的第二个
    >>> x = np.linspace(-np.pi, np.pi, 500)
    # 生成一个从 -π 到 π 的等间距数组，用于绘制 von Mises 密度函数
    >>> vonmises_pdf = vonmises.pdf(x, loc=loc, kappa=kappa)
    # 计算 von Mises 分布在指定 x 值处的概率密度函数值，使用给定的 loc 和 kappa 参数
    >>> ticks = [0, 0.15, 0.3]

    The left image contains the Cartesian plot.

    >>> left.plot(x, vonmises_pdf)
    # 在左子图中绘制 von Mises 分布的概率密度函数图像
    >>> left.set_yticks(ticks)
    # 设置左子图的 y 轴刻度
    >>> number_of_bins = int(np.sqrt(sample_size))
    # 计算直方图的 bin 数量，使用样本大小的平方根
    >>> left.hist(sample, density=True, bins=number_of_bins)
    # 在左子图中绘制 von Mises 分布的样本直方图
    >>> left.set_title("Cartesian plot")
    # 设置左子图的标题为 "Cartesian plot"
    >>> left.set_xlim(-np.pi, np.pi)
    # 设置左子图的 x 轴范围为 -π 到 π
    >>> left.grid(True)
    # 在左子图中显示网格线

    The right image contains the polar plot.

    >>> right.plot(x, vonmises_pdf, label="PDF")
    # 在极坐标子图中绘制 von Mises 分布的概率密度函数曲线，并添加标签 "PDF"
    >>> right.set_yticks(ticks)
    # 设置极坐标子图的 y 轴刻度
    >>> right.hist(sample, density=True, bins=number_of_bins,
    ...            label="Histogram")
    # 在极坐标子图中绘制 von Mises 分布的样本直方图，并添加标签 "Histogram"
    >>> right.set_title("Polar plot")
    # 设置极坐标子图的标题为 "Polar plot"
    >>> right.legend(bbox_to_anchor=(0.15, 1.06))
    # 在右上角添加图例

    """
    # 返回形状信息列表，描述 von Mises 分布的参数 "kappa" 的性质
    def _shape_info(self):
        return [_ShapeInfo("kappa", False, (0, np.inf), (True, False))]

    # 检查参数 kappa 是否为非负数，用于 von Mises 分布
    def _argcheck(self, kappa):
        return kappa >= 0

    # 使用指定的 kappa 参数和随机种子生成 von Mises 分布的随机样本
    def _rvs(self, kappa, size=None, random_state=None):
        return random_state.vonmises(0.0, kappa, size=size)

    @inherit_docstring_from(rv_continuous)
    # 从基类 rv_continuous 继承文档字符串
    def rvs(self, *args, **kwds):
        rvs = super().rvs(*args, **kwds)
        # 对生成的随机样本进行调整，使其处于 -π 到 π 的范围内
        return np.mod(rvs + np.pi, 2*np.pi) - np.pi

    # 计算 von Mises 分布在给定 x 和 kappa 参数下的概率密度函数值
    def _pdf(self, x, kappa):
        # vonmises.pdf(x, kappa) = exp(kappa * cosm1(x)) / (2*pi*i0e(kappa))
        return np.exp(kappa*sc.cosm1(x)) / (2*np.pi*sc.i0e(kappa))

    # 计算 von Mises 分布在给定 x 和 kappa 参数下的对数概率密度函数值
    def _logpdf(self, x, kappa):
        # vonmises.pdf(x, kappa) = exp(kappa * cosm1(x)) / (2*pi*i0e(kappa))
        return kappa * sc.cosm1(x) - np.log(2*np.pi) - np.log(sc.i0e(kappa))

    # 计算 von Mises 分布在给定 x 和 kappa 参数下的累积分布函数值
    def _cdf(self, x, kappa):
        return _stats.von_mises_cdf(kappa, x)

    # 跳过计算分布的一些统计量，返回固定值
    def _stats_skip(self, kappa):
        return 0, None, 0, None
    def _entropy(self, kappa):
        # 计算 von Mises 分布的熵
        # 公式来源于 von Mises 分布的熵定义：
        # vonmises.entropy(kappa) = -kappa * I[1](kappa) / I[0](kappa) +
        #                           log(2 * np.pi * I[0](kappa))
        # 其中 I[0](kappa) 和 I[1](kappa) 分别是修改过的零阶和一阶贝塞尔函数
        return (-kappa * sc.i1e(kappa) / sc.i0e(kappa) +
                np.log(2 * np.pi * sc.i0e(kappa)) + kappa)

    @extend_notes_in_docstring(rv_continuous, notes="""\
        The default limits of integration are endpoints of the interval
        of width ``2*pi`` centered at `loc` (e.g. ``[-pi, pi]`` when
        ``loc=0``).\n\n""")
    def expect(self, func=None, args=(), loc=0, scale=1, lb=None, ub=None,
               conditional=False, **kwds):
        _a, _b = -np.pi, np.pi

        # 如果未指定下界 lb，则默认为 loc + _a
        if lb is None:
            lb = loc + _a
        # 如果未指定上界 ub，则默认为 loc + _b
        if ub is None:
            ub = loc + _b

        # 调用父类的 expect 方法，返回期望值
        return super().expect(func, args, loc,
                              scale, lb, ub, conditional, **kwds)

    @_call_super_mom
    @extend_notes_in_docstring(rv_continuous, notes="""\
        Fit data is assumed to represent angles and will be wrapped onto the
        unit circle. `f0` and `fscale` are ignored; the returned shape is
        always the maximum likelihood estimate and the scale is always
        1. Initial guesses are ignored.\n\n""")
# 创建一个 von Mises 分布的随机变量生成器，用于生成符合 von Mises 分布的随机数
vonmises = vonmises_gen(name='vonmises')

# 创建一个 von Mises 分布的随机变量生成器，限定范围在 [-π, π] 内，用于生成符合 von Mises 分布的随机数
vonmises_line = vonmises_gen(a=-np.pi, b=np.pi, name='vonmises_line')

# 定义一个 Wald 分布的随机变量生成器，继承自 invgauss_gen 类
class wald_gen(invgauss_gen):
    r"""A Wald continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `wald` is:

    .. math::

        f(x) = \frac{1}{\sqrt{2\pi x^3}} \exp(- \frac{ (x-1)^2 }{ 2x })

    for :math:`x >= 0`.

    `wald` is a special case of `invgauss` with ``mu=1``.

    %(after_notes)s

    %(example)s
    """
    _support_mask = rv_continuous._open_support_mask

    # 返回一个空列表，表明没有额外的形状信息
    def _shape_info(self):
        return []

    # 生成符合 Wald 分布的随机数
    def _rvs(self, size=None, random_state=None):
        return random_state.wald(1.0, 1.0, size=size)

    # Wald 分布的概率密度函数
    def _pdf(self, x):
        # wald.pdf(x) = 1/sqrt(2*pi*x**3) * exp(-(x-1)**2/(2*x))
        return invgauss._pdf(x, 1.0)

    # Wald 分布的累积分布函数
    def _cdf(self, x):
        return invgauss._cdf(x, 1.0)

    # Wald 分布的生存函数 (1 - CDF)
    def _sf(self, x):
        return invgauss._sf(x, 1.0)

    # Wald 分布的分位点函数 (CDF 的反函数)
    def _ppf(self, x):
        return invgauss._ppf(x, 1.0)

    # Wald 分布的逆生存函数 (SF 的反函数)
    def _isf(self, x):
        return invgauss._isf(x, 1.0)

    # Wald 分布的对数概率密度函数
    def _logpdf(self, x):
        return invgauss._logpdf(x, 1.0)

    # Wald 分布的对数累积分布函数
    def _logcdf(self, x):
        return invgauss._logcdf(x, 1.0)

    # Wald 分布的对数生存函数
    def _logsf(self, x):
        return invgauss._logsf(x, 1.0)

    # Wald 分布的统计量：均值、方差、偏度、峰度
    def _stats(self):
        return 1.0, 1.0, 3.0, 15.0

    # Wald 分布的熵
    def _entropy(self):
        return invgauss._entropy(1.0)

# 创建一个 mu=1 的 Wald 分布的随机变量生成器
wald = wald_gen(a=0.0, name="wald")

# 定义一个 wrapped Cauchy 分布的随机变量生成器，继承自 rv_continuous 类
class wrapcauchy_gen(rv_continuous):
    r"""A wrapped Cauchy continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `wrapcauchy` is:

    .. math::

        f(x, c) = \frac{1-c^2}{2\pi (1+c^2 - 2c \cos(x))}

    for :math:`0 \le x \le 2\pi`, :math:`0 < c < 1`.

    `wrapcauchy` takes ``c`` as a shape parameter for :math:`c`.

    %(after_notes)s

    %(example)s

    """
    # 参数检查函数，确保 c 在 (0, 1) 之间
    def _argcheck(self, c):
        return (c > 0) & (c < 1)

    # 返回一个形状信息列表，描述了参数 c 的限制
    def _shape_info(self):
        return [_ShapeInfo("c", False, (0, 1), (False, False))]

    # wrapped Cauchy 分布的概率密度函数
    def _pdf(self, x, c):
        # wrapcauchy.pdf(x, c) = (1-c**2) / (2*pi*(1+c**2-2*c*cos(x)))
        return (1.0-c*c)/(2*np.pi*(1+c*c-2*c*np.cos(x)))

    # wrapped Cauchy 分布的累积分布函数
    def _cdf(self, x, c):

        # 在 0 <= x < pi 区间内的累积分布函数
        def f1(x, cr):
            return 1/np.pi * np.arctan(cr*np.tan(x/2))

        # 在 pi <= x <= 2*pi 区间内的累积分布函数
        def f2(x, cr):
            return 1 - 1/np.pi * np.arctan(cr*np.tan((2*np.pi - x)/2))

        cr = (1 + c)/(1 - c)
        return _lazywhere(x < np.pi, (x, cr), f=f1, f2=f2)

    # wrapped Cauchy 分布的分位点函数 (CDF 的反函数)
    def _ppf(self, q, c):
        val = (1.0-c)/(1.0+c)
        rcq = 2*np.arctan(val*np.tan(np.pi*q))
        rcmq = 2*np.pi-2*np.arctan(val*np.tan(np.pi*(1-q)))
        return np.where(q < 1.0/2, rcq, rcmq)

    # wrapped Cauchy 分布的熵
    def _entropy(self, c):
        return np.log(2*np.pi*(1-c*c))
    # 定义一个方法 `_fitstart`，用于初始化参数估计
    def _fitstart(self, data):
        # 如果传入的数据是 CensoredData 类型的对象，则先解除其被屏蔽的部分
        if isinstance(data, CensoredData):
            data = data._uncensor()
        # 返回初始的参数估计值：shape 参数为 0.5，
        # location 参数为数据中的最小值，
        # scale 参数为数据的峰-峰值除以（2*pi）
        return 0.5, np.min(data), np.ptp(data)/(2*np.pi)
# 创建一个名称为 wrapcauchy 的变量，使用 wrapcauchy_gen 函数生成，参数 a 和 b 分别设定为 0.0 和 2π，名称设置为 'wrapcauchy'
wrapcauchy = wrapcauchy_gen(a=0.0, b=2*np.pi, name='wrapcauchy')


# 创建一个名为 gennorm_gen 的类，继承自 rv_continuous 类
class gennorm_gen(rv_continuous):
    r"""A generalized normal continuous random variable.

    %(before_notes)s

    See Also
    --------
    laplace : Laplace distribution
    norm : normal distribution

    Notes
    -----
    The probability density function for `gennorm` is [1]_:

    .. math::

        f(x, \beta) = \frac{\beta}{2 \Gamma(1/\beta)} \exp(-|x|^\beta),

    where :math:`x` is a real number, :math:`\beta > 0` and
    :math:`\Gamma` is the gamma function (`scipy.special.gamma`).

    `gennorm` takes ``beta`` as a shape parameter for :math:`\beta`.
    For :math:`\beta = 1`, it is identical to a Laplace distribution.
    For :math:`\beta = 2`, it is identical to a normal distribution
    (with ``scale=1/sqrt(2)``).

    References
    ----------

    .. [1] "Generalized normal distribution, Version 1",
           https://en.wikipedia.org/wiki/Generalized_normal_distribution#Version_1

    .. [2] Nardon, Martina, and Paolo Pianca. "Simulation techniques for
           generalized Gaussian densities." Journal of Statistical
           Computation and Simulation 79.11 (2009): 1317-1329

    .. [3] Wicklin, Rick. "Simulate data from a generalized Gaussian
           distribution" in The DO Loop blog, September 21, 2016,
           https://blogs.sas.com/content/iml/2016/09/21/simulate-generalized-gaussian-sas.html

    %(example)s

    """
    
    # 定义返回形状信息的方法，这里返回一个 _ShapeInfo 对象的列表，描述了参数 beta 的特性
    def _shape_info(self):
        return [_ShapeInfo("beta", False, (0, np.inf), (False, False))]

    # 定义概率密度函数（PDF），接受参数 x 和 beta，返回 generalized normal 分布的概率密度函数值
    def _pdf(self, x, beta):
        return np.exp(self._logpdf(x, beta))

    # 定义对数概率密度函数（log PDF），接受参数 x 和 beta，返回 generalized normal 分布的对数概率密度函数值
    def _logpdf(self, x, beta):
        return np.log(0.5*beta) - sc.gammaln(1.0/beta) - abs(x)**beta

    # 定义累积分布函数（CDF），接受参数 x 和 beta，返回 generalized normal 分布的累积分布函数值
    def _cdf(self, x, beta):
        c = 0.5 * np.sign(x)
        # 通过先计算 (.5 + c)，可以防止数值上的抵消效应
        return (0.5 + c) - c * sc.gammaincc(1.0/beta, abs(x)**beta)

    # 定义累积分布函数的逆函数（percent point function, PPF），接受参数 x 和 beta，返回 generalized normal 分布的 PPF
    def _ppf(self, x, beta):
        c = np.sign(x - 0.5)
        # 通过先计算 (1. + c)，可以防止数值上的抵消效应
        return c * sc.gammainccinv(1.0/beta, (1.0 + c) - 2.0*c*x)**(1.0/beta)

    # 定义生存函数（survival function, SF），接受参数 x 和 beta，返回 generalized normal 分布的生存函数值
    def _sf(self, x, beta):
        return self._cdf(-x, beta)

    # 定义逆生存函数的方法（inverse survival function, ISF），接受参数 x 和 beta，返回 generalized normal 分布的 ISF
    def _isf(self, x, beta):
        return -self._ppf(x, beta)

    # 定义统计特性方法（stats），接受参数 beta，返回 generalized normal 分布的均值和方差
    def _stats(self, beta):
        c1, c3, c5 = sc.gammaln([1.0/beta, 3.0/beta, 5.0/beta])
        return 0., np.exp(c3 - c1), 0., np.exp(c5 + c1 - 2.0*c3) - 3.

    # 定义熵方法（entropy），接受参数 beta，返回 generalized normal 分布的熵值
    def _entropy(self, beta):
        return 1. / beta - np.log(.5 * beta) + sc.gammaln(1. / beta)

    # 定义随机变量生成方法（random variates, RVs），接受参数 beta 和 size（可选），返回 generalized normal 分布的随机样本
    def _rvs(self, beta, size=None, random_state=None):
        # 参考 [2]_ 中的算法
        # 参考 [3]_ 中 SAS 的实现方法
        z = random_state.gamma(1/beta, size=size)
        y = z ** (1/beta)
        # 将 y 转换为数组，以支持掩码操作
        y = np.asarray(y)
        mask = random_state.random(size=y.shape) < 0.5
        y[mask] = -y[mask]
        return y


# 创建一个名为 gennorm 的变量，实例化 gennorm_gen 类，名称设置为 'gennorm'
gennorm = gennorm_gen(name='gennorm')


# 创建一个名为 halfgennorm_gen 的类，继承自 rv_continuous 类
    r"""The upper half of a generalized normal continuous random variable.

    %(before_notes)s

    See Also
    --------
    gennorm : generalized normal distribution
    expon : exponential distribution
    halfnorm : half normal distribution

    Notes
    -----
    The probability density function for `halfgennorm` is:

    .. math::

        f(x, \beta) = \frac{\beta}{\Gamma(1/\beta)} \exp(-|x|^\beta)

    for :math:`x, \beta > 0`. :math:`\Gamma` is the gamma function
    (`scipy.special.gamma`).

    `halfgennorm` takes ``beta`` as a shape parameter for :math:`\beta`.
    For :math:`\beta = 1`, it is identical to an exponential distribution.
    For :math:`\beta = 2`, it is identical to a half normal distribution
    (with ``scale=1/sqrt(2)``).

    References
    ----------

    .. [1] "Generalized normal distribution, Version 1",
           https://en.wikipedia.org/wiki/Generalized_normal_distribution#Version_1

    %(example)s

    """
    # 定义一个私有方法 _shape_info，返回一个 _ShapeInfo 对象列表，描述了参数 beta 的形状信息
    def _shape_info(self):
        return [_ShapeInfo("beta", False, (0, np.inf), (False, False))]

    # 定义概率密度函数 _pdf，接受参数 x 和 beta，返回 halfgennorm 的概率密度函数值
    def _pdf(self, x, beta):
        # 返回概率密度函数的指数部分的值，即 exp(-|x|**beta)
        return np.exp(self._logpdf(x, beta))

    # 定义对数概率密度函数 _logpdf，接受参数 x 和 beta，返回 halfgennorm 的对数概率密度函数值
    def _logpdf(self, x, beta):
        # 返回对数概率密度函数的值，计算公式为 np.log(beta) - sc.gammaln(1.0/beta) - x**beta
        return np.log(beta) - sc.gammaln(1.0/beta) - x**beta

    # 定义累积分布函数 _cdf，接受参数 x 和 beta，返回 halfgennorm 的累积分布函数值
    def _cdf(self, x, beta):
        # 使用 scipy 中的 gammainc 函数计算累积分布函数值
        return sc.gammainc(1.0/beta, x**beta)

    # 定义反函数 _ppf，接受参数 x 和 beta，返回 halfgennorm 的分位点函数值
    def _ppf(self, x, beta):
        # 使用 scipy 中的 gammaincinv 函数计算分位点函数值
        return sc.gammaincinv(1.0/beta, x)**(1.0/beta)

    # 定义生存函数 _sf，接受参数 x 和 beta，返回 halfgennorm 的生存函数值
    def _sf(self, x, beta):
        # 使用 scipy 中的 gammaincc 函数计算生存函数值
        return sc.gammaincc(1.0/beta, x**beta)

    # 定义反生存函数 _isf，接受参数 x 和 beta，返回 halfgennorm 的反生存函数值
    def _isf(self, x, beta):
        # 使用 scipy 中的 gammainccinv 函数计算反生存函数值
        return sc.gammainccinv(1.0/beta, x)**(1.0/beta)

    # 定义熵函数 _entropy，接受参数 beta，返回 halfgennorm 的熵值
    def _entropy(self, beta):
        # 计算熵的值，公式为 1.0/beta - np.log(beta) + sc.gammaln(1.0/beta)
        return 1.0/beta - np.log(beta) + sc.gammaln(1.0/beta)
halfgennorm = halfgennorm_gen(a=0, name='halfgennorm')

class crystalball_gen(rv_continuous):
    r"""
    Crystalball distribution

    %(before_notes)s

    Notes
    -----
    The probability density function for `crystalball` is:

    .. math::

        f(x, \beta, m) =  \begin{cases}
                            N \exp(-x^2 / 2),  &\text{for } x > -\beta\\
                            N A (B - x)^{-m}  &\text{for } x \le -\beta
                          \end{cases}

    where :math:`A = (m / |\beta|)^m  \exp(-\beta^2 / 2)`,
    :math:`B = m/|\beta| - |\beta|` and :math:`N` is a normalisation constant.

    `crystalball` takes :math:`\beta > 0` and :math:`m > 1` as shape
    parameters.  :math:`\beta` defines the point where the pdf changes
    from a power-law to a Gaussian distribution.  :math:`m` is the power
    of the power-law tail.

    %(after_notes)s

    .. versionadded:: 0.19.0

    References
    ----------
    .. [1] "Crystal Ball Function",
           https://en.wikipedia.org/wiki/Crystal_Ball_function

    %(example)s
    """
    
    def _argcheck(self, beta, m):
        """
        Check the validity of shape parameters beta and m.
        """
        return (m > 1) & (beta > 0)

    def _shape_info(self):
        """
        Return the shape information for beta and m.
        """
        ibeta = _ShapeInfo("beta", False, (0, np.inf), (False, False))
        im = _ShapeInfo("m", False, (1, np.inf), (False, False))
        return [ibeta, im]

    def _fitstart(self, data):
        """
        Provide starting values for fitting based on data.
        """
        # Arbitrary, but the default m=1 is not valid
        return super()._fitstart(data, args=(1, 1.5))

    def _pdf(self, x, beta, m):
        """
        Return PDF of the crystalball function.

        Parameters:
        - x: value at which to compute the PDF
        - beta: shape parameter > 0
        - m: shape parameter > 1

        Returns:
        - PDF value at x

        """
        N = 1.0 / (m/beta / (m-1) * np.exp(-beta**2 / 2.0) +
                   _norm_pdf_C * _norm_cdf(beta))

        def rhs(x, beta, m):
            return np.exp(-x**2 / 2)

        def lhs(x, beta, m):
            return ((m/beta)**m * np.exp(-beta**2 / 2.0) *
                    (m/beta - beta - x)**(-m))

        return N * _lazywhere(x > -beta, (x, beta, m), f=rhs, f2=lhs)

    def _logpdf(self, x, beta, m):
        """
        Return the log of the PDF of the crystalball function.

        Parameters:
        - x: value at which to compute the log PDF
        - beta: shape parameter > 0
        - m: shape parameter > 1

        Returns:
        - Logarithm of the PDF value at x

        """
        N = 1.0 / (m/beta / (m-1) * np.exp(-beta**2 / 2.0) +
                   _norm_pdf_C * _norm_cdf(beta))

        def rhs(x, beta, m):
            return -x**2/2

        def lhs(x, beta, m):
            return m*np.log(m/beta) - beta**2/2 - m*np.log(m/beta - beta - x)

        return np.log(N) + _lazywhere(x > -beta, (x, beta, m), f=rhs, f2=lhs)
    def _cdf(self, x, beta, m):
        """
        Return CDF of the crystalball function
        """
        # 计算归一化系数 N
        N = 1.0 / (m/beta / (m-1) * np.exp(-beta**2 / 2.0) +
                   _norm_pdf_C * _norm_cdf(beta)

        # 定义右侧函数 rhs(x, beta, m)
        def rhs(x, beta, m):
            return ((m/beta) * np.exp(-beta**2 / 2.0) / (m-1) +
                    _norm_pdf_C * (_norm_cdf(x) - _norm_cdf(-beta)))

        # 定义左侧函数 lhs(x, beta, m)
        def lhs(x, beta, m):
            return ((m/beta)**m * np.exp(-beta**2 / 2.0) *
                    (m/beta - beta - x)**(-m+1) / (m-1))

        # 返回根据条件延迟求值的结果
        return N * _lazywhere(x > -beta, (x, beta, m), f=rhs, f2=lhs)

    def _ppf(self, p, beta, m):
        # 计算归一化系数 N
        N = 1.0 / (m/beta / (m-1) * np.exp(-beta**2 / 2.0) +
                   _norm_pdf_C * _norm_cdf(beta))
        
        # 计算 pbeta
        pbeta = N * (m/beta) * np.exp(-beta**2/2) / (m - 1)

        # 定义在 p < pbeta 时的反函数 ppf_less(p, beta, m)
        def ppf_less(p, beta, m):
            eb2 = np.exp(-beta**2/2)
            C = (m/beta) * eb2 / (m-1)
            N = 1/(C + _norm_pdf_C * _norm_cdf(beta))
            return (m/beta - beta -
                    ((m - 1)*(m/beta)**(-m)/eb2*p/N)**(1/(1-m)))

        # 定义在 p >= pbeta 时的反函数 ppf_greater(p, beta, m)
        def ppf_greater(p, beta, m):
            eb2 = np.exp(-beta**2/2)
            C = (m/beta) * eb2 / (m-1)
            N = 1/(C + _norm_pdf_C * _norm_cdf(beta))
            return _norm_ppf(_norm_cdf(-beta) + (1/_norm_pdf_C)*(p/N - C))

        # 返回根据条件延迟求值的结果
        return _lazywhere(p < pbeta, (p, beta, m), f=ppf_less, f2=ppf_greater)

    def _munp(self, n, beta, m):
        """
        Returns the n-th non-central moment of the crystalball function.
        """
        # 计算归一化系数 N
        N = 1.0 / (m/beta / (m-1) * np.exp(-beta**2 / 2.0) +
                   _norm_pdf_C * _norm_cdf(beta))

        # 定义第 n 阶非中心矩 n_th_moment(n, beta, m)
        def n_th_moment(n, beta, m):
            """
            Returns n-th moment. Defined only if n+1 < m
            Function cannot broadcast due to the loop over n
            """
            A = (m/beta)**m * np.exp(-beta**2 / 2.0)
            B = m/beta - beta
            rhs = (2**((n-1)/2.0) * sc.gamma((n+1)/2) *
                   (1.0 + (-1)**n * sc.gammainc((n+1)/2, beta**2 / 2)))
            lhs = np.zeros(rhs.shape)
            for k in range(n + 1):
                lhs += (sc.binom(n, k) * B**(n-k) * (-1)**k / (m - k - 1) *
                        (m/beta)**(-m + k + 1))
            return A * lhs + rhs

        # 返回根据条件延迟求值的结果
        return N * _lazywhere(n + 1 < m, (n, beta, m),
                              np.vectorize(n_th_moment, otypes=[np.float64]),
                              np.inf)
# 创建一个名为 crystalball 的变量，并调用 crystalball_gen 函数来生成一个名为 crystalball 的对象，设置其长名称为 "A Crystalball Function"
crystalball = crystalball_gen(name='crystalball', longname="A Crystalball Function")


def _argus_phi(chi):
    """
    Utility function for the argus distribution used in the pdf, sf and
    moment calculation.
    Note that for all x > 0:
    gammainc(1.5, x**2/2) = 2 * (_norm_cdf(x) - x * _norm_pdf(x) - 0.5).
    This can be verified directly by noting that the cdf of Gamma(1.5) can
    be written as erf(sqrt(x)) - 2*sqrt(x)*exp(-x)/sqrt(Pi).
    We use gammainc instead of the usual definition because it is more precise
    for small chi.
    """
    # 计算 argus 分布中的辅助函数，用于计算概率密度函数、生存函数和矩
    return sc.gammainc(1.5, chi**2/2) / 2


class argus_gen(rv_continuous):
    r"""
    Argus distribution

    %(before_notes)s

    Notes
    -----
    The probability density function for `argus` is:

    .. math::

        f(x, \chi) = \frac{\chi^3}{\sqrt{2\pi} \Psi(\chi)} x \sqrt{1-x^2}
                     \exp(-\chi^2 (1 - x^2)/2)

    for :math:`0 < x < 1` and :math:`\chi > 0`, where

    .. math::

        \Psi(\chi) = \Phi(\chi) - \chi \phi(\chi) - 1/2

    with :math:`\Phi` and :math:`\phi` being the CDF and PDF of a standard
    normal distribution, respectively.

    `argus` takes :math:`\chi` as shape a parameter. Details about sampling
    from the ARGUS distribution can be found in [2]_.

    %(after_notes)s

    References
    ----------
    .. [1] "ARGUS distribution",
           https://en.wikipedia.org/wiki/ARGUS_distribution
    .. [2] Christoph Baumgarten "Random variate generation by fast numerical
           inversion in the varying parameter case." Research in Statistics,
           vol. 1, 2023, doi:10.1080/27684520.2023.2279060.

    .. versionadded:: 0.19.0

    %(example)s
    """

    def _shape_info(self):
        # 定义分布的形状信息，此处指定了参数 chi 的范围和属性
        return [_ShapeInfo("chi", False, (0, np.inf), (False, False))]

    def _logpdf(self, x, chi):
        # 对于 x = 0 或 1，logpdf 返回 -np.inf
        with np.errstate(divide='ignore'):
            y = 1.0 - x*x
            # 计算对数概率密度函数，用于计算参数 chi 下的 argus 分布
            A = 3*np.log(chi) - _norm_pdf_logC - np.log(_argus_phi(chi))
            return A + np.log(x) + 0.5*np.log1p(-x*x) - chi**2 * y / 2

    def _pdf(self, x, chi):
        # 计算概率密度函数，用于计算参数 chi 下的 argus 分布
        return np.exp(self._logpdf(x, chi))

    def _cdf(self, x, chi):
        # 计算累积分布函数，用于计算参数 chi 下的 argus 分布
        return 1.0 - self._sf(x, chi)

    def _sf(self, x, chi):
        # 计算生存函数，用于计算参数 chi 下的 argus 分布
        return _argus_phi(chi * np.sqrt(1 - x**2)) / _argus_phi(chi)
    def _rvs(self, chi, size=None, random_state=None):
        # 将输入参数 chi 转换为 NumPy 数组
        chi = np.asarray(chi)
        # 如果 chi 的大小为 1，调用 _rvs_scalar 处理
        if chi.size == 1:
            out = self._rvs_scalar(chi, numsamples=size,
                                   random_state=random_state)
        else:
            # 检查 chi 的形状并处理广播
            shp, bc = _check_shape(chi.shape, size)
            # 计算样本数
            numsamples = int(np.prod(shp))
            # 创建输出数组
            out = np.empty(size)
            # 迭代器遍历 chi
            it = np.nditer([chi],
                           flags=['multi_index'],
                           op_flags=[['readonly']])
            while not it.finished:
                # 根据多索引创建索引元组
                idx = tuple((it.multi_index[j] if not bc[j] else slice(None))
                            for j in range(-len(size), 0))
                # 调用 _rvs_scalar 处理当前索引处的 chi 值
                r = self._rvs_scalar(it[0], numsamples=numsamples,
                                     random_state=random_state)
                # 将结果放入输出数组相应的位置
                out[idx] = r.reshape(shp)
                # 迭代到下一个元素
                it.iternext()

        # 如果 size 是空元组，将结果转为标量
        if size == ():
            out = out[()]
        # 返回处理后的结果
        return out

    def _stats(self, chi):
        # 将输入参数 chi 转换为 NumPy 数组，并确保其类型为 float
        # 这样可以确保后续的掩码操作适用于整数类型的输入
        chi = np.asarray(chi, dtype=float)
        # 计算 Argus 分布的 phi 参数
        phi = _argus_phi(chi)
        # 计算 m 值，根据公式计算
        m = np.sqrt(np.pi/8) * chi * sc.ive(1, chi**2/4) / phi
        # 计算第二矩，对于小的 chi 值使用 Taylor 展开 (<= 0.1)
        mu2 = np.empty_like(chi)
        mask = chi > 0.1
        c = chi[mask]
        mu2[mask] = 1 - 3 / c**2 + c * _norm_pdf(c) / phi[mask]
        c = chi[~mask]
        coef = [-358/65690625, 0, -94/1010625, 0, 2/2625, 0, 6/175, 0, 0.4]
        mu2[~mask] = np.polyval(coef, c)
        # 返回计算得到的 m 值、mu2 值、以及两个 None（表示未使用）
        return m, mu2 - m**2, None, None
# 使用指定参数调用argus_gen函数生成argus对象
argus = argus_gen(name='argus', longname="An Argus Function", a=0.0, b=1.0)

# 定义rv_histogram类，继承自rv_continuous类，用于根据直方图生成分布
class rv_histogram(rv_continuous):
    """
    Generates a distribution given by a histogram.
    This is useful to generate a template distribution from a binned
    datasample.

    As a subclass of the `rv_continuous` class, `rv_histogram` inherits from it
    a collection of generic methods (see `rv_continuous` for the full list),
    and implements them based on the properties of the provided binned
    datasample.

    Parameters
    ----------
    histogram : tuple of array_like
        Tuple containing two array_like objects.
        The first containing the content of n bins,
        the second containing the (n+1) bin boundaries.
        In particular, the return value of `numpy.histogram` is accepted.

    density : bool, optional
        If False, assumes the histogram is proportional to counts per bin;
        otherwise, assumes it is proportional to a density.
        For constant bin widths, these are equivalent, but the distinction
        is important when bin widths vary (see Notes).
        If None (default), sets ``density=True`` for backwards compatibility,
        but warns if the bin widths are variable. Set `density` explicitly
        to silence the warning.

        .. versionadded:: 1.10.0

    Notes
    -----
    When a histogram has unequal bin widths, there is a distinction between
    histograms that are proportional to counts per bin and histograms that are
    proportional to probability density over a bin. If `numpy.histogram` is
    called with its default ``density=False``, the resulting histogram is the
    number of counts per bin, so ``density=False`` should be passed to
    `rv_histogram`. If `numpy.histogram` is called with ``density=True``, the
    resulting histogram is in terms of probability density, so ``density=True``
    should be passed to `rv_histogram`. To avoid warnings, always pass
    ``density`` explicitly when the input histogram has unequal bin widths.

    There are no additional shape parameters except for the loc and scale.
    The pdf is defined as a stepwise function from the provided histogram.
    The cdf is a linear interpolation of the pdf.

    .. versionadded:: 0.19.0

    Examples
    --------

    Create a scipy.stats distribution from a numpy histogram

    >>> import scipy.stats
    >>> import numpy as np
    >>> data = scipy.stats.norm.rvs(size=100000, loc=0, scale=1.5,
    ...                             random_state=123)
    >>> hist = np.histogram(data, bins=100)
    >>> hist_dist = scipy.stats.rv_histogram(hist, density=False)

    Behaves like an ordinary scipy rv_continuous distribution

    >>> hist_dist.pdf(1.0)
    0.20538577847618705
    >>> hist_dist.cdf(2.0)
    0.90818568543056499

    PDF is zero above (below) the highest (lowest) bin of the histogram,
    defined by the max (min) of the original dataset

    >>> hist_dist.pdf(np.max(data))
    0.0
    >>> hist_dist.cdf(np.max(data))
    # 调用概率分布对象的 PDF 方法，计算数据中最小值的概率密度函数值
    >>> hist_dist.pdf(np.min(data))
    7.7591907244498314e-05
    # 调用概率分布对象的 CDF 方法，计算数据中最小值的累积分布函数值
    >>> hist_dist.cdf(np.min(data))
    0.0

    # PDF 和 CDF 与直方图相对应

    # 导入 matplotlib.pyplot 库，简称 plt
    >>> import matplotlib.pyplot as plt
    # 在 -5.0 到 5.0 范围内生成包含 100 个点的等间隔数组 X
    >>> X = np.linspace(-5.0, 5.0, 100)
    # 创建图形 fig 和坐标轴 ax
    >>> fig, ax = plt.subplots()
    # 设置坐标轴标题
    >>> ax.set_title("PDF from Template")
    # 绘制数据的直方图，设置为密度图并分成 100 个区间
    >>> ax.hist(data, density=True, bins=100)
    # 绘制概率密度函数 PDF 的曲线
    >>> ax.plot(X, hist_dist.pdf(X), label='PDF')
    # 绘制累积分布函数 CDF 的曲线
    >>> ax.plot(X, hist_dist.cdf(X), label='CDF')
    # 添加图例
    >>> ax.legend()
    # 显示图形
    >>> fig.show()

    """
    # 定义支持掩码，继承自 rv_continuous 的支持掩码
    _support_mask = rv_continuous._support_mask

    # 初始化方法，使用给定的直方图创建一个新的分布对象
    def __init__(self, histogram, *args, density=None, **kwargs):
        """
        使用给定的直方图创建一个新的分布对象

        Parameters
        ----------
        histogram : tuple of array_like
            包含两个 array_like 对象的元组。
            第一个包含 n 个柱的内容，
            第二个包含 (n+1) 个柱的边界。
            特别地，np.histogram 的返回值将被接受。
        density : bool, optional
            如果为 False，假设直方图与每个柱的计数成比例；
            否则假设它与密度成比例。
            对于恒定的柱宽度，这两者是等价的。
            如果为 None（默认），则为了向后兼容性，设置 `density=True`，
            但如果柱宽度可变，则会发出警告。显式设置 `density` 可以消除警告。
        """
        # 存储直方图和密度属性
        self._histogram = histogram
        self._density = density
        # 检查直方图的长度是否为2
        if len(histogram) != 2:
            raise ValueError("Expected length 2 for parameter histogram")
        # 将直方图内容和边界转换为 NumPy 数组
        self._hpdf = np.asarray(histogram[0])
        self._hbins = np.asarray(histogram[1])
        # 检查直方图内容和边界的元素数量是否匹配
        if len(self._hpdf) + 1 != len(self._hbins):
            raise ValueError("Number of elements in histogram content "
                             "and histogram boundaries do not match, "
                             "expected n and n+1.")
        # 计算直方图每个柱的宽度
        self._hbin_widths = self._hbins[1:] - self._hbins[:-1]
        # 检查柱宽度是否恒定，如果未指定密度且柱宽度可变，发出警告
        bins_vary = not np.allclose(self._hbin_widths, self._hbin_widths[0])
        if density is None and bins_vary:
            message = ("Bin widths are not constant. Assuming `density=True`."
                       "Specify `density` explicitly to silence this warning.")
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            density = True
        # 如果不是密度直方图，将直方图内容除以柱宽度以得到密度
        elif not density:
            self._hpdf = self._hpdf / self._hbin_widths

        # 将直方图内容归一化为总面积为1的概率密度函数
        self._hpdf = self._hpdf / float(np.sum(self._hpdf * self._hbin_widths))
        # 计算累积分布函数
        self._hcdf = np.cumsum(self._hpdf * self._hbin_widths)
        # 在概率密度函数数组两侧添加零元素，用于边界条件
        self._hpdf = np.hstack([0.0, self._hpdf, 0.0])
        self._hcdf = np.hstack([0.0, self._hcdf])
        # 设置支持区间的上下界
        kwargs['a'] = self.a = self._hbins[0]
        kwargs['b'] = self.b = self._hbins[-1]
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
    def _pdf(self, x):
        """
        PDF of the histogram
        
        根据给定的输入 x，返回直方图的概率密度函数值。
        """
        return self._hpdf[np.searchsorted(self._hbins, x, side='right')]

    def _cdf(self, x):
        """
        CDF calculated from the histogram
        
        根据直方图计算累积分布函数（CDF）。
        """
        return np.interp(x, self._hbins, self._hcdf)

    def _ppf(self, x):
        """
        Percentile function calculated from the histogram
        
        根据直方图计算百分位点函数（PPF）。
        """
        return np.interp(x, self._hcdf, self._hbins)

    def _munp(self, n):
        """
        Compute the n-th non-central moment
        
        计算直方图的 n 阶非中心矩。
        """
        integrals = (self._hbins[1:]**(n+1) - self._hbins[:-1]**(n+1)) / (n+1)
        return np.sum(self._hpdf[1:-1] * integrals)

    def _entropy(self):
        """
        Compute entropy of distribution
        
        计算分布的信息熵。
        """
        res = _lazywhere(self._hpdf[1:-1] > 0.0,
                         (self._hpdf[1:-1],),
                         np.log,
                         0.0)
        return -np.sum(self._hpdf[1:-1] * res * self._hbin_widths)

    def _updated_ctor_param(self):
        """
        Set the histogram as additional constructor argument
        
        将直方图设置为额外的构造函数参数。
        """
        dct = super()._updated_ctor_param()
        dct['histogram'] = self._histogram
        dct['density'] = self._density
        return dct
# 定义一个自定义的连续随机变量类 `studentized_range_gen`，继承自 `rv_continuous`
class studentized_range_gen(rv_continuous):
    # 一个学生化区间连续随机变量的描述性字符串
    r"""A studentized range continuous random variable.

    %(before_notes)s

    See Also
    --------
    t: Student's t distribution

    Notes
    -----
    The probability density function for `studentized_range` is:

    .. math::

         f(x; k, \nu) = \frac{k(k-1)\nu^{\nu/2}}{\Gamma(\nu/2)
                        2^{\nu/2-1}} \int_{0}^{\infty} \int_{-\infty}^{\infty}
                        s^{\nu} e^{-\nu s^2/2} \phi(z) \phi(sx + z)
                        [\Phi(sx + z) - \Phi(z)]^{k-2} \,dz \,ds

    for :math:`x ≥ 0`, :math:`k > 1`, and :math:`\nu > 0`.

    `studentized_range` takes ``k`` for :math:`k` and ``df`` for :math:`\nu`
    as shape parameters.

    When :math:`\nu` exceeds 100,000, an asymptotic approximation (infinite
    degrees of freedom) is used to compute the cumulative distribution
    function [4]_ and probability distribution function.

    %(after_notes)s

    References
    ----------

    .. [1] "Studentized range distribution",
           https://en.wikipedia.org/wiki/Studentized_range_distribution
    .. [2] Batista, Ben Dêivide, et al. "Externally Studentized Normal Midrange
           Distribution." Ciência e Agrotecnologia, vol. 41, no. 4, 2017, pp.
           378-389., doi:10.1590/1413-70542017414047716.
    .. [3] Harter, H. Leon. "Tables of Range and Studentized Range." The Annals
           of Mathematical Statistics, vol. 31, no. 4, 1960, pp. 1122-1147.
           JSTOR, www.jstor.org/stable/2237810. Accessed 18 Feb. 2021.
    .. [4] Lund, R. E., and J. R. Lund. "Algorithm AS 190: Probabilities and
           Upper Quantiles for the Studentized Range." Journal of the Royal
           Statistical Society. Series C (Applied Statistics), vol. 32, no. 2,
           1983, pp. 204-210. JSTOR, www.jstor.org/stable/2347300. Accessed 18
           Feb. 2021.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import studentized_range
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1, 1)

    Display the probability density function (``pdf``):

    >>> k, df = 3, 10
    >>> x = np.linspace(studentized_range.ppf(0.01, k, df),
    ...                 studentized_range.ppf(0.99, k, df), 100)
    >>> ax.plot(x, studentized_range.pdf(x, k, df),
    ...         'r-', lw=5, alpha=0.6, label='studentized_range pdf')

    Alternatively, the distribution object can be called (as a function)
    to fix the shape, location and scale parameters. This returns a "frozen"
    RV object holding the given parameters fixed.

    Freeze the distribution and display the frozen ``pdf``:

    >>> rv = studentized_range(k, df)
    >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

    Check accuracy of ``cdf`` and ``ppf``:

    >>> vals = studentized_range.ppf([0.001, 0.5, 0.999], k, df)
    >>> np.allclose([0.001, 0.5, 0.999], studentized_range.cdf(vals, k, df))
    True
    ```
    def _argcheck(self, k, df):
        # 检查参数 k 和 df 的有效性
        return (k > 1) & (df > 0)

    def _shape_info(self):
        # 返回参数 k 和 df 的形状信息
        ik = _ShapeInfo("k", False, (1, np.inf), (False, False))
        idf = _ShapeInfo("df", False, (0, np.inf), (False, False))
        return [ik, idf]

    def _fitstart(self, data):
        # 如果参数 k 默认为 1，则使用参数值 2 替代，因为 1 不是有效的参数值
        return super()._fitstart(data, args=(2, 1))

    def _munp(self, K, k, df):
        cython_symbol = '_studentized_range_moment'
        _a, _b = self._get_support()
        # 所有这三个值用于创建一个形状相同的 numpy 数组，因此它们必须具有相同的形状。

        def _single_moment(K, k, df):
            # 计算单个矩（moment）
            log_const = _stats._studentized_range_pdf_logconst(k, df)
            arg = [K, k, df, log_const]
            usr_data = np.array(arg, float).ctypes.data_as(ctypes.c_void_p)

            # 从 Cython 函数创建低级可调用对象
            llc = LowLevelCallable.from_cython(_stats, cython_symbol, usr_data)

            ranges = [(-np.inf, np.inf), (0, np.inf), (_a, _b)]
            opts = dict(epsabs=1e-11, epsrel=1e-12)

            # 数值积分计算
            return integrate.nquad(llc, ranges=ranges, opts=opts)[0]

        # 创建一个通用函数对象
        ufunc = np.frompyfunc(_single_moment, 3, 1)
        return np.asarray(ufunc(K, k, df), dtype=np.float64)[()]
    def _pdf(self, x, k, df):
        # 定义内部函数 _single_pdf，计算学生 t 分布的概率密度函数
        def _single_pdf(q, k, df):
            # 如果自由度 df 小于 100000，选择使用 _studentized_range_pdf
            # 计算学生 t 分布的概率密度函数的对数常数
            if df < 100000:
                cython_symbol = '_studentized_range_pdf'
                log_const = _stats._studentized_range_pdf_logconst(k, df)
                arg = [q, k, df, log_const]
                usr_data = np.array(arg, float).ctypes.data_as(ctypes.c_void_p)
                ranges = [(-np.inf, np.inf), (0, np.inf)]
            else:
                # 如果自由度 df 大于等于 100000，选择使用 _studentized_range_pdf_asymptotic
                cython_symbol = '_studentized_range_pdf_asymptotic'
                arg = [q, k]
                usr_data = np.array(arg, float).ctypes.data_as(ctypes.c_void_p)
                ranges = [(-np.inf, np.inf)]
            
            # 从 Cython 函数创建低级回调函数
            llc = LowLevelCallable.from_cython(_stats, cython_symbol, usr_data)
            opts = dict(epsabs=1e-11, epsrel=1e-12)
            # 使用数值积分计算概率密度函数
            return integrate.nquad(llc, ranges=ranges, opts=opts)[0]

        # 将 _single_pdf 函数转换为 NumPy 通用函数
        ufunc = np.frompyfunc(_single_pdf, 3, 1)
        # 返回 x, k, df 参数对应的概率密度函数值，转换为 float64 类型的数组后返回其第一个元素
        return np.asarray(ufunc(x, k, df), dtype=np.float64)[()]

    def _cdf(self, x, k, df):
        # 定义内部函数 _single_cdf，计算学生 t 分布的累积分布函数
        def _single_cdf(q, k, df):
            # 如果自由度 df 小于 100000，选择使用 _studentized_range_cdf
            # 计算学生 t 分布的累积分布函数的对数常数
            if df < 100000:
                cython_symbol = '_studentized_range_cdf'
                log_const = _stats._studentized_range_cdf_logconst(k, df)
                arg = [q, k, df, log_const]
                usr_data = np.array(arg, float).ctypes.data_as(ctypes.c_void_p)
                ranges = [(-np.inf, np.inf), (0, np.inf)]
            else:
                # 如果自由度 df 大于等于 100000，选择使用 _studentized_range_cdf_asymptotic
                cython_symbol = '_studentized_range_cdf_asymptotic'
                arg = [q, k]
                usr_data = np.array(arg, float).ctypes.data_as(ctypes.c_void_p)
                ranges = [(-np.inf, np.inf)]
            
            # 从 Cython 函数创建低级回调函数
            llc = LowLevelCallable.from_cython(_stats, cython_symbol, usr_data)
            opts = dict(epsabs=1e-11, epsrel=1e-12)
            # 使用数值积分计算累积分布函数
            return integrate.nquad(llc, ranges=ranges, opts=opts)[0]

        # 将 _single_cdf 函数转换为 NumPy 通用函数
        ufunc = np.frompyfunc(_single_cdf, 3, 1)

        # 计算 x, k, df 参数对应的累积分布函数值，并确保结果在 [0, 1] 范围内
        return np.clip(np.asarray(ufunc(x, k, df), dtype=np.float64)[()], 0, 1)
# 创建一个 Studentized Range 分布生成器对象，并命名为 studentized_range
studentized_range = studentized_range_gen(name='studentized_range', a=0,
                                          b=np.inf)


class rel_breitwigner_gen(rv_continuous):
    r"""A relativistic Breit-Wigner random variable.

    %(before_notes)s

    See Also
    --------
    cauchy: Cauchy distribution, also known as the Breit-Wigner distribution.

    Notes
    -----

    The probability density function for `rel_breitwigner` is

    .. math::

        f(x, \rho) = \frac{k}{(x^2 - \rho^2)^2 + \rho^2}

    where

    .. math::
        k = \frac{2\sqrt{2}\rho^2\sqrt{\rho^2 + 1}}
            {\pi\sqrt{\rho^2 + \rho\sqrt{\rho^2 + 1}}}

    The relativistic Breit-Wigner distribution is used in high energy physics
    to model resonances [1]_. It gives the uncertainty in the invariant mass,
    :math:`M` [2]_, of a resonance with characteristic mass :math:`M_0` and
    decay-width :math:`\Gamma`, where :math:`M`, :math:`M_0` and :math:`\Gamma`
    are expressed in natural units. In SciPy's parametrization, the shape
    parameter :math:`\rho` is equal to :math:`M_0/\Gamma` and takes values in
    :math:`(0, \infty)`.

    Equivalently, the relativistic Breit-Wigner distribution is said to give
    the uncertainty in the center-of-mass energy :math:`E_{\text{cm}}`. In
    natural units, the speed of light :math:`c` is equal to 1 and the invariant
    mass :math:`M` is equal to the rest energy :math:`Mc^2`. In the
    center-of-mass frame, the rest energy is equal to the total energy [3]_.

    %(after_notes)s

    :math:`\rho = M/\Gamma` and :math:`\Gamma` is the scale parameter. For
    example, if one seeks to model the :math:`Z^0` boson with :math:`M_0
    \approx 91.1876 \text{ GeV}` and :math:`\Gamma \approx 2.4952\text{ GeV}`
    [4]_ one can set ``rho=91.1876/2.4952`` and ``scale=2.4952``.

    To ensure a physically meaningful result when using the `fit` method, one
    should set ``floc=0`` to fix the location parameter to 0.

    References
    ----------
    .. [1] Relativistic Breit-Wigner distribution, Wikipedia,
           https://en.wikipedia.org/wiki/Relativistic_Breit-Wigner_distribution
    .. [2] Invariant mass, Wikipedia,
           https://en.wikipedia.org/wiki/Invariant_mass
    .. [3] Center-of-momentum frame, Wikipedia,
           https://en.wikipedia.org/wiki/Center-of-momentum_frame
    .. [4] M. Tanabashi et al. (Particle Data Group) Phys. Rev. D 98, 030001 -
           Published 17 August 2018

    %(example)s

    """
    # 定义参数检查函数，确保 rho 大于 0
    def _argcheck(self, rho):
        return rho > 0

    # 返回参数形状信息，这里定义了 rho 的范围为 (0, np.inf)
    def _shape_info(self):
        return [_ShapeInfo("rho", False, (0, np.inf), (False, False))]

    # 定义概率密度函数 PDF，计算并返回相对论 Breit-Wigner 分布的概率密度值
    def _pdf(self, x, rho):
        # 计算常数 C，用于归一化概率密度函数
        C = np.sqrt(
            2 * (1 + 1/rho**2) / (1 + np.sqrt(1 + 1/rho**2))
        ) * 2 / np.pi
        # 使用特定的数值处理上溢出错误，然后返回概率密度函数的值
        with np.errstate(over='ignore'):
            return C / (((x - rho)*(x + rho)/rho)**2 + 1)
    def _cdf(self, x, rho):
        # 计算常数 C
        C = np.sqrt(2/(1 + np.sqrt(1 + 1/rho**2)))/np.pi
        # 计算累积分布函数的结果
        result = (
            np.sqrt(-1 + 1j/rho)
            * np.arctan(x/np.sqrt(-rho*(rho + 1j)))
        )
        result = C * 2 * np.imag(result)
        # 对结果进行裁剪，确保不超过1
        return np.clip(result, None, 1)

    def _munp(self, n, rho):
        if n == 1:
            # 计算常数 C，对应 n = 1 的情况
            C = np.sqrt(
                2 * (1 + 1/rho**2) / (1 + np.sqrt(1 + 1/rho**2))
            ) / np.pi * rho
            # 计算原点矩的结果
            return C * (np.pi/2 + np.arctan(rho))
        if n == 2:
            # 计算常数 C，对应 n = 2 的情况
            C = np.sqrt(
                (1 + 1/rho**2) / (2 * (1 + np.sqrt(1 + 1/rho**2)))
            ) * rho
            # 计算二阶原点矩的结果
            result = (1 - rho * 1j) / np.sqrt(-1 - 1j/rho)
            return 2 * C * np.real(result)
        else:
            # 对于其他 n 的情况，返回无穷大
            return np.inf

    def _stats(self, rho):
        # 返回 None 表示不提供统计信息，公共统计函数将使用 _munp
        # NaN 值将在公共统计中被省略。偏度和峰度实际上是无穷大。
        return None, None, np.nan, np.nan

    @inherit_docstring_from(rv_continuous)
    def fit(self, data, *args, **kwds):
        # 重写 rv_continuous.fit 方法以更好地处理 floc 被设置的情况
        data, _, floc, fscale = _check_fit_input_parameters(
            self, data, args, kwds
        )

        censored = isinstance(data, CensoredData)
        if censored:
            if data.num_censored() == 0:
                # 数据中没有被审查的值，将 CensoredData 实例替换为普通数组
                data = data._uncensored
                censored = False

        if floc is None or censored:
            # 如果 floc 为 None 或数据中存在审查的值，则调用父类的 fit 方法
            return super().fit(data, *args, **kwds)

        if fscale is None:
            # 四分位距近似表示尺度参数 gamma
            # 中位数近似表示 rho * gamma
            p25, p50, p75 = np.quantile(data - floc, [0.25, 0.5, 0.75])
            scale_0 = p75 - p25
            rho_0 = p50 / scale_0
            if not args:
                args = [rho_0]
            if "scale" not in kwds:
                kwds["scale"] = scale_0
        else:
            # M_0 表示中位数，rho_0 表示 M_0 / fscale
            M_0 = np.median(data - floc)
            rho_0 = M_0 / fscale
            if not args:
                args = [rho_0]
        # 调用父类的 fit 方法，并返回其结果
        return super().fit(data, *args, **kwds)
# 使用 rel_breitwigner_gen 函数生成一个相对 Breit-Wigner 分布对象，参数 a 设置为 0.0，名称设置为 "rel_breitwigner"
rel_breitwigner = rel_breitwigner_gen(a=0.0, name="rel_breitwigner")

# 复制当前模块的全局变量字典，并将其转换为列表 pairs，其中每个元素是 (名称, 对象) 的键值对
pairs = list(globals().copy().items())

# 调用 get_distribution_names 函数，从 pairs 列表中提取分布类和分布生成函数的名称，
# 存储在 _distn_names 和 _distn_gen_names 变量中，这些名称由 rv_continuous 指定
_distn_names, _distn_gen_names = get_distribution_names(pairs, rv_continuous)

# 定义模块的公开接口列表 __all__，包括分布类的名称、分布生成函数的名称以及 'rv_histogram' 字符串
__all__ = _distn_names + _distn_gen_names + ['rv_histogram']
```