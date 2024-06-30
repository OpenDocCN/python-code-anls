# `D:\src\scipysrc\scipy\scipy\stats\_binomtest.py`

```
# 从 math 模块导入 sqrt 函数，用于计算平方根
from math import sqrt
# 导入 numpy 库并使用 np 别名
import numpy as np
# 从 scipy._lib._util 模块导入 _validate_int 函数，用于验证整数
from scipy._lib._util import _validate_int
# 从 scipy.optimize 模块导入 brentq 函数，用于求解方程
from scipy.optimize import brentq
# 从 scipy.special 模块导入 ndtri 函数，用于求正态分布的逆累积分布函数
from scipy.special import ndtri
# 从当前包的 _discrete_distns 模块导入 binom 函数
from ._discrete_distns import binom
# 从当前包的 _common 模块导入 ConfidenceInterval 类
from ._common import ConfidenceInterval

# 定义 BinomTestResult 类，用于保存二项分布检验的结果
class BinomTestResult:
    """
    Result of `scipy.stats.binomtest`.

    Attributes
    ----------
    k : int
        成功的次数（从 `binomtest` 输入复制而来）。
    n : int
        试验的总次数（从 `binomtest` 输入复制而来）。
    alternative : str
        指示在 `binomtest` 输入中指定的备择假设。可能是 ``'two-sided'``、``'greater'`` 或 ``'less'`` 之一。
    statistic: float
        成功比例的估计值。
    pvalue : float
        假设检验的 p 值。

    """
    
    # 初始化方法，用于设置类的属性
    def __init__(self, k, n, alternative, statistic, pvalue):
        self.k = k  # 设置成功次数 k 属性
        self.n = n  # 设置试验总次数 n 属性
        self.alternative = alternative  # 设置备择假设 alternative 属性
        self.statistic = statistic  # 设置成功比例估计值 statistic 属性
        self.pvalue = pvalue  # 设置假设检验 p 值属性
        
        # 为了向后兼容添加别名
        self.proportion_estimate = statistic  # 设置比例估计别名属性

    # 返回对象的字符串表示形式
    def __repr__(self):
        # 构建对象的字符串表示形式
        s = ("BinomTestResult("
             f"k={self.k}, "
             f"n={self.n}, "
             f"alternative={self.alternative!r}, "
             f"statistic={self.statistic}, "
             f"pvalue={self.pvalue})")
        return s  # 返回构建的字符串
    def proportion_ci(self, confidence_level=0.95, method='exact'):
        """
        Compute the confidence interval for ``statistic``.

        Parameters
        ----------
        confidence_level : float, optional
            Confidence level for the computed confidence interval
            of the estimated proportion. Default is 0.95.
        method : {'exact', 'wilson', 'wilsoncc'}, optional
            Selects the method used to compute the confidence interval
            for the estimate of the proportion:

            'exact' :
                Use the Clopper-Pearson exact method [1]_.
            'wilson' :
                Wilson's method, without continuity correction ([2]_, [3]_).
            'wilsoncc' :
                Wilson's method, with continuity correction ([2]_, [3]_).

            Default is ``'exact'``.

        Returns
        -------
        ci : ``ConfidenceInterval`` object
            The object has attributes ``low`` and ``high`` that hold the
            lower and upper bounds of the confidence interval.

        References
        ----------
        .. [1] C. J. Clopper and E. S. Pearson, The use of confidence or
               fiducial limits illustrated in the case of the binomial,
               Biometrika, Vol. 26, No. 4, pp 404-413 (Dec. 1934).
        .. [2] E. B. Wilson, Probable inference, the law of succession, and
               statistical inference, J. Amer. Stat. Assoc., 22, pp 209-212
               (1927).
        .. [3] Robert G. Newcombe, Two-sided confidence intervals for the
               single proportion: comparison of seven methods, Statistics
               in Medicine, 17, pp 857-872 (1998).

        Examples
        --------
        >>> from scipy.stats import binomtest
        >>> result = binomtest(k=7, n=50, p=0.1)
        >>> result.statistic
        0.14
        >>> result.proportion_ci()
        ConfidenceInterval(low=0.05819170033997342, high=0.26739600249700846)
        """
        # 检查给定的方法是否有效，如果无效则抛出错误
        if method not in ('exact', 'wilson', 'wilsoncc'):
            raise ValueError(f"method ('{method}') must be one of 'exact', "
                             "'wilson' or 'wilsoncc'.")
        # 检查置信水平是否在合理范围内，如果不在则抛出错误
        if not (0 <= confidence_level <= 1):
            raise ValueError(f'confidence_level ({confidence_level}) must be in '
                             'the interval [0, 1].')
        
        # 根据选择的方法计算置信区间的下界和上界
        if method == 'exact':
            # 使用精确法计算二项分布的置信区间
            low, high = _binom_exact_conf_int(self.k, self.n,
                                              confidence_level,
                                              self.alternative)
        else:
            # 使用Wilson方法计算二项分布的置信区间，可能包含连续性校正
            low, high = _binom_wilson_conf_int(self.k, self.n,
                                               confidence_level,
                                               self.alternative,
                                               correction=method == 'wilsoncc')
        
        # 返回置信区间对象，包含计算得到的下界和上界
        return ConfidenceInterval(low=low, high=high)
# 使用 Brent 方法寻找函数 `func` 在区间 [0, 1] 内的根
def _findp(func):
    try:
        p = brentq(func, 0, 1)
    except RuntimeError:
        # 如果数值解法在计算置信区间时未收敛，抛出 RuntimeError
        raise RuntimeError('numerical solver failed to converge when '
                           'computing the confidence limits') from None
    except ValueError as exc:
        # 如果 brentq 函数引发 ValueError，将其传递，并报告给 SciPy 开发者
        raise ValueError('brentq raised a ValueError; report this to the '
                         'SciPy developers') from exc
    # 返回找到的根 p
    return p


# 计算二项检验的估计值和置信区间
def _binom_exact_conf_int(k, n, confidence_level, alternative):
    """
    Compute the estimate and confidence interval for the binomial test.

    Returns proportion, prop_low, prop_high
    """
    if alternative == 'two-sided':
        # 计算双侧检验的 alpha 值
        alpha = (1 - confidence_level) / 2
        # 计算下界 plow
        if k == 0:
            plow = 0.0
        else:
            # 使用 `_findp` 函数找到使得累积分布函数与 alpha 最接近的概率值
            plow = _findp(lambda p: binom.sf(k-1, n, p) - alpha)
        # 计算上界 phigh
        if k == n:
            phigh = 1.0
        else:
            # 使用 `_findp` 函数找到使得累积分布函数与 alpha 最接近的概率值
            phigh = _findp(lambda p: binom.cdf(k, n, p) - alpha)
    elif alternative == 'less':
        # 计算单侧检验的 alpha 值
        alpha = 1 - confidence_level
        plow = 0.0
        # 计算上界 phigh
        if k == n:
            phigh = 1.0
        else:
            # 使用 `_findp` 函数找到使得累积分布函数与 alpha 最接近的概率值
            phigh = _findp(lambda p: binom.cdf(k, n, p) - alpha)
    elif alternative == 'greater':
        # 计算单侧检验的 alpha 值
        alpha = 1 - confidence_level
        # 计算下界 plow
        if k == 0:
            plow = 0.0
        else:
            # 使用 `_findp` 函数找到使得累积分布函数与 alpha 最接近的概率值
            plow = _findp(lambda p: binom.sf(k-1, n, p) - alpha)
        phigh = 1.0
    # 返回下界和上界
    return plow, phigh


# 计算 Wilson 法则下的二项分布置信区间
def _binom_wilson_conf_int(k, n, confidence_level, alternative, correction):
    # 假设参数已经通过验证，`alternative` 必须是 'two-sided', 'less' 或 'greater' 之一
    p = k / n
    if alternative == 'two-sided':
        # 计算双侧检验的 Z 分值
        z = ndtri(0.5 + 0.5*confidence_level)
    else:
        # 计算单侧检验的 Z 分值
        z = ndtri(confidence_level)

    # 根据 Newcombe (1998) 中的公式计算置信区间
    denom = 2*(n + z**2)
    center = (2*n*p + z**2)/denom
    q = 1 - p
    if correction:
        # 如果需要校正
        if alternative == 'less' or k == 0:
            lo = 0.0
        else:
            # 计算下界
            dlo = (1 + z*sqrt(z**2 - 2 - 1/n + 4*p*(n*q + 1))) / denom
            lo = center - dlo
        if alternative == 'greater' or k == n:
            hi = 1.0
        else:
            # 计算上界
            dhi = (1 + z*sqrt(z**2 + 2 - 1/n + 4*p*(n*q - 1))) / denom
            hi = center + dhi
    else:
        # 如果不需要校正
        delta = z/denom * sqrt(4*n*p*q + z**2)
        # 计算下界
        if alternative == 'less' or k == 0:
            lo = 0.0
        else:
            lo = center - delta
        # 计算上界
        if alternative == 'greater' or k == n:
            hi = 1.0
        else:
            hi = center + delta

    # 返回下界和上界
    return lo, hi


# 执行二项检验，检验成功概率是否为 `p`
def binomtest(k, n, p=0.5, alternative='two-sided'):
    """
    Perform a test that the probability of success is p.

    The binomial test [1]_ is a test of the null hypothesis that the
    probability of success in a Bernoulli experiment is `p`.

    Details of the test can be found in many texts on statistics, such
    as section 24.5 of [2]_.

    Parameters
    ```
    # Validate and ensure k is a non-negative integer
    k = _validate_int(k, 'k', minimum=0)
    # Validate and ensure n is a positive integer
    n = _validate_int(n, 'n', minimum=1)
    
    # Check if the number of successes (k) exceeds the number of trials (n), which is invalid
    if k > n:
        raise ValueError(f'k ({k}) must not be greater than n ({n}).')

    # Check if the hypothesized probability of success (p) is within the valid range [0, 1]
    if not (0 <= p <= 1):
        raise ValueError(f"p ({p}) must be in range [0,1]")

    # Check if the alternative hypothesis is correctly specified ('two-sided', 'less', 'greater')
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError(f"alternative ('{alternative}') not recognized; \n"
                         "must be 'two-sided', 'less' or 'greater'")
    # 如果 alternative 参数为 'less'，则计算二项分布累积分布函数的值
    pval = binom.cdf(k, n, p)
elif alternative == 'greater':
    # 如果 alternative 参数为 'greater'，则计算二项分布的生存函数的值
    pval = binom.sf(k-1, n, p)
else:
    # 如果 alternative 参数为 'two-sided'，则进行以下处理
    # 计算二项分布的概率质量函数的值
    d = binom.pmf(k, n, p)
    # 相对误差
    rerr = 1 + 1e-7
    if k == p * n:
        # 如果 k 等于 p*n，特殊情况的快捷处理，返回概率值为 1.0
        pval = 1.
    elif k < p * n:
        # 在 k 小于 p*n 的情况下，执行二分查找，找到符合条件的边界值
        ix = _binary_search_for_binom_tst(lambda x1: -binom.pmf(x1, n, p),
                                          -d*rerr, np.ceil(p * n), n)
        # y 是在模式和 n 之间的术语数，这些术语小于等于 d*rerr
        # ix 给出了第一个满足 a(ix) <= d*rerr < a(ix-1) 条件的位置
        # 如果第一个相等条件不成立，y=n-ix。否则，需要包括 ix，因为相等条件成立。
        # 由于 rerr 的存在，相等条件非常罕见。
        y = n - ix + int(d*rerr == binom.pmf(ix, n, p))
        # 计算右侧尾部和左侧尾部的累积分布函数，得到双侧检验的 p 值
        pval = binom.cdf(k, n, p) + binom.sf(n - y, n, p)
    else:
        # 在 k 大于 p*n 的情况下，执行二分查找，找到符合条件的边界值
        ix = _binary_search_for_binom_tst(lambda x1: binom.pmf(x1, n, p),
                                          d*rerr, 0, np.floor(p * n))
        # y 是在 0 和模式之间的术语数，这些术语小于等于 d*rerr
        # 需要添加 1 来解释 0 索引的情况
        y = ix + 1
        # 计算左侧尾部和右侧尾部的累积分布函数，得到双侧检验的 p 值
        pval = binom.cdf(y-1, n, p) + binom.sf(k-1, n, p)

    # 确保 p 值不超过 1.0
    pval = min(1.0, pval)

# 构造二项分布检验的结果对象，并返回
result = BinomTestResult(k=k, n=n, alternative=alternative,
                         statistic=k/n, pvalue=pval)
return result
def _binary_search_for_binom_tst(a, d, lo, hi):
    """
    Conducts an implicit binary search on a function specified by `a`.
    
    Meant to be used on the binomial PMF for the case of two-sided tests
    to obtain the value on the other side of the mode where the tail
    probability should be computed. The values on either side of
    the mode are always in order, meaning binary search is applicable.
    
    Parameters
    ----------
    a : callable
      The function over which to perform binary search. Its values
      for inputs lo and hi should be in ascending order.
    d : float
      The value to search.
    lo : int
      The lower end of range to search.
    hi : int
      The higher end of the range to search.
    
    Returns
    -------
    int
      The index, i between lo and hi
      such that a(i)<=d<a(i+1)
    """
    # 当下限 lo 小于上限 hi 时执行循环
    while lo < hi:
        # 计算中间点 mid
        mid = lo + (hi - lo) // 2
        # 获取中间点 mid 处的函数值 midval
        midval = a(mid)
        # 如果 midval 小于目标值 d，则将搜索范围缩小到 mid 右侧
        if midval < d:
            lo = mid + 1
        # 如果 midval 大于目标值 d，则将搜索范围缩小到 mid 左侧
        elif midval > d:
            hi = mid - 1
        # 如果 midval 等于目标值 d，则直接返回 mid
        else:
            return mid
    # 如果退出循环时 a(lo) 小于等于目标值 d，则返回 lo
    if a(lo) <= d:
        return lo
    # 否则返回 lo-1
    else:
        return lo - 1
```