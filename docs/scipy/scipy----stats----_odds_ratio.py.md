# `D:\src\scipysrc\scipy\scipy\stats\_odds_ratio.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算

from scipy.special import ndtri  # 导入 ndtri 函数，用于求正态分布的逆函数
from scipy.optimize import brentq  # 导入 brentq 函数，用于求方程的根
from ._discrete_distns import nchypergeom_fisher  # 从本地模块导入 nchypergeom_fisher 函数
from ._common import ConfidenceInterval  # 从本地模块导入 ConfidenceInterval 类


def _sample_odds_ratio(table):
    """
    Given a table [[a, b], [c, d]], compute a*d/(b*c).

    Return nan if the numerator and denominator are 0.
    Return inf if just the denominator is 0.
    """
    # table must be a 2x2 numpy array.
    if table[1, 0] > 0 and table[0, 1] > 0:
        oddsratio = table[0, 0] * table[1, 1] / (table[1, 0] * table[0, 1])
    elif table[0, 0] == 0 or table[1, 1] == 0:
        oddsratio = np.nan  # 如果分子或分母为0，则返回 NaN
    else:
        oddsratio = np.inf  # 如果分母为0，则返回 Inf
    return oddsratio


def _solve(func):
    """
    Solve func(nc) = 0.  func must be an increasing function.
    """
    # We could just as well call the variable `x` instead of `nc`, but we
    # always call this function with functions for which nc (the noncentrality
    # parameter) is the variable for which we are solving.
    nc = 1.0  # 初始化非中心参数 nc
    value = func(nc)  # 计算给定函数在 nc 处的值
    if value == 0:
        return nc  # 如果 func(nc) 等于 0，则直接返回 nc

    # Multiplicative factor by which to increase or decrease nc when
    # searching for a bracketing interval.
    factor = 2.0  # 初始化搜索间隔的乘法因子

    # Find a bracketing interval.
    if value > 0:
        nc /= factor
        while func(nc) > 0:
            nc /= factor  # 缩小 nc 直到 func(nc) <= 0
        lo = nc  # 设置搜索区间的下限
        hi = factor * nc  # 设置搜索区间的上限
    else:
        nc *= factor
        while func(nc) < 0:
            nc *= factor  # 扩大 nc 直到 func(nc) >= 0
        lo = nc / factor  # 设置搜索区间的下限
        hi = nc  # 设置搜索区间的上限

    # lo and hi bracket the solution for nc.
    nc = brentq(func, lo, hi, xtol=1e-13)  # 使用 brentq 方法找到 func 在 [lo, hi] 区间内的根
    return nc


def _nc_hypergeom_mean_inverse(x, M, n, N):
    """
    For the given noncentral hypergeometric parameters x, M, n, and N
    (table[0,0], total, row 0 sum and column 0 sum, resp., of a 2x2
    contingency table), find the noncentrality parameter of Fisher's
    noncentral hypergeometric distribution whose mean is x.
    """
    nc = _solve(lambda nc: nchypergeom_fisher.mean(M, n, N, nc) - x)
    return nc  # 返回使得 Fisher 非中心超几何分布均值为 x 的非中心参数 nc


def _hypergeom_params_from_table(table):
    # The notation M, n and N is consistent with stats.hypergeom and
    # stats.nchypergeom_fisher.
    x = table[0, 0]  # 表格中的第一行第一列的值
    M = table.sum()  # 表格中所有元素的和
    n = table[0].sum()  # 表格第一行的总和
    N = table[:, 0].sum()  # 表格第一列的总和
    return x, M, n, N  # 返回从表格中获取的参数 x, M, n, N


def _ci_upper(table, alpha):
    """
    Compute the upper end of the confidence interval.
    """
    if _sample_odds_ratio(table) == np.inf:
        return np.inf  # 如果样本的比率为无穷大，则返回无穷大

    x, M, n, N = _hypergeom_params_from_table(table)

    # nchypergeom_fisher.cdf is a decreasing function of nc, so we negate
    # it in the lambda expression.
    nc = _solve(lambda nc: -nchypergeom_fisher.cdf(x, M, n, N, nc) + alpha)
    return nc  # 返回置信区间的上限


def _ci_lower(table, alpha):
    """
    Compute the lower end of the confidence interval.
    """
    if _sample_odds_ratio(table) == 0:
        return 0  # 如果样本的比率为0，则返回0

    x, M, n, N = _hypergeom_params_from_table(table)

    nc = _solve(lambda nc: nchypergeom_fisher.sf(x - 1, M, n, N, nc) - alpha)
    return nc  # 返回置信区间的下限


def _conditional_oddsratio(table):
    """
    """
    Conditional MLE of the odds ratio for the 2x2 contingency table.
    """
    # 从给定的二维列联表中获取超几何分布的参数 x, M, n, N
    x, M, n, N = _hypergeom_params_from_table(table)
    
    # 获取非中心超几何分布的支持范围边界。对于参数 M, n, N 的非中心超几何分布，
    # 支持范围在所有非中心参数值下是相同的，因此我们在这里可以使用 1。
    lo, hi = nchypergeom_fisher.support(M, n, N, 1)
    
    # 检查 x 是否处于支持范围的极端位置。如果是，我们知道 odds ratio 可能是 0 或无穷大。
    if x == lo:
        # x 在支持范围的低端。
        return 0
    if x == hi:
        # x 在支持范围的高端。
        return np.inf
    
    # 计算非中心参数值，即条件最大似然估计的结果
    nc = _nc_hypergeom_mean_inverse(x, M, n, N)
    return nc
def _conditional_oddsratio_ci(table, confidence_level=0.95,
                              alternative='two-sided'):
    """
    Conditional exact confidence interval for the odds ratio.
    """
    # 根据备择假设类型确定 alpha 值
    if alternative == 'two-sided':
        alpha = 0.5*(1 - confidence_level)
        # 计算双侧置信区间的下限
        lower = _ci_lower(table, alpha)
        # 计算双侧置信区间的上限
        upper = _ci_upper(table, alpha)
    elif alternative == 'less':
        # 若备择假设为“小于”，置信区间的下限设为 0
        lower = 0.0
        # 计算小于型置信区间的上限
        upper = _ci_upper(table, 1 - confidence_level)
    else:
        # 若备择假设为“大于”，置信区间的上限设为正无穷
        lower = _ci_lower(table, 1 - confidence_level)
        upper = np.inf

    return lower, upper


def _sample_odds_ratio_ci(table, confidence_level=0.95,
                          alternative='two-sided'):
    # 计算样本的比率比置信区间
    oddsratio = _sample_odds_ratio(table)
    # 计算对数比率比
    log_or = np.log(oddsratio)
    # 计算标准误差
    se = np.sqrt((1/table).sum())
    if alternative == 'less':
        # 若备择假设为“小于”，计算对数比率比的下限
        z = ndtri(confidence_level)
        loglow = -np.inf
        # 计算对数比率比的上限
        loghigh = log_or + z*se
    elif alternative == 'greater':
        # 若备择假设为“大于”，计算对数比率比的上限
        z = ndtri(confidence_level)
        loglow = log_or - z*se
        loghigh = np.inf
    else:
        # 若备择假设为“双侧”，计算对数比率比的置信区间
        z = ndtri(0.5*confidence_level + 0.5)
        loglow = log_or - z*se
        loghigh = log_or + z*se

    return np.exp(loglow), np.exp(loghigh)


class OddsRatioResult:
    """
    Result of `scipy.stats.contingency.odds_ratio`.  See the
    docstring for `odds_ratio` for more details.

    Attributes
    ----------
    statistic : float
        The computed odds ratio.

        * If `kind` is ``'sample'``, this is sample (or unconditional)
          estimate, given by
          ``table[0, 0]*table[1, 1]/(table[0, 1]*table[1, 0])``.
        * If `kind` is ``'conditional'``, this is the conditional
          maximum likelihood estimate for the odds ratio. It is
          the noncentrality parameter of Fisher's noncentral
          hypergeometric distribution with the same hypergeometric
          parameters as `table` and whose mean is ``table[0, 0]``.

    Methods
    -------
    confidence_interval :
        Confidence interval for the odds ratio.
    """

    def __init__(self, _table, _kind, statistic):
        # 初始化 OddsRatioResult 类的实例
        # 目前不需要将 _table 和 _kind 设为公共属性，因为在 `scipy.stats` 结果中这种信息返回的很少
        self._table = _table
        self._kind = _kind
        self.statistic = statistic

    def __repr__(self):
        # 返回 OddsRatioResult 实例的字符串表示
        return f"OddsRatioResult(statistic={self.statistic})"
    # 计算条件比率比的置信区间
    def _conditional_odds_ratio_ci(self, confidence_level=0.95,
                                   alternative='two-sided'):
        """
        Confidence interval for the conditional odds ratio.
        """

        # 获取交叉表
        table = self._table
        
        # 检查交叉表中是否有任何行或列的和为0
        if 0 in table.sum(axis=0) or 0 in table.sum(axis=1):
            # 如果某行或列的和为0，则p值为1，比率比为NaN，置信区间为(0, inf)
            ci = (0, np.inf)
        else:
            # 调用计算条件比率比置信区间的函数
            ci = _conditional_oddsratio_ci(table,
                                           confidence_level=confidence_level,
                                           alternative=alternative)
        
        # 返回置信区间对象，包括低和高端点
        return ConfidenceInterval(low=ci[0], high=ci[1])

    # 计算样本比率比的置信区间
    def _sample_odds_ratio_ci(self, confidence_level=0.95,
                              alternative='two-sided'):
        """
        Confidence interval for the sample odds ratio.
        """
        
        # 检查置信水平是否在有效范围内
        if confidence_level < 0 or confidence_level > 1:
            raise ValueError('confidence_level must be between 0 and 1')

        # 获取交叉表
        table = self._table
        
        # 检查交叉表中是否有任何行或列的和为0
        if 0 in table.sum(axis=0) or 0 in table.sum(axis=1):
            # 如果某行或列的和为0，则p值为1，比率比为NaN，置信区间为(0, inf)
            ci = (0, np.inf)
        else:
            # 调用计算样本比率比置信区间的函数
            ci = _sample_odds_ratio_ci(table,
                                       confidence_level=confidence_level,
                                       alternative=alternative)
        
        # 返回置信区间对象，包括低和高端点
        return ConfidenceInterval(low=ci[0], high=ci[1])
# 定义计算二维列联表中的几率比的函数
def odds_ratio(table, *, kind='conditional'):
    """
    Compute the odds ratio for a 2x2 contingency table.

    Parameters
    ----------
    table : array_like of ints
        A 2x2 contingency table.  Elements must be non-negative integers.
    kind : str, optional
        Which kind of odds ratio to compute, either the sample
        odds ratio (``kind='sample'``) or the conditional odds ratio
        (``kind='conditional'``).  Default is ``'conditional'``.

    Returns
    -------
    result : `~scipy.stats._result_classes.OddsRatioResult` instance
        The returned object has two computed attributes:

        statistic : float
            * If `kind` is ``'sample'``, this is sample (or unconditional)
              estimate, given by
              ``table[0, 0]*table[1, 1]/(table[0, 1]*table[1, 0])``.
            * If `kind` is ``'conditional'``, this is the conditional
              maximum likelihood estimate for the odds ratio. It is
              the noncentrality parameter of Fisher's noncentral
              hypergeometric distribution with the same hypergeometric
              parameters as `table` and whose mean is ``table[0, 0]``.

        The object has the method `confidence_interval` that computes
        the confidence interval of the odds ratio.

    See Also
    --------
    scipy.stats.fisher_exact
    relative_risk

    Notes
    -----
    The conditional odds ratio was discussed by Fisher (see "Example 1"
    of [1]_).  Texts that cover the odds ratio include [2]_ and [3]_.

    .. versionadded:: 1.10.0

    References
    ----------
    .. [1] R. A. Fisher (1935), The logic of inductive inference,
           Journal of the Royal Statistical Society, Vol. 98, No. 1,
           pp. 39-82.
    .. [2] Breslow NE, Day NE (1980). Statistical methods in cancer research.
           Volume I - The analysis of case-control studies. IARC Sci Publ.
           (32):5-338. PMID: 7216345. (See section 4.2.)
    .. [3] H. Sahai and A. Khurshid (1996), Statistics in Epidemiology:
           Methods, Techniques, and Applications, CRC Press LLC, Boca
           Raton, Florida.
    .. [4] Berger, Jeffrey S. et al. "Aspirin for the Primary Prevention of
           Cardiovascular Events in Women and Men: A Sex-Specific
           Meta-analysis of Randomized Controlled Trials."
           JAMA, 295(3):306-313, :doi:`10.1001/jama.295.3.306`, 2006.

    Examples
    --------
    In epidemiology, individuals are classified as "exposed" or
    "unexposed" to some factor or treatment. If the occurrence of some
    illness is under study, those who have the illness are often
    classified as "cases", and those without it are "noncases".  The
    counts of the occurrences of these classes gives a contingency
    table::

                    exposed    unexposed
        cases          a           b
        noncases       c           d

    The sample odds ratio may be written ``(a/c) / (b/d)``.  ``a/c`` can
    be interpreted as the odds of being exposed among cases relative to
    the odds of being exposed among noncases.  The conditional odds ratio
    is computed as the maximum likelihood estimate of the odds ratio
    under the assumptions of Fisher's exact test.

    """
    # 根据参数 kind 确定计算哪种类型的几率比
    if kind == 'sample':
        # 计算样本几率比
        statistic = table[0, 0] * table[1, 1] / (table[0, 1] * table[1, 0])
    elif kind == 'conditional':
        # 计算条件几率比，这是 Fisher 检验的条件最大似然估计
        statistic = table[0, 0]
    else:
        raise ValueError("Unknown kind of odds ratio. Expected 'sample' or 'conditional'.")
    
    # 返回一个包含计算结果的 OddsRatioResult 实例
    return scipy.stats._result_classes.OddsRatioResult(statistic)
    """
    Check if `kind` parameter is valid, raise an error if not.

    Parameters:
    kind : str
        Type of odds ratio calculation, must be either 'conditional' or 'sample'.

    Raises:
    ValueError: If `kind` is not 'conditional' or 'sample'.
    """
    if kind not in ['conditional', 'sample']:
        raise ValueError("`kind` must be 'conditional' or 'sample'.")

    """
    Convert `table` to a NumPy array and perform validation checks.

    Parameters:
    table : list of list of int
        2x2 contingency table representing counts of cases.

    Raises:
    ValueError: If `table` is not of shape (2, 2), or contains non-integer values,
                or contains negative values.
    """
    c = np.asarray(table)

    if c.shape != (2, 2):
        raise ValueError(f"Invalid shape {c.shape}. The input `table` must be "
                         "of shape (2, 2).")

    if not np.issubdtype(c.dtype, np.integer):
        raise ValueError("`table` must be an array of integers, but got "
                         f"type {c.dtype}")
    c = c.astype(np.int64)

    if np.any(c < 0):
        raise ValueError("All values in `table` must be nonnegative.")
    # 检查是否存在任何行或列的总和为零
    if 0 in c.sum(axis=0) or 0 in c.sum(axis=1):
        # 如果某行或某列的总和为零，则返回结果包含 NaN 的概率值和 NaN 的比率
        result = OddsRatioResult(_table=c, _kind=kind, statistic=np.nan)
        return result

    # 根据不同的统计类型计算不同的比率
    if kind == 'sample':
        # 如果统计类型为样本，则计算样本比率
        oddsratio = _sample_odds_ratio(c)
    else:  # kind is 'conditional'
        # 如果统计类型为条件，则计算条件比率
        oddsratio = _conditional_oddsratio(c)

    # 返回包含计算出的比率的结果对象
    result = OddsRatioResult(_table=c, _kind=kind, statistic=oddsratio)
    return result
```