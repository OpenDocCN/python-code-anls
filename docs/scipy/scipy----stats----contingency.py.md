# `D:\src\scipysrc\scipy\scipy\stats\contingency.py`

```
# 导入reduce函数，用于对序列进行累积计算；导入math模块，提供数学函数支持；导入numpy库，并简化为np，用于科学计算
from functools import reduce
import math
import numpy as np

# 导入_power_divergence函数，用于执行功率分歧测试；导入_relative_risk模块中的relative_risk函数，用于计算相对风险；导入_crosstab模块中的crosstab函数，用于生成交叉表；导入_odds_ratio模块中的odds_ratio函数，用于计算赔率比；导入_scipy._lib._bunch模块中的_make_tuple_bunch函数，用于创建元组捆绑对象
from ._stats_py import power_divergence
from ._relative_risk import relative_risk
from ._crosstab import crosstab
from ._odds_ratio import odds_ratio
from scipy._lib._bunch import _make_tuple_bunch

# 声明模块中公开的函数和类名称列表，用于控制导出的符号
__all__ = ['margins', 'expected_freq', 'chi2_contingency', 'crosstab',
           'association', 'relative_risk', 'odds_ratio']

# 定义函数margins，用于计算数组a的边际总和
def margins(a):
    """Return a list of the marginal sums of the array `a`.

    Parameters
    ----------
    a : ndarray
        The array for which to compute the marginal sums.

    Returns
    -------
    margsums : list of ndarrays
        A list of length `a.ndim`.  `margsums[k]` is the result
        of summing `a` over all axes except `k`; it has the same
        number of dimensions as `a`, but the length of each axis
        except axis `k` will be 1.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats.contingency import margins

    >>> a = np.arange(12).reshape(2, 6)
    >>> a
    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11]])
    >>> m0, m1 = margins(a)
    >>> m0
    array([[15],
           [51]])
    >>> m1
    array([[ 6,  8, 10, 12, 14, 16]])

    >>> b = np.arange(24).reshape(2,3,4)
    >>> m0, m1, m2 = margins(b)
    >>> m0
    array([[[ 66]],
           [[210]]])
    >>> m1
    array([[[ 60],
            [ 92],
            [124]]])
    >>> m2
    array([[[60, 66, 72, 78]]])
    """
    margsums = []
    ranged = list(range(a.ndim))
    # 遍历数组的每个维度
    for k in ranged:
        # 对数组a按照除了第k个轴之外的其它轴进行求和
        marg = np.apply_over_axes(np.sum, a, [j for j in ranged if j != k])
        margsums.append(marg)
    return margsums


def expected_freq(observed):
    """
    Compute the expected frequencies from a contingency table.

    Given an n-dimensional contingency table of observed frequencies,
    compute the expected frequencies for the table based on the marginal
    sums under the assumption that the groups associated with each
    dimension are independent.

    Parameters
    ----------
    observed : array_like
        The table of observed frequencies.  (While this function can handle
        a 1-D array, that case is trivial.  Generally `observed` is at
        least 2-D.)

    Returns
    -------
    expected : ndarray of float64
        The expected frequencies, based on the marginal sums of the table.
        Same shape as `observed`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats.contingency import expected_freq
    >>> observed = np.array([[10, 10, 20],[20, 20, 20]])
    >>> expected_freq(observed)
    """
    array([[ 12.,  12.,  16.],
           [ 18.,  18.,  24.]])

    """
    # 典型情况下，`observed` 是一个整数数组。如果 `observed` 的维度很多或者包含大数值，
    # 一些计算可能会溢出，因此我们首先将其转换为浮点数类型。
    observed = np.asarray(observed, dtype=np.float64)

    # 创建边际和的列表。
    margsums = margins(observed)

    # 创建期望频率的数组。通过 apply_over_axes() 返回的边际和的形状正好符合
    # 我们在接下来的乘积中进行广播运算的需要。
    d = observed.ndim
    expected = reduce(np.multiply, margsums) / observed.sum() ** (d - 1)
    return expected
# 创建一个名为 Chi2ContingencyResult 的命名元组，表示卡方检验的结果
Chi2ContingencyResult = _make_tuple_bunch(
    'Chi2ContingencyResult',  # 命名元组的名称
    ['statistic', 'pvalue', 'dof', 'expected_freq'],  # 元组包含的字段名
    []  # 初始值为空列表
)


def chi2_contingency(observed, correction=True, lambda_=None):
    """Chi-square test of independence of variables in a contingency table.

    This function computes the chi-square statistic and p-value for the
    hypothesis test of independence of the observed frequencies in the
    contingency table [1]_ `observed`.  The expected frequencies are computed
    based on the marginal sums under the assumption of independence; see
    `scipy.stats.contingency.expected_freq`.  The number of degrees of
    freedom is (expressed using numpy functions and attributes)::

        dof = observed.size - sum(observed.shape) + observed.ndim - 1

    Parameters
    ----------
    observed : array_like
        The contingency table. The table contains the observed frequencies
        (i.e. number of occurrences) in each category.  In the two-dimensional
        case, the table is often described as an "R x C table".
    correction : bool, optional
        If True, *and* the degrees of freedom is 1, apply Yates' correction
        for continuity.  The effect of the correction is to adjust each
        observed value by 0.5 towards the corresponding expected value.
    lambda_ : float or str, optional
        By default, the statistic computed in this test is Pearson's
        chi-squared statistic [2]_.  `lambda_` allows a statistic from the
        Cressie-Read power divergence family [3]_ to be used instead.  See
        `scipy.stats.power_divergence` for details.

    Returns
    -------
    res : Chi2ContingencyResult
        An object containing attributes:

        statistic : float
            The test statistic.
        pvalue : float
            The p-value of the test.
        dof : int
            The degrees of freedom.
        expected_freq : ndarray, same shape as `observed`
            The expected frequencies, based on the marginal sums of the table.

    See Also
    --------
    scipy.stats.contingency.expected_freq
    scipy.stats.fisher_exact
    scipy.stats.chisquare
    scipy.stats.power_divergence
    scipy.stats.barnard_exact
    scipy.stats.boschloo_exact

    Notes
    -----
    An often quoted guideline for the validity of this calculation is that
    the test should be used only if the observed and expected frequencies
    in each cell are at least 5.

    This is a test for the independence of different categories of a
    population. The test is only meaningful when the dimension of
    `observed` is two or more.  Applying the test to a one-dimensional
    table will always result in `expected` equal to `observed` and a
    chi-square statistic equal to 0.

    This function does not handle masked arrays, because the calculation
    does not make sense with missing values.

    Like `scipy.stats.chisquare`, this function computes a chi-square
    """
    statistic; the convenience this function provides is to figure out the
    expected frequencies and degrees of freedom from the given contingency
    table. If these were already known, and if the Yates' correction was not
    required, one could use `scipy.stats.chisquare`.  That is, if one calls::

        res = chi2_contingency(obs, correction=False)

    then the following is true::

        (res.statistic, res.pvalue) == stats.chisquare(obs.ravel(),
                                                       f_exp=ex.ravel(),
                                                       ddof=obs.size - 1 - dof)

    The `lambda_` argument was added in version 0.13.0 of scipy.

    References
    ----------
    .. [1] "Contingency table",
           https://en.wikipedia.org/wiki/Contingency_table
    .. [2] "Pearson's chi-squared test",
           https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test
    .. [3] Cressie, N. and Read, T. R. C., "Multinomial Goodness-of-Fit
           Tests", J. Royal Stat. Soc. Series B, Vol. 46, No. 3 (1984),
           pp. 440-464.
    .. [4] Berger, Jeffrey S. et al. "Aspirin for the Primary Prevention of
           Cardiovascular Events in Women and Men: A Sex-Specific
           Meta-analysis of Randomized Controlled Trials."
           JAMA, 295(3):306-313, :doi:`10.1001/jama.295.3.306`, 2006.

    Examples
    --------
    In [4]_, the use of aspirin to prevent cardiovascular events in women
    and men was investigated. The study notably concluded:

        ...aspirin therapy reduced the risk of a composite of
        cardiovascular events due to its effect on reducing the risk of
        ischemic stroke in women [...]

    The article lists studies of various cardiovascular events. Let's
    focus on the ischemic stoke in women.

    The following table summarizes the results of the experiment in which
    participants took aspirin or a placebo on a regular basis for several
    years. Cases of ischemic stroke were recorded::

                          Aspirin   Control/Placebo
        Ischemic stroke     176           230
        No stroke         21035         21018

    Is there evidence that the aspirin reduces the risk of ischemic stroke?
    We begin by formulating a null hypothesis :math:`H_0`:

        The effect of aspirin is equivalent to that of placebo.

    Let's assess the plausibility of this hypothesis with
    a chi-square test.

    >>> import numpy as np
    >>> from scipy.stats import chi2_contingency
    >>> table = np.array([[176, 230], [21035, 21018]])
    >>> res = chi2_contingency(table)
    使用 `chi2_contingency` 函数计算卡方统计量和 p 值
    >>> res.statistic
    返回卡方统计量的值
    >>> res.pvalue
    返回卡方检验的 p 值，用于判断是否拒绝原假设
    Using a significance level of 5%, we would reject the null hypothesis in
    favor of the alternative hypothesis: "the effect of aspirin
    is not equivalent to the effect of placebo".
    Because `scipy.stats.contingency.chi2_contingency` performs a two-sided
    observed = np.asarray(observed)
    # 将输入的观测数据转换为 NumPy 数组形式

    if np.any(observed < 0):
        # 如果观测数据中存在小于零的值，抛出数值错误异常
        raise ValueError("All values in `observed` must be nonnegative.")

    if observed.size == 0:
        # 如果观测数据为空（大小为零），抛出数值错误异常
        raise ValueError("No data; `observed` has size 0.")

    expected = expected_freq(observed)
    # 计算期望的频率表，基于观测数据

    if np.any(expected == 0):
        # 如果期望的频率表中存在零元素，抛出数值错误异常，并指出具体位置
        zeropos = list(zip(*np.nonzero(expected == 0)))[0]
        raise ValueError("The internally computed table of expected "
                         f"frequencies has a zero element at {zeropos}.")

    # 计算自由度
    dof = expected.size - sum(expected.shape) + expected.ndim - 1

    if dof == 0:
        # 特殊情况：自由度为零时，通常是因为观测数据为一维，此时卡方值为零，p 值为 1.0
        chi2 = 0.0
        p = 1.0
    else:
        if dof == 1 and correction:
            # 如果自由度为 1 并且启用了 Yates 修正，则根据 Yates 修正调整观测数据
            diff = expected - observed
            direction = np.sign(diff)
            magnitude = np.minimum(0.5, np.abs(diff))
            observed = observed + magnitude * direction

        # 计算卡方值和 p 值，支持使用对数似然比（G-测试）
        chi2, p = power_divergence(observed, expected,
                                   ddof=observed.size - 1 - dof, axis=None,
                                   lambda_=lambda_)

    # 返回卡方检验结果对象，包括卡方值、p 值、自由度和期望频率表
    return Chi2ContingencyResult(chi2, p, dof, expected)
# 定义一个计算两个名义变量之间关联度的函数
def association(observed, method="cramer", correction=False, lambda_=None):
    """Calculates degree of association between two nominal variables.

    The function provides the option for computing one of three measures of
    association between two nominal variables from the data given in a 2d
    contingency table: Tschuprow's T, Pearson's Contingency Coefficient
    and Cramer's V.

    Parameters
    ----------
    observed : array-like
        The array of observed values
    method : {"cramer", "tschuprow", "pearson"} (default = "cramer")
        The association test statistic.
    correction : bool, optional
        Inherited from `scipy.stats.contingency.chi2_contingency()`
    lambda_ : float or str, optional
        Inherited from `scipy.stats.contingency.chi2_contingency()`

    Returns
    -------
    statistic : float
        Value of the test statistic

    Notes
    -----
    Cramer's V, Tschuprow's T and Pearson's Contingency Coefficient, all
    measure the degree to which two nominal or ordinal variables are related,
    or the level of their association. This differs from correlation, although
    many often mistakenly consider them equivalent. Correlation measures in
    what way two variables are related, whereas, association measures how
    related the variables are. As such, association does not subsume
    independent variables, and is rather a test of independence. A value of
    1.0 indicates perfect association, and 0.0 means the variables have no
    association.

    Both the Cramer's V and Tschuprow's T are extensions of the phi
    coefficient.  Moreover, due to the close relationship between the
    Cramer's V and Tschuprow's T the returned values can often be similar
    or even equivalent.  They are likely to diverge more as the array shape
    diverges from a 2x2.

    References
    ----------
    .. [1] "Tschuprow's T",
           https://en.wikipedia.org/wiki/Tschuprow's_T
    .. [2] Tschuprow, A. A. (1939)
           Principles of the Mathematical Theory of Correlation;
           translated by M. Kantorowitsch. W. Hodge & Co.
    .. [3] "Cramer's V", https://en.wikipedia.org/wiki/Cramer's_V
    .. [4] "Nominal Association: Phi and Cramer's V",
           http://www.people.vcu.edu/~pdattalo/702SuppRead/MeasAssoc/NominalAssoc.html
    .. [5] Gingrich, Paul, "Association Between Variables",
           http://uregina.ca/~gingrich/ch11a.pdf

    Examples
    --------
    An example with a 4x2 contingency table:

    >>> import numpy as np
    >>> from scipy.stats.contingency import association
    >>> obs4x2 = np.array([[100, 150], [203, 322], [420, 700], [320, 210]])

    Pearson's contingency coefficient

    >>> association(obs4x2, method="pearson")
    0.18303298140595667

    Cramer's V

    >>> association(obs4x2, method="cramer")
    0.18617813077483678

    Tschuprow's T

    >>> association(obs4x2, method="tschuprow")
    0.14146478765062995
    """
    # 将输入的观测数据转换为 NumPy 数组
    arr = np.asarray(observed)
    # 检查数组元素类型是否为整数，若不是则抛出数值错误异常
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("`observed` must be an integer array.")

    # 检查数组维度是否为二维，若不是则抛出数值错误异常
    if len(arr.shape) != 2:
        raise ValueError("method only accepts 2d arrays")

    # 计算列联表的卡方统计量，通过参数传递矫正方式和lambda值
    chi2_stat = chi2_contingency(arr, correction=correction,
                                 lambda_=lambda_)

    # 计算 phi-squared 统计量（卡方统计量除以总计数）
    phi2 = chi2_stat.statistic / arr.sum()

    # 获取数组的行数和列数
    n_rows, n_cols = arr.shape

    # 根据指定的方法计算相关系数的值
    if method == "cramer":
        # Cramer's V 方法计算相关系数
        value = phi2 / min(n_cols - 1, n_rows - 1)
    elif method == "tschuprow":
        # Tschuprow's T 方法计算相关系数
        value = phi2 / math.sqrt((n_rows - 1) * (n_cols - 1))
    elif method == 'pearson':
        # Pearson 相关系数方法计算相关系数
        value = phi2 / (1 + phi2)
    else:
        # 若方法参数不在预期值中，抛出数值错误异常
        raise ValueError("Invalid argument value: 'method' argument must "
                         "be 'cramer', 'tschuprow', or 'pearson'")

    # 返回相关系数的平方根作为最终结果
    return math.sqrt(value)
```