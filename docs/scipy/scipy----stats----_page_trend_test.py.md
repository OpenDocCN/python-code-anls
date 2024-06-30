# `D:\src\scipysrc\scipy\scipy\stats\_page_trend_test.py`

```
from itertools import permutations  # 导入 permutations 函数，用于生成可迭代对象的所有排列
import numpy as np  # 导入 numpy 库，并用 np 别名引用
import math  # 导入 math 库，提供数学函数
from ._continuous_distns import norm  # 从 _continuous_distns 模块导入 norm 函数
import scipy.stats  # 导入 scipy.stats 库，用于统计函数
from dataclasses import dataclass  # 导入 dataclass 类，用于定义数据类


@dataclass
class PageTrendTestResult:
    statistic: float  # Page's Test 统计量，浮点型
    pvalue: float  # Page's Test 的 p 值，浮点型
    method: str  # 执行 Page's Test 的方法，字符串类型


def page_trend_test(data, ranked=False, predicted_ranks=None, method='auto'):
    r"""
    Perform Page's Test, a measure of trend in observations between treatments.

    Page's Test (also known as Page's :math:`L` test) is useful when:

    * there are :math:`n \geq 3` treatments,
    * :math:`m \geq 2` subjects are observed for each treatment, and
    * the observations are hypothesized to have a particular order.

    Specifically, the test considers the null hypothesis that

    .. math::

        m_1 = m_2 = m_3 \cdots = m_n,

    where :math:`m_j` is the mean of the observed quantity under treatment
    :math:`j`, against the alternative hypothesis that

    .. math::

        m_1 \leq m_2 \leq m_3 \leq \cdots \leq m_n,

    where at least one inequality is strict.

    As noted by [4]_, Page's :math:`L` test has greater statistical power than
    the Friedman test against the alternative that there is a difference in
    trend, as Friedman's test only considers a difference in the means of the
    observations without considering their order. Whereas Spearman :math:`\rho`
    considers the correlation between the ranked observations of two variables
    (e.g. the airspeed velocity of a swallow vs. the weight of the coconut it
    carries), Page's :math:`L` is concerned with a trend in an observation
    (e.g. the airspeed velocity of a swallow) across several distinct
    treatments (e.g. carrying each of five coconuts of different weight) even
    as the observation is repeated with multiple subjects (e.g. one European
    swallow and one African swallow).

    Parameters
    ----------
    data : array-like
        A :math:`m \times n` array; the element in row :math:`i` and
        column :math:`j` is the observation corresponding with subject
        :math:`i` and treatment :math:`j`. By default, the columns are
        assumed to be arranged in order of increasing predicted mean.

    ranked : boolean, optional
        By default, `data` is assumed to be observations rather than ranks;
        it will be ranked with `scipy.stats.rankdata` along ``axis=1``. If
        `data` is provided in the form of ranks, pass argument ``True``.

    predicted_ranks : array-like, optional
        The predicted ranks of the column means. If not specified,
        the columns are assumed to be arranged in order of increasing
        predicted mean, so the default `predicted_ranks` are
        :math:`[1, 2, \dots, n-1, n]`.
    method : {'auto', 'asymptotic', 'exact'}, optional
        Selects the method used to calculate the *p*-value. The following
        options are available.

        * 'auto': selects between 'exact' and 'asymptotic' to
          achieve reasonably accurate results in reasonable time (default)
        * 'asymptotic': compares the standardized test statistic against
          the normal distribution
        * 'exact': computes the exact *p*-value by comparing the observed
          :math:`L` statistic against those realized by all possible
          permutations of ranks (under the null hypothesis that each
          permutation is equally likely)
        选择用于计算 *p*-value 的方法。提供以下选项：
        * 'auto': 在合理时间内选择 'exact' 和 'asymptotic' 以获得合理准确的结果（默认）
        * 'asymptotic': 将标准化的检验统计量与正态分布进行比较
        * 'exact': 通过比较观察到的 :math:`L` 统计量与所有可能的排名排列（在每个排列等可能的零假设下）来计算精确的 *p*-value

    Returns
    -------
    res : PageTrendTestResult
        An object containing attributes:

        statistic : float
            Page's :math:`L` test statistic.
            Page 的 :math:`L` 检验统计量。
        pvalue : float
            The associated *p*-value
            相关的 *p*-value
        method : {'asymptotic', 'exact'}
            The method used to compute the *p*-value
            用于计算 *p*-value 的方法

    See Also
    --------
    rankdata, friedmanchisquare, spearmanr

    Notes
    -----
    As noted in [1]_, "the :math:`n` 'treatments' could just as well represent
    :math:`n` objects or events or performances or persons or trials ranked."
    Similarly, the :math:`m` 'subjects' could equally stand for :math:`m`
    "groupings by ability or some other control variable, or judges doing
    the ranking, or random replications of some other sort."

    如 [1]_ 中所述，“ :math:`n` 'treatments'” 可以很好地表示 :math:`n` 个对象或事件或表现或人员或排名试验。
    类似地， :math:`m` 'subjects' 同样可以表示 :math:`m` 个“能力分组或其他控制变量，或者进行排名的评委，或者其他类型的随机复制”。

    The procedure for calculating the :math:`L` statistic, adapted from
    [1]_, is:

    计算 :math:`L` 统计量的步骤，改编自 [1]_，如下：

    1. "Predetermine with careful logic the appropriate hypotheses
       concerning the predicted ordering of the experimental results.
       If no reasonable basis for ordering any treatments is known, the
       :math:`L` test is not appropriate."
       “使用谨慎的逻辑预先确定关于实验结果预测排序的适当假设。如果没有合理的基础来排序任何处理，那么 :math:`L` 检验不合适。”

    2. "As in other experiments, determine at what level of confidence
       you will reject the null hypothesis that there is no agreement of
       experimental results with the monotonic hypothesis."
       “与其他实验类似，确定在何种置信水平下将拒绝零假设，即实验结果与单调假设不一致。”

    3. "Cast the experimental material into a two-way table of :math:`n`
       columns (treatments, objects ranked, conditions) and :math:`m`
       rows (subjects, replication groups, levels of control variables)."
       “将实验材料转化为 :math:`n` 列（处理、对象排名、条件）和 :math:`m` 行（主体、重复组、控制变量水平）的双向表格。”

    4. "When experimental observations are recorded, rank them across each
       row", e.g. ``ranks = scipy.stats.rankdata(data, axis=1)``.
       “记录实验观测结果时，对每一行进行排名”，例如 ``ranks = scipy.stats.rankdata(data, axis=1)``。

    5. "Add the ranks in each column", e.g.
       ``colsums = np.sum(ranks, axis=0)``.
       “将每一列中的排名相加”，例如 ``colsums = np.sum(ranks, axis=0)``。

    6. "Multiply each sum of ranks by the predicted rank for that same
       column", e.g. ``products = predicted_ranks * colsums``.
       “将每个排名和的积乘以相同列的预测排名”，例如 ``products = predicted_ranks * colsums``。

    7. "Sum all such products", e.g. ``L = products.sum()``.
       “将所有这些积相加”，例如 ``L = products.sum()``。

    [1]_ continues by suggesting use of the standardized statistic

    .. math::

        \chi_L^2 = \frac{\left[12L-3mn(n+1)^2\right]^2}{mn^2(n^2-1)(n+1)}

    "which is distributed approximately as chi-square with 1 degree of
    freedom. The ordinary use of :math:`\chi^2` tables would be
    equivalent to a two-sided test of agreement. If a one-sided test

    继续建议使用标准化的统计量

    .. math::

        \chi_L^2 = \frac{\left[12L-3mn(n+1)^2\right]^2}{mn^2(n^2-1)(n+1)}

    “这近似服从自由度为1的卡方分布。普通使用 :math:`\chi^2` 表相当于双侧一致性检验。如果进行单侧检验
    # 文章包含Page趋势检验的详细说明，这种方法用于评估排名数据是否显示出一致的线性趋势。
    # 它涉及将观察到的排名与预期的排名进行比较，以便确定它们是否显著相关或反相关。
    
    is desired, *as will almost always be the case*, the probability
    discovered in the chi-square table should be *halved*."
    
    However, this standardized statistic does not distinguish between the
    observed values being well correlated with the predicted ranks and being
    _anti_-correlated with the predicted ranks. Instead, we follow [2]_
    and calculate the standardized statistic
    
    .. math::
    
        \Lambda = \frac{L - E_0}{\sqrt{V_0}},
    
    where :math:`E_0 = \frac{1}{4} mn(n+1)^2` and
    :math:`V_0 = \frac{1}{144} mn^2(n+1)(n^2-1)`, "which is asymptotically
    normal under the null hypothesis".
    
    The *p*-value for ``method='exact'`` is generated by comparing the observed
    value of :math:`L` against the :math:`L` values generated for all
    :math:`(n!)^m` possible permutations of ranks. The calculation is performed
    using the recursive method of [5].
    
    The *p*-values are not adjusted for the possibility of ties. When
    ties are present, the reported  ``'exact'`` *p*-values may be somewhat
    larger (i.e. more conservative) than the true *p*-value [2]_. The
    ``'asymptotic'``` *p*-values, however, tend to be smaller (i.e. less
    conservative) than the ``'exact'`` *p*-values.
    
    References
    ----------
    .. [1] Ellis Batten Page, "Ordered hypotheses for multiple treatments:
       a significant test for linear ranks", *Journal of the American
       Statistical Association* 58(301), p. 216--230, 1963.
    
    .. [2] Markus Neuhauser, *Nonparametric Statistical Test: A computational
       approach*, CRC Press, p. 150--152, 2012.
    
    .. [3] Statext LLC, "Page's L Trend Test - Easy Statistics", *Statext -
       Statistics Study*, https://www.statext.com/practice/PageTrendTest03.php,
       Accessed July 12, 2020.
    
    .. [4] "Page's Trend Test", *Wikipedia*, WikimediaFoundation,
       https://en.wikipedia.org/wiki/Page%27s_trend_test,
       Accessed July 12, 2020.
    
    .. [5] Robert E. Odeh, "The exact distribution of Page's L-statistic in
       the two-way layout", *Communications in Statistics - Simulation and
       Computation*,  6(1), p. 49--61, 1977.
    
    Examples
    --------
    We use the example from [3]_: 10 students are asked to rate three
    teaching methods - tutorial, lecture, and seminar - on a scale of 1-5,
    with 1 being the lowest and 5 being the highest. We have decided that
    a confidence level of 99% is required to reject the null hypothesis in
    favor of our alternative: that the seminar will have the highest ratings
    and the tutorial will have the lowest. Initially, the data have been
    tabulated with each row representing an individual student's ratings of
    the three methods in the following order: tutorial, lecture, seminar.
    
    >>> table = [[3, 4, 3],
    ...          [2, 2, 4],
    ...          [3, 3, 5],
    ...          [1, 3, 2],
    ...          [2, 3, 2],
    ...          [2, 4, 5],
    ...          [1, 2, 4],
    ...          [3, 4, 4],
    ...          [2, 4, 5],
    ...          [1, 3, 4]]

    # 假设教程的评分最低，将对应教程排名的列放在第一位；研讨会的评分最高，将其列放在最后。
    # 由于列已按预测均值升序排列，因此可以直接将表传递给 `page_trend_test` 函数。

    >>> from scipy.stats import page_trend_test
    >>> res = page_trend_test(table)
    >>> res
    PageTrendTestResult(statistic=133.5, pvalue=0.0018191161948127822,
                        method='exact')

    # 此 *p*-value 表明，在零假设下，统计量 :math:`L` 达到如此极端值的概率为 0.1819%。
    # 因为 0.1819% 小于 1%，我们有证据拒绝零假设，支持备择假设，置信水平为 99%。

    # 统计量 :math:`L` 的值为 133.5。为了手动检查这一点，我们对数据进行排名，
    # 高分对应高排名，使用平均排名解决并列情况：

    >>> from scipy.stats import rankdata
    >>> ranks = rankdata(table, axis=1)
    >>> ranks
    array([[1.5, 3. , 1.5],
           [1.5, 1.5, 3. ],
           [1.5, 1.5, 3. ],
           [1. , 3. , 2. ],
           [1.5, 3. , 1.5],
           [1. , 2. , 3. ],
           [1. , 2. , 3. ],
           [1. , 2.5, 2.5],
           [1. , 2. , 3. ],
           [1. , 2. , 3. ]])

    # 将每列内的排名求和，乘以预测排名，然后求和得到统计量 L。

    >>> import numpy as np
    >>> m, n = ranks.shape
    >>> predicted_ranks = np.arange(1, n+1)
    >>> L = (predicted_ranks * np.sum(ranks, axis=0)).sum()
    >>> res.statistic == L
    True

    # 如 [3]_ 所述，*p*-value 的渐近逼近是标准化检验统计量处正态分布的生存函数：

    >>> from scipy.stats import norm
    >>> E0 = (m*n*(n+1)**2)/4
    >>> V0 = (m*n**2*(n+1)*(n**2-1))/144
    >>> Lambda = (L-E0)/np.sqrt(V0)
    >>> p = norm.sf(Lambda)
    >>> p
    0.0012693433690751756

    # 这与上述由 `page_trend_test` 报告的 *p*-value 不完全匹配。
    # 渐近分布在 :math:`m \leq 12` 且 :math:`n \leq 8` 时不够精确，因此 `page_trend_test`
    # 基于表的维度和 Page 原始论文 [1]_ 的建议，选择了 ``method='exact'``。要覆盖
    # `page_trend_test` 的选择，提供 `method` 参数。

    >>> res = page_trend_test(table, method="asymptotic")
    >>> res
    PageTrendTestResult(statistic=133.5, pvalue=0.0012693433690751756,
                        method='asymptotic')

    # 如果数据已经排名，可以传入 ``ranks`` 而不是 ``table``，以节省计算时间。
    # 设置一个示例调用 `page_trend_test` 函数，演示如何使用不同的参数组合进行测试
    res = page_trend_test(ranks,             # 数据的排名
                          ranked=True,       # 数据已经排好序
                          )
    
    # 打印调用函数返回的结果对象 `res`
    res
    PageTrendTestResult(statistic=133.5, pvalue=0.0018191161948127822,
                        method='exact')
    
    Suppose the raw data had been tabulated in an order different from the
    order of predicted means, say lecture, seminar, tutorial.
    
    # 将 `table` 数组重新排列，以符合假设的顺序
    table = np.asarray(table)[:, [1, 2, 0]]
    
    Since the arrangement of this table is not consistent with the assumed
    ordering, we can either rearrange the table or provide the
    `predicted_ranks`. Remembering that the lecture is predicted
    to have the middle rank, the seminar the highest, and tutorial the lowest,
    we pass:
    
    # 使用 `page_trend_test` 函数，传入原始表格数据 `table` 和我们预测的顺序
    res = page_trend_test(table,             # 原始表格数据
                          predicted_ranks=[2, 3, 1],  # 我们预测的顺序
                          )
    
    # 打印调用函数返回的结果对象 `res`
    res
    PageTrendTestResult(statistic=133.5, pvalue=0.0018191161948127822,
                        method='exact')
    
    """
    
    # 可选的 `method` 参数及对应用于评估 p 值的函数
    methods = {"asymptotic": _l_p_asymptotic,
               "exact": _l_p_exact,
               "auto": None}
    
    # 如果传入的 `method` 不在可选的方法中，则抛出 ValueError 异常
    if method not in methods:
        raise ValueError(f"`method` must be in {set(methods)}")
    
    # 将数据转换为 NumPy 数组，并检查是否为二维数组
    ranks = np.asarray(data)
    if ranks.ndim != 2:  # TODO: relax this to accept 3d arrays?
        raise ValueError("`data` must be a 2d array.")
    
    m, n = ranks.shape
    # 检查数据的行数和列数是否满足 Page's L 方法的要求
    if m < 2 or n < 3:
        raise ValueError("Page's L is only appropriate for data with two "
                         "or more rows and three or more columns.")
    
    # 检查数据中是否包含 NaN 值，因为无法对包含 NaN 的数据有意义地进行排名
    if np.any(np.isnan(data)):
        raise ValueError("`data` contains NaNs, which cannot be ranked "
                         "meaningfully")
    
    # 如果数据已经排好序，进行简单的检查，确保数据已经排好序
    if ranked:
        if not (ranks.min() >= 1 and ranks.max() <= ranks.shape[1]):
            raise ValueError("`data` is not properly ranked. Rank the data or "
                             "pass `ranked=False`.")
    else:
        # 如果数据未排序，则使用 scipy.stats.rankdata 对数据进行排名
        ranks = scipy.stats.rankdata(data, axis=-1)
    
    # 如果未提供 `predicted_ranks`，则生成预测的排名数组，确保它是有效的 NumPy 数组
    if predicted_ranks is None:
        predicted_ranks = np.arange(1, n+1)
    else:
        predicted_ranks = np.asarray(predicted_ranks)
        # 检查预测的排名数组是否包含了从 1 到 n 的每个整数，并且长度为 n
        if (predicted_ranks.ndim < 1 or
                (set(predicted_ranks) != set(range(1, n+1)) or
                 len(predicted_ranks) != n)):
            raise ValueError(f"`predicted_ranks` must include each integer "
                             f"from 1 to {n} (the number of columns in "
                             f"`data`) exactly once.")
    # 检查 `ranked` 是否为布尔型，如果不是则抛出类型错误异常
    if not isinstance(ranked, bool):
        raise TypeError("`ranked` must be boolean.")

    # 计算 L 统计量
    L = _l_vectorized(ranks, predicted_ranks)

    # 计算 p 值
    # 如果 method 参数为 "auto"，则根据 ranks 计算出的方法来选择具体的方法
    if method == "auto":
        method = _choose_method(ranks)
    # 根据 method 从 methods 字典中获取对应的函数
    p_fun = methods[method]  # get the function corresponding with the method
    # 使用选定的方法函数 p_fun 计算 p 值，传入 L, m, n 作为参数
    p = p_fun(L, m, n)

    # 创建 PageTrendTestResult 对象，包含统计量 L, p 值 p, 使用的方法 method
    page_result = PageTrendTestResult(statistic=L, pvalue=p, method=method)
    # 返回计算结果对象
    return page_result
# 选择自动计算 p 值的方法
def _choose_method(ranks):
    '''Choose method for computing p-value automatically'''
    # 获取矩阵 ranks 的形状 m, n
    m, n = ranks.shape
    # 根据条件选择计算 p 值的方法，参考文献 [1], [4]
    if n > 8 or (m > 12 and n > 3) or m > 20:
        method = "asymptotic"
    else:
        method = "exact"
    return method


# 计算每个页面的 Page's L 统计量的向量化版本
def _l_vectorized(ranks, predicted_ranks):
    '''Calculate's Page's L statistic for each page of a 3d array'''
    # 按照页面方向计算 ranks 的列和
    colsums = ranks.sum(axis=-2, keepdims=True)
    # 计算预测 ranks 与列和的乘积
    products = predicted_ranks * colsums
    # 计算每个页面的 L 统计量
    Ls = products.sum(axis=-1)
    # 如果只有一个 L 值，则将其展平为一维数组
    Ls = Ls[0] if Ls.size == 1 else Ls.ravel()
    return Ls


# 根据渐近分布计算 Page's L 统计量的 p 值
def _l_p_asymptotic(L, m, n):
    '''Calculate the p-value of Page's L from the asymptotic distribution'''
    # 使用 [1] 参考文献，根据渐近分布计算 p 值
    E0 = (m*n*(n+1)**2)/4
    V0 = (m*n**2*(n+1)*(n**2-1))/144
    Lambda = (L-E0)/np.sqrt(V0)
    # 这是一个单侧“大于”检验，计算在零假设下 L 统计量大于观察到的 L 统计量的概率
    p = norm.sf(Lambda)
    return p


# 精确计算 Page's L 统计量的 p 值
def _l_p_exact(L, m, n):
    '''Calculate the p-value of Page's L exactly'''
    # [1] 使用 m, n; [5] 使用 n, k 的约定。这里调整为 exact 计算的约定
    L, n, k = int(L), int(m), int(n)
    _pagel_state.set_k(k)
    return _pagel_state.sf(L, n)


# _PageL 类：在多次执行 page_trend_test 之间维护状态
class _PageL:
    '''Maintains state between `page_trend_test` executions'''

    def __init__(self):
        '''Lightweight initialization'''
        self.all_pmfs = {}

    def set_k(self, k):
        '''Calculate lower and upper limits of L for single row'''
        # 计算单行的 L 统计量的下限和上限，参考 [5] 第 52 页顶部
        self.k = k
        self.a, self.b = (k*(k+1)*(k+2))//6, (k*(k+1)*(2*k+1))//6

    def sf(self, l, n):
        '''Survival function of Page's L statistic'''
        # 计算 Page's L 统计量的生存函数
        ps = [self.pmf(l, n) for l in range(l, n*self.b + 1)]
        return np.sum(ps)

    def pmf(self, l, n):
        '''Probability mass function of Page's L statistic'''
        # 计算 Page's L 统计量的概率质量函数
        if (l, n) not in self.all_pmfs:
            # 使用公式 [5] 方程 (6) 计算每个 L 值的相对频率
            ranks = range(1, self.k+1)
            rank_perms = np.array(list(permutations(ranks)))
            Ls = (ranks*rank_perms).sum(axis=1)
            counts = np.histogram(Ls, np.arange(self.a-0.5, self.b+1.5))[0]
            self.all_pmfs[(l, n)] = counts/math.factorial(self.k)
        return self.all_pmfs[(l, n)]
    def pmf(self, l, n):
        '''递归函数，用于计算 p(l, k, n); 参见 [5] 方程 1'''

        # 如果 self.all_pmfs 中没有 n 这个键，则创建一个空字典
        if n not in self.all_pmfs:
            self.all_pmfs[n] = {}
        # 如果 self.all_pmfs[n] 中没有 self.k 这个键，则创建一个空字典
        if self.k not in self.all_pmfs[n]:
            self.all_pmfs[n][self.k] = {}

        # 检查是否已经计算过结果，以避免重复计算。最初可能使用 lru_cache，但似乎这种方法更快？还可以添加选项以保存计算结果以供将来查询。
        if l in self.all_pmfs[n][self.k]:
            return self.all_pmfs[n][self.k][l]

        # 当 n 等于 1 时的情况
        if n == 1:
            ps = self.p_l_k_1()  # [5] 方程 6
            ls = range(self.a, self.b+1)
            # 虽然不快，但我们只会进入这里一次
            self.all_pmfs[n][self.k] = {l: p for l, p in zip(ls, ps)}
            return self.all_pmfs[n][self.k][l]

        p = 0
        # 计算 low 和 high 的值，参见 [5] 方程 2
        low = max(l - (n-1) * self.b, self.a)
        high = min(l - (n-1) * self.a, self.b)

        # 计算 [5] 方程 1
        for t in range(low, high + 1):
            p1 = self.pmf(l - t, n - 1)
            p2 = self.pmf(t, 1)
            p += p1 * p2
        self.all_pmfs[n][self.k][l] = p
        return p
# 创建一个用于维护状态的对象 _PageL 的实例，以便在使用 page_trend_test 函数时能够更快地进行重复调用，
# 其中方法(method)参数设置为 'exact'。
_pagel_state = _PageL()
```