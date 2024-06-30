# `D:\src\scipysrc\scipy\scipy\stats\_bws_test.py`

```
import numpy as np  # 导入NumPy库，用于数值计算
from functools import partial  # 导入functools库中的partial函数，用于创建偏函数
from scipy import stats  # 导入SciPy库中的stats模块，用于统计计算


def _bws_input_validation(x, y, alternative, method):
    ''' Input validation and standardization for bws test'''
    x, y = np.atleast_1d(x, y)  # 将输入数据x和y转换为至少是一维数组
    if x.ndim > 1 or y.ndim > 1:  # 检查x和y是否为一维数组，否则引发异常
        raise ValueError('`x` and `y` must be exactly one-dimensional.')
    if np.isnan(x).any() or np.isnan(y).any():  # 检查x和y是否包含NaN值，是则引发异常
        raise ValueError('`x` and `y` must not contain NaNs.')
    if np.size(x) == 0 or np.size(y) == 0:  # 检查x和y的大小是否为非零，否则引发异常
        raise ValueError('`x` and `y` must be of nonzero size.')

    z = stats.rankdata(np.concatenate((x, y)))  # 计算合并后数据的秩次，返回秩次数组z
    x, y = z[:len(x)], z[len(x):]  # 将秩次数组z分割回x和y的秩次

    alternatives = {'two-sided', 'less', 'greater'}  # 定义备择假设的集合
    alternative = alternative.lower()  # 将alternative参数转换为小写
    if alternative not in alternatives:  # 检查alternative是否在备择假设集合中，否则引发异常
        raise ValueError(f'`alternative` must be one of {alternatives}.')

    method = stats.PermutationMethod() if method is None else method  # 如果method参数为None，则使用默认的PermutationMethod
    if not isinstance(method, stats.PermutationMethod):  # 检查method是否为PermutationMethod的实例，否则引发异常
        raise ValueError('`method` must be an instance of '
                         '`scipy.stats.PermutationMethod`')

    return x, y, alternative, method  # 返回验证和标准化后的x、y及其他参数


def _bws_statistic(x, y, alternative, axis):
    '''Compute the BWS test statistic for two independent samples'''
    # 公共函数当前不接受`axis`参数，但`permutation_test`使用`axis`进行向量化调用。

    Ri, Hj = np.sort(x, axis=axis), np.sort(y, axis=axis)  # 对x和y按照指定轴向排序，分别赋值给Ri和Hj
    n, m = Ri.shape[axis], Hj.shape[axis]  # 获取排序后数组的长度，分别赋值给n和m
    i, j = np.arange(1, n+1), np.arange(1, m+1)  # 创建1到n和1到m的数组，分别赋值给i和j

    Bx_num = Ri - (m + n)/n * i  # 计算Bx的分子部分
    By_num = Hj - (m + n)/m * j  # 计算By的分子部分

    if alternative == 'two-sided':  # 根据alternative参数选择计算方式
        Bx_num *= Bx_num
        By_num *= By_num
    else:
        Bx_num *= np.abs(Bx_num)
        By_num *= np.abs(By_num)

    Bx_den = i/(n+1) * (1 - i/(n+1)) * m*(m+n)/n  # 计算Bx的分母部分
    By_den = j/(m+1) * (1 - j/(m+1)) * n*(m+n)/m  # 计算By的分母部分

    Bx = 1/n * np.sum(Bx_num/Bx_den, axis=axis)  # 计算Bx的值
    By = 1/m * np.sum(By_num/By_den, axis=axis)  # 计算By的值

    B = (Bx + By) / 2 if alternative == 'two-sided' else (Bx - By) / 2  # 计算BWS统计量B

    return B  # 返回BWS统计量B


def bws_test(x, y, *, alternative="two-sided", method=None):
    r'''Perform the Baumgartner-Weiss-Schindler test on two independent samples.

    The Baumgartner-Weiss-Schindler (BWS) test is a nonparametric test of 
    the null hypothesis that the distribution underlying sample `x` 
    is the same as the distribution underlying sample `y`. Unlike 
    the Kolmogorov-Smirnov, Wilcoxon, and Cramer-Von Mises tests, 
    the BWS test weights the integral by the variance of the difference
    in cumulative distribution functions (CDFs), emphasizing the tails of the
    distributions, which increases the power of the test in many applications.

    Parameters
    ----------
    x, y : array-like
        1-d arrays of samples.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
    method : PermutationMethod, optional
        Defines the method used for permutations. If None, defaults to
        `stats.PermutationMethod()`.

    Returns
    -------
    float
        The BWS test statistic.

    '''
    alternative : {'two-sided', 'less', 'greater'}, optional
        # 定义备择假设。默认为 'two-sided'。
        # 让 *F(u)* 和 *G(u)* 分别为 `x` 和 `y` 底层分布的累积分布函数。
        # 下列备择假设可供选择：

        # * 'two-sided': 分布不相等，即至少存在一个 *u* 使得 *F(u) ≠ G(u)*。
        # * 'less': `x` 的底层分布比 `y` 的底层分布要随机地小，即对所有 *u* 都满足 *F(u) >= G(u)*。
        # * 'greater': `x` 的底层分布比 `y` 的底层分布要随机地大，即对所有 *u* 都满足 *F(u) <= G(u)*。

        # 在更严格的假设下，备择假设可以根据分布的位置来表达；详见 [2] 第 5.1 节。
    method : PermutationMethod, optional
        # 配置用于计算 p 值的方法。默认使用默认的 `PermutationMethod` 对象。

    Returns
    -------
    res : PermutationTestResult
        # 一个对象，具有以下属性：

    statistic : float
        # 数据的观察测试统计量。
    pvalue : float
        # 给定备择假设的 p 值。
    null_distribution : ndarray
        # 在零假设下生成的测试统计值。

    See also
    --------
    scipy.stats.wilcoxon, scipy.stats.mannwhitneyu, scipy.stats.ttest_ind

    Notes
    -----
    # 当 ``alternative=='two-sided'`` 时，统计量由 [1]_ 第 2 节中给出的方程定义。
    # 这个统计量不适用于单侧备择假设；在这种情况下，统计量是 [1]_ 第 2 节中给出的统计量的负值。
    # 因此，当第一个样本的分布随机大于第二个样本的分布时，统计量将趋向于正值。

    References
    ----------
    .. [1] Neuhäuser, M. (2005). Exact Tests Based on the
           Baumgartner-Weiss-Schindler Statistic: A Survey. Statistical Papers,
           46(1), 1-29.
    .. [2] Fay, M. P., & Proschan, M. A. (2010). Wilcoxon-Mann-Whitney or t-test?
           On assumptions for hypothesis tests and multiple interpretations of 
           decision rules. Statistics surveys, 4, 1.

    Examples
    --------
    # 我们按照 [1]_ 中表 3 的例子进行操作：将十四个孩子随机分为两组。他们在进行特定测试时的排名如下。

    >>> import numpy as np
    >>> x = [1, 2, 3, 4, 6, 7, 8]
    >>> y = [5, 9, 10, 11, 12, 13, 14]

    # 我们使用 BWS 测试来评估两组之间是否存在统计显著差异。
    # 零假设是两组之间的表现分布没有差异。我们决定显著性水平为
    '''
    1% is required to reject the null hypothesis in favor of the alternative
    that the distributions are different.
    Since the number of samples is very small, we can compare the observed test
    statistic against the *exact* distribution of the test statistic under the
    null hypothesis.

    >>> from scipy.stats import bws_test
    >>> res = bws_test(x, y)
    >>> print(res.statistic)
    5.132167152575315

    This agrees with :math:`B = 5.132` reported in [1]_. The *p*-value produced
    by `bws_test` also agrees with :math:`p = 0.0029` reported in [1]_.

    >>> print(res.pvalue)
    0.002913752913752914

    Because the p-value is below our threshold of 1%, we take this as evidence
    against the null hypothesis in favor of the alternative that there is a
    difference in performance between the two groups.
    '''
    # 对输入的数据进行验证和预处理，确保输入的合法性和一致性
    x, y, alternative, method = _bws_input_validation(x, y, alternative,
                                                      method)
    # 根据选择的备择假设（alternative）创建相应的统计量函数
    bws_statistic = partial(_bws_statistic, alternative=alternative)

    # 根据选择的备择假设确定置换检验的方向
    permutation_alternative = 'less' if alternative == 'less' else 'greater'
    
    # 执行置换检验，得到检验结果
    res = stats.permutation_test((x, y), bws_statistic,
                                 alternative=permutation_alternative,
                                 **method._asdict())

    # 返回置换检验的结果
    return res
```