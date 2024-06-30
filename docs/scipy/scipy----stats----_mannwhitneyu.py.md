# `D:\src\scipysrc\scipy\scipy\stats\_mannwhitneyu.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
from collections import namedtuple  # 导入 namedtuple 类型，用于创建命名元组
from scipy import special  # 导入 scipy 库的 special 模块，用于特殊数学函数
from scipy import stats  # 导入 scipy 库的 stats 模块，用于统计函数
from scipy.stats._stats_py import _rankdata  # 导入 scipy 库内部的 _rankdata 函数
from ._axis_nan_policy import _axis_nan_policy_factory  # 导入本地模块中的 _axis_nan_policy_factory 函数


def _broadcast_concatenate(x, y, axis):
    '''Broadcast then concatenate arrays, leaving concatenation axis last'''
    # 移动数组 x 和 y 的轴，使得连接轴在最后
    x = np.moveaxis(x, axis, -1)
    y = np.moveaxis(y, axis, -1)
    # 使用广播机制，扩展 x 和 y 到相同形状
    z = np.broadcast(x[..., 0], y[..., 0])
    x = np.broadcast_to(x, z.shape + (x.shape[-1],))
    y = np.broadcast_to(y, z.shape + (y.shape[-1],))
    # 在连接轴上连接数组 x 和 y，得到新数组 z
    z = np.concatenate((x, y), axis=-1)
    return x, y, z


class _MWU:
    '''Distribution of MWU statistic under the null hypothesis'''

    def __init__(self, n1, n2):
        self._reset(n1, n2)

    def set_shapes(self, n1, n2):
        # 将 n1 和 n2 设置为较小和较大值，确保 n1 <= n2
        n1, n2 = min(n1, n2), max(n1, n2)
        # 如果已经是当前实例的 n1 和 n2，则直接返回
        if (n1, n2) == (self.n1, self.n2):
            return

        # 更新实例的 n1 和 n2，并初始化相关数组
        self.n1 = n1
        self.n2 = n2
        self.s_array = np.zeros(0, dtype=int)
        self.configurations = np.zeros(0, dtype=np.uint64)

    def reset(self):
        # 重置实例的状态到初始状态
        self._reset(self.n1, self.n2)

    def _reset(self, n1, n2):
        # 私有方法，重置实例的状态到初始状态
        self.n1 = None
        self.n2 = None
        self.set_shapes(n1, n2)

    def pmf(self, k):
        # 概率质量函数，返回指定 k 值的概率质量

        # 在实际中，k 不会大于 m*n/2。
        # 如果 k 大于这个值，可以利用对称性优化计算：
        # k = np.array(k, copy=True)
        # k2 = m*n - k
        # i = k2 < k
        # k[i] = k2[i]

        # 构建并返回 k 值对应的概率质量数组
        pmfs = self.build_u_freqs_array(np.max(k))
        return pmfs[k]

    def cdf(self, k):
        '''Cumulative distribution function'''
        # 累积分布函数，返回指定 k 值的累积分布

        # 在实际中，k 不会大于 m*n/2。
        # 如果 k 大于这个值，可以在这里利用对称性优化计算，而不是在 sf 中优化
        pmfs = self.build_u_freqs_array(np.max(k))
        cdfs = np.cumsum(pmfs)
        return cdfs[k]

    def sf(self, k):
        '''Survival function'''
        # 生存函数，返回指定 k 值的生存函数值
        # 注意，CDF 和 SF 都包含 k 处的 PMF。p 值是从 SF 计算的，应包含 k 处的质量，这是可取的

        # 利用分布的对称性，从左侧计算累积和
        kc = np.asarray(self.n1*self.n2 - k)  # k 的补集
        i = k < kc
        if np.any(i):
            kc[i] = k[i]
            cdfs = np.asarray(self.cdf(kc))
            cdfs[i] = 1. - cdfs[i] + self.pmf(kc[i])
        else:
            cdfs = np.asarray(self.cdf(kc))
        return cdfs[()]

    # build_sigma_array 和 build_u_freqs_array 从 @toobaz 的代码中适配而来，经许可使用。
    # 感谢 @andreasloe 的建议。
    # 参见 https://github.com/scipy/scipy/pull/4933#issuecomment-1898082691
    # 构建 sigma 数组，用于计算组合数
    def build_sigma_array(self, a):
        # 解构赋值，获取对象属性 n1 和 n2
        n1, n2 = self.n1, self.n2
        # 如果 a+1 小于等于当前 s_array 的大小，则直接返回部分 s_array
        if a + 1 <= self.s_array.size:
            return self.s_array[1:a+1]

        # 初始化一个大小为 a+1 的零数组 s_array
        s_array = np.zeros(a + 1, dtype=int)

        # 计算每个 d 的倍数（不包括 0），并对应位置上加上 d
        for d in np.arange(1, n1 + 1):
            indices = np.arange(d, a + 1, d)
            s_array[indices] += d  # \epsilon_d = 1

        # 计算每个 d 的倍数（不包括 0），并对应位置上减去 d
        for d in np.arange(n2 + 1, n2 + n1 + 1):
            indices = np.arange(d, a + 1, d)
            s_array[indices] -= d  # \epsilon_d = -1

        # 更新对象的 s_array 属性
        self.s_array = s_array
        # 返回 s_array 去掉第一个元素后的部分
        return s_array[1:]

    # 构建 u 频率数组，用于表示 U=0 到 U=maxu 的概率分布
    def build_u_freqs_array(self, maxu):
        """
        构建从 0 到 maxu 的所有频率数组。
        假设：
          n1 <= n2
          maxu <= n1 * n2 / 2
        """
        # 解构赋值，获取对象属性 n1 和 n2
        n1, n2 = self.n1, self.n2
        # 计算组合数的总数
        total = special.binom(n1 + n2, n1)

        # 如果 maxu+1 小于等于当前 configurations 的大小，则直接返回部分 configurations
        if maxu + 1 <= self.configurations.size:
            return self.configurations[:maxu + 1] / total

        # 构建 sigma 数组，用于后续计算
        s_array = self.build_sigma_array(maxu)

        # 初始化一个大小为 maxu+1 的零数组 configurations，用于存储计算结果
        configurations = np.zeros(maxu + 1, dtype=np.uint64)
        configurations_is_uint = True  # 标记 configurations 是否为无符号整数类型
        uint_max = np.iinfo(np.uint64).max  # 获取 uint64 的最大值

        # 初始条件：U=0 时的概率为 1
        configurations[0] = 1

        # 计算从 U=1 到 U=maxu 的概率分布
        for u in np.arange(1, maxu + 1):
            # 取出 configurations[:u] 和 s_array[u-1::-1] 的内积，并除以 u 得到新的值
            coeffs = s_array[u - 1::-1]
            new_val = np.dot(configurations[:u], coeffs) / u

            # 如果新计算的值超过了 uint64 的最大值，并且 configurations 目前仍然为 uint 类型
            if new_val > uint_max and configurations_is_uint:
                # 切换为浮点数类型数组进行后续计算，以保证精度
                configurations = configurations.astype(float)
                configurations_is_uint = False

            # 更新 configurations 数组中的值为新计算得到的值
            configurations[u] = new_val

        # 更新对象的 configurations 属性
        self.configurations = configurations
        # 返回 configurations 数组除以总数 total 后的概率分布
        return configurations / total
_mwu_state = _MWU(0, 0)

# 初始化一个名为 `_mwu_state` 的全局变量，其值为 `_MWU` 类的一个实例，初始参数为 (0, 0)。


def _get_mwu_z(U, n1, n2, t, axis=0, continuity=True):
    '''Standardized MWU statistic'''

# 定义名为 `_get_mwu_z` 的函数，计算标准化的 MWU 统计量。接受参数 `U`, `n1`, `n2`, `t`，并且有可选参数 `axis` 和 `continuity`。


    # Follows mannwhitneyu [2]
    mu = n1 * n2 / 2
    n = n1 + n2

# 根据文献 [2]，计算 `mu` 为 `n1 * n2 / 2`，`n` 为 `n1 + n2`。


    # Tie correction according to [2], "Normal approximation and tie correction"
    # "A more computationally-efficient form..."
    tie_term = (t**3 - t).sum(axis=-1)
    s = np.sqrt(n1*n2/12 * ((n + 1) - tie_term/(n*(n-1))))

# 根据文献 [2] 中的描述，进行关于 "正态近似和结扎校正" 的结扎校正。计算 `tie_term` 和 `s` 的值。


    numerator = U - mu

# 计算 `numerator`，即 `U - mu`。


    # Continuity correction.
    # Because SF is always used to calculate the p-value, we can always
    # _subtract_ 0.5 for the continuity correction. This always increases the
    # p-value to account for the rest of the probability mass _at_ q = U.
    if continuity:
        numerator -= 0.5

# 如果 `continuity` 为 `True`，进行连续性校正，将 `numerator` 减去 0.5。


    # no problem evaluating the norm SF at an infinity
    with np.errstate(divide='ignore', invalid='ignore'):
        z = numerator / s
    return z

# 使用 NumPy 的错误状态管理，计算并返回标准化 MWU 统计量 `z`。


def _mwu_input_validation(x, y, use_continuity, alternative, axis, method):
    ''' Input validation and standardization for mannwhitneyu '''

# 定义名为 `_mwu_input_validation` 的函数，用于验证和标准化 `mannwhitneyu` 的输入参数 `x`, `y`, `use_continuity`, `alternative`, `axis`, `method`。


    # Would use np.asarray_chkfinite, but infs are OK
    x, y = np.atleast_1d(x), np.atleast_1d(y)
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError('`x` and `y` must not contain NaNs.')
    if np.size(x) == 0 or np.size(y) == 0:
        raise ValueError('`x` and `y` must be of nonzero size.')

# 使用 NumPy 函数确保 `x` 和 `y` 至少是一维数组，并检查它们不包含 `NaN` 值。如果有问题，则引发 `ValueError`。


    bools = {True, False}
    if use_continuity not in bools:
        raise ValueError(f'`use_continuity` must be one of {bools}.')

# 检查 `use_continuity` 是否为布尔类型的值。


    alternatives = {"two-sided", "less", "greater"}
    alternative = alternative.lower()
    if alternative not in alternatives:
        raise ValueError(f'`alternative` must be one of {alternatives}.')

# 将 `alternative` 转换为小写，并检查其是否为合法的字符串值。


    axis_int = int(axis)
    if axis != axis_int:
        raise ValueError('`axis` must be an integer.')

# 将 `axis` 转换为整数，并检查其是否为整数值。


    if not isinstance(method, stats.PermutationMethod):
        methods = {"asymptotic", "exact", "auto"}
        method = method.lower()
        if method not in methods:
            raise ValueError(f'`method` must be one of {methods}.')

# 检查 `method` 是否为 `stats.PermutationMethod` 的实例或者是合法的字符串值。


    return x, y, use_continuity, alternative, axis_int, method

# 返回经验证和标准化后的输入参数。


def _mwu_choose_method(n1, n2, ties):
    """Choose method 'asymptotic' or 'exact' depending on input size, ties"""

# 定义名为 `_mwu_choose_method` 的函数，根据输入的大小和结扎数，选择 `asymptotic` 或 `exact` 方法。


    # if both inputs are large, asymptotic is OK
    if n1 > 8 and n2 > 8:
        return "asymptotic"

    # if there are any ties, asymptotic is preferred
    if ties:
        return "asymptotic"

    return "exact"

# 根据输入的大小和是否存在结扎情况，返回合适的方法名称字符串。


MannwhitneyuResult = namedtuple('MannwhitneyuResult', ('statistic', 'pvalue'))

# 定义名为 `MannwhitneyuResult` 的命名元组，用于表示 `mannwhitneyu` 函数的结果，包含 `statistic` 和 `pvalue` 两个字段。


@_axis_nan_policy_factory(MannwhitneyuResult, n_samples=2)
def mannwhitneyu(x, y, use_continuity=True, alternative="two-sided",
                 axis=0, method="auto"):

# 使用 `_axis_nan_policy_factory` 装饰器定义 `mannwhitneyu` 函数，用于执行两独立样本的 Mann-Whitney U 排名检验。


    r'''Perform the Mann-Whitney U rank test on two independent samples.

    The Mann-Whitney U test is a nonparametric test of the null hypothesis
    that the distribution underlying sample `x` is the same as the
    distribution underlying sample `y`. It is often used as a test of

# `mannwhitneyu` 函数的文档字符串，说明它执行 Mann-Whitney U 排名检验，用于检验两个独立样本的分布是否相同。
    # 定义一个函数，用于执行 Mann-Whitney U 检验，用于比较两个样本的位置差异。
    
    Parameters
    ----------
    x, y : array-like
        N 维样本数组。除了 `axis` 给定的维度外，这些数组必须可以进行广播。
    use_continuity : bool, optional
        是否应用连续性校正（1/2）。当 `method` 是 ``'asymptotic'`` 时，默认为 True；否则无效。
    alternative : {'two-sided', 'less', 'greater'}, optional
        定义备择假设。默认为 'two-sided'。
        设 *F(u)* 和 *G(u)* 分别为 `x` 和 `y` 背后的累积分布函数。下列备择假设可供选择：
    
        * 'two-sided': 分布不相等，即对于至少一个 *u*，*F(u) ≠ G(u)*。
        * 'less': `x` 背后的分布在随机意义上小于 `y` 背后的分布，即对所有 *u*，*F(u) > G(u)*。
        * 'greater': `x` 背后的分布在随机意义上大于 `y` 背后的分布，即对所有 *u*，*F(u) < G(u)*。
    
        注意上述备择假设中的数学表达描述了背后分布的累积分布函数。不同方向的不等式看起来可能与自然语言描述不一致，但实际上不是这样。例如，假设 *X* 和 *Y* 是遵循具有累积分布函数 *F* 和 *G* 的随机变量。如果对所有 *u*，*F(u) > G(u)*，则从 *X* 中抽取的样本倾向于小于从 *Y* 中抽取的样本。
    
        在更严格的一组假设下，备择假设可以用分布位置的术语来表达；详见 [5] 章节 5.1。
    axis : int, optional
        执行检验的轴。默认为 0。
    method : {'auto', 'asymptotic', 'exact'} or `PermutationMethod` instance, optional
        选择用于计算 *p* 值的方法。默认为 'auto'。可选项如下。
    
        * ``'asymptotic'``: 将标准化的检验统计量与正态分布进行比较，修正绑定。
        * ``'exact'``: 通过将观察到的 :math:`U` 统计量与零假设下 :math:`U` 统计量的精确分布进行比较来计算精确的 *p* 值。不对绑定进行修正。
        * ``'auto'``: 当其中一个样本的大小小于或等于 8 且没有绑定时，选择 ``'exact'``；否则选择 ``'asymptotic'``。
        * `PermutationMethod` 实例。在这种情况下，使用提供的配置选项和其他适当设置使用 `permutation_test` 计算 *p* 值。
    
    Returns
    -------
    res : MannwhitneyuResult
        # 一个包含以下属性的对象：

        statistic : float
            # 对应于样本 `x` 的 Mann-Whitney U 统计量。参见注释以了解对应于样本 `y` 的测试统计量。
        pvalue : float
            # 所选 `alternative` 的相关 *p* 值。

    Notes
    -----
    如果 ``U1`` 是对应于样本 `x` 的统计量，则对应于样本 `y` 的统计量为
    ``U2 = x.shape[axis] * y.shape[axis] - U1``。

    `mannwhitneyu` 适用于独立样本。对于相关/配对样本，请考虑 `scipy.stats.wilcoxon`。

    `method` ``'exact'`` 在没有任何并列值且任一样本大小小于8时建议使用 [1]_。该实现遵循 [3]_ 中报告的算法。
    注意，精确方法未校正并列值，但是如果数据中有并列值，`mannwhitneyu` 不会引发错误或警告。
    如果数据中有并列值且任一样本较小（少于约10个观测值），考虑将 `PermutationMethod` 的实例作为 `method` 传递，执行置换检验。

    Mann-Whitney U 检验是独立样本的非参数 t 检验版本。当来自总体的样本均值服从正态分布时，考虑 `scipy.stats.ttest_ind`。

    See Also
    --------
    scipy.stats.wilcoxon, scipy.stats.ranksums, scipy.stats.ttest_ind

    References
    ----------
    .. [1] H.B. Mann 和 D.R. Whitney, "On a test of whether one of two random
           variables is stochastically larger than the other", The Annals of
           Mathematical Statistics, Vol. 18, pp. 50-60, 1947.
    .. [2] Mann-Whitney U Test, Wikipedia,
           http://en.wikipedia.org/wiki/Mann-Whitney_U_test
    .. [3] Andreas Löffler,
           "Über eine Partition der nat. Zahlen und ihr Anwendung beim U-Test",
           Wiss. Z. Univ. Halle, XXXII'83 pp. 87-89.
    .. [4] Rosie Shier, "Statistics: 2.3 The Mann-Whitney U Test", Mathematics
           Learning Support Centre, 2004.
    .. [5] Michael P. Fay 和 Michael A. Proschan. "Wilcoxon-Mann-Whitney
           or t-test? On assumptions for hypothesis tests and multiple \
           interpretations of decision rules." Statistics surveys, Vol. 4, pp.
           1-39, 2010. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2857732/

    Examples
    --------
    我们按照 [4]_ 中的示例：九名随机抽取的年轻成年人在以下年龄被诊断为 II 型糖尿病。

    >>> males = [19, 22, 16, 29, 24]
    >>> females = [20, 11, 17, 12]

    我们使用 Mann-Whitney U 检验评估男性和女性诊断年龄之间是否存在统计显著差异。
    零假设是男性诊断年龄的分布与女性相同。我们决定
    that a confidence level of 95% is required to reject the null hypothesis
    in favor of the alternative that the distributions are different.
    Since the number of samples is very small and there are no ties in the
    data, we can compare the observed test statistic against the *exact*
    distribution of the test statistic under the null hypothesis.

    >>> from scipy.stats import mannwhitneyu
    # 导入 mannwhitneyu 函数从 scipy.stats 模块
    >>> U1, p = mannwhitneyu(males, females, method="exact")
    # 使用 Mann-Whitney U 检验比较两组样本 males 和 females，选择精确方法
    >>> print(U1)
    17.0

    `mannwhitneyu` always reports the statistic associated with the first
    sample, which, in this case, is males. This agrees with :math:`U_M = 17`
    reported in [4]_. The statistic associated with the second statistic
    can be calculated:

    >>> nx, ny = len(males), len(females)
    # 计算两组样本的大小 nx 和 ny
    >>> U2 = nx*ny - U1
    # 计算第二个统计量 U2
    >>> print(U2)
    3.0

    This agrees with :math:`U_F = 3` reported in [4]_. The two-sided
    *p*-value can be calculated from either statistic, and the value produced
    by `mannwhitneyu` agrees with :math:`p = 0.11` reported in [4]_.

    >>> print(p)
    0.1111111111111111

    The exact distribution of the test statistic is asymptotically normal, so
    the example continues by comparing the exact *p*-value against the
    *p*-value produced using the normal approximation.

    >>> _, pnorm = mannwhitneyu(males, females, method="asymptotic")
    # 使用渐近方法计算 Mann-Whitney U 检验，得到对应的 p 值 pnorm
    >>> print(pnorm)
    0.11134688653314041

    Here `mannwhitneyu`'s reported *p*-value appears to conflict with the
    value :math:`p = 0.09` given in [4]_. The reason is that [4]_
    does not apply the continuity correction performed by `mannwhitneyu`;
    `mannwhitneyu` reduces the distance between the test statistic and the
    mean :math:`\mu = n_x n_y / 2` by 0.5 to correct for the fact that the
    discrete statistic is being compared against a continuous distribution.
    Here, the :math:`U` statistic used is less than the mean, so we reduce
    the distance by adding 0.5 in the numerator.

    >>> import numpy as np
    >>> from scipy.stats import norm
    # 导入 numpy 和 scipy.stats 模块中的 norm 函数
    >>> U = min(U1, U2)
    # 选择较小的 U1 和 U2 作为 U
    >>> N = nx + ny
    # 计算总样本数 N
    >>> z = (U - nx*ny/2 + 0.5) / np.sqrt(nx*ny * (N + 1)/ 12)
    # 计算 z 统计量，用于基于正态分布近似计算 p 值
    >>> p = 2 * norm.cdf(z)  # use CDF to get p-value from smaller statistic
    # 使用累积分布函数计算双侧 p 值
    >>> print(p)
    0.11134688653314041

    If desired, we can disable the continuity correction to get a result
    that agrees with that reported in [4]_.

    >>> _, pnorm = mannwhitneyu(males, females, use_continuity=False,
    ...                         method="asymptotic")
    # 禁用连续性修正，得到与文献 [4]_ 报告一致的结果
    >>> print(pnorm)
    0.0864107329737

    Regardless of whether we perform an exact or asymptotic test, the
    probability of the test statistic being as extreme or more extreme by
    chance exceeds 5%, so we do not consider the results statistically
    significant.

    Suppose that, before seeing the data, we had hypothesized that females
    would tend to be diagnosed at a younger age than males.
    In that case, it would be natural to provide the female ages as the
    '''
    The following code implements the Mann-Whitney U test for comparing two independent samples.
    It calculates the test statistic and p-value to determine if there is a significant difference
    between the distributions of the two samples.
    
    '''
    
    x, y, use_continuity, alternative, axis_int, method = (
        _mwu_input_validation(x, y, use_continuity, alternative, axis, method))
    
    x, y, xy = _broadcast_concatenate(x, y, axis)
    
    n1, n2 = x.shape[-1], y.shape[-1]
    
    # Follows [2]
    # 将两个样本合并后，计算排名和计算 T 值
    ranks, t = _rankdata(xy, 'average', return_ties=True)  # method 2, step 1
    # 计算第一个样本的秩和 R1
    R1 = ranks[..., :n1].sum(axis=-1)                      # method 2, step 2
    # 计算 U1 统计量
    U1 = R1 - n1*(n1+1)/2                                  # method 2, step 3
    # 根据 U1 + U2 = n1 * n2，计算 U2 统计量
    U2 = n1 * n2 - U1                                      # as U1 + U2 = n1 * n2
    
    if alternative == "greater":
        U, f = U1, 1  # 使用 U1 作为统计量，f 为缩放因子
    elif alternative == "less":
        U, f = U2, 1  # 由于对称性，使用 U2 的生存函数（Survival Function），而不是 U1 的累积分布函数（Cumulative Distribution Function）
    else:
        U, f = np.maximum(U1, U2), 2  # 对于双侧检验，使用两倍的生存函数
    
    if method == "auto":
        method = _mwu_choose_method(n1, n2, np.any(t > 1))
    
    if method == "exact":
        _mwu_state.set_shapes(n1, n2)
        # 计算确切法下的 p 值
        p = _mwu_state.sf(U.astype(int))
    elif method == "asymptotic":
        # 计算渐近法下的 p 值
        z = _get_mwu_z(U, n1, n2, t, continuity=use_continuity)
        p = stats.norm.sf(z)
    else:  # `PermutationMethod` instance (already validated)
        # 使用排列检验方法计算 p 值
        def statistic(x, y, axis):
            return mannwhitneyu(x, y, use_continuity=use_continuity,
                                alternative=alternative, axis=axis,
                                method="asymptotic").statistic
    
        res = stats.permutation_test((x, y), statistic, axis=axis,
                                     **method._asdict(), alternative=alternative)
        p = res.pvalue
        f = 1
    
    p *= f
    
    # 确保检验统计量不大于 1
    # 这种情况可能发生在确切检验中，当 U = m*n/2 时
    p = np.clip(p, 0, 1)
    
    # 返回 Mann-Whitney U 检验结果
    return MannwhitneyuResult(U1, p)
```