# `D:\src\scipysrc\scipy\scipy\stats\_survival.py`

```
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import warnings

import numpy as np
from scipy import special, interpolate, stats
from scipy.stats._censored_data import CensoredData
from scipy.stats._common import ConfidenceInterval

if TYPE_CHECKING:
    from typing import Literal
    import numpy.typing as npt


__all__ = ['ecdf', 'logrank']


@dataclass
class EmpiricalDistributionFunction:
    """An empirical distribution function produced by `scipy.stats.ecdf`

    Attributes
    ----------
    quantiles : ndarray
        The unique values of the sample from which the
        `EmpiricalDistributionFunction` was estimated.
    probabilities : ndarray
        The point estimates of the cumulative distribution function (CDF) or
        its complement, the survival function (SF), corresponding with
        `quantiles`.
    """
    quantiles: np.ndarray
    probabilities: np.ndarray
    # Exclude these from __str__
    _n: np.ndarray = field(repr=False)  # number "at risk"
    _d: np.ndarray = field(repr=False)  # number of "deaths"
    _sf: np.ndarray = field(repr=False)  # survival function for var estimate
    _kind: str = field(repr=False)  # type of function: "cdf" or "sf"

    def __init__(self, q, p, n, d, kind):
        self.probabilities = p
        self.quantiles = q
        self._n = n
        self._d = d
        self._sf = p if kind == 'sf' else 1 - p
        self._kind = kind

        # Determine leftmost function values
        f0 = 1 if kind == 'sf' else 0  # leftmost function value
        f1 = 1 - f0
        # Prepare for interpolation
        # Insert infinite values at edges to handle edge cases
        x = np.insert(q, [0, len(q)], [-np.inf, np.inf])
        y = np.insert(p, [0, len(p)], [f0, f1])
        # Create an interpolation function using previous value strategy
        self._f = interpolate.interp1d(x, y, kind='previous',
                                       assume_sorted=True)

    def evaluate(self, x):
        """Evaluate the empirical CDF/SF function at the input.

        Parameters
        ----------
        x : ndarray
            Argument to the CDF/SF

        Returns
        -------
        y : ndarray
            The CDF/SF evaluated at the input
        """
        return self._f(x)
    def plot(self, ax=None, **matplotlib_kwargs):
        """Plot the empirical distribution function

        Available only if ``matplotlib`` is installed.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to draw the plot onto, otherwise uses the current Axes.

        **matplotlib_kwargs : dict, optional
            Keyword arguments passed directly to `matplotlib.axes.Axes.step`.
            Unless overridden, ``where='post'``.

        Returns
        -------
        lines : list of `matplotlib.lines.Line2D`
            Objects representing the plotted data
        """
        try:
            import matplotlib  # noqa: F401
        except ModuleNotFoundError as exc:
            message = "matplotlib must be installed to use method `plot`."
            raise ModuleNotFoundError(message) from exc

        # 如果未提供 Axes 对象，则使用当前的 Axes 对象
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()

        # 设置默认的绘图参数，覆盖用户传递的参数
        kwargs = {'where': 'post'}
        kwargs.update(matplotlib_kwargs)

        # 计算超出样本边缘多少距离来绘制
        delta = np.ptp(self.quantiles)*0.05  # how far past sample edge to plot
        q = self.quantiles
        q = [q[0] - delta] + list(q) + [q[-1] + delta]

        # 使用 matplotlib 的 step 方法绘制图形，并返回绘制的线对象
        return ax.step(q, self.evaluate(q), **kwargs)

    def _linear_ci(self, confidence_level):
        sf, d, n = self._sf, self._d, self._n

        # 在 n == d 时，Greenwood's 公式会导致除以零的错误
        # 当 s != 0 时，可以忽略此问题：var == inf，CI 是 [0, 1]
        # 当 s == 0 时，会导致 NaN。生成一个有信息性的警告
        with np.errstate(divide='ignore', invalid='ignore'):
            var = sf ** 2 * np.cumsum(d / (n * (n - d)))

        se = np.sqrt(var)
        z = special.ndtri(1 / 2 + confidence_level / 2)

        z_se = z * se
        low = self.probabilities - z_se
        high = self.probabilities + z_se

        return low, high

    def _loglog_ci(self, confidence_level):
        sf, d, n = self._sf, self._d, self._n

        # 在计算过程中，忽略除以零和无效操作的错误
        with np.errstate(divide='ignore', invalid='ignore'):
            var = 1 / np.log(sf) ** 2 * np.cumsum(d / (n * (n - d)))

        se = np.sqrt(var)
        z = special.ndtri(1 / 2 + confidence_level / 2)

        # 计算 lnL 点
        with np.errstate(divide='ignore'):
            lnl_points = np.log(-np.log(sf))

        z_se = z * se
        low = np.exp(-np.exp(lnl_points + z_se))
        high = np.exp(-np.exp(lnl_points - z_se))

        # 如果是累积分布函数 (CDF)，则转换为区间 [1-high, 1-low]
        if self._kind == "cdf":
            low, high = 1 - high, 1 - low

        return low, high
# 使用 dataclass 装饰器定义一个结果对象 ECDFResult，用于存储 ECDF 函数的结果
@dataclass
class ECDFResult:
    """ Result object returned by `scipy.stats.ecdf`

    Attributes
    ----------
    cdf : `~scipy.stats._result_classes.EmpiricalDistributionFunction`
        An object representing the empirical cumulative distribution function.
    sf : `~scipy.stats._result_classes.EmpiricalDistributionFunction`
        An object representing the complement of the empirical cumulative
        distribution function.
    """
    # 定义两个属性 cdf 和 sf，类型为 EmpiricalDistributionFunction
    cdf: EmpiricalDistributionFunction
    sf: EmpiricalDistributionFunction

    def __init__(self, q, cdf, sf, n, d):
        # 初始化方法，接受 q, cdf, sf, n, d 参数，并用它们创建 EmpiricalDistributionFunction 对象
        self.cdf = EmpiricalDistributionFunction(q, cdf, n, d, "cdf")
        self.sf = EmpiricalDistributionFunction(q, sf, n, d, "sf")


def _iv_CensoredData(
    sample: npt.ArrayLike | CensoredData, param_name: str = 'sample'
) -> CensoredData:
    """Attempt to convert `sample` to `CensoredData`."""
    # 尝试将 sample 转换为 CensoredData 类型，如果不是，则尝试通过 CensoredData 的构造函数进行转换
    if not isinstance(sample, CensoredData):
        try:  # 处理输入的标准化和验证
            sample = CensoredData(uncensored=sample)
        except ValueError as e:
            # 处理错误，将异常消息中的 "uncensored" 替换为 param_name，并抛出新的异常
            message = str(e).replace('uncensored', param_name)
            raise type(e)(message) from e
    return sample


def ecdf(sample: npt.ArrayLike | CensoredData) -> ECDFResult:
    """Empirical cumulative distribution function of a sample.

    The empirical cumulative distribution function (ECDF) is a step function
    estimate of the CDF of the distribution underlying a sample. This function
    returns objects representing both the empirical distribution function and
    its complement, the empirical survival function.

    Parameters
    ----------
    sample : 1D array_like or `scipy.stats.CensoredData`
        Besides array_like, instances of `scipy.stats.CensoredData` containing
        uncensored and right-censored observations are supported. Currently,
        other instances of `scipy.stats.CensoredData` will result in a
        ``NotImplementedError``.

    Returns
    -------
    ECDFResult
        An instance containing the empirical cumulative distribution function
        and its complement.
    """
    res : `~scipy.stats._result_classes.ECDFResult`
        # 定义一个变量 res，表示 ECDF 结果对象，具有以下属性和方法

        cdf : `~scipy.stats._result_classes.EmpiricalDistributionFunction`
            # ECDF 对象，表示经验累积分布函数

        sf : `~scipy.stats._result_classes.EmpiricalDistributionFunction`
            # ECDF 对象，表示经验生存函数

        The `cdf` and `sf` attributes themselves have the following attributes.

        quantiles : ndarray
            # 样本中定义经验CDF/SF的唯一值

        probabilities : ndarray
            # 与 `quantiles` 对应的概率的点估计值

        And the following methods:

        evaluate(x) :
            # 在给定参数 x 处评估CDF/SF

        plot(ax) :
            # 在提供的坐标轴上绘制CDF/SF图像

        confidence_interval(confidence_level=0.95) :
            # 计算 `quantiles` 中值的CDF/SF的置信区间

    Notes
    -----
    # 当样本的每个观察值是精确测量时，ECDF在每个观察点上以 ``1/len(sample)`` 递增 [1]_.

    # 当观测数据是下界、上界或下界和上界时，数据被称为“被审查”，`sample` 可作为 `scipy.stats.CensoredData` 的实例提供。

    # 对于右侧被审查的数据，ECDF由Kaplan-Meier估计量给出 [2]_；目前不支持其他形式的审查。

    # 根据Greenwood公式或更近期的“指数Greenwood”公式计算置信区间，具体描述见 [4]_.

    References
    ----------
    .. [1] Conover, William Jay. Practical nonparametric statistics. Vol. 350.
           John Wiley & Sons, 1999.

    .. [2] Kaplan, Edward L., and Paul Meier. "Nonparametric estimation from
           incomplete observations." Journal of the American statistical
           association 53.282 (1958): 457-481.

    .. [3] Goel, Manish Kumar, Pardeep Khanna, and Jugal Kishore.
           "Understanding survival analysis: Kaplan-Meier estimate."
           International journal of Ayurveda research 1.4 (2010): 274.

    .. [4] Sawyer, Stanley. "The Greenwood and Exponential Greenwood Confidence
           Intervals in Survival Analysis."
           https://www.math.wustl.edu/~sawyer/handouts/greenwood.pdf

    Examples
    --------
    **Uncensored Data**

    As in the example from [1]_ page 79, five boys were selected at random from
    those in a single high school. Their one-mile run times were recorded as
    follows.

    >>> sample = [6.23, 5.58, 7.06, 6.42, 5.20]  # one-mile run times (minutes)

    The empirical distribution function, which approximates the distribution
    function of one-mile run times of the population from which the boys were
    sampled, is calculated as follows.
    # 导入 scipy 中的统计模块
    >>> from scipy import stats
    # 计算经验累积分布函数（ECDF）
    >>> res = stats.ecdf(sample)
    # 获取 ECDF 对象的累积分布函数的分位数数组
    >>> res.cdf.quantiles
    array([5.2 , 5.58, 6.23, 6.42, 7.06])
    # 获取 ECDF 对象的累积分布函数的概率数组
    >>> res.cdf.probabilities
    array([0.2, 0.4, 0.6, 0.8, 1. ])

    # 绘制结果为阶梯函数图
    To plot the result as a step function:
    # 导入 matplotlib 的 pyplot 模块
    >>> import matplotlib.pyplot as plt
    # 创建一个子图
    >>> ax = plt.subplot()
    # 在子图上绘制 ECDF 对象的累积分布函数图
    >>> res.cdf.plot(ax)
    # 设置 x 轴标签
    >>> ax.set_xlabel('One-Mile Run Time (minutes)')
    # 设置 y 轴标签
    >>> ax.set_ylabel('Empirical CDF')
    # 显示图形
    >>> plt.show()

    **Right-censored Data**

    As in the example from [1]_ page 91, the lives of ten car fanbelts were
    tested. Five tests concluded because the fanbelt being tested broke, but
    the remaining tests concluded for other reasons (e.g. the study ran out of
    funding, but the fanbelt was still functional). The mileage driven
    with the fanbelts were recorded as follows.

    # 右截尾数据：在示例中，测试了十个汽车传动带的寿命。五次测试因为传动带断裂而结束，但剩余的测试因其他原因结束（例如研究经费耗尽，但传动带仍然功能正常）。传动带的行驶里程如下记录。
    >>> broken = [77, 47, 81, 56, 80]  # in thousands of miles driven
    >>> unbroken = [62, 60, 43, 71, 37]

    # 那些在测试结束时仍然功能正常的传动带的精确寿命时间未知，但已知超过“unbroken”中记录的数值。因此，这些观察结果被称为“右截尾”，并使用“scipy.stats.CensoredData”来表示数据。
    >>> sample = stats.CensoredData(uncensored=broken, right=unbroken)

    # 计算经验存活函数
    >>> res = stats.ecdf(sample)
    # 获取 ECDF 对象的存活函数的分位数数组
    >>> res.sf.quantiles
    array([37., 43., 47., 56., 60., 62., 71., 77., 80., 81.])
    # 获取 ECDF 对象的存活函数的概率数组
    >>> res.sf.probabilities
    array([1.   , 1.   , 0.875, 0.75 , 0.75 , 0.75 , 0.75 , 0.5  , 0.25 , 0.   ])

    # 绘制结果为阶梯函数图
    To plot the result as a step function:
    # 创建一个子图
    >>> ax = plt.subplot()
    # 在子图上绘制 ECDF 对象的累积分布函数图
    >>> res.cdf.plot(ax)
    # 设置 x 轴标签
    >>> ax.set_xlabel('Fanbelt Survival Time (thousands of miles)')
    # 设置 y 轴标签
    >>> ax.set_ylabel('Empirical SF')
    # 显示图形
    >>> plt.show()

    """
    # 将样本数据转换为 _iv_CensoredData 格式
    sample = _iv_CensoredData(sample)

    # 根据样本数据是否完全未截尾或全部右截尾进行不同的 ECDF 计算处理
    if sample.num_censored() == 0:
        # 如果样本数据完全未截尾，则计算未截尾数据的 ECDF
        res = _ecdf_uncensored(sample._uncensor())
    elif sample.num_censored() == sample._right.size:
        # 如果样本数据全部右截尾，则计算右截尾数据的 ECDF
        res = _ecdf_right_censored(sample)
    else:
        # 如果存在其他截尾选项，提示后续 PR 可支持
        message = ("Currently, only uncensored and right-censored data is "
                   "supported.")
        raise NotImplementedError(message)

    # 从结果中获取时间、累积分布函数、存活函数、样本数量和样本密度，并返回 ECDFResult 对象
    t, cdf, sf, n, d = res
    return ECDFResult(t, cdf, sf, n, d)
# 计算未被截尾的经验累积分布函数（ECDF）
def _ecdf_uncensored(sample):
    # 对样本进行排序
    sample = np.sort(sample)
    # 计算每个唯一值及其出现次数
    x, counts = np.unique(sample, return_counts=True)
    
    # 计算累积事件数
    events = np.cumsum(counts)
    n = sample.size
    # 计算经验累积分布函数（CDF）
    cdf = events / n

    # 计算生存函数（SF），即样本值大于每个唯一值的相对频率
    sf = 1 - cdf
    
    # 计算每个时间点处的“处于风险中”的样本数
    at_risk = np.concatenate(([n], n - events[:-1]))
    
    # 返回结果：唯一值、CDF、SF、处于风险中的样本数、每个唯一值的出现次数
    return x, cdf, sf, at_risk, counts


# 计算右截尾的经验累积分布函数（ECDF）
def _ecdf_right_censored(sample):
    # 根据惯例，讨论右截尾数据时通常使用“生存时间”、“死亡”和“丢失”等术语
    tod = sample._uncensored  # 死亡时间
    tol = sample._right  # 丢失时间
    times = np.concatenate((tod, tol))
    died = np.asarray([1]*tod.size + [0]*tol.size)
    
    # 按时间排序数据
    i = np.argsort(times)
    times = times[i]
    died = died[i]
    at_risk = np.arange(times.size, 0, -1)
    
    # 找到唯一时间点的逻辑索引
    j = np.diff(times, prepend=-np.inf, append=np.inf) > 0
    j_l = j[:-1]  # 唯一时间点的第一个实例
    j_r = j[1:]   # 唯一时间点的最后一个实例
    
    # 获取每个唯一时间点处的处于风险中样本数和死亡数
    t = times[j_l]   # 唯一时间点
    n = at_risk[j_l]  # 每个唯一时间点处的处于风险中样本数
    cd = np.cumsum(died)[j_r]  # 截至每个唯一时间点的累积死亡数
    d = np.diff(cd, prepend=0)  # 每个唯一时间点的死亡数
    
    # 计算生存函数（SF）和累积分布函数（CDF）
    sf = np.cumprod((n - d) / n)
    cdf = 1 - sf
    
    # 返回结果：唯一时间点、CDF、SF、处于风险中的样本数、每个时间点的死亡数
    return t, cdf, sf, n, d


@dataclass
class LogRankResult:
    """`scipy.stats.logrank` 返回的结果对象。

    Attributes
    ----------
    statistic : float ndarray
        计算出的统计量（以下文定义）。其大小是大多数其他logrank测试实现返回的大小的平方根。
    pvalue : float ndarray
        测试的p值。
    """
    statistic: np.ndarray
    pvalue: np.ndarray


# 计算两个样本的生存分布是否有显著差异的logrank检验
def logrank(
    x: npt.ArrayLike | CensoredData,
    y: npt.ArrayLike | CensoredData,
    alternative: Literal['two-sided', 'less', 'greater'] = "two-sided"
) -> LogRankResult:
    r"""通过logrank检验比较两个样本的生存分布。

    Parameters
    ----------
    x, y : array_like or CensoredData
        要比较的样本，基于它们的经验生存函数。
    alternative : {'two-sided', 'less', 'greater'}, optional
        指定检验的双侧或单侧性质，默认为"two-sided"。
    alternative : {'two-sided', 'less', 'greater'}, optional
        # 定义备择假设的选择项。

        # 原假设是两组（例如 *X* 和 *Y*）的生存分布相同。

        # 下列备择假设 [4]_ 可供选择（默认为 'two-sided'）：

        # * 'two-sided': 两组的生存分布不相同。
        # * 'less': 组 *X* 的生存更有利：组 *X* 的失效率函数在某些时间点上小于组 *Y* 的失效率函数。
        # * 'greater': 组 *Y* 的生存更有利：组 *X* 的失效率函数在某些时间点上大于组 *Y* 的失效率函数。

    Returns
    -------
    res : `~scipy.stats._result_classes.LogRankResult`
        # 包含以下属性的对象：

        statistic : float ndarray
            # 计算得到的统计量（下文有定义）。其大小是大多数其他logrank测试实现返回值的平方根。
        pvalue : float ndarray
            # 测试的p值。

    See Also
    --------
    scipy.stats.ecdf

    Notes
    -----
    # logrank测试 [1]_ 比较观察到的事件数量和在两个样本从相同分布中抽取的原假设下预期的事件数量。
    # 统计量为

    .. math::

        Z_i = \frac{\sum_{j=1}^J(O_{i,j}-E_{i,j})}{\sqrt{\sum_{j=1}^J V_{i,j}}}
        \rightarrow \mathcal{N}(0,1)

    # 其中

    .. math::

        E_{i,j} = O_j \frac{N_{i,j}}{N_j},
        \qquad
        V_{i,j} = E_{i,j} \left(\frac{N_j-O_j}{N_j}\right)
        \left(\frac{N_j-N_{i,j}}{N_j-1}\right),

    # :math:`i` 表示组（例如可能是 :math:`x` 或 :math:`y`，或者可以省略以指代组合样本）
    # :math:`j` 表示时间（事件发生的时间）
    # :math:`N` 是事件发生前的风险人数，:math:`O` 是该时间点的观察到的事件数。

    # `logrank` 返回的 `statistic` :math:`Z_x` 是许多其他实现返回的统计量的（带符号的）平方根。
    # 在原假设下，:math:`Z_x**2` 渐近地按照自由度为一的卡方分布分布。
    # 因此，:math:`Z_x` 渐近地按照标准正态分布分布。
    # 使用 :math:`Z_x` 的优势在于保留了符号信息（即观察到的事件数量是否倾向于少于或多于原假设下预期的数量），允许 `scipy.stats.logrank` 提供单侧备择假设。

    References
    ----------
    """
    # 引用文献 [2] 比较了两种不同类型复发性恶性胶质瘤患者的生存时间。
    # 下面的样本记录了每位患者参与研究的时间（以周为单位）。
    # 使用 `scipy.stats.CensoredData` 类处理右截尾的数据：
    # 未截尾观测对应观察到的死亡事件，而截尾观测对应患者由于其他原因退出研究。

    >>> from scipy import stats
    >>> x = stats.CensoredData(
    ...     uncensored=[6, 13, 21, 30, 37, 38, 49, 50,
    ...                 63, 79, 86, 98, 202, 219],
    ...     right=[31, 47, 80, 82, 82, 149]
    ... )
    >>> y = stats.CensoredData(
    ...     uncensored=[10, 10, 12, 13, 14, 15, 16, 17, 18, 20, 24, 24,
    ...                 25, 28,30, 33, 35, 37, 40, 40, 46, 48, 76, 81,
    ...                 82, 91, 112, 181],
    ...     right=[34, 40, 70]
    ... )

    # 我们可以计算并可视化两组的经验生存函数如下。

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> ax = plt.subplot()
    >>> ecdf_x = stats.ecdf(x)
    >>> ecdf_x.sf.plot(ax, label='Astrocytoma')
    >>> ecdf_y = stats.ecdf(y)
    >>> ecdf_y.sf.plot(ax, label='Glioblastoma')
    >>> ax.set_xlabel('Time to death (weeks)')
    >>> ax.set_ylabel('Empirical SF')
    >>> plt.legend()
    >>> plt.show()

    # 通过视觉检查经验生存函数，我们发现两组的生存时间倾向于不同。
    # 为了正式评估这种差异在 1% 的显著水平下是否重要，我们使用对数秩检验。

    >>> res = stats.logrank(x=x, y=y)
    >>> res.statistic
    -2.73799
    >>> res.pvalue
    0.00618

    # p 值小于 1%，因此我们可以认为数据支持备择假设，即两个生存函数之间存在差异。

    """
    # 输入验证。`alternative` IV 在下面的 `_get_pvalue` 中处理。
    x = _iv_CensoredData(sample=x, param_name='x')
    y = _iv_CensoredData(sample=y, param_name='y')

    # 合并样本。在零假设下，两组是相同的。
    xy = CensoredData(
        uncensored=np.concatenate((x._uncensored, y._uncensored)),
        right=np.concatenate((x._right, y._right))
    )

    # 创建一个合并后的被审查数据集 `xy`，包括未审查的数据和右侧截尾数据的合并
    res = ecdf(xy)
    # 计算经验累积分布函数（ECDF）`res`，用于后续分析
    idx = res.sf._d.astype(bool)  # 获取观察到事件的索引
    times_xy = res.sf.quantiles[idx]  # 获得观察到事件的唯一时间点
    at_risk_xy = res.sf._n[idx]  # 合并样本中在风险中的主体数
    deaths_xy = res.sf._d[idx]  # 合并样本中发生事件的次数

    # 计算每个样本中在风险中的人数。
    # 首先计算组 X 中每个 `times_xy` 时间点的风险人数。
    # 可以使用 `interpolate_1d`，但这样更为紧凑。
    res_x = ecdf(x)
    i = np.searchsorted(res_x.sf.quantiles, times_xy)
    at_risk_x = np.append(res_x.sf._n, 0)[i]  # 在最后一个时间点后风险人数为 0
    # 从合并风险中减去 X 组的风险人数以获得 Y 组的风险人数
    at_risk_y = at_risk_xy - at_risk_x

    # 计算方差。
    num = at_risk_x * at_risk_y * deaths_xy * (at_risk_xy - deaths_xy)
    den = at_risk_xy**2 * (at_risk_xy - 1)
    # 注意：当 `at_risk_xy == 1` 时，分子和分母中的 `at_risk_xy - 1` 都会为 0。
    # 在符号化简分数时，我们总是会得到整体商为零，因此不计算它。
    i = at_risk_xy > 1
    sum_var = np.sum(num[i]/den[i])

    # 获取组 X 中观察到和预期的死亡人数
    n_died_x = x._uncensored.size
    sum_exp_deaths_x = np.sum(at_risk_x * (deaths_xy/at_risk_xy))

    # 计算统计量。这是参考文献中所述统计量的平方根。
    statistic = (n_died_x - sum_exp_deaths_x)/np.sqrt(sum_var)

    # 当 alternative='two-sided' 时，相当于 chi2(df=1).sf(statistic**2)
    norm = stats._stats_py._SimpleNormal()
    pvalue = stats._stats_py._get_pvalue(statistic, norm, alternative, xp=np)

    return LogRankResult(statistic=statistic[()], pvalue=pvalue[()])
```