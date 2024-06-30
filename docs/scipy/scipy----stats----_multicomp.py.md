# `D:\src\scipysrc\scipy\scipy\stats\_multicomp.py`

```
from __future__ import annotations
# 允许在类中使用注解的特性，用于类型提示和类型检查

import warnings
# 导入警告模块，用于处理警告信息

from dataclasses import dataclass, field
# 导入数据类相关模块，用于定义数据类和字段属性

from typing import TYPE_CHECKING
# 导入类型提示相关模块，用于类型检查

import numpy as np
# 导入NumPy库，用于数值计算

from scipy import stats
# 导入SciPy统计模块

from scipy.optimize import minimize_scalar
# 导入SciPy中的一维最小化函数

from scipy.stats._common import ConfidenceInterval
# 导入SciPy中的置信区间类

from scipy.stats._qmc import check_random_state
# 导入SciPy中的随机数状态检查函数

from scipy.stats._stats_py import _var
# 导入SciPy中的统计计算函数

if TYPE_CHECKING:
    import numpy.typing as npt
    from scipy._lib._util import DecimalNumber, SeedType
    from typing import Literal, Sequence  # noqa: UP035
    # 如果是类型检查模式，则导入特定的类型提示

__all__ = [
    'dunnett'
]
# 导出的符号列表，包含模块提供的公共接口名称

@dataclass
# 使用数据类装饰器定义类
class DunnettResult:
    """Result object returned by `scipy.stats.dunnett`.

    Attributes
    ----------
    statistic : float ndarray
        The computed statistic of the test for each comparison. The element
        at index ``i`` is the statistic for the comparison between
        groups ``i`` and the control.
    pvalue : float ndarray
        The computed p-value of the test for each comparison. The element
        at index ``i`` is the p-value for the comparison between
        group ``i`` and the control.
    """
    statistic: np.ndarray
    # 统计量数组，存储每个比较的测试统计量
    pvalue: np.ndarray
    # P值数组，存储每个比较的测试P值
    _alternative: Literal['two-sided', 'less', 'greater'] = field(repr=False)
    # 可选属性，指定假设检验的备择假设类型
    _rho: np.ndarray = field(repr=False)
    # 可选属性，用于存储相关系数数组
    _df: int = field(repr=False)
    # 可选属性，自由度数值
    _std: float = field(repr=False)
    # 可选属性，标准差值
    _mean_samples: np.ndarray = field(repr=False)
    # 可选属性，样本均值数组
    _mean_control: np.ndarray = field(repr=False)
    # 可选属性，控制组均值数组
    _n_samples: np.ndarray = field(repr=False)
    # 可选属性，样本数数组
    _n_control: int = field(repr=False)
    # 可选属性，控制组样本数
    _rng: SeedType = field(repr=False)
    # 可选属性，随机数生成器种子类型
    _ci: ConfidenceInterval | None = field(default=None, repr=False)
    # 可选属性，置信区间对象或None
    _ci_cl: DecimalNumber | None = field(default=None, repr=False)
    # 可选属性，置信水平或None

    def __str__(self):
        # 定义对象的字符串表示形式
        # 注意：`__str__` 方法打印最近一次调用 `confidence_interval` 方法生成的置信区间
        # 如果没有调用过，则使用默认置信水平0.95进行调用
        if self._ci is None:
            self.confidence_interval(confidence_level=.95)
        s = (
            "Dunnett's test"
            f" ({self._ci_cl*100:.1f}% Confidence Interval)\n"
            "Comparison               Statistic  p-value  Lower CI  Upper CI\n"
        )
        for i in range(self.pvalue.size):
            s += (f" (Sample {i} - Control) {self.statistic[i]:>10.3f}"
                  f"{self.pvalue[i]:>10.3f}"
                  f"{self._ci.low[i]:>10.3f}"
                  f"{self._ci.high[i]:>10.3f}\n")
        return s
    # 返回格式化后的Dunnett's test结果字符串，包含统计量、P值和置信区间上下限

    def _allowance(
        self, confidence_level: DecimalNumber = 0.95, tol: DecimalNumber = 1e-3
        # 定义方法_allowance，用于设置置信水平和容差
    ) -> float:
        """计算容差值。

        这是要从观察组的均值与对照组均值之间的差异中加/减的数量。结果提供置信区间。

        Parameters
        ----------
        confidence_level : float, optional
            计算置信区间的置信水平，默认为0.95。
        tol : float, optional
            数值优化的容差：容差将确保置信水平在指定水平的“10*tol*(1 - confidence_level)”内，否则会发出警告。
            由于目标函数的嘈杂评估，过于紧密的容差可能不切实际。
            默认为1e-3。

        Returns
        -------
        allowance : float
            均值周围的容差。
        """
        alpha = 1 - confidence_level

        def pvalue_from_stat(statistic):
            statistic = np.array(statistic)
            sf = _pvalue_dunnett(
                rho=self._rho, df=self._df,
                statistic=statistic, alternative=self._alternative,
                rng=self._rng
            )
            return abs(sf - alpha)/alpha

        # 由于使用RQMC评估multivariate_t.cdf，pvalue_from_stat的评估嘈杂。
        # `minimize_scalar`不设计用于容忍嘈杂的目标函数，可能无法准确找到最小值。
        # 我们通过下面的验证步骤来减轻这种可能性，但是实现一个容忍噪声的根查找器或最小化器将是一个值得欢迎的增强。
        # 参见 gh-18150。
        res = minimize_scalar(pvalue_from_stat, method='brent', tol=tol)
        critical_value = res.x

        # 验证步骤
        # tol*10是因为tol=1e-3意味着我们最多容忍1%的变化
        if res.success is False or res.fun >= tol*10:
            warnings.warn(
                "未能将置信区间的计算收敛到期望水平。返回区间对应的置信水平约为 "
                f"{alpha*(1+res.fun)}。",
                stacklevel=3
            )

        # 来自 [1] p. 1101 between (1) and (3)
        allowance = critical_value*self._std*np.sqrt(
            1/self._n_samples + 1/self._n_control
        )
        return abs(allowance)

    def confidence_interval(
        self, confidence_level: DecimalNumber = 0.95
    ) -> ConfidenceInterval:
        """Compute the confidence interval for the specified confidence level.

        Parameters
        ----------
        confidence_level : float, optional
            Confidence level for the computed confidence interval.
            Default is .95.

        Returns
        -------
        ci : ``ConfidenceInterval`` object
            The object has attributes ``low`` and ``high`` that hold the
            lower and upper bounds of the confidence intervals for each
            comparison. The high and low values are accessible for each
            comparison at index ``i`` for each group ``i``.

        """
        # 检查提供的置信水平是否与先前计算的置信水平一致，如果一致则直接返回已经计算的置信区间对象
        if (self._ci is not None) and (confidence_level == self._ci_cl):
            return self._ci

        # 如果置信水平不在 (0, 1) 之间，则抛出 ValueError 异常
        if not (0 < confidence_level < 1):
            raise ValueError("Confidence level must be between 0 and 1.")

        # 计算置信区间的宽度 allowance
        allowance = self._allowance(confidence_level=confidence_level)
        
        # 计算样本均值之差
        diff_means = self._mean_samples - self._mean_control

        # 计算置信区间的下限和上限
        low = diff_means - allowance
        high = diff_means + allowance

        # 根据备择假设的类型调整置信区间的边界
        if self._alternative == 'greater':
            high = [np.inf] * len(diff_means)  # 对于 'greater'，将上限设为正无穷大
        elif self._alternative == 'less':
            low = [-np.inf] * len(diff_means)  # 对于 'less'，将下限设为负无穷大

        # 更新对象的内部状态：置信水平和置信区间对象
        self._ci_cl = confidence_level
        self._ci = ConfidenceInterval(
            low=low,
            high=high
        )
        
        # 返回计算得到的置信区间对象
        return self._ci
# 定义 Dunnett's test 函数，用于对多个样本组的均值与一个控制组进行多重比较

def dunnett(
    *samples: npt.ArrayLike,  # noqa: D417
    control: npt.ArrayLike,
    alternative: Literal['two-sided', 'less', 'greater'] = "two-sided",
    random_state: SeedType = None
) -> DunnettResult:
    """Dunnett's test: multiple comparisons of means against a control group.

    This is an implementation of Dunnett's original, single-step test as
    described in [1]_.

    Parameters
    ----------
    *samples : 1D array_like
        样本测量值，每个实验组一个参数。
    control : 1D array_like
        控制组的样本测量值。
    alternative : {'two-sided', 'less', 'greater'}, optional
        定义备择假设。

        零假设是样本和控制组的分布均值相等。可选的备择假设包括（默认为 'two-sided'）：

        * 'two-sided': 样本和控制组的分布均值不相等。
        * 'less': 样本的分布均值小于控制组的分布均值。
        * 'greater': 样本的分布均值大于控制组的分布均值。
    random_state : {None, int, `numpy.random.Generator`}, optional
        如果 `random_state` 是 int 或 None，则使用 ``np.random.default_rng(random_state)`` 创建一个新的 `numpy.random.Generator`。
        如果 `random_state` 已经是 ``Generator`` 实例，则直接使用该实例。

        随机数生成器用于控制多元 t 分布的随机化拟合。

    Returns
    -------
    res : `~scipy.stats._result_classes.DunnettResult`
        包含以下属性的对象：

        statistic : float ndarray
            每个比较的测试统计量。索引为 ``i`` 的元素是组 ``i`` 与控制组之间的统计量。
        pvalue : float ndarray
            每个比较的 p 值。索引为 ``i`` 的元素是组 ``i`` 与控制组之间的 p 值。

        还包含以下方法：

        confidence_interval(confidence_level=0.95) :
            计算每个组与控制组的均值差异的置信区间。

    See Also
    --------
    tukey_hsd : 执行成对均值比较。

    Notes
    -----
    类似于独立样本 t 检验，Dunnett's test [1]_ 用于推断抽样分布的均值。然而，当在固定显著性水平下执行多个 t 检验时，
    "家族误差率" - 错误拒绝零假设的概率 - 可能会增加。

    [1] Dunnett, C. W. (1955). A multiple comparison procedure for comparing several treatments with a control.
    # 导入 Dunnett's test 方法从 scipy.stats 模块
    from scipy.stats import dunnett

    # 使用 Dunnett's test 检验 drug_a 和 drug_b 组的平均值与控制组 control 是否显著不同，
    # 控制家族误差率为5%。零假设是实验组与控制组的均值相同，备择假设是某个实验组与控制组的均值不同。
    res = dunnett(drug_a, drug_b, control=control)
    >>> res.pvalue
    array([0.62004941, 0.0059035 ])  # may vary
    # 打印出来的 p 值数组，这里是示例，实际数值可能会有所不同

    The p-value corresponding with the comparison between group A and control
    exceeds 0.05, so we do not reject the null hypothesis for that comparison.
    However, the p-value corresponding with the comparison between group B
    and control is less than 0.05, so we consider the experimental results
    to be evidence against the null hypothesis in favor of the alternative:
    group B has a different mean than the control group.

    """
    # 基于 Dunnett 方法进行多组间比较

    samples_, control_, rng = _iv_dunnett(
        samples=samples, control=control,
        alternative=alternative, random_state=random_state
    )
    # 调用 _iv_dunnett 函数计算处理后的样本、控制组及随机数生成器

    rho, df, n_group, n_samples, n_control = _params_dunnett(
        samples=samples_, control=control_
    )
    # 调用 _params_dunnett 函数计算参数 rho（相关性系数）、自由度 df、组数、样本数和控制组数

    statistic, std, mean_control, mean_samples = _statistic_dunnett(
        samples_, control_, df, n_samples, n_control
    )
    # 调用 _statistic_dunnett 函数计算统计量、标准差、样本组均值和控制组均值

    pvalue = _pvalue_dunnett(
        rho=rho, df=df, statistic=statistic, alternative=alternative, rng=rng
    )
    # 调用 _pvalue_dunnett 函数计算 p 值

    return DunnettResult(
        statistic=statistic, pvalue=pvalue,
        _alternative=alternative,
        _rho=rho, _df=df, _std=std,
        _mean_samples=mean_samples,
        _mean_control=mean_control,
        _n_samples=n_samples,
        _n_control=n_control,
        _rng=rng
    )
    # 返回 DunnettResult 对象，包含统计量、p 值及其他相关参数
# 输入验证函数，用于检查输入参数是否符合Dunnett's测试的要求
def _iv_dunnett(
    samples: Sequence[npt.ArrayLike],
    control: npt.ArrayLike,
    alternative: Literal['two-sided', 'less', 'greater'],
    random_state: SeedType
) -> tuple[list[np.ndarray], np.ndarray, SeedType]:
    """Input validation for Dunnett's test."""
    # 使用提供的随机种子初始化随机数生成器
    rng = check_random_state(random_state)

    # 检查alternative参数是否在预定义的集合中，如果不在则抛出ValueError异常
    if alternative not in {'two-sided', 'less', 'greater'}:
        raise ValueError(
            "alternative must be 'less', 'greater' or 'two-sided'"
        )

    # 控制信息消息，用于验证样本和控制组的维度和大小
    ndim_msg = "Control and samples groups must be 1D arrays"
    n_obs_msg = "Control and samples groups must have at least 1 observation"

    # 将控制组和样本转换为NumPy数组
    control = np.asarray(control)
    samples_ = [np.asarray(sample) for sample in samples]

    # 对所有样本（包括控制组）进行维度和大小的检查
    samples_control: list[np.ndarray] = samples_ + [control]
    for sample in samples_control:
        # 如果样本维度大于1，则引发ValueError异常
        if sample.ndim > 1:
            raise ValueError(ndim_msg)
        # 如果样本大小小于1，则引发ValueError异常
        if sample.size < 1:
            raise ValueError(n_obs_msg)

    # 返回验证后的样本列表、控制组和随机数生成器
    return samples_, control, rng


# 返回Dunnett's测试的特定参数
def _params_dunnett(
    samples: list[np.ndarray], control: np.ndarray
) -> tuple[np.ndarray, int, int, np.ndarray, int]:
    """Specific parameters for Dunnett's test.

    Degree of freedom is the number of observations minus the number of groups
    including the control.
    """
    # 计算每个样本组的大小
    n_samples = np.array([sample.size for sample in samples])

    # 计算总样本数、控制组大小及总群组数
    n_sample = n_samples.sum()
    n_control = control.size
    n = n_sample + n_control
    n_groups = len(samples)
    df = n - n_groups - 1

    # 计算相关系数矩阵rho，用于多元t分布的计算
    rho = n_control/n_samples + 1
    rho = 1/np.sqrt(rho[:, None] * rho[None, :])
    np.fill_diagonal(rho, 1)

    # 返回rho矩阵、自由度、群组数、每个样本组的大小及控制组大小
    return rho, df, n_groups, n_samples, n_control


# 返回Dunnett's测试的统计量
def _statistic_dunnett(
    samples: list[np.ndarray], control: np.ndarray, df: int,
    n_samples: np.ndarray, n_control: int
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Statistic of Dunnett's test.

    Computation based on the original single-step test from [1].
    """
    # 计算控制组和样本组的平均值
    mean_control = np.mean(control)
    mean_samples = np.array([np.mean(sample) for sample in samples])
    all_samples = [control] + samples
    all_means = np.concatenate([[mean_control], mean_samples])

    # 估算方差s^2，用于计算标准差
    s2 = np.sum([_var(sample, mean=mean)*sample.size
                 for sample, mean in zip(all_samples, all_means)]) / df
    std = np.sqrt(s2)

    # 计算z得分，用于后续的t分布计算
    z = (mean_samples - mean_control) / np.sqrt(1/n_samples + 1/n_control)

    # 返回标准化的z得分、标准差、控制组平均值和样本组平均值
    return z / std, std, mean_control, mean_samples


# 返回Dunnett's测试的p值，基于多元t分布
def _pvalue_dunnett(
    rho: np.ndarray, df: int, statistic: np.ndarray,
    alternative: Literal['two-sided', 'less', 'greater'],
    rng: SeedType = None
) -> np.ndarray:
    """pvalue from the multivariate t-distribution.

    Critical values come from the multivariate student-t distribution.
    """
    statistic = statistic.reshape(-1, 1)
    # 使用 stats 模块中的 multivariate_t 函数创建一个多变量 t 分布对象
    mvt = stats.multivariate_t(shape=rho, df=df, seed=rng)
    
    # 如果 alternative 参数为 "two-sided"，则取统计量 statistic 的绝对值
    if alternative == "two-sided":
        statistic = abs(statistic)
        # 计算双尾检验的 p 值，通过 1 减去多变量 t 分布的累积分布函数
        pvalue = 1 - mvt.cdf(statistic, lower_limit=-statistic)
    # 如果 alternative 参数为 "greater"
    elif alternative == "greater":
        # 计算单尾检验（大于型）的 p 值，通过 1 减去多变量 t 分布的累积分布函数
        pvalue = 1 - mvt.cdf(statistic, lower_limit=-np.inf)
    # 如果 alternative 参数既不是 "two-sided" 也不是 "greater"
    else:
        # 计算单尾检验（小于型）的 p 值，通过 1 减去多变量 t 分布的累积分布函数
        pvalue = 1 - mvt.cdf(np.inf, lower_limit=statistic)
    
    # 返回至少是一维的 p 值数组
    return np.atleast_1d(pvalue)
```