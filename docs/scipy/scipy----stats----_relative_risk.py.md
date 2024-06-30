# `D:\src\scipysrc\scipy\scipy\stats\_relative_risk.py`

```
# 导入运算符模块，用于类型转换
import operator
# 导入数据类模块，用于定义数据类
from dataclasses import dataclass
# 导入 NumPy 库，用于科学计算
import numpy as np
# 导入 SciPy 库的 ndtri 函数，用于正态分布的逆累积分布函数
from scipy.special import ndtri
# 导入本地模块中的 ConfidenceInterval 类
from ._common import ConfidenceInterval


def _validate_int(n, bound, name):
    # 定义错误消息，说明参数应为大于等于指定边界的整数
    msg = f'{name} must be an integer not less than {bound}, but got {n!r}'
    try:
        # 尝试将 n 转换为整数
        n = operator.index(n)
    except TypeError:
        # 如果转换失败，抛出类型错误，附带上述错误消息
        raise TypeError(msg) from None
    if n < bound:
        # 如果 n 小于指定边界，抛出值错误，附带上述错误消息
        raise ValueError(msg)
    # 返回验证后的整数 n
    return n


@dataclass
class RelativeRiskResult:
    """
    Result of `scipy.stats.contingency.relative_risk`.

    Attributes
    ----------
    relative_risk : float
        This is::

            (exposed_cases/exposed_total) / (control_cases/control_total)

        相对风险，表示暴露组病例率与非暴露组病例率的比值。

    exposed_cases : int
        The number of "cases" (i.e. occurrence of disease or other event
        of interest) among the sample of "exposed" individuals.
        暴露组中发生疾病或其他感兴趣事件的病例数。

    exposed_total : int
        The total number of "exposed" individuals in the sample.
        样本中暴露组的总人数。

    control_cases : int
        The number of "cases" among the sample of "control" or non-exposed
        individuals.
        非暴露组中发生疾病或其他事件的病例数。

    control_total : int
        The total number of "control" individuals in the sample.
        样本中非暴露组的总人数。

    Methods
    -------
    confidence_interval :
        Compute the confidence interval for the relative risk estimate.
        计算相对风险估计的置信区间。
    """
    def confidence_interval(self, confidence_level=0.95):
        """
        Compute the confidence interval for the relative risk.

        The confidence interval is computed using the Katz method
        (i.e. "Method C" of [1]_; see also [2]_, section 3.1.2).

        Parameters
        ----------
        confidence_level : float, optional
            The confidence level to use for the confidence interval.
            Default is 0.95.

        Returns
        -------
        ci : ConfidenceInterval instance
            The return value is an object with attributes ``low`` and
            ``high`` that hold the confidence interval.

        References
        ----------
        .. [1] D. Katz, J. Baptista, S. P. Azen and M. C. Pike, "Obtaining
               confidence intervals for the risk ratio in cohort studies",
               Biometrics, 34, 469-474 (1978).
        .. [2] Hardeo Sahai and Anwer Khurshid, Statistics in Epidemiology,
               CRC Press LLC, Boca Raton, FL, USA (1996).


        Examples
        --------
        >>> from scipy.stats.contingency import relative_risk
        >>> result = relative_risk(exposed_cases=10, exposed_total=75,
        ...                        control_cases=12, control_total=225)
        >>> result.relative_risk
        2.5
        >>> result.confidence_interval()
        ConfidenceInterval(low=1.1261564003469628, high=5.549850800541033)
        """
        
        # Check if confidence_level is within the valid range [0, 1]
        if not 0 <= confidence_level <= 1:
            raise ValueError('confidence_level must be in the interval '
                             '[0, 1].')

        # Handle special cases where either exposed_cases or control_cases is zero
        # Returns a ConfidenceInterval object with appropriate bounds
        if self.exposed_cases == 0 and self.control_cases == 0:
            return ConfidenceInterval(low=np.nan, high=np.nan)  # Relative risk is nan
        elif self.exposed_cases == 0:
            return ConfidenceInterval(low=0.0, high=np.nan)  # Relative risk is 0
        elif self.control_cases == 0:
            return ConfidenceInterval(low=np.nan, high=np.inf)  # Relative risk is inf

        # Calculate alpha (significance level) and z-score for the confidence interval
        alpha = 1 - confidence_level
        z = ndtri(1 - alpha/2)
        rr = self.relative_risk

        # Estimate standard error (se) based on exposed and control group sizes
        se = np.sqrt(1/self.exposed_cases - 1/self.exposed_total +
                     1/self.control_cases - 1/self.control_total)

        # Calculate the delta for confidence interval bounds
        delta = z * se

        # Compute the lower and upper bounds of the confidence interval using the Katz method
        katz_lo = rr * np.exp(-delta)
        katz_hi = rr * np.exp(delta)

        # Return a ConfidenceInterval object with computed bounds
        return ConfidenceInterval(low=katz_lo, high=katz_hi)
# 计算相对风险（又称风险比）

# 这个函数计算与2x2列联表相关联的相对风险（relative risk）。
# 相对风险是指在“暴露”个体样本中发生疾病或其他感兴趣事件的“病例”数与整体“暴露”个体样本总数之比，
# 与“对照”个体样本中发生疾病或事件的“病例”数与整体“对照”个体样本总数之比之比值。
# 这个函数不接受表格作为参数，而是分别接受用于计算相对风险的各个数字，以避免不清楚列联表的哪一行或列对应于“暴露”或“对照”情况。
# 与例如“比率比”不同，“相对风险”在交换行或列时不具不变性。

def relative_risk(exposed_cases, exposed_total, control_cases, control_total):
    """
    Compute the relative risk (also known as the risk ratio).

    This function computes the relative risk associated with a 2x2
    contingency table ([1]_, section 2.2.3; [2]_, section 3.1.2). Instead
    of accepting a table as an argument, the individual numbers that are
    used to compute the relative risk are given as separate parameters.
    This is to avoid the ambiguity of which row or column of the contingency
    table corresponds to the "exposed" cases and which corresponds to the
    "control" cases.  Unlike, say, the odds ratio, the relative risk is not
    invariant under an interchange of the rows or columns.

    Parameters
    ----------
    exposed_cases : nonnegative int
        The number of "cases" (i.e. occurrence of disease or other event
        of interest) among the sample of "exposed" individuals.
    exposed_total : positive int
        The total number of "exposed" individuals in the sample.
    control_cases : nonnegative int
        The number of "cases" among the sample of "control" or non-exposed
        individuals.
    control_total : positive int
        The total number of "control" individuals in the sample.

    Returns
    -------
    result : instance of `~scipy.stats._result_classes.RelativeRiskResult`
        The object has the float attribute ``relative_risk``, which is::

            rr = (exposed_cases/exposed_total) / (control_cases/control_total)

        The object also has the method ``confidence_interval`` to compute
        the confidence interval of the relative risk for a given confidence
        level.

    See Also
    --------
    odds_ratio

    Notes
    -----
    The R package epitools has the function `riskratio`, which accepts
    a table with the following layout::

                        disease=0   disease=1
        exposed=0 (ref)    n00         n01
        exposed=1          n10         n11

    With a 2x2 table in the above format, the estimate of the CI is
    computed by `riskratio` when the argument method="wald" is given,
    or with the function `riskratio.wald`.

    For example, in a test of the incidence of lung cancer among a
    sample of smokers and nonsmokers, the "exposed" category would
    correspond to "is a smoker" and the "disease" category would
    correspond to "has or had lung cancer".

    To pass the same data to ``relative_risk``, use::

        relative_risk(n11, n10 + n11, n01, n00 + n01)

    .. versionadded:: 1.7.0

    References
    ----------
    .. [1] Alan Agresti, An Introduction to Categorical Data Analysis
           (second edition), Wiley, Hoboken, NJ, USA (2007).
    .. [2] Hardeo Sahai and Anwer Khurshid, Statistics in Epidemiology,
           CRC Press LLC, Boca Raton, FL, USA (1996).

    Examples
    --------
    >>> from scipy.stats.contingency import relative_risk

    This example is from Example 3.1 of [2]_.  The results of a heart
    """
    Relative risk is a trivial calculation.  The nontrivial part is in the
    `confidence_interval` method of the RelativeRiskResult class.

    Validate and ensure `exposed_cases` is a non-negative integer.
    """
    exposed_cases = _validate_int(exposed_cases, 0, "exposed_cases")

    """
    Validate and ensure `exposed_total` is a positive integer.
    """
    exposed_total = _validate_int(exposed_total, 1, "exposed_total")

    """
    Validate and ensure `control_cases` is a non-negative integer.
    """
    control_cases = _validate_int(control_cases, 0, "control_cases")

    """
    Validate and ensure `control_total` is a positive integer.
    """
    control_total = _validate_int(control_total, 1, "control_total")

    """
    Ensure `exposed_cases` does not exceed `exposed_total`.
    """
    if exposed_cases > exposed_total:
        raise ValueError('exposed_cases must not exceed exposed_total.')

    """
    Ensure `control_cases` does not exceed `control_total`.
    """
    if control_cases > control_total:
        raise ValueError('control_cases must not exceed control_total.')

    """
    Calculate relative risk based on the validated values of cases and totals.

    If both exposed and control cases are zero, relative risk is NaN.
    If only exposed cases are zero, relative risk is 0.0.
    If only control cases are zero, relative risk is infinite.
    Otherwise, compute the relative risk ratio.
    """
    if exposed_cases == 0 and control_cases == 0:
        rr = np.nan
    elif exposed_cases == 0:
        rr = 0.0
    elif control_cases == 0:
        rr = np.inf
    else:
        p1 = exposed_cases / exposed_total
        p2 = control_cases / control_total
        rr = p1 / p2

    """
    Return an instance of RelativeRiskResult containing the calculated relative risk,
    along with the validated counts of exposed and control cases and totals.
    """
    return RelativeRiskResult(relative_risk=rr,
                              exposed_cases=exposed_cases,
                              exposed_total=exposed_total,
                              control_cases=control_cases,
                              control_total=control_total)
```