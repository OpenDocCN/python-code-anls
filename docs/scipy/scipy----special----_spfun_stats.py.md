# `D:\src\scipysrc\scipy\scipy\special\_spfun_stats.py`

```
# 导入所需库：NumPy用于数值计算，scipy.special中的gammaln函数用于计算对数Gamma函数
import numpy as np
from scipy.special import gammaln as loggam

# 定义在多元统计分析中可能有用的特殊函数

__all__ = ['multigammaln']

# 定义函数multigammaln，计算对数多元Gamma函数（也称为广义Gamma函数）

def multigammaln(a, d):
    r"""Returns the log of multivariate gamma, also sometimes called the
    generalized gamma.

    Parameters
    ----------
    a : ndarray
        The multivariate gamma is computed for each item of `a`.
    d : int
        The dimension of the space of integration.

    Returns
    -------
    res : ndarray
        The values of the log multivariate gamma at the given points `a`.

    Notes
    -----
    The formal definition of the multivariate gamma of dimension d for a real
    `a` is

    .. math::

        \Gamma_d(a) = \int_{A>0} e^{-tr(A)} |A|^{a - (d+1)/2} dA

    with the condition :math:`a > (d-1)/2`, and :math:`A > 0` being the set of
    all the positive definite matrices of dimension `d`.  Note that `a` is a
    scalar: the integrand only is multivariate, the argument is not (the
    function is defined over a subset of the real set).

    This can be proven to be equal to the much friendlier equation

    .. math::

        \Gamma_d(a) = \pi^{d(d-1)/4} \prod_{i=1}^{d} \Gamma(a - (i-1)/2).

    References
    ----------
    R. J. Muirhead, Aspects of multivariate statistical theory (Wiley Series in
    probability and mathematical statistics).
    """
    # 返回在给定点a处的对数多元Gamma函数值
    return loggam(a) * (d * (d - 1) // 4) + np.sum(loggam(a - np.arange(d) / 2.0))
    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import multigammaln, gammaln
    >>> a = 23.5  # 设定参数 a 为 23.5
    >>> d = 10  # 设定参数 d 为 10
    >>> multigammaln(a, d)  # 调用 multigammaln 函数计算结果，应为 454.1488605074416
    454.1488605074416

    Verify that the result agrees with the logarithm of the equation
    shown above:

    >>> d*(d-1)/4*np.log(np.pi) + gammaln(a - 0.5*np.arange(0, d)).sum()
    454.1488605074416
    """
    a = np.asarray(a)  # 将参数 a 转换为 numpy 数组
    if not np.isscalar(d) or (np.floor(d) != d):  # 检查参数 d 是否为正整数
        raise ValueError("d should be a positive integer (dimension)")
    if np.any(a <= 0.5 * (d - 1)):  # 检查条件是否满足
        raise ValueError(f"condition a ({a:f}) > 0.5 * (d-1) ({0.5 * (d-1):f}) not met")

    # 计算多重 gamma 函数的对数值
    res = (d * (d-1) * 0.25) * np.log(np.pi)
    res += np.sum(loggam([(a - (j - 1.)/2) for j in range(1, d+1)]), axis=0)
    return res
```