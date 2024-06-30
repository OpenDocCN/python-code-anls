# `D:\src\scipysrc\scipy\scipy\stats\_entropy.py`

```
# 导入未来版本兼容模块，使得代码在较旧版本的 Python 中也能正常运行
from __future__ import annotations
# 导入数学库和 numpy 库
import math
import numpy as np
# 导入 scipy 中的特殊函数模块
from scipy import special
# 导入本地模块 _axis_nan_policy 中的工厂函数和 _broadcast_arrays 函数
from ._axis_nan_policy import _axis_nan_policy_factory, _broadcast_arrays
# 导入 scipy._lib._array_api 模块中的 array_namespace 函数
from scipy._lib._array_api import array_namespace

# 模块中公开的函数名列表
__all__ = ['entropy', 'differential_entropy']

# 使用 _axis_nan_policy_factory 装饰器定义 entropy 函数
@_axis_nan_policy_factory(
    # 标识直接返回输入参数 x
    lambda x: x,
    # 如果传入参数中包含 qk 且 qk 不为 None，则返回值为 2，否则返回值为 1
    n_samples=lambda kwgs: (
        2 if ("qk" in kwgs and kwgs["qk"] is not None)
        else 1
    ),
    # 指定返回结果的数量为 1
    n_outputs=1,
    # 将结果转换成元组
    result_to_tuple=lambda x: (x,),
    # 指定 paired=True
    paired=True,
    # 指定 too_small=-1，表示 entropy 函数不接受太小的输入
    too_small=-1
)
# 定义 entropy 函数，计算给定分布的 Shannon 熵或相对熵
def entropy(pk: np.typing.ArrayLike,
            qk: np.typing.ArrayLike | None = None,
            base: float | None = None,
            axis: int = 0
            ) -> np.number | np.ndarray:
    """
    Calculate the Shannon entropy/relative entropy of given distribution(s).

    If only probabilities `pk` are given, the Shannon entropy is calculated as
    ``H = -sum(pk * log(pk))``.

    If `qk` is not None, then compute the relative entropy
    ``D = sum(pk * log(pk / qk))``. This quantity is also known
    as the Kullback-Leibler divergence.

    This routine will normalize `pk` and `qk` if they don't sum to 1.

    Parameters
    ----------
    pk : array_like
        Defines the (discrete) distribution. Along each axis-slice of ``pk``,
        element ``i`` is the  (possibly unnormalized) probability of event
        ``i``.
    qk : array_like, optional
        Sequence against which the relative entropy is computed. Should be in
        the same format as `pk`.
    base : float, optional
        The logarithmic base to use, defaults to ``e`` (natural logarithm).
    axis : int, optional
        The axis along which the entropy is calculated. Default is 0.

    Returns
    -------
    S : {float, array_like}
        The calculated entropy.

    Notes
    -----
    Informally, the Shannon entropy quantifies the expected uncertainty
    inherent in the possible outcomes of a discrete random variable.
    For example,
    if messages consisting of sequences of symbols from a set are to be
    encoded and transmitted over a noiseless channel, then the Shannon entropy
    ``H(pk)`` gives a tight lower bound for the average number of units of
    information needed per symbol if the symbols occur with frequencies
    governed by the discrete distribution `pk` [1]_. The choice of base
    determines the choice of units; e.g., ``e`` for nats, ``2`` for bits, etc.

    The relative entropy, ``D(pk|qk)``, quantifies the increase in the average
    number of units of information needed per symbol if the encoding is
    optimized for the probability distribution `qk` instead of the true
    distribution `pk`. Informally, the relative entropy quantifies the expected
    excess in surprise experienced if one believes the true distribution is
    `qk` when it is actually `pk`.

    A related quantity, the cross entropy ``CE(pk, qk)``, satisfies the
    """
    """
    Calculate the entropy (uncertainty measure) or cross entropy between probability distributions.
    
    Parameters
    ----------
    pk : array_like
        Probability distribution `pk`.
    qk : array_like, optional
        Optional probability distribution `qk` to compute relative entropy (Kullback-Leibler divergence).
    base : float, optional
        The logarithmic base to use for entropy calculation. Default is `e` (natural logarithm).
    
    Returns
    -------
    S : ndarray
        Entropy or cross entropy of the distributions `pk` or `pk` and `qk`, depending on input.
    
    Raises
    ------
    ValueError
        If `base` is non-positive.
    
    Notes
    -----
    The function computes entropy (`H(pk) = -sum(pk * log(pk))`) and cross entropy
    (`CE(pk, qk) = H(pk) + D(pk||qk)`) based on Shannon's information theory.
    
    See [2]_ for more information.
    
    References
    ----------
    .. [1] Shannon, C.E. (1948), A Mathematical Theory of Communication.
           Bell System Technical Journal, 27: 379-423.
           https://doi.org/10.1002/j.1538-7305.1948.tb01338.x
    .. [2] Thomas M. Cover and Joy A. Thomas. 2006. Elements of Information
           Theory (Wiley Series in Telecommunications and Signal Processing).
           Wiley-Interscience, USA.
    
    Examples
    --------
    The outcome of a fair coin is the most uncertain:
    
    >>> import numpy as np
    >>> from scipy.stats import entropy
    >>> base = 2  # work in units of bits
    >>> pk = np.array([1/2, 1/2])  # fair coin
    >>> H = entropy(pk, base=base)
    >>> H
    1.0
    >>> H == -np.sum(pk * np.log(pk)) / np.log(base)
    True
    
    The outcome of a biased coin is less uncertain:
    
    >>> qk = np.array([9/10, 1/10])  # biased coin
    >>> entropy(qk, base=base)
    0.46899559358928117
    
    The relative entropy between the fair coin and biased coin is calculated
    as:
    
    >>> D = entropy(pk, qk, base=base)
    >>> D
    0.7369655941662062
    >>> np.isclose(D, np.sum(pk * np.log(pk/qk)) / np.log(base), rtol=4e-16, atol=0)
    True
    
    The cross entropy can be calculated as the sum of the entropy and
    relative entropy`:
    
    >>> CE = entropy(pk, base=base) + entropy(pk, qk, base=base)
    >>> CE
    1.736965594166206
    >>> CE == -np.sum(pk * np.log(qk)) / np.log(base)
    True
    
    """
    if base is not None and base <= 0:
        raise ValueError("`base` must be a positive number or `None`.")
    
    xp = array_namespace(pk) if qk is None else array_namespace(pk, qk)
    
    pk = xp.asarray(pk)
    with np.errstate(invalid='ignore'):
        pk = 1.0*pk / xp.sum(pk, axis=axis, keepdims=True)  # type: ignore[operator]
    
    if qk is None:
        vec = special.entr(pk)
    else:
        qk = xp.asarray(qk)
        pk, qk = _broadcast_arrays((pk, qk), axis=None, xp=xp)  # don't ignore any axes
        sum_kwargs = dict(axis=axis, keepdims=True)
        qk = 1.0*qk / xp.sum(qk, **sum_kwargs)  # type: ignore[operator, call-overload]
        vec = special.rel_entr(pk, qk)
    
    S = xp.sum(vec, axis=axis)
    if base is not None:
        S /= math.log(base)
    return S
# 检查差分熵是否过小的私有函数
def _differential_entropy_is_too_small(samples, kwargs, axis=-1):
    # 从样本中取出第一个值作为参考
    values = samples[0]
    # 获取沿指定轴的样本长度
    n = values.shape[axis]
    # 获取窗口长度，如果未指定则使用默认的值
    window_length = kwargs.get("window_length",
                               math.floor(math.sqrt(n) + 0.5))
    # 如果窗口长度不符合预期范围则返回True，表示差分熵过小
    if not 2 <= 2 * window_length < n:
        return True
    # 否则返回False，表示差分熵不过小
    return False

# 差分熵计算函数，使用装饰器设置轴向NaN处理策略和窗口长度判断函数
@_axis_nan_policy_factory(
    lambda x: x,  # 将输入直接作为输出
    n_outputs=1,  # 只产生一个输出
    result_to_tuple=lambda x: (x,),  # 将结果包装成元组形式
    too_small=_differential_entropy_is_too_small  # 使用_differential_entropy_is_too_small函数判断是否差分熵过小
)
def differential_entropy(
    values: np.typing.ArrayLike,
    *,
    window_length: int | None = None,
    base: float | None = None,
    axis: int = 0,
    method: str = "auto",
) -> np.number | np.ndarray:
    r"""给定一个分布的样本，估算其差分熵。

    通过`method`参数提供多种估算方法。默认情况下，会根据样本的大小选择一个方法。

    Parameters
    ----------
    values : sequence
        连续分布的样本。
    window_length : int, optional
        用于计算Vasicek估计的窗口长度。必须是1到样本大小的一半之间的整数。
        如果为``None``（默认值），则使用以下启发式值

        .. math::
            \left \lfloor \sqrt{n} + 0.5 \right \rfloor

        其中 :math:`n` 是样本大小。此启发式方法最初在文献 [2]_ 中提出，并已广泛使用。
    base : float, optional
        用于对数的基数，默认为``e``（自然对数）。
    axis : int, optional
        计算差分熵的轴。
        默认为0。
    method : {'vasicek', 'van es', 'ebrahimi', 'correa', 'auto'}, optional
        用于从样本估算差分熵的方法。
        默认为``'auto'``。有关更多信息，请参见Notes。

    Returns
    -------
    entropy : float
        计算得到的差分熵值。

    Notes
    -----
    在下列极限条件下，此函数会收敛到真实的差分熵值

    .. math::
        n \to \infty, \quad m \to \infty, \quad \frac{m}{n} \to 0

    对于给定样本大小，最优的`window_length`取值取决于（未知的）分布。
    通常情况下，分布的密度越平滑，`window_length`的最优值越大 [1]_。

    `method`参数支持以下选项。

    * ``'vasicek'`` 使用文献 [1]_ 中提出的估算器。这是最早和最有影响力的差分熵估算器之一。
    * ``'van es'`` 使用文献 [3]_ 中的修正偏差估算器，它不仅是一致的，而且在某些条件下是渐近正态的。
    * ``'ebrahimi'`` 使用文献 [4]_ 中的估算器，在模拟中显示出比Vasicek估算器更小的偏差和均方误差。
    * ``'correa'`` uses the estimator presented in [5]_ based on local linear
      regression. In a simulation study, it had consistently smaller mean
      square error than the Vasiceck estimator, but it is more expensive to
      compute.
    * ``'auto'`` selects the method automatically (default). Currently,
      this selects ``'van es'`` for very small samples (<10), ``'ebrahimi'``
      for moderate sample sizes (11-1000), and ``'vasicek'`` for larger
      samples, but this behavior is subject to change in future versions.

# 'correa' 方法使用基于局部线性回归的估计器 [5]_。在模拟研究中，其均方误差始终小于 Vasiceck 估计器，但计算成本更高。
# 'auto' 方法自动选择估计方法（默认）。当前情况下，对于非常小的样本（<10），选择 'van es'；对于中等样本大小（11-1000），选择 'ebrahimi'；对于较大样本，则选择 'vasicek'。但这种行为可能在未来版本中更改。

    All estimators are implemented as described in [6]_.

# 所有的估计方法都按照 [6]_ 中描述的实现。

    References
    ----------
    .. [1] Vasicek, O. (1976). A test for normality based on sample entropy.
           Journal of the Royal Statistical Society:
           Series B (Methodological), 38(1), 54-59.
    .. [2] Crzcgorzewski, P., & Wirczorkowski, R. (1999). Entropy-based
           goodness-of-fit test for exponentiality. Communications in
           Statistics-Theory and Methods, 28(5), 1183-1202.
    .. [3] Van Es, B. (1992). Estimating functionals related to a density by a
           class of statistics based on spacings. Scandinavian Journal of
           Statistics, 61-72.
    .. [4] Ebrahimi, N., Pflughoeft, K., & Soofi, E. S. (1994). Two measures
           of sample entropy. Statistics & Probability Letters, 20(3), 225-234.
    .. [5] Correa, J. C. (1995). A new estimator of entropy. Communications
           in Statistics-Theory and Methods, 24(10), 2439-2449.
    .. [6] Noughabi, H. A. (2015). Entropy Estimation Using Numerical Methods.
           Annals of Data Science, 2(2), 231-241.
           https://link.springer.com/article/10.1007/s40745-015-0045-9

# 参考文献：
# .. [1] Vasicek, O. (1976). 基于样本熵的正态性检验。Journal of the Royal Statistical Society: Series B (Methodological), 38(1), 54-59.
# .. [2] Crzcgorzewski, P., & Wirczorkowski, R. (1999). 基于熵的指数分布拟合检验。Communications in Statistics-Theory and Methods, 28(5), 1183-1202.
# .. [3] Van Es, B. (1992). 通过间距统计的统计类估计密度相关的函数。Scandinavian Journal of Statistics, 61-72.
# .. [4] Ebrahimi, N., Pflughoeft, K., & Soofi, E. S. (1994). 两种样本熵度量。Statistics & Probability Letters, 20(3), 225-234.
# .. [5] Correa, J. C. (1995). 熵的新估计方法。Communications in Statistics-Theory and Methods, 24(10), 2439-2449.
# .. [6] Noughabi, H. A. (2015). 数值方法估计熵。Annals of Data Science, 2(2), 231-241. https://link.springer.com/article/10.1007/s40745-015-0045-9

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import differential_entropy, norm

    Entropy of a standard normal distribution:

    >>> rng = np.random.default_rng()
    >>> values = rng.standard_normal(100)
    >>> differential_entropy(values)
    1.3407817436640392

    Compare with the true entropy:

    >>> float(norm.entropy())
    1.4189385332046727

    For several sample sizes between 5 and 1000, compare the accuracy of
    the ``'vasicek'``, ``'van es'``, and ``'ebrahimi'`` methods. Specifically,
    compare the root mean squared error (over 1000 trials) between the estimate
    and the true differential entropy of the distribution.

    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt
    >>>
    >>>
    >>> def rmse(res, expected):
    ...     '''Root mean squared error'''
    ...     return np.sqrt(np.mean((res - expected)**2))
    >>>
    >>>
    >>> a, b = np.log10(5), np.log10(1000)
    >>> ns = np.round(np.logspace(a, b, 10)).astype(int)
    >>> reps = 1000  # number of repetitions for each sample size
    >>> expected = stats.expon.entropy()
    >>>
    >>> method_errors = {'vasicek': [], 'van es': [], 'ebrahimi': []}
    >>> for method in method_errors:
    ...     for n in ns:

# 示例：
# 标准正态分布的熵计算：
# >>> rng = np.random.default_rng()
# >>> values = rng.standard_normal(100)
# >>> differential_entropy(values)
# 1.3407817436640392
# 与真实熵值比较：
# >>> float(norm.entropy())
# 1.4189385332046727
# 对于样本大小在5到1000之间的几种情况，比较 'vasicek'、'van es' 和 'ebrahimi' 方法的准确性。具体来说，比较估计值与分布真实差分熵之间的均方根误差（1000次试验的平均）。
    # 将输入的 `values` 转换为 NumPy 数组
    values = np.asarray(values)
    # 将数据的轴从 `axis` 移动到数组的最后一个位置
    values = np.moveaxis(values, axis, -1)
    # 获取数据中最后一个轴的长度，即观测值的数量
    n = values.shape[-1]  # number of observations
    
    # 如果未指定窗口长度，则设置为 sqrt(n) 的整数部分加 0.5
    if window_length is None:
        window_length = math.floor(math.sqrt(n) + 0.5)
    
    # 检查窗口长度是否在有效范围内
    if not 2 <= 2 * window_length < n:
        raise ValueError(
            f"Window length ({window_length}) must be positive and less "
            f"than half the sample size ({n}).",
        )
    
    # 如果指定了 `base` 且 `base` 小于等于 0，则引发错误
    if base is not None and base <= 0:
        raise ValueError("`base` must be a positive number or `None`.")
    
    # 对数据进行排序
    sorted_data = np.sort(values, axis=-1)
    
    # 定义不同熵估计方法的函数映射
    methods = {"vasicek": _vasicek_entropy,
               "van es": _van_es_entropy,
               "correa": _correa_entropy,
               "ebrahimi": _ebrahimi_entropy,
               "auto": _vasicek_entropy}
    
    # 将 `method` 转换为小写
    method = method.lower()
    # 检查 `method` 是否在支持的方法列表中
    if method not in methods:
        message = f"`method` must be one of {set(methods)}"
        raise ValueError(message)
    
    # 如果 `method` 是 "auto"，根据数据长度选择合适的方法
    if method == "auto":
        if n <= 10:
            method = 'van es'
        elif n <= 1000:
            method = 'ebrahimi'
        else:
            method = 'vasicek'
    
    # 使用选定的方法计算数据的熵估计值
    res = methods[method](sorted_data, window_length)
    
    # 如果指定了 `base`，则将结果除以 log(base)
    if base is not None:
        res /= np.log(base)
    
    # 返回计算得到的熵估计值
    return res
# 在最后一个轴上为计算滚动窗口差异而填充数据
def _pad_along_last_axis(X, m):
    # 获取数据 X 的形状
    shape = np.array(X.shape)
    # 将最后一个轴的长度设置为 m
    shape[-1] = m
    # 将 X 的第一个和最后一个元素广播到新的形状中，以保持原始数据的形状
    Xl = np.broadcast_to(X[..., [0]], shape)  # [0] vs 0 to maintain shape
    Xr = np.broadcast_to(X[..., [-1]], shape)
    # 返回填充后的数据，通过在最后一个轴上连接 Xl、X 和 Xr
    return np.concatenate((Xl, X, Xr), axis=-1)


# 计算 Vasicek 估计量，如文献 [6] 中的 Eq. 1.3 所述
def _vasicek_entropy(X, m):
    # 获取最后一个轴的长度
    n = X.shape[-1]
    # 使用 _pad_along_last_axis 函数填充 X，以准备进行计算
    X = _pad_along_last_axis(X, m)
    # 计算差异
    differences = X[..., 2 * m:] - X[..., : -2 * m:]
    # 计算对数
    logs = np.log(n/(2*m) * differences)
    # 返回对数的均值，沿着最后一个轴求平均
    return np.mean(logs, axis=-1)


# 计算 van Es 估计量，如文献 [6] 中描述的方法
def _van_es_entropy(X, m):
    # 获取最后一个轴的长度
    n = X.shape[-1]
    # 计算差异
    difference = X[..., m:] - X[..., :-m]
    # 计算第一个术语
    term1 = 1/(n-m) * np.sum(np.log((n+1)/m * difference), axis=-1)
    # 计算 k 的和
    k = np.arange(m, n+1)
    # 返回最终估计量，包括求和和对数运算
    return term1 + np.sum(1/k) + np.log(m) - np.log(n+1)


# 计算 Ebrahimi 估计量，如文献 [6] 中描述的方法
def _ebrahimi_entropy(X, m):
    # 获取最后一个轴的长度
    n = X.shape[-1]
    # 使用 _pad_along_last_axis 函数填充 X，以准备进行计算
    X = _pad_along_last_axis(X, m)

    # 计算差异
    differences = X[..., 2 * m:] - X[..., : -2 * m:]

    # 计算 ci
    i = np.arange(1, n+1).astype(float)
    ci = np.ones_like(i)*2
    ci[i <= m] = 1 + (i[i <= m] - 1)/m
    ci[i >= n - m + 1] = 1 + (n - i[i >= n-m+1])/m

    # 计算对数
    logs = np.log(n * differences / (ci * m))
    # 返回对数的均值，沿着最后一个轴求平均
    return np.mean(logs, axis=-1)


# 计算 Correa 估计量，如文献 [6] 中描述的方法
def _correa_entropy(X, m):
    # 获取最后一个轴的长度
    n = X.shape[-1]
    # 使用 _pad_along_last_axis 函数填充 X，以准备进行计算
    X = _pad_along_last_axis(X, m)

    # 计算索引
    i = np.arange(1, n+1)
    dj = np.arange(-m, m+1)[:, None]
    j = i + dj
    j0 = j + m - 1  # 0-indexed version of j

    # 计算 Xibar
    Xibar = np.mean(X[..., j0], axis=-2, keepdims=True)
    # 计算差异
    difference = X[..., j0] - Xibar
    # 计算 num 和 den
    num = np.sum(difference*dj, axis=-2)  # dj is d-i
    den = n*np.sum(difference**2, axis=-2)
    # 返回 Correa 估计量，包括对数运算
    return -np.mean(np.log(num/den), axis=-1)
```