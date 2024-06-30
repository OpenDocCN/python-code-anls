# `D:\src\scipysrc\scipy\scipy\special\_add_newdocs.py`

```
# 生成的ufunc的文档字符串
#
# 这些语句用于生成看起来像从numpy.lib调用add_newdoc的函数文档字符串，
# 但实际上在这个文件中，add_newdoc将文档字符串放入一个字典中。这个字典在
# _generate_pyx.py中用于在编译时生成scipy.special中ufunc的C级别文档字符串。

# 创建一个空的字典docdict，用于存储ufunc的文档字符串
docdict: dict[str, str] = {}

# 根据名称从docdict中获取文档字符串
def get(name):
    return docdict.get(name)

# 向docdict中添加新的文档字符串
def add_newdoc(name, doc):
    docdict[name] = doc

# 添加名为"_sf_error_test_function"的新文档字符串
add_newdoc("_sf_error_test_function",
    """
    Private function; do not use.
    """)

# 添加名为"_cosine_cdf"的新文档字符串
add_newdoc("_cosine_cdf",
    """
    _cosine_cdf(x)

    Cosine分布的累积分布函数(CDF)：

                 {             0,              x < -pi
        cdf(x) = { (pi + x + sin(x))/(2*pi),   -pi <= x <= pi
                 {             1,              x > pi

    参数
    ----------
    x : array_like
        `x` 必须包含实数。

    返回
    -------
    scalar 或 ndarray
        在`x`处评估的Cosine分布的CDF。

    """)

# 添加名为"_cosine_invcdf"的新文档字符串
add_newdoc("_cosine_invcdf",
    """
    _cosine_invcdf(p)

    Cosine分布的累积分布函数(CDF)的反函数。

    Cosine分布的CDF为::

        cdf(x) = (pi + x + sin(x))/(2*pi)

    此函数计算cdf(x)的反函数。

    参数
    ----------
    p : array_like
        `p` 必须包含介于 ``0 <= p <= 1`` 的实数。
        对于超出区间[0, 1]的`p`值，返回`nan`。

    返回
    -------
    scalar 或 ndarray
        在`p`处评估的Cosine分布的CDF的反函数。

    """)

# 添加名为"_ellip_harm"的新文档字符串
add_newdoc("_ellip_harm",
    """
    Internal function, use `ellip_harm` instead.
    """)

# 添加名为"_ellip_norm"的新文档字符串
add_newdoc("_ellip_norm",
    """
    Internal function, use `ellip_norm` instead.
    """)

# 添加名为"voigt_profile"的新文档字符串
add_newdoc("voigt_profile",
    r"""
    voigt_profile(x, sigma, gamma, out=None)

    Voigt profile.

    Voigt profile是一维正态分布（标准差为`sigma`）与一维柯西分布（半最大宽度为`gamma`）的卷积。

    如果 `sigma = 0`，返回柯西分布的概率密度函数（PDF）。
    相反，如果 `gamma = 0`，返回正态分布的PDF。
    如果 `sigma = gamma = 0`，对于 `x = 0` 返回 `Inf`，对于所有其他 `x` 返回 `0`。

    参数
    ----------
    x : array_like
        实数参数
    sigma : array_like
        正态分布部分的标准差
    gamma : array_like
        柯西分布部分的半最大宽度
    out : ndarray, optional
        可选的输出数组用于存储函数值

    返回
    -------
    scalar 或 ndarray
        给定参数下的Voigt profile

    另请参阅
    --------
    wofz : Faddeeva函数

    注意
    -----
    可以用Faddeeva函数来表示

    """)
    .. math:: V(x; \sigma, \gamma) = \frac{Re[w(z)]}{\sigma\sqrt{2\pi}},
    定义 Voigt 分布函数 V(x; \sigma, \gamma)，表示为 Faddeeva 函数的实部除以标准差乘以根号下的常数。

    .. math:: z = \frac{x + i\gamma}{\sqrt{2}\sigma}
    定义变量 z，用于计算 Voigt 分布中的参数，结合输入 x、gamma 和 sigma。

    where :math:`w(z)` is the Faddeeva function.
    提示 w(z) 是 Faddeeva 函数，用于 Voigt 分布的计算。

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Voigt_profile
    提供了 Voigt 分布的参考链接，详细解释了 Voigt 分布的概念和应用。

    Examples
    --------
    Calculate the function at point 2 for ``sigma=1`` and ``gamma=1``.
    示例：计算在 sigma=1 和 gamma=1 时，x=2 处的 Voigt 分布函数值。

    >>> from scipy.special import voigt_profile
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> voigt_profile(2, 1., 1.)
    0.09071519942627544

    Calculate the function at several points by providing a NumPy array
    for `x`.
    示例：通过提供 NumPy 数组 x 来计算多个点的 Voigt 分布函数值。

    >>> values = np.array([-2., 0., 5])
    >>> voigt_profile(values, 1., 1.)
    array([0.0907152 , 0.20870928, 0.01388492])

    Plot the function for different parameter sets.
    示例：绘制不同参数集合下的 Voigt 分布函数图像。

    >>> fig, ax = plt.subplots(figsize=(8, 8))
    >>> x = np.linspace(-10, 10, 500)
    >>> parameters_list = [(1.5, 0., "solid"), (1.3, 0.5, "dashed"),
    ...                    (0., 1.8, "dotted"), (1., 1., "dashdot")]
    >>> for params in parameters_list:
    ...     sigma, gamma, linestyle = params
    ...     voigt = voigt_profile(x, sigma, gamma)
    ...     ax.plot(x, voigt, label=rf"$\sigma={sigma},\, \gamma={gamma}$",
    ...             ls=linestyle)
    >>> ax.legend()
    >>> plt.show()

    Verify visually that the Voigt profile indeed arises as the convolution
    of a normal and a Cauchy distribution.
    示例：通过图形验证 Voigt 分布确实是正态分布和柯西分布的卷积结果。

    >>> from scipy.signal import convolve
    >>> x, dx = np.linspace(-10, 10, 500, retstep=True)
    >>> def gaussian(x, sigma):
    ...     return np.exp(-0.5 * x**2/sigma**2)/(sigma * np.sqrt(2*np.pi))
    >>> def cauchy(x, gamma):
    ...     return gamma/(np.pi * (np.square(x)+gamma**2))
    >>> sigma = 2
    >>> gamma = 1
    >>> gauss_profile = gaussian(x, sigma)
    >>> cauchy_profile = cauchy(x, gamma)
    >>> convolved = dx * convolve(cauchy_profile, gauss_profile, mode="same")
    >>> voigt = voigt_profile(x, sigma, gamma)
    >>> fig, ax = plt.subplots(figsize=(8, 8))
    >>> ax.plot(x, gauss_profile, label="Gauss: $G$", c='b')
    >>> ax.plot(x, cauchy_profile, label="Cauchy: $C$", c='y', ls="dashed")
    >>> xx = 0.5*(x[1:] + x[:-1])  # midpoints
    >>> ax.plot(xx, convolved[1:], label="Convolution: $G * C$", ls='dashdot',
    ...         c='k')
    >>> ax.plot(x, voigt, label="Voigt", ls='dotted', c='r')
    >>> ax.legend()
    >>> plt.show()
# 添加新的文档字符串到指定函数 "wrightomega" 中
add_newdoc("wrightomega",
    r"""
    wrightomega(z, out=None)

    Wright Omega function.

    Defined as the solution to

    .. math::

        \omega + \log(\omega) = z

    where :math:`\log` is the principal branch of the complex logarithm.

    Parameters
    ----------
    z : array_like
        Points at which to evaluate the Wright Omega function
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    omega : scalar or ndarray
        Values of the Wright Omega function

    See Also
    --------
    lambertw : The Lambert W function

    Notes
    -----
    .. versionadded:: 0.19.0

    The function can also be defined as

    .. math::

        \omega(z) = W_{K(z)}(e^z)

    where :math:`K(z) = \lceil (\Im(z) - \pi)/(2\pi) \rceil` is the
    unwinding number and :math:`W` is the Lambert W function.

    The implementation here is taken from [1]_.

    References
    ----------
    .. [1] Lawrence, Corless, and Jeffrey, "Algorithm 917: Complex
           Double-Precision Evaluation of the Wright :math:`\omega`
           Function." ACM Transactions on Mathematical Software,
           2012. :doi:`10.1145/2168773.2168779`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import wrightomega, lambertw

    >>> wrightomega([-2, -1, 0, 1, 2])
    array([0.12002824, 0.27846454, 0.56714329, 1.        , 1.5571456 ])

    Complex input:

    >>> wrightomega(3 + 5j)
    (1.5804428632097158+3.8213626783287937j)

    Verify that ``wrightomega(z)`` satisfies ``w + log(w) = z``:

    >>> w = -5 + 4j
    >>> wrightomega(w + np.log(w))
    (-5+4j)

    Verify the connection to ``lambertw``:

    >>> z = 0.5 + 3j
    >>> wrightomega(z)
    (0.0966015889280649+1.4937828458191993j)
    >>> lambertw(np.exp(z))
    (0.09660158892806493+1.4937828458191993j)

    >>> z = 0.5 + 4j
    >>> wrightomega(z)
    (-0.3362123489037213+2.282986001579032j)
    >>> lambertw(np.exp(z), k=1)
    (-0.33621234890372115+2.282986001579032j)
    """)


# 添加新的文档字符串到指定函数 "agm" 中
add_newdoc("agm",
    """
    agm(a, b, out=None)

    Compute the arithmetic-geometric mean of `a` and `b`.

    Start with a_0 = a and b_0 = b and iteratively compute::

        a_{n+1} = (a_n + b_n)/2
        b_{n+1} = sqrt(a_n*b_n)

    a_n and b_n converge to the same limit as n increases; their common
    limit is agm(a, b).

    Parameters
    ----------
    a, b : array_like
        Real values only. If the values are both negative, the result
        is negative. If one value is negative and the other is positive,
        `nan` is returned.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        The arithmetic-geometric mean of `a` and `b`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import agm
    >>> a, b = 24.0, 6.0
    >>> agm(a, b)
    13.458171481725614

    Compare that result to the iteration:
    """
    # 当 a 不等于 b 时执行循环
    while a != b:
        # 更新 a 和 b 的值为它们的算术几何平均数
        a, b = (a + b)/2, np.sqrt(a*b)
        # 打印当前的 a 和 b 值，使用指定格式打印
        print("a = %19.16f  b=%19.16f" % (a, b))
    
    # 循环结束后输出的几组 a 和 b 的值
    a = 15.0000000000000000  b=12.0000000000000000
    a = 13.5000000000000000  b=13.4164078649987388
    a = 13.4582039324993694  b=13.4581390309909850
    a = 13.4581714817451772  b=13.4581714817060547
    a = 13.4581714817256159  b=13.4581714817256159
    
    # 当传入类数组参数时，会进行广播操作：
    # 创建一个形状为 (3, 1) 的数组 a
    a = np.array([[1.5], [3], [6]])
    # 创建一个形状为 (4,) 的数组 b
    b = np.array([6, 12, 24, 48])
    # 调用 agm 函数对 a 和 b 进行计算，返回一个新的数组
    agm(a, b)
    # 返回的数组结果如下：
    # array([[  3.36454287,   5.42363427,   9.05798751,  15.53650756],
    #        [  4.37037309,   6.72908574,  10.84726853,  18.11597502],
    #        [  6.        ,   8.74074619,  13.45817148,  21.69453707]])
# 添加新的文档字符串给名为"airy"的函数
add_newdoc("airy",
    r"""
    airy(z, out=None)

    Airy functions and their derivatives.

    Parameters
    ----------
    z : array_like
        Real or complex argument.
    out : tuple of ndarray, optional
        Optional output arrays for the function values

    Returns
    -------
    Ai, Aip, Bi, Bip : 4-tuple of scalar or ndarray
        Airy functions Ai and Bi, and their derivatives Aip and Bip.

    See Also
    --------
    airye : exponentially scaled Airy functions.

    Notes
    -----
    The Airy functions Ai and Bi are two independent solutions of

    .. math:: y''(x) = x y(x).

    For real `z` in [-10, 10], the computation is carried out by calling
    the Cephes [1]_ `airy` routine, which uses power series summation
    for small `z` and rational minimax approximations for large `z`.

    Outside this range, the AMOS [2]_ `zairy` and `zbiry` routines are
    employed.  They are computed using power series for :math:`|z| < 1` and
    the following relations to modified Bessel functions for larger `z`
    (where :math:`t \equiv 2 z^{3/2}/3`):

    .. math::

        Ai(z) = \frac{1}{\pi \sqrt{3}} K_{1/3}(t)

        Ai'(z) = -\frac{z}{\pi \sqrt{3}} K_{2/3}(t)

        Bi(z) = \sqrt{\frac{z}{3}} \left(I_{-1/3}(t) + I_{1/3}(t) \right)

        Bi'(z) = \frac{z}{\sqrt{3}} \left(I_{-2/3}(t) + I_{2/3}(t)\right)

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    .. [2] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/

    Examples
    --------
    Compute the Airy functions on the interval [-15, 5].

    >>> import numpy as np
    >>> from scipy import special
    >>> x = np.linspace(-15, 5, 201)
    >>> ai, aip, bi, bip = special.airy(x)

    Plot Ai(x) and Bi(x).

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, ai, 'r', label='Ai(x)')
    >>> plt.plot(x, bi, 'b--', label='Bi(x)')
    >>> plt.ylim(-0.5, 1.0)
    >>> plt.grid()
    >>> plt.legend(loc='upper left')
    >>> plt.show()

    """)

# 添加新的文档字符串给名为"airye"的函数
add_newdoc("airye",
    """
    airye(z, out=None)

    Exponentially scaled Airy functions and their derivatives.

    Scaling::

        eAi  = Ai  * exp(2.0/3.0*z*sqrt(z))
        eAip = Aip * exp(2.0/3.0*z*sqrt(z))
        eBi  = Bi  * exp(-abs(2.0/3.0*(z*sqrt(z)).real))
        eBip = Bip * exp(-abs(2.0/3.0*(z*sqrt(z)).real))

    Parameters
    ----------
    z : array_like
        Real or complex argument.
    out : tuple of ndarray, optional
        Optional output arrays for the function values

    Returns
    -------
    eAi, eAip, eBi, eBip : 4-tuple of scalar or ndarray
        Exponentially scaled Airy functions eAi and eBi, and their derivatives
        eAip and eBip

    See Also
    --------
    airy

    Notes
    -----
    Wrapper for the AMOS [1]_ routines `zairy` and `zbiry`.

    References
    ----------
    [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
        of a Complex Argument and Nonnegative Order",
        http://netlib.org/amos/
    """
)
    Examples
    --------
    We can compute exponentially scaled Airy functions and their derivatives:

    >>> import numpy as np
    >>> from scipy.special import airye
    >>> import matplotlib.pyplot as plt
    >>> z = np.linspace(0, 50, 500)
    >>> eAi, eAip, eBi, eBip = airye(z)
    >>> f, ax = plt.subplots(2, 1, sharex=True)
    >>> for ind, data in enumerate([[eAi, eAip, ["eAi", "eAip"]],
    ...                             [eBi, eBip, ["eBi", "eBip"]]]):
    ...     # Plot the exponentially scaled Airy functions and their derivatives
    ...     ax[ind].plot(z, data[0], "-r", z, data[1], "-b")
    ...     # Add legends for each function and derivative
    ...     ax[ind].legend(data[2])
    ...     # Enable grid for the subplot
    ...     ax[ind].grid(True)
    >>> # Display the plot
    >>> plt.show()

    We can compute these using usual non-scaled Airy functions by:

    >>> from scipy.special import airy
    >>> Ai, Aip, Bi, Bip = airy(z)
    >>> # Verify the relationship between scaled and non-scaled Airy functions
    >>> np.allclose(eAi, Ai * np.exp(2.0 / 3.0 * z * np.sqrt(z)))
    True
    >>> np.allclose(eAip, Aip * np.exp(2.0 / 3.0 * z * np.sqrt(z)))
    True
    >>> np.allclose(eBi, Bi * np.exp(-abs(np.real(2.0 / 3.0 * z * np.sqrt(z)))))
    True
    >>> np.allclose(eBip, Bip * np.exp(-abs(np.real(2.0 / 3.0 * z * np.sqrt(z)))))
    True

    Comparing non-scaled and exponentially scaled ones, the usual non-scaled
    function quickly underflows for large values, whereas the exponentially
    scaled function does not.

    >>> airy(200)
    (0.0, 0.0, nan, nan)
    >>> airye(200)
    (0.07501041684381093, -1.0609012305109042, 0.15003188417418148, 2.1215836725571093)
# 添加新的文档字符串给函数 "bdtr"
add_newdoc("bdtr",
    r"""
    bdtr(k, n, p, out=None)

    二项分布累积分布函数。

    计算二项概率密度从 0 到 `floor(k)` 的和。

    .. math::
        \mathrm{bdtr}(k, n, p) =
        \sum_{j=0}^{\lfloor k \rfloor} {{n}\choose{j}} p^j (1-p)^{n-j}

    Parameters
    ----------
    k : array_like
        成功次数（浮点数），向下舍入到最接近的整数。
    n : array_like
        事件次数（整数）。
    p : array_like
        单次事件成功的概率（浮点数）。
    out : ndarray, optional
        可选的输出数组用于函数值。

    Returns
    -------
    y : 标量或者 ndarray
        在 `n` 次独立事件中，成功次数不超过 `floor(k)` 的概率，成功概率为 `p`。

    Notes
    -----
    不直接求和项；而是使用正则化的不完全贝塔函数，根据以下公式计算：

    .. math::
        \mathrm{bdtr}(k, n, p) =
        I_{1 - p}(n - \lfloor k \rfloor, \lfloor k \rfloor + 1).

    Cephes [1]_ 库中 `bdtr` 的包装器。

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    """)

# 添加新的文档字符串给函数 "bdtrc"
add_newdoc("bdtrc",
    r"""
    bdtrc(k, n, p, out=None)

    二项分布生存函数。

    计算二项概率密度从 `floor(k) + 1` 到 `n` 的和。

    .. math::
        \mathrm{bdtrc}(k, n, p) =
        \sum_{j=\lfloor k \rfloor +1}^n {{n}\choose{j}} p^j (1-p)^{n-j}

    Parameters
    ----------
    k : array_like
        成功次数（浮点数），向下舍入到最接近的整数。
    n : array_like
        事件次数（整数）。
    p : array_like
        单次事件成功的概率（浮点数）。
    out : ndarray, optional
        可选的输出数组用于函数值。

    Returns
    -------
    y : 标量或者 ndarray
        在 `n` 次独立事件中，成功次数不少于 `floor(k) + 1` 的概率，成功概率为 `p`。

    See Also
    --------
    bdtr
    betainc

    Notes
    -----
    不直接求和项；而是使用正则化的不完全贝塔函数，根据以下公式计算：

    .. math::
        \mathrm{bdtrc}(k, n, p) = I_{p}(\lfloor k \rfloor + 1, n - \lfloor k \rfloor).

    Cephes [1]_ 库中 `bdtrc` 的包装器。

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    """)

# 添加新的文档字符串给函数 "bdtri"
add_newdoc("bdtri",
    r"""
    bdtri(k, n, y, out=None)

    对于 `p` 的 `bdtr` 的逆函数。

    找到事件概率 `p`，使得二项概率密度从 0 到 `k` 的和等于给定的累积概率 `y`。

    Parameters
    ----------
    k : array_like
        成功次数（浮点数），向下舍入到最接近的整数。
    n : array_like
        # 参数 n 是一个类数组对象，表示事件的数量（浮点数）

    y : array_like
        # 参数 y 是一个类数组对象，表示累积概率（即在 n 次事件中出现 k 次或更少成功的概率）。

    out : ndarray, optional
        # 可选参数，用于存储函数值的输出数组

    Returns
    -------
    p : scalar or ndarray
        # 返回事件概率 p，满足 bdtr(floor(k), n, p) = y。

    See Also
    --------
    bdtr
    betaincinv

    Notes
    -----
    # 计算使用反向贝塔积分函数和以下关系进行：
    # 1 - p = betaincinv(n - k, k + 1, y)。

    # 这是 Cephes [1]_ 库中 bdtri 函数的包装。

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
# 将新的文档添加到 `bdtrik` 模块中，提供了 `bdtrik` 函数的详细说明
add_newdoc("bdtrik",
    """
    bdtrik(y, n, p, out=None)

    Inverse function to `bdtr` with respect to `k`.

    Finds the number of successes `k` such that the sum of the terms 0 through
    `k` of the Binomial probability density for `n` events with probability
    `p` is equal to the given cumulative probability `y`.

    Parameters
    ----------
    y : array_like
        Cumulative probability (probability of `k` or fewer successes in `n`
        events).
    n : array_like
        Number of events (float).
    p : array_like
        Success probability (float).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    k : scalar or ndarray
        The number of successes `k` such that `bdtr(k, n, p) = y`.

    See Also
    --------
    bdtr

    Notes
    -----
    Formula 26.5.24 of [1]_ is used to reduce the binomial distribution to the
    cumulative incomplete beta distribution.

    Computation of `k` involves a search for a value that produces the desired
    value of `y`. The search relies on the monotonicity of `y` with `k`.

    Wrapper for the CDFLIB [2]_ Fortran routine `cdfbin`.

    References
    ----------
    .. [1] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.
    .. [2] Barry Brown, James Lovato, and Kathy Russell,
           CDFLIB: Library of Fortran Routines for Cumulative Distribution
           Functions, Inverses, and Other Parameters.
    """)

# 将新的文档添加到 `bdtrin` 模块中，提供了 `bdtrin` 函数的详细说明
add_newdoc("bdtrin",
    """
    bdtrin(k, y, p, out=None)

    Inverse function to `bdtr` with respect to `n`.

    Finds the number of events `n` such that the sum of the terms 0 through
    `k` of the Binomial probability density for events with probability `p` is
    equal to the given cumulative probability `y`.

    Parameters
    ----------
    k : array_like
        Number of successes (float).
    y : array_like
        Cumulative probability (probability of `k` or fewer successes in `n`
        events).
    p : array_like
        Success probability (float).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    n : scalar or ndarray
        The number of events `n` such that `bdtr(k, n, p) = y`.

    See Also
    --------
    bdtr

    Notes
    -----
    Formula 26.5.24 of [1]_ is used to reduce the binomial distribution to the
    cumulative incomplete beta distribution.

    Computation of `n` involves a search for a value that produces the desired
    value of `y`. The search relies on the monotonicity of `y` with `n`.

    Wrapper for the CDFLIB [2]_ Fortran routine `cdfbin`.

    References
    ----------
    .. [1] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.
    """)
    .. [2] Barry Brown, James Lovato, and Kathy Russell,
           CDFLIB: Library of Fortran Routines for Cumulative Distribution
           Functions, Inverses, and Other Parameters.
# 向指定模块添加新的文档字符串，描述函数 `btdtria` 的逆函数。
add_newdoc("btdtria",
    r"""
    btdtria(p, b, x, out=None)

    Inverse of `btdtr` with respect to `a`.

    This is the inverse of the beta cumulative distribution function, `btdtr`,
    considered as a function of `a`, returning the value of `a` for which
    `btdtr(a, b, x) = p`, or

    .. math::
        p = \int_0^x \frac{\Gamma(a + b)}{\Gamma(a)\Gamma(b)} t^{a-1} (1-t)^{b-1}\,dt

    Parameters
    ----------
    p : array_like
        Cumulative probability, in [0, 1].
    b : array_like
        Shape parameter (`b` > 0).
    x : array_like
        The quantile, in [0, 1].
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    a : scalar or ndarray
        The value of the shape parameter `a` such that `btdtr(a, b, x) = p`.

    See Also
    --------
    btdtr : Cumulative distribution function of the beta distribution.
    btdtri : Inverse with respect to `x`.
    btdtrib : Inverse with respect to `b`.

    Notes
    -----
    Wrapper for the CDFLIB [1]_ Fortran routine `cdfbet`.

    The cumulative distribution function `p` is computed using a routine by
    DiDinato and Morris [2]_. Computation of `a` involves a search for a value
    that produces the desired value of `p`. The search relies on the
    monotonicity of `p` with `a`.

    References
    ----------
    .. [1] Barry Brown, James Lovato, and Kathy Russell,
           CDFLIB: Library of Fortran Routines for Cumulative Distribution
           Functions, Inverses, and Other Parameters.
    .. [2] DiDinato, A. R. and Morris, A. H.,
           Algorithm 708: Significant Digit Computation of the Incomplete Beta
           Function Ratios. ACM Trans. Math. Softw. 18 (1993), 360-373.

    """)

# 向指定模块添加新的文档字符串，描述函数 `btdtrib` 的逆函数。
add_newdoc("btdtrib",
    r"""
    btdtria(a, p, x, out=None)

    Inverse of `btdtr` with respect to `b`.

    This is the inverse of the beta cumulative distribution function, `btdtr`,
    considered as a function of `b`, returning the value of `b` for which
    `btdtr(a, b, x) = p`, or

    .. math::
        p = \int_0^x \frac{\Gamma(a + b)}{\Gamma(a)\Gamma(b)} t^{a-1} (1-t)^{b-1}\,dt

    Parameters
    ----------
    a : array_like
        Shape parameter (`a` > 0).
    p : array_like
        Cumulative probability, in [0, 1].
    x : array_like
        The quantile, in [0, 1].
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    b : scalar or ndarray
        The value of the shape parameter `b` such that `btdtr(a, b, x) = p`.

    See Also
    --------
    btdtr : Cumulative distribution function of the beta distribution.
    btdtri : Inverse with respect to `x`.
    btdtria : Inverse with respect to `a`.

    Notes
    -----
    Wrapper for the CDFLIB [1]_ Fortran routine `cdfbet`.

    The cumulative distribution function `p` is computed using a routine by
    DiDinato and Morris [2]_. Computation of `b` involves a search for a value
    that produces the desired value of `p`. The search relies on the
    monotonicity of `p` with `b`.

    References
    ----------
    .. [1] Barry Brown, James Lovato, and Kathy Russell,
           CDFLIB: Library of Fortran Routines for Cumulative Distribution
           Functions, Inverses, and Other Parameters.
    .. [2] DiDinato, A. R. and Morris, A. H.,
           Algorithm 708: Significant Digit Computation of the Incomplete Beta
           Function Ratios. ACM Trans. Math. Softw. 18 (1993), 360-373.


注释：

    此处是一个长文档字符串（docstring），用于描述函数或模块的功能、输入、输出以及参考资料。

    函数的具体实现细节没有在这里展示，而是文档字符串用于提供函数背景和引用文献。
    
    文档字符串包含两部分内容：
    
    1. 描述了函数的目标，即生成期望的 `p` 值。搜索依赖于 `p` 随 `b` 的单调性。
    2. 引用了两个参考文献：
        - [1] Barry Brown, James Lovato, and Kathy Russell,
             有关累积分布函数、反函数和其他参数的Fortran例程的CDFLIB库。
        - [2] DiDinato, A. R. 和 Morris, A. H.,
             算法708：不完全Beta函数比率的有效数字计算。ACM Trans. Math. Softw. 18 (1993), 360-373。
    
    这些信息对于理解函数的背景和数学基础是非常重要的。
# 将函数 "besselpoly" 添加到模块 "besselpoly" 的文档中
add_newdoc("besselpoly",
    r"""
    besselpoly(a, lmb, nu, out=None)

    Weighted integral of the Bessel function of the first kind.

    Computes

    .. math::

       \int_0^1 x^\lambda J_\nu(2 a x) \, dx

    where :math:`J_\nu` is a Bessel function and :math:`\lambda=lmb`,
    :math:`\nu=nu`.

    Parameters
    ----------
    a : array_like
        Scale factor inside the Bessel function.
    lmb : array_like
        Power of `x`
    nu : array_like
        Order of the Bessel function.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Value of the integral.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    Evaluate the function for one parameter set.

    >>> from scipy.special import besselpoly
    >>> besselpoly(1, 1, 1)
    0.24449718372863877

    Evaluate the function for different scale factors.

    >>> import numpy as np
    >>> factors = np.array([0., 3., 6.])
    >>> besselpoly(factors, 1, 1)
    array([ 0.        , -0.00549029,  0.00140174])

    Plot the function for varying powers, orders and scales.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> powers = np.linspace(0, 10, 100)
    >>> orders = [1, 2, 3]
    >>> scales = [1, 2]
    >>> all_combinations = [(order, scale) for order in orders
    ...                     for scale in scales]
    >>> for order, scale in all_combinations:
    ...     ax.plot(powers, besselpoly(scale, powers, order),
    ...             label=rf"$\nu={order}, a={scale}$")
    >>> ax.legend()
    >>> ax.set_xlabel(r"$\lambda$")
    >>> ax.set_ylabel(r"$\int_0^1 x^{\lambda} J_{\nu}(2ax)\,dx$")
    >>> plt.show()
    """)

# 将函数 "beta" 添加到模块 "beta" 的文档中
add_newdoc("beta",
    r"""
    beta(a, b, out=None)

    Beta function.

    This function is defined in [1]_ as

    .. math::

        B(a, b) = \int_0^1 t^{a-1}(1-t)^{b-1}dt
                = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)},

    where :math:`\Gamma` is the gamma function.

    Parameters
    ----------
    a, b : array_like
        Real-valued arguments
    out : ndarray, optional
        Optional output array for the function result

    Returns
    -------
    scalar or ndarray
        Value of the beta function

    See Also
    --------
    gamma : the gamma function
    betainc :  the regularized incomplete beta function
    betaln : the natural logarithm of the absolute
             value of the beta function

    References
    ----------
    .. [1] NIST Digital Library of Mathematical Functions,
           Eq. 5.12.1. https://dlmf.nist.gov/5.12

    Examples
    --------
    >>> import scipy.special as sc

    The beta function relates to the gamma function by the
    definition given above:

    >>> sc.beta(2, 3)
    0.08333333333333333
    >>> sc.gamma(2)*sc.gamma(3)/sc.gamma(2 + 3)
    0.08333333333333333
    """)
    # 示例中的代码段是一个包含了注释的文档字符串（docstring），用于说明函数或模块的使用方法和示例。
    # 文档字符串通常用于提供函数的使用示例、输入输出的期望等信息，这里展示了使用标准库中的 scipy 的 beta 函数的几个示例。
    As this relationship demonstrates, the beta function
    is symmetric:

    >>> sc.beta(1.7, 2.4)
    0.16567527689031739
    >>> sc.beta(2.4, 1.7)
    0.16567527689031739

    This function satisfies :math:`B(1, b) = 1/b`:

    >>> sc.beta(1, 4)
    0.25

    """
# 在 "betainc" 函数中添加新的文档字符串
add_newdoc(
    "betainc",
    r"""
    betainc(a, b, x, out=None)

    Regularized incomplete beta function.

    Computes the regularized incomplete beta function, defined as [1]_:

    .. math::

        I_x(a, b) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} \int_0^x
        t^{a-1}(1-t)^{b-1}dt,

    for :math:`0 \leq x \leq 1`.

    This function is the cumulative distribution function for the beta
    distribution; its range is [0, 1].

    Parameters
    ----------
    a, b : array_like
           Positive, real-valued parameters
    x : array_like
        Real-valued such that :math:`0 \leq x \leq 1`,
        the upper limit of integration
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        Value of the regularized incomplete beta function

    See Also
    --------
    beta : beta function
    betaincinv : inverse of the regularized incomplete beta function
    betaincc : complement of the regularized incomplete beta function
    scipy.stats.beta : beta distribution

    Notes
    -----
    The term *regularized* in the name of this function refers to the
    scaling of the function by the gamma function terms shown in the
    formula.  When not qualified as *regularized*, the name *incomplete
    beta function* often refers to just the integral expression,
    without the gamma terms.  One can use the function `beta` from
    `scipy.special` to get this "nonregularized" incomplete beta
    function by multiplying the result of ``betainc(a, b, x)`` by
    ``beta(a, b)``.

    References
    ----------
    .. [1] NIST Digital Library of Mathematical Functions
           https://dlmf.nist.gov/8.17

    Examples
    --------

    Let :math:`B(a, b)` be the `beta` function.

    >>> import scipy.special as sc

    The coefficient in terms of `gamma` is equal to
    :math:`1/B(a, b)`. Also, when :math:`x=1`
    the integral is equal to :math:`B(a, b)`.
    Therefore, :math:`I_{x=1}(a, b) = 1` for any :math:`a, b`.

    >>> sc.betainc(0.2, 3.5, 1.0)
    1.0

    It satisfies
    :math:`I_x(a, b) = x^a F(a, 1-b, a+1, x)/ (aB(a, b))`,
    where :math:`F` is the hypergeometric function `hyp2f1`:

    >>> a, b, x = 1.4, 3.1, 0.5
    >>> x**a * sc.hyp2f1(a, 1 - b, a + 1, x)/(a * sc.beta(a, b))
    0.8148904036225295
    >>> sc.betainc(a, b, x)
    0.8148904036225296

    This functions satisfies the relationship
    :math:`I_x(a, b) = 1 - I_{1-x}(b, a)`:

    >>> sc.betainc(2.2, 3.1, 0.4)
    0.49339638807619446
    >>> 1 - sc.betainc(3.1, 2.2, 1 - 0.4)
    0.49339638807619446

    """)

# 在 "betaincc" 函数中添加新的文档字符串
add_newdoc(
    "betaincc",
    r"""
    betaincc(a, b, x, out=None)

    Complement of the regularized incomplete beta function.

    Computes the complement of the regularized incomplete beta function,
    defined as [1]_:

    ...
    .. math::

        \bar{I}_x(a, b) = 1 - I_x(a, b)
                        = 1 - \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} \int_0^x
                                  t^{a-1}(1-t)^{b-1}dt,

    for :math:`0 \leq x \leq 1`.

    Parameters
    ----------
    a, b : array_like
           Positive, real-valued parameters
           参数 a 和 b：数组类型，正实数参数

    x : array_like
        Real-valued such that :math:`0 \leq x \leq 1`,
        the upper limit of integration
        参数 x：数组类型，实数，满足 :math:`0 \leq x \leq 1`，积分的上限

    out : ndarray, optional
        Optional output array for the function values
        out：可选的输出数组，用于存储函数值

    Returns
    -------
    scalar or ndarray
        Value of the regularized incomplete beta function
        标量或数组：正则化不完全贝塔函数的值

    See Also
    --------
    betainc : regularized incomplete beta function
    betaincinv : inverse of the regularized incomplete beta function
    betainccinv :
        inverse of the complement of the regularized incomplete beta function
    beta : beta function
    scipy.stats.beta : beta distribution

    Notes
    -----
    .. versionadded:: 1.11.0
    版本说明：1.11.0 版本新增功能

    References
    ----------
    .. [1] NIST Digital Library of Mathematical Functions
           https://dlmf.nist.gov/8.17
    参考资料：NIST 数学函数数字图书馆，https://dlmf.nist.gov/8.17

    Examples
    --------
    >>> from scipy.special import betaincc, betainc

    The naive calculation ``1 - betainc(a, b, x)`` loses precision when
    the values of ``betainc(a, b, x)`` are close to 1:

    >>> 1 - betainc(0.5, 8, [0.9, 0.99, 0.999])
    array([2.0574632e-09, 0.0000000e+00, 0.0000000e+00])

    By using ``betaincc``, we get the correct values:

    >>> betaincc(0.5, 8, [0.9, 0.99, 0.999])
    array([2.05746321e-09, 1.97259354e-17, 1.96467954e-25])
    示例：使用 betaincc 可以得到正确的值
# 向 scipy.special 模块中的 betaincinv 函数添加新文档字符串
add_newdoc(
    "betaincinv",
    r"""
    betaincinv(a, b, y, out=None)

    Inverse of the regularized incomplete beta function.

    Computes :math:`x` such that:

    .. math::

        y = I_x(a, b) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}
        \int_0^x t^{a-1}(1-t)^{b-1}dt,

    where :math:`I_x` is the normalized incomplete beta function `betainc`
    and :math:`\Gamma` is the `gamma` function [1]_.

    Parameters
    ----------
    a, b : array_like
        Positive, real-valued parameters
    y : array_like
        Real-valued input
    out : ndarray, optional
        Optional output array for function values

    Returns
    -------
    scalar or ndarray
        Value of the inverse of the regularized incomplete beta function

    See Also
    --------
    betainc : regularized incomplete beta function
    gamma : gamma function

    References
    ----------
    .. [1] NIST Digital Library of Mathematical Functions
           https://dlmf.nist.gov/8.17

    Examples
    --------
    >>> import scipy.special as sc

    This function is the inverse of `betainc` for fixed
    values of :math:`a` and :math:`b`.

    >>> a, b = 1.2, 3.1
    >>> y = sc.betainc(a, b, 0.2)
    >>> sc.betaincinv(a, b, y)
    0.2
    >>>
    >>> a, b = 7.5, 0.4
    >>> x = sc.betaincinv(a, b, 0.5)
    >>> sc.betainc(a, b, x)
    0.5

    """)


# 向 scipy.special 模块中的 betainccinv 函数添加新文档字符串
add_newdoc(
    "betainccinv",
    r"""
    betainccinv(a, b, y, out=None)

    Inverse of the complemented regularized incomplete beta function.

    Computes :math:`x` such that:

    .. math::

        y = 1 - I_x(a, b) = 1 - \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}
        \int_0^x t^{a-1}(1-t)^{b-1}dt,

    where :math:`I_x` is the normalized incomplete beta function `betainc`
    and :math:`\Gamma` is the `gamma` function [1]_.

    Parameters
    ----------
    a, b : array_like
        Positive, real-valued parameters
    y : array_like
        Real-valued input
    out : ndarray, optional
        Optional output array for function values

    Returns
    -------
    scalar or ndarray
        Value of the inverse of the regularized incomplete beta function

    See Also
    --------
    betainc : regularized incomplete beta function
    betaincc : complement of the regularized incomplete beta function

    Notes
    -----
    .. versionadded:: 1.11.0

    References
    ----------
    .. [1] NIST Digital Library of Mathematical Functions
           https://dlmf.nist.gov/8.17

    Examples
    --------
    >>> from scipy.special import betainccinv, betaincc

    This function is the inverse of `betaincc` for fixed
    values of :math:`a` and :math:`b`.

    >>> a, b = 1.2, 3.1
    >>> y = betaincc(a, b, 0.2)
    >>> betainccinv(a, b, y)
    0.2

    >>> a, b = 7, 2.5
    >>> x = betainccinv(a, b, 0.875)
    >>> betaincc(a, b, x)
    0.875

    """)


# 向 scipy.special 模块中的 betaln 函数添加新文档字符串
add_newdoc("betaln",
    """
    betaln(a, b, out=None)

    Natural logarithm of absolute value of beta function.

    Computes ``ln(abs(beta(a, b)))``.
    Parameters
    ----------
    a, b : array_like
        Positive, real-valued parameters
        参数 a 和 b：类数组，必须为正实数参数

    out : ndarray, optional
        Optional output array for function values
        可选的输出数组，用于存储函数计算结果的 ndarray

    Returns
    -------
    scalar or ndarray
        Value of the betaln function
        betaln 函数的返回值，可以是标量或者 ndarray

    See Also
    --------
    gamma : the gamma function
    betainc :  the regularized incomplete beta function
    beta : the beta function
    参见：
    gamma：伽玛函数
    betainc：正则化不完全贝塔函数
    beta：贝塔函数

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import betaln, beta

    Verify that, for moderate values of ``a`` and ``b``, ``betaln(a, b)``
    is the same as ``log(beta(a, b))``:
    验证对于适度的 ``a`` 和 ``b`` 值，``betaln(a, b)`` 是否等同于 ``log(beta(a, b))``：

    >>> betaln(3, 4)
    -4.0943445622221

    >>> np.log(beta(3, 4))
    -4.0943445622221

    In the following ``beta(a, b)`` underflows to 0, so we can't compute
    the logarithm of the actual value.
    在以下情况下，``beta(a, b)`` 下溢为 0，因此无法计算实际值的对数。

    >>> a = 400
    >>> b = 900
    >>> beta(a, b)
    0.0

    We can compute the logarithm of ``beta(a, b)`` by using `betaln`:
    我们可以通过使用 `betaln` 计算 ``beta(a, b)`` 的对数：

    >>> betaln(a, b)
    -804.3069951764146
# 将文档添加到 "boxcox" 函数中
add_newdoc("boxcox",
    """
    boxcox(x, lmbda, out=None)

    Compute the Box-Cox transformation.

    The Box-Cox transformation is::

        y = (x**lmbda - 1) / lmbda  if lmbda != 0
            log(x)                  if lmbda == 0

    Returns `nan` if ``x < 0``.
    Returns `-inf` if ``x == 0`` and ``lmbda < 0``.

    Parameters
    ----------
    x : array_like
        Data to be transformed.
    lmbda : array_like
        Power parameter of the Box-Cox transform.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    y : scalar or ndarray
        Transformed data.

    Notes
    -----

    .. versionadded:: 0.14.0

    Examples
    --------
    >>> from scipy.special import boxcox
    >>> boxcox([1, 4, 10], 2.5)
    array([   0.        ,   12.4       ,  126.09110641])
    >>> boxcox(2, [0, 1, 2])
    array([ 0.69314718,  1.        ,  1.5       ])
    """)

# 将文档添加到 "boxcox1p" 函数中
add_newdoc("boxcox1p",
    """
    boxcox1p(x, lmbda, out=None)

    Compute the Box-Cox transformation of 1 + `x`.

    The Box-Cox transformation computed by `boxcox1p` is::

        y = ((1+x)**lmbda - 1) / lmbda  if lmbda != 0
            log(1+x)                    if lmbda == 0

    Returns `nan` if ``x < -1``.
    Returns `-inf` if ``x == -1`` and ``lmbda < 0``.

    Parameters
    ----------
    x : array_like
        Data to be transformed.
    lmbda : array_like
        Power parameter of the Box-Cox transform.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    y : scalar or ndarray
        Transformed data.

    Notes
    -----

    .. versionadded:: 0.14.0

    Examples
    --------
    >>> from scipy.special import boxcox1p
    >>> boxcox1p(1e-4, [0, 0.5, 1])
    array([  9.99950003e-05,   9.99975001e-05,   1.00000000e-04])
    >>> boxcox1p([0.01, 0.1], 0.25)
    array([ 0.00996272,  0.09645476])
    """)

# 将文档添加到 "inv_boxcox" 函数中
add_newdoc("inv_boxcox",
    """
    inv_boxcox(y, lmbda, out=None)

    Compute the inverse of the Box-Cox transformation.

    Find ``x`` such that::

        y = (x**lmbda - 1) / lmbda  if lmbda != 0
            log(x)                  if lmbda == 0

    Parameters
    ----------
    y : array_like
        Data to be transformed.
    lmbda : array_like
        Power parameter of the Box-Cox transform.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    x : scalar or ndarray
        Transformed data.

    Notes
    -----

    .. versionadded:: 0.16.0

    Examples
    --------
    >>> from scipy.special import boxcox, inv_boxcox
    >>> y = boxcox([1, 4, 10], 2.5)
    >>> inv_boxcox(y, 2.5)
    array([1., 4., 10.])
    """)

# 将文档添加到 "inv_boxcox1p" 函数中
add_newdoc("inv_boxcox1p",
    """
    inv_boxcox1p(y, lmbda, out=None)

    Compute the inverse of the Box-Cox transformation.

    Find ``x`` such that::

        y = ((1+x)**lmbda - 1) / lmbda  if lmbda != 0
            log(1+x)                    if lmbda == 0
    """)
    # Parameters 定义函数参数说明部分，以下是函数 boxcox1p 的参数说明：
    # y : array_like
    #     待转换的数据。
    # lmbda : array_like
    #     Box-Cox 变换的幂参数。
    # out : ndarray, optional
    #     可选参数，用于存储函数值的输出数组。

    # Returns 返回值说明部分，函数返回转换后的数据 x：
    # x : scalar or ndarray
    #     转换后的数据。

    # Notes 注意事项部分，版本 0.16.0 中添加了这个函数。

    # Examples 示例部分，演示了如何使用 boxcox1p 和 inv_boxcox1p 函数：
    # >>> from scipy.special import boxcox1p, inv_boxcox1p
    # >>> y = boxcox1p([1, 4, 10], 2.5)
    # >>> inv_boxcox1p(y, 2.5)
    # array([1., 4., 10.])
# 添加新的文档字符串给名为 "btdtr" 的函数
add_newdoc("btdtr",
    r"""
    btdtr(a, b, x, out=None)

    Beta分布的累积分布函数。

    返回从零到 `x` 的beta概率密度函数的积分，

    .. math::
        I = \int_0^x \frac{\Gamma(a + b)}{\Gamma(a)\Gamma(b)} t^{a-1} (1-t)^{b-1}\,dt

    其中 :math:`\Gamma` 是gamma函数。

    .. deprecated:: 1.12.0
        此函数已弃用，并将在SciPy 1.14.0中删除。
        请使用 `scipy.special.betainc` 替代。

    Parameters
    ----------
    a : array_like
        形状参数 (a > 0)。
    b : array_like
        形状参数 (b > 0)。
    x : array_like
        积分的上限，范围在 [0, 1] 之间。
    out : ndarray, optional
        可选的输出数组，用于存储函数值

    Returns
    -------
    I : scalar or ndarray
        beta分布的累积分布函数，在参数 `a` 和 `b`，在 `x` 处的值。

    See Also
    --------
    betainc

    Notes
    -----
    此函数与不完全贝塔积分函数 `betainc` 相同。

    Cephes [1]_ 库中 `btdtr` 函数的封装。

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    """)

# 添加新的文档字符串给名为 "btdtri" 的函数
add_newdoc("btdtri",
    r"""
    btdtri(a, b, p, out=None)

    Beta分布的第 `p` 个分位数。

    此函数是beta累积分布函数 `btdtr` 的逆函数，返回使得 `btdtr(a, b, x) = p` 的 `x` 值，或者

    .. math::
        p = \int_0^x \frac{\Gamma(a + b)}{\Gamma(a)\Gamma(b)} t^{a-1} (1-t)^{b-1}\,dt

    .. deprecated:: 1.12.0
        此函数已弃用，并将在SciPy 1.14.0中删除。
        请使用 `scipy.special.betaincinv` 替代。

    Parameters
    ----------
    a : array_like
        形状参数 (`a` > 0)。
    b : array_like
        形状参数 (`b` > 0)。
    p : array_like
        累积概率，范围在 [0, 1] 之间。
    out : ndarray, optional
        可选的输出数组，用于存储函数值

    Returns
    -------
    x : scalar or ndarray
        对应于 `p` 的分位数。

    See Also
    --------
    betaincinv
    btdtr

    Notes
    -----
    通过区间二分法或牛顿迭代法找到 `x` 的值。

    Cephes [1]_ 库中 `incbi` 函数的封装，解决等效于求不完全贝塔积分的逆的问题。

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    """)

# 添加新的文档字符串给名为 "cbrt" 的函数
add_newdoc("cbrt",
    """
    cbrt(x, out=None)

    `x` 的逐元素立方根。

    Parameters
    ----------
    x : array_like
        `x` 必须包含实数。
    out : ndarray, optional
        可选的输出数组，用于存储函数值

    Returns
    -------
    scalar or ndarray
        `x` 中每个值的立方根。

    Examples
    --------
    ```python
    >>> cbrt([1, 8, 27])
    array([1., 2., 3.])
    ```
    """
)
    # 导入 scipy.special 模块中的 cbrt 函数，用于计算立方根
    >>> from scipy.special import cbrt
    
    # 使用 cbrt 函数计算 8 的立方根，返回结果为 2.0
    >>> cbrt(8)
    2.0
    
    # 使用 cbrt 函数计算列表 [-8, -3, 0.125, 1.331] 中各元素的立方根，返回结果为一个 NumPy 数组
    >>> cbrt([-8, -3, 0.125, 1.331])
    array([-2.        , -1.44224957,  0.5       ,  1.1       ])
add_newdoc("chdtri",
    r"""
    chdtri(p, v)

    Inverse of the Chi square cumulative distribution function.

    Returns the inverse of the cumulative distribution function (i.e., quantile)
    for the Chi square distribution with `v` degrees of freedom at probability `p`.

    Parameters
    ----------
    p : array_like
        Probability values.
    v : array_like
        Degrees of freedom.

    Returns
    -------
    ndarray
        Values of the inverse cumulative distribution function.

    See Also
    --------
    chdtr, chdtrc, chdtriv

    Notes
    -----
    The inverse cumulative distribution function is computed using the
    relationship with the regularized lower incomplete gamma function `gammaincinv`.

    References
    ----------
    .. [1] Chi-Square distribution,
        https://www.itl.nist.gov/div898/handbook/eda/section3/eda3666.htm

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    Compute the inverse cumulative distribution function for a Chi square
    distribution with 2 degrees of freedom at probabilities 0.1 and 0.9.

    >>> v = 2
    >>> p = [0.1, 0.9]
    >>> sc.chdtri(p, v)
    array([0.21034037, 4.60517019])
    >>> sc.gammaincinv(v / 2, p)
    array([0.21034037, 4.60517019])

    """)
    chdtri(v, p, out=None)

    # `chdtri`函数是对`chdtrc`的逆操作，用于求解满足 `chdtrc(v, x) == p` 的 `x` 值。

    Parameters
    ----------
    v : array_like
        # 自由度，通常是一个数组或类数组对象。
    p : array_like
        # 概率，通常是一个数组或类数组对象。
    out : ndarray, optional
        # 可选的输出数组，用于存储函数结果。

    Returns
    -------
    x : scalar or ndarray
        # 满足条件的值 `x`，使得自由度为 `v` 的卡方随机变量大于 `x` 的概率等于 `p`。

    See Also
    --------
    chdtrc, chdtr, chdtriv
        # 相关函数：`chdtrc`、`chdtr`、`chdtriv`

    References
    ----------
    .. [1] Chi-Square distribution,
        https://www.itl.nist.gov/div898/handbook/eda/section3/eda3666.htm
        # 参考文献：卡方分布的相关信息，详见链接。

    Examples
    --------
    >>> import scipy.special as sc

    # `chdtri`函数的示例用法。

    It inverts `chdtrc`.

    >>> v, p = 1, 0.3
    >>> sc.chdtrc(v, sc.chdtri(v, p))
    0.3
    >>> x = 1
    >>> sc.chdtri(v, sc.chdtrc(v, x))
    1.0
# 添加新的文档字符串到模块 `chdtriv`
add_newdoc("chdtriv",
    """
    chdtriv(p, x, out=None)

    `chdtr` 的逆函数，关于 `v` 的逆。

    返回 `v`，使得 ``chdtr(v, x) == p``。

    Parameters
    ----------
    p : array_like
        Chi square 随机变量小于或等于 `x` 的概率。
    x : array_like
        非负输入。
    out : ndarray, optional
        函数结果的可选输出数组。

    Returns
    -------
    scalar or ndarray
        自由度。

    See Also
    --------
    chdtr, chdtrc, chdtri

    References
    ----------
    .. [1] Chi-Square distribution,
        https://www.itl.nist.gov/div898/handbook/eda/section3/eda3666.htm

    Examples
    --------
    >>> import scipy.special as sc

    反转 `chdtr`。

    >>> p, x = 0.5, 1
    >>> sc.chdtr(sc.chdtriv(p, x), x)
    0.5000000000202172
    >>> v = 1
    >>> sc.chdtriv(sc.chdtr(v, x), v)
    1.0000000000000013

    """)

# 添加新的文档字符串到模块 `chndtr`
add_newdoc("chndtr",
    r"""
    chndtr(x, df, nc, out=None)

    非中心卡方累积分布函数

    累积分布函数定义如下：

    .. math::

        P(\chi^{\prime 2} \vert \nu, \lambda) =\sum_{j=0}^{\infty}
        e^{-\lambda /2}
        \frac{(\lambda /2)^j}{j!} P(\chi^{\prime 2} \vert \nu + 2j),

    其中 :math:`\nu > 0` 是自由度 (``df``)，:math:`\lambda \geq 0` 是非中心参数 (``nc``)。

    Parameters
    ----------
    x : array_like
        积分的上限；必须满足 ``x >= 0``
    df : array_like
        自由度；必须满足 ``df > 0``
    nc : array_like
        非中心参数；必须满足 ``nc >= 0``
    out : ndarray, optional
        函数结果的可选输出数组

    Returns
    -------
    x : scalar or ndarray
        非中心卡方累积分布函数的值。

    See Also
    --------
    chndtrix, chndtridf, chndtrinc

    """)

# 添加新的文档字符串到模块 `chndtrix`
add_newdoc("chndtrix",
    """
    chndtrix(p, df, nc, out=None)

    `chndtr` 的逆函数，关于 `x`。

    使用搜索找到一个 `x` 的值，使得期望的 `p` 值成立。

    Parameters
    ----------
    p : array_like
        概率；必须满足 ``0 <= p < 1``
    df : array_like
        自由度；必须满足 ``df > 0``
    nc : array_like
        非中心参数；必须满足 ``nc >= 0``
    out : ndarray, optional
        函数结果的可选输出数组

    Returns
    -------
    x : scalar or ndarray
        使得具有 `df` 自由度和非中心性 `nc` 的非中心卡方随机变量大于 `x` 的概率等于 `p` 的值。

    See Also
    --------
    chndtr, chndtridf, chndtrinc

    """)

# 添加新的文档字符串到模块 `chndtridf`
add_newdoc("chndtridf",
    """
    chndtridf(x, p, nc, out=None)

    `chndtr` 的逆函数，关于 `df`。

    使用搜索找到一个 `df` 的值，使得期望的 `p` 值成立。
    
    """)
    desired value of `p`.

    Parameters
    ----------
    x : array_like
        Upper bound of the integral; must satisfy ``x >= 0``
    p : array_like
        Probability; must satisfy ``0 <= p < 1``
    nc : array_like
        Non-centrality parameter; must satisfy ``nc >= 0``
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    df : scalar or ndarray
        Degrees of freedom

    See Also
    --------
    chndtr, chndtrix, chndtrinc

    """
# 添加新文档字符串给模块 "chndtrinc"
add_newdoc("chndtrinc",
    """
    chndtrinc(x, df, p, out=None)

    Inverse to `chndtr` vs `nc`

    Calculated using a search to find a value for `df` that produces the
    desired value of `p`.

    Parameters
    ----------
    x : array_like
        Upper bound of the integral; must satisfy ``x >= 0``
    df : array_like
        Degrees of freedom; must satisfy ``df > 0``
    p : array_like
        Probability; must satisfy ``0 <= p < 1``
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    nc : scalar or ndarray
        Non-centrality

    See Also
    --------
    chndtr, chndtrix, chndtrinc
    """
)

# 添加新文档字符串给函数 "cosdg"
add_newdoc("cosdg",
    """
    cosdg(x, out=None)

    Cosine of the angle `x` given in degrees.

    Parameters
    ----------
    x : array_like
        Angle, given in degrees.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Cosine of the input.

    See Also
    --------
    sindg, tandg, cotdg

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    It is more accurate than using cosine directly.

    >>> x = 90 + 180 * np.arange(3)
    >>> sc.cosdg(x)
    array([-0.,  0., -0.])
    >>> np.cos(x * np.pi / 180)
    array([ 6.1232340e-17, -1.8369702e-16,  3.0616170e-16])

    """
)

# 添加新文档字符串给函数 "cosm1"
add_newdoc("cosm1",
    """
    cosm1(x, out=None)

    cos(x) - 1 for use when `x` is near zero.

    Parameters
    ----------
    x : array_like
        Real valued argument.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Values of ``cos(x) - 1``.

    See Also
    --------
    expm1, log1p

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    It is more accurate than computing ``cos(x) - 1`` directly for
    ``x`` around 0.

    >>> x = 1e-30
    >>> np.cos(x) - 1
    0.0
    >>> sc.cosm1(x)
    -5.0000000000000005e-61

    """
)

# 添加新文档字符串给函数 "cotdg"
add_newdoc("cotdg",
    """
    cotdg(x, out=None)

    Cotangent of the angle `x` given in degrees.

    Parameters
    ----------
    x : array_like
        Angle, given in degrees.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Cotangent at the input.

    See Also
    --------
    sindg, cosdg, tandg

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    It is more accurate than using cotangent directly.

    >>> x = 90 + 180 * np.arange(3)
    >>> sc.cotdg(x)
    array([0., 0., 0.])
    >>> 1 / np.tan(x * np.pi / 180)
    array([6.1232340e-17, 1.8369702e-16, 3.0616170e-16])

    """
)

# 添加新文档字符串给函数 "dawsn"
add_newdoc("dawsn",
    """
    dawsn(x, out=None)

    Dawson's integral.

    Computes::

        exp(-x**2) * integral(exp(t**2), t=0..x).

    Parameters
    ----------
    x : array_like
        Real valued argument.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        The value of Dawson's integral at `x`.

    """
)
    x : array_like
        函数参数，接受类数组类型的输入 x。
    out : ndarray, optional
        可选参数，用于存储函数计算结果的输出数组。

    Returns
    -------
    y : scalar or ndarray
        函数的积分值，可以是标量或者多维数组。

    See Also
    --------
    wofz, erf, erfc, erfcx, erfi
        相关函数：wofz, erf, erfc, erfcx, erfi

    References
    ----------
    .. [1] Steven G. Johnson, Faddeeva W function implementation.
       参考文献：Steven G. Johnson, Faddeeva W 函数的实现。
       http://ab-initio.mit.edu/Faddeeva
       相关链接：http://ab-initio.mit.edu/Faddeeva

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import special
    >>> import matplotlib.pyplot as plt
    导入必要的库：numpy、scipy.special 和 matplotlib.pyplot。
    >>> x = np.linspace(-15, 15, num=1000)
    生成从 -15 到 15 的 1000 个等间距点的数组 x。
    >>> plt.plot(x, special.dawsn(x))
    绘制函数 dawsn(x) 在 x 范围内的图像。
    >>> plt.xlabel('$x$')
    设置 x 轴标签。
    >>> plt.ylabel('$dawsn(x)$')
    设置 y 轴标签。
    >>> plt.show()
    显示绘制的图像。

    """
# 将文档字符串添加到"ellipe"函数的文档中，提供了关于完全椭圆积分第二类的详细描述和用法说明
add_newdoc("ellipe",
    r"""
    ellipe(m, out=None)

    Complete elliptic integral of the second kind

    This function is defined as

    .. math:: E(m) = \int_0^{\pi/2} [1 - m \sin(t)^2]^{1/2} dt

    Parameters
    ----------
    m : array_like
        Defines the parameter of the elliptic integral.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    E : scalar or ndarray
        Value of the elliptic integral.

    See Also
    --------
    ellipkm1 : Complete elliptic integral of the first kind, near `m` = 1
    ellipk : Complete elliptic integral of the first kind
    ellipkinc : Incomplete elliptic integral of the first kind
    ellipeinc : Incomplete elliptic integral of the second kind
    elliprd : Symmetric elliptic integral of the second kind.
    elliprg : Completely-symmetric elliptic integral of the second kind.

    Notes
    -----
    Wrapper for the Cephes [1]_ routine `ellpe`.

    For ``m > 0`` the computation uses the approximation,

    .. math:: E(m) \approx P(1-m) - (1-m) \log(1-m) Q(1-m),

    where :math:`P` and :math:`Q` are tenth-order polynomials.  For
    ``m < 0``, the relation

    .. math:: E(m) = E(m/(m - 1)) \sqrt(1-m)

    is used.

    The parameterization in terms of :math:`m` follows that of section
    17.2 in [2]_. Other parameterizations in terms of the
    complementary parameter :math:`1 - m`, modular angle
    :math:`\sin^2(\alpha) = m`, or modulus :math:`k^2 = m` are also
    used, so be careful that you choose the correct parameter.

    The Legendre E integral is related to Carlson's symmetric R_D or R_G
    functions in multiple ways [3]_. For example,

    .. math:: E(m) = 2 R_G(0, 1-k^2, 1) .

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    .. [2] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.
    .. [3] NIST Digital Library of Mathematical
           Functions. http://dlmf.nist.gov/, Release 1.0.28 of
           2020-09-15. See Sec. 19.25(i) https://dlmf.nist.gov/19.25#i

    Examples
    --------
    This function is used in finding the circumference of an
    ellipse with semi-major axis `a` and semi-minor axis `b`.

    >>> import numpy as np
    >>> from scipy import special

    >>> a = 3.5
    >>> b = 2.1
    >>> e_sq = 1.0 - b**2/a**2  # eccentricity squared

    Then the circumference is found using the following:

    >>> C = 4*a*special.ellipe(e_sq)  # circumference formula
    >>> C
    17.868899204378693

    When `a` and `b` are the same (meaning eccentricity is 0),
    this reduces to the circumference of a circle.

    >>> 4*a*special.ellipe(0.0)  # formula for ellipse with a = b
    21.991148575128552
    >>> 2*np.pi*a  # formula for circle of radius a
    21.991148575128552

    """)

# 将文档字符串添加到"ellipeinc"函数的文档中，提供了关于不完全椭圆积分第二类的详细描述和用法说明
add_newdoc("ellipeinc",
    r"""
    ```
    ellipeinc(phi, m, out=None)
    # 计算第二类不完全椭圆积分

    Incomplete elliptic integral of the second kind
    # 第二类不完全椭圆积分的定义

    This function is defined as
    # 函数定义如下

    .. math:: E(\phi, m) = \int_0^{\phi} [1 - m \sin(t)^2]^{1/2} dt
    # 其中 E(\phi, m) 表示不完全椭圆积分的第二类，积分表达式中包含参数 m

    Parameters
    ----------
    phi : array_like
        amplitude of the elliptic integral.
    # 椭圆积分的振幅参数 phi

    m : array_like
        parameter of the elliptic integral.
    # 椭圆积分的参数 m

    out : ndarray, optional
        Optional output array for the function values
    # 可选参数，用于存储函数值的输出数组

    Returns
    -------
    E : scalar or ndarray
        Value of the elliptic integral.
    # 返回值，椭圆积分的值，可以是标量或数组

    See Also
    --------
    ellipkm1 : Complete elliptic integral of the first kind, near `m` = 1
    ellipk : Complete elliptic integral of the first kind
    ellipkinc : Incomplete elliptic integral of the first kind
    ellipe : Complete elliptic integral of the second kind
    elliprd : Symmetric elliptic integral of the second kind.
    elliprf : Completely-symmetric elliptic integral of the first kind.
    elliprg : Completely-symmetric elliptic integral of the second kind.
    # 相关函数，与第一类和第二类椭圆积分相关的其他函数

    Notes
    -----
    Wrapper for the Cephes [1]_ routine `ellie`.
    # 使用 Cephes 库中的 ellie 程序包装器进行计算

    Computation uses arithmetic-geometric means algorithm.
    # 使用算术-几何均值算法进行计算

    The parameterization in terms of :math:`m` follows that of section
    17.2 in [2]_. Other parameterizations in terms of the
    complementary parameter :math:`1 - m`, modular angle
    :math:`\sin^2(\alpha) = m`, or modulus :math:`k^2 = m` are also
    used, so be careful that you choose the correct parameter.
    # 参数化的说明，关于参数 m 的选择及其不同的表示方式

    The Legendre E incomplete integral can be related to combinations
    of Carlson's symmetric integrals R_D, R_F, and R_G in multiple
    ways [3]_. For example, with :math:`c = \csc^2\phi`,
    # Legendre E 不完全积分可以用 Carlson 对称积分 R_D, R_F 和 R_G 的组合表示

    .. math::
      E(\phi, m) = R_F(c-1, c-k^2, c)
        - \frac{1}{3} k^2 R_D(c-1, c-k^2, c) .
    # 其中的一个例子，展示了 E(\phi, m) 的一个表达式

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    # 参考文献 1，Cephes 数学函数库

    .. [2] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.
    # 参考文献 2，数学函数手册

    .. [3] NIST Digital Library of Mathematical
           Functions. http://dlmf.nist.gov/, Release 1.0.28 of
           2020-09-15. See Sec. 19.25(i) https://dlmf.nist.gov/19.25#i
    # 参考文献 3，NIST 数学函数数字库
    ```
# 添加新的文档字符串到函数 "ellipj"
add_newdoc("ellipj",
    """
    ellipj(u, m, out=None)

    Jacobian elliptic functions

    Calculates the Jacobian elliptic functions of parameter `m` between
    0 and 1, and real argument `u`.

    Parameters
    ----------
    m : array_like
        Parameter.
    u : array_like
        Argument.
    out : tuple of ndarray, optional
        Optional output arrays for the function values

    Returns
    -------
    sn, cn, dn, ph : 4-tuple of scalar or ndarray
        The returned functions::

            sn(u|m), cn(u|m), dn(u|m)

        The value `ph` is such that if `u = ellipkinc(ph, m)`,
        then `sn(u|m) = sin(ph)` and `cn(u|m) = cos(ph)`.

    See Also
    --------
    ellipk : Complete elliptic integral of the first kind
    ellipkinc : Incomplete elliptic integral of the first kind

    Notes
    -----
    Wrapper for the Cephes [1]_ routine `ellpj`.

    These functions are periodic, with quarter-period on the real axis
    equal to the complete elliptic integral `ellipk(m)`.

    Relation to incomplete elliptic integral: If `u = ellipkinc(phi,m)`, then
    `sn(u|m) = sin(phi)`, and `cn(u|m) = cos(phi)`. The `phi` is called
    the amplitude of `u`.

    Computation is by means of the arithmetic-geometric mean algorithm,
    except when `m` is within 1e-9 of 0 or 1. In the latter case with `m`
    close to 1, the approximation applies only for `phi < pi/2`.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    """)

# 添加新的文档字符串到函数 "ellipkm1"
add_newdoc("ellipkm1",
    """
    ellipkm1(p, out=None)

    Complete elliptic integral of the first kind around `m` = 1

    This function is defined as

    .. math:: K(p) = \\int_0^{\\pi/2} [1 - m \\sin(t)^2]^{-1/2} dt

    where `m = 1 - p`.

    Parameters
    ----------
    p : array_like
        Defines the parameter of the elliptic integral as `m = 1 - p`.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    K : scalar or ndarray
        Value of the elliptic integral.

    See Also
    --------
    ellipk : Complete elliptic integral of the first kind
    ellipkinc : Incomplete elliptic integral of the first kind
    ellipe : Complete elliptic integral of the second kind
    ellipeinc : Incomplete elliptic integral of the second kind
    elliprf : Completely-symmetric elliptic integral of the first kind.

    Notes
    -----
    Wrapper for the Cephes [1]_ routine `ellpk`.

    For ``p <= 1``, computation uses the approximation,

    .. math:: K(p) \\approx P(p) - \\log(p) Q(p),

    where :math:`P` and :math:`Q` are tenth-order polynomials.  The
    argument `p` is used internally rather than `m` so that the logarithmic
    singularity at ``m = 1`` will be shifted to the origin; this preserves
    maximum accuracy.  For ``p > 1``, the identity

    .. math:: K(p) = K(1/p)/\\sqrt(p)

    is used.

    References
    ----------

    """)
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    """



# 注释引用了 Cephes 数学函数库的信息，提供了网址链接
# 添加新文档到函数 "ellipk"
add_newdoc("ellipk",
    r"""
    ellipk(m, out=None)

    Complete elliptic integral of the first kind.

    This function is defined as

    .. math:: K(m) = \int_0^{\pi/2} [1 - m \sin(t)^2]^{-1/2} dt

    Parameters
    ----------
    m : array_like
        The parameter of the elliptic integral.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    K : scalar or ndarray
        Value of the elliptic integral.

    See Also
    --------
    ellipkm1 : Complete elliptic integral of the first kind around m = 1
    ellipkinc : Incomplete elliptic integral of the first kind
    ellipe : Complete elliptic integral of the second kind
    ellipeinc : Incomplete elliptic integral of the second kind
    elliprf : Completely-symmetric elliptic integral of the first kind.

    Notes
    -----
    For more precision around point m = 1, use `ellipkm1`, which this
    function calls.

    The parameterization in terms of :math:`m` follows that of section
    17.2 in [1]_. Other parameterizations in terms of the
    complementary parameter :math:`1 - m`, modular angle
    :math:`\sin^2(\alpha) = m`, or modulus :math:`k^2 = m` are also
    used, so be careful that you choose the correct parameter.

    The Legendre K integral is related to Carlson's symmetric R_F
    function by [2]_:

    .. math:: K(m) = R_F(0, 1-k^2, 1) .

    References
    ----------
    .. [1] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.
    .. [2] NIST Digital Library of Mathematical
           Functions. http://dlmf.nist.gov/, Release 1.0.28 of
           2020-09-15. See Sec. 19.25(i) https://dlmf.nist.gov/19.25#i

    """)

# 添加新文档到函数 "ellipkinc"
add_newdoc("ellipkinc",
    r"""
    ellipkinc(phi, m, out=None)

    Incomplete elliptic integral of the first kind

    This function is defined as

    .. math:: K(\phi, m) = \int_0^{\phi} [1 - m \sin(t)^2]^{-1/2} dt

    This function is also called :math:`F(\phi, m)`.

    Parameters
    ----------
    phi : array_like
        amplitude of the elliptic integral
    m : array_like
        parameter of the elliptic integral
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    K : scalar or ndarray
        Value of the elliptic integral

    See Also
    --------
    ellipkm1 : Complete elliptic integral of the first kind, near `m` = 1
    ellipk : Complete elliptic integral of the first kind
    ellipe : Complete elliptic integral of the second kind
    ellipeinc : Incomplete elliptic integral of the second kind
    elliprf : Completely-symmetric elliptic integral of the first kind.

    Notes
    -----
    Wrapper for the Cephes [1]_ routine `ellik`.  The computation is
    carried out using the arithmetic-geometric mean algorithm.

    The parameterization in terms of :math:`m` follows that of section
    17.3 in [1]_. Other parameterizations in terms of the
    complementary parameter :math:`1 - m`, modular angle
    :math:`\sin^2(\alpha) = m`, or modulus :math:`k^2 = m` are also
    used, so be careful that you choose the correct parameter.

    References
    ----------
    .. [1] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.
    """)
    17.2 in [2]_. Other parameterizations in terms of the
    complementary parameter :math:`1 - m`, modular angle
    :math:`\sin^2(\alpha) = m`, or modulus :math:`k^2 = m` are also
    used, so be careful that you choose the correct parameter.

    The Legendre K incomplete integral (or F integral) is related to
    Carlson's symmetric R_F function [3]_.
    Setting :math:`c = \csc^2\phi`,

    .. math:: F(\phi, m) = R_F(c-1, c-k^2, c) .

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    .. [2] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.
    .. [3] NIST Digital Library of Mathematical
           Functions. http://dlmf.nist.gov/, Release 1.0.28 of
           2020-09-15. See Sec. 19.25(i) https://dlmf.nist.gov/19.25#i
# 将新文档添加到模块"elliprc"中，定义函数"elliprc"
add_newdoc(
    "elliprc",
    r"""
    elliprc(x, y, out=None)

    Degenerate symmetric elliptic integral.

    The function RC is defined as [1]_

    .. math::

        R_{\mathrm{C}}(x, y) =
           \frac{1}{2} \int_0^{+\infty} (t + x)^{-1/2} (t + y)^{-1} dt
           = R_{\mathrm{F}}(x, y, y)

    Parameters
    ----------
    x, y : array_like
        Real or complex input parameters. `x` can be any number in the
        complex plane cut along the negative real axis. `y` must be non-zero.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    R : scalar or ndarray
        Value of the integral. If `y` is real and negative, the Cauchy
        principal value is returned. If both of `x` and `y` are real, the
        return value is real. Otherwise, the return value is complex.

    See Also
    --------
    elliprf : Completely-symmetric elliptic integral of the first kind.
    elliprd : Symmetric elliptic integral of the second kind.
    elliprg : Completely-symmetric elliptic integral of the second kind.
    elliprj : Symmetric elliptic integral of the third kind.

    Notes
    -----
    RC is a degenerate case of the symmetric integral RF: ``elliprc(x, y) ==
    elliprf(x, y, y)``. It is an elementary function rather than an elliptic
    integral.

    The code implements Carlson's algorithm based on the duplication theorems
    and series expansion up to the 7th order. [2]_

    .. versionadded:: 1.8.0

    References
    ----------
    .. [1] B. C. Carlson, ed., Chapter 19 in "Digital Library of Mathematical
           Functions," NIST, US Dept. of Commerce.
           https://dlmf.nist.gov/19.16.E6
    .. [2] B. C. Carlson, "Numerical computation of real or complex elliptic
           integrals," Numer. Algorithm, vol. 10, no. 1, pp. 13-26, 1995.
           https://arxiv.org/abs/math/9409227
           https://doi.org/10.1007/BF02198293

    Examples
    --------
    Basic homogeneity property:

    >>> import numpy as np
    >>> from scipy.special import elliprc

    >>> x = 1.2 + 3.4j
    >>> y = 5.
    >>> scale = 0.3 + 0.4j
    >>> elliprc(scale*x, scale*y)
    (0.5484493976710874-0.4169557678995833j)

    >>> elliprc(x, y)/np.sqrt(scale)
    (0.5484493976710874-0.41695576789958333j)

    When the two arguments coincide, the integral is particularly
    simple:

    >>> x = 1.2 + 3.4j
    >>> elliprc(x, x)
    (0.4299173120614631-0.3041729818745595j)

    >>> 1/np.sqrt(x)
    (0.4299173120614631-0.30417298187455954j)

    Another simple case: the first argument vanishes:

    >>> y = 1.2 + 3.4j
    >>> elliprc(0, y)
    (0.6753125346116815-0.47779380263880866j)

    >>> np.pi/2/np.sqrt(y)
    (0.6753125346116815-0.4777938026388088j)

    When `x` and `y` are both positive, we can express
    :math:`R_C(x,y)` in terms of more elementary functions.  For the
    case :math:`0 \le x < y`,

    >>> x = 3.2
    >>> y = 6.
    >>> elliprc(x, y)
    0.44942991498453444
    """
    # 计算表达式 np.arctan(np.sqrt((y-x)/x))/np.sqrt(y-x) 的结果
    0.44942991498453433
    
    # 当条件 0 ≤ y < x 满足时，计算 elliprc(x, y) 的结果
    x = 6.
    y = 3.2
    elliprc(x, y)
    0.4989837501576147
    
    # 计算表达式 np.log((np.sqrt(x)+np.sqrt(x-y))/np.sqrt(y))/np.sqrt(x-y) 的结果
    0.49898375015761476
# 为 "elliprd" 函数添加文档字符串
add_newdoc(
    "elliprd",
    r"""
    elliprd(x, y, z, out=None)

    Symmetric elliptic integral of the second kind.

    The function RD is defined as [1]_

    .. math::

        R_{\mathrm{D}}(x, y, z) =
           \frac{3}{2} \int_0^{+\infty} [(t + x) (t + y)]^{-1/2} (t + z)^{-3/2}
           dt

    Parameters
    ----------
    x, y, z : array_like
        Real or complex input parameters. `x` or `y` can be any number in the
        complex plane cut along the negative real axis, but at most one of them
        can be zero, while `z` must be non-zero.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    R : scalar or ndarray
        Value of the integral. If all of `x`, `y`, and `z` are real, the
        return value is real. Otherwise, the return value is complex.

    See Also
    --------
    elliprc : Degenerate symmetric elliptic integral.
    elliprf : Completely-symmetric elliptic integral of the first kind.
    elliprg : Completely-symmetric elliptic integral of the second kind.
    elliprj : Symmetric elliptic integral of the third kind.

    Notes
    -----
    RD is a degenerate case of the elliptic integral RJ: ``elliprd(x, y, z) ==
    elliprj(x, y, z, z)``.

    The code implements Carlson's algorithm based on the duplication theorems
    and series expansion up to the 7th order. [2]_

    .. versionadded:: 1.8.0

    References
    ----------
    .. [1] B. C. Carlson, ed., Chapter 19 in "Digital Library of Mathematical
           Functions," NIST, US Dept. of Commerce.
           https://dlmf.nist.gov/19.16.E5
    .. [2] B. C. Carlson, "Numerical computation of real or complex elliptic
           integrals," Numer. Algorithm, vol. 10, no. 1, pp. 13-26, 1995.
           https://arxiv.org/abs/math/9409227
           https://doi.org/10.1007/BF02198293

    Examples
    --------
    Basic homogeneity property:

    >>> import numpy as np
    >>> from scipy.special import elliprd

    >>> x = 1.2 + 3.4j
    >>> y = 5.
    >>> z = 6.
    >>> scale = 0.3 + 0.4j
    >>> elliprd(scale*x, scale*y, scale*z)
    (-0.03703043835680379-0.24500934665683802j)

    >>> elliprd(x, y, z)*np.power(scale, -1.5)
    (-0.0370304383568038-0.24500934665683805j)

    All three arguments coincide:

    >>> x = 1.2 + 3.4j
    >>> elliprd(x, x, x)
    (-0.03986825876151896-0.14051741840449586j)

    >>> np.power(x, -1.5)
    (-0.03986825876151894-0.14051741840449583j)

    The so-called "second lemniscate constant":

    >>> elliprd(0, 2, 1)/3
    0.5990701173677961

    >>> from scipy.special import gamma
    >>> gamma(0.75)**2/np.sqrt(2*np.pi)
    0.5990701173677959

    """)

# 为 "elliprf" 函数添加文档字符串
add_newdoc(
    "elliprf",
    r"""
    elliprf(x, y, z, out=None)

    Completely-symmetric elliptic integral of the first kind.

    The function RF is defined as [1]_

    .. math::

        R_{\mathrm{F}}(x, y, z) =
           \frac{1}{2} \int_0^{+\infty} [(t + x) (t + y) (t + z)]^{-1/2} dt

    Parameters
    ----------
    x, y, z : array_like
        Real or complex input parameters. `x`, `y`, and `z` can be any number in the
        complex plane cut along the negative real axis.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    R : scalar or ndarray
        Value of the integral. If all of `x`, `y`, and `z` are real, the
        return value is real. Otherwise, the return value is complex.

    See Also
    --------
    elliprd : Symmetric elliptic integral of the second kind.
    elliprc : Degenerate symmetric elliptic integral.
    elliprg : Completely-symmetric elliptic integral of the second kind.
    elliprj : Symmetric elliptic integral of the third kind.

    Notes
    -----
    RF is related to the other elliptic integrals and is used in various
    mathematical and physical applications.

    .. versionadded:: 1.8.0

    References
    ----------
    .. [1] B. C. Carlson, ed., Chapter 19 in "Digital Library of Mathematical
           Functions," NIST, US Dept. of Commerce.
           https://dlmf.nist.gov/19.16.E5

    Examples
    --------
    Basic example:

    >>> from scipy.special import elliprf

    >>> x = 1.2
    >>> y = 2.3
    >>> z = 3.4
    >>> elliprf(x, y, z)
    0.38601309168605124

    """)
    ----------
    x, y, z : array_like
        Real or complex input parameters. `x`, `y`, or `z` can be any number in
        the complex plane cut along the negative real axis, but at most one of
        them can be zero.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    R : scalar or ndarray
        Value of the integral. If all of `x`, `y`, and `z` are real, the return
        value is real. Otherwise, the return value is complex.

    See Also
    --------
    elliprc : Degenerate symmetric integral.
    elliprd : Symmetric elliptic integral of the second kind.
    elliprg : Completely-symmetric elliptic integral of the second kind.
    elliprj : Symmetric elliptic integral of the third kind.

    Notes
    -----
    The code implements Carlson's algorithm based on the duplication theorems
    and series expansion up to the 7th order (cf.:
    https://dlmf.nist.gov/19.36.i) and the AGM algorithm for the complete
    integral. [2]_

    .. versionadded:: 1.8.0

    References
    ----------
    .. [1] B. C. Carlson, ed., Chapter 19 in "Digital Library of Mathematical
           Functions," NIST, US Dept. of Commerce.
           https://dlmf.nist.gov/19.16.E1
    .. [2] B. C. Carlson, "Numerical computation of real or complex elliptic
           integrals," Numer. Algorithm, vol. 10, no. 1, pp. 13-26, 1995.
           https://arxiv.org/abs/math/9409227
           https://doi.org/10.1007/BF02198293

    Examples
    --------
    Basic homogeneity property:

    >>> import numpy as np
    >>> from scipy.special import elliprf

    >>> x = 1.2 + 3.4j
    >>> y = 5.
    >>> z = 6.
    >>> scale = 0.3 + 0.4j
    >>> elliprf(scale*x, scale*y, scale*z)
    (0.5328051227278146-0.4008623567957094j)

    >>> elliprf(x, y, z)/np.sqrt(scale)
    (0.5328051227278147-0.4008623567957095j)

    All three arguments coincide:

    >>> x = 1.2 + 3.4j
    >>> elliprf(x, x, x)
    (0.42991731206146316-0.30417298187455954j)

    >>> 1/np.sqrt(x)
    (0.4299173120614631-0.30417298187455954j)

    The so-called "first lemniscate constant":

    >>> elliprf(0, 1, 2)
    1.3110287771460598

    >>> from scipy.special import gamma
    >>> gamma(0.25)**2/(4*np.sqrt(2*np.pi))
    1.3110287771460598

    """
# 在指定的模块中添加新的文档字符串，描述了 `elliprg` 函数的用途和数学定义
add_newdoc(
    "elliprg",
    r"""
    elliprg(x, y, z, out=None)

    Completely-symmetric elliptic integral of the second kind.

    The function RG is defined as [1]_

    .. math::

        R_{\mathrm{G}}(x, y, z) =
           \frac{1}{4} \int_0^{+\infty} [(t + x) (t + y) (t + z)]^{-1/2}
           \left(\frac{x}{t + x} + \frac{y}{t + y} + \frac{z}{t + z}\right) t
           dt

    Parameters
    ----------
    x, y, z : array_like
        Real or complex input parameters. `x`, `y`, or `z` can be any number in
        the complex plane cut along the negative real axis.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    R : scalar or ndarray
        Value of the integral. If all of `x`, `y`, and `z` are real, the return
        value is real. Otherwise, the return value is complex.

    See Also
    --------
    elliprc : Degenerate symmetric integral.
    elliprd : Symmetric elliptic integral of the second kind.
    elliprf : Completely-symmetric elliptic integral of the first kind.
    elliprj : Symmetric elliptic integral of the third kind.

    Notes
    -----
    The implementation uses the relation [1]_

    .. math::

        2 R_{\mathrm{G}}(x, y, z) =
           z R_{\mathrm{F}}(x, y, z) -
           \frac{1}{3} (x - z) (y - z) R_{\mathrm{D}}(x, y, z) +
           \sqrt{\frac{x y}{z}}

    and the symmetry of `x`, `y`, `z` when at least one non-zero parameter can
    be chosen as the pivot. When one of the arguments is close to zero, the AGM
    method is applied instead. Other special cases are computed following Ref.
    [2]_

    .. versionadded:: 1.8.0

    References
    ----------
    .. [1] B. C. Carlson, "Numerical computation of real or complex elliptic
           integrals," Numer. Algorithm, vol. 10, no. 1, pp. 13-26, 1995.
           https://arxiv.org/abs/math/9409227
           https://doi.org/10.1007/BF02198293
    .. [2] B. C. Carlson, ed., Chapter 19 in "Digital Library of Mathematical
           Functions," NIST, US Dept. of Commerce.
           https://dlmf.nist.gov/19.16.E1
           https://dlmf.nist.gov/19.20.ii

    Examples
    --------
    Basic homogeneity property:

    >>> import numpy as np
    >>> from scipy.special import elliprg

    >>> x = 1.2 + 3.4j
    >>> y = 5.
    >>> z = 6.
    >>> scale = 0.3 + 0.4j
    >>> elliprg(scale*x, scale*y, scale*z)
    (1.195936862005246+0.8470988320464167j)

    >>> elliprg(x, y, z)*np.sqrt(scale)
    (1.195936862005246+0.8470988320464165j)

    Simplifications:

    >>> elliprg(0, y, y)
    1.756203682760182

    >>> 0.25*np.pi*np.sqrt(y)
    1.7562036827601817

    >>> elliprg(0, 0, z)
    1.224744871391589

    >>> 0.5*np.sqrt(z)
    1.224744871391589

    The surface area of a triaxial ellipsoid with semiaxes ``a``, ``b``, and
    ``c`` is given by

    .. math::

        S = 4 \pi a b c R_{\mathrm{G}}(1 / a^2, 1 / b^2, 1 / c^2).

    >>> def ellipsoid_area(a, b, c):
    # 计算椭球表面积的函数
    r = 4.0 * np.pi * a * b * c
    # 返回椭球表面积，调用 elliprg 函数计算椭球体积
    return r * elliprg(1.0 / (a * a), 1.0 / (b * b), 1.0 / (c * c))
>>> print(ellipsoid_area(1, 3, 5))
# 打印计算出的椭球表面积值
108.62688289491807
# 在指定模块中添加新文档字符串
add_newdoc(
    "elliprj",
    r"""
    elliprj(x, y, z, p, out=None)

    Symmetric elliptic integral of the third kind.

    The function RJ is defined as [1]_

    .. math::

        R_{\mathrm{J}}(x, y, z, p) =
           \frac{3}{2} \int_0^{+\infty} [(t + x) (t + y) (t + z)]^{-1/2}
           (t + p)^{-1} dt

    .. warning::
        This function should be considered experimental when the inputs are
        unbalanced.  Check correctness with another independent implementation.

    Parameters
    ----------
    x, y, z, p : array_like
        Real or complex input parameters. `x`, `y`, or `z` are numbers in
        the complex plane cut along the negative real axis (subject to further
        constraints, see Notes), and at most one of them can be zero. `p` must
        be non-zero.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    R : scalar or ndarray
        Value of the integral. If all of `x`, `y`, `z`, and `p` are real, the
        return value is real. Otherwise, the return value is complex.

        If `p` is real and negative, while `x`, `y`, and `z` are real,
        non-negative, and at most one of them is zero, the Cauchy principal
        value is returned. [1]_ [2]_

    See Also
    --------
    elliprc : Degenerate symmetric integral.
    elliprd : Symmetric elliptic integral of the second kind.
    elliprf : Completely-symmetric elliptic integral of the first kind.
    elliprg : Completely-symmetric elliptic integral of the second kind.

    Notes
    -----
    实现了基于 Carlson 算法的代码，利用了重复定理和到第 7 阶的级数展开。[3]_
    算法与 [1]_ 中的早期版本略有不同，在内部循环中不再需要调用 `elliprc`（或 `atan`/`atanh`，参见 [4]_）。
    当参数的数量级差异很大时，使用渐近近似。[5]_

    对于复数输入，输入值需要满足一定的充分条件，但不是必要条件。特别地，`x`、`y` 和 `z` 的实部必须是非负的，除非其中两个是非负的共轭复数，而另一个是实数非负数。[1]_
    如果输入不满足参考文献 [1]_ 中描述的充分条件，则会被拒绝，并将输出设置为 NaN。

    当 `x`、`y` 或 `z` 中的一个等于 `p` 时，应优先选择函数 `elliprd`，因为它的定义域更宽松。

    .. versionadded:: 1.8.0

    References
    ----------
    .. [1] B. C. Carlson, "Numerical computation of real or complex elliptic
           integrals," Numer. Algorithm, vol. 10, no. 1, pp. 13-26, 1995.
           https://arxiv.org/abs/math/9409227
           https://doi.org/10.1007/BF02198293
    """
)
    .. [2] B. C. Carlson, ed., Chapter 19 in "Digital Library of Mathematical
           Functions," NIST, US Dept. of Commerce.
           https://dlmf.nist.gov/19.20.iii
    .. [3] B. C. Carlson, J. FitzSimmons, "Reduction Theorems for Elliptic
           Integrands with the Square Root of Two Quadratic Factors," J.
           Comput. Appl. Math., vol. 118, nos. 1-2, pp. 71-85, 2000.
           https://doi.org/10.1016/S0377-0427(00)00282-X
    .. [4] F. Johansson, "Numerical Evaluation of Elliptic Functions, Elliptic
           Integrals and Modular Forms," in J. Blumlein, C. Schneider, P.
           Paule, eds., "Elliptic Integrals, Elliptic Functions and Modular
           Forms in Quantum Field Theory," pp. 269-293, 2019 (Cham,
           Switzerland: Springer Nature Switzerland)
           https://arxiv.org/abs/1806.06725
           https://doi.org/10.1007/978-3-030-04480-0
    .. [5] B. C. Carlson, J. L. Gustafson, "Asymptotic Approximations for
           Symmetric Elliptic Integrals," SIAM J. Math. Anls., vol. 25, no. 2,
           pp. 288-303, 1994.
           https://arxiv.org/abs/math/9310223
           https://doi.org/10.1137/S0036141092228477

    Examples
    --------
    Basic homogeneity property:

    >>> import numpy as np
    >>> from scipy.special import elliprj

    >>> x = 1.2 + 3.4j
    >>> y = 5.
    >>> z = 6.
    >>> p = 7.
    >>> scale = 0.3 - 0.4j
    >>> elliprj(scale*x, scale*y, scale*z, scale*p)
    (0.10834905565679157+0.19694950747103812j)

    >>> elliprj(x, y, z, p)*np.power(scale, -1.5)
    (0.10834905565679556+0.19694950747103854j)

    Reduction to simpler elliptic integral:

    >>> elliprj(x, y, z, z)
    (0.08288462362195129-0.028376809745123258j)

    >>> from scipy.special import elliprd
    >>> elliprd(x, y, z)
    (0.08288462362195136-0.028376809745123296j)

    All arguments coincide:

    >>> elliprj(x, x, x, x)
    (-0.03986825876151896-0.14051741840449586j)

    >>> np.power(x, -1.5)
    (-0.03986825876151894-0.14051741840449583j)
# 给指定的函数名添加新的文档字符串和说明
add_newdoc("entr",
    r"""
    entr(x, out=None)

    Elementwise function for computing entropy.

    .. math:: \text{entr}(x) = \begin{cases} - x \log(x) & x > 0  \\ 0 & x = 0
              \\ -\infty & \text{otherwise} \end{cases}

    Parameters
    ----------
    x : ndarray
        Input array.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    res : scalar or ndarray
        The value of the elementwise entropy function at the given points `x`.

    See Also
    --------
    kl_div, rel_entr, scipy.stats.entropy

    Notes
    -----
    .. versionadded:: 0.15.0

    This function is concave.

    The origin of this function is in convex programming; see [1]_.
    Given a probability distribution :math:`p_1, \ldots, p_n`,
    the definition of entropy in the context of *information theory* is

    .. math::

        \sum_{i = 1}^n \mathrm{entr}(p_i).

    To compute the latter quantity, use `scipy.stats.entropy`.

    References
    ----------
    .. [1] Boyd, Stephen and Lieven Vandenberghe. *Convex optimization*.
           Cambridge University Press, 2004.
           :doi:`https://doi.org/10.1017/CBO9780511804441`

    """)

# 给指定的函数名添加新的文档字符串和说明
add_newdoc("erf",
    """
    erf(z, out=None)

    Returns the error function of complex argument.

    It is defined as ``2/sqrt(pi)*integral(exp(-t**2), t=0..z)``.

    Parameters
    ----------
    x : ndarray
        Input array.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    res : scalar or ndarray
        The values of the error function at the given points `x`.

    See Also
    --------
    erfc, erfinv, erfcinv, wofz, erfcx, erfi

    Notes
    -----
    The cumulative of the unit normal distribution is given by
    ``Phi(z) = 1/2[1 + erf(z/sqrt(2))]``.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Error_function
    .. [2] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover,
        1972. http://www.math.sfu.ca/~cbm/aands/page_297.htm
    .. [3] Steven G. Johnson, Faddeeva W function implementation.
       http://ab-initio.mit.edu/Faddeeva

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import special
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-3, 3)
    >>> plt.plot(x, special.erf(x))
    >>> plt.xlabel('$x$')
    >>> plt.ylabel('$erf(x)$')
    >>> plt.show()

    """)

# 给指定的函数名添加新的文档字符串和说明
add_newdoc("erfc",
    """
    erfc(x, out=None)

    Complementary error function, ``1 - erf(x)``.

    Parameters
    ----------
    x : array_like
        Real or complex valued argument
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of the complementary error function

    See Also
    --------
    erf, erfi, erfcx, dawsn, wofz


    """)
    References
    ----------
    .. [1] Steven G. Johnson, Faddeeva W function implementation.
       http://ab-initio.mit.edu/Faddeeva
    
    Examples
    --------
    >>> import numpy as np
    >>> from scipy import special
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-3, 3)  # 创建一个从-3到3的等间隔数组，用作 x 轴数据
    >>> plt.plot(x, special.erfc(x))  # 绘制 x 对应的正余弦积分函数的曲线
    >>> plt.xlabel('$x$')  # 设置 x 轴标签为数学模式的 x
    >>> plt.ylabel('$erfc(x)$')  # 设置 y 轴标签为数学模式的 erfc(x)
    >>> plt.show()  # 显示绘制的图形
# 添加新的文档字符串到 "erfi" 函数中，用于描述函数的功能、参数和返回值
add_newdoc("erfi",
    """
    erfi(z, out=None)

    Imaginary error function, ``-i erf(i z)``.

    Parameters
    ----------
    z : array_like
        Real or complex valued argument
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of the imaginary error function

    See Also
    --------
    erf, erfc, erfcx, dawsn, wofz

    Notes
    -----

    .. versionadded:: 0.12.0

    References
    ----------
    .. [1] Steven G. Johnson, Faddeeva W function implementation.
       http://ab-initio.mit.edu/Faddeeva

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import special
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-3, 3)
    >>> plt.plot(x, special.erfi(x))
    >>> plt.xlabel('$x$')
    >>> plt.ylabel('$erfi(x)$')
    >>> plt.show()

    """)

# 添加新的文档字符串到 "erfcx" 函数中，用于描述函数的功能、参数和返回值
add_newdoc("erfcx",
    """
    erfcx(x, out=None)

    Scaled complementary error function, ``exp(x**2) * erfc(x)``.

    Parameters
    ----------
    x : array_like
        Real or complex valued argument
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of the scaled complementary error function


    See Also
    --------
    erf, erfc, erfi, dawsn, wofz

    Notes
    -----

    .. versionadded:: 0.12.0

    References
    ----------
    .. [1] Steven G. Johnson, Faddeeva W function implementation.
       http://ab-initio.mit.edu/Faddeeva

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import special
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-3, 3)
    >>> plt.plot(x, special.erfcx(x))
    >>> plt.xlabel('$x$')
    >>> plt.ylabel('$erfcx(x)$')
    >>> plt.show()

    """)

# 添加新的文档字符串到 "erfinv" 函数中，用于描述函数的功能、参数和返回值
add_newdoc(
    "erfinv",
    """
    erfinv(y, out=None)

    Inverse of the error function.

    Computes the inverse of the error function.

    In the complex domain, there is no unique complex number w satisfying
    erf(w)=z. This indicates a true inverse function would be multivalued.
    When the domain restricts to the real, -1 < x < 1, there is a unique real
    number satisfying erf(erfinv(x)) = x.

    Parameters
    ----------
    y : ndarray
        Argument at which to evaluate. Domain: [-1, 1]
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    erfinv : scalar or ndarray
        The inverse of erf of y, element-wise

    See Also
    --------
    erf : Error function of a complex argument
    erfc : Complementary error function, ``1 - erf(x)``
    erfcinv : Inverse of the complementary error function

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.special import erfinv, erf

    >>> erfinv(0.5)
    0.4769362762044699

    >>> y = np.linspace(-1.0, 1.0, num=9)
    >>> x = erfinv(y)
    >>> x

    """)
    # 导入 numpy 库，并使用其别名 np
    >>> import numpy as np

    # 导入 matplotlib.pyplot 库，并使用其别名 plt
    >>> import matplotlib.pyplot as plt

    # 定义一个数组，包含 -inf（负无穷）、-0.81341985、-0.47693628 等值
    >>> array([       -inf, -0.81341985, -0.47693628, -0.22531206,  0.        ,
            0.22531206,  0.47693628,  0.81341985,         inf])

    # 验证 erf(erfinv(y)) 等于 y
    >>> Verify that ``erf(erfinv(y))`` is ``y``.

    # 调用 erf 函数计算给定数组的值
    >>> erf(x)
    # 返回包含 -1 到 1 的数组
    array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ])

    # 绘制函数 erfinv(y) 的图形
    >>> Plot the function:

    # 在 -1 到 1 的范围内生成包含 200 个点的等间隔数组 y
    >>> y = np.linspace(-1, 1, 200)

    # 创建一个图形窗口和轴对象
    >>> fig, ax = plt.subplots()

    # 在轴上绘制 y 对应的 erfinv(y) 函数曲线
    >>> ax.plot(y, erfinv(y))

    # 启用网格显示
    >>> ax.grid(True)

    # 设置 x 轴的标签为 'y'
    >>> ax.set_xlabel('y')

    # 设置图形的标题为 'erfinv(y)'
    >>> ax.set_title('erfinv(y)')

    # 显示绘制的图形
    >>> plt.show()
add_newdoc(
    "erfcinv",
    """
    erfcinv(y, out=None)

    Inverse of the complementary error function.

    Computes the inverse of the complementary error function.

    In the complex domain, there is no unique complex number w satisfying
    erfc(w)=z. This indicates a true inverse function would be multivalued.
    When the domain restricts to the real, 0 < x < 2, there is a unique real
    number satisfying erfc(erfcinv(x)) = erfcinv(erfc(x)).

    It is related to inverse of the error function by erfcinv(1-x) = erfinv(x)

    Parameters
    ----------
    y : ndarray
        Argument at which to evaluate. Domain: [0, 2]
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    erfcinv : scalar or ndarray
        The inverse of erfc of y, element-wise

    See Also
    --------
    erf : Error function of a complex argument
    erfc : Complementary error function, ``1 - erf(x)``
    erfinv : Inverse of the error function

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.special import erfcinv

    >>> erfcinv(0.5)
    0.4769362762044699

    >>> y = np.linspace(0.0, 2.0, num=11)
    >>> erfcinv(y)
    array([        inf,  0.9061938 ,  0.59511608,  0.37080716,  0.17914345,
           -0.        , -0.17914345, -0.37080716, -0.59511608, -0.9061938 ,
                  -inf])

    Plot the function:

    >>> y = np.linspace(0, 2, 200)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(y, erfcinv(y))
    >>> ax.grid(True)
    >>> ax.set_xlabel('y')
    >>> ax.set_title('erfcinv(y)')
    >>> plt.show()

    """
)

add_newdoc("eval_jacobi",
    r"""
    eval_jacobi(n, alpha, beta, x, out=None)

    Evaluate Jacobi polynomial at a point.

    The Jacobi polynomials can be defined via the Gauss hypergeometric
    function :math:`{}_2F_1` as

    .. math::

        P_n^{(\alpha, \beta)}(x) = \frac{(\alpha + 1)_n}{\Gamma(n + 1)}
          {}_2F_1(-n, 1 + \alpha + \beta + n; \alpha + 1; (1 - z)/2)

    where :math:`(\cdot)_n` is the Pochhammer symbol; see `poch`. When
    :math:`n` is an integer the result is a polynomial of degree
    :math:`n`. See 22.5.42 in [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer the result is
        determined via the relation to the Gauss hypergeometric
        function.
    alpha : array_like
        Parameter
    beta : array_like
        Parameter
    x : array_like
        Points at which to evaluate the polynomial
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    P : scalar or ndarray
        Values of the Jacobi polynomial

    See Also
    --------
    roots_jacobi : roots and quadrature weights of Jacobi polynomials
    jacobi : Jacobi polynomial object
    hyp2f1 : Gauss hypergeometric function

    References
    ----------
    Abramowitz, M. and Stegun, I.A. (Eds.). "Handbook of Mathematical Functions
    with Formulas, Graphs, and Mathematical Tables", Dover, 1964.

    """
)
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.



# 这是一个多行注释的示例，通常用于文档字符串或长注释内容
# 在这里，作者引用了一本数学函数手册的参考文献
# 文档字符串通常用于函数或类的开头，用来描述其功能、参数和返回值等信息
# 这里的文档字符串未被分配给任何函数或类，可能是一个错误或遗漏
# 添加新文档给 eval_sh_jacobi 函数
add_newdoc("eval_sh_jacobi",
    r"""
    eval_sh_jacobi(n, p, q, x, out=None)

    Evaluate shifted Jacobi polynomial at a point.

    Defined by

    .. math::

        G_n^{(p, q)}(x)
          = \binom{2n + p - 1}{n}^{-1} P_n^{(p - q, q - 1)}(2x - 1),

    where :math:`P_n^{(\cdot, \cdot)}` is the n-th Jacobi
    polynomial. See 22.5.2 in [AS]_ for details.

    Parameters
    ----------
    n : int
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to `binom` and `eval_jacobi`.
    p : float
        Parameter
    q : float
        Parameter
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    G : scalar or ndarray
        Values of the shifted Jacobi polynomial.

    See Also
    --------
    roots_sh_jacobi : roots and quadrature weights of shifted Jacobi
                      polynomials
    sh_jacobi : shifted Jacobi polynomial object
    eval_jacobi : evaluate Jacobi polynomials

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

# 添加新文档给 eval_gegenbauer 函数
add_newdoc("eval_gegenbauer",
    r"""
    eval_gegenbauer(n, alpha, x, out=None)

    Evaluate Gegenbauer polynomial at a point.

    The Gegenbauer polynomials can be defined via the Gauss
    hypergeometric function :math:`{}_2F_1` as

    .. math::

        C_n^{(\alpha)} = \frac{(2\alpha)_n}{\Gamma(n + 1)}
          {}_2F_1(-n, 2\alpha + n; \alpha + 1/2; (1 - z)/2).

    When :math:`n` is an integer the result is a polynomial of degree
    :math:`n`. See 22.5.46 in [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to the Gauss hypergeometric
        function.
    alpha : array_like
        Parameter
    x : array_like
        Points at which to evaluate the Gegenbauer polynomial
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    C : scalar or ndarray
        Values of the Gegenbauer polynomial

    See Also
    --------
    roots_gegenbauer : roots and quadrature weights of Gegenbauer
                       polynomials
    gegenbauer : Gegenbauer polynomial object
    hyp2f1 : Gauss hypergeometric function

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

# 添加新文档给 eval_chebyt 函数
add_newdoc("eval_chebyt",
    r"""
    eval_chebyt(n, x, out=None)

    Evaluate Chebyshev polynomial of the first kind at a point.

    The Chebyshev polynomials of the first kind can be defined via the
    Gauss hypergeometric function :math:`{}_2F_1` as

    .. math::

        T_n(x) = {}_2F_1(n, -n; 1/2; (1 - x)/2).

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    x : array_like
        Points at which to evaluate the Chebyshev polynomial
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    T : scalar or ndarray
        Values of the Chebyshev polynomial of the first kind

    """)
    When :math:`n` is an integer the result is a polynomial of degree
    :math:`n`. See 22.5.47 in [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to the Gauss hypergeometric
        function.
    x : array_like
        Points at which to evaluate the Chebyshev polynomial
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    T : scalar or ndarray
        Values of the Chebyshev polynomial

    See Also
    --------
    roots_chebyt : roots and quadrature weights of Chebyshev
                   polynomials of the first kind
    chebyu : Chebychev polynomial object
    eval_chebyu : evaluate Chebyshev polynomials of the second kind
    hyp2f1 : Gauss hypergeometric function
    numpy.polynomial.chebyshev.Chebyshev : Chebyshev series

    Notes
    -----
    This routine is numerically stable for `x` in ``[-1, 1]`` at least
    up to order ``10000``.

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.
# 添加一个新的文档字符串到 eval_chebyu 函数
add_newdoc("eval_chebyu",
    r"""
    eval_chebyu(n, x, out=None)

    Evaluate Chebyshev polynomial of the second kind at a point.

    The Chebyshev polynomials of the second kind can be defined via
    the Gauss hypergeometric function :math:`{}_2F_1` as

    .. math::

        U_n(x) = (n + 1) {}_2F_1(-n, n + 2; 3/2; (1 - x)/2).

    When :math:`n` is an integer the result is a polynomial of degree
    :math:`n`. See 22.5.48 in [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to the Gauss hypergeometric
        function.
    x : array_like
        Points at which to evaluate the Chebyshev polynomial
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    U : scalar or ndarray
        Values of the Chebyshev polynomial

    See Also
    --------
    roots_chebyu : roots and quadrature weights of Chebyshev
                   polynomials of the second kind
    chebyu : Chebyshev polynomial object
    eval_chebyt : evaluate Chebyshev polynomials of the first kind
    hyp2f1 : Gauss hypergeometric function

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

# 添加一个新的文档字符串到 eval_chebys 函数
add_newdoc("eval_chebys",
    r"""
    eval_chebys(n, x, out=None)

    Evaluate Chebyshev polynomial of the second kind on [-2, 2] at a
    point.

    These polynomials are defined as

    .. math::

        S_n(x) = U_n(x/2)

    where :math:`U_n` is a Chebyshev polynomial of the second
    kind. See 22.5.13 in [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to `eval_chebyu`.
    x : array_like
        Points at which to evaluate the Chebyshev polynomial
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    S : scalar or ndarray
        Values of the Chebyshev polynomial

    See Also
    --------
    roots_chebys : roots and quadrature weights of Chebyshev
                   polynomials of the second kind on [-2, 2]
    chebys : Chebyshev polynomial object
    eval_chebyu : evaluate Chebyshev polynomials of the second kind

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    They are a scaled version of the Chebyshev polynomials of the
    second kind.

    >>> x = np.linspace(-2, 2, 6)
    >>> sc.eval_chebys(3, x)
    array([-4.   ,  0.672,  0.736, -0.736, -0.672,  4.   ])
    >>> sc.eval_chebyu(3, x / 2)
    """
    array([-4.   ,  0.672,  0.736, -0.736, -0.672,  4.   ])



    # 创建一个 NumPy 数组，包含指定的浮点数值
    """


注释的重点是解释了这行代码创建了一个包含特定浮点数值的 NumPy 数组。
add_newdoc("eval_chebyc",
    r"""
    eval_chebyc(n, x, out=None)

    Evaluate Chebyshev polynomial of the first kind on [-2, 2] at a
    point.

    These polynomials are defined as

    .. math::

        C_n(x) = 2 T_n(x/2)

    where :math:`T_n` is a Chebyshev polynomial of the first kind. See
    22.5.11 in [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to `eval_chebyt`.
    x : array_like
        Points at which to evaluate the Chebyshev polynomial
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    C : scalar or ndarray
        Values of the Chebyshev polynomial

    See Also
    --------
    roots_chebyc : roots and quadrature weights of Chebyshev
                   polynomials of the first kind on [-2, 2]
    chebyc : Chebyshev polynomial object
    numpy.polynomial.chebyshev.Chebyshev : Chebyshev series
    eval_chebyt : evaluate Chebyshev polynomials of the first kind

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    They are a scaled version of the Chebyshev polynomials of the
    first kind.

    >>> x = np.linspace(-2, 2, 6)
    >>> sc.eval_chebyc(3, x)
    array([-2.   ,  1.872,  1.136, -1.136, -1.872,  2.   ])
    >>> 2 * sc.eval_chebyt(3, x / 2)
    array([-2.   ,  1.872,  1.136, -1.136, -1.872,  2.   ])

    """)

# 添加新的文档字符串到 eval_sh_chebyt 函数
add_newdoc("eval_sh_chebyt",
    r"""
    eval_sh_chebyt(n, x, out=None)

    Evaluate shifted Chebyshev polynomial of the first kind at a
    point.

    These polynomials are defined as

    .. math::

        T_n^*(x) = T_n(2x - 1)

    where :math:`T_n` is a Chebyshev polynomial of the first kind. See
    22.5.14 in [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to `eval_chebyt`.
    x : array_like
        Points at which to evaluate the shifted Chebyshev polynomial
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    T : scalar or ndarray
        Values of the shifted Chebyshev polynomial

    See Also
    --------
    roots_sh_chebyt : roots and quadrature weights of shifted
                      Chebyshev polynomials of the first kind
    sh_chebyt : shifted Chebyshev polynomial object
    eval_chebyt : evaluate Chebyshev polynomials of the first kind
    numpy.polynomial.chebyshev.Chebyshev : Chebyshev series

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.


    # 这是一个文档字符串，用来提供关于当前模块或函数的信息
    # 引用了Milton Abramowitz和Irene A. Stegun编辑的数学函数手册，出版于1972年的Dover版
    # 提供了与当前代码或函数相关的背景信息或引用来源
add_newdoc("eval_sh_chebyu",
    r"""
    eval_sh_chebyu(n, x, out=None)

    Evaluate shifted Chebyshev polynomial of the second kind at a
    point.

    These polynomials are defined as

    .. math::

        U_n^*(x) = U_n(2x - 1)

    where :math:`U_n` is a Chebyshev polynomial of the first kind. See
    22.5.15 in [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to `eval_chebyu`.
    x : array_like
        Points at which to evaluate the shifted Chebyshev polynomial
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    U : scalar or ndarray
        Values of the shifted Chebyshev polynomial

    See Also
    --------
    roots_sh_chebyu : roots and quadrature weights of shifted
                      Chebychev polynomials of the second kind
    sh_chebyu : shifted Chebyshev polynomial object
    eval_chebyu : evaluate Chebyshev polynomials of the second kind

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

add_newdoc("eval_legendre",
    r"""
    eval_legendre(n, x, out=None)

    Evaluate Legendre polynomial at a point.

    The Legendre polynomials can be defined via the Gauss
    hypergeometric function :math:`{}_2F_1` as

    .. math::

        P_n(x) = {}_2F_1(-n, n + 1; 1; (1 - x)/2).

    When :math:`n` is an integer the result is a polynomial of degree
    :math:`n`. See 22.5.49 in [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to the Gauss hypergeometric
        function.
    x : array_like
        Points at which to evaluate the Legendre polynomial
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    P : scalar or ndarray
        Values of the Legendre polynomial

    See Also
    --------
    roots_legendre : roots and quadrature weights of Legendre
                     polynomials
    legendre : Legendre polynomial object
    hyp2f1 : Gauss hypergeometric function
    numpy.polynomial.legendre.Legendre : Legendre series

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import eval_legendre

    Evaluate the zero-order Legendre polynomial at x = 0

    >>> eval_legendre(0, 0)
    1.0

    Evaluate the first-order Legendre polynomial between -1 and 1

    >>> X = np.linspace(-1, 1, 5)  # Domain of Legendre polynomials
    >>> eval_legendre(1, X)

    """)
    # 导入NumPy库，通常用于数值计算
    >>> import numpy as np

    # 在 x = 0 处计算零阶到四阶勒让德多项式的值
    array([-1. , -0.5,  0. ,  0.5,  1. ])

    # 评估在 x = 0 处零阶到四阶勒让德多项式的值
    >>> N = range(0, 5)
    >>> eval_legendre(N, 0)
    array([ 1.   ,  0.   , -0.5  ,  0.   ,  0.375])

    # 绘制零阶到四阶勒让德多项式的图像

    # 创建一个从 -1 到 1 的等间距数组作为 x 值
    >>> X = np.linspace(-1, 1)

    # 导入 matplotlib.pyplot 库，并用 plt 作为别名
    >>> import matplotlib.pyplot as plt

    # 对每个阶数 n 在 X 上计算对应的勒让德多项式的值并绘制图像
    >>> for n in range(0, 5):
    ...     y = eval_legendre(n, X)
    ...     plt.plot(X, y, label=r'$P_{}(x)$'.format(n))

    # 添加图标题
    >>> plt.title("Legendre Polynomials")
    # 添加 x 轴标签
    >>> plt.xlabel("x")
    # 添加 y 轴标签，使用 LaTeX 格式显示勒让德多项式的符号
    >>> plt.ylabel(r'$P_n(x)$')
    # 添加图例并设定位置在右下角
    >>> plt.legend(loc='lower right')
    # 显示绘制的图像
    >>> plt.show()

    """
# 向库中添加新文档条目 "eval_sh_legendre"
add_newdoc("eval_sh_legendre",
    r"""
    eval_sh_legendre(n, x, out=None)

    Evaluate shifted Legendre polynomial at a point.

    These polynomials are defined as

    .. math::

        P_n^*(x) = P_n(2x - 1)

    where :math:`P_n` is a Legendre polynomial. See 2.2.11 in [AS]_
    for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the value is
        determined via the relation to `eval_legendre`.
    x : array_like
        Points at which to evaluate the shifted Legendre polynomial
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    P : scalar or ndarray
        Values of the shifted Legendre polynomial

    See Also
    --------
    roots_sh_legendre : roots and quadrature weights of shifted
                        Legendre polynomials
    sh_legendre : shifted Legendre polynomial object
    eval_legendre : evaluate Legendre polynomials
    numpy.polynomial.legendre.Legendre : Legendre series

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

# 向库中添加新文档条目 "eval_genlaguerre"
add_newdoc("eval_genlaguerre",
    r"""
    eval_genlaguerre(n, alpha, x, out=None)

    Evaluate generalized Laguerre polynomial at a point.

    The generalized Laguerre polynomials can be defined via the
    confluent hypergeometric function :math:`{}_1F_1` as

    .. math::

        L_n^{(\alpha)}(x) = \binom{n + \alpha}{n}
          {}_1F_1(-n, \alpha + 1, x).

    When :math:`n` is an integer the result is a polynomial of degree
    :math:`n`. See 22.5.54 in [AS]_ for details. The Laguerre
    polynomials are the special case where :math:`\alpha = 0`.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to the confluent hypergeometric
        function.
    alpha : array_like
        Parameter; must have ``alpha > -1``
    x : array_like
        Points at which to evaluate the generalized Laguerre
        polynomial
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    L : scalar or ndarray
        Values of the generalized Laguerre polynomial

    See Also
    --------
    roots_genlaguerre : roots and quadrature weights of generalized
                        Laguerre polynomials
    genlaguerre : generalized Laguerre polynomial object
    hyp1f1 : confluent hypergeometric function
    eval_laguerre : evaluate Laguerre polynomials

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

# 向库中添加新文档条目 "eval_laguerre"
add_newdoc("eval_laguerre",
    r"""
    eval_laguerre(n, x, out=None)

    Evaluate Laguerre polynomial at a point.

    Laguerre polynomials are defined as

    .. math::

        L_n(x) = \sum_{k=0}^{n} (-1)^k \binom{n}{k} \frac{x^k}{k!}

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to `eval_genlaguerre`.
    x : array_like
        Points at which to evaluate the Laguerre polynomial
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    L : scalar or ndarray
        Values of the Laguerre polynomial

    See Also
    --------
    roots_laguerre : roots and quadrature weights of Laguerre polynomials
    laguerre : Laguerre polynomial object
    numpy.polynomial.laguerre.Laguerre : Laguerre series

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)
    Evaluate Laguerre polynomial at a point.

    The Laguerre polynomials can be defined via the confluent
    hypergeometric function :math:`{}_1F_1` as

    .. math::

        L_n(x) = {}_1F_1(-n, 1, x).

    See 22.5.16 and 22.5.54 in [AS]_ for details. When :math:`n` is an
    integer the result is a polynomial of degree :math:`n`.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer the result is
        determined via the relation to the confluent hypergeometric
        function.
    x : array_like
        Points at which to evaluate the Laguerre polynomial
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    L : scalar or ndarray
        Values of the Laguerre polynomial

    See Also
    --------
    roots_laguerre : roots and quadrature weights of Laguerre
                     polynomials
    laguerre : Laguerre polynomial object
    numpy.polynomial.laguerre.Laguerre : Laguerre series
    eval_genlaguerre : evaluate generalized Laguerre polynomials

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.
# 添加新的文档字符串到 eval_hermite 函数，用于描述评估物理学 Hermite 多项式的功能和用法
add_newdoc("eval_hermite",
    r"""
    eval_hermite(n, x, out=None)

    Evaluate physicist's Hermite polynomial at a point.

    Defined by

    .. math::

        H_n(x) = (-1)^n e^{x^2} \frac{d^n}{dx^n} e^{-x^2};

    :math:`H_n` is a polynomial of degree :math:`n`. See 22.11.7 in
    [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial
    x : array_like
        Points at which to evaluate the Hermite polynomial
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    H : scalar or ndarray
        Values of the Hermite polynomial

    See Also
    --------
    roots_hermite : roots and quadrature weights of physicist's
                    Hermite polynomials
    hermite : physicist's Hermite polynomial object
    numpy.polynomial.hermite.Hermite : Physicist's Hermite series
    eval_hermitenorm : evaluate Probabilist's Hermite polynomials

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

# 添加新的文档字符串到 eval_hermitenorm 函数，用于描述评估概率学归一化 Hermite 多项式的功能和用法
add_newdoc("eval_hermitenorm",
    r"""
    eval_hermitenorm(n, x, out=None)

    Evaluate probabilist's (normalized) Hermite polynomial at a
    point.

    Defined by

    .. math::

        He_n(x) = (-1)^n e^{x^2/2} \frac{d^n}{dx^n} e^{-x^2/2};

    :math:`He_n` is a polynomial of degree :math:`n`. See 22.11.8 in
    [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial
    x : array_like
        Points at which to evaluate the Hermite polynomial
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    He : scalar or ndarray
        Values of the Hermite polynomial

    See Also
    --------
    roots_hermitenorm : roots and quadrature weights of probabilist's
                        Hermite polynomials
    hermitenorm : probabilist's Hermite polynomial object
    numpy.polynomial.hermite_e.HermiteE : Probabilist's Hermite series
    eval_hermite : evaluate physicist's Hermite polynomials

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)


# 添加新的文档字符串到 exp10 函数，用于描述计算 10 的 x 次幂的功能和用法
add_newdoc("exp10",
    """
    exp10(x, out=None)

    Compute ``10**x`` element-wise.

    Parameters
    ----------
    x : array_like
        `x` must contain real numbers.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        ``10**x``, computed element-wise.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import exp10

    >>> exp10(3)
    1000.0
    >>> x = np.array([[-1, -0.5, 0], [0.5, 1, 1.5]])
    >>> exp10(x)

    """)
    # 创建一个二维数组，包含两个行，每行有三个元素
    array([[  0.1       ,   0.31622777,   1.        ],
           [  3.16227766,  10.        ,  31.6227766 ]])
# 将新文档添加到模块 "exp2" 中，提供了关于函数 exp2 的详细说明
add_newdoc("exp2",
    """
    exp2(x, out=None)

    Compute ``2**x`` element-wise.

    Parameters
    ----------
    x : array_like
        `x` must contain real numbers.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        ``2**x``, computed element-wise.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import exp2

    >>> exp2(3)
    8.0
    >>> x = np.array([[-1, -0.5, 0], [0.5, 1, 1.5]])
    >>> exp2(x)
    array([[ 0.5       ,  0.70710678,  1.        ],
           [ 1.41421356,  2.        ,  2.82842712]])
    """)

# 将新文档添加到模块 "expm1" 中，提供了关于函数 expm1 的详细说明
add_newdoc("expm1",
    """
    expm1(x, out=None)

    Compute ``exp(x) - 1``.

    When `x` is near zero, ``exp(x)`` is near 1, so the numerical calculation
    of ``exp(x) - 1`` can suffer from catastrophic loss of precision.
    ``expm1(x)`` is implemented to avoid the loss of precision that occurs when
    `x` is near zero.

    Parameters
    ----------
    x : array_like
        `x` must contain real numbers.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        ``exp(x) - 1`` computed element-wise.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import expm1

    >>> expm1(1.0)
    1.7182818284590451
    >>> expm1([-0.2, -0.1, 0, 0.1, 0.2])
    array([-0.18126925, -0.09516258,  0.        ,  0.10517092,  0.22140276])

    The exact value of ``exp(7.5e-13) - 1`` is::

        7.5000000000028125000000007031250000001318...*10**-13.

    Here is what ``expm1(7.5e-13)`` gives:

    >>> expm1(7.5e-13)
    7.5000000000028135e-13

    Compare that to ``exp(7.5e-13) - 1``, where the subtraction results in
    a "catastrophic" loss of precision:

    >>> np.exp(7.5e-13) - 1
    7.5006667543675576e-13

    """)

# 将新文档添加到模块 "expn" 中，提供了关于函数 expn 的详细说明
add_newdoc("expn",
    r"""
    expn(n, x, out=None)

    Generalized exponential integral En.

    For integer :math:`n \geq 0` and real :math:`x \geq 0` the
    generalized exponential integral is defined as [dlmf]_

    .. math::

        E_n(x) = x^{n - 1} \int_x^\infty \frac{e^{-t}}{t^n} dt.

    Parameters
    ----------
    n : array_like
        Non-negative integers
    x : array_like
        Real argument
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of the generalized exponential integral

    See Also
    --------
    exp1 : special case of :math:`E_n` for :math:`n = 1`
    expi : related to :math:`E_n` when :math:`n = 1`

    References
    ----------
    .. [dlmf] Digital Library of Mathematical Functions, 8.19.2
              https://dlmf.nist.gov/8.19#E2

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    Its domain is nonnegative n and x.

    >>> sc.expn(-1, 1.0), sc.expn(1, -1.0)
    (nan, nan)
    # 对于 n = 1, 2 的情况，在 x = 0 处有极点；对于更大的 n，该函数值为 `1 / (n - 1)`。
    >>> sc.expn([0, 1, 2, 3, 4], 0)
    # 返回一个 NumPy 数组，包含对应参数的指数积分值
    array([       inf,        inf, 1.        , 0.5       , 0.33333333])

    # 当 n = 0 时，函数化简为 `exp(-x) / x`。
    >>> x = np.array([1, 2, 3, 4])
    >>> sc.expn(0, x)
    # 返回一个 NumPy 数组，包含对应参数和数组 x 的指数积分值
    array([0.36787944, 0.06766764, 0.01659569, 0.00457891])
    >>> np.exp(-x) / x
    # 返回一个 NumPy 数组，包含对应数组 x 的 `exp(-x) / x` 值

    # 当 n = 1 时，函数化简为 `exp1`。
    >>> sc.expn(1, x)
    # 返回一个 NumPy 数组，包含对应参数和数组 x 的指数积分值
    array([0.21938393, 0.04890051, 0.01304838, 0.00377935])
    >>> sc.exp1(x)
    # 返回一个 NumPy 数组，包含对应数组 x 的 `exp1` 值
    array([0.21938393, 0.04890051, 0.01304838, 0.00377935])
# 将文档添加到名为 "fdtr" 的函数的文档字符串中，描述了 F 分布的累积分布函数
add_newdoc("fdtr",
    r"""
    fdtr(dfn, dfd, x, out=None)

    F cumulative distribution function.

    Returns the value of the cumulative distribution function of the
    F-distribution, also known as Snedecor's F-distribution or the
    Fisher-Snedecor distribution.

    The F-distribution with parameters :math:`d_n` and :math:`d_d` is the
    distribution of the random variable,

    .. math::
        X = \frac{U_n/d_n}{U_d/d_d},

    where :math:`U_n` and :math:`U_d` are random variables distributed
    :math:`\chi^2`, with :math:`d_n` and :math:`d_d` degrees of freedom,
    respectively.

    Parameters
    ----------
    dfn : array_like
        First parameter (positive float).
    dfd : array_like
        Second parameter (positive float).
    x : array_like
        Argument (nonnegative float).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    y : scalar or ndarray
        The CDF of the F-distribution with parameters `dfn` and `dfd` at `x`.

    See Also
    --------
    fdtrc : F distribution survival function
    fdtri : F distribution inverse cumulative distribution
    scipy.stats.f : F distribution

    Notes
    -----
    The regularized incomplete beta function is used, according to the
    formula,

    .. math::
        F(d_n, d_d; x) = I_{xd_n/(d_d + xd_n)}(d_n/2, d_d/2).

    Wrapper for the Cephes [1]_ routine `fdtr`. The F distribution is also
    available as `scipy.stats.f`. Calling `fdtr` directly can improve
    performance compared to the ``cdf`` method of `scipy.stats.f` (see last
    example below).

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    Calculate the function for ``dfn=1`` and ``dfd=2`` at ``x=1``.

    >>> import numpy as np
    >>> from scipy.special import fdtr
    >>> fdtr(1, 2, 1)
    0.5773502691896258

    Calculate the function at several points by providing a NumPy array for
    `x`.

    >>> x = np.array([0.5, 2., 3.])
    >>> fdtr(1, 2, x)
    array([0.4472136 , 0.70710678, 0.77459667])

    Plot the function for several parameter sets.

    >>> import matplotlib.pyplot as plt
    >>> dfn_parameters = [1, 5, 10, 50]
    >>> dfd_parameters = [1, 1, 2, 3]
    >>> linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
    >>> parameters_list = list(zip(dfn_parameters, dfd_parameters,
    ...                            linestyles))
    >>> x = np.linspace(0, 30, 1000)
    >>> fig, ax = plt.subplots()
    >>> for parameter_set in parameters_list:
    ...     dfn, dfd, style = parameter_set
    ...     fdtr_vals = fdtr(dfn, dfd, x)
    ...     ax.plot(x, fdtr_vals, label=rf"$d_n={dfn},\, d_d={dfd}$",
    ...             ls=style)
    >>> ax.legend()
    >>> ax.set_xlabel("$x$")
    >>> ax.set_title("F distribution cumulative distribution function")
    >>> plt.show()
    The F distribution is also available as `scipy.stats.f`. Using `fdtr`
    directly can be much faster than calling the ``cdf`` method of
    `scipy.stats.f`, especially for small arrays or individual values.
    To get the same results one must use the following parametrization:
    ``stats.f(dfn, dfd).cdf(x)=fdtr(dfn, dfd, x)``.

    >>> from scipy.stats import f
    >>> dfn, dfd = 1, 2
    >>> x = 1
    >>> fdtr_res = fdtr(dfn, dfd, x)  # this will often be faster than below
    >>> f_dist_res = f(dfn, dfd).cdf(x)
    >>> fdtr_res == f_dist_res  # test that results are equal
    True


    # 导入 scipy 库中的 F 分布模块
    from scipy.stats import f
    # 设置 F 分布的自由度参数 dfn = 1, dfd = 2
    dfn, dfd = 1, 2
    # 设置 x 值为 1
    x = 1
    # 使用 fdtr 函数计算 F 分布的累积分布函数值，这通常比下面的方法更快
    fdtr_res = fdtr(dfn, dfd, x)
    # 使用 scipy.stats.f 对象计算 F 分布的累积分布函数值
    f_dist_res = f(dfn, dfd).cdf(x)
    # 检查两种方法计算的结果是否相等
    fdtr_res == f_dist_res  # test that results are equal
    # 返回 True，表明两种方法计算的结果是相等的
    True
# 添加新的文档字符串到模块 "fdtrc" 中
add_newdoc("fdtrc",
    r"""
    fdtrc(dfn, dfd, x, out=None)

    F survival function.

    Returns the complemented F-distribution function (the integral of the
    density from `x` to infinity).

    Parameters
    ----------
    dfn : array_like
        First parameter (positive float).
    dfd : array_like
        Second parameter (positive float).
    x : array_like
        Argument (nonnegative float).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    y : scalar or ndarray
        The complemented F-distribution function with parameters `dfn` and
        `dfd` at `x`.

    See Also
    --------
    fdtr : F distribution cumulative distribution function
    fdtri : F distribution inverse cumulative distribution function
    scipy.stats.f : F distribution

    Notes
    -----
    使用正则化的不完全贝塔函数，根据下面的公式计算 F 分布的生存函数：

    .. math::
        F(d_n, d_d; x) = I_{d_d/(d_d + xd_n)}(d_d/2, d_n/2).

    这是 Cephes [1]_ 库中 `fdtrc` 的包装器。F 分布也可以通过 `scipy.stats.f` 获得。
    直接调用 `fdtrc` 可以比调用 `scipy.stats.f` 的 ``sf`` 方法提供更好的性能（参见最后一个示例）。

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    计算 ``dfn=1`` 和 ``dfd=2`` 在 ``x=1`` 处的函数值。

    >>> import numpy as np
    >>> from scipy.special import fdtrc
    >>> fdtrc(1, 2, 1)
    0.42264973081037427

    通过提供一个 NumPy 数组 `x` 来计算多个点的函数值。

    >>> x = np.array([0.5, 2., 3.])
    >>> fdtrc(1, 2, x)
    array([0.5527864 , 0.29289322, 0.22540333])

    绘制多组参数设置下的函数图像。

    >>> import matplotlib.pyplot as plt
    >>> dfn_parameters = [1, 5, 10, 50]
    >>> dfd_parameters = [1, 1, 2, 3]
    >>> linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
    >>> parameters_list = list(zip(dfn_parameters, dfd_parameters,
    ...                            linestyles))
    >>> x = np.linspace(0, 30, 1000)
    >>> fig, ax = plt.subplots()
    >>> for parameter_set in parameters_list:
    ...     dfn, dfd, style = parameter_set
    ...     fdtrc_vals = fdtrc(dfn, dfd, x)
    ...     ax.plot(x, fdtrc_vals, label=rf"$d_n={dfn},\, d_d={dfd}$",
    ...             ls=style)
    >>> ax.legend()
    >>> ax.set_xlabel("$x$")
    >>> ax.set_title("F distribution survival function")
    >>> plt.show()

    F 分布也可以通过 `scipy.stats.f` 获得。直接使用 `fdtrc` 可以比调用 `scipy.stats.f` 的 ``sf`` 方法更快，
    特别是对于小数组或单个值。为了获得相同的结果，必须使用以下参数化方式：
    ``stats.f(dfn, dfd).sf(x)=fdtrc(dfn, dfd, x)``.

    >>> from scipy.stats import f
    >>> dfn, dfd = 1, 2
    >>> x = 1
    >>> fdtrc_res = fdtrc(dfn, dfd, x)  # 调用函数 fdtrc 计算给定参数 dfn, dfd, x 的结果，并赋值给 fdtrc_res
    >>> f_dist_res = f(dfn, dfd).sf(x)  # 调用函数 f 计算给定参数 dfn, dfd 的结果，并对结果调用 sf 方法，将结果赋值给 f_dist_res
    >>> f_dist_res == fdtrc_res  # 检查两个结果是否相等
    True  # 返回结果为 True 表示两个结果相等
add_newdoc("fdtri",
    r"""
    fdtri(dfn, dfd, p, out=None)

    The `p`-th quantile of the F-distribution.

    This function is the inverse of the F-distribution CDF, `fdtr`, returning
    the `x` such that `fdtr(dfn, dfd, x) = p`.

    Parameters
    ----------
    dfn : array_like
        First parameter (positive float).
    dfd : array_like
        Second parameter (positive float).
    p : array_like
        Cumulative probability, in [0, 1].
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    x : scalar or ndarray
        The quantile corresponding to `p`.

    See Also
    --------
    fdtr : F distribution cumulative distribution function
    fdtrc : F distribution survival function
    scipy.stats.f : F distribution

    Notes
    -----
    The computation is carried out using the relation to the inverse
    regularized beta function, :math:`I^{-1}_x(a, b)`.  Let
    :math:`z = I^{-1}_p(d_d/2, d_n/2).`  Then,

    .. math::
        x = \frac{d_d (1 - z)}{d_n z}.

    If `p` is such that :math:`x < 0.5`, the following relation is used
    instead for improved stability: let
    :math:`z' = I^{-1}_{1 - p}(d_n/2, d_d/2).` Then,

    .. math::
        x = \frac{d_d z'}{d_n (1 - z')}.

    Wrapper for the Cephes [1]_ routine `fdtri`.

    The F distribution is also available as `scipy.stats.f`. Calling
    `fdtri` directly can improve performance compared to the ``ppf``
    method of `scipy.stats.f` (see last example below).

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    `fdtri` represents the inverse of the F distribution CDF which is
    available as `fdtr`. Here, we calculate the CDF for ``df1=1``, ``df2=2``
    at ``x=3``. `fdtri` then returns ``3`` given the same values for `df1`,
    `df2` and the computed CDF value.

    >>> import numpy as np
    >>> from scipy.special import fdtri, fdtr
    >>> df1, df2 = 1, 2
    >>> x = 3
    >>> cdf_value =  fdtr(df1, df2, x)
    >>> fdtri(df1, df2, cdf_value)
    3.000000000000006

    Calculate the function at several points by providing a NumPy array for
    `x`.

    >>> x = np.array([0.1, 0.4, 0.7])
    >>> fdtri(1, 2, x)
    array([0.02020202, 0.38095238, 1.92156863])

    Plot the function for several parameter sets.

    >>> import matplotlib.pyplot as plt
    >>> dfn_parameters = [50, 10, 1, 50]
    >>> dfd_parameters = [0.5, 1, 1, 5]
    >>> linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
    >>> parameters_list = list(zip(dfn_parameters, dfd_parameters,
    ...                            linestyles))
    >>> x = np.linspace(0, 1, 1000)
    >>> fig, ax = plt.subplots()
    >>> for parameter_set in parameters_list:
    ...     dfn, dfd, style = parameter_set
    ...     fdtri_vals = fdtri(dfn, dfd, x)
    ...     ax.plot(x, fdtri_vals, label=rf"$d_n={dfn},\, d_d={dfd}$",
    ...             ls=style)
    # 添加图例到当前图表
    ax.legend()
    # 设置 x 轴标签为数学表达式 "$x$"
    ax.set_xlabel("$x$")
    # 设置图表标题为指定的字符串
    title = "F distribution inverse cumulative distribution function"
    ax.set_title(title)
    # 设置 y 轴的数值范围为 0 到 30
    ax.set_ylim(0, 30)
    # 显示当前图表
    plt.show()

    """
    F 分布也可以通过 `scipy.stats.f` 进行访问。直接使用 `fdtri`
    可以比调用 `scipy.stats.f` 的 `ppf` 方法更快，特别是对于小数组或单个值。
    要获得相同的结果，必须使用以下参数化方式：
    ``stats.f(dfn, dfd).ppf(x)=fdtri(dfn, dfd, x)``。
    """

    # 导入 scipy.stats 模块中的 f 分布
    from scipy.stats import f
    # 设置 dfn 和 dfd 分别为 1 和 2
    dfn, dfd = 1, 2
    # 设置 x 的值为 0.7
    x = 0.7
    # 使用 fdtri 函数计算 F 分布的反函数结果，通常比下面的方法更快
    fdtri_res = fdtri(dfn, dfd, x)
    # 使用 scipy.stats.f 中的 ppf 方法计算 F 分布的反函数结果
    f_dist_res = f(dfn, dfd).ppf(x)
    # 检查两种方法得到的结果是否相等
    f_dist_res == fdtri_res  # test that results are equal
    """
    结果为 True
    """
# 将文档添加到名称为 "fresnel" 的函数中，描述其计算 Fresnel 积分的功能
add_newdoc("fresnel",
    r"""
    fresnel(z, out=None)

    Fresnel integrals.

    The Fresnel integrals are defined as

    .. math::

       S(z) &= \int_0^z \sin(\pi t^2 /2) dt \\
       C(z) &= \int_0^z \cos(\pi t^2 /2) dt.

    See [dlmf]_ for details.

    Parameters
    ----------
    z : array_like
        Real or complex valued argument
    out : 2-tuple of ndarrays, optional
        Optional output arrays for the function results

    Returns
    -------
    S, C : 2-tuple of scalar or ndarray
        Values of the Fresnel integrals

    See Also
    --------
    fresnel_zeros : zeros of the Fresnel integrals

    References
    ----------
    .. [dlmf] NIST Digital Library of Mathematical Functions
              https://dlmf.nist.gov/7.2#iii

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    As z goes to infinity along the real axis, S and C converge to 0.5.

    >>> S, C = sc.fresnel([0.1, 1, 10, 100, np.inf])
    >>> S
    """)
    # 调用 NumPy 库中的数组函数，创建一个包含浮点数的数组
    array([0.00052359, 0.43825915, 0.46816998, 0.4968169 , 0.5       ])
    # 变量 C 指向一个数组对象
    >>> C
    # 调用 SciPy 库中的函数，计算并返回一个包含浮点数的数组
    array([0.09999753, 0.7798934 , 0.49989869, 0.4999999 , 0.5       ])

    # 提示这些与误差函数 `erf` 有关。

    # 创建一个包含整数的 NumPy 数组对象
    >>> z = np.array([1, 2, 3, 4])
    # 创建一个复数值 NumPy 数组对象 zeta
    >>> zeta = 0.5 * np.sqrt(np.pi) * (1 - 1j) * z
    # 调用 SciPy 库中的函数 fresnel，返回两个复数值数组 S 和 C
    >>> S, C = sc.fresnel(z)
    # 返回一个复数值数组，由 C 和 S 组成
    array([0.7798934 +0.43825915j, 0.48825341+0.34341568j,
           0.60572079+0.496313j  , 0.49842603+0.42051575j])
    # 计算一个复数值数组，其中包含使用 erf 函数计算的值
    >>> 0.5 * (1 + 1j) * sc.erf(zeta)
    # 返回一个复数值数组
    array([0.7798934 +0.43825915j, 0.48825341+0.34341568j,
           0.60572079+0.496313j  , 0.49842603+0.42051575j])

    """
# 在 scipy.special 模块中添加新的文档字符串，说明 regularized lower incomplete gamma 函数的定义和用法
add_newdoc("gammainc",
    r"""
    gammainc(a, x, out=None)

    Regularized lower incomplete gamma function.

    It is defined as

    .. math::

        P(a, x) = \frac{1}{\Gamma(a)} \int_0^x t^{a - 1}e^{-t} dt

    for :math:`a > 0` and :math:`x \geq 0`. See [dlmf]_ for details.

    Parameters
    ----------
    a : array_like
        Positive parameter
    x : array_like
        Nonnegative argument
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        Values of the lower incomplete gamma function

    See Also
    --------
    gammaincc : regularized upper incomplete gamma function
    gammaincinv : inverse of the regularized lower incomplete gamma function
    gammainccinv : inverse of the regularized upper incomplete gamma function

    Notes
    -----
    The function satisfies the relation ``gammainc(a, x) +
    gammaincc(a, x) = 1`` where `gammaincc` is the regularized upper
    incomplete gamma function.

    The implementation largely follows that of [boost]_.

    References
    ----------
    .. [dlmf] NIST Digital Library of Mathematical functions
              https://dlmf.nist.gov/8.2#E4
    .. [boost] Maddock et. al., "Incomplete Gamma Functions",
       https://www.boost.org/doc/libs/1_61_0/libs/math/doc/html/math_toolkit/sf_gamma/igamma.html

    Examples
    --------
    >>> import scipy.special as sc

    It is the CDF of the gamma distribution, so it starts at 0 and
    monotonically increases to 1.

    >>> sc.gammainc(0.5, [0, 1, 10, 100])
    array([0.        , 0.84270079, 0.99999226, 1.        ])

    It is equal to one minus the upper incomplete gamma function.

    >>> a, x = 0.5, 0.4
    >>> sc.gammainc(a, x)
    0.6289066304773024
    >>> 1 - sc.gammaincc(a, x)
    0.6289066304773024

    """)

# 在 scipy.special 模块中添加新的文档字符串，说明 regularized upper incomplete gamma 函数的定义和用法
add_newdoc("gammaincc",
    r"""
    gammaincc(a, x, out=None)

    Regularized upper incomplete gamma function.

    It is defined as

    .. math::

        Q(a, x) = \frac{1}{\Gamma(a)} \int_x^\infty t^{a - 1}e^{-t} dt

    for :math:`a > 0` and :math:`x \geq 0`. See [dlmf]_ for details.

    Parameters
    ----------
    a : array_like
        Positive parameter
    x : array_like
        Nonnegative argument
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        Values of the upper incomplete gamma function

    See Also
    --------
    gammainc : regularized lower incomplete gamma function
    gammaincinv : inverse of the regularized lower incomplete gamma function
    gammainccinv : inverse of the regularized upper incomplete gamma function

    Notes
    -----
    The function satisfies the relation ``gammainc(a, x) +
    gammaincc(a, x) = 1`` where `gammainc` is the regularized lower
    incomplete gamma function.

    The implementation largely follows that of [boost]_.

    References
    ----------
    .. [dlmf] NIST Digital Library of Mathematical functions
              https://dlmf.nist.gov/8.2#E4
    .. [boost] Maddock et. al., "Incomplete Gamma Functions",
       https://www.boost.org/doc/libs/1_61_0/libs/math/doc/html/math_toolkit/sf_gamma/igamma.html
    """
    .. [dlmf] NIST Digital Library of Mathematical functions
              https://dlmf.nist.gov/8.2#E4
    .. [boost] Maddock et. al., "Incomplete Gamma Functions",
       https://www.boost.org/doc/libs/1_61_0/libs/math/doc/html/math_toolkit/sf_gamma/igamma.html

    Examples
    --------
    >>> import scipy.special as sc

    It is the survival function of the gamma distribution, so it
    starts at 1 and monotonically decreases to 0.

    >>> sc.gammaincc(0.5, [0, 1, 10, 100, 1000])
    array([1.00000000e+00, 1.57299207e-01, 7.74421643e-06, 2.08848758e-45,
           0.00000000e+00])

    It is equal to one minus the lower incomplete gamma function.

    >>> a, x = 0.5, 0.4
    >>> sc.gammaincc(a, x)
    0.37109336952269756
    >>> 1 - sc.gammainc(a, x)
    0.37109336952269756

    """


注释：

# 这部分是一个文档字符串，用于描述 gamma 函数的上不完全 gamma 函数及其用法示例。
# 文档提供了引用来源 [dlmf] 和 [boost]，分别指向 NIST 数学函数库和 Boost C++ 文档中关于不完全 gamma 函数的页面。
# 示例展示了如何使用 scipy.special 库中的 gammaincc 函数计算 gamma 分布的生存函数，以及它如何与下不完全 gamma 函数相关联。
# 第一个示例展示了 gammaincc 函数对不同参数和值的数组计算，结果为一个 numpy 数组。
# 第二个示例展示了如何计算特定参数 a 和 x 下的 gammaincc 值，并与 1 减去 gammainc(a, x) 的结果进行对比。
add_newdoc("gammainccinv",
    """
    gammainccinv(a, y, out=None)

    Inverse of the regularized upper incomplete gamma function.

    Given an input :math:`y` between 0 and 1, returns :math:`x` such
    that :math:`y = Q(a, x)`. Here :math:`Q` is the regularized upper
    incomplete gamma function; see `gammaincc`. This is well-defined
    because the upper incomplete gamma function is monotonic as can
    be seen from its definition in [dlmf]_.

    Parameters
    ----------
    a : array_like
        Positive parameter
    y : array_like
        Argument between 0 and 1, inclusive
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        Values of the inverse of the upper incomplete gamma function

    See Also
    --------
    gammaincc : regularized upper incomplete gamma function
    gammainc : regularized lower incomplete gamma function
    gammaincinv : inverse of the regularized lower incomplete gamma function

    References
    ----------
    .. [dlmf] NIST Digital Library of Mathematical Functions
              https://dlmf.nist.gov/8.2#E4

    Examples
    --------
    >>> import scipy.special as sc

    It starts at infinity and monotonically decreases to 0.

    >>> sc.gammainccinv(0.5, [0, 0.1, 0.5, 1])
    array([       inf, 1.35277173, 0.22746821, 0.        ])

    It inverts the upper incomplete gamma function.

    >>> a, x = 0.5, [0, 0.1, 0.5, 1]
    >>> sc.gammaincc(a, sc.gammainccinv(a, x))
    array([0. , 0.1, 0.5, 1. ])

    >>> a, x = 0.5, [0, 10, 50]
    >>> sc.gammainccinv(a, sc.gammaincc(a, x))
    array([ 0., 10., 50.])

    """)

add_newdoc("gammaincinv",
    """
    gammaincinv(a, y, out=None)

    Inverse to the regularized lower incomplete gamma function.

    Given an input :math:`y` between 0 and 1, returns :math:`x` such
    that :math:`y = P(a, x)`. Here :math:`P` is the regularized lower
    incomplete gamma function; see `gammainc`. This is well-defined
    because the lower incomplete gamma function is monotonic as can be
    seen from its definition in [dlmf]_.

    Parameters
    ----------
    a : array_like
        Positive parameter
    y : array_like
        Parameter between 0 and 1, inclusive
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        Values of the inverse of the lower incomplete gamma function

    See Also
    --------
    gammainc : regularized lower incomplete gamma function
    gammaincc : regularized upper incomplete gamma function
    gammainccinv : inverse of the regularized upper incomplete gamma function

    References
    ----------
    .. [dlmf] NIST Digital Library of Mathematical Functions
              https://dlmf.nist.gov/8.2#E4

    Examples
    --------
    >>> import scipy.special as sc

    It starts at 0 and monotonically increases to infinity.

    >>> sc.gammaincinv(0.5, [0, 0.1 ,0.5, 1])
    # 该函数返回给定参数 a 和 y 下的逆正则化下不完全 Gamma 函数的值，返回的是 x，使得 y = P(a, x)
    array([       inf, 0.03182837, 0.22603083, 0.77894304])
    
    """)
    # 计算下不完全伽玛函数的反函数值
    array([0.        , 0.00789539, 0.22746821,        inf])
    
    # 示例：使用 scipy 库计算下不完全伽玛函数的反函数值
    >>> a, x = 0.5, [0, 0.1, 0.5, 1]
    >>> sc.gammainc(a, sc.gammaincinv(a, x))
    array([0. , 0.1, 0.5, 1. ])
    
    # 示例：使用 scipy 库计算下不完全伽玛函数和其反函数的组合
    >>> a, x = 0.5, [0, 10, 25]
    >>> sc.gammaincinv(a, sc.gammainc(a, x))
    array([ 0.        , 10.        , 25.00001465])
# 添加新的文档字符串到模块的"gammasgn"函数
add_newdoc("gammasgn",
    r"""
    gammasgn(x, out=None)

    Gamma函数的符号函数。

    定义如下：

    .. math::

       \text{gammasgn}(x) =
       \begin{cases}
         +1 & \Gamma(x) > 0 \\
         -1 & \Gamma(x) < 0
       \end{cases}

    其中 :math:`\Gamma` 是Gamma函数；参见 `gamma`。此定义完整，因为Gamma函数永不为零；
    参见 [dlmf]_ 中的讨论。

    Parameters
    ----------
    x : array_like
        实数参数
    out : ndarray, optional
        可选的输出数组，用于存储函数值

    Returns
    -------
    scalar or ndarray
        Gamma函数的符号

    See Also
    --------
    gamma : Gamma函数
    gammaln : Gamma函数绝对值的对数
    loggamma : Gamma函数对数的解析延拓

    Notes
    -----
    Gamma函数可以通过 ``gammasgn(x) * np.exp(gammaln(x))`` 计算。

    References
    ----------
    .. [dlmf] NIST数学函数数字图书馆
              https://dlmf.nist.gov/5.2#E1

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    对于 ``x > 0`` 为1。

    >>> sc.gammasgn([1, 2, 3, 4])
    array([1., 1., 1., 1.])

    对于负整数交替为 -1 和 1。

    >>> sc.gammasgn([-0.5, -1.5, -2.5, -3.5])
    array([-1.,  1., -1.,  1.])

    可以用于计算Gamma函数。

    >>> x = [1.5, 0.5, -0.5, -1.5]
    >>> sc.gammasgn(x) * np.exp(sc.gammaln(x))
    array([ 0.88622693,  1.77245385, -3.5449077 ,  2.3632718 ])
    >>> sc.gamma(x)
    array([ 0.88622693,  1.77245385, -3.5449077 ,  2.3632718 ])

    """)

# 添加新的文档字符串到模块的"gdtr"函数
add_newdoc("gdtr",
    r"""
    gdtr(a, b, x, out=None)

    Gamma分布的累积分布函数。

    返回Gamma概率密度函数从0到`x`的积分，

    .. math::

        F = \int_0^x \frac{a^b}{\Gamma(b)} t^{b-1} e^{-at}\,dt,

    其中 :math:`\Gamma` 是Gamma函数。

    Parameters
    ----------
    a : array_like
        Gamma分布的速率参数，有时表示为 :math:`\beta`（float）。也是尺度参数 :math:`\theta` 的倒数。
    b : array_like
        Gamma分布的形状参数，有时表示为 :math:`\alpha`（float）。
    x : array_like
        分位数（积分上限；float）。
    out : ndarray, optional
        可选的输出数组，用于存储函数值

    Returns
    -------
    F : scalar or ndarray
        带有参数 `a` 和 `b` 的Gamma分布的累积分布函数在 `x` 处的值。

    See Also
    --------
    gdtrc : Gamma分布的补累积分布函数。
    scipy.stats.gamma: Gamma分布

    Notes
    -----
    评估使用与不完全Gamma积分（正则化Gamma函数）的关系。

    """
    Wrapper for the Cephes [1]_ routine `gdtr`. Calling `gdtr` directly can
    improve performance compared to the ``cdf`` method of `scipy.stats.gamma`
    (see last example below).

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    Compute the function for ``a=1``, ``b=2`` at ``x=5``.

    >>> import numpy as np
    >>> from scipy.special import gdtr
    >>> import matplotlib.pyplot as plt
    >>> gdtr(1., 2., 5.)
    0.9595723180054873

    Compute the function for ``a=1`` and ``b=2`` at several points by
    providing a NumPy array for `x`.

    >>> xvalues = np.array([1., 2., 3., 4])
    >>> gdtr(1., 1., xvalues)
    array([0.63212056, 0.86466472, 0.95021293, 0.98168436])

    `gdtr` can evaluate different parameter sets by providing arrays with
    broadcasting compatible shapes for `a`, `b` and `x`. Here we compute the
    function for three different `a` at four positions `x` and ``b=3``,
    resulting in a 3x4 array.

    >>> a = np.array([[0.5], [1.5], [2.5]])
    >>> x = np.array([1., 2., 3., 4])
    >>> a.shape, x.shape
    ((3, 1), (4,))

    >>> gdtr(a, 3., x)
    array([[0.01438768, 0.0803014 , 0.19115317, 0.32332358],
           [0.19115317, 0.57680992, 0.82642193, 0.9380312 ],
           [0.45618688, 0.87534798, 0.97974328, 0.9972306 ]])

    Plot the function for four different parameter sets.

    >>> a_parameters = [0.3, 1, 2, 6]
    >>> b_parameters = [2, 10, 15, 20]
    >>> linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
    >>> parameters_list = list(zip(a_parameters, b_parameters, linestyles))
    >>> x = np.linspace(0, 30, 1000)
    >>> fig, ax = plt.subplots()
    >>> for parameter_set in parameters_list:
    ...     a, b, style = parameter_set
    ...     gdtr_vals = gdtr(a, b, x)
    ...     ax.plot(x, gdtr_vals, label=fr"$a= {a},\, b={b}$", ls=style)
    >>> ax.legend()
    >>> ax.set_xlabel("$x$")
    >>> ax.set_title("Gamma distribution cumulative distribution function")
    >>> plt.show()

    The gamma distribution is also available as `scipy.stats.gamma`. Using
    `gdtr` directly can be much faster than calling the ``cdf`` method of
    `scipy.stats.gamma`, especially for small arrays or individual values.
    To get the same results one must use the following parametrization:
    ``stats.gamma(b, scale=1/a).cdf(x)=gdtr(a, b, x)``.

    >>> from scipy.stats import gamma
    >>> a = 2.
    >>> b = 3
    >>> x = 1.
    >>> gdtr_result = gdtr(a, b, x)  # this will often be faster than below
    >>> gamma_dist_result = gamma(b, scale=1/a).cdf(x)
    >>> gdtr_result == gamma_dist_result  # test that results are equal
    True
# 将文档字符串添加到 `gdtrc` 函数的文档中，用于描述其功能和用法
add_newdoc("gdtrc",
    r"""
    gdtrc(a, b, x, out=None)

    Gamma distribution survival function.

    Integral from `x` to infinity of the gamma probability density function,

    .. math::

        F = \int_x^\infty \frac{a^b}{\Gamma(b)} t^{b-1} e^{-at}\,dt,

    where :math:`\Gamma` is the gamma function.

    Parameters
    ----------
    a : array_like
        The rate parameter of the gamma distribution, sometimes denoted
        :math:`\beta` (float). It is also the reciprocal of the scale
        parameter :math:`\theta`.
    b : array_like
        The shape parameter of the gamma distribution, sometimes denoted
        :math:`\alpha` (float).
    x : array_like
        The quantile (lower limit of integration; float).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    F : scalar or ndarray
        The survival function of the gamma distribution with parameters `a`
        and `b` evaluated at `x`.

    See Also
    --------
    gdtr: Gamma distribution cumulative distribution function
    scipy.stats.gamma: Gamma distribution
    gdtrix

    Notes
    -----
    The evaluation is carried out using the relation to the incomplete gamma
    integral (regularized gamma function).

    Wrapper for the Cephes [1]_ routine `gdtrc`. Calling `gdtrc` directly can
    improve performance compared to the ``sf`` method of `scipy.stats.gamma`
    (see last example below).

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    Compute the function for ``a=1`` and ``b=2`` at ``x=5``.

    >>> import numpy as np
    >>> from scipy.special import gdtrc
    >>> import matplotlib.pyplot as plt
    >>> gdtrc(1., 2., 5.)
    0.04042768199451279

    Compute the function for ``a=1``, ``b=2`` at several points by providing
    a NumPy array for `x`.

    >>> xvalues = np.array([1., 2., 3., 4])
    >>> gdtrc(1., 1., xvalues)
    array([0.36787944, 0.13533528, 0.04978707, 0.01831564])

    `gdtrc` can evaluate different parameter sets by providing arrays with
    broadcasting compatible shapes for `a`, `b` and `x`. Here we compute the
    function for three different `a` at four positions `x` and ``b=3``,
    resulting in a 3x4 array.

    >>> a = np.array([[0.5], [1.5], [2.5]])
    >>> x = np.array([1., 2., 3., 4])
    >>> a.shape, x.shape
    ((3, 1), (4,))

    >>> gdtrc(a, 3., x)
    array([[0.98561232, 0.9196986 , 0.80884683, 0.67667642],
           [0.80884683, 0.42319008, 0.17357807, 0.0619688 ],
           [0.54381312, 0.12465202, 0.02025672, 0.0027694 ]])

    Plot the function for four different parameter sets.

    >>> a_parameters = [0.3, 1, 2, 6]
    >>> b_parameters = [2, 10, 15, 20]
    >>> linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
    >>> parameters_list = list(zip(a_parameters, b_parameters, linestyles))
    >>> x = np.linspace(0, 30, 1000)
    >>> fig, ax = plt.subplots()
    # 遍历参数列表，每个参数集合包含 a, b, style 三个元素
    >>> for parameter_set in parameters_list:
    # 将当前参数集合中的 a, b, style 分别解包
    ...     a, b, style = parameter_set
    # 使用 gdtrc 函数计算 Gamma 分布的右尾累积分布函数值 gdtrc_vals
    ...     gdtrc_vals = gdtrc(a, b, x)
    # 在图形 ax 上绘制 x 和 gdtrc_vals，标签为参数 a 和 b 的字符串表示
    ...     ax.plot(x, gdtrc_vals, label=fr"$a= {a},\, b={b}$", ls=style)
    # 添加图例到图形 ax 上
    >>> ax.legend()
    # 设置 x 轴标签为 "$x$"
    >>> ax.set_xlabel("$x$")
    # 设置图形标题为 "Gamma distribution survival function"
    >>> ax.set_title("Gamma distribution survival function")
    # 显示绘制的图形
    >>> plt.show()

    # 提示：Gamma 分布也可以使用 `scipy.stats.gamma` 获取
    # 直接使用 gdtrc 可能比调用 `scipy.stats.gamma` 的 `sf` 方法更快，特别是对于小数组或单个值。
    # 要获得相同的结果，必须使用以下参数化方式：``stats.gamma(b, scale=1/a).sf(x)=gdtrc(a, b, x)``.

    # 导入 `scipy.stats` 中的 gamma 模块
    >>> from scipy.stats import gamma
    # 设置参数 a, b 和 x 的值
    >>> a = 2
    >>> b = 3
    >>> x = 1.
    # 使用 gdtrc 函数计算 Gamma 分布的右尾累积分布函数值 gdtrc_result，通常比下面的方法更快
    >>> gdtrc_result = gdtrc(a, b, x)  # this will often be faster than below
    # 使用 `scipy.stats.gamma` 中的 `sf` 方法计算 Gamma 分布的右尾累积分布函数值 gamma_dist_result
    >>> gamma_dist_result = gamma(b, scale=1/a).sf(x)
    # 检查 gdtrc_result 和 gamma_dist_result 是否相等
    >>> gdtrc_result == gamma_dist_result  # test that results are equal
    # 结果为 True，说明两种方法得到的结果相等
    True
# 将函数 `gdtria` 添加到模块中，提供了对 gamma 分布累积分布函数 `gdtr` 相对于参数 `a` 的逆函数
add_newdoc("gdtria",
    """
    gdtria(p, b, x, out=None)

    Inverse of `gdtr` vs a.

    Returns the inverse with respect to the parameter `a` of ``p =
    gdtr(a, b, x)``, the cumulative distribution function of the gamma
    distribution.

    Parameters
    ----------
    p : array_like
        Probability values.
        概率值。
    b : array_like
        `gdtr(a, b, x)` 的 `b` 参数值。`b` 是 gamma 分布的形状参数。
    x : array_like
        非负实数值，来自 gamma 分布的定义域。
    out : ndarray, optional
        如果给定第四个参数，必须是一个大小与 `a`、`b` 和 `x` 广播结果匹配的 numpy.ndarray。
        `out` 是函数返回的数组。

    Returns
    -------
    a : scalar or ndarray
        `a` 参数的值，使得 ``p = gdtr(a, b, x)``。 ``1/a`` 是 gamma 分布的尺度参数。

    See Also
    --------
    gdtr : gamma 分布的累积分布函数。
    gdtrib : `gdtr(a, b, x)` 相对于 `b` 的逆函数。
    gdtrix : `gdtr(a, b, x)` 相对于 `x` 的逆函数。

    Notes
    -----
    对于 Fortran 程序 `cdfgam` 的 CDFLIB [1]_ 包装器。

    使用 DiDinato 和 Morris [2]_ 的例程计算累积分布函数 `p`。计算 `a` 包括搜索产生所需 `p` 值的 `a` 值。
    搜索依赖于 `p` 随 `a` 的单调性。

    References
    ----------
    .. [1] Barry Brown, James Lovato, and Kathy Russell,
           CDFLIB: Library of Fortran Routines for Cumulative Distribution
           Functions, Inverses, and Other Parameters.
    .. [2] DiDinato, A. R. and Morris, A. H.,
           Computation of the incomplete gamma function ratios and their
           inverse.  ACM Trans. Math. Softw. 12 (1986), 377-393.

    Examples
    --------
    先评估 `gdtr`。

    >>> from scipy.special import gdtr, gdtria
    >>> p = gdtr(1.2, 3.4, 5.6)
    >>> print(p)
    0.94378087442

    验证逆函数。

    >>> gdtria(p, 3.4, 5.6)
    1.2
    """)

# 将函数 `gdtrib` 添加到模块中，提供了对 gamma 分布累积分布函数 `gdtr` 相对于参数 `b` 的逆函数
add_newdoc("gdtrib",
    """
    gdtrib(a, p, x, out=None)

    Inverse of `gdtr` vs b.

    Returns the inverse with respect to the parameter `b` of ``p =
    gdtr(a, b, x)``, the cumulative distribution function of the gamma
    distribution.

    Parameters
    ----------
    a : array_like
        `gdtr(a, b, x)` 的 `a` 参数值。 ``1/a`` 是 gamma 分布的尺度参数。
    p : array_like
        概率值。
    x : array_like
        非负实数值，来自 gamma 分布的定义域。
    out : ndarray, optional
        如果给定第四个参数，必须是一个大小与 `a`、`b` 和 `x` 广播结果匹配的 numpy.ndarray。
        `out` 是函数返回的数组。

    Returns
    -------
    b : scalar or ndarray
        `b` 参数的值，使得 ``p = gdtr(a, b, x)``。 

    """)
    b : scalar or ndarray
        `b`参数的值，使得 `p = gdtr(a, b, x)` 成立。`b` 是伽玛分布的形状参数。

    See Also
    --------
    gdtr : 伽玛分布的累积分布函数。
    gdtria : 对 `a` 进行求逆的函数，即 `gdtr(a, b, x)` 的反函数。
    gdtrix : 对 `x` 进行求逆的函数，即 `gdtr(a, b, x)` 的反函数。

    Notes
    -----
    `cdfgam` 的 Python 封装。

    使用 DiDinato 和 Morris 的算法 [2]_ 计算累积分布函数 `p`。计算 `b` 的过程中，
    通过 `p` 关于 `b` 的单调性进行搜索，以找到合适的 `b` 值。

    References
    ----------
    .. [1] Barry Brown, James Lovato, and Kathy Russell,
           CDFLIB: Library of Fortran Routines for Cumulative Distribution
           Functions, Inverses, and Other Parameters.
    .. [2] DiDinato, A. R. and Morris, A. H.,
           Computation of the incomplete gamma function ratios and their
           inverse.  ACM Trans. Math. Softw. 12 (1986), 377-393.

    Examples
    --------
    首先评估 `gdtr`。

    >>> from scipy.special import gdtr, gdtrib
    >>> p = gdtr(1.2, 3.4, 5.6)
    >>> print(p)
    0.94378087442

    验证反函数。

    >>> gdtrib(1.2, p, 5.6)
    3.3999999999723882
# 添加新的文档字符串到模块 "gdtrix" 中
add_newdoc("gdtrix",
    """
    gdtrix(a, b, p, out=None)

    Inverse of `gdtr` vs x.

    Returns the inverse with respect to the parameter `x` of ``p =
    gdtr(a, b, x)``, the cumulative distribution function of the gamma
    distribution. This is also known as the pth quantile of the
    distribution.

    Parameters
    ----------
    a : array_like
        `a` parameter values of ``gdtr(a, b, x)``. ``1/a`` is the "scale"
        parameter of the gamma distribution.
    b : array_like
        `b` parameter values of ``gdtr(a, b, x)``. `b` is the "shape" parameter
        of the gamma distribution.
    p : array_like
        Probability values.
    out : ndarray, optional
        If a fourth argument is given, it must be a numpy.ndarray whose size
        matches the broadcast result of `a`, `b` and `x`. `out` is then the
        array returned by the function.

    Returns
    -------
    x : scalar or ndarray
        Values of the `x` parameter such that `p = gdtr(a, b, x)`.

    See Also
    --------
    gdtr : CDF of the gamma distribution.
    gdtria : Inverse with respect to `a` of ``gdtr(a, b, x)``.
    gdtrib : Inverse with respect to `b` of ``gdtr(a, b, x)``.

    Notes
    -----
    Wrapper for the CDFLIB [1]_ Fortran routine `cdfgam`.

    The cumulative distribution function `p` is computed using a routine by
    DiDinato and Morris [2]_. Computation of `x` involves a search for a value
    that produces the desired value of `p`. The search relies on the
    monotonicity of `p` with `x`.

    References
    ----------
    .. [1] Barry Brown, James Lovato, and Kathy Russell,
           CDFLIB: Library of Fortran Routines for Cumulative Distribution
           Functions, Inverses, and Other Parameters.
    .. [2] DiDinato, A. R. and Morris, A. H.,
           Computation of the incomplete gamma function ratios and their
           inverse.  ACM Trans. Math. Softw. 12 (1986), 377-393.

    Examples
    --------
    First evaluate `gdtr`.

    >>> from scipy.special import gdtr, gdtrix
    >>> p = gdtr(1.2, 3.4, 5.6)
    >>> print(p)
    0.94378087442

    Verify the inverse.

    >>> gdtrix(1.2, 3.4, p)
    5.5999999999999996
    """)

# 添加新的文档字符串到模块 "hankel1" 中，使用原始字符串表示方式
add_newdoc("hankel1",
    r"""
    hankel1(v, z, out=None)

    Hankel function of the first kind

    Parameters
    ----------
    v : array_like
        Order (float).
    z : array_like
        Argument (float or complex).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        Values of the Hankel function of the first kind.

    See Also
    --------
    hankel1e : ndarray
        This function with leading exponential behavior stripped off.

    Notes
    -----
    A wrapper for the AMOS [1]_ routine `zbesh`, which carries out the
    computation using the relation,

    .. math:: H^{(1)}_v(z) =
              \frac{2}{\imath\pi} \exp(-\imath \pi v/2) K_v(z \exp(-\imath\pi/2))

    """)
    where :math:`K_v` is the modified Bessel function of the second kind.
    对于 :math:`K_v`，这是第二类修正贝塞尔函数。

    For negative orders, the relation

    .. math:: H^{(1)}_{-v}(z) = H^{(1)}_v(z) \exp(\imath\pi v)

    is used.
    对于负阶数，使用以下关系式：
    当 :math:`v` 为负数时，有 :math:`H^{(1)}_{-v}(z) = H^{(1)}_v(z) \exp(\imath\pi v)`。

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/
    参考文献
    ----------
    参考文献 [1]：Donald E. Amos，《AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order》，网址：http://netlib.org/amos/
# 给 hankel1e 函数添加文档字符串
add_newdoc("hankel1e",
    r"""
    hankel1e(v, z, out=None)

    Exponentially scaled Hankel function of the first kind

    Defined as::

        hankel1e(v, z) = hankel1(v, z) * exp(-1j * z)

    Parameters
    ----------
    v : array_like
        Order (float).
        订单（浮点数）。
    z : array_like
        Argument (float or complex).
        参数（浮点数或复数）。
    out : ndarray, optional
        Optional output array for the function values
        可选的输出数组用于函数值。

    Returns
    -------
    scalar or ndarray
        Values of the exponentially scaled Hankel function.
        指数缩放的第一类汉克尔函数的值。

    Notes
    -----
    A wrapper for the AMOS [1]_ routine `zbesh`, which carries out the
    computation using the relation,

    .. math:: H^{(1)}_v(z) =
              \frac{2}{\imath\pi} \exp(-\imath \pi v/2) K_v(z \exp(-\imath\pi/2))

    where :math:`K_v` is the modified Bessel function of the second kind.
    For negative orders, the relation

    .. math:: H^{(1)}_{-v}(z) = H^{(1)}_v(z) \exp(\imath\pi v)

    is used.

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/
    """)
    
# 给 hankel2 函数添加文档字符串
add_newdoc("hankel2",
    r"""
    hankel2(v, z, out=None)

    Hankel function of the second kind

    Parameters
    ----------
    v : array_like
        Order (float).
        订单（浮点数）。
    z : array_like
        Argument (float or complex).
        参数（浮点数或复数）。
    out : ndarray, optional
        Optional output array for the function values
        可选的输出数组用于函数值。

    Returns
    -------
    scalar or ndarray
        Values of the Hankel function of the second kind.
        第二类汉克尔函数的值。

    See Also
    --------
    hankel2e : this function with leading exponential behavior stripped off.
    hankel2e：去除前导指数行为的此函数。

    Notes
    -----
    A wrapper for the AMOS [1]_ routine `zbesh`, which carries out the
    computation using the relation,

    .. math:: H^{(2)}_v(z) =
              -\frac{2}{\imath\pi} \exp(\imath \pi v/2) K_v(z \exp(\imath\pi/2))

    where :math:`K_v` is the modified Bessel function of the second kind.
    For negative orders, the relation

    .. math:: H^{(2)}_{-v}(z) = H^{(2)}_v(z) \exp(-\imath\pi v)

    is used.

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/
    """)

# 给 hankel2e 函数添加文档字符串
add_newdoc("hankel2e",
    r"""
    hankel2e(v, z, out=None)

    Exponentially scaled Hankel function of the second kind

    Defined as::

        hankel2e(v, z) = hankel2(v, z) * exp(1j * z)

    Parameters
    ----------
    v : array_like
        Order (float).
        订单（浮点数）。
    z : array_like
        Argument (float or complex).
        参数（浮点数或复数）。
    out : ndarray, optional
        Optional output array for the function values
        可选的输出数组用于函数值。

    Returns
    -------
    scalar or ndarray
        Values of the exponentially scaled Hankel function of the second kind.
        指数缩放的第二类汉克尔函数的值。

    Notes
    -----
    A wrapper for the AMOS [1]_ routine `zbesh`, which carries out the
    computation using the relation,
    这是 AMOS [1]_ 例程 `zbesh` 的包装器，用于根据以下关系进行计算，

    ```
    .. math:: H^{(2)}_v(z) =
              -\frac{2}{\imath\pi} \exp(\imath \pi v/2) K_v(z \exp(\imath\pi/2))
    ```

    where :math:`K_v` is the modified Bessel function of the second kind.
    For negative orders, the relation

    .. math:: H^{(2)}_{-v}(z) = H^{(2)}_v(z) \exp(-\imath\pi v)

    is used.

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/
    """)
    .. math:: H^{(2)}_v(z) = -\frac{2}{\imath\pi}
              \exp(\frac{\imath \pi v}{2}) K_v(z \exp(\frac{\imath\pi}{2}))

    where :math:`K_v` is the modified Bessel function of the second kind.
    For negative orders, the relation

    .. math:: H^{(2)}_{-v}(z) = H^{(2)}_v(z) \exp(-\imath\pi v)

    is used.

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/


注释：

# 此部分是数学公式和引用的说明文档，介绍了修正第二类贝塞尔函数的定义及其特性。
# 向模块添加新的文档字符串，描述 Huber 损失函数
add_newdoc("huber",
    r"""
    huber(delta, r, out=None)

    Huber loss function.

    .. math:: \text{huber}(\delta, r) = \begin{cases} \infty & \delta < 0  \\
              \frac{1}{2}r^2 & 0 \le \delta, | r | \le \delta \\
              \delta ( |r| - \frac{1}{2}\delta ) & \text{otherwise} \end{cases}

    Parameters
    ----------
    delta : ndarray
        输入数组，表示二次损失与线性损失的转折点。
    r : ndarray
        输入数组，可能表示残差。
    out : ndarray, optional
        可选的输出数组，用于存储函数值。

    Returns
    -------
    scalar or ndarray
        计算得到的 Huber 损失函数值。

    See Also
    --------
    pseudo_huber : 此函数的平滑近似

    Notes
    -----
    `huber` 在鲁棒统计或机器学习中作为损失函数非常有用，可以减少离群值的影响，相比常见的平方误差损失，大于 `delta` 的残差不会被平方处理[1]_。

    通常情况下，`r` 表示残差，即模型预测与数据之间的差异。当 :math:`|r|\leq\delta` 时，`huber` 类似于平方误差，当 :math:`|r|>\delta` 时类似于绝对误差。这种方式，Huber 损失函数在小残差时像平方误差损失函数一样快速收敛，并且在离群值（:math:`|r|>\delta`）时减少影响。因为 :math:`\delta` 是平方和绝对误差之间的切换点，需要针对每个问题进行精心调整。`huber` 也是凸函数，适合基于梯度的优化方法。

    .. versionadded:: 0.15.0

    References
    ----------
    .. [1] Peter Huber. "Robust Estimation of a Location Parameter",
           1964. Annals of Statistics. 53 (1): 73 - 101.

    Examples
    --------
    导入所有必要的模块。

    >>> import numpy as np
    >>> from scipy.special import huber
    >>> import matplotlib.pyplot as plt

    计算 `delta=1` 时，`r=2` 的函数值。

    >>> huber(1., 2.)
    1.5

    通过提供 NumPy 数组或列表作为 `delta`，计算不同 `delta` 的函数值。

    >>> huber([1., 3., 5.], 4.)
    array([3.5, 7.5, 8. ])

    通过提供 NumPy 数组或列表作为 `r`，计算不同 `r` 点的函数值。

    >>> huber(2., np.array([1., 1.5, 3.]))
    array([0.5  , 1.125, 4.   ])

    可以通过为 `delta` 和 `r` 同时提供形状兼容的数组来计算不同 `delta` 和 `r` 的函数值。

    >>> r = np.array([1., 2.5, 8., 10.])
    >>> deltas = np.array([[1.], [5.], [9.]])
    >>> print(r.shape, deltas.shape)
    (4,) (3, 1)

    >>> huber(deltas, r)
    array([[ 0.5  ,  2.   ,  7.5  ,  9.5  ],
           [ 0.5  ,  3.125, 27.5  , 37.5  ],
           [ 0.5  ,  3.125, 32.   , 49.5  ]])

    绘制不同 `delta` 的函数图像。
    ```
    >>> x = np.linspace(-4, 4, 500)
    >>> deltas = [1, 2, 3]
    >>> linestyles = ["dashed", "dotted", "dashdot"]
    >>> fig, ax = plt.subplots()
    >>> combined_plot_parameters = list(zip(deltas, linestyles))
    >>> for delta, style in combined_plot_parameters:
    ...     # 使用 Huber 函数绘制曲线，标签中显示当前 delta 值
    ...     ax.plot(x, huber(delta, x), label=fr"$\delta={delta}$", ls=style)
    >>> ax.legend(loc="upper center")  # 设置图例位置为图像上方中心
    >>> ax.set_xlabel("$x$")  # 设置 x 轴标签
    >>> ax.set_title(r"Huber loss function $h_{\delta}(x)$")  # 设置图像标题，显示 Huber 损失函数公式
    >>> ax.set_xlim(-4, 4)  # 设置 x 轴范围
    >>> ax.set_ylim(0, 8)  # 设置 y 轴范围
    >>> plt.show()  # 显示绘制的图像
# 为 `hyp0f1` 函数添加新的文档字符串和文档说明
add_newdoc("hyp0f1",
    r"""
    hyp0f1(v, z, out=None)

    Confluent hypergeometric limit function 0F1.

    Parameters
    ----------
    v : array_like
        Real-valued parameter
        实数参数 `v`
    z : array_like
        Real- or complex-valued argument
        实数或复数参数 `z`
    out : ndarray, optional
        Optional output array for the function results
        可选的输出数组用于存放函数结果

    Returns
    -------
    scalar or ndarray
        The confluent hypergeometric limit function
        返回共轭超几何极限函数的标量或数组形式结果

    Notes
    -----
    This function is defined as:

    .. math:: _0F_1(v, z) = \sum_{k=0}^{\infty}\frac{z^k}{(v)_k k!}.

    It's also the limit as :math:`q \to \infty` of :math:`_1F_1(q; v; z/q)`,
    and satisfies the differential equation :math:`f''(z) + vf'(z) =
    f(z)`. See [1]_ for more information.
    这个函数定义如下：

    .. math:: _0F_1(v, z) = \sum_{k=0}^{\infty}\frac{z^k}{(v)_k k!}.

    它也是当 :math:`q \to \infty` 时 :math:`_1F_1(q; v; z/q)` 的极限，
    并满足微分方程 :math:`f''(z) + vf'(z) = f(z)`。详细信息参见 [1]_。

    References
    ----------
    .. [1] Wolfram MathWorld, "Confluent Hypergeometric Limit Function",
           http://mathworld.wolfram.com/ConfluentHypergeometricLimitFunction.html

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    It is one when `z` is zero.
    当 `z` 为零时，函数值为1。

    >>> sc.hyp0f1(1, 0)
    1.0

    It is the limit of the confluent hypergeometric function as `q`
    goes to infinity.
    当 `q` 趋向无穷时，它是共轭超几何函数的极限。

    >>> q = np.array([1, 10, 100, 1000])
    >>> v = 1
    >>> z = 1
    >>> sc.hyp1f1(q, v, z / q)
    array([2.71828183, 2.31481985, 2.28303778, 2.27992985])
    >>> sc.hyp0f1(v, z)
    2.2795853023360673

    It is related to Bessel functions.
    与贝塞尔函数相关。

    >>> n = 1
    >>> x = np.linspace(0, 1, 5)
    >>> sc.jv(n, x)
    array([0.        , 0.12402598, 0.24226846, 0.3492436 , 0.44005059])
    >>> (0.5 * x)**n / sc.factorial(n) * sc.hyp0f1(n + 1, -0.25 * x**2)
    array([0.        , 0.12402598, 0.24226846, 0.3492436 , 0.44005059])
    """
)

# 为 `hyp1f1` 函数添加新的文档字符串和文档说明
add_newdoc("hyp1f1",
    r"""
    hyp1f1(a, b, x, out=None)

    Confluent hypergeometric function 1F1.

    The confluent hypergeometric function is defined by the series

    .. math::

       {}_1F_1(a; b; x) = \sum_{k = 0}^\infty \frac{(a)_k}{(b)_k k!} x^k.

    See [dlmf]_ for more details. Here :math:`(\cdot)_k` is the
    Pochhammer symbol; see `poch`.

    Parameters
    ----------
    a, b : array_like
        Real parameters
        实数参数 `a` 和 `b`
    x : array_like
        Real or complex argument
        实数或复数参数 `x`
    out : ndarray, optional
        Optional output array for the function results
        可选的输出数组用于存放函数结果

    Returns
    -------
    scalar or ndarray
        Values of the confluent hypergeometric function
        共轭超几何函数的值，可以是标量或数组形式

    See Also
    --------
    hyperu : another confluent hypergeometric function
    hyp0f1 : confluent hypergeometric limit function
    hyp2f1 : Gaussian hypergeometric function

    References
    ----------
    .. [dlmf] NIST Digital Library of Mathematical Functions
              https://dlmf.nist.gov/13.2#E2

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    It is one when `x` is zero:
    当 `x` 为零时，函数值为1。

    >>> sc.hyp1f1(0.5, 0.5, 0)
    1.0

    It is singular when `b` is a nonpositive integer.
    当 `b` 是非正整数时，函数是奇异的。

    >>> sc.hyp1f1(0.5, -1, 0)
    inf
    """
)
    # 导入所需的数学函数库 numpy 和 scipy.special
    import numpy as np
    import scipy.special as sc

    # 定义一个函数 sc.hyp1f1，用于计算超几何函数 1F1(a, b, x)
    # 当 a 是非正整数时，此函数代表一个多项式
    # 当 a = b 时，结果简化为指数函数的形式
    """
    It is a polynomial when `a` is a nonpositive integer.

    >>> a, b, x = -1, 0.5, np.array([1.0, 2.0, 3.0, 4.0])
    >>> sc.hyp1f1(a, b, x)
    array([-1., -3., -5., -7.])
    >>> 1 + (a / b) * x
    array([-1., -3., -5., -7.])

    It reduces to the exponential function when ``a = b``.

    >>> sc.hyp1f1(2, 2, [1, 2, 3, 4])
    array([ 2.71828183,  7.3890561 , 20.08553692, 54.59815003])
    >>> np.exp([1, 2, 3, 4])
    array([ 2.71828183,  7.3890561 , 20.08553692, 54.59815003])
    """
add_newdoc("hyperu",
    r"""
    hyperu(a, b, x, out=None)

    Confluent hypergeometric function U

    It is defined as the solution to the equation

    .. math::

       x \frac{d^2w}{dx^2} + (b - x) \frac{dw}{dx} - aw = 0

    which satisfies the property

    .. math::

       U(a, b, x) \sim x^{-a}

    as :math:`x \to \infty`. See [dlmf]_ for more details.

    Parameters
    ----------
    a, b : array_like
        Real-valued parameters
    x : array_like
        Real-valued argument
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        Values of `U`

    References
    ----------
    .. [dlmf] NIST Digital Library of Mathematics Functions
              https://dlmf.nist.gov/13.2#E6

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    It has a branch cut along the negative `x` axis.

    >>> x = np.linspace(-0.1, -10, 5)
    >>> sc.hyperu(1, 1, x)
    array([nan, nan, nan, nan, nan])

    It approaches zero as `x` goes to infinity.

    >>> x = np.array([1, 10, 100])
    >>> sc.hyperu(1, 1, x)
    array([0.59634736, 0.09156333, 0.00990194])

    It satisfies Kummer's transformation.

    >>> a, b, x = 2, 1, 1
    >>> sc.hyperu(a, b, x)
    0.1926947246463881
    >>> x**(1 - b) * sc.hyperu(a - b + 1, 2 - b, x)
    0.1926947246463881

    """)

add_newdoc("i0",
    r"""
    i0(x, out=None)

    Modified Bessel function of order 0.

    Defined as,

    .. math::
        I_0(x) = \sum_{k=0}^\infty \frac{(x^2/4)^k}{(k!)^2} = J_0(\imath x),

    where :math:`J_0` is the Bessel function of the first kind of order 0.

    Parameters
    ----------
    x : array_like
        Argument (float)
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    I : scalar or ndarray
        Value of the modified Bessel function of order 0 at `x`.

    See Also
    --------
    iv: Modified Bessel function of any order
    i0e: Exponentially scaled modified Bessel function of order 0

    Notes
    -----
    The range is partitioned into the two intervals [0, 8] and (8, infinity).
    Chebyshev polynomial expansions are employed in each interval.

    This function is a wrapper for the Cephes [1]_ routine `i0`.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    Calculate the function at one point:

    >>> from scipy.special import i0
    >>> i0(1.)
    1.2660658777520082

    Calculate at several points:

    >>> import numpy as np
    >>> i0(np.array([-2., 0., 3.5]))
    array([2.2795853 , 1.        , 7.37820343])

    Plot the function from -10 to 10.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(-10., 10., 1000)
    >>> y = i0(x)
    >>> ax.plot(x, y)
    >>> plt.show()

    """)

add_newdoc("i0e",
    """
    i0e(x, out=None)

    Exponentially scaled modified Bessel function of order 0.

    Defined as,

    .. math::
        I_{0e}(x) = e^{-|x|} \cdot I_0(x),

    where :math:`I_0` is the modified Bessel function of order 0.

    Parameters
    ----------
    x : array_like
        Argument (float)
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    I : scalar or ndarray
        Value of the exponentially scaled modified Bessel function of order 0 at `x`.

    See Also
    --------
    i0: Modified Bessel function of order 0
    iv: Modified Bessel function of any order

    Notes
    -----
    This function is a wrapper for the Cephes [1]_ routine `i0e`.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    Calculate the function at one point:

    >>> from scipy.special import i0e
    >>> i0e(1.)
    0.36787944117144233

    Calculate at several points:

    >>> import numpy as np
    >>> i0e(np.array([-2., 0., 3.5]))
    array([0.13533528, 1.        , 0.01456583])

    Plot the function from -10 to 10.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(-10., 10., 1000)
    >>> y = i0e(x)
    >>> ax.plot(x, y)
    >>> plt.show()

    """)
    Exponentially scaled modified Bessel function of order 0.

    Defined as::

        i0e(x) = exp(-abs(x)) * i0(x).

    Parameters
    ----------
    x : array_like
        Argument (float)
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    I : scalar or ndarray
        Value of the exponentially scaled modified Bessel function of order 0
        at `x`.

    See Also
    --------
    iv: Modified Bessel function of the first kind
    i0: Modified Bessel function of order 0

    Notes
    -----
    The range is partitioned into the two intervals [0, 8] and (8, infinity).
    Chebyshev polynomial expansions are employed in each interval. The
    polynomial expansions used are the same as those in `i0`, but
    they are not multiplied by the dominant exponential factor.

    This function is a wrapper for the Cephes [1]_ routine `i0e`. `i0e`
    is useful for large arguments `x`: for these, `i0` quickly overflows.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    In the following example `i0` returns infinity whereas `i0e` still returns
    a finite number.

    >>> from scipy.special import i0, i0e
    >>> i0(1000.), i0e(1000.)
    (inf, 0.012617240455891257)

    Calculate the function at several points by providing a NumPy array or
    list for `x`:

    >>> import numpy as np
    >>> i0e(np.array([-2., 0., 3.]))
    array([0.30850832, 1.        , 0.24300035])

    Plot the function from -10 to 10.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(-10., 10., 1000)
    >>> y = i0e(x)
    >>> ax.plot(x, y)
    >>> plt.show()
# 添加新的文档字符串到模块 "i1" 中，描述修改的贝塞尔函数。
add_newdoc("i1",
    r"""
    i1(x, out=None)

    Modified Bessel function of order 1.

    Defined as,

    .. math::
        I_1(x) = \frac{1}{2}x \sum_{k=0}^\infty \frac{(x^2/4)^k}{k! (k + 1)!}
               = -\imath J_1(\imath x),

    where :math:`J_1` is the Bessel function of the first kind of order 1.

    Parameters
    ----------
    x : array_like
        Argument (float)
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    I : scalar or ndarray
        Value of the modified Bessel function of order 1 at `x`.

    See Also
    --------
    iv: Modified Bessel function of the first kind
    i1e: Exponentially scaled modified Bessel function of order 1

    Notes
    -----
    The range is partitioned into the two intervals [0, 8] and (8, infinity).
    Chebyshev polynomial expansions are employed in each interval.

    This function is a wrapper for the Cephes [1]_ routine `i1`.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    Calculate the function at one point:

    >>> from scipy.special import i1
    >>> i1(1.)
    0.5651591039924851

    Calculate the function at several points:

    >>> import numpy as np
    >>> i1(np.array([-2., 0., 6.]))
    array([-1.59063685,  0.        , 61.34193678])

    Plot the function between -10 and 10.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(-10., 10., 1000)
    >>> y = i1(x)
    >>> ax.plot(x, y)
    >>> plt.show()

    """)

# 添加新的文档字符串到模块 "i1e" 中，描述指数尺度下的修改贝塞尔函数。
add_newdoc("i1e",
    """
    i1e(x, out=None)

    Exponentially scaled modified Bessel function of order 1.

    Defined as::

        i1e(x) = exp(-abs(x)) * i1(x)

    Parameters
    ----------
    x : array_like
        Argument (float)
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    I : scalar or ndarray
        Value of the exponentially scaled modified Bessel function of order 1
        at `x`.

    See Also
    --------
    iv: Modified Bessel function of the first kind
    i1: Modified Bessel function of order 1

    Notes
    -----
    The range is partitioned into the two intervals [0, 8] and (8, infinity).
    Chebyshev polynomial expansions are employed in each interval. The
    polynomial expansions used are the same as those in `i1`, but
    they are not multiplied by the dominant exponential factor.

    This function is a wrapper for the Cephes [1]_ routine `i1e`. `i1e`
    is useful for large arguments `x`: for these, `i1` quickly overflows.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    In the following example `i1` returns infinity whereas `i1e` still returns
    a finite number.

    >>> from scipy.special import i1, i1e
    >>> i1(1000.), i1e(1000.)
    (inf, 0.01261093025692863)
    # 定义一个元组，包含两个元素。这可能是一个函数返回的结果或者某种值的表示方式，具体含义需结合上下文来看。

    Calculate the function at several points by providing a NumPy array or
    list for `x`:
    # 使用提供的 NumPy 数组或列表 `x` 计算函数在多个点上的值：
    
    >>> import numpy as np
    >>> i1e(np.array([-2., 0., 6.]))
    array([-0.21526929,  0.        ,  0.15205146])
    # 示例：导入 NumPy 库，使用 `i1e` 函数计算在数组 `[-2., 0., 6.]` 上的函数值，返回一个 NumPy 数组。

    Plot the function between -10 and 10.
    # 绘制函数在区间 [-10, 10] 上的图像。

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(-10., 10., 1000)
    >>> y = i1e(x)
    >>> ax.plot(x, y)
    >>> plt.show()
    # 示例：导入 matplotlib 库，创建一个图形和坐标系，生成包含 1000 个点的从 -10 到 10 的等间距数组 `x`，
    # 计算函数在 `x` 上的值 `y`，然后绘制 `x` 和 `y` 的曲线图并显示出来。
# 添加新的文档字符串到 `_igam_fac` 函数
add_newdoc("_igam_fac",
    """
    Internal function, do not use.
    """)

# 添加新的文档字符串到 `iv` 函数
add_newdoc("iv",
    r"""
    iv(v, z, out=None)

    Modified Bessel function of the first kind of real order.

    Parameters
    ----------
    v : array_like
        Order. If `z` is of real type and negative, `v` must be integer
        valued.
    z : array_like of float or complex
        Argument.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        Values of the modified Bessel function.

    See Also
    --------
    ive : This function with leading exponential behavior stripped off.
    i0 : Faster version of this function for order 0.
    i1 : Faster version of this function for order 1.

    Notes
    -----
    For real `z` and :math:`v \in [-50, 50]`, the evaluation is carried out
    using Temme's method [1]_.  For larger orders, uniform asymptotic
    expansions are applied.

    For complex `z` and positive `v`, the AMOS [2]_ `zbesi` routine is
    called. It uses a power series for small `z`, the asymptotic expansion
    for large `abs(z)`, the Miller algorithm normalized by the Wronskian
    and a Neumann series for intermediate magnitudes, and the uniform
    asymptotic expansions for :math:`I_v(z)` and :math:`J_v(z)` for large
    orders. Backward recurrence is used to generate sequences or reduce
    orders when necessary.

    The calculations above are done in the right half plane and continued
    into the left half plane by the formula,

    .. math:: I_v(z \exp(\pm\imath\pi)) = \exp(\pm\pi v) I_v(z)

    (valid when the real part of `z` is positive).  For negative `v`, the
    formula

    .. math:: I_{-v}(z) = I_v(z) + \frac{2}{\pi} \sin(\pi v) K_v(z)

    is used, where :math:`K_v(z)` is the modified Bessel function of the
    second kind, evaluated using the AMOS routine `zbesk`.

    References
    ----------
    .. [1] Temme, Journal of Computational Physics, vol 21, 343 (1976)
    .. [2] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/

    Examples
    --------
    Evaluate the function of order 0 at one point.

    >>> from scipy.special import iv
    >>> iv(0, 1.)
    1.2660658777520084

    Evaluate the function at one point for different orders.

    >>> iv(0, 1.), iv(1, 1.), iv(1.5, 1.)
    (1.2660658777520084, 0.565159103992485, 0.2935253263474798)

    The evaluation for different orders can be carried out in one call by
    providing a list or NumPy array as argument for the `v` parameter:

    >>> iv([0, 1, 1.5], 1.)
    array([1.26606588, 0.5651591 , 0.29352533])

    Evaluate the function at several points for order 0 by providing an
    array for `z`.

    >>> import numpy as np
    >>> points = np.array([-2., 0., 3.])
    >>> iv(0, points)
    array([2.2795853 , 1.        , 4.88079259])
    # 导入numpy库，并使用“np”作为别名
    import numpy as np
    
    # 导入matplotlib.pyplot，并使用“plt”作为别名
    import matplotlib.pyplot as plt
    
    # 创建一个Figure对象和一个包含单个Axes对象的元组
    fig, ax = plt.subplots()
    
    # 生成一个包含1000个均匀分布在-5到5之间的数值的数组
    x = np.linspace(-5., 5., 1000)
    
    # 循环遍历0到3的整数，绘制不同阶数的贝塞尔函数曲线
    for i in range(4):
        # 在Axes对象上绘制第i阶的贝塞尔函数曲线，使用f-string格式化标签
        ax.plot(x, iv(i, x), label=f'$I_{i!r}$')
    
    # 在图形上添加图例，显示各阶贝塞尔函数的标签
    ax.legend()
    
    # 显示绘制的图形
    plt.show()
# 添加新文档或修改现有文档的帮助信息
add_newdoc("ive",
    r"""
    ive(v, z, out=None)

    Exponentially scaled modified Bessel function of the first kind.

    Defined as::

        ive(v, z) = iv(v, z) * exp(-abs(z.real))

    For imaginary numbers without a real part, returns the unscaled
    Bessel function of the first kind `iv`.

    Parameters
    ----------
    v : array_like of float
        Order.
    z : array_like of float or complex
        Argument.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        Values of the exponentially scaled modified Bessel function.

    See Also
    --------
    iv: Modified Bessel function of the first kind
    i0e: Faster implementation of this function for order 0
    i1e: Faster implementation of this function for order 1

    Notes
    -----
    For positive `v`, the AMOS [1]_ `zbesi` routine is called. It uses a
    power series for small `z`, the asymptotic expansion for large
    `abs(z)`, the Miller algorithm normalized by the Wronskian and a
    Neumann series for intermediate magnitudes, and the uniform asymptotic
    expansions for :math:`I_v(z)` and :math:`J_v(z)` for large orders.
    Backward recurrence is used to generate sequences or reduce orders when
    necessary.

    The calculations above are done in the right half plane and continued
    into the left half plane by the formula,

    .. math:: I_v(z \exp(\pm\imath\pi)) = \exp(\pm\pi v) I_v(z)

    (valid when the real part of `z` is positive).  For negative `v`, the
    formula

    .. math:: I_{-v}(z) = I_v(z) + \frac{2}{\pi} \sin(\pi v) K_v(z)

    is used, where :math:`K_v(z)` is the modified Bessel function of the
    second kind, evaluated using the AMOS routine `zbesk`.

    `ive` is useful for large arguments `z`: for these, `iv` easily overflows,
    while `ive` does not due to the exponential scaling.

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/

    Examples
    --------
    In the following example `iv` returns infinity whereas `ive` still returns
    a finite number.

    >>> from scipy.special import iv, ive
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> iv(3, 1000.), ive(3, 1000.)
    (inf, 0.01256056218254712)

    Evaluate the function at one point for different orders by
    providing a list or NumPy array as argument for the `v` parameter:

    >>> ive([0, 1, 1.5], 1.)
    array([0.46575961, 0.20791042, 0.10798193])

    Evaluate the function at several points for order 0 by providing an
    array for `z`.

    >>> points = np.array([-2., 0., 3.])
    >>> ive(0, points)
    array([0.30850832, 1.        , 0.24300035])

    Evaluate the function at several points for different orders by
    providing arrays for both `v` for `z`. Both arrays have to be
    # 创建一个图形对象和轴对象，用于绘制图形
    fig, ax = plt.subplots()
    
    # 生成一个包含从 -5 到 5 的 1000 个均匀间隔的数值作为 x 轴数据
    x = np.linspace(-5., 5., 1000)
    
    # 循环绘制阶数从 0 到 3 的函数曲线
    for i in range(4):
        # 绘制第 i 阶的函数曲线，使用 LaTeX 表示标签
        ax.plot(x, ive(i, x), label=fr'$I_{i!r}(z)\cdot e^{{-|z|}}$')
    
    # 添加图例
    ax.legend()
    
    # 设置 x 轴标签
    ax.set_xlabel(r"$z$")
    
    # 显示绘制的图形
    plt.show()
# 添加新文档到特定函数 "j0"
add_newdoc("j0",
    r"""
    j0(x, out=None)

    Bessel function of the first kind of order 0.

    Parameters
    ----------
    x : array_like
        Argument (float).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    J : scalar or ndarray
        Value of the Bessel function of the first kind of order 0 at `x`.

    See Also
    --------
    jv : Bessel function of real order and complex argument.
    spherical_jn : spherical Bessel functions.

    Notes
    -----
    The domain is divided into the intervals [0, 5] and (5, infinity). In the
    first interval the following rational approximation is used:

    .. math::

        J_0(x) \approx (w - r_1^2)(w - r_2^2) \frac{P_3(w)}{Q_8(w)},

    where :math:`w = x^2` and :math:`r_1`, :math:`r_2` are the zeros of
    :math:`J_0`, and :math:`P_3` and :math:`Q_8` are polynomials of degrees 3
    and 8, respectively.

    In the second interval, the Hankel asymptotic expansion is employed with
    two rational functions of degree 6/6 and 7/7.

    This function is a wrapper for the Cephes [1]_ routine `j0`.
    It should not be confused with the spherical Bessel functions (see
    `spherical_jn`).

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    Calculate the function at one point:

    >>> from scipy.special import j0
    >>> j0(1.)
    0.7651976865579665

    Calculate the function at several points:

    >>> import numpy as np
    >>> j0(np.array([-2., 0., 4.]))
    array([ 0.22389078,  1.        , -0.39714981])

    Plot the function from -20 to 20.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(-20., 20., 1000)
    >>> y = j0(x)
    >>> ax.plot(x, y)
    >>> plt.show()

    """)

# 添加新文档到特定函数 "j1"
add_newdoc("j1",
    """
    j1(x, out=None)

    Bessel function of the first kind of order 1.

    Parameters
    ----------
    x : array_like
        Argument (float).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    J : scalar or ndarray
        Value of the Bessel function of the first kind of order 1 at `x`.

    See Also
    --------
    jv: Bessel function of the first kind
    spherical_jn: spherical Bessel functions.

    Notes
    -----
    The domain is divided into the intervals [0, 8] and (8, infinity). In the
    first interval a 24 term Chebyshev expansion is used. In the second, the
    asymptotic trigonometric representation is employed using two rational
    functions of degree 5/5.

    This function is a wrapper for the Cephes [1]_ routine `j1`.
    It should not be confused with the spherical Bessel functions (see
    `spherical_jn`).

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------

    """)
    # 导入需要的库函数 scipy.special.j1 来计算第一类贝塞尔函数
    >>> from scipy.special import j1
    # 计算第一类贝塞尔函数在单个点上的值，例如 j1(1.0)
    >>> j1(1.)
    0.44005058574493355

    # 导入 numpy 库，并将 j1 函数应用在一个数组上，计算多个点的第一类贝塞尔函数值
    >>> import numpy as np
    >>> j1(np.array([-2., 0., 4.]))
    array([-0.57672481,  0.        , -0.06604333])

    # 导入 matplotlib.pyplot 库用于绘图，并创建一个新的图形和坐标轴对象
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    # 生成一个从 -20 到 20 的等间隔数组作为 x 轴数据
    >>> x = np.linspace(-20., 20., 1000)
    # 计算在上述 x 值处的第一类贝塞尔函数值作为 y 轴数据
    >>> y = j1(x)
    # 在坐标轴上绘制 x 和 y 数据的图形
    >>> ax.plot(x, y)
    # 显示图形
    >>> plt.show()
# 添加新的文档字符串到模块 "jn" 中，描述了 Bessel 函数的基本信息和使用方法
add_newdoc("jn",
    """
    jn(n, x, out=None)

    Bessel function of the first kind of integer order and real argument.

    Parameters
    ----------
    n : array_like
        order of the Bessel function
    x : array_like
        argument of the Bessel function
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        The value of the bessel function

    See Also
    --------
    jv
    spherical_jn : spherical Bessel functions.

    Notes
    -----
    `jn` is an alias of `jv`.
    Not to be confused with the spherical Bessel functions (see
    `spherical_jn`).

    """)

# 添加新的文档字符串到模块 "jv" 中，描述了 Bessel 函数的详细信息和使用方法
add_newdoc("jv",
    r"""
    jv(v, z, out=None)

    Bessel function of the first kind of real order and complex argument.

    Parameters
    ----------
    v : array_like
        Order (float).
    z : array_like
        Argument (float or complex).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    J : scalar or ndarray
        Value of the Bessel function, :math:`J_v(z)`.

    See Also
    --------
    jve : :math:`J_v` with leading exponential behavior stripped off.
    spherical_jn : spherical Bessel functions.
    j0 : faster version of this function for order 0.
    j1 : faster version of this function for order 1.

    Notes
    -----
    For positive `v` values, the computation is carried out using the AMOS
    [1]_ `zbesj` routine, which exploits the connection to the modified
    Bessel function :math:`I_v`,

    .. math::
        J_v(z) = \exp(v\pi\imath/2) I_v(-\imath z)\qquad (\Im z > 0)

        J_v(z) = \exp(-v\pi\imath/2) I_v(\imath z)\qquad (\Im z < 0)

    For negative `v` values the formula,

    .. math:: J_{-v}(z) = J_v(z) \cos(\pi v) - Y_v(z) \sin(\pi v)

    is used, where :math:`Y_v(z)` is the Bessel function of the second
    kind, computed using the AMOS routine `zbesy`.  Note that the second
    term is exactly zero for integer `v`; to improve accuracy the second
    term is explicitly omitted for `v` values such that `v = floor(v)`.

    Not to be confused with the spherical Bessel functions (see `spherical_jn`).

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/

    Examples
    --------
    Evaluate the function of order 0 at one point.

    >>> from scipy.special import jv
    >>> jv(0, 1.)
    0.7651976865579666

    Evaluate the function at one point for different orders.

    >>> jv(0, 1.), jv(1, 1.), jv(1.5, 1.)
    (0.7651976865579666, 0.44005058574493355, 0.24029783912342725)

    The evaluation for different orders can be carried out in one call by
    providing a list or NumPy array as argument for the `v` parameter:

    >>> jv([0, 1, 1.5], 1.)
    array([0.76519769, 0.44005059, 0.24029784])
    Evaluate the function at several points for order 0 by providing an
    array for `z`.
    >>> import numpy as np
    导入NumPy库，用于处理数组和数值计算
    >>> points = np.array([-2., 0., 3.])
    创建一个包含几个点的NumPy数组，这些点用于计算0阶贝塞尔函数的值
    >>> jv(0, points)
    调用jv函数，计算阶数为0时在points数组中各点的贝塞尔函数值，并返回一个数组
    array([ 0.22389078,  1.        , -0.26005195])
    
    If `z` is an array, the order parameter `v` must be broadcastable to
    the correct shape if different orders shall be computed in one call.
    如果`z`是一个数组，则阶数参数`v`必须能够广播到正确的形状，以便在一次调用中计算不同的阶数。
    To calculate the orders 0 and 1 for an 1D array:
    为了计算一个一维数组中的阶数0和1：
    
    >>> orders = np.array([[0], [1]])
    创建一个NumPy数组，包含阶数0和1的二维索引，以便在一次调用中计算多个不同阶数的贝塞尔函数值
    >>> orders.shape
    返回orders数组的形状
    (2, 1)
    
    >>> jv(orders, points)
    调用jv函数，计算在points数组中各点处阶数为0和1的贝塞尔函数值，并返回一个二维数组
    array([[ 0.22389078,  1.        , -0.26005195],
           [-0.57672481,  0.        ,  0.33905896]])
    
    Plot the functions of order 0 to 3 from -10 to 10.
    绘制阶数从0到3的贝塞尔函数在区间[-10, 10]上的图像。
    
    >>> import matplotlib.pyplot as plt
    导入matplotlib.pyplot库，用于绘图
    >>> fig, ax = plt.subplots()
    创建一个新的Figure对象和一个Axes对象
    >>> x = np.linspace(-10., 10., 1000)
    创建一个包含1000个点的NumPy数组，用于绘制x轴的数值范围
    >>> for i in range(4):
    循环迭代，依次计算并绘制阶数从0到3的贝塞尔函数曲线
    ...     ax.plot(x, jv(i, x), label=f'$J_{i!r}$')
    在Axes对象上绘制贝塞尔函数曲线，标签显示贝塞尔函数的阶数
    >>> ax.legend()
    显示图例
    >>> plt.show()
    显示图形窗口，并绘制贝塞尔函数曲线
# 添加新的文档字符串到名为 "jve" 的函数
add_newdoc("jve",
    r"""
    jve(v, z, out=None)

    Exponentially scaled Bessel function of the first kind of order `v`.

    Defined as::

        jve(v, z) = jv(v, z) * exp(-abs(z.imag))

    Parameters
    ----------
    v : array_like
        Order (float).
    z : array_like
        Argument (float or complex).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    J : scalar or ndarray
        Value of the exponentially scaled Bessel function.

    See Also
    --------
    jv: Unscaled Bessel function of the first kind

    Notes
    -----
    For positive `v` values, the computation is carried out using the AMOS
    [1]_ `zbesj` routine, which exploits the connection to the modified
    Bessel function :math:`I_v`,

    .. math::
        J_v(z) = \exp(v\pi\imath/2) I_v(-\imath z)\qquad (\Im z > 0)

        J_v(z) = \exp(-v\pi\imath/2) I_v(\imath z)\qquad (\Im z < 0)

    For negative `v` values the formula,

    .. math:: J_{-v}(z) = J_v(z) \cos(\pi v) - Y_v(z) \sin(\pi v)

    is used, where :math:`Y_v(z)` is the Bessel function of the second
    kind, computed using the AMOS routine `zbesy`.  Note that the second
    term is exactly zero for integer `v`; to improve accuracy the second
    term is explicitly omitted for `v` values such that `v = floor(v)`.

    Exponentially scaled Bessel functions are useful for large arguments `z`:
    for these, the unscaled Bessel functions can easily under-or overflow.

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/

    Examples
    --------
    Compare the output of `jv` and `jve` for large complex arguments for `z`
    by computing their values for order ``v=1`` at ``z=1000j``. We see that
    `jv` overflows but `jve` returns a finite number:

    >>> import numpy as np
    >>> from scipy.special import jv, jve
    >>> v = 1
    >>> z = 1000j
    >>> jv(v, z), jve(v, z)
    ((inf+infj), (7.721967686709077e-19+0.012610930256928629j))

    For real arguments for `z`, `jve` returns the same as `jv`.

    >>> v, z = 1, 1000
    >>> jv(v, z), jve(v, z)
    (0.004728311907089523, 0.004728311907089523)

    The function can be evaluated for several orders at the same time by
    providing a list or NumPy array for `v`:

    >>> jve([1, 3, 5], 1j)
    array([1.27304208e-17+2.07910415e-01j, -4.99352086e-19-8.15530777e-03j,
           6.11480940e-21+9.98657141e-05j])

    In the same way, the function can be evaluated at several points in one
    call by providing a list or NumPy array for `z`:

    >>> jve(1, np.array([1j, 2j, 3j]))
    array([1.27308412e-17+0.20791042j, 1.31814423e-17+0.21526929j,
           1.20521602e-17+0.19682671j])

    It is also possible to evaluate several orders at several points
    at the same time by providing arrays for `v` and `z` with
    """
    compatible shapes for broadcasting. Compute `jve` for two different orders
    `v` and three points `z` resulting in a 2x3 array.

    >>> v = np.array([[1], [3]])  # 创建一个2x1的NumPy数组v
    >>> z = np.array([1j, 2j, 3j])  # 创建一个包含三个复数的NumPy数组z
    >>> v.shape, z.shape  # 打印v和z的形状
    ((2, 1), (3,))  # v的形状为(2, 1)，z的形状为(3,)

    >>> jve(v, z)  # 调用函数jve计算v和z的交叉向量表达
    array([[1.27304208e-17+0.20791042j,  1.31810070e-17+0.21526929j,
            1.20517622e-17+0.19682671j],  # 第一行结果
           [-4.99352086e-19-0.00815531j, -1.76289571e-18-0.02879122j,
            -2.92578784e-18-0.04778332j]])  # 第二行结果
# 添加新文档到 "k0" 函数
add_newdoc("k0",
    r"""
    k0(x, out=None)

    Modified Bessel function of the second kind of order 0, :math:`K_0`.

    This function is also sometimes referred to as the modified Bessel
    function of the third kind of order 0.

    Parameters
    ----------
    x : array_like
        Argument (float).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    K : scalar or ndarray
        Value of the modified Bessel function :math:`K_0` at `x`.

    See Also
    --------
    kv: Modified Bessel function of the second kind of any order
    k0e: Exponentially scaled modified Bessel function of the second kind

    Notes
    -----
    The range is partitioned into the two intervals [0, 2] and (2, infinity).
    Chebyshev polynomial expansions are employed in each interval.

    This function is a wrapper for the Cephes [1]_ routine `k0`.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    Calculate the function at one point:

    >>> from scipy.special import k0
    >>> k0(1.)
    0.42102443824070823

    Calculate the function at several points:

    >>> import numpy as np
    >>> k0(np.array([0.5, 2., 3.]))
    array([0.92441907, 0.11389387, 0.0347395 ])

    Plot the function from 0 to 10.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(0., 10., 1000)
    >>> y = k0(x)
    >>> ax.plot(x, y)
    >>> plt.show()

    """)

# 添加新文档到 "k0e" 函数
add_newdoc("k0e",
    """
    k0e(x, out=None)

    Exponentially scaled modified Bessel function K of order 0

    Defined as::

        k0e(x) = exp(x) * k0(x).

    Parameters
    ----------
    x : array_like
        Argument (float)
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    K : scalar or ndarray
        Value of the exponentially scaled modified Bessel function K of order
        0 at `x`.

    See Also
    --------
    kv: Modified Bessel function of the second kind of any order
    k0: Modified Bessel function of the second kind

    Notes
    -----
    The range is partitioned into the two intervals [0, 2] and (2, infinity).
    Chebyshev polynomial expansions are employed in each interval.

    This function is a wrapper for the Cephes [1]_ routine `k0e`. `k0e` is
    useful for large arguments: for these, `k0` easily underflows.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    In the following example `k0` returns 0 whereas `k0e` still returns a
    useful finite number:

    >>> from scipy.special import k0, k0e
    >>> k0(1000.), k0e(1000)
    (0., 0.03962832160075422)

    Calculate the function at several points by providing a NumPy array or
    list for `x`:

    >>> import numpy as np
    >>> k0e(np.array([0.5, 2., 3.]))

    """)
    array([1.52410939, 0.84156822, 0.6977616 ])

    Plot the function from 0 to 10.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()  # 创建一个新的图形和坐标轴对象
    >>> x = np.linspace(0., 10., 1000)  # 在区间[0, 10]上生成1000个均匀分布的点作为x轴数据
    >>> y = k0e(x)  # 计算函数 k0e 在x轴数据上的取值，得到y轴数据
    >>> ax.plot(x, y)  # 在坐标轴上绘制(x, y)数据的折线图
    >>> plt.show()  # 显示绘制的图形
# 添加新的文档字符串到模块 "k1" 中，描述了修改的贝塞尔函数第二类一阶，即 :math:`K_1(x)`。

add_newdoc("k1",
    """
    k1(x, out=None)

    Modified Bessel function of the second kind of order 1, :math:`K_1(x)`.

    Parameters
    ----------
    x : array_like
        Argument (float)
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    K : scalar or ndarray
        Value of the modified Bessel function K of order 1 at `x`.

    See Also
    --------
    kv: Modified Bessel function of the second kind of any order
    k1e: Exponentially scaled modified Bessel function K of order 1

    Notes
    -----
    The range is partitioned into the two intervals [0, 2] and (2, infinity).
    Chebyshev polynomial expansions are employed in each interval.

    This function is a wrapper for the Cephes [1]_ routine `k1`.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    Calculate the function at one point:

    >>> from scipy.special import k1
    >>> k1(1.)
    0.6019072301972346

    Calculate the function at several points:

    >>> import numpy as np
    >>> k1(np.array([0.5, 2., 3.]))
    array([1.65644112, 0.13986588, 0.04015643])

    Plot the function from 0 to 10.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(0., 10., 1000)
    >>> y = k1(x)
    >>> ax.plot(x, y)
    >>> plt.show()

    """)

# 添加新的文档字符串到模块 "k1e" 中，描述了指数尺度修改的贝塞尔函数第二类一阶，即 `k1e(x) = exp(x) * k1(x)`。

add_newdoc("k1e",
    """
    k1e(x, out=None)

    Exponentially scaled modified Bessel function K of order 1

    Defined as::

        k1e(x) = exp(x) * k1(x)

    Parameters
    ----------
    x : array_like
        Argument (float)
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    K : scalar or ndarray
        Value of the exponentially scaled modified Bessel function K of order
        1 at `x`.

    See Also
    --------
    kv: Modified Bessel function of the second kind of any order
    k1: Modified Bessel function of the second kind of order 1

    Notes
    -----
    The range is partitioned into the two intervals [0, 2] and (2, infinity).
    Chebyshev polynomial expansions are employed in each interval.

    This function is a wrapper for the Cephes [1]_ routine `k1e`.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    In the following example `k1` returns 0 whereas `k1e` still returns a
    useful floating point number.

    >>> from scipy.special import k1, k1e
    >>> k1(1000.), k1e(1000.)
    (0., 0.03964813081296021)

    Calculate the function at several points by providing a NumPy array or
    list for `x`:

    >>> import numpy as np
    >>> k1e(np.array([0.5, 2., 3.]))
    array([2.73100971, 1.03347685, 0.80656348])

    Plot the function from 0 to 10.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()

    """)
    # 使用 numpy.linspace 函数生成从 0 到 10 的等间隔的 1000 个数的数组
    >>> x = np.linspace(0., 10., 1000)
    # 调用 k1e 函数计算数组 x 对应的 y 值
    >>> y = k1e(x)
    # 在图形对象 ax 上绘制 x 和 y 的图像
    >>> ax.plot(x, y)
    # 显示图形
    >>> plt.show()
`
# 添加文档字符串，描述 kelvin 函数，函数功能是处理复数形式的 Kelvin 函数
add_newdoc("kelvin",
    """
    kelvin(x, out=None)

    Kelvin functions as complex numbers

    Parameters
    ----------
    x : array_like
        Argument
    out : tuple of ndarray, optional
        Optional output arrays for the function values

    Returns
    -------
    Be, Ke, Bep, Kep : 4-tuple of scalar or ndarray
        The tuple (Be, Ke, Bep, Kep) contains complex numbers
        representing the real and imaginary Kelvin functions and their
        derivatives evaluated at `x`.  For example, kelvin(x)[0].real =
        ber x and kelvin(x)[0].imag = bei x with similar relationships
        for ker and kei.
    """)
    
# 添加文档字符串，描述 ker 函数，定义了 Kelvin 函数 ker 的公式
add_newdoc("ker",
    r"""
    ker(x, out=None)

    Kelvin function ker.

    Defined as

    .. math::

        \mathrm{ker}(x) = \Re[K_0(x e^{\pi i / 4})]

    Where :math:`K_0` is the modified Bessel function of the second
    kind (see `kv`). See [dlmf]_ for more details.

    Parameters
    ----------
    x : array_like
        Real argument.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Values of the Kelvin function.

    See Also
    --------
    kei : the corresponding imaginary part
    kerp : the derivative of ker
    kv : modified Bessel function of the second kind

    References
    ----------
    .. [dlmf] NIST, Digital Library of Mathematical Functions,
        https://dlmf.nist.gov/10.61

    Examples
    --------
    It can be expressed using the modified Bessel function of the
    second kind.

    >>> import numpy as np
    >>> import scipy.special as sc
    >>> x = np.array([1.0, 2.0, 3.0, 4.0])
    >>> sc.kv(0, x * np.exp(np.pi * 1j / 4)).real
    array([ 0.28670621, -0.04166451, -0.06702923, -0.03617885])
    >>> sc.ker(x)
    array([ 0.28670621, -0.04166451, -0.06702923, -0.03617885])
    """)
    
# 添加文档字符串，描述 kerp 函数，表示 Kelvin 函数 ker 的导数
add_newdoc("kerp",
    r"""
    kerp(x, out=None)

    Derivative of the Kelvin function ker.

    Parameters
    ----------
    x : array_like
        Real argument.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Values of the derivative of ker.

    See Also
    --------
    ker

    References
    ----------
    .. [dlmf] NIST, Digital Library of Mathematical Functions,
        https://dlmf.nist.gov/10#PT5
    """)
    
# 添加文档字符串，描述 kl_div 函数，用于计算 Kullback-Leibler 散度
add_newdoc("kl_div",
    r"""
    kl_div(x, y, out=None)

    Elementwise function for computing Kullback-Leibler divergence.

    .. math::

        \mathrm{kl\_div}(x, y) =
          \begin{cases}
            x \log(x / y) - x + y & x > 0, y > 0 \\
            y & x = 0, y \ge 0 \\
            \infty & \text{otherwise}
          \end{cases}

    Parameters
    ----------
    x, y : array_like
        Real arguments
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of the Kullback-Liebler divergence.
    """)
    See Also
    --------
    entr, rel_entr, scipy.stats.entropy

    Notes
    -----
    .. versionadded:: 0.15.0

    This function is non-negative and is jointly convex in `x` and `y`.

    The origin of this function is in convex programming; see [1]_ for
    details. This is why the function contains the extra :math:`-x
    + y` terms over what might be expected from the Kullback-Leibler
    divergence. For a version of the function without the extra terms,
    see `rel_entr`.

    References
    ----------
    .. [1] Boyd, Stephen and Lieven Vandenberghe. *Convex optimization*.
           Cambridge University Press, 2004.
           :doi:`https://doi.org/10.1017/CBO9780511804441`


注释：
这部分代码是一个文档字符串（docstring），用于描述函数或方法的相关信息、使用说明、版本信息以及参考文献。在Python中，文档字符串通常使用三重引号包裹起来，可以包含函数的详细说明、版本历史、相关参考等内容。
# 将文档字符串添加到名为 "kn" 的函数中，该函数返回修正贝塞尔函数第二类的整数阶 `n` 的值
add_newdoc("kn",
    r"""
    kn(n, x, out=None)

    Modified Bessel function of the second kind of integer order `n`

    Returns the modified Bessel function of the second kind for integer order
    `n` at real `z`.

    These are also sometimes called functions of the third kind, Basset
    functions, or Macdonald functions.

    Parameters
    ----------
    n : array_like of int
        Order of Bessel functions (floats will truncate with a warning)
    x : array_like of float
        Argument at which to evaluate the Bessel functions
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Value of the Modified Bessel function of the second kind,
        :math:`K_n(x)`.

    See Also
    --------
    kv : Same function, but accepts real order and complex argument
    kvp : Derivative of this function

    Notes
    -----
    Wrapper for AMOS [1]_ routine `zbesk`.  For a discussion of the
    algorithm used, see [2]_ and the references therein.

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/
    .. [2] Donald E. Amos, "Algorithm 644: A portable package for Bessel
           functions of a complex argument and nonnegative order", ACM
           TOMS Vol. 12 Issue 3, Sept. 1986, p. 265

    Examples
    --------
    Plot the function of several orders for real input:

    >>> import numpy as np
    >>> from scipy.special import kn
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0, 5, 1000)
    >>> for N in range(6):
    ...     plt.plot(x, kn(N, x), label='$K_{}(x)$'.format(N))
    >>> plt.ylim(0, 10)
    >>> plt.legend()
    >>> plt.title(r'Modified Bessel function of the second kind $K_n(x)$')
    >>> plt.show()

    Calculate for a single value at multiple orders:

    >>> kn([4, 5, 6], 1)
    array([   44.23241585,   360.9605896 ,  3653.83831186])
    """)

# 将文档字符串添加到名为 "kolmogi" 的函数中，该函数返回 Kolmogorov 分布的逆生存函数值
add_newdoc("kolmogi",
    """
    kolmogi(p, out=None)

    Inverse Survival Function of Kolmogorov distribution

    It is the inverse function to `kolmogorov`.
    Returns y such that ``kolmogorov(y) == p``.

    Parameters
    ----------
    p : float array_like
        Probability
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        The value(s) of kolmogi(p)

    See Also
    --------
    kolmogorov : The Survival Function for the distribution
    scipy.stats.kstwobign : Provides the functionality as a continuous distribution
    smirnov, smirnovi : Functions for the one-sided distribution

    Notes
    -----
    `kolmogorov` is used by `stats.kstest` in the application of the
    Kolmogorov-Smirnov Goodness of Fit test. For historical reasons this
    function is exposed in `scpy.special`, but the recommended way to achieve
    """
)
    the most accurate CDF/SF/PDF/PPF/ISF computations is to use the
    `stats.kstwobign` distribution.

    Examples
    --------
    >>> from scipy.special import kolmogi
    >>> kolmogi([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    array([        inf,  1.22384787,  1.01918472,  0.82757356,  0.67644769,
            0.57117327,  0.        ])

"""
# 向 `scipy.special` 模块添加新的文档字符串，介绍 Kolmogorov 分布的补充累积分布函数（生存函数）
add_newdoc("kolmogorov",
    r"""
    kolmogorov(y, out=None)

    Kolmogorov 分布的补充累积分布函数（生存函数）。

    返回 Kolmogorov 极限分布的补充累积分布函数（随着 n 趋向无穷大，`D_n*\sqrt(n)`），
    这是一个双侧检验的概率，用于测量经验分布函数与理论分布函数之间的一致性。它等于
    `sqrt(n) * max 绝对偏差 > y` 的概率（n 趋向无穷大时的极限）。

    Parameters
    ----------
    y : float array_like
      经验分布函数（ECDF）与目标分布函数之间的绝对偏差，乘以 sqrt(n)。
    out : ndarray, optional
        可选的输出数组用于存放函数结果。

    Returns
    -------
    scalar or ndarray
        kolmogorov(y) 的值或值数组。

    See Also
    --------
    kolmogi : 分布的逆生存函数
    scipy.stats.kstwobign : 提供作为连续分布的功能
    smirnov, smirnovi : 单侧分布函数

    Notes
    -----
    `kolmogorov` 在 Kolmogorov-Smirnov 拟合优度检验中由 `stats.kstest` 使用。
    基于历史原因，该函数在 `scpy.special` 中公开，但实现最精确的CDF/SF/PDF/PPF/ISF计算的
    推荐方式是使用 `stats.kstwobign` 分布。

    Examples
    --------
    显示至少为 0、0.5 和 1.0 的间隙的概率。

    >>> import numpy as np
    >>> from scipy.special import kolmogorov
    >>> from scipy.stats import kstwobign
    >>> kolmogorov([0, 0.5, 1.0])
    array([ 1.        ,  0.96394524,  0.26999967])

    比较大小为 1000 的拉普拉斯分布样本与目标分布，正态分布（0, 1）。

    >>> from scipy.stats import norm, laplace
    >>> rng = np.random.default_rng()
    >>> n = 1000
    >>> lap01 = laplace(0, 1)
    >>> x = np.sort(lap01.rvs(n, random_state=rng))
    >>> np.mean(x), np.std(x)
    (-0.05841730131499543, 1.3968109101997568)

    构建经验分布函数（ECDF）和 Kolmogorov-Smirnov 统计量 Dn。

    >>> target = norm(0,1)  # 正态分布均值 0，标准差 1
    >>> cdfs = target.cdf(x)
    >>> ecdfs = np.arange(n+1, dtype=float)/n
    >>> gaps = np.column_stack([cdfs - ecdfs[:n], ecdfs[1:] - cdfs])
    >>> Dn = np.max(gaps)
    >>> Kn = np.sqrt(n) * Dn
    >>> print('Dn=%f, sqrt(n)*Dn=%f' % (Dn, Kn))
    Dn=0.043363, sqrt(n)*Dn=1.371265
    >>> print(chr(10).join(['对大小为 n 的 N(0, 1) 分布样本:',
    ...   ' sqrt(n)*Dn>=%f 的 Kolmogorov 概率约为 %f' %
    ...    (Kn, kolmogorov(Kn)),
    ...   ' sqrt(n)*Dn<=%f 的 Kolmogorov 概率约为 %f' %
    ...    (Kn, kstwobign.cdf(Kn))]))
    """
    For a sample of size n drawn from a N(0, 1) distribution:
     the approximate Kolmogorov probability that sqrt(n)*Dn>=1.371265 is 0.046533
     the approximate Kolmogorov probability that sqrt(n)*Dn<=1.371265 is 0.953467

    Plot the Empirical CDF against the target N(0, 1) CDF.

    >>> import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，用于绘图
    >>> plt.step(np.concatenate([[-3], x]), ecdfs, where='post', label='Empirical CDF')  # 绘制阶梯状的经验累积分布函数图，连接点为 x 和 ecdfs
    >>> x3 = np.linspace(-3, 3, 100)  # 在区间 [-3, 3] 内生成 100 个均匀间隔的点，用于绘制目标 N(0, 1) 分布的累积分布函数图
    >>> plt.plot(x3, target.cdf(x3), label='CDF for N(0, 1)')  # 绘制 N(0, 1) 分布的累积分布函数图
    >>> plt.ylim([0, 1]); plt.grid(True); plt.legend();  # 设置纵轴范围为 [0, 1]，显示网格，显示图例

    >>> # Add vertical lines marking Dn+ and Dn-
    >>> iminus, iplus = np.argmax(gaps, axis=0)  # 找出 gaps 中的最大值的索引，即 Dn- 和 Dn+ 的位置
    >>> plt.vlines([x[iminus]], ecdfs[iminus], cdfs[iminus],
    ...            color='r', linestyle='dashed', lw=4)  # 添加标记 Dn- 的垂直虚线，红色，虚线样式，线宽为 4
    >>> plt.vlines([x[iplus]], cdfs[iplus], ecdfs[iplus+1],
    ...            color='r', linestyle='dashed', lw=4)  # 添加标记 Dn+ 的垂直虚线，红色，虚线样式，线宽为 4

    >>> plt.show()  # 显示绘制的图形
# 添加新的文档字符串至 "_kolmogc" 模块
add_newdoc("_kolmogc",
    r"""
    Internal function, do not use.
    """)

# 添加新的文档字符串至 "_kolmogci" 模块
add_newdoc("_kolmogci",
    r"""
    Internal function, do not use.
    """)

# 添加新的文档字符串至 "_kolmogp" 模块
add_newdoc("_kolmogp",
    r"""
    Internal function, do not use.
    """)

# 添加新的文档字符串至 "kv" 函数
add_newdoc("kv",
    r"""
    kv(v, z, out=None)

    Modified Bessel function of the second kind of real order `v`

    Returns the modified Bessel function of the second kind for real order
    `v` at complex `z`.

    These are also sometimes called functions of the third kind, Basset
    functions, or Macdonald functions.  They are defined as those solutions
    of the modified Bessel equation for which,

    .. math::
        K_v(x) \sim \sqrt{\pi/(2x)} \exp(-x)

    as :math:`x \to \infty` [3]_.

    Parameters
    ----------
    v : array_like of float
        Order of Bessel functions
    z : array_like of complex
        Argument at which to evaluate the Bessel functions
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        The results. Note that input must be of complex type to get complex
        output, e.g. ``kv(3, -2+0j)`` instead of ``kv(3, -2)``.

    See Also
    --------
    kve : This function with leading exponential behavior stripped off.
    kvp : Derivative of this function

    Notes
    -----
    Wrapper for AMOS [1]_ routine `zbesk`.  For a discussion of the
    algorithm used, see [2]_ and the references therein.

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/
    .. [2] Donald E. Amos, "Algorithm 644: A portable package for Bessel
           functions of a complex argument and nonnegative order", ACM
           TOMS Vol. 12 Issue 3, Sept. 1986, p. 265
    .. [3] NIST Digital Library of Mathematical Functions,
           Eq. 10.25.E3. https://dlmf.nist.gov/10.25.E3

    Examples
    --------
    Plot the function of several orders for real input:

    >>> import numpy as np
    >>> from scipy.special import kv
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0, 5, 1000)
    >>> for N in np.linspace(0, 6, 5):
    ...     plt.plot(x, kv(N, x), label='$K_{{{}}}(x)$'.format(N))
    >>> plt.ylim(0, 10)
    >>> plt.legend()
    >>> plt.title(r'Modified Bessel function of the second kind $K_\nu(x)$')
    >>> plt.show()

    Calculate for a single value at multiple orders:

    >>> kv([4, 4.5, 5], 1+2j)
    array([ 0.1992+2.3892j,  2.3493+3.6j   ,  7.2827+3.8104j])

    """)

# 添加新的文档字符串至 "kve" 函数
add_newdoc("kve",
    r"""
    kve(v, z, out=None)

    Exponentially scaled modified Bessel function of the second kind.

    Returns the exponentially scaled, modified Bessel function of the
    second kind (sometimes called the third kind) for real order `v` at
    complex `z`::

        kve(v, z) = kv(v, z) * exp(z)

    Parameters
    ----------
    v : array_like of float
        Order of Bessel functions
    z : array_like of complex
        Argument at which to evaluate the Bessel functions
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        The results. Note that input must be of complex type to get complex
        output, e.g. ``kve(3, -2+0j)`` instead of ``kve(3, -2)``.

    See Also
    --------
    kv : Modified Bessel function of the second kind without exponential scaling
    kvp : Derivative of this function

    Notes
    -----
    Wrapper for AMOS [1]_ routine `zbesk`.  For a discussion of the
    algorithm used, see [2]_ and the references therein.

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/
    .. [2] Donald E. Amos, "Algorithm 644: A portable package for Bessel
           functions of a complex argument and nonnegative order", ACM
           TOMS Vol. 12 Issue 3, Sept. 1986, p. 265

    """
)
    # v : array_like of float
    # Bessel函数的阶数，可以是一个浮点数数组或类似数组

    # z : array_like of complex
    # 要求Bessel函数的复数参数，可以是一个复数数组或类似数组

    # out : ndarray, optional
    # 可选的输出数组，用于存储函数计算的结果

    # Returns
    # -------
    # scalar or ndarray
    # 第二类修正Bessel函数的指数尺度版本，可以是标量或数组

    # See Also
    # --------
    # kv : 不带指数尺度的函数版本。
    # k0e : 用于阶数为0时的更快版本。
    # k1e : 用于阶数为1时的更快版本。

    # Notes
    # -----
    # AMOS [1]_ 的 `zbesk` 程序包装器。有关所用算法的讨论，请参见 [2]_ 及其引用。

    # References
    # ----------
    # .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
    #        of a Complex Argument and Nonnegative Order",
    #        http://netlib.org/amos/
    # .. [2] Donald E. Amos, "Algorithm 644: A portable package for Bessel
    #        functions of a complex argument and nonnegative order", ACM
    #        TOMS Vol. 12 Issue 3, Sept. 1986, p. 265

    # Examples
    # --------
    # 在以下示例中，`kv` 返回0，而 `kve` 返回一个有用的有限数。

    # >>> import numpy as np
    # >>> from scipy.special import kv, kve
    # >>> import matplotlib.pyplot as plt
    # >>> kv(3, 1000.), kve(3, 1000.)
    # (0.0, 0.03980696128440973)

    # 通过将列表或NumPy数组作为 `v` 参数的参数来评估不同阶数的一个点上的函数：

    # >>> kve([0, 1, 1.5], 1.)
    # array([1.14446308, 1.63615349, 2.50662827])

    # 通过为 `z` 提供一个数组，在阶数为0时，在几个点上评估函数。

    # >>> points = np.array([1., 3., 10.])
    # >>> kve(0, points)
    # array([1.14446308, 0.6977616 , 0.39163193])

    # 通过为 `v` 和 `z` 同时提供数组来评估不同阶数的多个点的函数。
    # 这两个数组必须能够广播到正确的形状。例如，计算阶数0、1和2在1D点阵列上的结果：

    # >>> kve([[0], [1], [2]], points)
    # array([[1.14446308, 0.6977616 , 0.39163193],
    #        [1.63615349, 0.80656348, 0.41076657],
    #        [4.41677005, 1.23547058, 0.47378525]])

    # 绘制从0到5的阶数0到3的函数。

    # >>> fig, ax = plt.subplots()
    # >>> x = np.linspace(0., 5., 1000)
    # >>> for i in range(4):
    # ...     ax.plot(x, kve(i, x), label=fr'$K_{i!r}(z)\cdot e^z$')
    # >>> ax.legend()
    # >>> ax.set_xlabel(r"$z$")
    # >>> ax.set_ylim(0, 4)
    # >>> ax.set_xlim(0, 5)
    # >>> plt.show()
# 添加新的文档说明 "_lanczos_sum_expg_scaled" 函数，提示不要使用此函数
add_newdoc("_lanczos_sum_expg_scaled",
    """
    Internal function, do not use.
    """)

# 添加新的文档说明 "_lgam1p" 函数，提示不要使用此函数
add_newdoc("_lgam1p",
    """
    Internal function, do not use.
    """)

# 添加新的文档说明 "log1p" 函数，计算 log(1 + x)，特别用于 x 接近 0 的情况
add_newdoc("log1p",
    """
    log1p(x, out=None)

    Calculates log(1 + x) for use when `x` is near zero.

    Parameters
    ----------
    x : array_like
        Real or complex valued input.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Values of ``log(1 + x)``.

    See Also
    --------
    expm1, cosm1

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    It is more accurate than using ``log(1 + x)`` directly for ``x``
    near 0. Note that in the below example ``1 + 1e-17 == 1`` to
    double precision.

    >>> sc.log1p(1e-17)
    1e-17
    >>> np.log(1 + 1e-17)
    0.0
    """)

# 添加新的文档说明 "_log1pmx" 函数，提示不要使用此函数
add_newdoc("_log1pmx",
    """
    Internal function, do not use.
    """)

# 添加新的文档说明 "lpmv" 函数，计算整数阶及实数次的关联Legendre函数
add_newdoc("lpmv",
    r"""
    lpmv(m, v, x, out=None)

    Associated Legendre function of integer order and real degree.

    Defined as

    .. math::

        P_v^m = (-1)^m (1 - x^2)^{m/2} \frac{d^m}{dx^m} P_v(x)

    where

    .. math::

        P_v = \sum_{k = 0}^\infty \frac{(-v)_k (v + 1)_k}{(k!)^2}
                \left(\frac{1 - x}{2}\right)^k

    is the Legendre function of the first kind. Here :math:`(\cdot)_k`
    is the Pochhammer symbol; see `poch`.

    Parameters
    ----------
    m : array_like
        Order (int or float). If passed a float not equal to an
        integer the function returns NaN.
    v : array_like
        Degree (float).
    x : array_like
        Argument (float). Must have ``|x| <= 1``.
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    pmv : scalar or ndarray
        Value of the associated Legendre function.

    See Also
    --------
    lpmn : Compute the associated Legendre function for all orders
           ``0, ..., m`` and degrees ``0, ..., n``.
    clpmn : Compute the associated Legendre function at complex
            arguments.

    Notes
    -----
    Note that this implementation includes the Condon-Shortley phase.

    References
    ----------
    .. [1] Zhang, Jin, "Computation of Special Functions", John Wiley
           and Sons, Inc, 1996.
    """)

# 添加新的文档说明 "modstruve" 函数，计算修改的Struve函数
add_newdoc("modstruve",
    r"""
    modstruve(v, x, out=None)

    Modified Struve function.

    Return the value of the modified Struve function of order `v` at `x`.  The
    modified Struve function is defined as,

    .. math::
        L_v(x) = -\imath \exp(-\pi\imath v/2) H_v(\imath x),

    where :math:`H_v` is the Struve function.

    Parameters
    ----------
    v : array_like
        Order of the modified Struve function (float).
    x : array_like
        Argument of the Struve function (float; must be positive unless `v` is
        an integer).
    out : ndarray, optional
        Optional output array for the function results
    """
)
    out : ndarray, optional
        # 可选的输出数组，用于存储函数的结果

    Returns
    -------
    L : scalar or ndarray
        # 修改后斯特鲁夫函数在给定参数 `v` 和 `x` 处的值

    See Also
    --------
    struve

    Notes
    -----
    # 使用三种方法计算函数的值 [1]_ :

    - 幂级数
    - 贝塞尔函数的展开（如果 :math:`|x| < |v| + 20`）
    - 渐近大 x 展开（如果 :math:`x \geq 0.7v + 12`）

    基于和的最大项估计舍入误差，并返回与最小误差相关联的结果。

    References
    ----------
    .. [1] NIST 数学函数数字图书馆
           https://dlmf.nist.gov/11

    Examples
    --------
    计算一阶修改后斯特鲁夫函数在 2 处的值。

    >>> import numpy as np
    >>> from scipy.special import modstruve
    >>> import matplotlib.pyplot as plt
    >>> modstruve(1, 2.)
    1.102759787367716

    计算给定参数 `v` 的修改后斯特鲁夫函数在 2 处的值，参数 `v` 为列表。

    >>> modstruve([1, 2, 3], 2.)
    array([1.10275979, 0.41026079, 0.11247294])

    通过提供数组 `x` 计算一阶修改后斯特鲁夫函数在多个点的值。

    >>> points = np.array([2., 5., 8.])
    >>> modstruve(1, points)
    array([  1.10275979,  23.72821578, 399.24709139])

    通过提供数组 `v` 和 `z` 计算多个参数 `v` 和 `z` 处的修改后斯特鲁夫函数值。
    这些数组必须能够广播到正确的形状。

    >>> orders = np.array([[1], [2], [3]])
    >>> points.shape, orders.shape
    ((3,), (3, 1))

    >>> modstruve(orders, points)
    array([[1.10275979e+00, 2.37282158e+01, 3.99247091e+02],
           [4.10260789e-01, 1.65535979e+01, 3.25973609e+02],
           [1.12472937e-01, 9.42430454e+00, 2.33544042e+02]])

    绘制修改后斯特鲁夫函数从 -5 到 5 的 0 到 3 阶。

    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(-5., 5., 1000)
    >>> for i in range(4):
    ...     ax.plot(x, modstruve(i, x), label=f'$L_{i!r}$')
    >>> ax.legend(ncol=2)
    >>> ax.set_xlim(-5, 5)
    >>> ax.set_title(r"修改后斯特鲁夫函数 $L_{\nu}$")
    >>> plt.show()
# 添加新的文档字符串到函数 `nbdtr` 中
add_newdoc("nbdtr",
    r"""
    nbdtr(k, n, p, out=None)

    Negative binomial cumulative distribution function.

    Returns the sum of the terms 0 through `k` of the negative binomial
    distribution probability mass function,

    .. math::

        F = \sum_{j=0}^k {{n + j - 1}\choose{j}} p^n (1 - p)^j.

    In a sequence of Bernoulli trials with individual success probabilities
    `p`, this is the probability that `k` or fewer failures precede the nth
    success.

    Parameters
    ----------
    k : array_like
        The maximum number of allowed failures (nonnegative int).
    n : array_like
        The target number of successes (positive int).
    p : array_like
        Probability of success in a single event (float).
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    F : scalar or ndarray
        The probability of `k` or fewer failures before `n` successes in a
        sequence of events with individual success probability `p`.

    See Also
    --------
    nbdtrc : Negative binomial survival function
    nbdtrik : Negative binomial quantile function
    scipy.stats.nbinom : Negative binomial distribution

    Notes
    -----
    If floating point values are passed for `k` or `n`, they will be truncated
    to integers.

    The terms are not summed directly; instead the regularized incomplete beta
    function is employed, according to the formula,

    .. math::
        \mathrm{nbdtr}(k, n, p) = I_{p}(n, k + 1).

    Wrapper for the Cephes [1]_ routine `nbdtr`.

    The negative binomial distribution is also available as
    `scipy.stats.nbinom`. Using `nbdtr` directly can improve performance
    compared to the ``cdf`` method of `scipy.stats.nbinom` (see last example).

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    Compute the function for ``k=10`` and ``n=5`` at ``p=0.5``.

    >>> import numpy as np
    >>> from scipy.special import nbdtr
    >>> nbdtr(10, 5, 0.5)
    0.940765380859375

    Compute the function for ``n=10`` and ``p=0.5`` at several points by
    providing a NumPy array or list for `k`.

    >>> nbdtr([5, 10, 15], 10, 0.5)
    array([0.15087891, 0.58809853, 0.88523853])

    Plot the function for four different parameter sets.

    >>> import matplotlib.pyplot as plt
    >>> k = np.arange(130)
    >>> n_parameters = [20, 20, 20, 80]
    >>> p_parameters = [0.2, 0.5, 0.8, 0.5]
    >>> linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
    >>> parameters_list = list(zip(p_parameters, n_parameters,
    ...                            linestyles))
    >>> fig, ax = plt.subplots(figsize=(8, 8))
    >>> for parameter_set in parameters_list:
    ...     p, n, style = parameter_set
    ...     nbdtr_vals = nbdtr(k, n, p)
    ...     ax.plot(k, nbdtr_vals, label=rf"$n={n},\, p={p}$",
    ...             ls=style)
    >>> ax.legend()
    # 设置 x 轴的标签为 "$k$"
    ax.set_xlabel("$k$")
    # 设置图表的标题为 "Negative binomial cumulative distribution function"
    ax.set_title("Negative binomial cumulative distribution function")
    # 显示图表
    plt.show()

    # 负二项分布也可以通过 `scipy.stats.nbinom` 来获取。直接使用 `nbdtr` 函数可能比调用 `scipy.stats.nbinom` 的 `cdf` 方法更快，特别是对于小数组或单个值来说。为了获得相同的结果，必须使用如下的参数设置：`nbinom(n, p).cdf(k)=nbdtr(k, n, p)`。
    
    # 导入需要的模块和函数
    from scipy.stats import nbinom
    # 设定参数 k, n, p
    k, n, p = 5, 3, 0.5
    # 使用 nbdtr 函数计算结果，这通常比下面的方法更快
    nbdtr_res = nbdtr(k, n, p)
    # 使用 scipy.stats.nbinom 的 cdf 方法计算结果
    stats_res = nbinom(n, p).cdf(k)
    # 检查两种方法计算的结果是否相等
    stats_res, nbdtr_res  # test that results are equal
    # 输出：(0.85546875, 0.85546875)

    # `nbdtr` 可以通过提供与 `k`, `n` 和 `p` 的广播兼容形状的数组来评估不同的参数设置。在此示例中，我们计算了三个不同 `k` 值在四个不同 `p` 值位置上的函数值，结果是一个 3x4 的数组。

    # 定义数组 k 和 p
    k = np.array([[5], [10], [15]])
    p = np.array([0.3, 0.5, 0.7, 0.9])
    # 输出数组 k 和 p 的形状
    k.shape, p.shape
    # 输出：((3, 1), (4,))

    # 使用 nbdtr 函数计算结果，展示了不同 `k` 值在固定 `n=5` 和不同 `p` 值上的结果
    nbdtr(k, 5, p)
    # 输出：
    # array([[0.15026833, 0.62304687, 0.95265101, 0.9998531 ],
    #        [0.48450894, 0.94076538, 0.99932777, 0.99999999],
    #        [0.76249222, 0.99409103, 0.99999445, 1.        ]])
add_newdoc("nbdtrc",
    r"""
    nbdtrc(k, n, p, out=None)

    Negative binomial survival function.

    Returns the sum of the terms `k + 1` to infinity of the negative binomial
    distribution probability mass function,

    .. math::

        F = \sum_{j=k + 1}^\infty {{n + j - 1}\choose{j}} p^n (1 - p)^j.

    In a sequence of Bernoulli trials with individual success probabilities
    `p`, this is the probability that more than `k` failures precede the nth
    success.

    Parameters
    ----------
    k : array_like
        The maximum number of allowed failures (nonnegative int).
    n : array_like
        The target number of successes (positive int).
    p : array_like
        Probability of success in a single event (float).
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    F : scalar or ndarray
        The probability of `k + 1` or more failures before `n` successes in a
        sequence of events with individual success probability `p`.

    See Also
    --------
    nbdtr : Negative binomial cumulative distribution function
    nbdtrik : Negative binomial percentile function
    scipy.stats.nbinom : Negative binomial distribution

    Notes
    -----
    If floating point values are passed for `k` or `n`, they will be truncated
    to integers.

    The terms are not summed directly; instead the regularized incomplete beta
    function is employed, according to the formula,

    .. math::
        \mathrm{nbdtrc}(k, n, p) = I_{1 - p}(k + 1, n).

    Wrapper for the Cephes [1]_ routine `nbdtrc`.

    The negative binomial distribution is also available as
    `scipy.stats.nbinom`. Using `nbdtrc` directly can improve performance
    compared to the ``sf`` method of `scipy.stats.nbinom` (see last example).

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    Compute the function for ``k=10`` and ``n=5`` at ``p=0.5``.

    >>> import numpy as np
    >>> from scipy.special import nbdtrc
    >>> nbdtrc(10, 5, 0.5)
    0.059234619140624986

    Compute the function for ``n=10`` and ``p=0.5`` at several points by
    providing a NumPy array or list for `k`.

    >>> nbdtrc([5, 10, 15], 10, 0.5)
    array([0.84912109, 0.41190147, 0.11476147])

    Plot the function for four different parameter sets.

    >>> import matplotlib.pyplot as plt
    >>> k = np.arange(130)
    >>> n_parameters = [20, 20, 20, 80]
    >>> p_parameters = [0.2, 0.5, 0.8, 0.5]
    >>> linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
    >>> parameters_list = list(zip(p_parameters, n_parameters,
    ...                            linestyles))
    >>> fig, ax = plt.subplots(figsize=(8, 8))
    >>> for parameter_set in parameters_list:
    ...     p, n, style = parameter_set
    ...     nbdtrc_vals = nbdtrc(k, n, p)
    ...     ax.plot(k, nbdtrc_vals, label=rf"$n={n},\, p={p}$",
    ...             linestyle=style)
    >>> ax.legend()
    >>> plt.show()
    """
    # 在指定风格下绘制负二项分布的生存函数图像
    >>> ax.plot(np.arange(0, 10, 0.01), nbdtrc(np.arange(0, 10, 0.01), 5, 0.5), ls=style)
    # 添加图例
    >>> ax.legend()
    # 设置 x 轴标签
    >>> ax.set_xlabel("$k$")
    # 设置图表标题
    >>> ax.set_title("Negative binomial distribution survival function")
    # 显示图表
    >>> plt.show()

    # 负二项分布也可以使用 `scipy.stats.nbinom` 获取，直接使用 `nbdtrc` 比调用 `scipy.stats.nbinom` 的 `sf` 方法更快，
    # 特别是对于小数组或单个值。为了获得相同的结果，必须使用以下参数化方式：`nbinom(n, p).sf(k) = nbdtrc(k, n, p)`。

    # 导入 `scipy.stats.nbinom`
    >>> from scipy.stats import nbinom
    # 设定参数 k, n, p
    >>> k, n, p = 3, 5, 0.5
    # 使用 `nbdtrc` 计算结果，这通常比以下的方法更快
    >>> nbdtr_res = nbdtrc(k, n, p)
    # 使用 `scipy.stats.nbinom` 的 `sf` 方法计算结果
    >>> stats_res = nbinom(n, p).sf(k)
    # 检验两种方法的结果是否相等
    >>> stats_res, nbdtr_res  # test that results are equal
    (0.6367187499999999, 0.6367187499999999)

    # `nbdtrc` 可以通过提供与 `k`、`n` 和 `p` 的广播兼容形状的数组来评估不同的参数组合。
    # 这里我们计算三个不同 `k` 和四个不同 `p` 的函数值，结果是一个 3x4 的数组。

    # 定义 `k` 的数组形状和 `p` 的数组形状
    >>> k = np.array([[5], [10], [15]])
    >>> p = np.array([0.3, 0.5, 0.7, 0.9])
    # 输出 `k` 和 `p` 的形状
    >>> k.shape, p.shape
    ((3, 1), (4,))

    # 使用 `nbdtrc` 计算结果
    >>> nbdtrc(k, 5, p)
    array([[8.49731667e-01, 3.76953125e-01, 4.73489874e-02, 1.46902600e-04],
           [5.15491059e-01, 5.92346191e-02, 6.72234070e-04, 9.29610100e-09],
           [2.37507779e-01, 5.90896606e-03, 5.55025308e-06, 3.26346760e-13]])
# 添加新的文档字符串到指定函数或模块中，此处为 "nbdtri" 函数
add_newdoc(
    "nbdtri",
    r"""
    nbdtri(k, n, y, out=None)

    Returns the inverse with respect to the parameter `p` of
    ``y = nbdtr(k, n, p)``, the negative binomial cumulative distribution
    function.

    Parameters
    ----------
    k : array_like
        The maximum number of allowed failures (nonnegative int).
    n : array_like
        The target number of successes (positive int).
    y : array_like
        The probability of `k` or fewer failures before `n` successes (float).
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    p : scalar or ndarray
        Probability of success in a single event (float) such that
        `nbdtr(k, n, p) = y`.

    See Also
    --------
    nbdtr : Cumulative distribution function of the negative binomial.
    nbdtrc : Negative binomial survival function.
    scipy.stats.nbinom : negative binomial distribution.
    nbdtrik : Inverse with respect to `k` of `nbdtr(k, n, p)`.
    nbdtrin : Inverse with respect to `n` of `nbdtr(k, n, p)`.
    scipy.stats.nbinom : Negative binomial distribution

    Notes
    -----
    Wrapper for the Cephes [1]_ routine `nbdtri`.

    The negative binomial distribution is also available as
    `scipy.stats.nbinom`. Using `nbdtri` directly can improve performance
    compared to the ``ppf`` method of `scipy.stats.nbinom`.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    `nbdtri` is the inverse of `nbdtr` with respect to `p`.
    Up to floating point errors the following holds:
    ``nbdtri(k, n, nbdtr(k, n, p))=p``.

    >>> import numpy as np
    >>> from scipy.special import nbdtri, nbdtr
    >>> k, n, y = 5, 10, 0.2
    >>> cdf_val = nbdtr(k, n, y)
    >>> nbdtri(k, n, cdf_val)
    0.20000000000000004

    Compute the function for ``k=10`` and ``n=5`` at several points by
    providing a NumPy array or list for `y`.

    >>> y = np.array([0.1, 0.4, 0.8])
    >>> nbdtri(3, 5, y)
    array([0.34462319, 0.51653095, 0.69677416])

    Plot the function for three different parameter sets.

    >>> import matplotlib.pyplot as plt
    >>> n_parameters = [5, 20, 30, 30]
    >>> k_parameters = [20, 20, 60, 80]
    >>> linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
    >>> parameters_list = list(zip(n_parameters, k_parameters, linestyles))
    >>> cdf_vals = np.linspace(0, 1, 1000)
    >>> fig, ax = plt.subplots(figsize=(8, 8))
    >>> for parameter_set in parameters_list:
    ...     n, k, style = parameter_set
    ...     nbdtri_vals = nbdtri(k, n, cdf_vals)
    ...     ax.plot(cdf_vals, nbdtri_vals, label=rf"$k={k},\ n={n}$",
    ...             ls=style)
    >>> ax.legend()
    >>> ax.set_ylabel("$p$")
    >>> ax.set_xlabel("$CDF$")
    >>> title = "nbdtri: inverse of negative binomial CDF with respect to $p$"
    >>> ax.set_title(title)
    >>> plt.show()
    """
)
    `nbdtri` can evaluate different parameter sets by providing arrays with
    shapes compatible for broadcasting for `k`, `n` and `p`. Here we compute
    the function for three different `k` at four locations `p`, resulting in
    a 3x4 array.

    >>> k = np.array([[5], [10], [15]])
    # 创建包含三个不同k值的二维 NumPy 数组
    >>> y = np.array([0.3, 0.5, 0.7, 0.9])
    # 创建包含四个不同p值的一维 NumPy 数组
    >>> k.shape, y.shape
    # 输出 k 和 y 的形状
    ((3, 1), (4,))

    >>> nbdtri(k, 5, y)
    # 使用 nbdtri 函数计算给定参数下的负二项分布累积分布函数值
    array([[0.37258157, 0.45169416, 0.53249956, 0.64578407],
           [0.24588501, 0.30451981, 0.36778453, 0.46397088],
           [0.18362101, 0.22966758, 0.28054743, 0.36066188]])
    """
# 将函数添加到模块的文档字符串中，描述了负二项分布的百分位函数
add_newdoc("nbdtrik",
    r"""
    nbdtrik(y, n, p, out=None)

    Negative binomial percentile function.

    Returns the inverse with respect to the parameter `k` of
    ``y = nbdtr(k, n, p)``, the negative binomial cumulative distribution
    function.

    Parameters
    ----------
    y : array_like
        The probability of `k` or fewer failures before `n` successes (float).
    n : array_like
        The target number of successes (positive int).
    p : array_like
        Probability of success in a single event (float).
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    k : scalar or ndarray
        The maximum number of allowed failures such that `nbdtr(k, n, p) = y`.

    See Also
    --------
    nbdtr : Cumulative distribution function of the negative binomial.
    nbdtrc : Survival function of the negative binomial.
    nbdtri : Inverse with respect to `p` of `nbdtr(k, n, p)`.
    nbdtrin : Inverse with respect to `n` of `nbdtr(k, n, p)`.
    scipy.stats.nbinom : Negative binomial distribution

    Notes
    -----
    Wrapper for the CDFLIB [1]_ Fortran routine `cdfnbn`.

    Formula 26.5.26 of [2]_,

    .. math::
        \sum_{j=k + 1}^\infty {{n + j - 1}
        \choose{j}} p^n (1 - p)^j = I_{1 - p}(k + 1, n),

    is used to reduce calculation of the cumulative distribution function to
    that of a regularized incomplete beta :math:`I`.

    Computation of `k` involves a search for a value that produces the desired
    value of `y`.  The search relies on the monotonicity of `y` with `k`.

    References
    ----------
    .. [1] Barry Brown, James Lovato, and Kathy Russell,
           CDFLIB: Library of Fortran Routines for Cumulative Distribution
           Functions, Inverses, and Other Parameters.
    .. [2] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    Compute the negative binomial cumulative distribution function for an
    exemplary parameter set.

    >>> import numpy as np
    >>> from scipy.special import nbdtr, nbdtrik
    >>> k, n, p = 5, 2, 0.5
    >>> cdf_value = nbdtr(k, n, p)
    >>> cdf_value
    0.9375

    Verify that `nbdtrik` recovers the original value for `k`.

    >>> nbdtrik(cdf_value, n, p)
    5.0

    Plot the function for different parameter sets.

    >>> import matplotlib.pyplot as plt
    >>> p_parameters = [0.2, 0.5, 0.7, 0.5]
    >>> n_parameters = [30, 30, 30, 80]
    >>> linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
    >>> parameters_list = list(zip(p_parameters, n_parameters, linestyles))
    >>> cdf_vals = np.linspace(0, 1, 1000)
    >>> fig, ax = plt.subplots(figsize=(8, 8))
    >>> for parameter_set in parameters_list:
    ...     p, n, style = parameter_set
    ...     nbdtrik_vals = nbdtrik(cdf_vals, n, p)
    # 在当前坐标轴上绘制图形，绘制负二项分布的累积分布函数图像
    ax.plot(cdf_vals, nbdtrik_vals, label=rf"$n={n},\ p={p}$", ls=style)
    # 添加图例，显示标签为负二项分布参数的描述
    ax.legend()
    # 设置纵轴标签为"$k$"
    ax.set_ylabel("$k$")
    # 设置横轴标签为"$CDF$"
    ax.set_xlabel("$CDF$")
    # 设置图像标题为"Negative binomial percentile function"
    ax.set_title("Negative binomial percentile function")
    # 显示图形
    plt.show()

    # 负二项分布也可以通过 scipy.stats.nbinom 访问。百分位数函数 "ppf" 返回 nbdtrik 方法的结果，将其四舍五入为整数：
    from scipy.stats import nbinom
    q, n, p = 0.6, 5, 0.5
    # 计算负二项分布的分位数，以及自定义实现的 nbdtrik 方法的结果
    nbinom.ppf(q, n, p), nbdtrik(q, n, p)
    # 输出结果 (5.0, 4.800428460273882)
# 添加新文档到函数 "nbdtrin"
add_newdoc("nbdtrin",
    r"""
    nbdtrin(k, y, p, out=None)

    Inverse of `nbdtr` vs `n`.

    Returns the inverse with respect to the parameter `n` of
    ``y = nbdtr(k, n, p)``, the negative binomial cumulative distribution
    function.

    Parameters
    ----------
    k : array_like
        The maximum number of allowed failures (nonnegative int).
    y : array_like
        The probability of `k` or fewer failures before `n` successes (float).
    p : array_like
        Probability of success in a single event (float).
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    n : scalar or ndarray
        The number of successes `n` such that `nbdtr(k, n, p) = y`.

    See Also
    --------
    nbdtr : Cumulative distribution function of the negative binomial.
    nbdtri : Inverse with respect to `p` of `nbdtr(k, n, p)`.
    nbdtrik : Inverse with respect to `k` of `nbdtr(k, n, p)`.

    Notes
    -----
    Wrapper for the CDFLIB [1]_ Fortran routine `cdfnbn`.

    Formula 26.5.26 of [2]_,

    .. math::
        \sum_{j=k + 1}^\infty {{n + j - 1}
        \choose{j}} p^n (1 - p)^j = I_{1 - p}(k + 1, n),

    is used to reduce calculation of the cumulative distribution function to
    that of a regularized incomplete beta :math:`I`.

    Computation of `n` involves a search for a value that produces the desired
    value of `y`.  The search relies on the monotonicity of `y` with `n`.

    References
    ----------
    .. [1] Barry Brown, James Lovato, and Kathy Russell,
           CDFLIB: Library of Fortran Routines for Cumulative Distribution
           Functions, Inverses, and Other Parameters.
    .. [2] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    Compute the negative binomial cumulative distribution function for an
    exemplary parameter set.

    >>> from scipy.special import nbdtr, nbdtrin
    >>> k, n, p = 5, 2, 0.5
    >>> cdf_value = nbdtr(k, n, p)
    >>> cdf_value
    0.9375

    Verify that `nbdtrin` recovers the original value for `n` up to floating
    point accuracy.

    >>> nbdtrin(k, cdf_value, p)
    1.999999999998137
    """)

# 添加新文档到函数 "ncfdtr"
add_newdoc("ncfdtr",
    r"""
    ncfdtr(dfn, dfd, nc, f, out=None)

    Cumulative distribution function of the non-central F distribution.

    The non-central F describes the distribution of,

    .. math::
        Z = \frac{X/d_n}{Y/d_d}

    where :math:`X` and :math:`Y` are independently distributed, with
    :math:`X` distributed non-central :math:`\chi^2` with noncentrality
    parameter `nc` and :math:`d_n` degrees of freedom, and :math:`Y`
    distributed :math:`\chi^2` with :math:`d_d` degrees of freedom.

    Parameters
    ----------
    dfn : array_like
        Degrees of freedom of the numerator sum of squares.  Range (0, inf).
    dfd : array_like
        Degrees of freedom of the denominator sum of squares.  Range (0, inf).
    nc : array_like
        Noncentrality parameter.  Should be in range (0, 1e4).
    f : array_like
        Quantiles, i.e. the upper limit of integration.
    out : ndarray, optional
        Optional output array for the function results


    Returns
    -------
    cdf : scalar or ndarray
        The calculated CDF.  If all inputs are scalar, the return will be a
        float.  Otherwise it will be an array.


    See Also
    --------
    ncfdtri : Quantile function; inverse of `ncfdtr` with respect to `f`.
    ncfdtridfd : Inverse of `ncfdtr` with respect to `dfd`.
    ncfdtridfn : Inverse of `ncfdtr` with respect to `dfn`.
    ncfdtrinc : Inverse of `ncfdtr` with respect to `nc`.


    Notes
    -----
    Wrapper for the CDFLIB [1]_ Fortran routine `cdffnc`.

    The cumulative distribution function is computed using Formula 26.6.20 of
    [2]_:

    .. math::
        F(d_n, d_d, n_c, f) = \sum_{j=0}^\infty e^{-n_c/2}
        \frac{(n_c/2)^j}{j!} I_{x}(\frac{d_n}{2} + j, \frac{d_d}{2}),

    where :math:`I` is the regularized incomplete beta function, and
    :math:`x = f d_n/(f d_n + d_d)`.

    The computation time required for this routine is proportional to the
    noncentrality parameter `nc`.  Very large values of this parameter can
    consume immense computer resources.  This is why the search range is
    bounded by 10,000.


    References
    ----------
    .. [1] Barry Brown, James Lovato, and Kathy Russell,
           CDFLIB: Library of Fortran Routines for Cumulative Distribution
           Functions, Inverses, and Other Parameters.
    .. [2] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.


    Examples
    --------
    >>> import numpy as np
    >>> from scipy import special
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt

    Plot the CDF of the non-central F distribution, for nc=0.  Compare with the
    F-distribution from scipy.stats:

    >>> x = np.linspace(-1, 8, num=500)
    >>> dfn = 3
    >>> dfd = 2
    >>> ncf_stats = stats.f.cdf(x, dfn, dfd)
    >>> ncf_special = special.ncfdtr(dfn, dfd, 0, x)

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot(x, ncf_stats, 'b-', lw=3)
    >>> ax.plot(x, ncf_special, 'r-')
    >>> plt.show()
# 添加新的文档字符串给函数 "ncfdtri"
add_newdoc("ncfdtri",
    """
    ncfdtri(dfn, dfd, nc, p, out=None)

    Inverse with respect to `f` of the CDF of the non-central F distribution.

    See `ncfdtr` for more details.

    Parameters
    ----------
    dfn : array_like
        Degrees of freedom of the numerator sum of squares.  Range (0, inf).
    dfd : array_like
        Degrees of freedom of the denominator sum of squares.  Range (0, inf).
    nc : array_like
        Noncentrality parameter.  Should be in range (0, 1e4).
    p : array_like
        Value of the cumulative distribution function.  Must be in the
        range [0, 1].
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    f : scalar or ndarray
        Quantiles, i.e., the upper limit of integration.

    See Also
    --------
    ncfdtr : CDF of the non-central F distribution.
    ncfdtridfd : Inverse of `ncfdtr` with respect to `dfd`.
    ncfdtridfn : Inverse of `ncfdtr` with respect to `dfn`.
    ncfdtrinc : Inverse of `ncfdtr` with respect to `nc`.

    Examples
    --------
    >>> from scipy.special import ncfdtr, ncfdtri

    Compute the CDF for several values of `f`:

    >>> f = [0.5, 1, 1.5]
    >>> p = ncfdtr(2, 3, 1.5, f)
    >>> p
    array([ 0.20782291,  0.36107392,  0.47345752])

    Compute the inverse.  We recover the values of `f`, as expected:

    >>> ncfdtri(2, 3, 1.5, p)
    array([ 0.5,  1. ,  1.5])

    """)

# 添加新的文档字符串给函数 "ncfdtridfd"
add_newdoc("ncfdtridfd",
    """
    ncfdtridfd(dfn, p, nc, f, out=None)

    Calculate degrees of freedom (denominator) for the noncentral F-distribution.

    This is the inverse with respect to `dfd` of `ncfdtr`.
    See `ncfdtr` for more details.

    Parameters
    ----------
    dfn : array_like
        Degrees of freedom of the numerator sum of squares.  Range (0, inf).
    p : array_like
        Value of the cumulative distribution function.  Must be in the
        range [0, 1].
    nc : array_like
        Noncentrality parameter.  Should be in range (0, 1e4).
    f : array_like
        Quantiles, i.e., the upper limit of integration.
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    dfd : scalar or ndarray
        Degrees of freedom of the denominator sum of squares.

    See Also
    --------
    ncfdtr : CDF of the non-central F distribution.
    ncfdtri : Quantile function; inverse of `ncfdtr` with respect to `f`.
    ncfdtridfn : Inverse of `ncfdtr` with respect to `dfn`.
    ncfdtrinc : Inverse of `ncfdtr` with respect to `nc`.

    Notes
    -----
    The value of the cumulative noncentral F distribution is not necessarily
    monotone in either degrees of freedom. There thus may be two values that
    provide a given CDF value. This routine assumes monotonicity and will
    find an arbitrary one of the two values.

    Examples
    --------
    >>> from scipy.special import ncfdtr, ncfdtridfd

    Compute the CDF for several values of `dfd`:
    """)
    # 创建一个包含整数的列表
    dfd = [1, 2, 3]
    # 调用 ncfdtr 函数计算非中心 F 分布的累积分布函数，返回一个包含概率值的数组
    p = ncfdtr(2, dfd, 0.25, 15)
    # 打印数组 p 的值
    p
    # 计算非中心 F 分布的累积分布函数的逆函数，传入参数 2 (自由度), p (概率值数组), 0.25 (非中心参数), 15 (自由度参数)
    # 预期结果应该是恢复原始的 dfd 数组值：
    
    # 调用 ncfdtridfd 函数，传入参数 2 (自由度), p (概率值数组), 0.25 (非中心参数), 15 (自由度参数)，返回一个数组
    array([ 1.,  2.,  3.])
# 向 scipy 库中的 ncfdtridfn 函数添加新文档字符串
add_newdoc("ncfdtridfn",
    """
    ncfdtridfn(p, dfd, nc, f, out=None)

    计算非中心 F 分布的分子自由度。

    这是相对于 `dfn` 的 `ncfdtr` 的反函数。
    更多细节请参见 `ncfdtr`。

    Parameters
    ----------
    p : array_like
        累积分布函数的值。必须在区间 [0, 1] 内。
    dfd : array_like
        分母平方和的自由度。取值范围为 (0, inf)。
    nc : array_like
        非中心参数。应在范围 (0, 1e4) 内。
    f : float
        分位数，即积分的上限。
    out : ndarray, optional
        可选的输出数组，用于存储函数结果。

    Returns
    -------
    dfn : scalar or ndarray
        分子平方和的自由度。

    See Also
    --------
    ncfdtr : 非中心 F 分布的累积分布函数。
    ncfdtri : 分位数函数；相对于 `f` 的 `ncfdtr` 的反函数。
    ncfdtridfd : 相对于 `dfd` 的 `ncfdtr` 的反函数。
    ncfdtrinc : 相对于 `nc` 的 `ncfdtr` 的反函数。

    Notes
    -----
    非中心 F 分布的累积分布值在自由度上不一定单调。因此可能存在给定累积分布函数值的两个值。
    该函数假设单调性，并将找到其中任意一个值。

    Examples
    --------
    >>> from scipy.special import ncfdtr, ncfdtridfn

    计算几个 `dfn` 值对应的累积分布函数值：

    >>> dfn = [1, 2, 3]
    >>> p = ncfdtr(dfn, 2, 0.25, 15)
    >>> p
    array([ 0.92562363,  0.93020416,  0.93188394])

    计算反函数。我们得到了预期的 `dfn` 值：

    >>> ncfdtridfn(p, 2, 0.25, 15)
    array([ 1.,  2.,  3.])

    """)

# 向 scipy 库中的 ncfdtrinc 函数添加新文档字符串
add_newdoc("ncfdtrinc",
    """
    ncfdtrinc(dfn, dfd, p, f, out=None)

    计算非中心 F 分布的非中心参数。

    这是相对于 `nc` 的 `ncfdtr` 的反函数。
    更多细节请参见 `ncfdtr`。

    Parameters
    ----------
    dfn : array_like
        分子平方和的自由度。取值范围为 (0, inf)。
    dfd : array_like
        分母平方和的自由度。取值范围为 (0, inf)。
    p : array_like
        累积分布函数的值。必须在区间 [0, 1] 内。
    f : array_like
        分位数，即积分的上限。
    out : ndarray, optional
        可选的输出数组，用于存储函数结果。

    Returns
    -------
    nc : scalar or ndarray
        非中心参数。

    See Also
    --------
    ncfdtr : 非中心 F 分布的累积分布函数。
    ncfdtri : 分位数函数；相对于 `f` 的 `ncfdtr` 的反函数。
    ncfdtridfd : 相对于 `dfd` 的 `ncfdtr` 的反函数。
    ncfdtridfn : 相对于 `dfn` 的 `ncfdtr` 的反函数。

    Examples
    --------
    >>> from scipy.special import ncfdtr, ncfdtrinc

    """
    Compute the CDF for several values of `nc`:
    
    >>> nc = [0.5, 1.5, 2.0]
    # 计算自由度参数为2和3的非中心 F 分布的累积分布函数(CDF)，对应于 `nc` 中的多个值
    >>> p = ncfdtr(2, 3, nc, 15)
    # p 是计算得到的累积分布函数值的数组
    >>> p
    # 打印结果，显示累积分布函数的计算结果
    array([ 0.96309246,  0.94327955,  0.93304098])
    
    Compute the inverse. We recover the values of `nc`, as expected:
    
    >>> ncfdtrinc(2, 3, p, 15)
    # 使用相同的自由度和非中心参数，以及累积分布函数的值 p，计算非中心 F 分布的逆函数
    array([ 0.5,  1.5,  2. ])
    
add_newdoc("nctdtr",
    """
    nctdtr(df, nc, t, out=None)

    Cumulative distribution function of the non-central `t` distribution.

    Parameters
    ----------
    df : array_like
        Degrees of freedom of the distribution. Should be in range (0, inf).
    nc : array_like
        Noncentrality parameter. Should be in range (-1e6, 1e6).
    t : array_like
        Quantiles, i.e., the upper limit of integration.
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    cdf : scalar or ndarray
        The calculated CDF. If all inputs are scalar, the return will be a
        float. Otherwise, it will be an array.

    See Also
    --------
    nctdtrit : Inverse CDF (iCDF) of the non-central t distribution.
    nctdtridf : Calculate degrees of freedom, given CDF and iCDF values.
    nctdtrinc : Calculate non-centrality parameter, given CDF iCDF values.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import special
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt

    Plot the CDF of the non-central t distribution, for nc=0. Compare with the
    t-distribution from scipy.stats:

    >>> x = np.linspace(-5, 5, num=500)
    >>> df = 3
    >>> nct_stats = stats.t.cdf(x, df)
    >>> nct_special = special.nctdtr(df, 0, x)

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot(x, nct_stats, 'b-', lw=3)
    >>> ax.plot(x, nct_special, 'r-')
    >>> plt.show()

    """)

add_newdoc("nctdtridf",
    """
    nctdtridf(p, nc, t, out=None)

    Calculate degrees of freedom for non-central t distribution.

    See `nctdtr` for more details.

    Parameters
    ----------
    p : array_like
        CDF values, in range (0, 1].
    nc : array_like
        Noncentrality parameter. Should be in range (-1e6, 1e6).
    t : array_like
        Quantiles, i.e., the upper limit of integration.
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    df : scalar or ndarray
        The degrees of freedom. If all inputs are scalar, the return will be a
        float. Otherwise, it will be an array.

    See Also
    --------
    nctdtr :  CDF of the non-central `t` distribution.
    nctdtrit : Inverse CDF (iCDF) of the non-central t distribution.
    nctdtrinc : Calculate non-centrality parameter, given CDF iCDF values.

    Examples
    --------
    >>> from scipy.special import nctdtr, nctdtridf

    Compute the CDF for several values of `df`:

    >>> df = [1, 2, 3]
    >>> p = nctdtr(df, 0.25, 1)
    >>> p
    array([0.67491974, 0.716464  , 0.73349456])

    Compute the inverse. We recover the values of `df`, as expected:

    >>> nctdtridf(p, 0.25, 1)
    array([1., 2., 3.])

    """)

add_newdoc("nctdtrinc",
    """
    nctdtrinc(df, p, t, out=None)

    Calculate non-centrality parameter for non-central t distribution.

    See `nctdtr` for more details.

    Parameters
    ----------
    df : array_like
        Degrees of freedom of the distribution. Should be in range (0, inf).
    p : array_like
        CDF values, in range (0, 1].
    t : array_like
        Quantiles, i.e., the upper limit of integration.
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    nc : scalar or ndarray
        The non-centrality parameter. If all inputs are scalar, the return will be a
        float. Otherwise, it will be an array.

    See Also
    --------
    nctdtr :  CDF of the non-central `t` distribution.
    nctdtrit : Inverse CDF (iCDF) of the non-central t distribution.
    nctdtridf : Calculate degrees of freedom, given CDF and iCDF values.

    Examples
    --------
    >>> from scipy.special import nctdtr, nctdtrinc

    Calculate the non-centrality parameter for different degrees of freedom:

    >>> df = [1, 2, 3]
    >>> p = [0.5, 0.7, 0.9]
    >>> nctdtrinc(df, p, 1)
    array([0.72674761, 0.64261595, 0.46465834])

    """)
    # df : array_like
    # Degrees of freedom of the distribution. Should be in range (0, inf).

    # p : array_like
    # CDF values, in range (0, 1].

    # t : array_like
    # Quantiles, i.e., the upper limit of integration.

    # out : ndarray, optional
    # Optional output array for the function results

    # Returns
    # -------
    # nc : scalar or ndarray
    # Noncentrality parameter

    # See Also
    # --------
    # nctdtr :  CDF of the non-central `t` distribution.
    # nctdtrit : Inverse CDF (iCDF) of the non-central t distribution.
    # nctdtridf : Calculate degrees of freedom, given CDF and iCDF values.

    # Examples
    # --------
    # >>> from scipy.special import nctdtr, nctdtrinc

    # Compute the CDF for several values of `nc`:

    # >>> nc = [0.5, 1.5, 2.5]
    # >>> p = nctdtr(3, nc, 1.5)
    # >>> p
    # array([0.77569497, 0.45524533, 0.1668691 ])

    # Compute the inverse. We recover the values of `nc`, as expected:

    # >>> nctdtrinc(3, p, 1.5)
    # array([0.5, 1.5, 2.5])
# 添加新文档到模块 "nctdtrit" 中，描述了非中心 t 分布的反向累积分布函数
add_newdoc("nctdtrit",
    """
    nctdtrit(df, nc, p, out=None)

    Inverse cumulative distribution function of the non-central t distribution.

    See `nctdtr` for more details.

    Parameters
    ----------
    df : array_like
        Degrees of freedom of the distribution. Should be in range (0, inf).
    nc : array_like
        Noncentrality parameter. Should be in range (-1e6, 1e6).
    p : array_like
        CDF values, in range (0, 1].
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    t : scalar or ndarray
        Quantiles

    See Also
    --------
    nctdtr :  CDF of the non-central `t` distribution.
    nctdtridf : Calculate degrees of freedom, given CDF and iCDF values.
    nctdtrinc : Calculate non-centrality parameter, given CDF iCDF values.

    Examples
    --------
    >>> from scipy.special import nctdtr, nctdtrit

    Compute the CDF for several values of `t`:

    >>> t = [0.5, 1, 1.5]
    >>> p = nctdtr(3, 1, t)
    >>> p
    array([0.29811049, 0.46922687, 0.6257559 ])

    Compute the inverse. We recover the values of `t`, as expected:

    >>> nctdtrit(3, 1, p)
    array([0.5, 1. , 1.5])

    """)

# 添加新文档到模块 "ndtr" 中，描述了标准正态分布的累积分布函数
add_newdoc("ndtr",
    r"""
    ndtr(x, out=None)

    Cumulative distribution of the standard normal distribution.

    Returns the area under the standard Gaussian probability
    density function, integrated from minus infinity to `x`

    .. math::

       \frac{1}{\sqrt{2\pi}} \int_{-\infty}^x \exp(-t^2/2) dt

    Parameters
    ----------
    x : array_like, real or complex
        Argument
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        The value of the normal CDF evaluated at `x`

    See Also
    --------
    log_ndtr : Logarithm of ndtr
    ndtri : Inverse of ndtr, standard normal percentile function
    erf : Error function
    erfc : 1 - erf
    scipy.stats.norm : Normal distribution

    Examples
    --------
    Evaluate `ndtr` at one point.

    >>> import numpy as np
    >>> from scipy.special import ndtr
    >>> ndtr(0.5)
    0.6914624612740131

    Evaluate the function at several points by providing a NumPy array
    or list for `x`.

    >>> ndtr([0, 0.5, 2])
    array([0.5       , 0.69146246, 0.97724987])

    Plot the function.

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-5, 5, 100)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, ndtr(x))
    >>> ax.set_title(r"Standard normal cumulative distribution function $\Phi$")
    >>> plt.show()
    """)

# 添加新文档到模块 "nrdtrimn" 中，描述了给定其他参数计算正态分布均值的函数
add_newdoc("nrdtrimn",
    """
    nrdtrimn(p, std, x, out=None)

    Calculate mean of normal distribution given other params.

    Parameters
    ----------
    p : array_like
        CDF values, in range (0, 1].
    std : array_like
        Standard deviation.
    x : array_like
        Quantiles, i.e. the upper limit of integration.
    out : ndarray, optional
        # 可选的输出数组，用于函数的结果

    Returns
    -------
    mn : scalar or ndarray
        # 正态分布的均值。

    See Also
    --------
    scipy.stats.norm : Normal distribution
        # 正态分布的文档链接
    ndtr : Standard normal cumulative probability distribution
        # 标准正态累积概率分布的函数链接
    ndtri : Inverse of standard normal CDF with respect to quantile
        # 标准正态分布的累积分布函数的逆函数，相对于分位数
    nrdtrisd : Inverse of normal distribution CDF with respect to
               standard deviation
        # 正态分布的累积分布函数的逆函数，相对于标准差

    Examples
    --------
    `nrdtrimn` can be used to recover the mean of a normal distribution
    if we know the CDF value `p` for a given quantile `x` and the
    standard deviation `std`. First, we calculate
    the normal distribution CDF for an exemplary parameter set.

    >>> from scipy.stats import norm
    >>> mean = 3.
    >>> std = 2.
    >>> x = 6.
    >>> p = norm.cdf(x, loc=mean, scale=std)
    >>> p
    0.9331927987311419

    Verify that `nrdtrimn` returns the original value for `mean`.

    >>> from scipy.special import nrdtrimn
    >>> nrdtrimn(p, std, x)
    3.0000000000000004

    """
# 添加新的文档字符串到 "nrdtrisd" 函数
add_newdoc("nrdtrisd",
    """
    nrdtrisd(mn, p, x, out=None)

    Calculate standard deviation of normal distribution given other params.

    Parameters
    ----------
    mn : scalar or ndarray
        正态分布的均值。
    p : array_like
        CDF 值，范围在 (0, 1]。
    x : array_like
        分位数，即积分的上限。

    out : ndarray, optional
        可选的输出数组用于函数结果。

    Returns
    -------
    std : scalar or ndarray
        标准差。

    See Also
    --------
    scipy.stats.norm : 正态分布
    ndtr : 标准正态累积概率分布
    ndtri : 标准正态分布的反函数
    nrdtrimn : 与均值相关的正态分布的反函数

    Examples
    --------
    `nrdtrisd` 可用于根据给定分位数 `x` 和均值 `mn` 知道 CDF 值 `p` 来恢复正态分布的标准差。首先，我们计算一个示例参数集的正态分布 CDF。

    >>> from scipy.stats import norm
    >>> mean = 3.
    >>> std = 2.
    >>> x = 6.
    >>> p = norm.cdf(x, loc=mean, scale=std)
    >>> p
    0.9331927987311419

    验证 `nrdtrisd` 返回原始的标准差值。

    >>> from scipy.special import nrdtrisd
    >>> nrdtrisd(mean, p, x)
    2.0000000000000004

    """)

# 添加新的文档字符串到 "log_ndtr" 函数
add_newdoc("log_ndtr",
    """
    log_ndtr(x, out=None)

    Logarithm of Gaussian cumulative distribution function.

    Returns the log of the area under the standard Gaussian probability
    density function, integrated from minus infinity to `x`::

        log(1/sqrt(2*pi) * integral(exp(-t**2 / 2), t=-inf..x))

    Parameters
    ----------
    x : array_like, real or complex
        参数
    out : ndarray, optional
        可选的输出数组用于函数结果

    Returns
    -------
    scalar or ndarray
        计算出的正态 CDF 的对数值

    See Also
    --------
    erf
    erfc
    scipy.stats.norm
    ndtr

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import log_ndtr, ndtr

    `log_ndtr(x)` 相比于朴素实现 `np.log(ndtr(x))` 的优势在于，对于中等到大的正值 `x`，表现最为明显：

    >>> x = np.array([6, 7, 9, 12, 15, 25])
    >>> log_ndtr(x)
    array([-9.86587646e-010, -1.27981254e-012, -1.12858841e-019,
           -1.77648211e-033, -3.67096620e-051, -3.05669671e-138])

    对于中等 `x` 值，朴素计算的结果仅有 5 到 6 位有效数字。对于大约大于 8.3 的 `x` 值，朴素表达式返回 0：

    >>> np.log(ndtr(x))
    array([-9.86587701e-10, -1.27986510e-12,  0.00000000e+00,
            0.00000000e+00,  0.00000000e+00,  0.00000000e+00])
    """)

# 添加新的文档字符串到 "ndtri" 函数
add_newdoc("ndtri",
    """
    ndtri(y, out=None)

    Inverse of standard normal cumulative distribution function (CDF).

    Parameters
    ----------
    y : array_like
        Cumulative probability.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Quantiles corresponding to the cumulative probability `y`.

    See Also
    --------
    scipy.stats.norm : Normal distribution
    ndtr : Standard normal cumulative probability distribution
    log_ndtr : Logarithm of standard normal CDF
    nrdtrisd : Standard deviation of normal distribution given other params

    Examples
    --------
    >>> from scipy.stats import norm
    >>> from scipy.special import ndtri

    Calculate the quantile corresponding to a cumulative probability.

    >>> p = 0.9331927987311419
    >>> mean = 3.
    >>> std = 2.
    >>> x = ndtri(p)
    >>> x
    6.000000000000001

    Verify that the CDF returns the original probability.

    >>> norm.cdf(x, loc=mean, scale=std)
    0.9331927987311419
    """)
    Inverse of `ndtr` vs x

    Returns the argument x for which the area under the standard normal
    probability density function (integrated from minus infinity to `x`)
    is equal to y.

    Parameters
    ----------
    p : array_like
        Probability
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    x : scalar or ndarray
        Value of x such that ``ndtr(x) == p``.

    See Also
    --------
    ndtr : Standard normal cumulative probability distribution
    ndtri_exp : Inverse of log_ndtr

    Examples
    --------
    `ndtri` is the percentile function of the standard normal distribution.
    This means it returns the inverse of the cumulative density `ndtr`. First,
    let us compute a cumulative density value.

    >>> import numpy as np
    >>> from scipy.special import ndtri, ndtr
    >>> cdf_val = ndtr(2)  # Compute the cumulative density at x=2
    >>> cdf_val  # Output the computed cumulative density value
    0.9772498680518208

    Verify that `ndtri` yields the original value for `x` up to floating point
    errors.

    >>> ndtri(cdf_val)  # Compute the inverse of cumulative density value
    2.0000000000000004  # Expected to match original x value

    Plot the function. For that purpose, we provide a NumPy array as argument.

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0.01, 1, 200)  # Generate an array of x values
    >>> fig, ax = plt.subplots()  # Create a new figure and axis object
    >>> ax.plot(x, ndtri(x))  # Plot ndtri function against x
    >>> ax.set_title("Standard normal percentile function")  # Set plot title
    >>> plt.show()  # Display the plot
# 向SciPy库的pdtr函数添加新的文档字符串
add_newdoc("pdtr",
    r"""
    pdtr(k, m, out=None)

    Poisson cumulative distribution function.

    Defined as the probability that a Poisson-distributed random
    variable with event rate :math:`m` is less than or equal to
    :math:`k`. More concretely, this works out to be [1]_

    .. math::

       \exp(-m) \sum_{j = 0}^{\lfloor{k}\rfloor} \frac{m^j}{j!}.

    Parameters
    ----------
    k : array_like
        Number of occurrences (nonnegative, real)
    m : array_like
        Shape parameter (nonnegative, real)
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of the Poisson cumulative distribution function

    See Also
    --------
    pdtrc : Poisson survival function
    pdtrik : inverse of `pdtr` with respect to `k`
    pdtri : inverse of `pdtr` with respect to `m`

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Poisson_distribution

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    It is a cumulative distribution function, so it converges to 1
    monotonically as `k` goes to infinity.

    >>> sc.pdtr([1, 10, 100, np.inf], 1)
    array([0.73575888, 0.99999999, 1.        , 1.        ])

    It is discontinuous at integers and constant between integers.

    >>> sc.pdtr([1, 1.5, 1.9, 2], 1)
    array([0.73575888, 0.73575888, 0.73575888, 0.9196986 ])

    """)

# 向SciPy库的pdtrc函数添加新的文档字符串
add_newdoc("pdtrc",
    """
    pdtrc(k, m, out=None)

    Poisson survival function

    Returns the sum of the terms from k+1 to infinity of the Poisson
    distribution: sum(exp(-m) * m**j / j!, j=k+1..inf) = gammainc(
    k+1, m). Arguments must both be non-negative doubles.

    Parameters
    ----------
    k : array_like
        Number of occurrences (nonnegative, real)
    m : array_like
        Shape parameter (nonnegative, real)
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of the Poisson survival function

    See Also
    --------
    pdtr : Poisson cumulative distribution function
    pdtrik : inverse of `pdtr` with respect to `k`
    pdtri : inverse of `pdtr` with respect to `m`

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    It is a survival function, so it decreases to 0
    monotonically as `k` goes to infinity.

    >>> k = np.array([1, 10, 100, np.inf])
    >>> sc.pdtrc(k, 1)
    array([2.64241118e-001, 1.00477664e-008, 3.94147589e-161, 0.00000000e+000])

    It can be expressed in terms of the lower incomplete gamma
    function `gammainc`.

    >>> sc.gammainc(k + 1, 1)
    array([2.64241118e-001, 1.00477664e-008, 3.94147589e-161, 0.00000000e+000])

    """)

# 向SciPy库的pdtri函数添加新的文档字符串
add_newdoc("pdtri",
    """
    pdtri(k, y, out=None)

    Inverse to `pdtr` vs m

    Returns the Poisson variable `m` such that the sum from 0 to `k` of
    the Poisson cumulative distribution function equals `y`.

    Parameters
    ----------
    k : array_like
        Number of occurrences (nonnegative, real)
    y : array_like
        Probability value (0 <= y <= 1)
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of the Poisson variable `m`

    See Also
    --------
    pdtr : Poisson cumulative distribution function
    pdtrc : Poisson survival function
    pdtrik : inverse of `pdtr` with respect to `k`

    """)
    the Poisson density is equal to the given probability `y`:
    calculated by ``gammaincinv(k + 1, y)``. `k` must be a nonnegative
    integer and `y` between 0 and 1.

    Parameters
    ----------
    k : array_like
        Number of occurrences (nonnegative, real)
        表示出现次数的数组或类数组对象（非负实数）
    y : array_like
        Probability
        表示概率的数组或类数组对象
    out : ndarray, optional
        Optional output array for the function results
        可选参数，用于存储函数结果的输出数组

    Returns
    -------
    scalar or ndarray
        Values of the shape parameter `m` such that ``pdtr(k, m) = p``
        返回形状参数 `m` 的值，使得 ``pdtr(k, m) = p``

    See Also
    --------
    pdtr : Poisson cumulative distribution function
        Poisson累积分布函数
    pdtrc : Poisson survival function
        Poisson生存函数
    pdtrik : inverse of `pdtr` with respect to `k`
        相对于 `k` 的 `pdtr` 的反函数

    Examples
    --------
    >>> import scipy.special as sc

    Compute the CDF for several values of `m`:

    >>> m = [0.5, 1, 1.5]
    >>> p = sc.pdtr(1, m)
    >>> p
    array([0.90979599, 0.73575888, 0.5578254 ])

    Compute the inverse. We recover the values of `m`, as expected:

    >>> sc.pdtri(1, p)
    array([0.5, 1. , 1.5])

    """
# 为 `pdtrik` 函数添加新的文档字符串
add_newdoc("pdtrik",
    """
    pdtrik(p, m, out=None)

    Inverse to `pdtr` vs `k`.

    Parameters
    ----------
    p : array_like
        概率
    m : array_like
        形状参数（非负实数）
    out : ndarray, optional
        可选的输出数组，用于存储函数的结果

    Returns
    -------
    scalar or ndarray
        满足 ``pdtr(k, m) = p`` 的次数 `k`

    See Also
    --------
    pdtr : 泊松累积分布函数
    pdtrc : 泊松生存函数
    pdtri : 关于 `m` 对 `pdtr` 的逆函数

    Examples
    --------
    >>> import scipy.special as sc

    计算多个 `k` 值的累积分布函数（CDF）：

    >>> k = [1, 2, 3]
    >>> p = sc.pdtr(k, 2)
    >>> p
    array([0.40600585, 0.67667642, 0.85712346])

    计算逆函数。我们得到了预期的 `k` 值：

    >>> sc.pdtrik(p, 2)
    array([1., 2., 3.])

    """)

# 为 `poch` 函数添加新的文档字符串
add_newdoc("poch",
    r"""
    poch(z, m, out=None)

    Pochhammer symbol.

    Pochhammer 符号（上升阶乘）定义为

    .. math::

        (z)_m = \frac{\Gamma(z + m)}{\Gamma(z)}

    当 `m` 是正整数时，它表示为

    .. math::

        (z)_m = z (z + 1) ... (z + m - 1)

    更多详细信息参见 [dlmf]_。

    Parameters
    ----------
    z, m : array_like
        实数参数。
    out : ndarray, optional
        可选的输出数组，用于存储函数的结果

    Returns
    -------
    scalar or ndarray
        函数的值。

    References
    ----------
    .. [dlmf] Nist, 数字数学函数库
        https://dlmf.nist.gov/5.2#iii

    Examples
    --------
    >>> import scipy.special as sc

    当 `m` 为 0 时，结果为 1。

    >>> sc.poch([1, 2, 3, 4], 0)
    array([1., 1., 1., 1.])

    当 `z` 等于 1 时，它等同于阶乘函数。

    >>> sc.poch(1, 5)
    120.0
    >>> 1 * 2 * 3 * 4 * 5
    120

    可以用 Gamma 函数表达它。

    >>> z, m = 3.7, 2.1
    >>> sc.poch(z, m)
    20.529581933776953
    >>> sc.gamma(z + m) / sc.gamma(z)
    20.52958193377696

    """)

# 为 `powm1` 函数添加新的文档字符串
add_newdoc("powm1", """
    powm1(x, y, out=None)

    Computes ``x**y - 1``.

    计算 ``x**y - 1``。

    当 `y` 接近 0 或 `x` 接近 1 时，此函数特别有用。

    此函数仅适用于实数类型（不像 ``numpy.power`` 接受复数输入）。

    Parameters
    ----------
    x : array_like
        底数。必须为实数类型（即整数或浮点数，不是复数）。
    y : array_like
        指数。必须为实数类型（即整数或浮点数，不是复数）。

    Returns
    -------
    array_like
        计算结果

    Notes
    -----
    .. versionadded:: 1.10.0

    底层代码仅为单精度和双精度浮点数实现。与 `numpy.power` 不同，`powm1` 的整数输入会被转换为浮点数，不接受复数输入。

    注意以下边缘情况：
    * ``powm1(x, 0)`` returns 0 for any ``x``, including 0, ``inf``
      and ``nan``.
    * ``powm1(1, y)`` returns 0 for any ``y``, including ``nan``
      and ``inf``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import powm1

    >>> x = np.array([1.2, 10.0, 0.9999999975])
    >>> y = np.array([1e-9, 1e-11, 0.1875])
    >>> powm1(x, y)
    array([ 1.82321557e-10,  2.30258509e-11, -4.68749998e-10])

    It can be verified that the relative errors in those results
    are less than 2.5e-16.

    Compare that to the result of ``x**y - 1``, where the
    relative errors are all larger than 8e-8:

    >>> x**y - 1
    array([ 1.82321491e-10,  2.30258035e-11, -4.68750039e-10])

    """
# 添加新的文档字符串给函数 "pseudo_huber"
add_newdoc("pseudo_huber",
    r"""
    pseudo_huber(delta, r, out=None)

    Pseudo-Huber loss function.

    .. math:: \mathrm{pseudo\_huber}(\delta, r) =
              \delta^2 \left( \sqrt{ 1 + \left( \frac{r}{\delta} \right)^2 } - 1 \right)

    Parameters
    ----------
    delta : array_like
        输入数组，表示软二次 vs. 线性损失的转折点。
    r : array_like
        输入数组，可能表示残差。
    out : ndarray, optional
        可选的输出数组，用于存储函数结果。

    Returns
    -------
    res : scalar or ndarray
        计算得到的 Pseudo-Huber 损失函数值。

    See Also
    --------
    huber: 类似的函数，该函数近似于 huber 函数

    Notes
    -----
    类似于 `huber`，`pseudo_huber` 经常用作统计学或机器学习中的稳健损失函数，
    以减少异常值的影响。与 `huber` 不同，`pseudo_huber` 是平滑的。

    通常，`r` 表示残差，即模型预测与数据之间的差异。因此，对于 :math:`|r|\leq\delta`，
    `pseudo_huber` 类似于平方误差，而对于 :math:`|r|>\delta`，则类似于绝对误差。
    这种方式，Pseudo-Huber 损失函数在小残差（如平方误差损失函数）的模型拟合中通常会
    达到快速收敛，同时仍然减少异常值（:math:`|r|>\delta`）的影响。因为 :math:`\delta`
    是平方误差和绝对误差之间的切换点，需要针对每个问题进行仔细调整。`pseudo_huber`
    也是凸的，适合于基于梯度的优化。 [1]_ [2]_

    .. versionadded:: 0.15.0

    References
    ----------
    .. [1] Hartley, Zisserman, "Multiple View Geometry in Computer Vision".
           2003. Cambridge University Press. p. 619
    .. [2] Charbonnier et al. "Deterministic edge-preserving regularization
           in computed imaging". 1997. IEEE Trans. Image Processing.
           6 (2): 298 - 311.

    Examples
    --------
    导入所有必要的模块。

    >>> import numpy as np
    >>> from scipy.special import pseudo_huber, huber
    >>> import matplotlib.pyplot as plt

    计算 ``delta=1`` 时， ``r=2`` 的函数值。

    >>> pseudo_huber(1., 2.)
    1.2360679774997898

    通过提供列表或 NumPy 数组给 `delta`，计算不同 `delta` 下的 `r=2` 的函数值。

    >>> pseudo_huber([1., 2., 4.], 3.)
    array([2.16227766, 3.21110255, 4.        ])

    通过提供列表或 NumPy 数组给 `r`，计算不同 `r` 下的 `delta=1` 的函数值。

    >>> pseudo_huber(2., np.array([1., 1.5, 3., 4.]))
    array([0.47213595, 1.        , 3.21110255, 4.94427191])

    通过提供形状兼容的数组来同时计算不同 `delta` 和 `r` 的函数值。

    >>> r = np.array([1., 2.5, 8., 10.])
    >>> deltas = np.array([[1.], [5.], [9.]])
    >>> print(r.shape, deltas.shape)
    ```
    # 定义一个元组 (4,) 和一个元组 (3, 1)，暂时没有使用说明
    >>> pseudo_huber(deltas, r)
    # 调用 pseudo_huber 函数，传入参数 deltas 和 r，并返回一个二维数组，数组元素表示伪胡贝尔损失函数的计算结果
    
    # 绘制不同 delta 值下的伪胡贝尔损失函数图像
    >>> x = np.linspace(-4, 4, 500)
    # 在区间 [-4, 4] 内生成 500 个等间距的数值作为 x 值
    
    >>> deltas = [1, 2, 3]
    # 定义 delta 参数列表
    
    >>> linestyles = ["dashed", "dotted", "dashdot"]
    # 定义线条样式列表
    
    >>> fig, ax = plt.subplots()
    # 创建一个新的图形和一个包含单个轴的 subplot 对象
    
    >>> combined_plot_parameters = list(zip(deltas, linestyles))
    # 将 delta 和对应的线条样式组合成一个列表
    
    >>> for delta, style in combined_plot_parameters:
    ...     ax.plot(x, pseudo_huber(delta, x), label=rf"$\delta={delta}$",
    ...             ls=style)
    # 遍历组合参数列表，绘制每个 delta 对应的伪胡贝尔损失函数曲线，设置标签和线条样式
    
    >>> ax.legend(loc="upper center")
    # 在图形中添加图例，位置为图形上部中心
    
    >>> ax.set_xlabel("$x$")
    # 设置 x 轴标签为 "$x$"
    
    >>> ax.set_title(r"Pseudo-Huber loss function $h_{\delta}(x)$")
    # 设置图形标题为 "Pseudo-Huber loss function $h_{\delta}(x)$"
    
    >>> ax.set_xlim(-4, 4)
    # 设置 x 轴的显示范围为 [-4, 4]
    
    >>> ax.set_ylim(0, 8)
    # 设置 y 轴的显示范围为 [0, 8]
    
    >>> plt.show()
    # 显示绘制的图形
    
    # 最后，通过绘制 huber 和 pseudo_huber 及其关于 r 的梯度来说明它们之间的差异。
    # 图中显示了 pseudo_huber 在点 $\pm\delta$ 处是连续可微的，而 huber 不是。
    >>> def huber_grad(delta, x):
    ...     grad = np.copy(x)
    ...     linear_area = np.argwhere(np.abs(x) > delta)
    ...     grad[linear_area]=delta*np.sign(x[linear_area])
    ...     return grad
    # 定义 huber 损失函数的梯度函数，根据 delta 和 x 返回对应的梯度值
    
    >>> def pseudo_huber_grad(delta, x):
    ...     return x* (1+(x/delta)**2)**(-0.5)
    # 定义伪胡贝尔损失函数的梯度函数，根据 delta 和 x 返回对应的梯度值
    
    >>> x=np.linspace(-3, 3, 500)
    # 在区间 [-3, 3] 内生成 500 个等间距的数值作为 x 值
    
    >>> delta = 1.
    # 定义 delta 参数为 1.0
    
    >>> fig, ax = plt.subplots(figsize=(7, 7))
    # 创建一个新的图形，设置图形大小为 7x7，并创建一个包含单个轴的 subplot 对象
    
    >>> ax.plot(x, huber(delta, x), label="Huber", ls="dashed")
    # 绘制 huber 函数的图像，设置线条样式为虚线，添加图例标签为 "Huber"
    
    >>> ax.plot(x, huber_grad(delta, x), label="Huber Gradient", ls="dashdot")
    # 绘制 huber 函数的梯度图像，设置线条样式为点划线，添加图例标签为 "Huber Gradient"
    
    >>> ax.plot(x, pseudo_huber(delta, x), label="Pseudo-Huber", ls="dotted")
    # 绘制 pseudo_huber 函数的图像，设置线条样式为点线，添加图例标签为 "Pseudo-Huber"
    
    >>> ax.plot(x, pseudo_huber_grad(delta, x), label="Pseudo-Huber Gradient",
    ...         ls="solid")
    # 绘制 pseudo_huber 函数的梯度图像，设置线条样式为实线，添加图例标签为 "Pseudo-Huber Gradient"
    
    >>> ax.legend(loc="upper center")
    # 在图形中添加图例，位置为图形上部中心
    
    >>> plt.show()
    # 显示绘制的图形
# 添加新的文档字符串到 "radian" 函数
add_newdoc("radian",
    """
    radian(d, m, s, out=None)

    Convert from degrees to radians.

    Returns the angle given in (d)egrees, (m)inutes, and (s)econds in
    radians.

    Parameters
    ----------
    d : array_like
        Degrees, can be real-valued.
    m : array_like
        Minutes, can be real-valued.
    s : array_like
        Seconds, can be real-valued.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Values of the inputs in radians.

    Examples
    --------
    >>> import scipy.special as sc

    There are many ways to specify an angle.

    >>> sc.radian(90, 0, 0)
    1.5707963267948966
    >>> sc.radian(0, 60 * 90, 0)
    1.5707963267948966
    >>> sc.radian(0, 0, 60**2 * 90)
    1.5707963267948966

    The inputs can be real-valued.

    >>> sc.radian(1.5, 0, 0)
    0.02617993877991494
    >>> sc.radian(1, 30, 0)
    0.02617993877991494

    """)

# 添加新的文档字符串到 "rel_entr" 函数
add_newdoc("rel_entr",
    r"""
    rel_entr(x, y, out=None)

    Elementwise function for computing relative entropy.

    .. math::

        \mathrm{rel\_entr}(x, y) =
            \begin{cases}
                x \log(x / y) & x > 0, y > 0 \\
                0 & x = 0, y \ge 0 \\
                \infty & \text{otherwise}
            \end{cases}

    Parameters
    ----------
    x, y : array_like
        Input arrays
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Relative entropy of the inputs

    See Also
    --------
    entr, kl_div, scipy.stats.entropy

    Notes
    -----
    .. versionadded:: 0.15.0

    This function is jointly convex in x and y.

    The origin of this function is in convex programming; see
    [1]_. Given two discrete probability distributions :math:`p_1,
    \ldots, p_n` and :math:`q_1, \ldots, q_n`, the definition of relative
    entropy in the context of *information theory* is

    .. math::

        \sum_{i = 1}^n \mathrm{rel\_entr}(p_i, q_i).

    To compute the latter quantity, use `scipy.stats.entropy`.

    See [2]_ for details.

    References
    ----------
    .. [1] Boyd, Stephen and Lieven Vandenberghe. *Convex optimization*.
           Cambridge University Press, 2004.
           :doi:`https://doi.org/10.1017/CBO9780511804441`
    .. [2] Kullback-Leibler divergence,
           https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

    """)

# 添加新的文档字符串到 "round" 函数
add_newdoc("round",
    """
    round(x, out=None)

    Round to the nearest integer.

    Returns the nearest integer to `x`.  If `x` ends in 0.5 exactly,
    the nearest even integer is chosen.

    Parameters
    ----------
    x : array_like
        Real valued input.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Rounded values.

    """
)
    scalar or ndarray
        # 接受标量或 ndarray 类型的输入参数，表示可以处理单个数值或多个数值数组。
        The nearest integers to the elements of `x`. The result is of
        # 返回与 `x` 中每个元素最接近的整数。结果是浮点类型，而不是整数类型。
        floating type, not integer type.
        # 返回结果是浮点类型，而不是整数类型。

    Examples
    --------
    >>> import scipy.special as sc

    It rounds to even.
    # 它是偶数舍入法（round to even）。

    >>> sc.round([0.5, 1.5])
    array([0., 2.])

    """
    # 此处是代码块的结束，包括了函数文档字符串的末尾引号和括号。
# 向给定的对象添加新的文档字符串，这里是一个函数 "shichi"
add_newdoc("shichi",
    r"""
    shichi(x, out=None)

    Hyperbolic sine and cosine integrals.

    The hyperbolic sine integral is

    .. math::

      \int_0^x \frac{\sinh{t}}{t}dt

    and the hyperbolic cosine integral is

    .. math::

      \gamma + \log(x) + \int_0^x \frac{\cosh{t} - 1}{t} dt

    where :math:`\gamma` is Euler's constant and :math:`\log` is the
    principal branch of the logarithm [1]_.

    Parameters
    ----------
    x : array_like
        实数或复数点，用于计算双曲正弦和余双曲正弦积分。
    out : tuple of ndarray, optional
        可选的输出数组，用于存储函数结果。

    Returns
    -------
    si : scalar or ndarray
        返回输入 x 的双曲正弦积分
    ci : scalar or ndarray
        返回输入 x 的双曲余弦积分

    See Also
    --------
    sici : 正弦和余弦积分。
    exp1 : 指数积分 E1。
    expi : 指数积分 Ei。

    Notes
    -----
    对于实数参数 x < 0，chi 是双曲余弦积分的实部。对于这样的点，
    chi(x) 和 chi(x + 0j) 之间相差一个因子 1j*pi。

    对于实数参数，该函数通过调用 Cephes 的 shichi 程序包来计算。
    对于复数参数，算法基于 Mpmath 的 shi 和 chi 程序包。

    References
    ----------
    .. [1] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.
           (See Section 5.2.)
    .. [2] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    .. [3] Fredrik Johansson and others.
           "mpmath: a Python library for arbitrary-precision floating-point
           arithmetic" (Version 0.19) http://mpmath.org/

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.special import shichi, sici

    `shichi` 接受实数或复数输入：

    >>> shichi(0.5)
    (0.5069967498196671, -0.05277684495649357)
    >>> shichi(0.5 + 2.5j)
    ((0.11772029666668238+1.831091777729851j),
     (0.29912435887648825+1.7395351121166562j))

    双曲正弦和余弦积分 Shi(z) 和 Chi(z) 与正弦和余弦积分 Si(z) 和 Ci(z) 相关：

    * Shi(z) = -i*Si(i*z)
    * Chi(z) = Ci(-i*z) + i*pi/2

    >>> z = 0.25 + 5j
    >>> shi, chi = shichi(z)
    >>> shi, -1j*sici(1j*z)[0]            # 应该相同。
    ((-0.04834719325101729+1.5469354086921228j),
     (-0.04834719325101729+1.5469354086921228j))
    >>> chi, sici(-1j*z)[1] + 1j*np.pi/2  # 应该相同。
    ((-0.19568708973868087+1.556276312103824j),
     (-0.19568708973868087+1.556276312103824j))

    在实轴上绘制函数的图像：

    >>> xp = np.geomspace(1e-8, 4.0, 250)
    >>> x = np.concatenate((-xp[::-1], xp))
    >>> shi, chi = shichi(x)
    """
    # 创建一个新的图形窗口，并返回一个包含图形和轴对象的元组
    fig, ax = plt.subplots()
    
    # 在轴对象上绘制以 x 为横轴，shi 为纵轴的线条，并添加标签 'Shi(x)'
    ax.plot(x, shi, label='Shi(x)')
    
    # 在同一个轴对象上绘制以 x 为横轴，chi 为纵轴的虚线，并添加标签 'Chi(x)'
    ax.plot(x, chi, '--', label='Chi(x)')
    
    # 设置 x 轴的标签文本为 'x'
    ax.set_xlabel('x')
    
    # 设置图形的标题为 'Hyperbolic Sine and Cosine Integrals'
    ax.set_title('Hyperbolic Sine and Cosine Integrals')
    
    # 在图例中添加标签，设置阴影效果为 True，设置图例框的透明度为 1，设置位置在右下角
    ax.legend(shadow=True, framealpha=1, loc='lower right')
    
    # 在轴对象上显示网格线
    ax.grid(True)
    
    # 显示绘制的图形
    plt.show()
# 向 scipy.special 模块添加新文档字符串
add_newdoc("sici",
    r"""
    sici(x, out=None)

    Sine and cosine integrals.

    The sine integral is

    .. math::

      \int_0^x \frac{\sin{t}}{t}dt

    and the cosine integral is

    .. math::

      \gamma + \log(x) + \int_0^x \frac{\cos{t} - 1}{t}dt

    where :math:`\gamma` is Euler's constant and :math:`\log` is the
    principal branch of the logarithm [1]_.

    Parameters
    ----------
    x : array_like
        Real or complex points at which to compute the sine and cosine
        integrals.
    out : tuple of ndarray, optional
        Optional output arrays for the function results

    Returns
    -------
    si : scalar or ndarray
        Sine integral at ``x``
    ci : scalar or ndarray
        Cosine integral at ``x``

    See Also
    --------
    shichi : Hyperbolic sine and cosine integrals.
    exp1 : Exponential integral E1.
    expi : Exponential integral Ei.

    Notes
    -----
    For real arguments with ``x < 0``, ``ci`` is the real part of the
    cosine integral. For such points ``ci(x)`` and ``ci(x + 0j)``
    differ by a factor of ``1j*pi``.

    For real arguments the function is computed by calling Cephes'
    [2]_ *sici* routine. For complex arguments the algorithm is based
    on Mpmath's [3]_ *si* and *ci* routines.

    References
    ----------
    .. [1] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.
           (See Section 5.2.)
    .. [2] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    .. [3] Fredrik Johansson and others.
           "mpmath: a Python library for arbitrary-precision floating-point
           arithmetic" (Version 0.19) http://mpmath.org/

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.special import sici, exp1

    `sici` accepts real or complex input:

    >>> sici(2.5)
    (1.7785201734438267, 0.2858711963653835)
    >>> sici(2.5 + 3j)
    ((4.505735874563953+0.06863305018999577j),
    (0.0793644206906966-2.935510262937543j))

    For z in the right half plane, the sine and cosine integrals are
    related to the exponential integral E1 (implemented in SciPy as
    `scipy.special.exp1`) by

    * Si(z) = (E1(i*z) - E1(-i*z))/2i + pi/2
    * Ci(z) = -(E1(i*z) + E1(-i*z))/2

    See [1]_ (equations 5.2.21 and 5.2.23).

    We can verify these relations:

    >>> z = 2 - 3j
    >>> sici(z)
    ((4.54751388956229-1.3991965806460565j),
    (1.408292501520851+2.9836177420296055j))

    >>> (exp1(1j*z) - exp1(-1j*z))/2j + np.pi/2  # Same as sine integral
    (4.54751388956229-1.3991965806460565j)

    >>> -(exp1(1j*z) + exp1(-1j*z))/2            # Same as cosine integral
    (1.408292501520851+2.9836177420296055j)

    Plot the functions evaluated on the real axis; the dotted horizontal
    lines are at pi/2 and -pi/2:

    >>> x = np.linspace(-16, 16, 150)
    # 调用函数 sici(x)，返回结果 si 和 ci
    >>> si, ci = sici(x)
    
    # 创建一个新的图形对象和一个包含单个子图的坐标轴对象
    >>> fig, ax = plt.subplots()
    
    # 在坐标轴上绘制 x 对应的 si 曲线，设置标签为 'Si(x)'
    >>> ax.plot(x, si, label='Si(x)')
    
    # 在坐标轴上绘制 x 对应的 ci 曲线，线型为虚线，设置标签为 'Ci(x)'
    >>> ax.plot(x, ci, '--', label='Ci(x)')
    
    # 在图上添加图例，设置阴影为 True，设置图例边框的透明度为 1，位置为左上角
    >>> ax.legend(shadow=True, framealpha=1, loc='upper left')
    
    # 设置 x 轴的标签为 'x'
    >>> ax.set_xlabel('x')
    
    # 设置图的标题为 'Sine and Cosine Integrals'
    >>> ax.set_title('Sine and Cosine Integrals')
    
    # 在坐标轴上添加一条 y 值为 np.pi/2 的水平虚线，线型为冒号，透明度为 0.5，颜色为黑色
    >>> ax.axhline(np.pi/2, linestyle=':', alpha=0.5, color='k')
    
    # 在坐标轴上添加一条 y 值为 -np.pi/2 的水平虚线，线型为冒号，透明度为 0.5，颜色为黑色
    >>> ax.axhline(-np.pi/2, linestyle=':', alpha=0.5, color='k')
    
    # 打开坐标轴的网格线显示
    >>> ax.grid(True)
    
    # 显示图形
    >>> plt.show()
add_newdoc("sindg",
    """
    sindg(x, out=None)

    计算角度为 `x` 的正弦值，角度以度数给出。

    Parameters
    ----------
    x : array_like
        角度，以度数给出。
    out : ndarray, optional
        可选参数，用于存储函数结果的输出数组。

    Returns
    -------
    scalar or ndarray
        输入角度的正弦值。

    See Also
    --------
    cosdg, tandg, cotdg

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    更精确地计算正弦值，比直接使用 sin 函数更为精确。

    >>> x = 180 * np.arange(3)
    >>> sc.sindg(x)
    array([ 0., -0.,  0.])
    >>> np.sin(x * np.pi / 180)
    array([ 0.0000000e+00,  1.2246468e-16, -2.4492936e-16])

    """)

add_newdoc("smirnov",
    r"""
    smirnov(n, d, out=None)

    Kolmogorov-Smirnov complementary cumulative distribution function

    返回 Kolmogorov-Smirnov 分布的补余累积分布函数（即生存函数），用于单侧等价性检验，
    测试经验分布函数与理论分布函数之间的差异。它表示在 `n` 个样本中，理论分布与经验分布
    最大差异大于 `d` 的概率。

    Parameters
    ----------
    n : int
      样本数量
    d : float array_like
      经验累积分布函数（ECDF）与目标累积分布函数之间的偏差
    out : ndarray, optional
        可选参数，用于存储函数结果的输出数组

    Returns
    -------
    scalar or ndarray
        smirnov(n, d) 的值，即 Prob(Dn+ >= d)（也即 Prob(Dn- >= d)）

    See Also
    --------
    smirnovi : 分布的逆生存函数
    scipy.stats.ksone : 作为连续分布的功能
    kolmogorov, kolmogi : 双侧分布函数

    Notes
    -----
    `smirnov` 在 Kolmogorov-Smirnov 拟合优度检验中被 `stats.kstest` 使用。由于历史原因，
    这个函数在 `scpy.special` 中暴露，但推荐的最准确的 CDF/SF/PDF/PPF/ISF 计算方法是使用
    `stats.ksone` 分布。

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import smirnov
    >>> from scipy.stats import norm

    显示样本大小为 5 时，至少有 0、0.5 和 1.0 差距的概率。

    >>> smirnov(5, [0, 0.5, 1.0])
    array([ 1.   ,  0.056,  0.   ])

    将大小为 5 的样本与 N(0, 1) 进行比较，即平均值为 0，标准差为 1 的标准正态分布。

    `x` 是样本。

    >>> x = np.array([-1.392, -0.135, 0.114, 0.190, 1.82])

    >>> target = norm(0, 1)
    >>> cdfs = target.cdf(x)
    >>> cdfs
    array([0.0819612 , 0.44630594, 0.5453811 , 0.57534543, 0.9656205 ])

    构建经验累积分布函数和 K-S 统计量（Dn+，Dn-，Dn）。

    >>> n = len(x)
    >>> ecdfs = np.arange(n+1, dtype=float)/n

    """
    >>> cols = np.column_stack([x, ecdfs[1:], cdfs, cdfs - ecdfs[:n],
    ...                        ecdfs[1:] - cdfs])
    # 将数组 x, ecdfs[1:], cdfs, cdfs - ecdfs[:n], ecdfs[1:] - cdfs 堆叠成一个二维数组 cols
    
    >>> with np.printoptions(precision=3):
    ...    print(cols)
    # 设置 numpy 打印选项，精度为三位小数，然后打印数组 cols
    
    [[ -1.392   0.2     0.082   0.082   0.118]
     [-0.135  0.4    0.446  0.246 -0.046]
     [ 0.114  0.6    0.545  0.145  0.055]
     [ 0.19   0.8    0.575 -0.025  0.225]
     [ 1.82   1.     0.966  0.166  0.034]]
    
    >>> gaps = cols[:, -2:]
    # 从二维数组 cols 中提取最后两列形成数组 gaps
    
    >>> Dnpm = np.max(gaps, axis=0)
    # 计算 gaps 数组中每列的最大值，存储在数组 Dnpm 中
    
    >>> print(f'Dn-={Dnpm[0]:f}, Dn+={Dnpm[1]:f}')
    # 打印 Dnpm 数组的第一个和第二个元素，格式化为浮点数
    
    Dn-=0.246306, Dn+=0.224655
    
    >>> probs = smirnov(n, Dnpm)
    # 调用函数 smirnov 计算给定 n 和 Dnpm 的概率
    
    >>> print(f'For a sample of size {n} drawn from N(0, 1):',
    ...       f' Smirnov n={n}: Prob(Dn- >= {Dnpm[0]:f}) = {probs[0]:.4f}',
    ...       f' Smirnov n={n}: Prob(Dn+ >= {Dnpm[1]:f}) = {probs[1]:.4f}',
    ...       sep='\n')
    # 打印关于 Smirnov 检验结果的信息，包括样本大小 n，以及 Dn- 和 Dn+ 的概率
    
    For a sample of size 5 drawn from N(0, 1):
     Smirnov n=5: Prob(Dn- >= 0.246306) = 0.4711
     Smirnov n=5: Prob(Dn+ >= 0.224655) = 0.5245
    
    Plot the empirical CDF and the standard normal CDF.
    
    >>> import matplotlib.pyplot as plt
    # 导入 matplotlib.pyplot 库，简称为 plt
    
    >>> plt.step(np.concatenate(([-2.5], x, [2.5])),
    ...          np.concatenate((ecdfs, [1])),
    ...          where='post', label='Empirical CDF')
    # 绘制阶梯图，表示经验累积分布函数（ECDF）
    
    >>> xx = np.linspace(-2.5, 2.5, 100)
    # 生成从 -2.5 到 2.5 的 100 个等间隔的点，存储在数组 xx 中
    
    >>> plt.plot(xx, target.cdf(xx), '--', label='CDF for N(0, 1)')
    # 绘制标准正态分布的累积分布函数（CDF）
    
    Add vertical lines marking Dn+ and Dn-.
    
    >>> iminus, iplus = np.argmax(gaps, axis=0)
    # 找到 gaps 中每列最大值的索引，iminus 是 Dn- 的索引，iplus 是 Dn+ 的索引
    
    >>> plt.vlines([x[iminus]], ecdfs[iminus], cdfs[iminus], color='r',
    ...            alpha=0.5, lw=4)
    # 在图上用红色绘制垂直线标记 Dn- 的位置
    
    >>> plt.vlines([x[iplus]], cdfs[iplus], ecdfs[iplus+1], color='m',
    ...            alpha=0.5, lw=4)
    # 在图上用品红色绘制垂直线标记 Dn+ 的位置
    
    >>> plt.grid(True)
    # 显示网格
    
    >>> plt.legend(framealpha=1, shadow=True)
    # 显示图例，设置透明度和阴影效果
    
    >>> plt.show()
    # 显示绘制的图形
# 向 `smirnovi` 函数添加新的文档字符串，描述其功能和用法
add_newdoc("smirnovi",
    """
    smirnovi(n, p, out=None)

    Inverse to `smirnov`

    Returns `d` such that ``smirnov(n, d) == p``, the critical value
    corresponding to `p`.

    Parameters
    ----------
    n : int
      Number of samples
    p : float array_like
        Probability
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        The value(s) of smirnovi(n, p), the critical values.

    See Also
    --------
    smirnov : The Survival Function (SF) for the distribution
    scipy.stats.ksone : Provides the functionality as a continuous distribution
    kolmogorov, kolmogi : Functions for the two-sided distribution
    scipy.stats.kstwobign : Two-sided Kolmogorov-Smirnov distribution, large n

    Notes
    -----
    `smirnov` is used by `stats.kstest` in the application of the
    Kolmogorov-Smirnov Goodness of Fit test. For historical reasons this
    function is exposed in `scpy.special`, but the recommended way to achieve
    the most accurate CDF/SF/PDF/PPF/ISF computations is to use the
    `stats.ksone` distribution.

    Examples
    --------
    >>> from scipy.special import smirnovi, smirnov

    >>> n = 24
    >>> deviations = [0.1, 0.2, 0.3]

    Use `smirnov` to compute the complementary CDF of the Smirnov
    distribution for the given number of samples and deviations.

    >>> p = smirnov(n, deviations)
    >>> p
    array([0.58105083, 0.12826832, 0.01032231])

    The inverse function ``smirnovi(n, p)`` returns ``deviations``.

    >>> smirnovi(n, p)
    array([0.1, 0.2, 0.3])

    """)

# 向 `_smirnovc` 函数添加新的文档字符串，指出其为内部函数，不建议使用
add_newdoc("_smirnovc",
    """
    _smirnovc(n, d)
     Internal function, do not use.
    """)

# 向 `_smirnovci` 函数添加新的文档字符串，指出其为内部函数，不建议使用
add_newdoc("_smirnovci",
    """
     Internal function, do not use.
    """)

# 向 `_smirnovp` 函数添加新的文档字符串，指出其为内部函数，不建议使用
add_newdoc("_smirnovp",
    """
    _smirnovp(n, p)
     Internal function, do not use.
    """)

# 向 `spence` 函数添加新的文档字符串，描述其功能、定义及使用
add_newdoc("spence",
    r"""
    spence(z, out=None)

    Spence's function, also known as the dilogarithm.

    It is defined to be

    .. math::
      \int_1^z \frac{\log(t)}{1 - t}dt

    for complex :math:`z`, where the contour of integration is taken
    to avoid the branch cut of the logarithm. Spence's function is
    analytic everywhere except the negative real axis where it has a
    branch cut.

    Parameters
    ----------
    z : array_like
        Points at which to evaluate Spence's function
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    s : scalar or ndarray
        Computed values of Spence's function

    Notes
    -----
    There is a different convention which defines Spence's function by
    the integral

    .. math::
      -\int_0^z \frac{\log(1 - t)}{t}dt;

    this is our ``spence(1 - z)``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import spence
    >>> import matplotlib.pyplot as plt

    The function is defined for complex inputs:
    """)
    # 对于输入为复数的情况，特别是在分支切割处（负实轴），函数返回具有正虚部的``z``的极限。
    # 例如，在以下示例中，注意输出的虚部对``z = -2``和``z = -2 - 1e-8j``的变化：

    >>> spence([-2 + 1e-8j, -2, -2 - 1e-8j])
    array([2.32018041-3.45139229j, 2.32018042-3.4513923j ,
           2.32018041+3.45139229j])

    # 对于实数输入落在分支切割处，函数返回``nan``：

    >>> spence(-1.5)
    nan

    # 验证一些特定值：``spence(0) = pi**2/6``, ``spence(1) = 0`` 和 ``spence(2) = -pi**2/12``。

    >>> spence([0, 1, 2])
    array([ 1.64493407,  0.        , -0.82246703])
    >>> np.pi**2/6, -np.pi**2/12
    (1.6449340668482264, -0.8224670334241132)

    # 验证恒等式::

        spence(z) + spence(1 - z) = pi**2/6 - log(z)*log(1 - z)

    >>> z = 3 + 4j
    >>> spence(z) + spence(1 - z)
    (-2.6523186143876067+1.8853470951513935j)
    >>> np.pi**2/6 - np.log(z)*np.log(1 - z)
    (-2.652318614387606+1.885347095151394j)

    # 绘制正实数输入情况下的函数图像。

    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(0, 6, 400)
    >>> ax.plot(x, spence(x))
    >>> ax.grid()
    >>> ax.set_xlabel('x')
    >>> ax.set_title('spence(x)')
    >>> plt.show()
add_newdoc(
    "stdtr",
    r"""
    stdtr(df, t, out=None)

    Student t distribution cumulative distribution function

    Returns the integral:

    .. math::
        \frac{\Gamma((df+1)/2)}{\sqrt{\pi df} \Gamma(df/2)}
        \int_{-\infty}^t (1+x^2/df)^{-(df+1)/2}\, dx

    Parameters
    ----------
    df : array_like
        Degrees of freedom
    t : array_like
        Upper bound of the integral
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Value of the Student t CDF at t

    See Also
    --------
    stdtridf : inverse of stdtr with respect to `df`
    stdtrit : inverse of stdtr with respect to `t`
    scipy.stats.t : student t distribution

    Notes
    -----
    The student t distribution is also available as `scipy.stats.t`.
    Calling `stdtr` directly can improve performance compared to the
    ``cdf`` method of `scipy.stats.t` (see last example below).

    Examples
    --------
    Calculate the function for ``df=3`` at ``t=1``.

    >>> import numpy as np
    >>> from scipy.special import stdtr
    >>> import matplotlib.pyplot as plt
    >>> stdtr(3, 1)
    0.8044988905221148

    Plot the function for three different degrees of freedom.

    >>> x = np.linspace(-10, 10, 1000)
    >>> fig, ax = plt.subplots()
    >>> parameters = [(1, "solid"), (3, "dashed"), (10, "dotted")]
    >>> for (df, linestyle) in parameters:
    ...     ax.plot(x, stdtr(df, x), ls=linestyle, label=f"$df={df}$")
    >>> ax.legend()
    >>> ax.set_title("Student t distribution cumulative distribution function")
    >>> plt.show()

    The function can be computed for several degrees of freedom at the same
    time by providing a NumPy array or list for `df`:

    >>> stdtr([1, 2, 3], 1)
    array([0.75      , 0.78867513, 0.80449889])

    It is possible to calculate the function at several points for several
    different degrees of freedom simultaneously by providing arrays for `df`
    and `t` with shapes compatible for broadcasting. Compute `stdtr` at
    4 points for 3 degrees of freedom resulting in an array of shape 3x4.

    >>> dfs = np.array([[1], [2], [3]])
    >>> t = np.array([2, 4, 6, 8])
    >>> dfs.shape, t.shape
    ((3, 1), (4,))

    >>> stdtr(dfs, t)
    array([[0.85241638, 0.92202087, 0.94743154, 0.96041658],
           [0.90824829, 0.97140452, 0.98666426, 0.99236596],
           [0.93033702, 0.98599577, 0.99536364, 0.99796171]])

    The t distribution is also available as `scipy.stats.t`. Calling `stdtr`
    directly can be much faster than calling the ``cdf`` method of
    `scipy.stats.t`. To get the same results, one must use the following
    parametrization: ``scipy.stats.t(df).cdf(x) = stdtr(df, x)``.

    >>> from scipy.stats import t
    >>> df, x = 3, 1
    >>> stdtr_result = stdtr(df, x)  # this can be faster than below
    >>> stats_result = t(df).cdf(x)
    >>> stats_result == stdtr_result  # test that results are equal

    """
)
    # 返回布尔值 True
    True
    # 多行字符串的开始，通常用作注释或文档字符串的一部分
    """
    """
# 将新文档添加到 "stdtridf" 函数中，提供了关于逆函数的文档说明
add_newdoc("stdtridf",
    """
    stdtridf(p, t, out=None)

    Inverse of `stdtr` vs df

    Returns the argument df such that stdtr(df, t) is equal to `p`.

    Parameters
    ----------
    p : array_like
        Probability
    t : array_like
        Upper bound of the integral
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    df : scalar or ndarray
        Value of `df` such that ``stdtr(df, t) == p``

    See Also
    --------
    stdtr : Student t CDF
    stdtrit : inverse of stdtr with respect to `t`
    scipy.stats.t : Student t distribution

    Examples
    --------
    Compute the student t cumulative distribution function for one
    parameter set.

    >>> from scipy.special import stdtr, stdtridf
    >>> df, x = 5, 2
    >>> cdf_value = stdtr(df, x)
    >>> cdf_value
    0.9490302605850709

    Verify that `stdtridf` recovers the original value for `df` given
    the CDF value and `x`.

    >>> stdtridf(cdf_value, x)
    5.0
    """)

# 将新文档添加到 "stdtrit" 函数中，提供了关于逆函数的文档说明
add_newdoc("stdtrit",
    """
    stdtrit(df, p, out=None)

    The `p`-th quantile of the student t distribution.

    This function is the inverse of the student t distribution cumulative
    distribution function (CDF), returning `t` such that `stdtr(df, t) = p`.

    Returns the argument `t` such that stdtr(df, t) is equal to `p`.

    Parameters
    ----------
    df : array_like
        Degrees of freedom
    p : array_like
        Probability
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    t : scalar or ndarray
        Value of `t` such that ``stdtr(df, t) == p``

    See Also
    --------
    stdtr : Student t CDF
    stdtridf : inverse of stdtr with respect to `df`
    scipy.stats.t : Student t distribution

    Notes
    -----
    The student t distribution is also available as `scipy.stats.t`. Calling
    `stdtrit` directly can improve performance compared to the ``ppf``
    method of `scipy.stats.t` (see last example below).

    Examples
    --------
    `stdtrit` represents the inverse of the student t distribution CDF which
    is available as `stdtr`. Here, we calculate the CDF for ``df`` at
    ``x=1``. `stdtrit` then returns ``1`` up to floating point errors
    given the same value for `df` and the computed CDF value.

    >>> import numpy as np
    >>> from scipy.special import stdtr, stdtrit
    >>> import matplotlib.pyplot as plt
    >>> df = 3
    >>> x = 1
    >>> cdf_value = stdtr(df, x)
    >>> stdtrit(df, cdf_value)
    0.9999999994418539

    Plot the function for three different degrees of freedom.

    >>> x = np.linspace(0, 1, 1000)
    >>> parameters = [(1, "solid"), (2, "dashed"), (5, "dotted")]
    >>> fig, ax = plt.subplots()
    >>> for (df, linestyle) in parameters:
    ...     ax.plot(x, stdtrit(df, x), ls=linestyle, label=f"$df={df}$")
    >>> ax.legend()
    >>> ax.set_ylim(-10, 10)
    """)
    # 设置图表的标题为 "Student t distribution quantile function"
    ax.set_title("Student t distribution quantile function")
    # 显示绘制的图表
    plt.show()

    # 使用 NumPy 数组或列表作为 `df` 参数，可以同时计算多个自由度的学生 t 分布的分位数函数值
    stdtrit([1, 2, 3], 0.7)
    # 返回结果为一个数组，对应不同自由度的计算结果
    array([0.72654253, 0.6172134 , 0.58438973])

    # 如果要同时计算多个自由度和多个概率水平 `p` 的分位数函数值，可以提供形状兼容的数组 `df` 和 `p`
    # 这里演示了在 3 个自由度和 4 个概率水平下的计算，结果是一个 3x4 的数组
    dfs = np.array([[1], [2], [3]])
    p = np.array([0.2, 0.4, 0.7, 0.8])
    # 打印出数组 `dfs` 和 `p` 的形状
    dfs.shape, p.shape
    ((3, 1), (4,))

    # 调用 `stdtrit` 计算多个自由度和多个概率水平下的分位数函数值
    stdtrit(dfs, p)
    # 返回一个 3x4 的数组，包含了在不同自由度和概率水平下的计算结果

    # 学生 t 分布也可以通过 `scipy.stats.t` 访问，直接调用 `stdtrit` 可能比调用 `scipy.stats.t` 的 `ppf` 方法更快
    # 为了获得相同的结果，需要使用以下参数化方式：`scipy.stats.t(df).ppf(x) = stdtrit(df, x)`
    from scipy.stats import t
    df, x = 3, 0.5
    # 使用 `stdtrit` 计算给定自由度和概率水平下的分位数函数值
    stdtrit_result = stdtrit(df, x)
    # 使用 `scipy.stats.t` 的 `ppf` 方法计算相同自由度和概率水平下的分位数函数值
    stats_result = t(df).ppf(x)
    # 检查两种方法计算的结果是否相等
    stats_result == stdtrit_result  # 返回 True 表示结果相等
# 将新的文档添加到特定的模块中
add_newdoc("struve",
    r"""
    struve(v, x, out=None)

    Struve function.

    Return the value of the Struve function of order `v` at `x`.  The Struve
    function is defined as,

    .. math::
        H_v(x) = (z/2)^{v + 1} \sum_{n=0}^\infty
        \frac{(-1)^n (z/2)^{2n}}{\Gamma(n + \frac{3}{2}) \Gamma(n + v + \frac{3}{2})},

    where :math:`\Gamma` is the gamma function.

    Parameters
    ----------
    v : array_like
        Order of the Struve function (float).
    x : array_like
        Argument of the Struve function (float; must be positive unless `v` is
        an integer).
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    H : scalar or ndarray
        Value of the Struve function of order `v` at `x`.

    See Also
    --------
    modstruve: Modified Struve function

    Notes
    -----
    Three methods discussed in [1]_ are used to evaluate the Struve function:

    - power series
    - expansion in Bessel functions (if :math:`|z| < |v| + 20`)
    - asymptotic large-z expansion (if :math:`z \geq 0.7v + 12`)

    Rounding errors are estimated based on the largest terms in the sums, and
    the result associated with the smallest error is returned.

    References
    ----------
    .. [1] NIST Digital Library of Mathematical Functions
           https://dlmf.nist.gov/11

    Examples
    --------
    Calculate the Struve function of order 1 at 2.

    >>> import numpy as np
    >>> from scipy.special import struve
    >>> import matplotlib.pyplot as plt
    >>> struve(1, 2.)
    0.6467637282835622

    Calculate the Struve function at 2 for orders 1, 2 and 3 by providing
    a list for the order parameter `v`.

    >>> struve([1, 2, 3], 2.)
    array([0.64676373, 0.28031806, 0.08363767])

    Calculate the Struve function of order 1 for several points by providing
    an array for `x`.

    >>> points = np.array([2., 5., 8.])
    >>> struve(1, points)
    array([0.64676373, 0.80781195, 0.48811605])

    Compute the Struve function for several orders at several points by
    providing arrays for `v` and `z`. The arrays have to be broadcastable
    to the correct shapes.

    >>> orders = np.array([[1], [2], [3]])
    >>> points.shape, orders.shape
    ((3,), (3, 1))

    >>> struve(orders, points)
    array([[0.64676373, 0.80781195, 0.48811605],
           [0.28031806, 1.56937455, 1.51769363],
           [0.08363767, 1.50872065, 2.98697513]])

    Plot the Struve functions of order 0 to 3 from -10 to 10.

    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(-10., 10., 1000)
    >>> for i in range(4):
    ...     ax.plot(x, struve(i, x), label=f'$H_{i!r}$')
    >>> ax.legend(ncol=2)
    >>> ax.set_xlim(-10, 10)
    >>> ax.set_title(r"Struve functions $H_{\nu}$")
    >>> plt.show()
    """)

# 计算角度为x度的正切值
add_newdoc("tandg",
    """
    tandg(x, out=None)

    Tangent of angle `x` given in degrees.

    Parameters
    ----------
    x : array_like
        Angle in degrees for which to compute the tangent.

    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    tangent : scalar or ndarray
        Tangent of the input angle `x` in degrees.

    Notes
    -----
    The tangent function is defined as the ratio of the sine to the cosine
    of the input angle.

    Examples
    --------
    Calculate the tangent of 45 degrees.

    >>> from scipy.special import tandg
    >>> tandg(45)
    1.0

    Calculate the tangent for an array of angles in degrees.

    >>> angles = [0, 30, 60, 90]
    >>> tandg(angles)
    array([ 0.        ,  0.57735027,  1.73205081,         inf])
    """)
    x : array_like
        Angle, given in degrees.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Tangent at the input.

    See Also
    --------
    sindg, cosdg, cotdg

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    It is more accurate than using tangent directly.

    >>> x = 180 * np.arange(3)
    调用 scipy.special 模块中的 tandg 函数计算给定角度的正切值
    >>> sc.tandg(x)
    array([0., 0., 0.])
    使用 numpy 库计算给定角度的正切值（使用弧度制）
    >>> np.tan(x * np.pi / 180)
    array([ 0.0000000e+00, -1.2246468e-16, -2.4492936e-16])

    """
# 向指定模块添加新的文档字符串，用于描述 Tukey lambda 分布的累积分布函数
add_newdoc(
    "tklmbda",
    r"""
    tklmbda(x, lmbda, out=None)

    Cumulative distribution function of the Tukey lambda distribution.

    Parameters
    ----------
    x, lmbda : array_like
        Parameters
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    cdf : scalar or ndarray
        Value of the Tukey lambda CDF

    See Also
    --------
    scipy.stats.tukeylambda : Tukey lambda distribution

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.special import tklmbda, expit

    Compute the cumulative distribution function (CDF) of the Tukey lambda
    distribution at several ``x`` values for `lmbda` = -1.5.

    >>> x = np.linspace(-2, 2, 9)
    >>> x
    array([-2. , -1.5, -1. , -0.5,  0. ,  0.5,  1. ,  1.5,  2. ])
    >>> tklmbda(x, -1.5)
    array([0.34688734, 0.3786554 , 0.41528805, 0.45629737, 0.5       ,
           0.54370263, 0.58471195, 0.6213446 , 0.65311266])

    When `lmbda` is 0, the function is the logistic sigmoid function,
    which is implemented in `scipy.special` as `expit`.

    >>> tklmbda(x, 0)
    array([0.11920292, 0.18242552, 0.26894142, 0.37754067, 0.5       ,
           0.62245933, 0.73105858, 0.81757448, 0.88079708])
    >>> expit(x)
    array([0.11920292, 0.18242552, 0.26894142, 0.37754067, 0.5       ,
           0.62245933, 0.73105858, 0.81757448, 0.88079708])

    When `lmbda` is 1, the Tukey lambda distribution is uniform on the
    interval [-1, 1], so the CDF increases linearly.

    >>> t = np.linspace(-1, 1, 9)
    >>> tklmbda(t, 1)
    array([0.   , 0.125, 0.25 , 0.375, 0.5  , 0.625, 0.75 , 0.875, 1.   ])

    In the following, we generate plots for several values of `lmbda`.

    The first figure shows graphs for `lmbda` <= 0.

    >>> styles = ['-', '-.', '--', ':']
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(-12, 12, 500)
    >>> for k, lmbda in enumerate([-1.0, -0.5, 0.0]):
    ...     y = tklmbda(x, lmbda)
    ...     ax.plot(x, y, styles[k], label=rf'$\lambda$ = {lmbda:-4.1f}')

    >>> ax.set_title(r'tklmbda(x, $\lambda$)')
    >>> ax.set_label('x')
    >>> ax.legend(framealpha=1, shadow=True)
    >>> ax.grid(True)

    The second figure shows graphs for `lmbda` > 0.  The dots in the
    graphs show the bounds of the support of the distribution.

    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(-4.2, 4.2, 500)
    >>> lmbdas = [0.25, 0.5, 1.0, 1.5]
    >>> for k, lmbda in enumerate(lmbdas):
    ...     y = tklmbda(x, lmbda)
    ...     ax.plot(x, y, styles[k], label=fr'$\lambda$ = {lmbda}')

    >>> ax.set_prop_cycle(None)
    >>> for l
    # 导入 scipy.stats 中的 tukeylambda 模块，用于处理 Tukey lambda 分布的统计功能
    from scipy.stats import tukeylambda
    # 导入 numpy 库，用于生成等间隔的数组
    import numpy as np
    
    # 创建一个等间距的包含9个元素的数组，范围从-2到2
    x = np.linspace(-2, 2, 9)
    
    # 使用 tukeylambda 模块中的 cdf 方法计算 Tukey lambda 分布的累积分布函数值，并存储在数组中
    tukeylambda.cdf(x, -0.5)
    array([0.21995157, 0.27093858, 0.33541677, 0.41328161, 0.5       ,
           0.58671839, 0.66458323, 0.72906142, 0.78004843])
    
    # tklmbda 是一个未定义的变量或函数，应该是指示例中自定义的函数或变量，但在给定的代码段中无法确定其定义和用途
    # 因此，这行代码可能需要进一步的上下文来理解其含义和用途
    
    # 上述例子说明 tukeylambda 模块提供了 Tukey lambda 分布的多种功能，包括概率密度函数 (pdf) 和累积分布函数的反函数 (ppf)，
    # 因此在处理 Tukey lambda 分布时，tukeylambda 模块更为全面和实用。而 tklmbda 的主要优势是比 tukeylambda.cdf 方法更快。
# 给指定函数添加新的文档字符串和描述
add_newdoc("wofz",
    """
    wofz(z, out=None)

    Faddeeva function

    返回复数参数 z 的 Faddeeva 函数值::

        exp(-z**2) * erfc(-i*z)

    Parameters
    ----------
    z : array_like
        复数参数
    out : ndarray, optional
        可选的输出数组，用于存储函数结果

    Returns
    -------
    scalar or ndarray
        Faddeeva 函数的值

    See Also
    --------
    dawsn, erf, erfc, erfcx, erfi

    References
    ----------
    .. [1] Steven G. Johnson, Faddeeva W function implementation.
       http://ab-initio.mit.edu/Faddeeva

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import special
    >>> import matplotlib.pyplot as plt

    >>> x = np.linspace(-3, 3)
    >>> z = special.wofz(x)

    >>> plt.plot(x, z.real, label='wofz(x).real')
    >>> plt.plot(x, z.imag, label='wofz(x).imag')
    >>> plt.xlabel('$x$')
    >>> plt.legend(framealpha=1, shadow=True)
    >>> plt.grid(alpha=0.25)
    >>> plt.show()

    """)

# 给指定函数添加新的文档字符串和描述
add_newdoc("xlogy",
    """
    xlogy(x, y, out=None)

    计算 ``x*log(y)``，当 ``x = 0`` 时结果为 0。

    Parameters
    ----------
    x : array_like
        乘数
    y : array_like
        参数
    out : ndarray, optional
        可选的输出数组，用于存储函数结果

    Returns
    -------
    z : scalar or ndarray
        计算得到的 x*log(y)

    Notes
    -----
    计算中使用的对数函数是自然对数。

    .. versionadded:: 0.13.0

    Examples
    --------
    我们可以使用这个函数来计算二元逻辑损失，也称为二元交叉熵。这个损失函数用于二分类问题，定义如下：

    .. math::
        L = 1/n * \\sum_{i=0}^n -(y_i*log(y\\_pred_i) + (1-y_i)*log(1-y\\_pred_i))

    其中，`x` 和 `y` 分别是 y 和 y_pred 的参数。y 是实际标签的数组，可以是 0 或 1。y_pred 是相对于正类（1）的预测概率数组。

    >>> import numpy as np
    >>> from scipy.special import xlogy
    >>> y = np.array([0, 1, 0, 1, 1, 0])
    >>> y_pred = np.array([0.3, 0.8, 0.4, 0.7, 0.9, 0.2])
    >>> n = len(y)
    >>> loss = -(xlogy(y, y_pred) + xlogy(1 - y, 1 - y_pred)).sum()
    >>> loss /= n
    >>> loss
    0.29597052165495025

    损失值越低通常越好，因为它表示预测与实际标签的相似性。在这个例子中，由于我们的预测概率接近实际标签，得到了一个相对较低且合适的总体损失值。

    """)

# 给指定函数添加新的文档字符串和描述
add_newdoc("xlog1py",
    """
    xlog1py(x, y, out=None)

    计算 ``x*log1p(y)``，当 ``x = 0`` 时结果为 0。

    Parameters
    ----------
    x : array_like
        乘数
    y : array_like
        参数
    out : ndarray, optional
        可选的输出数组，用于存储函数结果
    """)
    # 输出参数，可选的用于存放函数结果的数组
    out : ndarray, optional
    # 返回值
    Returns
    -------
    # 计算结果 x*log1p(y) 的标量或数组
    z : scalar or ndarray
    # 注意事项部分的说明
    Notes
    -----
    # 添加于版本 0.13.0 中
    .. versionadded:: 0.13.0

    # 示例部分开始
    Examples
    --------
    # 展示如何使用该函数计算几何离散随机变量的对数概率质量函数
    This example shows how the function can be used to calculate the log of
    the probability mass function for a geometric discrete random variable.
    # 几何分布的概率质量函数定义如下：
    The probability mass function of the geometric distribution is defined
    as follows:
    # 其中 p 是单次成功的概率
    .. math:: f(k) = (1-p)^{k-1} p
    # 其中 1-p 是单次失败的概率，k 是获得第一个成功所需的试验次数

    # 引入必要的库
    >>> import numpy as np
    >>> from scipy.special import xlog1py
    # 设定成功的概率
    >>> p = 0.5
    # 设定试验次数
    >>> k = 100
    # 计算几何分布的概率质量函数
    >>> _pmf = np.power(1 - p, k - 1) * p
    >>> _pmf
    7.888609052210118e-31

    # 当 k 较大时，概率质量函数的值可以变得非常小。
    # 在这种情况下，取概率质量函数的对数更合适，因为对数函数可以将值转换到更适合处理的范围内。
    >>> _log_pmf = xlog1py(k - 1, -p) + np.log(p)
    >>> _log_pmf
    -69.31471805599453

    # 通过取对数概率质量函数的指数，可以确认得到接近原始概率质量函数值的结果。
    >>> _orig_pmf = np.exp(_log_pmf)
    >>> np.isclose(_pmf, _orig_pmf)
    True

    # 示例结束
    """
# 将函数文档添加到 "y0" 函数的文档字符串中
add_newdoc("y0",
    r"""
    y0(x, out=None)

    Bessel function of the second kind of order 0.

    Parameters
    ----------
    x : array_like
        Argument (float).
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    Y : scalar or ndarray
        Value of the Bessel function of the second kind of order 0 at `x`.

    See Also
    --------
    j0: Bessel function of the first kind of order 0
    yv: Bessel function of the first kind

    Notes
    -----
    The domain is divided into the intervals [0, 5] and (5, infinity). In the
    first interval a rational approximation :math:`R(x)` is employed to
    compute,

    .. math::

        Y_0(x) = R(x) + \frac{2 \log(x) J_0(x)}{\pi},

    where :math:`J_0` is the Bessel function of the first kind of order 0.

    In the second interval, the Hankel asymptotic expansion is employed with
    two rational functions of degree 6/6 and 7/7.

    This function is a wrapper for the Cephes [1]_ routine `y0`.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    Calculate the function at one point:

    >>> from scipy.special import y0
    >>> y0(1.)
    0.08825696421567697

    Calculate at several points:

    >>> import numpy as np
    >>> y0(np.array([0.5, 2., 3.]))
    array([-0.44451873,  0.51037567,  0.37685001])

    Plot the function from 0 to 10.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(0., 10., 1000)
    >>> y = y0(x)
    >>> ax.plot(x, y)
    >>> plt.show()

    """)

# 将函数文档添加到 "y1" 函数的文档字符串中
add_newdoc("y1",
    """
    y1(x, out=None)

    Bessel function of the second kind of order 1.

    Parameters
    ----------
    x : array_like
        Argument (float).
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    Y : scalar or ndarray
        Value of the Bessel function of the second kind of order 1 at `x`.

    See Also
    --------
    j1: Bessel function of the first kind of order 1
    yn: Bessel function of the second kind
    yv: Bessel function of the second kind

    Notes
    -----
    The domain is divided into the intervals [0, 8] and (8, infinity). In the
    first interval a 25 term Chebyshev expansion is used, and computing
    :math:`J_1` (the Bessel function of the first kind) is required. In the
    second, the asymptotic trigonometric representation is employed using two
    rational functions of degree 5/5.

    This function is a wrapper for the Cephes [1]_ routine `y1`.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    Calculate the function at one point:

    >>> from scipy.special import y1
    >>> y1(1.)
    -0.7812128213002888

    Calculate at several points:

    >>> import numpy as np
    """
    )
    # 调用函数 y1，并传入一个 NumPy 数组作为参数，返回一个新的数组作为结果
    >>> y1(np.array([0.5, 2., 3.]))
    # 返回结果如下：
    array([-1.47147239, -0.10703243,  0.32467442])
    
    # 绘制函数的图像，横坐标范围从 0 到 10
    >>> import matplotlib.pyplot as plt
    # 创建一个新的图形对象和一个包含单个子图的坐标轴对象
    >>> fig, ax = plt.subplots()
    # 生成一个包含 1000 个均匀分布的点的 NumPy 数组，范围从 0 到 10
    >>> x = np.linspace(0., 10., 1000)
    # 计算函数 y1 在 x 数组上的值
    >>> y = y1(x)
    # 在坐标轴上绘制 x 和 y 数组的图像
    >>> ax.plot(x, y)
    # 显示图形
    >>> plt.show()
# 添加新的文档字符串给 `yn` 函数
add_newdoc("yn",
    r"""
    yn(n, x, out=None)

    Bessel function of the second kind of integer order and real argument.

    Parameters
    ----------
    n : array_like
        Order (integer).
        订单（整数）。
    x : array_like
        Argument (float).
        参数（浮点数）。
    out : ndarray, optional
        Optional output array for the function results
        可选的输出数组，用于存放函数的结果。

    Returns
    -------
    Y : scalar or ndarray
        Value of the Bessel function, :math:`Y_n(x)`.
        贝塞尔函数的值，:math:`Y_n(x)`。

    See Also
    --------
    yv : For real order and real or complex argument.
    y0: faster implementation of this function for order 0
    y1: faster implementation of this function for order 1
    有关实数阶数和实数或复数参数的贝塞尔函数 `yv`。
    阶数为 0 时的更快实现 `y0`。
    阶数为 1 时的更快实现 `y1`。

    Notes
    -----
    Wrapper for the Cephes [1]_ routine `yn`.
    使用 Cephes [1]_ 库中的 `yn` 函数进行封装。

    The function is evaluated by forward recurrence on `n`, starting with
    values computed by the Cephes routines `y0` and `y1`. If ``n = 0`` or 1,
    the routine for `y0` or `y1` is called directly.
    函数通过对 `n` 进行正向递归来进行计算，从由 Cephes 中的 `y0` 和 `y1` 计算的值开始。
    如果 `n = 0` 或 `n = 1`，则直接调用 `y0` 或 `y1` 的计算例程。

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    参考文献：Cephes 数学函数库，http://www.netlib.org/cephes/

    Examples
    --------
    Evaluate the function of order 0 at one point.
    在一个点上评估阶数为 0 的函数。

    >>> from scipy.special import yn
    >>> yn(0, 1.)
    0.08825696421567697

    Evaluate the function at one point for different orders.
    在不同阶数的一个点上评估函数。

    >>> yn(0, 1.), yn(1, 1.), yn(2, 1.)
    (0.08825696421567697, -0.7812128213002888, -1.6506826068162546)

    The evaluation for different orders can be carried out in one call by
    providing a list or NumPy array as argument for the `v` parameter:
    可以通过为 `v` 参数提供列表或 NumPy 数组来一次性评估不同阶数。

    >>> yn([0, 1, 2], 1.)
    array([ 0.08825696, -0.78121282, -1.65068261])

    Evaluate the function at several points for order 0 by providing an
    array for `z`.
    通过为 `z` 提供数组，在多个点上评估阶数为 0 的函数。

    >>> import numpy as np
    >>> points = np.array([0.5, 3., 8.])
    >>> yn(0, points)
    array([-0.44451873,  0.37685001,  0.22352149])

    If `z` is an array, the order parameter `v` must be broadcastable to
    the correct shape if different orders shall be computed in one call.
    如果 `z` 是一个数组，则阶数参数 `v` 必须能够广播到正确的形状，以便在一次调用中计算不同的阶数。
    To calculate the orders 0 and 1 for an 1D array:
    要计算一个 1 维数组的阶数 0 和 1：

    >>> orders = np.array([[0], [1]])
    >>> orders.shape
    (2, 1)

    >>> yn(orders, points)
    array([[-0.44451873,  0.37685001,  0.22352149],
           [-1.47147239,  0.32467442, -0.15806046]])

    Plot the functions of order 0 to 3 from 0 to 10.
    绘制阶数从 0 到 3 的函数在 0 到 10 之间的图像。

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(0., 10., 1000)
    >>> for i in range(4):
    ...     ax.plot(x, yn(i, x), label=f'$Y_{i!r}$')
    >>> ax.set_ylim(-3, 1)
    >>> ax.legend()
    >>> plt.show()
    """)

# 添加新的文档字符串给 `yv` 函数，因为此处的代码未给出完整，注释也不完整，应继续添加。
add_newdoc("yv",
    r"""
    yv(v, z, out=None)

    Bessel function of the second kind of real order and complex argument.

    Parameters
    ----------
    v : array_like
        Order (float).
        阶数（浮点数）。
    z : array_like
        Argument (float or complex).
        参数（浮点数或复数）。
    out : ndarray, optional
        Optional output array for the function results
        可选的输出数组，用于存放函数的结果。

    Returns
    -------

    """)
    # Y是标量或者数组，表示第二类贝塞尔函数Y_v(x)的值。

    # 参见
    # --------
    # yve : 去掉领先指数行为的Y_v。
    # y0: 用于0阶函数更快的实现。
    # y1: 用于1阶函数更快的实现。

    # 注意
    # -----
    # 对于正的v值，使用AMOS [1]_中的zbesy例程进行计算，该例程利用汉克尔贝塞尔函数
    # H_v^{(1)} 和 H_v^{(2)} 的关系进行计算。
    # Y_v(z) = \frac{1}{2\imath} (H_v^{(1)} - H_v^{(2)})。
    
    # 对于负的v值，使用以下公式：
    # Y_{-v}(z) = Y_v(z) \cos(\pi v) + J_v(z) \sin(\pi v)，
    # 其中J_v(z)是第一类贝塞尔函数，使用AMOS例程zbesj计算。
    # 注意，当v为整数时第二项是精确为零的；为了提高精度，在v为floor(v)的情况下显式省略第二项。

    # 参考文献
    # ----------
    # .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
    #        of a Complex Argument and Nonnegative Order",
    #        http://netlib.org/amos/

    # 示例
    # --------
    # 在一个点上评估0阶函数。

    # >>> from scipy.special import yv
    # >>> yv(0, 1.)
    # 0.088256964215677

    # 在不同阶数的一个点上评估函数。

    # >>> yv(0, 1.), yv(1, 1.), yv(1.5, 1.)
    # (0.088256964215677, -0.7812128213002889, -1.102495575160179)

    # 通过为参数`v`提供一个列表或NumPy数组，可以在一次调用中进行不同阶数的评估。

    # >>> yv([0, 1, 1.5], 1.)
    # array([ 0.08825696, -0.78121282, -1.10249558])

    # 通过为`z`提供一个数组，在0阶数上评估几个点。

    # >>> import numpy as np
    # >>> points = np.array([0.5, 3., 8.])
    # >>> yv(0, points)
    # array([-0.44451873,  0.37685001,  0.22352149])

    # 如果`z`是一个数组，阶数参数`v`必须广播到正确的形状，以便在一次调用中计算不同的阶数。
    # 例如，计算一个1维数组的阶数为0和1的情况：

    # >>> orders = np.array([[0], [1]])
    # >>> orders.shape
    # (2, 1)

    # >>> yv(orders, points)
    # array([[-0.44451873,  0.37685001,  0.22352149],
    #        [-1.47147239,  0.32467442, -0.15806046]])

    # 绘制从0到10的0到3阶函数。

    # >>> import matplotlib.pyplot as plt
    # >>> fig, ax = plt.subplots()
    # >>> x = np.linspace(0., 10., 1000)
    # >>> for i in range(4):
    # ...     ax.plot(x, yv(i, x), label=f'$Y_{i!r}$')
    # >>> ax.set_ylim(-3, 1)
    # >>> ax.legend()
    # >>> plt.show()
# 添加新的文档字符串到模块 "yve" 中，描述函数 yve 的功能和用法
add_newdoc("yve",
    r"""
    yve(v, z, out=None)

    Exponentially scaled Bessel function of the second kind of real order.

    Returns the exponentially scaled Bessel function of the second
    kind of real order `v` at complex `z`::

        yve(v, z) = yv(v, z) * exp(-abs(z.imag))

    Parameters
    ----------
    v : array_like
        Order (float).
    z : array_like
        Argument (float or complex).
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    Y : scalar or ndarray
        Value of the exponentially scaled Bessel function.

    See Also
    --------
    yv: Unscaled Bessel function of the second kind of real order.

    Notes
    -----
    For positive `v` values, the computation is carried out using the
    AMOS [1]_ `zbesy` routine, which exploits the connection to the Hankel
    Bessel functions :math:`H_v^{(1)}` and :math:`H_v^{(2)}`,

    .. math:: Y_v(z) = \frac{1}{2\imath} (H_v^{(1)} - H_v^{(2)}).

    For negative `v` values the formula,

    .. math:: Y_{-v}(z) = Y_v(z) \cos(\pi v) + J_v(z) \sin(\pi v)

    is used, where :math:`J_v(z)` is the Bessel function of the first kind,
    computed using the AMOS routine `zbesj`.  Note that the second term is
    exactly zero for integer `v`; to improve accuracy the second term is
    explicitly omitted for `v` values such that `v = floor(v)`.

    Exponentially scaled Bessel functions are useful for large `z`:
    for these, the unscaled Bessel functions can easily under-or overflow.

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/

    Examples
    --------
    Compare the output of `yv` and `yve` for large complex arguments for `z`
    by computing their values for order ``v=1`` at ``z=1000j``. We see that
    `yv` returns nan but `yve` returns a finite number:

    >>> import numpy as np
    >>> from scipy.special import yv, yve
    >>> v = 1
    >>> z = 1000j
    >>> yv(v, z), yve(v, z)
    ((nan+nanj), (-0.012610930256928629+7.721967686709076e-19j))

    For real arguments for `z`, `yve` returns the same as `yv` up to
    floating point errors.

    >>> v, z = 1, 1000
    >>> yv(v, z), yve(v, z)
    (-0.02478433129235178, -0.02478433129235179)

    The function can be evaluated for several orders at the same time by
    providing a list or NumPy array for `v`:

    >>> yve([1, 2, 3], 1j)
    array([-0.20791042+0.14096627j,  0.38053618-0.04993878j,
           0.00815531-1.66311097j])

    In the same way, the function can be evaluated at several points in one
    call by providing a list or NumPy array for `z`:

    >>> yve(1, np.array([1j, 2j, 3j]))
    array([-0.20791042+0.14096627j, -0.21526929+0.01205044j,
           -0.19682671+0.00127278j])

    It is also possible to evaluate several orders at several points
    """
    at the same time by providing arrays for `v` and `z` with
    broadcasting compatible shapes. Compute `yve` for two different orders
    `v` and three points `z` resulting in a 2x3 array.

    >>> v = np.array([[1], [2]])
    >>> z = np.array([3j, 4j, 5j])
    >>> v.shape, z.shape
    ((2, 1), (3,))



    # 导入 numpy 库并创建示例数组 `v` 和 `z`
    import numpy as np
    
    # 创建 `v` 数组，形状为 (2, 1)，包含两个行和一个列的数组
    v = np.array([[1], [2]])
    
    # 创建 `z` 数组，形状为 (3,)，包含三个虚数
    z = np.array([3j, 4j, 5j])
    
    # 打印 `v` 和 `z` 的形状信息，以确认广播兼容性
    v.shape, z.shape



    >>> yve(v, z)
    array([[-1.96826713e-01+1.27277544e-03j, -1.78750840e-01+1.45558819e-04j,
            -1.63972267e-01+1.73494110e-05j],
           [1.94960056e-03-1.11782545e-01j,  2.02902325e-04-1.17626501e-01j,
            2.27727687e-05-1.17951906e-01j]])



    # 调用 `yve` 函数计算给定 `v` 和 `z` 的结果，并返回一个形状为 (2, 3) 的数组
    yve(v, z)
# 在指定模块或对象上添加新的文档字符串
add_newdoc("zetac",
    """
    zetac(x, out=None)

    Riemann zeta function minus 1.

    This function is defined as

    .. math:: \\zeta(x) = \\sum_{k=2}^{\\infty} 1 / k^x,

    where ``x > 1``.  For ``x < 1`` the analytic continuation is
    computed. For more information on the Riemann zeta function, see
    [dlmf]_.

    Parameters
    ----------
    x : array_like of float
        Values at which to compute zeta(x) - 1 (must be real).
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of zeta(x) - 1.

    See Also
    --------
    zeta

    References
    ----------
    .. [dlmf] NIST Digital Library of Mathematical Functions
              https://dlmf.nist.gov/25

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import zetac, zeta

    Some special values:

    >>> zetac(2), np.pi**2/6 - 1
    (0.64493406684822641, 0.6449340668482264)

    >>> zetac(-1), -1.0/12 - 1
    (-1.0833333333333333, -1.0833333333333333)

    Compare ``zetac(x)`` to ``zeta(x) - 1`` for large `x`:

    >>> zetac(60), zeta(60) - 1
    (8.673617380119933e-19, 0.0)
    """)

# 内部函数，建议使用 `zeta` 替代
add_newdoc("_riemann_zeta",
    """
    Internal function, use `zeta` instead.
    """)

# 内部函数，用于测试 `struve` 和 `modstruve` 的大 Z 渐近展开
add_newdoc("_struve_asymp_large_z",
    """
    _struve_asymp_large_z(v, z, is_h)

    Internal function for testing `struve` & `modstruve`

    Evaluates using asymptotic expansion

    Returns
    -------
    v, err
    """)

# 内部函数，用于测试 `struve` 和 `modstruve` 的幂级数
add_newdoc("_struve_power_series",
    """
    _struve_power_series(v, z, is_h)

    Internal function for testing `struve` & `modstruve`

    Evaluates using power series

    Returns
    -------
    v, err
    """)

# 内部函数，用于测试 `struve` 和 `modstruve` 的贝塞尔函数级数
add_newdoc("_struve_bessel_series",
    """
    _struve_bessel_series(v, z, is_h)

    Internal function for testing `struve` & `modstruve`

    Evaluates using Bessel function series

    Returns
    -------
    v, err
    """)

# 内部函数，建议使用 `spherical_jn` 替代
add_newdoc("_spherical_jn",
    """
    Internal function, use `spherical_jn` instead.
    """)

# 内部函数，建议使用 `spherical_jn` 替代
add_newdoc("_spherical_jn_d",
    """
    Internal function, use `spherical_jn` instead.
    """)

# 内部函数，建议使用 `spherical_yn` 替代
add_newdoc("_spherical_yn",
    """
    Internal function, use `spherical_yn` instead.
    """)

# 内部函数，建议使用 `spherical_yn` 替代
add_newdoc("_spherical_yn_d",
    """
    Internal function, use `spherical_yn` instead.
    """)

# 内部函数，建议使用 `spherical_in` 替代
add_newdoc("_spherical_in",
    """
    Internal function, use `spherical_in` instead.
    """)

# 内部函数，建议使用 `spherical_in` 替代
add_newdoc("_spherical_in_d",
    """
    Internal function, use `spherical_in` instead.
    """)

# 内部函数，建议使用 `spherical_kn` 替代
add_newdoc("_spherical_kn",
    """
    Internal function, use `spherical_kn` instead.
    """)

# 内部函数，建议使用 `spherical_kn` 替代
add_newdoc("_spherical_kn_d",
    """
    Internal function, use `spherical_kn` instead.
    """)

# Owen's T 函数，用于计算 X 和 Y 为独立标准正态随机变量时的概率
add_newdoc("owens_t",
    """
    owens_t(h, a, out=None)

    Owen's T Function.

    The function T(h, a) gives the probability of the event
    (X > h and 0 < Y < a * X) where X and Y are independent
    standard normal random variables.
    """)
    Parameters
    ----------
    h: array_like
        输入值 h，可以是标量或数组。
    a: array_like
        输入值 a，可以是标量或数组。
    out : ndarray, optional
        可选的输出数组，用于存储函数结果。

    Returns
    -------
    t: scalar or ndarray
        事件的概率 (X > h and 0 < Y < a * X)，其中 X 和 Y 是独立的标准正态随机变量。

    References
    ----------
    .. [1] M. Patefield and D. Tandy, "Fast and accurate calculation of
           Owen's T Function", Statistical Software vol. 5, pp. 1-25, 2000.

    Examples
    --------
    >>> from scipy import special
    >>> a = 3.5
    >>> h = 0.78
    >>> special.owens_t(h, a)
    0.10877216734852274
# 将文档字符串添加到 "_factorial" 函数，说明这是一个内部函数，不应该被直接使用
add_newdoc("_factorial",
    """
    Internal function, do not use.
    """)

# 将文档字符串添加到 "ndtri_exp" 函数，解释该函数是标准正态分布的逆对数累积分布函数与指数函数的组合逆函数
add_newdoc("ndtri_exp",
    r"""
    ndtri_exp(y, out=None)

    Inverse of `log_ndtr` vs x. Allows for greater precision than
    `ndtri` composed with `numpy.exp` for very small values of y and for
    y close to 0.

    Parameters
    ----------
    y : array_like of float
        Function argument
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Inverse of the log CDF of the standard normal distribution, evaluated
        at y.

    See Also
    --------
    log_ndtr : log of the standard normal cumulative distribution function
    ndtr : standard normal cumulative distribution function
    ndtri : standard normal percentile function

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    `ndtri_exp` agrees with the naive implementation when the latter does
    not suffer from underflow.

    >>> sc.ndtri_exp(-1)
    -0.33747496376420244
    >>> sc.ndtri(np.exp(-1))
    -0.33747496376420244

    For extreme values of y, the naive approach fails

    >>> sc.ndtri(np.exp(-800))
    -inf
    >>> sc.ndtri(np.exp(-1e-20))
    inf

    whereas `ndtri_exp` is still able to compute the result to high precision.

    >>> sc.ndtri_exp(-800)
    -39.88469483825668
    >>> sc.ndtri_exp(-1e-20)
    9.262340089798409
    """)


# 将文档字符串添加到 "_stirling2_inexact" 函数，说明这是一个内部函数，不应该被直接使用
add_newdoc("_stirling2_inexact",
    r"""
    Internal function, do not use.
    """)

# 将文档字符串添加到 "_beta_pdf" 函数，解释该函数是贝塔分布的概率密度函数
add_newdoc(
    "_beta_pdf",
    r"""
    _beta_pdf(x, a, b)

    Probability density function of beta distribution.

    Parameters
    ----------
    x : array_like
        Real-valued such that :math:`0 \leq x \leq 1`,
        the upper limit of integration
    a, b : array_like
           Positive, real-valued parameters

    Returns
    -------
    scalar or ndarray

    """)

# 将文档字符串添加到 "_beta_ppf" 函数，解释该函数是贝塔分布的百分点函数（逆分布函数）
add_newdoc(
    "_beta_ppf",
    r"""
    _beta_ppf(x, a, b)

    Percent point function of beta distribution.

    Parameters
    ----------
    x : array_like
        Real-valued such that :math:`0 \leq x \leq 1`,
        the upper limit of integration
    a, b : array_like
           Positive, real-valued parameters

    Returns
    -------
    scalar or ndarray

    """)

# 将文档字符串添加到 "_invgauss_ppf" 函数，解释该函数是逆高斯分布的百分点函数（逆分布函数）
add_newdoc(
    "_invgauss_ppf",
    """
    _invgauss_ppf(x, mu)

    Percent point function of inverse gaussian distribution.

    Parameters
    ----------
    x : array_like
        Positive real-valued
    mu : array_like
        Positive, real-valued parameters

    Returns
    -------
    scalar or ndarray

    """)

# 将文档字符串添加到 "_invgauss_isf" 函数，解释该函数是逆高斯分布的逆生存函数（逆分布函数）
add_newdoc(
    "_invgauss_isf",
    """
    _invgauss_isf(x, mu, s)

    Inverse survival function of inverse gaussian distribution.

    Parameters
    ----------
    x : array_like
        Positive real-valued
    mu : array_like
        Positive, real-valued parameters
    s : array_like
        Positive, real-valued parameters

    Returns
    -------
    scalar or ndarray

    """)
    scalar or ndarray
    # 这是一个函数或方法的文档字符串，用于描述函数或方法的参数或返回值类型。
add_newdoc(
    "_ncf_isf",
    """
    _ncf_isf(x, v1, v2, l)

    Inverse survival function of noncentral F-distribution.

    Parameters
    ----------
    x : array_like
        Positive real-valued
    v1, v2, l : array_like
        Positive, real-valued parameters

    Returns
    -------
    scalar or ndarray

    """
)
    # 调用 _ncf_isf 函数，用于计算非中心 F 分布的反生存函数
    _ncf_isf(x, v1, v2, l)
    # 函数用于计算非中心 F 分布的反生存函数，参数如下：
    # - x: 数组，包含正实数值
    # - v1, v2, l: 数组，包含正实数参数
    # 返回值为标量或数组
# 向指定的库函数添加新的文档字符串，用于描述非中心 F 分布的均值计算函数
add_newdoc(
    "_ncf_mean",
    """
    _ncf_mean(v1, v2, l)

    Mean of noncentral F-distribution.

    Parameters
    ----------
    v1, v2, l : array_like
        Positive, real-valued parameters

    Returns
    -------
    scalar or ndarray

    """)

# 向指定的库函数添加新的文档字符串，用于描述非中心 F 分布的方差计算函数
add_newdoc(
    "_ncf_variance",
    """
    _ncf_variance(v1, v2, l)

    Variance of noncentral F-distribution.

    Parameters
    ----------
    v1, v2, l : array_like
        Positive, real-valued parameters

    Returns
    -------
    scalar or ndarray

    """)

# 向指定的库函数添加新的文档字符串，用于描述非中心 F 分布的偏度计算函数
add_newdoc(
    "_ncf_skewness",
    """
    _ncf_skewness(v1, v2, l)

    Skewness of noncentral F-distribution.

    Parameters
    ----------
    v1, v2, l : array_like
        Positive, real-valued parameters

    Returns
    -------
    scalar or ndarray

    """)

# 向指定的库函数添加新的文档字符串，用于描述非中心 F 分布的峰度过剩计算函数
add_newdoc(
    "_ncf_kurtosis_excess",
    """
    _ncf_kurtosis_excess(v1, v2, l)

    Kurtosis excess of noncentral F-distribution.

    Parameters
    ----------
    v1, v2, l : array_like
        Positive, real-valued parameters

    Returns
    -------
    scalar or ndarray

    """)

# 向指定的库函数添加新的文档字符串，用于描述非中心 t 分布的累积分布函数
add_newdoc(
    "_nct_cdf",
    """
    _nct_cdf(x, v, l)

    Cumulative density function of noncentral t-distribution.

    Parameters
    ----------
    x : array_like
        Real-valued
    v : array_like
        Positive, real-valued parameters
    l : array_like
        Real-valued parameters

    Returns
    -------
    scalar or ndarray

    """)

# 向指定的库函数添加新的文档字符串，用于描述非中心 t 分布的百分点函数
add_newdoc(
    "_nct_ppf",
    """
    _nct_ppf(x, v, l)

    Percent point function of noncentral t-distribution.

    Parameters
    ----------
    x : array_like
        Real-valued
    v : array_like
        Positive, real-valued parameters
    l : array_like
        Real-valued parameters

    Returns
    -------
    scalar or ndarray

    """)

# 向指定的库函数添加新的文档字符串，用于描述非中心 t 分布的生存函数
add_newdoc(
    "_nct_sf",
    """
    _nct_sf(x, v, l)

    Survival function of noncentral t-distribution.

    Parameters
    ----------
    x : array_like
        Real-valued
    v : array_like
        Positive, real-valued parameters
    l : array_like
        Real-valued parameters

    Returns
    -------
    scalar or ndarray

    """)

# 向指定的库函数添加新的文档字符串，用于描述非中心 t 分布的反向生存函数
add_newdoc(
    "_nct_isf",
    """
    _nct_isf(x, v, l)

    Inverse survival function of noncentral t-distribution.

    Parameters
    ----------
    x : array_like
        Real-valued
    v : array_like
        Positive, real-valued parameters
    l : array_like
        Real-valued parameters

    Returns
    -------
    scalar or ndarray

    """)

# 向指定的库函数添加新的文档字符串，用于描述非中心 t 分布的均值计算函数
add_newdoc(
    "_nct_mean",
    """
    _nct_mean(v, l)

    Mean of noncentral t-distribution.

    Parameters
    ----------
    v : array_like
        Positive, real-valued parameters
    l : array_like
        Real-valued parameters

    Returns
    -------
    scalar or ndarray

    """)

# 向指定的库函数添加新的文档字符串，用于描述非中心 t 分布的方差计算函数
add_newdoc(
    "_nct_variance",
    """
    _nct_variance(v, l)

    Variance of noncentral t-distribution.

    Parameters
    ----------
    v : array_like
        Positive, real-valued parameters
    l : array_like
        Real-valued parameters

    Returns
    -------
    scalar or ndarray

    """)
    # v : array_like
    #     输入参数 v，可以是类数组的结构，通常表示正实数值的参数
    # l : array_like
    #     输入参数 l，可以是类数组的结构，通常表示实数值的参数

    # Returns
    # -------
    # scalar or ndarray
    #     返回值可以是标量（scalar）或者 ndarray（多维数组），取决于函数的计算结果
# 添加新的文档字符串给指定的函数或方法，用于描述其作用、参数和返回值
add_newdoc(
    "_nct_skewness",
    """
    _nct_skewness(v, l)

    Skewness of noncentral t-distribution.

    Parameters
    ----------
    v : array_like
        Positive, real-valued parameters
    l : array_like
        Real-valued parameters

    Returns
    -------
    scalar or ndarray

    """
)

# 添加新的文档字符串给指定的函数或方法，用于描述其作用、参数和返回值
add_newdoc(
    "_nct_kurtosis_excess",
    """
    _nct_kurtosis_excess(v, l)

    Kurtosis excess of noncentral t-distribution.

    Parameters
    ----------
    v : array_like
        Positive, real-valued parameters
    l : array_like
        Real-valued parameters

    Returns
    -------
    scalar or ndarray

    """
)

# 添加新的文档字符串给指定的函数或方法，用于描述其作用、参数和返回值
add_newdoc(
    "_skewnorm_cdf",
    """
    _skewnorm_cdf(x, l, sc, sh)

    Cumulative density function of skewnorm distribution.

    Parameters
    ----------
    x : array_like
        Real-valued
    l : array_like
        Real-valued parameters
    sc : array_like
        Positive, Real-valued parameters
    sh : array_like
        Real-valued parameters

    Returns
    -------
    scalar or ndarray

    """
)

# 添加新的文档字符串给指定的函数或方法，用于描述其作用、参数和返回值
add_newdoc(
    "_skewnorm_ppf",
    """
    _skewnorm_ppf(x, l, sc, sh)

    Percent point function of skewnorm distribution.

    Parameters
    ----------
    x : array_like
        Real-valued
    l : array_like
        Real-valued parameters
    sc : array_like
        Positive, Real-valued parameters
    sh : array_like
        Real-valued parameters

    Returns
    -------
    scalar or ndarray

    """
)

# 添加新的文档字符串给指定的函数或方法，用于描述其作用、参数和返回值
add_newdoc(
    "_skewnorm_isf",
    """
    _skewnorm_isf(x, l, sc, sh)

    Inverse survival function of skewnorm distribution.

    Parameters
    ----------
    x : array_like
        Real-valued
    l : array_like
        Real-valued parameters
    sc : array_like
        Positive, Real-valued parameters
    sh : array_like
        Real-valued parameters

    Returns
    -------
    scalar or ndarray

    """
)

# 添加新的文档字符串给指定的函数或方法，用于描述其作用、参数和返回值
add_newdoc(
    "_binom_pmf",
    """
    _binom_pmf(x, n, p)

    Probability mass function of binomial distribution.

    Parameters
    ----------
    x : array_like
        Real-valued
    n : array_like
        Positive, integer-valued parameter
    p : array_like
        Positive, real-valued parameter

    Returns
    -------
    scalar or ndarray

    """
)

# 添加新的文档字符串给指定的函数或方法，用于描述其作用、参数和返回值
add_newdoc(
    "_binom_cdf",
    """
    _binom_cdf(x, n, p)

    Cumulative density function of binomial distribution.

    Parameters
    ----------
    x : array_like
        Real-valued
    n : array_like
        Positive, integer-valued parameter
    p : array_like
        Positive, real-valued parameter

    Returns
    -------
    scalar or ndarray

    """
)

# 添加新的文档字符串给指定的函数或方法，用于描述其作用、参数和返回值
add_newdoc(
    "_binom_ppf",
    """
    _binom_ppf(x, n, p)

    Percent point function of binomial distribution.

    Parameters
    ----------
    x : array_like
        Real-valued
    n : array_like
        Positive, integer-valued parameter
    p : array_like
        Positive, real-valued parameter

    Returns
    -------
    scalar or ndarray

    """
)
add_newdoc(
    "_binom_sf",
    """
    _binom_sf(x, n, p)

    Survival function of binomial distribution.

    Parameters
    ----------
    x : array_like
        Real-valued
    n : array_like
        Positive, integer-valued parameter
    p : array_like
        Positive, real-valued parameter

    Returns
    -------
    scalar or ndarray

    """
)

add_newdoc(
    "_binom_isf",
    """
    _binom_isf(x, n, p)

    Inverse survival function of binomial distribution.

    Parameters
    ----------
    x : array_like
        Real-valued
    n : array_like
        Positive, integer-valued parameter
    p : array_like
        Positive, real-valued parameter

    Returns
    -------
    scalar or ndarray

    """
)

add_newdoc(
    "_nbinom_pmf",
    """
    _nbinom_pmf(x, r, p)

    Probability mass function of negative binomial distribution.

    Parameters
    ----------
    x : array_like
        Real-valued
    r : array_like
        Positive, integer-valued parameter
    p : array_like
        Positive, real-valued parameter

    Returns
    -------
    scalar or ndarray

    """
)

add_newdoc(
    "_nbinom_cdf",
    """
    _nbinom_cdf(x, r, p)

    Cumulative density function of negative binomial distribution.

    Parameters
    ----------
    x : array_like
        Real-valued
    r : array_like
        Positive, integer-valued parameter
    p : array_like
        Positive, real-valued parameter

    Returns
    -------
    scalar or ndarray

    """
)

add_newdoc(
    "_nbinom_ppf",
    """
    _nbinom_ppf(x, r, p)

    Percent point function of negative binomial distribution.

    Parameters
    ----------
    x : array_like
        Real-valued
    r : array_like
        Positive, integer-valued parameter
    p : array_like
        Positive, real-valued parameter

    Returns
    -------
    scalar or ndarray

    """
)

add_newdoc(
    "_nbinom_sf",
    """
    _nbinom_sf(x, r, p)

    Survival function of negative binomial distribution.

    Parameters
    ----------
    x : array_like
        Real-valued
    r : array_like
        Positive, integer-valued parameter
    p : array_like
        Positive, real-valued parameter

    Returns
    -------
    scalar or ndarray

    """
)

add_newdoc(
    "_nbinom_isf",
    """
    _nbinom_isf(x, r, p)

    Inverse survival function of negative binomial distribution.

    Parameters
    ----------
    x : array_like
        Real-valued
    r : array_like
        Positive, integer-valued parameter
    p : array_like
        Positive, real-valued parameter

    Returns
    -------
    scalar or ndarray

    """
)

add_newdoc(
    "_nbinom_mean",
    """
    _nbinom_mean(r, p)

    Mean of negative binomial distribution.

    Parameters
    ----------
    r : array_like
        Positive, integer-valued parameter
    p : array_like
        Positive, real-valued parameter

    Returns
    -------
    scalar or ndarray

    """
)

add_newdoc(
    "_nbinom_variance",
    """
    _nbinom_variance(r, p)

    Variance of negative binomial distribution.

    Parameters
    ----------
    r : array_like
        Positive, integer-valued parameter
    p : array_like
        Positive, real-valued parameter

    Returns
    -------
    scalar or ndarray

    """
)
    Variance of negative binomial distribution.  # 负二项分布的方差计算。

    Parameters  # 参数说明部分开始
    ----------  # 分隔线，以下为参数说明
    r : array_like  # r 是一个类数组对象，表示正整数值的参数
        Positive, integer-valued parameter  # 必须是正整数值的参数
    p : array_like  # p 是一个类数组对象，表示正实数值的参数
        Positive, real-valued parameter  # 必须是正实数值的参数

    Returns  # 返回部分开始
    -------  # 分隔线，以下为返回值说明
    scalar or ndarray  # 返回标量或者 ndarray 数组

    """
# 添加新的文档字符串到指定函数或方法，用于描述负二项分布的偏度计算函数
add_newdoc(
    "_nbinom_skewness",
    """
    _nbinom_skewness(r, p)

    Skewness of negative binomial distribution.

    Parameters
    ----------
    r : array_like
        Positive, integer-valued parameter
    p : array_like
        Positive, real-valued parameter

    Returns
    -------
    scalar or ndarray

    """
)

# 添加新的文档字符串到指定函数或方法，用于描述负二项分布的峰度偏差计算函数
add_newdoc(
    "_nbinom_kurtosis_excess",
    """
    _nbinom_kurtosis_excess(r, p)

    Kurtosis excess of negative binomial distribution.

    Parameters
    ----------
    r : array_like
        Positive, integer-valued parameter
    p : array_like
        Positive, real-valued parameter

    Returns
    -------
    scalar or ndarray

    """
)

# 添加新的文档字符串到指定函数或方法，用于描述超几何分布的概率质量函数
add_newdoc(
    "_hypergeom_pmf",
    """
    _hypergeom_pmf(x, r, N, M)

    Probability mass function of hypergeometric distribution.

    Parameters
    ----------
    x : array_like
        Real-valued
    r, N, M : array_like
        Positive, integer-valued parameter

    Returns
    -------
    scalar or ndarray

    """
)

# 添加新的文档字符串到指定函数或方法，用于描述超几何分布的累积分布函数
add_newdoc(
    "_hypergeom_cdf",
    """
    _hypergeom_cdf(x, r, N, M)

    Cumulative density function of hypergeometric distribution.

    Parameters
    ----------
    x : array_like
        Real-valued
    r, N, M : array_like
        Positive, integer-valued parameter

    Returns
    -------
    scalar or ndarray
    """
)

# 添加新的文档字符串到指定函数或方法，用于描述超几何分布的生存函数
add_newdoc(
    "_hypergeom_sf",
    """
    _hypergeom_sf(x, r, N, M)

    Survival function of hypergeometric distribution.

    Parameters
    ----------
    x : array_like
        Real-valued
    r, N, M : array_like
        Positive, integer-valued parameter

    Returns
    -------
    scalar or ndarray
    """
)

# 添加新的文档字符串到指定函数或方法，用于描述超几何分布的均值计算函数
add_newdoc(
    "_hypergeom_mean",
    """
    _hypergeom_mean(r, N, M)

    Mean of hypergeometric distribution.

    Parameters
    ----------
    r, N, M : array_like
        Positive, integer-valued parameter

    Returns
    -------
    scalar or ndarray

    """
)

# 添加新的文档字符串到指定函数或方法，用于描述超几何分布的方差计算函数
add_newdoc(
    "_hypergeom_variance",
    """
    _hypergeom_variance(r, N, M)

    Mean of hypergeometric distribution.

    Parameters
    ----------
    r, N, M : array_like
        Positive, integer-valued parameter

    Returns
    -------
    scalar or ndarray

    """
)

# 添加新的文档字符串到指定函数或方法，用于描述超几何分布的偏度计算函数
add_newdoc(
    "_hypergeom_skewness",
    """
    _hypergeom_skewness(r, N, M)

    Skewness of hypergeometric distribution.

    Parameters
    ----------
    r, N, M : array_like
        Positive, integer-valued parameter

    Returns
    -------
    scalar or ndarray

    """
)
```