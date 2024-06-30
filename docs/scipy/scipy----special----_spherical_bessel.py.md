# `D:\src\scipysrc\scipy\scipy\special\_spherical_bessel.py`

```
# 导入NumPy库，用于数值计算
import numpy as np
# 从内部模块导入特定函数，用于计算球面贝塞尔函数及其导数
from ._ufuncs import (_spherical_jn, _spherical_yn, _spherical_in,
                      _spherical_kn, _spherical_jn_d, _spherical_yn_d,
                      _spherical_in_d, _spherical_kn_d)

# 定义计算球面贝塞尔函数第一类的函数
def spherical_jn(n, z, derivative=False):
    r"""Spherical Bessel function of the first kind or its derivative.

    Defined as [1]_,

    .. math:: j_n(z) = \sqrt{\frac{\pi}{2z}} J_{n + 1/2}(z),

    where :math:`J_n` is the Bessel function of the first kind.

    Parameters
    ----------
    n : int, array_like
        Order of the Bessel function (n >= 0).
    z : complex or float, array_like
        Argument of the Bessel function.
    derivative : bool, optional
        If True, the value of the derivative (rather than the function
        itself) is returned.

    Returns
    -------
    jn : ndarray
        Value of the spherical Bessel function or its derivative.

    Notes
    -----
    For real arguments greater than the order, the function is computed
    using the ascending recurrence [2]_. For small real or complex
    arguments, the definitional relation to the cylindrical Bessel function
    of the first kind is used.

    The derivative is computed using the relations [3]_,

    .. math::
        j_n'(z) = j_{n-1}(z) - \frac{n + 1}{z} j_n(z).

        j_0'(z) = -j_1(z)


    .. versionadded:: 0.18.0

    References
    ----------
    .. [1] https://dlmf.nist.gov/10.47.E3
    .. [2] https://dlmf.nist.gov/10.51.E1
    .. [3] https://dlmf.nist.gov/10.51.E2
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    The spherical Bessel functions of the first kind :math:`j_n` accept
    both real and complex second argument. They can return a complex type:

    >>> from scipy.special import spherical_jn
    >>> spherical_jn(0, 3+5j)
    (-9.878987731663194-8.021894345786002j)
    >>> type(spherical_jn(0, 3+5j))
    <class 'numpy.complex128'>

    We can verify the relation for the derivative from the Notes
    for :math:`n=3` in the interval :math:`[1, 2]`:

    >>> import numpy as np
    >>> x = np.arange(1.0, 2.0, 0.01)
    >>> np.allclose(spherical_jn(3, x, True),
    ...             spherical_jn(2, x) - 4/x * spherical_jn(3, x))
    True

    The first few :math:`j_n` with real argument:

    >>> import matplotlib.pyplot as plt
    >>> x = np.arange(0.0, 10.0, 0.01)
    >>> fig, ax = plt.subplots()
    >>> ax.set_ylim(-0.5, 1.5)
    >>> ax.set_title(r'Spherical Bessel functions $j_n$')
    >>> for n in np.arange(0, 4):
    ...     ax.plot(x, spherical_jn(n, x), label=rf'$j_{n}$')
    >>> plt.legend(loc='best')
    >>> plt.show()

    """
    # 将n转换为长整型数组
    n = np.asarray(n, dtype=np.dtype("long"))
    # 如果需要计算导数，调用_spherical_jn_d函数
    if derivative:
        return _spherical_jn_d(n, z)
    else:
        # 否则，调用_spherical_jn函数计算球面贝塞尔函数第一类的值
        return _spherical_jn(n, z)
    # 将输入的 n 转换为长整型数组，确保能处理多个输入的情况
    n = np.asarray(n, dtype=np.dtype("long"))
    # 如果 derivative 参数为 True，则计算球形贝塞尔函数的导数
    if derivative:
        return _spherical_yn_d(n, z)
    else:
        # 否则，计算球形贝塞尔函数的值
        return _spherical_yn(n, z)
# 修改后的球形贝塞尔函数的第一类或其导数。
def spherical_in(n, z, derivative=False):
    """
    Modified spherical Bessel function of the first kind or its derivative.

    Defined as [1],

    .. math:: i_n(z) = \sqrt{\frac{\pi}{2z}} I_{n + 1/2}(z),

    where :math:`I_n` is the modified Bessel function of the first kind.

    Parameters
    ----------
    n : int, array_like
        Order of the Bessel function (n >= 0).
    z : complex or float, array_like
        Argument of the Bessel function.
    derivative : bool, optional
        If True, the value of the derivative (rather than the function
        itself) is returned.

    Returns
    -------
    in : ndarray

    Notes
    -----
    The function is computed using its definitional relation to the
    modified cylindrical Bessel function of the first kind.

    The derivative is computed using the relations [2],

    .. math::
        i_n' = i_{n-1} - \frac{n + 1}{z} i_n.

        i_1' = i_0

    .. versionadded:: 0.18.0

    References
    ----------
    .. [1] https://dlmf.nist.gov/10.47.E7
    .. [2] https://dlmf.nist.gov/10.51.E5
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    The modified spherical Bessel functions of the first kind :math:`i_n`
    accept both real and complex second argument.
    They can return a complex type:

    >>> from scipy.special import spherical_in
    >>> spherical_in(0, 3+5j)
    (-1.1689867793369182-1.2697305267234222j)
    >>> type(spherical_in(0, 3+5j))
    <class 'numpy.complex128'>

    We can verify the relation for the derivative from the Notes
    for :math:`n=3` in the interval :math:`[1, 2]`:

    >>> import numpy as np
    >>> x = np.arange(1.0, 2.0, 0.01)
    >>> np.allclose(spherical_in(3, x, True),
    ...             spherical_in(2, x) - 4/x * spherical_in(3, x))
    True

    The first few :math:`i_n` with real argument:

    >>> import matplotlib.pyplot as plt
    >>> x = np.arange(0.0, 6.0, 0.01)
    >>> fig, ax = plt.subplots()
    >>> ax.set_ylim(-0.5, 5.0)
    >>> ax.set_title(r'Modified spherical Bessel functions $i_n$')
    >>> for n in np.arange(0, 4):
    ...     ax.plot(x, spherical_in(n, x), label=rf'$i_{n}$')
    >>> plt.legend(loc='best')
    >>> plt.show()
    """
    n = np.asarray(n, dtype=np.dtype("long"))
    if derivative:
        # 返回导数而不是函数本身
        return _spherical_in_d(n, z)
    else:
        # 返回函数本身
        return _spherical_in(n, z)


# 修改后的球形贝塞尔函数的第二类或其导数。
def spherical_kn(n, z, derivative=False):
    """
    Modified spherical Bessel function of the second kind or its derivative.

    Defined as [1],

    .. math:: k_n(z) = \sqrt{\frac{\pi}{2z}} K_{n + 1/2}(z),

    where :math:`K_n` is the modified Bessel function of the second kind.

    Parameters
    ----------
    n : int, array_like
        Order of the Bessel function (n >= 0).
    z : complex or float, array_like
        Argument of the Bessel function.
    derivative : bool, optional
        If True, the value of the derivative (rather than the function
        itself) is returned.

    """
    # 略去此处的详细注释，因为要求不将注释写到代码块之外
    pass
    # 将输入的 n 转换为 numpy 数组，数据类型为 long
    n = np.asarray(n, dtype=np.dtype("long"))
    # 如果 derivative 参数为 True，则计算修改球形贝塞尔函数的导数
    if derivative:
        return _spherical_kn_d(n, z)
    else:
        # 如果 derivative 参数为 False，则计算修改球形贝塞尔函数本身
        return _spherical_kn(n, z)
```