# `D:\src\scipysrc\scipy\scipy\special\_basic.py`

```
# 导入模块 operator，提供了一些对内置操作符的额外函数支持
import operator
# 导入模块 numpy，并使用别名 np
import numpy as np
# 导入模块 math，提供了基本的数学函数
import math
# 导入模块 warnings，用于处理警告信息
import warnings
# 导入 collections 模块中的 defaultdict 类，用于创建默认值为列表的字典
from collections import defaultdict
# 从 heapq 模块中导入 heapify 和 heappop 函数，用于堆操作
from heapq import heapify, heappop
# 从 numpy 模块中导入多个函数，包括常数 pi 和一些数组操作函数
from numpy import (pi, asarray, floor, isscalar, sqrt, where,
                   sin, place, issubdtype, extract, inexact, nan, zeros, sinc)
# 从当前包（.）中导入 _ufuncs 模块
from . import _ufuncs
# 从 _ufuncs 模块中导入一系列函数和常量，包括数学函数和特殊函数
from ._ufuncs import (mathieu_a, mathieu_b, iv, jv, gamma,
                      psi, hankel1, hankel2, yv, kv, poch, binom,
                      _stirling2_inexact)
# 从当前包（.）中导入 _gufuncs 模块中的 _sph_harm_all 函数
from ._gufuncs import (_lpn, _lpmn, _clpmn, _lqn, _lqmn, _rctj, _rcty,
                       _sph_harm_all as _sph_harm_all_gufunc)
# 从当前包（.）中导入 _specfun 模块
from . import _specfun
# 从当前包（.）中导入 _comb 模块中的 _comb_int 函数
from ._comb import _comb_int

# 定义 __all__ 列表，包含当前模块中公开的所有函数和常量名
__all__ = [
    'ai_zeros',
    'assoc_laguerre',
    'bei_zeros',
    'beip_zeros',
    'ber_zeros',
    'bernoulli',
    'berp_zeros',
    'bi_zeros',
    'clpmn',
    'comb',
    'digamma',
    'diric',
    'erf_zeros',
    'euler',
    'factorial',
    'factorial2',
    'factorialk',
    'fresnel_zeros',
    'fresnelc_zeros',
    'fresnels_zeros',
    'h1vp',
    'h2vp',
    'ivp',
    'jn_zeros',
    'jnjnp_zeros',
    'jnp_zeros',
    'jnyn_zeros',
    'jvp',
    'kei_zeros',
    'keip_zeros',
    'kelvin_zeros',
    'ker_zeros',
    'kerp_zeros',
    'kvp',
    'lmbda',
    'lpmn',
    'lpn',
    'lqmn',
    'lqn',
    'mathieu_even_coef',
    'mathieu_odd_coef',
    'obl_cv_seq',
    'pbdn_seq',
    'pbdv_seq',
    'pbvv_seq',
    'perm',
    'polygamma',
    'pro_cv_seq',
    'riccati_jn',
    'riccati_yn',
    'sinc',
    'stirling2',
    'y0_zeros',
    'y1_zeros',
    'y1p_zeros',
    'yn_zeros',
    'ynp_zeros',
    'yvp',
    'zeta'
]

# 定义字典 _FACTORIALK_LIMITS_64BITS，映射 k 到最大的 n，使得 factorialk(n, k) < np.iinfo(np.int64).max
_FACTORIALK_LIMITS_64BITS = {1: 20, 2: 33, 3: 44, 4: 54, 5: 65,
                             6: 74, 7: 84, 8: 93, 9: 101}
# 定义字典 _FACTORIALK_LIMITS_32BITS，映射 k 到最大的 n，使得 factorialk(n, k) < np.iinfo(np.int32).max
_FACTORIALK_LIMITS_32BITS = {1: 12, 2: 19, 3: 25, 4: 31, 5: 37,
                             6: 43, 7: 47, 8: 51, 9: 56}

# 定义函数 _nonneg_int_or_fail，用于验证并返回非负整数
def _nonneg_int_or_fail(n, var_name, strict=True):
    try:
        if strict:
            # 如果 strict=True，尝试将 n 转换为整数索引，如果 n 是浮点数则引发异常
            n = operator.index(n)
        elif n == floor(n):
            # 如果 strict=False，检查 n 是否是整数，是则转换为整数，否则引发异常
            n = int(n)
        else:
            raise ValueError()
        if n < 0:
            raise ValueError()
    except (ValueError, TypeError) as err:
        # 捕获可能的异常，并用特定的错误消息重新引发
        raise err.__class__(f"{var_name} must be a non-negative integer") from err
    return n

# 定义函数 diric，实现周期 sinc 函数（Dirichlet 函数）的计算
def diric(x, n):
    """Periodic sinc function, also called the Dirichlet function.

    The Dirichlet function is defined as::

        diric(x, n) = sin(x * n/2) / (n * sin(x / 2)),

    where `n` is a positive integer.

    Parameters
    ----------
    x : array_like
        Input data
    n : int
        Integer defining the periodicity.

    Returns
    -------
    diric : ndarray
        Output array containing the Dirichlet function evaluated at each point of `x`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import special
    """
    >>> import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，用于绘图

    >>> x = np.linspace(-8*np.pi, 8*np.pi, num=201)  # 在区间[-8π, 8π]上生成201个均匀间隔的点作为 x 值

    >>> plt.figure(figsize=(8, 8));  # 创建一个8x8英寸大小的图形窗口
    >>> for idx, n in enumerate([2, 3, 4, 9]):  # 遍历列表中的值和索引
    ...     plt.subplot(2, 2, idx+1)  # 创建2x2的子图布局，并选择当前子图
    ...     plt.plot(x, special.diric(x, n))  # 绘制 diric 函数的图像，使用特定的 n 值
    ...     plt.title('diric, n={}'.format(n))  # 设置子图标题
    >>> plt.show()  # 显示绘制的图形

    The following example demonstrates that `diric` gives the magnitudes
    (modulo the sign and scaling) of the Fourier coefficients of a
    rectangular pulse.

    Suppress output of values that are effectively 0:

    >>> np.set_printoptions(suppress=True)  # 设置 numpy 输出选项，抑制小数输出

    Create a signal `x` of length `m` with `k` ones:

    >>> m = 8  # 信号长度为 8
    >>> k = 3  # 信号中有 3 个 '1'
    >>> x = np.zeros(m)  # 创建长度为 m 的全零数组
    >>> x[:k] = 1  # 将前 k 个元素设为 1，形成矩形脉冲信号

    Use the FFT to compute the Fourier transform of `x`, and
    inspect the magnitudes of the coefficients:

    >>> np.abs(np.fft.fft(x))  # 计算信号 `x` 的傅里叶变换后的幅度谱

    Now find the same values (up to sign) using `diric`. We multiply
    by `k` to account for the different scaling conventions of
    `numpy.fft.fft` and `diric`:

    >>> theta = np.linspace(0, 2*np.pi, m, endpoint=False)  # 在 [0, 2π) 区间生成 m 个均匀间隔的角度值
    >>> k * special.diric(theta, k)  # 使用 diric 函数计算信号 `x` 的谐波系数（除去标度因子）

    """
    x, n = asarray(x), asarray(n)  # 将 x 和 n 转换为 numpy 数组
    n = asarray(n + (x-x))  # 将 n 转换为与 x 相同类型的数组，进行一些操作
    x = asarray(x + (n-n))  # 将 x 转换为与 n 相同类型的数组，进行一些操作
    if issubdtype(x.dtype, inexact):  # 检查 x 的数据类型是否是非精确类型
        ytype = x.dtype  # 如果是，将 ytype 设为 x 的数据类型
    else:
        ytype = float  # 否则将 ytype 设为 float 类型
    y = zeros(x.shape, ytype)  # 创建一个与 x 形状相同的全零数组，类型为 ytype

    # empirical minval for 32, 64 or 128 bit float computations
    # where sin(x/2) < minval, result is fixed at +1 or -1
    if np.finfo(ytype).eps < 1e-18:  # 根据 ytype 的精度选择 minval 的值
        minval = 1e-11
    elif np.finfo(ytype).eps < 1e-15:
        minval = 1e-7
    else:
        minval = 1e-3

    mask1 = (n <= 0) | (n != floor(n))  # 创建一个掩码，用于标记 n 的值符合条件的位置
    place(y, mask1, nan)  # 将 y 中掩码对应位置的值设为 NaN

    x = x / 2  # 将 x 的每个元素除以 2
    denom = sin(x)  # 计算 x 的每个元素的 sin 值
    mask2 = (1-mask1) & (abs(denom) < minval)  # 创建第二个掩码，标记满足条件的位置
    xsub = extract(mask2, x)  # 提取符合第二个掩码的 x 的子集
    nsub = extract(mask2, n)  # 提取符合第二个掩码的 n 的子集
    zsub = xsub / pi  # 对子集 xsub 进行一些操作
    place(y, mask2, pow(-1, np.round(zsub)*(nsub-1)))  # 将 y 中掩码对应位置的值设为一些操作的结果

    mask = (1-mask1) & (1-mask2)  # 创建一个综合掩码，标记满足条件的位置
    xsub = extract(mask, x)  # 提取符合综合掩码的 x 的子集
    nsub = extract(mask, n)  # 提取符合综合掩码的 n 的子集
    dsub = extract(mask, denom)  # 提取符合综合掩码的 denom 的子集
    place(y, mask, sin(nsub*xsub)/(nsub*dsub))  # 将 y 中综合掩码对应位置的值设为一些操作的结果
    return y  # 返回计算结果 y
# 计算整数阶贝塞尔函数 Jn 和 Jn' 的零点。

# 结果按零点的大小顺序排列。

# Parameters
# ----------
# nt : int
#     要计算的零点数量（<=1200）

# Returns
# -------
# zo[l-1] : ndarray
#     Jn(x) 和 Jn'(x) 的第l个零点的值。长度为 `nt`。
# n[l-1] : ndarray
#     与第l个零点相关联的 Jn(x) 或 Jn'(x) 的阶数。长度为 `nt`。
# m[l-1] : ndarray
#     与第l个零点相关联的 Jn(x) 或 Jn'(x) 的零点的序号。长度为 `nt`。
# t[l-1] : ndarray
#     如果 zo 中的第l个零点是 Jn(x) 的零点，则为0；如果是 Jn'(x) 的零点，则为1。长度为 `nt`。

# See Also
# --------
# jn_zeros, jnp_zeros : 获取分开的 Jn 和 Jn' 的零点数组。

# References
# ----------
# .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
#        Functions", John Wiley and Sons, 1996, chapter 5.
#        https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

def jnjnp_zeros(nt):
    if not isscalar(nt) or (floor(nt) != nt) or (nt > 1200):
        raise ValueError("Number must be integer <= 1200.")
    nt = int(nt)
    n, m, t, zo = _specfun.jdzo(nt)
    return zo[1:nt+1], n[:nt], m[:nt], t[:nt]


# 计算贝塞尔函数 Jn(x), Jn'(x), Yn(x), Yn'(x) 的 nt 个零点。

# 返回长度为 `nt` 的4个数组，分别对应 Jn(x), Jn'(x), Yn(x), Yn'(x) 的第一个 `nt` 个零点，按升序排列。

# Parameters
# ----------
# n : int
#     贝塞尔函数的阶数
# nt : int
#     要计算的零点数量（<=1200）

# Returns
# -------
# Jn : ndarray
#     Jn 的前 `nt` 个零点
# Jnp : ndarray
#     Jn' 的前 `nt` 个零点
# Yn : ndarray
#     Yn 的前 `nt` 个零点
# Ynp : ndarray
#     Yn' 的前 `nt` 个零点

# See Also
# --------
# jn_zeros, jnp_zeros, yn_zeros, ynp_zeros

# References
# ----------
# .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
#        Functions", John Wiley and Sons, 1996, chapter 5.
#        https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

# Examples
# --------
# 计算 J1, J1', Y1 和 Y1' 的前三个根。

# >>> from scipy.special import jnyn_zeros
# >>> jn_roots, jnp_roots, yn_roots, ynp_roots = jnyn_zeros(1, 3)
# >>> jn_roots, yn_roots
# (array([ 3.83170597,  7.01558667, 10.17346814]),
#  array([2.19714133, 5.42968104, 8.59600587]))

# 绘制 J1, J1', Y1 和 Y1' 及其零点。

# >>> import numpy as np
# >>> import matplotlib.pyplot as plt
# >>> from scipy.special import jnyn_zeros, jvp, jn, yvp, yn
# >>> jn_roots, jnp_roots, yn_roots, ynp_roots = jnyn_zeros(1, 3)
# >>> fig, ax = plt.subplots()
# >>> xmax= 11
# >>> x = np.linspace(0, xmax)
# >>> x[0] += 1e-15
    # 绘制 Bessel 函数和其导数的图像，以及它们的零点
    ax.plot(x, jn(1, x), label=r"$J_1$", c='r')
    # 绘制第一类贝塞尔函数 J_1 的图像，标签为 J_1，颜色为红色

    ax.plot(x, jvp(1, x, 1), label=r"$J_1'$", c='b')
    # 绘制第一类贝塞尔函数 J_1 的导数的图像，标签为 J_1'，颜色为蓝色

    ax.plot(x, yn(1, x), label=r"$Y_1$", c='y')
    # 绘制第二类贝塞尔函数 Y_1 的图像，标签为 Y_1，颜色为黄色

    ax.plot(x, yvp(1, x, 1), label=r"$Y_1'$", c='c')
    # 绘制第二类贝塞尔函数 Y_1 的导数的图像，标签为 Y_1'，颜色为青色

    zeros = np.zeros((3, ))
    # 创建一个形状为 (3,) 的零数组

    ax.scatter(jn_roots, zeros, s=30, c='r', zorder=5,
               label=r"$J_1$ roots")
    # 在图上以红色标记绘制第一类贝塞尔函数 J_1 的零点，标签为 J_1 roots

    ax.scatter(jnp_roots, zeros, s=30, c='b', zorder=5,
               label=r"$J_1'$ roots")
    # 在图上以蓝色标记绘制第一类贝塞尔函数 J_1 的导数的零点，标签为 J_1' roots

    ax.scatter(yn_roots, zeros, s=30, c='y', zorder=5,
               label=r"$Y_1$ roots")
    # 在图上以黄色标记绘制第二类贝塞尔函数 Y_1 的零点，标签为 Y_1 roots

    ax.scatter(ynp_roots, zeros, s=30, c='c', zorder=5,
               label=r"$Y_1'$ roots")
    # 在图上以青色标记绘制第二类贝塞尔函数 Y_1 的导数的零点，标签为 Y_1' roots

    ax.hlines(0, 0, xmax, color='k')
    # 绘制一条水平线，表示 y=0，从 x=0 到 x=xmax，颜色为黑色

    ax.set_ylim(-0.6, 0.6)
    # 设置 y 轴的显示范围为 -0.6 到 0.6

    ax.set_xlim(0, xmax)
    # 设置 x 轴的显示范围为 0 到 xmax

    ax.legend(ncol=2, bbox_to_anchor=(1., 0.75))
    # 添加图例，分为两列，放置在坐标轴的右上角

    plt.tight_layout()
    # 调整子图的布局，使其填充整个图像区域

    plt.show()
    # 显示绘制好的图像
# 定义函数 jn_zeros，计算整数阶贝塞尔函数 Jn 的零点
def jn_zeros(n, nt):
    r"""Compute zeros of integer-order Bessel functions Jn.

    Compute `nt` zeros of the Bessel functions :math:`J_n(x)` on the
    interval :math:`(0, \infty)`. The zeros are returned in ascending
    order. Note that this interval excludes the zero at :math:`x = 0`
    that exists for :math:`n > 0`.

    Parameters
    ----------
    n : int
        Order of Bessel function
    nt : int
        Number of zeros to return

    Returns
    -------
    ndarray
        First `nt` zeros of the Bessel function.

    See Also
    --------
    jv: Real-order Bessel functions of the first kind
    jnp_zeros: Zeros of :math:`Jn'`

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    Examples
    --------
    Compute the first four positive roots of :math:`J_3`.

    >>> from scipy.special import jn_zeros
    >>> jn_zeros(3, 4)
    array([ 6.3801619 ,  9.76102313, 13.01520072, 16.22346616])

    Plot :math:`J_3` and its first four positive roots. Note
    that the root located at 0 is not returned by `jn_zeros`.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.special import jn, jn_zeros
    >>> j3_roots = jn_zeros(3, 4)
    >>> xmax = 18
    >>> xmin = -1
    >>> x = np.linspace(xmin, xmax, 500)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, jn(3, x), label=r'$J_3$')
    >>> ax.scatter(j3_roots, np.zeros((4, )), s=30, c='r',
    ...            label=r"$J_3$_Zeros", zorder=5)
    >>> ax.scatter(0, 0, s=30, c='k',
    ...            label=r"Root at 0", zorder=5)
    >>> ax.hlines(0, 0, xmax, color='k')
    >>> ax.set_xlim(xmin, xmax)
    >>> plt.legend()
    >>> plt.show()
    """
    # 调用 jnyn_zeros 函数，返回其计算结果的第一个元素
    return jnyn_zeros(n, nt)[0]


# 定义函数 jnp_zeros，计算整数阶贝塞尔函数导数 Jn' 的零点
def jnp_zeros(n, nt):
    r"""Compute zeros of integer-order Bessel function derivatives Jn'.

    Compute `nt` zeros of the functions :math:`J_n'(x)` on the
    interval :math:`(0, \infty)`. The zeros are returned in ascending
    order. Note that this interval excludes the zero at :math:`x = 0`
    that exists for :math:`n > 1`.

    Parameters
    ----------
    n : int
        Order of Bessel function
    nt : int
        Number of zeros to return

    Returns
    -------
    ndarray
        First `nt` zeros of the Bessel function.

    See Also
    --------
    jvp: Derivatives of integer-order Bessel functions of the first kind
    jv: Float-order Bessel functions of the first kind

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    Examples
    --------
    Compute the first four roots of :math:`J_2'`.

    >>> from scipy.special import jnp_zeros
    ```
    # 调用 jnyn_zeros 函数计算并返回第二个返回值，即 J_n 的导数的根
    return jnyn_zeros(n, nt)[1]
# 计算整数阶贝塞尔函数 Y_n(x) 的零点

# 定义函数 ynp_zeros，用于计算整数阶贝塞尔函数导数 Y_n'(x) 的零点
def ynp_zeros(n, nt):
    r"""Compute zeros of integer-order Bessel function derivatives Yn'(x).

    Compute `nt` zeros of the functions :math:`Y_n'(x)` on the
    interval :math:`(0, \infty)`. The zeros are returned in ascending
    order.

    Parameters
    ----------
    n : int
        Order of Bessel function
    nt : int
        Number of zeros to return

    Returns
    -------
    ndarray
        First `nt` zeros of the Bessel derivative function.

    See Also
    --------
    yvp

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    Examples
    --------
    Compute the first four roots of the first derivative of the
    Bessel function of second kind for order 0 :math:`Y_0'`.

    >>> from scipy.special import ynp_zeros
    >>> ynp_zeros(0, 4)
    array([ 2.19714133,  5.42968104,  8.59600587, 11.74915483])

    Plot :math:`Y_0`, :math:`Y_0'` and confirm visually that the roots of
    :math:`Y_0'` are located at local extrema of :math:`Y_0`.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.special import yn, ynp_zeros, yvp
    >>> zeros = ynp_zeros(0, 4)
    """

    # 调用 jnyn_zeros 函数计算贝塞尔函数和导数的零点，返回第三个元素（即导数的零点）
    return jnyn_zeros(n, nt)[2]
    # 设置 xmax 的值为 13，用于定义 x 轴的范围
    >>> xmax = 13
    
    # 创建一个包含 500 个点的从 0 到 xmax 的均匀间隔的数组
    >>> x = np.linspace(0, xmax, 500)
    
    # 创建一个新的图形和一个包含单个子图(ax)的对象
    >>> fig, ax = plt.subplots()
    
    # 在子图上绘制 Y_0 的图像，使用 yn(0, x) 函数，添加标签为 $Y_0$
    >>> ax.plot(x, yn(0, x), label=r'$Y_0$')
    
    # 在子图上绘制 Y_0' 的图像，使用 yvp(0, x, 1) 函数，添加标签为 $Y_0'$
    >>> ax.plot(x, yvp(0, x, 1), label=r"$Y_0'$")
    
    # 在图上以红色标出 Y_0' 的零点，位置由变量 zeros 控制
    >>> ax.scatter(zeros, np.zeros((4, )), s=30, c='r', label=r"Roots of $Y_0'$", zorder=5)
    
    # 对于每一个零点 root，在图上画出 Y_0 在该点处的极值，使用 yn(0, root) 函数计算极值
    >>> for root in zeros:
    ...     y0_extremum =  yn(0, root)
    ...     lower = min(0, y0_extremum)
    ...     upper = max(0, y0_extremum)
    ...     ax.vlines(root, lower, upper, color='r')
    
    # 在图上画出一条水平线，表示 y=0 的位置
    >>> ax.hlines(0, 0, xmax, color='k')
    
    # 设置 y 轴的显示范围为 -0.6 到 0.6
    >>> ax.set_ylim(-0.6, 0.6)
    
    # 设置 x 轴的显示范围为 0 到 xmax
    >>> ax.set_xlim(0, xmax)
    
    # 在图上添加图例
    >>> plt.legend()
    
    # 显示图形
    >>> plt.show()
# 计算贝塞尔函数 Y0(z) 的前 nt 个零点及每个零点处的导数值。
# 导数由 Y0'(z0) = -Y1(z0) 给出。

def y0_zeros(nt, complex=False):
    """Compute nt zeros of Bessel function Y0(z), and derivative at each zero.

    The derivatives are given by Y0'(z0) = -Y1(z0) at each zero z0.

    Parameters
    ----------
    nt : int
        Number of zeros to return
    complex : bool, default False
        Set to False to return only the real zeros; set to True to return only
        the complex zeros with negative real part and positive imaginary part.
        Note that the complex conjugates of the latter are also zeros of the
        function, but are not returned by this routine.

    Returns
    -------
    z0n : ndarray
        Location of nth zero of Y0(z)
    y0pz0n : ndarray
        Value of derivative Y0'(z0) for nth zero

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    Examples
    --------
    Compute the first 4 real roots and the derivatives at the roots of
    :math:`Y_0`:

    >>> import numpy as np
    >>> from scipy.special import y0_zeros
    >>> zeros, grads = y0_zeros(4)
    >>> with np.printoptions(precision=5):
    ...     print(f"Roots: {zeros}")
    ...     print(f"Gradients: {grads}")
    Roots: [ 0.89358+0.j  3.95768+0.j  7.08605+0.j 10.22235+0.j]
    Gradients: [-0.87942+0.j  0.40254+0.j -0.3001 +0.j  0.2497 +0.j]

    Plot the real part of :math:`Y_0` and the first four computed roots.

    >>> import matplotlib.pyplot as plt
    >>> from scipy.special import y0
    >>> xmin = 0
    >>> xmax = 11
    >>> x = np.linspace(xmin, xmax, 500)
    >>> fig, ax = plt.subplots()
    >>> ax.hlines(0, xmin, xmax, color='k')
    >>> ax.plot(x, y0(x), label=r'$Y_0$')
    >>> zeros, grads = y0_zeros(4)
    >>> ax.scatter(zeros.real, np.zeros((4, )), s=30, c='r',
    ...            label=r'$Y_0$_zeros', zorder=5)
    >>> ax.set_ylim(-0.5, 0.6)
    >>> ax.set_xlim(xmin, xmax)
    >>> plt.legend(ncol=2)
    >>> plt.show()

    Compute the first 4 complex roots and the derivatives at the roots of
    :math:`Y_0` by setting ``complex=True``:

    >>> y0_zeros(4, True)
    (array([ -2.40301663+0.53988231j,  -5.5198767 +0.54718001j,
             -8.6536724 +0.54841207j, -11.79151203+0.54881912j]),
     array([ 0.10074769-0.88196771j, -0.02924642+0.5871695j ,
             0.01490806-0.46945875j, -0.00937368+0.40230454j]))
    """
    # 检查输入的 nt 是否为正整数标量
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("Arguments must be scalar positive integer.")
    # kf 为 0，kc 根据 complex 参数确定是否为 False
    kf = 0
    kc = not complex
    # 调用底层函数 _specfun.cyzo 计算 Y0(z) 的零点及导数
    return _specfun.cyzo(nt, kf, kc)


# 计算贝塞尔函数 Y1(z) 的前 nt 个零点及每个零点处的导数值。
# 导数由 Y1'(z1) = Y0(z1) 给出。

def y1_zeros(nt, complex=False):
    """Compute nt zeros of Bessel function Y1(z), and derivative at each zero.

    The derivatives are given by Y1'(z1) = Y0(z1) at each zero z1.

    Parameters
    ----------
    nt : int
        Number of zeros to return
    # 检查输入参数 nt 是否为标量的正整数，若不是则抛出数值错误异常
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("Arguments must be scalar positive integer.")
    # 根据 complex 参数确定是否计算复数根的情况
    kf = 1
    kc = not complex
    # 调用底层 cyzo 函数来计算特殊函数中的零点位置和导数值
    return _specfun.cyzo(nt, kf, kc)
# 根据 AMS55 提供的公式计算贝塞尔函数的导数差分表达式
def _bessel_diff_formula(v, z, n, L, phase):
    # 将 v 转换为数组形式，以便处理
    v = asarray(v)
    # 初始化系数 p
    p = 1.0
    # 根据给定的函数 L 计算 L(v-n, z)，这里 L 可以是贝塞尔函数的一种
    s = L(v-n, z)


这段代码实现了一个函数 `_bessel_diff_formula`，它根据 AMS55 提供的公式计算贝塞尔函数的导数差分表达式。在这个函数中，参数 `v` 是贝塞尔函数的阶数或者是一个数组，`z` 是函数中的一个参数，`n` 是一个整数用于调整函数 `L` 的参数，`L` 是一个函数，可以是贝塞尔函数的一种，用来计算特定的函数值，`phase` 是一个指示使用哪种函数的标志。
    for i in range(1, n+1):
        # 计算组合数 choose(k, i)，并乘以当前的相位值 phase
        p = phase * (p * (n-i+1)) / i   # = choose(k, i)
        # 将计算得到的 p 乘以 L(v-n + i*2, z)，并加到总和 s 上
        s += p*L(v-n + i*2, z)
    # 返回总和 s 除以 2 的 n 次方
    return s / (2.**n)
# 计算第一类贝塞尔函数的导数。

def jvp(v, z, n=1):
    """Compute derivatives of Bessel functions of the first kind.

    Compute the nth derivative of the Bessel function `Jv` with
    respect to `z`.

    Parameters
    ----------
    v : array_like or float
        Order of Bessel function
    z : complex
        Argument at which to evaluate the derivative; can be real or
        complex.
    n : int, default 1
        Order of derivative. For 0 returns the Bessel function `jv` itself.

    Returns
    -------
    scalar or ndarray
        Values of the derivative of the Bessel function.

    Notes
    -----
    The derivative is computed using the relation DLFM 10.6.7 [2]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    .. [2] NIST Digital Library of Mathematical Functions.
           https://dlmf.nist.gov/10.6.E7

    Examples
    --------

    Compute the Bessel function of the first kind of order 0 and
    its first two derivatives at 1.

    >>> from scipy.special import jvp
    >>> jvp(0, 1, 0), jvp(0, 1, 1), jvp(0, 1, 2)
    (0.7651976865579666, -0.44005058574493355, -0.3251471008130331)

    Compute the first derivative of the Bessel function of the first
    kind for several orders at 1 by providing an array for `v`.

    >>> jvp([0, 1, 2], 1, 1)
    array([-0.44005059,  0.3251471 ,  0.21024362])

    Compute the first derivative of the Bessel function of the first
    kind of order 0 at several points by providing an array for `z`.

    >>> import numpy as np
    >>> points = np.array([0., 1.5, 3.])
    >>> jvp(0, points, 1)
    array([-0.        , -0.55793651, -0.33905896])

    Plot the Bessel function of the first kind of order 1 and its
    first three derivatives.

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-10, 10, 1000)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, jvp(1, x, 0), label=r"$J_1$")
    >>> ax.plot(x, jvp(1, x, 1), label=r"$J_1'$")
    >>> ax.plot(x, jvp(1, x, 2), label=r"$J_1''$")
    >>> ax.plot(x, jvp(1, x, 3), label=r"$J_1'''$")
    >>> plt.legend()
    >>> plt.show()
    """
    # 确保 n 是非负整数
    n = _nonneg_int_or_fail(n, 'n')
    # 若 n 为 0，则返回贝塞尔函数 jv 自身的值
    if n == 0:
        return jv(v, z)
    else:
        # 否则，使用贝塞尔函数的差分公式计算导数
        return _bessel_diff_formula(v, z, n, jv, -1)


# 计算第二类贝塞尔函数的导数。

def yvp(v, z, n=1):
    """Compute derivatives of Bessel functions of the second kind.

    Compute the nth derivative of the Bessel function `Yv` with
    respect to `z`.

    Parameters
    ----------
    v : array_like of float
        Order of Bessel function
    z : complex
        Argument at which to evaluate the derivative
    n : int, default 1
        Order of derivative. For 0 returns the BEssel function `yv`

    Returns
    -------
    scalar or ndarray
        nth derivative of the Bessel function.

    See Also
    --------
    yv : Bessel functions of the second kind
    """
    # 这个函数的实现还未完成，需要添加相应的代码来完成第二类贝塞尔函数的导数计算。
    """
    根据给定的公式计算贝塞尔函数第二类的导数。

    Parameters
    ----------
    v : float or array_like
        贝塞尔函数的阶数。
    z : float or array_like
        计算点的数值。
    n : int
        所需计算的导数阶数。

    Returns
    -------
    float or ndarray
        贝塞尔函数第二类的导数值。

    Notes
    -----
    根据 DLFM 10.6.7 [2]_ 计算导数。

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    .. [2] NIST Digital Library of Mathematical Functions.
           https://dlmf.nist.gov/10.6.E7

    Examples
    --------
    计算阶数为 0 时的贝塞尔函数第二类值。

    >>> from scipy.special import yvp
    >>> yvp(0, 1, 0)
    0.088256964215677

    计算阶数为 1 和 2 时的贝塞尔函数第二类的第一阶导数。

    >>> yvp(0, 1, 1), yvp(0, 1, 2)
    (0.7812128213002889, -0.8694697855159659)

    提供多个阶数计算相同点 1 处的贝塞尔函数第二类的导数。

    >>> yvp([0, 1, 2], 1, 1)
    array([0.78121282, 0.86946979, 2.52015239])

    提供多个点计算阶数为 0 时的贝塞尔函数第二类的导数。

    >>> import numpy as np
    >>> points = np.array([0.5, 1.5, 3.])
    >>> yvp(0, points, 1)
    array([ 1.47147239,  0.41230863, -0.32467442])

    绘制阶数为 1 的贝塞尔函数第二类及其前三阶导数图像。

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0, 5, 1000)
    >>> x[0] += 1e-15
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, yvp(1, x, 0), label=r"$Y_1$")
    >>> ax.plot(x, yvp(1, x, 1), label=r"$Y_1'$")
    >>> ax.plot(x, yvp(1, x, 2), label=r"$Y_1''$")
    >>> ax.plot(x, yvp(1, x, 3), label=r"$Y_1'''$")
    >>> ax.set_ylim(-10, 10)
    >>> plt.legend()
    >>> plt.show()
    """
    n = _nonneg_int_or_fail(n, 'n')
    如果 n 为 0，则直接返回贝塞尔函数第二类的值。
    if n == 0:
        return yv(v, z)
    否则，根据指定的公式计算贝塞尔函数第二类的导数。
    else:
        return _bessel_diff_formula(v, z, n, yv, -1)
# 计算实数阶修改贝塞尔函数 Kv(z) 的导数
def kvp(v, z, n=1):
    # Kv(z) 是第二类修改贝塞尔函数
    # 计算关于 `z` 的导数

    # 确保 `n` 是非负整数
    n = _nonneg_int_or_fail(n, 'n')
    
    # 如果 `n` 为 0，返回贝塞尔函数 `kv` 本身
    if n == 0:
        return kv(v, z)
    else:
        # 否则，使用贝塞尔函数的差分公式计算导数
        return (-1)**n * _bessel_diff_formula(v, z, n, kv, 1)


# 计算第一类修改贝塞尔函数的导数
def ivp(v, z, n=1):
    # 计算第一类修改贝塞尔函数 `Iv` 的第 `n` 阶导数

    # 确保 `n` 是非负整数
    n = _nonneg_int_or_fail(n, 'n')
    
    # 如果 `n` 为 0，返回贝塞尔函数 `iv` 本身
    if n == 0:
        return iv(v, z)
    # 否则，根据输入的参数计算贝塞尔函数的差分公式
    else:
        return (-1)**n * _bessel_diff_formula(v, z, n, iv, 0)
    n = _nonneg_int_or_fail(n, 'n')
    # 调用函数 _nonneg_int_or_fail，确保 n 是非负整数，如果不是则抛出异常
    if n == 0:
        # 如果 n 等于 0，则直接返回修正贝塞尔函数 iv(v, z)
        return iv(v, z)
    else:
        # 否则，调用 _bessel_diff_formula 函数计算修正贝塞尔函数的导数
        # 这里传入的参数包括 v, z, n, iv, 1
        # 返回修正贝塞尔函数的第 n 阶导数
        return _bessel_diff_formula(v, z, n, iv, 1)
# 计算Hankel函数H2v(z)关于z的导数
def h2vp(v, z, n=1):
    # 确保n是非负整数，否则引发异常
    n = _nonneg_int_or_fail(n, 'n')
    # 如果n为0，返回Hankel函数h2v本身的值
    if n == 0:
        return hankel2(v, z)
    else:
        # 否则使用差分公式计算Hankel函数H2v(z)的导数
        return _bessel_diff_formula(v, z, n, hankel2, 1)


这段代码定义了一个函数 `h2vp`，用于计算Hankel函数H2v(z)关于z的导数。根据提供的参数，它可以计算指定阶数n的导数值。
    # 确保 n 是非负整数，如果不是则引发错误
    n = _nonneg_int_or_fail(n, 'n')
    # 如果 n 等于 0，调用 hankel2 函数计算第二类 Hankel 函数 H_v^(2)(z)
    if n == 0:
        return hankel2(v, z)
    else:
        # 否则，使用 _bessel_diff_formula 函数计算 Hankel 函数 H_v^(2)(z) 的导数
        # 这里的 -1 表示计算第一类贝塞尔函数的导数
        return _bessel_diff_formula(v, z, n, hankel2, -1)
def riccati_jn(n, x):
    r"""Compute Ricatti-Bessel function of the first kind and its derivative.

    The Ricatti-Bessel function of the first kind is defined as :math:`x
    j_n(x)`, where :math:`j_n` is the spherical Bessel function of the first
    kind of order :math:`n`.

    This function computes the value and first derivative of the
    Ricatti-Bessel function for all orders up to and including `n`.

    Parameters
    ----------
    n : int
        Maximum order of function to compute
    x : float
        Argument at which to evaluate

    Returns
    -------
    jn : ndarray
        Value of j0(x), ..., jn(x)
    jnp : ndarray
        First derivative j0'(x), ..., jn'(x)

    Notes
    -----
    The computation is carried out via backward recurrence, using the
    relation DLMF 10.51.1 [2]_.

    Wrapper for a Fortran routine created by Shanjie Zhang and Jianming
    Jin [1]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html
    .. [2] NIST Digital Library of Mathematical Functions.
           https://dlmf.nist.gov/10.51.E1

    """
    # 检查参数是否为标量，若不是则引发值错误异常
    if not (isscalar(n) and isscalar(x)):
        raise ValueError("arguments must be scalars.")
    # 将 n 转换为非负整数（如果 n == 0，则 n1 设为 1）
    n = _nonneg_int_or_fail(n, 'n', strict=False)
    if (n == 0):
        n1 = 1
    else:
        n1 = n

    # 创建一个长度为 n1+1 的空数组，用于存储 Ricatti-Bessel 函数值
    jn = np.empty((n1 + 1,), dtype=np.float64)
    # 创建与 jn 相同形状的空数组，用于存储 Ricatti-Bessel 函数的一阶导数值
    jnp = np.empty_like(jn)

    # 调用 _rctj 函数计算 Ricatti-Bessel 函数值和一阶导数，将结果存入 jn 和 jnp 中
    _rctj(x, out=(jn, jnp))
    # 返回计算结果中的前 (n+1) 项
    return jn[:(n+1)], jnp[:(n+1)]


def riccati_yn(n, x):
    """Compute Ricatti-Bessel function of the second kind and its derivative.

    The Ricatti-Bessel function of the second kind is defined as :math:`x
    y_n(x)`, where :math:`y_n` is the spherical Bessel function of the second
    kind of order :math:`n`.

    This function computes the value and first derivative of the function for
    all orders up to and including `n`.

    Parameters
    ----------
    n : int
        Maximum order of function to compute
    x : float
        Argument at which to evaluate

    Returns
    -------
    yn : ndarray
        Value of y0(x), ..., yn(x)
    ynp : ndarray
        First derivative y0'(x), ..., yn'(x)

    Notes
    -----
    The computation is carried out via ascending recurrence, using the
    relation DLMF 10.51.1 [2]_.

    Wrapper for a Fortran routine created by Shanjie Zhang and Jianming
    Jin [1]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html
    .. [2] NIST Digital Library of Mathematical Functions.
           https://dlmf.nist.gov/10.51.E1

    """
    # 检查参数是否为标量，若不是则引发值错误异常
    if not (isscalar(n) and isscalar(x)):
        raise ValueError("arguments must be scalars.")
    # 使用 _nonneg_int_or_fail 函数确保 n 是非负整数，如果 strict=False，可能会返回转换后的整数
    n = _nonneg_int_or_fail(n, 'n', strict=False)
    # 如果 n 等于 0，设置 n1 为 1；否则，n1 等于 n
    if (n == 0):
        n1 = 1
    else:
        n1 = n

    # 创建一个形状为 (n1 + 1,) 的空数组 yn，数据类型为 np.float64
    yn = np.empty((n1 + 1,), dtype=np.float64)
    # 创建一个与 yn 具有相同形状和数据类型的空数组 ynp
    ynp = np.empty_like(yn)
    # 调用 _rcty 函数，将 x 作为输入，并将结果分别存储在 yn 和 ynp 中
    _rcty(x, out=(yn, ynp))

    # 返回 yn 数组的前 (n+1) 个元素和 ynp 数组的前 (n+1) 个元素作为结果
    return yn[:(n+1)], ynp[:(n+1)]
# 计算第一象限中按绝对值排序的前 nt 个误差函数 erf(z) 的零点。

def erf_zeros(nt):
    """Compute the first nt zero in the first quadrant, ordered by absolute value.

    Zeros in the other quadrants can be obtained by using the symmetries
    erf(-z) = erf(z) and erf(conj(z)) = conj(erf(z)).

    Parameters
    ----------
    nt : int
        The number of zeros to compute

    Returns
    -------
    The locations of the zeros of erf : ndarray (complex)
        Complex values at which zeros of erf(z)

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    Examples
    --------
    >>> from scipy import special
    >>> special.erf_zeros(1)
    array([1.45061616+1.880943j])

    Check that erf is (close to) zero for the value returned by erf_zeros

    >>> special.erf(special.erf_zeros(1))
    array([4.95159469e-14-1.16407394e-16j])

    """
    # 检查输入参数是否为正整数标量
    if (floor(nt) != nt) or (nt <= 0) or not isscalar(nt):
        raise ValueError("Argument must be positive scalar integer.")
    # 调用底层函数计算 erf(z) 的零点
    return _specfun.cerzo(nt)


# 计算余弦 Fresnel 积分 C(z) 的 nt 个复数零点。

def fresnelc_zeros(nt):
    """Compute nt complex zeros of cosine Fresnel integral C(z).

    Parameters
    ----------
    nt : int
        Number of zeros to compute

    Returns
    -------
    fresnelc_zeros: ndarray
        Zeros of the cosine Fresnel integral

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    # 检查输入参数是否为正整数标量
    if (floor(nt) != nt) or (nt <= 0) or not isscalar(nt):
        raise ValueError("Argument must be positive scalar integer.")
    # 调用底层函数计算余弦 Fresnel 积分 C(z) 的零点
    return _specfun.fcszo(1, nt)


# 计算正弦 Fresnel 积分 S(z) 的 nt 个复数零点。

def fresnels_zeros(nt):
    """Compute nt complex zeros of sine Fresnel integral S(z).

    Parameters
    ----------
    nt : int
        Number of zeros to compute

    Returns
    -------
    fresnels_zeros: ndarray
        Zeros of the sine Fresnel integral

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    # 检查输入参数是否为正整数标量
    if (floor(nt) != nt) or (nt <= 0) or not isscalar(nt):
        raise ValueError("Argument must be positive scalar integer.")
    # 调用底层函数计算正弦 Fresnel 积分 S(z) 的零点
    return _specfun.fcszo(2, nt)


# 计算正弦和余弦 Fresnel 积分 S(z) 和 C(z) 的 nt 个复数零点。

def fresnel_zeros(nt):
    """Compute nt complex zeros of sine and cosine Fresnel integrals S(z) and C(z).

    Parameters
    ----------
    nt : int
        Number of zeros to compute

    Returns
    -------
    zeros_sine: ndarray
        Zeros of the sine Fresnel integral
    zeros_cosine : ndarray
        Zeros of the cosine Fresnel integral

    References
    ----------

    """
    # 检查输入参数是否为正整数标量
    if (floor(nt) != nt) or (nt <= 0) or not isscalar(nt):
        raise ValueError("Argument must be positive scalar integer.")
    # 调用底层函数计算正弦和余弦 Fresnel 积分 S(z) 和 C(z) 的零点
    return _specfun.fcszo(3, nt)
    """
    计算特殊函数，参考文献 [1] 中的说明。
    
    如果输入的nt不是正的标量整数或者不是标量，抛出数值错误异常。
    """
    if (floor(nt) != nt) or (nt <= 0) or not isscalar(nt):
        raise ValueError("Argument must be positive scalar integer.")
    # 调用特殊函数库中的fcszo函数，分别计算对应参数为2和1时的结果，并返回这两个结果作为元组。
    return _specfun.fcszo(2, nt), _specfun.fcszo(1, nt)
def assoc_laguerre(x, n, k=0.0):
    """Compute the generalized (associated) Laguerre polynomial of degree n and order k.

    The polynomial :math:`L^{(k)}_n(x)` is orthogonal over ``[0, inf)``,
    with weighting function ``exp(-x) * x**k`` with ``k > -1``.

    Parameters
    ----------
    x : float or ndarray
        Points where to evaluate the Laguerre polynomial
    n : int
        Degree of the Laguerre polynomial
    k : int
        Order of the Laguerre polynomial

    Returns
    -------
    assoc_laguerre: float or ndarray
        Associated Laguerre polynomial values

    Notes
    -----
    `assoc_laguerre` is a simple wrapper around `eval_genlaguerre`, with
    reversed argument order ``(x, n, k=0.0) --> (n, k, x)``.

    """
    return _ufuncs.eval_genlaguerre(n, k, x)


digamma = psi  # `digamma` function alias for `psi`


def polygamma(n, x):
    r"""Polygamma functions.

    Defined as :math:`\psi^{(n)}(x)` where :math:`\psi` is the
    `digamma` function. See [dlmf]_ for details.

    Parameters
    ----------
    n : array_like
        The order of the derivative of the digamma function; must be
        integral
    x : array_like
        Real valued input

    Returns
    -------
    ndarray
        Function results

    See Also
    --------
    digamma

    References
    ----------
    .. [dlmf] NIST, Digital Library of Mathematical Functions,
        https://dlmf.nist.gov/5.15

    Examples
    --------
    >>> from scipy import special
    >>> x = [2, 3, 25.5]
    >>> special.polygamma(1, x)
    array([ 0.64493407,  0.39493407,  0.03999467])
    >>> special.polygamma(0, x) == special.psi(x)
    array([ True,  True,  True], dtype=bool)

    """
    n, x = asarray(n), asarray(x)  # Convert `n` and `x` to numpy arrays if they are not already
    fac2 = (-1.0)**(n+1) * gamma(n+1.0) * zeta(n+1, x)  # Compute the factorial-related term
    return where(n == 0, psi(x), fac2)  # Return either `psi(x)` or `fac2` based on condition


def mathieu_even_coef(m, q):
    r"""Fourier coefficients for even Mathieu and modified Mathieu functions.

    The Fourier series of the even solutions of the Mathieu differential
    equation are of the form

    .. math:: \mathrm{ce}_{2n}(z, q) = \sum_{k=0}^{\infty} A_{(2n)}^{(2k)} \cos 2kz

    .. math:: \mathrm{ce}_{2n+1}(z, q) =
              \sum_{k=0}^{\infty} A_{(2n+1)}^{(2k+1)} \cos (2k+1)z

    This function returns the coefficients :math:`A_{(2n)}^{(2k)}` for even
    input m=2n, and the coefficients :math:`A_{(2n+1)}^{(2k+1)}` for odd input
    m=2n+1.

    Parameters
    ----------
    m : int
        Order of Mathieu functions.  Must be non-negative.
    q : float (>=0)
        Parameter of Mathieu functions.  Must be non-negative.

    Returns
    -------
    Ak : ndarray
        Even or odd Fourier coefficients, corresponding to even or odd m.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    """
    计算 Mathieu 函数的 Fourier 系数。
    
    参数：
    - m: float，Mathieu 函数的参数 m，必须是标量
    - q: float，Mathieu 函数的参数 q，必须是标量
    
    返回：
    - fc: numpy.ndarray，长度为 km 的 Fourier 系数数组
    
    异常：
    - ValueError: 如果 m 或 q 不是标量，或者 q < 0，或者 m 不是大于等于零的整数
    - RuntimeWarning: 如果计算得到的 km 超过 251，会发出警告
    
    参考：
    .. [2] NIST Digital Library of Mathematical Functions
           https://dlmf.nist.gov/28.4#i
    """
    
    # 检查 m 和 q 是否为标量，如果不是，则抛出异常
    if not (isscalar(m) and isscalar(q)):
        raise ValueError("m and q must be scalars.")
    # 检查 q 是否大于等于 0，如果不是，则抛出异常
    if (q < 0):
        raise ValueError("q >=0")
    # 检查 m 是否为大于等于零的整数，如果不是，则抛出异常
    if (m != floor(m)) or (m < 0):
        raise ValueError("m must be an integer >=0.")
    
    # 根据 q 的大小选择不同的公式计算 qm
    if (q <= 1):
        qm = 7.5 + 56.1*sqrt(q) - 134.7*q + 90.7*sqrt(q)*q
    else:
        qm = 17.0 + 3.1*sqrt(q) - .126*q + .0037*sqrt(q)*q
    
    # 计算 km，并将其四舍五入为最接近的整数
    km = int(qm + 0.5*m)
    
    # 如果 km 超过 251，发出运行时警告
    if km > 251:
        warnings.warn("Too many predicted coefficients.", RuntimeWarning, stacklevel=2)
    
    # 初始化 kd 为 1，如果 m 是奇数，则设置 kd 为 2
    kd = 1
    m = int(floor(m))
    if m % 2:
        kd = 2
    
    # 计算 Mathieu 函数的参数 a
    a = mathieu_a(m, q)
    
    # 调用 _specfun 模块的 fcoef 函数计算 Fourier 系数
    fc = _specfun.fcoef(kd, m, q, a)
    
    # 返回 Fourier 系数数组的前 km 个值
    return fc[:km]
def mathieu_odd_coef(m, q):
    r"""Fourier coefficients for even Mathieu and modified Mathieu functions.

    The Fourier series of the odd solutions of the Mathieu differential
    equation are of the form

    .. math:: \mathrm{se}_{2n+1}(z, q) =
              \sum_{k=0}^{\infty} B_{(2n+1)}^{(2k+1)} \sin (2k+1)z

    .. math:: \mathrm{se}_{2n+2}(z, q) =
              \sum_{k=0}^{\infty} B_{(2n+2)}^{(2k+2)} \sin (2k+2)z

    This function returns the coefficients :math:`B_{(2n+2)}^{(2k+2)}` for even
    input m=2n+2, and the coefficients :math:`B_{(2n+1)}^{(2k+1)}` for odd
    input m=2n+1.

    Parameters
    ----------
    m : int
        Order of Mathieu functions.  Must be non-negative.
    q : float (>=0)
        Parameter of Mathieu functions.  Must be non-negative.

    Returns
    -------
    Bk : ndarray
        Even or odd Fourier coefficients, corresponding to even or odd m.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """

    # Check if m and q are scalars; raise ValueError if not
    if not (isscalar(m) and isscalar(q)):
        raise ValueError("m and q must be scalars.")
    
    # Check if q is non-negative; raise ValueError if q < 0
    if (q < 0):
        raise ValueError("q >= 0")
    
    # Check if m is a positive integer; raise ValueError if not
    if (m != floor(m)) or (m <= 0):
        raise ValueError("m must be an integer > 0")

    # Calculate qm based on q value
    if (q <= 1):
        qm = 7.5 + 56.1 * sqrt(q) - 134.7 * q + 90.7 * sqrt(q) * q
    else:
        qm = 17.0 + 3.1 * sqrt(q) - .126 * q + .0037 * sqrt(q) * q
    
    # Round qm and compute km
    km = int(qm + 0.5 * m)
    
    # Issue warning if km exceeds 251
    if km > 251:
        warnings.warn("Too many predicted coefficients.", RuntimeWarning, stacklevel=2)
    
    # Initialize kd to 4 and adjust if m is odd
    kd = 4
    m = int(floor(m))
    if m % 2:
        kd = 3
    
    # Compute Fourier coefficients using helper functions
    b = mathieu_b(m, q)
    fc = _specfun.fcoef(kd, m, q, b)
    
    # Return coefficients up to km
    return fc[:km]
    """
    Calculate associated Legendre functions of the first kind and their derivatives.

    Parameters
    ----------
    n : int
        Degree of the Legendre function, must be a non-negative integer.
    m : int
        Order of the Legendre function, must satisfy abs(m) <= n.
    z : array_like
        Real-valued argument(s) at which the Legendre function is evaluated.

    Returns
    -------
    p : ndarray
        Values of the associated Legendre functions of the first kind P_n^m(z).
    pd : ndarray
        Values of the derivatives of the associated Legendre functions P_n^m(z).

    Notes
    -----
    Computes associated Legendre functions of the first kind P_n^m(z) and their derivatives
    using the underlying _lpmn function, which handles the computation for scalar and
    array-like inputs.

    The phase convention used for the intervals (1, inf) and (-inf, -1) ensures that
    the result is always real.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html
    .. [2] NIST Digital Library of Mathematical Functions
           https://dlmf.nist.gov/14.3

    """

    # Ensure n is a non-negative integer
    n = _nonneg_int_or_fail(n, 'n', strict=False)

    # Check if m is a scalar and abs(m) is within bounds
    if not isscalar(m) or (abs(m) > n):
        raise ValueError("m must be <= n.")

    # Check if n is a scalar and non-negative
    if not isscalar(n) or (n < 0):
        raise ValueError("n must be a non-negative integer.")

    # Ensure z is real-valued
    if np.iscomplexobj(z):
        raise ValueError("Argument must be real. Use clpmn instead.")

    # Convert m and n to integers (backwards compatibility)
    m, n = int(m), int(n)

    # Determine sign bit and absolute value of m
    if m < 0:
        m_signbit = True
        m_abs = -m
    else:
        m_signbit = False
        m_abs = m

    # Ensure z is an ndarray and of floating point type
    z = np.asarray(z)
    if not np.issubdtype(z.dtype, np.inexact):
        z = z.astype(np.float64)

    # Initialize arrays to store results
    p = np.empty((m_abs + 1, n + 1) + z.shape, dtype=np.float64)
    pd = np.empty_like(p)

    # Compute Legendre functions and their derivatives
    if z.ndim == 0:
        _lpmn(z, m_signbit, out=(p, pd))
    else:
        _lpmn(z, m_signbit, out=(np.moveaxis(p, (0, 1), (-2, -1)),
                                 np.moveaxis(pd, (0, 1), (-2, -1))))  # new axes must be last for the ufunc

    # Return computed Legendre functions and derivatives
    return p, pd
def clpmn(m, n, z, type=3):
    """Associated Legendre function of the first kind for complex arguments.

    Computes the associated Legendre function of the first kind of order m and
    degree n, ``Pmn(z)`` = :math:`P_n^m(z)`, and its derivative, ``Pmn'(z)``.
    Returns two arrays of size ``(m+1, n+1)`` containing ``Pmn(z)`` and
    ``Pmn'(z)`` for all orders from ``0..m`` and degrees from ``0..n``.

    Parameters
    ----------
    m : int
       ``|m| <= n``; the order of the Legendre function.
    n : int
       where ``n >= 0``; the degree of the Legendre function.  Often
       called ``l`` (lower case L) in descriptions of the associated
       Legendre function
    z : array_like, float or complex
        Input value.
    type : int, optional
       takes values 2 or 3
       2: cut on the real axis ``|x| > 1``
       3: cut on the real axis ``-1 < x < 1`` (default)

    Returns
    -------
    Pmn_z : (m+1, n+1) array
       Values for all orders ``0..m`` and degrees ``0..n``
    Pmn_d_z : (m+1, n+1) array
       Derivatives for all orders ``0..m`` and degrees ``0..n``

    See Also
    --------
    lpmn: associated Legendre functions of the first kind for real z

    Notes
    -----
    By default, i.e. for ``type=3``, phase conventions are chosen according
    to [1]_ such that the function is analytic. The cut lies on the interval
    (-1, 1). Approaching the cut from above or below in general yields a phase
    factor with respect to Ferrer's function of the first kind
    (cf. `lpmn`).

    For ``type=2`` a cut at ``|x| > 1`` is chosen. Approaching the real values
    on the interval (-1, 1) in the complex plane yields Ferrer's function
    of the first kind.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html
    .. [2] NIST Digital Library of Mathematical Functions
           https://dlmf.nist.gov/14.21

    """
    # 检查 m 是否为标量且不超过 n
    if not isscalar(m) or (abs(m) > n):
        raise ValueError("m must be <= n.")
    # 检查 n 是否为非负整数
    if not isscalar(n) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    # 检查 type 是否为合法取值
    if not (type == 2 or type == 3):
        raise ValueError("type must be either 2 or 3.")

    # 将 m 和 n 转换为整数以保持向后兼容性
    m, n = int(m), int(n)
    # 处理 m 的正负号
    if (m < 0):
        mp = -m
        m_signbit = True
    else:
        mp = m
        m_signbit = False

    # 将 z 转换为复数数组，如果不是浮点数或复数类型的话
    z = np.asarray(z)
    if (not np.issubdtype(z.dtype, np.inexact)):
        z = z.astype(np.complex128)

    # 创建存储结果的空数组
    p = np.empty((mp + 1, n + 1) + z.shape, dtype=np.complex128)
    pd = np.empty_like(p)

    # 调用底层函数 _clpmn 计算结果
    if (z.ndim == 0):
        _clpmn(z, type, m_signbit, out=(p, pd))
    else:
        _clpmn(z, type, m_signbit, out=(np.moveaxis(p, (0, 1), (-2, -1)),
                                        np.moveaxis(pd, (0, 1), (-2, -1))))  # ufunc 要求新轴在最后

    # 返回计算得到的结果数组
    return p, pd
def lqmn(m, n, z):
    """Sequence of associated Legendre functions of the second kind.

    Computes the associated Legendre function of the second kind of order m and
    degree n, ``Qmn(z)`` = :math:`Q_n^m(z)`, and its derivative, ``Qmn'(z)``.
    Returns two arrays of size ``(m+1, n+1)`` containing ``Qmn(z)`` and
    ``Qmn'(z)`` for all orders from ``0..m`` and degrees from ``0..n``.

    Parameters
    ----------
    m : int
       ``|m| <= n``; the order of the Legendre function.
    n : int
       where ``n >= 0``; the degree of the Legendre function.  Often
       called ``l`` (lower case L) in descriptions of the associated
       Legendre function
    z : array_like, complex
        Input value.

    Returns
    -------
    Qmn_z : (m+1, n+1) array
       Values for all orders 0..m and degrees 0..n
    Qmn_d_z : (m+1, n+1) array
       Derivatives for all orders 0..m and degrees 0..n

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    # 检查 m 是否为非负整数，若不是则引发 ValueError 异常
    if not isscalar(m) or (m < 0):
        raise ValueError("m must be a non-negative integer.")
    # 检查 n 是否为非负整数，若不是则引发 ValueError 异常
    if not isscalar(n) or (n < 0):
        raise ValueError("n must be a non-negative integer.")

    # 将 m 和 n 转换为整数，以保持向后兼容性
    m, n = int(m), int(n)
    # 确保 mm 和 nn 至少为 1，以防止 m 或 n 为 0 的情况
    mm = max(1, m)
    nn = max(1, n)

    # 将输入值 z 转换为 NumPy 数组
    z = np.asarray(z)
    # 如果 z 的数据类型不是浮点数，则转换为 np.float64 类型
    if not np.issubdtype(z.dtype, np.inexact):
        z = z.astype(np.float64)

    # 如果 z 是复数类型，则创建复数类型的数组 q 和 qd
    if np.iscomplexobj(z):
        q = np.empty((mm + 1, nn + 1) + z.shape, dtype=np.complex128)
    else:
        q = np.empty((mm + 1, nn + 1) + z.shape, dtype=np.float64)
    qd = np.empty_like(q)

    # 如果 z 是标量，则调用 _lqmn 函数，并将结果存储在 q 和 qd 中
    if z.ndim == 0:
        _lqmn(z, out=(q, qd))
    else:
        # 如果 z 是数组，则调用 _lqmn 函数，并通过 moveaxis 调整数组的轴顺序
        _lqmn(z, out=(np.moveaxis(q, (0, 1), (-2, -1)),
                      np.moveaxis(qd, (0, 1), (-2, -1))))  # new axes must be last for the ufunc

    # 返回结果数组 q 和 qd 的部分切片，分别为 0 到 m 和 0 到 n 的部分
    return q[:(m+1), :(n+1)], qd[:(m+1), :(n+1)]


def bernoulli(n):
    """Bernoulli numbers B0..Bn (inclusive).

    Parameters
    ----------
    n : int
        Indicated the number of terms in the Bernoulli series to generate.

    Returns
    -------
    ndarray
        The Bernoulli numbers ``[B(0), B(1), ..., B(n)]``.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html
    .. [2] "Bernoulli number", Wikipedia, https://en.wikipedia.org/wiki/Bernoulli_number

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import bernoulli, zeta
    >>> bernoulli(4)
    array([ 1.        , -0.5       ,  0.16666667,  0.        , -0.03333333])

    The Wikipedia article ([2]_) points out the relationship between the

    """
    # 返回前 n+1 个贝努利数的数组
    return np.asarray(bernoulli_numbers)[:n+1]
    """
    计算 Bernoulli 数和 zeta 函数的关系，``B_n^+ = -n * zeta(1 - n)``
    对于 ``n > 0`` 的情况。
    
    >>> n = np.arange(1, 5)
    >>> -n * zeta(1 - n)
    array([ 0.5       ,  0.16666667, -0.        , -0.03333333])
    
    注意，在维基百科文章中使用的符号表示法中，
    `bernoulli` 计算的是 ``B_n^-`` （即使用 ``B_1 = -1/2`` 的约定）。
    上述给出的关系是针对 ``B_n^+`` 的，因此 0.5 的符号与 ``bernoulli(4)`` 的输出不匹配。
    
    """
    if not isscalar(n) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    n = int(n)
    if (n < 2):
        n1 = 2
    else:
        n1 = n
    # 调用 C 库函数计算 Bernoulli 数，并返回结果的切片（长度为 n+1）
    return _specfun.bernob(int(n1))[:(n+1)]
# 计算欧拉数 E(0), E(1), ..., E(n) 的函数
def euler(n):
    """Euler numbers E(0), E(1), ..., E(n).

    The Euler numbers [1]_ are also known as the secant numbers.

    Because ``euler(n)`` returns floating point values, it does not give
    exact values for large `n`.  The first inexact value is E(22).

    Parameters
    ----------
    n : int
        欧拉数的最大索引。

    Returns
    -------
    ndarray
        欧拉数的数组 [E(0), E(1), ..., E(n)]。
        包括所有奇数欧拉数，其值为零。

    References
    ----------
    .. [1] Sequence A122045, The On-Line Encyclopedia of Integer Sequences,
           https://oeis.org/A122045
    .. [2] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import euler
    >>> euler(6)
    array([  1.,   0.,  -1.,   0.,   5.,   0., -61.])

    >>> euler(13).astype(np.int64)
    array([      1,       0,      -1,       0,       5,       0,     -61,
                 0,    1385,       0,  -50521,       0, 2702765,       0])

    >>> euler(22)[-1]  # Exact value of E(22) is -69348874393137901.
    -69348874393137976.0

    """
    # 检查输入是否为非负整数
    if not isscalar(n) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    n = int(n)
    if (n < 2):
        n1 = 2
    else:
        n1 = n
    # 调用特殊函数库计算欧拉数，并返回指定范围内的结果
    return _specfun.eulerb(n1)[:(n+1)]


# 计算第一类Legendre函数的函数
def lpn(n, z):
    """Legendre function of the first kind.

    Compute sequence of Legendre functions of the first kind (polynomials),
    Pn(z) and derivatives for all degrees from 0 to n (inclusive).

    See also special.legendre for polynomial class.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    # 确保输入的 n 是非负整数
    n = _nonneg_int_or_fail(n, 'n', strict=False)

    z = np.asarray(z)
    if (not np.issubdtype(z.dtype, np.inexact)):
        z = z.astype(np.float64)

    # 创建存储结果的数组
    pn = np.empty((n + 1,) + z.shape, dtype=z.dtype)
    pd = np.empty_like(pn)
    if (z.ndim == 0):
        # 调用特殊函数库计算第一类Legendre函数及其导数，并存入 pn 和 pd
        _lpn(z, out=(pn, pd))
    else:
        # 调用特殊函数库计算第一类Legendre函数及其导数，并存入 pn 和 pd
        _lpn(z, out=(np.moveaxis(pn, 0, -1),
                     np.moveaxis(pd, 0, -1)))  # 新的轴必须放在最后用于 ufunc

    return pn, pd


# 计算第二类Legendre函数的函数
def lqn(n, z):
    """Legendre function of the second kind.

    Compute sequence of Legendre functions of the second kind, Qn(z) and
    derivatives for all degrees from 0 to n (inclusive).

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    # 确保输入的 n 是非负整数
    n = _nonneg_int_or_fail(n, 'n', strict=False)
    # 如果 n 小于 1，则将 n1 设置为 1；否则将 n1 设置为 n 的值
    if (n < 1):
        n1 = 1
    else:
        n1 = n

    # 将 z 转换为 NumPy 数组
    z = np.asarray(z)

    # 如果 z 的数据类型不是浮点数或者复数类型的子类型，则将其转换为 float 类型
    if (not np.issubdtype(z.dtype, np.inexact)):
        z = z.astype(float)

    # 根据 z 是否为复数类型的对象，创建对应的空数组 qn 和 qd
    if np.iscomplexobj(z):
        qn = np.empty((n1 + 1,) + z.shape, dtype=np.complex128)
    else:
        qn = np.empty((n1 + 1,) + z.shape, dtype=np.float64)
    qd = np.empty_like(qn)

    # 如果 z 的维度为 0，则调用 _lqn 函数，将结果存储在 (qn, qd) 中
    if (z.ndim == 0):
        _lqn(z, out=(qn, qd))
    else:
        # 如果 z 的维度大于 0，则调用 _lqn 函数，将结果存储在移动轴后的数组中
        _lqn(z, out=(np.moveaxis(qn, 0, -1),
                     np.moveaxis(qd, 0, -1)))  # ufunc 需要新的轴位于最后

    # 返回前 (n+1) 项的 qn 和 qd
    return qn[:(n+1)], qd[:(n+1)]
# 定义函数 `ai_zeros`，计算 Airy 函数 Ai 及其导数的前 `nt` 个零点和对应的值
def ai_zeros(nt):
    # 设置指示函数类型的标志 kf，对于 Ai 函数设置为 1
    kf = 1
    # 检查输入参数 nt 是否为正整数标量，否则抛出数值错误异常
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be a positive integer scalar.")
    # 调用 C 函数 `_specfun.airyzo`，返回计算结果
    return _specfun.airyzo(nt, kf)


# 定义函数 `bi_zeros`，计算 Airy 函数 Bi 及其导数的前 `nt` 个零点和对应的值
def bi_zeros(nt):
    # 设置指示函数类型的标志 kf，对于 Bi 函数设置为 2
    kf = 2
    # 检查输入参数 nt 是否为正整数标量，否则抛出数值错误异常
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be a positive integer scalar.")
    # 调用 C 函数 `_specfun.airyzo`，返回计算结果
    return _specfun.airyzo(nt, kf)


# 定义函数 `lmbda(v, x)`
    r"""Jahnke-Emden Lambda function, Lambdav(x).

    This function is defined as [2]_,

    .. math:: \Lambda_v(x) = \Gamma(v+1) \frac{J_v(x)}{(x/2)^v},

    where :math:`\Gamma` is the gamma function and :math:`J_v` is the
    Bessel function of the first kind.

    Parameters
    ----------
    v : float
        Order of the Lambda function
    x : float
        Value at which to evaluate the function and derivatives

    Returns
    -------
    vl : ndarray
        Values of Lambda_vi(x), for vi=v-int(v), vi=1+v-int(v), ..., vi=v.
    dl : ndarray
        Derivatives Lambda_vi'(x), for vi=v-int(v), vi=1+v-int(v), ..., vi=v.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html
    .. [2] Jahnke, E. and Emde, F. "Tables of Functions with Formulae and
           Curves" (4th ed.), Dover, 1945
    """
    # 检查参数 v 和 x 是否为标量（scalar），若不是则抛出 ValueError 异常
    if not (isscalar(v) and isscalar(x)):
        raise ValueError("arguments must be scalars.")
    # 检查 v 是否小于 0，若是则抛出 ValueError 异常
    if (v < 0):
        raise ValueError("argument must be > 0.")
    # 将 v 转换为整数 n，v0 表示 v 减去 n 的小数部分
    n = int(v)
    v0 = v - n
    # 根据 n 的值确定 n1 的取值
    if (n < 1):
        n1 = 1
    else:
        n1 = n
    # 计算 v1，为 n1 + v0
    v1 = n1 + v0
    # 根据 v 是否为整数调用不同的函数计算 Lambda 函数及其导数
    if (v != floor(v)):
        vm, vl, dl = _specfun.lamv(v1, x)
    else:
        vm, vl, dl = _specfun.lamn(v1, x)
    # 返回结果 vl 和 dl 的前 (n+1) 项
    return vl[:(n+1)], dl[:(n+1)]
# 定义函数 pbdv_seq，计算抛物柱函数 Dv(x) 及其导数。

def pbdv_seq(v, x):
    """Parabolic cylinder functions Dv(x) and derivatives.

    Parameters
    ----------
    v : float
        抛物柱函数的阶数
    x : float
        待求函数及导数的值

    Returns
    -------
    dv : ndarray
        D_vi(x) 的值，其中 vi=v-int(v), vi=1+v-int(v), ..., vi=v.
    dp : ndarray
        D_vi'(x) 的导数，其中 vi=v-int(v), vi=1+v-int(v), ..., vi=v.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 13.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    # 检查输入参数是否为标量
    if not (isscalar(v) and isscalar(x)):
        raise ValueError("arguments must be scalars.")
    # 取 v 的整数部分
    n = int(v)
    # 计算 v 的小数部分
    v0 = v - n
    # 根据整数部分 n 计算 n1
    if (n < 1):
        n1 = 1
    else:
        n1 = n
    # 计算 v1
    v1 = n1 + v0
    # 调用底层函数 _specfun.pbdv 计算 Dv(x) 和其导数
    dv, dp, pdf, pdd = _specfun.pbdv(v1, x)
    # 返回结果，截取前 n1+1 个元素
    return dv[:n1+1], dp[:n1+1]


# 定义函数 pbvv_seq，计算抛物柱函数 Vv(x) 及其导数。

def pbvv_seq(v, x):
    """Parabolic cylinder functions Vv(x) and derivatives.

    Parameters
    ----------
    v : float
        抛物柱函数的阶数
    x : float
        待求函数及导数的值

    Returns
    -------
    dv : ndarray
        V_vi(x) 的值，其中 vi=v-int(v), vi=1+v-int(v), ..., vi=v.
    dp : ndarray
        V_vi'(x) 的导数，其中 vi=v-int(v), vi=1+v-int(v), ..., vi=v.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 13.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    # 检查输入参数是否为标量
    if not (isscalar(v) and isscalar(x)):
        raise ValueError("arguments must be scalars.")
    # 取 v 的整数部分
    n = int(v)
    # 计算 v 的小数部分
    v0 = v - n
    # 根据整数部分 n 计算 n1
    if (n <= 1):
        n1 = 1
    else:
        n1 = n
    # 计算 v1
    v1 = n1 + v0
    # 调用底层函数 _specfun.pbvv 计算 Vv(x) 和其导数
    dv, dp, pdf, pdd = _specfun.pbvv(v1, x)
    # 返回结果，截取前 n1+1 个元素
    return dv[:n1+1], dp[:n1+1]


# 定义函数 pbdn_seq，计算抛物柱函数 Dn(z) 及其导数。

def pbdn_seq(n, z):
    """Parabolic cylinder functions Dn(z) and derivatives.

    Parameters
    ----------
    n : int
        抛物柱函数的阶数
    z : complex
        待求函数及导数的值

    Returns
    -------
    dv : ndarray
        D_i(z) 的值，其中 i=0, ..., i=n.
    dp : ndarray
        D_i'(z) 的导数，其中 i=0, ..., i=n.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 13.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    # 检查输入参数是否为标量
    if not (isscalar(n) and isscalar(z)):
        raise ValueError("arguments must be scalars.")
    # 检查 n 是否为整数
    if (floor(n) != n):
        raise ValueError("n must be an integer.")
    # 判断 n 的绝对值是否小于等于 1，确定 n1
    if (abs(n) <= 1):
        n1 = 1
    else:
        n1 = n
    # 调用底层函数 _specfun.cpbdn 计算 Dn(z) 和其导数
    cpb, cpd = _specfun.cpbdn(n1, z)
    # 返回结果，截取前 n1+1 个元素
    return cpb[:n1+1], cpd[:n1+1]
    # 计算 Kelvin 函数 ber 的前 nt 个零点

    # 检查 nt 是否为标量且为正整数，如果不是则抛出 ValueError
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    
    # 调用底层函数 _specfun.klvnzo(nt, 1)，计算 Kelvin 函数 ber 的前 nt 个零点并返回结果
    return _specfun.klvnzo(nt, 1)
# 计算 Kelvin 函数 bei 的前 nt 个零点

def bei_zeros(nt):
    """Compute nt zeros of the Kelvin function bei.

    Parameters
    ----------
    nt : int
        Number of zeros to compute. Must be positive.

    Returns
    -------
    ndarray
        First `nt` zeros of the Kelvin function.

    See Also
    --------
    bei

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    # 检查输入参数 nt 是否为正整数标量，否则抛出值错误异常
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    # 调用底层特殊函数库中的函数，返回 Kelvin 函数 bei 的前 nt 个零点
    return _specfun.klvnzo(nt, 2)


# 计算 Kelvin 函数 ker 的前 nt 个零点

def ker_zeros(nt):
    """Compute nt zeros of the Kelvin function ker.

    Parameters
    ----------
    nt : int
        Number of zeros to compute. Must be positive.

    Returns
    -------
    ndarray
        First `nt` zeros of the Kelvin function.

    See Also
    --------
    ker

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    # 检查输入参数 nt 是否为正整数标量，否则抛出值错误异常
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    # 调用底层特殊函数库中的函数，返回 Kelvin 函数 ker 的前 nt 个零点
    return _specfun.klvnzo(nt, 3)


# 计算 Kelvin 函数 kei 的前 nt 个零点

def kei_zeros(nt):
    """Compute nt zeros of the Kelvin function kei.

    Parameters
    ----------
    nt : int
        Number of zeros to compute. Must be positive.

    Returns
    -------
    ndarray
        First `nt` zeros of the Kelvin function.

    See Also
    --------
    kei

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    # 检查输入参数 nt 是否为正整数标量，否则抛出值错误异常
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    # 调用底层特殊函数库中的函数，返回 Kelvin 函数 kei 的前 nt 个零点
    return _specfun.klvnzo(nt, 4)


# 计算 Kelvin 函数 ber 的导数的前 nt 个零点

def berp_zeros(nt):
    """Compute nt zeros of the derivative of the Kelvin function ber.

    Parameters
    ----------
    nt : int
        Number of zeros to compute. Must be positive.

    Returns
    -------
    ndarray
        First `nt` zeros of the derivative of the Kelvin function.

    See Also
    --------
    ber, berp

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    # 检查输入参数 nt 是否为正整数标量，否则抛出值错误异常
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    # 调用底层特殊函数库中的函数，返回 Kelvin 函数 ber 的导数的前 nt 个零点
    return _specfun.klvnzo(nt, 5)


# 计算 Kelvin 函数 bei 的导数的前 nt 个零点

def beip_zeros(nt):
    """Compute nt zeros of the derivative of the Kelvin function bei.

    Parameters
    ----------
    nt : int
        Number of zeros to compute. Must be positive.

    Returns
    -------
    ndarray
        First `nt` zeros of the derivative of the Kelvin function.

    See Also
    --------
    bei

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    # 检查输入参数 nt 是否为标量且为正整数
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        # 如果不满足条件，抛出数值错误异常
        raise ValueError("nt must be positive integer scalar.")
    # 调用特殊函数模块中的 klvnzo 函数，计算Kelvin函数的导数的前 `nt` 个零点
    return _specfun.klvnzo(nt, 6)
# 计算 Kelvin 函数 ker 的导数的前 nt 个零点

def kerp_zeros(nt):
    """Compute nt zeros of the derivative of the Kelvin function ker.

    Parameters
    ----------
    nt : int
        Number of zeros to compute. Must be positive.

    Returns
    -------
    ndarray
        First `nt` zeros of the derivative of the Kelvin function.

    See Also
    --------
    ker, kerp

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    # 检查输入参数 nt 是否为正整数标量
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    # 调用特殊函数库中的函数，计算 Kelvin 函数 ker 的导数前 nt 个零点
    return _specfun.klvnzo(nt, 7)


# 计算 Kelvin 函数 kei 的导数的前 nt 个零点

def keip_zeros(nt):
    """Compute nt zeros of the derivative of the Kelvin function kei.

    Parameters
    ----------
    nt : int
        Number of zeros to compute. Must be positive.

    Returns
    -------
    ndarray
        First `nt` zeros of the derivative of the Kelvin function.

    See Also
    --------
    kei, keip

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    # 检查输入参数 nt 是否为正整数标量
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    # 调用特殊函数库中的函数，计算 Kelvin 函数 kei 的导数前 nt 个零点
    return _specfun.klvnzo(nt, 8)


# 计算所有 Kelvin 函数的前 nt 个零点

def kelvin_zeros(nt):
    """Compute nt zeros of all Kelvin functions.

    Returned in a length-8 tuple of arrays of length nt.  The tuple contains
    the arrays of zeros of (ber, bei, ker, kei, ber', bei', ker', kei').

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    # 检查输入参数 nt 是否为正整数标量
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    # 调用特殊函数库中的函数，计算所有 Kelvin 函数的前 nt 个零点，返回为一个包含8个数组的元组
    return (_specfun.klvnzo(nt, 1),
            _specfun.klvnzo(nt, 2),
            _specfun.klvnzo(nt, 3),
            _specfun.klvnzo(nt, 4),
            _specfun.klvnzo(nt, 5),
            _specfun.klvnzo(nt, 6),
            _specfun.klvnzo(nt, 7),
            _specfun.klvnzo(nt, 8))


# 计算椭球波函数的特征值序列

def pro_cv_seq(m, n, c):
    """Characteristic values for prolate spheroidal wave functions.

    Compute a sequence of characteristic values for the prolate
    spheroidal wave functions for mode m and n'=m..n and spheroidal
    parameter c.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    # 检查输入参数 m, n, c 是否为标量
    if not (isscalar(m) and isscalar(n) and isscalar(c)):
        raise ValueError("Arguments must be scalars.")
    # 检查 n 和 m 是否为整数，如果不是则抛出数值错误异常
    if (n != floor(n)) or (m != floor(m)):
        raise ValueError("Modes must be integers.")
    # 检查 n 和 m 的差是否超过 199，如果超过则抛出数值错误异常
    if (n-m > 199):
        raise ValueError("Difference between n and m is too large.")
    # 计算最大的 L 值，即 n - m + 1
    maxL = n - m + 1
    # 调用特定函数 _specfun.segv(m, n, c, 1)，获取返回结果的第二个元素的前 maxL 个元素
    return _specfun.segv(m, n, c, 1)[1][:maxL]
# 计算椭球体波函数的特征值序列。

# 检查输入参数 m, n, c 是否为标量，若不是则引发 ValueError 异常。
def obl_cv_seq(m, n, c):
    """Characteristic values for oblate spheroidal wave functions.

    Compute a sequence of characteristic values for the oblate
    spheroidal wave functions for mode m and n'=m..n and spheroidal
    parameter c.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    if not (isscalar(m) and isscalar(n) and isscalar(c)):
        raise ValueError("Arguments must be scalars.")
    
    # 检查 m 和 n 是否为整数，若不是则引发 ValueError 异常。
    if (n != floor(n)) or (m != floor(m)):
        raise ValueError("Modes must be integers.")
    
    # 检查 n-m 是否大于 199，若是则引发 ValueError 异常。
    if (n-m > 199):
        raise ValueError("Difference between n and m is too large.")
    
    # 计算最大 L 值
    maxL = n-m+1
    
    # 调用 _specfun.segv 函数计算椭球体波函数的特征值序列，返回第二个元素中的前 maxL 个值。
    return _specfun.segv(m, n, c, -1)[1][:maxL]


# 计算从 N 个物体中取出 k 个的组合数。

# 检查是否开启了精确计算模式（exact=True），若是则提醒此特性将在未来版本中移除。
def comb(N, k, *, exact=False, repetition=False):
    """The number of combinations of N things taken k at a time.

    This is often expressed as "N choose k".

    Parameters
    ----------
    N : int, ndarray
        Number of things.
    k : int, ndarray
        Number of elements taken.
    exact : bool, optional
        For integers, if `exact` is False, then floating point precision is
        used, otherwise the result is computed exactly.

        .. deprecated:: 1.14.0
            ``exact=True`` is deprecated for non-integer `N` and `k` and will raise an
            error in SciPy 1.16.0
    repetition : bool, optional
        If `repetition` is True, then the number of combinations with
        repetition is computed.

    Returns
    -------
    val : int, float, ndarray
        The total number of combinations.

    See Also
    --------
    binom : Binomial coefficient considered as a function of two real
            variables.

    Notes
    -----
    - Array arguments accepted only for exact=False case.
    - If N < 0, or k < 0, then 0 is returned.
    - If k > N and repetition=False, then 0 is returned.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import comb
    >>> k = np.array([3, 4])
    >>> n = np.array([10, 10])
    >>> comb(n, k, exact=False)
    array([ 120.,  210.])
    >>> comb(10, 3, exact=True)
    120
    >>> comb(10, 3, exact=True, repetition=True)
    220

    """
    # 若 repetition=True，则计算带重复的组合数，返回结果。
    if repetition:
        return comb(N + k - 1, k, exact=exact)
    
    # 若 exact=True，则根据 N 和 k 的类型返回整数或浮点数，如果是非整数则发出警告并返回整数结果。
    if exact:
        if int(N) == N and int(k) == k:
            # _comb_int 将输入转换为整数，这里是安全且预期的行为。
            return _comb_int(N, k)
        # 否则，忽略 exact=True；对于非整数参数，此设置无意义。
        msg = ("`exact=True` is deprecated for non-integer `N` and `k` and will raise "
               "an error in SciPy 1.16.0")
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return comb(N, k)
    else:
        # 将 k 和 N 转换为 NumPy 数组
        k, N = asarray(k), asarray(N)
        # 创建条件数组，判断 k <= N，N >= 0，k >= 0
        cond = (k <= N) & (N >= 0) & (k >= 0)
        # 计算二项式系数，vals 可能是数组或者标量
        vals = binom(N, k)
        # 如果 vals 是数组，则将不满足条件的位置设为 0
        if isinstance(vals, np.ndarray):
            vals[~cond] = 0
        # 如果 vals 不是数组且条件不满足，则将 vals 设为 0 的浮点数
        elif not cond:
            vals = np.float64(0)
        # 返回计算得到的 vals
        return vals
# 计算 N 个元素中取出 k 个元素的排列数，即部分排列数。

def perm(N, k, exact=False):
    """Permutations of N things taken k at a time, i.e., k-permutations of N.

    It's also known as "partial permutations".

    Parameters
    ----------
    N : int, ndarray
        Number of things.
    k : int, ndarray
        Number of elements taken.
    exact : bool, optional
        If ``True``, calculate the answer exactly using long integer arithmetic (`N`
        and `k` must be scalar integers). If ``False``, a floating point approximation
        is calculated (more rapidly) using `poch`. Default is ``False``.

    Returns
    -------
    val : int, ndarray
        The number of k-permutations of N.

    Notes
    -----
    - Array arguments accepted only for exact=False case.
    - If k > N, N < 0, or k < 0, then a 0 is returned.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import perm
    >>> k = np.array([3, 4])
    >>> n = np.array([10, 10])
    >>> perm(n, k)
    array([  720.,  5040.])
    >>> perm(10, 3, exact=True)
    720

    """
    
    # 如果 exact=True，使用精确的长整数算术计算排列数
    if exact:
        N = np.squeeze(N)[()]  # for backward compatibility (accepted size 1 arrays)
        k = np.squeeze(k)[()]
        if not (isscalar(N) and isscalar(k)):
            raise ValueError("`N` and `k` must scalar integers be with `exact=True`.")

        floor_N, floor_k = int(N), int(k)
        non_integral = not (floor_N == N and floor_k == k)
        
        # 检查是否存在非整数的情况
        if (k > N) or (N < 0) or (k < 0):
            if non_integral:
                msg = ("Non-integer `N` and `k` with `exact=True` is deprecated and "
                       "will raise an error in SciPy 1.16.0.")
                warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return 0
        
        # 如果存在非整数情况，抛出异常
        if non_integral:
            raise ValueError("Non-integer `N` and `k` with `exact=True` is not "
                             "supported.")
        
        # 计算排列数的值
        val = 1
        for i in range(floor_N - floor_k + 1, floor_N + 1):
            val *= i
        return val
    
    # 如果 exact=False，使用浮点数近似计算排列数
    else:
        k, N = asarray(k), asarray(N)
        cond = (k <= N) & (N >= 0) & (k >= 0)
        
        # 计算排列数的近似值
        vals = poch(N - k + 1, k)
        
        # 处理结果不符合条件的情况
        if isinstance(vals, np.ndarray):
            vals[~cond] = 0
        elif not cond:
            vals = np.float64(0)
        
        return vals


# https://stackoverflow.com/a/16327037
def _range_prod(lo, hi, k=1):
    """
    Product of a range of numbers spaced k apart (from hi).

    For k=1, this returns the product of
    lo * (lo+1) * (lo+2) * ... * (hi-2) * (hi-1) * hi
    = hi! / (lo-1)!

    For k>1, it correspond to taking only every k'th number when
    counting down from hi - e.g. 18!!!! = _range_prod(1, 18, 4).

    Breaks into smaller products first for speed:
    _range_prod(2, 9) = ((2*3)*(4*5))*((6*7)*(8*9))
    """
    
    # 递归计算给定范围内以 k 为步长的数的乘积
    if lo + k < hi:
        mid = (hi + lo) // 2
        if k > 1:
            # 确保 mid 是离 hi 最近的 k 的倍数
            mid = mid - ((mid - hi) % k)
        
        # 分解为更小的乘积以提高速度
        return _range_prod(lo, mid, k) * _range_prod(mid + k, hi, k)
    # 如果 lo + k 等于 hi，则返回 lo 乘以 hi 的结果
    elif lo + k == hi:
        return lo * hi
    # 否则，返回 hi 的值
    else:
        return hi
# 精确计算数组 n 的阶乘（包括多阶乘），使用不同的数据类型处理不同情况
def _factorialx_array_exact(n, k=1):
    """
    Exact computation of factorial for an array.

    The factorials are computed in incremental fashion, by taking
    the sorted unique values of n and multiplying the intervening
    numbers between the different unique values.

    In other words, the factorial for the largest input is only
    computed once, with each other result computed in the process.

    k > 1 corresponds to the multifactorial.
    """
    # 从数组 n 中获取唯一值并排序
    un = np.unique(n)
    
    # 处理 numpy 1.21 版本之后的 nan 排序行为变化问题，删除数组中的 nan
    un = un[~np.isnan(un)]
    
    # 如果 n 中包含 nan，则将数据类型设为 float
    if np.isnan(n).any():
        dt = float
    # 根据 k 的不同取值确定数据类型
    elif k in _FACTORIALK_LIMITS_64BITS.keys():
        if un[-1] > _FACTORIALK_LIMITS_64BITS[k]:
            # 当最大值超过 np.int64 的限制时，使用 object 类型
            dt = object
        elif un[-1] > _FACTORIALK_LIMITS_32BITS[k]:
            # 当最大值超过 np.int32 的限制时，使用 np.int64 类型
            dt = np.int64
        else:
            # 否则使用 np.long 类型
            dt = np.dtype("long")
    else:
        # 对于 k >= 10，始终使用 object 类型
        dt = object
    
    # 创建一个和 n 数组相同大小的空数组，使用指定的数据类型 dt
    out = np.empty_like(n, dtype=dt)
    
    # 处理无效/特殊情况下的值
    un = un[un > 1]  # 筛选出大于 1 的值
    out[n < 2] = 1    # 对于 n < 2 的情况，阶乘值为 1
    out[n < 0] = 0    # 对于 n < 0 的情况，阶乘值为 0
    
    # 计算每个数值范围内的乘积
    # 如果值相差 k，则可以递增地进行乘积计算；因此将 un 分为 "lane"，即其 k 的余数
    for lane in range(0, k):
        ul = un[(un % k) == lane] if k > 1 else un
        if ul.size:
            # 对于每个唯一值 ul 中的第一个值，计算其阶乘
            val = _range_prod(1, int(ul[0]), k=k)
            out[n == ul[0]] = val
            for i in range(len(ul) - 1):
                # 对于 ul 中的连续值，递增计算其阶乘
                prev = ul[i]
                current = ul[i + 1]
                val *= _range_prod(int(prev + 1), int(current), k=k)
                out[n == current] = val
    
    # 如果 n 中包含 nan，则将输出结果转换为 np.float64 类型，并将 nan 值保留
    if np.isnan(n).any():
        out = out.astype(np.float64)
        out[np.isnan(n)] = np.nan
    
    # 返回计算得到的阶乘数组
    return out


# 为数组 n 和整数 k 计算近似的多阶乘值
def _factorialx_array_approx(n, k):
    """
    Calculate approximation to multifactorial for array n and integer k.

    Ensure we only call _factorialx_approx_core where necessary/required.
    """
    # 创建与 n 数组形状相同的结果数组，初始值为 0，保持 nans 不变
    result = zeros(n.shape)
    place(result, np.isnan(n), np.nan)
    
    # 仅计算 n >= 0 的情况（不包括 nans），其余情况结果为 0
    cond = (n >= 0)
    n_to_compute = extract(cond, n)
    # 调用函数 place，并传入 result、cond 和 _factorialx_approx_core(n_to_compute, k=k) 作为参数
    place(result, cond, _factorialx_approx_core(n_to_compute, k=k))
    # 返回变量 result 的值作为函数的结果
    return result
def factorial(n, exact=False):
    """
    The factorial of a number or array of numbers.

    The factorial of non-negative integer `n` is the product of all
    positive integers less than or equal to `n`::

        n! = n * (n - 1) * (n - 2) * ... * 1

    Parameters
    ----------
    n : int or array_like of ints
        Input values.  If ``n < 0``, the return value is 0.
    exact : bool, optional
        If True, calculate the answer exactly using long integer arithmetic.
        If False, result is approximated in floating point rapidly using the
        `gamma` function.
        Default is False.

    Returns
    -------
    nf : float or int or ndarray
        Factorial of `n`, as integer or float depending on `exact`.

    Notes
    -----
    For arrays with ``exact=True``, the factorial is computed only once, for
    the largest input, with each other result computed in the process.
    The output dtype is increased to ``int64`` or ``object`` if necessary.

    With ``exact=False`` the factorial is approximated using the gamma
    function:

    .. math:: n! = \\Gamma(n+1)

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import factorial
    >>> arr = np.array([3, 4, 5])
    >>> factorial(arr, exact=False)
    array([   6.,   24.,  120.])
    >>> factorial(arr, exact=True)
    array([  6,  24, 120])
    >>> factorial(5, exact=True)
    120

    """
    # don't use isscalar due to numpy/numpy#23574; 0-dim arrays treated below
    # 检查输入 `n` 是否为标量（scalar），因为numpy的一些问题，不能直接使用 isscalar
    # 处理 0 维数组的情况，以及统一处理数组和标量的计算效率
    if not isinstance(n, np.ndarray):
        # 单个数值的情况下，直接计算阶乘的近似值或精确值
        return (
            np.power(k, (n - n_mod_k) / k)
            * gamma(n / k + 1) / gamma(n_mod_k / k + 1)
            * max(n_mod_k, 1)
        )

    # 计算不同 `r` 值对应的修正因子，并根据 `r` 的唯一值迭代计算结果
    result = np.power(k, n / k) * gamma(n / k + 1)
    def corr(k, r): return np.power(k, -r / k) / gamma(r / k + 1) * r
    for r in np.unique(n_mod_k):
        if r == 0:
            continue
        result[n_mod_k == r] *= corr(k, int(r))
    return result
    # 检查输入是否是标量（scalar）且不是 NumPy 数组（np.ndarray）的情况
    if np.ndim(n) == 0 and not isinstance(n, np.ndarray):
        # 处理标量情况
        if n is None or np.isnan(n):
            return np.nan
        elif not (np.issubdtype(type(n), np.integer)
                  or np.issubdtype(type(n), np.floating)):
            # 如果数据类型不是整数或浮点数，抛出数值错误
            raise ValueError(
                f"Unsupported datatype for factorial: {type(n)}\n"
                "Permitted data types are integers and floating point numbers"
            )
        elif n < 0:
            # 处理负数情况，阶乘定义域不包括负数，返回 0
            return 0
        elif exact and np.issubdtype(type(n), np.integer):
            # 如果要求精确计算且输入是整数，返回整数 n 的阶乘
            return math.factorial(n)
        elif exact:
            # 如果要求精确计算但输入是非整数，抛出数值错误
            msg = ("Non-integer values of `n` together with `exact=True` are "
                   "not supported. Either ensure integer `n` or use `exact=False`.")
            raise ValueError(msg)
        # 对于其它情况，调用近似计算函数处理
        return _factorialx_approx_core(n, k=1)

    # 处理数组和类数组的情况
    n = asarray(n)
    if n.size == 0:
        # 对于空数组，直接返回空数组
        return n
    if not (np.issubdtype(n.dtype, np.integer)
            or np.issubdtype(n.dtype, np.floating)):
        # 如果数据类型不是整数或浮点数，抛出数值错误
        raise ValueError(
            f"Unsupported datatype for factorial: {n.dtype}\n"
            "Permitted data types are integers and floating point numbers"
        )
    if exact and not np.issubdtype(n.dtype, np.integer):
        # 如果要求精确计算但输入是非整数数组，抛出数值错误
        msg = ("factorial with `exact=True` does not "
               "support non-integral arrays")
        raise ValueError(msg)

    if exact:
        # 如果要求精确计算，调用精确数组阶乘函数处理
        return _factorialx_array_exact(n, k=1)
    # 否则调用近似数组阶乘函数处理
    return _factorialx_array_approx(n, k=1)
def factorial2(n, exact=False):
    """Double factorial.

    This is the factorial with every second value skipped.  E.g., ``7!! = 7 * 5
    * 3 * 1``.  It can be approximated numerically as::

      n!! = 2 ** (n / 2) * gamma(n / 2 + 1) * sqrt(2 / pi)  n odd
          = 2 ** (n / 2) * gamma(n / 2 + 1)                 n even
          = 2 ** (n / 2) * (n / 2)!                         n even

    Parameters
    ----------
    n : int or array_like
        Calculate ``n!!``.  If ``n < 0``, the return value is 0.
    exact : bool, optional
        The result can be approximated rapidly using the gamma-formula
        above (default).  If `exact` is set to True, calculate the
        answer exactly using integer arithmetic.

    Returns
    -------
    nff : float or int
        Double factorial of `n`, as an int or a float depending on
        `exact`.

    Examples
    --------
    >>> from scipy.special import factorial2
    >>> factorial2(7, exact=False)
    array(105.00000000000001)
    >>> factorial2(7, exact=True)
    105

    """

    # don't use isscalar due to numpy/numpy#23574; 0-dim arrays treated below
    # 检查输入参数 n 是否是标量 (scalar)，因为 numpy 的问题 numpy/numpy#23574; 处理 0 维数组的情况
    if np.ndim(n) == 0 and not isinstance(n, np.ndarray):
        # scalar cases
        # 如果 n 是 None 或者是 NaN，返回 NaN
        if n is None or np.isnan(n):
            return np.nan
        # 如果 n 不是整数类型，抛出 ValueError
        elif not np.issubdtype(type(n), np.integer):
            msg = "factorial2 does not support non-integral scalar arguments"
            raise ValueError(msg)
        # 如果 n 小于 0，返回 0
        elif n < 0:
            return 0
        # 如果 n 是 {0, 1} 中的一个，返回 1
        elif n in {0, 1}:
            return 1
        # 一般整数情况下
        if exact:
            # 如果 exact 为 True，精确计算 n!! 使用 _range_prod 函数
            return _range_prod(1, n, k=2)
        # 否则使用 _factorialx_approx_core 函数进行近似计算
        return _factorialx_approx_core(n, k=2)
    
    # arrays & array-likes
    # 处理数组和类数组的情况，将 n 转换为 ndarray
    n = asarray(n)
    # 如果 n 的大小为 0，返回空数组
    if n.size == 0:
        return n
    # 如果 n 不是整数类型的数组，抛出 ValueError
    if not np.issubdtype(n.dtype, np.integer):
        raise ValueError("factorial2 does not support non-integral arrays")
    # 如果 exact 为 True，使用 _factorialx_array_exact 函数精确计算
    if exact:
        return _factorialx_array_exact(n, k=2)
    # 否则使用 _factorialx_array_approx 函数进行近似计算
    return _factorialx_array_approx(n, k=2)
    # 导入 scipy.special 中的 factorialk 函数，用于计算 k-generalized factorial
    from scipy.special import factorialk
    # 计算 n=5, k=1 时的 factorialk 值，exact=True 表示精确计算
    factorialk(5, k=1, exact=True)
    # 返回结果为 120

    # 计算 n=5, k=3 时的 factorialk 值，exact=True 表示精确计算
    factorialk(5, k=3, exact=True)
    # 返回结果为 10

    # 计算数组 [5, 7, 9] 中每个元素的 factorialk 值，k=3，exact=True 表示精确计算
    factorialk([5, 7, 9], k=3, exact=True)
    # 返回数组 [10, 28, 162]

    # 计算数组 [5, 7, 9] 中每个元素的 factorialk 值，k=3，exact=False 表示近似计算
    factorialk([5, 7, 9], k=3, exact=False)
    # 返回数组 [10., 28., 162.]

    # 在 exact=False 时，可以根据公式进行 n!(k) 的近似计算，参考下文提供的公式和说明
    """
    Notes
    -----
    While less straight-forward than for the double-factorial, it's possible to
    calculate a general approximation formula of n!(k) by studying ``n`` for a given
    remainder ``r < k`` (thus ``n = m * k + r``, resp. ``r = n % k``), which can be
    put together into something valid for all integer values ``n >= 0`` & ``k > 0``::

      n!(k) = k ** ((n - r)/k) * gamma(n/k + 1) / gamma(r/k + 1) * max(r, 1)

    This is the basis of the approximation when ``exact=False``. Compare also [1].

    References
    ----------
    .. [1] Complex extension to multifactorial
            https://en.wikipedia.org/wiki/Double_factorial#Alternative_extension_of_the_multifactorial
    """

    # 如果 k 不是正整数或小于 1，则抛出 ValueError 异常
    if not np.issubdtype(type(k), np.integer) or k < 1:
        raise ValueError(f"k must be a positive integer, received: {k}")

    # 如果 exact 参数为 None，则发出警告并设定 exact=True
    if exact is None:
        msg = (
            "factorialk will default to `exact=False` starting from SciPy "
            "1.15.0. To avoid behaviour changes due to this, explicitly "
            "specify either `exact=False` (faster, returns floats), or the "
            "past default `exact=True` (slower, lossless result as integer)."
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        exact = True

    # 如果 k 为 1 或 2，则给出帮助信息，建议尝试使用 factorial 或 factorial2 函数
    helpmsg = ""
    if k in {1, 2}:
        func = "factorial" if k == 1 else "factorial2"
        helpmsg = f"\nYou can try to use {func} instead"

    # 如果 n 是标量且不是整数或者是 NaN，则返回 NaN
    if np.ndim(n) == 0 and not isinstance(n, np.ndarray):
        if n is None or np.isnan(n):
            return np.nan
        elif not np.issubdtype(type(n), np.integer):
            msg = "factorialk does not support non-integral scalar arguments!"
            raise ValueError(msg + helpmsg)
        elif n < 0:
            return 0
        elif n in {0, 1}:
            return 1
        # 对于一般的整数情况，根据 exact 参数选择精确计算或近似计算
        if exact:
            return _range_prod(1, n, k=k)
        return _factorialx_approx_core(n, k=k)

    # 如果 n 是数组或类数组，则进行如下处理
    n = asarray(n)
    # 如果数组为空，则返回空数组
    if n.size == 0:
        return n
    # 如果数组元素不是整数类型，则抛出异常
    if not np.issubdtype(n.dtype, np.integer):
        msg = "factorialk does not support non-integral arrays!"
        raise ValueError(msg + helpmsg)
    # 根据 exact 参数选择精确计算或近似计算数组中每个元素的 factorialk 值
    if exact:
        return _factorialx_array_exact(n, k=k)
    return _factorialx_array_approx(n, k=k)
# 生成第二类斯特林数（Stirling numbers of the second kind），用于计算将具有N个元素的集合分成K个非空子集的方法数。
def stirling2(N, K, *, exact=False):
    # 根据动态规划算法计算斯特林数，避免在解决方案的子问题中进行冗余计算。
    # 对于类数组输入，此实现还避免了不同斯特林数计算之间的冗余计算。
    
    # 如果输入N和K都是标量，则设置标志以表示输出是标量
    output_is_scalar = np.isscalar(N) and np.isscalar(K)
    
    # 将输入的N和K转换为数组形式
    N, K = asarray(N), asarray(K)
    
    # 检查N数组中是否只包含整数，若不是则抛出TypeError异常
    if not np.issubdtype(N.dtype, np.integer):
        raise TypeError("Argument `N` must contain only integers")
    
    # 检查K数组中是否只包含整数，若不是则抛出TypeError异常
    if not np.issubdtype(K.dtype, np.integer):
        raise TypeError("Argument `K` must contain only integers")
    # 如果不要求精确计算，则进行以下操作：
    if not exact:
        # 注意：这里允许 np.uint 类型通过先转换为双精度类型，然后再传递给私有 ufunc 调度器。
        # 所有被调用的函数都接受双精度类型的 (n, k) 参数，并返回双精度类型的结果。
        return _stirling2_inexact(N.astype(float), K.astype(float))
    
    # 将 N 和 K 转换成 (n, k) 对列表，并去重
    nk_pairs = list(
        set([(n.take(0), k.take(0))
             for n, k in np.nditer([N, K], ['refs_ok'])])
    )
    
    # 将 nk_pairs 转换成最小堆结构
    heapify(nk_pairs)
    
    # 创建一个默认字典，初始值为整数 0
    snsk_vals = defaultdict(int)
    
    # 初始化 snsk_vals 的基本映射，针对小数值
    for pair in [(0, 0), (1, 1), (2, 1), (2, 2)]:
        snsk_vals[pair] = 1
    
    # 初始化 n_old 和 n_row 数组
    n_old, n_row = 2, [0, 1, 1]
    
    # 当最小堆 nk_pairs 不为空时循环
    while nk_pairs:
        # 从最小堆中取出 n 和 k
        n, k = heappop(nk_pairs)
        
        # 根据不同情况计算斯特林第二类数，存入 snsk_vals 中
        if n < 2 or k > n or k <= 0:
            continue
        elif k == n or k == 1:
            snsk_vals[(n, k)] = 1
            continue
        elif n != n_old:
            # 计算从 n_old 到 n 的斯特林数
            num_iters = n - n_old
            while num_iters > 0:
                n_row.append(1)
                # 从后往前遍历，移除第二行
                for j in range(len(n_row)-2, 1, -1):
                    n_row[j] = n_row[j]*j + n_row[j-1]
                num_iters -= 1
            snsk_vals[(n, k)] = n_row[k]
        else:
            snsk_vals[(n, k)] = n_row[k]
        
        # 更新 n_old 和 n_row
        n_old, n_row = n, n_row
    
    # 根据精确性要求确定输出类型
    out_types = [object, object, object] if exact else [float, float, float]
    
    # 使用 np.nditer 遍历 N, K 和输出数组，设置对应的操作类型和数据类型
    it = np.nditer(
        [N, K, None],
        ['buffered', 'refs_ok'],
        [['readonly'], ['readonly'], ['writeonly', 'allocate']],
        op_dtypes=out_types,
    )
    
    # 进入 np.nditer 迭代器上下文
    with it:
        while not it.finished:
            # 使用 snsk_vals 中对应的 (int(it[0]), int(it[1])) 的值填充输出数组的元素
            it[2] = snsk_vals[(int(it[0]), int(it[1]))]
            it.iternext()
        
        # 获取操作数中的输出数组
        output = it.operands[2]
        
        # 如果 N 和 K 都是标量，则将输出转换为标量
        if output_is_scalar:
            output = output.take(0)
    
    # 返回计算结果
    return output
# 定义 Riemann 或 Hurwitz zeta 函数，用于计算黎曼或赫维兹函数的值
def zeta(x, q=None, out=None):
    r"""
    Riemann or Hurwitz zeta function.

    Parameters
    ----------
    x : array_like of float
        输入数据，必须是实数
    q : array_like of float, optional
        输入数据，必须是实数。默认为 Riemann zeta 函数。
    out : ndarray, optional
        用于存储计算值的输出数组。

    Returns
    -------
    out : array_like
        zeta(x) 的值。

    See Also
    --------
    zetac

    Notes
    -----
    两参数版本是 Hurwitz zeta 函数

    .. math::

        \zeta(x, q) = \sum_{k=0}^{\infty} \frac{1}{(k + q)^x};

    详见 [dlmf]_ 。当 ``q = 1`` 时，对应黎曼 zeta 函数。

    References
    ----------
    .. [dlmf] NIST, Digital Library of Mathematical Functions,
        https://dlmf.nist.gov/25.11#i

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import zeta, polygamma, factorial

    特定值示例：

    >>> zeta(2), np.pi**2/6
    (1.6449340668482266, 1.6449340668482264)

    >>> zeta(4), np.pi**4/90
    (1.0823232337111381, 1.082323233711138)

    与 `polygamma` 函数的关系：

    >>> m = 3
    >>> x = 1.25
    >>> polygamma(m, x)
    array(2.782144009188397)
    >>> (-1)**(m+1) * factorial(m) * zeta(m+1, x)
    2.7821440091883969

    """
    # 如果 q 未指定，则调用内部函数计算黎曼 zeta 函数的值并返回
    if q is None:
        return _ufuncs._riemann_zeta(x, out)
    # 否则调用内部函数计算 Hurwitz zeta 函数的值并返回
    else:
        return _ufuncs._zeta(x, q, out)


# 私有函数：计算所有球谐函数的值
def _sph_harm_all(m, n, theta, phi):
    """Private function. This may be removed or modified at any time."""

    # 将输入的 theta 转换为 numpy 数组，确保数据类型为浮点数
    theta = np.asarray(theta)
    if (not np.issubdtype(theta.dtype, np.inexact)):
        theta = theta.astype(np.float64)

    # 将输入的 phi 转换为 numpy 数组，确保数据类型为浮点数
    phi = np.asarray(phi)
    if (not np.issubdtype(phi.dtype, np.inexact)):
        phi = phi.astype(np.float64)

    # 创建一个空数组，用于存储球谐函数的计算结果，维度根据输入参数而定
    out = np.empty((2 * m + 1, n + 1) + np.broadcast_shapes(theta.shape, phi.shape),
        dtype = np.result_type(1j, theta.dtype, phi.dtype))

    # 调用内部函数计算球谐函数，将结果移动轴以匹配输出的形状
    _sph_harm_all_gufunc(theta, phi, out = np.moveaxis(out, (0, 1), (-2, -1)))

    # 返回计算结果数组
    return out
```