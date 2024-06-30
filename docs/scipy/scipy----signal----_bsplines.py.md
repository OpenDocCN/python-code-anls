# `D:\src\scipysrc\scipy\scipy\signal\_bsplines.py`

```
# 从 numpy 模块中导入多个函数和常量，包括数组转换、常数 pi、类似零数组、数组构造、反正切函数、正切函数、全一数组、范围数组、向下取整函数、r_函数、至少一维数组、平方根函数、指数函数、大于比较函数、余弦函数、加法函数、正弦函数、轴移动函数、绝对值函数、反正切函数、复数 64、浮点数 32
from numpy import (asarray, pi, zeros_like,
                   array, arctan2, tan, ones, arange, floor,
                   r_, atleast_1d, sqrt, exp, greater, cos, add, sin,
                   moveaxis, abs, arctan, complex64, float32)

# 从 scipy._lib._util 模块中导入 normalize_axis_index 函数
from scipy._lib._util import normalize_axis_index

# 从 splinemodule.c 文件中导入 sepfir2d 函数
from ._spline import sepfir2d

# 从 _splines 模块中导入 symiirorder1 和 symiirorder2 函数
from ._splines import symiirorder1, symiirorder2

# 从 _signaltools 模块中导入 lfilter, sosfilt, lfiltic 函数
from ._signaltools import lfilter, sosfilt, lfiltic

# 从 scipy.interpolate 模块中导入 BSpline 类
from scipy.interpolate import BSpline

# 定义本模块中可导出的函数和类名称列表
__all__ = ['spline_filter', 'gauss_spline',
           'cspline1d', 'qspline1d', 'qspline2d', 'cspline2d',
           'cspline1d_eval', 'qspline1d_eval']

# 定义一个函数 spline_filter，用于对二维数组进行 (三次) 平滑样条滤波
def spline_filter(Iin, lmbda=5.0):
    """Smoothing spline (cubic) filtering of a rank-2 array.

    Filter an input data set, `Iin`, using a (cubic) smoothing spline of
    fall-off `lmbda`.

    Parameters
    ----------
    Iin : array_like
        input data set
    lmbda : float, optional
        spline smooghing fall-off value, default is `5.0`.

    Returns
    -------
    res : ndarray
        filtered input data

    Examples
    --------
    We can filter an multi dimensional signal (ex: 2D image) using cubic
    B-spline filter:

    >>> import numpy as np
    >>> from scipy.signal import spline_filter
    >>> import matplotlib.pyplot as plt
    >>> orig_img = np.eye(20)  # create an image
    >>> orig_img[10, :] = 1.0
    >>> sp_filter = spline_filter(orig_img, lmbda=0.1)
    >>> f, ax = plt.subplots(1, 2, sharex=True)
    >>> for ind, data in enumerate([[orig_img, "original image"],
    ...                             [sp_filter, "spline filter"]]):
    ...     ax[ind].imshow(data[0], cmap='gray_r')
    ...     ax[ind].set_title(data[1])
    >>> plt.tight_layout()
    >>> plt.show()

    """
    # 确定输入数据的类型字符码
    intype = Iin.dtype.char
    # 定义一个长度为 3 的浮点数组 hcol，用于平滑滤波
    hcol = array([1.0, 4.0, 1.0], 'f') / 6.0
    # 根据输入数据类型进行不同的处理分支
    if intype in ['F', 'D']:
        # 如果是复数类型，则将实部和虚部分别进行三次样条插值
        Iin = Iin.astype('F')
        ckr = cspline2d(Iin.real, lmbda)
        cki = cspline2d(Iin.imag, lmbda)
        outr = sepfir2d(ckr, hcol, hcol)
        outi = sepfir2d(cki, hcol, hcol)
        out = (outr + 1j * outi).astype(intype)
    elif intype in ['f', 'd']:
        # 如果是实数类型，则直接进行二维三次样条插值
        ckr = cspline2d(Iin, lmbda)
        out = sepfir2d(ckr, hcol, hcol)
        out = out.astype(intype)
    else:
        # 如果不是以上类型，则抛出类型错误异常
        raise TypeError("Invalid data type for Iin")
    # 返回滤波后的结果数组
    return out

# 定义一个空的缓存字典，用于存储样条函数
_splinefunc_cache = {}

# 定义一个函数 gauss_spline，用于生成高斯函数与 B 样条基函数的近似
def gauss_spline(x, n):
    r"""Gaussian approximation to B-spline basis function of order n.

    Parameters
    ----------
    x : array_like
        a knot vector
    n : int
        The order of the spline. Must be non-negative, i.e., n >= 0

    Returns
    -------
    res : ndarray
        B-spline basis function values approximated by a zero-mean Gaussian
        function.

    Notes
    -----
    The B-spline basis function can be approximated well by a zero-mean
    Gaussian function with standard-deviation equal to :math:`\sigma=(n+1)/12`
    for large `n` :

    .. math::  \frac{1}{\sqrt {2\pi\sigma^2}}exp(-\frac{x^2}{2\sigma})
    x = asarray(x)
    # 将输入的 x 转换为 numpy 数组，确保可以进行数学运算

    signsq = (n + 1) / 12.0
    # 计算高斯分布的方差 signsq

    return 1 / sqrt(2 * pi * signsq) * exp(-x ** 2 / 2 / signsq)
    # 返回高斯分布的密度函数在给定 x 处的值，使用了给定的方差 signsq
# 计算三次样条基函数的值，并进行截断处理
def _cubic(x):
    # 将输入的 x 转换为浮点型数组
    x = asarray(x, dtype=float)
    # 创建 B 样条基函数对象，定义结点为 [-2, -1, 0, 1, 2]，不进行外推
    b = BSpline.basis_element([-2, -1, 0, 1, 2], extrapolate=False)
    # 计算 B 样条基函数在输入 x 上的取值
    out = b(x)
    # 对超出指定范围 [-2, 2] 的 x 值对应的基函数值置为 0
    out[(x < -2) | (x > 2)] = 0
    # 返回计算结果
    return out


# 计算二次样条基函数的值，并进行截断处理
def _quadratic(x):
    # 将输入的 x 取绝对值后转换为浮点型数组
    x = abs(asarray(x, dtype=float))
    # 创建 B 样条基函数对象，定义结点为 [-1.5, -0.5, 0.5, 1.5]，不进行外推
    b = BSpline.basis_element([-1.5, -0.5, 0.5, 1.5], extrapolate=False)
    # 计算 B 样条基函数在输入 x 上的取值
    out = b(x)
    # 对超出指定范围 [-1.5, 1.5] 的 x 值对应的基函数值置为 0
    out[(x < -1.5) | (x > 1.5)] = 0
    # 返回计算结果
    return out


# 计算光滑系数 rho 和角频率 omega
def _coeff_smooth(lam):
    # 计算 xi
    xi = 1 - 96 * lam + 24 * lam * sqrt(3 + 144 * lam)
    # 计算 omega
    omega = arctan2(sqrt(144 * lam - 1), sqrt(xi))
    # 计算 rho
    rho = (24 * lam - 1 - sqrt(xi)) / (24 * lam)
    rho = rho * sqrt((48 * lam + 24 * lam * sqrt(3 + 144 * lam)) / xi)
    # 返回 rho 和 omega
    return rho, omega


# 计算正向滤波器的初始状态 zi，根据给定的信号和光滑系数
def _hc(k, cs, rho, omega):
    return (cs / sin(omega) * (rho ** k) * sin(omega * (k + 1)) *
            greater(k, -1))


# 计算反向滤波器的初始状态 zi，根据给定的信号和光滑系数
def _hs(k, cs, rho, omega):
    # 计算常数项 c0
    c0 = (cs * cs * (1 + rho * rho) / (1 - rho * rho) /
          (1 - 2 * rho * rho * cos(2 * omega) + rho ** 4))
    # 计算 gamma
    gamma = (1 - rho * rho) / (1 + rho * rho) / tan(omega)
    # 计算 ak 的绝对值
    ak = abs(k)
    # 返回反向滤波器的初始状态
    return c0 * rho ** ak * (cos(omega * ak) + gamma * sin(omega * ak))


# 计算三次样条平滑系数，并应用滤波器对信号进行平滑处理
def _cubic_smooth_coeff(signal, lamb):
    # 计算光滑系数 rho 和角频率 omega
    rho, omega = _coeff_smooth(lamb)
    # 计算 cs
    cs = 1 - 2 * rho * cos(omega) + rho * rho
    # 获取信号长度 K
    K = len(signal)
    # 创建索引序列 k
    k = arange(K)

    # 计算正向滤波器初始状态的前两个值
    zi_2 = (_hc(0, cs, rho, omega) * signal[0] +
            add.reduce(_hc(k + 1, cs, rho, omega) * signal))
    zi_1 = (_hc(0, cs, rho, omega) * signal[0] +
            _hc(1, cs, rho, omega) * signal[1] +
            add.reduce(_hc(k + 2, cs, rho, omega) * signal))

    # 正向滤波：
    # for n in range(2, K):
    #     yp[n] = (cs * signal[n] + 2 * rho * cos(omega) * yp[n - 1] -
    #              rho * rho * yp[n - 2])
    # 利用 lfiltic 函数计算正向滤波器的初始状态 zi
    zi = lfiltic(cs, r_[1, -2 * rho * cos(omega), rho * rho], r_[zi_1, zi_2])
    zi = zi.reshape(1, -1)

    # 定义二阶段二阶节的系数向量 sos
    sos = r_[cs, 0, 0, 1, -2 * rho * cos(omega), rho * rho]
    sos = sos.reshape(1, -1)

    # 应用 sosfilt 函数进行滤波，获取平滑后的信号 yp
    yp, _ = sosfilt(sos, signal[2:], zi=zi)
    yp = r_[zi_2, zi_1, yp]

    # 反向滤波：
    # for n in range(K - 3, -1, -1):
    #     y[n] = (cs * yp[n] + 2 * rho * cos(omega) * y[n + 1] -
    #             rho * rho * y[n + 2])
    # 计算反向滤波器初始状态的前两个值
    zi_2 = add.reduce((_hs(k, cs, rho, omega) +
                       _hs(k + 1, cs, rho, omega)) * signal[::-1])
    zi_1 = add.reduce((_hs(k - 1, cs, rho, omega) +
                       _hs(k + 2, cs, rho, omega)) * signal[::-1])

    # 利用 lfiltic 函数计算反向滤波器的初始状态 zi
    zi = lfiltic(cs, r_[1, -2 * rho * cos(omega), rho * rho], r_[zi_1, zi_2])
    zi = zi.reshape(1, -1)

    # 应用 sosfilt 函数进行滤波，获取最终的平滑信号 y
    y, _ = sosfilt(sos, yp[-3::-1], zi=zi)
    y = r_[y[::-1], zi_1, zi_2]
    # 返回平滑后的信号 y
    return y


# 计算三次样条系数，并应用滤波器对信号进行插值处理
def _cubic_coeff(signal):
    # 计算 zi
    zi = -2 + sqrt(3)
    # 获取信号长度 K
    K = len(signal)
    # 计算幂次方向量 powers
    powers = zi ** arange(K)

    if K == 1:
        # 计算单个数据点的插值结果
        yplus = signal[0] + zi * add.reduce(powers * signal)
        output = zi / (zi - 1) * yplus
        return atleast_1d(output)

    # 正向滤波：
    # yplus[0] = signal[0] + zi * add.reduce(powers * signal)
    # for k in range(1, K):
    #     yplus[k] = signal[k] + zi * yplus[k - 1]
    # 使用 lfiltic 函数计算 IIR 滤波器的初始状态
    state = lfiltic(1, r_[1, -zi], atleast_1d(add.reduce(powers * signal)))

    # 创建分子系数 b 和分母系数 a 用于 IIR 滤波器
    b = ones(1)
    a = r_[1, -zi]

    # 应用 IIR 滤波器，计算 yplus，同时使用之前计算的状态 state
    yplus, _ = lfilter(b, a, signal, zi=state)

    # 反向滤波器:
    # 输出的最后一个元素的计算方式
    out_last = zi / (zi - 1) * yplus[K - 1]

    # 使用 lfiltic 函数计算反向滤波器的初始状态
    state = lfiltic(-zi, r_[1, -zi], atleast_1d(out_last))

    # 使用反向滤波器计算输出，并从倒序的 yplus[-2::-1] 开始，使用之前计算的状态 state
    b = asarray([-zi])
    output, _ = lfilter(b, a, yplus[-2::-1], zi=state)

    # 将反向滤波器的输出与 out_last 合并为最终输出
    output = r_[output[::-1], out_last]

    # 返回最终输出，并乘以 6.0
    return output * 6.0
def _quadratic_coeff(signal):
    # 计算常数 zi，使用信号长度 K
    zi = -3 + 2 * sqrt(2.0)
    K = len(signal)
    # 计算 zi 的幂，从 0 到 K-1
    powers = zi ** arange(K)

    # 如果信号长度为 1，则直接计算输出并返回
    if K == 1:
        yplus = signal[0] + zi * add.reduce(powers * signal)
        output = zi / (zi - 1) * yplus
        return atleast_1d(output)

    # 使用 lfiltic 函数生成初始状态
    state = lfiltic(1, r_[1, -zi], atleast_1d(add.reduce(powers * signal)))

    # 设置前向滤波器的系数
    b = ones(1)
    a = r_[1, -zi]
    # 使用 lfilter 函数进行前向滤波
    yplus, _ = lfilter(b, a, signal, zi=state)

    # 计算最后一个输出值
    out_last = zi / (zi - 1) * yplus[K - 1]
    # 使用 lfiltic 函数生成反向滤波器的初始状态
    state = lfiltic(-zi, r_[1, -zi], atleast_1d(out_last))

    # 设置反向滤波器的系数
    b = asarray([-zi])
    # 使用 lfilter 函数进行反向滤波
    output, _ = lfilter(b, a, yplus[-2::-1], zi=state)
    # 组合输出结果
    output = r_[output[::-1], out_last]
    return output * 8.0



def compute_root_from_lambda(lamb):
    # 计算中间临时变量 tmp
    tmp = sqrt(3 + 144 * lamb)
    # 计算 xi
    xi = 1 - 96 * lamb + 24 * lamb * tmp
    # 计算 omega
    omega = arctan(sqrt((144 * lamb - 1.0) / xi))
    # 计算另一个临时变量 tmp2
    tmp2 = sqrt(xi)
    # 计算 r
    r = ((24 * lamb - 1 - tmp2) / (24 * lamb) *
         sqrt(48*lamb + 24 * lamb * tmp) / tmp2)
    return r, omega



def cspline1d(signal, lamb=0.0):
    """
    Compute cubic spline coefficients for rank-1 array.

    Find the cubic spline coefficients for a 1-D signal assuming
    mirror-symmetric boundary conditions. To obtain the signal back from the
    spline representation mirror-symmetric-convolve these coefficients with a
    length 3 FIR window [1.0, 4.0, 1.0]/ 6.0 .

    Parameters
    ----------
    signal : ndarray
        A rank-1 array representing samples of a signal.
    lamb : float, optional
        Smoothing coefficient, default is 0.0.

    Returns
    -------
    c : ndarray
        Cubic spline coefficients.

    See Also
    --------
    cspline1d_eval : Evaluate a cubic spline at the new set of points.

    Examples
    --------
    We can filter a signal to reduce and smooth out high-frequency noise with
    a cubic spline:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.signal import cspline1d, cspline1d_eval
    >>> rng = np.random.default_rng()
    >>> sig = np.repeat([0., 1., 0.], 100)
    >>> sig += rng.standard_normal(len(sig))*0.05  # add noise
    >>> time = np.linspace(0, len(sig))
    >>> filtered = cspline1d_eval(cspline1d(sig), time)
    >>> plt.plot(sig, label="signal")
    >>> plt.plot(time, filtered, label="filtered")
    >>> plt.legend()
    >>> plt.show()

    """
    # 如果 lamb 不为 0，则调用 _cubic_smooth_coeff 函数
    if lamb != 0.0:
        return _cubic_smooth_coeff(signal, lamb)
    else:
        # 否则调用 _cubic_coeff 函数
        return _cubic_coeff(signal)



def qspline1d(signal, lamb=0.0):
    """Compute quadratic spline coefficients for rank-1 array.

    Parameters
    ----------
    signal : ndarray
        A rank-1 array representing samples of a signal.
    lamb : float, optional
        Smoothing coefficient, default is 0.0.
    """
    # 此函数尚未实现，可以根据需要编写相应的代码
    pass
    # lamb : float, optional
    # 平滑系数（目前必须为零）。

    # Returns
    # -------
    # c : ndarray
    # 二次样条插值的系数。

    # See Also
    # --------
    # qspline1d_eval : 在新的点集上评估二次样条插值。

    # Notes
    # -----
    # 找到一维信号的二次样条插值系数，假设使用镜像对称边界条件。
    # 要从插值表示中恢复信号，请使用长度为3的FIR窗口[1.0, 6.0, 1.0]/8.0与这些系数进行镜像对称卷积。

    # Examples
    # --------
    # 我们可以使用二次样条插值来过滤信号，减少和平滑高频噪声：

    # >>> import numpy as np
    # >>> import matplotlib.pyplot as plt
    # >>> from scipy.signal import qspline1d, qspline1d_eval
    # >>> rng = np.random.default_rng()
    # >>> sig = np.repeat([0., 1., 0.], 100)
    # >>> sig += rng.standard_normal(len(sig))*0.05  # 添加噪声
    # >>> time = np.linspace(0, len(sig))
    # >>> filtered = qspline1d_eval(qspline1d(sig), time)
    # >>> plt.plot(sig, label="signal")
    # >>> plt.plot(time, filtered, label="filtered")
    # >>> plt.legend()
    # >>> plt.show()

    """
    if lamb != 0.0:
        # 如果平滑系数不为零，则抛出值错误异常
        raise ValueError("Smoothing quadratic splines not supported yet.")
    else:
        # 否则，返回信号的二次插值系数
        return _quadratic_coeff(signal)
def collapse_2d(x, axis):
    # 将指定轴移动到最后一个位置，以便在最后一个维度上重塑数组
    x = moveaxis(x, axis, -1)
    # 记录重塑后的数组形状
    x_shape = x.shape
    # 将数组展平为二维数组，最后一个维度保持不变
    x = x.reshape(-1, x.shape[-1])
    # 如果数组不是C连续的，则进行复制以确保C连续性
    if not x.flags.c_contiguous:
        x = x.copy()
    # 返回重塑后的数组和原始形状信息
    return x, x_shape


def symiirorder_nd(func, input, *args, axis=-1, **kwargs):
    # 标准化轴索引，确保在输入数组的维度范围内
    axis = normalize_axis_index(axis, input.ndim)
    # 记录输入数组的原始形状
    input_shape = input.shape
    # 记录输入数组的维度数
    input_ndim = input.ndim
    # 如果输入数组的维度大于1，则将其折叠为二维数组
    if input.ndim > 1:
        input, input_shape = collapse_2d(input, axis)

    # 对输入数组应用指定的函数，传递额外的参数和关键字参数
    out = func(input, *args, **kwargs)

    # 如果输入数组的维度大于1，则对输出进行相应的重塑和轴移操作
    if input_ndim > 1:
        out = out.reshape(input_shape)
        out = moveaxis(out, -1, axis)
        # 如果输出数组不是C连续的，则进行复制以确保C连续性
        if not out.flags.c_contiguous:
            out = out.copy()
    # 返回处理后的输出数组
    return out


def qspline2d(signal, lamb=0.0, precision=-1.0):
    """
    Coefficients for 2-D quadratic (2nd order) B-spline.

    Return the second-order B-spline coefficients over a regularly spaced
    input grid for the two-dimensional input image.

    Parameters
    ----------
    input : ndarray
        The input signal.
    lamb : float
        Specifies the amount of smoothing in the transfer function.
    precision : float
        Specifies the precision for computing the infinite sum needed to apply
        mirror-symmetric boundary conditions.

    Returns
    -------
    output : ndarray
        The filtered signal.
    """
    # 如果precision小于0或者大于等于1，根据信号的数据类型设置默认precision
    if precision < 0.0 or precision >= 1.0:
        if signal.dtype in [float32, complex64]:
            precision = 1e-3
        else:
            precision = 1e-6

    # 如果lamb大于0，引发值错误，lambda必须为负数或零
    if lamb > 0:
        raise ValueError('lambda must be negative or zero')

    # 标准二次B样条插值
    r = -3 + 2 * sqrt(2.0)
    c0 = -r * 8.0
    z1 = r

    # 对信号应用符号IIR滤波器，沿着指定的轴(-1)应用
    out = symiirorder_nd(symiirorder1, signal, c0, z1, precision, axis=-1)
    # 对输出再次应用符号IIR滤波器，沿着0轴应用
    out = symiirorder_nd(symiirorder1, out, c0, z1, precision, axis=0)
    # 返回处理后的输出
    return out


def cspline2d(signal, lamb=0.0, precision=-1.0):
    """
    Coefficients for 2-D cubic (3rd order) B-spline.

    Return the third-order B-spline coefficients over a regularly spaced
    input grid for the two-dimensional input image.

    Parameters
    ----------
    input : ndarray
        The input signal.
    lamb : float
        Specifies the amount of smoothing in the transfer function.
    precision : float
        Specifies the precision for computing the infinite sum needed to apply
        mirror-symmetric boundary conditions.

    Returns
    -------
    output : ndarray
        The filtered signal.
    """
    # 如果precision小于0或者大于等于1，根据信号的数据类型设置默认precision
    if precision < 0.0 or precision >= 1.0:
        if signal.dtype in [float32, complex64]:
            precision = 1e-3
        else:
            precision = 1e-6

    # 如果lamb小于等于1/144.0，执行正常的三次B样条插值
    if lamb <= 1 / 144.0:
        r = -2 + sqrt(3.0)
        out = symiirorder_nd(
            symiirorder1, signal, -r * 6.0, r, precision=precision, axis=-1)
        out = symiirorder_nd(
            symiirorder1, out, -r * 6.0, r, precision=precision, axis=0)
        return out

    # 如果lamb大于1/144.0，通过计算lambda的根来执行其它操作
    r, omega = compute_root_from_lambda(lamb)
    # 对输入信号使用二维有限脉冲响应（FIR）滤波器进行滤波，返回滤波后的结果
    out = symiirorder_nd(symiirorder2, signal, r, omega,
                         precision=precision, axis=-1)
    # 对上一步滤波后的结果再次使用二维FIR滤波器进行滤波，返回最终的滤波结果
    out = symiirorder_nd(symiirorder2, out, r, omega,
                         precision=precision, axis=0)
    # 返回最终的滤波结果
    return out
# 根据新的点集评估一维立方样条插值函数。

# `dx` 是旧的采样间距，`x0` 是旧的起点。换句话说，`cj` 表示立方样条系数的旧样本点（节点）是均匀分布在以下点上的：
# oldx = x0 + j*dx，其中 j=0...N-1，这里 N=len(cj)

# 使用镜像对称边界条件处理边缘情况。

def cspline1d_eval(cj, newx, dx=1.0, x0=0):
    newx = (asarray(newx) - x0) / float(dx)  # 将新的点集转换为以旧的样本间距和起点为基础的坐标系
    res = zeros_like(newx, dtype=cj.dtype)  # 创建一个与 newx 形状相同的结果数组
    if res.size == 0:
        return res  # 如果结果数组为空，则直接返回空数组

    N = len(cj)  # 获取系数数组 `cj` 的长度
    cond1 = newx < 0  # 新的点集中小于 0 的条件
    cond2 = newx > (N - 1)  # 新的点集中大于 N-1 的条件
    cond3 = ~(cond1 | cond2)  # 非边缘条件的布尔掩码

    # 处理一般的镜像对称性
    res[cond1] = cspline1d_eval(cj, -newx[cond1])  # 对于小于 0 的点，使用镜像对称性求解
    res[cond2] = cspline1d_eval(cj, 2 * (N - 1) - newx[cond2])  # 对于大于 N-1 的点，使用镜像对称性求解

    newx = newx[cond3]  # 保留非边缘条件下的新的点集

    if newx.size == 0:
        return res  # 如果新的点集为空，则直接返回结果数组

    result = zeros_like(newx, dtype=cj.dtype)  # 创建一个与 newx 形状相同的结果数组
    jlower = floor(newx - 2).astype(int) + 1  # 计算每个新点对应的最低节点索引

    for i in range(4):
        thisj = jlower + i  # 计算当前节点的索引
        indj = thisj.clip(0, N - 1)  # 处理边界情况，将索引限制在有效范围内
        result += cj[indj] * _cubic(newx - thisj)  # 根据立方插值函数计算每个新点的插值结果

    res[cond3] = result  # 将计算得到的插值结果赋给非边缘条件的结果数组部分
    return res  # 返回最终的插值结果数组


# 根据新的点集评估一维二次样条插值函数。
def qspline1d_eval(cj, newx, dx=1.0, x0=0):
    # `dx` 是旧的采样间距，`x0` 是旧的起点。换句话说，`cj` 表示二次样条系数的旧样本点（节点）是均匀分布在以下点上的：

    # 参数和返回值同上
    newx = (asarray(newx) - x0) / float(dx)  # 将新的点集转换为以旧的样本间距和起点为基础的坐标系
    res = zeros_like(newx, dtype=cj.dtype)  # 创建一个与 newx 形状相同的结果数组
    if res.size == 0:
        return res  # 如果结果数组为空，则直接返回空数组

    N = len(cj)  # 获取系数数组 `cj` 的长度

    # 函数未完全提供，此处省略处理边缘情况的代码
    """
    newx = (asarray(newx) - x0) / dx
    将新的 x 值转换为相对于原始起始点 x0 的偏移量，并且按照步长 dx 进行归一化处理
    res = zeros_like(newx)
    初始化一个与 newx 相同形状的零数组作为结果容器
    if res.size == 0:
    如果结果容器为空，则返回空数组
        return res
    N = len(cj)
    获取样条系数 cj 的长度，用于后续条件判断
    cond1 = newx < 0
    创建条件1：新的 x 值小于 0 的布尔数组
    cond2 = newx > (N - 1)
    创建条件2：新的 x 值大于 (N - 1) 的布尔数组
    cond3 = ~(cond1 | cond2)
    创建条件3：既不满足条件1也不满足条件2的布尔数组
    # handle general mirror-symmetry
    处理一般的镜像对称边界条件
    res[cond1] = qspline1d_eval(cj, -newx[cond1])
    对满足条件1的部分使用二次样条插值求值函数 qspline1d_eval 进行处理
    res[cond2] = qspline1d_eval(cj, 2 * (N - 1) - newx[cond2])
    对满足条件2的部分使用二次样条插值求值函数 qspline1d_eval 进行处理
    newx = newx[cond3]
    更新 newx，仅保留满足条件3的部分
    if newx.size == 0:
    如果经过条件筛选后 newx 为空，则返回结果容器
        return res
    result = zeros_like(newx)
    初始化一个与新的 newx 相同形状的零数组作为结果容器
    jlower = floor(newx - 1.5).astype(int) + 1
    计算新的 newx 值向下取整后减去1.5并转换为整数，然后加1，得到 jlower 数组
    for i in range(3):
    循环3次：
        thisj = jlower + i
        计算当前的 thisj 值，即 jlower 加上当前循环变量 i
        indj = thisj.clip(0, N - 1)  # handle edge cases
        对 thisj 进行边界处理，确保不超出索引范围
        result += cj[indj] * _quadratic(newx - thisj)
        根据 thisj 索引取出对应的样条系数 cj，并使用二次插值函数 _quadratic 对新的 x 值进行处理并累加到结果中
    res[cond3] = result
    将最终处理好的结果赋值给满足条件3的部分
    return res
    返回最终的结果数组 res
    """
```