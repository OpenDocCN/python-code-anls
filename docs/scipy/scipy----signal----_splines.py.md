# `D:\src\scipysrc\scipy\scipy\signal\_splines.py`

```
# 导入NumPy库，用于处理数组和数学运算
import numpy as np

# 从内部模块中导入所需函数和类
from ._arraytools import axis_slice, axis_reverse
from ._signaltools import lfilter, sosfilt
from ._spline import symiirorder1_ic, symiirorder2_ic_fwd, symiirorder2_ic_bwd

# 指定模块中公开的函数和类
__all__ = ['symiirorder1', 'symiirorder2']

# 定义一个函数，实现一阶级联平滑IIR滤波器，使用镜像对称边界条件
def symiirorder1(signal, c0, z1, precision=-1.0):
    """
    Implement a smoothing IIR filter with mirror-symmetric boundary conditions
    using a cascade of first-order sections.  The second section uses a
    reversed sequence.  This implements a system with the following
    transfer function and mirror-symmetric boundary conditions::

                           c0
           H(z) = ---------------------
                   (1-z1/z) (1 - z1 z)

    The resulting signal will have mirror symmetric boundary conditions
    as well.

    Parameters
    ----------
    input : ndarray
        The input signal. If 2D, then the filter will be applied in a batched
        fashion across the last axis.
    c0, z1 : scalar
        Parameters in the transfer function.
    precision :
        Specifies the precision for calculating initial conditions
        of the recursive filter based on mirror-symmetric input.

    Returns
    -------
    output : ndarray
        The filtered signal.
    """
    # 检查z1的绝对值是否小于1，如果不是则引发ValueError异常
    if np.abs(z1) >= 1:
        raise ValueError('|z1| must be less than 1.0')

    # 检查输入信号的维度是否大于2，如果是则引发ValueError异常
    if signal.ndim > 2:
        raise ValueError('Input must be 1D or 2D')

    # 如果输入信号是一维的，则扩展维度为二维，以便在最后一个轴上进行批处理滤波
    squeeze_dim = False
    if signal.ndim == 1:
        signal = signal[None, :]
        squeeze_dim = True

    # 如果输入信号的数据类型为整数类型，则转换为浮点数类型
    if np.issubdtype(signal.dtype, np.integer):
        signal = signal.astype(np.float64)

    # 计算镜像对称条件下的初始状态y0
    y0 = symiirorder1_ic(signal, z1, precision)

    # 应用第一个系统 1 / (1 - z1 * z^-1)
    b = np.ones(1, dtype=signal.dtype)
    a = np.r_[1, -z1]
    a = a.astype(signal.dtype)

    # 计算lfilter的初始状态
    zii = y0 * z1

    # 对信号的第一个轴进行滤波，得到y1
    y1, _ = lfilter(b, a, axis_slice(signal, 1), zi=zii)
    y1 = np.c_[y0, y1]

    # 计算反向对称条件并应用系统 c0 / (1 - z1 * z)
    b = np.asarray([c0], dtype=signal.dtype)
    out_last = -c0 / (z1 - 1.0) * axis_slice(y1, -1)

    # 计算lfilter的初始状态
    zii = out_last * z1

    # 对信号的第一个轴进行反向滤波，得到out
    out, _ = lfilter(b, a, axis_slice(y1, -2, step=-1), zi=zii)
    out = np.c_[axis_reverse(out), out_last]

    # 如果之前进行了维度压缩，则恢复输出维度
    if squeeze_dim:
        out = out[0]

    return out


def symiirorder2(input, r, omega, precision=-1.0):
    """
    Implement a smoothing IIR filter with mirror-symmetric boundary conditions
    using a cascade of second-order sections.  The second section uses a
    reversed sequence.  This implements the following transfer function::

                                  cs^2
         H(z) = ---------------------------------------
                (1 - a2/z - a3/z^2) (1 - a2 z - a3 z^2 )

    where::
          a2 = 2 * r * cos(omega)
          a3 = - r ** 2
          cs = 1 - 2 * r * cos(omega) + r ** 2

    Parameters
    ----------
    input : ndarray
        The input signal.
    """
    # 函数实现了一个平滑的IIR滤波器，使用镜像对称边界条件，由二阶段级联实现
    # 第二阶段使用了反向序列
    r, omega : float
        转移函数中的参数。
    precision : float
        基于镜像对称输入计算递归滤波器初始条件的精度。

    Returns
    -------
    output : ndarray
        过滤后的信号。
    """
    # 如果 r 大于等于 1.0，则抛出值错误异常
    if r >= 1.0:
        raise ValueError('r must be less than 1.0')

    # 如果输入的维度大于 2，则抛出值错误异常
    if input.ndim > 2:
        raise ValueError('Input must be 1D or 2D')

    # 如果输入数组不是 C 连续的，则进行拷贝，以确保是 C 连续的
    if not input.flags.c_contiguous:
        input = input.copy()

    # 如果输入数组是 1 维的，则将其转换为二维数组，并标记需要压缩维度
    squeeze_dim = False
    if input.ndim == 1:
        input = input[None, :]
        squeeze_dim = True

    # 如果输入数组的数据类型是整数类型，则将其转换为 np.float64 类型
    if np.issubdtype(input.dtype, np.integer):
        input = input.astype(np.float64)

    # 计算 r 的平方和相关系数
    rsq = r * r
    a2 = 2 * r * np.cos(omega)
    a3 = -rsq
    cs = np.atleast_1d(1 - 2 * r * np.cos(omega) + rsq)
    sos = np.atleast_2d(np.r_[cs, 0, 0, 1, -a2, -a3]).astype(input.dtype)

    # 查找起始（前向）条件
    ic_fwd = symiirorder2_ic_fwd(input, r, omega, precision)

    # 首先应用系统 cs / (1 - a2 * z^-1 - a3 * z^-2)
    # 计算 sosfilt 预期形式的初始条件
    coef = np.r_[a3, a2, 0, a3].reshape(2, 2).astype(input.dtype)
    zi = np.matmul(coef, ic_fwd[:, :, None])[:, :, 0]

    y_fwd, _ = sosfilt(sos, axis_slice(input, 2), zi=zi[None])
    y_fwd = np.c_[ic_fwd, y_fwd]

    # 然后计算对称的后向起始条件
    ic_bwd = symiirorder2_ic_bwd(input, r, omega, precision)

    # 再次应用系统 cs / (1 - a2 * z^1 - a3 * z^2)
    # 计算 sosfilt 预期形式的初始条件
    zi = np.matmul(coef, ic_bwd[:, :, None])[:, :, 0]
    y, _ = sosfilt(sos, axis_slice(y_fwd, -3, step=-1), zi=zi[None])
    out = np.c_[axis_reverse(y), axis_reverse(ic_bwd)]

    # 如果之前进行了维度压缩，则恢复到原来的维度
    if squeeze_dim:
        out = out[0]

    # 返回处理后的输出
    return out
```