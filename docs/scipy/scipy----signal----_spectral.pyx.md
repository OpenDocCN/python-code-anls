# `D:\src\scipysrc\scipy\scipy\signal\_spectral.pyx`

```
# Author: Pim Schellart
# 2010 - 2011

# cython: cpow=True

"""Tools for spectral analysis of unequally sampled signals."""

# 导入 NumPy 库并声明 Cython 特定的 NumPy 扩展
import numpy as np
cimport numpy as np
cimport cython

# 导入 NumPy C API
np.import_array()

# 定义导出的函数名列表
__all__ = ['_lombscargle']

# 从 C 标准库 "math.h" 导入数学函数原型
cdef extern from "math.h":
    double cos(double)
    double sin(double)
    double atan2(double, double)

# 关闭 Cython 的边界检查优化
@cython.boundscheck(False)
# 定义名为 _lombscargle 的 Python 函数
def _lombscargle(np.ndarray[np.float64_t, ndim=1] x,
                np.ndarray[np.float64_t, ndim=1] y,
                np.ndarray[np.float64_t, ndim=1] freqs):
    """
    _lombscargle(x, y, freqs)

    Computes the Lomb-Scargle periodogram.

    Parameters
    ----------
    x : array_like
        Sample times.
    y : array_like
        Measurement values (must be registered so the mean is zero).
    freqs : array_like
        Angular frequencies for output periodogram.

    Returns
    -------
    pgram : array_like
        Lomb-Scargle periodogram.

    Raises
    ------
    ValueError
        If the input arrays `x` and `y` do not have the same shape.

    See also
    --------
    lombscargle

    """

    # 检查输入数组 x 和 y 的形状是否相同
    if x.shape[0] != y.shape[0]:
        raise ValueError("Input arrays do not have the same size.")

    # 创建一个空数组用于存储输出的 periodogram
    pgram = np.empty(freqs.shape[0], dtype=np.float64)

    # 定义局部变量
    cdef Py_ssize_t i, j
    cdef double c, s, xc, xs, cc, ss, cs
    cdef double tau, c_tau, s_tau, c_tau2, s_tau2, cs_tau

    # 遍历频率数组
    for i in range(freqs.shape[0]):

        # 初始化累加器
        xc = 0.
        xs = 0.
        cc = 0.
        ss = 0.
        cs = 0.

        # 遍历时间样本数组
        for j in range(x.shape[0]):

            # 计算余弦和正弦值
            c = cos(freqs[i] * x[j])
            s = sin(freqs[i] * x[j])

            # 更新累加器
            xc += y[j] * c
            xs += y[j] * s
            cc += c * c
            ss += s * s
            cs += c * s

        # 计算 tau 的值
        tau = atan2(2 * cs, cc - ss) / (2 * freqs[i])
        c_tau = cos(freqs[i] * tau)
        s_tau = sin(freqs[i] * tau)
        c_tau2 = c_tau * c_tau
        s_tau2 = s_tau * s_tau
        cs_tau = 2 * c_tau * s_tau

        # 计算 Lomb-Scargle periodogram 的值并存储
        pgram[i] = 0.5 * (((c_tau * xc + s_tau * xs)**2 / \
            (c_tau2 * cc + cs_tau * cs + s_tau2 * ss)) + \
            ((c_tau * xs - s_tau * xc)**2 / \
            (c_tau2 * ss - cs_tau * cs + s_tau2 * cc)))

    # 返回 Lomb-Scargle periodogram 数组
    return pgram
```