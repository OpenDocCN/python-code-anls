# `D:\src\scipysrc\scipy\scipy\signal\_spectral.py`

```
# Author: Pim Schellart
# 2010 - 2011

"""Tools for spectral analysis of unequally sampled signals."""

import numpy as np

# 定义了一个函数，用于计算Lomb-Scargle周期图
# 并通过pythran进行导出以提高性能
# 输入为x（样本时间）、y（测量值，要求均值为零）、freqs（输出周期图的角频率）
def _lombscargle(x, y, freqs):
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

    # 检查输入数组的大小是否相同
    if x.shape != y.shape:
        raise ValueError("Input arrays do not have the same size.")

    # 创建一个和freqs大小相同的空数组，用于存储周期图
    pgram = np.empty_like(freqs)

    # 创建空数组c和s，用于存储cos和sin的计算结果
    c = np.empty_like(x)
    s = np.empty_like(x)

    # 遍历freqs数组中的每一个频率
    for i in range(freqs.shape[0]):

        # 初始化一些变量
        xc = 0.
        xs = 0.
        cc = 0.
        ss = 0.
        cs = 0.

        # 计算cos和sin值，并存储到数组c和s中
        c[:] = np.cos(freqs[i] * x)
        s[:] = np.sin(freqs[i] * x)

        # 遍历输入数组x中的每一个时间点
        for j in range(x.shape[0]):
            # 根据Lomb-Scargle算法更新计算结果
            xc += y[j] * c[j]
            xs += y[j] * s[j]
            cc += c[j] * c[j]
            ss += s[j] * s[j]
            cs += c[j] * s[j]

        # 检查频率是否为零，若为零则抛出ZeroDivisionError异常
        if freqs[i] == 0:
            raise ZeroDivisionError()

        # 计算tau值
        tau = np.arctan2(2 * cs, cc - ss) / (2 * freqs[i])
        c_tau = np.cos(freqs[i] * tau)
        s_tau = np.sin(freqs[i] * tau)
        c_tau2 = c_tau * c_tau
        s_tau2 = s_tau * s_tau
        cs_tau = 2 * c_tau * s_tau

        # 计算并存储Lomb-Scargle周期图的值
        pgram[i] = 0.5 * (((c_tau * xc + s_tau * xs)**2 /
            (c_tau2 * cc + cs_tau * cs + s_tau2 * ss)) +
            ((c_tau * xs - s_tau * xc)**2 /
             (c_tau2 * ss - cs_tau * cs + s_tau2 * cc)))

    # 返回计算得到的Lomb-Scargle周期图
    return pgram
```