# `D:\src\scipysrc\scipy\scipy\linalg\_solve_toeplitz.pyx`

```
# Author: Robert T. McGibbon, December 2014
#
# cython: boundscheck=False, wraparound=False, cdivision=True

# 导入必要的模块和函数
from numpy import zeros, asarray, complex128, float64
from numpy.linalg import LinAlgError
from numpy cimport complex128_t, float64_t

# 定义复合类型 dz，包含 float64_t 和 complex128_t
cdef fused dz:
    float64_t
    complex128_t

# 定义函数 levinson，用于求解线性 Toeplitz 系统的问题
def levinson(const dz[::1] a, const dz[::1] b):
    """Solve a linear Toeplitz system using Levinson recursion.

    Parameters
    ----------
    a : array, dtype=double or complex128, shape=(2n-1,)
        The first column of the matrix in reverse order (without the diagonal)
        followed by the first (see below)
    b : array, dtype=double  or complex128, shape=(n,)
        The right hand side vector. Both a and b must have the same type
        (double or complex128).

    Notes
    -----
    For example, the 5x5 toeplitz matrix below should be represented as
    the linear array ``a`` on the right ::

        [ a0    a1   a2  a3  a4 ]
        [ a-1   a0   a1  a2  a3 ]
        [ a-2  a-1   a0  a1  a2 ] -> [a-4  a-3  a-2  a-1  a0  a1  a2  a3  a4]
        [ a-3  a-2  a-1  a0  a1 ]
        [ a-4  a-3  a-2  a-1 a0 ]

    Returns
    -------
    x : array, shape=(n,)
        The solution vector
    reflection_coeff : array, shape=(n+1,)
        Toeplitz reflection coefficients. When a is symmetric Toeplitz and
        ``b`` is ``a[n:]``, as in the solution of autoregressive systems,
        then ``reflection_coeff`` also correspond to the partial
        autocorrelation function.
    """
    # 从 toeplitz.f90 文件的 Alan Miller 修改而来，可在以下网址找到
    # http://jblevins.org/mirror/amiller/toeplitz.f90
    # 以公共领域声明发布。

    # 根据 dz 的类型选择正确的数据类型
    if dz is float64_t:
        dtype = float64
    else:
        dtype = complex128

    # 定义各种变量
    cdef ssize_t n, m, j, nmj, k, m2
    n = b.shape[0]
    cdef dz x_num, g_num, h_num, x_den, g_den
    cdef dz gj, gk, hj, hk, c1, c2
    # 初始化结果和工作空间
    cdef dz[:] x = zeros(n, dtype=dtype)  # 结果
    cdef dz[:] g = zeros(n, dtype=dtype)  # 工作空间
    cdef dz[:] h = zeros(n, dtype=dtype)  # 工作空间
    cdef dz[:] reflection_coeff = zeros(n+1, dtype=dtype)  # 历史记录
    assert len(a) == (2*n) - 1  # 断言确保 a 的长度符合预期

    # 检查主次列是否为奇异，若是则抛出异常
    if a[n-1] == 0:
        raise LinAlgError('Singular principal minor')

    # 计算第一个解和反射系数
    x[0] = b[0] / a[n-1]
    reflection_coeff[0] = 1
    reflection_coeff[1] = x[0]

    # 如果 n 等于 1，直接返回结果
    if (n == 1):
        return asarray(x), asarray(reflection_coeff)

    # 计算初始 g 和 h 的值
    g[0] = a[n-2] / a[n-1]
    h[0] = a[n] / a[n-1]
    for m in range(1, n):
        # 计算 x[m] 的分子和分母
        x_num = -b[m]  # 分子初始化为 -b[m]
        x_den = -a[n-1]  # 分母初始化为 -a[n-1]

        # 计算 x[m] 的分子和分母，根据先前计算的 x[j] 和 g[m-j-1]
        for j in range(m):
            nmj = n + m - (j+1)
            x_num = x_num + a[nmj] * x[j]  # 更新 x_num
            x_den = x_den + a[nmj] * g[m-j-1]  # 更新 x_den

        if x_den == 0:
            raise LinAlgError('Singular principal minor')  # 若分母为零，抛出异常

        x[m] = x_num / x_den  # 计算 x[m]
        reflection_coeff[m+1] = x[m]  # 更新 reflection_coeff

        # 更新 x[j]，根据计算的 x[m] 和 g[m-j-1]
        for j in range(m):
            x[j] = x[j] - x[m] * g[m-j-1]

        if m == n-1:
            return asarray(x), asarray(reflection_coeff)  # 若 m 达到 n-1，返回结果

        # 计算 g[m] 和 h[m] 的分子和分母
        g_num = -a[n-m-2]  # 计算 g[m] 的分子
        h_num = -a[n+m]  # 计算 h[m] 的分子
        g_den = -a[n-1]  # 计算 g[m] 的分母

        # 根据先前计算的 g[j] 和 h[m-j-1] 更新 g_num, h_num 和 g_den
        for j in range(m):
            g_num = g_num + a[n+j-m-1] * g[j]
            h_num = h_num + a[n+m-j-1] * h[j]
            g_den = g_den + a[n+j-m-1] * h[m-j-1]

        if g_den == 0.0:
            raise LinAlgError("Singular principal minor")  # 若 g_den 为零，抛出异常

        # 计算 g[m] 和 h[m]
        g[m] = g_num / g_den
        h[m] = h_num / x_den

        # 更新 g[j] 和 h[k]，根据计算的 c1 和 c2
        k = m - 1
        m2 = (m + 1) >> 1
        c1 = g[m]
        c2 = h[m]
        for j in range(m2):
            gj = g[j]
            gk = g[k]
            hj = h[j]
            hk = h[k]
            g[j] = gj - (c1 * hk)
            g[k] = gk - (c1 * hj)
            h[j] = hj - (c2 * gk)
            h[k] = hk - (c2 * gj)
            k -= 1
```