# `D:\src\scipysrc\scipy\scipy\special\_spence.pxd`

```
# cython: cpow=True

# 导入 Cython 模块，用于编写 C 扩展
import cython

# 从 _complexstuff 模块中导入 zlog1 和 zabs 函数
from ._complexstuff cimport zlog1, zabs

# 相对误差容限，用于级数计算
DEF TOL = 2.220446092504131e-16
# pi^2 / 6 的常数值
DEF PISQ_6 = 1.6449340668482264365


# 定义一个内联函数，计算复数参数 z 的 Spence 函数
@cython.cdivision(True)
cdef inline double complex cspence(double complex z) noexcept nogil:
    """
    计算复数参数 z 的 Spence 函数。计算策略如下：
    - 如果 z 接近 0，则使用以 0 为中心的级数。
    - 如果 z 远离 1，则使用反射公式

    spence(z) = -spence(z/(z - 1)) - pi**2/6 - ln(z - 1)**2/2

    将 z 移近 1。参见 [1]。
    - 如果 z 接近 1，则使用以 1 为中心的级数。
    """
    if zabs(z) < 0.5:
        # 这一步骤并非必需，但该级数收敛更快。
        return cspence_series0(z)
    elif zabs(1 - z) > 1:
        return -cspence_series1(z/(z - 1)) - PISQ_6 - 0.5*zlog1(z - 1)**2
    else:
        return cspence_series1(z)


# 定义一个内联函数，计算以 z = 0 为中心的级数
@cython.cdivision(True)
cdef inline double complex cspence_series0(double complex z) noexcept nogil:
    """
    以 z = 0 为中心的级数；参见

    http://functions.wolfram.com/10.07.06.0005.02

    """
    cdef:
        int n
        double complex zfac = 1
        double complex sum1 = 0
        double complex sum2 = 0
        double complex term1, term2

    if z == 0:
        return PISQ_6

    for n in range(1, 500):
        zfac *= z
        term1 = zfac/n**2
        sum1 += term1
        term2 = zfac/n
        sum2 += term2
        if zabs(term1) <= TOL*zabs(sum1) and zabs(term2) <= TOL*zabs(sum2):
            break
    return PISQ_6 - sum1 + zlog1(z)*sum2


# 定义一个内联函数，计算以 z = 1 为中心的级数
@cython.cdivision(True)
cdef inline double complex cspence_series1(double complex z) noexcept nogil:
    """
    以 z = 1 为中心的级数，比泰勒级数收敛更快。参见 [3]。
    使用的项数来自于将绝对容限限制在收敛半径边缘的位置，其中总和为 O(1)。

    """
    cdef:
        int n
        double complex zfac = 1
        double complex res = 0
        double complex term, zz

    if z == 1:
        return 0
    z = 1 - z
    zz = z**2
    for n in range(1, 500):
        zfac *= z
        # 逐个进行除法，以防止溢出
        term = ((zfac/n**2)/(n + 1)**2)/(n + 2)**2
        res += term
        if zabs(term) <= TOL*zabs(res):
            break
    res *= 4*zz
    res += 4*z + 5.75*zz + 3*(1 - zz)*zlog1(1 - z)
    res /= 1 + 4*z + zz
    return res
```