# `D:\src\scipysrc\scipy\scipy\special\_agm.pxd`

```
# cython: cpow=True
# 导入 Cython 模块

import cython

# 从 C 标准库的 math 模块中导入一些函数和常量
from libc.math cimport log, exp, fabs, sqrt, isnan, isinf, NAN, M_PI

# 从特定的 C 头文件 "special_wrappers.h" 中导入外部 C 函数
cdef extern from "special_wrappers.h" nogil:
    double cephes_ellpk_wrap(double x)

# 内联函数，计算算术-几何均值的迭代实现
cdef inline double _agm_iter(double a, double b) noexcept nogil:
    # a 和 b 必须是正数（不是零，不是 nan）。

    cdef double amean, gmean

    cdef int count = 20
    amean = 0.5*a + 0.5*b
    while (count > 0) and (amean != a and amean != b):
        gmean = sqrt(a)*sqrt(b)
        a = amean
        b = gmean
        amean = 0.5*a + 0.5*b
        count -= 1
    return amean

# 定义计算算术-几何均值的函数
@cython.cdivision(True)
cdef inline double agm(double a, double b) noexcept nogil:
    # 算术-几何均值

    # sqrthalfmax 是 sqrt(np.finfo(1.0).max/2) 的近似值
    # invsqrthalfmax 是 1/sqrthalfmax 的近似值
    cdef double sqrthalfmax = 9.480751908109176e+153
    cdef double invsqrthalfmax = 1.0547686614863e-154

    cdef double e
    cdef int sgn

    # 处理 nan 的情况
    if isnan(a) or isnan(b):
        return NAN

    # 处理其中一个参数为负数的情况
    if (a < 0 and b > 0) or (a > 0 and b < 0):
        # a 和 b 符号相反。
        return NAN

    # 处理其中一个为无穷大，另一个为零的情况
    if (isinf(a) or isinf(b)) and (a == 0 or b == 0):
        # 一个值为无穷大，另一个为零。
        return NAN

    # 处理至少有一个参数为零的情况
    if a == 0 or b == 0:
        # 至少有一个参数为零。
        return 0.0

    # 如果 a 和 b 相等，则直接返回其中一个
    if a == b:
        return a

    # 处理 a 小于零的情况
    sgn = 1
    if a < 0:
        sgn = -1
        a = -a
        b = -b

    # 现在，a 和 b 都是正数且不是 nan。

    # 处理参数范围在极端情况下（非常大或非常小）的情况，以避免溢出或下溢
    if (invsqrthalfmax < a < sqrthalfmax) and (invsqrthalfmax < b < sqrthalfmax):
        e = 4*a*b/(a + b)**2
        return sgn*(M_PI/4)*(a + b)/cephes_ellpk_wrap(e)
    else:
        # 至少一个值是 "极端值"（非常大或非常小）。
        # 使用迭代来避免溢出或下溢。
        return sgn*_agm_iter(a, b)
```