# `D:\src\scipysrc\scipy\scipy\special\orthogonal_eval.pxd`

```
# -*- cython -*-
# cython: cpow=True
"""
使用递推关系计算正交多项式的值。

参考文献
----------

.. [AMS55] Abramowitz & Stegun, Section 22.5.

.. [MH] Mason & Handscombe, Chebyshev Polynomials, CRC Press (2003).

.. [LP] P. Levrie & R. Piessens, A note on the evaluation of orthogonal
        polynomials using recurrence relations, Internal Report TW74 (1985)
        Dept. of Computer Science, K.U. Leuven, Belgium
        https://lirias.kuleuven.be/handle/123456789/131600

"""
#
# 作者：Pauli Virtanen, Eric Moore
#

#------------------------------------------------------------------------------
# 直接评估多项式
#------------------------------------------------------------------------------
cimport cython
# 导入数学库函数，包括 sqrt, exp, floor, fabs, log, sin, isnan, NAN, pi
from libc.math cimport sqrt, exp, floor, fabs, log, sin, isnan, NAN, M_PI as pi

# 导入 numpy 复数类型
from numpy cimport npy_cdouble
# 导入自定义模块中的函数和类型
from ._complexstuff cimport (
    number_t,
    npy_cdouble_from_double_complex,
    double_complex_from_npy_cdouble
)

# 导入自定义模块中的错误处理函数
from . cimport sf_error


# 从外部头文件中导入特殊函数包装器
cdef extern from "special_wrappers.h" nogil:
    npy_cdouble hyp2f1_complex_wrap(double a, double b, double c, npy_cdouble zp)
    double binom_wrap(double n, double k)
    double cephes_hyp2f1_wrap(double a, double b, double c, double x)
    double cephes_gamma_wrap(double x)
    double cephes_beta_wrap(double a, double b)
    double hyp1f1_wrap(double a, double b, double x) nogil
    npy_cdouble chyp1f1_wrap( double a, double b, npy_cdouble z) nogil


# 融合类型包装器

# 定义 hyp2f1 函数，根据 number_t 的类型选择调用不同的包装函数
cdef inline number_t hyp2f1(double a, double b, double c, number_t z) noexcept nogil:
    cdef npy_cdouble r
    if number_t is double:
        return cephes_hyp2f1_wrap(a, b, c, z)
    else:
        r = hyp2f1_complex_wrap(a, b, c, npy_cdouble_from_double_complex(z))
        return double_complex_from_npy_cdouble(r)

# 定义 hyp1f1 函数，根据 number_t 的类型选择调用不同的包装函数
cdef inline number_t hyp1f1(double a, double b, number_t z) noexcept nogil:
    cdef npy_cdouble r
    if number_t is double:
        return hyp1f1_wrap(a, b, z)
    else:
        r = chyp1f1_wrap(a, b, npy_cdouble_from_double_complex(z))
        return double_complex_from_npy_cdouble(r)


#-----------------------------------------------------------------------------
# Jacobi
#-----------------------------------------------------------------------------

# 定义 eval_jacobi 函数，评估 Jacobi 多项式
cdef inline number_t eval_jacobi(double n, double alpha, double beta, number_t x) noexcept nogil:
    cdef double a, b, c, d
    cdef number_t g

    d = binom_wrap(n+alpha, n)
    a = -n
    b = n + alpha + beta + 1
    c = alpha + 1
    g = 0.5*(1-x)
    return d * hyp2f1(a, b, c, g)

# 定义 eval_jacobi_l 函数，评估 Jacobi 多项式（长整型版本）
@cython.cdivision(True)
cdef inline double eval_jacobi_l(long n, double alpha, double beta, double x) noexcept nogil:
    cdef long kk
    cdef double p, d
    cdef double k, t

    if n < 0:
        return eval_jacobi(n, alpha, beta, x)
    elif n == 0:
        return 1.0
    elif n == 1:
        return 0.5*(2*(alpha+1)+(alpha+beta+2)*(x-1))
    else:
        # 计算连分式中的初始值
        d = (alpha+beta+2)*(x - 1) / (2*(alpha+1))
        # 设置初始的连分式项
        p = d + 1
        # 循环计算连分式的每一项
        for kk in range(n-1):
            # 计算当前项的序号
            k = kk+1.0
            # 计算连分式的下一个 t 值
            t = 2*k+alpha+beta
            # 计算连分式的下一个 d 值
            d = ((t*(t+1)*(t+2))*(x-1)*p + 2*k*(k+beta)*(t+2)*d) / (2*(k+alpha+1)*(k+alpha+beta+1)*t)
            # 更新连分式的当前值
            p = d + p
        # 返回计算出的最终结果
        return binom_wrap(n+alpha, n)*p
#-----------------------------------------------------------------------------
# Shifted Jacobi
#-----------------------------------------------------------------------------

# 定义内联函数 eval_sh_jacobi，计算偏移雅各比多项式
@cython.cdivision(True)
cdef inline number_t eval_sh_jacobi(double n, double p, double q, number_t x) noexcept nogil:
    # 调用 eval_jacobi 函数计算雅各比多项式的值，参数做了一定的变换
    return eval_jacobi(n, p-q, q-1, 2*x-1) / binom_wrap(2*n + p - 1, n)

# 定义内联函数 eval_sh_jacobi_l，计算偏移雅各比多项式（长整型参数版本）
@cython.cdivision(True)
cdef inline double eval_sh_jacobi_l(long n, double p, double q, double x) noexcept nogil:
    # 调用 eval_jacobi_l 函数计算长整型参数版本的雅各比多项式的值，参数做了一定的变换
    return eval_jacobi_l(n, p-q, q-1, 2*x-1) / binom_wrap(2*n + p - 1, n)

#-----------------------------------------------------------------------------
# Gegenbauer (Ultraspherical)
#-----------------------------------------------------------------------------

# 定义内联函数 eval_gegenbauer，计算赫尔默特·赫恩·蒙塔诺多项式（Gegenbauer polynomial）
@cython.cdivision(True)
cdef inline number_t eval_gegenbauer(double n, double alpha, number_t x) noexcept nogil:
    cdef double a, b, c, d
    cdef number_t g

    # 计算赫尔默特·赫恩·蒙塔诺多项式的系数和参数
    d = cephes_gamma_wrap(n+2*alpha)/cephes_gamma_wrap(1+n)/cephes_gamma_wrap(2*alpha)
    a = -n
    b = n + 2*alpha
    c = alpha + 0.5
    g = (1-x)/2.0
    # 返回赫尔默特·赫恩·蒙塔诺多项式的值
    return d * hyp2f1(a, b, c, g)

# 定义内联函数 eval_gegenbauer_l，计算赫尔默特·赫恩·蒙塔诺多项式（长整型参数版本）
@cython.cdivision(True)
cdef inline double eval_gegenbauer_l(long n, double alpha, double x) noexcept nogil:
    cdef long kk
    cdef long a, b
    cdef double p, d
    cdef double k

    # 处理特殊情况
    if isnan(alpha) or isnan(x):
        return NAN

    if n < 0:
        return 0.0
    elif n == 0:
        return 1.0
    elif n == 1:
        return 2*alpha*x
    elif alpha == 0.0:
        return eval_gegenbauer(n, alpha, x)
    elif fabs(x) < 1e-5:
        # 当 x 接近 0 时，采用幂级数展开，而非递推公式，以避免精度损失
        # 参考 Wolfram Functions 中 GegenbauerC3 的幂级数展开
        a = n//2

        d = 1 if a % 2 == 0 else -1
        d /= cephes_beta_wrap(alpha, 1 + a)
        if n == 2*a:
            d /= (a + alpha)
        else:
            d *= 2*x

        p = 0
        for kk in range(a+1):
            # 计算幂级数展开的值
            p += d
            d *= -4*x**2 * (a - kk) * (-a + alpha + kk + n) / (
                (n + 1 - 2*a + 2*kk) * (n + 2 - 2*a + 2*kk))
            if fabs(d) == 1e-20*fabs(p):
                # 收敛时退出循环
                break
        return p
    else:
        # 一般情况下采用递推公式计算赫尔默特·赫恩·蒙塔诺多项式的值
        d = x - 1
        p = x
        for kk in range(n-1):
            k = kk+1.0
            d = (2*(k+alpha)/(k+2*alpha))*(x-1)*p + (k/(k+2*alpha)) * d
            p = d + p

        if fabs(alpha/n) < 1e-8:
            # 避免精度损失
            return 2*alpha/n * p
        else:
            return binom_wrap(n+2*alpha-1, n)*p

#-----------------------------------------------------------------------------
# Chebyshev 1st kind (T)
#-----------------------------------------------------------------------------

# 定义内联函数 eval_chebyt，计算第一类切比雪夫多项式
cdef inline number_t eval_chebyt(double n, number_t x) noexcept nogil:
    cdef double a, b, c, d
    cdef number_t g

    # 计算第一类切比雪夫多项式的系数和参数
    d = 1.0
    a = -n
    b = n
    c = 0.5
    g = 0.5*(1-x)
    # 返回第一类切比雪夫多项式的值
    return hyp2f1(a, b, c, g)

# 定义内联函数 eval_chebyt_l，计算第一类切比雪夫多项式（长整型参数版本）
cdef inline double eval_chebyt_l(long k, double x) noexcept nogil:
    # 直接使用切比雪夫 T 递推公式，参考文献 [MH]
    cdef long m
    cdef double b2, b1, b0

    if k < 0:
        # 处理负数情况，利用对称性质
        k = -k

    b2 = 0
    b1 = -1
    b0 = 0
    x = 2*x
    for m in range(k+1):
        b2 = b1
        b1 = b0
        b0 = x*b1 - b2
    # 计算并返回结果
    return (b0 - b2)/2.0
#-----------------------------------------------------------------------------
# Chebyshev 2nd kind (U)
#-----------------------------------------------------------------------------

# 定义一个内联函数，用于计算 Chebyshev 第二类函数 U_n(x) 的值
cdef inline number_t eval_chebyu(double n, number_t x) noexcept nogil:
    cdef double a, b, c, d
    cdef number_t g

    d = n+1
    a = -n
    b = n+2
    c = 1.5
    g = 0.5*(1-x)
    # 使用超几何函数计算 U_n(x) 的值并返回
    return d*hyp2f1(a, b, c, g)

# 定义一个内联函数，用于计算 Chebyshev 第二类函数 U_n(x) 的长整型版本的值
cdef inline double eval_chebyu_l(long k, double x) noexcept nogil:
    cdef long m
    cdef int sign
    cdef double b2, b1, b0

    if k == -1:
        return 0
    elif k < -1:
        # 对称性处理，计算 U_{-k-2}(x) 的值
        k = -k - 2
        sign = -1
    else:
        sign = 1

    b2 = 0
    b1 = -1
    b0 = 0
    x = 2*x
    # 使用循环计算 Chebyshev 第二类函数 U_n(x) 的值
    for m in range(k+1):
        b2 = b1
        b1 = b0
        b0 = x*b1 - b2
    # 返回计算结果乘以符号
    return b0 * sign

#-----------------------------------------------------------------------------
# Chebyshev S
#-----------------------------------------------------------------------------

# 定义一个内联函数，用于计算 Chebyshev S 函数 T_n(x) 的值
cdef inline number_t eval_chebys(double n, number_t x) noexcept nogil:
    # 调用 Chebyshev 第二类函数 U_n(x) 的计算函数 eval_chebyu
    return eval_chebyu(n, 0.5*x)

# 定义一个内联函数，用于计算 Chebyshev S 函数 T_n(x) 的长整型版本的值
cdef inline double eval_chebys_l(long n, double x) noexcept nogil:
    # 调用 Chebyshev 第二类函数 U_n(x) 的长整型版本的计算函数 eval_chebyu_l
    return eval_chebyu_l(n, 0.5*x)

#-----------------------------------------------------------------------------
# Chebyshev C
#-----------------------------------------------------------------------------

# 定义一个内联函数，用于计算 Chebyshev C 函数 C_n(x) 的值
cdef inline number_t eval_chebyc(double n, number_t x) noexcept nogil:
    # 计算 Chebyshev T 函数 T_n(x) 的值并乘以 2 返回
    return 2*eval_chebyt(n, 0.5*x)

# 定义一个内联函数，用于计算 Chebyshev C 函数 C_n(x) 的长整型版本的值
cdef inline double eval_chebyc_l(long n, double x) noexcept nogil:
    # 计算 Chebyshev T 函数 T_n(x) 的长整型版本的值并乘以 2 返回
    return 2*eval_chebyt_l(n, 0.5*x)

#-----------------------------------------------------------------------------
# Chebyshev 1st kind shifted
#-----------------------------------------------------------------------------

# 定义一个内联函数，用于计算 Chebyshev 第一类函数 T_n(x) 的 shifted 版本的值
cdef inline number_t eval_sh_chebyt(double n, number_t x) noexcept nogil:
    # 调用 Chebyshev 第一类函数 T_n(x) 的计算函数 eval_chebyt，并进行参数变换
    return eval_chebyt(n, 2*x-1)

# 定义一个内联函数，用于计算 Chebyshev 第一类函数 T_n(x) 的 shifted 版本的长整型版本的值
cdef inline double eval_sh_chebyt_l(long n, double x) noexcept nogil:
    # 调用 Chebyshev 第一类函数 T_n(x) 的长整型版本的计算函数 eval_chebyt_l，并进行参数变换
    return eval_chebyt_l(n, 2*x-1)

#-----------------------------------------------------------------------------
# Chebyshev 2nd kind shifted
#-----------------------------------------------------------------------------

# 定义一个内联函数，用于计算 Chebyshev 第二类函数 U_n(x) 的 shifted 版本的值
cdef inline number_t eval_sh_chebyu(double n, number_t x) noexcept nogil:
    # 调用 Chebyshev 第二类函数 U_n(x) 的计算函数 eval_chebyu，并进行参数变换
    return eval_chebyu(n, 2*x-1)

# 定义一个内联函数，用于计算 Chebyshev 第二类函数 U_n(x) 的 shifted 版本的长整型版本的值
cdef inline double eval_sh_chebyu_l(long n, double x) noexcept nogil:
    # 调用 Chebyshev 第二类函数 U_n(x) 的长整型版本的计算函数 eval_chebyu_l，并进行参数变换
    return eval_chebyu_l(n, 2*x-1)

#-----------------------------------------------------------------------------
# Legendre
#-----------------------------------------------------------------------------

# 定义一个内联函数，用于计算 Legendre 多项式 P_n(x) 的值
cdef inline number_t eval_legendre(double n, number_t x) noexcept nogil:
    cdef double a, b, c, d
    cdef number_t g

    d = 1
    a = -n
    b = n+1
    c = 1
    g = 0.5*(1-x)
    # 使用超几何函数计算 Legendre 多项式 P_n(x) 的值并返回
    return d*hyp2f1(a, b, c, g)

# 定义一个内联函数，用于计算 Legendre 多项式 P_n(x) 的长整型版本的值
@cython.cdivision(True)
cdef inline double eval_legendre_l(long n, double x) noexcept nogil:
    cdef long kk, a
    cdef double p, d
    cdef double k

    if n < 0:
        # 对称性处理，计算 P_{-n-1}(x) 的值
        n = -n - 1
    如果 n == 0:
        # 如果 n 等于 0，返回 1.0，这是勒让德多项式的基础情况 P_0(x) = 1.0
        return 1.0
    elif n == 1:
        # 如果 n 等于 1，返回 x，这是勒让德多项式的基础情况 P_1(x) = x
        return x
    elif fabs(x) < 1e-5:
        # 如果 x 的绝对值小于 1e-5，使用幂级数而不是递归，因为递归会由于精度损失而不适用
        # 参考 Wolfram 函数网的勒让德多项式 P_n(x) 的幂级数展开：http://functions.wolfram.com/Polynomials/LegendreP/02/
        a = n // 2

        d = 1 if a % 2 == 0 else -1
        if n == 2 * a:
            # 如果 n 是 2*a，根据公式调整 d 的值
            d *= -2 / cephes_beta_wrap(a + 1, -0.5)
        else:
            # 如果 n 不是 2*a，根据公式调整 d 的值
            d *= 2 * x / cephes_beta_wrap(a + 1, 0.5)

        p = 0
        for kk in range(a + 1):
            # 计算幂级数的每一项，直到收敛或达到最大精度
            p += d
            d *= -2 * x**2 * (a - kk) * (2 * n + 1 - 2 * a + 2 * kk) / (
                (n + 1 - 2 * a + 2 * kk) * (n + 2 - 2 * a + 2 * kk))
            if fabs(d) == 1e-20 * fabs(p):
                # 如果收敛到足够小的值，跳出循环
                # converged 表示收敛
                break
        return p
    else:
        # 对于一般情况，使用递推关系计算勒让德多项式 P_n(x)
        d = x - 1
        p = x
        for kk in range(n - 1):
            k = kk + 1.0
            d = ((2 * k + 1) / (k + 1)) * (x - 1) * p + (k / (k + 1)) * d
            p = d + p
        return p
`
#-----------------------------------------------------------------------------
# Legendre Shifted
#-----------------------------------------------------------------------------

# 定义一个 Cython 内联函数，计算 Legendre 多项式在 x 的值，参数 n 是多项式的次数，x 是自变量
cdef inline number_t eval_sh_legendre(double n, number_t x) noexcept nogil:
    # 调用 eval_legendre 函数计算 Legendre 多项式的值，传入的 x 已经经过变换 2*x-1
    return eval_legendre(n, 2*x-1)

# 定义一个 Cython 内联函数，计算 Legendre 多项式在 x 的值，参数 n 是多项式的次数，x 是自变量
cdef inline double eval_sh_legendre_l(long n, double x) noexcept nogil:
    # 调用 eval_legendre_l 函数计算 Legendre 多项式的值，传入的 x 已经经过变换 2*x-1
    return eval_legendre_l(n, 2*x-1)

#-----------------------------------------------------------------------------
# Generalized Laguerre
#-----------------------------------------------------------------------------

# 定义一个 Cython 内联函数，计算广义 Laguerre 多项式在 x 的值，参数 n 是多项式的次数，alpha 是参数，x 是自变量
cdef inline number_t eval_genlaguerre(double n, double alpha, number_t x) noexcept nogil:
    cdef double a, b, d
    cdef number_t g

    # 如果 alpha <= -1，抛出错误
    if alpha <= -1:
        sf_error.error("eval_genlaguerre", sf_error.DOMAIN,
                       "polynomial defined only for alpha > -1")
        return NAN

    # 计算 binom_wrap(n+alpha, n)
    d = binom_wrap(n+alpha, n)
    a = -n
    b = alpha + 1
    g = x
    # 调用 hyp1f1 函数计算超几何函数值，返回结果乘以 d
    return d * hyp1f1(a, b, g)

# 定义一个 Cython 内联函数，计算广义 Laguerre 多项式在 x 的值，参数 n 是多项式的次数，alpha 是参数，x 是自变量
@cython.cdivision(True)
cdef inline double eval_genlaguerre_l(long n, double alpha, double x) noexcept nogil:
    cdef long kk
    cdef double p, d
    cdef double k

    # 如果 alpha <= -1，抛出错误
    if alpha <= -1:
        sf_error.error("eval_genlaguerre", sf_error.DOMAIN,
                       "polynomial defined only for alpha > -1")
        return NAN

    # 如果 alpha 或 x 是 NaN，返回 NaN
    if isnan(alpha) or isnan(x):
        return NAN

    # 如果 n < 0，返回 0.0
    if n < 0:
        return 0.0
    # 如果 n == 0，返回 1.0
    elif n == 0:
        return 1.0
    # 如果 n == 1，返回 -x + alpha + 1
    elif n == 1:
        return -x + alpha + 1
    else:
        # 初始化 d 为 -x/(alpha+1)，p 为 d + 1
        d = -x/(alpha+1)
        p = d + 1
        # 使用迭代计算多项式值
        for kk in range(n-1):
            k = kk + 1.0
            d = -x/(k + alpha + 1) * p + (k / (k + alpha + 1)) * d
            p = d + p
        # 返回 binom_wrap(n+alpha, n) 乘以多项式值
        return binom_wrap(n + alpha, n) * p

#-----------------------------------------------------------------------------
# Laguerre
#-----------------------------------------------------------------------------

# 定义一个 Cython 内联函数，计算 Laguerre 多项式在 x 的值，参数 n 是多项式的次数，x 是自变量
cdef inline number_t eval_laguerre(double n, number_t x) noexcept nogil:
    # 调用 eval_genlaguerre 函数，alpha 为 0
    return eval_genlaguerre(n, 0., x)

# 定义一个 Cython 内联函数，计算 Laguerre 多项式在 x 的值，参数 n 是多项式的次数，x 是自变量
cdef inline double eval_laguerre_l(long n, double x) noexcept nogil:
    # 调用 eval_genlaguerre_l 函数，alpha 为 0
    return eval_genlaguerre_l(n, 0., x)

#-----------------------------------------------------------------------------
# Hermite (statistician's)
#-----------------------------------------------------------------------------

# 定义一个 Cython 内联函数，计算统计学家定义的 Hermite 多项式在 x 的值，参数 n 是多项式的次数，x 是自变量
cdef inline double eval_hermitenorm(long n, double x) noexcept nogil:
    cdef long k
    cdef double y1, y2, y3

    # 如果 x 是 NaN，返回 x
    if isnan(x):
        return x

    # 如果 n < 0，抛出错误
    if n < 0:
        sf_error.error(
            "eval_hermitenorm",
            sf_error.DOMAIN,
            "polynomial only defined for nonnegative n",
        )
        return NAN
    # 如果 n == 0，返回 1.0
    elif n == 0:
        return 1.0
    # 如果 n == 1，返回 x
    elif n == 1:
        return x
    else:
        # 初始化 y3 为 0.0，y2 为 1.0
        y3 = 0.0
        y2 = 1.0
        # 使用迭代计算多项式值
        for k in range(n, 1, -1):
            y1 = x * y2 - k * y3
            y3 = y2
            y2 = y1
        # 返回 x 乘以 y2 减去 y3
        return x * y2 - y3

#-----------------------------------------------------------------------------
# Hermite (physicist's)
#-----------------------------------------------------------------------------
# 启用 Cython 的 C 除法优化
@cython.cdivision(True)
# 定义内联函数 eval_hermite，接受一个长整型参数 n 和一个双精度浮点数参数 x，
# 并声明为无异常、无全局解锁锁定的函数
cdef inline double eval_hermite(long n, double x) noexcept nogil:
    # 如果 n 小于 0，抛出错误并返回 NaN
    if n < 0:
        sf_error.error(
            "eval_hermite",
            sf_error.DOMAIN,
            "polynomial only defined for nonnegative n",
        )
        return NAN
    # 返回 Hermite 多项式在参数 n 和 sqrt(2)*x 处的评估结果，乘以 2^(n/2.0)
    return eval_hermitenorm(n, sqrt(2)*x) * 2**(n/2.0)
```