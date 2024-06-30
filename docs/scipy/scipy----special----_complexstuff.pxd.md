# `D:\src\scipysrc\scipy\scipy\special\_complexstuff.pxd`

```
# -*-cython-*-
# Cython文件的声明，指示这是一个Cython编译的文件

# 导入Cython模块
import cython
# 导入Cython版本的NumPy模块，使用np别名
cimport numpy as np
# 导入C语言级别的数学函数，包括isnan, isinf, isfinite, log1p, fabs, exp, log, sin, cos, sqrt, pow
from libc.math cimport (
    isnan, isinf, isfinite, log1p, fabs, exp, log, sin, cos, sqrt, pow
)

# 从npy_2_complexcompat.h头文件中导入函数声明
cdef extern from "npy_2_complexcompat.h":
    void NPY_CSETREAL(np.npy_cdouble *c, double real) nogil
    void NPY_CSETIMAG(np.npy_cdouble *c, double imag) nogil

# 从_complexstuff.h头文件中导入函数声明
cdef extern from "_complexstuff.h":
    double npy_cabs(np.npy_cdouble z) nogil
    double npy_carg(np.npy_cdouble z) nogil
    np.npy_cdouble npy_clog(np.npy_cdouble z) nogil
    np.npy_cdouble npy_cexp(np.npy_cdouble z) nogil
    np.npy_cdouble npy_csin(np.npy_cdouble z) nogil
    np.npy_cdouble npy_ccos(np.npy_cdouble z) nogil
    np.npy_cdouble npy_csqrt(np.npy_cdouble z) nogil
    np.npy_cdouble npy_cpow(np.npy_cdouble x, np.npy_cdouble y) nogil

# 定义double_complex类型别名，表示复数类型
ctypedef double complex double_complex

# 定义number_t融合类型，包括double和double_complex
ctypedef fused number_t:
    double
    double_complex

# 定义union类型_complex_pun，包括npy_cdouble和double_complex，用于类型转换
ctypedef union _complex_pun:
    np.npy_cdouble npy
    double_complex c99

# 定义内联函数，将double_complex转换为npy_cdouble类型
cdef inline np.npy_cdouble npy_cdouble_from_double_complex(
        double_complex x) noexcept nogil:
    cdef _complex_pun z
    z.c99 = x
    return z.npy

# 定义内联函数，将npy_cdouble类型转换为double_complex类型
cdef inline double_complex double_complex_from_npy_cdouble(
        np.npy_cdouble x) noexcept nogil:
    cdef _complex_pun z
    z.npy = x
    return z.c99

# 定义内联函数，判断复数或实数是否包含NaN
cdef inline bint zisnan(number_t x) noexcept nogil:
    if number_t is double_complex:
        return isnan(x.real) or isnan(x.imag)
    else:
        return isnan(x)

# 定义内联函数，判断复数或实数是否有限
cdef inline bint zisfinite(number_t x) noexcept nogil:
    if number_t is double_complex:
        return isfinite(x.real) and isfinite(x.imag)
    else:
        return isfinite(x)

# 定义内联函数，判断复数或实数是否为无穷大
cdef inline bint zisinf(number_t x) noexcept nogil:
    if number_t is double_complex:
        return not zisnan(x) and not zisfinite(x)
    else:
        return isinf(x)

# 定义内联函数，获取复数或实数的实部
cdef inline double zreal(number_t x) noexcept nogil:
    if number_t is double_complex:
        return x.real
    else:
        return x

# 定义内联函数，获取复数的模
cdef inline double zabs(number_t x) noexcept nogil:
    if number_t is double_complex:
        return npy_cabs(npy_cdouble_from_double_complex(x))
    else:
        return fabs(x)

# 定义内联函数，获取复数的幅角
cdef inline double zarg(double complex x) noexcept nogil:
    return npy_carg(npy_cdouble_from_double_complex(x))

# 定义内联函数，计算复数或实数的自然对数
cdef inline number_t zlog(number_t x) noexcept nogil:
    cdef np.npy_cdouble r
    if number_t is double_complex:
        r = npy_clog(npy_cdouble_from_double_complex(x))
        return double_complex_from_npy_cdouble(r)
    else:
        return log(x)

# 定义内联函数，计算复数或实数的指数函数
cdef inline number_t zexp(number_t x) noexcept nogil:
    cdef np.npy_cdouble r
    if number_t is double_complex:
        r = npy_cexp(npy_cdouble_from_double_complex(x))
        return double_complex_from_npy_cdouble(r)
    else:
        return exp(x)

# 定义内联函数，计算复数或实数的正弦值
cdef inline number_t zsin(number_t x) noexcept nogil:
    cdef np.npy_cdouble r
    # 如果变量 number_t 是 double_complex 类型
    if number_t is double_complex:
        # 调用 numpy 库中的 csin 函数，传入经过转换后的 x（从 double_complex 转为 npy_cdouble）
        r = npy_csin(npy_cdouble_from_double_complex(x))
        # 将 numpy 返回的 npy_cdouble 转换回 double_complex 类型，并返回结果
        return double_complex_from_npy_cdouble(r)
    else:
        # 如果 number_t 不是 double_complex 类型，则直接调用标准库中的 sin 函数
        return sin(x)
# 定义一个 Cython 的内联函数 zcos，用于计算余弦值
cdef inline number_t zcos(number_t x) noexcept nogil:
    # 声明复数类型变量 r
    cdef np.npy_cdouble r
    # 如果 number_t 是 double_complex 类型
    if number_t is double_complex:
        # 调用 npy_ccos 函数计算复数 x 的余弦值，结果存入 r
        r = npy_ccos(npy_cdouble_from_double_complex(x))
        # 将 np.npy_cdouble 类型转换为 double_complex，并返回结果
        return double_complex_from_npy_cdouble(r)
    else:
        # 如果 number_t 不是 double_complex 类型，直接计算实数 x 的余弦值并返回
        return cos(x)

# 定义一个 Cython 的内联函数 zsqrt，用于计算平方根
cdef inline number_t zsqrt(number_t x) noexcept nogil:
    # 声明复数类型变量 r
    cdef np.npy_cdouble r
    # 如果 number_t 是 double_complex 类型
    if number_t is double_complex:
        # 调用 npy_csqrt 函数计算复数 x 的平方根，结果存入 r
        r = npy_csqrt(npy_cdouble_from_double_complex(x))
        # 将 np.npy_cdouble 类型转换为 double_complex，并返回结果
        return double_complex_from_npy_cdouble(r)
    else:
        # 如果 number_t 不是 double_complex 类型，直接计算实数 x 的平方根并返回
        return sqrt(x)

# 定义一个 Cython 的内联函数 zpow，用于计算数的幂次方
cdef inline number_t zpow(number_t x, double y) noexcept nogil:
    # 声明复数类型变量 r 和 z
    cdef np.npy_cdouble r, z
    # FIXME：标记需要修复的问题，这里未提供具体修复内容

    # 如果 number_t 是 double_complex 类型
    if number_t is double_complex:
        # 将 y 设置为复数类型，虚部为 0
        NPY_CSETREAL(&z, y)
        NPY_CSETIMAG(&z, 0.0)
        # 调用 npy_cpow 函数计算复数 x 的 z 次幂，结果存入 r
        r = npy_cpow(npy_cdouble_from_double_complex(x), z)
        # 将 np.npy_cdouble 类型转换为 double_complex，并返回结果
        return double_complex_from_npy_cdouble(r)
    else:
        # 如果 number_t 不是 double_complex 类型，直接计算 x 的 y 次幂并返回
        return pow(x, y)

# 定义一个 Cython 的内联函数 zpack，用于创建复数
cdef inline double_complex zpack(double zr, double zi) noexcept nogil:
    # 声明复数类型变量 z
    cdef np.npy_cdouble z
    # 设置 z 的实部和虚部
    NPY_CSETREAL(&z, zr)
    NPY_CSETIMAG(&z, zi)
    # 将 np.npy_cdouble 类型转换为 double_complex，并返回结果
    return double_complex_from_npy_cdouble(z)

# 定义一个 Cython 的内联函数 zlog1，用于计算对数，特别关注在 z 接近 1 时的精度
@cython.cdivision(True)
cdef inline double complex zlog1(double complex z) noexcept nogil:
    """
    Compute log, paying special attention to accuracy around 1. We
    implement this ourselves because some systems (most notably the
    Travis CI machines) are weak in this regime.

    """
    # 声明整型变量 n，复数变量 coeff 和 res，以及双精度浮点变量 tol
    cdef:
        int n
        double complex coeff = -1
        double complex res = 0
        double tol = 2.220446092504131e-16

    # 如果 z 到 1 的距离超过 0.1
    if zabs(z - 1) > 0.1:
        # 调用标准库中的 zlog 函数计算 z 的对数并返回结果
        return zlog(z)
    
    # z 减去 1
    z = z - 1
    # 如果 z 等于 0，直接返回 0
    if z == 0:
        return 0
    
    # 循环计算级数展开，最多 16 次
    for n in range(1, 17):
        coeff *= -z
        res += coeff/n
        # 如果当前项除以 coeff 的绝对值小于给定的 tol，则结束循环
        if zabs(res/coeff) < tol:
            break
    
    # 返回计算得到的对数结果
    return res
```