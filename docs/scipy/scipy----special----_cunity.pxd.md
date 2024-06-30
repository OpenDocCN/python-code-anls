# `D:\src\scipysrc\scipy\scipy\special\_cunity.pxd`

```
# 导入必要的Cython和C库，这里使用numpy和一些数学函数
cimport numpy as np
from libc.math cimport fabs, sin, cos, exp, atan2

# 导入Cython代码中的复杂数据结构和函数声明
from ._complexstuff cimport (
    zisfinite, zabs, zpack, npy_cdouble_from_double_complex,
    double_complex_from_npy_cdouble)

# 导入外部C头文件中定义的函数和数据结构
cdef extern from "_complexstuff.h":
    np.npy_cdouble npy_clog(np.npy_cdouble x) nogil
    np.npy_cdouble npy_cexp(np.npy_cdouble x) nogil

cdef extern from "dd_real_wrappers.h":
    # 定义一个双精度结构体double2，包含两个double类型的成员hi和lo
    ctypedef struct double2:
        double hi
        double lo
    
    # 函数声明：将一个double类型转换为double2结构
    double2 dd_create_d(double x) nogil
    # 函数声明：双精度加法
    double2 dd_add(const double2* a, const double2* b) nogil
    # 函数声明：双精度乘法
    double2 dd_mul(const double2* a, const double2* b) nogil
    # 函数声明：将double2结构转换为double类型
    double dd_to_double(const double2* a) nogil

# 导入特殊函数的声明，这些函数用于数学运算
cdef extern from "special_wrappers.h" nogil:
    double cephes_cosm1_wrap(double x)
    double cephes_expm1_wrap(double x)
    double cephes_log1p_wrap(double x)

# 定义一个内联函数 clog1p，计算 log(z + 1)
# 当z非有限值时，将z + 1作为参数调用npy_clog函数
# 使用特殊情况处理小于0.707的az和zi，避免浮点数取消问题
# 使用双精度计算函数clog1p_ddouble处理浮点数取消问题
cdef inline double complex clog1p(double complex z) noexcept nogil:
    cdef double zr, zi, x, y, az, azi
    cdef np.npy_cdouble ret

    if not zisfinite(z):
        z = z + 1
        ret = npy_clog(npy_cdouble_from_double_complex(z))
        return double_complex_from_npy_cdouble(ret)

    zr = z.real
    zi = z.imag

    if zi == 0.0 and zr >= -1.0:
        return zpack(cephes_log1p_wrap(zr), 0.0)

    az = zabs(z)
    if az < 0.707:
        azi = fabs(zi)
        if zr < 0 and fabs(-zr - azi*azi/2)/(-zr) < 0.5:
            return clog1p_ddouble(zr, zi)
        else:
            x = 0.5 * cephes_log1p_wrap(az*(az + 2*zr/az))
            y = atan2(zi, zr + 1.0)
            return zpack(x, y)

    z = z + 1.0
    ret = npy_clog(npy_cdouble_from_double_complex(z))
    return double_complex_from_npy_cdouble(ret)

# 内联函数 clog1p_ddouble，用双精度处理浮点数取消问题
# 使用双精度算法计算log1p，并返回复数结果
cdef inline double complex clog1p_ddouble(double zr, double zi) noexcept nogil:
    cdef double x, y
    cdef double2 r, i, two, rsqr, isqr, rtwo, absm1

    r = dd_create_d(zr)
    i = dd_create_d(zi)
    two = dd_create_d(2.0)

    rsqr = dd_mul(&r,& r)
    isqr = dd_mul(&i, &i)
    rtwo = dd_mul(&two, &r)
    absm1 = dd_add(&rsqr, &isqr)
    absm1 = dd_add(&absm1, &rtwo)

    x = 0.5 * cephes_log1p_wrap(dd_to_double(&absm1))
    y = atan2(zi, zr+1.0)
    return zpack(x, y)

# cexpm1(z) = cexp(z) - 1
# 计算 cexpm1 的实部和虚部
# 实部使用exp(z.real)*sin(z.imag)计算，虚部使用exp(z.real)*cos(z.imag)计算
# 当z的模很小时，采用expm1的逼近公式，避免浮点数取消问题
# 定义一个内联函数 cexpm1，用于计算复数 z 的指数减一的结果
cdef inline double complex cexpm1(double complex z) noexcept nogil:
    # 定义双精度变量 zr 和 zi，用于分别存储 z 的实部和虚部
    cdef double zr, zi, ezr, x, y
    # 定义 numpy 复数 ret，用于存储计算结果

    # 如果 z 不是有限数，计算 npy_cexp(npy_cdouble_from_double_complex(z)) 的结果，
    # 并将其转换为双精度复数，然后减去 1.0
    if not zisfinite(z):
        ret = npy_cexp(npy_cdouble_from_double_complex(z))
        return double_complex_from_npy_cdouble(ret) - 1.0

    # 将 z 的实部和虚部分别存入 zr 和 zi
    zr = z.real
    zi = z.imag

    # 根据 zr 的值选择计算 x 的方式
    if zr <= -40:
        x = -1.0
    else:
        # 使用 cephes_expm1_wrap 计算 exp(zr) - 1
        ezr = cephes_expm1_wrap(zr)
        # 计算 x = ezr * cos(zi) + cephes_cosm1_wrap(zi)
        x = ezr * cos(zi) + cephes_cosm1_wrap(zi)

    # 根据 zr 的值选择计算 y 的方式
    # 不计算 exp(zr) 除非必要
    if zr > -1.0:
        # 计算 y = (ezr + 1.0) * sin(zi)
        y = (ezr + 1.0) * sin(zi)
    else:
        # 计算 y = exp(zr) * sin(zi)
        y = exp(zr) * sin(zi)

    # 返回复数 z 的包装结果 zpack(x, y)
    return zpack(x, y)
```