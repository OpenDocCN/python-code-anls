# `D:\src\scipysrc\scipy\scipy\special\_hyp0f1.pxd`

```
# 导入一些特定的数学函数和常量，这些函数和常量从C库中导入
from libc.math cimport pow, sqrt, floor, log, log1p, exp, M_PI, NAN, fabs, isinf
cimport numpy as np  # 导入Cython版的NumPy

# 导入一些特定的函数和结构体，这些从Cython文件中导入
from ._xlogy cimport xlogy
from ._complexstuff cimport (
    zsqrt, zpow, zabs, npy_cdouble_from_double_complex,
    double_complex_from_npy_cdouble)

# 从特定的头文件 "special_wrappers.h" 中导入一些C函数，使用 `nogil` 关键字来指示没有GIL
cdef extern from "special_wrappers.h" nogil:
    double cephes_iv_wrap(double v, double x)  # 调用 C 函数 cephes_iv_wrap
    double cephes_jv_wrap(double v, double x)  # 调用 C 函数 cephes_jv_wrap
    double cephes_gamma_wrap(double x)         # 调用 C 函数 cephes_gamma_wrap
    double cephes_lgam_wrap(double x)          # 调用 C 函数 cephes_lgam_wrap
    double cephes_gammasgn_wrap(double x)      # 调用 C 函数 cephes_gammasgn_wrap

# 从 "float.h" 头文件中导入一些特定的双精度浮点数常量
cdef extern from "float.h":
    double DBL_MAX, DBL_MIN

# 从特定的头文件 "special_wrappers.h" 中导入一些特定的C函数和结构体，使用 `nogil` 关键字来指示没有GIL
cdef extern from "special_wrappers.h":
    np.npy_cdouble special_ccyl_bessel_i(double v, np.npy_cdouble z) nogil  # 调用 C 函数 special_ccyl_bessel_i
    np.npy_cdouble special_ccyl_bessel_j(double v, np.npy_cdouble z) nogil  # 调用 C 函数 special_ccyl_bessel_j
    double special_sinpi(double x) nogil                                       # 调用 C 函数 special_sinpi

# 从 "numpy/npy_math.h" 头文件中导入一个特定的双精度浮点数函数
cdef extern from "numpy/npy_math.h":
    double npy_creal(np.npy_cdouble z) nogil  # 调用 C 函数 npy_creal

#
# 实数值核心函数
#
cdef inline double _hyp0f1_real(double v, double z) noexcept nogil:
    cdef double arg, v1, arg_exp, bess_val

    # 处理极点和零点情况
    if v <= 0.0 and v == floor(v):
        return NAN  # 如果 v 小于等于 0 且为整数，返回 NaN
    if z == 0.0 and v != 0.0:
        return 1.0  # 如果 z 等于 0 且 v 不等于 0，返回 1.0

    # 当 v 和 z 都很小的时候，将泰勒级数截断至 O(z**2)
    if fabs(z) < 1e-6*(1.0 + fabs(v)):
        return 1.0 + z/v + z*z/(2.0*v*(v+1.0))

    if z > 0:
        arg = sqrt(z)  # 计算 z 的平方根
        arg_exp = xlogy(1.0-v, arg) + cephes_lgam_wrap(v)  # 计算对数乘积 xlogy 和特殊 gamma 函数 cephes_lgam_wrap
        bess_val = cephes_iv_wrap(v-1, 2.0*arg)  # 调用函数 cephes_iv_wrap 计算修正的 Bessel 函数

        # 如果超过浮点数极限或者出现下溢，则返回渐近展开的结果
        if (arg_exp > log(DBL_MAX) or bess_val == 0 or
            arg_exp < log(DBL_MIN) or isinf(bess_val)):
            return _hyp0f1_asy(v, z)  # 调用渐近展开函数
        else:
            return exp(arg_exp) * cephes_gammasgn_wrap(v) * bess_val  # 计算最终结果
    else:
        arg = sqrt(-z)  # 计算 -z 的平方根
        return pow(arg, 1.0 - v) * cephes_gamma_wrap(v) * cephes_jv_wrap(v - 1, 2*arg)  # 计算最终结果

# 内联函数，用于计算渐近展开结果
cdef inline double _hyp0f1_asy(double v, double z) noexcept nogil:
    r"""Asymptotic expansion for I_{v-1}(2*sqrt(z)) * Gamma(v)
    for real $z > 0$ and $v\to +\infty$.

    Based off DLMF 10.41
    """
    cdef:
        double arg = sqrt(z)
        double v1 = fabs(v - 1)
        double x = 2.0 * arg / v1
        double p1 = sqrt(1.0 + x*x)
        double eta = p1 + log(x) - log1p(p1)
        double arg_exp_i, arg_exp_k
        double pp, p2, p4, p6, u1, u2, u3, u_corr_i, u_corr_k
        double result, gs

    arg_exp_i = -0.5*log(p1)
    arg_exp_i -= 0.5*log(2.0*M_PI*v1)
    arg_exp_i += cephes_lgam_wrap(v)
    gs = cephes_gammasgn_wrap(v)

    arg_exp_k = arg_exp_i
    arg_exp_i += v1 * eta
    arg_exp_k -= v1 * eta

    # 大 v 渐近修正，参考 DLMF 10.41.10
    pp = 1.0/p1
    p2 = pp*pp
    p4 = p2*p2
    p6 = p4*p2
    u1 = (3.0 - 5.0*p2) * pp / 24.0
    u2 = (81.0 - 462.0*p2 + 385.0*p4) * p2 / 1152.0
    u3 = (30375.0 - 369603.0*p2 + 765765.0*p4 - 425425.0*p6) * pp * p2 / 414720.0
    u_corr_i = 1.0 + u1/v1 + u2/(v1*v1) + u3/(v1*v1*v1)

    result = exp(arg_exp_i - xlogy(v1, arg)) * gs * u_corr_i  # 计算最终结果
    # 如果 v - 1 小于 0，则执行以下语句块
    if v - 1 < 0:
        # 根据 DLMF 10.27.2 公式计算修正系数 u_corr_k
        u_corr_k = 1.0 - u1/v1 + u2/(v1*v1) - u3/(v1*v1*v1)
        # 计算结果 result 增加 exp(arg_exp_k + xlogy(v1, arg)) 乘以修正系数 u_corr_k、gs、2.0 和 special_sinpi(v1) 的乘积
        result += exp(arg_exp_k + xlogy(v1, arg)) * gs * 2.0 * special_sinpi(v1) * u_corr_k

    # 返回最终的结果 result
    return result
#
# 复数内核函数
#
# 定义一个内联函数，计算复数输入的超几何函数 $_0F_1(v, z)$
cdef inline double complex _hyp0f1_cmplx(double v, double complex z) noexcept nogil:
    cdef:
        # 将复数 z 转换为 NumPy 中的复数类型
        np.npy_cdouble zz = npy_cdouble_from_double_complex(z)
        np.npy_cdouble r
        double complex arg, s
        double complex t1, t2

    # 处理极点和零点
    if v <= 0.0 and v == floor(v):
        # 如果 v 小于等于零且是整数，则返回 NaN
        return NAN
    if z.real == 0.0 and z.imag == 0.0 and v != 0.0:
        # 如果 z 是实部和虚部都为零且 v 不等于零，则返回 1.0
        return 1.0

    # 当 v 和 z 都很小：在 O(z**2) 处截断 Taylor 级数
    if zabs(z) < 1e-6*(1.0 + zabs(v)):
        # 需要按照这个顺序计算，因为 $v\approx -z \ll 1$ 时可能会失去精度
        t1 = 1.0 + z/v
        t2 = z*z / (2.0*v*(v+1.0))
        return t1 + t2

    # 如果 z 的实部大于 0
    if npy_creal(zz) > 0:
        # 计算 z 的平方根，并做相应的计算
        arg = zsqrt(z)
        s = 2.0 * arg
        # 调用特殊的修正圆柱贝塞尔函数 I_{v-1}(2*sqrt(z))
        r = special_ccyl_bessel_i(v-1.0, npy_cdouble_from_double_complex(s))
    else:
        # 计算 -z 的平方根，并做相应的计算
        arg = zsqrt(-z)
        s = 2.0 * arg
        # 调用特殊的修正圆柱贝塞尔函数 J_{v-1}(2*sqrt(-z))
        r = special_ccyl_bessel_j(v-1.0, npy_cdouble_from_double_complex(s))

    # 返回结果，乘以 gamma 函数的值和 z 的 (1.0 - v) 次幂
    return double_complex_from_npy_cdouble(r) * cephes_gamma_wrap(v) * zpow(arg, 1.0 - v)
```