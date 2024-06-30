# `D:\src\scipysrc\scipy\scipy\special\_sici.pxd`

```
# Implementation of sin/cos/sinh/cosh integrals for complex arguments
#
# Sources
# [1] Fredrik Johansson and others. mpmath: a Python library for
#     arbitrary-precision floating-point arithmetic (version 0.19),
#     December 2013. http://mpmath.org/.
# [2] NIST, "Digital Library of Mathematical Functions",
#     https://dlmf.nist.gov/
import cython  # 导入Cython模块，用于编写C扩展
from libc.math cimport M_PI, M_PI_2, NAN, INFINITY  # 从C标准库中导入常量和函数
cimport numpy as np  # 导入Cython版的NumPy库的部分

from . cimport sf_error  # 从当前包中导入sf_error模块

from ._complexstuff cimport (
    npy_cdouble_from_double_complex, double_complex_from_npy_cdouble,
    zabs, zlog, zpack)  # 从_Cython扩展模块_complexstuff中导入多个函数和类型

cdef extern from "special_wrappers.h":
    np.npy_cdouble special_cexpi(np.npy_cdouble) nogil  # 从C头文件中声明C函数special_cexpi，并指定不使用全局解释器锁（GIL）

DEF EULER = 0.577215664901532860606512090082402431  # 定义欧拉常数EULER

cdef inline double complex zexpi(double complex z) noexcept nogil:
    """计算复数参数z的指数函数e^z，使用特殊C函数special_cexpi进行计算。

    """
    cdef np.npy_cdouble r
    r = special_cexpi(npy_cdouble_from_double_complex(z))
    return double_complex_from_npy_cdouble(r)

@cython.cdivision(True)
cdef inline void power_series(int sgn, double complex z,
                             double complex *s, double complex *c) noexcept nogil:
    """计算复数参数z的幂级数展开，用于求解sin/cos或sinh/cosh积分。如果sgn = -1，计算si/ci；如果sgn = 1，计算shi/chi。

    """
    cdef:
        int n
        double complex fac, term1, term2
        int MAXITER = 100
        double tol = 2.220446092504131e-16
        
    fac = z
    s[0] = fac
    c[0] = 0        
    for n in range(1, MAXITER):
        fac *= sgn*z/(2*n)
        term2 = fac/(2*n)
        c[0] += term2
        fac *= z/(2*n + 1)
        term1 = fac/(2*n + 1)
        s[0] += term1
        if zabs(term1) < tol*zabs(s[0]) and zabs(term2) < tol*zabs(c[0]):
            break

    
cdef inline int csici(double complex z,
                      double complex *si, double complex *ci) noexcept nogil:
    """计算复数参数z的sin/cos积分。算法主要参考文献[1]。

    """
    cdef double complex jz, term1, term2

    if z == INFINITY:
        si[0] = M_PI_2
        ci[0] = 0
        return 0
    elif z == -INFINITY:
        si[0] = -M_PI_2
        ci[0] = 1j*M_PI
        return 0
    elif zabs(z) < 0.8:
        # 使用级数展开避免si中的取消
        power_series(-1, z, si, ci)
        if z == 0:
            sf_error.error("sici", sf_error.DOMAIN, NULL)
            ci[0] = zpack(-INFINITY, NAN)
        else:
            ci[0] += EULER + zlog(z)
        return 0
    
    # 使用DLMF 6.5.5/6.5.6加上DLMF 6.4.4/6.4.6/6.4.7的方法
    jz = 1j*z
    term1 = zexpi(jz)
    term2 = zexpi(-jz)
    si[0] = -0.5j*(term1 - term2)
    ci[0] = 0.5*(term1 + term2)
    if z.real == 0:
        if z.imag > 0:
            ci[0] += 1j*M_PI_2
        elif z.imag < 0:
            ci[0] -= 1j*M_PI_2
    elif z.real > 0:
        si[0] -= M_PI_2
    else:
        si[0] += M_PI_2
        if z.imag >= 0:
            ci[0] += 1j*M_PI
        else:
            ci[0] -= 1j*M_PI

    return 0
# 定义一个内联函数，计算复数参数 z 处的双曲正弦和双曲余弦积分。该算法主要基于参考文献 [1]。

cdef inline int cshichi(double complex z,
                        double complex *shi, double complex *chi) noexcept nogil:
    """Compute sinh/cosh integrals at complex arguments. The algorithm
    largely follows that of [1].
    计算复数参数 z 处的双曲正弦和双曲余弦积分。算法主要遵循参考文献 [1]。

    """
    # 定义两个复数变量 term1 和 term2
    cdef double complex term1, term2

    # 如果 z 等于正无穷
    if z == INFINITY:
        # 设置双曲正弦和双曲余弦积分结果为正无穷
        shi[0] = INFINITY
        chi[0] = INFINITY
        return 0
    # 如果 z 等于负无穷
    elif z == -INFINITY:
        # 设置双曲正弦积分结果为负无穷，双曲余弦积分结果为正无穷
        shi[0] = -INFINITY
        chi[0] = INFINITY
        return 0
    # 如果 z 的绝对值小于 0.8
    elif zabs(z) < 0.8:
        # 使用级数展开以避免在 shi 中的相消现象
        power_series(1, z, shi, chi)
        # 如果 z 等于 0
        if z == 0:
            # 报告错误，此时 chi[0] 被设置为复数无穷大
            sf_error.error("shichi", sf_error.DOMAIN, NULL)
            chi[0] = zpack(-INFINITY, NAN)
        else:
            # 计算双曲余弦积分的附加项，包括欧拉常数和 z 的对数
            chi[0] += EULER + zlog(z)
        return 0

    # 计算 term1 和 term2 的值，这两个值用于计算双曲正弦和双曲余弦积分
    term1 = zexpi(z)
    term2 = zexpi(-z)
    shi[0] = 0.5*(term1 - term2)
    chi[0] = 0.5*(term1 + term2)
    
    # 如果 z 的虚部大于 0
    if z.imag > 0:
        # 调整双曲正弦和双曲余弦积分的虚部，减去 0.5j*π
        shi[0] -= 0.5j*M_PI
        chi[0] += 0.5j*M_PI
    # 如果 z 的虚部小于 0
    elif z.imag < 0:
        # 调整双曲正弦和双曲余弦积分的虚部，加上 0.5j*π
        shi[0] += 0.5j*M_PI
        chi[0] -= 0.5j*M_PI
    # 如果 z 的实部小于 0
    elif z.real < 0:
        # 调整双曲余弦积分的虚部，加上 1j*π
        chi[0] += 1j*M_PI

    # 函数返回 0，表示计算完成
    return 0
```