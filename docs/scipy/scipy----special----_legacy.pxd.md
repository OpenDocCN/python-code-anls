# `D:\src\scipysrc\scipy\scipy\special\_legacy.pxd`

```
# -*- cython -*-
"""
Many Scipy special functions originally cast silently double input
arguments to integers.

Here, we define such unsafe wrappers manually.

"""

# 从 C 库中导入指定的函数和宏定义，这些函数在特殊情况下会使用 Cython 优化
from libc.math cimport isnan, isinf, NAN

# 导入 C 扩展中定义的错误处理函数
from . cimport sf_error

# 导入 C 扩展中定义的特殊椭圆函数
from ._ellip_harm cimport ellip_harmonic

# 从特殊的 C 函数库头文件中导入一系列具体的函数定义，这些函数将会在 Cython 中使用
cdef extern from "special_wrappers.h" nogil:
    double cephes_bdtrc_wrap(double k, int n, double p)
    double cephes_bdtr_wrap(double k, int n, double p)
    double cephes_bdtri_wrap(double k, int n, double y)
    double cephes_expn_wrap(int n, double x)
    double cephes_nbdtrc_wrap(int k, int n, double p)
    double cephes_nbdtr_wrap(int k, int n, double p)
    double cephes_nbdtri_wrap(int k, int n, double p)
    double cephes_pdtri_wrap(int k, double y)
    double cephes_yn_wrap(int n, double x)
    double cephes_smirnov_wrap(int n, double x)
    double cephes_smirnovc_wrap(int n, double x)
    double cephes_smirnovi_wrap(int n, double x)
    double cephes_smirnovci_wrap(int n, double x)
    double cephes_smirnovp_wrap(int n, double x)

# 从特殊的 C 函数库头文件中导入一个额外的函数定义，用于整数参数的特殊圆柱贝塞尔函数 K
cdef extern from "special_wrappers.h":
    double special_cyl_bessel_k_int(int n, double z) nogil

# 从 Python 的 C API 中导入异常处理相关定义，主要是警告相关的功能
cdef extern from "Python.h":
    # 忽略由 PyErr_WarnEx 抛出的异常 --- 假设 ufunc 将会收集这些异常
    int PyErr_WarnEx_noerr "PyErr_WarnEx" (object, char *, int)

# 定义一个内联函数，用于检查浮点数是否被截断为整数
cdef inline void _legacy_cast_check(char *func_name, double x, double y) noexcept nogil:
    if <int>x != x or <int>y != y:
        with gil:
            PyErr_WarnEx_noerr(RuntimeWarning,
                               "floating point number truncated to an integer",
                               1)

# 定义一个内联函数，用于警告关于不推荐使用的功能
cdef inline void _legacy_deprecation(char *func_name, double x, double y) noexcept nogil:
        with gil:
            PyErr_WarnEx_noerr(DeprecationWarning,
                               "non-integer arg n is deprecated, removed in SciPy 1.7.x",
                               1)

# 定义一个内联函数，用于计算不安全的椭圆函数，处理可能的 NaN 输入情况
cdef inline double ellip_harmonic_unsafe(double h2, double k2, double n,
                                         double p, double l, double signm,
                                         double signn) noexcept nogil:
    if isnan(n) or isnan(p):
        return NAN
    _legacy_cast_check("_ellip_harm", n, p)
    return ellip_harmonic(h2, k2, <int>n, <int>p, l, signm, signn)

# 定义一个内联函数，用于计算不安全的二项分布累积分布函数
cdef inline double bdtr_unsafe(double k, double n, double p) noexcept nogil:
    _legacy_deprecation("bdtr", k, n)
    if isnan(n) or isinf(n):
        return NAN
    else:
        return cephes_bdtr_wrap(k, <int>n, p)

# 定义一个内联函数，用于计算不安全的二项分布补充累积分布函数
cdef inline double bdtrc_unsafe(double k, double n, double p) noexcept nogil:
    _legacy_deprecation("bdtrc", k, n)
    if isnan(n) or isinf(n):
        return NAN
    else:
        return cephes_bdtrc_wrap(k, <int>n, p)

# 定义一个内联函数，用于计算不安全的二项分布反函数
cdef inline double bdtri_unsafe(double k, double n, double p) noexcept nogil:
    _legacy_deprecation("bdtri", k, n)
    if isnan(n) or isinf(n):
        return NAN
    else:
        return cephes_bdtri_wrap(k, <int>n, p)
# 计算特定函数的值，如果 n 是 NaN，则直接返回 n
cdef inline double expn_unsafe(double n, double x) noexcept nogil:
    if isnan(n):
        return n
    # 执行特定函数的类型检查和转换，此处检查并没有使用返回值
    _legacy_cast_check("expn", n, 0)
    # 调用特定函数 cephes_expn_wrap 计算函数 expn(n, x) 的值并返回
    return cephes_expn_wrap(<int>n, x)

# 计算负二项分布的累积分布函数的值，如果 k 或 n 是 NaN，则返回 NaN
cdef inline double nbdtrc_unsafe(double k, double n, double p) noexcept nogil:
    if isnan(k) or isnan(n):
        return NAN
    # 执行特定函数的类型检查和转换，此处检查并没有使用返回值
    _legacy_cast_check("nbdtrc", k, n)
    # 调用特定函数 cephes_nbdtrc_wrap 计算函数 nbdtrc(k, n, p) 的值并返回
    return cephes_nbdtrc_wrap(<int>k, <int>n, p)

# 计算负二项分布的分布函数的值，如果 k 或 n 是 NaN，则返回 NaN
cdef inline double nbdtr_unsafe(double k, double n, double p) noexcept nogil:
    if isnan(k) or isnan(n):
        return NAN
    # 执行特定函数的类型检查和转换，此处检查并没有使用返回值
    _legacy_cast_check("nbdtr", k, n)
    # 调用特定函数 cephes_nbdtr_wrap 计算函数 nbdtr(k, n, p) 的值并返回
    return cephes_nbdtr_wrap(<int>k, <int>n, p)

# 计算负二项分布的逆分布函数的值，如果 k 或 n 是 NaN，则返回 NaN
cdef inline double nbdtri_unsafe(double k, double n, double p) noexcept nogil:
    if isnan(k) or isnan(n):
        return NAN
    # 执行特定函数的类型检查和转换，此处检查并没有使用返回值
    _legacy_cast_check("nbdtri", k, n)
    # 调用特定函数 cephes_nbdtri_wrap 计算函数 nbdtri(k, n, p) 的值并返回
    return cephes_nbdtri_wrap(<int>k, <int>n, p)

# 计算 Poisson 分布的逆累积分布函数的值，如果 k 是 NaN，则返回 k
cdef inline double pdtri_unsafe(double k, double y) noexcept nogil:
    if isnan(k):
        return k
    # 执行特定函数的类型检查和转换，此处检查并没有使用返回值
    _legacy_cast_check("pdtri", k, 0)
    # 调用特定函数 cephes_pdtri_wrap 计算函数 pdtri(k, y) 的值并返回
    return cephes_pdtri_wrap(<int>k, y)

# 计算修正的 Bessel 函数的值，如果 n 是 NaN，则返回 n
cdef inline double kn_unsafe(double n, double x) noexcept nogil:
    if isnan(n):
        return n
    # 执行特定函数的类型检查和转换，此处检查并没有使用返回值
    _legacy_cast_check("kn", n, 0)
    # 调用特定函数 special_cyl_bessel_k_int 计算函数 kn(n, x) 的值并返回
    return special_cyl_bessel_k_int(<int>n, x)

# 计算第二类修正的 Bessel 函数的值，如果 n 是 NaN，则返回 n
cdef inline double yn_unsafe(double n, double x) noexcept nogil:
    if isnan(n):
        return n
    # 执行特定函数的类型检查和转换，此处检查并没有使用返回值
    _legacy_cast_check("yn", n, 0)
    # 调用特定函数 cephes_yn_wrap 计算函数 yn(n, x) 的值并返回
    return cephes_yn_wrap(<int>n, x)

# 计算 Smirnov 分布函数的值，如果 n 是 NaN，则返回 n
cdef inline double smirnov_unsafe(double n, double e) noexcept nogil:
    if isnan(n):
        return n
    # 执行特定函数的类型检查和转换，此处检查并没有使用返回值
    _legacy_cast_check("smirnov", n, 0)
    # 调用特定函数 cephes_smirnov_wrap 计算函数 smirnov(n, e) 的值并返回
    return cephes_smirnov_wrap(<int>n, e)

# 计算 Smirnov-Cramer-Von Mises 分布函数的值，如果 n 是 NaN，则返回 n
cdef inline double smirnovc_unsafe(double n, double e) noexcept nogil:
    if isnan(n):
        return n
    # 执行特定函数的类型检查和转换，此处检查并没有使用返回值
    _legacy_cast_check("smirnovc", n, 0)
    # 调用特定函数 cephes_smirnovc_wrap 计算函数 smirnovc(n, e) 的值并返回
    return cephes_smirnovc_wrap(<int>n, e)

# 计算 Smirnov 积分函数的值，如果 n 是 NaN，则返回 n
cdef inline double smirnovp_unsafe(double n, double e) noexcept nogil:
    if isnan(n):
        return n
    # 执行特定函数的类型检查和转换，此处检查并没有使用返回值
    _legacy_cast_check("smirnovp", n, 0)
    # 调用特定函数 cephes_smirnovp_wrap 计算函数 smirnovp(n, e) 的值并返回
    return cephes_smirnovp_wrap(<int>n, e)

# 计算 Smirnov 逆积分函数的值，如果 n 是 NaN，则返回 n
cdef inline double smirnovi_unsafe(double n, double p) noexcept nogil:
    if isnan(n):
        return n
    # 执行特定函数的类型检查和转换，此处检查并没有使用返回值
    _legacy_cast_check("smirnovi", n, 0)
    # 调用特定函数 cephes_smirnovi_wrap 计算函数 smirnovi(n, p) 的值并返回
    return cephes_smirnovi_wrap(<int>n, p)

# 计算 Smirnov-Cramer-Von Mises 逆积分函数的值，如果 n 是 NaN，则返回 n
cdef inline double smirnovci_unsafe(double n, double p) noexcept nogil:
    if isnan(n):
        return n
    # 执行特定函数的类型检查和转换，此处检查并没有使用返回值
    _legacy_cast_check("smirnovci", n, 0)
    # 调用特定函数 cephes_smirnovci_wrap 计算函数 smirnovci(n, p) 的值并返回
    return cephes_smirnovci_wrap(<int>n, p)
```