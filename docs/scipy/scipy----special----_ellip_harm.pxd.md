# `D:\src\scipysrc\scipy\scipy\special\_ellip_harm.pxd`

```
# 导入Cython模块，用于优化Python代码的性能
import cython

# 从当前包中导入sf_error模块
from . cimport sf_error

# 从C标准库中导入数学函数和常量
from libc.math cimport sqrt, fabs, pow, NAN
# 从C标准库中导入内存分配和释放函数
from libc.stdlib cimport malloc, free

# 从外部头文件"lapack_defs.h"中声明一个外部C函数
cdef extern from "lapack_defs.h":
    # 定义C语言类型CBLAS_INT，实际类型由头文件定义
    ctypedef int CBLAS_INT
    # lapack中的特定函数声明，使用Cython的nogil标记表示不需要GIL
    void c_dstevr(char *jobz, char *range, CBLAS_INT *n, double *d, double *e,
                  double *vl, double *vu, CBLAS_INT *il, CBLAS_INT *iu, double *abstol,
                  CBLAS_INT *m, double *w, double *z, CBLAS_INT *ldz, CBLAS_INT *isuppz,
                  double *work, CBLAS_INT *lwork, CBLAS_INT *iwork, CBLAS_INT *liwork,
                  CBLAS_INT *info) nogil

# 使用Cython装饰器定义函数的优化选项
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
# 使用Cython定义内联函数，返回一个指向double的指针
cdef inline double* lame_coefficients(double h2, double k2, int n, int p,
                                      void **bufferp, double signm,
                                      double signn) noexcept nogil:
    # 初始化缓冲区指针为空，以便安全地释放内存
    bufferp[0] = NULL

    # 检查n的有效性，若无效则报错并返回空指针
    if n < 0:
        sf_error.error("ellip_harm", sf_error.ARG, "invalid value for n")
        return NULL

    # 检查p的有效性，若无效则报错并返回空指针
    if p < 1 or p > 2*n + 1:
        sf_error.error("ellip_harm", sf_error.ARG, "invalid value for p")
        return NULL

    # 检查signm和signn的有效性，若无效则报错并返回空指针
    if fabs(signm) != 1 or fabs(signn) != 1:
        sf_error.error("ellip_harm", sf_error.ARG, "invalid signm or signn")
        return NULL

    # 定义一些双精度浮点数变量，用于计算椭圆函数的系数
    cdef double s2, alpha, beta, gamma, lamba_romain, pp, psi, t1, tol, vl, vu
    # 定义变量 r, tp, j, size, i, info, lwork, liwork, c, iu，其中包括整型和字符类型
    cdef CBLAS_INT r, tp, j, size, i, info, lwork, liwork, c, iu
    cdef Py_UCS4 t

    # 计算 r 的值，这里使用整数除法
    r = n/2
    # 设置 alpha, beta, gamma 的值
    alpha = h2
    beta = k2 - h2
    gamma = alpha - beta

    # 根据不同条件确定变量 t, tp, size 的值
    if p - 1 < r + 1:
        t, tp, size = 'K', p, r + 1
    elif p - 1 < (n - r) + (r + 1):
        t, tp, size = 'L', p - (r + 1), n - r
    elif p - 1 < (n - r) + (n - r) + (r + 1):
        t, tp, size = 'M', p - (n - r) - (r + 1), n - r
    elif p - 1 < 2*n + 1:
        t, tp, size = 'N', p - (n - r) - (n - r) - (r + 1), r
    else:
        # 如果条件不符合，输出错误信息并返回 NULL
        sf_error.error("ellip_harm", sf_error.ARG, "invalid condition on `p - 1`")
        return NULL

    # 根据 size 计算 lwork 和 liwork 的大小
    lwork = 60*size
    liwork = 30*size
    tol = 0.0
    vl = 0
    vu = 0

    # 分配内存空间，并检查分配是否成功
    cdef void *buffer = malloc((sizeof(double)*(7*size + lwork))
                               + (sizeof(CBLAS_INT)*(2*size + liwork)))
    bufferp[0] = buffer
    if not buffer:
        # 如果内存分配失败，输出错误信息并返回 NULL
        sf_error.error("ellip_harm", sf_error.NO_RESULT, "failed to allocate memory")
        return NULL

    # 将分配的内存区域分别映射到不同类型的指针
    cdef double *g = <double *>buffer
    cdef double *d = g + size
    cdef double *f = d + size
    cdef double *ss =  f + size
    cdef double *w =  ss + size
    cdef double *dd = w + size
    cdef double *eigv = dd + size
    cdef double *work = eigv + size

    # 将 iwork 和 isuppz 映射到不同类型的指针
    cdef CBLAS_INT *iwork = <CBLAS_INT *>(work + lwork)
    cdef CBLAS_INT *isuppz = iwork + liwork

    # 根据 t 的值执行不同的计算
    if t == 'K':
        # 根据公式计算数组 g, d, f 中的值
        for j in range(0, r + 1):
           g[j] = (-(2*j + 2)*(2*j + 1)*beta)
           if n%2:
               f[j] = (-alpha*(2*(r- (j + 1)) + 2)*(2*((j + 1) + r) + 1))
               d[j] = ((2*r + 1)*(2*r + 2) - 4*j*j)*alpha + (2*j + 1)*(2*j + 1)*beta
           else:
               f[j] = (-alpha*(2*(r - (j + 1)) + 2)*(2*(r + (j + 1)) - 1))
               d[j] = 2*r*(2*r + 1)*alpha - 4*j*j*gamma
        
    elif t == 'L':
        # 根据公式计算数组 g, d, f 中的值
        for j in range(0, n - r):
           g[j] = (-(2*j + 2)*(2*j + 3)*beta)
           if n%2:
               f[j] = (-alpha*(2*(r- (j + 1)) + 2)*(2*((j + 1) + r) + 1))
               d[j] = (2*r + 1)*(2*r + 2)*alpha - (2*j + 1)*(2*j + 1)*gamma
           else:
               f[j] = (-alpha*(2*(r - (j + 1)))*(2*(r+(j + 1)) + 1))
               d[j] = (2*r*(2*r + 1) - (2*j + 1)*(2*j + 1))*alpha + (2*j + 2)*(2*j + 2)*beta
        
    elif t == 'M':
        # 根据公式计算数组 g, d, f 中的值
        for j in range(0, n - r):
           g[j] = (-(2*j + 2)*(2*j + 1)*beta)
           if n%2:
               f[j] = (-alpha*(2*(r - (j + 1)) + 2)*(2*((j + 1) + r) + 1))
               d[j] = ((2*r + 1)*(2*r + 2) - (2*j + 1)*(2*j + 1))*alpha + 4*j*j*beta
           else:
               f[j] = (-alpha*(2*(r - (j + 1)))*(2*(r+(j + 1)) + 1))
               d[j] = 2*r*(2*r + 1)*alpha - (2*j + 1)*(2*j + 1)*gamma    
    # 如果 t 的值为 'N'，则执行以下代码块
    elif t == 'N':
        # 遍历 j 从 0 到 r-1
        for j in range(0, r):
            # 计算 g[j] 的值
            g[j] = (-(2*j + 2)*(2*j + 3)*beta)
            # 如果 n 除以 2 的余数为真（即 n 是奇数）
            if n % 2:
                # 计算 f[j] 的值（n 为奇数时）
                f[j] = (-alpha*(2*(r- (j + 1)))*(2*((j + 1) + r) + 3))
                # 计算 d[j] 的值（n 为奇数时）
                d[j] = (2*r + 1)*(2*r + 2)*alpha - (2*j + 2)*(2*j + 2)*gamma
            else:
                # 计算 f[j] 的值（n 为偶数时）
                f[j] = (-alpha*(2*(r - (j + 1)))*(2*(r+(j + 1)) + 1))
                # 计算 d[j] 的值（n 为偶数时）
                d[j] = 2*r*(2*r + 1)*alpha - (2*j + 2)*(2*j + 2)*alpha + (2*j + 1)*(2*j + 1)*beta

    # 对于 i 从 0 到 size-1 的范围进行迭代
    for i in range(0, size):
        # 如果 i 等于 0
        if i == 0:
            # 设置 ss[i] 的值为 1
            ss[i] = 1
        else:
            # 计算 ss[i] 的值（非首个元素）
            ss[i] = sqrt(g[i - 1] / f[i - 1]) * ss[i - 1]

    # 对于 i 从 0 到 size-2 的范围进行迭代
    for i in range(0, size-1):
        # 计算 dd[i] 的值
        dd[i] = g[i] * ss[i] / ss[i + 1]

    # 调用外部函数 c_dstevr 进行计算
    c_dstevr("V", "I", &size, d, dd, &vl, &vu, &tp, &tp, &tol, &c, w, eigv,
             &size, isuppz, work, &lwork, iwork, &liwork, &info)

    # 如果调用 c_dstevr 函数返回的 info 不等于 0
    if info != 0:
        # 报错并返回 NULL
        sf_error.error("ellip_harm", sf_error.NO_RESULT, "failed to allocate memory")
        return NULL

    # 对于 i 从 0 到 size-1 的范围进行迭代
    for i in range(0, size):
        # 将 eigv[i] 除以 ss[i]
        eigv[i] /= ss[i]

    # 对于 i 从 0 到 size-1 的范围进行迭代
    for i in range(0, size):
        # 将 eigv[i] 除以 eigv[size - 1] 除以 -h2 的 size - 1 次方
        eigv[i] = eigv[i] / (eigv[size - 1] / pow(-h2, size - 1))

    # 返回 eigv 数组作为函数的结果
    return eigv
# 设置 Cython 编译器选项，禁用循环边界检查和溢出检查，启用 C 语言风格的除法
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
# 定义一个内联函数 ellip_harm_eval，用于计算椭圆函数的值
cdef inline double ellip_harm_eval(double h2, double k2, int n, int p,
                                   double s, double *eigv, double signm,
                                   double signn) noexcept nogil:
    # 声明变量
    cdef int size, tp, r, j
    cdef double s2, pp, lambda_romain, psi
    # 计算 s 的平方
    s2 = s*s
    # 计算 n 的一半
    r = n/2
    # 根据 p - 1 的不同值选择不同的分支计算 size 和 psi
    if p - 1 < r + 1:
        size, psi = r + 1, pow(s, n - 2*r)
    elif p - 1 < (n - r) + (r + 1):
        size, psi = n - r, pow(s, 1 - n + 2*r)*signm*sqrt(fabs(s2 - h2))
    elif p - 1 < (n - r) + (n - r) + (r + 1):
        size, psi = n - r, pow(s, 1 - n + 2*r)*signn*sqrt(fabs(s2 - k2))
    elif p - 1 < 2*n + 1:
        size, psi = r, pow(s,  n - 2*r)*signm*signn*sqrt(fabs((s2 - h2)*(s2 - k2)))
    else:
        # 如果 p - 1 不在预期范围内，抛出错误并返回 NaN
        sf_error.error("ellip_harm", sf_error.ARG, "invalid condition on `p - 1`")
        return NAN

    # 计算 lambda_romain
    lambda_romain = 1.0 - <double>s2/<double>h2
    # 初始化 pp 为 eigv 中最后一个元素
    pp = eigv[size - 1]
    # 从倒数第二个元素开始向前计算 pp
    for j in range(size - 2, -1, -1):
        pp = pp*lambda_romain + eigv[j]

    # 最终乘以 psi
    pp = pp*psi
    # 返回计算结果 pp
    return pp


# 定义一个内联函数 ellip_harmonic，调用 ellip_harm_eval 计算椭圆函数的值
cdef inline double ellip_harmonic(double h2, double k2, int n, int p, double s,
                                  double signm, double signn) noexcept nogil:
    # 声明变量
    cdef double result
    cdef double *eigv
    cdef void *bufferp
    # 调用 lame_coefficients 函数获取 eigv 数组和 bufferp 指针
    eigv = lame_coefficients(h2, k2, n, p, &bufferp, signm, signn)
    # 如果 eigv 为空指针，则释放 bufferp 并返回 NaN
    if not eigv:
        free(bufferp)
        return NAN
    # 调用 ellip_harm_eval 计算椭圆函数的值
    result = ellip_harm_eval(h2, k2, n, p, s, eigv, signm, signn)
    # 释放 bufferp
    free(bufferp)
    # 返回计算结果 result
    return result
```