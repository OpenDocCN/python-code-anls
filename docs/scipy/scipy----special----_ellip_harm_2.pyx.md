# `D:\src\scipysrc\scipy\scipy\special\_ellip_harm_2.pyx`

```
# 导入 Python.h 中的 PyCapsule_New 函数，用于创建一个 Python 对象的“胶囊”，以封装一个 C 指针
cdef extern from "Python.h":
    object PyCapsule_New(void *pointer, char *name, void *destructor)

# 从 libc.math 中导入 sqrt 和 fabs 函数
# 从 libc.stdlib 中导入 free 函数
from libc.math cimport sqrt, fabs
from libc.stdlib cimport free

# 从 numpy 导入 nan 常量
from numpy import nan

# 从 scipy._lib._ccallback 导入 LowLevelCallable 类
# 从 ._ellip_harm 模块中导入 ellip_harmonic, ellip_harm_eval, lame_coefficients 函数
from scipy._lib._ccallback import LowLevelCallable
from ._ellip_harm cimport ellip_harmonic, ellip_harm_eval, lame_coefficients

# 定义一个 C 结构体 _ellip_data_t，包含双精度浮点数指针 eval，以及 h2、k2、n、p 四个整型变量
ctypedef struct _ellip_data_t:
    double *eval
    double h2, k2
    int n, p

# 定义一个 C 函数 _F_integrand，接受一个双精度浮点数 t 和一个指向 void 的指针 user_data，且是无异常抛出、无 GIL 的函数
cdef double _F_integrand(double t, void *user_data) noexcept nogil:
    # 将 user_data 解释为 _ellip_data_t 结构体指针
    cdef _ellip_data_t *data = <_ellip_data_t *>user_data
    cdef double h2, k2, t2, i, result
    cdef int n, p
    cdef double * eval

    # 计算 t 的平方
    t2 = t*t
    # 从 data 结构体获取 h2、k2、n、p 和 eval
    h2 = data[0].h2
    k2 = data[0].k2
    n = data[0].n
    p = data[0].p
    eval = data[0].eval

    # 调用 ellip_harm_eval 函数计算椭球谐函数
    i = ellip_harm_eval( h2, k2, n, p, 1/t, eval, 1, 1)
    # 计算并返回积分被求函数的结果
    result = 1/(i*i*sqrt(1 - t2*k2)*sqrt(1 - t2*h2))
    return result

# 定义函数 _F_integrand1，接受一个双精度浮点数 t 和一个指向 void 的指针 user_data，且是无异常抛出、无 GIL 的函数
cdef double _F_integrand1(double t, void *user_data) noexcept nogil:
    cdef _ellip_data_t *data = <_ellip_data_t *>user_data
    cdef double h2, k2, i, h, result
    cdef int n, p
    cdef double * eval

    # 从 data 结构体获取 h2、k2、n、p 和 eval
    h2 = data[0].h2
    k2 = data[0].k2
    n = data[0].n
    p = data[0].p
    eval = data[0].eval

    # 计算 h 和 k 的平方根
    h = sqrt(h2)
    k = sqrt(k2)
    # 调用 ellip_harm_eval 函数计算椭球谐函数
    i = ellip_harm_eval( h2, k2, n, p, t, eval, 1, 1)
    # 计算并返回积分被求函数的结果
    result = i*i/sqrt((t + h)*(t + k))
    return result

# 定义函数 _F_integrand2，接受一个双精度浮点数 t 和一个指向 void 的指针 user_data，且是无异常抛出、无 GIL 的函数
cdef double _F_integrand2(double t, void *user_data) noexcept nogil:
    cdef _ellip_data_t *data = <_ellip_data_t *>user_data
    cdef double h2, k2, t2, i, h, result
    cdef int n, p
    cdef double * eval

    # 计算 t 的平方
    t2 = t*t
    # 从 data 结构体获取 h2、k2、n、p 和 eval
    h2 = data[0].h2
    k2 = data[0].k2
    n = data[0].n
    p = data[0].p
    eval = data[0].eval

    # 计算 h 和 k 的平方根
    h = sqrt(h2)
    k = sqrt(k2)
    # 调用 ellip_harm_eval 函数计算椭球谐函数
    i = ellip_harm_eval( h2, k2, n, p, t, eval, 1, 1)
    # 计算并返回积分被求函数的结果
    result = t2*i*i/sqrt((t + h)*(t + k))
    return result

# 定义函数 _F_integrand3，接受一个双精度浮点数 t 和一个指向 void 的指针 user_data，且是无异常抛出、无 GIL 的函数
cdef double _F_integrand3(double t, void *user_data) noexcept nogil:
    cdef _ellip_data_t *data = <_ellip_data_t *>user_data
    cdef double h2, k2, t2, i, h, result
    cdef int n, p
    cdef double * eval

    # 计算 t 的平方
    t2 = t*t
    # 从 data 结构体获取 h2、k2、n、p 和 eval
    h2 = data[0].h2
    k2 = data[0].k2
    n = data[0].n
    p = data[0].p
    eval = data[0].eval

    # 计算 h 的平方根
    h = sqrt(h2)
    # 调用 ellip_harm_eval 函数计算椭球谐函数
    i = ellip_harm_eval( h2, k2, n, p, t, eval, 1, 1)
    # 计算并返回积分被求函数的结果
    result = i*i/sqrt((t + h)*(k2 - t2))
    return result

# 定义函数 _F_integrand4，接受一个双精度浮点数 t 和一个指向 void 的指针 user_data，且是无异常抛出、无 GIL 的函数
cdef double _F_integrand4(double t, void *user_data) noexcept nogil:
    cdef _ellip_data_t *data = <_ellip_data_t *>user_data
    cdef double h2, k2, t2, i, h, result
    cdef int n, p
    cdef double *eval

    # 计算 t 的平方
    t2 = t*t
    # 从 data 结构体获取 h2、k2、n、p 和 eval
    h2 = data[0].h2
    k2 = data[0].k2
    n = data[0].n
    p = data[0].p
    eval = data[0].eval

    # 计算 h 的平方根
    h = sqrt(h2)
    # 调用 ellip_harm_eval 函数计算椭球谐函数
    i = ellip_harm_eval( h2, k2, n, p, t, eval, 1, 1)
    # 计算并返回积分被求函数的结果
    result = i*i*t2/sqrt((t + h)*(k2 - t2))
    return result

# 定义 _ellipsoid 函数，接受双精度浮点数 h2、k2 和整型 n、p，以及双精度浮点数 s 作为参数
def _ellipsoid(double h2, double k2, int n, int p, double s):
    # 导入 scipy.special._ellip_harm_2 模块，并从 scipy.integrate 导入 quad 函数
    import scipy.special._ellip_h
    # 将变量 h2 赋值给 data 对象的属性 h2
    data.h2 = h2
    # 将变量 k2 赋值给 data 对象的属性 k2
    data.k2 = k2
    # 将变量 n 赋值给 data 对象的属性 n
    data.n = n
    # 将变量 p 赋值给 data 对象的属性 p
    data.p = p
    # 将内置函数 eval 赋值给 data 对象的属性 eval
    data.eval = eval

    # 声明并初始化两个 C 语言风格的双精度浮点变量 res 和 err
    cdef double res, err

    # 尝试执行以下代码块
    try:
        # 创建一个 Python Capsule 对象，封装 data 结构体的指针，无自定义析构函数
        capsule = PyCapsule_New(<void*>&data, NULL, NULL)
        # 使用 Cython 生成的 LowLevelCallable 对象调用 _F_integrand 函数进行数值积分
        # 在区间 [0, 1/s] 上计算，指定绝对误差 epsabs=1e-300 和相对误差 epsrel=1e-15
        res, err = quad(LowLevelCallable.from_cython(mod, "_F_integrand", capsule), 0, 1/s,
                                                     epsabs=1e-300, epsrel=1e-15)
    finally:
        # 释放先前分配的缓冲区指针
        free(bufferp)

    # 如果相对误差 err 大于阈值 1e-10*fabs(res) + 1e-290，则返回 NaN
    if err > 1e-10*fabs(res) + 1e-290:
        return nan
    
    # 计算最终结果 res，乘以 (2*n + 1) 和 ellip_harmonic 函数的返回值
    res = res*(2*n + 1)*ellip_harmonic( h2, k2, n, p, s, 1, 1)
    
    # 返回计算得到的结果 res
    return res
# 导入必要的模块和函数
import scipy.special._ellip_harm_2 as mod
from scipy.integrate import quad

# 定义 C 结构体 _ellip_data_t
cdef _ellip_data_t data

# 定义 void 指针 bufferp
cdef void *bufferp

# 调用 lame_coefficients 函数计算评估值
eval = lame_coefficients(h2, k2, n, p, &bufferp, 1, 1)
if not eval:
    return nan

# 填充数据结构体 data
data.h2 = h2
data.k2 = k2
data.n = n
data.p = p
data.eval = eval

# 定义用于积分计算的变量和误差变量
cdef double res, res1, res2, res3, err, err1, err2, err3

# 计算 h 和 k 的平方根
h = sqrt(h2)
k = sqrt(k2)

try:
    # 创建包含 C 结构体数据的 PyCapsule
    capsule = PyCapsule_New(<void*>&data, NULL, NULL)

    # 设置积分权重变量 wvar
    wvar = (-0.5, -0.5)

    # 执行第一个积分，计算 res 和 err
    res, err = quad(LowLevelCallable.from_cython(mod, "_F_integrand1", capsule), h, k,
                    epsabs=1e-300, epsrel=1e-15, weight="alg", wvar=wvar)

    # 执行第二个积分，计算 res1 和 err1
    res1, err1 = quad(LowLevelCallable.from_cython(mod, "_F_integrand2", capsule), h, k,
                      epsabs=1e-300, epsrel=1e-15, weight="alg", wvar=wvar)

    # 更新积分权重变量 wvar
    wvar = (0, -0.5)

    # 执行第三个积分，计算 res2 和 err2
    res2, err2 = quad(LowLevelCallable.from_cython(mod, "_F_integrand3", capsule), 0, h,
                      epsabs=1e-300, epsrel=1e-15, weight="alg", wvar=wvar)

    # 执行第四个积分，计算 res3 和 err3
    res3, err3 = quad(LowLevelCallable.from_cython(mod, "_F_integrand4", capsule), 0, h,
                      epsabs=1e-300, epsrel=1e-15, weight="alg", wvar=wvar)

finally:
    # 释放 bufferp 内存
    free(bufferp)

# 计算误差项
error = 8*(res2*err1 + err2*res1 + res*err3 + res3*err)
# 计算结果
result = 8*(res1*res2 - res*res3)

# 检查误差是否超出阈值，若超出则返回 nan
if error > 10e-8*fabs(result):
    return nan

# 返回计算结果
return result
```