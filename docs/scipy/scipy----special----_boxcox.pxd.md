# `D:\src\scipysrc\scipy\scipy\special\_boxcox.pxd`

```
# 导入一组数学函数，这些函数来自于 libc 库中的 math 模块，包括 log, log1p, expm1, exp, fabs, copysign
from libc.math cimport log, log1p, expm1, exp, fabs, copysign


# 定义一个内联函数 boxcox，接受两个 double 类型参数 x 和 lmbda，并且在无锁状态下运行，不会抛出异常
cdef inline double boxcox(double x, double lmbda) noexcept nogil:
    # 如果 lmbda 的绝对值小于 1e-19，则返回 log(x)
    if fabs(lmbda) < 1e-19:
        return log(x)
    # 如果 lmbda * log(x) 的结果小于 709.78，则返回 expm1(lmbda * log(x)) / lmbda
    elif lmbda * log(x) < 709.78:
        return expm1(lmbda * log(x)) / lmbda
    # 否则返回 copysign(1., lmbda) * exp(lmbda * log(x) - log(fabs(lmbda))) - 1 / lmbda
    else:
        return copysign(1., lmbda) * exp(lmbda * log(x) - log(fabs(lmbda))) - 1 / lmbda


# 定义一个内联函数 boxcox1p，接受两个 double 类型参数 x 和 lmbda，并且在无锁状态下运行，不会抛出异常
cdef inline double boxcox1p(double x, double lmbda) noexcept nogil:
    # 计算 log1p(x) 的值并赋给 lgx
    cdef double lgx = log1p(x)
    # 如果 lmbda 的绝对值小于 1e-19 或者 (lgx 的绝对值小于 1e-289 且 lmbda 的绝对值小于 1e273)，则返回 lgx
    if fabs(lmbda) < 1e-19 or (fabs(lgx) < 1e-289 and fabs(lmbda) < 1e273):
        return lgx
    # 如果 lmbda * lgx 的结果小于 709.78，则返回 expm1(lmbda * lgx) / lmbda
    elif lmbda * lgx < 709.78:
        return expm1(lmbda * lgx) / lmbda
    # 否则返回 copysign(1., lmbda) * exp(lmbda * lgx - log(fabs(lmbda))) - 1 / lmbda
    else:
        return copysign(1., lmbda) * exp(lmbda * lgx - log(fabs(lmbda))) - 1 / lmbda


# 定义一个内联函数 inv_boxcox，接受两个 double 类型参数 x 和 lmbda，并且在无锁状态下运行，不会抛出异常
cdef inline double inv_boxcox(double x, double lmbda) noexcept nogil:
    # 如果 lmbda 等于 0，则返回 exp(x)
    if lmbda == 0:
        return exp(x)
    # 如果 lmbda * x 的结果小于 1.79e308，则返回 exp(log1p(lmbda * x) / lmbda)
    elif lmbda * x < 1.79e308:
        return exp(log1p(lmbda * x) / lmbda)
    # 否则返回 exp((log(copysign(1., lmbda) * (x + 1 / lmbda)) + log(fabs(lmbda))) / lmbda)
    else:
        return exp((log(copysign(1., lmbda) * (x + 1 / lmbda)) + log(fabs(lmbda))) / lmbda)


# 定义一个内联函数 inv_boxcox1p，接受两个 double 类型参数 x 和 lmbda，并且在无锁状态下运行，不会抛出异常
cdef inline double inv_boxcox1p(double x, double lmbda) noexcept nogil:
    # 如果 lmbda 等于 0，则返回 expm1(x)
    if lmbda == 0:
        return expm1(x)
    # 如果 lmbda * x 的绝对值小于 1e-154，则返回 x
    elif fabs(lmbda * x) < 1e-154:
        return x
    # 如果 lmbda * x 的结果小于 1.79e308，则返回 expm1(log1p(lmbda * x) / lmbda)
    elif lmbda * x < 1.79e308:
        return expm1(log1p(lmbda * x) / lmbda)
    # 否则返回 expm1((log(copysign(1., lmbda) * (x + 1 / lmbda)) + log(fabs(lmbda))) / lmbda)
    else:
        return expm1((log(copysign(1., lmbda) * (x + 1 / lmbda)) + log(fabs(lmbda))) / lmbda)
```