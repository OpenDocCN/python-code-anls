# `D:\src\scipysrc\scipy\scipy\special\_factorial.pxd`

```
# 从特定的头文件 "special_wrappers.h" 中引入一个外部函数声明，使用 nogil 语句块避免全局解释器锁
cdef extern from "special_wrappers.h" nogil:
    # 声明一个双精度浮点数返回值的函数 cephes_gamma_wrap，接受一个双精度浮点数参数 x
    double cephes_gamma_wrap(double x)


# 定义一个内联函数 _factorial，返回类型为 double，无异常抛出，且不需要全局解释器锁
cdef inline double _factorial(double n) noexcept nogil:
    # 如果 n 小于 0，则返回 0
    if n < 0:
        return 0
    else:
        # 否则，调用 cephes_gamma_wrap 函数计算 n+1 的伽马函数值，作为阶乘的近似值返回
        return cephes_gamma_wrap(n + 1)
```