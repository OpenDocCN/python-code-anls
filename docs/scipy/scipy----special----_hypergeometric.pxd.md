# `D:\src\scipysrc\scipy\scipy\special\_hypergeometric.pxd`

```
# 从libc.math中导入一些函数和常量，这些函数和常量在C语言中定义在math.h中
from libc.math cimport fabs, exp, floor, isnan, M_PI, NAN, INFINITY

# 导入Cython模块，用于编写Cython代码
import cython

# 从当前包中的特定C文件中导入sf_error模块
from . cimport sf_error

# 从'special_wrappers.h'头文件中声明外部C函数，使用 nogil 来确保GIL（全局解释器锁）释放
cdef extern from 'special_wrappers.h':
    double hypU_wrap(double, double, double) nogil  # 声明hypU_wrap函数原型
    double cephes_poch_wrap(double x, double m) nogil  # 声明cephes_poch_wrap函数原型


# 使用cython.cdivision(True)开启Cython中的除法检查
@cython.cdivision(True)
# 定义一个内联函数hyperu，返回double类型，不抛出异常，且在无GIL状态下执行
cdef inline double hyperu(double a, double b, double x) noexcept nogil:
    # 检查参数a、b、x是否为NaN，如果是则返回NaN
    if isnan(a) or isnan(b) or isnan(x):
        return NAN

    # 如果x小于0.0，则报错并返回NaN
    if x < 0.0:
        sf_error.error("hyperu", sf_error.DOMAIN, NULL)
        return NAN

    # 如果x等于0.0
    if x == 0.0:
        # 如果b大于1.0，则报告奇异情况并返回无穷大
        if b > 1.0:
            # 参考DLMF 13.2.16-18节
            sf_error.error("hyperu", sf_error.SINGULAR, NULL)
            return INFINITY
        else:
            # 否则根据DLMF 13.2.14-15和13.2.19-21节计算并返回结果
            return cephes_poch_wrap(1.0 - b + a, -a)

    # 对于其他情况，调用外部声明的hypU_wrap函数来计算并返回结果
    return hypU_wrap(a, b, x)
```