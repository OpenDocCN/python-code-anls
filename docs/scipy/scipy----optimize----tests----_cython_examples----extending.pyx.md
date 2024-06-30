# `D:\src\scipysrc\scipy\scipy\optimize\tests\_cython_examples\extending.pyx`

```
#!/usr/bin/env python3
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
"""
Taken from docstring for scipy.optimize.cython_optimize module.
"""

# 从 scipy.optimize.cython_optimize 模块导入 brentq 函数
from scipy.optimize.cython_optimize cimport brentq

# 从 libc 中导入 math 模块
from libc cimport math

# 一个包含额外参数的字典
myargs = {'C0': 1.0, 'C1': 0.7}

# 搜索区间的下限和上限
XLO, XHI = 0.5, 1.0

# 其他求解器参数：容差、相对容差和最大迭代次数
XTOL, RTOL, MITR = 1e-3, 1e-3, 10

# 用户定义的结构体，用于存储额外参数
ctypedef struct test_params:
    double C0
    double C1

# 用户定义的回调函数
cdef double f(double x, void *args) noexcept:
    # 将 args 转换为 test_params 结构体指针
    cdef test_params *myargs = <test_params *> args
    return myargs.C0 - math.exp(-(x - myargs.C1))

# Cython 封装函数
cdef double brentq_wrapper_example(dict args, double xa, double xb,
                                    double xtol, double rtol, int mitr):
    # Cython 自动将字典转换为结构体 test_params
    cdef test_params myargs = args
    return brentq(
        f, xa, xb, <test_params *> &myargs, xtol, rtol, mitr, NULL)

# Python 函数
def brentq_example(args=myargs, xa=XLO, xb=XHI, xtol=XTOL, rtol=RTOL,
                    mitr=MITR):
    '''Calls Cython wrapper from Python.'''
    return brentq_wrapper_example(args, xa, xb, xtol, rtol, mitr)
```