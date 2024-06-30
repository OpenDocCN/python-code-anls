# `D:\src\scipysrc\scipy\scipy\special\tests\_cython_examples\extending.pyx`

```
#!/usr/bin/env python3
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
# 以上为 Cython 编译器的指令，指定 Python 代码的编译选项

# 从 scipy.special.cython_special 模块中导入 beta 和 gamma 函数
from scipy.special.cython_special cimport beta, gamma

# 声明一个 Cython 编译器可识别的 CPython 函数，返回类型为 double
cpdef double cy_beta(double a, double b):
    # 调用 Cython 版本的 beta 函数，并返回结果
    return beta(a, b)

# 声明一个 Cython 编译器可识别的 CPython 函数，返回类型为 double complex
cpdef double complex cy_gamma(double complex z):
    # 调用 Cython 版本的 gamma 函数，并返回结果
    return gamma(z)
```