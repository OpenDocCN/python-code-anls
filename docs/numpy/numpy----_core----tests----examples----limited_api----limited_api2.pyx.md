# `.\numpy\numpy\_core\tests\examples\limited_api\limited_api2.pyx`

```py
#cython: language_level=3
# 设置 Cython 的语言级别为 3，以支持较新的语言特性和语法

"""
Make sure cython can compile in limited API mode (see meson.build)
"""
# 引入 NumPy 的头文件中的数组对象声明
cdef extern from "numpy/arrayobject.h":
    pass
# 引入 NumPy 的标量数组头文件中的声明
cdef extern from "numpy/arrayscalars.h":
    pass
```