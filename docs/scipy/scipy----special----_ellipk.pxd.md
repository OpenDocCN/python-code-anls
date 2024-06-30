# `D:\src\scipysrc\scipy\scipy\special\_ellipk.pxd`

```
# 从特定的头文件 "special_wrappers.h" 中导入一个 C 函数声明，使用 nogil 以确保在 Cython 中没有全局解释器锁
cdef extern from "special_wrappers.h" nogil:
    # 声明一个名为 cephes_ellpk_wrap 的 C 函数，接受一个 double 类型参数 x，返回一个 double 类型结果
    double cephes_ellpk_wrap(double x)

# 定义一个内联函数 ellipk，接受一个 double 类型参数 m，声明不会抛出异常，并且在 Cython 中不会使用全局解释器锁 (nogil)
cdef inline double ellipk(double m) noexcept nogil:
    # 调用之前声明的 cephes_ellpk_wrap 函数，传入参数 1.0 - m，并将其返回值作为 ellipk 函数的返回值
    return cephes_ellpk_wrap(1.0 - m)
```