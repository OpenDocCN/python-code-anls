# `D:\src\scipysrc\pandas\pandas\_libs\lib.pxd`

```
# 导入NumPy库中的ndarray类型，cimport语句用于从Cython中导入特定的C变量或函数
from numpy cimport ndarray

# 定义Cython函数c_is_list_like，接受两个参数：object和bint，返回一个bint类型的值，如果出现异常则返回-1
cdef bint c_is_list_like(object, bint) except -1

# 定义Cython函数eq_NA_compat，接受两个参数：一个ndarray对象arr（元素类型为object）、一个object类型的key
# 返回一个ndarray对象，用于处理NA值的兼容性
cpdef ndarray eq_NA_compat(ndarray[object] arr, object key)
```