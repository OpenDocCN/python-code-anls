# `D:\src\scipysrc\pandas\pandas\_libs\arrays.pxd`

```
# 导入 numpy 库中的 cimport 模块，并引入其中的 ndarray 类型
from numpy cimport ndarray

# 定义一个 Cython 的 cdef 类 NDArrayBacked
cdef class NDArrayBacked:
    cdef:
        # 定义私有成员变量 _ndarray，类型为 readonly 的 numpy ndarray
        readonly ndarray _ndarray
        # 定义私有成员变量 _dtype，类型为 readonly 的 Python 对象
        readonly object _dtype

    # 声明一个公共方法，可以由 Python 调用，接受一个 ndarray 参数并返回 NDArrayBacked 对象
    cpdef NDArrayBacked _from_backing_data(self, ndarray values)
    
    # 声明一个特殊方法 __setstate__，可以由 Python 调用，用于设置对象的状态
    cpdef __setstate__(self, state)
```