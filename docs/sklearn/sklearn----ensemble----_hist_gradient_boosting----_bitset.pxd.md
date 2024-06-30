# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_hist_gradient_boosting\_bitset.pxd`

```
# 导入来自common模块的X_BINNED_DTYPE_C类型
from .common cimport X_BINNED_DTYPE_C
# 导入来自common模块的BITSET_DTYPE_C类型
from .common cimport BITSET_DTYPE_C
# 导入来自common模块的BITSET_INNER_DTYPE_C类型
from .common cimport BITSET_INNER_DTYPE_C
# 导入来自common模块的X_DTYPE_C类型
from .common cimport X_DTYPE_C

# 定义Cython函数，初始化位集(bitset)，并且不抛出异常、无需GIL
cdef void init_bitset(BITSET_DTYPE_C bitset) noexcept nogil

# 定义Cython函数，设置位集(bitset)中的某个值为指定的X_BINNED_DTYPE_C类型的值，并且不抛出异常、无需GIL
cdef void set_bitset(BITSET_DTYPE_C bitset, X_BINNED_DTYPE_C val) noexcept nogil

# 定义Cython函数，检查位集(bitset)中是否存在指定的X_BINNED_DTYPE_C类型的值，并且不抛出异常、无需GIL
cdef unsigned char in_bitset(BITSET_DTYPE_C bitset, X_BINNED_DTYPE_C val) noexcept nogil

# 定义Cython函数，使用内存视图(const BITSET_INNER_DTYPE_C[:])检查位集(bitset)中是否存在指定的X_BINNED_DTYPE_C类型的值，并且不抛出异常、无需GIL
cpdef unsigned char in_bitset_memoryview(const BITSET_INNER_DTYPE_C[:] bitset,
                                         X_BINNED_DTYPE_C val) noexcept nogil

# 定义Cython函数，使用二维内存视图(const BITSET_INNER_DTYPE_C[:, :])检查位集(bitset)的指定行(row)中是否存在指定的X_BINNED_DTYPE_C类型的值，并且不抛出异常、无需GIL
cdef unsigned char in_bitset_2d_memoryview(
    const BITSET_INNER_DTYPE_C[:, :] bitset,
    X_BINNED_DTYPE_C val,
    unsigned int row) noexcept nogil
```