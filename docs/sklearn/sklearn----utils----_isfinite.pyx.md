# `D:\src\scipysrc\scikit-learn\sklearn\utils\_isfinite.pyx`

```
# Author: John Kirkham, Meekail Zain, Thomas Fan

# 导入必要的库函数和类型定义
from libc.math cimport isnan, isinf
from cython cimport floating

# 定义枚举类型 FiniteStatus，表示数据集的有限状态
cpdef enum FiniteStatus:
    all_finite = 0       # 所有元素都是有限的
    has_nan = 1          # 数据集包含 NaN (非数值)
    has_infinite = 2     # 数据集包含无穷大值

# 定义函数 cy_isfinite，检查给定的浮点数数组是否包含 NaN 或无穷大值
def cy_isfinite(floating[::1] a, bint allow_nan=False):
    cdef FiniteStatus result
    # 使用 nogil 块来确保在 Cython 中不会释放全局解释器锁，优化性能
    with nogil:
        result = _isfinite(a, allow_nan)
    return result

# 定义内部函数 _isfinite，根据参数 allow_nan 决定是否允许 NaN，检查浮点数数组的有限性
cdef inline FiniteStatus _isfinite(floating[::1] a, bint allow_nan) noexcept nogil:
    cdef floating* a_ptr = &a[0]         # 获取浮点数数组的指针
    cdef Py_ssize_t length = len(a)      # 获取浮点数数组的长度
    if allow_nan:
        return _isfinite_allow_nan(a_ptr, length)    # 如果允许 NaN，调用允许 NaN 的检查函数
    else:
        return _isfinite_disable_nan(a_ptr, length)  # 如果不允许 NaN，调用禁止 NaN 的检查函数

# 定义内部函数 _isfinite_allow_nan，检查浮点数数组是否包含无穷大值
cdef inline FiniteStatus _isfinite_allow_nan(floating* a_ptr,
                                             Py_ssize_t length) noexcept nogil:
    cdef Py_ssize_t i
    cdef floating v
    for i in range(length):
        v = a_ptr[i]                      # 获取数组中的浮点数
        if isinf(v):                      # 如果是无穷大值
            return FiniteStatus.has_infinite  # 返回包含无穷大值的状态
    return FiniteStatus.all_finite         # 如果没有无穷大值，则返回所有元素都有限的状态

# 定义内部函数 _isfinite_disable_nan，检查浮点数数组是否包含 NaN 或无穷大值
cdef inline FiniteStatus _isfinite_disable_nan(floating* a_ptr,
                                               Py_ssize_t length) noexcept nogil:
    cdef Py_ssize_t i
    cdef floating v
    for i in range(length):
        v = a_ptr[i]                      # 获取数组中的浮点数
        if isnan(v):                      # 如果是 NaN
            return FiniteStatus.has_nan   # 返回包含 NaN 的状态
        elif isinf(v):                    # 如果是无穷大值
            return FiniteStatus.has_infinite  # 返回包含无穷大值的状态
    return FiniteStatus.all_finite         # 如果没有 NaN 或无穷大值，则返回所有元素都有限的状态
```