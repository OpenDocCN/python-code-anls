# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\period.pxd`

```
# 导入 `int64_t` 类型的整数定义，这是由 NumPy 提供的 C 扩展类型
from numpy cimport int64_t

# 导入 `.np_datetime` 模块中的 `npy_datetimestruct` 结构体，用于处理日期时间数据

# 定义一个 Cython 函数，检查给定对象是否是时间周期对象，并返回一个布尔值
cdef bint is_period_object(object obj)

# 定义一个 Cython 函数，计算给定日期时间结构体 `npy_datetimestruct` 的周期序数，
# 使用指定的频率 `freq`，并且声明此函数是无异常、无全局解锁且无全局解锁的（noexcept nogil）
cdef int64_t get_period_ordinal(npy_datetimestruct *dts, int freq) noexcept nogil
```