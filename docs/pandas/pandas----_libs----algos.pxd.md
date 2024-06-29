# `D:\src\scipysrc\pandas\pandas\_libs\algos.pxd`

```
# 导入 pandas 库的 C 语言接口中的特定数据类型
from pandas._libs.dtypes cimport (
    numeric_object_t,  # 定义数值对象类型，可能包含 NaN
    numeric_t,         # 定义数值类型
)

# 定义一个 C 语言函数原型，用于在没有 GIL 的情况下计算数组中第 k 小的数值
cdef numeric_t kth_smallest_c(numeric_t* arr, Py_ssize_t k, Py_ssize_t n) noexcept nogil

# 定义一个枚举类型，表示在处理平局时的不同策略
cdef enum TiebreakEnumType:
    TIEBREAK_AVERAGE            # 平局时取平均值
    TIEBREAK_MIN,               # 平局时取最小值
    TIEBREAK_MAX                # 平局时取最大值
    TIEBREAK_FIRST              # 平局时取首次出现的值
    TIEBREAK_FIRST_DESCENDING   # 平局时取首次出现的值（降序）
    TIEBREAK_DENSE              # 平局时取连续的密集排名

# 定义一个 C 语言函数，用于获取排名时用于填充 NaN 值的数值对象类型
cdef numeric_object_t get_rank_nan_fill_val(
    bint rank_nans_highest,     # 表示 NaN 的排名是否最高
    numeric_object_t val,       # 待处理的数值对象
    bint is_datetimelike=*     # 表示是否为类日期时间的数据类型
)
```