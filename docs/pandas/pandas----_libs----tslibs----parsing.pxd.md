# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\parsing.pxd`

```
# 从 cpython.datetime 模块中导入 datetime 类型，这是一个特定的 CPython 扩展
# cimport 用于在 Cython 中导入 CPython 的 C 扩展模块
from cpython.datetime cimport datetime

# 从 numpy 模块中导入 int64_t 类型，这是一个特定的 C 类型
# cimport 用于在 Cython 中导入 CPython 的 C 扩展模块
from numpy cimport int64_t

# 从 pandas._libs.tslibs.np_datetime 模块中导入 NPY_DATETIMEUNIT 常量
# cimport 用于在 Cython 中导入 CPython 的 C 扩展模块
from pandas._libs.tslibs.np_datetime cimport NPY_DATETIMEUNIT

# 声明一个 Cython 函数，返回类型为 str，名称为 get_rule_month，参数为 str 类型的 source
cpdef str get_rule_month(str source)

# 声明一个 Cython 函数，返回类型为 str，名称为 quarter_to_myear，参数为 int 类型的 year、quarter 和 str 类型的 freq
cpdef quarter_to_myear(int year, int quarter, str freq)

# 声明一个 Cython 函数，返回类型为 datetime，名称为 parse_datetime_string，
# 参数为 str 类型的 date_string、bint 类型的 dayfirst 和 yearfirst、NPY_DATETIMEUNIT 指针类型的 out_bestunit 和 int64_t 指针类型的 nanos
cdef datetime parse_datetime_string(
    str date_string,
    bint dayfirst,
    bint yearfirst,
    NPY_DATETIMEUNIT* out_bestunit,
    int64_t* nanos,
)
```