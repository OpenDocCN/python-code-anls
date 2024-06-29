# `D:\src\scipysrc\pandas\pandas\_libs\missing.pxd`

```
# 导入必要的类型和对象声明
from numpy cimport (
    ndarray,       # 导入 ndarray 类型，表示多维数组
    uint8_t,       # 导入 uint8_t 类型，表示无符号8位整数
)

# 声明一个 Cython 函数，用于检查两个对象是否匹配，可以处理缺失值的情况
cpdef bint is_matching_na(object left, object right, bint nan_matches_none=*)
    # 实现省略，未提供具体代码内容

# 声明一个 Cython 函数，用于检查两个元组对象是否不相等，处理缺失值情况
cpdef bint check_na_tuples_nonequal(object left, object right)
    # 实现省略，未提供具体代码内容

# 声明一个 Cython 函数，用于检查对象是否为 null（空值）
cpdef bint checknull(object val)
    # 实现省略，未提供具体代码内容

# 声明一个 Cython 函数，用于检查 ndarray 中的每个元素是否为缺失值
cpdef ndarray[uint8_t] isnaobj(ndarray arr)
    # 实现省略，未提供具体代码内容

# 声明一个 Cython 函数，用于检查 datetime64 类型的对象是否为 null
cdef bint is_null_datetime64(v)
    # 实现省略，未提供具体代码内容

# 声明一个 Cython 函数，用于检查 timedelta64 类型的对象是否为 null
cdef bint is_null_timedelta64(v)
    # 实现省略，未提供具体代码内容

# 声明一个 Cython 函数，用于检查对象是否为 null，支持 NAT 和 NA 类型
cdef bint checknull_with_nat_and_na(object obj)
    # 实现省略，未提供具体代码内容

# 声明一个 Cython 的类 C_NAType，用于表示特定的 NA 类型
cdef class C_NAType:
    pass
    # 类定义为空，表示没有额外的属性或方法

# 声明一个 Cython 的对象 C_NA，用于表示 NA 类型的实例
cdef C_NAType C_NA
    # 实现省略，未提供具体代码内容
```