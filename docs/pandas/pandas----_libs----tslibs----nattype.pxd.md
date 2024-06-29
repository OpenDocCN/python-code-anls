# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\nattype.pxd`

```
# 导入 cpython.datetime 模块中的 datetime 类
# cimport 用于从 Cython 编写的扩展模块导入 C 类型
from cpython.datetime cimport datetime

# 从 numpy 模块中导入 int64_t 类型
# cimport 用于从 Cython 编写的扩展模块导入 C 类型
from numpy cimport int64_t

# 定义一个 int64_t 类型的变量 NPY_NAT
cdef int64_t NPY_NAT

# 定义一个空的 set 类型变量 c_nat_strings
cdef set c_nat_strings

# 定义一个名为 _NaT 的 Cython 类，继承自 datetime 类
cdef class _NaT(datetime):
    # 声明一个只读的 int64_t 类型成员变量 _value
    cdef readonly:
        int64_t _value

# 定义一个名为 c_NaT 的 _NaT 类的对象
cdef _NaT c_NaT

# 定义一个名为 checknull_with_nat 的 Cython 函数，接收一个对象参数 val
cdef bint checknull_with_nat(object val)

# 定义一个名为 is_dt64nat 的 Cython 函数，接收一个对象参数 val
cdef bint is_dt64nat(object val)

# 定义一个名为 is_td64nat 的 Cython 函数，接收一个对象参数 val
cdef bint is_td64nat(object val)
```