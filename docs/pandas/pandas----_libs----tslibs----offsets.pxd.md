# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\offsets.pxd`

```
# 导入 cimport 命令用于从 Cython 中导入 int64_t 类型
from numpy cimport int64_t

# 声明一个 Cython 编译时可用的函数，将 Python 对象转换为偏移量
# obj: 要转换的对象
# is_period: 是否为周期性偏移量的标志，True 或 False，默认为 *
cpdef to_offset(object obj, bint is_period=*)

# 声明一个 Cython 编译时可用的函数，判断给定的对象是否为偏移量对象
# obj: 要检查的对象
cdef bint is_offset_object(object obj)

# 声明一个 Cython 编译时可用的函数，判断给定的对象是否为 tick 对象
# obj: 要检查的对象
cdef bint is_tick_object(object obj)

# 定义一个 Cython 编译时的类 BaseOffset
cdef class BaseOffset:
    # 以下是只读属性定义部分
    cdef readonly:
        # 定义一个 int64_t 类型的成员变量 n
        int64_t n
        # 定义一个 bint 类型的成员变量 normalize，用于表示是否进行标准化
        bint normalize
        # 定义一个字典类型的成员变量 _cache，用作缓存
        dict _cache
```