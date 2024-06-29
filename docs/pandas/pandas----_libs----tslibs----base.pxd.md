# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\base.pxd`

```
# 导入 CPython 中的 datetime 模块中的 datetime 类
from cpython.datetime cimport datetime

# 定义一个 C 扩展类 ABCTimestamp，继承自 datetime 类
cdef class ABCTimestamp(datetime):
    pass
```