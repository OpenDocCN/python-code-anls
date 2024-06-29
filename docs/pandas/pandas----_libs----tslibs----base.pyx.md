# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\base.pyx`

```
"""
We define base classes that will be inherited by Timestamp, Timedelta, etc
in order to allow for fast isinstance checks without circular dependency issues.

This is analogous to core.dtypes.generic.
"""

# 导入需要的 datetime 模块，使用 cimport 是为了从 Cython 导入 C 扩展模块
from cpython.datetime cimport datetime

# 定义 Cython 类 ABCTimestamp，继承自 datetime 类
cdef class ABCTimestamp(datetime):
    pass
```