# `D:\src\scipysrc\pandas\pandas\_libs\util.pxd`

```
# 导入 Cython 中的 numpy 模块，命名为 cnp
cimport numpy as cnp
# 从 libc.stdint 中导入以下整数类型的最大值和最小值常量
from libc.stdint cimport (
    INT8_MAX,    # 8位有符号整数的最大值
    INT8_MIN,    # 8位有符号整数的最小值
    INT16_MAX,   # 16位有符号整数的最大值
    INT16_MIN,   # 16位有符号整数的最小值
    INT32_MAX,   # 32位有符号整数的最大值
    INT32_MIN,   # 32位有符号整数的最小值
    INT64_MAX,   # 64位有符号整数的最大值
    INT64_MIN,   # 64位有符号整数的最小值
    UINT8_MAX,   # 8位无符号整数的最大值
    UINT16_MAX,  # 16位无符号整数的最大值
    UINT32_MAX,  # 32位无符号整数的最大值
    UINT64_MAX,  # 64位无符号整数的最大值
)
# 从 pandas._libs.tslibs.util 中导入所有内容
from pandas._libs.tslibs.util cimport *
```