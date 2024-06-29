# `D:\src\scipysrc\pandas\pandas\testing.py`

```
"""
Public testing utility functions.
"""

# 从 pandas._testing 模块中导入以下函数，用于测试断言
from pandas._testing import (
    assert_extension_array_equal,  # 导入断言扩展数组相等的函数
    assert_frame_equal,             # 导入断言数据框相等的函数
    assert_index_equal,             # 导入断言索引相等的函数
    assert_series_equal,            # 导入断言序列相等的函数
)

# 指定模块中可导出的函数名列表
__all__ = [
    "assert_extension_array_equal",  # 可导出的函数名：assert_extension_array_equal
    "assert_frame_equal",            # 可导出的函数名：assert_frame_equal
    "assert_series_equal",           # 可导出的函数名：assert_series_equal
    "assert_index_equal",            # 可导出的函数名：assert_index_equal
]
```