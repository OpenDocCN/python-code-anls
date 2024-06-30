# `D:\src\scipysrc\scikit-learn\sklearn\utils\_fast_dict.pxd`

```
# 导入必要的Cython库和类型定义
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
"""
使用 C++ 的 map 容器来实现快速的类似字典的行为，其中键为整数，值为浮点数。
"""

# 导入 C++ 的 map 作为 cpp_map
from libcpp.map cimport map as cpp_map

# 从 _typedefs 中导入必要的数据类型 float64_t 和 intp_t
from ._typedefs cimport float64_t, intp_t


###############################################################################
# Python 中使用的对象定义

# 使用 cdef 声明一个 Cython 类
cdef class IntFloatDict:
    # 声明一个私有的 C++ map 对象，键为 intp_t 类型，值为 float64_t 类型
    cdef cpp_map[intp_t, float64_t] my_map
    
    # 声明一个私有方法 _to_arrays，用于将键和值转换为 Cython 数组
    cdef _to_arrays(self, intp_t [:] keys, float64_t [:] values)
```