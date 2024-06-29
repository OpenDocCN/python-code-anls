# `D:\src\scipysrc\pandas\pandas\_libs\dtypes.pxd`

```
"""
Common location for shared fused types
"""

# 导入需要的模块和类型定义
from numpy cimport (
    float32_t,
    float64_t,
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)

# 定义一个融合类型 `numeric_t`，包括所有的数值类型，但不包括复数类型
ctypedef fused numeric_t:
    int8_t
    int16_t
    int32_t
    int64_t
    uint8_t
    uint16_t
    uint32_t
    uint64_t
    float32_t
    float64_t

# 定义一个融合类型 `numeric_object_t`，包括所有的数值类型和对象类型，但不包括复数类型
ctypedef fused numeric_object_t:
    numeric_t
    object
```