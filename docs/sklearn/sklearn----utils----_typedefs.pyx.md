# `D:\src\scipysrc\scikit-learn\sklearn\utils\_typedefs.pyx`

```
# _typedefs is a declaration only module
#
# The functions implemented here are for testing purpose only.

# 导入 NumPy 库，用于数组操作
import numpy as np

# 定义一个融合类型 fused testing_type_t，包括多种数据类型的定义
ctypedef fused testing_type_t:
    float32_t
    float64_t
    int8_t
    int32_t
    int64_t
    intp_t
    uint8_t
    uint32_t
    uint64_t

# 定义函数 testing_make_array_from_typed_val，接受一个 testing_type_t 类型的参数 val
def testing_make_array_from_typed_val(testing_type_t val):
    # 声明一个指向 val 的视图 val_view，类型为 testing_type_t 的数组
    cdef testing_type_t[:] val_view = <testing_type_t[:1]>&val
    # 将 val_view 转换为 NumPy 数组并返回
    return np.asarray(val_view)
```