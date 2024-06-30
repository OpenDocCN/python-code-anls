# `D:\src\scipysrc\scipy\scipy\ndimage\tests\__init__.py`

```
# 导入未来的 annotations 特性，使得类型提示中的类型可以是其本身，而不是字符串形式
from __future__ import annotations
# 导入 NumPy 库并将其命名为 np
import numpy as np

# 定义包含整数数据类型的列表
integer_types: list[type] = [
    np.int8, np.uint8, np.int16, np.uint16,
    np.int32, np.uint32, np.int64, np.uint64]

# 定义包含浮点数数据类型的列表
float_types: list[type] = [np.float32, np.float64]

# 定义包含复数数据类型的列表
complex_types: list[type] = [np.complex64, np.complex128]

# 定义包含所有数值数据类型的列表，即整数、浮点数和复数类型的组合
types: list[type] = integer_types + float_types
```