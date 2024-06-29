# `.\numpy\numpy\typing\tests\data\pass\numerictypes.py`

```
# 导入 NumPy 库，命名为 np
import numpy as np

# 检查 np.float64 是否为 np.int64 或 np.float64 类型
np.isdtype(np.float64, (np.int64, np.float64))

# 检查 np.int64 是否为有符号整数类型
np.isdtype(np.int64, "signed integer")

# 检查 "S1" 是否为 np.bytes_ 类型的子类型
np.issubdtype("S1", np.bytes_)

# 检查 np.float64 是否为 np.float32 类型的子类型
np.issubdtype(np.float64, np.float32)

# np.ScalarType 是所有标量类型的父类

# np.ScalarType[0] 是 NumPy 中的标量类型列表的第一个元素
# np.ScalarType[3] 是 NumPy 中的标量类型列表的第四个元素
# np.ScalarType[8] 是 NumPy 中的标量类型列表的第九个元素
# np.ScalarType[10] 是 NumPy 中的标量类型列表的第十一个元素

# np.typecodes["Character"] 返回 NumPy 中字符类型的代码
# np.typecodes["Complex"] 返回 NumPy 中复数类型的代码
# np.typecodes["All"] 返回 NumPy 中所有已定义类型的代码
```