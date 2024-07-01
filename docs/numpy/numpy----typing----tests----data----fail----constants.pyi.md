# `.\numpy\numpy\typing\tests\data\fail\constants.pyi`

```py
# 导入 NumPy 库，用于科学计算和数组操作
import numpy as np

# 尝试修改 NumPy 的 little_endian 属性，但是 little_endian 是不可修改的，因此会引发错误
np.little_endian = np.little_endian  # E: Cannot assign to final
```