# `.\numpy\numpy\typing\tests\data\pass\ma.py`

```py
# 导入必要的模块和类型
from typing import Any
# 导入 NumPy 库及其子模块，用于科学计算和数组处理
import numpy as np
import numpy.ma

# 创建一个带有掩码的 NumPy 掩码数组
# 数据数组为 [1.5, 2, 3]，掩码数组为 [True, False, True]
m : np.ma.MaskedArray[Any, np.dtype[np.float64]] = np.ma.masked_array([1.5, 2, 3], mask=[True, False, True])
```