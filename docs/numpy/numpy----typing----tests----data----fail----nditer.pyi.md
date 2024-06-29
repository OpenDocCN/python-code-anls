# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\nditer.pyi`

```
import numpy as np  # 导入 NumPy 库

class Test(np.nditer): ...  # E: 无法继承自 final 类

np.nditer([0, 1], flags=["test"])  # E: 类型不兼容，flags 应为整数或者空列表

np.nditer([0, 1], op_flags=[["test"]])  # E: 类型不兼容，op_flags 应为整数

np.nditer([0, 1], itershape=(1.0,))  # E: 类型不兼容，itershape 应为整数元组

np.nditer([0, 1], buffersize=1.0)  # E: 类型不兼容，buffersize 应为整数
```