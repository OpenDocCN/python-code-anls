# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\numerictypes.pyi`

```
# 导入 numpy 库，并将其重命名为 np
import numpy as np

# 使用 numpy 库中的函数 np.isdtype() 检查参数是否是指定数据类型 np.int64
np.isdtype(1, np.int64)  # E: incompatible type

# 使用 numpy 库中的函数 np.issubdtype() 检查参数是否是指定数据类型 np.int64 的子类型
np.issubdtype(1, np.int64)  # E: incompatible type
```