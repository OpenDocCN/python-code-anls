# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\shape_base.pyi`

```
import numpy as np  # 导入 numpy 库

class DTypeLike:  # 定义类 DTypeLike
    dtype: np.dtype[np.int_]  # 类属性 dtype，指定为 np.int_ 类型的 numpy 数据类型

dtype_like: DTypeLike  # 定义变量 dtype_like，类型为 DTypeLike 类的实例

np.expand_dims(dtype_like, (5, 10))  # 使用 numpy 函数 expand_dims 对变量 dtype_like 进行维度扩展，传入参数 (5, 10)，出现错误 "E: No overload variant"
```