# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\array_pad.pyi`

```
import numpy as np  # 导入 NumPy 库，用于科学计算
import numpy.typing as npt  # 导入 NumPy 的类型提示模块，用于类型注解

AR_i8: npt.NDArray[np.int64]  # 声明一个变量 AR_i8，类型为 int64 的 NumPy 数组

np.pad(AR_i8, 2, mode="bob")  # 对数组 AR_i8 进行填充操作，填充宽度为 2，使用非法填充模式 "bob"
                            # 此处会报错，因为 "bob" 不是合法的填充模式
```