# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\stride_tricks.pyi`

```py
import numpy as np
import numpy.typing as npt

# 导入 numpy 库以及 numpy.typing 模块，npt.NDArray 表示 numpy 数组的类型提示

AR_f8: npt.NDArray[np.float64]
# 定义一个类型为 np.float64 的 numpy 数组 AR_f8

np.lib.stride_tricks.as_strided(AR_f8, shape=8)
# 使用 numpy 的 stride_tricks 模块中的 as_strided 函数，尝试使用 shape 参数，但未找到匹配的重载变体

np.lib.stride_tricks.as_strided(AR_f8, strides=8)
# 使用 numpy 的 stride_tricks 模块中的 as_strided 函数，尝试使用 strides 参数，但未找到匹配的重载变体

np.lib.stride_tricks.sliding_window_view(AR_f8, axis=(1,))
# 使用 numpy 的 stride_tricks 模块中的 sliding_window_view 函数，尝试使用 axis 参数，但未找到匹配的重载变体
```