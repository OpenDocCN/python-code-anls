# `.\numpy\numpy\typing\tests\data\pass\lib_utils.py`

```
# 引入未来版本的注解支持
from __future__ import annotations

# 从 io 模块中引入 StringIO 类
from io import StringIO

# 引入 numpy 库，并将其命名为 np
import numpy as np

# 引入 numpy 库中的 array_utils 模块
import numpy.lib.array_utils as array_utils

# 创建一个字符串缓冲区对象 FILE
FILE = StringIO()

# 创建一个包含 0 到 9 的浮点数数组 AR，数据类型为 np.float64
AR = np.arange(10, dtype=np.float64)


# 定义一个函数 func，参数 a 是整数类型，返回值是布尔类型
def func(a: int) -> bool:
    return True


# 调用 array_utils 模块的 byte_bounds 函数，传入 AR 数组作为参数
array_utils.byte_bounds(AR)

# 调用 array_utils 模块的 byte_bounds 函数，传入 np.float64() 对象作为参数
array_utils.byte_bounds(np.float64())

# 调用 numpy 的 info 函数，打印关于数字 1 的信息，输出重定向到 FILE 对象
np.info(1, output=FILE)
```