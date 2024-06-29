# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\arrayprint.pyi`

```py
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

# 定义一个名为 AR 的 numpy 数组，其元素类型为 np.float64
AR: npt.NDArray[np.float64]
# 定义一个接受任意类型参数并返回字符串的函数签名
func1: Callable[[Any], str]
# 定义一个接受 numpy.integer 类型参数并返回字符串的函数签名
func2: Callable[[np.integer[Any]], str]

# 调用 np.array2string 函数，以下是各种参数传递方式的错误提示注释

# E: 意外的关键字参数 style
np.array2string(AR, style=None)
# E: 类型不兼容
np.array2string(AR, legacy="1.14")
# E: 类型不兼容
np.array2string(AR, sign="*")
# E: 类型不兼容
np.array2string(AR, floatmode="default")
# E: 类型不兼容
np.array2string(AR, formatter={"A": func1})
# E: 类型不兼容
np.array2string(AR, formatter={"float": func2})
```