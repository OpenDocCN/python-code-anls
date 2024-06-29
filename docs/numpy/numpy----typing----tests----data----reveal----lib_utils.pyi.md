# `D:\src\scipysrc\numpy\numpy\typing\tests\data\reveal\lib_utils.pyi`

```
import sys
from io import StringIO

import numpy as np
import numpy.typing as npt
import numpy.lib.array_utils as array_utils

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type

# 定义一个 numpy 数组类型 AR，元素类型为 np.float64
AR: npt.NDArray[np.float64]
# 定义一个字典类型 AR_DICT，键为字符串，值为 np.float64 类型的 numpy 数组
AR_DICT: dict[str, npt.NDArray[np.float64]]
# 定义一个 StringIO 对象 FILE，用于文件操作
FILE: StringIO

# 定义一个名为 func 的函数，参数为整数 a，返回布尔值，但函数体未实现
def func(a: int) -> bool: ...

# 对 array_utils.byte_bounds 函数调用 assert_type，期望返回类型是 tuple[int, int]
assert_type(array_utils.byte_bounds(AR), tuple[int, int])
# 对 array_utils.byte_bounds 函数以 np.float64 类型的实例为参数调用 assert_type
assert_type(array_utils.byte_bounds(np.float64()), tuple[int, int])

# 调用 np.info 函数，传入整数 1 和 FILE 对象作为输出参数，期望返回 None 类型
assert_type(np.info(1, output=FILE), None)
```