# `D:\src\scipysrc\numpy\numpy\typing\tests\data\reveal\ufunclike.pyi`

```
# 导入 sys 模块，用于系统相关操作
import sys
# 导入 Any 类型，表示可以是任何类型的变量
from typing import Any

# 导入 numpy 库，并将其命名为 np
import numpy as np
# 导入 numpy.typing 模块，用于类型提示
import numpy.typing as npt

# 如果 Python 版本大于等于 3.11，则导入 typing 模块的 assert_type 函数
if sys.version_info >= (3, 11):
    from typing import assert_type
# 否则，导入 typing_extensions 模块的 assert_type 函数
else:
    from typing_extensions import assert_type

# 定义多种列表类型的变量，用于类型提示
AR_LIKE_b: list[bool]
AR_LIKE_u: list[np.uint32]
AR_LIKE_i: list[int]
AR_LIKE_f: list[float]
AR_LIKE_O: list[np.object_]

# 定义一个字符串类型的 NumPy 数组变量 AR_U
AR_U: npt.NDArray[np.str_]

# 使用 assert_type 函数验证 np.fix 函数的返回类型，期望返回一个浮点数类型的 NumPy 数组
assert_type(np.fix(AR_LIKE_b), npt.NDArray[np.floating[Any]])
assert_type(np.fix(AR_LIKE_u), npt.NDArray[np.floating[Any]])
assert_type(np.fix(AR_LIKE_i), npt.NDArray[np.floating[Any]])
assert_type(np.fix(AR_LIKE_f), npt.NDArray[np.floating[Any]])
# 验证带有输出参数的 np.fix 函数，期望返回一个字符串类型的 NumPy 数组
assert_type(np.fix(AR_LIKE_O), npt.NDArray[np.object_])
assert_type(np.fix(AR_LIKE_f, out=AR_U), npt.NDArray[np.str_])

# 使用 assert_type 函数验证 np.isposinf 函数的返回类型，期望返回一个布尔类型的 NumPy 数组
assert_type(np.isposinf(AR_LIKE_b), npt.NDArray[np.bool])
assert_type(np.isposinf(AR_LIKE_u), npt.NDArray[np.bool])
assert_type(np.isposinf(AR_LIKE_i), npt.NDArray[np.bool])
assert_type(np.isposinf(AR_LIKE_f), npt.NDArray[np.bool])
# 验证带有输出参数的 np.isposinf 函数，期望返回一个字符串类型的 NumPy 数组
assert_type(np.isposinf(AR_LIKE_f, out=AR_U), npt.NDArray[np.str_])

# 使用 assert_type 函数验证 np.isneginf 函数的返回类型，期望返回一个布尔类型的 NumPy 数组
assert_type(np.isneginf(AR_LIKE_b), npt.NDArray[np.bool])
assert_type(np.isneginf(AR_LIKE_u), npt.NDArray[np.bool])
assert_type(np.isneginf(AR_LIKE_i), npt.NDArray[np.bool])
assert_type(np.isneginf(AR_LIKE_f), npt.NDArray[np.bool])
# 验证带有输出参数的 np.isneginf 函数，期望返回一个字符串类型的 NumPy 数组
assert_type(np.isneginf(AR_LIKE_f, out=AR_U), npt.NDArray[np.str_])
```