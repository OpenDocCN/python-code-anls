# `.\numpy\numpy\typing\tests\data\reveal\arrayprint.pyi`

```py
import sys
import contextlib
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy._core.arrayprint import _FormatOptions

# 如果 Python 版本大于等于 3.11，则使用标准库 typing 中的 assert_type
if sys.version_info >= (3, 11):
    from typing import assert_type
# 否则，使用 typing_extensions 中的 assert_type
else:
    from typing_extensions import assert_type

# 定义一个 numpy 的多维数组 AR，元素类型为 np.int64
AR: npt.NDArray[np.int64]
# 定义两个可调用对象，接受 np.floating 和 np.integer 类型的参数，返回字符串
func_float: Callable[[np.floating[Any]], str]
func_int: Callable[[np.integer[Any]], str]

# 使用 assert_type 确保 np.get_printoptions() 返回的类型是 _FormatOptions
assert_type(np.get_printoptions(), _FormatOptions)

# 使用 assert_type 确保 np.array2string(AR, formatter={'float_kind': func_float, 'int_kind': func_int}) 返回的类型是 str
assert_type(
    np.array2string(AR, formatter={'float_kind': func_float, 'int_kind': func_int}),
    str,
)

# 使用 assert_type 确保 np.format_float_scientific(1.0) 返回的类型是 str
assert_type(np.format_float_scientific(1.0), str)

# 使用 assert_type 确保 np.format_float_positional(1) 返回的类型是 str
assert_type(np.format_float_positional(1), str)

# 使用 assert_type 确保 np.array_repr(AR) 返回的类型是 str
assert_type(np.array_repr(AR), str)

# 使用 assert_type 确保 np.array_str(AR) 返回的类型是 str
assert_type(np.array_str(AR), str)

# 使用 assert_type 确保 np.printoptions() 返回的类型是 contextlib._GeneratorContextManager[_FormatOptions]
assert_type(np.printoptions(), contextlib._GeneratorContextManager[_FormatOptions])

# 进入 np.printoptions() 上下文管理器，使用 with 语句
with np.printoptions() as dct:
    # 使用 assert_type 确保上下文管理器 dct 的类型是 _FormatOptions
    assert_type(dct, _FormatOptions)
```