# `D:\src\scipysrc\numpy\numpy\typing\tests\data\reveal\ufunc_config.pyi`

```
"""Typing tests for `_core._ufunc_config`."""

# 导入 sys 模块，用于访问 Python 解释器相关信息
import sys
# 导入 Any 和 Protocol 类型，用于类型提示
from typing import Any, Protocol
# 导入 Callable 类型，用于类型提示
from collections.abc import Callable

# 导入 numpy 库，并使用 np 别名
import numpy as np

# 如果 Python 版本大于等于 3.11，则使用标准库中的 assert_type 函数
if sys.version_info >= (3, 11):
    from typing import assert_type
# 否则，使用 typing_extensions 中的 assert_type 函数
else:
    from typing_extensions import assert_type

# 声明一个空函数 func，参数为 a（str 类型）和 b（int 类型），返回 None
def func(a: str, b: int) -> None: ...

# 定义一个类 Write，包含一个方法 write，接收参数 value（str 类型），返回 None
class Write:
    def write(self, value: str) -> None: ...

# 定义一个 Protocol SupportsWrite，包含一个方法 write，接收参数 s（str 类型），返回 object 类型
class SupportsWrite(Protocol):
    def write(self, s: str, /) -> object: ...

# 调用 assert_type 函数，验证 np.seterr(all=None) 的返回类型为 np._core._ufunc_config._ErrDict
assert_type(np.seterr(all=None), np._core._ufunc_config._ErrDict)
assert_type(np.seterr(divide="ignore"), np._core._ufunc_config._ErrDict)
assert_type(np.seterr(over="warn"), np._core._ufunc_config._ErrDict)
assert_type(np.seterr(under="call"), np._core._ufunc_config._ErrDict)
assert_type(np.seterr(invalid="raise"), np._core._ufunc_config._ErrDict)
assert_type(np.geterr(), np._core._ufunc_config._ErrDict)

# 调用 assert_type 函数，验证 np.setbufsize(4096) 的返回类型为 int
assert_type(np.setbufsize(4096), int)
assert_type(np.getbufsize(), int)

# 调用 assert_type 函数，验证 np.seterrcall(func) 的返回类型为 Callable[[str, int], Any] | None | SupportsWrite
assert_type(np.seterrcall(func), Callable[[str, int], Any] | None | SupportsWrite)
assert_type(np.seterrcall(Write()), Callable[[str, int], Any] | None | SupportsWrite)
assert_type(np.geterrcall(), Callable[[str, int], Any] | None | SupportsWrite)

# 调用 assert_type 函数，验证 np.errstate(call=func, all="call") 的返回类型为 np.errstate
assert_type(np.errstate(call=func, all="call"), np.errstate)
assert_type(np.errstate(call=Write(), divide="log", over="log"), np.errstate)
```