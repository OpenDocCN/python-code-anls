# `D:\src\scipysrc\numpy\numpy\typing\tests\data\reveal\memmap.pyi`

```py
import sys
from typing import Any

import numpy as np  # 导入 NumPy 库

if sys.version_info >= (3, 11):
    from typing import assert_type  # 如果 Python 版本大于等于 3.11，则使用标准 typing 模块中的 assert_type 函数
else:
    from typing_extensions import assert_type  # 否则，从 typing_extensions 模块中导入 assert_type 函数

memmap_obj: np.memmap[Any, np.dtype[np.str_]]  # 定义一个类型为 np.memmap 的变量 memmap_obj

assert_type(np.memmap.__array_priority__, float)  # 断言 np.memmap.__array_priority__ 的类型为 float
assert_type(memmap_obj.__array_priority__, float)  # 断言 memmap_obj.__array_priority__ 的类型为 float
assert_type(memmap_obj.filename, str | None)  # 断言 memmap_obj.filename 的类型为 str 或 None
assert_type(memmap_obj.offset, int)  # 断言 memmap_obj.offset 的类型为 int
assert_type(memmap_obj.mode, str)  # 断言 memmap_obj.mode 的类型为 str
assert_type(memmap_obj.flush(), None)  # 断言 memmap_obj.flush() 的返回类型为 None

assert_type(np.memmap("file.txt", offset=5), np.memmap[Any, np.dtype[np.uint8]])  # 断言 np.memmap("file.txt", offset=5) 的类型
assert_type(np.memmap(b"file.txt", dtype=np.float64, shape=(10, 3)), np.memmap[Any, np.dtype[np.float64]])  # 断言 np.memmap(b"file.txt", dtype=np.float64, shape=(10, 3)) 的类型
with open("file.txt", "rb") as f:
    assert_type(np.memmap(f, dtype=float, order="K"), np.memmap[Any, np.dtype[Any]])  # 断言 np.memmap(f, dtype=float, order="K") 的类型

assert_type(memmap_obj.__array_finalize__(object()), None)  # 断言 memmap_obj.__array_finalize__(object()) 的返回类型为 None
```