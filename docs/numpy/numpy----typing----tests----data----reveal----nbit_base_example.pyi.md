# `D:\src\scipysrc\numpy\numpy\typing\tests\data\reveal\nbit_base_example.pyi`

```
import sys
from typing import TypeVar

import numpy as np
import numpy.typing as npt
from numpy._typing import _64Bit, _32Bit

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type

T1 = TypeVar("T1", bound=npt.NBitBase)  # 定义一个类型变量 T1，它必须是 numpy 的 NBitBase 的子类或实现类
T2 = TypeVar("T2", bound=npt.NBitBase)  # 定义一个类型变量 T2，它必须是 numpy 的 NBitBase 的子类或实现类

def add(a: np.floating[T1], b: np.integer[T2]) -> np.floating[T1 | T2]:
    # 定义一个函数 add，接收两个参数 a 和 b，分别为 numpy 浮点数类型和整数类型
    return a + b  # 返回 a 和 b 的和，结果类型为 np.floating[T1 | T2]，即 a 和 b 类型的联合

i8: np.int64  # 声明 i8 为 np.int64 类型
i4: np.int32  # 声明 i4 为 np.int32 类型
f8: np.float64  # 声明 f8 为 np.float64 类型
f4: np.float32  # 声明 f4 为 np.float32 类型

assert_type(add(f8, i8), np.float64)  # 断言 add 函数返回的类型为 np.float64
assert_type(add(f4, i8), np.floating[_32Bit | _64Bit])  # 断言 add 函数返回的类型为 np.floating[_32Bit | _64Bit]
assert_type(add(f8, i4), np.floating[_32Bit | _64Bit])  # 断言 add 函数返回的类型为 np.floating[_32Bit | _64Bit]
assert_type(add(f4, i4), np.float32)  # 断言 add 函数返回的类型为 np.float32
```