# `.\numpy\numpy\typing\tests\data\pass\array_like.py`

```py
# 导入未来的语法特性，用于支持类型注解中的 annotations
from __future__ import annotations

# 导入类型相关模块
from typing import Any

# 导入 NumPy 库
import numpy as np
# 导入 NumPy 中的类型定义
from numpy._typing import NDArray, ArrayLike, _SupportsArray

# 下面定义了多个变量，并注明它们的类型为 ArrayLike

# x1 是一个布尔类型的数组或类数组对象，初始值为 True
x1: ArrayLike = True
# x2 是一个整数类型的数组或类数组对象，初始值为 5
x2: ArrayLike = 5
# x3 是一个浮点数类型的数组或类数组对象，初始值为 1.0
x3: ArrayLike = 1.0
# x4 是一个复数类型的数组或类数组对象，初始值为 1 + 1j
x4: ArrayLike = 1 + 1j
# x5 是一个 int8 类型的数组或类数组对象，初始值为 1
x5: ArrayLike = np.int8(1)
# x6 是一个 float64 类型的数组或类数组对象，初始值为 1.0
x6: ArrayLike = np.float64(1)
# x7 是一个 complex128 类型的数组或类数组对象，初始值为 1
x7: ArrayLike = np.complex128(1)
# x8 是一个包含整数数组 [1, 2, 3] 的数组或类数组对象
x8: ArrayLike = np.array([1, 2, 3])
# x9 是一个包含整数列表 [1, 2, 3] 的数组或类数组对象
x9: ArrayLike = [1, 2, 3]
# x10 是一个包含整数元组 (1, 2, 3) 的数组或类数组对象
x10: ArrayLike = (1, 2, 3)
# x11 是一个字符串类型的数组或类数组对象，初始值为 "foo"
x11: ArrayLike = "foo"
# x12 是一个内存视图类型的数组或类数组对象，初始值为 b'foo'
x12: ArrayLike = memoryview(b'foo')

# 定义类 A
class A:
    # 类 A 定义了 __array__ 方法，返回一个包含浮点数数组的 NDArray
    def __array__(
        self, dtype: None | np.dtype[Any] = None
    ) -> NDArray[np.float64]:
        return np.array([1.0, 2.0, 3.0])

# x13 是类 A 的一个实例，它也被声明为数组或类数组对象
x13: ArrayLike = A()

# scalar 是一个支持数组的标量对象，其类型为 np.int64 的支持数组
scalar: _SupportsArray[np.dtype[np.int64]] = np.int64(1)
# 调用 scalar 对象的 __array__ 方法

scalar.__array__()

# array 是一个支持数组的对象，其类型为 np.int_ 的支持数组
array: _SupportsArray[np.dtype[np.int_]] = np.array(1)
# 调用 array 对象的 __array__ 方法
array.__array__()

# a 是类 A 的一个实例，其类型为 np.float64 的支持数组
a: _SupportsArray[np.dtype[np.float64]] = A()
# 连续两次调用 a 对象的 __array__ 方法
a.__array__()
a.__array__()

# 对于创建对象数组的转义，用于表示像对象数组一样的东西
# object_array_scalar 是一个生成器表达式，用于生成 0 到 9 的整数序列
object_array_scalar: object = (i for i in range(10))
# 使用 np.array 将 object_array_scalar 转换为数组对象
np.array(object_array_scalar)
```