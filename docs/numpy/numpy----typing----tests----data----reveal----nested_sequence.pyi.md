# `.\numpy\numpy\typing\tests\data\reveal\nested_sequence.pyi`

```py
# 导入 sys 模块，用于访问系统相关的功能
import sys
# 从 collections.abc 模块导入 Sequence 类型，用于声明序列类型的变量
from collections.abc import Sequence
# 从 typing 模块导入 Any 类型，用于表示任意类型的变量
from typing import Any

# 从 numpy._typing 模块导入 _NestedSequence 类型，用于声明嵌套序列类型的变量
from numpy._typing import _NestedSequence

# 根据 Python 版本不同选择不同的 assert_type 函数的来源
if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type

# 声明变量 a，其类型为整数序列的序列（二维列表或元组）
a: Sequence[int]
# 声明变量 b，其类型为整数序列的序列的序列（三维列表或元组）
b: Sequence[Sequence[int]]
# 声明变量 c，其类型为整数序列的序列的序列的序列（四维列表或元组）
c: Sequence[Sequence[Sequence[int]]]
# 声明变量 d，其类型为整数序列的序列的序列的序列的序列（五维列表或元组）
d: Sequence[Sequence[Sequence[Sequence[int]]]]
# 声明变量 e，其类型为布尔值序列
e: Sequence[bool]
# 声明变量 f，其类型为包含整数的元组，且元组长度不固定
f: tuple[int, ...]
# 声明变量 g，其类型为整数列表
g: list[int]
# 声明变量 h，其类型为任意类型的序列
h: Sequence[Any]

# 定义函数 func，接受一个 _NestedSequence[int] 类型的参数，无返回值
def func(a: _NestedSequence[int]) -> None:
    ...

# 对函数 func 的返回值进行类型断言，期望其返回 None 类型
assert_type(func(a), None)
assert_type(func(b), None)
assert_type(func(c), None)
assert_type(func(d), None)
assert_type(func(e), None)
assert_type(func(f), None)
assert_type(func(g), None)
assert_type(func(h), None)
# 对 func 函数参数为 range(15) 的返回值进行类型断言
assert_type(func(range(15)), None)
```