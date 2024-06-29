# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\bitwise_ops.pyi`

```py
import numpy as np

# 创建一个64位整数的 numpy 对象 i8
i8 = np.int64()
# 创建一个32位整数的 numpy 对象 i4
i4 = np.int32()
# 创建一个64位无符号整数的 numpy 对象 u8
u8 = np.uint64()
# 创建一个布尔值的 numpy 对象 b_
b_ = np.bool()
# 创建一个普通的 Python 整数对象 i
i = int()

# 创建一个64位浮点数的 numpy 对象 f8
f8 = np.float64()

# 尝试使用位运算符 >>，但在布尔值 b_ 和浮点数 f8 之间无法进行位移操作
b_ >> f8  # E: No overload variant
# 尝试使用位运算符 <<，但在整数 i8 和浮点数 f8 之间无法进行位移操作
i8 << f8  # E: No overload variant
# 尝试使用位或运算符 |，但在整数 i 和浮点数 f8 之间无法进行位或操作
i | f8  # E: Unsupported operand types
# 尝试使用位异或运算符 ^，但在整数 i8 和浮点数 f8 之间无法进行位异或操作
i8 ^ f8  # E: No overload variant
# 尝试使用位与运算符 &，但在无符号整数 u8 和浮点数 f8 之间无法进行位与操作
u8 & f8  # E: No overload variant
# 尝试使用位取反运算符 ~，但浮点数 f8 不能进行位运算
~f8  # E: Unsupported operand type

# TODO: Certain mixes like i4 << u8 go to float and thus should fail

# mypys' error message for `NoReturn` is unfortunately pretty bad
# TODO: Re-enable this once we add support for numerical precision for `number`s
# a = u8 | 0  # E: Need type annotation
```