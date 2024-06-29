# `D:\src\scipysrc\numpy\numpy\typing\tests\data\reveal\scalars.pyi`

```
import sys
from typing import Any, Literal  # 导入需要使用的类型提示工具

import numpy as np  # 导入NumPy库
import numpy.typing as npt  # 导入NumPy类型提示

if sys.version_info >= (3, 11):
    from typing import assert_type  # Python版本大于等于3.11时，使用标准库中的assert_type函数
else:
    from typing_extensions import assert_type  # Python版本小于3.11时，使用typing_extensions中的assert_type函数

b: np.bool  # 声明变量b为NumPy中的布尔类型
u8: np.uint64  # 声明变量u8为NumPy中的64位无符号整数类型
i8: np.int64  # 声明变量i8为NumPy中的64位有符号整数类型
f8: np.float64  # 声明变量f8为NumPy中的64位浮点数类型
c8: np.complex64  # 声明变量c8为NumPy中的64位复数类型
c16: np.complex128  # 声明变量c16为NumPy中的128位复数类型
m: np.timedelta64  # 声明变量m为NumPy中的时间间隔类型
U: np.str_  # 声明变量U为NumPy中的字符串类型
S: np.bytes_  # 声明变量S为NumPy中的字节类型
V: np.void  # 声明变量V为NumPy中的空类型

assert_type(c8.real, np.float32)  # 断言c8的实部为32位浮点数类型
assert_type(c8.imag, np.float32)  # 断言c8的虚部为32位浮点数类型

assert_type(c8.real.real, np.float32)  # 断言c8的实部的实部为32位浮点数类型
assert_type(c8.real.imag, np.float32)  # 断言c8的实部的虚部为32位浮点数类型

assert_type(c8.itemsize, int)  # 断言c8的元素大小为整数类型
assert_type(c8.shape, tuple[()])  # 断言c8的形状为空元组类型
assert_type(c8.strides, tuple[()])  # 断言c8的步幅为空元组类型

assert_type(c8.ndim, Literal[0])  # 断言c8的维度为字面值0
assert_type(c8.size, Literal[1])  # 断言c8的大小为字面值1

assert_type(c8.squeeze(), np.complex64)  # 断言c8经过压缩后的类型为NumPy中的64位复数类型
assert_type(c8.byteswap(), np.complex64)  # 断言c8进行字节交换后的类型为NumPy中的64位复数类型
assert_type(c8.transpose(), np.complex64)  # 断言c8进行转置后的类型为NumPy中的64位复数类型

assert_type(c8.dtype, np.dtype[np.complex64])  # 断言c8的数据类型为NumPy中的64位复数类型

assert_type(c8.real, np.float32)  # 断言c8的实部为32位浮点数类型
assert_type(c16.imag, np.float64)  # 断言c16的虚部为64位浮点数类型

assert_type(np.str_('foo'), np.str_)  # 断言字符串'foo'的类型为NumPy中的字符串类型

assert_type(V[0], Any)  # 断言V的第一个元素的类型为任意类型
assert_type(V["field1"], Any)  # 断言V中字段"field1"的类型为任意类型
assert_type(V[["field1", "field2"]], np.void)  # 断言V中字段"field1"和"field2"的类型为NumPy中的空类型
V[0] = 5  # 设置V的第一个元素为整数5

# 别名
assert_type(np.bool_(), np.bool)  # 断言np.bool_()的类型为NumPy中的布尔类型
assert_type(np.byte(), np.byte)  # 断言np.byte()的类型为NumPy中的字节类型
assert_type(np.short(), np.short)  # 断言np.short()的类型为NumPy中的短整型
assert_type(np.intc(), np.intc)  # 断言np.intc()的类型为NumPy中的C语言整型
assert_type(np.intp(), np.intp)  # 断言np.intp()的类型为NumPy中的整数指针类型
assert_type(np.int_(), np.int_)  # 断言np.int_()的类型为NumPy中的整数类型
assert_type(np.long(), np.long)  # 断言np.long()的类型为NumPy中的长整型
assert_type(np.longlong(), np.longlong)  # 断言np.longlong()的类型为NumPy中的长长整型

assert_type(np.ubyte(), np.ubyte)  # 断言np.ubyte()的类型为NumPy中的无符号字节类型
assert_type(np.ushort(), np.ushort)  # 断言np.ushort()的类型为NumPy中的无符号短整型
assert_type(np.uintc(), np.uintc)  # 断言np.uintc()的类型为NumPy中的无符号C语言整型
assert_type(np.uintp(), np.uintp)  # 断言np.uintp()的类型为NumPy中的无符号整数指针类型
assert_type(np.uint(), np.uint)  # 断言np.uint()的类型为NumPy中的无符号整数类型
assert_type(np.ulong(), np.ulong)  # 断言np.ulong()的类型为NumPy中的无符号长整型
assert_type(np.ulonglong(), np.ulonglong)  # 断言np.ulonglong()的类型为NumPy中的无符号长长整型

assert_type(np.half(), np.half)  # 断言np.half()的类型为NumPy中的半精度浮点数类型
assert_type(np.single(), np.single)  # 断言np.single()的类型为NumPy中的单精度浮点数类型
assert_type(np.double(), np.double)  # 断言np.double()的类型为NumPy中的双精度浮点数类型
assert_type(np.longdouble(), np.longdouble)  # 断言np.longdouble()的类型为NumPy中的长双精度浮点数类型

assert_type(np.csingle(), np.csingle)  # 断言np.csingle()的类型为NumPy中的单精度复数类型
assert_type(np.cdouble(), np.cdouble)  # 断言np.cdouble()的类型为NumPy中的双精度复数类型
assert_type(np.clongdouble(), np.clongdouble)  # 断言np.clongdouble()的类型为NumPy中的长双精度复数类型

assert_type(b.item(), bool)  # 断言b的元素的类型为布尔类型
assert_type(i8.item(), int)  # 断言i8的元素的类型为整数类型
assert_type(u8.item(), int)  # 断言u8的元素的类型为整数类型
assert_type(f8.item(), float)  # 断言f8的元素的类型为浮点数类型
assert_type(c16.item(), complex)  # 断言c16的元素的类型为复数类型
assert_type(U.item(), str)  # 断言U的元素的类型为字符串类型
assert_type(S.item(), bytes)  # 断言S的元素的类型为字节类型

assert_type(b.tolist(), bool)  # 断言b转换为列表后的类型为布尔类型
assert_type(i8.tolist(), int)  # 断言i8转换为列表后的类型为整数类型
assert_type(u8.tolist(), int)  # 断言u8转换为列表后的类型为整数类型
assert_type(f8.tolist(), float)  # 断言f8转换为列表后的类型为浮点数类型
assert_type(c16.tolist(), complex)  # 断言c16转换为列表后的类型为复数类型
assert_type(U.tolist(), str)  # 断言U转换为列表
# 断言数组 b 重塑为一维后的类型为 np.bool 类型的数组
assert_type(b.reshape(1), npt.NDArray[np.bool])
# 断言数组 i8 重塑为一维后的类型为 np.int64 类型的数组
assert_type(i8.reshape(1), npt.NDArray[np.int64])
# 断言数组 u8 重塑为一维后的类型为 np.uint64 类型的数组
assert_type(u8.reshape(1), npt.NDArray[np.uint64])
# 断言数组 f8 重塑为一维后的类型为 np.float64 类型的数组
assert_type(f8.reshape(1), npt.NDArray[np.float64])
# 断言数组 c16 重塑为一维后的类型为 np.complex128 类型的数组
assert_type(c16.reshape(1), npt.NDArray[np.complex128])
# 断言数组 U 重塑为一维后的类型为 np.str_ 类型的数组
assert_type(U.reshape(1), npt.NDArray[np.str_])
# 断言数组 S 重塑为一维后的类型为 np.bytes_ 类型的数组
assert_type(S.reshape(1), npt.NDArray[np.bytes_])

# 断言将 i8 转换为 float 类型后的类型为 Any
assert_type(i8.astype(float), Any)
# 断言将 i8 转换为 np.float64 类型后的类型为 np.float64
assert_type(i8.astype(np.float64), np.float64)

# 断言对 i8 使用 view 方法后返回的数组类型为 np.int64
assert_type(i8.view(), np.int64)
# 断言对 i8 使用 view 方法转换为 np.float64 后返回的数组类型为 np.float64
assert_type(i8.view(np.float64), np.float64)
# 断言对 i8 使用 view 方法转换为 float（无具体类型）后返回的数组类型为 Any
assert_type(i8.view(float), Any)
# 断言对 i8 使用 view 方法同时指定 np.float64 和 np.ndarray 后返回的数组类型为 np.float64
assert_type(i8.view(np.float64, np.ndarray), np.float64)

# 断言对 i8 使用 getfield 方法不指定参数时返回的类型为 Any
assert_type(i8.getfield(float), Any)
# 断言对 i8 使用 getfield 方法指定返回类型为 np.float64 后返回的类型为 np.float64
assert_type(i8.getfield(np.float64), np.float64)
# 断言对 i8 使用 getfield 方法指定返回类型为 np.float64 和 8 位的 np.float64 后返回的类型为 np.float64
assert_type(i8.getfield(np.float64, 8), np.float64)

# 断言对 f8 使用 as_integer_ratio 方法返回的结果类型为包含两个 int 的 tuple
assert_type(f8.as_integer_ratio(), tuple[int, int])
# 断言对 f8 使用 is_integer 方法返回的结果类型为 bool
assert_type(f8.is_integer(), bool)
# 断言对 f8 使用 __trunc__ 方法返回的结果类型为 int
assert_type(f8.__trunc__(), int)
# 断言对 f8 使用 __getformat__ 方法返回的结果类型为 str
assert_type(f8.__getformat__("float"), str)
# 断言对 f8 使用 hex 方法返回的结果类型为 str
assert_type(f8.hex(), str)
# 断言对 np.float64 使用 fromhex 方法解析字符串后返回的结果类型为 np.float64
assert_type(np.float64.fromhex("0x0.0p+0"), np.float64)

# 断言对 f8 使用 __getnewargs__ 方法返回的结果类型为包含一个 float 的 tuple
assert_type(f8.__getnewargs__(), tuple[float])
# 断言对 c16 使用 __getnewargs__ 方法返回的结果类型为包含两个 float 的 tuple
assert_type(c16.__getnewargs__(), tuple[float, float])

# 断言对 i8 的 numerator 属性返回的类型为 np.int64
assert_type(i8.numerator, np.int64)
# 断言对 i8 的 denominator 属性返回的类型为 Literal[1]
assert_type(i8.denominator, Literal[1])
# 断言对 u8 的 numerator 属性返回的类型为 np.uint64
assert_type(u8.numerator, np.uint64)
# 断言对 u8 的 denominator 属性返回的类型为 Literal[1]
assert_type(u8.denominator, Literal[1])
# 断言对 m 的 numerator 属性返回的类型为 np.timedelta64
assert_type(m.numerator, np.timedelta64)
# 断言对 m 的 denominator 属性返回的类型为 Literal[1]
assert_type(m.denominator, Literal[1])

# 断言对 i8 使用 round 函数返回的类型为 int
assert_type(round(i8), int)
# 断言对 i8 使用 round 函数保留三位小数后返回的类型为 np.int64
assert_type(round(i8, 3), np.int64)
# 断言对 u8 使用 round 函数返回的类型为 int
assert_type(round(u8), int)
# 断言对 u8 使用 round 函数保留三位小数后返回的类型为 np.uint64
assert_type(round(u8, 3), np.uint64)
# 断言对 f8 使用 round 函数返回的类型为 int
assert_type(round(f8), int)
# 断言对 f8 使用 round 函数保留三位小数后返回的类型为 np.float64
assert_type(round(f8, 3), np.float64)

# 断言对 f8 使用 __ceil__ 方法返回的类型为 int
assert_type(f8.__ceil__(), int)
# 断言对 f8 使用 __floor__ 方法返回的类型为 int
assert_type(f8.__floor__(), int)

# 断言对 i8 使用 is_integer 方法返回的结果类型为 Literal[True]
assert_type(i8.is_integer(), Literal[True])
```