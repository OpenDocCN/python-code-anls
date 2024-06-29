# `D:\src\scipysrc\numpy\numpy\dtypes.pyi`

```
# 导入 numpy 库
import numpy as np

# 声明一个全局变量 __all__，它是一个字符串列表，用于指定模块中公开的变量和函数名
__all__: list[str]

# 布尔类型数据的 numpy 数据类型
BoolDType = np.dtype[np.bool]

# 有符号和无符号的字节大小整数类型
Int8DType = np.dtype[np.int8]
UInt8DType = np.dtype[np.uint8]

# 有符号和无符号的16位整数类型
Int16DType = np.dtype[np.int16]
UInt16DType = np.dtype[np.uint16]

# 有符号和无符号的32位整数类型
Int32DType = np.dtype[np.int32]
UInt32DType = np.dtype[np.uint32]

# 有符号和无符号的64位整数类型
Int64DType = np.dtype[np.int64]
UInt64DType = np.dtype[np.uint64]

# 标准 C 语言命名版本的别名
ByteDType = np.dtype[np.byte]
UByteDType = np.dtype[np.ubyte]
ShortDType = np.dtype[np.short]
UShortDType = np.dtype[np.ushort]
IntDType = np.dtype[np.intc]
UIntDType = np.dtype[np.uintc]
LongDType = np.dtype[np.long]
ULongDType = np.dtype[np.ulong]
LongLongDType = np.dtype[np.longlong]
ULongLongDType = np.dtype[np.ulonglong]

# 浮点数类型，包括16位、32位和64位浮点数
Float16DType = np.dtype[np.float16]
Float32DType = np.dtype[np.float32]
Float64DType = np.dtype[np.float64]
LongDoubleDType = np.dtype[np.longdouble]

# 复数类型，包括64位和128位复数
Complex64DType = np.dtype[np.complex64]
Complex128DType = np.dtype[np.complex128]
CLongDoubleDType = np.dtype[np.clongdouble]

# 其他类型，包括对象、字节串、字符串、空类型、日期时间和时间间隔类型
ObjectDType = np.dtype[np.object_]
BytesDType = np.dtype[np.bytes_]
StrDType = np.dtype[np.str_]
VoidDType = np.dtype[np.void]
DateTime64DType = np.dtype[np.datetime64]
TimeDelta64DType = np.dtype[np.timedelta64]
```