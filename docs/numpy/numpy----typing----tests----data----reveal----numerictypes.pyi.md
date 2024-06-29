# `D:\src\scipysrc\numpy\numpy\typing\tests\data\reveal\numerictypes.pyi`

```
# 导入 sys 模块，用于访问系统相关的功能
import sys
# 导入 Literal 类型提示，用于指定字面量类型
from typing import Literal
# 导入 numpy 库，约定使用 np 别名
import numpy as np

# 根据 Python 版本选择不同的 assert_type 函数导入方式
if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type

# 使用 assert_type 函数断言 np.ScalarType 的类型约束
assert_type(
    np.ScalarType,
    tuple[
        type[int],             # 整数类型
        type[float],           # 浮点数类型
        type[complex],         # 复数类型
        type[bool],            # 布尔类型
        type[bytes],           # 字节类型
        type[str],             # 字符串类型
        type[memoryview],      # 内存视图类型
        type[np.bool],         # numpy 布尔类型
        type[np.csingle],      # numpy 单精度复数类型
        type[np.cdouble],      # numpy 双精度复数类型
        type[np.clongdouble],  # numpy 长双精度复数类型
        type[np.half],         # numpy 半精度浮点数类型
        type[np.single],       # numpy 单精度浮点数类型
        type[np.double],       # numpy 双精度浮点数类型
        type[np.longdouble],   # numpy 长双精度浮点数类型
        type[np.byte],         # numpy 字节类型
        type[np.short],        # numpy 短整型
        type[np.intc],         # numpy C 风格整型
        type[np.long],         # numpy 长整型
        type[np.longlong],     # numpy 长长整型
        type[np.timedelta64],  # numpy 时间间隔类型
        type[np.datetime64],   # numpy 日期时间类型
        type[np.object_],      # numpy 对象类型
        type[np.bytes_],       # numpy 字节类型
        type[np.str_],         # numpy 字符串类型
        type[np.ubyte],        # numpy 无符号字节类型
        type[np.ushort],       # numpy 无符号短整型
        type[np.uintc],        # numpy 无符号 C 风格整型
        type[np.ulong],        # numpy 无符号长整型
        type[np.ulonglong],    # numpy 无符号长长整型
        type[np.void],         # numpy 空类型
    ],
)

# 使用 assert_type 函数断言 np.ScalarType 元组的第一个元素为整数类型
assert_type(np.ScalarType[0], type[int])
# 使用 assert_type 函数断言 np.ScalarType 元组的第四个元素为布尔类型
assert_type(np.ScalarType[3], type[bool])
# 使用 assert_type 函数断言 np.ScalarType 元组的第九个元素为 numpy 单精度复数类型
assert_type(np.ScalarType[8], type[np.csingle])
# 使用 assert_type 函数断言 np.ScalarType 元组的第十一个元素为 numpy 长双精度复数类型
assert_type(np.ScalarType[10], type[np.clongdouble])

# 使用 assert_type 函数断言 numpy 布尔类型 np.bool_ 为 numpy 布尔类型
assert_type(np.bool_, type[np.bool])

# 使用 assert_type 函数断言 numpy 类型码表中字符类型 "c" 的字面量类型为 Literal["c"]
assert_type(np.typecodes["Character"], Literal["c"])
# 使用 assert_type 函数断言 numpy 类型码表中复数类型 "FDG" 的字面量类型为 Literal["FDG"]
assert_type(np.typecodes["Complex"], Literal["FDG"])
# 使用 assert_type 函数断言 numpy 类型码表中所有类型码 "All" 的字面量类型为 Literal["?bhilqpBHILQPefdgFDGSUVOMm"]
assert_type(np.typecodes["All"], Literal["?bhilqpBHILQPefdgFDGSUVOMm"])
```