# `D:\src\scipysrc\numpy\numpy\typing\tests\data\reveal\dtype.pyi`

```py
import sys
import ctypes as ct               # 导入 ctypes 库并使用别名 ct
from typing import Any           # 导入 typing 库中的 Any 类型

import numpy as np               # 导入 numpy 库并使用别名 np

if sys.version_info >= (3, 11):  # 检查 Python 版本是否大于等于 3.11
    from typing import assert_type  # 如果是，导入 typing 库中的 assert_type 函数
else:
    from typing_extensions import assert_type  # 否则，导入 typing_extensions 库中的 assert_type 函数

dtype_U: np.dtype[np.str_]       # 定义变量 dtype_U，类型为 numpy 的 str 类型
dtype_V: np.dtype[np.void]       # 定义变量 dtype_V，类型为 numpy 的 void 类型
dtype_i8: np.dtype[np.int64]     # 定义变量 dtype_i8，类型为 numpy 的 int64 类型

assert_type(np.dtype(np.float64), np.dtype[np.float64])  # 断言验证 np.float64 类型与其自身的类型相等
assert_type(np.dtype(np.float64, metadata={"test": "test"}), np.dtype[np.float64])  # 断言验证带元数据的 np.float64 类型与其自身的类型相等
assert_type(np.dtype(np.int64), np.dtype[np.int64])      # 断言验证 np.int64 类型与其自身的类型相等

# String aliases
assert_type(np.dtype("float64"), np.dtype[np.float64])    # 断言验证字符串别名 "float64" 与 np.float64 类型相等
assert_type(np.dtype("float32"), np.dtype[np.float32])    # 断言验证字符串别名 "float32" 与 np.float32 类型相等
assert_type(np.dtype("int64"), np.dtype[np.int64])        # 断言验证字符串别名 "int64" 与 np.int64 类型相等
assert_type(np.dtype("int32"), np.dtype[np.int32])        # 断言验证字符串别名 "int32" 与 np.int32 类型相等
assert_type(np.dtype("bool"), np.dtype[np.bool])          # 断言验证字符串别名 "bool" 与 np.bool 类型相等
assert_type(np.dtype("bytes"), np.dtype[np.bytes_])       # 断言验证字符串别名 "bytes" 与 np.bytes_ 类型相等
assert_type(np.dtype("str"), np.dtype[np.str_])           # 断言验证字符串别名 "str" 与 np.str_ 类型相等

# Python types
assert_type(np.dtype(complex), np.dtype[np.cdouble])      # 断言验证 Python 复数类型与 np.cdouble 类型相等
assert_type(np.dtype(float), np.dtype[np.double])         # 断言验证 Python 浮点数类型与 np.double 类型相等
assert_type(np.dtype(int), np.dtype[np.int_])             # 断言验证 Python 整数类型与 np.int_ 类型相等
assert_type(np.dtype(bool), np.dtype[np.bool])            # 断言验证 Python 布尔类型与 np.bool 类型相等
assert_type(np.dtype(str), np.dtype[np.str_])             # 断言验证 Python 字符串类型与 np.str_ 类型相等
assert_type(np.dtype(bytes), np.dtype[np.bytes_])         # 断言验证 Python 字节类型与 np.bytes_ 类型相等
assert_type(np.dtype(object), np.dtype[np.object_])       # 断言验证 Python 对象类型与 np.object_ 类型相等

# ctypes
assert_type(np.dtype(ct.c_double), np.dtype[np.double])   # 断言验证 ctypes 的 c_double 类型与 np.double 类型相等
assert_type(np.dtype(ct.c_longlong), np.dtype[np.longlong])  # 断言验证 ctypes 的 c_longlong 类型与 np.longlong 类型相等
assert_type(np.dtype(ct.c_uint32), np.dtype[np.uint32])    # 断言验证 ctypes 的 c_uint32 类型与 np.uint32 类型相等
assert_type(np.dtype(ct.c_bool), np.dtype[np.bool])        # 断言验证 ctypes 的 c_bool 类型与 np.bool 类型相等
assert_type(np.dtype(ct.c_char), np.dtype[np.bytes_])      # 断言验证 ctypes 的 c_char 类型与 np.bytes_ 类型相等
assert_type(np.dtype(ct.py_object), np.dtype[np.object_])  # 断言验证 ctypes 的 py_object 类型与 np.object_ 类型相等

# Special case for None
assert_type(np.dtype(None), np.dtype[np.double])           # 断言验证 None 类型与 np.double 类型相等

# Dtypes of dtypes
assert_type(np.dtype(np.dtype(np.float64)), np.dtype[np.float64])  # 断言验证 np.dtype(np.float64) 类型与 np.float64 类型相等

# Parameterized dtypes
assert_type(np.dtype("S8"), np.dtype[Any])                 # 断言验证参数化的 dtype "S8" 与 Any 类型相等

# Void
assert_type(np.dtype(("U", 10)), np.dtype[np.void])        # 断言验证 void 类型的 dtype ("U", 10) 与 np.void 类型相等

# Methods and attributes
assert_type(dtype_U.base, np.dtype[Any])                   # 断言验证 dtype_U 的 base 属性类型为 Any
assert_type(dtype_U.subdtype, None | tuple[np.dtype[Any], tuple[int, ...]])  # 断言验证 dtype_U 的 subdtype 属性为 None 或 (dtype[Any], tuple[int, ...])
assert_type(dtype_U.newbyteorder(), np.dtype[np.str_])      # 断言验证 dtype_U 调用 newbyteorder() 方法返回 np.str_ 类型
assert_type(dtype_U.type, type[np.str_])                    # 断言验证 dtype_U 的 type 属性为 type[np.str_]
assert_type(dtype_U.name, str)                              # 断言验证 dtype_U 的 name 属性为 str 类型
assert_type(dtype_U.names, None | tuple[str, ...])          # 断言验证 dtype_U 的 names 属性为 None 或 (str, ...)

assert_type(dtype_U * 0, np.dtype[np.str_])                 # 断言验证 dtype_U 乘以 0 的结果为 np.str_ 类型
assert_type(dtype_U * 1, np.dtype[np.str_])                 # 断言验证 dtype_U 乘以 1 的结果为 np.str_ 类型
assert_type(dtype_U * 2, np.dtype[np.str_])                 # 断言验证 dtype_U 乘以 2 的结果为 np.str_ 类型

assert_type(dtype_i8 * 0, np.dtype[np.void])               # 断言验证 dtype_i8 乘以 0 的结果为 np.void 类型
assert_type(dtype_i8 * 1, np.dtype[np.int64])              # 断言验证 dtype_i8 乘以 1 的结果为 np.int64 类型
assert_type(dtype_i8 * 2, np.dtype[np.void])               # 断言验证 dtype_i8 乘以 2 的结果为 np.void 类型

assert_type(0 * dtype_U, np.dtype[np.str_])                 # 断言验证 0 乘以 dtype_U 的结果为 np.str_ 类型
assert_type(1 * dtype_U, np.dtype[np.str_])                 # 断言验证 1 乘以 dtype_U 的结果为 np.str_ 类型
assert_type(2 * dtype_U, np.dtype[np.str_])                 # 断言验证 2 乘以 dtype_U 的结果为 np.str_ 类型

assert_type(0 * dtype_i8, np.dtype[Any])                    # 断言验证 0 乘以 dtype_i8 的结果为 Any 类型
assert_type(1 * dtype_i8, np.dtype[Any])                    # 断言验证 1 乘以 dtype_i8 的结果为 Any 类型
assert_type(2 * dtype_i8, np.dtype[Any])                    # 断言验证 2 乘以 dtype_i8 的结果为 Any 类型

assert_type(dtype_V["f0"], np.dtype[Any])                   # 断言验证 dtype_V 的索引 "f0" 结果为 Any 类型
assert_type(dtype_V[0], np.dtype[Any])                      # 断言验证 dtype_V 的索引 0 结果为 Any 类型
assert_type(dtype_V[["f0", "f1"]], np.dtype[np.void])       # 断言验证 dtype_V 的复合索引 ["f0", "f1"] 结果为 np.void 类型
assert_type(dtype_V[["f0"]], np.dtype[np.void])             # 断言验证 dtype_V 的复合索引 ["f0"] 结果为 np.void 类型
```