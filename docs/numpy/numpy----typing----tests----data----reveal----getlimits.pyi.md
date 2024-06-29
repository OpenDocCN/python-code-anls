# `D:\src\scipysrc\numpy\numpy\typing\tests\data\reveal\getlimits.pyi`

```py
# 导入系统模块 sys
import sys
# 导入 Any 类型，用于泛型类型注解
from typing import Any

# 导入 NumPy 库并使用别名 np
import numpy as np

# 如果 Python 版本大于等于 3.11，导入 typing 模块的 assert_type 函数
if sys.version_info >= (3, 11):
    from typing import assert_type
# 否则，从 typing_extensions 模块导入 assert_type 函数
else:
    from typing_extensions import assert_type

# 声明变量 f 为浮点数类型
f: float
# 声明变量 f8 为 NumPy 中的 np.float64 类型
f8: np.float64
# 声明变量 c8 为 NumPy 中的 np.complex64 类型

# 声明变量 i 为整数类型
i: int
# 声明变量 i8 为 NumPy 中的 np.int64 类型
i8: np.int64
# 声明变量 u4 为 NumPy 中的 np.uint32 类型

# 声明变量 finfo_f8 为 NumPy 中 np.finfo[np.float64] 类型
finfo_f8: np.finfo[np.float64]
# 声明变量 iinfo_i8 为 NumPy 中 np.iinfo[np.int64] 类型

# 使用 assert_type 函数验证 np.finfo(f) 的类型为 np.finfo[np.double]
assert_type(np.finfo(f), np.finfo[np.double])
# 使用 assert_type 函数验证 np.finfo(f8) 的类型为 np.finfo[np.float64]
assert_type(np.finfo(f8), np.finfo[np.float64])
# 使用 assert_type 函数验证 np.finfo(c8) 的类型为 np.finfo[np.float32]
assert_type(np.finfo(c8), np.finfo[np.float32])
# 使用 assert_type 函数验证 np.finfo('f2') 的类型为 np.finfo[np.floating[Any]]

# 使用 assert_type 函数验证 finfo_f8.dtype 的类型为 np.dtype[np.float64]
assert_type(finfo_f8.dtype, np.dtype[np.float64])
# 使用 assert_type 函数验证 finfo_f8.bits 的类型为 int
assert_type(finfo_f8.bits, int)
# 使用 assert_type 函数验证 finfo_f8.eps 的类型为 np.float64
assert_type(finfo_f8.eps, np.float64)
# 使用 assert_type 函数验证 finfo_f8.epsneg 的类型为 np.float64
assert_type(finfo_f8.epsneg, np.float64)
# 使用 assert_type 函数验证 finfo_f8.iexp 的类型为 int
assert_type(finfo_f8.iexp, int)
# 使用 assert_type 函数验证 finfo_f8.machep 的类型为 int
assert_type(finfo_f8.machep, int)
# 使用 assert_type 函数验证 finfo_f8.max 的类型为 np.float64
assert_type(finfo_f8.max, np.float64)
# 使用 assert_type 函数验证 finfo_f8.maxexp 的类型为 int
assert_type(finfo_f8.maxexp, int)
# 使用 assert_type 函数验证 finfo_f8.min 的类型为 np.float64
assert_type(finfo_f8.min, np.float64)
# 使用 assert_type 函数验证 finfo_f8.minexp 的类型为 int
assert_type(finfo_f8.minexp, int)
# 使用 assert_type 函数验证 finfo_f8.negep 的类型为 int
assert_type(finfo_f8.negep, int)
# 使用 assert_type 函数验证 finfo_f8.nexp 的类型为 int
assert_type(finfo_f8.nexp, int)
# 使用 assert_type 函数验证 finfo_f8.nmant 的类型为 int
assert_type(finfo_f8.nmant, int)
# 使用 assert_type 函数验证 finfo_f8.precision 的类型为 int
assert_type(finfo_f8.precision, int)
# 使用 assert_type 函数验证 finfo_f8.resolution 的类型为 np.float64
assert_type(finfo_f8.resolution, np.float64)
# 使用 assert_type 函数验证 finfo_f8.tiny 的类型为 np.float64
assert_type(finfo_f8.tiny, np.float64)
# 使用 assert_type 函数验证 finfo_f8.smallest_normal 的类型为 np.float64
assert_type(finfo_f8.smallest_normal, np.float64)
# 使用 assert_type 函数验证 finfo_f8.smallest_subnormal 的类型为 np.float64

# 使用 assert_type 函数验证 np.iinfo(i) 的类型为 np.iinfo[np.int_]
assert_type(np.iinfo(i), np.iinfo[np.int_])
# 使用 assert_type 函数验证 np.iinfo(i8) 的类型为 np.iinfo[np.int64]
assert_type(np.iinfo(i8), np.iinfo[np.int64])
# 使用 assert_type 函数验证 np.iinfo(u4) 的类型为 np.iinfo[np.uint32]
assert_type(np.iinfo(u4), np.iinfo[np.uint32])
# 使用 assert_type 函数验证 np.iinfo('i2') 的类型为 np.iinfo[Any]

# 使用 assert_type 函数验证 iinfo_i8.dtype 的类型为 np.dtype[np.int64]
assert_type(iinfo_i8.dtype, np.dtype[np.int64])
# 使用 assert_type 函数验证 iinfo_i8.kind 的类型为 str
assert_type(iinfo_i8.kind, str)
# 使用 assert_type 函数验证 iinfo_i8.bits 的类型为 int
assert_type(iinfo_i8.bits, int)
# 使用 assert_type 函数验证 iinfo_i8.key 的类型为 str
assert_type(iinfo_i8.key, str)
# 使用 assert_type 函数验证 iinfo_i8.min 的类型为 int
assert_type(iinfo_i8.min, int)
# 使用 assert_type 函数验证 iinfo_i8.max 的类型为 int
assert_type(iinfo_i8.max, int)
```