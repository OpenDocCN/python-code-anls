# `D:\src\scipysrc\numpy\numpy\typing\tests\data\reveal\chararray.pyi`

```py
# 导入sys模块，用于访问系统相关功能
import sys
# 导入Any类型用于灵活类型声明
from typing import Any

# 导入NumPy库，并引入NumPy类型注解
import numpy as np
import numpy.typing as npt

# 根据Python版本选择合适的assert_type函数导入
if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type

# 声明AR_U和AR_S为chararray类型的NumPy数组，可以包含任意类型的数据
AR_U: np.char.chararray[Any, np.dtype[np.str_]]
AR_S: np.char.chararray[Any, np.dtype[np.bytes_]]

# 使用assert_type函数验证AR_U和AR_S的比较运算结果为布尔类型的NumPy数组
assert_type(AR_U == AR_U, npt.NDArray[np.bool])
assert_type(AR_S == AR_S, npt.NDArray[np.bool])

assert_type(AR_U != AR_U, npt.NDArray[np.bool])
assert_type(AR_S != AR_S, npt.NDArray[np.bool])

assert_type(AR_U >= AR_U, npt.NDArray[np.bool])
assert_type(AR_S >= AR_S, npt.NDArray[np.bool])

assert_type(AR_U <= AR_U, npt.NDArray[np.bool])
assert_type(AR_S <= AR_S, npt.NDArray[np.bool])

assert_type(AR_U > AR_U, npt.NDArray[np.bool])
assert_type(AR_S > AR_S, npt.NDArray[np.bool])

assert_type(AR_U < AR_U, npt.NDArray[np.bool])
assert_type(AR_S < AR_S, npt.NDArray[np.bool])

# 使用assert_type函数验证AR_U和AR_S的数学运算结果为chararray类型的NumPy数组
assert_type(AR_U * 5, np.char.chararray[Any, np.dtype[np.str_]])
assert_type(AR_S * [5], np.char.chararray[Any, np.dtype[np.bytes_]])

assert_type(AR_U % "test", np.char.chararray[Any, np.dtype[np.str_]])
assert_type(AR_S % b"test", np.char.chararray[Any, np.dtype[np.bytes_]])

# 使用assert_type函数验证AR_U和AR_S的字符串操作结果为chararray类型的NumPy数组
assert_type(AR_U.capitalize(), np.char.chararray[Any, np.dtype[np.str_]])
assert_type(AR_S.capitalize(), np.char.chararray[Any, np.dtype[np.bytes_]])

assert_type(AR_U.center(5), np.char.chararray[Any, np.dtype[np.str_]])
assert_type(AR_S.center([2, 3, 4], b"a"), np.char.chararray[Any, np.dtype[np.bytes_]])

assert_type(AR_U.encode(), np.char.chararray[Any, np.dtype[np.bytes_]])
assert_type(AR_S.decode(), np.char.chararray[Any, np.dtype[np.str_]])

assert_type(AR_U.expandtabs(), np.char.chararray[Any, np.dtype[np.str_]])
assert_type(AR_S.expandtabs(tabsize=4), np.char.chararray[Any, np.dtype[np.bytes_]])

assert_type(AR_U.join("_"), np.char.chararray[Any, np.dtype[np.str_]])
assert_type(AR_S.join([b"_", b""]), np.char.chararray[Any, np.dtype[np.bytes_]])

assert_type(AR_U.ljust(5), np.char.chararray[Any, np.dtype[np.str_]])
assert_type(AR_S.ljust([4, 3, 1], fillchar=[b"a", b"b", b"c"]), np.char.chararray[Any, np.dtype[np.bytes_]])
assert_type(AR_U.rjust(5), np.char.chararray[Any, np.dtype[np.str_]])
assert_type(AR_S.rjust([4, 3, 1], fillchar=[b"a", b"b", b"c"]), np.char.chararray[Any, np.dtype[np.bytes_]])

assert_type(AR_U.lstrip(), np.char.chararray[Any, np.dtype[np.str_]])
assert_type(AR_S.lstrip(chars=b"_"), np.char.chararray[Any, np.dtype[np.bytes_]])
assert_type(AR_U.rstrip(), np.char.chararray[Any, np.dtype[np.str_]])
assert_type(AR_S.rstrip(chars=b"_"), np.char.chararray[Any, np.dtype[np.bytes_]])
assert_type(AR_U.strip(), np.char.chararray[Any, np.dtype[np.str_]])
assert_type(AR_S.strip(chars=b"_"), np.char.chararray[Any, np.dtype[np.bytes_]])

assert_type(AR_U.partition("\n"), np.char.chararray[Any, np.dtype[np.str_]])
assert_type(AR_S.partition([b"a", b"b", b"c"]), np.char.chararray[Any, np.dtype[np.bytes_]])
assert_type(AR_U.rpartition("\n"), np.char.chararray[Any, np.dtype[np.str_]])
# 使用 assert_type 函数验证 AR_S.rpartition([b"a", b"b", b"c"]) 返回的类型是否为 np.char.chararray[Any, np.dtype[np.bytes_]]
assert_type(AR_S.rpartition([b"a", b"b", b"c"]), np.char.chararray[Any, np.dtype[np.bytes_]])

# 使用 assert_type 函数验证 AR_U.replace("_", "-") 返回的类型是否为 np.char.chararray[Any, np.dtype[np.str_]]
assert_type(AR_U.replace("_", "-"), np.char.chararray[Any, np.dtype[np.str_]])

# 使用 assert_type 函数验证 AR_S.replace([b"_", b""], [b"a", b"b"]) 返回的类型是否为 np.char.chararray[Any, np.dtype[np.bytes_]]
assert_type(AR_S.replace([b"_", b""], [b"a", b"b"]), np.char.chararray[Any, np.dtype[np.bytes_]])

# 使用 assert_type 函数验证 AR_U.split("_") 返回的类型是否为 npt.NDArray[np.object_]
assert_type(AR_U.split("_"), npt.NDArray[np.object_])

# 使用 assert_type 函数验证 AR_S.split(maxsplit=[1, 2, 3]) 返回的类型是否为 npt.NDArray[np.object_]
assert_type(AR_S.split(maxsplit=[1, 2, 3]), npt.NDArray[np.object_])

# 使用 assert_type 函数验证 AR_U.rsplit("_") 返回的类型是否为 npt.NDArray[np.object_]
assert_type(AR_U.rsplit("_"), npt.NDArray[np.object_])

# 使用 assert_type 函数验证 AR_S.rsplit(maxsplit=[1, 2, 3]) 返回的类型是否为 npt.NDArray[np.object_]
assert_type(AR_S.rsplit(maxsplit=[1, 2, 3]), npt.NDArray[np.object_])

# 使用 assert_type 函数验证 AR_U.splitlines() 返回的类型是否为 npt.NDArray[np.object_]
assert_type(AR_U.splitlines(), npt.NDArray[np.object_])

# 使用 assert_type 函数验证 AR_S.splitlines(keepends=[True, True, False]) 返回的类型是否为 npt.NDArray[np.object_]
assert_type(AR_S.splitlines(keepends=[True, True, False]), npt.NDArray[np.object_])

# 使用 assert_type 函数验证 AR_U.swapcase() 返回的类型是否为 np.char.chararray[Any, np.dtype[np.str_]]
assert_type(AR_U.swapcase(), np.char.chararray[Any, np.dtype[np.str_]])

# 使用 assert_type 函数验证 AR_S.swapcase() 返回的类型是否为 np.char.chararray[Any, np.dtype[np.bytes_]]
assert_type(AR_S.swapcase(), np.char.chararray[Any, np.dtype[np.bytes_]])

# 使用 assert_type 函数验证 AR_U.title() 返回的类型是否为 np.char.chararray[Any, np.dtype[np.str_]]
assert_type(AR_U.title(), np.char.chararray[Any, np.dtype[np.str_]])

# 使用 assert_type 函数验证 AR_S.title() 返回的类型是否为 np.char.chararray[Any, np.dtype[np.bytes_]]
assert_type(AR_S.title(), np.char.chararray[Any, np.dtype[np.bytes_]])

# 使用 assert_type 函数验证 AR_U.upper() 返回的类型是否为 np.char.chararray[Any, np.dtype[np.str_]]
assert_type(AR_U.upper(), np.char.chararray[Any, np.dtype[np.str_]])

# 使用 assert_type 函数验证 AR_S.upper() 返回的类型是否为 np.char.chararray[Any, np.dtype[np.bytes_]]
assert_type(AR_S.upper(), np.char.chararray[Any, np.dtype[np.bytes_]])

# 使用 assert_type 函数验证 AR_U.zfill(5) 返回的类型是否为 np.char.chararray[Any, np.dtype[np.str_]]
assert_type(AR_U.zfill(5), np.char.chararray[Any, np.dtype[np.str_]])

# 使用 assert_type 函数验证 AR_S.zfill([2, 3, 4]) 返回的类型是否为 np.char.chararray[Any, np.dtype[np.bytes_]]
assert_type(AR_S.zfill([2, 3, 4]), np.char.chararray[Any, np.dtype[np.bytes_]])

# 使用 assert_type 函数验证 AR_U.count("a", start=[1, 2, 3]) 返回的类型是否为 npt.NDArray[np.int_]
assert_type(AR_U.count("a", start=[1, 2, 3]), npt.NDArray[np.int_])

# 使用 assert_type 函数验证 AR_S.count([b"a", b"b", b"c"], end=9) 返回的类型是否为 npt.NDArray[np.int_]
assert_type(AR_S.count([b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])

# 使用 assert_type 函数验证 AR_U.endswith("a", start=[1, 2, 3]) 返回的类型是否为 npt.NDArray[np.bool]
assert_type(AR_U.endswith("a", start=[1, 2, 3]), npt.NDArray[np.bool])

# 使用 assert_type 函数验证 AR_S.endswith([b"a", b"b", b"c"], end=9) 返回的类型是否为 npt.NDArray[np.bool]
assert_type(AR_S.endswith([b"a", b"b", b"c"], end=9), npt.NDArray[np.bool])

# 使用 assert_type 函数验证 AR_U.startswith("a", start=[1, 2, 3]) 返回的类型是否为 npt.NDArray[np.bool]
assert_type(AR_U.startswith("a", start=[1, 2, 3]), npt.NDArray[np.bool])

# 使用 assert_type 函数验证 AR_S.startswith([b"a", b"b", b"c"], end=9) 返回的类型是否为 npt.NDArray[np.bool]
assert_type(AR_S.startswith([b"a", b"b", b"c"], end=9), npt.NDArray[np.bool])

# 使用 assert_type 函数验证 AR_U.find("a", start=[1, 2, 3]) 返回的类型是否为 npt.NDArray[np.int_]
assert_type(AR_U.find("a", start=[1, 2, 3]), npt.NDArray[np.int_])

# 使用 assert_type 函数验证 AR_S.find([b"a", b"b", b"c"], end=9) 返回的类型是否为 npt.NDArray[np.int_]
assert_type(AR_S.find([b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])

# 使用 assert_type 函数验证 AR_U.rfind("a", start=[1, 2, 3]) 返回的类型是否为 npt.NDArray[np.int_]
assert_type(AR_U.rfind("a", start=[1, 2, 3]), npt.NDArray[np.int_])

# 使用 assert_type 函数验证 AR_S.rfind([b"a", b"b", b"c"], end=9) 返回的类型是否为 npt.NDArray[np.int_]
assert_type(AR_S.rfind([b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])

# 使用 assert_type 函数验证 AR_U.index("a", start=[1, 2, 3]) 返回的类型是否为 npt.NDArray[np.int_]
assert_type(AR_U.index("a", start=[1, 2, 3]), npt.NDArray[np.int_])

# 使用 assert_type 函数验证 AR_S.index([b"a", b"b", b"c"], end=9) 返回的类型是否为 npt.NDArray[np.int_]
assert_type(AR_S.index([b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])

# 使用 assert_type 函数验证 AR_U.rindex("a", start=[1, 2, 3]) 返回的类型是否为 npt.NDArray[np.int_]
assert_type(AR_U.rindex("a", start=[1, 2, 3]), npt.NDArray[np.int_])

# 使用 assert_type 函数验证 AR_S.rindex([b"a", b"b", b"c"], end=9) 返回的类型是否为 npt.NDArray[np.int_]
assert_type(AR_S.rindex([b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])

# 使用 assert_type 函数验证 AR_U.isalpha() 返回的类型是否为 npt.NDArray[np.bool]
assert_type(AR_U.isalpha(), npt.NDArray[np.bool])

# 使用 assert_type 函数验证 AR_S.isalpha() 返回的类型是否为 npt.NDArray[np.bool]
assert_type(AR_S.isalpha(), npt.NDArray[np.bool])

# 使用 assert_type 函数验证 AR_U.isalnum() 返回的类型是否为 npt.NDArray[np.bool]
assert_type(AR_U.isalnum(), npt.NDArray[np.bool])

# 使用 assert_type 函数验证 AR_S.isalnum() 返回的类型是否为 npt.NDArray[np.bool]
assert_type(AR_S.isalnum(), npt.NDArray[np.bool])

# 使用 assert_type 函数验证 AR_U.isdecimal() 返回的类型是否为 npt.NDArray[np.bool]
assert
# 断言检查 AR_U 是否每个元素的首字母大写，返回布尔数组
assert_type(AR_U.istitle(), npt.NDArray[np.bool])

# 断言检查 AR_S 是否每个元素的首字母大写，返回布尔数组
assert_type(AR_S.istitle(), npt.NDArray[np.bool])

# 断言检查 AR_U 是否每个元素都是大写字母，返回布尔数组
assert_type(AR_U.isupper(), npt.NDArray[np.bool])

# 断言检查 AR_S 是否每个元素都是大写字母，返回布尔数组
assert_type(AR_S.isupper(), npt.NDArray[np.bool])

# 调用 __array_finalize__ 方法并断言其返回值类型为 None
assert_type(AR_U.__array_finalize__(object()), None)

# 调用 __array_finalize__ 方法并断言其返回值类型为 None
assert_type(AR_S.__array_finalize__(object()), None)
```