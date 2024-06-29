# `D:\src\scipysrc\numpy\numpy\typing\tests\data\reveal\multiarray.pyi`

```py
# 导入系统模块 sys
import sys
# 导入 datetime 模块并使用别名 dt
import datetime as dt
# 导入类型提示模块 Any 和 TypeVar
from typing import Any, TypeVar

# 导入 numpy 库，并使用别名 np
import numpy as np
# 导入 numpy.typing 模块中的 NDArray 类型提示
import numpy.typing as npt

# 如果 Python 版本大于等于 3.11，导入 typing 模块中的 assert_type 函数
# 否则，从 typing_extensions 模块中导入 assert_type 函数
if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type

# 定义一个类型变量 _SCT，它是 np.generic 类型的子类型，并支持协变性
_SCT = TypeVar("_SCT", bound=np.generic, covariant=True)

# 定义一个名为 SubClass 的类，它继承自 npt.NDArray[_SCT] 类型
class SubClass(npt.NDArray[_SCT]): ...

# 定义 subclass 变量，其类型为 SubClass[np.float64]
subclass: SubClass[np.float64]

# 定义 AR_f8 变量，其类型为 npt.NDArray[np.float64]
AR_f8: npt.NDArray[np.float64]
# 定义 AR_i8 变量，其类型为 npt.NDArray[np.int64]
AR_i8: npt.NDArray[np.int64]
# 定义 AR_u1 变量，其类型为 npt.NDArray[np.uint8]
AR_u1: npt.NDArray[np.uint8]
# 定义 AR_m 变量，其类型为 npt.NDArray[np.timedelta64]
AR_m: npt.NDArray[np.timedelta64]
# 定义 AR_M 变量，其类型为 npt.NDArray[np.datetime64]
AR_M: npt.NDArray[np.datetime64]

# 定义 AR_LIKE_f 变量，其类型为 list[float]
AR_LIKE_f: list[float]
# 定义 AR_LIKE_i 变量，其类型为 list[int]
AR_LIKE_i: list[int]

# 定义 m 变量，其类型为 np.timedelta64
m: np.timedelta64
# 定义 M 变量，其类型为 np.datetime64
M: np.datetime64

# 使用 AR_f8 创建一个广播对象 b_f8
b_f8 = np.broadcast(AR_f8)
# 使用 AR_i8, AR_f8, AR_f8 创建一个广播对象 b_i8_f8_f8
b_i8_f8_f8 = np.broadcast(AR_i8, AR_f8, AR_f8)

# 定义一个迭代器对象 nditer_obj，其类型为 np.nditer
nditer_obj: np.nditer

# 定义 date_scalar 变量，其类型为 dt.date
date_scalar: dt.date
# 定义 date_seq 变量，其类型为 list[dt.date]
date_seq: list[dt.date]
# 定义 timedelta_seq 变量，其类型为 list[dt.timedelta]
timedelta_seq: list[dt.timedelta]

# 定义一个名为 func 的函数，它接受一个 int 类型的参数 a，并返回一个 bool 类型的值
def func(a: int) -> bool: ...

# 使用 assert_type 函数验证 b_f8 的下一个值的类型为 tuple[Any, ...]
assert_type(next(b_f8), tuple[Any, ...])
# 使用 assert_type 函数验证 b_f8 重置的返回类型为 None
assert_type(b_f8.reset(), None)
# 使用 assert_type 函数验证 b_f8 的索引值的类型为 int
assert_type(b_f8.index, int)
# 使用 assert_type 函数验证 b_f8 的迭代器元组的类型为 tuple[np.flatiter[Any], ...]
assert_type(b_f8.iters, tuple[np.flatiter[Any], ...])
# 使用 assert_type 函数验证 b_f8 的维度数的类型为 int
assert_type(b_f8.nd, int)
# 使用 assert_type 函数验证 b_f8 的维度数量的类型为 int
assert_type(b_f8.ndim, int)
# 使用 assert_type 函数验证 b_f8 的迭代器数量的类型为 int
assert_type(b_f8.numiter, int)
# 使用 assert_type 函数验证 b_f8 的形状元组的类型为 tuple[int, ...]
assert_type(b_f8.shape, tuple[int, ...])
# 使用 assert_type 函数验证 b_f8 的大小的类型为 int
assert_type(b_f8.size, int)

# 使用 assert_type 函数验证 b_i8_f8_f8 的下一个值的类型为 tuple[Any, ...]
assert_type(next(b_i8_f8_f8), tuple[Any, ...])
# 使用 assert_type 函数验证 b_i8_f8_f8 重置的返回类型为 None
assert_type(b_i8_f8_f8.reset(), None)
# 使用 assert_type 函数验证 b_i8_f8_f8 的索引值的类型为 int
assert_type(b_i8_f8_f8.index, int)
# 使用 assert_type 函数验证 b_i8_f8_f8 的迭代器元组的类型为 tuple[np.flatiter[Any], ...]
assert_type(b_i8_f8_f8.iters, tuple[np.flatiter[Any], ...])
# 使用 assert_type 函数验证 b_i8_f8_f8 的维度数的类型为 int
assert_type(b_i8_f8_f8.nd, int)
# 使用 assert_type 函数验证 b_i8_f8_f8 的维度数量的类型为 int
assert_type(b_i8_f8_f8.ndim, int)
# 使用 assert_type 函数验证 b_i8_f8_f8 的迭代器数量的类型为 int
assert_type(b_i8_f8_f8.numiter, int)
# 使用 assert_type 函数验证 b_i8_f8_f8 的形状元组的类型为 tuple[int, ...]
assert_type(b_i8_f8_f8.shape, tuple[int, ...])
# 使用 assert_type 函数验证 b_i8_f8_f8 的大小的类型为 int
assert_type(b_i8_f8_f8.size, int)

# 使用 assert_type 函数验证 np.inner(AR_f8, AR_i8) 的返回类型为 Any
assert_type(np.inner(AR_f8, AR_i8), Any)

# 使用 assert_type 函数验证 np.where([True, True, False]) 的返回类型为 tuple[npt.NDArray[np.intp], ...]
assert_type(np.where([True, True, False]), tuple[npt.NDArray[np.intp], ...])
# 使用 assert_type 函数验证 np.where([True, True, False], 1, 0) 的返回类型为 npt.NDArray[Any]
assert_type(np.where([True, True, False], 1, 0), npt.NDArray[Any])

# 使用 assert_type 函数验证 np.lexsort([0, 1, 2]) 的返回类型为 Any
assert_type(np.lexsort([0, 1, 2]), Any)

# 使用 assert_type 函数验证 np.can_cast(np.dtype("i8"), int) 的返回类型为 bool
assert_type(np.can_cast(np.dtype("i8"), int), bool)
# 使用 assert_type 函数验证 np.can_cast(AR_f8, "f8") 的返回类型为 bool
assert_type(np.can_cast(AR_f8, "f8"), bool)
# 使用 assert_type 函数验证 np.can_cast(AR_f8, np.complex128, casting="unsafe") 的返回类型为 bool
assert_type(np.can_cast(AR_f8, np.complex128, casting="unsafe"), bool)

# 使用 assert_type 函数验证 np.min_scalar_type([1]) 的返回类型为 np.dtype[Any]
assert_type(np.min_scalar_type([1]), np.dtype[Any])
# 使用 assert_type 函数验证 np.min_scalar_type(AR_f8) 的返回类型为 np.dtype[Any]
assert_type(np.min_scalar_type(AR_f8), np.dtype[Any])

# 使用 assert_type 函数验证 np.result_type(int, [1]) 的返回类型为 np.dtype[Any]
assert_type(np.result_type(int, [1]), np.dtype[Any])
# 使用 assert_type 函数验证 np.result_type(AR_f8, AR_u1) 的返回类型为 np.dtype[Any]
assert_type(np.result_type(AR_f8, AR_u1), np.dtype[Any])
# 使用 assert_type 函数验证 np.result_type(AR_f8, np.complex128) 的返回类型为 np.dtype[Any]
assert_type(np.result_type(AR_f8, np.complex128), np.dtype[Any])

# 使用 assert_type 函数验证 np.dot(AR_LIKE_f, AR_i8) 的返回类型为 Any
assert_type(np.dot(AR_LIKE_f, AR_i8), Any)
# 使用 assert_type 函数验证 np.dot(AR_u1, 1) 的返回类型为 Any
assert_type(np.dot(AR_u1, 1), Any)
# 使用 assert_type 函数验证 np.dot(1.
# 断言两个对象是否共享内存，返回布尔值
assert_type(np.may_share_memory(1, 2), bool)
# 断言两个数组是否共享内存，返回布尔值
assert_type(np.may_share_memory(AR_f8, AR_f8, max_work=1), bool)

# 推断两个数据类型的最小公共类型
assert_type(np.promote_types(np.int32, np.int64), np.dtype[Any])
# 推断两个数据类型的最小公共类型
assert_type(np.promote_types("f4", float), np.dtype[Any])

# 创建一个从 Python 函数到通用函数的转换器
assert_type(np.frompyfunc(func, 1, 1, identity=None), np.ufunc)

# 返回 datetime64 数据类型的元数据
assert_type(np.datetime_data("m8[D]"), tuple[str, int])
# 返回 datetime64 数据类型的元数据
assert_type(np.datetime_data(np.datetime64), tuple[str, int])
# 返回 timedelta64 数据类型的元数据
assert_type(np.datetime_data(np.dtype(np.timedelta64)), tuple[str, int])

# 计算两个日期之间的工作日数量
assert_type(np.busday_count("2011-01", "2011-02"), np.int_)
# 计算两个日期之间的工作日数量
assert_type(np.busday_count(["2011-01"], "2011-02"), npt.NDArray[np.int_])
# 计算两个日期之间的工作日数量
assert_type(np.busday_count(["2011-01"], date_scalar), npt.NDArray[np.int_])

# 计算工作日偏移后的日期
assert_type(np.busday_offset(M, m), np.datetime64)
# 计算工作日偏移后的日期
assert_type(np.busday_offset(date_scalar, m), np.datetime64)
# 计算工作日偏移后的日期
assert_type(np.busday_offset(M, 5), np.datetime64)
# 计算工作日偏移后的日期
assert_type(np.busday_offset(AR_M, m), npt.NDArray[np.datetime64])
# 计算工作日偏移后的日期
assert_type(np.busday_offset(M, timedelta_seq), npt.NDArray[np.datetime64])
# 计算工作日偏移后的日期
assert_type(np.busday_offset("2011-01", "2011-02", roll="forward"), np.datetime64)
# 计算工作日偏移后的日期
assert_type(np.busday_offset(["2011-01"], "2011-02", roll="forward"), npt.NDArray[np.datetime64])

# 判断日期是否是工作日
assert_type(np.is_busday("2012"), np.bool)
# 判断日期是否是工作日
assert_type(np.is_busday(date_scalar), np.bool)
# 判断日期是否是工作日
assert_type(np.is_busday(["2012"]), npt.NDArray[np.bool])

# 将日期时间数组转换为字符串数组
assert_type(np.datetime_as_string(M), np.str_)
# 将日期时间数组转换为字符串数组
assert_type(np.datetime_as_string(AR_M), npt.NDArray[np.str_])

# 创建一个工作日日历对象
assert_type(np.busdaycalendar(holidays=date_seq), np.busdaycalendar)
# 创建一个工作日日历对象
assert_type(np.busdaycalendar(holidays=[M]), np.busdaycalendar)

# 比较两个字符数组的元素
assert_type(np.char.compare_chararrays("a", "b", "!=", rstrip=False), npt.NDArray[np.bool])
# 比较两个字符数组的元素
assert_type(np.char.compare_chararrays(b"a", b"a", "==", True), npt.NDArray[np.bool])

# 创建嵌套迭代器对象
assert_type(np.nested_iters([AR_i8, AR_i8], [[0], [1]], flags=["c_index"]), tuple[np.nditer, ...])
# 创建嵌套迭代器对象
assert_type(np.nested_iters([AR_i8, AR_i8], [[0], [1]], op_flags=[["readonly", "readonly"]]), tuple[np.nditer, ...])
# 创建嵌套迭代器对象
assert_type(np.nested_iters([AR_i8, AR_i8], [[0], [1]], op_dtypes=np.int_), tuple[np.nditer, ...])
# 创建嵌套迭代器对象
assert_type(np.nested_iters([AR_i8, AR_i8], [[0], [1]], order="C", casting="no"), tuple[np.nditer, ...])
```