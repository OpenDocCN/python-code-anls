# `D:\src\scipysrc\numpy\numpy\typing\tests\data\reveal\array_constructors.pyi`

```
import sys
from typing import Any, TypeVar
from pathlib import Path
from collections import deque

import numpy as np
import numpy.typing as npt

# 导入 assert_type 函数，根据 Python 版本不同选择不同的来源
if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type

# 定义泛型类型变量 _SCT，继承自 np.generic 类型，是协变的
_SCT = TypeVar("_SCT", bound=np.generic, covariant=True)

# 定义一个子类 SubClass，继承自 npt.NDArray[_SCT] 类型
class SubClass(npt.NDArray[_SCT]): ...

# 定义一个变量 i8，类型为 np.int64
i8: np.int64

# 定义变量 A，类型为 npt.NDArray[np.float64]
A: npt.NDArray[np.float64]

# 定义变量 B，类型为 SubClass[np.float64]
B: SubClass[np.float64]

# 定义变量 C，类型为 list[int]
C: list[int]

# 定义函数 func，接受两个整数参数 i 和 j，以及任意关键字参数，返回类型为 SubClass[np.float64]
def func(i: int, j: int, **kwargs: Any) -> SubClass[np.float64]: ...

# 对 np.empty_like(A) 的返回类型进行断言，应为 npt.NDArray[np.float64]
assert_type(np.empty_like(A), npt.NDArray[np.float64])

# 对 np.empty_like(B) 的返回类型进行断言，应为 SubClass[np.float64]
assert_type(np.empty_like(B), SubClass[np.float64])

# 对 np.empty_like([1, 1.0]) 的返回类型进行断言，应为 npt.NDArray[Any]
assert_type(np.empty_like([1, 1.0]), npt.NDArray[Any])

# 对 np.empty_like(A, dtype=np.int64) 的返回类型进行断言，应为 npt.NDArray[np.int64]
assert_type(np.empty_like(A, dtype=np.int64), npt.NDArray[np.int64])

# 对 np.empty_like(A, dtype='c16') 的返回类型进行断言，应为 npt.NDArray[Any]
assert_type(np.empty_like(A, dtype='c16'), npt.NDArray[Any])

# 对 np.array(A) 的返回类型进行断言，应为 npt.NDArray[np.float64]
assert_type(np.array(A), npt.NDArray[np.float64])

# 对 np.array(B) 的返回类型进行断言，应为 npt.NDArray[np.float64]
assert_type(np.array(B), npt.NDArray[np.float64])

# 对 np.array(B, subok=True) 的返回类型进行断言，应为 SubClass[np.float64]
assert_type(np.array(B, subok=True), SubClass[np.float64])

# 对 np.array([1, 1.0]) 的返回类型进行断言，应为 npt.NDArray[Any]
assert_type(np.array([1, 1.0]), npt.NDArray[Any])

# 对 np.array(deque([1, 2, 3])) 的返回类型进行断言，应为 npt.NDArray[Any]
assert_type(np.array(deque([1, 2, 3])), npt.NDArray[Any])

# 对 np.array(A, dtype=np.int64) 的返回类型进行断言，应为 npt.NDArray[np.int64]
assert_type(np.array(A, dtype=np.int64), npt.NDArray[np.int64])

# 对 np.array(A, dtype='c16') 的返回类型进行断言，应为 npt.NDArray[Any]
assert_type(np.array(A, dtype='c16'), npt.NDArray[Any])

# 对 np.array(A, like=A) 的返回类型进行断言，应为 npt.NDArray[np.float64]
assert_type(np.array(A, like=A), npt.NDArray[np.float64])

# 对 np.zeros([1, 5, 6]) 的返回类型进行断言，应为 npt.NDArray[np.float64]
assert_type(np.zeros([1, 5, 6]), npt.NDArray[np.float64])

# 对 np.zeros([1, 5, 6], dtype=np.int64) 的返回类型进行断言，应为 npt.NDArray[np.int64]
assert_type(np.zeros([1, 5, 6], dtype=np.int64), npt.NDArray[np.int64])

# 对 np.zeros([1, 5, 6], dtype='c16') 的返回类型进行断言，应为 npt.NDArray[Any]
assert_type(np.zeros([1, 5, 6], dtype='c16'), npt.NDArray[Any])

# 对 np.empty([1, 5, 6]) 的返回类型进行断言，应为 npt.NDArray[np.float64]
assert_type(np.empty([1, 5, 6]), npt.NDArray[np.float64])

# 对 np.empty([1, 5, 6], dtype=np.int64) 的返回类型进行断言，应为 npt.NDArray[np.int64]
assert_type(np.empty([1, 5, 6], dtype=np.int64), npt.NDArray[np.int64])

# 对 np.empty([1, 5, 6], dtype='c16') 的返回类型进行断言，应为 npt.NDArray[Any]
assert_type(np.empty([1, 5, 6], dtype='c16'), npt.NDArray[Any])

# 对 np.concatenate(A) 的返回类型进行断言，应为 npt.NDArray[np.float64]
assert_type(np.concatenate(A), npt.NDArray[np.float64])

# 对 np.concatenate([A, A]) 的返回类型进行断言，类型不确定，断言为 Any
assert_type(np.concatenate([A, A]), Any)

# 对 np.concatenate([[1], A]) 的返回类型进行断言，应为 npt.NDArray[Any]
assert_type(np.concatenate([[1], A]), npt.NDArray[Any])

# 对 np.concatenate([[1], [1]]) 的返回类型进行断言，应为 npt.NDArray[Any]
assert_type(np.concatenate([[1], [1]]), npt.NDArray[Any])

# 对 np.concatenate((A, A)) 的返回类型进行断言，应为 npt.NDArray[np.float64]
assert_type(np.concatenate((A, A)), npt.NDArray[np.float64])

# 对 np.concatenate(([1], [1])) 的返回类型进行断言，应为 npt.NDArray[Any]
assert_type(np.concatenate(([1], [1])), npt.NDArray[Any])

# 对 np.concatenate([1, 1.0]) 的返回类型进行断言，应为 npt.NDArray[Any]
assert_type(np.concatenate([1, 1.0]), npt.NDArray[Any])

# 对 np.concatenate(A, dtype=np.int64) 的返回类型进行断言，应为 npt.NDArray[np.int64]
assert_type(np.concatenate(A, dtype=np.int64), npt.NDArray[np.int64])

# 对 np.concatenate(A, dtype='c16') 的返回类型进行断言，应为 npt.NDArray[Any]
assert_type(np.concatenate(A, dtype='c16'), npt.NDArray[Any])

# 对 np.concatenate([1, 1.0], out=A) 的返回类型进行断言，应为 npt.NDArray[np.float64]
assert_type(np.concatenate([1, 1.0], out=A), npt.NDArray[np.float64])

# 对 np.asarray(A) 的返回类型进行断言，应为 npt.NDArray[np.float64]
assert_type(np.asarray(A), npt.NDArray[np.float64])

# 对 np.asarray(B) 的返回类型进行断言，应为 npt.NDArray[np.float64]
assert_type(np.asarray(B), npt.NDArray[np.float64])

# 对 np.asarray([1, 1.0]) 的返回类型进行断言，应为 npt.NDArray[Any]
assert_type(np.asarray([1, 1.0]), npt.NDArray[Any])

# 对 np.asarray(A, dtype=np.int64) 的返回类型进行断言，应为 npt.NDArray[np.int64]
assert_type(np.asarray(A, dtype=np.int64), npt.NDArray[np.int64])

# 对 np.asarray(A, dtype='c16') 的返回类型进行断言，应为 npt.NDArray[Any]
assert_type(np.asarray(A, dtype='c16'), npt.NDArray[Any])

# 对 np.asanyarray(A) 的返回类型进行断言，应为 npt.NDArray[np.float64]
assert_type(np.asanyarray(A), npt.NDArray[np.float64])

# 对 np.asanyarray(B) 的返回类型进行断言，应为 SubClass[np.float64]
assert_type(np.asanyarray(B), SubClass[np.float64])

# 对 np.asanyarray([1, 1.0]) 的返回类型进行断言，应为 npt.NDArray[Any]
assert_type(np.asanyarray([1,
# 使用 np.ascontiguousarray 函数将数组 A 转换为连续的内存布局，并指定数据类型为 np.int64，然后进行类型断言
assert_type(np.ascontiguousarray(A, dtype=np.int64), npt.NDArray[np.int64])

# 使用 np.ascontiguousarray 函数将数组 A 转换为连续的内存布局，并指定数据类型为 'c16'（复数类型），然后进行类型断言
assert_type(np.ascontiguousarray(A, dtype='c16'), npt.NDArray[Any])

# 使用 np.asfortranarray 函数将数组 A 转换为 Fortran 风格的内存布局，并进行类型断言
assert_type(np.asfortranarray(A), npt.NDArray[np.float64])

# 使用 np.asfortranarray 函数将数组 B 转换为 Fortran 风格的内存布局，并进行类型断言
assert_type(np.asfortranarray(B), npt.NDArray[np.float64])

# 使用 np.asfortranarray 函数将列表 [1, 1.0] 转换为 Fortran 风格的内存布局，并进行类型断言
assert_type(np.asfortranarray([1, 1.0]), npt.NDArray[Any])

# 使用 np.asfortranarray 函数将数组 A 转换为 Fortran 风格的内存布局，并指定数据类型为 np.int64，然后进行类型断言
assert_type(np.asfortranarray(A, dtype=np.int64), npt.NDArray[np.int64])

# 使用 np.asfortranarray 函数将数组 A 转换为 Fortran 风格的内存布局，并指定数据类型为 'c16'（复数类型），然后进行类型断言
assert_type(np.asfortranarray(A, dtype='c16'), npt.NDArray[Any])

# 使用 np.fromstring 函数从字符串 "1 1 1" 中按照空格分隔解析出 np.float64 类型的数组，并进行类型断言
assert_type(np.fromstring("1 1 1", sep=" "), npt.NDArray[np.float64])

# 使用 np.fromstring 函数从字节串 b"1 1 1" 中按照空格分隔解析出 np.float64 类型的数组，并进行类型断言
assert_type(np.fromstring(b"1 1 1", sep=" "), npt.NDArray[np.float64])

# 使用 np.fromstring 函数从字符串 "1 1 1" 中按照空格分隔解析出 np.int64 类型的数组，并进行类型断言
assert_type(np.fromstring("1 1 1", dtype=np.int64, sep=" "), npt.NDArray[np.int64])

# 使用 np.fromstring 函数从字节串 b"1 1 1" 中按照空格分隔解析出 np.int64 类型的数组，并进行类型断言
assert_type(np.fromstring(b"1 1 1", dtype=np.int64, sep=" "), npt.NDArray[np.int64])

# 使用 np.fromstring 函数从字符串 "1 1 1" 中按照空格分隔解析出 dtype 为 'c16'（复数类型）的数组，并进行类型断言
assert_type(np.fromstring("1 1 1", dtype="c16", sep=" "), npt.NDArray[Any])

# 使用 np.fromstring 函数从字节串 b"1 1 1" 中按照空格分隔解析出 dtype 为 'c16'（复数类型）的数组，并进行类型断言
assert_type(np.fromstring(b"1 1 1", dtype="c16", sep=" "), npt.NDArray[Any])

# 使用 np.fromfile 函数从文件 "test.txt" 中按照空格分隔解析出 np.float64 类型的数组，并进行类型断言
assert_type(np.fromfile("test.txt", sep=" "), npt.NDArray[np.float64])

# 使用 np.fromfile 函数从文件 "test.txt" 中按照空格分隔解析出 dtype 为 np.int64 的数组，并进行类型断言
assert_type(np.fromfile("test.txt", dtype=np.int64, sep=" "), npt.NDArray[np.int64])

# 使用 np.fromfile 函数从文件 "test.txt" 中按照空格分隔解析出 dtype 为 'c16'（复数类型）的数组，并进行类型断言
assert_type(np.fromfile("test.txt", dtype="c16", sep=" "), npt.NDArray[Any])

# 使用 np.fromfile 函数从文件对象 f 中按照空格分隔解析出 np.float64 类型的数组，并进行类型断言
with open("test.txt") as f:
    assert_type(np.fromfile(f, sep=" "), npt.NDArray[np.float64])

# 使用 np.fromfile 函数从字节串 b"test.txt" 中按照空格分隔解析出 np.float64 类型的数组，并进行类型断言
assert_type(np.fromfile(b"test.txt", sep=" "), npt.NDArray[np.float64])

# 使用 np.fromfile 函数从路径对象 Path("test.txt") 中按照空格分隔解析出 np.float64 类型的数组，并进行类型断言
assert_type(np.fromfile(Path("test.txt"), sep=" "), npt.NDArray[np.float64])

# 使用 np.fromiter 函数从字符串 "12345" 中迭代解析出 np.float64 类型的数组，并进行类型断言
assert_type(np.fromiter("12345", np.float64), npt.NDArray[np.float64])

# 使用 np.fromiter 函数从字符串 "12345" 中迭代解析出 float 类型的数组，并进行类型断言
assert_type(np.fromiter("12345", float), npt.NDArray[Any])

# 使用 np.frombuffer 函数从数组 A 的缓冲区创建 np.float64 类型的数组，并进行类型断言
assert_type(np.frombuffer(A), npt.NDArray[np.float64])

# 使用 np.frombuffer 函数从数组 A 的缓冲区创建 dtype 为 np.int64 的数组，并进行类型断言
assert_type(np.frombuffer(A, dtype=np.int64), npt.NDArray[np.int64])

# 使用 np.frombuffer 函数从数组 A 的缓冲区创建 dtype 为 'c16'（复数类型）的数组，并进行类型断言
assert_type(np.frombuffer(A, dtype="c16"), npt.NDArray[Any])

# 使用 np.arange 函数生成一个从 False 到 True 的整数序列，并进行类型断言
assert_type(np.arange(False, True), npt.NDArray[np.signedinteger[Any]])

# 使用 np.arange 函数生成一个从 0 到 9 的整数序列，并进行类型断言
assert_type(np.arange(10), npt.NDArray[np.signedinteger[Any]])

# 使用 np.arange 函数生成一个从 0 到 10，步长为 2 的整数序列，并进行类型断言
assert_type(np.arange(0, 10, step=2), npt.NDArray[np.signedinteger[Any]])

# 使用 np.arange 函数生成一个从 0.0 到 9.0 的浮点数序列，并进行类型断言
assert_type(np.arange(10.0), npt.NDArray[np.floating[Any]])

# 使用 np.arange 函数生成一个从 0 到 9.0 的浮点数序列，并进行类型断言
assert_type(np.arange(start=0, stop=10.0), npt.NDArray[np.floating[Any]])

# 使用 np.arange 函数生成一个从 np.timedelta64(0) 开始到 np.timedelta64(10) 结束的时间序列，并进行类型断言
assert_type(np.arange(np.timedelta64(0)), npt.NDArray[np.timedelta64])

# 使用 np.arange 函数生成一个从 0 到 np.timedelta64(10) 的时间序列，并进行类型断言
assert_type(np.arange(0, np.timedelta64(10)), npt.NDArray[np.timedelta64])

# 使用 np.arange 函数生成一个从 np.datetime64("0") 到 np.datetime64("10") 的日期时间序列，并进行类型断言
assert_type(np.arange(np.datetime64("0"), np.datetime64("10")), npt.NDArray[np.datetime64])

# 使用 np.arange 函数生成一个从 0 到 9 的浮点数序列，并指定数据类型为 np.float64，并进行类型断言
assert_type(np.arange(10, dtype=np.float64), npt.NDArray[np.float64])

# 使用 np.arange 函数生成一个从 0 到 10，步长为 2 的整数序列，并指定数据类型为 np.int16，并进行类型断言
assert_type(np.arange(0, 10, step=2, dtype=np.int16), n
# 确保数组 B 满足指定的要求，返回类型为 npt.NDArray[Any]
assert_type(np.require(B, requirements={"F", "E"}), npt.NDArray[Any])

# 确保数组 B 满足指定的要求，返回类型为 SubClass[np.float64]
assert_type(np.require(B, requirements=["C", "OWNDATA"]), SubClass[np.float64])

# 确保数组 B 满足指定的要求，返回类型为 SubClass[np.float64]
assert_type(np.require(B, requirements="W"), SubClass[np.float64])

# 确保数组 B 满足指定的要求，返回类型为 SubClass[np.float64]
assert_type(np.require(B, requirements="A"), SubClass[np.float64])

# 确保数组 C 满足默认要求，返回类型为 npt.NDArray[Any]
assert_type(np.require(C), npt.NDArray[Any])

# 生成一个包含从 0 到 10 的均匀间隔的数组，返回类型为 npt.NDArray[np.floating[Any]]
assert_type(np.linspace(0, 10), npt.NDArray[np.floating[Any]])

# 生成一个包含从 0 到 10j（虚数单位）的均匀间隔的数组，返回类型为 npt.NDArray[np.complexfloating[Any, Any]]
assert_type(np.linspace(0, 10j), npt.NDArray[np.complexfloating[Any, Any]])

# 生成一个包含从 0 到 10 的均匀间隔的数组，元素类型为 np.int64，返回类型为 npt.NDArray[np.int64]
assert_type(np.linspace(0, 10, dtype=np.int64), npt.NDArray[np.int64])

# 生成一个包含从 0 到 10 的均匀间隔的数组，元素类型为 int（默认为 float），返回类型为 npt.NDArray[Any]
assert_type(np.linspace(0, 10, dtype=int), npt.NDArray[Any])

# 生成一个包含从 0 到 10 的均匀间隔的数组，并返回结果和步长的元组，结果类型为 npt.NDArray[np.floating[Any]]，步长类型为 np.floating[Any]
assert_type(np.linspace(0, 10, retstep=True), tuple[npt.NDArray[np.floating[Any]], np.floating[Any]])

# 生成一个包含从 0j（虚数单位）到 10 的均匀间隔的数组，并返回结果和步长的元组，结果类型为 npt.NDArray[np.complexfloating[Any, Any]]，步长类型为 np.complexfloating[Any, Any]
assert_type(np.linspace(0j, 10, retstep=True), tuple[npt.NDArray[np.complexfloating[Any, Any]], np.complexfloating[Any, Any]])

# 生成一个包含从 0 到 10 的均匀间隔的数组，并返回结果和步长的元组，结果类型为 npt.NDArray[np.int64]，步长类型为 np.int64
assert_type(np.linspace(0, 10, retstep=True, dtype=np.int64), tuple[npt.NDArray[np.int64], np.int64])

# 生成一个包含从 0j（虚数单位）到 10 的均匀间隔的数组，并返回结果和步长的元组，结果类型为 npt.NDArray[Any]，步长类型为 Any（默认为 complex）
assert_type(np.linspace(0j, 10, retstep=True, dtype=int), tuple[npt.NDArray[Any], Any])

# 生成一个包含从 0 到 10 的对数间隔的数组，返回类型为 npt.NDArray[np.floating[Any]]
assert_type(np.logspace(0, 10), npt.NDArray[np.floating[Any]])

# 生成一个包含从 0 到 10j（虚数单位）的对数间隔的数组，返回类型为 npt.NDArray[np.complexfloating[Any, Any]]
assert_type(np.logspace(0, 10j), npt.NDArray[np.complexfloating[Any, Any]])

# 生成一个包含从 0 到 10 的对数间隔的数组，元素类型为 np.int64，返回类型为 npt.NDArray[np.int64]
assert_type(np.logspace(0, 10, dtype=np.int64), npt.NDArray[np.int64])

# 生成一个包含从 0 到 10 的对数间隔的数组，元素类型为 int（默认为 float），返回类型为 npt.NDArray[Any]
assert_type(np.logspace(0, 10, dtype=int), npt.NDArray[Any])

# 生成一个包含从 0 到 10 的几何间隔的数组，返回类型为 npt.NDArray[np.floating[Any]]
assert_type(np.geomspace(0, 10), npt.NDArray[np.floating[Any]])

# 生成一个包含从 0 到 10j（虚数单位）的几何间隔的数组，返回类型为 npt.NDArray[np.complexfloating[Any, Any]]
assert_type(np.geomspace(0, 10j), npt.NDArray[np.complexfloating[Any, Any]])

# 生成一个包含从 0 到 10 的几何间隔的数组，元素类型为 np.int64，返回类型为 npt.NDArray[np.int64]
assert_type(np.geomspace(0, 10, dtype=np.int64), npt.NDArray[np.int64])

# 生成一个包含从 0 到 10 的几何间隔的数组，元素类型为 int（默认为 float），返回类型为 npt.NDArray[Any]
assert_type(np.geomspace(0, 10, dtype=int), npt.NDArray[Any])

# 返回与数组 A 结构相同且元素全为 0 的数组，返回类型为 npt.NDArray[np.float64]
assert_type(np.zeros_like(A), npt.NDArray[np.float64])

# 返回与数组 C 结构相同且元素全为 0 的数组，返回类型为 npt.NDArray[Any]
assert_type(np.zeros_like(C), npt.NDArray[Any])

# 返回与数组 A 结构相同且元素全为 0.0 的数组，返回类型为 npt.NDArray[Any]
assert_type(np.zeros_like(A, dtype=float), npt.NDArray[Any])

# 返回与数组 B 结构相同且元素全为 0 的数组，返回类型为 SubClass[np.float64]
assert_type(np.zeros_like(B), SubClass[np.float64])

# 返回与数组 B 结构相同且元素全为 0 的数组，元素类型为 np.int64，返回类型为 npt.NDArray[np.int64]
assert_type(np.zeros_like(B, dtype=np.int64), npt.NDArray[np.int64])

# 返回与数组 A 结构相同且元素全为 1 的数组，返回类型为 npt.NDArray[np.float64]
assert_type(np.ones_like(A), npt.NDArray[np.float64])

# 返回与数组 C 结构相同且元素全为 1 的数组，返回类型为 npt.NDArray[Any]
assert_type(np.ones_like(C), npt.NDArray[Any])

# 返回与数组 A 结构相同且元素全为 1.0 的数组，返回类型为 npt.NDArray[Any]
assert_type(np.ones_like(A, dtype=float), npt.NDArray[Any])

# 返回与数组 B 结构相同且元素全为 1 的数组，返回类型为 SubClass[np.float64]
assert_type(np.ones_like(B), SubClass[np.float64])

# 返回与数组 B 结构相同且元素全为 1 的数组，元素类型为 np.int64，返回类型为 npt.NDArray[np.int64]
assert_type(np.ones_like(B, dtype=np.int64), npt.NDArray[np.int64])

# 返回与数组 A 结构相同且元素全为 i8 的数组，元素类型为 np.float64，返回类型为 npt.NDArray[np.float64]
assert_type(np.full_like(A, i8), npt.NDArray[np.float64])

# 返回与数组 C 结构相同且元素全为 i8 的数组，返回类型为 npt.NDArray[Any]
assert_type(np.full_like(C, i8), npt.NDArray[Any])

# 返回与数组 A 结构相同且元素全为 i8 的数组，元素类型为 int，返回类型为 npt.NDArray[Any]
assert_type(np.full_like(A, i8, dtype=int), npt.NDArray[Any])

# 返回与数组 B 结构相同且元素全为 i8
# 检查 np.indices 函数返回值类型是否符合指定的 tuple 类型标注
assert_type(np.indices([1, 2, 3], sparse=True), tuple[npt.NDArray[np.int_], ...])

# 检查 np.fromfunction 函数返回值类型是否为 SubClass[np.float64]
assert_type(np.fromfunction(func, (3, 5)), SubClass[np.float64])

# 检查 np.identity 函数返回值类型是否为 npt.NDArray[np.float64]
assert_type(np.identity(10), npt.NDArray[np.float64])
# 检查带有 dtype 参数的 np.identity 函数返回值类型是否为 npt.NDArray[np.int64]
assert_type(np.identity(10, dtype=np.int64), npt.NDArray[np.int64])
# 检查带有 dtype 参数为 int 的 np.identity 函数返回值类型是否为 npt.NDArray[Any]
assert_type(np.identity(10, dtype=int), npt.NDArray[Any])

# 检查 np.atleast_1d 函数返回值类型是否为 npt.NDArray[np.float64]
assert_type(np.atleast_1d(A), npt.NDArray[np.float64])
# 检查 np.atleast_1d 函数返回值类型是否为 npt.NDArray[Any]
assert_type(np.atleast_1d(C), npt.NDArray[Any])
# 检查 np.atleast_1d 函数返回值类型是否为 tuple[npt.NDArray[Any], ...]
assert_type(np.atleast_1d(A, A), tuple[npt.NDArray[Any], ...])
# 检查 np.atleast_1d 函数返回值类型是否为 tuple[npt.NDArray[Any], ...]
assert_type(np.atleast_1d(A, C), tuple[npt.NDArray[Any], ...])
# 检查 np.atleast_1d 函数返回值类型是否为 tuple[npt.NDArray[Any], ...]
assert_type(np.atleast_1d(C, C), tuple[npt.NDArray[Any], ...])

# 检查 np.atleast_2d 函数返回值类型是否为 npt.NDArray[np.float64]
assert_type(np.atleast_2d(A), npt.NDArray[np.float64])
# 检查 np.atleast_2d 函数返回值类型是否为 tuple[npt.NDArray[Any], ...]
assert_type(np.atleast_2d(A, A), tuple[npt.NDArray[Any], ...])

# 检查 np.atleast_3d 函数返回值类型是否为 npt.NDArray[np.float64]
assert_type(np.atleast_3d(A), npt.NDArray[np.float64])
# 检查 np.atleast_3d 函数返回值类型是否为 tuple[npt.NDArray[Any], ...]
assert_type(np.atleast_3d(A, A), tuple[npt.NDArray[Any], ...])

# 检查 np.vstack 函数返回值类型是否为 np.ndarray[Any, Any]
assert_type(np.vstack([A, A]), np.ndarray[Any, Any])
# 检查 np.vstack 函数返回值类型是否为 npt.NDArray[np.float64]
assert_type(np.vstack([A, A], dtype=np.float64), npt.NDArray[np.float64])
# 检查 np.vstack 函数返回值类型是否为 npt.NDArray[Any]
assert_type(np.vstack([A, C]), npt.NDArray[Any])
# 检查 np.vstack 函数返回值类型是否为 npt.NDArray[Any]
assert_type(np.vstack([C, C]), npt.NDArray[Any])

# 检查 np.hstack 函数返回值类型是否为 np.ndarray[Any, Any]
assert_type(np.hstack([A, A]), np.ndarray[Any, Any])
# 检查 np.hstack 函数返回值类型是否为 npt.NDArray[np.float64]
assert_type(np.hstack([A, A], dtype=np.float64), npt.NDArray[np.float64])

# 检查 np.stack 函数返回值类型是否为 Any
assert_type(np.stack([A, A]), Any)
# 检查 np.stack 函数返回值类型是否为 npt.NDArray[np.float64]
assert_type(np.stack([A, A], dtype=np.float64), npt.NDArray[np.float64])
# 检查 np.stack 函数返回值类型是否为 npt.NDArray[Any]
assert_type(np.stack([A, C]), npt.NDArray[Any])
# 检查 np.stack 函数返回值类型是否为 npt.NDArray[Any]
assert_type(np.stack([C, C]), npt.NDArray[Any])
# 检查 np.stack 函数返回值类型是否为 Any
assert_type(np.stack([A, A], axis=0), Any)
# 检查 np.stack 函数返回值类型是否为 SubClass[np.float64]
assert_type(np.stack([A, A], out=B), SubClass[np.float64])

# 检查 np.block 函数返回值类型是否为 npt.NDArray[Any]
assert_type(np.block([[A, A], [A, A]]), npt.NDArray[Any])
# 检查 np.block 函数返回值类型是否为 npt.NDArray[Any]
assert_type(np.block(C), npt.NDArray[Any])

# 检查如果 Python 版本大于等于 3.12，则导入 Buffer 类型
if sys.version_info >= (3, 12):
    from collections.abc import Buffer

    # 定义函数 create_array，接受 npt.ArrayLike 类型参数，返回 npt.NDArray[Any]
    def create_array(obj: npt.ArrayLike) -> npt.NDArray[Any]: ...

    # 定义变量 buffer 类型为 Buffer
    buffer: Buffer
    # 检查 create_array 函数返回值类型是否为 npt.NDArray[Any]
    assert_type(create_array(buffer), npt.NDArray[Any])
```