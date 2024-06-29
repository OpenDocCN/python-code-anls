# `.\numpy\numpy\typing\tests\data\pass\array_constructors.py`

```py
from typing import Any  # 导入必要的类型提示模块

import numpy as np  # 导入 NumPy 库
import numpy.typing as npt  # 导入 NumPy 的类型提示模块

class Index:
    def __index__(self) -> int:
        return 0  # Index 类的方法返回整数 0

class SubClass(npt.NDArray[np.float64]):
    pass  # 定义一个 SubClass 类，继承自 NumPy 的 NDArray 类型，元素类型为 np.float64

def func(i: int, j: int, **kwargs: Any) -> SubClass:
    return B  # 函数 func 接收两个整数参数并返回变量 B

i8 = np.int64(1)  # 创建一个 np.int64 类型的变量 i8，值为 1

A = np.array([1])  # 创建一个包含单个元素 1 的 NumPy 数组 A
B = A.view(SubClass).copy()  # 创建一个 B 变量，通过 A 的视图转换为 SubClass 类型并复制
B_stack = np.array([[1], [1]]).view(SubClass)  # 创建一个 B_stack 变量，通过 NumPy 数组的视图转换为 SubClass 类型
C = [1]  # 创建一个包含单个元素 1 的普通 Python 列表

np.ndarray(Index())  # 创建一个 NumPy 数组，其元素为 Index() 返回的整数 0
np.ndarray([Index()])  # 创建一个 NumPy 数组，其元素为包含 Index() 返回的整数 0 的列表

np.array(1, dtype=float)  # 创建一个包含单个元素 1.0 的 NumPy 数组，数据类型为 float
np.array(1, copy=None)  # 创建一个包含单个元素 1 的 NumPy 数组，未指定 copy 参数
np.array(1, order='F')  # 创建一个包含单个元素 1 的 NumPy 数组，内存布局顺序为列优先 (Fortran 风格)
np.array(1, order=None)  # 创建一个包含单个元素 1 的 NumPy 数组，未指定内存布局顺序
np.array(1, subok=True)  # 创建一个包含单个元素 1 的 NumPy 数组，允许子类化
np.array(1, ndmin=3)  # 创建一个至少包含三个维度的 NumPy 数组，元素为单个元素 1
np.array(1, str, copy=True, order='C', subok=False, ndmin=2)  # 创建一个字符串类型的 NumPy 数组，指定多个参数

np.asarray(A)  # 将 A 转换为 NumPy 数组
np.asarray(B)  # 将 B 转换为 NumPy 数组
np.asarray(C)  # 将 C 转换为 NumPy 数组

np.asanyarray(A)  # 将 A 转换为 NumPy 数组（若已是 NumPy 数组则不变）
np.asanyarray(B)  # 将 B 转换为 NumPy 数组（若已是 NumPy 数组则不变）
np.asanyarray(B, dtype=int)  # 将 B 转换为指定数据类型的 NumPy 数组
np.asanyarray(C)  # 将 C 转换为 NumPy 数组（若已是 NumPy 数组则不变）

np.ascontiguousarray(A)  # 将 A 转换为内存连续的 NumPy 数组
np.ascontiguousarray(B)  # 将 B 转换为内存连续的 NumPy 数组
np.ascontiguousarray(C)  # 将 C 转换为内存连续的 NumPy 数组

np.asfortranarray(A)  # 将 A 转换为 Fortran 风格的 NumPy 数组
np.asfortranarray(B)  # 将 B 转换为 Fortran 风格的 NumPy 数组
np.asfortranarray(C)  # 将 C 转换为 Fortran 风格的 NumPy 数组

np.require(A)  # 返回 A 的一个新的要求数组
np.require(B)  # 返回 B 的一个新的要求数组
np.require(B, dtype=int)  # 返回 B 的一个新的要求数组，指定数据类型
np.require(B, requirements=None)  # 返回 B 的一个新的要求数组，未指定要求
np.require(B, requirements="E")  # 返回 B 的一个新的要求数组，指定要求为 "E"
np.require(B, requirements=["ENSUREARRAY"])  # 返回 B 的一个新的要求数组，指定多个要求
np.require(B, requirements={"F", "E"})  # 返回 B 的一个新的要求数组，指定多个要求为集合
np.require(B, requirements=["C", "OWNDATA"])  # 返回 B 的一个新的要求数组，指定多个要求
np.require(B, requirements="W")  # 返回 B 的一个新的要求数组，指定要求为 "W"
np.require(B, requirements="A")  # 返回 B 的一个新的要求数组，指定要求为 "A"
np.require(C)  # 返回 C 的一个新的要求数组

np.linspace(0, 2)  # 返回一个在 [0, 2] 区间内均匀间隔的 NumPy 数组，默认包含 50 个元素
np.linspace(0.5, [0, 1, 2])  # 返回一个在 [0.5, [0, 1, 2]] 区间内均匀间隔的 NumPy 数组
np.linspace([0, 1, 2], 3)  # 返回一个在 [[0, 1, 2], 3] 区间内均匀间隔的 NumPy 数组
np.linspace(0j, 2)  # 返回一个在 [0j, 2] 区间内均匀间隔的复数 NumPy 数组
np.linspace(0, 2, num=10)  # 返回一个在 [0, 2] 区间内均匀间隔的 NumPy 数组，包含 10 个元素
np.linspace(0, 2, endpoint=True)  # 返回一个在 [0, 2] 区间内均匀间隔的 NumPy 数组，包含终点值 2
np.linspace(0, 2, retstep=True)  # 返回一个在 [0, 2] 区间内均匀间隔的 NumPy 数组以及步长
np.linspace(0j, 2j, retstep=True)  # 返回一个在 [0j, 2j] 区间内均匀间隔的复数 NumPy 数组以及步长
np.linspace(0, 2, dtype=bool)  # 返回一个在 [0, 2] 区间内均匀间隔的 NumPy 数组，数据类型为 bool
np.linspace([0, 1], [2, 3], axis=Index())  # 返回一个在 [[0, 1], [2, 3]] 区间内均匀间隔的 NumPy 数组，指定轴

np.logspace(0, 2, base=2)  # 返回一个在 [2^0, 2^2] 区间内以 2 为底数均匀间隔的 NumPy 数组
np.logspace(0, 2, base=2)  # 返回一个在 [2^0, 2^2] 区间内以 2 为底数均匀间隔的 NumPy 数组
np.logspace(0, 2, base=[1j, 2j], num=2)  # 返回一个在 [1j^0, 2j^2] 区间内以复数为底数均匀间隔的 NumPy 数组，包含 2 个元素

np.geomspace(1, 2)  # 返回一个在 [1, 2] 区间内以几何级数间隔的 NumPy 数组

np.zeros_like(A)  # 返回一个与 A 形状和数据类型相同的全零 NumPy 数组
np.zeros_like(C)  # 返回一个与 C 形状和数据类型相同的全零 NumPy 数组
np.zeros_like(B)  # 返回一个与 B 形状和
```