# `.\numpy\numpy\typing\tests\data\pass\ndarray_misc.py`

```py
"""
Tests for miscellaneous (non-magic) ``np.ndarray``/``np.generic`` methods.

More extensive tests are performed for the methods'
function-based counterpart in `../from_numeric.py`.

"""

# 导入必要的模块和类型声明
from __future__ import annotations  # 使用未来版本的类型注解支持

import operator  # 导入运算符模块
from typing import cast, Any  # 导入类型提示相关模块

import numpy as np  # 导入 NumPy 库
import numpy.typing as npt  # 导入 NumPy 的类型提示模块

class SubClass(npt.NDArray[np.float64]): ...  # 定义一个继承自 np.float64 的子类 SubClass

i4 = np.int32(1)  # 创建一个 np.int32 类型的标量 i4
A: np.ndarray[Any, np.dtype[np.int32]] = np.array([[1]], dtype=np.int32)  # 创建一个形状为 (1, 1) 的 np.ndarray A，元素类型为 np.int32
B0 = np.empty((), dtype=np.int32).view(SubClass)  # 创建一个空数组 B0，将其视图转换为 SubClass 类型
B1 = np.empty((1,), dtype=np.int32).view(SubClass)  # 创建一个形状为 (1,) 的空数组 B1，将其视图转换为 SubClass 类型
B2 = np.empty((1, 1), dtype=np.int32).view(SubClass)  # 创建一个形状为 (1, 1) 的空数组 B2，将其视图转换为 SubClass 类型
C: np.ndarray[Any, np.dtype[np.int32]] = np.array([0, 1, 2], dtype=np.int32)  # 创建一个形状为 (3,) 的 np.ndarray C，元素类型为 np.int32
D = np.ones(3).view(SubClass)  # 创建一个包含三个元素的全为 1 的数组 D，并将其视图转换为 SubClass 类型

i4.all()  # 调用 i4 的 all 方法，返回所有元素是否为 True 的布尔值
A.all()  # 调用 A 的 all 方法，返回所有元素是否为 True 的布尔值
A.all(axis=0)  # 调用 A 的 all 方法，按列计算所有元素是否为 True 的布尔值
A.all(keepdims=True)  # 调用 A 的 all 方法，保持维度，返回所有元素是否为 True 的布尔值
A.all(out=B0)  # 调用 A 的 all 方法，将结果存入数组 B0 中

i4.any()  # 调用 i4 的 any 方法，返回是否存在任意元素为 True 的布尔值
A.any()  # 调用 A 的 any 方法，返回是否存在任意元素为 True 的布尔值
A.any(axis=0)  # 调用 A 的 any 方法，按列计算是否存在任意元素为 True 的布尔值
A.any(keepdims=True)  # 调用 A 的 any 方法，保持维度，返回是否存在任意元素为 True 的布尔值
A.any(out=B0)  # 调用 A 的 any 方法，将结果存入数组 B0 中

i4.argmax()  # 返回 i4 中最大元素的索引
A.argmax()  # 返回 A 中最大元素的索引
A.argmax(axis=0)  # 返回 A 中每列最大元素的索引
A.argmax(out=B0)  # 返回 A 中最大元素的索引，并将结果存入数组 B0 中

i4.argmin()  # 返回 i4 中最小元素的索引
A.argmin()  # 返回 A 中最小元素的索引
A.argmin(axis=0)  # 返回 A 中每列最小元素的索引
A.argmin(out=B0)  # 返回 A 中最小元素的索引，并将结果存入数组 B0 中

i4.argsort()  # 返回 i4 的元素排序后的索引
A.argsort()  # 返回 A 的元素排序后的索引

i4.choose([()])  # 根据索引数组选择 i4 中的元素
_choices = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int32)  # 创建一个选择数组 _choices
C.choose(_choices)  # 根据选择数组 _choices 选择 C 中的元素
C.choose(_choices, out=D)  # 根据选择数组 _choices 选择 C 中的元素，并将结果存入数组 D 中

i4.clip(1)  # 将 i4 的元素限制在指定范围内，小于 1 的变为 1
A.clip(1)  # 将 A 的元素限制在指定范围内，小于 1 的变为 1
A.clip(None, 1)  # 将 A 的元素限制在指定范围内，大于 1 的变为 1
A.clip(1, out=B2)  # 将 A 的元素限制在指定范围内，并将结果存入数组 B2 中
A.clip(None, 1, out=B2)  # 将 A 的元素限制在指定范围内，大于 1 的变为 1，并将结果存入数组 B2 中

i4.compress([1])  # 返回根据条件压缩后的 i4 的数组
A.compress([1])  # 返回根据条件压缩后的 A 的数组
A.compress([1], out=B1)  # 返回根据条件压缩后的 A 的数组，并将结果存入数组 B1 中

i4.conj()  # 返回 i4 的共轭复数
A.conj()  # 返回 A 的每个元素的共轭复数
B0.conj()  # 返回 B0 的每个元素的共轭复数

i4.conjugate()  # 返回 i4 的共轭复数
A.conjugate()  # 返回 A 的每个元素的共轭复数
B0.conjugate()  # 返回 B0 的每个元素的共轭复数

i4.cumprod()  # 返回 i4 的累积乘积
A.cumprod()  # 返回 A 的每个元素的累积乘积
A.cumprod(out=B1)  # 返回 A 的每个元素的累积乘积，并将结果存入数组 B1 中

i4.cumsum()  # 返回 i4 的累积和
A.cumsum()  # 返回 A 的每个元素的累积和
A.cumsum(out=B1)  # 返回 A 的每个元素的累积和，并将结果存入数组 B1 中

i4.max()  # 返回 i4 的最大值
A.max()  # 返回 A 的最大值
A.max(axis=0)  # 返回 A 中每列的最大值
A.max(keepdims=True)  # 返回 A 的最大值，并保持维度
A.max(out=B0)  # 返回 A 的最大值，并将结果存入数组 B0 中

i4.mean()  # 返回 i4 的平均值
A.mean()  # 返回 A 的平均值
A.mean(axis=0)  # 返回 A 中每列的平均值
A.mean(keepdims=True)  # 返回 A 的平均值，并保持维度
A.mean(out=B0)  # 返回 A 的平均值，并将结果存入数组 B0 中

i4.min()  # 返回 i4 的最小值
A.min()  # 返回 A 的最小值
A.min(axis=0)  # 返回 A 中每列的最小值
A.min(keepdims=True)  # 返回 A 的最小值，并保持维度
A.min(out=B0)  # 返回 A 的最小值，并将结果存入数组 B0 中

i4.prod()  # 返回 i4 的所有元素的乘积
A.prod()  # 返回 A 的所有元素的乘积
A.prod(axis=0)  # 返回 A 中每列元素的乘积
A.prod(keepdims=True)  # 返回 A 的所有元素的乘积，并保持维度
A.prod(out=B0)  # 返回 A 的所有元素的乘积，并将结果存入数组 B0 中

i4.round()  # 返回 i4
```