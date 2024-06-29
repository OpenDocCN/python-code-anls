# `.\numpy\numpy\typing\tests\data\pass\numeric.py`

```py
"""
Tests for :mod:`numpy._core.numeric`.

Does not include tests which fall under ``array_constructors``.
"""

# 导入 NumPy 库和类型定义模块
import numpy as np
import numpy.typing as npt

# 创建一个继承自 NDArray 的子类，元素类型为 np.float64
class SubClass(npt.NDArray[np.float64]):
    ...

# 创建一个 np.int64 类型的整数变量 i8，赋值为 1
i8 = np.int64(1)

# 使用 np.arange 创建一个长度为 27 的数组，并通过 reshape 转换为 3x3x3 的三维数组 A
A = np.arange(27).reshape(3, 3, 3)

# 将数组 A 转换为嵌套列表 B
B: list[list[list[int]]] = A.tolist()

# 创建一个形状为 (27, 27) 的空数组，并将其视图转换为 SubClass 类型 C
C = np.empty((27, 27)).view(SubClass)

# 计算 i8 中非零元素的数量
np.count_nonzero(i8)

# 计算数组 A 中非零元素的数量
np.count_nonzero(A)

# 计算嵌套列表 B 中非零元素的数量
np.count_nonzero(B)

# 计算数组 A 中非零元素的数量，并保持维度信息
np.count_nonzero(A, keepdims=True)

# 计算数组 A 沿着第 0 轴（行）的非零元素数量
np.count_nonzero(A, axis=0)

# 检查 i8 是否使用 Fortran 风格存储
np.isfortran(i8)

# 检查数组 A 是否使用 Fortran 风格存储
np.isfortran(A)

# 返回 i8 中非零元素的索引数组
np.argwhere(i8)

# 返回数组 A 中非零元素的索引数组
np.argwhere(A)

# 返回 i8 中扁平化后非零元素的索引数组
np.flatnonzero(i8)

# 返回数组 A 中扁平化后非零元素的索引数组
np.flatnonzero(A)

# 计算 B[0][0] 和 A 扁平化后的相关性，模式为 "valid"
np.correlate(B[0][0], A.ravel(), mode="valid")

# 计算数组 A 自身扁平化后的相关性，模式为 "same"
np.correlate(A.ravel(), A.ravel(), mode="same")

# 计算 B[0][0] 和 A 扁平化后的卷积，模式为 "valid"
np.convolve(B[0][0], A.ravel(), mode="valid")

# 计算数组 A 自身扁平化后的卷积，模式为 "same"
np.convolve(A.ravel(), A.ravel(), mode="same")

# 计算 i8 和数组 A 的外积
np.outer(i8, A)

# 计算嵌套列表 B 和数组 A 的外积
np.outer(B, A)

# 计算数组 A 自身的外积
np.outer(A, A)

# 计算数组 A 自身的外积，并将结果写入 C
np.outer(A, A, out=C)

# 计算 B 和数组 A 的张量积
np.tensordot(B, A)

# 计算数组 A 自身的张量积
np.tensordot(A, A)

# 计算数组 A 自身的张量积，轴为 (0, 1)
np.tensordot(A, A, axes=(0, 1))

# 检查 i8 是否为标量
np.isscalar(i8)

# 检查数组 A 是否为标量
np.isscalar(A)

# 检查嵌套列表 B 是否为标量
np.isscalar(B)

# 将数组 A 沿第 0 轴（行）滚动 1 个位置
np.roll(A, 1)

# 将数组 A 沿第 0 和第 1 轴滚动分别 1 和 2 个位置
np.roll(A, (1, 2))

# 将嵌套列表 B 沿第 0 轴滚动 1 个位置
np.roll(B, 1)

# 将数组 A 的第 0 轴（行）移动到第 1 轴（列）
np.rollaxis(A, 0, 1)

# 将数组 A 的第 0 轴（行）移动到第 1 轴（列），同时将第 1 轴（列）移动到第 2 轴（深度）
np.moveaxis(A, (0, 1), (1, 2))

# 计算嵌套列表 B 和数组 A 的叉乘
np.cross(B, A)

# 计算数组 A 自身的叉乘
np.cross(A, A)

# 返回一个形状为 [0, 1, 2] 的索引数组
np.indices([0, 1, 2])

# 返回一个形状为 [0, 1, 2] 的稀疏索引数组
np.indices([0, 1, 2], sparse=False)

# 返回一个形状为 [0, 1, 2] 的稀疏索引数组
np.indices([0, 1, 2], sparse=True)

# 返回整数 1 的二进制表示
np.binary_repr(1)

# 返回整数 1 的基数为 2 的表示
np.base_repr(1)

# 检查 i8 和数组 A 是否所有元素在容差范围内相等
np.allclose(i8, A)

# 检查嵌套列表 B 和数组 A 是否所有元素在容差范围内相等
np.allclose(B, A)

# 检查数组 A 自身所有元素在容差范围内是否相等
np.allclose(A, A)

# 检查 i8 和数组 A 是否所有元素在容差范围内近似相等
np.isclose(i8, A)

# 检查嵌套列表 B 和数组 A 是否所有元素在容差范围内近似相等
np.isclose(B, A)

# 检查数组 A 自身所有元素在容差范围内是否近似相等
np.isclose(A, A)

# 检查 i8 和数组 A 是否在形状和元素上完全相等
np.array_equal(i8, A)

# 检查嵌套列表 B 和数组 A 是否在形状和元素上完全相等
np.array_equal(B, A)

# 检查数组 A 自身是否在形状和元素上完全相等
np.array_equal(A, A)

# 检查 i8 和数组 A 是否在形状和元素上等效
np.array_equiv(i8, A)

# 检查嵌套列表 B 和数组 A 是否在形状和元素上等效
np.array_equiv(B, A)

# 检查数组 A 自身是否在形状和元素上等效
np.array_equiv(A, A)
```