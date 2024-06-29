# `.\numpy\numpy\typing\tests\data\pass\fromnumeric.py`

```py
"""Tests for :mod:`numpy._core.fromnumeric`."""

# 导入 numpy 库
import numpy as np

# 创建布尔类型的二维数组 A，初始化为 True
A = np.array(True, ndmin=2, dtype=bool)
# 创建单精度浮点数类型的二维数组 B，初始化为 1.0
B = np.array(1.0, ndmin=2, dtype=np.float32)
# 设置数组 A 的写保护标志，禁止写入操作
A.setflags(write=False)
# 设置数组 B 的写保护标志，禁止写入操作
B.setflags(write=False)

# 创建布尔类型标量 a，初始化为 True
a = np.bool(True)
# 创建单精度浮点数标量 b，初始化为 1.0
b = np.float32(1.0)
# 创建标量 c，初始化为 1.0
c = 1.0
# 创建单精度浮点数类型的数组 d，初始化为 1.0，可写入
d = np.array(1.0, dtype=np.float32)  # writeable

# 取出数组 a 中索引为 0 的元素
np.take(a, 0)
# 取出数组 b 中索引为 0 的元素
np.take(b, 0)
# 取出标量 c 的索引为 0 的元素，不支持
np.take(c, 0)
# 取出数组 A 中索引为 0 的元素
np.take(A, 0)
# 取出数组 B 中索引为 0 的元素
np.take(B, 0)
# 取出数组 A 中索引为 0 的元素组成的数组
np.take(A, [0])
# 取出数组 B 中索引为 0 的元素组成的数组
np.take(B, [0])

# 将标量 a 重塑为形状为 (1,) 的数组
np.reshape(a, 1)
# 将标量 b 重塑为形状为 (1,) 的数组
np.reshape(b, 1)
# 将标量 c 重塑为形状为 (1,) 的数组
np.reshape(c, 1)
# 将数组 A 重塑为形状为 (1,) 的数组
np.reshape(A, 1)
# 将数组 B 重塑为形状为 (1,) 的数组
np.reshape(B, 1)

# 从布尔类型标量 a 和给定选择数组中选择元素
np.choose(a, [True, True])
# 从数组 A 和给定选择数组中选择元素
np.choose(A, [1.0, 1.0])

# 对标量 a 重复 1 次
np.repeat(a, 1)
# 对标量 b 重复 1 次
np.repeat(b, 1)
# 对标量 c 重复 1 次
np.repeat(c, 1)
# 对数组 A 重复 1 次
np.repeat(A, 1)
# 对数组 B 重复 1 次
np.repeat(B, 1)

# 交换数组 A 的轴 0 和轴 0
np.swapaxes(A, 0, 0)
# 交换数组 B 的轴 0 和轴 0
np.swapaxes(B, 0, 0)

# 转置布尔类型标量 a
np.transpose(a)
# 转置单精度浮点数标量 b
np.transpose(b)
# 转置标量 c
np.transpose(c)
# 转置数组 A
np.transpose(A)
# 转置数组 B
np.transpose(B)

# 对布尔类型标量 a 进行轴向为 None 的分区排序
np.partition(a, 0, axis=None)
# 对单精度浮点数标量 b 进行轴向为 None 的分区排序
np.partition(b, 0, axis=None)
# 对标量 c 进行轴向为 None 的分区排序
np.partition(c, 0, axis=None)
# 对数组 A 进行以索引 0 为分割点的分区排序
np.partition(A, 0)
# 对数组 B 进行以索引 0 为分割点的分区排序
np.partition(B, 0)

# 返回数组 a 在轴 0 上的分区排序索引
np.argpartition(a, 0)
# 返回数组 b 在轴 0 上的分区排序索引
np.argpartition(b, 0)
# 返回标量 c 在轴 0 上的分区排序索引，不支持
np.argpartition(c, 0)
# 返回数组 A 在轴 0 上的分区排序索引
np.argpartition(A, 0)
# 返回数组 B 在轴 0 上的分区排序索引
np.argpartition(B, 0)

# 返回数组 A 沿轴 0 排序后的结果
np.sort(A, 0)
# 返回数组 B 沿轴 0 排序后的结果
np.sort(B, 0)

# 返回数组 A 沿轴 0 排序后的索引
np.argsort(A, 0)
# 返回数组 B 沿轴 0 排序后的索引
np.argsort(B, 0)

# 返回数组 A 中最大值的索引
np.argmax(A)
# 返回数组 B 中最大值的索引
np.argmax(B)
# 返回数组 A 在轴 0 上最大值的索引
np.argmax(A, axis=0)
# 返回数组 B 在轴 0 上最大值的索引
np.argmax(B, axis=0)

# 返回数组 A 中最小值的索引
np.argmin(A)
# 返回数组 B 中最小值的索引
np.argmin(B)
# 返回数组 A 在轴 0 上最小值的索引
np.argmin(A, axis=0)
# 返回数组 B 在轴 0 上最小值的索引
np.argmin(B, axis=0)

# 在数组 A[0] 中搜索值为 0 的插入点索引
np.searchsorted(A[0], 0)
# 在数组 B[0] 中搜索值为 0 的插入点索引
np.searchsorted(B[0], 0)
# 在数组 A[0] 中搜索值为 [0] 的插入点索引
np.searchsorted(A[0], [0])
# 在数组 B[0] 中搜索值为 [0] 的插入点索引
np.searchsorted(B[0], [0])

# 将标量 a 重塑为形状 (5, 5) 的数组
np.resize(a, (5, 5))
# 将标量 b 重塑为形状 (5, 5) 的数组
np.resize(b, (5, 5))
# 将标量 c 重塑为形状 (5, 5) 的数组
np.resize(c, (5, 5))
# 将数组 A 重塑为形状 (5, 5) 的数组
np.resize(A, (5, 5))
# 将数组 B 重塑为形状 (5, 5) 的数组
np.resize(B, (5, 5))

# 去除数组 a 的单维度条目
np.squeeze(a)
# 去除数组 b 的单维度条目
np.squeeze(b)
# 去除数组 c 的单维度条目
np.squeeze(c)
# 去除数组 A 的单维度条目
np.squeeze(A)
# 去除数组 B 的单维度条目
np.squeeze(B)

# 返回数组 A 的主对角线元素
np.diagonal(A)
# 返回数组 B 的主对角线元素
np.diagonal(B)

# 返回数组 A 的迹
np.trace(A)
# 返回数组 B 的迹
np.trace(B)

# 返回数组 a 的扁平化视图
np.ravel(a)
# 返回数组 b 的扁平化视图
np.ravel(b)
# 返回标量 c 的扁平化视图
np.ravel(c)
# 返回数组 A 的扁平化视图
np.ravel(A)
# 返回数组 B 的扁平化视图
np.ravel(B)

# 返回数组 A 中非零元素的索引
np.nonzero(A)
# 返回数组 B 中非零元素的索引
np.nonzero(B)

# 返回数组 a 的形状
np.shape(a)
# 返回数组 b 的形状
np.shape(b)
# 返回标量 c 的形状，不支持
np.shape(c)
# 返回数组 A 的形状
np.shape(A)
# 返回数组 B 的
# 计算数组 c 的累积乘积
np.cumprod(c)

# 计算数组 A 的累积乘积
np.cumprod(A)

# 计算数组 B 的累积乘积
np.cumprod(B)

# 返回数组 a 的维度数
np.ndim(a)

# 返回数组 b 的维度数
np.ndim(b)

# 返回数组 c 的维度数
np.ndim(c)

# 返回数组 A 的维度数
np.ndim(A)

# 返回数组 B 的维度数
np.ndim(B)

# 返回数组 a 中元素的总数
np.size(a)

# 返回数组 b 中元素的总数
np.size(b)

# 返回数组 c 中元素的总数
np.size(c)

# 返回数组 A 中元素的总数
np.size(A)

# 返回数组 B 中元素的总数
np.size(B)

# 返回数组 a 中每个元素四舍五入到整数
np.around(a)

# 返回数组 b 中每个元素四舍五入到整数
np.around(b)

# 返回数组 c 中每个元素四舍五入到整数
np.around(c)

# 返回数组 A 中每个元素四舍五入到整数
np.around(A)

# 返回数组 B 中每个元素四舍五入到整数
np.around(B)

# 返回数组 a 的平均值
np.mean(a)

# 返回数组 b 的平均值
np.mean(b)

# 返回数组 c 的平均值
np.mean(c)

# 返回数组 A 的所有元素的平均值
np.mean(A)

# 返回数组 B 的所有元素的平均值
np.mean(B)

# 返回数组 A 按行计算的平均值，axis=0 表示沿着第一个轴（行）计算
np.mean(A, axis=0)

# 返回数组 B 按行计算的平均值，axis=0 表示沿着第一个轴（行）计算
np.mean(B, axis=0)

# 返回数组 A 的平均值，并保持原有维度
np.mean(A, keepdims=True)

# 返回数组 B 的平均值，并保持原有维度
np.mean(B, keepdims=True)

# 返回数组 b 的平均值，将结果存储在数组 d 中
np.mean(b, out=d)

# 返回数组 B 的平均值，将结果存储在数组 d 中
np.mean(B, out=d)

# 返回数组 a 的标准差
np.std(a)

# 返回数组 b 的标准差
np.std(b)

# 返回数组 c 的标准差
np.std(c)

# 返回数组 A 的所有元素的标准差
np.std(A)

# 返回数组 B 的所有元素的标准差
np.std(B)

# 返回数组 A 按行计算的标准差，axis=0 表示沿着第一个轴（行）计算
np.std(A, axis=0)

# 返回数组 B 按行计算的标准差，axis=0 表示沿着第一个轴（行）计算
np.std(B, axis=0)

# 返回数组 A 的标准差，并保持原有维度
np.std(A, keepdims=True)

# 返回数组 B 的标准差，并保持原有维度
np.std(B, keepdims=True)

# 返回数组 b 的标准差，将结果存储在数组 d 中
np.std(b, out=d)

# 返回数组 B 的标准差，将结果存储在数组 d 中
np.std(B, out=d)

# 返回数组 a 的方差
np.var(a)

# 返回数组 b 的方差
np.var(b)

# 返回数组 c 的方差
np.var(c)

# 返回数组 A 的所有元素的方差
np.var(A)

# 返回数组 B 的所有元素的方差
np.var(B)

# 返回数组 A 按行计算的方差，axis=0 表示沿着第一个轴（行）计算
np.var(A, axis=0)

# 返回数组 B 按行计算的方差，axis=0 表示沿着第一个轴（行）计算
np.var(B, axis=0)

# 返回数组 A 的方差，并保持原有维度
np.var(A, keepdims=True)

# 返回数组 B 的方差，并保持原有维度
np.var(B, keepdims=True)

# 返回数组 b 的方差，将结果存储在数组 d 中
np.var(b, out=d)

# 返回数组 B 的方差，将结果存储在数组 d 中
np.var(B, out=d)
```