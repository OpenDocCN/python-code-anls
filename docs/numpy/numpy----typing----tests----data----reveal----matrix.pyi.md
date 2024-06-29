# `D:\src\scipysrc\numpy\numpy\typing\tests\data\reveal\matrix.pyi`

```
import sys
from typing import Any  # 导入需要的类型提示

import numpy as np  # 导入NumPy库
import numpy.typing as npt  # 导入NumPy类型提示

if sys.version_info >= (3, 11):
    from typing import assert_type  # Python版本支持assert_type函数
else:
    from typing_extensions import assert_type  # 使用typing_extensions中的assert_type函数

mat: np.matrix[Any, np.dtype[np.int64]]  # 声明一个NumPy整数矩阵mat
ar_f8: npt.NDArray[np.float64]  # 声明一个NumPy浮点数数组ar_f8

assert_type(mat * 5, np.matrix[Any, Any])  # 断言mat * 5的结果是NumPy矩阵
assert_type(5 * mat, np.matrix[Any, Any])  # 断言5 * mat的结果是NumPy矩阵
mat *= 5  # mat乘以5，原地更新mat矩阵

assert_type(mat**5, np.matrix[Any, Any])  # 断言mat的5次方的结果是NumPy矩阵
mat **= 5  # 将mat矩阵升幂至5次方，原地更新mat矩阵

assert_type(mat.sum(), Any)  # 断言mat所有元素的和的类型
assert_type(mat.mean(), Any)  # 断言mat所有元素的平均值的类型
assert_type(mat.std(), Any)  # 断言mat所有元素的标准差的类型
assert_type(mat.var(), Any)  # 断言mat所有元素的方差的类型
assert_type(mat.prod(), Any)  # 断言mat所有元素的乘积的类型
assert_type(mat.any(), np.bool)  # 断言mat是否有任意一个True值，返回NumPy布尔类型
assert_type(mat.all(), np.bool)  # 断言mat是否所有元素都为True，返回NumPy布尔类型
assert_type(mat.max(), np.int64)  # 断言mat所有元素的最大值的类型
assert_type(mat.min(), np.int64)  # 断言mat所有元素的最小值的类型
assert_type(mat.argmax(), np.intp)  # 断言mat最大值的索引的类型
assert_type(mat.argmin(), np.intp)  # 断言mat最小值的索引的类型
assert_type(mat.ptp(), np.int64)  # 断言mat所有元素的峰峰值的类型

assert_type(mat.sum(axis=0), np.matrix[Any, Any])  # 断言按列求和的结果是NumPy矩阵
assert_type(mat.mean(axis=0), np.matrix[Any, Any])  # 断言按列求平均值的结果是NumPy矩阵
assert_type(mat.std(axis=0), np.matrix[Any, Any])  # 断言按列求标准差的结果是NumPy矩阵
assert_type(mat.var(axis=0), np.matrix[Any, Any])  # 断言按列求方差的结果是NumPy矩阵
assert_type(mat.prod(axis=0), np.matrix[Any, Any])  # 断言按列求乘积的结果是NumPy矩阵
assert_type(mat.any(axis=0), np.matrix[Any, np.dtype[np.bool]])  # 断言按列判断是否有True值，返回NumPy布尔类型的矩阵
assert_type(mat.all(axis=0), np.matrix[Any, np.dtype[np.bool]])  # 断言按列判断是否所有元素都为True，返回NumPy布尔类型的矩阵
assert_type(mat.max(axis=0), np.matrix[Any, np.dtype[np.int64]])  # 断言按列求最大值的结果的类型
assert_type(mat.min(axis=0), np.matrix[Any, np.dtype[np.int64]])  # 断言按列求最小值的结果的类型
assert_type(mat.argmax(axis=0), np.matrix[Any, np.dtype[np.intp]])  # 断言按列求最大值索引的类型
assert_type(mat.argmin(axis=0), np.matrix[Any, np.dtype[np.intp]])  # 断言按列求最小值索引的类型
assert_type(mat.ptp(axis=0), np.matrix[Any, np.dtype[np.int64]])  # 断言按列求峰峰值的类型

assert_type(mat.sum(out=ar_f8), npt.NDArray[np.float64])  # 断言将按列求和结果存储到ar_f8的类型
assert_type(mat.mean(out=ar_f8), npt.NDArray[np.float64])  # 断言将按列求平均值结果存储到ar_f8的类型
assert_type(mat.std(out=ar_f8), npt.NDArray[np.float64])  # 断言将按列求标准差结果存储到ar_f8的类型
assert_type(mat.var(out=ar_f8), npt.NDArray[np.float64])  # 断言将按列求方差结果存储到ar_f8的类型
assert_type(mat.prod(out=ar_f8), npt.NDArray[np.float64])  # 断言将按列求乘积结果存储到ar_f8的类型
assert_type(mat.any(out=ar_f8), npt.NDArray[np.float64])  # 断言将按列判断是否有True值结果存储到ar_f8的类型
assert_type(mat.all(out=ar_f8), npt.NDArray[np.float64])  # 断言将按列判断是否所有元素都为True结果存储到ar_f8的类型
assert_type(mat.max(out=ar_f8), npt.NDArray[np.float64])  # 断言将按列求最大值结果存储到ar_f8的类型
assert_type(mat.min(out=ar_f8), npt.NDArray[np.float64])  # 断言将按列求最小值结果存储到ar_f8的类型
assert_type(mat.argmax(out=ar_f8), npt.NDArray[np.float64])  # 断言将按列求最大值索引结果存储到ar_f8的类型
assert_type(mat.argmin(out=ar_f8), npt.NDArray[np.float64])  # 断言将按列求最小值索引结果存储到ar_f8的类型
assert_type(mat.ptp(out=ar_f8), npt.NDArray[np.float64])  # 断言将按列求峰峰值结果存储到ar_f8的类型

assert_type(mat.T, np.matrix[Any, np.dtype[np.int64]])  # 断言mat的转置的类型
assert_type(mat.I, np.matrix[Any, Any])  # 断言mat的逆矩阵的类型
assert_type(mat.A, npt.NDArray[np.int64])  # 断言将mat转换为数组的类型
assert_type(mat.A1, npt.NDArray[np.int64])  # 断言将mat转换为一维数组的类型
assert_type(mat.H, np.matrix[Any, np.dtype[np.int64]])  # 断言mat的共轭转置的类型
assert_type(mat.getT(), np.matrix[Any, np.dtype[np.int64]])  # 断言获取mat的转置的类型
assert_type(mat.getI(), np.matrix[Any, Any])  # 断言获取mat的逆矩阵的类型
assert_type(mat.getA(), npt.NDArray[np.int64])  # 断言获取mat的数组的类型
assert_type(mat.getA1(), npt.NDArray[np.int64])  # 断言获取mat的一维数组的类型
assert_type(mat.getH(), np.matrix[Any, np.dtype[np.int64]])  # 断言获取mat的共轭转置的类型

assert_type(np.bmat(ar_f8), np.matrix[Any, Any])  # 断言将ar_f8作为矩阵块矩阵的类型
assert_type(np.bmat([[0, 1, 2]]), np.matrix[Any, Any])  # 断言将列表作为矩阵块矩阵的类型
assert_type(np.bmat("mat"), np.matrix[Any, Any])  # 断言将字符串作为矩阵块矩阵的类型

assert_type(np.asmatrix(ar_f8, dtype=np.int64), np.matrix[Any, Any])  # 断言将ar_f8转换为指定数据类型的矩阵的类型
```