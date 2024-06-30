# `D:\src\scipysrc\scipy\scipy\sparse\tests\test_csc.py`

```
import numpy as np  # 导入 NumPy 库，用于科学计算
from numpy.testing import assert_array_almost_equal, assert_  # 导入 NumPy 测试模块中的断言函数
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix  # 导入 SciPy 稀疏矩阵模块中的三种矩阵类型

import pytest  # 导入 Pytest 测试框架


def test_csc_getrow():
    N = 10  # 设定矩阵的大小 N × N
    np.random.seed(0)  # 设置随机数种子以便复现结果
    X = np.random.random((N, N))  # 生成一个随机 N × N 的数组
    X[X > 0.7] = 0  # 将数组中大于 0.7 的元素置为 0，稀疏化处理
    Xcsc = csc_matrix(X)  # 将数组转换为 CSC 格式的稀疏矩阵

    for i in range(N):
        arr_row = X[i:i + 1, :]  # 从密集数组中获取第 i 行作为密集数组
        csc_row = Xcsc.getrow(i)  # 从 CSC 稀疏矩阵中获取第 i 行

        assert_array_almost_equal(arr_row, csc_row.toarray())  # 断言稀疏矩阵行与密集数组行近似相等
        assert_(type(csc_row) is csr_matrix)  # 断言获取的行是 CSR 格式的稀疏矩阵


def test_csc_getcol():
    N = 10  # 设定矩阵的大小 N × N
    np.random.seed(0)  # 设置随机数种子以便复现结果
    X = np.random.random((N, N))  # 生成一个随机 N × N 的数组
    X[X > 0.7] = 0  # 将数组中大于 0.7 的元素置为 0，稀疏化处理
    Xcsc = csc_matrix(X)  # 将数组转换为 CSC 格式的稀疏矩阵

    for i in range(N):
        arr_col = X[:, i:i + 1]  # 从密集数组中获取第 i 列作为密集数组
        csc_col = Xcsc.getcol(i)  # 从 CSC 稀疏矩阵中获取第 i 列

        assert_array_almost_equal(arr_col, csc_col.toarray())  # 断言稀疏矩阵列与密集数组列近似相等
        assert_(type(csc_col) is csc_matrix)  # 断言获取的列是 CSC 格式的稀疏矩阵


@pytest.mark.parametrize("matrix_input, axis, expected_shape",
    [(csc_matrix([[1, 0],
                [0, 0],
                [0, 2]]),
      0, (0, 2)),
     (csc_matrix([[1, 0],
                [0, 0],
                [0, 2]]),
      1, (3, 0)),
     (csc_matrix([[1, 0],
                [0, 0],
                [0, 2]]),
      'both', (0, 0)),
     (csc_matrix([[0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 2, 3, 0, 1]]),
      0, (0, 6))])
def test_csc_empty_slices(matrix_input, axis, expected_shape):
    # 查看相关讨论 gh-11127
    slice_1 = matrix_input.toarray().shape[0] - 1
    slice_2 = slice_1
    slice_3 = slice_2 - 1

    if axis == 0:
        actual_shape_1 = matrix_input[slice_1:slice_2, :].toarray().shape  # 获取切片在 axis=0 上的实际形状
        actual_shape_2 = matrix_input[slice_1:slice_3, :].toarray().shape  # 获取另一个切片在 axis=0 上的实际形状
    elif axis == 1:
        actual_shape_1 = matrix_input[:, slice_1:slice_2].toarray().shape  # 获取切片在 axis=1 上的实际形状
        actual_shape_2 = matrix_input[:, slice_1:slice_3].toarray().shape  # 获取另一个切片在 axis=1 上的实际形状
    elif axis == 'both':
        actual_shape_1 = matrix_input[slice_1:slice_2, slice_1:slice_2].toarray().shape  # 获取双向切片的实际形状
        actual_shape_2 = matrix_input[slice_1:slice_3, slice_1:slice_3].toarray().shape  # 获取双向切片的另一个实际形状

    assert actual_shape_1 == expected_shape  # 断言实际形状与期望形状相等
    assert actual_shape_1 == actual_shape_2  # 断言两个实际形状相等


@pytest.mark.parametrize('ax', (-2, -1, 0, 1, None))
def test_argmax_overflow(ax):
    # 查看相关讨论 gh-13646: 对于大稀疏矩阵，Windows 整数溢出问题
    dim = (100000, 100000)  # 定义大型矩阵的维度
    A = lil_matrix(dim)  # 生成一个 LIL 格式的大型稀疏矩阵
    A[-2, -2] = 42  # 在指定位置设置一个值
    A[-3, -3] = 0.1234  # 在另一个位置设置一个值
    A = csc_matrix(A)  # 将其转换为 CSC 格式的稀疏矩阵
    idx = A.argmax(axis=ax)  # 沿指定轴找到最大值的索引

    if ax is None:
        # idx 是一个单个扁平化索引，我们需要将其转换为二维索引对；
        # 无法使用 np.unravel_index 因为维度太大
        ii = idx % dim[0]  # 计算行索引
        jj = idx // dim[0]  # 计算列索引
    else:
        # idx 是一个大小为 A.shape[ax] 的数组；
        # 检查最大索引以确保没有溢出问题
        assert np.count_nonzero(idx) == A.nnz  # 断言非零元素的数量等于 A 的非零元素数量
        ii, jj = np.max(idx), np.argmax(idx)  # 获取最大索引和最大值的索引

    assert A[ii, jj] == A[-2, -2]  # 断言指定位置的值与预期值相等
```