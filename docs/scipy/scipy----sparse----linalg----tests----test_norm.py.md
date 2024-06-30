# `D:\src\scipysrc\scipy\scipy\sparse\linalg\tests\test_norm.py`

```
"""Test functions for the sparse.linalg.norm module
"""

# 导入 pytest 库，用于测试框架
import pytest
# 导入 numpy 库，并重命名 norm 函数为 npnorm
import numpy as np
from numpy.linalg import norm as npnorm
# 导入 numpy.testing 库中的 assert_allclose 和 assert_equal 函数
from numpy.testing import assert_allclose, assert_equal
# 导入 pytest 库中的 raises 函数，并重命名为 assert_raises
from pytest import raises as assert_raises

# 导入 scipy.sparse 库
import scipy.sparse
# 导入 scipy.sparse.linalg 库中的 norm 函数，并重命名为 spnorm
from scipy.sparse.linalg import norm as spnorm


# https://github.com/scipy/scipy/issues/16031
# 测试稀疏数组的 norm 函数
def test_sparray_norm():
    # 创建稀疏矩阵的行、列、数据
    row = np.array([0, 0, 1, 1])
    col = np.array([0, 1, 2, 3])
    data = np.array([4, 5, 7, 9])
    # 使用 scipy.sparse 创建 COO 格式的稀疏数组和稀疏矩阵
    test_arr = scipy.sparse.coo_array((data, (row, col)), shape=(2, 4))
    test_mat = scipy.sparse.coo_matrix((data, (row, col)), shape=(2, 4))
    # 测试对稀疏数组和稀疏矩阵按列求 L1 范数，应与给定数组相等
    assert_equal(spnorm(test_arr, ord=1, axis=0), np.array([4, 5, 7, 9]))
    assert_equal(spnorm(test_mat, ord=1, axis=0), np.array([4, 5, 7, 9]))
    # 测试对稀疏数组和稀疏矩阵按行求 L1 范数，应与给定数组相等
    assert_equal(spnorm(test_arr, ord=1, axis=1), np.array([9, 16]))
    assert_equal(spnorm(test_mat, ord=1, axis=1), np.array([9, 16]))


# 定义测试类 TestNorm
class TestNorm:
    # 设置每个测试方法的初始化方法
    def setup_method(self):
        # 创建一个 numpy 数组 a，并转换为 CSR 格式的稀疏矩阵
        a = np.arange(9) - 4
        b = a.reshape((3, 3))
        self.b = scipy.sparse.csr_matrix(b)

    # 测试矩阵范数的方法
    def test_matrix_norm(self):

        # Frobenius 范数是默认的矩阵范数
        assert_allclose(spnorm(self.b), 7.745966692414834)        
        assert_allclose(spnorm(self.b, 'fro'), 7.745966692414834)

        # 测试按指定的 p 范数（inf、-inf、1、-1）计算矩阵范数
        assert_allclose(spnorm(self.b, np.inf), 9)
        assert_allclose(spnorm(self.b, -np.inf), 2)
        assert_allclose(spnorm(self.b, 1), 7)
        assert_allclose(spnorm(self.b, -1), 6)

        # 只有浮点数或复数浮点数 dtype 支持 svds
        with pytest.warns(UserWarning, match="The problem size"):
            assert_allclose(spnorm(self.b.astype(np.float64), 2),
                            7.348469228349534)

        # 对于稀疏矩阵，_multi_svd_norm 方法未实现
        assert_raises(NotImplementedError, spnorm, self.b, -2)

    # 测试矩阵范数的轴向方法
    def test_matrix_norm_axis(self):
        # 针对每个矩阵 m 和轴向 axis 的组合进行测试
        for m, axis in ((self.b, None), (self.b, (0, 1)), (self.b.T, (1, 0))):
            assert_allclose(spnorm(m, axis=axis), 7.745966692414834)        
            assert_allclose(spnorm(m, 'fro', axis=axis), 7.745966692414834)
            assert_allclose(spnorm(m, np.inf, axis=axis), 9)
            assert_allclose(spnorm(m, -np.inf, axis=axis), 2)
            assert_allclose(spnorm(m, 1, axis=axis), 7)
            assert_allclose(spnorm(m, -1, axis=axis), 6)

    # 测试向量范数的方法
    def test_vector_norm(self):
        # 定义一个向量 v
        v = [4.5825756949558398, 4.2426406871192848, 4.5825756949558398]
        # 针对每个矩阵 m 和轴 a 的组合进行测试
        for m, a in ((self.b, 0), (self.b.T, 1)):
            for axis in a, (a, ), a-2, (a-2, ):
                assert_allclose(spnorm(m, 1, axis=axis), [7, 6, 7])
                assert_allclose(spnorm(m, np.inf, axis=axis), [4, 3, 4])
                assert_allclose(spnorm(m, axis=axis), v)
                assert_allclose(spnorm(m, ord=2, axis=axis), v)
                assert_allclose(spnorm(m, ord=None, axis=axis), v)
    # 定义一个测试函数，用于测试 spnorm 函数的异常情况处理
    def test_norm_exceptions(self):
        # 将 self.b 赋值给 m，即测试对象
        m = self.b
        # 断言以下操作会引发 TypeError 异常：spnorm(m, None, 1.5)
        assert_raises(TypeError, spnorm, m, None, 1.5)
        # 断言以下操作会引发 TypeError 异常：spnorm(m, None, [2])
        assert_raises(TypeError, spnorm, m, None, [2])
        # 断言以下操作会引发 ValueError 异常：spnorm(m, None, ())
        assert_raises(ValueError, spnorm, m, None, ())
        # 断言以下操作会引发 ValueError 异常：spnorm(m, None, (0, 1, 2))
        assert_raises(ValueError, spnorm, m, None, (0, 1, 2))
        # 断言以下操作会引发 ValueError 异常：spnorm(m, None, (0, 0))
        assert_raises(ValueError, spnorm, m, None, (0, 0))
        # 断言以下操作会引发 ValueError 异常：spnorm(m, None, (0, 2))
        assert_raises(ValueError, spnorm, m, None, (0, 2))
        # 断言以下操作会引发 ValueError 异常：spnorm(m, None, (-3, 0))
        assert_raises(ValueError, spnorm, m, None, (-3, 0))
        # 断言以下操作会引发 ValueError 异常：spnorm(m, None, 2)
        assert_raises(ValueError, spnorm, m, None, 2)
        # 断言以下操作会引发 ValueError 异常：spnorm(m, None, -3)
        assert_raises(ValueError, spnorm, m, None, -3)
        # 断言以下操作会引发 ValueError 异常：spnorm(m, 'plate_of_shrimp', 0)
        assert_raises(ValueError, spnorm, m, 'plate_of_shrimp', 0)
        # 断言以下操作会引发 ValueError 异常：spnorm(m, 'plate_of_shrimp', (0, 1))
        assert_raises(ValueError, spnorm, m, 'plate_of_shrimp', (0, 1))
class TestVsNumpyNorm:
    # 定义稀疏矩阵类型元组
    _sparse_types = (
            scipy.sparse.bsr_matrix,
            scipy.sparse.coo_matrix,
            scipy.sparse.csc_matrix,
            scipy.sparse.csr_matrix,
            scipy.sparse.dia_matrix,
            scipy.sparse.dok_matrix,
            scipy.sparse.lil_matrix,
            )
    # 定义测试用矩阵列表
    _test_matrices = (
            (np.arange(9) - 4).reshape((3, 3)),
            [
                [1, 2, 3],
                [-1, 1, 4]],
            [
                [1, 0, 3],
                [-1, 1, 4j]],
            )

    # 测试稀疏矩阵的范数计算
    def test_sparse_matrix_norms(self):
        for sparse_type in self._sparse_types:
            for M in self._test_matrices:
                S = sparse_type(M)
                # 检查稀疏矩阵范数是否与 NumPy 矩阵范数相近
                assert_allclose(spnorm(S), npnorm(M))
                assert_allclose(spnorm(S, 'fro'), npnorm(M, 'fro'))
                assert_allclose(spnorm(S, np.inf), npnorm(M, np.inf))
                assert_allclose(spnorm(S, -np.inf), npnorm(M, -np.inf))
                assert_allclose(spnorm(S, 1), npnorm(M, 1))
                assert_allclose(spnorm(S, -1), npnorm(M, -1))

    # 测试带轴的稀疏矩阵范数计算
    def test_sparse_matrix_norms_with_axis(self):
        for sparse_type in self._sparse_types:
            for M in self._test_matrices:
                S = sparse_type(M)
                for axis in None, (0, 1), (1, 0):
                    # 检查带轴的稀疏矩阵范数是否与 NumPy 矩阵范数相近
                    assert_allclose(spnorm(S, axis=axis), npnorm(M, axis=axis))
                    for ord in 'fro', np.inf, -np.inf, 1, -1:
                        assert_allclose(spnorm(S, ord, axis=axis),
                                        npnorm(M, ord, axis=axis))
                # 一些 NumPy 矩阵范数不支持负轴
                for axis in (-2, -1), (-1, -2), (1, -2):
                    assert_allclose(spnorm(S, axis=axis), npnorm(M, axis=axis))
                    assert_allclose(spnorm(S, 'f', axis=axis),
                                    npnorm(M, 'f', axis=axis))
                    assert_allclose(spnorm(S, 'fro', axis=axis),
                                    npnorm(M, 'fro', axis=axis))

    # 测试稀疏向量的范数计算
    def test_sparse_vector_norms(self):
        for sparse_type in self._sparse_types:
            for M in self._test_matrices:
                S = sparse_type(M)
                for axis in (0, 1, -1, -2, (0, ), (1, ), (-1, ), (-2, )):
                    # 检查稀疏向量范数是否与 NumPy 向量范数相近
                    assert_allclose(spnorm(S, axis=axis), npnorm(M, axis=axis))
                    for ord in None, 2, np.inf, -np.inf, 1, 0.5, 0.42:
                        assert_allclose(spnorm(S, ord, axis=axis),
                                        npnorm(M, ord, axis=axis))
```