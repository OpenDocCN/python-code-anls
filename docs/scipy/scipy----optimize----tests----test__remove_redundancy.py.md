# `D:\src\scipysrc\scipy\scipy\optimize\tests\test__remove_redundancy.py`

```
"""
Unit test for Linear Programming via Simplex Algorithm.
"""

# TODO: add tests for:
# https://github.com/scipy/scipy/issues/5400
# https://github.com/scipy/scipy/issues/6690

import numpy as np
from numpy.testing import (
    assert_,
    assert_allclose,
    assert_equal)

# 从本地导入 magic_square 函数
from .test_linprog import magic_square

# 导入用于移除冗余的函数
from scipy.optimize._remove_redundancy import _remove_redundancy_svd
from scipy.optimize._remove_redundancy import _remove_redundancy_pivot_dense
from scipy.optimize._remove_redundancy import _remove_redundancy_pivot_sparse
from scipy.optimize._remove_redundancy import _remove_redundancy_id

# 导入稀疏矩阵类 csc_matrix
from scipy.sparse import csc_matrix


# 设置模块的初始化方法，用于设置随机种子
def setup_module():
    np.random.seed(2017)


# 定义一个函数，检查矩阵 A 是否只包含另一个矩阵 B 的独立行
def redundancy_removed(A, B):
    """Checks whether a matrix contains only independent rows of another"""
    for rowA in A:
        # `rowA in B` 不是一个可靠的检查方法
        for rowB in B:
            if np.all(rowA == rowB):
                break
        else:
            return False
    return A.shape[0] == np.linalg.matrix_rank(A) == np.linalg.matrix_rank(B)


# 定义一个测试类 RRCommonTests
class RRCommonTests:
    # 测试函数，检查没有冗余的情况
    def test_no_redundancy(self):
        m, n = 10, 10
        A0 = np.random.rand(m, n)
        b0 = np.random.rand(m)
        A1, b1, status, message = self.rr(A0, b0)
        assert_allclose(A0, A1)
        assert_allclose(b0, b1)
        assert_equal(status, 0)

    # 测试函数，检查存在不可行零行的情况
    def test_infeasible_zero_row(self):
        A = np.eye(3)
        A[1, :] = 0
        b = np.random.rand(3)
        A1, b1, status, message = self.rr(A, b)
        assert_equal(status, 2)

    # 测试函数，检查移除零行的情况
    def test_remove_zero_row(self):
        A = np.eye(3)
        A[1, :] = 0
        b = np.random.rand(3)
        b[1] = 0
        A1, b1, status, message = self.rr(A, b)
        assert_equal(status, 0)
        assert_allclose(A1, A[[0, 2], :])
        assert_allclose(b1, b[[0, 2]])

    # 测试函数，检查存在 m > n 不可行情况
    def test_infeasible_m_gt_n(self):
        m, n = 20, 10
        A0 = np.random.rand(m, n)
        b0 = np.random.rand(m)
        A1, b1, status, message = self.rr(A0, b0)
        assert_equal(status, 2)

    # 测试函数，检查存在 m = n 不可行情况
    def test_infeasible_m_eq_n(self):
        m, n = 10, 10
        A0 = np.random.rand(m, n)
        b0 = np.random.rand(m)
        A0[-1, :] = 2 * A0[-2, :]
        A1, b1, status, message = self.rr(A0, b0)
        assert_equal(status, 2)

    # 测试函数，检查存在 m < n 不可行情况
    def test_infeasible_m_lt_n(self):
        m, n = 9, 10
        A0 = np.random.rand(m, n)
        b0 = np.random.rand(m)
        A0[-1, :] = np.arange(m - 1).dot(A0[:-1])
        A1, b1, status, message = self.rr(A0, b0)
        assert_equal(status, 2)

    # 测试函数，检查存在 m > n 的情况
    def test_m_gt_n(self):
        np.random.seed(2032)
        m, n = 20, 10
        A0 = np.random.rand(m, n)
        b0 = np.random.rand(m)
        x = np.linalg.solve(A0[:n, :], b0[:n])
        b0[n:] = A0[n:, :].dot(x)
        A1, b1, status, message = self.rr(A0, b0)
        assert_equal(status, 0)
        assert_equal(A1.shape[0], n)
        assert_equal(np.linalg.matrix_rank(A1), n)
    # 定义一个测试函数，测试当 m > n 时的稀疏矩阵情况
    def test_m_gt_n_rank_deficient(self):
        # 设定矩阵的行数 m 和列数 n
        m, n = 20, 10
        # 创建一个 m 行 n 列的全零矩阵 A0
        A0 = np.zeros((m, n))
        # 将 A0 的所有行的第一列元素设为 1
        A0[:, 0] = 1
        # 创建一个长度为 m 的全一向量 b0
        b0 = np.ones(m)
        # 调用 rr 方法对 A0, b0 进行处理，返回处理后的结果 A1, b1, status, message
        A1, b1, status, message = self.rr(A0, b0)
        # 断言状态 status 为 0
        assert_equal(status, 0)
        # 断言 A1 的值与 A0 的第一行相等
        assert_allclose(A1, A0[0:1, :])
        # 断言 b1 的值与 b0 的第一个元素相等
        assert_allclose(b1, b0[0])

    # 定义一个测试函数，测试当 m < n 时的稀疏矩阵情况
    def test_m_lt_n_rank_deficient(self):
        # 设定矩阵的行数 m 和列数 n
        m, n = 9, 10
        # 创建一个 m 行 n 列的随机矩阵 A0
        A0 = np.random.rand(m, n)
        # 创建一个长度为 m 的随机向量 b0
        b0 = np.random.rand(m)
        # 修改 A0 的最后一行，使其满足线性依赖条件
        A0[-1, :] = np.arange(m - 1).dot(A0[:-1])
        # 修改 b0 的最后一个元素，使其满足线性依赖条件
        b0[-1] = np.arange(m - 1).dot(b0[:-1])
        # 调用 rr 方法对 A0, b0 进行处理，返回处理后的结果 A1, b1, status, message
        A1, b1, status, message = self.rr(A0, b0)
        # 断言状态 status 为 0
        assert_equal(status, 0)
        # 断言 A1 的行数为 8
        assert_equal(A1.shape[0], 8)
        # 断言 A1 的秩为 8
        assert_equal(np.linalg.matrix_rank(A1), 8)

    # 定义一个测试函数，测试密集矩阵情况1
    def test_dense1(self):
        # 创建一个 6x6 全一矩阵 A
        A = np.ones((6, 6))
        # 将 A 的第一行的前三列设为 0
        A[0, :3] = 0
        # 将 A 的第二行的第四列到最后设为 0
        A[1, 3:] = 0
        # 将 A 的第四行及之后的行的奇数列设为 -1
        A[3:, ::2] = -1
        # 将 A 的第四行的前两列设为 0
        A[3, :2] = 0
        # 将 A 的第五行的第三列及之后的列设为 0
        A[4, 2:] = 0
        # 创建一个长度为 6 的全零向量 b
        b = np.zeros(A.shape[0])
        # 调用 rr 方法对 A, b 进行处理，返回处理后的结果 A1, b1, status, message
        A1, b1, status, message = self.rr(A, b)
        # 断言使用 redundancy_removed 函数判断 A1 是否为 A 的冗余移除后的结果
        assert_(redundancy_removed(A1, A))
        # 断言状态 status 为 0
        assert_equal(status, 0)

    # 定义一个测试函数，测试密集矩阵情况2
    def test_dense2(self):
        # 创建一个 6x6 的单位矩阵 A
        A = np.eye(6)
        # 将 A 倒数第二行的最后一列设为 1
        A[-2, -1] = 1
        # 将 A 最后一行的所有元素设为 1
        A[-1, :] = 1
        # 创建一个长度为 6 的全零向量 b
        b = np.zeros(A.shape[0])
        # 调用 rr 方法对 A, b 进行处理，返回处理后的结果 A1, b1, status, message
        A1, b1, status, message = self.rr(A, b)
        # 断言使用 redundancy_removed 函数判断 A1 是否为 A 的冗余移除后的结果
        assert_(redundancy_removed(A1, A))
        # 断言状态 status 为 0
        assert_equal(status, 0)

    # 定义一个测试函数，测试密集矩阵情况3
    def test_dense3(self):
        # 创建一个 6x6 的单位矩阵 A
        A = np.eye(6)
        # 将 A 倒数第二行的最后一列设为 1
        A[-2, -1] = 1
        # 将 A 最后一行的所有元素设为 1
        A[-1, :] = 1
        # 创建一个长度为 6 的随机向量 b
        b = np.random.rand(A.shape[0])
        # 将 b 的最后一个元素设为前面所有元素之和
        b[-1] = np.sum(b[:-1])
        # 调用 rr 方法对 A, b 进行处理，返回处理后的结果 A1, b1, status, message
        A1, b1, status, message = self.rr(A, b)
        # 断言使用 redundancy_removed 函数判断 A1 是否为 A 的冗余移除后的结果
        assert_(redundancy_removed(A1, A))
        # 断言状态 status 为 0
        assert_equal(status, 0)

    # 定义一个测试函数，测试当 m > n 时的稀疏矩阵情况
    def test_m_gt_n_sparse(self):
        # 设定随机种子，确保结果可复现
        np.random.seed(2013)
        # 设定矩阵的行数 m 和列数 n
        m, n = 20, 5
        # 设定稀疏度 p
        p = 0.1
        # 创建一个 m 行 n 列的随机矩阵 A
        A = np.random.rand(m, n)
        # 将 A 中大于 p 的元素设为 0，使其稀疏化
        A[np.random.rand(m, n) > p] = 0
        # 计算 A 的秩
        rank = np.linalg.matrix_rank(A)
        # 创建一个长度为 m 的全零向量 b
        b = np.zeros(A.shape[0])
        # 调用 rr 方法对 A, b 进行处理，返回处理后的结果 A1, b1, status, message
        A1, b1, status, message = self.rr(A, b)
        # 断言状态 status 为 0
        assert_equal(status, 0)
        # 断言 A1 的行数与 A 的秩相等
        assert_equal(A1.shape[0], rank)
        # 断言 A1 的秩与 A 的秩相等
        assert_equal(np.linalg.matrix_rank(A1), rank)

    # 定义一个测试函数，测试当 m < n 时的稀疏矩阵情况
    def test_m_lt_n_sparse(self):
        # 设定随机种子，确保结果可复现
        np.random.seed(2017)
        # 设定矩阵的行数 m 和列数 n
        m, n = 20, 50
        # 设定稀疏度 p
        p = 0.05
        # 创建一个 m 行 n 列的随机矩阵 A
        A = np.random.rand
    # 定义测试函数，测试魔方阵生成函数的功能
    def test_magic_square(self):
        # 调用 magic_square 函数生成阶数为 3 的魔方阵的参数
        A, b, c, numbers, _ = magic_square(3)
        # 调用 self.rr 方法进行行约简，获取约简后的参数及状态信息
        A1, b1, status, message = self.rr(A, b)
        # 断言状态为0，表示行约简成功
        assert_equal(status, 0)
        # 断言约简后的矩阵 A1 的行数为23
        assert_equal(A1.shape[0], 23)
        # 断言约简后的矩阵 A1 的秩为23
        assert_equal(np.linalg.matrix_rank(A1), 23)

    # 定义第二个测试函数，测试魔方阵生成函数的功能
    def test_magic_square2(self):
        # 调用 magic_square 函数生成阶数为 4 的魔方阵的参数
        A, b, c, numbers, _ = magic_square(4)
        # 调用 self.rr 方法进行行约简，获取约简后的参数及状态信息
        A1, b1, status, message = self.rr(A, b)
        # 断言状态为0，表示行约简成功
        assert_equal(status, 0)
        # 断言约简后的矩阵 A1 的行数为39
        assert_equal(A1.shape[0], 39)
        # 断言约简后的矩阵 A1 的秩为39
        assert_equal(np.linalg.matrix_rank(A1), 39)
# 定义测试类 TestRRSVD，继承自 RRCommonTests 类
class TestRRSVD(RRCommonTests):
    # 定义 rr 方法，接受 A 和 b 两个参数
    def rr(self, A, b):
        # 调用 _remove_redundancy_svd 函数处理 A 和 b，返回处理结果
        return _remove_redundancy_svd(A, b)


# 定义测试类 TestRRPivotDense，继承自 RRCommonTests 类
class TestRRPivotDense(RRCommonTests):
    # 定义 rr 方法，接受 A 和 b 两个参数
    def rr(self, A, b):
        # 调用 _remove_redundancy_pivot_dense 函数处理 A 和 b，返回处理结果
        return _remove_redundancy_pivot_dense(A, b)


# 定义测试类 TestRRID，继承自 RRCommonTests 类
class TestRRID(RRCommonTests):
    # 定义 rr 方法，接受 A 和 b 两个参数
    def rr(self, A, b):
        # 调用 _remove_redundancy_id 函数处理 A 和 b，返回处理结果
        return _remove_redundancy_id(A, b)


# 定义测试类 TestRRPivotSparse，继承自 RRCommonTests 类
class TestRRPivotSparse(RRCommonTests):
    # 定义 rr 方法，接受 A 和 b 两个参数
    def rr(self, A, b):
        # 将稀疏矩阵 A 转换为压缩稀疏列格式，然后调用 _remove_redundancy_pivot_sparse 处理 A 和 b
        rr_res = _remove_redundancy_pivot_sparse(csc_matrix(A), b)
        # 将处理结果拆解为 A1, b1, status, message 四个变量
        A1, b1, status, message = rr_res
        # 将 A1 转换为普通的稠密数组，返回 A1, b1, status, message 四个值
        return A1.toarray(), b1, status, message
```