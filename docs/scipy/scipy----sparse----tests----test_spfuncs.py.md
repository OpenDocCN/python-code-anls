# `D:\src\scipysrc\scipy\scipy\sparse\tests\test_spfuncs.py`

```
# 导入必要的库函数和模块
from numpy import array, kron, diag
from numpy.testing import assert_, assert_equal

# 导入稀疏矩阵相关的函数和类
from scipy.sparse import _spfuncs as spfuncs
from scipy.sparse import csr_matrix, csc_matrix, bsr_matrix
from scipy.sparse._sparsetools import (csr_scale_rows, csr_scale_columns,
                                       bsr_scale_rows, bsr_scale_columns)

# 定义一个测试类 TestSparseFunctions，用于测试稀疏矩阵函数
class TestSparseFunctions:
    # 定义测试函数 test_scale_rows_and_cols，测试行和列的缩放操作
    def test_scale_rows_and_cols(self):
        # 创建一个二维数组 D
        D = array([[1, 0, 0, 2, 3],
                   [0, 4, 0, 5, 0],
                   [0, 0, 6, 7, 0]])

        # 使用 csr_matrix 将数组 D 转换为 CSR 稀疏矩阵 S
        S = csr_matrix(D)
        # 定义一个向量 v
        v = array([1, 2, 3])
        # 调用 csr_scale_rows 函数，对 S 进行行缩放操作
        csr_scale_rows(3, 5, S.indptr, S.indices, S.data, v)
        # 断言 S 转换为密集矩阵后，与对角矩阵 v@D 相等
        assert_equal(S.toarray(), diag(v) @ D)

        # 重新创建 CSR 稀疏矩阵 S 和向量 v
        S = csr_matrix(D)
        v = array([1, 2, 3, 4, 5])
        # 调用 csr_scale_columns 函数，对 S 进行列缩放操作
        csr_scale_columns(3, 5, S.indptr, S.indices, S.data, v)
        # 断言 S 转换为密集矩阵后，与矩阵 D@diag(v) 相等
        assert_equal(S.toarray(), D @ diag(v))

        # 创建一个二维数组 E，通过 kron 函数生成 D 的 Kronecker 乘积
        E = kron(D, [[1, 2], [3, 4]])
        # 使用 bsr_matrix 将数组 E 转换为 BSR 稀疏矩阵 S，设置块大小为 (2,2)
        S = bsr_matrix(E, blocksize=(2, 2))
        v = array([1, 2, 3, 4, 5, 6])
        # 调用 bsr_scale_rows 函数，对 S 进行行缩放操作
        bsr_scale_rows(3, 5, 2, 2, S.indptr, S.indices, S.data, v)
        # 断言 S 转换为密集矩阵后，与对角矩阵 v@E 相等
        assert_equal(S.toarray(), diag(v) @ E)

        # 重新创建 BSR 稀疏矩阵 S 和向量 v
        S = bsr_matrix(E, blocksize=(2, 2))
        v = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # 调用 bsr_scale_columns 函数，对 S 进行列缩放操作
        bsr_scale_columns(3, 5, 2, 2, S.indptr, S.indices, S.data, v)
        # 断言 S 转换为密集矩阵后，与矩阵 E@diag(v) 相等
        assert_equal(S.toarray(), E @ diag(v))

        # 创建一个二维数组 E，通过 kron 函数生成 D 的 Kronecker 乘积
        E = kron(D, [[1, 2, 3], [4, 5, 6]])
        # 使用 bsr_matrix 将数组 E 转换为 BSR 稀疏矩阵 S，设置块大小为 (2,3)
        S = bsr_matrix(E, blocksize=(2, 3))
        v = array([1, 2, 3, 4, 5, 6])
        # 调用 bsr_scale_rows 函数，对 S 进行行缩放操作
        bsr_scale_rows(3, 5, 2, 3, S.indptr, S.indices, S.data, v)
        # 断言 S 转换为密集矩阵后，与对角矩阵 v@E 相等
        assert_equal(S.toarray(), diag(v) @ E)

        # 重新创建 BSR 稀疏矩阵 S 和向量 v
        S = bsr_matrix(E, blocksize=(2, 3))
        v = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        # 调用 bsr_scale_columns 函数，对 S 进行列缩放操作
        bsr_scale_columns(3, 5, 2, 3, S.indptr, S.indices, S.data, v)
        # 断言 S 转换为密集矩阵后，与矩阵 E@diag(v) 相等
        assert_equal(S.toarray(), E @ diag(v))

    # 定义测试函数 test_estimate_blocksize，测试块大小的估算
    def test_estimate_blocksize(self):
        # 初始化几个测试用的二维数组和块数组
        mats = []
        mats.append([[0, 1], [1, 0]])
        mats.append([[1, 1, 0], [0, 0, 1], [1, 0, 1]])
        mats.append([[0], [0], [1]])
        mats = [array(x) for x in mats]

        blks = []
        blks.append([[1]])
        blks.append([[1, 1], [1, 1]])
        blks.append([[1, 1], [0, 1]])
        blks.append([[1, 1, 0], [1, 0, 1], [1, 1, 1]])
        blks = [array(x) for x in blks]

        # 遍历测试用的数组和块数组
        for A in mats:
            for B in blks:
                # 计算 A 和 B 的 Kronecker 乘积 X
                X = kron(A, B)
                # 使用 spfuncs.estimate_blocksize 估算 X 的块大小
                r, c = spfuncs.estimate_blocksize(X)
                # 断言行数 r 大于等于 B 的行数，列数 c 大于等于 B 的列数
                assert_(r >= B.shape[0])
                assert_(c >= B.shape[1])
    # 定义一个测试方法，用于测试 count_blocks 函数的准确性
    def test_count_blocks(self):
        # 定义一个参考函数 gold，用于计算矩阵 A 在给定块大小 bs 下的非零块数
        def gold(A, bs):
            R, C = bs
            # 找出矩阵 A 中非零元素的行列索引
            I, J = A.nonzero()
            # 使用块大小 R 和 C 对索引进行分块，并返回非零块的数量
            return len(set(zip(I // R, J // C)))

        # 创建一个测试用的矩阵列表 mats
        mats = []
        mats.append([[0]])
        mats.append([[1]])
        mats.append([[1, 0]])
        mats.append([[1, 1]])
        mats.append([[0, 1], [1, 0]])
        mats.append([[1, 1, 0], [0, 0, 1], [1, 0, 1]])
        mats.append([[0], [0], [1]])

        # 遍历矩阵列表 mats，对每对矩阵 A 和 B 进行以下操作
        for A in mats:
            for B in mats:
                # 计算 A 和 B 的 Kronecker 积 X
                X = kron(A, B)
                # 将 X 转换为稀疏矩阵类型 Y
                Y = csr_matrix(X)
                # 对每个可能的块大小 R 和 C 进行以下操作
                for R in range(1, 6):
                    for C in range(1, 6):
                        # 断言 count_blocks 函数对 Y 和块大小 (R, C) 的计算结果与 gold 函数的结果相等
                        assert_equal(spfuncs.count_blocks(Y, (R, C)), gold(X, (R, C)))

        # 创建一个特定的 Kronecker 积 X 和相应的稀疏矩阵 Y
        X = kron([[1, 1, 0], [0, 0, 1], [1, 0, 1]], [[1, 1]])
        Y = csc_matrix(X)
        # 断言 count_blocks 函数对 X 和块大小 (1, 2) 的计算结果与 gold 函数的结果相等
        assert_equal(spfuncs.count_blocks(X, (1, 2)), gold(X, (1, 2)))
        # 断言 count_blocks 函数对 Y 和块大小 (1, 2) 的计算结果与 gold 函数的结果相等
        assert_equal(spfuncs.count_blocks(Y, (1, 2)), gold(X, (1, 2)))
```