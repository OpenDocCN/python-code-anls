# `D:\src\scipysrc\scipy\scipy\optimize\_trustregion_constr\tests\test_projections.py`

```
import numpy as np  # 导入 NumPy 库，用于科学计算
import scipy.linalg  # 导入 SciPy 的线性代数模块
from scipy.sparse import csc_matrix  # 导入 SciPy 的稀疏矩阵模块中的 csc_matrix
from scipy.optimize._trustregion_constr.projections \
    import projections, orthogonality  # 从信任区域约束优化中导入 projections 和 orthogonality 函数
from numpy.testing import (TestCase, assert_array_almost_equal,
                           assert_equal, assert_allclose)  # 导入 NumPy 测试工具

try:
    from sksparse.cholmod import cholesky_AAt  # 尝试导入 sksparse 中的 cholesky_AAt 函数
    sksparse_available = True  # 如果导入成功，则设置 sksparse_available 为 True
    available_sparse_methods = ("NormalEquation", "AugmentedSystem")  # 可用的稀疏方法
except ImportError:
    sksparse_available = False  # 如果导入失败，则设置 sksparse_available 为 False
    available_sparse_methods = ("AugmentedSystem",)  # 只有一个可用的稀疏方法

available_dense_methods = ('QRFactorization', 'SVDFactorization')  # 可用的稠密方法


class TestProjections(TestCase):  # 定义测试类 TestProjections，继承自 TestCase

    def test_nullspace_and_least_squares_sparse(self):
        A_dense = np.array([[1, 2, 3, 4, 0, 5, 0, 7],  # 定义一个密集型 NumPy 数组 A_dense
                            [0, 8, 7, 0, 1, 5, 9, 0],
                            [1, 0, 0, 0, 0, 1, 2, 3]])
        At_dense = A_dense.T  # A_dense 的转置
        A = csc_matrix(A_dense)  # 将 A_dense 转换为稀疏的 csc_matrix 格式
        test_points = ([1, 2, 3, 4, 5, 6, 7, 8],  # 测试点列表，包含多个测试向量
                       [1, 10, 3, 0, 1, 6, 7, 8],
                       [1.12, 10, 0, 0, 100000, 6, 0.7, 8])

        for method in available_sparse_methods:  # 遍历可用的稀疏方法
            Z, LS, _ = projections(A, method)  # 调用 projections 函数获得 Z, LS 和 _ 结果
            for z in test_points:
                # Test if x is in the null_space
                x = Z.matvec(z)  # 计算 Z 乘以 z 的结果
                assert_array_almost_equal(A.dot(x), 0)  # 断言 A.dot(x) 结果近似于零
                # Test orthogonality
                assert_array_almost_equal(orthogonality(A, x), 0)  # 断言 A 和 x 正交的结果近似于零
                # Test if x is the least square solution
                x = LS.matvec(z)  # 计算 LS 乘以 z 的结果
                x2 = scipy.linalg.lstsq(At_dense, z)[0]  # 使用 scipy.linalg.lstsq 求解最小二乘解 x2
                assert_array_almost_equal(x, x2)  # 断言 x 和 x2 结果近似相等

    def test_iterative_refinements_sparse(self):
        A_dense = np.array([[1, 2, 3, 4, 0, 5, 0, 7],  # 定义一个密集型 NumPy 数组 A_dense
                            [0, 8, 7, 0, 1, 5, 9, 0],
                            [1, 0, 0, 0, 0, 1, 2, 3]])
        A = csc_matrix(A_dense)  # 将 A_dense 转换为稀疏的 csc_matrix 格式
        test_points = ([1, 2, 3, 4, 5, 6, 7, 8],  # 测试点列表，包含多个测试向量
                       [1, 10, 3, 0, 1, 6, 7, 8],
                       [1.12, 10, 0, 0, 100000, 6, 0.7, 8],
                       [1, 0, 0, 0, 0, 1, 2, 3+1e-10])

        for method in available_sparse_methods:  # 遍历可用的稀疏方法
            Z, LS, _ = projections(A, method, orth_tol=1e-18, max_refin=100)  # 调用 projections 函数
            for z in test_points:
                # Test if x is in the null_space
                x = Z.matvec(z)  # 计算 Z 乘以 z 的结果
                atol = 1e-13 * abs(x).max()  # 设置断言的绝对容差
                assert_allclose(A.dot(x), 0, atol=atol)  # 断言 A.dot(x) 结果近似于零
                # Test orthogonality
                assert_allclose(orthogonality(A, x), 0, atol=1e-13)  # 断言 A 和 x 正交的结果近似于零
    def test_rowspace_sparse(self):
        A_dense = np.array([[1, 2, 3, 4, 0, 5, 0, 7],  # 创建一个密集型（dense）的 NumPy 数组 A_dense
                            [0, 8, 7, 0, 1, 5, 9, 0],  # 第二行数据
                            [1, 0, 0, 0, 0, 1, 2, 3]])  # 第三行数据
        A = csc_matrix(A_dense)  # 将密集型数组 A_dense 转换为稀疏压缩列（CSC）格式的稀疏矩阵 A
        test_points = ([1, 2, 3],  # 测试点1
                       [1, 10, 3],  # 测试点2
                       [1.12, 10, 0])  # 测试点3

        for method in available_sparse_methods:  # 对于每个可用的稀疏方法循环
            _, _, Y = projections(A, method)  # 获取投影函数 projections 的输出
            for z in test_points:  # 对于每个测试点循环
                # 测试 x 是否是方程 A x = z 的解
                x = Y.matvec(z)  # 计算 Y 对 z 的乘积得到 x
                assert_array_almost_equal(A.dot(x), z)  # 断言 A.dot(x) 与 z 几乎相等
                # 测试 x 是否在 A 的行空间中
                A_ext = np.vstack((A_dense, x))  # 将 x 添加到 A_dense 的末尾形成 A_ext
                assert_equal(np.linalg.matrix_rank(A_dense),
                             np.linalg.matrix_rank(A_ext))  # 断言 A_dense 的秩与 A_ext 的秩相等

    def test_nullspace_and_least_squares_dense(self):
        A = np.array([[1, 2, 3, 4, 0, 5, 0, 7],  # 创建一个密集型（dense）的 NumPy 数组 A
                      [0, 8, 7, 0, 1, 5, 9, 0],  # 第二行数据
                      [1, 0, 0, 0, 0, 1, 2, 3]])  # 第三行数据
        At = A.T  # A 的转置矩阵
        test_points = ([1, 2, 3, 4, 5, 6, 7, 8],  # 测试点1
                       [1, 10, 3, 0, 1, 6, 7, 8],  # 测试点2
                       [1.12, 10, 0, 0, 100000, 6, 0.7, 8])  # 测试点3

        for method in available_dense_methods:  # 对于每个可用的密集方法循环
            Z, LS, _ = projections(A, method)  # 获取投影函数 projections 的输出
            for z in test_points:  # 对于每个测试点循环
                # 测试 x 是否在零空间中
                x = Z.matvec(z)  # 计算 Z 对 z 的乘积得到 x
                assert_array_almost_equal(A.dot(x), 0)  # 断言 A.dot(x) 几乎为零
                # 测试正交性
                assert_array_almost_equal(orthogonality(A, x), 0)  # 断言 A 和 x 的正交性几乎为零
                # 测试 x 是否是最小二乘解
                x = LS.matvec(z)  # 计算 LS 对 z 的乘积得到 x
                x2 = scipy.linalg.lstsq(At, z)[0]  # 使用最小二乘法计算得到 x2
                assert_array_almost_equal(x, x2)  # 断言 x 与 x2 几乎相等

    def test_compare_dense_and_sparse(self):
        D = np.diag(range(1, 101))  # 创建一个对角矩阵 D，对角线元素为 1 到 100
        A = np.hstack([D, D, D, D])  # 水平堆叠四个 D 形成 A
        A_sparse = csc_matrix(A)  # 将 A 转换为稀疏压缩列（CSC）格式的稀疏矩阵 A_sparse
        np.random.seed(0)  # 设置随机数种子

        Z, LS, Y = projections(A)  # 获取投影函数 projections 的输出
        Z_sparse, LS_sparse, Y_sparse = projections(A_sparse)  # 获取稀疏矩阵 A_sparse 的投影函数输出
        for k in range(20):  # 重复 20 次
            z = np.random.normal(size=(400,))  # 生成一个大小为 400 的正态分布随机向量 z
            assert_array_almost_equal(Z.dot(z), Z_sparse.dot(z))  # 断言 Z.dot(z) 与 Z_sparse.dot(z) 几乎相等
            assert_array_almost_equal(LS.dot(z), LS_sparse.dot(z))  # 断言 LS.dot(z) 与 LS_sparse.dot(z) 几乎相等
            x = np.random.normal(size=(100,))  # 生成一个大小为 100 的正态分布随机向量 x
            assert_array_almost_equal(Y.dot(x), Y_sparse.dot(x))  # 断言 Y.dot(x) 与 Y_sparse.dot(x) 几乎相等
    def test_compare_dense_and_sparse2(self):
        # 创建对角矩阵 D1, D2, D3
        D1 = np.diag([-1.7, 1, 0.5])
        D2 = np.diag([1, -0.6, -0.3])
        D3 = np.diag([-0.3, -1.5, 2])
        # 水平堆叠 D1, D2, D3 形成矩阵 A
        A = np.hstack([D1, D2, D3])
        # 将稠密矩阵 A 转换为稀疏的压缩列格式 (CSC) 矩阵 A_sparse
        A_sparse = csc_matrix(A)
        np.random.seed(0)

        # 进行投影操作，得到稠密矩阵的结果 Z, LS, Y
        Z, LS, Y = projections(A)
        # 对稀疏矩阵进行投影操作，得到结果 Z_sparse, LS_sparse, Y_sparse
        Z_sparse, LS_sparse, Y_sparse = projections(A_sparse)
        # 针对若干随机向量 z 和 x 进行断言比较
        for k in range(1):
            z = np.random.normal(size=(9,))
            # 断言稠密矩阵投影的结果与稀疏矩阵投影的结果近似相等
            assert_array_almost_equal(Z.dot(z), Z_sparse.dot(z))
            assert_array_almost_equal(LS.dot(z), LS_sparse.dot(z))
            x = np.random.normal(size=(3,))
            assert_array_almost_equal(Y.dot(x), Y_sparse.dot(x))

    def test_iterative_refinements_dense(self):
        # 定义测试用的稠密矩阵 A 和测试点 test_points
        A = np.array([[1, 2, 3, 4, 0, 5, 0, 7],
                      [0, 8, 7, 0, 1, 5, 9, 0],
                      [1, 0, 0, 0, 0, 1, 2, 3]])
        test_points = ([1, 2, 3, 4, 5, 6, 7, 8],
                       [1, 10, 3, 0, 1, 6, 7, 8],
                       [1, 0, 0, 0, 0, 1, 2, 3+1e-10])

        # 对每种可用的稠密方法进行迭代
        for method in available_dense_methods:
            # 进行投影操作，返回结果 Z, LS, _
            Z, LS, _ = projections(A, method, orth_tol=1e-18, max_refin=10)
            # 遍历测试点
            for z in test_points:
                # 计算 A*x 是否接近于零向量，检验 x 是否在零空间中
                x = Z.matvec(z)
                assert_allclose(A.dot(x), 0, rtol=0, atol=2.5e-14)
                # 检验正交性
                assert_allclose(orthogonality(A, x), 0, rtol=0, atol=5e-16)

    def test_rowspace_dense(self):
        # 定义测试用的稠密矩阵 A 和测试点 test_points
        A = np.array([[1, 2, 3, 4, 0, 5, 0, 7],
                      [0, 8, 7, 0, 1, 5, 9, 0],
                      [1, 0, 0, 0, 0, 1, 2, 3]])
        test_points = ([1, 2, 3],
                       [1, 10, 3],
                       [1.12, 10, 0])

        # 对每种可用的稠密方法进行迭代
        for method in available_dense_methods:
            # 进行投影操作，返回结果 _, _, Y
            _, _, Y = projections(A, method)
            # 遍历测试点
            for z in test_points:
                # 计算 A*x 是否接近于 z，检验 x 是否是方程 A*x = z 的解
                x = Y.matvec(z)
                assert_array_almost_equal(A.dot(x), z)
                # 检验 x 是否在 A 的行空间中
                A_ext = np.vstack((A, x))
                assert_equal(np.linalg.matrix_rank(A),
                             np.linalg.matrix_rank(A_ext))
# 定义一个测试类 TestOrthogonality，继承自 TestCase，用于测试 orthogonality 函数的功能
class TestOrthogonality(TestCase):

    # 定义测试方法 test_dense_matrix，用于测试在稠密矩阵上的正交性检验
    def test_dense_matrix(self):
        # 创建一个3x8的 NumPy 数组 A，表示一个稠密矩阵
        A = np.array([[1, 2, 3, 4, 0, 5, 0, 7],
                      [0, 8, 7, 0, 1, 5, 9, 0],
                      [1, 0, 0, 0, 0, 1, 2, 3]])
        # 定义两个测试向量，每个向量都包含8个浮点数
        test_vectors = ([-1.98931144, -1.56363389,
                         -0.84115584, 2.2864762,
                         5.599141, 0.09286976,
                         1.37040802, -0.28145812],
                        [697.92794044, -4091.65114008,
                         -3327.42316335, 836.86906951,
                         99434.98929065, -1285.37653682,
                         -4109.21503806, 2935.29289083])
        # 定义预期的正交性结果，这里是一个包含两个元素的元组
        test_expected_orth = (0, 0)

        # 遍历测试向量
        for i in range(len(test_vectors)):
            # 取出当前的测试向量 x
            x = test_vectors[i]
            # 取出当前的预期正交性结果 orth
            orth = test_expected_orth[i]
            # 调用 orthogonality 函数，并断言其结果与预期的正交性结果 orth 几乎相等
            assert_array_almost_equal(orthogonality(A, x), orth)

    # 定义测试方法 test_sparse_matrix，用于测试在稀疏矩阵上的正交性检验
    def test_sparse_matrix(self):
        # 创建一个3x8的稠密 NumPy 数组 A，然后将其转换为压缩稀疏列 (CSC) 格式的矩阵 A
        A = np.array([[1, 2, 3, 4, 0, 5, 0, 7],
                      [0, 8, 7, 0, 1, 5, 9, 0],
                      [1, 0, 0, 0, 0, 1, 2, 3]])
        A = csc_matrix(A)
        # 定义两个测试向量，每个向量都包含8个浮点数
        test_vectors = ([-1.98931144, -1.56363389,
                         -0.84115584, 2.2864762,
                         5.599141, 0.09286976,
                         1.37040802, -0.28145812],
                        [697.92794044, -4091.65114008,
                         -3327.42316335, 836.86906951,
                         99434.98929065, -1285.37653682,
                         -4109.21503806, 2935.29289083])
        # 定义预期的正交性结果，这里是一个包含两个元素的元组
        test_expected_orth = (0, 0)

        # 遍历测试向量
        for i in range(len(test_vectors)):
            # 取出当前的测试向量 x
            x = test_vectors[i]
            # 取出当前的预期正交性结果 orth
            orth = test_expected_orth[i]
            # 调用 orthogonality 函数，并断言其结果与预期的正交性结果 orth 几乎相等
            assert_array_almost_equal(orthogonality(A, x), orth)
```