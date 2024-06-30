# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\tests\test_linsolve.py`

```
import sys
import threading

import numpy as np
from numpy import array, finfo, arange, eye, all, unique, ones, dot
import numpy.random as random
from numpy.testing import (
        assert_array_almost_equal, assert_almost_equal,
        assert_equal, assert_array_equal, assert_, assert_allclose,
        assert_warns, suppress_warnings)
import pytest
from pytest import raises as assert_raises

import scipy.linalg
from scipy.linalg import norm, inv
from scipy.sparse import (spdiags, SparseEfficiencyWarning, csc_matrix,
        csr_matrix, identity, issparse, dok_matrix, lil_matrix, bsr_matrix)
from scipy.sparse.linalg import SuperLU
from scipy.sparse.linalg._dsolve import (spsolve, use_solver, splu, spilu,
        MatrixRankWarning, _superlu, spsolve_triangular, factorized)
import scipy.sparse

from scipy._lib._testutils import check_free_memory
from scipy._lib._util import ComplexWarning


# 创建一个警告过滤器，用于忽略稀疏矩阵效率警告
sup_sparse_efficiency = suppress_warnings()
sup_sparse_efficiency.filter(SparseEfficiencyWarning)

# 检查是否导入了 scikits.umfpack 模块，以确定是否使用 UMFPACK 求解器
try:
    import scikits.umfpack as umfpack
    has_umfpack = True
except ImportError:
    has_umfpack = False

# 将稀疏矩阵转换为稠密数组
def toarray(a):
    if issparse(a):
        return a.toarray()
    else:
        return a


# 设置用于测试的矩阵 A 和向量 b，解决 issue 8278
def setup_bug_8278():
    N = 2 ** 6
    h = 1/N
    # 创建一维 Ah1D 稀疏矩阵
    Ah1D = scipy.sparse.diags([-1, 2, -1], [-1, 0, 1],
                              shape=(N-1, N-1))/(h**2)
    eyeN = scipy.sparse.eye(N - 1)
    # 创建三维问题的稀疏矩阵 A
    A = (scipy.sparse.kron(eyeN, scipy.sparse.kron(eyeN, Ah1D))
         + scipy.sparse.kron(eyeN, scipy.sparse.kron(Ah1D, eyeN))
         + scipy.sparse.kron(Ah1D, scipy.sparse.kron(eyeN, eyeN)))
    # 创建随机的向量 b
    b = np.random.rand((N-1)**3)
    return A, b


# 测试因式分解的类
class TestFactorized:
    def setup_method(self):
        n = 5
        d = arange(n) + 1
        self.n = n
        # 创建一个对角占优的稀疏矩阵 A
        self.A = spdiags((d, 2*d, d[::-1]), (-3, 0, 5), n, n).tocsc()
        random.seed(1234)

    # 检查奇异情况的测试
    def _check_singular(self):
        A = csc_matrix((5,5), dtype='d')
        b = ones(5)
        # 断言 factorized(A)(b) 几乎等于零向量
        assert_array_almost_equal(0. * b, factorized(A)(b))

    # 检查非奇异情况的测试
    def _check_non_singular(self):
        # 创建一个对角线占优的随机稀疏矩阵 a
        n = 5
        a = csc_matrix(random.rand(n, n))
        b = ones(n)
        # 使用 splu 求解器解方程，验证 factorized(a)(b) 几乎等于期望值
        expected = splu(a).solve(b)
        assert_array_almost_equal(factorized(a)(b), expected)

    # 测试在没有 UMFPACK 的情况下奇异矩阵的测试
    def test_singular_without_umfpack(self):
        use_solver(useUmfpack=False)
        # 使用 assert_raises 断言捕获 RuntimeError 异常，验证因式分解是奇异的
        with assert_raises(RuntimeError, match="Factor is exactly singular"):
            self._check_singular()

    # 在有 UMFPACK 的情况下测试奇异矩阵的测试
    @pytest.mark.skipif(not has_umfpack, reason="umfpack not available")
    def test_singular_with_umfpack(self):
        use_solver(useUmfpack=True)
        # 使用 suppress_warnings 忽略特定的警告信息
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "divide by zero encountered in double_scalars")
            # 使用 assert_warns 断言捕获 UmfpackWarning 警告，验证因式分解是奇异的
            assert_warns(umfpack.UmfpackWarning, self._check_singular)
    # 定义测试函数，用于测试在未使用umfpack求解器时的非奇异性检查
    def test_non_singular_without_umfpack(self):
        # 设置使用umfpack参数为False
        use_solver(useUmfpack=False)
        # 调用检查非奇异性的辅助函数
        self._check_non_singular()

    # 标记为pytest跳过条件不满足时的测试函数
    @pytest.mark.skipif(not has_umfpack, reason="umfpack not available")
    # 测试在使用umfpack求解器时的非奇异性检查
    def test_non_singular_with_umfpack(self):
        # 设置使用umfpack参数为True
        use_solver(useUmfpack=True)
        # 调用检查非奇异性的辅助函数
        self._check_non_singular()

    # 测试在未使用umfpack求解器时，对非方阵矩阵无法因子化的情况
    def test_cannot_factorize_nonsquare_matrix_without_umfpack(self):
        # 设置使用umfpack参数为False
        use_solver(useUmfpack=False)
        # 定义异常消息
        msg = "can only factor square matrices"
        # 使用断言检查是否抛出预期的值错误异常，并匹配特定消息
        with assert_raises(ValueError, match=msg):
            factorized(self.A[:, :4])

    # 标记为pytest跳过条件不满足时的测试函数
    @pytest.mark.skipif(not has_umfpack, reason="umfpack not available")
    # 测试在使用umfpack求解器时，对非方阵矩阵进行因子化操作
    def test_factorizes_nonsquare_matrix_with_umfpack(self):
        # 设置使用umfpack参数为True
        use_solver(useUmfpack=True)
        # 执行因子化操作，验证不会抛出异常
        factorized(self.A[:,:4])

    # 测试在未使用umfpack求解器时，对尺寸不匹配的矩阵调用求解器的情况
    def test_call_with_incorrectly_sized_matrix_without_umfpack(self):
        # 设置使用umfpack参数为False
        use_solver(useUmfpack=False)
        # 执行矩阵因子化操作并获取其解决方案
        solve = factorized(self.A)
        # 创建不同尺寸的随机向量和矩阵
        b = random.rand(4)
        B = random.rand(4, 3)
        BB = random.rand(self.n, 3, 9)

        # 使用断言检查是否抛出预期的值错误异常，并匹配特定消息
        with assert_raises(ValueError, match="is of incompatible size"):
            solve(b)
        with assert_raises(ValueError, match="is of incompatible size"):
            solve(B)
        with assert_raises(ValueError,
                           match="object too deep for desired array"):
            solve(BB)

    # 标记为pytest跳过条件不满足时的测试函数
    @pytest.mark.skipif(not has_umfpack, reason="umfpack not available")
    # 测试在使用umfpack求解器时，对尺寸不匹配的矩阵调用求解器的情况
    def test_call_with_incorrectly_sized_matrix_with_umfpack(self):
        # 设置使用umfpack参数为True
        use_solver(useUmfpack=True)
        # 执行矩阵因子化操作并获取其解决方案
        solve = factorized(self.A)
        # 创建不同尺寸的随机向量和矩阵
        b = random.rand(4)
        B = random.rand(4, 3)
        BB = random.rand(self.n, 3, 9)

        # 验证调用不会抛出异常
        solve(b)
        # 定义异常消息
        msg = "object too deep for desired array"
        # 使用断言检查是否抛出预期的值错误异常，并匹配特定消息
        with assert_raises(ValueError, match=msg):
            solve(B)
        with assert_raises(ValueError, match=msg):
            solve(BB)

    # 测试在未使用umfpack求解器时，对转换为复数的矩阵调用求解器的情况
    def test_call_with_cast_to_complex_without_umfpack(self):
        # 设置使用umfpack参数为False
        use_solver(useUmfpack=False)
        # 执行矩阵因子化操作并获取其解决方案
        solve = factorized(self.A)
        # 创建随机向量
        b = random.rand(4)
        # 遍历复数类型数组
        for t in [np.complex64, np.complex128]:
            # 使用断言检查是否抛出预期的类型错误异常，并匹配特定消息
            with assert_raises(TypeError, match="Cannot cast array data"):
                solve(b.astype(t))

    # 标记为pytest跳过条件不满足时的测试函数
    @pytest.mark.skipif(not has_umfpack, reason="umfpack not available")
    # 测试在使用umfpack求解器时，对转换为复数的矩阵调用求解器的情况
    def test_call_with_cast_to_complex_with_umfpack(self):
        # 设置使用umfpack参数为True
        use_solver(useUmfpack=True)
        # 执行矩阵因子化操作并获取其解决方案
        solve = factorized(self.A)
        # 创建随机向量
        b = random.rand(4)
        # 遍历复数类型数组
        for t in [np.complex64, np.complex128]:
            # 使用断言检查是否产生复杂警告，调用solve函数
            assert_warns(ComplexWarning, solve, b.astype(t))

    # 标记为pytest跳过条件不满足时的测试函数
    @pytest.mark.skipif(not has_umfpack, reason="umfpack not available")
    # 定义一个测试函数，用于测试假设排序索引标志
    def test_assume_sorted_indices_flag(self):
        # 创建一个稀疏矩阵，其中包含未排序的索引
        unsorted_inds = np.array([2, 0, 1, 0])
        data = np.array([10, 16, 5, 0.4])
        indptr = np.array([0, 1, 2, 4])
        # 使用给定的数据创建一个压缩列格式 (CSC) 的稀疏矩阵 A
        A = csc_matrix((data, unsorted_inds, indptr), (3, 3))
        # 创建一个大小为 3 的单位向量 b
        b = ones(3)

        # 当假设索引已排序时，应当引发错误
        use_solver(useUmfpack=True, assumeSortedIndices=True)
        # 使用 assert_raises 来捕获期望的 RuntimeError，并匹配给定的错误消息
        with assert_raises(RuntimeError,
                           match="UMFPACK_ERROR_invalid_matrix"):
            factorized(A)

        # 当不假设索引已排序时，应对索引进行排序并成功
        use_solver(useUmfpack=True, assumeSortedIndices=False)
        # 计算使用 LU 分解求解的期望值
        expected = splu(A.copy()).solve(b)

        # 断言稀疏矩阵 A 的排序索引标志为 0 (未排序)
        assert_equal(A.has_sorted_indices, 0)
        # 使用 factorized(A) 对 b 进行求解，并断言结果与预期接近
        assert_array_almost_equal(factorized(A)(b), expected)

    @pytest.mark.slow
    @pytest.mark.skipif(not has_umfpack, reason="umfpack not available")
    # 定义一个测试函数，用于测试 Bug 8278
    def test_bug_8278(self):
        # 检查系统空闲内存是否足够（至少 8000 MB）
        check_free_memory(8000)
        # 使用 UMFPACK 求解器
        use_solver(useUmfpack=True)
        # 设置 Bug 8278 的测试数据 A 和 b
        A, b = setup_bug_8278()
        # 将 A 转换为压缩列格式 (CSC)
        A = A.tocsc()
        # 对 A 进行因子分解
        f = factorized(A)
        # 使用因子化后的对象 f 对 b 进行求解
        x = f(b)
        # 断言 A @ x 与 b 的近似相等
        assert_array_almost_equal(A @ x, b)
class TestLinsolve:
    # 在每个测试方法运行之前调用，设置使用 UMFPACK 求解器
    def setup_method(self):
        use_solver(useUmfpack=False)

    # 测试奇异矩阵情况
    def test_singular(self):
        # 创建一个空的稀疏矩阵 A，形状为 5x5
        A = csc_matrix((5,5), dtype='d')
        # 创建一个数组 b，包含元素 [1, 2, 3, 4, 5]
        b = array([1, 2, 3, 4, 5], dtype='d')
        # 使用 suppress_warnings 上下文管理器，过滤 MatrixRankWarning 警告
        with suppress_warnings() as sup:
            sup.filter(MatrixRankWarning, "Matrix is exactly singular")
            # 使用 spsolve 求解稀疏矩阵 A 的线性方程组 Ax=b
            x = spsolve(A, b)
        # 断言 x 中没有非有限数值
        assert_(not np.isfinite(x).any())

    # 测试特殊的奇异情况，应当引发 RuntimeError 异常
    def test_singular_gh_3312(self):
        # 定义一个坐标数组 ij 和值数组 v，创建稀疏矩阵 A
        ij = np.array([(17, 0), (17, 6), (17, 12), (10, 13)], dtype=np.int32)
        v = np.array([0.284213, 0.94933781, 0.15767017, 0.38797296])
        A = csc_matrix((v, ij.T), shape=(20, 20))
        # 创建数组 b，包含元素 [0, 1, ..., 19]
        b = np.arange(20)

        try:
            # 使用 suppress_warnings 上下文管理器，过滤 MatrixRankWarning 警告
            with suppress_warnings() as sup:
                sup.filter(MatrixRankWarning, "Matrix is exactly singular")
                # 使用 spsolve 求解稀疏矩阵 A 的线性方程组 Ax=b
                x = spsolve(A, b)
            # 断言 x 中没有非有限数值
            assert not np.isfinite(x).any()
        except RuntimeError:
            pass

    # 使用参数化测试格式和索引类型来测试两对角矩阵的解法
    @pytest.mark.parametrize('format', ['csc', 'csr'])
    @pytest.mark.parametrize('idx_dtype', [np.int32, np.int64])
    def test_twodiags(self, format: str, idx_dtype: np.dtype):
        # 创建两对角稀疏矩阵 A
        A = spdiags([[1, 2, 3, 4, 5], [6, 5, 8, 9, 10]], [0, 1], 5, 5,
                    format=format)
        # 创建数组 b，包含元素 [1, 2, 3, 4, 5]
        b = array([1, 2, 3, 4, 5])

        # 计算矩阵 A 的条件数
        cond_A = norm(A.toarray(), 2) * norm(inv(A.toarray()), 2)

        # 对于每种浮点数类型 t 进行测试
        for t in ['f','d','F','D']:
            # 获取浮点数类型 t 的机器精度 epsilon
            eps = finfo(t).eps  # floating point epsilon
            # 将数组 b 和矩阵 A 转换为浮点数类型 t
            b = b.astype(t)
            Asp = A.astype(t)
            # 将稀疏矩阵 A 的索引和指针数组转换为 idx_dtype 类型
            Asp.indices = Asp.indices.astype(idx_dtype, copy=False)
            Asp.indptr = Asp.indptr.astype(idx_dtype, copy=False)

            # 使用 spsolve 求解稀疏矩阵 Asp 的线性方程组 Asp*x=b
            x = spsolve(Asp, b)
            # 断言 Ax=b 的解 x 的误差小于 10 * cond_A * eps
            assert_(norm(b - Asp@x) < 10 * cond_A * eps)

    # 测试稠密矩阵向量乘法的情况
    def test_bvector_smoketest(self):
        # 创建稠密矩阵 Adense
        Adense = array([[0., 1., 1.],
                        [1., 0., 1.],
                        [0., 0., 1.]])
        # 将 Adense 转换为稀疏矩阵 As
        As = csc_matrix(Adense)
        # 设置随机种子
        random.seed(1234)
        # 创建随机数组 x
        x = random.randn(3)
        # 计算 b = As * x
        b = As@x
        # 使用 spsolve 求解稀疏矩阵 As 的线性方程组 As*x=b
        x2 = spsolve(As, b)

        # 断言 x 和 x2 很接近
        assert_array_almost_equal(x, x2)

    # 测试稠密矩阵乘法的情况
    def test_bmatrix_smoketest(self):
        # 创建稠密矩阵 Adense
        Adense = array([[0., 1., 1.],
                        [1., 0., 1.],
                        [0., 0., 1.]])
        # 将 Adense 转换为稀疏矩阵 As
        As = csc_matrix(Adense)
        # 设置随机种子
        random.seed(1234)
        # 创建随机矩阵 x，形状为 3x4
        x = random.randn(3, 4)
        # 计算稠密矩阵乘法 Bdense = As * x
        Bdense = As.dot(x)
        # 将 Bdense 转换为稀疏矩阵 Bs
        Bs = csc_matrix(Bdense)
        # 使用 spsolve 求解稀疏矩阵 As 的线性方程组 As*x=Bs
        x2 = spsolve(As, Bs)
        # 断言 x 和 x2 很接近
        assert_array_almost_equal(x, x2.toarray())
    # 定义一个测试函数，用于测试非方阵情况下的解决方案
    def test_non_square(self):
        # 创建一个形状为 (3, 4) 的全一矩阵 A
        A = ones((3, 4))
        # 创建一个形状为 (4, 1) 的全一矩阵 b
        b = ones((4, 1))
        # 断言：当 A 不是方阵时，调用 spsolve 函数会引发 ValueError 异常
        assert_raises(ValueError, spsolve, A, b)
        
        # 创建一个稀疏矩阵 A2，其形状为 (3, 3)，是单位矩阵的压缩列存储形式
        A2 = csc_matrix(eye(3))
        # 创建一个形状为 (2,) 的数组 b2
        b2 = array([1.0, 2.0])
        # 断言：当 A2 和 b2 的形状不兼容时，调用 spsolve 函数会引发 ValueError 异常
        assert_raises(ValueError, spsolve, A2, b2)

    # 使用 @sup_sparse_efficiency 装饰器标记的测试函数，用于比较例子
    def test_example_comparison(self):
        # 定义稀疏矩阵 sM 的行索引、列索引和数据
        row = array([0,0,1,2,2,2])
        col = array([0,2,2,0,1,2])
        data = array([1,2,3,-4,5,6])
        # 从行索引、列索引和数据创建稀疏矩阵 sM，形状为 (3, 3)，数据类型为 float
        sM = csr_matrix((data,(row,col)), shape=(3,3), dtype=float)
        # 将稀疏矩阵 sM 转换为密集矩阵 M
        M = sM.toarray()

        # 定义稀疏矩阵 sN 的行索引、列索引和数据
        row = array([0,0,1,1,0,0])
        col = array([0,2,1,1,0,0])
        data = array([1,1,1,1,1,1])
        # 从行索引、列索引和数据创建稀疏矩阵 sN，形状为 (3, 3)，数据类型为 float
        sN = csr_matrix((data, (row,col)), shape=(3,3), dtype=float)
        # 将稀疏矩阵 sN 转换为密集矩阵 N
        N = sN.toarray()

        # 使用 spsolve 函数解 sM * X = sN，得到稀疏矩阵 sX
        sX = spsolve(sM, sN)
        # 使用 scipy.linalg.solve 函数解 M * X = N，得到密集矩阵 X
        X = scipy.linalg.solve(M, N)

        # 断言：密集矩阵 X 和稀疏矩阵 sX 的数值近似相等
        assert_array_almost_equal(X, sX.toarray())

    # 使用 @sup_sparse_efficiency 装饰器标记的测试函数，如果没有 umfpack，则跳过该测试
    @sup_sparse_efficiency
    @pytest.mark.skipif(not has_umfpack, reason="umfpack not available")
    # 测试稀疏矩阵解决方案的形状兼容性
    def test_shape_compatibility(self):
        # 使用 UMFPACK 求解器
        use_solver(useUmfpack=True)
        # 创建一个稀疏的压缩列矩阵 A
        A = csc_matrix([[1., 0], [0, 2]])
        # 准备多个不同类型的右侧向量 bs
        bs = [
            [1, 6],                                # Python 列表
            array([1, 6]),                          # NumPy 数组
            [[1], [6]],                             # 嵌套 Python 列表
            array([[1], [6]]),                      # NumPy 二维数组
            csc_matrix([[1], [6]]),                 # 稀疏压缩列矩阵
            csr_matrix([[1], [6]]),                 # 稀疏压缩行矩阵
            dok_matrix([[1], [6]]),                 # 稀疏字典矩阵
            bsr_matrix([[1], [6]]),                 # 块压缩行矩阵
            array([[1., 2., 3.], [6., 8., 10.]]),    # NumPy 二维数组
            csc_matrix([[1., 2., 3.], [6., 8., 10.]]),  # 稀疏压缩列矩阵
            csr_matrix([[1., 2., 3.], [6., 8., 10.]]),  # 稀疏压缩行矩阵
            dok_matrix([[1., 2., 3.], [6., 8., 10.]]),  # 稀疏字典矩阵
            bsr_matrix([[1., 2., 3.], [6., 8., 10.]]),  # 块压缩行矩阵
        ]

        # 遍历所有的 bs 向量
        for b in bs:
            # 使用 numpy.linalg.solve 求解稀疏矩阵 A 和 b 的方程，转换为密集数组
            x = np.linalg.solve(A.toarray(), toarray(b))
            # 遍历不同的稀疏矩阵类型
            for spmattype in [csc_matrix, csr_matrix, dok_matrix, lil_matrix]:
                # 使用 spsolve 函数解决方程，使用 UMFPACK 加速
                x1 = spsolve(spmattype(A), b, use_umfpack=True)
                x2 = spsolve(spmattype(A), b, use_umfpack=False)

                # 检查解的一致性
                if x.ndim == 2 and x.shape[1] == 1:
                    # 将二维数组视为“向量”
                    x = x.ravel()

                # 断言 x1 和 x2 的数组近似相等
                assert_array_almost_equal(toarray(x1), x,
                                          err_msg=repr((b, spmattype, 1)))
                assert_array_almost_equal(toarray(x2), x,
                                          err_msg=repr((b, spmattype, 2)))

                # 密集 vs 稀疏输出的检查（“向量”始终是密集的）
                if issparse(b) and x.ndim > 1:
                    assert_(issparse(x1), repr((b, spmattype, 1)))
                    assert_(issparse(x2), repr((b, spmattype, 2)))
                else:
                    assert_(isinstance(x1, np.ndarray), repr((b, spmattype, 1)))
                    assert_(isinstance(x2, np.ndarray), repr((b, spmattype, 2)))

                # 检查输出的形状
                if x.ndim == 1:
                    # “向量”
                    assert_equal(x1.shape, (A.shape[1],))
                    assert_equal(x2.shape, (A.shape[1],))
                else:
                    # “矩阵”
                    assert_equal(x1.shape, x.shape)
                    assert_equal(x2.shape, x.shape)

        # 当 A 和 b 的形状不匹配时，断言会引发 ValueError
        A = csc_matrix((3, 3))
        b = csc_matrix((1, 3))
        assert_raises(ValueError, spsolve, A, b)

    # 使用 sup_sparse_efficiency 修饰器测试 ndarray 支持
    @sup_sparse_efficiency
    def test_ndarray_support(self):
        # 创建一个二维数组 A
        A = array([[1., 2.], [2., 0.]])
        # 创建一个二维数组 x
        x = array([[1., 1.], [0.5, -0.5]])
        # 创建一个二维数组 b
        b = array([[2., 0.], [2., 2.]])

        # 断言通过 spsolve 函数解 A 和 b 的方程得到的结果与 x 近似相等
        assert_array_almost_equal(x, spsolve(A, b))
    # 定义一个测试函数，用于测试在特定输入条件下的 _superlu.gssv 函数的行为
    def test_gssv_badinput(self):
        # 定义一个整数 N
        N = 10
        # 创建一个 N 元素的数组 d，数组元素为 1.0 到 N 的序列
        d = arange(N) + 1.0
        # 构造一个稀疏对角矩阵 A，使用 spdiags 函数，对角线的偏移为 (-3, 0, 5)，大小为 N x N
        A = spdiags((d, 2*d, d[::-1]), (-3, 0, 5), N, N)

        # 遍历稀疏矩阵类型 csc_matrix 和 csr_matrix
        for spmatrix in (csc_matrix, csr_matrix):
            # 将 A 转换为当前稀疏矩阵类型的对象
            A = spmatrix(A)
            # 创建一个长度为 N 的数组 b，元素为 0 到 N-1
            b = np.arange(N)

            # 定义几个函数，用于生成不良输入条件
            def not_c_contig(x):
                return x.repeat(2)[::2]

            def not_1dim(x):
                return x[:,None]

            def bad_type(x):
                return x.astype(bool)

            def too_short(x):
                return x[:-1]

            # 将不良操作函数组成的列表存储在 badops 变量中
            badops = [not_c_contig, not_1dim, bad_type, too_short]

            # 遍历每个不良操作函数
            for badop in badops:
                # 格式化消息字符串，描述当前测试条件
                msg = f"{spmatrix!r} {badop!r}"
                # 断言调用 _superlu.gssv 函数时会引发 ValueError 或 TypeError 异常，检查不良操作函数在不同参数上的影响
                assert_raises((ValueError, TypeError), _superlu.gssv,
                              N, A.nnz, badop(A.data), A.indices, A.indptr,
                              b, int(spmatrix == csc_matrix), err_msg=msg)
                assert_raises((ValueError, TypeError), _superlu.gssv,
                              N, A.nnz, A.data, badop(A.indices), A.indptr,
                              b, int(spmatrix == csc_matrix), err_msg=msg)
                assert_raises((ValueError, TypeError), _superlu.gssv,
                              N, A.nnz, A.data, A.indices, badop(A.indptr),
                              b, int(spmatrix == csc_matrix), err_msg=msg)

    # 测试函数，验证在求解线性方程组时是否能够保持稀疏矩阵的稀疏性
    def test_sparsity_preservation(self):
        # 创建一个单位稀疏矩阵 ident
        ident = csc_matrix([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]])
        # 创建一个稀疏矩阵 b
        b = csc_matrix([
            [0, 1],
            [1, 0],
            [0, 0]])
        # 解方程 ident * x = b，得到解 x
        x = spsolve(ident, b)
        # 断言 ident 矩阵的非零元素个数为 3
        assert_equal(ident.nnz, 3)
        # 断言 b 矩阵的非零元素个数为 2
        assert_equal(b.nnz, 2)
        # 断言解 x 的非零元素个数为 2
        assert_equal(x.nnz, 2)
        # 断言解 x 的数值数组近似等于 b 的数值数组，允许的误差为 1e-12
        assert_allclose(x.toarray(), b.toarray(), atol=1e-12, rtol=1e-12)

    # 测试函数，验证在不同数据类型下求解线性方程组时的数据类型转换
    def test_dtype_cast(self):
        # 创建一个实数的稀疏矩阵 A_real
        A_real = scipy.sparse.csr_matrix([[1, 2, 0],
                                          [0, 0, 3],
                                          [4, 0, 5]])
        # 创建一个复数的稀疏矩阵 A_complex
        A_complex = scipy.sparse.csr_matrix([[1, 2, 0],
                                             [0, 0, 3],
                                             [4, 0, 5 + 1j]])
        # 创建实数类型的向量 b_real 和复数类型的向量 b_complex
        b_real = np.array([1,1,1])
        b_complex = np.array([1,1,1]) + 1j*np.array([1,1,1])

        # 解方程 A_real * x = b_real，检查解 x 的数据类型是否为浮点数
        x = spsolve(A_real, b_real)
        assert_(np.issubdtype(x.dtype, np.floating))

        # 解方程 A_real * x = b_complex，检查解 x 的数据类型是否为复数
        x = spsolve(A_real, b_complex)
        assert_(np.issubdtype(x.dtype, np.complexfloating))

        # 解方程 A_complex * x = b_real，检查解 x 的数据类型是否为复数
        x = spsolve(A_complex, b_real)
        assert_(np.issubdtype(x.dtype, np.complexfloating))

        # 解方程 A_complex * x = b_complex，检查解 x 的数据类型是否为复数
        x = spsolve(A_complex, b_complex)
        assert_(np.issubdtype(x.dtype, np.complexfloating))

    # 标记为慢速测试，并且仅在 umfpack 可用时才执行的测试函数，验证特定 bug 的修复情况
    @pytest.mark.slow
    @pytest.mark.skipif(not has_umfpack, reason="umfpack not available")
    def test_bug_8278(self):
        # 检查系统剩余内存是否足够，至少需要 8000 单位
        check_free_memory(8000)
        # 使用 umfpack 求解器来解决线性方程组
        use_solver(useUmfpack=True)
        # 设置特定 bug 8278 的测试数据 A 和 b
        A, b = setup_bug_8278()
        # 解方程 A * x = b，得到解 x
        x = spsolve(A, b)
        # 断言解 x 满足 A * x = b 的近似精度
        assert_array_almost_equal(A @ x, b)
class TestSplu:
    # 设置方法，初始化测试环境
    def setup_method(self):
        # 关闭 UMFPACK 求解器
        use_solver(useUmfpack=False)
        # 设定矩阵维度为 40
        n = 40
        # 创建对角线数组
        d = arange(n) + 1
        # 将 n 和 A 分别设为实例属性
        self.n = n
        # 创建稀疏对角矩阵 A
        self.A = spdiags((d, 2*d, d[::-1]), (-3, 0, 5), n, n, format='csc')
        # 设定随机数种子
        random.seed(1234)

    # 执行矩阵 LU 分解的核心测试方法
    def _smoketest(self, spxlu, check, dtype, idx_dtype):
        # 如果数据类型是复数类型，则创建复数类型的矩阵 A
        if np.issubdtype(dtype, np.complexfloating):
            A = self.A + 1j*self.A.T
        else:
            A = self.A

        # 将矩阵 A 转换为指定的数据类型
        A = A.astype(dtype)
        # 将 A 的 indices 属性转换为指定的索引数据类型
        A.indices = A.indices.astype(idx_dtype, copy=False)
        # 将 A 的 indptr 属性转换为指定的索引数据类型
        A.indptr = A.indptr.astype(idx_dtype, copy=False)
        # 执行 LU 分解
        lu = spxlu(A)

        # 创建随机数生成器
        rng = random.RandomState(1234)

        # 测试不同的输入形状
        for k in [None, 1, 2, self.n, self.n+2]:
            msg = f"k={k!r}"

            # 根据 k 的值生成对应形状的随机向量 b
            if k is None:
                b = rng.rand(self.n)
            else:
                b = rng.rand(self.n, k)

            # 如果数据类型是复数类型，则给随机向量 b 添加虚部
            if np.issubdtype(dtype, np.complexfloating):
                b = b + 1j*rng.rand(*b.shape)
            # 将 b 转换为指定的数据类型
            b = b.astype(dtype)

            # 解方程 lu.solve(b)，并检查结果
            x = lu.solve(b)
            check(A, b, x, msg)

            # 解方程 lu.solve(b, 'T')，并检查结果
            x = lu.solve(b, 'T')
            check(A.T, b, x, msg)

            # 解方程 lu.solve(b, 'H')，并检查结果
            x = lu.solve(b, 'H')
            check(A.T.conj(), b, x, msg)

    # 用于测试 splu 的性能效率修饰器
    @sup_sparse_efficiency
    # 执行 splu 的核心测试方法
    def test_splu_smoketest(self):
        self._internal_test_splu_smoketest()

    # 内部方法，实际执行测试 splu 的核心方法
    def _internal_test_splu_smoketest(self):
        # 检查 splu 是否正常工作
        def check(A, b, x, msg=""):
            # 获取 A 的浮点数精度
            eps = np.finfo(A.dtype).eps
            # 计算残差 r
            r = A @ x
            # 断言残差的最大值小于给定的误差限
            assert_(abs(r - b).max() < 1e3*eps, msg)

        # 遍历不同的数据类型和索引数据类型进行测试
        for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
            for idx_dtype in [np.int32, np.int64]:
                self._smoketest(splu, check, dtype, idx_dtype)

    # 用于测试 spilu 的性能效率修饰器
    @sup_sparse_efficiency
    # 执行 spilu 的核心测试方法
    def test_spilu_smoketest(self):
        self._internal_test_spilu_smoketest()

    # 内部方法，实际执行测试 spilu 的核心方法
    def _internal_test_spilu_smoketest(self):
        # 错误列表，用于记录错误
        errors = []

        # 检查 spilu 的方法
        def check(A, b, x, msg=""):
            # 计算残差 r
            r = A @ x
            # 计算残差的最大值
            err = abs(r - b).max()
            # 断言残差的最大值小于给定的误差限
            assert_(err < 1e-2, msg)
            # 如果 b 的数据类型是 float64 或 complex128，则记录错误
            if b.dtype in (np.float64, np.complex128):
                errors.append(err)

        # 遍历不同的数据类型和索引数据类型进行测试
        for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
            for idx_dtype in [np.int32, np.int64]:
                self._smoketest(spilu, check, dtype, idx_dtype)

        # 断言最大错误大于给定的阈值
        assert_(max(errors) > 1e-5)

    # 用于测试 spilu 的性能效率修饰器
    @sup_sparse_efficiency
    # 执行测试 spilu 的方法，测试传递 drop_rule 参数
    def test_spilu_drop_rule(self):
        # 创建单位矩阵 A
        A = identity(2)

        # 不同的 drop_rule 规则列表
        rules = [
            b'basic,area'.decode('ascii'),  # 转换为 Unicode
            b'basic,area',  # ASCII
            [b'basic', b'area'.decode('ascii')]  # 列表形式
        ]
        # 遍历规则列表进行测试
        for rule in rules:
            # 断言传递规则参数后返回值是 SuperLU 类的实例
            assert_(isinstance(spilu(A, drop_rule=rule), SuperLU))

    # 测试 splu 处理稀疏矩阵 nnz=0 的情况
    def test_splu_nnz0(self):
        # 创建一个空的 5x5 的 csc 矩阵
        A = csc_matrix((5,5), dtype='d')
        # 断言调用 splu(A) 会抛出 RuntimeError 异常
        assert_raises(RuntimeError, splu, A)
    def test_spilu_nnz0(self):
        # 创建一个 5x5 的稀疏矩阵 A，数据类型为双精度浮点数
        A = csc_matrix((5,5), dtype='d')
        # 断言调用 spilu 函数时会引发 RuntimeError 异常，因为 A 是零矩阵
        assert_raises(RuntimeError, spilu, A)

    def test_splu_basic(self):
        # 测试 splu 基本功能
        n = 30
        rng = random.RandomState(12)
        # 创建一个 n x n 的随机数组 a
        a = rng.rand(n, n)
        # 将数组 a 中小于 0.95 的元素设为 0，生成稀疏矩阵 a_
        a[a < 0.95] = 0
        # 让矩阵 a 成为奇异矩阵的一个测试案例
        a[:, 0] = 0
        a_ = csc_matrix(a)
        # 断言调用 splu 函数时会引发 RuntimeError 异常，因为 a_ 是奇异的
        assert_raises(RuntimeError, splu, a_)

        # 将 a 变为对角占优，确保它不是奇异的
        a += 4*eye(n)
        a_ = csc_matrix(a)
        # 对 a_ 进行 LU 分解
        lu = splu(a_)
        b = ones(n)
        # 解方程 lu*x = b，验证解 x 是否满足 dot(a, x) ≈ b
        x = lu.solve(b)
        assert_almost_equal(dot(a, x), b)

    def test_splu_perm(self):
        # 测试 splu 暴露的排列向量
        n = 30
        a = random.random((n, n))
        a[a < 0.95] = 0
        # 将 a 变为对角占优，确保它不是奇异的
        a += 4*eye(n)
        a_ = csc_matrix(a)
        # 对 a_ 进行 LU 分解
        lu = splu(a_)
        # 检查排列索引是否属于 [0, n-1]
        for perm in (lu.perm_r, lu.perm_c):
            assert_(all(perm > -1))
            assert_(all(perm < n))
            assert_equal(len(unique(perm)), len(perm))

        # 现在使 a 对称，并测试两个排列向量是否相同
        # 注意：a += a.T 依赖于未定义的行为。
        a = a + a.T
        a_ = csc_matrix(a)
        lu = splu(a_)
        assert_array_equal(lu.perm_r, lu.perm_c)

    @pytest.mark.parametrize("splu_fun, rtol", [(splu, 1e-7), (spilu, 1e-1)])
    def test_natural_permc(self, splu_fun, rtol):
        # 测试 "NATURAL" permc_spec 不会对矩阵进行置换
        np.random.seed(42)
        n = 500
        p = 0.01
        # 创建一个稀疏随机矩阵 A
        A = scipy.sparse.random(n, n, p)
        x = np.random.rand(n)
        # 将 A 变为对角占优，确保它不是奇异的
        A += (n+1)*scipy.sparse.identity(n)
        A_ = csc_matrix(A)
        b = A_ @ x

        # 没有 permc_spec 时，排列不是恒等排列
        lu = splu_fun(A_)
        assert_(np.any(lu.perm_c != np.arange(n)))

        # 使用 permc_spec="NATURAL"，排列是恒等排列
        lu = splu_fun(A_, permc_spec="NATURAL")
        assert_array_equal(lu.perm_c, np.arange(n))

        # 同时，lu 分解是有效的
        x2 = lu.solve(b)
        assert_allclose(x, x2, rtol=rtol)

    @pytest.mark.skipif(not hasattr(sys, 'getrefcount'), reason="no sys.getrefcount")
    def test_lu_refcount(self):
        # 测试我们是否正确地跟踪了使用 splu 时的引用计数。
        n = 30
        # 创建一个 n x n 的随机矩阵 a
        a = random.random((n, n))
        # 将矩阵中小于 0.95 的元素置为 0，使其对角线占优，确保不是奇异矩阵
        a[a < 0.95] = 0
        # 对角线加上一个值，使其对角线占优
        a += 4 * eye(n)
        # 转换为压缩稀疏列格式的矩阵 a_
        a_ = csc_matrix(a)
        # 对 a_ 进行 LU 分解
        lu = splu(a_)

        # 现在测试我们没有引用计数 bug
        rc = sys.getrefcount(lu)
        # 检查 LU 分解对象的一些属性
        for attr in ('perm_r', 'perm_c'):
            # 获取 LU 分解对象 lu 的属性 perm_r 和 perm_c
            perm = getattr(lu, attr)
            # 断言 lu 的引用计数增加了一个
            assert_equal(sys.getrefcount(lu), rc + 1)
            # 删除 perm 对象后，断言 lu 的引用计数恢复到原始值
            del perm
            assert_equal(sys.getrefcount(lu), rc)

    def test_bad_inputs(self):
        # 将 self.A 转换为压缩稀疏列格式的矩阵 A
        A = self.A.tocsc()

        # 断言对于 A 的部分列切片会引发 ValueError 异常
        assert_raises(ValueError, splu, A[:, :4])
        assert_raises(ValueError, spilu, A[:, :4])

        # 对于 splu(A) 和 spilu(A) 进行一些测试
        for lu in [splu(A), spilu(A)]:
            b = random.rand(42)
            B = random.rand(42, 3)
            BB = random.rand(self.n, 3, 9)
            # 断言对 lu.solve(b), lu.solve(B), lu.solve(BB) 分别会引发 ValueError 异常
            assert_raises(ValueError, lu.solve, b)
            assert_raises(ValueError, lu.solve, B)
            assert_raises(ValueError, lu.solve, BB)
            # 断言对于复数类型输入会引发 TypeError 异常
            assert_raises(TypeError, lu.solve, b.astype(np.complex64))
            assert_raises(TypeError, lu.solve, b.astype(np.complex128))

    @sup_sparse_efficiency
    def test_superlu_dlamch_i386_nan(self):
        # 在 i386@linux 平台上，SuperLU 4.3 调用一些返回浮点数的函数时未声明类型。
        # 这会导致在调用后未清除浮点寄存器，可能导致下一次浮点操作中出现 NaN。
        #
        # 这里是一个触发该问题的测试用例。
        n = 8
        # 创建一个带有特定对角线值的稀疏矩阵 A
        d = np.arange(n) + 1
        A = spdiags((d, 2*d, d[::-1]), (-3, 0, 5), n, n)
        A = A.astype(np.float32)
        # 对 A 进行不同操作
        spilu(A)
        A = A + 1j * A
        B = A.toarray()
        # 断言 B 中不存在 NaN 值
        assert_(not np.isnan(B).any())

    @sup_sparse_efficiency
    def test_lu_attr(self):

        def check(dtype, complex_2=False):
            # 将 self.A 转换为指定类型 dtype 的矩阵 A
            A = self.A.astype(dtype)

            if complex_2:
                # 如果 complex_2 为 True，则对 A 添加复数部分
                A = A + 1j * A.T

            n = A.shape[0]
            # 对 A 进行 LU 分解
            lu = splu(A)

            # 检查 LU 分解是否如宣传的那样
            # 构建置换矩阵 Pc 和 Pr
            Pc = np.zeros((n, n))
            Pc[np.arange(n), lu.perm_c] = 1

            Pr = np.zeros((n, n))
            Pr[lu.perm_r, np.arange(n)] = 1

            Ad = A.toarray()
            # 计算左侧和右侧的乘积，使用指定的容差 atol
            lhs = Pr.dot(Ad).dot(Pc)
            rhs = (lu.L @ lu.U).toarray()

            eps = np.finfo(dtype).eps

            # 断言 lhs 和 rhs 在指定容差下全部相等
            assert_allclose(lhs, rhs, atol=100 * eps)

        # 分别对不同类型进行测试
        check(np.float32)
        check(np.float64)
        check(np.complex64)
        check(np.complex128)
        check(np.complex64, True)
        check(np.complex128, True)

    @pytest.mark.slow
    @sup_sparse_efficiency
    # 定义一个测试并发线程的方法，这是一个测试用例的一部分
    def test_threads_parallel(self):
        # 创建一个空列表 oks，用于存储线程执行成功的标志
        oks = []

        # 定义一个内部函数 worker，用于执行具体的测试任务
        def worker():
            try:
                # 调用测试方法 self.test_splu_basic()，执行基本测试
                self.test_splu_basic()
                # 调用内部测试方法 self._internal_test_splu_smoketest()，执行烟雾测试
                self._internal_test_splu_smoketest()
                # 调用内部测试方法 self._internal_test_spilu_smoketest()，执行烟雾测试
                self._internal_test_spilu_smoketest()
                # 若以上测试全部通过，将 True 添加到 oks 列表中表示成功
                oks.append(True)
            except Exception:
                # 若出现异常，捕获并忽略，不影响主线程的执行
                pass

        # 创建包含 20 个线程的列表，每个线程都执行 worker 函数
        threads = [threading.Thread(target=worker)
                   for k in range(20)]
        
        # 启动所有线程
        for t in threads:
            t.start()
        
        # 等待所有线程执行完毕
        for t in threads:
            t.join()

        # 断言 oks 列表的长度为 20，确保所有线程都成功执行了测试
        assert_equal(len(oks), 20)
# 定义一个测试类 TestGstrsErrors，用于测试 gstrs 函数的异常情况
class TestGstrsErrors:
    
    # 在每个测试方法运行之前设置测试环境
    def setup_method(self):
        # 初始化测试数据 A 和 b
        self.A = array([[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]], dtype=np.float64)
        self.b = np.array([[1.0],[2.0],[3.0]], dtype=np.float64)

    # 测试 gstrs 函数在传入错误的 trans 参数时是否能引发 ValueError 异常
    def test_trans(self):
        # 生成稀疏下三角矩阵 L 和上三角矩阵 U
        L = scipy.sparse.tril(self.A, format='csc')
        U = scipy.sparse.triu(self.A, k=1, format='csc')
        # 使用 assert_raises 检查 gstrs 调用时是否会引发指定异常，并匹配给定的错误消息
        with assert_raises(ValueError, match="trans must be N, T, or H"):
            _superlu.gstrs('X', L.shape[0], L.nnz, L.data, L.indices, L.indptr,
                                U.shape[0], U.nnz, U.data, U.indices, U.indptr, self.b)

    # 测试 gstrs 函数在传入形状不匹配的 L 和 U 矩阵时是否能引发 ValueError 异常
    def test_shape_LU(self):
        # 生成部分稀疏下三角矩阵 L 和完整上三角矩阵 U
        L = scipy.sparse.tril(self.A[0:2,0:2], format='csc')
        U = scipy.sparse.triu(self.A, k=1, format='csc')
        # 使用 assert_raises 检查 gstrs 调用时是否会引发指定异常，并匹配给定的错误消息
        with assert_raises(ValueError, match="L and U must have the same dimension"):
            _superlu.gstrs('N', L.shape[0], L.nnz, L.data, L.indices, L.indptr,
                                U.shape[0], U.nnz, U.data, U.indices, U.indptr, self.b)

    # 测试 gstrs 函数在传入形状不匹配的右侧向量 b 时是否能引发 ValueError 异常
    def test_shape_b(self):
        # 生成稀疏下三角矩阵 L 和上三角矩阵 U
        L = scipy.sparse.tril(self.A, format='csc')
        U = scipy.sparse.triu(self.A, k=1, format='csc')
        # 使用 assert_raises 检查 gstrs 调用时是否会引发指定异常，并匹配给定的错误消息
        with assert_raises(ValueError, match="right hand side array has invalid shape"):
            _superlu.gstrs('N', L.shape[0], L.nnz, L.data, L.indices, L.indptr,
                                U.shape[0], U.nnz, U.data, U.indices, U.indptr, 
                                self.b[0:2])

    # 测试 gstrs 函数在传入不同数据类型的 L 和 U 矩阵时是否能引发 TypeError 异常
    def test_types_differ(self):
        # 生成不同数据类型的稀疏下三角矩阵 L 和上三角矩阵 U
        L = scipy.sparse.tril(self.A.astype(np.float32), format='csc')
        U = scipy.sparse.triu(self.A, k=1, format='csc')
        # 使用 assert_raises 检查 gstrs 调用时是否会引发指定异常，并匹配给定的错误消息
        with assert_raises(TypeError, match="nzvals types of L and U differ"):
            _superlu.gstrs('N', L.shape[0], L.nnz, L.data, L.indices, L.indptr,
                                U.shape[0], U.nnz, U.data, U.indices, U.indptr, self.b)

    # 测试 gstrs 函数在传入不支持的数据类型的 L, U 或 b 时是否能引发 TypeError 异常
    def test_types_unsupported(self):
        # 生成不支持的数据类型的稀疏下三角矩阵 L 和上三角矩阵 U，以及不支持的数据类型的右侧向量 b
        L = scipy.sparse.tril(self.A.astype(np.uint8), format='csc')
        U = scipy.sparse.triu(self.A.astype(np.uint8), k=1, format='csc')
        # 使用 assert_raises 检查 gstrs 调用时是否会引发指定异常，并匹配给定的错误消息
        with assert_raises(TypeError, match="nzvals is not of a type supported"):
            _superlu.gstrs('N', L.shape[0], L.nnz, L.data, L.indices, L.indptr,
                                U.shape[0], U.nnz, U.data, U.indices, U.indptr, 
                                self.b.astype(np.uint8))

# 定义一个测试类 TestSpsolveTriangular，用于测试 spsolve_triangular 函数
class TestSpsolveTriangular:
    
    # 在每个测试方法运行之前设置测试环境
    def setup_method(self):
        # 关闭 UMFPACK 求解器
        use_solver(useUmfpack=False)

    # 使用 pytest.mark.parametrize 注解，参数化 fmt 参数，用于测试不同的稀疏矩阵格式
    @pytest.mark.parametrize("fmt",["csr","csc"])
    # 定义一个测试函数，用于测试零对角线的情况，使用给定的稀疏矩阵格式 fmt
    def test_zero_diagonal(self, fmt):
        # 矩阵的维度设为 5
        n = 5
        # 使用指定种子创建随机数生成器
        rng = np.random.default_rng(43876432987)
        # 生成一个 n x n 的标准正态分布的随机稀疏矩阵 A
        A = rng.standard_normal((n, n))
        # 创建一个长度为 n 的向量 b，其值为 [0, 1, 2, 3, 4]
        b = np.arange(n)
        # 将 A 转换为指定格式的下三角稀疏矩阵
        A = scipy.sparse.tril(A, k=0, format=fmt)

        # 使用 spsolve_triangular 解决三角线性系统 Ax=b，要求单位对角线并且是下三角
        x = spsolve_triangular(A, b, unit_diagonal=True, lower=True)

        # 将 A 的对角线元素设置为 1
        A.setdiag(1)
        # 断言 A 乘以 x 得到的结果接近于向量 b
        assert_allclose(A.dot(x), b)

        # 从 gh-15199 的回归测试
        # 创建一个具有特定数值的 3x3 浮点数数组 A 和长度为 3 的浮点数数组 b
        A = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float64)
        b = np.array([1., 2., 3.])
        # 使用上下文管理器 suppress_warnings 来忽略警告
        with suppress_warnings() as sup:
            # 设置警告过滤器，忽略 SparseEfficiencyWarning 类型的警告信息
            sup.filter(SparseEfficiencyWarning, "CSC or CSR matrix format is")
            # 使用 spsolve_triangular 解决三角线性系统 Ax=b，要求单位对角线
            spsolve_triangular(A, b, unit_diagonal=True)

    # 标记测试参数化，fmt 可以是 "csr" 或 "csc"
    @pytest.mark.parametrize("fmt", ["csr", "csc"])
    # 定义一个测试函数，用于测试奇异矩阵情况
    def test_singular(self, fmt):
        # 矩阵的维度设为 5
        n = 5
        # 根据 fmt 的取值选择创建一个稀疏矩阵 A，可以是 CSR 或 CSC 格式
        if fmt == "csr":
            A = csr_matrix((n, n))
        else:
            A = csc_matrix((n, n))
        # 创建一个长度为 n 的向量 b，其值为 [0, 1, 2, 3, 4]
        b = np.arange(n)
        # 遍历 lower 参数为 True 和 False 两种情况
        for lower in (True, False):
            # 断言求解三角线性系统 Ax=b 时会引发 LinAlgError 异常
            assert_raises(scipy.linalg.LinAlgError,
                          spsolve_triangular, A, b, lower=lower)

    # 标记测试为提高稀疏矩阵运算效率
    @sup_sparse_efficiency
    # 定义一个测试函数，用于测试不合适的矩阵形状情况
    def test_bad_shape(self):
        # 创建一个形状为 (3, 4) 的零矩阵 A
        A = np.zeros((3, 4))
        # 创建一个形状为 (4, 1) 的全为 1 的向量 b
        b = ones((4, 1))
        # 断言求解三角线性系统 Ax=b 时会引发 ValueError 异常
        assert_raises(ValueError, spsolve_triangular, A, b)
        # 创建一个 CSR 格式的单位矩阵 A2 和一个形状不兼容的向量 b2
        A2 = csr_matrix(eye(3))
        b2 = array([1.0, 2.0])
        # 断言求解三角线性系统 A2x=b2 时会引发 ValueError 异常
        assert_raises(ValueError, spsolve_triangular, A2, b2)

    # 标记测试为提高稀疏矩阵运算效率
    @sup_sparse_efficiency
    # 定义一个测试函数，用于测试输入数据类型的情况
    def test_input_types(self):
        # 创建一个 2x2 的浮点数数组 A 和一个 2x2 的浮点数数组 b
        A = array([[1., 0.], [1., 2.]])
        b = array([[2., 0.], [2., 2.]])
        # 遍历 matrix_type 参数为 array, csc_matrix, csr_matrix 三种情况
        for matrix_type in (array, csc_matrix, csr_matrix):
            # 使用 spsolve_triangular 解决三角线性系统 Ax=b，要求是下三角矩阵
            x = spsolve_triangular(matrix_type(A), b, lower=True)
            # 断言 A 乘以 x 得到的结果接近于向量 b
            assert_array_almost_equal(A.dot(x), b)

    # 标记测试为慢速测试
    @pytest.mark.slow
    # 标记测试为提高稀疏矩阵运算效率
    @sup_sparse_efficiency
    # 参数化测试，测试不同的参数组合
    @pytest.mark.parametrize("n", [10, 10**2, 10**3])
    @pytest.mark.parametrize("m", [1, 10])
    @pytest.mark.parametrize("lower", [True, False])
    @pytest.mark.parametrize("format", ["csr", "csc"])
    @pytest.mark.parametrize("unit_diagonal", [False, True])
    @pytest.mark.parametrize("choice_of_A", ["real", "complex"])
    @pytest.mark.parametrize("choice_of_b", ["floats", "ints", "complexints"])
    # 定义一个测试函数，用于测试随机生成的三角稀疏矩阵求解线性系统
    def test_random(self, n, m, lower, format, unit_diagonal, choice_of_A, choice_of_b):
        
        # 定义生成随机三角矩阵的内部函数
        def random_triangle_matrix(n, lower=True, format="csr", choice_of_A="real"):
            # 根据 choice_of_A 参数选择数据类型
            if choice_of_A == "real":
                dtype = np.float64
            elif choice_of_A == "complex":
                dtype = np.complex128
            else:
                raise ValueError("choice_of_A must be 'real' or 'complex'.")
            
            # 创建一个随机数生成器 rng
            rng = np.random.default_rng(789002319)
            # 使用 rng 来生成随机数 rvs
            rvs = rng.random
            # 使用 scipy.sparse.random 生成稀疏随机矩阵 A
            A = scipy.sparse.random(n, n, density=0.1, format='lil', dtype=dtype,
                    random_state=rng, data_rvs=rvs)
            
            # 如果 lower 参数为 True，则将 A 转换为下三角形式
            if lower:
                A = scipy.sparse.tril(A, format="lil")
            else:
                A = scipy.sparse.triu(A, format="lil")
            
            # 将对角线元素设为随机生成的数值加 1
            for i in range(n):
                A[i, i] = np.random.rand() + 1
            
            # 根据 format 参数将 A 转换为 CSC 或 CSR 格式
            if format == "csc":
                A = A.tocsc(copy=False)
            else:
                A = A.tocsr(copy=False)
            
            return A
        
        # 设定随机数种子
        np.random.seed(1234)
        # 生成随机三角矩阵 A
        A = random_triangle_matrix(n, lower=lower, format=format, choice_of_A=choice_of_A)
        
        # 根据 choice_of_b 参数生成向量 b
        if choice_of_b == "floats":
            b = np.random.rand(n, m)
        elif choice_of_b == "ints":
            b = np.random.randint(-9, 9, (n, m))
        elif choice_of_b == "complexints":
            b = np.random.randint(-9, 9, (n, m)) + np.random.randint(-9, 9, (n, m)) * 1j
        else:
            raise ValueError(
                "choice_of_b must be 'floats', 'ints', or 'complexints'.")
        
        # 使用 spsolve_triangular 求解线性系统 Ax = b
        x = spsolve_triangular(A, b, lower=lower, unit_diagonal=unit_diagonal)
        
        # 如果 unit_diagonal 参数为 True，则将 A 的对角线元素设为 1
        if unit_diagonal:
            A.setdiag(1)
        
        # 断言 A * x 与 b 的近似相等性
        assert_allclose(A.dot(x), b, atol=1.5e-6)
```