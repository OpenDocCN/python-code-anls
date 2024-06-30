# `D:\src\scipysrc\scipy\scipy\linalg\tests\test_interpolative.py`

```
# 导入 scipy.linalg.interpolative 模块中的 pymatrixid，这是一个用于矩阵插值分解的工具
import scipy.linalg.interpolative as pymatrixid
# 导入 numpy 库，并将其命名为 np
import numpy as np
# 从 scipy.linalg 中导入 hilbert、svdvals 和 norm 函数
from scipy.linalg import hilbert, svdvals, norm
# 从 scipy.sparse.linalg 中导入 aslinearoperator 函数
from scipy.sparse.linalg import aslinearoperator
# 从 scipy.linalg.interpolative 中导入 interp_decomp 函数
from scipy.linalg.interpolative import interp_decomp
# 导入 numpy.testing 模块中的各种断言函数：assert_, assert_allclose, assert_equal, assert_array_equal
from numpy.testing import (assert_, assert_allclose, assert_equal,
                           assert_array_equal)
# 导入 pytest 库，并将其中的 raises 函数命名为 assert_raises
import pytest
from pytest import raises as assert_raises
# 导入 sys 模块
import sys

# 检测系统是否为32位，如果是，则 _IS_32BIT 为 True
_IS_32BIT = (sys.maxsize < 2**32)

# 定义 eps 的 pytest fixture，返回值为 1e-12，用于比较浮点数的误差容忍度
@pytest.fixture()
def eps():
    yield 1e-12

# 定义 A 的 pytest fixture，参数化返回一个 Hilbert 矩阵的 numpy 数组，可以是 np.float64 或 np.complex128 类型
@pytest.fixture(params=[np.float64, np.complex128])
def A(request):
    # 构造 Hilbert 矩阵
    n = 300
    # 返回经 request.param 指定类型的 Hilbert 矩阵
    yield hilbert(n).astype(request.param)

# 定义 L 的 pytest fixture，返回 A 的线性操作符表示
@pytest.fixture()
def L(A):
    yield aslinearoperator(A)

# 定义 rank 的 pytest fixture，返回 A 的秩
@pytest.fixture()
def rank(A, eps):
    # 计算 A 的奇异值
    S = np.linalg.svd(A, compute_uv=False)
    try:
        # 寻找第一个小于 eps 的奇异值的索引，从而确定 A 的秩
        rank = np.nonzero(S < eps)[0][0]
    except IndexError:
        # 如果所有奇异值都大于等于 eps，则 A 的秩为其行数
        rank = A.shape[0]
    return rank

# 定义 TestInterpolativeDecomposition 类，用于测试插值分解的功能
class TestInterpolativeDecomposition:

    # 参数化测试用例，测试不同的随机化和线性操作符设置
    @pytest.mark.parametrize(
        "rand,lin_op",
        [(False, False), (True, False), (True, True)])
    def test_real_id_fixed_precision(self, A, L, eps, rand, lin_op):
        # 如果系统是32位且A的dtype是np.complex128并且rand为True，则预期测试失败并给出错误信息
        if _IS_32BIT and A.dtype == np.complex128 and rand:
            pytest.xfail("bug in external fortran code")
        # 在Hilbert矩阵上测试ID例程
        A_or_L = A if not lin_op else L

        # 使用interp_decomp函数对A_or_L进行插值分解，获取k（阶数）、idx（索引）、proj（投影矩阵）
        k, idx, proj = pymatrixid.interp_decomp(A_or_L, eps, rand=rand)
        # 使用reconstruct_matrix_from_id函数重建矩阵B
        B = pymatrixid.reconstruct_matrix_from_id(A[:, idx[:k]], idx, proj)
        # 断言A与B的所有元素在相对误差rtol和绝对误差atol范围内接近
        assert_allclose(A, B, rtol=eps, atol=1e-08)

    @pytest.mark.parametrize(
        "rand,lin_op",
        [(False, False), (True, False), (True, True)])
    def test_real_id_fixed_rank(self, A, L, eps, rank, rand, lin_op):
        # 如果系统是32位且A的dtype是np.complex128并且rand为True，则预期测试失败并给出错误信息
        if _IS_32BIT and A.dtype == np.complex128 and rand:
            pytest.xfail("bug in external fortran code")
        k = rank
        A_or_L = A if not lin_op else L

        # 使用interp_decomp函数对A_or_L进行插值分解，获取idx（索引）、proj（投影矩阵）
        idx, proj = pymatrixid.interp_decomp(A_or_L, k, rand=rand)
        # 使用reconstruct_matrix_from_id函数重建矩阵B
        B = pymatrixid.reconstruct_matrix_from_id(A[:, idx[:k]], idx, proj)
        # 断言A与B的所有元素在相对误差rtol和绝对误差atol范围内接近
        assert_allclose(A, B, rtol=eps, atol=1e-08)

    @pytest.mark.parametrize("rand,lin_op", [(False, False)])
    def test_real_id_skel_and_interp_matrices(
            self, A, L, eps, rank, rand, lin_op):
        k = rank
        A_or_L = A if not lin_op else L

        # 使用interp_decomp函数对A_or_L进行插值分解，获取idx（索引）、proj（投影矩阵）
        idx, proj = pymatrixid.interp_decomp(A_or_L, k, rand=rand)
        # 使用reconstruct_interp_matrix函数重建插值矩阵P
        P = pymatrixid.reconstruct_interp_matrix(idx, proj)
        # 使用reconstruct_skel_matrix函数重建骨架矩阵B
        B = pymatrixid.reconstruct_skel_matrix(A, k, idx)
        # 断言B与A的列子集在相对误差rtol和绝对误差atol范围内接近
        assert_allclose(B, A[:, idx[:k]], rtol=eps, atol=1e-08)
        # 断言B乘以P与A在相对误差rtol和绝对误差atol范围内接近
        assert_allclose(B @ P, A, rtol=eps, atol=1e-08)

    @pytest.mark.parametrize(
        "rand,lin_op",
        [(False, False), (True, False), (True, True)])
    def test_svd_fixed_precison(self, A, L, eps, rand, lin_op):
        # 如果系统是32位且A的dtype是np.complex128并且rand为True，则预期测试失败并给出错误信息
        if _IS_32BIT and A.dtype == np.complex128 and rand:
            pytest.xfail("bug in external fortran code")
        A_or_L = A if not lin_op else L

        # 使用svd函数对A_or_L进行奇异值分解，获取U（左奇异矩阵）、S（奇异值）、V（右奇异矩阵的共轭转置）
        U, S, V = pymatrixid.svd(A_or_L, eps, rand=rand)
        # 重建矩阵B
        B = U * S @ V.T.conj()
        # 断言A与B的所有元素在相对误差rtol和绝对误差atol范围内接近
        assert_allclose(A, B, rtol=eps, atol=1e-08)

    @pytest.mark.parametrize(
        "rand,lin_op",
        [(False, False), (True, False), (True, True)])
    def test_svd_fixed_rank(self, A, L, eps, rank, rand, lin_op):
        # 如果系统是32位且A的dtype是np.complex128并且rand为True，则预期测试失败并给出错误信息
        if _IS_32BIT and A.dtype == np.complex128 and rand:
            pytest.xfail("bug in external fortran code")
        k = rank
        A_or_L = A if not lin_op else L

        # 使用svd函数对A_or_L进行奇异值分解，获取U（左奇异矩阵）、S（奇异值）、V（右奇异矩阵的共轭转置）
        U, S, V = pymatrixid.svd(A_or_L, k, rand=rand)
        # 重建矩阵B
        B = U * S @ V.T.conj()
        # 断言A与B的所有元素在相对误差rtol和绝对误差atol范围内接近
        assert_allclose(A, B, rtol=eps, atol=1e-08)

    def test_id_to_svd(self, A, eps, rank):
        k = rank

        # 使用interp_decomp函数对A进行插值分解，获取idx（索引）、proj（投影矩阵）
        idx, proj = pymatrixid.interp_decomp(A, k, rand=False)
        # 使用id_to_svd函数将ID（插值分解）转换为SVD（奇异值分解），获取U（左奇异矩阵）、S（奇异值）、V（右奇异矩阵的共轭转置）
        U, S, V = pymatrixid.id_to_svd(A[:, idx[:k]], idx, proj)
        # 重建矩阵B
        B = U * S @ V.T.conj()
        # 断言A与B的所有元素在相对误差rtol和绝对误差atol范围内接近
        assert_allclose(A, B, rtol=eps, atol=1e-08)

    def test_estimate_spectral_norm(self, A):
        # 计算矩阵A的奇异值
        s = svdvals(A)
        # 估算矩阵A的谱范数
        norm_2_est = pymatrixid.estimate_spectral_norm(A)
        # 断言估算的谱范数与矩阵A的最大奇异值在相对误差1e-6和绝对误差1e-8范围内接近
        assert_allclose(norm_2_est, s[0], rtol=1e-6, atol=1e-8)
    # 定义一个测试函数，用于评估估计的谱范数差异
    def test_estimate_spectral_norm_diff(self, A):
        # 复制矩阵 A 到 B
        B = A.copy()
        # 修改 B 的第一列，使其乘以 1.2
        B[:, 0] *= 1.2
        # 计算矩阵 A 和 B 的奇异值，并存储在 s 中
        s = svdvals(A - B)
        # 使用 pymatrixid 库估计矩阵 A 和 B 的谱范数差异
        norm_2_est = pymatrixid.estimate_spectral_norm_diff(A, B)
        # 断言估计的谱范数差异与 s[0] 相近，允许一定的相对误差和绝对误差
        assert_allclose(norm_2_est, s[0], rtol=1e-6, atol=1e-8)

    # 定义一个测试函数，用于评估矩阵的秩估计（针对数组输入）
    def test_rank_estimates_array(self, A):
        # 创建一个固定的 B 矩阵作为比较对象
        B = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=A.dtype)

        # 对 A 和 B 进行循环遍历
        for M in [A, B]:
            # 设置秩的容差
            rank_tol = 1e-9
            # 使用 numpy 计算 M 的秩，基于其二范数乘以秩容差
            rank_np = np.linalg.matrix_rank(M, norm(M, 2) * rank_tol)
            # 使用 pymatrixid 库估计矩阵 M 的秩
            rank_est = pymatrixid.estimate_rank(M, rank_tol)
            # 断言估计的秩不小于 numpy 计算得到的秩
            assert_(rank_est >= rank_np)
            # 断言估计的秩不大于 numpy 计算得到的秩再加上 10
            assert_(rank_est <= rank_np + 10)

    # 定义一个测试函数，用于评估矩阵的秩估计（针对线性操作器输入）
    def test_rank_estimates_lin_op(self, A):
        # 创建一个固定的 B 矩阵作为比较对象
        B = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=A.dtype)

        # 对 A 和 B 进行循环遍历
        for M in [A, B]:
            # 将 M 转换为线性操作器
            ML = aslinearoperator(M)
            # 设置秩的容差
            rank_tol = 1e-9
            # 使用 numpy 计算 M 的秩，基于其二范数乘以秩容差
            rank_np = np.linalg.matrix_rank(M, norm(M, 2) * rank_tol)
            # 使用 pymatrixid 库估计线性操作器 ML 的秩
            rank_est = pymatrixid.estimate_rank(ML, rank_tol)
            # 断言估计的秩在不小于 numpy 计算得到的秩减 4 和不大于 numpy 计算得到的秩加 4 之间
            assert_(rank_est >= rank_np - 4)
            assert_(rank_est <= rank_np + 4)

    # 定义一个测试函数，用于测试随机数生成器
    def test_rand(self):
        # 使用默认种子生成随机数，并断言生成的随机数近似于指定的值
        pymatrixid.seed('default')
        assert_allclose(pymatrixid.rand(2), [0.8932059, 0.64500803],
                        rtol=1e-4, atol=1e-8)

        # 使用指定种子生成随机数，并断言生成的随机数近似于指定的值
        pymatrixid.seed(1234)
        x1 = pymatrixid.rand(2)
        assert_allclose(x1, [0.7513823, 0.06861718], rtol=1e-4, atol=1e-8)

        # 使用 numpy 的种子生成随机数，并重置 pymatrixid 的种子，并断言生成的随机数与之前一致
        np.random.seed(1234)
        pymatrixid.seed()
        x2 = pymatrixid.rand(2)

        # 使用 numpy 的种子生成随机数，并将其作为 pymatrixid 的种子，并断言生成的随机数与之前一致
        np.random.seed(1234)
        pymatrixid.seed(np.random.rand(55))
        x3 = pymatrixid.rand(2)

        # 断言 x1、x2、x3 三者近似相等
        assert_allclose(x1, x2)
        assert_allclose(x1, x3)

    # 定义一个测试函数，用于测试 pymatrixid.interp_decomp 函数在不良调用时是否抛出 ValueError
    def test_badcall(self):
        # 创建一个希尔伯特矩阵 A，并将其转换为 np.float32 类型
        A = hilbert(5).astype(np.float32)
        # 使用 assert_raises 断言调用 pymatrixid.interp_decomp 函数时会抛出 ValueError 异常
        with assert_raises(ValueError):
            pymatrixid.interp_decomp(A, 1e-6, rand=False)

    # 定义一个测试函数，用于测试当请求的秩过大时，pymatrixid.svd 函数是否正确处理异常
    def test_rank_too_large(self):
        # 创建一个全为 1 的 4x3 数组 a
        a = np.ones((4, 3))
        # 使用 assert_raises 断言调用 pymatrixid.svd 函数时会抛出 ValueError 异常
        with assert_raises(ValueError):
            pymatrixid.svd(a, 4)

    # 定义一个测试函数，用于测试在特定条件下 pymatrixid 库的函数是否正确工作
    def test_full_rank(self):
        eps = 1.0e-12

        # 创建一个随机的 16x8 的浮点数数组 A
        A = np.random.rand(16, 8)
        # 使用 pymatrixid.interp_decomp 函数对 A 进行插值分解，要求精度为 eps
        k, idx, proj = pymatrixid.interp_decomp(A, eps)
        # 断言插值分解得到的秩 k 等于 A 的列数
        assert_equal(k, A.shape[1])

        # 使用插值分解得到的 idx 和 proj 重构插值矩阵 P 和骨架矩阵 B
        P = pymatrixid.reconstruct_interp_matrix(idx, proj)
        B = pymatrixid.reconstruct_skel_matrix(A, k, idx)
        # 断言重构后的矩阵 B @ P 与原始矩阵 A 近似相等
        assert_allclose(A, B @ P)

        # 使用固定的秩 k 对 A 进行插值分解
        idx, proj = pymatrixid.interp_decomp(A, k)

        # 再次重构插值矩阵 P 和骨架矩阵 B
        P = pymatrixid.reconstruct_interp_matrix(idx, proj)
        B = pymatrixid.reconstruct_skel_matrix(A, k, idx)
        # 断言重构后的矩阵 B @ P 与原始矩阵 A 近似相等
        assert_allclose(A, B @ P)

    # 使用参数化测试标记，定义一组测试函数参数化的情况
    @pytest.mark.parametrize("dtype", [np.float64, np.complex128])
    @pytest.mark.parametrize("rand", [True, False])
    @pytest.mark.parametrize("eps", [1, 0.1])
    # 定义名为 test_bug_9793 的测试方法，接受参数 dtype、rand、eps
    def test_bug_9793(self, dtype, rand, eps):
        # 如果系统为32位且 dtype 为复数类型 np.complex128 且 rand 为真，标记测试为预期失败
        if _IS_32BIT and dtype == np.complex128 and rand:
            pytest.xfail("bug in external fortran code")
        
        # 创建一个二维数组 A，包含特定的整数值，使用给定的 dtype 和内存布局 "C"
        A = np.array([[-1, -1, -1, 0, 0, 0],
                      [0, 0, 0, 1, 1, 1],
                      [1, 0, 0, 1, 0, 0],
                      [0, 1, 0, 0, 1, 0],
                      [0, 0, 1, 0, 0, 1]],
                     dtype=dtype, order="C")
        
        # 复制数组 A，得到数组 B
        B = A.copy()
        
        # 调用 interp_decomp 函数，传入 A 的转置、eps 和 rand 参数，对 A 进行修改
        interp_decomp(A.T, eps, rand=rand)
        
        # 断言数组 A 和 B 相等，用于验证 interp_decomp 函数是否正确地修改了 A
        assert_array_equal(A, B)
```