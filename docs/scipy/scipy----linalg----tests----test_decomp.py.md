# `D:\src\scipysrc\scipy\scipy\linalg\tests\test_decomp.py`

```
# 导入 itertools 模块，用于生成迭代器的函数
import itertools
# 导入 platform 模块，用于访问底层平台信息的函数
import platform

# 导入 numpy 库，并将其重命名为 np
import numpy as np
# 从 numpy.testing 模块中导入多个断言函数，用于单元测试时比较数组或值
from numpy.testing import (assert_equal, assert_almost_equal,
                           assert_array_almost_equal, assert_array_equal,
                           assert_, assert_allclose)

# 导入 pytest 库，用于编写简单而有效的单元测试
import pytest
# 从 pytest 模块中导入 raises 函数，并将其重命名为 assert_raises
from pytest import raises as assert_raises

# 从 scipy.linalg 模块中导入多个线性代数函数
from scipy.linalg import (eig, eigvals, lu, svd, svdvals, cholesky, qr,
                          schur, rsf2csf, lu_solve, lu_factor, solve, diagsvd,
                          hessenberg, rq, eig_banded, eigvals_banded, eigh,
                          eigvalsh, qr_multiply, qz, orth, ordqz,
                          subspace_angles, hadamard, eigvalsh_tridiagonal,
                          eigh_tridiagonal, null_space, cdf2rdf, LinAlgError)

# 从 scipy.linalg.lapack 模块中导入多个 LAPACK 函数
from scipy.linalg.lapack import (dgbtrf, dgbtrs, zgbtrf, zgbtrs, dsbev,
                                 dsbevd, dsbevx, zhbevd, zhbevx)

# 从 scipy.linalg._misc 模块中导入 norm 函数
from scipy.linalg._misc import norm
# 从 scipy.linalg._decomp_qz 模块中导入 _select_function 函数
from scipy.linalg._decomp_qz import _select_function
# 从 scipy.stats 模块中导入 ortho_group 类，用于生成正交或酉矩阵
from scipy.stats import ortho_group

# 从 numpy 模块中导入多个函数和类
from numpy import (array, diag, full, linalg, argsort, zeros, arange,
                   float32, complex64, ravel, sqrt, iscomplex, shape, sort,
                   sign, asarray, isfinite, ndarray, eye,)

# 从 scipy.linalg._testutils 模块中导入 assert_no_overwrite 函数，用于测试不会覆盖数据
from scipy.linalg._testutils import assert_no_overwrite
# 从 scipy.sparse._sputils 模块中导入 matrix 类，用于创建矩阵
from scipy.sparse._sputils import matrix

# 从 scipy._lib._testutils 模块中导入 check_free_memory 函数，用于检查可用内存
from scipy._lib._testutils import check_free_memory
# 从 scipy.linalg.blas 模块中导入 HAS_ILP64 常量，用于指示 BLAS 库是否支持 ILP64 接口
from scipy.linalg.blas import HAS_ILP64
try:
    # 尝试从 scipy.__config__ 模块中导入 CONFIG 变量，用于配置信息
    from scipy.__config__ import CONFIG
except ImportError:
    # 如果导入失败，则将 CONFIG 设置为 None
    CONFIG = None


# 定义一个函数，用于生成随机的 Hermitian 对称矩阵
def _random_hermitian_matrix(n, posdef=False, dtype=float):
    "Generate random sym/hermitian array of the given size n"
    if dtype in COMPLEX_DTYPES:
        # 如果数据类型是复数类型，则生成随机的复数矩阵，并确保其共轭转置后为自身的 Hermitian 矩阵
        A = np.random.rand(n, n) + np.random.rand(n, n)*1.0j
        A = (A + A.conj().T) / 2
    else:
        # 如果数据类型是实数类型，则生成随机的实数矩阵，并确保其转置后为自身的对称矩阵
        A = np.random.rand(n, n)
        A = (A + A.T) / 2

    if posdef:
        # 如果需要生成正定矩阵，则加上一个单位矩阵的倍数
        A += sqrt(2 * n) * np.eye(n)

    return A.astype(dtype)


# 定义常量列表，分别包含实数和复数数据类型
REAL_DTYPES = [np.float32, np.float64]
COMPLEX_DTYPES = [np.complex64, np.complex128]
DTYPES = REAL_DTYPES + COMPLEX_DTYPES


# XXX: 这个函数不应该在这里定义，而应该在 scipy.linalg 命名空间的某个地方定义
# 定义一个函数，用于生成随机对称 (Hermitian) 矩阵
def symrand(dim_or_eigv, rng):
    """Return a random symmetric (Hermitian) matrix.

    If 'dim_or_eigv' is an integer N, return a NxN matrix, with eigenvalues
        uniformly distributed on (-1,1).

    If 'dim_or_eigv' is  1-D real array 'a', return a matrix whose
                      eigenvalues are 'a'.
    """
    if isinstance(dim_or_eigv, int):
        # 如果 dim_or_eigv 是整数，则生成一个 dim_or_eigv x dim_or_eigv 的随机矩阵，并且其特征值均匀分布在 (-1,1) 区间
        dim = dim_or_eigv
        d = rng.random(dim) * 2 - 1
    elif (isinstance(dim_or_eigv, ndarray) and
          len(dim_or_eigv.shape) == 1):
        # 如果 dim_or_eigv 是一维实数数组，则生成一个特征值为 dim_or_eigv 的矩阵
        dim = dim_or_eigv.shape[0]
        d = dim_or_eigv
    else:
        # 如果输入类型不支持，则抛出 TypeError 异常
        raise TypeError("input type not supported.")

    # 生成一个正交或酉矩阵
    v = ortho_group.rvs(dim)
    # 生成对称 (Hermitian) 矩阵，确保矩阵的对称性
    h = v.T.conj() @ diag(d) @ v
    # 为避免舍入误差，再次对矩阵进行对称化处理
    h = 0.5 * (h.T + h)
    return h


# 定义一个测试类 TestEigVals
class TestEigVals:
    # 这里会添加测试方法，但在这里不需要注释它们的作用
    # 定义一个测试方法，用于简单的矩阵求特征值
    def test_simple(self):
        # 创建一个二维列表作为输入矩阵
        a = [[1, 2, 3], [1, 2, 3], [2, 5, 6]]
        # 调用 eigvals 函数计算矩阵的特征值
        w = eigvals(a)
        # 预期的精确特征值列表
        exact_w = [(9+sqrt(93))/2, 0, (9-sqrt(93))/2]
        # 断言计算得到的特征值与预期特征值列表几乎相等
        assert_array_almost_equal(w, exact_w)

    # 定义一个测试方法，用于简单转置后的矩阵求特征值
    def test_simple_tr(self):
        # 创建一个二维数组作为输入矩阵，并将其转置为双精度类型
        a = array([[1, 2, 3], [1, 2, 3], [2, 5, 6]], 'd').T
        # 复制矩阵 a 的副本
        a = a.copy()
        # 再次转置矩阵 a
        a = a.T
        # 调用 eigvals 函数计算矩阵的特征值
        w = eigvals(a)
        # 预期的精确特征值列表
        exact_w = [(9+sqrt(93))/2, 0, (9-sqrt(93))/2]
        # 断言计算得到的特征值与预期特征值列表几乎相等
        assert_array_almost_equal(w, exact_w)

    # 定义一个测试方法，用于包含复数的矩阵求特征值
    def test_simple_complex(self):
        # 创建一个包含复数的二维列表作为输入矩阵
        a = [[1, 2, 3], [1, 2, 3], [2, 5, 6+1j]]
        # 调用 eigvals 函数计算矩阵的特征值
        w = eigvals(a)
        # 预期的精确特征值列表，包含复数
        exact_w = [(9+1j+sqrt(92+6j))/2,
                   0,
                   (9+1j-sqrt(92+6j))/2]
        # 断言计算得到的特征值与预期特征值列表几乎相等
        assert_array_almost_equal(w, exact_w)

    # 定义一个测试方法，用于在不检查有限性的情况下计算特征值
    def test_finite(self):
        # 创建一个二维列表作为输入矩阵
        a = [[1, 2, 3], [1, 2, 3], [2, 5, 6]]
        # 调用 eigvals 函数计算矩阵的特征值，不检查有限性
        w = eigvals(a, check_finite=False)
        # 预期的精确特征值列表
        exact_w = [(9+sqrt(93))/2, 0, (9-sqrt(93))/2]
        # 断言计算得到的特征值与预期特征值列表几乎相等
        assert_array_almost_equal(w, exact_w)

    # 使用 pytest 的参数化功能定义一个测试方法，用于测试空矩阵的特征值计算
    @pytest.mark.parametrize('dt', [int, float, float32, complex, complex64])
    def test_empty(self, dt):
        # 创建一个空的 numpy 数组，数据类型由参数 dt 指定
        a = np.empty((0, 0), dtype=dt)
        # 调用 eigvals 函数计算空矩阵的特征值
        w = eigvals(a)
        # 断言计算得到的特征值数组形状为 (0,)
        assert w.shape == (0,)
        # 断言计算得到的特征值数组数据类型与单位矩阵的特征值数据类型相同
        assert w.dtype == eigvals(np.eye(2, dtype=dt)).dtype

        # 使用 homogeneous_eigvals 参数再次调用 eigvals 函数计算特征值
        w = eigvals(a, homogeneous_eigvals=True)
        # 断言计算得到的特征值数组形状为 (2, 0)
        assert w.shape == (2, 0)
        # 断言计算得到的特征值数组数据类型与单位矩阵的特征值数据类型相同
        assert w.dtype == eigvals(np.eye(2, dtype=dt)).dtype
class TestEig:

    def test_simple(self):
        # 创建一个3x3的数组a
        a = array([[1, 2, 3], [1, 2, 3], [2, 5, 6]])
        # 计算矩阵a的特征值(w)和特征向量(v)
        w, v = eig(a)
        # 精确的特征值列表
        exact_w = [(9+sqrt(93))/2, 0, (9-sqrt(93))/2]
        # 预定义的特征向量
        v0 = array([1, 1, (1+sqrt(93)/3)/2])
        v1 = array([3., 0, -1])
        v2 = array([1, 1, (1-sqrt(93)/3)/2])
        # 归一化特征向量
        v0 = v0 / norm(v0)
        v1 = v1 / norm(v1)
        v2 = v2 / norm(v2)
        # 检查特征值是否近似相等
        assert_array_almost_equal(w, exact_w)
        # 检查特征向量是否近似相等
        assert_array_almost_equal(v0, v[:, 0]*sign(v[0, 0]))
        assert_array_almost_equal(v1, v[:, 1]*sign(v[0, 1]))
        assert_array_almost_equal(v2, v[:, 2]*sign(v[0, 2]))
        # 检查是否满足特征值分解的条件
        for i in range(3):
            assert_array_almost_equal(a @ v[:, i], w[i]*v[:, i])
        # 使用left=1, right=0重新计算特征值和特征向量
        w, v = eig(a, left=1, right=0)
        # 再次检查是否满足特征值分解的条件
        for i in range(3):
            assert_array_almost_equal(a.T @ v[:, i], w[i]*v[:, i])

    def test_simple_complex_eig(self):
        # 创建一个2x2的复数数组a
        a = array([[1, 2], [-2, 1]])
        # 计算复数矩阵a的特征值(w)，左特征向量(vl)，右特征向量(vr)
        w, vl, vr = eig(a, left=1, right=1)
        # 检查复数特征值是否近似相等
        assert_array_almost_equal(w, array([1+2j, 1-2j]))
        # 检查是否满足特征值分解的条件
        for i in range(2):
            assert_array_almost_equal(a @ vr[:, i], w[i]*vr[:, i])
        # 检查是否满足特征值分解的条件（共轭转置）
        for i in range(2):
            assert_array_almost_equal(a.conj().T @ vl[:, i],
                                      w[i].conj()*vl[:, i])

    def test_simple_complex(self):
        # 创建一个3x3的复数数组a
        a = array([[1, 2, 3], [1, 2, 3], [2, 5, 6+1j]])
        # 计算复数矩阵a的特征值(w)，左特征向量(vl)，右特征向量(vr)
        w, vl, vr = eig(a, left=1, right=1)
        # 检查是否满足特征值分解的条件
        for i in range(3):
            assert_array_almost_equal(a @ vr[:, i], w[i]*vr[:, i])
        # 检查是否满足特征值分解的条件（共轭转置）
        for i in range(3):
            assert_array_almost_equal(a.conj().T @ vl[:, i],
                                      w[i].conj()*vl[:, i])

    def test_gh_3054(self):
        # 创建简单的1x1矩阵a和b
        a = [[1]]
        b = [[0]]
        # 计算广义特征值(w)和广义右特征向量(vr)，指定homogeneous_eigvals=True
        w, vr = eig(a, b, homogeneous_eigvals=True)
        # 检查特定元素是否近似等于0
        assert_allclose(w[1, 0], 0)
        # 检查特定元素是否不等于0
        assert_(w[0, 0] != 0)
        # 检查是否所有元素都近似等于1
        assert_allclose(vr, 1)

        # 计算普通的特征值(w)和右特征向量(vr)
        w, vr = eig(a, b)
        # 检查是否所有特征值都为无穷大
        assert_equal(w, np.inf)
        # 检查是否所有元素都近似等于1
        assert_allclose(vr, 1)
    # 定义一个方法用于检查广义特征值问题，支持指定的容差阈值
    def _check_gen_eig(self, A, B, atol_homog=1e-13, rtol_homog=1e-13,
                                   atol=1e-13, rtol=1e-13):
        # 如果 B 不为 None，则将 A 和 B 转换为 ndarray
        if B is not None:
            A, B = asarray(A), asarray(B)
            # 将 B0 初始化为 B
            B0 = B
        else:
            # 将 A 转换为 ndarray
            A = asarray(A)
            # 将 B0 初始化为 None
            B0 = B
            # 将 B 初始化为 A 形状的单位矩阵
            B = np.eye(*A.shape)
        # 创建一个包含 A 和 B 的消息字符串
        msg = f"\n{A!r}\n{B!r}"

        # 计算广义特征值及其对应的特征向量
        w, vr = eig(A, B0, homogeneous_eigvals=True)
        # 计算广义特征值
        wt = eigvals(A, B0, homogeneous_eigvals=True)
        # 计算 val1 和 val2，用于后续的数值比较
        val1 = A @ vr * w[1, :]
        val2 = B @ vr * w[0, :]
        # 检查 val1 和 val2 是否在给定的容差范围内相等
        for i in range(val1.shape[1]):
            assert_allclose(val1[:, i], val2[:, i],
                            rtol=rtol_homog, atol=atol_homog, err_msg=msg)

        # 如果 B0 为 None，则进一步检查特定情况下的广义特征值
        if B0 is None:
            assert_allclose(w[1, :], 1)
            assert_allclose(wt[1, :], 1)

        # 对广义特征值进行排序，并比较其排序结果
        perm = np.lexsort(w)
        permt = np.lexsort(wt)
        assert_allclose(w[:, perm], wt[:, permt], atol=1e-7, rtol=1e-7,
                        err_msg=msg)

        # 计算特征向量的长度，并检查其是否接近于单位长度
        length = np.empty(len(vr))
        for i in range(len(vr)):
            length[i] = norm(vr[:, i])
        assert_allclose(length, np.ones(length.size), err_msg=msg,
                        atol=1e-7, rtol=1e-7)

        # 转换为非齐次坐标系下的广义特征值，并进行比较
        beta_nonzero = (w[1, :] != 0)
        wh = w[0, beta_nonzero] / w[1, beta_nonzero]

        # 计算标准坐标系下的广义特征值及其对应的特征向量，并进行数值比较
        w, vr = eig(A, B0)
        wt = eigvals(A, B0)
        val1 = A @ vr
        val2 = B @ vr * w
        res = val1 - val2
        for i in range(res.shape[1]):
            if np.all(isfinite(res[:, i])):
                assert_allclose(res[:, i], 0,
                                rtol=rtol, atol=atol, err_msg=msg)

        # 对有效的实部部分的广义特征值进行排序，包括复共轭对
        w_fin = w[isfinite(w)]
        wt_fin = wt[isfinite(wt)]
        w_fin = -1j * np.real_if_close(1j*w_fin, tol=1e-10)
        wt_fin = -1j * np.real_if_close(1j*wt_fin, tol=1e-10)
        perm = argsort(abs(w_fin) + w_fin.imag)
        permt = argsort(abs(wt_fin) + wt_fin.imag)
        assert_allclose(w_fin[perm], wt_fin[permt],
                        atol=1e-7, rtol=1e-7, err_msg=msg)

        # 计算特征向量的长度，并检查其是否接近于单位长度
        length = np.empty(len(vr))
        for i in range(len(vr)):
            length[i] = norm(vr[:, i])
        assert_allclose(length, np.ones(length.size), err_msg=msg)

        # 比较齐次和非齐次坐标系下的特征值排序结果
        assert_allclose(sort(wh), sort(w[np.isfinite(w)]))
    def test_singular(self):
        # 从以下网址获取的示例
        # https://web.archive.org/web/20040903121217/http://www.cs.umu.se/research/nla/singular_pairs/guptri/matlab.html
        # 定义矩阵 A 和 B
        A = array([[22, 34, 31, 31, 17],
                   [45, 45, 42, 19, 29],
                   [39, 47, 49, 26, 34],
                   [27, 31, 26, 21, 15],
                   [38, 44, 44, 24, 30]])
        B = array([[13, 26, 25, 17, 24],
                   [31, 46, 40, 26, 37],
                   [26, 40, 19, 25, 25],
                   [16, 25, 27, 14, 23],
                   [24, 35, 18, 21, 22]])

        # 忽略所有的 NumPy 错误
        with np.errstate(all='ignore'):
            # 调用 _check_gen_eig 方法进行通用特征值问题的检查
            self._check_gen_eig(A, B, atol_homog=5e-13, atol=5e-13)

    def test_falker(self):
        # 测试导致一些广义特征值为 NaN 的矩阵
        M = diag(array([1, 0, 3]))
        K = array(([2, -1, -1], [-1, 2, -1], [-1, -1, 2]))
        D = array(([1, -1, 0], [-1, 1, 0], [0, 0, 0]))
        Z = zeros((3, 3))
        I3 = eye(3)
        # 构建矩阵 A 和 B
        A = np.block([[I3, Z], [Z, -K]])
        B = np.block([[Z, I3], [M, D]])

        # 忽略所有的 NumPy 错误
        with np.errstate(all='ignore'):
            # 调用 _check_gen_eig 方法进行通用特征值问题的检查
            self._check_gen_eig(A, B)

    def test_bad_geneig(self):
        # Ticket #709 (strange return values from DGGEV)

        def matrices(omega):
            c1 = -9 + omega**2
            c2 = 2*omega
            A = [[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, c1, 0],
                 [0, 0, 0, c1]]
            B = [[0, 0, 1, 0],
                 [0, 0, 0, 1],
                 [1, 0, 0, -c2],
                 [0, 1, c2, 0]]
            return A, B

        # 当使用有问题的 LAPACK 时，该测试可能在不同机器上使用不同的 omega 值失败
        with np.errstate(all='ignore'):
            # 对于多个 omega 值，生成 A 和 B 并调用 _check_gen_eig 方法
            for k in range(100):
                A, B = matrices(omega=k*5./100)
                self._check_gen_eig(A, B)

    def test_make_eigvals(self):
        # 逐步检查 _make_eigvals 方法中的所有路径
        # 实数特征值
        rng = np.random.RandomState(1234)
        A = symrand(3, rng)
        # 调用 _check_gen_eig 方法，检查 A 的特征值
        self._check_gen_eig(A, None)
        B = symrand(3, rng)
        # 调用 _check_gen_eig 方法，检查 A 和 B 的特征值
        self._check_gen_eig(A, B)
        # 复数特征值
        A = rng.random((3, 3)) + 1j*rng.random((3, 3))
        # 调用 _check_gen_eig 方法，检查 A 的特征值
        self._check_gen_eig(A, None)
        B = rng.random((3, 3)) + 1j*rng.random((3, 3))
        # 调用 _check_gen_eig 方法，检查 A 和 B 的特征值
        self._check_gen_eig(A, B)
    # 定义一个测试方法，用于验证在不检查有限性的情况下的特征值和特征向量计算是否正确
    def test_check_finite(self):
        # 创建一个矩阵 `a`，包含三个行向量
        a = [[1, 2, 3], [1, 2, 3], [2, 5, 6]]
        # 调用 `eig` 函数计算特征值 `w` 和特征向量 `v`，设置 `check_finite=False` 禁用有限性检查
        w, v = eig(a, check_finite=False)
        # 精确的特征值列表，计算结果需要接近这些值
        exact_w = [(9+sqrt(93))/2, 0, (9-sqrt(93))/2]
        # 精确的第一个特征向量 `v0`
        v0 = array([1, 1, (1+sqrt(93)/3)/2])
        # 精确的第二个特征向量 `v1`
        v1 = array([3., 0, -1])
        # 精确的第三个特征向量 `v2`
        v2 = array([1, 1, (1-sqrt(93)/3)/2])
        # 将每个特征向量归一化为单位向量
        v0 = v0 / norm(v0)
        v1 = v1 / norm(v1)
        v2 = v2 / norm(v2)
        # 断言特征值 `w` 接近精确值 `exact_w`
        assert_array_almost_equal(w, exact_w)
        # 断言计算出的第 i 列特征向量 `v[:, i]` 与精确的第 i 个特征向量 `v0`, `v1`, `v2` 之间的相似性
        assert_array_almost_equal(v0, v[:, 0]*sign(v[0, 0]))
        assert_array_almost_equal(v1, v[:, 1]*sign(v[0, 1]))
        assert_array_almost_equal(v2, v[:, 2]*sign(v[0, 2]))
        # 对每个特征向量 `v[:, i]` 断言矩阵乘积 `a @ v[:, i]` 等于特征值 `w[i]` 乘以 `v[:, i]`
        for i in range(3):
            assert_array_almost_equal(a @ v[:, i], w[i]*v[:, i])

    # 定义一个测试方法，验证传入非方阵数组时是否会引发 ValueError 异常
    def test_not_square_error(self):
        """Check that passing a non-square array raises a ValueError."""
        # 创建一个非方阵数组 `A`
        A = np.arange(6).reshape(3, 2)
        # 断言调用 `eig` 函数时会引发 ValueError 异常
        assert_raises(ValueError, eig, A)

    # 定义一个测试方法，验证传入形状不匹配的数组时是否会引发 ValueError 异常
    def test_shape_mismatch(self):
        """Check that passing arrays with different shapes
        raises a ValueError."""
        # 创建两个数组 `A` 和 `B`
        A = eye(2)
        B = np.arange(9.0).reshape(3, 3)
        # 断言调用 `eig` 函数时会引发 ValueError 异常，分别传入 `A`, `B` 和 `B`, `A`
        assert_raises(ValueError, eig, A, B)
        assert_raises(ValueError, eig, B, A)

    # 定义一个测试方法，验证特定问题的修复是否成功，关于 gh-11577 的问题
    def test_gh_11577(self):
        # https://github.com/scipy/scipy/issues/11577
        # `A - lambda B` 应该具有特征值 4 和 8，这在某些平台上显然是破损的
        # 创建两个矩阵 `A` 和 `B`
        A = np.array([[12.0, 28.0, 76.0, 220.0],
                      [16.0, 32.0, 80.0, 224.0],
                      [24.0, 40.0, 88.0, 232.0],
                      [40.0, 56.0, 104.0, 248.0]], dtype='float64')
        B = np.array([[2.0, 4.0, 10.0, 28.0],
                      [3.0, 5.0, 11.0, 29.0],
                      [5.0, 7.0, 13.0, 31.0],
                      [9.0, 11.0, 17.0, 35.0]], dtype='float64')

        # 计算矩阵 `A` 和 `B` 的特征值 `D` 和特征向量 `V`
        D, V = eig(A, B)

        # 问题是不稳定的，另外两个特征值取决于 ATLAS/OpenBLAS 版本、编译器版本等
        # 详见 gh-11577 的讨论
        #
        # 注意：虽然 `assert_allclose(D[:2], [4, 8])` 看起来更符合，但是特征值的顺序在不同系统上也会有所不同。
        with np.testing.suppress_warnings() as sup:
            # isclose 函数在处理无穷大/NaN 值时会出错
            sup.filter(RuntimeWarning, "invalid value encountered in multiply")
            # 断言 `D` 中存在接近于 4.0 的特征值，允许误差 `atol=1e-14`
            assert np.isclose(D, 4.0, atol=1e-14).any()
            # 断言 `D` 中存在接近于 8.0 的特征值，允许误差 `atol=1e-14`
            assert np.isclose(D, 8.0, atol=1e-14).any()
    # 定义一个测试方法，测试空数组情况
    def test_empty(self, dt):
        # 创建一个空的 NumPy 数组 `a`，数据类型为 `dt`
        a = np.empty((0, 0), dtype=dt)
        # 对数组 `a` 进行特征值分解，返回特征值 `w` 和特征向量 `vr`
        w, vr = eig(a)

        # 对单位矩阵进行特征值分解，返回特征值 `w_n` 和特征向量 `vr_n`
        w_n, vr_n = eig(np.eye(2, dtype=dt))

        # 断言特征值 `w` 的形状应为 `(0,)`
        assert w.shape == (0,)
        # 断言特征值 `w` 的数据类型与 `w_n` 相同
        assert w.dtype == w_n.dtype

        # 断言特征向量 `vr` 应当接近一个空的 `(0, 0)` 数组
        assert_allclose(vr, np.empty((0, 0)))
        # 断言特征向量 `vr` 的形状应为 `(0, 0)`
        assert vr.shape == (0, 0)
        # 断言特征向量 `vr` 的数据类型与 `vr_n` 相同
        assert vr.dtype == vr_n.dtype

        # 对数组 `a` 进行特征值分解，使用 `homogeneous_eigvals=True` 参数
        w, vr = eig(a, homogeneous_eigvals=True)
        # 断言特征值 `w` 的形状应为 `(2, 0)`
        assert w.shape == (2, 0)
        # 断言特征值 `w` 的数据类型与 `w_n` 相同
        assert w.dtype == w_n.dtype

        # 断言特征向量 `vr` 的形状应为 `(0, 0)`
        assert vr.shape == (0, 0)
        # 断言特征向量 `vr` 的数据类型与 `vr_n` 相同
        assert vr.dtype == vr_n.dtype
class TestEigBanded:
    def setup_method(self):
        self.create_bandmat()

    #####################################################################

    def test_dsbev(self):
        """Compare dsbev eigenvalues and eigenvectors with
           the result of linalg.eig."""
        # 调用 dsbev 函数计算带状对称矩阵的特征值和特征向量
        w, evec, info = dsbev(self.bandmat_sym, compute_v=1)
        # 对特征值 w 和特征向量 evec 按特征值排序后进行比较
        evec_ = evec[:, argsort(w)]
        assert_array_almost_equal(sort(w), self.w_sym_lin)
        assert_array_almost_equal(abs(evec_), abs(self.evec_sym_lin))

    def test_dsbevd(self):
        """Compare dsbevd eigenvalues and eigenvectors with
           the result of linalg.eig."""
        # 调用 dsbevd 函数计算带状对称矩阵的特征值和特征向量
        w, evec, info = dsbevd(self.bandmat_sym, compute_v=1)
        # 对特征值 w 和特征向量 evec 按特征值排序后进行比较
        evec_ = evec[:, argsort(w)]
        assert_array_almost_equal(sort(w), self.w_sym_lin)
        assert_array_almost_equal(abs(evec_), abs(self.evec_sym_lin))

    def test_dsbevx(self):
        """Compare dsbevx eigenvalues and eigenvectors
           with the result of linalg.eig."""
        # 获取对称矩阵的形状，并将其赋值给 N
        N, N = shape(self.sym_mat)
        # 使用 dsbevx 函数计算带状对称矩阵的特征值和特征向量
        # Achtung: 参数 0.0,0.0,range?
        w, evec, num, ifail, info = dsbevx(self.bandmat_sym, 0.0, 0.0, 1, N,
                                           compute_v=1, range=2)
        # 对特征值 w 和特征向量 evec 按特征值排序后进行比较
        evec_ = evec[:, argsort(w)]
        assert_array_almost_equal(sort(w), self.w_sym_lin)
        assert_array_almost_equal(abs(evec_), abs(self.evec_sym_lin))

    def test_zhbevd(self):
        """Compare zhbevd eigenvalues and eigenvectors
           with the result of linalg.eig."""
        # 调用 zhbevd 函数计算带状 Hermite 矩阵的特征值和特征向量
        w, evec, info = zhbevd(self.bandmat_herm, compute_v=1)
        # 对特征值 w 和特征向量 evec 按特征值排序后进行比较
        evec_ = evec[:, argsort(w)]
        assert_array_almost_equal(sort(w), self.w_herm_lin)
        assert_array_almost_equal(abs(evec_), abs(self.evec_herm_lin))

    def test_zhbevx(self):
        """Compare zhbevx eigenvalues and eigenvectors
           with the result of linalg.eig."""
        # 获取 Hermite 矩阵的形状，并将其赋值给 N
        N, N = shape(self.herm_mat)
        # 使用 zhbevx 函数计算带状 Hermite 矩阵的特征值和特征向量
        # Achtung: 参数 0.0,0.0,range?
        w, evec, num, ifail, info = zhbevx(self.bandmat_herm, 0.0, 0.0, 1, N,
                                           compute_v=1, range=2)
        # 对特征值 w 和特征向量 evec 按特征值排序后进行比较
        evec_ = evec[:, argsort(w)]
        assert_array_almost_equal(sort(w), self.w_herm_lin)
        assert_array_almost_equal(abs(evec_), abs(self.evec_herm_lin))
    def test_eigvals_banded(self):
        """Compare eigenvalues of eigvals_banded with those of linalg.eig."""
        # 计算对称带状矩阵的特征值
        w_sym = eigvals_banded(self.bandmat_sym)
        # 取实部并排序
        w_sym = w_sym.real
        # 断言特征值数组近似相等
        assert_array_almost_equal(sort(w_sym), self.w_sym_lin)

        # 计算厄米带状矩阵的特征值
        w_herm = eigvals_banded(self.bandmat_herm)
        # 取实部并排序
        w_herm = w_herm.real
        # 断言特征值数组近似相等
        assert_array_almost_equal(sort(w_herm), self.w_herm_lin)

        # 按照索引范围提取特征值
        ind1 = 2
        ind2 = np.longlong(6)
        w_sym_ind = eigvals_banded(self.bandmat_sym,
                                   select='i', select_range=(ind1, ind2))
        # 断言提取的特征值数组近似相等
        assert_array_almost_equal(sort(w_sym_ind),
                                  self.w_sym_lin[ind1:ind2+1])
        w_herm_ind = eigvals_banded(self.bandmat_herm,
                                    select='i', select_range=(ind1, ind2))
        # 断言提取的特征值数组近似相等
        assert_array_almost_equal(sort(w_herm_ind),
                                  self.w_herm_lin[ind1:ind2+1])

        # 按照数值范围提取特征值
        v_lower = self.w_sym_lin[ind1] - 1.0e-5
        v_upper = self.w_sym_lin[ind2] + 1.0e-5
        w_sym_val = eigvals_banded(self.bandmat_sym,
                                   select='v', select_range=(v_lower, v_upper))
        # 断言提取的特征值数组近似相等
        assert_array_almost_equal(sort(w_sym_val),
                                  self.w_sym_lin[ind1:ind2+1])

        v_lower = self.w_herm_lin[ind1] - 1.0e-5
        v_upper = self.w_herm_lin[ind2] + 1.0e-5
        w_herm_val = eigvals_banded(self.bandmat_herm,
                                    select='v',
                                    select_range=(v_lower, v_upper))
        # 断言提取的特征值数组近似相等
        assert_array_almost_equal(sort(w_herm_val),
                                  self.w_herm_lin[ind1:ind2+1])

        # 计算对称带状矩阵的特征值（不检查有限性）
        w_sym = eigvals_banded(self.bandmat_sym, check_finite=False)
        # 取实部并排序
        w_sym = w_sym.real
        # 断言特征值数组近似相等
        assert_array_almost_equal(sort(w_sym), self.w_sym_lin)

    def test_dgbtrf(self):
        """Compare dgbtrf  LU factorisation with the LU factorisation result
           of linalg.lu."""
        # 获取实矩阵的形状
        M, N = shape(self.real_mat)
        # 使用 dgbtrf 进行对称带状矩阵的 LU 分解
        lu_symm_band, ipiv, info = dgbtrf(self.bandmat_real, self.KL, self.KU)

        # 从 lu_symm_band 中提取矩阵 U
        u = diag(lu_symm_band[2*self.KL, :])
        for i in range(self.KL + self.KU):
            u += diag(lu_symm_band[2*self.KL-1-i, i+1:N], i+1)

        # 使用 linalg.lu 进行 LU 分解
        p_lin, l_lin, u_lin = lu(self.real_mat, permute_l=0)
        # 断言 U 矩阵近似相等
        assert_array_almost_equal(u, u_lin)
    ``
class TestEigTridiagonal:
    def setup_method(self):
        self.create_trimat()

    def create_trimat(self):
        """Create the full matrix `self.fullmat`, `self.d`, and `self.e`."""
        N = 10

        # symmetric band matrix
        self.d = full(N, 1.0)  # 创建大小为 N 的对角线向量 self.d，元素值为 1.0
        self.e = full(N-1, -1.0)  # 创建大小为 N-1 的副对角线向量 self.e，元素值为 -1.0
        self.full_mat = (diag(self.d) + diag(self.e, -1) + diag(self.e, 1))  # 构建对称带状矩阵 self.full_mat

        # 计算 self.full_mat 的特征值和特征向量
        ew, ev = linalg.eig(self.full_mat)
        ew = ew.real  # 提取特征值的实部
        args = argsort(ew)  # 对特征值排序并记录索引
        self.w = ew[args]  # 按索引重新排列特征值
        self.evec = ev[:, args]  # 按索引重新排列特征向量

    def test_degenerate(self):
        """Test error conditions."""
        # 错误的尺寸
        assert_raises(ValueError, eigvalsh_tridiagonal, self.d, self.e[:-1])
        # 必须是实数
        assert_raises(TypeError, eigvalsh_tridiagonal, self.d, self.e * 1j)
        # 错误的驱动程序
        assert_raises(TypeError, eigvalsh_tridiagonal, self.d, self.e,
                      lapack_driver=1.)
        assert_raises(ValueError, eigvalsh_tridiagonal, self.d, self.e,
                      lapack_driver='foo')
        # 错误的边界条件
        assert_raises(ValueError, eigvalsh_tridiagonal, self.d, self.e,
                      select='i', select_range=(0, -1))

    def test_eigvalsh_tridiagonal(self):
        """Compare eigenvalues of eigvalsh_tridiagonal with those of eig."""
        # 不能使用 ?STERF 进行子选择
        for driver in ('sterf', 'stev', 'stebz', 'stemr', 'auto'):
            w = eigvalsh_tridiagonal(self.d, self.e, lapack_driver=driver)  # 计算三对角矩阵的特征值
            assert_array_almost_equal(sort(w), self.w)  # 断言计算得到的特征值与预期的特征值 self.w 排序后相近

        for driver in ('sterf', 'stev'):
            assert_raises(ValueError, eigvalsh_tridiagonal, self.d, self.e,
                          lapack_driver='stev', select='i',
                          select_range=(0, 1))
        for driver in ('stebz', 'stemr', 'auto'):
            # 提取特定索引范围内的特征值
            w_ind = eigvalsh_tridiagonal(
                self.d, self.e, select='i', select_range=(0, len(self.d)-1),
                lapack_driver=driver)
            assert_array_almost_equal(sort(w_ind), self.w)

            # 提取特定索引范围内的特征值
            ind1 = 2
            ind2 = 6
            w_ind = eigvalsh_tridiagonal(
                self.d, self.e, select='i', select_range=(ind1, ind2),
                lapack_driver=driver)
            assert_array_almost_equal(sort(w_ind), self.w[ind1:ind2+1])

            # 提取特定值范围内的特征值
            v_lower = self.w[ind1] - 1.0e-5
            v_upper = self.w[ind2] + 1.0e-5
            w_val = eigvalsh_tridiagonal(
                self.d, self.e, select='v', select_range=(v_lower, v_upper),
                lapack_driver=driver)
            assert_array_almost_equal(sort(w_val), self.w[ind1:ind2+1])
    def test_eigh_tridiagonal(self):
        """比较 eigh_tridiagonal 的特征值和特征向量
           与 eig 函数的结果。"""
        # 当请求特征向量时不能使用 ?STERF 驱动
        assert_raises(ValueError, eigh_tridiagonal, self.d, self.e,
                      lapack_driver='sterf')
        # 使用不同的驱动进行测试
        for driver in ('stebz', 'stev', 'stemr', 'auto'):
            # 调用 eigh_tridiagonal 函数获取特征值 w 和特征向量 evec
            w, evec = eigh_tridiagonal(self.d, self.e, lapack_driver=driver)
            # 根据特征值 w 对特征向量 evec 进行排序
            evec_ = evec[:, argsort(w)]
            # 断言特征值 w 排序后与预期结果 self.w 几乎相等
            assert_array_almost_equal(sort(w), self.w)
            # 断言特征向量 evec 的绝对值与预期结果 self.evec 的绝对值几乎相等
            assert_array_almost_equal(abs(evec_), abs(self.evec))

        # 当使用 'stev' 驱动且指定了选择条件时，抛出 ValueError 异常
        assert_raises(ValueError, eigh_tridiagonal, self.d, self.e,
                      lapack_driver='stev', select='i', select_range=(0, 1))
        # 再次使用不同的驱动进行测试
        for driver in ('stebz', 'stemr', 'auto'):
            # 根据索引范围提取特征值和特征向量
            ind1 = 0
            ind2 = len(self.d)-1
            w, evec = eigh_tridiagonal(
                self.d, self.e, select='i', select_range=(ind1, ind2),
                lapack_driver=driver)
            # 断言特征值 w 排序后与预期结果 self.w 几乎相等
            assert_array_almost_equal(sort(w), self.w)
            # 断言特征向量 evec 的绝对值与预期结果 self.evec 的绝对值几乎相等
            assert_array_almost_equal(abs(evec), abs(self.evec))
            # 设置新的索引范围
            ind1 = 2
            ind2 = 6
            w, evec = eigh_tridiagonal(
                self.d, self.e, select='i', select_range=(ind1, ind2),
                lapack_driver=driver)
            # 断言特征值 w 排序后与预期结果 self.w 的指定范围内几乎相等
            assert_array_almost_equal(sort(w), self.w[ind1:ind2+1])
            # 断言特征向量 evec 的绝对值与预期结果 self.evec 的指定范围内几乎相等
            assert_array_almost_equal(abs(evec),
                                      abs(self.evec[:, ind1:ind2+1]))

            # 根据值范围提取特征值和特征向量
            v_lower = self.w[ind1] - 1.0e-5
            v_upper = self.w[ind2] + 1.0e-5
            w, evec = eigh_tridiagonal(
                self.d, self.e, select='v', select_range=(v_lower, v_upper),
                lapack_driver=driver)
            # 断言特征值 w 排序后与预期结果 self.w 的指定范围内几乎相等
            assert_array_almost_equal(sort(w), self.w[ind1:ind2+1])
            # 断言特征向量 evec 的绝对值与预期结果 self.evec 的指定范围内几乎相等
            assert_array_almost_equal(abs(evec),
                                      abs(self.evec[:, ind1:ind2+1]))

    def test_eigh_tridiagonal_1x1(self):
        """查看 gh-20075"""
        # 创建 1x1 的特征三对角矩阵
        a = np.array([-2.0])
        b = np.array([])
        # 获取只有特征值的结果 x
        x = eigh_tridiagonal(a, b, eigvals_only=True)
        # 断言 x 的维度为 1
        assert x.ndim == 1
        # 断言 x 与预期值 a 几乎相等
        assert_allclose(x, a)
        # 获取特征值 x 和特征向量 V 的结果
        x, V = eigh_tridiagonal(a, b, select="i", select_range=(0, 0))
        # 断言 x 的维度为 1
        assert x.ndim == 1
        # 断言 V 的维度为 2
        assert V.ndim == 2
        # 断言 x 与预期值 a 几乎相等
        assert_allclose(x, a)
        # 断言 V 与预期值 [[1.]] 几乎相等
        assert_allclose(V, array([[1.]]))

        # 获取特征值 x 和特征向量 V 的结果
        x, V = eigh_tridiagonal(a, b, select="v", select_range=(-2, 0))
        # 断言 x 的大小为 0
        assert x.size == 0
        # 断言 x 的形状为 (0,)
        assert x.shape == (0,)
        # 断言 V 的形状为 (1, 0)
        assert V.shape == (1, 0)
class TestEigh:
    # 设置测试类的初始化方法
    def setup_class(self):
        # 设置随机种子为1234
        np.random.seed(1234)

    # 测试错误输入情况
    def test_wrong_inputs(self):
        # 非方阵 a
        assert_raises(ValueError, eigh, np.ones([1, 2]))
        # 非方阵 b
        assert_raises(ValueError, eigh, np.ones([2, 2]), np.ones([2, 1]))
        # 不兼容的 a, b 大小
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([2, 2]))
        # 广义问题的错误类型参数
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]),
                      type=4)
        # 请求值和索引子集
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]),
                      subset_by_value=[1, 2], subset_by_index=[2, 4])
        # 无效的上限索引规范
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]),
                      subset_by_index=[0, 4])
        # 无效的下限索引
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]),
                      subset_by_index=[-2, 2])
        # 无效的索引规范 #2
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]),
                      subset_by_index=[2, 0])
        # 无效的值规范
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]),
                      subset_by_value=[2, 0])
        # 无效的驱动程序名称
        assert_raises(ValueError, eigh, np.ones([2, 2]), driver='wrong')
        # 选择广义驱动程序但没有 b
        assert_raises(ValueError, eigh, np.ones([3, 3]), None, driver='gvx')
        # 带 b 的标准驱动程序
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]),
                      driver='evr')
        # 从无效驱动程序请求子集
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]),
                      driver='gvd', subset_by_index=[1, 2])
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]),
                      driver='gvd', subset_by_index=[1, 2])

    # 测试 b 非正定情况
    def test_nonpositive_b(self):
        assert_raises(LinAlgError, eigh, np.ones([3, 3]), np.ones([3, 3]))

    # 基于值的子集在传统测试 test_eigh() 中完成
    def test_value_subsets(self):
        for ind, dt in enumerate(DTYPES):

            a = _random_hermitian_matrix(20, dtype=dt)
            w, v = eigh(a, subset_by_value=[-2, 2])
            assert_equal(v.shape[1], len(w))
            assert all((w > -2) & (w < 2))

            b = _random_hermitian_matrix(20, posdef=True, dtype=dt)
            w, v = eigh(a, b, subset_by_value=[-2, 2])
            assert_equal(v.shape[1], len(w))
            assert all((w > -2) & (w < 2))

    # 测试 eigh 方法的整数输入
    def test_eigh_integer(self):
        a = array([[1, 2], [2, 7]])
        b = array([[3, 1], [1, 5]])
        w, z = eigh(a)
        w, z = eigh(a, b)
    # 测试稀疏矩阵情况下的特征值和特征向量计算是否能够正确地拒绝无法处理的输入。
    def test_eigh_of_sparse(self):
        import scipy.sparse
        # 创建一个稀疏的单位矩阵，转换为压缩列格式
        a = scipy.sparse.identity(2).tocsc()
        # 将稀疏矩阵转换为至少二维的数组
        b = np.atleast_2d(a)
        # 检查是否会引发 ValueError 异常，要求 eigh 处理稀疏矩阵 a
        assert_raises(ValueError, eigh, a)
        # 检查是否会引发 ValueError 异常，要求 eigh 处理二维数组 b
        assert_raises(ValueError, eigh, b)

    # 使用不同的驱动程序和数据类型测试标准的特征值和特征向量计算
    @pytest.mark.parametrize('dtype_', DTYPES)
    @pytest.mark.parametrize('driver', ("ev", "evd", "evr", "evx"))
    def test_various_drivers_standard(self, driver, dtype_):
        # 生成一个随机的 Hermite 矩阵，数据类型为 dtype_
        a = _random_hermitian_matrix(n=20, dtype=dtype_)
        # 计算 Hermite 矩阵 a 的特征值和特征向量，使用指定的驱动程序
        w, v = eigh(a, driver=driver)
        # 检查是否满足数值精度，要求 a @ v 等于 v * w
        assert_allclose(a @ v - (v * w), 0.,
                        atol=1000*np.finfo(dtype_).eps,
                        rtol=0.)

    # 使用不同的驱动程序测试 1x1 矩阵的特征值和特征向量计算
    @pytest.mark.parametrize('driver', ("ev", "evd", "evr", "evx"))
    def test_1x1_lwork(self, driver):
        # 计算 [[1]] 的特征值和特征向量
        w, v = eigh([[1]], driver=driver)
        # 检查特征值是否等于 1.0，数值精度为 1e-15
        assert_allclose(w, array([1.]), atol=1e-15)
        # 检查特征向量是否等于 [[1.]]，数值精度为 1e-15
        assert_allclose(v, array([[1.]]), atol=1e-15)

        # 复数情况
        # 计算 [[1j]] 的特征值和特征向量
        w, v = eigh([[1j]], driver=driver)
        # 检查特征值是否等于 0，数值精度为 1e-15
        assert_allclose(w, array([0]), atol=1e-15)
        # 检查特征向量是否等于 [[1.]]，数值精度为 1e-15
        assert_allclose(v, array([[1.]]), atol=1e-15)

    # 使用不同类型和驱动程序测试广义特征值和特征向量计算
    @pytest.mark.parametrize('type', (1, 2, 3))
    @pytest.mark.parametrize('driver', ("gv", "gvd", "gvx"))
    def test_various_drivers_generalized(self, driver, type):
        # 设置数值精度
        atol = np.spacing(5000.)
        # 生成一个随机的 Hermite 矩阵 a 和正定矩阵 b
        a = _random_hermitian_matrix(20)
        b = _random_hermitian_matrix(20, posdef=True)
        # 计算广义特征值和特征向量，使用指定的驱动程序和类型
        w, v = eigh(a=a, b=b, driver=driver, type=type)
        # 根据不同类型进行检查
        if type == 1:
            # 检查是否满足数值精度，要求 a @ v 等于 w*(b @ v)
            assert_allclose(a @ v - w*(b @ v), 0., atol=atol, rtol=0.)
        elif type == 2:
            # 检查是否满足数值精度，要求 a @ b @ v 等于 v * w
            assert_allclose(a @ b @ v - v * w, 0., atol=atol, rtol=0.)
        else:
            # 检查是否满足数值精度，要求 b @ a @ v 等于 v * w
            assert_allclose(b @ a @ v - v * w, 0., atol=atol, rtol=0.)

    # 测试 eigvalsh 函数的新参数
    def test_eigvalsh_new_args(self):
        # 生成一个随机的 Hermite 矩阵 a
        a = _random_hermitian_matrix(5)
        # 使用索引子集计算特征值
        w = eigvalsh(a, subset_by_index=[1, 2])
        # 检查特征值的长度是否为 2
        assert_equal(len(w), 2)

        # 再次使用相同的索引子集计算特征值
        w2 = eigvalsh(a, subset_by_index=[1, 2])
        # 检查特征值的长度是否为 2
        assert_equal(len(w2), 2)
        # 检查两次计算得到的特征值是否近似相等
        assert_allclose(w, w2)

        # 生成一个对角矩阵 b
        b = np.diag([1, 1.2, 1.3, 1.5, 2])
        # 使用值子集计算特征值
        w3 = eigvalsh(b, subset_by_value=[1, 1.4])
        # 检查特征值的长度是否为 2
        assert_equal(len(w3), 2)
        # 检查计算得到的特征值是否近似等于 [1.2, 1.3]
        assert_allclose(w3, np.array([1.2, 1.3]))

    # 测试空矩阵情况下的特征值和特征向量计算
    @pytest.mark.parametrize('dt', [int, float, np.float32, complex, np.complex64])
    def test_empty(self, dt):
        # 生成一个空的数组 a，数据类型为 dt
        a = np.empty((0, 0), dtype=dt)
        # 计算空数组的特征值和特征向量
        w, v = eigh(a)

        # 计算单位矩阵的特征值和特征向量
        w_n, v_n = eigh(np.eye(2, dtype=dt))

        # 检查特征值的形状是否为 (0,)
        assert w.shape == (0,)
        # 检查特征值的数据类型是否与 w_n 的数据类型相同
        assert w.dtype == w_n.dtype

        # 检查特征向量的形状是否为 (0, 0)
        assert v.shape == (0, 0)
        # 检查特征向量的数据类型是否与 v_n 的数据类型相同
        assert v.dtype == v_n.dtype

        # 计算只返回特征值的情况
        w = eigh(a, eigvals_only=True)
        # 检查特征值的形状是否为 (0,)
        assert_allclose(w, np.empty((0,)))

        # 再次检查特征值的形状是否为 (0,)
        assert w.shape == (0,)
        # 再次检查特征值的数据类型是否与 w_n 的数据类型相同
        assert w.dtype == w_n.dtype
# 定义一个测试类 TestSVD_GESDD，用于测试 SVD 算法的不同情况
class TestSVD_GESDD:
    # 指定 LAPACK 驱动程序为 'gesdd'
    lapack_driver = 'gesdd'

    # 测试特殊情况：输入矩阵为单元素列表时，期望引发 TypeError 异常
    def test_degenerate(self):
        # 断言调用 svd 函数时，传入单元素列表会引发 TypeError 异常
        assert_raises(TypeError, svd, [[1.]], lapack_driver=1.)
        # 断言调用 svd 函数时，传入不支持的 lapack_driver 参数会引发 ValueError 异常
        assert_raises(ValueError, svd, [[1.]], lapack_driver='foo')

    # 测试简单情况：对于给定矩阵 a 的 SVD 分解
    def test_simple(self):
        # 定义一个简单的测试矩阵 a
        a = [[1, 2, 3], [1, 20, 3], [2, 5, 6]]
        # 遍历是否返回完整矩阵的选项
        for full_matrices in (True, False):
            # 调用 svd 函数进行奇异值分解，得到左奇异矩阵 u，奇异值向量 s，右奇异矩阵 vh
            u, s, vh = svd(a, full_matrices=full_matrices, lapack_driver=self.lapack_driver)
            # 断言 u 的转置乘以 u 等于单位矩阵
            assert_array_almost_equal(u.T @ u, eye(3))
            # 断言 vh 的转置乘以 vh 等于单位矩阵
            assert_array_almost_equal(vh.T @ vh, eye(3))
            # 构造对角阵 sigma，将奇异值 s 放在对角线上
            sigma = zeros((u.shape[0], vh.shape[0]), s.dtype.char)
            for i in range(len(s)):
                sigma[i, i] = s[i]
            # 断言 u * sigma * vh 等于原始矩阵 a
            assert_array_almost_equal(u @ sigma @ vh, a)

    # 测试简单情况：对于存在相同奇异值的矩阵 a 的 SVD 分解
    def test_simple_singular(self):
        # 定义一个存在相同奇异值的测试矩阵 a
        a = [[1, 2, 3], [1, 2, 3], [2, 5, 6]]
        # 遍历是否返回完整矩阵的选项
        for full_matrices in (True, False):
            # 调用 svd 函数进行奇异值分解，得到左奇异矩阵 u，奇异值向量 s，右奇异矩阵 vh
            u, s, vh = svd(a, full_matrices=full_matrices, lapack_driver=self.lapack_driver)
            # 断言 u 的转置乘以 u 等于单位矩阵
            assert_array_almost_equal(u.T @ u, eye(3))
            # 断言 vh 的转置乘以 vh 等于单位矩阵
            assert_array_almost_equal(vh.T @ vh, eye(3))
            # 构造对角阵 sigma，将奇异值 s 放在对角线上
            sigma = zeros((u.shape[0], vh.shape[0]), s.dtype.char)
            for i in range(len(s)):
                sigma[i, i] = s[i]
            # 断言 u * sigma * vh 等于原始矩阵 a
            assert_array_almost_equal(u @ sigma @ vh, a)

    # 测试简单情况：对于欠定矩阵 a 的 SVD 分解
    def test_simple_underdet(self):
        # 定义一个欠定矩阵 a
        a = [[1, 2, 3], [4, 5, 6]]
        # 遍历是否返回完整矩阵的选项
        for full_matrices in (True, False):
            # 调用 svd 函数进行奇异值分解，得到左奇异矩阵 u，奇异值向量 s，右奇异矩阵 vh
            u, s, vh = svd(a, full_matrices=full_matrices, lapack_driver=self.lapack_driver)
            # 断言 u 的转置乘以 u 等于单位矩阵
            assert_array_almost_equal(u.T @ u, eye(u.shape[0]))
            # 构造对角阵 sigma，将奇异值 s 放在对角线上
            sigma = zeros((u.shape[0], vh.shape[0]), s.dtype.char)
            for i in range(len(s)):
                sigma[i, i] = s[i]
            # 断言 u * sigma * vh 等于原始矩阵 a
            assert_array_almost_equal(u @ sigma @ vh, a)

    # 测试简单情况：对于超定矩阵 a 的 SVD 分解
    def test_simple_overdet(self):
        # 定义一个超定矩阵 a
        a = [[1, 2], [4, 5], [3, 4]]
        # 遍历是否返回完整矩阵的选项
        for full_matrices in (True, False):
            # 调用 svd 函数进行奇异值分解，得到左奇异矩阵 u，奇异值向量 s，右奇异矩阵 vh
            u, s, vh = svd(a, full_matrices=full_matrices, lapack_driver=self.lapack_driver)
            # 断言 u 的转置乘以 u 等于单位矩阵
            assert_array_almost_equal(u.T @ u, eye(u.shape[1]))
            # 断言 vh 的转置乘以 vh 等于单位矩阵
            assert_array_almost_equal(vh.T @ vh, eye(2))
            # 构造对角阵 sigma，将奇异值 s 放在对角线上
            sigma = zeros((u.shape[1], vh.shape[0]), s.dtype.char)
            for i in range(len(s)):
                sigma[i, i] = s[i]
            # 断言 u * sigma * vh 等于原始矩阵 a
            assert_array_almost_equal(u @ sigma @ vh, a)
    # 测试随机生成的数据
    def test_random(self):
        # 创建指定种子的随机数生成器
        rng = np.random.RandomState(1234)
        # 定义矩阵的维度
        n = 20
        m = 15
        # 执行三次循环以测试不同情况
        for i in range(3):
            # 对于不同的数组结构进行测试
            for a in [rng.random([n, m]), rng.random([m, n])]:
                # 对于两种不同的全矩阵计算方式进行测试
                for full_matrices in (True, False):
                    # 执行奇异值分解
                    u, s, vh = svd(a, full_matrices=full_matrices,
                                   lapack_driver=self.lapack_driver)
                    # 断言奇异向量U的转置乘以自身接近单位矩阵
                    assert_array_almost_equal(u.T @ u, eye(u.shape[1]))
                    # 断言奇异向量V的转置乘以自身接近单位矩阵
                    assert_array_almost_equal(vh @ vh.T, eye(vh.shape[0]))
                    # 构造奇异值矩阵Sigma
                    sigma = zeros((u.shape[1], vh.shape[0]), s.dtype.char)
                    for i in range(len(s)):
                        sigma[i, i] = s[i]
                    # 断言恢复的矩阵接近原始矩阵a
                    assert_array_almost_equal(u @ sigma @ vh, a)

    # 测试包含复数的简单矩阵
    def test_simple_complex(self):
        # 定义包含复数的矩阵a
        a = [[1, 2, 3], [1, 2j, 3], [2, 5, 6]]
        # 对于两种不同的全矩阵计算方式进行测试
        for full_matrices in (True, False):
            # 执行奇异值分解
            u, s, vh = svd(a, full_matrices=full_matrices,
                           lapack_driver=self.lapack_driver)
            # 断言奇异向量U的共轭转置乘以自身接近单位矩阵
            assert_array_almost_equal(u.conj().T @ u, eye(u.shape[1]))
            # 断言奇异向量V的共轭转置乘以自身接近单位矩阵
            assert_array_almost_equal(vh.conj().T @ vh, eye(vh.shape[0]))
            # 构造奇异值矩阵Sigma
            sigma = zeros((u.shape[0], vh.shape[0]), s.dtype.char)
            for i in range(len(s)):
                sigma[i, i] = s[i]
            # 断言恢复的矩阵接近原始矩阵a
            assert_array_almost_equal(u @ sigma @ vh, a)

    # 测试包含复数的随机生成数据
    def test_random_complex(self):
        # 创建指定种子的随机数生成器
        rng = np.random.RandomState(1234)
        # 定义矩阵的维度
        n = 20
        m = 15
        # 执行三次循环以测试不同情况
        for i in range(3):
            # 对于两种不同的全矩阵计算方式进行测试
            for full_matrices in (True, False):
                # 对于两种不同的数组结构进行测试，并加入复数部分
                for a in [rng.random([n, m]), rng.random([m, n])]:
                    a = a + 1j * rng.random(list(a.shape))
                    # 执行奇异值分解
                    u, s, vh = svd(a, full_matrices=full_matrices,
                                   lapack_driver=self.lapack_driver)
                    # 断言奇异向量U的共轭转置乘以自身接近单位矩阵
                    assert_array_almost_equal(u.conj().T @ u,
                                              eye(u.shape[1]))
                    # 注释掉的代码行预期失败，当数组形状为[m,n]时
                    # assert_array_almost_equal(vh.conj().T @ vh,
                    #                        eye(len(vh),dtype=vh.dtype.char))
                    # 构造奇异值矩阵Sigma
                    sigma = zeros((u.shape[1], vh.shape[0]), s.dtype.char)
                    for i in range(len(s)):
                        sigma[i, i] = s[i]
                    # 断言恢复的矩阵接近原始矩阵a
                    assert_array_almost_equal(u @ sigma @ vh, a)

    # 测试解决问题 #1580
    def test_crash_1580(self):
        # 创建指定种子的随机数生成器
        rng = np.random.RandomState(1234)
        # 定义不同尺寸的矩阵
        sizes = [(13, 23), (30, 50), (60, 100)]
        # 对于不同的矩阵大小和数据类型进行测试
        for sz in sizes:
            for dt in [np.float32, np.float64, np.complex64, np.complex128]:
                # 生成随机数据并转换为指定数据类型的数组
                a = rng.rand(*sz).astype(dt)
                # 执行奇异值分解，预期不会崩溃
                svd(a, lapack_driver=self.lapack_driver)
    def test_check_finite(self):
        # 创建一个测试用的二维数组
        a = [[1, 2, 3], [1, 20, 3], [2, 5, 6]]
        # 对数组进行奇异值分解，禁用有限性检查，使用给定的 LAPACK 驱动器
        u, s, vh = svd(a, check_finite=False, lapack_driver=self.lapack_driver)
        # 断言 u 的转置乘以 u 等于单位矩阵
        assert_array_almost_equal(u.T @ u, eye(3))
        # 断言 vh 的转置乘以 vh 等于单位矩阵
        assert_array_almost_equal(vh.T @ vh, eye(3))
        # 创建一个零矩阵，用来构建奇异值矩阵 sigma
        sigma = zeros((u.shape[0], vh.shape[0]), s.dtype.char)
        # 构建奇异值矩阵 sigma
        for i in range(len(s)):
            sigma[i, i] = s[i]
        # 断言 u @ sigma @ vh 等于原始数组 a
        assert_array_almost_equal(u @ sigma @ vh, a)

    def test_gh_5039(self):
        # 这是一个针对 https://github.com/scipy/scipy/issues/5039 的烟雾测试
        #
        # 以下代码据报道会引发 "ValueError: On entry to DGESDD
        # parameter number 12 had an illegal value" 错误。
        # `interp1d([1,2,3,4], [1,2,3,4], kind='cubic')`
        # 据报道，这种情况只在 LAPACK 3.0.3 上出现。
        #
        # 下面的矩阵来自于 interpolate._find_smoothest 中对
        # `B = _fitpack._bsplmat(order, xk)` 的调用
        b = np.array(
            [[0.16666667, 0.66666667, 0.16666667, 0., 0., 0.],
             [0., 0.16666667, 0.66666667, 0.16666667, 0., 0.],
             [0., 0., 0.16666667, 0.66666667, 0.16666667, 0.],
             [0., 0., 0., 0.16666667, 0.66666667, 0.16666667]])
        # 对矩阵 b 进行奇异值分解，使用给定的 LAPACK 驱动器
        svd(b, lapack_driver=self.lapack_driver)

    @pytest.mark.skipif(not HAS_ILP64, reason="64-bit LAPACK required")
    @pytest.mark.slow
    def test_large_matrix(self):
        # 检查可用的自由内存，至少需要 17000 MB
        check_free_memory(free_mb=17000)
        # 创建一个非常大的零矩阵 A，其中只有最后一个元素为 1
        A = np.zeros([1, 2**31], dtype=np.float32)
        A[0, -1] = 1
        # 对矩阵 A 进行奇异值分解，不返回完整的 u 和 v 矩阵
        u, s, vh = svd(A, full_matrices=False)
        # 断言奇异值 s 的第一个元素接近 1.0
        assert_allclose(s[0], 1.0)
        # 断言 u 的第一个行向量的第一个元素乘以 vh 的最后一列的第一个元素接近 1.0
        assert_allclose(u[0, 0] * vh[0, -1], 1.0)

    @pytest.mark.parametrize("m", [0, 1, 2])
    @pytest.mark.parametrize("n", [0, 1, 2])
    @pytest.mark.parametrize('dtype', DTYPES)
    def test_shape_dtype(self, m, n, dtype):
        # 创建一个指定形状和数据类型的零矩阵 a
        a = np.zeros((m, n), dtype=dtype)
        # 计算奇异值分解，并返回完整的 u、s、v 矩阵
        u, s, v = svd(a)
        # 断言 u 的形状为 (m, m)，数据类型为指定的 dtype
        assert_equal(u.shape, (m, m))
        assert_equal(u.dtype, dtype)
        # 断言 s 的形状为 (k,)，数据类型为实数类型的字符表示
        k = min(m, n)
        dchar = a.dtype.char
        real_dchar = dchar.lower() if dchar in 'FD' else dchar
        assert_equal(s.shape, (k,))
        assert_equal(s.dtype, np.dtype(real_dchar))
        # 断言 v 的形状为 (n, n)，数据类型为指定的 dtype
        assert_equal(v.shape, (n, n))
        assert_equal(v.dtype, dtype)

        # 计算奇异值分解，不返回完整的 u 和 v 矩阵
        u, s, v = svd(a, full_matrices=False)
        # 断言 u 的形状为 (m, k)，数据类型为指定的 dtype
        assert_equal(u.shape, (m, k))
        assert_equal(u.dtype, dtype)
        # 断言 s 的形状为 (k,)，数据类型为实数类型的字符表示
        assert_equal(s.shape, (k,))
        assert_equal(s.dtype, np.dtype(real_dchar))
        # 断言 v 的形状为 (k, n)，数据类型为指定的 dtype
        assert_equal(v.shape, (k, n))
        assert_equal(v.dtype, dtype)

        # 计算奇异值，不返回 u 和 v 矩阵
        s = svd(a, compute_uv=False)
        # 断言 s 的形状为 (k,)，数据类型为实数类型的字符表示
        assert_equal(s.shape, (k,))
        assert_equal(s.dtype, np.dtype(real_dchar))

    @pytest.mark.parametrize('dt', [int, float, np.float32, complex, np.complex64])
    @pytest.mark.parametrize(("m", "n"), [(0, 0), (0, 2), (2, 0)])
    # 定义一个测试方法，用于测试空矩阵情况下的奇异值分解
    def test_empty(self, dt, m, n):
        # 创建一个单位矩阵，数据类型为指定的 dt
        a0 = np.eye(3, dtype=dt)
        # 对单位矩阵进行奇异值分解，返回左奇异向量、奇异值和右奇异向量
        u0, s0, v0 = svd(a0)

        # 创建一个空矩阵，形状为 (m, n)，数据类型为指定的 dt
        a = np.empty((m, n), dtype=dt)
        # 对空矩阵进行奇异值分解，返回左奇异向量、奇异值和右奇异向量
        u, s, v = svd(a)
        # 断言左奇异向量接近单位矩阵
        assert_allclose(u, np.identity(m))
        # 断言奇异值接近空数组
        assert_allclose(s, np.empty((0,)))
        # 断言右奇异向量接近单位矩阵
        assert_allclose(v, np.identity(n))

        # 断言左奇异向量的数据类型与 a0 中的相同
        assert u.dtype == u0.dtype
        # 断言右奇异向量的数据类型与 a0 中的相同
        assert v.dtype == v0.dtype
        # 断言奇异值的数据类型与 a0 中的相同
        assert s.dtype == s0.dtype

        # 对空矩阵进行奇异值分解，限制返回的奇异向量数量
        u, s, v = svd(a, full_matrices=False)
        # 断言左奇异向量接近空数组形状为 (m, 0)
        assert_allclose(u, np.empty((m, 0)))
        # 断言奇异值接近空数组
        assert_allclose(s, np.empty((0,)))
        # 断言右奇异向量接近空数组形状为 (0, n)
        assert_allclose(v, np.empty((0, n)))

        # 断言左奇异向量的数据类型与 a0 中的相同
        assert u.dtype == u0.dtype
        # 断言右奇异向量的数据类型与 a0 中的相同
        assert v.dtype == v0.dtype
        # 断言奇异值的数据类型与 a0 中的相同
        assert s.dtype == s0.dtype

        # 对空矩阵进行奇异值分解，仅计算奇异值
        s = svd(a, compute_uv=False)
        # 断言奇异值接近空数组
        assert_allclose(s, np.empty((0,)))

        # 断言奇异值的数据类型与 a0 中的相同
        assert s.dtype == s0.dtype
# 继承自 TestSVD_GESDD 类的 TestSVD_GESVD 类
class TestSVD_GESVD(TestSVD_GESDD):
    # 使用 gesvd LAPACK 驱动程序
    lapack_driver = 'gesvd'


# 使用 pytest.mark.fail_slow(10) 装饰器标记的测试函数
@pytest.mark.fail_slow(10)
def test_svd_gesdd_nofegfault():
    # 当 {U,VT}.size > INT_MAX 时，svd(a) 不会导致段错误
    # 参考 https://github.com/scipy/scipy/issues/14001
    df = np.ones((4799, 53130), dtype=np.float64)
    # 断言捕获 ValueError 异常
    with assert_raises(ValueError):
        svd(df)


# TestSVDVals 类定义
class TestSVDVals:

    # 使用 pytest.mark.parametrize 装饰器参数化的测试函数
    @pytest.mark.parametrize('dt', [int, float, np.float32, complex, np.complex64])
    def test_empty(self, dt):
        # 对于各种空数组的情况进行测试
        for a in [[]], np.empty((2, 0)), np.ones((0, 3)):
            a = np.array(a, dtype=dt)
            # 调用 svdvals 函数，期望得到一个空数组
            s = svdvals(a)
            assert_equal(s, np.empty(0))

            # 对单位矩阵调用 svdvals 函数
            s0 = svdvals(np.eye(2, dtype=dt))
            # 断言 s 和 s0 的数据类型一致
            assert s.dtype == s0.dtype

    # 测试简单情况的 svdvals 函数调用
    def test_simple(self):
        a = [[1, 2, 3], [1, 2, 3], [2, 5, 6]]
        s = svdvals(a)
        # 断言返回的奇异值数量为 3
        assert_(len(s) == 3)
        # 断言奇异值降序排列
        assert_(s[0] >= s[1] >= s[2])

    # 测试欠定情况的 svdvals 函数调用
    def test_simple_underdet(self):
        a = [[1, 2, 3], [4, 5, 6]]
        s = svdvals(a)
        # 断言返回的奇异值数量为 2
        assert_(len(s) == 2)
        # 断言奇异值降序排列
        assert_(s[0] >= s[1])

    # 测试超定情况的 svdvals 函数调用
    def test_simple_overdet(self):
        a = [[1, 2], [4, 5], [3, 4]]
        s = svdvals(a)
        # 断言返回的奇异值数量为 2
        assert_(len(s) == 2)
        # 断言奇异值降序排列
        assert_(s[0] >= s[1])

    # 测试复数情况的 svdvals 函数调用
    def test_simple_complex(self):
        a = [[1, 2, 3], [1, 20, 3j], [2, 5, 6]]
        s = svdvals(a)
        # 断言返回的奇异值数量为 3
        assert_(len(s) == 3)
        # 断言奇异值降序排列
        assert_(s[0] >= s[1] >= s[2])

    # 测试复数情况下的欠定 svdvals 函数调用
    def test_simple_underdet_complex(self):
        a = [[1, 2, 3], [4, 5j, 6]]
        s = svdvals(a)
        # 断言返回的奇异值数量为 2
        assert_(len(s) == 2)
        # 断言奇异值降序排列
        assert_(s[0] >= s[1])

    # 测试复数情况下的超定 svdvals 函数调用
    def test_simple_overdet_complex(self):
        a = [[1, 2], [4, 5], [3j, 4]]
        s = svdvals(a)
        # 断言返回的奇异值数量为 2
        assert_(len(s) == 2)
        # 断言奇异值降序排列
        assert_(s[0] >= s[1])

    # 测试禁用有限性检查的 svdvals 函数调用
    def test_check_finite(self):
        a = [[1, 2, 3], [1, 2, 3], [2, 5, 6]]
        s = svdvals(a, check_finite=False)
        # 断言返回的奇异值数量为 3
        assert_(len(s) == 3)
        # 断言奇异值降序排列
        assert_(s[0] >= s[1] >= s[2])

    # 使用 pytest.mark.slow 标记的慢速测试
    @pytest.mark.slow
    def test_crash_2609(self):
        np.random.seed(1234)
        a = np.random.rand(1500, 2800)
        # 不应该崩溃的情况
        svdvals(a)


# TestDiagSVD 类定义
class TestDiagSVD:

    # 测试简单情况的 diagsvd 函数调用
    def test_simple(self):
        assert_array_almost_equal(diagsvd([1, 0, 0], 3, 3),
                                  [[1, 0, 0], [0, 0, 0], [0, 0, 0]])


# TestQR 类定义
class TestQR:

    # 测试简单情况的 qr 函数调用
    def test_simple(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        q, r = qr(a)
        # 断言 q 的转置与 q 的乘积为单位矩阵
        assert_array_almost_equal(q.T @ q, eye(3))
        # 断言 q 和 r 的乘积等于原始矩阵 a
        assert_array_almost_equal(q @ r, a)

    # 测试简单情况下的 qr_multiply 函数调用（左乘）
    def test_simple_left(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        q, r = qr(a)
        c = [1, 2, 3]
        qc, r2 = qr_multiply(a, c, "left")
        # 断言 q 乘以 c 等于 qc
        assert_array_almost_equal(q @ c, qc)
        # 断言 r 等于 r2
        assert_array_almost_equal(r, r2)
        qc, r2 = qr_multiply(a, eye(3), "left")
        # 断言 q 等于 qc
        assert_array_almost_equal(q, qc)
    # 定义一个测试函数，测试非奇异矩阵的 QR 分解和乘法运算
    def test_simple_right(self):
        # 创建一个 3x3 的矩阵 a
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        # 进行 QR 分解，返回正交矩阵 q 和 上三角矩阵 r
        q, r = qr(a)
        # 创建一个长度为 3 的向量 c
        c = [1, 2, 3]
        # 使用 QR 分解的结果 q 和 r，将矩阵 a 和向量 c 进行乘法运算
        qc, r2 = qr_multiply(a, c)
        # 断言 qc 与 c @ q 几乎相等
        assert_array_almost_equal(c @ q, qc)
        # 断言 r 与 r2 几乎相等
        assert_array_almost_equal(r, r2)
        # 使用单位矩阵进行 QR 乘法运算，返回 qc 和 r
        qc, r = qr_multiply(a, eye(3))
        # 断言 q 与 qc 几乎相等
        assert_array_almost_equal(q, qc)

    # 定义一个测试函数，测试带主元素置换的 QR 分解
    def test_simple_pivoting(self):
        # 创建一个 3x3 的 numpy 数组 a
        a = np.asarray([[8, 2, 3], [2, 9, 3], [5, 3, 6]])
        # 进行带主元素置换的 QR 分解，返回正交矩阵 q、上三角矩阵 r 和 置换向量 p
        q, r, p = qr(a, pivoting=True)
        # 计算 r 的对角线元素的绝对值
        d = abs(diag(r))
        # 断言 r 的对角线元素单调非增
        assert_(np.all(d[1:] <= d[:-1]))
        # 断言 q 的转置与自身的乘积几乎等于单位矩阵
        assert_array_almost_equal(q.T @ q, eye(3))
        # 断言 q 与 r 的乘积几乎等于矩阵 a 的列按照 p 的顺序排列
        assert_array_almost_equal(q @ r, a[:, p])
        # 对按照 p 排列的 a 进行标准 QR 分解，返回 q2 和 r2
        q2, r2 = qr(a[:, p])
        # 断言 q 与 q2 几乎相等
        assert_array_almost_equal(q, q2)
        # 断言 r 与 r2 几乎相等
        assert_array_almost_equal(r, r2)

    # 定义一个测试函数，测试左乘 QR 分解和带主元素置换
    def test_simple_left_pivoting(self):
        # 创建一个 3x3 的矩阵 a
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        # 进行带主元素置换的 QR 分解，返回正交矩阵 q、上三角矩阵 r 和 置换向量 jpvt
        q, r, jpvt = qr(a, pivoting=True)
        # 创建一个长度为 3 的向量 c
        c = [1, 2, 3]
        # 使用左乘 QR 分解的结果 q 和 jpvt，将矩阵 a 和向量 c 进行乘法运算
        qc, r, jpvt = qr_multiply(a, c, "left", True)
        # 断言 q @ c 与 qc 几乎相等
        assert_array_almost_equal(q @ c, qc)

    # 定义一个测试函数，测试右乘 QR 分解和带主元素置换
    def test_simple_right_pivoting(self):
        # 创建一个 3x3 的矩阵 a
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        # 进行带主元素置换的 QR 分解，返回正交矩阵 q、上三角矩阵 r 和 置换向量 jpvt
        q, r, jpvt = qr(a, pivoting=True)
        # 创建一个长度为 3 的向量 c
        c = [1, 2, 3]
        # 使用右乘 QR 分解的结果 q 和 jpvt，将矩阵 a 和向量 c 进行乘法运算
        qc, r, jpvt = qr_multiply(a, c, pivoting=True)
        # 断言 c @ q 与 qc 几乎相等
        assert_array_almost_equal(c @ q, qc)

    # 定义一个测试函数，测试奇异矩阵的 QR 分解
    def test_simple_trap(self):
        # 创建一个 2x3 的矩阵 a
        a = [[8, 2, 3], [2, 9, 3]]
        # 进行 QR 分解，返回正交矩阵 q 和 上三角矩阵 r
        q, r = qr(a)
        # 断言 q 的转置与自身的乘积几乎等于单位矩阵
        assert_array_almost_equal(q.T @ q, eye(2))
        # 断言 q 与 r 的乘积几乎等于矩阵 a
        assert_array_almost_equal(q @ r, a)

    # 定义一个测试函数，测试奇异矩阵的带主元素置换的 QR 分解
    def test_simple_trap_pivoting(self):
        # 创建一个 2x3 的 numpy 数组 a
        a = np.asarray([[8, 2, 3], [2, 9, 3]])
        # 进行带主元素置换的 QR 分解，返回正交矩阵 q、上三角矩阵 r 和 置换向量 p
        q, r, p = qr(a, pivoting=True)
        # 计算 r 的对角线元素的绝对值
        d = abs(diag(r))
        # 断言 r 的对角线元素单调非增
        assert_(np.all(d[1:] <= d[:-1]))
        # 断言 q 的转置与自身的乘积几乎等于单位矩阵
        assert_array_almost_equal(q.T @ q, eye(2))
        # 断言 q 与 r 的乘积几乎等于矩阵 a 的列按照 p 的顺序排列
        assert_array_almost_equal(q @ r, a[:, p])
        # 对按照 p 排列的 a 进行标准 QR 分解，返回 q2 和 r2
        q2, r2 = qr(a[:, p])
        # 断言 q 与 q2 几乎相等
        assert_array_almost_equal(q, q2)
        # 断言 r 与 r2 几乎相等
        assert_array_almost_equal(r, r2)

    # 定义一个测试函数，测试长方形矩阵的 QR 分解
    def test_simple_tall(self):
        # 创建一个 3x2 的矩阵 a
        a = [[8, 2], [2, 9], [5, 3]]
        # 进行 QR 分解，返回正交矩阵 q 和 上三角矩阵 r
        q, r = qr(a)
        # 断言 q 的转置与自身的乘积几乎等于单位矩阵
        assert_array_almost_equal(q.T @ q, eye(3))
        # 断言 q 与 r 的乘积几乎等于矩阵 a
        assert_array_almost_equal(q @ r, a)

    # 定义一个测试函数，测试长方形矩阵的带主元素置换的 QR 分解
    def test_simple_tall_pivoting(self):
        # 创建一个 3x2 的 numpy 数组 a
        a = np.asarray([[8, 2], [2, 9], [5, 3]])
        # 进行带主元素置换的 QR 分解，
    def test_simple_tall_e_pivoting(self):
        # 测试经济模式下的 QR 分解，使用偏置
        a = np.asarray([[8, 2], [2, 9], [5, 3]])
        # 执行 QR 分解，返回正交矩阵 q，上三角矩阵 r，以及列置换向量 p
        q, r, p = qr(a, pivoting=True, mode='economic')
        # 获取 r 的对角线元素的绝对值
        d = abs(diag(r))
        # 断言：所有对角线元素都满足非增序列
        assert_(np.all(d[1:] <= d[:-1]))
        # 断言：q 的转置乘以 q 等于单位矩阵
        assert_array_almost_equal(q.T @ q, eye(2))
        # 断言：q 乘以 r 等于原始矩阵 a 按列置换后的结果
        assert_array_almost_equal(q @ r, a[:, p])
        # 执行二次 QR 分解，验证结果与首次分解相同
        q2, r2 = qr(a[:, p], mode='economic')
        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(r, r2)

    def test_simple_tall_left(self):
        # 测试经济模式下的 QR 分解，左乘特定向量或矩阵
        a = [[8, 2], [2, 9], [5, 3]]
        # 执行 QR 分解，返回正交矩阵 q 和上三角矩阵 r
        q, r = qr(a, mode="economic")
        c = [1, 2]
        # 左乘向量 c 到 QR 分解结果，返回 qc 和 r2
        qc, r2 = qr_multiply(a, c, "left")
        assert_array_almost_equal(q @ c, qc)
        assert_array_almost_equal(r, r2)
        c = array([1, 2, 0])
        # 左乘矩阵 c 到 QR 分解结果，覆盖输入矩阵 c，返回 qc 和 r2
        qc, r2 = qr_multiply(a, c, "left", overwrite_c=True)
        assert_array_almost_equal(q @ c[:2], qc)
        qc, r = qr_multiply(a, eye(2), "left")
        assert_array_almost_equal(qc, q)

    def test_simple_tall_left_pivoting(self):
        # 测试经济模式下的 QR 分解，左乘特定向量或矩阵，并进行列置换
        a = [[8, 2], [2, 9], [5, 3]]
        # 执行 QR 分解，返回正交矩阵 q，上三角矩阵 r，和列置换向量 jpvt
        q, r, jpvt = qr(a, mode="economic", pivoting=True)
        c = [1, 2]
        # 左乘向量 c 到 QR 分解结果，返回 qc 和 r，以及列置换向量 kpvt
        qc, r, kpvt = qr_multiply(a, c, "left", True)
        assert_array_equal(jpvt, kpvt)
        assert_array_almost_equal(q @ c, qc)
        qc, r, jpvt = qr_multiply(a, eye(2), "left", True)
        assert_array_almost_equal(qc, q)

    def test_simple_tall_right(self):
        # 测试经济模式下的 QR 分解，右乘特定向量或矩阵
        a = [[8, 2], [2, 9], [5, 3]]
        # 执行 QR 分解，返回正交矩阵 q 和上三角矩阵 r
        q, r = qr(a, mode="economic")
        c = [1, 2, 3]
        # 右乘向量 c 到 QR 分解结果，返回 cq 和 r2
        cq, r2 = qr_multiply(a, c)
        assert_array_almost_equal(c @ q, cq)
        assert_array_almost_equal(r, r2)
        cq, r = qr_multiply(a, eye(3))
        assert_array_almost_equal(cq, q)

    def test_simple_tall_right_pivoting(self):
        # 测试经济模式下的 QR 分解，右乘特定向量或矩阵，并进行列置换
        a = [[8, 2], [2, 9], [5, 3]]
        # 执行 QR 分解，返回正交矩阵 q，上三角矩阵 r，和列置换向量 jpvt
        q, r, jpvt = qr(a, pivoting=True, mode="economic")
        c = [1, 2, 3]
        # 右乘向量 c 到 QR 分解结果，返回 cq 和 r，以及列置换向量 jpvt
        cq, r, jpvt = qr_multiply(a, c, pivoting=True)
        assert_array_almost_equal(c @ q, cq)
        cq, r, jpvt = qr_multiply(a, eye(3), pivoting=True)
        assert_array_almost_equal(cq, q)

    def test_simple_fat(self):
        # 测试完整版本的 QR 分解
        a = [[8, 2, 5], [2, 9, 3]]
        # 执行 QR 分解，返回正交矩阵 q 和上三角矩阵 r
        q, r = qr(a)
        assert_array_almost_equal(q.T @ q, eye(2))
        assert_array_almost_equal(q @ r, a)
        assert_equal(q.shape, (2, 2))
        assert_equal(r.shape, (2, 3))

    def test_simple_fat_pivoting(self):
        # 测试完整版本的 QR 分解，使用列置换
        a = np.asarray([[8, 2, 5], [2, 9, 3]])
        # 执行 QR 分解，返回正交矩阵 q，上三角矩阵 r，和列置换向量 p
        q, r, p = qr(a, pivoting=True)
        d = abs(diag(r))
        assert_(np.all(d[1:] <= d[:-1]))
        assert_array_almost_equal(q.T @ q, eye(2))
        assert_array_almost_equal(q @ r, a[:, p])
        assert_equal(q.shape, (2, 2))
        assert_equal(r.shape, (2, 3))
        q2, r2 = qr(a[:, p])
        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(r, r2)
    # 定义测试函数，测试 QR 分解对于“瘦”矩阵（行数大于列数）的经济版本
    def test_simple_fat_e(self):
        # 创建一个包含数值的二维列表作为输入矩阵
        a = [[8, 2, 3], [2, 9, 5]]
        # 进行 QR 分解，选择经济模式（只返回最小必需的输出）
        q, r = qr(a, mode='economic')
        # 断言：计算 Q 的转置乘以 Q 应该等于单位矩阵
        assert_array_almost_equal(q.T @ q, eye(2))
        # 断言：Q 乘以 R 应该等于原始输入矩阵 A
        assert_array_almost_equal(q @ r, a)
        # 断言：Q 的形状应该是 (2, 2)
        assert_equal(q.shape, (2, 2))
        # 断言：R 的形状应该是 (2, 3)
        assert_equal(r.shape, (2, 3))

    # 定义测试函数，测试带主元素选取的 QR 分解对于“瘦”矩阵的经济版本
    def test_simple_fat_e_pivoting(self):
        # 创建一个 NumPy 数组作为输入矩阵
        a = np.asarray([[8, 2, 3], [2, 9, 5]])
        # 进行 QR 分解，选择经济模式和启用主元素选取
        q, r, p = qr(a, pivoting=True, mode='economic')
        # 计算 R 的对角线元素，并检查它们是否按非升序排列
        d = abs(diag(r))
        assert_(np.all(d[1:] <= d[:-1]))
        # 断言：计算 Q 的转置乘以 Q 应该等于单位矩阵
        assert_array_almost_equal(q.T @ q, eye(2))
        # 断言：Q 乘以 R 应该等于原始输入矩阵 A 的列按照主元素选取的顺序
        assert_array_almost_equal(q @ r, a[:, p])
        # 断言：Q 的形状应该是 (2, 2)
        assert_equal(q.shape, (2, 2))
        # 断言：R 的形状应该是 (2, 3)
        assert_equal(r.shape, (2, 3))
        # 对 A 按照主元素选取重新进行 QR 分解，检查结果是否与原始的 Q 和 R 相等
        q2, r2 = qr(a[:, p], mode='economic')
        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(r, r2)

    # 定义测试函数，测试 QR 乘法对于“瘦”矩阵的左乘操作
    def test_simple_fat_left(self):
        # 创建一个包含数值的二维列表作为输入矩阵
        a = [[8, 2, 3], [2, 9, 5]]
        # 进行 QR 分解，选择经济模式
        q, r = qr(a, mode="economic")
        # 创建一个向量作为左乘操作的矩阵
        c = [1, 2]
        # 执行 QR 左乘操作，返回乘积结果 qc 和 R2
        qc, r2 = qr_multiply(a, c, "left")
        # 断言：Q 乘以 c 应该等于 qc
        assert_array_almost_equal(q @ c, qc)
        # 断言：R 应该等于 R2
        assert_array_almost_equal(r, r2)
        # 对单位矩阵按照左乘操作进行 QR 分解，检查结果是否与原始的 Q 相等
        qc, r = qr_multiply(a, eye(2), "left")
        assert_array_almost_equal(qc, q)

    # 定义测试函数，测试带主元素选取的 QR 乘法对于“瘦”矩阵的左乘操作
    def test_simple_fat_left_pivoting(self):
        # 创建一个包含数值的二维列表作为输入矩阵
        a = [[8, 2, 3], [2, 9, 5]]
        # 进行 QR 分解，选择经济模式和启用主元素选取
        q, r, jpvt = qr(a, mode="economic", pivoting=True)
        # 创建一个向量作为左乘操作的矩阵
        c = [1, 2]
        # 执行带主元素选取的 QR 左乘操作，返回乘积结果 qc、R 和 jpvt
        qc, r, jpvt = qr_multiply(a, c, "left", True)
        # 断言：Q 乘以 c 应该等于 qc
        assert_array_almost_equal(q @ c, qc)
        # 对单位矩阵按照左乘操作进行带主元素选取的 QR 分解，检查结果是否与原始的 Q 和 jpvt 相等
        qc, r, jpvt = qr_multiply(a, eye(2), "left", True)
        assert_array_almost_equal(qc, q)

    # 定义测试函数，测试 QR 乘法对于“瘦”矩阵的右乘操作
    def test_simple_fat_right(self):
        # 创建一个包含数值的二维列表作为输入矩阵
        a = [[8, 2, 3], [2, 9, 5]]
        # 进行 QR 分解，选择经济模式
        q, r = qr(a, mode="economic")
        # 创建一个向量作为右乘操作的矩阵
        c = [1, 2]
        # 执行 QR 右乘操作，返回乘积结果 cq 和 R2
        cq, r2 = qr_multiply(a, c)
        # 断言：c 乘以 Q 应该等于 cq
        assert_array_almost_equal(c @ q, cq)
        # 断言：R 应该等于 R2
        assert_array_almost_equal(r, r2)
        # 对单位矩阵按照右乘操作进行 QR 分解，检查结果是否与原始的 Q 相等
        cq, r = qr_multiply(a, eye(2))
        assert_array_almost_equal(cq, q)

    # 定义测试函数，测试带主元素选取的 QR 乘法对于“瘦”矩阵的右乘操作
    def test_simple_fat_right_pivoting(self):
        # 创建一个包含数值的二维列表作为输入矩阵
        a = [[8, 2, 3], [2, 9, 5]]
        # 进行 QR 分解，选择经济模式和启用主元素选取
        q, r, jpvt = qr(a, pivoting=True, mode="economic")
        # 创建一个向量作为右乘操作的矩阵
        c = [1, 2]
        # 执行带主元素选取的 QR 右乘操作，返回乘积结果 cq、R 和 jpvt
        cq, r, jpvt = qr_multiply(a, c, pivoting=True)
        # 断言：c 乘以 Q 应该等于 cq
        assert_array_almost_equal(c @ q, cq)
        # 对单位矩阵按照右乘操作进行带主元素选取的 QR 分解，检查结果是否与原始的 Q 和 jpvt 相等
        cq, r, jpvt = qr_multiply(a, eye(2), pivoting=True)
        assert_array_almost_equal(cq, q)

    # 定义测试函数，测试 QR 分解对于复数矩阵的操作
    def test_simple_complex(self):
        # 创建一个包含复数的二维列表作为输入矩阵
        a = [[3, 3+4j, 5], [5, 2, 2+7j], [3, 2, 7]]
        # 进行 QR 分解
        q, r = qr(a)
        # 断言：计算 Q 的共轭转置乘以 Q 应该等于单位矩阵
        assert_array
    # 测试简单复杂右乘
    def test_simple_complex_right(self):
        # 创建一个包含复数的二维数组
        a = [[3, 3+4j, 5], [5, 2, 2+7j], [3, 2, 7]]
        # 计算矩阵的 QR 分解
        q, r = qr(a)
        # 创建一个复数数组
        c = [1, 2, 3+4j]
        # 使用 QR 分解结果进行矩阵乘法
        qc, r = qr_multiply(a, c)
        # 断言两个数组几乎相等
        assert_array_almost_equal(c @ q, qc)
        # 使用单位矩阵进行矩阵乘法
        qc, r = qr_multiply(a, eye(3))
        # 断言两个数组几乎相等
        assert_array_almost_equal(q, qc)

    # 测试简单高瘦复杂左乘
    def test_simple_tall_complex_left(self):
        # 创建一个包含复数的二维数组
        a = [[8, 2+3j], [2, 9], [5+7j, 3]]
        # 计算经济型 QR 分解
        q, r = qr(a, mode="economic")
        # 创建一个复数数组
        c = [1, 2+2j]
        # 使用 QR 分解结果进行矩阵乘法，并指定左乘
        qc, r2 = qr_multiply(a, c, "left")
        # 断言两个数组几乎相等
        assert_array_almost_equal(q @ c, qc)
        # 断言两个数组几乎相等
        assert_array_almost_equal(r, r2)
        # 创建一个数组
        c = array([1, 2, 0])
        # 使用 QR 分解结果进行矩阵乘法，并指定左乘和覆盖 C
        qc, r2 = qr_multiply(a, c, "left", overwrite_c=True)
        # 断言两个数组几乎相等
        assert_array_almost_equal(q @ c[:2], qc)
        # 使用单位矩阵进行矩阵乘法，并指定左乘
        qc, r = qr_multiply(a, eye(2), "left")
        # 断言两个数组几乎相等
        assert_array_almost_equal(qc, q)

    # 测试简单复杂左乘共轭
    def test_simple_complex_left_conjugate(self):
        # 创建一个包含复数的二维数组
        a = [[3, 3+4j, 5], [5, 2, 2+7j], [3, 2, 7]]
        # 计算矩阵的 QR 分解
        q, r = qr(a)
        # 创建一个复数数组
        c = [1, 2, 3+4j]
        # 使用 QR 分解结果进行矩阵乘法，并指定左乘和共轭
        qc, r = qr_multiply(a, c, "left", conjugate=True)
        # 断言两个数组几乎相等
        assert_array_almost_equal(q.conj() @ c, qc)

    # 测试简单复杂高瘦左乘共轭
    def test_simple_complex_tall_left_conjugate(self):
        # 创建一个包含复数的二维数组
        a = [[3, 3+4j], [5, 2+2j], [3, 2]]
        # 计算经济型 QR 分解
        q, r = qr(a, mode='economic')
        # 创建一个复数数组
        c = [1, 3+4j]
        # 使用 QR 分解结果进行矩阵乘法，并指定左乘和共轭
        qc, r = qr_multiply(a, c, "left", conjugate=True)
        # 断言两个数组几乎相等
        assert_array_almost_equal(q.conj() @ c, qc)

    # 测试简单复杂右乘共轭
    def test_simple_complex_right_conjugate(self):
        # 创建一个包含复数的二维数组
        a = [[3, 3+4j, 5], [5, 2, 2+7j], [3, 2, 7]]
        # 计算矩阵的 QR 分解
        q, r = qr(a)
        # 创建一个复数数组
        c = np.array([1, 2, 3+4j])
        # 使用 QR 分解结果进行矩阵乘法，并指定右乘共轭
        qc, r = qr_multiply(a, c, conjugate=True)
        # 断言两个数组几乎相等
        assert_array_almost_equal(c @ q.conj(), qc)

    # 测试简单复杂带枢轴
    def test_simple_complex_pivoting(self):
        # 创建一个包含复数的二维数组
        a = array([[3, 3+4j, 5], [5, 2, 2+7j], [3, 2, 7]])
        # 计算带枢轴的 QR 分解
        q, r, p = qr(a, pivoting=True)
        # 计算对角线元素的绝对值
        d = abs(diag(r))
        # 断言对角线元素非增
        assert_(np.all(d[1:] <= d[:-1]))
        # 断言两个数组几乎相等
        assert_array_almost_equal(q.conj().T @ q, eye(3))
        # 断言两个数组几乎相等
        assert_array_almost_equal(q @ r, a[:, p])
        # 再次计算无枢轴的 QR 分解
        q2, r2 = qr(a[:, p])
        # 断言两个数组几乎相等
        assert_array_almost_equal(q, q2)
        # 断言两个数组几乎相等
        assert_array_almost_equal(r, r2)

    # 测试简单复杂带枢轴左乘
    def test_simple_complex_left_pivoting(self):
        # 创建一个包含复数的二维数组
        a = array([[3, 3+4j, 5], [5, 2, 2+7j], [3, 2, 7]])
        # 计算带枢轴的 QR 分解
        q, r, jpvt = qr(a, pivoting=True)
        # 创建一个复数数组
        c = [1, 2, 3+4j]
        # 使用带枢轴的 QR 分解结果进行矩阵乘法，并指定左乘
        qc, r, jpvt = qr_multiply(a, c, "left", True)
        # 断言两个数组几乎相等
        assert_array_almost_equal(q @ c, qc)

    # 测试简单复杂带枢轴右乘
    def test_simple_complex_right_pivoting(self):
        # 创建一个包含复数的二维数组
        a = array([[3, 3+4j, 5], [5, 2, 2+7j], [3, 2, 7]])
        # 计算带枢轴的 QR 分解
        q, r, jpvt = qr(a, pivoting=True)
        # 创建一个复数数组
        c = [1, 2, 3+4j]
        # 使用带枢轴的 QR 分解结果进行矩阵乘法，并指定右乘带枢轴
        qc, r, jpvt = qr_multiply(a, c, pivoting=True)
        # 断言两个数组几乎相等
        assert_array_almost_equal(c @ q, qc)

    # 测试随机生成的矩阵
    def test_random(self):
        # 创建一个随机数生成器
        rng = np.random.RandomState(1234)
        # 矩阵的维数
        n = 20
        # 进行多次测试
        for k in range(2):
            # 生成一个随机矩阵
            a = rng.random([n, n])
            # 计算矩阵的 QR 分解
            q, r = qr(a)
            # 断言两个数组几乎相等
            assert_array_almost_equal(q.T @
    # 定义一个测试函数，用于测试随机生成的左乘QR分解的功能
    def test_random_left(self):
        # 创建一个随机数生成器，并指定种子以保持结果的可重复性
        rng = np.random.RandomState(1234)
        # 矩阵维度设定为 n × n
        n = 20
        # 重复执行两次测试
        for k in range(2):
            # 生成一个随机的 n × n 矩阵 a
            a = rng.random([n, n])
            # 对矩阵 a 进行 QR 分解
            q, r = qr(a)
            # 生成一个随机的 n 维向量 c
            c = rng.random([n])
            # 使用左乘的方式计算 QR 分解后的乘积 qc
            qc, r = qr_multiply(a, c, "left")
            # 断言乘积结果 qc 等于 q @ c
            assert_array_almost_equal(q @ c, qc)
            # 再次使用左乘的方式计算 QR 分解后的乘积，这次乘以单位矩阵 eye(n)
            qc, r = qr_multiply(a, eye(n), "left")
            # 断言乘积结果 qc 等于 q
            assert_array_almost_equal(q, qc)

    # 定义一个测试函数，用于测试随机生成的右乘QR分解的功能
    def test_random_right(self):
        # 创建一个随机数生成器，并指定种子以保持结果的可重复性
        rng = np.random.RandomState(1234)
        # 矩阵维度设定为 n × n
        n = 20
        # 重复执行两次测试
        for k in range(2):
            # 生成一个随机的 n × n 矩阵 a
            a = rng.random([n, n])
            # 对矩阵 a 进行 QR 分解
            q, r = qr(a)
            # 生成一个随机的 n 维向量 c
            c = rng.random([n])
            # 使用右乘的方式计算 QR 分解后的乘积 cq
            cq, r = qr_multiply(a, c)
            # 断言乘积结果 c @ q 等于 cq
            assert_array_almost_equal(c @ q, cq)
            # 再次使用右乘的方式计算 QR 分解后的乘积，这次乘以单位矩阵 eye(n)
            cq, r = qr_multiply(a, eye(n))
            # 断言乘积结果 cq 等于 q
            assert_array_almost_equal(q, cq)

    # 定义一个测试函数，用于测试随机生成的带有列主元素的QR分解的功能
    def test_random_pivoting(self):
        # 创建一个随机数生成器，并指定种子以保持结果的可重复性
        rng = np.random.RandomState(1234)
        # 矩阵维度设定为 n × n
        n = 20
        # 重复执行两次测试
        for k in range(2):
            # 生成一个随机的 n × n 矩阵 a
            a = rng.random([n, n])
            # 对矩阵 a 进行带有列主元素的 QR 分解
            q, r, p = qr(a, pivoting=True)
            # 计算 QR 分解后上三角矩阵的对角线元素的绝对值
            d = abs(np.diag(r))
            # 断言上三角矩阵对角线元素递减
            assert_(np.all(d[1:] <= d[:-1]))
            # 断言 q 的转置与 q 的矩阵乘积等于单位矩阵
            assert_array_almost_equal(q.T @ q, np.eye(n))
            # 断言 q 与 r 的矩阵乘积等于原始矩阵 a 的列重排结果
            assert_array_almost_equal(q @ r, a[:, p])
            # 对经列重排的矩阵 a[:, p] 再次进行 QR 分解
            q2, r2 = qr(a[:, p])
            # 断言两次 QR 分解得到的 q 和 r 矩阵近似相等
            assert_array_almost_equal(q, q2)
            assert_array_almost_equal(r, r2)

    # 定义一个测试函数，用于测试随机生成的高瘦矩阵的QR分解功能
    def test_random_tall(self):
        # 创建一个随机数生成器，并指定种子以保持结果的可重复性
        rng = np.random.RandomState(1234)
        # 设定矩阵的维度为 m × n，其中 m > n
        m = 200
        n = 100
        # 重复执行两次测试
        for k in range(2):
            # 生成一个随机的 m × n 矩阵 a
            a = rng.random([m, n])
            # 对矩阵 a 进行 QR 分解
            q, r = qr(a)
            # 断言 q 的转置与 q 的矩阵乘积等于单位矩阵
            assert_array_almost_equal(q.T @ q, np.eye(m))
            # 断言 q 与 r 的矩阵乘积等于原始矩阵 a
            assert_array_almost_equal(q @ r, a)

    # 定义一个测试函数，用于测试随机生成的高瘦矩阵的左乘QR分解功能
    def test_random_tall_left(self):
        # 创建一个随机数生成器，并指定种子以保持结果的可重复性
        rng = np.random.RandomState(1234)
        # 设定矩阵的维度为 m × n，其中 m > n
        m = 200
        n = 100
        # 重复执行两次测试
        for k in range(2):
            # 生成一个随机的 m × n 矩阵 a
            a = rng.random([m, n])
            # 对矩阵 a 进行经济型左乘 QR 分解
            q, r = qr(a, mode="economic")
            # 生成一个随机的 n 维向量 c
            c = rng.random([n])
            # 使用左乘的方式计算 QR 分解后的乘积 qc
            qc, r = qr_multiply(a, c, "left")
            # 断言乘积结果 q @ c 等于 qc
            assert_array_almost_equal(q @ c, qc)
            # 再次使用左乘的方式计算 QR 分解后的乘积，这次乘以单位矩阵 eye(n)
            qc, r = qr_multiply(a, eye(n), "left")
            # 断言乘积结果 qc 等于 q
            assert_array_almost_equal(qc, q)

    # 定义一个测试函数，用于测试随机生成的高瘦矩阵的右乘QR分解功能
    def test_random_tall_right(self):
        # 创建一个随机数生成器，并指定种子以保持结果的可重复性
        rng = np.random.RandomState(1234)
        # 设定矩阵的维度为 m × n，其中 m > n
        m = 200
        n = 100
        # 重复执行两次测试
        for k in range(2):
            # 生成一个随机的 m × n 矩阵 a
            a = rng.random([m, n])
            # 对矩阵 a 进行经济型右乘 QR 分解
            q, r = qr(a, mode="economic")
            # 生成一个随机的 m 维向量 c
            c = rng.random([m])
            # 使用右乘的方式计算 QR 分解后的乘积 cq
            cq, r = qr_multiply(a, c)
            # 断言乘积结果 c @ q 等于 cq
            assert_array_almost_equal(c @ q, cq)
            # 再次使用右乘的方式计算 QR 分解后的乘积，这次乘以单位矩阵 eye(m)
            cq, r =
    def test_random_tall_pivoting(self):
        rng = np.random.RandomState(1234)
        # 使用种子为1234的随机数生成器创建对象rng

        # 定义矩阵维度
        m = 200
        n = 100

        # 迭代两次
        for k in range(2):
            # 生成m行n列的随机矩阵a
            a = rng.random([m, n])
            
            # 对矩阵a进行QR分解，启用完整的列主元素选取
            q, r, p = qr(a, pivoting=True)
            
            # 计算QR分解后的对角线元素绝对值
            d = abs(diag(r))
            
            # 断言：QR分解后得到的对角线元素是非递增的
            assert_(np.all(d[1:] <= d[:-1]))
            
            # 断言：q的转置乘以q等于单位矩阵
            assert_array_almost_equal(q.T @ q, eye(m))
            
            # 断言：q乘以r等于原始矩阵a的列重新排列后的结果
            assert_array_almost_equal(q @ r, a[:, p])
            
            # 对重新排列后的a[:, p]再次进行QR分解
            q2, r2 = qr(a[:, p])
            
            # 断言：两次QR分解得到的结果应该相等
            assert_array_almost_equal(q, q2)
            assert_array_almost_equal(r, r2)

    def test_random_tall_e(self):
        rng = np.random.RandomState(1234)
        # 使用种子为1234的随机数生成器创建对象rng

        # 定义矩阵维度
        m = 200
        n = 100

        # 迭代两次
        for k in range(2):
            # 生成m行n列的随机矩阵a
            a = rng.random([m, n])
            
            # 对矩阵a进行经济QR分解
            q, r = qr(a, mode='economic')
            
            # 断言：q的转置乘以q等于单位矩阵
            assert_array_almost_equal(q.T @ q, eye(n))
            
            # 断言：q乘以r等于原始矩阵a
            assert_array_almost_equal(q @ r, a)
            
            # 断言：q的形状应该是(m, n)，r的形状应该是(n, n)
            assert_equal(q.shape, (m, n))
            assert_equal(r.shape, (n, n))

    def test_random_tall_e_pivoting(self):
        rng = np.random.RandomState(1234)
        # 使用种子为1234的随机数生成器创建对象rng

        # 定义矩阵维度
        m = 200
        n = 100

        # 迭代两次
        for k in range(2):
            # 生成m行n列的随机矩阵a
            a = rng.random([m, n])
            
            # 对矩阵a进行经济QR分解，并启用完整的列主元素选取
            q, r, p = qr(a, pivoting=True, mode='economic')
            
            # 计算QR分解后的对角线元素绝对值
            d = abs(diag(r))
            
            # 断言：QR分解后得到的对角线元素是非递增的
            assert_(np.all(d[1:] <= d[:-1]))
            
            # 断言：q的转置乘以q等于单位矩阵
            assert_array_almost_equal(q.T @ q, eye(n))
            
            # 断言：q乘以r等于原始矩阵a的列重新排列后的结果
            assert_array_almost_equal(q @ r, a[:, p])
            
            # 断言：q的形状应该是(m, n)，r的形状应该是(n, n)
            assert_equal(q.shape, (m, n))
            assert_equal(r.shape, (n, n))
            
            # 对重新排列后的a[:, p]再次进行经济QR分解
            q2, r2 = qr(a[:, p], mode='economic')
            
            # 断言：两次QR分解得到的结果应该相等
            assert_array_almost_equal(q, q2)
            assert_array_almost_equal(r, r2)

    def test_random_trap(self):
        rng = np.random.RandomState(1234)
        # 使用种子为1234的随机数生成器创建对象rng

        # 定义矩阵维度
        m = 100
        n = 200

        # 迭代两次
        for k in range(2):
            # 生成m行n列的随机矩阵a
            a = rng.random([m, n])
            
            # 对矩阵a进行QR分解
            q, r = qr(a)
            
            # 断言：q的转置乘以q等于单位矩阵
            assert_array_almost_equal(q.T @ q, eye(m))
            
            # 断言：q乘以r等于原始矩阵a
            assert_array_almost_equal(q @ r, a)

    def test_random_trap_pivoting(self):
        rng = np.random.RandomState(1234)
        # 使用种子为1234的随机数生成器创建对象rng

        # 定义矩阵维度
        m = 100
        n = 200

        # 迭代两次
        for k in range(2):
            # 生成m行n列的随机矩阵a
            a = rng.random([m, n])
            
            # 对矩阵a进行QR分解，启用完整的列主元素选取
            q, r, p = qr(a, pivoting=True)
            
            # 计算QR分解后的对角线元素绝对值
            d = abs(diag(r))
            
            # 断言：QR分解后得到的对角线元素是非递增的
            assert_(np.all(d[1:] <= d[:-1]))
            
            # 断言：q的转置乘以q等于单位矩阵
            assert_array_almost_equal(q.T @ q, eye(m))
            
            # 断言：q乘以r等于原始矩阵a的列重新排列后的结果
            assert_array_almost_equal(q @ r, a[:, p])
            
            # 对重新排列后的a[:, p]再次进行QR分解
            q2, r2 = qr(a[:, p])
            
            # 断言：两次QR分解得到的结果应该相等
            assert_array_almost_equal(q, q2)
            assert_array_almost_equal(r, r2)

    def test_random_complex(self):
        rng = np.random.RandomState(1234)
        # 使用种子为1234的随机数生成器创建对象rng

        # 定义矩阵维度
        n = 20

        # 迭代两次
        for k in range(2):
            # 生成n行n列的复数随机矩阵a
            a = rng.random([n, n]) + 1j * rng.random([n, n])
            
            # 对复数随机矩阵a进行QR分解
            q, r = qr(a)
            
            # 断言：q的共轭转置乘以q等于单位矩阵
            assert_array_almost_equal(q.conj().T @ q, eye(n))
            
            # 断言：q乘以r等于原始矩阵a
            assert_array_almost_equal(q @ r, a)
    def test_random_complex_left(self):
        # 使用种子 1234 初始化随机数生成器 rng
        rng = np.random.RandomState(1234)
        # 设置矩阵维度 n = 20
        n = 20
        # 执行两次循环
        for k in range(2):
            # 创建一个复数矩阵 a，实部和虚部都是从 rng 生成的随机数
            a = rng.random([n, n]) + 1j * rng.random([n, n])
            # 对矩阵 a 进行 QR 分解，得到正交矩阵 q 和上三角矩阵 r
            q, r = qr(a)
            # 创建一个复数向量 c，实部和虚部都是从 rng 生成的随机数
            c = rng.random([n]) + 1j * rng.random([n])
            # 使用 qr_multiply 函数左乘矩阵 a 和向量 c，得到结果 qc 和新的 r
            qc, r = qr_multiply(a, c, "left")
            # 检查 qc 是否与 q @ c 近似相等
            assert_array_almost_equal(q @ c, qc)
            # 再次使用 qr_multiply 函数左乘矩阵 a 和单位矩阵 eye(n)，得到结果 qc 和新的 r
            qc, r = qr_multiply(a, eye(n), "left")
            # 检查 qc 是否与 q 近似相等
            assert_array_almost_equal(q, qc)

    def test_random_complex_right(self):
        # 使用种子 1234 初始化随机数生成器 rng
        rng = np.random.RandomState(1234)
        # 设置矩阵维度 n = 20
        n = 20
        # 执行两次循环
        for k in range(2):
            # 创建一个复数矩阵 a，实部和虚部都是从 rng 生成的随机数
            a = rng.random([n, n]) + 1j * rng.random([n, n])
            # 对矩阵 a 进行 QR 分解，得到正交矩阵 q 和上三角矩阵 r
            q, r = qr(a)
            # 创建一个复数向量 c，实部和虚部都是从 rng 生成的随机数
            c = rng.random([n]) + 1j * rng.random([n])
            # 使用 qr_multiply 函数右乘矩阵 a 和向量 c，得到结果 cq 和新的 r
            cq, r = qr_multiply(a, c)
            # 检查 cq 是否与 c @ q 近似相等
            assert_array_almost_equal(c @ q, cq)
            # 再次使用 qr_multiply 函数右乘矩阵 a 和单位矩阵 eye(n)，得到结果 cq 和新的 r
            cq, r = qr_multiply(a, eye(n))
            # 检查 cq 是否与 q 近似相等
            assert_array_almost_equal(q, cq)

    def test_random_complex_pivoting(self):
        # 使用种子 1234 初始化随机数生成器 rng
        rng = np.random.RandomState(1234)
        # 设置矩阵维度 n = 20
        n = 20
        # 执行两次循环
        for k in range(2):
            # 创建一个复数矩阵 a，实部和虚部都是从 rng 生成的随机数
            a = rng.random([n, n]) + 1j * rng.random([n, n])
            # 对矩阵 a 进行 QR 分解，开启枢轴选取选项，得到正交矩阵 q、上三角矩阵 r 和枢轴数组 p
            q, r, p = qr(a, pivoting=True)
            # 计算 r 的对角线元素的绝对值，并检查是否递减
            d = abs(diag(r))
            assert_(np.all(d[1:] <= d[:-1]))
            # 检查 q 的共轭转置与自身的乘积是否近似等于单位矩阵
            assert_array_almost_equal(q.conj().T @ q, eye(n))
            # 检查 q @ r 是否近似等于矩阵 a 按照枢轴数组 p 的重新排列
            assert_array_almost_equal(q @ r, a[:, p])
            # 对重新排列后的子矩阵进行 QR 分解，得到正交矩阵 q2 和上三角矩阵 r2
            q2, r2 = qr(a[:, p])
            # 检查 q 和 q2 是否近似相等
            assert_array_almost_equal(q, q2)
            # 检查 r 和 r2 是否近似相等
            assert_array_almost_equal(r, r2)

    def test_check_finite(self):
        # 创建一个普通列表作为输入矩阵 a
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        # 对矩阵 a 进行 QR 分解，关闭有限性检查选项，得到正交矩阵 q 和上三角矩阵 r
        q, r = qr(a, check_finite=False)
        # 检查 q 的转置与自身的乘积是否近似等于单位矩阵
        assert_array_almost_equal(q.T @ q, eye(3))
        # 检查 q @ r 是否近似等于矩阵 a
        assert_array_almost_equal(q @ r, a)

    def test_lwork(self):
        # 创建一个普通列表作为输入矩阵 a
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        # 获取默认 lwork 下的 QR 分解结果 q 和 r
        q, r = qr(a, lwork=None)

        # 测试使用最小有效 lwork 值
        q2, r2 = qr(a, lwork=3)
        assert_array_almost_equal(q2, q)
        assert_array_almost_equal(r2, r)

        # 测试使用较大的 lwork 值
        q3, r3 = qr(a, lwork=10)
        assert_array_almost_equal(q3, q)
        assert_array_almost_equal(r3, r)

        # 测试使用显式的 lwork=-1
        q4, r4 = qr(a, lwork=-1)
        assert_array_almost_equal(q4, q)
        assert_array_almost_equal(r4, r)

        # 测试使用无效的 lwork 值
        assert_raises(Exception, qr, (a,), {'lwork': 0})
        assert_raises(Exception, qr, (a,), {'lwork': 2})

    @pytest.mark.parametrize("m", [0, 1, 2])
    @pytest.mark.parametrize("n", [0, 1, 2])
    @pytest.mark.parametrize("pivoting", [False, True])
    @pytest.mark.parametrize('dtype', DTYPES)
    # 定义一个测试函数，用于测试 QR 分解的形状和数据类型
    def test_shape_dtype(self, m, n, pivoting, dtype):
        # 取 m 和 n 的较小值作为 k
        k = min(m, n)

        # 创建一个 m 行 n 列的零矩阵 a，数据类型为指定的 dtype
        a = np.zeros((m, n), dtype=dtype)
        
        # 对矩阵 a 进行 QR 分解，返回 Q, R 和其他可能的输出
        q, r, *other = qr(a, pivoting=pivoting)
        
        # 断言 Q 的形状为 (m, m)，数据类型为指定的 dtype
        assert_equal(q.shape, (m, m))
        assert_equal(q.dtype, dtype)
        
        # 断言 R 的形状为 (m, n)，数据类型为指定的 dtype
        assert_equal(r.shape, (m, n))
        assert_equal(r.dtype, dtype)
        
        # 如果进行了列主元素的分解，则检查附加输出的形状和数据类型
        assert len(other) == (1 if pivoting else 0)
        if pivoting:
            p, = other
            assert_equal(p.shape, (n,))
            assert_equal(p.dtype, np.int32)

        # 重复以上过程，使用不同的 mode 参数进行测试

        # mode='r'，只返回 R
        r, *other = qr(a, mode='r', pivoting=pivoting)
        assert_equal(r.shape, (m, n))
        assert_equal(r.dtype, dtype)
        assert len(other) == (1 if pivoting else 0)
        if pivoting:
            p, = other
            assert_equal(p.shape, (n,))
            assert_equal(p.dtype, np.int32)

        # mode='economic'，返回经济 QR 分解的 Q 和 R
        q, r, *other = qr(a, mode='economic', pivoting=pivoting)
        assert_equal(q.shape, (m, k))
        assert_equal(q.dtype, dtype)
        assert_equal(r.shape, (k, n))
        assert_equal(r.dtype, dtype)
        assert len(other) == (1 if pivoting else 0)
        if pivoting:
            p, = other
            assert_equal(p.shape, (n,))
            assert_equal(p.dtype, np.int32)

        # mode='raw'，返回原始输出的 Q, R 和其他可能的输出
        (raw, tau), r, *other = qr(a, mode='raw', pivoting=pivoting)
        assert_equal(raw.shape, (m, n))
        assert_equal(raw.dtype, dtype)
        assert_equal(tau.shape, (k,))
        assert_equal(tau.dtype, dtype)
        assert_equal(r.shape, (k, n))
        assert_equal(r.dtype, dtype)
        assert len(other) == (1 if pivoting else 0)
        if pivoting:
            p, = other
            assert_equal(p.shape, (n,))
            assert_equal(p.dtype, np.int32)

    # 使用 pytest 的参数化装饰器，定义一个测试空矩阵的函数
    @pytest.mark.parametrize(("m", "n"), [(0, 0), (0, 2), (2, 0)])
    def test_empty(self, m, n):
        # 取 m 和 n 的较小值作为 k
        k = min(m, n)

        # 创建一个 m 行 n 列的空矩阵 a
        a = np.empty((m, n))
        
        # 对空矩阵 a 进行 QR 分解，返回 Q, R
        q, r = qr(a)
        
        # 断言 Q 为单位矩阵（对角线为 1，其余为 0）
        assert_allclose(q, np.identity(m))
        
        # 断言 R 的值与空矩阵的形状相同
        assert_allclose(r, np.empty((m, n)))

        # 对带主元素的 QR 分解进行类似的断言
        q, r, p = qr(a, pivoting=True)
        assert_allclose(q, np.identity(m))
        assert_allclose(r, np.empty((m, n)))
        assert_allclose(p, np.arange(n))

        # mode='r'，只返回 R
        r, = qr(a, mode='r')
        assert_allclose(r, np.empty((m, n)))

        # mode='economic'，返回经济 QR 分解的 Q 和 R
        q, r = qr(a, mode='economic')
        assert_allclose(q, np.empty((m, k)))
        assert_allclose(r, np.empty((k, n)))

        # mode='raw'，返回原始输出的 Q, R 和其他可能的输出
        (raw, tau), r = qr(a, mode='raw')
        assert_allclose(raw, np.empty((m, n)))
        assert_allclose(tau, np.empty((k,)))
        assert_allclose(r, np.empty((k, n)))

    # 定义一个测试函数，用于测试乘法运算中空矩阵的情况
    def test_multiply_empty(self):
        # 创建两个 0x0 空矩阵 a 和 c
        a = np.empty((0, 0))
        c = np.empty((0, 0))
        
        # 对空矩阵 a 和 c 进行 QR 分解后乘法运算
        cq, r = qr_multiply(a, c)
        
        # 断言乘法结果 cq 也为 0x0 空矩阵
        assert_allclose(cq, np.empty((0, 0)))

        # 创建两个空矩阵，其中一个行数为 0，列数为 2；另一个行数为 2，列数为 0
        a = np.empty((0, 2))
        c = np.empty((2, 0))
        
        # 对这两个矩阵进行 QR 分解后乘法运算
        cq, r = qr_multiply(a, c)
        
        # 断言乘法结果 cq 的形状为 (2, 0)
        assert_allclose(cq, np.empty((2, 0)))

        # 创建两个空矩阵，其中一个行数为 2，列数为 0；另一个行数为 0，列数为 2
        a = np.empty((2, 0))
        c = np.empty((0, 2))
        
        # 对这两个矩阵进行 QR 分解后乘法运算
        cq, r = qr_multiply(a, c)
        
        # 断言乘法结果 cq 的形状为 (0, 2)
        assert_allclose(cq, np.empty((0, 2)))
class TestRQ:
    # 测试简单情况下的RQ分解
    def test_simple(self):
        # 创建一个3x3的矩阵a
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        # 进行RQ分解，并返回结果r和q
        r, q = rq(a)
        # 断言q乘以其转置等于单位矩阵
        assert_array_almost_equal(q @ q.T, eye(3))
        # 断言r乘以q等于原始矩阵a
        assert_array_almost_equal(r @ q, a)

    # 测试获取R矩阵的情况
    def test_r(self):
        # 创建一个3x3的矩阵a
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        # 进行RQ分解，并返回结果r和q
        r, q = rq(a)
        # 再次进行RQ分解获取r2
        r2 = rq(a, mode='r')
        # 断言r和r2近似相等
        assert_array_almost_equal(r, r2)

    # 测试随机生成矩阵的情况
    def test_random(self):
        # 创建一个随机数生成器，种子为1234
        rng = np.random.RandomState(1234)
        # 设置矩阵维度为20
        n = 20
        # 进行两次循环
        for k in range(2):
            # 生成一个n x n的随机矩阵a
            a = rng.random([n, n])
            # 进行RQ分解，并返回结果r和q
            r, q = rq(a)
            # 断言q乘以其转置等于单位矩阵
            assert_array_almost_equal(q @ q.T, eye(n))
            # 断言r乘以q等于原始矩阵a
            assert_array_almost_equal(r @ q, a)

    # 测试矩阵较短的情况
    def test_simple_trap(self):
        # 创建一个2x3的矩阵a
        a = [[8, 2, 3], [2, 9, 3]]
        # 进行RQ分解，并返回结果r和q
        r, q = rq(a)
        # 断言q的转置乘以q等于单位矩阵
        assert_array_almost_equal(q.T @ q, eye(3))
        # 断言r乘以q等于原始矩阵a
        assert_array_almost_equal(r @ q, a)

    # 测试矩阵较高的情况
    def test_simple_tall(self):
        # 创建一个3x2的矩阵a
        a = [[8, 2], [2, 9], [5, 3]]
        # 进行RQ分解，并返回结果r和q
        r, q = rq(a)
        # 断言q的转置乘以q等于单位矩阵
        assert_array_almost_equal(q.T @ q, eye(2))
        # 断言r乘以q等于原始矩阵a
        assert_array_almost_equal(r @ q, a)

    # 测试矩阵较宽的情况
    def test_simple_fat(self):
        # 创建一个2x3的矩阵a
        a = [[8, 2, 5], [2, 9, 3]]
        # 进行RQ分解，并返回结果r和q
        r, q = rq(a)
        # 断言q乘以其转置等于单位矩阵
        assert_array_almost_equal(q @ q.T, eye(3))
        # 断言r乘以q等于原始矩阵a
        assert_array_almost_equal(r @ q, a)

    # 测试复杂数据类型的情况
    def test_simple_complex(self):
        # 创建一个包含复数的3x3矩阵a
        a = [[3, 3+4j, 5], [5, 2, 2+7j], [3, 2, 7]]
        # 进行RQ分解，并返回结果r和q
        r, q = rq(a)
        # 断言q乘以其共轭转置等于单位矩阵
        assert_array_almost_equal(q @ q.conj().T, eye(3))
        # 断言r乘以q等于原始矩阵a
        assert_array_almost_equal(r @ q, a)

    # 测试随机生成的高瘦矩阵的情况
    def test_random_tall(self):
        # 创建一个随机数生成器，种子为1234
        rng = np.random.RandomState(1234)
        # 设置矩阵维度为200x100
        m = 200
        n = 100
        # 进行两次循环
        for k in range(2):
            # 生成一个m x n的随机矩阵a
            a = rng.random([m, n])
            # 进行RQ分解，并返回结果r和q
            r, q = rq(a)
            # 断言q乘以其转置等于单位矩阵
            assert_array_almost_equal(q @ q.T, eye(n))
            # 断言r乘以q等于原始矩阵a
            assert_array_almost_equal(r @ q, a)

    # 测试随机生成的高胖矩阵的情况
    def test_random_trap(self):
        # 创建一个随机数生成器，种子为1234
        rng = np.random.RandomState(1234)
        # 设置矩阵维度为100x200
        m = 100
        n = 200
        # 进行两次循环
        for k in range(2):
            # 生成一个m x n的随机矩阵a
            a = rng.random([m, n])
            # 进行RQ分解，并返回结果r和q
            r, q = rq(a)
            # 断言q乘以其转置等于单位矩阵
            assert_array_almost_equal(q @ q.T, eye(n))
            # 断言r乘以q等于原始矩阵a
            assert_array_almost_equal(r @ q, a)

    # 测试随机生成的高胖矩阵的经济模式情况
    def test_random_trap_economic(self):
        # 创建一个随机数生成器，种子为1234
        rng = np.random.RandomState(1234)
        # 设置矩阵维度为100x200
        m = 100
        n = 200
        # 进行两次循环
        for k in range(2):
            # 生成一个m x n的随机矩阵a
            a = rng.random([m, n])
            # 进行经济模式的RQ分解，并返回结果r和q
            r, q = rq(a, mode='economic')
            # 断言q乘以其转置等于单位矩阵
            assert_array_almost_equal(q @ q.T, eye(m))
            # 断言r乘以q等于原始矩阵a
            assert_array_almost_equal(r @ q, a)
            # 断言q和r的形状正确
            assert_equal(q.shape, (m, n))
            assert_equal(r.shape, (m, m))

    # 测试复杂数据类型的情况
    def test_random_complex(self):
        # 创建一个包含复数的20x20的随机矩阵a
        rng = np.random.RandomState(1234)
        n = 20
        # 进行两次循环
        for k in range(2):
            # 生成一个n x n的复数随机矩阵a
            a = rng.random([n, n]) + 1j*rng.random([n, n])
            # 进行RQ分解，并返回结果r和q
            r, q = rq(a)
            # 断言q乘以其共轭转置等于单位矩阵
            assert
    # 定义测试方法，用于测试随机复数经济型RQ分解
    def test_random_complex_economic(self):
        # 使用种子1234初始化随机数生成器
        rng = np.random.RandomState(1234)
        # 设置矩阵维度
        m = 100
        n = 200
        # 循环两次
        for k in range(2):
            # 创建随机复数矩阵a，实部和虚部分别由随机数生成
            a = rng.random([m, n]) + 1j*rng.random([m, n])
            # 进行经济型RQ分解
            r, q = rq(a, mode='economic')
            # 断言q乘以其共轭转置应接近单位矩阵
            assert_array_almost_equal(q @ q.conj().T, eye(m))
            # 断言r乘以q应接近a
            assert_array_almost_equal(r @ q, a)
            # 断言q的形状应为(m, n)
            assert_equal(q.shape, (m, n))
            # 断言r的形状应为(m, m)
            assert_equal(r.shape, (m, m))

    # 定义测试方法，用于测试非有限检查的RQ分解
    def test_check_finite(self):
        # 定义输入矩阵a
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        # 进行RQ分解，关闭有限性检查
        r, q = rq(a, check_finite=False)
        # 断言q乘以其转置应接近单位矩阵
        assert_array_almost_equal(q @ q.T, eye(3))
        # 断言r乘以q应接近a
        assert_array_almost_equal(r @ q, a)

    # 定义测试方法，测试形状和数据类型
    @pytest.mark.parametrize("m", [0, 1, 2])
    @pytest.mark.parametrize("n", [0, 1, 2])
    @pytest.mark.parametrize('dtype', DTYPES)
    def test_shape_dtype(self, m, n, dtype):
        # 计算k为m和n的最小值
        k = min(m, n)

        # 创建指定数据类型的零矩阵a
        a = np.zeros((m, n), dtype=dtype)
        # 进行RQ分解
        r, q = rq(a)
        # 断言q的形状应为(n, n)
        assert_equal(q.shape, (n, n))
        # 断言r的形状应为(m, n)
        assert_equal(r.shape, (m, n))
        # 断言r的数据类型应为指定的dtype
        assert_equal(r.dtype, dtype)
        # 断言q的数据类型应为指定的dtype
        assert_equal(q.dtype, dtype)

        # 进行只返回r的RQ分解
        r = rq(a, mode='r')
        # 断言r的形状应为(m, n)
        assert_equal(r.shape, (m, n))
        # 断言r的数据类型应为指定的dtype
        assert_equal(r.dtype, dtype)

        # 进行经济型RQ分解
        r, q = rq(a, mode='economic')
        # 断言r的形状应为(m, k)
        assert_equal(r.shape, (m, k))
        # 断言r的数据类型应为指定的dtype
        assert_equal(r.dtype, dtype)
        # 断言q的形状应为(k, n)
        assert_equal(q.shape, (k, n))
        # 断言q的数据类型应为指定的dtype
        assert_equal(q.dtype, dtype)

    # 定义测试方法，测试空矩阵情况
    @pytest.mark.parametrize(("m", "n"), [(0, 0), (0, 2), (2, 0)])
    def test_empty(self, m, n):
        # 计算k为m和n的最小值
        k = min(m, n)

        # 创建空矩阵a
        a = np.empty((m, n))
        # 进行RQ分解
        r, q = rq(a)
        # 断言r接近空矩阵
        assert_allclose(r, np.empty((m, n)))
        # 断言q接近单位矩阵
        assert_allclose(q, np.identity(n))

        # 进行只返回r的RQ分解
        r = rq(a, mode='r')
        # 断言r接近空矩阵
        assert_allclose(r, np.empty((m, n)))

        # 进行经济型RQ分解
        r, q = rq(a, mode='economic')
        # 断言r接近空矩阵
        assert_allclose(r, np.empty((m, k)))
        # 断言q接近空矩阵
        assert_allclose(q, np.empty((k, n)))
class TestSchur:

    def check_schur(self, a, t, u, rtol, atol):
        # Check that the Schur decomposition is correct.
        assert_allclose(u @ t @ u.conj().T, a, rtol=rtol, atol=atol,
                        err_msg="Schur decomposition does not match 'a'")
        # The expected value of u @ u.H - I is all zeros, so test
        # with absolute tolerance only.
        assert_allclose(u @ u.conj().T - np.eye(len(u)), 0, rtol=0, atol=atol,
                        err_msg="u is not unitary")

    def test_simple(self):
        a = [[8, 12, 3], [2, 9, 3], [10, 3, 6]]
        t, z = schur(a)
        self.check_schur(a, t, z, rtol=1e-14, atol=5e-15)
        tc, zc = schur(a, 'complex')
        assert_(np.any(ravel(iscomplex(zc))) and np.any(ravel(iscomplex(tc))))
        self.check_schur(a, tc, zc, rtol=1e-14, atol=5e-15)
        tc2, zc2 = rsf2csf(tc, zc)
        self.check_schur(a, tc2, zc2, rtol=1e-14, atol=5e-15)

    @pytest.mark.parametrize(
        'sort, expected_diag',
        [('lhp', [-np.sqrt(2), -0.5, np.sqrt(2), 0.5]),
         ('rhp', [np.sqrt(2), 0.5, -np.sqrt(2), -0.5]),
         ('iuc', [-0.5, 0.5, np.sqrt(2), -np.sqrt(2)]),
         ('ouc', [np.sqrt(2), -np.sqrt(2), -0.5, 0.5]),
         (lambda x: x >= 0.0, [np.sqrt(2), 0.5, -np.sqrt(2), -0.5])]
    )
    def test_sort(self, sort, expected_diag):
        # The exact eigenvalues of this matrix are
        #   -sqrt(2), sqrt(2), -1/2, 1/2.
        a = [[4., 3., 1., -1.],
             [-4.5, -3.5, -1., 1.],
             [9., 6., -4., 4.5],
             [6., 4., -3., 3.5]]
        # Compute Schur decomposition with specified sorting order
        t, u, sdim = schur(a, sort=sort)
        self.check_schur(a, t, u, rtol=1e-14, atol=5e-15)
        # Check if the diagonal elements of t match the expected eigenvalues
        assert_allclose(np.diag(t), expected_diag, rtol=1e-12)
        # Check if sdim (number of selected eigenvalues) equals 2
        assert_equal(2, sdim)

    def test_sort_errors(self):
        a = [[4., 3., 1., -1.],
             [-4.5, -3.5, -1., 1.],
             [9., 6., -4., 4.5],
             [6., 4., -3., 3.5]]
        # Check that ValueError is raised for unsupported sorting criteria
        assert_raises(ValueError, schur, a, sort='unsupported')
        # Check that ValueError is raised for non-string or callable sort criteria
        assert_raises(ValueError, schur, a, sort=1)

    def test_check_finite(self):
        a = [[8, 12, 3], [2, 9, 3], [10, 3, 6]]
        # Compute Schur decomposition without checking for finite values
        t, z = schur(a, check_finite=False)
        # Check if the reconstructed matrix matches the original matrix
        assert_array_almost_equal(z @ t @ z.conj().T, a)

    @pytest.mark.parametrize('dt', [int, float, np.float32, complex, np.complex64])
    def test_empty(self, dt):
        # Create an empty matrix of specified dtype
        a = np.empty((0, 0), dtype=dt)
        # Compute Schur decomposition of empty matrix
        t, z = schur(a)
        # Compute Schur decomposition of 2x2 identity matrix with specified dtype
        t0, z0 = schur(np.eye(2, dtype=dt))
        # Check if both t and z are empty matrices
        assert_allclose(t, np.empty((0, 0)))
        assert_allclose(z, np.empty((0, 0)))
        # Check if dtypes of t and z match the dtype of t0 and z0 respectively
        assert t.dtype == t0.dtype
        assert z.dtype == z0.dtype

        # Compute Schur decomposition with sorting order 'lhp' of empty matrix
        t, z, sdim = schur(a, sort='lhp')
        # Check if t and z remain empty matrices
        assert_allclose(t, np.empty((0, 0)))
        assert_allclose(z, np.empty((0, 0)))
        # Check if sdim equals 0
        assert_equal(sdim, 0)
        # Check if dtypes of t and z match the dtype of t0 and z0 respectively
        assert t.dtype == t0.dtype
        assert z.dtype == z0.dtype


class TestHessenberg:
    # 定义一个测试方法，测试简单的情况
    def test_simple(self):
        # 创建一个实数矩阵 a
        a = [[-149, -50, -154],
             [537, 180, 546],
             [-27, -9, -25]]
        # 预期的上Hessenberg矩阵 h1
        h1 = [[-149.0000, 42.2037, -156.3165],
              [-537.6783, 152.5511, -554.9272],
              [0, 0.0728, 2.4489]]
        # 调用 hessenberg 函数，计算 h 和 q
        h, q = hessenberg(a, calc_q=1)
        # 断言计算结果：q^T @ a @ q 应等于 h
        assert_array_almost_equal(q.T @ a @ q, h)
        # 断言 h 应接近于预期的 h1，精度为 4 位小数
        assert_array_almost_equal(h, h1, decimal=4)

    # 定义一个测试方法，测试复数情况
    def test_simple_complex(self):
        # 创建一个复数矩阵 a
        a = [[-149, -50, -154],
             [537, 180j, 546],
             [-27j, -9, -25]]
        # 调用 hessenberg 函数，计算 h 和 q
        h, q = hessenberg(a, calc_q=1)
        # 断言计算结果：q^H @ a @ q 应等于 h
        assert_array_almost_equal(q.conj().T @ a @ q, h)

    # 定义一个测试方法，测试另一个简单情况
    def test_simple2(self):
        # 创建另一个实数矩阵 a
        a = [[1, 2, 3, 4, 5, 6, 7],
             [0, 2, 3, 4, 6, 7, 2],
             [0, 2, 2, 3, 0, 3, 2],
             [0, 0, 2, 8, 0, 0, 2],
             [0, 3, 1, 2, 0, 1, 2],
             [0, 1, 2, 3, 0, 1, 0],
             [0, 0, 0, 0, 0, 1, 2]]
        # 调用 hessenberg 函数，计算 h 和 q
        h, q = hessenberg(a, calc_q=1)
        # 断言计算结果：q^T @ a @ q 应等于 h
        assert_array_almost_equal(q.T @ a @ q, h)

    # 定义一个测试方法，测试单位矩阵情况
    def test_simple3(self):
        # 创建一个单位矩阵 a，并修改其元素
        a = np.eye(3)
        a[-1, 0] = 2
        # 调用 hessenberg 函数，计算 h 和 q
        h, q = hessenberg(a, calc_q=1)
        # 断言计算结果：q^T @ a @ q 应等于 h
        assert_array_almost_equal(q.T @ a @ q, h)

    # 定义一个测试方法，测试随机生成的实数矩阵
    def test_random(self):
        # 使用随机数生成器创建矩阵 a
        rng = np.random.RandomState(1234)
        n = 20
        for k in range(2):
            a = rng.random([n, n])
            # 调用 hessenberg 函数，计算 h 和 q
            h, q = hessenberg(a, calc_q=1)
            # 断言计算结果：q^T @ a @ q 应等于 h
            assert_array_almost_equal(q.T @ a @ q, h)

    # 定义一个测试方法，测试随机生成的复数矩阵
    def test_random_complex(self):
        # 使用随机数生成器创建复数矩阵 a
        rng = np.random.RandomState(1234)
        n = 20
        for k in range(2):
            a = rng.random([n, n]) + 1j*rng.random([n, n])
            # 调用 hessenberg 函数，计算 h 和 q
            h, q = hessenberg(a, calc_q=1)
            # 断言计算结果：q^H @ a @ q 应等于 h
            assert_array_almost_equal(q.conj().T @ a @ q, h)

    # 定义一个测试方法，测试非有限数的情况
    def test_check_finite(self):
        # 创建一个实数矩阵 a
        a = [[-149, -50, -154],
             [537, 180, 546],
             [-27, -9, -25]]
        # 预期的上Hessenberg矩阵 h1
        h1 = [[-149.0000, 42.2037, -156.3165],
              [-537.6783, 152.5511, -554.9272],
              [0, 0.0728, 2.4489]]
        # 调用 hessenberg 函数，计算 h 和 q，不检查有限性
        h, q = hessenberg(a, calc_q=1, check_finite=False)
        # 断言计算结果：q^T @ a @ q 应等于 h
        assert_array_almost_equal(q.T @ a @ q, h)
        # 断言 h 应接近于预期的 h1，精度为 4 位小数
        assert_array_almost_equal(h, h1, decimal=4)

    # 定义一个测试方法，测试2x2矩阵
    def test_2x2(self):
        # 创建一个2x2实数矩阵 a
        a = [[2, 1], [7, 12]]
        # 调用 hessenberg 函数，计算 h 和 q
        h, q = hessenberg(a, calc_q=1)
        # 断言 q 应等于单位矩阵
        assert_array_almost_equal(q, np.eye(2))
        # 断言 h 应等于原始矩阵 a
        assert_array_almost_equal(h, a)

        # 创建一个2x2复数矩阵 b
        b = [[2-7j, 1+2j], [7+3j, 12-2j]]
        # 调用 hessenberg 函数，计算 h 和 q
        h2, q2 = hessenberg(b, calc_q=1)
        # 断言 q2 应等于单位矩阵
        assert_array_almost_equal(q2, np.eye(2))
        # 断言 h2 应等于原始矩阵 b
        assert_array_almost_equal(h2, b)

    # 定义一个测试方法，测试空矩阵情况
    @pytest.mark.parametrize('dt', [int, float, float32, complex, complex64])
    def test_empty(self, dt):
        # 创建一个空矩阵 a，指定数据类型为 dt
        a = np.empty((0, 0), dtype=dt)
        # 调用 hessenberg 函数，计算 h
        h = hessenberg(a)
        # 断言 h 的形状应为 (0, 0)
        assert h.shape == (0, 0)
        # 断言 h 的数据类型应与单位矩阵的数据类型相同
        assert h.dtype == hessenberg(np.eye(3, dtype=dt)).dtype

        # 再次调用 hessenberg 函数，计算 h 和 q
        h, q = hessenberg(a, calc_q=True)
        h3, q3 = hessenberg(a, calc_q=True)
        # 断言 h 的形状应为 (0, 0)
        assert h.shape == (0, 0)
        # 断言 h 的数据类型应与 h3 的数据类型相同
        assert h.dtype == h3.dtype

        # 断言 q 的形状应为 (0, 0)
        assert q.shape == (0, 0)
        # 断言 q 的数据类型应与 q3 的数据类型相同
        assert q.dtype == q3.dtype
# 初始化变量 `blas_provider` 和 `blas_version`，初始值设为 `None`
blas_provider = blas_version = None

# 如果 `CONFIG` 不为 `None`，则从配置中获取 BLAS 的名称和版本
if CONFIG is not None:
    blas_provider = CONFIG['Build Dependencies']['blas']['name']
    blas_version = CONFIG['Build Dependencies']['blas']['version']


# 定义一个测试类 `TestQZ`
class TestQZ:
    
    # 测试单精度浮点数 QZ 分解
    def test_qz_single(self):
        # 初始化随机数生成器 `rng`
        rng = np.random.RandomState(12345)
        # 设置矩阵维度 `n`
        n = 5
        # 生成随机矩阵 A 和 B，转换为单精度浮点数类型
        A = rng.random([n, n]).astype(float32)
        B = rng.random([n, n]).astype(float32)
        # 进行 QZ 分解
        AA, BB, Q, Z = qz(A, B)
        # 检查 Q * AA * Z^T 是否近似等于 A，精度为 5 位小数
        assert_array_almost_equal(Q @ AA @ Z.T, A, decimal=5)
        # 检查 Q * BB * Z^T 是否近似等于 B，精度为 5 位小数
        assert_array_almost_equal(Q @ BB @ Z.T, B, decimal=5)
        # 检查 Q * Q^T 是否近似等于单位矩阵
        assert_array_almost_equal(Q @ Q.T, eye(n), decimal=5)
        # 检查 Z * Z^T 是否近似等于单位矩阵
        assert_array_almost_equal(Z @ Z.T, eye(n), decimal=5)
        # 检查 BB 的对角线元素是否都大于等于 0
        assert_(np.all(diag(BB) >= 0))

    # 测试双精度浮点数 QZ 分解
    def test_qz_double(self):
        rng = np.random.RandomState(12345)
        n = 5
        A = rng.random([n, n])
        B = rng.random([n, n])
        AA, BB, Q, Z = qz(A, B)
        assert_array_almost_equal(Q @ AA @ Z.T, A)
        assert_array_almost_equal(Q @ BB @ Z.T, B)
        assert_array_almost_equal(Q @ Q.T, eye(n))
        assert_array_almost_equal(Z @ Z.T, eye(n))
        assert_(np.all(diag(BB) >= 0))

    # 测试复数 QZ 分解
    def test_qz_complex(self):
        rng = np.random.RandomState(12345)
        n = 5
        # 生成复数随机矩阵 A 和 B
        A = rng.random([n, n]) + 1j*rng.random([n, n])
        B = rng.random([n, n]) + 1j*rng.random([n, n])
        # 进行复数 QZ 分解
        AA, BB, Q, Z = qz(A, B)
        # 检查 Q * AA * Z^H 是否近似等于 A
        assert_array_almost_equal(Q @ AA @ Z.conj().T, A)
        # 检查 Q * BB * Z^H 是否近似等于 B
        assert_array_almost_equal(Q @ BB @ Z.conj().T, B)
        # 检查 Q * Q^H 是否近似等于单位矩阵
        assert_array_almost_equal(Q @ Q.conj().T, eye(n))
        # 检查 Z * Z^H 是否近似等于单位矩阵
        assert_array_almost_equal(Z @ Z.conj().T, eye(n))
        # 检查 BB 的对角线元素是否都大于等于 0
        assert_(np.all(diag(BB) >= 0))
        # 检查 BB 的对角线元素的虚部是否为 0
        assert_(np.all(diag(BB).imag == 0))

    # 测试复数64位 QZ 分解
    def test_qz_complex64(self):
        rng = np.random.RandomState(12345)
        n = 5
        # 生成复数64位随机矩阵 A 和 B
        A = (rng.random([n, n]) + 1j*rng.random([n, n])).astype(complex64)
        B = (rng.random([n, n]) + 1j*rng.random([n, n])).astype(complex64)
        # 进行复数64位 QZ 分解
        AA, BB, Q, Z = qz(A, B)
        # 检查 Q * AA * Z^H 是否近似等于 A，精度为 5 位小数
        assert_array_almost_equal(Q @ AA @ Z.conj().T, A, decimal=5)
        # 检查 Q * BB * Z^H 是否近似等于 B，精度为 5 位小数
        assert_array_almost_equal(Q @ BB @ Z.conj().T, B, decimal=5)
        # 检查 Q * Q^H 是否近似等于单位矩阵，精度为 5 位小数
        assert_array_almost_equal(Q @ Q.conj().T, eye(n), decimal=5)
        # 检查 Z * Z^H 是否近似等于单位矩阵，精度为 5 位小数
        assert_array_almost_equal(Z @ Z.conj().T, eye(n), decimal=5)
        # 检查 BB 的对角线元素是否都大于等于 0
        assert_(np.all(diag(BB) >= 0))
        # 检查 BB 的对角线元素的虚部是否为 0
        assert_(np.all(diag(BB).imag == 0))

    # 测试双精度浮点数和复数混合 QZ 分解
    def test_qz_double_complex(self):
        rng = np.random.RandomState(12345)
        n = 5
        A = rng.random([n, n])
        B = rng.random([n, n])
        # 进行双精度浮点数和复数混合 QZ 分解
        AA, BB, Q, Z = qz(A, B, output='complex')
        # 计算 Q * AA * Z^H
        aa = Q @ AA @ Z.conj().T
        # 检查 aa 的实部是否近似等于 A
        assert_array_almost_equal(aa.real, A)
        # 检查 aa 的虚部是否近似等于 0
        assert_array_almost_equal(aa.imag, 0)
        # 计算 Q * BB * Z^H
        bb = Q @ BB @ Z.conj().T
        # 检查 bb 的实部是否近似等于 B
        assert_array_almost_equal(bb.real, B)
        # 检查 bb 的虚部是否近似等于 0
        assert_array_almost_equal(bb.imag, 0)
        # 检查 Q * Q^H 是否近似等于单位矩阵
        assert_array_almost_equal(Q @ Q.conj().T, eye(n))
        # 检查 Z * Z^H 是否近似等于单位矩阵
        assert_array_almost_equal(Z @ Z.conj().T, eye(n))
        # 检查 BB 的对角线元素是否都大于等于 0
        assert_(np.all(diag(BB) >= 0))
    # 定义一个测试函数，用于检查矩阵的有限性
    def test_check_finite(self):
        # 使用随机数种子创建随机数生成器
        rng = np.random.RandomState(12345)
        # 定义矩阵的大小
        n = 5
        # 生成随机矩阵 A
        A = rng.random([n, n])
        # 生成随机矩阵 B
        B = rng.random([n, n])
        # 对矩阵 A 和 B 进行 QZ 分解，不检查有限性
        AA, BB, Q, Z = qz(A, B, check_finite=False)
        # 断言 Q * AA * Z^T 等于 A，即 QZ 分解的正确性
        assert_array_almost_equal(Q @ AA @ Z.T, A)
        # 断言 Q * BB * Z^T 等于 B，即 QZ 分解的正确性
        assert_array_almost_equal(Q @ BB @ Z.T, B)
        # 断言 Q * Q^T 等于单位矩阵，即 Q 是正交矩阵
        assert_array_almost_equal(Q @ Q.T, eye(n))
        # 断言 Z * Z^T 等于单位矩阵，即 Z 是正交矩阵
        assert_array_almost_equal(Z @ Z.T, eye(n))
        # 断言 BB 的对角线元素均大于等于零
        assert_(np.all(np.diag(BB) >= 0))
class TestOrdQZ:
    @classmethod
    def setup_class(cls):
        # 定义复杂矩阵 A1
        A1 = np.array([[-21.10 - 22.50j, 53.5 - 50.5j, -34.5 + 127.5j,
                        7.5 + 0.5j],
                       [-0.46 - 7.78j, -3.5 - 37.5j, -15.5 + 58.5j,
                        -10.5 - 1.5j],
                       [4.30 - 5.50j, 39.7 - 17.1j, -68.5 + 12.5j,
                        -7.5 - 3.5j],
                       [5.50 + 4.40j, 14.4 + 43.3j, -32.5 - 46.0j,
                        -19.0 - 32.5j]])

        # 定义复杂矩阵 B1
        B1 = np.array([[1.0 - 5.0j, 1.6 + 1.2j, -3 + 0j, 0.0 - 1.0j],
                       [0.8 - 0.6j, .0 - 5.0j, -4 + 3j, -2.4 - 3.2j],
                       [1.0 + 0.0j, 2.4 + 1.8j, -4 - 5j, 0.0 - 3.0j],
                       [0.0 + 1.0j, -1.8 + 2.4j, 0 - 4j, 4.0 - 5.0j]])

        # 定义矩阵 A2
        A2 = np.array([[3.9, 12.5, -34.5, -0.5],
                       [4.3, 21.5, -47.5, 7.5],
                       [4.3, 21.5, -43.5, 3.5],
                       [4.4, 26.0, -46.0, 6.0]])

        # 定义矩阵 B2
        B2 = np.array([[1, 2, -3, 1],
                       [1, 3, -5, 4],
                       [1, 3, -4, 3],
                       [1, 3, -4, 4]])

        # 定义矩阵 A3
        A3 = np.array([[5., 1., 3., 3.],
                       [4., 4., 2., 7.],
                       [7., 4., 1., 3.],
                       [0., 4., 8., 7.]])
        
        # 定义矩阵 B3
        B3 = np.array([[8., 10., 6., 10.],
                       [7., 7., 2., 9.],
                       [9., 1., 6., 6.],
                       [5., 1., 4., 7.]])

        # 定义矩阵 A4，单位矩阵
        A4 = np.eye(2)
        
        # 定义矩阵 B4，对角矩阵
        B4 = np.diag([0, 1])

        # 定义矩阵 A5，对角矩阵
        A5 = np.diag([1, 0])

        # 将所有矩阵存入类属性 A 和 B 中
        cls.A = [A1, A2, A3, A4, A5]
        cls.B = [B1, B2, B3, B4, A5]

    def qz_decomp(self, sort):
        # 设置numpy的错误处理，遇到错误则抛出异常
        with np.errstate(all='raise'):
            # 对类属性 A 和 B 中的每对矩阵调用 ordqz 函数进行QZ分解
            ret = [ordqz(Ai, Bi, sort=sort) for Ai, Bi in zip(self.A, self.B)]
        # 返回QZ分解的结果
        return tuple(ret)
    def check(self, A, B, sort, AA, BB, alpha, beta, Q, Z):
        # 构造单位矩阵，确保 Q 和 Z 是正交的
        Id = np.eye(*A.shape)
        assert_array_almost_equal(Q @ Q.T.conj(), Id)  # 检查 Q 是否正交
        assert_array_almost_equal(Z @ Z.T.conj(), Id)  # 检查 Z 是否正交
        # 检查分解
        assert_array_almost_equal(Q @ AA, A @ Z)  # 检查 Q @ AA 是否等于 A @ Z
        assert_array_almost_equal(Q @ BB, B @ Z)  # 检查 Q @ BB 是否等于 B @ Z
        # 检查 AA 和 BB 的形状
        assert_array_equal(np.tril(AA, -2), np.zeros(AA.shape))  # 检查 AA 的下三角（超过-2位置）是否为零
        assert_array_equal(np.tril(BB, -1), np.zeros(BB.shape))  # 检查 BB 的下三角（超过-1位置）是否为零
        # 检查特征值
        for i in range(A.shape[0]):
            # 当前对角线元素是否属于已经检查过的2×2块？
            if i > 0 and A[i, i - 1] != 0:
                continue
            # 处理2×2块
            if i < AA.shape[0] - 1 and AA[i + 1, i] != 0:
                evals, _ = eig(AA[i:i + 2, i:i + 2], BB[i:i + 2, i:i + 2])
                # 确保复共轭特征值对按顺序排列（正虚部优先）
                if evals[0].imag < 0:
                    evals = evals[[1, 0]]
                tmp = alpha[i:i + 2] / beta[i:i + 2]
                if tmp[0].imag < 0:
                    tmp = tmp[[1, 0]]
                assert_array_almost_equal(evals, tmp)
            else:
                if alpha[i] == 0 and beta[i] == 0:
                    assert_equal(AA[i, i], 0)
                    assert_equal(BB[i, i], 0)
                elif beta[i] == 0:
                    assert_equal(BB[i, i], 0)
                else:
                    assert_almost_equal(AA[i, i] / BB[i, i], alpha[i] / beta[i])
        # 根据排序函数选择排序方法
        sortfun = _select_function(sort)
        lastsort = True
        for i in range(A.shape[0]):
            cursort = sortfun(np.array([alpha[i]]), np.array([beta[i]]))
            # 如果排序标准不符合，后续的特征值也不应该匹配
            if not lastsort:
                assert not cursort
            lastsort = cursort

    def check_all(self, sort):
        # 执行 QZ 分解并检查所有对
        ret = self.qz_decomp(sort)

        for reti, Ai, Bi in zip(ret, self.A, self.B):
            self.check(Ai, Bi, sort, *reti)

    def test_lhp(self):
        # 执行所有左半平面测试
        self.check_all('lhp')

    def test_rhp(self):
        # 执行所有右半平面测试
        self.check_all('rhp')

    def test_iuc(self):
        # 执行所有内逆稳定测试
        self.check_all('iuc')

    def test_ouc(self):
        # 执行所有外逆稳定测试
        self.check_all('ouc')

    def test_ref(self):
        # 首先测试实特征值（左上角）
        def sort(x, y):
            out = np.empty_like(x, dtype=bool)
            nonzero = (y != 0)
            out[~nonzero] = False
            out[nonzero] = (x[nonzero] / y[nonzero]).imag == 0
            return out

        self.check_all(sort)
    def test_cef(self):
        # 复杂特征值优先（左上角）
        def sort(x, y):
            # 创建一个与 x 相同形状的空数组，数据类型为布尔型
            out = np.empty_like(x, dtype=bool)
            # 找出 y 不为零的索引
            nonzero = (y != 0)
            # 对于 y 不为零的元素，将其对应位置的 out 数组设为 x/y 的虚部不为零的布尔值
            out[~nonzero] = False
            out[nonzero] = (x[nonzero]/y[nonzero]).imag != 0
            return out

        self.check_all(sort)

    def test_diff_input_types(self):
        # 对 ordqz 函数使用 self.A[1] 和 self.B[2] 进行计算，并指定排序方式为 'lhp'
        ret = ordqz(self.A[1], self.B[2], sort='lhp')
        # 检查返回值是否符合预期
        self.check(self.A[1], self.B[2], 'lhp', *ret)

        # 对 ordqz 函数使用 self.B[2] 和 self.A[1] 进行计算，并指定排序方式为 'lhp'
        ret = ordqz(self.B[2], self.A[1], sort='lhp')
        # 检查返回值是否符合预期
        self.check(self.B[2], self.A[1], 'lhp', *ret)

    def test_sort_explicit(self):
        # 测试在 2x2 情况下特征值的顺序，其中可以显式计算解
        A1 = np.eye(2)
        B1 = np.diag([-2, 0.5])
        expected1 = [('lhp', [-0.5, 2]),
                     ('rhp', [2, -0.5]),
                     ('iuc', [-0.5, 2]),
                     ('ouc', [2, -0.5])]
        
        A2 = np.eye(2)
        B2 = np.diag([-2 + 1j, 0.5 + 0.5j])
        expected2 = [('lhp', [1/(-2 + 1j), 1/(0.5 + 0.5j)]),
                     ('rhp', [1/(0.5 + 0.5j), 1/(-2 + 1j)]),
                     ('iuc', [1/(-2 + 1j), 1/(0.5 + 0.5j)]),
                     ('ouc', [1/(0.5 + 0.5j), 1/(-2 + 1j)])]
        
        A3 = np.eye(2)
        B3 = np.diag([2, 0])
        expected3 = [('rhp', [0.5, np.inf]),
                     ('iuc', [0.5, np.inf]),
                     ('ouc', [np.inf, 0.5])]
        
        A4 = np.eye(2)
        B4 = np.diag([-2, 0])
        expected4 = [('lhp', [-0.5, np.inf]),
                     ('iuc', [-0.5, np.inf]),
                     ('ouc', [np.inf, -0.5])]
        
        A5 = np.diag([0, 1])
        B5 = np.diag([0, 0.5])
        expected5 = [('rhp', [2, np.nan]),
                     ('ouc', [2, np.nan])]

        A = [A1, A2, A3, A4, A5]
        B = [B1, B2, B3, B4, B5]
        expected = [expected1, expected2, expected3, expected4, expected5]
        for Ai, Bi, expectedi in zip(A, B, expected):
            for sortstr, expected_eigvals in expectedi:
                # 调用 ordqz 函数计算特征值，并返回结果
                _, _, alpha, beta, _, _ = ordqz(Ai, Bi, sort=sortstr)
                # 找出 alpha 和 beta 等于零的位置
                azero = (alpha == 0)
                bzero = (beta == 0)
                # 创建一个与 alpha 相同形状的空数组
                x = np.empty_like(alpha)
                # 根据 alpha 和 beta 的值填充 x 数组
                x[azero & bzero] = np.nan
                x[~azero & bzero] = np.inf
                x[~bzero] = alpha[~bzero]/beta[~bzero]
                # 断言计算得到的特征值与预期值接近
                assert_allclose(expected_eigvals, x)
class TestOrdQZWorkspaceSize:
    # 设置标记以便在测试运行时指定速度较慢的测试失败阈值为5秒
    @pytest.mark.fail_slow(5)
    # 定义测试函数，用于测试 ordqz 函数的分解功能
    def test_decompose(self):
        # 使用随机种子12345初始化随机数生成器
        rng = np.random.RandomState(12345)
        # 定义矩阵维度 N
        N = 202
        # 对于浮点数类型 np.float32 和 np.float64 进行测试
        for ddtype in [np.float32, np.float64]:
            # 生成随机的 N×N 浮点数矩阵 A 和 B
            A = rng.random((N, N)).astype(ddtype)
            B = rng.random((N, N)).astype(ddtype)
            # 调用 ordqz 函数，对矩阵进行排序并返回实部结果
            _ = ordqz(A, B, sort=lambda alpha, beta: alpha < beta,
                      output='real')

        # 对于复数类型 np.complex128 和 np.complex64 进行测试
        for ddtype in [np.complex128, np.complex64]:
            # 生成随机的 N×N 复数矩阵 A 和 B
            A = rng.random((N, N)).astype(ddtype)
            B = rng.random((N, N)).astype(ddtype)
            # 调用 ordqz 函数，对矩阵进行排序并返回复数结果
            _ = ordqz(A, B, sort=lambda alpha, beta: alpha < beta,
                      output='complex')

    # 设置标记以便在测试运行时指定速度较慢的测试
    @pytest.mark.slow
    # 定义测试函数，用于测试 ordqz 函数的分解功能（使用 'ouc' 排序）
    def test_decompose_ouc(self):
        # 使用随机种子12345初始化随机数生成器
        rng = np.random.RandomState(12345)
        # 定义矩阵维度 N
        N = 202
        # 对于浮点数和复数类型 np.float32, np.float64, np.complex128, np.complex64 进行测试
        for ddtype in [np.float32, np.float64, np.complex128, np.complex64]:
            # 生成随机的 N×N 矩阵 A 和 B
            A = rng.random((N, N)).astype(ddtype)
            B = rng.random((N, N)).astype(ddtype)
            # 调用 ordqz 函数，对矩阵进行 'ouc' 排序并返回多个结果
            S, T, alpha, beta, U, V = ordqz(A, B, sort='ouc')


class TestDatacopied:
    # 定义测试函数，用于测试 _datacopied 函数
    def test_datacopied(self):
        # 导入 _datacopied 函数
        from scipy.linalg._decomp import _datacopied

        # 创建矩阵 M
        M = matrix([[0, 1], [2, 3]])
        # 将矩阵 M 转换为 ndarray 类型的 A
        A = asarray(M)
        # 将矩阵 M 转换为列表类型的 L
        L = M.tolist()
        # 复制矩阵 M 到 M2
        M2 = M.copy()

        # 定义 Fake1 类，实现 __array__ 方法返回 A
        class Fake1:
            def __array__(self, dtype=None, copy=None):
                return A

        # 定义 Fake2 类，实现 __array_interface__ 属性与 A 相同
        class Fake2:
            __array_interface__ = A.__array_interface__

        # 实例化 Fake1 和 Fake2 类
        F1 = Fake1()
        F2 = Fake2()

        # 遍历测试用例 [(M, False), (A, False), (L, True), (M2, False), (F1, False), (F2, False)]
        for item, status in [(M, False), (A, False), (L, True),
                             (M2, False), (F1, False), (F2, False)]:
            # 将 item 转换为 ndarray 类型的 arr
            arr = asarray(item)
            # 调用 _datacopied 函数检查 arr 是否复制了 item
            assert_equal(_datacopied(arr, item), status,
                         err_msg=repr(item))


# 定义测试函数，用于测试 linalg 在非对齐内存上的工作（float32）
def test_aligned_mem_float():
    """Check linalg works with non-aligned memory (float32)"""
    # 分配402字节内存（在边界上分配）
    a = arange(402, dtype=np.uint8)

    # 使用偏移量2创建一个数组，从 a.data 开始，总计100个元素，数据类型为 float32
    z = np.frombuffer(a.data, offset=2, count=100, dtype=float32)
    # 调整数组形状为10×10
    z.shape = 10, 10

    # 对 z 进行特征值分解，覆盖原始数据
    eig(z, overwrite_a=True)
    # 对 z 的转置进行特征值分解，覆盖原始数据
    eig(z.T, overwrite_a=True)


# 设置标记以便在特定平台上跳过测试（如果是 'ppc64le' 平台，则跳过）
@pytest.mark.skipif(platform.machine() == 'ppc64le',
                    reason="crashes on ppc64le")
# 定义测试函数，用于测试 linalg 在非对齐内存上的工作（float64）
def test_aligned_mem():
    """Check linalg works with non-aligned memory (float64)"""
    # 分配804字节内存（在边界上分配）
    a = arange(804, dtype=np.uint8)

    # 使用偏移量4创建一个数组，从 a.data 开始，总计100个元素，数据类型为 float64
    z = np.frombuffer(a.data, offset=4, count=100, dtype=float)
    # 调整数组形状为10×10
    z.shape = 10, 10

    # 对 z 进行特征值分解，覆盖原始数据
    eig(z, overwrite_a=True)
    # 对 z 的转置进行特征值分解，覆盖原始数据
    eig(z.T, overwrite_a=True)


# 定义测试函数，用于测试复杂对象在不完全对齐内存上的工作
def test_aligned_mem_complex():
    """Check that complex objects don't need to be completely aligned"""
    # 分配1608字节内存（在边界上分配）
    a = zeros(1608, dtype=np.uint8)
    # 使用 np.frombuffer() 从给定的字节数据 `a.data` 中创建一个数组
    # `offset=8` 表示从偏移量为8的位置开始读取数据
    # `count=100` 表示读取100个元素
    # `dtype=complex` 指定数组的元素类型为复数
    z = np.frombuffer(a.data, offset=8, count=100, dtype=complex)
    
    # 将数组 `z` 重新调整形状为 10x10 的二维数组
    z.shape = 10, 10
    
    # 对数组 `z` 进行特征值分解，`overwrite_a=True` 表示可以重用数组 `z` 的存储空间
    eig(z, overwrite_a=True)
    
    # 对数组 `z` 的转置进行特征值分解，`overwrite_a=True` 表示可以重用数组 `z` 的存储空间
    # 这里不需要特殊处理，仅是对 `z` 的转置进行相同的操作
    eig(z.T, overwrite_a=True)
# 定义一个函数用于检查是否存在 LAPACK 调用时的内存对齐问题
def check_lapack_misaligned(func, args, kwargs):
    # 将参数 args 转换为列表
    args = list(args)
    # 遍历 args 的索引范围
    for i in range(len(args)):
        # 复制一份 args 的副本
        a = args[:]
        # 如果 args[i] 是 numpy 数组
        if isinstance(a[i], np.ndarray):
            # 创建一个新的、稍微错位的数组 aa，用于复现问题
            aa = np.zeros(a[i].size * a[i].dtype.itemsize + 8, dtype=np.uint8)
            aa = np.frombuffer(aa.data, offset=4, count=a[i].size,
                               dtype=a[i].dtype)
            aa.shape = a[i].shape
            aa[...] = a[i]
            # 将 a[i] 替换为 aa
            a[i] = aa
            # 调用函数 func，并传入修改后的参数 a 和 kwargs
            func(*a, **kwargs)
            # 如果 a[i] 的维度大于 1
            if len(a[i].shape) > 1:
                # 对 a[i] 进行转置操作
                a[i] = a[i].T
                # 再次调用函数 func，并传入修改后的参数 a 和 kwargs
                func(*a, **kwargs)


# 为了跳过此测试，使用 pytest 的 xfail 标记，并提供原因
@pytest.mark.xfail(run=False,
                   reason="Ticket #1152, triggers a segfault in rare cases.")
# 定义一个测试函数，用于测试 LAPACK 在不正确对齐时的行为
def test_lapack_misaligned():
    # 创建多个测试用例
    M = np.eye(10, dtype=float)
    R = np.arange(100)
    R.shape = 10, 10
    S = np.arange(20000, dtype=np.uint8)
    S = np.frombuffer(S.data, offset=4, count=100, dtype=float)
    S.shape = 10, 10
    b = np.ones(10)
    # 对每个测试用例执行 LAPACK 函数调用，并检查其行为
    LU, piv = lu_factor(S)
    for (func, args, kwargs) in [
            (eig, (S,), dict(overwrite_a=True)),  # crash
            (eigvals, (S,), dict(overwrite_a=True)),  # no crash
            (lu, (S,), dict(overwrite_a=True)),  # no crash
            (lu_factor, (S,), dict(overwrite_a=True)),  # no crash
            (lu_solve, ((LU, piv), b), dict(overwrite_b=True)),
            (solve, (S, b), dict(overwrite_a=True, overwrite_b=True)),
            (svd, (M,), dict(overwrite_a=True)),  # no crash
            (svd, (R,), dict(overwrite_a=True)),  # no crash
            (svd, (S,), dict(overwrite_a=True)),  # crash
            (svdvals, (S,), dict()),  # no crash
            (svdvals, (S,), dict(overwrite_a=True)),  # crash
            (cholesky, (M,), dict(overwrite_a=True)),  # no crash
            (qr, (S,), dict(overwrite_a=True)),  # crash
            (rq, (S,), dict(overwrite_a=True)),  # crash
            (hessenberg, (S,), dict(overwrite_a=True)),  # crash
            (schur, (S,), dict(overwrite_a=True)),  # crash
            ]:
        # 对每个测试用例调用检查函数
        check_lapack_misaligned(func, args, kwargs)


# 定义一个测试类 TestOverwrite，用于测试多个 LAPACK 函数的参数不重叠情况
class TestOverwrite:
    # 定义测试方法 test_eig，验证 eig 函数的参数是否不重叠
    def test_eig(self):
        assert_no_overwrite(eig, [(3, 3)])
        assert_no_overwrite(eig, [(3, 3), (3, 3)])

    # 定义测试方法 test_eigh，验证 eigh 函数的参数是否不重叠
    def test_eigh(self):
        assert_no_overwrite(eigh, [(3, 3)])
        assert_no_overwrite(eigh, [(3, 3), (3, 3)])

    # 定义测试方法 test_eig_banded，验证 eig_banded 函数的参数是否不重叠
    def test_eig_banded(self):
        assert_no_overwrite(eig_banded, [(3, 2)])

    # 定义测试方法 test_eigvals，验证 eigvals 函数的参数是否不重叠
    def test_eigvals(self):
        assert_no_overwrite(eigvals, [(3, 3)])

    # 定义测试方法 test_eigvalsh，验证 eigvalsh 函数的参数是否不重叠
    def test_eigvalsh(self):
        assert_no_overwrite(eigvalsh, [(3, 3)])

    # 定义测试方法 test_eigvals_banded，验证 eigvals_banded 函数的参数是否不重叠
    def test_eigvals_banded(self):
        assert_no_overwrite(eigvals_banded, [(3, 2)])

    # 定义测试方法 test_hessenberg，验证 hessenberg 函数的参数是否不重叠
    def test_hessenberg(self):
        assert_no_overwrite(hessenberg, [(3, 3)])

    # 定义测试方法 test_lu_factor，验证 lu_factor 函数的参数是否不重叠
    def test_lu_factor(self):
        assert_no_overwrite(lu_factor, [(3, 3)])
    # 定义一个测试方法，用于测试 lu_solve 函数的行为
    def test_lu_solve(self):
        # 创建一个 3x3 的 NumPy 数组 x
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 8]])
        # 对数组 x 进行 LU 分解，并返回分解后的结果
        xlu = lu_factor(x)
        # 断言 lu_solve 函数在不覆盖数据的情况下解方程组，输入维度为 (3,)
        assert_no_overwrite(lambda b: lu_solve(xlu, b), [(3,)])

    # 定义一个测试方法，用于测试 lu 函数的行为
    def test_lu(self):
        # 断言 lu 函数在不覆盖数据的情况下执行 LU 分解，输入维度为 (3, 3)
        assert_no_overwrite(lu, [(3, 3)])

    # 定义一个测试方法，用于测试 qr 函数的行为
    def test_qr(self):
        # 断言 qr 函数在不覆盖数据的情况下执行 QR 分解，输入维度为 (3, 3)
        assert_no_overwrite(qr, [(3, 3)])

    # 定义一个测试方法，用于测试 rq 函数的行为
    def test_rq(self):
        # 断言 rq 函数在不覆盖数据的情况下执行 RQ 分解，输入维度为 (3, 3)
        assert_no_overwrite(rq, [(3, 3)])

    # 定义一个测试方法，用于测试 schur 函数的行为
    def test_schur(self):
        # 断言 schur 函数在不覆盖数据的情况下执行 Schur 分解，输入维度为 (3, 3)
        assert_no_overwrite(schur, [(3, 3)])

    # 定义一个测试方法，用于测试 schur 函数处理复数时的行为
    def test_schur_complex(self):
        # 断言带有复数输入的 schur 函数在不覆盖数据的情况下执行 Schur 分解，输入维度为 (3, 3)
        # 要求输入数据类型为 np.float32 或 np.float64
        assert_no_overwrite(lambda a: schur(a, 'complex'), [(3, 3)],
                            dtypes=[np.float32, np.float64])

    # 定义一个测试方法，用于测试 svd 函数的行为
    def test_svd(self):
        # 断言 svd 函数在不覆盖数据的情况下执行奇异值分解，输入维度为 (3, 3)
        assert_no_overwrite(svd, [(3, 3)])
        # 断言带有 lapack_driver 参数的 svd 函数在不覆盖数据的情况下执行奇异值分解，输入维度为 (3, 3)
        assert_no_overwrite(lambda a: svd(a, lapack_driver='gesvd'), [(3, 3)])

    # 定义一个测试方法，用于测试 svdvals 函数的行为
    def test_svdvals(self):
        # 断言 svdvals 函数在不覆盖数据的情况下执行奇异值分解，输入维度为 (3, 3)
        assert_no_overwrite(svdvals, [(3, 3)])
# 定义一个函数来检查正交性，参数包括 n（矩阵维度）、dtype（数据类型）、skip_big（是否跳过大数据量的检查，默认为False）
def _check_orth(n, dtype, skip_big=False):
    # 创建一个 n 行 2 列的全一矩阵，并指定数据类型为 dtype
    X = np.ones((n, 2), dtype=float).astype(dtype)

    # 获取指定数据类型的机器精度
    eps = np.finfo(dtype).eps
    # 设置容忍度为 1000 倍的机器精度
    tol = 1000 * eps

    # 对 X 进行正交化处理，返回正交化后的矩阵 Y
    Y = orth(X)
    # 断言 Y 的形状为 (n, 1)
    assert_equal(Y.shape, (n, 1))
    # 断言 Y 的所有元素近似相等于其均值，容忍度为 atol=tol
    assert_allclose(Y, Y.mean(), atol=tol)

    # 对 X 的转置进行正交化处理，返回正交化后的矩阵 Y
    Y = orth(X.T)
    # 断言 Y 的形状为 (2, 1)
    assert_equal(Y.shape, (2, 1))
    # 断言 Y 的所有元素近似相等于其均值，容忍度为 atol=tol
    assert_allclose(Y, Y.mean(), atol=tol)

    # 如果 n 大于 5 并且未设置 skip_big=True，则生成一个随机 n 行 5 列的矩阵，并进行后续处理
    if n > 5 and not skip_big:
        np.random.seed(1)
        # 生成随机 n 行 5 列的矩阵，用随机数矩阵乘积增加噪声，转换为指定数据类型 dtype
        X = np.random.rand(n, 5) @ np.random.rand(5, n)
        X = X + 1e-4 * np.random.rand(n, 1) @ np.random.rand(1, n)
        X = X.astype(dtype)

        # 对 X 进行正交化处理，设置奇异值小于 1e-3 的条件
        Y = orth(X, rcond=1e-3)
        # 断言 Y 的形状为 (n, 5)
        assert_equal(Y.shape, (n, 5))

        # 对 X 进行正交化处理，设置奇异值小于 1e-6 的条件
        Y = orth(X, rcond=1e-6)
        # 断言 Y 的形状为 (n, 6)
        assert_equal(Y.shape, (n, 6))


# 使用 pytest 的标记，标记该测试函数为慢速执行，并且只在 64 位系统上运行
@pytest.mark.slow
@pytest.mark.skipif(np.dtype(np.intp).itemsize < 8,
                    reason="test only on 64-bit, else too slow")
# 定义一个测试函数，用于测试正交化函数的内存效率
def test_orth_memory_efficiency():
    # 选择一个使得 16*n 字节合理而 8*n*n 字节不合理的 n 值
    # 注意，@pytest.mark.slow 标记的测试可能在支持 4Gb+ 内存的配置下运行
    n = 10*1000*1000
    try:
        # 调用 _check_orth 函数，传入 n、数据类型为 np.float64，并跳过大数据量处理
        _check_orth(n, np.float64, skip_big=True)
    except MemoryError as e:
        raise AssertionError(
            'memory error perhaps caused by orth regression'
        ) from e


# 定义一个测试函数，用于测试不同数据类型和大小的正交化函数
def test_orth():
    dtypes = [np.float32, np.float64, np.complex64, np.complex128]
    sizes = [1, 2, 3, 10, 100]
    # 遍历数据类型和大小的组合
    for dt, n in itertools.product(dtypes, sizes):
        # 调用 _check_orth 函数，传入 n 和当前数据类型 dt
        _check_orth(n, dt)


# 使用 pytest 的参数化标记，传入不同的数据类型 dt 进行测试
@pytest.mark.parametrize('dt', [int, float, np.float32, complex, np.complex64])
# 定义一个测试函数，用于测试空矩阵的正交化函数
def test_orth_empty(dt):
    # 创建一个空的 0x0 矩阵 a，指定数据类型为 dt
    a = np.empty((0, 0), dtype=dt)
    # 创建一个单位矩阵 a0，数据类型与 a 相同
    a0 = np.eye(2, dtype=dt)

    # 对空矩阵 a 进行正交化处理，返回正交化后的矩阵 oa
    oa = orth(a)
    # 断言 oa 的数据类型与正交化后的 a0 的数据类型相同
    assert oa.dtype == orth(a0).dtype
    # 断言 oa 的形状为 (0, 0)
    assert oa.shape == (0, 0)


# 定义一个测试函数，用于测试零空间函数
def test_null_space():
    np.random.seed(1)

    dtypes = [np.float32, np.float64, np.complex64, np.complex128]
    sizes = [1, 2, 3, 10, 100]

    # 遍历数据类型和大小的组合
    for dt, n in itertools.product(dtypes, sizes):
        # 创建一个全一矩阵 X，形状为 (2, n)，数据类型为 dt
        X = np.ones((2, n), dtype=dt)

        # 获取指定数据类型的机器精度
        eps = np.finfo(dt).eps
        # 设置容忍度为 1000 倍的机器精度
        tol = 1000 * eps

        # 对 X 进行零空间计算，返回零空间矩阵 Y
        Y = null_space(X)
        # 断言 Y 的形状为 (n, n-1)
        assert_equal(Y.shape, (n, n-1))
        # 断言 X @ Y 的所有元素近似等于 0，容忍度为 atol=tol
        assert_allclose(X @ Y, 0, atol=tol)

        # 对 X 的转置进行零空间计算，返回零空间矩阵 Y
        Y = null_space(X.T)
        # 断言 Y 的形状为 (2, 1)
        assert_equal(Y.shape, (2, 1))
        # 断言 X.T @ Y 的所有元素近似等于 0，容忍度为 atol=tol
        assert_allclose(X.T @ Y, 0, atol=tol)

        # 生成一个随机矩阵 X，形状为 (1 + n//2, n)，数据类型为 dt
        X = np.random.randn(1 + n//2, n)
        # 对 X 进行零空间计算，返回零空间矩阵 Y
        Y = null_space(X)
        # 断言 Y 的形状为 (n, n - 1 - n//2)
        assert_equal(Y.shape, (n, n - 1 - n//2))
        # 断言 X @ Y 的所有元素近似等于 0，容忍度为 atol=tol
        assert_allclose(X @ Y, 0, atol=tol)

        # 如果 n 大于 5，则生成一个随机 n 行 5 列的矩阵 X，并进行后续处理
        if n > 5:
            np.random.seed(1)
            # 生成随机 n 行 5 列的矩阵 X，用随机数矩阵乘积增加噪声，转换为指定数据类型 dt
            X = np.random.rand(n, 5) @ np.random.rand(5, n)
            X = X + 1e-4 * np.random.rand(n, 1) @ np.random.rand(1, n)
            X = X.astype(dt)

            # 对 X 进行零空间计算，设置奇
    # 创建一个2x2的单位矩阵，使用给定的数据类型dt
    a0 = np.eye(2, dtype=dt)
    
    # 计算矩阵a的零空间（null space），即其零特征值对应的特征向量构成的空间
    nsa = null_space(a)

    # 断言零空间的形状应为(0, 0)，即零空间中没有非零向量
    assert nsa.shape == (0, 0)
    
    # 断言零空间的数据类型与a0的零空间数据类型相同
    assert nsa.dtype == null_space(a0).dtype
# 定义一个测试函数，用于测试子空间角度计算函数的准确性
def test_subspace_angles():
    # 使用哈达玛矩阵生成8x8的浮点数矩阵H
    H = hadamard(8, float)
    # 取H的前3列作为矩阵A
    A = H[:, :3]
    # 取H的第4列及之后的列作为矩阵B
    B = H[:, 3:]
    # 断言子空间角度函数计算A和B的角度，期望值为π/2，允许误差为1e-14
    assert_allclose(subspace_angles(A, B), [np.pi / 2.] * 3, atol=1e-14)
    # 断言子空间角度函数计算B和A的角度，期望值为π/2，允许误差为1e-14
    assert_allclose(subspace_angles(B, A), [np.pi / 2.] * 3, atol=1e-14)
    # 遍历矩阵A和B，断言子空间角度函数计算每个矩阵自身的角度为0，允许误差为1e-14
    for x in (A, B):
        assert_allclose(subspace_angles(x, x), np.zeros(x.shape[1]),
                        atol=1e-14)
    
    # MATLAB函数"subspace"的测试数据，只返回计算得到的最后一个值
    x = np.array(
        [[0.537667139546100, 0.318765239858981, 3.578396939725760, 0.725404224946106],  # noqa: E501
         [1.833885014595086, -1.307688296305273, 2.769437029884877, -0.063054873189656],  # noqa: E501
         [-2.258846861003648, -0.433592022305684, -1.349886940156521, 0.714742903826096],  # noqa: E501
         [0.862173320368121, 0.342624466538650, 3.034923466331855, -0.204966058299775]])  # noqa: E501
    expected = 1.481454682101605
    # 断言子空间角度函数计算x的第1列和第2列的角度，期望值为expected，相对误差允许为1e-12
    assert_allclose(subspace_angles(x[:, :2], x[:, 2:])[0], expected,
                    rtol=1e-12)
    # 断言子空间角度函数计算x的第3列和第4列的角度，期望值为expected，相对误差允许为1e-12
    assert_allclose(subspace_angles(x[:, 2:], x[:, :2])[0], expected,
                    rtol=1e-12)
    expected = 0.746361174247302
    # 断言子空间角度函数计算x的第1列和第3列的角度，期望值为expected，相对误差允许为1e-12
    assert_allclose(subspace_angles(x[:, :2], x[:, [2]]), expected, rtol=1e-12)
    # 断言子空间角度函数计算x的第3列和第1列的角度，期望值为expected，相对误差允许为1e-12
    assert_allclose(subspace_angles(x[:, [2]], x[:, :2]), expected, rtol=1e-12)
    expected = 0.487163718534313
    # 断言子空间角度函数计算x的前3列和第4列的角度，期望值为expected，相对误差允许为1e-12
    assert_allclose(subspace_angles(x[:, :3], x[:, [3]]), expected, rtol=1e-12)
    # 断言子空间角度函数计算x的第4列和前3列的角度，期望值为expected，相对误差允许为1e-12
    assert_allclose(subspace_angles(x[:, [3]], x[:, :3]), expected, rtol=1e-12)
    expected = 0.328950515907756
    # 断言子空间角度函数计算x的第1列和第2列及之后的列的角度，期望值为[expected, 0]，允许误差为1e-12
    assert_allclose(subspace_angles(x[:, :2], x[:, 1:]), [expected, 0],
                    atol=1e-12)
    
    # 异常情况测试
    # 断言子空间角度函数在异常条件下会引发ValueError异常
    assert_raises(ValueError, subspace_angles, x[0], x)
    assert_raises(ValueError, subspace_angles, x, x[0])
    assert_raises(ValueError, subspace_angles, x[:-1], x)

    # 测试条件分支，当mask.any为True时的情况
    A = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1],
                  [0, 0, 0],
                  [0, 0, 0]])
    B = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 1]])
    expected = np.array([np.pi/2, 0, 0])
    # 断言子空间角度函数计算A和B的角度，期望值为expected，相对误差允许为1e-12
    assert_allclose(subspace_angles(A, B), expected, rtol=1e-12)

    # 复数情况
    # 设置复数矩阵a和b，断言子空间角度函数计算它们的角度为0，允许误差为1e-14
    a = [[1 + 1j], [0]]
    b = [[1 - 1j, 0], [0, 1]]
    assert_allclose(subspace_angles(a, b), 0., atol=1e-14)
    assert_allclose(subspace_angles(b, a), 0., atol=1e-14)

    # 空矩阵情况
    # 设置空的矩阵a和b，断言子空间角度函数计算它们的结果为空矩阵
    a = np.empty((0, 0))
    b = np.empty((0, 0))
    assert_allclose(subspace_angles(a, b), np.empty((0,)))
    a = np.empty((2, 0))
    b = np.empty((2, 0))
    assert_allclose(subspace_angles(a, b), np.empty((0,)))
    a = np.empty((0, 2))
    b = np.empty((0, 3))
    assert_allclose(subspace_angles(a, b), np.empty((0,)))
    # 定义矩阵乘法方法，使用 Einstein 求和约定进行张量乘积计算
    def matmul(self, a, b):
        return np.einsum('...ij,...jk->...ik', a, b)

    # 断言特征值分解结果的有效性：验证 v*w == x*v 成立
    def assert_eig_valid(self, w, v, x):
        assert_array_almost_equal(
            self.matmul(v, w),  # 计算 v*w
            self.matmul(x, v)   # 计算 x*v
        )

    # 测试对于 0x0 大小的实数组，使用 cdf2rdf 转换后的特征值和特征向量是否满足特征值分解的有效性
    def test_single_array0x0real(self):
        # eig 在旧版本的 numpy 中不支持 0x0 大小的数组
        X = np.empty((0, 0))
        w, v = np.empty(0), np.empty((0, 0))
        wr, vr = cdf2rdf(w, v)
        self.assert_eig_valid(wr, vr, X)

    # 测试对于 2x2 实数组，使用 cdf2rdf 转换后的特征值和特征向量是否满足特征值分解的有效性
    def test_single_array2x2_real(self):
        X = np.array([[1, 2], [3, -1]])
        w, v = np.linalg.eig(X)
        wr, vr = cdf2rdf(w, v)
        self.assert_eig_valid(wr, vr, X)

    # 测试对于 2x2 复数数组，使用 cdf2rdf 转换后的特征值和特征向量是否满足特征值分解的有效性
    def test_single_array2x2_complex(self):
        X = np.array([[1, 2], [-2, 1]])
        w, v = np.linalg.eig(X)
        wr, vr = cdf2rdf(w, v)
        self.assert_eig_valid(wr, vr, X)

    # 测试对于 3x3 实数组，使用 cdf2rdf 转换后的特征值和特征向量是否满足特征值分解的有效性
    def test_single_array3x3_real(self):
        X = np.array([[1, 2, 3], [1, 2, 3], [2, 5, 6]])
        w, v = np.linalg.eig(X)
        wr, vr = cdf2rdf(w, v)
        self.assert_eig_valid(wr, vr, X)

    # 测试对于 3x3 复数数组，使用 cdf2rdf 转换后的特征值和特征向量是否满足特征值分解的有效性
    def test_single_array3x3_complex(self):
        X = np.array([[1, 2, 3], [0, 4, 5], [0, -5, 4]])
        w, v = np.linalg.eig(X)
        wr, vr = cdf2rdf(w, v)
        self.assert_eig_valid(wr, vr, X)

    # 测试随机生成的 1 维堆叠数组，验证其特征值和特征向量是否满足特征值分解的有效性
    def test_random_1d_stacked_arrays(self):
        # 由于旧版本的 numpy 存在 bug，无法测试 M == 0 的情况
        for M in range(1, 7):
            np.random.seed(999999999)
            X = np.random.rand(100, M, M)
            w, v = np.linalg.eig(X)
            wr, vr = cdf2rdf(w, v)
            self.assert_eig_valid(wr, vr, X)

    # 测试随机生成的 2 维堆叠数组，验证其特征值和特征向量是否满足特征值分解的有效性
    def test_random_2d_stacked_arrays(self):
        # 由于旧版本的 numpy 存在 bug，无法测试 M == 0 的情况
        for M in range(1, 7):
            X = np.random.rand(10, 10, M, M)
            w, v = np.linalg.eig(X)
            wr, vr = cdf2rdf(w, v)
            self.assert_eig_valid(wr, vr, X)

    # 测试低维度情况下的错误：w 是空数组，v 是 1 维非空数组，预期引发 ValueError
    def test_low_dimensionality_error(self):
        w, v = np.empty(()), np.array((2,))
        assert_raises(ValueError, cdf2rdf, w, v)

    # 测试非方阵情况下的错误：传入非方阵数组，预期引发 ValueError
    def test_not_square_error(self):
        w, v = np.arange(3), np.arange(6).reshape(3, 2)
        assert_raises(ValueError, cdf2rdf, w, v)

    # 测试交换 v 和 w 引发的错误：传入 v 和 w 的顺序颠倒，预期引发 ValueError
    def test_swapped_v_w_error(self):
        X = np.array([[1, 2, 3], [0, 4, 5], [0, -5, 4]])
        w, v = np.linalg.eig(X)
        assert_raises(ValueError, cdf2rdf, v, w)

    # 测试非关联特征向量引发的错误：传入非关联特征向量，预期引发 ValueError
    def test_non_associated_error(self):
        w, v = np.arange(3), np.arange(16).reshape(4, 4)
        assert_raises(ValueError, cdf2rdf, w, v)
    def test_not_conjugate_pairs(self):
        # 检查传递非共轭对是否引发 ValueError 异常。
        X = np.array([[1, 2, 3], [1, 2, 3], [2, 5, 6+1j]])
        # 计算矩阵 X 的特征值和特征向量
        w, v = np.linalg.eig(X)
        # 断言调用 cdf2rdf 函数使用 w 和 v 会引发 ValueError 异常
        assert_raises(ValueError, cdf2rdf, w, v)

        # 不同的数组堆栈，因此它们不是共轭的
        X = np.array([
            [[1, 2, 3], [1, 2, 3], [2, 5, 6+1j]],
            [[1, 2, 3], [1, 2, 3], [2, 5, 6-1j]],
        ])
        # 计算张量 X 的特征值和特征向量
        w, v = np.linalg.eig(X)
        # 断言调用 cdf2rdf 函数使用 w 和 v 会引发 ValueError 异常
        assert_raises(ValueError, cdf2rdf, w, v)
```