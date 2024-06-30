# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_eigen\lobpcg\tests\test_lobpcg.py`

```
""" Test functions for the sparse.linalg._eigen.lobpcg module
"""

# 导入必要的模块和库
import itertools  # 导入 itertools 模块
import platform   # 导入 platform 模块
import sys        # 导入 sys 模块
import pytest     # 导入 pytest 模块
import numpy as np  # 导入 numpy 库，并使用 np 别名
from numpy import ones, r_, diag  # 从 numpy 中导入 ones, r_, diag 函数
from numpy.testing import (assert_almost_equal, assert_equal,
                           assert_allclose, assert_array_less)  # 从 numpy.testing 导入断言函数

from scipy import sparse  # 导入 scipy 库中的 sparse 模块
from scipy.linalg import eig, eigh, toeplitz, orth  # 导入 scipy.linalg 库中的函数
from scipy.sparse import spdiags, diags, eye, csr_matrix  # 导入 scipy.sparse 库中的函数
from scipy.sparse.linalg import eigs, LinearOperator  # 导入 scipy.sparse.linalg 库中的函数
from scipy.sparse.linalg._eigen.lobpcg import lobpcg  # 导入 lobpcg 函数
from scipy.sparse.linalg._eigen.lobpcg.lobpcg import _b_orthonormalize  # 导入 _b_orthonormalize 函数
from scipy._lib._util import np_long, np_ulong  # 导入 scipy._lib._util 库中的数据类型

_IS_32BIT = (sys.maxsize < 2**32)  # 检测系统是否为 32 位

INT_DTYPES = {np.intc, np_long, np.longlong, np.uintc, np_ulong, np.ulonglong}  # 整数数据类型集合
# np.half is unsupported on many test systems so excluded
REAL_DTYPES = {np.float32, np.float64, np.longdouble}  # 实数数据类型集合
COMPLEX_DTYPES = {np.complex64, np.complex128, np.clongdouble}  # 复数数据类型集合
# 使用排序列表确保测试顺序固定
VDTYPES = sorted(REAL_DTYPES ^ COMPLEX_DTYPES, key=str)  # 混合实数类型集合
MDTYPES = sorted(INT_DTYPES ^ REAL_DTYPES ^ COMPLEX_DTYPES, key=str)  # 混合数据类型集合


def sign_align(A, B):
    """Align signs of columns of A match those of B: column-wise remove
    sign of A by multiplying with its sign then multiply in sign of B.
    """
    # 对矩阵 A 的每一列，调整其符号使其与矩阵 B 的对应列符号一致
    return np.array([col_A * np.sign(col_A[0]) * np.sign(col_B[0])
                     for col_A, col_B in zip(A.T, B.T)]).T


def ElasticRod(n):
    """Build the matrices for the generalized eigenvalue problem of the
    fixed-free elastic rod vibration model.
    """
    L = 1.0  # 弹性杆长度
    le = L/n  # 每段长度
    rho = 7.85e3  # 密度
    S = 1.e-4  # 横截面积
    E = 2.1e11  # 弹性模量
    mass = rho*S*le/6.  # 质量
    k = E*S/le  # 刚度系数
    # 构建广义特征值问题的系数矩阵 A 和 B
    A = k*(diag(r_[2.*ones(n-1), 1])-diag(ones(n-1), 1)-diag(ones(n-1), -1))
    B = mass*(diag(r_[4.*ones(n-1), 2])+diag(ones(n-1), 1)+diag(ones(n-1), -1))
    return A, B  # 返回系数矩阵 A 和 B


def MikotaPair(n):
    """Build a pair of full diagonal matrices for the generalized eigenvalue
    problem. The Mikota pair acts as a nice test since the eigenvalues are the
    squares of the integers n, n=1,2,...
    """
    x = np.arange(1, n+1)  # 构建向量 x
    B = diag(1./x)  # 构建对角矩阵 B
    y = np.arange(n-1, 0, -1)  # 构建向量 y
    z = np.arange(2*n-1, 0, -2)  # 构建向量 z
    A = diag(z)-diag(y, -1)-diag(y, 1)  # 构建对角矩阵 A
    return A, B  # 返回对角矩阵 A 和 B


def compare_solutions(A, B, m):
    """Check eig vs. lobpcg consistency.
    """
    n = A.shape[0]  # 获取矩阵 A 的大小
    rnd = np.random.RandomState(0)  # 使用种子 0 创建随机状态
    V = rnd.random((n, m))  # 生成随机矩阵 V
    X = orth(V)  # 对 V 进行正交化得到矩阵 X
    eigvals, _ = lobpcg(A, X, B=B, tol=1e-2, maxiter=50, largest=False)  # 使用 lobpcg 求解广义特征值问题
    eigvals.sort()  # 对求解得到的特征值排序
    w, _ = eig(A, b=B)  # 使用 eig 函数求解广义特征值问题
    w.sort()  # 对求解得到的特征值排序
    assert_almost_equal(w[:int(m/2)], eigvals[:int(m/2)], decimal=2)  # 断言两种方法求解得到的前一半特征值近似相等


def test_Small():
    """Test case for small problems using ElasticRod and MikotaPair.
    """
    A, B = ElasticRod(10)  # 构建弹性杆问题的系数矩阵
    with pytest.warns(UserWarning, match="The problem size"):
        compare_solutions(A, B, 10)  # 检查使用不同方法求解结果的一致性
    A, B = MikotaPair(10)  # 构建 Mikota 对问题的系数矩阵
    with pytest.warns(UserWarning, match="The problem size"):
        compare_solutions(A, B, 10)  # 检查使用不同方法求解结果的一致性


def test_ElasticRod():
    """Test case for ElasticRod problem with n=20.
    """
    A, B = ElasticRod(20)  # 构建弹性杆问题的系数矩阵
    # 定义一个正则表达式模式，用于匹配两种可能的警告消息
    msg = "Exited at iteration.*|Exited postprocessing with accuracies.*"
    
    # 使用 pytest 模块的 warn 函数来捕获特定类型的警告，并检查其匹配给定的正则表达式模式
    # 如果捕获到符合模式的警告消息，则测试通过；否则，测试失败
    with pytest.warns(UserWarning, match=msg):
        # 调用 compare_solutions 函数，传递参数 A, B, 2，并期望捕获特定警告
        compare_solutions(A, B, 2)
# 定义名为 test_MikotaPair 的测试函数
def test_MikotaPair():
    # 调用 MikotaPair 函数，返回元组 A, B，其中 A 和 B 是 MikotaPair 的结果
    A, B = MikotaPair(20)
    # 调用 compare_solutions 函数，比较 A 和 B 的解决方案，期望精度为 2
    compare_solutions(A, B, 2)


# 使用 pytest 的 parametrize 装饰器定义参数化测试
@pytest.mark.parametrize("n", [50])
@pytest.mark.parametrize("m", [1, 2, 10])
@pytest.mark.parametrize("Vdtype", sorted(REAL_DTYPES, key=str))
@pytest.mark.parametrize("Bdtype", sorted(REAL_DTYPES, key=str))
@pytest.mark.parametrize("BVdtype", sorted(REAL_DTYPES, key=str))
def test_b_orthonormalize(n, m, Vdtype, Bdtype, BVdtype):
    """测试通过可调用的 'B' 实现 B-正交化。
    函数 '_b_orthonormalize' 在 LOBPCG 中非常关键，但可能导致数值不稳定性。
    输入向量通常缩放不良，因此函数需要具有尺度不变性的 Cholesky 分解；
    参见 https://netlib.org/lapack/lawnspdf/lawn14.pdf。
    """
    # 使用随机种子 0 初始化随机数生成器
    rnd = np.random.RandomState(0)
    # 创建大小为 (n, m) 的标准正态分布随机数组，类型转换为 Vdtype
    X = rnd.standard_normal((n, m)).astype(Vdtype)
    # 复制 X，类型转换为 Bdtype，构造对角矩阵 B
    Xcopy = np.copy(X)
    vals = np.arange(1, n+1, dtype=float)
    B = diags([vals], [0], (n, n)).astype(Bdtype)
    # 计算 B @ X，结果类型转换为 BVdtype
    BX = B @ X
    BX = BX.astype(BVdtype)
    # 选择 X、B、BX 三者中的最小数据类型
    dtype = min(X.dtype, B.dtype, BX.dtype)
    # 计算公差 atol，以 m * n 乘以当前数据类型的机器精度或 np.float64 的机器精度为最大值
    atol = m * n * max(np.finfo(dtype).eps, np.finfo(np.float64).eps)

    # 调用 _b_orthonormalize 函数，对 X 和 BX 进行 B-正交化操作，返回结果 Xo, BXo 和辅助信息
    Xo, BXo, _ = _b_orthonormalize(lambda v: B @ v, X, BX)
    # 检查 Xo 是否与 X 相等
    assert_equal(X, Xo)
    # 检查 Xo 和 X 是否为同一对象
    assert_equal(id(X), id(Xo))
    # 检查 BXo 是否与 BX 相等
    assert_equal(BX, BXo)
    # 检查 BXo 和 BX 是否为同一对象
    assert_equal(id(BX), id(BXo))
    # 检查 BXo 是否满足 B-正交性
    assert_allclose(B @ Xo, BXo, atol=atol, rtol=atol)
    # 检查 Xo 是否满足 B-正交性
    assert_allclose(Xo.T.conj() @ B @ Xo, np.identity(m),
                    atol=atol, rtol=atol)

    # 复制 X，重新执行 _b_orthonormalize 函数，返回结果 Xo1, BXo1 和辅助信息
    X = np.copy(Xcopy)
    Xo1, BXo1, _ = _b_orthonormalize(lambda v: B @ v, X)
    # 检查 Xo 和 Xo1 是否近似相等
    assert_allclose(Xo, Xo1, atol=atol, rtol=atol)
    # 检查 BXo 和 BXo1 是否近似相等
    assert_allclose(BXo, BXo1, atol=atol, rtol=atol)
    # 检查 Xo1 是否与 X 相等
    assert_equal(X, Xo1)
    # 检查 Xo1 和 X 是否为同一对象
    assert_equal(id(X), id(Xo1))
    # 检查 BXo1 是否满足 B-正交性
    assert_allclose(B @ Xo1, BXo1, atol=atol, rtol=atol)

    # 对 X 进行列缩放
    scaling = 1.0 / np.geomspace(10, 1e10, num=m)
    X = Xcopy * scaling
    X = X.astype(Vdtype)
    BX = B @ X
    BX = BX.astype(BVdtype)
    # 重新执行 _b_orthonormalize 函数，返回结果 Xo1, BXo1 和辅助信息
    Xo1, BXo1, _ = _b_orthonormalize(lambda v: B @ v, X, BX)
    # 检查 Xo1 是否与 Xo 近似相等，考虑数值公差
    assert_allclose(Xo, Xo1, atol=atol, rtol=atol)
    # 检查 BXo1 是否与 BXo 近似相等，考虑数值公差
    assert_allclose(BXo, BXo1, atol=atol, rtol=atol)


# 使用 pytest 的 filterwarnings 装饰器忽略特定警告信息
@pytest.mark.filterwarnings("ignore:Exited at iteration 0")
@pytest.mark.filterwarnings("ignore:Exited postprocessing")
def test_nonhermitian_warning(capsys):
    """检查非厄米矩阵警告，通过输入非厄米矩阵触发警告。
    同时检查标准输出，因为 verbosityLevel=1 并且没有标准错误输出。
    """
    # 设置矩阵维度为 10，创建类型为 np.float32 的二维数组 X 和 A
    n = 10
    X = np.arange(n * 2).reshape(n, 2).astype(np.float32)
    A = np.arange(n * n).reshape(n, n).astype(np.float32)
    # 检查是否在运行lobpcg时产生了UserWarning，并且警告信息包含"Matrix gramA"
    with pytest.warns(UserWarning, match="Matrix gramA"):
        # 调用lobpcg函数，但设置最大迭代次数为0，并捕获其返回值
        _, _ = lobpcg(A, X, verbosityLevel=1, maxiter=0)
    
    # 捕获当前标准输出和标准错误输出
    out, err = capsys.readouterr()
    
    # 断言标准输出以指定字符串"Solving standard eigenvalue"开头
    assert out.startswith("Solving standard eigenvalue")
    
    # 断言标准错误输出为空
    assert err == ''
    
    # 使矩阵A变为对称矩阵，以消除UserWarning警告
    A += A.T
    
    # 重新运行lobpcg函数，捕获其输出
    _, _ = lobpcg(A, X, verbosityLevel=1, maxiter=0)
    
    # 捕获更新后的标准输出和标准错误输出
    out, err = capsys.readouterr()
    
    # 再次断言标准输出以指定字符串"Solving standard eigenvalue"开头
    assert out.startswith("Solving standard eigenvalue")
    
    # 再次断言标准错误输出为空
    assert err == ''
def test_regression():
    """Check the eigenvalue of the identity matrix is one.
    """
    # 设置矩阵大小为10
    n = 10
    # 创建一个 n x 1 的全为 1 的数组
    X = np.ones((n, 1))
    # 创建一个 n x n 的单位矩阵
    A = np.identity(n)
    # 使用 lobpcg 方法计算特征值 w，忽略特征向量
    w, _ = lobpcg(A, X)
    # 断言特征值 w 的值接近于 [1]
    assert_allclose(w, [1])


@pytest.mark.filterwarnings("ignore:The problem size")
@pytest.mark.parametrize('n, m, m_excluded', [(30, 4, 3), (4, 2, 0)])
def test_diagonal(n, m, m_excluded):
    """Test ``m - m_excluded`` eigenvalues and eigenvectors of
    diagonal matrices of the size ``n`` varying matrix formats:
    dense array, sparse matrix, and ``LinearOperator`` for both
    matrices in the generalized eigenvalue problem ``Av = cBv``
    and for the preconditioner.
    """
    rnd = np.random.RandomState(0)

    # 定义对角矩阵 A 和 B，以及线性操作 LinearOperator M
    # A 是对角线上元素为 1 到 n 的对角矩阵
    vals = np.arange(1, n+1, dtype=float)
    A_s = diags([vals], [0], (n, n))
    A_a = A_s.toarray()

    def A_f(x):
        return A_s @ x

    A_lo = LinearOperator(matvec=A_f,
                          matmat=A_f,
                          shape=(n, n), dtype=float)

    # B 是单位矩阵
    B_a = eye(n)
    B_s = csr_matrix(B_a)

    def B_f(x):
        return B_a @ x

    B_lo = LinearOperator(matvec=B_f,
                          matmat=B_f,
                          shape=(n, n), dtype=float)

    # M 是 A 的逆矩阵作为预处理器
    M_s = diags([1./vals], [0], (n, n))
    M_a = M_s.toarray()

    def M_f(x):
        return M_s @ x

    M_lo = LinearOperator(matvec=M_f,
                          matmat=M_f,
                          shape=(n, n), dtype=float)

    # 随机生成初始向量 X
    X = rnd.normal(size=(n, m))

    # 要求返回的特征向量在前几个标准基向量的正交补空间中
    if m_excluded > 0:
        Y = np.eye(n, m_excluded)
    else:
        Y = None

    # 对每种 A, B, M 的组合进行 lobpcg 方法求解广义特征值问题
    for A in [A_a, A_s, A_lo]:
        for B in [B_a, B_s, B_lo]:
            for M in [M_a, M_s, M_lo]:
                eigvals, vecs = lobpcg(A, X, B, M=M, Y=Y,
                                       maxiter=40, largest=False)

                # 断言返回的特征值接近于 [1+m_excluded, 2+m_excluded, ..., m+m_excluded]
                assert_allclose(eigvals, np.arange(1+m_excluded,
                                                   1+m_excluded+m))
                # 检查特征值和特征向量的正确性
                _check_eigen(A, eigvals, vecs, rtol=1e-3, atol=1e-3)


def _check_eigen(M, w, V, rtol=1e-8, atol=1e-14):
    """Check if the eigenvalue residual is small.
    """
    # 计算特征向量和特征值的乘积和矩阵 M 乘以特征向量的结果
    mult_wV = np.multiply(w, V)
    dot_MV = M.dot(V)
    # 断言乘积结果接近于矩阵乘积的结果
    assert_allclose(mult_wV, dot_MV, rtol=rtol, atol=atol)


def _check_fiedler(n, p):
    """Check the Fiedler vector computation.
    """
    # 这不一定是计算 Fiedler 向量的推荐方法
    col = np.zeros(n)
    col[1] = 1
    # 构造对称矩阵 A
    A = toeplitz(col)
    D = np.diag(A.sum(axis=1))
    L = D - A
    # 使用一些技巧计算完整的特征分解
    # 根据论文提供的方法构造一个长度为 n 的等间距的角度数组
    tmp = np.pi * np.arange(n) / n
    # 使用解析方法计算对应的特征值
    analytic_w = 2 * (1 - np.cos(tmp))
    # 使用解析方法计算对应的特征向量矩阵
    analytic_V = np.cos(np.outer(np.arange(n) + 1/2, tmp))
    # 检查解析方法计算的特征值和特征向量是否与给定的 L 矩阵的特征对匹配
    _check_eigen(L, analytic_w, analytic_V)
    
    # 使用 eigh 方法计算 L 矩阵的全部特征值和特征向量
    eigh_w, eigh_V = eigh(L)
    # 检查 eigh 方法计算的特征值和特征向量是否与给定的 L 矩阵的特征对匹配
    _check_eigen(L, eigh_w, eigh_V)
    # 检查第一个特征值是否接近零，并且其余特征值是否一致
    assert_array_less(np.abs([eigh_w[0], analytic_w[0]]), 1e-14)
    assert_allclose(eigh_w[1:], analytic_w[1:])
    
    # 使用 lobpcg 方法计算 L 矩阵的部分特征值和对应的特征向量（最小的 p 个特征值）
    X = analytic_V[:, :p]
    lobpcg_w, lobpcg_V = lobpcg(L, X, largest=False)
    # 检查 lobpcg 方法计算的特征值数组的形状是否为 (p,)
    assert_equal(lobpcg_w.shape, (p,))
    # 检查 lobpcg 方法计算的特征向量数组的形状是否为 (n, p)
    assert_equal(lobpcg_V.shape, (n, p))
    # 检查 lobpcg 方法计算的特征值和特征向量是否与给定的 L 矩阵的特征对匹配
    _check_eigen(L, lobpcg_w, lobpcg_V)
    # 检查 lobpcg 方法计算的最小特征值是否接近零
    assert_array_less(np.abs(np.min(lobpcg_w)), 1e-14)
    # 检查 lobpcg 方法计算的除第一个特征值外其余特征值是否与解析方法计算的一致
    assert_allclose(np.sort(lobpcg_w)[1:], analytic_w[1:p])
    
    # 使用 lobpcg 方法计算 L 矩阵的部分特征值和对应的特征向量（最大的 p 个特征值）
    X = analytic_V[:, -p:]
    lobpcg_w, lobpcg_V = lobpcg(L, X, largest=True)
    # 检查 lobpcg 方法计算的特征值数组的形状是否为 (p,)
    assert_equal(lobpcg_w.shape, (p,))
    # 检查 lobpcg 方法计算的特征向量数组的形状是否为 (n, p)
    assert_equal(lobpcg_V.shape, (n, p))
    # 检查 lobpcg 方法计算的特征值和特征向量是否与给定的 L 矩阵的特征对匹配
    _check_eigen(L, lobpcg_w, lobpcg_V)
    # 检查 lobpcg 方法计算的最大特征值是否与解析方法计算的最大特征值一致
    assert_allclose(np.sort(lobpcg_w), analytic_w[-p:])
    
    # 使用 lobpcg 方法寻找 Fiedler 向量，使用略有不同的初始猜测
    fiedler_guess = np.concatenate((np.ones(n//2), -np.ones(n-n//2)))
    X = np.vstack((np.ones(n), fiedler_guess)).T
    lobpcg_w, _ = lobpcg(L, X, largest=False)
    # 数学上，较小的特征值应该接近零，较大的特征值应该等于代数连通性
    lobpcg_w = np.sort(lobpcg_w)
    # 检查 lobpcg 方法计算的特征值是否与解析方法计算的前两个特征值一致
    assert_allclose(lobpcg_w, analytic_w[:2], atol=1e-14)
# 测试小规模矩阵的费德勒检查，触发了稠密路径
def test_fiedler_small_8():
    """Check the dense workaround path for small matrices.
    """
    # 这里触发了稠密路径，因为 8 < 2*5。
    with pytest.warns(UserWarning, match="The problem size"):
        _check_fiedler(8, 2)


# 测试大规模矩阵的费德勒检查，避免了稠密路径
def test_fiedler_large_12():
    """Check the dense workaround path avoided for non-small matrices.
    """
    # 这里不会触发稠密路径，因为 2*5 <= 12。
    _check_fiedler(12, 2)


# 测试迭代运行失败情况
def test_failure_to_run_iterations():
    """Check that the code exits gracefully without breaking. Issue #10974.
    The code may or not issue a warning, filtered out. Issue #15935, #17954.
    """
    # 使用随机数种子创建随机数生成器
    rnd = np.random.RandomState(0)
    # 生成一个随机正态分布的矩阵 X
    X = rnd.standard_normal((100, 10))
    # 计算 X 的转置乘积，构成对称矩阵 A
    A = X @ X.T
    # 创建一个随机正态分布的矩阵 Q
    Q = rnd.standard_normal((X.shape[0], 4))
    # 使用 lobpcg 方法求解 A 的特征值，最大迭代次数为 40，容差为 1e-12
    eigenvalues, _ = lobpcg(A, Q, maxiter=40, tol=1e-12)
    # 断言特征值的最大值大于 0
    assert np.max(eigenvalues) > 0


# 测试迭代运行失败情况（非对称矩阵）
def test_failure_to_run_iterations_nonsymmetric():
    """Check that the code exists gracefully without breaking
    if the matrix in not symmetric.
    """
    # 创建一个全零的矩阵 A
    A = np.zeros((10, 10))
    # 将 A 的第一行第二列设为 1，使其成为非对称矩阵
    A[0, 1] = 1
    # 创建一个全为 1 的矩阵 Q
    Q = np.ones((10, 1))
    # 设置匹配字符串，用于过滤警告信息
    msg = "Exited at iteration 2|Exited postprocessing with accuracies.*"
    # 断言 lobpcg 方法调用时会产生 UserWarning，且警告信息匹配 msg
    with pytest.warns(UserWarning, match=msg):
        eigenvalues, _ = lobpcg(A, Q, maxiter=20)
    # 断言特征值的最大值大于 0
    assert np.max(eigenvalues) > 0


# 测试复数埃尔米特矩阵情况
@pytest.mark.filterwarnings("ignore:The problem size")
def test_hermitian():
    """Check complex-value Hermitian cases.
    """
    # 使用随机数种子创建随机数生成器
    rnd = np.random.RandomState(0)

    # 定义不同的大小、k值和生成器选项
    sizes = [3, 12]
    ks = [1, 2]
    gens = [True, False]

    # 遍历所有大小、k值和生成器选项的组合
    for s, k, gen, dh, dx, db in (
        itertools.product(sizes, ks, gens, gens, gens, gens)
        H = rnd.random((s, s)) + 1.j * rnd.random((s, s))
        # 生成一个随机的复数 Hermitian 矩阵 H
        H = 10 * np.eye(s) + H + H.T.conj()
        # 将 H 转换为复数类型，如果 dh 为真则转换为 np.complex128，否则转换为 np.complex64
        H = H.astype(np.complex128) if dh else H.astype(np.complex64)

        X = rnd.standard_normal((s, k))
        # 生成一个随机的复数矩阵 X
        X = X + 1.j * rnd.standard_normal((s, k))
        # 将 X 转换为复数类型，如果 dx 为真则转换为 np.complex128，否则转换为 np.complex64
        X = X.astype(np.complex128) if dx else X.astype(np.complex64)

        if not gen:
            B = np.eye(s)
            # 使用 lobpcg 函数计算 H 和 X 的特征值和特征向量，最大迭代次数为 99，关闭详细输出
            w, v = lobpcg(H, X, maxiter=99, verbosityLevel=0)
            # 测试复数 H 和实数 B 混合的情况
            wb, _ = lobpcg(H, X, B, maxiter=99, verbosityLevel=0)
            # 断言 w 和 wb 的特征值数组非常接近，相对误差小于 1e-6
            assert_allclose(w, wb, rtol=1e-6)
            # 计算 H 的实部特征值和特征向量
            w0, _ = eigh(H)
        else:
            B = rnd.random((s, s)) + 1.j * rnd.random((s, s))
            B = 10 * np.eye(s) + B.dot(B.T.conj())
            # 将 B 转换为复数类型，如果 db 为真则转换为 np.complex128，否则转换为 np.complex64
            B = B.astype(np.complex128) if db else B.astype(np.complex64)
            # 使用 lobpcg 函数计算 H、X 和 B 的特征值和特征向量，最大迭代次数为 99，关闭详细输出
            w, v = lobpcg(H, X, B, maxiter=99, verbosityLevel=0)
            # 计算 H、B 的特征值和特征向量
            w0, _ = eigh(H, B)

        for wx, vx in zip(w, v.T):
            # 检查特征向量是否正确
            assert_allclose(np.linalg.norm(H.dot(vx) - B.dot(vx) * wx)
                            / np.linalg.norm(H.dot(vx)),
                            0, atol=5e-2, rtol=0)
            # 比较特征值是否一致
            j = np.argmin(abs(w0 - wx))
            assert_allclose(wx, w0[j], rtol=1e-4)
# 测试 lobpcg 函数对于不同问题大小和精度要求的一致性
# n=5 的情况测试使用 eigh() 的小矩阵代码路径
@pytest.mark.filterwarnings("ignore:The problem size")
@pytest.mark.parametrize('n, atol', [(20, 1e-3), (5, 1e-8)])
def test_eigs_consistency(n, atol):
    """Check eigs vs. lobpcg consistency.
    """
    # 创建一个对角矩阵 A，对角线元素从 1 到 n
    vals = np.arange(1, n+1, dtype=np.float64)
    A = spdiags(vals, 0, n, n)
    # 创建一个随机矩阵 X
    rnd = np.random.RandomState(0)
    X = rnd.standard_normal((n, 2))
    # 使用 lobpcg 求解最大的两个特征值和特征向量
    lvals, lvecs = lobpcg(A, X, largest=True, maxiter=100)
    # 使用 eigs 函数求解矩阵 A 的两个特征值
    vals, _ = eigs(A, k=2)

    # 检查 lobpcg 求得的特征值和特征向量的正确性
    _check_eigen(A, lvals, lvecs, atol=atol, rtol=0)
    # 检查排序后的特征值是否一致
    assert_allclose(np.sort(vals), np.sort(lvals), atol=1e-14)


# 测试非零详细级别代码是否正常运行
def test_verbosity():
    """Check that nonzero verbosity level code runs.
    """
    rnd = np.random.RandomState(0)
    X = rnd.standard_normal((10, 10))
    A = X @ X.T
    Q = rnd.standard_normal((X.shape[0], 1))
    msg = "Exited at iteration.*|Exited postprocessing with accuracies.*"
    # 检查 lobpcg 函数在详细级别为 9 时是否会引发 UserWarning，并匹配给定的正则表达式消息
    with pytest.warns(UserWarning, match=msg):
        _, _ = lobpcg(A, Q, maxiter=3, verbosityLevel=9)


# 测试在 float32 精度下 lobpcg 函数是否能达到可接受的容差
@pytest.mark.xfail(_IS_32BIT and sys.platform == 'win32',
                   reason="tolerance violation on windows")
@pytest.mark.xfail(platform.machine() == 'ppc64le',
                   reason="fails on ppc64le")
@pytest.mark.filterwarnings("ignore:Exited postprocessing")
def test_tolerance_float32():
    """Check lobpcg for attainable tolerance in float32.
    """
    rnd = np.random.RandomState(0)
    n = 50
    m = 3
    vals = -np.arange(1, n + 1)
    # 创建一个对角矩阵 A，对角线元素从 -1 到 -n
    A = diags([vals], [0], (n, n))
    A = A.astype(np.float32)
    # 创建一个随机矩阵 X
    X = rnd.standard_normal((n, m))
    X = X.astype(np.float32)
    # 使用 lobpcg 求解 A 的前 3 个特征值和特征向量，设置容差和迭代次数上限
    eigvals, _ = lobpcg(A, X, tol=1.25e-5, maxiter=50, verbosityLevel=0)
    # 检查 lobpcg 求得的特征值是否接近预期值 -1 到 -1-m
    assert_allclose(eigvals, -np.arange(1, 1 + m), atol=2e-5, rtol=1e-5)


# 测试 lobpcg 函数在不同数据类型下的行为
@pytest.mark.parametrize("vdtype", VDTYPES)
@pytest.mark.parametrize("mdtype", MDTYPES)
@pytest.mark.parametrize("arr_type", [np.array,
                                      sparse.csr_matrix,
                                      sparse.coo_matrix])
def test_dtypes(vdtype, mdtype, arr_type):
    """Test lobpcg in various dtypes.
    """
    rnd = np.random.RandomState(0)
    n = 12
    m = 2
    # 创建一个对角矩阵 A，对角线元素从 1 到 n
    A = arr_type(np.diag(np.arange(1, n + 1)).astype(mdtype))
    # 创建一个随机矩阵 X，元素类型为 vdtype
    X = rnd.random((n, m))
    X = X.astype(vdtype)
    # 使用 lobpcg 求解 A 的前 2 个特征值和特征向量，设置容差和最大迭代次数
    eigvals, eigvecs = lobpcg(A, X, tol=1e-2, largest=False)
    # 检查 lobpcg 求得的特征值是否接近预期值 1 到 1+m
    assert_allclose(eigvals, np.arange(1, 1 + m), atol=1e-1)
    # 检查特征向量是否几乎是实数
    assert_allclose(np.sum(np.abs(eigvecs - eigvecs.conj())), 0, atol=1e-2)


# 测试 lobpcg 函数在 _b_orthonormalize 函数中对于不支持就地操作的警告是否正常
@pytest.mark.filterwarnings("ignore:Exited at iteration")
@pytest.mark.filterwarnings("ignore:Exited postprocessing")
def test_inplace_warning():
    """Check lobpcg gives a warning in '_b_orthonormalize'
    that in-place orthogonalization is impossible due to dtype mismatch.
    """
    rnd = np.random.RandomState(0)
    n = 6
    m = 1
    vals = -np.arange(1, n + 1)
    # 创建一个对角矩阵 A，对角线元素从 -1 到 -n，元素类型为复数
    A = diags([vals], [0], (n, n))
    A = A.astype(np.cdouble)
    # 生成一个 n 行 m 列的标准正态分布随机矩阵 X
    X = rnd.standard_normal((n, m))
    # 使用 pytest 模块监测警告信息，当匹配到 "Inplace update" 的警告时进行处理
    with pytest.warns(UserWarning, match="Inplace update"):
        # 使用 lobpcg 函数进行特征值计算，其中 A 是输入的稀疏矩阵，X 是初始的随机矩阵，
        # maxiter=2 表示最大迭代次数为 2，verbosityLevel=1 表示设置详细程度为 1
        eigvals, _ = lobpcg(A, X, maxiter=2, verbosityLevel=1)
# 定义一个测试函数，用于验证 lobpcg 算法在不同参数下的行为。
def test_maxit():
    """Check lobpcg if maxit=maxiter runs maxiter iterations and
    if maxit=None runs 20 iterations (the default)
    by checking the size of the iteration history output, which should
    be the number of iterations plus 3 (initial, final, and postprocessing)
    typically when maxiter is small and the choice of the best is passive.
    """
    # 使用固定的随机种子创建随机数生成器
    rnd = np.random.RandomState(0)
    # 设置问题的维度和特征向量数量
    n = 50
    m = 4
    # 创建一个负对角线的对角矩阵 A
    vals = -np.arange(1, n + 1)
    A = diags([vals], [0], (n, n))
    # 将矩阵 A 转换为 np.float32 类型
    A = A.astype(np.float32)
    # 使用随机数生成器创建 n 行 m 列的矩阵 X
    X = rnd.standard_normal((n, m))
    # 将矩阵 X 转换为 np.float64 类型
    X = X.astype(np.float64)
    # 正则表达式模式，用于匹配 lobpcg 函数的警告信息
    msg = "Exited at iteration.*|Exited postprocessing with accuracies.*"
    
    # 循环测试不同的 maxiter 值
    for maxiter in range(1, 4):
        # 在 pytest 的警告上下文中运行 lobpcg 函数，并获取返回结果
        with pytest.warns(UserWarning, match=msg):
            _, _, l_h, r_h = lobpcg(A, X, tol=1e-8, maxiter=maxiter,
                                    retLambdaHistory=True,
                                    retResidualNormsHistory=True)
        # 断言历史记录数组的长度是否符合预期（maxiter + 初始 + 最终 + 后处理）
        assert_allclose(np.shape(l_h)[0], maxiter + 3)
        assert_allclose(np.shape(r_h)[0], maxiter + 3)
    
    # 单独测试默认 maxiter=None 的情况
    with pytest.warns(UserWarning, match=msg):
        l, _, l_h, r_h = lobpcg(A, X, tol=1e-8,
                                retLambdaHistory=True,
                                retResidualNormsHistory=True)
    # 断言默认情况下历史记录数组的长度是否为 20 + 初始 + 最终 + 后处理
    assert_allclose(np.shape(l_h)[0], 20 + 3)
    assert_allclose(np.shape(r_h)[0], 20 + 3)
    
    # 检查计算出的最终特征值是否与历史记录中的最后一个一致
    assert_allclose(l, l_h[-1])
    # 确保历史记录输出是列表形式
    assert isinstance(l_h, list)
    assert isinstance(r_h, list)
    # 确保历史记录列表是类似数组的结构
    assert_allclose(np.shape(l_h), np.shape(np.asarray(l_h)))
    assert_allclose(np.shape(r_h), np.shape(np.asarray(r_h)))


@pytest.mark.slow
@pytest.mark.parametrize("n", [15])
@pytest.mark.parametrize("m", [1, 2])
@pytest.mark.filterwarnings("ignore:Exited at iteration")
@pytest.mark.filterwarnings("ignore:Exited postprocessing")
def test_diagonal_data_types(n, m):
    """Check lobpcg for diagonal matrices for all matrix types.
    Constraints are imposed, so a dense eigensolver eig cannot run.
    """
    # 使用固定的随机种子创建随机数生成器
    rnd = np.random.RandomState(0)
    # 定义对角线元素为 1 到 n 的数组
    vals = np.arange(1, n + 1)

    # 设置不同的稀疏矩阵格式进行测试
    list_sparse_format = ['bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil']
```