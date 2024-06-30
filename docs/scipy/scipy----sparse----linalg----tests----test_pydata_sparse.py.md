# `D:\src\scipysrc\scipy\scipy\sparse\linalg\tests\test_pydata_sparse.py`

```
# 导入 pytest 模块，用于单元测试
import pytest

# 导入 numpy 库，并使用别名 np
import numpy as np

# 导入 scipy.sparse 库，并使用别名 sp
import scipy.sparse as sp

# 导入 scipy.sparse.linalg 模块，并使用别名 splin
import scipy.sparse.linalg as splin

# 从 numpy.testing 模块中导入 assert_allclose 和 assert_equal 函数
from numpy.testing import assert_allclose, assert_equal

# 尝试导入 sparse 库，如果失败则将 sparse 设为 None
try:
    import sparse
except Exception:
    sparse = None

# 如果 sparse 为 None，则使用 pytest.mark.skipif 标记跳过测试，给出原因
pytestmark = pytest.mark.skipif(sparse is None,
                                reason="pydata/sparse not installed")

# 若 sparse 为 None，设定消息为指定版本未实现必要操作
msg = "pydata/sparse (0.15.1) does not implement necessary operations"

# 使用 pytest.param 创建参数元组 sparse_params，包含 "COO" 和 "DOK" 两个参数
sparse_params = (pytest.param("COO"),
                 pytest.param("DOK", marks=[pytest.mark.xfail(reason=msg)]))

# 定义 scipy 稀疏矩阵类列表
scipy_sparse_classes = [
    sp.bsr_matrix,
    sp.csr_matrix,
    sp.coo_matrix,
    sp.csc_matrix,
    sp.dia_matrix,
    sp.dok_matrix
]

# 使用 pytest.fixture 注册 sparse_cls 作为参数化 fixture，根据请求返回相应的 sparse 类
@pytest.fixture(params=sparse_params)
def sparse_cls(request):
    return getattr(sparse, request.param)

# 使用 pytest.fixture 注册 sp_sparse_cls 作为参数化 fixture，根据请求返回相应的 scipy 稀疏矩阵类
@pytest.fixture(params=scipy_sparse_classes)
def sp_sparse_cls(request):
    return request.param

# 定义 same_matrix fixture，生成相同的稠密和稀疏矩阵对
@pytest.fixture
def same_matrix(sparse_cls, sp_sparse_cls):
    np.random.seed(1234)
    A_dense = np.random.rand(9, 9)
    return sp_sparse_cls(A_dense), sparse_cls(A_dense)

# 定义 matrices fixture，生成稠密矩阵、对应的稀疏矩阵和向量 b
@pytest.fixture
def matrices(sparse_cls):
    np.random.seed(1234)
    A_dense = np.random.rand(9, 9)
    A_dense = A_dense @ A_dense.T  # 计算 A_dense 的转置乘积
    A_sparse = sparse_cls(A_dense)  # 使用 sparse_cls 构造稀疏矩阵 A_sparse
    b = np.random.rand(9)  # 生成随机向量 b
    return A_dense, A_sparse, b

# 定义 test_isolve_gmres 单元测试函数，测试 GMRES 解线性方程组
def test_isolve_gmres(matrices):
    # 测试中的迭代求解器使用相同的 isolve.utils.make_system 包装代码，只测试其中的一个
    A_dense, A_sparse, b = matrices
    x, info = splin.gmres(A_sparse, b, atol=1e-15)
    assert info == 0
    assert isinstance(x, np.ndarray)
    assert_allclose(A_sparse @ x, b)

# 定义 test_lsmr 单元测试函数，测试 LSMR 最小二乘问题求解
def test_lsmr(matrices):
    A_dense, A_sparse, b = matrices
    res0 = splin.lsmr(A_dense, b)
    res = splin.lsmr(A_sparse, b)
    assert_allclose(res[0], res0[0], atol=1e-3)

# 定义 test_lsmr_output_shape 单元测试函数，测试 LSMR 输出的形状
def test_lsmr_output_shape():
    x = splin.lsmr(A=np.ones((10, 1)), b=np.zeros(10), x0=np.ones(1))[0]
    assert_equal(x.shape, (1,))

# 定义 test_lsqr 单元测试函数，测试 LSQR 最小二乘问题求解
def test_lsqr(matrices):
    A_dense, A_sparse, b = matrices
    res0 = splin.lsqr(A_dense, b)
    res = splin.lsqr(A_sparse, b)
    assert_allclose(res[0], res0[0], atol=1e-5)

# 定义 test_eigs 单元测试函数，测试特征值问题求解
def test_eigs(matrices):
    A_dense, A_sparse, v0 = matrices

    M_dense = np.diag(v0**2)
    M_sparse = A_sparse.__class__(M_dense)

    w_dense, v_dense = splin.eigs(A_dense, k=3, v0=v0)
    w, v = splin.eigs(A_sparse, k=3, v0=v0)
    assert_allclose(w, w_dense)
    assert_allclose(v, v_dense)

    for M in [M_sparse, M_dense]:
        w_dense, v_dense = splin.eigs(A_dense, M=M_dense, k=3, v0=v0)
        w, v = splin.eigs(A_sparse, M=M, k=3, v0=v0)
        assert_allclose(w, w_dense)
        assert_allclose(v, v_dense)

        w_dense, v_dense = splin.eigsh(A_dense, M=M_dense, k=3, v0=v0)
        w, v = splin.eigsh(A_sparse, M=M, k=3, v0=v0)
        assert_allclose(w, w_dense)
        assert_allclose(v, v_dense)

# 定义 test_svds 单元测试函数，测试奇异值分解求解
def test_svds(matrices):
    A_dense, A_sparse, v0 = matrices

    u0, s0, vt0 = splin.svds(A_dense, k=2, v0=v0)
    # 对稀疏矩阵 A_sparse 进行截断奇异值分解（SVD），返回左奇异向量 u、奇异值 s 和右奇异向量的转置 vt
    u, s, vt = splin.svds(A_sparse, k=2, v0=v0)

    # 断言检查，确保计算得到的奇异值 s 与预期值 s0 接近
    assert_allclose(s, s0)
    
    # 断言检查，确保计算得到的左奇异向量 u 的绝对值与预期值 u0 的绝对值接近
    assert_allclose(np.abs(u), np.abs(u0))
    
    # 断言检查，确保计算得到的右奇异向量的转置 vt 的绝对值与预期值 vt0 的绝对值接近
    assert_allclose(np.abs(vt), np.abs(vt0))
# 测试 lobpcg 函数，用于求解稠密和稀疏矩阵的特征值问题
def test_lobpcg(matrices):
    # 解包输入参数
    A_dense, A_sparse, x = matrices
    # 将 x 转置为列向量
    X = x[:,None]

    # 对稠密矩阵 A_dense 进行 lobpcg 求解
    w_dense, v_dense = splin.lobpcg(A_dense, X)
    # 对稀疏矩阵 A_sparse 进行 lobpcg 求解
    w, v = splin.lobpcg(A_sparse, X)

    # 断言稠密和稀疏矩阵的特征值近似相等
    assert_allclose(w, w_dense)
    assert_allclose(v, v_dense)


# 测试 spsolve 函数，用于解线性方程组
def test_spsolve(matrices):
    # 解包输入参数
    A_dense, A_sparse, b = matrices
    # 随机生成新的右端向量 b2
    b2 = np.random.rand(len(b), 3)

    # 使用 spsolve 解稠密矩阵 A_dense 的线性方程组
    x0 = splin.spsolve(sp.csc_matrix(A_dense), b)
    # 使用 spsolve 解稀疏矩阵 A_sparse 的线性方程组
    x = splin.spsolve(A_sparse, b)
    assert isinstance(x, np.ndarray)
    assert_allclose(x, x0)

    # 使用 spsolve 解稠密矩阵 A_dense 的线性方程组（使用 umfpack）
    x0 = splin.spsolve(sp.csc_matrix(A_dense), b)
    x = splin.spsolve(A_sparse, b, use_umfpack=True)
    assert isinstance(x, np.ndarray)
    assert_allclose(x, x0)

    # 使用 spsolve 解稠密矩阵 A_dense 的线性方程组（右端向量为 b2）
    x0 = splin.spsolve(sp.csc_matrix(A_dense), b2)
    x = splin.spsolve(A_sparse, b2)
    assert isinstance(x, np.ndarray)
    assert_allclose(x, x0)

    # 使用 spsolve 解稠密矩阵 A_dense 的线性方程组（右端向量为稠密矩阵 A_dense 自身）
    x0 = splin.spsolve(sp.csc_matrix(A_dense),
                       sp.csc_matrix(A_dense))
    x = splin.spsolve(A_sparse, A_sparse)
    assert isinstance(x, type(A_sparse))
    assert_allclose(x.todense(), x0.todense())


# 测试 splu 函数，用于求解稀疏矩阵的 LU 分解
def test_splu(matrices):
    # 解包输入参数
    A_dense, A_sparse, b = matrices
    n = len(b)
    sparse_cls = type(A_sparse)

    # 对稀疏矩阵 A_sparse 进行 LU 分解
    lu = splin.splu(A_sparse)

    # 断言 LU 分解结果是稀疏矩阵类型
    assert isinstance(lu.L, sparse_cls)
    assert isinstance(lu.U, sparse_cls)

    # 构造置换矩阵 Pr 和 Pc
    _Pr_scipy = sp.csc_matrix((np.ones(n), (lu.perm_r, np.arange(n))))
    _Pc_scipy = sp.csc_matrix((np.ones(n), (np.arange(n), lu.perm_c)))
    Pr = sparse_cls.from_scipy_sparse(_Pr_scipy)
    Pc = sparse_cls.from_scipy_sparse(_Pc_scipy)
    A2 = Pr.T @ lu.L @ lu.U @ Pc.T

    # 断言 LU 分解重构的稀疏矩阵 A2 与原始稀疏矩阵 A_sparse 近似相等
    assert_allclose(A2.todense(), A_sparse.todense())

    # 使用 LU 分解求解线性方程组，并断言解接近单位矩阵
    z = lu.solve(A_sparse.todense())
    assert_allclose(z, np.eye(n), atol=1e-10)


# 测试 spilu 函数，用于求解稀疏矩阵的不完全 LU 分解
def test_spilu(matrices):
    # 解包输入参数
    A_dense, A_sparse, b = matrices
    sparse_cls = type(A_sparse)

    # 对稀疏矩阵 A_sparse 进行不完全 LU 分解
    lu = splin.spilu(A_sparse)

    # 断言不完全 LU 分解结果是稀疏矩阵类型
    assert isinstance(lu.L, sparse_cls)
    assert isinstance(lu.U, sparse_cls)

    # 使用不完全 LU 分解求解线性方程组，并断言解接近单位矩阵
    z = lu.solve(A_sparse.todense())
    assert_allclose(z, np.eye(len(b)), atol=1e-3)


# 测试 spsolve_triangular 函数，用于求解稀疏下三角矩阵的线性方程组
def test_spsolve_triangular(matrices):
    # 解包输入参数
    A_dense, A_sparse, b = matrices
    # 将稀疏矩阵 A_sparse 转换为其下三角部分
    A_sparse = sparse.tril(A_sparse)

    # 使用 spsolve_triangular 求解稀疏下三角矩阵 A_sparse 的线性方程组
    x = splin.spsolve_triangular(A_sparse, b)
    # 断言解 x 满足方程 A_sparse @ x = b
    assert_allclose(A_sparse @ x, b)


# 测试 onenormest 函数，用于计算矩阵的一范数估计
def test_onenormest(matrices):
    # 解包输入参数
    A_dense, A_sparse, b = matrices
    # 计算稠密矩阵 A_dense 的一范数估计
    est0 = splin.onenormest(A_dense)
    # 计算稀疏矩阵 A_sparse 的一范数估计
    est = splin.onenormest(A_sparse)
    # 断言两者估计值近似相等
    assert_allclose(est, est0)


# 测试 norm 函数，用于计算矩阵的范数
def test_norm(matrices):
    # 解包输入参数
    A_dense, A_sparse, b = matrices
    # 计算稠密矩阵 A_dense 的范数
    norm0 = splin.norm(sp.csr_matrix(A_dense))
    # 计算稀疏矩阵 A_sparse 的范数
    norm = splin.norm(A_sparse)
    # 断言两者范数近似相等
    assert_allclose(norm, norm0)


# 测试 inv 函数，用于计算矩阵的逆
def test_inv(matrices):
    # 解包输入参数
    A_dense, A_sparse, b = matrices
    # 计算稠密矩阵 A_dense 的逆
    x0 = splin.inv(sp.csc_matrix(A_dense))
    # 计算稀疏矩阵 A_sparse 的逆
    x = splin.inv(A_sparse)
    # 断言两者逆矩阵近似相等
    assert_allclose(x.todense(), x0.todense())


# 测试 expm 函数，用于计算矩阵的指数函数
def test_expm(matrices):
    # 解包输入参数
    A_dense, A_sparse, b = matrices
    # 计算稠密矩阵 A_dense 的指数函数
    x0 = splin.expm(sp.csc_matrix(A_dense))
    # 计算稀疏矩阵 A_sparse 的指数函数
    x = splin.expm(A_sparse)
    # 断言两者指数函数矩阵近似相等
    assert_allclose(x.todense(), x0.todense())


# 测试 expm_multiply 函数，用于计算矩阵与向量的指数乘积
def test_expm_multiply(matrices):
    # 解包输入参数
    A_dense, A_sparse, b = matrices
    # 计算稠
    # 使用稀疏矩阵 A_sparse 和向量 b 来计算指数乘积
    x = splin.expm_multiply(A_sparse, b)
    # 断言 x 和预期的 x0 在数值上非常接近
    assert_allclose(x, x0)
# 定义一个测试函数，用于比较两个稀疏矩阵对象是否相等
def test_eq(same_matrix):
    # 从传入的参数中获取两个稀疏矩阵对象
    sp_sparse, pd_sparse = same_matrix
    # 使用 NumPy 的 all() 函数检查两个稀疏矩阵对象是否逐元素相等，并断言结果为真
    assert (sp_sparse == pd_sparse).all()

# 定义另一个测试函数，用于比较两个稀疏矩阵对象是否不相等
def test_ne(same_matrix):
    # 从传入的参数中获取两个稀疏矩阵对象
    sp_sparse, pd_sparse = same_matrix
    # 使用 NumPy 的 any() 函数检查两个稀疏矩阵对象是否至少有一个元素不相等，并断言结果为假（即全都相等）
    assert not (sp_sparse != pd_sparse).any()
```