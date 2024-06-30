# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_extmath.py`

```
# 导入所需的库和模块
import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于编写和运行测试用例
from scipy import linalg, sparse  # 导入SciPy的线性代数和稀疏矩阵模块
from scipy.linalg import eigh  # 导入SciPy的特征值求解函数
from scipy.sparse.linalg import eigsh  # 导入SciPy的稀疏矩阵特征值求解函数
from scipy.special import expit  # 导入SciPy的逻辑函数

# 导入scikit-learn的数据生成和工具函数
from sklearn.datasets import make_low_rank_matrix, make_sparse_spd_matrix
from sklearn.utils import gen_batches  # 导入批处理生成器函数
from sklearn.utils._arpack import _init_arpack_v0  # 导入ARPACK初始化函数
from sklearn.utils._testing import (
    assert_allclose,  # 导入用于断言数组近似相等的函数
    assert_allclose_dense_sparse,  # 导入用于断言稠密和稀疏数组近似相等的函数
    assert_almost_equal,  # 导入用于断言两个数值近似相等的函数
    assert_array_almost_equal,  # 导入用于断言两个数组近似相等的函数
    assert_array_equal,  # 导入用于断言两个数组完全相等的函数
    skip_if_32bit,  # 导入用于在32位系统上跳过测试的函数
)
# 导入scikit-learn的扩展数学函数
from sklearn.utils.extmath import (
    _approximate_mode,  # 导入用于近似众数的函数
    _deterministic_vector_sign_flip,  # 导入用于确定性向量符号翻转的函数
    _incremental_mean_and_var,  # 导入用于增量计算均值和方差的函数
    _randomized_eigsh,  # 导入用于随机特征值求解的函数
    _safe_accumulator_op,  # 导入用于安全累加操作的函数
    cartesian,  # 导入笛卡尔积计算函数
    density,  # 导入用于计算稀疏矩阵密度的函数
    log_logistic,  # 导入用于计算对数逻辑函数的函数
    randomized_svd,  # 导入用于随机奇异值分解的函数
    row_norms,  # 导入用于计算行范数的函数
    safe_sparse_dot,  # 导入用于稀疏矩阵乘法的函数
    softmax,  # 导入用于计算Softmax函数的函数
    stable_cumsum,  # 导入用于计算稳定累加和的函数
    svd_flip,  # 导入用于奇异值分解后翻转的函数
    weighted_mode,  # 导入用于加权众数计算的函数
)
# 导入scikit-learn的修复函数
from sklearn.utils.fixes import (
    COO_CONTAINERS,  # 导入COO格式容器列表
    CSC_CONTAINERS,  # 导入CSC格式容器列表
    CSR_CONTAINERS,  # 导入CSR格式容器列表
    DOK_CONTAINERS,  # 导入DOK格式容器列表
    LIL_CONTAINERS,  # 导入LIL格式容器列表
    _mode,  # 导入用于众数计算的函数
)


@pytest.mark.parametrize(
    "sparse_container",
    COO_CONTAINERS + CSC_CONTAINERS + CSR_CONTAINERS + LIL_CONTAINERS,
)
def test_density(sparse_container):
    # 创建随机数生成器
    rng = np.random.RandomState(0)
    # 生成一个随机整数矩阵X，大小为(10, 5)
    X = rng.randint(10, size=(10, 5))
    # 将矩阵X的特定位置置为0，以创建稀疏模式
    X[1, 2] = 0
    X[5, 3] = 0

    # 断言稀疏容器对象(sparse_container)创建的稀疏矩阵的密度等于X的密度
    assert density(sparse_container(X)) == density(X)


def test_uniform_weights():
    # 当权重均匀时，结果应与stats.mode完全一致
    rng = np.random.RandomState(0)
    # 生成一个随机整数矩阵x，大小为(10, 5)
    x = rng.randint(10, size=(10, 5))
    # 权重为每个元素都是1的矩阵
    weights = np.ones(x.shape)

    # 对于所有轴（None, 0, 1）进行循环
    for axis in (None, 0, 1):
        # 计算x在给定轴上的众数和分数
        mode, score = _mode(x, axis)
        # 计算加权模式和分数
        mode2, score2 = weighted_mode(x, weights, axis=axis)

        # 断言众数和加权模式相等
        assert_array_equal(mode, mode2)
        # 断言分数和加权分数相等
        assert_array_equal(score, score2)


def test_random_weights():
    # 设置每行加权模式的预期结果为6，
    # 并且得分是可以轻松复现的
    mode_result = 6

    rng = np.random.RandomState(0)
    # 生成一个随机整数矩阵x，大小为(100, 10)
    x = rng.randint(mode_result, size=(100, 10))
    # 生成一个随机浮点数矩阵w，大小与x相同
    w = rng.random_sample(x.shape)

    # 将x的前5列设置为mode_result
    x[:, :5] = mode_result
    # 将w的前5列加1
    w[:, :5] += 1

    # 计算加权模式和分数
    mode, score = weighted_mode(x, w, axis=1)

    # 断言加权模式等于预期结果
    assert_array_equal(mode, mode_result)
    # 断言加权分数的展平版本与加权的前5列之和相等
    assert_array_almost_equal(score.ravel(), w[:, :5].sum(1))


@pytest.mark.parametrize("dtype", (np.int32, np.int64, np.float32, np.float64))
def test_randomized_svd_low_rank_all_dtypes(dtype):
    # 检查extmath.randomized_svd与linalg.svd的一致性
    n_samples = 100
    n_features = 500
    rank = 5
    k = 10
    # 根据dtype选择精度
    decimal = 5 if dtype == np.float32 else 7
    dtype = np.dtype(dtype)

    # 生成一个近似有效秩为rank的矩阵X，没有噪声成分（非常结构化的信号）
    X = make_low_rank_matrix(
        n_samples=n_samples,
        n_features=n_features,
        effective_rank=rank,
        tail_strength=0.0,
        random_state=0,
    ).astype(dtype, copy=False)
    # 断言确保输入的矩阵 X 的形状为 (n_samples, n_features)
    assert X.shape == (n_samples, n_features)

    # 使用精确但较慢的方法计算矩阵 X 的奇异值分解
    U, s, Vt = linalg.svd(X, full_matrices=False)

    # 将奇异值转换为特定的数据类型 dtype
    U = U.astype(dtype, copy=False)
    s = s.astype(dtype, copy=False)
    Vt = Vt.astype(dtype, copy=False)

    # 针对不同的正则化器 normalizer，使用快速的近似方法计算矩阵 X 的奇异值分解
    for normalizer in ["auto", "LU", "QR"]:  # 'none' 不稳定，不予考虑
        Ua, sa, Va = randomized_svd(
            X, k, power_iteration_normalizer=normalizer, random_state=0
        )

        # 如果输入的 dtype 是浮点数类型，则输出的 dtype 也应该是相同位数的浮点数
        # 如果输入的 dtype 是整数类型，则输出的 dtype 应为 float64
        if dtype.kind == "f":
            assert Ua.dtype == dtype
            assert sa.dtype == dtype
            assert Va.dtype == dtype
        else:
            assert Ua.dtype == np.float64
            assert sa.dtype == np.float64
            assert Va.dtype == np.float64

        # 断言确保计算得到的 Ua, sa, Va 的形状符合预期
        assert Ua.shape == (n_samples, k)
        assert sa.shape == (k,)
        assert Va.shape == (k, n_features)

        # 确保两种方法得到的奇异值在实际秩（rank）上是相等的
        assert_almost_equal(s[:k], sa, decimal=decimal)

        # 检查奇异向量，忽略其符号的影响
        assert_almost_equal(
            np.dot(U[:, :k], Vt[:k, :]), np.dot(Ua, Va), decimal=decimal
        )

        # 检查稀疏矩阵的表示方式
        for csr_container in CSR_CONTAINERS:
            X = csr_container(X)

            # 使用快速的近似方法计算稀疏矩阵 X 的奇异值分解
            Ua, sa, Va = randomized_svd(
                X, k, power_iteration_normalizer=normalizer, random_state=0
            )

            # 根据输入的 dtype 类型进行断言
            if dtype.kind == "f":
                assert Ua.dtype == dtype
                assert sa.dtype == dtype
                assert Va.dtype == dtype
            else:
                assert Ua.dtype.kind == "f"
                assert sa.dtype.kind == "f"
                assert Va.dtype.kind == "f"

            # 确保稀疏矩阵 X 计算得到的奇异值与实际秩上的奇异值近似相等
            assert_almost_equal(s[:rank], sa[:rank], decimal=decimal)
# 使用 pytest.mark.parametrize 装饰器，为 test_randomized_eigsh 函数添加参数化测试
@pytest.mark.parametrize("dtype", (np.int32, np.int64, np.float32, np.float64))
def test_randomized_eigsh(dtype):
    """Test that `_randomized_eigsh` returns the appropriate components"""

    # 创建随机数生成器 rng，种子为 42
    rng = np.random.RandomState(42)
    # 创建对角矩阵 X，其中包含指定数据类型的对角元素
    X = np.diag(np.array([1.0, -2.0, 0.0, 3.0], dtype=dtype))
    # 随机旋转矩阵，保持 X 的特征值不变
    rand_rot = np.linalg.qr(rng.normal(size=X.shape))[0]
    X = rand_rot @ X @ rand_rot.T

    # 使用 'module' 方法选择特征值，返回特征值和特征向量
    eigvals, eigvecs = _randomized_eigsh(X, n_components=2, selection="module")
    # 断言特征值的形状为 (2,)
    assert eigvals.shape == (2,)
    # 检查特征值数组准确度，包含一个负特征值
    assert_array_almost_equal(eigvals, [3.0, -2.0])  # negative eigenvalue here
    # 断言特征向量的形状为 (4, 2)
    assert eigvecs.shape == (4, 2)

    # 使用 'value' 方法选择特征值，预期引发 NotImplementedError 异常
    with pytest.raises(NotImplementedError):
        _randomized_eigsh(X, n_components=2, selection="value")


# 使用 pytest.mark.parametrize 装饰器，为 test_randomized_eigsh_compared_to_others 函数添加参数化测试
@pytest.mark.parametrize("k", (10, 50, 100, 199, 200))
def test_randomized_eigsh_compared_to_others(k):
    """Check that `_randomized_eigsh` is similar to other `eigsh`

    Tests that for a random PSD matrix, `_randomized_eigsh` provides results
    comparable to LAPACK (scipy.linalg.eigh) and ARPACK
    (scipy.sparse.linalg.eigsh).

    Note: some versions of ARPACK do not support k=n_features.
    """

    # 创建一个随机正定对称矩阵 X
    n_features = 200
    X = make_sparse_spd_matrix(n_features, random_state=0)

    # 比较两种版本的 randomized eigsh
    # 粗略且快速的计算方式
    eigvals, eigvecs = _randomized_eigsh(
        X, n_components=k, selection="module", n_iter=25, random_state=0
    )
    # 更精确但速度较慢的计算方式，使用 QR 方法进行正交化
    eigvals_qr, eigvecs_qr = _randomized_eigsh(
        X,
        n_components=k,
        n_iter=25,
        n_oversamples=20,
        random_state=0,
        power_iteration_normalizer="QR",
        selection="module",
    )

    # 使用 LAPACK 进行计算
    eigvals_lapack, eigvecs_lapack = eigh(
        X, subset_by_index=(n_features - k, n_features - 1)
    )
    indices = eigvals_lapack.argsort()[::-1]
    eigvals_lapack = eigvals_lapack[indices]
    eigvecs_lapack = eigvecs_lapack[:, indices]

    # -- 比较特征值
    assert eigvals_lapack.shape == (k,)
    # 比较精度为小数点后 6 位
    assert_array_almost_equal(eigvals, eigvals_lapack, decimal=6)
    assert_array_almost_equal(eigvals_qr, eigvals_lapack, decimal=6)

    # -- 比较特征向量
    assert eigvecs_lapack.shape == (n_features, k)
    # 翻转特征向量的符号以确保确定性输出
    dummy_vecs = np.zeros_like(eigvecs).T
    eigvecs, _ = svd_flip(eigvecs, dummy_vecs)
    eigvecs_qr, _ = svd_flip(eigvecs_qr, dummy_vecs)
    eigvecs_lapack, _ = svd_flip(eigvecs_lapack, dummy_vecs)
    assert_array_almost_equal(eigvecs, eigvecs_lapack, decimal=4)
    assert_array_almost_equal(eigvecs_qr, eigvecs_lapack, decimal=6)
    # 如果 k 小于特征数量 n_features，则进行以下操作
    if k < n_features:
        # 使用 _init_arpack_v0 函数生成初始向量 v0，用于 ARPACK 算法
        v0 = _init_arpack_v0(n_features, random_state=0)
        # 使用 eigsh 函数计算 X 的前 k 个最大代数特征值和对应的特征向量
        # which="LA" 表示选择最大代数特征值，tol=0 表示容差为零，maxiter=None 表示不限制迭代次数，v0 是初始向量
        eigvals_arpack, eigvecs_arpack = eigsh(
            X, k, which="LA", tol=0, maxiter=None, v0=v0
        )
        # 对 ARPACK 算法得到的特征值按从大到小排序的索引
        indices = eigvals_arpack.argsort()[::-1]
        # 按索引重新排列特征值和特征向量
        eigvals_arpack = eigvals_arpack[indices]  # 特征值重新排序
        # 检查 ARPACK 算法计算的特征值是否与 LAPACK 算法的特征值几乎相等
        assert_array_almost_equal(eigvals_lapack, eigvals_arpack, decimal=10)
        # 对 ARPACK 算法得到的特征向量进行相同的重新排列
        eigvecs_arpack = eigvecs_arpack[:, indices]  # 特征向量重新排序
        # 使用 svd_flip 函数调整特征向量的符号
        eigvecs_arpack, _ = svd_flip(eigvecs_arpack, dummy_vecs)
        # 检查 ARPACK 算法计算的特征向量是否与 LAPACK 算法的特征向量几乎相等
        assert_array_almost_equal(eigvecs_arpack, eigvecs_lapack, decimal=8)
# 使用 pytest 的 mark.parametrize 装饰器标记，定义了多个测试参数组合
@pytest.mark.parametrize(
    "n,rank",
    [
        (10, 7),
        (100, 10),
        (100, 80),
        (500, 10),
        (500, 250),
        (500, 400),
    ],
)
# 定义测试函数 test_randomized_eigsh_reconst_low_rank，用于测试随机化特征值分解的重构能力
def test_randomized_eigsh_reconst_low_rank(n, rank):
    """Check that randomized_eigsh is able to reconstruct a low rank psd matrix

    Tests that the decomposition provided by `_randomized_eigsh` leads to
    orthonormal eigenvectors, and that a low rank PSD matrix can be effectively
    reconstructed with good accuracy using it.
    """
    # 断言 rank 应小于 n，确保测试条件成立
    assert rank < n

    # 创建一个低秩正定对称矩阵 A
    rng = np.random.RandomState(69)
    X = rng.randn(n, rank)
    A = X @ X.T

    # 使用 "_randomized_eigsh" 近似 A 的右特征向量和特征值
    S, V = _randomized_eigsh(A, n_components=rank, random_state=rng)
    # 检查特征向量是否正交
    assert_array_almost_equal(np.linalg.norm(V, axis=0), np.ones(S.shape))
    assert_array_almost_equal(V.T @ V, np.diag(np.ones(S.shape)))
    # 重构 A
    A_reconstruct = V @ np.diag(S) @ V.T

    # 检查重构是否准确
    assert_array_almost_equal(A_reconstruct, A, decimal=6)


# 使用 pytest 的 mark.parametrize 装饰器标记，定义了测试 dtype 参数的多个值
# 同时使用 CSR_CONTAINERS 中的不同稀疏矩阵容器进行测试
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 定义测试函数 test_row_norms，用于测试行范数计算函数 row_norms 的精度
def test_row_norms(dtype, csr_container):
    X = np.random.RandomState(42).randn(100, 100)
    if dtype is np.float32:
        precision = 4
    else:
        precision = 5

    X = X.astype(dtype, copy=False)
    sq_norm = (X**2).sum(axis=1)

    # 检查 row_norms 函数计算的平方范数是否与预期值一致
    assert_array_almost_equal(sq_norm, row_norms(X, squared=True), precision)
    # 检查 row_norms 函数计算的非平方范数是否与预期值一致
    assert_array_almost_equal(np.sqrt(sq_norm), row_norms(X), precision)

    for csr_index_dtype in [np.int32, np.int64]:
        Xcsr = csr_container(X, dtype=dtype)
        # csr_matrix 默认使用 int32 索引，必要时将其升级为 int64
        if csr_index_dtype is np.int64:
            Xcsr.indptr = Xcsr.indptr.astype(csr_index_dtype, copy=False)
            Xcsr.indices = Xcsr.indices.astype(csr_index_dtype, copy=False)
        assert Xcsr.indices.dtype == csr_index_dtype
        assert Xcsr.indptr.dtype == csr_index_dtype
        # 检查稀疏矩阵 Xcsr 的行范数计算结果是否正确
        assert_array_almost_equal(sq_norm, row_norms(Xcsr, squared=True), precision)
        assert_array_almost_equal(np.sqrt(sq_norm), row_norms(Xcsr), precision)


# 定义测试函数 test_randomized_svd_low_rank_with_noise，用于测试随机化奇异值分解处理带噪声矩阵的能力
def test_randomized_svd_low_rank_with_noise():
    # Check that extmath.randomized_svd can handle noisy matrices
    n_samples = 100
    n_features = 500
    rank = 5
    k = 10

    # 生成具有结构近似秩 `rank` 和重要噪声成分的矩阵 X
    X = make_low_rank_matrix(
        n_samples=n_samples,
        n_features=n_features,
        effective_rank=rank,
        tail_strength=0.1,
        random_state=0,
    )
    assert X.shape == (n_samples, n_features)

    # 使用传统的精确方法计算 X 的奇异值
    _, s, _ = linalg.svd(X, full_matrices=False)
    # 对于每种标准化方法，依次进行以下操作：["auto", "none", "LU", "QR"]
    for normalizer in ["auto", "none", "LU", "QR"]:
        # 使用快速近似方法计算矩阵 X 的奇异值，不使用迭代幂方法
        _, sa, _ = randomized_svd(
            X, k, n_iter=0, power_iteration_normalizer=normalizer, random_state=0
        )

        # 检查近似值是否容忍噪音：
        assert np.abs(s[:k] - sa).max() > 0.01

        # 使用带有迭代幂方法的快速近似方法计算矩阵 X 的奇异值
        _, sap, _ = randomized_svd(
            X, k, power_iteration_normalizer=normalizer, random_state=0
        )

        # 迭代幂方法有助于消除噪音：
        assert_almost_equal(s[:k], sap, decimal=3)
# 定义一个测试函数，用于测试 randomized_svd 在处理无穷秩矩阵时的情况
def test_randomized_svd_infinite_rank():
    # 设定样本数和特征数
    n_samples = 100
    n_features = 500
    # 矩阵的秩
    rank = 5
    # 奇异值个数
    k = 10

    # 创建一个低秩矩阵 X，其秩为 rank，具有完整尾部强度，随机状态为 0
    X = make_low_rank_matrix(
        n_samples=n_samples,
        n_features=n_features,
        effective_rank=rank,
        tail_strength=1.0,
        random_state=0,
    )
    # 断言矩阵 X 的形状为 (n_samples, n_features)
    assert X.shape == (n_samples, n_features)

    # 使用精确但较慢的方法计算矩阵 X 的奇异值分解
    _, s, _ = linalg.svd(X, full_matrices=False)

    # 对于不同的正则化方式，使用快速近似方法计算矩阵 X 的奇异值
    # 无迭代幂法
    for normalizer in ["auto", "none", "LU", "QR"]:
        _, sa, _ = randomized_svd(
            X, k, n_iter=0, power_iteration_normalizer=normalizer, random_state=0
        )

        # 近似方法对噪声不具容忍性
        assert np.abs(s[:k] - sa).max() > 0.1

        # 使用迭代幂法
        _, sap, _ = randomized_svd(
            X, k, n_iter=5, power_iteration_normalizer=normalizer, random_state=0
        )

        # 迭代幂法仍能在请求的秩上获取大部分结构
        assert_almost_equal(s[:k], sap, decimal=3)


def test_randomized_svd_transpose_consistency():
    # 检查转置设计矩阵的一致性影响
    n_samples = 100
    n_features = 500
    rank = 4
    k = 10

    # 创建一个低秩矩阵 X，其秩为 rank，尾部强度为 0.5，随机状态为 0
    X = make_low_rank_matrix(
        n_samples=n_samples,
        n_features=n_features,
        effective_rank=rank,
        tail_strength=0.5,
        random_state=0,
    )
    # 断言矩阵 X 的形状为 (n_samples, n_features)
    assert X.shape == (n_samples, n_features)

    # 使用 randomized_svd 进行奇异值分解，不同的 transpose 参数
    U1, s1, V1 = randomized_svd(X, k, n_iter=3, transpose=False, random_state=0)
    U2, s2, V2 = randomized_svd(X, k, n_iter=3, transpose=True, random_state=0)
    U3, s3, V3 = randomized_svd(X, k, n_iter=3, transpose="auto", random_state=0)
    U4, s4, V4 = linalg.svd(X, full_matrices=False)

    # 断言近似奇异值与精确计算结果的一致性
    assert_almost_equal(s1, s4[:k], decimal=3)
    assert_almost_equal(s2, s4[:k], decimal=3)
    assert_almost_equal(s3, s4[:k], decimal=3)

    # 断言 U1*V1 和精确计算结果 U4[:,:k]*V4[:k,:] 的一致性
    assert_almost_equal(np.dot(U1, V1), np.dot(U4[:, :k], V4[:k, :]), decimal=2)
    assert_almost_equal(np.dot(U2, V2), np.dot(U4[:, :k], V4[:k, :]), decimal=2)

    # 在这种情况下，'auto' 相当于 transpose
    assert_almost_equal(s2, s3)


def test_randomized_svd_power_iteration_normalizer():
    # 对于大量幂迭代，在此数据集上，power_iteration_normalizer='none' 会发散
    rng = np.random.RandomState(42)
    X = make_low_rank_matrix(100, 500, effective_rank=50, random_state=rng)
    X += 3 * rng.randint(0, 2, size=X.shape)
    n_components = 50
    # 使用随机化奇异值分解（randomized_svd）来计算矩阵 X 的前 n_components 个奇异值分解
    # 这里进行了两次迭代，不使用任何归一化器（power_iteration_normalizer="none"），随机种子为 0
    U, s, Vt = randomized_svd(
        X, n_components, n_iter=2, power_iteration_normalizer="none", random_state=0
    )
    # 计算矩阵 X 与近似重建的差异矩阵 A
    A = X - U.dot(np.diag(s).dot(Vt))
    # 计算 Frobenius 范数（矩阵的二范数）作为误差度量
    error_2 = linalg.norm(A, ord="fro")

    # 再次使用随机化奇异值分解，进行20次迭代，其他参数与上面相同
    U, s, Vt = randomized_svd(
        X, n_components, n_iter=20, power_iteration_normalizer="none", random_state=0
    )
    # 计算新的差异矩阵 A
    A = X - U.dot(np.diag(s).dot(Vt))
    # 计算新的误差度量
    error_20 = linalg.norm(A, ord="fro")
    # 断言两次迭代所得误差的绝对值大于100，用于验证算法的收敛性
    assert np.abs(error_2 - error_20) > 100

    # 对不同的归一化器（normalizer）进行迭代测试，包括 "LU", "QR", "auto"
    for normalizer in ["LU", "QR", "auto"]:
        # 使用不同归一化器进行两次迭代的随机化奇异值分解
        U, s, Vt = randomized_svd(
            X,
            n_components,
            n_iter=2,
            power_iteration_normalizer=normalizer,
            random_state=0,
        )
        # 计算差异矩阵 A
        A = X - U.dot(np.diag(s).dot(Vt))
        # 计算误差度量
        error_2 = linalg.norm(A, ord="fro")

        # 对于不同的迭代次数 [5, 10, 50]，再次进行随机化奇异值分解
        for i in [5, 10, 50]:
            U, s, Vt = randomized_svd(
                X,
                n_components,
                n_iter=i,
                power_iteration_normalizer=normalizer,
                random_state=0,
            )
            # 计算新的差异矩阵 A
            A = X - U.dot(np.diag(s).dot(Vt))
            # 计算新的误差度量
            error = linalg.norm(A, ord="fro")
            # 断言误差度量的绝对值小于15，用于验证算法的稳定性
            assert 15 > np.abs(error_2 - error)
# 使用 pytest.mark.parametrize 装饰器，为 test_randomized_svd_sparse_warnings 函数参数化 sparse_container
@pytest.mark.parametrize("sparse_container", DOK_CONTAINERS + LIL_CONTAINERS)
def test_randomized_svd_sparse_warnings(sparse_container):
    # 设置随机数生成器 rng，种子为 42
    rng = np.random.RandomState(42)
    # 创建一个低秩矩阵 X，大小为 50x20，有效秩为 10，使用指定的随机数生成器
    X = make_low_rank_matrix(50, 20, effective_rank=10, random_state=rng)
    # 设定 SVD 组件数为 5

    # 将 X 转换为 sparse_container 类型的稀疏矩阵
    X = sparse_container(X)
    # 构造警告消息，指出对于 sparse_container，计算 SVD 操作较为耗时，建议使用 csr_matrix 更高效
    warn_msg = (
        "Calculating SVD of a {} is expensive. csr_matrix is more efficient.".format(
            sparse_container.__name__
        )
    )
    # 使用 pytest.warns 检查是否会产生 SparseEfficiencyWarning 警告，并匹配 warn_msg
    with pytest.warns(sparse.SparseEfficiencyWarning, match=warn_msg):
        # 调用 randomized_svd 函数进行奇异值分解，设置参数为 X, n_components, n_iter=1, power_iteration_normalizer="none"
        randomized_svd(X, n_components, n_iter=1, power_iteration_normalizer="none")


# 定义测试函数 test_svd_flip
def test_svd_flip():
    # 创建随机数生成器 rs，种子为 1999
    rs = np.random.RandomState(1999)
    n_samples = 20
    n_features = 10
    # 生成一个随机矩阵 X，大小为 n_samples x n_features
    X = rs.randn(n_samples, n_features)

    # 对 X 进行奇异值分解，返回 U、S、Vt
    U, S, Vt = linalg.svd(X, full_matrices=False)
    # 使用 svd_flip 函数进行 U 和 Vt 的翻转，不使用基于 u 的决策
    U1, V1 = svd_flip(U, Vt, u_based_decision=False)
    # 断言重构的矩阵与原始矩阵 X 相近
    assert_almost_equal(np.dot(U1 * S, V1), X, decimal=6)

    # 对 X 的转置进行奇异值分解
    XT = X.T
    U, S, Vt = linalg.svd(XT, full_matrices=False)
    # 使用 svd_flip 函数进行 U 和 Vt 的翻转，使用基于 u 的决策
    U2, V2 = svd_flip(U, Vt, u_based_decision=True)
    # 断言转置后的矩阵与原始转置矩阵 XT 相近
    assert_almost_equal(np.dot(U2 * S, V2), XT, decimal=6)

    # 检查不同翻转方法在重构下的等效性
    U_flip1, V_flip1 = svd_flip(U, Vt, u_based_decision=True)
    assert_almost_equal(np.dot(U_flip1 * S, V_flip1), XT, decimal=6)
    U_flip2, V_flip2 = svd_flip(U, Vt, u_based_decision=False)
    assert_almost_equal(np.dot(U_flip2 * S, V_flip2), XT, decimal=6)


# 使用 pytest.mark.parametrize 装饰器，为 test_svd_flip_max_abs_cols 函数参数化 n_samples 和 n_features
@pytest.mark.parametrize("n_samples, n_features", [(3, 4), (4, 3)])
def test_svd_flip_max_abs_cols(n_samples, n_features, global_random_seed):
    # 创建随机数生成器 rs，种子为 global_random_seed
    rs = np.random.RandomState(global_random_seed)
    # 生成一个随机矩阵 X，大小为 n_samples x n_features
    X = rs.randn(n_samples, n_features)
    # 对 X 进行奇异值分解，返回 U、S、Vt
    U, _, Vt = linalg.svd(X, full_matrices=False)

    # 使用 svd_flip 函数进行 U 和 Vt 的翻转，使用基于 u 的决策
    U1, _ = svd_flip(U, Vt, u_based_decision=True)
    # 找出每列中绝对值最大的行索引，并验证其大于等于 0
    max_abs_U1_row_idx_for_col = np.argmax(np.abs(U1), axis=0)
    assert (U1[max_abs_U1_row_idx_for_col, np.arange(U1.shape[1])] >= 0).all()

    # 使用 svd_flip 函数进行 U 和 Vt 的翻转，不使用基于 u 的决策
    _, V2 = svd_flip(U, Vt, u_based_decision=False)
    # 找出每行中绝对值最大的列索引，并验证其大于等于 0
    max_abs_V2_col_idx_for_row = np.argmax(np.abs(V2), axis=1)
    assert (V2[np.arange(V2.shape[0]), max_abs_V2_col_idx_for_row] >= 0).all()


# 定义测试函数 test_randomized_svd_sign_flip
def test_randomized_svd_sign_flip():
    # 创建一个示例矩阵 a
    a = np.array([[2.0, 0.0], [0.0, 1.0]])
    # 进行随机化奇异值分解，flip_sign=True，种子为 41
    u1, s1, v1 = randomized_svd(a, 2, flip_sign=True, random_state=41)
    # 循环进行多次随机化奇异值分解，并断言结果与第一次一致
    for seed in range(10):
        u2, s2, v2 = randomized_svd(a, 2, flip_sign=True, random_state=seed)
        assert_almost_equal(u1, u2)
        assert_almost_equal(v1, v2)
        assert_almost_equal(np.dot(u2 * s2, v2), a)
        assert_almost_equal(np.dot(u2.T, u2), np.eye(2))
        assert_almost_equal(np.dot(v2.T, v2), np.eye(2))


# 定义测试函数 test_randomized_svd_sign_flip_with_transpose
def test_randomized_svd_sign_flip_with_transpose():
    # 检查 randomized_svd 的符号翻转是否始终基于 u 进行，无论是否进行了转置
    # 定义一个函数，用于检查两个矩阵的最大负载是否为正的布尔值元组
    def max_loading_is_positive(u, v):
        """
        返回一个布尔值元组，指示在矩阵 u 的所有行和矩阵 v 的所有列中，
        是否最大化 np.abs 时的值是否都为正。
        """
        # 检查基于 u 的条件：是否所有行的最大绝对值等于每行的最大值
        u_based = (np.abs(u).max(axis=0) == u.max(axis=0)).all()
        # 检查基于 v 的条件：是否所有列的最大绝对值等于每列的最大值
        v_based = (np.abs(v).max(axis=1) == v.max(axis=1)).all()
        return u_based, v_based
    
    # 创建一个 10x8 的矩阵 mat，其中元素从 0 到 79
    mat = np.arange(10 * 8).reshape(10, -1)
    
    # 使用 randomized_svd 函数对矩阵 mat 进行奇异值分解，不进行转置
    u_flipped, _, v_flipped = randomized_svd(mat, 3, flip_sign=True, random_state=0)
    # 检查得到的 u_flipped 和 v_flipped 是否满足最大负载为正的条件
    u_based, v_based = max_loading_is_positive(u_flipped, v_flipped)
    assert u_based  # 断言 u_based 应为 True
    assert not v_based  # 断言 v_based 应为 False
    
    # 使用 randomized_svd 函数对矩阵 mat 进行奇异值分解，并进行转置
    u_flipped_with_transpose, _, v_flipped_with_transpose = randomized_svd(
        mat, 3, flip_sign=True, transpose=True, random_state=0
    )
    # 检查得到的 u_flipped_with_transpose 和 v_flipped_with_transpose 是否满足最大负载为正的条件
    u_based, v_based = max_loading_is_positive(
        u_flipped_with_transpose, v_flipped_with_transpose
    )
    assert u_based  # 断言 u_based 应为 True
    assert not v_based  # 断言 v_based 应为 False
@pytest.mark.parametrize("n", [50, 100, 300])
@pytest.mark.parametrize("m", [50, 100, 300])
@pytest.mark.parametrize("k", [10, 20, 50])
@pytest.mark.parametrize("seed", range(5))
def test_randomized_svd_lapack_driver(n, m, k, seed):
    # 检查不同的 SVD 驱动程序提供一致的结果

    # 创建随机数生成器
    rng = np.random.RandomState(seed)
    # 生成一个 n x m 的随机矩阵 X
    X = rng.rand(n, m)

    # 使用 gesdd 驱动程序进行随机化 SVD 分解
    u1, s1, vt1 = randomized_svd(X, k, svd_lapack_driver="gesdd", random_state=0)
    # 使用 gesvd 驱动程序进行随机化 SVD 分解
    u2, s2, vt2 = randomized_svd(X, k, svd_lapack_driver="gesvd", random_state=0)

    # 检查形状和内容是否一致
    assert u1.shape == u2.shape
    assert_allclose(u1, u2, atol=0, rtol=1e-3)

    assert s1.shape == s2.shape
    assert_allclose(s1, s2, atol=0, rtol=1e-3)

    assert vt1.shape == vt2.shape
    assert_allclose(vt1, vt2, atol=0, rtol=1e-3)


def test_cartesian():
    # 检查笛卡尔积是否提供正确的结果

    # 定义多个轴
    axes = (np.array([1, 2, 3]), np.array([4, 5]), np.array([6, 7]))

    # 预期的输出结果
    true_out = np.array(
        [
            [1, 4, 6],
            [1, 4, 7],
            [1, 5, 6],
            [1, 5, 7],
            [2, 4, 6],
            [2, 4, 7],
            [2, 5, 6],
            [2, 5, 7],
            [3, 4, 6],
            [3, 4, 7],
            [3, 5, 6],
            [3, 5, 7],
        ]
    )

    # 计算笛卡尔积
    out = cartesian(axes)
    # 断言结果是否与预期一致
    assert_array_equal(true_out, out)

    # 检查单个轴的情况
    x = np.arange(3)
    assert_array_equal(x[:, np.newaxis], cartesian((x,)))


@pytest.mark.parametrize(
    "arrays, output_dtype",
    [
        (
            [np.array([1, 2, 3], dtype=np.int32), np.array([4, 5], dtype=np.int64)],
            np.dtype(np.int64),
        ),
        (
            [np.array([1, 2, 3], dtype=np.int32), np.array([4, 5], dtype=np.float64)],
            np.dtype(np.float64),
        ),
        (
            [np.array([1, 2, 3], dtype=np.int32), np.array(["x", "y"], dtype=object)],
            np.dtype(object),
        ),
    ],
)
def test_cartesian_mix_types(arrays, output_dtype):
    """检查混合类型情况下的笛卡尔积是否有效。"""
    output = cartesian(arrays)

    assert output.dtype == output_dtype


# TODO(1.6): remove this test
def test_logistic_sigmoid():
    # 检查 logistic sigmoid 实现的正确性和健壮性
    def naive_log_logistic(x):
        return np.log(expit(x))

    # 创建测试用例
    x = np.linspace(-2, 2, 50)
    warn_msg = "`log_logistic` is deprecated and will be removed"
    # 断言使用 log_logistic 函数的结果与 naive_log_logistic 函数的结果接近
    with pytest.warns(FutureWarning, match=warn_msg):
        assert_array_almost_equal(log_logistic(x), naive_log_logistic(x))

    # 检查极端情况
    extreme_x = np.array([-100.0, 100.0])
    with pytest.warns(FutureWarning, match=warn_msg):
        assert_array_almost_equal(log_logistic(extreme_x), [-100, 0])


@pytest.fixture()
def rng():
    return np.random.RandomState(42)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_incremental_weighted_mean_and_variance_simple(rng, dtype):
    """检查简单的增量加权均值和方差计算是否正确。"""
    mult = 10
    # 使用随机数生成器 rng 创建一个形状为 (1000, 20) 的二维数组 X，并且将其类型转换为指定的 dtype，并乘以 mult
    X = rng.rand(1000, 20).astype(dtype) * mult
    
    # 使用随机数生成器 rng 创建一个长度为 X.shape[0] 的数组作为样本权重 sample_weight，并乘以 mult
    sample_weight = rng.rand(X.shape[0]) * mult
    
    # 调用 _incremental_mean_and_var 函数计算 X 数据集的加权平均值 mean 和加权方差 var，
    # 初始的平均值和方差设为 0，同时使用 sample_weight 作为样本权重
    mean, var, _ = _incremental_mean_and_var(X, 0, 0, 0, sample_weight=sample_weight)
    
    # 使用 np.average 函数计算加权平均值，weights 参数使用 sample_weight，axis=0 表示沿着列的方向计算平均值
    expected_mean = np.average(X, weights=sample_weight, axis=0)
    
    # 使用 np.average 函数计算加权平均值的平方，weights 参数使用 sample_weight，axis=0 表示沿着列的方向计算平均值，
    # 然后减去平均值的平方，得到加权方差 expected_var
    expected_var = np.average(X**2, weights=sample_weight, axis=0) - expected_mean**2
    
    # 断言检查计算得到的 mean 和 expected_mean 的近似程度，保留的小数点位数很高，以确保精确度
    assert_almost_equal(mean, expected_mean)
    
    # 断言检查计算得到的 var 和 expected_var 的近似程度，保留的小数点位数很高，以确保精确度
    assert_almost_equal(var, expected_var)
@pytest.mark.parametrize("mean", [0, 1e7, -1e7])
@pytest.mark.parametrize("var", [1, 1e-8, 1e5])
@pytest.mark.parametrize(
    "weight_loc, weight_scale", [(0, 1), (0, 1e-8), (1, 1e-8), (10, 1), (1e7, 1)]
)
# 定义测试函数，用于逐步验证加权平均和方差的正确性和数值稳定性
def test_incremental_weighted_mean_and_variance(
    mean, var, weight_loc, weight_scale, rng
):
    # 内部辅助函数，断言函数结果与期望值的接近程度
    def _assert(X, sample_weight, expected_mean, expected_var):
        n = X.shape[0]
        # 针对不同的块大小进行迭代
        for chunk_size in [1, n // 10 + 1, n // 4 + 1, n // 2 + 1, n]:
            last_mean, last_weight_sum, last_var = 0, 0, 0
            # 对数据集按照块迭代，并调用增量计算均值和方差的函数
            for batch in gen_batches(n, chunk_size):
                last_mean, last_var, last_weight_sum = _incremental_mean_and_var(
                    X[batch],
                    last_mean,
                    last_var,
                    last_weight_sum,
                    sample_weight=sample_weight[batch],
                )
            # 断言最终计算得到的均值与期望值接近
            assert_allclose(last_mean, expected_mean)
            # 断言最终计算得到的方差与期望值接近，设置允许的绝对误差为1e-6
            assert_allclose(last_var, expected_var, atol=1e-6)

    size = (100, 20)
    # 生成指定均值和方差分布的权重数组
    weight = rng.normal(loc=weight_loc, scale=weight_scale, size=size[0])

    # 使用随机数生成器生成指定均值和方差的数据集 X
    X = rng.normal(loc=mean, scale=var, size=size)
    # 使用增量安全累加操作函数计算加权平均值，作为期望的均值
    expected_mean = _safe_accumulator_op(np.average, X, weights=weight, axis=0)
    # 使用增量安全累加操作函数计算加权方差，作为期望的方差
    expected_var = _safe_accumulator_op(
        np.average, (X - expected_mean) ** 2, weights=weight, axis=0
    )
    # 调用内部断言函数进行验证
    _assert(X, weight, expected_mean, expected_var)

    # 与非加权平均值 np.mean 进行比较
    X = rng.normal(loc=mean, scale=var, size=size)
    ones_weight = np.ones(size[0])
    # 使用增量安全累加操作函数计算非加权平均值，作为期望的均值
    expected_mean = _safe_accumulator_op(np.mean, X, axis=0)
    # 使用增量安全累加操作函数计算方差，作为期望的方差
    expected_var = _safe_accumulator_op(np.var, X, axis=0)
    # 再次调用内部断言函数进行验证
    _assert(X, ones_weight, expected_mean, expected_var)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
# 测试忽略 NaN 值的增量加权均值和方差计算
def test_incremental_weighted_mean_and_variance_ignore_nan(dtype):
    # 初始化旧的均值、方差和权重总和
    old_means = np.array([535.0, 535.0, 535.0, 535.0])
    old_variances = np.array([4225.0, 4225.0, 4225.0, 4225.0])
    old_weight_sum = np.array([2, 2, 2, 2], dtype=np.int32)
    # 创建样本权重数组，包含 NaN 值和不含 NaN 值的情况
    sample_weights_X = np.ones(3)
    sample_weights_X_nan = np.ones(4)

    # 创建包含 NaN 值的数据集 X_nan 和不含 NaN 值的数据集 X
    X = np.array(
        [[170, 170, 170, 170], [430, 430, 430, 430], [300, 300, 300, 300]]
    ).astype(dtype)

    X_nan = np.array(
        [
            [170, np.nan, 170, 170],
            [np.nan, 170, 430, 430],
            [430, 430, np.nan, 300],
            [300, 300, 300, np.nan],
        ]
    ).astype(dtype)

    # 使用增量均值和方差计算函数计算 X 的均值、方差和计数
    X_means, X_variances, X_count = _incremental_mean_and_var(
        X, old_means, old_variances, old_weight_sum, sample_weight=sample_weights_X
    )
    # 使用增量均值和方差计算函数计算 X_nan 的均值、方差和计数
    X_nan_means, X_nan_variances, X_nan_count = _incremental_mean_and_var(
        X_nan,
        old_means,
        old_variances,
        old_weight_sum,
        sample_weight=sample_weights_X_nan,
    )

    # 断言计算出的包含 NaN 和不包含 NaN 的均值、方差和计数在允许误差范围内接近
    assert_allclose(X_nan_means, X_means)
    assert_allclose(X_nan_variances, X_variances)
    assert_allclose(X_nan_count, X_count)
def test_incremental_variance_update_formulas():
    # Test Youngs and Cramer incremental variance formulas.
    # 测试Youngs和Cramer的增量方差公式。
    # Doggie data from https://www.mathsisfun.com/data/standard-deviation.html
    A = np.array(
        [
            [600, 470, 170, 430, 300],
            [600, 470, 170, 430, 300],
            [600, 470, 170, 430, 300],
            [600, 470, 170, 430, 300],
        ]
    ).T
    # 转置矩阵A以使每列代表一个变量
    idx = 2
    X1 = A[:idx, :]  # 取前idx行作为X1
    X2 = A[idx:, :]  # 取从idx行开始的剩余部分作为X2

    old_means = X1.mean(axis=0)  # 计算X1每列的均值作为旧均值
    old_variances = X1.var(axis=0)  # 计算X1每列的方差作为旧方差
    old_sample_count = np.full(X1.shape[1], X1.shape[0], dtype=np.int32)  # 创建一个全为X1行数的整数数组作为旧样本计数

    # 调用增量均值和方差函数计算新的均值、方差和样本计数
    final_means, final_variances, final_count = _incremental_mean_and_var(
        X2, old_means, old_variances, old_sample_count
    )

    # 断言最终计算的均值与整体矩阵A的均值接近到小数点后第六位
    assert_almost_equal(final_means, A.mean(axis=0), 6)
    # 断言最终计算的方差与整体矩阵A的方差接近到小数点后第六位
    assert_almost_equal(final_variances, A.var(axis=0), 6)
    # 断言最终计算的样本计数等于整体矩阵A的行数
    assert_almost_equal(final_count, A.shape[0])


def test_incremental_mean_and_variance_ignore_nan():
    old_means = np.array([535.0, 535.0, 535.0, 535.0])
    old_variances = np.array([4225.0, 4225.0, 4225.0, 4225.0])
    old_sample_count = np.array([2, 2, 2, 2], dtype=np.int32)

    X = np.array([[170, 170, 170, 170], [430, 430, 430, 430], [300, 300, 300, 300]])

    X_nan = np.array(
        [
            [170, np.nan, 170, 170],
            [np.nan, 170, 430, 430],
            [430, 430, np.nan, 300],
            [300, 300, 300, np.nan],
        ]
    )

    # 调用增量均值和方差函数计算不考虑NaN值的均值、方差和样本计数
    X_means, X_variances, X_count = _incremental_mean_and_var(
        X, old_means, old_variances, old_sample_count
    )
    # 调用增量均值和方差函数计算考虑NaN值的均值、方差和样本计数
    X_nan_means, X_nan_variances, X_nan_count = _incremental_mean_and_var(
        X_nan, old_means, old_variances, old_sample_count
    )

    # 断言不考虑NaN值的均值与考虑NaN值的均值接近
    assert_allclose(X_nan_means, X_means)
    # 断言不考虑NaN值的方差与考虑NaN值的方差接近
    assert_allclose(X_nan_variances, X_variances)
    # 断言不考虑NaN值的样本计数与考虑NaN值的样本计数接近
    assert_allclose(X_nan_count, X_count)


@skip_if_32bit
def test_incremental_variance_numerical_stability():
    # Test Youngs and Cramer incremental variance formulas.
    # 测试Youngs和Cramer的增量方差公式。

    def np_var(A):
        return A.var(axis=0)

    # Naive one pass variance computation - not numerically stable
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    def one_pass_var(X):
        n = X.shape[0]
        exp_x2 = (X**2).sum(axis=0) / n
        expx_2 = (X.sum(axis=0) / n) ** 2
        return exp_x2 - expx_2

    # Two-pass algorithm, stable.
    # We use it as a benchmark. It is not an online algorithm
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Two-pass_algorithm
    def two_pass_var(X):
        mean = X.mean(axis=0)
        Y = X.copy()
        return np.mean((Y - mean) ** 2, axis=0)

    # Naive online implementation
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
    # This works only for chunks for size 1
    # 定义一个函数用于更新均值和方差，采用朴素方法
    def naive_mean_variance_update(x, last_mean, last_variance, last_sample_count):
        # 更新样本数量
        updated_sample_count = last_sample_count + 1
        # 计算样本比率
        samples_ratio = last_sample_count / float(updated_sample_count)
        # 更新均值
        updated_mean = x / updated_sample_count + last_mean * samples_ratio
        # 更新方差
        updated_variance = (
            last_variance * samples_ratio
            + (x - last_mean) * (x - updated_mean) / updated_sample_count
        )
        return updated_mean, updated_variance, updated_sample_count

    # 设置公差阈值
    tol = 200
    # 设置特征数和样本数
    n_features = 2
    n_samples = 10000
    # 定义两个不同的浮点数值
    x1 = np.array(1e8, dtype=np.float64)
    x2 = np.log(1e-5, dtype=np.float64)
    # 创建两个大小为 (5000, 2) 的数组 A0 和 A1，分别填充 x1 和 x2
    A0 = np.full((n_samples // 2, n_features), x1, dtype=np.float64)
    A1 = np.full((n_samples // 2, n_features), x2, dtype=np.float64)
    # 沿垂直方向堆叠数组 A0 和 A1，形成数组 A
    A = np.vstack((A0, A1))

    # 检验朴素方法与一次过程方法在计算方差时的误差情况
    # 对比数组 A 的方差的差异
    assert np.abs(np_var(A) - one_pass_var(A)).max() > tol

    # 在数组 A0 后面作为在线算法的起始点

    # 检验朴素实现方法的方差计算结果
    mean, var, n = A0[0, :], np.zeros(n_features), n_samples // 2
    # 遍历数组 A1 的每一行，并使用朴素方法更新均值和方差
    for i in range(A1.shape[0]):
        mean, var, n = naive_mean_variance_update(A1[i, :], mean, var, n)
    # 确保样本数量 n 与数组 A 的总行数相同
    assert n == A.shape[0]
    # 验证均值的稳定性
    assert np.abs(A.mean(axis=0) - mean).max() > 1e-6
    # 验证方差的稳定性
    assert np.abs(np_var(A) - var).max() > tol

    # 检验鲁棒实现方法的方差计算结果
    mean, var = A0[0, :], np.zeros(n_features)
    n = np.full(n_features, n_samples // 2, dtype=np.int32)
    # 遍历数组 A1 的每一行，并使用增量均值和方差更新函数
    for i in range(A1.shape[0]):
        mean, var, n = _incremental_mean_and_var(
            A1[i, :].reshape((1, A1.shape[1])), mean, var, n
        )
    # 确保数组 n 中的每个元素与数组 A 的总行数相同
    assert_array_equal(n, A.shape[0])
    # 验证数组 A 的均值与计算结果 mean 的近似一致性
    assert_array_almost_equal(A.mean(axis=0), mean)
    # 验证数组 A 的方差与计算结果 var 的差异在公差阈值内
    assert tol > np.abs(np_var(A) - var).max()
def test_incremental_variance_ddof():
    # Test that degrees of freedom parameter for calculations are correct.
    # 使用种子值1999初始化随机数生成器
    rng = np.random.RandomState(1999)
    # 生成一个大小为50x10的随机正态分布矩阵X
    X = rng.randn(50, 10)
    # 获取矩阵X的样本数和特征数
    n_samples, n_features = X.shape
    # 针对不同的批处理大小进行迭代
    for batch_size in [11, 20, 37]:
        # 计算步长数组，用于分割数据
        steps = np.arange(0, X.shape[0], batch_size)
        # 确保最后一步可以包含所有剩余的样本
        if steps[-1] != X.shape[0]:
            steps = np.hstack([steps, n_samples])

        # 遍历步长数组，依次处理数据批次
        for i, j in zip(steps[:-1], steps[1:]):
            # 从X中获取当前批次的数据
            batch = X[i:j, :]
            # 如果是第一批次，计算增量均值、增量方差和样本数
            if i == 0:
                incremental_means = batch.mean(axis=0)
                incremental_variances = batch.var(axis=0)
                incremental_count = batch.shape[0]
                sample_count = np.full(batch.shape[1], batch.shape[0], dtype=np.int32)
            else:
                # 否则调用增量计算函数计算均值、方差和样本数
                result = _incremental_mean_and_var(
                    batch, incremental_means, incremental_variances, sample_count
                )
                (incremental_means, incremental_variances, incremental_count) = result
                sample_count += batch.shape[0]

            # 计算整体数据截至当前步长的均值和方差
            calculated_means = np.mean(X[:j], axis=0)
            calculated_variances = np.var(X[:j], axis=0)
            # 断言增量计算的均值和方差与整体计算结果接近
            assert_almost_equal(incremental_means, calculated_means, 6)
            assert_almost_equal(incremental_variances, calculated_variances, 6)
            # 断言增量计算的样本数与累计样本数相等
            assert_array_equal(incremental_count, sample_count)


def test_vector_sign_flip():
    # Testing that sign flip is working & largest value has positive sign
    # 生成一个5x5的随机正态分布数据矩阵
    data = np.random.RandomState(36).randn(5, 5)
    # 计算每行绝对值最大值的索引
    max_abs_rows = np.argmax(np.abs(data), axis=1)
    # 对数据进行确定性的符号翻转操作
    data_flipped = _deterministic_vector_sign_flip(data)
    # 计算翻转后每行的最大值索引
    max_rows = np.argmax(data_flipped, axis=1)
    # 断言绝对值最大值索引和翻转后最大值索引相等
    assert_array_equal(max_abs_rows, max_rows)
    # 获取原始数据每行绝对值最大值的符号
    signs = np.sign(data[range(data.shape[0]), max_abs_rows])
    # 断言翻转后的数据乘以相应符号后与原始数据相等
    assert_array_equal(data, data_flipped * signs[:, np.newaxis])


def test_softmax():
    # 生成一个3x5的随机正态分布矩阵X
    rng = np.random.RandomState(0)
    X = rng.randn(3, 5)
    # 计算矩阵X的指数值
    exp_X = np.exp(X)
    # 计算指数值的和
    sum_exp_X = np.sum(exp_X, axis=1).reshape((-1, 1))
    # 断言softmax函数的输出与期望的结果相等
    assert_array_almost_equal(softmax(X), exp_X / sum_exp_X)


def test_stable_cumsum():
    # 断言稳定累加函数对于一维数组的输出与numpy.cumsum函数相等
    assert_array_equal(stable_cumsum([1, 2, 3]), np.cumsum([1, 2, 3]))
    # 生成一个长度为100000的随机数组r
    r = np.random.RandomState(0).rand(100000)
    # 使用 pytest 捕获 RuntimeWarning
    with pytest.warns(RuntimeWarning):
        # 断言稳定累加函数对数组r的输出与numpy.cumsum函数相等
        stable_cumsum(r, rtol=0, atol=0)

    # 测试多维数组的不同轴向的稳定累加
    A = np.random.RandomState(36).randint(1000, size=(5, 5, 5))
    assert_array_equal(stable_cumsum(A, axis=0), np.cumsum(A, axis=0))
    assert_array_equal(stable_cumsum(A, axis=1), np.cumsum(A, axis=1))
    assert_array_equal(stable_cumsum(A, axis=2), np.cumsum(A, axis=2))
def test_safe_sparse_dot_2d(A_container, B_container):
    # 使用种子0初始化随机数生成器
    rng = np.random.RandomState(0)

    # 创建随机的30x10矩阵A和10x20矩阵B
    A = rng.random_sample((30, 10))
    B = rng.random_sample((10, 20))
    # 计算期望的矩阵乘积
    expected = np.dot(A, B)

    # 将A和B分别封装到A_container和B_container中
    A = A_container(A)
    B = B_container(B)
    # 调用safe_sparse_dot函数计算实际的矩阵乘积，期望输出为密集矩阵
    actual = safe_sparse_dot(A, B, dense_output=True)

    # 使用assert_allclose函数验证actual和expected是否接近
    assert_allclose(actual, expected)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_safe_sparse_dot_nd(csr_container):
    # 使用种子0初始化随机数生成器
    rng = np.random.RandomState(0)

    # dense ND / sparse
    # 创建随机的2x3x4x5x6数组A和6x7矩阵B
    A = rng.random_sample((2, 3, 4, 5, 6))
    B = rng.random_sample((6, 7))
    # 计算期望的张量乘积
    expected = np.dot(A, B)
    # 将B转换为csr_container类型
    B = csr_container(B)
    # 调用safe_sparse_dot函数计算实际的张量乘积
    actual = safe_sparse_dot(A, B)
    # 使用assert_allclose函数验证actual和expected是否接近
    assert_allclose(actual, expected)

    # sparse / dense ND
    # 创建随机的2x3数组A和4x5x3x6数组B
    A = rng.random_sample((2, 3))
    B = rng.random_sample((4, 5, 3, 6))
    # 计算期望的张量乘积
    expected = np.dot(A, B)
    # 将A转换为csr_container类型
    A = csr_container(A)
    # 调用safe_sparse_dot函数计算实际的张量乘积
    actual = safe_sparse_dot(A, B)
    # 使用assert_allclose函数验证actual和expected是否接近
    assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "container",
    [np.array, *CSR_CONTAINERS],
    ids=["dense"] + [container.__name__ for container in CSR_CONTAINERS],
)
def test_safe_sparse_dot_2d_1d(container):
    # 使用种子0初始化随机数生成器
    rng = np.random.RandomState(0)
    # 创建随机的长度为10的向量B
    B = rng.random_sample((10))

    # 2D @ 1D
    # 创建随机的30x10矩阵A
    A = rng.random_sample((30, 10))
    # 计算期望的矩阵-向量乘积
    expected = np.dot(A, B)
    # 调用safe_sparse_dot函数计算实际的矩阵-向量乘积
    actual = safe_sparse_dot(container(A), B)
    # 使用assert_allclose函数验证actual和expected是否接近
    assert_allclose(actual, expected)

    # 1D @ 2D
    # 创建随机的10x30矩阵A
    A = rng.random_sample((10, 30))
    # 计算期望的向量-矩阵乘积
    expected = np.dot(B, A)
    # 调用safe_sparse_dot函数计算实际的向量-矩阵乘积
    actual = safe_sparse_dot(B, container(A))
    # 使用assert_allclose函数验证actual和expected是否接近
    assert_allclose(actual, expected)


@pytest.mark.parametrize("dense_output", [True, False])
def test_safe_sparse_dot_dense_output(dense_output):
    # 使用种子0初始化随机数生成器
    rng = np.random.RandomState(0)

    # 创建稀疏矩阵A和B，密度为0.1
    A = sparse.random(30, 10, density=0.1, random_state=rng)
    B = sparse.random(10, 20, density=0.1, random_state=rng)

    # 计算期望的稠密乘积
    expected = A.dot(B)
    # 调用safe_sparse_dot函数计算实际的乘积，根据参数dense_output决定输出稠密还是稀疏矩阵
    actual = safe_sparse_dot(A, B, dense_output=dense_output)

    # 使用assert语句验证actual的稀疏性质是否与dense_output参数一致
    assert sparse.issparse(actual) == (not dense_output)

    # 如果dense_output为True，则将expected转换为稠密数组
    if dense_output:
        expected = expected.toarray()
    # 使用assert_allclose_dense_sparse函数验证actual和expected是否接近
    assert_allclose_dense_sparse(actual, expected)


def test_approximate_mode():
    """Make sure sklearn.utils.extmath._approximate_mode returns valid
    results for cases where "class_counts * n_draws" is enough
    to overflow 32-bit signed integer.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/20774
    """
    # 创建一个包含两个元素的整数数组X，类型为int32
    X = np.array([99000, 1000], dtype=np.int32)
    # 调用_approximate_mode函数计算结果
    ret = _approximate_mode(class_counts=X, n_draws=25000, rng=0)

    # 预期结果为抽样的25%值，分别是24750和250
    assert_array_equal(ret, [24750, 250])
```