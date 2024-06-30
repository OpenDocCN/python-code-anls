# `D:\src\scipysrc\scikit-learn\sklearn\decomposition\tests\test_pca.py`

```
import re
import warnings

import numpy as np
import pytest
import scipy as sp
from numpy.testing import assert_array_equal

from sklearn import config_context, datasets
from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification, make_low_rank_matrix
from sklearn.decomposition import PCA
from sklearn.decomposition._pca import _assess_dimension, _infer_dimension
from sklearn.utils._array_api import (
    _atol_for_type,
    _convert_to_numpy,
    yield_namespace_device_dtype_combinations,
)
from sklearn.utils._array_api import device as array_device
from sklearn.utils._testing import _array_api_for_tests, assert_allclose
from sklearn.utils.estimator_checks import (
    _get_check_estimator_ids,
    check_array_api_input_and_values,
)
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS

iris = datasets.load_iris()
PCA_SOLVERS = ["full", "covariance_eigh", "arpack", "randomized", "auto"]

# `SPARSE_M` and `SPARSE_N` could be larger, but be aware:
# * SciPy's generation of random sparse matrix can be costly
# * A (SPARSE_M, SPARSE_N) dense array is allocated to compare against
SPARSE_M, SPARSE_N = 1000, 300  # arbitrary
SPARSE_MAX_COMPONENTS = min(SPARSE_M, SPARSE_N)


def _check_fitted_pca_close(pca1, pca2, rtol=1e-7, atol=1e-12):
    # Assert that PCA components are numerically close
    assert_allclose(pca1.components_, pca2.components_, rtol=rtol, atol=atol)
    assert_allclose(pca1.explained_variance_, pca2.explained_variance_, rtol=rtol, atol=atol)
    assert_allclose(pca1.singular_values_, pca2.singular_values_, rtol=rtol, atol=atol)
    assert_allclose(pca1.mean_, pca2.mean_, rtol=rtol, atol=atol)
    assert_allclose(pca1.noise_variance_, pca2.noise_variance_, rtol=rtol, atol=atol)

    # Assert other PCA attributes are identical
    assert pca1.n_components_ == pca2.n_components_
    assert pca1.n_samples_ == pca2.n_samples_
    assert pca1.n_features_in_ == pca2.n_features_in_


@pytest.mark.parametrize("svd_solver", PCA_SOLVERS)
@pytest.mark.parametrize("n_components", range(1, iris.data.shape[1]))
def test_pca(svd_solver, n_components):
    X = iris.data
    pca = PCA(n_components=n_components, svd_solver=svd_solver)

    # check the shape of fit.transform
    X_r = pca.fit(X).transform(X)
    assert X_r.shape[1] == n_components

    # check the equivalence of fit.transform and fit_transform
    X_r2 = pca.fit_transform(X)
    assert_allclose(X_r, X_r2)
    X_r = pca.transform(X)
    assert_allclose(X_r, X_r2)

    # Test get_covariance and get_precision
    cov = pca.get_covariance()
    precision = pca.get_precision()
    assert_allclose(np.dot(cov, precision), np.eye(X.shape[1]), atol=1e-12)


@pytest.mark.parametrize("density", [0.01, 0.1, 0.30])
@pytest.mark.parametrize("n_components", [1, 2, 10])
@pytest.mark.parametrize("sparse_container", CSR_CONTAINERS + CSC_CONTAINERS)
@pytest.mark.parametrize("svd_solver", ["arpack", "covariance_eigh"])
@pytest.mark.parametrize("scale", [1, 10, 100])
def test_pca_sparse(
    # 定义全局变量 global_random_seed, svd_solver, sparse_container, n_components, density, scale
    global_random_seed, svd_solver, sparse_container, n_components, density, scale
# 测试函数：验证稀疏和密集输入的结果是否相同
def test_pca_sparse_fit_transform(global_random_seed, sparse_container):
    random_state = np.random.default_rng(global_random_seed)
    
    # 创建稀疏矩阵 X，使用指定的稀疏容器和密度
    X = sparse_container(
        sp.sparse.random(
            SPARSE_M,
            SPARSE_N,
            random_state=random_state,
            density=0.01,
        )
    )
    
    # 创建另一个稀疏矩阵 X2，与 X 具有相同的参数
    X2 = sparse_container(
        sp.sparse.random(
            SPARSE_M,
            SPARSE_N,
            random_state=random_state,
            density=0.01,
        )
    )
    
    # 初始化 PCA 对象 pca_fit 和 pca_fit_transform，使用相同的随机种子和求解器
    pca_fit = PCA(n_components=10, svd_solver="arpack", random_state=global_random_seed)
    pca_fit_transform = PCA(
        n_components=10, svd_solver="arpack", random_state=global_random_seed
    )
    
    # 对 X 进行 PCA 拟合和转换
    pca_fit.fit(X)
    transformed_X = pca_fit_transform.fit_transform(X)
    
    # 验证拟合后的 PCA 对象的近似相等性
    _check_fitted_pca_close(pca_fit, pca_fit_transform)
    
    # 使用 assert_allclose 验证转换后的结果的近似性
    assert_allclose(transformed_X, pca_fit_transform.transform(X))
    assert_allclose(transformed_X, pca_fit.transform(X))
    
    # 使用 assert_allclose 验证对 X2 的转换结果的近似性
    assert_allclose(pca_fit.transform(X2), pca_fit_transform.transform(X2))


# 参数化测试函数：验证不同求解器下稀疏 PCA 的错误情况
@pytest.mark.parametrize("svd_solver", ["randomized", "full"])
@pytest.mark.parametrize("sparse_container", CSR_CONTAINERS + CSC_CONTAINERS)
def test_sparse_pca_solver_error(global_random_seed, svd_solver, sparse_container):
    random_state = np.random.RandomState(global_random_seed)
    
    # 创建稀疏矩阵 X，使用指定的稀疏容器和随机状态
    X = sparse_container(
        sp.sparse.random(
            SPARSE_M,
            SPARSE_N,
            random_state=random_state,
        )
    )
    # 使用PCA进行降维处理，设置主成分数量为30，并指定奇异值分解求解器为svd_solver
    pca = PCA(n_components=30, svd_solver=svd_solver)
    
    # 设置错误消息的模式，用于匹配预期的错误消息内容，当传递的求解器不支持稀疏输入时抛出TypeError异常
    error_msg_pattern = (
        'PCA only support sparse inputs with the "arpack" and "covariance_eigh"'
        f' solvers, while "{svd_solver}" was passed'
    )
    
    # 使用pytest来捕获预期的TypeError异常，并检查异常消息是否与error_msg_pattern匹配
    with pytest.raises(TypeError, match=error_msg_pattern):
        # 尝试拟合PCA模型到输入数据X
        pca.fit(X)
# 使用 pytest 的 mark.parametrize 装饰器，为 test_sparse_pca_auto_arpack_singluar_values_consistency 函数参数化 sparse_container
@pytest.mark.parametrize("sparse_container", CSR_CONTAINERS + CSC_CONTAINERS)
def test_sparse_pca_auto_arpack_singluar_values_consistency(
    global_random_seed, sparse_container
):
    """Check that "auto" and "arpack" solvers are equivalent for sparse inputs."""
    # 设定随机种子
    random_state = np.random.RandomState(global_random_seed)
    # 生成一个稀疏矩阵 X，使用给定的 sparse_container 生成方法，形状为 (SPARSE_M, SPARSE_N)，使用随机状态 random_state
    X = sparse_container(
        sp.sparse.random(
            SPARSE_M,
            SPARSE_N,
            random_state=random_state,
        )
    )
    # 使用 arpack 解算器拟合 PCA 模型，设定主成分数量为 10
    pca_arpack = PCA(n_components=10, svd_solver="arpack").fit(X)
    # 使用 auto 解算器拟合 PCA 模型，设定主成分数量为 10
    pca_auto = PCA(n_components=10, svd_solver="auto").fit(X)
    # 断言 arpack 和 auto 解算器计算出的奇异值（singular values）相近，相对误差小于 5e-3
    assert_allclose(pca_arpack.singular_values_, pca_auto.singular_values_, rtol=5e-3)


# 定义 test_no_empty_slice_warning 函数，用于测试在空数组上计算时是否避免了 numpy 警告
def test_no_empty_slice_warning():
    # 设定主成分数量为 10
    n_components = 10
    # 设定特征数量为主成分数量加 2
    n_features = n_components + 2  # anything > n_comps triggered it in 0.16
    # 生成一个随机数组 X，形状为 (n_components, n_features)，值在 [-1, 1] 之间均匀分布
    X = np.random.uniform(-1, 1, size=(n_components, n_features))
    # 创建 PCA 对象，设定主成分数量为 n_components
    pca = PCA(n_components=n_components)
    # 使用 warnings.catch_warnings 上下文管理器捕获 RuntimeWarning 警告
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        # 对数组 X 进行 PCA 拟合
        pca.fit(X)


# 使用 pytest 的 mark.parametrize 装饰器，为 test_whitening 函数参数化 copy 和 solver
@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("solver", PCA_SOLVERS)
def test_whitening(solver, copy):
    # 检查 PCA 输出是否具有单位方差
    rng = np.random.RandomState(0)
    n_samples = 100
    n_features = 80
    n_components = 30
    rank = 50

    # 创建一个低秩数据 X，具有相关特征
    X = np.dot(
        rng.randn(n_samples, rank),
        np.dot(np.diag(np.linspace(10.0, 1.0, rank)), rng.randn(rank, n_features)),
    )
    # 前 50 个特征的分量方差是其余 30 个特征的平均分量方差的 3 倍
    X[:, :50] *= 3

    assert X.shape == (n_samples, n_features)

    # 第一轮 50 个特征的分量方差高度变化：
    assert X.std(axis=0).std() > 43.8

    # 对数据进行白化（whiten），同时投影到较低维度子空间
    X_ = X.copy()  # 确保我们在迭代中保留原始数据。
    # 创建 PCA 对象，设定主成分数量为 n_components，启用白化，复制数据（根据 copy 参数），选择解算器（solver 参数）
    pca = PCA(
        n_components=n_components,
        whiten=True,
        copy=copy,
        svd_solver=solver,
        random_state=0,
        iterated_power=7,
    )
    # 测试 fit_transform 方法
    X_whitened = pca.fit_transform(X_.copy())
    # 断言白化后的数据形状为 (n_samples, n_components)
    assert X_whitened.shape == (n_samples, n_components)
    # 使用 transform 方法再次转换 X_
    X_whitened2 = pca.transform(X_)
    # 断言两次转换的结果非常接近，相对误差小于 5e-4
    assert_allclose(X_whitened, X_whitened2, rtol=5e-4)

    # 断言白化后的数据每个主成分的方差为 1
    assert_allclose(X_whitened.std(ddof=1, axis=0), np.ones(n_components))
    # 断言白化后的数据每个主成分的均值接近 0，绝对误差小于等于 1e-12
    assert_allclose(X_whitened.mean(axis=0), np.zeros(n_components), atol=1e-12)

    X_ = X.copy()
    # 创建 PCA 对象，设定主成分数量为 n_components，禁用白化，复制数据（根据 copy 参数），选择解算器（solver 参数）
    pca = PCA(
        n_components=n_components, whiten=False, copy=copy, svd_solver=solver
    ).fit(X_.copy())
    # 使用 PCA 对象转换数据 X_
    X_unwhitened = pca.transform(X_)
    # 断言未白化的数据形状为 (n_samples, n_components)
    assert X_unwhitened.shape == (n_samples, n_components)

    # 在这种情况下，输出的主成分仍然具有不同的方差
    assert X_unwhitened.std(axis=0).std() == pytest.approx(74.1, rel=1e-1)
    # 我们始终居中处理，因此不需要检查不居中的情况。
    we always center, so no test for non-centering.
@pytest.mark.parametrize(
    "other_svd_solver", sorted(list(set(PCA_SOLVERS) - {"full", "auto"}))
)
# 使用 pytest 的 parametrize 装饰器，为 other_svd_solver 参数生成多个测试参数，这些参数来自于 PCA_SOLVERS 集合中除去 "full" 和 "auto" 的元素，并按字母顺序排序

@pytest.mark.parametrize("data_shape", ["tall", "wide"])
# 继续使用 pytest 的 parametrize 装饰器，为 data_shape 参数生成两个测试参数： "tall" 和 "wide"

@pytest.mark.parametrize("rank_deficient", [False, True])
# 继续使用 pytest 的 parametrize 装饰器，为 rank_deficient 参数生成两个测试参数： False 和 True

@pytest.mark.parametrize("whiten", [False, True])
# 继续使用 pytest 的 parametrize 装饰器，为 whiten 参数生成两个测试参数： False 和 True

def test_pca_solver_equivalence(
    other_svd_solver,
    data_shape,
    rank_deficient,
    whiten,
    global_random_seed,
    global_dtype,
):
    # 根据 data_shape 参数设置不同的样本数和特征数
    if data_shape == "tall":
        n_samples, n_features = 100, 30
    else:
        n_samples, n_features = 30, 100

    n_samples_test = 10

    # 根据 rank_deficient 参数决定生成 X 矩阵的方式
    if rank_deficient:
        rng = np.random.default_rng(global_random_seed)
        rank = min(n_samples, n_features) // 2
        # 生成一个秩不足的数据矩阵 X
        X = rng.standard_normal(
            size=(n_samples + n_samples_test, rank)
        ) @ rng.standard_normal(size=(rank, n_features))
    else:
        # 使用 make_low_rank_matrix 函数生成近似全秩的数据矩阵 X
        X = make_low_rank_matrix(
            n_samples=n_samples + n_samples_test,
            n_features=n_features,
            tail_strength=0.5,
            random_state=global_random_seed,
        )
        rank = min(n_samples, n_features)  # 数据实际上是全秩的

    # 将 X 转换为指定的全局数据类型 global_dtype
    X = X.astype(global_dtype, copy=False)
    X_train, X_test = X[:n_samples], X[n_samples:]

    # 根据 global_dtype 设置容忍度和方差阈值
    if global_dtype == np.float32:
        tols = dict(atol=3e-2, rtol=1e-5)
        variance_threshold = 1e-5
    else:
        tols = dict(atol=1e-10, rtol=1e-12)
        variance_threshold = 1e-12

    extra_other_kwargs = {}

    # 根据 other_svd_solver 参数设置不同的额外关键字参数
    if other_svd_solver == "randomized":
        # 仅使用大量迭代次数来检查截断结果，以确保能恢复精确结果
        n_components = 10
        extra_other_kwargs = {"iterated_power": 50}
    elif other_svd_solver == "arpack":
        # 测试所有分量，除了 arpack 不能估计的最后一个分量
        n_components = np.minimum(n_samples, n_features) - 1
    else:
        # 测试所有分量，精确到高精度
        n_components = None

    # 创建两个 PCA 对象，一个使用 "full" svd_solver，一个使用 other_svd_solver
    pca_full = PCA(n_components=n_components, svd_solver="full", whiten=whiten)
    pca_other = PCA(
        n_components=n_components,
        svd_solver=other_svd_solver,
        whiten=whiten,
        random_state=global_random_seed,
        **extra_other_kwargs,
    )

    # 对训练数据 X_train 进行 PCA 变换并验证结果
    X_trans_full_train = pca_full.fit_transform(X_train)
    assert np.isfinite(X_trans_full_train).all()
    assert X_trans_full_train.dtype == global_dtype

    X_trans_other_train = pca_other.fit_transform(X_train)
    assert np.isfinite(X_trans_other_train).all()
    assert X_trans_other_train.dtype == global_dtype

    # 验证两个 PCA 对象的解释方差是否非负且接近
    assert (pca_full.explained_variance_ >= 0).all()
    assert_allclose(pca_full.explained_variance_, pca_other.explained_variance_, **tols)
    assert_allclose(
        pca_full.explained_variance_ratio_,
        pca_other.explained_variance_ratio_,
        **tols,
    )

    # 验证参考成分的有限性
    reference_components = pca_full.components_
    assert np.isfinite(reference_components).all()
    other_components = pca_other.components_
    # 获取另一个PCA对象的主成分

    assert np.isfinite(other_components).all()
    # 断言另一个PCA对象的所有主成分都是有限的（即不含NaN或无穷大值）

    # 对于某些n_components和数据分布的选择，可能有些主成分纯粹是噪声，我们在比较时忽略它们：
    stable = pca_full.explained_variance_ > variance_threshold
    # 稳定性指标：使用方差阈值判断哪些主成分是稳定的
    assert stable.sum() > 1
    # 断言稳定的主成分数量大于1
    assert_allclose(reference_components[stable], other_components[stable], **tols)
    # 使用给定的容差（**tols参数）检查稳定的主成分在两个PCA对象中的相似性

    # 因此，fit_transform的输出应该是相同的：
    assert_allclose(
        X_trans_other_train[:, stable], X_trans_full_train[:, stable], **tols
    )
    # 使用给定的容差检查在稳定主成分上训练数据的变换输出是否相似

    # 对于新数据的transform输出也应该类似（除了最后一个可能未确定的成分）：
    X_trans_full_test = pca_full.transform(X_test)
    assert np.isfinite(X_trans_full_test).all()
    assert X_trans_full_test.dtype == global_dtype
    X_trans_other_test = pca_other.transform(X_test)
    assert np.isfinite(X_trans_other_test).all()
    assert X_trans_other_test.dtype == global_dtype
    assert_allclose(X_trans_other_test[:, stable], X_trans_full_test[:, stable], **tols)
    # 使用给定的容差检查在稳定主成分上测试数据的变换输出是否相似

    # 检查两个解码器的逆变换重建是否兼容。
    X_recons_full_test = pca_full.inverse_transform(X_trans_full_test)
    assert np.isfinite(X_recons_full_test).all()
    assert X_recons_full_test.dtype == global_dtype
    X_recons_other_test = pca_other.inverse_transform(X_trans_other_test)
    assert np.isfinite(X_recons_other_test).all()
    assert X_recons_other_test.dtype == global_dtype

    if pca_full.components_.shape[0] == pca_full.components_.shape[1]:
        # 在这种情况下，模型应该学会了相同的可逆转换。因此它们都应该能够重构测试数据。
        assert_allclose(X_recons_full_test, X_test, **tols)
        assert_allclose(X_recons_other_test, X_test, **tols)
    elif pca_full.components_.shape[0] < rank:
        # 在没有噪声成分的情况下，两个模型应该能够重构原始数据的相同低秩近似。
        assert pca_full.explained_variance_.min() > variance_threshold
        assert_allclose(X_recons_full_test, X_recons_other_test, **tols)
    else:
        # 当n_features > n_samples并且n_components大于训练集的秩时，inverse_transform函数的输出是不确定的。
        # 我们只能检查经过另一轮transform后是否达到相同的稳定点：
        assert_allclose(
            pca_full.transform(X_recons_full_test)[:, stable],
            pca_other.transform(X_recons_other_test)[:, stable],
            **tols,
        )
@pytest.mark.parametrize(
    "X",
    [
        np.random.RandomState(0).randn(100, 80),  # 生成一个服从标准正态分布的随机矩阵
        datasets.make_classification(100, 80, n_informative=78, random_state=0)[0],  # 使用make_classification生成相关矩阵
        np.random.RandomState(0).randn(10, 100),  # 另一个随机矩阵，形状为10x100
    ],
    ids=["random-tall", "correlated-tall", "random-wide"],  # 为每个参数化的测试案例提供ID标识
)
@pytest.mark.parametrize("svd_solver", PCA_SOLVERS)  # 参数化测试，针对不同的PCA求解器进行测试
def test_pca_explained_variance_empirical(X, svd_solver):
    pca = PCA(n_components=2, svd_solver=svd_solver, random_state=0)  # 创建PCA对象，指定参数
    X_pca = pca.fit_transform(X)  # 对数据进行PCA降维处理
    assert_allclose(pca.explained_variance_, np.var(X_pca, ddof=1, axis=0))  # 断言PCA解释方差与变换后数据方差的一致性

    expected_result = np.linalg.eig(np.cov(X, rowvar=False))[0]  # 计算原始数据的协方差矩阵的特征值
    expected_result = sorted(expected_result, reverse=True)[:2]  # 取最大的两个特征值
    assert_allclose(pca.explained_variance_, expected_result, rtol=5e-3)  # 断言PCA解释方差与期望结果的一致性


@pytest.mark.parametrize("svd_solver", ["arpack", "randomized"])
def test_pca_singular_values_consistency(svd_solver):
    rng = np.random.RandomState(0)
    n_samples, n_features = 100, 80
    X = rng.randn(n_samples, n_features)  # 生成一个服从标准正态分布的随机矩阵

    pca_full = PCA(n_components=2, svd_solver="full", random_state=rng)  # 创建PCA对象，完全SVD求解
    pca_other = PCA(n_components=2, svd_solver=svd_solver, random_state=rng)  # 创建PCA对象，指定求解器

    pca_full.fit(X)  # 对数据进行PCA处理
    pca_other.fit(X)  # 对数据进行PCA处理

    assert_allclose(pca_full.singular_values_, pca_other.singular_values_, rtol=5e-3)  # 断言不同求解器计算的奇异值一致性


@pytest.mark.parametrize("svd_solver", PCA_SOLVERS)
def test_pca_singular_values(svd_solver):
    rng = np.random.RandomState(0)
    n_samples, n_features = 100, 80
    X = rng.randn(n_samples, n_features)  # 生成一个服从标准正态分布的随机矩阵

    pca = PCA(n_components=2, svd_solver=svd_solver, random_state=rng)  # 创建PCA对象，指定求解器和随机种子
    X_trans = pca.fit_transform(X)  # 对数据进行PCA降维处理

    # 比较与Frobenius范数的关系
    assert_allclose(
        np.sum(pca.singular_values_**2), np.linalg.norm(X_trans, "fro") ** 2
    )
    # 比较与得分向量的二范数关系
    assert_allclose(pca.singular_values_, np.sqrt(np.sum(X_trans**2, axis=0)))

    # 设置奇异值并查看返回结果
    n_samples, n_features = 100, 110
    X = rng.randn(n_samples, n_features)  # 生成另一个服从标准正态分布的随机矩阵

    pca = PCA(n_components=3, svd_solver=svd_solver, random_state=rng)  # 创建PCA对象，指定求解器和随机种子
    X_trans = pca.fit_transform(X)  # 对数据进行PCA降维处理
    X_trans /= np.sqrt(np.sum(X_trans**2, axis=0))  # 对转换后的数据进行归一化处理
    X_trans[:, 0] *= 3.142  # 缩放第一个主成分
    X_trans[:, 1] *= 2.718  # 缩放第二个主成分
    X_hat = np.dot(X_trans, pca.components_)  # 使用主成分重构数据
    pca.fit(X_hat)  # 对重构后的数据进行PCA处理
    assert_allclose(pca.singular_values_, [3.142, 2.718, 1.0])  # 断言奇异值与期望结果的一致性


@pytest.mark.parametrize("svd_solver", PCA_SOLVERS)
def test_pca_check_projection(svd_solver):
    # 测试数据投影的正确性
    rng = np.random.RandomState(0)
    n, p = 100, 3
    X = rng.randn(n, p) * 0.1  # 生成一个服从标准正态分布的随机矩阵，并进行缩放
    X[:10] += np.array([3, 4, 5])  # 改变部分数据的值

    Xt = 0.1 * rng.randn(1, p) + np.array([3, 4, 5])  # 生成一个测试数据，并进行缩放

    Yt = PCA(n_components=2, svd_solver=svd_solver).fit(X).transform(Xt)  # 对数据进行PCA处理，并将测试数据投影到主成分上
    Yt /= np.sqrt((Yt**2).sum())  # 对投影后的数据进行归一化处理

    assert_allclose(np.abs(Yt[0][0]), 1.0, rtol=5e-3)  # 断言投影数据的正确性


@pytest.mark.parametrize("svd_solver", PCA_SOLVERS)
def test_pca_check_projection_list(svd_solver):
    # 测试数据投影的正确性
    # 创建一个包含两个样本的二维数组 X
    X = [[1.0, 0.0], [0.0, 1.0]]
    
    # 使用 PCA 进行降维处理，设置主成分数量为 1，使用给定的 svd_solver 和随机种子
    pca = PCA(n_components=1, svd_solver=svd_solver, random_state=0)
    
    # 对输入数据 X 进行 PCA 变换，得到降维后的结果 X_trans
    X_trans = pca.fit_transform(X)
    
    # 断言降维后的数据形状应为 (2, 1)
    assert X_trans.shape, (2, 1)
    
    # 断言降维后数据的平均值应接近 0.00，允许误差为 1e-12
    assert_allclose(X_trans.mean(), 0.00, atol=1e-12)
    
    # 断言降维后数据的标准差应接近 0.71，相对误差率为 5e-3
    assert_allclose(X_trans.std(), 0.71, rtol=5e-3)
# 使用 pytest 的装饰器标记参数化测试，测试不同的 svd_solver 和 whiten 参数组合
@pytest.mark.parametrize("svd_solver", ["full", "arpack", "randomized"])
@pytest.mark.parametrize("whiten", [False, True])
def test_pca_inverse(svd_solver, whiten):
    # 测试数据的投影能否被反转
    rng = np.random.RandomState(0)
    n, p = 50, 3
    X = rng.randn(n, p)  # 创建具有球形数据分布的随机数据
    X[:, 1] *= 0.00001  # 将数据的中间分量变得相对较小
    X += [5, 4, 3]  # 给数据添加一个较大的均值

    # 验证：能够从变换后的信号中找回原始数据（因为数据几乎是 n_components 秩）
    pca = PCA(n_components=2, svd_solver=svd_solver, whiten=whiten).fit(X)
    Y = pca.transform(X)  # 对数据进行 PCA 变换
    Y_inverse = pca.inverse_transform(Y)  # 反转 PCA 变换
    assert_allclose(X, Y_inverse, rtol=5e-6)  # 断言原始数据与反转后数据的接近程度


@pytest.mark.parametrize(
    "data", [np.array([[0, 1, 0], [1, 0, 0]]), np.array([[0, 1, 0], [1, 0, 0]]).T]
)
@pytest.mark.parametrize(
    "svd_solver, n_components, err_msg",
    [
        ("arpack", 0, r"must be between 1 and min\(n_samples, n_features\)"),
        ("randomized", 0, r"must be between 1 and min\(n_samples, n_features\)"),
        ("arpack", 2, r"must be strictly less than min"),
        (
            "auto",
            3,
            (
                r"n_components=3 must be between 0 and min\(n_samples, "
                r"n_features\)=2 with svd_solver='full'"
            ),
        ),
    ],
)
def test_pca_validation(svd_solver, data, n_components, err_msg):
    # 确保针对特定 solver 的极端 n_components 参数输入会引发错误
    smallest_d = 2  # 最小维度
    pca_fitted = PCA(n_components, svd_solver=svd_solver)

    with pytest.raises(ValueError, match=err_msg):
        pca_fitted.fit(data)

    # arpack 的额外情况
    if svd_solver == "arpack":
        n_components = smallest_d

        err_msg = (
            "n_components={}L? must be strictly less than "
            r"min\(n_samples, n_features\)={}L? with "
            "svd_solver='arpack'".format(n_components, smallest_d)
        )
        with pytest.raises(ValueError, match=err_msg):
            PCA(n_components, svd_solver=svd_solver).fit(data)


@pytest.mark.parametrize(
    "solver, n_components_",
    [
        ("full", min(iris.data.shape)),
        ("arpack", min(iris.data.shape) - 1),
        ("randomized", min(iris.data.shape)),
    ],
)
@pytest.mark.parametrize("data", [iris.data, iris.data.T])
def test_n_components_none(data, solver, n_components_):
    pca = PCA(svd_solver=solver)
    pca.fit(data)
    assert pca.n_components_ == n_components_


@pytest.mark.parametrize("svd_solver", ["auto", "full"])
def test_n_components_mle(svd_solver):
    # 确保当 n_components == 'mle' 时对于 auto/full 不会引发错误
    rng = np.random.RandomState(0)
    n_samples, n_features = 600, 10
    X = rng.randn(n_samples, n_features)
    pca = PCA(n_components="mle", svd_solver=svd_solver)
    pca.fit(X)
    assert pca.n_components_ == 1
@pytest.mark.parametrize("svd_solver", ["arpack", "randomized"])
# 使用pytest的@parametrize装饰器，针对不同的svd_solver参数执行多次测试
def test_n_components_mle_error(svd_solver):
    # 确保当n_components == 'mle'时，对不支持的求解器会引发错误
    rng = np.random.RandomState(0)
    n_samples, n_features = 600, 10
    X = rng.randn(n_samples, n_features)
    # 创建PCA对象，指定n_components为'mle'，svd_solver根据参数svd_solver设置
    pca = PCA(n_components="mle", svd_solver=svd_solver)
    # 构建错误信息字符串
    err_msg = "n_components='mle' cannot be a string with svd_solver='{}'".format(
        svd_solver
    )
    # 使用pytest的raises断言，确保fit过程中会引发ValueError，并且错误信息符合预期
    with pytest.raises(ValueError, match=err_msg):
        pca.fit(X)


def test_pca_dim():
    # 检查自动化设置维度
    rng = np.random.RandomState(0)
    n, p = 100, 5
    X = rng.randn(n, p) * 0.1
    X[:10] += np.array([3, 4, 5, 1, 2])
    # 创建PCA对象，指定n_components为'mle'，svd_solver为'full'，并fit数据X
    pca = PCA(n_components="mle", svd_solver="full").fit(X)
    # 断言PCA对象的n_components属性为'mle'
    assert pca.n_components == "mle"
    # 断言PCA对象的n_components_属性为1，表示自动推断的主成分个数
    assert pca.n_components_ == 1


def test_infer_dim_1():
    # TODO: explain what this is testing
    # Or at least use explicit variable names...
    n, p = 1000, 5
    rng = np.random.RandomState(0)
    X = (
        rng.randn(n, p) * 0.1
        + rng.randn(n, 1) * np.array([3, 4, 5, 1, 2])
        + np.array([1, 0, 7, 4, 6])
    )
    # 创建PCA对象，指定n_components为p，svd_solver为'full'，并fit数据X
    pca = PCA(n_components=p, svd_solver="full")
    pca.fit(X)
    spect = pca.explained_variance_
    # 计算不同主成分数目下的评估结果，确保第二个主成分的评估值大于最大值与0.01*n的差值
    ll = np.array([_assess_dimension(spect, k, n) for k in range(1, p)])
    assert ll[1] > ll.max() - 0.01 * n


def test_infer_dim_2():
    # TODO: explain what this is testing
    # Or at least use explicit variable names...
    n, p = 1000, 5
    rng = np.random.RandomState(0)
    X = rng.randn(n, p) * 0.1
    X[:10] += np.array([3, 4, 5, 1, 2])
    X[10:20] += np.array([6, 0, 7, 2, -1])
    # 创建PCA对象，指定n_components为p，svd_solver为'full'，并fit数据X
    pca = PCA(n_components=p, svd_solver="full")
    pca.fit(X)
    spect = pca.explained_variance_
    # 断言通过_infer_dimension函数推断的维度大于1
    assert _infer_dimension(spect, n) > 1


def test_infer_dim_3():
    n, p = 100, 5
    rng = np.random.RandomState(0)
    X = rng.randn(n, p) * 0.1
    X[:10] += np.array([3, 4, 5, 1, 2])
    X[10:20] += np.array([6, 0, 7, 2, -1])
    X[30:40] += 2 * np.array([-1, 1, -1, 1, -1])
    # 创建PCA对象，指定n_components为p，svd_solver为'full'，并fit数据X
    pca = PCA(n_components=p, svd_solver="full")
    pca.fit(X)
    spect = pca.explained_variance_
    # 断言通过_infer_dimension函数推断的维度大于2
    assert _infer_dimension(spect, n) > 2


@pytest.mark.parametrize(
    "X, n_components, n_components_validated",
    [
        (iris.data, 0.95, 2),  # row > col
        (iris.data, 0.01, 1),  # row > col
        (np.random.RandomState(0).rand(5, 20), 0.5, 2),
    ],  # row < col
)
# 使用pytest的@parametrize装饰器，对不同的参数组合(X, n_components, n_components_validated)执行多次测试
def test_infer_dim_by_explained_variance(X, n_components, n_components_validated):
    # 创建PCA对象，指定n_components为n_components，svd_solver为'full'，并fit数据X
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(X)
    # 断言PCA对象的n_components属性接近于参数n_components
    assert pca.n_components == pytest.approx(n_components)
    # 断言PCA对象的n_components_属性等于n_components_validated，表示自动验证的主成分数目
    assert pca.n_components_ == n_components_validated


@pytest.mark.parametrize("svd_solver", PCA_SOLVERS)
# 使用pytest的@parametrize装饰器，对PCA_SOLVERS中的每种svd_solver参数执行测试
def test_pca_score(svd_solver):
    # 测试概率PCA评分是否得到合理的分数
    n, p = 1000, 3
    rng = np.random.RandomState(0)
    X = rng.randn(n, p) * 0.1 + np.array([3, 4, 5])
    # 创建一个PCA对象，指定要降到的维度为2，并选择奇异值分解的求解器
    pca = PCA(n_components=2, svd_solver=svd_solver)
    # 使用PCA对象拟合数据集X，学习主成分
    pca.fit(X)

    # 计算PCA模型在数据集X上的对数似然评分
    ll1 = pca.score(X)
    # 计算一个固定模型下的信息熵，用于后续的断言比较
    h = -0.5 * np.log(2 * np.pi * np.exp(1) * 0.1**2) * p
    # 断言 ll1 / h 的结果应接近1，相对误差容忍度为5e-2
    assert_allclose(ll1 / h, 1, rtol=5e-2)

    # 在一个扰动数据集上重新计算PCA模型的对数似然评分
    ll2 = pca.score(rng.randn(n, p) * 0.2 + np.array([3, 4, 5]))
    # 断言未扰动数据集上的对数似然评分 ll1 大于扰动数据集上的 ll2
    assert ll1 > ll2

    # 使用whiten参数设置为True创建一个新的PCA对象，同时指定降维后要白化数据
    pca = PCA(n_components=2, whiten=True, svd_solver=svd_solver)
    # 使用新的PCA对象拟合数据集X
    pca.fit(X)
    # 计算白化后的PCA模型在数据集X上的对数似然评分
    ll2 = pca.score(X)
    # 断言未白化数据集上的对数似然评分 ll1 大于白化后数据集上的 ll2
    assert ll1 > ll2
def test_pca_score3():
    # 检查概率主成分分析是否选择正确的模型

    # 设置样本数和特征数
    n, p = 200, 3
    # 创建随机数生成器
    rng = np.random.RandomState(0)
    # 生成具有特定均值和方差的数据集
    Xl = rng.randn(n, p) + rng.randn(n, 1) * np.array([3, 4, 5]) + np.array([1, 0, 7])
    Xt = rng.randn(n, p) + rng.randn(n, 1) * np.array([3, 4, 5]) + np.array([1, 0, 7])
    # 初始化用于保存对数似然值的数组
    ll = np.zeros(p)
    # 对每个主成分数进行迭代
    for k in range(p):
        # 创建 PCA 模型对象
        pca = PCA(n_components=k, svd_solver="full")
        # 拟合训练数据
        pca.fit(Xl)
        # 计算测试数据的对数似然值，并保存到数组中
        ll[k] = pca.score(Xt)

    # 断言选择的对数似然值最大的主成分索引为 1
    assert ll.argmax() == 1


@pytest.mark.parametrize("svd_solver", PCA_SOLVERS)
def test_pca_sanity_noise_variance(svd_solver):
    # 对 noise_variance_ 进行的合理性检查。详细信息请参见以下问题链接

    # 加载手写数字数据集
    X, _ = datasets.load_digits(return_X_y=True)
    # 创建 PCA 模型对象
    pca = PCA(n_components=30, svd_solver=svd_solver, random_state=0)
    # 拟合数据
    pca.fit(X)
    # 断言解释方差和噪声方差之差都大于等于 0
    assert np.all((pca.explained_variance_ - pca.noise_variance_) >= 0)


@pytest.mark.parametrize("svd_solver", ["arpack", "randomized"])
def test_pca_score_consistency_solvers(svd_solver):
    # 检查不同求解器之间得分的一致性

    # 加载手写数字数据集
    X, _ = datasets.load_digits(return_X_y=True)
    # 创建不同求解器下的 PCA 模型对象
    pca_full = PCA(n_components=30, svd_solver="full", random_state=0)
    pca_other = PCA(n_components=30, svd_solver=svd_solver, random_state=0)
    # 分别拟合数据
    pca_full.fit(X)
    pca_other.fit(X)
    # 断言不同求解器下的得分是接近的
    assert_allclose(pca_full.score(X), pca_other.score(X), rtol=5e-6)


# arpack raises ValueError for n_components == min(n_samples,  n_features)
@pytest.mark.parametrize("svd_solver", ["full", "randomized"])
def test_pca_zero_noise_variance_edge_cases(svd_solver):
    # 确保在边缘情况下 noise_variance_ 为 0
    # 当 n_components == min(n_samples, n_features) 时

    # 设置样本数和特征数
    n, p = 100, 3
    # 创建随机数生成器
    rng = np.random.RandomState(0)
    # 生成具有特定均值和方差的数据集
    X = rng.randn(n, p) * 0.1 + np.array([3, 4, 5])

    # 创建 PCA 模型对象
    pca = PCA(n_components=p, svd_solver=svd_solver)
    # 拟合数据
    pca.fit(X)
    # 断言噪声方差为 0
    assert pca.noise_variance_ == 0
    # 对 gh-12489 的非回归测试
    # 确保在 n_components == n_features < n_samples 时不会出现除以零的错误
    pca.score(X)

    # 对数据的转置进行 PCA 拟合
    pca.fit(X.T)
    # 再次断言噪声方差为 0
    assert pca.noise_variance_ == 0
    # 对 gh-12489 的非回归测试
    # 确保在 n_components == n_samples < n_features 时不会出现除以零的错误
    pca.score(X.T)


@pytest.mark.parametrize(
    "n_samples, n_features, n_components, expected_solver",
    [
        # case: n_samples < 10 * n_features and max(X.shape) <= 500 => 'full'
        (10, 50, 5, "full"),
        # case: n_samples > 10 * n_features and n_features < 500 => 'covariance_eigh'
        (1000, 50, 50, "covariance_eigh"),
        # case: n_components >= .8 * min(X.shape) => 'full'
        (1000, 500, 400, "full"),
        # n_components >= 1 and n_components < .8*min(X.shape) => 'randomized'
        (1000, 500, 10, "randomized"),
        # case: n_components in (0,1) => 'full'
        (1000, 500, 0.5, "full"),
    ],
# 定义一个测试函数，用于测试 PCA 类的自动选择 SVD 求解器
def test_pca_svd_solver_auto(n_samples, n_features, n_components, expected_solver):
    # 创建随机数据矩阵，用于 PCA 分析
    data = np.random.RandomState(0).uniform(size=(n_samples, n_features))
    # 创建 PCA 对象，自动选择求解器
    pca_auto = PCA(n_components=n_components, random_state=0)
    # 创建 PCA 对象，指定预期的求解器类型
    pca_test = PCA(
        n_components=n_components, svd_solver=expected_solver, random_state=0
    )
    # 对随机数据进行 PCA 拟合
    pca_auto.fit(data)
    # 断言自动选择的求解器与预期的求解器类型相符
    assert pca_auto._fit_svd_solver == expected_solver
    # 对比自动选择与指定求解器的主成分
    pca_test.fit(data)
    assert_allclose(pca_auto.components_, pca_test.components_)


# 使用参数化测试来测试 PCA 的确定性输出
@pytest.mark.parametrize("svd_solver", PCA_SOLVERS)
def test_pca_deterministic_output(svd_solver):
    # 创建随机数生成器
    rng = np.random.RandomState(0)
    # 创建一个随机矩阵
    X = rng.rand(10, 10)

    # 创建一个转换后的矩阵，用于存储每次 PCA 转换的结果
    transformed_X = np.zeros((20, 2))
    for i in range(20):
        # 创建 PCA 对象，指定求解器类型
        pca = PCA(n_components=2, svd_solver=svd_solver, random_state=rng)
        # 对随机数据进行 PCA 转换，并记录第一主成分
        transformed_X[i, :] = pca.fit_transform(X)[0]
    # 断言所有转换后的结果与第一次的结果一致
    assert_allclose(transformed_X, np.tile(transformed_X[0, :], 20).reshape(20, 2))


# 使用参数化测试来测试 PCA 类在不同数据类型下的保留性
@pytest.mark.parametrize("svd_solver", PCA_SOLVERS)
def test_pca_dtype_preservation(svd_solver, global_random_seed):
    # 调用函数检查 PCA 在浮点数类型下的数据保留性
    check_pca_float_dtype_preservation(svd_solver, global_random_seed)
    # 调用函数检查 PCA 在整数类型下的数据向上转型至双精度浮点型
    check_pca_int_dtype_upcast_to_double(svd_solver)


# 检查 PCA 在浮点数类型下的数据保留性
def check_pca_float_dtype_preservation(svd_solver, seed):
    # 确保 PCA 在输入为 float32 时不会提升数据类型至 float64
    X = np.random.RandomState(seed).rand(1000, 4)
    X_float64 = X.astype(np.float64, copy=False)
    X_float32 = X.astype(np.float32)

    # 创建 PCA 对象，指定求解器类型
    pca_64 = PCA(n_components=3, svd_solver=svd_solver, random_state=seed).fit(
        X_float64
    )
    pca_32 = PCA(n_components=3, svd_solver=svd_solver, random_state=seed).fit(
        X_float32
    )

    # 断言主成分的数据类型为 float64
    assert pca_64.components_.dtype == np.float64
    assert pca_32.components_.dtype == np.float32
    # 断言转换后数据的数据类型为对应的输入数据类型
    assert pca_64.transform(X_float64).dtype == np.float64
    assert pca_32.transform(X_float32).dtype == np.float32

    # 通过设定的绝对误差和相对误差来检查两者之间的接近程度
    assert_allclose(pca_64.components_, pca_32.components_, rtol=1e-3, atol=1e-3)


# 检查 PCA 在整数类型下的数据向上转型至双精度浮点型
def check_pca_int_dtype_upcast_to_double(svd_solver):
    # 确保所有整数类型在进行 PCA 操作时将被提升至 float64
    X_i64 = np.random.RandomState(0).randint(0, 1000, (1000, 4))
    X_i64 = X_i64.astype(np.int64, copy=False)
    X_i32 = X_i64.astype(np.int32, copy=False)

    # 创建 PCA 对象，指定求解器类型
    pca_64 = PCA(n_components=3, svd_solver=svd_solver, random_state=0).fit(X_i64)
    pca_32 = PCA(n_components=3, svd_solver=svd_solver, random_state=0).fit(X_i32)

    # 断言主成分的数据类型为 float64
    assert pca_64.components_.dtype == np.float64
    assert pca_32.components_.dtype == np.float64
    # 断言转换后数据的数据类型为 float64
    assert pca_64.transform(X_i64).dtype == np.float64
    assert pca_32.transform(X_i32).dtype == np.float64

    # 通过设定的绝对误差和相对误差来检查两者之间的接近程度
    assert_allclose(pca_64.components_, pca_32.components_, rtol=1e-4)


# 测试 PCA 主成分分析时主要解释方差比例是否符合预期
def test_pca_n_components_mostly_explained_variance_ratio():
    # 当 n_components 是第二高累积和
    # 载入鸢尾花数据集，并将特征矩阵 X 和标签 y 分别赋值
    X, y = load_iris(return_X_y=True)
    # 使用 PCA 对特征矩阵 X 进行拟合
    pca1 = PCA().fit(X, y)

    # 计算累积解释方差贡献率，并选择倒数第二个作为新的主成分数量
    n_components = pca1.explained_variance_ratio_.cumsum()[-2]
    # 使用新的主成分数量进行 PCA 拟合
    pca2 = PCA(n_components=n_components).fit(X, y)
    # 断言新的 PCA 模型的主成分数量等于特征矩阵 X 的列数
    assert pca2.n_components_ == X.shape[1]
def test_assess_dimension_bad_rank():
    # 当测试的秩不在 [1, n_features - 1] 范围内时，测试错误
    spectrum = np.array([1, 1e-30, 1e-30, 1e-30])
    n_samples = 10
    for rank in (0, 5):
        with pytest.raises(ValueError, match=r"should be in \[1, n_features - 1\]"):
            _assess_dimension(spectrum, rank, n_samples)


def test_small_eigenvalues_mle():
    # 测试与微小特征值相关的秩，其对数似然应为 -inf。推断的秩将为 1
    spectrum = np.array([1, 1e-30, 1e-30, 1e-30])

    assert _assess_dimension(spectrum, rank=1, n_samples=10) > -np.inf

    for rank in (2, 3):
        assert _assess_dimension(spectrum, rank, 10) == -np.inf

    assert _infer_dimension(spectrum, 10) == 1


def test_mle_redundant_data():
    # 使用病态数据 'mle' 测试：仅一个相关特征应该给出秩 1
    X, _ = datasets.make_classification(
        n_features=20,
        n_informative=1,
        n_repeated=18,
        n_redundant=1,
        n_clusters_per_class=1,
        random_state=42,
    )
    pca = PCA(n_components="mle").fit(X)
    assert pca.n_components_ == 1


def test_fit_mle_too_few_samples():
    # 测试当样本数小于特征数时，进行 'mle' 拟合时会触发错误
    X, _ = datasets.make_classification(n_samples=20, n_features=21, random_state=42)

    pca = PCA(n_components="mle", svd_solver="full")
    with pytest.raises(
        ValueError,
        match="n_components='mle' is only supported if n_samples >= n_features",
    ):
        pca.fit(X)


def test_mle_simple_case():
    # 针对问题 https://github.com/scikit-learn/scikit-learn/issues/16730 的非回归测试
    n_samples, n_dim = 1000, 10
    X = np.random.RandomState(0).randn(n_samples, n_dim)
    X[:, -1] = np.mean(X[:, :-1], axis=-1)  # 真实 X 维度为 ndim - 1
    pca_skl = PCA("mle", svd_solver="full")
    pca_skl.fit(X)
    assert pca_skl.n_components_ == n_dim - 1


def test_assess_dimesion_rank_one():
    # 确保 assess_dimension 在秩为 1 的矩阵上正常工作
    n_samples, n_features = 9, 6
    X = np.ones((n_samples, n_features))  # 秩为 1 的矩阵
    _, s, _ = np.linalg.svd(X, full_matrices=True)
    # 除了秩为 1，所有特征值为 0 或接近 0（浮点数）
    assert_allclose(s[1:], np.zeros(n_features - 1), atol=1e-12)

    assert np.isfinite(_assess_dimension(s, rank=1, n_samples=n_samples))
    for rank in range(2, n_features):
        assert _assess_dimension(s, rank, n_samples) == -np.inf


def test_pca_randomized_svd_n_oversamples():
    """检查当 `X` 具有大量特征时，暴露和设置 `n_oversamples` 是否会提供准确结果

    非回归测试：
    https://github.com/scikit-learn/scikit-learn/issues/20589
    """
    rng = np.random.RandomState(0)
    n_features = 100
    X = rng.randn(1_000, n_features)
    # 使用 PCA 进行降维分析，n_components 设置为 1 表示降维后的维度为 1
    # 使用 randomized 方法进行 SVD 分解，n_oversamples 参数设置为特征数以保证准确性
    # 使用随机种子 0 来确保结果的可重复性
    pca_randomized = PCA(
        n_components=1,
        svd_solver="randomized",
        n_oversamples=n_features,  # 强制设置为特征数，以确保结果准确
        random_state=0,
    ).fit(X)

    # 使用 PCA 进行降维分析，n_components 设置为 1 表示降维后的维度为 1
    # 使用 full 方法进行 SVD 分解
    pca_full = PCA(n_components=1, svd_solver="full").fit(X)

    # 使用 PCA 进行降维分析，n_components 设置为 1 表示降维后的维度为 1
    # 使用 arpack 方法进行 SVD 分解，使用随机种子 0 来确保结果的可重复性
    pca_arpack = PCA(n_components=1, svd_solver="arpack", random_state=0).fit(X)

    # 断言两个 PCA 对象的主成分的绝对值近似相等
    assert_allclose(np.abs(pca_full.components_), np.abs(pca_arpack.components_))

    # 断言两个 PCA 对象的主成分的绝对值近似相等
    assert_allclose(np.abs(pca_randomized.components_), np.abs(pca_arpack.components_))
# 定义测试函数，检查PCA的特征名称输出是否正确
def test_feature_names_out():
    # 使用PCA拟合鸢尾花数据，设置主成分数量为2
    pca = PCA(n_components=2).fit(iris.data)
    
    # 获取PCA对象中的特征名称输出
    names = pca.get_feature_names_out()
    # 断言特征名称应为 ["pca0", "pca1"]
    assert_array_equal([f"pca{i}" for i in range(2)], names)


# 使用参数化测试，检查PCA内部方差计算的准确性
@pytest.mark.parametrize("copy", [True, False])
def test_variance_correctness(copy):
    # 设定随机数种子
    rng = np.random.RandomState(0)
    # 生成服从标准正态分布的数据矩阵X，维度为1000x200
    X = rng.randn(1000, 200)
    # 使用PCA拟合数据X
    pca = PCA().fit(X)
    # 计算PCA对象的方差与方差比率之比
    pca_var = pca.explained_variance_ / pca.explained_variance_ratio_
    # 计算数据矩阵X的总体方差
    true_var = np.var(X, ddof=1, axis=0).sum()
    # 使用 np.testing.assert_allclose 函数断言pca_var与true_var的近似程度
    np.testing.assert_allclose(pca_var, true_var)


# 检查数组API的精度获取函数
def check_array_api_get_precision(name, estimator, array_namespace, device, dtype_name):
    # 根据数组API命名空间和设备类型选择对应的扩展库
    xp = _array_api_for_tests(array_namespace, device)
    # 将鸢尾花数据转换为指定的数据类型并使用所选的设备
    iris_np = iris.data.astype(dtype_name)
    iris_xp = xp.asarray(iris_np, device=device)

    # 使用估计器拟合鸢尾花数据（numpy版本）
    estimator.fit(iris_np)
    # 获取numpy版本的精度和协方差
    precision_np = estimator.get_precision()
    covariance_np = estimator.get_covariance()

    # 设定相对误差限
    rtol = 2e-4 if iris_np.dtype == "float32" else 2e-7
    # 使用数组API调度上下文，允许在指定设备上使用对应的扩展库
    with config_context(array_api_dispatch=True):
        # 克隆估计器对象并使用数组API版本拟合鸢尾花数据
        estimator_xp = clone(estimator).fit(iris_xp)
        # 获取扩展库版本的精度
        precision_xp = estimator_xp.get_precision()
        # 断言扩展库版本的精度形状为(4, 4)，与数据类型一致
        assert precision_xp.shape == (4, 4)
        assert precision_xp.dtype == iris_xp.dtype

        # 将扩展库版本的精度转换为numpy数组并与numpy版本进行比较，验证精度的准确性
        assert_allclose(
            _convert_to_numpy(precision_xp, xp=xp),
            precision_np,
            rtol=rtol,
            atol=_atol_for_type(dtype_name),
        )
        # 获取扩展库版本的协方差
        covariance_xp = estimator_xp.get_covariance()
        # 断言扩展库版本的协方差形状为(4, 4)，与数据类型一致
        assert covariance_xp.shape == (4, 4)
        assert covariance_xp.dtype == iris_xp.dtype

        # 将扩展库版本的协方差转换为numpy数组并与numpy版本进行比较，验证协方差的准确性
        assert_allclose(
            _convert_to_numpy(covariance_xp, xp=xp),
            covariance_np,
            rtol=rtol,
            atol=_atol_for_type(dtype_name),
        )


# 参数化测试，检查PCA在数组API上的兼容性
@pytest.mark.parametrize(
    "array_namespace, device, dtype_name", yield_namespace_device_dtype_combinations()
)
@pytest.mark.parametrize(
    "check",
    [check_array_api_input_and_values, check_array_api_get_precision],
    ids=_get_check_estimator_ids,
)
@pytest.mark.parametrize(
    "estimator",
    [
        PCA(n_components=2, svd_solver="full"),
        PCA(n_components=2, svd_solver="full", whiten=True),
        PCA(n_components=0.1, svd_solver="full", whiten=True),
        PCA(n_components=2, svd_solver="covariance_eigh"),
        PCA(n_components=2, svd_solver="covariance_eigh", whiten=True),
        PCA(
            n_components=2,
            svd_solver="randomized",
            power_iteration_normalizer="QR",
            random_state=0,
        ),
    ],
    ids=_get_check_estimator_ids,
)
# 测试PCA在数组API上的兼容性
def test_pca_array_api_compliance(
    estimator, check, array_namespace, device, dtype_name
):
    # 获取估计器的类名
    name = estimator.__class__.__name__
    # 调用测试函数进行检查
    check(name, estimator, array_namespace, device=device, dtype_name=dtype_name)
    # 使用 yield_namespace_device_dtype_combinations() 生成器函数生成的三元组作为参数传入
    "array_namespace, device, dtype_name", yield_namespace_device_dtype_combinations()
# 使用 pytest 提供的装饰器标记此函数为测试函数，参数化测试数据将由 pytest 根据参数值生成多个测试实例
@pytest.mark.parametrize(
    # 定义参数 "check"，其值为 check_array_api_get_precision 函数的引用
    "check",
    [check_array_api_get_precision],
    # 定义参数化测试实例的标识符，使用 _get_check_estimator_ids 函数生成
    ids=_get_check_estimator_ids,
)
# 参数化测试实例，用不同的 PCA 实例来测试
@pytest.mark.parametrize(
    # 定义参数 "estimator"，值为包含单个 PCA 实例的列表
    "estimator",
    [
        # 创建一个 PCA 实例，使用 "mle" 作为 n_components 参数，"full" 作为 svd_solver 参数
        # 这里注释指出，使用 check_array_api_input_and_values 进行检查会因为噪声组件的舍入误差而失败，
        # 即使检查 `components_` 的形状也有问题，因为组件数量取决于 mle 算法的截断阈值，可能依赖于设备特定的舍入误差。
        PCA(n_components="mle", svd_solver="full"),
    ],
    # 定义参数化测试实例的标识符，使用 _get_check_estimator_ids 函数生成
    ids=_get_check_estimator_ids,
)
# 定义测试函数 test_pca_mle_array_api_compliance，接受多个参数：estimator, check, array_namespace, device, dtype_name
def test_pca_mle_array_api_compliance(
    estimator, check, array_namespace, device, dtype_name
):
    # 获取 estimator 的类名
    name = estimator.__class__.__name__
    # 调用 check 函数，传递 estimator, array_namespace, device, dtype_name 参数进行检查
    check(name, estimator, array_namespace, device=device, dtype_name=dtype_name)

    # 使用特定于 PCA mle-trimmed 组件的通用 check_array_api_input 检查器的简化变体
    xp = _array_api_for_tests(array_namespace, device)

    # 生成分类数据 X 和 y
    X, y = make_classification(random_state=42)
    # 将 X 转换为指定的 dtype_name 类型，in-place 操作
    X = X.astype(dtype_name, copy=False)
    # 获取 X 的类型相关的 atol
    atol = _atol_for_type(X.dtype)

    # 克隆 estimator
    est = clone(estimator)

    # 使用 xp 将 X 和 y 转换为特定设备上的数组
    X_xp = xp.asarray(X, device=device)
    y_xp = xp.asarray(y, device=device)

    # 使用 X 和 y 拟合 estimator
    est.fit(X, y)

    # 获取 components_ 和 explained_variance_ 属性
    components_np = est.components_
    explained_variance_np = est.explained_variance_

    # 克隆 estimator
    est_xp = clone(est)
    # 使用 array_api_dispatch 上下文配置，使用 xp 执行拟合操作
    with config_context(array_api_dispatch=True):
        est_xp.fit(X_xp, y_xp)
        # 获取 components_ 和 explained_variance_ 的 xp 版本
        components_xp = est_xp.components_
        # 断言 components_xp 的设备与 X_xp 相同
        assert array_device(components_xp) == array_device(X_xp)
        # 将 components_xp 转换为 numpy 数组
        components_xp_np = _convert_to_numpy(components_xp, xp=xp)

        # 获取 explained_variance_xp
        explained_variance_xp = est_xp.explained_variance_
        # 断言 explained_variance_xp 的设备与 X_xp 相同
        assert array_device(explained_variance_xp) == array_device(X_xp)
        # 将 explained_variance_xp 转换为 numpy 数组
        explained_variance_xp_np = _convert_to_numpy(explained_variance_xp, xp=xp)

    # 断言 components_xp_np 的 dtype 与 components_np 相同
    assert components_xp_np.dtype == components_np.dtype
    # 断言 components_xp_np 的列数与 components_np 相同
    assert components_xp_np.shape[1] == components_np.shape[1]
    # 断言 explained_variance_xp_np 的 dtype 与 explained_variance_np 相同
    assert explained_variance_xp_np.dtype == explained_variance_np.dtype

    # 检查共同组件的解释方差值是否匹配
    min_components = min(components_xp_np.shape[0], components_np.shape[0])
    # 使用 assert_allclose 检查 explained_variance_xp_np 和 explained_variance_np 的前 min_components 个元素是否在给定的 atol 范围内相等
    assert_allclose(
        explained_variance_xp_np[:min_components],
        explained_variance_np[:min_components],
        atol=atol,
    )

    # 如果 components_xp_np 的行数与 components_np 不同，检查修剪组件的解释方差是否非常小
    if components_xp_np.shape[0] != components_np.shape[0]:
        # 获取 reference_variance
        reference_variance = explained_variance_np[-1]
        # 获取额外的解释方差值
        extra_variance_np = explained_variance_np[min_components:]
        extra_variance_xp_np = explained_variance_xp_np[min_components:]
        # 断言额外的解释方差值与 reference_variance 之间的绝对差小于 atol
        assert all(np.abs(extra_variance_np - reference_variance) < atol)
        assert all(np.abs(extra_variance_xp_np - reference_variance) < atol)
    # 导入array_api_compat模块，如果导入失败则跳过该测试
    pytest.importorskip("array_api_compat")
    # 导入array_api_strict模块，如果导入失败则跳过该测试，并将导入的模块赋值给xp变量
    xp = pytest.importorskip("array_api_strict")
    # 将iris.data转换为Array API兼容的数组表示
    iris_xp = xp.asarray(iris.data)

    # 使用PCA进行降维，设置参数n_components为2，svd_solver为"arpack"，随机状态为0
    pca = PCA(n_components=2, svd_solver="arpack", random_state=0)
    # 构建预期的错误信息，用于检查是否抛出特定异常
    expected_msg = re.escape(
        "PCA with svd_solver='arpack' is not supported for Array API inputs."
    )
    # 检查是否抛出值错误异常，并匹配预期的错误信息
    with pytest.raises(ValueError, match=expected_msg):
        # 在array_api_dispatch上下文中，执行pca.fit(iris_xp)
        with config_context(array_api_dispatch=True):
            pca.fit(iris_xp)

    # 设置PCA对象的参数，svd_solver为"randomized"，power_iteration_normalizer为"LU"
    pca.set_params(svd_solver="randomized", power_iteration_normalizer="LU")
    # 构建预期的错误信息，用于检查是否抛出特定异常
    expected_msg = re.escape(
        "Array API does not support LU factorization. Set"
        " `power_iteration_normalizer='QR'` instead."
    )
    # 检查是否抛出值错误异常，并匹配预期的错误信息
    with pytest.raises(ValueError, match=expected_msg):
        # 在array_api_dispatch上下文中，执行pca.fit(iris_xp)
        with config_context(array_api_dispatch=True):
            pca.fit(iris_xp)

    # 设置PCA对象的参数，svd_solver为"randomized"，power_iteration_normalizer为"auto"
    pca.set_params(svd_solver="randomized", power_iteration_normalizer="auto")
    # 构建预期的警告信息，用于检查是否抛出特定的用户警告
    expected_msg = re.escape(
        "Array API does not support LU factorization, falling back to QR instead. Set"
        " `power_iteration_normalizer='QR'` explicitly to silence this warning."
    )
    # 检查是否抛出用户警告，并匹配预期的警告信息
    with pytest.warns(UserWarning, match=expected_msg):
        # 在array_api_dispatch上下文中，执行pca.fit(iris_xp)
        with config_context(array_api_dispatch=True):
            pca.fit(iris_xp)
```