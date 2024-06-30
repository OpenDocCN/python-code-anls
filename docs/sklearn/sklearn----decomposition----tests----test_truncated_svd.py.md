# `D:\src\scipysrc\scikit-learn\sklearn\decomposition\tests\test_truncated_svd.py`

```
"""Test truncated SVD transformer."""

import numpy as np
import pytest
import scipy.sparse as sp

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, assert_array_less

# 可用的 SVD 求解器列表
SVD_SOLVERS = ["arpack", "randomized"]


@pytest.fixture(scope="module")
def X_sparse():
    # 创建一个类似小型 tf-idf 矩阵的稀疏矩阵 X
    rng = check_random_state(42)
    X = sp.random(60, 55, density=0.2, format="csr", random_state=rng)
    X.data[:] = 1 + np.log(X.data)
    return X


@pytest.mark.parametrize("solver", ["randomized"])
@pytest.mark.parametrize("kind", ("dense", "sparse"))
def test_solvers(X_sparse, solver, kind):
    X = X_sparse if kind == "sparse" else X_sparse.toarray()
    # 使用 arpack 算法进行 SVD 分解
    svd_a = TruncatedSVD(30, algorithm="arpack")
    # 使用指定的 solver 和参数进行 SVD 分解
    svd = TruncatedSVD(30, algorithm=solver, random_state=42, n_oversamples=100)

    # 对 X 应用 SVD 分解，并截取前6个主成分
    Xa = svd_a.fit_transform(X)[:, :6]
    Xr = svd.fit_transform(X)[:, :6]
    assert_allclose(Xa, Xr, rtol=2e-3)

    # 获取 SVD 分解后的主成分，取绝对值
    comp_a = np.abs(svd_a.components_)
    comp = np.abs(svd.components_)
    # 检查前9个主成分的接近程度
    assert_allclose(comp_a[:9], comp[:9], rtol=1e-3)
    # 检查后面的主成分的接近程度，允许更大的绝对误差
    assert_allclose(comp_a[9:], comp[9:], atol=1e-2)


@pytest.mark.parametrize("n_components", (10, 25, 41, 55))
def test_attributes(n_components, X_sparse):
    n_features = X_sparse.shape[1]
    # 测试 TruncatedSVD 的属性
    tsvd = TruncatedSVD(n_components).fit(X_sparse)
    assert tsvd.n_components == n_components
    assert tsvd.components_.shape == (n_components, n_features)


@pytest.mark.parametrize(
    "algorithm, n_components",
    [
        ("arpack", 55),
        ("arpack", 56),
        ("randomized", 56),
    ],
)
def test_too_many_components(X_sparse, algorithm, n_components):
    # 测试当指定的组件数过多时是否引发 ValueError
    tsvd = TruncatedSVD(n_components=n_components, algorithm=algorithm)
    with pytest.raises(ValueError):
        tsvd.fit(X_sparse)


@pytest.mark.parametrize("fmt", ("array", "csr", "csc", "coo", "lil"))
def test_sparse_formats(fmt, X_sparse):
    n_samples = X_sparse.shape[0]
    # 根据不同的格式 fmt 转换稀疏矩阵 Xfmt
    Xfmt = X_sparse.toarray() if fmt == "dense" else getattr(X_sparse, "to" + fmt)()
    # 使用 TruncatedSVD 进行拟合和转换
    tsvd = TruncatedSVD(n_components=11)
    Xtrans = tsvd.fit_transform(Xfmt)
    assert Xtrans.shape == (n_samples, 11)
    Xtrans = tsvd.transform(Xfmt)
    assert Xtrans.shape == (n_samples, 11)


@pytest.mark.parametrize("algo", SVD_SOLVERS)
def test_inverse_transform(algo, X_sparse):
    # 测试逆转换是否能够近似恢复原始数据
    tsvd = TruncatedSVD(n_components=52, random_state=42, algorithm=algo)
    Xt = tsvd.fit_transform(X_sparse)
    Xinv = tsvd.inverse_transform(Xt)
    assert_allclose(Xinv, X_sparse.toarray(), rtol=1e-1, atol=2e-1)


def test_integers(X_sparse):
    n_samples = X_sparse.shape[0]
    # 将稀疏矩阵 X 转换为整数类型
    Xint = X_sparse.astype(np.int64)
    # 使用 TruncatedSVD 进行拟合和转换
    tsvd = TruncatedSVD(n_components=6)
    Xtrans = tsvd.fit_transform(Xint)
    # 确保Xtrans的形状与指定的(n_samples, tsvd.n_components)相匹配
    assert Xtrans.shape == (n_samples, tsvd.n_components)
# 使用 pytest 的参数化装饰器，为以下测试用例创建多组参数组合
@pytest.mark.parametrize("kind", ("dense", "sparse"))
@pytest.mark.parametrize("n_components", [10, 20])
@pytest.mark.parametrize("solver", SVD_SOLVERS)
def test_explained_variance(X_sparse, kind, n_components, solver):
    # 根据稀疏性选择输入数据类型
    X = X_sparse if kind == "sparse" else X_sparse.toarray()
    
    # 使用截断SVD进行降维，设置算法和组件数
    svd = TruncatedSVD(n_components, algorithm=solver)
    X_tr = svd.fit_transform(X)
    
    # 断言所有的解释方差比例大于0
    assert_array_less(0.0, svd.explained_variance_ratio_)
    
    # 断言总解释方差比例小于1
    assert_array_less(svd.explained_variance_ratio_.sum(), 1.0)
    
    # 测试解释方差比例是否正确
    total_variance = np.var(X_sparse.toarray(), axis=0).sum()
    variances = np.var(X_tr, axis=0)
    true_explained_variance_ratio = variances / total_variance
    
    # 断言所有解释方差比例与预期值接近
    assert_allclose(
        svd.explained_variance_ratio_,
        true_explained_variance_ratio,
    )


# 使用 pytest 的参数化装饰器，为以下测试用例创建多组参数组合
@pytest.mark.parametrize("kind", ("dense", "sparse"))
@pytest.mark.parametrize("solver", SVD_SOLVERS)
def test_explained_variance_components_10_20(X_sparse, kind, solver):
    # 根据稀疏性选择输入数据类型
    X = X_sparse if kind == "sparse" else X_sparse.toarray()
    
    # 创建两个截断SVD对象，分别计算10和20个组件
    svd_10 = TruncatedSVD(10, algorithm=solver, n_iter=10).fit(X)
    svd_20 = TruncatedSVD(20, algorithm=solver, n_iter=10).fit(X)
    
    # 断言第一个组件相等
    assert_allclose(
        svd_10.explained_variance_ratio_,
        svd_20.explained_variance_ratio_[:10],
        rtol=5e-3,
    )
    
    # 断言20个组件的解释方差比10个组件更高
    assert (
        svd_20.explained_variance_ratio_.sum() > svd_10.explained_variance_ratio_.sum()
    )


# 使用 pytest 的参数化装饰器，为以下测试用例创建多组参数组合
@pytest.mark.parametrize("solver", SVD_SOLVERS)
def test_singular_values_consistency(solver):
    # 检查截断SVD输出的正确奇异值
    rng = np.random.RandomState(0)
    n_samples, n_features = 100, 80
    X = rng.randn(n_samples, n_features)
    
    # 使用截断SVD进行降维，设置算法和随机状态
    pca = TruncatedSVD(n_components=2, algorithm=solver, random_state=rng).fit(X)
    
    # 与Frobenius范数比较
    X_pca = pca.transform(X)
    assert_allclose(
        np.sum(pca.singular_values_**2.0),
        np.linalg.norm(X_pca, "fro") ** 2.0,
        rtol=1e-2,
    )
    
    # 与得分向量的2-范数比较
    assert_allclose(
        pca.singular_values_, np.sqrt(np.sum(X_pca**2.0, axis=0)), rtol=1e-2
    )


# 使用 pytest 的参数化装饰器，为以下测试用例创建多组参数组合
@pytest.mark.parametrize("solver", SVD_SOLVERS)
def test_singular_values_expected(solver):
    # 设置奇异值并检查返回结果
    rng = np.random.RandomState(0)
    n_samples = 100
    n_features = 110
    X = rng.randn(n_samples, n_features)
    
    # 使用截断SVD进行降维，设置算法和随机状态
    pca = TruncatedSVD(n_components=3, algorithm=solver, random_state=rng)
    X_pca = pca.fit_transform(X)
    
    # 标准化X_pca并调整第一、第二列
    X_pca /= np.sqrt(np.sum(X_pca**2.0, axis=0))
    X_pca[:, 0] *= 3.142
    X_pca[:, 1] *= 2.718
    
    # 用新的X_pca与主成分计算重构X，并重新拟合截断SVD
    X_hat_pca = np.dot(X_pca, pca.components_)
    pca.fit(X_hat_pca)
    
    # 断言所有奇异值与预期值接近
    assert_allclose(pca.singular_values_, [3.142, 2.718, 1.0], rtol=1e-14)


这段代码使用了pytest的参数化装饰器来创建多个测试用例，每个测试用例都测试了TruncatedSVD对象的不同行为和属性。每个函数都包含了多个断言来验证其结果的正确性。
# TruncatedSVD 应该与 PCA 在中心化数据上产生相同的结果

def test_truncated_svd_eq_pca(X_sparse):
    # 将稀疏矩阵转换为稠密矩阵
    X_dense = X_sparse.toarray()

    # 计算中心化后的数据
    X_c = X_dense - X_dense.mean(axis=0)

    # 设定参数字典
    params = dict(n_components=10, random_state=42)

    # 创建 TruncatedSVD 对象
    svd = TruncatedSVD(algorithm="arpack", **params)
    
    # 创建 PCA 对象
    pca = PCA(svd_solver="arpack", **params)

    # 使用 TruncatedSVD 对象拟合并转换数据
    Xt_svd = svd.fit_transform(X_c)

    # 使用 PCA 对象拟合并转换数据
    Xt_pca = pca.fit_transform(X_c)

    # 断言 TruncatedSVD 和 PCA 转换后的结果非常接近
    assert_allclose(Xt_svd, Xt_pca, rtol=1e-9)
    
    # 断言 PCA 的均值接近于零
    assert_allclose(pca.mean_, 0, atol=1e-9)
    
    # 断言 TruncatedSVD 和 PCA 的主成分非常接近
    assert_allclose(svd.components_, pca.components_)


@pytest.mark.parametrize(
    "algorithm, tol", [("randomized", 0.0), ("arpack", 1e-6), ("arpack", 0.0)]
)
@pytest.mark.parametrize("kind", ("dense", "sparse"))
def test_fit_transform(X_sparse, algorithm, tol, kind):
    # fit_transform(X) 应该等于 fit(X).transform(X)

    # 根据稀疏或稠密类型，选择对应的数据类型
    X = X_sparse if kind == "sparse" else X_sparse.toarray()

    # 创建 TruncatedSVD 对象，设定参数
    svd = TruncatedSVD(
        n_components=5, n_iter=7, random_state=42, algorithm=algorithm, tol=tol
    )

    # 第一种方法：使用 fit_transform 进行拟合和转换
    X_transformed_1 = svd.fit_transform(X)

    # 第二种方法：先拟合后转换
    X_transformed_2 = svd.fit(X).transform(X)

    # 断言两种方法得到的结果非常接近
    assert_allclose(X_transformed_1, X_transformed_2)
```