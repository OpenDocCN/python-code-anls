# `D:\src\scipysrc\scikit-learn\sklearn\decomposition\tests\test_incremental_pca.py`

```
# 导入警告模块，用于管理警告信息的显示
import warnings

# 导入 NumPy 库并重命名为 np
import numpy as np

# 导入 pytest 库，用于编写和运行测试用例
import pytest

# 导入相关的测试工具函数和断言函数
from numpy.testing import assert_allclose, assert_array_equal

# 导入 sklearn 中的数据集模块和 PCA 降维算法模块
from sklearn import datasets
from sklearn.decomposition import PCA, IncrementalPCA

# 导入 sklearn 中的测试工具函数和容器修复函数
from sklearn.utils._testing import (
    assert_allclose_dense_sparse,
    assert_almost_equal,
    assert_array_almost_equal,
)
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS

# 加载鸢尾花数据集
iris = datasets.load_iris()


def test_incremental_pca():
    # 在稠密数组上进行增量 PCA
    X = iris.data
    
    # 设置批处理大小为总样本数的三分之一
    batch_size = X.shape[0] // 3
    
    # 创建 IncrementalPCA 对象，指定主成分数为 2，并使用上述批处理大小
    ipca = IncrementalPCA(n_components=2, batch_size=batch_size)
    
    # 创建 PCA 对象，指定主成分数为 2，并在 X 上进行拟合和变换
    pca = PCA(n_components=2)
    pca.fit_transform(X)

    # 对 X 应用增量 PCA，并获取转换后的数据
    X_transformed = ipca.fit_transform(X)

    # 断言转换后的数据形状应为 (样本数, 2)
    assert X_transformed.shape == (X.shape[0], 2)
    
    # 断言增量 PCA 的解释方差比例之和与普通 PCA 的相同
    np.testing.assert_allclose(
        ipca.explained_variance_ratio_.sum(),
        pca.explained_variance_ratio_.sum(),
        rtol=1e-3,
    )

    # 针对不同的主成分数进行增量 PCA 测试
    for n_components in [1, 2, X.shape[1]]:
        # 创建指定主成分数的增量 PCA 对象
        ipca = IncrementalPCA(n_components, batch_size=batch_size)
        
        # 在 X 上进行增量 PCA 的拟合
        ipca.fit(X)
        
        # 获取增量 PCA 的协方差矩阵
        cov = ipca.get_covariance()
        
        # 获取增量 PCA 的精度矩阵
        precision = ipca.get_precision()
        
        # 断言协方差矩阵与精度矩阵的乘积应接近单位矩阵
        np.testing.assert_allclose(
            np.dot(cov, precision), np.eye(X.shape[1]), atol=1e-13
        )


@pytest.mark.parametrize(
    "sparse_container", CSC_CONTAINERS + CSR_CONTAINERS + LIL_CONTAINERS
)
def test_incremental_pca_sparse(sparse_container):
    # 在稀疏数组上进行增量 PCA
    X = iris.data
    
    # 创建 PCA 对象，指定主成分数为 2，并在 X 上进行拟合和变换
    pca = PCA(n_components=2)
    pca.fit_transform(X)
    
    # 将 X 转换为指定稀疏容器类型的稀疏数组
    X_sparse = sparse_container(X)
    
    # 设置批处理大小为总样本数的三分之一
    batch_size = X_sparse.shape[0] // 3
    
    # 创建 IncrementalPCA 对象，指定主成分数为 2，并使用上述批处理大小
    ipca = IncrementalPCA(n_components=2, batch_size=batch_size)

    # 对 X_sparse 应用增量 PCA，并获取转换后的数据
    X_transformed = ipca.fit_transform(X_sparse)

    # 断言转换后的数据形状应为 (样本数, 2)
    assert X_transformed.shape == (X_sparse.shape[0], 2)
    
    # 断言增量 PCA 的解释方差比例之和与普通 PCA 的相同
    np.testing.assert_allclose(
        ipca.explained_variance_ratio_.sum(),
        pca.explained_variance_ratio_.sum(),
        rtol=1e-3,
    )

    # 针对不同的主成分数进行增量 PCA 测试
    for n_components in [1, 2, X.shape[1]]:
        # 创建指定主成分数的增量 PCA 对象
        ipca = IncrementalPCA(n_components, batch_size=batch_size)
        
        # 在 X_sparse 上进行增量 PCA 的拟合
        ipca.fit(X_sparse)
        
        # 获取增量 PCA 的协方差矩阵
        cov = ipca.get_covariance()
        
        # 获取增量 PCA 的精度矩阵
        precision = ipca.get_precision()
        
        # 断言协方差矩阵与精度矩阵的乘积应接近单位矩阵
        np.testing.assert_allclose(
            np.dot(cov, precision), np.eye(X_sparse.shape[1]), atol=1e-13
        )

    # 使用 pytest 断言，验证对稀疏输入进行增量 PCA 的部分拟合会引发 TypeError 异常
    with pytest.raises(
        TypeError,
        match=(
            "IncrementalPCA.partial_fit does not support "
            "sparse input. Either convert data to dense "
            "or use IncrementalPCA.fit to do so in batches."
        ),
    ):
        ipca.partial_fit(X_sparse)


def test_incremental_pca_check_projection():
    # 测试数据投影的正确性
    rng = np.random.RandomState(1999)
    n, p = 100, 3
    
    # 生成均值为 3，4，5，标准差为 0.1 的正态分布随机数
    X = rng.randn(n, p) * 0.1
    X[:10] += np.array([3, 4, 5])
    
    # 生成均值为 3，4，5，标准差为 0.1 的正态分布随机数，作为测试数据 Xt
    Xt = 0.1 * rng.randn(1, p) + np.array([3, 4, 5])

    # 获取生成数据 X 的重构投影
    # 注意：Xt 与 X 具有相同的“组件”，只是分开的
    # 使用增量主成分分析（IncrementalPCA）对数据集 X 进行拟合，并对数据集 Xt 进行转换，得到降维后的结果 Yt
    Yt = IncrementalPCA(n_components=2).fit(X).transform(Xt)

    # 对 Yt 进行归一化处理，使其每个元素的平方和开平方等于 1
    Yt /= np.sqrt((Yt**2).sum())

    # 确保 Yt 的第一个元素的绝对值接近于 1，这表示重构工作按预期进行
    assert_almost_equal(np.abs(Yt[0][0]), 1.0, 1)
def test_incremental_pca_inverse():
    # Test that the projection of data can be inverted.

    # 使用种子为1999的随机数生成器创建随机数对象
    rng = np.random.RandomState(1999)
    # 设置样本数和特征数
    n, p = 50, 3
    # 生成随机的球形数据
    X = rng.randn(n, p)
    # 将数据的第二列乘以一个较小的数，使其相对较小
    X[:, 1] *= 0.00001
    # 将数据加上一个较大的均值向量[5, 4, 3]
    X += [5, 4, 3]

    # 检查能否从转换后的信号中找到原始数据（因为数据几乎是满秩的）
    ipca = IncrementalPCA(n_components=2, batch_size=10).fit(X)
    # 对数据进行变换
    Y = ipca.transform(X)
    # 对变换后的数据进行反变换
    Y_inverse = ipca.inverse_transform(Y)
    # 断言反变换后的数据与原始数据非常接近，精确度为3位小数
    assert_almost_equal(X, Y_inverse, decimal=3)


def test_incremental_pca_validation():
    # Test that n_components is <= n_features.

    X = np.array([[0, 1, 0], [1, 0, 0]])
    # 获取样本数和特征数
    n_samples, n_features = X.shape
    n_components = 4
    # 使用 pytest 检查是否会引发预期的 ValueError 异常
    with pytest.raises(
        ValueError,
        match=(
            "n_components={} invalid"
            " for n_features={}, need more rows than"
            " columns for IncrementalPCA"
            " processing".format(n_components, n_features)
        ),
    ):
        # 使用 n_components 和 batch_size 初始化 IncrementalPCA 对象，并拟合数据
        IncrementalPCA(n_components, batch_size=10).fit(X)

    # Tests that n_components is also <= n_samples.
    n_components = 3
    with pytest.raises(
        ValueError,
        match=(
            "n_components={} must be"
            " less or equal to the batch number of"
            " samples {}".format(n_components, n_samples)
        ),
    ):
        # 使用 n_components 初始化 IncrementalPCA 对象，并进行部分拟合
        IncrementalPCA(n_components=n_components).partial_fit(X)


def test_n_samples_equal_n_components():
    # Ensures no warning is raised when n_samples==n_components
    # Non-regression test for gh-19050

    # 创建 IncrementalPCA 对象
    ipca = IncrementalPCA(n_components=5)
    # 使用警告捕获上下文，检查在 n_samples==n_components 时是否会引发 RuntimeWarning
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        # 对随机生成的数据进行部分拟合
        ipca.partial_fit(np.random.randn(5, 7))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        # 对随机生成的数据进行完全拟合
        ipca.fit(np.random.randn(5, 7))


def test_n_components_none():
    # Ensures that n_components == None is handled correctly

    # 使用种子为1999的随机数生成器创建随机数对象
    rng = np.random.RandomState(1999)
    # 遍历多组样本数和特征数的组合
    for n_samples, n_features in [(50, 10), (10, 50)]:
        # 生成随机数据
        X = rng.rand(n_samples, n_features)
        # 创建 IncrementalPCA 对象，n_components 设置为 None
        ipca = IncrementalPCA(n_components=None)

        # 第一次部分拟合，ipca.n_components_ 将从 min(X.shape) 推断出来
        ipca.partial_fit(X)
        # 断言 ipca.n_components_ 等于 min(X.shape)
        assert ipca.n_components_ == min(X.shape)

        # 第二次部分拟合，ipca.n_components_ 将从第一次拟合计算得到的 ipca.components_ 推断出来
        ipca.partial_fit(X)
        # 断言 ipca.n_components_ 等于 ipca.components_ 的行数
        assert ipca.n_components_ == ipca.components_.shape[0]


def test_incremental_pca_set_params():
    # Test that components_ sign is stable over batch sizes.

    # 使用种子为1999的随机数生成器创建随机数对象
    rng = np.random.RandomState(1999)
    n_samples = 100
    n_features = 20
    # 生成随机数据
    X = rng.randn(n_samples, n_features)
    X2 = rng.randn(n_samples, n_features)
    X3 = rng.randn(n_samples, n_features)
    # 创建 IncrementalPCA 对象，设置 n_components=20
    ipca = IncrementalPCA(n_components=20)
    # 对数据 X 进行拟合
    ipca.fit(X)
    # 减少主成分的数量设置为10
    ipca.set_params(n_components=10)
    # 使用 pytest 检查是否会抛出 ValueError 异常
    with pytest.raises(ValueError):
        # 对模型进行部分拟合，验证是否抛出异常
        ipca.partial_fit(X2)
    
    # 增加主成分的数量设置为15
    ipca.set_params(n_components=15)
    # 使用 pytest 检查是否会抛出 ValueError 异常
    with pytest.raises(ValueError):
        # 对模型进行部分拟合，验证是否抛出异常
        ipca.partial_fit(X3)
    
    # 恢复到原始的主成分数量设置为20
    ipca.set_params(n_components=20)
    # 对模型进行部分拟合，使用原始的主成分数量设置
    ipca.partial_fit(X)
# 测试增量PCA中n_components更改是否会引发错误
def test_incremental_pca_num_features_change():
    # 创建随机数生成器对象
    rng = np.random.RandomState(1999)
    # 设置样本数
    n_samples = 100
    # 生成服从标准正态分布的数据矩阵X（100行，20列）
    X = rng.randn(n_samples, 20)
    # 生成不同维度的数据矩阵X2（100行，50列）
    X2 = rng.randn(n_samples, 50)
    # 初始化增量PCA对象，不限定主成分数目
    ipca = IncrementalPCA(n_components=None)
    # 对数据矩阵X进行拟合
    ipca.fit(X)
    # 使用pytest检查是否会引发值错误
    with pytest.raises(ValueError):
        ipca.partial_fit(X2)


# 测试增量PCA中主成分符号在不同批次大小下的稳定性
def test_incremental_pca_batch_signs():
    # 创建随机数生成器对象
    rng = np.random.RandomState(1999)
    # 设置样本数和特征数
    n_samples = 100
    n_features = 3
    # 生成服从标准正态分布的数据矩阵X（100行，3列）
    X = rng.randn(n_samples, n_features)
    # 存储所有批次下的主成分数组
    all_components = []
    # 设置不同的批次大小范围
    batch_sizes = np.arange(10, 20)
    # 遍历每个批次大小
    for batch_size in batch_sizes:
        # 初始化增量PCA对象，不限定主成分数目，使用当前批次大小拟合数据矩阵X
        ipca = IncrementalPCA(n_components=None, batch_size=batch_size).fit(X)
        # 将当前批次的主成分数组添加到列表中
        all_components.append(ipca.components_)
    
    # 检查相邻两个批次下的主成分符号是否稳定（精度为小数点后6位）
    for i, j in zip(all_components[:-1], all_components[1:]):
        assert_almost_equal(np.sign(i), np.sign(j), decimal=6)


# 测试增量PCA中主成分数值在不同批次大小下的稳定性
def test_incremental_pca_batch_values():
    # 创建随机数生成器对象
    rng = np.random.RandomState(1999)
    # 设置样本数和特征数
    n_samples = 100
    n_features = 3
    # 生成服从标准正态分布的数据矩阵X（100行，3列）
    X = rng.randn(n_samples, n_features)
    # 存储所有批次下的主成分数组
    all_components = []
    # 设置不同的批次大小范围
    batch_sizes = np.arange(20, 40, 3)
    # 遍历每个批次大小
    for batch_size in batch_sizes:
        # 初始化增量PCA对象，不限定主成分数目，使用当前批次大小拟合数据矩阵X
        ipca = IncrementalPCA(n_components=None, batch_size=batch_size).fit(X)
        # 将当前批次的主成分数组添加到列表中
        all_components.append(ipca.components_)
    
    # 检查相邻两个批次下的主成分数值是否稳定（精度为小数点后1位）
    for i, j in zip(all_components[:-1], all_components[1:]):
        assert_almost_equal(i, j, decimal=1)


# 测试增量PCA中每个批次中的样本数始终大于或等于n_components
def test_incremental_pca_batch_rank():
    # 创建随机数生成器对象
    rng = np.random.RandomState(1999)
    # 设置样本数和特征数
    n_samples = 100
    n_features = 20
    # 生成服从标准正态分布的数据矩阵X（100行，20列）
    X = rng.randn(n_samples, n_features)
    # 存储所有批次下的主成分数组
    all_components = []
    # 设置不同的批次大小范围
    batch_sizes = np.arange(20, 90, 3)
    # 遍历每个批次大小
    for batch_size in batch_sizes:
        # 初始化增量PCA对象，设定主成分数目为20，使用当前批次大小拟合数据矩阵X
        ipca = IncrementalPCA(n_components=20, batch_size=batch_size).fit(X)
        # 将当前批次的主成分数组添加到列表中
        all_components.append(ipca.components_)
    
    # 检查相邻两个批次下的主成分数组是否稳定（使用assert_allclose_dense_sparse函数）
    for components_i, components_j in zip(all_components[:-1], all_components[1:]):
        assert_allclose_dense_sparse(components_i, components_j)


# 测试增量PCA中fit和partial_fit方法是否得到等价结果
def test_incremental_pca_partial_fit():
    # 创建随机数生成器对象
    rng = np.random.RandomState(1999)
    n, p = 50, 3
    # 生成球形数据矩阵X（50行，3列）
    X = rng.randn(n, p)
    # 将数据矩阵X的第二列乘以一个很小的数，使得中间成分相对较小
    X[:, 1] *= 0.00001
    # 将数据矩阵X的每一行加上一个向量[5, 4, 3]，使得数据有一个较大的均值
    X += [5, 4, 3]
    
    # 设置批次大小为10的增量PCA对象，设定主成分数目为2，对数据矩阵X进行拟合
    ipca = IncrementalPCA(n_components=2, batch_size=10).fit(X)
    # 设置批次大小为10的增量PCA对象，设定主成分数目为2
    pipca = IncrementalPCA(n_components=2, batch_size=10)
    # 生成批次迭代器数组（0到n+1，步长为batch_size）
    batch_itr = np.arange(0, n + 1, 10)
    # 遍历每个批次的起始索引和结束索引
    for i, j in zip(batch_itr[:-1], batch_itr[1:]):
        # 使用partial_fit方法拟合数据矩阵X的每个批次
        pipca.partial_fit(X[i:j, :])
    
    # 检查使用fit方法和partial_fit方法得到的主成分数组是否近似相等（精度为小数点后3位）
    assert_almost_equal(ipca.components_, pipca.components_, decimal=3)


# 与PCA在鸢尾花数据集上的性能进行比较的测试（未提供完整代码）
    # 使用 Iris 数据集中的数据 X 进行测试 IncrementalPCA 和 PCA 是否近似（可能存在符号翻转）。
    X = iris.data
    
    # 使用 PCA 对象拟合数据 X，并降维到 2 维
    Y_pca = PCA(n_components=2).fit_transform(X)
    
    # 使用 IncrementalPCA 对象拟合数据 X，设置批处理大小为 25，并降维到 2 维
    Y_ipca = IncrementalPCA(n_components=2, batch_size=25).fit_transform(X)
    
    # 断言两个转换后的结果 Y_pca 和 Y_ipca 的绝对值近似相等，精度为小数点后一位
    assert_almost_equal(np.abs(Y_pca), np.abs(Y_ipca), 1)
def test_incremental_pca_against_pca_random_data():
    # Test that IncrementalPCA and PCA are approximate (to a sign flip).

    # 设置随机数生成器
    rng = np.random.RandomState(1999)
    # 设置样本数和特征数
    n_samples = 100
    n_features = 3
    # 生成随机数据矩阵 X
    X = rng.randn(n_samples, n_features) + 5 * rng.rand(1, n_features)

    # 对数据 X 进行 PCA 变换
    Y_pca = PCA(n_components=3).fit_transform(X)
    # 对数据 X 进行 IncrementalPCA 变换
    Y_ipca = IncrementalPCA(n_components=3, batch_size=25).fit_transform(X)

    # 断言 PCA 和 IncrementalPCA 的结果在绝对值上近似相等，精度为1
    assert_almost_equal(np.abs(Y_pca), np.abs(Y_ipca), 1)


def test_explained_variances():
    # Test that PCA and IncrementalPCA calculations match

    # 生成低秩矩阵 X 用于测试
    X = datasets.make_low_rank_matrix(
        1000, 100, tail_strength=0.0, effective_rank=10, random_state=1999
    )
    prec = 3
    n_samples, n_features = X.shape
    for nc in [None, 99]:
        # 计算 PCA
        pca = PCA(n_components=nc).fit(X)
        # 计算 IncrementalPCA
        ipca = IncrementalPCA(n_components=nc, batch_size=100).fit(X)
        
        # 断言 PCA 和 IncrementalPCA 的解释方差近似相等，精度为 prec
        assert_almost_equal(
            pca.explained_variance_, ipca.explained_variance_, decimal=prec
        )
        # 断言 PCA 和 IncrementalPCA 的解释方差比例近似相等，精度为 prec
        assert_almost_equal(
            pca.explained_variance_ratio_, ipca.explained_variance_ratio_, decimal=prec
        )
        # 断言 PCA 和 IncrementalPCA 的噪声方差近似相等，精度为 prec
        assert_almost_equal(pca.noise_variance_, ipca.noise_variance_, decimal=prec)


def test_singular_values():
    # Check that the IncrementalPCA output has the correct singular values

    # 设置随机数生成器
    rng = np.random.RandomState(0)
    n_samples = 1000
    n_features = 100

    # 生成低秩矩阵 X 用于测试
    X = datasets.make_low_rank_matrix(
        n_samples, n_features, tail_strength=0.0, effective_rank=10, random_state=rng
    )

    # 计算 PCA 并比较奇异值
    pca = PCA(n_components=10, svd_solver="full", random_state=rng).fit(X)
    ipca = IncrementalPCA(n_components=10, batch_size=100).fit(X)
    assert_array_almost_equal(pca.singular_values_, ipca.singular_values_, 2)

    # 比较到 Frobenius 范数
    X_pca = pca.transform(X)
    X_ipca = ipca.transform(X)
    assert_array_almost_equal(
        np.sum(pca.singular_values_**2.0), np.linalg.norm(X_pca, "fro") ** 2.0, 12
    )
    assert_array_almost_equal(
        np.sum(ipca.singular_values_**2.0), np.linalg.norm(X_ipca, "fro") ** 2.0, 2
    )

    # 比较到分数向量的 2-范数
    assert_array_almost_equal(
        pca.singular_values_, np.sqrt(np.sum(X_pca**2.0, axis=0)), 12
    )
    assert_array_almost_equal(
        ipca.singular_values_, np.sqrt(np.sum(X_ipca**2.0, axis=0)), 2
    )

    # 设置奇异值并查看返回值
    rng = np.random.RandomState(0)
    n_samples = 100
    n_features = 110

    X = datasets.make_low_rank_matrix(
        n_samples, n_features, tail_strength=0.0, effective_rank=3, random_state=rng
    )

    pca = PCA(n_components=3, svd_solver="full", random_state=rng)
    ipca = IncrementalPCA(n_components=3, batch_size=100)

    X_pca = pca.fit_transform(X)
    X_pca /= np.sqrt(np.sum(X_pca**2.0, axis=0))
    X_pca[:, 0] *= 3.142
    X_pca[:, 1] *= 2.718

    X_hat = np.dot(X_pca, pca.components_)
    pca.fit(X_hat)
    ipca.fit(X_hat)
    # 使用断言检查 PCA 对象的奇异值是否几乎等于给定值列表，精度为 1e-14
    assert_array_almost_equal(pca.singular_values_, [3.142, 2.718, 1.0], 14)
    # 使用断言检查 IPCA 对象的奇异值是否几乎等于给定值列表，精度为 1e-14
    assert_array_almost_equal(ipca.singular_values_, [3.142, 2.718, 1.0], 14)
# 测试白化处理的函数，使用全局随机种子
def test_whitening(global_random_seed):
    # 创建一个低秩矩阵 X，用于测试，确保 PCA 和 IncrementalPCA 转换匹配到符号翻转
    X = datasets.make_low_rank_matrix(
        1000, 10, tail_strength=0.0, effective_rank=2, random_state=global_random_seed
    )
    atol = 1e-3
    # 遍历不同的 n_components 值进行测试，包括 None 和 9
    for nc in [None, 9]:
        # 使用 PCA 对象进行拟合，启用白化处理
        pca = PCA(whiten=True, n_components=nc).fit(X)
        # 使用 IncrementalPCA 对象进行拟合，启用白化处理和指定的 n_components
        ipca = IncrementalPCA(whiten=True, n_components=nc, batch_size=250).fit(X)

        # 由于数据是秩亏的，一些主成分是纯噪声，我们在比较前需要将它们过滤掉
        stable_mask = pca.explained_variance_ratio_ > 1e-12

        # 使用 PCA 和 IncrementalPCA 转换 X
        Xt_pca = pca.transform(X)
        Xt_ipca = ipca.transform(X)
        # 断言两者在稳定维度上的绝对值近似相等
        assert_allclose(
            np.abs(Xt_pca)[:, stable_mask],
            np.abs(Xt_ipca)[:, stable_mask],
            atol=atol,
        )

        # 噪声维度位于逆变换的零空间中，因此它们不会影响重构，这里不需要应用稳定性掩码
        Xinv_ipca = ipca.inverse_transform(Xt_ipca)
        Xinv_pca = pca.inverse_transform(Xt_pca)
        # 断言原始数据与逆变换后的数据近似相等
        assert_allclose(X, Xinv_ipca, atol=atol)
        assert_allclose(X, Xinv_pca, atol=atol)
        # 断言两个逆变换后的结果近似相等
        assert_allclose(Xinv_pca, Xinv_ipca, atol=atol)


# 检查在所有 Python 版本中是否使用浮点除法的测试
def test_incremental_pca_partial_fit_float_division():
    rng = np.random.RandomState(0)
    A = rng.randn(5, 3) + 2
    B = rng.randn(7, 3) + 5

    # 创建 IncrementalPCA 对象并进行部分拟合
    pca = IncrementalPCA(n_components=2)
    pca.partial_fit(A)
    # 将 n_samples_seen_ 设置为浮点数而不是整数
    pca.n_samples_seen_ = float(pca.n_samples_seen_)
    pca.partial_fit(B)
    singular_vals_float_samples_seen = pca.singular_values_

    # 创建另一个 IncrementalPCA 对象并进行部分拟合
    pca2 = IncrementalPCA(n_components=2)
    pca2.partial_fit(A)
    pca2.partial_fit(B)
    singular_vals_int_samples_seen = pca2.singular_values_

    # 断言两个对象计算出的奇异值近似相等
    np.testing.assert_allclose(
        singular_vals_float_samples_seen, singular_vals_int_samples_seen
    )


# 在 Windows 操作系统上测试溢出错误的函数
def test_incremental_pca_fit_overflow_error():
    rng = np.random.RandomState(0)
    A = rng.rand(500000, 2)

    # 创建 IncrementalPCA 对象并进行拟合
    ipca = IncrementalPCA(n_components=2, batch_size=10000)
    ipca.fit(A)

    # 创建 PCA 对象并进行拟合
    pca = PCA(n_components=2)
    pca.fit(A)

    # 断言两个对象计算出的奇异值近似相等
    np.testing.assert_allclose(ipca.singular_values_, pca.singular_values_)


# 检查 IncrementalPCA 的特征名输出是否符合预期
def test_incremental_pca_feature_names_out():
    ipca = IncrementalPCA(n_components=2).fit(iris.data)

    # 获取特征名输出
    names = ipca.get_feature_names_out()
    # 断言特征名是否与预期的列表相等
    assert_array_equal([f"incrementalpca{i}" for i in range(2)], names)
```