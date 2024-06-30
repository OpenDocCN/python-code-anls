# `D:\src\scipysrc\scikit-learn\sklearn\cluster\tests\test_bisect_k_means.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试

from sklearn.cluster import BisectingKMeans  # 导入 BisectingKMeans 聚类算法
from sklearn.metrics import v_measure_score  # 导入 v_measure_score 用于聚类评估
from sklearn.utils._testing import assert_allclose, assert_array_equal  # 导入测试工具函数
from sklearn.utils.fixes import CSR_CONTAINERS  # 导入 CSR 数据结构支持


@pytest.mark.parametrize("bisecting_strategy", ["biggest_inertia", "largest_cluster"])  # 参数化测试，测试不同的分裂策略
@pytest.mark.parametrize("init", ["k-means++", "random"])  # 参数化测试，测试不同的初始化策略
def test_three_clusters(bisecting_strategy, init):
    """Tries to perform bisect k-means for three clusters to check
    if splitting data is performed correctly.
    """
    X = np.array(
        [[1, 1], [10, 1], [3, 1], [10, 0], [2, 1], [10, 2], [10, 8], [10, 9], [10, 10]]
    )  # 创建一个包含9个样本的二维数组作为数据集

    bisect_means = BisectingKMeans(
        n_clusters=3,
        random_state=0,
        bisecting_strategy=bisecting_strategy,
        init=init,
    )  # 初始化 BisectingKMeans 聚类模型

    bisect_means.fit(X)  # 使用数据集进行聚类

    expected_centers = [[2, 1], [10, 1], [10, 9]]  # 预期的聚类中心
    expected_labels = [0, 1, 0, 1, 0, 1, 2, 2, 2]  # 预期的样本标签

    assert_allclose(
        sorted(expected_centers), sorted(bisect_means.cluster_centers_.tolist())
    )  # 断言聚类中心是否接近预期值
    assert_allclose(v_measure_score(expected_labels, bisect_means.labels_), 1.0)  # 断言 V-measure 分数是否为1.0


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sparse(csr_container):
    """Test Bisecting K-Means with sparse data.

    Checks if labels and centers are the same between dense and sparse.
    """

    rng = np.random.RandomState(0)  # 使用种子为0的随机数生成器

    X = rng.rand(20, 2)  # 创建一个20x2的随机数组成的稠密数据集
    X[X < 0.8] = 0  # 将数据集中小于0.8的值置为0
    X_csr = csr_container(X)  # 将稠密数据集转换为指定的 CSR 格式

    bisect_means = BisectingKMeans(n_clusters=3, random_state=0)  # 初始化 BisectingKMeans 聚类模型

    bisect_means.fit(X_csr)  # 使用 CSR 格式的数据集进行聚类
    sparse_centers = bisect_means.cluster_centers_  # 获取稀疏数据集的聚类中心

    bisect_means.fit(X)  # 使用稠密数据集进行聚类
    normal_centers = bisect_means.cluster_centers_  # 获取稠密数据集的聚类中心

    # Check if results is the same for dense and sparse data
    assert_allclose(normal_centers, sparse_centers, atol=1e-8)  # 断言稠密和稀疏数据集的聚类中心是否接近


@pytest.mark.parametrize("n_clusters", [4, 5])
def test_n_clusters(n_clusters):
    """Test if resulting labels are in range [0, n_clusters - 1]."""

    rng = np.random.RandomState(0)  # 使用种子为0的随机数生成器
    X = rng.rand(10, 2)  # 创建一个10x2的随机数组成的数据集

    bisect_means = BisectingKMeans(n_clusters=n_clusters, random_state=0)  # 初始化 BisectingKMeans 聚类模型
    bisect_means.fit(X)  # 使用数据集进行聚类

    assert_array_equal(np.unique(bisect_means.labels_), np.arange(n_clusters))  # 断言聚类标签是否在 [0, n_clusters-1] 范围内


def test_one_cluster():
    """Test single cluster."""

    X = np.array([[1, 2], [10, 2], [10, 8]])  # 创建一个包含3个样本的二维数组作为数据集

    bisect_means = BisectingKMeans(n_clusters=1, random_state=0).fit(X)  # 初始化并拟合单个聚类模型

    # All labels from fit or predict should be equal 0
    assert all(bisect_means.labels_ == 0)  # 断言所有样本的标签都等于0
    assert all(bisect_means.predict(X) == 0)  # 断言使用 predict 函数预测的标签都等于0

    assert_allclose(bisect_means.cluster_centers_, X.mean(axis=0).reshape(1, -1))  # 断言聚类中心是否接近样本均值


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS + [None])
def test_fit_predict(csr_container):
    """Check if labels from fit(X) method are same as from fit(X).predict(X)."""
    rng = np.random.RandomState(0)  # 使用种子为0的随机数生成器

    X = rng.rand(10, 2)  # 创建一个10x2的随机数组成的数据集

    if csr_container is not None:
        X[X < 0.8] = 0  # 将数据集中小于0.8的值置为0
        X = csr_container(X)  # 将数据集转换为指定的 CSR 格式
    # 使用 BisectingKMeans 算法初始化一个聚类器对象，设置聚类数为 3，随机数种子为 0
    bisect_means = BisectingKMeans(n_clusters=3, random_state=0)
    # 对数据集 X 进行聚类训练
    bisect_means.fit(X)
    # 断言 bisect_means 对象的 labels_ 属性与对数据集 X 的预测结果相等，用于验证聚类器的预测结果
    assert_array_equal(bisect_means.labels_, bisect_means.predict(X))
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS + [None])
def test_dtype_preserved(csr_container, global_dtype):
    """Check that centers dtype is the same as input data dtype."""
    # 设置随机数种子为0
    rng = np.random.RandomState(0)
    # 生成一个形状为(10, 2)的随机数组，数据类型为全局设定的数据类型
    X = rng.rand(10, 2).astype(global_dtype, copy=False)

    # 如果 csr_container 不为 None，则进行以下操作
    if csr_container is not None:
        # 将 X 中小于0.8的元素置为0
        X[X < 0.8] = 0
        # 将 X 转换为 csr_container 类型
        X = csr_container(X)

    # 创建 BisectingKMeans 对象，设置簇数为3，随机种子为0，然后对 X 进行拟合
    km = BisectingKMeans(n_clusters=3, random_state=0)
    km.fit(X)

    # 断言聚类中心的数据类型与全局数据类型相同
    assert km.cluster_centers_.dtype == global_dtype


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS + [None])
def test_float32_float64_equivalence(csr_container):
    """Check that the results are the same between float32 and float64."""
    # 设置随机数种子为0
    rng = np.random.RandomState(0)
    # 生成一个形状为(10, 2)的随机数组
    X = rng.rand(10, 2)

    # 如果 csr_container 不为 None，则进行以下操作
    if csr_container is not None:
        # 将 X 中小于0.8的元素置为0
        X[X < 0.8] = 0
        # 将 X 转换为 csr_container 类型
        X = csr_container(X)

    # 创建 BisectingKMeans 对象，设置簇数为3，随机种子为0，然后分别对 X 和 X.astype(np.float32) 进行拟合
    km64 = BisectingKMeans(n_clusters=3, random_state=0).fit(X)
    km32 = BisectingKMeans(n_clusters=3, random_state=0).fit(X.astype(np.float32))

    # 使用 assert_allclose 检查两个模型的聚类中心是否相等
    assert_allclose(km32.cluster_centers_, km64.cluster_centers_)
    # 使用 assert_array_equal 检查两个模型的标签是否相等
    assert_array_equal(km32.labels_, km64.labels_)


@pytest.mark.parametrize("algorithm", ("lloyd", "elkan"))
def test_no_crash_on_empty_bisections(algorithm):
    # Non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/27081
    # 设置随机数种子为0
    rng = np.random.RandomState(0)
    # 生成一个形状为(3000, 10)的随机数组
    X_train = rng.rand(3000, 10)
    # 创建 BisectingKMeans 对象，设置簇数为10，算法为 algorithm，然后对 X_train 进行拟合
    bkm = BisectingKMeans(n_clusters=10, algorithm=algorithm).fit(X_train)

    # 对缩放后的数据 X_test 进行预测，以触发内部空分割的极端情况
    X_test = 50 * rng.rand(100, 10)
    labels = bkm.predict(X_test)  # 预测标签，不应因除以0而崩溃
    # 使用断言确保所有预测标签都在 [0, 9] 范围内
    assert np.isin(np.unique(labels), np.arange(10)).all()


def test_one_feature():
    # Check that no error is raised when there is only one feature
    # Non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/27236
    # 生成一个形状为(128, 1)的正态分布随机数组
    X = np.random.normal(size=(128, 1))
    # 创建 BisectingKMeans 对象，设置分割策略为"biggest_inertia"，随机种子为0，然后对 X 进行拟合
    BisectingKMeans(bisecting_strategy="biggest_inertia", random_state=0).fit(X)
```