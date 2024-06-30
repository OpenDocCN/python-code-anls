# `D:\src\scipysrc\scikit-learn\sklearn\cluster\tests\test_birch.py`

```
"""
Tests for the birch clustering algorithm.
"""

# 导入所需的库和模块
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试

# 导入 sklearn 中的聚类算法和相关函数
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.cluster.tests.common import generate_clustered_data  # 导入用于生成聚类数据的函数
from sklearn.datasets import make_blobs  # 导入 make_blobs 函数，用于生成聚类测试数据
from sklearn.exceptions import ConvergenceWarning  # 导入收敛警告异常类
from sklearn.metrics import pairwise_distances_argmin, v_measure_score  # 导入计算距离和评估聚类效果的函数
from sklearn.utils._testing import assert_allclose, assert_array_equal  # 导入用于测试的函数和方法
from sklearn.utils.fixes import CSR_CONTAINERS  # 导入 CSR 容器相关的修复方法


def test_n_samples_leaves_roots(global_random_seed, global_dtype):
    # Sanity check for the number of samples in leaves and roots
    X, y = make_blobs(n_samples=10, random_state=global_random_seed)  # 生成包含10个样本的聚类数据
    X = X.astype(global_dtype, copy=False)  # 将数据类型转换为指定的全局数据类型
    brc = Birch()  # 创建 Birch 聚类器对象
    brc.fit(X)  # 对数据进行聚类
    # 计算根节点下所有子簇中样本的总数
    n_samples_root = sum([sc.n_samples_ for sc in brc.root_.subclusters_])
    # 计算叶子节点下所有子簇中样本的总数
    n_samples_leaves = sum(
        [sc.n_samples_ for leaf in brc._get_leaves() for sc in leaf.subclusters_]
    )
    # 断言叶子节点中样本数等于原始数据的样本数
    assert n_samples_leaves == X.shape[0]
    # 断言根节点中样本数等于原始数据的样本数
    assert n_samples_root == X.shape[0]


def test_partial_fit(global_random_seed, global_dtype):
    # Test that fit is equivalent to calling partial_fit multiple times
    X, y = make_blobs(n_samples=100, random_state=global_random_seed)  # 生成包含100个样本的聚类数据
    X = X.astype(global_dtype, copy=False)  # 将数据类型转换为指定的全局数据类型
    brc = Birch(n_clusters=3)  # 创建 Birch 聚类器对象，设置簇数为3
    brc.fit(X)  # 对数据进行聚类
    brc_partial = Birch(n_clusters=None)  # 创建不限定簇数的 Birch 聚类器对象
    brc_partial.partial_fit(X[:50])  # 对前50个样本进行部分拟合
    brc_partial.partial_fit(X[50:])  # 对后50个样本进行部分拟合
    # 断言部分拟合后得到的子簇中心与完全拟合后的子簇中心相近
    assert_allclose(brc_partial.subcluster_centers_, brc.subcluster_centers_)

    # Test that same global labels are obtained after calling partial_fit
    # with None
    brc_partial.set_params(n_clusters=3)  # 设置簇数为3
    brc_partial.partial_fit(None)  # 对所有样本进行部分拟合
    # 断言部分拟合后得到的子簇标签与完全拟合后的子簇标签相等
    assert_array_equal(brc_partial.subcluster_labels_, brc.subcluster_labels_)


def test_birch_predict(global_random_seed, global_dtype):
    # Test the predict method predicts the nearest centroid.
    rng = np.random.RandomState(global_random_seed)  # 创建指定随机种子的随机状态生成器
    X = generate_clustered_data(n_clusters=3, n_features=3, n_samples_per_cluster=10)  # 生成聚类数据
    X = X.astype(global_dtype, copy=False)  # 将数据类型转换为指定的全局数据类型

    # n_samples * n_samples_per_cluster
    shuffle_indices = np.arange(30)  # 创建包含30个元素的索引数组
    rng.shuffle(shuffle_indices)  # 打乱索引数组顺序
    X_shuffle = X[shuffle_indices, :]  # 根据打乱的索引重新排序数据
    brc = Birch(n_clusters=4, threshold=1.0)  # 创建 Birch 聚类器对象，设置簇数为4，阈值为1.0
    brc.fit(X_shuffle)  # 对打乱顺序后的数据进行聚类

    # Birch must preserve inputs' dtype
    assert brc.subcluster_centers_.dtype == global_dtype  # 断言子簇中心的数据类型与全局指定的数据类型相同

    assert_array_equal(brc.labels_, brc.predict(X_shuffle))  # 断言预测标签与实际标签相等
    centroids = brc.subcluster_centers_  # 获取子簇中心
    nearest_centroid = brc.subcluster_labels_[  # 获取最近的质心的子簇标签
        pairwise_distances_argmin(X_shuffle, centroids)
    ]
    # 断言最近质心的子簇标签与预测标签的 V-Measure 分数接近1.0
    assert_allclose(v_measure_score(nearest_centroid, brc.labels_), 1.0)


def test_n_clusters(global_random_seed, global_dtype):
    # Test that n_clusters param works properly
    X, y = make_blobs(n_samples=100, centers=10, random_state=global_random_seed)  # 生成包含100个样本的聚类数据，共10个中心
    X = X.astype(global_dtype, copy=False)  # 将数据类型转换为指定的全局数据类型
    brc1 = Birch(n_clusters=10)  # 创建 Birch 聚类器对象，设置簇数为10
    brc1.fit(X)  # 对数据进行聚类
    # 断言子簇中心的数量大于设定的簇数
    assert len(brc1.subcluster_centers_) > 10
    # 确保聚类后的标签数量为10个的断言
    assert len(np.unique(brc1.labels_)) == 10
    
    # 测试使用层次聚类（Agglomerative Clustering）时是否能得到相同的结果
    gc = AgglomerativeClustering(n_clusters=10)
    # 创建另一个 Birch 对象 brc2，并使用上面创建的层次聚类 gc 进行聚类
    brc2 = Birch(n_clusters=gc)
    # 对数据 X 进行聚类
    brc2.fit(X)
    # 断言 brc1 的子簇标签与 brc2 的子簇标签相等
    assert_array_equal(brc1.subcluster_labels_, brc2.subcluster_labels_)
    # 断言 brc1 的完整标签与 brc2 的完整标签相等
    assert_array_equal(brc1.labels_, brc2.labels_)
    
    # 测试当聚类数较少时是否会触发警告
    # 创建一个新的 Birch 对象 brc4，设置阈值为 10000.0
    brc4 = Birch(threshold=10000.0)
    # 使用 pytest 的 warns 方法捕获 ConvergenceWarning 警告
    with pytest.warns(ConvergenceWarning):
        # 对数据 X 进行聚类
        brc4.fit(X)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 使用pytest的参数化标记，用于多次运行同一个测试函数，每次使用不同的参数
def test_sparse_X(global_random_seed, global_dtype, csr_container):
    # 测试稀疏数据和密集数据的结果是否相同
    X, y = make_blobs(n_samples=100, centers=10, random_state=global_random_seed)
    # 将数据类型转换为全局设定的数据类型，并且不复制数据
    X = X.astype(global_dtype, copy=False)
    # 初始化Birch聚类器，设定簇的数量为10
    brc = Birch(n_clusters=10)
    # 对密集数据进行拟合
    brc.fit(X)

    # 使用csr_container函数处理稀疏数据
    csr = csr_container(X)
    # 初始化另一个Birch聚类器，设定簇的数量为10，并对稀疏数据进行拟合
    brc_sparse = Birch(n_clusters=10)
    brc_sparse.fit(csr)

    # Birch聚类器必须保持输入数据的数据类型
    assert brc_sparse.subcluster_centers_.dtype == global_dtype

    # 检查密集数据和稀疏数据的标签是否相等
    assert_array_equal(brc.labels_, brc_sparse.labels_)
    # 检查密集数据和稀疏数据的子簇中心是否接近
    assert_allclose(brc.subcluster_centers_, brc_sparse.subcluster_centers_)


def test_partial_fit_second_call_error_checks():
    # 第二次调用partial_fit时，如果特征数与第一次调用不一致，将会出错
    X, y = make_blobs(n_samples=100)
    # 初始化Birch聚类器，设定簇的数量为3
    brc = Birch(n_clusters=3)
    # 第一次部分拟合
    brc.partial_fit(X, y)

    # 出错信息应包含以下内容
    msg = "X has 1 features, but Birch is expecting 2 features"
    # 使用pytest断言检查是否抛出预期的值错误，并包含特定的消息
    with pytest.raises(ValueError, match=msg):
        brc.partial_fit(X[:, [0]], y)


def check_branching_factor(node, branching_factor):
    # 检查节点的子簇是否最多有branching_factor个
    subclusters = node.subclusters_
    assert branching_factor >= len(subclusters)
    for cluster in subclusters:
        if cluster.child_:
            # 递归检查每个子簇的分支因子
            check_branching_factor(cluster.child_, branching_factor)


def test_branching_factor(global_random_seed, global_dtype):
    # 测试节点的子簇是否最多有指定的分支因子个数
    X, y = make_blobs(random_state=global_random_seed)
    X = X.astype(global_dtype, copy=False)
    branching_factor = 9

    # 故意设置一个较低的阈值以最大化子簇数
    # 初始化Birch聚类器，不设定具体簇的数量，指定分支因子和阈值
    brc = Birch(n_clusters=None, branching_factor=branching_factor, threshold=0.01)
    brc.fit(X)
    # 检查根节点的分支因子
    check_branching_factor(brc.root_, branching_factor)
    brc = Birch(n_clusters=3, branching_factor=branching_factor, threshold=0.01)
    brc.fit(X)
    # 检查根节点的分支因子
    check_branching_factor(brc.root_, branching_factor)


def check_threshold(birch_instance, threshold):
    """使用叶子链接列表进行遍历"""
    current_leaf = birch_instance.dummy_leaf_.next_leaf_
    while current_leaf:
        subclusters = current_leaf.subclusters_
        for sc in subclusters:
            # 断言叶子节点的子簇的半径是否小于等于阈值
            assert threshold >= sc.radius
        current_leaf = current_leaf.next_leaf_


def test_threshold(global_random_seed, global_dtype):
    # 测试叶子节点的子簇是否具有小于半径的阈值
    X, y = make_blobs(n_samples=80, centers=4, random_state=global_random_seed)
    X = X.astype(global_dtype, copy=False)
    # 初始化Birch聚类器，不设定具体簇的数量，指定阈值
    brc = Birch(threshold=0.5, n_clusters=None)
    brc.fit(X)
    # 检查阈值是否小于等于半径
    check_threshold(brc, 0.5)

    brc = Birch(threshold=5.0, n_clusters=None)
    brc.fit(X)
    # 检查阈值是否小于等于半径
    check_threshold(brc, 5.0)


def test_birch_n_clusters_long_int():
    # 检查Birch是否支持具有np.int64数据类型的簇数量，例如来自np.arange
    X, _ = make_blobs(random_state=0)
    # 设置簇数量为np.int64类型，例如来自np.arange
    n_clusters = np.int64(5)
    # 使用 Birch 算法对输入的数据 X 进行聚类，其中 n_clusters 是指定的聚类数目
    # 返回一个训练好的聚类器对象，这里没有显式地将其分配给变量，因此结果未存储
    Birch(n_clusters=n_clusters).fit(X)
# 定义测试函数，验证 `get_feature_names_out` 方法在 Birch 聚类算法中的输出结果
def test_feature_names_out():
    """Check `get_feature_names_out` for `Birch`."""
    # 生成包含 80 个样本和 4 个特征的随机数据集
    X, _ = make_blobs(n_samples=80, n_features=4, random_state=0)
    # 创建 Birch 聚类器对象，设置簇数为 4
    brc = Birch(n_clusters=4)
    # 对数据集 X 进行聚类
    brc.fit(X)
    # 获取输出的特征名列表
    n_clusters = brc.subcluster_centers_.shape[0]
    names_out = brc.get_feature_names_out()
    # 验证输出的特征名列表是否符合预期
    assert_array_equal([f"birch{i}" for i in range(n_clusters)], names_out)


# 定义测试函数，验证在不同数据类型下进行聚类变换后的结果一致性
def test_transform_match_across_dtypes(global_random_seed):
    X, _ = make_blobs(n_samples=80, n_features=4, random_state=global_random_seed)
    # 创建 Birch 聚类器对象，设置簇数为 4，阈值为 1.1
    brc = Birch(n_clusters=4, threshold=1.1)
    # 分别对原始数据 X 和转换成 np.float32 类型后的数据 X 进行聚类变换
    Y_64 = brc.fit_transform(X)
    Y_32 = brc.fit_transform(X.astype(np.float32))
    # 验证两种数据类型下的聚类结果是否在指定误差范围内相等
    assert_allclose(Y_64, Y_32, atol=1e-6)


# 定义测试函数，验证子簇中心点数据类型与全局设定一致性
def test_subcluster_dtype(global_dtype):
    X = make_blobs(n_samples=80, n_features=4, random_state=0)[0].astype(
        global_dtype, copy=False
    )
    # 创建 Birch 聚类器对象，设置簇数为 4
    brc = Birch(n_clusters=4)
    # 对数据集 X 进行聚类，并验证子簇中心点的数据类型是否与全局设定一致
    assert brc.fit(X).subcluster_centers_.dtype == global_dtype


# 定义测试函数，验证在节点分裂时，即使存在重复数据点，两个子簇都得到更新的情况
def test_both_subclusters_updated():
    """Check that both subclusters are updated when a node a split, even when there are
    duplicated data points. Non-regression test for #23269.
    """
    # 定义包含重复数据点的数据集 X
    X = np.array(
        [
            [-2.6192791, -1.5053215],
            [-2.9993038, -1.6863596],
            [-2.3724914, -1.3438171],
            [-2.336792, -1.3417323],
            [-2.4089134, -1.3290224],
            [-2.3724914, -1.3438171],
            [-3.364009, -1.8846745],
            [-2.3724914, -1.3438171],
            [-2.617677, -1.5003285],
            [-2.2960556, -1.3260119],
            [-2.3724914, -1.3438171],
            [-2.5459878, -1.4533926],
            [-2.25979, -1.3003055],
            [-2.4089134, -1.3290224],
            [-2.3724914, -1.3438171],
            [-2.4089134, -1.3290224],
            [-2.5459878, -1.4533926],
            [-2.3724914, -1.3438171],
            [-2.9720619, -1.7058647],
            [-2.336792, -1.3417323],
            [-2.3724914, -1.3438171],
        ],
        dtype=np.float32,
    )
    # 创建 Birch 聚类器对象，设置分支因子为 5，阈值为 1e-5，簇数为 None（自动确定）
    # 用于测试在存在重复数据点时节点分裂后两个子簇都得到更新的情况
    Birch(branching_factor=5, threshold=1e-5, n_clusters=None).fit(X)


# TODO(1.8): Remove
# 定义测试函数，验证警告提示功能是否正常，检查 'copy' 参数在 Birch 构造函数中的使用是否已过时
def test_birch_copy_deprecated():
    X, _ = make_blobs(n_samples=80, n_features=4, random_state=0)
    # 创建 Birch 聚类器对象，设置簇数为 4，使用了已过时的 'copy' 参数
    brc = Birch(n_clusters=4, copy=True)
    # 使用 pytest 检测是否生成 FutureWarning 警告，内容包含 "`copy` was deprecated"
    with pytest.warns(FutureWarning, match="`copy` was deprecated"):
        brc.fit(X)
```