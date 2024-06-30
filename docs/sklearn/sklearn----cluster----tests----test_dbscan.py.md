# `D:\src\scipysrc\scikit-learn\sklearn\cluster\tests\test_dbscan.py`

```
"""
Tests for DBSCAN clustering algorithm
"""

import pickle  # 导入pickle模块，用于序列化和反序列化对象
import warnings  # 导入warnings模块，用于管理警告

import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于编写和运行测试

from scipy.spatial import distance  # 从scipy.spatial模块导入distance，用于计算距离

from sklearn.cluster import DBSCAN, dbscan  # 导入DBSCAN聚类算法
from sklearn.cluster.tests.common import generate_clustered_data  # 从测试公共模块导入生成聚类数据的函数
from sklearn.metrics.pairwise import pairwise_distances  # 导入计算成对距离的函数
from sklearn.neighbors import NearestNeighbors  # 导入最近邻模块
from sklearn.utils._testing import assert_array_equal  # 导入数组比较函数
from sklearn.utils.fixes import CSR_CONTAINERS, LIL_CONTAINERS  # 导入稀疏矩阵容器修复相关函数

n_clusters = 3  # 设置聚类簇的数量
X = generate_clustered_data(n_clusters=n_clusters)  # 生成指定数量聚类簇的数据


def test_dbscan_similarity():
    # Tests the DBSCAN algorithm with a similarity array.
    # Parameters chosen specifically for this task.
    eps = 0.15  # 设置邻域半径
    min_samples = 10  # 设置最小样本数
    # Compute similarities
    D = distance.squareform(distance.pdist(X))  # 计算数据点之间的距离矩阵并转换成方阵
    D /= np.max(D)  # 归一化距离矩阵
    # Compute DBSCAN
    core_samples, labels = dbscan(
        D, metric="precomputed", eps=eps, min_samples=min_samples
    )  # 使用预计算的距离矩阵执行DBSCAN聚类
    # number of clusters, ignoring noise if present
    n_clusters_1 = len(set(labels)) - (1 if -1 in labels else 0)  # 计算聚类簇的数量，忽略噪声点
    assert n_clusters_1 == n_clusters

    db = DBSCAN(metric="precomputed", eps=eps, min_samples=min_samples)  # 创建DBSCAN对象
    labels = db.fit(D).labels_  # 对数据进行DBSCAN聚类并获取标签

    n_clusters_2 = len(set(labels)) - int(-1 in labels)  # 计算聚类簇的数量，忽略噪声点
    assert n_clusters_2 == n_clusters


def test_dbscan_feature():
    # Tests the DBSCAN algorithm with a feature vector array.
    # Parameters chosen specifically for this task.
    # Different eps to other test, because distance is not normalised.
    eps = 0.8  # 设置邻域半径
    min_samples = 10  # 设置最小样本数
    metric = "euclidean"  # 设置距离度量方式为欧氏距离
    # Compute DBSCAN
    core_samples, labels = dbscan(X, metric=metric, eps=eps, min_samples=min_samples)  # 使用特征向量数组执行DBSCAN聚类

    # number of clusters, ignoring noise if present
    n_clusters_1 = len(set(labels)) - int(-1 in labels)  # 计算聚类簇的数量，忽略噪声点
    assert n_clusters_1 == n_clusters

    db = DBSCAN(metric=metric, eps=eps, min_samples=min_samples)  # 创建DBSCAN对象
    labels = db.fit(X).labels_  # 对数据进行DBSCAN聚类并获取标签

    n_clusters_2 = len(set(labels)) - int(-1 in labels)  # 计算聚类簇的数量，忽略噪声点
    assert n_clusters_2 == n_clusters


@pytest.mark.parametrize("lil_container", LIL_CONTAINERS)
def test_dbscan_sparse(lil_container):
    core_sparse, labels_sparse = dbscan(lil_container(X), eps=0.8, min_samples=10)  # 使用稀疏矩阵容器执行DBSCAN聚类
    core_dense, labels_dense = dbscan(X, eps=0.8, min_samples=10)  # 使用密集矩阵执行DBSCAN聚类
    assert_array_equal(core_dense, core_sparse)  # 断言核心点结果相等
    assert_array_equal(labels_dense, labels_sparse)  # 断言标签结果相等


@pytest.mark.parametrize("include_self", [False, True])
def test_dbscan_sparse_precomputed(include_self):
    D = pairwise_distances(X)  # 计算数据点之间的距离矩阵
    nn = NearestNeighbors(radius=0.9).fit(X)  # 使用最近邻方法拟合数据
    X_ = X if include_self else None  # 根据include_self条件选择是否包含自身
    D_sparse = nn.radius_neighbors_graph(X=X_, mode="distance")  # 根据半径获取最近邻图的距离矩阵
    # Ensure it is sparse not merely on diagonals:
    assert D_sparse.nnz < D.shape[0] * (D.shape[0] - 1)  # 断言稀疏距离矩阵确保非对角线上的元素少于全连接情况
    core_sparse, labels_sparse = dbscan(
        D_sparse, eps=0.8, min_samples=10, metric="precomputed"
    )  # 使用预计算的稀疏距离矩阵执行DBSCAN聚类
    core_dense, labels_dense = dbscan(D, eps=0.8, min_samples=10, metric="precomputed")  # 使用预计算的密集距离矩阵执行DBSCAN聚类
    # 断言两个数组 core_dense 和 core_sparse 是否相等
    assert_array_equal(core_dense, core_sparse)
    # 断言两个数组 labels_dense 和 labels_sparse 是否相等
    assert_array_equal(labels_dense, labels_sparse)
def test_dbscan_sparse_precomputed_different_eps():
    # 测试当预计算的邻居图是使用比 DBSCAN 的 eps 更大的半径计算时，它会被过滤掉。
    lower_eps = 0.2
    # 使用最近邻居算法拟合数据 X，使用较小的半径 lower_eps
    nn = NearestNeighbors(radius=lower_eps).fit(X)
    # 生成稀疏的邻居距离图 D_sparse
    D_sparse = nn.radius_neighbors_graph(X, mode="distance")
    # 使用预计算的距离矩阵 D_sparse 运行 DBSCAN，设置 eps 为 lower_eps，指定 metric 为 "precomputed"
    dbscan_lower = dbscan(D_sparse, eps=lower_eps, metric="precomputed")

    higher_eps = lower_eps + 0.7
    # 使用较大的半径 higher_eps 重新拟合最近邻居算法
    nn = NearestNeighbors(radius=higher_eps).fit(X)
    # 重新生成稀疏的邻居距离图 D_sparse
    D_sparse = nn.radius_neighbors_graph(X, mode="distance")
    # 使用预计算的距离矩阵 D_sparse 运行 DBSCAN，设置 eps 为 lower_eps，指定 metric 为 "precomputed"
    dbscan_higher = dbscan(D_sparse, eps=lower_eps, metric="precomputed")

    # 断言两次运行 DBSCAN 的结果应该是相同的
    assert_array_equal(dbscan_lower[0], dbscan_higher[0])
    assert_array_equal(dbscan_lower[1], dbscan_higher[1])


@pytest.mark.parametrize("metric", ["precomputed", "minkowski"])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS + [None])
def test_dbscan_input_not_modified(metric, csr_container):
    # 测试确保 DBSCAN 不会修改输入数据
    X = np.random.RandomState(0).rand(10, 10)
    X = csr_container(X) if csr_container is not None else X
    X_copy = X.copy()
    # 运行 DBSCAN 算法
    dbscan(X, metric=metric)

    if csr_container is not None:
        # 如果使用了 CSR 容器，断言 X 没有被修改
        assert_array_equal(X.toarray(), X_copy.toarray())
    else:
        # 如果没有使用 CSR 容器，断言 X 没有被修改
        assert_array_equal(X, X_copy)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_dbscan_input_not_modified_precomputed_sparse_nodiag(csr_container):
    """检查我们不会就地修改预先计算的稀疏矩阵。

    非回归测试用例:
    https://github.com/scikit-learn/scikit-learn/issues/27508
    """
    X = np.random.RandomState(0).rand(10, 10)
    # 在创建稀疏矩阵时，添加对角线上的零值，如果 X 被就地修改，对角线上的零将变为显式零。
    np.fill_diagonal(X, 0)
    X = csr_container(X)
    # 确保稀疏矩阵 X 中的每对非零元素的行和列不相等
    assert all(row != col for row, col in zip(*X.nonzero()))
    X_copy = X.copy()
    # 运行 DBSCAN 算法，指定 metric 为 "precomputed"
    dbscan(X, metric="precomputed")
    # 确保我们没有就地修改 X，即使在创建显式零值时
    assert X.nnz == X_copy.nnz
    assert_array_equal(X.toarray(), X_copy.toarray())


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_dbscan_no_core_samples(csr_container):
    rng = np.random.RandomState(0)
    X = rng.rand(40, 10)
    X[X < 0.8] = 0

    for X_ in [X, csr_container(X)]:
        # 运行 DBSCAN 算法，最小样本数为 6
        db = DBSCAN(min_samples=6).fit(X_)
        # 断言结果中没有核心样本
        assert_array_equal(db.components_, np.empty((0, X_.shape[1])))
        assert_array_equal(db.labels_, -1)
        assert db.core_sample_indices_.shape == (0,)


def test_dbscan_callable():
    # 使用可调用的度量函数测试 DBSCAN 算法。
    # 参数根据具体任务选择。
    eps = 0.8
    min_samples = 10
    # metric 是函数引用，而不是字符串键。
    metric = distance.euclidean
    # 计算 DBSCAN
    # 根据任务选择的参数
    # 使用 DBSCAN 算法对数据集 X 进行聚类，返回核心样本和对应的簇标签
    core_samples, labels = dbscan(
        X, metric=metric, eps=eps, min_samples=min_samples, algorithm="ball_tree"
    )

    # 计算聚类后的簇数量，忽略可能存在的噪声
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    # 断言确保计算得到的簇数量与预期的 n_clusters 相等
    assert n_clusters_1 == n_clusters

    # 创建 DBSCAN 对象，并使用该对象对数据集 X 进行拟合，获取聚类标签
    db = DBSCAN(metric=metric, eps=eps, min_samples=min_samples, algorithm="ball_tree")
    labels = db.fit(X).labels_

    # 计算再次聚类后的簇数量，忽略可能存在的噪声
    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    # 断言确保计算得到的簇数量与预期的 n_clusters 相等
    assert n_clusters_2 == n_clusters
def test_dbscan_metric_params():
    # Tests that DBSCAN works with the metrics_params argument.
    eps = 0.8  # 设置 DBSCAN 的参数 eps
    min_samples = 10  # 设置 DBSCAN 的参数 min_samples
    p = 1  # 设置 Minkowski 距离的参数 p

    # 使用 metric_params 参数计算 DBSCAN

    with warnings.catch_warnings(record=True) as warns:
        # 创建 DBSCAN 对象，使用 ball_tree 算法和 Minkowski 距离
        db = DBSCAN(
            metric="minkowski",
            metric_params={"p": p},  # 设置 Minkowski 距离的参数 p
            eps=eps,
            p=None,  # 忽略 p 参数，因为已经使用了 metric_params
            min_samples=min_samples,
            algorithm="ball_tree",
        ).fit(X)
    assert not warns, warns[0].message

    core_sample_1, labels_1 = db.core_sample_indices_, db.labels_

    # 测试样本标签是否与直接传递 Minkowski 'p' 相同
    db = DBSCAN(
        metric="minkowski", eps=eps, min_samples=min_samples, algorithm="ball_tree", p=p
    ).fit(X)
    core_sample_2, labels_2 = db.core_sample_indices_, db.labels_

    assert_array_equal(core_sample_1, core_sample_2)
    assert_array_equal(labels_1, labels_2)

    # 使用 p=1 的 Minkowski 应等同于曼哈顿距离
    db = DBSCAN(
        metric="manhattan", eps=eps, min_samples=min_samples, algorithm="ball_tree"
    ).fit(X)
    core_sample_3, labels_3 = db.core_sample_indices_, db.labels_

    assert_array_equal(core_sample_1, core_sample_3)
    assert_array_equal(labels_1, labels_3)

    with pytest.warns(
        SyntaxWarning,
        match=(
            "Parameter p is found in metric_params. "
            "The corresponding parameter from __init__ "
            "is ignored."
        ),
    ):
        # 测试确认 p 被忽略，而是使用 metric_params={'p': <val>}
        db = DBSCAN(
            metric="minkowski",
            metric_params={"p": p},  # 设置 Minkowski 距离的参数 p
            eps=eps,
            p=p + 1,  # p 被忽略，因为在 metric_params 中已经指定了
            min_samples=min_samples,
            algorithm="ball_tree",
        ).fit(X)
        core_sample_4, labels_4 = db.core_sample_indices_, db.labels_

    assert_array_equal(core_sample_1, core_sample_4)
    assert_array_equal(labels_1, labels_4)


def test_dbscan_balltree():
    # Tests the DBSCAN algorithm with balltree for neighbor calculation.
    eps = 0.8  # 设置 DBSCAN 的参数 eps
    min_samples = 10  # 设置 DBSCAN 的参数 min_samples

    D = pairwise_distances(X)  # 计算 X 的成对距离矩阵
    core_samples, labels = dbscan(
        D, metric="precomputed", eps=eps, min_samples=min_samples
    )

    # 计算聚类的数量，忽略可能存在的噪声
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    assert n_clusters_1 == n_clusters

    # 创建 DBSCAN 对象，使用 ball_tree 算法和 p=2.0
    db = DBSCAN(p=2.0, eps=eps, min_samples=min_samples, algorithm="ball_tree")
    labels = db.fit(X).labels_

    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    assert n_clusters_2 == n_clusters

    # 创建 DBSCAN 对象，使用 kd_tree 算法和 p=2.0
    db = DBSCAN(p=2.0, eps=eps, min_samples=min_samples, algorithm="kd_tree")
    labels = db.fit(X).labels_

    n_clusters_3 = len(set(labels)) - int(-1 in labels)
    assert n_clusters_3 == n_clusters

    # 创建 DBSCAN 对象，使用 ball_tree 算法和 p=1.0
    db = DBSCAN(p=1.0, eps=eps, min_samples=min_samples, algorithm="ball_tree")
    labels = db.fit(X).labels_

    n_clusters_4 = len(set(labels)) - int(-1 in labels)
    # 断言确保两个变量相等，验证聚类数目是否与预期一致
    assert n_clusters_4 == n_clusters
    
    # 使用DBSCAN算法进行密度聚类分析，设置叶子大小为20，密度半径为eps，最小样本数为min_samples，使用ball_tree算法
    db = DBSCAN(leaf_size=20, eps=eps, min_samples=min_samples, algorithm="ball_tree")
    
    # 对输入数据X进行DBSCAN聚类，并获取每个样本的聚类标签
    labels = db.fit(X).labels_
    
    # 计算实际聚类数目，排除标签中可能的噪声点（标签为-1）
    n_clusters_5 = len(set(labels)) - int(-1 in labels)
    
    # 断言确保新计算的聚类数目与预期一致
    assert n_clusters_5 == n_clusters
def test_input_validation():
    # DBSCAN.fit 应该接受一个列表的列表作为输入数据
    X = [[1.0, 2.0], [3.0, 4.0]]
    DBSCAN().fit(X)  # 不应该抛出异常


def test_pickle():
    obj = DBSCAN()
    s = pickle.dumps(obj)
    # 序列化并反序列化对象，确保类型保持不变
    assert type(pickle.loads(s)) == obj.__class__


def test_boundaries():
    # 确保 min_samples 包含核心点
    core, _ = dbscan([[0], [1]], eps=2, min_samples=2)
    assert 0 in core
    # 确保 eps 包含周围点
    core, _ = dbscan([[0], [1], [1]], eps=1, min_samples=2)
    assert 0 in core
    core, _ = dbscan([[0], [1], [1]], eps=0.99, min_samples=2)
    assert 0 not in core


def test_weighted_dbscan(global_random_seed):
    # 确保 sample_weight 被验证
    with pytest.raises(ValueError):
        dbscan([[0], [1]], sample_weight=[2])
    with pytest.raises(ValueError):
        dbscan([[0], [1]], sample_weight=[2, 3, 4])

    # 确保 sample_weight 生效
    assert_array_equal([], dbscan([[0], [1]], sample_weight=None, min_samples=6)[0])
    assert_array_equal([], dbscan([[0], [1]], sample_weight=[5, 5], min_samples=6)[0])
    assert_array_equal([0], dbscan([[0], [1]], sample_weight=[6, 5], min_samples=6)[0])
    assert_array_equal(
        [0, 1], dbscan([[0], [1]], sample_weight=[6, 6], min_samples=6)[0]
    )

    # 确保在 eps 范围内的点
    assert_array_equal(
        [0, 1], dbscan([[0], [1]], eps=1.5, sample_weight=[5, 1], min_samples=6)[0]
    )
    # 确保非正数和非整数的 sample_weight 的影响
    assert_array_equal(
        [], dbscan([[0], [1]], sample_weight=[5, 0], eps=1.5, min_samples=6)[0]
    )
    assert_array_equal(
        [0, 1], dbscan([[0], [1]], sample_weight=[5.9, 0.1], eps=1.5, min_samples=6)[0]
    )
    assert_array_equal(
        [0, 1], dbscan([[0], [1]], sample_weight=[6, 0], eps=1.5, min_samples=6)[0]
    )
    assert_array_equal(
        [], dbscan([[0], [1]], sample_weight=[6, -1], eps=1.5, min_samples=6)[0]
    )

    # 对于非负的 sample_weight，核心应该与重复后的数据相同
    rng = np.random.RandomState(global_random_seed)
    sample_weight = rng.randint(0, 5, X.shape[0])
    core1, label1 = dbscan(X, sample_weight=sample_weight)
    assert len(label1) == len(X)

    X_repeated = np.repeat(X, sample_weight, axis=0)
    core_repeated, label_repeated = dbscan(X_repeated)
    core_repeated_mask = np.zeros(X_repeated.shape[0], dtype=bool)
    core_repeated_mask[core_repeated] = True
    core_mask = np.zeros(X.shape[0], dtype=bool)
    core_mask[core1] = True
    assert_array_equal(np.repeat(core_mask, sample_weight), core_repeated_mask)

    # sample_weight 应该与预先计算的距离矩阵一起工作
    D = pairwise_distances(X)
    core3, label3 = dbscan(D, sample_weight=sample_weight, metric="precomputed")
    assert_array_equal(core1, core3)
    assert_array_equal(label1, label3)

    # sample_weight 应该与估计器一起工作
    est = DBSCAN().fit(X, sample_weight=sample_weight)
    # 获取当前 DBSCAN 模型的核心样本索引
    core4 = est.core_sample_indices_
    # 获取当前 DBSCAN 模型的标签
    label4 = est.labels_
    # 断言：验证之前计算的核心样本索引与当前模型计算的核心样本索引是否相等
    assert_array_equal(core1, core4)
    # 断言：验证之前计算的标签与当前模型计算的标签是否相等
    assert_array_equal(label1, label4)

    # 初始化一个新的 DBSCAN 模型
    est = DBSCAN()
    # 使用给定数据拟合 DBSCAN 模型，并预测数据的标签，支持样本权重
    label5 = est.fit_predict(X, sample_weight=sample_weight)
    # 获取当前 DBSCAN 模型的核心样本索引
    core5 = est.core_sample_indices_
    # 断言：验证之前计算的核心样本索引与当前模型计算的核心样本索引是否相等
    assert_array_equal(core1, core5)
    # 断言：验证之前计算的标签与当前模型计算的标签是否相等
    assert_array_equal(label1, label5)
    # 断言：验证之前计算的标签与当前 DBSCAN 模型的标签是否相等
    assert_array_equal(label1, est.labels_)
# 使用 pytest.mark.parametrize 装饰器，定义了一个参数化测试函数，参数为算法的选择
@pytest.mark.parametrize("algorithm", ["brute", "kd_tree", "ball_tree"])
def test_dbscan_core_samples_toy(algorithm):
    # 定义一个简单的数据集 X，包含多个单维度的样本
    X = [[0], [2], [3], [4], [6], [8], [10]]
    n_samples = len(X)

    # 对于 eps=1, min_samples=1 的情况，测试所有样本都是核心样本的情况
    core_samples, labels = dbscan(X, algorithm=algorithm, eps=1, min_samples=1)
    assert_array_equal(core_samples, np.arange(n_samples))
    assert_array_equal(labels, [0, 1, 1, 1, 2, 3, 4])

    # 对于 eps=1, min_samples=2 的情况，测试只有密集区域内的样本被标记为核心样本
    core_samples, labels = dbscan(X, algorithm=algorithm, eps=1, min_samples=2)
    assert_array_equal(core_samples, [1, 2, 3])
    assert_array_equal(labels, [-1, 0, 0, 0, -1, -1, -1])

    # 对于 eps=1, min_samples=3 的情况，测试只有密集区域中间的样本被标记为核心样本
    core_samples, labels = dbscan(X, algorithm=algorithm, eps=1, min_samples=3)
    assert_array_equal(core_samples, [2])
    assert_array_equal(labels, [-1, 0, 0, 0, -1, -1, -1])

    # 对于 eps=1, min_samples=4 的情况，测试无法提取核心样本，所有样本都被视为噪声
    core_samples, labels = dbscan(X, algorithm=algorithm, eps=1, min_samples=4)
    assert_array_equal(core_samples, [])
    assert_array_equal(labels, np.full(n_samples, -1.0))


# 定义一个测试函数，测试使用预计算距离矩阵的情况下的 DBSCAN 算法
def test_dbscan_precomputed_metric_with_degenerate_input_arrays():
    # 创建一个单位矩阵 X，使用预计算的距离度量进行 DBSCAN
    X = np.eye(10)
    labels = DBSCAN(eps=0.5, metric="precomputed").fit(X).labels_
    assert len(set(labels)) == 1

    # 创建一个全零矩阵 X，使用预计算的距离度量进行 DBSCAN
    X = np.zeros((10, 10))
    labels = DBSCAN(eps=0.5, metric="precomputed").fit(X).labels_
    assert len(set(labels)) == 1


# 使用 pytest.mark.parametrize 装饰器，定义一个参数化测试函数，测试不同的 CSR 容器
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_dbscan_precomputed_metric_with_initial_rows_zero(csr_container):
    # 创建一个稀疏矩阵 ar，并将其封装成 CSR 容器，然后使用预计算的距离度量进行 DBSCAN
    ar = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.3],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
            [0.0, 0.0, 0.0, 0.0, 0.3, 0.1, 0.0],
        ]
    )
    matrix = csr_container(ar)
    labels = DBSCAN(eps=0.2, metric="precomputed", min_samples=2).fit(matrix).labels_
    assert_array_equal(labels, [-1, -1, 0, 0, 0, 1, 1])
```