# `D:\src\scipysrc\scikit-learn\sklearn\cluster\tests\test_hdbscan.py`

```
"""
Tests for HDBSCAN clustering algorithm
Based on the DBSCAN test code
"""

# 导入所需的库和模块
import numpy as np
import pytest
from scipy import stats
from scipy.spatial import distance

from sklearn.cluster import HDBSCAN  # 导入HDBSCAN聚类算法
from sklearn.cluster._hdbscan._tree import (
    CONDENSED_dtype,  # 导入用于HDBSCAN的树结构相关模块
    _condense_tree,
    _do_labelling,
)
from sklearn.cluster._hdbscan.hdbscan import _OUTLIER_ENCODING  # 导入HDBSCAN的异常值编码
from sklearn.datasets import make_blobs  # 导入生成数据集的函数
from sklearn.metrics import fowlkes_mallows_score  # 导入评估聚类质量的指标
from sklearn.metrics.pairwise import _VALID_METRICS, euclidean_distances  # 导入距离计算相关模块
from sklearn.neighbors import BallTree, KDTree  # 导入用于快速近邻搜索的数据结构
from sklearn.preprocessing import StandardScaler  # 导入数据预处理模块
from sklearn.utils import shuffle  # 导入用于数据洗牌的工具函数
from sklearn.utils._testing import assert_allclose, assert_array_equal  # 导入用于测试的工具函数
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS  # 导入修复工具函数

# 生成样本数据集并标准化处理
X, y = make_blobs(n_samples=200, random_state=10)
X, y = shuffle(X, y, random_state=7)
X = StandardScaler().fit_transform(X)

# 可选的聚类算法列表
ALGORITHMS = [
    "kd_tree",
    "ball_tree",
    "brute",
    "auto",
]

# 定义用于标记异常值的集合
OUTLIER_SET = {-1} | {out["label"] for _, out in _OUTLIER_ENCODING.items()}


def check_label_quality(labels, threshold=0.99):
    """
    检查聚类标签的质量是否满足预期的阈值。

    参数：
    labels : array-like
        聚类算法生成的标签
    threshold : float, optional (default=0.99)
        指定的质量阈值

    断言：
    - 簇的数量等于3
    - 使用Fowlkes-Mallows指数评估标签的质量，要求大于阈值
    """
    n_clusters = len(set(labels) - OUTLIER_SET)
    assert n_clusters == 3
    assert fowlkes_mallows_score(labels, y) > threshold


@pytest.mark.parametrize("outlier_type", _OUTLIER_ENCODING)
def test_outlier_data(outlier_type):
    """
    测试处理特殊异常值（np.inf和np.nan）的功能是否正常。
    """
    outlier = {
        "infinite": np.inf,
        "missing": np.nan,
    }[outlier_type]
    prob_check = {
        "infinite": lambda x, y: x == y,
        "missing": lambda x, y: np.isnan(x),
    }[outlier_type]
    label = _OUTLIER_ENCODING[outlier_type]["label"]
    prob = _OUTLIER_ENCODING[outlier_type]["prob"]

    # 创建包含特殊异常值的新数据集
    X_outlier = X.copy()
    X_outlier[0] = [outlier, 1]
    X_outlier[5] = [outlier, outlier]

    # 使用HDBSCAN模型拟合数据集
    model = HDBSCAN().fit(X_outlier)

    # 验证异常值的标签
    (missing_labels_idx,) = (model.labels_ == label).nonzero()
    assert_array_equal(missing_labels_idx, [0, 5])

    # 验证异常值的概率
    (missing_probs_idx,) = (prob_check(model.probabilities_, prob)).nonzero()
    assert_array_equal(missing_probs_idx, [0, 5])

    # 清理数据集并重新拟合模型
    clean_indices = list(range(1, 5)) + list(range(6, 200))
    clean_model = HDBSCAN().fit(X_outlier[clean_indices])
    assert_array_equal(clean_model.labels_, model.labels_[clean_indices])


def test_hdbscan_distance_matrix():
    """
    测试HDBSCAN是否支持预先计算的距离矩阵，并在需要时抛出适当的错误。
    """
    # 计算原始的欧几里得距离矩阵
    D = euclidean_distances(X)
    D_original = D.copy()

    # 使用预先计算的距离矩阵拟合HDBSCAN模型
    labels = HDBSCAN(metric="precomputed", copy=True).fit_predict(D)

    # 验证距离矩阵未被修改
    assert_allclose(D, D_original)

    # 检查生成的聚类标签的质量
    check_label_quality(labels)

    # 测试传递非对称距离矩阵时是否抛出预期的错误
    msg = r"The precomputed distance matrix.*has shape"
    with pytest.raises(ValueError, match=msg):
        HDBSCAN(metric="precomputed", copy=True).fit_predict(X)

    # 测试传递有问题的距离矩阵时是否抛出预期的错误
    msg = r"The precomputed distance matrix.*values"
    D[0, 1] = 10  # 使距离矩阵不对称
    with pytest.raises(ValueError, match=msg):
        HDBSCAN(metric="precomputed", copy=True).fit_predict(D)
    # 在矩阵 D 的位置 (1, 0) 设置值为 1
    D[1, 0] = 1
    # 使用 pytest 库来测试异常情况，期望捕获 ValueError 异常并匹配指定的错误信息 msg
    with pytest.raises(ValueError, match=msg):
        # 使用 HDBSCAN 算法，基于预先计算的距离矩阵 (precomputed) 进行聚类预测
        HDBSCAN(metric="precomputed").fit_predict(D)
@pytest.mark.parametrize("sparse_constructor", [*CSR_CONTAINERS, *CSC_CONTAINERS])
# 使用pytest的@parametrize装饰器，对sparse_constructor参数进行参数化测试，参数值包括CSR_CONTAINERS和CSC_CONTAINERS中的所有元素
def test_hdbscan_sparse_distance_matrix(sparse_constructor):
    """
    Tests that HDBSCAN works with sparse distance matrices.
    """
    # 计算数据集X的距离矩阵，并将其转换为紧凑形式
    D = distance.squareform(distance.pdist(X))
    # 将距离矩阵D进行归一化处理，除以其最大值
    D /= np.max(D)

    # 计算D的50%分位数作为阈值
    threshold = stats.scoreatpercentile(D.flatten(), 50)

    # 将D中大于等于阈值的元素设为0
    D[D >= threshold] = 0.0
    # 使用sparse_constructor构造稀疏矩阵
    D = sparse_constructor(D)
    # 消除稀疏矩阵中的零元素
    D.eliminate_zeros()

    # 使用metric="precomputed"的HDBSCAN对象对稀疏矩阵D进行聚类并预测标签
    labels = HDBSCAN(metric="precomputed").fit_predict(D)
    # 检查聚类标签的质量
    check_label_quality(labels)


def test_hdbscan_feature_array():
    """
    Tests that HDBSCAN works with feature array, including an arbitrary
    goodness of fit check. Note that the check is a simple heuristic.
    """
    # 使用feature array X对HDBSCAN进行聚类并预测标签
    labels = HDBSCAN().fit_predict(X)

    # 检查聚类的任意好坏度
    # 这是一种启发式方法，用于防止回归
    check_label_quality(labels)


@pytest.mark.parametrize("algo", ALGORITHMS)
@pytest.mark.parametrize("metric", _VALID_METRICS)
# 使用pytest的@parametrize装饰器，对algo和metric参数进行参数化测试，参数值分别为ALGORITHMS和_VALID_METRICS中的所有元素
def test_hdbscan_algorithms(algo, metric):
    """
    Tests that HDBSCAN works with the expected combinations of algorithms and
    metrics, or raises the expected errors.
    """
    # 使用指定的algorithm和metric对HDBSCAN进行聚类并预测标签
    labels = HDBSCAN(algorithm=algo).fit_predict(X)
    # 检查聚类标签的质量
    check_label_quality(labels)

    # 如果algorithm是"brute"或"auto"，则由`pairwise_distances`函数处理brute算法的验证
    if algo in ("brute", "auto"):
        return

    # 对于不支持的metric，期望抛出ValueError异常
    ALGOS_TREES = {
        "kd_tree": KDTree,
        "ball_tree": BallTree,
    }
    metric_params = {
        "mahalanobis": {"V": np.eye(X.shape[1])},
        "seuclidean": {"V": np.ones(X.shape[1])},
        "minkowski": {"p": 2},
        "wminkowski": {"p": 2, "w": np.ones(X.shape[1])},
    }.get(metric, None)

    hdb = HDBSCAN(
        algorithm=algo,
        metric=metric,
        metric_params=metric_params,
    )

    # 如果metric不在ALGOS_TREES[algo].valid_metrics中，预期抛出ValueError异常
    if metric not in ALGOS_TREES[algo].valid_metrics:
        with pytest.raises(ValueError):
            hdb.fit(X)
    # 对于metric为"wminkowski"，预期会有FutureWarning警告
    elif metric == "wminkowski":
        with pytest.warns(FutureWarning):
            hdb.fit(X)
    else:
        hdb.fit(X)


def test_dbscan_clustering():
    """
    Tests that HDBSCAN can generate a sufficiently accurate dbscan clustering.
    This test is more of a sanity check than a rigorous evaluation.
    """
    # 使用默认参数对数据集X进行HDBSCAN聚类
    clusterer = HDBSCAN().fit(X)
    # 使用阈值0.3对聚类结果进行DBSCAN聚类，返回标签
    labels = clusterer.dbscan_clustering(0.3)

    # 由于DBSCAN生成更加紧凑的聚类表示，因此使用较宽松的阈值
    check_label_quality(labels, threshold=0.92)


@pytest.mark.parametrize("cut_distance", (0.1, 0.5, 1))
# 使用pytest的@parametrize装饰器，对cut_distance参数进行参数化测试，参数值为(0.1, 0.5, 1)
def test_dbscan_clustering_outlier_data(cut_distance):
    """
    Tests if np.inf and np.nan data are each treated as special outliers.
    """
    # 获取缺失数据的标签和无限数据的标签
    missing_label = _OUTLIER_ENCODING["missing"]["label"]
    infinite_label = _OUTLIER_ENCODING["infinite"]["label"]

    # 复制数据集X，将第0行的数据修改为[inf, 1]，将第2行的数据修改为[1, nan]，将第5行的数据修改为[inf, nan]
    X_outlier = X.copy()
    X_outlier[0] = [np.inf, 1]
    X_outlier[2] = [1, np.nan]
    X_outlier[5] = [np.inf, np.nan]
    # 使用默认参数对X_outlier进行HDBSCAN聚类
    model = HDBSCAN().fit(X_outlier)
    # 使用 DBSCAN 聚类模型对数据进行聚类，并指定切割距离
    labels = model.dbscan_clustering(cut_distance=cut_distance)

    # 找出标签为 missing_label 的索引位置
    missing_labels_idx = np.flatnonzero(labels == missing_label)
    # 断言确保找到的索引与预期的数组相等
    assert_array_equal(missing_labels_idx, [2, 5])

    # 找出标签为 infinite_label 的索引位置
    infinite_labels_idx = np.flatnonzero(labels == infinite_label)
    # 断言确保找到的索引与预期的数组相等
    assert_array_equal(infinite_labels_idx, [0])

    # 找出干净数据的索引，即既不是 missing_label 也不是 infinite_label 的数据索引
    clean_idx = list(set(range(200)) - set(missing_labels_idx + infinite_labels_idx))
    # 使用清理后的数据训练新的 HDBSCAN 模型
    clean_model = HDBSCAN().fit(X_outlier[clean_idx])
    # 对清理后的数据再次进行 DBSCAN 聚类
    clean_labels = clean_model.dbscan_clustering(cut_distance=cut_distance)
    # 断言确保清理后的标签与原始标签在干净数据索引处的标签相等
    assert_array_equal(clean_labels, labels[clean_idx])
# 测试使用 `BallTree` 的 HDBSCAN 是否有效运行
def test_hdbscan_best_balltree_metric():
    labels = HDBSCAN(
        metric="seuclidean", metric_params={"V": np.ones(X.shape[1])}
    ).fit_predict(X)
    check_label_quality(labels)


# 测试当数据的 `min_cluster_size` 太大时，HDBSCAN 能否正确地不生成有效聚类
def test_hdbscan_no_clusters():
    labels = HDBSCAN(min_cluster_size=len(X) - 1).fit_predict(X)
    assert set(labels).issubset(OUTLIER_SET)


# 测试最小的非噪声聚类是否至少包含 `min_cluster_size` 个点
def test_hdbscan_min_cluster_size():
    for min_cluster_size in range(2, len(X), 1):
        labels = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(X)
        true_labels = [label for label in labels if label != -1]
        if len(true_labels) != 0:
            assert np.min(np.bincount(true_labels)) >= min_cluster_size


# 测试当传递可调用的度量方法给 HDBSCAN 时是否有效运行
def test_hdbscan_callable_metric():
    metric = distance.euclidean
    labels = HDBSCAN(metric=metric).fit_predict(X)
    check_label_quality(labels)


# 使用参数化测试，测试当传递预计算数据且请求基于树的算法时，HDBSCAN 是否能正确地引发错误
@pytest.mark.parametrize("tree", ["kd_tree", "ball_tree"])
def test_hdbscan_precomputed_non_brute(tree):
    hdb = HDBSCAN(metric="precomputed", algorithm=tree)
    msg = "precomputed is not a valid metric for"
    with pytest.raises(ValueError, match=msg):
        hdb.fit(X)


# 使用稀疏特征数据时，测试 HDBSCAN 是否能正确运行，并与稠密数组的结果进行比较
def test_hdbscan_sparse(csr_container):
    dense_labels = HDBSCAN().fit(X).labels_
    check_label_quality(dense_labels)

    _X_sparse = csr_container(X)
    X_sparse = _X_sparse.copy()
    sparse_labels = HDBSCAN().fit(X_sparse).labels_
    assert_array_equal(dense_labels, sparse_labels)

    # 比较稀疏和稠密的非预计算例程是否返回相同的标签，其中第0个观测包含异常值
    for outlier_val, outlier_type in ((np.inf, "infinite"), (np.nan, "missing")):
        X_dense = X.copy()
        X_dense[0, 0] = outlier_val
        dense_labels = HDBSCAN().fit(X_dense).labels_
        check_label_quality(dense_labels)
        assert dense_labels[0] == _OUTLIER_ENCODING[outlier_type]["label"]

        X_sparse = _X_sparse.copy()
        X_sparse[0, 0] = outlier_val
        sparse_labels = HDBSCAN().fit(X_sparse).labels_
        assert_array_equal(dense_labels, sparse_labels)

    msg = "Sparse data matrices only support algorithm `brute`."
    with pytest.raises(ValueError, match=msg):
        HDBSCAN(metric="euclidean", algorithm="ball_tree").fit(X_sparse)
# 使用 pytest 的 @pytest.mark.parametrize 装饰器，参数化测试，测试不同的算法
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_hdbscan_centers(algorithm):
    """
    Tests that HDBSCAN centers are calculated and stored properly, and are
    accurate to the data.
    """
    # 定义中心点坐标
    centers = [(0.0, 0.0), (3.0, 3.0)]
    # 生成包含中心点的数据集
    H, _ = make_blobs(n_samples=2000, random_state=0, centers=centers, cluster_std=0.5)
    # 使用 HDBSCAN 计算聚类中心和中心点，并存储结果
    hdb = HDBSCAN(store_centers="both").fit(H)

    # 遍历中心点、聚类中心和中心点列表，并检查它们之间的距离是否在允许范围内
    for center, centroid, medoid in zip(centers, hdb.centroids_, hdb.medoids_):
        assert_allclose(center, centroid, rtol=1, atol=0.05)
        assert_allclose(center, medoid, rtol=1, atol=0.05)

    # 确保对于噪声点不做任何处理
    hdb = HDBSCAN(
        algorithm=algorithm, store_centers="both", min_cluster_size=X.shape[0]
    ).fit(X)
    assert hdb.centroids_.shape[0] == 0
    assert hdb.medoids_.shape[0] == 0


def test_hdbscan_allow_single_cluster_with_epsilon():
    """
    Tests that HDBSCAN single-cluster selection with epsilon works correctly.
    """
    # 创建随机数据集
    rng = np.random.RandomState(0)
    no_structure = rng.rand(150, 2)
    # 使用 HDBSCAN 进行聚类，设置 epsilon 为 0.0，采用 EOM 方法，允许单一聚类
    labels = HDBSCAN(
        min_cluster_size=5,
        cluster_selection_epsilon=0.0,
        cluster_selection_method="eom",
        allow_single_cluster=True,
    ).fit_predict(no_structure)
    # 计算唯一标签和其出现次数
    unique_labels, counts = np.unique(labels, return_counts=True)
    assert len(unique_labels) == 2

    # 检查噪声点的数量大于 30 的启发式断言
    assert counts[unique_labels == -1] > 30

    # 使用随机种子，设置 epsilon 为 0.18，使用 KD 树算法，验证单一聚类的噪声点数量
    labels = HDBSCAN(
        min_cluster_size=5,
        cluster_selection_epsilon=0.18,
        cluster_selection_method="eom",
        allow_single_cluster=True,
        algorithm="kd_tree",
    ).fit_predict(no_structure)
    unique_labels, counts = np.unique(labels, return_counts=True)
    assert len(unique_labels) == 2
    assert counts[unique_labels == -1] == 2


def test_hdbscan_better_than_dbscan():
    """
    Validate that HDBSCAN can properly cluster this difficult synthetic
    dataset. Note that DBSCAN fails on this (see HDBSCAN plotting
    example)
    """
    # 定义复杂的合成数据集
    centers = [[-0.85, -0.85], [-0.85, 0.85], [3, 3], [3, -3]]
    X, y = make_blobs(
        n_samples=750,
        centers=centers,
        cluster_std=[0.2, 0.35, 1.35, 1.35],
        random_state=0,
    )
    # 使用 HDBSCAN 对数据集进行聚类，获取标签
    labels = HDBSCAN().fit(X).labels_

    # 计算聚类数量，排除噪声点
    n_clusters = len(set(labels)) - int(-1 in labels)
    assert n_clusters == 4
    # 使用 Fowlkes-Mallows 指数评估聚类结果与真实标签的相似性
    fowlkes_mallows_score(labels, y) > 0.99


@pytest.mark.parametrize(
    "kwargs, X",
    [
        ({"metric": "precomputed"}, np.array([[1, np.inf], [np.inf, 1]])),
        ({"metric": "precomputed"}, [[1, 2], [2, 1]]),
        ({}, [[1, 2], [3, 4]]),
    ],
)
def test_hdbscan_usable_inputs(X, kwargs):
    """
    Tests that HDBSCAN works correctly for array-likes and precomputed inputs
    with non-finite points.
    """
    # 使用HDBSCAN算法对数据集X进行聚类，其中min_samples=1表示最小样本数，**kwargs是其他可选参数
    HDBSCAN(min_samples=1, **kwargs).fit(X)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 使用 pytest 的参数化装饰器，对每个 CSR 容器执行以下测试函数
def test_hdbscan_sparse_distances_too_few_nonzero(csr_container):
    """
    Tests that HDBSCAN raises the correct error when there are too few
    non-zero distances.
    """
    # 使用零矩阵创建稀疏 CSR 格式数据
    X = csr_container(np.zeros((10, 10)))

    # 错误消息内容
    msg = "There exists points with fewer than"
    # 断言捕获到 ValueError 并匹配指定消息
    with pytest.raises(ValueError, match=msg):
        HDBSCAN(metric="precomputed").fit(X)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 使用 pytest 的参数化装饰器，对每个 CSR 容器执行以下测试函数
def test_hdbscan_sparse_distances_disconnected_graph(csr_container):
    """
    Tests that HDBSCAN raises the correct error when the distance matrix
    has multiple connected components.
    """
    # 创建包含两个连接组件的对称稀疏矩阵
    X = np.zeros((20, 20))
    X[:5, :5] = 1
    X[5:, 15:] = 1
    X = X + X.T
    X = csr_container(X)
    # 错误消息内容
    msg = "HDBSCAN cannot be perfomed on a disconnected graph"
    # 断言捕获到 ValueError 并匹配指定消息
    with pytest.raises(ValueError, match=msg):
        HDBSCAN(metric="precomputed").fit(X)


def test_hdbscan_tree_invalid_metric():
    """
    Tests that HDBSCAN correctly raises an error for invalid metric choices.
    """
    # 定义一个可调用对象作为无效的度量
    metric_callable = lambda x: x
    msg = (
        ".* is not a valid metric for a .*-based algorithm\\. Please select a different"
        " metric\\."
    )

    # 对于 kd_tree 算法，不支持调用对象作为度量
    with pytest.raises(ValueError, match=msg):
        HDBSCAN(algorithm="kd_tree", metric=metric_callable).fit(X)
    # 对于 ball_tree 算法，同样不支持调用对象作为度量
    with pytest.raises(ValueError, match=msg):
        HDBSCAN(algorithm="ball_tree", metric=metric_callable).fit(X)

    # 在编写此测试时，KDTree 支持的度量是 BallTree 支持度量的严格子集
    metrics_not_kd = list(set(BallTree.valid_metrics) - set(KDTree.valid_metrics))
    if len(metrics_not_kd) > 0:
        # 对于不适用于 KDTree 的度量，同样应该捕获到 ValueError
        with pytest.raises(ValueError, match=msg):
            HDBSCAN(algorithm="kd_tree", metric=metrics_not_kd[0]).fit(X)


def test_hdbscan_too_many_min_samples():
    """
    Tests that HDBSCAN correctly raises an error when setting `min_samples`
    larger than the number of samples.
    """
    # 设置超过样本数目的 min_samples 参数
    hdb = HDBSCAN(min_samples=len(X) + 1)
    # 错误消息内容
    msg = r"min_samples (.*) must be at most"
    # 断言捕获到 ValueError 并匹配指定消息
    with pytest.raises(ValueError, match=msg):
        hdb.fit(X)


def test_hdbscan_precomputed_dense_nan():
    """
    Tests that HDBSCAN correctly raises an error when providing precomputed
    distances with `np.nan` values.
    """
    # 复制 X 矩阵，并将其中一个元素设置为 np.nan
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    # 错误消息内容
    msg = "np.nan values found in precomputed-dense"
    # 创建 HDBSCAN 实例，使用预计算距离度量
    hdb = HDBSCAN(metric="precomputed")
    # 断言捕获到 ValueError 并匹配指定消息
    with pytest.raises(ValueError, match=msg):
        hdb.fit(X_nan)


@pytest.mark.parametrize("allow_single_cluster", [True, False])
@pytest.mark.parametrize("epsilon", [0, 0.1])
def test_labelling_distinct(global_random_seed, allow_single_cluster, epsilon):
    """
    Tests that the `_do_labelling` helper function correctly assigns labels.
    """
    n_samples = 48
    X, y = make_blobs(
        n_samples,
        random_state=global_random_seed,
        # Ensure the clusters are distinct with no overlap
        centers=[
            [0, 0],
            [10, 0],
            [0, 10],
        ],
    )


# 生成具有指定中心点的虚拟数据集 X 和对应的真实标签 y
# n_samples: 数据集中样本的数量
# random_state: 随机数种子，确保结果的可重复性
# centers: 指定生成数据的中心点，确保生成的聚类之间无重叠



    est = HDBSCAN().fit(X)
    condensed_tree = _condense_tree(
        est._single_linkage_tree_, min_cluster_size=est.min_cluster_size
    )
    clusters = {n_samples + 2, n_samples + 3, n_samples + 4}
    cluster_label_map = {n_samples + 2: 0, n_samples + 3: 1, n_samples + 4: 2}
    labels = _do_labelling(
        condensed_tree=condensed_tree,
        clusters=clusters,
        cluster_label_map=cluster_label_map,
        allow_single_cluster=allow_single_cluster,
        cluster_selection_epsilon=epsilon,
    )


# 使用 HDBSCAN 算法对数据集 X 进行聚类
# est: HDBSCAN 的聚类器对象
# condensed_tree: 使用 _condense_tree 函数对 est 的单链接树进行压缩
# min_cluster_size: est 对象的最小聚类大小设定
# clusters: 人为指定的集群（cluster）集合
# cluster_label_map: 将特定集群映射到标签的字典
# labels: 聚类结果的标签，通过 _do_labelling 函数生成
# condensed_tree: 压缩的聚类树对象
# allow_single_cluster: 是否允许单独的集群存在的标志
# cluster_selection_epsilon: 聚类选择的 epsilon 参数



    first_with_label = {_y: np.where(y == _y)[0][0] for _y in list(set(y))}
    y_to_labels = {_y: labels[first_with_label[_y]] for _y in list(set(y))}
    aligned_target = np.vectorize(y_to_labels.get)(y)
    assert_array_equal(labels, aligned_target)


# 建立真实标签 y 到聚类标签的映射，以便进行标签对齐
# first_with_label: 字典，将每个真实标签 _y 映射到其首个出现的索引
# y_to_labels: 字典，将真实标签映射到对应的聚类标签
# aligned_target: 通过 y_to_labels 映射得到的标签对齐结果
# assert_array_equal: 断言验证聚类结果 labels 与 aligned_target 是否一致
# 测试 `_do_labelling` 辅助函数，验证其对给定的 lambda 值使用不同的 `cluster_selection_epsilon` 正确进行阈值化处理。

def test_labelling_thresholding():
    """
    Tests that the `_do_labelling` helper function correctly thresholds the
    incoming lambda values given various `cluster_selection_epsilon` values.
    """
    # 设置样本数
    n_samples = 5
    # 定义最大的 lambda 值
    MAX_LAMBDA = 1.5
    # 构造紧凑树的数组表示
    condensed_tree = np.array(
        [
            (5, 2, MAX_LAMBDA, 1),
            (5, 1, 0.1, 1),
            (5, 0, MAX_LAMBDA, 1),
            (5, 3, 0.2, 1),
            (5, 4, 0.3, 1),
        ],
        dtype=CONDENSED_dtype,
    )
    # 调用 `_do_labelling` 函数进行标记
    labels = _do_labelling(
        condensed_tree=condensed_tree,
        clusters={n_samples},
        cluster_label_map={n_samples: 0, n_samples + 1: 1},
        allow_single_cluster=True,
        cluster_selection_epsilon=1,
    )
    # 计算标记为噪声点的数量
    num_noise = condensed_tree["value"] < 1
    # 断言：标记为噪声点的数量应该与标签为-1的数量相等
    assert sum(num_noise) == sum(labels == -1)

    # 使用不同的 `cluster_selection_epsilon` 再次调用 `_do_labelling`
    labels = _do_labelling(
        condensed_tree=condensed_tree,
        clusters={n_samples},
        cluster_label_map={n_samples: 0, n_samples + 1: 1},
        allow_single_cluster=True,
        cluster_selection_epsilon=0,
    )
    # 根据最大兄弟节点的 lambda 值来计算阈值
    # 在本例中，所有点都是兄弟节点，最大值恰好是 MAX_LAMBDA
    num_noise = condensed_tree["value"] < MAX_LAMBDA
    # 断言：标记为噪声点的数量应该与标签为-1的数量相等
    assert sum(num_noise) == sum(labels == -1)


# TODO(1.6): Remove
# 测试在将来版本中移除的算法名称警告信息
def test_hdbscan_warning_on_deprecated_algorithm_name():
    # 测试当 `algorithm='kdtree'` 时是否显示警告信息
    msg = (
        "`algorithm='kdtree'` has been deprecated in 1.4 and will be renamed"
        " to `algorithm='kd_tree'` in 1.6. To keep the past behaviour, set `algorithm='kd_tree'`."
    )
    with pytest.warns(FutureWarning, match=msg):
        HDBSCAN(algorithm="kdtree").fit(X)

    # 测试当 `algorithm='balltree'` 时是否显示警告信息
    msg = (
        "`algorithm='balltree'` has been deprecated in 1.4 and will be renamed"
        " to `algorithm='ball_tree'` in 1.6. To keep the past behaviour, set"
        " `algorithm='ball_tree'`."
    )
    with pytest.warns(FutureWarning, match=msg):
        HDBSCAN(algorithm="balltree").fit(X)


# 使用参数化测试，检查当 `store_centers` 设置为 "centroid" 或 "medoid" 时，如果请求了预计算的输入矩阵，则是否引发错误
# 非回归测试：确保在 https://github.com/scikit-learn/scikit-learn/issues/27893 中的问题得到解决
@pytest.mark.parametrize("store_centers", ["centroid", "medoid"])
def test_hdbscan_error_precomputed_and_store_centers(store_centers):
    """Check that we raise an error if the centers are requested together with
    a precomputed input matrix.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/27893
    """
    rng = np.random.RandomState(0)
    X = rng.random((100, 2))
    X_dist = euclidean_distances(X)
    err_msg = "Cannot store centers when using a precomputed distance matrix."
    # 断言：使用预计算的距离矩阵时，如果请求存储中心点，则应该引发 ValueError
    with pytest.raises(ValueError, match=err_msg):
        HDBSCAN(metric="precomputed", store_centers=store_centers).fit(X_dist)


# 使用参数化测试，检查当算法设置为 "auto" 或 "brute" 时，HDBSCAN 是否能够正常处理 "cosine" 距离度量
@pytest.mark.parametrize("valid_algo", ["auto", "brute"])
def test_hdbscan_cosine_metric_valid_algorithm(valid_algo):
    """Test that HDBSCAN works with the "cosine" metric when the algorithm is set
    to either "auto" or "brute".
    """
    # 使用 HDBSCAN 算法对给定的数据 X 进行聚类预测
    # 使用 cosine 距离作为度量方式
    # valid_algo 是一个有效的算法选择，可以是 "brute" 或 "auto"
    # 这是一个非回归测试，用于检查问题 #28631 是否修复
    HDBSCAN(metric="cosine", algorithm=valid_algo).fit_predict(X)
# 使用 pytest.mark.parametrize 装饰器标记该测试函数，参数化测试输入的无效算法列表
@pytest.mark.parametrize("invalid_algo", ["kd_tree", "ball_tree"])
# 定义测试函数，测试当在 "cosine" 距离度量中使用不支持的算法时，HDBSCAN 是否会引发详细错误信息
def test_hdbscan_cosine_metric_invalid_algorithm(invalid_algo):
    """Test that HDBSCAN raises an informative error is raised when an unsupported
    algorithm is used with the "cosine" metric.
    """
    # 创建 HDBSCAN 对象，使用 "cosine" 距离度量和参数化的无效算法
    hdbscan = HDBSCAN(metric="cosine", algorithm=invalid_algo)
    # 使用 pytest.raises 检查是否会引发 ValueError，并验证错误信息是否包含指定文本
    with pytest.raises(ValueError, match="cosine is not a valid metric"):
        # 调用 HDBSCAN 的 fit_predict 方法，尝试拟合和预测，预期会引发错误
        hdbscan.fit_predict(X)
```