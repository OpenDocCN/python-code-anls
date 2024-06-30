# `D:\src\scipysrc\scikit-learn\sklearn\neighbors\tests\test_neighbors.py`

```
import re
import warnings
from itertools import product

import joblib
import numpy as np
import pytest
from scipy.sparse import issparse

from sklearn import (
    config_context,
    datasets,
    metrics,
    neighbors,
)
from sklearn.base import clone
from sklearn.exceptions import DataConversionWarning, EfficiencyWarning, NotFittedError
from sklearn.metrics._dist_metrics import (
    DistanceMetric,
)
from sklearn.metrics.pairwise import PAIRWISE_BOOLEAN_FUNCTIONS, pairwise_distances
from sklearn.metrics.tests.test_dist_metrics import BOOL_METRICS
from sklearn.metrics.tests.test_pairwise_distances_reduction import (
    assert_compatible_argkmin_results,
    assert_compatible_radius_results,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import (
    VALID_METRICS_SPARSE,
    KNeighborsRegressor,
)
from sklearn.neighbors._base import (
    KNeighborsMixin,
    _check_precomputed,
    _is_sorted_by_data,
    sort_graph_by_row_values,
)
from sklearn.pipeline import make_pipeline
from sklearn.utils._testing import (
    assert_allclose,
    assert_array_equal,
    ignore_warnings,
)
from sklearn.utils.fixes import (
    BSR_CONTAINERS,
    COO_CONTAINERS,
    CSC_CONTAINERS,
    CSR_CONTAINERS,
    DIA_CONTAINERS,
    DOK_CONTAINERS,
    LIL_CONTAINERS,
    parse_version,
    sp_version,
)
from sklearn.utils.validation import check_random_state

rng = np.random.RandomState(0)
# 加载并打乱鸢尾花数据集
iris = datasets.load_iris()
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# 加载并打乱手写数字数据集
digits = datasets.load_digits()
perm = rng.permutation(digits.target.size)
digits.data = digits.data[perm]
digits.target = digits.target[perm]

SPARSE_TYPES = tuple(
    BSR_CONTAINERS
    + COO_CONTAINERS
    + CSC_CONTAINERS
    + CSR_CONTAINERS
    + DOK_CONTAINERS
    + LIL_CONTAINERS
)
SPARSE_OR_DENSE = SPARSE_TYPES + (np.asarray,)

ALGORITHMS = ("ball_tree", "brute", "kd_tree", "auto")
COMMON_VALID_METRICS = sorted(
    set.intersection(*map(set, neighbors.VALID_METRICS.values()))
)  # type: ignore

P = (1, 2, 3, 4, np.inf)

# 过滤废弃警告
neighbors.kneighbors_graph = ignore_warnings(neighbors.kneighbors_graph)
neighbors.radius_neighbors_graph = ignore_warnings(neighbors.radius_neighbors_graph)

# A list containing metrics where the string specifies the use of the
# DistanceMetric object directly (as resolved in _parse_metric)
DISTANCE_METRIC_OBJS = ["DM_euclidean"]


def _parse_metric(metric: str, dtype=None):
    """
    Helper function for properly building a type-specialized DistanceMetric instances.

    Constructs a type-specialized DistanceMetric instance from a string
    beginning with "DM_" while allowing a pass-through for other metric-specifying
    strings. This is necessary since we wish to parameterize dtype independent of
    metric, yet DistanceMetric requires it for construction.

    """
    # 在字符串metric以"DM_"开头时构建指定类型的DistanceMetric实例，允许其他指定距离度量的字符串直接传递。
    # 这是因为我们希望能够独立于metric参数来参数化dtype，而DistanceMetric在构建时需要dtype参数。
    pass
    # 如果 metric 字符串以 "DM_" 开头
    if metric[:3] == "DM_":
        # 返回根据剔除 "DM_" 后的部分获取的距离度量对象，可以指定 dtype 参数
        return DistanceMetric.get_metric(metric[3:], dtype=dtype)
    
    # 如果 metric 字符串不以 "DM_" 开头，则直接返回 metric 字符串本身
    return metric
# 生成用于测试的 DistanceMetric 的参数列表的函数
def _generate_test_params_for(metric: str, n_features: int):
    """Return list of DistanceMetric kwargs for tests."""

    # 使用种子为1的随机数生成器创建 RNG 对象
    rng = np.random.RandomState(1)

    # 如果 metric 是 "minkowski"
    if metric == "minkowski":
        # 定义多个不同参数的 Minkowski 距离的字典列表
        minkowski_kwargs = [dict(p=1.5), dict(p=2), dict(p=3), dict(p=np.inf)]
        # 如果 scipy 版本 >= 1.8.0.dev0
        if sp_version >= parse_version("1.8.0.dev0"):
            # TODO: 在不再支持 scipy < 1.8.0 时移除此测试。
            # 较新版本的 scipy 直接接受 Minkowski 距离中的权重参数:
            # type: ignore
            minkowski_kwargs.append(dict(p=3, w=rng.rand(n_features)))
        return minkowski_kwargs

    # 如果 metric 是 "seuclidean"
    if metric == "seuclidean":
        # 返回包含 V 参数的字典列表
        return [dict(V=rng.rand(n_features))]

    # 如果 metric 是 "mahalanobis"
    if metric == "mahalanobis":
        # 使用随机数生成器创建 n_features x n_features 的随机矩阵 A
        A = rng.rand(n_features, n_features)
        # 使得矩阵变为对称正定矩阵
        VI = A + A.T + 3 * np.eye(n_features)
        # 返回包含 VI 参数的字典列表
        return [dict(VI=VI)]

    # 对于 "euclidean", "manhattan", "chebyshev", "haversine" 或其他任何 metric
    # 这些情况下不需要额外的参数，返回一个空字典作为参数
    return [{}]


# 替代 lambda d: d ** -2 的权重函数
def _weight_func(dist):
    """Weight function to replace lambda d: d ** -2.
    The lambda function is not valid because:
    if d==0 then 0^-2 is not valid."""

    # dist 可能是多维的，将其展平以便进行循环计算
    with np.errstate(divide="ignore"):
        retval = 1.0 / dist
    return retval**2


# 定义权重函数的列表
WEIGHTS = ["uniform", "distance", _weight_func]


# 使用 pytest 的参数化装饰器定义多组参数进行测试
@pytest.mark.parametrize(
    "n_samples, n_features, n_query_pts, n_neighbors",
    [
        (100, 100, 10, 100),  # 第一组参数
        (1000, 5, 100, 1),    # 第二组参数
    ],
)
@pytest.mark.parametrize("query_is_train", [False, True])  # 是否将查询点作为训练点
@pytest.mark.parametrize("metric", COMMON_VALID_METRICS + DISTANCE_METRIC_OBJS)  # 测试的距离度量方法列表 # type: ignore # noqa
def test_unsupervised_kneighbors(
    global_dtype,
    n_samples,
    n_features,
    n_query_pts,
    n_neighbors,
    query_is_train,
    metric,
):
    # 不同算法在常见的度量方法上必须返回相同的结果，包括返回距离和不返回距离的情况

    # 解析 metric 参数成 DistanceMetric 对象
    metric = _parse_metric(metric, global_dtype)

    # 在本地重新定义随机数生成器，使用相同生成的 X
    local_rng = np.random.RandomState(0)
    X = local_rng.rand(n_samples, n_features).astype(global_dtype, copy=False)

    # 如果 query_is_train 为 True，查询点为训练点 X
    # 否则，生成 n_query_pts x n_features 大小的随机查询点数组
    query = (
        X
        if query_is_train
        else local_rng.rand(n_query_pts, n_features).astype(global_dtype, copy=False)
    )

    # 存储不包含距离的结果的列表
    results_nodist = []
    # 存储包含距离的结果的列表
    results = []
    # 遍历可用的算法列表 ALGORITHMS
    for algorithm in ALGORITHMS:
        # 检查 metric 是否是 DistanceMetric 的实例，并且全局数据类型 global_dtype 是 np.float32
        if isinstance(metric, DistanceMetric) and global_dtype == np.float32:
            # 如果算法名称中包含 "tree"，则跳过当前测试，不进行覆盖率检查
            if "tree" in algorithm:  # pragma: nocover
                pytest.skip(
                    "Neither KDTree nor BallTree support 32-bit distance metric"
                    " objects."
                )
        
        # 创建最近邻对象 neigh，使用给定的参数 n_neighbors, algorithm 和 metric
        neigh = neighbors.NearestNeighbors(
            n_neighbors=n_neighbors, algorithm=algorithm, metric=metric
        )
        
        # 将数据集 X 进行最近邻对象的拟合
        neigh.fit(X)
        
        # 将不返回距离的查询结果添加到 results_nodist 列表中
        results_nodist.append(neigh.kneighbors(query, return_distance=False))
        
        # 将返回距离的查询结果添加到 results 列表中
        results.append(neigh.kneighbors(query, return_distance=True))

    # 遍历结果列表 results 的长度减一次
    for i in range(len(results) - 1):
        # 获取当前算法和下一个算法的名称
        algorithm = ALGORITHMS[i]
        next_algorithm = ALGORITHMS[i + 1]

        # 获取当前结果中的索引列表和下一个结果中的索引列表
        indices_no_dist = results_nodist[i]
        distances, next_distances = results[i][0], results[i + 1][0]
        indices, next_indices = results[i][1], results[i + 1][1]
        
        # 断言当前结果的不返回距离的索引和返回距离的索引相同
        assert_array_equal(
            indices_no_dist,
            indices,
            err_msg=(
                f"The '{algorithm}' algorithm returns different"
                "indices depending on 'return_distances'."
            ),
        )
        
        # 断言当前结果的索引和下一个结果的索引相同
        assert_array_equal(
            indices,
            next_indices,
            err_msg=(
                f"The '{algorithm}' and '{next_algorithm}' "
                "algorithms return different indices."
            ),
        )
        
        # 断言当前结果的距离和下一个结果的距离在指定的误差范围内相同
        assert_allclose(
            distances,
            next_distances,
            err_msg=(
                f"The '{algorithm}' and '{next_algorithm}' "
                "algorithms return different distances."
            ),
            atol=1e-6,
        )
@pytest.mark.parametrize(
    "n_samples, n_features, n_query_pts",
    [
        (100, 100, 10),  # 参数化测试：设定不同的样本数、特征数和查询点数
        (1000, 5, 100),
    ],
)
@pytest.mark.parametrize("metric", COMMON_VALID_METRICS + DISTANCE_METRIC_OBJS)  # type: ignore # noqa
@pytest.mark.parametrize("n_neighbors, radius", [(1, 100), (50, 500), (100, 1000)])  # 参数化测试：设定不同的近邻数和半径
@pytest.mark.parametrize(
    "NeighborsMixinSubclass",
    [
        neighbors.KNeighborsClassifier,  # 测试不同的近邻分类器和回归器
        neighbors.KNeighborsRegressor,
        neighbors.RadiusNeighborsClassifier,
        neighbors.RadiusNeighborsRegressor,
    ],
)
def test_neigh_predictions_algorithm_agnosticity(
    global_dtype,
    n_samples,
    n_features,
    n_query_pts,
    metric,
    n_neighbors,
    radius,
    NeighborsMixinSubclass,
):
    # The different algorithms must return identical predictions results
    # on their common metrics.
    # 测试不同算法在相同的度量标准下返回相同的预测结果

    metric = _parse_metric(metric, global_dtype)  # 解析度量标准

    if isinstance(metric, DistanceMetric):
        if "Classifier" in NeighborsMixinSubclass.__name__:
            pytest.skip(
                "Metrics of type `DistanceMetric` are not yet supported for"
                " classifiers."
            )  # 如果是分类器且使用了距离度量，则跳过测试

        if "Radius" in NeighborsMixinSubclass.__name__:
            pytest.skip(
                "Metrics of type `DistanceMetric` are not yet supported for"
                " radius-neighbor estimators."
            )  # 如果是半径邻居估计器且使用了距离度量，则跳过测试

    # Redefining the rng locally to use the same generated X
    # 在本地重新定义随机数生成器以使用相同的生成数据集 X
    local_rng = np.random.RandomState(0)
    X = local_rng.rand(n_samples, n_features).astype(global_dtype, copy=False)
    y = local_rng.randint(3, size=n_samples)  # 生成随机分类标签

    query = local_rng.rand(n_query_pts, n_features).astype(global_dtype, copy=False)  # 生成随机查询点集合

    predict_results = []

    parameter = (
        n_neighbors if issubclass(NeighborsMixinSubclass, KNeighborsMixin) else radius
    )  # 根据算法类型选择参数

    for algorithm in ALGORITHMS:  # 遍历预定义的算法列表
        if isinstance(metric, DistanceMetric) and global_dtype == np.float32:
            if "tree" in algorithm:  # pragma: nocover
                pytest.skip(
                    "Neither KDTree nor BallTree support 32-bit distance metric"
                    " objects."
                )  # 如果是树结构算法且使用了32位距离度量，则跳过测试

        neigh = NeighborsMixinSubclass(parameter, algorithm=algorithm, metric=metric)  # 初始化邻居模型
        neigh.fit(X, y)  # 拟合模型

        predict_results.append(neigh.predict(query))  # 预测查询点并添加结果

    for i in range(len(predict_results) - 1):
        algorithm = ALGORITHMS[i]
        next_algorithm = ALGORITHMS[i + 1]

        predictions, next_predictions = predict_results[i], predict_results[i + 1]

        assert_allclose(
            predictions,
            next_predictions,
            err_msg=(
                f"The '{algorithm}' and '{next_algorithm}' "
                "algorithms return different predictions."
            ),
        )  # 断言检查：确保相邻算法返回相同的预测结果


@pytest.mark.parametrize(
    "KNeighborsMixinSubclass",
    [
        neighbors.KNeighborsClassifier,  # 参数化测试：测试不同的近邻分类器和回归器以及最近邻模型
        neighbors.KNeighborsRegressor,
        neighbors.NearestNeighbors,
    ],
)
# 测试无监督输入的邻居估计器
def test_unsupervised_inputs(global_dtype, KNeighborsMixinSubclass):
    # 使用全局数据类型创建一个形状为 (10, 3) 的随机数组，作为输入数据 X
    X = rng.random_sample((10, 3)).astype(global_dtype, copy=False)
    # 创建一个形状为 (10,) 的随机整数数组作为目标值 y
    y = rng.randint(3, size=10)
    # 创建 NearestNeighbors 对象 nbrs_fid，设置邻居数为 1，并拟合 X 数据
    nbrs_fid = neighbors.NearestNeighbors(n_neighbors=1)
    nbrs_fid.fit(X)

    # 计算 X 中每个样本的最近邻距离和索引
    dist1, ind1 = nbrs_fid.kneighbors(X)

    # 创建 KNeighborsMixinSubclass 对象 nbrs，设置邻居数为 1
    nbrs = KNeighborsMixinSubclass(n_neighbors=1)

    # 遍历不同的数据源 data，并使用 nbrs 拟合每个数据源和目标值 y
    for data in (nbrs_fid, neighbors.BallTree(X), neighbors.KDTree(X)):
        nbrs.fit(data, y)

        # 计算每个数据源中每个样本的最近邻距离和索引
        dist2, ind2 = nbrs.kneighbors(X)

        # 断言两种方法（nbrs_fid 和 nbrs）计算的距离和索引相等
        assert_allclose(dist1, dist2)
        assert_array_equal(ind1, ind2)


# 测试未拟合错误是否被正确抛出
def test_not_fitted_error_gets_raised():
    # 创建一个包含单个样本的列表 X
    X = [[1]]
    # 创建 NearestNeighbors 对象 neighbors_
    neighbors_ = neighbors.NearestNeighbors()
    # 使用 pytest 检查是否抛出 NotFittedError 异常
    with pytest.raises(NotFittedError):
        neighbors_.kneighbors_graph(X)
    with pytest.raises(NotFittedError):
        neighbors_.radius_neighbors_graph(X)


@pytest.mark.filterwarnings("ignore:EfficiencyWarning")
def check_precomputed(make_train_test, estimators):
    """Tests unsupervised NearestNeighbors with a distance matrix."""
    # 注意：较小的样本可能导致测试结果错误
    rng = np.random.RandomState(42)
    # 创建形状为 (10, 4) 的随机数组 X 和 (3, 4) 的随机数组 Y
    X = rng.random_sample((10, 4))
    Y = rng.random_sample((3, 4))
    # 使用 make_train_test 函数创建训练和测试数据集 DXX, DYX
    DXX, DYX = make_train_test(X, Y)
    # 遍历方法列表，目前只包括 "kneighbors"
    for method in [
        "kneighbors",
    ]:
        # 使用 NearestNeighbors 对象 nbrs_X，设置邻居数为 3，并拟合 X 数据
        nbrs_X = neighbors.NearestNeighbors(n_neighbors=3)
        nbrs_X.fit(X)
        # 计算 Y 中每个样本到 X 中最近邻的距离和索引
        dist_X, ind_X = getattr(nbrs_X, method)(Y)

        # 使用 NearestNeighbors 对象 nbrs_D，设置邻居数为 3，算法为 "brute"，度量为 "precomputed"，并拟合 DXX 数据
        nbrs_D = neighbors.NearestNeighbors(
            n_neighbors=3, algorithm="brute", metric="precomputed"
        )
        nbrs_D.fit(DXX)
        # 计算 DYX 中每个样本到 DXX 中最近邻的距离和索引
        dist_D, ind_D = getattr(nbrs_D, method)(DYX)
        # 断言通过两种方法计算的距离和索引应该相等
        assert_allclose(dist_X, dist_D)
        assert_array_equal(ind_X, ind_D)

        # 使用 NearestNeighbors 对象 nbrs_D，设置邻居数为 3，算法为 "auto"，度量为 "precomputed"，并拟合 DXX 数据
        nbrs_D = neighbors.NearestNeighbors(
            n_neighbors=3, algorithm="auto", metric="precomputed"
        )
        nbrs_D.fit(DXX)
        # 再次计算 DYX 中每个样本到 DXX 中最近邻的距离和索引
        dist_D, ind_D = getattr(nbrs_D, method)(DYX)
        # 断言通过两种方法计算的距离和索引应该相等
        assert_allclose(dist_X, dist_D)
        assert_array_equal(ind_X, ind_D)

        # 检查当 X=None 时的预测情况
        dist_X, ind_X = getattr(nbrs_X, method)(None)
        dist_D, ind_D = getattr(nbrs_D, method)(None)
        # 断言通过两种方法计算的距离和索引应该相等
        assert_allclose(dist_X, dist_D)
        assert_array_equal(ind_X, ind_D)

        # 如果矩阵形状不正确，必须引发 ValueError 异常
        with pytest.raises(ValueError):
            getattr(nbrs_D, method)(X)

    # 创建目标数组 target，其值为 X 的样本数
    target = np.arange(X.shape[0])
    # 遍历估计器列表 estimators
    for Est in estimators:
        # 创建 Est 对象 est，度量为 "euclidean"，邻居数和半径都设置为 1，并用 X 和 target 拟合
        est = Est(metric="euclidean")
        est.radius = est.n_neighbors = 1
        # 使用 X 和 target 训练 est 模型，并对 Y 进行预测
        pred_X = est.fit(X, target).predict(Y)
        # 将度量设置为 "precomputed"，并用 DXX 和 target 拟合 est
        est.metric = "precomputed"
        pred_D = est.fit(DXX, target).predict(DYX)
        # 断言通过两种方法预测的结果应该相等
        assert_allclose(pred_X, pred_D)
    # 定义一个函数 make_train_test，用于生成训练集和测试集的距离矩阵
    def make_train_test(X_train, X_test):
        # 返回训练集内部样本之间的距离矩阵
        return (
            metrics.pairwise_distances(X_train),
            # 返回测试集样本到训练集样本的距离矩阵
            metrics.pairwise_distances(X_test, X_train),
        )
    
    # 创建一个包含多个机器学习估算器类的列表
    estimators = [
        neighbors.KNeighborsClassifier,            # K最近邻分类器
        neighbors.KNeighborsRegressor,             # K最近邻回归器
        neighbors.RadiusNeighborsClassifier,       # 半径最近邻分类器
        neighbors.RadiusNeighborsRegressor,        # 半径最近邻回归器
    ]
    
    # 调用 check_precomputed 函数，验证 make_train_test 函数是否支持估算器列表中的估算器
    check_precomputed(make_train_test, estimators)
@pytest.mark.parametrize("fmt", ["csr", "lil"])
def test_precomputed_sparse_knn(fmt):
    # 定义内部函数make_train_test，用于生成训练集和测试集的稀疏k近邻图
    def make_train_test(X_train, X_test):
        # 创建一个具有3个最近邻的最近邻模型，并对训练集进行拟合
        nn = neighbors.NearestNeighbors(n_neighbors=3 + 1).fit(X_train)
        # 返回训练集和测试集的k近邻图，以指定的稀疏格式表示
        return (
            nn.kneighbors_graph(X_train, mode="distance").asformat(fmt),
            nn.kneighbors_graph(X_test, mode="distance").asformat(fmt),
        )

    # 我们不测试基于半径的最近邻分类器和回归器，因为预计算的邻居图仅使用k近邻
    estimators = [
        neighbors.KNeighborsClassifier,
        neighbors.KNeighborsRegressor,
    ]
    # 调用通用函数，检查预计算邻居图在指定的估算器上的表现
    check_precomputed(make_train_test, estimators)


@pytest.mark.parametrize("fmt", ["csr", "lil"])
def test_precomputed_sparse_radius(fmt):
    # 定义内部函数make_train_test，用于生成训练集和测试集的基于半径的稀疏邻居图
    def make_train_test(X_train, X_test):
        # 创建一个基于半径的最近邻模型，并对训练集进行拟合
        nn = neighbors.NearestNeighbors(radius=1).fit(X_train)
        # 返回训练集和测试集的基于半径的邻居图，以指定的稀疏格式表示
        return (
            nn.radius_neighbors_graph(X_train, mode="distance").asformat(fmt),
            nn.radius_neighbors_graph(X_test, mode="distance").asformat(fmt),
        )

    # 我们不测试k近邻分类器和回归器，因为预计算的邻居图使用了基于半径的方法
    estimators = [
        neighbors.RadiusNeighborsClassifier,
        neighbors.RadiusNeighborsRegressor,
    ]
    # 调用通用函数，检查预计算邻居图在指定的估算器上的表现
    check_precomputed(make_train_test, estimators)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_is_sorted_by_data(csr_container):
    # 测试_is_sorted_by_data函数的预期行为。在CSR稀疏矩阵中，每行的条目可以按索引、数据排序或未排序。
    # 当条目按数据排序时，_is_sorted_by_data应返回True；在所有其他情况下返回False。

    # 测试排序的单行稀疏数组
    X = csr_container(np.arange(10).reshape(1, 10))
    assert _is_sorted_by_data(X)
    # 测试未排序的1D数组
    X[0, 2] = 5
    assert not _is_sorted_by_data(X)

    # 当每个样本的数据排序时，但不一定在样本之间排序时的测试
    X = csr_container([[0, 1, 2], [3, 0, 0], [3, 4, 0], [1, 0, 2]])
    assert _is_sorted_by_data(X)

    # 测试X.indptr中存在重复条目的情况
    data, indices, indptr = [0, 4, 2, 2], [0, 1, 1, 1], [0, 2, 2, 4]
    X = csr_container((data, indices, indptr), shape=(3, 3))
    assert _is_sorted_by_data(X)


@pytest.mark.filterwarnings("ignore:EfficiencyWarning")
@pytest.mark.parametrize("function", [sort_graph_by_row_values, _check_precomputed])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sort_graph_by_row_values(function, csr_container):
    # 测试sort_graph_by_row_values函数返回按行值排序的图的功能
    X = csr_container(np.abs(np.random.RandomState(42).randn(10, 10)))
    assert not _is_sorted_by_data(X)
    Xt = function(X)
    assert _is_sorted_by_data(Xt)

    # 测试具有每个样本不同非零条目数量的情况
    mask = np.random.RandomState(42).randint(2, size=(10, 10))
    # 将稀疏矩阵转换为密集矩阵表示
    X = X.toarray()
    # 根据掩码条件将部分元素置为零
    X[mask == 1] = 0
    # 将修改后的矩阵重新封装为稀疏矩阵
    X = csr_container(X)
    # 断言函数 `_is_sorted_by_data` 返回 False，即确认矩阵 X 按数据未排序
    assert not _is_sorted_by_data(X)
    # 对矩阵 X 进行函数 `function` 的处理
    Xt = function(X)
    # 断言函数 `_is_sorted_by_data` 返回 True，即确认处理后的矩阵 Xt 按数据排序
    assert _is_sorted_by_data(Xt)
# 标记测试以忽略效率警告
@pytest.mark.filterwarnings("ignore:EfficiencyWarning")
# 参数化测试，使用不同的 CSR 容器进行测试
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 测试 sort_graph_by_row_values 函数，验证其在复制时是否按预期排序
def test_sort_graph_by_row_values_copy(csr_container):
    # 创建 CSR 矩阵 X_，元素为随机生成的绝对值
    X_ = csr_container(np.abs(np.random.RandomState(42).randn(10, 10)))
    # 断言 X_ 不是按数据排序的
    assert not _is_sorted_by_data(X_)

    # 复制 X_ 到 X
    X = X_.copy()
    # 断言 sort_graph_by_row_values 函数在不复制时是原地操作，并且返回的数据与 X 的数据相同
    assert sort_graph_by_row_values(X).data is X.data

    X = X_.copy()
    # 断言 sort_graph_by_row_values 函数在 copy=False 时是原地操作，并且返回的数据与 X 的数据相同
    assert sort_graph_by_row_values(X, copy=False).data is X.data

    X = X_.copy()
    # 断言 sort_graph_by_row_values 函数在 copy=True 时不是原地操作，并且返回的数据与 X 的数据不同
    assert sort_graph_by_row_values(X, copy=True).data is not X.data

    # 复制 X_ 到 X
    X = X_.copy()
    # 断言 _check_precomputed 函数从不是原地操作，并且返回的数据与 X 的数据不同
    assert _check_precomputed(X).data is not X.data

    # 当 X 不是 CSR 格式且 copy=True 时，不会引发异常
    sort_graph_by_row_values(X.tocsc(), copy=True)

    # 当 X 不是 CSR 格式且 copy=False 时，会引发 ValueError 异常，匹配 "Use copy=True to allow the conversion"
    with pytest.raises(ValueError, match="Use copy=True to allow the conversion"):
        sort_graph_by_row_values(X.tocsc(), copy=False)


# 参数化测试，使用不同的 CSR 容器进行测试
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 测试 sort_graph_by_row_values 函数的警告功能
def test_sort_graph_by_row_values_warning(csr_container):
    # 创建 CSR 矩阵 X，元素为随机生成的绝对值
    X = csr_container(np.abs(np.random.RandomState(42).randn(10, 10)))
    # 断言 X 不是按数据排序的
    assert not _is_sorted_by_data(X)

    # 测试警告功能
    # 使用 pytest.warns 检查 EfficiencyWarning 警告，并匹配 "was not sorted by row values"
    with pytest.warns(EfficiencyWarning, match="was not sorted by row values"):
        sort_graph_by_row_values(X, copy=True)
    with pytest.warns(EfficiencyWarning, match="was not sorted by row values"):
        sort_graph_by_row_values(X, copy=True, warn_when_not_sorted=True)
    with pytest.warns(EfficiencyWarning, match="was not sorted by row values"):
        _check_precomputed(X)

    # 使用 warnings.catch_warnings() 确保没有警告
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sort_graph_by_row_values(X, copy=True, warn_when_not_sorted=False)


# 参数化测试，使用不同的稀疏容器进行测试
@pytest.mark.parametrize(
    "sparse_container", DOK_CONTAINERS + BSR_CONTAINERS + DIA_CONTAINERS
)
# 测试 sort_graph_by_row_values 和 _check_precomputed 函数在不支持的稀疏格式上是否会引发 TypeError
def test_sort_graph_by_row_values_bad_sparse_format(sparse_container):
    # 创建不支持的稀疏矩阵 X，元素为随机生成的绝对值
    X = sparse_container(np.abs(np.random.RandomState(42).randn(10, 10)))
    # 使用 pytest.raises 检查是否会引发 TypeError 异常，并匹配 "format is not supported"
    with pytest.raises(TypeError, match="format is not supported"):
        sort_graph_by_row_values(X)
    with pytest.raises(TypeError, match="format is not supported"):
        _check_precomputed(X)


# 标记测试以忽略效率警告
@pytest.mark.filterwarnings("ignore:EfficiencyWarning")
# 参数化测试，使用不同的 CSR 容器进行测试
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 测试 precomputed 稀疏矩阵在无效输入时是否会引发异常
def test_precomputed_sparse_invalid(csr_container):
    # 创建 CSR 格式的距离矩阵 dist_csr，元素为随机生成的距离值
    dist = np.array([[0.0, 2.0, 1.0], [2.0, 0.0, 3.0], [1.0, 3.0, 0.0]])
    dist_csr = csr_container(dist)
    # 创建 NearestNeighbors 对象 neigh，使用 metric="precomputed"，并拟合 dist_csr
    neigh = neighbors.NearestNeighbors(n_neighbors=1, metric="precomputed")
    neigh.fit(dist_csr)
    # 查询最近邻
    neigh.kneighbors(None, n_neighbors=1)
    neigh.kneighbors(np.array([[0.0, 0.0, 0.0]]), n_neighbors=2)

    # 确保足够数量的最近邻
    # 创建一个 3x3 的 NumPy 数组作为距离矩阵，表示样本之间的距离
    dist = np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 3.0], [0.0, 3.0, 0.0]])
    # 将 NumPy 数组转换为 CSR 格式的稀疏矩阵
    dist_csr = csr_container(dist)
    # 使用邻居对象(neigh)拟合(训练)稀疏矩阵
    neigh.fit(dist_csr)
    # 定义一个错误消息，指示每个样本至少需要 2 个邻居，但某些样本只有 1 个
    msg = "2 neighbors per samples are required, but some samples have only 1"
    # 使用 pytest 检测是否会抛出 ValueError 异常，并且异常消息与预期的错误消息相匹配
    with pytest.raises(ValueError, match=msg):
        # 调用邻居对象(neigh)的 kneighbors 方法，并传递 None 和 n_neighbors=1 参数
        neigh.kneighbors(None, n_neighbors=1)

    # Checks error with inconsistent distance matrix
    # 创建另一个不一致的距离矩阵作为 NumPy 数组
    dist = np.array([[5.0, 2.0, 1.0], [-2.0, 0.0, 3.0], [1.0, 3.0, 0.0]])
    # 将新的距离矩阵转换为 CSR 格式的稀疏矩阵
    dist_csr = csr_container(dist)
    # 定义一个错误消息，指示数据中存在负值，不符合预期的预计算距离矩阵格式
    msg = "Negative values in data passed to precomputed distance matrix."
    # 使用 pytest 检测是否会抛出 ValueError 异常，并且异常消息与预期的错误消息相匹配
    with pytest.raises(ValueError, match=msg):
        # 调用邻居对象(neigh)的 kneighbors 方法，并传递 dist_csr 和 n_neighbors=1 参数
        neigh.kneighbors(dist_csr, n_neighbors=1)
# 测试预先计算的交叉验证功能
def test_precomputed_cross_validation():
    # 创建一个特定的随机数生成器实例
    rng = np.random.RandomState(0)
    # 创建一个随机数组，大小为20x2，元素值在[0,1)之间
    X = rng.rand(20, 2)
    # 计算数据集X中每对样本之间的欧氏距离，返回距离矩阵D
    D = pairwise_distances(X, metric="euclidean")
    # 随机生成一个长度为20的整数数组，元素值在[0,3)之间，用作分类标签y
    y = rng.randint(3, size=20)
    # 遍历不同的估计器Est，进行交叉验证评估
    for Est in (
        neighbors.KNeighborsClassifier,
        neighbors.RadiusNeighborsClassifier,
        neighbors.KNeighborsRegressor,
        neighbors.RadiusNeighborsRegressor,
    ):
        # 使用默认参数进行交叉验证，返回基于距离度量的得分
        metric_score = cross_val_score(Est(), X, y)
        # 使用预计算的距离矩阵D进行交叉验证，返回基于距离度量的得分
        precomp_score = cross_val_score(Est(metric="precomputed"), D, y)
        # 断言两种方式的评分结果应当一致
        assert_array_equal(metric_score, precomp_score)


def test_unsupervised_radius_neighbors(
    global_dtype, n_samples=20, n_features=5, n_query_pts=2, radius=0.5, random_state=0
):
    # 测试无监督的基于半径的查询
    rng = np.random.RandomState(random_state)
    # 创建一个大小为n_samples x n_features的随机浮点数数组X
    X = rng.rand(n_samples, n_features).astype(global_dtype, copy=False)

    # 创建一个大小为n_query_pts x n_features的随机浮点数数组test
    test = rng.rand(n_query_pts, n_features).astype(global_dtype, copy=False)

    for p in P:  # P应为外部定义的可迭代对象
        results = []

        for algorithm in ALGORITHMS:  # ALGORITHMS应为外部定义的可迭代对象
            # 使用给定的半径、算法和距离度量p创建最近邻搜索器实例neigh
            neigh = neighbors.NearestNeighbors(radius=radius, algorithm=algorithm, p=p)
            # 在数据集X上拟合最近邻搜索器neigh
            neigh.fit(X)

            # 对测试数据集test进行半径查询，返回每个点的邻居索引ind1
            ind1 = neigh.radius_neighbors(test, return_distance=False)

            # 对测试数据集test进行半径查询，返回每个点的邻居距离和索引dist、ind
            dist, ind = neigh.radius_neighbors(test, return_distance=True)

            # 对查询结果进行排序，半径查询结果不会自动排序
            for d, i, i1 in zip(dist, ind, ind1):
                j = d.argsort()
                d[:] = d[j]
                i[:] = i[j]
                i1[:] = i1[j]

            # 将排序后的结果(dist, ind)添加到results列表中
            results.append((dist, ind))

            # 断言排序后的索引结果应当与未排序的索引结果ind1一致
            assert_allclose(np.concatenate(list(ind)), np.concatenate(list(ind1)))

        # 断言不同算法的结果应当一致
        for i in range(len(results) - 1):
            assert_allclose(
                np.concatenate(list(results[i][0])),
                np.concatenate(list(results[i + 1][0])),
            )
            assert_allclose(
                np.concatenate(list(results[i][1])),
                np.concatenate(list(results[i + 1][1])),
            )


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("weights", WEIGHTS)
def test_kneighbors_classifier(
    global_dtype,
    algorithm,
    weights,
    n_samples=40,
    n_features=5,
    n_test_pts=10,
    n_neighbors=5,
    random_state=0,
):
    # 测试K近邻分类器
    rng = np.random.RandomState(random_state)
    # 创建一个大小为n_samples x n_features的随机浮点数数组X
    X = 2 * rng.rand(n_samples, n_features).astype(global_dtype, copy=False) - 1
    # 根据X的欧氏距离判断样本类别标签，生成二分类标签y
    y = ((X**2).sum(axis=1) < 0.5).astype(int)
    # 将y转换为字符串类型，用作K近邻分类器的分类标签
    y_str = y.astype(str)

    # 创建K近邻分类器实例knn，设置参数：近邻数、权重方式、算法
    knn = neighbors.KNeighborsClassifier(
        n_neighbors=n_neighbors, weights=weights, algorithm=algorithm
    )
    # 在数据集X上拟合K近邻分类器knn
    knn.fit(X, y)
    # 创建一个微小扰动epsilon，对前n_test_pts个样本进行预测
    epsilon = 1e-5 * (2 * rng.rand(1, n_features) - 1)
    y_pred = knn.predict(X[:n_test_pts] + epsilon)
    # 断言预测结果应当与真实标签y[:n_test_pts]一致
    assert_array_equal(y_pred, y[:n_test_pts])
    
    # 使用字符串类型的标签y_str重新拟合K近邻分类器knn
    knn.fit(X, y_str)
    # 对前n_test_pts个样本进行预测
    y_pred = knn.predict(X[:n_test_pts] + epsilon)
    # 使用断言检查预测值数组 `y_pred` 是否与真实标签数组 `y_str` 的前 `n_test_pts` 个元素相等
    assert_array_equal(y_pred, y_str[:n_test_pts])
# 测试K近邻分类器，使用浮点标签
def test_kneighbors_classifier_float_labels(
    global_dtype,  # 全局数据类型
    n_samples=40,  # 样本数量，默认为40
    n_features=5,  # 特征数量，默认为5
    n_test_pts=10,  # 测试点数量，默认为10
    n_neighbors=5,  # 近邻数量，默认为5
    random_state=0,  # 随机数种子，默认为0
):
    # 随机数生成器，使用指定的随机数种子
    rng = np.random.RandomState(random_state)
    # 生成服从指定数据类型的随机样本矩阵，数值范围在[-1, 1)
    X = 2 * rng.rand(n_samples, n_features).astype(global_dtype, copy=False) - 1
    # 创建分类标签，基于每行向量元素平方和是否小于0.5进行判定，转换为整数类型
    y = ((X**2).sum(axis=1) < 0.5).astype(int)

    # 创建K近邻分类器对象
    knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    # 使用K近邻算法拟合模型，将标签转换为浮点类型
    knn.fit(X, y.astype(float))
    # 生成微小扰动
    epsilon = 1e-5 * (2 * rng.rand(1, n_features) - 1)
    # 对前n_test_pts个样本进行预测，加入扰动
    y_pred = knn.predict(X[:n_test_pts] + epsilon)
    # 断言预测结果与真实标签一致
    assert_array_equal(y_pred, y[:n_test_pts])


# 测试K近邻分类器的predict_proba()方法
def test_kneighbors_classifier_predict_proba(global_dtype):
    # 创建样本特征矩阵，使用指定的全局数据类型
    X = np.array(
        [[0, 2, 0], [0, 2, 1], [2, 0, 0], [2, 2, 0], [0, 0, 2], [0, 0, 1]]
    ).astype(global_dtype, copy=False)
    # 创建分类标签数组
    y = np.array([4, 4, 5, 5, 1, 1])
    # 创建K近邻分类器对象，使用曼哈顿距离（p=1）
    cls = neighbors.KNeighborsClassifier(n_neighbors=3, p=1)
    # 使用样本数据拟合分类器
    cls.fit(X, y)
    # 预测每个类的概率
    y_prob = cls.predict_proba(X)
    # 真实概率数组，按行归一化
    real_prob = (
        np.array(
            [
                [0, 2, 1],
                [1, 2, 0],
                [1, 0, 2],
                [0, 1, 2],
                [2, 1, 0],
                [2, 1, 0],
            ]
        )
        / 3.0
    )
    # 断言预测概率与真实概率数组一致
    assert_array_equal(real_prob, y_prob)
    # 检查非整数标签下是否仍有效
    cls.fit(X, y.astype(str))
    y_prob = cls.predict_proba(X)
    assert_array_equal(real_prob, y_prob)
    # 检查使用weights='distance'参数是否有效
    cls = neighbors.KNeighborsClassifier(n_neighbors=2, p=1, weights="distance")
    cls.fit(X, y)
    y_prob = cls.predict_proba(np.array([[0, 2, 0], [2, 2, 2]]))
    real_prob = np.array([[0, 1, 0], [0, 0.4, 0.6]])
    assert_allclose(real_prob, y_prob)


# 使用参数化测试框架测试半径邻居分类器
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("weights", WEIGHTS)
def test_radius_neighbors_classifier(
    global_dtype,  # 全局数据类型
    algorithm,  # 算法选择参数
    weights,  # 权重参数
    n_samples=40,  # 样本数量，默认为40
    n_features=5,  # 特征数量，默认为5
    n_test_pts=10,  # 测试点数量，默认为10
    radius=0.5,  # 邻域半径，默认为0.5
    random_state=0,  # 随机数种子，默认为0
):
    # 随机数生成器，使用指定的随机数种子
    rng = np.random.RandomState(random_state)
    # 生成服从指定数据类型的随机样本矩阵，数值范围在[-1, 1)
    X = 2 * rng.rand(n_samples, n_features).astype(global_dtype, copy=False) - 1
    # 创建分类标签，基于每行向量元素平方和是否小于半径进行判定，转换为整数类型
    y = ((X**2).sum(axis=1) < radius).astype(int)
    # 将整数标签转换为字符串类型
    y_str = y.astype(str)

    # 创建半径邻居分类器对象
    neigh = neighbors.RadiusNeighborsClassifier(
        radius=radius, weights=weights, algorithm=algorithm
    )
    # 使用半径邻居算法拟合模型
    neigh.fit(X, y)
    # 生成微小扰动
    epsilon = 1e-5 * (2 * rng.rand(1, n_features) - 1)
    # 对前n_test_pts个样本进行预测，加入扰动
    y_pred = neigh.predict(X[:n_test_pts] + epsilon)
    # 断言预测结果与真实标签一致
    assert_array_equal(y_pred, y[:n_test_pts])
    # 使用字符串类型的标签重新拟合模型
    neigh.fit(X, y_str)
    # 对前n_test_pts个样本进行预测，加入扰动
    y_pred = neigh.predict(X[:n_test_pts] + epsilon)
    # 断言预测结果与真实字符串标签一致
    assert_array_equal(y_pred, y_str[:n_test_pts])


# 参数化测试框架的组合测试，测试半径邻居分类器当没有邻居时的情况
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("weights", WEIGHTS)
@pytest.mark.parametrize("outlier_label", [0, -1, None])
def test_radius_neighbors_classifier_when_no_neighbors(
    global_dtype, algorithm, weights, outlier_label


    # 声明全局变量：global_dtype、algorithm、weights、outlier_label
# Test radius-based classifier when no neighbors found.
# In this case it should raise an informative exception

X = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=global_dtype)
y = np.array([1, 2])
radius = 0.1

# no outliers
z1 = np.array([[1.01, 1.01], [2.01, 2.01]], dtype=global_dtype)

# one outlier
z2 = np.array([[1.01, 1.01], [1.4, 1.4]], dtype=global_dtype)

# Create an instance of RadiusNeighborsClassifier
rnc = neighbors.RadiusNeighborsClassifier
clf = rnc(
    radius=radius,
    weights=weights,
    algorithm=algorithm,
    outlier_label=outlier_label,
)
# Fit the classifier with training data
clf.fit(X, y)
# Assert that predictions for z1 match the correct labels
assert_array_equal(np.array([1, 2]), clf.predict(z1))
# If outlier_label is None, ensure a ValueError is raised when predicting z2
if outlier_label is None:
    with pytest.raises(ValueError):
        clf.predict(z2)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("weights", WEIGHTS)
def test_radius_neighbors_classifier_outlier_labeling(global_dtype, algorithm, weights):
    # Test radius-based classifier when no neighbors found and outliers
    # are labeled.

    X = np.array(
        [[1.0, 1.0], [2.0, 2.0], [0.99, 0.99], [0.98, 0.98], [2.01, 2.01]],
        dtype=global_dtype,
    )
    y = np.array([1, 2, 1, 1, 2])
    radius = 0.1

    # no outliers
    z1 = np.array([[1.01, 1.01], [2.01, 2.01]], dtype=global_dtype)

    # one outlier
    z2 = np.array([[1.4, 1.4], [1.01, 1.01], [2.01, 2.01]], dtype=global_dtype)

    correct_labels1 = np.array([1, 2])
    correct_labels2 = np.array([-1, 1, 2])
    outlier_proba = np.array([0, 0])

    # Create a RadiusNeighborsClassifier instance with specified parameters
    clf = neighbors.RadiusNeighborsClassifier(
        radius=radius, weights=weights, algorithm=algorithm, outlier_label=-1
    )
    # Fit the classifier with training data
    clf.fit(X, y)
    # Assert that predictions for z1 match correct_labels1
    assert_array_equal(correct_labels1, clf.predict(z1))
    # Assert that predicting z2 raises a UserWarning due to outlier label mismatch
    with pytest.warns(UserWarning, match="Outlier label -1 is not in training classes"):
        assert_array_equal(correct_labels2, clf.predict(z2))
    # Assert that predicting probabilities for z2 also raises a warning
    with pytest.warns(UserWarning, match="Outlier label -1 is not in training classes"):
        assert_allclose(outlier_proba, clf.predict_proba(z2)[0])

    # test outlier_labeling of using predict_proba()
    RNC = neighbors.RadiusNeighborsClassifier
    X = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]], dtype=global_dtype)
    y = np.array([0, 2, 2, 1, 1, 1, 3, 3, 3, 3])

    # Define a function to check for TypeError when outlier_label is an array
    def check_array_exception():
        clf = RNC(radius=1, outlier_label=[[5]])
        clf.fit(X, y)

    # Assert that check_array_exception raises TypeError
    with pytest.raises(TypeError):
        check_array_exception()

    # Define a function to check for TypeError when outlier_label has invalid dtype
    def check_dtype_exception():
        clf = RNC(radius=1, outlier_label="a")
        clf.fit(X, y)

    # Assert that check_dtype_exception raises TypeError
    with pytest.raises(TypeError):
        check_dtype_exception()

    # Test setting outlier_label to "most_frequent"
    clf = RNC(radius=1, outlier_label="most_frequent")
    clf.fit(X, y)
    # Predict probabilities for [1] and [15] and assert that [15] has [0, 0, 0, 1]
    proba = clf.predict_proba([[1], [15]])
    assert_array_equal(proba[1, :], [0, 0, 0, 1])

    # Test manual label assignment in y
    clf = RNC(radius=1, outlier_label=1)
    clf.fit(X, y)
    # 使用分类器 clf 对输入数据进行预测，返回预测类别的概率分布
    proba = clf.predict_proba([[1], [15]])
    # 验证预测的概率分布是否与期望的数组相等
    assert_array_equal(proba[1, :], [0, 1, 0, 0])
    # 使用分类器 clf 对输入数据进行预测，返回预测的类别
    pred = clf.predict([[1], [15]])
    # 验证预测的类别是否与期望的数组相等
    assert_array_equal(pred, [2, 1])

    # 测试手动标签超出 y 范围的警告
    def check_warning():
        # 创建一个 RNC 分类器，设置半径为 1，异常标签为 4
        clf = RNC(radius=1, outlier_label=4)
        # 使用输入数据 X, y 进行拟合
        clf.fit(X, y)
        # 使用分类器进行预测概率分布，验证是否产生用户警告
        clf.predict_proba([[1], [15]])

    # 预期代码块会产生 UserWarning 警告
    with pytest.warns(UserWarning):
        check_warning()

    # 测试多输出相同异常标签的情况
    y_multi = [
        [0, 1],
        [2, 1],
        [2, 2],
        [1, 2],
        [1, 2],
        [1, 3],
        [3, 3],
        [3, 3],
        [3, 0],
        [3, 0],
    ]
    # 创建一个 RNC 分类器，设置半径为 1，异常标签为 1
    clf = RNC(radius=1, outlier_label=1)
    # 使用输入数据 X, y_multi 进行拟合
    clf.fit(X, y_multi)
    # 使用分类器进行预测概率分布
    proba = clf.predict_proba([[7], [15]])
    # 验证预测的第二个输出的概率分布是否与期望的数组相等
    assert_array_equal(proba[1][1, :], [0, 1, 0, 0])
    # 使用分类器进行预测，验证预测的第二个输出的类别是否与期望的数组相等
    pred = clf.predict([[7], [15]])
    assert_array_equal(pred[1, :], [1, 1])

    # 测试多输出不同异常标签的情况
    y_multi = [
        [0, 0],
        [2, 2],
        [2, 2],
        [1, 1],
        [1, 1],
        [1, 1],
        [3, 3],
        [3, 3],
        [3, 3],
        [3, 3],
    ]
    # 创建一个 RNC 分类器，设置半径为 1，异常标签分别为 [0, 1]
    clf = RNC(radius=1, outlier_label=[0, 1])
    # 使用输入数据 X, y_multi 进行拟合
    clf.fit(X, y_multi)
    # 使用分类器进行预测概率分布
    proba = clf.predict_proba([[7], [15]])
    # 验证预测的第一个输出的第二个输出的概率分布是否与期望的数组相等
    assert_array_equal(proba[0][1, :], [1, 0, 0, 0])
    # 验证预测的第二个输出的第二个输出的概率分布是否与期望的数组相等
    assert_array_equal(proba[1][1, :], [0, 1, 0, 0])
    # 使用分类器进行预测，验证预测的第二个输出的类别是否与期望的数组相等
    pred = clf.predict([[7], [15]])
    assert_array_equal(pred[1, :], [0, 1])

    # 测试异常标签列表长度不一致的情况
    def check_exception():
        # 创建一个 RNC 分类器，设置半径为 1，异常标签为 [0, 1, 2]
        clf = RNC(radius=1, outlier_label=[0, 1, 2])
        # 使用输入数据 X, y_multi 进行拟合
        clf.fit(X, y_multi)

    # 预期代码块会产生 ValueError 异常
    with pytest.raises(ValueError):
        check_exception()
def test_radius_neighbors_classifier_zero_distance():
    # Test radius-based classifier, when distance to a sample is zero.

    X = np.array([[1.0, 1.0], [2.0, 2.0]])
    y = np.array([1, 2])
    radius = 0.1

    z1 = np.array([[1.01, 1.01], [2.0, 2.0]])
    correct_labels1 = np.array([1, 2])

    weight_func = _weight_func

    for algorithm in ALGORITHMS:
        for weights in ["uniform", "distance", weight_func]:
            clf = neighbors.RadiusNeighborsClassifier(
                radius=radius, weights=weights, algorithm=algorithm
            )
            clf.fit(X, y)
            with np.errstate(invalid="ignore"):
                # Ignore the warning raised in _weight_func when making
                # predictions with null distances resulting in np.inf values.
                assert_array_equal(correct_labels1, clf.predict(z1))


def test_neighbors_regressors_zero_distance():
    # Test radius-based regressor, when distance to a sample is zero.

    X = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.5, 2.5]])
    y = np.array([1.0, 1.5, 2.0, 0.0])
    radius = 0.2
    z = np.array([[1.1, 1.1], [2.0, 2.0]])

    rnn_correct_labels = np.array([1.25, 2.0])

    knn_correct_unif = np.array([1.25, 1.0])
    knn_correct_dist = np.array([1.25, 2.0])

    for algorithm in ALGORITHMS:
        # we don't test for weights=_weight_func since user will be expected
        # to handle zero distances themselves in the function.
        for weights in ["uniform", "distance"]:
            rnn = neighbors.RadiusNeighborsRegressor(
                radius=radius, weights=weights, algorithm=algorithm
            )
            rnn.fit(X, y)
            assert_allclose(rnn_correct_labels, rnn.predict(z))

        for weights, corr_labels in zip(
            ["uniform", "distance"], [knn_correct_unif, knn_correct_dist]
        ):
            knn = neighbors.KNeighborsRegressor(
                n_neighbors=2, weights=weights, algorithm=algorithm
            )
            knn.fit(X, y)
            assert_allclose(corr_labels, knn.predict(z))


def test_radius_neighbors_boundary_handling():
    """Test whether points lying on boundary are handled consistently

    Also ensures that even with only one query point, an object array
    is returned rather than a 2d array.
    """

    X = np.array([[1.5], [3.0], [3.01]])
    radius = 3.0

    for algorithm in ALGORITHMS:
        nbrs = neighbors.NearestNeighbors(radius=radius, algorithm=algorithm).fit(X)
        results = nbrs.radius_neighbors([[0.0]], return_distance=False)
        assert results.shape == (1,)
        assert results.dtype == object
        assert_array_equal(results[0], [0, 1])


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_radius_neighbors_returns_array_of_objects(csr_container):
    # check that we can pass precomputed distances to
    # NearestNeighbors.radius_neighbors()
    # non-regression test for
    # https://github.com/scikit-learn/scikit-learn/issues/16036
    # 创建一个稀疏矩阵对象 `X`，包含4x4的全1矩阵
    X = csr_container(np.ones((4, 4)))
    # 将矩阵 `X` 的对角线元素设置为0
    X.setdiag([0, 0, 0, 0])
    
    # 使用最近邻算法初始化一个 `NearestNeighbors` 对象 `nbrs`
    # 设置搜索半径为0.5，算法自动选择，叶子大小为30，使用预先计算的度量方式
    nbrs = neighbors.NearestNeighbors(
        radius=0.5, algorithm="auto", leaf_size=30, metric="precomputed"
    ).fit(X)
    # 计算 `X` 中每个样本点的半径邻居
    neigh_dist, neigh_ind = nbrs.radius_neighbors(X, return_distance=True)
    
    # 创建一个数组 `expected_dist`，包含与 `X` 形状相同的对象数组
    # 每个元素都是长度为1的数组，初始值为 [np.array([0]), np.array([0]), np.array([0]), np.array([0])]
    expected_dist = np.empty(X.shape[0], dtype=object)
    expected_dist[:] = [np.array([0]), np.array([0]), np.array([0]), np.array([0])]
    # 创建一个数组 `expected_ind`，与 `expected_dist` 类似，但每个元素是索引数组
    expected_ind = np.empty(X.shape[0], dtype=object)
    expected_ind[:] = [np.array([0]), np.array([1]), np.array([2]), np.array([3])]
    
    # 使用断言验证 `neigh_dist` 和 `expected_dist` 数组是否相等
    assert_array_equal(neigh_dist, expected_dist)
    # 使用断言验证 `neigh_ind` 和 `expected_ind` 数组是否相等
    assert_array_equal(neigh_ind, expected_ind)
# 使用 pytest 的装饰器标记测试参数化，参数为算法选择
@pytest.mark.parametrize("algorithm", ["ball_tree", "kd_tree", "brute"])
def test_query_equidistant_kth_nn(algorithm):
    # 对于多个候选的第 k 个最近邻位置，应选择第一个候选
    query_point = np.array([[0, 0]])
    equidistant_points = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    # 第 3 和第 4 点不应替换第 2 点作为第 2 个最近邻位置
    k = 2
    knn_indices = np.array([[0, 1]])
    # 创建最近邻对象，并对等距点进行拟合
    nn = neighbors.NearestNeighbors(algorithm=algorithm).fit(equidistant_points)
    # 检索查询点的 k 个最近邻的索引
    indices = np.sort(nn.kneighbors(query_point, n_neighbors=k, return_distance=False))
    # 断言索引是否与预期的 k 个最近邻索引相等
    assert_array_equal(indices, knn_indices)


@pytest.mark.parametrize(
    ["algorithm", "metric"],
    list(
        product(
            ("kd_tree", "ball_tree", "brute"),
            ("euclidean", *DISTANCE_METRIC_OBJS),
        )
    )
    + [
        ("brute", "euclidean"),
        ("brute", "precomputed"),
    ],
)
def test_radius_neighbors_sort_results(algorithm, metric):
    # 测试 sort_results=True 时的 radius_neighbors[_graph] 输出

    # 解析度量标准
    metric = _parse_metric(metric, np.float64)
    if isinstance(metric, DistanceMetric):
        # 跳过不支持的距离度量类型
        pytest.skip(
            "Metrics of type `DistanceMetric` are not yet supported for radius-neighbor"
            " estimators."
        )
    n_samples = 10
    rng = np.random.RandomState(42)
    X = rng.random_sample((n_samples, 4))

    if metric == "precomputed":
        # 如果度量为预先计算的，生成 radius_neighbors_graph
        X = neighbors.radius_neighbors_graph(X, radius=np.inf, mode="distance")
    # 创建最近邻对象并拟合数据
    model = neighbors.NearestNeighbors(algorithm=algorithm, metric=metric)
    model.fit(X)

    # 获取半径范围内的邻居点及其距离，并排序结果
    distances, indices = model.radius_neighbors(X=X, radius=np.inf, sort_results=True)
    for ii in range(n_samples):
        # 断言每个样本的距离按升序排序
        assert_array_equal(distances[ii], np.sort(distances[ii]))

    # 测试 sort_results=True 且 return_distance=False
    if metric != "precomputed":  # 对于预先计算的图形不需要引发异常
        # 应引发 ValueError，提示 return_distance 必须为 True
        with pytest.raises(ValueError, match="return_distance must be True"):
            model.radius_neighbors(
                X=X, radius=np.inf, sort_results=True, return_distance=False
            )

    # 获取半径范围内的邻居图形，以距离为模式，并确保按数据排序
    graph = model.radius_neighbors_graph(
        X=X, radius=np.inf, mode="distance", sort_results=True
    )
    # 断言图形是否按数据排序
    assert _is_sorted_by_data(graph)


def test_RadiusNeighborsClassifier_multioutput():
    # 在多输出数据上测试 k-NN 分类器
    rng = check_random_state(0)
    n_features = 2
    n_samples = 40
    n_output = 3

    X = rng.rand(n_samples, n_features)
    y = rng.randint(0, 3, (n_samples, n_output))

    # 将数据拆分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # 测试多种权重参数
    weights = [None, "uniform", "distance", _weight_func]
    # 对于每一种算法和权重组合，进行迭代计算
    for algorithm, weights in product(ALGORITHMS, weights):
        # 存储单一输出的预测结果
        y_pred_so = []
        # 遍历每一个输出维度
        for o in range(n_output):
            # 使用 RadiusNeighborsClassifier 初始化 RNN 模型
            rnn = neighbors.RadiusNeighborsClassifier(
                weights=weights, algorithm=algorithm
            )
            # 使用训练数据拟合 RNN 模型，针对当前输出维度 o
            rnn.fit(X_train, y_train[:, o])
            # 将预测结果添加到单一输出预测列表中
            y_pred_so.append(rnn.predict(X_test))

        # 将单一输出预测结果转换为 numpy 数组，并进行转置
        y_pred_so = np.vstack(y_pred_so).T
        # 断言单一输出预测结果的形状与测试数据 y_test 相同
        assert y_pred_so.shape == y_test.shape

        # 多输出预测
        # 使用 RadiusNeighborsClassifier 初始化多输出 RNN 模型
        rnn_mo = neighbors.RadiusNeighborsClassifier(
            weights=weights, algorithm=algorithm
        )
        # 使用训练数据拟合多输出 RNN 模型
        rnn_mo.fit(X_train, y_train)
        # 对测试数据进行预测
        y_pred_mo = rnn_mo.predict(X_test)

        # 断言多输出预测结果的形状与测试数据 y_test 相同
        assert y_pred_mo.shape == y_test.shape
        # 断言多输出预测结果与单一输出预测结果 y_pred_so 相等
        assert_array_equal(y_pred_mo, y_pred_so)
# 测试稀疏矩阵上的 k-NN 分类器
def test_kneighbors_classifier_sparse(
    n_samples=40, n_features=5, n_test_pts=10, n_neighbors=5, random_state=0
):
    # 使用稀疏矩阵测试 k-NN 分类器
    rng = np.random.RandomState(random_state)
    # 生成随机数并创建稀疏矩阵 X
    X = 2 * rng.rand(n_samples, n_features) - 1
    X *= X > 0.2
    # 根据条件重新赋值 y
    y = ((X**2).sum(axis=1) < 0.5).astype(int)

    # 遍历不同类型的稀疏矩阵
    for sparsemat in SPARSE_TYPES:
        # 创建 k-NN 分类器对象
        knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm="auto")
        # 使用稀疏矩阵 sparsemat(X) 训练 knn 分类器
        knn.fit(sparsemat(X), y)
        # 根据随机生成的 epsilon，扰动部分样本点
        epsilon = 1e-5 * (2 * rng.rand(1, n_features) - 1)
        # 遍历稀疏向量和 np.asarray
        for sparsev in SPARSE_TYPES + (np.asarray,):
            # 生成扰动后的稀疏向量 X_eps
            X_eps = sparsev(X[:n_test_pts] + epsilon)
            # 预测 X_eps 的类别
            y_pred = knn.predict(X_eps)
            # 断言预测结果与真实结果一致
            assert_array_equal(y_pred, y[:n_test_pts])


# 测试 KNeighborsClassifier 多输出情况
def test_KNeighborsClassifier_multioutput():
    # 使用多输出数据测试 KNeighborsClassifier
    rng = check_random_state(0)
    n_features = 5
    n_samples = 50
    n_output = 3

    # 创建随机数生成的特征矩阵 X 和输出矩阵 y
    X = rng.rand(n_samples, n_features)
    y = rng.randint(0, 3, (n_samples, n_output))

    # 将数据集分割为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    weights = [None, "uniform", "distance", _weight_func]

    # 遍历不同的算法和权重配置
    for algorithm, weights in product(ALGORITHMS, weights):
        # 按单输出预测结果堆叠
        y_pred_so = []
        y_pred_proba_so = []
        for o in range(n_output):
            # 创建 KNeighborsClassifier 对象，针对单个输出 y_train[:, o] 进行训练
            knn = neighbors.KNeighborsClassifier(weights=weights, algorithm=algorithm)
            knn.fit(X_train, y_train[:, o])
            y_pred_so.append(knn.predict(X_test))
            y_pred_proba_so.append(knn.predict_proba(X_test))

        # 将单输出预测结果堆叠成矩阵
        y_pred_so = np.vstack(y_pred_so).T
        # 断言堆叠后的预测结果形状与测试集标签形状一致
        assert y_pred_so.shape == y_test.shape
        assert len(y_pred_proba_so) == n_output

        # 多输出预测
        knn_mo = neighbors.KNeighborsClassifier(weights=weights, algorithm=algorithm)
        knn_mo.fit(X_train, y_train)
        y_pred_mo = knn_mo.predict(X_test)

        # 断言多输出预测结果形状与测试集标签形状一致，并且与单输出预测结果一致
        assert y_pred_mo.shape == y_test.shape
        assert_array_equal(y_pred_mo, y_pred_so)

        # 检查概率预测结果
        y_pred_proba_mo = knn_mo.predict_proba(X_test)
        assert len(y_pred_proba_mo) == n_output

        # 断言多输出概率预测结果与单输出概率预测结果一致
        for proba_mo, proba_so in zip(y_pred_proba_mo, y_pred_proba_so):
            assert_array_equal(proba_mo, proba_so)


# 测试 k-NN 回归器
def test_kneighbors_regressor(
    n_samples=40, n_features=5, n_test_pts=10, n_neighbors=3, random_state=0
):
    # 测试 k-NN 回归器
    rng = np.random.RandomState(random_state)
    # 生成随机数并创建特征矩阵 X
    X = 2 * rng.rand(n_samples, n_features) - 1
    # 根据特征矩阵 X 计算目标值 y
    y = np.sqrt((X**2).sum(1))
    y /= y.max()

    # 获取目标值 y 的部分测试样本
    y_target = y[:n_test_pts]

    # 设置回归权重函数
    weight_func = _weight_func
    # 遍历每种算法配置
    for algorithm in ALGORITHMS:
        # 遍历权重选项，包括均匀权重、距离权重和自定义权重函数
        for weights in ["uniform", "distance", weight_func]:
            # 创建 K 最近邻回归器对象，配置邻居数量、权重类型和算法
            knn = neighbors.KNeighborsRegressor(
                n_neighbors=n_neighbors, weights=weights, algorithm=algorithm
            )
            # 使用训练数据 X 和目标数据 y 来拟合 K 最近邻模型
            knn.fit(X, y)
            # 添加小的扰动 epsilon 到部分测试数据 X 中，以验证模型的稳定性
            epsilon = 1e-5 * (2 * rng.rand(1, n_features) - 1)
            # 使用带扰动的测试数据进行预测
            y_pred = knn.predict(X[:n_test_pts] + epsilon)
            # 断言预测值 y_pred 与目标值 y_target 的差异在0.3以内
            assert np.all(abs(y_pred - y_target) < 0.3)
def test_KNeighborsRegressor_multioutput_uniform_weight():
    # Test k-neighbors in multi-output regression with uniform weight

    # 使用固定种子值生成随机数生成器
    rng = check_random_state(0)
    # 设定特征数为5，样本数为40，输出数为4
    n_features = 5
    n_samples = 40
    n_output = 4

    # 生成随机样本数据和输出数据
    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples, n_output)

    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # 使用product函数生成不同的算法和权重组合
    for algorithm, weights in product(ALGORITHMS, [None, "uniform"]):
        # 创建K近邻回归模型对象
        knn = neighbors.KNeighborsRegressor(weights=weights, algorithm=algorithm)
        # 在训练集上训练模型
        knn.fit(X_train, y_train)

        # 获取测试集样本的最近邻的索引
        neigh_idx = knn.kneighbors(X_test, return_distance=False)
        # 根据最近邻索引计算预测值
        y_pred_idx = np.array([np.mean(y_train[idx], axis=0) for idx in neigh_idx])

        # 进行预测
        y_pred = knn.predict(X_test)

        # 断言预测结果的形状与真实输出的形状相同
        assert y_pred.shape == y_test.shape
        assert y_pred_idx.shape == y_test.shape
        # 断言预测结果与索引预测结果接近
        assert_allclose(y_pred, y_pred_idx)


def test_kneighbors_regressor_multioutput(
    n_samples=40, n_features=5, n_test_pts=10, n_neighbors=3, random_state=0
):
    # Test k-neighbors in multi-output regression

    # 使用固定种子值生成随机数生成器
    rng = np.random.RandomState(random_state)
    # 生成符合要求的随机样本数据
    X = 2 * rng.rand(n_samples, n_features) - 1
    # 根据样本数据生成输出数据
    y = np.sqrt((X**2).sum(1))
    y /= y.max()
    y = np.vstack([y, y]).T

    # 设置目标输出数据为部分样本的输出
    y_target = y[:n_test_pts]

    # 定义不同的权重类型
    weights = ["uniform", "distance", _weight_func]
    # 使用product函数生成不同的算法和权重组合
    for algorithm, weights in product(ALGORITHMS, weights):
        # 创建K近邻回归模型对象
        knn = neighbors.KNeighborsRegressor(
            n_neighbors=n_neighbors, weights=weights, algorithm=algorithm
        )
        # 在整体样本数据上训练模型
        knn.fit(X, y)
        # 在部分样本数据上进行预测
        epsilon = 1e-5 * (2 * rng.rand(1, n_features) - 1)
        y_pred = knn.predict(X[:n_test_pts] + epsilon)
        # 断言预测结果的形状与目标输出数据的形状相同
        assert y_pred.shape == y_target.shape

        # 断言预测结果与目标输出数据的差值绝对值小于0.3
        assert np.all(np.abs(y_pred - y_target) < 0.3)


def test_radius_neighbors_regressor(
    n_samples=40, n_features=3, n_test_pts=10, radius=0.5, random_state=0
):
    # Test radius-based neighbors regression

    # 使用固定种子值生成随机数生成器
    rng = np.random.RandomState(random_state)
    # 生成符合要求的随机样本数据
    X = 2 * rng.rand(n_samples, n_features) - 1
    # 根据样本数据生成输出数据
    y = np.sqrt((X**2).sum(1))
    y /= y.max()

    # 设置目标输出数据为部分样本的输出
    y_target = y[:n_test_pts]

    # 定义权重函数
    weight_func = _weight_func

    # 使用不同的算法进行迭代
    for algorithm in ALGORITHMS:
        # 使用不同的权重类型进行迭代
        for weights in ["uniform", "distance", weight_func]:
            # 创建基于半径的K近邻回归模型对象
            neigh = neighbors.RadiusNeighborsRegressor(
                radius=radius, weights=weights, algorithm=algorithm
            )
            # 在整体样本数据上训练模型
            neigh.fit(X, y)
            # 在部分样本数据上进行预测
            epsilon = 1e-5 * (2 * rng.rand(1, n_features) - 1)
            y_pred = neigh.predict(X[:n_test_pts] + epsilon)
            # 断言预测结果与目标输出数据的差值绝对值小于半径的一半
            assert np.all(abs(y_pred - y_target) < radius / 2)

    # test that nan is returned when no nearby observations
    # 对于每种权重计算方式进行迭代：uniform（均匀）和 distance（距离加权）
    for weights in ["uniform", "distance"]:
        # 创建一个半径邻居回归器对象，设定半径、权重方式和算法（自动选择）
        neigh = neighbors.RadiusNeighborsRegressor(
            radius=radius, weights=weights, algorithm="auto"
        )
        # 使用给定的特征向量 X 和目标向量 y 来拟合回归器模型
        neigh.fit(X, y)
        # 创建一个测试用的特征向量 X_test_nan，所有值初始化为 -1.0
        X_test_nan = np.full((1, n_features), -1.0)
        # 准备一个空警告信息字符串，用于在预测时捕获警告
        empty_warning_msg = (
            "One or more samples have no neighbors "
            "within specified radius; predicting NaN."
        )
        # 在执行预测时，捕获可能的用户警告，匹配预定义的空警告信息
        with pytest.warns(UserWarning, match=re.escape(empty_warning_msg)):
            # 使用邻居回归器预测给定的测试数据 X_test_nan
            pred = neigh.predict(X_test_nan)
        # 断言预测结果中的所有值都是 NaN（Not a Number）
        assert np.all(np.isnan(pred))
# 定义测试函数，测试 RadiusNeighborsRegressor 在多输出回归中使用均匀权重的情况
def test_RadiusNeighborsRegressor_multioutput_with_uniform_weight():
    # 随机数生成器，种子为0
    rng = check_random_state(0)
    # 特征数为5，样本数为40，输出为4个
    n_features = 5
    n_samples = 40
    n_output = 4

    # 生成随机的样本特征和输出值
    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples, n_output)
    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # 对算法和权重进行排列组合，使用 product 函数
    for algorithm, weights in product(ALGORITHMS, [None, "uniform"]):
        # 创建 RadiusNeighborsRegressor 对象
        rnn = neighbors.RadiusNeighborsRegressor(weights=weights, algorithm=algorithm)
        # 使用训练数据拟合模型
        rnn.fit(X_train, y_train)

        # 获取测试集的邻居索引
        neigh_idx = rnn.radius_neighbors(X_test, return_distance=False)
        # 计算每个测试点对应的预测输出值
        y_pred_idx = np.array([np.mean(y_train[idx], axis=0) for idx in neigh_idx])

        # 将预测值转换为数组
        y_pred_idx = np.array(y_pred_idx)
        # 使用模型预测测试集的输出
        y_pred = rnn.predict(X_test)

        # 断言预测值的形状与真实输出值的形状相同
        assert y_pred_idx.shape == y_test.shape
        assert y_pred.shape == y_test.shape
        # 使用 assert_allclose 断言预测值与真实输出值之间的接近程度
        assert_allclose(y_pred, y_pred_idx)


# 定义测试函数，测试 RadiusNeighborsRegressor 在多输出回归中的情况
def test_RadiusNeighborsRegressor_multioutput(
    n_samples=40, n_features=5, n_test_pts=10, random_state=0
):
    # 测试使用不同权重进行多输出回归
    rng = np.random.RandomState(random_state)
    # 生成随机的样本特征数据
    X = 2 * rng.rand(n_samples, n_features) - 1
    # 根据特征数据计算对应的输出值
    y = np.sqrt((X**2).sum(1))
    y /= y.max()
    # 将输出值重复为两列，模拟多输出情况
    y = np.vstack([y, y]).T

    # 目标输出为前 n_test_pts 个样本的输出值
    y_target = y[:n_test_pts]
    # 使用不同的算法和权重类型进行排列组合
    weights = ["uniform", "distance", _weight_func]

    for algorithm, weights in product(ALGORITHMS, weights):
        # 创建 RadiusNeighborsRegressor 对象
        rnn = neighbors.RadiusNeighborsRegressor(weights=weights, algorithm=algorithm)
        # 使用全部样本数据拟合模型
        rnn.fit(X, y)
        # 生成一个微小的扰动 epsilon，用于测试集输入数据
        epsilon = 1e-5 * (2 * rng.rand(1, n_features) - 1)
        # 使用模型预测测试集的输出值
        y_pred = rnn.predict(X[:n_test_pts] + epsilon)

        # 断言预测值的形状与目标输出的形状相同
        assert y_pred.shape == y_target.shape
        # 断言预测值与目标输出之间的差距小于0.3
        assert np.all(np.abs(y_pred - y_target) < 0.3)


# 使用 pytest 的标记，忽略警告信息
@pytest.mark.filterwarnings("ignore:EfficiencyWarning")
def test_kneighbors_regressor_sparse(
    n_samples=40, n_features=5, n_test_pts=10, n_neighbors=5, random_state=0
):
    # 测试稀疏矩阵上的基于半径的回归
    # 类似上面的测试，但使用各种类型的稀疏矩阵
    rng = np.random.RandomState(random_state)
    # 生成随机的样本特征数据
    X = 2 * rng.rand(n_samples, n_features) - 1
    # 根据特征数据生成二元分类的输出值
    y = ((X**2).sum(axis=1) < 0.25).astype(int)
    # 遍历稀疏矩阵类型列表
    for sparsemat in SPARSE_TYPES:
        # 创建 K 最近邻回归器对象，使用自动选择算法
        knn = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors, algorithm="auto")
        # 使用稀疏矩阵 sparsemat 对象拟合模型
        knn.fit(sparsemat(X), y)

        # 创建 K 最近邻回归器对象，使用预先计算的距离矩阵作为度量
        knn_pre = neighbors.KNeighborsRegressor(
            n_neighbors=n_neighbors, metric="precomputed"
        )
        # 使用预先计算的欧氏距离矩阵拟合模型
        knn_pre.fit(pairwise_distances(X, metric="euclidean"), y)

        # 遍历稀疏或密集矩阵类型列表
        for sparsev in SPARSE_OR_DENSE:
            # 对输入数据 X 应用稀疏或密集矩阵转换 sparsev
            X2 = sparsev(X)
            # 断言预测结果的平均值大于 0.95
            assert np.mean(knn.predict(X2).round() == y) > 0.95

            # 对预先计算的欧氏距离矩阵应用稀疏或密集矩阵转换 sparsev
            X2_pre = sparsev(pairwise_distances(X, metric="euclidean"))
            # 如果 sparsev 属于 DOK_CONTAINERS 或 BSR_CONTAINERS，则断言引发 TypeError 异常，匹配特定消息
            if sparsev in DOK_CONTAINERS + BSR_CONTAINERS:
                msg = "not supported due to its handling of explicit zeros"
                with pytest.raises(TypeError, match=msg):
                    knn_pre.predict(X2_pre)
            else:
                # 断言预测结果的平均值大于 0.95
                assert np.mean(knn_pre.predict(X2_pre).round() == y) > 0.95
def test_neighbors_iris():
    # Iris 数据集的基本检查
    # 将每个标签的三个点放置在平面上，并在接近决策边界的点上执行最近邻查询。

    for algorithm in ALGORITHMS:
        # 使用指定算法创建最近邻分类器对象
        clf = neighbors.KNeighborsClassifier(n_neighbors=1, algorithm=algorithm)
        # 使用 iris 数据集训练分类器
        clf.fit(iris.data, iris.target)
        # 断言预测结果与目标值数组相等
        assert_array_equal(clf.predict(iris.data), iris.target)

        # 更新分类器参数，并重新训练
        clf.set_params(n_neighbors=9, algorithm=algorithm)
        clf.fit(iris.data, iris.target)
        # 断言分类器准确率大于 95%
        assert np.mean(clf.predict(iris.data) == iris.target) > 0.95

        # 使用指定算法创建最近邻回归器对象
        rgs = neighbors.KNeighborsRegressor(n_neighbors=5, algorithm=algorithm)
        # 使用 iris 数据集训练回归器
        rgs.fit(iris.data, iris.target)
        # 断言回归器准确率大于 95%
        assert np.mean(rgs.predict(iris.data).round() == iris.target) > 0.95


def test_neighbors_digits():
    # 手写数字数据集的基本检查
    # 当输入数据类型为 uint8 时，“brute”算法由于距离计算中的溢出而可能失败。

    X = digits.data.astype("uint8")
    Y = digits.target
    (n_samples, n_features) = X.shape
    train_test_boundary = int(n_samples * 0.8)
    train = np.arange(0, train_test_boundary)
    test = np.arange(train_test_boundary, n_samples)
    (X_train, Y_train, X_test, Y_test) = X[train], Y[train], X[test], Y[test]

    # 使用“brute”算法创建最近邻分类器对象
    clf = neighbors.KNeighborsClassifier(n_neighbors=1, algorithm="brute")
    # 分别计算 uint8 和 float 类型输入的分类器得分，并断言它们相等
    score_uint8 = clf.fit(X_train, Y_train).score(X_test, Y_test)
    score_float = clf.fit(X_train.astype(float, copy=False), Y_train).score(
        X_test.astype(float, copy=False), Y_test
    )
    assert score_uint8 == score_float


def test_kneighbors_graph():
    # 测试 kneighbors_graph 函数构建 k 近邻图。
    X = np.array([[0, 1], [1.01, 1.0], [2, 0]])

    # n_neighbors = 1
    # 使用 mode="connectivity" 参数构建 k=1 的邻接矩阵，并断言与单位矩阵相等
    A = neighbors.kneighbors_graph(X, 1, mode="connectivity", include_self=True)
    assert_array_equal(A.toarray(), np.eye(A.shape[0]))

    # 使用 mode="distance" 参数构建 k=1 的距离矩阵，并断言数值近似相等
    A = neighbors.kneighbors_graph(X, 1, mode="distance")
    assert_allclose(
        A.toarray(), [[0.00, 1.01, 0.0], [1.01, 0.0, 0.0], [0.00, 1.40716026, 0.0]]
    )

    # n_neighbors = 2
    # 使用 mode="connectivity" 参数构建 k=2 的邻接矩阵，并断言与预期矩阵相等
    A = neighbors.kneighbors_graph(X, 2, mode="connectivity", include_self=True)
    assert_array_equal(A.toarray(), [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])

    # 使用 mode="distance" 参数构建 k=2 的距离矩阵，并断言数值近似相等
    A = neighbors.kneighbors_graph(X, 2, mode="distance")
    assert_allclose(
        A.toarray(),
        [
            [0.0, 1.01, 2.23606798],
            [1.01, 0.0, 1.40716026],
            [2.23606798, 1.40716026, 0.0],
        ],
    )

    # n_neighbors = 3
    # 使用 mode="connectivity" 参数构建 k=3 的邻接矩阵，并断言数值近似相等
    A = neighbors.kneighbors_graph(X, 3, mode="connectivity", include_self=True)
    assert_allclose(A.toarray(), [[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    # 使用 kneighbors_graph 函数构建稀疏输入的 k-最近邻图。
    # 初始化一个随机数生成器 rng，使用指定的种子 seed。
    rng = np.random.RandomState(seed)
    
    # 生成一个 10x10 的随机数组 X。
    X = rng.randn(10, 10)
    
    # 将数组 X 转换为稀疏格式，存储在 Xcsr 变量中。
    Xcsr = csr_container(X)
    
    # 断言两个 k-最近邻图的稀疏矩阵表示在给定的模式（mode）下是相等的。
    assert_allclose(
        neighbors.kneighbors_graph(X, n_neighbors, mode=mode).toarray(),
        neighbors.kneighbors_graph(Xcsr, n_neighbors, mode=mode).toarray(),
    )
def test_radius_neighbors_graph():
    # 测试 radius_neighbors_graph 函数以构建最近邻图。
    X = np.array([[0, 1], [1.01, 1.0], [2, 0]])

    # 创建连接性模式下的最近邻图，包括自身节点
    A = neighbors.radius_neighbors_graph(X, 1.5, mode="connectivity", include_self=True)
    assert_array_equal(A.toarray(), [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])

    # 创建距离模式下的最近邻图
    A = neighbors.radius_neighbors_graph(X, 1.5, mode="distance")
    assert_allclose(
        A.toarray(), [[0.0, 1.01, 0.0], [1.01, 0.0, 1.40716026], [0.0, 1.40716026, 0.0]]
    )


@pytest.mark.parametrize("n_neighbors", [1, 2, 3])
@pytest.mark.parametrize("mode", ["connectivity", "distance"])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_radius_neighbors_graph_sparse(n_neighbors, mode, csr_container, seed=36):
    # 测试 radius_neighbors_graph 函数对稀疏输入构建最近邻图。
    rng = np.random.RandomState(seed)
    X = rng.randn(10, 10)
    Xcsr = csr_container(X)

    assert_allclose(
        # 比较稠密和稀疏输入情况下的最近邻图
        neighbors.radius_neighbors_graph(X, n_neighbors, mode=mode).toarray(),
        neighbors.radius_neighbors_graph(Xcsr, n_neighbors, mode=mode).toarray(),
    )


@pytest.mark.parametrize(
    "Estimator",
    [
        neighbors.KNeighborsClassifier,
        neighbors.RadiusNeighborsClassifier,
        neighbors.KNeighborsRegressor,
        neighbors.RadiusNeighborsRegressor,
    ],
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_neighbors_validate_parameters(Estimator, csr_container):
    """*Neighbors* 估算器的额外参数验证，未包含在通用验证中。"""
    X = rng.random_sample((10, 2))
    Xsparse = csr_container(X)
    X3 = rng.random_sample((10, 3))
    y = np.ones(10)

    nbrs = Estimator(algorithm="ball_tree", metric="haversine")
    msg = "instance is not fitted yet"
    with pytest.raises(ValueError, match=msg):
        nbrs.predict(X)
    msg = "Metric 'haversine' not valid for sparse input."
    with pytest.raises(ValueError, match=msg):
        ignore_warnings(nbrs.fit(Xsparse, y))

    nbrs = Estimator(metric="haversine", algorithm="brute")
    nbrs.fit(X3, y)
    msg = "Haversine distance only valid in 2 dimensions"
    with pytest.raises(ValueError, match=msg):
        nbrs.predict(X3)

    nbrs = Estimator()
    msg = re.escape("Found array with 0 sample(s)")
    with pytest.raises(ValueError, match=msg):
        nbrs.fit(np.ones((0, 2)), np.ones(0))

    msg = "Found array with dim 3"
    with pytest.raises(ValueError, match=msg):
        nbrs.fit(X[:, :, None], y)
    nbrs.fit(X, y)

    msg = re.escape("Found array with 0 feature(s)")
    with pytest.raises(ValueError, match=msg):
        nbrs.predict([[]])


@pytest.mark.parametrize(
    "Estimator",
    [
        neighbors.KNeighborsClassifier,
        neighbors.RadiusNeighborsClassifier,
        neighbors.KNeighborsRegressor,
        neighbors.RadiusNeighborsRegressor,
    ],
)
@pytest.mark.parametrize("n_features", [2, 100])
# 使用 pytest.mark.parametrize 装饰器，定义测试函数 test_neighbors_minkowski_semimetric_algo_warn，参数化测试算法 "auto" 和 "brute"
@pytest.mark.parametrize("algorithm", ["auto", "brute"])
def test_neighbors_minkowski_semimetric_algo_warn(Estimator, n_features, algorithm):
    """
    Validation of all classes extending NeighborsBase with
    Minkowski semi-metrics (i.e. when 0 < p < 1). That proper
    Warning is raised for `algorithm="auto"` and "brute".
    """
    # 创建随机数数组作为特征数据 X 和标签数据 y
    X = rng.random_sample((10, n_features))
    y = np.ones(10)

    # 根据给定的 Estimator 类和算法参数创建模型对象
    model = Estimator(p=0.1, algorithm=algorithm)
    # 定义警告消息
    msg = (
        "Mind that for 0 < p < 1, Minkowski metrics are not distance"
        " metrics. Continuing the execution with `algorithm='brute'`."
    )
    # 使用 pytest.warns 验证是否会触发 UserWarning，并匹配预期的警告消息
    with pytest.warns(UserWarning, match=msg):
        model.fit(X, y)

    # 断言模型的拟合方法为 "brute"
    assert model._fit_method == "brute"


# 使用 pytest.mark.parametrize 装饰器，定义测试函数 test_neighbors_minkowski_semimetric_algo_error，参数化 Estimator 类、特征数 n_features 和算法参数 "kd_tree" 和 "ball_tree"
@pytest.mark.parametrize(
    "Estimator",
    [
        neighbors.KNeighborsClassifier,
        neighbors.RadiusNeighborsClassifier,
        neighbors.KNeighborsRegressor,
        neighbors.RadiusNeighborsRegressor,
    ],
)
@pytest.mark.parametrize("n_features", [2, 100])
@pytest.mark.parametrize("algorithm", ["kd_tree", "ball_tree"])
def test_neighbors_minkowski_semimetric_algo_error(Estimator, n_features, algorithm):
    """Check that we raise a proper error if `algorithm!='brute'` and `p<1`."""
    # 创建随机数数组作为特征数据 X 和标签数据 y
    X = rng.random_sample((10, 2))
    y = np.ones(10)

    # 根据给定的 Estimator 类、算法参数和 p 值创建模型对象
    model = Estimator(algorithm=algorithm, p=0.1)
    # 定义错误消息
    msg = (
        f'algorithm="{algorithm}" does not support 0 < p < 1 for '
        "the Minkowski metric. To resolve this problem either "
        'set p >= 1 or algorithm="brute".'
    )
    # 使用 pytest.raises 验证是否会触发 ValueError，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        model.fit(X, y)


# TODO: remove when NearestNeighbors methods uses parameter validation mechanism
# 定义测试函数 test_nearest_neighbors_validate_params，验证 NearestNeighbors 的参数
def test_nearest_neighbors_validate_params():
    """Validate parameter of NearestNeighbors."""
    # 创建随机数数组作为特征数据 X
    X = rng.random_sample((10, 2))

    # 创建 NearestNeighbors 对象并拟合数据 X
    nbrs = neighbors.NearestNeighbors().fit(X)
    # 定义错误消息
    msg = (
        'Unsupported mode, must be one of "connectivity", or "distance" but got "blah"'
        " instead"
    )
    # 使用 pytest.raises 验证是否会触发 ValueError，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        nbrs.kneighbors_graph(X, mode="blah")
    with pytest.raises(ValueError, match=msg):
        nbrs.radius_neighbors_graph(X, mode="blah")


# 使用 pytest.mark.parametrize 装饰器，定义测试函数 test_neighbors_metrics，参数化 metric 参数
@pytest.mark.parametrize(
    "metric",
    sorted(
        set(neighbors.VALID_METRICS["ball_tree"]).intersection(
            neighbors.VALID_METRICS["brute"]
        )
        - set(["pyfunc", *BOOL_METRICS])
    )
    + DISTANCE_METRIC_OBJS,
)
def test_neighbors_metrics(
    global_dtype,
    global_random_seed,
    metric,
    n_samples=20,
    n_features=3,
    n_query_pts=2,
    n_neighbors=5,
):
    rng = np.random.RandomState(global_random_seed)

    # 解析度量标准 metric，并根据全局数据类型创建 X_train 和 X_test
    metric = _parse_metric(metric, global_dtype)

    # 测试使用不同度量标准计算邻居
    algorithms = ["brute", "ball_tree", "kd_tree"]
    X_train = rng.rand(n_samples, n_features).astype(global_dtype, copy=False)
    X_test = rng.rand(n_query_pts, n_features).astype(global_dtype, copy=False)
    # 生成针对指定距离度量(metric)和特征数(n_features)的测试参数列表
    metric_params_list = _generate_test_params_for(metric, n_features)

    # 遍历测试参数列表
    for metric_params in metric_params_list:
        # 检查是否应该排除使用 KDTree 的某些度量（如加权闵可夫斯基距离）
        exclude_kd_tree = (
            False
            if isinstance(metric, DistanceMetric)  # 如果度量(metric)是距离度量对象
            else metric not in neighbors.VALID_METRICS["kd_tree"]  # 或者度量不在 KDTree 支持的度量列表中
                 or ("minkowski" in metric and "w" in metric_params)  # 或者是闵可夫斯基距离且包含权重参数
        )
        results = {}

        # 从参数中弹出距离度量参数中的 "p" 值，默认为 2
        p = metric_params.pop("p", 2)

        # 遍历算法列表
        for algorithm in algorithms:
            # 如果度量是距离度量对象并且全局数据类型是 np.float32
            if isinstance(metric, DistanceMetric) and global_dtype == np.float32:
                # 如果算法包含 "tree"，则跳过此测试（不覆盖）
                if "tree" in algorithm:  # pragma: nocover
                    pytest.skip(
                        "Neither KDTree nor BallTree support 32-bit distance metric"
                        " objects."
                    )

            # 创建最近邻对象(neighbors.NearestNeighbors)
            neigh = neighbors.NearestNeighbors(
                n_neighbors=n_neighbors,
                algorithm=algorithm,
                metric=metric,
                p=p,
                metric_params=metric_params,
            )

            # 如果需要排除使用 KDTree 并且当前算法是 "kd_tree"
            if exclude_kd_tree and algorithm == "kd_tree":
                # 确保会引发 ValueError 异常
                with pytest.raises(ValueError):
                    neigh.fit(X_train)
                continue

            # 如果距离度量是 "haversine"，则只接受二维数据
            if metric == "haversine":
                feature_sl = slice(None, 2)
                # 将训练数据和测试数据限制为二维
                X_train = np.ascontiguousarray(X_train[:, feature_sl])
                X_test = np.ascontiguousarray(X_test[:, feature_sl])

            # 使用训练数据拟合最近邻模型
            neigh.fit(X_train)
            # 计算最近邻和对应的距离，存储在结果字典中
            results[algorithm] = neigh.kneighbors(X_test, return_distance=True)

        # 从结果中获取 brute 算法的距离和索引
        brute_dst, brute_idx = results["brute"]
        # 从结果中获取 ball_tree 算法的距离和索引
        ball_tree_dst, ball_tree_idx = results["ball_tree"]

        # 无论输入数据类型如何，返回的距离始终为 float64，根据输入数据类型调整容差
        rtol = 1e-7 if global_dtype == np.float64 else 1e-4

        # 检查 brute 和 ball_tree 算法返回的距离是否在容差范围内相等
        assert_allclose(brute_dst, ball_tree_dst, rtol=rtol)
        # 检查 brute 和 ball_tree 算法返回的索引是否完全相等
        assert_array_equal(brute_idx, ball_tree_idx)

        # 如果不需要排除使用 KDTree
        if not exclude_kd_tree:
            # 从结果中获取 kd_tree 算法的距离和索引
            kd_tree_dst, kd_tree_idx = results["kd_tree"]
            # 检查 brute 和 kd_tree 算法返回的距离是否在容差范围内相等
            assert_allclose(brute_dst, kd_tree_dst, rtol=rtol)
            # 检查 brute 和 kd_tree 算法返回的索引是否完全相等
            assert_array_equal(brute_idx, kd_tree_idx)

            # 检查 ball_tree 和 kd_tree 算法返回的距离是否在容差范围内相等
            assert_allclose(ball_tree_dst, kd_tree_dst, rtol=rtol)
            # 检查 ball_tree 和 kd_tree 算法返回的索引是否完全相等
            assert_array_equal(ball_tree_idx, kd_tree_idx)
# 使用 pytest 的参数化功能来定义测试用例，参数化使用的指标是 'brute' 算法的有效指标集合去除了 'precomputed' 后的结果
@pytest.mark.parametrize(
    "metric", sorted(set(neighbors.VALID_METRICS["brute"]) - set(["precomputed"]))
)
# 定义测试函数 test_kneighbors_brute_backend，测试 'brute' 算法的后端实现
def test_kneighbors_brute_backend(
    metric,
    global_dtype,
    global_random_seed,
    n_samples=2000,
    n_features=30,
    n_query_pts=5,
    n_neighbors=5,
):
    # 使用全局随机种子创建随机数生成器
    rng = np.random.RandomState(global_random_seed)
    
    # 生成随机的训练数据和测试数据，数据类型为 global_dtype
    X_train = rng.rand(n_samples, n_features).astype(global_dtype, copy=False)
    X_test = rng.rand(n_query_pts, n_features).astype(global_dtype, copy=False)

    # 对于 'haversine' 距离，只接受二维数据，所以进行切片操作
    if metric == "haversine":
        feature_sl = slice(None, 2)
        X_train = np.ascontiguousarray(X_train[:, feature_sl])
        X_test = np.ascontiguousarray(X_test[:, feature_sl])

    # 如果指标在 PAIRWISE_BOOLEAN_FUNCTIONS 中，将数据转换为布尔类型
    if metric in PAIRWISE_BOOLEAN_FUNCTIONS:
        X_train = X_train > 0.5
        X_test = X_test > 0.5

    # 生成测试用的 metric 参数列表
    metric_params_list = _generate_test_params_for(metric, n_features)

    # 遍历 metric 参数列表
    for metric_params in metric_params_list:
        # 从 metric_params 中弹出 'p' 参数，默认值为 2
        p = metric_params.pop("p", 2)

        # 创建 NearestNeighbors 实例 neigh，使用 'brute' 算法和给定的 metric 和 metric_params
        neigh = neighbors.NearestNeighbors(
            n_neighbors=n_neighbors,
            algorithm="brute",
            metric=metric,
            p=p,
            metric_params=metric_params,
        )

        # 使用训练数据 X_train 拟合模型
        neigh.fit(X_train)

        # 在禁用 Cython 加速的上下文中计算 legacy_brute_dst 和 legacy_brute_idx
        with config_context(enable_cython_pairwise_dist=False):
            legacy_brute_dst, legacy_brute_idx = neigh.kneighbors(
                X_test, return_distance=True
            )
        
        # 在启用 Cython 加速的上下文中计算 pdr_brute_dst 和 pdr_brute_idx
        with config_context(enable_cython_pairwise_dist=True):
            pdr_brute_dst, pdr_brute_idx = neigh.kneighbors(
                X_test, return_distance=True
            )

        # 断言 legacy_brute 和 pdr_brute 的计算结果应该兼容
        assert_compatible_argkmin_results(
            legacy_brute_dst, pdr_brute_dst, legacy_brute_idx, pdr_brute_idx
        )


# 定义测试函数 test_callable_metric，测试可调用的自定义 metric 函数
def test_callable_metric():
    # 定义自定义的 metric 函数 custom_metric，计算欧几里得距离的平方根
    def custom_metric(x1, x2):
        return np.sqrt(np.sum(x1**2 + x2**2))

    # 生成随机数据 X
    X = np.random.RandomState(42).rand(20, 2)
    
    # 创建 NearestNeighbors 实例 nbrs1 和 nbrs2，使用自定义的 metric 函数
    nbrs1 = neighbors.NearestNeighbors(
        n_neighbors=3, algorithm="auto", metric=custom_metric
    )
    nbrs2 = neighbors.NearestNeighbors(
        n_neighbors=3, algorithm="brute", metric=custom_metric
    )

    # 分别使用 X 拟合 nbrs1 和 nbrs2
    nbrs1.fit(X)
    nbrs2.fit(X)

    # 计算 nbrs1 和 nbrs2 对于 X 的最近邻距离和索引
    dist1, ind1 = nbrs1.kneighbors(X)
    dist2, ind2 = nbrs2.kneighbors(X)

    # 断言两种算法的计算结果应该非常接近
    assert_allclose(dist1, dist2)


# 使用 pytest 的参数化功能，参数化指标是 'brute' 算法的有效指标集合和 DISTANCE_METRIC_OBJS
@pytest.mark.parametrize(
    "metric", neighbors.VALID_METRICS["brute"] + DISTANCE_METRIC_OBJS
)
# 参数化 csr_container，用于测试不同的数据结构
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 定义测试函数 test_valid_brute_metric_for_auto_algorithm，测试 'auto' 算法中 'brute' 算法的有效指标
def test_valid_brute_metric_for_auto_algorithm(
    global_dtype, metric, csr_container, n_samples=20, n_features=12
):
    # 将 metric 转换为适合使用的格式，使用全局数据类型 global_dtype
    metric = _parse_metric(metric, global_dtype)

    # 生成随机数据 X，使用全局随机数生成器 rng
    X = rng.rand(n_samples, n_features).astype(global_dtype, copy=False)
    
    # 将 X 转换为 csr_container 格式
    Xcsr = csr_container(X)

    # 生成测试用的 metric 参数列表
    metric_params_list = _generate_test_params_for(metric, n_features)
    # 如果距离度量(metric)是 "precomputed"，执行以下代码块
    if metric == "precomputed":
        # 生成一个 10x4 的随机数组 X_precomputed
        X_precomputed = rng.random_sample((10, 4))
        # 生成一个 3x4 的随机数组 Y_precomputed
        Y_precomputed = rng.random_sample((3, 4))
        # 计算基于欧氏距离的 X_precomputed 的距离矩阵 DXX
        DXX = metrics.pairwise_distances(X_precomputed, metric="euclidean")
        # 计算基于欧氏距离的 Y_precomputed 和 X_precomputed 之间的距离矩阵 DYX
        DYX = metrics.pairwise_distances(
            Y_precomputed, X_precomputed, metric="euclidean"
        )
        # 创建一个 NearestNeighbors 对象 nb_p，使用预计算的距离矩阵 DXX
        nb_p = neighbors.NearestNeighbors(n_neighbors=3, metric="precomputed")
        nb_p.fit(DXX)
        # 找出 DYX 中每个样本的最近邻
        nb_p.kneighbors(DYX)
    else:
        # 对于 metric_params_list 中的每个 metric_params
        for metric_params in metric_params_list:
            # 创建一个 NearestNeighbors 对象 nn，设置参数 n_neighbors=3, algorithm="auto", metric=metric, metric_params=metric_params
            nn = neighbors.NearestNeighbors(
                n_neighbors=3,
                algorithm="auto",
                metric=metric,
                metric_params=metric_params,
            )
            # 如果距离度量是 "haversine"，只接受二维数据，将 X 的特征切片到前两列
            if metric == "haversine":
                feature_sl = slice(None, 2)
                X = np.ascontiguousarray(X[:, feature_sl])
            
            # 将数据 X 加载到 NearestNeighbors 对象 nn 中
            nn.fit(X)
            # 找出 X 中每个样本的最近邻
            nn.kneighbors(X)

            # 如果 metric 属于 VALID_METRICS_SPARSE["brute"] 中的度量标准
            if metric in VALID_METRICS_SPARSE["brute"]:
                # 创建一个 NearestNeighbors 对象 nn，使用稀疏矩阵 Xcsr
                nn = neighbors.NearestNeighbors(
                    n_neighbors=3, algorithm="auto", metric=metric
                ).fit(Xcsr)
                # 找出稀疏矩阵 Xcsr 中每个样本的最近邻
                nn.kneighbors(Xcsr)
def test_metric_params_interface():
    # 使用随机数生成器创建一个5x5的数组X
    X = rng.rand(5, 5)
    # 用随机整数填充一个长度为5的数组y
    y = rng.randint(0, 2, 5)
    # 创建一个K最近邻分类器，指定距离度量参数为 {"p": 3}
    est = neighbors.KNeighborsClassifier(metric_params={"p": 3})
    # 断言会产生SyntaxWarning警告
    with pytest.warns(SyntaxWarning):
        # 对数据X, y进行拟合
        est.fit(X, y)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_predict_sparse_ball_kd_tree(csr_container):
    rng = np.random.RandomState(0)
    # 用随机数生成器创建一个5x5的数组X
    X = rng.rand(5, 5)
    # 用随机整数填充一个长度为5的数组y
    y = rng.randint(0, 2, 5)
    # 创建一个K最近邻分类器，使用kd_tree算法
    nbrs1 = neighbors.KNeighborsClassifier(1, algorithm="kd_tree")
    # 创建一个K最近邻回归器，使用ball_tree算法
    nbrs2 = neighbors.KNeighborsRegressor(1, algorithm="ball_tree")
    # 遍历模型列表[nbrs1, nbrs2]
    for model in [nbrs1, nbrs2]:
        # 对数据X, y进行拟合
        model.fit(X, y)
        # 断言会产生ValueError错误
        with pytest.raises(ValueError):
            # 对csr_container类型的X进行预测
            model.predict(csr_container(X))


def test_non_euclidean_kneighbors():
    rng = np.random.RandomState(0)
    # 用随机数生成器创建一个5x5的数组X
    X = rng.rand(5, 5)

    # 找到一个合理的半径值
    dist_array = pairwise_distances(X).flatten()
    np.sort(dist_array)
    radius = dist_array[15]

    # 测试kneighbors_graph函数
    for metric in ["manhattan", "chebyshev"]:
        # 生成基于指定距离度量的最近邻图
        nbrs_graph = neighbors.kneighbors_graph(
            X, 3, metric=metric, mode="connectivity", include_self=True
        ).toarray()
        # 创建一个最近邻模型，使用指定距离度量
        nbrs1 = neighbors.NearestNeighbors(n_neighbors=3, metric=metric).fit(X)
        # 断言两个最近邻图是否相等
        assert_array_equal(nbrs_graph, nbrs1.kneighbors_graph(X).toarray())

    # 测试radiusneighbors_graph函数
    for metric in ["manhattan", "chebyshev"]:
        # 生成基于指定距离度量和半径的最近邻图
        nbrs_graph = neighbors.radius_neighbors_graph(
            X, radius, metric=metric, mode="connectivity", include_self=True
        ).toarray()
        # 创建一个最近邻模型，使用指定距离度量和半径
        nbrs1 = neighbors.NearestNeighbors(metric=metric, radius=radius).fit(X)
        # 断言两个最近邻图是否相等
        assert_array_equal(nbrs_graph, nbrs1.radius_neighbors_graph(X).toarray())

    # 当提供错误参数时抛出错误
    X_nbrs = neighbors.NearestNeighbors(n_neighbors=3, metric="manhattan")
    X_nbrs.fit(X)
    with pytest.raises(ValueError):
        # 尝试使用错误的距离度量参数调用kneighbors_graph函数
        neighbors.kneighbors_graph(X_nbrs, 3, metric="euclidean")
    X_nbrs = neighbors.NearestNeighbors(radius=radius, metric="manhattan")
    X_nbrs.fit(X)
    with pytest.raises(ValueError):
        # 尝试使用错误的距离度量参数调用radius_neighbors_graph函数
        neighbors.radius_neighbors_graph(X_nbrs, radius, metric="euclidean")


def check_object_arrays(nparray, list_check):
    # 检查对象数组nparray中的元素是否与列表list_check中的对应元素相等
    for ind, ele in enumerate(nparray):
        assert_array_equal(ele, list_check[ind])


def test_k_and_radius_neighbors_train_is_not_query():
    # 测试当查询数据不是训练数据时的kneighbors等函数
    # 对于每个算法，创建最近邻对象，设置邻居数量为1
    for algorithm in ALGORITHMS:
        nn = neighbors.NearestNeighbors(n_neighbors=1, algorithm=algorithm)
        
        # 定义数据集 X 并将其用于训练最近邻模型
        X = [[0], [1]]
        nn.fit(X)
        
        # 准备测试数据集
        test_data = [[2], [1]]
        
        # 测试最近邻搜索结果
        dist, ind = nn.kneighbors(test_data)
        assert_array_equal(dist, [[1], [0]])  # 断言距离数组与预期相同
        assert_array_equal(ind, [[1], [1]])   # 断言索引数组与预期相同
        
        # 使用半径进行最近邻搜索
        dist, ind = nn.radius_neighbors([[2], [1]], radius=1.5)
        check_object_arrays(dist, [[1], [1, 0]])   # 检查距离数组是否匹配预期
        check_object_arrays(ind, [[1], [0, 1]])    # 检查索引数组是否匹配预期
        
        # 测试图形变体
        assert_array_equal(
            nn.kneighbors_graph(test_data).toarray(), [[0.0, 1.0], [0.0, 1.0]]
        )  # 断言最近邻图的稀疏数组与预期相同
        
        assert_array_equal(
            nn.kneighbors_graph([[2], [1]], mode="distance").toarray(),
            np.array([[0.0, 1.0], [0.0, 0.0]]),
        )  # 断言带有距离模式的最近邻图的稀疏数组与预期相同
        
        # 使用半径生成最近邻图
        rng = nn.radius_neighbors_graph([[2], [1]], radius=1.5)
        assert_array_equal(rng.toarray(), [[0, 1], [1, 1]])  # 断言半径最近邻图的稀疏数组与预期相同
@pytest.mark.parametrize("algorithm", ALGORITHMS)
# 使用 pytest 的 parametrize 装饰器，对算法参数进行参数化测试
def test_k_and_radius_neighbors_X_None(algorithm):
    # 测试当查询为 None 时的 kneighbors 等方法
    nn = neighbors.NearestNeighbors(n_neighbors=1, algorithm=algorithm)
    # 创建最近邻对象 nn，设定邻居数为 1，选择算法为传入的 algorithm

    X = [[0], [1]]
    # 定义样本数据 X
    nn.fit(X)
    # 对样本数据 X 进行拟合

    dist, ind = nn.kneighbors()
    # 调用 kneighbors 方法获取距离和索引
    assert_array_equal(dist, [[1], [1]])
    # 断言距离的数组是否与预期一致
    assert_array_equal(ind, [[1], [0]])
    # 断言索引的数组是否与预期一致

    dist, ind = nn.radius_neighbors(None, radius=1.5)
    # 调用 radius_neighbors 方法，查询半径内的邻居，并获取距离和索引
    check_object_arrays(dist, [[1], [1]])
    # 使用 check_object_arrays 函数检查距离数组
    check_object_arrays(ind, [[1], [0]])
    # 使用 check_object_arrays 函数检查索引数组

    # 测试图形变体。
    rng = nn.radius_neighbors_graph(None, radius=1.5)
    # 调用 radius_neighbors_graph 方法获取半径邻居的图形表示
    kng = nn.kneighbors_graph(None)
    # 调用 kneighbors_graph 方法获取 k 近邻的图形表示
    for graph in [rng, kng]:
        # 遍历图形表示列表
        assert_array_equal(graph.toarray(), [[0, 1], [1, 0]])
        # 断言图形表示的稀疏矩阵是否与预期一致
        assert_array_equal(graph.data, [1, 1])
        # 断言图形表示的数据数组是否与预期一致
        assert_array_equal(graph.indices, [1, 0])
        # 断言图形表示的索引数组是否与预期一致

    X = [[0, 1], [0, 1], [1, 1]]
    # 重新定义样本数据 X
    nn = neighbors.NearestNeighbors(n_neighbors=2, algorithm=algorithm)
    # 创建新的最近邻对象 nn，设定邻居数为 2，选择算法为传入的 algorithm
    nn.fit(X)
    # 对样本数据 X 进行拟合
    assert_array_equal(
        nn.kneighbors_graph().toarray(),
        np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0]])
    )
    # 断言最近邻图的稀疏矩阵是否与预期一致


@pytest.mark.parametrize("algorithm", ALGORITHMS)
# 使用 pytest 的 parametrize 装饰器，对算法参数进行参数化测试
def test_k_and_radius_neighbors_duplicates(algorithm):
    # 测试在查询中存在重复数据时 kneighbors 方法的行为
    nn = neighbors.NearestNeighbors(n_neighbors=1, algorithm=algorithm)
    # 创建最近邻对象 nn，设定邻居数为 1，选择算法为传入的 algorithm
    duplicates = [[0], [1], [3]]

    nn.fit(duplicates)
    # 对重复数据进行拟合

    # 不对重复数据执行任何特殊处理。
    kng = nn.kneighbors_graph(duplicates, mode="distance")
    # 调用 kneighbors_graph 方法获取距离图形表示
    assert_allclose(
        kng.toarray(), np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    )
    # 断言距离图形表示的稀疏矩阵是否与预期一致
    assert_allclose(kng.data, [0.0, 0.0, 0.0])
    # 断言距离图形表示的数据数组是否与预期一致
    assert_allclose(kng.indices, [0, 1, 2])
    # 断言距离图形表示的索引数组是否与预期一致

    dist, ind = nn.radius_neighbors([[0], [1]], radius=1.5)
    # 调用 radius_neighbors 方法查询半径内的邻居，并获取距离和索引
    check_object_arrays(dist, [[0, 1], [1, 0]])
    # 使用 check_object_arrays 函数检查距离数组
    check_object_arrays(ind, [[0, 1], [0, 1]])
    # 使用 check_object_arrays 函数检查索引数组

    rng = nn.radius_neighbors_graph(duplicates, radius=1.5)
    # 调用 radius_neighbors_graph 方法获取半径邻居的图形表示
    assert_allclose(
        rng.toarray(), np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )
    # 断言半径邻居的图形表示的稀疏矩阵是否与预期一致

    rng = nn.radius_neighbors_graph([[0], [1]], radius=1.5, mode="distance")
    # 以距离模式调用 radius_neighbors_graph 方法获取半径邻居的图形表示
    rng.sort_indices()
    # 对索引进行排序
    assert_allclose(rng.toarray(), [[0, 1, 0], [1, 0, 0]])
    # 断言半径邻居的距离图形表示的稀疏矩阵是否与预期一致
    assert_allclose(rng.indices, [0, 1, 0, 1])
    # 断言半径邻居的距离图形表示的索引数组是否与预期一致
    assert_allclose(rng.data, [0, 1, 1, 0])
    # 断言半径邻居的距离图形表示的数据数组是否与预期一致

    # 当 n_duplicates > n_neighbors 时，屏蔽第一个重复数据。
    X = np.ones((3, 1))
    # 创建全为 1 的样本数据 X
    nn = neighbors.NearestNeighbors(n_neighbors=1, algorithm="brute")
    # 创建最近邻对象 nn，设定邻居数为 1，选择算法为 brute
    nn.fit(X)
    # 对样本数据 X 进行拟合
    dist, ind = nn.kneighbors()
    # 调用 kneighbors 方法获取距离和索引
    assert_allclose(dist, np.zeros((3, 1)))
    # 断言距离的数组是否全为 0
    assert_allclose(ind, [[1], [0], [1]])
    # 断言索引的数组是否与预期一致

    # 断言在 kneighbors_graph 中显式标记了零值。
    kng = nn.kneighbors_graph(mode="distance")
    # 以距离模式调用 kneighbors_graph 方法获取最近邻图形表示
    assert_allclose(kng.toarray(), np.zeros((3, 3)))
    # 断言最近邻图形表示的稀疏矩阵是否全为 0
    assert_allclose(kng.data, np.zeros(3))
    # 断言最近邻图形表示的数据数组是否全为 0
    assert_allclose(kng.indices, [1, 0, 1])
    # 断言最近邻图形表示的索引数组是否与预期一致
    assert_allclose(
        nn.kneighbors_graph().toarray(),
        np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    )
    # 断言最近邻图的稀疏矩阵是否与预
    # Test include_self parameter in neighbors_graph

    # 创建一个包含两个样本的特征矩阵
    X = [[2, 3], [4, 5]]

    # 使用 kneighbors_graph 函数生成带有自连接的 k 近邻图，转换为数组形式
    kng = neighbors.kneighbors_graph(X, 1, include_self=True).toarray()

    # 使用 kneighbors_graph 函数生成不带自连接的 k 近邻图，转换为数组形式
    kng_not_self = neighbors.kneighbors_graph(X, 1, include_self=False).toarray()

    # 断言带自连接的 k 近邻图是否正确
    assert_array_equal(kng, [[1.0, 0.0], [0.0, 1.0]])

    # 断言不带自连接的 k 近邻图是否正确
    assert_array_equal(kng_not_self, [[0.0, 1.0], [1.0, 0.0]])

    # 使用 radius_neighbors_graph 函数生成带自连接的半径邻居图，转换为数组形式
    rng = neighbors.radius_neighbors_graph(X, 5.0, include_self=True).toarray()

    # 使用 radius_neighbors_graph 函数生成不带自连接的半径邻居图，转换为数组形式
    rng_not_self = neighbors.radius_neighbors_graph(
        X, 5.0, include_self=False
    ).toarray()

    # 断言带自连接的半径邻居图是否正确
    assert_array_equal(rng, [[1.0, 1.0], [1.0, 1.0]])

    # 断言不带自连接的半径邻居图是否正确
    assert_array_equal(rng_not_self, [[0.0, 1.0], [1.0, 0.0]])
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_same_knn_parallel(algorithm):
    # 使用参数化测试，对每个算法进行测试
    X, y = datasets.make_classification(
        n_samples=30, n_features=5, n_redundant=0, random_state=0
    )
    # 生成分类数据集，包括特征矩阵 X 和标签向量 y
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # 将数据集划分为训练集和测试集

    clf = neighbors.KNeighborsClassifier(n_neighbors=3, algorithm=algorithm)
    # 初始化 K 近邻分类器，设定邻居数为 3 和算法为指定的算法
    clf.fit(X_train, y_train)
    # 使用训练集训练分类器
    y = clf.predict(X_test)
    # 对测试集进行预测，得到预测标签 y
    dist, ind = clf.kneighbors(X_test)
    # 计算测试集中每个样本的最近邻距离和索引
    graph = clf.kneighbors_graph(X_test, mode="distance").toarray()
    # 构建测试集的最近邻图，以距离为权重

    clf.set_params(n_jobs=3)
    # 设置分类器使用的并行作业数为 3
    clf.fit(X_train, y_train)
    # 使用训练集重新训练分类器
    y_parallel = clf.predict(X_test)
    # 对测试集进行并行预测，得到并行预测标签 y_parallel
    dist_parallel, ind_parallel = clf.kneighbors(X_test)
    # 计算并行预测的最近邻距离和索引
    graph_parallel = clf.kneighbors_graph(X_test, mode="distance").toarray()
    # 构建并行预测的最近邻图，以距离为权重

    assert_array_equal(y, y_parallel)
    # 检查串行预测和并行预测的预测标签是否一致
    assert_allclose(dist, dist_parallel)
    # 检查串行预测和并行预测的最近邻距离是否接近
    assert_array_equal(ind, ind_parallel)
    # 检查串行预测和并行预测的最近邻索引是否一致
    assert_allclose(graph, graph_parallel)
    # 检查串行预测和并行预测的最近邻图是否接近


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_same_radius_neighbors_parallel(algorithm):
    # 使用参数化测试，对每个算法进行测试
    X, y = datasets.make_classification(
        n_samples=30, n_features=5, n_redundant=0, random_state=0
    )
    # 生成分类数据集，包括特征矩阵 X 和标签向量 y
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # 将数据集划分为训练集和测试集

    clf = neighbors.RadiusNeighborsClassifier(radius=10, algorithm=algorithm)
    # 初始化半径最近邻分类器，设定半径为 10 和算法为指定的算法
    clf.fit(X_train, y_train)
    # 使用训练集训练分类器
    y = clf.predict(X_test)
    # 对测试集进行预测，得到预测标签 y
    dist, ind = clf.radius_neighbors(X_test)
    # 计算测试集中每个样本的半径内邻居距离和索引
    graph = clf.radius_neighbors_graph(X_test, mode="distance").toarray()
    # 构建测试集的半径内邻居图，以距离为权重

    clf.set_params(n_jobs=3)
    # 设置分类器使用的并行作业数为 3
    clf.fit(X_train, y_train)
    # 使用训练集重新训练分类器
    y_parallel = clf.predict(X_test)
    # 对测试集进行并行预测，得到并行预测标签 y_parallel
    dist_parallel, ind_parallel = clf.radius_neighbors(X_test)
    # 计算并行预测的半径内邻居距离和索引
    graph_parallel = clf.radius_neighbors_graph(X_test, mode="distance").toarray()
    # 构建并行预测的半径内邻居图，以距离为权重

    assert_array_equal(y, y_parallel)
    # 检查串行预测和并行预测的预测标签是否一致
    for i in range(len(dist)):
        assert_allclose(dist[i], dist_parallel[i])
        # 检查串行预测和并行预测的每个样本的半径内邻居距离是否接近
        assert_array_equal(ind[i], ind_parallel[i])
        # 检查串行预测和并行预测的每个样本的半径内邻居索引是否一致
    assert_allclose(graph, graph_parallel)
    # 检查串行预测和并行预测的半径内邻居图是否接近


@pytest.mark.parametrize("backend", ["threading", "loky"])
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_knn_forcing_backend(backend, algorithm):
    # 参数化测试，测试不同的并行后端和每个算法
    # 非回归测试，确保 knn 方法在强制使用全局 joblib 后端时正常工作
    with joblib.parallel_backend(backend):
        # 使用指定的并行后端执行以下代码块
        X, y = datasets.make_classification(
            n_samples=30, n_features=5, n_redundant=0, random_state=0
        )
        # 生成分类数据集，包括特征矩阵 X 和标签向量 y
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        # 将数据集划分为训练集和测试集

        clf = neighbors.KNeighborsClassifier(
            n_neighbors=3, algorithm=algorithm, n_jobs=2
        )
        # 初始化 K 近邻分类器，设定邻居数为 3，算法为指定的算法，使用 2 个并行作业
        clf.fit(X_train, y_train)
        # 使用训练集训练分类器
        clf.predict(X_test)
        # 对测试集进行预测
        clf.kneighbors(X_test)
        # 计算测试集中每个样本的最近邻距离和索引
        clf.kneighbors_graph(X_test, mode="distance")
        # 构建测试集的最近邻图，以距离为权重


def test_dtype_convert():
    # 测试数据类型转换函数
    classifier = neighbors.KNeighborsClassifier(n_neighbors=1)
    # 初始化 K 近邻分类器，设定邻居数为 1
    CLASSES = 15
    X = np.eye(CLASSES)
    # 创建一个单位矩阵作为特征矩阵 X
    y = [ch for ch in "ABCDEFGHIJKLMNOPQRSTU"[:CLASSES]]
    # 创建相应长度的标签列表 y

    result = classifier.fit(X, y).predict(X)
    # 对数据集 X, y 进行拟合和预测
    assert_array_equal(result, y)
    # 检查预测结果是否与真实标签 y 一致
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sparse_metric_callable(csr_container):
    # 定义一个接受稀疏矩阵输入的度量函数
    def sparse_metric(x, y):
        assert issparse(x) and issparse(y)  # 确保输入是稀疏矩阵
        return x.dot(y.T).toarray().item()  # 计算稀疏矩阵 x 和 y 的点积并返回其标量值

    X = csr_container(
        [[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [0, 0, 1, 0, 0]]  # 人口矩阵
    )

    Y = csr_container([[1, 1, 0, 1, 1], [1, 0, 0, 1, 1]])  # 查询矩阵

    nn = neighbors.NearestNeighbors(
        algorithm="brute", n_neighbors=2, metric=sparse_metric
    ).fit(X)
    N = nn.kneighbors(Y, return_distance=False)

    # `sparse_metric` 对应的 `X` 中最近邻居的索引
    gold_standard_nn = np.array([[2, 1], [2, 1]])

    assert_array_equal(N, gold_standard_nn)


# 忽略在 pairwise_distances 中的布尔值转换警告
@ignore_warnings(category=DataConversionWarning)
def test_pairwise_boolean_distance():
    # 非回归测试 #4523
    # 'brute': 通过 pairwise_distances 使用 scipy.spatial.distance
    # 'ball_tree': 使用 sklearn.neighbors._dist_metrics
    rng = np.random.RandomState(0)
    X = rng.uniform(size=(6, 5))
    NN = neighbors.NearestNeighbors

    nn1 = NN(metric="jaccard", algorithm="brute").fit(X)
    nn2 = NN(metric="jaccard", algorithm="ball_tree").fit(X)
    assert_array_equal(nn1.kneighbors(X)[0], nn2.kneighbors(X)[0])


def test_radius_neighbors_predict_proba():
    for seed in range(5):
        X, y = datasets.make_classification(
            n_samples=50,
            n_features=5,
            n_informative=3,
            n_redundant=0,
            n_classes=3,
            random_state=seed,
        )
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)
        outlier_label = int(2 - seed)
        clf = neighbors.RadiusNeighborsClassifier(radius=2, outlier_label=outlier_label)
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        proba = clf.predict_proba(X_te)
        proba_label = proba.argmax(axis=1)
        proba_label = np.where(proba.sum(axis=1) == 0, outlier_label, proba_label)
        assert_array_equal(pred, proba_label)


def test_pipeline_with_nearest_neighbors_transformer():
    # 测试将 KNeighborsTransformer 和分类器/回归器链式化
    rng = np.random.RandomState(0)
    X = 2 * rng.rand(40, 5) - 1
    X2 = 2 * rng.rand(40, 5) - 1
    y = rng.rand(40, 1)

    n_neighbors = 12
    radius = 1.5
    # 预计算比必要的邻居更多，以保证在半径邻居转换器后的 k-邻居估算器和反之间的等价性
    factor = 2

    k_trans = neighbors.KNeighborsTransformer(n_neighbors=n_neighbors, mode="distance")
    k_trans_factor = neighbors.KNeighborsTransformer(
        n_neighbors=int(n_neighbors * factor), mode="distance"
    )

    r_trans = neighbors.RadiusNeighborsTransformer(radius=radius, mode="distance")
    # 创建一个基于半径的邻居转换器对象，用于距离模式
    r_trans_factor = neighbors.RadiusNeighborsTransformer(
        radius=int(radius * factor), mode="distance"
    )

    # 创建一个 K 近邻回归器对象，设定邻居数为 n_neighbors
    k_reg = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors)
    
    # 创建一个基于半径的邻居回归器对象，设定半径为 radius
    r_reg = neighbors.RadiusNeighborsRegressor(radius=radius)

    # 创建一个测试列表，包含不同的转换器和回归器组合
    test_list = [
        (k_trans, k_reg),           # 使用 k_trans 和 k_reg 的组合
        (k_trans_factor, r_reg),    # 使用 k_trans_factor 和 r_reg 的组合
        (r_trans, r_reg),           # 使用 r_trans 和 r_reg 的组合
        (r_trans_factor, k_reg),    # 使用 r_trans_factor 和 k_reg 的组合
    ]

    # 遍历测试列表中的每个转换器和回归器组合
    for trans, reg in test_list:
        # 克隆回归器对象，生成紧凑版本和预计算版本
        reg_compact = clone(reg)
        reg_precomp = clone(reg)
        reg_precomp.set_params(metric="precomputed")

        # 创建管道，包含给定的转换器和预计算版本的回归器
        reg_chain = make_pipeline(clone(trans), reg_precomp)

        # 在训练数据 X 上拟合管道模型，并预测测试数据 X2
        y_pred_chain = reg_chain.fit(X, y).predict(X2)
        # 在训练数据 X 上拟合紧凑版本的回归器，并预测测试数据 X2
        y_pred_compact = reg_compact.fit(X, y).predict(X2)
        
        # 断言管道预测结果与紧凑版本预测结果近似相等
        assert_allclose(y_pred_chain, y_pred_compact)
# 使用 pytest.mark.parametrize 装饰器，为 test_auto_algorithm 函数定义多组参数化测试数据
@pytest.mark.parametrize(
    "X, metric, metric_params, expected_algo",
    [
        # 第一组参数化测试数据：使用随机整数数组，指定 metric 为 "precomputed"，无需额外参数，期望的算法是 "brute"
        (np.random.randint(10, size=(10, 10)), "precomputed", None, "brute"),
        # 第二组参数化测试数据：使用随机正态分布数组，metric 为 "euclidean"，无需额外参数，期望的算法是 "brute"
        (np.random.randn(10, 20), "euclidean", None, "brute"),
        # 第三组参数化测试数据：使用不同大小的随机正态分布数组，metric 为 "euclidean"，无需额外参数，期望的算法是 "brute"
        (np.random.randn(8, 5), "euclidean", None, "brute"),
        # 第四组参数化测试数据：使用随机正态分布数组，metric 为 "euclidean"，无需额外参数，期望的算法是 "kd_tree"
        (np.random.randn(10, 5), "euclidean", None, "kd_tree"),
        # 第五组参数化测试数据：使用随机正态分布数组，metric 为 "seuclidean"，参数包括 V=[2]*5，期望的算法是 "ball_tree"
        (np.random.randn(10, 5), "seuclidean", {"V": [2] * 5}, "ball_tree"),
        # 第六组参数化测试数据：使用随机正态分布数组，metric 为 "correlation"，无需额外参数，期望的算法是 "brute"
        (np.random.randn(10, 5), "correlation", None, "brute"),
    ],
)
# 定义测试函数 test_auto_algorithm，测试 NearestNeighbors 模型的自动算法选择功能
def test_auto_algorithm(X, metric, metric_params, expected_algo):
    # 创建 NearestNeighbors 模型对象，设置 n_neighbors=4，algorithm="auto"，指定 metric 和 metric_params
    model = neighbors.NearestNeighbors(
        n_neighbors=4, algorithm="auto", metric=metric, metric_params=metric_params
    )
    # 将模型拟合到数据 X 上
    model.fit(X)
    # 断言模型的 _fit_method 属性与期望的算法名 expected_algo 相符
    assert model._fit_method == expected_algo


# 使用 pytest.mark.parametrize 装饰器，为 test_radius_neighbors_brute_backend 函数定义参数化测试数据
@pytest.mark.parametrize(
    "metric", sorted(set(neighbors.VALID_METRICS["brute"]) - set(["precomputed"]))
)
# 定义测试函数 test_radius_neighbors_brute_backend，测试 radius_neighbors 方法在 "brute" 算法下的后端一致性
def test_radius_neighbors_brute_backend(
    metric,
    global_random_seed,
    global_dtype,
    n_samples=2000,
    n_features=30,
    n_query_pts=5,
    radius=1.0,
):
    # 使用全局随机种子创建随机数生成器
    rng = np.random.RandomState(global_random_seed)
    # 创建训练数据 X_train 和测试数据 X_test，分别是随机数数组，指定数据类型为 global_dtype
    X_train = rng.rand(n_samples, n_features).astype(global_dtype, copy=False)
    X_test = rng.rand(n_query_pts, n_features).astype(global_dtype, copy=False)

    # 对于 metric 为 "haversine"，只接受二维数据
    if metric == "haversine":
        feature_sl = slice(None, 2)
        X_train = np.ascontiguousarray(X_train[:, feature_sl])
        X_test = np.ascontiguousarray(X_test[:, feature_sl])

    # 生成指定 metric 下的测试参数列表
    metric_params_list = _generate_test_params_for(metric, n_features)

    # 遍历测试参数列表
    for metric_params in metric_params_list:
        p = metric_params.pop("p", 2)

        # 创建 NearestNeighbors 模型对象，使用 "brute" 算法，指定 radius、metric、p 和 metric_params
        neigh = neighbors.NearestNeighbors(
            radius=radius,
            algorithm="brute",
            metric=metric,
            p=p,
            metric_params=metric_params,
        )

        # 将模型拟合到训练数据 X_train 上
        neigh.fit(X_train)

        # 在禁用 Cython 优化的环境下执行 radius_neighbors 方法，得到 legacy_brute_dst 和 legacy_brute_idx
        with config_context(enable_cython_pairwise_dist=False):
            legacy_brute_dst, legacy_brute_idx = neigh.radius_neighbors(
                X_test, return_distance=True
            )
        # 在启用 Cython 优化的环境下执行 radius_neighbors 方法，得到 pdr_brute_dst 和 pdr_brute_idx
        with config_context(enable_cython_pairwise_dist=True):
            pdr_brute_dst, pdr_brute_idx = neigh.radius_neighbors(
                X_test, return_distance=True
            )

        # 断言 legacy 和 pdr 两种后端计算结果的一致性
        assert_compatible_radius_results(
            legacy_brute_dst,
            pdr_brute_dst,
            legacy_brute_idx,
            pdr_brute_idx,
            radius=radius,
            check_sorted=False,
        )


# 定义测试函数 test_valid_metrics_has_no_duplicate，检查 neighbors.VALID_METRICS 中的指标是否无重复
def test_valid_metrics_has_no_duplicate():
    for val in neighbors.VALID_METRICS.values():
        assert len(val) == len(set(val))


# 定义测试函数 test_regressor_predict_on_arraylikes，确保在 weights 为可调用对象时，predict 方法可以处理数组样式的输入
# 这是对 issue #22687 的非回归测试
def test_regressor_predict_on_arraylikes():
    pass  # 此处不执行具体测试，只是函数的简要描述
    # 定义输入特征矩阵 X，每行表示一个样本，每列表示一个特征
    X = [[5, 1], [3, 1], [4, 3], [0, 3]]
    # 定义目标值向量 y，每个元素对应 X 中相应样本的目标值
    y = [2, 3, 5, 6]
    
    # 定义用于计算权重的函数 _weights，该函数返回与输入距离相同形状的全一数组
    def _weights(dist):
        return np.ones_like(dist)
    
    # 创建 K 最近邻回归器对象 est，设置邻居数为 1，算法为 "brute"，权重函数为 _weights
    est = KNeighborsRegressor(n_neighbors=1, algorithm="brute", weights=_weights)
    # 使用输入数据 X 和目标值 y 来训练回归器对象 est
    est.fit(X, y)
    # 断言预测样本 [[0, 2.5]] 的预测值接近 [6]
    assert_allclose(est.predict([[0, 2.5]]), [6])
# 检查 KNN 在数据框上的预测是否正常工作

# 非回归测试，用于解决问题＃26768
def test_predict_dataframe():
    pd = pytest.importorskip("pandas")  # 导入并检查是否存在 pandas 库

    # 创建一个包含数据的数据框 X 和目标变量 y
    X = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), columns=["a", "b"])
    y = np.array([1, 2, 3, 4])

    # 使用 KNeighborsClassifier 拟合数据
    knn = neighbors.KNeighborsClassifier(n_neighbors=2).fit(X, y)
    
    # 对数据框 X 进行预测
    knn.predict(X)


# 检查 NearestNeighbors 在 p 在 (0,1) 区间内时的工作情况，当算法是 "auto" 或 "brute" 时
# 不管 X 的数据类型是什么

# 非回归测试，用于解决问题＃26548
def test_nearest_neighbours_works_with_p_less_than_1():
    X = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])

    # 创建 NearestNeighbors 对象，使用 brute 算法和 p=0.5 的参数
    neigh = neighbors.NearestNeighbors(
        n_neighbors=3, algorithm="brute", metric_params={"p": 0.5}
    )
    neigh.fit(X)

    # 对数据 X[0] 使用半径邻居搜索，半径为 4，返回不考虑距离的最近邻居索引
    y = neigh.radius_neighbors(X[0].reshape(1, -1), radius=4, return_distance=False)
    assert_allclose(y[0], [0, 1, 2])  # 断言确保结果与预期接近

    # 对数据 X[0] 使用 k 近邻搜索，返回不考虑距离的最近 k 个邻居的索引
    y = neigh.kneighbors(X[0].reshape(1, -1), return_distance=False)
    assert_allclose(y[0], [0, 1, 2])  # 断言确保结果与预期接近


# 检查在所有零权重样本上，KNeighborsClassifier 的 `predict` 和 `predict_proba` 是否会引发异常

# 相关问题＃25854
def test_KNeighborsClassifier_raise_on_all_zero_weights():
    X = [[0, 1], [1, 2], [2, 3], [3, 4]]
    y = [0, 0, 1, 1]

    # 定义一个返回零或一权重的函数 _weights
    def _weights(dist):
        return np.vectorize(lambda x: 0 if x > 0.5 else 1)(dist)

    # 使用定义的权重函数 _weights 创建 KNeighborsClassifier 对象
    est = neighbors.KNeighborsClassifier(n_neighbors=3, weights=_weights)
    est.fit(X, y)

    # 预测一个样本时，所有邻居的权重都是零，期望引发 ValueError 异常
    msg = (
        "All neighbors of some sample is getting zero weights. "
        "Please modify 'weights' to avoid this case if you are "
        "using a user-defined function."
    )

    # 使用 pytest 断言预期的异常消息
    with pytest.raises(ValueError, match=msg):
        est.predict([[1.1, 1.1]])

    # 使用 pytest 断言预期的异常消息
    with pytest.raises(ValueError, match=msg):
        est.predict_proba([[1.1, 1.1]])
```