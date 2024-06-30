# `D:\src\scipysrc\scikit-learn\sklearn\cluster\tests\test_optics.py`

```
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入警告模块
import warnings

# 导入 NumPy 库并重命名为 np
import numpy as np

# 导入 pytest 测试框架
import pytest

# 导入 DBSCAN 和 OPTICS 聚类算法
from sklearn.cluster import DBSCAN, OPTICS

# 导入 _extend_region 和 _extract_xi_labels 函数
from sklearn.cluster._optics import _extend_region, _extract_xi_labels

# 导入生成聚类数据的函数
from sklearn.cluster.tests.common import generate_clustered_data

# 导入生成高斯分布的数据集函数
from sklearn.datasets import make_blobs

# 导入数据转换警告和效率警告
from sklearn.exceptions import DataConversionWarning, EfficiencyWarning

# 导入用于计算聚类指标的函数
from sklearn.metrics.cluster import contingency_matrix

# 导入计算两两距离的函数
from sklearn.metrics.pairwise import pairwise_distances

# 导入数据洗牌函数
from sklearn.utils import shuffle

# 导入用于断言的函数
from sklearn.utils._testing import assert_allclose, assert_array_equal

# 导入 CSR 格式容器的修复函数
from sklearn.utils.fixes import CSR_CONTAINERS

# 创建一个随机数生成器对象
rng = np.random.RandomState(0)

# 每个聚类点数目
n_points_per_cluster = 10

# 生成具有不同中心的聚类数据集
C1 = [-5, -2] + 0.8 * rng.randn(n_points_per_cluster, 2)
C2 = [4, -1] + 0.1 * rng.randn(n_points_per_cluster, 2)
C3 = [1, -2] + 0.2 * rng.randn(n_points_per_cluster, 2)
C4 = [-2, 3] + 0.3 * rng.randn(n_points_per_cluster, 2)
C5 = [3, -2] + 1.6 * rng.randn(n_points_per_cluster, 2)
C6 = [5, 6] + 2 * rng.randn(n_points_per_cluster, 2)

# 将所有数据堆叠在一起形成最终的数据集
X = np.vstack((C1, C2, C3, C4, C5, C6))


# 使用 pytest 的参数化装饰器进行多组测试
@pytest.mark.parametrize(
    ("r_plot", "end"),
    [
        [[10, 8.9, 8.8, 8.7, 7, 10], 3],  # 测试降低区域的情况
        [[10, 8.9, 8.8, 8.7, 8.6, 7, 10], 0],  # 测试非降低区域的情况
        [[10, 8.9, 8.8, 8.7, 7, 6, np.inf], 4],  # 测试包含无穷大的情况
        [[10, 8.9, 8.8, 8.7, 7, 6, np.inf], 4],  # 再次测试相同情况，确认结果
    ],
)
def test_extend_downward(r_plot, end):
    # 将 r_plot 转换为 NumPy 数组
    r_plot = np.array(r_plot)
    
    # 计算相邻元素的比率
    ratio = r_plot[:-1] / r_plot[1:]
    
    # 筛选出急剧下降的区域
    steep_downward = ratio >= 1 / 0.9
    
    # 筛选出上升的区域
    upward = ratio < 1
    
    # 执行 _extend_region 函数进行测试
    e = _extend_region(steep_downward, upward, 0, 2)
    
    # 断言测试结果是否符合预期
    assert e == end


# 使用 pytest 的参数化装饰器进行多组测试
@pytest.mark.parametrize(
    ("r_plot", "end"),
    [
        [[1, 2, 2.1, 2.2, 4, 8, 8, np.inf], 6],  # 测试上升区域的情况
        [[1, 2, 2.1, 2.2, 2.3, 4, 8, 8, np.inf], 0],  # 测试非上升区域的情况
        [[1, 2, 2.1, 2, np.inf], 0],  # 测试包含无穷大的情况
        [[1, 2, 2.1, np.inf], 2],  # 测试只有一个急剧上升的区域
    ],
)
def test_extend_upward(r_plot, end):
    # 将 r_plot 转换为 NumPy 数组
    r_plot = np.array(r_plot)
    
    # 计算相邻元素的比率
    ratio = r_plot[:-1] / r_plot[1:]
    
    # 筛选出急剧上升的区域
    steep_upward = ratio <= 0.9
    
    # 筛选出下降的区域
    downward = ratio > 1
    
    # 执行 _extend_region 函数进行测试
    e = _extend_region(steep_upward, downward, 0, 2)
    
    # 断言测试结果是否符合预期
    assert e == end


# 使用 pytest 的参数化装饰器进行多组测试
@pytest.mark.parametrize(
    ("ordering", "clusters", "expected"),
    [
        [[0, 1, 2, 3], [[0, 1], [2, 3]], [0, 0, 1, 1]],  # 测试基本情况
        [[0, 1, 2, 3], [[0, 1], [3, 3]], [0, 0, -1, 1]],  # 测试包含噪声点的情况
        [[0, 1, 2, 3], [[0, 1], [3, 3], [0, 3]], [0, 0, -1, 1]],  # 测试多个簇的情况
        [[3, 1, 2, 0], [[0, 1], [3, 3], [0, 3]], [1, 0, -1, 0]],  # 测试不同顺序的情况
    ],
)
def test_the_extract_xi_labels(ordering, clusters, expected):
    # 调用 _extract_xi_labels 函数
    labels = _extract_xi_labels(ordering, clusters)
    
    # 断言测试结果是否符合预期
    assert_array_equal(labels, expected)


def test_extract_xi(global_dtype):
    # small and easy test (no clusters around other clusters)
    # but with a clear noise data.
    
    # 使用固定的随机数种子重新初始化随机数生成器
    rng = np.random.RandomState(0)
    
    # 每个聚类点数目
    n_points_per_cluster = 5

    # 生成具有不同中心的简单聚类数据集
    C1 = [-5, -2] + 0.8 * rng.randn(n_points_per_cluster, 2)
    C2 = [4, -1] + 0.1 * rng.randn(n_points_per_cluster, 2)
    C3 = [1, -2] + 0.2 * rng.randn(n_points_per_cluster, 2)
    # 创建数据集 C4，包含指定均值和随机扰动的点集
    C4 = [-2, 3] + 0.3 * rng.randn(n_points_per_cluster, 2)
    # 创建数据集 C5，包含指定均值和随机扰动的点集
    C5 = [3, -2] + 0.6 * rng.randn(n_points_per_cluster, 2)
    # 创建数据集 C6，包含指定均值和随机扰动的点集
    C6 = [5, 6] + 0.2 * rng.randn(n_points_per_cluster, 2)

    # 将所有数据集堆叠成一个新的数据集 X，并转换其数据类型为 global_dtype
    X = np.vstack((C1, C2, C3, C4, C5, np.array([[100, 100]]), C6)).astype(
        global_dtype, copy=False
    )
    # 创建预期的标签数组 expected_labels，并进行随机打乱
    expected_labels = np.r_[[2] * 5, [0] * 5, [1] * 5, [3] * 5, [1] * 5, -1, [4] * 5]
    X, expected_labels = shuffle(X, expected_labels, random_state=rng)

    # 使用 OPTICS 算法对数据集 X 进行聚类，设置参数并拟合模型
    clust = OPTICS(
        min_samples=3, min_cluster_size=2, max_eps=20, cluster_method="xi", xi=0.4
    ).fit(X)
    # 断言聚类结果与预期标签一致
    assert_array_equal(clust.labels_, expected_labels)

    # 检查浮点数类型的 min_samples 和 min_cluster_size
    clust = OPTICS(
        min_samples=0.1, min_cluster_size=0.08, max_eps=20, cluster_method="xi", xi=0.4
    ).fit(X)
    # 断言聚类结果与预期标签一致
    assert_array_equal(clust.labels_, expected_labels)

    # 修改 X 数据集的一个部分并重新设置预期标签
    X = np.vstack((C1, C2, C3, C4, C5, np.array([[100, 100]] * 2), C6)).astype(
        global_dtype, copy=False
    )
    expected_labels = np.r_[
        [1] * 5, [3] * 5, [2] * 5, [0] * 5, [2] * 5, -1, -1, [4] * 5
    ]
    X, expected_labels = shuffle(X, expected_labels, random_state=rng)

    # 使用 OPTICS 算法再次对修改后的数据集 X 进行聚类，设置参数并拟合模型
    clust = OPTICS(
        min_samples=3, min_cluster_size=3, max_eps=20, cluster_method="xi", xi=0.3
    ).fit(X)
    # 断言聚类结果与重新设置的预期标签一致，这可能失败，除非前任校正起作用！
    assert_array_equal(clust.labels_, expected_labels)

    # 创建数据集 C1、C2、C3，并堆叠成新的数据集 X
    C1 = [[0, 0], [0, 0.1], [0, -0.1], [0.1, 0]]
    C2 = [[10, 10], [10, 9], [10, 11], [9, 10]]
    C3 = [[100, 100], [100, 90], [100, 110], [90, 100]]
    X = np.vstack((C1, C2, C3)).astype(global_dtype, copy=False)
    # 创建预期的标签数组 expected_labels，并进行随机打乱
    expected_labels = np.r_[[0] * 4, [1] * 4, [2] * 4]
    X, expected_labels = shuffle(X, expected_labels, random_state=rng)

    # 使用 OPTICS 算法对数据集 X 进行聚类，设置参数并拟合模型
    clust = OPTICS(
        min_samples=2, min_cluster_size=2, max_eps=np.inf, cluster_method="xi", xi=0.04
    ).fit(X)
    # 断言聚类结果与预期标签一致
    assert_array_equal(clust.labels_, expected_labels)
def test_cluster_hierarchy_(global_dtype):
    # 创建一个随机数生成器对象，用于生成随机数据
    rng = np.random.RandomState(0)
    # 每个聚类点的数量
    n_points_per_cluster = 100
    # 创建第一个聚类中心，并生成随机数据，类型为全局指定的数据类型
    C1 = [0, 0] + 2 * rng.randn(n_points_per_cluster, 2).astype(
        global_dtype, copy=False
    )
    # 创建第二个聚类中心，并生成随机数据，类型为全局指定的数据类型
    C2 = [0, 0] + 50 * rng.randn(n_points_per_cluster, 2).astype(
        global_dtype, copy=False
    )
    # 将两个聚类中心的数据堆叠起来形成输入数据 X，并进行随机打乱
    X = np.vstack((C1, C2))
    X = shuffle(X, random_state=0)

    # 使用 OPTICS 算法拟合数据 X，并获取聚类层次
    clusters = OPTICS(min_samples=20, xi=0.1).fit(X).cluster_hierarchy_
    # 断言聚类结果的形状为 (2, 2)
    assert clusters.shape == (2, 2)
    # 计算与预期聚类结果的差异
    diff = np.sum(clusters - np.array([[0, 99], [0, 199]]))
    # 断言差异在数据总量的比例小于 0.05
    assert diff / len(X) < 0.05


@pytest.mark.parametrize(
    "csr_container, metric",
    [(None, "minkowski")] + [(container, "euclidean") for container in CSR_CONTAINERS],
)
def test_correct_number_of_clusters(metric, csr_container):
    # 在 'auto' 模式下

    # 设定聚类数目为 3
    n_clusters = 3
    # 生成聚类数据 X
    X = generate_clustered_data(n_clusters=n_clusters)
    # 为本任务特别选择的参数
    # 计算 OPTICS
    clust = OPTICS(max_eps=5.0 * 6.0, min_samples=4, xi=0.1, metric=metric)
    # 根据是否有 csr_container，对 X 进行处理后拟合 OPTICS
    clust.fit(csr_container(X) if csr_container is not None else X)
    # 获取聚类数目，忽略噪声数据
    n_clusters_1 = len(set(clust.labels_)) - int(-1 in clust.labels_)
    # 断言聚类数目与预期相符
    assert n_clusters_1 == n_clusters

    # 检查属性的类型和大小
    assert clust.labels_.shape == (len(X),)
    assert clust.labels_.dtype.kind == "i"

    assert clust.reachability_.shape == (len(X),)
    assert clust.reachability_.dtype.kind == "f"

    assert clust.core_distances_.shape == (len(X),)
    assert clust.core_distances_.dtype.kind == "f"

    assert clust.ordering_.shape == (len(X),)
    assert clust.ordering_.dtype.kind == "i"
    assert set(clust.ordering_) == set(range(len(X)))


def test_minimum_number_of_sample_check():
    # 测试是否检查了最小样本数
    msg = "min_samples must be no greater than"

    # 计算 OPTICS
    X = [[1, 1]]
    clust = OPTICS(max_eps=5.0 * 0.3, min_samples=10, min_cluster_size=1.0)

    # 运行拟合过程
    with pytest.raises(ValueError, match=msg):
        clust.fit(X)


def test_bad_extract():
    # 测试提取的 epsilon 是否过于接近原始的 epsilon
    msg = "Specify an epsilon smaller than 0.15. Got 0.3."
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(
        n_samples=750, centers=centers, cluster_std=0.4, random_state=0
    )

    # 计算 OPTICS
    clust = OPTICS(max_eps=5.0 * 0.03, cluster_method="dbscan", eps=0.3, min_samples=10)
    with pytest.raises(ValueError, match=msg):
        clust.fit(X)


def test_bad_reachability():
    msg = "All reachability values are inf. Set a larger max_eps."
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(
        n_samples=750, centers=centers, cluster_std=0.4, random_state=0
    )

    with pytest.warns(UserWarning, match=msg):
        clust = OPTICS(max_eps=5.0 * 0.003, min_samples=10, eps=0.015)
        clust.fit(X)


def test_nowarn_if_metric_bool_data_bool():
    # 这是一个空的测试函数，用于测试特定情况下是否发出警告
    pass
    # 确保在metric和data都是布尔类型时不会引发警告
    # 非回归测试
    # https://github.com/scikit-learn/scikit-learn/issues/18996

    # 设置pairwise_metric变量为"rogerstanimoto"
    pairwise_metric = "rogerstanimoto"
    
    # 生成一个5行2列的随机布尔类型数组X
    X = np.random.randint(2, size=(5, 2), dtype=bool)

    # 使用warnings模块捕获警告
    with warnings.catch_warnings():
        # 设置警告过滤器，使得DataConversionWarning变为错误
        warnings.simplefilter("error", DataConversionWarning)

        # 使用OPTICS算法，使用指定的metric参数（pairwise_metric）拟合数据X
        OPTICS(metric=pairwise_metric).fit(X)
# 确保如果度量是布尔型而数据不是，则会触发单个的转换警告
# 针对 https://github.com/scikit-learn/scikit-learn/issues/18996 的非回归测试

def test_warn_if_metric_bool_data_no_bool():
    pairwise_metric = "rogerstanimoto"  # 设置成对度量的字符串
    X = np.random.randint(2, size=(5, 2), dtype=np.int32)  # 创建一个5x2的随机整数数组
    msg = f"Data will be converted to boolean for metric {pairwise_metric}"  # 创建警告消息字符串

    with pytest.warns(DataConversionWarning, match=msg) as warn_record:  # 捕获特定警告并记录到 warn_record
        OPTICS(metric=pairwise_metric).fit(X)  # 使用 OPTICS 算法拟合数据 X
        assert len(warn_record) == 1  # 断言确保只有一个警告记录


# 确保如果度量不是布尔型，则不会触发任何转换警告，无论数据类型如何
def test_nowarn_if_metric_no_bool():
    pairwise_metric = "minkowski"  # 设置成对度量的字符串
    X_bool = np.random.randint(2, size=(5, 2), dtype=bool)  # 创建一个5x2的随机布尔数组
    X_num = np.random.randint(2, size=(5, 2), dtype=np.int32)  # 创建一个5x2的随机整数数组

    with warnings.catch_warnings():  # 捕获警告
        warnings.simplefilter("error", DataConversionWarning)  # 将 DataConversionWarning 警告转换为错误

        OPTICS(metric=pairwise_metric).fit(X_bool)  # 使用 OPTICS 算法拟合布尔数据
        OPTICS(metric=pairwise_metric).fit(X_num)   # 使用 OPTICS 算法拟合整数数据


# 测试当提取 eps 接近于缩放后的 max_eps 时的情况
def test_close_extract():
    centers = [[1, 1], [-1, -1], [1, -1]]  # 设置数据中心点坐标
    X, labels_true = make_blobs(  # 创建聚类数据 X 和真实标签
        n_samples=750, centers=centers, cluster_std=0.4, random_state=0
    )

    # 计算 OPTICS
    clust = OPTICS(max_eps=1.0, cluster_method="dbscan", eps=0.3, min_samples=10).fit(X)
    # 聚类排序从 0 开始；最大聚类标签为 2 表示 3 个聚类
    assert max(clust.labels_) == 2


# 参数化测试，测试 OPTICS 聚类标签与 DBSCAN 标签的一致性
@pytest.mark.parametrize("eps", [0.1, 0.3, 0.5])
@pytest.mark.parametrize("min_samples", [3, 10, 20])
@pytest.mark.parametrize(
    "csr_container, metric",
    [(None, "minkowski"), (None, "euclidean")]
    + [(container, "euclidean") for container in CSR_CONTAINERS],
)
def test_dbscan_optics_parity(eps, min_samples, metric, global_dtype, csr_container):
    centers = [[1, 1], [-1, -1], [1, -1]]  # 设置数据中心点坐标
    X, labels_true = make_blobs(  # 创建聚类数据 X 和真实标签
        n_samples=150, centers=centers, cluster_std=0.4, random_state=0
    )
    X = csr_container(X) if csr_container is not None else X  # 如果有 CSR 容器，将 X 转换为 CSR 格式

    X = X.astype(global_dtype, copy=False)  # 将 X 转换为全局数据类型

    # 使用 OPTICS 计算与 DBSCAN 提取的 0.3 epsilon 的聚类
    op = OPTICS(
        min_samples=min_samples, cluster_method="dbscan", eps=eps, metric=metric
    ).fit(X)

    # 计算 DBSCAN 标签
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)

    # 计算争议矩阵
    contingency = contingency_matrix(db.labels_, op.labels_)
    agree = min(
        np.sum(np.max(contingency, axis=0)), np.sum(np.max(contingency, axis=1))
    )
    disagree = X.shape[0] - agree

    percent_mismatch = np.round((disagree - 1) / X.shape[0], 2)

    # 验证标签不匹配率不超过 5%
    assert percent_mismatch <= 0.05


# 测试 min_samples 边界情况
def test_min_samples_edge_case(global_dtype):
    C1 = [[0, 0], [0, 0.1], [0, -0.1]]  # 设置数据 C1 的坐标
    # 定义一个包含三个点的簇 C2
    C2 = [[10, 10], [10, 9], [10, 11]]
    # 定义一个包含三个点的簇 C3
    C3 = [[100, 100], [100, 96], [100, 106]]
    # 将三个簇 C1、C2、C3 垂直堆叠，转换为指定的数据类型并在原地修改
    X = np.vstack((C1, C2, C3)).astype(global_dtype, copy=False)

    # 定义预期的标签数组，依次为 0、0、0、1、1、1、2、2、2
    expected_labels = np.r_[[0] * 3, [1] * 3, [2] * 3]
    # 使用 OPTICS 算法拟合数据 X，设置最小样本数为 3，最大半径为 7，聚类方法为 xi，xi 值为 0.04
    clust = OPTICS(min_samples=3, max_eps=7, cluster_method="xi", xi=0.04).fit(X)
    # 断言聚类后的标签与预期标签相等
    assert_array_equal(clust.labels_, expected_labels)

    # 定义另一组预期的标签数组，依次为 0、0、0、1、1、1、-1、-1、-1
    expected_labels = np.r_[[0] * 3, [1] * 3, [-1] * 3]
    # 使用 OPTICS 算法拟合数据 X，设置最小样本数为 3，最大半径为 3，聚类方法为 xi，xi 值为 0.04
    clust = OPTICS(min_samples=3, max_eps=3, cluster_method="xi", xi=0.04).fit(X)
    # 断言聚类后的标签与预期标签相等
    assert_array_equal(clust.labels_, expected_labels)

    # 定义另一组预期的标签数组，全为 -1，表示所有点未被分类到任何簇
    expected_labels = np.r_[[-1] * 9]
    # 使用 OPTICS 算法拟合数据 X，设置最小样本数为 4，最大半径为 3，聚类方法为 xi，xi 值为 0.04
    # 预期触发 UserWarning，匹配消息中包含 "All reachability values"
    with pytest.warns(UserWarning, match="All reachability values"):
        clust = OPTICS(min_samples=4, max_eps=3, cluster_method="xi", xi=0.04).fit(X)
        # 断言聚类后的标签与预期标签相等
        assert_array_equal(clust.labels_, expected_labels)
# 使用 pytest 的 parametrize 装饰器，对 min_cluster_size 参数进行多组测试
@pytest.mark.parametrize("min_cluster_size", range(2, X.shape[0] // 10, 23))
def test_min_cluster_size(min_cluster_size, global_dtype):
    # 将数据集 X 每隔一个样本取一个，转换为指定的全局数据类型，以提高速度
    redX = X[::2].astype(global_dtype, copy=False)  # reduce for speed
    
    # 使用指定的参数创建 OPTICS 聚类对象，并对减少后的数据 redX 进行拟合
    clust = OPTICS(min_samples=9, min_cluster_size=min_cluster_size).fit(redX)
    
    # 统计聚类结果中每个簇的样本数
    cluster_sizes = np.bincount(clust.labels_[clust.labels_ != -1])
    
    # 如果存在簇，则验证最小簇大小约束
    if cluster_sizes.size:
        assert min(cluster_sizes) >= min_cluster_size
    
    # 当 min_cluster_size 是一个分数时，验证行为是否与整数时相同
    clust_frac = OPTICS(
        min_samples=9,
        min_cluster_size=min_cluster_size / redX.shape[0],
    )
    clust_frac.fit(redX)
    
    # 验证两种参数形式下的聚类标签是否一致
    assert_array_equal(clust.labels_, clust_frac.labels_)


# 使用 pytest 的 parametrize 装饰器，对 csr_container 参数进行测试
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_min_cluster_size_invalid2(csr_container):
    # 测试当 min_cluster_size 超出数据长度时是否会抛出 ValueError 异常
    clust = OPTICS(min_cluster_size=len(X) + 1)
    with pytest.raises(ValueError, match="must be no greater than the "):
        clust.fit(X)

    # 测试当 min_cluster_size 超出数据长度且指定了距离度量时是否会抛出 ValueError 异常
    clust = OPTICS(min_cluster_size=len(X) + 1, metric="euclidean")
    with pytest.raises(ValueError, match="must be no greater than the "):
        clust.fit(csr_container(X))


# 测试 OPTICS 算法的处理顺序
def test_processing_order():
    # 确保在选择下一个点时，考虑所有未处理的点，而不仅仅是直接邻居
    Y = [[0], [10], [-10], [25]]

    # 使用指定的参数创建 OPTICS 聚类对象，并对数据 Y 进行拟合
    clust = OPTICS(min_samples=3, max_eps=15).fit(Y)
    
    # 验证计算得到的可达距离（reachability）与预期一致
    assert_array_equal(clust.reachability_, [np.inf, 10, 10, 15])
    
    # 验证计算得到的核心点距离（core distances）与预期一致
    assert_array_equal(clust.core_distances_, [10, 15, np.inf, np.inf])
    
    # 验证数据点的处理顺序与预期一致
    assert_array_equal(clust.ordering_, [0, 1, 2, 3])


# 与 ELKI 的结果进行比较
def test_compare_to_ELKI():
    # 预期值是使用 ELKI 0.7.5 计算得到的，使用命令行工具生成的固定编号的 CSV 数据
    # java -jar elki.jar cli -dbc.in csv -dbc.filter FixedDBIDsFilter
    #   -algorithm clustering.optics.OPTICSHeap -optics.minpts 5
    # 其中 FixedDBIDsFilter 提供了从 0 开始的索引编号。
    # 定义包含浮点数的列表 r1，用于存储一组特定的数值
    r1 = [
        np.inf,  # 正无穷大
        1.0574896366427478,
        0.7587934993548423,
        0.7290174038973836,
        0.7290174038973836,
        0.7290174038973836,
        0.6861627576116127,
        0.7587934993548423,
        0.9280118450166668,
        1.1748022534146194,
        3.3355455741292257,
        0.49618389254482587,
        0.2552805046961355,
        0.2552805046961355,
        0.24944622248445714,
        0.24944622248445714,
        0.24944622248445714,
        0.2552805046961355,
        0.2552805046961355,
        0.3086779122185853,
        4.163024452756142,
        1.623152630340929,
        0.45315840475822655,
        0.25468325192031926,
        0.2254004358159971,
        0.18765711877083036,
        0.1821471333893275,
        0.1821471333893275,
        0.18765711877083036,
        0.18765711877083036,
        0.2240202988740153,
        1.154337614548715,
        1.342604473837069,
        1.323308536402633,
        0.8607514948648837,
        0.27219111215810565,
        0.13260875220533205,
        0.13260875220533205,
        0.09890587675958984,
        0.09890587675958984,
        0.13548790801634494,
        0.1575483940837384,
        0.17515137170530226,
        0.17575920159442388,
        0.27219111215810565,
        0.6101447895405373,
        1.3189208094864302,
        1.323308536402633,
        2.2509184159764577,
        2.4517810628594527,
        3.675977064404973,
        3.8264795626020365,
        2.9130735341510614,
        2.9130735341510614,
        2.9130735341510614,
        2.9130735341510614,
        2.8459300127258036,
        2.8459300127258036,
        2.8459300127258036,
        3.0321982337972537,
    ]

    # 定义包含整数的列表 o1，用于存储一组特定的整数
    o1 = [
        0,
        3,
        6,
        4,
        7,
        8,
        2,
        9,
        5,
        1,
        31,
        30,
        32,
        34,
        33,
        38,
        39,
        35,
        37,
        36,
        44,
        21,
        23,
        24,
        22,
        25,
        27,
        29,
        26,
        28,
        20,
        40,
        45,
        46,
        10,
        15,
        11,
        13,
        17,
        19,
        18,
        12,
        16,
        14,
        47,
        49,
        43,
        48,
        42,
        41,
        53,
        57,
        51,
        52,
        56,
        59,
        54,
        55,
        58,
        50,
    ]
    p1 = [
        -1,  # 初始化 p1 数组，包含一系列整数值
        0,   # 索引 1
        3,   # 索引 2
        6,   # 索引 3-6
        6,
        6,
        8,   # 索引 7
        3,   # 索引 8
        7,   # 索引 9
        5,   # 索引 10
        1,   # 索引 11
        31,  # 索引 12
        30,  # 索引 13-15
        30,
        34,  # 索引 16-18
        34,
        34,
        32,  # 索引 19-21
        32,
        37,  # 索引 22-23
        36,
        44,  # 索引 24
        21,  # 索引 25-28
        23,
        24,
        22,  # 索引 29-32
        25,
        25,
        22,
        22,
        22,
        21,  # 索引 33-36
        40,  # 索引 37
        45,  # 索引 38-39
        46,
        10,  # 索引 40-42
        15,
        15,
        13,
        13,
        15,
        11,  # 索引 43-47
        19,
        15,
        10,
        47,
        12,
        45,
        14,
        43,
        42,  # 索引 48-52
        53,
        57,  # 索引 53-56
        57,
        57,
        57,
        59,  # 索引 57-59
        59,
        59,
        58,  # 索引 60
    ]

    # 使用 OPTICS 算法对数据集 X 进行聚类分析
    clust1 = OPTICS(min_samples=5).fit(X)

    # 断言验证聚类结果的顺序是否与给定数组 o1 相等
    assert_array_equal(clust1.ordering_, np.array(o1))

    # 断言验证聚类结果中每个点的前趋节点是否与给定数组 p1 相等
    assert_array_equal(clust1.predecessor_[clust1.ordering_], np.array(p1))

    # 断言验证聚类结果中每个点的可达距离是否与给定数组 r1 相近
    assert_allclose(clust1.reachability_[clust1.ordering_], np.array(r1))

    # 遍历聚类结果的顺序数组，确保可达距离与核心距离的一致性
    for i in clust1.ordering_[1:]:
        assert clust1.reachability_[i] >= clust1.core_distances_[clust1.predecessor_[i]]

    # 预期的可达距离数组，使用 ELKI 0.7.5 计算得出
    r2 = [
        np.inf,  # 索引 0
        np.inf,  # 索引 1
        np.inf,  # 索引 2
        np.inf,  # 索引 3
        np.inf,  # 索引 4
        np.inf,  # 索引 5
        np.inf,  # 索引 6
        np.inf,  # 索引 7
        np.inf,  # 索引 8
        np.inf,  # 索引 9
        np.inf,  # 索引 10
        0.27219111215810565,  # 索引 11
        0.13260875220533205,  # 索引 12-13
        0.13260875220533205,
        0.09890587675958984,  # 索引 14-15
        0.09890587675958984,
        0.13548790801634494,  # 索引 16
        0.1575483940837384,   # 索引 17
        0.17515137170530226,  # 索引 18
        0.17575920159442388,  # 索引 19
        0.27219111215810565,  # 索引 20
        0.4928068613197889,   # 索引 21
        np.inf,  # 索引 22
        0.2666183922512113,   # 索引 23
        0.18765711877083036,  # 索引 24-25
        0.1821471333893275,   # 索引 26-27
        0.1821471333893275,
        0.1821471333893275,
        0.18715928772277457,  # 索引 28
        0.18765711877083036,  # 索引 29-30
        0.18765711877083036,
        0.25468325192031926,  # 索引 31
        np.inf,  # 索引 32
        0.2552805046961355,   # 索引 33-34
        0.2552805046961355,
        0.24944622248445714,  # 索引 35-37
        0.24944622248445714,
        0.24944622248445714,
        0.2552805046961355,   # 索引 38-39
        0.2552805046961355,
        0.3086779122185853,   # 索引 40
        0.34466409325984865,  # 索引 41
        np.inf,  # 索引 42
        np.inf,  # 索引 43
        np.inf,  # 索引 44
        np.inf,  # 索引 45
        np.inf,  # 索引 46
        np.inf,  # 索引 47
        np.inf,  # 索引 48
        np.inf,  # 索引 49
        np.inf,  # 索引 50
        np.inf,  # 索引 51
        np.inf,  # 索引 52
        np.inf,  # 索引 53
        np.inf,  # 索引 54
        np.inf,  # 索引 55
        np.inf,  # 索引 56
        np.inf,  # 索引 57
        np.inf,  # 索引 58
        np.inf,  # 索引 59
        np.inf,  # 索引 60
    ]
    o2 = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        15,
        11,
        13,
        17,
        19,
        18,
        12,
        16,
        14,
        47,
        46,
        20,
        22,
        25,
        23,
        27,
        29,
        24,
        26,
        28,
        21,
        30,
        32,
        34,
        33,
        38,
        39,
        35,
        37,
        36,
        31,
        40,
        41,
        42,
        43,
        44,
        45,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
    ]
    p2 = [
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        10,
        15,
        15,
        13,
        13,
        15,
        11,
        19,
        15,
        10,
        47,
        -1,
        20,
        22,
        25,
        25,
        25,
        25,
        22,
        22,
        23,
        -1,
        30,
        30,
        34,
        34,
        34,
        32,
        32,
        37,
        38,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
    ]
    clust2 = OPTICS(min_samples=5, max_eps=0.5).fit(X)
    # 使用OPTICS算法拟合数据，生成聚类结果对象clust2

    assert_array_equal(clust2.ordering_, np.array(o2))
    # 断言clust2的聚类顺序与给定的顺序数组o2相等，即验证聚类顺序是否正确

    assert_array_equal(clust2.predecessor_[clust2.ordering_], np.array(p2))
    # 断言clust2在给定的聚类顺序下的前驱数组与给定的前驱数组p2相等，验证前驱关系是否正确

    assert_allclose(clust2.reachability_[clust2.ordering_], np.array(r2))
    # 断言clust2在给定的聚类顺序下的可达距离数组与给定的可达距离数组r2在数值上全部接近，验证可达距离是否正确

    index = np.where(clust1.core_distances_ <= 0.5)[0]
    # 找出clust1中核心距离小于等于0.5的数据点的索引

    assert_allclose(clust1.core_distances_[index], clust2.core_distances_[index])
    # 断言clust1和clust2在相同索引下的核心距离在数值上全部接近，验证核心距离是否正确
# 测试用例：测试简单的 DBSCAN 案例。不包括密度不同的簇。
def test_extract_dbscan(global_dtype):
    # 使用种子为0的随机数生成器创建一个新的随机数生成器实例
    rng = np.random.RandomState(0)
    # 每个簇中的点数
    n_points_per_cluster = 20
    # 生成第一个簇的数据，以及对其加入一些随机噪声
    C1 = [-5, -2] + 0.2 * rng.randn(n_points_per_cluster, 2)
    # 生成第二个簇的数据，以及对其加入一些随机噪声
    C2 = [4, -1] + 0.2 * rng.randn(n_points_per_cluster, 2)
    # 生成第三个簇的数据，以及对其加入一些随机噪声
    C3 = [1, 2] + 0.2 * rng.randn(n_points_per_cluster, 2)
    # 生成第四个簇的数据，以及对其加入一些随机噪声
    C4 = [-2, 3] + 0.2 * rng.randn(n_points_per_cluster, 2)
    # 将所有簇的数据堆叠在一起，并转换为指定的全局数据类型
    X = np.vstack((C1, C2, C3, C4)).astype(global_dtype, copy=False)

    # 使用 OPTICS 算法执行聚类，使用 DBSCAN 方法，设置 eps 为 0.5
    clust = OPTICS(cluster_method="dbscan", eps=0.5).fit(X)
    # 断言确保聚类标签的唯一值是已排序的 [0, 1, 2, 3]
    assert_array_equal(np.sort(np.unique(clust.labels_)), [0, 1, 2, 3])


@pytest.mark.parametrize("csr_container", [None] + CSR_CONTAINERS)
def test_precomputed_dists(global_dtype, csr_container):
    # 从 X 中每隔一个元素选择一个子集 redX，并转换为指定的全局数据类型
    redX = X[::2].astype(global_dtype, copy=False)
    # 计算 redX 中数据点之间的欧几里得距离
    dists = pairwise_distances(redX, metric="euclidean")
    # 如果 csr_container 不为 None，则将距离矩阵转换为指定容器类型
    dists = csr_container(dists) if csr_container is not None else dists
    # 忽略效率警告，并使用预先计算的距离矩阵执行 OPTICS 算法，算法选择 "brute"，距离度量选择 "precomputed"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EfficiencyWarning)
        clust1 = OPTICS(min_samples=10, algorithm="brute", metric="precomputed").fit(
            dists
        )
    # 对 redX 使用常规的欧几里得距离执行 OPTICS 算法
    clust2 = OPTICS(min_samples=10, algorithm="brute", metric="euclidean").fit(redX)

    # 断言确保 clust1 的 reachability_ 与 clust2 的 reachability_ 全部接近
    assert_allclose(clust1.reachability_, clust2.reachability_)
    # 断言确保 clust1 的标签与 clust2 的标签完全相等
    assert_array_equal(clust1.labels_, clust2.labels_)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_optics_input_not_modified_precomputed_sparse_nodiag(csr_container):
    """检查我们不会对预先计算的稀疏矩阵进行就地修改。
    非回归测试，用于检查：
    https://github.com/scikit-learn/scikit-learn/issues/27508
    """
    # 创建一个随机 6x6 的数组 X
    X = np.random.RandomState(0).rand(6, 6)
    # 在对角线上添加零，这些零将在创建稀疏矩阵时隐式存在
    np.fill_diagonal(X, 0)
    # 将 X 转换为 csr_container 类型的稀疏矩阵
    X = csr_container(X)
    # 断言确保稀疏矩阵的每一行和列都不相等
    assert all(row != col for row, col in zip(*X.nonzero()))
    # 创建 X 的一个副本 X_copy
    X_copy = X.copy()
    # 使用 OPTICS 算法执行距离度量为 "precomputed" 的聚类
    OPTICS(metric="precomputed").fit(X)
    # 确保我们没有就地修改 X，即使已经创建了显式的零值
    assert X.nnz == X_copy.nnz
    assert_array_equal(X.toarray(), X_copy.toarray())


def test_optics_predecessor_correction_ordering():
    """检查使用前驱者校正的聚类是否按预期工作。

    在以下示例中，前驱者校正未正常工作，因为没有使用正确的索引。

    这个非回归测试检查重新排序数据是否会改变结果。

    非回归测试，用于检查：
    https://github.com/scikit-learn/scikit-learn/issues/26324
    """
    # 创建一个一维数组 X_1
    X_1 = np.array([1, 2, 3, 1, 8, 8, 7, 100]).reshape(-1, 1)
    # 根据指定的重新排序索引重新排列 X_1，得到数组 X_2
    reorder = [0, 1, 2, 4, 5, 6, 7, 3]
    X_2 = X_1[reorder]

    # 使用 OPTICS 算法执行最小样本数为3，距离度量为欧几里得距离的聚类
    optics_1 = OPTICS(min_samples=3, metric="euclidean").fit(X_1)
    optics_2 = OPTICS(min_samples=3, metric="euclidean").fit(X_2)
    # 使用断言检查两个 OPTICS 算法生成的标签数组是否相等
    assert_array_equal(optics_1.labels_[reorder], optics_2.labels_)
```