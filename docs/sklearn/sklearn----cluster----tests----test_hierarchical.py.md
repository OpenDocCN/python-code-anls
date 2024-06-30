# `D:\src\scipysrc\scikit-learn\sklearn\cluster\tests\test_hierarchical.py`

```
"""
Several basic tests for hierarchical clustering procedures

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
# 导入所需的库和模块
import itertools
import shutil
from functools import partial
from tempfile import mkdtemp

import numpy as np
import pytest
from scipy.cluster import hierarchy
from scipy.sparse.csgraph import connected_components

# 导入需要测试的聚类算法和函数
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration, ward_tree
from sklearn.cluster._agglomerative import (
    _TREE_BUILDERS,
    _fix_connectivity,
    _hc_cut,
    linkage_tree,
)
from sklearn.cluster._hierarchical_fast import (
    average_merge,
    max_merge,
    mst_linkage_core,
)
from sklearn.datasets import make_circles, make_moons
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.metrics import DistanceMetric
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import (
    PAIRED_DISTANCES,
    cosine_distances,
    manhattan_distances,
    pairwise_distances,
)
from sklearn.metrics.tests.test_dist_metrics import METRICS_DEFAULT_PARAMS
from sklearn.neighbors import kneighbors_graph
from sklearn.utils._fast_dict import IntFloatDict
from sklearn.utils._testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    create_memmap_backed_data,
    ignore_warnings,
)
from sklearn.utils.fixes import LIL_CONTAINERS


def test_linkage_misc():
    # Misc tests on linkage

    # 创建一个随机数生成器
    rng = np.random.RandomState(42)
    # 创建一个随机的5x5数组
    X = rng.normal(size=(5, 5))

    # 测试链接时引发异常
    with pytest.raises(ValueError):
        linkage_tree(X, linkage="foo")

    # 测试链接时引发异常，使用一个全1的连接性矩阵
    with pytest.raises(ValueError):
        linkage_tree(X, connectivity=np.ones((4, 4)))

    # 对FeatureAgglomeration进行基本测试
    FeatureAgglomeration().fit(X)

    # 测试在预计算距离矩阵上进行层次聚类
    dis = cosine_distances(X)
    res = linkage_tree(dis, affinity="precomputed")
    assert_array_equal(res[0], linkage_tree(X, affinity="cosine")[0])

    # 测试在预计算距离矩阵上进行层次聚类，使用曼哈顿距离
    res = linkage_tree(X, affinity=manhattan_distances)
    assert_array_equal(res[0], linkage_tree(X, affinity="manhattan")[0])


def test_structured_linkage_tree():
    # Check that we obtain the correct solution for structured linkage trees.

    # 创建一个随机数生成器
    rng = np.random.RandomState(0)
    # 创建一个具有结构的掩码
    mask = np.ones([10, 10], dtype=bool)
    # 避免只有'True'条目的掩码
    mask[4:7, 4:7] = 0
    # 创建一个随机的50x100数组
    X = rng.randn(50, 100)
    # 使用网格创建图的连接性矩阵
    connectivity = grid_to_graph(*mask.shape)
    # 遍历_TREE_BUILDERS字典中的每个树构建器函数
    for tree_builder in _TREE_BUILDERS.values():
        # 使用当前树构建器函数构建树，并返回相关结果
        children, n_components, n_leaves, parent = tree_builder(
            X.T, connectivity=connectivity
        )
        # 计算预期的树节点数
        n_nodes = 2 * X.shape[1] - 1
        # 断言：树的子节点数与叶子节点数之和应等于总节点数
        assert len(children) + n_leaves == n_nodes
        # 检查ward_tree在使用错误形状的连接矩阵时是否引发ValueError异常
        with pytest.raises(ValueError):
            tree_builder(X.T, connectivity=np.ones((4, 4)))
        # 检查在没有样本的情况下拟合是否引发错误
        with pytest.raises(ValueError):
            tree_builder(X.T[:0], connectivity=connectivity)
def test_unstructured_linkage_tree():
    # 检查对于非结构化连接树能否获得正确的解决方案。
    rng = np.random.RandomState(0)
    X = rng.randn(50, 100)
    for this_X in (X, X[0]):
        # 为了引发警告并测试警告代码，指定一个固定的聚类数
        with ignore_warnings():
            with pytest.warns(UserWarning):
                children, n_nodes, n_leaves, parent = ward_tree(this_X.T, n_clusters=10)
        # 计算预期的节点数
        n_nodes = 2 * X.shape[1] - 1
        assert len(children) + n_leaves == n_nodes

    for tree_builder in _TREE_BUILDERS.values():
        for this_X in (X, X[0]):
            with ignore_warnings():
                with pytest.warns(UserWarning):
                    children, n_nodes, n_leaves, parent = tree_builder(
                        this_X.T, n_clusters=10
                    )
            # 计算预期的节点数
            n_nodes = 2 * X.shape[1] - 1
            assert len(children) + n_leaves == n_nodes


def test_height_linkage_tree():
    # 检查连接树结果的高度是否排序。
    rng = np.random.RandomState(0)
    mask = np.ones([10, 10], dtype=bool)
    X = rng.randn(50, 100)
    connectivity = grid_to_graph(*mask.shape)
    for linkage_func in _TREE_BUILDERS.values():
        children, n_nodes, n_leaves, parent = linkage_func(
            X.T, connectivity=connectivity
        )
        # 计算预期的节点数
        n_nodes = 2 * X.shape[1] - 1
        assert len(children) + n_leaves == n_nodes


def test_zero_cosine_linkage_tree():
    # 检查当 X 中存在零向量并使用 'cosine' 亲和度时是否会产生错误。
    X = np.array([[0, 1], [0, 0]])
    msg = "Cosine affinity cannot be used when X contains zero vectors"
    with pytest.raises(ValueError, match=msg):
        linkage_tree(X, affinity="cosine")


@pytest.mark.parametrize("n_clusters, distance_threshold", [(None, 0.5), (10, None)])
@pytest.mark.parametrize("compute_distances", [True, False])
@pytest.mark.parametrize("linkage", ["ward", "complete", "average", "single"])
def test_agglomerative_clustering_distances(
    n_clusters, compute_distances, distance_threshold, linkage
):
    # 检查当 `compute_distances` 为 True 或 `distance_threshold` 被给定时，
    # 拟合的模型是否有 `distances_` 属性。
    rng = np.random.RandomState(0)
    mask = np.ones([10, 10], dtype=bool)
    n_samples = 100
    X = rng.randn(n_samples, 50)
    connectivity = grid_to_graph(*mask.shape)

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        connectivity=connectivity,
        linkage=linkage,
        distance_threshold=distance_threshold,
        compute_distances=compute_distances,
    )
    clustering.fit(X)
    if compute_distances or (distance_threshold is not None):
        assert hasattr(clustering, "distances_")
        n_children = clustering.children_.shape[0]
        n_nodes = n_children + 1
        assert clustering.distances_.shape == (n_nodes - 1,)
    else:
        # 如果不满足前面的条件（即 clustering 对象没有 distances_ 属性）
        # 断言：确保 clustering 对象没有 distances_ 属性，如果有，则会触发 AssertionError
        assert not hasattr(clustering, "distances_")
@pytest.mark.parametrize("lil_container", LIL_CONTAINERS)
# 使用pytest的@parametrize装饰器，允许以参数化方式运行此测试函数，参数为LIL_CONTAINERS中的每个值

def test_agglomerative_clustering(global_random_seed, lil_container):
    # 测试聚类算法的准确性，确保使用凝聚聚类获得正确的聚类数目

    rng = np.random.RandomState(global_random_seed)
    # 使用全局随机种子创建NumPy随机状态对象rng

    mask = np.ones([10, 10], dtype=bool)
    # 创建一个10x10的布尔类型全为True的掩码数组mask

    n_samples = 100
    # 定义样本数目n_samples为100

    X = rng.randn(n_samples, 50)
    # 从rng中生成一个形状为(n_samples, 50)的标准正态分布随机数组X作为样本数据

    connectivity = grid_to_graph(*mask.shape)
    # 根据mask的形状生成连接性矩阵connectivity，用于聚类算法中的连接性参数

    for linkage in ("ward", "complete", "average", "single"):
        # 针对不同的链接方式(linkage)，依次执行以下操作：

        clustering = AgglomerativeClustering(
            n_clusters=10, connectivity=connectivity, linkage=linkage
        )
        # 创建凝聚聚类对象clustering，设定聚类数目为10，连接性参数为connectivity，链接方式为linkage

        clustering.fit(X)
        # 使用样本数据X进行聚类

        # test caching
        try:
            tempdir = mkdtemp()
            # 创建一个临时目录tempdir

            clustering = AgglomerativeClustering(
                n_clusters=10,
                connectivity=connectivity,
                memory=tempdir,
                linkage=linkage,
            )
            # 使用临时目录作为缓存，创建新的凝聚聚类对象clustering

            clustering.fit(X)
            # 使用样本数据X进行聚类

            labels = clustering.labels_
            # 获取聚类标签

            assert np.size(np.unique(labels)) == 10
            # 断言聚类标签的唯一值数量为10，确保聚类结果正确

        finally:
            shutil.rmtree(tempdir)
            # 最终删除临时目录tempdir

        # Turn caching off now
        clustering = AgglomerativeClustering(
            n_clusters=10, connectivity=connectivity, linkage=linkage
        )
        # 创建新的凝聚聚类对象clustering，关闭缓存功能

        # Check that we obtain the same solution with early-stopping of the
        # tree building
        clustering.compute_full_tree = False
        # 设置不完整树构建标志为False，启用早停止树构建策略

        clustering.fit(X)
        # 使用样本数据X进行聚类

        assert_almost_equal(normalized_mutual_info_score(clustering.labels_, labels), 1)
        # 断言使用早停止树构建策略后的聚类结果与之前相同，通过归一化互信息得分检验

        clustering.connectivity = None
        # 将连接性参数设置为None

        clustering.fit(X)
        # 使用样本数据X进行聚类

        assert np.size(np.unique(clustering.labels_)) == 10
        # 断言聚类标签的唯一值数量为10，确保聚类结果正确

        # Check that we raise a TypeError on dense matrices
        clustering = AgglomerativeClustering(
            n_clusters=10,
            connectivity=lil_container(connectivity.toarray()[:10, :10]),
            linkage=linkage,
        )
        # 创建新的凝聚聚类对象clustering，使用lil_container转换后的密集连接性矩阵切片作为连接性参数

        with pytest.raises(ValueError):
            clustering.fit(X)
        # 使用样本数据X进行聚类，并断言会抛出值错误异常(ValueError)

    # Test that using ward with another metric than euclidean raises an
    # exception
    clustering = AgglomerativeClustering(
        n_clusters=10,
        connectivity=connectivity.toarray(),
        metric="manhattan",
        linkage="ward",
    )
    # 创建新的凝聚聚类对象clustering，使用曼哈顿距离作为度量标准，链接方式为ward

    with pytest.raises(ValueError):
        clustering.fit(X)
    # 使用样本数据X进行聚类，并断言会抛出值错误异常(ValueError)

    # Test using another metric than euclidean works with linkage complete
    for metric in PAIRED_DISTANCES.keys():
        # 遍历PAIRED_DISTANCES字典中的度量标准metric

        # Compare our (structured) implementation to scipy
        clustering = AgglomerativeClustering(
            n_clusters=10,
            connectivity=np.ones((n_samples, n_samples)),
            metric=metric,
            linkage="complete",
        )
        # 创建新的凝聚聚类对象clustering，使用全为1的连接性矩阵，度量标准为metric，链接方式为complete

        clustering.fit(X)
        # 使用样本数据X进行聚类

        clustering2 = AgglomerativeClustering(
            n_clusters=10, connectivity=None, metric=metric, linkage="complete"
        )
        # 创建第二个凝聚聚类对象clustering2，不使用连接性矩阵，度量标准为metric，链接方式为complete

        clustering2.fit(X)
        # 使用样本数据X进行聚类

        assert_almost_equal(
            normalized_mutual_info_score(clustering2.labels_, clustering.labels_), 1
        )
        # 断言自定义实现和scipy的实现在聚类结果上的归一化互信息得分接近1
    # 使用距离矩阵作为相似性度量（affinity = 'precomputed'）测试是否与使用连接约束的结果相同

    # 使用AgglomerativeClustering聚类算法，指定10个聚类簇，使用给定的连接约束
    clustering = AgglomerativeClustering(
        n_clusters=10, connectivity=connectivity, linkage="complete"
    )

    # 对数据集X进行聚类
    clustering.fit(X)

    # 计算数据集X中样本点两两之间的距离矩阵
    X_dist = pairwise_distances(X)

    # 使用AgglomerativeClustering聚类算法，指定10个聚类簇，使用预先计算的距离矩阵作为度量(metric="precomputed")，使用给定的连接约束和完全连接(linkage="complete")
    clustering2 = AgglomerativeClustering(
        n_clusters=10,
        connectivity=connectivity,
        metric="precomputed",
        linkage="complete",
    )

    # 对基于预先计算的距离矩阵X_dist进行聚类
    clustering2.fit(X_dist)

    # 断言两个聚类结果的标签是否相等
    assert_array_equal(clustering.labels_, clustering2.labels_)
# 测试 AgglomerativeClustering 是否能在内存映射数据集上正常工作。
# 这是针对问题 #19875 的非回归测试。
def test_agglomerative_clustering_memory_mapped():
    rng = np.random.RandomState(0)
    # 创建一个内存映射数据集 Xmm，使用随机数生成器 rng 的随机数据填充
    Xmm = create_memmap_backed_data(rng.randn(50, 100))
    # 使用 AgglomerativeClustering 对象，使用欧氏距离和单链接方法进行聚类
    AgglomerativeClustering(metric="euclidean", linkage="single").fit(Xmm)


# 测试 Ward 聚合
def test_ward_agglomeration(global_random_seed):
    # 确保在简单情况下获得正确的解决方案
    rng = np.random.RandomState(global_random_seed)
    # 创建一个形状为 [10, 10] 的布尔掩码数组
    mask = np.ones([10, 10], dtype=bool)
    X = rng.randn(50, 100)
    # 从掩码创建连接性图
    connectivity = grid_to_graph(*mask.shape)
    # 创建 FeatureAgglomeration 对象，指定聚类数为 5，使用连接性信息
    agglo = FeatureAgglomeration(n_clusters=5, connectivity=connectivity)
    agglo.fit(X)
    # 断言聚类标签的唯一值数量为 5
    assert np.size(np.unique(agglo.labels_)) == 5

    # 对 X 进行降维处理
    X_red = agglo.transform(X)
    # 断言降维后的特征数量为 5
    assert X_red.shape[1] == 5
    # 通过逆变换还原降维后的数据
    X_full = agglo.inverse_transform(X_red)
    # 断言还原后数据第一行的唯一值数量为 5
    assert np.unique(X_full[0]).size == 5
    # 断言逆变换后的数据与原始降维数据 X_red 很接近
    assert_array_almost_equal(agglo.transform(X_full), X_red)

    # 检查使用空样本拟合是否引发 ValueError 异常
    with pytest.raises(ValueError):
        agglo.fit(X[:0])


# 测试单链接聚类
def test_single_linkage_clustering():
    # 检查在两个典型案例中是否获得正确的结果

    # 创建月亮形数据集 moons 和相应标签 moon_labels
    moons, moon_labels = make_moons(noise=0.05, random_state=42)
    # 使用 AgglomerativeClustering 对象，聚类数为 2，链接方法为单链接
    clustering = AgglomerativeClustering(n_clusters=2, linkage="single")
    clustering.fit(moons)
    # 断言聚类标签与真实标签的归一化互信息得分接近于 1
    assert_almost_equal(
        normalized_mutual_info_score(clustering.labels_, moon_labels), 1
    )

    # 创建圆形数据集 circles 和相应标签 circle_labels
    circles, circle_labels = make_circles(factor=0.5, noise=0.025, random_state=42)
    # 使用 AgglomerativeClustering 对象，聚类数为 2，链接方法为单链接
    clustering = AgglomerativeClustering(n_clusters=2, linkage="single")
    clustering.fit(circles)
    # 断言聚类标签与真实标签的归一化互信息得分接近于 1
    assert_almost_equal(
        normalized_mutual_info_score(clustering.labels_, circle_labels), 1
    )


# 辅助函数，用于与 scipy 进行比较
def assess_same_labelling(cut1, cut2):
    co_clust = []
    for cut in [cut1, cut2]:
        n = len(cut)
        k = cut.max() + 1
        ecut = np.zeros((n, k))
        ecut[np.arange(n), cut] = 1
        co_clust.append(np.dot(ecut, ecut.T))
    # 断言两个聚类结果的一致性
    assert (co_clust[0] == co_clust[1]).all()


# 测试稀疏数据的 scikit-learn 和 scipy 的比较
def test_sparse_scikit_vs_scipy(global_random_seed):
    # 使用全连接性（即无结构性）检测 scikit-learn 的链接方法与 scipy 的差异
    n, p, k = 10, 5, 3
    rng = np.random.RandomState(global_random_seed)

    # 这里没有使用 lil_matrix，只是为了检查非稀疏矩阵的处理能力
    connectivity = np.ones((n, n))
    # 遍历所有树构建器的链接方法
    for linkage in _TREE_BUILDERS.keys():
        # 对每种链接方法循环5次
        for i in range(5):
            # 生成一个(n, p)大小的随机数组，每个元素从标准正态分布中取样乘以0.1
            X = 0.1 * rng.normal(size=(n, p))
            # 每行的每个元素减去该行索引的四倍
            X -= 4.0 * np.arange(n)[:, np.newaxis]
            # 每行减去其均值，保持维度一致
            X -= X.mean(axis=1)[:, np.newaxis]

            # 使用当前数据 X 应用层次聚类算法，采用给定的链接方法
            out = hierarchy.linkage(X, method=linkage)

            # 从输出中提取子节点信息，并将其转换为整数类型
            children_ = out[:, :2].astype(int, copy=False)
            # 使用指定的链接方法构建树，并返回子节点、叶子节点数及连接信息
            children, _, n_leaves, _ = _TREE_BUILDERS[linkage](
                X, connectivity=connectivity
            )

            # 对每行子节点按列排序，以确保一致性
            children.sort(axis=1)
            # 断言排序后的子节点与直接从层次聚类输出的子节点相等
            assert_array_equal(
                children,
                children_,
                "linkage tree differs from scipy impl for linkage: " + linkage,
            )

            # 使用 _hc_cut 函数进行聚类结果的切割，得到预期的 k 个簇
            cut = _hc_cut(k, children, n_leaves)
            # 使用 _hc_cut 函数对原始输出进行相同操作，验证结果一致性
            cut_ = _hc_cut(k, children_, n_leaves)
            # 检查两种切割结果是否相同
            assess_same_labelling(cut, cut_)

    # 测试 _hc_cut 函数对异常情况的处理，预期会引发 ValueError
    with pytest.raises(ValueError):
        _hc_cut(n_leaves + 1, children, n_leaves)


这段代码主要是用于测试层次聚类算法中的树构建和切割过程，包括随机数据生成、层次聚类的应用、子节点的排序和比较、以及对异常情况的处理进行验证。
# 确保我们的自定义 mst_linkage_core 和 scipy 的内置实现产生相同的结果
def test_vector_scikit_single_vs_scipy_single(global_random_seed):
    # 定义样本数、特征数和聚类数
    n_samples, n_features, n_clusters = 10, 5, 3
    # 使用全局随机种子创建随机数生成器
    rng = np.random.RandomState(global_random_seed)
    # 生成服从正态分布的随机数据，并对每个样本减去一个线性下降的偏移
    X = 0.1 * rng.normal(size=(n_samples, n_features))
    X -= 4.0 * np.arange(n_samples)[:, np.newaxis]
    X -= X.mean(axis=1)[:, np.newaxis]

    # 使用 scipy 的单链接聚类方法计算聚类树
    out = hierarchy.linkage(X, method="single")
    # 提取出聚类树中的子节点信息
    children_scipy = out[:, :2].astype(int)

    # 调用内部函数获取自定义聚类树的子节点信息
    children, _, n_leaves, _ = _TREE_BUILDERS["single"](X)

    # 对每行子节点信息进行排序，确保结果一致性
    children.sort(axis=1)
    # 断言自定义聚类树与 scipy 实现的结果一致
    assert_array_equal(
        children,
        children_scipy,
        "linkage tree differs from scipy impl for single linkage.",
    )

    # 使用自定义函数计算聚类树的剪枝结果
    cut = _hc_cut(n_clusters, children, n_leaves)
    cut_scipy = _hc_cut(n_clusters, children_scipy, n_leaves)
    # 对比自定义方法与 scipy 的剪枝结果是否一致
    assess_same_labelling(cut, cut_scipy)


@pytest.mark.parametrize("metric_param_grid", METRICS_DEFAULT_PARAMS)
def test_mst_linkage_core_memory_mapped(metric_param_grid):
    """
    MST-LINKAGE-CORE 算法必须能够在内存映射的数据集上运行。

    针对问题 #19875 的非回归测试。
    """
    # 使用固定随机种子创建随机数生成器
    rng = np.random.RandomState(seed=1)
    # 生成正态分布的随机数据集
    X = rng.normal(size=(20, 4))
    # 创建内存映射的数据集
    Xmm = create_memmap_backed_data(X)
    # 获取测试参数中的距离度量和参数网格
    metric, param_grid = metric_param_grid
    keys = param_grid.keys()
    # 对参数网格中的每个参数组合进行迭代测试
    for vals in itertools.product(*param_grid.values()):
        kwargs = dict(zip(keys, vals))
        # 根据参数创建距离度量对象
        distance_metric = DistanceMetric.get_metric(metric, **kwargs)
        # 分别使用原始数据和内存映射数据运行 MST-LINKAGE-CORE 算法
        mst = mst_linkage_core(X, distance_metric)
        mst_mm = mst_linkage_core(Xmm, distance_metric)
        # 断言原始数据和内存映射数据上算法结果的一致性
        np.testing.assert_equal(mst, mst_mm)


def test_identical_points():
    """
    确保在使用稀疏连接矩阵进行 MST 聚类时，处理相同点的情况。

    """
    # 创建包含相同点的数据集和真实标签
    X = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2]])
    true_labels = np.array([0, 0, 1, 1, 2, 2])
    # 创建 K 近邻图的稀疏连接矩阵
    connectivity = kneighbors_graph(X, n_neighbors=3, include_self=False)
    connectivity = 0.5 * (connectivity + connectivity.T)
    # 修正连接性矩阵和计算数据集的连通组件数
    connectivity, n_components = _fix_connectivity(X, connectivity, "euclidean")

    # 对每种链接方法进行迭代
    for linkage in ("single", "average", "average", "ward"):
        # 使用凝聚聚类方法创建聚类器
        clustering = AgglomerativeClustering(
            n_clusters=3, linkage=linkage, connectivity=connectivity
        )
        # 对数据集进行聚类
        clustering.fit(X)

        # 断言聚类结果与真实标签的归一化互信息分数接近于 1
        assert_almost_equal(
            normalized_mutual_info_score(clustering.labels_, true_labels), 1
        )


def test_connectivity_propagation():
    """
    检查在 ward 树中连接性如何在合并过程中正确传播。

    """
    # 创建一个 NumPy 数组 X，包含了一组二维数据点
    X = np.array(
        [
            (0.014, 0.120),
            (0.014, 0.099),
            (0.014, 0.097),
            (0.017, 0.153),
            (0.017, 0.153),
            (0.018, 0.153),
            (0.018, 0.153),
            (0.018, 0.153),
            (0.018, 0.153),
            (0.018, 0.153),
            (0.018, 0.153),
            (0.018, 0.153),
            (0.018, 0.152),
            (0.018, 0.149),
            (0.018, 0.144),
        ]
    )
    # 使用 kneighbors_graph 函数计算 X 的连接图，设置每个点的邻居数量为 10，不包括自身
    connectivity = kneighbors_graph(X, 10, include_self=False)
    # 创建 AgglomerativeClustering 聚类对象，设置聚类数为 4，使用连接图作为连接性约束，使用 ward 连接方式
    ward = AgglomerativeClustering(
        n_clusters=4, connectivity=connectivity, linkage="ward"
    )
    # 如果修改未正确传播，fit 方法会因 IndexError 而崩溃
    # 对数据 X 进行聚类
    ward.fit(X)
# 检查在结构化和非结构化版本的ward_tree中子节点的顺序是否相同。

# 在五个随机数据集上进行测试
n, p = 10, 5
rng = np.random.RandomState(global_random_seed)

# 创建全连接性矩阵
connectivity = np.ones((n, n))
for i in range(5):
    # 生成服从正态分布的随机数据，并对数据进行修正
    X = 0.1 * rng.normal(size=(n, p))
    X -= 4.0 * np.arange(n)[:, np.newaxis]
    X -= X.mean(axis=1)[:, np.newaxis]

    # 调用ward_tree函数，分别获取结构化和非结构化版本的输出
    out_unstructured = ward_tree(X)
    out_structured = ward_tree(X, connectivity=connectivity)

    # 断言两个输出的第一个元素（子节点）是否相等
    assert_array_equal(out_unstructured[0], out_structured[0])


# 测试链接和ward树的return_distance选项

# 当return_distance设置为True时，验证结构化和非结构化聚类是否得到相同的输出
n, p = 10, 5
rng = np.random.RandomState(global_random_seed)

connectivity = np.ones((n, n))
for i in range(5):
    X = 0.1 * rng.normal(size=(n, p))
    X -= 4.0 * np.arange(n)[:, np.newaxis]
    X -= X.mean(axis=1)[:, np.newaxis]

    # 调用ward_tree函数，返回包括距离的结果
    out_unstructured = ward_tree(X, return_distance=True)
    out_structured = ward_tree(X, connectivity=connectivity, return_distance=True)

    # 获取子节点
    children_unstructured = out_unstructured[0]
    children_structured = out_structured[0]

    # 检查是否得到相同的聚类结果
    assert_array_equal(children_unstructured, children_structured)

    # 获取距离信息
    dist_unstructured = out_unstructured[-1]
    dist_structured = out_structured[-1]

    # 断言距离信息是否几乎相等
    assert_array_almost_equal(dist_unstructured, dist_structured)

    # 对于average、complete、single三种链接方式分别进行测试
    for linkage in ["average", "complete", "single"]:
        # 调用linkage_tree函数，返回包括距离的结果
        structured_items = linkage_tree(
            X, connectivity=connectivity, linkage=linkage, return_distance=True
        )[-1]
        unstructured_items = linkage_tree(X, linkage=linkage, return_distance=True)[
            -1
        ]
        structured_dist = structured_items[-1]
        unstructured_dist = unstructured_items[-1]
        structured_children = structured_items[0]
        unstructured_children = unstructured_items[0]
        
        # 断言结构化和非结构化的距离信息几乎相等
        assert_array_almost_equal(structured_dist, unstructured_dist)
        # 断言结构化和非结构化的子节点信息几乎相等
        assert_array_almost_equal(structured_children, unstructured_children)
    # 创建包含 ward 方法聚类结果的 linkage 矩阵
    linkage_X_ward = np.array(
        [
            [3.0, 4.0, 0.36265956, 2.0],  # 合并簇3和簇4，距离为0.36265956，共包含2个样本
            [1.0, 5.0, 1.77045373, 2.0],  # 合并簇1和簇5，距离为1.77045373，共包含2个样本
            [0.0, 2.0, 2.55760419, 2.0],  # 合并簇0和簇2，距离为2.55760419，共包含2个样本
            [6.0, 8.0, 9.10208346, 4.0],  # 合并簇6和簇8，距离为9.10208346，共包含4个样本
            [7.0, 9.0, 24.7784379, 6.0],  # 合并簇7和簇9，距离为24.7784379，共包含6个样本
        ]
    )

    # 创建包含 complete 方法聚类结果的 linkage 矩阵
    linkage_X_complete = np.array(
        [
            [3.0, 4.0, 0.36265956, 2.0],  # 合并簇3和簇4，距离为0.36265956，共包含2个样本
            [1.0, 5.0, 1.77045373, 2.0],  # 合并簇1和簇5，距离为1.77045373，共包含2个样本
            [0.0, 2.0, 2.55760419, 2.0],  # 合并簇0和簇2，距离为2.55760419，共包含2个样本
            [6.0, 8.0, 6.96742194, 4.0],  # 合并簇6和簇8，距离为6.96742194，共包含4个样本
            [7.0, 9.0, 18.77445997, 6.0],  # 合并簇7和簇9，距离为18.77445997，共包含6个样本
        ]
    )

    # 创建包含 average 方法聚类结果的 linkage 矩阵
    linkage_X_average = np.array(
        [
            [3.0, 4.0, 0.36265956, 2.0],  # 合并簇3和簇4，距离为0.36265956，共包含2个样本
            [1.0, 5.0, 1.77045373, 2.0],  # 合并簇1和簇5，距离为1.77045373，共包含2个样本
            [0.0, 2.0, 2.55760419, 2.0],  # 合并簇0和簇2，距离为2.55760419，共包含2个样本
            [6.0, 8.0, 6.55832839, 4.0],  # 合并簇6和簇8，距离为6.55832839，共包含4个样本
            [7.0, 9.0, 15.44089605, 6.0],  # 合并簇7和簇9，距离为15.44089605，共包含6个样本
        ]
    )

    # 计算样本矩阵 X 的行数和列数
    n_samples, n_features = np.shape(X)

    # 创建一个全为1的连接矩阵，用于 ward 和 linkage 方法
    connectivity_X = np.ones((n_samples, n_samples))

    # 使用 ward 方法生成无结构连接的聚类结果
    out_X_unstructured = ward_tree(X, return_distance=True)

    # 使用 ward 方法生成有结构连接的聚类结果，传入自定义的连接矩阵
    out_X_structured = ward_tree(X, connectivity=connectivity_X, return_distance=True)

    # 检查标签是否一致
    assert_array_equal(linkage_X_ward[:, :2], out_X_unstructured[0])
    assert_array_equal(linkage_X_ward[:, :2], out_X_structured[0])

    # 检查距离是否正确
    assert_array_almost_equal(linkage_X_ward[:, 2], out_X_unstructured[4])
    assert_array_almost_equal(linkage_X_ward[:, 2], out_X_structured[4])

    # 定义要测试的 linkage 方法选项
    linkage_options = ["complete", "average", "single"]

    # 定义每种 linkage 方法对应的真实聚类结果
    X_linkage_truth = [linkage_X_complete, linkage_X_average]

    # 遍历每种 linkage 方法及其对应的真实聚类结果
    for linkage, X_truth in zip(linkage_options, X_linkage_truth):
        # 使用 linkage 方法生成无结构连接的聚类结果
        out_X_unstructured = linkage_tree(X, return_distance=True, linkage=linkage)

        # 使用 linkage 方法生成有结构连接的聚类结果，传入自定义的连接矩阵
        out_X_structured = linkage_tree(
            X, connectivity=connectivity_X, linkage=linkage, return_distance=True
        )

        # 检查标签是否一致
        assert_array_equal(X_truth[:, :2], out_X_unstructured[0])
        assert_array_equal(X_truth[:, :2], out_X_structured[0])

        # 检查距离是否正确
        assert_array_almost_equal(X_truth[:, 2], out_X_unstructured[4])
        assert_array_almost_equal(X_truth[:, 2], out_X_structured[4])
def test_connectivity_fixing_non_lil():
    # 检查非LIL格式的连接性修复是否不会回退到旧的Bug，
    # 如果提供一个具有多个组件的不可分配连接性。
    
    # 创建虚拟数据
    x = np.array([[0, 0], [1, 1]])
    
    # 创建一个具有多个组件的掩码，以强制连接性修复
    m = np.array([[True, False], [False, True]])
    
    # 将网格转换为图形，使用给定的网格大小和掩码
    c = grid_to_graph(n_x=2, n_y=2, mask=m)
    
    # 使用Ward方法进行聚类，使用创建的连接性
    w = AgglomerativeClustering(connectivity=c, linkage="ward")
    
    # 断言会触发UserWarning警告
    with pytest.warns(UserWarning):
        w.fit(x)


def test_int_float_dict():
    rng = np.random.RandomState(0)
    keys = np.unique(rng.randint(100, size=10).astype(np.intp, copy=False))
    values = rng.rand(len(keys))

    # 创建一个IntFloatDict实例，使用给定的键和值
    d = IntFloatDict(keys, values)
    
    # 对每个键值对进行断言，验证IntFloatDict的正确性
    for key, value in zip(keys, values):
        assert d[key] == value

    other_keys = np.arange(50, dtype=np.intp)[::2]
    other_values = np.full(50, 0.5)[::2]
    other = IntFloatDict(other_keys, other_values)
    
    # 使用max_merge函数合并两个IntFloatDict对象，进行完整性测试
    max_merge(d, other, mask=np.ones(100, dtype=np.intp), n_a=1, n_b=1)
    
    # 使用average_merge函数合并两个IntFloatDict对象，进行完整性测试
    average_merge(d, other, mask=np.ones(100, dtype=np.intp), n_a=1, n_b=1)


def test_connectivity_callable():
    rng = np.random.RandomState(0)
    X = rng.rand(20, 5)
    
    # 使用k-neighbors图创建连接性矩阵，不包括自身
    connectivity = kneighbors_graph(X, 3, include_self=False)
    
    # 创建AgglomerativeClustering实例，使用上述连接性
    aglc1 = AgglomerativeClustering(connectivity=connectivity)
    
    # 使用partial函数包装k-neighbors图函数，创建AgglomerativeClustering实例
    aglc2 = AgglomerativeClustering(
        connectivity=partial(kneighbors_graph, n_neighbors=3, include_self=False)
    )
    
    # 分别对两个实例进行拟合
    aglc1.fit(X)
    aglc2.fit(X)
    
    # 断言两个实例的标签数组相等
    assert_array_equal(aglc1.labels_, aglc2.labels_)


def test_connectivity_ignores_diagonal():
    rng = np.random.RandomState(0)
    X = rng.rand(20, 5)
    
    # 创建不包括自身的k-neighbors图连接性矩阵
    connectivity = kneighbors_graph(X, 3, include_self=False)
    
    # 创建包括自身的k-neighbors图连接性矩阵
    connectivity_include_self = kneighbors_graph(X, 3, include_self=True)
    
    # 创建两个AgglomerativeClustering实例，分别使用上述连接性
    aglc1 = AgglomerativeClustering(connectivity=connectivity)
    aglc2 = AgglomerativeClustering(connectivity=connectivity_include_self)
    
    # 分别对两个实例进行拟合
    aglc1.fit(X)
    aglc2.fit(X)
    
    # 断言两个实例的标签数组相等
    assert_array_equal(aglc1.labels_, aglc2.labels_)


def test_compute_full_tree():
    # 测试当n_clusters较小时，是否计算完整的树
    rng = np.random.RandomState(0)
    X = rng.randn(10, 2)
    
    # 创建k-neighbors图连接性矩阵，不包括自身
    connectivity = kneighbors_graph(X, 5, include_self=False)
    
    # 当n_clusters较小时，应构建完整的树，即合并次数应为n_samples - 1
    agc = AgglomerativeClustering(n_clusters=2, connectivity=connectivity)
    agc.fit(X)
    n_samples = X.shape[0]
    n_nodes = agc.children_.shape[0]
    assert n_nodes == n_samples - 1
    
    # 当n_clusters较大时，应停止合并，直到达到n_clusters
    n_clusters = 101
    X = rng.randn(200, 2)
    connectivity = kneighbors_graph(X, 10, include_self=False)
    agc = AgglomerativeClustering(n_clusters=n_clusters, connectivity=connectivity)
    agc.fit(X)
    n_samples = X.shape[0]
    n_nodes = agc.children_.shape[0]
    assert n_nodes == n_samples - n_clusters
def test_n_components():
    # 测试 linkage、average 和 ward 树返回的 n_components
    rng = np.random.RandomState(0)
    X = rng.rand(5, 5)

    # 连通性矩阵，包含五个组件
    connectivity = np.eye(5)

    # 对于每个 linkage 函数，验证其返回的第二个元素为 5
    for linkage_func in _TREE_BUILDERS.values():
        assert ignore_warnings(linkage_func)(X, connectivity=connectivity)[1] == 5


def test_affinity_passed_to_fix_connectivity():
    # 测试确保 affinity 参数有效传递给 pairwise 函数

    size = 2
    rng = np.random.RandomState(0)
    X = rng.randn(size, size)
    mask = np.array([True, False, False, True])

    # 生成连接性矩阵
    connectivity = grid_to_graph(n_x=size, n_y=size, mask=mask, return_as=np.ndarray)

    # 定义一个假的 Affinity 类
    class FakeAffinity:
        def __init__(self):
            self.counter = 0

        def increment(self, *args, **kwargs):
            self.counter += 1
            return self.counter

    fa = FakeAffinity()

    # 调用 linkage_tree 函数，验证 affinity 参数是否被传递
    linkage_tree(X, connectivity=connectivity, affinity=fa.increment)

    # 验证 fa.counter 是否增加到 3
    assert fa.counter == 3


@pytest.mark.parametrize("linkage", ["ward", "complete", "average"])
def test_agglomerative_clustering_with_distance_threshold(linkage, global_random_seed):
    # 检查在设置 distance_threshold 的情况下，使用聚合聚类获得正确的聚类数量

    rng = np.random.RandomState(global_random_seed)
    mask = np.ones([10, 10], dtype=bool)
    n_samples = 100
    X = rng.randn(n_samples, 50)
    connectivity = grid_to_graph(*mask.shape)

    # 设置距离阈值为 10 进行测试
    distance_threshold = 10
    for conn in [None, connectivity]:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            connectivity=conn,
            linkage=linkage,
        )
        clustering.fit(X)
        clusters_produced = clustering.labels_

        # 获取聚合聚类后的簇数
        num_clusters_produced = len(np.unique(clustering.labels_))

        # 检查聚类数是否与 linkage 树中距离超过阈值的点匹配
        tree_builder = _TREE_BUILDERS[linkage]
        children, n_components, n_leaves, parent, distances = tree_builder(
            X, connectivity=conn, n_clusters=None, return_distance=True
        )
        num_clusters_at_threshold = (
            np.count_nonzero(distances >= distance_threshold) + 1
        )

        # 测试产生的簇数是否正确
        assert num_clusters_at_threshold == num_clusters_produced

        # 测试产生的簇是否正确
        clusters_at_threshold = _hc_cut(
            n_clusters=num_clusters_produced, children=children, n_leaves=n_leaves
        )
        assert np.array_equiv(clusters_produced, clusters_at_threshold)


def test_small_distance_threshold(global_random_seed):
    rng = np.random.RandomState(global_random_seed)
    n_samples = 10
    X = rng.randint(-300, 300, size=(n_samples, 3))
    # 假设所有数据都应该在自己的簇中，给定
    # 使用层次聚类算法（AgglomerativeClustering），以单链接（single linkage）方式，对数据集 X 进行聚类
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1.0, linkage="single"
    ).fit(X)
    # 检查所有成对样本之间的距离是否都大于0.1
    all_distances = pairwise_distances(X, metric="minkowski", p=2)
    # 将对角线上的距离设为无穷大，以排除样本与自身的距离
    np.fill_diagonal(all_distances, np.inf)
    # 断言所有的成对距离都大于0.1
    assert np.all(all_distances > 0.1)
    # 断言聚类后的簇数与样本数相同
    assert clustering.n_clusters_ == n_samples
# 测试聚类算法在给定距离阈值下的距离计算
def test_cluster_distances_with_distance_threshold(global_random_seed):
    # 使用全局随机种子创建随机数生成器
    rng = np.random.RandomState(global_random_seed)
    # 设定样本数量
    n_samples = 100
    # 生成随机样本数据
    X = rng.randint(-10, 10, size=(n_samples, 3))
    # 设定距离阈值
    distance_threshold = 4
    # 使用凝聚聚类算法，根据给定距离阈值和链接方式构建聚类模型
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=distance_threshold, linkage="single"
    ).fit(X)
    # 获取聚类标签
    labels = clustering.labels_
    # 计算样本之间的距离矩阵
    D = pairwise_distances(X, metric="minkowski", p=2)
    # 将距离矩阵对角线元素设置为无穷大，以避免在最小值计算中使用对角线元素
    np.fill_diagonal(D, np.inf)
    # 遍历每个唯一的聚类标签
    for label in np.unique(labels):
        # 获取当前聚类中的样本掩码
        in_cluster_mask = labels == label
        # 计算当前聚类内部样本间的最大距离
        max_in_cluster_distance = (
            D[in_cluster_mask][:, in_cluster_mask].min(axis=0).max()
        )
        # 计算当前聚类与其他聚类之间的最小距离
        min_out_cluster_distance = (
            D[in_cluster_mask][:, ~in_cluster_mask].min(axis=0).min()
        )
        # 对于具有多于一个样本点的聚类，验证最大内部距离小于设定的距离阈值
        if in_cluster_mask.sum() > 1:
            assert max_in_cluster_distance < distance_threshold
        # 验证最小外部距离大于等于设定的距离阈值
        assert min_out_cluster_distance >= distance_threshold


# 使用参数化测试框架对不同的链接方式和距离阈值进行边界情况测试
@pytest.mark.parametrize("linkage", ["ward", "complete", "average"])
@pytest.mark.parametrize(
    ("threshold", "y_true"), [(0.5, [1, 0]), (1.0, [1, 0]), (1.5, [0, 0])]
)
def test_agglomerative_clustering_with_distance_threshold_edge_case(
    linkage, threshold, y_true
):
    # 测试距离阈值与距离匹配的边界情况
    X = [[0], [1]]
    # 使用凝聚聚类算法，根据给定的距离阈值、链接方式构建聚类模型
    clusterer = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold, linkage=linkage
    )
    # 进行聚类并预测类别
    y_pred = clusterer.fit_predict(X)
    # 验证预测结果与真实结果的调整兰德指数为1
    assert adjusted_rand_score(y_true, y_pred) == 1


# 测试凝聚聚类算法在参数无效时是否能够抛出异常
def test_dist_threshold_invalid_parameters():
    X = [[0], [1]]
    # 验证在未指定距离阈值时是否会抛出值错误异常
    with pytest.raises(ValueError, match="Exactly one of "):
        AgglomerativeClustering(n_clusters=None, distance_threshold=None).fit(X)

    # 验证在指定了聚类数目时是否会抛出值错误异常
    with pytest.raises(ValueError, match="Exactly one of "):
        AgglomerativeClustering(n_clusters=2, distance_threshold=1).fit(X)

    X = [[0], [1]]
    # 验证在未设置完整树计算标志时是否会抛出值错误异常
    with pytest.raises(ValueError, match="compute_full_tree must be True if"):
        AgglomerativeClustering(
            n_clusters=None, distance_threshold=1, compute_full_tree=False
        ).fit(X)


# 测试在预先计算距离矩阵时，是否会检测并抛出非方阵错误
def test_invalid_shape_precomputed_dist_matrix():
    # 检查当距离矩阵设置为预先计算时，是否会在传递非方阵时抛出错误
    rng = np.random.RandomState(0)
    X = rng.rand(5, 3)
    with pytest.raises(
        ValueError,
        match=r"Distance matrix should be square, got matrix of shape \(5, 3\)",
    ):
        AgglomerativeClustering(metric="precomputed", linkage="complete").fit(X)


# 测试在预先计算连接性和距离矩阵时，多个连接组件的情况
def test_precomputed_connectivity_metric_with_2_connected_components():
    """Check that connecting components works when connectivity and
    affinity are both precomputed and the number of connected components is
    greater than 1. Non-regression test for #16151.
    """
    # 创建一个 5x5 的 NumPy 数组，表示一个连接性矩阵
    connectivity_matrix = np.array(
        [
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]
    )
    # 确保 connectivity_matrix 有两个连接的组件
    assert connected_components(connectivity_matrix)[0] == 2

    # 使用随机数生成器创建一个 5x10 的 NumPy 数组 X
    rng = np.random.RandomState(0)
    X = rng.randn(5, 10)

    # 计算 X 中样本点之间的距离，存储在 X_dist 中
    X_dist = pairwise_distances(X)

    # 使用预计算的距离矩阵进行层次聚类，连接性矩阵和连接方式为完全连接
    clusterer_precomputed = AgglomerativeClustering(
        metric="precomputed", connectivity=connectivity_matrix, linkage="complete"
    )
    
    # 准备用于警告测试的消息
    msg = "Completing it to avoid stopping the tree early"
    
    # 在运行过程中捕获特定警告，并与消息 msg 匹配
    with pytest.warns(UserWarning, match=msg):
        clusterer_precomputed.fit(X_dist)

    # 使用连接性矩阵进行层次聚类，连接方式为完全连接
    clusterer = AgglomerativeClustering(
        connectivity=connectivity_matrix, linkage="complete"
    )
    
    # 在运行过程中捕获特定警告，并与消息 msg 匹配
    with pytest.warns(UserWarning, match=msg):
        clusterer.fit(X)

    # 断言聚类结果的标签相等
    assert_array_equal(clusterer.labels_, clusterer_precomputed.labels_)
    # 断言聚类结果的子节点相等
    assert_array_equal(clusterer.children_, clusterer_precomputed.children_)
# TODO(1.6): remove in 1.6
# 使用 pytest 框架的 parametrize 装饰器，为测试函数提供多个参数化的输入
@pytest.mark.parametrize(
    # 参数化的参数名为 "Agglomeration"
    "Agglomeration", [AgglomerativeClustering, FeatureAgglomeration]
)
# 定义一个测试函数，测试警告在指定条件下的行为
def test_deprecation_warning_metric_None(Agglomeration):
    # 创建一个示例数据集
    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    # 定义一个警告信息字符串，提醒 "`metric=None`" 的用法在 1.4 版本中已不推荐，并将在将来版本中移除
    warn_msg = "`metric=None` is deprecated in version 1.4 and will be removed"
    # 使用 pytest 的 warn 捕获语法，检查是否触发 FutureWarning，并匹配预期的警告信息
    with pytest.warns(FutureWarning, match=warn_msg):
        # 使用 Agglomeration 类型的算法对象，使用 metric=None 参数进行拟合
        Agglomeration(metric=None).fit(X)
```