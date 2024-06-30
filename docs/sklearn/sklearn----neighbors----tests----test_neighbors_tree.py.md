# `D:\src\scipysrc\scikit-learn\sklearn\neighbors\tests\test_neighbors_tree.py`

```
# SPDX-License-Identifier: BSD-3-Clause
# 导入必要的库和模块
import itertools  # 提供迭代工具的模块
import pickle  # 提供序列化和反序列化 Python 对象的功能

import numpy as np  # 科学计算库 NumPy
import pytest  # 测试框架 pytest
from numpy.testing import assert_allclose, assert_array_almost_equal  # NumPy 测试模块

from sklearn.metrics import DistanceMetric  # 距离度量类
from sklearn.neighbors._ball_tree import (  # Ball Tree 相关函数和类
    BallTree,
    kernel_norm,
)
from sklearn.neighbors._ball_tree import (  # Ball Tree 的堆操作类
    NeighborsHeap64 as NeighborsHeapBT,
)
from sklearn.neighbors._ball_tree import (  # Ball Tree 的节点堆排序函数
    nodeheap_sort as nodeheap_sort_bt,
)
from sklearn.neighbors._ball_tree import (  # Ball Tree 的并行排序函数
    simultaneous_sort as simultaneous_sort_bt,
)
from sklearn.neighbors._kd_tree import (  # KD Tree 相关函数和类
    KDTree,
)
from sklearn.neighbors._kd_tree import (  # KD Tree 的堆操作类
    NeighborsHeap64 as NeighborsHeapKDT,
)
from sklearn.neighbors._kd_tree import (  # KD Tree 的节点堆排序函数
    nodeheap_sort as nodeheap_sort_kdt,
)
from sklearn.neighbors._kd_tree import (  # KD Tree 的并行排序函数
    simultaneous_sort as simultaneous_sort_kdt,
)
from sklearn.utils import check_random_state  # 随机状态检查函数

# 设定随机数生成器
rng = np.random.RandomState(42)

# 生成 Mahalanobis 距离所需的协方差矩阵
V_mahalanobis = rng.rand(3, 3)
V_mahalanobis = np.dot(V_mahalanobis, V_mahalanobis.T)

# 定义维度常量
DIMENSION = 3

# 定义不同距离度量的参数字典
METRICS = {
    "euclidean": {},  # 欧氏距离
    "manhattan": {},  # 曼哈顿距离
    "minkowski": dict(p=3),  # Minkowski 距离（p=3）
    "chebyshev": {},  # 切比雪夫距离
    "seuclidean": dict(V=rng.random_sample(DIMENSION)),  # 标准化欧氏距离
    "mahalanobis": dict(V=V_mahalanobis),  # Mahalanobis 距离
}

# KD 树可以使用的距离度量列表
KD_TREE_METRICS = ["euclidean", "manhattan", "chebyshev", "minkowski"]

# Ball 树可以使用的距离度量列表（使用所有 METRICS 中定义的度量）
BALL_TREE_METRICS = list(METRICS)


# 定义计算距离的函数
def dist_func(x1, x2, p):
    return np.sum((x1 - x2) ** p) ** (1.0 / p)


# 计算核密度估计的慢速版本函数
def compute_kernel_slow(Y, X, kernel, h):
    # 计算点之间的距离
    d = np.sqrt(((Y[:, None, :] - X) ** 2).sum(-1))
    # 计算核函数的归一化常数
    norm = kernel_norm(h, X.shape[1], kernel)

    # 根据不同的核函数类型进行计算
    if kernel == "gaussian":
        return norm * np.exp(-0.5 * (d * d) / (h * h)).sum(-1)
    elif kernel == "tophat":
        return norm * (d < h).sum(-1)
    elif kernel == "epanechnikov":
        return norm * ((1.0 - (d * d) / (h * h)) * (d < h)).sum(-1)
    elif kernel == "exponential":
        return norm * (np.exp(-d / h)).sum(-1)
    elif kernel == "linear":
        return norm * ((1 - d / h) * (d < h)).sum(-1)
    elif kernel == "cosine":
        return norm * (np.cos(0.5 * np.pi * d / h) * (d < h)).sum(-1)
    else:
        raise ValueError("kernel not recognized")


# 暴力搜索最近邻的函数
def brute_force_neighbors(X, Y, k, metric, **kwargs):
    # 使用指定的距离度量类计算距离矩阵 D
    D = DistanceMetric.get_metric(metric, **kwargs).pairwise(Y, X)
    # 对距离矩阵 D 按照距离排序，找出每个 Y 中点最近的 k 个 X 中点的索引
    ind = np.argsort(D, axis=1)[:, :k]
    # 提取对应的距离
    dist = D[np.arange(Y.shape[0])[:, None], ind]
    return dist, ind


# 使用 pytest 的参数化装饰器定义多个测试参数组合进行测试
@pytest.mark.parametrize("Cls", [KDTree, BallTree])
@pytest.mark.parametrize(
    "kernel", ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]
)
@pytest.mark.parametrize("h", [0.01, 0.1, 1])
@pytest.mark.parametrize("rtol", [0, 1e-5])
@pytest.mark.parametrize("atol", [1e-6, 1e-2])
@pytest.mark.parametrize("breadth_first", [True, False])
# 核密度估计的测试函数
def test_kernel_density(
    Cls, kernel, h, rtol, atol, breadth_first, n_samples=100, n_features=3
):
    # 设置随机数生成器
    rng = check_random_state(1)
    # 生成随机样本数据 X 和 Y
    X = rng.random_sample((n_samples, n_features))
    Y = rng.random_sample((n_samples, n_features))
    # 使用给定的慢速方法计算核密度估计，将结果赋给 dens_true
    dens_true = compute_kernel_slow(Y, X, kernel, h)
    
    # 使用 Cls 类创建一个树结构对象 tree，设置叶子大小为 10
    tree = Cls(X, leaf_size=10)
    
    # 使用 tree 对象计算核密度估计，返回结果赋给 dens
    dens = tree.kernel_density(
        Y, h, atol=atol, rtol=rtol, kernel=kernel, breadth_first=breadth_first
    )
    
    # 断言检查 dens 与 dens_true 的近似程度，指定允许的绝对误差和相对误差
    assert_allclose(dens, dens_true, atol=atol, rtol=max(rtol, 1e-7))
@pytest.mark.parametrize("Cls", [KDTree, BallTree])
# 使用 pytest.mark.parametrize 装饰器，对 Cls 参数进行参数化测试，分别传入 KDTree 和 BallTree 作为参数
def test_neighbor_tree_query_radius(Cls, n_samples=100, n_features=10):
    # 设置随机数生成器，并生成 n_samples 行 n_features 列的随机数矩阵 X
    rng = check_random_state(0)
    X = 2 * rng.random_sample(size=(n_samples, n_features)) - 1
    # 创建一个全零的长度为 n_features 的查询点 query_pt
    query_pt = np.zeros(n_features, dtype=float)

    # 设置 eps 为 1e-15，用于处理舍入误差，避免测试失败
    eps = 1e-15
    # 使用 Cls 构造一个树结构，传入数据 X 和叶子大小 leaf_size=5
    tree = Cls(X, leaf_size=5)
    # 计算每个样本点到查询点的欧式距离的平方根，存储在 rad 中
    rad = np.sqrt(((X - query_pt) ** 2).sum(1))

    # 对于从 rad[0] 到 rad[-1] 等间距取 100 个值的范围内的每个半径 r
    for r in np.linspace(rad[0], rad[-1], 100):
        # 查询距离小于等于 r + eps 的点的索引，返回结果的第一个元素
        ind = tree.query_radius([query_pt], r + eps)[0]
        # 找到所有距离小于等于 r + eps 的索引 i
        i = np.where(rad <= r + eps)[0]

        # 对索引进行排序
        ind.sort()
        i.sort()

        # 断言 i 和 ind 的值几乎相等
        assert_array_almost_equal(i, ind)


@pytest.mark.parametrize("Cls", [KDTree, BallTree])
# 使用 pytest.mark.parametrize 装饰器，对 Cls 参数进行参数化测试，分别传入 KDTree 和 BallTree 作为参数
def test_neighbor_tree_query_radius_distance(Cls, n_samples=100, n_features=10):
    # 设置随机数生成器，并生成 n_samples 行 n_features 列的随机数矩阵 X
    rng = check_random_state(0)
    X = 2 * rng.random_sample(size=(n_samples, n_features)) - 1
    # 创建一个全零的长度为 n_features 的查询点 query_pt
    query_pt = np.zeros(n_features, dtype=float)

    # 设置 eps 为 1e-15，用于处理舍入误差，避免测试失败
    eps = 1e-15
    # 使用 Cls 构造一个树结构，传入数据 X 和叶子大小 leaf_size=5
    tree = Cls(X, leaf_size=5)
    # 计算每个样本点到查询点的欧式距离的平方根，存储在 rad 中
    rad = np.sqrt(((X - query_pt) ** 2).sum(1))

    # 对于从 rad[0] 到 rad[-1] 等间距取 100 个值的范围内的每个半径 r
    for r in np.linspace(rad[0], rad[-1], 100):
        # 查询距离小于等于 r + eps 的点的索引和距离，返回结果的第一个元素
        ind, dist = tree.query_radius([query_pt], r + eps, return_distance=True)

        # 取出第一个元素中的索引和距离
        ind = ind[0]
        dist = dist[0]

        # 计算查询点与 ind 所指示的点之间的欧式距离
        d = np.sqrt(((query_pt - X[ind]) ** 2).sum(1))

        # 断言 d 与 dist 几乎相等
        assert_array_almost_equal(d, dist)


@pytest.mark.parametrize("Cls", [KDTree, BallTree])
@pytest.mark.parametrize("dualtree", (True, False))
# 使用 pytest.mark.parametrize 装饰器，对 Cls 和 dualtree 参数进行参数化测试，分别传入 KDTree 和 BallTree 作为 Cls，True 和 False 作为 dualtree
def test_neighbor_tree_two_point(Cls, dualtree, n_samples=100, n_features=3):
    # 设置随机数生成器，并生成 n_samples 行 n_features 列的随机数矩阵 X 和 Y
    rng = check_random_state(0)
    X = rng.random_sample((n_samples, n_features))
    Y = rng.random_sample((n_samples, n_features))
    # 在 0 到 1 之间等间距取 10 个值的范围内的每个半径 r
    r = np.linspace(0, 1, 10)
    # 使用 Cls 构造一个树结构，传入数据 X 和叶子大小 leaf_size=10
    tree = Cls(X, leaf_size=10)

    # 计算 Y 中每对点与 X 中每个点之间的欧氏距离，并保存在 D 中
    D = DistanceMetric.get_metric("euclidean").pairwise(Y, X)
    # 计算每个 r 下真实的点对计数 counts_true
    counts_true = [(D <= ri).sum() for ri in r]

    # 计算树结构中每个 r 下的点对计数 counts
    counts = tree.two_point_correlation(Y, r=r, dualtree=dualtree)
    # 断言 counts 与 counts_true 几乎相等
    assert_array_almost_equal(counts, counts_true)


@pytest.mark.parametrize("NeighborsHeap", [NeighborsHeapBT, NeighborsHeapKDT])
# 使用 pytest.mark.parametrize 装饰器，对 NeighborsHeap 参数进行参数化测试，分别传入 NeighborsHeapBT 和 NeighborsHeapKDT 作为参数
def test_neighbors_heap(NeighborsHeap, n_pts=5, n_nbrs=10):
    # 创建一个 NeighborsHeap 对象，传入 n_pts 和 n_nbrs 作为参数
    heap = NeighborsHeap(n_pts, n_nbrs)
    # 设置随机数生成器，并生成长度为 2*n_nbrs 的浮点数随机数组 d_in，以及长度为 2*n_nbrs 的整数数组 i_in
    rng = check_random_state(0)

    for row in range(n_pts):
        d_in = rng.random_sample(2 * n_nbrs).astype(np.float64, copy=False)
        i_in = np.arange(2 * n_nbrs, dtype=np.intp)
        for d, i in zip(d_in, i_in):
            heap.push(row, d, i)

        # 对 d_in 和 i_in 按 d_in 排序
        ind = np.argsort(d_in)
        d_in = d_in[ind]
        i_in = i_in[ind]

        # 从堆中获取排序后的数组 d_heap 和 i_heap
        d_heap, i_heap = heap.get_arrays(sort=True)

        # 断言 d_heap[row] 的前 n_nbrs 个元素与 d_in 的前 n_nbrs 个元素几乎相等
        assert_array_almost_equal(d_in[:n_nbrs], d_heap[row])
        # 断言 i_heap[row] 的前 n_nbrs 个元素与 i_in 的前 n_nbrs 个元素几乎相等
        assert_array_almost_equal(i_in[:n_nbrs], i_heap[row])


@pytest.mark.parametrize("nodeheap_sort", [nodeheap_sort_bt, nodeheap_sort_kdt])
# 使用 pytest.mark.parametrize 装饰器，对 nodeheap_sort 参数进行参数化测试，分别传入 nodeheap_sort_bt 和 nodeheap_sort_kdt 作为参数
def test_node_heap(nodeheap_sort, n_nodes=50):
    # 设置随机数生成器，并生成长度为 n_nodes 的浮点数随机数组 vals
    rng = check_random_state(0)
    vals = rng.random_sample(n_nodes).astype(np.float64, copy=False)

    # 对 vals 中的元素进行排序，返回排序后的值和索引
    i1 = np.argsort(vals)
    # 使用 nodeheap_sort 函数对 vals 进行堆排序，返回排序后的值 vals2 和索引 i2
    vals2, i2 = nodeheap_sort(vals)

    # 断言 i1 与 i2 几乎相等
    assert_array_almost_equal(i1, i2)
    # 使用 NumPy 提供的函数检查两个数组的元素是否几乎相等，并引发 AssertionError 如果不是
    assert_array_almost_equal(vals[i1], vals2)
@pytest.mark.parametrize(
    "simultaneous_sort", [simultaneous_sort_bt, simultaneous_sort_kdt]
)
# 定义测试函数：测试同时排序功能，接受不同的排序函数作为参数
def test_simultaneous_sort(simultaneous_sort, n_rows=10, n_pts=201):
    # 使用随机数生成器创建随机状态
    rng = check_random_state(0)
    # 创建随机数组成的距离矩阵，浮点数类型
    dist = rng.random_sample((n_rows, n_pts)).astype(np.float64, copy=False)
    # 创建索引矩阵，每行为一组索引值，整数类型
    ind = (np.arange(n_pts) + np.zeros((n_rows, 1))).astype(np.intp, copy=False)

    # 复制距离矩阵和索引矩阵，备份用于后续对比
    dist2 = dist.copy()
    ind2 = ind.copy()

    # 使用给定的排序函数同时对距离矩阵和索引矩阵的各行进行排序
    simultaneous_sort(dist, ind)

    # 使用numpy库对距离矩阵和索引矩阵的各行进行排序
    i = np.argsort(dist2, axis=1)
    row_ind = np.arange(n_rows)[:, None]
    dist2 = dist2[row_ind, i]
    ind2 = ind2[row_ind, i]

    # 检查排序后的距离矩阵和索引矩阵是否几乎相等
    assert_array_almost_equal(dist, dist2)
    assert_array_almost_equal(ind, ind2)


@pytest.mark.parametrize("Cls", [KDTree, BallTree])
# 定义测试函数：测试高斯核密度估计（KDE）功能，接受KDTree和BallTree作为参数
def test_gaussian_kde(Cls, n_samples=1000):
    # 导入高斯核密度估计函数
    from scipy.stats import gaussian_kde

    # 使用随机数生成器创建随机状态
    rng = check_random_state(0)
    # 从正态分布中生成一组随机样本
    x_in = rng.normal(0, 1, n_samples)
    # 在指定区间内生成一组等间距的数据点
    x_out = np.linspace(-5, 5, 30)

    # 对于不同的带宽参数h，分别使用Cls构建树结构，以及使用scipy.stats中的高斯核密度估计函数进行比较
    for h in [0.01, 0.1, 1]:
        tree = Cls(x_in[:, None])
        gkde = gaussian_kde(x_in, bw_method=h / np.std(x_in))

        # 使用树结构计算指定数据点的核密度估计值，并归一化
        dens_tree = tree.kernel_density(x_out[:, None], h) / n_samples
        # 使用高斯核密度估计函数计算指定数据点的核密度估计值
        dens_gkde = gkde.evaluate(x_out)

        # 检查两种方法计算的核密度估计值是否几乎相等
        assert_array_almost_equal(dens_tree, dens_gkde, decimal=3)


@pytest.mark.parametrize(
    "Cls, metric",
    itertools.chain(
        [(KDTree, metric) for metric in KD_TREE_METRICS],
        [(BallTree, metric) for metric in BALL_TREE_METRICS],
    ),
)
@pytest.mark.parametrize("k", (1, 3, 5))
@pytest.mark.parametrize("dualtree", (True, False))
@pytest.mark.parametrize("breadth_first", (True, False))
# 定义测试函数：测试最近邻查询功能，接受不同的树结构、距离度量、邻居数量和查询选项作为参数
def test_nn_tree_query(Cls, metric, k, dualtree, breadth_first):
    # 使用随机数生成器创建随机状态
    rng = check_random_state(0)
    # 生成指定维度的随机样本数据集X和Y
    X = rng.random_sample((40, DIMENSION))
    Y = rng.random_sample((10, DIMENSION))

    # 根据指定的距离度量参数获取额外的参数设置
    kwargs = METRICS[metric]

    # 使用给定的树结构类型、距离度量和其他参数构建KDTree或BallTree
    kdt = Cls(X, leaf_size=1, metric=metric, **kwargs)
    # 使用树结构进行最近邻查询，返回距离和索引
    dist1, ind1 = kdt.query(Y, k, dualtree=dualtree, breadth_first=breadth_first)
    # 使用暴力法进行最近邻查询，返回距离和索引
    dist2, ind2 = brute_force_neighbors(X, Y, k, metric, **kwargs)

    # 不检查索引：如果有重复的距离，索引可能不匹配。距离不应该有这个问题。
    # 检查两种方法计算的距离是否几乎相等
    assert_array_almost_equal(dist1, dist2)


@pytest.mark.parametrize(
    "Cls, metric",
    [(KDTree, "euclidean"), (BallTree, "euclidean"), (BallTree, dist_func)],
)
@pytest.mark.parametrize("protocol", (0, 1, 2))
# 定义测试函数：测试树结构的序列化和反序列化功能，接受树结构类型、距离度量和序列化协议作为参数
def test_pickle(Cls, metric, protocol):
    # 使用随机数生成器创建随机状态
    rng = check_random_state(0)
    # 生成随机样本数据集X
    X = rng.random_sample((10, 3))

    # 如果距离度量是一个函数，则设置额外的参数
    if hasattr(metric, "__call__"):
        kwargs = {"p": 2}
    else:
        kwargs = {}

    # 使用给定的树结构类型、距离度量和其他参数构建KDTree或BallTree
    tree1 = Cls(X, leaf_size=1, metric=metric, **kwargs)

    # 查询树结构的索引和距离
    ind1, dist1 = tree1.query(X)

    # 序列化树结构对象，根据指定的协议
    s = pickle.dumps(tree1, protocol=protocol)
    # 反序列化树结构对象
    tree2 = pickle.loads(s)

    # 查询反序列化后的树结构对象的索引和距离
    ind2, dist2 = tree2.query(X)

    # 检查序列化和反序列化后的树结构对象的索引和距离是否几乎相等
    assert_array_almost_equal(ind1, ind2)
    assert_array_almost_equal(dist1, dist2)
    # 断言语句，用于检查变量 tree2 是否为 Cls 类的实例
    assert isinstance(tree2, Cls)
```