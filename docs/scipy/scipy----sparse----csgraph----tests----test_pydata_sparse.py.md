# `D:\src\scipysrc\scipy\scipy\sparse\csgraph\tests\test_pydata_sparse.py`

```
# 导入pytest模块，用于测试和标记
import pytest

# 导入numpy和scipy中的稀疏矩阵相关模块
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as spgraph
from scipy._lib import _pep440

# 导入用于测试的numpy断言函数
from numpy.testing import assert_equal

# 尝试导入pydata/sparse库，如果失败则设为None
try:
    import sparse
except Exception:
    sparse = None

# 标记，如果sparse为None，则跳过当前测试，给出相应理由
pytestmark = pytest.mark.skipif(sparse is None,
                                reason="pydata/sparse not installed")

# 提示消息，用于标记失败的测试用例
msg = "pydata/sparse (0.15.1) does not implement necessary operations"

# 定义参数化的sparse_cls，指定COO和DOK类型的稀疏矩阵，并标记DOK类型的测试为预期失败
sparse_params = (pytest.param("COO"),
                 pytest.param("DOK", marks=[pytest.mark.xfail(reason=msg)]))


# 检查sparse版本是否满足要求的fixture函数
def check_sparse_version(min_ver):
    if sparse is None:
        return pytest.mark.skip(reason="sparse is not installed")
    return pytest.mark.skipif(
        _pep440.parse(sparse.__version__) < _pep440.Version(min_ver),
        reason=f"sparse version >= {min_ver} required"
    )


# 参数化fixture，根据参数选择不同类型的稀疏矩阵类
@pytest.fixture(params=sparse_params)
def sparse_cls(request):
    return getattr(sparse, request.param)


# fixture，提供图数据用于测试，返回密集矩阵和对应的稀疏矩阵
@pytest.fixture
def graphs(sparse_cls):
    graph = [
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ]
    A_dense = np.array(graph)
    A_sparse = sparse_cls(A_dense)
    return A_dense, A_sparse


# 参数化测试函数，测试稀疏图算法的等效性
@pytest.mark.parametrize(
    "func",
    [
        spgraph.shortest_path,
        spgraph.dijkstra,
        spgraph.floyd_warshall,
        spgraph.bellman_ford,
        spgraph.johnson,
        spgraph.reverse_cuthill_mckee,
        spgraph.maximum_bipartite_matching,
        spgraph.structural_rank,
    ]
)
def test_csgraph_equiv(func, graphs):
    A_dense, A_sparse = graphs
    actual = func(A_sparse)
    desired = func(sp.csc_matrix(A_dense))
    assert_equal(actual, desired)


# 测试连通组件的函数
def test_connected_components(graphs):
    A_dense, A_sparse = graphs
    func = spgraph.connected_components

    actual_comp, actual_labels = func(A_sparse)
    desired_comp, desired_labels, = func(sp.csc_matrix(A_dense))

    assert actual_comp == desired_comp
    assert_equal(actual_labels, desired_labels)


# 测试拉普拉斯矩阵的函数
def test_laplacian(graphs):
    A_dense, A_sparse = graphs
    sparse_cls = type(A_sparse)
    func = spgraph.laplacian

    actual = func(A_sparse)
    desired = func(sp.csc_matrix(A_dense))

    assert isinstance(actual, sparse_cls)

    assert_equal(actual.todense(), desired.todense())


# 参数化测试函数，测试广度优先和深度优先搜索的函数
@pytest.mark.parametrize(
    "func", [spgraph.breadth_first_order, spgraph.depth_first_order]
)
def test_order_search(graphs, func):
    A_dense, A_sparse = graphs

    actual = func(A_sparse, 0)
    desired = func(sp.csc_matrix(A_dense), 0)

    assert_equal(actual, desired)


# 参数化测试函数，测试广度优先和深度优先搜索生成树的函数
@pytest.mark.parametrize(
    "func", [spgraph.breadth_first_tree, spgraph.depth_first_tree]
)
def test_tree_search(graphs, func):
    A_dense, A_sparse = graphs
    sparse_cls = type(A_sparse)

    actual = func(A_sparse, 0)
    desired = func(sp.csc_matrix(A_dense), 0)

    assert isinstance(actual, sparse_cls)

    assert_equal(actual.todense(), desired.todense())
# 测试最小生成树函数
def test_minimum_spanning_tree(graphs):
    # 解包参数 graphs 中的稠密图 A_dense 和稀疏图 A_sparse
    A_dense, A_sparse = graphs
    # 获取稀疏图 A_sparse 的类型
    sparse_cls = type(A_sparse)
    # 获取最小生成树函数
    func = spgraph.minimum_spanning_tree

    # 计算稀疏图 A_sparse 的最小生成树
    actual = func(A_sparse)
    # 计算稠密图 A_dense 的最小生成树
    desired = func(sp.csc_matrix(A_dense))

    # 断言 actual 的类型与 A_sparse 相同
    assert isinstance(actual, sparse_cls)
    # 断言 actual 的稠密表示与 desired 的稠密表示相同
    assert_equal(actual.todense(), desired.todense())


# 测试最大流函数
def test_maximum_flow(graphs):
    # 解包参数 graphs 中的稠密图 A_dense 和稀疏图 A_sparse
    A_dense, A_sparse = graphs
    # 获取稀疏图 A_sparse 的类型
    sparse_cls = type(A_sparse)
    # 获取最大流函数
    func = spgraph.maximum_flow

    # 计算稀疏图 A_sparse 的最大流，从节点 0 到节点 2
    actual = func(A_sparse, 0, 2)
    # 计算稠密图 A_dense 的最大流，从节点 0 到节点 2
    desired = func(sp.csr_matrix(A_dense), 0, 2)

    # 断言 actual 的流量值与 desired 的流量值相等
    assert actual.flow_value == desired.flow_value
    # 断言 actual 的流与 desired 的流稀疏表示相同
    assert isinstance(actual.flow, sparse_cls)
    assert_equal(actual.flow.todense(), desired.flow.todense())


# 测试最小权重全二分图匹配函数
def test_min_weight_full_bipartite_matching(graphs):
    # 解包参数 graphs 中的稠密图 A_dense 和稀疏图 A_sparse
    A_dense, A_sparse = graphs
    # 获取最小权重全二分图匹配函数
    func = spgraph.min_weight_full_bipartite_matching

    # 计算稀疏图 A_sparse 的部分子图的最小权重全二分图匹配
    actual = func(A_sparse[0:2, 1:3])
    # 计算稠密图 A_dense 的部分子图的最小权重全二分图匹配
    desired = func(sp.csc_matrix(A_dense)[0:2, 1:3])

    # 断言 actual 与 desired 相等
    assert_equal(actual, desired)


# 使用特定填充值测试路径算法函数
@check_sparse_version("0.15.4")
@pytest.mark.parametrize(
    "func",
    [
        spgraph.shortest_path,
        spgraph.dijkstra,
        spgraph.floyd_warshall,
        spgraph.bellman_ford,
        spgraph.johnson,
        spgraph.minimum_spanning_tree,
    ]
)
@pytest.mark.parametrize(
    "fill_value, comp_func",
    [(np.inf, np.isposinf), (np.nan, np.isnan)],
)
def test_nonzero_fill_value(graphs, func, fill_value, comp_func):
    # 解包参数 graphs 中的稠密图 A_dense 和稀疏图 A_sparse
    A_dense, A_sparse = graphs
    # 将 A_sparse 转换为浮点类型并设置填充值
    A_sparse = A_sparse.astype(float)
    A_sparse.fill_value = fill_value
    # 获取稀疏图 A_sparse 的类型
    sparse_cls = type(A_sparse)

    # 计算稀疏图 A_sparse 使用 func 算法的结果
    actual = func(A_sparse)
    # 计算稠密图 A_dense 使用 func 算法的结果
    desired = func(sp.csc_matrix(A_dense))

    # 如果 func 是最小生成树函数，执行以下断言
    if func == spgraph.minimum_spanning_tree:
        # 断言 actual 的类型与 A_sparse 相同
        assert isinstance(actual, sparse_cls)
        # 断言 actual 的填充值满足 comp_func 函数的条件
        assert comp_func(actual.fill_value)
        # 将 actual 转换为稠密表示并将满足 comp_func 的值置为 0.0
        actual = actual.todense()
        actual[comp_func(actual)] = 0.0
        # 断言 actual 的稠密表示与 desired 的稠密表示相同
        assert_equal(actual, desired.todense())
    else:
        # 对于其他 func 函数，断言 actual 与 desired 相等
        assert_equal(actual, desired)
```