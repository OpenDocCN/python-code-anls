# `D:\src\scipysrc\scipy\scipy\sparse\csgraph\tests\test_flow.py`

```
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.csgraph import maximum_flow
from scipy.sparse.csgraph._flow import (
    _add_reverse_edges, _make_edge_pointers, _make_tails
)

# 定义两种最大流算法
methods = ['edmonds_karp', 'dinic']

# 测试：当输入为稠密矩阵时抛出 TypeError 异常
def test_raises_on_dense_input():
    with pytest.raises(TypeError):
        graph = np.array([[0, 1], [0, 0]])
        maximum_flow(graph, 0, 1)
        maximum_flow(graph, 0, 1, method='edmonds_karp')

# 测试：当输入为 csc 稀疏矩阵时抛出 TypeError 异常
def test_raises_on_csc_input():
    with pytest.raises(TypeError):
        graph = csc_matrix([[0, 1], [0, 0]])
        maximum_flow(graph, 0, 1)
        maximum_flow(graph, 0, 1, method='edmonds_karp')

# 测试：当输入包含浮点数时抛出 ValueError 异常
def test_raises_on_floating_point_input():
    with pytest.raises(ValueError):
        graph = csr_matrix([[0, 1.5], [0, 0]], dtype=np.float64)
        maximum_flow(graph, 0, 1)
        maximum_flow(graph, 0, 1, method='edmonds_karp')

# 测试：当输入矩阵不是方阵时抛出 ValueError 异常
def test_raises_on_non_square_input():
    with pytest.raises(ValueError):
        graph = csr_matrix([[0, 1, 2], [2, 1, 0]])
        maximum_flow(graph, 0, 1)

# 测试：当源点与汇点相同时抛出 ValueError 异常
def test_raises_when_source_is_sink():
    with pytest.raises(ValueError):
        graph = csr_matrix([[0, 1], [0, 0]])
        maximum_flow(graph, 0, 0)
        maximum_flow(graph, 0, 0, method='edmonds_karp')

# 参数化测试：测试源点越界时抛出 ValueError 异常
@pytest.mark.parametrize('method', methods)
@pytest.mark.parametrize('source', [-1, 2, 3])
def test_raises_when_source_is_out_of_bounds(source, method):
    with pytest.raises(ValueError):
        graph = csr_matrix([[0, 1], [0, 0]])
        maximum_flow(graph, source, 1, method=method)

# 参数化测试：测试汇点越界时抛出 ValueError 异常
@pytest.mark.parametrize('method', methods)
@pytest.mark.parametrize('sink', [-1, 2, 3])
def test_raises_when_sink_is_out_of_bounds(sink, method):
    with pytest.raises(ValueError):
        graph = csr_matrix([[0, 1], [0, 0]])
        maximum_flow(graph, 0, sink, method=method)

# 参数化测试：测试简单的流网络图例子
@pytest.mark.parametrize('method', methods)
def test_simple_graph(method):
    # This graph looks as follows:
    #     (0) --5--> (1)
    graph = csr_matrix([[0, 5], [0, 0]])
    res = maximum_flow(graph, 0, 1, method=method)
    assert res.flow_value == 5
    expected_flow = np.array([[0, 5], [-5, 0]])
    assert_array_equal(res.flow.toarray(), expected_flow)

# 参数化测试：测试瓶颈流网络图例子
@pytest.mark.parametrize('method', methods)
def test_bottle_neck_graph(method):
    # This graph cannot use the full capacity between 0 and 1:
    #     (0) --5--> (1) --3--> (2)
    graph = csr_matrix([[0, 5, 0], [0, 0, 3], [0, 0, 0]])
    res = maximum_flow(graph, 0, 2, method=method)
    assert res.flow_value == 3
    expected_flow = np.array([[0, 3, 0], [-3, 0, 3], [0, -3, 0]])
    assert_array_equal(res.flow.toarray(), expected_flow)

# 参数化测试：测试反向流问题
@pytest.mark.parametrize('method', methods)
def test_backwards_flow(method):
    # This example causes backwards flow between vertices 3 and 4,
    # and so this test ensures that we handle that accordingly. See
    #     https://stackoverflow.com/q/38843963/5085211
    # 创建一个稀疏矩阵表示的图结构，用于最大流算法的输入
    graph = csr_matrix([[0, 10, 0, 0, 10, 0, 0, 0],
                        [0, 0, 10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 10, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 10],
                        [0, 0, 0, 10, 0, 10, 0, 0],
                        [0, 0, 0, 0, 0, 0, 10, 0],
                        [0, 0, 0, 0, 0, 0, 0, 10],
                        [0, 0, 0, 0, 0, 0, 0, 0]])
    
    # 调用最大流算法，计算从节点 0 到节点 7 的最大流
    res = maximum_flow(graph, 0, 7, method=method)
    
    # 确保计算得到的最大流值为 20
    assert res.flow_value == 20
    
    # 预期的流量分布矩阵，用于验证算法计算的正确性
    expected_flow = np.array([[0, 10, 0, 0, 10, 0, 0, 0],
                              [-10, 0, 10, 0, 0, 0, 0, 0],
                              [0, -10, 0, 10, 0, 0, 0, 0],
                              [0, 0, -10, 0, 0, 0, 0, 10],
                              [-10, 0, 0, 0, 0, 10, 0, 0],
                              [0, 0, 0, 0, -10, 0, 10, 0],
                              [0, 0, 0, 0, 0, -10, 0, 10],
                              [0, 0, 0, -10, 0, 0, -10, 0]])
    
    # 确保最大流算法计算得到的流量矩阵与预期一致
    assert_array_equal(res.flow.toarray(), expected_flow)
@pytest.mark.parametrize('method', methods)
def test_example_from_clrs_chapter_26_1(method):
    # 使用 pytest 框架执行单元测试，参数化方法使用变量 methods 中的方法
    # 测试 CLRS 第二版第659页的示例，但是找到的最大流略有不同；
    # 我们将流量推送到 v_1 而不是 v_2。
    graph = csr_matrix([[0, 16, 13, 0, 0, 0],    # 创建一个压缩稀疏行矩阵 graph
                        [0, 0, 10, 12, 0, 0],     # 定义图的结构和边的容量
                        [0, 4, 0, 0, 14, 0],      # 每一行代表一个顶点的出边情况
                        [0, 0, 9, 0, 0, 20],      # 0 表示没有边连接，非零值表示边的容量
                        [0, 0, 0, 7, 0, 4],       # 具体的数字表示容量大小
                        [0, 0, 0, 0, 0, 0]])      # 例如，16 表示顶点 0 到顶点 1 的边容量为 16
    res = maximum_flow(graph, 0, 5, method=method)  # 调用最大流算法计算从顶点 0 到顶点 5 的最大流
    assert res.flow_value == 23    # 断言最大流的值为 23
    expected_flow = np.array([[0, 12, 11, 0, 0, 0],    # 定义预期的流量分布数组
                              [-12, 0, 0, 12, 0, 0],    # 包括每条边的流量变化
                              [-11, 0, 0, 0, 11, 0],    # 负值表示反向流
                              [0, -12, 0, 0, -7, 19],   # 非负值表示正向流
                              [0, 0, -11, 7, 0, 4],     # 数组维度与 graph 的边数一致
                              [0, 0, 0, -19, -4, 0]])   # np.array 用于处理数组操作
    assert_array_equal(res.flow.toarray(), expected_flow)  # 断言实际计算的流量数组与预期的一致


@pytest.mark.parametrize('method', methods)
def test_disconnected_graph(method):
    # 这个测试检验了以下的不连通图：
    #     (0) --5--> (1)    (2) --3--> (3)
    graph = csr_matrix([[0, 5, 0, 0],    # 创建一个压缩稀疏行矩阵 graph
                        [0, 0, 0, 0],    # 定义图的结构，但没有边的容量
                        [0, 0, 9, 3],    # 每一行代表一个顶点的出边情况
                        [0, 0, 0, 0]])   # 0 表示没有边连接
    res = maximum_flow(graph, 0, 3, method=method)  # 调用最大流算法计算从顶点 0 到顶点 3 的最大流
    assert res.flow_value == 0    # 断言最大流的值为 0，因为图是不连通的
    expected_flow = np.zeros((4, 4), dtype=np.int32)  # 创建一个预期的流量分布数组，全为 0
    assert_array_equal(res.flow.toarray(), expected_flow)  # 断言实际计算的流量数组与预期的一致


@pytest.mark.parametrize('method', methods)
def test_add_reverse_edges_large_graph(method):
    # 对于 https://github.com/scipy/scipy/issues/14385 的回归测试
    n = 100_000    # 定义顶点数量
    indices = np.arange(1, n)    # 创建索引数组
    indptr = np.array(list(range(n)) + [n - 1])    # 创建索引指针数组
    data = np.ones(n - 1, dtype=np.int32)    # 创建数据数组，全为 1
    graph = csr_matrix((data, indices, indptr), shape=(n, n))    # 创建一个非常大的压缩稀疏行矩阵 graph
    res = maximum_flow(graph, 0, n - 1, method=method)  # 调用最大流算法计算从顶点 0 到顶点 n-1 的最大流
    assert res.flow_value == 1    # 断言最大流的值为 1
    expected_flow = graph - graph.transpose()    # 计算预期的流量分布，即原图减去其转置
    assert_array_equal(res.flow.data, expected_flow.data)    # 断言实际计算的流量数据与预期的一致
    assert_array_equal(res.flow.indices, expected_flow.indices)  # 断言实际计算的流量索引与预期的一致
    assert_array_equal(res.flow.indptr, expected_flow.indptr)    # 断言实际计算的流量指针与预期的一致


@pytest.mark.parametrize("a,b_data_expected", [
    ([[]], []),    # 参数化测试：空数组的情况
    ([[0], [0]], []),    # 参数化测试：所有边容量为 0 的情况
    ([[1, 0, 2], [0, 0, 0], [0, 3, 0]], [1, 2, 0, 0, 3]),    # 参数化测试：正常图的情况
    ([[9, 8, 7], [4, 5, 6], [0, 0, 0]], [9, 8, 7, 4, 5, 6, 0, 0])    # 参数化测试：包含容量大于 0 的情况
])
def test_add_reverse_edges(a, b_data_expected):
    """测试反转输入图的边是否按预期工作。"""
    a = csr_matrix(a, dtype=np.int32, shape=(len(a), len(a)))    # 创建压缩稀疏行矩阵 a
    b = _add_reverse_edges(a)    # 调用函数 _add_reverse_edges 反转图的边
    assert_array_equal(b.data, b_data_expected)    # 断言反转后的边的数据与预期一致


@pytest.mark.parametrize("a,expected", [
    ([[]], []),    # 参数化测试：空数组的情况
    ([[0]], []),    # 参数化测试：所有边容量为 0 的情况
    ([[1]], [0]),    # 参数化测试：单条边容量大于 0 的情况
    ([[0, 1], [10, 0]], [1, 0]),    # 参数化测试：正常图的情况
    ([[1, 0, 2], [0, 0, 3], [4, 5, 0]], [0, 3, 4, 1, 2])    # 参数化测试：包含容量大于 0 的情况
])
def test_make_edge_pointers(a, expected):
    a = csr_matrix(a, dtype=np.int32)    # 创建压缩稀疏行矩阵 a
    # 调用函数 _make_edge_pointers(a)，生成反向边指针数组
    rev_edge_ptr = _make_edge_pointers(a)
    # 使用断言检查 rev_edge_ptr 是否与预期的数组 expected 相等
    assert_array_equal(rev_edge_ptr, expected)
# 使用 pytest 的 parametrize 装饰器，定义多组参数化测试数据和期望结果
@pytest.mark.parametrize("a,expected", [
    # 第一组参数化测试数据和期望结果
    ([[]], []),           # 空列表的情况，期望返回空列表
    ([[0]], []),          # 单个元素为 0 的二维列表，期望返回空列表
    ([[1]], [0]),         # 单个元素为 1 的二维列表，期望返回 [0]
    ([[0, 1], [10, 0]], [0, 1]),  # 含有非零元素的二维列表，期望返回 [0, 1]
    ([[1, 0, 2], [0, 0, 3], [4, 5, 0]], [0, 0, 1, 2, 2])  # 复杂的二维列表，期望返回 [0, 0, 1, 2, 2]
])
def test_make_tails(a, expected):
    # 将二维列表 a 转换为 csr_matrix 类型的稀疏矩阵，数据类型为 np.int32
    a = csr_matrix(a, dtype=np.int32)
    # 调用 _make_tails 函数，获取其返回值，即尾部元素组成的列表
    tails = _make_tails(a)
    # 使用 assert_array_equal 函数断言 tails 是否等于期望结果 expected
    assert_array_equal(tails, expected)
```