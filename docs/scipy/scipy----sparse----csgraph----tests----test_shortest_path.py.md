# `D:\src\scipysrc\scipy\scipy\sparse\csgraph\tests\test_shortest_path.py`

```
# 导入需要的库
from io import StringIO
import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_allclose
from pytest import raises as assert_raises
from scipy.sparse.csgraph import (shortest_path, dijkstra, johnson,
                                  bellman_ford, construct_dist_matrix, yen,
                                  NegativeCycleError)
import scipy.sparse
from scipy.io import mmread
import pytest

# 定义有向图的邻接矩阵
directed_G = np.array([[0, 3, 3, 0, 0],
                       [0, 0, 0, 2, 4],
                       [0, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0],
                       [2, 0, 0, 2, 0]], dtype=float)

# 定义无向图的邻接矩阵
undirected_G = np.array([[0, 3, 3, 1, 2],
                         [3, 0, 0, 2, 4],
                         [3, 0, 0, 0, 0],
                         [1, 2, 0, 0, 2],
                         [2, 4, 0, 2, 0]], dtype=float)

# 生成无向图的邻接矩阵的布尔表示（是否有边）
unweighted_G = (directed_G > 0).astype(float)

# 有向图的预计算最短路径距离矩阵
directed_SP = [[0, 3, 3, 5, 7],
               [3, 0, 6, 2, 4],
               [np.inf, np.inf, 0, np.inf, np.inf],
               [1, 4, 4, 0, 8],
               [2, 5, 5, 2, 0]]

# 有向图从节点0到节点3的两个最短路径长度矩阵
directed_2SP_0_to_3 = [[-9999, 0, -9999, 1, -9999],
                       [-9999, 0, -9999, 4, 1]]

# 稀疏有向图的零权重边的邻接矩阵表示
directed_sparse_zero_G = scipy.sparse.csr_matrix(
    (
        [0, 1, 2, 3, 1],
        ([0, 1, 2, 3, 4], [1, 2, 0, 4, 3]),
    ),
    shape=(5, 5),
)

# 稀疏有向图的预计算最短路径距离矩阵
directed_sparse_zero_SP = [[0, 0, 1, np.inf, np.inf],
                           [3, 0, 1, np.inf, np.inf],
                           [2, 2, 0, np.inf, np.inf],
                           [np.inf, np.inf, np.inf, 0, 3],
                           [np.inf, np.inf, np.inf, 1, 0]]

# 稀疏无向图的零权重边的邻接矩阵表示
undirected_sparse_zero_G = scipy.sparse.csr_matrix(
    (
        [0, 0, 1, 1, 2, 2, 1, 1],
        ([0, 1, 1, 2, 2, 0, 3, 4], [1, 0, 2, 1, 0, 2, 4, 3])
    ),
    shape=(5, 5),
)

# 稀疏无向图的预计算最短路径距离矩阵
undirected_sparse_zero_SP = [[0, 0, 1, np.inf, np.inf],
                             [0, 0, 1, np.inf, np.inf],
                             [1, 1, 0, np.inf, np.inf],
                             [np.inf, np.inf, np.inf, 0, 1],
                             [np.inf, np.inf, np.inf, 1, 0]]

# 有向图的预测距离矩阵，即节点间的最短路径前驱矩阵
directed_pred = np.array([[-9999, 0, 0, 1, 1],
                          [3, -9999, 0, 1, 1],
                          [-9999, -9999, -9999, -9999, -9999],
                          [3, 0, 0, -9999, 1],
                          [4, 0, 0, 4, -9999]], dtype=float)

# 无向图的预计算最短路径距离矩阵
undirected_SP = np.array([[0, 3, 3, 1, 2],
                          [3, 0, 6, 2, 4],
                          [3, 6, 0, 4, 5],
                          [1, 2, 4, 0, 2],
                          [2, 4, 5, 2, 0]], dtype=float)

# 限制最大权重为2的无向图的预计算最短路径距离矩阵
undirected_SP_limit_2 = np.array([[0, np.inf, np.inf, 1, 2],
                                  [np.inf, 0, np.inf, 2, np.inf],
                                  [np.inf, np.inf, 0, np.inf, np.inf],
                                  [1, 2, np.inf, 0, 2],
                                  [2, np.inf, np.inf, 2, 0]], dtype=float)

# 所有边权重为0的无向图的预计算最短路径距离矩阵
undirected_SP_limit_0 = np.ones((5, 5), dtype=float) - np.eye(5)
# 将 undirected_SP_limit_0 中大于 0 的元素设为无穷大
undirected_SP_limit_0[undirected_SP_limit_0 > 0] = np.inf

# 初始化一个大小为 5x5 的浮点数数组，表示无向图的先驱节点矩阵，-9999 表示未连接
undirected_pred = np.array([[-9999, 0, 0, 0, 0],
                            [1, -9999, 0, 1, 1],
                            [2, 0, -9999, 0, 0],
                            [3, 3, 0, -9999, 3],
                            [4, 4, 0, 4, -9999]], dtype=float)

# 初始化一个大小为 3x3 的浮点数数组，表示带负权重的有向图
directed_negative_weighted_G = np.array([[0, 0, 0],
                                         [-1, 0, 0],
                                         [0, -1, 0]], dtype=float)

# 初始化一个大小为 3x3 的浮点数数组，表示带负权重的有向图的最短路径矩阵，np.inf 表示不可达
directed_negative_weighted_SP = np.array([[0, np.inf, np.inf],
                                          [-1, 0, np.inf],
                                          [-2, -1, 0]], dtype=float)

# 方法列表，用于测试不同的路径算法
methods = ['auto', 'FW', 'D', 'BF', 'J']


def test_dijkstra_limit():
    # 限制列表，包括 0、2 和无穷大
    limits = [0, 2, np.inf]
    # 对应的结果列表，包括 undirected_SP_limit_0、undirected_SP_limit_2 和 undirected_SP
    results = [undirected_SP_limit_0,
               undirected_SP_limit_2,
               undirected_SP]

    def check(limit, result):
        # 调用 dijkstra 算法计算最短路径，并断言结果与期望结果相近
        SP = dijkstra(undirected_G, directed=False, limit=limit)
        assert_array_almost_equal(SP, result)

    # 对每个限制和结果进行测试
    for limit, result in zip(limits, results):
        check(limit, result)


def test_directed():
    def check(method):
        # 调用 shortest_path 算法计算有向图的最短路径，并断言结果与期望结果相近
        SP = shortest_path(directed_G, method=method, directed=True,
                           overwrite=False)
        assert_array_almost_equal(SP, directed_SP)

    # 对每种方法进行测试
    for method in methods:
        check(method)


def test_undirected():
    def check(method, directed_in):
        if directed_in:
            # 若 directed_in 为真，则计算有向图的无向化最短路径，并断言结果与 undirected_SP 相近
            SP1 = shortest_path(directed_G, method=method, directed=False,
                                overwrite=False)
            assert_array_almost_equal(SP1, undirected_SP)
        else:
            # 若 directed_in 为假，则计算无向图的有向化最短路径，并断言结果与 undirected_SP 相近
            SP2 = shortest_path(undirected_G, method=method, directed=True,
                                overwrite=False)
            assert_array_almost_equal(SP2, undirected_SP)

    # 对每种方法和 directed_in 进行测试
    for method in methods:
        for directed_in in (True, False):
            check(method, directed_in)


def test_directed_sparse_zero():
    # 测试带零权重边和两个连接组件的稀疏有向图
    def check(method):
        # 调用 shortest_path 算法计算带零权重边的稀疏有向图的最短路径，并断言结果与期望结果相近
        SP = shortest_path(directed_sparse_zero_G, method=method, directed=True,
                           overwrite=False)
        assert_array_almost_equal(SP, directed_sparse_zero_SP)

    # 对每种方法进行测试
    for method in methods:
        check(method)


def test_undirected_sparse_zero():
    def check(method, directed_in):
        if directed_in:
            # 若 directed_in 为真，则计算带零权重边的稀疏有向图的无向化最短路径，并断言结果与 undirected_sparse_zero_SP 相近
            SP1 = shortest_path(directed_sparse_zero_G, method=method, directed=False,
                                overwrite=False)
            assert_array_almost_equal(SP1, undirected_sparse_zero_SP)
        else:
            # 若 directed_in 为假，则计算带零权重边的稀疏无向图的有向化最短路径，并断言结果与 undirected_sparse_zero_SP 相近
            SP2 = shortest_path(undirected_sparse_zero_G, method=method, directed=True,
                                overwrite=False)
            assert_array_almost_equal(SP2, undirected_sparse_zero_SP)

    # 对每种方法和 directed_in 进行测试
    for method in methods:
        for directed_in in (True, False):
            check(method, directed_in)
# 使用 pytest 模块的 parametrize 装饰器为函数 test_dijkstra_indices_min_only 添加参数化测试
@pytest.mark.parametrize('directed, SP_ans',
                         ((True, directed_SP),  # 设置 directed=True，使用 directed_SP 作为预期最短路径数组
                          (False, undirected_SP)))  # 设置 directed=False，使用 undirected_SP 作为预期最短路径数组
@pytest.mark.parametrize('indices', ([0, 2, 4], [0, 4], [3, 4], [0, 0]))  # 参数化 indices 参数，测试多种索引组合
def test_dijkstra_indices_min_only(directed, SP_ans, indices):
    SP_ans = np.array(SP_ans)  # 将预期最短路径数组转换为 NumPy 数组
    indices = np.array(indices, dtype=np.int64)  # 将 indices 转换为 NumPy int64 数组
    min_ind_ans = indices[np.argmin(SP_ans[indices, :], axis=0)]  # 根据 SP_ans 计算最小路径索引数组
    min_d_ans = np.zeros(SP_ans.shape[0], SP_ans.dtype)  # 创建全零数组作为最短路径值的容器
    for k in range(SP_ans.shape[0]):
        min_d_ans[k] = SP_ans[min_ind_ans[k], k]  # 填充最短路径值数组
    min_ind_ans[np.isinf(min_d_ans)] = -9999  # 将无穷大路径的索引设置为 -9999

    # 调用 dijkstra 函数执行最短路径算法，并获取返回的最短路径、前驱节点、源节点数组
    SP, pred, sources = dijkstra(directed_G,
                                 directed=directed,
                                 indices=indices,
                                 min_only=True,
                                 return_predecessors=True)
    assert_array_almost_equal(SP, min_d_ans)  # 断言计算得到的最短路径数组与预期的最短路径数组相近
    assert_array_equal(min_ind_ans, sources)  # 断言计算得到的源节点数组与预期的源节点数组相等

    # 再次调用 dijkstra 函数，不返回前驱节点数组，只返回最短路径数组
    SP = dijkstra(directed_G,
                  directed=directed,
                  indices=indices,
                  min_only=True,
                  return_predecessors=False)
    assert_array_almost_equal(SP, min_d_ans)  # 断言计算得到的最短路径数组与预期的最短路径数组相近


# 使用 pytest 模块的 parametrize 装饰器为函数 test_dijkstra_min_only_random 添加参数化测试
@pytest.mark.parametrize('n', (10, 100, 1000))  # 参数化 n 参数，测试不同的图大小
def test_dijkstra_min_only_random(n):
    np.random.seed(1234)  # 设置随机数种子以保证测试结果可重复
    data = scipy.sparse.rand(n, n, density=0.5, format='lil',
                             random_state=42, dtype=np.float64)  # 生成稀疏随机矩阵作为测试图的邻接矩阵表示
    data.setdiag(np.zeros(n, dtype=np.bool_))  # 将对角线元素置为零
    v = np.arange(n)  # 创建顶点索引数组
    np.random.shuffle(v)  # 随机打乱顶点索引数组
    indices = v[:int(n*.1)]  # 选取部分顶点作为起始索引数组
    # 调用 dijkstra 函数执行最短路径算法，并获取返回的最短路径、前驱节点、源节点数组
    ds, pred, sources = dijkstra(data,
                                 directed=True,
                                 indices=indices,
                                 min_only=True,
                                 return_predecessors=True)
    # 遍历每个节点，验证其前驱节点和源节点的关系
    for k in range(n):
        p = pred[k]
        s = sources[k]
        while p != -9999:
            assert sources[p] == s  # 断言当前节点的前驱节点的源节点与当前节点的源节点相同
            p = pred[p]  # 获取下一个前驱节点


def test_dijkstra_random():
    # 重现在 gh-17782 中观察到的 hang 问题
    n = 10
    indices = [0, 4, 4, 5, 7, 9, 0, 6, 2, 3, 7, 9, 1, 2, 9, 2, 5, 6]  # 指定起始索引数组
    indptr = [0, 0, 2, 5, 6, 7, 8, 12, 15, 18, 18]  # 指定 CSR 格式中的行指针数组
    data = [0.33629, 0.40458, 0.47493, 0.42757, 0.11497, 0.91653, 0.69084,
            0.64979, 0.62555, 0.743, 0.01724, 0.99945, 0.31095, 0.15557,
            0.02439, 0.65814, 0.23478, 0.24072]  # 指定图的权重数据
    graph = scipy.sparse.csr_matrix((data, indices, indptr), shape=(n, n))  # 使用 CSR 矩阵表示图
    dijkstra(graph, directed=True, return_predecessors=True)  # 调用 dijkstra 函数执行最短路径算法


def test_gh_17782_segfault():
    # 待补充测试内容
    # 定义一个多行字符串，包含 MatrixMarket 格式的稀疏矩阵数据
    text = """%%MatrixMarket matrix coordinate real general
                84 84 22
                2 1 4.699999809265137e+00
                6 14 1.199999973177910e-01
                9 6 1.199999973177910e-01
                10 16 2.012000083923340e+01
                11 10 1.422000026702881e+01
                12 1 9.645999908447266e+01
                13 18 2.012000083923340e+01
                14 13 4.679999828338623e+00
                15 11 1.199999973177910e-01
                16 12 1.199999973177910e-01
                18 15 1.199999973177910e-01
                32 2 2.299999952316284e+00
                33 20 6.000000000000000e+00
                33 32 5.000000000000000e+00
                36 9 3.720000028610229e+00
                36 37 3.720000028610229e+00
                36 38 3.720000028610229e+00
                37 44 8.159999847412109e+00
                38 32 7.903999328613281e+01
                43 20 2.400000000000000e+01
                43 33 4.000000000000000e+00
                44 43 6.028000259399414e+01
    """
    
    # 使用 StringIO 将上述字符串转换为文件对象，然后使用 mmread 函数解析该对象中的稀疏矩阵数据
    data = mmread(StringIO(text))
    
    # 调用 dijkstra 算法处理解析后的稀疏矩阵数据
    # 设置 directed=True 表示处理有向图，设置 return_predecessors=True 表示需要返回前驱节点信息
    dijkstra(data, directed=True, return_predecessors=True)
def test_shortest_path_indices():
    # 创建一个长度为4的numpy数组
    indices = np.arange(4)

    def check(func, indshape):
        # 将indshape扩展为(indshape[0], 5)的形状
        outshape = indshape + (5,)
        # 使用func计算最短路径，传入的indices为重新形状为indshape的数组
        SP = func(directed_G, directed=False,
                  indices=indices.reshape(indshape))
        # 断言计算得到的SP与预期的undirected_SP的形状一致
        assert_array_almost_equal(SP, undirected_SP[indices].reshape(outshape))

    # 遍历不同的indshape形状，以及四种不同的最短路径计算函数
    for indshape in [(4,), (4, 1), (2, 2)]:
        for func in (dijkstra, bellman_ford, johnson, shortest_path):
            check(func, indshape)

    # 断言当使用不支持的方法时，会引发ValueError异常
    assert_raises(ValueError, shortest_path, directed_G, method='FW',
                  indices=indices)


def test_predecessors():
    # 指定不同情况下的预期结果
    SP_res = {True: directed_SP,
              False: undirected_SP}
    pred_res = {True: directed_pred,
                False: undirected_pred}

    def check(method, directed):
        # 获取最短路径和前驱节点数组
        SP, pred = shortest_path(directed_G, method, directed=directed,
                                 overwrite=False,
                                 return_predecessors=True)
        # 断言计算得到的最短路径与预期的结果一致
        assert_array_almost_equal(SP, SP_res[directed])
        # 断言计算得到的前驱节点数组与预期的结果一致
        assert_array_almost_equal(pred, pred_res[directed])

    # 遍历所有方法和两种方向（有向和无向）
    for method in methods:
        for directed in (True, False):
            check(method, directed)


def test_construct_shortest_path():
    def check(method, directed):
        # 计算最短路径和前驱节点数组
        SP1, pred = shortest_path(directed_G,
                                  directed=directed,
                                  overwrite=False,
                                  return_predecessors=True)
        # 使用前驱节点数组构建距离矩阵
        SP2 = construct_dist_matrix(directed_G, pred, directed=directed)
        # 断言两种方法计算得到的最短路径矩阵一致
        assert_array_almost_equal(SP1, SP2)

    # 遍历所有方法和两种方向（有向和无向）
    for method in methods:
        for directed in (True, False):
            check(method, directed)


def test_unweighted_path():
    def check(method, directed):
        # 计算有权重图和无权重图的最短路径
        SP1 = shortest_path(directed_G,
                            directed=directed,
                            overwrite=False,
                            unweighted=True)
        SP2 = shortest_path(unweighted_G,
                            directed=directed,
                            overwrite=False,
                            unweighted=False)
        # 断言两种方法计算得到的最短路径矩阵一致
        assert_array_almost_equal(SP1, SP2)

    # 遍历所有方法和两种方向（有向和无向）
    for method in methods:
        for directed in (True, False):
            check(method, directed)


def test_negative_cycles():
    # 创建一个带有负环的小图
    graph = np.ones([5, 5])
    graph.flat[::6] = 0
    graph[1, 2] = -2

    def check(method, directed):
        # 断言使用不同方法计算带负权重的图时会引发NegativeCycleError异常
        assert_raises(NegativeCycleError, shortest_path, graph, method,
                      directed)

    # 遍历所有方法和两种方向（有向和无向）
    for directed in (True, False):
        for method in ['FW', 'J', 'BF']:
            check(method, directed)

        # 断言在yen算法中，对带有负权重的图计算时会引发NegativeCycleError异常
        assert_raises(NegativeCycleError, yen, graph, 0, 1, 1,
                      directed=directed)


@pytest.mark.parametrize("method", ['FW', 'J', 'BF'])
def test_negative_weights(method):
    # 使用指定方法计算带负权重的有向图的最短路径
    SP = shortest_path(directed_negative_weighted_G, method, directed=True)
    # 断言计算得到的最短路径矩阵与预期结果一致
    assert_allclose(SP, directed_negative_weighted_SP, atol=1e-10)
def test_masked_input():
    np.ma.masked_equal(directed_G, 0)
    # 对 directed_G 应用掩码，将值为 0 的元素掩盖（即忽略）

    def check(method):
        SP = shortest_path(directed_G, method=method, directed=True,
                           overwrite=False)
        # 使用指定方法计算 directed_G 的最短路径，保持图的有向性，不覆盖原图
        assert_array_almost_equal(SP, directed_SP)
        # 断言计算得到的最短路径 SP 与预期的 directed_SP 几乎相等

    for method in methods:
        check(method)
        # 对每个方法调用 check 函数，验证最短路径计算的正确性


def test_overwrite():
    G = np.array([[0, 3, 3, 1, 2],
                  [3, 0, 0, 2, 4],
                  [3, 0, 0, 0, 0],
                  [1, 2, 0, 0, 2],
                  [2, 4, 0, 2, 0]], dtype=float)
    foo = G.copy()
    # 复制 G 到 foo
    shortest_path(foo, overwrite=False)
    # 在不覆盖的情况下计算 foo 的最短路径
    assert_array_equal(foo, G)
    # 断言 foo 与原始 G 在不覆盖的情况下应保持不变


@pytest.mark.parametrize('method', methods)
def test_buffer(method):
    # Smoke test that sparse matrices with read-only buffers (e.g., those from
    # joblib workers) do not cause::
    #
    #     ValueError: buffer source array is read-only
    #
    G = scipy.sparse.csr_matrix([[1.]])
    G.data.flags['WRITEABLE'] = False
    # 设置 G 的数据部分为不可写
    shortest_path(G, method=method)
    # 使用指定方法计算稀疏矩阵 G 的最短路径


def test_NaN_warnings():
    with warnings.catch_warnings(record=True) as record:
        shortest_path(np.array([[0, 1], [np.nan, 0]]))
    # 捕获警告，因为计算中含有 NaN 值
    for r in record:
        assert r.category is not RuntimeWarning
        # 断言捕获的警告类型不是运行时警告


def test_sparse_matrices():
    # Test that using lil,csr and csc sparse matrix do not cause error
    G_dense = np.array([[0, 3, 0, 0, 0],
                        [0, 0, -1, 0, 0],
                        [0, 0, 0, 2, 0],
                        [0, 0, 0, 0, 4],
                        [0, 0, 0, 0, 0]], dtype=float)
    SP = shortest_path(G_dense)
    # 计算稠密矩阵 G_dense 的最短路径
    G_csr = scipy.sparse.csr_matrix(G_dense)
    # 转换为 CSR 格式的稀疏矩阵
    G_csc = scipy.sparse.csc_matrix(G_dense)
    # 转换为 CSC 格式的稀疏矩阵
    G_lil = scipy.sparse.lil_matrix(G_dense)
    # 转换为 LIL 格式的稀疏矩阵
    assert_array_almost_equal(SP, shortest_path(G_csr))
    # 断言使用 CSR 格式的稀疏矩阵计算的最短路径与稠密矩阵的最短路径几乎相等
    assert_array_almost_equal(SP, shortest_path(G_csc))
    # 断言使用 CSC 格式的稀疏矩阵计算的最短路径与稠密矩阵的最短路径几乎相等
    assert_array_almost_equal(SP, shortest_path(G_lil))
    # 断言使用 LIL 格式的稀疏矩阵计算的最短路径与稠密矩阵的最短路径几乎相等


def test_yen_directed():
    distances, predecessors = yen(
                            directed_G,
                            source=0,
                            sink=3,
                            K=2,
                            return_predecessors=True
                        )
    # 使用 Yen's K 最短路径算法计算 directed_G 中从源点 0 到汇点 3 的两条最短路径，并返回前驱节点信息
    assert_allclose(distances, [5., 9.])
    # 断言计算得到的路径长度数组与预期的 [5., 9.] 几乎相等
    assert_allclose(predecessors, directed_2SP_0_to_3)
    # 断言计算得到的前驱节点数组与预期的 directed_2SP_0_to_3 相等


def test_yen_undirected():
    distances = yen(
        undirected_G,
        source=0,
        sink=3,
        K=4,
    )
    # 使用 Yen's K 最短路径算法计算无向图 undirected_G 中从源点 0 到汇点 3 的四条最短路径
    assert_allclose(distances, [1., 4., 5., 8.])
    # 断言计算得到的路径长度数组与预期的 [1., 4., 5., 8.] 几乎相等


def test_yen_unweighted():
    # Ask for more paths than there are, verify only the available paths are returned
    distances, predecessors = yen(
        directed_G,
        source=0,
        sink=3,
        K=4,
        unweighted=True,
        return_predecessors=True,
    )
    # 使用 Yen's K 最短路径算法计算无权重的 directed_G 中从源点 0 到汇点 3 的四条最短路径，并返回前驱节点信息
    assert_allclose(distances, [2., 3.])
    # 断言计算得到的路径长度数组与预期的 [2., 3.] 几乎相等
    assert_allclose(predecessors, directed_2SP_0_to_3)
    # 断言计算得到的前驱节点数组与预期的 directed_2SP_0_to_3 相等


def test_yen_no_paths():
    distances = yen(
        directed_G,
        source=2,
        sink=3,
        K=1,
    )
    # 使用 Yen's K 最短路径算法计算 directed_G 中从源点 2 到汇点 3 的一条最短路径
    assert distances.size == 0
    # 断言计算得到的路径数组大小为 0，即没有找到路径


def test_yen_negative_weights():
    # 使用 Yen's K 最短路径算法计算有向负权重图 directed_negative_weighted_G 中从节点 2 到节点 0 的单个最短路径
    distances = yen(
        directed_negative_weighted_G,  # 给定的有向负权重图对象
        source=2,  # 起始节点为 2
        sink=0,  # 目标节点为 0
        K=1,  # 计算一条最短路径
    )
    
    # 断言计算得到的最短路径距离与预期值 [-2.] 接近
    assert_allclose(distances, [-2.])
# 使用 pytest 的 parametrize 装饰器设置多个参数化测试用例，每个参数都会生成一组测试
@pytest.mark.parametrize("min_only", (True, False))
@pytest.mark.parametrize("directed", (True, False))
@pytest.mark.parametrize("return_predecessors", (True, False))
@pytest.mark.parametrize("index_dtype", (np.int32, np.int64))
@pytest.mark.parametrize("indices", (None, [1]))
def test_20904(min_only, directed, return_predecessors, index_dtype, indices):
    """Test two failures from gh-20904: int32 and indices-as-None."""
    
    # 创建一个 4x4 的稀疏单位矩阵，格式为 CSR
    adj_mat = scipy.sparse.eye(4, format="csr")
    
    # 将原始的 CSR 格式的数据转换成指定类型的 CSR 稀疏数组
    adj_mat = scipy.sparse.csr_array(
        (
            adj_mat.data,  # 数据部分保持不变
            adj_mat.indices.astype(index_dtype),  # 索引部分转换为指定的数据类型
            adj_mat.indptr.astype(index_dtype),  # 指针部分也转换为指定的数据类型
        ),
    )
    
    # 调用 Dijkstra 算法，传入稀疏矩阵、方向性标志、索引、最小化标志、返回前驱标志等参数
    dijkstra(
        adj_mat,
        directed,
        indices=indices,
        min_only=min_only,
        return_predecessors=return_predecessors,
    )
```