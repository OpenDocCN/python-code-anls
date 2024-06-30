# `D:\src\scipysrc\scipy\scipy\sparse\csgraph\tests\test_traversal.py`

```
# 导入必要的库
import numpy as np  # 导入 NumPy 库并使用别名 np
import pytest  # 导入 pytest 库
from numpy.testing import assert_array_almost_equal  # 导入 NumPy 测试模块中的数组近似相等断言函数
from scipy.sparse import csr_array  # 导入 SciPy 稀疏矩阵中的 CSR 格式
from scipy.sparse.csgraph import (breadth_first_tree, depth_first_tree,  # 从 SciPy 稀疏图算法模块中导入广度优先搜索树和深度优先搜索树等函数
    csgraph_to_dense, csgraph_from_dense)


def test_graph_breadth_first():
    # 定义一个稠密图的邻接矩阵
    csgraph = np.array([[0, 1, 2, 0, 0],
                        [1, 0, 0, 0, 3],
                        [2, 0, 0, 7, 0],
                        [0, 0, 7, 0, 1],
                        [0, 3, 0, 1, 0]])
    # 将稠密图转换为稀疏图的 CSR 格式
    csgraph = csgraph_from_dense(csgraph, null_value=0)

    # 预期的广度优先搜索树的邻接矩阵
    bfirst = np.array([[0, 1, 2, 0, 0],
                       [0, 0, 0, 0, 3],
                       [0, 0, 0, 7, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]])

    # 对于有向和无向图，分别测试广度优先搜索树的生成
    for directed in [True, False]:
        bfirst_test = breadth_first_tree(csgraph, 0, directed)
        # 断言生成的树的邻接矩阵近似等于预期的邻接矩阵
        assert_array_almost_equal(csgraph_to_dense(bfirst_test),
                                  bfirst)


def test_graph_depth_first():
    # 定义一个稠密图的邻接矩阵
    csgraph = np.array([[0, 1, 2, 0, 0],
                        [1, 0, 0, 0, 3],
                        [2, 0, 0, 7, 0],
                        [0, 0, 7, 0, 1],
                        [0, 3, 0, 1, 0]])
    # 将稠密图转换为稀疏图的 CSR 格式
    csgraph = csgraph_from_dense(csgraph, null_value=0)

    # 预期的深度优先搜索树的邻接矩阵
    dfirst = np.array([[0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 3],
                       [0, 0, 0, 0, 0],
                       [0, 0, 7, 0, 0],
                       [0, 0, 0, 1, 0]])

    # 对于有向和无向图，分别测试深度优先搜索树的生成
    for directed in [True, False]:
        dfirst_test = depth_first_tree(csgraph, 0, directed)
        # 断言生成的树的邻接矩阵近似等于预期的邻接矩阵
        assert_array_almost_equal(csgraph_to_dense(dfirst_test),
                                  dfirst)


def test_graph_breadth_first_trivial_graph():
    # 定义一个只有一个节点的稠密图的邻接矩阵
    csgraph = np.array([[0]])
    # 将稠密图转换为稀疏图的 CSR 格式
    csgraph = csgraph_from_dense(csgraph, null_value=0)

    # 预期的广度优先搜索树的邻接矩阵
    bfirst = np.array([[0]])

    # 对于有向和无向图，分别测试广度优先搜索树的生成
    for directed in [True, False]:
        bfirst_test = breadth_first_tree(csgraph, 0, directed)
        # 断言生成的树的邻接矩阵近似等于预期的邻接矩阵
        assert_array_almost_equal(csgraph_to_dense(bfirst_test),
                                  bfirst)


def test_graph_depth_first_trivial_graph():
    # 定义一个只有一个节点的稠密图的邻接矩阵
    csgraph = np.array([[0]])
    # 将稠密图转换为稀疏图的 CSR 格式
    csgraph = csgraph_from_dense(csgraph, null_value=0)

    # 预期的深度优先搜索树的邻接矩阵
    bfirst = np.array([[0]])

    # 对于有向和无向图，分别测试深度优先搜索树的生成
    for directed in [True, False]:
        bfirst_test = depth_first_tree(csgraph, 0, directed)
        # 断言生成的树的邻接矩阵近似等于预期的邻接矩阵
        assert_array_almost_equal(csgraph_to_dense(bfirst_test),
                                  bfirst)


@pytest.mark.parametrize('directed', [True, False])
@pytest.mark.parametrize('tree_func', [breadth_first_tree, depth_first_tree])
def test_int64_indices(tree_func, directed):
    # 见 https://github.com/scipy/scipy/issues/18716
    # 创建一个稀疏图的 CSR 格式，其中的索引数据类型是 int64
    g = csr_array(([1], np.array([[0], [1]], dtype=np.int64)), shape=(2, 2))
    # 断言稀疏图的索引数据类型是 int64
    assert g.indices.dtype == np.int64
    # 使用参数化的函数测试生成树的邻接矩阵
    tree = tree_func(g, 0, directed=directed)
    # 断言生成的树的邻接矩阵近似等于预期的邻接矩阵
    assert_array_almost_equal(csgraph_to_dense(tree), [[0, 1], [0, 0]])
```