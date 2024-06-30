# `D:\src\scipysrc\scipy\scipy\sparse\csgraph\tests\test_spanning_tree.py`

```
"""Test the minimum spanning tree function"""
# 导入必要的库和模块
import numpy as np
from numpy.testing import assert_
import numpy.testing as npt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

# 定义测试最小生成树函数
def test_minimum_spanning_tree():

    # 创建一个包含两个连通分量的图
    graph = [[0,1,0,0,0],
             [1,0,0,0,0],
             [0,0,0,8,5],
             [0,0,8,0,1],
             [0,0,5,1,0]]
    graph = np.asarray(graph)

    # 创建预期的生成树
    expected = [[0,1,0,0,0],
                [0,0,0,0,0],
                [0,0,0,0,5],
                [0,0,0,0,1],
                [0,0,0,0,0]]
    expected = np.asarray(expected)

    # 确保最小生成树算法产生了预期的输出
    csgraph = csr_matrix(graph)
    mintree = minimum_spanning_tree(csgraph)
    mintree_array = mintree.toarray()
    npt.assert_array_equal(mintree_array, expected,
                           'Incorrect spanning tree found.')

    # 确保原始图未被修改
    npt.assert_array_equal(csgraph.toarray(), graph,
        'Original graph was modified.')

    # 让算法原地修改 csgraph
    mintree = minimum_spanning_tree(csgraph, overwrite=True)
    npt.assert_array_equal(mintree.toarray(), expected,
        'Graph was not properly modified to contain MST.')

    np.random.seed(1234)
    for N in (5, 10, 15, 20):

        # 创建一个随机图
        graph = 3 + np.random.random((N, N))
        csgraph = csr_matrix(graph)

        # 生成树最多有 N - 1 条边
        mintree = minimum_spanning_tree(csgraph)
        assert_(mintree.nnz < N)

        # 将子对角线设置为1以创建已知生成树
        idx = np.arange(N-1)
        graph[idx,idx+1] = 1
        csgraph = csr_matrix(graph)
        mintree = minimum_spanning_tree(csgraph)

        # 我们预期在生成树中看到这种模式，其他地方为零
        expected = np.zeros((N, N))
        expected[idx, idx+1] = 1

        npt.assert_array_equal(mintree.toarray(), expected,
            'Incorrect spanning tree found.')
```