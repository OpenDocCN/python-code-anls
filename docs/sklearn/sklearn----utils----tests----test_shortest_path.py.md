# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_shortest_path.py`

```
from collections import defaultdict  # 导入默认字典模块

import numpy as np  # 导入NumPy库
from numpy.testing import assert_array_almost_equal  # 导入NumPy测试模块

from sklearn.utils.graph import single_source_shortest_path_length  # 导入sklearn中的单源最短路径长度函数


def floyd_warshall_slow(graph, directed=False):
    N = graph.shape[0]  # 获取图的大小（节点数量）

    # 将非零元素设置为无穷大
    graph[np.where(graph == 0)] = np.inf

    # 将对角线元素设置为零
    graph.flat[:: N + 1] = 0

    if not directed:
        graph = np.minimum(graph, graph.T)  # 若图是无向图，则将其转换为对称的

    # Floyd-Warshall算法的三重循环实现
    for k in range(N):
        for i in range(N):
            for j in range(N):
                graph[i, j] = min(graph[i, j], graph[i, k] + graph[k, j])

    graph[np.where(np.isinf(graph))] = 0  # 将无穷大元素重新设置为零

    return graph  # 返回计算后的图


def generate_graph(N=20):
    # 生成随机的距离矩阵
    rng = np.random.RandomState(0)
    dist_matrix = rng.random_sample((N, N))

    # 使距离矩阵对称化：距离不依赖于方向
    dist_matrix = dist_matrix + dist_matrix.T

    # 使图变稀疏：随机将一半的距离设为零
    i = (rng.randint(N, size=N * N // 2), rng.randint(N, size=N * N // 2))
    dist_matrix[i] = 0

    # 将对角线元素设为零
    dist_matrix.flat[:: N + 1] = 0

    return dist_matrix  # 返回生成的距离矩阵


def test_shortest_path():
    dist_matrix = generate_graph(20)  # 生成一个大小为20的距离矩阵

    # 将非零距离设置为1，用于路径长度比较
    dist_matrix[dist_matrix != 0] = 1

    # 测试有向和无向情况下的最短路径函数
    for directed in (True, False):
        if not directed:
            dist_matrix = np.minimum(dist_matrix, dist_matrix.T)  # 若图是无向的，则使其对称化

        graph_py = floyd_warshall_slow(dist_matrix.copy(), directed)  # 调用Floyd-Warshall算法计算最短路径

        for i in range(dist_matrix.shape[0]):
            # 生成从节点i出发的单源最短路径长度字典
            dist_dict = defaultdict(int)
            dist_dict.update(single_source_shortest_path_length(dist_matrix, i))

            for j in range(graph_py[i].shape[0]):
                # 检查计算出的路径长度与预期的路径长度是否几乎相等
                assert_array_almost_equal(dist_dict[j], graph_py[i, j])
```