# `numpy-ml\numpy_ml\tests\test_utils.py`

```
# 禁用 flake8 的警告
import numpy as np

# 导入 scipy 库
import scipy
# 导入 networkx 库并重命名为 nx
import networkx as nx

# 从 sklearn.neighbors 模块中导入 BallTree 类并重命名为 sk_BallTree
from sklearn.neighbors import BallTree as sk_BallTree
# 从 sklearn.metrics.pairwise 模块中导入 rbf_kernel 函数并重命名为 sk_rbf
from sklearn.metrics.pairwise import rbf_kernel as sk_rbf
# 从 sklearn.metrics.pairwise 模块中导入 linear_kernel 函数并重命名为 sk_linear
from sklearn.metrics.pairwise import linear_kernel as sk_linear
# 从 sklearn.metrics.pairwise 模块中导入 polynomial_kernel 函数并重命名为 sk_poly
from sklearn.metrics.pairwise import polynomial_kernel as sk_poly

# 从 numpy_ml.utils.distance_metrics 模块中导入多个距离度量函数
from numpy_ml.utils.distance_metrics import (
    hamming,
    euclidean,
    chebyshev,
    manhattan,
    minkowski,
)
# 从 numpy_ml.utils.kernels 模块中导入 LinearKernel、PolynomialKernel、RBFKernel 类
from numpy_ml.utils.kernels import LinearKernel, PolynomialKernel, RBFKernel
# 从 numpy_ml.utils.data_structures 模块中导入 BallTree 类
from numpy_ml.utils.data_structures import BallTree
# 从 numpy_ml.utils.graphs 模块中导入多个图相关的类和函数
from numpy_ml.utils.graphs import (
    Edge,
    DiGraph,
    UndirectedGraph,
    random_DAG,
    random_unweighted_graph,
)

#######################################################################
#                               Kernels                               #
#######################################################################

# 定义测试线性核函数的函数
def test_linear_kernel(N=1):
    # 设置随机种子
    np.random.seed(12345)
    i = 0
    while i < N:
        # 生成随机数 N、M、C
        N = np.random.randint(1, 100)
        M = np.random.randint(1, 100)
        C = np.random.randint(1, 1000)

        # 生成随机矩阵 X 和 Y
        X = np.random.rand(N, C)
        Y = np.random.rand(M, C)

        # 计算自定义线性核函数的结果
        mine = LinearKernel()(X, Y)
        # 计算 sklearn 中线性核函数的结果
        gold = sk_linear(X, Y)

        # 使用 np.testing.assert_almost_equal 函数比较两个结果的近似程度
        np.testing.assert_almost_equal(mine, gold)
        # 打印测试通过信息
        print("PASSED")
        i += 1

# 定义测试多项式核函数的函数
def test_polynomial_kernel(N=1):
    np.random.seed(12345)
    i = 0
    while i < N:
        N = np.random.randint(1, 100)
        M = np.random.randint(1, 100)
        C = np.random.randint(1, 1000)
        gamma = np.random.rand()
        d = np.random.randint(1, 5)
        c0 = np.random.rand()

        X = np.random.rand(N, C)
        Y = np.random.rand(M, C)

        mine = PolynomialKernel(gamma=gamma, d=d, c0=c0)(X, Y)
        gold = sk_poly(X, Y, gamma=gamma, degree=d, coef0=c0)

        np.testing.assert_almost_equal(mine, gold)
        print("PASSED")
        i += 1

# 定义测试径向基核函数的函数
def test_radial_basis_kernel(N=1):
    np.random.seed(12345)
    i = 0
    # 当 i 小于 N 时执行循环
    while i < N:
        # 生成随机整数 N，范围在 [1, 100)
        N = np.random.randint(1, 100)
        # 生成随机整数 M，范围在 [1, 100)
        M = np.random.randint(1, 100)
        # 生成随机整数 C，范围在 [1, 1000)
        C = np.random.randint(1, 1000)
        # 生成随机浮点数 gamma，范围在 [0, 1)
        gamma = np.random.rand()

        # 生成 N 行 C 列的随机数组 X
        X = np.random.rand(N, C)
        # 生成 M 行 C 列的随机数组 Y
        Y = np.random.rand(M, C)

        # sklearn (gamma) <-> mine (sigma) 转换:
        # gamma = 1 / (2 * sigma^2)
        # sigma = np.sqrt(1 / 2 * gamma)

        # 使用 RBFKernel 类计算 RBF 核函数值，传入参数 sigma
        mine = RBFKernel(sigma=np.sqrt(1 / (2 * gamma)))(X, Y)
        # 使用 sk_rbf 函数计算 RBF 核函数值，传入参数 gamma
        gold = sk_rbf(X, Y, gamma=gamma)

        # 使用 np.testing.assert_almost_equal 函数检查 mine 和 gold 是否近似相等
        np.testing.assert_almost_equal(mine, gold)
        # 打印 "PASSED" 表示测试通过
        print("PASSED")
        # i 自增 1
        i += 1
# 距离度量函数的测试

# 测试欧几里德距离函数
def test_euclidean(N=1):
    # 设置随机种子
    np.random.seed(12345)
    i = 0
    while i < N:
        # 生成随机长度的向量
        N = np.random.randint(1, 100)
        x = np.random.rand(N)
        y = np.random.rand(N)
        # 计算自定义的欧几里德距离
        mine = euclidean(x, y)
        # 使用 SciPy 库计算欧几里德距离
        theirs = scipy.spatial.distance.euclidean(x, y)
        # 断言两者的结果几乎相等
        np.testing.assert_almost_equal(mine, theirs)
        print("PASSED")
        i += 1

# 测试汉明距离函数
def test_hamming(N=1):
    np.random.seed(12345)
    i = 0
    while i < N:
        N = np.random.randint(1, 100)
        # 生成随机整数向量
        x = (np.random.rand(N) * 100).round().astype(int)
        y = (np.random.rand(N) * 100).round().astype(int)
        # 计算自定义的汉明距离
        mine = hamming(x, y)
        # 使用 SciPy 库计算汉明距离
        theirs = scipy.spatial.distance.hamming(x, y)
        # 断言两者的结果几乎相等
        np.testing.assert_almost_equal(mine, theirs)
        print("PASSED")
        i += 1

# 测试闵可夫斯基距离函数
def test_minkowski(N=1):
    np.random.seed(12345)
    i = 0
    while i < N:
        N = np.random.randint(1, 100)
        p = 1 + np.random.rand() * 10
        x = np.random.rand(N)
        y = np.random.rand(N)
        # 计算自定义的闵可夫斯基距离
        mine = minkowski(x, y, p)
        # 使用 SciPy 库计算闵可夫斯基距离
        theirs = scipy.spatial.distance.minkowski(x, y, p)
        # 断言两者的结果几乎相等
        np.testing.assert_almost_equal(mine, theirs)
        print("PASSED")
        i += 1

# 测试切比雪夫距离函数
def test_chebyshev(N=1):
    np.random.seed(12345)
    i = 0
    while i < N:
        N = np.random.randint(1, 100)
        x = np.random.rand(N)
        y = np.random.rand(N)
        # 计算自定义的切比雪夫距离
        mine = chebyshev(x, y)
        # 使用 SciPy 库计算切比雪夫距离
        theirs = scipy.spatial.distance.chebyshev(x, y)
        # 断言两者的结果几乎相等
        np.testing.assert_almost_equal(mine, theirs)
        print("PASSED")
        i += 1

# 测试曼哈顿距离函数
def test_manhattan(N=1):
    np.random.seed(12345)
    i = 0
    # 当 i 小于 N 时执行循环
    while i < N:
        # 生成一个随机整数 N，范围在 [1, 100)
        N = np.random.randint(1, 100)
        # 生成一个包含 N 个随机浮点数的数组 x
        x = np.random.rand(N)
        # 生成一个包含 N 个随机浮点数的数组 y
        y = np.random.rand(N)
        # 调用 manhattan 函数计算曼哈顿距离
        mine = manhattan(x, y)
        # 调用 scipy 库中的 cityblock 函数计算曼哈顿距离
        theirs = scipy.spatial.distance.cityblock(x, y)
        # 使用 np.testing.assert_almost_equal 函数比较两个距离的近似相等性
        np.testing.assert_almost_equal(mine, theirs)
        # 打印 "PASSED" 表示测试通过
        print("PASSED")
        # i 自增
        i += 1
# 定义一个函数用于测试 BallTree 数据结构
def test_ball_tree(N=1):
    # 设置随机种子
    np.random.seed(12345)
    # 初始化计数器 i
    i = 0
    # 循环 N 次
    while i < N:
        # 生成随机数 N 和 M
        N = np.random.randint(2, 100)
        M = np.random.randint(2, 100)
        # 生成随机数 k
        k = np.random.randint(1, N)
        # 生成随机数 ls
        ls = np.min([np.random.randint(1, 10), N - 1])

        # 生成随机数据矩阵 X
        X = np.random.rand(N, M)
        # 创建 BallTree 对象 BT
        BT = BallTree(leaf_size=ls, metric=euclidean)
        # 对数据矩阵 X 进行拟合
        BT.fit(X)

        # 生成随机向量 x
        x = np.random.rand(M)
        # 获取最近的 k 个邻居
        mine = BT.nearest_neighbors(k, x)
        # 断言返回的邻居数量等于 k
        assert len(mine) == k

        # 提取邻居的键和距离
        mine_neighb = np.array([n.key for n in mine])
        mine_dist = np.array([n.distance for n in mine])

        # 对距离进行排序
        sort_ix = np.argsort(mine_dist)
        mine_dist = mine_dist[sort_ix]
        mine_neighb = mine_neighb[sort_ix]

        # 创建 scikit-learn 的 BallTree 对象 sk
        sk = sk_BallTree(X, leaf_size=ls)
        # 使用 sk 查询最近的 k 个邻居
        theirs_dist, ind = sk.query(x.reshape(1, -1), k=k)
        sort_ix = np.argsort(theirs_dist.flatten())

        theirs_dist = theirs_dist.flatten()[sort_ix]
        theirs_neighb = X[ind.flatten()[sort_ix]]

        # 断言我的结果与 scikit-learn 的结果一致
        for j in range(len(theirs_dist)):
            np.testing.assert_almost_equal(mine_neighb[j], theirs_neighb[j])
            np.testing.assert_almost_equal(mine_dist[j], theirs_dist[j])

        # 打印测试通过信息
        print("PASSED")
        # 更新计数器 i
        i += 1


# 将 networkx 图转换为自定义图表示
def from_networkx(G_nx):
    V = list(G_nx.nodes)
    edges = list(G_nx.edges)
    # 检查是否是带权重的图
    is_weighted = "weight" in G_nx[edges[0][0]][edges[0][1]]

    E = []
    for e in edges:
        if is_weighted:
            # 如果是带权重的边，则添加带权重的边
            E.append(Edge(e[0], e[1], G_nx[e[0]][e[1]]["weight"]))
        else:
            # 如果不带权重，则添加不带权重的边
            E.append(Edge(e[0], e[1]))
    # 如果输入的图 G_nx 是有向图，则返回一个有向图对象 DiGraph(V, E)
    # 否则返回一个无向图对象 UndirectedGraph(V, E)
    return DiGraph(V, E) if nx.is_directed(G_nx) else UndirectedGraph(V, E)
# 将自定义的图形表示转换为 networkx 图形
def to_networkx(G):
    # 如果图是有向的，则创建有向图，否则创建无向图
    G_nx = nx.DiGraph() if G.is_directed else nx.Graph()
    # 获取图中所有顶点
    V = list(G._V2I.keys())
    # 将所有顶点添加到 networkx 图中
    G_nx.add_nodes_from(V)

    for v in V:
        # 获取顶点在图中的索引
        fr_i = G._V2I[v]
        # 获取与该顶点相连的边
        edges = G._G[fr_i]

        for edge in edges:
            # 将边添加到 networkx 图中，包括权重信息
            G_nx.add_edge(edge.fr, edge.to, weight=edge._w)
    return G_nx


def test_all_paths(N=1):
    np.random.seed(12345)
    i = 0
    while i < N:
        # 生成随机概率值
        p = np.random.rand()
        # 生成随机值来确定图是否有向
        directed = np.random.rand() < 0.5
        # 创建一个随机无权重图
        G = random_unweighted_graph(n_vertices=5, edge_prob=p, directed=directed)

        nodes = G._I2V.keys()
        G_nx = to_networkx(G)

        # 对于每个图，测试所有起点和终点顶点对的 all_paths 方法
        # 注意图不一定是连通的，所以许多路径可能为空
        for s_i in nodes:
            for e_i in nodes:
                if s_i == e_i:
                    continue

                # 获取自定义图中的所有路径
                paths = G.all_paths(s_i, e_i)
                # 获取 networkx 图中的所有简单路径
                paths_nx = nx.all_simple_paths(G_nx, source=s_i, target=e_i, cutoff=10)

                # 对路径进行排序
                paths = sorted(paths)
                paths_nx = sorted(list(paths_nx))

                # 断言两个路径列表是否相等
                for p1, p2 in zip(paths, paths_nx):
                    np.testing.assert_array_equal(p1, p2)

                print("PASSED")
                i += 1


def test_random_DAG(N=1):
    np.random.seed(12345)
    i = 0
    while i < N:
        # 生成指定范围内的随机概率值
        p = np.random.uniform(0.25, 1)
        # 生成指定范围内的随机顶点数量
        n_v = np.random.randint(5, 50)

        # 创建一个随机有向无环图
        G = random_DAG(n_v, p)
        G_nx = to_networkx(G)

        # 断言 networkx 图是否是有向无环图
        assert nx.is_directed_acyclic_graph(G_nx)
        print("PASSED")
        i += 1


def test_topological_ordering(N=1):
    np.random.seed(12345)
    i = 0
    # 当 i 小于 N 时执行循环
    while i < N:
        # 生成一个介于 0.25 和 1 之间的随机数
        p = np.random.uniform(0.25, 1)
        # 生成一个介于 5 和 10 之间的随机整数
        n_v = np.random.randint(5, 10)

        # 生成一个随机有向无环图
        G = random_DAG(n_v, p)
        # 将生成的图转换为 NetworkX 图
        G_nx = to_networkx(G)

        # 如果转换后的图是有向无环图
        if nx.is_directed_acyclic_graph(G_nx):
            # 获取图的拓扑排序
            topo_order = G.topological_ordering()

            # 测试拓扑排序
            seen_it = set()
            for n_i in topo_order:
                seen_it.add(n_i)
                # 断言：对于每个节点 n_i，其邻居节点 c_i 不在已经遍历过的节点集合中
                assert any([c_i in seen_it for c_i in G.get_neighbors(n_i)]) == False

            # 打印 "PASSED" 表示测试通过
            print("PASSED")
            # 增加 i 的值，继续下一次循环
            i += 1
# 测试图是否为无环图
def test_is_acyclic(N=1):
    # 设置随机种子
    np.random.seed(12345)
    # 初始化计数器
    i = 0
    # 循环N次
    while i < N:
        # 生成随机概率值
        p = np.random.rand()
        # 生成随机布尔值，表示是否为有向图
        directed = np.random.rand() < 0.5
        # 生成一个随机无权图
        G = random_unweighted_graph(n_vertices=10, edge_prob=p, directed=True)
        # 将生成的图转换为 NetworkX 图
        G_nx = to_networkx(G)

        # 断言当前图是否为无环图，与 NetworkX 中的判断结果进行比较
        assert G.is_acyclic() == nx.is_directed_acyclic_graph(G_nx)
        # 打印测试通过信息
        print("PASSED")
        # 更新计数器
        i += 1
```