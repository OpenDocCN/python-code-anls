# `D:\src\scipysrc\scipy\benchmarks\benchmarks\sparse_csgraph_matching.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import scipy.sparse  # 导入 SciPy 稀疏矩阵模块
from scipy.spatial.distance import cdist  # 导入 SciPy 距离计算模块

from .common import Benchmark, safe_import  # 导入自定义模块和函数

with safe_import():  # 使用安全导入上下文管理器，确保依赖模块可用
    from scipy.sparse.csgraph import maximum_bipartite_matching,\
        min_weight_full_bipartite_matching  # 导入最大二分匹配和最小权重全二分匹配函数


class MaximumBipartiteMatching(Benchmark):  # 定义最大二分匹配类，继承自 Benchmark 类
    params = [[5000, 7500, 10000], [0.0001, 0.0005, 0.001]]  # 参数设置：n 和 density 的不同取值
    param_names = ['n', 'density']  # 参数名称：n 和 density

    def setup(self, n, density):  # 设置函数，用于生成测试所需的二分图
        # 创建随机稀疏矩阵。虽然可以使用 scipy.sparse.rand 来实现，但简单使用 np.random
        # 并忽略重复项会更快一些。
        rng = np.random.default_rng(42)  # 使用种子为 42 的随机数生成器
        d = rng.integers(0, n, size=(int(n*n*density), 2))  # 生成大小为 (n*n*density) 的随机整数对
        graph = scipy.sparse.csr_matrix((np.ones(len(d)), (d[:, 0], d[:, 1])),  # 创建 CSR 格式的稀疏矩阵
                                        shape=(n, n))
        self.graph = graph  # 将生成的稀疏矩阵赋值给类成员变量 self.graph

    def time_maximum_bipartite_matching(self, n, density):  # 最大二分匹配的性能测试函数
        maximum_bipartite_matching(self.graph)  # 调用最大二分匹配函数对 self.graph 进行匹配


# 用于基准测试最小权重全二分匹配的函数需要依赖 Burkard, Dell'Amico, Martello 的一些类定义。
def random_uniform(shape, rng):  # 生成均匀随机稀疏矩阵的函数
    return scipy.sparse.csr_matrix(rng.uniform(1, 100, shape))


def random_uniform_sparse(shape, rng):  # 生成稀疏均匀随机稀疏矩阵的函数
    return scipy.sparse.random(shape[0], shape[1],
                               density=0.1, format='csr', random_state=rng)


def random_uniform_integer(shape, rng):  # 生成整数均匀随机稀疏矩阵的函数
    return scipy.sparse.csr_matrix(rng.integers(1, 1000, shape))


def random_geometric(shape, rng):  # 生成几何随机稀疏矩阵的函数
    P = rng.integers(1, 1000, size=(shape[0], 2))
    Q = rng.integers(1, 1000, size=(shape[1], 2))
    return scipy.sparse.csr_matrix(cdist(P, Q, 'sqeuclidean'))


def random_two_cost(shape, rng):  # 生成具有两种成本的随机稀疏矩阵的函数
    return scipy.sparse.csr_matrix(rng.choice((1, 1000000), shape))


def machol_wien(shape, rng):  # 生成 Machol--Wien 实例的函数，较其他例子更难
    # 由于 Machol--Wien 实例较难，我们将实例的大小缩小了 5 倍。
    return scipy.sparse.csr_matrix(
        np.outer(np.arange(shape[0]//5) + 1, np.arange(shape[1]//5) + 1))


class MinWeightFullBipartiteMatching(Benchmark):  # 定义最小权重全二分匹配类，继承自 Benchmark 类

    sizes = range(100, 401, 100)  # 定义不同矩阵大小的范围
    param_names = ['shapes', 'input_type']  # 参数名称：形状和输入类型
    params = [
        [(i, i) for i in sizes] + [(i, 2 * i) for i in sizes],  # 不同大小的形状组合
        ['random_uniform', 'random_uniform_sparse', 'random_uniform_integer',
         'random_geometric', 'random_two_cost', 'machol_wien']  # 不同的输入类型
    ]

    def setup(self, shape, input_type):  # 设置函数，用于生成测试所需的二分图
        rng = np.random.default_rng(42)  # 使用种子为 42 的随机数生成器
        input_func = {'random_uniform': random_uniform,
                      'random_uniform_sparse': random_uniform_sparse,
                      'random_uniform_integer': random_uniform_integer,
                      'random_geometric': random_geometric,
                      'random_two_cost': random_two_cost,
                      'machol_wien': machol_wien}[input_type]  # 根据输入类型选择相应的生成函数

        self.biadjacency_matrix = input_func(shape, rng)  # 生成具体的二分图矩阵
    # 定义一个方法 `time_evaluation`，接受任意数量的参数
    def time_evaluation(self, *args):
        # 调用 `min_weight_full_bipartite_matching` 函数，传入 `self.biadjacency_matrix` 参数
        min_weight_full_bipartite_matching(self.biadjacency_matrix)
```