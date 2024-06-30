# `D:\src\scipysrc\scipy\benchmarks\benchmarks\sparse_csgraph_dijkstra.py`

```
"""benchmarks for the scipy.sparse.csgraph module"""
# 导入必要的库
import numpy as np
import scipy.sparse

# 导入Benchmark类和safe_import函数，来自.common模块
from .common import Benchmark, safe_import

# 定义一个上下文管理器，用于安全导入
with safe_import():
    # 从scipy.sparse.csgraph模块导入dijkstra函数
    from scipy.sparse.csgraph import dijkstra

# 定义一个Benchmark的子类Dijkstra，用于测试Dijkstra算法的性能
class Dijkstra(Benchmark):
    # 参数化测试用例
    params = [
        [30, 300, 900],    # 不同的图大小n
        [True, False],     # 是否仅计算最短路径min_only
        ['random', 'star'] # 不同的图结构format
    ]
    param_names = ['n', 'min_only', 'format']

    # 设置测试用例的准备阶段
    def setup(self, n, min_only, format):
        # 使用固定种子1234创建随机数生成器rng
        rng = np.random.default_rng(1234)
        if format == 'random':
            # 创建一个随机的稀疏连接矩阵data，密度为0.2
            data = scipy.sparse.rand(n, n, density=0.2, format='csc',
                                     random_state=42, dtype=np.bool_)
            # 将对角线元素设为False
            data.setdiag(np.zeros(n, dtype=np.bool_))
            self.data = data
        elif format == 'star':
            # 创建一个星型图的稀疏矩阵self.data
            rows = [0 for i in range(n - 1)] + [i + 1 for i in range(n - 1)]
            cols = [i + 1 for i in range(n - 1)] + [0 for i in range(n - 1)]
            weights = [i + 1 for i in range(n - 1)] * 2
            self.data = scipy.sparse.csr_matrix((weights, (rows, cols)),
                                                shape=(n, n))
        # 随机选择一些顶点作为起始点indices
        v = np.arange(n)
        rng.shuffle(v)
        self.indices = v[:int(n*.1)]

    # 测试函数，用于测试多次调用Dijkstra算法的性能
    def time_dijkstra_multi(self, n, min_only, format):
        dijkstra(self.data,
                 directed=False,
                 indices=self.indices,
                 min_only=min_only)
```