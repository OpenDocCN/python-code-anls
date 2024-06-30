# `D:\src\scipysrc\scipy\benchmarks\benchmarks\cluster.py`

```
# 导入 NumPy 库，并将其命名为 np
import numpy as np
# 导入 suppress_warnings 函数，用于抑制警告
from numpy.testing import suppress_warnings

# 从当前目录下的 common 模块中导入 Benchmark 类和 safe_import 函数
from .common import Benchmark, safe_import

# 使用 safe_import 上下文管理器来安全导入以下模块：linkage, kmeans, kmeans2, vq
with safe_import():
    # 从 scipy.cluster.hierarchy 模块导入 linkage 函数
    from scipy.cluster.hierarchy import linkage
    # 从 scipy.cluster.vq 模块导入 kmeans, kmeans2, vq 函数
    from scipy.cluster.vq import kmeans, kmeans2, vq

# 定义一个名为 HierarchyLinkage 的类，继承自 Benchmark 类
class HierarchyLinkage(Benchmark):
    # 参数列表，包含不同的聚类方法名称
    params = ['single', 'complete', 'average', 'weighted', 'centroid',
              'median', 'ward']
    # 参数名称列表，只有一个参数 method
    param_names = ['method']

    # 初始化方法
    def __init__(self):
        # 创建一个名为 rnd 的随机数生成器对象，种子为 0
        rnd = np.random.RandomState(0)
        # 创建一个形状为 (2000, 2) 的随机数组 X
        self.X = rnd.randn(2000, 2)

    # 测试方法 time_linkage，用于测试 linkage 函数的性能
    def time_linkage(self, method):
        # 调用 linkage 函数，传入参数 self.X 和 method
        linkage(self.X, method=method)

# 定义一个名为 KMeans 的类，继承自 Benchmark 类
class KMeans(Benchmark):
    # 参数列表，包含不同的聚类数量
    params = [2, 10, 50]
    # 参数名称列表，只有一个参数 k
    param_names = ['k']

    # 初始化方法
    def __init__(self):
        # 创建一个名为 rnd 的随机数生成器对象，种子为 0
        rnd = np.random.RandomState(0)
        # 创建一个形状为 (1000, 5) 的随机数组 obs
        self.obs = rnd.rand(1000, 5)

    # 测试方法 time_kmeans，用于测试 kmeans 函数的性能
    def time_kmeans(self, k):
        # 调用 kmeans 函数，传入参数 self.obs, k 和 iter=10
        kmeans(self.obs, k, iter=10)

# 定义一个名为 KMeans2 的类，继承自 Benchmark 类
class KMeans2(Benchmark):
    # 参数列表，包含不同的聚类数量和不同的初始化方法
    params = [[2, 10, 50], ['random', 'points', '++']]
    # 参数名称列表，分别为 k 和 init
    param_names = ['k', 'init']

    # 初始化方法
    def __init__(self):
        # 创建一个名为 rnd 的随机数生成器对象，种子为 0
        rnd = np.random.RandomState(0)
        # 创建一个形状为 (1000, 5) 的随机数组 obs
        self.obs = rnd.rand(1000, 5)

    # 测试方法 time_kmeans2，用于测试 kmeans2 函数的性能
    def time_kmeans2(self, k, init):
        # 使用 suppress_warnings 上下文管理器来抑制特定警告
        with suppress_warnings() as sup:
            # 设置过滤条件，过滤特定的 UserWarning 类型警告
            sup.filter(UserWarning,
                       "One of the clusters is empty. Re-run kmeans with a "
                       "different initialization")
            # 调用 kmeans2 函数，传入参数 self.obs, k, minit=init 和 iter=10
            kmeans2(self.obs, k, minit=init, iter=10)

# 定义一个名为 VQ 的类，继承自 Benchmark 类
class VQ(Benchmark):
    # 参数列表，包含不同的聚类数量和不同的数据类型
    params = [[2, 10, 50], ['float32', 'float64']]
    # 参数名称列表，分别为 k 和 dtype
    param_names = ['k', 'dtype']

    # 初始化方法
    def __init__(self):
        # 创建一个名为 rnd 的随机数生成器对象，种子为 0
        rnd = np.random.RandomState(0)
        # 创建一个形状为 (5000, 5) 的随机数组 data 和一个形状为 (50, 5) 的随机数组 cbook_source
        self.data = rnd.rand(5000, 5)
        self.cbook_source = rnd.rand(50, 5)

    # 设置方法，根据参数 k 和 dtype 设置观察数据 self.obs 和参考数据 self.cbook
    def setup(self, k, dtype):
        self.obs = self.data.astype(dtype)
        self.cbook = self.cbook_source[:k].astype(dtype)

    # 测试方法 time_vq，用于测试 vq 函数的性能
    def time_vq(self, k, dtype):
        # 调用 vq 函数，传入参数 self.obs 和 self.cbook
        vq(self.obs, self.cbook)
```