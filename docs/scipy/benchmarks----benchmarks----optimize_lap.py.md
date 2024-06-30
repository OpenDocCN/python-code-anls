# `D:\src\scipysrc\scipy\benchmarks\benchmarks\optimize_lap.py`

```
# 导入必要的库和模块
from concurrent.futures import ThreadPoolExecutor, wait

# 导入 NumPy 库，并从本地模块 common 中导入 Benchmark 和 safe_import 函数
import numpy as np
from .common import Benchmark, safe_import

# 使用 safe_import 函数安全导入 scipy 库中的 linear_sum_assignment 和 spatial.distance 中的 cdist 函数
with safe_import():
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

# 定义一个函数，生成指定形状的服从均匀分布的随机数数组
def random_uniform(shape):
    return np.random.uniform(-20, 20, shape)

# 定义一个函数，生成指定形状的服从对数均匀分布的随机数数组
def random_logarithmic(shape):
    return 10**np.random.uniform(-20, 20, shape)

# 定义一个函数，生成指定形状的随机整数数组
def random_integer(shape):
    return np.random.randint(-1000, 1000, shape)

# 定义一个函数，生成指定形状的随机二进制数组（0或1）
def random_binary(shape):
    return np.random.randint(0, 2, shape)

# 定义一个函数，生成两个随机点集之间的平方欧氏距离矩阵
def random_spatial(shape):
    P = np.random.uniform(-1, 1, size=(shape[0], 2))
    Q = np.random.uniform(-1, 1, size=(shape[1], 2))
    return cdist(P, Q, 'sqeuclidean')

# 定义一个继承自 Benchmark 的类 LinearAssignment
class LinearAssignment(Benchmark):

    # 定义矩阵大小的范围
    sizes = range(100, 401, 100)
    # 定义不同形状的矩阵
    shapes = [(i, i) for i in sizes]
    shapes.extend([(i, 2 * i) for i in sizes])
    shapes.extend([(2 * i, i) for i in sizes])
    # 定义不同代价类型
    cost_types = ['uniform', 'spatial', 'logarithmic', 'integer', 'binary']
    # 参数名列表
    param_names = ['shape', 'cost_type']
    # 参数组合
    params = [shapes, cost_types]

    # 初始化方法，根据给定的形状和代价类型选择相应的随机数生成函数
    def setup(self, shape, cost_type):
        cost_func = {'uniform': random_uniform,
                     'spatial': random_spatial,
                     'logarithmic': random_logarithmic,
                     'integer': random_integer,
                     'binary': random_binary}[cost_type]
        # 生成代价矩阵
        self.cost_matrix = cost_func(shape)

    # 性能评估方法，执行线性分配算法并计时
    def time_evaluation(self, *args):
        linear_sum_assignment(self.cost_matrix)


# 定义一个继承自 Benchmark 的类 ParallelLinearAssignment
class ParallelLinearAssignment(Benchmark):
    # 固定矩阵形状
    shape = (100, 100)
    # 参数名列表
    param_names = ['threads']
    # 参数组合
    params = [[1, 2, 4]]

    # 初始化方法，生成多个具有固定形状的随机数矩阵
    def setup(self, threads):
        self.cost_matrices = [random_uniform(self.shape) for _ in range(20)]

    # 性能评估方法，使用线程池并发执行线性分配算法，并计时
    def time_evaluation(self, threads):
        with ThreadPoolExecutor(max_workers=threads) as pool:
            # 提交每个随机数矩阵的线性分配任务给线程池
            wait({pool.submit(linear_sum_assignment, cost_matrix)
                  for cost_matrix in self.cost_matrices})
```