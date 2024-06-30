# `D:\src\scipysrc\scipy\benchmarks\benchmarks\sparse_linalg_solve.py`

```
"""
Check the speed of the conjugate gradient solver.
"""
# 导入必要的库
import numpy as np
from numpy.testing import assert_equal

# 从本地文件中导入 Benchmark 和 safe_import 函数
from .common import Benchmark, safe_import

# 安全导入以下函数和模块
with safe_import():
    from scipy import linalg, sparse
    from scipy.sparse.linalg import cg, minres, gmres, tfqmr, spsolve
with safe_import():
    from scipy.sparse.linalg import lgmres
with safe_import():
    from scipy.sparse.linalg import gcrotmk

# 创建一个稀疏的一维 Poisson 方程矩阵
def _create_sparse_poisson1d(n):
    # 构建 Gilbert Strang 最喜欢的矩阵
    # 参考：http://www-math.mit.edu/~gs/PIX/cupcakematrix.jpg
    P1d = sparse.diags([[-1]*(n-1), [2]*n, [-1]*(n-1)], [-1, 0, 1])
    assert_equal(P1d.shape, (n, n))  # 断言确保矩阵形状正确
    return P1d

# 创建一个稀疏的二维 Poisson 方程矩阵
def _create_sparse_poisson2d(n):
    P1d = _create_sparse_poisson1d(n)
    P2d = sparse.kronsum(P1d, P1d)
    assert_equal(P2d.shape, (n*n, n*n))  # 断言确保矩阵形状正确
    return P2d.tocsr()

# Benchmark 类用于性能测试
class Bench(Benchmark):
    # 参数化的参数和求解器
    params = [
        [4, 6, 10, 16, 25, 40, 64, 100],  # n 的取值
        ['dense', 'spsolve', 'cg', 'minres', 'gmres', 'lgmres', 'gcrotmk', 'tfqmr']  # 求解器的选择
    ]
    # 将求解器名称映射到实际的求解函数
    mapping = {'spsolve': spsolve, 'cg': cg, 'minres': minres, 'gmres': gmres,
               'lgmres': lgmres, 'gcrotmk': gcrotmk, 'tfqmr': tfqmr}
    param_names = ['(n,n)', 'solver']  # 参数名称为 n 和 solver

    # 初始化函数，设置测试环境
    def setup(self, n, solver):
        # 如果选择了 'dense' 求解器且 n >= 25，则抛出未实现错误
        if solver == 'dense' and n >= 25:
            raise NotImplementedError()

        self.b = np.ones(n*n)  # 创建一个长度为 n*n 的全为 1 的向量
        self.P_sparse = _create_sparse_poisson2d(n)  # 创建稀疏的二维 Poisson 矩阵

        # 如果选择了 'dense' 求解器，则创建该矩阵的密集版本
        if solver == 'dense':
            self.P_dense = self.P_sparse.toarray()

    # 定义时间测试函数，执行求解器的运行时间测试
    def time_solve(self, n, solver):
        # 如果选择了 'dense' 求解器，则使用 linalg.solve 求解
        if solver == 'dense':
            linalg.solve(self.P_dense, self.b)
        else:
            self.mapping[solver](self.P_sparse, self.b)  # 否则使用选定的稀疏求解器求解

# Lgmres 类用于测试 lgmres 方法
class Lgmres(Benchmark):
    # 参数化的参数 n 和 m
    params = [
        [10, 50, 100, 1000, 10000],  # n 的取值
        [10, 30, 60, 90, 180],  # m 的取值
    ]
    param_names = ['n', 'm']  # 参数名称为 n 和 m

    # 初始化函数，设置测试环境
    def setup(self, n, m):
        rng = np.random.default_rng(1234)
        self.A = sparse.eye(n, n) + sparse.rand(n, n, density=0.01, random_state=rng)  # 创建稀疏矩阵 A
        self.b = np.ones(n)  # 创建长度为 n 的全为 1 的向量

    # 定义时间测试函数，执行 lgmres 方法的运行时间测试
    def time_inner(self, n, m):
        lgmres(self.A, self.b, inner_m=m, maxiter=1)  # 调用 lgmres 方法进行求解
```