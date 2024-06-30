# `D:\src\scipysrc\scipy\benchmarks\benchmarks\sparse_linalg_spsolve_triangular.py`

```
"""
Check the speed of the sparse triangular solve function.
"""
# 导入必要的库
import numpy as np
from numpy.testing import assert_equal

# 从当前目录的 common 模块中安全导入 Benchmark 和 safe_import 函数
from .common import Benchmark, safe_import

# 使用安全导入引入 scipy 库中的 sparse 和 sparse.linalg 中的 spsolve 和 spsolve_triangular 函数
with safe_import():
    from scipy import sparse
    from scipy.sparse.linalg import spsolve, spsolve_triangular

# 创建一个稀疏的一维 Poisson 方程系数矩阵的函数
def _create_sparse_poisson1d(n):
    # 构造 Gilbert Strang 的喜爱矩阵，并取其下三角部分
    # 参考：http://www-math.mit.edu/~gs/PIX/cupcakematrix.jpg
    P1d = sparse.diags([[-1]*(n-1), [2]*n, [-1]*(n-1)], [-1, 0, 1])
    assert_equal(P1d.shape, (n, n))  # 断言确保矩阵形状正确
    return P1d

# 创建一个稀疏的二维 Poisson 方程系数矩阵的函数，取其下三角部分并转换为 CSR 格式
def _create_sparse_poisson2d_half(n):
    P1d = _create_sparse_poisson1d(n)
    P2d = sparse.kronsum(P1d, P1d)
    assert_equal(P2d.shape, (n*n, n*n))  # 断言确保矩阵形状正确
    return sparse.tril(P2d).tocsr()

# 继承 Benchmark 类定义 Bench 类
class Bench(Benchmark):
    params = [
        [100,1000],  # 参数1: n 取 100 和 1000
        ["spsolve", "spsolve_triangular"],  # 参数2: 方法为 spsolve 或 spsolve_triangular
    ]
    param_names = ['(n,n)', "method"]  # 参数名称

    # 设置函数，在每次性能测试前创建所需的数据和对象
    def setup(self, n, method):
        self.b = np.ones(n*n)  # 创建长度为 n*n 的全为1的数组
        self.P_sparse = _create_sparse_poisson2d_half(n)  # 创建稀疏的二维 Poisson 矩阵

    # 性能测试函数，根据选择的方法调用相应的稀疏三角解法函数
    def time_solve(self, n, method):
        if method == "spsolve":
            spsolve(self.P_sparse, self.b)  # 使用 spsolve 解方程
        elif method == "spsolve_triangular":
            spsolve_triangular(self.P_sparse, self.b)  # 使用 spsolve_triangular 解方程
        else:
            raise NotImplementedError()  # 抛出未实现错误，以防万一
```