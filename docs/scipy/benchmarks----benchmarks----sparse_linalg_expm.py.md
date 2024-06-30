# `D:\src\scipysrc\scipy\benchmarks\benchmarks\sparse_linalg_expm.py`

```
"""benchmarks for the scipy.sparse.linalg._expm_multiply module"""
# 导入数学库
import math

# 导入numpy库并重命名为np
import numpy as np
# 从当前包中导入Benchmark类和safe_import函数
from .common import Benchmark, safe_import

# 使用safe_import装饰器导入scipy.linalg，sp_expm和expm_multiply函数
with safe_import():
    import scipy.linalg
    from scipy.sparse.linalg import expm as sp_expm
    from scipy.sparse.linalg import expm_multiply


# 定义生成随机稀疏CSR矩阵的函数
def random_sparse_csr(m, n, nnz_per_row):
    # 从scipy.sparse基准中复制代码
    rows = np.arange(m).repeat(nnz_per_row)
    cols = np.random.randint(0, n, size=nnz_per_row*m)
    vals = np.random.random_sample(m*nnz_per_row)
    M = scipy.sparse.coo_matrix((vals, (rows, cols)), (m, n), dtype=float)
    return M.tocsr()


# 定义生成随机稀疏CSC矩阵的函数
def random_sparse_csc(m, n, nnz_per_row, rng):
    # 从scipy.sparse基准中复制代码
    rows = np.arange(m).repeat(nnz_per_row)
    cols = rng.integers(0, n, size=nnz_per_row*m)
    vals = rng.random(m*nnz_per_row)
    M = scipy.sparse.coo_matrix((vals, (rows, cols)), (m, n), dtype=float)
    # 使用CSC格式而不是CSR格式，因为使用CSR格式时，稀疏LU分解会引发警告。
    return M.tocsc()


# 定义Benchmark类的子类ExpmMultiply
class ExpmMultiply(Benchmark):
    # 设置函数，初始化参数
    def setup(self):
        self.n = 2000
        self.i = 100
        self.j = 200
        nnz_per_row = 25
        # 生成随机稀疏CSR矩阵
        self.A = random_sparse_csr(self.n, self.n, nnz_per_row)

    # 定义时间测量expm_multiply函数的方法
    def time_expm_multiply(self):
        # 计算稀疏矩阵的指定列j的expm结果
        v = np.zeros(self.n, dtype=float)
        v[self.j] = 1
        A_expm_col_j = expm_multiply(self.A, v)
        A_expm_col_j[self.i]


# 定义Benchmark类的子类Expm
class Expm(Benchmark):
    # 定义参数和参数名称
    params = [
        [30, 100, 300],
        ['sparse', 'dense']
    ]
    param_names = ['n', 'format']

    # 设置函数，根据参数生成数据
    def setup(self, n, format):
        rng = np.random.default_rng(1234)

        # 让每行非零条目的数量按矩阵阶数的对数增长
        nnz_per_row = int(math.ceil(math.log(n)))

        # 测量生成随机稀疏矩阵的时间
        self.A_sparse = random_sparse_csc(n, n, nnz_per_row, rng)

        # 将稀疏矩阵转换为稠密矩阵
        self.A_dense = self.A_sparse.toarray()

    # 定义时间测量expm函数的方法
    def time_expm(self, n, format):
        if format == 'sparse':
            sp_expm(self.A_sparse)
        elif format == 'dense':
            scipy.linalg.expm(self.A_dense)
```