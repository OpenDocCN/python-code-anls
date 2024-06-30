# `D:\src\scipysrc\scipy\benchmarks\benchmarks\sparse_linalg_onenormest.py`

```
"""Compare the speed of exact one-norm calculation vs. its estimation.
"""
import numpy as np

from .common import Benchmark, safe_import

# 导入必要的库，并使用安全导入
with safe_import():
    import scipy.sparse
    import scipy.special  # import cycle workaround for some versions
    import scipy.sparse.linalg

# 定义一个 Benchmark 类用于性能测试
class BenchmarkOneNormEst(Benchmark):
    # 参数设置：n为矩阵大小，solver为求解器类型
    params = [
        [2, 3, 5, 10, 30, 100, 300, 500, 1000, 1e4, 1e5, 1e6],
        ['exact', 'onenormest']
    ]
    param_names = ['n', 'solver']

    # 初始化函数，生成随机矩阵用于测试
    def setup(self, n, solver):
        rng = np.random.default_rng(1234)
        nrepeats = 100
        shape = (int(n), int(n))

        if solver == 'exact' and n >= 300:
            # 如果是精确求解器并且矩阵较大，则跳过
            raise NotImplementedError()

        if n <= 1000:
            # 生成随机矩阵
            self.matrices = []
            for i in range(nrepeats):
                M = rng.standard_normal(shape)
                self.matrices.append(M)
        else:
            max_nnz = 100000
            nrepeats = 1

            self.matrices = []
            for i in range(nrepeats):
                # 生成稀疏矩阵
                M = scipy.sparse.rand(
                    shape[0],
                    shape[1],
                    min(max_nnz/(shape[0]*shape[1]), 1e-5),
                    random_state=rng,
                )
                self.matrices.append(M)

    # 计算 onenormest 函数的运行时间
    def time_onenormest(self, n, solver):
        if solver == 'exact':
            # 对于精确求解器，计算矩阵平方的一范数
            for M in self.matrices:
                M.dot(M)
                scipy.sparse.linalg._matfuncs._onenorm(M)
        elif solver == 'onenormest':
            # 对于估计求解器，计算矩阵平方的一范数估计值
            for M in self.matrices:
                scipy.sparse.linalg._matfuncs._onenormest_matrix_power(M, 2)

    # 保留旧的基准测试结果（如果更改基准测试，则删除此行）
    time_onenormest.version = (
        "f7b31b4bf5caa50d435465e78dab6e133f3c263a52c4523eec785446185fdb6f"
    )
```