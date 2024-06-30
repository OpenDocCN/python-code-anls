# `D:\src\scipysrc\scipy\benchmarks\benchmarks\sparse_linalg_svds.py`

```
import os
import numpy as np
from .common import Benchmark, safe_import

# 使用安全导入，确保依赖库安全导入或不导入
with safe_import():
    # 从 scipy.sparse.linalg 中导入 svds 函数
    from scipy.sparse.linalg import svds

# 定义 BenchSVDS 类，继承自 Benchmark 类
class BenchSVDS(Benchmark):
    # 参数设置：矩阵维度 k 的值
    params = [
        [25],
        # 待测试的问题名称列表
        ["abb313", "illc1033", "illc1850", "qh1484", "rbs480a", "tols4000",
         "well1033", "well1850", "west0479", "west2021"],
        # 待测试的求解器列表
        # TODO: 重新包含 propack
        ['arpack', 'lobpcg']  # 'propack' 失败（2023 年 8 月）
    ]
    # 参数名称列表
    param_names = ['k', 'problem', 'solver']

    # 初始化方法，设置测试环境
    def setup(self, k, problem, solver):
        # 获取当前文件所在目录的绝对路径
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # 构建数据文件的完整路径
        datafile = os.path.join(dir_path, "svds_benchmark_files",
                                "svds_benchmark_files.npz")
        # 使用 numpy 加载数据文件，允许 pickle 序列化
        matrices = np.load(datafile, allow_pickle=True)
        # 从加载的数据中获取问题对应的矩阵 A
        self.A = matrices[problem][()]

    # 测试方法，计算 svds 函数的执行时间
    def time_svds(self, k, problem, solver):
        # 设置随机数种子为 0，以保证结果可重复
        np.random.seed(0)
        # 调用 scipy.sparse.linalg 中的 svds 函数进行奇异值分解
        svds(self.A, k=k, solver=solver)
```