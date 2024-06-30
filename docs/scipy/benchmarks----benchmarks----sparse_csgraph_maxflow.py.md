# `D:\src\scipysrc\scipy\benchmarks\benchmarks\sparse_csgraph_maxflow.py`

```
# 导入必要的库 numpy 和 scipy.sparse
import numpy as np
import scipy.sparse

# 从当前目录的 common 模块中导入 Benchmark 和 safe_import 函数
from .common import Benchmark, safe_import

# 使用 safe_import 上下文管理器安全导入 scipy.sparse.csgraph 中的 maximum_flow 函数
with safe_import():
    from scipy.sparse.csgraph import maximum_flow

# MaximumFlow 类继承自 Benchmark 类
class MaximumFlow(Benchmark):
    # 定义参数化测试的参数
    params = [[200, 500, 1500], [0.1, 0.3, 0.5]]
    param_names = ['n', 'density']

    # 设置方法，在测试开始前创建稀疏矩阵 data
    def setup(self, n, density):
        # 使用 scipy.sparse.rand 创建稀疏随机矩阵 data，格式为 lil_matrix
        data = (scipy.sparse.rand(n, n, density=density, format='lil',
                                  random_state=42) * 100).astype(np.int32)
        # 将对角线元素设置为 0
        data.setdiag(np.zeros(n, dtype=np.int32))
        # 转换为 CSR 格式的稀疏矩阵，并存储在 self.data 中
        self.data = scipy.sparse.csr_matrix(data)

    # 定义测试方法，测量 maximum_flow 函数的执行时间
    def time_maximum_flow(self, n, density):
        # 调用 maximum_flow 计算稀疏矩阵 self.data 中从节点 0 到节点 n-1 的最大流
        maximum_flow(self.data, 0, n - 1)
```