# `D:\src\scipysrc\scipy\benchmarks\benchmarks\sparse_csgraph.py`

```
"""benchmarks for the scipy.sparse.csgraph module"""
# 导入所需的库
import numpy as np
import scipy.sparse

# 从本地的 common 模块中安全导入 Benchmark 和 safe_import 函数
from .common import Benchmark, safe_import

# 使用 safe_import 上下文管理器安全导入 laplacian 函数
with safe_import():
    from scipy.sparse.csgraph import laplacian


# 创建 Laplacian 类，继承自 Benchmark 类
class Laplacian(Benchmark):
    # 定义参数列表
    params = [
        [30, 300, 900],        # n 的取值
        ['dense', 'coo', 'csc', 'csr', 'dia'],  # format 的取值
        [True, False]          # normed 的取值
    ]
    # 定义参数名称
    param_names = ['n', 'format', 'normed']

    # 设置方法，在每次运行 benchmark 前调用
    def setup(self, n, format, normed):
        # 创建稀疏矩阵数据，密度为 0.5，随机种子为 42
        data = scipy.sparse.rand(9, n, density=0.5, random_state=42).toarray()
        # 将数据堆叠起来形成一个 18xN 的数据集
        data = np.vstack((data, data))
        # 创建对角线的列表
        diags = list(range(-9, 0)) + list(range(1, 10))
        # 使用 spdiags 函数创建稀疏对角矩阵 A
        A = scipy.sparse.spdiags(data, diags, n, n)
        # 根据 format 的取值选择将 A 转换为密集矩阵或者其他格式的稀疏矩阵
        if format == 'dense':
            self.A = A.toarray()
        else:
            self.A = A.asformat(format)

    # 定义用于计时 laplacian 函数执行时间的方法
    def time_laplacian(self, n, format, normed):
        # 调用 laplacian 函数计算 Laplacian 矩阵
        laplacian(self.A, normed=normed)
```