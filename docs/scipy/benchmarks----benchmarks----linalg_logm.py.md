# `D:\src\scipysrc\scipy\benchmarks\benchmarks\linalg_logm.py`

```
""" Benchmark linalg.logm for various blocksizes.

"""
# 导入必要的库
import numpy as np
from .common import Benchmark, safe_import

# 使用安全导入上下文管理器导入 scipy.linalg
with safe_import():
    import scipy.linalg

# 定义 Logm 类，继承自 Benchmark 类
class Logm(Benchmark):
    # 参数定义
    params = [
        ['float64', 'complex128'],  # 数据类型为 float64 或 complex128
        [64, 256],                  # 方阵大小 n 可选值为 64 或 256
        ['gen', 'her', 'pos']       # 矩阵结构可以是一般性 (gen)，Hermitian (her)，正定 (pos)
    ]
    param_names = ['dtype', 'n', 'structure']  # 参数名列表

    # 初始化方法，设置矩阵 A 的值
    def setup(self, dtype, n, structure):
        n = int(n)               # 将 n 转换为整数类型
        dtype = np.dtype(dtype)  # 使用 numpy 定义的数据类型

        A = np.random.rand(n, n)  # 生成随机 n x n 矩阵 A
        if dtype == np.complex128:
            A = A + 1j*np.random.rand(n, n)  # 如果数据类型是 complex128，则生成复数矩阵

        # 根据结构参数调整矩阵 A
        if structure == 'pos':
            A = A @ A.T.conj()      # 正定矩阵 A = A @ A^H
        elif structure == 'her':
            A = A + A.T.conj()      # Hermitian 矩阵 A = A + A^H

        self.A = A  # 将生成的矩阵 A 赋值给实例变量 self.A

    # 测试 logm 函数的性能
    def time_logm(self, dtype, n, structure):
        scipy.linalg.logm(self.A, disp=False)  # 调用 scipy.linalg 中的 logm 函数，计算矩阵 self.A 的对数
```