# `D:\src\scipysrc\scipy\benchmarks\benchmarks\linalg_sqrtm.py`

```
""" Benchmark linalg.sqrtm for various blocksizes.

"""
import numpy as np

# 导入Benchmark类和safe_import函数，用于性能测试和安全导入
from .common import Benchmark, safe_import

# 使用safe_import安全导入scipy.linalg模块
with safe_import():
    import scipy.linalg


class Sqrtm(Benchmark):
    # 参数设置：数据类型、矩阵大小n、块大小blocksize
    params = [
        ['float64', 'complex128'],  # 测试数据类型：双精度浮点数、复数
        [64, 256],  # 测试矩阵大小n：64x64、256x256
        [32, 64, 256]  # 测试块大小blocksize：32、64、256
    ]
    param_names = ['dtype', 'n', 'blocksize']  # 参数名称列表

    def setup(self, dtype, n, blocksize):
        n = int(n)  # 将n转换为整数
        dtype = np.dtype(dtype)  # 转换数据类型为numpy的dtype
        blocksize = int(blocksize)  # 将blocksize转换为整数
        A = np.random.rand(n, n)  # 创建一个大小为n x n的随机数矩阵A
        if dtype == np.complex128:
            A = A + 1j*np.random.rand(n, n)  # 如果数据类型是complex128，则A为复数矩阵
        self.A = A  # 将A存储为对象的属性

        if blocksize > n:
            raise NotImplementedError()  # 如果块大小大于矩阵大小n，抛出NotImplementedError异常

    def time_sqrtm(self, dtype, n, blocksize):
        # 测试scipy.linalg.sqrtm函数的性能
        scipy.linalg.sqrtm(self.A, disp=False, blocksize=blocksize)
```