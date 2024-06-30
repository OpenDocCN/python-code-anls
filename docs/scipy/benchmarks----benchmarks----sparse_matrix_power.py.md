# `D:\src\scipysrc\scipy\benchmarks\benchmarks\sparse_matrix_power.py`

```
# 从.common模块中导入Benchmark类和safe_import函数
from .common import Benchmark, safe_import

# 使用safe_import上下文管理器，安全导入所需的模块
with safe_import():
    # 从scipy.sparse模块导入random函数
    from scipy.sparse import random

# 定义一个名为BenchMatrixPower的类，继承自Benchmark类
class BenchMatrixPower(Benchmark):
    # 定义参数列表，包括一个整数列表、一个包含一个整数的列表、一个浮点数密度列表
    params = [
        [0, 1, 2, 3, 8, 9],  # x的取值列表
        [1000],              # N的取值列表，只包含一个值1000
        [1e-6, 1e-3],        # density的取值列表
    ]
    # 定义参数名称列表，对应params中的三个参数
    param_names = ['x', 'N', 'density']

    # 设置方法，在每次测试之前调用，初始化self.A为一个稀疏随机矩阵
    def setup(self, x: int, N: int, density: float):
        self.A = random(N, N, density=density, format='csr')

    # 定义一个测试方法，测试矩阵的幂运算时间
    def time_matrix_power(self, x: int, N: int, density: float):
        self.A ** x
```