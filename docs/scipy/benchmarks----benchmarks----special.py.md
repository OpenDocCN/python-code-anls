# `D:\src\scipysrc\scipy\benchmarks\benchmarks\special.py`

```
# 导入 NumPy 库，并使用 np 别名
import numpy as np

# 从当前目录中的 common 模块导入 Benchmark、with_attributes 和 safe_import 函数
from .common import Benchmark, with_attributes, safe_import

# 使用 safe_import 函数安全地导入 scipy.special 中的 ai_zeros、bi_zeros、erf 和 expn 函数
with safe_import():
    from scipy.special import ai_zeros, bi_zeros, erf, expn

# 使用 safe_import 函数安全地导入 scipy.special 中的 comb 函数
with safe_import():
    # 由于 comb 函数不总是在 scipy.special 中，因此单独导入
    from scipy.special import comb

# 使用 safe_import 函数安全地导入 scipy.special 中的 loggamma 函数
with safe_import():
    from scipy.special import loggamma


# 定义 Airy 类，继承自 Benchmark 类
class Airy(Benchmark):
    
    # 定义 time_ai_zeros 方法
    def time_ai_zeros(self):
        # 调用 ai_zeros 函数，计算前 100000 个 Airy 函数的零点
        ai_zeros(100000)

    # 定义 time_bi_zeros 方法
    def time_bi_zeros(self):
        # 调用 bi_zeros 函数，计算前 100000 个修改过的 Airy 函数的零点
        bi_zeros(100000)


# 定义 Erf 类，继承自 Benchmark 类
class Erf(Benchmark):
    
    # 设置方法，初始化随机数数组 rand
    def setup(self, *args):
        self.rand = np.random.rand(100000)

    # 定义 time_real 方法，测试 erf 函数的性能
    def time_real(self, offset):
        # 调用 erf 函数，计算随机数数组 self.rand 加上给定偏移量 offset 的结果
        erf(self.rand + offset)

    # 设置 time_real 方法的参数和参数名称
    time_real.params = [0.0, 2.0]
    time_real.param_names = ['offset']


# 定义 Comb 类，继承自 Benchmark 类
class Comb(Benchmark):
    
    # 设置方法，初始化数组 N 和 k
    def setup(self, *args):
        self.N = np.arange(1, 1000, 50)
        self.k = np.arange(1, 1000, 50)

    # 使用 with_attributes 装饰器设置 time_comb_exact 方法的参数和参数名称
    @with_attributes(params=[(10, 100, 1000, 10000), (1, 10, 100)],
                     param_names=['N', 'k'])
    # 定义 time_comb_exact 方法，测试 comb 函数在精确模式下的性能
    def time_comb_exact(self, N, k):
        comb(N, k, exact=True)

    # 定义 time_comb_float 方法，测试 comb 函数在浮点数模式下的性能
    def time_comb_float(self):
        comb(self.N[:,None], self.k[None,:])


# 定义 Loggamma 类，继承自 Benchmark 类
class Loggamma(Benchmark):
    
    # 设置方法，初始化大数值网格 self.large_z
    def setup(self):
        x, y = np.logspace(3, 5, 10), np.logspace(3, 5, 10)
        x, y = np.meshgrid(x, y)
        self.large_z = x + 1j*y

    # 定义 time_loggamma_asymptotic 方法，测试 loggamma 函数的性能
    def time_loggamma_asymptotic(self):
        loggamma(self.large_z)


# 定义 Expn 类，继承自 Benchmark 类
class Expn(Benchmark):
    
    # 设置方法，初始化大数值网格 self.n 和 self.x
    def setup(self):
        n, x = np.arange(50, 500), np.logspace(0, 20, 100)
        n, x = np.meshgrid(n, x)
        self.n, self.x = n, x

    # 定义 time_expn_large_n 方法，测试 expn 函数在大数值 n 下的性能
    def time_expn_large_n(self):
        expn(self.n, self.x)
```