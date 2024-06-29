# `.\numpy\benchmarks\benchmarks\bench_trim_zeros.py`

```
# 从.common模块中导入Benchmark类，用于性能基准测试
from .common import Benchmark

# 导入NumPy库，并定义几种特定的数据类型
import numpy as np

# 定义全局变量，表示不同数据类型的NumPy数据类型对象
_FLOAT = np.dtype('float64')
_COMPLEX = np.dtype('complex128')
_INT = np.dtype('int64')
_BOOL = np.dtype('bool')

# 定义一个继承自Benchmark类的TrimZeros类
class TrimZeros(Benchmark):
    # 参数名列表，用于性能测试参数化
    param_names = ["dtype", "size"]
    # 参数列表，包含数据类型和数组大小的组合
    params = [
        [_INT, _FLOAT, _COMPLEX, _BOOL],  # 数据类型
        [3000, 30_000, 300_000]           # 数组大小
    ]

    # 设置方法，在每个性能测试之前调用，初始化数组
    def setup(self, dtype, size):
        # 计算数组的长度为总大小的三分之一
        n = size // 3
        # 创建一个由三部分组成的NumPy数组：前后各有一部分零元素，中间部分为随机均匀分布的元素
        self.array = np.hstack([
            np.zeros(n),                      # 前部分零元素
            np.random.uniform(size=n),        # 中间部分随机均匀分布的元素
            np.zeros(n),                      # 后部分零元素
        ]).astype(dtype)                      # 转换成指定的数据类型

    # 性能测试方法，测试np.trim_zeros函数的性能
    def time_trim_zeros(self, dtype, size):
        np.trim_zeros(self.array)  # 调用np.trim_zeros函数处理初始化的数组
```