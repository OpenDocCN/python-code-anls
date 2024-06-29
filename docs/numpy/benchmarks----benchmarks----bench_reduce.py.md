# `.\numpy\benchmarks\benchmarks\bench_reduce.py`

```
from .common import Benchmark, TYPES1, get_squares  # 导入所需模块和函数

import numpy as np  # 导入 NumPy 库


class AddReduce(Benchmark):
    def setup(self):
        self.squares = get_squares().values()  # 获取方形矩阵数据集

    def time_axis_0(self):
        [np.add.reduce(a, axis=0) for a in self.squares]
        # 对每个方形矩阵沿着 axis=0 的方向进行 reduce 操作，即按列相加

    def time_axis_1(self):
        [np.add.reduce(a, axis=1) for a in self.squares]
        # 对每个方形矩阵沿着 axis=1 的方向进行 reduce 操作，即按行相加


class AddReduceSeparate(Benchmark):
    params = [[0, 1], TYPES1]  # 参数化设置：axis 可选 0 或 1，typename 取自 TYPES1
    param_names = ['axis', 'type']  # 参数名称

    def setup(self, axis, typename):
        self.a = get_squares()[typename]  # 获取特定类型的方形矩阵数据

    def time_reduce(self, axis, typename):
        np.add.reduce(self.a, axis=axis)
        # 对给定的方形矩阵进行 reduce 操作，根据参数 axis 指定是按列还是按行相加


class AnyAll(Benchmark):
    def setup(self):
        # 初始化全为 0 或全为 1 的数组，用于测试 any 和 all 方法的性能
        self.zeros = np.full(100000, 0, bool)
        self.ones = np.full(100000, 1, bool)

    def time_all_fast(self):
        self.zeros.all()
        # 使用 NumPy 的 all 方法检查数组中的所有元素是否为 True

    def time_all_slow(self):
        self.ones.all()
        # 使用 NumPy 的 all 方法检查数组中的所有元素是否为 True

    def time_any_fast(self):
        self.ones.any()
        # 使用 NumPy 的 any 方法检查数组中是否有任意元素为 True

    def time_any_slow(self):
        self.zeros.any()
        # 使用 NumPy 的 any 方法检查数组中是否有任意元素为 True


class StatsReductions(Benchmark):
    params = ['int64', 'uint64', 'float32', 'float64', 'complex64', 'bool_']  # 参数化设置：不同数据类型
    param_names = ['dtype']  # 参数名称

    def setup(self, dtype):
        self.data = np.ones(200, dtype=dtype)  # 创建指定数据类型的长度为 200 的数组
        if dtype.startswith('complex'):
            self.data = self.data * self.data.T*1j  # 若数据类型为复数，则进行相应初始化操作

    def time_min(self, dtype):
        np.min(self.data)
        # 计算数组中元素的最小值

    def time_max(self, dtype):
        np.max(self.data)
        # 计算数组中元素的最大值

    def time_mean(self, dtype):
        np.mean(self.data)
        # 计算数组中元素的平均值

    def time_std(self, dtype):
        np.std(self.data)
        # 计算数组中元素的标准差

    def time_prod(self, dtype):
        np.prod(self.data)
        # 计算数组中元素的乘积

    def time_var(self, dtype):
        np.var(self.data)
        # 计算数组中元素的方差


class FMinMax(Benchmark):
    params = [np.float32, np.float64]  # 参数化设置：浮点数类型
    param_names = ['dtype']  # 参数名称

    def setup(self, dtype):
        self.d = np.ones(20000, dtype=dtype)  # 创建指定数据类型的长度为 20000 的数组

    def time_min(self, dtype):
        np.fmin.reduce(self.d)
        # 使用 NumPy 的 fmin 方法对数组进行 reduce 操作，计算最小值

    def time_max(self, dtype):
        np.fmax.reduce(self.d)
        # 使用 NumPy 的 fmax 方法对数组进行 reduce 操作，计算最大值


class ArgMax(Benchmark):
    params = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32,
              np.int64, np.uint64, np.float32, np.float64, bool]  # 参数化设置：不同数据类型
    param_names = ['dtype']  # 参数名称

    def setup(self, dtype):
        self.d = np.zeros(200000, dtype=dtype)  # 创建指定数据类型的长度为 200000 的数组

    def time_argmax(self, dtype):
        np.argmax(self.d)
        # 找出数组中最大元素的索引


class ArgMin(Benchmark):
    params = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32,
              np.int64, np.uint64, np.float32, np.float64, bool]  # 参数化设置：不同数据类型
    param_names = ['dtype']  # 参数名称

    def setup(self, dtype):
        self.d = np.ones(200000, dtype=dtype)  # 创建指定数据类型的长度为 200000 的数组

    def time_argmin(self, dtype):
        np.argmin(self.d)
        # 找出数组中最小元素的索引


class SmallReduction(Benchmark):
    def setup(self):
        self.d = np.ones(100, dtype=np.float32)  # 创建长度为 100 的单精度浮点数数组

    def time_small(self):
        np.sum(self.d)
        # 计算数组中所有元素的和
```