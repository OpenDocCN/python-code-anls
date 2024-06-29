# `.\numpy\benchmarks\benchmarks\bench_lib.py`

```py
"""Benchmarks for `numpy.lib`."""

# 从common模块导入Benchmark类
from .common import Benchmark
# 导入numpy库并简写为np
import numpy as np

# 定义Pad类，继承Benchmark类
class Pad(Benchmark):
    """Benchmarks for `numpy.pad`.

    When benchmarking the pad function it is useful to cover scenarios where
    the ratio between the size of the input array and the output array differs
    significantly (original area vs. padded area). This allows to evaluate for
    which scenario a padding algorithm is optimized. Furthermore involving
    large range of array sizes ensures that the effects of CPU-bound caching is
    visible.

    The table below shows the sizes of the arrays involved in this benchmark:

    +-----------------+----------+-----------+-----------+-----------------+
    | shape           | original | padded: 1 | padded: 8 | padded: (0, 32) |
    +=================+==========+===========+===========+=================+
    | (2 ** 22,)      | 32 MiB   | 32.0 MiB  | 32.0 MiB  | 32.0 MiB        |
    +-----------------+----------+-----------+-----------+-----------------+
    | (1024, 1024)    | 8 MiB    | 8.03 MiB  | 8.25 MiB  | 8.51 MiB        |
    +-----------------+----------+-----------+-----------+-----------------+
    | (256, 256, 1)   | 256 KiB  | 786 KiB   | 5.08 MiB  | 11.6 MiB        |
    +-----------------+----------+-----------+-----------+-----------------+
    | (4, 4, 4, 4)    | 2 KiB    | 10.1 KiB  | 1.22 MiB  | 12.8 MiB        |
    +-----------------+----------+-----------+-----------+-----------------+
    | (1, 1, 1, 1, 1) | 8 B      | 1.90 MiB  | 10.8 MiB  | 299 MiB         |
    +-----------------+----------+-----------+-----------+-----------------+
    """

    # 参数名称列表
    param_names = ["shape", "pad_width", "mode"]
    # 参数组合
    params = [
        # Shape of the input arrays
        [(2 ** 22,), (1024, 1024), (256, 128, 1),
         (4, 4, 4, 4), (1, 1, 1, 1, 1)],
        # Tested pad widths
        [1, 8, (0, 32)],
        # Tested modes: mean, median, minimum & maximum use the same code path
        #               reflect & symmetric share a lot of their code path
        ["constant", "edge", "linear_ramp", "mean", "reflect", "wrap"],
    ]

    # 设置方法，在此方法中填充数组以确保在计时阶段之前触发操作系统的页面错误
    def setup(self, shape, pad_width, mode):
        self.array = np.full(shape, fill_value=1, dtype=np.float64)

    # 计时方法，调用numpy的pad函数进行计时
    def time_pad(self, shape, pad_width, mode):
        np.pad(self.array, pad_width, mode)


# 定义Nan类，继承Benchmark类
class Nan(Benchmark):
    """Benchmarks for nan functions"""

    # 参数名称列表
    param_names = ["array_size", "percent_nans"]
    # 参数组合
    params = [
            # sizes of the 1D arrays
            [200, int(2e5)],
            # percent of np.nan in arrays
            [0, 0.1, 2., 50., 90.],
            ]
    # 设置数组大小和 NaN 值的百分比
    def setup(self, array_size, percent_nans):
        # 使用指定种子创建随机状态生成器
        rnd = np.random.RandomState(1819780348)
        # 生成一个随机打乱顺序的数组，其大约包含指定百分比的 np.nan 值
        base_array = rnd.uniform(size=array_size)
        base_array[base_array < percent_nans / 100.] = np.nan
        # 将生成的数组赋值给对象的实例变量 arr
        self.arr = base_array

    # 计算数组中的最小值，忽略 NaN 值
    def time_nanmin(self, array_size, percent_nans):
        np.nanmin(self.arr)

    # 计算数组中的最大值，忽略 NaN 值
    def time_nanmax(self, array_size, percent_nans):
        np.nanmax(self.arr)

    # 返回数组中最小值的索引，忽略 NaN 值
    def time_nanargmin(self, array_size, percent_nans):
        np.nanargmin(self.arr)

    # 返回数组中最大值的索引，忽略 NaN 值
    def time_nanargmax(self, array_size, percent_nans):
        np.nanargmax(self.arr)

    # 计算数组中的元素总和，忽略 NaN 值
    def time_nansum(self, array_size, percent_nans):
        np.nansum(self.arr)

    # 计算数组中的元素乘积，忽略 NaN 值
    def time_nanprod(self, array_size, percent_nans):
        np.nanprod(self.arr)

    # 计算数组的累积和，忽略 NaN 值
    def time_nancumsum(self, array_size, percent_nans):
        np.nancumsum(self.arr)

    # 计算数组的累积乘积，忽略 NaN 值
    def time_nancumprod(self, array_size, percent_nans):
        np.nancumprod(self.arr)

    # 计算数组中的平均值，忽略 NaN 值
    def time_nanmean(self, array_size, percent_nans):
        np.nanmean(self.arr)

    # 计算数组中的方差，忽略 NaN 值
    def time_nanvar(self, array_size, percent_nans):
        np.nanvar(self.arr)

    # 计算数组中的标准差，忽略 NaN 值
    def time_nanstd(self, array_size, percent_nans):
        np.nanstd(self.arr)

    # 计算数组中的中位数，忽略 NaN 值
    def time_nanmedian(self, array_size, percent_nans):
        np.nanmedian(self.arr)

    # 计算数组中指定分位数对应的值，忽略 NaN 值
    def time_nanquantile(self, array_size, percent_nans):
        np.nanquantile(self.arr, q=0.2)

    # 计算数组中指定百分位数对应的值，忽略 NaN 值
    def time_nanpercentile(self, array_size, percent_nans):
        np.nanpercentile(self.arr, q=50)
# 定义一个继承自 Benchmark 的 Unique 类，用于评估包含 np.nan 值的 np.unique 函数的性能

param_names = ["array_size", "percent_nans"]
params = [
    # 1D 数组的大小
    [200, int(2e5)],
    # 数组中 np.nan 的百分比
    [0, 0.1, 2., 50., 90.],
]

def setup(self, array_size, percent_nans):
    # 设置随机种子为 123
    np.random.seed(123)
    # 创建一个随机打乱顺序的数组，并设置大约指定百分比的 np.nan 内容
    base_array = np.random.uniform(size=array_size)
    n_nan = int(percent_nans * array_size)
    nan_indices = np.random.choice(np.arange(array_size), size=n_nan)
    base_array[nan_indices] = np.nan
    self.arr = base_array

def time_unique_values(self, array_size, percent_nans):
    # 评估 np.unique 函数在数组中查找唯一值时的性能，不返回索引、逆向索引或计数
    np.unique(self.arr, return_index=False,
              return_inverse=False, return_counts=False)

def time_unique_counts(self, array_size, percent_nans):
    # 评估 np.unique 函数在数组中查找唯一值及其计数时的性能
    np.unique(self.arr, return_index=False,
              return_inverse=False, return_counts=True)

def time_unique_inverse(self, array_size, percent_nans):
    # 评估 np.unique 函数在数组中查找唯一值及其逆向索引时的性能
    np.unique(self.arr, return_index=False,
              return_inverse=True, return_counts=False)

def time_unique_all(self, array_size, percent_nans):
    # 评估 np.unique 函数在数组中查找唯一值、索引及其计数及逆向索引时的性能
    np.unique(self.arr, return_index=True,
              return_inverse=True, return_counts=True)


# 定义一个继承自 Benchmark 的 Isin 类，用于评估 numpy.isin 函数的性能

param_names = ["size", "highest_element"]
params = [
    [10, 100000, 3000000],
    [10, 10000, int(1e8)]
]

def setup(self, size, highest_element):
    # 创建一个大小为 size 的随机整数数组，元素范围在 0 到 highest_element 之间
    self.array = np.random.randint(
            low=0, high=highest_element, size=size)
    # 创建一个大小为 size 的随机整数数组，用于检查是否存在于 self.array 中，元素范围同样在 0 到 highest_element 之间
    self.in_array = np.random.randint(
            low=0, high=highest_element, size=size)

def time_isin(self, size, highest_element):
    # 评估 numpy.isin 函数在数组中检查元素是否存在时的性能
    np.isin(self.array, self.in_array)
```