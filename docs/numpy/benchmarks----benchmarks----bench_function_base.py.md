# `.\numpy\benchmarks\benchmarks\bench_function_base.py`

```
# 从 common 模块中导入 Benchmark 类
from .common import Benchmark

# 导入 numpy 库并使用别名 np
import numpy as np

try:
    # 尝试导入 SkipNotImplemented 类，该类自 asv_runner.benchmarks.mark 模块中引入
    from asv_runner.benchmarks.mark import SkipNotImplemented
except ImportError:
    # 如果 ImportError 异常发生，则将 SkipNotImplemented 设置为 NotImplementedError
    SkipNotImplemented = NotImplementedError


# 创建 Linspace 类，继承自 Benchmark 类
class Linspace(Benchmark):
    # 设置方法，在每次测试前初始化数据
    def setup(self):
        self.d = np.array([1, 2, 3])

    # 定义时间测试方法 time_linspace_scalar
    def time_linspace_scalar(self):
        # 调用 numpy 中的 linspace 函数，生成一个包含两个元素的等差数列
        np.linspace(0, 10, 2)

    # 定义时间测试方法 time_linspace_array
    def time_linspace_array(self):
        # 调用 numpy 中的 linspace 函数，使用实例变量 self.d 作为起始值，生成一个等差数列
        np.linspace(self.d, 10, 10)


# 创建 Histogram1D 类，继承自 Benchmark 类
class Histogram1D(Benchmark):
    # 设置方法，在每次测试前初始化数据
    def setup(self):
        # 使用 numpy 中的 linspace 函数生成一个包含 100000 个元素的等差数列
        self.d = np.linspace(0, 100, 100000)

    # 定义时间测试方法 time_full_coverage
    def time_full_coverage(self):
        # 调用 numpy 中的 histogram 函数，对 self.d 进行直方图统计，使用 200 个 bin
        np.histogram(self.d, 200, (0, 100))

    # 定义时间测试方法 time_small_coverage
    def time_small_coverage(self):
        # 调用 numpy 中的 histogram 函数，对 self.d 进行直方图统计，使用 200 个 bin，但仅计算区间 (50, 51)
        np.histogram(self.d, 200, (50, 51))

    # 定义时间测试方法 time_fine_binning
    def time_fine_binning(self):
        # 调用 numpy 中的 histogram 函数，对 self.d 进行直方图统计，使用 10000 个 bin
        np.histogram(self.d, 10000, (0, 100))


# 创建 Histogram2D 类，继承自 Benchmark 类
class Histogram2D(Benchmark):
    # 设置方法，在每次测试前初始化数据
    def setup(self):
        # 使用 numpy 中的 linspace 函数生成一个包含 200000 个元素的等差数列，并将其重塑为二维数组
        self.d = np.linspace(0, 100, 200000).reshape((-1,2))

    # 定义时间测试方法 time_full_coverage
    def time_full_coverage(self):
        # 调用 numpy 中的 histogramdd 函数，对 self.d 进行多维直方图统计，使用 200x200 个 bin
        np.histogramdd(self.d, (200, 200), ((0, 100), (0, 100)))

    # 定义时间测试方法 time_small_coverage
    def time_small_coverage(self):
        # 调用 numpy 中的 histogramdd 函数，对 self.d 进行多维直方图统计，使用 200x200 个 bin，但仅计算区间 ((50, 51), (50, 51))
        np.histogramdd(self.d, (200, 200), ((50, 51), (50, 51)))

    # 定义时间测试方法 time_fine_binning
    def time_fine_binning(self):
        # 调用 numpy 中的 histogramdd 函数，对 self.d 进行多维直方图统计，使用 10000x10000 个 bin
        np.histogramdd(self.d, (10000, 10000), ((0, 100), (0, 100)))


# 创建 Bincount 类，继承自 Benchmark 类
class Bincount(Benchmark):
    # 设置方法，在每次测试前初始化数据
    def setup(self):
        # 使用 numpy 中的 arange 函数生成一个包含 80000 个元素的数组，数据类型为 np.intp
        self.d = np.arange(80000, dtype=np.intp)
        # 将 self.d 转换为 np.float64 类型，并赋值给实例变量 self.e
        self.e = self.d.astype(np.float64)

    # 定义时间测试方法 time_bincount
    def time_bincount(self):
        # 调用 numpy 中的 bincount 函数，统计 self.d 中每个值出现的次数
        np.bincount(self.d)

    # 定义时间测试方法 time_weights
    def time_weights(self):
        # 调用 numpy 中的 bincount 函数，同时根据权重 self.e 统计 self.d 中每个值出现的加权次数
        np.bincount(self.d, weights=self.e)


# 创建 Mean 类，继承自 Benchmark 类
class Mean(Benchmark):
    # 定义参数名列表 param_names 和参数值列表 params
    param_names = ['size']
    params = [[1, 10, 100_000]]

    # 设置方法，在每次测试前根据 size 初始化数据
    def setup(self, size):
        # 使用 numpy 中的 arange 函数生成一个包含 2*size 个元素的数组，并将其重塑为二维数组
        self.array = np.arange(2*size).reshape(2, size)

    # 定义时间测试方法 time_mean，计算 self.array 的平均值
    def time_mean(self, size):
        np.mean(self.array)

    # 定义时间测试方法 time_mean_axis，计算 self.array 沿指定轴的平均值
    def time_mean_axis(self, size):
        np.mean(self.array, axis=1)


# 创建 Median 类，继承自 Benchmark 类
class Median(Benchmark):
    # 设置方法，在每次测试前初始化数据
    def setup(self):
        # 使用 numpy 中的 arange 函数生成一个包含 10000 个元素的 np.float32 类型的数组
        self.e = np.arange(10000, dtype=np.float32)
        # 使用 numpy 中的 arange 函数生成一个包含 10001 个元素的 np.float32 类型的数组
        self.o = np.arange(10001, dtype=np.float32)
        # 使用 numpy 中的 random 函数生成一个形状为 (10000, 20) 的随机数组
        self.tall = np.random.random((10000, 20))
        # 使用 numpy 中的 random 函数生成一个形状为 (20, 10000) 的随机数组
        self.wide = np.random.random((20, 10000))

    # 定义时间测试方法 time_even，计算 self.e 的中位数
    def time_even(self):
        np.median(self.e)

    # 定义时间测试方法 time_odd，计算 self.o 的中位数
    def time_odd(self):
        np.median(self.o)

    # 定义时间测试方法 time_even_inplace，计算 self.e 的中位数，且允许在计算过程中覆盖输入数据
    def time_even_inplace(self):
        np.median(self.e, overwrite_input=True)

    # 定义时间测试方法 time_odd_inplace，计算 self.o 的中位数，且允许在计算过程中覆盖输入数据
    def time_odd_inplace(self):
        np.median(self.o, overwrite_input=True)

    # 定义时间测试方法 time_even_small，计算 self.e 的前 500 个元素的中位数，且允许在计算过程中覆盖输入数据
    def time_even_small(self):
        np.median(self.e[:500], overwrite_input=True)

    # 定义时间测试方法 time_odd_small，计算 self.o 的前 500 个元素的中位数，且允许在计算过程中覆盖输入数据
    def time_odd_small(self):
        np.median(self.o[:500], overwrite_input=True)

    # 定义时间测试方法 time_tall，计算 self.tall 沿最后一个轴的中位数
    def time_tall(self):
        np.median(self.tall, axis=-1)

    # 定义时间测试方法 time_wide，计算 self.wide 沿第一个轴的中位数
    def time_wide(self):
        np.median(self.wide, axis=0)


# 创建 Percent
class Select(Benchmark):
    # Benchmark 类的子类，用于选择性能测试
    def setup(self):
        # 创建一个长度为 20000 的 NumPy 数组 d，包含整数序列
        self.d = np.arange(20000)
        # 将数组 d 复制给数组 e
        self.e = self.d.copy()
        # 定义两个条件列表，用于条件选择
        self.cond = [(self.d > 4), (self.d < 2)]
        # 大规模条件列表，包含多个重复的条件
        self.cond_large = [(self.d > 4), (self.d < 2)] * 10

    def time_select(self):
        # 使用 np.select 函数根据条件 self.cond 选择数组 self.d 或 self.e 的值
        np.select(self.cond, [self.d, self.e])

    def time_select_larger(self):
        # 使用 np.select 函数根据大规模条件 self.cond_large 选择数组 self.d 或 self.e 的值
        np.select(self.cond_large, ([self.d, self.e] * 10))


def memoize(f):
    # 缓存装饰器，用于存储函数计算结果，避免重复计算
    _memoized = {}
    def wrapped(*args):
        # 如果参数 args 不在缓存中，则计算并存储结果
        if args not in _memoized:
            _memoized[args] = f(*args)

        return _memoized[args].copy()  # 返回结果的副本

    return f


class SortGenerator:
    # 随机未排序区域的大小，用于基准测试
    AREA_SIZE = 100
    # 部分有序子数组的大小，用于基准测试
    BUBBLE_SIZE = 100

    @staticmethod
    @memoize
    def random(size, dtype, rnd):
        """
        Returns a randomly-shuffled array.
        """
        # 创建一个随机打乱顺序的数组
        arr = np.arange(size, dtype=dtype)
        rnd = np.random.RandomState(1792364059)
        np.random.shuffle(arr)
        rnd.shuffle(arr)
        return arr

    @staticmethod
    @memoize
    def ordered(size, dtype, rnd):
        """
        Returns an ordered array.
        """
        # 创建一个有序的数组
        return np.arange(size, dtype=dtype)

    @staticmethod
    @memoize
    def reversed(size, dtype, rnd):
        """
        Returns an array that's in descending order.
        """
        # 创建一个降序排列的数组
        dtype = np.dtype(dtype)
        try:
            with np.errstate(over="raise"):
                res = dtype.type(size-1)
        except (OverflowError, FloatingPointError):
            raise SkipNotImplemented("Cannot construct arange for this size.")

        return np.arange(size-1, -1, -1, dtype=dtype)

    @staticmethod
    @memoize
    def uniform(size, dtype, rnd):
        """
        Returns an array that has the same value everywhere.
        """
        # 创建一个所有元素均相同的数组
        return np.ones(size, dtype=dtype)

    @staticmethod
    @memoize
    def sorted_block(size, dtype, block_size, rnd):
        """
        Returns an array with blocks that are all sorted.
        """
        # 创建一个包含排序块的数组
        a = np.arange(size, dtype=dtype)
        b = []
        if size < block_size:
            return a
        block_num = size // block_size
        for i in range(block_num):
            b.extend(a[i::block_num])
        return np.array(b)


class Sort(Benchmark):
    """
    This benchmark tests sorting performance with several
    different types of arrays that are likely to appear in
    real-world applications.
    """
    # 排序基准测试类，用于测试不同类型的数组排序性能
    params = [
        # 在 NumPy 1.17 及更新版本中，'merge' 可以是多种稳定排序算法之一，不一定是归并排序。
        ['quick', 'merge', 'heap'],
        ['float64', 'int64', 'float32', 'uint32', 'int32', 'int16', 'float16'],
        [
            ('random',),
            ('ordered',),
            ('reversed',),
            ('uniform',),
            ('sorted_block', 10),
            ('sorted_block', 100),
            ('sorted_block', 1000),
        ],
    ]
    # 定义包含参数名称的列表
    param_names = ['kind', 'dtype', 'array_type']

    # 定义被基准测试数组的大小
    ARRAY_SIZE = 10000

    # 设置函数，用于生成和准备基准测试所需的数组
    def setup(self, kind, dtype, array_type):
        # 使用指定种子创建随机数生成器
        rnd = np.random.RandomState(507582308)
        # 从 array_type 中获取数组类型的类名
        array_class = array_type[0]
        # 调用 SortGenerator 类的相应方法生成数组，存储在 self.arr 中
        self.arr = getattr(SortGenerator, array_class)(self.ARRAY_SIZE, dtype, *array_type[1:], rnd)

    # 基准测试排序算法执行时间
    def time_sort(self, kind, dtype, array_type):
        # 使用 np.sort(...) 而不是 arr.sort(...)，因为它会生成副本
        # 这很重要，因为数据只需准备一次，但会跨多次运行使用
        np.sort(self.arr, kind=kind)

    # 基准测试 argsort 函数执行时间
    def time_argsort(self, kind, dtype, array_type):
        # 使用 np.argsort 对数组进行排序并返回索引
        np.argsort(self.arr, kind=kind)
class Partition(Benchmark):
    # 参数列表，包括数据类型、数组类型和 k 值的不同组合
    params = [
        ['float64', 'int64', 'float32', 'int32', 'int16', 'float16'],
        [
            ('random',),
            ('ordered',),
            ('reversed',),
            ('uniform',),
            ('sorted_block', 10),
            ('sorted_block', 100),
            ('sorted_block', 1000),
        ],
        [10, 100, 1000],
    ]
    # 参数名列表，对应 params 中的各个参数
    param_names = ['dtype', 'array_type', 'k']

    # 被基准测试数组的大小
    ARRAY_SIZE = 100000

    def setup(self, dtype, array_type, k):
        # 设置随机种子，并根据指定的数组类型生成相应大小和类型的数组
        rnd = np.random.seed(2136297818)
        array_class = array_type[0]
        self.arr = getattr(SortGenerator, array_class)(
            self.ARRAY_SIZE, dtype, *array_type[1:], rnd)

    def time_partition(self, dtype, array_type, k):
        # 对 self.arr 数组进行分区操作，并记录时间
        temp = np.partition(self.arr, k)

    def time_argpartition(self, dtype, array_type, k):
        # 对 self.arr 数组进行参数分区操作，并记录时间
        temp = np.argpartition(self.arr, k)


class SortWorst(Benchmark):
    def setup(self):
        # 创建一个最坏情况下的快速排序数组
        # 使用快速排序的中位数为 3 的最坏情况
        self.worst = np.arange(1000000)
        x = self.worst
        while x.size > 3:
            mid = x.size // 2
            x[mid], x[-2] = x[-2], x[mid]
            x = x[:-2]

    def time_sort_worst(self):
        # 对 self.worst 数组进行排序，并记录时间
        np.sort(self.worst)

    # 为了向后兼容性，保留旧基准测试名称
    time_sort_worst.benchmark_name = "bench_function_base.Sort.time_sort_worst"


class Where(Benchmark):
    def setup(self):
        # 创建几种用于测试的数组和条件
        self.d = np.arange(20000)
        self.d_o = self.d.astype(object)
        self.e = self.d.copy()
        self.e_o = self.d_o.copy()
        self.cond = (self.d > 5000)
        size = 1024 * 1024 // 8
        rnd_array = np.random.rand(size)
        self.rand_cond_01 = rnd_array > 0.01
        self.rand_cond_20 = rnd_array > 0.20
        self.rand_cond_30 = rnd_array > 0.30
        self.rand_cond_40 = rnd_array > 0.40
        self.rand_cond_50 = rnd_array > 0.50
        self.all_zeros = np.zeros(size, dtype=bool)
        self.all_ones = np.ones(size, dtype=bool)
        self.rep_zeros_2 = np.arange(size) % 2 == 0
        self.rep_zeros_4 = np.arange(size) % 4 == 0
        self.rep_zeros_8 = np.arange(size) % 8 == 0
        self.rep_ones_2 = np.arange(size) % 2 > 0
        self.rep_ones_4 = np.arange(size) % 4 > 0
        self.rep_ones_8 = np.arange(size) % 8 > 0

    def time_1(self):
        # 测试 np.where 函数在条件 self.cond 下的性能
        np.where(self.cond)

    def time_2(self):
        # 测试 np.where 函数在条件 self.cond 下的性能，并根据条件选择 self.d 或 self.e
        np.where(self.cond, self.d, self.e)

    def time_2_object(self):
        # 测试 np.where 函数在条件 self.cond 下的性能，特别考虑对象和字节交换数组的情况
        np.where(self.cond, self.d_o, self.e_o)

    def time_2_broadcast(self):
        # 测试 np.where 函数在条件 self.cond 下的性能，使用广播方式选择 self.d 或 0
        np.where(self.cond, self.d, 0)

    def time_all_zeros(self):
        # 测试 np.where 函数在全零数组 self.all_zeros 下的性能
        np.where(self.all_zeros)

    def time_random_01_percent(self):
        # 测试 np.where 函数在随机条件 self.rand_cond_01 下的性能
        np.where(self.rand_cond_01)

    def time_random_20_percent(self):
        # 测试 np.where 函数在随机条件 self.rand_cond_20 下的性能
        np.where(self.rand_cond_20)

    def time_random_30_percent(self):
        # 测试 np.where 函数在随机条件 self.rand_cond_30 下的性能
        np.where(self.rand_cond_30)
    # 使用 NumPy 的 where 函数执行条件筛选，针对预定义的随机条件 self.rand_cond_40
    def time_random_40_percent(self):
        np.where(self.rand_cond_40)
    
    # 使用 NumPy 的 where 函数执行条件筛选，针对预定义的随机条件 self.rand_cond_50
    def time_random_50_percent(self):
        np.where(self.rand_cond_50)
    
    # 使用 NumPy 的 where 函数执行条件筛选，针对预定义的全为1的条件 self.all_ones
    def time_all_ones(self):
        np.where(self.all_ones)
    
    # 使用 NumPy 的 where 函数执行条件筛选，针对预定义的交错0的条件 self.rep_zeros_2
    def time_interleaved_zeros_x2(self):
        np.where(self.rep_zeros_2)
    
    # 使用 NumPy 的 where 函数执行条件筛选，针对预定义的交错0的条件 self.rep_zeros_4
    def time_interleaved_zeros_x4(self):
        np.where(self.rep_zeros_4)
    
    # 使用 NumPy 的 where 函数执行条件筛选，针对预定义的交错0的条件 self.rep_zeros_8
    def time_interleaved_zeros_x8(self):
        np.where(self.rep_zeros_8)
    
    # 使用 NumPy 的 where 函数执行条件筛选，针对预定义的交错1的条件 self.rep_ones_2
    def time_interleaved_ones_x2(self):
        np.where(self.rep_ones_2)
    
    # 使用 NumPy 的 where 函数执行条件筛选，针对预定义的交错1的条件 self.rep_ones_4
    def time_interleaved_ones_x4(self):
        np.where(self.rep_ones_4)
    
    # 使用 NumPy 的 where 函数执行条件筛选，针对预定义的交错1的条件 self.rep_ones_8
    def time_interleaved_ones_x8(self):
        np.where(self.rep_ones_8)
```