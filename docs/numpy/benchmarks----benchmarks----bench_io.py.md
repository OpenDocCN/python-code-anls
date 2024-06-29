# `.\numpy\benchmarks\benchmarks\bench_io.py`

```
from .common import Benchmark, get_squares, get_squares_
import numpy as np
from io import SEEK_SET, StringIO, BytesIO

class Copy(Benchmark):
    params = ["int8", "int16", "float32", "float64",
              "complex64", "complex128"]
    param_names = ['type']

    def setup(self, typename):
        dtype = np.dtype(typename)
        # 创建一个大小为 500x50 的 NumPy 数组，数据类型由参数指定
        self.d = np.arange((50 * 500), dtype=dtype).reshape((500, 50))
        # 创建一个大小为 50x500 的 NumPy 数组，数据类型由参数指定
        self.e = np.arange((50 * 500), dtype=dtype).reshape((50, 500))
        # 将数组 e 重新形状为数组 d 的形状
        self.e_d = self.e.reshape(self.d.shape)
        # 创建一个大小为 25000 的一维 NumPy 数组，数据类型由参数指定
        self.dflat = np.arange((50 * 500), dtype=dtype)

    def time_memcpy(self, typename):
        # 使用 NumPy 的广播功能，将数组 e_d 的内容复制到数组 d 中
        self.d[...] = self.e_d

    def time_memcpy_large_out_of_place(self, typename):
        # 创建一个大小为 1024x1024 的全为 1 的 NumPy 数组，数据类型由参数指定
        l = np.ones(1024**2, dtype=np.dtype(typename))
        # 使用 NumPy 的 copy 方法复制数组 l
        l.copy()

    def time_cont_assign(self, typename):
        # 将数组 d 中所有元素赋值为 1
        self.d[...] = 1

    def time_strided_copy(self, typename):
        # 使用 NumPy 的转置操作，将数组 e 的转置内容复制到数组 d 中
        self.d[...] = self.e.T

    def time_strided_assign(self, typename):
        # 将数组 dflat 中偶数索引位置的元素赋值为 2
        self.dflat[::2] = 2


class CopyTo(Benchmark):
    def setup(self):
        # 创建一个大小为 50000 的全为 1 的一维 NumPy 数组
        self.d = np.ones(50000)
        # 复制数组 d 到数组 e
        self.e = self.d.copy()
        # 创建一个布尔掩码数组，标记数组 d 中值为 1 的位置
        self.m = (self.d == 1)
        # 取反布尔掩码数组
        self.im = (~ self.m)
        # 复制数组 m 到数组 m8
        self.m8 = self.m.copy()
        # 在数组 m8 中每隔 8 个元素取反
        self.m8[::8] = (~ self.m[::8])
        # 取反布尔掩码数组 m8
        self.im8 = (~ self.m8)

    def time_copyto(self):
        # 使用 np.copyto 将数组 e 的内容复制到数组 d
        np.copyto(self.d, self.e)

    def time_copyto_sparse(self):
        # 使用 np.copyto 将数组 e 的内容复制到数组 d，仅在数组 m 为 True 的位置进行复制
        np.copyto(self.d, self.e, where=self.m)

    def time_copyto_dense(self):
        # 使用 np.copyto 将数组 e 的内容复制到数组 d，仅在数组 im 为 True 的位置进行复制
        np.copyto(self.d, self.e, where=self.im)

    def time_copyto_8_sparse(self):
        # 使用 np.copyto 将数组 e 的内容复制到数组 d，仅在数组 m8 为 True 的位置进行复制
        np.copyto(self.d, self.e, where=self.m8)

    def time_copyto_8_dense(self):
        # 使用 np.copyto 将数组 e 的内容复制到数组 d，仅在数组 im8 为 True 的位置进行复制
        np.copyto(self.d, self.e, where=self.im8)


class Savez(Benchmark):
    def setup(self):
        # 获取由 get_squares 函数返回的字典
        self.squares = get_squares()

    def time_vb_savez_squares(self):
        # 将字典内容保存到文件 tmp.npz
        np.savez('tmp.npz', **self.squares)


class LoadNpyOverhead(Benchmark):
    def setup(self):
        # 创建一个 BytesIO 对象
        self.buffer = BytesIO()
        # 将 get_squares_()['float32'] 保存到 buffer 中
        np.save(self.buffer, get_squares_()['float32'])

    def time_loadnpy_overhead(self):
        # 将 buffer 的指针位置移动到开头
        self.buffer.seek(0, SEEK_SET)
        # 从 buffer 中加载 NumPy 数组
        np.load(self.buffer)


class LoadtxtCSVComments(Benchmark):
    # 用于测试 np.loadtxt 在读取 CSV 文件时处理注释的性能
    params = [10, int(1e2), int(1e4), int(1e5)]
    param_names = ['num_lines']

    def setup(self, num_lines):
        # 创建包含注释的 CSV 数据
        data = ['1,2,3 # comment'] * num_lines
        # 创建一个 StringIO 对象，用于保存 CSV 数据
        self.data_comments = StringIO('\n'.join(data))
    def time_comment_loadtxt_csv(self, num_lines):
        # 定义一个方法用于测试处理带有注释的行数
        # 从 CSV 文件中加载数据时的性能

        # 受到 pandas 中 read_csv 的类似基准测试的启发

        # 需要在每次正确的时间测试调用前重置 StringIO 对象的位置
        # 这会在一定程度上影响计时结果
        np.loadtxt(self.data_comments,
                   delimiter=',')
        # 将 StringIO 对象的位置重置到文件开头，以备下次读取使用
        self.data_comments.seek(0)
class LoadtxtCSVdtypes(Benchmark):
    # 对 np.loadtxt 进行性能基准测试，测试不同的数据类型从 CSV 文件解析/转换

    params = (['float32', 'float64', 'int32', 'int64',
               'complex128', 'str', 'object'],
              [10, int(1e2), int(1e4), int(1e5)])
    param_names = ['dtype', 'num_lines']

    def setup(self, dtype, num_lines):
        # 设置测试数据：生成包含 num_lines 行 '5, 7, 888' 的 CSV 数据
        data = ['5, 7, 888'] * num_lines
        self.csv_data = StringIO('\n'.join(data))

    def time_loadtxt_dtypes_csv(self, dtype, num_lines):
        # 基准测试加载各种数据类型的数组从 CSV 文件

        # 因为基准测试依赖状态和时间测量，需要重置 StringIO 对象的指针
        np.loadtxt(self.csv_data,
                   delimiter=',',
                   dtype=dtype)
        self.csv_data.seek(0)

class LoadtxtCSVStructured(Benchmark):
    # 对 np.loadtxt 进行性能基准测试，测试结构化数据类型与 CSV 文件的解析

    def setup(self):
        # 设置测试数据：生成包含 50000 行的 "M, 21, 72, X, 155" 结构化 CSV 数据
        num_lines = 50000
        data = ["M, 21, 72, X, 155"] * num_lines
        self.csv_data = StringIO('\n'.join(data))

    def time_loadtxt_csv_struct_dtype(self):
        # 在每次迭代重复之间强制重置 StringIO 对象的指针

        np.loadtxt(self.csv_data,
                   delimiter=',',
                   dtype=[('category_1', 'S1'),
                          ('category_2', 'i4'),
                          ('category_3', 'f8'),
                          ('category_4', 'S1'),
                          ('category_5', 'f8')])
        self.csv_data.seek(0)


class LoadtxtCSVSkipRows(Benchmark):
    # 对 loadtxt 进行性能基准测试，在读取 CSV 文件数据时跳过行；pandas asv 套件中也有类似的基准测试

    params = [0, 500, 10000]
    param_names = ['skiprows']

    def setup(self, skiprows):
        # 设置测试数据：生成包含 100000 行和 3 列的随机数据，并将其保存为 CSV 文件
        np.random.seed(123)
        test_array = np.random.rand(100000, 3)
        self.fname = 'test_array.csv'
        np.savetxt(fname=self.fname,
                   X=test_array,
                   delimiter=',')

    def time_skiprows_csv(self, skiprows):
        # 基准测试跳过指定行数后加载 CSV 文件数据

        np.loadtxt(self.fname,
                   delimiter=',',
                   skiprows=skiprows)

class LoadtxtReadUint64Integers(Benchmark):
    # pandas 有一个类似的 CSV 读取基准测试，这里修改以适应 np.loadtxt

    params = [550, 1000, 10000]
    param_names = ['size']

    def setup(self, size):
        # 设置测试数据：生成 uint64 类型的数组，并将其保存为 StringIO 对象
        arr = np.arange(size).astype('uint64') + 2**63
        self.data1 = StringIO('\n'.join(arr.astype(str).tolist()))
        arr = arr.astype(object)
        arr[500] = -1
        self.data2 = StringIO('\n'.join(arr.astype(str).tolist()))

    def time_read_uint64(self, size):
        # 在每次迭代重复之间强制重置 StringIO 对象的指针

        np.loadtxt(self.data1)
        self.data1.seek(0)
    # 定义一个方法用于处理读取 uint64 类型的负值时间
    def time_read_uint64_neg_values(self, size):
        # 强制重置 StringIO 对象的指针位置到文件开头
        np.loadtxt(self.data2)
        # 将 StringIO 对象的指针位置移动到文件开头
        self.data2.seek(0)
class LoadtxtUseColsCSV(Benchmark):
    # benchmark selective column reading from CSV files
    # using np.loadtxt

    params = [2, [1, 3], [1, 3, 5, 7]]
    param_names = ['usecols']

    def setup(self, usecols):
        # 准备数据：生成包含大量行的 CSV 数据
        num_lines = 5000
        data = ['0, 1, 2, 3, 4, 5, 6, 7, 8, 9'] * num_lines
        self.csv_data = StringIO('\n'.join(data))

    def time_loadtxt_usecols_csv(self, usecols):
        # 由于文件读取的状态依赖性，必须重新定位 StringIO 对象
        np.loadtxt(self.csv_data,
                   delimiter=',',
                   usecols=usecols)
        self.csv_data.seek(0)

class LoadtxtCSVDateTime(Benchmark):
    # benchmarks for np.loadtxt operating with
    # datetime data in a CSV file

    params = [20, 200, 2000, 20000]
    param_names = ['num_lines']

    def setup(self, num_lines):
        # 创建一个包含日期字符串和随机浮点数数据的模拟两列 CSV 文件
        dates = np.arange('today', 20, dtype=np.datetime64)
        np.random.seed(123)
        values = np.random.rand(20)
        date_line = ''

        for date, value in zip(dates, values):
            date_line += (str(date) + ',' + str(value) + '\n')

        # 扩展数据至指定行数
        data = date_line * (num_lines // 20)
        self.csv_data = StringIO(data)

    def time_loadtxt_csv_datetime(self, num_lines):
        # 重置 StringIO 对象的位置，因为时间迭代的计时依赖于对象状态
        X = np.loadtxt(self.csv_data,
                       delimiter=',',
                       dtype=([('dates', 'M8[us]'),
                               ('values', 'float64')]))
        self.csv_data.seek(0)
```