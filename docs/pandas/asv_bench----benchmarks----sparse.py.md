# `D:\src\scipysrc\pandas\asv_bench\benchmarks\sparse.py`

```
import numpy as np  # 导入NumPy库，用于科学计算
import scipy.sparse  # 导入SciPy的稀疏矩阵模块

import pandas as pd  # 导入Pandas库，用于数据分析
from pandas import (  # 从Pandas库中导入以下子模块和类
    MultiIndex,  # 多级索引
    Series,  # 系列数据结构
    date_range,  # 日期范围生成函数
)
from pandas.arrays import SparseArray  # 导入Pandas稀疏数组模块


def make_array(size, dense_proportion, fill_value, dtype):
    dense_size = int(size * dense_proportion)
    arr = np.full(size, fill_value, dtype)  # 创建一个大小为size的数组，填充为fill_value，数据类型为dtype
    indexer = np.random.choice(np.arange(size), dense_size, replace=False)  # 从0到size-1随机选择dense_size个索引
    arr[indexer] = np.random.choice(np.arange(100, dtype=dtype), dense_size)  # 在选定的索引位置上填充随机数（数据类型为dtype）
    return arr


class SparseSeriesToFrame:
    def setup(self):
        K = 50
        N = 50001
        rng = date_range("1/1/2000", periods=N, freq="min")  # 生成一个包含50001个时间点的日期范围
        self.series = {}
        for i in range(1, K):
            data = np.random.randn(N)[:-i]  # 生成一个包含随机数的长度为N-i的数组
            idx = rng[:-i]  # 使用前N-i个日期作为索引
            data[100:] = np.nan  # 将数组中索引大于等于100的元素设置为NaN
            self.series[i] = Series(SparseArray(data), index=idx)  # 创建稀疏系列对象，使用稀疏数组data和日期索引idx

    def time_series_to_frame(self):
        pd.DataFrame(self.series)  # 将稀疏系列转换为DataFrame


class SparseArrayConstructor:
    params = ([0.1, 0.01], [0, np.nan], [np.int64, np.float64, object])
    param_names = ["dense_proportion", "fill_value", "dtype"]

    def setup(self, dense_proportion, fill_value, dtype):
        N = 10**6
        self.array = make_array(N, dense_proportion, fill_value, dtype)  # 创建一个包含N个元素的稀疏数组

    def time_sparse_array(self, dense_proportion, fill_value, dtype):
        SparseArray(self.array, fill_value=fill_value, dtype=dtype)  # 使用数组创建稀疏数组对象


class SparseDataFrameConstructor:
    def setup(self):
        N = 1000
        self.sparse = scipy.sparse.rand(N, N, 0.005)  # 生成一个稀疏矩阵，大小为N x N，稀疏度为0.005

    def time_from_scipy(self):
        pd.DataFrame.sparse.from_spmatrix(self.sparse)  # 从SciPy稀疏矩阵创建稀疏DataFrame


class FromCoo:
    def setup(self):
        self.matrix = scipy.sparse.coo_matrix(
            ([3.0, 1.0, 2.0], ([1, 0, 0], [0, 2, 3])), shape=(100, 100)
        )  # 创建一个COO格式的稀疏矩阵，数据为[3.0, 1.0, 2.0]

    def time_sparse_series_from_coo(self):
        Series.sparse.from_coo(self.matrix)  # 从COO格式的稀疏矩阵创建稀疏系列


class ToCoo:
    params = [True, False]
    param_names = ["sort_labels"]

    def setup(self, sort_labels):
        s = Series([np.nan] * 10000)  # 创建一个包含10000个NaN的系列
        s[0] = 3.0  # 设置索引为0的元素为3.0
        s[100] = -1.0  # 设置索引为100的元素为-1.0
        s[999] = 12.1  # 设置索引为999的元素为12.1

        s_mult_lvl = s.set_axis(MultiIndex.from_product([range(10)] * 4))  # 创建一个多级索引的系列
        self.ss_mult_lvl = s_mult_lvl.astype("Sparse")  # 将系列转换为稀疏格式

        s_two_lvl = s.set_axis(MultiIndex.from_product([range(100)] * 2))  # 创建一个两级索引的系列
        self.ss_two_lvl = s_two_lvl.astype("Sparse")  # 将系列转换为稀疏格式

    def time_sparse_series_to_coo(self, sort_labels):
        self.ss_mult_lvl.sparse.to_coo(
            row_levels=[0, 1], column_levels=[2, 3], sort_labels=sort_labels
        )  # 将多级索引的稀疏系列转换为COO格式

    def time_sparse_series_to_coo_single_level(self, sort_labels):
        self.ss_two_lvl.sparse.to_coo(sort_labels=sort_labels)  # 将两级索引的稀疏系列转换为COO格式


class ToCooFrame:
    def setup(self):
        N = 10000
        k = 10
        arr = np.zeros((N, k), dtype=float)  # 创建一个大小为N x k的全零数组，数据类型为float
        arr[0, 0] = 3.0  # 设置数组第一行第一列的元素为3.0
        arr[12, 7] = -1.0  # 设置数组第13行第8列的元素为-1.0
        arr[0, 9] = 11.2  # 设置数组第一行第10列的元素为11.2
        self.df = pd.DataFrame(arr, dtype=pd.SparseDtype("float", fill_value=0.0))  # 创建一个包含稀疏数据类型的DataFrame

    def time_to_coo(self):
        self.df.sparse.to_coo()  # 将稀疏DataFrame转换为COO格式


class Arithmetic:
    pass  # 空类，用于未来实现
    # 定义参数列表，包含两个元组，每个元组包含两个参数值
    params = ([0.1, 0.01], [0, np.nan])
    # 定义参数名称列表，分别对应 dense_proportion 和 fill_value
    param_names = ["dense_proportion", "fill_value"]

    # 设置方法，用于初始化两个 SparseArray 实例
    def setup(self, dense_proportion, fill_value):
        # 定义数组大小
        N = 10**6
        # 创建稀疏数组 arr1，使用 make_array 函数生成数组数据
        arr1 = make_array(N, dense_proportion, fill_value, np.int64)
        # 使用 SparseArray 类创建 self.array1，指定填充值 fill_value
        self.array1 = SparseArray(arr1, fill_value=fill_value)
        # 创建稀疏数组 arr2，使用 make_array 函数生成数组数据
        arr2 = make_array(N, dense_proportion, fill_value, np.int64)
        # 使用 SparseArray 类创建 self.array2，指定填充值 fill_value
        self.array2 = SparseArray(arr2, fill_value=fill_value)

    # 用于衡量执行 make_union 方法的时间
    def time_make_union(self, dense_proportion, fill_value):
        # 调用 SparseArray 的 sp_index 对象的 make_union 方法
        self.array1.sp_index.make_union(self.array2.sp_index)

    # 用于衡量执行 intersect 方法的时间
    def time_intersect(self, dense_proportion, fill_value):
        # 调用 SparseArray 的 sp_index 对象的 intersect 方法
        self.array1.sp_index.intersect(self.array2.sp_index)

    # 用于衡量执行加法操作的时间
    def time_add(self, dense_proportion, fill_value):
        # 执行 SparseArray 的加法操作
        self.array1 + self.array2

    # 用于衡量执行除法操作的时间
    def time_divide(self, dense_proportion, fill_value):
        # 执行 SparseArray 的除法操作
        self.array1 / self.array2
class ArithmeticBlock:
    params = [np.nan, 0]
    param_names = ["fill_value"]

    def setup(self, fill_value):
        # 设置数组长度
        N = 10**6
        # 创建第一个稀疏数组
        self.arr1 = self.make_block_array(
            length=N, num_blocks=1000, block_size=10, fill_value=fill_value
        )
        # 创建第二个稀疏数组
        self.arr2 = self.make_block_array(
            length=N, num_blocks=1000, block_size=10, fill_value=fill_value
        )

    def make_block_array(self, length, num_blocks, block_size, fill_value):
        # 创建一个全是 fill_value 的数组
        arr = np.full(length, fill_value)
        # 在随机位置生成块值
        indices = np.random.choice(
            np.arange(0, length, block_size), num_blocks, replace=False
        )
        for ind in indices:
            arr[ind : ind + block_size] = np.random.randint(0, 100, block_size)
        # 返回稀疏数组对象
        return SparseArray(arr, fill_value=fill_value)

    def time_make_union(self, fill_value):
        # 计算两个稀疏数组的并集
        self.arr1.sp_index.make_union(self.arr2.sp_index)

    def time_intersect(self, fill_value):
        # 计算两个稀疏数组的交集
        self.arr2.sp_index.intersect(self.arr2.sp_index)

    def time_addition(self, fill_value):
        # 计算两个稀疏数组的加法
        self.arr1 + self.arr2

    def time_division(self, fill_value):
        # 计算两个稀疏数组的除法
        self.arr1 / self.arr2


class MinMax:
    params = (["min", "max"], [0.0, np.nan])
    param_names = ["func", "fill_value"]

    def setup(self, func, fill_value):
        # 设置数组长度
        N = 1_000_000
        # 创建具有指定填充值的数组
        arr = make_array(N, 1e-5, fill_value, np.float64)
        # 创建稀疏数组对象
        self.sp_arr = SparseArray(arr, fill_value=fill_value)

    def time_min_max(self, func, fill_value):
        # 调用稀疏数组对象的最小值或最大值方法
        getattr(self.sp_arr, func)()


class Take:
    params = ([np.array([0]), np.arange(100_000), np.full(100_000, -1)], [True, False])
    param_names = ["indices", "allow_fill"]

    def setup(self, indices, allow_fill):
        # 设置数组长度
        N = 1_000_000
        fill_value = 0.0
        # 创建具有指定填充值的数组
        arr = make_array(N, 1e-5, fill_value, np.float64)
        # 创建稀疏数组对象
        self.sp_arr = SparseArray(arr, fill_value=fill_value)

    def time_take(self, indices, allow_fill):
        # 调用稀疏数组对象的取值方法
        self.sp_arr.take(indices, allow_fill=allow_fill)


class GetItem:
    def setup(self):
        # 设置数组长度
        N = 1_000_000
        d = 1e-5
        # 创建具有指定填充值的数组
        arr = make_array(N, d, np.nan, np.float64)
        # 创建稀疏数组对象
        self.sp_arr = SparseArray(arr)

    def time_integer_indexing(self):
        # 访问稀疏数组对象的整数索引
        self.sp_arr[78]

    def time_slice(self):
        # 访问稀疏数组对象的切片
        self.sp_arr[1:]


class GetItemMask:
    params = [True, False, np.nan]
    param_names = ["fill_value"]

    def setup(self, fill_value):
        # 设置数组长度
        N = 1_000_000
        d = 1e-5
        # 创建具有指定填充值的数组
        arr = make_array(N, d, np.nan, np.float64)
        # 创建稀疏数组对象
        self.sp_arr = SparseArray(arr)
        # 创建布尔数组，表示是否使用 fill_value
        b_arr = np.full(shape=N, fill_value=fill_value, dtype=np.bool_)
        # 随机选择索引并设置布尔值
        fv_inds = np.unique(
            np.random.randint(low=0, high=N - 1, size=int(N * d), dtype=np.int32)
        )
        b_arr[fv_inds] = True if pd.isna(fill_value) else not fill_value
        # 创建稀疏布尔数组对象
        self.sp_b_arr = SparseArray(b_arr, dtype=np.bool_, fill_value=fill_value)

    def time_mask(self, fill_value):
        # 使用稀疏布尔数组对象来进行掩码操作
        self.sp_arr[self.sp_b_arr]
```