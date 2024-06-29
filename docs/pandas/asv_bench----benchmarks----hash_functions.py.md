# `D:\src\scipysrc\pandas\asv_bench\benchmarks\hash_functions.py`

```
import numpy as np  # 导入 NumPy 库
import pandas as pd  # 导入 Pandas 库


class UniqueForLargePyObjectInts:
    def setup(self):
        # 创建一个包含左移操作的列表，每个元素左移 32 位，范围是 0 到 4999
        lst = [x << 32 for x in range(5000)]
        # 将列表转换为 NumPy 数组，数据类型为 np.object_
        self.arr = np.array(lst, dtype=np.object_)

    def time_unique(self):
        # 调用 Pandas 的 unique 函数，返回数组中的唯一值
        pd.unique(self.arr)


class Float64GroupIndex:
    # 类说明：GH28303
    def setup(self):
        # 创建包含从 "2018年1月1日" 到 "2018年1月2日" 的日期范围，共 10**6 个时间戳
        self.df = pd.date_range(
            start="1/1/2018", end="1/2/2018", periods=10**6
        ).to_frame()
        # 将日期索引转换为整数后除以 10**9 并四舍五入，作为分组索引
        self.group_index = np.round(self.df.index.astype(int) / 10**9)

    def time_groupby(self):
        # 对 DataFrame 按照分组索引进行分组，并取每组的最后一个元素
        self.df.groupby(self.group_index).last()


class UniqueAndFactorizeArange:
    params = range(4, 16)
    param_names = ["exponent"]

    def setup(self, exponent):
        # 创建一个包含 10000 个元素的浮点数数组 a
        a = np.arange(10**4, dtype="float64")
        # 将数组 a 中的每个元素加上 10**exponent，重复 100 次
        self.a2 = (a + 10**exponent).repeat(100)

    def time_factorize(self, exponent):
        # 调用 Pandas 的 factorize 函数，返回数组的唯一值和对应的标签
        pd.factorize(self.a2)

    def time_unique(self, exponent):
        # 调用 Pandas 的 unique 函数，返回数组中的唯一值
        pd.unique(self.a2)


class Unique:
    params = ["Int64", "Float64"]
    param_names = ["dtype"]

    def setup(self, dtype):
        # 创建一个包含重复值的 Pandas Series 对象
        self.ser = pd.Series(([1, pd.NA, 2] + list(range(100_000))) * 3, dtype=dtype)
        # 创建一个包含唯一值的 Pandas Series 对象
        self.ser_unique = pd.Series(list(range(300_000)) + [pd.NA], dtype=dtype)

    def time_unique_with_duplicates(self, exponent):
        # 调用 Pandas 的 unique 函数，返回数组中的唯一值
        pd.unique(self.ser)

    def time_unique(self, exponent):
        # 调用 Pandas 的 unique 函数，返回数组中的唯一值
        pd.unique(self.ser_unique)


class NumericSeriesIndexing:
    params = [
        (np.int64, np.uint64, np.float64),
        (10**4, 10**5, 5 * 10**5, 10**6, 5 * 10**6),
    ]
    param_names = ["dtype", "N"]

    def setup(self, dtype, N):
        # 创建一个包含索引值的 Pandas Index 对象
        vals = np.array(list(range(55)) + [54] + list(range(55, N - 1)), dtype=dtype)
        indices = pd.Index(vals)
        # 创建一个包含从 0 到 N-1 的整数的 Pandas Series 对象
        self.data = pd.Series(np.arange(N), index=indices)

    def time_loc_slice(self, index, N):
        # 触发索引映射的构建
        self.data.loc[:800]


class NumericSeriesIndexingShuffled:
    params = [
        (np.int64, np.uint64, np.float64),
        (10**4, 10**5, 5 * 10**5, 10**6, 5 * 10**6),
    ]
    param_names = ["dtype", "N"]

    def setup(self, dtype, N):
        # 创建一个包含索引值的 NumPy 数组，并对其进行随机重排
        vals = np.array(list(range(55)) + [54] + list(range(55, N - 1)), dtype=dtype)
        np.random.shuffle(vals)
        indices = pd.Index(vals)
        # 创建一个包含从 0 到 N-1 的整数的 Pandas Series 对象
        self.data = pd.Series(np.arange(N), index=indices)

    def time_loc_slice(self, index, N):
        # 触发索引映射的构建
        self.data.loc[:800]
```