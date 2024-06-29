# `D:\src\scipysrc\pandas\asv_bench\benchmarks\frame_ctor.py`

```
import numpy as np

import pandas as pd
from pandas import (
    NA,
    Categorical,
    DataFrame,
    Float64Dtype,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
)

try:
    from pandas.tseries.offsets import (
        Hour,
        Nano,
    )
except ImportError:
    # 对于兼容性较老的版本
    from pandas.core.datetools import (
        Hour,
        Nano,
    )


class FromDicts:
    def setup(self):
        N, K = 5000, 50
        # 创建一个包含对象类型索引的索引对象
        self.index = pd.Index([f"i-{i}" for i in range(N)], dtype=object)
        # 创建一个包含对象类型索引的列索引对象
        self.columns = pd.Index([f"i-{i}" for i in range(K)], dtype=object)
        # 生成一个随机数据帧，并转换成字典形式
        frame = DataFrame(np.random.randn(N, K), index=self.index, columns=self.columns)
        self.data = frame.to_dict()
        # 将数据帧转换为记录（行）的字典列表
        self.dict_list = frame.to_dict(orient="records")
        # 创建一个嵌套字典，用于测试
        self.data2 = {i: {j: float(j) for j in range(100)} for i in range(2000)}

        # 不需要合并的分类数组字典
        self.dict_of_categoricals = {i: Categorical(np.arange(N)) for i in range(K)}

    def time_list_of_dict(self):
        DataFrame(self.dict_list)

    def time_nested_dict(self):
        DataFrame(self.data)

    def time_nested_dict_index(self):
        DataFrame(self.data, index=self.index)

    def time_nested_dict_columns(self):
        DataFrame(self.data, columns=self.columns)

    def time_nested_dict_index_columns(self):
        DataFrame(self.data, index=self.index, columns=self.columns)

    def time_nested_dict_int64(self):
        # 嵌套字典，带整数索引，用于回归测试
        DataFrame(self.data2)

    def time_dict_of_categoricals(self):
        # 不需要合并的分类数组字典
        DataFrame(self.dict_of_categoricals)


class FromSeries:
    def setup(self):
        # 创建一个包含100x100 MultiIndex 的Series
        mi = MultiIndex.from_product([range(100), range(100)])
        self.s = Series(np.random.randn(10000), index=mi)

    def time_mi_series(self):
        DataFrame(self.s)


class FromDictwithTimestamp:
    params = [Nano(1), Hour(1)]
    param_names = ["offset"]

    def setup(self, offset):
        N = 10**3
        # 使用给定的偏移量创建日期范围
        idx = date_range(Timestamp("1/1/1900"), freq=offset, periods=N)
        df = DataFrame(np.random.randn(N, 10), index=idx)
        self.d = df.to_dict()

    def time_dict_with_timestamp_offsets(self, offset):
        DataFrame(self.d)


class FromRecords:
    params = [None, 1000]
    param_names = ["nrows"]

    # 生成器在使用后被消耗，所以在每次调用前都需要运行setup
    number = 1
    repeat = (3, 250, 10)

    def setup(self, nrows):
        N = 100000
        # 创建一个生成器，生成元组 (x, x*20, x*100) 的序列
        self.gen = ((x, (x * 20), (x * 100)) for x in range(N))

    def time_frame_from_records_generator(self, nrows):
        # issue-6700
        # 使用从生成器中生成的元组创建数据帧
        self.df = DataFrame.from_records(self.gen, nrows=nrows)


class FromNDArray:
    def setup(self):
        N = 100000
        # 创建一个包含随机数据的NumPy数组
        self.data = np.random.randn(N)

    def time_frame_from_ndarray(self):
        # 使用NumPy数组创建数据帧
        self.df = DataFrame(self.data)


class FromLists:
    goal_time = 0.2
    # 在类的setup方法中初始化数据
    def setup(self):
        # 设定常量N为1000
        N = 1000
        # 设定常量M为100
        M = 100
        # 创建一个包含N个子列表的列表，每个子列表包含从0到M-1的整数
        self.data = [list(range(M)) for i in range(N)]

    # 创建DataFrame对象，使用类中的self.data作为数据源
    def time_frame_from_lists(self):
        self.df = DataFrame(self.data)
# 定义一个名为 FromRange 的类，用于创建 DataFrame 对象
class FromRange:
    # 设定一个类级别的变量 goal_time，并初始化为 0.2
    goal_time = 0.2

    # 定义 setup 方法，用于设置数据
    def setup(self):
        # 设定变量 N 为 1,000,000
        N = 1_000_000
        # 创建一个包含 N 个元素的 range 对象，并赋值给实例变量 self.data
        self.data = range(N)

    # 定义一个名为 time_frame_from_range 的方法，用于生成 DataFrame 对象
    def time_frame_from_range(self):
        # 使用 self.data 创建 DataFrame 对象，并赋值给实例变量 self.df
        self.df = DataFrame(self.data)


# 定义一个名为 FromScalar 的类，用于创建不同类型的 DataFrame 对象
class FromScalar:
    # 定义 setup 方法，用于设置数据
    def setup(self):
        # 设定实例变量 self.nrows 为 100,000
        self.nrows = 100_000

    # 定义一个名为 time_frame_from_scalar_ea_float64 的方法，生成指定类型的 DataFrame 对象
    def time_frame_from_scalar_ea_float64(self):
        # 创建一个包含浮点数 1.0 的 DataFrame 对象，指定索引和列名，并使用 Float64Dtype 类型
        DataFrame(
            1.0,
            index=range(self.nrows),
            columns=list("abc"),
            dtype=Float64Dtype(),
        )

    # 定义一个名为 time_frame_from_scalar_ea_float64_na 的方法，生成包含 NA 值的 DataFrame 对象
    def time_frame_from_scalar_ea_float64_na(self):
        # 创建一个包含 NA 值的 DataFrame 对象，指定索引和列名，并使用 Float64Dtype 类型
        DataFrame(
            NA,
            index=range(self.nrows),
            columns=list("abc"),
            dtype=Float64Dtype(),
        )


# 定义一个名为 FromArrays 的类，用于从数组创建 DataFrame 对象
class FromArrays:
    # 设定一个类级别的变量 goal_time，并初始化为 0.2
    goal_time = 0.2

    # 定义 setup 方法，用于设置数据
    def setup(self):
        # 设定 N_rows 和 N_cols 的值分别为 1000
        N_rows = 1000
        N_cols = 1000
        # 创建包含 N_cols 个元素的浮点数数组列表，并赋值给实例变量 self.float_arrays
        self.float_arrays = [np.random.randn(N_rows) for _ in range(N_cols)]
        # 创建包含 N_cols 个稀疏数组列表，并赋值给实例变量 self.sparse_arrays
        self.sparse_arrays = [
            pd.arrays.SparseArray(np.random.randint(0, 2, N_rows), dtype="float64")
            for _ in range(N_cols)
        ]
        # 创建包含 N_cols 个整数数组列表，并赋值给实例变量 self.int_arrays
        self.int_arrays = [
            pd.array(np.random.randint(1000, size=N_rows), dtype="Int64")
            for _ in range(N_cols)
        ]
        # 创建一个包含 N_rows 个元素的索引，并赋值给实例变量 self.index
        self.index = pd.Index(range(N_rows))
        # 创建一个包含 N_cols 个元素的列索引，并赋值给实例变量 self.columns
        self.columns = pd.Index(range(N_cols))

    # 定义一个名为 time_frame_from_arrays_float 的方法，从浮点数数组创建 DataFrame 对象
    def time_frame_from_arrays_float(self):
        # 使用 self.float_arrays 创建 DataFrame 对象，并指定索引和列索引
        self.df = DataFrame._from_arrays(
            self.float_arrays,
            index=self.index,
            columns=self.columns,
            verify_integrity=False,
        )

    # 定义一个名为 time_frame_from_arrays_int 的方法，从整数数组创建 DataFrame 对象
    def time_frame_from_arrays_int(self):
        # 使用 self.int_arrays 创建 DataFrame 对象，并指定索引和列索引
        self.df = DataFrame._from_arrays(
            self.int_arrays,
            index=self.index,
            columns=self.columns,
            verify_integrity=False,
        )

    # 定义一个名为 time_frame_from_arrays_sparse 的方法，从稀疏数组创建 DataFrame 对象
    def time_frame_from_arrays_sparse(self):
        # 使用 self.sparse_arrays 创建 DataFrame 对象，并指定索引和列索引
        self.df = DataFrame._from_arrays(
            self.sparse_arrays,
            index=self.index,
            columns=self.columns,
            verify_integrity=False,
        )


# 导入 pandas_vb_common 模块中的 setup 函数，用于设置环境
from .pandas_vb_common import setup  # noqa: F401 isort:skip
```