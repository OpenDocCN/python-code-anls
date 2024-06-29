# `D:\src\scipysrc\pandas\asv_bench\benchmarks\frame_methods.py`

```
# 导入字符串和警告模块
import string
import warnings

# 导入 NumPy 库并使用 np 别名
import numpy as np

# 从 pandas 库中导入特定类和函数
from pandas import (
    DataFrame,        # 数据帧类
    Index,            # 索引类
    MultiIndex,       # 多重索引类
    NaT,              # 表示缺失时间的常量
    Series,           # 系列类
    date_range,       # 生成日期范围的函数
    isnull,           # 检查是否为空值的函数
    period_range,     # 生成周期范围的函数
    timedelta_range,  # 生成时间差范围的函数
)

# 定义类 AsType
class AsType:
    # 参数化测试参数
    params = [
        [
            # 同类型转换
            ("Float64", "Float64"),
            ("float64[pyarrow]", "float64[pyarrow]"),
            # 从非EA到EA类型转换
            ("float64", "Float64"),
            ("float64", "float64[pyarrow]"),
            # 从EA到非EA类型转换
            ("Float64", "float64"),
            ("float64[pyarrow]", "float64"),
            # EA到EA类型转换
            ("Int64", "Float64"),
            ("int64[pyarrow]", "float64[pyarrow]"),
        ],
        [False, True],
    ]
    # 参数名字
    param_names = ["from_to_dtypes", "copy"]

    # 设置测试的初始化
    def setup(self, from_to_dtypes, copy):
        from_dtype = from_to_dtypes[0]
        # 根据不同的类型生成随机数据
        if from_dtype in ("float64", "Float64", "float64[pyarrow]"):
            data = np.random.randn(100, 100)
        elif from_dtype in ("int64", "Int64", "int64[pyarrow]"):
            data = np.random.randint(0, 1000, (100, 100))
        else:
            raise NotImplementedError
        # 创建数据帧对象并存储在 self.df 中
        self.df = DataFrame(data, dtype=from_dtype)

    # 测试类型转换函数的性能
    def time_astype(self, from_to_dtypes, copy):
        self.df.astype(from_to_dtypes[1], copy=copy)


# 定义类 Clip
class Clip:
    # 参数化测试参数
    params = [
        ["float64", "Float64", "float64[pyarrow]"],
    ]
    # 参数名字
    param_names = ["dtype"]

    # 设置测试的初始化
    def setup(self, dtype):
        # 生成随机数据并创建数据帧对象并存储在 self.df 中
        data = np.random.randn(100_000, 10)
        df = DataFrame(data, dtype=dtype)
        self.df = df

    # 测试数据帧对象的剪裁功能的性能
    def time_clip(self, dtype):
        self.df.clip(-1.0, 1.0)


# 定义类 GetNumericData
class GetNumericData:
    # 设置测试的初始化
    def setup(self):
        # 生成随机数据并创建数据帧对象并存储在 self.df 中
        self.df = DataFrame(np.random.randn(10000, 25))
        self.df["foo"] = "bar"
        self.df["bar"] = "baz"
        # 优化数据帧对象以仅保留数值类型的列并重新存储在 self.df 中
        self.df = self.df._consolidate()

    # 测试获取数据帧对象数值数据的性能
    def time_frame_get_numeric_data(self):
        self.df._get_numeric_data()


# 定义类 Reindex
class Reindex:
    # 设置测试的初始化
    def setup(self):
        N = 10**3
        # 生成大量随机数据并创建数据帧对象并存储在 self.df 中
        self.df = DataFrame(np.random.randn(N * 10, N))
        # 创建一个索引数组并存储在 self.idx 中
        self.idx = np.arange(4 * N, 7 * N)
        # 创建一个列索引数组并存储在 self.idx_cols 中
        self.idx_cols = np.random.randint(0, N, N)
        # 创建一个包含不同数据类型的数据帧对象并存储在 self.df2 中
        self.df2 = DataFrame(
            {
                c: {
                    0: np.random.randint(0, 2, N).astype(np.bool_),
                    1: np.random.randint(0, N, N).astype(np.int16),
                    2: np.random.randint(0, N, N).astype(np.int32),
                    3: np.random.randint(0, N, N).astype(np.int64),
                }[np.random.randint(0, 4)]
                for c in range(N)
            }
        )

    # 测试在轴0上重新索引数据帧对象的性能
    def time_reindex_axis0(self):
        self.df.reindex(self.idx)

    # 测试在轴1上重新索引数据帧对象的性能
    def time_reindex_axis1(self):
        self.df.reindex(columns=self.idx_cols)

    # 测试在轴1上重新索引数据帧对象的性能（用与索引数组不匹配的列索引）
    def time_reindex_axis1_missing(self):
        self.df.reindex(columns=self.idx)

    # 测试同时在两个轴上重新索引数据帧对象的性能
    def time_reindex_both_axes(self):
        self.df.reindex(index=self.idx, columns=self.idx_cols)
    # 定义一个方法 `time_reindex_upcast`，用于重新索引 self.df2 数据框，将索引随机重排。
    def time_reindex_upcast(self):
        # 使用 np.random.permutation 生成一个 0 到 1199 的随机排列，然后用这个排列来重新索引 self.df2 数据框。
        self.df2.reindex(np.random.permutation(range(1200)))
class Rename:
    # 定义一个重命名类
    def setup(self):
        # 设置函数，初始化数据框 df，使用随机数据
        N = 10**3
        self.df = DataFrame(np.random.randn(N * 10, N))
        # 创建一个包含 10*N 行、N 列的随机数数据框 df
        self.idx = np.arange(4 * N, 7 * N)
        # 创建一个包含从 4*N 到 7*N 的索引数组 idx
        self.dict_idx = {k: k for k in self.idx}
        # 创建一个以 idx 为键和值的字典 dict_idx
        self.df2 = DataFrame(
            {
                c: {
                    0: np.random.randint(0, 2, N).astype(np.bool_),
                    1: np.random.randint(0, N, N).astype(np.int16),
                    2: np.random.randint(0, N, N).astype(np.int32),
                    3: np.random.randint(0, N, N).astype(np.int64),
                }[np.random.randint(0, 4)]
                for c in range(N)
            }
        )
        # 创建一个包含多种数据类型的数据框 df2

    def time_rename_single(self):
        # 测试单轴重命名的性能
        self.df.rename({0: 0})

    def time_rename_axis0(self):
        # 测试轴 0 重命名的性能
        self.df.rename(self.dict_idx)

    def time_rename_axis1(self):
        # 测试轴 1 重命名的性能
        self.df.rename(columns=self.dict_idx)

    def time_rename_both_axes(self):
        # 测试同时重命名轴 0 和轴 1 的性能
        self.df.rename(index=self.dict_idx, columns=self.dict_idx)

    def time_dict_rename_both_axes(self):
        # 测试使用字典重命名同时轴 0 和轴 1 的性能
        self.df.rename(index=self.dict_idx, columns=self.dict_idx)


class Iteration:
    # 迭代类
    # mem_itertuples_* benchmarks are slow
    timeout = 120  # 设置超时时间为 120 秒

    def setup(self):
        # 设置函数，初始化多个数据框，使用随机数据
        N = 1000
        self.df = DataFrame(np.random.randn(N * 10, N))
        # 创建一个包含 10*N 行、N 列的随机数数据框 df
        self.df2 = DataFrame(np.random.randn(N * 50, 10))
        # 创建一个包含 50*N 行、10 列的随机数数据框 df2
        self.df3 = DataFrame(
            np.random.randn(N, 5 * N), columns=["C" + str(c) for c in range(N * 5)]
        )
        # 创建一个包含 N 行、5*N 列的随机数数据框 df3，并设置列名为 C0 至 C4999
        self.df4 = DataFrame(np.random.randn(N * 1000, 10))
        # 创建一个包含 1000*N 行、10 列的随机数数据框 df4

    def time_items(self):
        # 测试遍历列名和列数据的性能
        # (monitor no-copying behaviour)
        for name, col in self.df.items():
            pass

    def time_iteritems_indexing(self):
        # 测试通过遍历 df3 的列名来访问列数据的性能
        for col in self.df3:
            self.df3[col]

    def time_itertuples_start(self):
        # 测试开始迭代 df4 的性能
        self.df4.itertuples()

    def time_itertuples_read_first(self):
        # 测试读取 df4 的第一个元组的性能
        next(self.df4.itertuples())

    def time_itertuples(self):
        # 测试遍历 df4 所有元组的性能
        for row in self.df4.itertuples():
            pass

    def time_itertuples_to_list(self):
        # 测试将 df4 所有元组转换为列表的性能
        list(self.df4.itertuples())

    def mem_itertuples_start(self):
        # 返回开始迭代 df4 时的内存使用情况
        return self.df4.itertuples()

    def peakmem_itertuples_start(self):
        # 记录开始迭代 df4 时的内存峰值使用情况
        self.df4.itertuples()

    def mem_itertuples_read_first(self):
        # 返回读取 df4 第一个元组时的内存使用情况
        return next(self.df4.itertuples())

    def peakmem_itertuples(self):
        # 记录遍历 df4 所有元组时的内存峰值使用情况
        for row in self.df4.itertuples():
            pass

    def mem_itertuples_to_list(self):
        # 返回将 df4 所有元组转换为列表时的内存使用情况
        return list(self.df4.itertuples())

    def peakmem_itertuples_to_list(self):
        # 记录将 df4 所有元组转换为列表时的内存峰值使用情况
        list(self.df4.itertuples())

    def time_itertuples_raw_start(self):
        # 测试开始迭代 df4（无索引，无名称）的性能
        self.df4.itertuples(index=False, name=None)

    def time_itertuples_raw_read_first(self):
        # 测试读取 df4 的第一个元组（无索引，无名称）的性能
        next(self.df4.itertuples(index=False, name=None))

    def time_itertuples_raw_tuples(self):
        # 测试遍历 df4 所有元组（无索引，无名称）的性能
        for row in self.df4.itertuples(index=False, name=None):
            pass

    def time_itertuples_raw_tuples_to_list(self):
        # 测试将 df4 所有元组（无索引，无名称）转换为列表的性能
        list(self.df4.itertuples(index=False, name=None))
    # 返回一个不包含索引的DataFrame的迭代器，生成的元组中不包含行索引
    def mem_itertuples_raw_start(self):
        return self.df4.itertuples(index=False, name=None)

    # 在peak memory模式下，调用DataFrame的itertuples方法，但不返回结果
    def peakmem_itertuples_raw_start(self):
        self.df4.itertuples(index=False, name=None)

    # 在peak memory模式下，通过调用next函数获取DataFrame的第一个元组
    def peakmem_itertuples_raw_read_first(self):
        next(self.df4.itertuples(index=False, name=None))

    # 在peak memory模式下，遍历DataFrame的每一行元组，但不执行任何操作
    def peakmem_itertuples_raw(self):
        for row in self.df4.itertuples(index=False, name=None):
            pass

    # 返回一个包含DataFrame所有行元组的列表
    def mem_itertuples_raw_to_list(self):
        return list(self.df4.itertuples(index=False, name=None))

    # 在peak memory模式下，调用DataFrame的itertuples方法，将生成的元组转换为列表，但不返回结果
    def peakmem_itertuples_raw_to_list(self):
        list(self.df4.itertuples(index=False, name=None))

    # 遍历DataFrame的每一行，但不执行任何操作
    def time_iterrows(self):
        for row in self.df.iterrows():
            pass
class ToString:
    # 初始化方法，生成一个包含随机数据的 DataFrame 对象
    def setup(self):
        self.df = DataFrame(np.random.randn(100, 10))

    # 将 DataFrame 对象转换为字符串表示
    def time_to_string_floats(self):
        self.df.to_string()


class ToHTML:
    # 初始化方法，生成一个包含随机数据的 DataFrame 对象，包含两列特殊数据
    def setup(self):
        nrows = 500
        self.df2 = DataFrame(np.random.randn(nrows, 10))
        self.df2[0] = period_range("2000", periods=nrows)
        self.df2[1] = range(nrows)

    # 将 DataFrame 对象转换为 HTML 表格表示
    def time_to_html_mixed(self):
        self.df2.to_html()


class ToDict:
    # 定义参数和参数名称列表
    params = [["dict", "list", "series", "split", "records", "index"]]
    param_names = ["orient"]

    # 初始化方法，生成包含随机整数数据和时间相关数据的 DataFrame 对象
    def setup(self, orient):
        data = np.random.randint(0, 1000, size=(10000, 4))
        self.int_df = DataFrame(data)
        self.datetimelike_df = self.int_df.astype("timedelta64[ns]")

    # 将整数类型的 DataFrame 转换为字典对象
    def time_to_dict_ints(self, orient):
        self.int_df.to_dict(orient=orient)

    # 将时间相关类型的 DataFrame 转换为字典对象
    def time_to_dict_datetimelike(self, orient):
        self.datetimelike_df.to_dict(orient=orient)


class ToNumpy:
    # 初始化方法，生成包含随机数据的不同形态的 DataFrame 对象
    def setup(self):
        N = 10000
        M = 10
        self.df_tall = DataFrame(np.random.randn(N, M))
        self.df_wide = DataFrame(np.random.randn(M, N))
        self.df_mixed_tall = self.df_tall.copy()
        self.df_mixed_tall["foo"] = "bar"
        self.df_mixed_tall[0] = period_range("2000", periods=N)
        self.df_mixed_tall[1] = range(N)
        self.df_mixed_wide = self.df_wide.copy()
        self.df_mixed_wide["foo"] = "bar"
        self.df_mixed_wide[0] = period_range("2000", periods=M)
        self.df_mixed_wide[1] = range(M)

    # 将高维的 DataFrame 对象转换为 NumPy 数组
    def time_to_numpy_tall(self):
        self.df_tall.to_numpy()

    # 将宽维的 DataFrame 对象转换为 NumPy 数组
    def time_to_numpy_wide(self):
        self.df_wide.to_numpy()

    # 将包含混合数据的高维 DataFrame 对象转换为 NumPy 数组
    def time_to_numpy_mixed_tall(self):
        self.df_mixed_tall.to_numpy()

    # 将包含混合数据的宽维 DataFrame 对象转换为 NumPy 数组
    def time_to_numpy_mixed_wide(self):
        self.df_mixed_wide.to_numpy()

    # 获取高维 DataFrame 对象的值作为 NumPy 数组
    def time_values_tall(self):
        self.df_tall.values

    # 获取宽维 DataFrame 对象的值作为 NumPy 数组
    def time_values_wide(self):
        self.df_wide.values

    # 获取包含混合数据的高维 DataFrame 对象的值作为 NumPy 数组
    def time_values_mixed_tall(self):
        self.df_mixed_tall.values

    # 获取包含混合数据的宽维 DataFrame 对象的值作为 NumPy 数组
    def time_values_mixed_wide(self):
        self.df_mixed_wide.values


class ToRecords:
    # 初始化方法，生成包含随机数据的 DataFrame 对象，其中包含一个 MultiIndex
    def setup(self):
        N = 100_000
        data = np.random.randn(N, 2)
        mi = MultiIndex.from_arrays(
            [
                np.arange(N),
                date_range("1970-01-01", periods=N, freq="ms"),
            ]
        )
        self.df = DataFrame(data)
        self.df_mi = DataFrame(data, index=mi)

    # 将 DataFrame 对象转换为结构化 NumPy 数组
    def time_to_records(self):
        self.df.to_records(index=True)

    # 将包含 MultiIndex 的 DataFrame 对象转换为结构化 NumPy 数组
    def time_to_records_multiindex(self):
        self.df_mi.to_records(index=True)


class Repr:
    # 此处为示例代码的最后一个类，不包含任何方法或属性，无需注释
    pass
    # 设置函数，在测试中初始化数据和DataFrame对象
    def setup(self):
        # 定义行数为10000
        nrows = 10000
        # 创建一个形状为(nrows, 10)的随机数据数组
        data = np.random.randn(nrows, 10)
        # 创建一个形状为(3, nrows // 100)的随机数据数组，重复100次，形成一个数组
        arrays = np.tile(np.random.randn(3, nrows // 100), 100)
        # 使用数组创建一个多级索引对象
        idx = MultiIndex.from_arrays(arrays)
        # 创建一个DataFrame对象self.df3，数据为data，索引为idx
        self.df3 = DataFrame(data, index=idx)
        # 创建一个DataFrame对象self.df4，数据为data，索引为形状为(nrows,)的随机数据数组
        self.df4 = DataFrame(data, index=np.random.randn(nrows))
        # 创建一个形状为(nrows, 10)的随机数据的DataFrame对象self.df_tall
        self.df_tall = DataFrame(np.random.randn(nrows, 10))
        # 创建一个形状为(10, nrows)的随机数据的DataFrame对象self.df_wide
        self.df_wide = DataFrame(np.random.randn(10, nrows))

    # 测试方法，生成self.df3的HTML表示形式（截断）
    def time_html_repr_trunc_mi(self):
        self.df3._repr_html_()

    # 测试方法，生成self.df4的HTML表示形式（截断）
    def time_html_repr_trunc_si(self):
        self.df4._repr_html_()

    # 测试方法，生成self.df_tall的字符串表示形式
    def time_repr_tall(self):
        repr(self.df_tall)

    # 测试方法，生成self.df_wide的字符串表示形式
    def time_frame_repr_wide(self):
        repr(self.df_wide)
class MaskBool:
    # 初始化方法，生成随机数据并创建 DataFrame 对象
    def setup(self):
        data = np.random.randn(1000, 500)
        df = DataFrame(data)
        # 将 DataFrame 中小于等于 0 的元素替换为 NaN
        df = df.where(df > 0)
        # 生成布尔型 DataFrame，标记大于 0 的元素为 True，其余为 False
        self.bools = df > 0
        # 生成 NaN 掩码，标记 DataFrame 中的 NaN 元素
        self.mask = isnull(df)

    # 用于测试的方法，未修改 self.bools 的值
    def time_frame_mask_bools(self):
        self.bools.mask(self.mask)

    # 用于测试的方法，将 self.bools 转换为 float 类型后应用 mask
    def time_frame_mask_floats(self):
        self.bools.astype(float).mask(self.mask)


class Isnull:
    # 初始化方法，生成多种类型数据的 DataFrame 对象
    def setup(self):
        N = 10**3
        self.df_no_null = DataFrame(np.random.randn(N, N))

        # 生成包含 NaN 的数据
        sample = np.array([np.nan, 1.0])
        data = np.random.choice(sample, (N, N))
        self.df = DataFrame(data)

        # 生成包含字符串的数据
        sample = np.array(list(string.ascii_letters + string.whitespace))
        data = np.random.choice(sample, (N, N))
        self.df_strings = DataFrame(data)

        # 生成包含各种对象类型的数据
        sample = np.array(
            [
                NaT,
                np.nan,
                None,
                np.datetime64("NaT"),
                np.timedelta64("NaT"),
                0,
                1,
                2.0,
                "",
                "abcd",
            ]
        )
        data = np.random.choice(sample, (N, N))
        self.df_obj = DataFrame(data)

    # 用于测试的方法，检查无 NaN 数据的 DataFrame
    def time_isnull_floats_no_null(self):
        isnull(self.df_no_null)

    # 用于测试的方法，检查包含 NaN 数据的 DataFrame
    def time_isnull(self):
        isnull(self.df)

    # 用于测试的方法，检查包含字符串数据的 DataFrame
    def time_isnull_strngs(self):
        isnull(self.df_strings)

    # 用于测试的方法，检查包含对象类型数据的 DataFrame
    def time_isnull_obj(self):
        isnull(self.df_obj)


class Fillna:
    params = (
        [True, False],  # 填充是否原地修改的参数
        [
            "float64",
            "float32",
            "object",
            "Int64",
            "Float64",
            "datetime64[ns]",
            "datetime64[ns, tz]",
            "timedelta64[ns]",
        ],  # 数据类型的参数
    )
    param_names = ["inplace", "dtype"]  # 参数的名称

    # 初始化方法，生成指定数据类型的 DataFrame 对象
    def setup(self, inplace, dtype):
        N, M = 10000, 100
        if dtype in ("datetime64[ns]", "datetime64[ns, tz]", "timedelta64[ns]"):
            # 生成指定类型的时间数据
            data = {
                "datetime64[ns]": date_range("2011-01-01", freq="h", periods=N),
                "datetime64[ns, tz]": date_range(
                    "2011-01-01", freq="h", periods=N, tz="Asia/Tokyo"
                ),
                "timedelta64[ns]": timedelta_range(start="1 day", periods=N, freq="1D"),
            }
            self.df = DataFrame({f"col_{i}": data[dtype] for i in range(M)})
            # 随机将部分数据设为 NaN
            self.df[::2] = None
        else:
            # 生成随机数据
            values = np.random.randn(N, M)
            # 随机将部分数据设为 NaN
            values[::2] = np.nan
            if dtype == "Int64":
                values = values.round()
            self.df = DataFrame(values, dtype=dtype)
        # 获取第一个有效索引位置的填充值字典
        self.fill_values = self.df.iloc[self.df.first_valid_index()].to_dict()

    # 用于测试的方法，填充 NaN 值
    def time_fillna(self, inplace, dtype):
        self.df.fillna(value=self.fill_values, inplace=inplace)

    # 用于测试的方法，向前填充缺失值
    def time_ffill(self, inplace, dtype):
        self.df.ffill(inplace=inplace)

    # 用于测试的方法，向后填充缺失值
    def time_bfill(self, inplace, dtype):
        self.df.bfill(inplace=inplace)


class Dropna:
    params = (["all", "any"], [0, 1])  # 参数列表：如何丢弃和轴向
    param_names = ["how", "axis"]  # 参数名称列表
    # 设置函数，用于初始化数据帧，并插入缺失值
    def setup(self, how, axis):
        # 创建一个 10000x1000 的随机数数据帧
        self.df = DataFrame(np.random.randn(10000, 1000))
        # 在指定的区域插入 NaN 值
        self.df.iloc[50:1000, 20:50] = np.nan
        # 在指定的行范围内全部插入 NaN 值
        self.df.iloc[2000:3000] = np.nan
        # 在指定的列范围内全部插入 NaN 值
        self.df.iloc[:, 60:70] = np.nan
        # 复制数据帧以创建混合数据类型的版本
        self.df_mixed = self.df.copy()
        # 在复制的数据帧中添加一列 'foo'，所有值设为 'bar'
        self.df_mixed["foo"] = "bar"

    # 测试函数，用于测试在指定条件下删除 NaN 值的性能
    def time_dropna(self, how, axis):
        # 调用数据帧的 dropna 方法删除 NaN 值，不返回结果
        self.df.dropna(how=how, axis=axis)

    # 测试函数，用于测试在混合数据类型的数据帧中删除 NaN 值的性能
    def time_dropna_axis_mixed_dtypes(self, how, axis):
        # 调用混合数据类型数据帧的 dropna 方法删除 NaN 值，不返回结果
        self.df_mixed.dropna(how=how, axis=axis)
class Isna:
    # 参数列表，指定了三种可能的数据类型
    params = ["float64", "Float64", "float64[pyarrow]"]
    # 参数名称，仅有一个参数 dtype
    param_names = ["dtype"]

    # 设置方法，用于初始化测试数据
    def setup(self, dtype):
        # 生成一个 10000x1000 的随机数据数组
        data = np.random.randn(10000, 1000)
        # 将第 600 到 800 列的数据设置为全为 NaN
        data[:, 600:800] = np.nan
        # 将第 800 到 1000 行、第 4000 到 5000 列的数据设置为 NaN
        data[800:1000, 4000:5000] = np.nan
        # 使用 DataFrame 类创建 DataFrame 对象，传入数据和指定的 dtype
        self.df = DataFrame(data, dtype=dtype)

    # 测试 isna 方法的执行时间
    def time_isna(self, dtype):
        # 调用 DataFrame 的 isna 方法
        self.df.isna()


class Count:
    # 参数列表，axis 参数可以是 0 或 1
    params = [0, 1]
    # 参数名称，仅有一个参数 axis
    param_names = ["axis"]

    # 设置方法，用于初始化测试数据
    def setup(self, axis):
        # 生成一个 10000x1000 的随机数据数组
        self.df = DataFrame(np.random.randn(10000, 1000))
        # 将第 50 到 1000 行、第 20 到 50 列的数据设置为 NaN
        self.df.iloc[50:1000, 20:50] = np.nan
        # 将第 2000 到 3000 行的所有数据设置为 NaN
        self.df.iloc[2000:3000] = np.nan
        # 将第 60 到 70 列的所有数据设置为 NaN
        self.df.iloc[:, 60:70] = np.nan
        # 复制 DataFrame 对象到 df_mixed
        self.df_mixed = self.df.copy()
        # 在 df_mixed 中新增一个名为 "foo" 的列，所有行的值为 "bar"
        self.df_mixed["foo"] = "bar"

    # 测试 count 方法的执行时间，指定 axis 参数
    def time_count(self, axis):
        # 调用 DataFrame 的 count 方法
        self.df.count(axis=axis)

    # 测试 count 方法的执行时间，对包含不同数据类型的 DataFrame 调用
    def time_count_mixed_dtypes(self, axis):
        # 调用 DataFrame 的 count 方法
        self.df_mixed.count(axis=axis)


class Apply:
    # 设置方法，用于初始化测试数据
    def setup(self):
        # 生成一个 1000x100 的随机数据数组，创建 DataFrame 对象
        self.df = DataFrame(np.random.randn(1000, 100))

        # 生成一个包含 1028 个浮点数的 Series 对象
        self.s = Series(np.arange(1028.0))
        # 创建一个包含 1028 列，每列的数据都是上面的 Series 对象的 DataFrame 对象
        self.df2 = DataFrame({i: self.s for i in range(1028)})
        # 生成一个 1000x3 的随机数据数组，指定列名为 ["A", "B", "C"]，创建 DataFrame 对象
        self.df3 = DataFrame(np.random.randn(1000, 3), columns=list("ABC"))

    # 测试 apply 方法的执行时间，传入一个 lambda 函数
    def time_apply_user_func(self):
        # 对 df2 调用 apply 方法，传入 lambda 函数计算相关系数
        self.df2.apply(lambda x: np.corrcoef(x, self.s)[(0, 1)])

    # 测试 apply 方法的执行时间，指定 axis=1
    def time_apply_axis_1(self):
        # 对 df 调用 apply 方法，传入 lambda 函数，每行加 1
        self.df.apply(lambda x: x + 1, axis=1)

    # 测试 apply 方法的执行时间，传入 lambda 函数计算平均值
    def time_apply_lambda_mean(self):
        # 对 df 调用 apply 方法，传入 lambda 函数计算每列的平均值
        self.df.apply(lambda x: x.mean())

    # 测试 apply 方法的执行时间，传入字符串 "mean"
    def time_apply_str_mean(self):
        # 对 df 调用 apply 方法，传入字符串 "mean"，计算每列的平均值
        self.df.apply("mean")

    # 测试 apply 方法的执行时间，传入 lambda 函数直接返回输入
    def time_apply_pass_thru(self):
        # 对 df 调用 apply 方法，传入 lambda 函数直接返回输入
        self.df.apply(lambda x: x)

    # 测试 apply 方法的执行时间，传入 lambda 函数引用列名计算
    def time_apply_ref_by_name(self):
        # 对 df3 调用 apply 方法，传入 lambda 函数引用列名 "A" 和 "B"，计算它们的和
        self.df3.apply(lambda x: x["A"] + x["B"], axis=1)


class Dtypes:
    # 设置方法，用于初始化测试数据
    def setup(self):
        # 生成一个 1000x1000 的随机数据数组，创建 DataFrame 对象
        self.df = DataFrame(np.random.randn(1000, 1000))

    # 测试 dtypes 方法的执行时间
    def time_frame_dtypes(self):
        # 调用 DataFrame 的 dtypes 方法
        self.df.dtypes


class Equals:
    # 设置方法，用于初始化测试数据
    def setup(self):
        N = 10**3
        # 生成一个大小为 N*N 的随机浮点数数组，创建 DataFrame 对象
        self.float_df = DataFrame(np.random.randn(N, N))
        # 复制 float_df 对象到 float_df_nan
        self.float_df_nan = self.float_df.copy()
        # 将 float_df_nan 的最后一个元素设置为 NaN
        self.float_df_nan.iloc[-1, -1] = np.nan

        # 生成一个大小为 N*N 的字符串 "foo" 组成的 DataFrame 对象
        self.object_df = DataFrame("foo", index=range(N), columns=range(N))
        # 复制 object_df 对象到 object_df_nan
        self.object_df_nan = self.object_df.copy()
        # 将 object_df_nan 的最后一个元素设置为 NaN
        self.object_df_nan.iloc[-1, -1] = np.nan

        # 复制 object_df 对象到 nonunique_cols
        self.nonunique_cols = self.object_df.copy()
        # 将 nonunique_cols 的所有列名设置为 "A"
        self.nonunique_cols.columns = ["A"] * len(self.nonunique_cols.columns)
        # 复制 nonunique_cols 对象到 nonunique_cols_nan
        self.nonunique_cols_nan = self.nonunique_cols.copy()
        # 将 nonunique_cols_nan 的最后一个元素设置为 NaN
        self.nonunique_cols_nan.iloc[-1, -1] = np.nan

    # 测试 equals 方法的执行时间，比较两个相同的 float 类型的 DataFrame 对象
    def time_frame_float_equal(self):
        # 调用 DataFrame 的 equals 方法比较两个 float 类型的 DataFrame 对象
        self.float_df.equals(self.float_df)

    # 测试 equals 方法的执行时间，比较一个包含 NaN 的 float 类型的 DataFrame 对象与原对象
    def time_frame_float_unequal(self):
        # 调用 DataFrame 的 equals 方法比较一个包含 NaN 的 float 类型的 DataFrame 对象与原对象
        self.float_df.equals(self.float_df_nan)

    # 测试 equals 方法的执行时间，比较两个相同的非唯一列名的 DataFrame 对象
    def time_frame_nonunique_equal(self):
        # 调用 DataFrame 的 equals 方法比较两个相同的非唯一列名的 DataFrame 对象
        self.nonunique_cols.equals(self.nonunique_cols)

    # 测试 equals 方法的执行时间，比较一个包含 NaN 的非唯一列名的 DataFrame 对象与原对象
    def time_frame_nonunique_unequal(self):
        # 调用 DataFrame 的 equals 方法比较一个包含 NaN 的非唯一列名的 DataFrame 对象与原对象
        self.nonunique_cols.equals(self.nonunique_cols_nan)

    # 测试 equals 方法的执行时间，比较两个相同的 object 类型的 DataFrame 对象
    def time_frame_object_equal(self):
        # 调用 DataFrame 的 equals 方法比较两个相同的 object 类型
    # 定义一个方法 `time_frame_object_unequal`，它属于当前类的实例方法
    def time_frame_object_unequal(self):
        # 调用 `equals` 方法比较 `object_df` 和 `object_df_nan` 是否相等，但没有返回或保存结果
        self.object_df.equals(self.object_df_nan)
class Interpolate:
    # 插值类
    def setup(self):
        # 设定数组大小为10000
        N = 10000
        # 创建一个形状为(N, 100)的随机数组，将每第二行设为NaN
        arr = np.random.randn(N, 100)
        arr[::2] = np.nan

        # 使用DataFrame封装数组arr
        self.df = DataFrame(arr)

        # 创建另一个DataFrame对象self.df2
        self.df2 = DataFrame(
            {
                "A": np.arange(0, N),  # 列'A'为0到N-1的整数
                "B": np.random.randint(0, 100, N),  # 列'B'为0到99的随机整数
                "C": np.random.randn(N),  # 列'C'为标准正态分布随机数
                "D": np.random.randn(N),  # 列'D'为标准正态分布随机数
            }
        )
        # 每隔5行，将self.df2的'A'和'C'列的值设为NaN
        self.df2.loc[1::5, "A"] = np.nan
        self.df2.loc[1::5, "C"] = np.nan

    # 时间插值方法
    def time_interpolate(self):
        self.df.interpolate()

    # 时间插值（部分列）方法
    def time_interpolate_some_good(self):
        self.df2.interpolate()


class Shift:
    # 框架移位速度问题-5609
    params = [0, 1]  # 参数为0和1
    param_names = ["axis"]  # 参数名称为'axis'

    # 设置方法，接受axis参数
    def setup(self, axis):
        # 创建一个形状为(10000, 500)的随机数组成的DataFrame对象self.df
        self.df = DataFrame(np.random.rand(10000, 500))

    # 时间移位方法，接受axis参数
    def time_shift(self, axis):
        self.df.shift(1, axis=axis)


class Nunique:
    # 唯一值计数类
    def setup(self):
        # 创建一个形状为(10000, 1000)的随机数组成的DataFrame对象self.df
        self.df = DataFrame(np.random.randn(10000, 1000))

    # 时间帧唯一值计数方法
    def time_frame_nunique(self):
        self.df.nunique()


class SeriesNuniqueWithNan:
    # 具有NaN的序列唯一值计数类
    def setup(self):
        # 创建一个包含大量NaN值和一些数字的Series对象self.ser
        values = 100 * [np.nan] + list(range(100))
        self.ser = Series(np.tile(values, 10000), dtype=float)

    # 时间序列唯一值计数（包括NaN）方法
    def time_series_nunique_nan(self):
        self.ser.nunique()


class Duplicated:
    # 重复值类
    def setup(self):
        # 创建一个包含大量数据的DataFrame对象self.df
        n = 1 << 20
        t = date_range("2015-01-01", freq="s", periods=(n // 64))
        xs = np.random.randn(n // 64).round(2)
        self.df = DataFrame(
            {
                "a": np.random.randint(-1 << 8, 1 << 8, n),
                "b": np.random.choice(t, n),
                "c": np.random.choice(xs, n),
            }
        )
        # 创建一个形状为(1000, 100)的随机数组成的DataFrame对象self.df2，并转置
        self.df2 = DataFrame(np.random.randn(1000, 100).astype(str)).T

    # 时间帧重复值方法
    def time_frame_duplicated(self):
        self.df.duplicated()

    # 时间帧重复值（宽格式）方法
    def time_frame_duplicated_wide(self):
        self.df2.duplicated()

    # 时间帧重复值（部分列）方法
    def time_frame_duplicated_subset(self):
        self.df.duplicated(subset=["a"])


class XS:
    params = [0, 1]  # 参数为0和1
    param_names = ["axis"]  # 参数名称为'axis'

    # 设置方法，接受axis参数
    def setup(self, axis):
        self.N = 10**4  # 设置常量N为10000
        # 创建一个形状为(N, N)的随机数组成的DataFrame对象self.df
        self.df = DataFrame(np.random.randn(self.N, self.N))

    # 时间帧选取方法，接受axis参数
    def time_frame_xs(self, axis):
        self.df.xs(self.N / 2, axis=axis)


class SortValues:
    params = [True, False]  # 参数为True和False
    param_names = ["ascending"]  # 参数名称为'ascending'

    # 设置方法，接受ascending参数
    def setup(self, ascending):
        # 创建一个形状为(1000000, 2)的随机数组成的DataFrame对象self.df
        self.df = DataFrame(np.random.randn(1000000, 2), columns=list("AB"))

    # 时间帧排序方法，接受ascending参数
    def time_frame_sort_values(self, ascending):
        self.df.sort_values(by="A", ascending=ascending)


class SortMultiKey:
    params = [True, False]  # 参数为True和False
    param_names = ["monotonic"]  # 参数名称为'monotonic'
    # 设置函数，用于初始化数据框架
    def setup(self, monotonic):
        # 定义数据框的行数和重复次数
        N = 10000
        K = 10
        # 创建一个数据框，包含三列：key1、key2 和 value
        df = DataFrame(
            {
                # key1 列包含 N 个以 "i-" 开头的字符串，重复 K 次
                "key1": Index([f"i-{i}" for i in range(N)], dtype=object).values.repeat(
                    K
                ),
                # key2 列同样包含 N 个以 "i-" 开头的字符串，重复 K 次
                "key2": Index([f"i-{i}" for i in range(N)], dtype=object).values.repeat(
                    K
                ),
                # value 列包含 N*K 个随机数
                "value": np.random.randn(N * K),
            }
        )
        # 如果需要按照 key1 和 key2 列排序，则进行排序
        if monotonic:
            df = df.sort_values(["key1", "key2"])
        # 将按列排序后的数据框赋给实例变量 df_by_columns
        self.df_by_columns = df
        # 将按照 key1 和 key2 列设置索引后的数据框赋给实例变量 df_by_index
        self.df_by_index = df.set_index(["key1", "key2"])

    # 时间测试函数，用于测试按列排序的性能
    def time_sort_values(self, monotonic):
        # 对按列排序后的数据框再次进行排序，以测试性能
        self.df_by_columns.sort_values(by=["key1", "key2"])

    # 时间测试函数，用于测试按索引排序的性能
    def time_sort_index(self, monotonic):
        # 对按索引排序后的数据框进行排序，以测试性能
        self.df_by_index.sort_index()
class Quantile:
    # 默认参数值为 [0, 1]
    params = [0, 1]
    # 参数名列表为 ["axis"]
    param_names = ["axis"]

    # 初始化方法，根据给定的轴生成一个随机数据框
    def setup(self, axis):
        self.df = DataFrame(np.random.randn(1000, 3), columns=list("ABC"))

    # 计算指定轴上的分位数
    def time_frame_quantile(self, axis):
        self.df.quantile([0.1, 0.5], axis=axis)


class Rank:
    # 参数名列表为 ["dtype"]
    param_names = ["dtype"]
    # 参数选项包括不同的数据类型字符串
    params = [
        ["int", "uint", "float", "object"],
    ]

    # 初始化方法，生成一个具有指定数据类型的随机数据框
    def setup(self, dtype):
        self.df = DataFrame(
            np.random.randn(10000, 10).astype(dtype), columns=range(10), dtype=dtype
        )

    # 对数据框中的数据进行排名操作
    def time_rank(self, dtype):
        self.df.rank()


class GetDtypeCounts:
    # 初始化方法，生成一个随机的数据框
    def setup(self):
        self.df = DataFrame(np.random.randn(10, 10000))

    # 计算数据框中每种数据类型的数量
    def time_frame_get_dtype_counts(self):
        with warnings.catch_warnings(record=True):
            self.df.dtypes.value_counts()

    # 输出数据框的基本信息
    def time_info(self):
        self.df.info()


class NSort:
    # 参数选项为 ["first", "last", "all"]
    params = ["first", "last", "all"]
    # 参数名列表为 ["keep"]
    param_names = ["keep"]

    # 初始化方法，生成一个具有随机数据的大型数据框
    def setup(self, keep):
        self.df = DataFrame(np.random.randn(100000, 3), columns=list("ABC"))

    # 计算指定列中最大的若干个值
    def time_nlargest_one_column(self, keep):
        self.df.nlargest(100, "A", keep=keep)

    # 计算指定两列中最大的若干个值
    def time_nlargest_two_columns(self, keep):
        self.df.nlargest(100, ["A", "B"], keep=keep)

    # 计算指定列中最小的若干个值
    def time_nsmallest_one_column(self, keep):
        self.df.nsmallest(100, "A", keep=keep)

    # 计算指定两列中最小的若干个值
    def time_nsmallest_two_columns(self, keep):
        self.df.nsmallest(100, ["A", "B"], keep=keep)


class Describe:
    # 初始化方法，生成一个包含大量随机数据的数据框
    def setup(self):
        self.df = DataFrame(
            {
                "a": np.random.randint(0, 100, 10**6),
                "b": np.random.randint(0, 100, 10**6),
                "c": np.random.randint(0, 100, 10**6),
            }
        )

    # 对数据框中的单一列进行描述性统计
    def time_series_describe(self):
        self.df["a"].describe()

    # 对整个数据框进行描述性统计
    def time_dataframe_describe(self):
        self.df.describe()


class MemoryUsage:
    # 初始化方法，生成两个包含大量数据的数据框，其中一个将列转换为对象类型
    def setup(self):
        self.df = DataFrame(np.random.randn(100000, 2), columns=list("AB"))
        self.df2 = self.df.copy()
        self.df2["A"] = self.df2["A"].astype("object")

    # 计算数据框内存使用情况，包括深度复制
    def time_memory_usage(self):
        self.df.memory_usage(deep=True)

    # 计算包含对象数据类型的数据框内存使用情况
    def time_memory_usage_object_dtype(self):
        self.df2.memory_usage(deep=True)


class Round:
    # 初始化方法，生成一个包含大量随机数据的数据框，并创建其转置副本
    def setup(self):
        self.df = DataFrame(np.random.randn(10000, 10))
        self.df_t = self.df.transpose(copy=True)

    # 对数据框中的所有值进行四舍五入操作
    def time_round(self):
        self.df.round()

    # 对转置后数据框中的所有值进行四舍五入操作
    def time_round_transposed(self):
        self.df_t.round()

    # 记录四舍五入操作的峰值内存使用情况
    def peakmem_round(self):
        self.df.round()

    # 记录转置后数据框四舍五入操作的峰值内存使用情况
    def peakmem_round_transposed(self):
        self.df_t.round()


class Where:
    # 参数选项为 [True, False] 和不同的数据类型字符串
    params = (
        [True, False],
        ["float64", "Float64", "float64[pyarrow]"],
    )
    # 参数名列表为 ["dtype"]
    param_names = ["dtype"]

    # 初始化方法，生成一个包含大量随机数据的数据框，并创建一个布尔掩码
    def setup(self, inplace, dtype):
        self.df = DataFrame(np.random.randn(100_000, 10), dtype=dtype)
        self.mask = self.df < 0

    # 使用条件表达式替换数据框中的值
    def time_where(self, inplace, dtype):
        self.df.where(self.mask, other=0.0, inplace=inplace)
class FindValidIndex:
    # 参数名称列表
    param_names = ["dtype"]
    # 参数值列表
    params = [
        ["float", "Float64", "float64[pyarrow]"],
    ]

    # 准备方法，初始化数据帧 df
    def setup(self, dtype):
        # 创建具有随机数据的 DataFrame，包括两列 A 和 B
        df = DataFrame(
            np.random.randn(100000, 2),
            columns=list("AB"),
            dtype=dtype,
        )
        # 将前100行第一列设为 None（缺失值）
        df.iloc[:100, 0] = None
        # 将前200行第二列设为 None（缺失值）
        df.iloc[:200, 1] = None
        # 将后100行第一列设为 None（缺失值）
        df.iloc[-100:, 0] = None
        # 将后200行第二列设为 None（缺失值）
        df.iloc[-200:, 1] = None
        # 将创建的 DataFrame 赋给实例变量 self.df
        self.df = df

    # 计算函数执行时间：找到第一个非空值的索引
    def time_first_valid_index(self, dtype):
        self.df.first_valid_index()

    # 计算函数执行时间：找到最后一个非空值的索引
    def time_last_valid_index(self, dtype):
        self.df.last_valid_index()


class Update:
    # 准备方法，初始化数据帧 df、df_random 和 df_sample
    def setup(self):
        # 使用随机数生成器创建包含 1000000 行和 10 列的 DataFrame
        rng = np.random.default_rng()
        self.df = DataFrame(rng.uniform(size=(1_000_000, 10)))

        # 从范围为 0 到 999999 的数字中选择不重复的索引，创建 df_random
        idx = rng.choice(range(1_000_000), size=1_000_000, replace=False)
        self.df_random = DataFrame(self.df, index=idx)

        # 从范围为 0 到 999999 的数字中选择不重复的索引和两列，创建 df_sample
        idx = rng.choice(range(1_000_000), size=100_000, replace=False)
        cols = rng.choice(range(10), size=2, replace=False)
        self.df_sample = DataFrame(
            rng.uniform(size=(100_000, 2)), index=idx, columns=cols
        )

    # 计算函数执行时间：使用小的 DataFrame 更新大的 DataFrame
    def time_to_update_big_frame_small_arg(self):
        self.df.update(self.df_sample)

    # 计算函数执行时间：使用随机索引更新随机索引的 DataFrame
    def time_to_update_random_indices(self):
        self.df_random.update(self.df_sample)

    # 计算函数执行时间：使用大的 DataFrame 更新小的 DataFrame
    def time_to_update_small_frame_big_arg(self):
        self.df_sample.update(self.df)


from .pandas_vb_common import setup  # 导入 pandas_vb_common 模块的 setup 函数，跳过 isort 检查
```