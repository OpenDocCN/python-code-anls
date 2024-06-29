# `D:\src\scipysrc\pandas\asv_bench\benchmarks\join_merge.py`

```
# 导入 string 模块，用于处理字符串相关操作
import string

# 导入 numpy 库，并使用 np 别名
import numpy as np

# 从 pandas 库中导入多个模块：DataFrame, Index, MultiIndex, Series, array, concat, date_range, merge, merge_asof
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    array,
    concat,
    date_range,
    merge,
    merge_asof,
)

# 尝试从 pandas 库中导入 merge_ordered 模块，如果导入失败则使用 ordered_merge 作为别名
try:
    from pandas import merge_ordered
except ImportError:
    from pandas import ordered_merge as merge_ordered


# 定义 Concat 类
class Concat:
    # 参数列表包含一个名为 axis 的参数，可选值为 0 或 1
    params = [0, 1]
    # 参数名为 "axis"
    param_names = ["axis"]

    # 设置方法，根据传入的 axis 参数初始化数据
    def setup(self, axis):
        # N 的值为 1000
        N = 1000
        # 创建一个 Series 对象 s，包含 N 个元素，索引由字符串 "i-0" 到 "i-999"
        s = Series(N, index=Index([f"i-{i}" for i in range(N)], dtype=object))
        # self.series 为一个列表，包含 9 个 Series 对象的切片，每个切片长度逐渐减少，重复 50 次
        self.series = [s[i:-i] for i in range(1, 10)] * 50
        # self.small_frames 为一个列表，包含 1000 个元素，每个元素是一个 5x4 的随机数 DataFrame 对象
        self.small_frames = [DataFrame(np.random.randn(5, 4))] * 1000
        # 创建一个 DataFrame 对象 df，包含一列名为 "A"，值为 0 到 999，索引从 "20130101" 开始，以秒为频率
        df = DataFrame(
            {"A": range(N)}, index=date_range("20130101", periods=N, freq="s")
        )
        # self.empty_left 为一个列表，包含两个元素，分别为一个空的 DataFrame 对象和 df
        self.empty_left = [DataFrame(), df]
        # self.empty_right 为一个列表，包含两个元素，分别为 df 和一个空的 DataFrame 对象
        self.empty_right = [df, DataFrame()]
        # self.mixed_ndims 为一个列表，包含两个元素，分别为 df 和 df 的前半部分构成的 DataFrame 对象
        self.mixed_ndims = [df, df.head(N // 2)]

    # 测试拼接 Series 对象的性能
    def time_concat_series(self, axis):
        concat(self.series, axis=axis, sort=False)

    # 测试拼接小型 DataFrame 对象的性能
    def time_concat_small_frames(self, axis):
        concat(self.small_frames, axis=axis)

    # 测试拼接右边为空的 DataFrame 对象的性能
    def time_concat_empty_right(self, axis):
        concat(self.empty_right, axis=axis)

    # 测试拼接左边为空的 DataFrame 对象的性能
    def time_concat_empty_left(self, axis):
        concat(self.empty_left, axis=axis)

    # 测试拼接维度不同的 DataFrame 对象的性能
    def time_concat_mixed_ndims(self, axis):
        concat(self.mixed_ndims, axis=axis)


# 定义 ConcatDataFrames 类
class ConcatDataFrames:
    # 参数列表包含两个参数：axis 和 ignore_index，分别取值为 0 或 1，True 或 False
    params = ([0, 1], [True, False])
    # 参数名为 "axis" 和 "ignore_index"
    param_names = ["axis", "ignore_index"]

    # 设置方法，根据传入的 axis 和 ignore_index 参数初始化数据
    def setup(self, axis, ignore_index):
        # 创建一个元素全部为 0 的大型 DataFrame 对象 frame_c，形状为 (10000, 200)，数据类型为 np.float32，按 C 风格存储
        frame_c = DataFrame(np.zeros((10000, 200), dtype=np.float32, order="C"))
        # self.frame_c 为一个列表，包含 20 个元素，每个元素为 frame_c 的引用
        self.frame_c = [frame_c] * 20
        # 创建一个元素全部为 0 的大型 DataFrame 对象 frame_f，形状为 (10000, 200)，数据类型为 np.float32，按 Fortran 风格存储
        frame_f = DataFrame(np.zeros((10000, 200), dtype=np.float32, order="F"))
        # self.frame_f 为一个列表，包含 20 个元素，每个元素为 frame_f 的引用
        self.frame_f = [frame_f] * 20

    # 测试拼接按 C 风格存储的 DataFrame 对象的性能
    def time_c_ordered(self, axis, ignore_index):
        concat(self.frame_c, axis=axis, ignore_index=ignore_index)

    # 测试拼接按 Fortran 风格存储的 DataFrame 对象的性能
    def time_f_ordered(self, axis, ignore_index):
        concat(self.frame_f, axis=axis, ignore_index=ignore_index)


# 定义 ConcatIndexDtype 类
class ConcatIndexDtype:
    # 参数列表包含四个参数：dtype, structure, axis, sort
    params = (
        [
            "datetime64[ns]",
            "int64",
            "Int64",
            "int64[pyarrow]",
            "string[python]",
            "string[pyarrow]",
        ],
        ["monotonic", "non_monotonic", "has_na"],
        [0, 1],
        [True, False],
    )
    # 参数名为 "dtype", "structure", "axis", "sort"
    # 设置函数，用于初始化数据结构和类型
    def setup(self, dtype, structure, axis, sort):
        # 定义数据元素个数
        N = 10_000
        # 根据数据类型选择合适的数值范围或日期范围
        if dtype == "datetime64[ns]":
            vals = date_range("1970-01-01", periods=N)
        elif dtype in ("int64", "Int64", "int64[pyarrow]"):
            vals = np.arange(N, dtype=np.int64)
        elif dtype in ("string[python]", "string[pyarrow]"):
            # 生成以'i-'开头的字符串列表
            vals = Index([f"i-{i}" for i in range(N)], dtype=object)
        else:
            # 抛出未实现的错误，如果数据类型不在预期范围内
            raise NotImplementedError

        # 根据生成的值创建索引对象
        idx = Index(vals, dtype=dtype)

        # 根据结构参数对索引进行不同的排序或处理
        if structure == "monotonic":
            # 如果结构为单调递增，则对索引进行排序
            idx = idx.sort_values()
        elif structure == "non_monotonic":
            # 如果结构为非单调，则逆序排列索引
            idx = idx[::-1]
        elif structure == "has_na":
            # 如果结构包含缺失值，检查是否可以容纳缺失值
            if not idx._can_hold_na:
                raise NotImplementedError
            # 在索引前添加一个包含None的索引
            idx = Index([None], dtype=dtype).append(idx)
        else:
            # 抛出未实现的错误，如果结构参数不在预期范围内
            raise NotImplementedError

        # 使用创建的索引对象，初始化self.series列表
        self.series = [Series(i, idx[:-i]) for i in range(1, 6)]

    # 时间串联系列数据的函数
    def time_concat_series(self, dtype, structure, axis, sort):
        # 调用concat函数将self.series列表中的Series对象串联起来
        concat(self.series, axis=axis, sort=sort)
class Join:
    # 设定参数列表，包含排序选项
    params = [True, False]
    # 参数名列表，仅包含排序
    param_names = ["sort"]

    # 设置函数，初始化各种数据结构和随机数据
    def setup(self, sort):
        # 创建第一级和第二级索引数组
        level1 = Index([f"i-{i}" for i in range(10)], dtype=object).values
        level2 = Index([f"i-{i}" for i in range(1000)], dtype=object).values
        # 重复和铺设索引代码以创建 MultiIndex 对象
        codes1 = np.arange(10).repeat(1000)
        codes2 = np.tile(np.arange(1000), 10)
        index2 = MultiIndex(levels=[level1, level2], codes=[codes1, codes2])
        # 创建包含随机数据的 DataFrame，使用 MultiIndex 作为索引
        self.df_multi = DataFrame(
            np.random.randn(len(index2), 4), index=index2, columns=["A", "B", "C", "D"]
        )

        # 创建 key1 和 key2 数组，用于单级索引的测试
        self.key1 = np.tile(level1.take(codes1), 10)
        self.key2 = np.tile(level2.take(codes2), 10)
        # 创建包含随机数据的 DataFrame，包括 data1, data2, key1, key2 列
        self.df = DataFrame(
            {
                "data1": np.random.randn(100000),
                "data2": np.random.randn(100000),
                "key1": self.key1,
                "key2": self.key2,
            }
        )

        # 创建包含随机数据的 DataFrame，使用 level1 作为索引
        self.df_key1 = DataFrame(
            np.random.randn(len(level1), 4), index=level1, columns=["A", "B", "C", "D"]
        )
        # 创建包含随机数据的 DataFrame，使用 level2 作为索引
        self.df_key2 = DataFrame(
            np.random.randn(len(level2), 4), index=level2, columns=["A", "B", "C", "D"]
        )

        # 随机重排 self.df，并存储在 self.df_shuf 中
        shuf = np.arange(100000)
        np.random.shuffle(shuf)
        self.df_shuf = self.df.reindex(self.df.index[shuf])

    # 测试函数：测试 DataFrame 和 MultiIndex 的连接
    def time_join_dataframe_index_multi(self, sort):
        self.df.join(self.df_multi, on=["key1", "key2"], sort=sort)

    # 测试函数：测试 DataFrame 和单级索引的连接（右边 DataFrame 大）
    def time_join_dataframe_index_single_key_bigger(self, sort):
        self.df.join(self.df_key2, on="key2", sort=sort)

    # 测试函数：测试 DataFrame 和单级索引的连接（左边 DataFrame 大）
    def time_join_dataframe_index_single_key_small(self, sort):
        self.df.join(self.df_key1, on="key1", sort=sort)

    # 测试函数：测试 DataFrame 和单级索引的连接（使用随机重排的 DataFrame）
    def time_join_dataframe_index_shuffle_key_bigger_sort(self, sort):
        self.df_shuf.join(self.df_key2, on="key2", sort=sort)

    # 测试函数：测试 DataFrame 的交叉连接
    def time_join_dataframes_cross(self, sort):
        self.df.loc[:2000].join(self.df_key1, how="cross", sort=sort)


class JoinIndex:
    # 设置函数：初始化左右两个包含随机整数的 DataFrame
    def setup(self):
        N = 5000
        self.left = DataFrame(
            np.random.randint(1, N / 50, (N, 2)), columns=["jim", "joe"]
        )
        self.right = DataFrame(
            np.random.randint(1, N / 50, (N, 2)), columns=["jolie", "jolia"]
        ).set_index("jolie")

    # 测试函数：测试左外连接（使用索引列）
    def time_left_outer_join_index(self):
        self.left.join(self.right, on="jim")


class JoinMultiindexSubset:
    # 设置函数：初始化包含 MultiIndex 的左右两个 DataFrame
    def setup(self):
        N = 100_000
        mi1 = MultiIndex.from_arrays([np.arange(N)] * 4, names=["a", "b", "c", "d"])
        mi2 = MultiIndex.from_arrays([np.arange(N)] * 2, names=["a", "b"])
        self.left = DataFrame({"col1": 1}, index=mi1)
        self.right = DataFrame({"col2": 2}, index=mi2)

    # 测试函数：测试 MultiIndex 的子集连接
    def time_join_multiindex_subset(self):
        self.left.join(self.right)


class JoinEmpty:
    # 设置函数：初始化包含整数列的 DataFrame 和一个空 DataFrame
    def setup(self):
        N = 100_000
        self.df = DataFrame({"A": np.arange(N)})
        self.df_empty = DataFrame(columns=["B", "C"], dtype="int64")

    # 测试函数：测试空 DataFrame 和非空 DataFrame 的内连接
    def time_inner_join_left_empty(self):
        self.df_empty.join(self.df, how="inner")
    # 定义一个方法，用于执行右连接的内连接操作，其中右侧DataFrame为空
    def time_inner_join_right_empty(self):
        # 调用当前对象的DataFrame的join方法，指定连接方式为内连接（保留匹配的行）
        # 这里的self.df代表当前对象的DataFrame，self.df_empty代表另一个空的DataFrame
        # 返回的结果是一个新的DataFrame，表示两个DataFrame的内连接结果
        self.df.join(self.df_empty, how="inner")
class JoinNonUnique:
    # 外连接非唯一
    # GH 6329
    def setup(self):
        # 创建一个日期范围，从 "01-Jan-2013" 到 "23-Jan-2013"，频率为每分钟
        date_index = date_range("01-Jan-2013", "23-Jan-2013", freq="min")
        # 将日期索引转换为每日周期，并转换为时间戳秒和秒
        daily_dates = date_index.to_period("D").to_timestamp("s", "s")
        # 计算每个时间戳与所在日的时间差
        self.fracofday = date_index.values - daily_dates.values
        # 将时间差转换为纳秒级的时间增量
        self.fracofday = self.fracofday.astype("timedelta64[ns]")
        # 将时间增量转换为秒，并将结果转换为浮点数
        self.fracofday = self.fracofday.astype(np.float64) / 86_400_000_000_000
        # 创建时间序列，以每日日期为索引，时间差为数据
        self.fracofday = Series(self.fracofday, daily_dates)
        # 根据日期范围创建索引，并创建值为 1.0 的系列，使用时间差的索引进行筛选
        index = date_range(date_index.min(), date_index.max(), freq="D")
        self.temp = Series(1.0, index)[self.fracofday.index]

    def time_join_non_unique_equal(self):
        # 计算时间差和临时系列的乘积，但未使用结果
        self.fracofday * self.temp


class Merge:
    params = [True, False]
    param_names = ["sort"]

    def setup(self, sort):
        N = 10000
        # 创建包含字符串索引的数组，用于左侧和右侧的数据帧
        indices = Index([f"i-{i}" for i in range(N)], dtype=object).values
        indices2 = Index([f"i-{i}" for i in range(N)], dtype=object).values
        # 在左右两侧创建重复的索引数组
        key = np.tile(indices[:8000], 10)
        key2 = np.tile(indices2[:8000], 10)
        # 创建左侧和右侧的数据帧，包含键和随机值
        self.left = DataFrame(
            {"key": key, "key2": key2, "value": np.random.randn(80000)}
        )
        self.right = DataFrame(
            {
                "key": indices[2000:],
                "key2": indices2[2000:],
                "value2": np.random.randn(8000),
            }
        )

        # 创建带有两个整数键的数据帧，以及带有整数键和随机值的数据帧
        self.df = DataFrame(
            {
                "key1": np.tile(np.arange(500).repeat(10), 2),
                "key2": np.tile(np.arange(250).repeat(10), 4),
                "value": np.random.randn(10000),
            }
        )
        self.df2 = DataFrame({"key1": np.arange(500), "value2": np.random.randn(500)})
        # 创建数据帧的切片，包含前 5000 行
        self.df3 = self.df[:5000]

    def time_merge_2intkey(self, sort):
        # 执行合并操作，使用左侧和右侧数据帧，根据排序参数确定是否排序
        merge(self.left, self.right, sort=sort)

    def time_merge_dataframe_integer_2key(self, sort):
        # 执行合并操作，使用整数键的数据帧和切片的数据帧，根据排序参数确定是否排序
        merge(self.df, self.df3, sort=sort)

    def time_merge_dataframe_integer_key(self, sort):
        # 执行合并操作，使用整数键的数据帧和带有整数键和随机值的数据帧，根据排序参数确定是否排序
        merge(self.df, self.df2, on="key1", sort=sort)

    def time_merge_dataframe_empty_right(self, sort):
        # 执行合并操作，使用左侧和右侧数据帧的切片，右侧为空，根据排序参数确定是否排序
        merge(self.left, self.right.iloc[:0], sort=sort)

    def time_merge_dataframe_empty_left(self, sort):
        # 执行合并操作，使用左侧数据帧的切片和右侧数据帧，左侧为空，根据排序参数确定是否排序
        merge(self.left.iloc[:0], self.right, sort=sort)

    def time_merge_dataframes_cross(self, sort):
        # 执行交叉合并操作，使用左侧和右侧数据帧的切片，根据排序参数确定是否排序
        merge(self.left.loc[:2000], self.right.loc[:2000], how="cross", sort=sort)


class MergeEA:
    params = [
        [
            "Int64",
            "Int32",
            "Int16",
            "UInt64",
            "UInt32",
            "UInt16",
            "Float64",
            "Float32",
        ],
        [True, False],
    ]
    param_names = ["dtype", "monotonic"]
    # 设置函数，用于初始化数据结构
    def setup(self, dtype, monotonic):
        # 设定常量 N 为 10000
        N = 10_000
        # 创建从 1 到 N-1 的整数索引数组
        indices = np.arange(1, N)
        # 将索引数组的前 8000 个元素重复 10 次，构成关键字数组
        key = np.tile(indices[:8000], 10)
        # 创建名为 left 的 DataFrame 对象，包含两列：key 和 value
        self.left = DataFrame(
            {"key": Series(key, dtype=dtype), "value": np.random.randn(80000)}
        )
        # 创建名为 right 的 DataFrame 对象，包含两列：key 和 value2
        self.right = DataFrame(
            {
                "key": Series(indices[2000:], dtype=dtype),
                "value2": np.random.randn(7999),
            }
        )
        # 如果 monotonic 参数为真，对 left 和 right 的数据按 key 列进行排序
        if monotonic:
            self.left = self.left.sort_values("key")
            self.right = self.right.sort_values("key")

    # 时间合并函数，将 left 和 right 的数据进行合并
    def time_merge(self, dtype, monotonic):
        merge(self.left, self.right)
class I8Merge:
    # 可选的合并方式参数列表
    params = ["inner", "outer", "left", "right"]
    # 参数名称列表，只包含一个参数名
    param_names = ["how"]

    # 设置方法，用于准备数据
    def setup(self, how):
        # 定义随机数范围和生成数据数量
        low, high, n = -1000, 1000, 10**6
        # 创建左侧数据帧，包含随机整数列和汇总列
        self.left = DataFrame(
            np.random.randint(low, high, (n, 7)), columns=list("ABCDEFG")
        )
        self.left["left"] = self.left.sum(axis=1)
        # 从左侧数据帧随机抽样并重命名为右侧数据帧
        self.right = self.left.sample(frac=1).rename({"left": "right"}, axis=1)
        self.right = self.right.reset_index(drop=True)
        # 将右侧数据帧的"right"列乘以-1
        self.right["right"] *= -1

    # 测试函数，执行合并操作
    def time_i8merge(self, how):
        merge(self.left, self.right, how=how)


class UniqueMerge:
    # 唯一元素数量参数列表
    params = [4_000_000, 1_000_000]
    # 参数名称列表，只包含一个参数名
    param_names = ["unique_elements"]

    # 设置方法，用于准备数据
    def setup(self, unique_elements):
        N = 1_000_000
        # 创建具有随机整数列的左右数据帧
        self.left = DataFrame({"a": np.random.randint(1, unique_elements, (N,))})
        self.right = DataFrame({"a": np.random.randint(1, unique_elements, (N,))})
        # 从右侧数据帧的"a"列中去除重复值
        uniques = self.right.a.drop_duplicates()
        # 将右侧数据帧的"a"列与递减序列连接起来
        self.right["a"] = concat(
            [uniques, Series(np.arange(0, -(N - len(uniques)), -1))], ignore_index=True
        )

    # 测试函数，执行合并操作
    def time_unique_merge(self, unique_elements):
        merge(self.left, self.right, how="inner")


class MergeDatetime:
    # 单元和时区参数列表
    params = [
        [
            ("ns", "ns"),
            ("ms", "ms"),
            ("ns", "ms"),
        ],
        [None, "Europe/Brussels"],
        [True, False],
    ]
    # 参数名称列表，包括单元、时区和单调性
    param_names = ["units", "tz", "monotonic"]

    # 设置方法，用于准备数据
    def setup(self, units, tz, monotonic):
        unit_left, unit_right = units
        N = 10_000
        # 生成时间序列作为键的左侧数据帧
        keys = Series(date_range("2012-01-01", freq="min", periods=N, tz=tz))
        self.left = DataFrame(
            {
                "key": keys.sample(N * 10, replace=True).dt.as_unit(unit_left),
                "value1": np.random.randn(N * 10),
            }
        )
        # 生成时间序列作为键的右侧数据帧
        self.right = DataFrame(
            {
                "key": keys[:8000].dt.as_unit(unit_right),
                "value2": np.random.randn(8000),
            }
        )
        if monotonic:
            # 如果单调性为真，则对左右数据帧按键进行排序
            self.left = self.left.sort_values("key")
            self.right = self.right.sort_values("key")

    # 测试函数，执行合并操作
    def time_merge(self, units, tz, monotonic):
        merge(self.left, self.right)


class MergeCategoricals:
    # 待实现的类，尚未提供代码内容
    # 设置测试环境
    def setup(self):
        # 创建左侧数据框对象，包含随机生成的 X 和 Y 列数据
        self.left_object = DataFrame(
            {
                "X": np.random.choice(range(10), size=(10000,)),
                "Y": np.random.choice(["one", "two", "three"], size=(10000,)),
            }
        )

        # 创建右侧数据框对象，包含随机生成的 X 和 Z 列数据
        self.right_object = DataFrame(
            {
                "X": np.random.choice(range(10), size=(10000,)),
                "Z": np.random.choice(["jjj", "kkk", "sss"], size=(10000,)),
            }
        )

        # 对左侧数据框的 Y 列进行类型转换为分类类型
        self.left_cat = self.left_object.assign(
            Y=self.left_object["Y"].astype("category")
        )
        
        # 对右侧数据框的 Z 列进行类型转换为分类类型
        self.right_cat = self.right_object.assign(
            Z=self.right_object["Z"].astype("category")
        )

        # 对左侧数据框的 X 列进行类型转换为分类类型
        self.left_cat_col = self.left_object.astype({"X": "category"})
        
        # 对右侧数据框的 X 列进行类型转换为分类类型
        self.right_cat_col = self.right_object.astype({"X": "category"})

        # 将左侧数据框按 X 列设置为索引
        self.left_cat_idx = self.left_cat_col.set_index("X")
        
        # 将右侧数据框按 X 列设置为索引
        self.right_cat_idx = self.right_cat_col.set_index("X")

    # 测试合并非分类数据框对象
    def time_merge_object(self):
        merge(self.left_object, self.right_object, on="X")

    # 测试合并分类数据框对象
    def time_merge_cat(self):
        merge(self.left_cat, self.right_cat, on="X")

    # 测试合并分类数据框对象，其中 X 列为分类类型
    def time_merge_on_cat_col(self):
        merge(self.left_cat_col, self.right_cat_col, on="X")

    # 测试合并分类数据框对象，其中 X 列为索引
    def time_merge_on_cat_idx(self):
        merge(self.left_cat_idx, self.right_cat_idx, on="X")
class MergeOrdered:
    # 设置测试数据的方法
    def setup(self):
        # 创建一个包含十个分组的索引
        groups = Index([f"i-{i}" for i in range(10)], dtype=object).values
        # 创建左侧数据帧，包括"group"列、"key"列和"lvalue"列
        self.left = DataFrame(
            {
                "group": groups.repeat(5000),  # 每个分组重复5000次
                "key": np.tile(np.arange(0, 10000, 2), 10),  # 从0到9998的偶数，每个分组重复10次
                "lvalue": np.random.randn(50000),  # 包含50000个随机数的数组
            }
        )
        # 创建右侧数据帧，包括"key"列和"rvalue"列
        self.right = DataFrame(
            {"key": np.arange(10000),  # 包含0到9999的整数的数组
             "rvalue": np.random.randn(10000)}  # 包含10000个随机数的数组
        )

    # 执行merge_ordered函数的方法
    def time_merge_ordered(self):
        merge_ordered(self.left, self.right, on="key", left_by="group")


class MergeAsof:
    # 定义参数列表
    params = [["backward", "forward", "nearest"], [None, 5]]
    # 定义参数名
    param_names = ["direction", "tolerance"]

    # 设置测试数据的方法，根据输入的方向和容差生成不同的数据框
    def setup(self, direction, tolerance):
        # 设置两个数据框的初始行数
        one_count = 200000
        two_count = 1000000

        # 创建第一个数据框df1，包括"time"、"key"、"key2"和"value1"列
        df1 = DataFrame(
            {
                "time": np.random.randint(0, one_count / 20, one_count),  # 0到9999之间的随机整数数组
                "key": np.random.choice(list(string.ascii_uppercase), one_count),  # 从大写字母中随机选择的数组
                "key2": np.random.randint(0, 25, one_count),  # 0到24之间的随机整数数组
                "value1": np.random.randn(one_count),  # 包含200000个随机数的数组
            }
        )
        # 创建第二个数据框df2，包括"time"、"key"、"key2"和"value2"列
        df2 = DataFrame(
            {
                "time": np.random.randint(0, two_count / 20, two_count),  # 0到49999之间的随机整数数组
                "key": np.random.choice(list(string.ascii_uppercase), two_count),  # 从大写字母中随机选择的数组
                "key2": np.random.randint(0, 25, two_count),  # 0到24之间的随机整数数组
                "value2": np.random.randn(two_count),  # 包含1000000个随机数的数组
            }
        )

        # 按"time"列对df1和df2进行排序
        df1 = df1.sort_values("time")
        df2 = df2.sort_values("time")

        # 将"time"列转换为int32类型，并分别赋值给df1和df2的"time32"列
        df1["time32"] = np.int32(df1.time)
        df2["time32"] = np.int32(df2.time)

        # 将"time"列转换为uint64类型，并分别赋值给df1和df2的"timeu64"列
        df1["timeu64"] = np.uint64(df1.time)
        df2["timeu64"] = np.uint64(df2.time)

        # 根据不同的类型选择不同的列子集作为测试数据，分别赋值给self的属性
        self.df1a = df1[["time", "value1"]]
        self.df2a = df2[["time", "value2"]]
        self.df1b = df1[["time", "key", "value1"]]
        self.df2b = df2[["time", "key", "value2"]]
        self.df1c = df1[["time", "key2", "value1"]]
        self.df2c = df2[["time", "key2", "value2"]]
        self.df1d = df1[["time32", "value1"]]
        self.df2d = df2[["time32", "value2"]]
        self.df1e = df1[["time", "key", "key2", "value1"]]
        self.df2e = df2[["time", "key", "key2", "value2"]]
        self.df1f = df1[["timeu64", "value1"]]
        self.df2f = df2[["timeu64", "value2"]]

    # 执行merge_asof函数的方法，使用"time"列作为连接键
    def time_on_int(self, direction, tolerance):
        merge_asof(
            self.df1a, self.df2a, on="time", direction=direction, tolerance=tolerance
        )

    # 执行merge_asof函数的方法，使用"time32"列作为连接键
    def time_on_int32(self, direction, tolerance):
        merge_asof(
            self.df1d, self.df2d, on="time32", direction=direction, tolerance=tolerance
        )

    # 执行merge_asof函数的方法，使用"timeu64"列作为连接键
    def time_on_uint64(self, direction, tolerance):
        merge_asof(
            self.df1f, self.df2f, on="timeu64", direction=direction, tolerance=tolerance
        )
    # 根据指定条件使用 merge_asof 函数合并两个DataFrame，按对象方式合并
    def time_by_object(self, direction, tolerance):
        merge_asof(
            self.df1b,          # 第一个DataFrame对象
            self.df2b,          # 第二个DataFrame对象
            on="time",          # 指定连接键为"time"
            by="key",           # 指定合并键为"key"
            direction=direction,    # 合并方向参数
            tolerance=tolerance,    # 容差参数
        )

    # 根据指定条件使用 merge_asof 函数合并两个DataFrame，按整数位置方式合并
    def time_by_int(self, direction, tolerance):
        merge_asof(
            self.df1c,          # 第一个DataFrame对象
            self.df2c,          # 第二个DataFrame对象
            on="time",          # 指定连接键为"time"
            by="key2",          # 指定合并键为"key2"
            direction=direction,    # 合并方向参数
            tolerance=tolerance,    # 容差参数
        )

    # 根据指定条件使用 merge_asof 函数合并两个DataFrame，按多个键方式合并
    def time_multiby(self, direction, tolerance):
        merge_asof(
            self.df1e,          # 第一个DataFrame对象
            self.df2e,          # 第二个DataFrame对象
            on="time",          # 指定连接键为"time"
            by=["key", "key2"],     # 指定多个合并键为["key", "key2"]
            direction=direction,    # 合并方向参数
            tolerance=tolerance,    # 容差参数
        )
class MergeMultiIndex:
    params = [
        [
            ("int64", "int64"),
            ("datetime64[ns]", "int64"),
            ("Int64", "Int64"),
        ],
        ["left", "right", "inner", "outer"],
    ]
    param_names = ["dtypes", "how"]

    def setup(self, dtypes, how):
        # 设置数据规模和偏移量
        n = 100_000
        offset = 50_000
        # 创建第一个多级索引
        mi1 = MultiIndex.from_arrays(
            [
                array(np.arange(n), dtype=dtypes[0]),
                array(np.arange(n), dtype=dtypes[1]),
            ]
        )
        # 创建第二个多级索引，偏移后的值
        mi2 = MultiIndex.from_arrays(
            [
                array(np.arange(offset, n + offset), dtype=dtypes[0]),
                array(np.arange(offset, n + offset), dtype=dtypes[1]),
            ]
        )
        # 创建 DataFrame df1，使用 mi1 作为索引
        self.df1 = DataFrame({"col1": 1}, index=mi1)
        # 创建 DataFrame df2，使用 mi2 作为索引
        self.df2 = DataFrame({"col2": 2}, index=mi2)

    def time_merge_sorted_multiindex(self, dtypes, how):
        # 复制 df1 和 df2 以避免 MultiIndex._values 缓存
        df1 = self.df1.copy()
        df2 = self.df2.copy()
        # 调用 merge 函数，按照指定的参数进行合并
        merge(df1, df2, how=how, left_index=True, right_index=True)


class Align:
    def setup(self):
        # 设置序列的大小
        size = 5 * 10**5
        # 生成一个范围为 [0, 10^13) 的整数数组
        rng = np.arange(0, 10**13, 10**7)
        # 生成时间戳数组，当前时间加上 rng 中的值
        stamps = np.datetime64("now").view("i8") + rng
        # 从 stamps 中随机选择 size 个元素并排序，作为 idx1
        idx1 = np.sort(np.random.choice(stamps, size, replace=False))
        # 从 stamps 中再次随机选择 size 个元素并排序，作为 idx2
        idx2 = np.sort(np.random.choice(stamps, size, replace=False))
        # 创建 Series ts1，使用 idx1 作为索引，数据为随机生成的大小为 size 的数组
        self.ts1 = Series(np.random.randn(size), idx1)
        # 创建 Series ts2，使用 idx2 作为索引，数据为随机生成的大小为 size 的数组
        self.ts2 = Series(np.random.randn(size), idx2)

    def time_series_align_int64_index(self):
        # 执行时间序列的加法操作
        self.ts1 + self.ts2

    def time_series_align_left_monotonic(self):
        # 调用 align 方法，将 ts1 和 ts2 进行左对齐操作
        self.ts1.align(self.ts2, join="left")
```