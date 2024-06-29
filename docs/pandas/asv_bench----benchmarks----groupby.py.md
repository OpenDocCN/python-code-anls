# `D:\src\scipysrc\pandas\asv_bench\benchmarks\groupby.py`

```
# 导入模块函数 partial，用于创建部分函数应用
# 导入 itertools 中的 product 函数，用于生成迭代器的笛卡尔积
# 导入 string 中的 ascii_letters，包含所有大小写字母的字符串
from functools import partial
from itertools import product
from string import ascii_letters

# 导入 numpy 库，并用 np 作为别名
import numpy as np

# 从 pandas 库中导入多个类和函数，包括 NA、Categorical、DataFrame、Index、MultiIndex、Series、Timestamp、date_range、period_range 和 to_timedelta
from pandas import (
    NA,
    Categorical,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
    period_range,
    to_timedelta,
)

# 定义一个字典 method_blocklist，包含不同数据类型下不支持的方法集合
method_blocklist = {
    "object": {
        "diff",
        "median",
        "prod",
        "sem",
        "cumsum",
        "sum",
        "cummin",
        "mean",
        "max",
        "skew",
        "cumprod",
        "cummax",
        "pct_change",
        "min",
        "var",
        "describe",
        "std",
        "quantile",
    },
    "datetime": {
        "median",
        "prod",
        "sem",
        "cumsum",
        "sum",
        "mean",
        "skew",
        "cumprod",
        "cummax",
        "pct_change",
        "var",
        "describe",
        "std",
    },
}

# 定义一个列表 _numba_unsupported_methods，包含尚未实现支持的方法名
_numba_unsupported_methods = [
    "all",
    "any",
    "bfill",
    "count",
    "cumcount",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "describe",
    "diff",
    "ffill",
    "first",
    "head",
    "idxmax",
    "idxmin",
    "last",
    "median",
    "nunique",
    "pct_change",
    "prod",
    "quantile",
    "rank",
    "sem",
    "shift",
    "size",
    "skew",
    "tail",
    "unique",
    "value_counts",
]

# 定义一个类 ApplyDictReturn
class ApplyDictReturn:
    # 设置方法 setup，用于初始化数据
    def setup(self):
        # 创建一个长度为 1000 的序列 self.labels，每个值重复 10 次
        self.labels = np.arange(1000).repeat(10)
        # 创建一个 Series 对象 self.data，包含随机正态分布的数据，长度与 self.labels 相同
        self.data = Series(np.random.randn(len(self.labels)))

    # 定义方法 time_groupby_apply_dict_return，对 self.data 按 self.labels 分组，应用 lambda 函数返回字典
    def time_groupby_apply_dict_return(self):
        self.data.groupby(self.labels).apply(
            lambda x: {"first": x.values[0], "last": x.values[-1]}
        )


# 定义一个类 Apply
class Apply:
    # 参数列表 param_names 包含名为 "factor" 的参数
    param_names = ["factor"]
    # 参数列表 params 包含两个元素 4 和 5
    params = [4, 5]

    # 设置方法 setup，根据不同的 factor 初始化 DataFrame self.df
    def setup(self, factor):
        # 根据 factor 的值计算 N
        N = 10**factor
        # 根据不同的 factor 生成 labels 和 labels2
        labels = np.random.randint(0, 2000 if factor == 4 else 20, size=N)
        labels2 = np.random.randint(0, 3, size=N)
        # 创建 DataFrame 对象 self.df，包含 key、key2、value1 和 value2 列
        self.df = DataFrame(
            {
                "key": labels,
                "key2": labels2,
                "value1": np.random.randn(N),
                "value2": ["foo", "bar", "baz", "qux"] * (N // 4),
            }
        )

    # 定义方法 time_scalar_function_multi_col，对 self.df 按 ["key", "key2"] 分组，应用 lambda 函数返回标量 1
    def time_scalar_function_multi_col(self, factor):
        self.df.groupby(["key", "key2"]).apply(lambda x: 1)

    # 定义方法 time_scalar_function_single_col，对 self.df 按 "key" 分组，应用 lambda 函数返回标量 1
    def time_scalar_function_single_col(self, factor):
        self.df.groupby("key").apply(lambda x: 1)

    # 定义静态方法 df_copy_function，对分组 g 执行 g.name 操作并返回 g 的副本
    @staticmethod
    def df_copy_function(g):
        g.name
        return g.copy()

    # 定义方法 time_copy_function_multi_col，对 self.df 按 ["key", "key2"] 分组，应用 df_copy_function 方法
    def time_copy_function_multi_col(self, factor):
        self.df.groupby(["key", "key2"]).apply(self.df_copy_function)
    # 定义一个方法 `time_copy_overhead_single_col`，接受 `self` 和 `factor` 两个参数
    def time_copy_overhead_single_col(self, factor):
        # 使用 Pandas 中的 `groupby` 方法，按照 "key" 列分组 DataFrame `self.df`
        # 然后对每个分组应用 `self.df_copy_function` 函数
        self.df.groupby("key").apply(self.df_copy_function)
class ApplyNonUniqueUnsortedIndex:
    # 类：应用于非唯一且未排序索引
    def setup(self):
        # GH 46527
        # 创建逆序的 0 到 99 的索引数组
        idx = np.arange(100)[::-1]
        # 通过重复 idx 数组的内容来创建索引对象 Index，命名为 "key"
        idx = Index(np.repeat(idx, 200), name="key")
        # 创建一个 DataFrame，形状为(len(idx), 10)，并填充随机标准正态分布的数据，使用 idx 作为索引
        self.df = DataFrame(np.random.randn(len(idx), 10), index=idx)

    def time_groupby_apply_non_unique_unsorted_index(self):
        # 对 DataFrame self.df 按照 "key" 列进行分组，不使用分组键，然后应用 lambda 函数
        self.df.groupby("key", group_keys=False).apply(lambda x: x)


class Groups:
    param_names = ["key"]
    params = ["int64_small", "int64_large", "object_small", "object_large"]

    def setup_cache(self):
        # 设置缓存数据大小为 10^6
        size = 10**6
        data = {
            "int64_small": Series(np.random.randint(0, 100, size=size)),
            "int64_large": Series(np.random.randint(0, 10000, size=size)),
            "object_small": Series(
                Index([f"i-{i}" for i in range(100)], dtype=object).take(
                    np.random.randint(0, 100, size=size)
                )
            ),
            "object_large": Series(
                Index([f"i-{i}" for i in range(10000)], dtype=object).take(
                    np.random.randint(0, 10000, size=size)
                )
            ),
        }
        return data

    def setup(self, data, key):
        # 设置数据序列为参数 key 所对应的序列
        self.ser = data[key]

    def time_series_groups(self, data, key):
        # 对序列 self.ser 按值进行分组，并返回分组后的组索引
        self.ser.groupby(self.ser).groups

    def time_series_indices(self, data, key):
        # 对序列 self.ser 按值进行分组，并返回分组后的组索引，相当于 time_series_groups 的变体
        self.ser.groupby(self.ser).indices


class GroupManyLabels:
    params = [1, 1000]
    param_names = ["ncols"]

    def setup(self, ncols):
        # 创建形状为 (1000, ncols) 的随机数据
        N = 1000
        data = np.random.randn(N, ncols)
        # 创建包含 1000 个元素的随机标签
        self.labels = np.random.randint(0, 100, size=N)
        # 创建一个 DataFrame，形状为 (1000, ncols)，使用随机数据填充
        self.df = DataFrame(data)

    def time_sum(self, ncols):
        # 对 DataFrame self.df 按照 self.labels 列进行分组，并计算每个分组的和
        self.df.groupby(self.labels).sum()


class Nth:
    param_names = ["dtype"]
    params = ["float32", "float64", "datetime", "object"]

    def setup(self, dtype):
        N = 10**5
        # 如果 dtype 为 "datetime"，则创建一个日期范围从 "1/1/2011" 开始，频率为秒的日期序列
        if dtype == "datetime":
            values = date_range("1/1/2011", periods=N, freq="s")
        # 如果 dtype 为 "object"，则创建一个包含 N 个 "foo" 的字符串列表
        elif dtype == "object":
            values = ["foo"] * N
        # 否则，创建一个从 0 到 N-1 的整数数组，并转换为指定的 dtype 类型
        else:
            values = np.arange(N).astype(dtype)

        # 创建一个 DataFrame，包含两列：'key' 列和 'values' 列，其中 'values' 列包含上面创建的 values 数据
        key = np.arange(N)
        self.df = DataFrame({"key": key, "values": values})
        # 在 DataFrame 中插入缺失数据，将第二行 'values' 列的第一个元素设为 NaN
        self.df.iloc[1, 1] = np.nan  # insert missing data

    def time_frame_nth_any(self, dtype):
        # 对 DataFrame self.df 按照 'key' 列进行分组，并取每个分组的第一个元素，如果存在缺失数据则丢弃
        self.df.groupby("key").nth(0, dropna="any")

    def time_groupby_nth_all(self, dtype):
        # 对 DataFrame self.df 按照 'key' 列进行分组，并取每个分组的第一个元素，只有当所有元素都缺失时才丢弃
        self.df.groupby("key").nth(0, dropna="all")

    def time_frame_nth(self, dtype):
        # 对 DataFrame self.df 按照 'key' 列进行分组，并取每个分组的第一个元素
        self.df.groupby("key").nth(0)

    def time_series_nth_any(self, dtype):
        # 对 Series self.df['values'] 按照 self.df['key'] 列进行分组，并取每个分组的第一个元素，如果存在缺失数据则丢弃
        self.df["values"].groupby(self.df["key"]).nth(0, dropna="any")

    def time_series_nth_all(self, dtype):
        # 对 Series self.df['values'] 按照 self.df['key'] 列进行分组，并取每个分组的第一个元素，只有当所有元素都缺失时才丢弃
        self.df["values"].groupby(self.df["key"]).nth(0, dropna="all")

    def time_series_nth(self, dtype):
        # 对 Series self.df['values'] 按照 self.df['key'] 列进行分组，并取每个分组的第一个元素
        self.df["values"].groupby(self.df["key"]).nth(0)


class DateAttributes:
    # 这是一个空类，没有任何代码需要注释
    pass
    # 在类中定义一个设置方法 `setup`
    def setup(self):
        # 生成一个日期范围，从 "1/1/2000" 到 "12/31/2005"，频率为每小时
        rng = date_range("1/1/2000", "12/31/2005", freq="h")
        # 将日期范围中的年、月、日分别赋值给对象实例的属性 year, month, day
        self.year, self.month, self.day = rng.year, rng.month, rng.day
        # 创建一个时间序列 `ts`，使用随机生成的数据，索引为日期范围 `rng`
        self.ts = Series(np.random.randn(len(rng)), index=rng)
    
    # 在类中定义一个方法 `time_len_groupby_object`
    def time_len_groupby_object(self):
        # 对时间序列 `ts` 进行分组，按照年、月、日进行分组，返回分组后的对象
        len(self.ts.groupby([self.year, self.month, self.day]))
class Int64:
    # 初始化函数，生成一个大数组，以及对应的 DataFrame 对象
    def setup(self):
        # 生成大小为 2^17 × 5 的随机整数数组
        arr = np.random.randint(-1 << 12, 1 << 12, (1 << 17, 5))
        # 随机选择并复制原数组的部分行，并将其垂直堆叠到原数组上
        i = np.random.choice(len(arr), len(arr) * 5)
        arr = np.vstack((arr, arr[i]))
        # 对数组的行进行随机重排列
        i = np.random.permutation(len(arr))
        arr = arr[i]
        # 创建 DataFrame 对象，列名为 ['a', 'b', 'c', 'd', 'e']
        self.cols = list("abcde")
        self.df = DataFrame(arr, columns=self.cols)
        # 向 DataFrame 对象中添加两列，列名为 'jim' 和 'joe'，值为标准正态分布随机数乘以 10
        self.df["jim"], self.df["joe"] = np.random.randn(2, len(self.df)) * 10

    # 对 DataFrame 对象按照列进行分组，并计算每组的最大值
    def time_overflow(self):
        self.df.groupby(self.cols).max()


class CountMultiDtype:
    # 设置缓存数据，生成包含多种数据类型的 DataFrame 对象
    def setup_cache(self):
        n = 10000
        # 生成随机的 timedelta64 和 datetime64 数组
        offsets = np.random.randint(n, size=n).astype("timedelta64[ns]")
        dates = np.datetime64("now") + offsets
        dates[np.random.rand(n) > 0.5] = np.datetime64("nat")
        offsets[np.random.rand(n) > 0.5] = np.timedelta64("nat")
        # 生成包含随机 NaN 值的浮点数数组
        value2 = np.random.randn(n)
        value2[np.random.rand(n) > 0.5] = np.nan
        # 生成包含随机 NaN 值的对象类型数组
        obj = np.random.choice(list("ab"), size=n).astype(object)
        obj[np.random.randn(n) > 0.5] = np.nan
        # 创建包含以上数据的 DataFrame 对象
        df = DataFrame(
            {
                "key1": np.random.randint(0, 500, size=n),
                "key2": np.random.randint(0, 100, size=n),
                "dates": dates,
                "value2": value2,
                "value3": np.random.randn(n),
                "ints": np.random.randint(0, 1000, size=n),
                "obj": obj,
                "offsets": offsets,
            }
        )
        return df

    # 对包含多种数据类型的 DataFrame 对象按照 key1 和 key2 列进行分组，并计算每组的数量
    def time_multi_count(self, df):
        df.groupby(["key1", "key2"]).count()


class CountMultiInt:
    # 设置缓存数据，生成包含多个整数类型的 DataFrame 对象
    def setup_cache(self):
        n = 10000
        # 创建包含整数数据的 DataFrame 对象
        df = DataFrame(
            {
                "key1": np.random.randint(0, 500, size=n),
                "key2": np.random.randint(0, 100, size=n),
                "ints": np.random.randint(0, 1000, size=n),
                "ints2": np.random.randint(0, 1000, size=n),
            }
        )
        return df

    # 对包含多个整数类型的 DataFrame 对象按照 key1 和 key2 列进行分组，并计算每组的数量
    def time_multi_int_count(self, df):
        df.groupby(["key1", "key2"]).count()

    # 对包含多个整数类型的 DataFrame 对象按照 key1 和 key2 列进行分组，并计算每组的唯一值数量
    def time_multi_int_nunique(self, df):
        df.groupby(["key1", "key2"]).nunique()


class AggFunctions:
    # 设置缓存数据，生成包含多种数据类型的大 DataFrame 对象
    def setup_cache(self):
        N = 10**5
        fac1 = np.array(["A", "B", "C"], dtype="O")
        fac2 = np.array(["one", "two"], dtype="O")
        # 创建包含多种数据类型的大 DataFrame 对象
        df = DataFrame(
            {
                "key1": fac1.take(np.random.randint(0, 3, size=N)),
                "key2": fac2.take(np.random.randint(0, 2, size=N)),
                "value1": np.random.randn(N),
                "value2": np.random.randn(N),
                "value3": np.random.randn(N),
            }
        )
        return df

    # 对包含多种数据类型的大 DataFrame 对象按照 key1 和 key2 列进行分组，并计算每组的平均值、方差和总和
    def time_different_str_functions(self, df):
        df.groupby(["key1", "key2"]).agg(
            {"value1": "mean", "value2": "var", "value3": "sum"}
        )

    # 对包含多种数据类型的大 DataFrame 对象按照 key1 和 key2 列进行分组，并计算每组的总和、最小值和最大值
    def time_different_str_functions_multicol(self, df):
        df.groupby(["key1", "key2"]).agg(["sum", "min", "max"])
    # 定义一个方法来处理 DataFrame，计算单列的多种统计函数
    def time_different_str_functions_singlecol(self, df):
        # 按照 "key1" 列对 DataFrame 进行分组，然后对 "value1" 列计算均值，"value2" 列计算方差，"value3" 列计算总和
        df.groupby("key1").agg({"value1": "mean", "value2": "var", "value3": "sum"})
# 定义 GroupStrings 类，用于操作字符串分组
class GroupStrings:
    # 设置函数，用于生成随机数据并初始化 DataFrame 对象
    def setup(self):
        # 设定数据集大小
        n = 2 * 10**5
        # 生成所有可能的四字母组合
        alpha = list(map("".join, product(ascii_letters, repeat=4)))
        # 从 alpha 中随机选择一部分作为数据集
        data = np.random.choice(alpha, (n // 5, 4), replace=False)
        # 扩展数据集，使每个数据重复五次
        data = np.repeat(data, 5, axis=0)
        # 创建 DataFrame 对象，包含列 'abcd' 和 'joe'
        self.df = DataFrame(data, columns=list("abcd"))
        # 为 'joe' 列赋值，使用正态分布生成随机数并四舍五入到三位小数
        self.df["joe"] = (np.random.randn(len(self.df)) * 10).round(3)
        # 对 DataFrame 进行随机行的重排
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    # 时间函数，对 DataFrame 对象按 'abcd' 列进行分组，并取每组的最大值
    def time_multi_columns(self):
        self.df.groupby(list("abcd")).max()


# 定义 MultiColumn 类，用于处理多列数据操作
class MultiColumn:
    # 设置缓存函数，生成包含随机数据的 DataFrame 对象
    def setup_cache(self):
        # 设定数据集大小
        N = 10**5
        # 创建两个包含重复和随机化的键数组
        key1 = np.tile(np.arange(100, dtype=object), 1000)
        key2 = key1.copy()
        np.random.shuffle(key1)
        np.random.shuffle(key2)
        # 创建 DataFrame 对象，包含列 'key1', 'key2', 'data1', 'data2'
        df = DataFrame(
            {
                "key1": key1,
                "key2": key2,
                "data1": np.random.randn(N),
                "data2": np.random.randn(N),
            }
        )
        return df

    # 时间函数，使用 lambda 函数对 DataFrame 对象按 ['key1', 'key2'] 列进行分组，并聚合求和
    def time_lambda_sum(self, df):
        df.groupby(["key1", "key2"]).agg(lambda x: x.values.sum())

    # 时间函数，直接对 DataFrame 对象按 ['key1', 'key2'] 列进行分组，并求和
    def time_cython_sum(self, df):
        df.groupby(["key1", "key2"]).sum()

    # 时间函数，使用 lambda 函数对 DataFrame 对象按 ['key1', 'key2'] 列分组，并对 'data1' 列聚合求和
    def time_col_select_lambda_sum(self, df):
        df.groupby(["key1", "key2"])["data1"].agg(lambda x: x.values.sum())

    # 时间函数，直接对 DataFrame 对象按 ['key1', 'key2'] 列分组，并对 'data1' 列聚合求和
    def time_col_select_str_sum(self, df):
        df.groupby(["key1", "key2"])["data1"].agg("sum")


# 定义 Size 类，处理包含不同数据类型的 DataFrame 对象
class Size:
    # 设置函数，生成包含日期和数值数据的 DataFrame 对象
    def setup(self):
        # 设定数据集大小
        n = 10**5
        # 创建随机时间偏移数组和日期数组
        offsets = np.random.randint(n, size=n).astype("timedelta64[ns]")
        dates = np.datetime64("now") + offsets
        # 创建 DataFrame 对象，包含列 'key1', 'key2', 'value1', 'value2', 'value3', 'dates'
        self.df = DataFrame(
            {
                "key1": np.random.randint(0, 500, size=n),
                "key2": np.random.randint(0, 100, size=n),
                "value1": np.random.randn(n),
                "value2": np.random.randn(n),
                "value3": np.random.randn(n),
                "dates": dates,
            }
        )
        # 创建包含随机数的 Series 对象
        self.draws = Series(np.random.randn(n))
        # 创建分类标签的 Series 对象
        labels = Series(["foo", "bar", "baz", "qux"] * (n // 4))
        self.cats = labels.astype("category")

    # 时间函数，对 DataFrame 对象按 ['key1', 'key2'] 列进行分组，并返回每组的大小
    def time_multi_size(self):
        self.df.groupby(["key1", "key2"]).size()

    # 时间函数，对 Series 对象按分类标签进行分组，并返回每组的大小
    def time_category_size(self):
        self.draws.groupby(self.cats, observed=True).size()


# 定义 Shift 类，处理 DataFrame 对象的分组位移操作
class Shift:
    # 设置函数，生成包含 'g' 和 'v' 列的 DataFrame 对象
    def setup(self):
        # 设定数据集大小
        N = 18
        # 创建 DataFrame 对象，包含列 'g' 和 'v'
        self.df = DataFrame({"g": ["a", "b"] * 9, "v": list(range(N))})

    # 时间函数，对 DataFrame 对象按 'g' 列进行分组，并对 'v' 列进行位移操作（默认情况）
    def time_defaults(self):
        self.df.groupby("g").shift()

    # 时间函数，对 DataFrame 对象按 'g' 列进行分组，并对 'v' 列进行位移操作（设定填充值为 99）
    def time_fill_value(self):
        self.df.groupby("g").shift(fill_value=99)


# 定义 Fillna 类，处理 DataFrame 对象的缺失值填充操作
class Fillna:
    # 设置函数，生成包含缺失值的 DataFrame 对象，并设置 'group' 列为索引
    def setup(self):
        # 设定数据集大小
        N = 100
        # 创建 DataFrame 对象，包含列 'group', 'value'，并将 'group' 列设为索引
        self.df = DataFrame(
            {"group": [1] * N + [2] * N, "value": [np.nan, 1.0] * N}
        ).set_index("group")

    # 时间函数，对 DataFrame 对象按 'group' 列进行分组，并对缺失值使用前向填充
    def time_df_ffill(self):
        self.df.groupby("group").ffill()

    # 时间函数，对 DataFrame 对象按 'group' 列进行分组，并对缺失值使用后向填充
    def time_df_bfill(self):
        self.df.groupby("group").bfill()

    # 时间函数，对 Series 对象按 'group' 列进行分组，并对缺失值使用前向填充
    def time_srs_ffill(self):
        self.df.groupby("group")["value"].ffill()

    # 时间函数，对 Series 对象按 'group' 列进行分组，并对缺失值使用后向填充
    def time_srs_bfill(self):
        self.df.groupby("group")["value"].bfill()
# 定义一个名为 GroupByMethods 的类，用于存储不同参数和方法的列表
class GroupByMethods:
    # 参数名列表，包括数据类型、方法、应用和列数
    param_names = ["dtype", "method", "application", "ncols"]
    # 参数值列表，分别列出了数据类型、方法、应用和列数的可能取值
    params = [
        ["int", "int16", "float", "object", "datetime", "uint"],  # 数据类型的可能取值
        [
            "all", "any", "bfill", "count", "cumcount", "cummax",  # 不同方法的可能取值
            "cummin", "cumprod", "cumsum", "describe", "diff",
            "ffill", "first", "head", "last", "max", "min", "median",
            "mean", "nunique", "pct_change", "prod", "quantile",
            "rank", "sem", "shift", "size", "skew", "std", "sum",
            "tail", "unique", "value_counts", "var"
        ],
        ["direct", "transformation"],  # 应用的可能取值
        [1, 5],  # 列数的可能取值
        ["cython", "numba"]  # 应用程序的可能取值
    ]
    def setup(self, dtype, method, application, ncols, engine):
        if method in method_blocklist.get(dtype, {}):
            # 如果方法在特定数据类型的阻止列表中，则抛出未实现错误，跳过基准测试
            raise NotImplementedError  # skip benchmark

        if ncols != 1 and method in ["value_counts", "unique"]:
            # 当列数不为1且方法为"value_counts"或"unique"时，DataFrameGroupBy不支持这些方法
            raise NotImplementedError

        if application == "transformation" and method in [
            "describe",
            "head",
            "tail",
            "unique",
            "value_counts",
            "size",
        ]:
            # 当应用为"transformation"且方法为描述性统计或数据检索时，DataFrameGroupBy不支持这些方法
            raise NotImplementedError

        # Numba目前不支持多个转换函数、字符串作为转换函数、多列分组，以及一些方法的内核缺失
        if (
            engine == "numba"
            and method in _numba_unsupported_methods
            or ncols > 1
            or application == "transformation"
            or dtype == "datetime"
        ):
            raise NotImplementedError

        if method == "describe":
            ngroups = 20
        elif method == "skew":
            ngroups = 100
        else:
            ngroups = 1000
        size = ngroups * 2
        rng = np.arange(ngroups).reshape(-1, 1)
        rng = np.broadcast_to(rng, (len(rng), ncols))
        taker = np.random.randint(0, ngroups, size=size)
        values = rng.take(taker, axis=0)
        if dtype == "int":
            key = np.random.randint(0, size, size=size)
        elif dtype in ("int16", "uint"):
            key = np.random.randint(0, size, size=size, dtype=dtype)
        elif dtype == "float":
            key = np.concatenate(
                [np.random.random(ngroups) * 0.1, np.random.random(ngroups) * 10.0]
            )
        elif dtype == "object":
            key = ["foo"] * size
        elif dtype == "datetime":
            key = date_range("1/1/2011", periods=size, freq="s")

        cols = [f"values{n}" for n in range(ncols)]
        df = DataFrame(values, columns=cols)
        df["key"] = key

        if len(cols) == 1:
            cols = cols[0]

        # 不是所有操作都支持engine关键字参数
        kwargs = {}
        if engine == "numba":
            kwargs["engine"] = engine

        if application == "transformation":
            # 根据"key"分组并对每列进行变换操作
            self.as_group_method = lambda: df.groupby("key")[cols].transform(
                method, **kwargs
            )
            # 根据列名分组并对"key"进行变换操作
            self.as_field_method = lambda: df.groupby(cols)["key"].transform(
                method, **kwargs
            )
        else:
            # 部分函数调用，将方法应用到根据"key"分组的列上
            self.as_group_method = partial(
                getattr(df.groupby("key")[cols], method), **kwargs
            )
            # 部分函数调用，将方法应用到根据列名分组的"key"上
            self.as_field_method = partial(
                getattr(df.groupby(cols)["key"], method), **kwargs
            )

    def time_dtype_as_group(self, dtype, method, application, ncols, engine):
        # 调用"as_group_method"执行操作
        self.as_group_method()
    # 定义一个类方法 time_dtype_as_field，接收参数 self, dtype, method, application, ncols, engine
    def time_dtype_as_field(self, dtype, method, application, ncols, engine):
        # 调用对象的 as_field_method 方法，但没有传递任何参数，可能需要补充参数以完整调用
        self.as_field_method()
class GroupByCythonAgg:
    """
    Benchmarks specifically targeting our cython aggregation algorithms
    (using a big enough dataframe with simple key, so a large part of the
    time is actually spent in the grouped aggregation).
    """

    # 定义参数名称列表
    param_names = ["dtype", "method"]
    # 定义参数组合列表
    params = [
        ["float64"],  # 数据类型为 float64
        [
            "sum",      # 聚合方法为求和
            "prod",     # 聚合方法为求积
            "min",      # 聚合方法为最小值
            "max",      # 聚合方法为最大值
            "idxmin",   # 聚合方法为最小值的索引
            "idxmax",   # 聚合方法为最大值的索引
            "mean",     # 聚合方法为均值
            "median",   # 聚合方法为中位数
            "var",      # 聚合方法为方差
            "first",    # 聚合方法为第一个值
            "last",     # 聚合方法为最后一个值
            "any",      # 聚合方法为逻辑或
            "all",      # 聚合方法为逻辑与
        ],
    ]

    def setup(self, dtype, method):
        # 生成包含随机数据的大型 DataFrame
        N = 1_000_000
        df = DataFrame(np.random.randn(N, 10), columns=list("abcdefghij"))
        # 在 DataFrame 中添加一个随机生成的 key 列
        df["key"] = np.random.randint(0, 100, size=N)
        self.df = df

    def time_frame_agg(self, dtype, method):
        # 对 DataFrame 根据 key 列进行分组，并应用指定的聚合方法
        self.df.groupby("key").agg(method)


class GroupByNumbaAgg(GroupByCythonAgg):
    """
    Benchmarks specifically targeting our numba aggregation algorithms
    (using a big enough dataframe with simple key, so a large part of the
    time is actually spent in the grouped aggregation).
    """

    def setup(self, dtype, method):
        # 如果方法不被 numba 支持，则抛出 NotImplementedError 异常
        if method in _numba_unsupported_methods:
            raise NotImplementedError
        # 调用父类的 setup 方法，生成数据
        super().setup(dtype, method)

    def time_frame_agg(self, dtype, method):
        # 对 DataFrame 根据 key 列进行分组，并应用指定的聚合方法，使用 numba 引擎
        self.df.groupby("key").agg(method, engine="numba")


class GroupByCythonAggEaDtypes:
    """
    Benchmarks specifically targeting our cython aggregation algorithms
    (using a big enough dataframe with simple key, so a large part of the
    time is actually spent in the grouped aggregation).
    """

    # 定义参数名称列表
    param_names = ["dtype", "method"]
    # 定义参数组合列表
    params = [
        ["Float64", "Int64", "Int32"],  # 数据类型为 Float64, Int64, Int32
        [
            "sum",      # 聚合方法为求和
            "prod",     # 聚合方法为求积
            "min",      # 聚合方法为最小值
            "max",      # 聚合方法为最大值
            "mean",     # 聚合方法为均值
            "median",   # 聚合方法为中位数
            "var",      # 聚合方法为方差
            "first",    # 聚合方法为第一个值
            "last",     # 聚合方法为最后一个值
            "any",      # 聚合方法为逻辑或
            "all",      # 聚合方法为逻辑与
        ],
    ]

    def setup(self, dtype, method):
        # 生成包含随机数据的大型 DataFrame，指定列的数据类型
        N = 1_000_000
        df = DataFrame(
            np.random.randint(0, high=100, size=(N, 10)),
            columns=list("abcdefghij"),
            dtype=dtype,
        )
        # 将每第五行的数据设置为缺失值
        df.loc[list(range(1, N, 5)), list("abcdefghij")] = NA
        # 在 DataFrame 中添加一个随机生成的 key 列
        df["key"] = np.random.randint(0, 100, size=N)
        self.df = df

    def time_frame_agg(self, dtype, method):
        # 对 DataFrame 根据 key 列进行分组，并应用指定的聚合方法
        self.df.groupby("key").agg(method)


class Cumulative:
    # 定义参数名称列表
    param_names = ["dtype", "method", "with_nans"]
    # 定义参数组合列表
    params = [
        ["float64", "int64", "Float64", "Int64"],  # 数据类型为 float64, int64, Float64, Int64
        ["cummin", "cummax", "cumsum"],            # 累积方法为最小值、最大值、求和
        [True, False],                            # 是否包含 NaN 值
    ]
    # 设置函数，用于初始化数据帧
    def setup(self, dtype, method, with_nans):
        # 如果需要处理 NaN 值且数据类型为 int64，则抛出未实现错误
        if with_nans and dtype == "int64":
            raise NotImplementedError("Construction of df would raise")

        # 设置数据点数 N 为 500,000
        N = 500_000
        # 生成随机整数键数组，范围在 0 到 99 之间
        keys = np.random.randint(0, 100, size=N)
        # 生成随机整数值数组，范围在 -10 到 9 之间，形状为 (N, 5)
        vals = np.random.randint(-10, 10, (N, 5))

        # 如果需要处理 NaN 值
        if with_nans:
            # 将 vals 数组复制为浮点数类型的空值数组
            null_vals = vals.astype(float, copy=True)
            # 将每隔两行的值设置为 NaN
            null_vals[::2, :] = np.nan
            # 将每隔三行的值设置为 NaN
            null_vals[::3, :] = np.nan
            # 创建 DataFrame 对象 df，使用指定的列名和数据类型
            df = DataFrame(null_vals, columns=list("abcde"), dtype=dtype)
            # 将键数组作为新列 "key" 添加到 df 中
            df["key"] = keys
            # 将创建的数据帧赋值给类属性 self.df
            self.df = df
        else:
            # 创建 DataFrame 对象 df，使用 vals 数组和指定的列名
            # 直接将数据类型转换为指定的 dtype，无需复制
            df = DataFrame(vals, columns=list("abcde")).astype(dtype, copy=False)
            # 将键数组作为新列 "key" 添加到 df 中
            df["key"] = keys
            # 将创建的数据帧赋值给类属性 self.df
            self.df = df

    # 时间框架转换函数，对数据帧按键分组并进行指定方法的转换操作
    def time_frame_transform(self, dtype, method, with_nans):
        self.df.groupby("key").transform(method)
# 定义一个名为 RankWithTies 的类，用于处理带有平级排名的情况
class RankWithTies:
    # 类属性：参数名称列表
    param_names = ["dtype", "tie_method"]
    # 类属性：参数组合列表
    params = [
        ["float64", "float32", "int64", "datetime64"],
        ["first", "average", "dense", "min", "max"],
    ]

    # 设置函数，初始化数据框和数据
    def setup(self, dtype, tie_method):
        # 设置数据量 N
        N = 10**4
        # 根据 dtype 初始化数据
        if dtype == "datetime64":
            # 若 dtype 是 datetime64 类型，则创建日期时间数据
            data = np.array([Timestamp("2011/01/01")] * N, dtype=dtype)
        else:
            # 否则创建一列全为 1 的数据，数据类型由 dtype 决定
            data = np.ones(N, dtype=dtype)
        # 创建 DataFrame 对象 df，包含 "values" 和 "key" 两列数据
        self.df = DataFrame({"values": data, "key": ["foo"] * N})

    # 时间函数：计算带有平级排名的操作时间
    def time_rank_ties(self, dtype, tie_method):
        # 使用 groupby 方法对 "key" 列进行分组，并进行带有指定 tie_method 的排名计算
        self.df.groupby("key").rank(method=tie_method)


# 定义一个名为 Float32 的类，用于处理 float32 类型的数据操作
class Float32:
    # 类属性：GH 13335
    def setup(self):
        # 创建两个长度为 10000 的随机浮点数数组，并转换为 float32 类型
        tmp1 = (np.random.random(10000) * 0.1).astype(np.float32)
        tmp2 = (np.random.random(10000) * 10.0).astype(np.float32)
        # 连接两个数组为一个新数组
        tmp = np.concatenate((tmp1, tmp2))
        # 重复 tmp 数组中的数据 10 次，创建 DataFrame 对象 df，包含两列 "a" 和 "b"
        arr = np.repeat(tmp, 10)
        self.df = DataFrame({"a": arr, "b": arr})

    # 时间函数：计算对 "b" 列进行分组，并计算分组后的总和
    def time_sum(self):
        self.df.groupby(["a"])["b"].sum()


# 定义一个名为 String 的类，用于处理字符串类型的数据操作
class String:
    # 类属性：GH#41596
    param_names = ["dtype", "method"]
    params = [
        ["str", "string[python]"],
        [
            "sum",
            "min",
            "max",
            "first",
            "last",
            "any",
            "all",
        ],
    ]

    # 设置函数，初始化具有随机数据的 DataFrame 对象 df
    def setup(self, dtype, method):
        # 创建列名列表 cols
        cols = list("abcdefghjkl")
        # 创建一个形状为 (10000, len(cols)) 的随机整数数组，作为数据，数据类型为 dtype
        self.df = DataFrame(
            np.random.randint(0, 100, size=(10_000, len(cols))),
            columns=cols,
            dtype=dtype,
        )

    # 时间函数：计算对 "a" 列进行分组，并应用指定的字符串处理方法
    def time_str_func(self, dtype, method):
        # 对 "a" 列以外的所有列应用指定的方法 method 进行聚合操作
        self.df.groupby("a")[self.df.columns[1:]].agg(method)


# 定义一个名为 Categories 的类，用于处理分类数据类型的操作
class Categories:
    # 类属性：参数 observed 的两个可能取值
    params = [True, False]
    param_names = ["observed"]

    # 设置函数，根据 observed 值创建三种不同的 DataFrame 对象
    def setup(self, observed):
        # 设置数据量 N
        N = 10**5
        # 创建随机数组 arr
        arr = np.random.random(N)
        
        # 创建三种不同类型的数据字典，并分别创建对应的 DataFrame 对象
        data = {"a": Categorical(np.random.randint(10000, size=N)), "b": arr}
        self.df = DataFrame(data)
        
        data = {
            "a": Categorical(np.random.randint(10000, size=N), ordered=True),
            "b": arr,
        }
        self.df_ordered = DataFrame(data)
        
        data = {
            "a": Categorical(
                np.random.randint(100, size=N), categories=np.arange(10000)
            ),
            "b": arr,
        }
        self.df_extra_cat = DataFrame(data)

    # 时间函数：计算对 "a" 列进行分组，并计算每组中 "b" 列的计数，排序观察值是否启用
    def time_groupby_sort(self, observed):
        self.df.groupby("a", observed=observed)["b"].count()

    # 时间函数：计算对 "a" 列进行分组，并计算每组中 "b" 列的计数，不排序观察值是否启用
    def time_groupby_nosort(self, observed):
        self.df.groupby("a", observed=observed, sort=False)["b"].count()

    # 时间函数：计算对 "a" 列进行分组（有序分类），并计算每组中 "b" 列的计数，排序观察值是否启用
    def time_groupby_ordered_sort(self, observed):
        self.df_ordered.groupby("a", observed=observed)["b"].count()

    # 时间函数：计算对 "a" 列进行分组（有序分类），并计算每组中 "b" 列的计数，不排序观察值是否启用
    def time_groupby_ordered_nosort(self, observed):
        self.df_ordered.groupby("a", observed=observed, sort=False)["b"].count()

    # 时间函数：计算对 "a" 列进行分组（额外分类），并计算每组中 "b" 列的计数，排序观察值是否启用
    def time_groupby_extra_cat_sort(self, observed):
        self.df_extra_cat.groupby("a", observed=observed)["b"].count()

    # 时间函数：计算对 "a" 列进行分组（额外分类），并计算每组中 "b" 列的计数，不排序观察值是否启用
    def time_groupby_extra_cat_nosort(self, observed):
        self.df_extra_cat.groupby("a", observed=observed, sort=False)["b"].count()
class MultipleCategories:
    # 设置方法，初始化包含不同类型数据的 DataFrame
    def setup(self):
        # 定义数据规模 N
        N = 10**3
        # 创建随机数组 arr
        arr = np.random.random(N)
        # 创建包含随机分类数据和随机数组的字典 data
        data = {
            "a1": Categorical(np.random.randint(10000, size=N)),
            "a2": Categorical(np.random.randint(10000, size=N)),
            "b": arr,
        }
        # 创建 DataFrame self.df
        self.df = DataFrame(data)
        # 创建包含有序随机分类数据和随机数组的字典 data
        data = {
            "a1": Categorical(np.random.randint(10000, size=N), ordered=True),
            "a2": Categorical(np.random.randint(10000, size=N), ordered=True),
            "b": arr,
        }
        # 创建有序 DataFrame self.df_ordered
        self.df_ordered = DataFrame(data)
        # 创建包含额外分类数据和随机数组的字典 data
        data = {
            "a1": Categorical(np.random.randint(100, size=N), categories=np.arange(N)),
            "a2": Categorical(np.random.randint(100, size=N), categories=np.arange(N)),
            "b": arr,
        }
        # 创建包含额外分类 DataFrame self.df_extra_cat
        self.df_extra_cat = DataFrame(data)

    # 测试分组并排序的性能方法
    def time_groupby_sort(self):
        self.df.groupby(["a1", "a2"], observed=False)["b"].count()

    # 测试分组不排序的性能方法
    def time_groupby_nosort(self):
        self.df.groupby(["a1", "a2"], observed=False, sort=False)["b"].count()

    # 测试有序数据分组并排序的性能方法
    def time_groupby_ordered_sort(self):
        self.df_ordered.groupby(["a1", "a2"], observed=False)["b"].count()

    # 测试有序数据分组不排序的性能方法
    def time_groupby_ordered_nosort(self):
        self.df_ordered.groupby(["a1", "a2"], observed=False, sort=False)["b"].count()

    # 测试包含额外分类数据分组并排序的性能方法
    def time_groupby_extra_cat_sort(self):
        self.df_extra_cat.groupby(["a1", "a2"], observed=False)["b"].count()

    # 测试包含额外分类数据分组不排序的性能方法
    def time_groupby_extra_cat_nosort(self):
        self.df_extra_cat.groupby(["a1", "a2"], observed=False, sort=False)["b"].count()

    # 测试分组并应用累积和的性能方法
    def time_groupby_transform(self):
        self.df_extra_cat.groupby(["a1", "a2"], observed=False)["b"].cumsum()


class Datelike:
    # GH 14338
    # 定义参数和参数名
    params = ["period_range", "date_range", "date_range_tz"]
    param_names = ["grouper"]

    # 初始化方法，根据选择的 grouper 创建时间序列和随机数据的 DataFrame
    def setup(self, grouper):
        # 定义数据规模 N
        N = 10**4
        # 创建时间范围映射
        rng_map = {
            "period_range": period_range,
            "date_range": date_range,
            "date_range_tz": partial(date_range, tz="US/Central"),
        }
        # 根据选择的 grouper 创建时间序列 self.grouper
        self.grouper = rng_map[grouper]("1900-01-01", freq="D", periods=N)
        # 创建随机数据 DataFrame self.df
        self.df = DataFrame(np.random.randn(10**4, 2))

    # 测试分组并求和的性能方法
    def time_sum(self, grouper):
        self.df.groupby(self.grouper).sum()


class SumBools:
    # GH 2692
    # 初始化方法，创建包含布尔值的 DataFrame
    def setup(self):
        # 定义数据规模 N
        N = 500
        # 创建包含索引和布尔列的 DataFrame self.df
        self.df = DataFrame({"ii": range(N), "bb": [True] * N})

    # 测试分组并求布尔值和的性能方法
    def time_groupby_sum_booleans(self):
        self.df.groupby("ii").sum()


class SumMultiLevel:
    # GH 9049
    # 设置超时时间
    timeout = 120.0

    # 初始化方法，创建多级索引 DataFrame self.df
    def setup(self):
        # 定义数据规模 N
        N = 50
        # 创建包含多级索引的 DataFrame self.df
        self.df = DataFrame(
            {"A": list(range(N)) * 2, "B": range(N * 2), "C": 1}
        ).set_index(["A", "B"])

    # 测试多级索引分组并求和的性能方法
    def time_groupby_sum_multiindex(self):
        self.df.groupby(level=[0, 1]).sum()


class SumTimeDelta:
    # GH 20660
    # 待实现，暂无代码内容
    pass
    # 设置函数，用于初始化测试数据
    def setup(self):
        # 设定数据量 N 为 10000
        N = 10**4
        # 创建一个 DataFrame，包含 N 行，每行包含 100 列随机整数数据（范围在 1000 到 100000 之间）
        self.df = DataFrame(
            np.random.randint(1000, 100000, (N, 100)),
            # 设置 DataFrame 的索引为 N 行随机生成的整数（范围在 0 到 199 之间）
            index=np.random.randint(200, size=(N,)),
        ).astype("timedelta64[ns]")  # 将 DataFrame 的数据类型转换为 timedelta64[ns]
    
        # 创建一个 df_int 变量，它是 df 的深拷贝，并转换数据类型为 int64
        self.df_int = self.df.copy().astype("int64")
    
    # 定义函数，用于测试按时间间隔分组后求和操作的性能
    def time_groupby_sum_timedelta(self):
        # 对 df 执行按索引值分组并对每组执行求和操作
        self.df.groupby(lambda x: x).sum()
    
    # 定义函数，用于测试按整数类型数据分组后求和操作的性能
    def time_groupby_sum_int(self):
        # 对 df_int 执行按索引值分组并对每组执行求和操作
        self.df_int.groupby(lambda x: x).sum()
class Transform:
    # 设置函数，初始化各种数据和DataFrame对象
    def setup(self):
        # 定义变量 n1 和 n2
        n1 = 400
        n2 = 250
        # 创建多级索引对象 MultiIndex
        index = MultiIndex(
            # 设置索引的层次结构和代码
            levels=[np.arange(n1), Index([f"i-{i}" for i in range(n2)], dtype=object)],
            codes=[np.repeat(range(n1), n2).tolist(), list(range(n2)) * n1],
            names=["lev1", "lev2"],
        )
        # 创建一个 n1 * n2 大小的随机数组 arr，并在特定位置设置 NaN 值
        arr = np.random.randn(n1 * n2, 3)
        arr[::10000, 0] = np.nan
        arr[1::10000, 1] = np.nan
        arr[2::10000, 2] = np.nan
        # 根据随机数组创建 DataFrame 对象 data，使用自定义的索引和列名
        data = DataFrame(arr, index=index, columns=["col1", "col20", "col3"])
        self.df = data  # 将 data 赋值给实例变量 df

        # 创建一个 n 行 n 列的随机数组，作为宽格式的 DataFrame 对象 df_wide
        n = 1000
        self.df_wide = DataFrame(
            np.random.randn(n, n),
            index=np.random.choice(range(10), n),
        )

        # 创建一个 n 行 3 列的随机数组，作为高格式的 DataFrame 对象 df_tall
        n = 1_000_000
        self.df_tall = DataFrame(
            np.random.randn(n, 3),
            index=np.random.randint(0, 5, n),
        )

        # 创建一个 n 行 3 列的随机整数数组，作为 DataFrame 对象 df1
        n = 20000
        self.df1 = DataFrame(
            np.random.randint(1, n, (n, 3)), columns=["jim", "joe", "jolie"]
        )
        self.df2 = self.df1.copy()  # 复制 df1 到 df2
        self.df2["jim"] = self.df2["joe"]  # 修改 df2 的 jim 列为 joe 列的值

        # 创建一个 n 行 3 列的随机整数数组，作为 DataFrame 对象 df3
        self.df3 = DataFrame(
            np.random.randint(1, (n / 10), (n, 3)), columns=["jim", "joe", "jolie"]
        )
        self.df4 = self.df3.copy()  # 复制 df3 到 df4
        self.df4["jim"] = self.df4["joe"]  # 修改 df4 的 jim 列为 joe 列的值

    # 使用 lambda 函数计算每个分组的最大值
    def time_transform_lambda_max(self):
        self.df.groupby(level="lev1").transform(lambda x: max(x))

    # 使用字符串参数计算每个分组的最大值
    def time_transform_str_max(self):
        self.df.groupby(level="lev1").transform("max")

    # 使用 lambda 函数计算每个分组在高格式 DataFrame 对象 df_tall 上的最大值
    def time_transform_lambda_max_tall(self):
        self.df_tall.groupby(level=0).transform(lambda x: np.max(x, axis=0))

    # 使用 lambda 函数计算每个分组在宽格式 DataFrame 对象 df_wide 上的最大值
    def time_transform_lambda_max_wide(self):
        self.df_wide.groupby(level=0).transform(lambda x: np.max(x, axis=0))

    # 在 df1 上按多个键（"jim" 和 "joe"）计算 "jolie" 列的最大值
    def time_transform_multi_key1(self):
        self.df1.groupby(["jim", "joe"])["jolie"].transform("max")

    # 在 df2 上按多个键（"jim" 和 "joe"）计算 "jolie" 列的最大值
    def time_transform_multi_key2(self):
        self.df2.groupby(["jim", "joe"])["jolie"].transform("max")

    # 在 df3 上按多个键（"jim" 和 "joe"）计算 "jolie" 列的最大值
    def time_transform_multi_key3(self):
        self.df3.groupby(["jim", "joe"])["jolie"].transform("max")

    # 在 df4 上按多个键（"jim" 和 "joe"）计算 "jolie" 列的最大值
    def time_transform_multi_key4(self):
        self.df4.groupby(["jim", "joe"])["jolie"].transform("max")


class TransformBools:
    # 初始化函数，设置转换点和转换信号 DataFrame 对象
    def setup(self):
        N = 120000
        transition_points = np.sort(np.random.choice(np.arange(N), 1400))
        transitions = np.zeros(N, dtype=np.bool_)
        transitions[transition_points] = True
        self.g = transitions.cumsum()  # 计算转换点的累积和
        self.df = DataFrame({"signal": np.random.rand(N)})  # 创建包含信号的随机数组 DataFrame 对象

    # 计算信号列的均值，并按转换点分组
    def time_transform_mean(self):
        self.df["signal"].groupby(self.g).transform("mean")


class TransformNaN:
    # GH 12737
    # 初始化函数，创建包含 NaN 值的 DataFrame 对象
    def setup(self):
        self.df_nans = DataFrame(
            {"key": np.repeat(np.arange(1000), 10), "B": np.nan, "C": np.nan}
        )
        # 在特定位置填充 NaN 值
        self.df_nans.loc[4::10, "B":"C"] = 5

    # 按 key 列分组，并计算每组的第一个非 NaN 值
    def time_first(self):
        self.df_nans.groupby("key").transform("first")


class TransformEngine:
    param_names = ["parallel"]
    params = [[True, False]]
    # 设置方法，初始化数据框并根据第一列分组
    def setup(self, parallel):
        # 设定常量 N 为 1000
        N = 10**3
        # 创建数据框，包含两列数据：第一列是从 '0' 到 '99' 重复 N 次，第二列是 0 到 99 重复 N 次
        data = DataFrame(
            {0: [str(i) for i in range(100)] * N, 1: list(range(100)) * N},
            columns=[0, 1],
        )
        # 将并行参数赋值给对象的并行属性
        self.parallel = parallel
        # 根据第一列对数据框进行分组
        self.grouper = data.groupby(0)

    # 使用 Numba 引擎处理时间序列数据
    def time_series_numba(self, parallel):
        # 定义一个函数，将给定的值乘以 5
        def function(values, index):
            return values * 5

        # 对分组后的第二列数据应用函数 function，使用 Numba 引擎，并传递并行参数
        self.grouper[1].transform(
            function, engine="numba", engine_kwargs={"parallel": self.parallel}
        )

    # 使用 Cython 引擎处理时间序列数据
    def time_series_cython(self, parallel):
        # 定义一个函数，将给定的值乘以 5
        def function(values):
            return values * 5

        # 对分组后的第二列数据应用函数 function，使用 Cython 引擎
        self.grouper[1].transform(function, engine="cython")

    # 使用 Numba 引擎处理整个数据框
    def time_dataframe_numba(self, parallel):
        # 定义一个函数，将给定的值乘以 5
        def function(values, index):
            return values * 5

        # 对整个分组后的数据框应用函数 function，使用 Numba 引擎，并传递并行参数
        self.grouper.transform(
            function, engine="numba", engine_kwargs={"parallel": self.parallel}
        )

    # 使用 Cython 引擎处理整个数据框
    def time_dataframe_cython(self, parallel):
        # 定义一个函数，将给定的值乘以 5
        def function(values):
            return values * 5

        # 对整个分组后的数据框应用函数 function，使用 Cython 引擎
        self.grouper.transform(function, engine="cython")
class AggEngine:
    # 类级别变量，定义了参数名称列表
    param_names = ["parallel"]
    # 类级别变量，定义了参数值列表
    params = [[True, False]]

    # 设置方法，用于初始化数据和设置分组
    def setup(self, parallel):
        # 定义常量 N，并创建 DataFrame 对象 data
        N = 10**3
        data = DataFrame(
            {0: [str(i) for i in range(100)] * N, 1: list(range(100)) * N},
            columns=[0, 1],
        )
        # 设置类实例的并行属性
        self.parallel = parallel
        # 使用列 0 进行分组，创建分组器对象
        self.grouper = data.groupby(0)

    # 使用 numba 引擎对时间序列数据进行聚合
    def time_series_numba(self, parallel):
        # 定义内部函数 function，对传入的值进行运算
        def function(values, index):
            total = 0
            for i, value in enumerate(values):
                if i % 2:
                    total += value + 5
                else:
                    total += value * 2
            return total

        # 使用 groupby 对象的 agg 方法，使用 numba 引擎进行聚合操作
        self.grouper[1].agg(
            function, engine="numba", engine_kwargs={"parallel": self.parallel}
        )

    # 使用 cython 引擎对时间序列数据进行聚合
    def time_series_cython(self, parallel):
        # 定义内部函数 function，对传入的值进行运算
        def function(values):
            total = 0
            for i, value in enumerate(values):
                if i % 2:
                    total += value + 5
                else:
                    total += value * 2
            return total

        # 使用 groupby 对象的 agg 方法，使用 cython 引擎进行聚合操作
        self.grouper[1].agg(function, engine="cython")

    # 使用 numba 引擎对 DataFrame 进行聚合
    def time_dataframe_numba(self, parallel):
        # 定义内部函数 function，对传入的值进行运算
        def function(values, index):
            total = 0
            for i, value in enumerate(values):
                if i % 2:
                    total += value + 5
                else:
                    total += value * 2
            return total

        # 使用 groupby 对象的 agg 方法，使用 numba 引擎进行聚合操作
        self.grouper.agg(
            function, engine="numba", engine_kwargs={"parallel": self.parallel}
        )

    # 使用 cython 引擎对 DataFrame 进行聚合
    def time_dataframe_cython(self, parallel):
        # 定义内部函数 function，对传入的值进行运算
        def function(values):
            total = 0
            for i, value in enumerate(values):
                if i % 2:
                    total += value + 5
                else:
                    total += value * 2
            return total

        # 使用 groupby 对象的 agg 方法，使用 cython 引擎进行聚合操作
        self.grouper.agg(function, engine="cython")


class Sample:
    # 示例类的设置方法，用于初始化数据
    def setup(self):
        # 定义常量 N，并创建包含零数组的 DataFrame 对象
        N = 10**3
        self.df = DataFrame({"a": np.zeros(N)})
        # 创建分组数组 groups 和权重数组 weights
        self.groups = np.arange(0, N)
        self.weights = np.ones(N)

    # 对 DataFrame 进行分组抽样
    def time_sample(self):
        self.df.groupby(self.groups).sample(n=1)

    # 对 DataFrame 进行带权重的分组抽样
    def time_sample_weights(self):
        self.df.groupby(self.groups).sample(n=1, weights=self.weights)


class Resample:
    # GH 28635，示例类的设置方法，用于初始化数据
    def setup(self):
        num_timedeltas = 20_000
        num_groups = 3

        # 使用 MultiIndex 创建具有不同时间戳的数据
        index = MultiIndex.from_product(
            [
                np.arange(num_groups),
                to_timedelta(np.arange(num_timedeltas), unit="s"),
            ],
            names=["groups", "timedeltas"],
        )
        data = np.random.randint(0, 1000, size=(len(index)))

        # 创建具有 MultiIndex 的 DataFrame 对象
        self.df = DataFrame(data, index=index).reset_index("timedeltas")
        self.df_multiindex = DataFrame(data, index=index)

    # 对 DataFrame 进行分组重采样
    def time_resample(self):
        self.df.groupby(level="groups").resample("10s", on="timedeltas").mean()
    # 定义一个方法用于在多层索引的DataFrame上进行时间重采样
    def time_resample_multiindex(self):
        # 按照 "groups" 层级进行分组，并且按照 "timedeltas" 层级以10秒为间隔进行时间重采样
        self.df_multiindex.groupby(level="groups").resample(
            "10s", level="timedeltas"
        ).mean()
# 导入 `setup` 函数从 `.pandas_vb_common` 模块
from .pandas_vb_common import setup  # noqa: F401 isort:skip
```