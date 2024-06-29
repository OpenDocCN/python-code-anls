# `D:\src\scipysrc\pandas\asv_bench\benchmarks\categoricals.py`

```
# 导入所需的库
import string  # 导入字符串处理模块
import sys  # 导入系统相关模块
import warnings  # 导入警告处理模块

import numpy as np  # 导入数值计算库numpy

import pandas as pd  # 导入数据处理和分析库pandas

try:
    from pandas.api.types import union_categoricals  # 尝试从pandas的API中导入union_categoricals函数
except ImportError:
    try:
        from pandas.types.concat import union_categoricals  # 如果上述导入失败，则从concat模块导入union_categoricals函数
    except ImportError:
        pass  # 如果两者导入都失败，则忽略继续执行

class Constructor:
    def setup(self):
        N = 10**5  # 定义常量N，用于数据重复生成

        # 初始化数据
        self.categories = list("abcde")  # 创建包含字符列表的类别
        self.cat_idx = pd.Index(self.categories)  # 使用pandas创建类别的索引
        self.values = np.tile(self.categories, N)  # 重复类别N次，形成数据值
        self.codes = np.tile(range(len(self.categories)), N)  # 生成类别的代码，重复N次

        # 创建日期时间序列
        self.datetimes = pd.Series(
            pd.date_range("1995-01-01 00:00:00", periods=N // 10, freq="s")
        )
        self.datetimes_with_nat = self.datetimes.copy()  # 复制日期时间序列
        self.datetimes_with_nat.iloc[-1] = pd.NaT  # 将最后一个元素设置为NaT（Not a Time）

        # 创建包含NaN的数据值列表
        self.values_some_nan = list(np.tile(self.categories + [np.nan], N))
        # 创建全部为NaN的数据值列表
        self.values_all_nan = [np.nan] * len(self.values)
        # 创建全部为int8类型1的数据数组
        self.values_all_int8 = np.ones(N, "int8")
        # 创建pandas的分类数据类型对象
        self.categorical = pd.Categorical(self.values, self.categories)
        self.series = pd.Series(self.categorical)  # 创建分类数据类型的Series对象
        # 创建pandas的时间间隔对象
        self.intervals = pd.interval_range(0, 1, periods=N // 10)

    def time_regular(self):
        pd.Categorical(self.values, self.categories)  # 使用普通方式创建分类数据类型

    def time_fastpath(self):
        dtype = pd.CategoricalDtype(categories=self.cat_idx)  # 创建分类数据类型的数据类型对象
        pd.Categorical._simple_new(self.codes, dtype)  # 使用快速路径创建分类数据类型

    def time_datetimes(self):
        pd.Categorical(self.datetimes)  # 使用日期时间创建分类数据类型

    def time_interval(self):
        pd.Categorical(self.datetimes, categories=self.datetimes)  # 使用日期时间创建分类数据类型

    def time_datetimes_with_nat(self):
        pd.Categorical(self.datetimes_with_nat)  # 使用包含NaT的日期时间创建分类数据类型

    def time_with_nan(self):
        pd.Categorical(self.values_some_nan)  # 使用包含NaN的数据值列表创建分类数据类型

    def time_all_nan(self):
        pd.Categorical(self.values_all_nan)  # 使用全部为NaN的数据值列表创建分类数据类型

    def time_from_codes_all_int8(self):
        pd.Categorical.from_codes(self.values_all_int8, self.categories)  # 使用代码和类别创建分类数据类型

    def time_existing_categorical(self):
        pd.Categorical(self.categorical)  # 使用现有的分类数据类型对象创建分类数据类型

    def time_existing_series(self):
        pd.Categorical(self.series)  # 使用现有的Series对象创建分类数据类型


class AsType:
    def setup(self):
        N = 10**5  # 定义常量N，用于数据重复生成

        random_pick = np.random.default_rng().choice  # 从numpy的随机数生成器中选择随机数

        # 定义不同数据类型的类别和数据生成方法
        categories = {
            "str": list(string.ascii_letters),  # 字符串类别为ASCII字母列表
            "int": np.random.randint(2**16, size=154),  # 整数类别为指定范围内的随机整数
            "float": sys.maxsize * np.random.random((38,)),  # 浮点数类别为指定形状的随机浮点数
            "timestamp": [
                pd.Timestamp(x, unit="s") for x in np.random.randint(2**18, size=578)
            ],  # 时间戳类别为指定单位的随机时间戳列表
        }

        # 使用随机选择的类别生成DataFrame
        self.df = pd.DataFrame(
            {col: random_pick(cats, N) for col, cats in categories.items()}
        )

        # 将部分列转换为字符串
        for col in ("int", "float", "timestamp"):
            self.df[f"{col}_as_str"] = self.df[col].astype(str)

        # 将DataFrame的所有列转换为分类数据类型
        for col in self.df.columns:
            self.df[col] = self.df[col].astype("category")

    def astype_str(self):
        [self.df[col].astype("str") for col in "int float timestamp".split()]  # 将指定列转换为字符串类型
    # 将指定列的数据类型转换为整数
    def astype_int(self):
        # 使用列表推导式遍历列名列表，将每列的数据类型转换为整数
        [self.df[col].astype("int") for col in "int_as_str timestamp".split()]

    # 将指定列的数据类型转换为浮点数
    def astype_float(self):
        # 使用列表推导式遍历列名列表，将每列的数据类型转换为浮点数
        [
            self.df[col].astype("float")
            for col in "float_as_str int int_as_str timestamp".split()
        ]

    # 将名为"float"的列数据类型转换为带有时区信息的日期时间类型
    def astype_datetime(self):
        self.df["float"].astype(pd.DatetimeTZDtype(tz="US/Pacific"))
class Concat:
    # 设置函数，初始化各种数据结构和对象
    def setup(self):
        # 设置数据长度
        N = 10**5
        # 创建一个包含大量重复数据的 Pandas Series，并将其转换为分类类型
        self.s = pd.Series(list("aabbcd") * N).astype("category")

        # 创建两个不同的分类类型的 Pandas Categorical 对象
        self.a = pd.Categorical(list("aabbcd") * N)
        self.b = pd.Categorical(list("bbcdjk") * N)

        # 创建两个不同的分类索引对象
        self.idx_a = pd.CategoricalIndex(range(N), range(N))
        self.idx_b = pd.CategoricalIndex(range(N + 1), range(N + 1))

        # 创建两个 DataFrame，使用分类索引作为行索引
        self.df_a = pd.DataFrame(range(N), columns=["a"], index=self.idx_a)
        self.df_b = pd.DataFrame(range(N + 1), columns=["a"], index=self.idx_b)

    # 测试合并操作的性能
    def time_concat(self):
        pd.concat([self.s, self.s])

    # 测试合并分类数据的性能
    def time_union(self):
        union_categoricals([self.a, self.b])

    # 测试在有重叠索引时追加操作的性能
    def time_append_overlapping_index(self):
        self.idx_a.append(self.idx_a)

    # 测试在没有重叠索引时追加操作的性能
    def time_append_non_overlapping_index(self):
        self.idx_a.append(self.idx_b)

    # 测试在有重叠索引时合并 DataFrame 的性能
    def time_concat_overlapping_index(self):
        pd.concat([self.df_a, self.df_a])

    # 测试在没有重叠索引时合并 DataFrame 的性能
    def time_concat_non_overlapping_index(self):
        pd.concat([self.df_a, self.df_b])


class ValueCounts:
    # 参数化测试，测试是否在计算中包含 NaN 值
    params = [True, False]
    param_names = ["dropna"]

    # 初始化函数，生成包含大量随机数据的分类类型 Series 对象
    def setup(self, dropna):
        n = 5 * 10**5
        arr = [f"s{i:04d}" for i in np.random.randint(0, n // 10, size=n)]
        self.ts = pd.Series(arr).astype("category")

    # 测试 value_counts() 方法的性能
    def time_value_counts(self, dropna):
        self.ts.value_counts(dropna=dropna)


class Repr:
    # 设置函数，初始化包含单个字符串的分类类型 Series 对象
    def setup(self):
        self.sel = pd.Series(["s1234"]).astype("category")

    # 测试将分类类型 Series 对象转换为字符串表示的性能
    def time_rendering(self):
        str(self.sel)


class SetCategories:
    # 设置函数，初始化包含大量随机数据的分类类型 Series 对象
    def setup(self):
        n = 5 * 10**5
        arr = [f"s{i:04d}" for i in np.random.randint(0, n // 10, size=n)]
        self.ts = pd.Series(arr).astype("category")

    # 测试设置分类类型的性能
    def time_set_categories(self):
        self.ts.cat.set_categories(self.ts.cat.categories[::2])


class RemoveCategories:
    # 设置函数，初始化包含大量随机数据的分类类型 Series 对象
    def setup(self):
        n = 5 * 10**5
        arr = [f"s{i:04d}" for i in np.random.randint(0, n // 10, size=n)]
        self.ts = pd.Series(arr).astype("category")

    # 测试移除分类类型的性能
    def time_remove_categories(self):
        self.ts.cat.remove_categories(self.ts.cat.categories[::2])


class Rank:
    # 设置函数，初始化包含大量随机数据的 Series 对象，包括字符串和整数类型
    def setup(self):
        N = 10**5
        ncats = 15

        # 创建包含随机整数的 Series 对象，并将其转换为分类类型
        self.s_str = pd.Series(np.random.randint(0, ncats, size=N).astype(str))
        self.s_str_cat = pd.Series(self.s_str, dtype="category")
        with warnings.catch_warnings(record=True):
            # 创建具有顺序的字符串分类类型
            str_cat_type = pd.CategoricalDtype(set(self.s_str), ordered=True)
            self.s_str_cat_ordered = self.s_str.astype(str_cat_type)

        # 创建包含随机整数的 Series 对象，并将其转换为分类类型
        self.s_int = pd.Series(np.random.randint(0, ncats, size=N))
        self.s_int_cat = pd.Series(self.s_int, dtype="category")
        with warnings.catch_warnings(record=True):
            # 创建具有顺序的整数分类类型
            int_cat_type = pd.CategoricalDtype(set(self.s_int), ordered=True)
            self.s_int_cat_ordered = self.s_int.astype(int_cat_type)

    # 测试对字符串类型的 Series 执行 rank() 方法的性能
    def time_rank_string(self):
        self.s_str.rank()

    # 测试对分类类型的 Series 执行 rank() 方法的性能
    def time_rank_string_cat(self):
        self.s_str_cat.rank()
    # 调用对象的 `rank` 方法对字符串类型的序列进行排名
    def time_rank_string_cat_ordered(self):
        self.s_str_cat_ordered.rank()

    # 调用对象的 `rank` 方法对整数类型的序列进行排名
    def time_rank_int(self):
        self.s_int.rank()

    # 调用对象的 `rank` 方法对整数和类别混合类型的序列进行排名
    def time_rank_int_cat(self):
        self.s_int_cat.rank()

    # 调用对象的 `rank` 方法对整数、类别和顺序混合类型的序列进行排名
    def time_rank_int_cat_ordered(self):
        self.s_int_cat_ordered.rank()
class IsMonotonic:
    # 设置测试数据
    def setup(self):
        N = 1000
        # 创建一个包含大量重复值的分类索引
        self.c = pd.CategoricalIndex(list("a" * N + "b" * N + "c" * N))
        # 根据分类索引创建一个系列
        self.s = pd.Series(self.c)

    # 测试分类索引是否单调递增
    def time_categorical_index_is_monotonic_increasing(self):
        self.c.is_monotonic_increasing

    # 测试分类索引是否单调递减
    def time_categorical_index_is_monotonic_decreasing(self):
        self.c.is_monotonic_decreasing

    # 测试分类系列是否单调递增
    def time_categorical_series_is_monotonic_increasing(self):
        self.s.is_monotonic_increasing

    # 测试分类系列是否单调递减
    def time_categorical_series_is_monotonic_decreasing(self):
        self.s.is_monotonic_decreasing


class Contains:
    # 设置测试数据
    def setup(self):
        N = 10**5
        # 创建一个包含从0到N-1的分类索引
        self.ci = pd.CategoricalIndex(np.arange(N))
        # 获取分类索引的值数组
        self.c = self.ci.values
        # 获取分类索引的第一个类别
        self.key = self.ci.categories[0]

    # 测试分类索引中是否包含指定键值
    def time_categorical_index_contains(self):
        self.key in self.ci

    # 测试分类索引的值数组中是否包含指定键值
    def time_categorical_contains(self):
        self.key in self.c


class CategoricalSlicing:
    params = ["monotonic_incr", "monotonic_decr", "non_monotonic"]
    param_names = ["index"]

    # 设置测试数据
    def setup(self, index):
        N = 10**6
        categories = ["a", "b", "c"]
        if index == "monotonic_incr":
            # 创建一个单调递增的分类数据
            codes = np.repeat([0, 1, 2], N)
        elif index == "monotonic_decr":
            # 创建一个单调递减的分类数据
            codes = np.repeat([2, 1, 0], N)
        elif index == "non_monotonic":
            # 创建一个非单调的分类数据
            codes = np.tile([0, 1, 2], N)
        else:
            raise ValueError(f"Invalid index param: {index}")

        # 根据分类代码和类别创建分类数据
        self.data = pd.Categorical.from_codes(codes, categories=categories)
        self.scalar = 10000
        self.list = list(range(10000))
        self.cat_scalar = "b"

    # 测试通过标量索引访问数据
    def time_getitem_scalar(self, index):
        self.data[self.scalar]

    # 测试通过切片访问数据
    def time_getitem_slice(self, index):
        self.data[: self.scalar]

    # 测试通过类似列表的索引访问数据
    def time_getitem_list_like(self, index):
        self.data[[self.scalar]]

    # 测试通过列表索引访问数据
    def time_getitem_list(self, index):
        self.data[self.list]

    # 测试通过布尔数组索引访问数据
    def time_getitem_bool_array(self, index):
        self.data[self.data == self.cat_scalar]


class Indexing:
    # 设置测试数据
    def setup(self):
        N = 10**5
        # 创建一个分类索引
        self.index = pd.CategoricalIndex(range(N), range(N))
        # 创建一个分类系列，并按索引排序
        self.series = pd.Series(range(N), index=self.index).sort_index()
        # 获取索引中的一个类别
        self.category = self.index[500]

    # 测试获取指定类别的位置
    def time_get_loc(self):
        self.index.get_loc(self.category)

    # 测试创建浅拷贝
    def time_shallow_copy(self):
        self.index._view()

    # 测试对齐操作
    def time_align(self):
        pd.DataFrame({"a": self.series, "b": self.series[:500]})

    # 测试交集操作
    def time_intersection(self):
        self.index[:750].intersection(self.index[250:])

    # 测试获取唯一值
    def time_unique(self):
        self.index.unique()

    # 测试重新索引操作
    def time_reindex(self):
        self.index.reindex(self.index[:500])

    # 测试重新索引缺失值处理
    def time_reindex_missing(self):
        self.index.reindex(["a", "b", "c", "d"])

    # 测试排序操作
    def time_sort_values(self):
        self.index.sort_values(ascending=False)


class SearchSorted:
    # 设置方法，用于初始化对象
    def setup(self):
        # 定义常量 N，并赋值为 100000
        N = 10**5
        # 创建一个 Pandas 的 CategoricalIndex 对象，包含从 0 到 N-1 的整数，并进行排序
        self.ci = pd.CategoricalIndex(np.arange(N)).sort_values()
        # 获取 CategoricalIndex 对象的内部数组并赋值给 self.c
        self.c = self.ci.values
        # 获取 CategoricalIndex 对象的第二个分类，并赋值给 self.key
        self.key = self.ci.categories[1]
    
    # 测试方法，用于测试 CategoricalIndex 对象的 searchsorted 方法的性能
    def time_categorical_index_contains(self):
        # 调用 CategoricalIndex 对象的 searchsorted 方法，寻找 self.key 的位置
        self.ci.searchsorted(self.key)
    
    # 测试方法，用于测试普通数组的 searchsorted 方法的性能
    def time_categorical_contains(self):
        # 调用普通数组 self.c 的 searchsorted 方法，寻找 self.key 的位置
        self.c.searchsorted(self.key)
# 从当前目录下的 pandas_vb_common 模块导入 setup 函数
# noqa: F401 用于告知 linter 忽略 F401 错误，即未使用的导入项
# isort:skip 用于告知 isort 工具跳过对导入顺序的检查
from .pandas_vb_common import setup  # noqa: F401 isort:skip
```