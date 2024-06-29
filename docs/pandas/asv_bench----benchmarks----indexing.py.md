# `D:\src\scipysrc\pandas\asv_bench\benchmarks\indexing.py`

```
"""
These benchmarks are for Series and DataFrame indexing methods.  For the
lower-level methods directly on Index and subclasses, see index_object.py,
indexing_engine.py, and index_cached.py
"""

from datetime import datetime  # 导入 datetime 模块中的 datetime 类
import warnings  # 导入 warnings 模块

import numpy as np  # 导入 NumPy 库，并使用 np 别名

from pandas import (  # 从 Pandas 库中导入多个模块和类
    NA,  # Pandas 中的缺失值标志
    CategoricalIndex,  # 分类索引类
    DataFrame,  # 数据框类
    Index,  # 索引基类
    IntervalIndex,  # 区间索引类
    MultiIndex,  # 多重索引类
    Series,  # 系列类
    concat,  # 合并函数
    date_range,  # 创建日期范围函数
    option_context,  # 上下文选项函数
    period_range,  # 创建周期范围函数
)


class NumericSeriesIndexing:
    params = [  # 定义性能测试参数
        (np.int64, np.uint64, np.float64),  # 数据类型参数
        ("unique_monotonic_inc", "nonunique_monotonic_inc"),  # 索引结构参数
    ]
    param_names = ["dtype", "index_structure"]  # 参数名称列表

    def setup(self, dtype, index_structure):
        N = 10**6  # 定义数据大小
        indices = {
            "unique_monotonic_inc": Index(range(N), dtype=dtype),  # 创建唯一单调递增索引
            "nonunique_monotonic_inc": Index(  # 创建非唯一单调递增索引
                list(range(55)) + [54] + list(range(55, N - 1)), dtype=dtype
            ),
        }
        self.data = Series(np.random.rand(N), index=indices[index_structure])  # 创建具有指定索引的随机数系列
        self.array = np.arange(10000)  # 创建一个 NumPy 数组
        self.array_list = self.array.tolist()  # 将数组转换为列表

    def time_getitem_scalar(self, index, index_structure):
        self.data[800000]  # 获取指定位置的元素

    def time_getitem_slice(self, index, index_structure):
        self.data[:800000]  # 获取切片范围内的元素

    def time_getitem_list_like(self, index, index_structure):
        self.data[[800000]]  # 获取类似列表的索引位置的元素

    def time_getitem_array(self, index, index_structure):
        self.data[self.array]  # 使用数组索引获取元素

    def time_getitem_lists(self, index, index_structure):
        self.data[self.array_list]  # 使用列表索引获取元素

    def time_iloc_array(self, index, index_structure):
        self.data.iloc[self.array]  # 使用整数位置索引数组获取元素

    def time_iloc_list_like(self, index, index_structure):
        self.data.iloc[[800000]]  # 使用类似列表的整数位置索引获取元素

    def time_iloc_scalar(self, index, index_structure):
        self.data.iloc[800000]  # 使用整数位置索引获取单个元素

    def time_iloc_slice(self, index, index_structure):
        self.data.iloc[:800000]  # 使用整数位置索引获取切片范围内的元素

    def time_loc_array(self, index, index_structure):
        self.data.loc[self.array]  # 使用标签索引数组获取元素

    def time_loc_list_like(self, index, index_structure):
        self.data.loc[[800000]]  # 使用类似列表的标签索引获取元素

    def time_loc_scalar(self, index, index_structure):
        self.data.loc[800000]  # 使用标签索引获取单个元素

    def time_loc_slice(self, index, index_structure):
        self.data.loc[:800000]  # 使用标签索引获取切片范围内的元素


class NumericMaskedIndexing:
    monotonic_list = list(range(10**6))  # 创建单调递增列表
    non_monotonic_list = list(range(50)) + [54, 53, 52, 51] + list(range(55, 10**6 - 1))  # 创建非单调递增列表

    params = [  # 定义性能测试参数
        ("Int64", "UInt64", "Float64"),  # 数据类型参数
        (True, False),  # 单调性参数
    ]
    param_names = ["dtype", "monotonic"]  # 参数名称列表

    def setup(self, dtype, monotonic):
        indices = {
            True: Index(self.monotonic_list, dtype=dtype),  # 创建单调或非单调索引
            False: Index(self.non_monotonic_list, dtype=dtype).append(  # 创建非单调索引并追加缺失值
                Index([NA], dtype=dtype)
            ),
        }
        self.data = indices[monotonic]  # 根据参数选择索引对象
        self.indexer = np.arange(300, 1_000)  # 创建用于索引的数组
        self.data_dups = self.data.append(self.data)  # 将索引对象复制并连接起来
    # 定义一个方法用于获取索引器，参数包括数据类型和是否单调
    def time_get_indexer(self, dtype, monotonic):
        # 调用数据对象的get_indexer方法，传入索引器参数
        self.data.get_indexer(self.indexer)

    # 定义一个方法用于获取重复值的索引器，参数包括数据类型和是否单调
    def time_get_indexer_dups(self, dtype, monotonic):
        # 调用数据对象的get_indexer_for方法，传入索引器参数
        self.data.get_indexer_for(self.indexer)
class NonNumericSeriesIndexing:
    # 定义参数组合，用于测试索引类型和结构
    params = [
        ("string", "datetime", "period"),
        ("unique_monotonic_inc", "nonunique_monotonic_inc", "non_monotonic"),
    ]
    # 参数名称列表
    param_names = ["index_dtype", "index_structure"]

    # 设置测试前的准备工作
    def setup(self, index, index_structure):
        # 设置数据规模
        N = 10**6
        # 根据不同的索引类型进行初始化
        if index == "string":
            index = Index([f"i-{i}" for i in range(N)], dtype=object)
        elif index == "datetime":
            index = date_range("1900", periods=N, freq="s")
        elif index == "period":
            index = period_range("1900", periods=N, freq="s")
        # 对索引进行排序
        index = index.sort_values()
        # 断言索引唯一且单调递增
        assert index.is_unique and index.is_monotonic_increasing
        # 根据索引结构类型进行相应处理
        if index_structure == "nonunique_monotonic_inc":
            index = index.insert(item=index[2], loc=2)[:-1]
        elif index_structure == "non_monotonic":
            index = index[::2].append(index[1::2])
            assert len(index) == N
        # 使用随机数据创建 Series
        self.s = Series(np.random.rand(N), index=index)
        # 设置用于测试的标签
        self.lbl = index[80000]
        # 预热索引映射
        self.s[self.lbl]

    # 测试根据标签进行切片获取元素的性能
    def time_getitem_label_slice(self, index, index_structure):
        self.s[: self.lbl]

    # 测试根据位置范围进行切片获取元素的性能
    def time_getitem_pos_slice(self, index, index_structure):
        self.s[:80000]

    # 测试根据标量标签获取元素的性能
    def time_getitem_scalar(self, index, index_structure):
        self.s[self.lbl]

    # 测试根据类似列表结构获取元素的性能
    def time_getitem_list_like(self, index, index_structure):
        self.s[[self.lbl]]


class DataFrameStringIndexing:
    # 设置数据框的索引和列
    def setup(self):
        index = Index([f"i-{i}" for i in range(1000)], dtype=object)
        columns = Index([f"i-{i}" for i in range(30)], dtype=object)
        # 创建具有随机数据的数据框
        with warnings.catch_warnings(record=True):
            self.df = DataFrame(np.random.randn(1000, 30), index=index, columns=columns)
        # 设置用于测试的标量索引和列标签
        self.idx_scalar = index[100]
        self.col_scalar = columns[10]
        # 布尔索引
        self.bool_indexer = self.df[self.col_scalar] > 0
        # 将布尔索引转换为对象类型
        self.bool_obj_indexer = self.bool_indexer.astype(object)
        # 将布尔索引转换为布尔类型
        self.boolean_indexer = (self.df[self.col_scalar] > 0).astype("boolean")

    # 测试使用 loc 方法获取元素的性能
    def time_loc(self):
        self.df.loc[self.idx_scalar, self.col_scalar]

    # 测试使用 at 方法获取元素的性能
    def time_at(self):
        self.df.at[self.idx_scalar, self.col_scalar]

    # 测试使用 at 方法设置元素的性能
    def time_at_setitem(self):
        self.df.at[self.idx_scalar, self.col_scalar] = 0.0

    # 测试使用标量索引获取元素的性能
    def time_getitem_scalar(self):
        self.df[self.col_scalar][self.idx_scalar]

    # 测试使用布尔行索引获取元素的性能
    def time_boolean_rows(self):
        self.df[self.bool_indexer]

    # 测试使用对象类型布尔行索引获取元素的性能
    def time_boolean_rows_object(self):
        self.df[self.bool_obj_indexer]

    # 测试使用布尔类型布尔行索引获取元素的性能
    def time_boolean_rows_boolean(self):
        self.df[self.boolean_indexer]


class DataFrameNumericIndexing:
    # 参数化测试的参数组合
    params = [
        (np.int64, np.uint64, np.float64),
        ("unique_monotonic_inc", "nonunique_monotonic_inc"),
    ]
    # 参数名称
    param_names = ["dtype", "index_structure"]
    # 设置函数，用于初始化测试数据
    def setup(self, dtype, index_structure):
        # 设置数据集大小为 10^5
        N = 10**5
        # 创建不同类型的索引结构
        indices = {
            "unique_monotonic_inc": Index(range(N), dtype=dtype),  # 唯一单调递增索引
            "nonunique_monotonic_inc": Index(
                list(range(55)) + [54] + list(range(55, N - 1)), dtype=dtype
            ),  # 非唯一单调递增索引
        }
        # 创建索引重复的 numpy 数组
        self.idx_dupe = np.array(range(30)) * 99
        # 创建包含随机数据的 DataFrame，使用指定的索引结构
        self.df = DataFrame(np.random.randn(N, 5), index=indices[index_structure])
        # 创建 DataFrame 的重复版本
        self.df_dup = concat([self.df, 2 * self.df, 3 * self.df])
        # 创建用于布尔索引的列表
        self.bool_indexer = [True] * (N // 2) + [False] * (N - N // 2)

    # 测试 iloc 方法处理重复索引的性能
    def time_iloc_dups(self, index, index_structure):
        self.df_dup.iloc[self.idx_dupe]

    # 测试 loc 方法处理重复索引的性能
    def time_loc_dups(self, index, index_structure):
        self.df_dup.loc[self.idx_dupe]

    # 测试 iloc 方法的性能
    def time_iloc(self, index, index_structure):
        self.df.iloc[:100, 0]

    # 测试 loc 方法的性能
    def time_loc(self, index, index_structure):
        self.df.loc[:100, 0]

    # 测试布尔索引的性能
    def time_bool_indexer(self, index, index_structure):
        self.df[self.bool_indexer]
class Take:
    # 参数定义：包含的参数类型和名称
    params = ["int", "datetime"]
    # 参数名称定义：对应每个参数的名称
    param_names = ["index"]

    # 设置方法：初始化测试所需的数据
    def setup(self, index):
        # 定义数组长度
        N = 100000
        # 索引字典，包含不同类型的索引数组
        indexes = {
            "int": Index(np.arange(N), dtype=np.int64),
            "datetime": date_range("2011-01-01", freq="s", periods=N),
        }
        # 根据参数选择相应的索引数组
        index = indexes[index]
        # 创建随机数序列，以选定的索引为索引
        self.s = Series(np.random.rand(N), index=index)
        # 创建随机整数索引器
        self.indexer = np.random.randint(0, N, size=N)

    # 测试方法：测试 Series 对象的 take 方法
    def time_take(self, index):
        self.s.take(self.indexer)


class MultiIndexing:
    # 参数定义：包含的参数类型
    params = [True, False]
    # 参数名称定义：对应每个参数的名称
    param_names = ["unique_levels"]

    # 设置方法：初始化测试所需的数据
    def setup(self, unique_levels):
        # 设置层级数
        self.nlevels = 2
        # 根据参数选择创建 MultiIndex 对象
        if unique_levels:
            mi = MultiIndex.from_arrays([range(1000000)] * self.nlevels)
        else:
            mi = MultiIndex.from_product([range(1000)] * self.nlevels)
        # 创建 DataFrame 对象，用随机数填充，以选择的 MultiIndex 作为索引
        self.df = DataFrame(np.random.randn(len(mi)), index=mi)

        # 定义不同的目标索引对象
        self.tgt_slice = slice(200, 800)
        self.tgt_null_slice = slice(None)
        self.tgt_list = list(range(0, 1000, 10))
        self.tgt_scalar = 500

        # 创建布尔型索引器，每100个值为 True
        bool_indexer = np.zeros(len(mi), dtype=np.bool_)
        bool_indexer[slice(0, len(mi), 100)] = True
        self.tgt_bool_indexer = bool_indexer

    # 测试方法：测试 loc 方法使用部分键和切片的性能
    def time_loc_partial_key_slice(self, unique_levels):
        self.df.loc[self.tgt_slice, :]

    def time_loc_partial_key_null_slice(self, unique_levels):
        self.df.loc[self.tgt_null_slice, :]

    def time_loc_partial_key_list(self, unique_levels):
        self.df.loc[self.tgt_list, :]

    def time_loc_partial_key_scalar(self, unique_levels):
        self.df.loc[self.tgt_scalar, :]

    def time_loc_partial_key_bool_indexer(self, unique_levels):
        self.df.loc[self.tgt_bool_indexer, :]

    def time_loc_all_slices(self, unique_levels):
        # 使用多重切片选择数据
        target = tuple([self.tgt_slice] * self.nlevels)
        self.df.loc[target, :]

    def time_loc_all_null_slices(self, unique_levels):
        # 使用多重空切片选择数据
        target = tuple([self.tgt_null_slice] * self.nlevels)
        self.df.loc[target, :]

    def time_loc_all_lists(self, unique_levels):
        # 使用多重列表选择数据
        target = tuple([self.tgt_list] * self.nlevels)
        self.df.loc[target, :]

    def time_loc_all_scalars(self, unique_levels):
        # 使用多重标量选择数据
        target = tuple([self.tgt_scalar] * self.nlevels)
        self.df.loc[target, :]

    def time_loc_all_bool_indexers(self, unique_levels):
        # 使用多重布尔型索引选择数据
        target = tuple([self.tgt_bool_indexer] * self.nlevels)
        self.df.loc[target, :]

    def time_loc_slice_plus_null_slice(self, unique_levels):
        # 组合切片和空切片选择数据
        target = (self.tgt_slice, self.tgt_null_slice)
        self.df.loc[target, :]

    def time_loc_null_slice_plus_slice(self, unique_levels):
        # 组合空切片和切片选择数据
        target = (self.tgt_null_slice, self.tgt_slice)
        self.df.loc[target, :]

    def time_loc_multiindex(self, unique_levels):
        # 使用 xs 方法选择层级为 0 的数据
        target = self.df.index[::10]
        self.df.loc[target]

    def time_xs_level_0(self, unique_levels):
        # 使用 xs 方法选择特定层级的数据
        target = self.tgt_scalar
        self.df.xs(target, level=0)
    # 定义一个方法用于从数据框中按照第二级索引提取特定数据
    def time_xs_level_1(self, unique_levels):
        # 从实例变量中获取目标标量值
        target = self.tgt_scalar
        # 使用 Pandas 的 xs 方法从数据框中提取第二级索引为目标标量值的数据，但未进行返回操作
        self.df.xs(target, level=1)

    # 定义一个方法用于从数据框中按照完整的索引键提取特定数据
    def time_xs_full_key(self, unique_levels):
        # 创建一个包含多个目标标量值的元组，其长度由实例变量 nlevels 决定
        target = tuple([self.tgt_scalar] * self.nlevels)
        # 使用 Pandas 的 xs 方法从数据框中提取索引键为目标元组的数据，但未进行返回操作
        self.df.xs(target)
# 定义一个名为 IntervalIndexing 的类，用于演示区间索引操作
class IntervalIndexing:
    # 设置缓存的方法，返回一个单调递增的序列
    def setup_cache(self):
        # 使用 np.arange 创建区间索引对象 idx
        idx = IntervalIndex.from_breaks(np.arange(1000001))
        # 创建一个 Series 对象 monotonic，其索引为 idx，数值为从 0 到 999999
        monotonic = Series(np.arange(1000000), index=idx)
        return monotonic

    # 计算使用 [] 运算符获取单个元素的时间
    def time_getitem_scalar(self, monotonic):
        monotonic[80000]

    # 计算使用 .loc 获取单个元素的时间
    def time_loc_scalar(self, monotonic):
        monotonic.loc[80000]

    # 计算使用 [] 运算符获取切片的时间
    def time_getitem_list(self, monotonic):
        monotonic[80000:]

    # 计算使用 .loc 获取切片的时间
    def time_loc_list(self, monotonic):
        monotonic.loc[80000:]


# 定义一个名为 DatetimeIndexIndexing 的类，用于演示日期时间索引操作
class DatetimeIndexIndexing:
    # 设置方法，在 US/Pacific 时区从 "2016-01-01" 开始的 10000 个时间点作为索引
    def setup(self):
        dti = date_range("2016-01-01", periods=10000, tz="US/Pacific")
        # 将索引 dti 转换为 UTC 时区
        dti2 = dti.tz_convert("UTC")
        self.dti = dti
        self.dti2 = dti2

    # 获取 dti 中 dti2 中存在的索引的时间
    def time_get_indexer_mismatched_tz(self):
        # 通过例如 ser = Series(range(len(dti)), index=dti) 这样的方法获取
        # ser[dti2]，返回 dti 中 dti2 的索引
        self.dti.get_indexer(self.dti2)


# 定义一个名为 SortedAndUnsortedDatetimeIndexLoc 的类，用于演示排序和未排序的日期时间索引的 .loc 操作
class SortedAndUnsortedDatetimeIndexLoc:
    # 设置方法，创建两个 DataFrame，一个使用未排序的索引，一个使用排序后的索引
    def setup(self):
        dti = date_range("2016-01-01", periods=10000, tz="US/Pacific")
        index = np.array(dti)

        unsorted_index = index.copy()
        unsorted_index[10] = unsorted_index[20]

        # 创建未排序索引的 DataFrame
        self.df_unsorted = DataFrame(index=unsorted_index, data={"a": 1})
        # 创建排序后索引的 DataFrame
        self.df_sort = DataFrame(index=index, data={"a": 1})

    # 使用 .loc 获取未排序索引的 "2016-6-11" 的行
    def time_loc_unsorted(self):
        self.df_unsorted.loc["2016-6-11"]

    # 使用 .loc 获取排序后索引的 "2016-6-11" 的行
    def time_loc_sorted(self):
        self.df_sort.loc["2016-6-11"]


# 定义一个名为 CategoricalIndexIndexing 的类，用于演示分类索引操作
class CategoricalIndexIndexing:
    params = ["monotonic_incr", "monotonic_decr", "non_monotonic"]
    param_names = ["index"]

    # 设置方法，根据不同的索引类型初始化数据
    def setup(self, index):
        N = 10**5
        values = list("a" * N + "b" * N + "c" * N)
        indices = {
            "monotonic_incr": CategoricalIndex(values),
            "monotonic_decr": CategoricalIndex(reversed(values)),
            "non_monotonic": CategoricalIndex(list("abc" * N)),
        }
        self.data = indices[index]
        # 创建一个唯一值的分类索引
        self.data_unique = CategoricalIndex([str(i) for i in range(N * 3)])

        self.int_scalar = 10000
        self.int_list = list(range(10000))

        self.cat_scalar = "b"
        self.cat_list = ["1", "3"]

    # 使用 [] 运算符获取单个元素的时间
    def time_getitem_scalar(self, index):
        self.data[self.int_scalar]

    # 使用 [] 运算符获取切片的时间
    def time_getitem_slice(self, index):
        self.data[: self.int_scalar]

    # 使用 [] 运算符获取类似列表的时间
    def time_getitem_list_like(self, index):
        self.data[[self.int_scalar]]

    # 使用 [] 运算符获取列表的时间
    def time_getitem_list(self, index):
        self.data[self.int_list]

    # 使用 [] 运算符获取布尔数组的时间
    def time_getitem_bool_array(self, index):
        self.data[self.data == self.cat_scalar]

    # 获取分类索引中指定元素的位置
    def time_get_loc_scalar(self, index):
        self.data.get_loc(self.cat_scalar)

    # 获取唯一值分类索引中多个元素的位置
    def time_get_indexer_list(self, index):
        self.data_unique.get_indexer(self.cat_list)


# 定义一个名为 MethodLookup 的类，用于演示方法查找操作
class MethodLookup:
    # 设置缓存的方法，返回一个空的 Series 对象
    def setup_cache(self):
        s = Series()
        return s

    # 计算获取 iloc 属性的时间
    def time_lookup_iloc(self, s):
        s.iloc

    # 计算获取 loc 属性的时间
    def time_lookup_loc(self, s):
        s.loc
    # 设置函数，用于初始化测试数据框架的字符串列和整数列
    def setup(self):
        # 创建一个包含3000行随机数的DataFrame，列名为"A"
        self.df_string_col = DataFrame(np.random.randn(3000, 1), columns=["A"])
        # 创建一个包含3000行随机数的DataFrame，未指定列名
        self.df_int_col = DataFrame(np.random.randn(3000, 1))
    
    # 测试函数，用于测试从数据框架中获取单列标签为"A"的时间性能
    def time_frame_getitem_single_column_label(self):
        # 获取数据框架self.df_string_col中标签为"A"的列
        self.df_string_col["A"]
    
    # 测试函数，用于测试从数据框架中获取单列整数索引为0的时间性能
    def time_frame_getitem_single_column_int(self):
        # 获取数据框架self.df_int_col中整数索引为0的列
        self.df_int_col[0]
class IndexSingleRow:
    # 定义类变量 params 和 param_names
    params = [True, False]
    param_names = ["unique_cols"]

    def setup(self, unique_cols):
        # 创建一个包含 10**7 个元素的数组，reshape 成 1000000 行，每行 10 列的 DataFrame
        arr = np.arange(10**7).reshape(-1, 10)
        df = DataFrame(arr)
        # 指定每列的数据类型为给定的 dtypes
        dtypes = ["u1", "u2", "u4", "u8", "i1", "i2", "i4", "i8", "f8", "f4"]
        for i, d in enumerate(dtypes):
            df[i] = df[i].astype(d)

        if not unique_cols:
            # 如果 unique_cols 为 False，则将 DataFrame 的列名设置为 ["A", "A"] + 原列名列表
            # 这是为了解决 GH#33032 中的性能问题
            df.columns = ["A", "A"] + list(df.columns[2:])

        # 将设置好的 DataFrame 赋值给实例变量 self.df
        self.df = df

    def time_iloc_row(self, unique_cols):
        # 计算用 iloc 获取第 10000 行的时间
        self.df.iloc[10000]

    def time_loc_row(self, unique_cols):
        # 计算用 loc 获取第 10000 行的时间
        self.df.loc[10000]


class AssignTimeseriesIndex:
    def setup(self):
        # 创建包含 N=100000 个时间序列索引的 DataFrame，随机值填充
        N = 100000
        idx = date_range("1/1/2000", periods=N, freq="h")
        self.df = DataFrame(np.random.randn(N, 1), columns=["A"], index=idx)

    def time_frame_assign_timeseries_index(self):
        # 测试将时间序列索引赋值给新列 "date" 的时间
        self.df["date"] = self.df.index


class InsertColumns:
    def setup(self):
        # 初始化一个包含 1000 行的空 DataFrame 和一个包含随机值的 1000 行 2 列 DataFrame
        self.N = 10**3
        self.df = DataFrame(index=range(self.N))
        self.df2 = DataFrame(np.random.randn(self.N, 2))

    def time_insert(self):
        # 测试在 DataFrame 前 100 列随机插入数据的时间
        for i in range(100):
            self.df.insert(0, i, np.random.randn(self.N), allow_duplicates=True)

    def time_insert_middle(self):
        # 测试在 DataFrame 中间列插入数据的时间，与在前后插入不同
        for i in range(100):
            self.df2.insert(
                1, "colname", np.random.randn(self.N), allow_duplicates=True
            )

    def time_assign_with_setitem(self):
        # 测试使用 setitem 方法将随机数据赋值给 DataFrame 的时间
        for i in range(100):
            self.df[i] = np.random.randn(self.N)

    def time_assign_list_like_with_setitem(self):
        # 测试将随机数据赋值给 DataFrame 的多列的时间
        self.df[list(range(100))] = np.random.randn(self.N, 100)

    def time_assign_list_of_columns_concat(self):
        # 测试将随机数据添加到 DataFrame 的末尾并进行合并的时间
        df = DataFrame(np.random.randn(self.N, 100))
        concat([self.df, df], axis=1)


class Setitem:
    def setup(self):
        # 初始化一个包含 500000 行 500 列随机浮点数的 DataFrame
        N = 500_000
        cols = 500
        self.df = DataFrame(np.random.rand(N, cols))

    def time_setitem(self):
        # 测试将整列赋值为相同值的时间
        self.df[100] = 100

    def time_setitem_list(self):
        # 测试将多列赋值为相同值的时间
        self.df[[100, 200, 300]] = 100


class SetitemObjectDtype:
    # GH#19299

    def setup(self):
        # 初始化一个包含 1000 行 500 列对象类型数据的 DataFrame
        N = 1000
        cols = 500
        self.df = DataFrame(index=range(N), columns=range(cols), dtype=object)

    def time_setitem_object_dtype(self):
        # 测试在对象类型 DataFrame 中赋值的时间
        self.df.loc[0, 1] = 1.0


class ChainIndexing:
    params = [None, "warn"]
    param_names = ["mode"]

    def setup(self, mode):
        # 初始化包含 1000000 行两列数据的 DataFrame
        self.N = 1000000
        self.df = DataFrame({"A": np.arange(self.N), "B": "foo"})

    def time_chained_indexing(self, mode):
        df = self.df
        N = self.N
        with warnings.catch_warnings(record=True):
            with option_context("mode.chained_assignment", mode):
                # 测试链式索引的性能，生成新 DataFrame 并添加新列的时间
                df2 = df[df.A > N // 2]
                df2["C"] = 1.0
class Block:
    # 类变量 params 包含两个元组，每个元组包含一个布尔值和其对应的描述字符串
    params = [
        (True, "True"),
        (np.array(True), "np.array(True)"),
    ]

    # 初始化方法，设置对象的 df 属性为一个 DataFrame 对象
    def setup(self, true_value, mode):
        # 创建一个空的 DataFrame 对象，列数为500，索引从"2010-01-01"到"2011-01-01"
        self.df = DataFrame(
            False,
            columns=np.arange(500).astype(str),
            index=date_range("2010-01-01", "2011-01-01"),
        )

        # 将传入的 true_value 参数赋值给对象的 true_value 属性
        self.true_value = true_value

    # 时间测试方法，用于设置指定时间范围内的 DataFrame 列为给定的 true_value
    def time_test(self, true_value, mode):
        # 设定开始时间和结束时间
        start = datetime(2010, 5, 1)
        end = datetime(2010, 9, 1)
        # 将指定时间范围内的所有行的数据列设置为 true_value
        self.df.loc[start:end, :] = true_value


from .pandas_vb_common import setup  # noqa: F401 isort:skip
```