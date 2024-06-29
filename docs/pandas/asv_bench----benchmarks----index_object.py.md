# `D:\src\scipysrc\pandas\asv_bench\benchmarks\index_object.py`

```
import gc  # 导入垃圾回收模块

import numpy as np  # 导入 NumPy 库

from pandas import (  # 从 Pandas 库中导入以下模块
    DatetimeIndex,  # 日期时间索引类
    Index,  # 基础索引类
    IntervalIndex,  # 区间索引类
    MultiIndex,  # 多重索引类
    RangeIndex,  # 范围索引类
    Series,  # 系列数据结构
    date_range,  # 日期范围生成函数
)


class SetOperations:
    params = (
        ["monotonic", "non_monotonic"],  # 索引结构参数：单调和非单调
        ["datetime", "date_string", "int", "strings", "ea_int"],  # 数据类型参数：日期时间、日期字符串、整数、字符串、可选整数
        ["intersection", "union", "symmetric_difference"],  # 操作方法参数：交集、并集、对称差集
    )
    param_names = ["index_structure", "dtype", "method"]  # 参数名称列表

    def setup(self, index_structure, dtype, method):
        N = 10**5  # 数据量级设定
        dates_left = date_range("1/1/2000", periods=N, freq="min")  # 生成日期时间范围
        fmt = "%Y-%m-%d %H:%M:%S"  # 日期时间格式
        date_str_left = Index(dates_left.strftime(fmt))  # 创建日期字符串索引
        int_left = Index(np.arange(N))  # 创建整数索引
        ea_int_left = Index(np.arange(N), dtype="Int64")  # 创建带有可选整数索引
        str_left = Index([f"i-{i}" for i in range(N)], dtype=object)  # 创建字符串索引

        data = {
            "datetime": dates_left,
            "date_string": date_str_left,
            "int": int_left,
            "strings": str_left,
            "ea_int": ea_int_left,
        }  # 数据字典按不同类型存储不同的索引对象

        if index_structure == "non_monotonic":
            data = {k: mi[::-1] for k, mi in data.items()}  # 如果索引结构为非单调，则反转索引顺序

        data = {k: {"left": idx, "right": idx[:-1]} for k, idx in data.items()}  # 为每种数据类型创建左右索引字典

        self.left = data[dtype]["left"]  # 设置当前对象的左侧索引
        self.right = data[dtype]["right"]  # 设置当前对象的右侧索引

    def time_operation(self, index_structure, dtype, method):
        getattr(self.left, method)(self.right)  # 执行指定方法操作


class SetDisjoint:
    def setup(self):
        N = 10**5  # 数据量级设定
        B = N + 20000  # 增加的数据量级
        self.datetime_left = DatetimeIndex(range(N))  # 创建日期时间索引
        self.datetime_right = DatetimeIndex(range(N, B))  # 创建另一个日期时间索引

    def time_datetime_difference_disjoint(self):
        self.datetime_left.difference(self.datetime_right)  # 计算两个日期时间索引的差集


class UnionWithDuplicates:
    def setup(self):
        self.left = Index(np.repeat(np.arange(1000), 100))  # 创建带有重复元素的索引
        self.right = Index(np.tile(np.arange(500, 1500), 50))  # 创建带有重复元素的索引

    def time_union_with_duplicates(self):
        self.left.union(self.right)  # 计算带有重复元素索引的并集


class Range:
    def setup(self):
        self.idx_inc = RangeIndex(start=0, stop=10**6, step=3)  # 创建递增步长的范围索引
        self.idx_dec = RangeIndex(start=10**6, stop=-1, step=-3)  # 创建递减步长的范围索引

    def time_max(self):
        self.idx_inc.max()  # 计算递增范围索引的最大值

    def time_max_trivial(self):
        self.idx_dec.max()  # 计算递减范围索引的最大值

    def time_min(self):
        self.idx_dec.min()  # 计算递减范围索引的最小值

    def time_min_trivial(self):
        self.idx_inc.min()  # 计算递增范围索引的最小值

    def time_get_loc_inc(self):
        self.idx_inc.get_loc(900_000)  # 获取递增范围索引中特定值的位置

    def time_get_loc_dec(self):
        self.idx_dec.get_loc(100_000)  # 获取递减范围索引中特定值的位置

    def time_iter_inc(self):
        for _ in self.idx_inc:  # 迭代递增范围索引
            pass

    def time_iter_dec(self):
        for _ in self.idx_dec:  # 迭代递减范围索引
            pass

    def time_sort_values_asc(self):
        self.idx_inc.sort_values()  # 对递增范围索引进行升序排序

    def time_sort_values_des(self):
        self.idx_inc.sort_values(ascending=False)  # 对递增范围索引进行降序排序


class IndexEquals:
    # 定义一个方法 `setup`，用于设置对象的初始状态
    def setup(self):
        # 创建一个包含 100,000 个整数的 RangeIndex 对象，用于大量数据且需要快速访问
        idx_large_fast = RangeIndex(100_000)
        # 创建一个包含单个日期的时间索引 RangeIndex 对象，适用于少量数据但访问速度较慢
        idx_small_slow = date_range(start="1/1/2012", periods=1)
        # 创建一个由两个索引对象的笛卡尔积构成的 MultiIndex 对象，用于大量数据且需要较慢访问
        self.mi_large_slow = MultiIndex.from_product([idx_large_fast, idx_small_slow])

        # 创建一个只包含一个元素的 RangeIndex 对象，适用于非对象数据类型的索引
        self.idx_non_object = RangeIndex(1)

    # 定义一个方法 `time_non_object_equals_multiindex`，用于比较非对象索引和多级索引是否相等
    def time_non_object_equals_multiindex(self):
        # 调用非对象索引的 `equals` 方法，比较其与多级索引 `self.mi_large_slow` 是否相等
        self.idx_non_object.equals(self.mi_large_slow)
class IndexAppend:
    # 设置索引的初始状态
    def setup(self):
        # 定义常量 N 为 10000
        N = 10_000
        # 创建一个范围索引对象，范围从 0 到 99
        self.range_idx = RangeIndex(0, 100)
        # 将范围索引对象转换为整数索引对象
        self.int_idx = self.range_idx.astype(int)
        # 将整数索引对象转换为字符串索引对象
        self.obj_idx = self.int_idx.astype(str)
        # 初始化空的范围索引对象列表、整数索引对象列表和字符串索引对象列表
        self.range_idxs = []
        self.int_idxs = []
        self.object_idxs = []
        # 循环创建 N 个范围索引对象，每个范围从 i*100 到 (i+1)*100
        for i in range(1, N):
            r_idx = RangeIndex(i * 100, (i + 1) * 100)
            self.range_idxs.append(r_idx)
            i_idx = r_idx.astype(int)
            self.int_idxs.append(i_idx)
            o_idx = i_idx.astype(str)
            self.object_idxs.append(o_idx)
        # 将 self.range_idx 复制 N 次形成列表
        self.same_range_idx = [self.range_idx] * N

    # 测试向范围索引对象追加范围索引对象列表的性能
    def time_append_range_list(self):
        self.range_idx.append(self.range_idxs)

    # 测试向范围索引对象追加相同范围索引对象列表的性能
    def time_append_range_list_same(self):
        self.range_idx.append(self.same_range_idx)

    # 测试向整数索引对象追加整数索引对象列表的性能
    def time_append_int_list(self):
        self.int_idx.append(self.int_idxs)

    # 测试向字符串索引对象追加字符串索引对象列表的性能
    def time_append_obj_list(self):
        self.obj_idx.append(self.object_idxs)


class Indexing:
    # 参数化测试的参数为 ["String", "Float", "Int"]
    params = ["String", "Float", "Int"]
    # 参数化测试的参数名称为 "dtype"
    param_names = ["dtype"]

    # 设置不同数据类型的索引对象
    def setup(self, dtype):
        # 定义常量 N 为 1000000
        N = 10**6
        # 根据数据类型创建不同的索引对象
        if dtype == "String":
            self.idx = Index([f"i-{i}" for i in range(N)], dtype=object)
        elif dtype == "Float":
            self.idx = Index(np.arange(N), dtype=np.float64)
        elif dtype == "Int":
            self.idx = Index(np.arange(N), dtype=np.int64)
        # 创建一个布尔数组，元素为是否能被 3 整除的结果
        self.array_mask = (np.arange(N) % 3) == 0
        # 创建一个布尔系列，使用布尔数组作为数据源
        self.series_mask = Series(self.array_mask)
        # 对索引对象进行排序
        self.sorted = self.idx.sort_values()
        # 将索引对象的前半部分重复一次，形成非唯一索引对象
        half = N // 2
        self.non_unique = self.idx[:half].append(self.idx[:half])
        # 对排序后的索引对象的前半部分重复两次，形成非唯一排序索引对象
        self.non_unique_sorted = self.sorted[:half].repeat(2)
        # 获取排序后索引对象的四分之一处的值作为关键值
        self.key = self.sorted[N // 4]

    # 测试通过布尔数组进行索引的性能
    def time_boolean_array(self, dtype):
        self.idx[self.array_mask]

    # 测试通过布尔系列进行索引的性能
    def time_boolean_series(self, dtype):
        self.idx[self.series_mask]

    # 测试获取索引对象中单个元素的性能
    def time_get(self, dtype):
        self.idx[1]

    # 测试对索引对象进行切片操作的性能
    def time_slice(self, dtype):
        self.idx[:-1]

    # 测试对索引对象进行带步长的切片操作的性能
    def time_slice_step(self, dtype):
        self.idx[::2]

    # 测试获取索引对象中特定值的位置的性能
    def time_get_loc(self, dtype):
        self.idx.get_loc(self.key)

    # 测试获取排序后索引对象中特定值的位置的性能
    def time_get_loc_sorted(self, dtype):
        self.sorted.get_loc(self.key)

    # 测试获取非唯一索引对象中特定值的位置的性能
    def time_get_loc_non_unique(self, dtype):
        self.non_unique.get_loc(self.key)

    # 测试获取非唯一排序索引对象中特定值的位置的性能
    def time_get_loc_non_unique_sorted(self, dtype):
        self.non_unique_sorted.get_loc(self.key)


class Float64IndexMethod:
    # GH 13166
    # 设置浮点数索引的初始状态
    def setup(self):
        # 定义常量 N 为 100000
        N = 100_000
        # 创建一个浮点数数组，类型为 np.float64
        a = np.arange(N, dtype=np.float64)
        # 使用浮点数数组创建索引对象
        self.ind = Index(a * 4.8000000418824129e-08)

    # 测试获取特定值位置的性能
    def time_get_loc(self):
        self.ind.get_loc(0)


class IntervalIndexMethod:
    # GH 24813
    # 参数化测试的参数为 [1000, 100000]
    params = [10**3, 10**5]
    # 设置函数，用于初始化数据结构，接受一个参数 N
    def setup(self, N):
        # 创建左侧索引，将 0 到 N-1 的整数数组和 [0] 合并
        left = np.append(np.arange(N), np.array(0))
        # 创建右侧索引，将 1 到 N 和 [1] 合并
        right = np.append(np.arange(1, N + 1), np.array(1))
        # 使用左右索引数组创建 IntervalIndex 对象并赋值给 self.intv
        self.intv = IntervalIndex.from_arrays(left, right)
        # 访问 IntervalIndex 对象的 _engine 属性

        # 使用左右索引数组创建 IntervalIndex 对象并赋值给 self.intv2
        self.intv2 = IntervalIndex.from_arrays(left + 1, right + 1)
        # 访问 IntervalIndex 对象的 _engine 属性

        # 使用 np.arange(N) 创建 IntervalIndex 对象并赋值给 self.left
        self.left = IntervalIndex.from_breaks(np.arange(N))
        # 使用 np.arange(N-3, 2*N-3) 创建 IntervalIndex 对象并赋值给 self.right

    # 时间性能测试函数，测试 self.intv 的单调递增性
    def time_monotonic_inc(self, N):
        self.intv.is_monotonic_increasing

    # 时间性能测试函数，测试 self.intv 的唯一性
    def time_is_unique(self, N):
        self.intv.is_unique

    # 时间性能测试函数，计算 self.left 和 self.right 的交集
    def time_intersection(self, N):
        self.left.intersection(self.right)

    # 时间性能测试函数，计算 self.intv 和 self.right 的交集
    def time_intersection_one_duplicate(self, N):
        self.intv.intersection(self.right)

    # 时间性能测试函数，计算 self.intv 和 self.intv2 的交集
    def time_intersection_both_duplicate(self, N):
        self.intv.intersection(self.intv2)
class GC:
    # 类级别的属性，包含一个列表作为参数
    params = [1, 2, 5]

    # 定义一个方法用于创建、使用和丢弃索引
    def create_use_drop(self):
        # 创建一个包含100万个元素的索引对象
        idx = Index(list(range(1_000_000)))
        idx._engine  # 访问索引对象的引擎属性

    # 方法用于多次调用 create_use_drop 方法以测试内存峰值和垃圾收集
    def peakmem_gc_instances(self, N):
        try:
            # 禁用垃圾收集器
            gc.disable()

            # 循环调用 create_use_drop 方法 N 次
            for _ in range(N):
                self.create_use_drop()
        finally:
            # 最终重新启用垃圾收集器
            gc.enable()


# 导入 pandas_vb_common 模块中的 setup 函数，忽略 F401 警告，并跳过 isort 检查
from .pandas_vb_common import setup  # noqa: F401 isort:skip
```