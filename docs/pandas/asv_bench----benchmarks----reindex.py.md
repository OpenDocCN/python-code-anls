# `D:\src\scipysrc\pandas\asv_bench\benchmarks\reindex.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算

from pandas import (  # 从 Pandas 库中导入以下模块
    DataFrame,  # 数据框
    Index,  # 索引
    MultiIndex,  # 多重索引
    Series,  # 系列
    date_range,  # 日期范围生成器
    period_range,  # 时间段范围生成器
)


class Reindex:
    def setup(self):
        rng = date_range(start="1/1/1970", periods=10000, freq="1min")  # 生成一个日期范围
        self.df = DataFrame(np.random.rand(10000, 10), index=rng, columns=range(10))  # 创建一个随机数填充的数据框
        self.df["foo"] = "bar"  # 添加一个名为 "foo" 的列，并填充为 "bar"
        self.rng_subset = Index(rng[::2])  # 从日期范围中取出每隔两个时间点的索引
        self.df2 = DataFrame(  # 创建一个包含随机数的数据框，行索引为范围内的整数，列索引为范围内的整数
            index=range(10000), data=np.random.rand(10000, 30), columns=range(30)
        )
        N = 5000
        K = 200
        level1 = Index([f"i-{i}" for i in range(N)], dtype=object).values.repeat(K)  # 创建第一级多重索引
        level2 = np.tile(Index([f"i-{i}" for i in range(K)], dtype=object).values, N)  # 创建第二级多重索引
        index = MultiIndex.from_arrays([level1, level2])  # 使用两级索引数组创建多重索引对象
        self.s = Series(np.random.randn(N * K), index=index)  # 创建一个随机数填充的系列，使用上面创建的多重索引作为索引
        self.s_subset = self.s[::2]  # 选择系列中每隔两个索引位置的子集
        self.s_subset_no_cache = self.s[::2].copy()  # 复制系列中每隔两个索引位置的子集，避免缓存

        mi = MultiIndex.from_product([rng, range(100)])  # 使用日期范围和整数范围创建多重索引
        self.s2 = Series(np.random.randn(len(mi)), index=mi)  # 创建一个随机数填充的系列，使用上述多重索引作为索引
        self.s2_subset = self.s2[::2].copy()  # 复制系列中每隔两个索引位置的子集

    def time_reindex_dates(self):
        self.df.reindex(self.rng_subset)  # 根据指定的日期索引子集重新索引数据框

    def time_reindex_columns(self):
        self.df2.reindex(columns=self.df.columns[1:5])  # 根据指定的列索引子集重新索引数据框

    def time_reindex_multiindex_with_cache(self):
        # 多重索引._values 被缓存
        self.s.reindex(self.s_subset.index)  # 根据指定的索引子集重新索引系列

    def time_reindex_multiindex_no_cache(self):
        # 复制以避免多重索引._values 被缓存
        self.s.reindex(self.s_subset_no_cache.index.copy())  # 根据指定的索引子集的副本重新索引系列

    def time_reindex_multiindex_no_cache_dates(self):
        # 复制以避免多重索引._values 被缓存
        self.s2_subset.reindex(self.s2.index.copy())  # 根据指定的索引子集的副本重新索引系列


class ReindexMethod:
    params = [["pad", "backfill"], [date_range, period_range]]
    param_names = ["method", "constructor"]

    def setup(self, method, constructor):
        N = 100000
        self.idx = constructor("1/1/2000", periods=N, freq="1min")  # 使用指定的构造器创建时间序列索引
        self.ts = Series(np.random.randn(N), index=self.idx)[::2]  # 创建一个随机数填充的系列，选择每隔两个索引位置的子集

    def time_reindex_method(self, method, constructor):
        self.ts.reindex(self.idx, method=method)  # 使用指定的方法重新索引系列


class LevelAlign:
    def setup(self):
        self.index = MultiIndex(
            levels=[np.arange(10), np.arange(100), np.arange(100)],  # 创建一个三级多重索引
            codes=[
                np.arange(10).repeat(10000),  # 每个级别的代码
                np.tile(np.arange(100).repeat(100), 10),
                np.tile(np.tile(np.arange(100), 100), 10),
            ],
        )
        self.df = DataFrame(np.random.randn(len(self.index), 4), index=self.index)  # 创建一个随机数填充的数据框，使用多重索引作为索引
        self.df_level = DataFrame(np.random.randn(100, 4), index=self.index.levels[1])  # 创建一个随机数填充的数据框，使用多重索引的第二级作为索引

    def time_align_level(self):
        self.df.align(self.df_level, level=1, copy=False)  # 在指定级别上对齐数据框

    def time_reindex_level(self):
        self.df_level.reindex(self.index, level=1)  # 根据指定级别重新索引数据框


class DropDuplicates:
    params = [True, False]
    param_names = ["inplace"]  # 参数：是否原地操作
    # 设置函数，初始化数据结构
    def setup(self, inplace):
        # 定义常量 N 和 K
        N = 10000
        K = 10
        # 创建 key1 和 key2，它们是包含重复值的索引对象
        key1 = Index([f"i-{i}" for i in range(N)], dtype=object).values.repeat(K)
        key2 = Index([f"i-{i}" for i in range(N)], dtype=object).values.repeat(K)
        # 创建 DataFrame 对象 self.df，包含 key1、key2 和随机生成的值
        self.df = DataFrame(
            {"key1": key1, "key2": key2, "value": np.random.randn(N * K)}
        )
        # 复制 DataFrame self.df 到 self.df_nan，并将前 10000 行置为 NaN
        self.df_nan = self.df.copy()
        self.df_nan.iloc[:10000, :] = np.nan

        # 创建 Series 对象 self.s，包含随机整数值
        self.s = Series(np.random.randint(0, 1000, size=10000))
        # 创建 Series 对象 self.s_str，包含重复的字符串索引
        self.s_str = Series(
            np.tile(Index([f"i-{i}" for i in range(1000)], dtype=object).values, 10)
        )

        # 更新 N 和 K 的值
        N = 1000000
        K = 10000
        # 创建 DataFrame 对象 self.df_int，包含随机整数 key1 列
        key1 = np.random.randint(0, K, size=N)
        self.df_int = DataFrame({"key1": key1})
        # 创建 DataFrame 对象 self.df_bool，包含随机布尔值
        self.df_bool = DataFrame(np.random.randint(0, 2, size=(K, 10), dtype=bool))

    # 时间测试函数：删除 DataFrame self.df 中的重复行，基于 key1 和 key2 列
    def time_frame_drop_dups(self, inplace):
        self.df.drop_duplicates(["key1", "key2"], inplace=inplace)

    # 时间测试函数：删除 DataFrame self.df_nan 中的重复行，基于 key1 和 key2 列
    def time_frame_drop_dups_na(self, inplace):
        self.df_nan.drop_duplicates(["key1", "key2"], inplace=inplace)

    # 时间测试函数：删除 Series self.s 中的重复值
    def time_series_drop_dups_int(self, inplace):
        self.s.drop_duplicates(inplace=inplace)

    # 时间测试函数：删除 Series self.s_str 中的重复值
    def time_series_drop_dups_string(self, inplace):
        self.s_str.drop_duplicates(inplace=inplace)

    # 时间测试函数：删除 DataFrame self.df_int 中的重复行，基于 key1 列
    def time_frame_drop_dups_int(self, inplace):
        self.df_int.drop_duplicates(inplace=inplace)

    # 时间测试函数：删除 DataFrame self.df_bool 中的重复行
    def time_frame_drop_dups_bool(self, inplace):
        self.df_bool.drop_duplicates(inplace=inplace)
# 定义一个名为 Align 的类，用于数据对齐操作
class Align:
    # 设置类的初始化方法，用于准备数据
    def setup(self):
        # 设置变量 n 为 50000，创建包含字符串 "i-0" 到 "i-49999" 的索引对象
        n = 50000
        indices = Index([f"i-{i}" for i in range(n)], dtype=object)
        # 设定子采样大小为 40000，生成随机数填充长度为 n 的 Series 对象 self.x
        self.x = Series(np.random.randn(n), indices)
        # 从 indices 中随机选择 subsample_size 个不重复的索引，创建随机数填充的 Series 对象 self.y
        subsample_size = 40000
        self.y = Series(
            np.random.randn(subsample_size),
            index=np.random.choice(indices, subsample_size, replace=False),
        )

    # 定义一个方法用于计算两个 Series 对象的元素求和
    def time_align_series_irregular_string(self):
        self.x + self.y

# 导入 pandas_vb_common 模块中的 setup 函数，用于数据初始化，忽略未使用的导入警告
from .pandas_vb_common import setup  # noqa: F401 isort:skip
```