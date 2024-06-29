# `D:\src\scipysrc\pandas\asv_bench\benchmarks\timedelta.py`

```
"""
Timedelta benchmarks with non-tslibs dependencies.  See
benchmarks.tslibs.timedelta for benchmarks that rely only on tslibs.
"""

# 导入所需的模块和函数
from pandas import (
    DataFrame,
    Series,
    timedelta_range,
)

# 定义一个类，用于测试时间访问器
class DatetimeAccessor:
    # 设置缓存数据的方法
    def setup_cache(self):
        # 设置数据点数 N
        N = 100000
        # 创建一个时间间隔序列，从 "1 days" 开始，以每小时 ("h") 为频率，共计 N 个时间间隔
        series = Series(timedelta_range("1 days", periods=N, freq="h"))
        return series

    # 测试时间访问器方法的性能
    def time_dt_accessor(self, series):
        # 访问时间访问器
        series.dt

    # 测试获取时间间隔的天数方法的性能
    def time_timedelta_days(self, series):
        # 访问时间间隔的天数属性
        series.dt.days

    # 测试获取时间间隔的秒数方法的性能
    def time_timedelta_seconds(self, series):
        # 访问时间间隔的秒数属性
        series.dt.seconds

    # 测试获取时间间隔的微秒数方法的性能
    def time_timedelta_microseconds(self, series):
        # 访问时间间隔的微秒数属性
        series.dt.microseconds

    # 测试获取时间间隔的纳秒数方法的性能
    def time_timedelta_nanoseconds(self, series):
        # 访问时间间隔的纳秒数属性
        series.dt.nanoseconds


# 定义一个类，用于测试时间间隔索引的性能
class TimedeltaIndexing:
    # 设置方法，初始化数据
    def setup(self):
        # 创建一个时间间隔索引，从 "1985" 年开始，每天 ("D") 为频率，共计 1000 个时间间隔
        self.index = timedelta_range(start="1985", periods=1000, freq="D")
        # 创建另一个时间间隔索引，从 "1986" 年开始，每天 ("D") 为频率，共计 1000 个时间间隔
        self.index2 = timedelta_range(start="1986", periods=1000, freq="D")
        # 创建一个带索引的系列，索引为 self.index，值为范围在 [0, 1000) 的整数
        self.series = Series(range(1000), index=self.index)
        # 获取 self.index 中的第 500 个时间间隔
        self.timedelta = self.index[500]

    # 测试获取时间间隔在索引中位置的性能
    def time_get_loc(self):
        # 获取 self.timedelta 在 self.index 中的位置
        self.index.get_loc(self.timedelta)

    # 测试时间间隔索引的浅复制方法的性能
    def time_shallow_copy(self):
        # 执行时间间隔索引的浅复制操作
        self.index._view()

    # 测试系列对象根据时间间隔索引进行位置访问的性能
    def time_series_loc(self):
        # 根据时间间隔索引访问系列对象的位置
        self.series.loc[self.timedelta]

    # 测试创建 DataFrame 并进行对齐操作的性能
    def time_align(self):
        # 创建一个 DataFrame，包含两列 "a" 和 "b"，分别使用 self.series 和 self.series[:500]
        DataFrame({"a": self.series, "b": self.series[:500]})

    # 测试时间间隔索引的交集操作的性能
    def time_intersection(self):
        # 计算 self.index 和 self.index2 的交集
        self.index.intersection(self.index2)

    # 测试时间间隔索引的并集操作的性能
    def time_union(self):
        # 计算 self.index 和 self.index2 的并集
        self.index.union(self.index2)

    # 测试时间间隔索引的唯一值操作的性能
    def time_unique(self):
        # 获取时间间隔索引的唯一值
        self.index.unique()
```