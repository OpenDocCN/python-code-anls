# `D:\src\scipysrc\pandas\asv_bench\benchmarks\period.py`

```
"""
Period benchmarks with non-tslibs dependencies.  See
benchmarks.tslibs.period for benchmarks that rely only on tslibs.
"""

从pandas库中导入所需的类和函数
from pandas import (
    DataFrame,                # 导入DataFrame类，用于操作数据框
    Period,                   # 导入Period类，表示时间段
    PeriodIndex,              # 导入PeriodIndex类，表示时间段索引
    Series,                   # 导入Series类，表示序列数据结构
    date_range,               # 导入date_range函数，生成日期范围
    period_range,             # 导入period_range函数，生成时间段范围
)

从pandas.tseries.frequencies模块中导入to_offset函数
from pandas.tseries.frequencies import to_offset

定义一个类PeriodIndexConstructor
class PeriodIndexConstructor:
    定义类参数params，包含两个列表的组合
    params = [["D"], [True, False]]
    定义类参数param_names，包含两个字符串，表示参数的名称
    param_names = ["freq", "is_offset"]

    定义setup方法，接受freq和is_offset作为参数
    def setup(self, freq, is_offset):
        生成一个日期范围，存储在self.rng中，从"1985"开始，持续1000个时间点
        self.rng = date_range("1985", periods=1000)
        生成一个日期时间范围，存储在self.rng2中，从"1985"开始，持续1000个时间点，并转换为Python datetime格式
        self.rng2 = date_range("1985", periods=1000).to_pydatetime()
        生成一个从2000到2999的整数列表，存储在self.ints中
        self.ints = list(range(2000, 3000))
        生成一个日期范围，以指定的频率freq生成1000个日期，并转换为整数格式存储在self.daily_ints中
        self.daily_ints = (
            date_range("1/1/2000", periods=1000, freq=freq).strftime("%Y%m%d").map(int)
        )
        如果is_offset为True，则将频率转换为偏移量对象，否则保持频率不变
        if is_offset:
            self.freq = to_offset(freq)
        else:
            self.freq = freq

    定义time_from_date_range方法，接受freq和is_offset作为参数
    def time_from_date_range(self, freq, is_offset):
        使用日期范围self.rng创建PeriodIndex对象，指定频率为freq

    定义time_from_pydatetime方法，接受freq和is_offset作为参数
    def time_from_pydatetime(self, freq, is_offset):
        使用日期时间范围self.rng2创建PeriodIndex对象，指定频率为freq

    定义time_from_ints方法，接受freq和is_offset作为参数
    def time_from_ints(self, freq, is_offset):
        使用整数列表self.ints创建PeriodIndex对象，指定频率为freq

    定义time_from_ints_daily方法，接受freq和is_offset作为参数
    def time_from_ints_daily(self, freq, is_offset):
        使用整数日期列表self.daily_ints创建PeriodIndex对象，指定频率为freq

定义一个类DataFramePeriodColumn
class DataFramePeriodColumn:
    定义setup方法
    def setup(self):
        生成一个时间段范围，从"1/1/1990"开始，以秒为频率，持续20000个时间段，存储在self.rng中
        self.rng = period_range(start="1/1/1990", freq="s", periods=20000)
        生成一个空的DataFrame对象，索引为self.rng的长度范围内的整数，存储在self.df中
        self.df = DataFrame(index=range(len(self.rng)))

    定义time_setitem_period_column方法
    def time_setitem_period_column(self):
        将时间段范围self.rng赋值给DataFrame的列"col"

    定义time_set_index方法
    def time_set_index(self):
        将时间段范围self.rng赋值给DataFrame的列"col2"
        将"col2"列设置为索引，使用append=True选项

定义一个类Algorithms
class Algorithms:
    定义类参数params，包含一个字符串列表
    params = ["index", "series"]
    定义类参数param_names，包含一个字符串列表，表示参数的名称
    param_names = ["typ"]

    定义setup方法，接受typ作为参数
    def setup(self, typ):
        创建一个Period对象列表data，包含4个Period对象，每个对象表示一个月份的时间段
        data = [
            Period("2011-01", freq="M"),
            Period("2011-02", freq="M"),
            Period("2011-03", freq="M"),
            Period("2011-04", freq="M"),
        ]
        根据typ参数的值，初始化self.vector：
        - 如果typ为"index"，则创建一个包含1000个重复data列表的PeriodIndex对象，指定频率为"M"
        - 如果typ为"series"，则创建一个包含1000个重复data列表的Series对象

    定义time_drop_duplicates方法，接受typ作为参数
    def time_drop_duplicates(self, typ):
        对self.vector执行drop_duplicates()操作

    定义time_value_counts方法，接受typ作为参数
    def time_value_counts(self, typ):
        对self.vector执行value_counts()操作

定义一个类Indexing
class Indexing:
    定义setup方法
    def setup(self):
        生成一个时间段范围，从"1985"开始，以天为频率，持续1000个时间段，存储在self.index中
        self.index = period_range(start="1985", periods=1000, freq="D")
        生成一个Series对象，索引为self.index，值为0到999的整数，存储在self.series中
        self.series = Series(range(1000), index=self.index)
        设置self.period为self.index的第500个时间段

    定义time_get_loc方法
    def time_get_loc(self):
        查询self.index中self.period的位置索引

    定义time_shallow_copy方法
    def time_shallow_copy(self):
        返回self.index的视图对象，表示浅复制

    定义time_series_loc方法
    def time_series_loc(self):
        查询self.series中self.period的值

    定义time_align方法
    def time_align(self):
        创建一个DataFrame对象，包含两列，"a"列和"b"列，分别使用self.series和self.series的前500个值作为数据

    定义time_intersection方法
    def time_intersection(self):
        对self.index的前750个时间段与self.index从第250个时间段开始的交集进行计算

    定义time_unique方法
    def time_unique(self):
        计算self.index的唯一时间段
```