# `D:\src\scipysrc\pandas\asv_bench\benchmarks\ctors.py`

```
import numpy as np  # 导入NumPy库，用于数值计算

from pandas import (  # 从Pandas库中导入以下模块
    DatetimeIndex,    # 日期时间索引模块
    Index,            # 索引模块
    MultiIndex,       # 多级索引模块
    Series,           # 系列模块
    Timestamp,        # 时间戳模块
    date_range,       # 日期范围生成模块
)


def no_change(arr):
    return arr  # 返回原始输入数组


def list_of_str(arr):
    return list(arr.astype(str))  # 将输入数组转换为字符串类型列表并返回


def gen_of_str(arr):
    return (x for x in arr.astype(str))  # 返回一个生成器，生成输入数组元素的字符串形式


def arr_dict(arr):
    return dict(zip(range(len(arr)), arr))  # 返回由输入数组的索引和元素组成的字典


def list_of_tuples(arr):
    return [(i, -i) for i in arr]  # 返回由输入数组元素和其相反数组成的元组列表


def gen_of_tuples(arr):
    return ((i, -i) for i in arr)  # 返回一个生成器，生成输入数组元素及其相反数的元组


def list_of_lists(arr):
    return [[i, -i] for i in arr]  # 返回由输入数组元素及其相反数组成的列表


def list_of_tuples_with_none(arr):
    return [(i, -i) for i in arr][:-1] + [None]  # 返回由输入数组元素及其相反数组成的元组列表，最后加入一个None值


def list_of_lists_with_none(arr):
    return [[i, -i] for i in arr][:-1] + [None]  # 返回由输入数组元素及其相反数组成的列表，最后加入一个None值


class SeriesConstructors:
    param_names = ["data_fmt", "with_index", "dtype"]  # 定义参数名列表
    params = [  # 定义参数组合列表
        [
            no_change,
            list,
            list_of_str,
            gen_of_str,
            arr_dict,
            list_of_tuples,
            gen_of_tuples,
            list_of_lists,
            list_of_tuples_with_none,
            list_of_lists_with_none,
        ],
        [False, True],   # 是否使用索引的布尔值列表
        ["float", "int"],  # 数据类型列表
    ]

    # Generators get exhausted on use, so run setup before every call
    number = 1
    repeat = (3, 250, 10)

    def setup(self, data_fmt, with_index, dtype):
        if data_fmt in (gen_of_str, gen_of_tuples) and with_index:
            raise NotImplementedError(
                "Series constructors do not support using generators with indexes"
            )  # 如果数据格式是生成器类型且需要索引，则抛出NotImplementedError异常
        N = 10**4  # 数组长度
        if dtype == "float":
            arr = np.random.randn(N)  # 生成随机浮点数数组
        else:
            arr = np.arange(N)  # 生成整数数组
        self.data = data_fmt(arr)  # 使用指定数据格式处理数组并存储
        self.index = np.arange(N) if with_index else None  # 根据是否需要索引生成索引数组或None

    def time_series_constructor(self, data_fmt, with_index, dtype):
        Series(self.data, index=self.index)  # 使用指定数据和索引构造Series对象


class SeriesDtypesConstructors:
    def setup(self):
        N = 10**4  # 数组长度
        self.arr = np.random.randn(N)  # 生成随机浮点数数组
        self.arr_str = np.array(["foo", "bar", "baz"], dtype=object)  # 生成包含字符串的数组
        self.s = Series(
            [Timestamp("20110101"), Timestamp("20120101"), Timestamp("20130101")]
            * N
            * 10
        )  # 生成包含多个时间戳的Series对象

    def time_index_from_array_string(self):
        Index(self.arr_str)  # 使用字符串数组生成索引对象

    def time_index_from_array_floats(self):
        Index(self.arr)  # 使用浮点数数组生成索引对象

    def time_dtindex_from_series(self):
        DatetimeIndex(self.s)  # 使用Series对象生成日期时间索引对象

    def time_dtindex_from_index_with_series(self):
        Index(self.s)  # 使用Series对象生成索引对象


class MultiIndexConstructor:
    def setup(self):
        N = 10**4  # 数组长度
        self.iterables = [Index([f"i-{i}" for i in range(N)], dtype=object), range(20)]  # 生成多级索引的可迭代对象列表

    def time_multiindex_from_iterables(self):
        MultiIndex.from_product(self.iterables)  # 使用可迭代对象列表生成多级索引对象


class DatetimeIndexConstructor:
    pass  # 日期时间索引构造器类暂未实现
    # 设置函数，用于初始化测试数据
    def setup(self):
        # 定义时间序列的长度
        N = 20_000
        # 生成一个从"1900-01-01"开始的长度为N的日期时间索引
        dti = date_range("1900-01-01", periods=N)

        # 将日期时间索引转换为不同格式的列表，并保存在对象的属性中
        self.list_of_timestamps = dti.tolist()          # 转换为时间戳列表
        self.list_of_dates = dti.date.tolist()          # 转换为日期列表
        self.list_of_datetimes = dti.to_pydatetime().tolist()  # 转换为日期时间对象列表
        self.list_of_str = dti.strftime("%Y-%m-%d").tolist()   # 转换为字符串列表

    # 从时间戳列表创建DatetimeIndex对象的函数
    def time_from_list_of_timestamps(self):
        DatetimeIndex(self.list_of_timestamps)

    # 从日期列表创建DatetimeIndex对象的函数
    def time_from_list_of_dates(self):
        DatetimeIndex(self.list_of_dates)

    # 从日期时间对象列表创建DatetimeIndex对象的函数
    def time_from_list_of_datetimes(self):
        DatetimeIndex(self.list_of_datetimes)

    # 从字符串列表创建DatetimeIndex对象的函数
    def time_from_list_of_str(self):
        DatetimeIndex(self.list_of_str)
# 从当前目录的pandas_vb_common模块导入setup函数
from .pandas_vb_common import setup  # noqa: F401 isort:skip
```