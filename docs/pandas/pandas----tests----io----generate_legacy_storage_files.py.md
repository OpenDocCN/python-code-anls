# `D:\src\scipysrc\pandas\pandas\tests\io\generate_legacy_storage_files.py`

```
"""
self-contained to write legacy storage pickle files

To use this script. Create an environment where you want
generate pickles, say its for 0.20.3, with your pandas clone
in ~/pandas

. activate pandas_0.20.3
cd ~/pandas/pandas

$ python -m tests.io.generate_legacy_storage_files \
    tests/io/data/legacy_pickle/0.20.3/ pickle

This script generates a storage file for the current arch, system,
and python version
  pandas version: 0.20.3
  output dir    : pandas/pandas/tests/io/data/legacy_pickle/0.20.3/
  storage format: pickle
created pickle file: 0.20.3_x86_64_darwin_3.5.2.pickle

The idea here is you are using the *current* version of the
generate_legacy_storage_files with an *older* version of pandas to
generate a pickle file. We will then check this file into a current
branch, and test using test_pickle.py. This will load the *older*
pickles and test versus the current data that is generated
(with main). These are then compared.

If we have cases where we changed the signature (e.g. we renamed
offset -> freq in Timestamp). Then we have to conditionally execute
in the generate_legacy_storage_files.py to make it
run under the older AND the newer version.

"""

from datetime import timedelta  # 导入timedelta类用于处理时间跨度
import os  # 导入os模块，用于操作系统相关功能
import pickle  # 导入pickle模块，用于序列化和反序列化Python对象
import platform as pl  # 导入platform模块，并使用pl作为别名，用于访问平台相关信息
import sys  # 导入sys模块，用于系统级参数和函数

# Remove script directory from path, otherwise Python will try to
# import the JSON test directory as the json module
sys.path.pop(0)  # 从sys.path中移除第一个路径，避免导入错误的JSON模块路径

import numpy as np  # 导入NumPy库，并使用np作为别名

import pandas  # 导入pandas库
from pandas import (  # 从pandas库中导入多个类和函数
    Categorical,
    DataFrame,
    Index,
    MultiIndex,
    NaT,
    Period,
    RangeIndex,
    Series,
    Timestamp,
    bdate_range,
    date_range,
    interval_range,
    period_range,
    timedelta_range,
)
from pandas.arrays import SparseArray  # 从pandas.arrays子模块导入SparseArray类

from pandas.tseries.offsets import (  # 从pandas.tseries.offsets子模块导入多个时间偏移类
    FY5253,
    BusinessDay,
    BusinessHour,
    CustomBusinessDay,
    DateOffset,
    Day,
    Easter,
    Hour,
    LastWeekOfMonth,
    Minute,
    MonthBegin,
    MonthEnd,
    QuarterBegin,
    QuarterEnd,
    SemiMonthBegin,
    SemiMonthEnd,
    Week,
    WeekOfMonth,
    YearBegin,
    YearEnd,
)


def _create_sp_series():
    nan = np.nan  # 设置变量nan为NumPy中的NaN值

    # nan-based
    arr = np.arange(15, dtype=np.float64)  # 创建一个长度为15的浮点数数组
    arr[7:12] = nan  # 将索引为7到11的元素设为NaN
    arr[-1:] = nan  # 将最后一个元素设为NaN

    bseries = Series(SparseArray(arr, kind="block"))  # 使用稀疏数组创建Series对象
    bseries.name = "bseries"  # 设置Series对象的名称为"bseries"
    return bseries  # 返回创建的Series对象


def _create_sp_tsseries():
    nan = np.nan  # 设置变量nan为NumPy中的NaN值

    # nan-based
    arr = np.arange(15, dtype=np.float64)  # 创建一个长度为15的浮点数数组
    arr[7:12] = nan  # 将索引为7到11的元素设为NaN
    arr[-1:] = nan  # 将最后一个元素设为NaN

    date_index = bdate_range("1/1/2011", periods=len(arr))  # 创建一个工作日时间索引
    bseries = Series(SparseArray(arr, kind="block"), index=date_index)  # 使用稀疏数组和日期索引创建Series对象
    bseries.name = "btsseries"  # 设置Series对象的名称为"btsseries"
    return bseries  # 返回创建的Series对象


def _create_sp_frame():
    nan = np.nan  # 设置变量nan为NumPy中的NaN值

    data = {
        "A": [nan, nan, nan, 0, 1, 2, 3, 4, 5, 6],  # 设置键'A'的值为一个包含NaN和整数的列表
        "B": [0, 1, 2, nan, nan, nan, 3, 4, 5, 6],  # 设置键'B'的值为一个包含整数和NaN的列表
        "C": np.arange(10).astype(np.int64),  # 创建一个长度为10的整数数组，并将其类型转换为int64
        "D": [0, 1, 2, 3, 4, 5, nan, nan, nan, nan],  # 设置键'D'的值为一个包含整数和NaN的列表
    }

    dates = bdate_range("1/1/2011", periods=10)  # 创建一个包含10个工作日日期的时间索引
    # 返回一个 DataFrame 对象，使用给定的数据和日期作为索引
    return DataFrame(data, index=dates).apply(SparseArray)
# 定义函数用于创建 pickle 数据
def create_pickle_data():
    # 创建包含不同数据类型的字典
    data = {
        "A": [0.0, 1.0, 2.0, 3.0, np.nan],  # 浮点数和缺失值组成的列表
        "B": [0, 1, 0, 1, 0],  # 整数列表
        "C": ["foo1", "foo2", "foo3", "foo4", "foo5"],  # 字符串列表
        "D": date_range("1/1/2009", periods=5),  # 日期时间索引
        "E": [0.0, 1, Timestamp("20100101"), "foo", 2.0],  # 混合数据列表
    }

    # 创建包含标量的字典
    scalars = {"timestamp": Timestamp("20130101"), "period": Period("2012", "M")}

    # 创建不同类型的索引对象
    index = {
        "int": Index(np.arange(10)),  # 整数索引
        "date": date_range("20130101", periods=10),  # 日期时间索引
        "period": period_range("2013-01-01", freq="M", periods=10),  # 时期索引
        "float": Index(np.arange(10, dtype=np.float64)),  # 浮点数索引
        "uint": Index(np.arange(10, dtype=np.uint64)),  # 无符号整数索引
        "timedelta": timedelta_range("00:00:00", freq="30min", periods=10),  # 时间间隔索引
    }

    index["range"] = RangeIndex(10)  # 范围索引

    index["interval"] = interval_range(0, periods=10)  # 区间索引

    # 创建多层次索引对象
    mi = {
        "reg2": MultiIndex.from_tuples(
            tuple(
                zip(
                    *[
                        ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],  # 第一层索引标签
                        ["one", "two", "one", "two", "one", "two", "one", "two"],  # 第二层索引标签
                    ]
                )
            ),
            names=["first", "second"],  # 设置索引层级名称
        )
    }

    # 创建包含不同类型 Series 的字典
    series = {
        "float": Series(data["A"]),  # 浮点数 Series
        "int": Series(data["B"]),  # 整数 Series
        "mixed": Series(data["E"]),  # 混合数据 Series
        "ts": Series(  # 时间戳类型 Series
            np.arange(10).astype(np.int64), index=date_range("20130101", periods=10)
        ),
        "mi": Series(  # 多层次索引类型 Series
            np.arange(5).astype(np.float64),
            index=MultiIndex.from_tuples(
                tuple(zip(*[[1, 1, 2, 2, 2], [3, 4, 3, 4, 5]])), names=["one", "two"]
            ),
        ),
        "dup": Series(  # 包含重复索引标签的 Series
            np.arange(5).astype(np.float64), index=["A", "B", "C", "D", "A"]
        ),
        "cat": Series(Categorical(["foo", "bar", "baz"])),  # 分类类型 Series
        "dt": Series(date_range("20130101", periods=5)),  # 日期时间类型 Series
        "dt_tz": Series(date_range("20130101", periods=5, tz="US/Eastern")),  # 时区日期时间类型 Series
        "period": Series([Period("2000Q1")] * 5),  # 时期类型 Series
    }

    # 创建 DataFrame 并设置列标签为 ABCDA
    mixed_dup_df = DataFrame(data)
    mixed_dup_df.columns = list("ABCDA")
    # 创建一个字典 `frame` 包含不同类型的 Pandas DataFrame 对象
    frame = {
        # 创建一个浮点数类型的 DataFrame，包含两列 'A' 和 'B'， 'B' 列为 'float' 列加一
        "float": DataFrame({"A": series["float"], "B": series["float"] + 1}),
        
        # 创建一个整数类型的 DataFrame，包含两列 'A' 和 'B'， 'B' 列为 'int' 列加一
        "int": DataFrame({"A": series["int"], "B": series["int"] + 1}),
        
        # 创建一个混合类型的 DataFrame，包含 'A', 'B', 'C', 'D' 列
        "mixed": DataFrame({k: data[k] for k in ["A", "B", "C", "D"]}),
        
        # 创建一个包含浮点数和整数的 MultiIndex DataFrame，定义了两级索引 'first' 和 'second'
        "mi": DataFrame(
            {"A": np.arange(5).astype(np.float64), "B": np.arange(5).astype(np.int64)},
            index=MultiIndex.from_tuples(
                tuple(
                    zip(
                        *[
                            ["bar", "bar", "baz", "baz", "baz"],
                            ["one", "two", "one", "two", "three"],
                        ]
                    )
                ),
                names=["first", "second"],
            ),
        ),
        
        # 创建一个包含重复列名的 DataFrame，列名为 'A', 'B', 'A'
        "dup": DataFrame(
            np.arange(15).reshape(5, 3).astype(np.float64), columns=["A", "B", "A"]
        ),
        
        # 创建一个包含单列分类数据的 DataFrame，列名为 'A'
        "cat_onecol": DataFrame({"A": Categorical(["foo", "bar"])}),
        
        # 创建一个包含分类数据和整数数据的 DataFrame，列名为 'A' 和 'B'
        "cat_and_float": DataFrame(
            {
                "A": Categorical(["foo", "bar", "baz"]),
                "B": np.arange(3).astype(np.int64),
            }
        ),
        
        # 使用已定义的 `mixed_dup_df` 创建一个名为 'mixed_dup' 的 DataFrame
        "mixed_dup": mixed_dup_df,
        
        # 创建一个包含带时区的时间戳数据的 DataFrame，列名为 'A' 和 'B'，索引为 0 到 4
        "dt_mixed_tzs": DataFrame(
            {
                "A": Timestamp("20130102", tz="US/Eastern"),
                "B": Timestamp("20130603", tz="CET"),
            },
            index=range(5),
        ),
        
        # 创建一个包含带时区的时间戳数据的 DataFrame，列名为 'A', 'B', 'C'，索引为 0 到 4
        "dt_mixed2_tzs": DataFrame(
            {
                "A": Timestamp("20130102", tz="US/Eastern"),
                "B": Timestamp("20130603", tz="CET"),
                "C": Timestamp("20130603", tz="UTC"),
            },
            index=range(5),
        ),
    }

    # 创建一个字典 `cat` 包含不同类型的 Pandas Categorical 对象
    cat = {
        # 创建一个包含小写字母 'a' 到 'g' 的 Categorical 对象，dtype 为 int8
        "int8": Categorical(list("abcdefg")),
        
        # 创建一个包含从 0 到 999 的整数的 Categorical 对象，dtype 为 int16
        "int16": Categorical(np.arange(1000)),
        
        # 创建一个包含从 0 到 9999 的整数的 Categorical 对象，dtype 为 int32
        "int32": Categorical(np.arange(10000)),
    }

    # 创建一个字典 `timestamp` 包含不同类型的 Pandas Timestamp 对象
    timestamp = {
        # 创建一个普通的时间戳对象，表示 2011 年 1 月 1 日
        "normal": Timestamp("2011-01-01"),
        
        # 创建一个 NaT（Not a Time）对象
        "nat": NaT,
        
        # 创建一个带时区的时间戳对象，表示 2011 年 1 月 1 日，时区为 'US/Eastern'
        "tz": Timestamp("2011-01-01", tz="US/Eastern"),
    }

    # 创建一个字典 `off` 包含不同类型的 Pandas Offset 对象
    off = {
        # 创建一个表示年份偏移量的 DateOffset 对象，偏移量为 1 年
        "DateOffset": DateOffset(years=1),
        
        # 创建一个表示小时和纳秒偏移量的 DateOffset 对象，小时为 6，纳秒为 5824
        "DateOffset_h_ns": DateOffset(hour=6, nanoseconds=5824),
        
        # 创建一个表示工作日偏移量的 BusinessDay 对象，偏移量为 9 秒
        "BusinessDay": BusinessDay(offset=timedelta(seconds=9)),
        
        # 创建一个表示工作小时偏移量的 BusinessHour 对象，工作小时从 00:00 到 15:14
        "BusinessHour": BusinessHour(normalize=True, n=6, end="15:14"),
        
        # 创建一个表示自定义工作日偏移量的 CustomBusinessDay 对象，工作日为周一至周五
        "CustomBusinessDay": CustomBusinessDay(weekmask="Mon Fri"),
        
        # 创建一个表示半月初的 SemiMonthBegin 对象，每月初的第 9 天
        "SemiMonthBegin": SemiMonthBegin(day_of_month=9),
        
        # 创建一个表示半月末的 SemiMonthEnd 对象，每月末的第 24 天
        "SemiMonthEnd": SemiMonthEnd(day_of_month=24),
        
        # 创建一个表示月初的 MonthBegin 对象，每月初
        "MonthBegin": MonthBegin(1),
        
        # 创建一个表示月末的 MonthEnd 对象，每月末
        "MonthEnd": MonthEnd(1),
        
        # 创建一个表示季初的 QuarterBegin 对象，每季初
        "QuarterBegin": QuarterBegin(1),
        
        # 创建一个表示季末的 QuarterEnd 对象，每季末
        "QuarterEnd": QuarterEnd(1),
        
        # 创建一个表示日历日的 Day 对象，每日
        "Day": Day(1),
        
        # 创建一个表示年初的 YearBegin 对象，每年初
        "YearBegin": YearBegin(1),
        
        # 创建一个表示年末的 YearEnd 对象，每年末
        "YearEnd": YearEnd(1),
        
        # 创建一个表示周的 Week 对象，每周
        "Week": Week(1),
        
        # 创建一个表示第二周二的 Week 对象，不进行标准化，周二是第 1 天
        "Week_Tues": Week(2, normalize=False, weekday=1),
        
        # 创建一个表示每月第三周的 WeekOfMonth 对象，周四是第 4 天
        "WeekOfMonth": WeekOfMonth(week=3, weekday=4),
        
        # 创建一个表示每月最后一周的 LastWeekOfMonth 对象，周三是第 3 天
        "LastWeekOfMonth": LastWeekOfMonth(n=1, weekday=3),
        
        # 创建一个表示财年（FY5253）的 FY5253 对象，财年从第 7 个月开始，以 'last' 方式变化
        "FY5253": FY5253(n=2, weekday=6, startingMonth=7, variation="last"),
        
        # 创建一个表示复活节的 Easter 对象
        "Easter": Easter(),
        
        # 创建一个表示小时的 Hour 对象，每小时
        "Hour": Hour(1),
        
        # 创建一个表示分钟的 Minute 对象，每分钟
        "Minute": Minute(1),
    }
    `
    # 返回一个包含多个键值对的字典，键对应不同的数据类型或结构
    return {
        # 键为 "series"，值为变量 series 的内容
        "series": series,
        # 键为 "frame"，值为变量 frame 的内容
        "frame": frame,
        # 键为 "index"，值为变量 index 的内容
        "index": index,
        # 键为 "scalars"，值为变量 scalars 的内容
        "scalars": scalars,
        # 键为 "mi"，值为变量 mi 的内容
        "mi": mi,
        # 键为 "sp_series"，值为一个字典，包含两个键 "float" 和 "ts"，分别对应调用 _create_sp_series() 和 _create_sp_tsseries() 的结果
        "sp_series": {"float": _create_sp_series(), "ts": _create_sp_tsseries()},
        # 键为 "sp_frame"，值为一个字典，包含一个键 "float"，对应调用 _create_sp_frame() 的结果
        "sp_frame": {"float": _create_sp_frame()},
        # 键为 "cat"，值为变量 cat 的内容
        "cat": cat,
        # 键为 "timestamp"，值为变量 timestamp 的内容
        "timestamp": timestamp,
        # 键为 "offsets"，值为变量 off 的内容
        "offsets": off,
    }
# 定义函数 platform_name，返回一个由多个平台信息组成的字符串
def platform_name():
    return "_".join(
        [
            str(pandas.__version__),  # 获取 pandas 库的版本号并转换为字符串
            str(pl.machine()),  # 获取当前机器的架构信息并转换为字符串
            str(pl.system().lower()),  # 获取当前操作系统名称并转换为小写字符串
            str(pl.python_version()),  # 获取当前 Python 解释器版本并转换为字符串
        ]
    )


# 定义函数 write_legacy_pickles，生成旧版 pickle 文件
def write_legacy_pickles(output_dir):
    version = pandas.__version__  # 获取 pandas 库的版本号

    # 打印脚本生成存储文件的信息，包括 pandas 版本、输出目录和存储格式
    print(
        "This script generates a storage file for the current arch, system, "
        "and python version"
    )
    print(f"  pandas version: {version}")  # 打印 pandas 版本号
    print(f"  output dir    : {output_dir}")  # 打印输出目录
    print("  storage format: pickle")  # 指定存储格式为 pickle

    pth = f"{platform_name()}.pickle"  # 生成特定平台名称的 pickle 文件路径

    # 打开输出目录下的 pickle 文件，并使用 pickle 协议将数据写入文件
    with open(os.path.join(output_dir, pth), "wb") as fh:
        pickle.dump(create_pickle_data(), fh, pickle.DEFAULT_PROTOCOL)

    print(f"created pickle file: {pth}")  # 打印成功创建 pickle 文件的消息


# 定义函数 write_legacy_file，生成旧版存储文件
def write_legacy_file():
    # 将当前目录添加到系统路径中，以便首先搜索
    sys.path.insert(0, "")

    # 检查命令行参数个数是否在指定范围内，否则退出程序并打印用法信息
    if not 3 <= len(sys.argv) <= 4:
        sys.exit(
            "Specify output directory and storage type: generate_legacy_"
            "storage_files.py <output_dir> <storage_type> "
        )

    output_dir = str(sys.argv[1])  # 获取输出目录参数
    storage_type = str(sys.argv[2])  # 获取存储类型参数

    # 如果输出目录不存在，则创建它
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 根据存储类型参数决定生成哪种类型的存储文件
    if storage_type == "pickle":
        write_legacy_pickles(output_dir=output_dir)  # 生成 pickle 格式的存储文件
    else:
        sys.exit("storage_type must be one of {'pickle'}")  # 存储类型错误，退出程序


# 如果该脚本作为主程序运行，则执行生成旧版存储文件的函数
if __name__ == "__main__":
    write_legacy_file()
```