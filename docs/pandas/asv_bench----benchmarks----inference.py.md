# `D:\src\scipysrc\pandas\asv_bench\benchmarks\inference.py`

```
"""
The functions benchmarked in this file depend _almost_ exclusively on
_libs, but not in a way that is easy to formalize.

If a PR does not change anything in pandas/_libs/ or pandas/core/tools/, then
it is likely that these benchmarks will be unaffected.
"""

import numpy as np  # 导入NumPy库

from pandas import (  # 从Pandas库导入以下模块
    Index,
    NaT,
    Series,
    date_range,
    to_datetime,
    to_numeric,
    to_timedelta,
)

from .pandas_vb_common import lib  # 从相对路径导入pandas_vb_common模块中的lib对象


class ToNumeric:  # 定义ToNumeric类
    def setup(self):  # 定义setup方法
        N = 10000
        self.float = Series(np.random.randn(N))  # 创建包含随机浮点数的Series对象
        self.numstr = self.float.astype("str")  # 将float类型的Series对象转换为字符串类型
        self.str = Series(Index([f"i-{i}" for i in range(N)], dtype=object))  # 创建包含对象索引的Series对象

    def time_from_float(self):  # 定义time_from_float方法
        to_numeric(self.float, errors="coerce")  # 将浮点数转换为数字，处理错误为"coerce"

    def time_from_numeric_str(self):  # 定义time_from_numeric_str方法
        to_numeric(self.numstr, errors="coerce")  # 将字符串数字转换为数字，处理错误为"coerce"

    def time_from_str(self):  # 定义time_from_str方法
        to_numeric(self.str, errors="coerce")  # 将字符串对象索引转换为数字，处理错误为"coerce"


class ToNumericDowncast:  # 定义ToNumericDowncast类
    param_names = ["dtype", "downcast"]  # 参数名称列表
    params = [  # 参数列表
        [
            "string-float",
            "string-int",
            "string-nint",
            "datetime64",
            "int-list",
            "int32",
        ],
        [None, "integer", "signed", "unsigned", "float"],
    ]

    N = 500000  # N的值为500000
    N2 = N // 2  # N2的值为N整除2的结果

    data_dict = {  # 数据字典
        "string-int": ["1"] * N2 + [2] * N2,  # 字符串整数数据
        "string-nint": ["-1"] * N2 + [2] * N2,  # 负数字符串整数数据
        "datetime64": np.repeat(  # 重复生成日期时间数据
            np.array(["1970-01-01", "1970-01-02"], dtype="datetime64[D]"), N
        ),
        "string-float": ["1.1"] * N2 + [2] * N2,  # 字符串浮点数数据
        "int-list": [1] * N2 + [2] * N2,  # 整数列表数据
        "int32": np.repeat(np.int32(1), N),  # 重复生成int32类型数据
    }

    def setup(self, dtype, downcast):  # 定义setup方法，接受dtype和downcast参数
        self.data = self.data_dict[dtype]  # 根据dtype选择相应的数据

    def time_downcast(self, dtype, downcast):  # 定义time_downcast方法，接受dtype和downcast参数
        to_numeric(self.data, downcast=downcast)  # 将数据转换为数字类型，指定downcast方式


class MaybeConvertNumeric:  # 定义MaybeConvertNumeric类
    # maybe_convert_numeric depends _exclusively_ on _libs, could
    #  go in benchmarks/libs.py

    def setup_cache(self):  # 定义setup_cache方法
        N = 10**6  # N的值为1000000
        arr = np.repeat([2**63], N) + np.arange(N).astype("uint64")  # 创建包含uint64类型数据的数组
        data = arr.astype(object)  # 将数组转换为对象类型
        data[1::2] = arr[1::2].astype(str)  # 将部分数组元素转换为字符串类型
        data[-1] = -1  # 修改最后一个元素的值为-1
        return data  # 返回处理后的数据

    def time_convert(self, data):  # 定义time_convert方法，接受data参数
        lib.maybe_convert_numeric(data, set(), coerce_numeric=False)  # 使用lib对象的方法处理数据


class MaybeConvertObjects:  # 定义MaybeConvertObjects类
    # maybe_convert_objects depends _almost_ exclusively on _libs, but
    #  does have some run-time imports from outside of _libs

    def setup(self):  # 定义setup方法
        N = 10**5  # N的值为100000

        data = list(range(N))  # 创建包含N个整数的列表
        data[0] = NaT  # 将第一个元素设置为NaT
        data = np.array(data)  # 将列表转换为NumPy数组
        self.data = data  # 将数据保存到实例变量中

    def time_maybe_convert_objects(self):  # 定义time_maybe_convert_objects方法
        lib.maybe_convert_objects(self.data)  # 使用lib对象的方法处理数据


class ToDatetimeFromIntsFloats:
    # 此处省略该类的定义，因为在提供的代码片段中未定义该类
    # 在设置方法中初始化一个包含秒级时间戳的 Series 对象，范围从1521080307到1521685106，数据类型为int64
    self.ts_sec = Series(range(1521080307, 1521685107), dtype="int64")
    # 初始化一个包含秒级时间戳的 Series 对象，数据类型为uint64
    self.ts_sec_uint = Series(range(1521080307, 1521685107), dtype="uint64")
    # 将秒级时间戳的 Series 对象转换为浮点数类型
    self.ts_sec_float = self.ts_sec.astype("float64")

    # 创建纳秒级时间戳的 Series 对象，通过将秒级时间戳乘以1,000,000得到
    self.ts_nanosec = 1_000_000 * self.ts_sec
    # 创建纳秒级时间戳的 Series 对象，数据类型为uint64
    self.ts_nanosec_uint = 1_000_000 * self.ts_sec_uint
    # 将纳秒级时间戳的 Series 对象转换为浮点数类型
    self.ts_nanosec_float = self.ts_nanosec.astype("float64")

# 定义性能测试函数：用于测试处理纳秒级int64时间戳的速度
def time_nanosec_int64(self):
    to_datetime(self.ts_nanosec, unit="ns")

# 定义性能测试函数：用于测试处理纳秒级uint64时间戳的速度
def time_nanosec_uint64(self):
    to_datetime(self.ts_nanosec_uint, unit="ns")

# 定义性能测试函数：用于测试处理纳秒级float64时间戳的速度
def time_nanosec_float64(self):
    to_datetime(self.ts_nanosec_float, unit="ns")

# 定义性能测试函数：用于测试处理秒级uint64时间戳的速度
def time_sec_uint64(self):
    to_datetime(self.ts_sec_uint, unit="s")

# 定义性能测试函数：用于测试处理秒级int64时间戳的速度
def time_sec_int64(self):
    to_datetime(self.ts_sec, unit="s")

# 定义性能测试函数：用于测试处理秒级float64时间戳的速度
def time_sec_float64(self):
    to_datetime(self.ts_sec_float, unit="s")
class ToDatetimeYYYYMMDD:
    # 定义一个类用于处理日期格式转换为 YYYYMMDD 格式
    def setup(self):
        # 创建一个日期范围从 "2000-01-01" 开始，10000天，每天一次的日期序列
        rng = date_range(start="1/1/2000", periods=10000, freq="D")
        # 将日期序列转换为字符串形式，格式为 "%Y%m%d"，存储在实例变量 stringsD 中
        self.stringsD = Series(rng.strftime("%Y%m%d"))

    # 将字符串格式为 YYYYMMDD 的日期转换为 datetime 对象
    def time_format_YYYYMMDD(self):
        to_datetime(self.stringsD, format="%Y%m%d")


class ToDatetimeCacheSmallCount:
    # 定义一个类用于测试不同缓存和计数条件下的日期转换性能
    params = ([True, False], [50, 500, 5000, 100000])
    param_names = ["cache", "count"]

    # 设置方法，根据不同的缓存和计数条件创建日期范围并转换为字符串
    def setup(self, cache, count):
        rng = date_range(start="1/1/1971", periods=count)
        self.unique_date_strings = rng.strftime("%Y-%m-%d").tolist()

    # 测试方法，根据不同的缓存和计数条件将日期字符串转换为 datetime 对象
    def time_unique_date_strings(self, cache, count):
        to_datetime(self.unique_date_strings, cache=cache)


class ToDatetimeISO8601:
    # 定义一个类用于测试 ISO 8601 格式的日期转换性能
    def setup(self):
        # 创建一个包含 20000 个小时级别的日期序列
        rng = date_range(start="1/1/2000", periods=20000, freq="h")
        # 将日期序列分别格式化为 ISO 8601 标准的字符串，存储在不同的实例变量中
        self.strings = rng.strftime("%Y-%m-%d %H:%M:%S").tolist()
        self.strings_nosep = rng.strftime("%Y%m%d %H:%M:%S").tolist()
        self.strings_tz_space = [
            x.strftime("%Y-%m-%d %H:%M:%S") + " -0800" for x in rng
        ]
        self.strings_zero_tz = [x.strftime("%Y-%m-%d %H:%M:%S") + "Z" for x in rng]

    # 测试方法，将带有分隔符的 ISO 8601 格式的日期字符串转换为 datetime 对象
    def time_iso8601(self):
        to_datetime(self.strings)

    # 测试方法，将不带分隔符的 ISO 8601 格式的日期字符串转换为 datetime 对象
    def time_iso8601_nosep(self):
        to_datetime(self.strings_nosep)

    # 测试方法，根据指定格式将 ISO 8601 格式的日期字符串转换为 datetime 对象
    def time_iso8601_format(self):
        to_datetime(self.strings, format="%Y-%m-%d %H:%M:%S")

    # 测试方法，根据指定格式将不带分隔符的 ISO 8601 格式的日期字符串转换为 datetime 对象
    def time_iso8601_format_no_sep(self):
        to_datetime(self.strings_nosep, format="%Y%m%d %H:%M:%S")

    # 测试方法，处理带有时区的 ISO 8601 格式的日期字符串转换为 datetime 对象
    def time_iso8601_tz_spaceformat(self):
        to_datetime(self.strings_tz_space)

    # 测试方法，处理零时区的 ISO 8601 格式的日期字符串转换为 datetime 对象
    def time_iso8601_infer_zero_tz_fromat(self):
        # GH 41047
        to_datetime(self.strings_zero_tz)


class ToDatetimeNONISO8601:
    # 定义一个类用于测试非 ISO 8601 格式的日期转换性能
    def setup(self):
        N = 10000
        half = N // 2
        ts_string_1 = "March 1, 2018 12:00:00+0400"
        ts_string_2 = "March 1, 2018 12:00:00+0500"
        # 创建两种不同时区偏移量的日期字符串列表
        self.same_offset = [ts_string_1] * N
        self.diff_offset = [ts_string_1] * half + [ts_string_2] * half

    # 测试方法，处理相同时区偏移量的日期字符串列表转换为 datetime 对象
    def time_same_offset(self):
        to_datetime(self.same_offset)

    # 测试方法，处理不同时区偏移量的日期字符串列表转换为 datetime 对象，强制 UTC
    def time_different_offset(self):
        to_datetime(self.diff_offset, utc=True)


class ToDatetimeFormatQuarters:
    # 定义一个类用于测试包含季度信息的日期格式转换性能
    def setup(self):
        # 创建一个包含大量季度信息的字符串序列
        self.s = Series(["2Q2005", "2Q05", "2005Q1", "05Q1"] * 10000)

    # 测试方法，将包含季度信息的日期字符串转换为 datetime 对象
    def time_infer_quarter(self):
        to_datetime(self.s)


class ToDatetimeFormat:
    # 定义一个类用于测试特定格式的日期字符串转换性能
    def setup(self):
        N = 100000
        # 创建一个包含大量特定格式日期字符串的序列
        self.s = Series(["19MAY11", "19MAY11:00:00:00"] * N)
        # 创建一个去掉时间部分的日期字符串序列
        self.s2 = self.s.str.replace(":\\S+$", "", regex=True)

        # 创建具有相同偏移量的日期时间字符串列表
        self.same_offset = ["10/11/2018 00:00:00.045-07:00"] * N
        # 创建具有不同偏移量的日期时间字符串列表
        self.diff_offset = [
            f"10/11/2018 00:00:00.045-0{offset}:00" for offset in range(10)
        ] * (N // 10)

    # 测试方法，将精确格式的日期字符串转换为 datetime 对象
    def time_exact(self):
        to_datetime(self.s2, format="%d%b%y")

    # 测试方法，将非精确格式的日期字符串转换为 datetime 对象
    def time_no_exact(self):
        to_datetime(self.s, format="%d%b%y", exact=False)

    # 测试方法，处理具有相同偏移量的日期时间字符串列表转换为 datetime 对象
    def time_same_offset(self):
        to_datetime(self.same_offset, format="%m/%d/%Y %H:%M:%S.%f%z")
    # 将具有相同偏移量的时间转换为 UTC 时间
    def time_same_offset_to_utc(self):
        # 使用给定的时间字符串和格式将时间转换为 UTC 时间
        to_datetime(self.same_offset, format="%m/%d/%Y %H:%M:%S.%f%z", utc=True)
    
    # 将具有不同偏移量的时间转换为 UTC 时间
    def time_different_offset_to_utc(self):
        # 使用给定的时间字符串和格式将时间转换为 UTC 时间
        to_datetime(self.diff_offset, format="%m/%d/%Y %H:%M:%S.%f%z", utc=True)
# 定义一个缓存时间转换结果的类
class ToDatetimeCache:
    # 参数列表，包括是否使用缓存的选项
    params = [True, False]
    # 参数名称列表
    param_names = ["cache"]

    # 设置方法，初始化数据
    def setup(self, cache):
        # 创建一个包含10000个不重复的数字秒的列表
        N = 10000
        self.unique_numeric_seconds = list(range(N))
        # 创建一个包含10000个重复值为1000的数字秒的列表
        self.dup_numeric_seconds = [1000] * N
        # 创建一个包含10000个重复日期字符串"2000-02-11"的列表
        self.dup_string_dates = ["2000-02-11"] * N
        # 创建一个包含10000个带有时区偏移的日期字符串"2000-02-11 15:00:00-0800"的列表
        self.dup_string_with_tz = ["2000-02-11 15:00:00-0800"] * N

    # 测试函数，用于测试时间转换函数处理不同参数时的性能
    def time_unique_seconds_and_unit(self, cache):
        # 调用时间转换函数处理不重复数字秒的列表，单位为秒，根据缓存选项决定是否缓存结果
        to_datetime(self.unique_numeric_seconds, unit="s", cache=cache)

    # 测试函数，用于测试时间转换函数处理重复数字秒的列表时的性能
    def time_dup_seconds_and_unit(self, cache):
        # 调用时间转换函数处理重复数字秒的列表，单位为秒，根据缓存选项决定是否缓存结果
        to_datetime(self.dup_numeric_seconds, unit="s", cache=cache)

    # 测试函数，用于测试时间转换函数处理重复日期字符串的列表时的性能
    def time_dup_string_dates(self, cache):
        # 调用时间转换函数处理重复日期字符串的列表，根据缓存选项决定是否缓存结果
        to_datetime(self.dup_string_dates, cache=cache)

    # 测试函数，用于测试时间转换函数处理带有格式的重复日期字符串的列表时的性能
    def time_dup_string_dates_and_format(self, cache):
        # 调用时间转换函数处理带有格式的重复日期字符串的列表，根据缓存选项决定是否缓存结果
        to_datetime(self.dup_string_dates, format="%Y-%m-%d", cache=cache)

    # 测试函数，用于测试时间转换函数处理带有时区偏移的日期字符串的列表时的性能
    def time_dup_string_tzoffset_dates(self, cache):
        # 调用时间转换函数处理带有时区偏移的日期字符串的列表，根据缓存选项决定是否缓存结果
        to_datetime(self.dup_string_with_tz, cache=cache)


# 定义一个时间间隔转换的类
class ToTimedelta:
    # 设置方法，初始化数据
    def setup(self):
        # 创建一个包含10000个随机整数的列表，表示秒数
        self.ints = np.random.randint(0, 60, size=10000)
        # 创建一个空列表，用于存储表示天数的字符串
        self.str_days = []
        # 创建一个空列表，用于存储表示秒数的字符串
        self.str_seconds = []
        # 遍历随机整数列表，将每个整数转换为"X days"格式的字符串并添加到列表中
        for i in self.ints:
            self.str_days.append(f"{i} days")
            # 将每个整数转换为"00:00:XX"格式的字符串并添加到列表中
            self.str_seconds.append(f"00:00:{i:02d}")

    # 测试函数，用于测试时间间隔转换函数处理整数列表时的性能
    def time_convert_int(self):
        # 调用时间间隔转换函数处理整数列表，单位为秒
        to_timedelta(self.ints, unit="s")

    # 测试函数，用于测试时间间隔转换函数处理表示天数的字符串列表时的性能
    def time_convert_string_days(self):
        # 调用时间间隔转换函数处理表示天数的字符串列表
        to_timedelta(self.str_days)

    # 测试函数，用于测试时间间隔转换函数处理表示秒数的字符串列表时的性能
    def time_convert_string_seconds(self):
        # 调用时间间隔转换函数处理表示秒数的字符串列表
        to_timedelta(self.str_seconds)


# 定义一个处理时间间隔转换异常的类
class ToTimedeltaErrors:
    # 设置方法，初始化数据
    def setup(self):
        # 创建一个包含10000个随机整数的列表，表示天数
        ints = np.random.randint(0, 60, size=10000)
        # 将每个随机整数转换为"X days"格式的字符串，最后一个元素修改为"apple"
        self.arr = [f"{i} days" for i in ints]
        self.arr[-1] = "apple"

    # 测试函数，用于测试时间间隔转换函数处理异常情况时的性能
    def time_convert(self):
        # 调用时间间隔转换函数处理包含异常数据的列表，将无法转换的值设为NaN
        to_timedelta(self.arr, errors="coerce")


# 导入设置函数，用于处理测试环境的配置
from .pandas_vb_common import setup  # noqa: F401 isort:skip
```