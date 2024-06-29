# `D:\src\scipysrc\pandas\asv_bench\benchmarks\tslibs\period.py`

```
"""
Period benchmarks that rely only on tslibs. See benchmarks.period for
Period benchmarks that rely on other parts of pandas.
"""

import numpy as np  # 导入NumPy库，用于数值计算

from pandas._libs.tslibs.period import (  # 从pandas中导入期间相关的模块
    Period,  # 导入Period类，用于处理时间段
    periodarr_to_dt64arr,  # 导入将期间数组转换为datetime64数组的函数
)

from pandas.tseries.frequencies import to_offset  # 导入将字符串频率转换为Offset对象的函数

from .tslib import (  # 导入自定义的tslib模块中的内容
    _sizes,  # 导入_sizes变量，用于时间数组的大小设置
    _tzs,  # 导入_tzs变量，用于时区设置
    tzlocal_obj,  # 导入tzlocal_obj对象，用于本地时区设置
)

try:
    from pandas._libs.tslibs.vectorized import dt64arr_to_periodarr  # 尝试导入将datetime64数组转换为期间数组的函数
except ImportError:
    from pandas._libs.tslibs.period import dt64arr_to_periodarr  # 导入将datetime64数组转换为期间数组的函数（备选）

class PeriodProperties:
    params = (  # 参数设置，包括频率和属性列表
        ["M", "min"],  # 频率列表
        [
            "year",  # 年份
            "month",  # 月份
            "day",  # 天数
            "hour",  # 小时
            "minute",  # 分钟
            "second",  # 秒数
            "is_leap_year",  # 是否是闰年
            "quarter",  # 季度
            "qyear",  # 季度年
            "week",  # 周数
            "daysinmonth",  # 当月天数
            "dayofweek",  # 星期几
            "dayofyear",  # 年中的第几天
            "start_time",  # 起始时间
            "end_time",  # 结束时间
        ],
    )
    param_names = ["freq", "attr"]  # 参数名称列表

    def setup(self, freq, attr):  # 初始化方法，设置频率和属性
        self.per = Period("2012-06-01", freq=freq)  # 创建Period对象，设置起始日期和频率

    def time_property(self, freq, attr):  # 时间属性方法，用于获取Period对象的属性
        getattr(self.per, attr)  # 获取Period对象的指定属性值


class PeriodUnaryMethods:
    params = ["M", "min"]  # 参数设置，包括频率列表
    param_names = ["freq"]  # 参数名称列表

    def setup(self, freq):  # 初始化方法，设置频率
        self.per = Period("2012-06-01", freq=freq)  # 创建Period对象，设置起始日期和频率
        if freq == "M":  # 如果频率是月份
            self.default_fmt = "%Y-%m"  # 设置默认格式为年-月
        elif freq == "min":  # 如果频率是分钟
            self.default_fmt = "%Y-%m-%d %H:%M"  # 设置默认格式为年-月-日 时:分

    def time_to_timestamp(self, freq):  # 将Period对象转换为时间戳方法
        self.per.to_timestamp()  # 调用Period对象的to_timestamp方法将其转换为时间戳

    def time_now(self, freq):  # 获取当前时间方法
        self.per.now(freq)  # 调用Period对象的now方法获取当前时间

    def time_asfreq(self, freq):  # 转换为指定频率方法
        self.per.asfreq("Y")  # 调用Period对象的asfreq方法将其转换为年度频率

    def time_str(self, freq):  # 将Period对象转换为字符串方法
        str(self.per)  # 调用Period对象的__str__方法将其转换为字符串

    def time_repr(self, freq):  # 将Period对象转换为表示形式方法
        repr(self.per)  # 调用Period对象的__repr__方法将其转换为表示形式字符串

    def time_strftime_default(self, freq):  # 使用默认格式进行格式化时间方法
        self.per.strftime(None)  # 调用Period对象的strftime方法，使用默认格式进行格式化

    def time_strftime_default_explicit(self, freq):  # 使用显式默认格式进行格式化时间方法
        self.per.strftime(self.default_fmt)  # 调用Period对象的strftime方法，使用显式默认格式进行格式化

    def time_strftime_custom(self, freq):  # 使用自定义格式进行格式化时间方法
        self.per.strftime("%b. %d, %Y was a %A")  # 调用Period对象的strftime方法，使用自定义格式进行格式化


class PeriodConstructor:
    params = [["D"], [True, False]]  # 参数设置，包括频率和是否偏移列表
    param_names = ["freq", "is_offset"]  # 参数名称列表

    def setup(self, freq, is_offset):  # 初始化方法，设置频率和是否偏移
        if is_offset:  # 如果是偏移
            self.freq = to_offset(freq)  # 将频率转换为Offset对象
        else:
            self.freq = freq  # 否则直接使用频率值

    def time_period_constructor(self, freq, is_offset):  # Period对象构造方法
        Period("2012-06-01", freq=freq)  # 创建Period对象，设置起始日期和频率


_freq_ints = [  # 频率整数列表
    1000,  # 年度 - 十一月末
    1011,  # 年度 - 十一月末
    2000,  # 季度 - 十一月末
    2011,  # 季度 - 十一月末
    3000,  # 周度 - 星期六末
    4000,  # 每天
    4006,  # 每周 - 星期六末
    5000,  # 小时
    6000,  # 分钟
    7000,  # 每秒
    8000,  # 每秒
    9000,  # 每秒
    10000,  # 每秒
    11000,  # 每秒
    12000,  # 每秒
]

class TimePeriodArrToDT64Arr:
    params = [  # 参数设置，包括大小和频率整数列表
        _sizes,  # 时间数组的大小设置
        _freq_ints,  # 频率整数列表
    ]
    param_names = ["size", "freq"]  # 参数名称列表

    def setup(self, size, freq):  # 初始化方法，设置大小和频率
        arr = np.arange(10, dtype="i8").repeat(size // 10)  # 创建整数数组，重复填充以满足指定大小
        self.i8values = arr  # 设置实例变量i8values为创建的数组

    def time_periodarray_to_dt64arr(self, size, freq):  # 将期间数组转换为datetime64数组方法
        periodarr_to_dt64arr(self.i8values, freq)  # 调用periodarr_to_dt64arr函数，将期间数组转换为datetime64数组
    # 定义一个参数列表，包含三个参数 _sizes, _freq_ints, _tzs
    params = [
        _sizes,
        _freq_ints,
        _tzs,
    ]
    # 定义参数名列表，分别对应 _sizes -> "size", _freq_ints -> "freq", _tzs -> "tz"
    param_names = ["size", "freq", "tz"]

    # 定义设置方法 setup，接受三个参数 size, freq, tz
    def setup(self, size, freq, tz):
        # 如果 size 为 10**6 并且 tz 为 tzlocal_obj，则抛出未实现错误
        if size == 10**6 and tz is tzlocal_obj:
            # tzlocal 运行速度太慢，因此跳过以保持运行时效率
            raise NotImplementedError

        # 我们选择 2**55 是因为较小的值会导致 npy_datetimestruct_to_datetime
        # 在 NPY_FR_Y 频率下返回 -1，这会人为地减慢函数速度，因为 -1 也是错误标记
        # 创建一个 int64 类型的数组 arr，范围从 2**55 到 2**55 + 10，重复 size // 10 次
        arr = np.arange(2**55, 2**55 + 10, dtype="i8").repeat(size // 10)
        # 将数组 arr 赋值给实例变量 self.i8values
        self.i8values = arr

    # 定义时间转换方法 time_dt64arr_to_periodarr，接受三个参数 size, freq, tz
    def time_dt64arr_to_periodarr(self, size, freq, tz):
        # 调用 dt64arr_to_periodarr 函数，传递 self.i8values, freq, tz 作为参数
        dt64arr_to_periodarr(self.i8values, freq, tz)
```