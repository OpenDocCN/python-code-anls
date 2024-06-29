# `D:\src\scipysrc\pandas\asv_bench\benchmarks\tslibs\fields.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算

from pandas._libs.tslibs.fields import (  # 从 pandas 库中导入时间序列字段相关的函数
    get_date_field,  # 获取日期字段的值
    get_start_end_field,  # 获取起止时间字段的值
    get_timedelta_field,  # 获取时间差字段的值
)

from .tslib import _sizes  # 从当前包的 tslib 模块导入 _sizes 变量


class TimeGetTimedeltaField:  # 定义一个类 TimeGetTimedeltaField，用于时间差字段的性能测试
    params = [  # 定义参数组合列表
        _sizes,  # 使用 _sizes 变量作为大小参数
        ["seconds", "microseconds", "nanoseconds"],  # 时间差单位参数列表
    ]
    param_names = ["size", "field"]  # 定义参数名称

    def setup(self, size, field):  # 设置测试环境
        arr = np.random.randint(0, 10, size=size, dtype="i8")  # 创建随机整数数组
        self.i8data = arr  # 将数组保存在实例变量中，用于后续测试
        arr = np.random.randint(-86400 * 1_000_000_000, 0, size=size, dtype="i8")  # 创建负数随机整数数组
        self.i8data_negative = arr  # 将负数数组保存在实例变量中，用于后续测试

    def time_get_timedelta_field(self, size, field):  # 定义测试函数，测试获取时间差字段的性能
        get_timedelta_field(self.i8data, field)  # 调用 get_timedelta_field 函数获取时间差字段值

    def time_get_timedelta_field_negative_td(self, size, field):  # 定义测试函数，测试获取负数时间差字段的性能
        get_timedelta_field(self.i8data_negative, field)  # 调用 get_timedelta_field 函数获取负数时间差字段值


class TimeGetDateField:  # 定义一个类 TimeGetDateField，用于日期字段的性能测试
    params = [  # 定义参数组合列表
        _sizes,  # 使用 _sizes 变量作为大小参数
        [
            "Y",  # 年份
            "M",  # 月份
            "D",  # 天数
            "h",  # 小时
            "m",  # 分钟
            "s",  # 秒数
            "us",  # 微秒数
            "ns",  # 纳秒数
            "doy",  # 年内的天数
            "dow",  # 周内的天数
            "woy",  # 年内的周数
            "q",   # 季度
            "dim",  # 月份的天数
            "is_leap_year",  # 是否闰年
        ],
    ]
    param_names = ["size", "field"]  # 定义参数名称

    def setup(self, size, field):  # 设置测试环境
        arr = np.random.randint(0, 10, size=size, dtype="i8")  # 创建随机整数数组
        self.i8data = arr  # 将数组保存在实例变量中，用于后续测试

    def time_get_date_field(self, size, field):  # 定义测试函数，测试获取日期字段的性能
        get_date_field(self.i8data, field)  # 调用 get_date_field 函数获取日期字段值


class TimeGetStartEndField:  # 定义一个类 TimeGetStartEndField，用于起止时间字段的性能测试
    params = [  # 定义参数组合列表
        _sizes,  # 使用 _sizes 变量作为大小参数
        ["start", "end"],  # 起止时间标识参数列表
        ["month", "quarter", "year"],  # 时间周期参数列表
        ["B", None, "QS"],  # 频率字符串参数列表
        [12, 3, 5],  # 月份关键字参数列表
    ]
    param_names = ["size", "side", "period", "freqstr", "month_kw"]  # 定义参数名称

    def setup(self, size, side, period, freqstr, month_kw):  # 设置测试环境
        arr = np.random.randint(0, 10, size=size, dtype="i8")  # 创建随机整数数组
        self.i8data = arr  # 将数组保存在实例变量中，用于后续测试

        self.attrname = f"is_{period}_{side}"  # 根据时间周期和起止时间标识生成属性名称字符串

    def time_get_start_end_field(self, size, side, period, freqstr, month_kw):  # 定义测试函数，测试获取起止时间字段的性能
        get_start_end_field(self.i8data, self.attrname, freqstr, month_kw=month_kw)  # 调用 get_start_end_field 函数获取起止时间字段值


from ..pandas_vb_common import setup  # 从上级包中导入 pandas_vb_common 模块的 setup 函数
```