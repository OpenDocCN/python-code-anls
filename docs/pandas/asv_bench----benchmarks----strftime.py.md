# `D:\src\scipysrc\pandas\asv_bench\benchmarks\strftime.py`

```
import numpy as np  # 导入NumPy库，用于数值计算

import pandas as pd  # 导入Pandas库，用于数据处理
from pandas import offsets  # 导入Pandas的offsets模块，用于时间偏移计算


class DatetimeStrftime:
    timeout = 1500  # 设置超时时间为1500毫秒
    params = [1000, 10000]  # 参数列表包含1000和10000
    param_names = ["nobs"]  # 参数名称为nobs

    def setup(self, nobs):
        d = "2018-11-29"  # 设置日期字符串d为"2018-11-29"
        dt = "2018-11-26 11:18:27.0"  # 设置日期时间字符串dt为"2018-11-26 11:18:27.0"
        self.data = pd.DataFrame(  # 创建一个Pandas DataFrame对象，包含以下列
            {
                "dt": [np.datetime64(dt)] * nobs,  # 列"dt"为nobs个np.datetime64格式的dt值
                "d": [np.datetime64(d)] * nobs,    # 列"d"为nobs个np.datetime64格式的d值
                "r": [np.random.uniform()] * nobs,  # 列"r"为nobs个随机均匀分布的数值
            }
        )

    def time_frame_date_to_str(self, nobs):
        self.data["d"].astype(str)  # 将"d"列的日期数据转换为字符串格式

    def time_frame_date_formatting_default(self, nobs):
        self.data["d"].dt.strftime(date_format=None)  # 使用Pandas将"d"列的日期数据按默认格式转换为字符串

    def time_frame_date_formatting_default_explicit(self, nobs):
        self.data["d"].dt.strftime(date_format="%Y-%m-%d")  # 使用Pandas将"d"列的日期数据按指定格式"%Y-%m-%d"转换为字符串

    def time_frame_date_formatting_custom(self, nobs):
        self.data["d"].dt.strftime(date_format="%Y---%m---%d")  # 使用Pandas将"d"列的日期数据按自定义格式"%Y---%m---%d"转换为字符串

    def time_frame_datetime_to_str(self, nobs):
        self.data["dt"].astype(str)  # 将"dt"列的日期时间数据转换为字符串格式

    def time_frame_datetime_formatting_default(self, nobs):
        self.data["dt"].dt.strftime(date_format=None)  # 使用Pandas将"dt"列的日期时间数据按默认格式转换为字符串

    def time_frame_datetime_formatting_default_explicit_date_only(self, nobs):
        self.data["dt"].dt.strftime(date_format="%Y-%m-%d")  # 使用Pandas将"dt"列的日期时间数据按日期格式"%Y-%m-%d"转换为字符串

    def time_frame_datetime_formatting_default_explicit(self, nobs):
        self.data["dt"].dt.strftime(date_format="%Y-%m-%d %H:%M:%S")  # 使用Pandas将"dt"列的日期时间数据按完整日期时间格式"%Y-%m-%d %H:%M:%S"转换为字符串

    def time_frame_datetime_formatting_default_with_float(self, nobs):
        self.data["dt"].dt.strftime(date_format="%Y-%m-%d %H:%M:%S.%f")  # 使用Pandas将"dt"列的日期时间数据按带微秒的完整日期时间格式"%Y-%m-%d %H:%M:%S.%f"转换为字符串

    def time_frame_datetime_formatting_custom(self, nobs):
        self.data["dt"].dt.strftime(date_format="%Y-%m-%d --- %H:%M:%S")  # 使用Pandas将"dt"列的日期时间数据按自定义格式"%Y-%m-%d --- %H:%M:%S"转换为字符串


class PeriodStrftime:
    timeout = 1500  # 设置超时时间为1500毫秒
    params = ([1000, 10000], ["D", "h"])  # 参数列表包含1000和10000，以及频率列表["D", "h"]
    param_names = ["nobs", "freq"]  # 参数名称为nobs和freq

    def setup(self, nobs, freq):
        self.data = pd.DataFrame(  # 创建一个Pandas DataFrame对象，包含以下列
            {
                "p": pd.period_range(start="2000-01-01", periods=nobs, freq=freq),  # 列"p"为以指定频率freq生成的nobs个时期
                "r": [np.random.uniform()] * nobs,  # 列"r"为nobs个随机均匀分布的数值
            }
        )
        self.data["i"] = self.data["p"]  # 添加一列"i"，值与"p"列相同
        self.data.set_index("i", inplace=True)  # 将"i"列设为索引，修改数据对象
        if freq == "D":
            self.default_fmt = "%Y-%m-%d"  # 如果频率为"D"，设置默认日期格式
        elif freq == "h":
            self.default_fmt = "%Y-%m-%d %H:00"  # 如果频率为"h"，设置默认日期时间格式

    def time_frame_period_to_str(self, nobs, freq):
        self.data["p"].astype(str)  # 将"p"列的时期数据转换为字符串格式

    def time_frame_period_formatting_default(self, nobs, freq):
        self.data["p"].dt.strftime(date_format=None)  # 使用Pandas将"p"列的时期数据按默认格式转换为字符串

    def time_frame_period_formatting_default_explicit(self, nobs, freq):
        self.data["p"].dt.strftime(date_format=self.default_fmt)  # 使用Pandas将"p"列的时期数据按默认或自定义格式self.default_fmt转换为字符串

    def time_frame_period_formatting_custom(self, nobs, freq):
        self.data["p"].dt.strftime(date_format="%Y-%m-%d --- %H:%M:%S")  # 使用Pandas将"p"列的时期数据按自定义格式"%Y-%m-%d --- %H:%M:%S"转换为字符串

    def time_frame_period_formatting_iso8601_strftime_Z(self, nobs, freq):
        self.data["p"].dt.strftime(date_format="%Y-%m-%dT%H:%M:%SZ")  # 使用Pandas将"p"列的时期数据按ISO 8601格式"%Y-%m-%dT%H:%M:%SZ"转换为字符串
    def time_frame_period_formatting_iso8601_strftime_offset(self, nobs, freq):
        """时间框架期间的 ISO8601 格式化字符串偏移"""
        # 尚未优化，因为 `convert_strftime_format` 不支持 %z
        self.data["p"].dt.strftime(date_format="%Y-%m-%dT%H:%M:%S%z")
class BusinessHourStrftime:
    # 设置超时时间为1500
    timeout = 1500
    # 设置参数列表为[1000, 10000]
    params = [1000, 10000]
    # 设置参数名称列表为["nobs"]
    param_names = ["nobs"]

    # 设置数据初始化函数，创建一个包含多个offsets.BusinessHour()的DataFrame
    def setup(self, nobs):
        self.data = pd.DataFrame(
            {
                "off": [offsets.BusinessHour()] * nobs,
            }
        )

    # 将"off"列中的每个元素转换为字符串并返回
    def time_frame_offset_str(self, nobs):
        self.data["off"].apply(str)

    # 将"off"列中的每个元素转换为它们的 Python 表示形式字符串并返回
    def time_frame_offset_repr(self, nobs):
        self.data["off"].apply(repr)
```