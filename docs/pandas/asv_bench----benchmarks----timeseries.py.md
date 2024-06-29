# `D:\src\scipysrc\pandas\asv_bench\benchmarks\timeseries.py`

```
from datetime import timedelta  # 导入 timedelta 类用于处理时间差

import dateutil  # 导入 dateutil 库用于处理日期时间
import numpy as np  # 导入 NumPy 库

from pandas import (  # 从 pandas 库中导入多个子模块和函数
    DataFrame,  # 数据框类
    Series,  # 系列类
    date_range,  # 日期范围生成函数
    period_range,  # 时期范围生成函数
    timedelta_range,  # 时间差范围生成函数
)

from pandas.tseries.frequencies import infer_freq  # 从 pandas 时间序列频率模块导入 infer_freq 函数

try:
    from pandas.plotting._matplotlib.converter import DatetimeConverter  # 尝试从 pandas 绘图模块中导入日期时间转换器
except ImportError:
    from pandas.tseries.converter import DatetimeConverter  # 如果导入失败，则从 pandas 时间序列模块导入日期时间转换器


class DatetimeIndex:  # 定义日期时间索引类
    params = ["dst", "repeated", "tz_aware", "tz_local", "tz_naive"]  # 参数列表
    param_names = ["index_type"]  # 参数名列表

    def setup(self, index_type):  # 设置方法，根据不同的 index_type 进行初始化
        N = 100000  # 设定数据点数
        dtidxes = {
            "dst": date_range(  # 夏令时日期时间范围
                start="10/29/2000 1:00:00", end="10/29/2000 1:59:59", freq="s"
            ),
            "repeated": date_range(  # 重复日期时间范围
                start="2000", periods=N // 10, freq="s"
            ).repeat(10),
            "tz_aware": date_range(  # 时区感知日期时间范围
                start="2000", periods=N, freq="s", tz="US/Eastern"
            ),
            "tz_local": date_range(  # 本地时区日期时间范围
                start="2000", periods=N, freq="s", tz=dateutil.tz.tzlocal()
            ),
            "tz_naive": date_range(  # 无时区日期时间范围
                start="2000", periods=N, freq="s"
            ),
        }
        self.index = dtidxes[index_type]  # 根据 index_type 选择相应的日期时间索引

    def time_add_timedelta(self, index_type):  # 添加时间差方法
        self.index + timedelta(minutes=2)  # 增加两分钟时间差

    def time_normalize(self, index_type):  # 标准化时间方法
        self.index.normalize()  # 标准化时间索引

    def time_unique(self, index_type):  # 唯一化时间方法
        self.index.unique()  # 获取唯一时间值

    def time_to_time(self, index_type):  # 获取时间方法
        self.index.time  # 获取时间属性

    def time_get(self, index_type):  # 获取时间点方法
        self.index[0]  # 获取第一个时间点

    def time_timeseries_is_month_start(self, index_type):  # 判断是否月初方法
        self.index.is_month_start  # 检查是否是月初

    def time_to_date(self, index_type):  # 转换为日期方法
        self.index.date  # 获取日期属性

    def time_to_pydatetime(self, index_type):  # 转换为 Python datetime 方法
        self.index.to_pydatetime()  # 转换为 Python datetime 对象

    def time_is_dates_only(self, index_type):  # 检查是否仅日期方法
        self.index._is_dates_only  # 检查索引是否仅包含日期


class TzLocalize:  # 时区本地化类
    params = [None, "US/Eastern", "UTC", dateutil.tz.tzutc()]  # 参数列表
    param_names = "tz"  # 参数名

    def setup(self, tz):  # 设置方法
        dst_rng = date_range(  # 夏令时日期时间范围
            start="10/29/2000 1:00:00", end="10/29/2000 1:59:59", freq="s"
        )
        self.index = date_range(  # 标准日期时间范围
            start="10/29/2000", end="10/29/2000 00:59:59", freq="s"
        )
        self.index = self.index.append(dst_rng)  # 添加夏令时日期时间范围
        self.index = self.index.append(dst_rng)  # 再次添加夏令时日期时间范围
        self.index = self.index.append(  # 添加标准日期时间范围
            date_range(start="10/29/2000 2:00:00", end="10/29/2000 3:00:00", freq="s")
        )

    def time_infer_dst(self, tz):  # 推断夏令时方法
        self.index.tz_localize(tz, ambiguous="infer")  # 使用指定时区本地化时间索引


class ResetIndex:  # 重置索引类
    params = [None, "US/Eastern"]  # 参数列表
    param_names = "tz"  # 参数名

    def setup(self, tz):  # 设置方法
        idx = date_range(start="1/1/2000", periods=1000, freq="h", tz=tz)  # 创建具有时区信息的日期时间范围
        self.df = DataFrame(np.random.randn(1000, 2), index=idx)  # 创建随机数据框，使用日期时间索引

    def time_reset_datetimeindex(self, tz):  # 重置日期时间索引方法
        self.df.reset_index()  # 重置数据框的索引


class InferFreq:  # 推断频率类
    # This depends mostly on code in _libs/, tseries/, and core.algos.unique
    params = [None, "D", "B"]  # 参数列表
    param_names = ["freq"]  # 参数名
    # 设置函数，用于初始化日期索引
    def setup(self, freq):
        # 如果频率参数为 None，则创建每日频率的日期范围索引，周期为10000天
        if freq is None:
            self.idx = date_range(start="1/1/1700", freq="D", periods=10000)
            # 将创建的日期索引对象的频率设为 None
            self.idx._data._freq = None
        else:
            # 根据给定的频率创建日期范围索引，周期为10000天
            self.idx = date_range(start="1/1/1700", freq=freq, periods=10000)

    # 时间推断频率函数，推断当前日期索引的频率
    def time_infer_freq(self, freq):
        infer_freq(self.idx)
class TimeDatetimeConverter:
    # 时间日期转换器类

    def setup(self):
        # 设置方法，初始化时间范围为从"1/1/2000"开始，频率为每分钟，共N=100000个时间点
        N = 100000
        self.rng = date_range(start="1/1/2000", periods=N, freq="min")

    def time_convert(self):
        # 执行时间转换，调用DatetimeConverter类的convert方法，传入时间范围self.rng
        DatetimeConverter.convert(self.rng, None, None)


class Iteration:
    # 迭代器类

    params = [date_range, period_range, timedelta_range]
    param_names = ["time_index"]

    def setup(self, time_index):
        # 设置方法，根据time_index不同类型设置self.idx不同时间范围的索引
        N = 10**6
        if time_index is timedelta_range:
            # 如果time_index是timedelta_range，则设置从0开始，每分钟频率，共N个时间点的索引
            self.idx = time_index(start=0, freq="min", periods=N)
        else:
            # 否则，设置从"20140101"开始，每分钟频率，共N个时间点的索引
            self.idx = time_index(start="20140101", freq="min", periods=N)
        self.exit = 10000

    def time_iter(self, time_index):
        # 迭代时间索引self.idx
        for _ in self.idx:
            pass

    def time_iter_preexit(self, time_index):
        # 在预设退出点前迭代时间索引self.idx
        for i, _ in enumerate(self.idx):
            if i > self.exit:
                break


class ResampleDataFrame:
    # 数据框重新采样类

    params = ["max", "mean", "min"]
    param_names = ["method"]

    def setup(self, method):
        # 设置方法，创建一个时间范围为从"20130101"开始，频率为每50毫秒，共100000个时间点的数据框df
        rng = date_range(start="20130101", periods=100000, freq="50ms")
        df = DataFrame(np.random.randn(100000, 2), index=rng)
        # 根据method选择相应的重新采样方法，保存到self.resample中
        self.resample = getattr(df.resample("1s"), method)

    def time_method(self, method):
        # 测试执行选定的重新采样方法
        self.resample()


class ResampleSeries:
    # 序列重新采样类

    params = (["period", "datetime"], ["5min", "1D"], ["mean", "ohlc"])
    param_names = ["index", "freq", "method"]

    def setup(self, index, freq, method):
        # 设置方法，根据index选择时间范围，创建对应索引的时间序列ts
        indexes = {
            "period": period_range(start="1/1/2000", end="1/1/2001", freq="min"),
            "datetime": date_range(start="1/1/2000", end="1/1/2001", freq="min"),
        }
        idx = indexes[index]
        ts = Series(np.random.randn(len(idx)), index=idx)
        # 根据freq和method进行重新采样，保存到self.resample中
        self.resample = getattr(ts.resample(freq), method)

    def time_resample(self, index, freq, method):
        # 测试执行选定的重新采样方法
        self.resample()


class ResampleDatetetime64:
    # datetime64重新采样类

    # GH 7754

    def setup(self):
        # 设置方法，创建时间范围从"2000-01-01 00:00:00"到"2000-01-01 10:00:00"，频率为每555000微秒的时间序列self.dt_ts
        rng3 = date_range(
            start="2000-01-01 00:00:00", end="2000-01-01 10:00:00", freq="555000us"
        )
        self.dt_ts = Series(5, rng3, dtype="datetime64[ns]")

    def time_resample(self):
        # 测试执行每秒重新采样，并取最后一个值
        self.dt_ts.resample("1s").last()


class AsOf:
    # AsOf类

    params = ["DataFrame", "Series"]
    param_names = ["constructor"]

    def setup(self, constructor):
        # 设置方法，创建时间范围从"1/1/1990"开始，频率为每53秒，长度为N=10000的时间序列self.ts
        N = 10000
        M = 10
        rng = date_range(start="1/1/1990", periods=N, freq="53s")
        data = {
            "DataFrame": DataFrame(np.random.randn(N, M)),
            "Series": Series(np.random.randn(N)),
        }
        self.ts = data[constructor]
        self.ts.index = rng
        # 复制self.ts，将部分元素设置为NaN，形成self.ts2和self.ts3
        self.ts2 = self.ts.copy()
        self.ts2.iloc[250:5000] = np.nan
        self.ts3 = self.ts.copy()
        self.ts3.iloc[-5000:] = np.nan
        # 创建时间范围从"1/1/1990"开始，频率为每5秒，长度为N*10的时间序列self.dates，并设置self.date和相应的日期范围
        self.dates = date_range(start="1/1/1990", periods=N * 10, freq="5s")
        self.date = self.dates[0]
        self.date_last = self.dates[-1]
        self.date_early = self.date - timedelta(10)

    # test speed of pre-computing NAs.
    # 测试预先计算NaN的速度
    # 使用给定的构造函数对时间戳进行处理
    def time_asof(self, constructor):
        # 使用时间戳对象的asof方法，传入日期列表进行处理
        self.ts.asof(self.dates)

    # 应该和上面的函数大致相同
    def time_asof_nan(self, constructor):
        # 使用时间戳对象的asof方法，传入日期列表进行处理
        self.ts2.asof(self.dates)

    # 测试代码路径在标量索引时的速度
    # 不使用while循环
    def time_asof_single(self, constructor):
        # 使用时间戳对象的asof方法，传入单个日期进行处理
        self.ts.asof(self.date)

    # 测试代码路径在早于起始日期的标量索引时的速度
    # 应该与上面的函数相同
    def time_asof_single_early(self, constructor):
        # 使用时间戳对象的asof方法，传入早期日期进行处理
        self.ts.asof(self.date_early)

    # 测试代码路径在带有长while循环的标量索引时的速度
    # 应该仍然比预先计算所有NA值要快得多
    def time_asof_nan_single(self, constructor):
        # 使用时间戳对象的asof方法，传入最后日期进行处理
        self.ts3.asof(self.date_last)
class SortIndex:
    # 参数列表，表示是否按单调顺序排序，默认为 True
    params = [True, False]
    # 参数名称列表，只有一个参数 "monotonic"
    param_names = ["monotonic"]

    # 设置方法，初始化数据
    def setup(self, monotonic):
        # N 设置为 10^5
        N = 10**5
        # 创建一个时间序列索引，从 "1/1/2000" 开始，每秒一个时间点，共 N 个时间点
        idx = date_range(start="1/1/2000", periods=N, freq="s")
        # 创建一个随机数值序列，使用标准正态分布，索引为 idx
        self.s = Series(np.random.randn(N), index=idx)
        # 如果不按单调顺序排序，则随机打乱 self.s 序列
        if not monotonic:
            self.s = self.s.sample(frac=1)

    # 测试排序索引所需的时间方法
    def time_sort_index(self, monotonic):
        # 对 self.s 进行排序
        self.s.sort_index()

    # 测试获取切片所需的时间方法
    def time_get_slice(self, monotonic):
        # 获取 self.s 的前 10000 个元素的切片
        self.s[:10000]


class Lookup:
    # 初始化方法
    def setup(self):
        # N 设置为 1500000
        N = 1500000
        # 创建一个时间序列索引，从 "1/1/2000" 开始，每秒一个时间点，共 N 个时间点
        rng = date_range(start="1/1/2000", periods=N, freq="s")
        # 创建一个数值全部为 1 的序列，索引为 rng
        self.ts = Series(1, index=rng)
        # 设置查找值为 rng 的中间值
        self.lookup_val = rng[N // 2]

    # 测试查找和清理所需的时间方法
    def time_lookup_and_cleanup(self):
        # 查找 self.ts 中的 self.lookup_val 值
        self.ts[self.lookup_val]
        # 清理 self.ts 的索引
        self.ts.index._cleanup()


class DatetimeAccessor:
    # 参数列表，包含时区参数和 None
    params = [None, "US/Eastern", "UTC", dateutil.tz.tzutc()]
    # 参数名称为 "tz"
    param_names = "tz"

    # 设置方法，初始化数据
    def setup(self, tz):
        # N 设置为 100000
        N = 100000
        # 创建一个时间序列，从 "1/1/2000" 开始，每分钟一个时间点，共 N 个时间点，使用指定时区 tz
        self.series = Series(date_range(start="1/1/2000", periods=N, freq="min", tz=tz))

    # 测试日期时间访问器的时间方法
    def time_dt_accessor(self, tz):
        # 访问 self.series 的 dt 属性
        self.series.dt

    # 测试日期时间访问器的 normalize 方法的时间方法
    def time_dt_accessor_normalize(self, tz):
        # 对 self.series 的日期时间进行标准化处理
        self.series.dt.normalize()

    # 测试日期时间访问器的 month_name 方法的时间方法
    def time_dt_accessor_month_name(self, tz):
        # 获取 self.series 的月份名称
        self.series.dt.month_name()

    # 测试日期时间访问器的 day_name 方法的时间方法
    def time_dt_accessor_day_name(self, tz):
        # 获取 self.series 的星期几名称
        self.series.dt.day_name()

    # 测试日期时间访问器的 time 方法的时间方法
    def time_dt_accessor_time(self, tz):
        # 获取 self.series 的时间部分
        self.series.dt.time

    # 测试日期时间访问器的 date 方法的时间方法
    def time_dt_accessor_date(self, tz):
        # 获取 self.series 的日期部分
        self.series.dt.date

    # 测试日期时间访问器的 year 方法的时间方法
    def time_dt_accessor_year(self, tz):
        # 获取 self.series 的年份
        self.series.dt.year


from .pandas_vb_common import setup  # noqa: F401 isort:skip
```