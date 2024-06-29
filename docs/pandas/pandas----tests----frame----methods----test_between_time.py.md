# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_between_time.py`

```
# 导入需要的模块和类
from datetime import (
    datetime,
    time,
)

import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入Pytest库，用于单元测试

from pandas._libs.tslibs import timezones  # 导入时间区域相关的模块
import pandas.util._test_decorators as td  # 导入测试装饰器相关的模块

from pandas import (
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm  # 导入Pandas测试工具相关的模块


class TestBetweenTime:
    @td.skip_if_not_us_locale
    def test_between_time_formats(self, frame_or_series):
        # GH#11818
        # 创建一个时间范围，每5分钟一个数据点
        rng = date_range("1/1/2000", "1/5/2000", freq="5min")
        # 创建一个随机数填充的DataFrame，索引为时间范围rng
        ts = DataFrame(
            np.random.default_rng(2).standard_normal((len(rng), 2)), index=rng
        )
        # 根据输入的frame_or_series获取对象
        ts = tm.get_obj(ts, frame_or_series)

        # 定义多种时间格式的字符串列表
        strings = [
            ("2:00", "2:30"),
            ("0200", "0230"),
            ("2:00am", "2:30am"),
            ("0200am", "0230am"),
            ("2:00:00", "2:30:00"),
            ("020000", "023000"),
            ("2:00:00am", "2:30:00am"),
            ("020000am", "023000am"),
        ]
        expected_length = 28  # 预期的结果长度

        # 遍历每种时间格式的字符串对，检验过滤结果的长度是否符合预期
        for time_string in strings:
            assert len(ts.between_time(*time_string)) == expected_length

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_localized_between_time(self, tzstr, frame_or_series):
        # 获取时区对象
        tz = timezones.maybe_get_tz(tzstr)

        # 创建一个时间范围，每小时一个数据点
        rng = date_range("4/16/2012", "5/1/2012", freq="h")
        # 创建一个随机数填充的Series，索引为时间范围rng
        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
        if frame_or_series is DataFrame:
            ts = ts.to_frame()

        # 将Series对象本地化为指定时区
        ts_local = ts.tz_localize(tzstr)

        t1, t2 = time(10, 0), time(11, 0)  # 定义开始和结束时间
        # 对本地化后的时间序列进行时间范围过滤
        result = ts_local.between_time(t1, t2)
        # 获取未本地化的时间序列在指定时间范围内的预期结果
        expected = ts.between_time(t1, t2).tz_localize(tzstr)
        tm.assert_equal(result, expected)  # 断言结果是否相等
        assert timezones.tz_compare(result.index.tz, tz)  # 断言结果的时区与预期时区相同

    def test_between_time_types(self, frame_or_series):
        # GH11818
        # 创建一个时间范围，每5分钟一个数据点
        rng = date_range("1/1/2000", "1/5/2000", freq="5min")
        # 创建一个包含单列的DataFrame，索引为时间范围rng
        obj = DataFrame({"A": 0}, index=rng)
        # 根据输入的frame_or_series获取对象
        obj = tm.get_obj(obj, frame_or_series)

        # 准备捕获的错误消息正则表达式
        msg = r"Cannot convert arg \[datetime\.datetime\(2010, 1, 2, 1, 0\)\] to a time"
        # 使用Pytest断言检测是否抛出特定的值错误，并且错误消息与预期相匹配
        with pytest.raises(ValueError, match=msg):
            obj.between_time(datetime(2010, 1, 2, 1), datetime(2010, 1, 2, 5))
    # 定义一个测试方法，用于测试时间范围过滤函数 `between_time`
    def test_between_time(self, inclusive_endpoints_fixture, frame_or_series):
        # 创建一个时间范围，频率为每5分钟
        rng = date_range("1/1/2000", "1/5/2000", freq="5min")
        # 创建一个DataFrame，填充随机正态分布的数据，索引为时间范围
        ts = DataFrame(
            np.random.default_rng(2).standard_normal((len(rng), 2)), index=rng
        )
        # 将DataFrame转换为测试对象，根据输入选择DataFrame或Series
        ts = tm.get_obj(ts, frame_or_series)

        # 定义起始时间和结束时间
        stime = time(0, 0)
        etime = time(1, 0)
        # 获取是否包含端点的设置
        inclusive = inclusive_endpoints_fixture

        # 使用 `between_time` 函数过滤时间范围内的数据
        filtered = ts.between_time(stime, etime, inclusive=inclusive)
        # 计算预期的过滤后的长度
        exp_len = 13 * 4 + 1

        # 根据端点包含的设置，调整预期长度
        if inclusive in ["right", "neither"]:
            exp_len -= 5
        if inclusive in ["left", "neither"]:
            exp_len -= 4

        # 断言过滤后的长度与预期长度相等
        assert len(filtered) == exp_len

        # 遍历过滤后的索引，进行进一步的时间验证
        for rs in filtered.index:
            t = rs.time()
            if inclusive in ["left", "both"]:
                assert t >= stime
            else:
                assert t > stime

            if inclusive in ["right", "both"]:
                assert t <= etime
            else:
                assert t < etime

        # 对比调用 "00:00" 到 "01:00" 时间范围和指定范围起始时间和结束时间的过滤结果
        result = ts.between_time("00:00", "01:00")
        expected = ts.between_time(stime, etime)
        tm.assert_equal(result, expected)

        # 跨越午夜的测试情况
        rng = date_range("1/1/2000", "1/5/2000", freq="5min")
        ts = DataFrame(
            np.random.default_rng(2).standard_normal((len(rng), 2)), index=rng
        )
        ts = tm.get_obj(ts, frame_or_series)
        stime = time(22, 0)
        etime = time(9, 0)

        # 使用 `between_time` 函数过滤跨越午夜的时间范围内的数据
        filtered = ts.between_time(stime, etime, inclusive=inclusive)
        # 计算预期的过滤后的长度
        exp_len = (12 * 11 + 1) * 4 + 1
        if inclusive in ["right", "neither"]:
            exp_len -= 4
        if inclusive in ["left", "neither"]:
            exp_len -= 4

        # 断言过滤后的长度与预期长度相等
        assert len(filtered) == exp_len

        # 遍历过滤后的索引，进行进一步的时间验证
        for rs in filtered.index:
            t = rs.time()
            if inclusive in ["left", "both"]:
                assert (t >= stime) or (t <= etime)
            else:
                assert (t > stime) or (t <= etime)

            if inclusive in ["right", "both"]:
                assert (t <= etime) or (t >= stime)
            else:
                assert (t < etime) or (t >= stime)

    # 定义一个测试方法，测试在非DatetimeIndex索引下调用 `between_time` 函数是否会引发TypeError异常
    def test_between_time_raises(self, frame_or_series):
        # 创建一个普通的DataFrame对象
        obj = DataFrame([[1, 2, 3], [4, 5, 6]])
        # 将DataFrame转换为测试对象，根据输入选择DataFrame或Series
        obj = tm.get_obj(obj, frame_or_series)

        # 定义期望的错误消息
        msg = "Index must be DatetimeIndex"
        # 使用pytest的断言，验证调用 `between_time` 函数是否会抛出TypeError，并检查错误消息
        with pytest.raises(TypeError, match=msg):  # index is not a DatetimeIndex
            obj.between_time(start_time="00:00", end_time="12:00")
    def test_between_time_axis(self, frame_or_series):
        # GH#8839
        # 创建一个日期范围，从"1/1/2000"开始，共100个时间点，每隔10分钟一个时间点
        rng = date_range("1/1/2000", periods=100, freq="10min")
        # 创建一个 Series，包含随机标准正态分布数据，索引为上面创建的日期范围
        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
        # 如果传入的 frame_or_series 是 DataFrame 类型，则将 ts 转换为 DataFrame
        if frame_or_series is DataFrame:
            ts = ts.to_frame()

        # 设置起始时间和结束时间
        stime, etime = ("08:00:00", "09:00:00")
        # 预期的长度为 7
        expected_length = 7

        # 断言选取起始时间和结束时间之间的数据长度是否等于预期长度
        assert len(ts.between_time(stime, etime)) == expected_length
        # 断言在指定轴上选取起始时间和结束时间之间的数据长度是否等于预期长度
        assert len(ts.between_time(stime, etime, axis=0)) == expected_length
        # 准备错误消息，用于检查在对象类型为 ts 的维度上是否存在名为 ts.ndim 的轴
        msg = f"No axis named {ts.ndim} for object type {type(ts).__name__}"
        # 使用 pytest 检查调用 between_time 方法时是否会引发 ValueError 异常，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            ts.between_time(stime, etime, axis=ts.ndim)

    def test_between_time_axis_aliases(self, axis):
        # GH#8839
        # 创建一个日期范围，从"1/1/2000"开始，共100个时间点，每隔10分钟一个时间点
        rng = date_range("1/1/2000", periods=100, freq="10min")
        # 创建一个 DataFrame，包含随机标准正态分布数据，行和列的长度为日期范围长度
        ts = DataFrame(np.random.default_rng(2).standard_normal((len(rng), len(rng))))
        # 设置起始时间和结束时间
        stime, etime = ("08:00:00", "09:00:00")
        # 预期的长度为 7
        exp_len = 7

        # 如果 axis 是 "index" 或者 0，则将 DataFrame 的索引设置为 rng，并执行断言
        if axis in ["index", 0]:
            ts.index = rng
            # 断言选取起始时间和结束时间之间的数据长度是否等于预期长度
            assert len(ts.between_time(stime, etime)) == exp_len
            # 断言在指定轴上选取起始时间和结束时间之间的数据长度是否等于预期长度
            assert len(ts.between_time(stime, etime, axis=0)) == exp_len

        # 如果 axis 是 "columns" 或者 1，则将 DataFrame 的列设置为 rng，并执行断言
        if axis in ["columns", 1]:
            ts.columns = rng
            # 选择在指定轴上选取起始时间和结束时间之间的数据，然后断言其长度是否等于预期长度
            selected = ts.between_time(stime, etime, axis=1).columns
            assert len(selected) == exp_len

    def test_between_time_axis_raises(self, axis):
        # issue 8839
        # 创建一个日期范围，从"1/1/2000"开始，共100个时间点，每隔10分钟一个时间点
        rng = date_range("1/1/2000", periods=100, freq="10min")
        # 创建一个与 rng 长度相同的范围数组
        mask = np.arange(0, len(rng))
        # 创建一个 DataFrame，包含随机标准正态分布数据，行和列的长度为 rng 长度
        rand_data = np.random.default_rng(2).standard_normal((len(rng), len(rng)))
        ts = DataFrame(rand_data, index=rng, columns=rng)
        # 设置起始时间和结束时间
        stime, etime = ("08:00:00", "09:00:00")

        # 准备错误消息，用于检查索引必须是 DatetimeIndex 类型
        msg = "Index must be DatetimeIndex"
        # 如果 axis 是 "columns" 或者 1，则将 DataFrame 的索引设置为 mask，并执行断言
        if axis in ["columns", 1]:
            ts.index = mask
            # 使用 pytest 检查调用 between_time 方法时是否会引发 TypeError 异常，并匹配错误消息
            with pytest.raises(TypeError, match=msg):
                ts.between_time(stime, etime)
            with pytest.raises(TypeError, match=msg):
                ts.between_time(stime, etime, axis=0)

        # 如果 axis 是 "index" 或者 0，则将 DataFrame 的列设置为 mask，并执行断言
        if axis in ["index", 0]:
            ts.columns = mask
            # 使用 pytest 检查调用 between_time 方法时是否会引发 TypeError 异常，并匹配错误消息
            with pytest.raises(TypeError, match=msg):
                ts.between_time(stime, etime, axis=1)

    def test_between_time_datetimeindex(self):
        # 创建一个日期范围，从"2012-01-01"开始，到"2012-01-05"结束，每隔30分钟一个时间点
        index = date_range("2012-01-01", "2012-01-05", freq="30min")
        # 创建一个 DataFrame，包含随机标准正态分布数据，行数为日期范围长度，列数为5
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(index), 5)), index=index
        )
        # 创建一个时间片段，从 13:00:00 到 14:00:00
        bkey = slice(time(13, 0, 0), time(14, 0, 0))
        # 预期选取的行索引位置
        binds = [26, 27, 28, 74, 75, 76, 122, 123, 124, 170, 171, 172]

        # 使用 between_time 方法选取指定时间段的数据，并与预期结果进行比较
        result = df.between_time(bkey.start, bkey.stop)
        expected = df.loc[bkey]
        expected2 = df.iloc[binds]
        # 使用 assert_frame_equal 方法比较 result 和 expected，保证它们相等
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result, expected2)
        # 断言 result 的长度是否为 12
        assert len(result) == 12
    def test_between_time_incorrect_arg_inclusive(self):
        # 定义测试函数，用于验证 `between_time` 方法的错误参数 inclusive
        # GH40245 是这个测试用例的标识符或引用
        rng = date_range("1/1/2000", "1/5/2000", freq="5min")
        # 创建一个日期范围，从 "1/1/2000" 到 "1/5/2000"，频率为每5分钟一次
        ts = DataFrame(
            np.random.default_rng(2).standard_normal((len(rng), 2)), index=rng
        )
        # 创建一个数据帧，其中包含随机标准正态分布的数据，行索引为 rng

        stime = time(0, 0)
        # 定义开始时间为 00:00（午夜）
        etime = time(1, 0)
        # 定义结束时间为 01:00

        inclusive = "bad_string"
        # 设置 inclusive 参数为错误的字符串 "bad_string"

        msg = "Inclusive has to be either 'both', 'neither', 'left' or 'right'"
        # 定义异常信息消息，指出正确的 inclusive 参数应该是 'both', 'neither', 'left' 或 'right'

        with pytest.raises(ValueError, match=msg):
            # 使用 pytest 的 raises 方法检测是否会抛出 ValueError 异常，并匹配特定的消息
            ts.between_time(stime, etime, inclusive=inclusive)
            # 调用被测试的 between_time 方法，传入定义的参数进行测试
```