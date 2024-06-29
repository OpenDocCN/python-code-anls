# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_to_timestamp.py`

```
# 从 datetime 模块中导入 timedelta 类
from datetime import timedelta

# 导入 numpy 库并使用别名 np
import numpy as np

# 导入 pytest 库
import pytest

# 从 pandas 库中导入多个模块
from pandas import (
    DataFrame,
    DatetimeIndex,
    PeriodIndex,
    Series,
    Timedelta,
    date_range,
    period_range,
    to_datetime,
)

# 导入 pandas 内部测试模块，并使用别名 tm
import pandas._testing as tm


# 定义一个函数 _get_with_delta，根据给定的 delta 和 freq 参数生成日期范围
def _get_with_delta(delta, freq="YE-DEC"):
    return date_range(
        to_datetime("1/1/2001") + delta,  # 计算起始日期
        to_datetime("12/31/2009") + delta,  # 计算结束日期
        freq=freq,  # 使用指定的频率
    )


# 定义一个测试类 TestToTimestamp
class TestToTimestamp:
    
    # 定义测试方法 test_to_timestamp，接受 frame_or_series 参数
    def test_to_timestamp(self, frame_or_series):
        K = 5  # 设置 K 的值为 5
        index = period_range(freq="Y", start="1/1/2001", end="12/1/2009")  # 生成一个周期索引对象
        obj = DataFrame(  # 创建 DataFrame 对象
            np.random.default_rng(2).standard_normal((len(index), K)),  # 生成随机数填充的 DataFrame
            index=index,  # 指定索引为之前生成的周期索引
            columns=["A", "B", "C", "D", "E"],  # 指定列名
        )
        obj["mix"] = "a"  # 添加一个名为 'mix' 的列，每行值为 'a'
        obj = tm.get_obj(obj, frame_or_series)  # 调用测试模块的 get_obj 方法，转换 DataFrame 或 Series

        exp_index = date_range("1/1/2001", end="12/31/2009", freq="YE-DEC")  # 生成期望的日期范围
        exp_index = exp_index + Timedelta(1, "D") - Timedelta(1, "ns")  # 对日期范围进行微调
        result = obj.to_timestamp("D", "end")  # 将对象转换为时间戳，以每天结束时间为基准
        tm.assert_index_equal(result.index, exp_index)  # 断言索引相等
        tm.assert_numpy_array_equal(result.values, obj.values)  # 断言值数组相等
        if frame_or_series is Series:
            assert result.name == "A"  # 如果对象为 Series，则断言结果的名称为 "A"

        exp_index = date_range("1/1/2001", end="1/1/2009", freq="YS-JAN")  # 生成另一组期望的日期范围
        result = obj.to_timestamp("D", "start")  # 将对象转换为时间戳，以每天开始时间为基准
        tm.assert_index_equal(result.index, exp_index)  # 断言索引相等

        result = obj.to_timestamp(how="start")  # 将对象转换为时间戳，以每天开始时间为基准（使用关键字参数指定）
        tm.assert_index_equal(result.index, exp_index)  # 断言索引相等

        delta = timedelta(hours=23)  # 定义一个 timedelta 对象，表示 23 小时
        result = obj.to_timestamp("H", "end")  # 将对象转换为时间戳，以每小时结束时间为基准
        exp_index = _get_with_delta(delta)  # 调用 _get_with_delta 函数生成期望的日期范围
        exp_index = exp_index + Timedelta(1, "h") - Timedelta(1, "ns")  # 对日期范围进行微调
        tm.assert_index_equal(result.index, exp_index)  # 断言索引相等

        delta = timedelta(hours=23, minutes=59)  # 定义一个 timedelta 对象，表示 23 小时 59 分钟
        result = obj.to_timestamp("T", "end")  # 将对象转换为时间戳，以每分钟结束时间为基准
        exp_index = _get_with_delta(delta)  # 调用 _get_with_delta 函数生成期望的日期范围
        exp_index = exp_index + Timedelta(1, "m") - Timedelta(1, "ns")  # 对日期范围进行微调
        tm.assert_index_equal(result.index, exp_index)  # 断言索引相等

        result = obj.to_timestamp("S", "end")  # 将对象转换为时间戳，以每秒结束时间为基准
        delta = timedelta(hours=23, minutes=59, seconds=59)  # 定义一个 timedelta 对象，表示 23 小时 59 分钟 59 秒
        exp_index = _get_with_delta(delta)  # 调用 _get_with_delta 函数生成期望的日期范围
        exp_index = exp_index + Timedelta(1, "s") - Timedelta(1, "ns")  # 对日期范围进行微调
        tm.assert_index_equal(result.index, exp_index)  # 断言索引相等
    # 定义一个测试方法，用于测试 DataFrame 的时间戳转换功能
    def test_to_timestamp_columns(self):
        # 设定变量 K 为 5
        K = 5
        # 生成一个时间索引，频率为年，从 "1/1/2001" 到 "12/1/2009"
        index = period_range(freq="Y", start="1/1/2001", end="12/1/2009")
        # 创建一个 DataFrame，数据为随机标准正态分布，行索引为 index，列为 ["A", "B", "C", "D", "E"]
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(index), K)),
            index=index,
            columns=["A", "B", "C", "D", "E"],
        )
        # 在 DataFrame 中添加一个名为 "mix" 的新列，所有行赋值为 "a"
        df["mix"] = "a"

        # 将 DataFrame 转置
        df = df.T

        # 生成预期的时间索引，频率为每年最后一天，从 "1/1/2001" 到 "12/31/2009"
        exp_index = date_range("1/1/2001", end="12/31/2009", freq="YE-DEC")
        # 调整时间索引，使其精确到一天，然后微调纳秒级别
        exp_index = exp_index + Timedelta(1, "D") - Timedelta(1, "ns")
        # 执行 DataFrame 的时间戳转换，按列处理，结束时间为单位
        result = df.to_timestamp("D", "end", axis=1)
        # 断言结果的列索引与预期一致
        tm.assert_index_equal(result.columns, exp_index)
        # 断言结果的数值与原始 DataFrame 的数值一致
        tm.assert_numpy_array_equal(result.values, df.values)

        # 生成预期的时间索引，频率为每年第一天，从 "1/1/2001" 到 "1/1/2009"
        exp_index = date_range("1/1/2001", end="1/1/2009", freq="YS-JAN")
        # 执行 DataFrame 的时间戳转换，按列处理，开始时间为单位
        result = df.to_timestamp("D", "start", axis=1)
        # 断言结果的列索引与预期一致
        tm.assert_index_equal(result.columns, exp_index)

        # 设定时间增量为 23 小时
        delta = timedelta(hours=23)
        # 执行 DataFrame 的时间戳转换，按列处理，结束时间为单位，每小时
        result = df.to_timestamp("H", "end", axis=1)
        # 生成预期的时间索引，根据时间增量微调
        exp_index = _get_with_delta(delta)
        exp_index = exp_index + Timedelta(1, "h") - Timedelta(1, "ns")
        # 断言结果的列索引与预期一致
        tm.assert_index_equal(result.columns, exp_index)

        # 设定时间增量为 23 小时 59 分钟
        delta = timedelta(hours=23, minutes=59)
        # 执行 DataFrame 的时间戳转换，按列处理，结束时间为单位，每分钟
        result = df.to_timestamp("min", "end", axis=1)
        # 生成预期的时间索引，根据时间增量微调
        exp_index = _get_with_delta(delta)
        exp_index = exp_index + Timedelta(1, "m") - Timedelta(1, "ns")
        # 断言结果的列索引与预期一致
        tm.assert_index_equal(result.columns, exp_index)

        # 执行 DataFrame 的时间戳转换，按列处理，结束时间为单位，每秒钟
        result = df.to_timestamp("S", "end", axis=1)
        # 设定时间增量为 23 小时 59 分钟 59 秒
        delta = timedelta(hours=23, minutes=59, seconds=59)
        # 生成预期的时间索引，根据时间增量微调
        exp_index = _get_with_delta(delta)
        exp_index = exp_index + Timedelta(1, "s") - Timedelta(1, "ns")
        # 断言结果的列索引与预期一致
        tm.assert_index_equal(result.columns, exp_index)

        # 执行 DataFrame 的时间戳转换，按列处理，每 5 分钟
        result1 = df.to_timestamp("5min", axis=1)
        # 执行 DataFrame 的时间戳转换，按列处理，每分钟
        result2 = df.to_timestamp("min", axis=1)
        # 生成预期的时间范围，从 "2001-01-01" 到 "2009-01-01"，频率为每年第一天
        expected = date_range("2001-01-01", "2009-01-01", freq="YS")
        # 断言结果的列索引类型为 DatetimeIndex
        assert isinstance(result1.columns, DatetimeIndex)
        assert isinstance(result2.columns, DatetimeIndex)
        # 断言结果的数值与预期一致（转换为整数后比较）
        tm.assert_numpy_array_equal(result1.columns.asi8, expected.asi8)
        tm.assert_numpy_array_equal(result2.columns.asi8, expected.asi8)
        # 断言结果的频率字符串为 "YS-JAN"
        assert result1.columns.freqstr == "YS-JAN"
        assert result2.columns.freqstr == "YS-JAN"

    # 定义一个测试方法，用于测试 DataFrame 时间戳转换的无效轴
    def test_to_timestamp_invalid_axis(self):
        # 生成一个时间索引，频率为年，从 "1/1/2001" 到 "12/1/2009"
        index = period_range(freq="Y", start="1/1/2001", end="12/1/2009")
        # 创建一个 DataFrame，数据为随机标准正态分布，行索引为 index，5 列
        obj = DataFrame(
            np.random.default_rng(2).standard_normal((len(index), 5)), index=index
        )

        # 执行无效轴测试，期望引发 ValueError 异常，匹配错误消息 "axis"
        with pytest.raises(ValueError, match="axis"):
            obj.to_timestamp(axis=2)
    # 定义一个测试方法，用于将给定的时间序列或数据帧转换为时间戳的测试
    def test_to_timestamp_hourly(self, frame_or_series):
        # 创建一个小时频率的时间范围索引，从"2001年1月1日"到"2001年1月2日"
        index = period_range(freq="h", start="1/1/2001", end="1/2/2001")
        # 创建一个包含索引的 Series 对象，每个值为 1，名称为 "foo"
        obj = Series(1, index=index, name="foo")
        # 如果传入的 frame_or_series 不是 Series 类型，则将 obj 转换为数据帧
        if frame_or_series is not Series:
            obj = obj.to_frame()

        # 期望的索引范围，每小时从"2001年1月1日 00:59:59"到"2001年1月2日 00:59:59"
        exp_index = date_range("1/1/2001 00:59:59", end="1/2/2001 00:59:59", freq="h")
        # 将 obj 转换为时间戳，并设定时间戳为每小时结束时刻
        result = obj.to_timestamp(how="end")
        # 调整期望索引，使其精确到纳秒
        exp_index = exp_index + Timedelta(1, "s") - Timedelta(1, "ns")
        # 断言结果的索引与期望索引相等
        tm.assert_index_equal(result.index, exp_index)
        # 如果 frame_or_series 是 Series 类型，则断言结果的名称为 "foo"
        if frame_or_series is Series:
            assert result.name == "foo"

    # 定义一个测试方法，用于测试将索引转换为时间戳时是否引发异常
    def test_to_timestamp_raises(self, index, frame_or_series):
        # 使用给定的索引和数据类型创建一个 Series 或数据帧对象
        obj = frame_or_series(index=index, dtype=object)

        # 如果索引不是 PeriodIndex 类型，则生成一个特定的错误消息
        if not isinstance(index, PeriodIndex):
            msg = f"unsupported Type {type(index).__name__}"
            # 使用 pytest 断言来检查调用 obj.to_timestamp() 是否会引发 TypeError，并匹配特定的错误消息
            with pytest.raises(TypeError, match=msg):
                obj.to_timestamp()
```