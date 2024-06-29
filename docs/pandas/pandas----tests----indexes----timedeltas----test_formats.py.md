# `D:\src\scipysrc\pandas\pandas\tests\indexes\timedeltas\test_formats.py`

```
import pytest  # 导入 pytest 测试框架

import pandas as pd  # 导入 pandas 库并使用 pd 别名
from pandas import (  # 从 pandas 中导入以下模块
    Series,  # 导入 Series 类
    TimedeltaIndex,  # 导入 TimedeltaIndex 类
)


class TestTimedeltaIndexRendering:
    def test_repr_round_days_non_nano(self):
        # GH#55405
        # 测试需求：确保在非纳秒精度下得到 "1 days" 而不是 "1 days 00:00:00"
        tdi = TimedeltaIndex(["1 days"], freq="D").as_unit("s")  # 创建 TimedeltaIndex 对象 tdi，精度为秒
        result = repr(tdi)  # 获取 tdi 的字符串表示形式
        expected = "TimedeltaIndex(['1 days'], dtype='timedelta64[s]', freq='D')"  # 预期的字符串表示形式
        assert result == expected  # 断言结果与预期相符

        result2 = repr(Series(tdi))  # 使用 tdi 创建 Series 对象并获取其字符串表示形式
        expected2 = "0   1 days\ndtype: timedelta64[s]"  # 预期的 Series 对象字符串表示形式
        assert result2 == expected2  # 断言结果与预期相符

    @pytest.mark.parametrize("method", ["__repr__", "__str__"])
    def test_representation(self, method):
        idx1 = TimedeltaIndex([], freq="D")  # 创建空的 TimedeltaIndex 对象 idx1
        idx2 = TimedeltaIndex(["1 days"], freq="D")  # 创建包含 "1 days" 的 TimedeltaIndex 对象 idx2
        idx3 = TimedeltaIndex(["1 days", "2 days"], freq="D")  # 创建包含 "1 days" 和 "2 days" 的 TimedeltaIndex 对象 idx3
        idx4 = TimedeltaIndex(["1 days", "2 days", "3 days"], freq="D")  # 创建包含 "1 days", "2 days" 和 "3 days" 的 TimedeltaIndex 对象 idx4
        idx5 = TimedeltaIndex(["1 days 00:00:01", "2 days", "3 days"])  # 创建包含不同格式的 TimedeltaIndex 对象 idx5

        exp1 = "TimedeltaIndex([], dtype='timedelta64[ns]', freq='D')"  # 预期的空 TimedeltaIndex 对象字符串表示形式

        exp2 = "TimedeltaIndex(['1 days'], dtype='timedelta64[ns]', freq='D')"  # 预期包含 "1 days" 的 TimedeltaIndex 对象字符串表示形式

        exp3 = "TimedeltaIndex(['1 days', '2 days'], dtype='timedelta64[ns]', freq='D')"  # 预期包含 "1 days" 和 "2 days" 的 TimedeltaIndex 对象字符串表示形式

        exp4 = (
            "TimedeltaIndex(['1 days', '2 days', '3 days'], "
            "dtype='timedelta64[ns]', freq='D')"  # 预期包含 "1 days", "2 days" 和 "3 days" 的 TimedeltaIndex 对象字符串表示形式
        )

        exp5 = (
            "TimedeltaIndex(['1 days 00:00:01', '2 days 00:00:00', "
            "'3 days 00:00:00'], dtype='timedelta64[ns]', freq=None)"  # 预期包含不同格式的 TimedeltaIndex 对象字符串表示形式
        )

        with pd.option_context("display.width", 300):  # 设置 pandas 显示宽度为 300
            for idx, expected in zip(
                [idx1, idx2, idx3, idx4, idx5], [exp1, exp2, exp3, exp4, exp5]
            ):
                result = getattr(idx, method)()  # 调用指定的方法获取对象的字符串表示形式
                assert result == expected  # 断言结果与预期相符

    # TODO: this is a Series.__repr__ test
    def test_representation_to_series(self):
        idx1 = TimedeltaIndex([], freq="D")  # 创建空的 TimedeltaIndex 对象 idx1
        idx2 = TimedeltaIndex(["1 days"], freq="D")  # 创建包含 "1 days" 的 TimedeltaIndex 对象 idx2
        idx3 = TimedeltaIndex(["1 days", "2 days"], freq="D")  # 创建包含 "1 days" 和 "2 days" 的 TimedeltaIndex 对象 idx3
        idx4 = TimedeltaIndex(["1 days", "2 days", "3 days"], freq="D")  # 创建包含 "1 days", "2 days" 和 "3 days" 的 TimedeltaIndex 对象 idx4
        idx5 = TimedeltaIndex(["1 days 00:00:01", "2 days", "3 days"])  # 创建包含不同格式的 TimedeltaIndex 对象 idx5

        exp1 = """Series([], dtype: timedelta64[ns])"""  # 预期的空 Series 对象字符串表示形式

        exp2 = "0   1 days\ndtype: timedelta64[ns]"  # 预期包含 "1 days" 的 Series 对象字符串表示形式

        exp3 = "0   1 days\n1   2 days\ndtype: timedelta64[ns]"  # 预期包含 "1 days" 和 "2 days" 的 Series 对象字符串表示形式

        exp4 = "0   1 days\n1   2 days\n2   3 days\ndtype: timedelta64[ns]"  # 预期包含 "1 days", "2 days" 和 "3 days" 的 Series 对象字符串表示形式

        exp5 = (
            "0   1 days 00:00:01\n"
            "1   2 days 00:00:00\n"
            "2   3 days 00:00:00\n"
            "dtype: timedelta64[ns]"  # 预期包含不同格式的 Series 对象字符串表示形式
        )

        with pd.option_context("display.width", 300):  # 设置 pandas 显示宽度为 300
            for idx, expected in zip(
                [idx1, idx2, idx3, idx4, idx5], [exp1, exp2, exp3, exp4, exp5]
            ):
                result = repr(Series(idx))  # 使用 TimedeltaIndex 对象创建 Series 对象并获取其字符串表示形式
                assert result == expected  # 断言结果与预期相符
    # 定义一个测试方法，用于测试 TimedeltaIndex 类的 _summary() 方法
    def test_summary(self):
        # 创建一个空的 TimedeltaIndex 对象，频率为每天，没有任何条目
        idx1 = TimedeltaIndex([], freq="D")
        # 创建一个包含单个时间增量的 TimedeltaIndex 对象，频率为每天
        idx2 = TimedeltaIndex(["1 days"], freq="D")
        # 创建一个包含两个时间增量的 TimedeltaIndex 对象，频率为每天
        idx3 = TimedeltaIndex(["1 days", "2 days"], freq="D")
        # 创建一个包含三个时间增量的 TimedeltaIndex 对象，频率为每天
        idx4 = TimedeltaIndex(["1 days", "2 days", "3 days"], freq="D")
        # 创建一个包含三个时间增量的 TimedeltaIndex 对象，没有指定频率
        idx5 = TimedeltaIndex(["1 days 00:00:01", "2 days", "3 days"])

        # 预期的输出结果，描述了 TimedeltaIndex 对象的内容和频率
        exp1 = "TimedeltaIndex: 0 entries\nFreq: D"
        exp2 = "TimedeltaIndex: 1 entries, 1 days to 1 days\nFreq: D"
        exp3 = "TimedeltaIndex: 2 entries, 1 days to 2 days\nFreq: D"
        exp4 = "TimedeltaIndex: 3 entries, 1 days to 3 days\nFreq: D"
        exp5 = "TimedeltaIndex: 3 entries, 1 days 00:00:01 to 3 days 00:00:00"

        # 对每个 TimedeltaIndex 对象和其对应的预期输出结果进行迭代测试
        for idx, expected in zip(
            [idx1, idx2, idx3, idx4, idx5], [exp1, exp2, exp3, exp4, exp5]
        ):
            # 调用 TimedeltaIndex 对象的 _summary() 方法，获取实际输出结果
            result = idx._summary()
            # 断言实际输出结果与预期输出结果相等
            assert result == expected
```