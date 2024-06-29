# `D:\src\scipysrc\pandas\pandas\tests\indexes\timedeltas\methods\test_shift.py`

```
import pytest  # 导入 pytest 模块

from pandas.errors import NullFrequencyError  # 从 pandas.errors 模块导入 NullFrequencyError 异常类

import pandas as pd  # 导入 pandas 库并使用 pd 别名
from pandas import TimedeltaIndex  # 从 pandas 导入 TimedeltaIndex 类
import pandas._testing as tm  # 导入 pandas._testing 模块并使用 tm 别名


class TestTimedeltaIndexShift:
    # -------------------------------------------------------------
    # TimedeltaIndex.shift is used by __add__/__sub__
    
    def test_tdi_shift_empty(self):
        # GH#9903
        idx = TimedeltaIndex([], name="xxx")  # 创建一个空的 TimedeltaIndex 对象，指定名称为 "xxx"
        tm.assert_index_equal(idx.shift(0, freq="h"), idx)  # 断言调用 shift 方法后，结果与原始索引相等
        tm.assert_index_equal(idx.shift(3, freq="h"), idx)  # 断言调用 shift 方法后，结果与原始索引相等

    def test_tdi_shift_hours(self):
        # GH#9903
        idx = TimedeltaIndex(["5 hours", "6 hours", "9 hours"], name="xxx")  # 创建一个包含时间差的 TimedeltaIndex 对象，指定名称为 "xxx"
        tm.assert_index_equal(idx.shift(0, freq="h"), idx)  # 断言调用 shift 方法后，结果与原始索引相等
        exp = TimedeltaIndex(["8 hours", "9 hours", "12 hours"], name="xxx")  # 期望的结果为包含增加小时后的 TimedeltaIndex 对象
        tm.assert_index_equal(idx.shift(3, freq="h"), exp)  # 断言调用 shift 方法后，结果与期望的索引相等
        exp = TimedeltaIndex(["2 hours", "3 hours", "6 hours"], name="xxx")  # 期望的结果为包含减少小时后的 TimedeltaIndex 对象
        tm.assert_index_equal(idx.shift(-3, freq="h"), exp)  # 断言调用 shift 方法后，结果与期望的索引相等

    def test_tdi_shift_minutes(self):
        # GH#9903
        idx = TimedeltaIndex(["5 hours", "6 hours", "9 hours"], name="xxx")  # 创建一个包含时间差的 TimedeltaIndex 对象，指定名称为 "xxx"
        tm.assert_index_equal(idx.shift(0, freq="min"), idx)  # 断言调用 shift 方法后，结果与原始索引相等
        exp = TimedeltaIndex(["05:03:00", "06:03:00", "9:03:00"], name="xxx")  # 期望的结果为包含增加分钟后的 TimedeltaIndex 对象
        tm.assert_index_equal(idx.shift(3, freq="min"), exp)  # 断言调用 shift 方法后，结果与期望的索引相等
        exp = TimedeltaIndex(["04:57:00", "05:57:00", "8:57:00"], name="xxx")  # 期望的结果为包含减少分钟后的 TimedeltaIndex 对象
        tm.assert_index_equal(idx.shift(-3, freq="min"), exp)  # 断言调用 shift 方法后，结果与期望的索引相等

    def test_tdi_shift_int(self):
        # GH#8083
        tdi = pd.to_timedelta(range(5), unit="D")  # 创建一个以天为单位的时间差序列
        trange = tdi._with_freq("infer") + pd.offsets.Hour(1)  # 添加一小时的时间偏移量
        result = trange.shift(1)  # 将时间序列向后移动一步
        expected = TimedeltaIndex(  # 期望的结果为包含增加一天并加上一小时后的 TimedeltaIndex 对象
            [
                "1 days 01:00:00",
                "2 days 01:00:00",
                "3 days 01:00:00",
                "4 days 01:00:00",
                "5 days 01:00:00",
            ],
            freq="D",  # 结果的频率为天
        )
        tm.assert_index_equal(result, expected)  # 断言调用 shift 方法后，结果与期望的索引相等

    def test_tdi_shift_nonstandard_freq(self):
        # GH#8083
        tdi = pd.to_timedelta(range(5), unit="D")  # 创建一个以天为单位的时间差序列
        trange = tdi._with_freq("infer") + pd.offsets.Hour(1)  # 添加一小时的时间偏移量
        result = trange.shift(3, freq="2D 1s")  # 将时间序列向后移动三步，使用非标准频率 "2D 1s"
        expected = TimedeltaIndex(  # 期望的结果为增加六天一小时三秒后的 TimedeltaIndex 对象
            [
                "6 days 01:00:03",
                "7 days 01:00:03",
                "8 days 01:00:03",
                "9 days 01:00:03",
                "10 days 01:00:03",
            ],
            freq="D",  # 结果的频率为天
        )
        tm.assert_index_equal(result, expected)  # 断言调用 shift 方法后，结果与期望的索引相等

    def test_shift_no_freq(self):
        # GH#19147
        tdi = TimedeltaIndex(["1 days 01:00:00", "2 days 01:00:00"], freq=None)  # 创建一个没有频率的 TimedeltaIndex 对象
        with pytest.raises(NullFrequencyError, match="Cannot shift with no freq"):  # 使用 pytest 断言捕获 NullFrequencyError 异常，断言异常信息包含 "Cannot shift with no freq"
            tdi.shift(2)  # 尝试对没有频率的时间序列进行移动操作，预期会抛出异常
```