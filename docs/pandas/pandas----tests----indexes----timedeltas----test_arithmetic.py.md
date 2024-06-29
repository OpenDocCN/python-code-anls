# `D:\src\scipysrc\pandas\pandas\tests\indexes\timedeltas\test_arithmetic.py`

```
# Arithmetic tests for TimedeltaIndex are generally about the result's `freq` attribute.
# Other cases can be shared in tests.arithmetic.test_timedelta64
# 导入 NumPy 库
import numpy as np

# 从 Pandas 库中导入需要的对象
from pandas import (
    NaT,
    Timedelta,
    timedelta_range,
)
# 导入 Pandas 测试模块
import pandas._testing as tm

# 定义测试类 TestTimedeltaIndexArithmetic
class TestTimedeltaIndexArithmetic:
    # 定义测试函数 test_arithmetic_zero_freq
    def test_arithmetic_zero_freq(self):
        # GH#51575 don't get a .freq with freq.n = 0
        # 创建时间增量索引对象 tdi，步长为 1 纳秒，共计 100 个数据点
        tdi = timedelta_range(0, periods=100, freq="ns")
        # 对 tdi 进行除法操作，结果赋给 result
        result = tdi / 2
        # 断言结果的 freq 属性为 None
        assert result.freq is None
        # 创建预期结果 expected，为 tdi 的前 50 个数据点重复两次组成的索引
        expected = tdi[:50].repeat(2)
        # 断言 result 与 expected 相等
        tm.assert_index_equal(result, expected)

        # 对 tdi 进行整数除法操作，结果赋给 result2
        result2 = tdi // 2
        # 断言 result2 的 freq 属性为 None
        assert result2.freq is None
        # 将 expected2 设为与 expected 相同的索引
        expected2 = expected
        # 断言 result2 与 expected2 相等
        tm.assert_index_equal(result2, expected2)

        # 对 tdi 进行乘法操作，结果赋给 result3
        result3 = tdi * 0
        # 断言 result3 的 freq 属性为 None
        assert result3.freq is None
        # 创建预期结果 expected3，为 tdi 的前 1 个数据点重复 100 次组成的索引
        expected3 = tdi[:1].repeat(100)
        # 断言 result3 与 expected3 相等
        tm.assert_index_equal(result3, expected3)

    # 定义测试函数 test_tdi_division，参数为 index_or_series
    def test_tdi_division(self, index_or_series):
        # doc example

        # 创建标量 scalar，代表 31 天的时间增量
        scalar = Timedelta(days=31)
        # 创建时间增量索引对象 td，包含四个元素：三个 scalar 和一个 NaT（Not a Time）
        td = index_or_series(
            [scalar, scalar, scalar + Timedelta(minutes=5, seconds=3), NaT],
            dtype="m8[ns]",
        )

        # 将 td 中每个时间增量除以 1 天的时间增量，结果赋给 result
        result = td / np.timedelta64(1, "D")
        # 创建预期结果 expected，为对应 td 中每个元素按天进行换算的结果
        expected = index_or_series(
            [31, 31, (31 * 86400 + 5 * 60 + 3) / 86400.0, np.nan]
        )
        # 使用测试模块中的函数断言 result 与 expected 相等
        tm.assert_equal(result, expected)

        # 将 td 中每个时间增量除以 1 秒的时间增量，结果赋给 result
        result = td / np.timedelta64(1, "s")
        # 创建预期结果 expected，为对应 td 中每个元素按秒进行换算的结果
        expected = index_or_series(
            [31 * 86400, 31 * 86400, 31 * 86400 + 5 * 60 + 3, np.nan]
        )
        # 使用测试模块中的函数断言 result 与 expected 相等
        tm.assert_equal(result, expected)
```