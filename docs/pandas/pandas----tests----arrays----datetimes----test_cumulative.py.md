# `D:\src\scipysrc\pandas\pandas\tests\arrays\datetimes\test_cumulative.py`

```
import pytest  # 导入 pytest 库

import pandas._testing as tm  # 导入 pandas 测试模块，命名为 tm
from pandas.core.arrays import DatetimeArray  # 导入 DatetimeArray 类


class TestAccumulator:
    def test_accumulators_freq(self):
        # GH#50297
        # 创建包含日期时间字符串的 DatetimeArray 对象，指定日期时间格式为纳秒级
        arr = DatetimeArray._from_sequence(
            [
                "2000-01-01",
                "2000-01-02",
                "2000-01-03",
            ],
            dtype="M8[ns]",
        )._with_freq("infer")
        # 对 DatetimeArray 对象应用累积函数 cummin
        result = arr._accumulate("cummin")
        # 创建预期结果 DatetimeArray 对象，所有值为 "2000-01-01"
        expected = DatetimeArray._from_sequence(["2000-01-01"] * 3, dtype="M8[ns]")
        # 断言结果与预期结果相等
        tm.assert_datetime_array_equal(result, expected)

        # 对 DatetimeArray 对象应用累积函数 cummax
        result = arr._accumulate("cummax")
        # 创建预期结果 DatetimeArray 对象，值分别为 "2000-01-01", "2000-01-02", "2000-01-03"
        expected = DatetimeArray._from_sequence(
            [
                "2000-01-01",
                "2000-01-02",
                "2000-01-03",
            ],
            dtype="M8[ns]",
        )
        # 断言结果与预期结果相等
        tm.assert_datetime_array_equal(result, expected)

    @pytest.mark.parametrize("func", ["cumsum", "cumprod"])
    def test_accumulators_disallowed(self, func):
        # GH#50297
        # 创建包含日期时间字符串的 DatetimeArray 对象，指定日期时间格式为纳秒级
        arr = DatetimeArray._from_sequence(
            [
                "2000-01-01",
                "2000-01-02",
            ],
            dtype="M8[ns]",
        )._with_freq("infer")
        # 使用 pytest 框架断言应该抛出 TypeError 异常，异常信息包含 func 的值
        with pytest.raises(TypeError, match=f"Accumulation {func}"):
            # 对 DatetimeArray 对象应用累积函数 func
            arr._accumulate(func)
```