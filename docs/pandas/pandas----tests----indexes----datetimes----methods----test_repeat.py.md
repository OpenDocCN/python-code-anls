# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\methods\test_repeat.py`

```
import numpy as np  # 导入 NumPy 库，用于数组操作
import pytest  # 导入 pytest 库，用于单元测试

from pandas import (  # 从 pandas 库中导入以下模块：
    DatetimeIndex,  # 日期时间索引
    Timestamp,      # 时间戳
    date_range,     # 日期范围生成器
)
import pandas._testing as tm  # 导入 pandas 内部测试模块

class TestRepeat:
    def test_repeat_range(self, tz_naive_fixture):
        rng = date_range("1/1/2000", "1/1/2001")  # 创建一个日期范围对象 rng，从 2000 年 1 月 1 日到 2001 年 1 月 1 日

        result = rng.repeat(5)  # 将日期范围 rng 中的每个元素重复 5 次
        assert result.freq is None  # 检查结果对象的频率是否为空
        assert len(result) == 5 * len(rng)  # 检查结果对象的长度是否为原始范围长度的 5 倍

    def test_repeat_range2(self, tz_naive_fixture, unit):
        tz = tz_naive_fixture  # 设置时区
        index = date_range("2001-01-01", periods=2, freq="D", tz=tz, unit=unit)  # 创建一个日期范围对象 index，每日频率，2 天

        exp = DatetimeIndex(  # 创建预期的日期时间索引对象 exp
            ["2001-01-01", "2001-01-01", "2001-01-02", "2001-01-02"], tz=tz
        ).as_unit(unit)  # 将预期的索引对象转换为指定单位（例如 ns）

        for res in [index.repeat(2), np.repeat(index, 2)]:  # 对 index 进行两次重复操作，分别用 repeat 和 np.repeat
            tm.assert_index_equal(res, exp)  # 使用测试模块验证结果 res 是否与预期 exp 相等
            assert res.freq is None  # 检查结果对象的频率是否为空

    def test_repeat_range3(self, tz_naive_fixture, unit):
        tz = tz_naive_fixture  # 设置时区
        index = date_range("2001-01-01", periods=2, freq="2D", tz=tz, unit=unit)  # 创建一个日期范围对象 index，2 天频率

        exp = DatetimeIndex(  # 创建预期的日期时间索引对象 exp
            ["2001-01-01", "2001-01-01", "2001-01-03", "2001-01-03"], tz=tz
        ).as_unit(unit)  # 将预期的索引对象转换为指定单位（例如 ns）

        for res in [index.repeat(2), np.repeat(index, 2)]:  # 对 index 进行两次重复操作，分别用 repeat 和 np.repeat
            tm.assert_index_equal(res, exp)  # 使用测试模块验证结果 res 是否与预期 exp 相等
            assert res.freq is None  # 检查结果对象的频率是否为空

    def test_repeat_range4(self, tz_naive_fixture, unit):
        tz = tz_naive_fixture  # 设置时区
        index = DatetimeIndex(["2001-01-01", "NaT", "2003-01-01"], tz=tz).as_unit(unit)  # 创建日期时间索引对象 index，包含 NaT

        exp = DatetimeIndex(  # 创建预期的日期时间索引对象 exp
            [
                "2001-01-01", "2001-01-01", "2001-01-01",  # 三个 "2001-01-01"
                "NaT", "NaT", "NaT",  # 三个 NaT
                "2003-01-01", "2003-01-01", "2003-01-01",  # 三个 "2003-01-01"
            ],
            tz=tz,
        ).as_unit(unit)  # 将预期的索引对象转换为指定单位（例如 ns）

        for res in [index.repeat(3), np.repeat(index, 3)]:  # 对 index 进行三次重复操作，分别用 repeat 和 np.repeat
            tm.assert_index_equal(res, exp)  # 使用测试模块验证结果 res 是否与预期 exp 相等
            assert res.freq is None  # 检查结果对象的频率是否为空

    def test_repeat(self, tz_naive_fixture, unit):
        tz = tz_naive_fixture  # 设置时区
        reps = 2  # 设置重复次数
        msg = "the 'axis' parameter is not supported"  # 错误信息

        rng = date_range(start="2016-01-01", periods=2, freq="30Min", tz=tz, unit=unit)  # 创建一个日期范围对象 rng，30 分钟频率

        expected_rng = DatetimeIndex(  # 创建预期的日期时间索引对象 expected_rng
            [
                Timestamp("2016-01-01 00:00:00", tz=tz),
                Timestamp("2016-01-01 00:00:00", tz=tz),
                Timestamp("2016-01-01 00:30:00", tz=tz),
                Timestamp("2016-01-01 00:30:00", tz=tz),
            ]
        ).as_unit(unit)  # 将预期的索引对象转换为指定单位（例如 ns）

        res = rng.repeat(reps)  # 将日期范围 rng 中的每个元素重复 reps 次
        tm.assert_index_equal(res, expected_rng)  # 使用测试模块验证结果 res 是否与预期 expected_rng 相等
        assert res.freq is None  # 检查结果对象的频率是否为空

        tm.assert_index_equal(np.repeat(rng, reps), expected_rng)  # 使用 NumPy 对 rng 进行 reps 次重复操作，验证结果是否与预期 expected_rng 相等
        with pytest.raises(ValueError, match=msg):  # 使用 pytest 测试是否会引发 ValueError 异常，异常信息为 msg
            np.repeat(rng, reps, axis=1)  # 尝试使用 NumPy 在轴向 1 上对 rng 进行 reps 次重复操作
```