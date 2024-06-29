# `D:\src\scipysrc\pandas\pandas\tests\indexes\timedeltas\test_timedelta.py`

```
import numpy as np  # 导入 NumPy 库，用于科学计算
import pytest  # 导入 pytest 库，用于单元测试

from pandas import (  # 从 pandas 库中导入以下模块：
    Index,  # 索引对象
    Series,  # 数据序列对象
    Timedelta,  # 时间增量对象
    timedelta_range,  # 时间增量范围生成器
)
import pandas._testing as tm  # 导入 pandas 内部测试模块

class TestTimedeltaIndex:
    def test_misc_coverage(self):
        rng = timedelta_range("1 day", periods=5)  # 创建一个时间增量范围对象，从 "1 day" 开始，包含 5 个时间增量
        result = rng.groupby(rng.days)  # 根据天数对时间增量范围进行分组
        assert isinstance(next(iter(result.values()))[0], Timedelta)  # 断言分组结果的第一个值的类型为 Timedelta 类型对象

    def test_map(self):
        # test_map_dictlike generally tests
        rng = timedelta_range("1 day", periods=10)  # 创建一个时间增量范围对象，从 "1 day" 开始，包含 10 个时间增量

        f = lambda x: x.days  # 定义一个函数 f，用于提取时间增量对象的天数
        result = rng.map(f)  # 将函数 f 应用到时间增量范围中的每个元素
        exp = Index([f(x) for x in rng], dtype=np.int64)  # 创建预期的索引对象，包含时间增量范围中每个元素应用函数 f 后的结果
        tm.assert_index_equal(result, exp)  # 断言计算结果与预期结果相等

    def test_fields(self):
        rng = timedelta_range("1 days, 10:11:12.100123456", periods=2, freq="s")  # 创建一个时间增量范围对象，包含 2 个时间增量，频率为秒

        tm.assert_index_equal(rng.days, Index([1, 1], dtype=np.int64))  # 断言时间增量范围中的天数索引与预期索引相等
        tm.assert_index_equal(
            rng.seconds,
            Index([10 * 3600 + 11 * 60 + 12, 10 * 3600 + 11 * 60 + 13], dtype=np.int32),
        )  # 断言时间增量范围中的秒数索引与预期索引相等
        tm.assert_index_equal(
            rng.microseconds,
            Index([100 * 1000 + 123, 100 * 1000 + 123], dtype=np.int32),
        )  # 断言时间增量范围中的微秒数索引与预期索引相等
        tm.assert_index_equal(rng.nanoseconds, Index([456, 456], dtype=np.int32))  # 断言时间增量范围中的纳秒数索引与预期索引相等

        msg = "'TimedeltaIndex' object has no attribute '{}'"
        with pytest.raises(AttributeError, match=msg.format("hours")):  # 断言访问不存在的属性 "hours" 时会引发 AttributeError 异常
            rng.hours
        with pytest.raises(AttributeError, match=msg.format("minutes")):  # 断言访问不存在的属性 "minutes" 时会引发 AttributeError 异常
            rng.minutes
        with pytest.raises(AttributeError, match=msg.format("milliseconds")):  # 断言访问不存在的属性 "milliseconds" 时会引发 AttributeError 异常
            rng.milliseconds

        # with nat
        s = Series(rng)  # 使用时间增量范围创建一个数据序列对象
        s[1] = np.nan  # 设置序列中索引为 1 的值为 NaN

        tm.assert_series_equal(s.dt.days, Series([1, np.nan], index=[0, 1]))  # 断言序列中时间增量的天数属性与预期结果相等
        tm.assert_series_equal(
            s.dt.seconds, Series([10 * 3600 + 11 * 60 + 12, np.nan], index=[0, 1])
        )  # 断言序列中时间增量的秒数属性与预期结果相等

        # preserve name (GH15589)
        rng.name = "name"  # 设置时间增量范围对象的名称为 "name"
        assert rng.days.name == "name"  # 断言时间增量范围对象的天数属性的名称为 "name"
```