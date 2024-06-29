# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimelike_\test_equals.py`

```
"""
Tests shared for DatetimeIndex/TimedeltaIndex/PeriodIndex
"""

from datetime import (
    datetime,             # 导入 datetime 模块中的 datetime 类
    timedelta,            # 导入 datetime 模块中的 timedelta 类
)

import numpy as np       # 导入 NumPy 库，并使用 np 别名
import pytest             # 导入 pytest 测试框架

import pandas as pd       # 导入 Pandas 库，并使用 pd 别名
from pandas import (      # 从 Pandas 中导入多个类和函数
    CategoricalIndex,     # 类别索引
    DatetimeIndex,        # 日期时间索引
    Index,                # 标准索引
    PeriodIndex,          # 时间段索引
    TimedeltaIndex,       # 时间差索引
    date_range,           # 生成日期范围
    period_range,         # 生成时间段范围
    timedelta_range,      # 生成时间差范围
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块

class EqualsTests:
    def test_not_equals_numeric(self, index):
        assert not index.equals(Index(index.asi8))  # 断言索引与其 64 位整数表示不相等
        assert not index.equals(Index(index.asi8.astype("u8")))  # 断言索引与其 64 位无符号整数表示不相等
        assert not index.equals(Index(index.asi8).astype("f8"))  # 断言索引与其 64 位浮点数表示不相等

    def test_equals(self, index):
        assert index.equals(index)  # 断言索引与自身相等
        assert index.equals(index.astype(object))  # 断言索引与转换为对象类型后的自身相等
        assert index.equals(CategoricalIndex(index))  # 断言索引与其转换为类别索引后相等
        assert index.equals(CategoricalIndex(index.astype(object)))  # 断言索引与转换为类别索引后的对象类型相等

    def test_not_equals_non_arraylike(self, index):
        assert not index.equals(list(index))  # 断言索引与其列表形式不相等

    def test_not_equals_strings(self, index):
        other = Index([str(x) for x in index], dtype=object)  # 创建包含索引字符串表示的对象索引
        assert not index.equals(other)  # 断言索引与另一个对象索引不相等
        assert not index.equals(CategoricalIndex(other))  # 断言索引与其转换为类别索引后不相等

    def test_not_equals_misc_strs(self, index):
        other = Index(list("abc"))  # 创建一个包含字符串列表的对象索引
        assert not index.equals(other)  # 断言索引与另一个对象索引不相等


class TestPeriodIndexEquals(EqualsTests):
    @pytest.fixture
    def index(self):
        return period_range("2013-01-01", periods=5, freq="D")  # 返回一个日期时间范围的时间段索引

    # TODO: de-duplicate with other test_equals2 methods
    @pytest.mark.parametrize("freq", ["D", "M"])
    def test_equals2(self, freq):
        # GH#13107
        idx = PeriodIndex(["2011-01-01", "2011-01-02", "NaT"], freq=freq)  # 创建指定频率的时间段索引
        assert idx.equals(idx)  # 断言时间段索引与自身相等
        assert idx.equals(idx.copy())  # 断言时间段索引与其副本相等
        assert idx.equals(idx.astype(object))  # 断言时间段索引与转换为对象类型后的自身相等
        assert idx.astype(object).equals(idx)  # 断言转换为对象类型的时间段索引与自身相等
        assert idx.astype(object).equals(idx.astype(object))  # 断言转换为对象类型的时间段索引与自身相等
        assert not idx.equals(list(idx))  # 断言时间段索引与其列表形式不相等
        assert not idx.equals(pd.Series(idx))  # 断言时间段索引与其 Pandas Series 形式不相等

        idx2 = PeriodIndex(["2011-01-01", "2011-01-02", "NaT"], freq="h")  # 创建指定小时频率的时间段索引
        assert not idx.equals(idx2)  # 断言时间段索引与另一个不同频率的索引不相等
        assert not idx.equals(idx2.copy())  # 断言时间段索引与另一个不同频率的索引副本不相等
        assert not idx.equals(idx2.astype(object))  # 断言时间段索引与另一个不同频率的索引转换为对象类型后不相等
        assert not idx.astype(object).equals(idx2)  # 断言转换为对象类型的时间段索引与另一个不同频率的索引不相等
        assert not idx.equals(list(idx2))  # 断言时间段索引与另一个不同频率的索引列表形式不相等
        assert not idx.equals(pd.Series(idx2))  # 断言时间段索引与另一个不同频率的索引 Pandas Series 形式不相等

        # same internal, different tz
        idx3 = PeriodIndex._simple_new(
            idx._values._simple_new(idx._values.asi8, dtype=pd.PeriodDtype("h"))
        )  # 创建新的时间段索引，内部类型相同但时区不同
        tm.assert_numpy_array_equal(idx.asi8, idx3.asi8)  # 使用测试模块检查两个时间段索引的内部数据是否相等
        assert not idx.equals(idx3)  # 断言时间段索引与另一个相同内部数据但不同时区的索引不相等
        assert not idx.equals(idx3.copy())  # 断言时间段索引与另一个相同内部数据但不同时区的索引副本不相等
        assert not idx.equals(idx3.astype(object))  # 断言时间段索引与另一个相同内部数据但不同时区的索引转换为对象类型后不相等
        assert not idx.astype(object).equals(idx3)  # 断言转换为对象类型的时间段索引与另一个相同内部数据但不同时区的索引不相等
        assert not idx.equals(list(idx3))  # 断言时间段索引与另一个相同内部数据但不同时区的索引列表形式不相等
        assert not idx.equals(pd.Series(idx3))  # 断言时间段索引与另一个相同内部数据但不同时区的索引 Pandas Series 形式不相等


class TestDatetimeIndexEquals(EqualsTests):
    @pytest.fixture
    def index(self):
        return date_range("2013-01-01", periods=5)  # 返回一个日期时间范围的日期时间索引
    # 定义一个测试方法，用于验证 DatetimeIndex 对象的相等性
    def test_equals2(self):
        # GH#13107：这个测试用例的编号或引用
        idx = DatetimeIndex(["2011-01-01", "2011-01-02", "NaT"])
        # 断言自身相等性
        assert idx.equals(idx)
        # 断言与副本的相等性
        assert idx.equals(idx.copy())
        # 断言与转换为 object 类型后的相等性
        assert idx.equals(idx.astype(object))
        # 断言与转换后的对象再转回 DatetimeIndex 后的相等性
        assert idx.astype(object).equals(idx)
        # 断言与两个都转换为 object 类型后的相等性
        assert idx.astype(object).equals(idx.astype(object))
        # 断言与转换为列表后的不相等性
        assert not idx.equals(list(idx))
        # 断言与转换为 Series 后的不相等性
        assert not idx.equals(pd.Series(idx))

        idx2 = DatetimeIndex(["2011-01-01", "2011-01-02", "NaT"], tz="US/Pacific")
        # 断言与具有不同时区的 DatetimeIndex 的不相等性
        assert not idx.equals(idx2)
        # 断言与其副本的不相等性
        assert not idx.equals(idx2.copy())
        # 断言与转换为 object 类型后的不相等性
        assert not idx.equals(idx2.astype(object))
        # 断言与转换后的对象再转回 DatetimeIndex 后的不相等性
        assert not idx.astype(object).equals(idx2)
        # 断言与转换为列表后的不相等性
        assert not idx.equals(list(idx2))
        # 断言与转换为 Series 后的不相等性
        assert not idx.equals(pd.Series(idx2))

        # 创建一个具有相同内部表示但不同时区的 DatetimeIndex 对象
        idx3 = DatetimeIndex(idx.asi8, tz="US/Pacific")
        # 使用 numpy 数组比较断言它们的 asi8 属性相等
        tm.assert_numpy_array_equal(idx.asi8, idx3.asi8)
        # 断言与具有相同内部表示但不同时区的 DatetimeIndex 的不相等性
        assert not idx.equals(idx3)
        # 断言与其副本的不相等性
        assert not idx.equals(idx3.copy())
        # 断言与转换为 object 类型后的不相等性
        assert not idx.equals(idx3.astype(object))
        # 断言与转换后的对象再转回 DatetimeIndex 后的不相等性
        assert not idx.astype(object).equals(idx3)
        # 断言与转换为列表后的不相等性
        assert not idx.equals(list(idx3))
        # 断言与转换为 Series 后的不相等性
        assert not idx.equals(pd.Series(idx3))

        # 检查与 OutOfBounds 对象比较时不引发异常
        oob = Index([datetime(2500, 1, 1)] * 3, dtype=object)
        # 断言与 OutOfBounds 对象的不相等性
        assert not idx.equals(oob)
        assert not idx2.equals(oob)
        assert not idx3.equals(oob)

        # 检查与转换为 np.datetime64 后的 OutOfBounds 对象比较时不引发异常
        oob2 = oob.map(np.datetime64)
        # 断言与 np.datetime64 类型的 OutOfBounds 对象的不相等性
        assert not idx.equals(oob2)
        assert not idx2.equals(oob2)
        assert not idx3.equals(oob2)

    # 使用 pytest 的参数化装饰器，测试不同频率下的日期范围对象是否相等
    @pytest.mark.parametrize("freq", ["B", "C"])
    def test_not_equals_bday(self, freq):
        # 创建一个按照给定频率 freq 的日期范围
        rng = date_range("2009-01-01", "2010-01-01", freq=freq)
        # 断言日期范围对象与其转换为列表后的不相等性
        assert not rng.equals(list(rng))
class TestTimedeltaIndexEquals(EqualsTests):
    @pytest.fixture
    def index(self):
        # 返回一个包含10个时间增量的时间增量范围对象作为测试用例的索引
        return timedelta_range("1 day", periods=10)

    def test_equals2(self):
        # GH#13107: 指示这些断言是为了验证与GitHub问题编号13107相关的功能

        # 创建一个时间增量索引对象idx，包含三个元素："1 days", "2 days", "NaT"
        idx = TimedeltaIndex(["1 days", "2 days", "NaT"])

        # 验证idx与其自身相等
        assert idx.equals(idx)
        # 验证idx与其副本相等
        assert idx.equals(idx.copy())
        # 验证idx与将其转换为对象类型后的对象相等
        assert idx.equals(idx.astype(object))
        # 验证将idx转换为对象类型后的对象与idx相等
        assert idx.astype(object).equals(idx)
        # 验证将idx和其对象类型的副本相等
        assert idx.astype(object).equals(idx.astype(object))
        # 验证idx与其转换为列表后不相等
        assert not idx.equals(list(idx))
        # 验证idx与其转换为Series后不相等
        assert not idx.equals(pd.Series(idx))

        # 创建另一个时间增量索引对象idx2，包含三个元素："2 days", "1 days", "NaT"
        idx2 = TimedeltaIndex(["2 days", "1 days", "NaT"])

        # 验证idx与idx2不相等
        assert not idx.equals(idx2)
        # 验证idx与idx2的副本不相等
        assert not idx.equals(idx2.copy())
        # 验证idx与将idx2转换为对象类型后不相等
        assert not idx.equals(idx2.astype(object))
        # 验证将idx2转换为对象类型后的对象与idx不相等
        assert not idx.astype(object).equals(idx2)
        # 验证将idx2和其对象类型的副本不相等
        assert not idx.astype(object).equals(idx2.astype(object))
        # 验证idx与idx2转换为列表后不相等
        assert not idx.equals(list(idx2))
        # 验证idx与将idx2转换为Series后不相等
        assert not idx.equals(pd.Series(idx2))

        # 创建一个超出实现范围的索引对象oob，包含三个时间增量对象，每个为10^6天
        oob = Index([timedelta(days=10**6)] * 3, dtype=object)
        # 验证idx与oob不相等
        assert not idx.equals(oob)
        # 验证idx2与oob不相等
        assert not idx2.equals(oob)

        # 创建一个与oob相同的索引对象oob2，但元素为np.timedelta64类型
        oob2 = Index([np.timedelta64(x) for x in oob], dtype=object)
        # 验证oob与oob2的所有元素相等
        assert (oob == oob2).all()
        # 验证idx与oob2不相等
        assert not idx.equals(oob2)
        # 验证idx2与oob2不相等
        assert not idx2.equals(oob2)

        # 创建一个将oob转换为np.timedelta64类型的索引对象oob3
        oob3 = oob.map(np.timedelta64)
        # 验证oob3与oob的所有元素相等
        assert (oob3 == oob).all()
        # 验证idx与oob3不相等
        assert not idx.equals(oob3)
        # 验证idx2与oob3不相等
        assert not idx2.equals(oob3)
```