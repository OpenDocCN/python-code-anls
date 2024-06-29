# `D:\src\scipysrc\pandas\pandas\tests\scalar\timestamp\test_comparisons.py`

```
from datetime import (
    datetime,            # 导入 datetime 模块中的 datetime 类
    timedelta,           # 导入 datetime 模块中的 timedelta 类
)
import operator         # 导入 operator 模块，用于比较操作

import numpy as np     # 导入 NumPy 库，并简称为 np
import pytest           # 导入 Pytest 测试框架

from pandas import Timestamp    # 从 Pandas 库中导入 Timestamp 类
import pandas._testing as tm    # 导入 Pandas 内部测试模块

class TestTimestampComparison:
    def test_compare_non_nano_dt64(self):
        # 在 __richcmp__ 中将 dt64 转换为 Timestamp 时不抛出异常
        dt = np.datetime64("1066-10-14")
        ts = Timestamp(dt)

        assert dt == ts   # 断言 dt 和 ts 相等

    def test_comparison_dt64_ndarray(self):
        ts = Timestamp("2021-01-01")
        ts2 = Timestamp("2019-04-05")
        arr = np.array([[ts.asm8, ts2.asm8]], dtype="M8[ns]")

        result = ts == arr   # 比较 Timestamp ts 与数组 arr 的相等性
        expected = np.array([[True, False]], dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

        result = arr == ts   # 比较数组 arr 与 Timestamp ts 的相等性
        tm.assert_numpy_array_equal(result, expected)

        result = ts != arr   # 比较 Timestamp ts 与数组 arr 的不等性
        tm.assert_numpy_array_equal(result, ~expected)

        result = arr != ts   # 比较数组 arr 与 Timestamp ts 的不等性
        tm.assert_numpy_array_equal(result, ~expected)

        result = ts2 < arr   # 比较 Timestamp ts2 与数组 arr 的大小关系
        tm.assert_numpy_array_equal(result, expected)

        result = arr < ts2   # 比较数组 arr 与 Timestamp ts2 的大小关系
        tm.assert_numpy_array_equal(result, np.array([[False, False]], dtype=bool))

        result = ts2 <= arr   # 比较 Timestamp ts2 与数组 arr 的小于等于关系
        tm.assert_numpy_array_equal(result, np.array([[True, True]], dtype=bool))

        result = arr <= ts2   # 比较数组 arr 与 Timestamp ts2 的小于等于关系
        tm.assert_numpy_array_equal(result, ~expected)

        result = ts >= arr   # 比较 Timestamp ts 与数组 arr 的大于等于关系
        tm.assert_numpy_array_equal(result, np.array([[True, True]], dtype=bool))

        result = arr >= ts   # 比较数组 arr 与 Timestamp ts 的大于等于关系
        tm.assert_numpy_array_equal(result, np.array([[True, False]], dtype=bool))

    @pytest.mark.parametrize("reverse", [True, False])
    def test_comparison_dt64_ndarray_tzaware(self, reverse, comparison_op):
        ts = Timestamp("2021-01-01 00:00:00.00000", tz="UTC")
        arr = np.array([ts.asm8, ts.asm8], dtype="M8[ns]")

        left, right = ts, arr
        if reverse:
            left, right = arr, ts

        if comparison_op is operator.eq:
            expected = np.array([False, False], dtype=bool)
            result = comparison_op(left, right)
            tm.assert_numpy_array_equal(result, expected)
        elif comparison_op is operator.ne:
            expected = np.array([True, True], dtype=bool)
            result = comparison_op(left, right)
            tm.assert_numpy_array_equal(result, expected)
        else:
            msg = "Cannot compare tz-naive and tz-aware timestamps"
            with pytest.raises(TypeError, match=msg):
                comparison_op(left, right)
    def test_comparison_object_array(self):
        # GH#15183
        # 创建 Timestamp 对象，表示特定时刻，带有时区信息
        ts = Timestamp("2011-01-03 00:00:00-0500", tz="US/Eastern")
        # 创建另一个 Timestamp 对象，表示不同的时刻，带有相同的时区信息
        other = Timestamp("2011-01-01 00:00:00-0500", tz="US/Eastern")
        # 创建一个没有时区信息的 Timestamp 对象
        naive = Timestamp("2011-01-01 00:00:00")

        # 创建一个包含 Timestamp 对象的 NumPy 数组，dtype 为 object
        arr = np.array([other, ts], dtype=object)
        # 执行比较操作，比较数组中的每个元素是否等于 ts
        res = arr == ts
        # 期望的比较结果，创建一个布尔类型的 NumPy 数组
        expected = np.array([False, True], dtype=bool)
        # 断言所有比较结果与期望结果相符
        assert (res == expected).all()

        # 二维数组的情况
        arr = np.array([[other, ts], [ts, other]], dtype=object)
        # 执行比较操作，比较数组中的每个元素是否不等于 ts
        res = arr != ts
        # 期望的比较结果，创建一个二维布尔类型的 NumPy 数组
        expected = np.array([[True, False], [False, True]], dtype=bool)
        # 断言比较结果的形状与期望结果相同
        assert res.shape == expected.shape
        # 断言所有比较结果与期望结果相符
        assert (res == expected).all()

        # 时区不匹配的情况
        arr = np.array([naive], dtype=object)
        # 检查是否会引发 TypeError 异常，错误消息应包含特定信息
        msg = "Cannot compare tz-naive and tz-aware timestamps"
        with pytest.raises(TypeError, match=msg):
            arr < ts

    def test_comparison(self):
        # 创建一个时间戳表示特定时刻的整数值
        # 5-18-2012 00:00:00.000 的时间戳
        stamp = 1337299200000000000

        # 创建一个 Timestamp 对象，使用上述时间戳值
        val = Timestamp(stamp)

        # 各种比较操作及相应的断言，验证时间戳对象的比较行为

        assert val == val
        assert not val != val
        assert not val < val
        assert val <= val
        assert not val > val
        assert val >= val

        # 创建另一个 datetime 对象
        other = datetime(2012, 5, 18)
        # 使用 datetime 对象与 Timestamp 对象进行比较

        assert val == other
        assert not val != other
        assert not val < other
        assert val <= other
        assert not val > other
        assert val >= other

        # 创建另一个 Timestamp 对象，与之前的 Timestamp 对象进行比较
        other = Timestamp(stamp + 100)

        assert val != other
        assert val != other
        assert val < other
        assert val <= other
        assert other > val
        assert other >= val

    def test_compare_invalid(self):
        # GH#8058
        # 创建一个 Timestamp 对象，表示特定时刻
        val = Timestamp("20130101 12:01:02")

        # 执行一系列无效比较的断言，验证是否会得到 False 的结果

        assert not val == "foo"
        assert not val == 10.0
        assert not val == 1
        assert not val == []
        assert not val == {"foo": 1}
        assert not val == np.float64(1)
        assert not val == np.int64(1)

        assert val != "foo"
        assert val != 10.0
        assert val != 1
        assert val != []
        assert val != {"foo": 1}
        assert val != np.float64(1)
        assert val != np.int64(1)

    @pytest.mark.parametrize("tz", [None, "US/Pacific"])
    def test_compare_date(self, tz):
        # GH#36131 comparing Timestamp with date object is deprecated

        # 创建一个 Timestamp 对象，表示特定日期和时间，使用给定的时区 tz
        ts = Timestamp("2021-01-01 00:00:00.00000", tz=tz)
        # 将 Timestamp 对象转换为 Python 标准库的 datetime 对象，再获取其日期部分
        dt = ts.to_pydatetime().date()

        # 在版本 2.0 中，禁止比较 datetime.date 对象和 Timestamp 对象，
        # 遵循标准库 datetime 的行为。
        msg = "Cannot compare Timestamp with datetime.date"
        # 对于每一对 (left, right)，执行以下断言和异常测试：
        for left, right in [(ts, dt), (dt, ts)]:
            assert not left == right  # 确保 left 不等于 right
            assert left != right  # 确保 left 不等于 right

            # 使用 pytest 检查以下比较操作是否抛出预期的 TypeError 异常
            with pytest.raises(TypeError, match=msg):
                left < right
            with pytest.raises(TypeError, match=msg):
                left <= right
            with pytest.raises(TypeError, match=msg):
                left > right
            with pytest.raises(TypeError, match=msg):
                left >= right

    def test_cant_compare_tz_naive_w_aware(self, utc_fixture):
        # see GH#1404

        # 创建两个 Timestamp 对象，分别表示两个不同的日期，其中一个具有时区信息
        a = Timestamp("3/12/2012")
        b = Timestamp("3/12/2012", tz=utc_fixture)

        # 错误消息，用于检查异常时显示
        msg = "Cannot compare tz-naive and tz-aware timestamps"
        
        # 对 a 和 b 进行比较操作，并使用断言和异常测试确认预期行为
        assert not a == b  # 确保 a 不等于 b
        assert a != b  # 确保 a 不等于 b
        with pytest.raises(TypeError, match=msg):
            a < b
        with pytest.raises(TypeError, match=msg):
            a <= b
        with pytest.raises(TypeError, match=msg):
            a > b
        with pytest.raises(TypeError, match=msg):
            a >= b

        # 对 b 和 a 进行相同的比较操作，验证其对称性
        assert not b == a  # 确保 b 不等于 a
        assert b != a  # 确保 b 不等于 a
        with pytest.raises(TypeError, match=msg):
            b < a
        with pytest.raises(TypeError, match=msg):
            b <= a
        with pytest.raises(TypeError, match=msg):
            b > a
        with pytest.raises(TypeError, match=msg):
            b >= a

        # 检查 Timestamp 对象与其转换为 datetime 对象后的比较
        assert not a == b.to_pydatetime()  # 确保 a 不等于 b 转换为 datetime 后的结果
        assert not a.to_pydatetime() == b  # 确保 a 转换为 datetime 后不等于 b

    def test_timestamp_compare_scalars(self):
        # case where ndim == 0

        # 创建 numpy 的 datetime64 和 Pandas 的 Timestamp 对象
        lhs = np.datetime64(datetime(2013, 12, 6))
        rhs = Timestamp("now")
        nat = Timestamp("nat")

        # 定义操作符和其对应的反向操作符
        ops = {"gt": "lt", "lt": "gt", "ge": "le", "le": "ge", "eq": "eq", "ne": "ne"}

        # 对每一对操作符进行测试
        for left, right in ops.items():
            left_f = getattr(operator, left)  # 获取左操作符的函数
            right_f = getattr(operator, right)  # 获取右操作符的函数

            # 计算预期结果
            expected = left_f(lhs, rhs)

            # 执行右操作符的函数，验证其结果是否与预期相符
            result = right_f(rhs, lhs)
            assert result == expected

            # 计算另一个预期结果
            expected = left_f(rhs, nat)

            # 执行右操作符的函数，验证其结果是否与预期相符
            result = right_f(nat, rhs)
            assert result == expected
    # 测试时间戳与早期日期时间的比较
    def test_timestamp_compare_with_early_datetime(self):
        # 创建一个时间戳对象，表示"2012-01-01"
        stamp = Timestamp("2012-01-01")

        # 断言时间戳不等于最小日期时间
        assert not stamp == datetime.min
        # 断言时间戳不等于指定日期时间
        assert not stamp == datetime(1600, 1, 1)
        assert not stamp == datetime(2700, 1, 1)
        # 断言时间戳不等于最小日期时间
        assert stamp != datetime.min
        # 断言时间戳不等于指定日期时间
        assert stamp != datetime(1600, 1, 1)
        assert stamp != datetime(2700, 1, 1)
        # 断言时间戳大于指定日期时间
        assert stamp > datetime(1600, 1, 1)
        assert stamp >= datetime(1600, 1, 1)
        # 断言时间戳小于指定日期时间
        assert stamp < datetime(2700, 1, 1)
        assert stamp <= datetime(2700, 1, 1)

        # 创建一个时间戳对象，表示最小日期时间，并转换为 Python datetime 对象
        other = Timestamp.min.to_pydatetime(warn=False)
        # 断言最小日期时间减去一微秒小于最小时间戳
        assert other - timedelta(microseconds=1) < Timestamp.min

    # 测试时间戳与超出范围的 np.datetime64 对象的比较
    def test_timestamp_compare_oob_dt64(self):
        # 创建一个微秒级的 np.timedelta64 对象
        us = np.timedelta64(1, "us")
        # 创建一个 np.datetime64 对象，表示最小日期时间，并转换为微秒级
        other = np.datetime64(Timestamp.min).astype("M8[us]")

        # 断言最小时间戳大于其他 np.datetime64 对象
        assert Timestamp.min > other
        # 注意：numpy 在反向比较时会出错

        # 创建一个 np.datetime64 对象，表示最大日期时间，并转换为微秒级
        other = np.datetime64(Timestamp.max).astype("M8[us]")
        # 断言最大时间戳大于其他 np.datetime64 对象
        assert Timestamp.max > other  # 实际上不是超出范围
        assert other < Timestamp.max

        # 断言最大时间戳小于其他 np.datetime64 对象加上一微秒
        assert Timestamp.max < other + us
        # 注意：numpy 在反向比较时会出错

        # 创建一个日期时间对象
        other = datetime(9999, 9, 9)
        # 断言最小时间戳小于日期时间对象
        assert Timestamp.min < other
        assert other > Timestamp.min
        # 断言最大时间戳小于日期时间对象
        assert Timestamp.max < other
        assert other > Timestamp.max

        other = datetime(1, 1, 1)
        # 断言最大时间戳大于日期时间对象
        assert Timestamp.max > other
        assert other < Timestamp.max
        # 断言最小时间戳大于日期时间对象
        assert Timestamp.min > other
        assert other < Timestamp.min

    # 测试与零维数组的比较
    def test_compare_zerodim_array(self, fixed_now_ts):
        # 创建一个固定的时间戳对象
        ts = fixed_now_ts
        # 创建一个 np.datetime64 对象，表示"2016-01-01"，单位为纳秒
        dt64 = np.datetime64("2016-01-01", "ns")
        # 创建一个包含 np.datetime64 对象的零维数组
        arr = np.array(dt64)
        # 断言数组的维度为零
        assert arr.ndim == 0

        # 比较数组中的值与时间戳对象的大小关系
        result = arr < ts
        assert result is np.bool_(True)
        result = arr > ts
        assert result is np.bool_(False)
def test_rich_comparison_with_unsupported_type():
    # Comparisons with unsupported objects should return NotImplemented
    # (it previously raised TypeError, see #24011)

    # 定义一个类 Inf，用于模拟一个特殊的对象
    class Inf:
        # 小于操作符重载，始终返回 False
        def __lt__(self, o):
            return False

        # 小于等于操作符重载，返回 o 是否为 Inf 的实例
        def __le__(self, o):
            return isinstance(o, Inf)

        # 大于操作符重载，返回 o 是否不为 Inf 的实例
        def __gt__(self, o):
            return not isinstance(o, Inf)

        # 大于等于操作符重载，始终返回 True
        def __ge__(self, o):
            return True

        # 等于操作符重载，返回 other 是否为 Inf 的实例
        def __eq__(self, other) -> bool:
            return isinstance(other, Inf)

    # 创建 Inf 类的实例
    inf = Inf()
    # 创建一个 Timestamp 类的实例，日期为 "2018-11-30"
    timestamp = Timestamp("2018-11-30")

    # 对于每一个组合 (inf, timestamp) 和 (timestamp, inf)，执行以下断言：
    for left, right in [(inf, timestamp), (timestamp, inf)]:
        # 左边大于右边，或者左边小于右边
        assert left > right or left < right
        # 左边大于等于右边，或者左边小于等于右边
        assert left >= right or left <= right
        # 左边不等于右边
        assert not left == right
        # 左边不等于右边
        assert left != right
```