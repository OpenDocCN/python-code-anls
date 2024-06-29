# `D:\src\scipysrc\pandas\pandas\tests\indexes\timedeltas\methods\test_insert.py`

```
# 导入 timedelta 类型用于处理时间差
from datetime import timedelta

# 导入 numpy 库，用于科学计算
import numpy as np

# 导入 pytest 库，用于编写和运行测试
import pytest

# 导入 pandas 库
import pandas as pd

# 从 pandas 库中导入特定模块和类
from pandas import (
    Index,              # 索引类
    Timedelta,          # 时间差类
    TimedeltaIndex,     # 时间差索引类
    timedelta_range,    # 时间差范围生成函数
)

# 导入 pandas 内部测试工具模块
import pandas._testing as tm


# 定义测试类 TestTimedeltaIndexInsert
class TestTimedeltaIndexInsert:
    
    # 定义测试方法 test_insert
    def test_insert(self):
        # 创建 TimedeltaIndex 对象 idx，包含三个时间差字符串，并指定名称为 "idx"
        idx = TimedeltaIndex(["4day", "1day", "2day"], name="idx")

        # 在索引位置 2 处插入时间差为 5 天的 Timedelta 对象
        result = idx.insert(2, timedelta(days=5))
        # 预期的 TimedeltaIndex 对象，包含插入后的时间差
        exp = TimedeltaIndex(["4day", "1day", "5day", "2day"], name="idx")
        # 断言插入结果与预期结果相等
        tm.assert_index_equal(result, exp)

        # 插入非日期时间数据将强制转换为对象索引
        result = idx.insert(1, "inserted")
        # 预期的 Index 对象，包含转换后的时间差和字符串
        expected = Index(
            [Timedelta("4day"), "inserted", Timedelta("1day"), Timedelta("2day")],
            name="idx",
        )
        # 断言结果不是 TimedeltaIndex 类型，并且与预期结果相等
        assert not isinstance(result, TimedeltaIndex)
        tm.assert_index_equal(result, expected)
        # 断言结果的名称与预期结果相等
        assert result.name == expected.name

        # 使用 timedelta_range 生成指定间隔的时间差索引对象 idx
        idx = timedelta_range("1day 00:00:01", periods=3, freq="s", name="idx")

        # 预期的 TimedeltaIndex 对象，包含指定频率的时间差字符串
        expected_0 = TimedeltaIndex(
            ["1day", "1day 00:00:01", "1day 00:00:02", "1day 00:00:03"],
            name="idx",
            freq="s",
        )
        expected_3 = TimedeltaIndex(
            ["1day 00:00:01", "1day 00:00:02", "1day 00:00:03", "1day 00:00:04"],
            name="idx",
            freq="s",
        )

        # 重设频率为 None 的预期 TimedeltaIndex 对象
        expected_1_nofreq = TimedeltaIndex(
            ["1day 00:00:01", "1day 00:00:01", "1day 00:00:02", "1day 00:00:03"],
            name="idx",
            freq=None,
        )
        expected_3_nofreq = TimedeltaIndex(
            ["1day 00:00:01", "1day 00:00:02", "1day 00:00:03", "1day 00:00:05"],
            name="idx",
            freq=None,
        )

        # 定义测试用例列表，每个元素包含插入位置、时间差和预期的 TimedeltaIndex 对象
        cases = [
            (0, Timedelta("1day"), expected_0),
            (-3, Timedelta("1day"), expected_0),
            (3, Timedelta("1day 00:00:04"), expected_3),
            (1, Timedelta("1day 00:00:01"), expected_1_nofreq),
            (3, Timedelta("1day 00:00:05"), expected_3_nofreq),
        ]

        # 遍历测试用例，执行插入操作并断言结果与预期相等
        for n, d, expected in cases:
            result = idx.insert(n, d)
            tm.assert_index_equal(result, expected)
            assert result.name == expected.name
            assert result.freq == expected.freq

    # 使用 pytest 的参数化装饰器，测试插入 pd.NaT 等特殊值
    @pytest.mark.parametrize(
        "null", [None, np.nan, np.timedelta64("NaT"), pd.NaT, pd.NA]
    )
    # 定义测试方法 test_insert_nat
    def test_insert_nat(self, null):
        # 创建时间差范围对象 idx，包含 "1day" 到 "3day" 的时间差
        idx = timedelta_range("1day", "3day")
        # 在索引位置 1 处插入 null（例如 pd.NaT）
        result = idx.insert(1, null)
        # 预期的 TimedeltaIndex 对象，包含插入 null 后的时间差
        expected = TimedeltaIndex(["1day", pd.NaT, "2day", "3day"])
        # 断言插入结果与预期结果相等
        tm.assert_index_equal(result, expected)
    # 定义一个测试方法，用于测试在插入无效 NaT（Not a Time）时的情况
    def test_insert_invalid_na(self):
        # 创建一个 TimedeltaIndex 对象，包含字符串数组和指定名称
        idx = TimedeltaIndex(["4day", "1day", "2day"], name="idx")

        # 创建一个 NaT（Not a Time）的 np.datetime64 对象
        item = np.datetime64("NaT")
        # 在索引位置 0 处插入 NaT 对象
        result = idx.insert(0, item)

        # 期望的结果是一个 Index 对象，包含插入的 NaT 对象和原始索引 idx 的内容
        expected = Index([item] + list(idx), dtype=object, name="idx")
        # 检查插入操作的结果是否符合期望
        tm.assert_index_equal(result, expected)

        # 再次插入一个不同的 NaT 对象，应该仍然符合期望结果
        item2 = np.datetime64("NaT")
        result = idx.insert(0, item2)
        tm.assert_index_equal(result, expected)

    # 使用 pytest 的参数化装饰器，测试在插入类型不匹配的情况下是否会引发异常
    @pytest.mark.parametrize(
        "item", [0, np.int64(0), np.float64(0), np.array(0), np.datetime64(456, "us")]
    )
    def test_insert_mismatched_types_raises(self, item):
        # 创建一个 TimedeltaIndex 对象，包含字符串数组和指定名称
        tdi = TimedeltaIndex(["4day", "1day", "2day"], name="idx")

        # 在索引位置 1 处插入指定的 item 对象
        result = tdi.insert(1, item)

        # 期望的结果是一个 Index 对象，包含插入的 item 和原始索引 tdi 的内容
        expected = Index(
            [tdi[0], lib.item_from_zerodim(item)] + list(tdi[1:]),
            dtype=object,
            name="idx",
        )
        # 检查插入操作的结果是否符合期望
        tm.assert_index_equal(result, expected)

    # 测试在插入可转换为字符串的对象时的情况
    def test_insert_castable_str(self):
        # 创建一个时间差范围的 TimedeltaIndex 对象
        idx = timedelta_range("1day", "3day")

        # 在索引位置 0 处插入字符串 "1 Day"
        result = idx.insert(0, "1 Day")

        # 期望的结果是一个 TimedeltaIndex 对象，包含插入的字符串 "1 Day" 和原始索引 idx 的内容
        expected = TimedeltaIndex([idx[0]] + list(idx))
        # 检查插入操作的结果是否符合期望
        tm.assert_index_equal(result, expected)

    # 测试在插入不可转换为字符串的对象时的情况
    def test_insert_non_castable_str(self):
        # 创建一个时间差范围的 TimedeltaIndex 对象
        idx = timedelta_range("1day", "3day")

        # 在索引位置 0 处插入字符串 "foo"
        result = idx.insert(0, "foo")

        # 期望的结果是一个 Index 对象，包含插入的字符串 "foo" 和原始索引 idx 的内容
        expected = Index(["foo"] + list(idx), dtype=object)
        # 检查插入操作的结果是否符合期望
        tm.assert_index_equal(result, expected)

    # 测试在空索引中插入元素的情况
    def test_insert_empty(self):
        # 对于长度为零的索引，插入操作不应引发 IndexError
        # 创建一个时间差范围的 TimedeltaIndex 对象，长度为 3
        idx = timedelta_range("1 Day", periods=3)
        # 获取索引位置 0 处的时间差对象
        td = idx[0]

        # 在长度为 0 的 idx 切片中插入时间差对象 td 到索引位置 0 处
        result = idx[:0].insert(0, td)
        # 检查插入操作后结果的频率是否与原始相同
        assert result.freq == "D"

        # 当在空索引中插入位置为 1 或 -1 处时，应该引发 IndexError 异常
        with pytest.raises(IndexError, match="loc must be an integer between"):
            result = idx[:0].insert(1, td)

        with pytest.raises(IndexError, match="loc must be an integer between"):
            result = idx[:0].insert(-1, td)
```