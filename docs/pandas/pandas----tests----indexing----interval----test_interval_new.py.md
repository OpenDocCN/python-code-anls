# `D:\src\scipysrc\pandas\pandas\tests\indexing\interval\test_interval_new.py`

```
# 导入正则表达式模块
import re

# 导入 numpy 库，并使用别名 np
import numpy as np

# 导入 pytest 测试框架
import pytest

# 从 pandas.compat 模块中导入 IS64 常量
from pandas.compat import IS64

# 从 pandas 模块中导入 Index, Interval, IntervalIndex, Series 类
from pandas import (
    Index,
    Interval,
    IntervalIndex,
    Series,
)

# 导入 pandas._testing 模块，并使用别名 tm
import pandas._testing as tm

# 定义 TestIntervalIndex 测试类
class TestIntervalIndex:
    
    # 定义 pytest fixture，生成包含区间索引的 Series 对象
    @pytest.fixture
    def series_with_interval_index(self):
        return Series(np.arange(5), IntervalIndex.from_breaks(np.arange(6)))

    # 测试方法：测试带有区间索引的 loc 方法
    def test_loc_with_interval(self, series_with_interval_index, indexer_sl):
        # loc 使用单个标签或标签列表：
        #   - 对于区间：只有精确匹配的情况下才返回
        #   - 对于标量：返回包含该标量的区间

        ser = series_with_interval_index.copy()

        expected = 0
        result = indexer_sl(ser)[Interval(0, 1)]
        assert result == expected

        expected = ser.iloc[3:5]
        result = indexer_sl(ser)[[Interval(3, 4), Interval(4, 5)]]
        tm.assert_series_equal(expected, result)

        # 处理缺失或不精确匹配的情况
        with pytest.raises(KeyError, match=re.escape("Interval(3, 5, closed='left')")):
            indexer_sl(ser)[Interval(3, 5, closed="left")]

        with pytest.raises(KeyError, match=re.escape("Interval(3, 5, closed='right')")):
            indexer_sl(ser)[Interval(3, 5)]

        with pytest.raises(
            KeyError, match=re.escape("Interval(-2, 0, closed='right')")
        ):
            indexer_sl(ser)[Interval(-2, 0)]

        with pytest.raises(KeyError, match=re.escape("Interval(5, 6, closed='right')")):
            indexer_sl(ser)[Interval(5, 6)]

    # 测试方法：测试带有标量的 loc 方法
    def test_loc_with_scalar(self, series_with_interval_index, indexer_sl):
        # loc 使用单个标签或标签列表：
        #   - 对于区间：只有精确匹配的情况下才返回
        #   - 对于标量：返回包含该标量的区间

        ser = series_with_interval_index.copy()

        assert indexer_sl(ser)[1] == 0
        assert indexer_sl(ser)[1.5] == 1
        assert indexer_sl(ser)[2] == 1

        expected = ser.iloc[1:4]
        tm.assert_series_equal(expected, indexer_sl(ser)[[1.5, 2.5, 3.5]])
        tm.assert_series_equal(expected, indexer_sl(ser)[[2, 3, 4]])
        tm.assert_series_equal(expected, indexer_sl(ser)[[1.5, 3, 4]])

        expected = ser.iloc[[1, 1, 2, 1]]
        tm.assert_series_equal(expected, indexer_sl(ser)[[1.5, 2, 2.5, 1.5]])

        expected = ser.iloc[2:5]
        tm.assert_series_equal(expected, indexer_sl(ser)[ser >= 2])
    def test_loc_with_slices(self, series_with_interval_index, indexer_sl):
        # loc with slices:
        #   - Interval objects: only works with exact matches
        #   - scalars: only works for non-overlapping, monotonic intervals,
        #     and start/stop select location based on the interval that
        #     contains them:
        #    (slice_loc(start, stop) == (idx.get_loc(start), idx.get_loc(stop))

        ser = series_with_interval_index.copy()

        # slice of interval

        # 从序列中切片获取前三个元素的期望结果
        expected = ser.iloc[:3]
        # 使用索引器函数获取指定区间的结果并进行断言比较
        result = indexer_sl(ser)[Interval(0, 1) : Interval(2, 3)]
        tm.assert_series_equal(expected, result)

        # 从序列中切片获取从第四个元素到最后的期望结果
        expected = ser.iloc[3:]
        # 使用索引器函数获取指定区间的结果并进行断言比较
        result = indexer_sl(ser)[Interval(3, 4) :]
        tm.assert_series_equal(expected, result)

        # 检查使用 Interval 对象进行切片时抛出 NotImplementedError 异常
        msg = "Interval objects are not currently supported"
        with pytest.raises(NotImplementedError, match=msg):
            indexer_sl(ser)[Interval(3, 6) :]

        with pytest.raises(NotImplementedError, match=msg):
            indexer_sl(ser)[Interval(3, 4, closed="left") :]

    def test_slice_step_ne1(self, series_with_interval_index):
        # GH#31658 slice of scalar with step != 1
        ser = series_with_interval_index.copy()
        # 获取序列中从第一个到第四个元素，步长为2的期望结果
        expected = ser.iloc[0:4:2]

        # 使用序列的切片功能获取从第一个到第四个元素，步长为2的结果
        result = ser[0:4:2]
        tm.assert_series_equal(result, expected)

        # 使用序列的切片功能获取从第一个到第四个元素，然后再步长为2进行切片的结果
        result2 = ser[0:4][::2]
        tm.assert_series_equal(result2, expected)

    def test_slice_float_start_stop(self, series_with_interval_index):
        # GH#31658 slicing with integers is positional, with floats is not
        #  supported
        ser = series_with_interval_index.copy()

        # 检查使用浮点数作为起始和结束标签时抛出 ValueError 异常
        msg = "label-based slicing with step!=1 is not supported for IntervalIndex"
        with pytest.raises(ValueError, match=msg):
            ser[1.5:9.5:2]

    def test_slice_interval_step(self, series_with_interval_index):
        # GH#31658 allows for integer step!=1, not Interval step
        ser = series_with_interval_index.copy()
        # 检查使用 Interval 对象作为步长时抛出 ValueError 异常
        msg = "label-based slicing with step!=1 is not supported for IntervalIndex"
        with pytest.raises(ValueError, match=msg):
            ser[0 : 4 : Interval(0, 1)]
    def test_loc_with_overlap(self, indexer_sl):
        # 创建一个包含两个间隔的 IntervalIndex 对象
        idx = IntervalIndex.from_tuples([(1, 5), (3, 7)])
        # 创建一个 Series 对象，使用上述的 IntervalIndex 对象作为索引
        ser = Series(range(len(idx)), index=idx)

        # 测试标量索引
        expected = ser
        result = indexer_sl(ser)[4]  # 使用 indexer_sl 函数对 ser 进行索引操作，索引值为 4
        tm.assert_series_equal(expected, result)

        result = indexer_sl(ser)[[4]]  # 使用 indexer_sl 函数对 ser 进行索引操作，索引值为 [4]
        tm.assert_series_equal(expected, result)

        # 测试间隔索引
        expected = 0
        result = indexer_sl(ser)[Interval(1, 5)]  # 使用 indexer_sl 函数对 ser 进行间隔索引，间隔为 (1, 5)
        assert expected == result

        expected = ser
        result = indexer_sl(ser)[[Interval(1, 5), Interval(3, 7)]]  # 使用 indexer_sl 函数对 ser 进行间隔索引，间隔为 [(1, 5), (3, 7)]
        tm.assert_series_equal(expected, result)

        # 测试抛出 KeyError 的情况，匹配特定的错误信息
        with pytest.raises(KeyError, match=re.escape("Interval(3, 5, closed='right')")):
            indexer_sl(ser)[Interval(3, 5)]

        msg = (
            r"None of \[IntervalIndex\(\[\(3, 5\]\], "
            r"dtype='interval\[int64, right\]'\)\] are in the \[index\]"
        )
        with pytest.raises(KeyError, match=msg):  # 使用 pytest 检查抛出的 KeyError，匹配特定的错误信息
            indexer_sl(ser)[[Interval(3, 5)]]

        # 测试间隔与切片结合的情况（仅精确匹配）
        expected = ser
        result = indexer_sl(ser)[Interval(1, 5) : Interval(3, 7)]  # 使用 indexer_sl 函数对 ser 进行切片索引，切片范围为 (1, 5) 到 (3, 7)
        tm.assert_series_equal(expected, result)

        msg = (
            "'can only get slices from an IntervalIndex if bounds are "
            "non-overlapping and all monotonic increasing or decreasing'"
        )
        with pytest.raises(KeyError, match=msg):  # 使用 pytest 检查抛出的 KeyError，匹配特定的错误信息
            indexer_sl(ser)[Interval(1, 6) : Interval(3, 8)]

        if indexer_sl is tm.loc:
            # 测试与标量结合的切片，期望抛出 KeyError
            # TODO KeyError is the appropriate error?
            with pytest.raises(KeyError, match=msg):
                ser.loc[1:4]

    def test_non_unique(self, indexer_sl):
        # 创建一个包含两个重叠间隔的 IntervalIndex 对象
        idx = IntervalIndex.from_tuples([(1, 3), (3, 7)])
        # 创建一个 Series 对象，使用上述的 IntervalIndex 对象作为索引
        ser = Series(range(len(idx)), index=idx)

        result = indexer_sl(ser)[Interval(1, 3)]  # 使用 indexer_sl 函数对 ser 进行间隔索引，间隔为 (1, 3)
        assert result == 0

        result = indexer_sl(ser)[[Interval(1, 3)]]  # 使用 indexer_sl 函数对 ser 进行间隔索引，间隔为 [(1, 3)]
        expected = ser.iloc[0:1]
        tm.assert_series_equal(expected, result)

    def test_non_unique_moar(self, indexer_sl):
        # 创建一个包含三个重叠间隔的 IntervalIndex 对象
        idx = IntervalIndex.from_tuples([(1, 3), (1, 3), (3, 7)])
        # 创建一个 Series 对象，使用上述的 IntervalIndex 对象作为索引
        ser = Series(range(len(idx)), index=idx)

        expected = ser.iloc[[0, 1]]
        result = indexer_sl(ser)[Interval(1, 3)]  # 使用 indexer_sl 函数对 ser 进行间隔索引，间隔为 (1, 3)
        tm.assert_series_equal(expected, result)

        expected = ser
        result = indexer_sl(ser)[Interval(1, 3) :]  # 使用 indexer_sl 函数对 ser 进行切片索引，切片范围从 (1, 3) 开始到末尾
        tm.assert_series_equal(expected, result)

        expected = ser.iloc[[0, 1]]
        result = indexer_sl(ser)[[Interval(1, 3)]]  # 使用 indexer_sl 函数对 ser 进行间隔索引，间隔为 [(1, 3)]
        tm.assert_series_equal(expected, result)

    def test_loc_getitem_missing_key_error_message(
        self, frame_or_series, series_with_interval_index
    ):
        # GH#27365
        ser = series_with_interval_index.copy()
        obj = frame_or_series(ser)
        with pytest.raises(KeyError, match=r"\[6\]"):  # 使用 pytest 检查抛出的 KeyError，匹配特定的错误信息
            obj.loc[[4, 5, 6]]
# 标记此测试用例为预期失败，如果不是64位系统，则会失败，原因是GH 23440
@pytest.mark.xfail(not IS64, reason="GH 23440")
# 参数化测试，测试不同的间隔集合
@pytest.mark.parametrize(
    "intervals",
    [
        ([Interval(-np.inf, 0.0), Interval(0.0, 1.0)]),    # 第一个间隔集合：负无穷到0.0，0.0到1.0
        ([Interval(-np.inf, -2.0), Interval(-2.0, -1.0)]),   # 第二个间隔集合：负无穷到-2.0，-2.0到-1.0
        ([Interval(-1.0, 0.0), Interval(0.0, np.inf)]),   # 第三个间隔集合：-1.0到0.0，0.0到正无穷
        ([Interval(1.0, 2.0), Interval(2.0, np.inf)]),    # 第四个间隔集合：1.0到2.0，2.0到正无穷
    ],
)
# 定义测试函数，测试具有无穷值的重复间隔索引
def test_repeating_interval_index_with_infs(intervals):
    # GH 46658

    # 创建索引对象，其中每个间隔重复51次
    interval_index = Index(intervals * 51)

    # 期望结果是从1开始，步长为2，共101个元素的整数数组
    expected = np.arange(1, 102, 2, dtype=np.intp)
    # 获取给定间隔的索引器结果
    result = interval_index.get_indexer_for([intervals[1]])

    # 使用测试工具函数断言结果等于期望值
    tm.assert_equal(result, expected)
```