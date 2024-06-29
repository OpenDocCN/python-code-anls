# `D:\src\scipysrc\pandas\pandas\tests\indexes\interval\test_indexing.py`

```
# 导入正则表达式模块
import re

# 导入 NumPy 库，并用 np 别名表示
import numpy as np

# 导入 pytest 测试框架
import pytest

# 导入 pandas 库中的异常类 InvalidIndexError
from pandas.errors import InvalidIndexError

# 从 pandas 库中导入多个类和函数
from pandas import (
    NA,
    CategoricalIndex,
    DatetimeIndex,
    Index,
    Interval,
    IntervalIndex,
    MultiIndex,
    NaT,
    Timedelta,
    Timestamp,
    array,
    date_range,
    interval_range,
    isna,
    period_range,
    timedelta_range,
)

# 导入 pandas 内部测试模块，并用 tm 别名表示
import pandas._testing as tm

# 定义测试类 TestGetItem
class TestGetItem:

    # 定义测试方法 test_getitem，参数为 closed
    def test_getitem(self, closed):
        # 创建一个区间索引对象 idx，从数组创建，包括 NaN 值，使用给定的 closed 参数
        idx = IntervalIndex.from_arrays((0, 1, np.nan), (1, 2, np.nan), closed=closed)
        
        # 断言索引的第一个元素等于 Interval 对象，区间为 (0.0, 1.0)，使用给定的 closed 参数
        assert idx[0] == Interval(0.0, 1.0, closed=closed)
        
        # 断言索引的第二个元素等于 Interval 对象，区间为 (1.0, 2.0)，使用给定的 closed 参数
        assert idx[1] == Interval(1.0, 2.0, closed=closed)
        
        # 断言索引的第三个元素是 NaN
        assert isna(idx[2])

        # 获取 idx 的切片 [0:1]，并赋值给 result
        result = idx[0:1]
        
        # 创建一个期望的 IntervalIndex 对象，从数组创建，区间为 (0.0, 1.0)，使用给定的 closed 参数
        expected = IntervalIndex.from_arrays((0.0,), (1.0,), closed=closed)
        
        # 使用测试模块 tm 的方法 assert_index_equal 来断言 result 等于 expected
        tm.assert_index_equal(result, expected)

        # 获取 idx 的切片 [0:2]，并赋值给 result
        result = idx[0:2]
        
        # 创建一个期望的 IntervalIndex 对象，从数组创建，区间为 (0.0, 1.0) 和 (1.0, 2.0)，使用给定的 closed 参数
        expected = IntervalIndex.from_arrays((0.0, 1), (1.0, 2.0), closed=closed)
        
        # 使用测试模块 tm 的方法 assert_index_equal 来断言 result 等于 expected
        tm.assert_index_equal(result, expected)

        # 获取 idx 的切片 [1:3]，并赋值给 result
        result = idx[1:3]
        
        # 创建一个期望的 IntervalIndex 对象，从数组创建，区间为 (1.0, NaN) 和 (2.0, NaN)，使用给定的 closed 参数
        expected = IntervalIndex.from_arrays(
            (1.0, np.nan), (2.0, np.nan), closed=closed
        )
        
        # 使用测试模块 tm 的方法 assert_index_equal 来断言 result 等于 expected
        tm.assert_index_equal(result, expected)

    # 定义测试方法 test_getitem_2d_deprecated
    def test_getitem_2d_deprecated(self):
        # 创建一个区间索引对象 idx，从断点列表创建，使用 'right' 作为 closed 参数
        idx = IntervalIndex.from_breaks(range(11), closed="right")
        
        # 使用 pytest.raises 来断言会抛出 ValueError 异常，并检查异常信息中是否包含 "multi-dimensional indexing not allowed"
        with pytest.raises(ValueError, match="multi-dimensional indexing not allowed"):
            idx[:, None]
        
        # 使用 pytest.raises 来断言会抛出 ValueError 异常，并检查异常信息中是否包含 "multi-dimensional indexing not allowed"
        with pytest.raises(ValueError, match="multi-dimensional indexing not allowed"):
            # GH#44051
            idx[True]
        
        # 使用 pytest.raises 来断言会抛出 ValueError 异常，并检查异常信息中是否包含 "multi-dimensional indexing not allowed"
        with pytest.raises(ValueError, match="multi-dimensional indexing not allowed"):
            # GH#44051
            idx[False]


# 定义测试类 TestWhere
class TestWhere:

    # 定义测试方法 test_where，参数为 listlike_box
    def test_where(self, listlike_box):
        # 从断点列表创建区间索引对象 idx，使用 'right' 作为 closed 参数
        idx = IntervalIndex.from_breaks(range(11), closed="right")
        
        # 创建一个条件列表 cond，长度与 idx 相同，所有元素为 True
        cond = [True] * len(idx)
        
        # 创建一个期望的 IntervalIndex 对象 expected，与 idx 相同
        expected = idx
        
        # 使用 klass(cond) 来处理 expected，结果赋值给 result
        result = expected.where(listlike_box(cond))
        
        # 使用测试模块 tm 的方法 assert_index_equal 来断言 result 等于 expected
        tm.assert_index_equal(result, expected)

        # 更新条件列表 cond，第一个元素为 False，其余为 True
        cond = [False] + [True] * len(idx[1:])
        
        # 创建一个期望的 IntervalIndex 对象 expected，第一个元素为 NaN，其余与 idx[1:] 相同
        expected = IntervalIndex([np.nan] + idx[1:].tolist())
        
        # 使用 idx.where(klass(cond)) 处理 idx，结果赋值给 result
        result = idx.where(listlike_box(cond))
        
        # 使用测试模块 tm 的方法 assert_index_equal 来断言 result 等于 expected
        tm.assert_index_equal(result, expected)


# 定义测试类 TestTake
class TestTake:

    # 定义测试方法 test_take，参数为 closed
    def test_take(self, closed):
        # 从断点列表创建区间索引对象 index，使用给定的 closed 参数
        index = IntervalIndex.from_breaks(range(11), closed=closed)
        
        # 获取 index 的前 10 个元素，赋值给 result
        result = index.take(range(10))
        
        # 使用测试模块 tm 的方法 assert_index_equal 来断言 result 等于 index
        tm.assert_index_equal(result, index)

        # 获取 index 中指定位置的元素，赋值给 result
        result = index.take([0, 0, 1])
        
        # 创建一个期望的 IntervalIndex 对象 expected，从数组创建，区间为 [0, 0, 1] 和 [1, 1, 2]，使用给定的 closed 参数
        expected = IntervalIndex.from_arrays([0, 0, 1], [1, 1, 2], closed=closed)
        
        # 使用测试模块 tm 的方法 assert_index_equal 来断言 result 等于 expected
        tm.assert_index_equal(result, expected)


# 定义测试类 TestGetLoc
class TestGetLoc:

    # 参数化测试方法 side，参数为 "right", "left", "both", "neither"
    @pytest.mark.parametrize("side", ["right", "left", "both", "neither"])
    def test_get_loc_interval(self, closed, side):
        # 创建一个 IntervalIndex 对象，包含两个区间 [(0, 1), (2, 3)]，根据传入的闭合参数设置闭合性质
        idx = IntervalIndex.from_tuples([(0, 1), (2, 3)], closed=closed)

        # 遍历不同的边界条件
        for bound in [[0, 1], [1, 2], [2, 3], [3, 4], [0, 2], [2.5, 3], [-1, 4]]:
            # 如果 get_loc 接收到一个区间作为输入，则应该只搜索完全匹配的区间，而不是重叠或覆盖的区间，否则会引发 KeyError
            msg = re.escape(f"Interval({bound[0]}, {bound[1]}, closed='{side}')")
            if closed == side:
                if bound == [0, 1]:
                    # 断言特定区间的索引位置是否为预期值
                    assert idx.get_loc(Interval(0, 1, closed=side)) == 0
                elif bound == [2, 3]:
                    # 断言特定区间的索引位置是否为预期值
                    assert idx.get_loc(Interval(2, 3, closed=side)) == 1
                else:
                    # 使用 pytest 断言预期引发 KeyError，并匹配特定的错误信息
                    with pytest.raises(KeyError, match=msg):
                        idx.get_loc(Interval(*bound, closed=side))
            else:
                # 使用 pytest 断言预期引发 KeyError，并匹配特定的错误信息
                with pytest.raises(KeyError, match=msg):
                    idx.get_loc(Interval(*bound, closed=side))

    @pytest.mark.parametrize("scalar", [-0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
    def test_get_loc_scalar(self, closed, scalar):
        # correct = {side: {query: answer}}.
        # 如果查询不在字典中，则应该引发 KeyError
        correct = {
            "right": {0.5: 0, 1: 0, 2.5: 1, 3: 1},
            "left": {0: 0, 0.5: 0, 2: 1, 2.5: 1},
            "both": {0: 0, 0.5: 0, 1: 0, 2: 1, 2.5: 1, 3: 1},
            "neither": {0.5: 0, 2.5: 1},
        }

        # 创建一个 IntervalIndex 对象，包含两个区间 [(0, 1), (2, 3)]，根据传入的闭合参数设置闭合性质
        idx = IntervalIndex.from_tuples([(0, 1), (2, 3)], closed=closed)

        # 如果 scalar 在 correct[closed] 的键集合中
        if scalar in correct[closed].keys():
            # 断言 idx.get_loc(scalar) 返回的索引值与预期值相等
            assert idx.get_loc(scalar) == correct[closed][scalar]
        else:
            # 使用 pytest 断言预期引发 KeyError，并匹配 scalar 的字符串表示
            with pytest.raises(KeyError, match=str(scalar)):
                idx.get_loc(scalar)

    @pytest.mark.parametrize("scalar", [-1, 0, 0.5, 3, 4.5, 5, 6])
    def test_get_loc_length_one_scalar(self, scalar, closed):
        # GH 20921
        # 创建一个包含单个区间 (0, 5) 的 IntervalIndex 对象，根据传入的闭合参数设置闭合性质
        index = IntervalIndex.from_tuples([(0, 5)], closed=closed)
        if scalar in index[0]:
            # 如果 scalar 在区间内
            result = index.get_loc(scalar)
            # 断言结果等于预期值 0
            assert result == 0
        else:
            # 使用 pytest 断言预期引发 KeyError，并匹配 scalar 的字符串表示
            with pytest.raises(KeyError, match=str(scalar)):
                index.get_loc(scalar)

    @pytest.mark.parametrize("left, right", [(0, 5), (-1, 4), (-1, 6), (6, 7)])
    def test_get_loc_length_one_interval(self, left, right, closed, other_closed):
        # GH 20921
        # 创建一个包含单个区间 (0, 5) 的 IntervalIndex 对象，根据传入的闭合参数设置闭合性质
        index = IntervalIndex.from_tuples([(0, 5)], closed=closed)
        # 创建一个指定左右端点和闭合性质的 Interval 对象
        interval = Interval(left, right, closed=other_closed)
        if interval == index[0]:
            # 如果 interval 等于 index 中的第一个区间
            result = index.get_loc(interval)
            # 断言结果等于预期值 0
            assert result == 0
        else:
            # 使用 pytest 断言预期引发 KeyError，并匹配特定的错误信息
            with pytest.raises(
                KeyError,
                match=re.escape(f"Interval({left}, {right}, closed='{other_closed}')"),
            ):
                index.get_loc(interval)
    # 使用 pytest.mark.parametrize 装饰器为 test_get_loc_datetimelike_nonoverlapping 函数定义参数化测试用例
    @pytest.mark.parametrize(
        "breaks",
        [
            date_range("20180101", periods=4),  # 生成一个日期范围作为参数
            date_range("20180101", periods=4, tz="US/Eastern"),  # 生成一个带时区的日期范围作为参数
            timedelta_range("0 days", periods=4),  # 生成一个时间间隔范围作为参数
        ],
        ids=lambda x: str(x.dtype),  # 设置参数化测试用例的标识符为参数的数据类型字符串表示
    )
    # 定义测试函数 test_get_loc_datetimelike_nonoverlapping，参数为 breaks
    def test_get_loc_datetimelike_nonoverlapping(self, breaks):
        # GH 20636
        # 使用 IntervalIndex.from_breaks 方法根据 breaks 创建 IntervalIndex 对象
        index = IntervalIndex.from_breaks(breaks)

        # 获取第一个间隔的中点值
        value = index[0].mid
        # 调用 index 对象的 get_loc 方法，查找 value 的位置
        result = index.get_loc(value)
        expected = 0  # 期望找到的位置是第一个位置
        assert result == expected

        # 创建一个与第一个间隔相同范围的 Interval 对象
        interval = Interval(index[0].left, index[0].right)
        # 调用 index 对象的 get_loc 方法，查找 interval 的位置
        result = index.get_loc(interval)
        assert result == expected  # 期望找到的位置是第一个位置

    # 使用 pytest.mark.parametrize 装饰器为 test_get_loc_datetimelike_overlapping 函数定义参数化测试用例
    @pytest.mark.parametrize(
        "arrays",
        [
            (date_range("20180101", periods=4), date_range("20180103", periods=4)),  # 生成两个日期范围作为参数
            (
                date_range("20180101", periods=4, tz="US/Eastern"),
                date_range("20180103", periods=4, tz="US/Eastern"),  # 生成两个带时区的日期范围作为参数
            ),
            (
                timedelta_range("0 days", periods=4),
                timedelta_range("2 days", periods=4),  # 生成两个时间间隔范围作为参数
            ),
        ],
        ids=lambda x: str(x[0].dtype),  # 设置参数化测试用例的标识符为第一个参数的数据类型字符串表示
    )
    # 定义测试函数 test_get_loc_datetimelike_overlapping，参数为 arrays
    def test_get_loc_datetimelike_overlapping(self, arrays):
        # GH 20636
        # 使用 IntervalIndex.from_arrays 方法根据 arrays 创建 IntervalIndex 对象
        index = IntervalIndex.from_arrays(*arrays)

        # 计算第一个间隔的中点值加上 12 小时后的值
        value = index[0].mid + Timedelta("12 hours")
        # 调用 index 对象的 get_loc 方法，查找 value 的位置
        result = index.get_loc(value)
        expected = slice(0, 2, None)  # 期望找到的位置是一个切片对象，表示位置在 0 到 2 之间
        assert result == expected

        # 创建一个与第一个间隔相同范围的 Interval 对象
        interval = Interval(index[0].left, index[0].right)
        # 调用 index 对象的 get_loc 方法，查找 interval 的位置
        result = index.get_loc(interval)
        expected = 0  # 期望找到的位置是第一个位置
        assert result == expected

    # 使用 pytest.mark.parametrize 装饰器为 test_get_loc_decreasing 函数定义参数化测试用例
    @pytest.mark.parametrize(
        "values",
        [
            date_range("2018-01-04", periods=4, freq="-1D"),  # 生成一个递减的日期范围作为参数
            date_range("2018-01-04", periods=4, freq="-1D", tz="US/Eastern"),  # 生成一个带时区的递减的日期范围作为参数
            timedelta_range("3 days", periods=4, freq="-1D"),  # 生成一个递减的时间间隔范围作为参数
            np.arange(3.0, -1.0, -1.0),  # 生成一个递减的浮点数数组作为参数
            np.arange(3, -1, -1),  # 生成一个递减的整数数组作为参数
        ],
        ids=lambda x: str(x.dtype),  # 设置参数化测试用例的标识符为参数的数据类型字符串表示
    )
    # 定义测试函数 test_get_loc_decreasing，参数为 values
    def test_get_loc_decreasing(self, values):
        # GH 25860
        # 使用 IntervalIndex.from_arrays 方法根据 values 创建 IntervalIndex 对象
        index = IntervalIndex.from_arrays(values[1:], values[:-1])
        # 调用 index 对象的 get_loc 方法，查找 index[0] 的位置
        result = index.get_loc(index[0])
        expected = 0  # 期望找到的位置是第一个位置
        assert result == expected

    # 使用 pytest.mark.parametrize 装饰器为 test_get_loc_non_scalar_errors 函数定义参数化测试用例
    @pytest.mark.parametrize("key", [[5], (2, 3)])
    # 定义测试函数 test_get_loc_non_scalar_errors，参数为 key
    def test_get_loc_non_scalar_errors(self, key):
        # GH 31117
        # 使用 IntervalIndex.from_tuples 方法根据给定元组列表创建 IntervalIndex 对象
        idx = IntervalIndex.from_tuples([(1, 3), (2, 4), (3, 5), (7, 10), (3, 10)])

        # 将 key 转换为字符串，作为异常信息的一部分
        msg = str(key)
        # 使用 pytest.raises 检测是否会抛出指定类型的异常，并且异常信息包含 msg
        with pytest.raises(InvalidIndexError, match=msg):
            idx.get_loc(key)
    # 定义测试方法 test_get_indexer_with_nans，用于测试索引器处理 NaN 值的情况
    def test_get_indexer_with_nans(self):
        # GH#41831: 关联 GitHub issue 编号 41831
        # 创建一个 IntervalIndex 对象，包含 NaN、Interval(1, 2)、NaN 三个元素
        index = IntervalIndex([np.nan, Interval(1, 2), np.nan])

        # 预期的结果是一个布尔数组 [True, False, True]
        expected = np.array([True, False, True])

        # 对于 [None, np.nan, NA] 中的每个键 key 进行测试
        for key in [None, np.nan, NA]:
            # 断言 key 在 index 中
            assert key in index
            # 获取 key 的位置索引
            result = index.get_loc(key)
            # 使用测试工具函数确保 result 与预期结果 expected 相等
            tm.assert_numpy_array_equal(result, expected)

        # 对于 [NaT, np.timedelta64("NaT", "ns"), np.datetime64("NaT", "ns")] 中的每个键 key 进行测试
        for key in [NaT, np.timedelta64("NaT", "ns"), np.datetime64("NaT", "ns")]:
            # 预期会抛出 KeyError 异常，并匹配 key 的字符串描述
            with pytest.raises(KeyError, match=str(key)):
                index.get_loc(key)
# 定义一个测试类 TestGetIndexer，用于测试 IntervalIndex 的 get_indexer 方法
class TestGetIndexer:

    # 使用 pytest 的 parametrize 装饰器，为 test_get_indexer_with_interval 方法提供多组参数化测试数据
    @pytest.mark.parametrize(
        "query, expected",
        [
            ([Interval(2, 4, closed="right")], [1]),       # 测试闭区间右端点查询，期望返回索引 1
            ([Interval(2, 4, closed="left")], [-1]),       # 测试闭区间左端点查询，期望返回索引 -1
            ([Interval(2, 4, closed="both")], [-1]),       # 测试闭区间两端点都闭合查询，期望返回索引 -1
            ([Interval(2, 4, closed="neither")], [-1]),    # 测试闭区间两端点都不闭合查询，期望返回索引 -1
            ([Interval(1, 4, closed="right")], [-1]),      # 测试第一个区间右端点闭合查询，期望返回索引 -1
            ([Interval(0, 4, closed="right")], [-1]),      # 测试第一个区间右端点闭合查询，期望返回索引 -1
            ([Interval(0.5, 1.5, closed="right")], [-1]),  # 测试第一个区间右端点闭合查询，期望返回索引 -1
            ([Interval(2, 4, closed="right"), Interval(0, 1, closed="right")], [1, -1]),  # 测试两个区间右端点闭合查询，期望返回 [1, -1]
            ([Interval(2, 4, closed="right"), Interval(2, 4, closed="right")], [1, 1]),    # 测试两个相同区间右端点闭合查询，期望返回 [1, 1]
            ([Interval(5, 7, closed="right"), Interval(2, 4, closed="right")], [2, 1]),    # 测试不同区间右端点闭合查询，期望返回 [2, 1]
            ([Interval(2, 4, closed="right"), Interval(2, 4, closed="left")], [1, -1]),    # 测试一个右端点闭合，一个左端点闭合查询，期望返回 [1, -1]
        ],
    )
    # 定义测试方法 test_get_indexer_with_interval，接受 query 和 expected 作为参数
    def test_get_indexer_with_interval(self, query, expected):
        # 定义区间元组列表
        tuples = [(0, 2), (2, 4), (5, 7)]
        # 使用右端点闭合方式创建 IntervalIndex 对象
        index = IntervalIndex.from_tuples(tuples, closed="right")

        # 调用 get_indexer 方法，传入查询 query，获取结果 result
        result = index.get_indexer(query)
        # 创建预期结果的 numpy 数组，指定数据类型为 intp
        expected = np.array(expected, dtype="intp")
        # 使用 pytest 提供的断言方法，验证 result 和 expected 是否相等
        tm.assert_numpy_array_equal(result, expected)

    # 使用 pytest 的 parametrize 装饰器，为 test_get_indexer_with_int_and_float 方法提供多组参数化测试数据
    @pytest.mark.parametrize(
        "query, expected",
        [
            ([-0.5], [-1]),
            ([0], [-1]),
            ([0.5], [0]),
            ([1], [0]),
            ([1.5], [1]),
            ([2], [1]),
            ([2.5], [-1]),
            ([3], [-1]),
            ([3.5], [2]),
            ([4], [2]),
            ([4.5], [-1]),
            ([1, 2], [0, 1]),
            ([1, 2, 3], [0, 1, -1]),
            ([1, 2, 3, 4], [0, 1, -1, 2]),
            ([1, 2, 3, 4, 2], [0, 1, -1, 2, 1]),
        ],
    )
    # 定义测试方法 test_get_indexer_with_int_and_float，接受 query 和 expected 作为参数
    def test_get_indexer_with_int_and_float(self, query, expected):
        # 定义区间元组列表
        tuples = [(0, 1), (1, 2), (3, 4)]
        # 使用右端点闭合方式创建 IntervalIndex 对象
        index = IntervalIndex.from_tuples(tuples, closed="right")

        # 调用 get_indexer 方法，传入查询 query，获取结果 result
        result = index.get_indexer(query)
        # 创建预期结果的 numpy 数组，指定数据类型为 intp
        expected = np.array(expected, dtype="intp")
        # 使用 pytest 提供的断言方法，验证 result 和 expected 是否相等
        tm.assert_numpy_array_equal(result, expected)

    # 使用 pytest 的 parametrize 装饰器，为 test_get_indexer_length_one 方法提供多组参数化测试数据
    @pytest.mark.parametrize("item", [[3], np.arange(0.5, 5, 0.5)])
    # 定义测试方法 test_get_indexer_length_one，接受 item 和 closed 作为参数
    def test_get_indexer_length_one(self, item, closed):
        # GH 17284
        # 使用指定的闭合方式创建 IntervalIndex 对象
        index = IntervalIndex.from_tuples([(0, 5)], closed=closed)
        # 调用 get_indexer 方法，传入查询 item，获取结果 result
        result = index.get_indexer(item)
        # 创建预期结果的 numpy 数组，长度与 item 相同，数据类型为 intp，填充值为 0
        expected = np.array([0] * len(item), dtype="intp")
        # 使用 pytest 提供的断言方法，验证 result 和 expected 是否相等
        tm.assert_numpy_array_equal(result, expected)

    # 使用 pytest 的 parametrize 装饰器，为 test_get_indexer_length_one_interval 方法提供多组参数化测试数据
    @pytest.mark.parametrize("size", [1, 5])
    # 定义测试方法 test_get_indexer_length_one_interval，接受 size 和 closed 作为参数
    def test_get_indexer_length_one_interval(self, size, closed):
        # GH 17284
        # 使用指定的闭合方式创建 IntervalIndex 对象
        index = IntervalIndex.from_tuples([(0, 5)], closed=closed)
        # 创建一个包含指定区间的列表，长度为 size
        intervals = [Interval(0, 5, closed)] * size
        # 调用 get_indexer 方法，传入查询 intervals，获取结果 result
        result = index.get_indexer(intervals)
        # 创建预期结果的 numpy 数组，长度与 size 相同，数据类型为 intp，填充值为 0
        expected = np.array([0] * size, dtype="intp")
        # 使用 pytest 提供的断言方法，验证 result 和 expected 是否相等
        tm.assert_numpy_array_equal(result, expected)
    @pytest.mark.parametrize(
        "target",
        [
            IntervalIndex.from_tuples([(7, 8), (1, 2), (3, 4), (0, 1)]),
            IntervalIndex.from_tuples([(0, 1), (1, 2), (3, 4), np.nan]),
            IntervalIndex.from_tuples([(0, 1), (1, 2), (3, 4)], closed="both"),
            [-1, 0, 0.5, 1, 2, 2.5, np.nan],
            ["foo", "foo", "bar", "baz"],
        ],
    )


        # 参数化测试用例，对不同的 target 参数进行测试
        def test_get_indexer_categorical(self, target, ordered):
            # GH 30063: categorical and non-categorical results should be consistent
            # 创建一个 IntervalIndex 对象作为测试数据
            index = IntervalIndex.from_tuples([(0, 1), (1, 2), (3, 4)])
            # 使用 CategoricalIndex 类创建一个分类索引对象
            categorical_target = CategoricalIndex(target, ordered=ordered)

            # 获取 index 对象对 categorical_target 的索引器结果
            result = index.get_indexer(categorical_target)
            # 获取 index 对象对 target 的预期索引器结果
            expected = index.get_indexer(target)
            # 使用测试框架中的断言函数，验证 result 和 expected 是否相等
            tm.assert_numpy_array_equal(result, expected)

        def test_get_indexer_categorical_with_nans(self):
            # GH#41934 nans in both index and in target
            # 创建一个 IntervalIndex 对象 ii，通过 from_breaks 方法
            ii = IntervalIndex.from_breaks(range(5))
            # 向 ii 中添加一个包含 np.nan 的 IntervalIndex 对象 ii2
            ii2 = ii.append(IntervalIndex([np.nan]))
            # 使用 CategoricalIndex 类创建一个分类索引对象 ci2
            ci2 = CategoricalIndex(ii2)

            # 获取 ii2 对象对 ci2 的索引器结果
            result = ii2.get_indexer(ci2)
            # 创建一个预期的索引结果数组 expected
            expected = np.arange(5, dtype=np.intp)
            # 使用测试框架中的断言函数，验证 result 和 expected 是否相等
            tm.assert_numpy_array_equal(result, expected)

            # 对 ii2[1:] 对象对 ci2[::-1] 的索引器结果进行测试
            result = ii2[1:].get_indexer(ci2[::-1])
            # 创建一个预期的索引结果数组 expected
            expected = np.array([3, 2, 1, 0, -1], dtype=np.intp)
            # 使用测试框架中的断言函数，验证 result 和 expected 是否相等
            tm.assert_numpy_array_equal(result, expected)

            # 对 ii2 对象对 ci2.append(ci2) 的索引器结果进行测试
            result = ii2.get_indexer(ci2.append(ci2))
            # 创建一个预期的索引结果数组 expected
            expected = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.intp)
            # 使用测试框架中的断言函数，验证 result 和 expected 是否相等
            tm.assert_numpy_array_equal(result, expected)

        def test_get_indexer_datetime(self):
            # 创建一个 IntervalIndex 对象 ii，通过 from_breaks 方法
            ii = IntervalIndex.from_breaks(date_range("2018-01-01", periods=4))
            # 创建一个 DatetimeIndex 对象 target，包含一个日期 "2018-01-02"
            target = DatetimeIndex(["2018-01-02"], dtype="M8[ns]")

            # 获取 ii 对象对 target 的索引器结果
            result = ii.get_indexer(target)
            # 创建一个预期的索引结果数组 expected
            expected = np.array([0], dtype=np.intp)
            # 使用测试框架中的断言函数，验证 result 和 expected 是否相等
            tm.assert_numpy_array_equal(result, expected)

            # 获取 ii 对象对 target.astype(str) 的索引器结果
            result = ii.get_indexer(target.astype(str))
            # 使用测试框架中的断言函数，验证 result 和 expected 是否相等
            tm.assert_numpy_array_equal(result, expected)

            # 获取 ii 对象对 target.asi8 的索引器结果
            result = ii.get_indexer(target.asi8)
            # 创建一个预期的索引结果数组 expected
            expected = np.array([-1], dtype=np.intp)
            # 使用测试框架中的断言函数，验证 result 和 expected 是否相等
            tm.assert_numpy_array_equal(result, expected)

        @pytest.mark.parametrize(
            "tuples, closed",
            [
                ([(0, 2), (1, 3), (3, 4)], "neither"),
                ([(0, 5), (1, 4), (6, 7)], "left"),
                ([(0, 1), (0, 1), (1, 2)], "right"),
                ([(0, 1), (2, 3), (3, 4)], "both"),
            ],
        )
    # 定义一个测试方法，用于测试 IntervalIndex 的错误处理
    def test_get_indexer_errors(self, tuples, closed):
        # 根据给定的元组列表和闭合方式创建 IntervalIndex 对象
        index = IntervalIndex.from_tuples(tuples, closed=closed)

        # 定义错误信息消息字符串
        msg = (
            "cannot handle overlapping indices; use "
            "IntervalIndex.get_indexer_non_unique"
        )
        # 使用 pytest 检查是否抛出 InvalidIndexError 异常，并匹配特定消息
        with pytest.raises(InvalidIndexError, match=msg):
            index.get_indexer([0, 2])

    # 使用 pytest 的参数化标记来定义多组测试参数
    @pytest.mark.parametrize(
        "query, expected",
        [
            ([-0.5], ([-1], [0])),
            ([0], ([0], [])),
            ([0.5], ([0], [])),
            ([1], ([0, 1], [])),
            ([1.5], ([0, 1], [])),
            ([2], ([0, 1, 2], [])),
            ([2.5], ([1, 2], [])),
            ([3], ([2], [])),
            ([3.5], ([2], [])),
            ([4], ([-1], [0])),
            ([4.5], ([-1], [0])),
            ([1, 2], ([0, 1, 0, 1, 2], [])),
            ([1, 2, 3], ([0, 1, 0, 1, 2, 2], [])),
            ([1, 2, 3, 4], ([0, 1, 0, 1, 2, 2, -1], [3])),
            ([1, 2, 3, 4, 2], ([0, 1, 0, 1, 2, 2, -1, 0, 1, 2], [3])),
        ],
    )
    # 定义测试方法，测试处理包含整数和浮点数的非唯一索引
    def test_get_indexer_non_unique_with_int_and_float(self, query, expected):
        # 定义元组列表
        tuples = [(0, 2.5), (1, 3), (2, 4)]
        # 根据元组列表和指定闭合方式创建 IntervalIndex 对象
        index = IntervalIndex.from_tuples(tuples, closed="left")

        # 调用被测试的方法，获取索引器和缺失值列表
        result_indexer, result_missing = index.get_indexer_non_unique(query)
        # 将预期结果转换为 NumPy 数组
        expected_indexer = np.array(expected[0], dtype="intp")
        expected_missing = np.array(expected[1], dtype="intp")

        # 使用 pandas.testing 模块的方法比较 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result_indexer, expected_indexer)
        tm.assert_numpy_array_equal(result_missing, expected_missing)

        # TODO 我们可能还希望测试 get_indexer 方法，处理重复区间、递减区间、非单调区间等情况

    # 定义测试方法，测试处理非单调区间的情况
    def test_get_indexer_non_monotonic(self):
        # GH 16410
        # 创建两个不同的 IntervalIndex 对象
        idx1 = IntervalIndex.from_tuples([(2, 3), (4, 5), (0, 1)])
        idx2 = IntervalIndex.from_tuples([(0, 1), (2, 3), (6, 7), (8, 9)])
        # 调用 get_indexer 方法，获取索引器
        result = idx1.get_indexer(idx2)
        # 定义预期的 NumPy 数组结果
        expected = np.array([2, 0, -1, -1], dtype=np.intp)
        # 使用 pandas.testing 模块的方法比较 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)

        # 再次调用 get_indexer 方法，测试单调性
        result = idx1.get_indexer(idx1[1:])
        # 定义预期的 NumPy 数组结果
        expected = np.array([1, 2], dtype=np.intp)
        # 使用 pandas.testing 模块的方法比较 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)

    # 定义测试方法，测试处理包含 NaN 的情况
    def test_get_indexer_with_nans(self):
        # GH#41831
        # 创建包含 NaN 的 IntervalIndex 对象
        index = IntervalIndex([np.nan, np.nan])
        other = IntervalIndex([np.nan])

        # 断言检查是否 index 不是作为唯一索引
        assert not index._index_as_unique

        # 调用 get_indexer_for 方法，获取索引器
        result = index.get_indexer_for(other)
        # 定义预期的 NumPy 数组结果
        expected = np.array([0, 1], dtype=np.intp)
        # 使用 pandas.testing 模块的方法比较 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)
    def test_get_index_non_unique_non_monotonic(self):
        # GH#44084 (root cause)
        # 创建一个 IntervalIndex 对象，包含非唯一非单调的区间列表
        index = IntervalIndex.from_tuples(
            [(0.0, 1.0), (1.0, 2.0), (0.0, 1.0), (1.0, 2.0)]
        )

        # 调用 get_indexer_non_unique 方法，查找给定区间 [Interval(1.0, 2.0)] 的索引
        result, _ = index.get_indexer_non_unique([Interval(1.0, 2.0)])
        # 期望得到的索引数组
        expected = np.array([1, 3], dtype=np.intp)
        # 使用测试工具函数验证 result 是否与 expected 相等
        tm.assert_numpy_array_equal(result, expected)

    def test_get_indexer_multiindex_with_intervals(self):
        # GH#44084 (MultiIndex case as reported)
        # 创建一个带有名称 "interval" 的 IntervalIndex 对象，包含区间元组列表
        interval_index = IntervalIndex.from_tuples(
            [(2.0, 3.0), (0.0, 1.0), (1.0, 2.0)], name="interval"
        )
        # 创建一个名称为 "foo" 的 Index 对象
        foo_index = Index([1, 2, 3], name="foo")

        # 使用 from_product 创建一个 MultiIndex 对象，结合 foo_index 和 interval_index
        multi_index = MultiIndex.from_product([foo_index, interval_index])

        # 调用 get_level_values("interval").get_indexer_for 方法，查找给定区间 [Interval(0.0, 1.0)] 的索引
        result = multi_index.get_level_values("interval").get_indexer_for(
            [Interval(0.0, 1.0)]
        )
        # 期望得到的索引数组
        expected = np.array([1, 4, 7], dtype=np.intp)
        # 使用测试工具函数验证 result 是否与 expected 相等
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("box", [IntervalIndex, array, list])
    def test_get_indexer_interval_index(self, box):
        # GH#30178
        # 创建一个日期范围对象 rng，包含三个日期
        rng = period_range("2022-07-01", freq="D", periods=3)
        # 使用 box 创建一个 IntervalIndex 对象 idx，包含一个时间间隔范围
        idx = box(interval_range(Timestamp("2022-07-01"), freq="3D", periods=3))

        # 调用 get_indexer 方法，查找 rng 对象中 idx 的索引
        actual = rng.get_indexer(idx)
        # 期望得到的索引数组，这里期望全为 -1，因为日期范围和区间范围不匹配
        expected = np.array([-1, -1, -1], dtype=np.intp)
        # 使用测试工具函数验证 actual 是否与 expected 相等
        tm.assert_numpy_array_equal(actual, expected)

    def test_get_indexer_read_only(self):
        # 创建一个时间间隔范围对象 idx，从 0 到 5
        idx = interval_range(start=0, end=5)
        # 创建一个不可写的数组 arr，包含 [1, 2]
        arr = np.array([1, 2])
        arr.flags.writeable = False
        # 调用 get_indexer 方法，查找 idx 对象中 arr 的索引
        result = idx.get_indexer(arr)
        # 期望得到的索引数组
        expected = np.array([0, 1])
        # 使用测试工具函数验证 result 是否与 expected 相等，不检查数据类型
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

        # 调用 get_indexer_non_unique 方法，查找 idx 对象中 arr 的索引（非唯一情况）
        result = idx.get_indexer_non_unique(arr)[0]
        # 使用测试工具函数验证 result 是否与 expected 相等，不检查数据类型
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)
class TestSliceLocs:
    def test_slice_locs_with_ints_and_floats_succeeds(self):
        # 创建一个 IntervalIndex 对象，包含给定的区间元组列表
        index = IntervalIndex.from_tuples([(0, 1), (1, 2), (3, 4)])

        # 断言：对于给定的起始和结束值，返回相应的切片位置元组
        assert index.slice_locs(0, 1) == (0, 1)
        assert index.slice_locs(0, 2) == (0, 2)
        assert index.slice_locs(0, 3) == (0, 2)
        assert index.slice_locs(3, 1) == (2, 1)
        assert index.slice_locs(3, 4) == (2, 3)
        assert index.slice_locs(0, 4) == (0, 3)

        # 创建一个新的 IntervalIndex 对象，包含给定的区间元组列表（递减顺序）
        index = IntervalIndex.from_tuples([(3, 4), (1, 2), (0, 1)])
        # 断言：对于给定的起始和结束值，返回相应的切片位置元组
        assert index.slice_locs(0, 1) == (3, 3)
        assert index.slice_locs(0, 2) == (3, 2)
        assert index.slice_locs(0, 3) == (3, 1)
        assert index.slice_locs(3, 1) == (1, 3)
        assert index.slice_locs(3, 4) == (1, 1)
        assert index.slice_locs(0, 4) == (3, 1)

    @pytest.mark.parametrize("query", [[0, 1], [0, 2], [0, 3], [0, 4]])
    @pytest.mark.parametrize(
        "tuples",
        [
            [(0, 2), (1, 3), (2, 4)],
            [(2, 4), (1, 3), (0, 2)],
            [(0, 2), (0, 2), (2, 4)],
            [(0, 2), (2, 4), (0, 2)],
            [(0, 2), (0, 2), (2, 4), (1, 3)],
        ],
    )
    def test_slice_locs_with_ints_and_floats_errors(self, tuples, query):
        start, stop = query
        # 创建一个 IntervalIndex 对象，包含给定的区间元组列表
        index = IntervalIndex.from_tuples(tuples)
        # 使用 pytest 断言，检查在给定条件下是否会引发 KeyError 异常
        with pytest.raises(
            KeyError,
            match=(
                "'can only get slices from an IntervalIndex if bounds are "
                "non-overlapping and all monotonic increasing or decreasing'"
            ),
        ):
            # 调用 slice_locs 方法，预期会引发 KeyError 异常
            index.slice_locs(start, stop)


class TestPutmask:
    @pytest.mark.parametrize("tz", ["US/Pacific", None])
    def test_putmask_dt64(self, tz):
        # 创建一个 DatetimeIndex 对象，从指定日期开始，生成一系列日期时间
        dti = date_range("2016-01-01", periods=9, tz=tz)
        # 使用日期时间创建一个 IntervalIndex 对象
        idx = IntervalIndex.from_breaks(dti)
        # 创建一个布尔类型的掩码数组，初始化为全 False
        mask = np.zeros(idx.shape, dtype=bool)
        # 将掩码的前三个位置设置为 True
        mask[0:3] = True

        # 调用 putmask 方法，根据掩码替换 IntervalIndex 中的值
        result = idx.putmask(mask, idx[-1])
        # 创建一个期望的 IntervalIndex 对象，预期结果中的前三个值与替换的值相同
        expected = IntervalIndex([idx[-1]] * 3 + list(idx[3:]))
        # 使用 assert_index_equal 断言两个 IntervalIndex 对象是否相等
        tm.assert_index_equal(result, expected)

    def test_putmask_td64(self):
        # 创建一个 DatetimeIndex 对象，从指定日期开始，生成一系列日期时间
        dti = date_range("2016-01-01", periods=9)
        # 计算日期时间之间的时间差，并创建一个 IntervalIndex 对象
        tdi = dti - dti[0]
        idx = IntervalIndex.from_breaks(tdi)
        # 创建一个布尔类型的掩码数组，初始化为全 False
        mask = np.zeros(idx.shape, dtype=bool)
        # 将掩码的前三个位置设置为 True
        mask[0:3] = True

        # 调用 putmask 方法，根据掩码替换 IntervalIndex 中的值
        result = idx.putmask(mask, idx[-1])
        # 创建一个期望的 IntervalIndex 对象，预期结果中的前三个值与替换的值相同
        expected = IntervalIndex([idx[-1]] * 3 + list(idx[3:]))
        # 使用 assert_index_equal 断言两个 IntervalIndex 对象是否相等
        tm.assert_index_equal(result, expected)


class TestContains:
    # .__contains__, not .contains
    # 定义一个测试方法，用于测试 IntervalIndex 对象的 __contains__ 方法
    def test_contains_dunder(self):
        # 从指定的两个数组创建 IntervalIndex 对象，closed 参数设置为 "right"
        index = IntervalIndex.from_arrays([0, 1], [1, 2], closed="right")

        # __contains__ 方法要求完全匹配区间才返回 True
        assert 0 not in index  # 检查是否不包含整数 0
        assert 1 not in index  # 检查是否不包含整数 1
        assert 2 not in index  # 检查是否不包含整数 2

        # 检查是否包含 Interval(0, 1, closed="right") 这个区间对象
        assert Interval(0, 1, closed="right") in index
        # 检查是否不包含 Interval(0, 2, closed="right") 这个区间对象
        assert Interval(0, 2, closed="right") not in index
        # 检查是否不包含 Interval(0, 0.5, closed="right") 这个区间对象
        assert Interval(0, 0.5, closed="right") not in index
        # 检查是否不包含 Interval(3, 5, closed="right") 这个区间对象
        assert Interval(3, 5, closed="right") not in index
        # 检查是否不包含 Interval(-1, 0, closed="left") 这个区间对象
        assert Interval(-1, 0, closed="left") not in index
        # 检查是否不包含 Interval(0, 1, closed="left") 这个区间对象
        assert Interval(0, 1, closed="left") not in index
        # 检查是否不包含 Interval(0, 1, closed="both") 这个区间对象
        assert Interval(0, 1, closed="both") not in index
```