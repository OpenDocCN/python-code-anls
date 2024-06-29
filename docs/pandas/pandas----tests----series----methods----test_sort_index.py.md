# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_sort_index.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于编写和运行测试用例

from pandas import (  # 从 Pandas 库中导入以下模块
    DatetimeIndex,  # 日期时间索引
    IntervalIndex,  # 区间索引
    MultiIndex,  # 多重索引
    Series,  # 系列数据结构
)
import pandas._testing as tm  # 导入 Pandas 内部测试工具模块


@pytest.fixture(params=["quicksort", "mergesort", "heapsort", "stable"])
def sort_kind(request):
    return request.param  # 使用参数化夹具，返回请求的排序方式参数


class TestSeriesSortIndex:
    def test_sort_index_name(self, datetime_series):
        result = datetime_series.sort_index(ascending=False)
        assert result.name == datetime_series.name  # 断言排序后的索引名称与原始系列的名称相同

    def test_sort_index(self, datetime_series):
        datetime_series.index = datetime_series.index._with_freq(None)

        rindex = list(datetime_series.index)
        np.random.default_rng(2).shuffle(rindex)  # 随机打乱索引顺序

        random_order = datetime_series.reindex(rindex)  # 按新顺序重新索引
        sorted_series = random_order.sort_index()  # 默认升序排序
        tm.assert_series_equal(sorted_series, datetime_series)  # 断言排序后的系列与原始系列相等

        # 降序排序
        sorted_series = random_order.sort_index(ascending=False)
        tm.assert_series_equal(
            sorted_series, datetime_series.reindex(datetime_series.index[::-1])
        )  # 断言降序排序后的系列与原始系列逆序索引相等

        # 按级别兼容性排序
        sorted_series = random_order.sort_index(level=0)
        tm.assert_series_equal(sorted_series, datetime_series)  # 断言按级别排序后的系列与原始系列相等

        # 按轴兼容性排序
        sorted_series = random_order.sort_index(axis=0)
        tm.assert_series_equal(sorted_series, datetime_series)  # 断言按轴排序后的系列与原始系列相等

        msg = "No axis named 1 for object type Series"
        with pytest.raises(ValueError, match=msg):
            random_order.sort_values(axis=1)  # 对不存在的轴进行排序，预期抛出 ValueError 异常

        sorted_series = random_order.sort_index(level=0, axis=0)
        tm.assert_series_equal(sorted_series, datetime_series)  # 断言按级别和轴排序后的系列与原始系列相等

        with pytest.raises(ValueError, match=msg):
            random_order.sort_index(level=0, axis=1)  # 对不存在的轴进行排序，预期抛出 ValueError 异常

    def test_sort_index_inplace(self, datetime_series):
        datetime_series.index = datetime_series.index._with_freq(None)

        # 用于 GH#11402
        rindex = list(datetime_series.index)
        np.random.default_rng(2).shuffle(rindex)  # 随机打乱索引顺序

        # 降序排序
        random_order = datetime_series.reindex(rindex)
        result = random_order.sort_index(ascending=False, inplace=True)

        assert result is None  # 断言 inplace 操作返回 None
        expected = datetime_series.reindex(datetime_series.index[::-1])
        expected.index = expected.index._with_freq(None)
        tm.assert_series_equal(random_order, expected)  # 断言降序排序后的系列与预期结果相等

        # 升序排序
        random_order = datetime_series.reindex(rindex)
        result = random_order.sort_index(ascending=True, inplace=True)

        assert result is None  # 断言 inplace 操作返回 None
        expected = datetime_series.copy()
        expected.index = expected.index._with_freq(None)
        tm.assert_series_equal(random_order, expected)  # 断言升序排序后的系列与预期结果相等
    # 定义一个测试方法，用于测试多层索引排序的功能
    def test_sort_index_level(self):
        # 创建一个多层索引对象，索引由两个子列表组成，每个子列表代表一个层级的索引值
        mi = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list("ABC"))
        # 创建一个序列对象，使用上面创建的多层索引对象作为索引
        s = Series([1, 2], mi)
        # 创建一个反向排序的序列对象，选择两个特定位置的索引数据
        backwards = s.iloc[[1, 0]]

        # 按照指定的层级"A"对序列进行排序，返回排序后的结果
        res = s.sort_index(level="A")
        # 使用测试工具比较反向排序和排序后的结果序列是否相等
        tm.assert_series_equal(backwards, res)

        # 按照多个层级["A", "B"]对序列进行排序，返回排序后的结果
        res = s.sort_index(level=["A", "B"])
        # 使用测试工具比较反向排序和排序后的结果序列是否相等
        tm.assert_series_equal(backwards, res)

        # 按照层级"A"对序列进行排序，但不对剩余的层级进行排序，返回排序后的结果
        res = s.sort_index(level="A", sort_remaining=False)
        # 使用测试工具比较原始序列和排序后的结果序列是否相等
        tm.assert_series_equal(s, res)

        # 按照多个层级["A", "B"]对序列进行排序，但不对剩余的层级进行排序，返回排序后的结果
        res = s.sort_index(level=["A", "B"], sort_remaining=False)
        # 使用测试工具比较原始序列和排序后的结果序列是否相等
        tm.assert_series_equal(s, res)

    # 使用pytest的参数化功能，测试多层索引排序的不同参数设置
    @pytest.mark.parametrize("level", ["A", 0])  # GH#21052
    def test_sort_index_multiindex(self, level):
        # 创建一个多层索引对象，索引由两个子列表组成，每个子列表代表一个层级的索引值
        mi = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list("ABC"))
        # 创建一个序列对象，使用上面创建的多层索引对象作为索引
        s = Series([1, 2], mi)
        # 创建一个反向排序的序列对象，选择两个特定位置的索引数据
        backwards = s.iloc[[1, 0]]

        # 按照指定的层级(level参数)对序列进行排序，返回排序后的结果
        res = s.sort_index(level=level)
        # 使用测试工具比较反向排序和排序后的结果序列是否相等
        tm.assert_series_equal(backwards, res)

        # GH#13496
        # 当设置sort_remaining=False时，排序操作不会影响剩余的层级索引
        res = s.sort_index(level=level, sort_remaining=False)
        # 使用测试工具比较原始序列和排序后的结果序列是否相等
        tm.assert_series_equal(s, res)

    # 定义一个测试方法，用于测试根据不同的排序算法(kind参数)对序列进行排序
    def test_sort_index_kind(self, sort_kind):
        # 创建一个对象，索引为整数列表，数据类型为object
        series = Series(index=[3, 2, 1, 4, 3], dtype=object)
        # 创建一个期望的排序后的序列对象
        expected_series = Series(index=[1, 2, 3, 3, 4], dtype=object)

        # 根据指定的排序算法(kind参数)对序列进行排序，返回排序后的结果
        index_sorted_series = series.sort_index(kind=sort_kind)
        # 使用测试工具比较期望的排序结果和实际排序后的结果序列是否相等
        tm.assert_series_equal(expected_series, index_sorted_series)

    # 定义一个测试方法，用于测试在索引中包含NaN值时的排序操作
    def test_sort_index_na_position(self):
        # 创建一个对象，索引为整数列表和NaN，数据类型为object
        series = Series(index=[3, 2, 1, 4, 3, np.nan], dtype=object)
        # 创建期望的排序结果序列对象，将NaN值放在第一个位置
        expected_series_first = Series(index=[np.nan, 1, 2, 3, 3, 4], dtype=object)

        # 根据指定的NaN位置(na_position参数)对序列进行排序，返回排序后的结果
        index_sorted_series = series.sort_index(na_position="first")
        # 使用测试工具比较期望的排序结果和实际排序后的结果序列是否相等
        tm.assert_series_equal(expected_series_first, index_sorted_series)

        # 创建期望的排序结果序列对象，将NaN值放在最后一个位置
        expected_series_last = Series(index=[1, 2, 3, 3, 4, np.nan], dtype=object)

        # 根据指定的NaN位置(na_position参数)对序列进行排序，返回排序后的结果
        index_sorted_series = series.sort_index(na_position="last")
        # 使用测试工具比较期望的排序结果和实际排序后的结果序列是否相等
        tm.assert_series_equal(expected_series_last, index_sorted_series)

    # 定义一个测试方法，用于测试在区间索引对象上的排序操作
    def test_sort_index_intervals(self):
        # 创建一个序列对象，索引为区间索引对象，数据包含NaN值
        s = Series(
            [np.nan, 1, 2, 3], IntervalIndex.from_arrays([0, 1, 2, 3], [1, 2, 3, 4])
        )

        # 根据默认设置对序列进行排序，返回排序后的结果
        result = s.sort_index()
        # 创建期望的排序结果序列对象
        expected = s
        # 使用测试工具比较期望的排序结果和实际排序后的结果序列是否相等
        tm.assert_series_equal(result, expected)

        # 根据降序排序设置对序列进行排序，返回排序后的结果
        result = s.sort_index(ascending=False)
        # 创建期望的排序结果序列对象，同时更新区间索引的数组范围
        expected = Series(
            [3, 2, 1, np.nan], IntervalIndex.from_arrays([3, 2, 1, 0], [4, 3, 2, 1])
        )
        # 使用测试工具比较期望的排序结果和实际排序后的结果序列是否相等
        tm.assert_series_equal(result, expected)
    # 使用 pytest 的 parametrize 装饰器为 test_sort_index_ignore_index 方法提供多组参数化测试数据
    @pytest.mark.parametrize(
        "original_list, sorted_list, ascending, ignore_index, output_index",
        [
            ([2, 3, 6, 1], [2, 3, 6, 1], True, True, [0, 1, 2, 3]),  # 测试保持顺序且保留索引的情况
            ([2, 3, 6, 1], [2, 3, 6, 1], True, False, [0, 1, 2, 3]),  # 测试保持顺序且不保留索引的情况
            ([2, 3, 6, 1], [1, 6, 3, 2], False, True, [0, 1, 2, 3]),  # 测试逆序且保留索引的情况
            ([2, 3, 6, 1], [1, 6, 3, 2], False, False, [3, 2, 1, 0]),  # 测试逆序且不保留索引的情况
        ],
    )
    # 定义测试方法 test_sort_index_ignore_index，测试 pandas 的排序功能
    def test_sort_index_ignore_index(
        self, inplace, original_list, sorted_list, ascending, ignore_index, output_index
    ):
        # GH 30114
        # 创建 Series 对象 ser，用原始列表 original_list 初始化
        ser = Series(original_list)
        # 根据 sorted_list 和 output_index 创建期望的 Series 对象 expected
        expected = Series(sorted_list, index=output_index)
        # 构建排序所需的参数字典 kwargs
        kwargs = {
            "ascending": ascending,
            "ignore_index": ignore_index,
            "inplace": inplace,
        }

        # 如果 inplace 参数为 True，先复制 ser，再原地排序
        if inplace:
            result_ser = ser.copy()
            result_ser.sort_index(**kwargs)
        # 如果 inplace 参数为 False，返回排序后的新 Series
        else:
            result_ser = ser.sort_index(**kwargs)

        # 断言排序后的结果 result_ser 与期望结果 expected 相等
        tm.assert_series_equal(result_ser, expected)
        # 断言原始 Series ser 不受排序操作影响
        tm.assert_series_equal(ser, Series(original_list))

    # 定义测试方法 test_sort_index_ascending_list，测试多级索引下的排序功能
    def test_sort_index_ascending_list(self):
        # GH#16934

        # 设置一个包含三级 MultiIndex 的 Series
        arrays = [
            ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
            [4, 3, 2, 1, 4, 3, 2, 1],
        ]
        # 创建 MultiIndex 对象 mi
        tuples = zip(*arrays)
        mi = MultiIndex.from_tuples(tuples, names=["first", "second", "third"])
        ser = Series(range(8), index=mi)

        # 按 boolean 类型的 ascending 参数排序
        result = ser.sort_index(level=["third", "first"], ascending=False)
        # 创建预期的排序结果 expected
        expected = ser.iloc[[4, 0, 5, 1, 6, 2, 7, 3]]
        tm.assert_series_equal(result, expected)

        # 按 boolean 类型的列表 ascending 参数排序
        result = ser.sort_index(level=["third", "first"], ascending=[False, True])
        # 创建预期的排序结果 expected
        expected = ser.iloc[[0, 4, 1, 5, 2, 6, 3, 7]]
        tm.assert_series_equal(result, expected)

    # 使用 pytest 的 parametrize 装饰器为 test_sort_index_ascending_bad_value_raises 方法提供多组参数化测试数据
    @pytest.mark.parametrize(
        "ascending",
        [
            None,  # 测试 ascending 参数为 None 的情况
            (True, None),  # 测试 ascending 参数为 (True, None) 的情况
            (False, "True"),  # 测试 ascending 参数为 (False, "True") 的情况
        ],
    )
    # 定义测试方法 test_sort_index_ascending_bad_value_raises，测试当 ascending 参数类型不正确时是否会抛出 ValueError 异常
    def test_sort_index_ascending_bad_value_raises(self, ascending):
        # 创建一个带索引的 Series 对象 ser
        ser = Series(range(10), index=[0, 3, 2, 1, 4, 5, 7, 6, 8, 9])
        # 匹配的错误信息
        match = 'For argument "ascending" expected type bool'
        # 使用 pytest 的 assertRaises 检查是否抛出 ValueError 异常并验证错误信息
        with pytest.raises(ValueError, match=match):
            ser.sort_index(ascending=ascending)
class TestSeriesSortIndexKey:
    def test_sort_index_multiindex_key(self):
        # 创建一个多级索引对象，包含两个级别 A 和 B，并设置索引名为 ABC
        mi = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list("ABC"))
        # 创建一个 Series 对象，使用 mi 作为索引，数据为 [1, 2]
        s = Series([1, 2], mi)
        # 反向选择 s 的行，生成一个新的 Series 对象
        backwards = s.iloc[[1, 0]]

        # 按照 C 级别的值对 s 进行排序，使用 lambda 函数进行降序排序
        result = s.sort_index(level="C", key=lambda x: -x)
        tm.assert_series_equal(s, result)

        # 按照 C 级别的值对 s 进行排序，使用 lambda 函数进行升序排序，期望结果不变
        result = s.sort_index(level="C", key=lambda x: x)  # nothing happens
        tm.assert_series_equal(backwards, result)

    def test_sort_index_multiindex_key_multi_level(self):
        # 创建一个多级索引对象，包含两个级别 A 和 B，并设置索引名为 ABC
        mi = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list("ABC"))
        # 创建一个 Series 对象，使用 mi 作为索引，数据为 [1, 2]
        s = Series([1, 2], mi)
        # 反向选择 s 的行，生成一个新的 Series 对象
        backwards = s.iloc[[1, 0]]

        # 按照多级别 A 和 C 的值对 s 进行排序，使用 lambda 函数进行降序排序
        result = s.sort_index(level=["A", "C"], key=lambda x: -x)
        tm.assert_series_equal(s, result)

        # 按照多级别 A 和 C 的值对 s 进行排序，使用 lambda 函数进行升序排序，期望结果不变
        result = s.sort_index(level=["A", "C"], key=lambda x: x)  # nothing happens
        tm.assert_series_equal(backwards, result)

    def test_sort_index_key(self):
        # 创建一个 Series 对象，使用列表作为索引，数据为 [0, 1, 2, 3, 4, 5]
        series = Series(np.arange(6, dtype="int64"), index=list("aaBBca"))

        # 按索引排序，默认升序
        result = series.sort_index()
        expected = series.iloc[[2, 3, 0, 1, 5, 4]]
        tm.assert_series_equal(result, expected)

        # 按索引排序，使用 lambda 函数对索引的小写形式排序
        result = series.sort_index(key=lambda x: x.str.lower())
        expected = series.iloc[[0, 1, 5, 2, 3, 4]]
        tm.assert_series_equal(result, expected)

        # 按索引排序，使用 lambda 函数对索引的小写形式进行降序排序
        result = series.sort_index(key=lambda x: x.str.lower(), ascending=False)
        expected = series.iloc[[4, 2, 3, 0, 1, 5]]
        tm.assert_series_equal(result, expected)

    def test_sort_index_key_int(self):
        # 创建一个 Series 对象，索引和数据都是 [0, 1, 2, 3, 4, 5]
        series = Series(np.arange(6, dtype="int64"), index=np.arange(6, dtype="int64"))

        # 按索引排序，默认升序
        result = series.sort_index()
        tm.assert_series_equal(result, series)

        # 按索引排序，使用 lambda 函数对索引的负值进行排序，等同于降序排序
        result = series.sort_index(key=lambda x: -x)
        expected = series.sort_index(ascending=False)
        tm.assert_series_equal(result, expected)

        # 按索引排序，使用 lambda 函数对索引乘以 2 进行排序，期望结果不变
        result = series.sort_index(key=lambda x: 2 * x)
        tm.assert_series_equal(result, series)

    def test_sort_index_kind_key(self, sort_kind, sort_by_key):
        # GH #14444 & #13589:  Add support for sort algo choosing
        # 创建一个对象，索引为 [3, 2, 1, 4, 3]，数据类型为 object
        series = Series(index=[3, 2, 1, 4, 3], dtype=object)
        expected_series = Series(index=[1, 2, 3, 3, 4], dtype=object)

        # 使用 sort_kind 和 sort_by_key 进行索引排序
        index_sorted_series = series.sort_index(kind=sort_kind, key=sort_by_key)
        tm.assert_series_equal(expected_series, index_sorted_series)

    def test_sort_index_kind_neg_key(self, sort_kind):
        # GH #14444 & #13589:  Add support for sort algo choosing
        # 创建一个对象，索引为 [3, 2, 1, 4, 3]，数据类型为 object
        series = Series(index=[3, 2, 1, 4, 3], dtype=object)
        expected_series = Series(index=[4, 3, 3, 2, 1], dtype=object)

        # 使用 sort_kind 和 lambda 函数对索引的负值进行排序
        index_sorted_series = series.sort_index(kind=sort_kind, key=lambda x: -x)
        tm.assert_series_equal(expected_series, index_sorted_series)
    # 测试函数，用于测试根据索引排序并指定处理缺失值位置和键函数的功能
    def test_sort_index_na_position_key(self, sort_by_key):
        # 创建一个带有特定索引的 Series 对象，包括一个 NaN 值，数据类型为对象
        series = Series(index=[3, 2, 1, 4, 3, np.nan], dtype=object)
        # 创建预期的排序后的 Series 对象，首个位置是 NaN 值
        expected_series_first = Series(index=[np.nan, 1, 2, 3, 3, 4], dtype=object)

        # 根据索引排序 Series，将 NaN 值放在首位，并使用给定的键函数排序
        index_sorted_series = series.sort_index(na_position="first", key=sort_by_key)
        # 断言排序后的 Series 与预期的 Series 相等
        tm.assert_series_equal(expected_series_first, index_sorted_series)

        # 创建另一个预期的排序后的 Series 对象，最后一个位置是 NaN 值
        expected_series_last = Series(index=[1, 2, 3, 3, 4, np.nan], dtype=object)

        # 根据索引排序 Series，将 NaN 值放在末尾，并使用给定的键函数排序
        index_sorted_series = series.sort_index(na_position="last", key=sort_by_key)
        # 断言排序后的 Series 与预期的 Series 相等
        tm.assert_series_equal(expected_series_last, index_sorted_series)

    # 测试函数，测试当排序索引时，使用不同的键函数引发异常
    def test_changes_length_raises(self):
        # 创建一个简单的 Series 对象
        s = Series([1, 2, 3])
        # 使用 pytest 检查排序索引时是否引发 ValueError 异常，并验证异常消息
        with pytest.raises(ValueError, match="change the shape"):
            s.sort_index(key=lambda x: x[:1])

    # 测试函数，测试根据键函数对 Series 进行排序
    def test_sort_values_key_type(self):
        # 创建一个带有日期时间索引的 Series 对象
        s = Series([1, 2, 3], index=DatetimeIndex(["2008-10-24", "2008-11-23", "2007-12-22"]))

        # 根据月份对 Series 进行排序
        result = s.sort_index(key=lambda x: x.month)
        expected = s.iloc[[0, 1, 2]]
        tm.assert_series_equal(result, expected)

        # 根据日期对 Series 进行排序
        result = s.sort_index(key=lambda x: x.day)
        expected = s.iloc[[2, 1, 0]]
        tm.assert_series_equal(result, expected)

        # 根据年份对 Series 进行排序
        result = s.sort_index(key=lambda x: x.year)
        expected = s.iloc[[2, 0, 1]]
        tm.assert_series_equal(result, expected)

        # 根据月份名称对 Series 进行排序
        result = s.sort_index(key=lambda x: x.month_name())
        expected = s.iloc[[2, 1, 0]]
        tm.assert_series_equal(result, expected)

    # 使用 pytest 参数化装饰器标记的测试函数，测试已经单调排序的多级索引
    @pytest.mark.parametrize(
        "ascending",
        [
            [True, False],
            [False, True],
        ],
    )
    def test_sort_index_multi_already_monotonic(self, ascending):
        # 创建一个多级索引对象
        mi = MultiIndex.from_product([[1, 2], [3, 4]])
        # 创建一个带有多级索引的 Series 对象
        ser = Series(range(len(mi)), index=mi)
        # 根据索引排序 Series，根据指定的升序或降序排序顺序
        result = ser.sort_index(ascending=ascending)
        # 根据不同的排序顺序设置预期的 Series
        if ascending == [True, False]:
            expected = ser.take([1, 0, 3, 2])
        elif ascending == [False, True]:
            expected = ser.take([2, 3, 0, 1])
        # 断言排序后的 Series 与预期的 Series 相等
        tm.assert_series_equal(result, expected)
```