# `D:\src\scipysrc\pandas\pandas\tests\indexes\ranges\test_range.py`

```
# 导入必要的库
import numpy as np  # 导入 NumPy 库
import pytest  # 导入 Pytest 库用于单元测试

# 从 Pandas 库中导入需要的模块和函数
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
    Index,
    RangeIndex,
)
import pandas._testing as tm
from pandas.core.indexes.range import min_fitting_element

# 定义测试类 TestRangeIndex
class TestRangeIndex:
    # 定义 Pytest 的 fixture 函数 simple_index，返回一个 RangeIndex 对象
    @pytest.fixture
    def simple_index(self):
        return RangeIndex(start=0, stop=20, step=2)

    # 测试 RangeIndex 构造函数是否正确解析索引
    def test_constructor_unwraps_index(self):
        result = RangeIndex(1, 3)
        expected = np.array([1, 2], dtype=np.int64)
        tm.assert_numpy_array_equal(result._data, expected)

    # 测试 RangeIndex 是否能够容纳标识符
    def test_can_hold_identifiers(self, simple_index):
        idx = simple_index
        key = idx[0]
        assert idx._can_hold_identifiers_and_holds_name(key) is False

    # 测试设置过多名称时是否引发 ValueError 异常
    def test_too_many_names(self, simple_index):
        index = simple_index
        with pytest.raises(ValueError, match="^Length"):
            index.names = ["roger", "harold"]

    # 使用参数化测试多个不同的 RangeIndex 实例，验证其 start、stop 和 step 属性
    @pytest.mark.parametrize(
        "index, start, stop, step",
        [
            (RangeIndex(5), 0, 5, 1),
            (RangeIndex(0, 5), 0, 5, 1),
            (RangeIndex(5, step=2), 0, 5, 2),
            (RangeIndex(1, 5, 2), 1, 5, 2),
        ],
    )
    def test_start_stop_step_attrs(self, index, start, stop, step):
        # GH 25710
        assert index.start == start
        assert index.stop == stop
        assert index.step == step

    # 测试 RangeIndex 的复制方法
    def test_copy(self):
        i = RangeIndex(5, name="Foo")
        i_copy = i.copy()
        assert i_copy is not i
        assert i_copy.identical(i)
        assert i_copy._range == range(0, 5, 1)
        assert i_copy.name == "Foo"

    # 测试 RangeIndex 的字符串表示形式
    def test_repr(self):
        i = RangeIndex(5, name="Foo")
        result = repr(i)
        expected = "RangeIndex(start=0, stop=5, step=1, name='Foo')"
        assert result == expected

        result = eval(result)
        tm.assert_index_equal(result, i, exact=True)

        i = RangeIndex(5, 0, -1)
        result = repr(i)
        expected = "RangeIndex(start=5, stop=0, step=-1)"
        assert result == expected

        result = eval(result)
        tm.assert_index_equal(result, i, exact=True)

    # 测试 RangeIndex 的插入方法
    def test_insert(self):
        idx = RangeIndex(5, name="Foo")
        result = idx[1:4]

        # 测试插入第一个元素
        tm.assert_index_equal(idx[0:4], result.insert(0, idx[0]), exact="equiv")

        # GH 18295 (测试缺失)
        expected = Index([0, np.nan, 1, 2, 3, 4], dtype=np.float64)
        for na in [np.nan, None, pd.NA]:
            result = RangeIndex(5).insert(1, na)
            tm.assert_index_equal(result, expected)

        result = RangeIndex(5).insert(1, pd.NaT)
        expected = Index([0, pd.NaT, 1, 2, 3, 4], dtype=object)
        tm.assert_index_equal(result, expected)
    # 定义测试方法，验证插入操作对 RangeIndex 的保持
    def test_insert_edges_preserves_rangeindex(self):
        # 创建一个 RangeIndex 对象，范围为 4 到 9（不包括），步长为 2
        idx = Index(range(4, 9, 2))

        # 在索引 0 处插入值 2
        result = idx.insert(0, 2)
        # 预期结果是一个 RangeIndex 对象，范围为 2 到 9（不包括），步长为 2
        expected = Index(range(2, 9, 2))
        # 使用 assert_index_equal 函数比较 result 和 expected 是否相等
        tm.assert_index_equal(result, expected, exact=True)

        # 再次进行插入操作，在索引 3 处插入值 10
        result = idx.insert(3, 10)
        # 预期结果是一个 RangeIndex 对象，范围为 4 到 11（不包括），步长为 2
        expected = Index(range(4, 11, 2))
        # 使用 assert_index_equal 函数比较 result 和 expected 是否相等
        tm.assert_index_equal(result, expected, exact=True)

    # 定义测试方法，验证插入操作在 RangeIndex 中间的保持
    def test_insert_middle_preserves_rangeindex(self):
        # 创建一个 RangeIndex 对象，范围为 0 到 3（不包括），步长为 2
        idx = Index(range(0, 3, 2))
        # 在索引 1 处插入值 1
        result = idx.insert(1, 1)
        # 预期结果是一个 RangeIndex 对象，范围为 0 到 3（不包括）
        expected = Index(range(3))
        # 使用 assert_index_equal 函数比较 result 和 expected 是否相等
        tm.assert_index_equal(result, expected, exact=True)

        # 对 idx 执行乘法操作，将所有值乘以 2
        idx = idx * 2
        # 再次进行插入操作，在索引 1 处插入值 2
        result = idx.insert(1, 2)
        # 预期结果是一个 RangeIndex 对象，范围为 0 到 6（不包括）
        expected = expected * 2
        # 使用 assert_index_equal 函数比较 result 和 expected 是否相等
        tm.assert_index_equal(result, expected, exact=True)

    # 定义测试方法，验证删除操作的效果
    def test_delete(self):
        # 创建一个 RangeIndex 对象，范围为 0 到 5，命名为 "Foo"
        idx = RangeIndex(5, name="Foo")

        # 预期结果是 idx 的第 1 位开始的子序列
        expected = idx[1:]
        # 删除索引 0 处的值
        result = idx.delete(0)
        # 使用 assert_index_equal 函数比较 result 和 expected 是否相等
        tm.assert_index_equal(result, expected, exact=True)
        # 断言 result 的名称与 expected 的名称相同
        assert result.name == expected.name

        # 预期结果是 idx 的最后一位之前的子序列
        expected = idx[:-1]
        # 删除最后一位处的值
        result = idx.delete(-1)
        # 使用 assert_index_equal 函数比较 result 和 expected 是否相等
        tm.assert_index_equal(result, expected, exact=True)
        # 断言 result 的名称与 expected 的名称相同
        assert result.name == expected.name

        # 预期的错误消息，当索引超出范围时，应该引发 IndexError 或 ValueError
        msg = "index 5 is out of bounds for axis 0 with size 5"
        # 使用 pytest.raises 检查是否引发了预期的异常，并匹配错误消息
        with pytest.raises((IndexError, ValueError), match=msg):
            # 删除超出范围的索引
            result = idx.delete(len(idx))

    # 定义测试方法，验证删除操作对 RangeIndex 的保持
    def test_delete_preserves_rangeindex(self):
        # 创建一个 RangeIndex 对象，范围为 0 到 1，命名为 "foo"
        idx = Index(range(2), name="foo")

        # 删除索引 1 处的值
        result = idx.delete([1])
        # 预期结果是一个 RangeIndex 对象，范围为 0 到 0（不包括），命名为 "foo"
        expected = Index(range(1), name="foo")
        # 使用 assert_index_equal 函数比较 result 和 expected 是否相等
        tm.assert_index_equal(result, expected, exact=True)

        # 再次进行删除操作，删除索引 1 处的值
        result = idx.delete(1)
        # 使用 assert_index_equal 函数比较 result 和 expected 是否相等
        tm.assert_index_equal(result, expected, exact=True)

    # 定义测试方法，验证删除操作在 RangeIndex 中间的保持
    def test_delete_preserves_rangeindex_middle(self):
        # 创建一个 RangeIndex 对象，范围为 0 到 2，命名为 "foo"
        idx = Index(range(3), name="foo")
        # 删除索引 1 处的值
        result = idx.delete(1)
        # 预期结果是一个 RangeIndex 对象，包含索引 0 和 2
        expected = idx[::2]
        # 使用 assert_index_equal 函数比较 result 和 expected 是否相等
        tm.assert_index_equal(result, expected, exact=True)

        # 再次进行删除操作，删除倒数第二个索引处的值
        result = idx.delete(-2)
        # 使用 assert_index_equal 函数比较 result 和 expected 是否相等
        tm.assert_index_equal(result, expected, exact=True)

    # 定义测试方法，验证删除操作在 RangeIndex 列表末尾的保持
    def test_delete_preserves_rangeindex_list_at_end(self):
        # 创建一个 RangeIndex 对象，范围为 0 到 6，步长为 1
        idx = RangeIndex(0, 6, 1)

        # 要删除的索引位置列表
        loc = [2, 3, 4, 5]
        # 删除 loc 列表中指定的索引位置处的值
        result = idx.delete(loc)
        # 预期结果是一个 RangeIndex 对象，范围为 0 到 2，步长为 1
        expected = idx[:2]
        # 使用 assert_index_equal 函数比较 result 和 expected 是否相等
        tm.assert_index_equal(result, expected, exact=True)

        # 再次进行删除操作，删除 loc 列表（倒序）中指定的索引位置处的值
        result = idx.delete(loc[::-1])
        # 使用 assert_index_equal 函数比较 result 和 expected 是否相等
        tm.assert_index_equal(result, expected, exact=True)

    # 定义测试方法，验证删除操作在 RangeIndex 列表中间的保持
    def test_delete_preserves_rangeindex_list_middle(self):
        # 创建一个 RangeIndex 对象，范围为 0 到 6，步长为 1
        idx = RangeIndex(0, 6, 1)

        # 要删除的索引位置列表
        loc = [1, 2, 3, 4]
        # 删除 loc 列表中指定的索引位置处的值
        result = idx.delete(loc)
        # 预期结果是一个 RangeIndex 对象，范围为 0 到 6，步长为 5
        expected = RangeIndex(0, 6, 5)
        # 使用 assert_index_equal 函数比较 result 和 expected 是否相等
        tm.assert_index_equal(result, expected, exact=True)

        # 再次进行删除操作，删除 loc 列表（倒序）中指定的索引位置处的值
        result = idx.delete(loc[::-1])
        # 使用 assert_index_equal 函数比较 result 和 expected 是否相等
        tm.assert_index_equal(result, expected, exact=True)
    # 定义测试方法，验证删除操作保留 RangeIndex 的行为
    def test_delete_all_preserves_rangeindex(self):
        # 创建 RangeIndex 对象，范围从 0 到 5，步长为 1
        idx = RangeIndex(0, 6, 1)

        # 指定要删除的位置列表
        loc = [0, 1, 2, 3, 4, 5]
        # 执行删除操作
        result = idx.delete(loc)
        # 预期结果是一个空的 RangeIndex
        expected = idx[:0]
        # 验证删除后的结果与预期结果相等
        tm.assert_index_equal(result, expected, exact=True)

        # 反向删除操作
        result = idx.delete(loc[::-1])
        # 验证删除后的结果与预期结果相等
        tm.assert_index_equal(result, expected, exact=True)

    # 定义测试方法，验证删除操作不保留 RangeIndex 的行为
    def test_delete_not_preserving_rangeindex(self):
        # 创建 RangeIndex 对象，范围从 0 到 5，步长为 1
        idx = RangeIndex(0, 6, 1)

        # 指定要删除的位置列表
        loc = [0, 3, 5]
        # 执行删除操作
        result = idx.delete(loc)
        # 预期结果是一个普通的 Index 对象，包含删除后的索引值
        expected = Index([1, 2, 4])
        # 验证删除后的结果与预期结果相等
        tm.assert_index_equal(result, expected, exact=True)

        # 反向删除操作
        result = idx.delete(loc[::-1])
        # 验证删除后的结果与预期结果相等
        tm.assert_index_equal(result, expected, exact=True)

    # 定义测试方法，验证 RangeIndex 的视图功能
    def test_view(self):
        # 创建一个命名为 "Foo" 的 RangeIndex 对象
        i = RangeIndex(0, name="Foo")
        # 创建该 RangeIndex 对象的视图
        i_view = i.view()
        # 断言视图的名称与原始 RangeIndex 对象的名称相同
        assert i_view.name == "Foo"

        # 创建特定类型的视图
        i_view = i.view("i8")
        # 验证视图的值与原始 RangeIndex 对象的值相等
        tm.assert_numpy_array_equal(i.values, i_view)

    # 定义测试方法，验证索引对象的数据类型
    def test_dtype(self, simple_index):
        # 获取一个简单的索引对象
        index = simple_index
        # 断言索引对象的数据类型是 np.int64
        assert index.dtype == np.int64

    # 定义测试方法，验证 RangeIndex 的缓存功能
    def test_cache(self):
        # GH 26565, GH26617, GH35432, GH53387
        # 此测试检查 _cache 是否已设置。
        # 调用 RangeIndex._cache["_data"] 创建与 RangeIndex 长度相同的 int64 数组，并将其存储在 _cache 中。
        idx = RangeIndex(0, 100, 10)

        # 断言初始时 _cache 为空字典
        assert idx._cache == {}

        # 调用 repr 方法不会修改索引缓存
        repr(idx)
        assert idx._cache == {}

        # 调用 str 方法不会修改索引缓存
        str(idx)
        assert idx._cache == {}

        # 调用 get_loc 方法不会修改索引缓存
        idx.get_loc(20)
        assert idx._cache == {}

        # 使用 in 操作符查找索引值 90，不会修改索引缓存
        90 in idx  # True
        assert idx._cache == {}

        # 使用 in 操作符查找索引值 91，不会修改索引缓存
        91 in idx  # False
        assert idx._cache == {}

        # 调用 all 方法不会修改索引缓存
        idx.all()
        assert idx._cache == {}

        # 调用 any 方法不会修改索引缓存
        idx.any()
        assert idx._cache == {}

        # 遍历索引对象不会修改索引缓存
        for _ in idx:
            pass
        assert idx._cache == {}

        # 创建一个包含 RangeIndex 的 DataFrame 对象
        df = pd.DataFrame({"a": range(10)}, index=idx)

        # 调用 str 方法不会修改索引缓存
        str(df)
        assert idx._cache == {}

        # 通过 loc 方法访问 DataFrame 的特定行，不会修改索引缓存
        df.loc[50]
        assert idx._cache == {}

        # 通过 loc 方法访问 DataFrame 的不存在行，不会修改索引缓存
        with pytest.raises(KeyError, match="51"):
            df.loc[51]
        assert idx._cache == {}

        # 通过 loc 方法切片访问 DataFrame，不会修改索引缓存
        df.loc[10:50]
        assert idx._cache == {}

        # 通过 iloc 方法切片访问 DataFrame，不会修改索引缓存
        df.iloc[5:10]
        assert idx._cache == {}

        # 调用 take 方法可能会修改索引缓存的其它键，但不包括 "_data"
        idx.take([3, 0, 1])
        assert "_data" not in idx._cache

        # 通过 loc 方法访问单个元素，不会修改索引缓存的 "_data"
        df.loc[[50]]
        assert "_data" not in idx._cache

        # 通过 iloc 方法访问多个元素，不会修改索引缓存的 "_data"
        df.iloc[[5, 6, 7, 8, 9]]
        assert "_data" not in idx._cache

        # 调用 _data 属性后，索引缓存应包含 "_data" 条目
        idx._data
        assert isinstance(idx._data, np.ndarray)
        assert idx._data is idx._data  # 检查缓存的值是否被重用
        assert "_data" in idx._cache
        expected = np.arange(0, 100, 10, dtype="int64")
        tm.assert_numpy_array_equal(idx._cache["_data"], expected)
    # 定义测试方法，用于测试 RangeIndex 对象的单调性

    # 创建一个 RangeIndex 对象，从 0 到 20（不包括），步长为 2
    index = RangeIndex(0, 20, 2)
    # 断言该索引是否单调递增
    assert index.is_monotonic_increasing is True
    # 再次断言该索引是否单调递增
    assert index.is_monotonic_increasing is True
    # 断言该索引不是单调递减
    assert index.is_monotonic_decreasing is False
    # 断言该索引是否严格单调递增
    assert index._is_strictly_monotonic_increasing is True
    # 断言该索引不是严格单调递减
    assert index._is_strictly_monotonic_decreasing is False

    # 创建一个 RangeIndex 对象，从 4 到 0（不包括），步长为 -1
    index = RangeIndex(4, 0, -1)
    # 断言该索引不是单调递增
    assert index.is_monotonic_increasing is False
    # 断言该索引不是严格单调递增
    assert index._is_strictly_monotonic_increasing is False
    # 断言该索引是单调递减
    assert index.is_monotonic_decreasing is True
    # 断言该索引是严格单调递减
    assert index._is_strictly_monotonic_decreasing is True

    # 创建一个 RangeIndex 对象，从 1 到 2（不包括）
    index = RangeIndex(1, 2)
    # 断言该索引是单调递增
    assert index.is_monotonic_increasing is True
    # 再次断言该索引是单调递增
    assert index.is_monotonic_increasing is True
    # 断言该索引是单调递减
    assert index.is_monotonic_decreasing is True
    # 断言该索引是严格单调递增
    assert index._is_strictly_monotonic_increasing is True
    # 断言该索引是严格单调递减
    assert index._is_strictly_monotonic_decreasing is True

    # 创建一个 RangeIndex 对象，从 2 到 1（不包括）
    index = RangeIndex(2, 1)
    # 断言该索引是单调递增
    assert index.is_monotonic_increasing is True
    # 再次断言该索引是单调递增
    assert index.is_monotonic_increasing is True
    # 断言该索引是单调递减
    assert index.is_monotonic_decreasing is True
    # 断言该索引是严格单调递增
    assert index._is_strictly_monotonic_increasing is True
    # 断言该索引是严格单调递减
    assert index._is_strictly_monotonic_decreasing is True

    # 创建一个 RangeIndex 对象，从 1 到 1（不包括）
    index = RangeIndex(1, 1)
    # 断言该索引是单调递增
    assert index.is_monotonic_increasing is True
    # 再次断言该索引是单调递增
    assert index.is_monotonic_increasing is True
    # 断言该索引是单调递减
    assert index.is_monotonic_decreasing is True
    # 断言该索引是严格单调递增
    assert index._is_strictly_monotonic_increasing is True
    # 断言该索引是严格单调递减
    assert index._is_strictly_monotonic_decreasing is True

    # 使用 pytest 的 parametrize 标记参数化测试
    @pytest.mark.parametrize(
        "left,right",
        [
            # 比较两个 RangeIndex 对象，从 0 到 9（不包括），步长为 2 和从 0 到 10（不包括），步长为 2
            (RangeIndex(0, 9, 2), RangeIndex(0, 10, 2)),
            # 比较两个 RangeIndex 对象，从 0 到 1（不包括） 和从 1 到 -1（步长为 3）
            (RangeIndex(0), RangeIndex(1, -1, 3)),
            # 比较两个 RangeIndex 对象，从 1 到 2（不包括），步长为 3 和从 1 到 3（不包括），步长为 4
            (RangeIndex(1, 2, 3), RangeIndex(1, 3, 4)),
            # 比较两个 RangeIndex 对象，从 0 到 -9（步长为 -2） 和从 0 到 -10（步长为 -2）
            (RangeIndex(0, -9, -2), RangeIndex(0, -10, -2)),
        ],
    )
    # 定义测试方法，用于测试 equals 方法
    def test_equals_range(self, left, right):
        # 断言两个 RangeIndex 对象相等
        assert left.equals(right)
        # 再次断言两个 RangeIndex 对象相等

        assert right.equals(left)

    # 测试逻辑兼容性方法，接受 simple_index 作为参数
    def test_logical_compat(self, simple_index):
        # 使用简单索引对象 simple_index，断言其所有元素与其值的所有元素相等
        idx = simple_index
        assert idx.all() == idx.values.all()
        # 断言简单索引对象 simple_index 是否有任意元素
        assert idx.any() == idx.values.any()

    # 测试 identical 方法，接受 simple_index 作为参数
    def test_identical(self, simple_index):
        # 创建索引对象 index，复制 simple_index
        index = simple_index
        # 创建索引对象 i，复制 index
        i = Index(index.copy())
        # 断言 i 和 index 是相同的
        assert i.identical(index)

        # 如果 simple_index 是 RangeIndex 类型，则不允许对象数据类型为 object
        if isinstance(index, RangeIndex):
            return

        # 创建同值但不同类型的索引对象 same_values_different_type
        same_values_different_type = Index(i, dtype=object)
        # 断言 i 和 same_values_different_type 不是相同的
        assert not i.identical(same_values_different_type)

        # 将 index 复制为对象数据类型为 object 的索引对象 i
        i = index.copy(dtype=object)
        # 将索引对象 i 重命名为 "foo"
        i = i.rename("foo")
        # 创建同值但数据类型为 object 的索引对象 same_values
        same_values = Index(i, dtype=object)
        # 断言 same_values 和 index 是相同的
        assert same_values.identical(index.copy(dtype=object))

        # 断言 i 和 index 不是相同的
        assert not i.identical(index)
        # 断言与 same_values 相同值但名称为 "foo" 和数据类型为 object 的索引对象 i 是相同的
        assert Index(same_values, name="foo", dtype=object).identical(i)

        # 断言对象数据类型为 object 的 index 和数据类型为 int64 的 index 不是相同的
        assert not index.copy(dtype=object).identical(index.copy(dtype="int64"))
    # 测试 RangeIndex 对象的 nbytes 方法，验证内存节省量是否大于使用整数索引的对象的十分之一
    def test_nbytes(self):
        idx = RangeIndex(0, 1000)
        # 断言 RangeIndex 对象的 nbytes 小于使用整数索引的对象转换为 Index 后的 nbytes 的十分之一
        assert idx.nbytes < Index(idx._values).nbytes / 10

        # 验证当 RangeIndex 对象的范围较小时，其 nbytes 与另一个较小范围的 RangeIndex 对象的 nbytes 相等
        i2 = RangeIndex(0, 10)
        assert idx.nbytes == i2.nbytes

    # 使用 pytest 的参数化功能，测试 RangeIndex 对象在无法或不应该转换时的行为
    @pytest.mark.parametrize(
        "start,stop,step",
        [
            # 测试无法转换的情况
            ("foo", "bar", "baz"),
            # 测试不应该转换的情况
            ("0", "1", "2"),
        ],
    )
    def test_cant_or_shouldnt_cast(self, start, stop, step):
        msg = f"Wrong type {type(start)} for value {start}"
        # 使用 pytest 的 raises 方法断言在创建 RangeIndex 对象时会抛出 TypeError 异常，异常信息需匹配特定消息
        with pytest.raises(TypeError, match=msg):
            RangeIndex(start, stop, step)

    # 测试阻止在索引对象上执行数据类型转换的行为
    def test_view_index(self, simple_index):
        index = simple_index
        msg = (
            "Cannot change data-type for array of references.|"
            "Cannot change data-type for object array.|"
        )
        # 使用 pytest 的 raises 方法断言在尝试将索引对象视图转换为 Index 类型时会抛出 TypeError 异常，异常信息需匹配特定消息
        with pytest.raises(TypeError, match=msg):
            index.view(Index)

    # 测试阻止将索引对象转换为不同数据类型的行为
    def test_prevent_casting(self, simple_index):
        index = simple_index
        # 验证在将索引对象转换为对象类型（'O'）后，其数据类型确实为 numpy 的对象类型
        result = index.astype("O")
        assert result.dtype == np.object_

    # 测试索引对象在使用 repr 和 eval 转换后能够保持一致性
    def test_repr_roundtrip(self, simple_index):
        index = simple_index
        # 使用 assert_index_equal 函数验证索引对象经过 repr 和 eval 转换后与原始对象保持相等
        tm.assert_index_equal(eval(repr(index)), index)

    # 测试索引对象切片后名称保持不变的行为
    def test_slice_keep_name(self):
        idx = RangeIndex(1, 2, name="asdf")
        # 验证对索引对象进行切片后，切片对象的名称与原始对象的名称保持一致
        assert idx.name == idx[1:].name

    # 使用 pytest 的参数化功能，测试具有不同属性的 RangeIndex 对象是否存在重复值
    @pytest.mark.parametrize(
        "index",
        [
            RangeIndex(start=0, stop=20, step=2, name="foo"),
            RangeIndex(start=18, stop=-1, step=-2, name="bar"),
        ],
        ids=["index_inc", "index_dec"],
    )
    def test_has_duplicates(self, index):
        # 断言测试的 RangeIndex 对象没有重复值
        assert index.is_unique
        # 断言测试的 RangeIndex 对象不存在重复值
        assert not index.has_duplicates

    # 测试 RangeIndex 对象的扩展欧几里得算法
    def test_extended_gcd(self, simple_index):
        index = simple_index
        # 验证扩展欧几里得算法能够正确计算给定的数的最大公约数和系数
        result = index._extended_gcd(6, 10)
        assert result[0] == result[1] * 6 + result[2] * 10
        assert 2 == result[0]

        result = index._extended_gcd(10, 6)
        assert 2 == result[1] * 10 + result[2] * 6
        assert 2 == result[0]

    # 测试 min_fitting_element 函数的行为
    def test_min_fitting_element(self):
        # 验证 min_fitting_element 函数能够返回在给定参数范围内的最小适配元素
        result = min_fitting_element(0, 2, 1)
        assert 2 == result

        result = min_fitting_element(1, 1, 1)
        assert 1 == result

        result = min_fitting_element(18, -2, 1)
        assert 2 == result

        result = min_fitting_element(5, -1, 1)
        assert 1 == result

        big_num = 500000000000000000000000
        result = min_fitting_element(5, 1, big_num)
        assert big_num == result
    # 定义一个测试方法，用于测试特定索引对象的切片操作
    def test_slice_specialised(self, simple_index):
        # 将传入的简单索引对象赋值给本地变量 index
        index = simple_index
        # 设置索引对象的名称为 "foo"
        index.name = "foo"

        # 标量索引
        # 获取索引位置为 1 的元素
        res = index[1]
        expected = 2
        # 断言获取的结果与期望值相等
        assert res == expected

        # 获取倒数第一个元素
        res = index[-1]
        expected = 18
        assert res == expected

        # 切片操作
        # 对整个索引对象进行切片
        index_slice = index[:]
        expected = index
        # 使用测试框架的函数验证切片结果与期望值相等
        tm.assert_index_equal(index_slice, expected)

        # 正向切片
        # 获取索引从第 7 到第 10 个元素，步长为 2
        index_slice = index[7:10:2]
        expected = Index([14, 18], name="foo")
        tm.assert_index_equal(index_slice, expected, exact="equiv")

        # 负向切片
        # 获取索引从倒数第一个到倒数第五个元素，步长为 -2
        index_slice = index[-1:-5:-2]
        expected = Index([18, 14], name="foo")
        tm.assert_index_equal(index_slice, expected, exact="equiv")

        # 超出索引范围的停止位置
        # 获取索引从第 2 到第 100 个元素，步长为 4
        index_slice = index[2:100:4]
        expected = Index([4, 12], name="foo")
        tm.assert_index_equal(index_slice, expected, exact="equiv")

        # 反向切片
        # 对整个索引对象进行反向切片
        index_slice = index[::-1]
        expected = Index(index.values[::-1], name="foo")
        tm.assert_index_equal(index_slice, expected, exact="equiv")

        # 从倒数第八个元素开始反向切片
        index_slice = index[-8::-1]
        expected = Index([4, 2, 0], name="foo")
        tm.assert_index_equal(index_slice, expected, exact="equiv")

        # 从超出索引范围的位置开始反向切片
        index_slice = index[-40::-1]
        expected = Index(np.array([], dtype=np.int64), name="foo")
        tm.assert_index_equal(index_slice, expected, exact="equiv")

        # 从超出索引范围的位置开始正向切片
        index_slice = index[40::-1]
        expected = Index(index.values[40::-1], name="foo")
        tm.assert_index_equal(index_slice, expected, exact="equiv")

        # 从第 10 个元素开始反向切片
        index_slice = index[10::-1]
        expected = Index(index.values[::-1], name="foo")
        tm.assert_index_equal(index_slice, expected, exact="equiv")

    # 使用参数化装饰器指定 step 参数的取值范围，但不包括 0
    @pytest.mark.parametrize("step", set(range(-5, 6)) - {0})
    # 定义一个测试方法，用于测试特定步长的 RangeIndex 对象的长度计算
    def test_len_specialised(self, step):
        # 确保我们的长度与 np.arange 函数计算的结果一致
        start, stop = (0, 5) if step > 0 else (5, 0)

        # 使用 np.arange 函数生成一个数组 arr
        arr = np.arange(start, stop, step)
        # 创建一个 RangeIndex 对象 index
        index = RangeIndex(start, stop, step)
        # 断言索引对象的长度与数组 arr 的长度相等
        assert len(index) == len(arr)

        # 创建一个反向的 RangeIndex 对象 index
        index = RangeIndex(stop, start, step)
        # 断言反向索引对象的长度为 0
        assert len(index) == 0
    @pytest.mark.parametrize(
        "indices, expected",
        [
            # 参数化测试数据，每个元组包含索引对象和期望的结果对象
            ([RangeIndex(1, 12, 5)], RangeIndex(1, 12, 5)),
            ([RangeIndex(0, 6, 4)], RangeIndex(0, 6, 4)),
            ([RangeIndex(1, 3), RangeIndex(3, 7)], RangeIndex(1, 7)),
            ([RangeIndex(1, 5, 2), RangeIndex(5, 6)], RangeIndex(1, 6, 2)),
            ([RangeIndex(1, 3, 2), RangeIndex(4, 7, 3)], RangeIndex(1, 7, 3)),
            ([RangeIndex(-4, 3, 2), RangeIndex(4, 7, 2)], RangeIndex(-4, 7, 2)),
            ([RangeIndex(-4, -8), RangeIndex(-8, -12)], RangeIndex(0, 0)),
            ([RangeIndex(-4, -8), RangeIndex(3, -4)], RangeIndex(0, 0)),
            ([RangeIndex(-4, -8), RangeIndex(3, 5)], RangeIndex(3, 5)),
            ([RangeIndex(-4, -2), RangeIndex(3, 5)], Index([-4, -3, 3, 4])),
            ([RangeIndex(-2), RangeIndex(3, 5)], RangeIndex(3, 5)),
            ([RangeIndex(2), RangeIndex(2)], Index([0, 1, 0, 1])),
            ([RangeIndex(2), RangeIndex(2, 5), RangeIndex(5, 8, 4)], RangeIndex(0, 6)),
            (
                [RangeIndex(2), RangeIndex(3, 5), RangeIndex(5, 8, 4)],
                Index([0, 1, 3, 4, 5]),
            ),
            (
                [RangeIndex(-2, 2), RangeIndex(2, 5), RangeIndex(5, 8, 4)],
                RangeIndex(-2, 6),
            ),
            ([RangeIndex(3), Index([-1, 3, 15])], Index([0, 1, 2, -1, 3, 15])),
            ([RangeIndex(3), Index([-1, 3.1, 15.0])], Index([0, 1, 2, -1, 3.1, 15.0])),
            ([RangeIndex(3), Index(["a", None, 14])], Index([0, 1, 2, "a", None, 14])),
            ([RangeIndex(3, 1), Index(["a", None, 14])], Index(["a", None, 14])),
        ],
    )
    # 测试方法：验证 append 方法的行为是否符合预期
    def test_append(self, indices, expected):
        # GH16212
        # 调用 append 方法并验证结果是否与期望相等
        result = indices[0].append(indices[1:])
        tm.assert_index_equal(result, expected, exact=True)

        if len(indices) == 2:
            # Append 单个项而不是列表
            result2 = indices[0].append(indices[1])
            tm.assert_index_equal(result2, expected, exact=True)

    # 测试方法：验证无引擎查找的行为
    def test_engineless_lookup(self):
        # GH 16685
        # RangeIndex 上的标准查找不需要创建引擎
        idx = RangeIndex(2, 10, 3)

        # 断言：确保查找到的位置为预期值
        assert idx.get_loc(5) == 1
        # 使用 assert_numpy_array_equal 验证 get_indexer 方法的行为是否符合预期
        tm.assert_numpy_array_equal(
            idx.get_indexer([2, 8]), ensure_platform_int(np.array([0, 2]))
        )
        # 使用 pytest.raises 检查是否抛出预期的 KeyError 异常
        with pytest.raises(KeyError, match="3"):
            idx.get_loc(3)

        # 断言：确保在缓存中没有 _engine 属性
        assert "_engine" not in idx._cache

        # 不同类型的标量可以立即排除，无需使用 _engine
        with pytest.raises(KeyError, match="'a'"):
            idx.get_loc("a")

        # 断言：确保在缓存中没有 _engine 属性
        assert "_engine" not in idx._cache

    @pytest.mark.parametrize(
        "ri",
        [
            RangeIndex(0, -1, -1),
            RangeIndex(0, 1, 1),
            RangeIndex(1, 3, 2),
            RangeIndex(0, -1, -2),
            RangeIndex(-3, -5, -2),
        ],
    )
    # 定义测试方法，测试在空列表上调用 append 方法后的行为
    def test_append_len_one(self, ri):
        # GH39401: 测试用例编号
        result = ri.append([])
        # 断言结果与原始索引相等，确保精确匹配
        tm.assert_index_equal(result, ri, exact=True)

    # 参数化测试，验证 base 对象是否在指定范围内
    @pytest.mark.parametrize("base", [RangeIndex(0, 2), Index([0, 1])])
    def test_isin_range(self, base):
        # GH#41151: 测试用例编号
        values = RangeIndex(0, 1)
        # 检查 base 中的元素是否在 values 中出现
        result = base.isin(values)
        expected = np.array([True, False])
        # 断言 numpy 数组的相等性
        tm.assert_numpy_array_equal(result, expected)

    # 测试排序方法，使用指定的键排序 RangeIndex
    def test_sort_values_key(self):
        # GH#43666, GH#52764: 测试用例编号
        sort_order = {8: 2, 6: 0, 4: 8, 2: 10, 0: 12}
        values = RangeIndex(0, 10, 2)
        # 使用 lambda 函数对 values 进行排序，并获取排序后的结果
        result = values.sort_values(key=lambda x: x.map(sort_order))
        expected = Index([6, 8, 4, 2, 0], dtype="int64")
        # 断言排序后的结果与期望结果相等，确保精确匹配
        tm.assert_index_equal(result, expected, check_exact=True)

        # 验证结果与 Series.sort_values 方法的行为是否一致
        ser = values.to_series()
        result2 = ser.sort_values(key=lambda x: x.map(sort_order))
        # 断言 Series 排序后的结果与期望结果相等，确保精确匹配
        tm.assert_series_equal(result2, expected.to_series(), check_exact=True)

    # 测试 RangeIndex 减去常数的结果
    def test_range_index_rsub_by_const(self):
        # GH#53255: 测试用例编号
        result = 3 - RangeIndex(0, 4, 1)
        expected = RangeIndex(3, -1, -1)
        # 断言结果与期望的 RangeIndex 相等
        tm.assert_index_equal(result, expected)
@pytest.mark.parametrize(
    "rng, decimals",
    [
        [range(5), 0],  # 参数化测试数据：范围为 0 到 4，小数位数为 0
        [range(5), 2],  # 参数化测试数据：范围为 0 到 4，小数位数为 2
        [range(10, 30, 10), -1],  # 参数化测试数据：范围为 10 到 20，逆序，小数位数为 -1
        [range(30, 10, -10), -1],  # 参数化测试数据：范围为 30 到 20，逆序，小数位数为 -1
    ],
)
def test_range_round_returns_rangeindex(rng, decimals):
    ri = RangeIndex(rng)  # 创建 RangeIndex 对象
    expected = ri.copy()  # 复制 ri 对象作为预期结果
    result = ri.round(decimals=decimals)  # 调用 round 方法，对 ri 进行舍入操作
    tm.assert_index_equal(result, expected, exact=True)  # 断言结果与预期相等


@pytest.mark.parametrize(
    "rng, decimals",
    [
        [range(10, 30, 1), -1],  # 参数化测试数据：范围为 10 到 29，递增 1，小数位数为 -1
        [range(30, 10, -1), -1],  # 参数化测试数据：范围为 30 到 11，逆序，小数位数为 -1
        [range(11, 14), -10],  # 参数化测试数据：范围为 11 到 13，小数位数为 -10
    ],
)
def test_range_round_returns_index(rng, decimals):
    ri = RangeIndex(rng)  # 创建 RangeIndex 对象
    expected = Index(list(rng)).round(decimals=decimals)  # 创建预期结果的 Index 对象
    result = ri.round(decimals=decimals)  # 调用 round 方法，对 ri 进行舍入操作
    tm.assert_index_equal(result, expected, exact=True)  # 断言结果与预期相等


def test_reindex_1_value_returns_rangeindex():
    ri = RangeIndex(0, 10, 2, name="foo")  # 创建具有特定参数的 RangeIndex 对象
    result, result_indexer = ri.reindex([2])  # 调用 reindex 方法，返回重新索引的结果和索引器
    expected = RangeIndex(2, 4, 2, name="foo")  # 创建预期的 RangeIndex 对象
    tm.assert_index_equal(result, expected, exact=True)  # 断言结果与预期相等

    expected_indexer = np.array([1], dtype=np.intp)  # 预期的索引器数组
    tm.assert_numpy_array_equal(result_indexer, expected_indexer)  # 断言索引器数组与预期相等


def test_reindex_empty_returns_rangeindex():
    ri = RangeIndex(0, 10, 2, name="foo")  # 创建具有特定参数的 RangeIndex 对象
    result, result_indexer = ri.reindex([])  # 调用 reindex 方法，返回重新索引的结果和索引器
    expected = RangeIndex(0, 0, 2, name="foo")  # 创建预期的 RangeIndex 对象
    tm.assert_index_equal(result, expected, exact=True)  # 断言结果与预期相等

    expected_indexer = np.array([], dtype=np.intp)  # 预期的索引器数组（空数组）
    tm.assert_numpy_array_equal(result_indexer, expected_indexer)  # 断言索引器数组与预期相等


def test_insert_empty_0_loc():
    ri = RangeIndex(0, step=10, name="foo")  # 创建具有特定参数的 RangeIndex 对象
    result = ri.insert(0, 5)  # 调用 insert 方法，在位置 0 处插入值 5
    expected = RangeIndex(5, 15, 10, name="foo")  # 创建预期的 RangeIndex 对象
    tm.assert_index_equal(result, expected, exact=True)  # 断言结果与预期相等


def test_append_non_rangeindex_return_rangeindex():
    ri = RangeIndex(1)  # 创建具有起始值 1 的 RangeIndex 对象
    result = ri.append(Index([1]))  # 调用 append 方法，将 Index 对象 [1] 添加到 ri 中
    expected = RangeIndex(2)  # 创建预期的 RangeIndex 对象
    tm.assert_index_equal(result, expected, exact=True)  # 断言结果与预期相等


def test_append_non_rangeindex_return_index():
    ri = RangeIndex(1)  # 创建具有起始值 1 的 RangeIndex 对象
    result = ri.append(Index([1, 3, 4]))  # 调用 append 方法，将 Index 对象 [1, 3, 4] 添加到 ri 中
    expected = Index([0, 1, 3, 4])  # 创建预期的 Index 对象
    tm.assert_index_equal(result, expected, exact=True)  # 断言结果与预期相等


def test_reindex_returns_rangeindex():
    ri = RangeIndex(2, name="foo")  # 创建具有特定参数的 RangeIndex 对象
    result, result_indexer = ri.reindex([1, 2, 3])  # 调用 reindex 方法，返回重新索引的结果和索引器
    expected = RangeIndex(1, 4, name="foo")  # 创建预期的 RangeIndex 对象
    tm.assert_index_equal(result, expected, exact=True)  # 断言结果与预期相等

    expected_indexer = np.array([1, -1, -1], dtype=np.intp)  # 预期的索引器数组
    tm.assert_numpy_array_equal(result_indexer, expected_indexer)  # 断言索引器数组与预期相等


def test_reindex_returns_index():
    ri = RangeIndex(4, name="foo")  # 创建具有特定参数的 RangeIndex 对象
    result, result_indexer = ri.reindex([0, 1, 3])  # 调用 reindex 方法，返回重新索引的结果和索引器
    expected = Index([0, 1, 3], name="foo")  # 创建预期的 Index 对象
    tm.assert_index_equal(result, expected, exact=True)  # 断言结果与预期相等

    expected_indexer = np.array([0, 1, 3], dtype=np.intp)  # 预期的索引器数组
    tm.assert_numpy_array_equal(result_indexer, expected_indexer)  # 断言索引器数组与预期相等


def test_take_return_rangeindex():
    ri = RangeIndex(5, name="foo")  # 创建具有范围 0 到 4 的 RangeIndex 对象
    result = ri.take([])  # 调用 take 方法，获取空列表的结果
    expected = RangeIndex(0, name="foo")  # 创建预期的 RangeIndex 对象
    tm.assert_index_equal(result, expected, exact=True)  # 断言结果与预期相等
    # 调用 ri 对象的 take 方法，从中获取索引为 [3, 4] 的部分索引对象，存储在 result 变量中
    result = ri.take([3, 4])
    # 创建一个预期的 RangeIndex 对象，范围从 3 到 5（不包括 5），命名为 "foo"
    expected = RangeIndex(3, 5, name="foo")
    # 使用测试工具包中的 assert_index_equal 方法，确保 result 和 expected 的索引相等，精确匹配
    tm.assert_index_equal(result, expected, exact=True)
@pytest.mark.parametrize(  # 使用 pytest 提供的参数化功能，定义多组测试参数
    "rng, exp_rng",  # 定义测试函数的参数名称
    [  # 参数化的测试数据列表
        [range(5), range(3, 4)],  # 第一组测试数据：输入范围和期望的输出范围
        [range(0, -10, -2), range(-6, -8, -2)],  # 第二组测试数据
        [range(0, 10, 2), range(6, 8, 2)],  # 第三组测试数据
    ],
)
def test_take_1_value_returns_rangeindex(rng, exp_rng):  # 定义测试函数，验证取单个值时的行为
    ri = RangeIndex(rng, name="foo")  # 创建 RangeIndex 对象
    result = ri.take([3])  # 执行 take 方法，获取索引为 3 的值
    expected = RangeIndex(exp_rng, name="foo")  # 创建预期的 RangeIndex 对象
    tm.assert_index_equal(result, expected, exact=True)  # 使用 tm.assert_index_equal 断言结果与预期相等


def test_append_one_nonempty_preserve_step():  # 测试 append 方法在添加非空对象时的行为
    expected = RangeIndex(0, -1, -1)  # 创建预期的 RangeIndex 对象
    result = RangeIndex(0).append([expected])  # 调用 append 方法
    tm.assert_index_equal(result, expected, exact=True)  # 使用断言验证结果与预期相等


def test_getitem_boolmask_all_true():  # 测试使用布尔掩码获取所有为真的元素
    ri = RangeIndex(3, name="foo")  # 创建 RangeIndex 对象
    expected = ri.copy()  # 复制原始 RangeIndex 对象作为预期结果
    result = ri[[True] * 3]  # 使用布尔掩码获取所有元素
    tm.assert_index_equal(result, expected, exact=True)  # 使用断言验证结果与预期相等


def test_getitem_boolmask_all_false():  # 测试使用布尔掩码获取所有为假的元素
    ri = RangeIndex(3, name="foo")  # 创建 RangeIndex 对象
    result = ri[[False] * 3]  # 使用布尔掩码获取所有元素
    expected = RangeIndex(0, name="foo")  # 创建预期的 RangeIndex 对象
    tm.assert_index_equal(result, expected, exact=True)  # 使用断言验证结果与预期相等


def test_getitem_boolmask_returns_rangeindex():  # 测试使用布尔掩码获取 RangeIndex 对象
    ri = RangeIndex(3, name="foo")  # 创建 RangeIndex 对象
    result = ri[[False, True, True]]  # 使用布尔掩码获取指定元素
    expected = RangeIndex(1, 3, name="foo")  # 创建预期的 RangeIndex 对象
    tm.assert_index_equal(result, expected, exact=True)  # 使用断言验证结果与预期相等

    result = ri[[True, False, True]]  # 使用布尔掩码获取指定元素
    expected = RangeIndex(0, 3, 2, name="foo")  # 创建预期的 RangeIndex 对象
    tm.assert_index_equal(result, expected, exact=True)  # 使用断言验证结果与预期相等


def test_getitem_boolmask_returns_index():  # 测试使用布尔掩码获取 Index 对象
    ri = RangeIndex(4, name="foo")  # 创建 RangeIndex 对象
    result = ri[[True, True, False, True]]  # 使用布尔掩码获取指定元素
    expected = Index([0, 1, 3], name="foo")  # 创建预期的 Index 对象
    tm.assert_index_equal(result, expected)  # 使用断言验证结果与预期相等


def test_getitem_boolmask_wrong_length():  # 测试布尔掩码长度不匹配的情况
    ri = RangeIndex(4, name="foo")  # 创建 RangeIndex 对象
    with pytest.raises(IndexError, match="Boolean index has wrong length"):  # 使用 pytest 的断言捕获异常
        ri[[True]]  # 使用不匹配长度的布尔掩码


def test_pos_returns_rangeindex():  # 测试正号运算符返回 RangeIndex 对象
    ri = RangeIndex(2, name="foo")  # 创建 RangeIndex 对象
    expected = ri.copy()  # 复制原始 RangeIndex 对象作为预期结果
    result = +ri  # 执行正号运算
    tm.assert_index_equal(result, expected, exact=True)  # 使用断言验证结果与预期相等


def test_neg_returns_rangeindex():  # 测试负号运算符返回 RangeIndex 对象
    ri = RangeIndex(2, name="foo")  # 创建 RangeIndex 对象
    result = -ri  # 执行负号运算
    expected = RangeIndex(0, -2, -1, name="foo")  # 创建预期的 RangeIndex 对象
    tm.assert_index_equal(result, expected, exact=True)  # 使用断言验证结果与预期相等

    ri = RangeIndex(-2, 2, name="foo")  # 创建 RangeIndex 对象
    result = -ri  # 执行负号运算
    expected = RangeIndex(2, -2, -1, name="foo")  # 创建预期的 RangeIndex 对象
    tm.assert_index_equal(result, expected, exact=True)  # 使用断言验证结果与预期相等


@pytest.mark.parametrize(  # 使用 pytest 提供的参数化功能，定义多组测试参数
    "rng",  # 定义测试函数的参数名称
    [
        [range(0)],  # 第一组测试数据
        [range(10)],  # 第二组测试数据
        [range(-2, 1, 1)],  # 第三组测试数据
        [range(0, -10, -1)],  # 第四组测试数据
    ],
)
def test_abs_returns_rangeindex(rng):  # 测试绝对值函数返回 RangeIndex 对象
    ri = RangeIndex(rng, name="foo")  # 创建 RangeIndex 对象
    expected = RangeIndex(exp_rng, name="foo")  # 创建预期的 RangeIndex 对象
    result = abs(ri)  # 执行绝对值函数
    tm.assert_index_equal(result, expected, exact=True)  # 使用断言验证结果与预期相等


def test_abs_returns_index():  # 测试绝对值函数返回 Index 对象
    ri = RangeIndex(-2, 2, name="foo")  # 创建 RangeIndex 对象
    result = abs(ri)  # 执行绝对值函数
    expected = Index([2, 1, 0, 1], name="foo")  # 创建预期的 Index 对象
    tm.assert_index_equal(result, expected, exact=True)  # 使用断言验证结果与预期相等
    [
        # 空范围，不包含任何元素的迭代器
        range(0),
        # 从0到4的整数范围，不包括5
        range(5),
        # 从0到-5的整数范围，步长为-1，包括0但不包括-5
        range(0, -5, -1),
        # 从-2到1的整数范围，步长为1，包括-2但不包括2
        range(-2, 2, 1),
        # 从2到-2的整数范围，步长为-2，包括2但不包括-2
        range(2, -2, -2),
        # 从0到4的整数范围，步长为2，包括0但不包括5
        range(0, 5, 2),
    ],
# 定义测试函数，用于测试 RangeIndex 的反转操作是否返回正确的 RangeIndex 对象
def test_invert_returns_rangeindex(rng):
    # 创建一个 RangeIndex 对象 ri，并指定名称为 "foo"
    ri = RangeIndex(rng, name="foo")
    # 对 ri 进行按位取反操作
    result = ~ri
    # 断言 result 是 RangeIndex 类型的对象
    assert isinstance(result, RangeIndex)
    # 创建一个预期的 Index 对象 expected，对原始列表 rng 进行按位取反操作，并指定名称为 "foo"
    expected = ~Index(list(rng), name="foo")
    # 使用测试工具 tm 检查 result 是否与 expected 相等，允许非精确匹配
    tm.assert_index_equal(result, expected, exact=False)


# 使用 pytest 的 parametrize 装饰器，定义多组输入参数 rng
@pytest.mark.parametrize(
    "rng",
    [
        range(0, 5, 1),
        range(0, 5, 2),
        range(10, 15, 1),
        range(10, 5, -1),
        range(10, 5, -2),
        range(5, 0, -1),
    ],
)
# 使用 pytest 的 parametrize 装饰器，定义多组输入参数 meth
@pytest.mark.parametrize("meth", ["argmax", "argmin"])
# 定义测试函数，测试 RangeIndex 的 argmin 和 argmax 方法
def test_arg_min_max(rng, meth):
    # 创建一个 RangeIndex 对象 ri，使用给定的 rng
    ri = RangeIndex(rng)
    # 创建一个 Index 对象 idx，使用 rng 转换成列表
    idx = Index(list(rng))
    # 使用 getattr 函数调用 ri 和 idx 的 meth 方法，并断言它们的返回值相等
    assert getattr(ri, meth)() == getattr(idx, meth)()


# 定义测试函数，测试当 RangeIndex 为空时调用 argmin 和 argmax 方法是否会引发 ValueError 异常
@pytest.mark.parametrize("meth", ["argmin", "argmax"])
def test_empty_argmin_argmax_raises(meth):
    # 使用 pytest.raises 检查调用 RangeIndex(0) 的 meth 方法时是否抛出 ValueError 异常
    with pytest.raises(ValueError, match=f"attempt to get {meth} of an empty sequence"):
        getattr(RangeIndex(0), meth)()


# 定义测试函数，测试 RangeIndex 对象的 __getitem__ 方法返回 RangeIndex 对象的情况
def test_getitem_integers_return_rangeindex():
    # 对 RangeIndex 对象进行切片操作，选择索引为 [0, -1] 的元素
    result = RangeIndex(0, 10, 2, name="foo")[[0, -1]]
    # 创建一个预期的 RangeIndex 对象 expected，指定起始值、结束值、步长和名称
    expected = RangeIndex(start=0, stop=16, step=8, name="foo")
    # 使用测试工具 tm 检查 result 是否与 expected 相等，要求精确匹配
    tm.assert_index_equal(result, expected, exact=True)

    # 对 RangeIndex 对象进行切片操作，选择索引为 [3] 的元素
    result = RangeIndex(0, 10, 2, name="foo")[[3]]
    # 创建一个预期的 RangeIndex 对象 expected，指定起始值、结束值、步长和名称
    expected = RangeIndex(start=6, stop=8, step=2, name="foo")
    # 使用测试工具 tm 检查 result 是否与 expected 相等，要求精确匹配
    tm.assert_index_equal(result, expected, exact=True)


# 定义测试函数，测试 RangeIndex 对象的 __getitem__ 方法返回空的 RangeIndex 对象的情况
def test_getitem_empty_return_rangeindex():
    # 对 RangeIndex 对象进行切片操作，选择索引为空列表的元素
    result = RangeIndex(0, 10, 2, name="foo")[[]]
    # 创建一个预期的 RangeIndex 对象 expected，指定起始值、结束值、步长和名称
    expected = RangeIndex(start=0, stop=0, step=1, name="foo")
    # 使用测试工具 tm 检查 result 是否与 expected 相等，要求精确匹配
    tm.assert_index_equal(result, expected, exact=True)


# 定义测试函数，测试 RangeIndex 对象的 __getitem__ 方法返回 Index 对象的情况
def test_getitem_integers_return_index():
    # 对 RangeIndex 对象进行切片操作，选择索引为 [0, 1, -1] 的元素
    result = RangeIndex(0, 10, 2, name="foo")[[0, 1, -1]]
    # 创建一个预期的 Index 对象 expected，指定元素列表、数据类型、名称
    expected = Index([0, 2, 8], dtype="int64", name="foo")
    # 使用测试工具 tm 检查 result 是否与 expected 相等，不检查索引类型
    tm.assert_index_equal(result, expected)


# 使用 pytest 的 parametrize 装饰器，定义多组输入参数 normalize 和 rng
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize(
    "rng",
    [
        range(3),
        range(0),
        range(0, 3, 2),
        range(3, -3, -2),
    ],
)
# 定义测试函数，测试 RangeIndex 对象的 value_counts 方法
def test_value_counts(sort, dropna, ascending, normalize, rng):
    # 创建一个 RangeIndex 对象 ri，指定名称为 "A"，使用给定的 rng
    ri = RangeIndex(rng, name="A")
    # 调用 ri 的 value_counts 方法，传入参数 normalize、sort、ascending 和 dropna
    result = ri.value_counts(
        normalize=normalize, sort=sort, ascending=ascending, dropna=dropna
    )
    # 创建一个 Index 对象 expected，指定元素列表、名称，并调用其 value_counts 方法
    expected = Index(list(rng), name="A").value_counts(
        normalize=normalize, sort=sort, ascending=ascending, dropna=dropna
    )
    # 使用测试工具 tm 检查 result 是否与 expected 相等，不检查索引类型
    tm.assert_series_equal(result, expected, check_index_type=False)


# 使用 pytest 的 parametrize 装饰器，定义多组输入参数 side 和 value
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("value", [0, -5, 5, -3, np.array([-5, -3, 0, 5])])
# 定义测试函数，测试 RangeIndex 对象的 searchsorted 方法
def test_searchsorted(side, value):
    # 创建一个 RangeIndex 对象 ri，指定起始值 -3，结束值 3，步长 2
    ri = RangeIndex(-3, 3, 2)
    # 调用 ri 的 searchsorted 方法，传入参数 value 和 side
    result = ri.searchsorted(value=value, side=side)
    # 创建一个 Index 对象 expected，使用 ri 转换成列表，并调用其 searchsorted 方法
    expected = Index(list(ri)).searchsorted(value=value, side=side)
    # 如果 value 是整数，则断言 result 等于 expected
    if isinstance(value, int):
        assert result == expected
    else:
        # 否则，使用测试工具 tm 检查 result 是否与 expected 相等（numpy 数组比较）
        tm.assert_numpy_array_equal(result, expected)
```