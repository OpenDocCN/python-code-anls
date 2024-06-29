# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_rename_axis.py`

```
import pytest  # 导入 pytest 库

from pandas import (  # 从 pandas 库中导入 Index, MultiIndex, Series
    Index,
    MultiIndex,
    Series,
)
import pandas._testing as tm  # 导入 pandas 内部测试模块


class TestSeriesRenameAxis:
    def test_rename_axis_mapper(self):
        # GH 19978
        # 创建一个 MultiIndex，其中包含元素为 ["a", "b", "c"] 和 [1, 2] 的笛卡尔积，设置列名为 ["ll", "nn"]
        mi = MultiIndex.from_product([["a", "b", "c"], [1, 2]], names=["ll", "nn"])
        # 创建一个 Series，其索引为 mi，数值为 mi 的长度范围
        ser = Series(list(range(len(mi))), index=mi)

        # 使用 mapper 对索引进行重命名，将 "ll" 改为 "foo"
        result = ser.rename_axis(index={"ll": "foo"})
        # 断言结果索引的名称为 ["foo", "nn"]
        assert result.index.names == ["foo", "nn"]

        # 使用函数 str.upper 对索引进行重命名，axis=0 表示对行索引进行操作
        result = ser.rename_axis(index=str.upper, axis=0)
        # 断言结果索引的名称为 ["LL", "NN"]
        assert result.index.names == ["LL", "NN"]

        # 使用列表 ["foo", "goo"] 对索引进行重命名
        result = ser.rename_axis(index=["foo", "goo"])
        # 断言结果索引的名称为 ["foo", "goo"]
        assert result.index.names == ["foo", "goo"]

        # 使用 pytest 来断言是否抛出 TypeError，匹配异常消息为 "unexpected"
        with pytest.raises(TypeError, match="unexpected"):
            ser.rename_axis(columns="wrong")

    def test_rename_axis_inplace(self, datetime_series):
        # GH 15704
        # 对 datetime_series 应用 rename_axis 方法并赋值给 expected
        expected = datetime_series.rename_axis("foo")
        # 直接将 datetime_series 赋值给 result
        result = datetime_series
        # 调用 rename_axis 方法，并设置 inplace=True，不返回任何结果
        no_return = result.rename_axis("foo", inplace=True)

        # 断言 no_return 的返回值为 None
        assert no_return is None
        # 使用 pandas._testing 模块的 assert_series_equal 方法断言 result 和 expected 相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("kwargs", [{"mapper": None}, {"index": None}, {}])
    def test_rename_axis_none(self, kwargs):
        # GH 25034
        # 创建一个 Index，包含元素为 ["a", "b", "c"]，名称为 "foo"
        index = Index(list("abc"), name="foo")
        # 创建一个 Series，其索引为 index，数值为 [1, 2, 3]
        ser = Series([1, 2, 3], index=index)

        # 使用 kwargs 中的参数对索引进行重命名
        result = ser.rename_axis(**kwargs)
        # 如果 kwargs 不为空，则使用 index.rename(None) 重置索引名称，否则使用原始的 index
        expected_index = index.rename(None) if kwargs else index
        # 创建一个期望的 Series，索引为 expected_index，数值为 [1, 2, 3]
        expected = Series([1, 2, 3], index=expected_index)
        # 使用 pandas._testing 模块的 assert_series_equal 方法断言 result 和 expected 相等
        tm.assert_series_equal(result, expected)
```