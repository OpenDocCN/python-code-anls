# `D:\src\scipysrc\pandas\pandas\tests\indexing\test_at.py`

```
# 从 datetime 模块中导入 datetime 和 timezone 类
from datetime import (
    datetime,
    timezone,
)

# 导入 numpy 库，并使用别名 np
import numpy as np

# 导入 pytest 库，用于测试
import pytest

# 从 pandas.errors 模块中导入 InvalidIndexError 类
from pandas.errors import InvalidIndexError

# 从 pandas 库中导入多个类和函数
from pandas import (
    CategoricalDtype,
    CategoricalIndex,
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    Timestamp,
)

# 导入 pandas._testing 模块，并使用别名 tm
import pandas._testing as tm

# 定义测试函数 test_at_timezone
def test_at_timezone():
    # 创建包含 datetime 对象的 DataFrame
    result = DataFrame({"foo": [datetime(2000, 1, 1)]})
    
    # 使用 tm.assert_produces_warning 断言捕获 FutureWarning，并匹配给定字符串
    with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
        # 在 DataFrame 的特定位置设置带有时区信息的 datetime 对象
        result.at[0, "foo"] = datetime(2000, 1, 2, tzinfo=timezone.utc)
    
    # 创建期望的 DataFrame 对象，包含带有时区信息的 datetime 对象
    expected = DataFrame(
        {"foo": [datetime(2000, 1, 2, tzinfo=timezone.utc)]}, dtype=object
    )
    
    # 使用 tm.assert_frame_equal 断言两个 DataFrame 对象相等
    tm.assert_frame_equal(result, expected)

# 定义测试函数 test_selection_methods_of_assigned_col
def test_selection_methods_of_assigned_col():
    # 创建包含两列数据的 DataFrame 对象
    df = DataFrame(data={"a": [1, 2, 3], "b": [4, 5, 6]})
    
    # 创建包含一列数据的 DataFrame 对象，并指定索引
    df2 = DataFrame(data={"c": [7, 8, 9]}, index=[2, 1, 0])
    
    # 将 df2 的 "c" 列数据赋值给 df 的 "c" 列
    df["c"] = df2["c"]
    
    # 使用 at 方法在指定位置设置新值
    df.at[1, "c"] = 11
    
    # 将结果赋值给 result 变量
    result = df
    
    # 创建期望的 DataFrame 对象，包含更新后的值
    expected = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [9, 11, 7]})
    
    # 使用 tm.assert_frame_equal 断言两个 DataFrame 对象相等
    tm.assert_frame_equal(result, expected)
    
    # 获取 df 中指定位置的值，并赋值给 result 变量
    result = df.at[1, "c"]
    
    # 使用 assert 断言结果是否符合预期
    assert result == 11
    
    # 获取 df 的 "c" 列，并赋值给 result 变量
    result = df["c"]
    
    # 创建期望的 Series 对象，包含更新后的 "c" 列数据
    expected = Series([9, 11, 7], name="c")
    
    # 使用 tm.assert_series_equal 断言两个 Series 对象相等
    tm.assert_series_equal(result, expected)
    
    # 获取 df 的包含 "c" 列的 DataFrame 对象，并赋值给 result 变量
    result = df[["c"]]
    
    # 创建期望的 DataFrame 对象，包含更新后的 "c" 列数据
    expected = DataFrame({"c": [9, 11, 7]})
    
    # 使用 tm.assert_frame_equal 断言两个 DataFrame 对象相等
    tm.assert_frame_equal(result, expected)

# 定义 TestAtSetItem 类
class TestAtSetItem:
    # 定义测试方法 test_at_setitem_item_cache_cleared
    def test_at_setitem_item_cache_cleared(self):
        # 创建一个只有一个索引为 0 的 DataFrame 对象
        df = DataFrame(index=[0])
        
        # 在 DataFrame 中添加 "x" 和 "cost" 列，并设置初始值
        df["x"] = 1
        df["cost"] = 2
        
        # 访问 df["cost"]，将其添加到 _item_cache 中
        df["cost"]
        
        # 使用 loc 方法查询索引为 [0] 的数据，并触发 _item_cache
        df.loc[[0]]
        
        # 使用 at 方法分别设置指定位置的值
        df.at[0, "x"] = 4
        df.at[0, "cost"] = 789
        
        # 创建期望的 DataFrame 对象，包含更新后的值
        expected = DataFrame(
            {"x": [4], "cost": 789},
            index=[0],
            columns=Index(["x", "cost"], dtype=object),
        )
        
        # 使用 tm.assert_frame_equal 断言两个 DataFrame 对象相等
        tm.assert_frame_equal(df, expected)
        
        # 使用 tm.assert_series_equal 断言 Series 对象相等，检查 _item_cache 是否正确更新
        tm.assert_series_equal(df["cost"], expected["cost"])

    # 定义测试方法 test_at_setitem_mixed_index_assignment
    def test_at_setitem_mixed_index_assignment(self):
        # 创建带有混合索引的 Series 对象
        ser = Series([1, 2, 3, 4, 5], index=["a", "b", "c", 1, 2])
        
        # 使用 at 方法设置指定索引位置的值
        ser.at["a"] = 11
        
        # 使用 assert 断言获取的值是否符合预期
        assert ser.iat[0] == 11
        
        # 使用 at 方法设置指定索引位置的值
        ser.at[1] = 22
        
        # 使用 assert 断言获取的值是否符合预期
        assert ser.iat[3] == 22
    # 定义测试方法：测试在分类数据框中使用 `at` 方法设置缺失值的行为
    def test_at_setitem_categorical_missing(self):
        # 创建一个包含分类数据类型的 DataFrame，索引为 0 到 2，列也为 0 到 2
        df = DataFrame(
            index=range(3), columns=range(3), dtype=CategoricalDtype(["foo", "bar"])
        )
        # 使用 `at` 方法设置索引为 1，列为 1 的位置为 "foo"
        df.at[1, 1] = "foo"

        # 创建预期的 DataFrame，包含 NaN 值，数据类型为分类数据类型
        expected = DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [np.nan, "foo", np.nan],
                [np.nan, np.nan, np.nan],
            ],
            dtype=CategoricalDtype(["foo", "bar"]),
        )

        # 使用 `assert_frame_equal` 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

    # 定义测试方法：测试在多级索引中使用 `at` 方法设置值的行为
    def test_at_setitem_multiindex(self):
        # 创建一个包含 int64 类型数据的 DataFrame，形状为 3x2，列为多级索引 [("a", 0), ("a", 1)]
        df = DataFrame(
            np.zeros((3, 2), dtype="int64"),
            columns=MultiIndex.from_tuples([("a", 0), ("a", 1)]),
        )
        # 使用 `at` 方法设置索引为 0，列为 "a" 的位置为 10
        df.at[0, "a"] = 10

        # 创建预期的 DataFrame，其中第一行的 "a" 列值为 10，其余为 0
        expected = DataFrame(
            [[10, 10], [0, 0], [0, 0]],
            columns=MultiIndex.from_tuples([("a", 0), ("a", 1)]),
        )

        # 使用 `assert_frame_equal` 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

    # 使用 pytest 的参数化装饰器定义测试方法，测试在日期时间索引中使用 `at` 方法设置值的行为
    @pytest.mark.parametrize("row", (Timestamp("2019-01-01"), "2019-01-01"))
    def test_at_datetime_index(self, row):
        # 创建一个包含日期时间索引和 float64 数据类型的 DataFrame
        df = DataFrame(
            data=[[1] * 2], index=DatetimeIndex(data=["2019-01-01", "2019-01-02"])
        ).astype({0: "float64"})
        # 创建预期的 DataFrame，设置第一列中 "2019-01-01" 对应的值为 0.5
        expected = DataFrame(
            data=[[0.5, 1], [1.0, 1]],
            index=DatetimeIndex(data=["2019-01-01", "2019-01-02"]),
        )

        # 使用 `at` 方法设置行为 row，列为 0 的值为 0.5
        df.at[row, 0] = 0.5
        # 使用 `assert_frame_equal` 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)
class TestAtSetItemWithExpansion:
    # 定义测试类 TestAtSetItemWithExpansion，测试 Series 的 at 操作扩展性
    def test_at_setitem_expansion_series_dt64tz_value(self, tz_naive_fixture):
        # GH#25506
        # 如果 tz_naive_fixture 不为空，则创建带有时区的 Timestamp 对象，否则创建不带时区的 Timestamp 对象
        ts = (
            Timestamp("2017-08-05 00:00:00+0100", tz=tz_naive_fixture)
            if tz_naive_fixture is not None
            else Timestamp("2017-08-05 00:00:00+0100")
        )
        # 创建一个 Series 对象，包含 ts 对象作为数据
        result = Series(ts)
        # 在索引 1 处插入 ts 对象
        result.at[1] = ts
        # 创建一个预期的 Series 对象，包含两个 ts 对象
        expected = Series([ts, ts])
        # 断言 result 和 expected 相等
        tm.assert_series_equal(result, expected)


class TestAtWithDuplicates:
    # 定义测试类 TestAtWithDuplicates，测试 DataFrame 中带有重复标签的 at 操作
    def test_at_with_duplicate_axes_requires_scalar_lookup(self):
        # GH#33041 检查在回退到 loc 操作时不允许非标量参数的情况

        # 创建一个随机数组，初始化一个 DataFrame，包含重复的列标签 "A"
        arr = np.random.default_rng(2).standard_normal(6).reshape(3, 2)
        df = DataFrame(arr, columns=["A", "A"])

        msg = "Invalid call for scalar access"
        # 使用 pytest 检查索引操作的 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            df.at[[1, 2]]
        with pytest.raises(ValueError, match=msg):
            df.at[1, ["A"]]
        with pytest.raises(ValueError, match=msg):
            df.at[:, "A"]

        with pytest.raises(ValueError, match=msg):
            df.at[[1, 2]] = 1
        with pytest.raises(ValueError, match=msg):
            df.at[1, ["A"]] = 1
        with pytest.raises(ValueError, match=msg):
            df.at[:, "A"] = 1


class TestAtErrors:
    # TODO: De-duplicate/parametrize
    #  test_at_series_raises_key_error2, test_at_frame_raises_key_error2

    def test_at_series_raises_key_error(self, indexer_al):
        # GH#31724 .at 应该与 .loc 一致

        # 创建一个带有索引的 Series 对象
        ser = Series([1, 2, 3], index=[3, 2, 1])
        # 使用 indexer_al 函数获取索引 1 对应的值
        result = indexer_al(ser)[1]
        assert result == 3

        # 使用 pytest 检查 KeyErro 异常，消息包含 "a"
        with pytest.raises(KeyError, match="a"):
            indexer_al(ser)["a"]

    def test_at_frame_raises_key_error(self, indexer_al):
        # GH#31724 .at 应该与 .loc 一致

        # 创建一个带有索引和列的 DataFrame 对象
        df = DataFrame({0: [1, 2, 3]}, index=[3, 2, 1])

        # 使用 indexer_al 函数获取索引 (1, 0) 对应的值
        result = indexer_al(df)[1, 0]
        assert result == 3

        # 使用 pytest 检查 KeyErro 异常，消息包含 "a"
        with pytest.raises(KeyError, match="a"):
            indexer_al(df)["a", 0]

        # 使用 pytest 检查 KeyErro 异常，消息包含 "a"
        with pytest.raises(KeyError, match="a"):
            indexer_al(df)[1, "a"]

    def test_at_series_raises_key_error2(self, indexer_al):
        # at 操作不应该回退
        # GH#7814
        # GH#31724 .at 应该与 .loc 一致

        # 创建一个带有索引的 Series 对象
        ser = Series([1, 2, 3], index=list("abc"))
        # 使用 indexer_al 函数获取索引 "a" 对应的值
        result = indexer_al(ser)["a"]
        assert result == 1

        # 使用 pytest 检查 KeyErro 异常，消息以 "^0$" 开头
        with pytest.raises(KeyError, match="^0$"):
            indexer_al(ser)[0]

    def test_at_frame_raises_key_error2(self, indexer_al):
        # GH#31724 .at 应该与 .loc 一致

        # 创建一个带有索引和列的 DataFrame 对象
        df = DataFrame({"A": [1, 2, 3]}, index=list("abc"))
        # 使用 indexer_al 函数获取索引 ("a", "A") 对应的值
        result = indexer_al(df)["a", "A"]
        assert result == 1

        # 使用 pytest 检查 KeyErro 异常，消息以 "^0$" 开头
        with pytest.raises(KeyError, match="^0$"):
            indexer_al(df)["a", 0]
    # 测试函数：test_at_frame_multiple_columns
    def test_at_frame_multiple_columns(self):
        # GH#48296 - at shouldn't modify multiple columns
        # 创建一个包含两列的DataFrame，每列各有两个元素
        df = DataFrame({"a": [1, 2], "b": [3, 4]})
        # 新行数据
        new_row = [6, 7]
        # 使用pytest断言，验证设置非法索引时是否会引发InvalidIndexError异常
        with pytest.raises(
            InvalidIndexError,
            match=f"You can only assign a scalar value not a \\{type(new_row)}",
        ):
            # 尝试在索引为5的位置设置新行数据
            df.at[5] = new_row

    # 测试函数：test_at_getitem_mixed_index_no_fallback
    def test_at_getitem_mixed_index_no_fallback(self):
        # GH#19860
        # 创建一个包含混合索引的Series
        ser = Series([1, 2, 3, 4, 5], index=["a", "b", "c", 1, 2])
        # 使用pytest断言，验证当索引为0时是否会引发KeyError异常
        with pytest.raises(KeyError, match="^0$"):
            ser.at[0]
        # 使用pytest断言，验证当索引为4时是否会引发KeyError异常
        with pytest.raises(KeyError, match="^4$"):
            ser.at[4]

    # 测试函数：test_at_categorical_integers
    def test_at_categorical_integers(self):
        # CategoricalIndex具有整数类别，但这些类别不匹配Categorical的代码
        # 创建一个CategoricalIndex，包含整数类别[3, 4]
        ci = CategoricalIndex([3, 4])

        # 创建一个2x2的二维数组
        arr = np.arange(4).reshape(2, 2)
        # 创建一个DataFrame，使用ci作为索引
        frame = DataFrame(arr, index=ci)

        # 遍历frame和其转置，验证使用at访问不存在的整数索引时是否会引发KeyError异常
        for df in [frame, frame.T]:
            for key in [0, 1]:
                with pytest.raises(KeyError, match=str(key)):
                    df.at[key, key]

    # 测试函数：test_at_applied_for_rows
    def test_at_applied_for_rows(self):
        # GH#48729 .at should raise InvalidIndexError when assigning rows
        # 创建一个具有单行的DataFrame，设置了index和columns
        df = DataFrame(index=["a"], columns=["col1", "col2"])
        # 新行数据
        new_row = [123, 15]
        # 使用pytest断言，验证设置非法行索引时是否会引发InvalidIndexError异常
        with pytest.raises(
            InvalidIndexError,
            match=f"You can only assign a scalar value not a \\{type(new_row)}",
        ):
            # 尝试在索引为"a"的位置设置新行数据
            df.at["a"] = new_row
```