# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_reset_index.py`

```
# 导入必要的模块和库
from datetime import datetime
from itertools import product

import numpy as np
import pytest

# 导入 pandas 相关模块和类
from pandas.core.dtypes.common import (
    is_float_dtype,
    is_integer_dtype,
)
import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    Index,
    Interval,
    IntervalIndex,
    MultiIndex,
    RangeIndex,
    Series,
    Timestamp,
    cut,
    date_range,
)
import pandas._testing as tm

# 定义 pytest 的 fixture 函数 multiindex_df
@pytest.fixture
def multiindex_df():
    # 创建一个多级索引 DataFrame
    levels = [["A", ""], ["B", "b"]]
    return DataFrame([[0, 2], [1, 3]], columns=MultiIndex.from_tuples(levels))

# 测试类 TestResetIndex
class TestResetIndex:
    # 测试方法 test_reset_index_empty_rangeindex
    def test_reset_index_empty_rangeindex(self):
        # GH#45230
        # 创建一个空的 RangeIndex 的 DataFrame
        df = DataFrame(
            columns=["brand"], dtype=np.int64, index=RangeIndex(0, 0, 1, name="foo")
        )
        # 对 df 设置新的索引为原索引和 "brand" 列
        df2 = df.set_index([df.index, "brand"])
        # 对 df2 进行索引重置，移除第一级索引
        result = df2.reset_index([1], drop=True)
        # 使用 pandas 测试工具断言结果与预期的空 DataFrame 相等，检查索引类型
        tm.assert_frame_equal(result, df[[]], check_index_type=True)

    # 测试方法 test_set_reset
    def test_set_reset(self):
        # 创建一个 Index 对象 idx，包含三个超大整数作为索引值
        idx = Index([2**63, 2**63 + 5, 2**63 + 10], name="foo")
        # 创建一个 DataFrame df，设置索引为 idx，包含一列数据 "A"
        df = DataFrame({"A": [0, 1, 2]}, index=idx)
        # 对 df 进行索引重置
        result = df.reset_index()
        # 断言结果 DataFrame 的 "foo" 列的数据类型为 uint64
        assert result["foo"].dtype == np.dtype("uint64")
        # 将结果 DataFrame 再次设置索引为 "foo" 列
        df = result.set_index("foo")
        # 使用 pandas 测试工具断言结果 DataFrame 的索引与原索引 idx 相等
        tm.assert_index_equal(df.index, idx)

    # 测试方法 test_set_index_reset_index_dt64tz
    def test_set_index_reset_index_dt64tz(self):
        # 创建一个具有时区信息的日期范围索引 idx
        idx = Index(date_range("20130101", periods=3, tz="US/Eastern"), name="foo")
        # 创建一个 DataFrame df，设置索引为 idx，包含一列数据 "A"
        df = DataFrame({"A": [0, 1, 2]}, index=idx)
        # 对 df 进行索引重置
        result = df.reset_index()
        # 断言结果 DataFrame 的 "foo" 列的数据类型为 datetime64[ns, US/Eastern]
        assert result["foo"].dtype == "datetime64[ns, US/Eastern]"
        # 将结果 DataFrame 再次设置索引为 "foo" 列
        df = result.set_index("foo")
        # 使用 pandas 测试工具断言结果 DataFrame 的索引与原索引 idx 相等
        tm.assert_index_equal(df.index, idx)

    # 测试方法 test_reset_index_tz
    def test_reset_index_tz(self, tz_aware_fixture):
        # GH 3950
        # 创建一个带有时区信息的日期范围索引 idx
        tz = tz_aware_fixture
        idx = date_range("1/1/2011", periods=5, freq="D", tz=tz, name="idx")
        # 创建一个 DataFrame df，包含两列数据 "a" 和 "b"，索引为 idx
        df = DataFrame({"a": range(5), "b": ["A", "B", "C", "D", "E"]}, index=idx)
        # 创建预期结果 DataFrame，包含三列 "idx"、"a"、"b"
        expected = DataFrame(
            {
                "idx": idx,
                "a": range(5),
                "b": ["A", "B", "C", "D", "E"],
            },
            columns=["idx", "a", "b"],
        )
        # 对 df 进行索引重置
        result = df.reset_index()
        # 使用 pandas 测试工具断言结果 DataFrame 与预期结果相等
        tm.assert_frame_equal(result, expected)

    # 测试方法 test_frame_reset_index_tzaware_index，使用参数化测试
    @pytest.mark.parametrize("tz", ["US/Eastern", "dateutil/US/Eastern"])
    def test_frame_reset_index_tzaware_index(self, tz):
        # 创建一个带有时区信息的日期范围 dr
        dr = date_range("2012-06-02", periods=10, tz=tz)
        # 创建一个带有日期索引的随机数值 DataFrame df
        df = DataFrame(np.random.default_rng(2).standard_normal(len(dr)), dr)
        # 对 df 进行索引重置，并再次设置索引为 "index"
        roundtripped = df.reset_index().set_index("index")
        # 检查原始索引 dr 和往返处理后的索引的时区是否一致
        xp = df.index.tz
        rs = roundtripped.index.tz
        assert xp == rs
    # 定义一个测试函数，用于测试重置索引和间隔的操作
    def test_reset_index_with_intervals(self):
        # 创建一个 IntervalIndex，从 0 到 10，命名为 "x"
        idx = IntervalIndex.from_breaks(np.arange(11), name="x")
        # 创建一个 DataFrame，包含列 "x" 和 "y"，其中 "x" 使用上面创建的 IntervalIndex，"y" 是从 0 到 9 的数组
        original = DataFrame({"x": idx, "y": np.arange(10)})[["x", "y"]]

        # 将 DataFrame 根据列 "x" 设置为索引
        result = original.set_index("x")
        # 创建一个预期的 DataFrame，只包含列 "y"，并且索引与 idx 相同
        expected = DataFrame({"y": np.arange(10)}, index=idx)
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

        # 对 result 进行重置索引操作
        result2 = result.reset_index()
        # 断言重置索引后的 DataFrame 是否与原始的 original 相等
        tm.assert_frame_equal(result2, original)
    # 定义一个测试方法，用于测试重置索引的功能，参数为 float_frame
    def test_reset_index(self, float_frame):
        # 将 float_frame 进行堆叠操作，每隔两个取一个元素
        stacked = float_frame.stack()[::2]
        # 创建一个 DataFrame，以堆叠后的数据作为 "foo" 和 "bar" 列的值
        stacked = DataFrame({"foo": stacked, "bar": stacked})

        # 定义索引的名称为 ["first", "second"]
        names = ["first", "second"]
        # 将 stacked 的索引名称设置为 names
        stacked.index.names = names
        # 重置索引并返回一个新的 DataFrame，保存为 deleveled
        deleveled = stacked.reset_index()

        # 遍历 stacked 的索引级别和对应的编码
        for i, (lev, level_codes) in enumerate(
            zip(stacked.index.levels, stacked.index.codes)
        ):
            # 根据编码取出级别对应的值
            values = lev.take(level_codes)
            # 取出当前名称
            name = names[i]
            # 断言 values 和 deleveled[name] 的索引相等
            tm.assert_index_equal(values, Index(deleveled[name]))

        # 将 stacked 的索引名称清空
        stacked.index.names = [None, None]
        # 再次重置索引并返回新的 DataFrame，保存为 deleveled2
        deleveled2 = stacked.reset_index()

        # 断言 deleveled["first"] 和 deleveled2["level_0"] 的 Series 相等，忽略名称检查
        tm.assert_series_equal(
            deleveled["first"], deleveled2["level_0"], check_names=False
        )
        # 断言 deleveled["second"] 和 deleveled2["level_1"] 的 Series 相等，忽略名称检查
        tm.assert_series_equal(
            deleveled["second"], deleveled2["level_1"], check_names=False
        )

        # 默认分配名称
        rdf = float_frame.reset_index()
        # 期望值为 float_frame 的索引值构成的 Series，名称为 "index"
        exp = Series(float_frame.index.values, name="index")
        # 断言 rdf["index"] 和 exp 的 Series 相等
        tm.assert_series_equal(rdf["index"], exp)

        # 默认分配名称，特殊情况
        df = float_frame.copy()
        # 在 df 中添加一列 "index"，值为 "foo"
        df["index"] = "foo"
        # 重置索引并返回新的 DataFrame，保存为 rdf
        rdf = df.reset_index()
        # 期望值为 float_frame 的索引值构成的 Series，名称为 "level_0"
        exp = Series(float_frame.index.values, name="level_0")
        # 断言 rdf["level_0"] 和 exp 的 Series 相等
        tm.assert_series_equal(rdf["level_0"], exp)

        # 但这是可以的，将 float_frame 的索引名称设置为 "index"
        float_frame.index.name = "index"
        # 重置索引并返回新的 DataFrame，保存为 deleveled
        deleveled = float_frame.reset_index()
        # 断言 deleveled["index"] 和 float_frame 的索引构成的 Series 相等
        tm.assert_series_equal(deleveled["index"], Series(float_frame.index))
        # 断言 deleveled 的索引和 range(len(deleveled)) 的 Index 相等，精确匹配
        tm.assert_index_equal(deleveled.index, Index(range(len(deleveled))), exact=True)

        # 保留列名
        float_frame.columns.name = "columns"
        # 重置索引并返回新的 DataFrame，保存为 reset
        reset = float_frame.reset_index()
        # 断言 reset 的列名为 "columns"
        assert reset.columns.name == "columns"

        # 只移除特定的列
        # 先重置索引再设置新的索引为 ["index", "A", "B"] 的 DataFrame，保存为 df
        df = float_frame.reset_index().set_index(["index", "A", "B"])
        # 根据 ["A", "B"] 再次重置索引，返回新的 DataFrame，保存为 rs
        rs = df.reset_index(["A", "B"])
        # 断言 rs 和 float_frame 的 DataFrame 相等
        tm.assert_frame_equal(rs, float_frame)

        # 再次根据 ["index", "A", "B"] 重置索引，返回新的 DataFrame，保存为 rs
        rs = df.reset_index(["index", "A", "B"])
        # 断言 rs 和 float_frame 的 DataFrame 相等
        tm.assert_frame_equal(rs, float_frame.reset_index())

        # 再次根据 ["index", "A", "B"] 重置索引，返回新的 DataFrame，保存为 rs
        rs = df.reset_index(["index", "A", "B"])
        # 断言 rs 和 float_frame 的 DataFrame 相等
        tm.assert_frame_equal(rs, float_frame.reset_index())

        # 根据 "A" 重置索引，返回新的 DataFrame，保存为 rs
        rs = df.reset_index("A")
        # 期望值为 float_frame 先重置索引再设置新的索引为 ["index", "B"] 的 DataFrame，保存为 xp
        xp = float_frame.reset_index().set_index(["index", "B"])
        # 断言 rs 和 xp 的 DataFrame 相等
        tm.assert_frame_equal(rs, xp)

        # 测试原地重置索引
        df = float_frame.copy()
        # 重置索引并返回新的 DataFrame，保存为 reset
        reset = float_frame.reset_index()
        # 将 df 原地重置索引
        return_value = df.reset_index(inplace=True)
        # 断言返回值为 None
        assert return_value is None
        # 断言 df 和 reset 的 DataFrame 相等
        tm.assert_frame_equal(df, reset)

        # 先重置索引再设置新的索引为 ["index", "A", "B"] 的 DataFrame，保存为 df
        df = float_frame.reset_index().set_index(["index", "A", "B"])
        # 根据 "A" 重置索引，且不保留 "A" 列，返回新的 DataFrame，保存为 rs
        rs = df.reset_index("A", drop=True)
        # 创建 xp 为 float_frame 的副本
        xp = float_frame.copy()
        # 删除 xp 中的 "A" 列，并设置新的索引为 ["B"]，并追加到原来的索引中
        del xp["A"]
        xp = xp.set_index(["B"], append=True)
        # 断言 rs 和 xp 的 DataFrame 相等
        tm.assert_frame_equal(rs, xp)
    # 定义测试方法，用于测试 DataFrame 的 reset_index 方法
    def test_reset_index_name(self):
        # 创建一个 DataFrame，包含两行数据，每行四列，指定列名和索引名
        df = DataFrame(
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            columns=["A", "B", "C", "D"],
            index=Index(range(2), name="x"),
        )
        # 断言 reset_index() 后的索引名应为 None
        assert df.reset_index().index.name is None
        # 断言 reset_index(drop=True) 后的索引名应为 None
        assert df.reset_index(drop=True).index.name is None
        # 执行 inplace=True 的 reset_index() 方法
        return_value = df.reset_index(inplace=True)
        # 断言返回值为 None
        assert return_value is None
        # 断言 DataFrame 的索引名应为 None
        assert df.index.name is None

    # 使用 pytest.mark.parametrize 标记的参数化测试方法，用于测试 reset_index 方法的多级索引重置
    @pytest.mark.parametrize("levels", [["A", "B"], [0, 1]])
    def test_reset_index_level(self, levels):
        # 创建一个 DataFrame，包含两行数据，四列，未指定索引
        df = DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=["A", "B", "C", "D"])

        # 多级索引情况下的测试
        # 设置多级索引 ["A", "B"]，并重置指定层级 levels[0]
        result = df.set_index(["A", "B"]).reset_index(level=levels[0])
        # 断言重置后的 DataFrame 与设置 ["B"] 后的 DataFrame 相等
        tm.assert_frame_equal(result, df.set_index("B"))

        # 设置多级索引 ["A", "B"]，并重置指定层级 levels[:1]，即 ["A"]
        result = df.set_index(["A", "B"]).reset_index(level=levels[:1])
        # 断言重置后的 DataFrame 与原始 DataFrame 相等
        tm.assert_frame_equal(result, df)

        # 设置多级索引 ["A", "B"]，并重置所有层级 levels
        result = df.set_index(["A", "B"]).reset_index(level=levels)
        # 断言重置后的 DataFrame 与原始 DataFrame 相等
        tm.assert_frame_equal(result, df)

        # 设置多级索引 ["A", "B"]，并重置所有层级 levels，同时设置 drop=True
        result = df.set_index(["A", "B"]).reset_index(level=levels, drop=True)
        # 断言重置后的 DataFrame 与仅包含 ["C", "D"] 列的 DataFrame 相等
        tm.assert_frame_equal(result, df[["C", "D"]])

        # 单级索引情况下的测试 (GH 16263)
        # 设置单级索引 "A"，并重置指定层级 levels[0]，即 "A"
        result = df.set_index("A").reset_index(level=levels[0])
        # 断言重置后的 DataFrame 与原始 DataFrame 相等
        tm.assert_frame_equal(result, df)

        # 设置单级索引 "A"，并重置指定层级 levels[:1]，即 ["A"]
        result = df.set_index("A").reset_index(level=levels[:1])
        # 断言重置后的 DataFrame 与原始 DataFrame 相等
        tm.assert_frame_equal(result, df)

        # 设置单级索引 "A"，并重置指定层级 levels[0]，同时设置 drop=True
        result = df.set_index(["A"]).reset_index(level=levels[0], drop=True)
        # 断言重置后的 DataFrame 与仅包含 ["B", "C", "D"] 列的 DataFrame 相等
        tm.assert_frame_equal(result, df[["B", "C", "D"]])

    # 测试方法，用于测试 reset_index 方法在数据类型上的正确性
    def test_reset_index_right_dtype(self):
        # 创建一个时间序列和对应的 DataFrame
        time = np.arange(0.0, 10, np.sqrt(2) / 2)
        s1 = Series((9.81 * time**2) / 2, index=Index(time, name="time"), name="speed")
        df = DataFrame(s1)

        # 对时间序列进行 reset_index() 操作
        reset = s1.reset_index()
        # 断言重置后的 "time" 列的数据类型为 np.float64
        assert reset["time"].dtype == np.float64

        # 对 DataFrame 进行 reset_index() 操作
        reset = df.reset_index()
        # 断言重置后的 "time" 列的数据类型为 np.float64
        assert reset["time"].dtype == np.float64
    # 定义一个测试方法，用于测试 DataFrame 的 reset_index 方法在多级索引列上的功能

    # 创建一个3x3的随机数矩阵，并将其转换为对象类型
    vals = np.random.default_rng(2).standard_normal((3, 3)).astype(object)
    
    # 定义索引列表
    idx = ["x", "y", "z"]
    
    # 将索引与随机数矩阵合并成完整的数据
    full = np.hstack(([[x] for x in idx], vals))
    
    # 创建一个 DataFrame 对象，指定数据、行索引（带名称"a"）、列名（多级）
    df = DataFrame(
        vals,
        Index(idx, name="a"),
        columns=[["b", "b", "c"], ["mean", "median", "mean"]],
    )
    
    # 使用 reset_index 方法重置 DataFrame 索引，并进行数据比较
    rs = df.reset_index()
    xp = DataFrame(
        full, columns=[["a", "b", "b", "c"], ["", "mean", "median", "mean"]]
    )
    tm.assert_frame_equal(rs, xp)

    # 使用 reset_index 方法并设置 col_fill=None，进行索引重置，进行数据比较
    rs = df.reset_index(col_fill=None)
    xp = DataFrame(
        full, columns=[["a", "b", "b", "c"], ["a", "mean", "median", "mean"]]
    )
    tm.assert_frame_equal(rs, xp)

    # 使用 reset_index 方法并指定 col_level=1 和 col_fill="blah"，进行索引重置，进行数据比较
    rs = df.reset_index(col_level=1, col_fill="blah")
    xp = DataFrame(
        full, columns=[["blah", "b", "b", "c"], ["a", "mean", "median", "mean"]]
    )
    tm.assert_frame_equal(rs, xp)

    # 创建一个具有多级行索引的 DataFrame 对象
    df = DataFrame(
        vals,
        MultiIndex.from_arrays([[0, 1, 2], ["x", "y", "z"]], names=["d", "a"]),
        columns=[["b", "b", "c"], ["mean", "median", "mean"]],
    )
    
    # 使用 reset_index 方法并指定重置索引"a"，进行索引重置，进行数据比较
    rs = df.reset_index("a")
    xp = DataFrame(
        full,
        Index([0, 1, 2], name="d"),
        columns=[["a", "b", "b", "c"], ["", "mean", "median", "mean"]],
    )
    tm.assert_frame_equal(rs, xp)

    # 使用 reset_index 方法并指定重置索引"a"和 col_fill=None，进行索引重置，进行数据比较
    rs = df.reset_index("a", col_fill=None)
    xp = DataFrame(
        full,
        Index(range(3), name="d"),
        columns=[["a", "b", "b", "c"], ["a", "mean", "median", "mean"]],
    )
    tm.assert_frame_equal(rs, xp)

    # 使用 reset_index 方法并指定重置索引"a"、col_fill="blah"和 col_level=1，进行索引重置，进行数据比较
    rs = df.reset_index("a", col_fill="blah", col_level=1)
    xp = DataFrame(
        full,
        Index(range(3), name="d"),
        columns=[["blah", "b", "b", "c"], ["a", "mean", "median", "mean"]],
    )
    tm.assert_frame_equal(rs, xp)
    def test_reset_index_multiindex_nan(self):
        # GH#6322, testing reset_index on MultiIndexes
        # when we have a nan or all nan
        
        # 创建一个包含列"A", "B", "C"的DataFrame对象，其中:
        # - "A"列包含字符串"a", "b", "c"
        # - "B"列包含0, 1, 和 NaN
        # - "C"列包含从随机数生成器中生成的长度为3的随机浮点数数组
        df = DataFrame(
            {
                "A": ["a", "b", "c"],
                "B": [0, 1, np.nan],
                "C": np.random.default_rng(2).random(3),
            }
        )
        
        # 对DataFrame执行set_index操作，设置"A", "B"为索引，然后重置索引
        rs = df.set_index(["A", "B"]).reset_index()
        
        # 使用tm.assert_frame_equal断言重置后的DataFrame与原始DataFrame相等
        tm.assert_frame_equal(rs, df)
        
        # 创建一个包含列"A", "B", "C"的DataFrame对象，其中:
        # - "A"列包含NaN, "b", "c"
        # - "B"列包含0, 1, 2
        # - "C"列包含从随机数生成器中生成的长度为3的随机浮点数数组
        df = DataFrame(
            {
                "A": [np.nan, "b", "c"],
                "B": [0, 1, 2],
                "C": np.random.default_rng(2).random(3),
            }
        )
        
        # 对DataFrame执行set_index操作，设置"A", "B"为索引，然后重置索引
        rs = df.set_index(["A", "B"]).reset_index()
        
        # 使用tm.assert_frame_equal断言重置后的DataFrame与原始DataFrame相等
        tm.assert_frame_equal(rs, df)
        
        # 创建一个包含列"A", "B", "C"的DataFrame对象，其中:
        # - "A"列包含字符串"a", "b", "c"
        # - "B"列包含0, 1, 2
        # - "C"列包含NaN, 1.1, 2.2
        df = DataFrame({"A": ["a", "b", "c"], "B": [0, 1, 2], "C": [np.nan, 1.1, 2.2]})
        
        # 对DataFrame执行set_index操作，设置"A", "B"为索引，然后重置索引
        rs = df.set_index(["A", "B"]).reset_index()
        
        # 使用tm.assert_frame_equal断言重置后的DataFrame与原始DataFrame相等
        tm.assert_frame_equal(rs, df)
        
        # 创建一个包含列"A", "B", "C"的DataFrame对象，其中:
        # - "A"列包含字符串"a", "b", "c"
        # - "B"列包含NaN, NaN, NaN
        # - "C"列包含从随机数生成器中生成的长度为3的随机浮点数数组
        df = DataFrame(
            {
                "A": ["a", "b", "c"],
                "B": [np.nan, np.nan, np.nan],
                "C": np.random.default_rng(2).random(3),
            }
        )
        
        # 对DataFrame执行set_index操作，设置"A", "B"为索引，然后重置索引
        rs = df.set_index(["A", "B"]).reset_index()
        
        # 使用tm.assert_frame_equal断言重置后的DataFrame与原始DataFrame相等
        tm.assert_frame_equal(rs, df)

    @pytest.mark.parametrize(
        "name",
        [
            None,
            "foo",
            2,
            3.0,
            pd.Timedelta(6),
            Timestamp("2012-12-30", tz="UTC"),
            "2012-12-31",
        ],
    )
    def test_reset_index_with_datetimeindex_cols(self, name):
        # GH#5818
        
        # 创建一个包含数值的DataFrame对象，其中:
        # - 列数为2，行数为2
        # - 列名为datetime从"1/1/2013"到"1/2/2013"的范围内
        # - 索引为"A", "B"
        df = DataFrame(
            [[1, 2], [3, 4]],
            columns=date_range("1/1/2013", "1/2/2013"),
            index=["A", "B"],
        )
        
        # 设置DataFrame的索引名称为参数"name"
        df.index.name = name
        
        # 对DataFrame执行重置索引操作
        result = df.reset_index()
        
        # 根据参数"name"确定预期的列名，构建期望的DataFrame对象
        item = name if name is not None else "index"
        columns = Index([item, datetime(2013, 1, 1), datetime(2013, 1, 2)])
        if isinstance(item, str) and item == "2012-12-31":
            columns = columns.astype("datetime64[ns]")
        else:
            assert columns.dtype == object
        
        expected = DataFrame(
            [["A", 1, 2], ["B", 3, 4]],
            columns=columns,
        )
        
        # 使用tm.assert_frame_equal断言重置后的DataFrame与预期DataFrame相等
        tm.assert_frame_equal(result, expected)

    def test_reset_index_range(self):
        # GH#12071
        
        # 创建一个包含数值的DataFrame对象，其中:
        # - 列名为"A", "B"
        # - 索引为RangeIndex(stop=2)
        df = DataFrame([[0, 0], [1, 1]], columns=["A", "B"], index=RangeIndex(stop=2))
        
        # 对DataFrame执行重置索引操作
        result = df.reset_index()
        
        # 断言重置后的索引类型为RangeIndex
        assert isinstance(result.index, RangeIndex)
        
        # 构建期望的DataFrame对象
        expected = DataFrame(
            [[0, 0, 0], [1, 1, 1]],
            columns=["index", "A", "B"],
            index=RangeIndex(stop=2),
        )
        
        # 使用tm.assert_frame_equal断言重置后的DataFrame与预期DataFrame相等
        tm.assert_frame_equal(result, expected)
    # 定义测试方法，测试对多级索引 DataFrame 进行 reset_index 操作
    def test_reset_index_multiindex_columns(self, multiindex_df):
        # 对多级索引 DataFrame 只选择"B"列，重置索引，并将索引命名为"A"，然后进行断言比较
        result = multiindex_df[["B"]].rename_axis("A").reset_index()
        tm.assert_frame_equal(result, multiindex_df)

        # GH#16120: 已存在的列
        # 验证重命名索引后插入已存在的列会抛出 ValueError 异常，匹配特定的错误信息
        msg = r"cannot insert \('A', ''\), already exists"
        with pytest.raises(ValueError, match=msg):
            multiindex_df.rename_axis("A").reset_index()

        # GH#16164: 多级索引 (元组) 的完整键
        # 设置索引为元组 ("A", "")，然后重置索引，进行断言比较
        result = multiindex_df.set_index([("A", "")]).reset_index()
        tm.assert_frame_equal(result, multiindex_df)

        # 带有额外的（未命名的）索引级别
        # 创建一个 DataFrame，包含一个未命名的索引级别，然后预期的结果是将它与多级索引 DataFrame 的部分列合并
        idx_col = DataFrame(
            [[0], [1]], columns=MultiIndex.from_tuples([("level_0", "")])
        )
        expected = pd.concat([idx_col, multiindex_df[[("B", "b"), ("A", "")]]], axis=1)
        result = multiindex_df.set_index([("B", "b")], append=True).reset_index()
        tm.assert_frame_equal(result, expected)

        # 使用长度过长的元组作为索引名会抛出 ValueError 异常
        msg = "Item must have length equal to number of levels."
        with pytest.raises(ValueError, match=msg):
            multiindex_df.rename_axis([("C", "c", "i")]).reset_index()

        # 或者使用长度不匹配的元组也会抛出 ValueError 异常
        levels = [["A", "a", ""], ["B", "b", "i"]]
        df2 = DataFrame([[0, 2], [1, 3]], columns=MultiIndex.from_tuples(levels))
        idx_col = DataFrame(
            [[0], [1]], columns=MultiIndex.from_tuples([("C", "c", "ii")])
        )
        expected = pd.concat([idx_col, df2], axis=1)
        result = df2.rename_axis([("C", "c")]).reset_index(col_fill="ii")
        tm.assert_frame_equal(result, expected)

        # 使用不兼容 col_fill=None 的长度不匹配的元组作为索引名会抛出 ValueError 异常
        with pytest.raises(
            ValueError,
            match=(
                "col_fill=None is incompatible with "
                r"incomplete column name \('C', 'c'\)"
            ),
        ):
            df2.rename_axis([("C", "c")]).reset_index(col_fill=None)

        # 使用 col_level != 0 的元组作为索引名会进行重置索引，进行断言比较
        result = df2.rename_axis([("c", "ii")]).reset_index(col_level=1, col_fill="C")
        tm.assert_frame_equal(result, expected)
    # GH#44755 reset_index with duplicate column labels
    # 重命名多级索引数据框的索引轴为"A"
    df = multiindex_df.rename_axis("A")
    # 设置数据框的标志，允许重复的标签
    df = df.set_flags(allows_duplicate_labels=flag)

    # 如果标志为真并且允许重复，则重置索引
    if flag and allow_duplicates:
        # 执行重置索引操作，允许重复的索引
        result = df.reset_index(allow_duplicates=allow_duplicates)
        # 定义多级索引的预期结构
        levels = [["A", ""], ["A", ""], ["B", "b"]]
        expected = DataFrame(
            [[0, 0, 2], [1, 1, 3]], columns=MultiIndex.from_tuples(levels)
        )
        # 断言重置索引后的数据框与预期结果相等
        tm.assert_frame_equal(result, expected)
    else:
        # 如果标志为假或者不允许重复，则抛出值错误异常
        if not flag and allow_duplicates:
            msg = (
                "Cannot specify 'allow_duplicates=True' when "
                "'self.flags.allows_duplicate_labels' is False"
            )
        else:
            msg = r"cannot insert \('A', ''\), already exists"
        # 使用 pytest 断言抛出值错误异常，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            df.reset_index(allow_duplicates=allow_duplicates)

@pytest.mark.parametrize("flag", [False, True])
def test_reset_index_duplicate_columns_default(self, multiindex_df, flag):
    # 重命名多级索引数据框的索引轴为"A"
    df = multiindex_df.rename_axis("A")
    # 设置数据框的标志，允许或禁止重复的标签
    df = df.set_flags(allows_duplicate_labels=flag)

    # 预期的错误消息
    msg = r"cannot insert \('A', ''\), already exists"
    # 使用 pytest 断言抛出值错误异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        # 调用 reset_index 方法，不指定是否允许重复索引
        df.reset_index()

@pytest.mark.parametrize("allow_duplicates", ["bad value"])
def test_reset_index_allow_duplicates_check(self, multiindex_df, allow_duplicates):
    # 使用 pytest 断言抛出值错误异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match="expected type bool"):
        # 调用 reset_index 方法，传入错误的 allow_duplicates 参数
        multiindex_df.reset_index(allow_duplicates=allow_duplicates)

def test_reset_index_datetime(self, tz_naive_fixture):
    # GH#3950
    # 获取时区信息
    tz = tz_naive_fixture
    # 创建日期范围索引 idx1
    idx1 = date_range("1/1/2011", periods=5, freq="D", tz=tz, name="idx1")
    # 创建整数索引 idx2
    idx2 = Index(range(5), name="idx2", dtype="int64")
    # 创建多级索引 idx，由 idx1 和 idx2 组成
    idx = MultiIndex.from_arrays([idx1, idx2])
    # 创建数据框 df，包含列"a"和"b"，并使用 idx 作为索引
    df = DataFrame(
        {"a": np.arange(5, dtype="int64"), "b": ["A", "B", "C", "D", "E"]},
        index=idx,
    )

    # 创建预期的数据框，重置索引后的列顺序为["idx1", "idx2", "a", "b"]
    expected = DataFrame(
        {
            "idx1": idx1,
            "idx2": np.arange(5, dtype="int64"),
            "a": np.arange(5, dtype="int64"),
            "b": ["A", "B", "C", "D", "E"],
        },
        columns=["idx1", "idx2", "a", "b"],
    )

    # 断言重置索引后的数据框与预期结果相等
    tm.assert_frame_equal(df.reset_index(), expected)
    def test_reset_index_datetime2(self, tz_naive_fixture):
        # 测试函数：test_reset_index_datetime2，重置索引操作
        tz = tz_naive_fixture
        # 创建第一个时间范围索引，从"1/1/2011"开始，周期为5天，时区为tz，名称为"idx1"
        idx1 = date_range("1/1/2011", periods=5, freq="D", tz=tz, name="idx1")
        # 创建第二个整数索引，范围为0到4，名称为"idx2"，数据类型为int64
        idx2 = Index(range(5), name="idx2", dtype="int64")
        # 创建第三个时间范围索引，从"1/1/2012"开始，周期为5个月，时区为"Europe/Paris"，名称为"idx3"
        idx3 = date_range(
            "1/1/2012", periods=5, freq="MS", tz="Europe/Paris", name="idx3"
        )
        # 使用三个索引创建多级索引对象
        idx = MultiIndex.from_arrays([idx1, idx2, idx3])
        # 创建数据框，包含两列："a"为0到4的整数，"b"为字符列表["A", "B", "C", "D", "E"]，索引为idx
        df = DataFrame(
            {"a": np.arange(5, dtype="int64"), "b": ["A", "B", "C", "D", "E"]},
            index=idx,
        )

        # 创建期望结果数据框，重置索引列为"idx1", "idx2", "idx3", "a", "b"
        expected = DataFrame(
            {
                "idx1": idx1,
                "idx2": np.arange(5, dtype="int64"),
                "idx3": idx3,
                "a": np.arange(5, dtype="int64"),
                "b": ["A", "B", "C", "D", "E"],
            },
            columns=["idx1", "idx2", "idx3", "a", "b"],
        )
        # 执行重置索引操作，结果存储在result中
        result = df.reset_index()
        # 使用测试工具tm.assert_frame_equal检查结果是否与期望一致
        tm.assert_frame_equal(result, expected)

    def test_reset_index_datetime3(self, tz_naive_fixture):
        # 测试函数：test_reset_index_datetime3，重置索引操作，针对GH#7793问题
        tz = tz_naive_fixture
        # 创建时间范围索引，从"20130101"开始，周期为3天，时区为tz
        dti = date_range("20130101", periods=3, tz=tz)
        # 创建多级索引，包含"a", "b"两个级别，每个级别与dti相乘
        idx = MultiIndex.from_product([["a", "b"], dti])
        # 创建数据框，包含一列"a"，值为0到5的整数，索引为idx
        df = DataFrame(
            np.arange(6, dtype="int64").reshape(6, 1), columns=["a"], index=idx
        )

        # 创建期望结果数据框，重置索引列为"level_0", "level_1", "a"
        expected = DataFrame(
            {
                "level_0": "a a a b b b".split(),
                "level_1": dti.append(dti),
                "a": np.arange(6, dtype="int64"),
            },
            columns=["level_0", "level_1", "a"],
        )
        # 执行重置索引操作，结果存储在result中
        result = df.reset_index()
        # 使用测试工具tm.assert_frame_equal检查结果是否与期望一致
        tm.assert_frame_equal(result, expected)

    def test_reset_index_period(self):
        # 测试函数：test_reset_index_period，重置索引操作，针对GH#7746问题
        # 创建多级索引，包含时间周期范围和特征列表
        idx = MultiIndex.from_product(
            [pd.period_range("20130101", periods=3, freq="M"), list("abc")],
            names=["month", "feature"],
        )

        # 创建数据框，包含一列"a"，值为0到8的整数，索引为idx
        df = DataFrame(
            np.arange(9, dtype="int64").reshape(-1, 1), index=idx, columns=["a"]
        )
        # 创建期望结果数据框，重置索引列为"month", "feature", "a"
        expected = DataFrame(
            {
                "month": (
                    [pd.Period("2013-01", freq="M")] * 3
                    + [pd.Period("2013-02", freq="M")] * 3
                    + [pd.Period("2013-03", freq="M")] * 3
                ),
                "feature": ["a", "b", "c"] * 3,
                "a": np.arange(9, dtype="int64"),
            },
            columns=["month", "feature", "a"],
        )
        # 执行重置索引操作，结果存储在result中
        result = df.reset_index()
        # 使用测试工具tm.assert_frame_equal检查结果是否与期望一致
        tm.assert_frame_equal(result, expected)

    def test_reset_index_delevel_infer_dtype(self):
        # 测试函数：test_reset_index_delevel_infer_dtype，重置索引操作，确认推断数据类型
        # 创建包含元组的列表
        tuples = list(product(["foo", "bar"], [10, 20], [1.0, 1.1]))
        # 创建多级索引，名称为"prm0", "prm1", "prm2"
        index = MultiIndex.from_tuples(tuples, names=["prm0", "prm1", "prm2"])
        # 创建数据框，包含三列"A", "B", "C"，值为标准正态分布随机数，索引为index
        df = DataFrame(
            np.random.default_rng(2).standard_normal((8, 3)),
            columns=["A", "B", "C"],
            index=index,
        )
        # 执行重置索引操作，结果存储在deleveled中
        deleveled = df.reset_index()
        # 断言确保重置后的"prm1"列为整数类型
        assert is_integer_dtype(deleveled["prm1"])
        # 断言确保重置后的"prm2"列为浮点类型
        assert is_float_dtype(deleveled["prm2"])
    # 定义一个测试方法，测试在删除索引的情况下DataFrame的重置
    def test_reset_index_with_drop(
        self, multiindex_year_month_day_dataframe_random_data
    ):
        # 从参数传入的数据中获取多级索引的DataFrame
        ymd = multiindex_year_month_day_dataframe_random_data

        # 使用 drop=True 参数重置索引，返回新的 DataFrame
        deleveled = ymd.reset_index(drop=True)
        # 断言重置后的 DataFrame 的列数与原始 DataFrame 的列数相同
        assert len(deleveled.columns) == len(ymd.columns)
        # 断言重置后的 DataFrame 的索引名称与原始 DataFrame 的索引名称相同
        assert deleveled.index.name == ymd.index.name

    # 使用 pytest 的参数化装饰器定义多个参数组合的测试用例
    @pytest.mark.parametrize(
        "ix_data, exp_data",
        [
            (
                [(pd.NaT, 1), (pd.NaT, 2)],
                {"a": [pd.NaT, pd.NaT], "b": [1, 2], "x": [11, 12]},
            ),
            (
                [(pd.NaT, 1), (Timestamp("2020-01-01"), 2)],
                {"a": [pd.NaT, Timestamp("2020-01-01")], "b": [1, 2], "x": [11, 12]},
            ),
            (
                [(pd.NaT, 1), (pd.Timedelta(123, "D"), 2)],
                {"a": [pd.NaT, pd.Timedelta(123, "D")], "b": [1, 2], "x": [11, 12]},
            ),
        ],
    )
    # 定义测试方法，测试在包含 NaT 值的多级索引中调用 reset_index() 方法
    def test_reset_index_nat_multiindex(self, ix_data, exp_data):
        # 创建一个多级索引对象，指定索引的名称
        ix = MultiIndex.from_tuples(ix_data, names=["a", "b"])
        # 创建一个 DataFrame 对象，指定数据列和索引
        result = DataFrame({"x": [11, 12]}, index=ix)
        # 调用 reset_index() 方法重置索引
        result = result.reset_index()

        # 创建一个期望结果的 DataFrame 对象
        expected = DataFrame(exp_data)
        # 使用 assert_frame_equal 方法断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    # 使用 pytest 的参数化装饰器定义多个参数组合的测试用例
    @pytest.mark.parametrize(
        "codes", ([[0, 0, 1, 1], [0, 1, 0, 1]], [[0, 0, -1, 1], [0, 1, 0, 1]])
    )
    # 定义测试方法，测试在包含分类数据和缺失值的多级索引中调用 reset_index() 方法
    def test_rest_index_multiindex_categorical_with_missing_values(self, codes):
        # GH#24206

        # 创建一个多级索引对象，包含两个类别索引，以及给定的 codes 数组
        index = MultiIndex(
            [CategoricalIndex(["A", "B"]), CategoricalIndex(["a", "b"])], codes
        )
        # 创建一个数据字典，包含一个名为 "col" 的列，长度与索引相同
        data = {"col": range(len(index))}
        # 创建一个 DataFrame 对象，指定数据和索引
        df = DataFrame(data=data, index=index)

        # 创建一个期望结果的 DataFrame 对象，使用 from_codes 方法创建分类对象
        expected = DataFrame(
            {
                "level_0": Categorical.from_codes(codes[0], categories=["A", "B"]),
                "level_1": Categorical.from_codes(codes[1], categories=["a", "b"]),
                "col": range(4),
            }
        )

        # 调用 reset_index() 方法重置索引，并使用 assert_frame_equal 方法断言结果是否与期望一致
        res = df.reset_index()
        tm.assert_frame_equal(res, expected)

        # 执行往返测试，将期望结果重新设置为索引，再次调用 reset_index() 方法，并使用 assert_frame_equal 方法断言结果是否与期望一致
        res = expected.set_index(["level_0", "level_1"]).reset_index()
        tm.assert_frame_equal(res, expected)
@pytest.mark.parametrize(
    "array, dtype",
    [
        (["a", "b"], object),  # 参数化测试：使用字符串数组和对象类型
        (
            pd.period_range("12-1-2000", periods=2, freq="Q-DEC"),  # 参数化测试：使用周期范围和周期数据类型
            pd.PeriodDtype(freq="Q-DEC"),
        ),
    ],
)
def test_reset_index_dtypes_on_empty_frame_with_multiindex(
    array, dtype, using_infer_string
):
    # GH 19602 - Preserve dtype on empty DataFrame with MultiIndex
    # 创建多级索引对象
    idx = MultiIndex.from_product([[0, 1], [0.5, 1.0], array])
    # 创建空DataFrame并重置索引，获取结果的数据类型
    result = DataFrame(index=idx)[:0].reset_index().dtypes
    # 如果使用推断字符串并且dtype为object，则将dtype改为"string"
    if using_infer_string and dtype == object:
        dtype = "string"
    # 期望的结果数据类型
    expected = Series({"level_0": np.int64, "level_1": np.float64, "level_2": dtype})
    # 断言结果与期望一致
    tm.assert_series_equal(result, expected)


def test_reset_index_empty_frame_with_datetime64_multiindex():
    # https://github.com/pandas-dev/pandas/issues/35606
    # 创建日期时间索引对象
    dti = pd.DatetimeIndex(["2020-07-20 00:00:00"], dtype="M8[ns]")
    # 创建空DataFrame，设定多级索引并重置索引
    idx = MultiIndex.from_product([dti, [3, 4]], names=["a", "b"])[:0]
    df = DataFrame(index=idx, columns=["c", "d"])
    # 获取重置索引后的结果DataFrame
    result = df.reset_index()
    # 创建期望的结果DataFrame
    expected = DataFrame(
        columns=list("abcd"), index=RangeIndex(start=0, stop=0, step=1)
    )
    # 将"a"列和"b"列转换为指定数据类型
    expected["a"] = expected["a"].astype("datetime64[ns]")
    expected["b"] = expected["b"].astype("int64")
    # 断言结果与期望一致
    tm.assert_frame_equal(result, expected)


def test_reset_index_empty_frame_with_datetime64_multiindex_from_groupby(
    using_infer_string,
):
    # https://github.com/pandas-dev/pandas/issues/35657
    # 创建日期时间索引对象
    dti = pd.DatetimeIndex(["2020-01-01"], dtype="M8[ns]")
    # 创建DataFrame，使用groupby方法后获取头部为0的结果，并对"c2"列和"c3"列进行求和
    df = df.head(0).groupby(["c2", "c3"])[["c1"]].sum()
    # 重置索引获取结果DataFrame
    result = df.reset_index()
    # 创建期望的结果DataFrame
    expected = DataFrame(
        columns=["c2", "c3", "c1"], index=RangeIndex(start=0, stop=0, step=1)
    )
    # 将"c3"列转换为指定数据类型
    expected["c3"] = expected["c3"].astype("datetime64[ns]")
    expected["c1"] = expected["c1"].astype("float64")
    # 如果使用推断字符串，则将"c2"列转换为指定数据类型
    if using_infer_string:
        expected["c2"] = expected["c2"].astype("string[pyarrow_numpy]")
    # 断言结果与期望一致
    tm.assert_frame_equal(result, expected)


def test_reset_index_multiindex_nat():
    # GH 11479
    # 创建索引范围和时间戳对象
    idx = range(3)
    tstamp = date_range("2015-07-01", freq="D", periods=3)
    # 创建DataFrame，设置多级索引并重置"tstamp"列的索引
    df = DataFrame({"id": idx, "tstamp": tstamp, "a": list("abc")})
    # 在索引为2的位置设置为NaT
    df.loc[2, "tstamp"] = pd.NaT
    # 重置索引获取结果DataFrame
    result = df.set_index(["id", "tstamp"]).reset_index("id")
    # 期望的结果时间戳索引对象
    exp_dti = pd.DatetimeIndex(
        ["2015-07-01", "2015-07-02", "NaT"], dtype="M8[ns]", name="tstamp"
    )
    # 创建期望的结果DataFrame
    expected = DataFrame(
        {"id": range(3), "a": list("abc")},
        index=exp_dti,
    )
    # 断言结果与期望一致
    tm.assert_frame_equal(result, expected)


def test_reset_index_interval_columns_object_cast():
    # GH 19136
    # 创建DataFrame，设置索引和列对象
    df = DataFrame(
        np.eye(2), index=Index([1, 2], name="Year"), columns=cut([1, 2], [0, 1, 2])
    )
    # 重置索引获取结果DataFrame
    result = df.reset_index()
    # 创建期望的结果DataFrame
    expected = DataFrame(
        [[1, 1.0, 0.0], [2, 0.0, 1.0]],
        columns=Index(["Year", Interval(0, 1), Interval(1, 2)]),
    )
    # 断言结果与期望一致
    tm.assert_frame_equal(result, expected)
# 定义一个测试函数，用于验证 reset_index() 方法在不同情况下的行为
def test_reset_index_rename(float_frame):
    # GH 6878
    # 对 float_frame 进行重置索引，使用字符串 "new_name" 作为新的索引名
    result = float_frame.reset_index(names="new_name")
    # 从 float_frame 中获取索引值，创建一个名为 "new_name" 的 Series 对象
    expected = Series(float_frame.index.values, name="new_name")
    # 验证 result 中的 "new_name" 列与 expected 相等
    tm.assert_series_equal(result["new_name"], expected)

    # 对 float_frame 进行重置索引，使用整数 123 作为新的索引名
    result = float_frame.reset_index(names=123)
    # 从 float_frame 中获取索引值，创建一个名为 123 的 Series 对象
    expected = Series(float_frame.index.values, name=123)
    # 验证 result 中的 123 列与 expected 相等
    tm.assert_series_equal(result[123], expected)


# 定义一个测试函数，用于验证带有多级索引的 DataFrame 在重置索引时的行为
def test_reset_index_rename_multiindex(float_frame):
    # GH 6878
    # 对 float_frame 进行堆叠操作，每隔两行取一个值
    stacked_df = float_frame.stack()[::2]
    # 创建一个 DataFrame，包含两列 "foo" 和 "bar"，均为 stacked_df 的值
    stacked_df = DataFrame({"foo": stacked_df, "bar": stacked_df})

    # 将 stacked_df 的索引命名为 ["first", "second"]
    names = ["first", "second"]
    stacked_df.index.names = names

    # 对 stacked_df 进行重置索引，并验证结果
    result = stacked_df.reset_index()
    # 重置索引后，命名为 ["new_first", "new_second"]，并验证结果
    expected = stacked_df.reset_index(names=["new_first", "new_second"])
    # 验证 result 中的 "first" 列与 expected 中的 "new_first" 列相等，不检查列名
    tm.assert_series_equal(result["first"], expected["new_first"], check_names=False)
    # 验证 result 中的 "second" 列与 expected 中的 "new_second" 列相等，不检查列名
    tm.assert_series_equal(result["second"], expected["new_second"], check_names=False)


# 定义一个测试函数，用于验证在错误情况下重置索引时的行为
def test_errorreset_index_rename(float_frame):
    # GH 6878
    # 对 float_frame 进行堆叠操作，每隔两行取一个值
    stacked_df = float_frame.stack()[::2]
    # 创建一个 DataFrame，包含两列 "first" 和 "second"，均为 stacked_df 的值
    stacked_df = DataFrame({"first": stacked_df, "second": stacked_df})

    # 使用 pytest 检测是否会抛出 ValueError 异常，匹配指定的错误信息
    with pytest.raises(
        ValueError, match="Index names must be str or 1-dimensional list"
    ):
        # 尝试以字典形式命名索引，会引发异常
        stacked_df.reset_index(names={"first": "new_first", "second": "new_second"})

    # 使用 pytest 检测是否会抛出 IndexError 异常，匹配指定的错误信息
    with pytest.raises(IndexError, match="list index out of range"):
        # 尝试以单个元素列表命名索引，会引发异常
        stacked_df.reset_index(names=["new_first"])


# 定义一个测试函数，用于验证重置索引时索引名为 False 的情况
def test_reset_index_false_index_name():
    # 创建一个 Series，数据为 range(5, 10)，索引为 range(5)，索引名设为 False
    result_series = Series(data=range(5, 10), index=range(5))
    # 对 result_series 进行重置索引操作
    result_series.reset_index()
    # 创建一个期望的 Series，数据和索引与 result_series 相同，索引名为 False
    expected_series = Series(range(5, 10), RangeIndex(range(5), name=False))
    # 验证 result_series 与 expected_series 相等
    tm.assert_series_equal(result_series, expected_series)

    # GH 38147
    # 创建一个 DataFrame，数据为 range(5, 10)，索引为 range(5)，索引名设为 False
    result_frame = DataFrame(data=range(5, 10), index=range(5))
    # 对 result_frame 进行重置索引操作
    result_frame.reset_index()
    # 创建一个期望的 DataFrame，数据和索引与 result_frame 相同，索引名为 False
    expected_frame = DataFrame(range(5, 10), RangeIndex(range(5), name=False))
    # 验证 result_frame 与 expected_frame 相等
    tm.assert_frame_equal(result_frame, expected_frame)
```