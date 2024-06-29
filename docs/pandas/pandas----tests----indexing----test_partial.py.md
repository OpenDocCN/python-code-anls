# `D:\src\scipysrc\pandas\pandas\tests\indexing\test_partial.py`

```
"""
test setting *parts* of objects both positionally and label based

TODO: these should be split among the indexer tests
"""

# 导入必要的库
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
)
# 导入日期相关的函数和对象
from pandas import Timestamp, date_range, period_range
import pandas._testing as tm

# 定义测试类 TestEmptyFrameSetitemExpansion
class TestEmptyFrameSetitemExpansion:
    
    # 测试空数据框设置项目并保留索引名称
    def test_empty_frame_setitem_index_name_retained(self):
        # 创建一个空数据框，使用指定的索引名称
        df = DataFrame({}, index=pd.RangeIndex(0, name="df_index"))
        # 创建一个序列对象，指定索引名称
        series = Series(1.23, index=pd.RangeIndex(4, name="series_index"))

        # 向数据框中插入序列对象
        df["series"] = series
        
        # 预期的数据框，包含指定的列和索引
        expected = DataFrame(
            {"series": [1.23] * 4},
            index=pd.RangeIndex(4, name="df_index"),
            columns=Index(["series"], dtype=object),
        )

        # 使用测试工具函数验证数据框是否符合预期
        tm.assert_frame_equal(df, expected)

    # 测试空数据框设置项目并继承索引名称
    def test_empty_frame_setitem_index_name_inherited(self):
        # 创建一个空数据框
        df = DataFrame()
        # 创建一个序列对象，指定索引名称
        series = Series(1.23, index=pd.RangeIndex(4, name="series_index"))
        
        # 向数据框中插入序列对象
        df["series"] = series
        
        # 预期的数据框，包含指定的列和索引
        expected = DataFrame(
            {"series": [1.23] * 4},
            index=pd.RangeIndex(4, name="series_index"),
            columns=Index(["series"], dtype=object),
        )
        # 使用测试工具函数验证数据框是否符合预期
        tm.assert_frame_equal(df, expected)

    # 测试通过 loc 方法设置长度为零的序列，确保列对齐
    def test_loc_setitem_zerolen_series_columns_align(self):
        # 创建一个具有指定列名的数据框
        df = DataFrame(columns=["A", "B"])
        # 使用 loc 方法向指定行插入序列对象
        df.loc[0] = Series(1, index=range(4))
        
        # 预期的数据框，包含指定的列和索引
        expected = DataFrame(columns=["A", "B"], index=[0], dtype=np.float64)
        
        # 使用测试工具函数验证数据框是否符合预期
        tm.assert_frame_equal(df, expected)

        # 再次创建具有指定列名的数据框
        df = DataFrame(columns=["A", "B"])
        # 使用 loc 方法向指定行插入序列对象
        df.loc[0] = Series(1, index=["B"])

        # 预期的数据框，包含指定的列和索引
        exp = DataFrame([[np.nan, 1]], columns=["A", "B"], index=[0], dtype="float64")
        
        # 使用测试工具函数验证数据框是否符合预期
        tm.assert_frame_equal(df, exp)

    # 测试通过 loc 方法设置长度为零的列表，确保长度匹配数据框列数
    def test_loc_setitem_zerolen_list_length_must_match_columns(self):
        # 创建一个具有指定列名的数据框
        df = DataFrame(columns=["A", "B"])

        # 预期的错误信息
        msg = "cannot set a row with mismatched columns"
        
        # 使用 pytest 的断言检查是否引发了预期的 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            df.loc[0] = [1, 2, 3]

        # 再次创建具有指定列名的数据框
        df = DataFrame(columns=["A", "B"])
        # 使用 loc 方法向指定行插入列表对象
        df.loc[3] = [6, 7]  # 长度与 len(df.columns) 匹配 --> 正确!

        # 预期的数据框，包含指定的列和索引
        exp = DataFrame([[6, 7]], index=[3], columns=["A", "B"], dtype=np.int64)
        
        # 使用测试工具函数验证数据框是否符合预期
        tm.assert_frame_equal(df, exp)

    # 测试部分设置空数据框
    def test_partial_set_empty_frame(self):
        # 创建一个空数据框
        df = DataFrame()

        # 预期的错误信息
        msg = "cannot set a frame with no defined columns"

        # 使用 pytest 的断言检查是否引发了预期的 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            df.loc[1] = 1

        with pytest.raises(ValueError, match=msg):
            df.loc[1] = Series([1], index=["foo"])

        msg = "cannot set a frame with no defined index and a scalar"
        with pytest.raises(ValueError, match=msg):
            df.loc[:, 1] = 1
    def test_partial_set_empty_frame2(self):
        # 这些操作并不真正改变任何内容，只是改变了索引
        # GH#5632
        expected = DataFrame(
            columns=Index(["foo"], dtype=object), index=Index([], dtype="object")
        )

        # 创建一个空的 DataFrame，指定了索引为空的情况
        df = DataFrame(index=Index([], dtype="object"))
        # 在 DataFrame 中添加一个名为 "foo" 的空 Series，指定数据类型为 object
        df["foo"] = Series([], dtype="object")

        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

        # 创建一个空的 DataFrame，索引为空
        df = DataFrame(index=Index([]))
        # 在 DataFrame 中添加一个名为 "foo" 的 Series，其数据来源于 DataFrame 的索引
        df["foo"] = Series(df.index)

        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

        # 创建一个空的 DataFrame，索引为空
        df = DataFrame(index=Index([]))
        # 在 DataFrame 中添加一个名为 "foo" 的 Series，其数据为 DataFrame 的索引
        df["foo"] = df.index

        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

    def test_partial_set_empty_frame3(self):
        expected = DataFrame(
            columns=Index(["foo"], dtype=object), index=Index([], dtype="int64")
        )
        # 将 "foo" 列的数据类型从 object 改为 float64
        expected["foo"] = expected["foo"].astype("float64")

        # 创建一个空的 DataFrame，指定索引为 int64 类型
        df = DataFrame(index=Index([], dtype="int64"))
        # 在 DataFrame 中添加一个名为 "foo" 的空列表
        df["foo"] = []

        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

        # 创建一个空的 DataFrame，指定索引为 int64 类型
        df = DataFrame(index=Index([], dtype="int64"))
        # 在 DataFrame 中添加一个名为 "foo" 的 Series，其数据为从 0 到长度减一的整数，数据类型为 float64
        df["foo"] = Series(np.arange(len(df)), dtype="float64")

        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

    def test_partial_set_empty_frame4(self):
        # 创建一个空的 DataFrame，指定索引为 int64 类型
        df = DataFrame(index=Index([], dtype="int64"))
        # 在 DataFrame 中添加一个名为 "foo" 的 Series，其数据为从 0 到长度减一的整数
        df["foo"] = range(len(df))

        expected = DataFrame(
            columns=Index(["foo"], dtype=object), index=Index([], dtype="int64")
        )
        # 将 "foo" 列的数据类型从 object 改为 int64
        expected["foo"] = expected["foo"].astype("int64")

        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

    def test_partial_set_empty_frame5(self):
        # 创建一个空的 DataFrame
        df = DataFrame()
        # 断言 DataFrame 的列索引是否与 RangeIndex(0) 相等
        tm.assert_index_equal(df.columns, pd.RangeIndex(0))
        # 创建另一个空的 DataFrame
        df2 = DataFrame()
        # 向 DataFrame 中添加一个名为 1 的列，其值为 Series([1])，索引为 ["foo"]
        df2[1] = Series([1], index=["foo"])
        # 在 df 的所有行中的列 1 中插入 Series([1])，索引为 ["foo"]
        df.loc[:, 1] = Series([1], index=["foo"])
        # 断言两个 DataFrame 是否相等，期望结果为包含值为 1 的 DataFrame，索引为 ["foo"]，列为 [1]
        tm.assert_frame_equal(df, DataFrame([[1]], index=["foo"], columns=[1]))
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(df, df2)

    def test_partial_set_empty_frame_no_index(self):
        # 开始时没有索引
        # 创建一个 DataFrame，列为 ["A", "B"]，无索引
        expected = DataFrame({0: Series(1, index=range(4))}, columns=["A", "B", 0])

        # 创建一个空的 DataFrame，列为 ["A", "B"]
        df = DataFrame(columns=["A", "B"])
        # 在 DataFrame 中添加一个名为 0 的列，其值为 Series(1, index=range(4))
        df[0] = Series(1, index=range(4))
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

        # 创建一个空的 DataFrame，列为 ["A", "B"]
        df = DataFrame(columns=["A", "B"])
        # 在 df 的所有行中的列 0 中插入 Series(1, index=range(4))
        df.loc[:, 0] = Series(1, index=range(4))
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)
    def test_partial_set_empty_frame_row(self):
        # GH#5720, GH#5744
        # GH#5720 和 GH#5744 的相关问题参考
        # 当空时不创建行
        expected = DataFrame(columns=["A", "B", "New"], index=Index([], dtype="int64"))
        # 将列"A"转换为整数类型"int64"
        expected["A"] = expected["A"].astype("int64")
        # 将列"B"转换为浮点数类型"float64"
        expected["B"] = expected["B"].astype("float64")
        # 将列"New"转换为浮点数类型"float64"
        expected["New"] = expected["New"].astype("float64")

        df = DataFrame({"A": [1, 2, 3], "B": [1.2, 4.2, 5.2]})
        y = df[df.A > 5]
        # 设置"New"列的值为NaN
        y["New"] = np.nan
        # 断言DataFrame y 和预期的 DataFrame 相等
        tm.assert_frame_equal(y, expected)

        expected = DataFrame(columns=["a", "b", "c c", "d"])
        # 将列"d"转换为整数类型"int64"
        expected["d"] = expected["d"].astype("int64")
        df = DataFrame(columns=["a", "b", "c c"])
        # 将列"d"设置为值3
        df["d"] = 3
        # 断言DataFrame df 和预期的 DataFrame 相等
        tm.assert_frame_equal(df, expected)
        # 断言Series df["c c"] 和预期的 Series 相等
        tm.assert_series_equal(df["c c"], Series(name="c c", dtype=object))

        # 重新索引列是可以的
        df = DataFrame({"A": [1, 2, 3], "B": [1.2, 4.2, 5.2]})
        y = df[df.A > 5]
        # 对y进行列重新索引为["A", "B", "C"]
        result = y.reindex(columns=["A", "B", "C"])
        expected = DataFrame(columns=["A", "B", "C"])
        # 将列"A"转换为整数类型"int64"
        expected["A"] = expected["A"].astype("int64")
        # 将列"B"转换为浮点数类型"float64"
        expected["B"] = expected["B"].astype("float64")
        # 将列"C"转换为浮点数类型"float64"
        expected["C"] = expected["C"].astype("float64")
        # 断言DataFrame result 和预期的 DataFrame 相等
        tm.assert_frame_equal(result, expected)

    def test_partial_set_empty_frame_set_series(self):
        # GH#5756
        # 使用空Series进行设置
        df = DataFrame(Series(dtype=object))
        expected = DataFrame({0: Series(dtype=object)})
        # 断言DataFrame df 和预期的 DataFrame 相等
        tm.assert_frame_equal(df, expected)

        df = DataFrame(Series(name="foo", dtype=object))
        expected = DataFrame({"foo": Series(dtype=object)})
        # 断言DataFrame df 和预期的 DataFrame 相等
        tm.assert_frame_equal(df, expected)

    def test_partial_set_empty_frame_empty_copy_assignment(self):
        # GH#5932
        # 在空DataFrame上进行复制和赋值
        df = DataFrame(index=[0])
        # 复制DataFrame df
        df = df.copy()
        # 将"a"列设置为值0
        df["a"] = 0
        expected = DataFrame(0, index=[0], columns=Index(["a"], dtype=object))
        # 断言DataFrame df 和预期的 DataFrame 相等
        tm.assert_frame_equal(df, expected)

    def test_partial_set_empty_frame_empty_consistencies(self, using_infer_string):
        # GH#6171
        # 空DataFrame的一致性
        df = DataFrame(columns=["x", "y"])
        # 将"x"列设置为[1, 2]
        df["x"] = [1, 2]
        expected = DataFrame({"x": [1, 2], "y": [np.nan, np.nan]})
        # 断言DataFrame df 和预期的 DataFrame 相等，不检查数据类型
        tm.assert_frame_equal(df, expected, check_dtype=False)

        df = DataFrame(columns=["x", "y"])
        # 将"x"列设置为["1", "2"]
        df["x"] = ["1", "2"]
        expected = DataFrame(
            {
                "x": Series(
                    ["1", "2"],
                    dtype=object if not using_infer_string else "string[pyarrow_numpy]",
                ),
                "y": Series([np.nan, np.nan], dtype=object),
            }
        )
        # 断言DataFrame df 和预期的 DataFrame 相等
        tm.assert_frame_equal(df, expected)

        df = DataFrame(columns=["x", "y"])
        # 在位置(0, "x")设置值1
        df.loc[0, "x"] = 1
        expected = DataFrame({"x": [1], "y": [np.nan]})
        # 断言DataFrame df 和预期的 DataFrame 相等，不检查数据类型
        tm.assert_frame_equal(df, expected, check_dtype=False)
class TestPartialSetting:
    def test_partial_setting(self):
        # GH2578, allow ix and friends to partially set
        # GH2578号问题，允许ix等函数进行部分设置操作

        # series
        # 创建原始Series对象
        s_orig = Series([1, 2, 3])

        # 复制原始Series对象
        s = s_orig.copy()
        # 在索引位置5处设置值为5
        s[5] = 5
        # 期望的Series对象，新增索引5，值为5
        expected = Series([1, 2, 3, 5], index=[0, 1, 2, 5])
        tm.assert_series_equal(s, expected)

        # 复制原始Series对象
        s = s_orig.copy()
        # 使用loc方法在索引位置5处设置值为5
        s.loc[5] = 5
        # 期望的Series对象，新增索引5，值为5
        expected = Series([1, 2, 3, 5], index=[0, 1, 2, 5])
        tm.assert_series_equal(s, expected)

        # 复制原始Series对象
        s = s_orig.copy()
        # 在索引位置5处设置值为5.0
        s[5] = 5.0
        # 期望的Series对象，新增索引5，值为5.0
        expected = Series([1, 2, 3, 5.0], index=[0, 1, 2, 5])
        tm.assert_series_equal(s, expected)

        # 复制原始Series对象
        s = s_orig.copy()
        # 使用loc方法在索引位置5处设置值为5.0
        s.loc[5] = 5.0
        # 期望的Series对象，新增索引5，值为5.0
        expected = Series([1, 2, 3, 5.0], index=[0, 1, 2, 5])
        tm.assert_series_equal(s, expected)

        # iloc/iat raise
        # 复制原始Series对象
        s = s_orig.copy()

        # 测试iloc方法，预期会引发IndexError异常
        msg = "iloc cannot enlarge its target object"
        with pytest.raises(IndexError, match=msg):
            s.iloc[3] = 5.0

        # 测试iat方法，预期会引发IndexError异常
        msg = "index 3 is out of bounds for axis 0 with size 3"
        with pytest.raises(IndexError, match=msg):
            s.iat[3] = 5.0
    # 定义测试方法，验证部分设置框架功能
    def test_partial_setting_frame(self):
        # 创建原始数据框 df_orig，包含6个元素，重塑为3行2列，列名为"A"和"B"，数据类型为int64
        df_orig = DataFrame(
            np.arange(6).reshape(3, 2), columns=["A", "B"], dtype="int64"
        )

        # 复制 df_orig 到 df
        df = df_orig.copy()

        # 定义错误消息
        msg = "iloc cannot enlarge its target object"
        # 使用 pytest 的上下文管理器，验证通过 iloc 设置超出范围的值是否引发 IndexError，并匹配特定消息
        with pytest.raises(IndexError, match=msg):
            df.iloc[4, 2] = 5.0

        # 重新定义错误消息
        msg = "index 2 is out of bounds for axis 0 with size 2"
        # 使用 pytest 的上下文管理器，验证通过 iat 设置超出范围的值是否引发 IndexError，并匹配特定消息
        with pytest.raises(IndexError, match=msg):
            df.iat[4, 2] = 5.0

        # 定义预期结果数据框 expected，将 df_orig 的第二行内容复制到 df 的第一行
        expected = DataFrame({"A": [0, 4, 4], "B": [1, 5, 5]})
        df = df_orig.copy()
        df.iloc[1] = df.iloc[2]
        # 使用 assert_frame_equal 验证 df 和 expected 是否相等
        tm.assert_frame_equal(df, expected)

        # 重新定义预期结果数据框 expected，将 df_orig 的第二行内容复制到 df 的第一行
        expected = DataFrame({"A": [0, 4, 4], "B": [1, 5, 5]})
        df = df_orig.copy()
        df.loc[1] = df.loc[2]
        # 使用 assert_frame_equal 验证 df 和 expected 是否相等
        tm.assert_frame_equal(df, expected)

        # 与2578类似，使用 loc 进行部分设置，并保持数据类型不变
        expected = DataFrame({"A": [0, 2, 4, 4], "B": [1, 3, 5, 5]})
        df = df_orig.copy()
        df.loc[3] = df.loc[2]
        # 使用 assert_frame_equal 验证 df 和 expected 是否相等
        tm.assert_frame_equal(df, expected)

        # 单一数据类型的数据框，进行数据覆盖
        expected = DataFrame({"A": [0, 2, 4], "B": [0, 2, 4]})
        df = df_orig.copy()
        df.loc[:, "B"] = df.loc[:, "A"]
        # 使用 assert_frame_equal 验证 df 和 expected 是否相等
        tm.assert_frame_equal(df, expected)

        # 混合数据类型的数据框，将列"B"的数据类型转换为 np.float64，并进行数据覆盖
        expected = DataFrame({"A": [0, 2, 4], "B": Series([0.0, 2.0, 4.0])})
        df = df_orig.copy()
        df["B"] = df["B"].astype(np.float64)
        # 从版本2.0开始，尝试在 inplace 设置 df.loc[:, "B"] = ... 成功
        df.loc[:, "B"] = df.loc[:, "A"]
        # 使用 assert_frame_equal 验证 df 和 expected 是否相等
        tm.assert_frame_equal(df, expected)

        # 单一数据类型的数据框，进行部分设置
        expected = df_orig.copy()
        expected["C"] = df["A"]
        df = df_orig.copy()
        df.loc[:, "C"] = df.loc[:, "A"]
        # 使用 assert_frame_equal 验证 df 和 expected 是否相等
        tm.assert_frame_equal(df, expected)

        # 混合数据类型的数据框，进行部分设置
        expected = df_orig.copy()
        expected["C"] = df["A"]
        df = df_orig.copy()
        df.loc[:, "C"] = df.loc[:, "A"]
        # 使用 assert_frame_equal 验证 df 和 expected 是否相等
        tm.assert_frame_equal(df, expected)
    def test_partial_setting2(self):
        # 测试GH 8473功能点
        # 创建日期范围对象，从"1/1/2000"开始，生成8个时间点
        dates = date_range("1/1/2000", periods=8)
        # 创建一个包含随机标准正态分布数据的DataFrame，8行4列，行索引为dates，列名为["A", "B", "C", "D"]
        df_orig = DataFrame(
            np.random.default_rng(2).standard_normal((8, 4)),
            index=dates,
            columns=["A", "B", "C", "D"],
        )

        # 生成预期的DataFrame，通过在最后一个日期后追加一个新行{"A": 7}，并排序列
        expected = pd.concat(
            [df_orig, DataFrame({"A": 7}, index=dates[-1:] + dates.freq)], sort=True
        )
        # 复制原始DataFrame
        df = df_orig.copy()
        # 使用loc方法设置特定位置的值为7
        df.loc[dates[-1] + dates.freq, "A"] = 7
        # 断言DataFrame是否与预期一致
        tm.assert_frame_equal(df, expected)
        # 再次复制原始DataFrame
        df = df_orig.copy()
        # 使用at方法设置特定位置的值为7
        df.at[dates[-1] + dates.freq, "A"] = 7
        # 断言DataFrame是否与预期一致
        tm.assert_frame_equal(df, expected)

        # 创建预期的DataFrame，通过在最后一个日期后追加一个新列{0: 7}
        exp_other = DataFrame({0: 7}, index=dates[-1:] + dates.freq)
        expected = pd.concat([df_orig, exp_other], axis=1)

        # 复制原始DataFrame
        df = df_orig.copy()
        # 使用loc方法设置特定位置的值为7
        df.loc[dates[-1] + dates.freq, 0] = 7
        # 断言DataFrame是否与预期一致
        tm.assert_frame_equal(df, expected)
        # 再次复制原始DataFrame
        df = df_orig.copy()
        # 使用at方法设置特定位置的值为7
        df.at[dates[-1] + dates.freq, 0] = 7
        # 断言DataFrame是否与预期一致
        tm.assert_frame_equal(df, expected)
    # 定义测试函数 test_series_partial_set_with_name
    def test_series_partial_set_with_name(self):
        # 问题编号 GH 11497

        # 创建一个索引对象 idx，包含整数值 [1, 2]，数据类型为 int64，命名为 "idx"
        idx = Index([1, 2], dtype="int64", name="idx")
        
        # 创建一个序列对象 ser，包含浮点数值 [0.1, 0.2]，使用上述索引 idx，并命名为 "s"
        ser = Series([0.1, 0.2], index=idx, name="s")

        # 使用 loc 方法进行索引操作
        # 测试：期望引发 KeyError 异常，匹配正则表达式 "\[3\] not in index"
        with pytest.raises(KeyError, match=r"\[3\] not in index"):
            ser.loc[[3, 2, 3]]

        # 测试：期望引发 KeyError 异常，匹配字符串 "not in index"
        with pytest.raises(KeyError, match=r"not in index"):
            ser.loc[[3, 2, 3, "x"]]

        # 创建一个期望的索引对象 exp_idx，包含整数值 [2, 2, 1]，数据类型为 int64，命名为 "idx"
        exp_idx = Index([2, 2, 1], dtype="int64", name="idx")
        
        # 创建一个期望的序列对象 expected，包含浮点数值 [0.2, 0.2, 0.1]，使用上述 exp_idx，并命名为 "s"
        expected = Series([0.2, 0.2, 0.1], index=exp_idx, name="s")
        
        # 执行 ser.loc[[2, 2, 1]]，并将结果存储在 result 中
        result = ser.loc[[2, 2, 1]]
        
        # 使用测试工具 tm.assert_series_equal 检查 result 和 expected 是否相等，检查索引类型
        tm.assert_series_equal(result, expected, check_index_type=True)

        # 测试：期望引发 KeyError 异常，匹配正则表达式 "\['x'\] not in index"
        with pytest.raises(KeyError, match=r"\['x'\] not in index"):
            ser.loc[[2, 2, "x", 1]]

        # 测试：期望引发 KeyError 异常，匹配 msg 变量中定义的详细消息字符串
        msg = (
            rf"\"None of \[Index\(\[3, 3, 3\], dtype='{np.dtype(int)}', "
            r"name='idx'\)\] are in the \[index\]\""
        )
        with pytest.raises(KeyError, match=msg):
            ser.loc[[3, 3, 3]]

        # 测试：期望引发 KeyError 异常，匹配字符串 "not in index"
        with pytest.raises(KeyError, match="not in index"):
            ser.loc[[2, 2, 3]]

        # 创建一个索引对象 idx，包含整数值 [1, 2, 3]，数据类型为 int64，命名为 "idx"
        idx = Index([1, 2, 3], dtype="int64", name="idx")
        
        # 测试：期望引发 KeyError 异常，匹配字符串 "not in index"
        with pytest.raises(KeyError, match="not in index"):
            Series([0.1, 0.2, 0.3], index=idx, name="s").loc[[3, 4, 4]]

        # 创建一个索引对象 idx，包含整数值 [1, 2, 3, 4]，数据类型为 int64，命名为 "idx"
        idx = Index([1, 2, 3, 4], dtype="int64", name="idx")
        
        # 测试：期望引发 KeyError 异常，匹配字符串 "not in index"
        with pytest.raises(KeyError, match="not in index"):
            Series([0.1, 0.2, 0.3, 0.4], index=idx, name="s").loc[[5, 3, 3]]

        # 创建一个索引对象 idx，包含整数值 [1, 2, 3, 4]，数据类型为 int64，命名为 "idx"
        idx = Index([1, 2, 3, 4], dtype="int64", name="idx")
        
        # 测试：期望引发 KeyError 异常，匹配字符串 "not in index"
        with pytest.raises(KeyError, match="not in index"):
            Series([0.1, 0.2, 0.3, 0.4], index=idx, name="s").loc[[5, 4, 4]]

        # 创建一个索引对象 idx，包含整数值 [4, 5, 6, 7]，数据类型为 int64，命名为 "idx"
        idx = Index([4, 5, 6, 7], dtype="int64", name="idx")
        
        # 测试：期望引发 KeyError 异常，匹配字符串 "not in index"
        with pytest.raises(KeyError, match="not in index"):
            Series([0.1, 0.2, 0.3, 0.4], index=idx, name="s").loc[[7, 2, 2]]

        # 创建一个索引对象 idx，包含整数值 [1, 2, 3, 4]，数据类型为 int64，命名为 "idx"
        idx = Index([1, 2, 3, 4], dtype="int64", name="idx")
        
        # 测试：期望引发 KeyError 异常，匹配字符串 "not in index"
        with pytest.raises(KeyError, match="not in index"):
            Series([0.1, 0.2, 0.3, 0.4], index=idx, name="s").loc[[4, 5, 5]]

        # 使用 iloc 方法进行索引操作
        # 创建一个期望的索引对象 exp_idx，包含整数值 [2, 2, 1, 1]，数据类型为 int64，命名为 "idx"
        exp_idx = Index([2, 2, 1, 1], dtype="int64", name="idx")
        
        # 创建一个期望的序列对象 expected，包含浮点数值 [0.2, 0.2, 0.1, 0.1]，使用上述 exp_idx，并命名为 "s"
        expected = Series([0.2, 0.2, 0.1, 0.1], index=exp_idx, name="s")
        
        # 执行 ser.iloc[[1, 1, 0, 0]]，并将结果存储在 result 中
        result = ser.iloc[[1, 1, 0, 0]]
        
        # 使用测试工具 tm.assert_series_equal 检查 result 和 expected 是否相等，检查索引类型
        tm.assert_series_equal(result, expected, check_index_type=True)
    # 定义测试函数，用于测试将数值扩展到日期时间索引中的设置操作
    def test_setitem_with_expansion_numeric_into_datetimeindex(self, key):
        # GH#4940 插入非字符串类型数据
        # 创建一个原始的 DataFrame，包含随机生成的数据，列名为 ['A', 'B', 'C', 'D']，索引为工作日频率的日期范围
        orig = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        # 复制原始 DataFrame
        df = orig.copy()

        # 将索引为 key 的行设为第一行的数据
        df.loc[key, :] = df.iloc[0]

        # 扩展后的索引，包含原始索引和新的 key
        ex_index = Index(list(orig.index) + [key], dtype=object, name=orig.index.name)
        # 扩展后的数据，包含原始数据和第一行数据的复制
        ex_data = np.concatenate([orig.values, df.iloc[[0]].values], axis=0)
        # 期望的 DataFrame，使用扩展后的数据和索引
        expected = DataFrame(ex_data, index=ex_index, columns=orig.columns)

        # 使用测试工具函数比较 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

    # 定义测试函数，用于测试部分设置无效值的情况
    def test_partial_set_invalid(self):
        # GH 4940 允许只设置“有效”的值

        # 创建一个原始的 DataFrame，包含随机生成的数据，列名为 ['A', 'B', 'C', 'D']，索引为工作日频率的日期范围
        orig = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )

        # 在这里允许对象转换
        df = orig.copy()

        # 将索引为 "a" 的行设为第一行的数据
        df.loc["a", :] = df.iloc[0]

        # 创建一个 Series，包含第一行数据，并命名为 "a"
        ser = Series(df.iloc[0], name="a")
        # 期望的 DataFrame，将原始数据和新的 Series 水平连接
        exp = pd.concat([orig, DataFrame(ser).T.infer_objects()])

        # 使用测试工具函数比较 DataFrame 是否相等
        tm.assert_frame_equal(df, exp)
        # 使用测试工具函数比较索引是否相等
        tm.assert_index_equal(df.index, Index(orig.index.tolist() + ["a"]))
        # 断言 DataFrame 的索引类型为对象
        assert df.index.dtype == "object"

    # 使用 pytest 的参数化装饰器，定义测试函数，测试使用字符串列表表示日期时间的 loc 方法
    @pytest.mark.parametrize(
        "idx,labels,expected_idx",
        [
            (
                period_range(start="2000", periods=20, freq="D"),
                ["2000-01-04", "2000-01-08", "2000-01-12"],
                [
                    Period("2000-01-04", freq="D"),
                    Period("2000-01-08", freq="D"),
                    Period("2000-01-12", freq="D"),
                ],
            ),
            (
                date_range(start="2000", periods=20, freq="D", unit="s"),
                ["2000-01-04", "2000-01-08", "2000-01-12"],
                [
                    Timestamp("2000-01-04"),
                    Timestamp("2000-01-08"),
                    Timestamp("2000-01-12"),
                ],
            ),
            (
                pd.timedelta_range(start="1 day", periods=20),
                ["4D", "8D", "12D"],
                [pd.Timedelta("4 day"), pd.Timedelta("8 day"), pd.Timedelta("12 day")],
            ),
        ],
    )
    # 定义测试函数，测试使用字符串列表表示日期时间的 loc 方法
    def test_loc_with_list_of_strings_representing_datetimes(
        self, idx, labels, expected_idx, frame_or_series
    ):
        # GH 11278

        # 创建一个以 frame_or_series 为构造函数，索引为 idx 的对象
        obj = frame_or_series(range(20), index=idx)

        # 期望的值，使用 frame_or_series 构造器，数据为 [3, 7, 11]，索引为 expected_idx
        expected_value = [3, 7, 11]
        expected = frame_or_series(expected_value, expected_idx)

        # 使用测试工具函数比较 obj.loc[labels] 是否等于 expected
        tm.assert_equal(expected, obj.loc[labels])
        # 如果 frame_or_series 是 Series，则使用测试工具函数比较 expected 和 obj[labels] 是否相等
        if frame_or_series is Series:
            tm.assert_series_equal(expected, obj[labels])
    @pytest.mark.parametrize(
        "idx,labels",
        [  # 参数化测试用例，idx 表示索引，labels 表示标签列表
            (
                period_range(start="2000", periods=20, freq="D"),  # 创建一个时间段的索引，从2000年开始，20个时间点，每日频率
                ["2000-01-04", "2000-01-30"],  # 预期的索引标签列表
            ),
            (
                date_range(start="2000", periods=20, freq="D"),  # 创建一个日期范围的索引，从2000年开始，20个日期，每日频率
                ["2000-01-04", "2000-01-30"],  # 预期的索引标签列表
            ),
            (
                pd.timedelta_range(start="1 day", periods=20),  # 创建一个时间增量范围的索引，从1天开始，20个时间增量
                ["3 day", "30 day"],  # 预期的索引标签列表
            ),
        ],
    )
    def test_loc_with_list_of_strings_representing_datetimes_missing_value(
        self, idx, labels
    ):
        # GH 11278：参考GitHub issue编号11278
        ser = Series(range(20), index=idx)  # 创建一个具有给定索引的Series对象
        df = DataFrame(range(20), index=idx)  # 创建一个具有给定索引的DataFrame对象
        msg = r"not in index"  # 错误消息字符串，指示索引中缺少值的情况

        with pytest.raises(KeyError, match=msg):  # 断言捕获到 KeyError 异常，并且异常消息匹配预期的消息
            ser.loc[labels]  # 在Series上使用loc属性，根据标签列表访问数据，预期抛出KeyError异常
        with pytest.raises(KeyError, match=msg):
            ser[labels]  # 在Series上直接使用标签列表访问数据，预期抛出KeyError异常
        with pytest.raises(KeyError, match=msg):
            df.loc[labels]  # 在DataFrame上使用loc属性，根据标签列表访问数据，预期抛出KeyError异常

    @pytest.mark.parametrize(
        "idx,labels,msg",
        [  # 参数化测试用例，idx 表示索引，labels 表示标签，msg 表示预期的错误消息
            (
                period_range(start="2000", periods=20, freq="D"),  # 创建一个时间段的索引，从2000年开始，20个时间点，每日频率
                Index(["4D", "8D"], dtype=object),  # 创建一个索引对象，包含字符串 "4D" 和 "8D"
                (
                    r"None of \[Index\(\['4D', '8D'\], dtype='object'\)\] "
                    r"are in the \[index\]"  # 预期的错误消息，指示索引中缺少指定的标签
                ),
            ),
            (
                date_range(start="2000", periods=20, freq="D"),  # 创建一个日期范围的索引，从2000年开始，20个日期，每日频率
                Index(["4D", "8D"], dtype=object),  # 创建一个索引对象，包含字符串 "4D" 和 "8D"
                (
                    r"None of \[Index\(\['4D', '8D'\], dtype='object'\)\] "
                    r"are in the \[index\]"  # 预期的错误消息，指示索引中缺少指定的标签
                ),
            ),
            (
                pd.timedelta_range(start="1 day", periods=20),  # 创建一个时间增量范围的索引，从1天开始，20个时间增量
                Index(["2000-01-04", "2000-01-08"], dtype=object),  # 创建一个索引对象，包含日期字符串 "2000-01-04" 和 "2000-01-08"
                (
                    r"None of \[Index\(\['2000-01-04', '2000-01-08'\], "
                    r"dtype='object'\)\] are in the \[index\]"  # 预期的错误消息，指示索引中缺少指定的标签
                ),
            ),
        ],
    )
    def test_loc_with_list_of_strings_representing_datetimes_not_matched_type(
        self, idx, labels, msg
    ):
        # GH 11278：参考GitHub issue编号11278
        ser = Series(range(20), index=idx)  # 创建一个具有给定索引的Series对象
        df = DataFrame(range(20), index=idx)  # 创建一个具有给定索引的DataFrame对象

        with pytest.raises(KeyError, match=msg):  # 断言捕获到 KeyError 异常，并且异常消息匹配预期的消息
            ser.loc[labels]  # 在Series上使用loc属性，根据标签列表访问数据，预期抛出KeyError异常
        with pytest.raises(KeyError, match=msg):
            ser[labels]  # 在Series上直接使用标签列表访问数据，预期抛出KeyError异常
        with pytest.raises(KeyError, match=msg):
            df.loc[labels]  # 在DataFrame上使用loc属性，根据标签列表访问数据，预期抛出KeyError异常
class TestStringSlicing:
    def test_slice_irregular_datetime_index_with_nan(self):
        # 创建一个日期时间索引，包括一个空值（None）
        # GH36953
        index = pd.to_datetime(["2012-01-01", "2012-01-02", "2012-01-03", None])
        # 创建一个 DataFrame，索引为上述日期时间索引，数据为索引位置
        df = DataFrame(range(len(index)), index=index)
        # 创建预期的 DataFrame，索引为前三个有效日期时间索引，数据为相应索引位置
        expected = DataFrame(range(len(index[:3])), index=index[:3])
        # 使用 pytest 断言检测 KeyError 异常，匹配指定的错误信息
        with pytest.raises(KeyError, match="non-existing keys is not allowed"):
            # 对 DataFrame 进行切片，范围为 "2012-01-01" 到 "2012-01-04"
            # 上限不在索引内（因为索引是无序的）
            # GH53983
            # GH37819
            df["2012-01-01":"2012-01-04"]
        # 对 DataFrame 进行精确的时间切片，右边界精确到纳秒级别
        # 因为右边界切片会被"舍入"到提供的时间点之前的最大时间点
        # 例如，2012-01-03 会被舍入到 2012-01-04 的前一纳秒
        result = df["2012-01-01":"2012-01-03 00:00:00.000000000"]
        # 使用测试工具 tm.assert_frame_equal 检查结果与预期是否相等
        tm.assert_frame_equal(result, expected)
```