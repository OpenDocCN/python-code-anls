# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_join.py`

```
# 导入 datetime 模块中的 datetime 类和 zoneinfo 模块
from datetime import datetime
import zoneinfo

# 导入 numpy 和 pytest 模块
import numpy as np
import pytest

# 从 pandas.errors 模块中导入 MergeError 异常类
from pandas.errors import MergeError

# 导入 pandas 模块并重命名为 pd
import pandas as pd
# 从 pandas 模块中导入以下类和函数
from pandas import (
    DataFrame,       # 数据帧类
    Index,           # 索引类
    MultiIndex,      # 多重索引类
    date_range,      # 日期范围生成函数
    period_range,    # 时期范围生成函数
)
# 导入 pandas._testing 模块并重命名为 tm
import pandas._testing as tm
# 从 pandas.core.reshape.concat 模块中导入 concat 函数
from pandas.core.reshape.concat import concat


# 定义左侧无重复值的测试数据生成 fixture 函数
@pytest.fixture
def left_no_dup():
    return DataFrame(
        {"a": ["a", "b", "c", "d"], "b": ["cat", "dog", "weasel", "horse"]},
        index=range(4),
    )


# 定义右侧无重复值的测试数据生成 fixture 函数
@pytest.fixture
def right_no_dup():
    return DataFrame(
        {
            "a": ["a", "b", "c", "d", "e"],
            "c": ["meow", "bark", "um... weasel noise?", "nay", "chirp"],
        },
        index=range(5),
    ).set_index("a")


# 定义左侧有重复值的测试数据生成 fixture 函数
@pytest.fixture
def left_w_dups(left_no_dup):
    return concat(
        [left_no_dup, DataFrame({"a": ["a"], "b": ["cow"]}, index=[3])], sort=True
    )


# 定义右侧有重复值的测试数据生成 fixture 函数
@pytest.fixture
def right_w_dups(right_no_dup):
    return concat(
        [right_no_dup, DataFrame({"a": ["e"], "c": ["moo"]}, index=[3])]
    ).set_index("a")


# 定义测试用例 test_join，参数化多种情况进行测试
@pytest.mark.parametrize(
    "how, sort, expected",
    [
        ("inner", False, DataFrame({"a": [20, 10], "b": [200, 100]}, index=[2, 1])),
        ("inner", True, DataFrame({"a": [10, 20], "b": [100, 200]}, index=[1, 2])),
        (
            "left",
            False,
            DataFrame({"a": [20, 10, 0], "b": [200, 100, np.nan]}, index=[2, 1, 0]),
        ),
        (
            "left",
            True,
            DataFrame({"a": [0, 10, 20], "b": [np.nan, 100, 200]}, index=[0, 1, 2]),
        ),
        (
            "right",
            False,
            DataFrame({"a": [np.nan, 10, 20], "b": [300, 100, 200]}, index=[3, 1, 2]),
        ),
        (
            "right",
            True,
            DataFrame({"a": [10, 20, np.nan], "b": [100, 200, 300]}, index=[1, 2, 3]),
        ),
        (
            "outer",
            False,
            DataFrame(
                {"a": [0, 10, 20, np.nan], "b": [np.nan, 100, 200, 300]},
                index=[0, 1, 2, 3],
            ),
        ),
        (
            "outer",
            True,
            DataFrame(
                {"a": [0, 10, 20, np.nan], "b": [np.nan, 100, 200, 300]},
                index=[0, 1, 2, 3],
            ),
        ),
    ],
)
def test_join(how, sort, expected):
    left = DataFrame({"a": [20, 10, 0]}, index=[2, 1, 0])
    right = DataFrame({"b": [300, 100, 200]}, index=[3, 1, 2])
    result = left.join(right, how=how, sort=sort, validate="1:1")
    tm.assert_frame_equal(result, expected)


# 定义测试用例 test_suffix_on_list_join，测试在多个 DataFrame 进行连接时是否能正确抛出异常
def test_suffix_on_list_join():
    first = DataFrame({"key": [1, 2, 3, 4, 5]})
    second = DataFrame({"key": [1, 8, 3, 2, 5], "v1": [1, 2, 3, 4, 5]})
    third = DataFrame({"keys": [5, 2, 3, 4, 1], "v2": [1, 2, 3, 4, 5]})

    # 检查是否正确抛出异常信息
    msg = "Suffixes not supported when joining multiple DataFrames"
    with pytest.raises(ValueError, match=msg):
        first.join([second], lsuffix="y")
    # 使用 pytest 来测试是否会引发 ValueError 异常，并检查异常消息是否与给定的 msg 匹配
    with pytest.raises(ValueError, match=msg):
        # 调用 DataFrame 的 join 方法，传入第二个和第三个 DataFrame，设置 rsuffix 参数为 "x"
        first.join([second, third], rsuffix="x")
    
    # 使用 pytest 来测试是否会引发 ValueError 异常，并检查异常消息是否与给定的 msg 匹配
    with pytest.raises(ValueError, match=msg):
        # 调用 DataFrame 的 join 方法，传入第二个和第三个 DataFrame，设置 lsuffix 参数为 "y"，rsuffix 参数为 "x"
        first.join([second, third], lsuffix="y", rsuffix="x")
    
    # 使用 pytest 来测试是否会引发 ValueError 异常，并检查异常消息是否为 "Indexes have overlapping values"
    with pytest.raises(ValueError, match="Indexes have overlapping values"):
        # 调用 DataFrame 的 join 方法，传入第二个和第三个 DataFrame，未指定任何后缀参数
        first.join([second, third])
    
    # 没有引发异常的情况下进行的操作
    # 调用 DataFrame 的 join 方法，仅传入第三个 DataFrame，返回合并后的结果 arr_joined
    arr_joined = first.join([third])
    # 调用 DataFrame 的 join 方法，仅传入第三个 DataFrame，返回合并后的结果 norm_joined
    norm_joined = first.join(third)
    # 使用 pandas.testing.assert_frame_equal 方法来比较两个 DataFrame 是否相等
    tm.assert_frame_equal(arr_joined, norm_joined)
def test_join_invalid_validate(left_no_dup, right_no_dup):
    # GH 46622
    # 检查无效参数
    msg = (
        '"invalid" is not a valid argument. '
        "Valid arguments are:\n"
        '- "1:1"\n'
        '- "1:m"\n'
        '- "m:1"\n'
        '- "m:m"\n'
        '- "one_to_one"\n'
        '- "one_to_many"\n'
        '- "many_to_one"\n'
        '- "many_to_many"'
    )
    # 断言抛出 ValueError 异常，并匹配特定的错误消息
    with pytest.raises(ValueError, match=msg):
        left_no_dup.merge(right_no_dup, on="a", validate="invalid")


@pytest.mark.parametrize("dtype", ["object", "string[pyarrow]"])
def test_join_on_single_col_dup_on_right(left_no_dup, right_w_dups, dtype):
    # GH 46622
    # right_w_dups 允许重复，符合 one_to_many 约束
    if dtype == "string[pyarrow]":
        pytest.importorskip("pyarrow")
    left_no_dup = left_no_dup.astype(dtype)
    right_w_dups.index = right_w_dups.index.astype(dtype)
    left_no_dup.join(
        right_w_dups,
        on="a",
        validate="one_to_many",
    )

    # left_no_dup 不允许 right_w_dups 中的重复，不符合 one_to_one 约束
    msg = "Merge keys are not unique in right dataset; not a one-to-one merge"
    with pytest.raises(MergeError, match=msg):
        left_no_dup.join(
            right_w_dups,
            on="a",
            validate="one_to_one",
        )


def test_join_on_single_col_dup_on_left(left_w_dups, right_no_dup):
    # GH 46622
    # left_w_dups 允许重复，符合 many_to_one 约束
    left_w_dups.join(
        right_no_dup,
        on="a",
        validate="many_to_one",
    )

    # left_w_dups 不允许 right_no_dup 中的重复，不符合 one_to_one 约束
    msg = "Merge keys are not unique in left dataset; not a one-to-one merge"
    with pytest.raises(MergeError, match=msg):
        left_w_dups.join(
            right_no_dup,
            on="a",
            validate="one_to_one",
        )


def test_join_on_single_col_dup_on_both(left_w_dups, right_w_dups):
    # GH 46622
    # left_w_dups 和 right_w_dups 均允许重复，符合 many_to_many 约束
    left_w_dups.join(right_w_dups, on="a", validate="many_to_many")

    # left_w_dups 和 right_w_dups 均不允许重复，不符合 many_to_one 约束
    msg = "Merge keys are not unique in right dataset; not a many-to-one merge"
    with pytest.raises(MergeError, match=msg):
        left_w_dups.join(
            right_w_dups,
            on="a",
            validate="many_to_one",
        )

    # left_w_dups 和 right_w_dups 均不允许重复，不符合 one_to_many 约束
    msg = "Merge keys are not unique in left dataset; not a one-to-many merge"
    with pytest.raises(MergeError, match=msg):
        left_w_dups.join(
            right_w_dups,
            on="a",
            validate="one_to_many",
        )


def test_join_on_multi_col_check_dup():
    # GH 46622
    # 两列连接，左右均有重复，但联合后无重复
    left = DataFrame(
        {
            "a": ["a", "a", "b", "b"],
            "b": [0, 1, 0, 1],
            "c": ["cat", "dog", "weasel", "horse"],
        },
        index=range(4),
    ).set_index(["a", "b"])
    # 创建一个 DataFrame 对象 `right`，包含三列数据，其中'a'列为字符串列表，'b'列为整数列表，'d'列为字符串列表，同时设置'a'和'b'列作为索引
    right = DataFrame(
        {
            "a": ["a", "a", "b"],
            "b": [0, 1, 0],
            "d": ["meow", "bark", "um... weasel noise?"],
        },
        index=range(3),
    ).set_index(["a", "b"])
    
    # 创建一个 DataFrame 对象 `expected_multi`，包含四列数据，其中'a'列为字符串列表，'b'列为整数列表，'c'列为字符串列表，'d'列为字符串列表，同时设置'a'和'b'列作为索引
    expected_multi = DataFrame(
        {
            "a": ["a", "a", "b"],
            "b": [0, 1, 0],
            "c": ["cat", "dog", "weasel"],
            "d": ["meow", "bark", "um... weasel noise?"],
        },
        index=range(3),
    ).set_index(["a", "b"])
    
    # 使用左连接 `left.join(right, how="inner", validate="1:1")`，连接 `left` 和 `right` DataFrame 对象，保证连接满足一对一的验证条件
    result = left.join(right, how="inner", validate="1:1")
    
    # 使用 `tm.assert_frame_equal` 方法断言 `result` DataFrame 是否与 `expected_multi` DataFrame 相等
    tm.assert_frame_equal(result, expected_multi)
def test_join_index(float_frame):
    # left / right

    # 从 float_frame 中选择前 10 行的 A 和 B 列组成新的 DataFrame f
    f = float_frame.loc[float_frame.index[:10], ["A", "B"]]
    
    # 从 float_frame 中选择从第 5 行开始的 C 和 D 列，并逆序排列，组成新的 DataFrame f2
    f2 = float_frame.loc[float_frame.index[5:], ["C", "D"]].iloc[::-1]

    # 使用默认的 inner 连接方式将 f 和 f2 连接起来，得到 joined DataFrame
    joined = f.join(f2)
    # 断言 f 的索引与 joined 的索引相等
    tm.assert_index_equal(f.index, joined.index)
    # 期望的列名为 ["A", "B", "C", "D"]，断言 joined 的列名与期望的列名相等
    expected_columns = Index(["A", "B", "C", "D"])
    tm.assert_index_equal(joined.columns, expected_columns)

    # 使用 left 连接方式将 f 和 f2 连接起来，得到 joined DataFrame
    joined = f.join(f2, how="left")
    tm.assert_index_equal(joined.index, f.index)
    tm.assert_index_equal(joined.columns, expected_columns)

    # 使用 right 连接方式将 f 和 f2 连接起来，得到 joined DataFrame
    joined = f.join(f2, how="right")
    tm.assert_index_equal(joined.index, f2.index)
    tm.assert_index_equal(joined.columns, expected_columns)

    # inner 连接方式将 f 和 f2 连接起来，得到 joined DataFrame
    joined = f.join(f2, how="inner")
    tm.assert_index_equal(joined.index, f.index[5:10])
    tm.assert_index_equal(joined.columns, expected_columns)

    # outer 连接方式将 f 和 f2 连接起来，得到 joined DataFrame
    joined = f.join(f2, how="outer")
    tm.assert_index_equal(joined.index, float_frame.index.sort_values())
    tm.assert_index_equal(joined.columns, expected_columns)

    # 使用不支持的连接方式 "foo" 连接 f 和 f2，预期会引发 ValueError 异常
    with pytest.raises(ValueError, match="join method"):
        f.join(f2, how="foo")

    # corner case - 重叠的列名
    # 对于每种连接方式 ("outer", "left", "inner")，检查如果列名重叠且没有后缀时是否会引发 ValueError 异常
    msg = "columns overlap but no suffix"
    for how in ("outer", "left", "inner"):
        with pytest.raises(ValueError, match=msg):
            float_frame.join(float_frame, how=how)


def test_join_index_more(float_frame):
    # 从 float_frame 中选择所有行的 A 和 B 列组成新的 DataFrame af
    af = float_frame.loc[:, ["A", "B"]]
    # 从 float_frame 中选择每隔一行的 C 和 D 列组成新的 DataFrame bf
    bf = float_frame.loc[::2, ["C", "D"]]

    # 创建期望的 joined DataFrame，将 af 的数据与每隔一行的 bf 的 C 和 D 列数据合并
    expected = af.copy()
    expected["C"] = float_frame["C"][::2]
    expected["D"] = float_frame["D"][::2]

    # 使用默认的 inner 连接方式将 af 和 bf 连接起来，得到 result DataFrame
    result = af.join(bf)
    tm.assert_frame_equal(result, expected)

    # 使用 right 连接方式将 af 和 bf 连接起来，得到 result DataFrame
    result = af.join(bf, how="right")
    tm.assert_frame_equal(result, expected[::2])

    # 使用 right 连接方式将 bf 和 af 连接起来，得到 result DataFrame
    result = bf.join(af, how="right")
    tm.assert_frame_equal(result, expected.loc[:, result.columns])


def test_join_index_series(float_frame):
    # 复制 float_frame 数据，将最后一列转换为 Series，并与 float_frame 的其余列进行连接
    df = float_frame.copy()
    ser = df.pop(float_frame.columns[-1])
    joined = df.join(ser)

    # 断言 joined 与原始 float_frame 相等
    tm.assert_frame_equal(joined, float_frame)

    # 修改 Series 的名称为 None，预期会引发 ValueError 异常
    ser.name = None
    with pytest.raises(ValueError, match="must have a name"):
        df.join(ser)


def test_join_overlap(float_frame):
    # 从 float_frame 中选择所有行的 A、B、C 列组成新的 DataFrame df1
    df1 = float_frame.loc[:, ["A", "B", "C"]]
    # 从 float_frame 中选择所有行的 B、C、D 列组成新的 DataFrame df2
    df2 = float_frame.loc[:, ["B", "C", "D"]]

    # 使用 lsuffix 和 rsuffix 来处理重叠的列名，将 df1 和 df2 进行连接
    joined = df1.join(df2, lsuffix="_df1", rsuffix="_df2")
    df1_suf = df1.loc[:, ["B", "C"]].add_suffix("_df1")
    df2_suf = df2.loc[:, ["B", "C"]].add_suffix("_df2")

    # 期望的结果为 df1_suf、df2_suf 和 float_frame 的 A、D 列的连接结果
    expected = df1_suf.join(df2_suf).join(float_frame.loc[:, ["A", "D"]])

    # 列的顺序不一定排序，使用 assert_frame_equal 断言 joined 与期望的结果相等
    tm.assert_frame_equal(joined, expected.loc[:, joined.columns])


def test_join_period_index():
    # 创建一个具有周期索引的 DataFrame frame_with_period_index
    frame_with_period_index = DataFrame(
        data=np.arange(20).reshape(4, 5),
        columns=list("abcde"),
        index=period_range(start="2000", freq="Y", periods=4),
    )
    # 使用 rename 方法重命名 frame_with_period_index 的列，每个列名后面加上另一个相同的列名
    other = frame_with_period_index.rename(columns=lambda key: f"{key}{key}")
    # 将带有周期索引的数据帧的值沿着列方向重复一次，形成一个扩展后的数组
    joined_values = np.concatenate([frame_with_period_index.values] * 2, axis=1)
    
    # 将带有周期索引的数据帧的列与另一个数据帧的列合并
    joined_cols = frame_with_period_index.columns.append(other.columns)
    
    # 使用周期索引的数据帧与另一个数据帧进行连接操作
    joined = frame_with_period_index.join(other)
    
    # 创建一个期望的数据帧，其中数据来自扩展后的值数组，列名来自合并后的列索引，索引保持不变
    expected = DataFrame(
        data=joined_values, columns=joined_cols, index=frame_with_period_index.index
    )
    
    # 使用测试框架中的函数验证连接后的数据帧与期望的数据帧是否相等
    tm.assert_frame_equal(joined, expected)
def test_join_left_sequence_non_unique_index():
    # 创建第一个 DataFrame，列名为'a'，索引为[1, 2, 3]
    df1 = DataFrame({"a": [0, 10, 20]}, index=[1, 2, 3])
    # 创建第二个 DataFrame，列名为'b'，索引为[4, 3, 2]
    df2 = DataFrame({"b": [100, 200, 300]}, index=[4, 3, 2])
    # 创建第三个 DataFrame，列名为'c'，索引为[2, 2, 4]
    df3 = DataFrame({"c": [400, 500, 600]}, index=[2, 2, 4])

    # 使用左连接将 df2 和 df3 加入 df1
    joined = df1.join([df2, df3], how="left")

    # 创建预期的 DataFrame，包含列'a', 'b', 'c'，对应的索引为[1, 2, 2, 3]
    expected = DataFrame(
        {
            "a": [0, 10, 10, 20],
            "b": [np.nan, 300, 300, 200],
            "c": [np.nan, 400, 500, np.nan],
        },
        index=[1, 2, 2, 3],
    )

    # 使用断言检查 joined 和 expected 是否相等
    tm.assert_frame_equal(joined, expected)


def test_join_list_series(float_frame):
    # GH#46850
    # 使用一个包含 Series 和 DataFrame 的列表来连接 DataFrame
    left = float_frame.A.to_frame()
    right = [float_frame.B, float_frame[["C", "D"]]]
    result = left.join(right)

    # 使用断言检查 result 和 float_frame 是否相等
    tm.assert_frame_equal(result, float_frame)


def test_suppress_future_warning_with_sort_kw(sort):
    sort_kw = sort
    # 创建 DataFrame a，包含列'col1'，索引为['c', 'a']
    a = DataFrame({"col1": [1, 2]}, index=["c", "a"])

    # 创建 DataFrame b，包含列'col2'，索引为['b', 'a']
    b = DataFrame({"col2": [4, 5]}, index=["b", "a"])

    # 创建 DataFrame c，包含列'col3'，索引为['a', 'b']
    c = DataFrame({"col3": [7, 8]}, index=["a", "b"])

    # 创建预期的 DataFrame，包含列'col1', 'col2', 'col3'，根据 sort_kw 是否为 False 进行索引重排
    expected = DataFrame(
        {
            "col1": {"a": 2.0, "b": float("nan"), "c": 1.0},
            "col2": {"a": 5.0, "b": 4.0, "c": float("nan")},
            "col3": {"a": 7.0, "b": 8.0, "c": float("nan")},
        }
    )
    if sort_kw is False:
        expected = expected.reindex(index=["c", "a", "b"])

    # 使用 tm.assert_produces_warning(None) 来忽略未来警告，执行连接操作
    with tm.assert_produces_warning(None):
        result = a.join([b, c], how="outer", sort=sort_kw)
    # 使用断言检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


class TestDataFrameJoin:
    def test_join(self, multiindex_dataframe_random_data):
        # 获取 multiindex_dataframe_random_data
        frame = multiindex_dataframe_random_data

        # 从 frame 中选择前5行的'A'列作为 DataFrame a
        a = frame.loc[frame.index[:5], ["A"]]
        # 从 frame 中选择从第3行开始的'B'和'C'列作为 DataFrame b
        b = frame.loc[frame.index[2:], ["B", "C"]]

        # 使用外连接将 a 和 b 连接，并按 frame 的索引重新排序
        joined = a.join(b, how="outer").reindex(frame.index)

        # 复制 frame 的值作为预期结果 expected
        expected = frame.copy().values.copy()
        expected[np.isnan(joined.values)] = np.nan
        expected = DataFrame(expected, index=frame.index, columns=frame.columns)

        # 使用断言检查 joined 和 expected 是否相等
        assert not np.isnan(joined.values).all()
        tm.assert_frame_equal(joined, expected)

    def test_join_segfault(self):
        # GH#1532
        # 创建 DataFrame df1，包含列'a', 'b', 'x'，并将索引设置为 ['a', 'b']
        df1 = DataFrame({"a": [1, 1], "b": [1, 2], "x": [1, 2]})
        # 创建 DataFrame df2，包含列'a', 'b', 'y'，并将索引设置为 ['a', 'b']
        df2 = DataFrame({"a": [2, 2], "b": [1, 2], "y": [1, 2]})
        # 尝试不同的连接方式来避免段错误
        for how in ["left", "right", "outer"]:
            df1.join(df2, how=how)

    def test_join_str_datetime(self):
        # 创建字符串日期列表 str_dates 和日期时间列表 dt_dates
        str_dates = ["20120209", "20120222"]
        dt_dates = [datetime(2012, 2, 9), datetime(2012, 2, 22)]

        # 创建 DataFrame A，列名为'aa'，索引为[0, 1]
        A = DataFrame(str_dates, index=range(2), columns=["aa"])
        # 创建 DataFrame C，值为[[1, 2], [3, 4]]，索引为 str_dates，列名为 dt_dates
        C = DataFrame([[1, 2], [3, 4]], index=str_dates, columns=dt_dates)

        # 使用字符串列'aa'连接 DataFrame A 和 DataFrame C
        tst = A.join(C, on="aa")

        # 使用断言检查 tst 的列数是否为 3
        assert len(tst.columns) == 3
    # 定义一个测试函数，用于测试多重索引数据框的合并操作
    def test_join_multiindex_leftright(self):
        # GH 10741: GitHub issue reference
        # 创建第一个数据框 df1，包含带有两级索引的数据，及其对应的数值列
        df1 = DataFrame(
            [
                ["a", "x", 0.471780],
                ["a", "y", 0.774908],
                ["a", "z", 0.563634],
                ["b", "x", -0.353756],
                ["b", "y", 0.368062],
                ["b", "z", -1.721840],
                ["c", "x", 1],
                ["c", "y", 2],
                ["c", "z", 3],
            ],
            columns=["first", "second", "value1"],  # 指定列名
        ).set_index(["first", "second"])  # 设置索引为 'first' 和 'second'

        # 创建第二个数据框 df2，包含带有一级索引的数据
        df2 = DataFrame([["a", 10], ["b", 20]], columns=["first", "value2"]).set_index(
            ["first"]  # 设置索引为 'first'
        )

        # 创建期望的结果数据框 exp，将 df1 和 df2 按照索引合并，保留左侧数据框的索引结构
        exp = DataFrame(
            [
                [0.471780, 10],
                [0.774908, 10],
                [0.563634, 10],
                [-0.353756, 20],
                [0.368062, 20],
                [-1.721840, 20],
                [1.000000, np.nan],
                [2.000000, np.nan],
                [3.000000, np.nan],
            ],
            index=df1.index,  # 使用 df1 的索引
            columns=["value1", "value2"],  # 指定列名
        )

        # 断言 df1 左连接 df2 的结果等于期望的 exp
        tm.assert_frame_equal(df1.join(df2, how="left"), exp)

        # 断言 df2 右连接 df1 的结果等于期望的 exp，但是列的顺序相反
        tm.assert_frame_equal(df2.join(df1, how="right"), exp[["value2", "value1"]])

        # 创建期望的索引 exp_idx，由给定的两个列表的笛卡尔积形成
        exp_idx = MultiIndex.from_product(
            [["a", "b"], ["x", "y", "z"]], names=["first", "second"]
        )
        
        # 创建期望的结果数据框 exp，将 df1 和 df2 按照索引合并，保留右侧数据框的索引结构
        exp = DataFrame(
            [
                [0.471780, 10],
                [0.774908, 10],
                [0.563634, 10],
                [-0.353756, 20],
                [0.368062, 20],
                [-1.721840, 20],
            ],
            index=exp_idx,  # 使用 exp_idx 作为索引
            columns=["value1", "value2"],  # 指定列名
        )

        # 断言 df1 右连接 df2 的结果等于期望的 exp
        tm.assert_frame_equal(df1.join(df2, how="right"), exp)

        # 断言 df2 左连接 df1 的结果等于期望的 exp，但是列的顺序相反
        tm.assert_frame_equal(df2.join(df1, how="left"), exp[["value2", "value1"]])

    # 定义一个测试函数，用于测试多重索引数据框与日期的合并操作
    def test_join_multiindex_dates(self):
        # GH 33692: GitHub issue reference
        # 创建一个日期对象
        date = pd.Timestamp(2000, 1, 1).date()

        # 创建带有多级索引的数据框 df1，包含一列数据 'col1'
        df1_index = MultiIndex.from_tuples([(0, date)], names=["index_0", "date"])
        df1 = DataFrame({"col1": [0]}, index=df1_index)

        # 创建带有多级索引的数据框 df2，包含一列数据 'col2'
        df2_index = MultiIndex.from_tuples([(0, date)], names=["index_0", "date"])
        df2 = DataFrame({"col2": [0]}, index=df2_index)

        # 创建带有多级索引的数据框 df3，包含一列数据 'col3'
        df3_index = MultiIndex.from_tuples([(0, date)], names=["index_0", "date"])
        df3 = DataFrame({"col3": [0]}, index=df3_index)

        # 对 df1 进行左连接 df2 和 df3，将结果保存在 result 中
        result = df1.join([df2, df3])

        # 创建期望的索引 expected_index
        expected_index = MultiIndex.from_tuples([(0, date)], names=["index_0", "date"])

        # 创建期望的结果数据框 expected，包含三列 'col1', 'col2', 'col3'，并使用 expected_index 作为索引
        expected = DataFrame(
            {"col1": [0], "col2": [0], "col3": [0]}, index=expected_index
        )

        # 断言 result 与期望的 expected 相等
        tm.assert_equal(result, expected)
    # 定义测试函数，用于测试不同层级的数据框合并操作是否会引发异常
    def test_merge_join_different_levels_raises():
        # GH#9455
        # GH 40993: For raising, enforced in 2.0

        # 创建第一个数据框
        df1 = DataFrame(columns=["a", "b"], data=[[1, 11], [0, 22]])

        # 创建第二个数据框，包含多级索引
        columns = MultiIndex.from_tuples([("a", ""), ("c", "c1")])
        df2 = DataFrame(columns=columns, data=[[1, 33], [0, 44]])

        # 使用 pytest 检查合并操作是否会引发 MergeError 异常，匹配特定错误信息
        with pytest.raises(
            MergeError, match="Not allowed to merge between different levels"
        ):
            pd.merge(df1, df2, on="a")

        # 使用 pytest 检查连接操作是否会引发 MergeError 异常，匹配特定错误信息
        with pytest.raises(
            MergeError, match="Not allowed to merge between different levels"
        ):
            df1.join(df2, on="a")

    # 定义测试函数，用于测试具有时区信息的数据框连接操作
    def test_frame_join_tzaware():
        # 创建时区对象
        tz = zoneinfo.ZoneInfo("US/Central")

        # 创建第一个数据框，设置具有时区信息的时间索引
        test1 = DataFrame(
            np.zeros((6, 3)),
            index=date_range("2012-11-15 00:00:00", periods=6, freq="100ms", tz=tz),
        )

        # 创建第二个数据框，设置具有时区信息的时间索引和列标签
        test2 = DataFrame(
            np.zeros((3, 3)),
            index=date_range("2012-11-15 00:00:00", periods=3, freq="250ms", tz=tz),
            columns=range(3, 6),
        )

        # 执行外连接操作
        result = test1.join(test2, how="outer")

        # 期望的索引为两个数据框索引的并集
        expected = test1.index.union(test2.index)

        # 使用 assert 检查实际结果的索引是否等于预期的索引
        tm.assert_index_equal(result.index, expected)

        # 使用 assert 检查结果的索引是否具有正确的时区信息
        assert result.index.tz.key == "US/Central"
```