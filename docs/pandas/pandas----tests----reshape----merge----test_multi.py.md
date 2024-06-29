# `D:\src\scipysrc\pandas\pandas\tests\reshape\merge\test_multi.py`

```
# 导入必要的库
import numpy as np  # 导入numpy库，用于数值计算
import pytest  # 导入pytest库，用于单元测试

import pandas.util._test_decorators as td  # 导入测试装饰器

import pandas as pd  # 导入pandas库
from pandas import (  # 从pandas库中导入多个类和函数
    DataFrame,
    Index,
    MultiIndex,
    RangeIndex,
    Series,
    Timestamp,
    option_context,
)
import pandas._testing as tm  # 导入测试相关的函数和类
from pandas.core.reshape.concat import concat  # 导入concat函数
from pandas.core.reshape.merge import merge  # 导入merge函数


@pytest.fixture
def left():
    """left dataframe (not multi-indexed) for multi-index join tests"""
    # 创建一个左侧数据框，用于多索引连接测试，不是多重索引
    # 包含一个示例数据集和NaN值
    key1 = ["bar", "bar", "bar", "foo", "foo", "baz", "baz", "qux", "qux", "snap"]
    key2 = ["two", "one", "three", "one", "two", "one", "two", "two", "three", "one"]

    data = np.random.default_rng(2).standard_normal(len(key1))
    return DataFrame({"key1": key1, "key2": key2, "data": data})


@pytest.fixture
def right(multiindex_dataframe_random_data):
    """right dataframe (multi-indexed) for multi-index join tests"""
    # 创建一个右侧数据框，用于多索引连接测试，是多重索引
    df = multiindex_dataframe_random_data
    df.index.names = ["key1", "key2"]

    df.columns = ["j_one", "j_two", "j_three"]
    return df


@pytest.fixture
def left_multi():
    # 创建一个左侧多重索引数据框
    return DataFrame(
        {
            "Origin": ["A", "A", "B", "B", "C"],
            "Destination": ["A", "B", "A", "C", "A"],
            "Period": ["AM", "AM", "IP", "AM", "OP"],
            "TripPurp": ["hbw", "nhb", "hbo", "nhb", "hbw"],
            "Trips": [1987, 3647, 2470, 4296, 4444],
        },
        columns=["Origin", "Destination", "Period", "TripPurp", "Trips"],
    ).set_index(["Origin", "Destination", "Period", "TripPurp"])


@pytest.fixture
def right_multi():
    # 创建一个右侧多重索引数据框
    return DataFrame(
        {
            "Origin": ["A", "A", "B", "B", "C", "C", "E"],
            "Destination": ["A", "B", "A", "B", "A", "B", "F"],
            "Period": ["AM", "AM", "IP", "AM", "OP", "IP", "AM"],
            "LinkType": ["a", "b", "c", "b", "a", "b", "a"],
            "Distance": [100, 80, 90, 80, 75, 35, 55],
        },
        columns=["Origin", "Destination", "Period", "LinkType", "Distance"],
    ).set_index(["Origin", "Destination", "Period", "LinkType"])


@pytest.fixture
def on_cols_multi():
    # 返回多重索引连接的键列表
    return ["Origin", "Destination", "Period"]


class TestMergeMulti:
    def test_merge_on_multikey(self, left, right, join_type):
        # 测试多键连接功能
        on_cols = ["key1", "key2"]
        # 使用左侧数据框和右侧数据框的多键进行连接，指定连接类型
        result = left.join(right, on=on_cols, how=join_type).reset_index(drop=True)

        # 期望的连接结果，使用merge函数来实现
        expected = merge(left, right.reset_index(), on=on_cols, how=join_type)

        # 断言连接结果与期望结果一致
        tm.assert_frame_equal(result, expected)

        # 使用排序参数进行连接，并断言结果是否与期望一致
        result = left.join(right, on=on_cols, how=join_type, sort=True).reset_index(
            drop=True
        )

        expected = merge(
            left, right.reset_index(), on=on_cols, how=join_type, sort=True
        )

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "infer_string", [False, pytest.param(True, marks=td.skip_if_no("pyarrow"))]
    )
    def test_left_join_multi_index(self, sort, infer_string):
        # 定义测试函数，测试左连接多索引操作
        with option_context("future.infer_string", infer_string):
            # 设置上下文选项，用于推断字符串类型的行为
            icols = ["1st", "2nd", "3rd"]

            def bind_cols(df):
                # 定义内部函数，将列映射为整数，计算并返回复合计算结果
                iord = lambda a: 0 if a != a else ord(a)
                f = lambda ts: ts.map(iord) - ord("a")
                return f(df["1st"]) + f(df["3rd"]) * 1e2 + df["2nd"].fillna(0) * 10

            def run_asserts(left, right, sort):
                # 执行左连接，并运行多个断言以验证结果
                res = left.join(right, on=icols, how="left", sort=sort)

                assert len(left) < len(res) + 1
                assert not res["4th"].isna().any()
                assert not res["5th"].isna().any()

                tm.assert_series_equal(res["4th"], -res["5th"], check_names=False)
                result = bind_cols(res.iloc[:, :-2])
                tm.assert_series_equal(res["4th"], result, check_names=False)
                assert result.name is None

                if sort:
                    tm.assert_frame_equal(res, res.sort_values(icols, kind="mergesort"))

                out = merge(left, right.reset_index(), on=icols, sort=sort, how="left")

                res.index = RangeIndex(len(res))
                tm.assert_frame_equal(out, res)

            lc = list(map(chr, np.arange(ord("a"), ord("z") + 1)))
            left = DataFrame(
                np.random.default_rng(2).choice(lc, (50, 2)), columns=["1st", "3rd"]
            )
            # Explicit cast to float to avoid implicit cast when setting nan
            left.insert(
                1,
                "2nd",
                np.random.default_rng(2).integers(0, 10, len(left)).astype("float"),
            )
            right = left.sample(frac=1, random_state=np.random.default_rng(2))

            left["4th"] = bind_cols(left)
            right["5th"] = -bind_cols(right)
            right.set_index(icols, inplace=True)

            run_asserts(left, right, sort)

            # inject some nulls
            left.loc[1::4, "1st"] = np.nan
            left.loc[2::5, "2nd"] = np.nan
            left.loc[3::6, "3rd"] = np.nan
            left["4th"] = bind_cols(left)

            i = np.random.default_rng(2).permutation(len(left))
            right = left.iloc[i, :-1]
            right["5th"] = -bind_cols(right)
            right.set_index(icols, inplace=True)

            run_asserts(left, right, sort)

    def test_merge_right_vs_left(self, left, right, sort):
        # 比较左连接和右连接的结果，使用多个键
        on_cols = ["key1", "key2"]
        merged_left_right = left.merge(
            right, left_on=on_cols, right_index=True, how="left", sort=sort
        )

        merge_right_left = right.merge(
            left, right_on=on_cols, left_index=True, how="right", sort=sort
        )

        # 重新排序列，使右连接的列顺序与左连接相同
        merge_right_left = merge_right_left[merged_left_right.columns]

        tm.assert_frame_equal(merged_left_right, merge_right_left)
    def test_merge_multiple_cols_with_mixed_cols_index(self):
        # 测试用例名称: 混合列索引的多列合并
        # GH29522 是 GitHub 上的 issue 或 PR 编号
        s = Series(
            range(6),
            MultiIndex.from_product([["A", "B"], [1, 2, 3]], names=["lev1", "lev2"]),
            name="Amount",
        )
        # 创建包含多级索引的 Series 对象 s，名称为 "Amount"
        
        df = DataFrame({"lev1": list("AAABBB"), "lev2": [1, 2, 3, 1, 2, 3], "col": 0})
        # 创建包含 lev1、lev2 和 col 列的 DataFrame 对象 df
        
        result = merge(df, s.reset_index(), on=["lev1", "lev2"])
        # 使用 lev1 和 lev2 列将 df 和 s 重置索引后的结果进行合并，生成结果 DataFrame result
        
        expected = DataFrame(
            {
                "lev1": list("AAABBB"),
                "lev2": [1, 2, 3, 1, 2, 3],
                "col": [0] * 6,
                "Amount": range(6),
            }
        )
        # 创建期望的结果 DataFrame expected，包含 lev1、lev2、col 和 Amount 列
        
        tm.assert_frame_equal(result, expected)
        # 使用测试工具 tm.assert_frame_equal 检查 result 和 expected 是否相等，并返回结果
    
    def test_compress_group_combinations(self):
        # 测试用例名称: 压缩组合情况
        # ~ 40000000 possible unique groups 表示可能的唯一组合数量约为 4000 万
        key1 = [str(i) for i in range(10000)]
        # 创建包含 10000 个字符串形式的数字的列表 key1
        
        key1 = np.tile(key1, 2)
        # 使用 np.tile 将 key1 扩展成长度为原来的两倍
        
        key2 = key1[::-1]
        # 创建 key1 的逆序列表 key2
        
        df = DataFrame(
            {
                "key1": key1,
                "key2": key2,
                "value1": np.random.default_rng(2).standard_normal(20000),
            }
        )
        # 创建包含 key1、key2 和 value1 列的 DataFrame 对象 df
        
        df2 = DataFrame(
            {
                "key1": key1[::2],
                "key2": key2[::2],
                "value2": np.random.default_rng(2).standard_normal(10000),
            }
        )
        # 创建包含 key1[::2]、key2[::2] 和 value2 列的 DataFrame 对象 df2
        
        # just to hit the label compression code path
        # 只是为了触发标签压缩的代码路径
        merge(df, df2, how="outer")
        # 使用外连接方式合并 df 和 df2，并返回结果
    # 定义一个测试方法，测试左连接并保持索引顺序不变
    def test_left_join_index_preserve_order(self):
        # 左连接的关键列
        on_cols = ["k1", "k2"]
        
        # 创建左侧数据帧
        left = DataFrame(
            {
                "k1": [0, 1, 2] * 8,
                "k2": ["foo", "bar"] * 12,
                "v": np.array(np.arange(24), dtype=np.int64),
            }
        )

        # 创建右侧数据帧，并使用指定的多重索引
        index = MultiIndex.from_tuples([(2, "bar"), (1, "foo")])
        right = DataFrame({"v2": [5, 7]}, index=index)

        # 执行左连接操作
        result = left.join(right, on=on_cols)

        # 复制左侧数据帧作为预期结果
        expected = left.copy()
        # 在预期结果中添加一个新列'v2'，并初始化为NaN
        expected["v2"] = np.nan
        # 根据条件更新预期结果中'v2'列的值
        expected.loc[(expected.k1 == 2) & (expected.k2 == "bar"), "v2"] = 5
        expected.loc[(expected.k1 == 1) & (expected.k2 == "foo"), "v2"] = 7

        # 使用测试工具库比较实际结果和预期结果的数据帧内容
        tm.assert_frame_equal(result, expected)

        # 对实际结果按照指定的列排序，使用稳定排序算法"mergesort"
        result.sort_values(on_cols, kind="mergesort", inplace=True)
        # 重新执行左连接操作，并设置排序标志为True
        expected = left.join(right, on=on_cols, sort=True)

        # 使用测试工具库比较经过排序的实际结果和预期结果的数据帧内容
        tm.assert_frame_equal(result, expected)

        # 测试包含多个数据类型块的连接操作
        # 创建包含多个数据类型的左侧数据帧
        left = DataFrame(
            {
                "k1": [0, 1, 2] * 8,
                "k2": ["foo", "bar"] * 12,
                "k3": np.array([0, 1, 2] * 8, dtype=np.float32),
                "v": np.array(np.arange(24), dtype=np.int32),
            }
        )

        # 再次创建右侧数据帧，并使用相同的多重索引
        index = MultiIndex.from_tuples([(2, "bar"), (1, "foo")])
        right = DataFrame({"v2": [5, 7]}, index=index)

        # 执行左连接操作
        result = left.join(right, on=on_cols)

        # 复制左侧数据帧作为预期结果
        expected = left.copy()
        # 在预期结果中添加一个新列'v2'，并初始化为NaN
        expected["v2"] = np.nan
        # 根据条件更新预期结果中'v2'列的值
        expected.loc[(expected.k1 == 2) & (expected.k2 == "bar"), "v2"] = 5
        expected.loc[(expected.k1 == 1) & (expected.k2 == "foo"), "v2"] = 7

        # 使用测试工具库比较实际结果和预期结果的数据帧内容
        tm.assert_frame_equal(result, expected)

        # 对实际结果按照指定的列排序，使用稳定排序算法"mergesort"
        result = result.sort_values(on_cols, kind="mergesort")
        # 重新执行左连接操作，并设置排序标志为True
        expected = left.join(right, on=on_cols, sort=True)

        # 使用测试工具库比较经过排序的实际结果和预期结果的数据帧内容
        tm.assert_frame_equal(result, expected)
    # 定义一个测试函数，测试左连接多列匹配的情况，使用多级索引
    def test_left_join_index_multi_match_multiindex(self):
        # 创建左侧数据帧，包含多个行和列，同时设置了索引
        left = DataFrame(
            [
                ["X", "Y", "C", "a"],
                ["W", "Y", "C", "e"],
                ["V", "Q", "A", "h"],
                ["V", "R", "D", "i"],
                ["X", "Y", "D", "b"],
                ["X", "Y", "A", "c"],
                ["W", "Q", "B", "f"],
                ["W", "R", "C", "g"],
                ["V", "Y", "C", "j"],
                ["X", "Y", "B", "d"],
            ],
            columns=["cola", "colb", "colc", "tag"],  # 设置列名
            index=[3, 2, 0, 1, 7, 6, 4, 5, 9, 8],  # 设置索引
        )

        # 创建右侧数据帧，包含多个行和列，同时设置了多级索引
        right = DataFrame(
            [
                ["W", "R", "C", 0],
                ["W", "Q", "B", 3],
                ["W", "Q", "B", 8],
                ["X", "Y", "A", 1],
                ["X", "Y", "A", 4],
                ["X", "Y", "B", 5],
                ["X", "Y", "C", 6],
                ["X", "Y", "C", 9],
                ["X", "Q", "C", -6],
                ["X", "R", "C", -9],
                ["V", "Y", "C", 7],
                ["V", "R", "D", 2],
                ["V", "R", "D", -1],
                ["V", "Q", "A", -3],
            ],
            columns=["col1", "col2", "col3", "val"],  # 设置列名
        ).set_index(["col1", "col2", "col3"])  # 将列作为多级索引

        # 执行左连接操作，基于 ["cola", "colb", "colc"] 列进行左连接，连接方式为左连接
        result = left.join(right, on=["cola", "colb", "colc"], how="left")

        # 预期的结果数据帧，包含左连接后的期望值
        expected = DataFrame(
            [
                ["X", "Y", "C", "a", 6],
                ["X", "Y", "C", "a", 9],
                ["W", "Y", "C", "e", np.nan],
                ["V", "Q", "A", "h", -3],
                ["V", "R", "D", "i", 2],
                ["V", "R", "D", "i", -1],
                ["X", "Y", "D", "b", np.nan],
                ["X", "Y", "A", "c", 1],
                ["X", "Y", "A", "c", 4],
                ["W", "Q", "B", "f", 3],
                ["W", "Q", "B", "f", 8],
                ["W", "R", "C", "g", 0],
                ["V", "Y", "C", "j", 7],
                ["X", "Y", "B", "d", 5],
            ],
            columns=["cola", "colb", "colc", "tag", "val"],  # 设置列名
            index=[3, 3, 2, 0, 1, 1, 7, 6, 6, 4, 4, 5, 9, 8],  # 设置索引
        )

        # 使用测试模块中的断言函数，比较结果和预期数据帧是否相等
        tm.assert_frame_equal(result, expected)

        # 再次执行左连接操作，但这次增加了 sort=True 参数，对结果进行排序
        result = left.join(right, on=["cola", "colb", "colc"], how="left", sort=True)

        # 对期望的结果数据帧进行排序，根据 ["cola", "colb", "colc"] 列，采用 mergesort 算法
        expected = expected.sort_values(["cola", "colb", "colc"], kind="mergesort")

        # 使用测试模块中的断言函数，比较排序后的结果和预期数据帧是否相等
        tm.assert_frame_equal(result, expected)
    # 定义测试函数，测试左连接索引多匹配情况
    def test_left_join_index_multi_match(self):
        # 创建左侧数据帧 DataFrame，包括标签和值两列，以及自定义索引
        left = DataFrame(
            [["c", 0], ["b", 1], ["a", 2], ["b", 3]],
            columns=["tag", "val"],
            index=[2, 0, 1, 3],
        )

        # 创建右侧数据帧 DataFrame，包括标签和字符两列，并将标签列设为索引
        right = DataFrame(
            [
                ["a", "v"],
                ["c", "w"],
                ["c", "x"],
                ["d", "y"],
                ["a", "z"],
                ["c", "r"],
                ["e", "q"],
                ["c", "s"],
            ],
            columns=["tag", "char"],
        ).set_index("tag")

        # 使用左连接将左右数据帧合并，按照左侧数据帧的标签列进行匹配，保留左侧所有行
        result = left.join(right, on="tag", how="left")

        # 创建预期的结果数据帧，包括左侧标签、值和右侧字符列，并设定新的索引
        expected = DataFrame(
            [
                ["c", 0, "w"],
                ["c", 0, "x"],
                ["c", 0, "r"],
                ["c", 0, "s"],
                ["b", 1, np.nan],
                ["a", 2, "v"],
                ["a", 2, "z"],
                ["b", 3, np.nan],
            ],
            columns=["tag", "val", "char"],
            index=[2, 2, 2, 2, 0, 1, 1, 3],
        )

        # 断言结果数据帧与预期数据帧相等
        tm.assert_frame_equal(result, expected)

        # 对结果数据帧进行排序，并与另一个预期数据帧进行比较
        result = left.join(right, on="tag", how="left", sort=True)
        expected2 = expected.sort_values("tag", kind="mergesort")
        
        # 断言排序后的结果数据帧与第二个预期数据帧相等
        tm.assert_frame_equal(result, expected2)

        # GH7331 - 在左连接中保持左侧数据帧的顺序
        # 使用 merge 函数进行左连接，将右侧数据帧重新索引后与左侧数据帧进行比较
        result = merge(left, right.reset_index(), how="left", on="tag")
        expected.index = RangeIndex(len(expected))
        # 断言合并后的结果数据帧与第三个预期数据帧相等
        tm.assert_frame_equal(result, expected)

    # 测试左连接时的缺失值 buglet
    def test_left_merge_na_buglet(self):
        # 创建左侧数据帧 DataFrame，包括标识符、多列随机值和虚拟列
        left = DataFrame(
            {
                "id": list("abcde"),
                "v1": np.random.default_rng(2).standard_normal(5),
                "v2": np.random.default_rng(2).standard_normal(5),
                "dummy": list("abcde"),
                "v3": np.random.default_rng(2).standard_normal(5),
            },
            columns=["id", "v1", "v2", "dummy", "v3"],
        )

        # 创建右侧数据帧 DataFrame，包括标识符和单列值
        right = DataFrame(
            {
                "id": ["a", "b", np.nan, np.nan, np.nan],
                "sv3": [1.234, 5.678, np.nan, np.nan, np.nan],
            }
        )

        # 使用左连接将左右数据帧合并，按照标识符列进行匹配，保留左侧所有行
        result = merge(left, right, on="id", how="left")

        # 从右侧数据帧中删除标识符列，然后与左侧数据帧进行连接
        rdf = right.drop(["id"], axis=1)
        expected = left.join(rdf)

        # 断言结果数据帧与预期数据帧相等
        tm.assert_frame_equal(result, expected)
    def test_merge_na_keys(self):
        # 准备测试数据，包括年份、面板和数据值的列表
        data = [
            [1950, "A", 1.5],
            [1950, "B", 1.5],
            [1955, "B", 1.5],
            [1960, "B", np.nan],
            [1970, "B", 4.0],
            [1950, "C", 4.0],
            [1960, "C", np.nan],
            [1965, "C", 3.0],
            [1970, "C", 4.0],
        ]

        # 创建 DataFrame，列名为"year", "panel", "data"
        frame = DataFrame(data, columns=["year", "panel", "data"])

        # 准备另一个测试数据，包括年份、面板和数据值的列表
        other_data = [
            [1960, "A", np.nan],
            [1970, "A", np.nan],
            [1955, "A", np.nan],
            [1965, "A", np.nan],
            [1965, "B", np.nan],
            [1955, "C", np.nan],
        ]
        # 创建另一个 DataFrame，列名为"year", "panel", "data"
        other = DataFrame(other_data, columns=["year", "panel", "data"])

        # 执行 merge 操作，使用 outer 连接方式
        result = frame.merge(other, how="outer")

        # 创建期望结果的 DataFrame，并对 NaN 值进行填充和替换
        expected = frame.fillna(-999).merge(other.fillna(-999), how="outer")
        expected = expected.replace(-999, np.nan)

        # 使用 assert_frame_equal 函数比较实际结果和期望结果
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("klass", [None, np.asarray, Series, Index])
    def test_merge_datetime_index(self, klass):
        # 测试用例标记为 gh-19038
        # 创建包含整数数据的 DataFrame，索引为日期字符串
        df = DataFrame(
            [1, 2, 3], ["2016-01-01", "2017-01-01", "2018-01-01"], columns=["a"]
        )
        # 将索引转换为日期时间格式
        df.index = pd.to_datetime(df.index)
        # 根据索引年份创建一个数组
        on_vector = df.index.year

        # 如果 klass 不为 None，则将 on_vector 转换为 klass 类型
        if klass is not None:
            on_vector = klass(on_vector)

        # 创建期望结果的 DataFrame，包含列"a"和"key_1"
        exp_years = np.array([2016, 2017, 2018], dtype=np.int32)
        expected = DataFrame({"a": [1, 2, 3], "key_1": exp_years})

        # 执行 merge 操作，使用 inner 连接方式
        result = df.merge(df, on=["a", on_vector], how="inner")
        # 比较实际结果和期望结果
        tm.assert_frame_equal(result, expected)

        # 创建期望结果的 DataFrame，包含列"key_0", "a_x"和"a_y"
        expected = DataFrame({"key_0": exp_years, "a_x": [1, 2, 3], "a_y": [1, 2, 3]})

        # 执行 merge 操作，使用 inner 连接方式，按照年份索引进行连接
        result = df.merge(df, on=[df.index.year], how="inner")
        # 比较实际结果和期望结果
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("merge_type", ["left", "right"])
    # 定义测试方法，用于测试在指定合并类型下处理空数据框的情况
    def test_merge_datetime_multi_index_empty_df(self, merge_type):
        # 标识这段代码是为了解决 GitHub issue #36895

        # 创建左侧数据框，包含"data"列和多级索引
        left = DataFrame(
            data={
                "data": [1.5, 1.5],
            },
            index=MultiIndex.from_tuples(
                [[Timestamp("1950-01-01"), "A"], [Timestamp("1950-01-02"), "B"]],
                names=["date", "panel"],
            ),
        )

        # 创建右侧空数据框，仅包含多级索引，列名为"state"
        right = DataFrame(
            index=MultiIndex.from_tuples([], names=["date", "panel"]), columns=["state"]
        )

        # 创建预期的多级索引
        expected_index = MultiIndex.from_tuples(
            [[Timestamp("1950-01-01"), "A"], [Timestamp("1950-01-02"), "B"]],
            names=["date", "panel"],
        )

        # 根据合并类型选择不同的预期结果数据框
        if merge_type == "left":
            # 如果是左连接，预期结果包含"data"和"state"两列，其中"state"列的值为NaN
            expected = DataFrame(
                data={
                    "data": [1.5, 1.5],
                    "state": np.array([np.nan, np.nan], dtype=object),
                },
                index=expected_index,
            )
            # 执行左连接，并将结果与预期结果比较
            results_merge = left.merge(right, how="left", on=["date", "panel"])
            # 执行左连接（join操作），并将结果与预期结果比较
            results_join = left.join(right, how="left")
        else:
            # 如果是右连接或其他类型，预期结果包含"state"和"data"两列，其中"data"列的值不变
            expected = DataFrame(
                data={
                    "state": np.array([np.nan, np.nan], dtype=object),
                    "data": [1.5, 1.5],
                },
                index=expected_index,
            )
            # 执行右连接，将右侧数据框与左侧数据框合并，并将结果与预期结果比较
            results_merge = right.merge(left, how="right", on=["date", "panel"])
            # 执行右连接（join操作），将右侧数据框与左侧数据框合并，并将结果与预期结果比较
            results_join = right.join(left, how="right")

        # 使用测试框架中的方法断言合并后的结果与预期结果相等
        tm.assert_frame_equal(results_merge, expected)
        # 使用测试框架中的方法断言join后的结果与预期结果相等
        tm.assert_frame_equal(results_join, expected)
    def expected(self):
        # 创建一个预期的 DataFrame，包含以下列：male, wealth, name, share, household_id, asset_id
        expected = (
            DataFrame(
                {
                    "male": [0, 1, 1, 0, 0, 0],
                    "wealth": [
                        196087.3,
                        316478.7,
                        316478.7,
                        294750.0,
                        294750.0,
                        294750.0,
                    ],
                    "name": [
                        "ABN Amro",
                        "Robeco",
                        "Royal Dutch Shell",
                        "Royal Dutch Shell",
                        "AAB Eastern Europe Equity Fund",
                        "Postbank BioTech Fonds",
                    ],
                    "share": [1.00, 0.40, 0.60, 0.15, 0.60, 0.25],
                    "household_id": [1, 2, 2, 3, 3, 3],
                    "asset_id": [
                        "nl0000301109",
                        "nl0000289783",
                        "gb00b03mlx29",
                        "gb00b03mlx29",
                        "lu0197800237",
                        "nl0000289965",
                    ],
                }
            )
            # 将 "household_id" 和 "asset_id" 设为索引，并按指定列顺序重新索引
            .set_index(["household_id", "asset_id"])
            .reindex(columns=["male", "wealth", "name", "share"])
        )
        return expected

    def test_join_multi_levels(self, portfolio, household, expected):
        portfolio = portfolio.copy()
        household = household.copy()

        # GH 3662
        # 对多级索引进行合并
        result = household.join(portfolio, how="inner")
        tm.assert_frame_equal(result, expected)

    def test_join_multi_levels_merge_equivalence(self, portfolio, household, expected):
        portfolio = portfolio.copy()
        household = household.copy()

        # 判断等价性
        result = merge(
            household.reset_index(),
            portfolio.reset_index(),
            on=["household_id"],
            how="inner",
        ).set_index(["household_id", "asset_id"])
        tm.assert_frame_equal(result, expected)

    def test_join_multi_levels_outer(self, portfolio, household, expected):
        portfolio = portfolio.copy()
        household = household.copy()

        # 外连接
        result = household.join(portfolio, how="outer")
        
        # 创建预期的 DataFrame，包含额外的一行数据，并按指定列重新索引
        expected = concat(
            [
                expected,
                (
                    DataFrame(
                        {"share": [1.00]},
                        index=MultiIndex.from_tuples(
                            [(4, np.nan)], names=["household_id", "asset_id"]
                        ),
                    )
                ),
            ],
            axis=0,
            sort=True,
        ).reindex(columns=expected.columns)
        
        # 断言两个 DataFrame 是否相等，忽略索引类型的检查
        tm.assert_frame_equal(result, expected, check_index_type=False)
    # 定义一个测试方法，用于测试无效的多层级连接情况，接受两个参数：portfolio 和 household
    def test_join_multi_levels_invalid(self, portfolio, household):
        # 复制 portfolio 和 household，以便在测试中使用它们的副本，而不改变原始数据
        portfolio = portfolio.copy()
        household = household.copy()

        # 设置 household 的索引名称为 "foo"，这是一个无效的情况
        household.index.name = "foo"

        # 使用 pytest 的上下文管理器检查是否会引发 ValueError 异常，
        # 并且异常消息应包含 "cannot join with no overlapping index names"
        with pytest.raises(
            ValueError, match="cannot join with no overlapping index names"
        ):
            # 尝试将 household 和 portfolio 进行内连接（inner join）
            household.join(portfolio, how="inner")

        # 复制 portfolio 以便进行第二个测试
        portfolio2 = portfolio.copy()
        # 设置 portfolio2 的索引名称为 ["household_id", "foo"]
        portfolio2.index.set_names(["household_id", "foo"])

        # 使用 pytest 的上下文管理器检查是否会引发 ValueError 异常，
        # 并且异常消息应包含 "columns overlap but no suffix specified"
        with pytest.raises(ValueError, match="columns overlap but no suffix specified"):
            # 尝试将 portfolio2 和 portfolio 进行内连接（inner join）
            portfolio2.join(portfolio, how="inner")
    # 定义一个测试类 TestJoinMultiMulti，用于测试多重索引数据帧的合并操作
    class TestJoinMultiMulti:
        
        # 定义测试方法 test_join_multi_multi，测试多重索引数据帧的多重合并操作
        def test_join_multi_multi(self, left_multi, right_multi, join_type, on_cols_multi):
            # 获取左侧数据帧的索引名称列表
            left_names = left_multi.index.names
            # 获取右侧数据帧的索引名称列表
            right_names = right_multi.index.names
            # 根据连接类型确定合并后的索引顺序
            if join_type == "right":
                level_order = right_names + left_names.difference(right_names)
            else:
                level_order = left_names + right_names.difference(left_names)
            
            # 构建期望的合并结果
            expected = (
                merge(
                    left_multi.reset_index(),
                    right_multi.reset_index(),
                    how=join_type,
                    on=on_cols_multi,
                )
                .set_index(level_order)  # 设置合并后的多重索引
                .sort_index()  # 按索引排序
            )
            
            # 执行实际的合并操作，并按索引排序
            result = left_multi.join(right_multi, how=join_type).sort_index()
            # 使用测试框架断言实际结果与期望结果相等
            tm.assert_frame_equal(result, expected)
        
        # 定义测试方法 test_join_multi_empty_frames，测试在空数据帧上的多重索引合并操作
        def test_join_multi_empty_frames(
            self, left_multi, right_multi, join_type, on_cols_multi
        ):
            # 清空左侧数据帧的列
            left_multi = left_multi.drop(columns=left_multi.columns)
            # 清空右侧数据帧的列
            right_multi = right_multi.drop(columns=right_multi.columns)
            
            # 获取左侧数据帧的索引名称列表
            left_names = left_multi.index.names
            # 获取右侧数据帧的索引名称列表
            right_names = right_multi.index.names
            # 根据连接类型确定合并后的索引顺序
            if join_type == "right":
                level_order = right_names + left_names.difference(right_names)
            else:
                level_order = left_names + right_names.difference(left_names)
            
            # 构建期望的合并结果
            expected = (
                merge(
                    left_multi.reset_index(),
                    right_multi.reset_index(),
                    how=join_type,
                    on=on_cols_multi,
                )
                .set_index(level_order)  # 设置合并后的多重索引
                .sort_index()  # 按索引排序
            )
            
            # 执行实际的合并操作，并按索引排序
            result = left_multi.join(right_multi, how=join_type).sort_index()
            # 使用测试框架断言实际结果与期望结果相等
            tm.assert_frame_equal(result, expected)
        
        # 使用 pytest.mark.parametrize 装饰器标记参数化测试方法 test_merge_datetime_index
        @pytest.mark.parametrize("box", [None, np.asarray, Series, Index])
        # 定义测试方法 test_merge_datetime_index，测试日期时间索引的合并操作
        def test_merge_datetime_index(self, box):
            # 创建包含日期时间索引的数据帧
            df = DataFrame(
                [1, 2, 3], ["2016-01-01", "2017-01-01", "2018-01-01"], columns=["a"]
            )
            # 将索引转换为 datetime 类型
            df.index = pd.to_datetime(df.index)
            # 根据索引年份生成一个用于合并的向量
            on_vector = df.index.year
            
            # 如果 box 参数不为 None，则将 on_vector 转换为 box 类型
            if box is not None:
                on_vector = box(on_vector)
            
            # 期望的合并结果，包含年份和数据列 'a'
            exp_years = np.array([2016, 2017, 2018], dtype=np.int32)
            expected = DataFrame({"a": [1, 2, 3], "key_1": exp_years})
            
            # 执行实际的合并操作，并使用测试框架断言结果与期望相等
            result = df.merge(df, on=["a", on_vector], how="inner")
            tm.assert_frame_equal(result, expected)
            
            # 另一种期望的合并结果，包含年份和数据列 'a_x'、'a_y'
            expected = DataFrame({"key_0": exp_years, "a_x": [1, 2, 3], "a_y": [1, 2, 3]})
            
            # 执行另一次合并操作，并使用测试框架断言结果与期望相等
            result = df.merge(df, on=[df.index.year], how="inner")
            tm.assert_frame_equal(result, expected)
    def test_single_common_level(self):
        # 创建一个多级索引对象 `index_left`，包含元组列表 [("K0", "X0"), ("K0", "X1"), ("K1", "X2")，并指定索引名称为 ["key", "X"]
        index_left = MultiIndex.from_tuples(
            [("K0", "X0"), ("K0", "X1"), ("K1", "X2")], names=["key", "X"]
        )

        # 使用 `index_left` 作为索引创建一个 DataFrame `left`，包含列 {"A": ["A0", "A1", "A2"], "B": ["B0", "B1", "B2"]}
        left = DataFrame(
            {"A": ["A0", "A1", "A2"], "B": ["B0", "B1", "B2"]}, index=index_left
        )

        # 创建一个多级索引对象 `index_right`，包含元组列表 [("K0", "Y0"), ("K1", "Y1"), ("K2", "Y2"), ("K2", "Y3")]，并指定索引名称为 ["key", "Y"]
        index_right = MultiIndex.from_tuples(
            [("K0", "Y0"), ("K1", "Y1"), ("K2", "Y2"), ("K2", "Y3")], names=["key", "Y"]
        )

        # 使用 `index_right` 作为索引创建一个 DataFrame `right`，包含列 {"C": ["C0", "C1", "C2", "C3"], "D": ["D0", "D1", "D2", "D3"]}
        right = DataFrame(
            {"C": ["C0", "C1", "C2", "C3"], "D": ["D0", "D1", "D2", "D3"]},
            index=index_right,
        )

        # 对 `left` 和 `right` 进行连接操作，生成结果 DataFrame `result`
        result = left.join(right)

        # 使用 `merge` 函数将 `left` 和 `right` 按照 "key" 列进行内连接，重新设置索引为 ["key", "X", "Y"]，生成期望的 DataFrame `expected`
        expected = merge(
            left.reset_index(), right.reset_index(), on=["key"], how="inner"
        ).set_index(["key", "X", "Y"])

        # 使用 `tm.assert_frame_equal` 断言 `result` 和 `expected` 的内容相等
        tm.assert_frame_equal(result, expected)

    def test_join_multi_wrong_order(self):
        # GH 25760
        # GH 28956

        # 创建一个多级索引对象 `midx1`，包含由 [1, 2] 和 [3, 4] 的笛卡尔积生成的元组列表，索引名称为 ["a", "b"]
        midx1 = MultiIndex.from_product([[1, 2], [3, 4]], names=["a", "b"])
        
        # 创建一个多级索引对象 `midx3`，包含元组列表 [(4, 1), (3, 2), (3, 1)]，并指定索引名称为 ["b", "a"]
        midx3 = MultiIndex.from_tuples([(4, 1), (3, 2), (3, 1)], names=["b", "a"])

        # 使用 `midx1` 作为索引创建一个 DataFrame `left`，只包含一列 "x"，数据为 [10, 20, 30, 40]
        left = DataFrame(index=midx1, data={"x": [10, 20, 30, 40]})
        
        # 使用 `midx3` 作为索引创建一个 DataFrame `right`，只包含一列 "y"，数据为 ["foo", "bar", "fing"]
        right = DataFrame(index=midx3, data={"y": ["foo", "bar", "fing"]})

        # 对 `left` 和 `right` 进行连接操作，生成结果 DataFrame `result`
        result = left.join(right)

        # 创建一个期望的 DataFrame `expected`，与 `left` 结构相同，但 "y" 列的数据按照匹配规则填充
        expected = DataFrame(
            index=midx1,
            data={"x": [10, 20, 30, 40], "y": ["fing", "foo", "bar", np.nan]},
        )

        # 使用 `tm.assert_frame_equal` 断言 `result` 和 `expected` 的内容相等
        tm.assert_frame_equal(result, expected)
```