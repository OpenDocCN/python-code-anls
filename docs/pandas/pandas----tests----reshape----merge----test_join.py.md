# `D:\src\scipysrc\pandas\pandas\tests\reshape\merge\test_join.py`

```
import re  # 导入正则表达式模块

import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest，用于编写和运行测试用例

import pandas.util._test_decorators as td  # 导入Pandas的测试装饰器模块

import pandas as pd  # 导入Pandas库
from pandas import (  # 从Pandas库中导入多个子模块和函数
    Categorical,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    bdate_range,
    concat,
    merge,
    option_context,
)
import pandas._testing as tm  # 导入Pandas的测试模块

def get_test_data(ngroups=8, n=50):
    unique_groups = list(range(ngroups))
    arr = np.asarray(np.tile(unique_groups, n // ngroups))

    if len(arr) < n:
        arr = np.asarray(list(arr) + unique_groups[: n - len(arr)])

    np.random.default_rng(2).shuffle(arr)
    return arr  # 返回包含测试数据的NumPy数组

class TestJoin:
    # 定义测试用例类TestJoin

    @pytest.fixture
    def df(self):
        # 创建DataFrame对象df，包含"key1"、"key2"、"data1"、"data2"四列数据
        df = DataFrame(
            {
                "key1": get_test_data(),
                "key2": get_test_data(),
                "data1": np.random.default_rng(2).standard_normal(50),
                "data2": np.random.default_rng(2).standard_normal(50),
            }
        )

        # 对df进行过滤，只保留"key2"列大于1的行
        df = df[df["key2"] > 1]
        return df  # 返回过滤后的DataFrame对象df作为测试用例的fixture

    @pytest.fixture
    def df2(self):
        # 创建DataFrame对象df2，包含"key1"、"key2"、"value"三列数据
        return DataFrame(
            {
                "key1": get_test_data(n=10),
                "key2": get_test_data(ngroups=4, n=10),
                "value": np.random.default_rng(2).standard_normal(10),
            }
        )

    @pytest.fixture
    def target_source(self):
        # 创建目标DataFrame对象target和源DataFrame对象source
        data = {
            "A": [0.0, 1.0, 2.0, 3.0, 4.0],
            "B": [0.0, 1.0, 0.0, 1.0, 0.0],
            "C": ["foo1", "foo2", "foo3", "foo4", "foo5"],
            "D": bdate_range("1/1/2009", periods=5),
        }
        target = DataFrame(data, index=Index(["a", "b", "c", "d", "e"], dtype=object))

        # 创建源DataFrame对象source，以"C"列作为索引
        source = DataFrame(
            {"MergedA": data["A"], "MergedD": data["D"]}, index=data["C"]
        )
        return target, source  # 返回目标DataFrame对象target和源DataFrame对象source作为测试用例的fixture

    def test_left_outer_join(self, df, df2):
        # 测试左外连接
        joined_key2 = merge(df, df2, on="key2")  # 使用"key2"列进行连接
        _check_join(df, df2, joined_key2, ["key2"], how="left")  # 调用辅助函数_check_join进行验证

        joined_both = merge(df, df2)  # 不指定连接键，默认使用公共列进行连接
        _check_join(df, df2, joined_both, ["key1", "key2"], how="left")  # 调用辅助函数_check_join进行验证

    def test_right_outer_join(self, df, df2):
        # 测试右外连接
        joined_key2 = merge(df, df2, on="key2", how="right")  # 使用"key2"列进行右外连接
        _check_join(df, df2, joined_key2, ["key2"], how="right")  # 调用辅助函数_check_join进行验证

        joined_both = merge(df, df2, how="right")  # 不指定连接键，默认使用公共列进行右外连接
        _check_join(df, df2, joined_both, ["key1", "key2"], how="right")  # 调用辅助函数_check_join进行验证

    def test_full_outer_join(self, df, df2):
        # 测试全外连接
        joined_key2 = merge(df, df2, on="key2", how="outer")  # 使用"key2"列进行全外连接
        _check_join(df, df2, joined_key2, ["key2"], how="outer")  # 调用辅助函数_check_join进行验证

        joined_both = merge(df, df2, how="outer")  # 不指定连接键，默认使用公共列进行全外连接
        _check_join(df, df2, joined_both, ["key1", "key2"], how="outer")  # 调用辅助函数_check_join进行验证
    # 定义测试函数，用于测试内连接操作
    def test_inner_join(self, df, df2):
        # 执行基于 'key2' 列的内连接，并返回结果
        joined_key2 = merge(df, df2, on="key2", how="inner")
        # 检查内连接的结果是否符合预期
        _check_join(df, df2, joined_key2, ["key2"], how="inner")

        # 执行基于所有公共列的内连接，并返回结果
        joined_both = merge(df, df2, how="inner")
        # 检查内连接的结果是否符合预期
        _check_join(df, df2, joined_both, ["key1", "key2"], how="inner")

    # 定义测试函数，测试处理重叠列名的情况
    def test_handle_overlap(self, df, df2):
        # 执行基于 'key2' 列的合并操作，使用自定义后缀区分重叠列
        joined = merge(df, df2, on="key2", suffixes=(".foo", ".bar"))

        # 断言合并后的列名中包含 'key1.foo' 和 'key1.bar'
        assert "key1.foo" in joined
        assert "key1.bar" in joined

    # 定义测试函数，测试在指定不同列名的情况下处理重叠列名的情况
    def test_handle_overlap_arbitrary_key(self, df, df2):
        # 执行基于不同列名 'key2' 和 'key1' 的合并操作，使用自定义后缀区分重叠列
        joined = merge(
            df,
            df2,
            left_on="key2",
            right_on="key1",
            suffixes=(".foo", ".bar"),
        )

        # 断言合并后的列名中包含 'key1.foo' 和 'key2.bar'
        assert "key1.foo" in joined
        assert "key2.bar" in joined

    # 使用参数化装饰器，定义测试函数，测试在指定列 'C' 上进行连接操作
    @pytest.mark.parametrize(
        "infer_string", [False, pytest.param(True, marks=td.skip_if_no("pyarrow"))]
    )
    def test_join_on(self, target_source, infer_string):
        # 获取目标和源数据
        target, source = target_source

        # 执行基于列 'C' 的连接操作，并返回结果
        merged = target.join(source, on="C")
        # 断言连接后的 'MergedA' 列与目标数据中的 'A' 列相等
        tm.assert_series_equal(merged["MergedA"], target["A"], check_names=False)
        # 断言连接后的 'MergedD' 列与目标数据中的 'D' 列相等
        tm.assert_series_equal(merged["MergedD"], target["D"], check_names=False)

        # 测试当存在重复值时的连接操作，期望结果与预期的 DataFrame 相等
        df = DataFrame({"key": ["a", "a", "b", "b", "c"]})
        df2 = DataFrame({"value": [0, 1, 2]}, index=["a", "b", "c"])
        joined = df.join(df2, on="key")
        expected = DataFrame(
            {"key": ["a", "a", "b", "b", "c"], "value": [0, 0, 1, 1, 2]}
        )
        tm.assert_frame_equal(joined, expected)

        # 测试当部分数据缺失时的连接操作，确保缺失值为 NaN
        df_a = DataFrame([[1], [2], [3]], index=["a", "b", "c"], columns=["one"])
        df_b = DataFrame([["foo"], ["bar"]], index=[1, 2], columns=["two"])
        df_c = DataFrame([[1], [2]], index=[1, 2], columns=["three"])
        joined = df_a.join(df_b, on="one")
        joined = joined.join(df_c, on="one")
        assert np.isnan(joined["two"]["c"])
        assert np.isnan(joined["three"]["c"])

        # 测试在目标数据中不存在的列 'E' 上进行连接操作，期望抛出 KeyError 异常
        with pytest.raises(KeyError, match="^'E'$"):
            target.join(source, on="E")

        # 测试当存在重叠列名时的连接操作，期望抛出 ValueError 异常
        msg = (
            "You are trying to merge on float64 and object|string columns for key "
            "'A'. If you wish to proceed you should use pd.concat"
        )
        with pytest.raises(ValueError, match=msg):
            target.join(source, on="A")
    # 定义一个测试函数，测试当右侧索引与左侧列不匹配时是否会引发异常
    def test_join_on_fails_with_different_right_index(self):
        # 创建一个 DataFrame df，包含两列："a"为随机选择的"m"或"f"，"b"为标准正态分布随机数
        df = DataFrame(
            {
                "a": np.random.default_rng(2).choice(["m", "f"], size=3),
                "b": np.random.default_rng(2).standard_normal(3),
            }
        )
        # 创建另一个 DataFrame df2，包含两列："a"为随机选择的"m"或"f"，"b"为标准正态分布随机数，索引为两层 MultiIndex
        df2 = DataFrame(
            {
                "a": np.random.default_rng(2).choice(["m", "f"], size=10),
                "b": np.random.default_rng(2).standard_normal(10),
            },
            index=MultiIndex.from_product([range(5), ["A", "B"]]),
        )
        # 准备一个错误消息字符串，指示左侧列长度与右侧索引层级数不匹配
        msg = r'len\(left_on\) must equal the number of levels in the index of "right"'
        # 使用 pytest 的 raises 方法验证调用 merge 函数时是否会引发 ValueError 异常，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            merge(df, df2, left_on="a", right_index=True)

    # 定义一个测试函数，测试当左侧索引与右侧列不匹配时是否会引发异常
    def test_join_on_fails_with_different_left_index(self):
        # 创建一个 DataFrame df，包含两列："a"为随机选择的"m"或"f"，"b"为标准正态分布随机数，索引为两层 MultiIndex
        df = DataFrame(
            {
                "a": np.random.default_rng(2).choice(["m", "f"], size=3),
                "b": np.random.default_rng(2).standard_normal(3),
            },
            index=MultiIndex.from_arrays([range(3), list("abc")]),
        )
        # 创建另一个 DataFrame df2，包含两列："a"为随机选择的"m"或"f"，"b"为标准正态分布随机数
        df2 = DataFrame(
            {
                "a": np.random.default_rng(2).choice(["m", "f"], size=10),
                "b": np.random.default_rng(2).standard_normal(10),
            }
        )
        # 准备一个错误消息字符串，指示右侧列长度与左侧索引层级数不匹配
        msg = r'len\(right_on\) must equal the number of levels in the index of "left"'
        # 使用 pytest 的 raises 方法验证调用 merge 函数时是否会引发 ValueError 异常，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            merge(df, df2, right_on="b", left_index=True)

    # 定义一个测试函数，测试当左侧列与右侧列数量不匹配时是否会引发异常
    def test_join_on_fails_with_different_column_counts(self):
        # 创建一个 DataFrame df，包含两列："a"为随机选择的"m"或"f"，"b"为标准正态分布随机数
        df = DataFrame(
            {
                "a": np.random.default_rng(2).choice(["m", "f"], size=3),
                "b": np.random.default_rng(2).standard_normal(3),
            }
        )
        # 创建另一个 DataFrame df2，包含两列："a"为随机选择的"m"或"f"，"b"为标准正态分布随机数，索引为两层 MultiIndex
        df2 = DataFrame(
            {
                "a": np.random.default_rng(2).choice(["m", "f"], size=10),
                "b": np.random.default_rng(2).standard_normal(10),
            },
            index=MultiIndex.from_product([range(5), ["A", "B"]]),
        )
        # 准备一个错误消息字符串，指示左侧列与右侧列数量不匹配
        msg = r"len\(right_on\) must equal len\(left_on\)"
        # 使用 pytest 的 raises 方法验证调用 merge 函数时是否会引发 ValueError 异常，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            merge(df, df2, right_on="a", left_on=["a", "b"])

    # 使用 pytest 的 parametrize 装饰器，定义一个测试函数，测试当传入错误类型的对象时是否会引发异常
    @pytest.mark.parametrize("wrong_type", [2, "str", None, np.array([0, 1])])
    def test_join_on_fails_with_wrong_object_type(self, wrong_type):
        # GH12081 - 原始问题
        # GH21220 - 现在允许合并 Series 和 DataFrame，编辑测试以从测试参数中移除 Series 对象

        # 创建一个简单的 DataFrame df，包含一个列 "a"，值为 [1, 1]
        df = DataFrame({"a": [1, 1]})
        # 准备一个错误消息字符串，指示只能合并 Series 或 DataFrame 对象，但传入了错误类型的对象
        msg = (
            "Can only merge Series or DataFrame objects, "
            f"a {type(wrong_type)} was passed"
        )
        # 使用 pytest 的 raises 方法验证调用 merge 函数时是否会引发 TypeError 异常，并匹配预期的错误消息
        with pytest.raises(TypeError, match=msg):
            merge(wrong_type, df, left_on="a", right_on="a")
        with pytest.raises(TypeError, match=msg):
            merge(df, wrong_type, left_on="a", right_on="a")
    # 对目标 DataFrame 和源 DataFrame 进行连接操作，并生成预期的连接结果
    def test_join_on_pass_vector(self, target_source):
        # 解包目标和源 DataFrame
        target, source = target_source
        # 根据指定列 "C" 进行连接操作，并重新命名连接列为 "key_0"
        expected = target.join(source, on="C")
        # 重命名连接后的 DataFrame 的列名 "C" 为 "key_0"
        expected = expected.rename(columns={"C": "key_0"})
        # 选择特定列的子集作为预期结果
        expected = expected[["key_0", "A", "B", "D", "MergedA", "MergedD"]]

        # 弹出目标 DataFrame 的列 "C" 用作连接的列，并进行连接操作
        join_col = target.pop("C")
        result = target.join(source, on=join_col)
        # 使用测试框架断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    # 测试在空数据集上进行连接操作的情况
    def test_join_with_len0(self, target_source):
        # 提示：nothing to merge
        # 解包目标和源 DataFrame
        target, source = target_source
        # 在源 DataFrame 上进行空重索引后，使用列 "C" 进行连接操作
        merged = target.join(source.reindex([]), on="C")
        # 验证所有源 DataFrame 列都在连接结果中，且其值全部为 NaN
        for col in source:
            assert col in merged
            assert merged[col].isna().all()

        # 在内连接的情况下，在源 DataFrame 上进行空重索引后，使用列 "C" 进行连接操作
        merged2 = target.join(source.reindex([]), on="C", how="inner")
        # 使用测试框架断言连接结果的列索引与 merged 的列索引相等
        tm.assert_index_equal(merged2.columns, merged.columns)
        # 使用测试框架断言连接结果的长度为 0
        assert len(merged2) == 0

    # 测试在内连接模式下使用指定列进行连接操作
    def test_join_on_inner(self):
        # 创建 DataFrame 和 DataFrame2，分别包含列 "key" 和 "value"
        df = DataFrame({"key": ["a", "a", "d", "b", "b", "c"]})
        df2 = DataFrame({"value": [0, 1]}, index=["a", "b"])

        # 使用列 "key" 在内连接模式下连接 df 和 df2
        joined = df.join(df2, on="key", how="inner")

        # 创建预期的连接结果，包含所有非 NaN 值的行
        expected = df.join(df2, on="key")
        expected = expected[expected["value"].notna()]

        # 使用测试框架断言连接结果的 "key" 列相等
        tm.assert_series_equal(joined["key"], expected["key"])
        # 使用测试框架断言连接结果的 "value" 列相等（不检查数据类型）
        tm.assert_series_equal(joined["value"], expected["value"], check_dtype=False)
        # 使用测试框架断言连接结果的索引相等
        tm.assert_index_equal(joined.index, expected.index)

    # 测试在连接时使用单个键的列表
    def test_join_on_singlekey_list(self):
        # 创建 DataFrame 和 DataFrame2，分别包含列 "key" 和 "value"
        df = DataFrame({"key": ["a", "a", "b", "b", "c"]})
        df2 = DataFrame({"value": [0, 1, 2]}, index=["a", "b", "c"])

        # 在单个键 "key" 的情况下进行连接操作
        joined = df.join(df2, on=["key"])
        # 创建预期的连接结果
        expected = df.join(df2, on="key")

        # 使用测试框架断言两个 DataFrame 是否相等
        tm.assert_frame_equal(joined, expected)

    # 测试在连接时使用 Series 进行连接操作
    def test_join_on_series(self, target_source):
        # 解包目标和源 DataFrame
        target, source = target_source
        # 使用源 DataFrame 中的 "MergedA" 列进行连接操作
        result = target.join(source["MergedA"], on="C")
        # 使用源 DataFrame 中的 "MergedA" 列的 DataFrame 进行连接操作
        expected = target.join(source[["MergedA"]], on="C")
        # 使用测试框架断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    # 测试在连接时遇到的特定 Bug
    def test_join_on_series_buglet(self):
        # GH #638
        # 创建 DataFrame 和 Series
        df = DataFrame({"a": [1, 1]})
        ds = Series([2], index=[1], name="b")
        # 使用列 "a" 进行连接操作
        result = df.join(ds, on="a")
        # 创建预期的连接结果 DataFrame
        expected = DataFrame({"a": [1, 1], "b": [2, 2]}, index=df.index)
        # 使用测试框架断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    # 测试在混合索引上进行连接操作的情况
    def test_join_index_mixed(self, join_type):
        # 创建包含布尔和字符串列的 DataFrame1
        df1 = DataFrame(index=np.arange(10))
        df1["bool"] = True
        df1["string"] = "foo"

        # 创建包含整数和浮点数列的 DataFrame2
        df2 = DataFrame(index=np.arange(5, 15))
        df2["int"] = 1
        df2["float"] = 1.0

        # 使用指定的连接方式在 df1 和 df2 之间进行连接操作
        joined = df1.join(df2, how=join_type)
        # 调用手动编写的连接函数 _join_by_hand 获取预期的连接结果
        expected = _join_by_hand(df1, df2, how=join_type)
        # 使用测试框架断言两个 DataFrame 是否相等
        tm.assert_frame_equal(joined, expected)

        # 使用指定的连接方式在 df2 和 df1 之间进行连接操作
        joined = df2.join(df1, how=join_type)
        # 调用手动编写的连接函数 _join_by_hand 获取预期的连接结果
        expected = _join_by_hand(df2, df1, how=join_type)
        # 使用测试框架断言两个 DataFrame 是否相等
        tm.assert_frame_equal(joined, expected)
    def test_join_index_mixed_overlap(self):
        df1 = DataFrame(
            {"A": 1.0, "B": 2, "C": "foo", "D": True},
            index=np.arange(10),
            columns=["A", "B", "C", "D"],
        )
        # 断言列'B'的数据类型为np.int64
        assert df1["B"].dtype == np.int64
        # 断言列'D'的数据类型为np.bool_
        assert df1["D"].dtype == np.bool_

        df2 = DataFrame(
            {"A": 1.0, "B": 2, "C": "foo", "D": True},
            index=np.arange(0, 10, 2),
            columns=["A", "B", "C", "D"],
        )

        # overlap
        # 使用left和right后缀连接df1和df2
        joined = df1.join(df2, lsuffix="_one", rsuffix="_two")
        expected_columns = [
            "A_one",
            "B_one",
            "C_one",
            "D_one",
            "A_two",
            "B_two",
            "C_two",
            "D_two",
        ]
        # 重命名df1和df2的列名
        df1.columns = expected_columns[:4]
        df2.columns = expected_columns[4:]
        # 手动调用_join_by_hand函数进行期望结果的合并
        expected = _join_by_hand(df1, df2)
        # 断言joined和expected的DataFrame是否相等
        tm.assert_frame_equal(joined, expected)

    def test_join_empty_bug(self):
        # 在0.4.3版本中生成异常
        x = DataFrame()
        x.join(DataFrame([3], index=[0], columns=["A"]), how="outer")

    def test_join_unconsolidated(self):
        # GH #331
        a = DataFrame(
            np.random.default_rng(2).standard_normal((30, 2)), columns=["a", "b"]
        )
        c = Series(np.random.default_rng(2).standard_normal(30))
        a["c"] = c
        d = DataFrame(np.random.default_rng(2).standard_normal((30, 1)), columns=["q"])

        # it works!
        # 对a和d进行连接操作
        a.join(d)
        d.join(a)

    def test_join_multiindex(self):
        index1 = MultiIndex.from_arrays(
            [["a", "a", "a", "b", "b", "b"], [1, 2, 3, 1, 2, 3]],
            names=["first", "second"],
        )

        index2 = MultiIndex.from_arrays(
            [["b", "b", "b", "c", "c", "c"], [1, 2, 3, 1, 2, 3]],
            names=["first", "second"],
        )

        df1 = DataFrame(
            data=np.random.default_rng(2).standard_normal(6),
            index=index1,
            columns=["var X"],
        )
        df2 = DataFrame(
            data=np.random.default_rng(2).standard_normal(6),
            index=index2,
            columns=["var Y"],
        )

        # 按第0级索引排序df1和df2
        df1 = df1.sort_index(level=0)
        df2 = df2.sort_index(level=0)

        # 使用outer方式对df1和df2进行连接
        joined = df1.join(df2, how="outer")
        # 构建期望的连接结果
        ex_index = Index(index1.values).union(Index(index2.values))
        expected = df1.reindex(ex_index).join(df2.reindex(ex_index))
        expected.index.names = index1.names
        # 断言joined和expected的DataFrame是否相等
        tm.assert_frame_equal(joined, expected)
        # 断言joined的索引名称与index1的索引名称相同
        assert joined.index.names == index1.names

        # 按第1级索引排序df1和df2
        df1 = df1.sort_index(level=1)
        df2 = df2.sort_index(level=1)

        # 使用outer方式对df1和df2进行连接，并按第0级索引排序
        joined = df1.join(df2, how="outer").sort_index(level=0)
        # 构建期望的连接结果
        ex_index = Index(index1.values).union(Index(index2.values))
        expected = df1.reindex(ex_index).join(df2.reindex(ex_index))
        expected.index.names = index1.names

        # 断言joined和expected的DataFrame是否相等
        tm.assert_frame_equal(joined, expected)
        # 断言joined的索引名称与index1的索引名称相同
        assert joined.index.names == index1.names
    def test_join_inner_multiindex(self, lexsorted_two_level_string_multiindex):
        # 定义第一个键列表
        key1 = ["bar", "bar", "bar", "foo", "foo", "baz", "baz", "qux", "qux", "snap"]
        # 定义第二个键列表
        key2 = [
            "two",
            "one",
            "three",
            "one",
            "two",
            "one",
            "two",
            "two",
            "three",
            "one",
        ]

        # 生成随机数据
        data = np.random.default_rng(2).standard_normal(len(key1))
        # 创建包含键和数据的DataFrame
        data = DataFrame({"key1": key1, "key2": key2, "data": data})

        # 使用给定的多级索引创建DataFrame
        index = lexsorted_two_level_string_multiindex
        # 创建具有特定形状和列名的DataFrame
        to_join = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)),
            index=index,
            columns=["j_one", "j_two", "j_three"],
        )

        # 在指定键上进行内部连接
        joined = data.join(to_join, on=["key1", "key2"], how="inner")
        # 预期结果通过合并两个DataFrame来生成
        expected = merge(
            data,
            to_join.reset_index(),
            left_on=["key1", "key2"],
            right_on=["first", "second"],
            how="inner",
            sort=False,
        )

        # 通过索引进行另一种预期内部连接
        expected2 = merge(
            to_join,
            data,
            right_on=["key1", "key2"],
            left_index=True,
            how="inner",
            sort=False,
        )
        # 断言连接后的DataFrame与预期结果的重建版相等
        tm.assert_frame_equal(joined, expected2.reindex_like(joined))

        # 从预期结果中移除指定列并将索引设为连接后的DataFrame的索引
        expected = expected.drop(["first", "second"], axis=1)
        expected.index = joined.index

        # 断言连接后的DataFrame的索引是单调递增的
        assert joined.index.is_monotonic_increasing
        # 断言连接后的DataFrame与预期结果相等
        tm.assert_frame_equal(joined, expected)

        # 使用pytest检查预期的异常情况是否引发
        with pytest.raises(
            pd.errors.MergeError, match="Not allowed to merge between different levels"
        ):
            merge(new_df, other_df, left_index=True, right_index=True)
    # 定义一个测试函数，用于测试 DataFrame 对象的连接操作，其中包含不同数据类型的列
    def test_join_float64_float32(self):
        # 创建一个包含 10 行 2 列随机浮点数的 DataFrame 对象 a，列名为 "a" 和 "b"，数据类型为 np.float64
        a = DataFrame(
            np.random.default_rng(2).standard_normal((10, 2)),
            columns=["a", "b"],
            dtype=np.float64,
        )
        # 创建一个包含 10 行 1 列随机浮点数的 DataFrame 对象 b，列名为 "c"，数据类型为 np.float32
        b = DataFrame(
            np.random.default_rng(2).standard_normal((10, 1)),
            columns=["c"],
            dtype=np.float32,
        )
        # 对 DataFrame 对象 a 和 b 进行连接操作，生成新的 DataFrame 对象 joined
        joined = a.join(b)
        # 断言新生成的 joined 对象中列 'a' 的数据类型为 "float64"
        assert joined.dtypes["a"] == "float64"
        # 断言新生成的 joined 对象中列 'b' 的数据类型为 "float64"
        assert joined.dtypes["b"] == "float64"
        # 断言新生成的 joined 对象中列 'c' 的数据类型为 "float32"
        assert joined.dtypes["c"] == "float32"

        # 创建一个包含 100 个元素的随机整数数组 a，数据类型为 "int64"
        a = np.random.default_rng(2).integers(0, 5, 100).astype("int64")
        # 创建一个包含 100 个元素的随机浮点数数组 b，数据类型为 "float64"
        b = np.random.default_rng(2).random(100).astype("float64")
        # 创建一个包含 100 个元素的随机浮点数数组 c，数据类型为 "float32"
        c = np.random.default_rng(2).random(100).astype("float32")
        # 创建一个包含列 'a', 'b', 'c' 的 DataFrame 对象 df
        df = DataFrame({"a": a, "b": b, "c": c})
        # 创建一个与 df 结构相同的 DataFrame 对象 xpdf
        xpdf = DataFrame({"a": a, "b": b, "c": c})
        # 创建一个包含 5 个随机浮点数的 Series 对象 s，列名为 "md"，数据类型为 "float32"
        s = DataFrame(
            np.random.default_rng(2).random(5).astype("float32"), columns=["md"]
        )
        # 使用 df 对象与 s 对象进行合并操作，使用列 'a' 和索引进行连接，生成结果 DataFrame 对象 rs
        rs = df.merge(s, left_on="a", right_index=True)
        # 断言结果 DataFrame 对象 rs 中列 'a' 的数据类型为 "int64"
        assert rs.dtypes["a"] == "int64"
        # 断言结果 DataFrame 对象 rs 中列 'b' 的数据类型为 "float64"
        assert rs.dtypes["b"] == "float64"
        # 断言结果 DataFrame 对象 rs 中列 'c' 的数据类型为 "float32"
        assert rs.dtypes["c"] == "float32"
        # 断言结果 DataFrame 对象 rs 中列 'md' 的数据类型为 "float32"
        assert rs.dtypes["md"] == "float32"

        # 使用 xpdf 对象与 s 对象进行合并操作，使用列 'a' 和索引进行连接，生成结果 DataFrame 对象 xp
        xp = xpdf.merge(s, left_on="a", right_index=True)
        # 使用测试工具库 tm 中的方法 assert_frame_equal 比较 rs 和 xp 两个 DataFrame 对象，确保它们相等
        tm.assert_frame_equal(rs, xp)
    # 定义测试方法，测试多个数据框的联接（当索引不唯一时）
    def test_join_many_non_unique_index(self):
        # 创建第一个数据框 df1
        df1 = DataFrame({"a": [1, 1], "b": [1, 1], "c": [10, 20]})
        # 创建第二个数据框 df2
        df2 = DataFrame({"a": [1, 1], "b": [1, 2], "d": [100, 200]})
        # 创建第三个数据框 df3
        df3 = DataFrame({"a": [1, 1], "b": [1, 2], "e": [1000, 2000]})
        # 使用列 'a' 和 'b' 设置 df1 的索引，并赋值给 idf1
        idf1 = df1.set_index(["a", "b"])
        # 使用列 'a' 和 'b' 设置 df2 的索引，并赋值给 idf2
        idf2 = df2.set_index(["a", "b"])
        # 使用列 'a' 和 'b' 设置 df3 的索引，并赋值给 idf3
        idf3 = df3.set_index(["a", "b"])

        # 对 idf1 与 [idf2, idf3] 进行外连接，并将结果赋值给 result
        result = idf1.join([idf2, idf3], how="outer")

        # 将 df1 与 df2 按列 'a' 和 'b' 外连接，并将结果赋值给 df_partially_merged
        df_partially_merged = merge(df1, df2, on=["a", "b"], how="outer")
        # 将 df_partially_merged 与 df3 按列 'a' 和 'b' 外连接，并将结果赋值给 expected
        expected = merge(df_partially_merged, df3, on=["a", "b"], how="outer")

        # 将 result 重置索引，并赋值回 result
        result = result.reset_index()
        # 从 expected 中选择与 result 列相同的列，并赋值给 expected
        expected = expected[result.columns]
        # 将 expected 中的列 'a' 和 'b' 的数据类型转换为 int64
        expected["a"] = expected.a.astype("int64")
        expected["b"] = expected.b.astype("int64")
        # 使用 pytest 的 assert_frame_equal 函数比较 result 和 expected
        tm.assert_frame_equal(result, expected)

        # 创建第一个数据框 df1
        df1 = DataFrame({"a": [1, 1, 1], "b": [1, 1, 1], "c": [10, 20, 30]})
        # 创建第二个数据框 df2
        df2 = DataFrame({"a": [1, 1, 1], "b": [1, 1, 2], "d": [100, 200, 300]})
        # 创建第三个数据框 df3
        df3 = DataFrame({"a": [1, 1, 1], "b": [1, 1, 2], "e": [1000, 2000, 3000]})
        # 使用列 'a' 和 'b' 设置 df1 的索引，并赋值给 idf1
        idf1 = df1.set_index(["a", "b"])
        # 使用列 'a' 和 'b' 设置 df2 的索引，并赋值给 idf2
        idf2 = df2.set_index(["a", "b"])
        # 使用列 'a' 和 'b' 设置 df3 的索引，并赋值给 idf3
        idf3 = df3.set_index(["a", "b"])
        # 对 idf1 与 [idf2, idf3] 进行内连接，并将结果赋值给 result
        result = idf1.join([idf2, idf3], how="inner")

        # 将 df1 与 df2 按列 'a' 和 'b' 内连接，并将结果赋值给 df_partially_merged
        df_partially_merged = merge(df1, df2, on=["a", "b"], how="inner")
        # 将 df_partially_merged 与 df3 按列 'a' 和 'b' 内连接，并将结果赋值给 expected
        expected = merge(df_partially_merged, df3, on=["a", "b"], how="inner")

        # 将 result 重置索引，并赋值回 result
        result = result.reset_index()

        # 使用 pytest 的 assert_frame_equal 函数比较 result 和 expected 中 result 列对应的部分
        tm.assert_frame_equal(result, expected.loc[:, result.columns])

        # 创建包含列 'A', 'B', 'C', 'D' 的数据框 df，其中 'C' 和 'D' 使用随机数生成
        df = DataFrame(
            {
                "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
                "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
                "C": np.random.default_rng(2).standard_normal(8),
                "D": np.random.default_rng(2).standard_normal(8),
            }
        )
        # 创建名为 'TEST' 的 Series，使用重复的序列和索引来定义
        s = Series(
            np.repeat(np.arange(8), 2), index=np.repeat(np.arange(8), 2), name="TEST"
        )
        # 使用内连接方式将 df 与 s 连接，并将结果赋值给 inner
        inner = df.join(s, how="inner")
        # 使用外连接方式将 df 与 s 连接，并将结果赋值给 outer
        outer = df.join(s, how="outer")
        # 使用左连接方式将 df 与 s 连接，并将结果赋值给 left
        left = df.join(s, how="left")
        # 使用右连接方式将 df 与 s 连接，并将结果赋值给 right
        right = df.join(s, how="right")
        # 使用 pytest 的 assert_frame_equal 函数比较 inner 和 outer
        tm.assert_frame_equal(inner, outer)
        # 使用 pytest 的 assert_frame_equal 函数比较 inner 和 left
        tm.assert_frame_equal(inner, left)
        # 使用 pytest 的 assert_frame_equal 函数比较 inner 和 right
        tm.assert_frame_equal(inner, right)
    # 定义一个测试方法，用于测试 DataFrame 的 join 和排序功能
    def test_join_sort(self, infer_string):
        # 设置上下文选项，用于指定将来推断字符串的方式
        with option_context("future.infer_string", infer_string):
            # 创建左侧 DataFrame，包含键和对应值的列
            left = DataFrame(
                {"key": ["foo", "bar", "baz", "foo"], "value": [1, 2, 3, 4]}
            )
            # 创建右侧 DataFrame，包含值和索引的列
            right = DataFrame({"value2": ["a", "b", "c"]}, index=["bar", "baz", "foo"])

            # 执行按键值连接左右 DataFrame，并排序结果
            joined = left.join(right, on="key", sort=True)
            # 创建预期的 DataFrame，包含键、左侧值、右侧值，并指定索引
            expected = DataFrame(
                {
                    "key": ["bar", "baz", "foo", "foo"],
                    "value": [2, 3, 1, 4],
                    "value2": ["a", "b", "c", "c"],
                },
                index=[1, 2, 0, 3],
            )
            # 比较实际连接结果和预期结果，确认它们相等
            tm.assert_frame_equal(joined, expected)

            # 冒烟测试，执行按键值连接左右 DataFrame，不排序结果
            joined = left.join(right, on="key", sort=False)
            # 检查连接后的索引是否与预期的范围相等
            tm.assert_index_equal(joined.index, Index(range(4)), exact=True)

    # 定义一个测试方法，用于测试包含混合非唯一索引的 DataFrame 连接
    def test_join_mixed_non_unique_index(self):
        # 创建第一个 DataFrame，包含列 'a' 和非唯一索引
        df1 = DataFrame({"a": [1, 2, 3, 4]}, index=[1, 2, 3, "a"])
        # 创建第二个 DataFrame，包含列 'b' 和索引
        df2 = DataFrame({"b": [5, 6, 7, 8]}, index=[1, 3, 3, 4])
        # 执行按索引连接第一个和第二个 DataFrame
        result = df1.join(df2)
        # 创建预期的 DataFrame，包含列 'a' 和 'b'，并指定索引
        expected = DataFrame(
            {"a": [1, 2, 3, 3, 4], "b": [5, np.nan, 6, 7, np.nan]},
            index=[1, 2, 3, 3, "a"],
        )
        # 比较实际连接结果和预期结果，确认它们相等
        tm.assert_frame_equal(result, expected)

        # 创建第三个 DataFrame，包含列 'a' 和混合非唯一索引
        df3 = DataFrame({"a": [1, 2, 3, 4]}, index=[1, 2, 2, "a"])
        # 创建第四个 DataFrame，包含列 'b' 和索引
        df4 = DataFrame({"b": [5, 6, 7, 8]}, index=[1, 2, 3, 4])
        # 执行按索引连接第三个和第四个 DataFrame
        result = df3.join(df4)
        # 创建预期的 DataFrame，包含列 'a' 和 'b'，并指定索引
        expected = DataFrame(
            {"a": [1, 2, 3, 4], "b": [5, 6, 6, np.nan]}, index=[1, 2, 2, "a"]
        )
        # 比较实际连接结果和预期结果，确认它们相等
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，用于测试包含非唯一期间索引的 DataFrame 连接
    def test_join_non_unique_period_index(self):
        # 创建一个时间期间范围的索引
        index = pd.period_range("2016-01-01", periods=16, freq="M")
        # 创建 DataFrame，包含从 0 到索引长度的数据，列名为 'pnum'
        df = DataFrame(list(range(len(index))), index=index, columns=["pnum"])
        # 将同一个 DataFrame 连接到自身
        df2 = concat([df, df])
        # 执行按内连接方式连接 DataFrame 和其自身
        result = df.join(df2, how="inner", rsuffix="_df2")
        # 创建预期的 DataFrame，包含 'pnum' 和 'pnum_df2' 列，按索引排序
        expected = DataFrame(
            np.tile(np.arange(16, dtype=np.int64).repeat(2).reshape(-1, 1), 2),
            columns=["pnum", "pnum_df2"],
            index=df2.sort_index().index,
        )
        # 比较实际连接结果和预期结果，确认它们相等
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，用于测试带有后缀的混合类型连接
    def test_mixed_type_join_with_suffix(self):
        # 创建一个包含随机标准正态分布数据的 DataFrame，列名为 'a' 到 'f'
        df = DataFrame(
            np.random.default_rng(2).standard_normal((20, 6)),
            columns=["a", "b", "c", "d", "e", "f"],
        )
        # 在第一列插入名为 'id' 的列，所有值为 0
        df.insert(0, "id", 0)
        # 在第六列插入名为 'dt' 的列，所有值为 'foo'
        df.insert(5, "dt", "foo")

        # 按 'id' 列分组 DataFrame
        grouped = df.groupby("id")
        # 期望抛出类型错误异常消息
        msg = re.escape("agg function failed [how->mean,dtype->")
        with pytest.raises(TypeError, match=msg):
            # 尝试对分组后的 DataFrame 执行均值聚合
            grouped.mean()
        # 对数值列进行均值聚合
        mn = grouped.mean(numeric_only=True)
        # 对分组后的 DataFrame 执行计数
        cn = grouped.count()

        # 执行按 'id' 列连接均值和计数结果，指定后缀 '_right'
        mn.join(cn, rsuffix="_right")
    def test_join_many(self):
        # 创建一个 DataFrame，包含随机生成的数据，共10行6列
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 6)), columns=list("abcdef")
        )
        # 将 DataFrame 按列切片形成列表 df_list，每个元素为一个包含部分列的 DataFrame
        df_list = [df[["a", "b"]], df[["c", "d"]], df[["e", "f"]]]

        # 对 df_list 中的第一个 DataFrame 与剩余 DataFrame 进行 join 操作
        joined = df_list[0].join(df_list[1:])
        # 断言 joined 与原始 df 相等
        tm.assert_frame_equal(joined, df)

        # 重新设置 df_list，每个 DataFrame 切片不同的行
        df_list = [df[["a", "b"]][:-2], df[["c", "d"]][2:], df[["e", "f"]][1:9]]

        def _check_diff_index(df_list, result, exp_index):
            # 对 df_list 中的每个 DataFrame 执行重新索引操作，以匹配给定的索引 exp_index
            reindexed = [x.reindex(exp_index) for x in df_list]
            # 对重新索引后的 DataFrame 列表进行 join 操作
            expected = reindexed[0].join(reindexed[1:])
            # 断言 result 与 expected 相等
            tm.assert_frame_equal(result, expected)

        # 使用不同的 join 类型对 df_list 中的 DataFrame 进行 join 操作
        joined = df_list[0].join(df_list[1:], how="outer")
        # 调用 _check_diff_index 函数进行断言
        _check_diff_index(df_list, joined, df.index)

        joined = df_list[0].join(df_list[1:])
        _check_diff_index(df_list, joined, df_list[0].index)

        joined = df_list[0].join(df_list[1:], how="inner")
        _check_diff_index(df_list, joined, df.index[2:8])

        # 测试使用非索引字段进行 join 操作时是否抛出 ValueError 异常
        msg = "Joining multiple DataFrames only supported for joining on index"
        with pytest.raises(ValueError, match=msg):
            df_list[0].join(df_list[1:], on="a")

    def test_join_many_mixed(self):
        # 创建一个 DataFrame，包含随机生成的数据，共8行4列，并包含名为 'key' 的列
        df = DataFrame(
            np.random.default_rng(2).standard_normal((8, 4)),
            columns=["A", "B", "C", "D"],
        )
        # 向 DataFrame 添加名为 'key' 的列，值为交替的 'foo' 和 'bar'
        df["key"] = ["foo", "bar"] * 4
        # 根据列名选择部分列形成新的 DataFrame df1, df2, df3
        df1 = df.loc[:, ["A", "B"]]
        df2 = df.loc[:, ["C", "D"]]
        df3 = df.loc[:, ["key"]]

        # 将 df1、df2 和 df3 进行 join 操作
        result = df1.join([df2, df3])
        # 断言 result 与原始 df 相等
        tm.assert_frame_equal(result, df)
   `
    # 测试连接重复列的情况
    def test_join_dups(self):
        # 创建包含重复列的数据框 df
        df = concat(
            [
                DataFrame(
                    np.random.default_rng(2).standard_normal((10, 4)),
                    columns=["A", "A", "B", "B"],
                ),
                DataFrame(
                    np.random.default_rng(2).integers(0, 10, size=20).reshape(10, 2),
                    columns=["A", "C"],
                ),
            ],
            axis=1,
        )

        # 预期的结果是将 df 自身连接起来
        expected = concat([df, df], axis=1)
        # 执行 df 的自连接操作，并使用后缀 "_2" 标记右侧数据框
        result = df.join(df, rsuffix="_2")
        # 将结果的列名设为预期结果的列名
        result.columns = expected.columns
        # 断言结果与预期相等
        tm.assert_frame_equal(result, expected)

        # GH 4975, 测试在重复列上进行无效连接的情况
        w = DataFrame(
            np.random.default_rng(2).standard_normal((4, 2)), columns=["x", "y"]
        )
        x = DataFrame(
            np.random.default_rng(2).standard_normal((4, 2)), columns=["x", "y"]
        )
        y = DataFrame(
            np.random.default_rng(2).standard_normal((4, 2)), columns=["x", "y"]
        )
        z = DataFrame(
            np.random.default_rng(2).standard_normal((4, 2)), columns=["x", "y"]
        )

        # 将 x, y, z 数据框进行外连接，并存储在 dta 变量中
        dta = x.merge(y, left_index=True, right_index=True).merge(
            z, left_index=True, right_index=True, how="outer"
        )
        # GH 40991: 自 2.0 版本起会导致列重复
        with pytest.raises(
            pd.errors.MergeError,
            match="Passing 'suffixes' which cause duplicate columns",
        ):
            # 在 dta 上执行与 w 的连接操作，使用左右索引连接
            dta.merge(w, left_index=True, right_index=True)

    # 测试多对多连接的情况
    def test_join_multi_to_multi(self, join_type):
        # GH 20475
        # 创建左侧数据框的多级索引
        leftindex = MultiIndex.from_product(
            [list("abc"), list("xy"), [1, 2]], names=["abc", "xy", "num"]
        )
        # 创建左侧数据框 left，包含列 "v1"，并以 leftindex 作为索引
        left = DataFrame({"v1": range(12)}, index=leftindex)

        # 创建右侧数据框的多级索引
        rightindex = MultiIndex.from_product(
            [list("abc"), list("xy")], names=["abc", "xy"]
        )
        # 创建右侧数据框 right，包含列 "v2"，并以 rightindex 作为索引
        right = DataFrame({"v2": [100 * i for i in range(1, 7)]}, index=rightindex)

        # 执行左右数据框的连接，连接键为 ["abc", "xy"]，连接方式由参数 join_type 指定
        result = left.join(right, on=["abc", "xy"], how=join_type)
        # 构造预期结果，通过重置索引并执行基于 ["abc", "xy"] 的内连接来实现
        expected = (
            left.reset_index()
            .merge(right.reset_index(), on=["abc", "xy"], how=join_type)
            .set_index(["abc", "xy", "num"])
        )
        # 断言预期结果与实际结果相等
        tm.assert_frame_equal(expected, result)

        # 定义错误消息，用于在连接键数量与右侧数据框索引级别数不匹配时引发 ValueError
        msg = r'len\(left_on\) must equal the number of levels in the index of "right"'
        # 预期通过连接键 "xy" 与右侧数据框 left 进行连接会引发 ValueError
        with pytest.raises(ValueError, match=msg):
            left.join(right, on="xy", how=join_type)

        # 预期通过连接键 ["abc", "xy"] 与左侧数据框 right 进行连接会引发 ValueError
        with pytest.raises(ValueError, match=msg):
            right.join(left, on=["abc", "xy"], how=join_type)
    def test_join_on_tz_aware_datetimeindex(self):
        # 测试用例：GH 23931, 26335
        # 创建第一个DataFrame，包含日期范围和时区信息
        df1 = DataFrame(
            {
                "date": pd.date_range(
                    start="2018-01-01", periods=5, tz="America/Chicago"
                ),
                "vals": list("abcde"),
            }
        )

        # 创建第二个DataFrame，包含日期范围和时区信息
        df2 = DataFrame(
            {
                "date": pd.date_range(
                    start="2018-01-03", periods=5, tz="America/Chicago"
                ),
                "vals_2": list("tuvwx"),
            }
        )
        # 使用date列进行连接，并生成结果DataFrame
        result = df1.join(df2.set_index("date"), on="date")
        # 复制df1作为期望结果
        expected = df1.copy()
        # 将vals_2列添加到期望结果中，并设置缺失值
        expected["vals_2"] = Series([np.nan] * 2 + list("tuv"))
        # 断言结果DataFrame与期望DataFrame相等
        tm.assert_frame_equal(result, expected)

    def test_join_datetime_string(self):
        # 测试用例：GH 5647
        # 创建第一个DataFrame，包含日期字符串
        dfa = DataFrame(
            [
                ["2012-08-02", "L", 10],
                ["2012-08-02", "J", 15],
                ["2013-04-06", "L", 20],
                ["2013-04-06", "J", 25],
            ],
            columns=["x", "y", "a"],
        )
        # 将x列转换为datetime类型
        dfa["x"] = pd.to_datetime(dfa["x"]).astype("M8[ns]")
        # 创建第二个DataFrame，包含日期字符串
        dfb = DataFrame(
            [["2012-08-02", "J", 1], ["2013-04-06", "L", 2]],
            columns=["x", "y", "z"],
            index=[2, 4],
        )
        # 将x列转换为datetime类型
        dfb["x"] = pd.to_datetime(dfb["x"]).astype("M8[ns]")
        # 使用x和y列进行连接，并生成结果DataFrame
        result = dfb.join(dfa.set_index(["x", "y"]), on=["x", "y"])
        # 创建期望结果DataFrame，包含特定日期时间戳
        expected = DataFrame(
            [
                [Timestamp("2012-08-02 00:00:00"), "J", 1, 15],
                [Timestamp("2013-04-06 00:00:00"), "L", 2, 20],
            ],
            index=[2, 4],
            columns=["x", "y", "z", "a"],
        )
        # 将x列转换为datetime类型
        expected["x"] = expected["x"].astype("M8[ns]")
        # 断言结果DataFrame与期望DataFrame相等
        tm.assert_frame_equal(result, expected)

    def test_join_with_categorical_index(self):
        # 测试用例：GH47812
        # 创建分类索引
        ix = ["a", "b"]
        id1 = pd.CategoricalIndex(ix, categories=ix)
        id2 = pd.CategoricalIndex(reversed(ix), categories=reversed(ix))

        # 创建第一个DataFrame，使用分类索引
        df1 = DataFrame({"c1": ix}, index=id1)
        # 创建第二个DataFrame，使用反转的分类索引
        df2 = DataFrame({"c2": reversed(ix)}, index=id2)
        # 使用索引进行连接，并生成结果DataFrame
        result = df1.join(df2)
        # 创建期望结果DataFrame，包含指定的分类索引
        expected = DataFrame(
            {"c1": ["a", "b"], "c2": ["a", "b"]},
            index=pd.CategoricalIndex(["a", "b"], categories=["a", "b"]),
        )
        # 断言结果DataFrame与期望DataFrame相等
        tm.assert_frame_equal(result, expected)
# 检查连接列是否不含缺失值，进行一些基本的测试
def _check_join(left, right, result, join_col, how="left", lsuffix="_x", rsuffix="_y"):
    for c in join_col:
        assert result[c].notna().all()  # 确保连接列在结果中没有缺失值

    # 按连接列对左表和右表进行分组
    left_grouped = left.groupby(join_col)
    right_grouped = right.groupby(join_col)

    # 对结果按连接列进行分组，然后进行以下操作
    for group_key, group in result.groupby(join_col):
        # 从组中选择左表和右表的列，根据指定的后缀
        l_joined = _restrict_to_columns(group, left.columns, lsuffix)
        r_joined = _restrict_to_columns(group, right.columns, rsuffix)

        # 尝试获取左表中与当前分组键匹配的组
        try:
            lgroup = left_grouped.get_group(group_key)
        except KeyError as err:
            # 如果当前分组键在左表中不存在，根据连接方式进行断言检查
            if how in ("left", "inner"):
                raise AssertionError(
                    f"key {group_key} should not have been in the join"
                ) from err

            _assert_all_na(l_joined, left.columns, join_col)  # 如果应该为右连接或内连接，则确保左连接列中全部为缺失值
        else:
            _assert_same_contents(l_joined, lgroup)  # 检查左表连接的内容与分组内容是否一致

        # 尝试获取右表中与当前分组键匹配的组
        try:
            rgroup = right_grouped.get_group(group_key)
        except KeyError as err:
            # 如果当前分组键在右表中不存在，根据连接方式进行断言检查
            if how in ("right", "inner"):
                raise AssertionError(
                    f"key {group_key} should not have been in the join"
                ) from err

            _assert_all_na(r_joined, right.columns, join_col)  # 如果应该为左连接或内连接，则确保右连接列中全部为缺失值
        else:
            _assert_same_contents(r_joined, rgroup)  # 检查右表连接的内容与分组内容是否一致


# 从给定的组中筛选出与指定列匹配的列，并根据后缀进行重命名
def _restrict_to_columns(group, columns, suffix):
    found = [
        c for c in group.columns if c in columns or c.replace(suffix, "") in columns
    ]

    # 根据筛选出的列重新设置组的内容
    group = group.loc[:, found]

    # 去除列名中的后缀（如果有的话）
    group = group.rename(columns=lambda x: x.replace(suffix, ""))

    # 根据指定的列顺序重新排列组的列
    group = group.loc[:, columns]

    return group


# 断言两个数据框的内容是否相同
def _assert_same_contents(join_chunk, source):
    NA_SENTINEL = -1234567  # drop_duplicates not so NA-friendly...

    # 填充缺失值并去除重复行，然后获取其值
    jvalues = join_chunk.fillna(NA_SENTINEL).drop_duplicates().values
    svalues = source.fillna(NA_SENTINEL).drop_duplicates().values

    # 使用集合来检查两个数据框的内容是否完全相同
    rows = {tuple(row) for row in jvalues}
    assert len(rows) == len(source)
    assert all(tuple(row) in rows for row in svalues)


# 断言给定的连接块是否所有列都为缺失值（除了连接列）
def _assert_all_na(join_chunk, source_columns, join_col):
    for c in source_columns:
        if c in join_col:
            continue
        assert join_chunk[c].isna().all()


# 手动实现数据框连接操作，根据指定的连接方式
def _join_by_hand(a, b, how="left"):
    # 使用指定的连接方式对索引进行连接
    join_index = a.index.join(b.index, how=how)

    # 根据连接后的索引重新索引左右数据框
    a_re = a.reindex(join_index)
    b_re = b.reindex(join_index)

    # 合并结果数据框的列
    result_columns = a.columns.append(b.columns)

    # 将右数据框的列添加到左数据框中，并重新索引结果列
    for col, s in b_re.items():
        a_re[col] = s
    return a_re.reindex(columns=result_columns)


# 测试多索引的内连接操作，验证特定的 GitHub 问题
def test_join_inner_multiindex_deterministic_order():
    # GH: 36910
    # 创建左右数据框，使用多级索引进行设置
    left = DataFrame(
        data={"e": 5},
        index=MultiIndex.from_tuples([(1, 2, 4)], names=("a", "b", "d")),
    )
    right = DataFrame(
        data={"f": 6}, index=MultiIndex.from_tuples([(2, 3)], names=("b", "c"))
    )
    # 执行内连接操作
    result = left.join(right, how="inner")
    # 创建预期的 DataFrame 对象，包含列 'e' 和 'f'，并设定 MultiIndex 索引
    expected = DataFrame(
        {"e": [5], "f": [6]},
        index=MultiIndex.from_tuples([(1, 2, 4, 3)], names=("a", "b", "d", "c")),
    )
    # 使用测试工具（tm.assert_frame_equal）比较计算结果（result）与预期结果（expected）
    tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize(
    # 参数化测试：定义输入列和输出列的参数组合
    ("input_col", "output_cols"), [("b", ["a", "b"]), ("a", ["a_x", "a_y"])]
)
def test_join_cross(input_col, output_cols):
    # GH#5401: GitHub issue编号，表示这段代码与该问题相关
    # 创建左侧DataFrame，包含列'a'，数据为[1, 3]
    left = DataFrame({"a": [1, 3]})
    # 创建右侧DataFrame，根据输入的列名(input_col)动态生成，数据为[3, 4]
    right = DataFrame({input_col: [3, 4]})
    # 执行交叉连接(join)，左侧列名后缀为"_x"，右侧列名后缀为"_y"
    result = left.join(right, how="cross", lsuffix="_x", rsuffix="_y")
    # 期望结果DataFrame，根据输出列的名称动态生成
    expected = DataFrame({output_cols[0]: [1, 1, 3, 3], output_cols[1]: [3, 4, 3, 4]})
    # 使用测试框架中的断言，比较实际结果和期望结果DataFrame
    tm.assert_frame_equal(result, expected)


def test_join_multiindex_one_level(join_type):
    # GH#36909: GitHub issue编号，表示这段代码与该问题相关
    # 创建左侧DataFrame，包含单层MultiIndex，数据为{"c": 3}，索引为[(1, 2)]
    left = DataFrame(
        data={"c": 3}, index=MultiIndex.from_tuples([(1, 2)], names=("a", "b"))
    )
    # 创建右侧DataFrame，包含单层MultiIndex，数据为{"d": 4}，索引为[(2,)]
    right = DataFrame(data={"d": 4}, index=MultiIndex.from_tuples([(2,)], names=("b",)))
    # 执行连接操作，连接方式根据传入的join_type参数确定
    result = left.join(right, how=join_type)
    # 根据连接类型(join_type)生成期望的DataFrame
    if join_type == "right":
        expected = DataFrame(
            {"c": [3], "d": [4]},
            index=MultiIndex.from_tuples([(2, 1)], names=["b", "a"]),
        )
    else:
        expected = DataFrame(
            {"c": [3], "d": [4]},
            index=MultiIndex.from_tuples([(1, 2)], names=["a", "b"]),
        )
    # 使用测试框架中的断言，比较实际结果和期望结果DataFrame
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "categories, values",
    [
        # 不按字母顺序的分类数据连接测试
        (["Y", "X"], ["Y", "X", "X"]),
        ([2, 1], [2, 1, 1]),
        ([2.5, 1.5], [2.5, 1.5, 1.5]),
        (
            [Timestamp("2020-12-31"), Timestamp("2019-12-31")],
            [Timestamp("2020-12-31"), Timestamp("2019-12-31"), Timestamp("2019-12-31")],
        ),
    ],
)
def test_join_multiindex_not_alphabetical_categorical(categories, values):
    # GH#38502: GitHub issue编号，表示这段代码与该问题相关
    # 创建左侧DataFrame，包含MultiIndex，其中包括分类数据（不按字母顺序排列）
    left = DataFrame(
        {
            "first": ["A", "A"],
            "second": Categorical(categories, categories=categories),
            "value": [1, 2],
        }
    ).set_index(["first", "second"])
    # 创建右侧DataFrame，包含MultiIndex，其中包括分类数据（不按字母顺序排列）
    right = DataFrame(
        {
            "first": ["A", "A", "B"],
            "second": Categorical(values, categories=categories),
            "value": [3, 4, 5],
        }
    ).set_index(["first", "second"])
    # 执行连接操作，左侧列名后缀为"_left"，右侧列名后缀为"_right"
    result = left.join(right, lsuffix="_left", rsuffix="_right")

    # 生成期望的DataFrame，左侧列名和右侧列名带有后缀
    expected = DataFrame(
        {
            "first": ["A", "A"],
            "second": Categorical(categories, categories=categories),
            "value_left": [1, 2],
            "value_right": [3, 4],
        }
    ).set_index(["first", "second"])
    # 使用测试框架中的断言，比较实际结果和期望结果DataFrame
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "left_empty, how, exp",
    [
        # 空DataFrame连接测试，根据连接方式定义期望结果
        (False, "left", "left"),
        (False, "right", "empty"),
        (False, "inner", "empty"),
        (False, "outer", "left"),
        (False, "cross", "empty"),
        (True, "left", "empty"),
        (True, "right", "right"),
        (True, "inner", "empty"),
        (True, "outer", "right"),
        (True, "cross", "empty"),
    ],
)
def test_join_empty(left_empty, how, exp):
    # 创建左侧DataFrame，根据left_empty参数决定是否为空
    left = DataFrame({"A": [2, 1], "B": [3, 4]}, dtype="int64").set_index("A")
    # 创建右侧DataFrame，无论如何包含一行数据，以及列'A'和'C'
    right = DataFrame({"A": [1], "C": [5]}, dtype="int64").set_index("A")
    # 如果 left_empty 为真，则将 left 设置为一个空的 DataFrame
    if left_empty:
        left = left.head(0)
    else:
        # 否则，将 right 设置为一个空的 DataFrame
        right = right.head(0)

    # 对 left 和 right 进行连接操作，使用指定的连接方式（how）
    result = left.join(right, how=how)

    # 根据预期结果类型（exp）设置期望的 DataFrame
    if exp == "left":
        expected = DataFrame({"A": [2, 1], "B": [3, 4], "C": [np.nan, np.nan]})
        expected = expected.set_index("A")
    elif exp == "right":
        expected = DataFrame({"B": [np.nan], "A": [1], "C": [5]})
        expected = expected.set_index("A")
    elif exp == "empty":
        expected = DataFrame(columns=["B", "C"], dtype="int64")
        # 如果连接方式不是 "cross"，则将索引命名为 "A"
        if how != "cross":
            expected = expected.rename_axis("A")

    # 如果连接方式为 "outer"，按索引排序期望结果
    if how == "outer":
        expected = expected.sort_index()

    # 使用测试工具（tm.assert_frame_equal）比较结果和期望结果的DataFrame
    tm.assert_frame_equal(result, expected)
# 定义一个测试函数，用于测试在处理空和不可比较的列时的数据框连接操作
def test_join_empty_uncomparable_columns():
    # GH 57048
    # 创建一个空的数据框 df1
    df1 = DataFrame()
    # 创建一个只包含 "test" 列的数据框 df2
    df2 = DataFrame(columns=["test"])
    # 创建一个包含 ("bar", "baz") 列的数据框 df3
    df3 = DataFrame(columns=["foo", ("bar", "baz")])

    # 执行 df1 和 df2 的列相加操作，返回结果存储在 result 中
    result = df1 + df2
    # 创建预期的结果数据框，只包含 "test" 列
    expected = DataFrame(columns=["test"])
    # 断言两个数据框的内容是否相等
    tm.assert_frame_equal(result, expected)

    # 执行 df2 和 df3 的列相加操作，返回结果存储在 result 中
    result = df2 + df3
    # 创建预期的结果数据框，列的顺序为 ("bar", "baz"), "foo", "test"
    expected = DataFrame(columns=[("bar", "baz"), "foo", "test"])
    # 断言两个数据框的内容是否相等
    tm.assert_frame_equal(result, expected)

    # 执行 df1 和 df3 的列相加操作，返回结果存储在 result 中
    result = df1 + df3
    # 创建预期的结果数据框，列的顺序为 ("bar", "baz"), "foo"
    expected = DataFrame(columns=[("bar", "baz"), "foo"])
    # 断言两个数据框的内容是否相等
    tm.assert_frame_equal(result, expected)


# 使用 pytest 的参数化装饰器定义多个参数组合的测试用例
@pytest.mark.parametrize(
    "how, values",
    [
        ("inner", [0, 1, 2]),
        ("outer", [0, 1, 2]),
        ("left", [0, 1, 2]),
        ("right", [0, 2, 1]),
    ],
)
# 定义一个测试函数，测试多级索引和分类数据在连接操作中的输出索引类型
def test_join_multiindex_categorical_output_index_dtype(how, values):
    # GH#50906
    # 创建一个具有多级索引的数据框 df1
    df1 = DataFrame(
        {
            "a": Categorical([0, 1, 2]),
            "b": Categorical([0, 1, 2]),
            "c": [0, 1, 2],
        }
    ).set_index(["a", "b"])

    # 创建另一个具有多级索引的数据框 df2
    df2 = DataFrame(
        {
            "a": Categorical([0, 2, 1]),
            "b": Categorical([0, 2, 1]),
            "d": [0, 2, 1],
        }
    ).set_index(["a", "b"])

    # 创建预期的结果数据框，包含多级索引和分类数据
    expected = DataFrame(
        {
            "a": Categorical(values),
            "b": Categorical(values),
            "c": values,
            "d": values,
        }
    ).set_index(["a", "b"])

    # 执行数据框 df1 和 df2 的连接操作，根据指定的连接方式 how
    result = df1.join(df2, how=how)
    # 断言两个数据框的内容是否相等
    tm.assert_frame_equal(result, expected)
```