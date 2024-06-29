# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_sort_index.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试

import pandas as pd  # 导入 Pandas 库，用于数据处理
from pandas import (  # 导入 Pandas 中的多个子模块和类
    CategoricalDtype,
    CategoricalIndex,
    DataFrame,
    IntervalIndex,
    MultiIndex,
    RangeIndex,
    Series,
    Timestamp,
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块


class TestDataFrameSortIndex:
    def test_sort_index_and_reconstruction_doc_example(self):
        # 文档示例
        df = DataFrame(  # 创建一个 DataFrame 对象
            {"value": [1, 2, 3, 4]},  # 数据为字典形式，包含 "value" 列
            index=MultiIndex(  # 指定索引为多重索引 MultiIndex 对象
                levels=[["a", "b"], ["bb", "aa"]],  # 索引的多层级别
                codes=[[0, 0, 1, 1], [0, 1, 0, 1]],  # 层级对应的编码
            ),
        )
        assert df.index._is_lexsorted()  # 断言索引已经按字典序排序
        assert not df.index.is_monotonic_increasing  # 断言索引不是单调递增的

        # 对 DataFrame 进行排序
        expected = DataFrame(  # 创建预期的 DataFrame 对象
            {"value": [2, 1, 4, 3]},  # 调整后的 "value" 列数据
            index=MultiIndex(  # 使用 MultiIndex 重新定义索引
                levels=[["a", "b"], ["aa", "bb"]],  # 调整后的层级
                codes=[[0, 0, 1, 1], [0, 1, 0, 1]],  # 调整后的编码
            ),
        )
        result = df.sort_index()  # 对 DataFrame 进行排序
        assert result.index.is_monotonic_increasing  # 断言排序后索引单调递增
        tm.assert_frame_equal(result, expected)  # 使用测试工具检验结果与预期是否相等

        # 重新构建
        result = df.sort_index().copy()  # 对排序后的 DataFrame 进行复制
        result.index = result.index._sort_levels_monotonic()  # 对索引进行单调排序
        assert result.index.is_monotonic_increasing  # 断言索引单调递增
        tm.assert_frame_equal(result, expected)  # 使用测试工具检验结果与预期是否相等

    def test_sort_index_non_existent_label_multiindex(self):
        # GH#12261
        df = DataFrame(0, columns=[], index=MultiIndex.from_product([[], []]))  # 创建空的 MultiIndex DataFrame
        with tm.assert_produces_warning(None):  # 禁止警告输出
            df.loc["b", "2"] = 1  # 在不存在的标签处添加数据
            df.loc["a", "3"] = 1  # 在不存在的标签处添加数据
        result = df.sort_index().index.is_monotonic_increasing  # 对索引进行排序并检查是否单调递增
        assert result is True  # 断言结果为 True

    def test_sort_index_reorder_on_ops(self):
        # GH#15687
        df = DataFrame(  # 创建一个 DataFrame 对象
            np.random.default_rng(2).standard_normal((8, 2)),  # 使用随机数据填充
            index=MultiIndex.from_product(  # 使用 MultiIndex 生成多层级索引
                [["a", "b"], ["big", "small"], ["red", "blu"]],  # 不同层级的索引值
                names=["letter", "size", "color"],  # 每个层级的名称
            ),
            columns=["near", "far"],  # 列名称
        )
        df = df.sort_index()  # 对 DataFrame 进行排序

        def my_func(group):  # 定义一个操作函数
            group.index = ["newz", "newa"]  # 重新定义索引值
            return group

        result = df.groupby(level=["letter", "size"]).apply(my_func).sort_index()  # 对分组后的结果应用操作函数并排序
        expected = MultiIndex.from_product(  # 生成预期的 MultiIndex 结果
            [["a", "b"], ["big", "small"], ["newa", "newz"]],  # 调整后的层级值
            names=["letter", "size", None],  # 调整后的名称
        )

        tm.assert_index_equal(result.index, expected)  # 使用测试工具检验结果与预期是否相等
    # 定义一个测试方法，用于验证多重索引排序中处理 NaN 值的正确性
    def test_sort_index_nan_multiindex(self):
        # GH#14784: GitHub issue number 14784，用于跟踪此问题
        # incorrect sorting w.r.t. nans，指出此处问题是与 NaN 值相关的排序错误

        # 创建一个包含 NaN 值的多重索引对象
        tuples = [[12, 13], [np.nan, np.nan], [np.nan, 3], [1, 2]]
        mi = MultiIndex.from_tuples(tuples)

        # 创建一个 DataFrame，使用前述的多重索引 mi 和列名列表 "ABCD"，填充数据从 0 到 15
        df = DataFrame(np.arange(16).reshape(4, 4), index=mi, columns=list("ABCD"))
        
        # 创建一个 Series，使用前述的多重索引 mi，填充数据从 0 到 3
        s = Series(np.arange(4), index=mi)

        # 创建一个新的 DataFrame，包含多列数据，以日期和用户 ID 作为索引
        df2 = DataFrame(
            {
                "date": pd.DatetimeIndex(
                    [
                        "20121002",
                        "20121007",
                        "20130130",
                        "20130202",
                        "20130305",
                        "20121002",
                        "20121207",
                        "20130130",
                        "20130202",
                        "20130305",
                        "20130202",
                        "20130305",
                    ]
                ),
                "user_id": [1, 1, 1, 1, 1, 3, 3, 3, 5, 5, 5, 5],
                "whole_cost": [
                    1790,
                    np.nan,
                    280,
                    259,
                    np.nan,
                    623,
                    90,
                    312,
                    np.nan,
                    301,
                    359,
                    801,
                ],
                "cost": [12, 15, 10, 24, 39, 1, 0, np.nan, 45, 34, 1, 12],
            }
        ).set_index(["date", "user_id"])

        # 对 DataFrame df 进行排序，默认情况下 NaN 值被视为最后
        result = df.sort_index()
        expected = df.iloc[[3, 0, 2, 1], :]
        tm.assert_frame_equal(result, expected)

        # 对 DataFrame df 进行排序，指定 NaN 值排序在最后
        result = df.sort_index(na_position="last")
        expected = df.iloc[[3, 0, 2, 1], :]
        tm.assert_frame_equal(result, expected)

        # 对 DataFrame df 进行排序，指定 NaN 值排序在最前面
        result = df.sort_index(na_position="first")
        expected = df.iloc[[1, 2, 3, 0], :]
        tm.assert_frame_equal(result, expected)

        # 对 DataFrame df2 删除 NaN 值后进行排序
        result = df2.dropna().sort_index()
        expected = df2.sort_index().dropna()
        tm.assert_frame_equal(result, expected)

        # 对 Series s 进行排序，默认情况下 NaN 值被视为最后
        result = s.sort_index()
        expected = s.iloc[[3, 0, 2, 1]]
        tm.assert_series_equal(result, expected)

        # 对 Series s 进行排序，指定 NaN 值排序在最后
        result = s.sort_index(na_position="last")
        expected = s.iloc[[3, 0, 2, 1]]
        tm.assert_series_equal(result, expected)

        # 对 Series s 进行排序，指定 NaN 值排序在最前面
        result = s.sort_index(na_position="first")
        expected = s.iloc[[1, 2, 3, 0]]
        tm.assert_series_equal(result, expected)
    def test_sort_index_nan(self):
        # GH#3917
        # 测试处理含有NaN标签的DataFrame

        # 创建一个包含NaN标签的DataFrame
        df = DataFrame(
            {"A": [1, 2, np.nan, 1, 6, 8, 4], "B": [9, np.nan, 5, 2, 5, 4, 5]},
            index=[1, 2, 3, 4, 5, 6, np.nan],
        )

        # 使用'quicksort'算法对DataFrame按索引排序，NaN标签排在最后
        sorted_df = df.sort_index(kind="quicksort", ascending=True, na_position="last")
        expected = DataFrame(
            {"A": [1, 2, np.nan, 1, 6, 8, 4], "B": [9, np.nan, 5, 2, 5, 4, 5]},
            index=[1, 2, 3, 4, 5, 6, np.nan],
        )
        tm.assert_frame_equal(sorted_df, expected)

        # 使用默认的'quicksort'算法对DataFrame按索引排序，NaN标签排在最前
        sorted_df = df.sort_index(na_position="first")
        expected = DataFrame(
            {"A": [4, 1, 2, np.nan, 1, 6, 8], "B": [5, 9, np.nan, 5, 2, 5, 4]},
            index=[np.nan, 1, 2, 3, 4, 5, 6],
        )
        tm.assert_frame_equal(sorted_df, expected)

        # 使用'quicksort'算法对DataFrame按索引降序排序，NaN标签排在最后
        sorted_df = df.sort_index(kind="quicksort", ascending=False)
        expected = DataFrame(
            {"A": [8, 6, 1, np.nan, 2, 1, 4], "B": [4, 5, 2, 5, np.nan, 9, 5]},
            index=[6, 5, 4, 3, 2, 1, np.nan],
        )
        tm.assert_frame_equal(sorted_df, expected)

        # 使用'quicksort'算法对DataFrame按索引降序排序，NaN标签排在最前
        sorted_df = df.sort_index(
            kind="quicksort", ascending=False, na_position="first"
        )
        expected = DataFrame(
            {"A": [4, 8, 6, 1, np.nan, 2, 1], "B": [5, 4, 5, 2, 5, np.nan, 9]},
            index=[np.nan, 6, 5, 4, 3, 2, 1],
        )
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_index_multi_index(self):
        # GH#25775, 测试多级索引排序功能

        # 创建一个包含多级索引的DataFrame，并按指定的级别排序
        df = DataFrame(
            {"a": [3, 1, 2], "b": [0, 0, 0], "c": [0, 1, 2], "d": list("abc")}
        )
        result = df.set_index(list("abc")).sort_index(level=list("ba"))

        expected = DataFrame(
            {"a": [1, 2, 3], "b": [0, 0, 0], "c": [1, 2, 0], "d": list("bca")}
        )
        expected = expected.set_index(list("abc"))

        tm.assert_frame_equal(result, expected)
    def test_sort_index_inplace(self):
        frame = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=[1, 2, 3, 4],
            columns=["A", "B", "C", "D"],
        )

        # axis=0，按行排序
        unordered = frame.loc[[3, 2, 4, 1]]
        a_values = unordered["A"]
        df = unordered.copy()
        return_value = df.sort_index(inplace=True)  # 在原地排序DataFrame
        assert return_value is None
        expected = frame  # 预期结果是原始DataFrame
        tm.assert_frame_equal(df, expected)
        # GH 44153 related
        # Used to be a_id != id(df["A"]), but flaky in the CI
        assert a_values is not df["A"]  # 检查未排序列"A"的身份是否与排序后的不同

        df = unordered.copy()
        return_value = df.sort_index(ascending=False, inplace=True)  # 在原地降序排序
        assert return_value is None
        expected = frame[::-1]  # 预期结果是原始DataFrame的逆序
        tm.assert_frame_equal(df, expected)

        # axis=1，按列排序
        unordered = frame.loc[:, ["D", "B", "C", "A"]]
        df = unordered.copy()
        return_value = df.sort_index(axis=1, inplace=True)  # 在原地按列排序
        assert return_value is None
        expected = frame  # 预期结果是原始DataFrame
        tm.assert_frame_equal(df, expected)

        df = unordered.copy()
        return_value = df.sort_index(axis=1, ascending=False, inplace=True)  # 在原地按列降序排序
        assert return_value is None
        expected = frame.iloc[:, ::-1]  # 预期结果是原始DataFrame列的逆序
        tm.assert_frame_equal(df, expected)
    def test_sort_index_level(self):
        # 创建一个包含多级索引的 DataFrame
        mi = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list("ABC"))
        df = DataFrame([[1, 2], [3, 4]], mi)

        # 按照指定级别 "A" 排序，不排序剩余级别
        result = df.sort_index(level="A", sort_remaining=False)
        expected = df
        tm.assert_frame_equal(result, expected)

        # 按照多个级别 ["A", "B"] 排序，不排序剩余级别
        result = df.sort_index(level=["A", "B"], sort_remaining=False)
        expected = df
        tm.assert_frame_equal(result, expected)

        # 当首个索引级别 "C" 在排序时抛出错误 (GH#26053)
        result = df.sort_index(level=["C", "B", "A"])
        expected = df.iloc[[1, 0]]
        tm.assert_frame_equal(result, expected)

        # 按照级别 ["B", "C", "A"] 排序
        result = df.sort_index(level=["B", "C", "A"])
        expected = df.iloc[[1, 0]]
        tm.assert_frame_equal(result, expected)

        # 按照级别 ["C", "A"] 排序
        result = df.sort_index(level=["C", "A"])
        expected = df.iloc[[1, 0]]
        tm.assert_frame_equal(result, expected)

    def test_sort_index_categorical_index(self):
        # 创建一个带有分类索引的 DataFrame
        df = DataFrame(
            {
                "A": np.arange(6, dtype="int64"),
                "B": Series(list("aabbca")).astype(CategoricalDtype(list("cab"))),
            }
        ).set_index("B")

        # 按照索引排序
        result = df.sort_index()
        expected = df.iloc[[4, 0, 1, 5, 2, 3]]
        tm.assert_frame_equal(result, expected)

        # 按照索引排序，降序
        result = df.sort_index(ascending=False)
        expected = df.iloc[[2, 3, 0, 1, 5, 4]]
        tm.assert_frame_equal(result, expected)

    def test_sort_index(self):
        # GH#13496

        # 创建一个简单的 DataFrame
        frame = DataFrame(
            np.arange(16).reshape(4, 4),
            index=[1, 2, 3, 4],
            columns=["A", "B", "C", "D"],
        )

        # axis=0 : 按照行索引标签排序
        unordered = frame.loc[[3, 2, 4, 1]]
        result = unordered.sort_index(axis=0)
        expected = frame
        tm.assert_frame_equal(result, expected)

        # axis=0 : 按照行索引标签排序，降序
        result = unordered.sort_index(ascending=False)
        expected = frame[::-1]
        tm.assert_frame_equal(result, expected)

        # axis=1 : 按照列名排序
        unordered = frame.iloc[:, [2, 1, 3, 0]]
        result = unordered.sort_index(axis=1)
        tm.assert_frame_equal(result, frame)

        # axis=1 : 按照列名排序，降序
        result = unordered.sort_index(axis=1, ascending=False)
        expected = frame.iloc[:, ::-1]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("level", ["A", 0])  # GH#21052
    def test_sort_index_multiindex(self, level):
        # GH#13496
        # 根据 GitHub 上的 issue 编号进行测试

        # sort rows by specified level of multi-index
        # 使用指定的多级索引级别对行进行排序
        mi = MultiIndex.from_tuples(
            [[2, 1, 3], [2, 1, 2], [1, 1, 1]], names=list("ABC")
        )
        df = DataFrame([[1, 2], [3, 4], [5, 6]], index=mi)

        expected_mi = MultiIndex.from_tuples(
            [[1, 1, 1], [2, 1, 2], [2, 1, 3]], names=list("ABC")
        )
        expected = DataFrame([[5, 6], [3, 4], [1, 2]], index=expected_mi)
        result = df.sort_index(level=level)
        tm.assert_frame_equal(result, expected)

        # sort_remaining=False
        # 当 sort_remaining=False 时

        expected_mi = MultiIndex.from_tuples(
            [[1, 1, 1], [2, 1, 3], [2, 1, 2]], names=list("ABC")
        )
        expected = DataFrame([[5, 6], [1, 2], [3, 4]], index=expected_mi)
        result = df.sort_index(level=level, sort_remaining=False)
        tm.assert_frame_equal(result, expected)

    def test_sort_index_intervalindex(self):
        # this is a de-facto sort via unstack
        # confirming that we sort in the order of the bins
        # 这是通过 unstack 实现的排序，确认我们按照区间的顺序进行排序
        y = Series(np.random.default_rng(2).standard_normal(100))
        x1 = Series(np.sign(np.random.default_rng(2).standard_normal(100)))
        x2 = pd.cut(
            Series(np.random.default_rng(2).standard_normal(100)),
            bins=[-3, -0.5, 0, 0.5, 3],
        )
        model = pd.concat([y, x1, x2], axis=1, keys=["Y", "X1", "X2"])

        result = model.groupby(["X1", "X2"], observed=True).mean().unstack()
        expected = IntervalIndex.from_tuples(
            [(-3.0, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 3.0)], closed="right"
        )
        result = result.columns.levels[1].categories
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize(
        "original_dict, sorted_dict, ascending, ignore_index, output_index",
        [
            ({"A": [1, 2, 3]}, {"A": [2, 3, 1]}, False, True, [0, 1, 2]),
            ({"A": [1, 2, 3]}, {"A": [1, 3, 2]}, True, True, [0, 1, 2]),
            ({"A": [1, 2, 3]}, {"A": [2, 3, 1]}, False, False, [5, 3, 2]),
            ({"A": [1, 2, 3]}, {"A": [1, 3, 2]}, True, False, [2, 3, 5]),
        ],
    )
    def test_sort_index_ignore_index(
        self, inplace, original_dict, sorted_dict, ascending, ignore_index, output_index
    ):
        # GH 30114
        # 根据 GitHub 上的 issue 编号进行测试

        original_index = [2, 5, 3]
        df = DataFrame(original_dict, index=original_index)
        expected_df = DataFrame(sorted_dict, index=output_index)
        kwargs = {
            "ascending": ascending,
            "ignore_index": ignore_index,
            "inplace": inplace,
        }

        if inplace:
            # 如果 inplace=True，则复制数据框并对其进行排序
            result_df = df.copy()
            result_df.sort_index(**kwargs)
        else:
            # 如果 inplace=False，则对原始数据框进行排序
            result_df = df.sort_index(**kwargs)

        tm.assert_frame_equal(result_df, expected_df)
        tm.assert_frame_equal(df, DataFrame(original_dict, index=original_index))
    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize("ignore_index", [True, False])
    def test_respect_ignore_index(self, inplace, ignore_index):
        # GH 43591
        # 创建一个包含指定列和索引的 DataFrame 对象
        df = DataFrame({"a": [1, 2, 3]}, index=RangeIndex(4, -1, -2))
        # 调用 sort_index 方法对 DataFrame 进行排序，并返回结果
        result = df.sort_index(
            ascending=False, ignore_index=ignore_index, inplace=inplace
        )

        if inplace:
            # 如果 inplace 为 True，则更新 result 为原始 DataFrame
            result = df
        if ignore_index:
            # 根据 ignore_index 的值创建预期的 DataFrame 对象
            expected = DataFrame({"a": [1, 2, 3]})
        else:
            # 根据 ignore_index 的值和特定的索引创建预期的 DataFrame 对象
            expected = DataFrame({"a": [1, 2, 3]}, index=RangeIndex(4, -1, -2))

        # 使用 assert_frame_equal 方法比较实际结果和预期结果的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize(
        "original_dict, sorted_dict, ascending, ignore_index, output_index",
        [
            (
                {"M1": [1, 2], "M2": [3, 4]},
                {"M1": [1, 2], "M2": [3, 4]},
                True,
                True,
                [0, 1],
            ),
            (
                {"M1": [1, 2], "M2": [3, 4]},
                {"M1": [2, 1], "M2": [4, 3]},
                False,
                True,
                [0, 1],
            ),
            (
                {"M1": [1, 2], "M2": [3, 4]},
                {"M1": [1, 2], "M2": [3, 4]},
                True,
                False,
                MultiIndex.from_tuples([(2, 1), (3, 4)], names=list("AB")),
            ),
            (
                {"M1": [1, 2], "M2": [3, 4]},
                {"M1": [2, 1], "M2": [4, 3]},
                False,
                False,
                MultiIndex.from_tuples([(3, 4), (2, 1)], names=list("AB")),
            ),
        ],
    )
    def test_sort_index_ignore_index_multi_index(
        self, inplace, original_dict, sorted_dict, ascending, ignore_index, output_index
    ):
        # GH 30114, 这是为了测试在索引的 MultiIndex 上应用 ignore_index
        # 使用给定的字典和 MultiIndex 创建 DataFrame 对象
        mi = MultiIndex.from_tuples([(2, 1), (3, 4)], names=list("AB"))
        df = DataFrame(original_dict, index=mi)
        # 根据预期的字典和输出的索引创建预期的 DataFrame 对象
        expected_df = DataFrame(sorted_dict, index=output_index)

        kwargs = {
            "ascending": ascending,
            "ignore_index": ignore_index,
            "inplace": inplace,
        }

        if inplace:
            # 如果 inplace 为 True，则对原始 DataFrame 执行排序并更新 result_df
            result_df = df.copy()
            result_df.sort_index(**kwargs)
        else:
            # 如果 inplace 为 False，则创建一个新的已排序 DataFrame
            result_df = df.sort_index(**kwargs)

        # 使用 assert_frame_equal 方法比较实际结果和预期结果的 DataFrame 是否相等
        tm.assert_frame_equal(result_df, expected_df)
        # 再次使用 assert_frame_equal 方法验证原始 DataFrame 是否保持不变
        tm.assert_frame_equal(df, DataFrame(original_dict, index=mi))
    # 定义一个测试方法，用于验证多级分类索引的排序功能
    def test_sort_index_categorical_multiindex(self):
        # GH#15058: 关联 GitHub 问题编号，指出测试的背景或目的
        # 创建一个 DataFrame 对象，包含列 'a', 'l1', 'l2'，其中 'l1' 是有序分类数据
        df = DataFrame(
            {
                "a": range(6),
                "l1": pd.Categorical(
                    ["a", "a", "b", "b", "c", "c"],
                    categories=["c", "a", "b"],
                    ordered=True,
                ),
                "l2": [0, 1, 0, 1, 0, 1],
            }
        )
        # 对 DataFrame 进行设置多级索引并排序
        result = df.set_index(["l1", "l2"]).sort_index()
        # 创建期望的 DataFrame，包含预期的数据和索引结构
        expected = DataFrame(
            [4, 5, 0, 1, 2, 3],
            columns=["a"],
            index=MultiIndex(
                levels=[
                    CategoricalIndex(
                        ["c", "a", "b"],
                        categories=["c", "a", "b"],
                        ordered=True,
                        name="l1",
                        dtype="category",
                    ),
                    [0, 1],
                ],
                codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]],
                names=["l1", "l2"],
            ),
        )
        # 使用测试工具比较实际结果和期望结果是否相等
        tm.assert_frame_equal(result, expected)
    def test_sort_index_and_reconstruction(self):
        # GH#15622
        # lexsortedness should be identical
        # across MultiIndex construction methods

        # 创建一个简单的 DataFrame，指定行索引为列表 "ab"
        df = DataFrame([[1, 1], [2, 2]], index=list("ab"))
        
        # 创建预期的 DataFrame，使用 MultiIndex.from_tuples 方法设置索引
        expected = DataFrame(
            [[1, 1], [2, 2], [1, 1], [2, 2]],
            index=MultiIndex.from_tuples(
                [(0.5, "a"), (0.5, "b"), (0.8, "a"), (0.8, "b")]
            ),
        )
        
        # 断言预期的索引是 lexsorted
        assert expected.index._is_lexsorted()

        # 创建一个结果 DataFrame，使用 MultiIndex.from_product 方法设置索引
        result = DataFrame(
            [[1, 1], [2, 2], [1, 1], [2, 2]],
            index=MultiIndex.from_product([[0.5, 0.8], list("ab")]),
        )
        
        # 对结果 DataFrame 进行排序
        result = result.sort_index()
        
        # 断言结果 DataFrame 的索引是单调递增的
        assert result.index.is_monotonic_increasing

        # 使用 tm.assert_frame_equal 断言结果与预期的 DataFrame 相等
        tm.assert_frame_equal(result, expected)

        # 创建另一个结果 DataFrame，使用 MultiIndex 方法设置索引
        result = DataFrame(
            [[1, 1], [2, 2], [1, 1], [2, 2]],
            index=MultiIndex(
                levels=[[0.5, 0.8], ["a", "b"]], codes=[[0, 0, 1, 1], [0, 1, 0, 1]]
            ),
        )
        
        # 对结果 DataFrame 进行排序
        result = result.sort_index()
        
        # 断言结果 DataFrame 的索引是 lexsorted
        assert result.index._is_lexsorted()

        # 使用 tm.assert_frame_equal 断言结果与预期的 DataFrame 相等
        tm.assert_frame_equal(result, expected)

        # 连接 df 和 df 的副本，指定 keys 为 0.8 和 0.5
        concatted = pd.concat([df, df], keys=[0.8, 0.5])
        
        # 对连接后的 DataFrame 进行排序
        result = concatted.sort_index()

        # 断言结果 DataFrame 的索引是单调递增的
        assert result.index.is_monotonic_increasing

        # 使用 tm.assert_frame_equal 断言结果与预期的 DataFrame 相等

        # GH#14015
        # 创建一个 DataFrame，指定列为 MultiIndex，其中包含日期字符串
        df = DataFrame(
            [[1, 2], [6, 7]],
            columns=MultiIndex.from_tuples(
                [(0, "20160811 12:00:00"), (0, "20160809 12:00:00")],
                names=["l1", "Date"],
            ),
        )

        # 将列索引的日期字符串转换为日期时间对象，并设置为新的列索引
        df.columns = df.columns.set_levels(
            pd.to_datetime(df.columns.levels[1]), level=1
        )
        
        # 断言列索引不是单调递增的
        assert not df.columns.is_monotonic_increasing
        
        # 对 DataFrame 按列索引排序
        result = df.sort_index(axis=1)
        
        # 断言排序后的列索引是单调递增的
        assert result.columns.is_monotonic_increasing
        
        # 按列索引的第二级别进行排序
        result = df.sort_index(axis=1, level=1)
        
        # 断言排序后的列索引是单调递增的
        assert result.columns.is_monotonic_increasing

    # TODO: better name, de-duplicate with test_sort_index_level above
    def test_sort_index_level2(self, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data

        # 复制 DataFrame，并重新设置行索引为整数序列
        df = frame.copy()
        df.index = np.arange(len(df))

        # axis=1

        # 对 Series 按第一级别的索引排序
        a_sorted = frame["A"].sort_index(level=0)

        # 断言排序后的索引名称与原始 DataFrame 的相同
        assert a_sorted.index.names == frame.index.names

        # 对原始 DataFrame 按第一级别的索引排序（就地排序）
        rs = frame.copy()
        return_value = rs.sort_index(level=0, inplace=True)
        
        # 断言 inplace 操作返回 None
        assert return_value is None
        
        # 使用 tm.assert_frame_equal 断言就地排序后的 DataFrame 与原始 DataFrame 相等
        tm.assert_frame_equal(rs, frame.sort_index(level=0))
    # 定义测试方法，用于测试在大基数情况下的多级索引排序
    def test_sort_index_level_large_cardinality(self):
        # 创建一个包含三个相同数组的多级索引，每个数组包含从0到3999的整数
        index = MultiIndex.from_arrays([np.arange(4000)] * 3)
        # 创建一个包含随机标准正态分布整数数据的 DataFrame，使用索引作为行索引
        df = DataFrame(
            np.random.default_rng(2).standard_normal(4000).astype("int64"), index=index
        )

        # 对 DataFrame 根据第一级索引排序
        result = df.sort_index(level=0)
        # 断言排序后的索引的深度为3
        assert result.index._lexsort_depth == 3

        # 重新创建相同结构的索引，但数据类型为 int32
        index = MultiIndex.from_arrays([np.arange(4000)] * 3)
        df = DataFrame(
            np.random.default_rng(2).standard_normal(4000).astype("int32"), index=index
        )

        # 对 DataFrame 根据第一级索引排序
        result = df.sort_index(level=0)
        # 断言排序后的列数据类型与原始 DataFrame 的数据类型相同
        assert (result.dtypes.values == df.dtypes.values).all()
        # 再次断言排序后的索引的深度为3
        assert result.index._lexsort_depth == 3

    # 定义测试方法，用于测试按名称对多级索引进行排序
    def test_sort_index_level_by_name(self, multiindex_dataframe_random_data):
        # 使用预设的多级索引随机数据 DataFrame
        frame = multiindex_dataframe_random_data

        # 将索引的层级名称设置为 ["first", "second"]
        frame.index.names = ["first", "second"]
        # 对 DataFrame 根据第二级索引排序
        result = frame.sort_index(level="second")
        # 期望的排序结果，根据第二级索引排序
        expected = frame.sort_index(level=1)
        # 断言实际排序后的 DataFrame 与期望的排序结果一致
        tm.assert_frame_equal(result, expected)

    # 定义测试方法，用于测试混合操作下的多级索引排序
    def test_sort_index_level_mixed(self, multiindex_dataframe_random_data):
        # 使用预设的多级索引随机数据 DataFrame
        frame = multiindex_dataframe_random_data

        # 在排序前先记录按第二级索引排序后的 DataFrame
        sorted_before = frame.sort_index(level=1)

        # 复制 DataFrame
        df = frame.copy()
        # 添加一列名为 "foo" 的数据，值为 "bar"
        df["foo"] = "bar"
        # 按第二级索引排序后的 DataFrame
        sorted_after = df.sort_index(level=1)
        # 断言排序前后的 DataFrame 相等，但不包括新添加的 "foo" 列
        tm.assert_frame_equal(sorted_before, sorted_after.drop(["foo"], axis=1))

        # 对 DataFrame 进行转置
        dft = frame.T
        # 按第二级索引在列方向排序
        sorted_before = dft.sort_index(level=1, axis=1)
        # 向转置后的 DataFrame 添加一个新的 ("foo", "three") 列，值为 "bar"

        dft["foo", "three"] = "bar"

        # 按第二级索引在列方向排序后的 DataFrame
        sorted_after = dft.sort_index(level=1, axis=1)
        # 断言排序前后的 DataFrame 相等，但不包括新添加的 ("foo", "three") 列
        tm.assert_frame_equal(
            sorted_before.drop([("foo", "three")], axis=1),
            sorted_after.drop([("foo", "three")], axis=1),
        )

    # 定义测试方法，用于测试在保留层级名称的情况下对多级索引进行排序
    def test_sort_index_preserve_levels(self, multiindex_dataframe_random_data):
        # 使用预设的多级索引随机数据 DataFrame
        frame = multiindex_dataframe_random_data

        # 对 DataFrame 进行排序，保留层级名称
        result = frame.sort_index()
        # 断言排序后的索引层级名称与原始 DataFrame 的索引层级名称相同
        assert result.index.names == frame.index.names

    # 使用 pytest 的参数化功能，定义测试方法和额外参数的组合
    @pytest.mark.parametrize(
        "gen,extra",
        [
            ([1.0, 3.0, 2.0, 5.0], 4.0),
            ([1, 3, 2, 5], 4),
            (
                [
                    Timestamp("20130101"),
                    Timestamp("20130103"),
                    Timestamp("20130102"),
                    Timestamp("20130105"),
                ],
                Timestamp("20130104"),
            ),
            (["1one", "3one", "2one", "5one"], "4one"),
        ],
    )
    # 定义一个测试方法，用于测试多级索引排序的表现，标识符为 8017
    def test_sort_index_multilevel_repr_8017(self, gen, extra):
        # 生成一个随机数组，形状为 (3, 4)
        data = np.random.default_rng(2).standard_normal((3, 4))

        # 创建一个多级索引的列
        columns = MultiIndex.from_tuples([("red", i) for i in gen])
        
        # 使用指定的数据、索引和列创建 DataFrame
        df = DataFrame(data, index=list("def"), columns=columns)
        
        # 创建第二个 DataFrame，将其与第一个 DataFrame 连接起来
        df2 = pd.concat(
            [
                df,
                DataFrame(
                    "world",
                    index=list("def"),
                    columns=MultiIndex.from_tuples([("red", extra)]),
                ),
            ],
            axis=1,
        )

        # 检查 DataFrame 的字符串表示是否正确
        assert str(df2).splitlines()[0].split() == ["red"]

        # GH 8017
        # 在添加列之后，排序失败的问题

        # 构造单一数据类型，然后进行排序
        result = df.copy().sort_index(axis=1)
        expected = df.iloc[:, [0, 2, 1, 3]]
        tm.assert_frame_equal(result, expected)

        # 对 df2 进行排序，并与预期结果进行比较
        result = df2.sort_index(axis=1)
        expected = df2.iloc[:, [0, 2, 1, 4, 3]]
        tm.assert_frame_equal(result, expected)

        # 对 df 进行复制并设置元素，然后进行排序
        result = df.copy()
        result[("red", extra)] = "world"

        # 对结果进行排序，并与预期结果进行比较
        result = result.sort_index(axis=1)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "categories",
        [
            pytest.param(["a", "b", "c"], id="str"),
            pytest.param(
                [pd.Interval(0, 1), pd.Interval(1, 2), pd.Interval(2, 3)],
                id="pd.Interval",
            ),
        ],
    )
    # 测试带有分类数据的索引排序，标识符为 GH#23452
    def test_sort_index_with_categories(self, categories):
        # 创建一个 DataFrame，使用分类索引
        df = DataFrame(
            {"foo": range(len(categories))},
            index=CategoricalIndex(
                data=categories, categories=categories, ordered=True
            ),
        )
        # 重新排列分类索引的类别，并进行排序
        df.index = df.index.reorder_categories(df.index.categories[::-1])
        
        # 对 DataFrame 进行排序，并与预期结果进行比较
        result = df.sort_index()
        expected = DataFrame(
            {"foo": reversed(range(len(categories)))},
            index=CategoricalIndex(
                data=categories[::-1], categories=categories[::-1], ordered=True
            ),
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "ascending",
        [
            None,
            [True, None],
            [False, "True"],
        ],
    )
    # 测试在 ascending 参数为非预期值时是否会引发 ValueError，标识符为 GH 39434
    def test_sort_index_ascending_bad_value_raises(self, ascending):
        # 创建一个包含 64 个元素的 DataFrame
        df = DataFrame(np.arange(64))
        length = len(df.index)
        
        # 修改 DataFrame 的索引
        df.index = [(i - length / 2) % length for i in range(length)]
        
        # 匹配错误信息，检查是否引发 ValueError
        match = 'For argument "ascending" expected type bool'
        with pytest.raises(ValueError, match=match):
            df.sort_index(axis=0, ascending=ascending, na_position="first")

    @pytest.mark.parametrize(
        "ascending",
        [(True, False), [True, False]],
    )
    # 定义一个测试方法，用于测试按索引排序的功能
    def test_sort_index_ascending_tuple(self, ascending):
        # 创建一个 DataFrame 对象 df，包含一列名为 "legs" 的数据
        df = DataFrame(
            {
                "legs": [4, 2, 4, 2, 2],
            },
            # 使用 MultiIndex.from_tuples 方法创建多级索引
            index=MultiIndex.from_tuples(
                [
                    ("mammal", "dog"),
                    ("bird", "duck"),
                    ("mammal", "horse"),
                    ("bird", "penguin"),
                    ("mammal", "kangaroo"),
                ],
                # 指定多级索引的名称为 ["class", "animal"]
                names=["class", "animal"],
            ),
        )

        # 调用 DataFrame 的 sort_index 方法，按照指定的索引级别排序
        # 参数 level=(0, 1) 表示按第一级和第二级索引排序
        # 参数 ascending=ascending 用于指定升序或降序排序，其值由外部传入
        result = df.sort_index(level=(0, 1), ascending=ascending)

        # 创建预期的 DataFrame 对象 expected，包含一列名为 "legs" 的数据
        expected = DataFrame(
            {
                "legs": [2, 2, 2, 4, 4],
            },
            # 使用 MultiIndex.from_tuples 方法创建预期的多级索引
            index=MultiIndex.from_tuples(
                [
                    ("bird", "penguin"),
                    ("bird", "duck"),
                    ("mammal", "kangaroo"),
                    ("mammal", "horse"),
                    ("mammal", "dog"),
                ],
                # 指定多级索引的名称为 ["class", "animal"]
                names=["class", "animal"],
            ),
        )

        # 使用 tm.assert_frame_equal 方法断言测试结果是否与预期一致
        tm.assert_frame_equal(result, expected)
class TestDataFrameSortIndexKey:
    def test_sort_multi_index_key(self):
        # GH 25775, testing that sorting by index works with a multi-index.
        # 创建一个包含列"a", "b", "c", "d"的DataFrame，并将"a", "b", "c"设为索引
        df = DataFrame(
            {"a": [3, 1, 2], "b": [0, 0, 0], "c": [0, 1, 2], "d": list("abc")}
        ).set_index(list("abc"))

        # 根据多级索引列表["a", "c"]对DataFrame进行排序，排序关键字为默认
        result = df.sort_index(level=list("ac"), key=lambda x: x)

        # 期望的排序结果DataFrame
        expected = DataFrame(
            {"a": [1, 2, 3], "b": [0, 0, 0], "c": [1, 2, 0], "d": list("bca")}
        ).set_index(list("abc"))
        # 检验结果是否与期望一致
        tm.assert_frame_equal(result, expected)

        # 根据多级索引列表["a", "c"]对DataFrame进行排序，排序关键字为-x
        result = df.sort_index(level=list("ac"), key=lambda x: -x)
        # 期望的排序结果DataFrame
        expected = DataFrame(
            {"a": [3, 2, 1], "b": [0, 0, 0], "c": [0, 2, 1], "d": list("acb")}
        ).set_index(list("abc"))
        # 检验结果是否与期望一致
        tm.assert_frame_equal(result, expected)

    def test_sort_index_key(self):  # issue 27237
        # 创建一个包含6个元素的DataFrame，使用列表"aaBBca"作为索引
        df = DataFrame(np.arange(6, dtype="int64"), index=list("aaBBca"))

        # 根据索引默认排序
        result = df.sort_index()
        # 期望的排序结果DataFrame
        expected = df.iloc[[2, 3, 0, 1, 5, 4]]
        # 检验结果是否与期望一致
        tm.assert_frame_equal(result, expected)

        # 根据索引的小写形式进行排序
        result = df.sort_index(key=lambda x: x.str.lower())
        # 期望的排序结果DataFrame
        expected = df.iloc[[0, 1, 5, 2, 3, 4]]
        # 检验结果是否与期望一致
        tm.assert_frame_equal(result, expected)

        # 根据索引的小写形式进行排序，降序排列
        result = df.sort_index(key=lambda x: x.str.lower(), ascending=False)
        # 期望的排序结果DataFrame
        expected = df.iloc[[4, 2, 3, 0, 1, 5]]
        # 检验结果是否与期望一致
        tm.assert_frame_equal(result, expected)

    def test_sort_index_key_int(self):
        # 创建一个包含6个元素的DataFrame，使用0到5作为索引
        df = DataFrame(np.arange(6, dtype="int64"), index=np.arange(6, dtype="int64"))

        # 根据索引默认排序
        result = df.sort_index()
        # 检验结果是否与原始DataFrame一致
        tm.assert_frame_equal(result, df)

        # 根据索引-x进行排序
        result = df.sort_index(key=lambda x: -x)
        # 根据降序索引排序
        expected = df.sort_index(ascending=False)
        # 检验结果是否与期望一致
        tm.assert_frame_equal(result, expected)

        # 根据索引2*x进行排序，结果应与原始DataFrame一致
        result = df.sort_index(key=lambda x: 2 * x)
        tm.assert_frame_equal(result, df)

    def test_sort_multi_index_key_str(self):
        # GH 25775, testing that sorting by index works with a multi-index.
        # 创建一个包含列"a", "b", "c", "d"的DataFrame，并将"a", "b", "c"设为索引
        df = DataFrame(
            {"a": ["B", "a", "C"], "b": [0, 1, 0], "c": list("abc"), "d": [0, 1, 2]}
        ).set_index(list("abc"))

        # 根据索引级别"a"进行排序，排序关键字为小写形式
        result = df.sort_index(level="a", key=lambda x: x.str.lower())

        # 期望的排序结果DataFrame
        expected = DataFrame(
            {"a": ["a", "B", "C"], "b": [1, 0, 0], "c": list("bac"), "d": [1, 0, 2]}
        ).set_index(list("abc"))
        # 检验结果是否与期望一致
        tm.assert_frame_equal(result, expected)

        # 根据索引级别列表["a", "c"]进行排序，排序关键字根据列名选择是否小写形式
        result = df.sort_index(
            level=list("abc"),  # 可以引用名称
            key=lambda x: x.str.lower() if x.name in ["a", "c"] else -x,
        )

        # 期望的排序结果DataFrame
        expected = DataFrame(
            {"a": ["a", "B", "C"], "b": [1, 0, 0], "c": list("bac"), "d": [1, 0, 2]}
        ).set_index(list("abc"))
        # 检验结果是否与期望一致
        tm.assert_frame_equal(result, expected)

    def test_changes_length_raises(self):
        # 创建一个包含列"A"的DataFrame
        df = DataFrame({"A": [1, 2, 3]})
        # 使用pytest断言，测试排序时尝试更改DataFrame形状是否会引发ValueError异常
        with pytest.raises(ValueError, match="change the shape"):
            df.sort_index(key=lambda x: x[:1])
    def test_sort_index_multiindex_sparse_column(self):
        # GH 29735, testing that sort_index on a multiindexed frame with sparse
        # columns fills with 0.
        expected = DataFrame(
            {
                i: pd.array([0.0, 0.0, 0.0, 0.0], dtype=pd.SparseDtype("float64", 0.0))
                for i in range(4)
            },
            index=MultiIndex.from_product([[1, 2], [1, 2]]),
        )

        result = expected.sort_index(level=0)

        # Assert that the sorted result matches the expected DataFrame
        tm.assert_frame_equal(result, expected)

    def test_sort_index_na_position(self):
        # GH#51612
        # Create a DataFrame with two rows, one of which has pd.NA as an index value
        df = DataFrame([1, 2], index=MultiIndex.from_tuples([(1, 1), (1, pd.NA)]))
        expected = df.copy()
        # Sort the DataFrame by the specified levels, placing NaNs at the end
        result = df.sort_index(level=[0, 1], na_position="last")
        # Assert that the sorted result matches the expected DataFrame
        tm.assert_frame_equal(result, expected)

    def test_sort_index_multiindex_sort_remaining(self, ascending):
        # GH #24247
        # Create a DataFrame with two columns and a multiindex
        df = DataFrame(
            {"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]},
            index=MultiIndex.from_tuples(
                [("a", "x"), ("a", "y"), ("b", "x"), ("b", "y"), ("c", "x")]
            ),
        )

        # Sort the DataFrame by the second level of the multiindex without sorting remaining levels
        result = df.sort_index(level=1, sort_remaining=False, ascending=ascending)

        if ascending:
            # Define the expected DataFrame when sorting in ascending order
            expected = DataFrame(
                {"A": [1, 3, 5, 2, 4], "B": [10, 30, 50, 20, 40]},
                index=MultiIndex.from_tuples(
                    [("a", "x"), ("b", "x"), ("c", "x"), ("a", "y"), ("b", "y")]
                ),
            )
        else:
            # Define the expected DataFrame when sorting in descending order
            expected = DataFrame(
                {"A": [2, 4, 1, 3, 5], "B": [20, 40, 10, 30, 50]},
                index=MultiIndex.from_tuples(
                    [("a", "y"), ("b", "y"), ("a", "x"), ("b", "x"), ("c", "x")]
                ),
            )

        # Assert that the sorted result matches the expected DataFrame
        tm.assert_frame_equal(result, expected)
def test_sort_index_with_sliced_multiindex():
    # GH 55379
    # 创建一个多级索引对象 mi，包含元组列表和列名
    mi = MultiIndex.from_tuples(
        [
            ("a", "10"),
            ("a", "18"),
            ("a", "25"),
            ("b", "16"),
            ("b", "26"),
            ("a", "45"),
            ("b", "28"),
            ("a", "5"),
            ("a", "50"),
            ("a", "51"),
            ("b", "4"),
        ],
        names=["group", "str"],
    )

    # 创建 DataFrame 对象 df，包含列"x"，使用 mi 作为索引
    df = DataFrame({"x": range(len(mi))}, index=mi)
    # 对 df 进行切片和排序，并保存结果到 result
    result = df.iloc[0:6].sort_index()

    # 创建预期的 DataFrame 对象 expected，包含列"x"，使用预定义的多级索引作为索引
    expected = DataFrame(
        {"x": [0, 1, 2, 5, 3, 4]},
        index=MultiIndex.from_tuples(
            [
                ("a", "10"),
                ("a", "18"),
                ("a", "25"),
                ("a", "45"),
                ("b", "16"),
                ("b", "26"),
            ],
            names=["group", "str"],
        ),
    )
    # 使用测试框架比较 result 和 expected 的内容是否相等
    tm.assert_frame_equal(result, expected)


def test_axis_columns_ignore_index():
    # GH 56478
    # 创建一个简单的 DataFrame 对象 df，包含特定的列名
    df = DataFrame([[1, 2]], columns=["d", "c"])
    # 对 df 按列名排序，忽略索引，并保存结果到 result
    result = df.sort_index(axis="columns", ignore_index=True)
    # 创建预期的 DataFrame 对象 expected，包含重新排序后的列
    expected = DataFrame([[2, 1]])
    # 使用测试框架比较 result 和 expected 的内容是否相等
    tm.assert_frame_equal(result, expected)


def test_axis_columns_ignore_index_ascending_false():
    # GH 57293
    # 创建一个复杂的 DataFrame 对象 df，包含不同类型的列和索引
    df = DataFrame(
        {
            "b": [1.0, 3.0, np.nan],
            "a": [1, 4, 3],
            1: ["a", "b", "c"],
            "e": [3, 1, 4],
            "d": [1, 2, 8],
        }
    ).set_index(["b", "a", 1])
    # 对 df 按列名排序，忽略索引，并且按降序排序，保存结果到 result
    result = df.sort_index(axis="columns", ignore_index=True, ascending=False)
    # 创建预期的 DataFrame 对象 expected，包含与 df 相同的内容，但列名为 RangeIndex
    expected = df.copy()
    expected.columns = RangeIndex(2)
    # 使用测试框架比较 result 和 expected 的内容是否相等
    tm.assert_frame_equal(result, expected)


def test_sort_index_stable_sort():
    # GH 57151
    # 创建一个 DataFrame 对象 df，包含时间戳和相关数值
    df = DataFrame(
        data=[
            (Timestamp("2024-01-30 13:00:00"), 13.0),
            (Timestamp("2024-01-30 13:00:00"), 13.1),
            (Timestamp("2024-01-30 12:00:00"), 12.0),
            (Timestamp("2024-01-30 12:00:00"), 12.1),
        ],
        columns=["dt", "value"],
    ).set_index(["dt"])
    # 对 df 按指定的级别进行稳定排序，保存结果到 result
    result = df.sort_index(level="dt", kind="stable")
    # 创建预期的 DataFrame 对象 expected，包含与 df 相同的内容，按稳定排序后的结果
    expected = DataFrame(
        data=[
            (Timestamp("2024-01-30 12:00:00"), 12.0),
            (Timestamp("2024-01-30 12:00:00"), 12.1),
            (Timestamp("2024-01-30 13:00:00"), 13.0),
            (Timestamp("2024-01-30 13:00:00"), 13.1),
        ],
        columns=["dt", "value"],
    ).set_index(["dt"])
    # 使用测试框架比较 result 和 expected 的内容是否相等
    tm.assert_frame_equal(result, expected)
```