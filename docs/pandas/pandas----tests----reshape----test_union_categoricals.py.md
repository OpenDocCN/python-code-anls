# `D:\src\scipysrc\pandas\pandas\tests\reshape\test_union_categoricals.py`

```
import numpy as np  # 导入 NumPy 库，用于支持数组操作
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from pandas.core.dtypes.concat import union_categoricals  # 从 pandas 库中导入 union_categoricals 函数

import pandas as pd  # 导入 pandas 库，并使用 pd 别名
from pandas import (  # 从 pandas 库中导入多个类
    Categorical,
    CategoricalIndex,
    Series,
)
import pandas._testing as tm  # 导入 pandas 内部测试模块


class TestUnionCategoricals:
    @pytest.mark.parametrize(  # 使用 pytest.mark.parametrize 装饰器来参数化测试
        "a, b, combined",  # 参数名
        [  # 参数化的测试数据
            (list("abc"), list("abd"), list("abcabd")),  # 字符串列表的联合
            ([0, 1, 2], [2, 3, 4], [0, 1, 2, 2, 3, 4]),  # 整数列表的联合
            ([0, 1.2, 2], [2, 3.4, 4], [0, 1.2, 2, 2, 3.4, 4]),  # 混合数据类型列表的联合
            (
                ["b", "b", np.nan, "a"],
                ["a", np.nan, "c"],
                ["b", "b", np.nan, "a", "a", np.nan, "c"],
            ),  # 包含 NaN 值的字符串列表的联合
            (
                pd.date_range("2014-01-01", "2014-01-05"),
                pd.date_range("2014-01-06", "2014-01-07"),
                pd.date_range("2014-01-01", "2014-01-07"),
            ),  # 日期范围的联合
            (
                pd.date_range("2014-01-01", "2014-01-05", tz="US/Central"),
                pd.date_range("2014-01-06", "2014-01-07", tz="US/Central"),
                pd.date_range("2014-01-01", "2014-01-07", tz="US/Central"),
            ),  # 包含时区信息的日期范围的联合
            (
                pd.period_range("2014-01-01", "2014-01-05"),
                pd.period_range("2014-01-06", "2014-01-07"),
                pd.period_range("2014-01-01", "2014-01-07"),
            ),  # 时间段范围的联合
        ],
    )
    @pytest.mark.parametrize("box", [Categorical, CategoricalIndex, Series])  # 参数化测试数据类型
    def test_union_categorical(self, a, b, combined, box):
        # GH 13361
        result = union_categoricals([box(Categorical(a)), box(Categorical(b))])  # 调用 union_categoricals 函数进行分类数据的联合
        expected = Categorical(combined)  # 生成预期的联合分类数据
        tm.assert_categorical_equal(result, expected)  # 使用 pandas._testing 模块的 assert_categorical_equal 函数进行比较

    def test_union_categorical_ordered_appearance(self):
        # new categories ordered by appearance
        s = Categorical(["x", "y", "z"])  # 创建分类数据 s
        s2 = Categorical(["a", "b", "c"])  # 创建分类数据 s2
        result = union_categoricals([s, s2])  # 调用 union_categoricals 函数进行分类数据的联合
        expected = Categorical(
            ["x", "y", "z", "a", "b", "c"], categories=["x", "y", "z", "a", "b", "c"]
        )  # 生成预期的联合分类数据，并指定分类的顺序
        tm.assert_categorical_equal(result, expected)  # 使用 pandas._testing 模块的 assert_categorical_equal 函数进行比较

    def test_union_categorical_ordered_true(self):
        s = Categorical([0, 1.2, 2], ordered=True)  # 创建有序分类数据 s
        s2 = Categorical([0, 1.2, 2], ordered=True)  # 创建有序分类数据 s2
        result = union_categoricals([s, s2])  # 调用 union_categoricals 函数进行分类数据的联合
        expected = Categorical([0, 1.2, 2, 0, 1.2, 2], ordered=True)  # 生成预期的有序联合分类数据
        tm.assert_categorical_equal(result, expected)  # 使用 pandas._testing 模块的 assert_categorical_equal 函数进行比较

    def test_union_categorical_match_types(self):
        # must exactly match types
        s = Categorical([0, 1.2, 2])  # 创建分类数据 s
        s2 = Categorical([2, 3, 4])  # 创建分类数据 s2
        msg = "dtype of categories must be the same"  # 定义错误消息字符串
        with pytest.raises(TypeError, match=msg):  # 检查是否会引发 TypeError 异常，并验证错误消息
            union_categoricals([s, s2])  # 调用 union_categoricals 函数进行分类数据的联合

    def test_union_categorical_empty(self):
        msg = "No Categoricals to union"  # 定义错误消息字符串
        with pytest.raises(ValueError, match=msg):  # 检查是否会引发 ValueError 异常，并验证错误消息
            union_categoricals([])  # 尝试对空列表进行分类数据的联合
    def test_union_categoricals_nan(self):
        # GH 13759
        # 调用 union_categoricals 函数，传入包含多个 Categorical 对象的列表
        res = union_categoricals(
            [Categorical([1, 2, np.nan]), Categorical([3, 2, np.nan])]
        )
        # 期望的 Categorical 对象，包含合并后的数据
        exp = Categorical([1, 2, np.nan, 3, 2, np.nan])
        # 使用 assert_categorical_equal 检查 res 和 exp 是否相等
        tm.assert_categorical_equal(res, exp)

        # 调用 union_categoricals 函数，传入包含多个 Categorical 对象的列表
        res = union_categoricals(
            [Categorical(["A", "B"]), Categorical(["B", "B", np.nan])]
        )
        # 期望的 Categorical 对象，包含合并后的数据
        exp = Categorical(["A", "B", "B", "B", np.nan])
        # 使用 assert_categorical_equal 检查 res 和 exp 是否相等
        tm.assert_categorical_equal(res, exp)

        # 创建两个包含 Timestamp 的列表
        val1 = [pd.Timestamp("2011-01-01"), pd.Timestamp("2011-03-01"), pd.NaT]
        val2 = [pd.NaT, pd.Timestamp("2011-01-01"), pd.Timestamp("2011-02-01")]

        # 调用 union_categoricals 函数，传入包含两个 Categorical 对象的列表
        res = union_categoricals([Categorical(val1), Categorical(val2)])
        # 期望的 Categorical 对象，包含合并后的数据和指定的 categories
        exp = Categorical(
            val1 + val2,
            categories=[
                pd.Timestamp("2011-01-01"),
                pd.Timestamp("2011-03-01"),
                pd.Timestamp("2011-02-01"),
            ],
        )
        # 使用 assert_categorical_equal 检查 res 和 exp 是否相等
        tm.assert_categorical_equal(res, exp)

        # 如果全部为 NaN
        # 调用 union_categoricals 函数，传入包含两个 Categorical 对象的列表
        res = union_categoricals(
            [
                Categorical(np.array([np.nan, np.nan], dtype=object)),
                Categorical(["X"], categories=pd.Index(["X"], dtype=object)),
            ]
        )
        # 期望的 Categorical 对象，包含合并后的数据
        exp = Categorical([np.nan, np.nan, "X"])
        # 使用 assert_categorical_equal 检查 res 和 exp 是否相等
        tm.assert_categorical_equal(res, exp)

        # 调用 union_categoricals 函数，传入包含两个 Categorical 对象的列表
        res = union_categoricals(
            [Categorical([np.nan, np.nan]), Categorical([np.nan, np.nan])]
        )
        # 期望的 Categorical 对象，包含合并后的数据
        exp = Categorical([np.nan, np.nan, np.nan, np.nan])
        # 使用 assert_categorical_equal 检查 res 和 exp 是否相等
        tm.assert_categorical_equal(res, exp)

    @pytest.mark.parametrize("val", [[], ["1"]])
    def test_union_categoricals_empty(self, val, request, using_infer_string):
        # GH 13759
        # 如果使用 infer_string 并且 val 为 ["1"]，则标记为预期失败
        if using_infer_string and val == ["1"]:
            request.applymarker(pytest.mark.xfail("object and strings dont match"))
        # 调用 union_categoricals 函数，传入包含两个 Categorical 对象的列表
        res = union_categoricals([Categorical([]), Categorical(val)])
        # 期望的 Categorical 对象，与输入的 val 相同
        exp = Categorical(val)
        # 使用 assert_categorical_equal 检查 res 和 exp 是否相等
        tm.assert_categorical_equal(res, exp)

    def test_union_categorical_same_category(self):
        # 检查快速路径
        # 创建两个拥有相同 categories 的 Categorical 对象
        c1 = Categorical([1, 2, 3, 4], categories=[1, 2, 3, 4])
        c2 = Categorical([3, 2, 1, np.nan], categories=[1, 2, 3, 4])
        # 调用 union_categoricals 函数，传入包含两个 Categorical 对象的列表
        res = union_categoricals([c1, c2])
        # 期望的 Categorical 对象，包含合并后的数据和指定的 categories
        exp = Categorical([1, 2, 3, 4, 3, 2, 1, np.nan], categories=[1, 2, 3, 4])
        # 使用 assert_categorical_equal 检查 res 和 exp 是否相等
        tm.assert_categorical_equal(res, exp)

    def test_union_categorical_same_category_str(self):
        # 创建两个拥有相同 categories 的 Categorical 对象
        c1 = Categorical(["z", "z", "z"], categories=["x", "y", "z"])
        c2 = Categorical(["x", "x", "x"], categories=["x", "y", "z"])
        # 调用 union_categoricals 函数，传入包含两个 Categorical 对象的列表
        res = union_categoricals([c1, c2])
        # 期望的 Categorical 对象，包含合并后的数据和指定的 categories
        exp = Categorical(["z", "z", "z", "x", "x", "x"], categories=["x", "y", "z"])
        # 使用 assert_categorical_equal 检查 res 和 exp 是否相等
        tm.assert_categorical_equal(res, exp)
    def test_union_categorical_same_categories_different_order(self):
        # 测试用例：https://github.com/pandas-dev/pandas/issues/19096
        # 创建第一个分类变量 c1，包含 ["a", "b", "c"]，指定分类为 ["a", "b", "c"]
        c1 = Categorical(["a", "b", "c"], categories=["a", "b", "c"])
        # 创建第二个分类变量 c2，包含 ["a", "b", "c"]，指定分类为 ["b", "a", "c"]
        c2 = Categorical(["a", "b", "c"], categories=["b", "a", "c"])
        # 调用 union_categoricals 函数，将 c1 和 c2 合并
        result = union_categoricals([c1, c2])
        # 创建预期的合并结果 expected，包含 ["a", "b", "c", "a", "b", "c"]，指定分类为 ["a", "b", "c"]
        expected = Categorical(
            ["a", "b", "c", "a", "b", "c"], categories=["a", "b", "c"]
        )
        # 断言 result 和 expected 是否相等
        tm.assert_categorical_equal(result, expected)

    def test_union_categoricals_ordered(self):
        # 创建第一个有序分类变量 c1，包含 [1, 2, 3]，指定为有序
        c1 = Categorical([1, 2, 3], ordered=True)
        # 创建第二个非有序分类变量 c2，包含 [1, 2, 3]
        c2 = Categorical([1, 2, 3], ordered=False)

        # 定义异常消息
        msg = "Categorical.ordered must be the same"
        # 使用 pytest 检查是否引发 TypeError 异常，异常消息必须匹配 msg
        with pytest.raises(TypeError, match=msg):
            union_categoricals([c1, c2])

        # 合并两个相同的有序分类变量 c1
        res = union_categoricals([c1, c1])
        # 创建预期的合并结果 exp，包含 [1, 2, 3, 1, 2, 3]，指定为有序
        exp = Categorical([1, 2, 3, 1, 2, 3], ordered=True)
        # 断言 res 和 exp 是否相等
        tm.assert_categorical_equal(res, exp)

        # 创建包含 NaN 的有序分类变量 c1
        c1 = Categorical([1, 2, 3, np.nan], ordered=True)
        # 创建包含部分分类的有序分类变量 c2
        c2 = Categorical([3, 2], categories=[1, 2, 3], ordered=True)

        # 合并 c1 和 c2
        res = union_categoricals([c1, c2])
        # 创建预期的合并结果 exp，包含 [1, 2, 3, np.nan, 3, 2]，指定为有序
        exp = Categorical([1, 2, 3, np.nan, 3, 2], ordered=True)
        # 断言 res 和 exp 是否相等
        tm.assert_categorical_equal(res, exp)

        # 创建第一个有序分类变量 c1，包含 [1, 2, 3]，指定为有序
        c1 = Categorical([1, 2, 3], ordered=True)
        # 创建第二个有序分类变量 c2，包含 [1, 2, 3]，指定分类为 [3, 2, 1]
        c2 = Categorical([1, 2, 3], categories=[3, 2, 1], ordered=True)

        # 定义异常消息
        msg = "to union ordered Categoricals, all categories must be the same"
        # 使用 pytest 检查是否引发 TypeError 异常，异常消息必须匹配 msg
        with pytest.raises(TypeError, match=msg):
            union_categoricals([c1, c2])
    # 定义一个测试方法，用于验证合并有序分类数据时的行为
    def test_union_categoricals_ignore_order(self):
        # GH 15219
        # 创建两个有序分类数据对象 c1 和 c2
        c1 = Categorical([1, 2, 3], ordered=True)
        c2 = Categorical([1, 2, 3], ordered=False)

        # 调用 union_categoricals 函数，忽略分类顺序
        res = union_categoricals([c1, c2], ignore_order=True)
        # 创建期望结果的有序分类数据对象
        exp = Categorical([1, 2, 3, 1, 2, 3])
        # 断言结果与期望是否相等
        tm.assert_categorical_equal(res, exp)

        # 定义错误消息
        msg = "Categorical.ordered must be the same"
        # 使用 pytest 断言引发 TypeError 异常，错误消息匹配指定的消息内容
        with pytest.raises(TypeError, match=msg):
            union_categoricals([c1, c2], ignore_order=False)

        # 测试合并两个相同的有序分类数据对象
        res = union_categoricals([c1, c1], ignore_order=True)
        exp = Categorical([1, 2, 3, 1, 2, 3])
        tm.assert_categorical_equal(res, exp)

        # 测试合并两个相同的有序分类数据对象，但不忽略顺序
        res = union_categoricals([c1, c1], ignore_order=False)
        exp = Categorical([1, 2, 3, 1, 2, 3], categories=[1, 2, 3], ordered=True)
        tm.assert_categorical_equal(res, exp)

        # 创建包含 NaN 值的有序分类数据对象 c1 和指定分类的有序分类数据对象 c2
        c1 = Categorical([1, 2, 3, np.nan], ordered=True)
        c2 = Categorical([3, 2], categories=[1, 2, 3], ordered=True)

        # 测试合并这两个分类数据对象，忽略顺序
        res = union_categoricals([c1, c2], ignore_order=True)
        exp = Categorical([1, 2, 3, np.nan, 3, 2])
        tm.assert_categorical_equal(res, exp)

        # 创建两个有序分类数据对象 c1 和 c2，其中 c2 的分类顺序不同
        c1 = Categorical([1, 2, 3], ordered=True)
        c2 = Categorical([1, 2, 3], categories=[3, 2, 1], ordered=True)

        # 测试合并这两个分类数据对象，忽略顺序
        res = union_categoricals([c1, c2], ignore_order=True)
        exp = Categorical([1, 2, 3, 1, 2, 3])
        tm.assert_categorical_equal(res, exp)

        # 测试合并这两个分类数据对象，忽略顺序并排序分类
        res = union_categoricals([c2, c1], ignore_order=True, sort_categories=True)
        exp = Categorical([1, 2, 3, 1, 2, 3], categories=[1, 2, 3])
        tm.assert_categorical_equal(res, exp)

        # 创建两个有序分类数据对象 c1 和 c2，它们的分类不同
        c1 = Categorical([1, 2, 3], ordered=True)
        c2 = Categorical([4, 5, 6], ordered=True)

        # 测试合并这两个分类数据对象，忽略顺序
        result = union_categoricals([c1, c2], ignore_order=True)
        expected = Categorical([1, 2, 3, 4, 5, 6])
        tm.assert_categorical_equal(result, expected)

        # 定义错误消息
        msg = "to union ordered Categoricals, all categories must be the same"
        # 使用 pytest 断言引发 TypeError 异常，错误消息匹配指定的消息内容
        with pytest.raises(TypeError, match=msg):
            union_categoricals([c1, c2], ignore_order=False)

        # 再次使用 pytest 断言引发 TypeError 异常，错误消息匹配指定的消息内容
        with pytest.raises(TypeError, match=msg):
            union_categoricals([c1, c2])
    # 定义测试函数，用于测试 union_categoricals 函数在 sort=True 时的行为
    def test_union_categoricals_sort(self):
        # 创建第一个分类变量 c1，包含元素 "x", "y", "z"
        c1 = Categorical(["x", "y", "z"])
        # 创建第二个分类变量 c2，包含元素 "a", "b", "c"
        c2 = Categorical(["a", "b", "c"])
        # 调用 union_categoricals 函数，对 c1 和 c2 进行合并，并排序其分类值
        result = union_categoricals([c1, c2], sort_categories=True)
        # 创建预期的分类变量 expected，包含合并后的所有元素，并按指定顺序排序
        expected = Categorical(
            ["x", "y", "z", "a", "b", "c"], categories=["a", "b", "c", "x", "y", "z"]
        )
        # 断言 result 和 expected 的内容相等
        tm.assert_categorical_equal(result, expected)

        # fastpath 快速路径测试
        # 创建第一个分类变量 c1，包含元素 "a", "b"，并指定其排序顺序
        c1 = Categorical(["a", "b"], categories=["b", "a", "c"])
        # 创建第二个分类变量 c2，包含元素 "b", "c"，并指定其排序顺序
        c2 = Categorical(["b", "c"], categories=["b", "a", "c"])
        # 调用 union_categoricals 函数，对 c1 和 c2 进行合并，并排序其分类值
        result = union_categoricals([c1, c2], sort_categories=True)
        # 创建预期的分类变量 expected，包含合并后的所有元素，并按指定顺序排序
        expected = Categorical(["a", "b", "b", "c"], categories=["a", "b", "c"])
        # 断言 result 和 expected 的内容相等
        tm.assert_categorical_equal(result, expected)

        # 创建第一个分类变量 c1，包含元素 "a", "b"，并指定其排序顺序
        c1 = Categorical(["a", "b"], categories=["c", "a", "b"])
        # 创建第二个分类变量 c2，包含元素 "b", "c"，并指定其排序顺序
        c2 = Categorical(["b", "c"], categories=["c", "a", "b"])
        # 调用 union_categoricals 函数，对 c1 和 c2 进行合并，并排序其分类值
        result = union_categoricals([c1, c2], sort_categories=True)
        # 创建预期的分类变量 expected，包含合并后的所有元素，并按指定顺序排序
        expected = Categorical(["a", "b", "b", "c"], categories=["a", "b", "c"])
        # 断言 result 和 expected 的内容相等
        tm.assert_categorical_equal(result, expected)

        # fastpath - skip resort 快速路径测试 - 跳过重新排序
        # 创建第一个分类变量 c1，包含元素 "a", "b"，并指定其排序顺序
        c1 = Categorical(["a", "b"], categories=["a", "b", "c"])
        # 创建第二个分类变量 c2，包含元素 "b", "c"，并指定其排序顺序
        c2 = Categorical(["b", "c"], categories=["a", "b", "c"])
        # 调用 union_categoricals 函数，对 c1 和 c2 进行合并，并排序其分类值
        result = union_categoricals([c1, c2], sort_categories=True)
        # 创建预期的分类变量 expected，包含合并后的所有元素，并按指定顺序排序
        expected = Categorical(["a", "b", "b", "c"], categories=["a", "b", "c"])
        # 断言 result 和 expected 的内容相等
        tm.assert_categorical_equal(result, expected)

        # 创建第一个分类变量 c1，包含元素 "x" 和 np.nan（空值）
        c1 = Categorical(["x", np.nan])
        # 创建第二个分类变量 c2，包含元素 np.nan（空值）和 "b"
        c2 = Categorical([np.nan, "b"])
        # 调用 union_categoricals 函数，对 c1 和 c2 进行合并，并排序其分类值
        result = union_categoricals([c1, c2], sort_categories=True)
        # 创建预期的分类变量 expected，包含合并后的所有元素，并按指定顺序排序
        expected = Categorical(["x", np.nan, np.nan, "b"], categories=["b", "x"])
        # 断言 result 和 expected 的内容相等
        tm.assert_categorical_equal(result, expected)

        # 创建第一个分类变量 c1，包含元素 np.nan（空值）
        c1 = Categorical([np.nan])
        # 创建第二个分类变量 c2，包含元素 np.nan（空值）
        c2 = Categorical([np.nan])
        # 调用 union_categoricals 函数，对 c1 和 c2 进行合并，并排序其分类值
        result = union_categoricals([c1, c2], sort_categories=True)
        # 创建预期的分类变量 expected，包含合并后的所有元素
        expected = Categorical([np.nan, np.nan])
        # 断言 result 和 expected 的内容相等
        tm.assert_categorical_equal(result, expected)

        # 创建两个空的分类变量 c1 和 c2
        c1 = Categorical([])
        c2 = Categorical([])
        # 调用 union_categoricals 函数，对 c1 和 c2 进行合并，并排序其分类值
        result = union_categoricals([c1, c2], sort_categories=True)
        # 创建预期的空分类变量 expected
        expected = Categorical([])
        # 断言 result 和 expected 的内容相等
        tm.assert_categorical_equal(result, expected)

        # 创建第一个分类变量 c1，包含元素 "b", "a"，并指定其排序顺序，同时标记为有序分类
        c1 = Categorical(["b", "a"], categories=["b", "a", "c"], ordered=True)
        # 创建第二个分类变量 c2，包含元素 "a", "c"，并指定其排序顺序，同时标记为有序分类
        c2 = Categorical(["a", "c"], categories=["b", "a", "c"], ordered=True)
        # 准备错误消息内容
        msg = "Cannot use sort_categories=True with ordered Categoricals"
        # 期望调用 union_categoricals 函数会引发 TypeError 异常，且异常消息包含预期的错误消息
        with pytest.raises(TypeError, match=msg):
            union_categoricals([c1, c2], sort_categories=True)

    # 定义测试函数，用于测试 union_categoricals 函数在 sort=False 时的行为
    def test_union_categoricals_sort_false(self):
        # GH 13846
        # 创建第一个分类变量 c1，包含元素 "x", "y", "z"
        c1 = Categorical(["x", "y", "z"])
        # 创建第二个分类变量 c2，包含元素 "a", "b", "c"
        c2 = Categorical(["a", "b", "c"])
        # 调用 union_categoricals 函数，对 c1 和 c2 进行合并，不排序其分类值
        result = union_categoricals([c1, c2], sort_categories=False)
        # 创建预期的分类变量 expected，包含合并后的所有元素，且不进行排序
        expected = Categorical(
            ["x", "y", "z", "a", "b", "c"], categories=["x", "y", "z", "a", "b", "c"]
        )
        # 断言 result 和 expected 的内容相等
        tm.assert_categorical_equal(result, expected)
    def test_union_categoricals_sort_false_fastpath(self):
        # 测试快速路径
        # 创建两个分类变量 c1 和 c2，分别指定其类别顺序
        c1 = Categorical(["a", "b"], categories=["b", "a", "c"])
        c2 = Categorical(["b", "c"], categories=["b", "a", "c"])
        # 调用 union_categoricals 函数，不排序类别，将 c1 和 c2 合并
        result = union_categoricals([c1, c2], sort_categories=False)
        # 创建预期的分类变量 expected，指定其类别顺序
        expected = Categorical(["a", "b", "b", "c"], categories=["b", "a", "c"])
        # 断言 result 和 expected 相等
        tm.assert_categorical_equal(result, expected)

    def test_union_categoricals_sort_false_skipresort(self):
        # 测试快速路径 - 跳过重新排序
        # 创建两个分类变量 c1 和 c2，都已按指定类别顺序排列
        c1 = Categorical(["a", "b"], categories=["a", "b", "c"])
        c2 = Categorical(["b", "c"], categories=["a", "b", "c"])
        # 调用 union_categoricals 函数，不排序类别，将 c1 和 c2 合并
        result = union_categoricals([c1, c2], sort_categories=False)
        # 创建预期的分类变量 expected，指定其类别顺序
        expected = Categorical(["a", "b", "b", "c"], categories=["a", "b", "c"])
        # 断言 result 和 expected 相等
        tm.assert_categorical_equal(result, expected)

    def test_union_categoricals_sort_false_one_nan(self):
        # 创建两个分类变量 c1 和 c2，包含 NaN 值
        c1 = Categorical(["x", np.nan])
        c2 = Categorical([np.nan, "b"])
        # 调用 union_categoricals 函数，不排序类别，将 c1 和 c2 合并
        result = union_categoricals([c1, c2], sort_categories=False)
        # 创建预期的分类变量 expected，指定其类别顺序
        expected = Categorical(["x", np.nan, np.nan, "b"], categories=["x", "b"])
        # 断言 result 和 expected 相等
        tm.assert_categorical_equal(result, expected)

    def test_union_categoricals_sort_false_only_nan(self):
        # 创建两个分类变量 c1 和 c2，都只包含 NaN 值
        c1 = Categorical([np.nan])
        c2 = Categorical([np.nan])
        # 调用 union_categoricals 函数，不排序类别，将 c1 和 c2 合并
        result = union_categoricals([c1, c2], sort_categories=False)
        # 创建预期的分类变量 expected，不指定类别顺序
        expected = Categorical([np.nan, np.nan])
        # 断言 result 和 expected 相等
        tm.assert_categorical_equal(result, expected)

    def test_union_categoricals_sort_false_empty(self):
        # 创建两个空的分类变量 c1 和 c2
        c1 = Categorical([])
        c2 = Categorical([])
        # 调用 union_categoricals 函数，不排序类别，将 c1 和 c2 合并
        result = union_categoricals([c1, c2], sort_categories=False)
        # 创建预期的空分类变量 expected
        expected = Categorical([])
        # 断言 result 和 expected 相等
        tm.assert_categorical_equal(result, expected)

    def test_union_categoricals_sort_false_ordered_true(self):
        # 创建两个有序的分类变量 c1 和 c2，指定其有序属性
        c1 = Categorical(["b", "a"], categories=["b", "a", "c"], ordered=True)
        c2 = Categorical(["a", "c"], categories=["b", "a", "c"], ordered=True)
        # 调用 union_categoricals 函数，不排序类别，将 c1 和 c2 合并
        result = union_categoricals([c1, c2], sort_categories=False)
        # 创建预期的有序分类变量 expected，指定其有序属性
        expected = Categorical(
            ["b", "a", "a", "c"], categories=["b", "a", "c"], ordered=True
        )
        # 断言 result 和 expected 相等
        tm.assert_categorical_equal(result, expected)

    def test_union_categorical_unwrap(self):
        # GH 14173
        # 创建分类变量 c1 和 c2，以及预期的分类变量 expected
        c1 = Categorical(["a", "b"])
        c2 = Series(["b", "c"], dtype="category")
        result = union_categoricals([c1, c2])
        expected = Categorical(["a", "b", "b", "c"])
        # 断言 result 和 expected 相等
        tm.assert_categorical_equal(result, expected)

        # 将 c2 转换为 CategoricalIndex 对象，再次调用 union_categoricals 函数
        c2 = CategoricalIndex(c2)
        result = union_categoricals([c1, c2])
        # 断言 result 和 expected 相等
        tm.assert_categorical_equal(result, expected)

        # 将 c1 转换为 Series 对象，再次调用 union_categoricals 函数
        c1 = Series(c1)
        result = union_categoricals([c1, c2])
        # 断言 result 和 expected 相等
        tm.assert_categorical_equal(result, expected)

        # 测试异常情况，传入不是 Categorical 类型的对象
        msg = "all components to combine must be Categorical"
        with pytest.raises(TypeError, match=msg):
            union_categoricals([c1, ["a", "b", "c"]])
```