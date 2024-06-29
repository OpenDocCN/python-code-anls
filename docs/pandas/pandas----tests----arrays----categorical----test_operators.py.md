# `D:\src\scipysrc\pandas\pandas\tests\arrays\categorical\test_operators.py`

```
# 导入 NumPy 库，并将其重命名为 np
import numpy as np
# 导入 Pytest 库
import pytest

# 导入 Pandas 库，并从中导入以下模块
import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Series,
    Timestamp,
    date_range,
)
# 导入 Pandas 内部测试模块
import pandas._testing as tm

# 定义一个测试类 TestCategoricalOpsWithFactor
class TestCategoricalOpsWithFactor:
    # 定义测试方法 test_categories_none_comparisons
    def test_categories_none_comparisons(self):
        # 创建一个有序分类变量 factor，包含指定的数据
        factor = Categorical(["a", "b", "b", "a", "a", "c", "c", "c"], ordered=True)
        # 使用 Pandas 测试模块验证 factor 与自身是否相等
        tm.assert_categorical_equal(factor, factor)

# 定义一个测试类 TestCategoricalOps
class TestCategoricalOps:
    # 使用 pytest.mark.parametrize 装饰器，测试不同的分类变量 categories
    @pytest.mark.parametrize(
        "categories",
        [["a", "b"], [0, 1], [Timestamp("2019"), Timestamp("2020")]],
    )
    # 定义测试方法 test_not_equal_with_na，参数 categories 为不同的分类变量
    def test_not_equal_with_na(self, categories):
        # 创建一个分类变量 c1，从指定的 codes 中生成，并指定 categories
        c1 = Categorical.from_codes([-1, 0], categories=categories)
        # 创建另一个分类变量 c2，从指定的 codes 中生成，并指定 categories
        c2 = Categorical.from_codes([0, 1], categories=categories)

        # 比较两个分类变量 c1 和 c2 是否不等，结果存储在 result 中
        result = c1 != c2

        # 使用 assert 断言，验证 result 中的所有值都为 True
        assert result.all()

    # 定义测试方法 test_compare_frame，测试分类变量与 DataFrame 的比较
    def test_compare_frame(self):
        # 创建一个包含混合类型数据的列表 data
        data = ["a", "b", 2, "a"]
        # 创建一个分类变量 cat，从 data 中生成
        cat = Categorical(data)

        # 创建一个 DataFrame df，将分类变量 cat 转置后生成
        df = DataFrame(cat)

        # 比较分类变量 cat 与 DataFrame df 转置后的结果，并将结果存储在 result 中
        result = cat == df.T
        # 创建一个期望的 DataFrame 对象，包含预期的比较结果
        expected = DataFrame([[True, True, True, True]])
        # 使用 Pandas 测试模块验证 result 与 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # 再次比较分类变量 cat 与 DataFrame df 转置后的结果（反向比较），并将结果存储在 result 中
        result = cat[::-1] != df.T
        # 创建另一个期望的 DataFrame 对象，包含反向比较的预期结果
        expected = DataFrame([[False, True, True, False]])
        # 使用 Pandas 测试模块验证 result 与 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 定义测试方法 test_compare_frame_raises，测试分类变量与 DataFrame 比较时引发异常情况
    def test_compare_frame_raises(self, comparison_op):
        # 获取比较操作符
        op = comparison_op
        # 创建一个包含混合类型数据的分类变量 cat
        cat = Categorical(["a", "b", 2, "a"])
        # 创建一个 DataFrame df，将分类变量 cat 转换为 DataFrame
        df = DataFrame(cat)
        # 设置预期的异常消息
        msg = "Unable to coerce to Series, length must be 1: given 4"
        # 使用 pytest 断言，验证执行 op(cat, df) 操作时是否引发 ValueError 异常，并检查异常消息是否匹配
        with pytest.raises(ValueError, match=msg):
            op(cat, df)

    # 定义测试方法 test_datetime_categorical_comparison，测试日期时间分类变量的比较
    def test_datetime_categorical_comparison(self):
        # 创建一个有序日期时间分类变量 dt_cat，从指定日期范围生成
        dt_cat = Categorical(date_range("2014-01-01", periods=3), ordered=True)
        # 使用 Pandas 测试模块验证 dt_cat 与 dt_cat[0] 的比较结果是否与预期一致
        tm.assert_numpy_array_equal(dt_cat > dt_cat[0], np.array([False, True, True]))
        # 使用 Pandas 测试模块验证 dt_cat[0] 与 dt_cat 的比较结果是否与预期一致
        tm.assert_numpy_array_equal(dt_cat[0] < dt_cat, np.array([False, True, True]))

    # 定义测试方法 test_reflected_comparison_with_scalars，测试标量与分类变量的反射比较
    def test_reflected_comparison_with_scalars(self):
        # 创建一个有序整数分类变量 cat，包含指定的数据
        cat = Categorical([1, 2, 3], ordered=True)
        # 使用 Pandas 测试模块验证 cat 与 cat[0] 的比较结果是否与预期一致
        tm.assert_numpy_array_equal(cat > cat[0], np.array([False, True, True]))
        # 使用 Pandas 测试模块验证 cat[0] 与 cat 的比较结果是否与预期一致
        tm.assert_numpy_array_equal(cat[0] < cat, np.array([False, True, True]))
    def test_comparison_with_unknown_scalars(self):
        # https://github.com/pandas-dev/pandas/issues/9836#issuecomment-92123057
        # and following comparisons with scalars not in categories should raise
        # for unequal comps, but not for equal/not equal
        # 创建一个有序的分类对象，包含整数 1, 2, 3
        cat = Categorical([1, 2, 3], ordered=True)

        # 准备一个错误消息，用于比较分类对象和整数时的异常匹配
        msg = "Invalid comparison between dtype=category and int"
        # 检查小于运算符与分类对象和整数 4 的比较，应该引发 TypeError 异常并匹配错误消息
        with pytest.raises(TypeError, match=msg):
            cat < 4
        # 检查大于运算符与分类对象和整数 4 的比较，应该引发 TypeError 异常并匹配错误消息
        with pytest.raises(TypeError, match=msg):
            cat > 4
        # 检查小于运算符与整数 4 和分类对象的比较，应该引发 TypeError 异常并匹配错误消息
        with pytest.raises(TypeError, match=msg):
            4 < cat
        # 检查大于运算符与整数 4 和分类对象的比较，应该引发 TypeError 异常并匹配错误消息
        with pytest.raises(TypeError, match=msg):
            4 > cat

        # 检查相等运算符与整数 4 和分类对象的比较，返回一个布尔数组，指示是否相等
        tm.assert_numpy_array_equal(cat == 4, np.array([False, False, False]))
        # 检查不等运算符与整数 4 和分类对象的比较，返回一个布尔数组，指示是否不相等
        tm.assert_numpy_array_equal(cat != 4, np.array([True, True, True]))

    def test_comparison_with_tuple(self):
        # 创建一个包含对象数组的分类对象，包含字符串 "foo" 和元组 (0, 1) 等
        cat = Categorical(np.array(["foo", (0, 1), 3, (0, 1)], dtype=object))

        # 检查相等运算符与字符串 "foo" 和分类对象的比较，返回一个布尔数组，指示是否相等
        result = cat == "foo"
        expected = np.array([True, False, False, False], dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

        # 检查相等运算符与元组 (0, 1) 和分类对象的比较，返回一个布尔数组，指示是否相等
        result = cat == (0, 1)
        expected = np.array([False, True, False, True], dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

        # 检查不等运算符与元组 (0, 1) 和分类对象的比较，返回一个布尔数组，指示是否不相等
        result = cat != (0, 1)
        tm.assert_numpy_array_equal(result, ~expected)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_comparison_of_ordered_categorical_with_nan_to_scalar(
        self, compare_operators_no_eq_ne
    ):
        # https://github.com/pandas-dev/pandas/issues/26504
        # BUG: fix ordered categorical comparison with missing values (#26504 )
        # 创建一个有序的分类对象，包含整数 1, 2, 3 和缺失值 None
        cat = Categorical([1, 2, 3, None], categories=[1, 2, 3], ordered=True)
        scalar = 2
        # 使用给定的比较运算符（compare_operators_no_eq_ne）测试分类对象和标量的比较，期望的结果存储在 expected 中
        expected = getattr(np.array(cat), compare_operators_no_eq_ne)(scalar)
        # 执行分类对象和标量的比较，实际结果存储在 actual 中
        actual = getattr(cat, compare_operators_no_eq_ne)(scalar)
        # 断言实际结果与期望结果相等
        tm.assert_numpy_array_equal(actual, expected)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_comparison_of_ordered_categorical_with_nan_to_listlike(
        self, compare_operators_no_eq_ne
    ):
        # https://github.com/pandas-dev/pandas/issues/26504
        # 创建一个有序的分类对象，包含整数 1, 2, 3 和缺失值 None
        cat = Categorical([1, 2, 3, None], categories=[1, 2, 3], ordered=True)
        # 创建另一个有序的分类对象，包含整数 2，并且与 cat 具有相同的分类
        other = Categorical([2, 2, 2, 2], categories=[1, 2, 3], ordered=True)
        # 使用给定的比较运算符（compare_operators_no_eq_ne）测试分类对象和 list-like 对象的比较，期望的结果存储在 expected 中
        expected = getattr(np.array(cat), compare_operators_no_eq_ne)(2)
        # 执行分类对象和 list-like 对象的比较，实际结果存储在 actual 中
        actual = getattr(cat, compare_operators_no_eq_ne)(other)
        # 断言实际结果与期望结果相等
        tm.assert_numpy_array_equal(actual, expected)

    @pytest.mark.parametrize(
        "data,reverse,base",
        [(list("abc"), list("cba"), list("bbb")), ([1, 2, 3], [3, 2, 1], [2, 2, 2])],
    )
    # 定义一个测试方法，用于比较不同的类别数据、反向数据和基础数据
    def test_comparisons(self, data, reverse, base):
        # 创建一个分类数据的 Series 对象，使用指定的反向类别列表，并指定为有序
        cat_rev = Series(Categorical(data, categories=reverse, ordered=True))
        # 创建另一个分类数据的 Series 对象，使用相同的反向类别列表，并指定为有序
        cat_rev_base = Series(Categorical(base, categories=reverse, ordered=True))
        # 创建一个分类数据的 Series 对象，指定为有序
        cat = Series(Categorical(data, ordered=True))
        # 创建另一个分类数据的 Series 对象，使用基础数据的类别，并指定为有序
        cat_base = Series(
            Categorical(base, categories=cat.cat.categories, ordered=True)
        )
        # 创建一个 Series 对象，根据基础数据的类型来决定其数据类型（对象或者 None 类型）
        s = Series(base, dtype=object if base == list("bbb") else None)
        # 创建一个基于基础数据的 NumPy 数组
        a = np.array(base)

        # 进行比较操作时需要考虑类别的顺序
        res_rev = cat_rev > cat_rev_base
        exp_rev = Series([True, False, False])
        tm.assert_series_equal(res_rev, exp_rev)

        res_rev = cat_rev < cat_rev_base
        exp_rev = Series([False, False, True])
        tm.assert_series_equal(res_rev, exp_rev)

        res = cat > cat_base
        exp = Series([False, False, True])
        tm.assert_series_equal(res, exp)

        # 使用基础数据的标量值进行比较
        scalar = base[1]
        res = cat > scalar
        exp = Series([False, False, True])
        exp2 = cat.values > scalar
        tm.assert_series_equal(res, exp)
        tm.assert_numpy_array_equal(res.values, exp2)
        res_rev = cat_rev > scalar
        exp_rev = Series([True, False, False])
        exp_rev2 = cat_rev.values > scalar
        tm.assert_series_equal(res_rev, exp_rev)
        tm.assert_numpy_array_equal(res_rev.values, exp_rev2)

        # 只有具有相同类别的分类数据可以进行比较
        msg = "Categoricals can only be compared if 'categories' are the same"
        with pytest.raises(TypeError, match=msg):
            cat > cat_rev

        # 分类数据不能与 Series 或者 NumPy 数组进行比较，反之亦然
        msg = (
            "Cannot compare a Categorical for op __gt__ with type "
            r"<class 'numpy\.ndarray'>"
        )
        with pytest.raises(TypeError, match=msg):
            cat > s
        with pytest.raises(TypeError, match=msg):
            cat_rev > s
        with pytest.raises(TypeError, match=msg):
            cat > a
        with pytest.raises(TypeError, match=msg):
            cat_rev > a

        with pytest.raises(TypeError, match=msg):
            s < cat
        with pytest.raises(TypeError, match=msg):
            s < cat_rev

        with pytest.raises(TypeError, match=msg):
            a < cat
        with pytest.raises(TypeError, match=msg):
            a < cat_rev
    # 定义一个测试方法，用于验证无序分类数据的比较结果是否相等，基于给定的盒子函数（box）
    def test_unordered_different_order_equal(self, box):
        # 创建两个无序分类数据对象 c1 和 c2，传入的数据相同但顺序不同
        c1 = box(Categorical(["a", "b"], categories=["a", "b"], ordered=False))
        c2 = box(Categorical(["a", "b"], categories=["b", "a"], ordered=False))
        # 断言两个对象完全相等
        assert (c1 == c2).all()

        # 创建两个无序分类数据对象 c1 和 c2，传入的数据不同
        c1 = box(Categorical(["a", "b"], categories=["a", "b"], ordered=False))
        c2 = box(Categorical(["b", "a"], categories=["b", "a"], ordered=False))
        # 断言两个对象不相等
        assert (c1 != c2).all()

        # 创建两个无序分类数据对象 c1 和 c2，传入的数据相同但顺序不同
        c1 = box(Categorical(["a", "a"], categories=["a", "b"], ordered=False))
        c2 = box(Categorical(["b", "b"], categories=["b", "a"], ordered=False))
        # 断言两个对象不相等
        assert (c1 != c2).all()

        # 创建两个无序分类数据对象 c1 和 c2，传入的数据和顺序不同
        c1 = box(Categorical(["a", "a"], categories=["a", "b"], ordered=False))
        c2 = box(Categorical(["a", "b"], categories=["b", "a"], ordered=False))
        # 对比 c1 和 c2，将结果存储在 result 中
        result = c1 == c2
        # 使用测试工具库（tm）来断言两个 numpy 数组的相等性
        tm.assert_numpy_array_equal(np.array(result), np.array([True, False]))

    # 定义一个测试方法，用于验证当两个分类数据对象的类别不同时是否会抛出异常
    def test_unordered_different_categories_raises(self):
        # 创建两个分类数据对象 c1 和 c2，传入的类别列表不同
        c1 = Categorical(["a", "b"], categories=["a", "b"], ordered=False)
        c2 = Categorical(["a", "c"], categories=["c", "a"], ordered=False)

        # 使用 pytest 的断言功能，预期会抛出 TypeError 异常，并且异常信息匹配特定的字符串
        with pytest.raises(TypeError, match=("Categoricals can only be compared")):
            c1 == c2

    # 定义一个测试方法，用于验证当两个分类数据对象的长度不同时是否会抛出异常
    def test_compare_different_lengths(self):
        # 创建两个分类数据对象 c1 和 c2，传入的数据列表为空，但类别列表不同
        c1 = Categorical([], categories=["a", "b"])
        c2 = Categorical([], categories=["a"])

        # 准备一个异常消息字符串
        msg = "Categoricals can only be compared if 'categories' are the same."
        # 使用 pytest 的断言功能，预期会抛出 TypeError 异常，并且异常信息匹配预设的消息
        with pytest.raises(TypeError, match=msg):
            c1 == c2

    # 定义一个测试方法，用于验证当两个无序分类数据对象的顺序不同时，它们是否不相等
    def test_compare_unordered_different_order(self):
        # 创建两个无序分类数据对象 a 和 b，传入的数据相同但类别顺序不同
        a = Categorical(["a"], categories=["a", "b"])
        b = Categorical(["b"], categories=["b", "a"])
        # 断言 a 和 b 不相等
        assert not a.equals(b)
    # 定义一个测试函数，用于测试 DataFrame 的数值操作
    def test_numeric_like_ops(self):
        # 创建一个包含随机整数值的 DataFrame
        df = DataFrame({"value": np.random.default_rng(2).integers(0, 10000, 100)})
        # 创建一组标签，每个标签表示一组数值范围
        labels = [f"{i} - {i + 499}" for i in range(0, 10000, 500)]
        # 创建一个分类变量，使用上述标签作为值和类别标签
        cat_labels = Categorical(labels, labels)

        # 按照 'value' 列对 DataFrame 进行升序排序
        df = df.sort_values(by=["value"], ascending=True)
        # 将 'value' 列的数值划分为指定范围，并使用 cat_labels 作为标签
        df["value_group"] = pd.cut(
            df.value, range(0, 10500, 500), right=False, labels=cat_labels
        )

        # 对数值运算进行测试，预期应该会抛出 TypeError 异常
        for op, str_rep in [
            ("__add__", r"\+"),
            ("__sub__", "-"),
            ("__mul__", r"\*"),
            ("__truediv__", "/"),
        ]:
            msg = f"Series cannot perform the operation {str_rep}|unsupported operand"
            with pytest.raises(TypeError, match=msg):
                getattr(df, op)(df)

        # 对约简操作进行测试，预期应该会抛出 TypeError 异常，除非特别定义支持的操作（如 min/max）
        s = df["value_group"]
        for op in ["kurt", "skew", "var", "std", "mean", "sum", "median"]:
            msg = f"does not support operation '{op}'"
            with pytest.raises(TypeError, match=msg):
                getattr(s, op)(numeric_only=False)

    # 测试 Series 上的数值操作
    def test_numeric_like_ops_series(self):
        # 使用分类变量创建一个 Series
        s = Series(Categorical([1, 2, 3, 4]))
        # 测试 numpy 操作，预期应该会抛出 TypeError 异常，提示不支持 'sum' 操作
        with pytest.raises(TypeError, match="does not support operation 'sum'"):
            np.sum(s)

    # 参数化测试，测试 Series 上的数值操作（算术运算）
    @pytest.mark.parametrize(
        "op, str_rep",
        [
            ("__add__", r"\+"),
            ("__sub__", "-"),
            ("__mul__", r"\*"),
            ("__truediv__", "/"),
        ],
    )
    def test_numeric_like_ops_series_arith(self, op, str_rep):
        # 使用分类变量创建一个 Series
        s = Series(Categorical([1, 2, 3, 4]))
        # 测试 Series 上的数值运算，预期应该会抛出 TypeError 异常，提示不支持特定操作
        msg = f"Series cannot perform the operation {str_rep}|unsupported operand"
        with pytest.raises(TypeError, match=msg):
            getattr(s, op)(2)

    # 测试不支持的 ufunc 操作
    def test_numeric_like_ops_series_invalid(self):
        # 使用分类变量创建一个 Series
        s = Series(Categorical([1, 2, 3, 4]))
        # 测试不支持的 numpy ufunc 操作，预期应该会抛出 TypeError 异常，提示不支持 'log' 操作
        msg = "Object with dtype category cannot perform the numpy op log"
        with pytest.raises(TypeError, match=msg):
            np.log(s)
```