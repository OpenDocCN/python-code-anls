# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_align.py`

```
from datetime import timezone  # 导入时区相关的功能

import numpy as np  # 导入NumPy库
import pytest  # 导入pytest库

import pandas as pd  # 导入Pandas库
from pandas import (  # 从Pandas库中导入多个模块和函数
    DataFrame,
    Index,
    Series,
    date_range,
)
import pandas._testing as tm  # 导入Pandas的测试工具模块


class TestDataFrameAlign:
    def test_frame_align_aware(self):
        # 创建具有时区信息的时间索引
        idx1 = date_range("2001", periods=5, freq="h", tz="US/Eastern")
        idx2 = date_range("2001", periods=5, freq="2h", tz="US/Eastern")
        # 创建随机数据填充的DataFrame对象，使用NumPy生成随机数
        df1 = DataFrame(np.random.default_rng(2).standard_normal((len(idx1), 3)), idx1)
        df2 = DataFrame(np.random.default_rng(2).standard_normal((len(idx2), 3)), idx2)
        # 对齐DataFrame对象
        new1, new2 = df1.align(df2)
        assert df1.index.tz == new1.index.tz  # 断言新的索引时区与原始索引相同
        assert df2.index.tz == new2.index.tz  # 断言新的索引时区与原始索引相同

        # 不同的时区会被转换为UTC

        # DataFrame与DataFrame对齐
        df1_central = df1.tz_convert("US/Central")
        new1, new2 = df1.align(df1_central)
        assert new1.index.tz is timezone.utc  # 断言新的索引时区为UTC
        assert new2.index.tz is timezone.utc  # 断言新的索引时区为UTC

        # DataFrame与Series对齐
        new1, new2 = df1.align(df1_central[0], axis=0)
        assert new1.index.tz is timezone.utc  # 断言新的索引时区为UTC
        assert new2.index.tz is timezone.utc  # 断言新的索引时区为UTC

        df1[0].align(df1_central, axis=0)
        assert new1.index.tz is timezone.utc  # 断言新的索引时区为UTC
        assert new2.index.tz is timezone.utc  # 断言新的索引时区为UTC

    def test_align_float(self, float_frame):
        # 对齐两个浮点数DataFrame对象
        af, bf = float_frame.align(float_frame)
        assert af._mgr is not float_frame._mgr  # 断言对齐后的DataFrame不是原始DataFrame的引用

        af, bf = float_frame.align(float_frame)
        assert af._mgr is not float_frame._mgr  # 断言对齐后的DataFrame不是原始DataFrame的引用

        # 在axis=0轴上对齐
        other = float_frame.iloc[:-5, :3]
        af, bf = float_frame.align(other, axis=0, fill_value=-1)

        tm.assert_index_equal(bf.columns, other.columns)  # 使用测试工具验证列索引相等

        # 测试填充值
        join_idx = float_frame.index.join(other.index)
        diff_a = float_frame.index.difference(join_idx)
        diff_a_vals = af.reindex(diff_a).values
        assert (diff_a_vals == -1).all()  # 断言填充后的值全部为-1

        af, bf = float_frame.align(other, join="right", axis=0)
        tm.assert_index_equal(bf.columns, other.columns)  # 使用测试工具验证列索引相等
        tm.assert_index_equal(bf.index, other.index)  # 使用测试工具验证行索引相等
        tm.assert_index_equal(af.index, other.index)  # 使用测试工具验证行索引相等

        # 在axis=1轴上对齐
        other = float_frame.iloc[:-5, :3].copy()
        af, bf = float_frame.align(other, axis=1)
        tm.assert_index_equal(bf.columns, float_frame.columns)  # 使用测试工具验证列索引相等
        tm.assert_index_equal(bf.index, other.index)  # 使用测试工具验证行索引相等

        # 测试填充值
        join_idx = float_frame.index.join(other.index)
        diff_a = float_frame.index.difference(join_idx)
        diff_a_vals = af.reindex(diff_a).values
        assert (diff_a_vals == -1).all()  # 断言填充后的值全部为-1

        af, bf = float_frame.align(other, join="inner", axis=1)
        tm.assert_index_equal(bf.columns, other.columns)  # 使用测试工具验证列索引相等

        # 尝试在错误的轴上将DataFrame与Series对齐
        msg = "No axis named 2 for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            float_frame.align(af.iloc[0, :3], join="inner", axis=2)
    def test_align_frame_with_series(self, float_frame):
        # align dataframe to series with broadcast or not
        # 获取输入数据帧的索引
        idx = float_frame.index
        # 创建一个序列，其索引与输入数据帧相同，值为索引位置
        s = Series(range(len(idx)), index=idx)

        # 将输入数据帧与序列按行对齐
        left, right = float_frame.align(s, axis=0)
        # 断言左对齐结果的索引与输入数据帧相同
        tm.assert_index_equal(left.index, float_frame.index)
        # 断言右对齐结果的索引与输入数据帧相同
        tm.assert_index_equal(right.index, float_frame.index)
        # 断言右对齐结果为序列类型
        assert isinstance(right, Series)

    def test_align_series_condition(self):
        # see gh-9558
        # 创建一个数据帧
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        # 使用条件筛选数据帧的行，选择"a"列为2的行
        result = df[df["a"] == 2]
        # 创建预期的数据帧
        expected = DataFrame([[2, 5]], index=[1], columns=["a", "b"])
        # 断言条件筛选的结果与预期的数据帧相等
        tm.assert_frame_equal(result, expected)

        # 将数据帧中不满足条件的元素替换为0
        result = df.where(df["a"] == 2, 0)
        # 创建预期的数据帧
        expected = DataFrame({"a": [0, 2, 0], "b": [0, 5, 0]})
        # 断言条件替换的结果与预期的数据帧相等
        tm.assert_frame_equal(result, expected)

    def test_align_mixed_float(self, mixed_float_frame):
        # mixed floats/ints
        # 创建另一个数据帧
        other = DataFrame(index=range(5), columns=["A", "B", "C"])
        # 将混合浮点数和整数数据帧与另一个数据帧的"B"列按列对齐
        af, bf = mixed_float_frame.align(
            other.iloc[:, 0], join="inner", axis=1, fill_value=0
        )
        # 断言对齐后的结果的索引为空索引
        tm.assert_index_equal(bf.index, Index([]))

    def test_align_mixed_int(self, mixed_int_frame):
        # 创建另一个数据帧
        other = DataFrame(index=range(5), columns=["A", "B", "C"])
        # 将混合整数数据帧与另一个数据帧的"B"列按列对齐
        af, bf = mixed_int_frame.align(
            other.iloc[:, 0], join="inner", axis=1, fill_value=0
        )
        # 断言对齐后的结果的索引为空索引
        tm.assert_index_equal(bf.index, Index([]))

    @pytest.mark.parametrize(
        "l_ordered,r_ordered,expected",
        [
            [True, True, pd.CategoricalIndex],
            [True, False, Index],
            [False, True, Index],
            [False, False, pd.CategoricalIndex],
        ],
    )
    def test_align_categorical(self, l_ordered, r_ordered, expected):
        # GH-28397
        # 创建第一个数据帧，包含"A"列和按条件排序的"B"列
        df_1 = DataFrame(
            {
                "A": np.arange(6, dtype="int64"),
                "B": Series(list("aabbca")).astype(
                    pd.CategoricalDtype(list("cab"), ordered=l_ordered)
                ),
            }
        ).set_index("B")
        # 创建第二个数据帧，包含"A"列和按条件排序的"B"列
        df_2 = DataFrame(
            {
                "A": np.arange(5, dtype="int64"),
                "B": Series(list("babca")).astype(
                    pd.CategoricalDtype(list("cab"), ordered=r_ordered)
                ),
            }
        ).set_index("B")

        # 将两个数据帧按索引对齐
        aligned_1, aligned_2 = df_1.align(df_2)
        # 断言对齐后的第一个结果的索引类型与预期相同
        assert isinstance(aligned_1.index, expected)
        # 断言对齐后的第二个结果的索引类型与预期相同
        assert isinstance(aligned_2.index, expected)
        # 断言对齐后的第一个结果的索引与对齐后的第二个结果的索引相同
        tm.assert_index_equal(aligned_1.index, aligned_2.index)
    def test_align_multiindex(self):
        # 定义一个测试函数，用于测试多级索引的对齐功能
        # GH#10665
        # 与 test_series.py 中的 test_align_multiindex 具有相同的测试用例

        # 创建一个三层的 MultiIndex，分别从各个范围生成，命名为 ("a", "b", "c")
        midx = pd.MultiIndex.from_product(
            [range(2), range(3), range(2)], names=("a", "b", "c")
        )
        # 创建一个单级索引，从 0 到 1，命名为 "b"
        idx = Index(range(2), name="b")
        # 创建一个包含 12 个元素的 DataFrame，使用 int64 类型，以 midx 作为索引
        df1 = DataFrame(np.arange(12, dtype="int64"), index=midx)
        # 创建一个包含 2 个元素的 DataFrame，使用 int64 类型，以 idx 作为索引
        df2 = DataFrame(np.arange(2, dtype="int64"), index=idx)

        # 对齐操作，使用左连接，得到两组结果 res1l, res1r 和 res2l, res2r
        res1l, res1r = df1.align(df2, join="left")
        res2l, res2r = df2.align(df1, join="right")

        # 预期 expl 应与 res1l 和 res2r 相等
        expl = df1
        tm.assert_frame_equal(expl, res1l)
        tm.assert_frame_equal(expl, res2r)
        # 预期 expr 应与 res1r 和 res2l 相等
        expr = DataFrame([0, 0, 1, 1, np.nan, np.nan] * 2, index=midx)
        tm.assert_frame_equal(expr, res1r)
        tm.assert_frame_equal(expr, res2l)

        # 对齐操作，使用右连接，得到两组结果 res1l, res1r 和 res2l, res2r
        res1l, res1r = df1.align(df2, join="right")
        res2l, res2r = df2.align(df1, join="left")

        # 创建一个新的 MultiIndex，从各个范围生成，命名为 ("a", "b", "c")
        exp_idx = pd.MultiIndex.from_product(
            [range(2), range(2), range(2)], names=("a", "b", "c")
        )
        # 预期 expl 应与 res1l 和 res2r 相等
        expl = DataFrame([0, 1, 2, 3, 6, 7, 8, 9], index=exp_idx)
        tm.assert_frame_equal(expl, res1l)
        tm.assert_frame_equal(expl, res2r)
        # 预期 expr 应与 res1r 和 res2l 相等
        expr = DataFrame([0, 0, 1, 1] * 2, index=exp_idx)
        tm.assert_frame_equal(expr, res1r)
        tm.assert_frame_equal(expr, res2l)

    def test_align_series_combinations(self):
        # 定义一个测试函数，用于测试不同 DataFrame 和 Series 的对齐组合

        # 创建一个 DataFrame，包含两列 "a" 和 "b"，以及对应的索引
        df = DataFrame({"a": [1, 3, 5], "b": [1, 3, 5]}, index=list("ACE"))
        # 创建一个 Series，包含三个元素，以及对应的索引和名称 "x"
        s = Series([1, 2, 4], index=list("ABD"), name="x")

        # 对齐操作，axis=0 表示按行对齐，得到两个结果 res1 和 res2
        res1, res2 = df.align(s, axis=0)
        # 创建预期的 DataFrame exp1 和 Series exp2
        exp1 = DataFrame(
            {"a": [1, np.nan, 3, np.nan, 5], "b": [1, np.nan, 3, np.nan, 5]},
            index=list("ABCDE"),
        )
        exp2 = Series([1, 2, np.nan, 4, np.nan], index=list("ABCDE"), name="x")

        tm.assert_frame_equal(res1, exp1)
        tm.assert_series_equal(res2, exp2)

        # 对齐操作，Series 对 DataFrame，得到两个结果 res1 和 res2
        res1, res2 = s.align(df)
        tm.assert_series_equal(res1, exp2)
        tm.assert_frame_equal(res2, exp1)

    def test_multiindex_align_to_series_with_common_index_level(self):
        # 定义一个测试函数，用于测试多级索引 DataFrame 对 Series 的对齐，当有公共索引级别时

        # 创建一个名为 "foo" 的单级索引，包含 1、2、3 三个元素
        foo_index = Index([1, 2, 3], name="foo")
        # 创建一个名为 "bar" 的单级索引，包含 1、2 两个元素
        bar_index = Index([1, 2], name="bar")

        # 创建一个 Series，包含两个元素，以 bar_index 作为索引，命名为 "foo_series"
        series = Series([1, 2], index=bar_index, name="foo_series")
        # 创建一个包含 6 个元素的 DataFrame，使用 np.arange(6) 作为数据，以 foo_index 和 bar_index 的笛卡尔积作为 MultiIndex
        df = DataFrame(
            {"col": np.arange(6)},
            index=pd.MultiIndex.from_product([foo_index, bar_index]),
        )

        # 预期的结果 expected_r 应与 result_l 和 result_r 相等
        expected_r = Series([1, 2] * 3, index=df.index, name="foo_series")
        result_l, result_r = df.align(series, axis=0)

        tm.assert_frame_equal(result_l, df)
        tm.assert_series_equal(result_r, expected_r)
    # 定义一个测试函数，用于验证多级索引数据框与系列对齐时，左侧缺少共同索引级别的情况
    def test_multiindex_align_to_series_with_common_index_level_missing_in_left(self):
        # GH-46001：测试用例编号
        foo_index = Index([1, 2, 3], name="foo")  # 创建名为 "foo" 的索引对象
        bar_index = Index([1, 2], name="bar")  # 创建名为 "bar" 的索引对象

        series = Series(
            [1, 2, 3, 4], index=Index([1, 2, 3, 4], name="bar"), name="foo_series"
        )  # 创建一个名为 "foo_series" 的系列对象，具有 "bar" 索引和指定名称
        df = DataFrame(
            {"col": np.arange(6)},  # 创建一个数据框，包含一列名为 "col" 的 NumPy 数组
            index=pd.MultiIndex.from_product([foo_index, bar_index]),  # 使用 foo_index 和 bar_index 创建多级索引
        )

        expected_r = Series([1, 2] * 3, index=df.index, name="foo_series")  # 期望的结果系列对象
        result_l, result_r = df.align(series, axis=0)  # 执行数据框与系列对象的对齐操作，axis=0 表示按行对齐

        tm.assert_frame_equal(result_l, df)  # 断言：验证 result_l 与 df 相等
        tm.assert_series_equal(result_r, expected_r)  # 断言：验证 result_r 与 expected_r 相等

    # 定义一个测试函数，用于验证多级索引数据框与系列对齐时，右侧缺少共同索引级别的情况
    def test_multiindex_align_to_series_with_common_index_level_missing_in_right(self):
        # GH-46001：测试用例编号
        foo_index = Index([1, 2, 3], name="foo")  # 创建名为 "foo" 的索引对象
        bar_index = Index([1, 2, 3, 4], name="bar")  # 创建名为 "bar" 的索引对象

        series = Series([1, 2], index=Index([1, 2], name="bar"), name="foo_series")  # 创建一个名为 "foo_series" 的系列对象，具有 "bar" 索引和指定名称
        df = DataFrame(
            {"col": np.arange(12)},  # 创建一个数据框，包含一列名为 "col" 的 NumPy 数组
            index=pd.MultiIndex.from_product([foo_index, bar_index]),  # 使用 foo_index 和 bar_index 创建多级索引
        )

        expected_r = Series(
            [1, 2, np.nan, np.nan] * 3, index=df.index, name="foo_series"
        )  # 期望的结果系列对象，包含 NaN 值
        result_l, result_r = df.align(series, axis=0)  # 执行数据框与系列对象的对齐操作，axis=0 表示按行对齐

        tm.assert_frame_equal(result_l, df)  # 断言：验证 result_l 与 df 相等
        tm.assert_series_equal(result_r, expected_r)  # 断言：验证 result_r 与 expected_r 相等

    # 定义一个测试函数，用于验证多级索引数据框与系列对齐时，左右两侧都缺少共同索引级别的情况
    def test_multiindex_align_to_series_with_common_index_level_missing_in_both(self):
        # GH-46001：测试用例编号
        foo_index = Index([1, 2, 3], name="foo")  # 创建名为 "foo" 的索引对象
        bar_index = Index([1, 3, 4], name="bar")  # 创建名为 "bar" 的索引对象

        series = Series(
            [1, 2, 3], index=Index([1, 2, 4], name="bar"), name="foo_series"
        )  # 创建一个名为 "foo_series" 的系列对象，具有 "bar" 索引和指定名称
        df = DataFrame(
            {"col": np.arange(9)},  # 创建一个数据框，包含一列名为 "col" 的 NumPy 数组
            index=pd.MultiIndex.from_product([foo_index, bar_index]),  # 使用 foo_index 和 bar_index 创建多级索引
        )

        expected_r = Series([1, np.nan, 3] * 3, index=df.index, name="foo_series")  # 期望的结果系列对象，包含 NaN 值
        result_l, result_r = df.align(series, axis=0)  # 执行数据框与系列对象的对齐操作，axis=0 表示按行对齐

        tm.assert_frame_equal(result_l, df)  # 断言：验证 result_l 与 df 相等
        tm.assert_series_equal(result_r, expected_r)  # 断言：验证 result_r 与 expected_r 相等

    # 定义一个测试函数，用于验证多级索引数据框与系列对齐时，左侧具有非唯一列名称的情况
    def test_multiindex_align_to_series_with_common_index_level_non_unique_cols(self):
        # GH-46001：测试用例编号
        foo_index = Index([1, 2, 3], name="foo")  # 创建名为 "foo" 的索引对象
        bar_index = Index([1, 2], name="bar")  # 创建名为 "bar" 的索引对象

        series = Series([1, 2], index=bar_index, name="foo_series")  # 创建一个名为 "foo_series" 的系列对象，具有 "bar" 索引和指定名称
        df = DataFrame(
            np.arange(18).reshape(6, 3),  # 创建一个数据框，包含一个 6x3 的 NumPy 数组
            index=pd.MultiIndex.from_product([foo_index, bar_index]),  # 使用 foo_index 和 bar_index 创建多级索引
        )
        df.columns = ["cfoo", "cbar", "cfoo"]  # 修改数据框的列名称为指定列表

        expected = Series([1, 2] * 3, index=df.index, name="foo_series")  # 期望的结果系列对象
        result_left, result_right = df.align(series, axis=0)  # 执行数据框与系列对象的对齐操作，axis=0 表示按行对齐

        tm.assert_series_equal(result_right, expected)  # 断言：验证 result_right 与 expected 相等
        tm.assert_index_equal(result_left.columns, df.columns)  # 断言：验证 result_left 的列索引与 df 的列索引相等
    def test_missing_axis_specification_exception(self):
        # 创建一个包含 10 行 5 列的 DataFrame，元素为 0 到 49 的整数
        df = DataFrame(np.arange(50).reshape((10, 5)))
        # 创建一个包含 5 个元素的 Series，元素为 0 到 4 的整数
        series = Series(np.arange(5))

        # 使用 pytest 检查是否会抛出 ValueError 异常，并匹配错误消息中包含 "axis=0 or 1"
        with pytest.raises(ValueError, match=r"axis=0 or 1"):
            # 调用 DataFrame 的 align 方法，对 Series 进行对齐操作
            df.align(series)

    def test_align_series_check_copy(self):
        # GH# 标识 GitHub 上的 issue 编号
        df = DataFrame({0: [1, 2]})
        # 创建一个包含单列名为 0 的 DataFrame，数据为 [1, 2]
        ser = Series([1], name=0)
        # 复制 Series 对象，创建一个新的期望结果
        expected = ser.copy()
        # 调用 DataFrame 的 align 方法，对 Series 进行对齐操作，指定轴向为列（axis=1）
        result, other = df.align(ser, axis=1)
        # 修改原始 Series 的第一个元素为 100
        ser.iloc[0] = 100
        # 使用 pytest 的 assert 比较两个 Series 是否相等
        tm.assert_series_equal(other, expected)

    def test_align_identical_different_object(self):
        # GH#51032 标识 GitHub 上的 issue 编号
        df = DataFrame({"a": [1, 2]})
        # 创建一个包含单列名为 "a" 的 DataFrame，数据为 [1, 2]
        ser = Series([3, 4])
        # 调用 DataFrame 的 align 方法，对 Series 进行对齐操作，指定轴向为行（axis=0）
        result, result2 = df.align(ser, axis=0)
        # 使用 pytest 的 assert 比较两个 DataFrame 是否相等
        tm.assert_frame_equal(result, df)
        # 使用 pytest 的 assert 比较两个 Series 是否相等
        tm.assert_series_equal(result2, ser)
        # 检查 df 和 result 是否为不同对象
        assert df is not result
        # 检查 ser 和 result2 是否为不同对象
        assert ser is not result2

    def test_align_identical_different_object_columns(self):
        # GH#51032 标识 GitHub 上的 issue 编号
        df = DataFrame({"a": [1, 2]})
        # 创建一个包含单列名为 "a" 的 DataFrame，数据为 [1, 2]
        ser = Series([1], index=["a"])
        # 调用 DataFrame 的 align 方法，对 Series 进行对齐操作，指定轴向为列（axis=1）
        result, result2 = df.align(ser, axis=1)
        # 使用 pytest 的 assert 比较两个 DataFrame 是否相等
        tm.assert_frame_equal(result, df)
        # 使用 pytest 的 assert 比较两个 Series 是否相等
        tm.assert_series_equal(result2, ser)
        # 检查 df 和 result 是否为不同对象
        assert df is not result
        # 检查 ser 和 result2 是否为不同对象
        assert ser is not result2
```