# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_clip.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from pandas import (  # 从 pandas 库中导入 DataFrame 和 Series 类
    DataFrame,
    Series,
)
import pandas._testing as tm  # 导入 pandas 内部测试工具模块 tm


class TestDataFrameClip:
    def test_clip(self, float_frame):
        median = float_frame.median().median()  # 计算浮点数 DataFrame 的中位数的中位数
        original = float_frame.copy()  # 复制原始的浮点数 DataFrame

        double = float_frame.clip(upper=median, lower=median)  # 将浮点数 DataFrame 的值剪切到中位数的范围内
        assert not (double.values != median).any()  # 断言剪切后的值是否全为中位数

        # 验证 float_frame 是否没有原地修改
        assert (float_frame.values == original.values).all()  # 断言浮点数 DataFrame 的值是否与原始值相同

    def test_inplace_clip(self, float_frame):
        # GH#15388
        median = float_frame.median().median()  # 计算浮点数 DataFrame 的中位数的中位数
        frame_copy = float_frame.copy()  # 复制原始的浮点数 DataFrame

        return_value = frame_copy.clip(upper=median, lower=median, inplace=True)  # 在原地将值剪切到中位数的范围内
        assert return_value is None  # 断言 inplace 操作返回值为 None
        assert not (frame_copy.values != median).any()  # 断言剪切后的值是否全为中位数

    def test_dataframe_clip(self):
        # GH#2747
        df = DataFrame(np.random.default_rng(2).standard_normal((1000, 2)))  # 创建一个随机数填充的 DataFrame

        for lb, ub in [(-1, 1), (1, -1)]:
            clipped_df = df.clip(lb, ub)  # 将 DataFrame 的值剪切到指定的上下界

            lb, ub = min(lb, ub), max(ub, lb)
            lb_mask = df.values <= lb  # 创建下界掩码
            ub_mask = df.values >= ub  # 创建上界掩码
            mask = ~lb_mask & ~ub_mask  # 创建既不是下界也不是上界的掩码
            assert (clipped_df.values[lb_mask] == lb).all()  # 断言剪切后下界的值是否符合预期
            assert (clipped_df.values[ub_mask] == ub).all()  # 断言剪切后上界的值是否符合预期
            assert (clipped_df.values[mask] == df.values[mask]).all()  # 断言剪切后其他值是否与原始 DataFrame 相同

    def test_clip_mixed_numeric(self):
        # clip on mixed integer or floats
        # GH#24162, clipping now preserves numeric types per column
        df = DataFrame({"A": [1, 2, 3], "B": [1.0, np.nan, 3.0]})  # 创建一个包含混合整数和浮点数的 DataFrame
        result = df.clip(1, 2)  # 将 DataFrame 的值剪切到指定的上下界
        expected = DataFrame({"A": [1, 2, 2], "B": [1.0, np.nan, 2.0]})  # 预期的剪切后的 DataFrame
        tm.assert_frame_equal(result, expected)  # 使用测试工具 tm 断言结果与预期相等

        df = DataFrame([[1, 2, 3.4], [3, 4, 5.6]], columns=["foo", "bar", "baz"])  # 创建一个指定列名的 DataFrame
        expected = df.dtypes  # 获取预期的数据类型
        result = df.clip(upper=3).dtypes  # 将 DataFrame 的值剪切到指定的上界，并获取剪切后的数据类型
        tm.assert_series_equal(result, expected)  # 使用测试工具 tm 断言结果的数据类型与预期相等

    @pytest.mark.parametrize("inplace", [True, False])
    def test_clip_against_series(self, inplace):
        # GH#6966

        df = DataFrame(np.random.default_rng(2).standard_normal((1000, 2)))  # 创建一个随机数填充的 DataFrame
        lb = Series(np.random.default_rng(2).standard_normal(1000))  # 创建一个随机数填充的 Series 作为下界
        ub = lb + 1  # 创建一个上界 Series

        original = df.copy()  # 复制原始的 DataFrame
        clipped_df = df.clip(lb, ub, axis=0, inplace=inplace)  # 在原地将 DataFrame 的值剪切到指定的上下界

        if inplace:
            clipped_df = df  # 如果 inplace 为 True，则更新 clipped_df 为剪切后的 DataFrame

        for i in range(2):
            lb_mask = original.iloc[:, i] <= lb  # 创建下界掩码
            ub_mask = original.iloc[:, i] >= ub  # 创建上界掩码
            mask = ~lb_mask & ~ub_mask  # 创建既不是下界也不是上界的掩码

            result = clipped_df.loc[lb_mask, i]  # 获取剪切后下界的 Series
            tm.assert_series_equal(result, lb[lb_mask], check_names=False)  # 使用测试工具 tm 断言结果与预期相等，不检查名称
            assert result.name == i  # 断言结果的名称是否正确

            result = clipped_df.loc[ub_mask, i]  # 获取剪切后上界的 Series
            tm.assert_series_equal(result, ub[ub_mask], check_names=False)  # 使用测试工具 tm 断言结果与预期相等，不检查名称
            assert result.name == i  # 断言结果的名称是否正确

            tm.assert_series_equal(clipped_df.loc[mask, i], df.loc[mask, i])  # 使用测试工具 tm 断言剪切后其他值是否与原始 DataFrame 相同
    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize("lower", [[2, 3, 4], np.asarray([2, 3, 4])])
    @pytest.mark.parametrize(
        "axis,res",
        [
            (0, [[2.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 7.0, 7.0]]),
            (1, [[2.0, 3.0, 4.0], [4.0, 5.0, 6.0], [5.0, 6.0, 7.0]]),
        ],
    )
    def test_clip_against_list_like(self, inplace, lower, axis, res):
        # 测试用例，用于验证 DataFrame 的 clip 方法对列表类型进行裁剪的行为
        # GH#15390
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    
        # 创建一个 DataFrame 对象，用于测试的原始数据
        original = DataFrame(
            arr, columns=["one", "two", "three"], index=["a", "b", "c"]
        )
    
        # 调用 DataFrame 的 clip 方法，生成裁剪后的结果
        result = original.clip(lower=lower, upper=[5, 6, 7], axis=axis, inplace=inplace)
    
        # 创建预期的 DataFrame 对象，用于和实际结果比较
        expected = DataFrame(res, columns=original.columns, index=original.index)
        if inplace:
            # 如果 inplace 为 True，则返回原始对象
            result = original
        # 使用 assert_frame_equal 方法比较实际结果和预期结果，确保一致性
        tm.assert_frame_equal(result, expected, check_exact=True)
    
    @pytest.mark.parametrize("axis", [0, 1, None])
    def test_clip_against_frame(self, axis):
        # 测试用例，用于验证 DataFrame 的 clip 方法对整个 DataFrame 进行裁剪的行为
    
        # 创建一个包含随机数据的 DataFrame 对象
        df = DataFrame(np.random.default_rng(2).standard_normal((1000, 2)))
        lb = DataFrame(np.random.default_rng(2).standard_normal((1000, 2)))
        ub = lb + 1
    
        # 调用 DataFrame 的 clip 方法，生成裁剪后的结果
        clipped_df = df.clip(lb, ub, axis=axis)
    
        # 创建不同的掩码，用于比较裁剪后的结果与预期值
        lb_mask = df <= lb
        ub_mask = df >= ub
        mask = ~lb_mask & ~ub_mask
    
        # 使用 assert_frame_equal 方法分别比较裁剪结果和预期结果的一致性
        tm.assert_frame_equal(clipped_df[lb_mask], lb[lb_mask])
        tm.assert_frame_equal(clipped_df[ub_mask], ub[ub_mask])
        tm.assert_frame_equal(clipped_df[mask], df[mask])
    
    def test_clip_against_unordered_columns(self):
        # 测试用例，用于验证 DataFrame 的 clip 方法在存在无序列的列名时的行为
        # GH#20911
    
        # 创建多个包含随机数据的 DataFrame 对象，列名无序
        df1 = DataFrame(
            np.random.default_rng(2).standard_normal((1000, 4)),
            columns=["A", "B", "C", "D"],
        )
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((1000, 4)),
            columns=["D", "A", "B", "C"],
        )
        df3 = DataFrame(df2.values - 1, columns=["B", "D", "C", "A"])
    
        # 使用不同的参数调用 DataFrame 的 clip 方法，生成裁剪后的结果
        result_upper = df1.clip(lower=0, upper=df2)
        expected_upper = df1.clip(lower=0, upper=df2[df1.columns])
        result_lower = df1.clip(lower=df3, upper=3)
        expected_lower = df1.clip(lower=df3[df1.columns], upper=3)
        result_lower_upper = df1.clip(lower=df3, upper=df2)
        expected_lower_upper = df1.clip(lower=df3[df1.columns], upper=df2[df1.columns])
    
        # 使用 assert_frame_equal 方法比较裁剪结果和预期结果的一致性
        tm.assert_frame_equal(result_upper, expected_upper)
        tm.assert_frame_equal(result_lower, expected_lower)
        tm.assert_frame_equal(result_lower_upper, expected_lower_upper)
    def test_clip_with_na_args(self, float_frame):
        """Should process np.nan argument as None"""
        # GH#17276
        # 使用 tm.assert_frame_equal 函数比较 float_frame 调用 clip 方法后的结果和原始 float_frame 是否相等
        tm.assert_frame_equal(float_frame.clip(np.nan), float_frame)
        # 使用 tm.assert_frame_equal 函数比较 float_frame 调用 clip 方法（指定上下界为 np.nan）后的结果和原始 float_frame 是否相等
        tm.assert_frame_equal(float_frame.clip(upper=np.nan, lower=np.nan), float_frame)

        # GH#19992 and adjusted in GH#40420
        # 创建一个 DataFrame 对象 df，包含三列，每列对应一个字典中的键值对
        df = DataFrame({"col_0": [1, 2, 3], "col_1": [4, 5, 6], "col_2": [7, 8, 9]})

        # 对 df 应用 clip 方法，指定 lower 参数为 [4, 5, np.nan]，axis 参数为 0
        result = df.clip(lower=[4, 5, np.nan], axis=0)
        # 创建期望的 DataFrame 对象 expected，包含与 result 对象相同结构的数据，但是 col_0 列的第一行数值替换为 4，数据类型为 float
        expected = DataFrame(
            {
                "col_0": Series([4, 5, 3], dtype="float"),
                "col_1": [4, 5, 6],
                "col_2": [7, 8, 9],
            }
        )
        # 使用 tm.assert_frame_equal 函数比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # 对 df 应用 clip 方法，指定 lower 参数为 [4, 5, np.nan]，axis 参数为 1
        result = df.clip(lower=[4, 5, np.nan], axis=1)
        # 创建期望的 DataFrame 对象 expected，包含与 result 对象相同结构的数据，但是 col_0 列的每一行的值替换为 4
        expected = DataFrame(
            {"col_0": [4, 4, 4], "col_1": [5, 5, 6], "col_2": [7, 8, 9]}
        )
        # 使用 tm.assert_frame_equal 函数比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # GH#40420
        # 创建一个字典 data，包含两列数据
        data = {"col_0": [9, -3, 0, -1, 5], "col_1": [-2, -7, 6, 8, -5]}
        # 创建 DataFrame 对象 df，使用 data 字典作为数据源
        df = DataFrame(data)
        # 创建一个 Series 对象 t，包含数值和一个 np.nan 值
        t = Series([2, -4, np.nan, 6, 3])
        # 对 df 应用 clip 方法，指定 lower 参数为 t，axis 参数为 0
        result = df.clip(lower=t, axis=0)
        # 创建期望的 DataFrame 对象 expected，包含与 result 对象相同结构的数据，但是 col_0 列的第四行的值替换为 6，col_1 列的第三行的值替换为 6，数据类型为 float
        expected = DataFrame(
            {"col_0": [9, -3, 0, 6, 5], "col_1": [2, -4, 6, 8, 3]}, dtype="float"
        )
        # 使用 tm.assert_frame_equal 函数比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    def test_clip_int_data_with_float_bound(self):
        # GH51472
        # 创建一个 DataFrame 对象 df，包含一列数据 a，数据为 [1, 2, 3]
        df = DataFrame({"a": [1, 2, 3]})
        # 对 df 应用 clip 方法，指定 lower 参数为 1.5
        result = df.clip(lower=1.5)
        # 创建期望的 DataFrame 对象 expected，包含一列数据 a，数据为 [1.5, 2.0, 3.0]
        expected = DataFrame({"a": [1.5, 2.0, 3.0]})
        # 使用 tm.assert_frame_equal 函数比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    def test_clip_with_list_bound(self):
        # GH#54817
        # 创建一个 DataFrame 对象 df，包含一列数据 [1, 5]
        df = DataFrame([1, 5])
        # 创建期望的 DataFrame 对象 expected，包含一列数据 [3, 5]
        expected = DataFrame([3, 5])
        # 对 df 应用 clip 方法，指定 lower 参数为 [3]
        result = df.clip([3])
        # 使用 tm.assert_frame_equal 函数比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # 创建期望的 DataFrame 对象 expected，包含一列数据 [1, 3]
        expected = DataFrame([1, 3])
        # 对 df 应用 clip 方法，指定 upper 参数为 [3]
        result = df.clip(upper=[3])
        # 使用 tm.assert_frame_equal 函数比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
```