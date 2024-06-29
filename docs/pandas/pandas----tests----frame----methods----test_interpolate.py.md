# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_interpolate.py`

```
import numpy as np
import pytest

# 导入 pandas 库中的一些模块和类
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
    DataFrame,
    NaT,
    Series,
    date_range,
)
import pandas._testing as tm

# 定义一个测试类 TestDataFrameInterpolate
class TestDataFrameInterpolate:
    
    # 测试插值复杂数据
    def test_interpolate_complex(self):
        # GH#53635
        # 创建一个包含复数、NaN 和复数的 Series
        ser = Series([complex("1+1j"), float("nan"), complex("2+2j")])
        assert ser.dtype.kind == "c"  # 断言确认数据类型为复数类型
        
        # 对 Series 进行插值处理
        res = ser.interpolate()
        expected = Series([ser[0], ser[0] * 1.5, ser[2]])  # 期望的插值结果
        tm.assert_series_equal(res, expected)  # 断言插值结果与期望结果一致
        
        # 将 Series 转换为 DataFrame，再进行插值处理
        df = ser.to_frame()
        res = df.interpolate()
        expected = expected.to_frame()
        tm.assert_frame_equal(res, expected)  # 断言插值后的 DataFrame 与期望的 DataFrame 一致

    # 测试插值日期时间数据
    def test_interpolate_datetimelike_values(self, frame_or_series):
        # GH#11312, GH#51005
        orig = Series(date_range("2012-01-01", periods=5))  # 创建一个日期时间序列
        ser = orig.copy()
        ser[2] = NaT  # 将第二个位置的值设为 NaT (Not a Time)

        # 对序列进行插值处理
        res = frame_or_series(ser).interpolate()
        expected = frame_or_series(orig)
        tm.assert_equal(res, expected)  # 断言插值结果与期望结果一致

        # 将序列转换为带时区信息的序列进行插值处理
        ser_tz = ser.dt.tz_localize("US/Pacific")
        res_tz = frame_or_series(ser_tz).interpolate()
        expected_tz = frame_or_series(orig.dt.tz_localize("US/Pacific"))
        tm.assert_equal(res_tz, expected_tz)  # 断言插值后的时区序列与期望的时区序列一致

        # 将序列转换为时间间隔序列进行插值处理
        ser_td = ser - ser[0]
        res_td = frame_or_series(ser_td).interpolate()
        expected_td = frame_or_series(orig - orig[0])
        tm.assert_equal(res_td, expected_td)  # 断言插值后的时间间隔序列与期望的时间间隔序列一致

    # 测试就地插值处理
    def test_interpolate_inplace(self, frame_or_series, request):
        # GH#44749
        obj = frame_or_series([1, np.nan, 2])  # 创建一个包含 NaN 的序列或 DataFrame
        orig = obj.values

        obj.interpolate(inplace=True)  # 就地对序列或 DataFrame 进行插值处理
        expected = frame_or_series([1, 1.5, 2])
        tm.assert_equal(obj, expected)  # 断言插值后的结果与期望结果一致

        # 检查是否真正就地操作
        assert np.shares_memory(orig, obj.values)
        assert orig.squeeze()[1] == 1.5

    @pytest.mark.xfail(
        using_pyarrow_string_dtype(), reason="interpolate doesn't work for string"
    )
    # 测试基本插值功能（预期失败的测试用例）
    def test_interp_basic(self):
        df = DataFrame(
            {
                "A": [1, 2, np.nan, 4],
                "B": [1, 4, 9, np.nan],
                "C": [1, 2, 3, 5],
                "D": list("abcd"),
            }
        )
        msg = "DataFrame cannot interpolate with object dtype"
        with pytest.raises(TypeError, match=msg):
            df.interpolate()  # 预期插值操作会引发 TypeError 异常

        cvalues = df["C"]._values
        dvalues = df["D"].values
        with pytest.raises(TypeError, match=msg):
            df.interpolate(inplace=True)  # 预期就地插值操作会引发 TypeError 异常

        # 检查是否真正就地操作
        assert np.shares_memory(df["C"]._values, cvalues)
        assert np.shares_memory(df["D"]._values, dvalues)
    # 定义一个测试方法，测试在非范围索引条件下的基本插值操作
    def test_interp_basic_with_non_range_index(self, using_infer_string):
        # 创建一个包含不同数据类型列的 DataFrame
        df = DataFrame(
            {
                "A": [1, 2, np.nan, 4],
                "B": [1, 4, 9, np.nan],
                "C": [1, 2, 3, 5],
                "D": list("abcd"),
            }
        )

        # 错误消息，指出DataFrame在对象数据类型上无法插值
        msg = "DataFrame cannot interpolate with object dtype"
        # 如果不使用infer_string，则应该引发 TypeError 异常并匹配给定消息
        if not using_infer_string:
            with pytest.raises(TypeError, match=msg):
                df.set_index("C").interpolate()
        else:
            # 否则，执行插值操作并验证结果
            result = df.set_index("C").interpolate()
            expected = df.set_index("C")
            expected.loc[3, "A"] = 2.66667  # 预期在索引3处的A列值
            expected.loc[5, "B"] = 9  # 预期在索引5处的B列值
            tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，测试空 DataFrame 的插值操作
    def test_interp_empty(self):
        # 创建一个空的 DataFrame
        df = DataFrame()
        # 执行插值操作
        result = df.interpolate()
        # 确保结果不是原始 DataFrame 的引用
        assert result is not df
        expected = df  # 预期结果是原始的空 DataFrame
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，测试使用错误方法参数时的插值操作
    def test_interp_bad_method(self):
        # 创建一个包含不同数据类型列的 DataFrame
        df = DataFrame(
            {
                "A": [1, 2, np.nan, 4],
                "B": [1, 4, 9, np.nan],
                "C": [1, 2, 3, 5],
            }
        )
        # 错误消息，指出不能使用指定的方法进行插值
        msg = "Can not interpolate with method=not_a_method"
        # 应该引发 ValueError 异常并匹配给定消息
        with pytest.raises(ValueError, match=msg):
            df.interpolate(method="not_a_method")

    # 定义一个测试方法，测试单列插值操作
    def test_interp_combo(self):
        # 创建一个包含不同数据类型列的 DataFrame
        df = DataFrame(
            {
                "A": [1.0, 2.0, np.nan, 4.0],
                "B": [1, 4, 9, np.nan],
                "C": [1, 2, 3, 5],
                "D": list("abcd"),
            }
        )

        # 执行对列'A'的插值操作
        result = df["A"].interpolate()
        # 预期结果是一个Series，包含特定插值后的值
        expected = Series([1.0, 2.0, 3.0, 4.0], name="A")
        tm.assert_series_equal(result, expected)

    # 定义一个测试方法，测试在索引列包含NaN值时的插值操作
    def test_interp_nan_idx(self):
        # 创建一个包含NaN值的索引列的 DataFrame
        df = DataFrame({"A": [1, 2, np.nan, 4], "B": [np.nan, 2, 3, 4]})
        df = df.set_index("A")
        # 错误消息，指出索引列包含NaN时无法进行插值操作
        msg = (
            "Interpolation with NaNs in the index has not been implemented. "
            "Try filling those NaNs before interpolating."
        )
        # 应该引发 NotImplementedError 异常并匹配给定消息
        with pytest.raises(NotImplementedError, match=msg):
            df.interpolate(method="values")
    # 测试不同的插值方法是否能正常导入 scipy 库
    def test_interp_various(self):
        pytest.importorskip("scipy")
        # 创建一个包含空值的 DataFrame，列'A'包含 NaN 值
        df = DataFrame(
            {"A": [1, 2, np.nan, 4, 5, np.nan, 7], "C": [1, 2, 3, 5, 8, 13, 21]}
        )
        # 将列'C'设置为索引
        df = df.set_index("C")
        # 复制预期的 DataFrame 结果
        expected = df.copy()
        # 使用多项式插值法（一阶）填充 DataFrame，并将结果存储在变量 result 中
        result = df.interpolate(method="polynomial", order=1)

        # 更新预期结果中特定位置的值，用于后续的比较
        expected.loc[3, "A"] = 2.66666667
        expected.loc[13, "A"] = 5.76923076
        # 比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # 使用三次样条插值法填充 DataFrame
        result = df.interpolate(method="cubic")
        # GH #15662. 注释，但没有实际修改

        # 更新预期结果中特定位置的值，用于后续的比较
        expected.loc[3, "A"] = 2.81547781
        expected.loc[13, "A"] = 5.52964175
        # 比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # 使用最近邻插值法填充 DataFrame
        result = df.interpolate(method="nearest")
        # 更新预期结果中特定位置的值，用于后续的比较
        expected.loc[3, "A"] = 2
        expected.loc[13, "A"] = 5
        # 比较 result 和 expected 是否相等，不检查数据类型
        tm.assert_frame_equal(result, expected, check_dtype=False)

        # 使用二次插值法填充 DataFrame
        result = df.interpolate(method="quadratic")
        # 更新预期结果中特定位置的值，用于后续的比较
        expected.loc[3, "A"] = 2.82150771
        expected.loc[13, "A"] = 6.12648668
        # 比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # 使用线性插值法填充 DataFrame
        result = df.interpolate(method="slinear")
        # 更新预期结果中特定位置的值，用于后续的比较
        expected.loc[3, "A"] = 2.66666667
        expected.loc[13, "A"] = 5.76923077
        # 比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # 使用零值填充 DataFrame
        result = df.interpolate(method="zero")
        # 更新预期结果中特定位置的值，用于后续的比较
        expected.loc[3, "A"] = 2.0
        expected.loc[13, "A"] = 5
        # 比较 result 和 expected 是否相等，不检查数据类型
        tm.assert_frame_equal(result, expected, check_dtype=False)

    # 测试使用 scipy 库的备选插值方法
    def test_interp_alt_scipy(self):
        pytest.importorskip("scipy")
        # 创建一个包含空值的 DataFrame
        df = DataFrame(
            {"A": [1, 2, np.nan, 4, 5, np.nan, 7], "C": [1, 2, 3, 5, 8, 13, 21]}
        )
        # 使用巴氏中心插值法填充 DataFrame
        result = df.interpolate(method="barycentric")
        # 复制预期的 DataFrame 结果
        expected = df.copy()
        # 更新预期结果中特定位置的值，用于后续的比较
        expected.loc[2, "A"] = 3
        expected.loc[5, "A"] = 6
        # 比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # 使用克罗格插值法填充 DataFrame
        result = df.interpolate(method="krogh")
        # 复制预期的 DataFrame 结果，但只更新'A'列
        expectedk = df.copy()
        expectedk["A"] = expected["A"]
        # 比较 result 和 expectedk 是否相等
        tm.assert_frame_equal(result, expectedk)

        # 使用 PCHIP 插值法填充 DataFrame
        result = df.interpolate(method="pchip")
        # 更新预期结果中特定位置的值，用于后续的比较
        expected.loc[2, "A"] = 3
        expected.loc[5, "A"] = 6.0
        # 比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 测试行向插值方法
    def test_interp_rowwise(self):
        # 创建一个包含空值的 DataFrame
        df = DataFrame(
            {
                0: [1, 2, np.nan, 4],
                1: [2, 3, 4, np.nan],
                2: [np.nan, 4, 5, 6],
                3: [4, np.nan, 6, 7],
                4: [1, 2, 3, 4],
            }
        )
        # 使用行向插值法填充 DataFrame
        result = df.interpolate(axis=1)
        # 复制预期的 DataFrame 结果
        expected = df.copy()
        # 更新预期结果中特定位置的值，用于后续的比较
        expected.loc[3, 1] = 5
        expected.loc[0, 2] = 3
        expected.loc[1, 3] = 3
        # 将列 4 的数据类型转换为 float64
        expected[4] = expected[4].astype(np.float64)
        # 比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # 使用值插值法填充 DataFrame（沿轴 1）
        result = df.interpolate(axis=1, method="values")
        # 比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # 使用列向插值法填充 DataFrame
        result = df.interpolate(axis=0)
        # 生成预期结果（默认插值）
        expected = df.interpolate()
        # 比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
    @pytest.mark.parametrize(
        "axis_name, axis_number",
        [  # 使用 pytest 的参数化功能，定义了三组参数，分别测试不同的轴名称和数字
            pytest.param("rows", 0, id="rows_0"),  # 测试轴名称为 "rows" 对应数字为 0
            pytest.param("index", 0, id="index_0"),  # 测试轴名称为 "index" 对应数字为 0
            pytest.param("columns", 1, id="columns_1"),  # 测试轴名称为 "columns" 对应数字为 1
        ],
    )
    def test_interp_axis_names(self, axis_name, axis_number):
        # GH 29132: test axis names
        # 准备测试数据，一个包含 NaN 值的字典
        data = {0: [0, np.nan, 6], 1: [1, np.nan, 7], 2: [2, 5, 8]}
        
        # 创建 DataFrame 对象，数据类型为 np.float64
        df = DataFrame(data, dtype=np.float64)
        
        # 对 DataFrame 进行插值处理，使用指定的轴名称和插值方法 "linear"
        result = df.interpolate(axis=axis_name, method="linear")
        
        # 根据给定的数字轴进行相同的插值处理
        expected = df.interpolate(axis=axis_number, method="linear")
        
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    def test_rowwise_alt(self):
        # 准备测试数据，一个包含 NaN 值的字典
        df = DataFrame(
            {
                0: [0, 0.5, 1.0, np.nan, 4, 8, np.nan, np.nan, 64],
                1: [1, 2, 3, 4, 3, 2, 1, 0, -1],
            }
        )
        
        # 对 DataFrame 进行行向插值处理，填充 NaN 值
        df.interpolate(axis=0)
        # TODO: assert something? -- 在此处可能需要添加断言以验证插值的结果

    @pytest.mark.parametrize(
        "check_scipy", [False, pytest.param(True, marks=td.skip_if_no("scipy"))]
    )
    def test_interp_leading_nans(self, check_scipy):
        # 准备测试数据，一个包含 NaN 值的字典
        df = DataFrame(
            {"A": [np.nan, np.nan, 0.5, 0.25, 0], "B": [np.nan, -3, -3.5, np.nan, -4]}
        )
        
        # 对 DataFrame 进行插值处理，填充 NaN 值
        result = df.interpolate()
        
        # 创建一个期望的 DataFrame，填充了 NaN 值的位置使用了预期的插值结果
        expected = df.copy()
        expected.loc[3, "B"] = -3.75
        
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)
        
        # 如果 check_scipy 为 True，则使用多项式插值方法进行额外的测试
        if check_scipy:
            result = df.interpolate(method="polynomial", order=1)
            tm.assert_frame_equal(result, expected)

    def test_interp_raise_on_only_mixed(self, axis):
        # 准备测试数据，一个包含不同数据类型的字典
        df = DataFrame(
            {
                "A": [1, 2, np.nan, 4],
                "B": ["a", "b", "c", "d"],
                "C": [np.nan, 2, 5, 7],
                "D": [np.nan, np.nan, 9, 9],
                "E": [1, 2, 3, 4],
            }
        )
        
        # 准备错误消息，用于断言插值过程中会引发 TypeError 异常
        msg = "DataFrame cannot interpolate with object dtype"
        
        # 使用 pytest 的断言检查插值操作是否会引发预期的 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            df.astype("object").interpolate(axis=axis)

    def test_interp_raise_on_all_object_dtype(self):
        # GH 22985
        # 准备测试数据，一个完全由对象类型组成的 DataFrame
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, dtype="object")
        
        # 准备错误消息，用于断言插值过程中会引发 TypeError 异常
        msg = "DataFrame cannot interpolate with object dtype"
        
        # 使用 pytest 的断言检查插值操作是否会引发预期的 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            df.interpolate()

    def test_interp_inplace(self):
        # 准备测试数据，一个包含 NaN 值的 DataFrame
        df = DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
        
        # 复制 DataFrame 以备后续比较
        expected = df.copy()
        result = df.copy()

        # 使用 tm 库捕获预期的 chained assignment 错误
        with tm.raises_chained_assignment_error():
            # 调用插值方法，并设置 inplace=True
            return_value = result["a"].interpolate(inplace=True)
        
        # 检查返回值是否为 None
        assert return_value is None
        
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)
    def test_interp_inplace_row(self):
        # GH 10395
        # 创建一个包含特定数据的DataFrame对象
        result = DataFrame(
            {"a": [1.0, 2.0, 3.0, 4.0], "b": [np.nan, 2.0, 3.0, 4.0], "c": [3, 2, 2, 2]}
        )
        # 使用线性插值法在行方向上插值，返回新的DataFrame对象
        expected = result.interpolate(method="linear", axis=1, inplace=False)
        # 在行方向上进行原地插值，返回值为None
        return_value = result.interpolate(method="linear", axis=1, inplace=True)
        # 断言返回值为None
        assert return_value is None
        # 断言result与expected相等
        tm.assert_frame_equal(result, expected)

    def test_interp_ignore_all_good(self):
        # GH
        # 创建包含特定数据的DataFrame对象
        df = DataFrame(
            {
                "A": [1, 2, np.nan, 4],
                "B": [1, 2, 3, 4],
                "C": [1.0, 2.0, np.nan, 4.0],
                "D": [1.0, 2.0, 3.0, 4.0],
            }
        )
        # 创建期望的DataFrame对象，使用插值填充缺失值
        expected = DataFrame(
            {
                "A": np.array([1, 2, 3, 4], dtype="float64"),
                "B": np.array([1, 2, 3, 4], dtype="int64"),
                "C": np.array([1.0, 2.0, 3, 4.0], dtype="float64"),
                "D": np.array([1.0, 2.0, 3.0, 4.0], dtype="float64"),
            }
        )
        # 对DataFrame对象进行插值填充，返回结果DataFrame对象
        result = df.interpolate()
        # 断言result与expected相等
        tm.assert_frame_equal(result, expected)

        # 对"B"和"D"列进行插值填充，断言结果与原始DataFrame对象相等
        result = df[["B", "D"]].interpolate()
        tm.assert_frame_equal(result, df[["B", "D"]])

    def test_interp_time_inplace_axis(self):
        # GH 9687
        # 设定时间段数
        periods = 5
        # 生成日期范围作为索引
        idx = date_range(start="2014-01-01", periods=periods)
        # 生成随机数据矩阵，含有NaN值
        data = np.random.default_rng(2).random((periods, periods))
        data[data < 0.5] = np.nan
        # 创建期望的DataFrame对象，使用时间插值方法填充NaN值
        expected = DataFrame(index=idx, columns=idx, data=data)

        # 在列方向上进行时间插值填充，返回新的DataFrame对象
        result = expected.interpolate(axis=0, method="time")
        # 在列方向上进行原地时间插值填充，返回值为None
        return_value = expected.interpolate(axis=0, method="time", inplace=True)
        # 断言返回值为None
        assert return_value is None
        # 断言result与expected相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("axis_name, axis_number", [("index", 0), ("columns", 1)])
    def test_interp_string_axis(self, axis_name, axis_number):
        # https://github.com/pandas-dev/pandas/issues/25190
        # 生成一系列的x和y数据
        x = np.linspace(0, 100, 1000)
        y = np.sin(x)
        # 创建DataFrame对象，包含重复y数据的矩阵，并重新索引列
        df = DataFrame(
            data=np.tile(y, (10, 1)), index=np.arange(10), columns=x
        ).reindex(columns=x * 1.005)
        # 对DataFrame对象进行线性插值填充，根据给定的轴名称或轴编号
        result = df.interpolate(method="linear", axis=axis_name)
        # 对DataFrame对象进行线性插值填充，根据给定的轴编号
        expected = df.interpolate(method="linear", axis=axis_number)
        # 断言result与expected相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("multiblock", [True, False])
    @pytest.mark.parametrize("method", ["ffill", "bfill", "pad"])
    # 定义一个测试函数，用于测试插值和填充方法的功能
    def test_interp_fillna_methods(self, axis, multiblock, method):
        # 创建一个包含列"A", "B", "C"的DataFrame，其中包括NaN值
        df = DataFrame(
            {
                "A": [1.0, 2.0, 3.0, 4.0, np.nan, 5.0],
                "B": [2.0, 4.0, 6.0, np.nan, 8.0, 10.0],
                "C": [3.0, 6.0, 9.0, np.nan, np.nan, 30.0],
            }
        )
        # 如果参数multiblock为True，则添加额外的列"D"和"E"
        if multiblock:
            df["D"] = np.nan
            df["E"] = 1.0

        # 设置错误消息，用于检查是否会引发值错误
        msg = f"Can not interpolate with method={method}"
        # 使用pytest的断言检查是否引发值错误，并匹配指定的错误消息
        with pytest.raises(ValueError, match=msg):
            df.interpolate(method=method, axis=axis)

    # 定义一个测试函数，用于测试对空DataFrame进行插值的情况
    def test_interpolate_empty_df(self):
        # 创建一个空DataFrame
        df = DataFrame()
        # 创建一个预期结果DataFrame，与原始DataFrame相同
        expected = df.copy()
        # 在原地对空DataFrame进行插值
        result = df.interpolate(inplace=True)
        # 使用pytest断言检查插值结果返回None，并且原始DataFrame与预期DataFrame相等
        assert result is None
        tm.assert_frame_equal(df, expected)

    # 定义一个测试函数，用于测试整数类型的DataFrame列的插值情况
    def test_interpolate_ea(self, any_int_ea_dtype):
        # 创建一个包含整数和NaN值的DataFrame，指定数据类型为any_int_ea_dtype
        df = DataFrame({"a": [1, None, None, None, 3]}, dtype=any_int_ea_dtype)
        # 复制原始DataFrame以备后用
        orig = df.copy()
        # 对DataFrame进行插值，设置插值限制为2
        result = df.interpolate(limit=2)
        # 创建预期的DataFrame，包含插值后的结果，数据类型为"Float64"
        expected = DataFrame({"a": [1, 1.5, 2.0, None, 3]}, dtype="Float64")
        # 使用pytest断言检查插值后的结果与预期结果相等，并且原始DataFrame未被修改
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(df, orig)

    # 使用pytest的参数化标记，定义一个测试函数，测试浮点数类型DataFrame列的插值情况
    @pytest.mark.parametrize(
        "dtype",
        [
            "Float64",
            "Float32",
            pytest.param("float32[pyarrow]", marks=td.skip_if_no("pyarrow")),
            pytest.param("float64[pyarrow]", marks=td.skip_if_no("pyarrow")),
        ],
    )
    def test_interpolate_ea_float(self, dtype):
        # 创建一个包含浮点数和NaN值的DataFrame，数据类型由参数dtype指定
        df = DataFrame({"a": [1, None, None, None, 3]}, dtype=dtype)
        # 复制原始DataFrame以备后用
        orig = df.copy()
        # 对DataFrame进行插值，设置插值限制为2
        result = df.interpolate(limit=2)
        # 创建预期的DataFrame，包含插值后的结果，数据类型与原始DataFrame相同
        expected = DataFrame({"a": [1, 1.5, 2.0, None, 3]}, dtype=dtype)
        # 使用pytest断言检查插值后的结果与预期结果相等，并且原始DataFrame未被修改
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(df, orig)

    # 使用pytest的参数化标记，定义一个测试函数，测试使用Arrow数据类型进行插值的情况
    @pytest.mark.parametrize(
        "dtype",
        ["int64", "uint64", "int32", "int16", "int8", "uint32", "uint16", "uint8"],
    )
    def test_interpolate_arrow(self, dtype):
        # 导入pyarrow模块，如果导入失败则跳过测试
        pytest.importorskip("pyarrow")
        # 创建一个包含整数和NaN值的DataFrame，数据类型为dtype + "[pyarrow]"
        df = DataFrame({"a": [1, None, None, None, 3]}, dtype=dtype + "[pyarrow]")
        # 对DataFrame进行插值，设置插值限制为2
        result = df.interpolate(limit=2)
        # 创建预期的DataFrame，包含插值后的结果，数据类型为"float64[pyarrow]"
        expected = DataFrame({"a": [1, 1.5, 2.0, None, 3]}, dtype="float64[pyarrow]")
        # 使用pytest断言检查插值后的结果与预期结果相等
        tm.assert_frame_equal(result, expected)
```