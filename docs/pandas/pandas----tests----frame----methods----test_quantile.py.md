# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_quantile.py`

```
# 导入 numpy 库，简写为 np
import numpy as np
# 导入 pytest 库
import pytest

# 导入 pandas 库，并分别导入 DataFrame, Index, Series, Timestamp 这几个类
import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    Timestamp,
)
# 导入 pandas 内部测试模块
import pandas._testing as tm

# 定义装饰器 fixture interp_method，返回参数为 [["linear", "single"], ["nearest", "table"]] 的两个元素
@pytest.fixture(
    params=[["linear", "single"], ["nearest", "table"]], ids=lambda x: "-".join(x)
)
def interp_method(request):
    """(interpolation, method) arguments for quantile"""
    return request.param

# 定义 TestDataFrameQuantile 类
class TestDataFrameQuantile:
    # 使用 pytest.mark.parametrize 装饰器标记的测试方法，测试稀疏数据框 quantile 方法
    @pytest.mark.parametrize(
        "df,expected",
        [
            [
                DataFrame(
                    {
                        0: Series(pd.arrays.SparseArray([1, 2])),
                        1: Series(pd.arrays.SparseArray([3, 4])),
                    }
                ),
                Series([1.5, 3.5], name=0.5),
            ],
            [
                DataFrame(Series([0.0, None, 1.0, 2.0], dtype="Sparse[float]")),
                Series([1.0], name=0.5),
            ],
        ],
    )
    def test_quantile_sparse(self, df, expected):
        # GH#17198
        # GH#24600
        # 调用 DataFrame 的 quantile 方法计算结果
        result = df.quantile()
        # 将期望的结果转换为 Sparse[float] 类型
        expected = expected.astype("Sparse[float]")
        # 使用测试模块中的 assert_series_equal 方法比较结果和期望
        tm.assert_series_equal(result, expected)

    # 测试 quantile 方法的普通情况
    def test_quantile(self, datetime_frame, interp_method, request):
        # 从 interp_method 参数中获取 interpolation 和 method
        interpolation, method = interp_method
        # 从参数 datetime_frame 获取数据框 df
        df = datetime_frame
        # 调用 df 的 quantile 方法，计算结果
        result = df.quantile(
            0.1, axis=0, numeric_only=True, interpolation=interpolation, method=method
        )
        # 创建期望的 Series 结果，包括每列的 10th 百分位数
        expected = Series(
            [np.percentile(df[col], 10) for col in df.columns],
            index=df.columns,
            name=0.1,
        )
        # 如果插值方式为 "linear"，则使用 assert_series_equal 方法比较 result 和 expected
        if interpolation == "linear":
            # np.percentile 的值只能与线性插值进行比较
            tm.assert_series_equal(result, expected)
        else:
            # 否则，比较 result 和 expected 的索引
            tm.assert_index_equal(result.index, expected.index)
            # 确保 result 的名称与 expected 的名称相同
            assert result.name == expected.name

        # 继续测试另一种轴向的 quantile 方法
        result = df.quantile(
            0.9, axis=1, numeric_only=True, interpolation=interpolation, method=method
        )
        # 创建期望的 Series 结果，包括每行的 90th 百分位数
        expected = Series(
            [np.percentile(df.loc[date], 90) for date in df.index],
            index=df.index,
            name=0.9,
        )
        # 如果插值方式为 "linear"，则使用 assert_series_equal 方法比较 result 和 expected
        if interpolation == "linear":
            # np.percentile 的值只能与线性插值进行比较
            tm.assert_series_equal(result, expected)
        else:
            # 否则，比较 result 和 expected 的索引
            tm.assert_index_equal(result.index, expected.index)
            # 确保 result 的名称与 expected 的名称相同
            assert result.name == expected.name

    # 测试空数据框的 quantile 方法
    def test_empty(self, interp_method):
        # 从 interp_method 参数中获取 interpolation 和 method
        interpolation, method = interp_method
        # 创建一个空的 DataFrame，并调用 quantile 方法
        q = DataFrame({"x": [], "y": []}).quantile(
            0.1, axis=0, numeric_only=True, interpolation=interpolation, method=method
        )
        # 确保返回的结果为 NaN
        assert np.isnan(q["x"]) and np.isnan(q["y"])
    def test_non_numeric_exclusion(self, interp_method, request):
        # 对于测试非数值排除功能的函数
        interpolation, method = interp_method
        # 创建一个包含列"col1"和"col2"的DataFrame对象
        df = DataFrame({"col1": ["A", "A", "B", "B"], "col2": [1, 2, 3, 4]})
        # 计算DataFrame的中位数，并指定使用插值方法和计算方法
        rs = df.quantile(
            0.5, numeric_only=True, interpolation=interpolation, method=method
        )
        # 通过计算中位数来获取期望的结果Series对象
        xp = df.median(numeric_only=True).rename(0.5)
        # 如果插值方法是"nearest"，则将期望的结果Series对象加上0.5并转换为np.int64类型
        if interpolation == "nearest":
            xp = (xp + 0.5).astype(np.int64)
        # 使用测试工具（tm）来比较计算结果rs和期望结果xp是否相等
        tm.assert_series_equal(rs, xp)

    def test_axis(self, interp_method):
        # 对于测试轴参数功能的函数
        interpolation, method = interp_method
        # 创建一个包含列"A"和"B"的DataFrame对象，并指定索引
        df = DataFrame({"A": [1, 2, 3], "B": [2, 3, 4]}, index=[1, 2, 3])
        # 计算DataFrame的指定分位数，并指定轴参数、插值方法和计算方法
        result = df.quantile(0.5, axis=1, interpolation=interpolation, method=method)
        # 创建期望的结果Series对象，指定索引名为0.5
        expected = Series([1.5, 2.5, 3.5], index=[1, 2, 3], name=0.5)
        # 如果插值方法是"nearest"，则将期望的结果Series对象转换为np.int64类型
        if interpolation == "nearest":
            expected = expected.astype(np.int64)
        # 使用测试工具（tm）来比较计算结果result和期望结果expected是否相等
        tm.assert_series_equal(result, expected)

        # 再次计算DataFrame的多个分位数，并指定轴参数、插值方法和计算方法
        result = df.quantile(
            [0.5, 0.75], axis=1, interpolation=interpolation, method=method
        )
        # 创建期望的结果DataFrame对象，指定列名和索引名
        expected = DataFrame(
            {1: [1.5, 1.75], 2: [2.5, 2.75], 3: [3.5, 3.75]}, index=[0.5, 0.75]
        )
        # 如果插值方法是"nearest"，则对期望的DataFrame对象进行特定的数值调整和转换为np.int64类型
        if interpolation == "nearest":
            expected.iloc[0, :] -= 0.5
            expected.iloc[1, :] += 0.25
            expected = expected.astype(np.int64)
        # 使用测试工具（tm）来比较计算结果result和期望结果expected是否相等，并检查索引类型
        tm.assert_frame_equal(result, expected, check_index_type=True)

    def test_axis_numeric_only_true(self, interp_method):
        # 对于测试在numeric_only=True条件下轴参数功能的函数
        # 可能将来会更改API以排除同一轴上的非数值
        # 详见GitHub issue #7312
        interpolation, method = interp_method
        # 创建一个包含不同类型元素的DataFrame对象
        df = DataFrame([[1, 2, 3], ["a", "b", 4]])
        # 计算DataFrame的指定分位数，并指定轴参数、numeric_only=True条件、插值方法和计算方法
        result = df.quantile(
            0.5, axis=1, numeric_only=True, interpolation=interpolation, method=method
        )
        # 创建期望的结果Series对象，指定索引名为0.5
        expected = Series([3.0, 4.0], index=[0, 1], name=0.5)
        # 如果插值方法是"nearest"，则将期望的结果Series对象转换为np.int64类型
        if interpolation == "nearest":
            expected = expected.astype(np.int64)
        # 使用测试工具（tm）来比较计算结果result和期望结果expected是否相等
        tm.assert_series_equal(result, expected)

    def test_quantile_date_range(self, interp_method):
        # GitHub issue 2460
        # 对于测试日期范围的分位数功能的函数
        interpolation, method = interp_method
        # 创建一个包含日期时间索引的Series对象
        dti = pd.date_range("2016-01-01", periods=3, tz="US/Pacific")
        ser = Series(dti)
        # 根据Series对象创建一个DataFrame对象
        df = DataFrame(ser)

        # 计算DataFrame的指定分位数，并指定是否包含非数值、插值方法和计算方法
        result = df.quantile(
            numeric_only=False, interpolation=interpolation, method=method
        )
        # 创建期望的结果Series对象，指定名称和数据类型
        expected = Series(
            ["2016-01-02 00:00:00"], name=0.5, dtype="datetime64[ns, US/Pacific]"
        )

        # 使用测试工具（tm）来比较计算结果result和期望结果expected是否相等
        tm.assert_series_equal(result, expected)
    # 测试混合轴参数的分位数计算，axis=1表示在每行上执行计算
    def test_quantile_axis_mixed(self, interp_method):
        # 根据interp_method参数确定插值方法和计算方法
        interpolation, method = interp_method
        # 创建一个包含不同数据类型的DataFrame
        df = DataFrame(
            {
                "A": [1, 2, 3],
                "B": [2.0, 3.0, 4.0],
                "C": pd.date_range("20130101", periods=3),
                "D": ["foo", "bar", "baz"],
            }
        )
        # 计算每行的分位数，numeric_only=True表示仅处理数值列，interpolation指定插值方法
        result = df.quantile(
            0.5, axis=1, numeric_only=True, interpolation=interpolation, method=method
        )
        # 创建预期结果，Series对象，包含每行的预期分位数结果
        expected = Series([1.5, 2.5, 3.5], name=0.5)
        # 如果插值方法是"nearest"，调整预期结果
        if interpolation == "nearest":
            expected -= 0.5
        # 使用测试工具函数tm.assert_series_equal比较结果和预期结果
        tm.assert_series_equal(result, expected)

        # 必须引发异常的情况
        msg = "'<' not supported between instances of 'Timestamp' and 'float'"
        # 使用pytest的raises断言来检查是否引发预期异常类型和消息
        with pytest.raises(TypeError, match=msg):
            df.quantile(0.5, axis=1, numeric_only=False)

    # 测试轴参数的分位数计算，检查GH 9543/9544问题
    def test_quantile_axis_parameter(self, interp_method):
        # 根据interp_method参数确定插值方法和计算方法
        interpolation, method = interp_method
        # 创建一个带索引的DataFrame
        df = DataFrame({"A": [1, 2, 3], "B": [2, 3, 4]}, index=[1, 2, 3])

        # 按列(axis=0)计算分位数，返回Series对象，包含每列的预期分位数结果
        result = df.quantile(0.5, axis=0, interpolation=interpolation, method=method)

        # 创建预期结果，Series对象，包含每列的预期分位数结果，索引为列名
        expected = Series([2.0, 3.0], index=["A", "B"], name=0.5)
        # 如果插值方法是"nearest"，将预期结果转换为np.int64类型
        if interpolation == "nearest":
            expected = expected.astype(np.int64)
        # 使用测试工具函数tm.assert_series_equal比较结果和预期结果
        tm.assert_series_equal(result, expected)

        # 使用字符串形式的轴参数"index"计算相同的结果
        expected = df.quantile(
            0.5, axis="index", interpolation=interpolation, method=method
        )
        if interpolation == "nearest":
            expected = expected.astype(np.int64)
        tm.assert_series_equal(result, expected)

        # 按行(axis=1)计算分位数，返回Series对象，包含每行的预期分位数结果
        result = df.quantile(0.5, axis=1, interpolation=interpolation, method=method)

        # 创建预期结果，Series对象，包含每行的预期分位数结果，索引为行索引
        expected = Series([1.5, 2.5, 3.5], index=[1, 2, 3], name=0.5)
        # 如果插值方法是"nearest"，将预期结果转换为np.int64类型
        if interpolation == "nearest":
            expected = expected.astype(np.int64)
        # 使用测试工具函数tm.assert_series_equal比较结果和预期结果
        tm.assert_series_equal(result, expected)

        # 使用字符串形式的轴参数"columns"计算相同的结果
        result = df.quantile(
            0.5, axis="columns", interpolation=interpolation, method=method
        )
        tm.assert_series_equal(result, expected)

        # 检查传入无效轴参数时是否引发预期的值错误异常
        msg = "No axis named -1 for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            df.quantile(0.1, axis=-1, interpolation=interpolation, method=method)
        msg = "No axis named column for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            df.quantile(0.1, axis="column")
    def test_quantile_interpolation(self):
        # see gh-10174
        # 测试分位数插值方法

        # interpolation method other than default linear
        # 使用非默认的线性插值方法
        df = DataFrame({"A": [1, 2, 3], "B": [2, 3, 4]}, index=[1, 2, 3])
        result = df.quantile(0.5, axis=1, interpolation="nearest")
        expected = Series([1, 2, 3], index=[1, 2, 3], name=0.5)
        tm.assert_series_equal(result, expected)

        # cross-check interpolation=nearest results in original dtype
        # 交叉检查插值方法为 nearest 时的结果与原始数据类型
        exp = np.percentile(
            np.array([[1, 2, 3], [2, 3, 4]]),
            0.5,
            axis=0,
            method="nearest",
        )
        expected = Series(exp, index=[1, 2, 3], name=0.5, dtype="int64")
        tm.assert_series_equal(result, expected)

        # float
        # 测试浮点数情况
        df = DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 3.0, 4.0]}, index=[1, 2, 3])
        result = df.quantile(0.5, axis=1, interpolation="nearest")
        expected = Series([1.0, 2.0, 3.0], index=[1, 2, 3], name=0.5)
        tm.assert_series_equal(result, expected)
        exp = np.percentile(
            np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
            0.5,
            axis=0,
            method="nearest",
        )
        expected = Series(exp, index=[1, 2, 3], name=0.5, dtype="float64")
        tm.assert_series_equal(result, expected)

        # axis
        # 测试轴向计算
        result = df.quantile([0.5, 0.75], axis=1, interpolation="lower")
        expected = DataFrame(
            {1: [1.0, 1.0], 2: [2.0, 2.0], 3: [3.0, 3.0]}, index=[0.5, 0.75]
        )
        tm.assert_frame_equal(result, expected)

        # test degenerate case
        # 测试极端情况
        df = DataFrame({"x": [], "y": []})
        q = df.quantile(0.1, axis=0, interpolation="higher")
        assert np.isnan(q["x"]) and np.isnan(q["y"])

        # multi
        # 多维度测试
        df = DataFrame([[1, 1, 1], [2, 2, 2], [3, 3, 3]], columns=["a", "b", "c"])
        result = df.quantile([0.25, 0.5], interpolation="midpoint")

        # https://github.com/numpy/numpy/issues/7163
        # 引用外部问题链接
        expected = DataFrame(
            [[1.5, 1.5, 1.5], [2.0, 2.0, 2.0]],
            index=[0.25, 0.5],
            columns=["a", "b", "c"],
        )
        tm.assert_frame_equal(result, expected)

    def test_quantile_interpolation_datetime(self, datetime_frame):
        # see gh-10174
        # 测试日期时间分位数插值方法

        # interpolation = linear (default case)
        # 插值方法为 linear（默认情况）
        df = datetime_frame
        q = df.quantile(0.1, axis=0, numeric_only=True, interpolation="linear")
        assert q["A"] == np.percentile(df["A"], 10)

    def test_quantile_interpolation_int(self, int_frame):
        # see gh-10174
        # 测试整数分位数插值方法

        df = int_frame
        # interpolation = linear (default case)
        # 插值方法为 linear（默认情况）
        q = df.quantile(0.1)
        assert q["A"] == np.percentile(df["A"], 10)

        # test with and without interpolation keyword
        # 测试使用和不使用插值关键字的情况
        q1 = df.quantile(0.1, axis=0, interpolation="linear")
        assert q1["A"] == np.percentile(df["A"], 10)
        tm.assert_series_equal(q, q1)
    # 定义一个测试方法，用于测试 DataFrame 的 quantile 方法在多个情况下的行为
    def test_quantile_multi(self, interp_method):
        # 解包插值方法和方法类型
        interpolation, method = interp_method
        # 创建一个包含多行数据的 DataFrame
        df = DataFrame([[1, 1, 1], [2, 2, 2], [3, 3, 3]], columns=["a", "b", "c"])
        # 调用 quantile 方法计算指定分位数处的值，并指定插值和方法
        result = df.quantile([0.25, 0.5], interpolation=interpolation, method=method)
        # 创建预期的 DataFrame，包含预期的分位数处的值
        expected = DataFrame(
            [[1.5, 1.5, 1.5], [2.0, 2.0, 2.0]],
            index=[0.25, 0.5],
            columns=["a", "b", "c"],
        )
        # 如果插值方法是 "nearest"，则将预期结果转换为整数类型
        if interpolation == "nearest":
            expected = expected.astype(np.int64)
        # 断言计算结果与预期结果是否相等
        tm.assert_frame_equal(result, expected)

    # 定义另一个测试方法，测试 DataFrame 的 quantile 方法在指定轴向上的行为
    def test_quantile_multi_axis_1(self, interp_method):
        # 解包插值方法和方法类型
        interpolation, method = interp_method
        # 创建一个包含多行数据的 DataFrame
        df = DataFrame([[1, 1, 1], [2, 2, 2], [3, 3, 3]], columns=["a", "b", "c"])
        # 调用 quantile 方法计算指定分位数处的值，指定轴向、插值和方法
        result = df.quantile(
            [0.25, 0.5], axis=1, interpolation=interpolation, method=method
        )
        # 创建预期的 DataFrame，包含预期的分位数处的值
        expected = DataFrame(
            [[1.0, 2.0, 3.0]] * 2, index=[0.25, 0.5], columns=[0, 1, 2]
        )
        # 如果插值方法是 "nearest"，则将预期结果转换为整数类型
        if interpolation == "nearest":
            expected = expected.astype(np.int64)
        # 断言计算结果与预期结果是否相等
        tm.assert_frame_equal(result, expected)

    # 定义第三个测试方法，测试在空 DataFrame 上调用 quantile 方法的行为
    def test_quantile_multi_empty(self, interp_method):
        # 解包插值方法和方法类型
        interpolation, method = interp_method
        # 调用 quantile 方法计算指定分位数处的值，指定轴向、插值和方法
        result = DataFrame({"x": [], "y": []}).quantile(
            [0.1, 0.9], axis=0, interpolation=interpolation, method=method
        )
        # 创建预期的 DataFrame，包含 NaN 值
        expected = DataFrame(
            {"x": [np.nan, np.nan], "y": [np.nan, np.nan]}, index=[0.1, 0.9]
        )
        # 断言计算结果与预期结果是否相等
        tm.assert_frame_equal(result, expected)
    # 定义一个测试函数，用于测试在指定时间单位下的量化方法
    def test_quantile_datetime(self, unit):
        # 创建一个日期时间索引对象，将年份转换为指定时间单位
        dti = pd.to_datetime(["2010", "2011"]).as_unit(unit)
        # 创建一个包含日期时间和整数列的数据帧
        df = DataFrame({"a": dti, "b": [0, 5]})

        # 对于数值列，计算指定分位数的分位数值
        result = df.quantile(0.5, numeric_only=True)
        # 预期结果是一个包含单个值的序列，索引为'b'，名称为0.5
        expected = Series([2.5], index=["b"], name=0.5)
        tm.assert_series_equal(result, expected)

        # 对于包含日期时间的列，计算指定分位数的分位数值
        result = df.quantile(0.5, numeric_only=False)
        # 预期结果是一个包含日期时间和整数的序列，索引为'a', 'b'，名称为0.5
        expected = Series(
            [Timestamp("2010-07-02 12:00:00"), 2.5], index=["a", "b"], name=0.5
        )
        tm.assert_series_equal(result, expected)

        # 对于包含日期时间的列，计算指定分位数的分位数值，结果作为数据帧返回
        result = df.quantile([0.5], numeric_only=False)
        # 预期结果是一个数据帧，包含日期时间和整数，索引为[0.5]
        expected = DataFrame(
            {"a": Timestamp("2010-07-02 12:00:00").as_unit(unit), "b": 2.5},
            index=[0.5],
        )
        tm.assert_frame_equal(result, expected)

        # 在轴向为1时，计算指定分位数的分位数值，仅针对指定的列
        df["c"] = pd.to_datetime(["2011", "2012"]).as_unit(unit)
        result = df[["a", "c"]].quantile(0.5, axis=1, numeric_only=False)
        # 预期结果是一个包含日期时间的序列，索引为[0, 1]，名称为0.5，数据类型为指定时间单位的日期时间
        expected = Series(
            [Timestamp("2010-07-02 12:00:00"), Timestamp("2011-07-02 12:00:00")],
            index=[0, 1],
            name=0.5,
            dtype=f"M8[{unit}]",
        )
        tm.assert_series_equal(result, expected)

        # 在轴向为1时，计算指定分位数的分位数值，结果作为数据帧返回
        result = df[["a", "c"]].quantile([0.5], axis=1, numeric_only=False)
        # 预期结果是一个数据帧，包含日期时间，索引为[0.5]，列名为[0, 1]，数据类型为指定时间单位的日期时间
        expected = DataFrame(
            [[Timestamp("2010-07-02 12:00:00"), Timestamp("2011-07-02 12:00:00")]],
            index=[0.5],
            columns=[0, 1],
            dtype=f"M8[{unit}]",
        )
        tm.assert_frame_equal(result, expected)

        # 当numeric_only=True时，对于不包含日期时间的列，返回一个空的序列
        result = df[["a", "c"]].quantile(0.5, numeric_only=True)
        # 预期结果是一个空的序列，索引为空，数据类型为np.float64，名称为0.5
        expected = Series([], index=[], dtype=np.float64, name=0.5)
        tm.assert_series_equal(result, expected)

        # 当numeric_only=True时，对于不包含日期时间的列，返回一个空的数据帧
        result = df[["a", "c"]].quantile([0.5], numeric_only=True)
        # 预期结果是一个空的数据帧，索引为[0.5]，列为空
        expected = DataFrame(index=[0.5], columns=[])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype",
        [
            "datetime64[ns]",
            "datetime64[ns, US/Pacific]",
            "timedelta64[ns]",
            "Period[D]",
        ],
    )
    # 定义一个测试函数，用于测试在特定数据类型下的空数据帧的量化方法
    def test_quantile_dt64_empty(self, dtype, interp_method):
        # GH#41544
        # 从参数中获取插值和方法
        interpolation, method = interp_method
        # 创建一个指定数据类型的空数据帧
        df = DataFrame(columns=["a", "b"], dtype=dtype)

        # 对空数据帧进行量化，计算指定分位数的分位数值
        res = df.quantile(
            0.5, axis=1, numeric_only=False, interpolation=interpolation, method=method
        )
        # 预期结果是一个空的序列，索引为空，数据类型与输入数据类型相同，名称为0.5
        expected = Series([], index=[], name=0.5, dtype=dtype)
        tm.assert_series_equal(res, expected)

        # 对空数据帧进行量化，计算指定分位数的分位数值，结果作为数据帧返回
        res = df.quantile(
            [0.5],
            axis=1,
            numeric_only=False,
            interpolation=interpolation,
            method=method,
        )
        # 预期结果是一个空的数据帧，索引为[0.5]，列为空
        expected = DataFrame(index=[0.5], columns=[])
        tm.assert_frame_equal(res, expected)
    # 使用 pytest 的参数化装饰器，定义了多个无效的测试参数
    @pytest.mark.parametrize("invalid", [-1, 2, [0.5, -1], [0.5, 2]])
    # 测试 quantile 方法对于无效参数的异常处理
    def test_quantile_invalid(self, invalid, datetime_frame, interp_method):
        # 定义期望抛出的 ValueError 异常信息
        msg = "percentiles should all be in the interval \\[0, 1\\]"
        interpolation, method = interp_method
        # 使用 pytest 的断言，验证 quantile 方法对于无效参数是否能正确抛出异常
        with pytest.raises(ValueError, match=msg):
            datetime_frame.quantile(invalid, interpolation=interpolation, method=method)

    # 测试 quantile 方法在数据框 DataFrame 上的盒形图分位数计算
    def test_quantile_box(self, interp_method):
        interpolation, method = interp_method
        # 创建一个示例的 DataFrame 对象 df
        df = DataFrame(
            {
                "A": [
                    Timestamp("2011-01-01"),
                    Timestamp("2011-01-02"),
                    Timestamp("2011-01-03"),
                ],
                "B": [
                    Timestamp("2011-01-01", tz="US/Eastern"),
                    Timestamp("2011-01-02", tz="US/Eastern"),
                    Timestamp("2011-01-03", tz="US/Eastern"),
                ],
                "C": [
                    pd.Timedelta("1 days"),
                    pd.Timedelta("2 days"),
                    pd.Timedelta("3 days"),
                ],
            }
        )

        # 计算 df 的中位数（分位数为 0.5）并保存结果到 res
        res = df.quantile(
            0.5, numeric_only=False, interpolation=interpolation, method=method
        )

        # 期望的中位数结果 exp，作为 Series 对象
        exp = Series(
            [
                Timestamp("2011-01-02"),
                Timestamp("2011-01-02", tz="US/Eastern"),
                pd.Timedelta("2 days"),
            ],
            name=0.5,
            index=["A", "B", "C"],
        )
        # 使用 pandas 测试模块中的 assert_series_equal 验证 res 和 exp 是否相等
        tm.assert_series_equal(res, exp)

        # 计算 df 的多个分位数（这里只计算了一个分位数，即 0.5）并保存结果到 res
        res = df.quantile(
            [0.5], numeric_only=False, interpolation=interpolation, method=method
        )
        # 期望的多个分位数结果 exp，作为 DataFrame 对象
        exp = DataFrame(
            [
                [
                    Timestamp("2011-01-02"),
                    Timestamp("2011-01-02", tz="US/Eastern"),
                    pd.Timedelta("2 days"),
                ]
            ],
            index=[0.5],
            columns=["A", "B", "C"],
        )
        # 使用 pandas 测试模块中的 assert_frame_equal 验证 res 和 exp 是否相等
        tm.assert_frame_equal(res, exp)
    def test_quantile_box_nat(self):
        # 定义一个测试方法，测试处理带有 NaT（Not a Time）的 DateTimeLikeBlock 的情况
        df = DataFrame(
            {
                "A": [
                    Timestamp("2011-01-01"),
                    pd.NaT,
                    Timestamp("2011-01-02"),
                    Timestamp("2011-01-03"),
                ],
                "a": [
                    Timestamp("2011-01-01"),
                    Timestamp("2011-01-02"),
                    pd.NaT,
                    Timestamp("2011-01-03"),
                ],
                "B": [
                    Timestamp("2011-01-01", tz="US/Eastern"),
                    pd.NaT,
                    Timestamp("2011-01-02", tz="US/Eastern"),
                    Timestamp("2011-01-03", tz="US/Eastern"),
                ],
                "b": [
                    Timestamp("2011-01-01", tz="US/Eastern"),
                    Timestamp("2011-01-02", tz="US/Eastern"),
                    pd.NaT,
                    Timestamp("2011-01-03", tz="US/Eastern"),
                ],
                "C": [
                    pd.Timedelta("1 days"),
                    pd.Timedelta("2 days"),
                    pd.Timedelta("3 days"),
                    pd.NaT,
                ],
                "c": [
                    pd.NaT,
                    pd.Timedelta("1 days"),
                    pd.Timedelta("2 days"),
                    pd.Timedelta("3 days"),
                ],
            },
            columns=list("AaBbCc"),
        )

        # 计算数据框中各列的 50% 分位数，包括 NA/NaN 值
        res = df.quantile(0.5, numeric_only=False)
        # 期望的结果，是一个包含各列 50% 分位数的序列对象
        exp = Series(
            [
                Timestamp("2011-01-02"),
                Timestamp("2011-01-02"),
                Timestamp("2011-01-02", tz="US/Eastern"),
                Timestamp("2011-01-02", tz="US/Eastern"),
                pd.Timedelta("2 days"),
                pd.Timedelta("2 days"),
            ],
            name=0.5,
            index=list("AaBbCc"),
        )
        # 断言序列 res 和期望的序列 exp 相等
        tm.assert_series_equal(res, exp)

        # 计算数据框中各列的指定分位数（此处为 50%），包括 NA/NaN 值
        res = df.quantile([0.5], numeric_only=False)
        # 期望的结果，是一个包含各列指定分位数（此处为 50%）的数据框对象
        exp = DataFrame(
            [
                [
                    Timestamp("2011-01-02"),
                    Timestamp("2011-01-02"),
                    Timestamp("2011-01-02", tz="US/Eastern"),
                    Timestamp("2011-01-02", tz="US/Eastern"),
                    pd.Timedelta("2 days"),
                    pd.Timedelta("2 days"),
                ]
            ],
            index=[0.5],
            columns=list("AaBbCc"),
        )
        # 断言数据框 res 和期望的数据框 exp 相等
        tm.assert_frame_equal(res, exp)
    # 定义一个测试函数，用于测试在特定插值方法下的分位数计算
    def test_quantile_nan(self, interp_method):
        # 解包插值方法元组，分别赋值给 interpolation 和 method 变量
        interpolation, method = interp_method
        
        # 创建一个包含两列的数据框，其中一列包含缺失值
        df = DataFrame({"a": np.arange(1, 6.0), "b": np.arange(1, 6.0)})
        df.iloc[-1, 1] = np.nan

        # 计算数据框 df 的中位数，使用给定的插值方法和计算方法
        res = df.quantile(0.5, interpolation=interpolation, method=method)
        # 创建预期结果的序列对象，根据插值方法选择不同的期望值
        exp = Series(
            [3.0, 2.5 if interpolation == "linear" else 3.0], index=["a", "b"], name=0.5
        )
        # 断言计算结果与预期结果相等
        tm.assert_series_equal(res, exp)

        # 计算数据框 df 的多个分位数，使用给定的插值方法和计算方法
        res = df.quantile([0.5, 0.75], interpolation=interpolation, method=method)
        # 创建预期结果的数据框对象，根据插值方法选择不同的期望值
        exp = DataFrame(
            {
                "a": [3.0, 4.0],
                "b": [2.5, 3.25] if interpolation == "linear" else [3.0, 4.0],
            },
            index=[0.5, 0.75],
        )
        # 断言计算结果与预期结果相等
        tm.assert_frame_equal(res, exp)

        # 沿着数据框 df 的列轴计算中位数，使用给定的插值方法和计算方法
        res = df.quantile(0.5, axis=1, interpolation=interpolation, method=method)
        # 创建预期结果的序列对象，期望值为数据框 df 各列的中位数
        exp = Series(np.arange(1.0, 6.0), name=0.5)
        # 断言计算结果与预期结果相等
        tm.assert_series_equal(res, exp)

        # 沿着数据框 df 的列轴计算多个分位数，使用给定的插值方法和计算方法
        res = df.quantile(
            [0.5, 0.75], axis=1, interpolation=interpolation, method=method
        )
        # 创建预期结果的数据框对象，期望值为数据框 df 各列的中位数
        exp = DataFrame([np.arange(1.0, 6.0)] * 2, index=[0.5, 0.75])
        # 如果插值方法为 "nearest"，则在期望结果的指定位置设置为 NaN
        if interpolation == "nearest":
            exp.iloc[1, -1] = np.nan
        # 断言计算结果与预期结果相等
        tm.assert_frame_equal(res, exp)

        # 将数据框 df 的列 "b" 设置为全为 NaN 的列
        df["b"] = np.nan

        # 计算数据框 df 的中位数，使用给定的插值方法和计算方法
        res = df.quantile(0.5, interpolation=interpolation, method=method)
        # 创建预期结果的序列对象，其中包含一个值为 NaN 的条目
        exp = Series([3.0, np.nan], index=["a", "b"], name=0.5)
        # 断言计算结果与预期结果相等
        tm.assert_series_equal(res, exp)

        # 计算数据框 df 的多个分位数，使用给定的插值方法和计算方法
        res = df.quantile([0.5, 0.75], interpolation=interpolation, method=method)
        # 创建预期结果的数据框对象，其中包含一个列完全为 NaN 的条目
        exp = DataFrame({"a": [3.0, 4.0], "b": [np.nan, np.nan]}, index=[0.5, 0.75])
        # 断言计算结果与预期结果相等
        tm.assert_frame_equal(res, exp)
    def test_quantile_nat(self, interp_method, unit):
        interpolation, method = interp_method
        # 创建一个数据框，包含一个全为 NaT 的列 'a'，并指定数据类型为给定单位的日期时间类型
        df = DataFrame({"a": [pd.NaT, pd.NaT, pd.NaT]}, dtype=f"M8[{unit}]")

        # 计算数据框 df 的分位数为 0.5 的结果，允许插值，并使用指定的方法
        res = df.quantile(
            0.5, numeric_only=False, interpolation=interpolation, method=method
        )
        # 期望的结果，一个包含 NaN 的 Series，索引为 'a'，名称为 0.5，数据类型为给定单位的日期时间类型
        exp = Series([pd.NaT], index=["a"], name=0.5, dtype=f"M8[{unit}]")
        # 断言结果是否与期望相等
        tm.assert_series_equal(res, exp)

        # 计算数据框 df 的分位数为 [0.5] 的结果，允许插值，并使用指定的方法
        res = df.quantile(
            [0.5], numeric_only=False, interpolation=interpolation, method=method
        )
        # 期望的结果，一个包含 NaN 的 DataFrame，列为 'a'，索引为 [0.5]，数据类型为给定单位的日期时间类型
        exp = DataFrame({"a": [pd.NaT]}, index=[0.5], dtype=f"M8[{unit}]")
        # 断言结果是否与期望相等
        tm.assert_frame_equal(res, exp)

        # 创建一个数据框，包含一个混合了非空值和全为 NaT 的列 'a' 和 'b'，并指定数据类型为给定单位的日期时间类型
        df = DataFrame(
            {
                "a": [
                    Timestamp("2012-01-01"),
                    Timestamp("2012-01-02"),
                    Timestamp("2012-01-03"),
                ],
                "b": [pd.NaT, pd.NaT, pd.NaT],
            },
            dtype=f"M8[{unit}]",
        )

        # 计算数据框 df 的分位数为 0.5 的结果，允许插值，并使用指定的方法
        res = df.quantile(
            0.5, numeric_only=False, interpolation=interpolation, method=method
        )
        # 期望的结果，一个包含时间戳和 NaN 的 Series，索引为 'a', 'b'，名称为 0.5，数据类型为给定单位的日期时间类型
        exp = Series(
            [Timestamp("2012-01-02"), pd.NaT],
            index=["a", "b"],
            name=0.5,
            dtype=f"M8[{unit}]",
        )
        # 断言结果是否与期望相等
        tm.assert_series_equal(res, exp)

        # 计算数据框 df 的分位数为 [0.5] 的结果，允许插值，并使用指定的方法
        res = df.quantile(
            [0.5], numeric_only=False, interpolation=interpolation, method=method
        )
        # 期望的结果，一个包含时间戳和 NaN 的 DataFrame，列为 'a', 'b'，索引为 [0.5]，数据类型为给定单位的日期时间类型
        exp = DataFrame(
            [[Timestamp("2012-01-02"), pd.NaT]],
            index=[0.5],
            columns=["a", "b"],
            dtype=f"M8[{unit}]",
        )
        # 断言结果是否与期望相等
        tm.assert_frame_equal(res, exp)

    def test_quantile_empty_no_rows_floats(self, interp_method):
        interpolation, method = interp_method

        # 创建一个空数据框，列为 'a', 'b'，数据类型为 float64
        df = DataFrame(columns=["a", "b"], dtype="float64")

        # 计算数据框 df 的分位数为 0.5 的结果，允许插值，并使用指定的方法
        res = df.quantile(0.5, interpolation=interpolation, method=method)
        # 期望的结果，一个包含 NaN 的 Series，索引为 'a', 'b'，名称为 0.5
        exp = Series([np.nan, np.nan], index=["a", "b"], name=0.5)
        # 断言结果是否与期望相等
        tm.assert_series_equal(res, exp)

        # 计算数据框 df 的分位数为 [0.5] 的结果，允许插值，并使用指定的方法
        res = df.quantile([0.5], interpolation=interpolation, method=method)
        # 期望的结果，一个包含 NaN 的 DataFrame，列为 'a', 'b'，索引为 [0.5]
        exp = DataFrame([[np.nan, np.nan]], columns=["a", "b"], index=[0.5])
        # 断言结果是否与期望相等
        tm.assert_frame_equal(res, exp)

        # 计算数据框 df 沿着轴 1 的分位数为 0.5 的结果，允许插值，并使用指定的方法
        res = df.quantile(0.5, axis=1, interpolation=interpolation, method=method)
        # 期望的结果，一个空的 Series，索引为空，数据类型为 float64，名称为 0.5
        exp = Series([], index=[], dtype="float64", name=0.5)
        # 断言结果是否与期望相等
        tm.assert_series_equal(res, exp)

        # 计算数据框 df 沿着轴 1 的分位数为 [0.5] 的结果，允许插值，并使用指定的方法
        res = df.quantile([0.5], axis=1, interpolation=interpolation, method=method)
        # 期望的结果，一个空的 DataFrame，列为空，索引为 [0.5]
        exp = DataFrame(columns=[], index=[0.5])
        # 断言结果是否与期望相等
        tm.assert_frame_equal(res, exp)

    def test_quantile_empty_no_rows_ints(self, interp_method):
        interpolation, method = interp_method
        # 创建一个空数据框，列为 'a', 'b'，数据类型为 int64
        df = DataFrame(columns=["a", "b"], dtype="int64")

        # 计算数据框 df 的分位数为 0.5 的结果，允许插值，并使用指定的方法
        res = df.quantile(0.5, interpolation=interpolation, method=method)
        # 期望的结果，一个包含 NaN 的 Series，索引为 'a', 'b'，名称为 0.5
        exp = Series([np.nan, np.nan], index=["a", "b"], name=0.5)
        # 断言结果是否与期望相等
        tm.assert_series_equal(res, exp)
    def test_quantile_empty_no_rows_dt64(self, interp_method):
        # 定义测试函数，用于测试空数据框和日期时间类型列的分位数计算
        interpolation, method = interp_method
        # 创建一个空的数据框，包含两列，数据类型为 datetime64[ns]
        df = DataFrame(columns=["a", "b"], dtype="datetime64[ns]")

        # 计算分位数并保存结果
        res = df.quantile(
            0.5, numeric_only=False, interpolation=interpolation, method=method
        )
        # 创建预期的结果 Series，包含 NaT（Not a Time）值
        exp = Series(
            [pd.NaT, pd.NaT], index=["a", "b"], dtype="datetime64[ns]", name=0.5
        )
        # 使用测试工具比较计算结果和预期结果
        tm.assert_series_equal(res, exp)

        # 对于混合的 dt64/dt64tz 类型
        # 将列 'a' 的日期时间类型设置为带时区信息的 'US/Central'
        df["a"] = df["a"].dt.tz_localize("US/Central")
        # 重新计算分位数
        res = df.quantile(
            0.5, numeric_only=False, interpolation=interpolation, method=method
        )
        # 将预期结果转换为 object 类型
        exp = exp.astype(object)
        # 如果插值方法为 "nearest"，则将预期结果中的 NaT 值填充为 NaN
        if interpolation == "nearest":
            # GH#18463 TODO: would we prefer NaTs here?
            exp = exp.fillna(np.nan)
        # 使用测试工具比较计算结果和预期结果
        tm.assert_series_equal(res, exp)

        # 对于两列均为 dt64tz 类型的情况
        # 将列 'b' 的日期时间类型设置为带时区信息的 'US/Central'
        df["b"] = df["b"].dt.tz_localize("US/Central")
        # 重新计算分位数
        res = df.quantile(
            0.5, numeric_only=False, interpolation=interpolation, method=method
        )
        # 将预期结果转换为与列 'b' 相同的数据类型
        exp = exp.astype(df["b"].dtype)
        # 使用测试工具比较计算结果和预期结果
        tm.assert_series_equal(res, exp)

    def test_quantile_empty_no_columns(self, interp_method):
        # 定义测试函数，用于测试空数据框和无列名称的情况
        # GH#23925 _get_numeric_data 可能会删除所有列
        interpolation, method = interp_method
        # 创建一个数据框，包含一个日期范围
        df = DataFrame(pd.date_range("1/1/18", periods=5))
        # 设置列名为 "captain tightpants"
        df.columns.name = "captain tightpants"
        # 计算分位数并保存结果
        result = df.quantile(
            0.5, numeric_only=True, interpolation=interpolation, method=method
        )
        # 创建预期的结果 Series，为空，数据类型为 np.float64
        expected = Series([], name=0.5, dtype=np.float64)
        # 设置预期结果的索引名为 "captain tightpants"
        expected.index.name = "captain tightpants"
        # 使用测试工具比较计算结果和预期结果
        tm.assert_series_equal(result, expected)

        # 计算分位数并保存结果，此时传入的分位数是一个列表
        result = df.quantile(
            [0.5], numeric_only=True, interpolation=interpolation, method=method
        )
        # 创建预期的结果 DataFrame，为空，索引为 [0.5]
        expected = DataFrame([], index=[0.5])
        # 设置预期结果的列名为 "captain tightpants"
        expected.columns.name = "captain tightpants"
        # 使用测试工具比较计算结果和预期结果
        tm.assert_frame_equal(result, expected)

    def test_quantile_item_cache(self, interp_method):
        # 定义测试函数，用于测试分位数计算的缓存项行为
        # 先前的行为是错误的，保留了无效的 _item_cache 条目
        interpolation, method = interp_method
        # 创建一个数据框，包含随机生成的标准正态分布数据
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 3)), columns=["A", "B", "C"]
        )
        # 添加一个新列 'D'，其值为列 'A' 的两倍
        df["D"] = df["A"] * 2
        # 获取列 'A' 的一个切片
        ser = df["A"]
        # 断言数据框的内部块数为 2
        assert len(df._mgr.blocks) == 2

        # 计算分位数，不关心数值类型
        df.quantile(numeric_only=False, interpolation=interpolation, method=method)

        # 修改 ser 中的第一个元素为 99
        ser.iloc[0] = 99
        # 断言数据框第一行第一列的值与 df["A"][0] 相等
        assert df.iloc[0, 0] == df["A"][0]
        # 断言数据框第一行第一列的值不等于 99
        assert df.iloc[0, 0] != 99

    def test_invalid_method(self):
        # 测试函数，用于测试传入无效方法参数时的异常抛出
        with pytest.raises(ValueError, match="Invalid method: foo"):
            # 创建一个包含一个整数的数据框，并计算分位数，期望抛出 ValueError 异常
            DataFrame(range(1)).quantile(0.5, method="foo")

    def test_table_invalid_interpolation(self):
        # 测试函数，用于测试传入无效插值方法参数时的异常抛出
        with pytest.raises(ValueError, match="Invalid interpolation: foo"):
            # 创建一个包含一个整数的数据框，并计算分位数，期望抛出 ValueError 异常
            DataFrame(range(1)).quantile(0.5, method="table", interpolation="foo")
class TestQuantileExtensionDtype:
    # TODO: tests for axis=1?
    # TODO: empty case?

    @pytest.fixture(
        params=[
            pytest.param(
                pd.IntervalIndex.from_breaks(range(10)),
                marks=pytest.mark.xfail(reason="raises when trying to add Intervals"),
            ),  # 使用pytest.param创建测试参数，标记为xfail以便于测试失败时有意义的原因
            pd.period_range("2016-01-01", periods=9, freq="D"),  # 创建日期周期范围的测试参数
            pd.date_range("2016-01-01", periods=9, tz="US/Pacific"),  # 创建带时区的日期范围测试参数
            pd.timedelta_range("1 Day", periods=9),  # 创建时间差范围测试参数
            pd.array(np.arange(9), dtype="Int64"),  # 使用numpy数组创建dtype为Int64的Pandas数组测试参数
            pd.array(np.arange(9), dtype="Float64"),  # 使用numpy数组创建dtype为Float64的Pandas数组测试参数
        ],
        ids=lambda x: str(x.dtype),
    )
    def index(self, request):
        # NB: not actually an Index object
        idx = request.param  # 获取测试参数
        idx.name = "A"  # 设置索引名称为"A"
        return idx  # 返回测试参数作为fixture的索引

    @pytest.fixture
    def obj(self, index, frame_or_series):
        # bc index is not always an Index (yet), we need to re-patch .name
        obj = frame_or_series(index).copy()  # 复制索引并创建相应的DataFrame或Series对象

        if frame_or_series is Series:
            obj.name = "A"  # 如果是Series，则设置对象名称为"A"
        else:
            obj.columns = ["A"]  # 如果是DataFrame，则设置列名称为["A"]
        return obj  # 返回创建的对象作为fixture

    def compute_quantile(self, obj, qs):
        if isinstance(obj, Series):
            result = obj.quantile(qs)  # 如果对象是Series，则计算给定分位数的分位数值
        else:
            result = obj.quantile(qs, numeric_only=False)  # 如果对象是DataFrame，则计算给定分位数的分位数值，包括非数值列
        return result  # 返回计算结果

    def test_quantile_ea(self, request, obj, index):
        # result should be invariant to shuffling
        indexer = np.arange(len(index), dtype=np.intp)  # 创建索引器，用于对对象进行随机化处理
        np.random.default_rng(2).shuffle(indexer)  # 使用随机数生成器对索引器进行随机化处理
        obj = obj.iloc[indexer]  # 根据索引器对对象进行重新排序

        qs = [0.5, 0, 1]  # 定义分位数列表
        result = self.compute_quantile(obj, qs)  # 计算给定对象和分位数的分位数值

        exp_dtype = index.dtype  # 获取索引的数据类型作为预期数据类型
        if index.dtype == "Int64":
            exp_dtype = "Float64"  # 如果索引数据类型是Int64，则预期数据类型为Float64，以匹配非空值的类型转换行为

        # expected here assumes len(index) == 9
        expected = Series(
            [index[4], index[0], index[-1]], dtype=exp_dtype, index=qs, name="A"
        )  # 创建预期的Series对象，包括指定的数据类型、索引和名称

        expected = type(obj)(expected)  # 根据对象类型重新创建预期值对象

        tm.assert_equal(result, expected)  # 使用Pandas测试工具比较结果与预期值

    def test_quantile_ea_with_na(self, obj, index):
        obj.iloc[0] = index._na_value  # 将对象的第一个值设置为索引的NA值
        obj.iloc[-1] = index._na_value  # 将对象的最后一个值设置为索引的NA值

        # result should be invariant to shuffling
        indexer = np.arange(len(index), dtype=np.intp)  # 创建索引器，用于对对象进行随机化处理
        np.random.default_rng(2).shuffle(indexer)  # 使用随机数生成器对索引器进行随机化处理
        obj = obj.iloc[indexer]  # 根据索引器对对象进行重新排序

        qs = [0.5, 0, 1]  # 定义分位数列表
        result = self.compute_quantile(obj, qs)  # 计算给定对象和分位数的分位数值

        # expected here assumes len(index) == 9
        expected = Series(
            [index[4], index[1], index[-2]], dtype=index.dtype, index=qs, name="A"
        )  # 创建预期的Series对象，包括指定的数据类型、索引和名称

        expected = type(obj)(expected)  # 根据对象类型重新创建预期值对象

        tm.assert_equal(result, expected)  # 使用Pandas测试工具比较结果与预期值
    @pytest.mark.parametrize(
        "dtype, expected_data, expected_index, axis",
        [
            # 第一个参数组合：float64 类型，空列表预期数据，空列表预期索引，沿着第一轴
            ["float64", [], [], 1],
            # 第二个参数组合：int64 类型，空列表预期数据，空列表预期索引，沿着第一轴
            ["int64", [], [], 1],
            # 第三个参数组合：float64 类型，包含两个 NaN 的列表预期数据，包含字符串 'a' 和 'b' 的索引列表，沿着第零轴
            ["float64", [np.nan, np.nan], ["a", "b"], 0],
            # 第四个参数组合：int64 类型，包含两个 NaN 的列表预期数据，包含字符串 'a' 和 'b' 的索引列表，沿着第零轴
            ["int64", [np.nan, np.nan], ["a", "b"], 0],
        ],
    )
    # 定义一个测试方法，测试处理空数值的情况
    def test_empty_numeric(self, dtype, expected_data, expected_index, axis):
        # GH 14564
        # 创建一个指定数据类型和列名的空 DataFrame
        df = DataFrame(columns=["a", "b"], dtype=dtype)
        # 计算 DataFrame 的中位数（0.5 分位数），沿着指定轴
        result = df.quantile(0.5, axis=axis)
        # 创建一个期望的 Series 对象，包括预期数据、名称和索引，数据类型为 float64
        expected = Series(
            expected_data, name=0.5, index=Index(expected_index), dtype="float64"
        )
        # 断言结果与期望值相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype, expected_data, expected_index, axis, expected_dtype",
        [
            # 第一个参数组合：datetime64[ns] 类型，空列表预期数据，空列表预期索引，沿着第一轴，预期数据类型为 datetime64[ns]
            ["datetime64[ns]", [], [], 1, "datetime64[ns]"],
            # 第二个参数组合：datetime64[ns] 类型，包含两个 NaT 的列表预期数据，包含字符串 'a' 和 'b' 的索引列表，沿着第零轴，预期数据类型为 datetime64[ns]
            ["datetime64[ns]", [pd.NaT, pd.NaT], ["a", "b"], 0, "datetime64[ns]"],
        ],
    )
    # 定义一个测试方法，测试处理空日期类数据的情况
    def test_empty_datelike(
        self, dtype, expected_data, expected_index, axis, expected_dtype
    ):
        # GH 14564
        # 创建一个指定数据类型和列名的空 DataFrame
        df = DataFrame(columns=["a", "b"], dtype=dtype)
        # 计算 DataFrame 的中位数（0.5 分位数），沿着指定轴，包括非数值类型的列
        result = df.quantile(0.5, axis=axis, numeric_only=False)
        # 创建一个期望的 Series 对象，包括预期数据、名称、索引和数据类型
        expected = Series(
            expected_data, name=0.5, index=Index(expected_index), dtype=expected_dtype
        )
        # 断言结果与期望值相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "expected_data, expected_index, axis",
        [
            # 第一个参数组合：包含两个 NaN 的列表预期数据，索引为 0 到 1，沿着第一轴
            [[np.nan, np.nan], range(2), 1],
            # 第二个参数组合：空列表预期数据，空列表预期索引，沿着第零轴
            [[], [], 0],
        ],
    )
    # 定义一个测试方法，用于验证数值型数据在指定轴上的分位数计算
    def test_datelike_numeric_only(self, expected_data, expected_index, axis):
        # GH 14564
        # 创建一个 DataFrame 对象 df，包含三列数据：
        # - "a": 包含日期时间对象的列，转换自字符串列表 ["2010", "2011"]
        # - "b": 包含整数数据的列，值为 [0, 5]
        # - "c": 包含日期时间对象的列，转换自字符串列表 ["2011", "2012"]
        df = DataFrame(
            {
                "a": pd.to_datetime(["2010", "2011"]),
                "b": [0, 5],
                "c": pd.to_datetime(["2011", "2012"]),
            }
        )
        # 计算 DataFrame df 中列 "a" 和 "c" 的指定分位数（此处为中位数）并返回结果
        result = df[["a", "c"]].quantile(0.5, axis=axis, numeric_only=True)
        # 创建预期的 Series 对象 expected，包含预期的数据 expected_data、名称为 0.5、索引为 expected_index、数据类型为 np.float64
        expected = Series(
            expected_data, name=0.5, index=Index(expected_index), dtype=np.float64
        )
        # 使用测试工具 tm.assert_series_equal 比较 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)
# 定义一个测试函数，用于测试多分位数计算，仅保留数值列
def test_multi_quantile_numeric_only_retains_columns():
    # 创建一个包含单列字符列表的数据框
    df = DataFrame(list("abc"))
    # 计算指定分位数（0.5 和 0.7）的数值列的分位数值
    result = df.quantile([0.5, 0.7], numeric_only=True)
    # 创建一个预期结果数据框，其索引为指定的分位数
    expected = DataFrame(index=[0.5, 0.7])
    # 使用测试工具检查结果数据框与预期数据框是否相等，包括索引类型和列类型的检查
    tm.assert_frame_equal(
        result, expected, check_index_type=True, check_column_type=True
    )
```