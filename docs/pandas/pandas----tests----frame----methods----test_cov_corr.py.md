# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_cov_corr.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试

import pandas.util._test_decorators as td  # 导入 pandas 内部测试装饰器模块

import pandas as pd  # 导入 pandas 库，用于数据分析
from pandas import (  # 从 pandas 中导入多个模块和函数
    DataFrame,  # 数据框结构
    Index,  # 索引结构
    Series,  # 序列结构
    date_range,  # 日期范围生成函数
    isna,  # 判断缺失值函数
)
import pandas._testing as tm  # 导入 pandas 内部测试模块

class TestDataFrameCov:
    def test_cov(self, float_frame, float_string_frame):
        # min_periods no NAs (corner case)
        expected = float_frame.cov()  # 计算浮点数据框的协方差矩阵
        result = float_frame.cov(min_periods=len(float_frame))  # 计算浮点数据框的协方差矩阵，指定最小期数

        tm.assert_frame_equal(expected, result)  # 使用测试模块验证期望结果和计算结果是否相等

        result = float_frame.cov(min_periods=len(float_frame) + 1)  # 计算浮点数据框的协方差矩阵，期数超过实际数据长度
        assert isna(result.values).all()  # 断言结果中是否全为缺失值

        # with NAs
        frame = float_frame.copy()  # 复制浮点数据框
        frame.iloc[:5, frame.columns.get_loc("A")] = np.nan  # 将前五行"A"列数据设为 NaN
        frame.iloc[5:10, frame.columns.get_loc("B")] = np.nan  # 将第五至第十行"B"列数据设为 NaN
        result = frame.cov(min_periods=len(frame) - 8)  # 计算浮点数据框的协方差矩阵，指定最小期数
        expected = frame.cov()  # 计算期望的协方差矩阵
        expected.loc["A", "B"] = np.nan  # 期望结果中"A"和"B"之间的协方差为 NaN
        expected.loc["B", "A"] = np.nan  # 期望结果中"B"和"A"之间的协方差为 NaN
        tm.assert_frame_equal(result, expected)  # 使用测试模块验证期望结果和计算结果是否相等

        # regular
        result = frame.cov()  # 计算浮点数据框的协方差矩阵
        expected = frame["A"].cov(frame["C"])  # 计算"A"列和"C"列之间的协方差
        tm.assert_almost_equal(result["A"]["C"], expected)  # 使用测试模块验证两者的近似相等性

        # fails on non-numeric types
        with pytest.raises(ValueError, match="could not convert string to float"):  # 使用 pytest 断言捕获 ValueError 异常
            float_string_frame.cov()  # 计算包含字符串的浮点数据框的协方差矩阵
        result = float_string_frame.cov(numeric_only=True)  # 计算数值类型的协方差矩阵
        expected = float_string_frame.loc[:, ["A", "B", "C", "D"]].cov()  # 计算选定列的协方差矩阵
        tm.assert_frame_equal(result, expected)  # 使用测试模块验证期望结果和计算结果是否相等

        # Single column frame
        df = DataFrame(np.linspace(0.0, 1.0, 10))  # 创建包含从 0.0 到 1.0 的 10 个等间距数值的数据框
        result = df.cov()  # 计算数据框的协方差矩阵
        expected = DataFrame(  # 创建期望的协方差矩阵
            np.cov(df.values.T).reshape((1, 1)), index=df.columns, columns=df.columns
        )
        tm.assert_frame_equal(result, expected)  # 使用测试模块验证期望结果和计算结果是否相等
        df.loc[0] = np.nan  # 将第一行数据设为 NaN
        result = df.cov()  # 重新计算数据框的协方差矩阵
        expected = DataFrame(  # 创建期望的协方差矩阵
            np.cov(df.values[1:].T).reshape((1, 1)),
            index=df.columns,
            columns=df.columns,
        )
        tm.assert_frame_equal(result, expected)  # 使用测试模块验证期望结果和计算结果是否相等

    @pytest.mark.parametrize("test_ddof", [None, 0, 1, 2, 3])
    def test_cov_ddof(self, test_ddof):
        # GH#34611
        np_array1 = np.random.default_rng(2).random(10)  # 生成随机数列 np_array1
        np_array2 = np.random.default_rng(2).random(10)  # 生成随机数列 np_array2
        df = DataFrame({0: np_array1, 1: np_array2})  # 创建包含随机数列的数据框
        result = df.cov(ddof=test_ddof)  # 根据自由度参数计算数据框的协方差矩阵
        expected_np = np.cov(np_array1, np_array2, ddof=test_ddof)  # 计算期望的协方差矩阵
        expected = DataFrame(expected_np)  # 转换期望结果为数据框
        tm.assert_frame_equal(result, expected)  # 使用测试模块验证期望结果和计算结果是否相等

    @pytest.mark.parametrize(
        "other_column", [pd.array([1, 2, 3]), np.array([1.0, 2.0, 3.0])]
    )
    def test_cov_nullable_integer(self, other_column):
        # https://github.com/pandas-dev/pandas/issues/33803
        data = DataFrame({"a": pd.array([1, 2, None]), "b": other_column})  # 创建包含可空整数和其他列的数据框
        result = data.cov()  # 计算数据框的协方差矩阵
        arr = np.array([[0.5, 0.5], [0.5, 1.0]])  # 创建期望的协方差矩阵数组
        expected = DataFrame(arr, columns=["a", "b"], index=["a", "b"])  # 创建期望的协方差矩阵
        tm.assert_frame_equal(result, expected)  # 使用测试模块验证期望结果和计算结果是否相等
    # 使用 pytest 的参数化功能，对下面的测试方法进行多组参数化测试
    @pytest.mark.parametrize("numeric_only", [True, False])
    # 定义一个测试方法，用于测试在不同条件下计算协方差时的行为
    def test_cov_numeric_only(self, numeric_only):
        # 创建一个包含不同数据类型的 DataFrame
        df = DataFrame({"a": [1, 0], "c": ["x", "y"]})
        # 创建一个预期的 DataFrame，所有元素为 0.5
        expected = DataFrame(0.5, index=["a"], columns=["a"])
        
        # 根据 numeric_only 参数的值进行不同的测试分支
        if numeric_only:
            # 如果 numeric_only 为 True，则计算协方差并将结果存储在 result 中
            result = df.cov(numeric_only=numeric_only)
            # 断言计算结果与预期结果相等
            tm.assert_frame_equal(result, expected)
        else:
            # 如果 numeric_only 为 False，则预期会抛出 ValueError 异常，异常信息包含指定的字符串
            with pytest.raises(ValueError, match="could not convert string to float"):
                # 调用 cov 方法，期望抛出 ValueError 异常
                df.cov(numeric_only=numeric_only)
    # 定义 TestDataFrameCorr 类，用于测试 DataFrame 的相关性计算功能

    # 使用 pytest.mark.parametrize 装饰器为 test_corr_scipy_method 方法添加参数化测试，测试不同的相关性计算方法
    @pytest.mark.parametrize("method", ["pearson", "kendall", "spearman"])
    def test_corr_scipy_method(self, float_frame, method):
        # 导入 scipy 库，如果导入失败则跳过该测试
        pytest.importorskip("scipy")
        
        # 将 float_frame 的前五行的 "A" 列设为 NaN
        float_frame.loc[float_frame.index[:5], "A"] = np.nan
        # 将 float_frame 的第 5-9 行的 "B" 列设为 NaN
        float_frame.loc[float_frame.index[5:10], "B"] = np.nan
        # 将 float_frame 的前十行的 "A" 列设为 float_frame 中第 10-19 行 "A" 列的拷贝
        float_frame.loc[float_frame.index[:10], "A"] = float_frame["A"][10:20].copy()
        
        # 计算 float_frame 的相关性矩阵，使用给定的 method 参数
        correls = float_frame.corr(method=method)
        # 计算 float_frame 中 "A" 列与 "C" 列的相关性，使用给定的 method 参数
        expected = float_frame["A"].corr(float_frame["C"], method=method)
        # 使用 pytest 的 assert_almost_equal 断言函数比较计算结果与期望值的接近程度
        tm.assert_almost_equal(correls["A"]["C"], expected)

    # ---------------------------------------------------------------------

    # 定义 test_corr_non_numeric 方法，测试在 DataFrame 包含非数值类型数据时的相关性计算
    def test_corr_non_numeric(self, float_string_frame):
        # 使用 pytest.raises 断言捕获 ValueError 异常，检查在包含字符串时是否会引发异常
        with pytest.raises(ValueError, match="could not convert string to float"):
            float_string_frame.corr()
        
        # 计算只包含数值列的 float_string_frame 的相关性矩阵
        result = float_string_frame.corr(numeric_only=True)
        # 计算 float_string_frame 中 "A", "B", "C", "D" 列之间的期望相关性矩阵
        expected = float_string_frame.loc[:, ["A", "B", "C", "D"]].corr()
        # 使用 pandas.testing.assert_frame_equal 检查计算结果与期望值是否相等
        tm.assert_frame_equal(result, expected)

    # 使用 pytest.mark.parametrize 装饰器为 test_corr_nooverlap 方法添加参数化测试，测试在数据列没有重叠时的相关性计算
    @pytest.mark.parametrize("meth", ["pearson", "kendall", "spearman"])
    def test_corr_nooverlap(self, meth):
        # 输出"nothing in common"，表示测试数据集中没有共同的数据
        # 导入 scipy 库，如果导入失败则跳过该测试
        pytest.importorskip("scipy")
        
        # 创建包含特定数据的 DataFrame df
        df = DataFrame(
            {
                "A": [1, 1.5, 1, np.nan, np.nan, np.nan],
                "B": [np.nan, np.nan, np.nan, 1, 1.5, 1],
                "C": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )
        
        # 计算 df 的相关性矩阵，使用给定的 meth 参数
        rs = df.corr(meth)
        
        # 使用 assert 断言函数检查相关性矩阵中的特定关系
        assert isna(rs.loc["A", "B"])
        assert isna(rs.loc["B", "A"])
        assert rs.loc["A", "A"] == 1
        assert rs.loc["B", "B"] == 1
        assert isna(rs.loc["C", "C"])

    # 使用 pytest.mark.parametrize 装饰器为 test_corr_constant 方法添加参数化测试，测试在数据列包含常量值时的相关性计算
    @pytest.mark.parametrize("meth", ["pearson", "spearman"])
    def test_corr_constant(self, meth):
        # 输出"constant --> all NA"，表示测试数据集中的数据列包含常量值时会导致所有相关性值为 NA
        # 创建包含特定数据的 DataFrame df
        df = DataFrame(
            {
                "A": [1, 1, 1, np.nan, np.nan, np.nan],
                "B": [np.nan, np.nan, np.nan, 1, 1, 1],
            }
        )
        
        # 计算 df 的相关性矩阵，使用给定的 meth 参数
        rs = df.corr(meth)
        
        # 使用 assert 断言函数检查所有相关性值是否为 NA
        assert isna(rs.values).all()

    # 使用 pytest.mark.filterwarnings 装饰器忽略 RuntimeWarning 警告
    # 使用 pytest.mark.parametrize 装饰器为 test_corr_int_and_boolean 方法添加参数化测试，测试包含整数和布尔值类型数据时的相关性计算
    @pytest.mark.parametrize("meth", ["pearson", "kendall", "spearman"])
    def test_corr_int_and_boolean(self, meth):
        # 输出"when dtypes of pandas series are different"，表示测试数据集中包含不同类型的数据时需要适当处理
        # 导入 scipy 库，如果导入失败则跳过该测试
        pytest.importorskip("scipy")
        
        # 创建包含布尔值和整数的 DataFrame df
        df = DataFrame({"a": [True, False], "b": [1, 0]})
        
        # 创建期望的相关性矩阵 expected，所有值为 1
        expected = DataFrame(np.ones((2, 2)), index=["a", "b"], columns=["a", "b"])
        
        # 计算 df 的相关性矩阵，使用给定的 meth 参数
        result = df.corr(meth)
        
        # 使用 pandas.testing.assert_frame_equal 检查计算结果与期望值是否相等
        tm.assert_frame_equal(result, expected)
    # 定义一个测试方法，用于验证相关性和协方差计算时，独立性索引和列的情况
    def test_corr_cov_independent_index_column(self, method):
        # GH#14617
        # 创建一个包含随机标准正态分布数据的 DataFrame，共有10行4列
        df = DataFrame(
            np.random.default_rng(2).standard_normal(4 * 10).reshape(10, 4),
            columns=list("abcd"),
        )
        # 调用指定的 method 方法计算相关性或协方差
        result = getattr(df, method)()
        # 断言结果的行索引与列索引不相同
        assert result.index is not result.columns
        # 断言结果的行索引与列索引相等
        assert result.index.equals(result.columns)

    # 定义一个测试方法，用于验证当指定无效的计算方法时是否抛出 ValueError 异常
    def test_corr_invalid_method(self):
        # GH#22298
        # 创建一个包含随机正态分布数据的 DataFrame，共有10行2列
        df = DataFrame(np.random.default_rng(2).normal(size=(10, 2)))
        # 预期的错误消息
        msg = "method must be either 'pearson', 'spearman', 'kendall', or a callable, "
        # 使用 pytest 检查是否抛出 ValueError 异常，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            df.corr(method="____")

    # 定义一个测试方法，用于验证非 float64 类型数据的相关性计算情况
    def test_corr_int(self):
        # dtypes other than float64 GH#1761
        # 创建一个包含整数列的 DataFrame
        df = DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]})

        # 计算协方差
        df.cov()
        # 计算相关性
        df.corr()

    # 使用参数化测试标记，定义一个测试方法，验证对包含可空整数和其他列的数据进行相关性计算
    @pytest.mark.parametrize(
        "nullable_column", [pd.array([1, 2, 3]), pd.array([1, 2, None])]
    )
    @pytest.mark.parametrize(
        "other_column",
        [pd.array([1, 2, 3]), np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, np.nan])],
    )
    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
    def test_corr_nullable_integer(self, nullable_column, other_column, method):
        # https://github.com/pandas-dev/pandas/issues/33803
        # 导入 scipy，如果导入失败则跳过测试
        pytest.importorskip("scipy")
        # 创建包含可空整数和其他列的 DataFrame
        data = DataFrame({"a": nullable_column, "b": other_column})
        # 计算相关性
        result = data.corr(method=method)
        # 期望的结果 DataFrame，全为1
        expected = DataFrame(np.ones((2, 2)), columns=["a", "b"], index=["a", "b"])
        # 使用测试工具函数比较实际结果和期望结果
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，验证在计算相关性后是否正确处理了 item_cache 中的条目
    def test_corr_item_cache(self):
        # Check that corr does not lead to incorrect entries in item_cache

        # 创建一个包含整数列的 DataFrame
        df = DataFrame({"A": range(10)})
        df["B"] = range(10)[::-1]

        ser = df["A"]  # populate item_cache
        # 断言 DataFrame 内部块的数量为2
        assert len(df._mgr.blocks) == 2

        _ = df.corr(numeric_only=True)

        # 修改 Series 中的一个元素
        ser.iloc[0] = 99
        # 断言 DataFrame 中指定位置的元素未改变
        assert df.loc[0, "A"] == 0

    # 使用参数化测试标记，定义一个测试方法，验证对包含常数列的 DataFrame 进行相关性计算
    @pytest.mark.parametrize("length", [2, 20, 200, 2000])
    def test_corr_for_constant_columns(self, length):
        # GH: 37448
        # 创建一个包含常数列的 DataFrame
        df = DataFrame(length * [[0.4, 0.1]], columns=["A", "B"])
        # 计算相关性
        result = df.corr()
        # 期望的结果 DataFrame，全为 NaN
        expected = DataFrame(
            {"A": [np.nan, np.nan], "B": [np.nan, np.nan]}, index=["A", "B"]
        )
        # 使用测试工具函数比较实际结果和期望结果
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，验证在计算包含极小数值的 DataFrame 的相关性时是否正确
    def test_calc_corr_small_numbers(self):
        # GH: 37452
        # 创建一个包含极小数值的 DataFrame
        df = DataFrame(
            {"A": [1.0e-20, 2.0e-20, 3.0e-20], "B": [1.0e-20, 2.0e-20, 3.0e-20]}
        )
        # 计算相关性
        result = df.corr()
        # 期望的结果 DataFrame，全为1
        expected = DataFrame({"A": [1.0, 1.0], "B": [1.0, 1.0]}, index=["A", "B"])
        # 使用测试工具函数比较实际结果和期望结果
        tm.assert_frame_equal(result, expected)

    # 使用参数化测试标记，定义一个测试方法，验证不同方法下对 DataFrame 进行相关性计算
    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
    # 定义一个测试方法，测试在最小数据点数大于数据框长度时的相关性计算
    def test_corr_min_periods_greater_than_length(self, method):
        # 导入 scipy 库，如果不存在则跳过测试
        pytest.importorskip("scipy")
        # 创建一个包含两列（"A"和"B"）的数据框
        df = DataFrame({"A": [1, 2], "B": [1, 2]})
        # 调用数据框的相关性计算方法，设置最小数据点数为3
        result = df.corr(method=method, min_periods=3)
        # 创建预期的结果数据框，包含NaN值，与索引["A", "B"]对应
        expected = DataFrame(
            {"A": [np.nan, np.nan], "B": [np.nan, np.nan]}, index=["A", "B"]
        )
        # 使用测试工具比较计算结果和预期结果
        tm.assert_frame_equal(result, expected)

    # 使用参数化测试来测试不同的相关性方法和数值参数选项
    @pytest.mark.parametrize("meth", ["pearson", "kendall", "spearman"])
    @pytest.mark.parametrize("numeric_only", [True, False])
    def test_corr_numeric_only(self, meth, numeric_only):
        # 当 pandas Series 的数据类型不同时，
        # ndarray 的数据类型将为 object 类型，
        # 因此需要适当处理这种情况
        pytest.importorskip("scipy")
        # 创建一个包含三列（"a", "b", "c"）的数据框
        df = DataFrame({"a": [1, 0], "b": [1, 0], "c": ["x", "y"]})
        # 创建预期的结果数据框，全部为1.0，与索引["a", "b"]和列名["a", "b"]对应
        expected = DataFrame(np.ones((2, 2)), index=["a", "b"], columns=["a", "b"])
        # 如果 numeric_only 为 True，则执行相关性计算
        if numeric_only:
            result = df.corr(meth, numeric_only=numeric_only)
            # 使用测试工具比较计算结果和预期结果
            tm.assert_frame_equal(result, expected)
        else:
            # 否则，预期会抛出值错误，并匹配特定的错误消息
            with pytest.raises(ValueError, match="could not convert string to float"):
                df.corr(meth, numeric_only=numeric_only)
    @pytest.mark.parametrize(
        "dtype",
        [
            "float64",  # 定义参数化测试的数据类型为 float64
            "Float64",  # 定义参数化测试的数据类型为 Float64
            pytest.param("float64[pyarrow]", marks=td.skip_if_no("pyarrow")),  # 如果没有 pyarrow 模块，则跳过该测试
        ],
    )
    def test_corrwith(self, datetime_frame, dtype):
        datetime_frame = datetime_frame.astype(dtype)  # 将 datetime_frame 转换为指定的 dtype 类型

        a = datetime_frame  # 设置变量 a 为 datetime_frame

        # 生成随机噪声数据，并与 datetime_frame 相加形成 b
        noise = Series(np.random.default_rng(2).standard_normal(len(a)), index=a.index)
        b = datetime_frame.add(noise, axis=0)

        # 保证列的顺序和行的顺序不影响结果
        b = b.reindex(columns=b.columns[::-1], index=b.index[::-1][len(a) // 2 :])

        del b["B"]  # 删除 b 中的列 "B"

        # 计算列之间的相关性，并断言与单独计算 a["A"] 和 b["A"] 相关性的结果几乎相等
        colcorr = a.corrwith(b, axis=0)
        tm.assert_almost_equal(colcorr["A"], a["A"].corr(b["A"]))

        # 计算行之间的相关性，并断言与转置后计算的结果几乎相等
        rowcorr = a.corrwith(b, axis=1)
        tm.assert_series_equal(rowcorr, a.T.corrwith(b.T, axis=0))

        # 计算列之间的相关性，确保删除了指定的列后相关性结果与未删除时相等，并断言 "B" 不在 dropped 中
        dropped = a.corrwith(b, axis=0, drop=True)
        tm.assert_almost_equal(dropped["A"], a["A"].corr(b["A"]))
        assert "B" not in dropped

        # 计算行之间的相关性，确保删除了指定的行后指定行不在 dropped 的索引中
        dropped = a.corrwith(b, axis=1, drop=True)
        assert a.index[-1] not in dropped.index

    def test_corrwith_non_timeseries_data(self):
        # 创建随机数据帧 df1 和 df2，进行行相关性的断言测试
        index = ["a", "b", "c", "d", "e"]
        columns = ["one", "two", "three", "four"]
        df1 = DataFrame(
            np.random.default_rng(2).standard_normal((5, 4)),
            index=index,
            columns=columns,
        )
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=index[:4],
            columns=columns,
        )
        correls = df1.corrwith(df2, axis=1)
        for row in index[:4]:
            tm.assert_almost_equal(correls[row], df1.loc[row].corr(df2.loc[row]))

    def test_corrwith_with_objects(self, using_infer_string):
        # 创建包含对象数据类型的数据帧 df1 和 df2，并进行相关性测试
        df1 = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        df2 = df1.copy()
        cols = ["A", "B", "C", "D"]

        # 向 df1 和 df2 添加对象类型的列，并根据 using_infer_string 的条件进行不同的异常断言测试
        df1["obj"] = "foo"
        df2["obj"] = "bar"

        if using_infer_string:
            import pyarrow as pa

            # 如果使用 pyarrow，则断言会抛出 ArrowNotImplementedError 异常
            with pytest.raises(pa.lib.ArrowNotImplementedError, match="has no kernel"):
                df1.corrwith(df2)
        else:
            # 否则断言会抛出 TypeError 异常
            with pytest.raises(TypeError, match="Could not convert"):
                df1.corrwith(df2)

        # 对数值列进行相关性计算，并与预期结果进行比较
        result = df1.corrwith(df2, numeric_only=True)
        expected = df1.loc[:, cols].corrwith(df2.loc[:, cols])
        tm.assert_series_equal(result, expected)

        # 对于 axis=1 的相关性计算，断言会抛出 TypeError 异常
        with pytest.raises(TypeError, match="unsupported operand type"):
            df1.corrwith(df2, axis=1)

        # 再次计算 axis=1 的相关性，并与预期结果进行比较
        result = df1.corrwith(df2, axis=1, numeric_only=True)
        expected = df1.loc[:, cols].corrwith(df2.loc[:, cols], axis=1)
        tm.assert_series_equal(result, expected)
    # 定义一个测试方法，用于检验DataFrame中某一列与其它列的相关性
    def test_corrwith_series(self, datetime_frame):
        # 计算datetime_frame中各列与"A"列的相关性系数，并返回结果Series
        result = datetime_frame.corrwith(datetime_frame["A"])
        # 对datetime_frame中每一列应用与"A"列相关性的计算函数，得到期望的相关性系数
        expected = datetime_frame.apply(datetime_frame["A"].corr)

        # 使用测试框架检验计算结果与期望值是否相等
        tm.assert_series_equal(result, expected)

    # 定义一个测试方法，验证corrwith方法与np.corrcoef函数的结果接近
    def test_corrwith_matches_corrcoef(self):
        # 创建两个DataFrame，分别包含0到99的整数和其平方的列
        df1 = DataFrame(np.arange(100), columns=["a"])
        df2 = DataFrame(np.arange(100) ** 2, columns=["a"])
        # 使用corrwith方法计算两个DataFrame中"a"列的相关性系数
        c1 = df1.corrwith(df2)["a"]
        # 使用np.corrcoef计算两个DataFrame中"a"列的相关性系数
        c2 = np.corrcoef(df1["a"], df2["a"])[0][1]

        # 使用测试框架验证corrwith方法和np.corrcoef计算结果的近似程度
        tm.assert_almost_equal(c1, c2)
        # 使用断言确保相关性系数小于1
        assert c1 < 1

    # 使用参数化测试标记，定义一个测试方法，验证在混合数据类型的情况下corrwith方法的行为
    @pytest.mark.parametrize("numeric_only", [True, False])
    def test_corrwith_mixed_dtypes(self, numeric_only):
        # 创建包含整数和字符串列的DataFrame
        df = DataFrame(
            {"a": [1, 4, 3, 2], "b": [4, 6, 7, 3], "c": ["a", "b", "c", "d"]}
        )
        # 创建一个Series
        s = Series([0, 6, 7, 3])
        if numeric_only:
            # 如果numeric_only为True，使用corrwith方法计算数值列与Series的相关性系数
            result = df.corrwith(s, numeric_only=numeric_only)
            # 计算数值列与Series的相关性系数，期望得到一个Series
            corrs = [df["a"].corr(s), df["b"].corr(s)]
            expected = Series(data=corrs, index=["a", "b"])
            # 使用测试框架验证计算结果与期望值的一致性
            tm.assert_series_equal(result, expected)
        else:
            # 如果numeric_only为False，测试是否会抛出值错误异常
            with pytest.raises(
                ValueError,
                match="could not convert string to float",
            ):
                # 使用corrwith方法计算数值列与Series的相关性系数，预期会抛出异常
                df.corrwith(s, numeric_only=numeric_only)

    # 定义一个测试方法，验证corrwith方法在处理索引交集时的行为
    def test_corrwith_index_intersection(self):
        # 创建两个具有相同随机值的DataFrame
        df1 = DataFrame(
            np.random.default_rng(2).random(size=(10, 2)), columns=["a", "b"]
        )
        df2 = DataFrame(
            np.random.default_rng(2).random(size=(10, 3)), columns=["a", "b", "c"]
        )

        # 使用corrwith方法计算df1和df2在索引交集上的相关性，并获取结果的索引并排序
        result = df1.corrwith(df2, drop=True).index.sort_values()
        # 获取df1和df2的列交集，并排序得到期望的索引
        expected = df1.columns.intersection(df2.columns).sort_values()
        # 使用测试框架验证计算结果的索引与期望的索引的一致性
        tm.assert_index_equal(result, expected)

    # 定义一个测试方法，验证corrwith方法在处理索引并集时的行为
    def test_corrwith_index_union(self):
        # 创建两个具有相同随机值的DataFrame
        df1 = DataFrame(
            np.random.default_rng(2).random(size=(10, 2)), columns=["a", "b"]
        )
        df2 = DataFrame(
            np.random.default_rng(2).random(size=(10, 3)), columns=["a", "b", "c"]
        )

        # 使用corrwith方法计算df1和df2在索引并集上的相关性，并获取结果的索引并排序
        result = df1.corrwith(df2, drop=False).index.sort_values()
        # 获取df1和df2的列并集，并排序得到期望的索引
        expected = df1.columns.union(df2.columns).sort_values()
        # 使用测试框架验证计算结果的索引与期望的索引的一致性
        tm.assert_index_equal(result, expected)

    # 定义一个测试方法，验证corrwith方法在处理重复列时的行为
    def test_corrwith_dup_cols(self):
        # 创建一个具有重复列的DataFrame
        df1 = DataFrame(np.vstack([np.arange(10)] * 3).T)
        df2 = df1.copy()
        df2 = pd.concat((df2, df2[0]), axis=1)

        # 使用corrwith方法计算df1和df2的相关性系数
        result = df1.corrwith(df2)
        # 期望得到一个Series，包含所有列与自身的相关性系数为1
        expected = Series(np.ones(4), index=[0, 0, 1, 2])
        # 使用测试框架验证计算结果与期望值的一致性
        tm.assert_series_equal(result, expected)

    # 定义一个测试方法，验证DataFrame的corr方法在处理数值不稳定性时的行为
    def test_corr_numerical_instabilities(self):
        # 创建一个包含浮点数的DataFrame
        df = DataFrame([[0.2, 0.4], [0.4, 0.2]])
        # 使用corr方法计算相关性矩阵
        result = df.corr()
        # 期望得到一个DataFrame，与数值稳定性修正后的参考DataFrame相等
        expected = DataFrame({0: [1.0, -1.0], 1: [-1.0, 1.0]})
        # 使用测试框架验证计算结果与期望值的一致性，允许误差为1e-17
        tm.assert_frame_equal(result - 1, expected - 1, atol=1e-17)
    def test_corrwith_spearman(self):
        # GH#21925
        # 确保 scipy 库已导入，否则跳过该测试
        pytest.importorskip("scipy")
        # 创建一个包含100行和3列随机数的数据框
        df = DataFrame(np.random.default_rng(2).random(size=(100, 3)))
        # 计算数据框中每一列与自身平方的斯皮尔曼相关系数
        result = df.corrwith(df**2, method="spearman")
        # 期望结果是与结果长度相同的全1系列
        expected = Series(np.ones(len(result)))
        # 断言结果与期望相等
        tm.assert_series_equal(result, expected)

    def test_corrwith_kendall(self):
        # GH#21925
        # 确保 scipy 库已导入，否则跳过该测试
        pytest.importorskip("scipy")
        # 创建一个包含100行和3列随机数的数据框
        df = DataFrame(np.random.default_rng(2).random(size=(100, 3)))
        # 计算数据框中每一列与自身平方的肯德尔相关系数
        result = df.corrwith(df**2, method="kendall")
        # 期望结果是与结果长度相同的全1系列
        expected = Series(np.ones(len(result)))
        # 断言结果与期望相等
        tm.assert_series_equal(result, expected)

    def test_corrwith_spearman_with_tied_data(self):
        # GH#48826
        # 确保 scipy 库已导入，否则跳过该测试
        pytest.importorskip("scipy")
        # 创建第一个数据框，包含列"A", "B", "C"，其中包括NaN值和布尔值
        df1 = DataFrame(
            {
                "A": [1, np.nan, 7, 8],
                "B": [False, True, True, False],
                "C": [10, 4, 9, 3],
            }
        )
        # 从第一个数据框中选择"B"和"C"列，创建第二个数据框
        df2 = df1[["B", "C"]]
        # 计算第一个数据框加1后与第二个数据框"B"列之间的斯皮尔曼相关系数
        result = (df1 + 1).corrwith(df2.B, method="spearman")
        # 期望结果是包含索引"A", "B", "C"的系列，其值分别为0.0, 1.0, 0.0
        expected = Series([0.0, 1.0, 0.0], index=["A", "B", "C"])
        # 断言结果与期望相等
        tm.assert_series_equal(result, expected)

        # 创建包含布尔值的第一个数据框
        df_bool = DataFrame(
            {"A": [True, True, False, False], "B": [True, False, False, True]}
        )
        # 创建包含布尔值的系列
        ser_bool = Series([True, True, False, True])
        # 计算第一个数据框与布尔值系列之间的相关系数
        result = df_bool.corrwith(ser_bool)
        # 期望结果是包含索引"A", "B"的系列，其值分别约为0.57735
        expected = Series([0.57735, 0.57735], index=["A", "B"])
        # 断言结果与期望相等
        tm.assert_series_equal(result, expected)

    def test_corrwith_min_periods_method(self):
        # GH#9490
        # 确保 scipy 库已导入，否则跳过该测试
        pytest.importorskip("scipy")
        # 创建第一个数据框，包含列"A", "B", "C"，其中包括NaN值和布尔值
        df1 = DataFrame(
            {
                "A": [1, np.nan, 7, 8],
                "B": [False, True, True, False],
                "C": [10, 4, 9, 3],
            }
        )
        # 从第一个数据框中选择"B"和"C"列，创建第二个数据框
        df2 = df1[["B", "C"]]
        # 计算第一个数据框加1后与第二个数据框"B"列之间的斯皮尔曼相关系数，最小观测期数为2
        result = (df1 + 1).corrwith(df2.B, method="spearman", min_periods=2)
        # 期望结果是包含索引"A", "B", "C"的系列，其值分别为0.0, 1.0, 0.0
        expected = Series([0.0, 1.0, 0.0], index=["A", "B", "C"])
        # 断言结果与期望相等
        tm.assert_series_equal(result, expected)

    def test_corrwith_min_periods_boolean(self):
        # GH#9490
        # 创建包含布尔值的数据框
        df_bool = DataFrame(
            {"A": [True, True, False, False], "B": [True, False, False, True]}
        )
        # 创建包含布尔值的系列
        ser_bool = Series([True, True, False, True])
        # 计算数据框与布尔值系列之间的相关系数，最小观测期数为3
        result = df_bool.corrwith(ser_bool, min_periods=3)
        # 期望结果是包含索引"A", "B"的系列，其值分别约为0.57735
        expected = Series([0.57735, 0.57735], index=["A", "B"])
        # 断言结果与期望相等
        tm.assert_series_equal(result, expected)
```