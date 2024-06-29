# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_astype.py`

```
    from datetime import (  # 导入datetime和timedelta模块
        datetime,         # 导入datetime类，用于处理日期和时间
        timedelta,        # 导入timedelta类，用于时间间隔的计算
    )
    from importlib import reload  # 导入reload函数，用于重新加载模块
    import string  # 导入string模块，包含处理字符串的相关函数和常量
    import sys  # 导入sys模块，提供对Python解释器的访问

    import numpy as np  # 导入NumPy库，并用np作为别名
    import pytest  # 导入pytest库，用于编写和运行测试

    from pandas._libs.tslibs import iNaT  # 导入iNaT，时间序列的空值表示
    import pandas.util._test_decorators as td  # 导入测试装饰器模块

    from pandas import (  # 从pandas库中导入多个类和函数
        NA,                   # 缺失数据的标记
        Categorical,          # 用于表示分类数据的类
        CategoricalDtype,     # 用于指定分类数据类型的类
        DatetimeTZDtype,      # 带有时区的日期时间数据类型
        Index,                # 表示pandas对象的索引结构
        Interval,             # 表示一段连续的数值区间
        NaT,                  # 表示不可用的日期时间数据
        Series,               # 表示一维标签数组，可以存储各种数据类型
        Timedelta,            # 表示时间间隔
        Timestamp,            # 表示时间戳
        cut,                  # 用于根据数值范围或分位数将数值变量分段的函数
        date_range,           # 生成日期范围
        to_datetime,          # 将输入转换为DatetimeIndex或Series的日期时间工具
    )
    import pandas._testing as tm  # 导入pandas测试模块，并用tm作为别名

    def rand_str(nchars: int) -> str:  # 定义函数rand_str，生成指定长度的随机字符串
        """
        Generate one random byte string.
        """
        RANDS_CHARS = np.array(  # 创建包含字母和数字的数组
            list(string.ascii_letters + string.digits), dtype=(np.str_, 1)
        )
        return "".join(np.random.default_rng(2).choice(RANDS_CHARS, nchars))  # 生成指定长度的随机字符串并返回

    class TestAstypeAPI:  # 定义测试类TestAstypeAPI
        def test_astype_unitless_dt64_raises(self):  # 测试方法，测试astype对unitless dt64的异常处理
            # GH#47844
            ser = Series(["1970-01-01", "1970-01-01", "1970-01-01"], dtype="datetime64[ns]")  # 创建包含日期字符串的Series对象
            df = ser.to_frame()  # 将Series对象转换为DataFrame对象

            msg = "Casting to unit-less dtype 'datetime64' is not supported"  # 异常消息
            with pytest.raises(TypeError, match=msg):  # 断言捕获TypeError并检查异常消息
                ser.astype(np.datetime64)
            with pytest.raises(TypeError, match=msg):
                df.astype(np.datetime64)
            with pytest.raises(TypeError, match=msg):
                ser.astype("datetime64")
            with pytest.raises(TypeError, match=msg):
                df.astype("datetime64")

        def test_arg_for_errors_in_astype(self):  # 测试方法，测试astype中错误参数的处理
            # see GH#14878
            ser = Series([1, 2, 3])  # 创建包含整数的Series对象

            msg = (
                r"Expected value of kwarg 'errors' to be one of \['raise', "
                r"'ignore'\]\. Supplied value is 'False'"
            )  # 异常消息的正则表达式
            with pytest.raises(ValueError, match=msg):  # 断言捕获ValueError并检查异常消息
                ser.astype(np.float64, errors=False)

            ser.astype(np.int8, errors="raise")  # 将Series对象转换为np.int8类型，错误参数为"raise"

        @pytest.mark.parametrize("dtype_class", [dict, Series])  # 参数化测试，测试dtype_class参数
    # 定义一个测试方法，用于测试将 Series 转换为类似字典的类型
    def test_astype_dict_like(self, dtype_class):
        # 创建一个 Series 对象，包含从 0 到 8 的偶数，命名为 "abc"
        ser = Series(range(0, 10, 2), name="abc")

        # 创建一个指定了 "abc" 列为字符串类型的 dtype 对象，并将 Series 转换为该类型
        dt1 = dtype_class({"abc": str})
        result = ser.astype(dt1)
        # 期望结果是将 Series 的值转换为字符串类型，与原名和类型保持一致
        expected = Series(["0", "2", "4", "6", "8"], name="abc", dtype=object)
        tm.assert_series_equal(result, expected)

        # 创建一个指定了 "abc" 列为 float64 类型的 dtype 对象，并将 Series 转换为该类型
        dt2 = dtype_class({"abc": "float64"})
        result = ser.astype(dt2)
        # 期望结果是将 Series 的值转换为 float64 类型，与原名和类型保持一致
        expected = Series([0.0, 2.0, 4.0, 6.0, 8.0], dtype="float64", name="abc")
        tm.assert_series_equal(result, expected)

        # 创建一个指定了 "abc" 和 "def" 列为字符串类型的 dtype 对象，并尝试将 Series 转换为该类型
        dt3 = dtype_class({"abc": str, "def": str})
        # 检查是否会引发 KeyError 异常，提示只有 Series 名称可以用于 Series dtype 映射的键
        msg = (
            "Only the Series name can be used for the key in Series dtype "
            r"mappings\."
        )
        with pytest.raises(KeyError, match=msg):
            ser.astype(dt3)

        # 创建一个指定了索引 0 为字符串类型的 dtype 对象，并尝试将 Series 转换为该类型
        dt4 = dtype_class({0: str})
        # 再次检查是否会引发 KeyError 异常，与前面相同的异常消息
        with pytest.raises(KeyError, match=msg):
            ser.astype(dt4)

        # 如果 dtype_class 是 Series 类型，创建一个空的对象，尝试将 Series 转换为该类型
        # 预期会引发与前面相同的 KeyError 异常消息
        if dtype_class is Series:
            dt5 = dtype_class({}, dtype=object)
        else:
            dt5 = dtype_class({})
        with pytest.raises(KeyError, match=msg):
            ser.astype(dt5)
class TestAstype:
    @pytest.mark.parametrize("tz", [None, "UTC", "US/Pacific"])
    def test_astype_object_to_dt64_non_nano(self, tz):
        # GH#55756, GH#54620
        # 创建一个指定日期的时间戳对象
        ts = Timestamp("2999-01-01")
        # 设置目标数据类型为微秒精度的日期时间类型
        dtype = "M8[us]"
        if tz is not None:
            # 如果时区不为空，则在数据类型中包含时区信息
            dtype = f"M8[us, {tz}]"
        # 包含多种数据类型的序列
        vals = [ts, "2999-01-02 03:04:05.678910", 2500]
        # 创建一个对象类型的序列
        ser = Series(vals, dtype=object)
        # 将序列转换为指定的数据类型
        result = ser.astype(dtype)

        # 将数值 2500 解释为微秒，与从 vals[:2] 和 vals[2:] 创建 DatetimeIndexes 后合并结果一致
        # 通过对前两个和最后一个值进行日期时间操作得到 pointwise 列表
        pointwise = [
            vals[0].tz_localize(tz),
            Timestamp(vals[1], tz=tz),
            to_datetime(vals[2], unit="us", utc=True).tz_convert(tz),
        ]
        # 期望的数值列表，转换为微秒精度的日期时间类型
        exp_vals = [x.as_unit("us").asm8 for x in pointwise]
        # 创建一个 NumPy 数组，指定数据类型为微秒精度的日期时间类型
        exp_arr = np.array(exp_vals, dtype="M8[us]")
        expected = Series(exp_arr, dtype="M8[us]")
        if tz is not None:
            # 如果时区不为空，则在预期结果中进行时区本地化和转换
            expected = expected.dt.tz_localize("UTC").dt.tz_convert(tz)
        # 断言两个序列是否相等
        tm.assert_series_equal(result, expected)

    def test_astype_mixed_object_to_dt64tz(self):
        # pre-2.0 this raised ValueError bc of tz mismatch
        # xref GH#32581
        # 创建带有指定时区的时间戳对象
        ts = Timestamp("2016-01-04 05:06:07", tz="US/Pacific")
        # 将 ts 对象转换为指定时区的时间戳对象
        ts2 = ts.tz_convert("Asia/Tokyo")

        # 创建一个包含两个对象类型时间戳的序列
        ser = Series([ts, ts2], dtype=object)
        # 将序列中的元素转换为指定的日期时间类型
        res = ser.astype("datetime64[ns, Europe/Brussels]")
        expected = Series(
            [ts.tz_convert("Europe/Brussels"), ts2.tz_convert("Europe/Brussels")],
            dtype="datetime64[ns, Europe/Brussels]",
        )
        # 断言两个序列是否相等
        tm.assert_series_equal(res, expected)

    @pytest.mark.parametrize("dtype", np.typecodes["All"])
    def test_astype_empty_constructor_equality(self, dtype):
        # see GH#15524
        # 如果数据类型不是以下几种，则创建一个空序列，将其转换为指定的数据类型，然后断言两者是否相等
        if dtype not in (
            "S",
            "V",  # poor support (if any) currently
            "M",
            "m",  # Generic timestamps raise a ValueError. Already tested.
        ):
            init_empty = Series([], dtype=dtype)
            as_type_empty = Series([]).astype(dtype)
            tm.assert_series_equal(init_empty, as_type_empty)

    @pytest.mark.parametrize("dtype", [str, np.str_])
    @pytest.mark.parametrize(
        "data",
        [
            [string.digits * 10, rand_str(63), rand_str(64), rand_str(1000)],
            [string.digits * 10, rand_str(63), rand_str(64), np.nan, 1.0],
        ],
    )
    def test_astype_str_map(self, dtype, data, using_infer_string):
        # see GH#4405
        # 创建一个包含指定数据的序列
        series = Series(data)
        # 将序列中的元素转换为指定的数据类型
        result = series.astype(dtype)
        # 期望结果是将序列中的每个元素转换为字符串
        expected = series.map(str)
        if using_infer_string:
            # 如果使用了推断字符串，则将期望结果转换为对象类型
            expected = expected.astype(object)
        # 断言两个序列是否相等
        tm.assert_series_equal(result, expected)

    def test_astype_float_to_period(self):
        # 将包含 NaN 值的序列转换为周期类型
        result = Series([np.nan]).astype("period[D]")
        expected = Series([NaT], dtype="period[D]")
        # 断言两个序列是否相等
        tm.assert_series_equal(result, expected)
    # 测试当未指定 Pandas 数据类型时的 astype 方法行为
    def test_astype_no_pandas_dtype(self):
        # 使用 Series 创建一个包含整数的序列，指定数据类型为 int64
        ser = Series([1, 2], dtype="int64")
        # 由于没有公共 API 中的 NumpyEADtype，因此使用 `.array.dtype`，
        # 它是一个 NumpyEADtype。
        result = ser.astype(ser.array.dtype)
        # 断言结果序列与原序列相等
        tm.assert_series_equal(result, ser)

    # 使用 pytest 的参数化功能测试 astype 方法对日期时间和时间间隔的行为
    @pytest.mark.parametrize("dtype", [np.datetime64, np.timedelta64])
    def test_astype_generic_timestamp_no_frequency(self, dtype, request):
        # 参考 GitHub 问题 GH#15524 和 GH#15987
        data = [1]
        ser = Series(data)

        # 如果 dtype 的名称不是 "timedelta64" 或 "datetime64"，
        # 则标记测试为失败，原因是 GH#33890 要求分配 ns 单位
        if np.dtype(dtype).name not in ["timedelta64", "datetime64"]:
            mark = pytest.mark.xfail(reason="GH#33890 Is assigned ns unit")
            request.applymarker(mark)

        # 构建错误消息，指出特定 dtype 没有单位，请使用正确格式
        msg = (
            rf"The '{dtype.__name__}' dtype has no unit\. "
            rf"Please pass in '{dtype.__name__}\[ns\]' instead."
        )
        # 断言抛出 ValueError 并匹配特定的错误消息
        with pytest.raises(ValueError, match=msg):
            ser.astype(dtype)

    # 测试将 datetime64 转换为字符串的正确性
    def test_astype_dt64_to_str(self):
        # GH#10442：验证将 Series/DatetimeIndex 转换为 str 的正确性
        dti = date_range("2012-01-01", periods=3)
        result = Series(dti).astype(str)
        expected = Series(["2012-01-01", "2012-01-02", "2012-01-03"], dtype=object)
        # 断言结果序列与期望序列相等
        tm.assert_series_equal(result, expected)

    # 测试将带时区信息的 datetime64 转换为字符串的正确性
    def test_astype_dt64tz_to_str(self):
        # GH#10442：验证将 Series/DatetimeIndex 转换为 str 的正确性
        dti_tz = date_range("2012-01-01", periods=3, tz="US/Eastern")
        result = Series(dti_tz).astype(str)
        expected = Series(
            [
                "2012-01-01 00:00:00-05:00",
                "2012-01-02 00:00:00-05:00",
                "2012-01-03 00:00:00-05:00",
            ],
            dtype=object,
        )
        # 断言结果序列与期望序列相等
        tm.assert_series_equal(result, expected)

    # 测试将 datetime64 转换为其他对象类型的正确性
    def test_astype_datetime(self, unit):
        # 使用 iNaT 创建一个 Series，指定数据类型为特定的日期时间单位
        ser = Series(iNaT, dtype=f"M8[{unit}]", index=range(5))

        # 将序列转换为一般对象类型
        ser = ser.astype("O")
        # 断言序列的 dtype 为 np.object_
        assert ser.dtype == np.object_

        # 使用包含单个 datetime 的 Series
        ser = Series([datetime(2001, 1, 2, 0, 0)])

        # 将序列转换为一般对象类型
        ser = ser.astype("O")
        # 断言序列的 dtype 为 np.object_
        assert ser.dtype == np.object_

        # 使用包含多个 datetime 的 Series，指定数据类型为特定的日期时间单位
        ser = Series(
            [datetime(2001, 1, 2, 0, 0) for i in range(3)], dtype=f"M8[{unit}]"
        )

        # 修改序列的第二个元素为 np.nan
        ser[1] = np.nan
        # 断言序列的 dtype 仍为特定的日期时间单位
        assert ser.dtype == f"M8[{unit}]"

        # 将序列转换为一般对象类型
        ser = ser.astype("O")
        # 断言序列的 dtype 为 np.object_
        assert ser.dtype == np.object_
    def test_astype_datetime64tz(self):
        # 创建一个带有时区的日期时间序列
        ser = Series(date_range("20130101", periods=3, tz="US/Eastern"))

        # 使用 astype 方法将序列转换为 object 类型
        result = ser.astype(object)
        expected = Series(ser.astype(object), dtype=object)
        tm.assert_series_equal(result, expected)

        # 将日期时间序列的值转换为 UTC 时区后再转换回原始时区，验证转换后的结果与原始序列一致
        result = Series(ser.values).dt.tz_localize("UTC").dt.tz_convert(ser.dt.tz)
        tm.assert_series_equal(result, ser)

        # 使用 astype 将序列转换为 object 类型，验证结果与预期相同
        result = Series(ser.astype(object))
        expected = ser.astype(object)
        tm.assert_series_equal(result, expected)

        # 使用 astype 将序列转换为 datetime64[ns, tz] 类型时，预期会抛出 TypeError 异常
        msg = "Cannot use .astype to convert from timezone-naive"
        with pytest.raises(TypeError, match=msg):
            Series(ser.values).astype("datetime64[ns, US/Eastern]")

        with pytest.raises(TypeError, match=msg):
            Series(ser.values).astype(ser.dtype)

        # 使用 astype 将序列转换为指定时区的 datetime64[ns, tz] 类型，验证转换结果与预期相同
        result = ser.astype("datetime64[ns, CET]")
        expected = Series(date_range("20130101 06:00:00", periods=3, tz="CET"))
        tm.assert_series_equal(result, expected)

    def test_astype_str_cast_dt64(self):
        # 测试将日期时间序列转换为字符串类型时的情况，参考 GitHub issue #9757
        ts = Series([Timestamp("2010-01-04 00:00:00")])
        res = ts.astype(str)

        expected = Series(["2010-01-04"], dtype=object)
        tm.assert_series_equal(res, expected)

        ts = Series([Timestamp("2010-01-04 00:00:00", tz="US/Eastern")])
        res = ts.astype(str)

        expected = Series(["2010-01-04 00:00:00-05:00"], dtype=object)
        tm.assert_series_equal(res, expected)

    def test_astype_str_cast_td64(self):
        # 测试将时间增量序列转换为字符串类型时的情况，参考 GitHub issue #9757
        td = Series([Timedelta(1, unit="D")])
        ser = td.astype(str)

        expected = Series(["1 days"], dtype=object)
        tm.assert_series_equal(ser, expected)

    def test_dt64_series_astype_object(self):
        # 测试将 datetime64 类型的序列转换为 object 类型时的情况
        dt64ser = Series(date_range("20130101", periods=3))
        result = dt64ser.astype(object)
        assert isinstance(result.iloc[0], datetime)
        assert result.dtype == np.object_

    def test_td64_series_astype_object(self):
        # 测试将 timedelta64 类型的序列转换为 object 类型时的情况
        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="timedelta64[ns]")
        result = tdser.astype(object)
        assert isinstance(result.iloc[0], timedelta)
        assert result.dtype == np.object_

    @pytest.mark.parametrize(
        "data, dtype",
        [
            (["x", "y", "z"], "string[python]"),
            pytest.param(
                ["x", "y", "z"],
                "string[pyarrow]",
                marks=td.skip_if_no("pyarrow"),
            ),
            (["x", "y", "z"], "category"),
            (3 * [Timestamp("2020-01-01", tz="UTC")], None),
            (3 * [Interval(0, 1)], None),
        ],
    )
    @pytest.mark.parametrize("errors", ["raise", "ignore"])
    # 测试方法：测试对于扩展数据类型的 astype 方法是否忽略错误
    def test_astype_ignores_errors_for_extension_dtypes(self, data, dtype, errors):
        # GitHub issue链接：https://github.com/pandas-dev/pandas/issues/35471
        # 创建一个 Series 对象，使用给定的数据和数据类型
        ser = Series(data, dtype=dtype)
        # 如果 errors 参数为 "ignore"，则期望结果与原始序列相同，执行 astype 操作时忽略错误
        if errors == "ignore":
            expected = ser
            result = ser.astype(float, errors="ignore")
            # 断言结果与期望相等
            tm.assert_series_equal(result, expected)
        else:
            # 否则，期望抛出 ValueError 或 TypeError 异常，匹配错误信息
            msg = "(Cannot cast)|(could not convert)"
            with pytest.raises((ValueError, TypeError), match=msg):
                ser.astype(float, errors=errors)

    # 测试方法：测试从浮点数到字符串的 astype 方法
    def test_astype_from_float_to_str(self, any_float_dtype):
        # GitHub issue链接：https://github.com/pandas-dev/pandas/issues/36451
        # 创建一个包含单个浮点数的 Series 对象，使用任意浮点数类型
        ser = Series([0.1], dtype=any_float_dtype)
        # 执行 astype 操作，将数据类型转换为字符串
        result = ser.astype(str)
        # 创建一个期望的 Series 对象，其中数据被转换为字符串
        expected = Series(["0.1"], dtype=object)
        # 断言结果与期望相等
        tm.assert_series_equal(result, expected)

    # 测试方法：测试 astype 方法将特定值转换为字符串时保留 NA 值
    def test_astype_to_str_preserves_na(self, value, string_value):
        # GitHub issue链接：https://github.com/pandas-dev/pandas/issues/36904
        # 创建一个包含字符串和特定值的 Series 对象，数据类型为对象类型
        ser = Series(["a", "b", value], dtype=object)
        # 执行 astype 操作，将数据类型转换为字符串
        result = ser.astype(str)
        # 创建一个期望的 Series 对象，保留了特定值的字符串表示
        expected = Series(["a", "b", string_value], dtype=object)
        # 断言结果与期望相等
        tm.assert_series_equal(result, expected)

    # 测试方法：测试 astype 方法将随机生成的数据转换为指定的数据类型
    @pytest.mark.parametrize("dtype", ["float32", "float64", "int64", "int32"])
    def test_astype(self, dtype):
        # 使用随机生成的正态分布数据创建一个 Series 对象
        ser = Series(np.random.default_rng(2).standard_normal(5), name="foo")
        # 执行 astype 操作，将数据类型转换为指定的 dtype
        as_typed = ser.astype(dtype)

        # 断言转换后的数据类型与指定的 dtype 相等
        assert as_typed.dtype == dtype
        # 断言转换后的 Series 名称与原始 Series 的名称相同
        assert as_typed.name == ser.name

    # 测试方法：测试 astype 方法将 NaN 和 Inf 转换为整数时是否抛出异常
    @pytest.mark.parametrize("value", [np.nan, np.inf])
    def test_astype_cast_nan_inf_int(self, any_int_numpy_dtype, value):
        # GitHub issue链接：https://github.com/pandas-dev/pandas/issues/14265
        # 创建一个包含特定值的 Series 对象
        ser = Series([value])
        # 期望抛出异常的错误信息
        msg = "Cannot convert non-finite values \\(NA or inf\\) to integer"

        # 断言执行 astype 操作时抛出 ValueError 异常，并匹配预期的错误信息
        with pytest.raises(ValueError, match=msg):
            ser.astype(any_int_numpy_dtype)

    # 测试方法：测试 astype 方法将对象类型转换为整数时是否抛出异常
    def test_astype_cast_object_int_fail(self, any_int_numpy_dtype):
        # 创建一个包含字符串的 Series 对象
        arr = Series(["car", "house", "tree", "1"])
        # 期望抛出异常的错误信息
        msg = r"invalid literal for int\(\) with base 10: 'car'"

        # 断言执行 astype 操作时抛出 ValueError 异常，并匹配预期的错误信息
        with pytest.raises(ValueError, match=msg):
            arr.astype(any_int_numpy_dtype)

    # 测试方法：测试 astype 方法将浮点数转换为无符号整数时是否抛出异常
    def test_astype_float_to_uint_negatives_raise(
        self, float_numpy_dtype, any_unsigned_int_numpy_dtype
    ):
        # GH#45151 我们不会将负数转换为无意义的值
        # TODO: 对于 EA 浮点数/无符号整数数据类型，有符号整数也应该类似处理？
        arr = np.arange(5).astype(float_numpy_dtype) - 3  # 包括负数
        ser = Series(arr)

        msg = "Cannot losslessly cast from .* to .*"
        with pytest.raises(ValueError, match=msg):
            ser.astype(any_unsigned_int_numpy_dtype)

        with pytest.raises(ValueError, match=msg):
            ser.to_frame().astype(any_unsigned_int_numpy_dtype)

        with pytest.raises(ValueError, match=msg):
            # 我们当前在 Index.astype 中捕获并重新引发异常
            Index(ser).astype(any_unsigned_int_numpy_dtype)

        with pytest.raises(ValueError, match=msg):
            ser.array.astype(any_unsigned_int_numpy_dtype)

    def test_astype_cast_object_int(self):
        arr = Series(["1", "2", "3", "4"], dtype=object)
        result = arr.astype(int)

        tm.assert_series_equal(result, Series(np.arange(1, 5)))

    def test_astype_unicode(self, using_infer_string):
        # 参见 GH#7758: 设置默认编码为 utf-8 需要一点技巧
        digits = string.digits
        test_series = [
            Series([digits * 10, rand_str(63), rand_str(64), rand_str(1000)]),
            Series(["データーサイエンス、お前はもう死んでいる"]),
        ]

        former_encoding = None

        if sys.getdefaultencoding() == "utf-8":
            # GH#45326 从 2.0 开始，Series.astype 与 Index.astype 匹配，通过 obj.decode() 处理字节而非 str(obj)
            item = "野菜食べないとやばい"
            ser = Series([item.encode()])
            result = ser.astype(np.str_)
            expected = Series([item], dtype=object)
            tm.assert_series_equal(result, expected)

        for ser in test_series:
            res = ser.astype(np.str_)
            expec = ser.map(str)
            if using_infer_string:
                expec = expec.astype(object)
            tm.assert_series_equal(res, expec)

        # 恢复先前的编码设置
        if former_encoding is not None and former_encoding != "utf-8":
            reload(sys)
            sys.setdefaultencoding(former_encoding)

    def test_astype_bytes(self):
        # GH#39474
        result = Series(["foo", "bar", "baz"]).astype(bytes)
        assert result.dtypes == np.dtype("S3")

    def test_astype_nan_to_bool(self):
        # GH#43018
        ser = Series(np.nan, dtype="object")
        result = ser.astype("bool")
        expected = Series(True, dtype="bool")
        tm.assert_series_equal(result, expected)
    # 测试函数：测试将任何数值类型的序列转换为带时区的日期时间数据类型
    def test_astype_ea_to_datetimetzdtype(self, any_numeric_ea_dtype):
        # 创建一个包含数值的序列，指定数据类型为传入的任何数值扩展数组数据类型
        ser = Series([4, 0, 9], dtype=any_numeric_ea_dtype)
        # 将序列的数据类型转换为带有指定时区（US/Pacific）的日期时间数据类型
        result = ser.astype(DatetimeTZDtype(tz="US/Pacific"))

        # 预期结果：创建一个包含时间戳的序列，每个时间戳都带有指定时区（US/Pacific）
        expected = Series(
            {
                0: Timestamp("1969-12-31 16:00:00.000000004-08:00", tz="US/Pacific"),
                1: Timestamp("1969-12-31 16:00:00.000000000-08:00", tz="US/Pacific"),
                2: Timestamp("1969-12-31 16:00:00.000000009-08:00", tz="US/Pacific"),
            }
        )

        # 使用测试框架检查结果序列与预期序列是否相等
        tm.assert_series_equal(result, expected)

    # 测试函数：测试在类型转换后保留序列的属性
    def test_astype_retain_attrs(self, any_numpy_dtype):
        # 创建一个包含整数的序列
        ser = Series([0, 1, 2, 3])
        # 为序列设置自定义属性"Location"为"Michigan"
        ser.attrs["Location"] = "Michigan"

        # 将序列的数据类型转换为传入的任何NumPy数据类型，并获取转换后的序列的属性
        result = ser.astype(any_numpy_dtype).attrs
        # 期望结果：与原始序列的属性相同
        expected = ser.attrs

        # 使用测试框架检查结果属性与预期属性是否相等
        tm.assert_dict_equal(expected, result)
# 定义一个测试类 TestAstypeString，用于测试字符串类型转换的功能
class TestAstypeString:
    # 使用 pytest 的参数化标记，定义多组参数化测试数据
    @pytest.mark.parametrize(
        "data, dtype",
        [
            ([True, NA], "boolean"),  # 布尔类型转换测试数据
            (["A", NA], "category"),  # 分类类型转换测试数据
            (["2020-10-10", "2020-10-10"], "datetime64[ns]"),  # 日期时间类型转换测试数据
            (["2020-10-10", "2020-10-10", NaT], "datetime64[ns]"),  # 含缺失值的日期时间类型转换测试数据
            (
                ["2012-01-01 00:00:00-05:00", NaT],
                "datetime64[ns, US/Eastern]",
            ),  # 带时区信息的日期时间类型转换测试数据
            ([1, None], "UInt16"),  # 无符号整数类型转换测试数据
            (["1/1/2021", "2/1/2021"], "period[M]"),  # 时期类型转换测试数据
            (["1/1/2021", "2/1/2021", NaT], "period[M]"),  # 含缺失值的时期类型转换测试数据
            (["1 Day", "59 Days", NaT], "timedelta64[ns]"),  # 时间间隔类型转换测试数据
            # 目前无法从字符串列表解析 IntervalArray
        ],
    )
    # 定义测试方法 test_astype_string_to_extension_dtype_roundtrip，接受参数化测试数据
    def test_astype_string_to_extension_dtype_roundtrip(
        self, data, dtype, request, nullable_string_dtype
    ):
        # 如果数据类型为 boolean，则标记为预期失败，添加失败原因说明
        if dtype == "boolean":
            mark = pytest.mark.xfail(
                reason="TODO StringArray.astype() with missing values #GH40566"
            )
            request.applymarker(mark)
        
        # 创建 Series 对象 ser，使用指定的数据和数据类型
        ser = Series(data, dtype=dtype)

        # 注意：直接使用 .astype(dtype) 转换时，对于 dtype="category" 可能会失败，
        # 因为 ser.dtype.categories 可能是对象类型，而 result.dtype.categories 将具有字符串类型
        # 所以先转换为 nullable_string_dtype 类型，然后再转回原始的 ser.dtype 类型
        result = ser.astype(nullable_string_dtype).astype(ser.dtype)

        # 使用 assert_series_equal 断言结果 result 和原始 ser 是否相等
        tm.assert_series_equal(result, ser)


# 定义一个测试类 TestAstypeCategorical，用于测试分类数据类型转换的功能
class TestAstypeCategorical:
    # 测试将分类数据转换为其他类型的方法
    def test_astype_categorical_to_other(self):
        # 创建一个分类变量，包含格式为 "i - i + 499" 的字符串，范围是从0到99500，步长为500
        cat = Categorical([f"{i} - {i + 499}" for i in range(0, 10000, 500)])
        # 创建一个包含100个随机整数的序列，并按升序排序
        ser = Series(np.random.default_rng(2).integers(0, 10000, 100)).sort_values()
        # 将序列根据指定的区间分段，并使用分类变量作为标签
        ser = cut(ser, range(0, 10500, 500), right=False, labels=cat)

        # 期望的结果是与原序列相同
        expected = ser
        # 检查序列转换为 "category" 后是否与期望结果相同
        tm.assert_series_equal(ser.astype("category"), expected)
        # 检查序列转换为 CategoricalDtype 后是否与期望结果相同
        tm.assert_series_equal(ser.astype(CategoricalDtype()), expected)

        # 期望引发值错误异常，信息为 "Cannot cast object|string dtype to float64"
        msg = r"Cannot cast object|string dtype to float64"
        with pytest.raises(ValueError, match=msg):
            ser.astype("float64")

        # 创建一个包含分类数据的序列
        cat = Series(Categorical(["a", "b", "b", "a", "a", "c", "c", "c"]))
        # 创建一个期望的结果序列，数据类型为对象
        exp = Series(["a", "b", "b", "a", "a", "c", "c", "c"], dtype=object)
        # 检查将分类数据转换为字符串后是否与期望结果相同
        tm.assert_series_equal(cat.astype("str"), exp)

        # 创建一个包含字符串类型的分类序列
        s2 = Series(Categorical(["1", "2", "3", "4"]))
        # 创建一个期望的结果序列，数据类型为整数
        exp2 = Series([1, 2, 3, 4]).astype("int")
        # 检查将分类数据转换为整数后是否与期望结果相同
        tm.assert_series_equal(s2.astype("int"), exp2)

        # 定义一个比较函数 cmp，用于比较两个序列的唯一值
        # 对象类型不正确排序，因此只比较是否具有相同的值
        def cmp(a, b):
            tm.assert_almost_equal(np.sort(np.unique(a)), np.sort(np.unique(b)))

        # 创建期望的结果序列，数据类型为对象
        expected = Series(np.array(ser.values), name="value_group")
        # 检查将序列转换为对象类型后是否与期望结果相同
        cmp(ser.astype("object"), expected)
        # 检查将序列转换为 np.object_ 类型后是否与期望结果相同
        cmp(ser.astype(np.object_), expected)

        # 检查将序列转换为数组后是否与序列的值数组相近
        tm.assert_almost_equal(np.array(ser), np.array(ser.values))

        # 检查将序列转换为 "category" 后是否与原序列相同
        tm.assert_series_equal(ser.astype("category"), ser)
        # 检查将序列转换为 CategoricalDtype 后是否与原序列相同
        tm.assert_series_equal(ser.astype(CategoricalDtype()), ser)

        # 将序列转换为对象类型后再转换为分类类型，期望结果是去除未使用的分类并按照分类值排序
        roundtrip_expected = ser.cat.set_categories(
            ser.cat.categories.sort_values()
        ).cat.remove_unused_categories()
        result = ser.astype("object").astype("category")
        # 检查结果与期望的回转序列是否相同
        tm.assert_series_equal(result, roundtrip_expected)
        result = ser.astype("object").astype(CategoricalDtype())
        # 检查结果与期望的回转序列是否相同
        tm.assert_series_equal(result, roundtrip_expected)

    # 测试无效的分类转换方法
    def test_astype_categorical_invalid_conversions(self):
        # 创建一个分类变量，包含格式为 "i - i + 499" 的字符串，范围是从0到99500，步长为500
        cat = Categorical([f"{i} - {i + 499}" for i in range(0, 10000, 500)])
        # 创建一个包含100个随机整数的序列，并按升序排序
        ser = Series(np.random.default_rng(2).integers(0, 10000, 100)).sort_values()
        # 将序列根据指定的区间分段，并使用分类变量作为标签
        ser = cut(ser, range(0, 10500, 500), right=False, labels=cat)

        # 期望引发类型错误异常，信息为 "dtype '<class 'pandas.core.arrays.categorical.Categorical'>' not understood"
        msg = (
            "dtype '<class 'pandas.core.arrays.categorical.Categorical'>' "
            "not understood"
        )
        with pytest.raises(TypeError, match=msg):
            ser.astype(Categorical)
        with pytest.raises(TypeError, match=msg):
            ser.astype("object").astype(Categorical)
    def test_astype_categoricaldtype(self):
        # 创建一个包含字符串的Series对象
        ser = Series(["a", "b", "a"])
        # 将Series转换为指定的有序分类数据类型，并获取结果
        result = ser.astype(CategoricalDtype(["a", "b"], ordered=True))
        # 创建预期结果的Series对象，使用有序分类数据类型
        expected = Series(Categorical(["a", "b", "a"], ordered=True))
        # 检查结果和预期结果是否相等
        tm.assert_series_equal(result, expected)

        # 将Series转换为指定的无序分类数据类型，并获取结果
        result = ser.astype(CategoricalDtype(["a", "b"], ordered=False))
        # 创建预期结果的Series对象，使用无序分类数据类型
        expected = Series(Categorical(["a", "b", "a"], ordered=False))
        # 检查结果和预期结果是否相等
        tm.assert_series_equal(result, expected)

        # 将Series转换为指定的无序分类数据类型（包含额外的类别），并获取结果
        result = ser.astype(CategoricalDtype(["a", "b", "c"], ordered=False))
        # 创建预期结果的Series对象，使用无序分类数据类型，并指定额外的类别
        expected = Series(
            Categorical(["a", "b", "a"], categories=["a", "b", "c"], ordered=False)
        )
        # 检查结果和预期结果是否相等
        tm.assert_series_equal(result, expected)
        # 检查结果的分类类别是否与预期一致
        tm.assert_index_equal(result.cat.categories, Index(["a", "b", "c"]))

    @pytest.mark.parametrize("name", [None, "foo"])
    @pytest.mark.parametrize("dtype_ordered", [True, False])
    @pytest.mark.parametrize("series_ordered", [True, False])
    def test_astype_categorical_to_categorical(
        self, name, dtype_ordered, series_ordered
    ):
        # GH#10696, GH#18593
        # 创建一个包含字符列表的数据
        s_data = list("abcaacbab")
        # 创建一个指定的分类数据类型，包含有序性信息
        s_dtype = CategoricalDtype(list("bac"), ordered=series_ordered)
        # 创建一个Series对象，指定名称和数据类型
        ser = Series(s_data, dtype=s_dtype, name=name)

        # 创建一个未指定类别的分类数据类型
        dtype = CategoricalDtype(ordered=dtype_ordered)
        # 将Series转换为指定的分类数据类型，并获取结果
        result = ser.astype(dtype)
        # 创建期望结果的数据类型，基于输入的分类数据类型
        exp_dtype = CategoricalDtype(s_dtype.categories, dtype_ordered)
        # 创建期望结果的Series对象，基于输入的数据和名称
        expected = Series(s_data, name=name, dtype=exp_dtype)
        # 检查结果和预期结果是否相等
        tm.assert_series_equal(result, expected)

        # 创建一个具有不同类别的分类数据类型
        dtype = CategoricalDtype(list("adc"), dtype_ordered)
        # 将Series转换为指定的分类数据类型，并获取结果
        result = ser.astype(dtype)
        # 创建期望结果的Series对象，基于输入的数据、名称和数据类型
        expected = Series(s_data, name=name, dtype=dtype)
        # 检查结果和预期结果是否相等
        tm.assert_series_equal(result, expected)

        if dtype_ordered is False:
            # 如果未指定有序性，则只测试一次
            expected = ser
            # 将Series转换为"category"类型，并获取结果
            result = ser.astype("category")
            # 检查结果和预期结果是否相等
            tm.assert_series_equal(result, expected)

    def test_astype_bool_missing_to_categorical(self):
        # GH-19182
        # 创建一个包含布尔值和缺失值的Series对象
        ser = Series([True, False, np.nan])
        # 断言Series对象的数据类型为np.object_
        assert ser.dtypes == np.object_

        # 将Series转换为指定的分类数据类型，并获取结果
        result = ser.astype(CategoricalDtype(categories=[True, False]))
        # 创建预期结果的Series对象，使用指定的分类数据类型
        expected = Series(Categorical([True, False, np.nan], categories=[True, False]))
        # 检查结果和预期结果是否相等
        tm.assert_series_equal(result, expected)

    def test_astype_categories_raises(self):
        # deprecated GH#17636, removed in GH#27141
        # 创建一个包含字符串的Series对象
        ser = Series(["a", "b", "a"])
        # 使用pytest断言，预期会抛出TypeError，并匹配特定的错误信息
        with pytest.raises(TypeError, match="got an unexpected"):
            # 将Series转换为"category"类型，同时指定类别和有序性
            ser.astype("category", categories=["a", "b"], ordered=True)

    @pytest.mark.parametrize("items", [["a", "b", "c", "a"], [1, 2, 3, 1]])
    def test_astype_from_categorical(self, items):
        # 创建一个包含指定项目的Series对象
        ser = Series(items)
        # 创建预期结果的Series对象，将数据转换为分类类型
        exp = Series(Categorical(items))
        # 将Series转换为"category"类型，并获取结果
        res = ser.astype("category")
        # 检查结果和预期结果是否相等
        tm.assert_series_equal(res, exp)
    # 定义一个测试方法，用于测试从分类数据类型转换的情况，带有关键字参数
    def test_astype_from_categorical_with_keywords(self):
        # 创建一个包含字符串的列表
        lst = ["a", "b", "c", "a"]
        # 根据列表创建一个 Pandas Series 对象
        ser = Series(lst)
        # 创建一个期望的结果 Series，将输入列表转换为有序的分类数据类型
        exp = Series(Categorical(lst, ordered=True))
        # 使用 astype 方法将 ser 转换为有序的分类数据类型，并将结果存储在 res 中
        res = ser.astype(CategoricalDtype(None, ordered=True))
        # 使用 Pandas 测试模块比较 res 和 exp 是否相等
        tm.assert_series_equal(res, exp)

        # 创建另一个期望的结果 Series，将输入列表转换为指定的分类数据类型
        exp = Series(Categorical(lst, categories=list("abcdef"), ordered=True))
        # 使用 astype 方法将 ser 转换为指定的分类数据类型，并将结果存储在 res 中
        res = ser.astype(CategoricalDtype(list("abcdef"), ordered=True))
        # 使用 Pandas 测试模块比较 res 和 exp 是否相等
        tm.assert_series_equal(res, exp)

    # 定义一个测试方法，用于测试 timedelta64 数据类型转换，并处理 np.nan 的情况
    def test_astype_timedelta64_with_np_nan(self):
        # 创建一个包含 Timedelta 对象和 np.nan 的 Series，数据类型为 timedelta64[ns]
        result = Series([Timedelta(1), np.nan], dtype="timedelta64[ns]")
        # 创建一个期望的结果 Series，将 np.nan 转换为 NaT
        expected = Series([Timedelta(1), NaT], dtype="timedelta64[ns]")
        # 使用 Pandas 测试模块比较 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

    # 根据条件跳过测试，如果未安装 pyarrow 库，则跳过该测试
    @td.skip_if_no("pyarrow")
    def test_astype_int_na_string(self):
        # 创建一个包含整数和 NA 值的 Series，数据类型为 Int64[pyarrow]
        ser = Series([12, NA], dtype="Int64[pyarrow]")
        # 使用 astype 方法将数据类型转换为 string[pyarrow]
        result = ser.astype("string[pyarrow]")
        # 创建一个期望的结果 Series，将整数转换为对应的字符串，保留 NA 值
        expected = Series(["12", NA], dtype="string[pyarrow]")
        # 使用 Pandas 测试模块比较 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)
```