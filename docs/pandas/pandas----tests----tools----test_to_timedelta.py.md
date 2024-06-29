# `D:\src\scipysrc\pandas\pandas\tests\tools\test_to_timedelta.py`

```
    @pytest.mark.parametrize(
        "dtype, unit",
        [
            ["int64", "s"],
            ["int64", "m"],
            ["int64", "h"],
            ["timedelta64[s]", "s"],
            ["timedelta64[D]", "D"],
        ],
    )
    def test_to_timedelta_units_dtypes(self, dtype, unit):
        # 定义参数化测试：测试不同的数据类型和时间单位组合
        arr = np.array([1] * 5, dtype=dtype)
        # 调用 to_timedelta 函数，将数组 arr 转换为时间增量，使用给定的单位
        result = to_timedelta(arr, unit=unit)
        # 根据参数化提供的单位创建预期的 TimedeltaIndex 对象
        exp_dtype = "m8[ns]" if dtype == "int64" else "m8[s]"
        expected = TimedeltaIndex([np.timedelta64(1, unit)] * 5, dtype=exp_dtype)
        # 断言结果与预期相等
        tm.assert_index_equal(result, expected)
    def test_to_timedelta_oob_non_nano(self):
        # 创建一个包含一个超出范围值的 timedelta64 数组
        arr = np.array([pd.NaT._value + 1], dtype="timedelta64[m]")

        # 设置错误消息，指明无法将 -9223372036854775807 分钟转换为 timedelta64[s] 而不溢出
        msg = (
            "Cannot convert -9223372036854775807 minutes to "
            r"timedelta64\[s\] without overflow"
        )

        # 测试 to_timedelta 函数是否引发 OutOfBoundsTimedelta 异常，且异常消息匹配指定的 msg
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            to_timedelta(arr)

        # 测试 TimedeltaIndex 构造函数是否引发 OutOfBoundsTimedelta 异常，且异常消息匹配指定的 msg
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            TimedeltaIndex(arr)

        # 测试 TimedeltaArray._from_sequence 方法是否引发 OutOfBoundsTimedelta 异常，且异常消息匹配指定的 msg
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            TimedeltaArray._from_sequence(arr, dtype="m8[s]")

    @pytest.mark.parametrize("box", [lambda x: x, pd.DataFrame])
    @pytest.mark.parametrize("errors", ["raise", "coerce"])
    def test_to_timedelta_dataframe(self, box, errors):
        # GH 11776，测试 DataFrame 输入时的异常情况
        arg = box(np.arange(10).reshape(2, 5))
        
        # 使用 pytest 检查是否引发了 TypeError 异常，异常消息中包含 "1-d array"
        with pytest.raises(TypeError, match="1-d array"):
            to_timedelta(arg, errors=errors)

    def test_to_timedelta_invalid_errors(self):
        # 错误的 errors 参数值测试
        msg = "errors must be one of"
        
        # 使用 pytest 检查是否引发了 ValueError 异常，异常消息中包含 "errors must be one of"
        with pytest.raises(ValueError, match=msg):
            to_timedelta(["foo"], errors="never")

    @pytest.mark.parametrize("arg", [[1, 2], 1])
    def test_to_timedelta_invalid_unit(self, arg):
        # 无效的单位测试
        msg = "invalid unit abbreviation: foo"
        
        # 使用 pytest 检查是否引发了 ValueError 异常，异常消息中包含 "invalid unit abbreviation: foo"
        with pytest.raises(ValueError, match=msg):
            to_timedelta(arg, unit="foo")

    def test_to_timedelta_time(self):
        # 时间类型当前不支持的信息提示
        msg = (
            "Value must be Timedelta, string, integer, float, timedelta or convertible"
        )
        
        # 使用 pytest 检查是否引发了 ValueError 异常，异常消息中包含指定的 msg
        with pytest.raises(ValueError, match=msg):
            to_timedelta(time(second=1))
        
        # 断言将 time(second=1) 转换为 timedelta 时，结果为 pd.NaT
        assert to_timedelta(time(second=1), errors="coerce") is pd.NaT

    def test_to_timedelta_bad_value(self):
        # 错误值转换为 NumPy timedelta 的异常情况
        msg = "Could not convert 'foo' to NumPy timedelta"
        
        # 使用 pytest 检查是否引发了 ValueError 异常，异常消息中包含指定的 msg
        with pytest.raises(ValueError, match=msg):
            to_timedelta(["foo", "bar"])

    def test_to_timedelta_bad_value_coerce(self):
        # 强制转换模式下的错误值处理
        tm.assert_index_equal(
            TimedeltaIndex([pd.NaT, pd.NaT]),
            to_timedelta(["foo", "bar"], errors="coerce"),
        )

        tm.assert_index_equal(
            TimedeltaIndex(["1 day", pd.NaT, "1 min"]),
            to_timedelta(["1 day", "bar", "1 min"], errors="coerce"),
        )

    @pytest.mark.parametrize(
        "val, errors",
        [
            ("1M", True),
            ("1 M", True),
            ("1Y", True),
            ("1 Y", True),
            ("1y", True),
            ("1 y", True),
            ("1m", False),
            ("1 m", False),
            ("1 day", False),
            ("2day", False),
        ],
    )
    # 定义测试方法，用于验证不含歧义的时间增量值
    def test_unambiguous_timedelta_values(self, val, errors):
        # 提示信息，警告不再支持使用字符串 'M', 'Y', 'm' 或 'y' 作为单位
        msg = "Units 'M', 'Y' and 'y' do not represent unambiguous timedelta"
        # 如果设置了错误标志，则验证函数应该引发 ValueError 异常并匹配指定消息
        if errors:
            with pytest.raises(ValueError, match=msg):
                to_timedelta(val)
        else:
            # 否则，验证函数不应该引发异常
            to_timedelta(val)

    # 定义测试方法，用于验证通过 apply 方法将字符串转换为 timedelta 对象
    def test_to_timedelta_via_apply(self):
        # 期望的结果 Series 对象，包含一个 timedelta64 类型的值
        expected = Series([np.timedelta64(1, "s")])
        # 使用 apply 方法将字符串 "00:00:01" 转换为 timedelta 对象
        result = Series(["00:00:01"]).apply(to_timedelta)
        # 验证转换结果与期望结果是否相等
        tm.assert_series_equal(result, expected)

        # 直接调用 to_timedelta 将字符串 "00:00:01" 转换为 timedelta 对象
        result = Series([to_timedelta("00:00:01")])
        # 验证转换结果与期望结果是否相等
        tm.assert_series_equal(result, expected)

    # 定义测试方法，验证在推断时间增量时不会产生警告
    def test_to_timedelta_inference_without_warning(self):
        # 列表 vals 包含字符串 "00:00:01" 和 pd.NaT（Not a Time）值
        vals = ["00:00:01", pd.NaT]
        # 使用 assert_produces_warning 上下文管理器验证不会产生任何警告
        with tm.assert_produces_warning(None):
            result = to_timedelta(vals)

        # 期望的 TimedeltaIndex 对象，包含两个 Timedelta 对象
        expected = TimedeltaIndex([pd.Timedelta(seconds=1), pd.NaT])
        # 验证转换结果与期望结果是否相等
        tm.assert_index_equal(result, expected)

    # 定义测试方法，验证在处理缺失值时的 to_timedelta 函数行为
    def test_to_timedelta_on_missing_values(self):
        # 定义 timedelta_NaT 为 np.timedelta64("NaT")
        timedelta_NaT = np.timedelta64("NaT")

        # 将字符串 Series 转换为 timedelta Series
        actual = to_timedelta(Series(["00:00:01", np.nan]))
        # 期望的结果 Series，包含两个 np.timedelta64 类型的值
        expected = Series(
            [np.timedelta64(1000000000, "ns"), timedelta_NaT],
            dtype=f"{tm.ENDIAN}m8[ns]",
        )
        # 验证转换结果与期望结果是否相等
        tm.assert_series_equal(actual, expected)

        # 将带有 pd.NaT 的字符串 Series 直接转换为 timedelta Series
        ser = Series(["00:00:01", pd.NaT], dtype="m8[ns]")
        actual = to_timedelta(ser)
        # 验证转换结果与期望结果是否相等
        tm.assert_series_equal(actual, expected)

    # 使用 pytest 参数化标记，定义测试方法，验证在处理缺失值时的 to_timedelta 函数行为
    @pytest.mark.parametrize("val", [np.nan, pd.NaT, pd.NA])
    def test_to_timedelta_on_missing_values_scalar(self, val):
        # 调用 to_timedelta 函数将单个缺失值转换为 timedelta 对象
        actual = to_timedelta(val)
        # 断言转换结果的数值表示与 np.timedelta64("NaT") 的 int64 表示相等
        assert actual._value == np.timedelta64("NaT").astype("int64")

    # 使用 pytest 参数化标记，定义测试方法，验证在处理缺失值列表时的 to_timedelta 函数行为
    @pytest.mark.parametrize("val", [np.nan, pd.NaT, pd.NA])
    def test_to_timedelta_on_missing_values_list(self, val):
        # 调用 to_timedelta 函数将列表中的缺失值转换为 timedelta 对象列表
        actual = to_timedelta([val])
        # 断言第一个转换结果的数值表示与 np.timedelta64("NaT") 的 int64 表示相等
        assert actual[0]._value == np.timedelta64("NaT").astype("int64")

    # 使用 pytest 跳过标记（skipif），定义测试方法，验证在处理浮点数时的 to_timedelta 函数行为
    @pytest.mark.skipif(WASM, reason="No fp exception support in WASM")
    @pytest.mark.xfail(not IS64, reason="Floating point error")
    def test_to_timedelta_float(self):
        # 创建一个浮点数数组 arr，范围从 0 到 1，步长为 1e-6，取最后十个元素
        arr = np.arange(0, 1, 1e-6)[-10:]
        # 使用 to_timedelta 函数将浮点数数组转换为 timedelta 数组，单位为秒
        result = to_timedelta(arr, unit="s")
        # 期望的 int64 数组，从 999990000 到 10^9，步长为 1000
        expected_asi8 = np.arange(999990000, 10**9, 1000, dtype="int64")
        # 验证转换结果的 asi8 属性与期望结果的数组是否相等
        tm.assert_numpy_array_equal(result.asi8, expected_asi8)

    # 定义测试方法，验证在处理字符串数组时的 to_timedelta 函数行为
    def test_to_timedelta_coerce_strings_unit(self):
        # 创建一个包含整数和字符串 "error" 的数组 arr
        arr = np.array([1, 2, "error"], dtype=object)
        # 使用 to_timedelta 函数将数组 arr 中的元素转换为 timedelta 数组，单位为纳秒，错误处理为 coerce
        result = to_timedelta(arr, unit="ns", errors="coerce")
        # 期望的 TimedeltaIndex 对象，包含两个 Timedelta 对象和一个 pd.NaT 值
        expected = to_timedelta([1, 2, pd.NaT], unit="ns")
        # 验证转换结果与期望结果是否相等
        tm.assert_index_equal(result, expected)

    # 使用 pytest 参数化标记，定义测试方法，验证在推断时间增量时的 to_timedelta 函数行为
    @pytest.mark.parametrize(
        "expected_val, result_val", [[timedelta(days=2), 2], [None, None]]
    )
    # 测试函数，用于测试处理可空的 int64 数据类型转换为 timedelta 对象
    def test_to_timedelta_nullable_int64_dtype(self, expected_val, result_val):
        # 创建预期结果的 Series，包含 timedelta 对象和 expected_val
        expected = Series([timedelta(days=1), expected_val])
        # 创建待测试的 Series，并转换为 timedelta 对象，单位为天
        result = to_timedelta(Series([1, result_val], dtype="Int64"), unit="days")
        
        # 使用测试工具比较两个 Series 是否相等
        tm.assert_series_equal(result, expected)

    # 参数化测试，测试不同输入下的精度处理情况
    @pytest.mark.parametrize(
        ("input", "expected"),
        [
            ("8:53:08.71800000001", "8:53:08.718"),
            ("8:53:08.718001", "8:53:08.718001"),
            ("8:53:08.7180000001", "8:53:08.7180000001"),
            ("-8:53:08.71800000001", "-8:53:08.718"),
            ("8:53:08.7180000089", "8:53:08.718000008"),
        ],
    )
    # 参数化测试，测试不同函数下的精度处理情况
    @pytest.mark.parametrize("func", [pd.Timedelta, to_timedelta])
    def test_to_timedelta_precision_over_nanos(self, input, expected, func):
        # GH: 36738
        # 将预期字符串转换为 Timedelta 对象
        expected = pd.Timedelta(expected)
        # 对输入执行相应的函数转换
        result = func(input)
        # 断言结果与预期相等
        assert result == expected

    # 测试函数，测试处理零维数据情况下的异常情况
    def test_to_timedelta_zerodim(self, fixed_now_ts):
        # 将时间戳转换为 datetime64 对象
        dt64 = fixed_now_ts.to_datetime64()
        # 创建包含 datetime64 对象的 numpy 数组
        arg = np.array(dt64)

        # 准备异常消息字符串
        msg = (
            "Value must be Timedelta, string, integer, float, timedelta "
            "or convertible, not datetime64"
        )
        # 使用 pytest 断言捕获 ValueError 异常并匹配指定消息
        with pytest.raises(ValueError, match=msg):
            to_timedelta(arg)

        # 将 arg 转换为 m8[ns] 类型的数组
        arg2 = arg.view("m8[ns]")
        # 使用 to_timedelta 函数处理转换后的数组
        result = to_timedelta(arg2)
        # 断言 result 是 pd.Timedelta 类型
        assert isinstance(result, pd.Timedelta)
        # 断言 result 的值与 dt64 的视图值相等
        assert result._value == dt64.view("i8")

    # 测试函数，测试处理数值型数据的情况
    def test_to_timedelta_numeric_ea(self, any_numeric_ea_dtype):
        # GH#48796
        # 创建包含数值和 pd.NA 的 Series
        ser = Series([1, pd.NA], dtype=any_numeric_ea_dtype)
        # 使用 to_timedelta 函数处理 Series
        result = to_timedelta(ser)
        # 创建预期结果的 Series
        expected = Series([pd.Timedelta(1, unit="ns"), pd.NaT])
        # 使用测试工具比较两个 Series 是否相等
        tm.assert_series_equal(result, expected)

    # 测试函数，测试处理分数的情况
    def test_to_timedelta_fraction(self):
        # 使用 to_timedelta 处理分数，单位为小时
        result = to_timedelta(1.0 / 3, unit="h")
        # 创建预期的 Timedelta 对象
        expected = pd.Timedelta("0 days 00:19:59.999999998")
        # 断言结果与预期相等
        assert result == expected
# 定义测试函数，测试从箭头类型转换为时间增量类型
def test_from_numeric_arrow_dtype(any_numeric_ea_dtype):
    # 导入 pytest，如果找不到 pyarrow 模块，则跳过这个测试
    pytest.importorskip("pyarrow")
    # 创建一个包含整数 1 和 2 的 Pandas Series，指定其数据类型为给定数值箭头类型的小写形式加上 "[pyarrow]"
    ser = Series([1, 2], dtype=f"{any_numeric_ea_dtype.lower()}[pyarrow]")
    # 调用 to_timedelta 函数，将 Series 转换为时间增量类型
    result = to_timedelta(ser)
    # 创建一个预期的 Pandas Series，其中包含整数 1 和 2，数据类型为 "timedelta64[ns]"
    expected = Series([1, 2], dtype="timedelta64[ns]")
    # 使用 Pandas 的 assert_series_equal 函数比较结果和预期的 Series 是否相等
    tm.assert_series_equal(result, expected)


# 使用参数化装饰器定义测试函数，测试从时间增量类型转换为箭头类型
@pytest.mark.parametrize("unit", ["ns", "ms"])
def test_from_timedelta_arrow_dtype(unit):
    # GH 54298
    # 导入 pytest，如果找不到 pyarrow 模块，则跳过这个测试
    pytest.importorskip("pyarrow")
    # 创建一个包含 timedelta(1) 的 Pandas Series，指定其数据类型为 "duration[unit][pyarrow]"
    expected = Series([timedelta(1)], dtype=f"duration[{unit}][pyarrow]")
    # 调用 to_timedelta 函数，将 Series 转换为时间增量类型
    result = to_timedelta(expected)
    # 使用 Pandas 的 assert_series_equal 函数比较结果和预期的 Series 是否相等
    tm.assert_series_equal(result, expected)
```