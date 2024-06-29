# `D:\src\scipysrc\pandas\pandas\tests\indexes\timedeltas\test_constructors.py`

```
    def test_array_of_dt64_nat_raises(self):
        # GH#39462
        # 创建一个 NaT (Not a Time) 对象，使用纳秒精度的 np.datetime64 类型
        nat = np.datetime64("NaT", "ns")
        # 创建一个包含 nat 对象的数组，数据类型为 object
        arr = np.array([nat], dtype=object)

        # 预期错误信息
        msg = "Invalid type for timedelta scalar"
        
        # 使用 TimedeltaIndex 初始化时，预期会抛出 TypeError，并匹配错误信息
        with pytest.raises(TypeError, match=msg):
            TimedeltaIndex(arr)

        # 使用 TimedeltaArray._from_sequence 初始化时，预期会抛出 TypeError，并匹配错误信息
        with pytest.raises(TypeError, match=msg):
            TimedeltaArray._from_sequence(arr, dtype="m8[ns]")

        # 使用 to_timedelta 函数时，预期会抛出 TypeError，并匹配错误信息
        with pytest.raises(TypeError, match=msg):

    def test_int64_nocopy(self):
        # GH#23539 检查当传递 int64 数据且设置 copy=False 时不会复制数据
        arr = np.arange(10, dtype=np.int64)
        # 使用 TimedeltaIndex 初始化，设置 copy=False
        tdi = TimedeltaIndex(arr, copy=False)
        # 断言 tdi 的底层数据与 arr 相同
        assert tdi._data._ndarray.base is arr

    def test_infer_from_tdi(self):
        # GH#23539
        # 当传递的数据已经有频率信息时，快速推断出频率
        tdi = timedelta_range("1 second", periods=10**7, freq="1s")

        # 使用 TimedeltaIndex 初始化时，设置 freq="infer"，预期结果的频率与 tdi 的频率相同
        result = TimedeltaIndex(tdi, freq="infer")
        assert result.freq == tdi.freq

        # 检查是否通过检查 "_cache" 属性来验证 inferred_freq 方法未被调用
        assert "inferred_freq" not in getattr(result, "_cache", {})

    def test_infer_from_tdi_mismatch(self):
        # GH#23539
        # 当传递的数据已有频率信息，但与指定的 freq 参数不匹配时，快速引发异常
        tdi = timedelta_range("1 second", periods=100, freq="1s")

        # 预期错误信息
        msg = (
            "Inferred frequency .* from passed values does "
            "not conform to passed frequency"
        )

        # 使用 TimedeltaIndex 初始化时，传递不匹配的 freq 参数，预期会抛出 ValueError，并匹配错误信息
        with pytest.raises(ValueError, match=msg):
            TimedeltaIndex(tdi, freq="D")

        # 使用 TimedeltaIndex 初始化时，传递不匹配的 freq 参数，预期会抛出 ValueError，并匹配错误信息
        with pytest.raises(ValueError, match=msg):
            TimedeltaIndex(tdi._data, freq="D")

    def test_dt64_data_invalid(self):
        # GH#23539
        # 传递具有时区信息的 DatetimeIndex 会引发异常，但没有时区信息或是 ndarray[datetime64] 类型的数据不会
        dti = pd.date_range("2016-01-01", periods=3)

        # 预期错误信息
        msg = "cannot be converted to timedelta64"

        # 使用 TimedeltaIndex 初始化时，传递带有时区信息的 DatetimeIndex，预期会抛出 TypeError，并匹配错误信息
        with pytest.raises(TypeError, match=msg):
            TimedeltaIndex(dti.tz_localize("Europe/Brussels"))

        # 使用 TimedeltaIndex 初始化时，传递 DatetimeIndex，预期会抛出 TypeError，并匹配错误信息
        with pytest.raises(TypeError, match=msg):
            TimedeltaIndex(dti)

        # 使用 TimedeltaIndex 初始化时，传递 ndarray[datetime64] 类型的数据，预期会抛出 TypeError，并匹配错误信息
        with pytest.raises(TypeError, match=msg):
            TimedeltaIndex(np.asarray(dti))
    def test_float64_ns_rounded(self):
        # GH#23539 如果没有指定单位，浮点数会被视为纳秒，小数部分会被截断
        tdi = TimedeltaIndex([2.3, 9.7])
        expected = TimedeltaIndex([2, 9])
        tm.assert_index_equal(tdi, expected)

        # 整数浮点数不会有数据损失
        tdi = TimedeltaIndex([2.0, 9.0])
        expected = TimedeltaIndex([2, 9])
        tm.assert_index_equal(tdi, expected)

        # NaN 被转换为 NaT（Not a Time）
        tdi = TimedeltaIndex([2.0, np.nan])
        expected = TimedeltaIndex([Timedelta(nanoseconds=2), pd.NaT])
        tm.assert_index_equal(tdi, expected)

    def test_float64_unit_conversion(self):
        # GH#23539
        tdi = to_timedelta([1.5, 2.25], unit="D")
        expected = TimedeltaIndex([Timedelta(days=1.5), Timedelta(days=2.25)])
        tm.assert_index_equal(tdi, expected)

    def test_construction_base_constructor(self):
        arr = [Timedelta("1 days"), pd.NaT, Timedelta("3 days")]
        tm.assert_index_equal(pd.Index(arr), TimedeltaIndex(arr))
        tm.assert_index_equal(pd.Index(np.array(arr)), TimedeltaIndex(np.array(arr)))

        arr = [np.nan, pd.NaT, Timedelta("1 days")]
        tm.assert_index_equal(pd.Index(arr), TimedeltaIndex(arr))
        tm.assert_index_equal(pd.Index(np.array(arr)), TimedeltaIndex(np.array(arr)))

    def test_constructor(self):
        expected = TimedeltaIndex(
            [
                "1 days",
                "1 days 00:00:05",
                "2 days",
                "2 days 00:00:02",
                "0 days 00:00:03",
            ]
        )
        result = TimedeltaIndex(
            [
                "1 days",
                "1 days, 00:00:05",
                np.timedelta64(2, "D"),
                timedelta(days=2, seconds=2),
                pd.offsets.Second(3),
            ]
        )
        tm.assert_index_equal(result, expected)

    def test_constructor_iso(self):
        # GH #21877
        expected = timedelta_range("1s", periods=9, freq="s")
        durations = [f"P0DT0H0M{i}S" for i in range(1, 10)]
        result = to_timedelta(durations)
        tm.assert_index_equal(result, expected)

    def test_timedelta_range_fractional_period(self):
        msg = "periods must be an integer"
        with pytest.raises(TypeError, match=msg):
            timedelta_range("1 days", periods=10.5)
    # 测试函数：测试timedelta_range构造函数对异常情况的处理
    def test_constructor_coverage(self):
        # 定义错误消息，期望抛出TypeError异常，检查是否捕获到正确的异常信息
        msg = "periods must be an integer, got foo"
        with pytest.raises(TypeError, match=msg):
            # 调用timedelta_range函数，传入错误参数"foo"作为periods，预期抛出TypeError异常
            timedelta_range(start="1 days", periods="foo", freq="D")

        # 定义错误消息，期望抛出TypeError异常，检查是否捕获到正确的异常信息
        msg = (
            r"TimedeltaIndex\(\.\.\.\) must be called with a collection of some kind, "
            "'1 days' was passed"
        )
        with pytest.raises(TypeError, match=msg):
            # 创建TimedeltaIndex对象，传入错误参数"1 days"，期望抛出TypeError异常
            TimedeltaIndex("1 days")

        # 生成器表达式
        gen = (timedelta(i) for i in range(10))
        # 使用生成器初始化TimedeltaIndex对象
        result = TimedeltaIndex(gen)
        # 期望的TimedeltaIndex对象，使用列表推导式生成
        expected = TimedeltaIndex([timedelta(i) for i in range(10)])
        # 断言两个TimedeltaIndex对象相等
        tm.assert_index_equal(result, expected)

        # NumPy字符串数组
        strings = np.array(["1 days", "2 days", "3 days"])
        # 使用字符串数组初始化TimedeltaIndex对象
        result = TimedeltaIndex(strings)
        # 期望的TimedeltaIndex对象，使用to_timedelta函数生成
        expected = to_timedelta([1, 2, 3], unit="D")
        # 断言两个TimedeltaIndex对象相等
        tm.assert_index_equal(result, expected)

        # 使用整数初始化TimedeltaIndex对象
        from_ints = TimedeltaIndex(expected.asi8)
        # 断言两个TimedeltaIndex对象相等
        tm.assert_index_equal(from_ints, expected)

        # 非标准的频率
        # 定义错误消息，期望抛出ValueError异常，检查是否捕获到正确的异常信息
        msg = (
            "Inferred frequency None from passed values does not conform to "
            "passed frequency D"
        )
        with pytest.raises(ValueError, match=msg):
            # 创建TimedeltaIndex对象，传入不符合要求的参数列表和频率"D"，期望抛出ValueError异常
            TimedeltaIndex(["1 days", "2 days", "4 days"], freq="D")

        # 定义错误消息，期望抛出ValueError异常，检查是否捕获到正确的异常信息
        msg = (
            "Of the four parameters: start, end, periods, and freq, exactly "
            "three must be specified"
        )
        with pytest.raises(ValueError, match=msg):
            # 调用timedelta_range函数，传入参数periods和freq，但未指定其它必需的参数，期望抛出ValueError异常
            timedelta_range(periods=10, freq="D")
    # 测试从分类数据创建 TimedeltaIndex 的功能
    def test_from_categorical(self):
        # 创建一个时间增量范围对象，从1开始，包含5个周期
        tdi = timedelta_range(1, periods=5)

        # 将时间增量范围对象 tdi 转换为 Pandas 的分类数据类型
        cat = pd.Categorical(tdi)

        # 使用分类数据创建 TimedeltaIndex 对象
        result = TimedeltaIndex(cat)
        # 断言两个 TimedeltaIndex 对象是否相等
        tm.assert_index_equal(result, tdi)

        # 将时间增量范围对象 tdi 转换为 Pandas 的分类索引类型
        ci = pd.CategoricalIndex(tdi)
        # 使用分类索引创建 TimedeltaIndex 对象
        result = TimedeltaIndex(ci)
        # 断言两个 TimedeltaIndex 对象是否相等
        tm.assert_index_equal(result, tdi)

    @pytest.mark.parametrize(
        "unit,unit_depr",
        [
            ("W", "w"),
            ("D", "d"),
            ("min", "MIN"),
            ("s", "S"),
            ("h", "H"),
            ("ms", "MS"),
            ("us", "US"),
        ],
    )
    # 测试已废弃的时间单位警告功能
    def test_unit_deprecated(self, unit, unit_depr):
        # 创建警告消息，指出特定单位(unit_depr)在将来的版本中将被移除
        msg = f"'{unit_depr}' is deprecated and will be removed in a future version."

        # 创建期望的 TimedeltaIndex 对象，使用非废弃单位(unit)
        expected = TimedeltaIndex([f"1{unit}", f"2{unit}"])
        # 断言在产生 FutureWarning 警告时，使用废弃单位(unit_depr)创建 TimedeltaIndex 对象时的行为
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = TimedeltaIndex([f"1{unit_depr}", f"2{unit_depr}"])
        # 断言生成的 TimedeltaIndex 对象与期望的对象相等
        tm.assert_index_equal(result, expected)

        # 断言在产生 FutureWarning 警告时，使用废弃单位(unit_depr)创建时间增量对象的行为
        with tm.assert_produces_warning(FutureWarning, match=msg):
            tdi = to_timedelta([1, 2], unit=unit_depr)
        # 断言生成的时间增量对象与期望的对象相等
        tm.assert_index_equal(tdi, expected)
```