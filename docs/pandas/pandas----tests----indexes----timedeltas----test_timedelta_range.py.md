# `D:\src\scipysrc\pandas\pandas\tests\indexes\timedeltas\test_timedelta_range.py`

```
    # 导入 numpy 库并重命名为 np
    import numpy as np
    # 导入 pytest 库
    import pytest

    # 从 pandas 库中导入以下模块：
    # Timedelta: 表示时间差的数据结构
    # TimedeltaIndex: 用于存储 Timedelta 对象的索引
    # timedelta_range: 用于生成一系列 Timedelta 对象的函数
    # to_timedelta: 将数组或序列转换为 Timedelta 对象
    from pandas import (
        Timedelta,
        TimedeltaIndex,
        timedelta_range,
        to_timedelta,
    )
    # 导入 pandas._testing 模块，并重命名为 tm
    import pandas._testing as tm

    # 从 pandas.tseries.offsets 模块中导入以下类：
    # Day: 表示天数的时间偏移量
    # Second: 表示秒数的时间偏移量
    from pandas.tseries.offsets import (
        Day,
        Second,
    )


    # 定义一个测试类 TestTimedeltas
    class TestTimedeltas:
        
        # 定义测试方法 test_timedelta_range_unit
        def test_timedelta_range_unit(self):
            # GH#49824
            # 创建一个 TimedeltaIndex 对象 tdi，包含 10 个元素，每个元素间隔 100000 天
            tdi = timedelta_range("0 Days", periods=10, freq="100000D", unit="s")
            # 生成预期的 numpy 数组 exp_arr
            exp_arr = (np.arange(10, dtype="i8") * 100_000).view("m8[D]").astype("m8[s]")
            # 使用 pandas._testing 模块的方法检查 tdi 和 exp_arr 是否相等
            tm.assert_numpy_array_equal(tdi.to_numpy(), exp_arr)

        # 定义测试方法 test_timedelta_range
        def test_timedelta_range(self):
            # 测试第一种情况：创建长度为 5 的 TimedeltaIndex 对象，每个元素间隔 1 天
            expected = to_timedelta(np.arange(5), unit="D")
            result = timedelta_range("0 days", periods=5, freq="D")
            tm.assert_index_equal(result, expected)

            # 测试第二种情况：创建长度为 11 的 TimedeltaIndex 对象，每个元素间隔 1 天
            expected = to_timedelta(np.arange(11), unit="D")
            result = timedelta_range("0 days", "10 days", freq="D")
            tm.assert_index_equal(result, expected)

            # 测试第三种情况：创建长度为 5 的 TimedeltaIndex 对象，每个元素间隔 2 天
            expected = to_timedelta(np.arange(5), unit="D") + Second(2) + Day()
            result = timedelta_range("1 days, 00:00:02", "5 days, 00:00:02", freq="D")
            tm.assert_index_equal(result, expected)

            # 测试第四种情况：创建长度为 5 的 TimedeltaIndex 对象，每个元素间隔 2 天
            expected = to_timedelta([1, 3, 5, 7, 9], unit="D") + Second(2)
            result = timedelta_range("1 days, 00:00:02", periods=5, freq="2D")
            tm.assert_index_equal(result, expected)

            # 测试第五种情况：创建长度为 50 的 TimedeltaIndex 对象，每个元素间隔 30 分钟
            expected = to_timedelta(np.arange(50), unit="min") * 30
            result = timedelta_range("0 days", freq="30min", periods=50)
            tm.assert_index_equal(result, expected)

        # 标记使用 pytest 参数化装饰器，参数为 "depr_unit" 和 "unit"
        @pytest.mark.parametrize("depr_unit, unit", [("H", "hour"), ("S", "second")])
        # 定义测试方法 test_timedelta_units_H_S_deprecated，测试不推荐使用的时间单位
        def test_timedelta_units_H_S_deprecated(self, depr_unit, unit):
            # GH#52536
            # 构造弃用消息
            depr_msg = (
                f"'{depr_unit}' is deprecated and will be removed in a future version."
            )
            # 生成预期的 TimedeltaIndex 对象 expected
            expected = to_timedelta(np.arange(5), unit=unit)
            # 使用 pandas._testing 模块的方法检查 to_timedelta 是否会产生 FutureWarning 警告
            with tm.assert_produces_warning(FutureWarning, match=depr_msg):
                result = to_timedelta(np.arange(5), unit=depr_unit)
                tm.assert_index_equal(result, expected)

        # 标记使用 pytest 参数化装饰器，参数为 "unit"，测试不支持的时间单位
        @pytest.mark.parametrize("unit", ["T", "t", "L", "l", "U", "u", "N", "n"])
        # 定义测试方法 test_timedelta_unit_T_L_U_N_raises，测试不支持的时间单位引发异常
        def test_timedelta_unit_T_L_U_N_raises(self, unit):
            # 构造错误消息
            msg = f"invalid unit abbreviation: {unit}"

            # 使用 pytest 的 pytest.raises 方法检查是否引发 ValueError 异常，异常消息匹配 msg
            with pytest.raises(ValueError, match=msg):
                to_timedelta(np.arange(5), unit=unit)

        # 标记使用 pytest 参数化装饰器，参数为 "periods" 和 "freq"
        @pytest.mark.parametrize(
            "periods, freq", [(3, "2D"), (5, "D"), (6, "19h12min"), (7, "16h"), (9, "12h")]
        )
        # 定义测试方法 test_linspace_behavior，测试 timedelta_range 方法的行为
        def test_linspace_behavior(self, periods, freq):
            # GH 20976
            # 创建 result 和 expected 分别为不同参数设置下的 TimedeltaIndex 对象
            result = timedelta_range(start="0 days", end="4 days", periods=periods)
            expected = timedelta_range(start="0 days", end="4 days", freq=freq)
            # 使用 pandas._testing 模块的方法检查 result 和 expected 是否相等
            tm.assert_index_equal(result, expected)
    # 定义测试函数，用于测试 timedelta_range 函数中 'H' 频率的废弃情况
    def test_timedelta_range_H_deprecated(self):
        # 指定废弃警告信息
        msg = "'H' is deprecated and will be removed in a future version."

        # 调用 timedelta_range 函数生成时间间隔序列
        result = timedelta_range(start="0 days", end="4 days", periods=6)
        # 断言确保产生 FutureWarning 警告并匹配给定的消息
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 使用废弃的 'H' 频率生成预期结果的时间间隔序列
            expected = timedelta_range(start="0 days", end="4 days", freq="19H12min")
        # 断言两个时间间隔序列相等
        tm.assert_index_equal(result, expected)

    # 定义测试函数，用于测试 timedelta_range 函数中 'T' 频率抛出异常的情况
    def test_timedelta_range_T_raises(self):
        # 指定无效频率的错误消息
        msg = "Invalid frequency: T"

        # 使用 pytest 检查 timedelta_range 函数对 'T' 频率抛出 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            timedelta_range(start="0 days", end="4 days", freq="19h12T")

    # 定义测试函数，用于测试 timedelta_range 函数中错误参数的情况
    def test_errors(self):
        # 指定参数数量不足时的错误消息
        msg = (
            "Of the four parameters: start, end, periods, and freq, "
            "exactly three must be specified"
        )

        # 使用 pytest 分别测试 timedelta_range 函数的各种参数组合是否会抛出 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            timedelta_range(start="0 days")

        with pytest.raises(ValueError, match=msg):
            timedelta_range(end="5 days")

        with pytest.raises(ValueError, match=msg):
            timedelta_range(periods=2)

        with pytest.raises(ValueError, match=msg):
            timedelta_range()

        # 指定参数数量过多时的错误消息，测试是否会抛出 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            timedelta_range(start="0 days", end="5 days", periods=10, freq="h")

    # 使用 pytest.mark.parametrize 标记的测试函数，用于测试 timedelta_range 函数中各种频率的情况
    @pytest.mark.parametrize(
        "start, end, freq, expected_periods",
        [
            ("1D", "10D", "2D", (10 - 1) // 2 + 1),
            ("2D", "30D", "3D", (30 - 2) // 3 + 1),
            ("2s", "50s", "5s", (50 - 2) // 5 + 1),
            # GH 33498 之前的测试：
            ("4D", "16D", "3D", (16 - 4) // 3 + 1),
            ("8D", "16D", "40s", (16 * 3600 * 24 - 8 * 3600 * 24) // 40 + 1),
        ],
    )
    # 定义测试函数，用于测试 timedelta_range 函数中根据结束时间自动推断频率的情况
    def test_timedelta_range_freq_divide_end(self, start, end, freq, expected_periods):
        # GH 33498 中仅在 `(end % freq) == 0` 的情况下才会失败
        # 调用 timedelta_range 函数生成时间间隔序列
        res = timedelta_range(start=start, end=end, freq=freq)
        # 断言首个时间间隔与给定的起始时间相等
        assert Timedelta(start) == res[0]
        # 断言最后一个时间间隔不超过给定的结束时间
        assert Timedelta(end) >= res[-1]
        # 断言生成的时间间隔序列长度符合预期
        assert len(res) == expected_periods

    # 定义测试函数，用于测试 timedelta_range 函数中自动推断频率为 None 的情况
    def test_timedelta_range_infer_freq(self):
        # 检查在一些情况下 timedelta_range 函数生成的时间间隔序列的频率为 None
        result = timedelta_range("0s", "1s", periods=31)
        assert result.freq is None

    # 使用 pytest.mark.parametrize 标记的测试函数，用于测试 timedelta_range 函数中废弃频率的情况
    @pytest.mark.parametrize(
        "freq_depr, start, end, expected_values, expected_freq",
        [
            (
                "3.5S",
                "05:03:01",
                "05:03:10",
                ["0 days 05:03:01", "0 days 05:03:04.500000", "0 days 05:03:08"],
                "3500ms",
            ),
        ],
    )
    # 定义测试函数，用于测试 timedelta_range 函数中废弃频率的情况
    def test_timedelta_range_deprecated_freq(
        self, freq_depr, start, end, expected_values, expected_freq
    ):
        # GH#52536: 标识 GitHub 上的 issue 编号，提醒此处的代码修复了特定问题
        msg = (
            f"'{freq_depr[-1]}' is deprecated and will be removed in a future version."
        )
        
        # 使用 pytest 提供的 assert_produces_warning 上下文管理器，验证是否产生 FutureWarning 警告，并检查警告消息是否匹配预期
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 调用 timedelta_range 函数，传入参数进行测试
            result = timedelta_range(start=start, end=end, freq=freq_depr)
        
        # 创建预期的 TimedeltaIndex 对象，用于与返回的 result 进行比较
        expected = TimedeltaIndex(
            expected_values, dtype="timedelta64[ns]", freq=expected_freq
        )
        
        # 使用 assert_index_equal 函数验证 result 与 expected 是否相等
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "freq_depr, start, end",
        [
            (
                "3.5l",
                "05:03:01",
                "05:03:10",
            ),
            (
                "2.5T",
                "5 hours",
                "5 hours 8 minutes",
            ),
        ],
    )
    # 定义测试函数 test_timedelta_range_removed_freq，使用 pytest 的 parametrize 装饰器传入多组参数进行参数化测试
    def test_timedelta_range_removed_freq(self, freq_depr, start, end):
        # 构造错误消息，用于验证是否会引发 ValueError 异常
        msg = f"Invalid frequency: {freq_depr}"
        
        # 使用 pytest.raises 检测是否引发 ValueError 异常，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            # 调用 timedelta_range 函数，传入参数进行测试
            timedelta_range(start=start, end=end, freq=freq_depr)
```