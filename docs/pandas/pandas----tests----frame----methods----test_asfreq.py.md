# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_asfreq.py`

```
    # 导入 datetime 模块中的 datetime 类
    from datetime import datetime

    # 导入 numpy 库，并使用 np 作为别名
    import numpy as np

    # 导入 pytest 库
    import pytest

    # 从 pandas._libs.tslibs.offsets 模块导入 MonthEnd 类
    from pandas._libs.tslibs.offsets import MonthEnd

    # 从 pandas 库中导入多个对象和函数
    from pandas import (
        DataFrame,
        DatetimeIndex,
        PeriodIndex,
        Series,
        date_range,
        period_range,
        to_datetime,
    )

    # 导入 pandas._testing 库并使用 tm 作为别名
    import pandas._testing as tm

    # 从 pandas.tseries 模块导入 offsets 对象
    from pandas.tseries import offsets


    class TestAsFreq:
        # 测试函数，测试 asfreq 方法的不同用法
        def test_asfreq2(self, frame_or_series):
            # 创建时间序列 ts，包含三个值和指定索引
            ts = frame_or_series(
                [0.0, 1.0, 2.0],
                index=DatetimeIndex(
                    [
                        datetime(2009, 10, 30),
                        datetime(2009, 11, 30),
                        datetime(2009, 12, 31),
                    ],
                    dtype="M8[ns]",  # 索引的数据类型为 datetime64[ns]
                    freq="BME",      # 索引的频率为 Business Month End
                ),
            )

            # 使用 "B" 频率重新采样为每工作日频率
            daily_ts = ts.asfreq("B")

            # 使用 "BME" 频率重新采样为每月的 Business Month End 频率
            monthly_ts = daily_ts.asfreq("BME")

            # 断言 monthly_ts 和 ts 相等
            tm.assert_equal(monthly_ts, ts)

            # 使用 "pad" 方法将缺失值填充后，重新采样为每工作日频率
            daily_ts = ts.asfreq("B", method="pad")

            # 使用 "BME" 频率重新采样为每月的 Business Month End 频率
            monthly_ts = daily_ts.asfreq("BME")

            # 断言 monthly_ts 和 ts 相等
            tm.assert_equal(monthly_ts, ts)

            # 使用 offsets.BDay() 对象重新采样为每工作日频率
            daily_ts = ts.asfreq(offsets.BDay())

            # 使用 offsets.BMonthEnd() 对象重新采样为每月的工作日最后一天频率
            monthly_ts = daily_ts.asfreq(offsets.BMonthEnd())

            # 断言 monthly_ts 和 ts 相等
            tm.assert_equal(monthly_ts, ts)

            # 对空时间序列进行 asfreq 操作，预期返回空序列
            result = ts[:0].asfreq("ME")

            # 断言结果长度为 0
            assert len(result) == 0

            # 断言 result 不是原始 ts 对象的引用
            assert result is not ts

            # 如果 frame_or_series 是 Series 类型
            if frame_or_series is Series:
                # 使用 fill_value=-1 参数将缺失值填充为 -1，重新采样为每日频率
                daily_ts = ts.asfreq("D", fill_value=-1)

                # 计算每个值的频数并按索引排序
                result = daily_ts.value_counts().sort_index()

                # 预期的结果序列
                expected = Series(
                    [60, 1, 1, 1], index=[-1.0, 2.0, 1.0, 0.0], name="count"
                ).sort_index()

                # 断言 result 和 expected 序列相等
                tm.assert_series_equal(result, expected)

        # 测试函数，测试对空 DatetimeIndex 的 asfreq 操作
        def test_asfreq_datetimeindex_empty(self, frame_or_series):
            # 创建单个日期时间索引对象
            index = DatetimeIndex(["2016-09-29 11:00"])

            # 使用空的 DatetimeIndex 调用 frame_or_series 函数，期望返回空序列
            expected = frame_or_series(index=index, dtype=object).asfreq("h")

            # 使用包含单个值的 DatetimeIndex 调用 frame_or_series 函数，并重新采样为每小时频率
            result = frame_or_series([3], index=index.copy()).asfreq("h")

            # 断言 expected 和 result 的索引相等
            tm.assert_index_equal(expected.index, result.index)

        # 参数化测试函数，测试带有时区的 asfreq 方法
        @pytest.mark.parametrize("tz", ["US/Eastern", "dateutil/US/Eastern"])
        def test_tz_aware_asfreq_smoke(self, tz, frame_or_series):
            # 创建时区为 tz 的日期范围对象
            dr = date_range("2011-12-01", "2012-07-20", freq="D", tz=tz)

            # 创建带有随机标准正态分布值的 frame_or_series 对象，并设置索引为 dr
            obj = frame_or_series(
                np.random.default_rng(2).standard_normal(len(dr)), index=dr
            )

            # 调用 asfreq 方法，重新采样为每分钟频率
            obj.asfreq("min")

        # 测试函数，测试 normalize 参数为 True 时的 asfreq 方法
        def test_asfreq_normalize(self, frame_or_series):
            # 创建从 "1/1/2000 09:30" 开始的时间范围对象，共 20 个周期
            rng = date_range("1/1/2000 09:30", periods=20)

            # 创建从 "1/1/2000" 开始的时间范围对象，共 20 个周期
            norm = date_range("1/1/2000", periods=20)

            # 创建随机标准正态分布值的 DataFrame 对象，设置索引为 rng
            vals = np.random.default_rng(2).standard_normal((20, 3))
            obj = DataFrame(vals, index=rng)

            # 创建预期的 DataFrame 对象，设置索引为 norm
            expected = DataFrame(vals, index=norm)

            # 如果 frame_or_series 是 Series 类型，仅使用第一列数据
            if frame_or_series is Series:
                obj = obj[0]
                expected = expected[0]

            # 使用 normalize=True 参数调用 asfreq 方法，重新采样为每日频率
            result = obj.asfreq("D", normalize=True)

            # 断言 result 和 expected DataFrame 对象相等
            tm.assert_equal(result, expected)
    # 测试保持索引名称不变的情况
    def test_asfreq_keep_index_name(self, frame_or_series):
        # 定义索引名称为 "bar"
        index_name = "bar"
        # 创建日期范围为 2013-01-01 至 20 天后，索引名称为 index_name
        index = date_range("20130101", periods=20, name=index_name)
        # 创建包含 0 到 19 的 DataFrame，列名为 ["foo"]，索引为上述日期范围的 obj
        obj = DataFrame(list(range(20)), columns=["foo"], index=index)
        # 根据 frame_or_series 参数获取适当的对象，并赋值给 obj
        obj = tm.get_obj(obj, frame_or_series)

        # 断言索引名称是否保持不变
        assert index_name == obj.index.name
        # 断言经过 asfreq("10D") 处理后的索引名称是否为 index_name
        assert index_name == obj.asfreq("10D").index.name

    # 测试时间序列的 asfreq 方法
    def test_asfreq_ts(self, frame_or_series):
        # 创建频率为年的日期范围，从 2001-01-01 到 2010-12-31
        index = period_range(freq="Y", start="1/1/2001", end="12/31/2010")
        # 创建一个包含随机正态分布数据的 DataFrame，行数为 index 的长度，列数为 3
        obj = DataFrame(
            np.random.default_rng(2).standard_normal((len(index), 3)), index=index
        )
        # 根据 frame_or_series 参数获取适当的对象，并赋值给 obj
        obj = tm.get_obj(obj, frame_or_series)

        # 测试以 "end" 方式处理的 asfreq("D") 方法
        result = obj.asfreq("D", how="end")
        # 期望的索引通过 asfreq("D", how="end") 处理后的结果
        exp_index = index.asfreq("D", how="end")
        # 断言结果长度是否与 obj 的长度相同
        assert len(result) == len(obj)
        # 断言结果索引是否与期望索引相等
        tm.assert_index_equal(result.index, exp_index)

        # 测试以 "start" 方式处理的 asfreq("D") 方法
        result = obj.asfreq("D", how="start")
        # 期望的索引通过 asfreq("D", how="start") 处理后的结果
        exp_index = index.asfreq("D", how="start")
        # 断言结果长度是否与 obj 的长度相同
        assert len(result) == len(obj)
        # 断言结果索引是否与期望索引相等
        tm.assert_index_equal(result.index, exp_index)

    # 测试 asfreq 和 resample 方法设置正确频率
    def test_asfreq_resample_set_correct_freq(self, frame_or_series):
        # GH#5613
        # 测试 .asfreq() 和 .resample() 是否正确设置了 .freq 的值
        dti = to_datetime(["2012-01-01", "2012-01-02", "2012-01-03"])
        # 创建包含 {"col": [1, 2, 3]} 的 DataFrame，索引为 dti
        obj = DataFrame({"col": [1, 2, 3]}, index=dti)
        # 根据 frame_or_series 参数获取适当的对象，并赋值给 obj
        obj = tm.get_obj(obj, frame_or_series)

        # 在调用 .asfreq() 和 .resample() 之前测试设置
        # 断言 obj 的索引频率是否为 None
        assert obj.index.freq is None
        # 断言 obj 推断的频率是否为 "D"
        assert obj.index.inferred_freq == "D"

        # 测试 .asfreq() 方法是否正确设置了 .freq
        assert obj.asfreq("D").index.freq == "D"

        # 测试 .resample() 方法是否正确设置了 .freq
        assert obj.resample("D").asfreq().index.freq == "D"

    # 测试在长度为 0 的 DataFrame 上调用 asfreq 方法不会出错
    def test_asfreq_empty(self, datetime_frame):
        # 创建长度为 0 的 datetime_frame 的副本
        zero_length = datetime_frame.reindex([])
        # 对长度为 0 的 DataFrame 调用 asfreq("BME") 方法
        result = zero_length.asfreq("BME")
        # 断言结果不是 zero_length 的引用
        assert result is not zero_length

    # 测试 asfreq 方法在 DateTimeIndex 上的使用
    def test_asfreq(self, datetime_frame):
        # 使用 offsets.BMonthEnd() 对 datetime_frame 进行 asfreq 处理
        offset_monthly = datetime_frame.asfreq(offsets.BMonthEnd())
        # 使用 "BME" 对 datetime_frame 进行 asfreq 处理
        rule_monthly = datetime_frame.asfreq("BME")

        # 断言 offset_monthly 和 rule_monthly 是否相等
        tm.assert_frame_equal(offset_monthly, rule_monthly)

        # 使用 "pad" 方法处理 asfreq("B") 后的 rule_monthly
        rule_monthly.asfreq("B", method="pad")
        # TODO: 实际检查这是否有效。

        # 不要忘记！
        rule_monthly.asfreq("B", method="pad")

    # 测试在 DateTimeIndex 上进行 asfreq 方法
    def test_asfreq_datetimeindex(self):
        # 创建包含 {"A": [1, 2, 3]} 的 DataFrame，索引为指定的日期时间对象
        df = DataFrame(
            {"A": [1, 2, 3]},
            index=[datetime(2011, 11, 1), datetime(2011, 11, 2), datetime(2011, 11, 3)],
        )
        # 对 DataFrame 使用 asfreq("B") 方法
        df = df.asfreq("B")
        # 断言 df 的索引是否为 DatetimeIndex 类型
        assert isinstance(df.index, DatetimeIndex)

        # 对 df["A"] 应用 asfreq("B") 方法
        ts = df["A"].asfreq("B")
        # 断言 ts 的索引是否为 DatetimeIndex 类型
        assert isinstance(ts.index, DatetimeIndex)
    # 定义一个测试函数，用于测试在上采样过程中填充值的情况，与问题3715相关

    # 设置时间序列的日期范围，频率为每2秒一次
    rng = date_range("1/1/2016", periods=10, freq="2s")
    # 创建一个序列对象，其值为从0开始的连续整数，索引为上面定义的日期范围，数据类型为浮点数
    ts = Series(np.arange(len(rng)), index=rng, dtype="float")
    # 创建一个数据框，包含名为"one"的列，列的数据为上述创建的时间序列对象
    df = DataFrame({"one": ts})

    # 在数据框中插入一个已存在的缺失值
    df.loc["2016-01-01 00:00:08", "one"] = None

    # 对数据框进行频率转换，转换为每秒一次，使用指定的填充值填充缺失值
    actual_df = df.asfreq(freq="1s", fill_value=9.0)
    # 对数据框进行频率转换，转换为每秒一次，然后对结果进行缺失值填充，填充值为9.0
    expected_df = df.asfreq(freq="1s").fillna(9.0)
    # 在期望的数据框中重新设置之前插入的缺失值
    expected_df.loc["2016-01-01 00:00:08", "one"] = None
    # 比较预期的数据框和实际的数据框，确保它们相等
    tm.assert_frame_equal(expected_df, actual_df)

    # 对时间序列进行频率转换，转换为每秒一次，然后对结果进行缺失值填充，填充值为9.0
    expected_series = ts.asfreq(freq="1s").fillna(9.0)
    # 对时间序列进行频率转换，转换为每秒一次，使用指定的填充值填充缺失值
    actual_series = ts.asfreq(freq="1s", fill_value=9.0)
    # 比较预期的时间序列和实际的时间序列，确保它们相等
    tm.assert_series_equal(expected_series, actual_series)

def test_asfreq_with_date_object_index(self, frame_or_series):
    # 创建一个日期范围，从"1/1/2000"开始，包含20个日期
    rng = date_range("1/1/2000", periods=20)
    # 使用随机数生成器创建一个具有标准正态分布随机数的时间序列或数据框，索引为上述日期范围
    ts = frame_or_series(np.random.default_rng(2).standard_normal(20), index=rng)

    # 创建时间序列的副本，并将其索引转换为日期对象
    ts2 = ts.copy()
    ts2.index = [x.date() for x in ts2.index]

    # 对副本的时间序列进行频率转换，转换为每4小时一次，使用前向填充方法填充缺失值
    result = ts2.asfreq("4h", method="ffill")
    # 对原始时间序列进行频率转换，转换为每4小时一次，使用前向填充方法填充缺失值
    expected = ts.asfreq("4h", method="ffill")
    # 比较预期的结果和实际的结果，确保它们相等
    tm.assert_equal(result, expected)

def test_asfreq_with_unsorted_index(self, frame_or_series):
    # GH#39805
    # 测试当日期时间索引未排序时，行不会被丢弃

    # 创建一个日期时间索引，包含四个日期，但是未排序
    index = to_datetime(["2021-01-04", "2021-01-02", "2021-01-03", "2021-01-01"])
    # 使用给定的索引创建一个时间序列或数据框，其值为0到3
    result = frame_or_series(range(4), index=index)

    # 对期望的结果进行重新索引，按日期时间排序
    expected = result.reindex(sorted(index))
    # 将期望的结果的索引设置为与推断的频率相匹配
    expected.index = expected.index._with_freq("infer")

    # 对结果进行频率转换，转换为每天一次
    result = result.asfreq("D")
    # 比较预期的结果和实际的结果，确保它们相等
    tm.assert_equal(result, expected)

def test_asfreq_after_normalize(self, unit):
    # https://github.com/pandas-dev/pandas/issues/50727
    # 创建一个日期时间索引，包含两个日期，然后将其单位化并标准化为每天一次
    result = DatetimeIndex(
        date_range("2000", periods=2).as_unit(unit).normalize(), freq="D"
    )
    # 创建一个预期的日期时间索引，包含两个日期，单位化为指定单位
    expected = DatetimeIndex(["2000-01-01", "2000-01-02"], freq="D").as_unit(unit)
    # 比较预期的结果和实际的结果，确保它们相等
    tm.assert_index_equal(result, expected)

@pytest.mark.parametrize(
    "freq, freq_half",
    [
        ("2ME", "ME"),
        (MonthEnd(2), MonthEnd(1)),
    ],
)
def test_asfreq_2ME(self, freq, freq_half):
    # 创建一个日期范围，从"1/1/2000"开始，包含6个日期，频率为指定的一半月
    index = date_range("1/1/2000", periods=6, freq=freq_half)
    # 创建一个数据框，包含名为"s"的列，列的数据为从0到5，索引为上述日期范围
    df = DataFrame({"s": Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], index=index)})
    # 对数据框进行频率转换，转换为指定频率
    expected = df.asfreq(freq=freq)

    # 创建一个日期范围，从"1/1/2000"开始，包含3个日期，频率为指定的频率
    index = date_range("1/1/2000", periods=3, freq=freq)
    # 创建一个数据框，包含名为"s"的列，列的数据为0、2和4，索引为上述日期范围
    result = DataFrame({"s": Series([0.0, 2.0, 4.0], index=index)})
    # 比较预期的结果和实际的结果，确保它们相等
    tm.assert_frame_equal(result, expected)
    @pytest.mark.parametrize(
        "freq, freq_depr",
        [  # 定义参数化测试的参数，freq和freq_depr分别是原始频率和被弃用的频率
            ("2ME", "2M"),  # 测试用例1：原始频率为"2ME"，被弃用频率为"2M"
            ("2ME", "2m"),  # 测试用例2：原始频率为"2ME"，被弃用频率为"2m"
            ("2QE", "2Q"),   # 测试用例3：原始频率为"2QE"，被弃用频率为"2Q"
            ("2QE-SEP", "2Q-SEP"),  # 测试用例4：原始频率为"2QE-SEP"，被弃用频率为"2Q-SEP"
            ("1BQE", "1BQ"),  # 测试用例5：原始频率为"1BQE"，被弃用频率为"1BQ"
            ("2BQE-SEP", "2BQ-SEP"),  # 测试用例6：原始频率为"2BQE-SEP"，被弃用频率为"2BQ-SEP"
            ("2BQE-SEP", "2bq-sep"),  # 测试用例7：原始频率为"2BQE-SEP"，被弃用频率为"2bq-sep"
            ("1YE", "1y"),   # 测试用例8：原始频率为"1YE"，被弃用频率为"1y"
            ("2YE-MAR", "2Y-MAR"),  # 测试用例9：原始频率为"2YE-MAR"，被弃用频率为"2Y-MAR"
        ],
    )
    def test_asfreq_frequency_M_Q_Y_raises(self, freq, freq_depr):
        # 准备错误消息，指出被弃用频率的无效性
        msg = f"Invalid frequency: {freq_depr}"

        # 创建时间索引，使用给定的原始频率，共4个时间点
        index = date_range("1/1/2000", periods=4, freq=f"{freq[1:]}")

        # 创建DataFrame对象，包含一个Series列，索引为上面创建的时间索引
        df = DataFrame({"s": Series([0.0, 1.0, 2.0, 3.0], index=index)})

        # 使用pytest.raises检查是否抛出值错误异常，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            df.asfreq(freq=freq_depr)

    @pytest.mark.parametrize(
        "freq, error_msg",
        [  # 定义参数化测试的参数，freq和error_msg分别是不支持的频率和期望的错误消息
            (
                "2MS",  # 测试用例1：不支持的频率为"2MS"
                "Invalid frequency: 2MS",  # 预期的错误消息
            ),
            (
                offsets.MonthBegin(),  # 测试用例2：不支持的频率为MonthBegin对象
                r"\<MonthBegin\> is not supported as period frequency",  # 预期的错误消息
            ),
            (
                offsets.DateOffset(months=2),  # 测试用例3：不支持的频率为DateOffset对象，月份为2
                r"\<DateOffset: months=2\> is not supported as period frequency",  # 预期的错误消息
            ),
        ],
    )
    def test_asfreq_unsupported_freq(self, freq, error_msg):
        # 准备时间段索引，频率为"M"
        index = PeriodIndex(["2020-01-01", "2021-01-01"], freq="M")

        # 创建DataFrame对象，包含一个列名为"a"的Series列，索引为上面创建的时间段索引
        df = DataFrame({"a": Series([0, 1], index=index)})

        # 使用pytest.raises检查是否抛出值错误异常，并匹配预期的错误消息
        with pytest.raises(ValueError, match=error_msg):
            df.asfreq(freq=freq)

    @pytest.mark.parametrize(
        "freq, freq_depr",
        [  # 定义参数化测试的参数，freq和freq_depr分别是原始频率和被弃用的频率
            ("2YE", "2A"),  # 测试用例1：原始频率为"2YE"，被弃用频率为"2A"
            ("2BYE-MAR", "2BA-MAR"),  # 测试用例2：原始频率为"2BYE-MAR"，被弃用频率为"2BA-MAR"
        ],
    )
    def test_asfreq_frequency_A_BA_raises(self, freq, freq_depr):
        # 准备错误消息，指出被弃用频率的无效性
        msg = f"Invalid frequency: {freq_depr}"

        # 创建时间索引，使用给定的原始频率
        index = date_range("1/1/2000", periods=4, freq=freq)

        # 创建DataFrame对象，包含一个Series列，索引为上面创建的时间索引
        df = DataFrame({"s": Series([0.0, 1.0, 2.0, 3.0], index=index)})

        # 使用pytest.raises检查是否抛出值错误异常，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            df.asfreq(freq=freq_depr)
```