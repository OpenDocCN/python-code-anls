# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\test_indexing.py`

```
    # 导入必要的日期和时间类
    from datetime import (
        date,
        datetime,
        time,
        timedelta,
    )

    # 导入 numpy 库，并使用别名 np
    import numpy as np

    # 导入 pytest 库
    import pytest

    # 导入 pandas 库中的 index 模块作为 libindex
    from pandas._libs import index as libindex

    # 从 pandas.compat.numpy 导入 np_long
    from pandas.compat.numpy import np_long

    # 导入 pandas 库，并使用别名 pd
    import pandas as pd

    # 从 pandas 中导入以下类和函数
    from pandas import (
        DatetimeIndex,
        Index,
        Timestamp,
        bdate_range,
        date_range,
        notna,
    )

    # 导入 pandas._testing 库，并使用别名 tm
    import pandas._testing as tm

    # 从 pandas.tseries.frequencies 中导入 to_offset 函数
    from pandas.tseries.frequencies import to_offset

    # 定义起始日期和结束日期常量
    START, END = datetime(2009, 1, 1), datetime(2010, 1, 1)

    # 测试类 TestGetItem
    class TestGetItem:

        # 测试方法 test_getitem_slice_keeps_name
        def test_getitem_slice_keeps_name(self):
            # GH4226
            # 创建带有时区的 Timestamp 对象 st 和 et
            st = Timestamp("2013-07-01 00:00:00", tz="America/Los_Angeles")
            et = Timestamp("2013-07-02 00:00:00", tz="America/Los_Angeles")
            # 创建日期范围 dr，频率为每小时，并指定名称为 "timebucket"
            dr = date_range(st, et, freq="h", name="timebucket")
            # 断言切片后的日期范围的名称与原始的名称相同
            assert dr[1:].name == dr.name

        # 参数化测试方法 test_getitem
        @pytest.mark.parametrize("tz", [None, "Asia/Tokyo"])
        def test_getitem(self, tz):
            # 创建日期范围 idx，从 "2011-01-01" 到 "2011-01-31"，频率为每日，时区为 tz，名称为 "idx"
            idx = date_range("2011-01-01", "2011-01-31", freq="D", tz=tz, name="idx")

            # 测试获取索引为 0 的结果
            result = idx[0]
            assert result == Timestamp("2011-01-01", tz=idx.tz)

            # 测试切片获取索引为 0 到 4 的结果
            result = idx[0:5]
            expected = date_range(
                "2011-01-01", "2011-01-05", freq="D", tz=idx.tz, name="idx"
            )
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq

            # 测试以步长为 2 切片获取索引为 0 到 9 的结果
            result = idx[0:10:2]
            expected = date_range(
                "2011-01-01", "2011-01-09", freq="2D", tz=idx.tz, name="idx"
            )
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq

            # 测试从倒数第 20 到倒数第 5 个元素，步长为 3 的结果
            result = idx[-20:-5:3]
            expected = date_range(
                "2011-01-12", "2011-01-24", freq="3D", tz=idx.tz, name="idx"
            )
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq

            # 测试反向切片获取索引为 4 到 0 的结果
            result = idx[4::-1]
            expected = DatetimeIndex(
                ["2011-01-05", "2011-01-04", "2011-01-03", "2011-01-02", "2011-01-01"],
                dtype=idx.dtype,
                freq="-1D",
                name="idx",
            )
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq

        # 参数化测试方法 test_dti_business_getitem
        @pytest.mark.parametrize("freq", ["B", "C"])
        def test_dti_business_getitem(self, freq):
            # 创建工作日日期范围 rng，从 START 到 END，指定频率为 freq
            rng = bdate_range(START, END, freq=freq)

            # 测试获取前 5 个元素的结果
            smaller = rng[:5]
            exp = DatetimeIndex(rng.view(np.ndarray)[:5], freq=freq)
            tm.assert_index_equal(smaller, exp)
            assert smaller.freq == exp.freq
            assert smaller.freq == rng.freq

            # 测试以步长为 5 获取的日期范围
            sliced = rng[::5]
            assert sliced.freq == to_offset(freq) * 5

            # 测试根据指定索引顺序获取的结果
            fancy_indexed = rng[[4, 3, 2, 1, 0]]
            assert len(fancy_indexed) == 5
            assert isinstance(fancy_indexed, DatetimeIndex)
            assert fancy_indexed.freq is None

            # 测试在 32 位和 64 位平台上的索引取值
            # 检查第 4 个元素
            assert rng[4] == rng[np_long(4)]
    # 定义一个测试方法，用于测试通过业务日时间范围获取索引值时的Matplotlib兼容性处理
    def test_dti_business_getitem_matplotlib_hackaround(self, freq):
        # 生成一个业务日时间范围对象rng，包括起始时间START到结束时间END，按照给定频率freq生成
        rng = bdate_range(START, END, freq=freq)
        # 使用pytest的断言检查，预期会抛出值错误(ValueError)，并且错误信息中会包含"Multi-dimensional indexing"
        with pytest.raises(ValueError, match="Multi-dimensional indexing"):
            # 在这里进行多维索引的操作，这种操作已经被弃用（GH#30588）
            rng[:, None]

    # 定义另一个测试方法，用于测试通过整数列表获取索引值时的行为
    def test_getitem_int_list(self):
        # 创建一个日期范围对象dti，从"1/1/2005"到"12/1/2005"，频率为"ME"（每月结束）
        dti = date_range(start="1/1/2005", end="12/1/2005", freq="ME")
        # 通过整数列表[1, 3, 5]获取日期范围对象dti的子集dti2
        dti2 = dti[[1, 3, 5]]

        # 获取dti2中的具体值，并赋给v1、v2、v3
        v1 = dti2[0]  # 第一个元素
        v2 = dti2[1]  # 第二个元素
        v3 = dti2[2]  # 第三个元素

        # 使用断言检查具体值是否与预期的日期时间戳相匹配
        assert v1 == Timestamp("2/28/2005")
        assert v2 == Timestamp("4/30/2005")
        assert v3 == Timestamp("6/30/2005")

        # 使用断言检查在使用非切片的情况下，获取的日期范围对象dti2的频率是否为None
        assert dti2.freq is None
class TestWhere:
    # 测试函数：测试 where 方法在不保留频率的情况下的行为
    def test_where_doesnt_retain_freq(self):
        # 创建一个日期范围，从 "20130101" 开始，包含三个日期，频率为每日 ("D")，命名为 "idx"
        dti = date_range("20130101", periods=3, freq="D", name="idx")
        # 条件数组，用于指示哪些日期保留
        cond = [True, True, False]
        # 期望的结果，创建一个 DatetimeIndex，包含指定的日期，并且不保留频率，命名为 "idx"
        expected = DatetimeIndex([dti[0], dti[1], dti[0]], freq=None, name="idx")

        # 使用 where 方法应用条件并返回结果
        result = dti.where(cond, dti[::-1])
        # 断言结果与期望相等
        tm.assert_index_equal(result, expected)

    # 测试函数：测试 where 方法中其他情况的行为
    def test_where_other(self):
        # 创建一个日期范围，从 "20130101" 开始，包含三个日期，时区为 "US/Eastern"
        i = date_range("20130101", periods=3, tz="US/Eastern")

        # 遍历不同的数组或索引情况
        for arr in [np.nan, pd.NaT]:
            # 应用 where 方法，其中 other 参数是 ndarray 或 Index
            result = i.where(notna(i), other=arr)
            # 期望的结果与原始索引相同
            expected = i
            tm.assert_index_equal(result, expected)

        # 创建一个副本 i2
        i2 = i.copy()
        # 修改副本 i2 为一个新的 Index，包含 NaN 值和部分原始 i 的值
        i2 = Index([pd.NaT, pd.NaT] + i[2:].tolist())
        # 使用 where 方法，其中 other 参数是 Index 对象
        result = i.where(notna(i2), i2)
        # 断言结果与期望相等
        tm.assert_index_equal(result, i2)

        # 再次创建 i2 的副本
        i2 = i.copy()
        # 修改副本 i2 为一个新的 Index，包含 NaN 值和部分原始 i 的值
        i2 = Index([pd.NaT, pd.NaT] + i[2:].tolist())
        # 使用 where 方法，其中 other 参数是 i2 的值数组
        result = i.where(notna(i2), i2._values)
        # 断言结果与期望相等
        tm.assert_index_equal(result, i2)

    # 测试函数：测试 where 方法中的无效数据类型情况
    def test_where_invalid_dtypes(self):
        # 创建一个日期范围，从 "20130101" 开始，包含三个日期，时区为 "US/Eastern"
        dti = date_range("20130101", periods=3, tz="US/Eastern")

        # 获取部分日期列表
        tail = dti[2:].tolist()
        # 创建一个新的 Index，包含 NaN 值和部分原始 dti 的值
        i2 = Index([pd.NaT, pd.NaT] + tail)

        # 创建一个 mask，指示 i2 中的非 NaN 值
        mask = notna(i2)

        # 应用 where 方法，其中 other 参数是 i2 的值数组
        result = dti.where(mask, i2.values)
        # 期望的结果，创建一个新的 Index，包含指定的值和类型
        expected = Index([pd.NaT.asm8, pd.NaT.asm8] + tail, dtype=object)
        tm.assert_index_equal(result, expected)

        # 将 dti 转换为无时区的日期时间索引 naive
        naive = dti.tz_localize(None)
        # 应用 where 方法，其中 other 参数是 i2
        result = naive.where(mask, i2)
        # 期望的结果，创建一个新的 Index，包含指定的值和类型
        expected = Index([i2[0], i2[1]] + naive[2:].tolist(), dtype=object)
        tm.assert_index_equal(result, expected)

        # 将 i2 转换为周期性的日期时间索引 pi
        pi = i2.tz_localize(None).to_period("D")
        # 应用 where 方法，其中 other 参数是 pi
        result = dti.where(mask, pi)
        # 期望的结果，创建一个新的 Index，包含指定的值和类型
        expected = Index([pi[0], pi[1]] + tail, dtype=object)
        tm.assert_index_equal(result, expected)

        # 将 i2 的值视为 timedelta64 类型的数据数组 tda
        tda = i2.astype('timedelta64[ns]').view('timedelta64[ns]')
        # 应用 where 方法，其中 other 参数是 tda
        result = dti.where(mask, tda)
        # 期望的结果，创建一个新的 Index，包含指定的值和类型
        expected = Index([tda[0], tda[1]] + tail, dtype=object)
        assert isinstance(expected[0], np.timedelta64)
        tm.assert_index_equal(result, expected)

        # 应用 where 方法，其中 other 参数是 i2 的值数组的视图
        result = dti.where(mask, i2.astype('int64'))
        # 期望的结果，创建一个新的 Index，包含指定的值和类型
        expected = Index([pd.NaT.value, pd.NaT.value] + tail, dtype=object)
        assert isinstance(expected[0], int)
        tm.assert_index_equal(result, expected)

        # 创建一个 timedelta 类型的标量 td
        td = pd.Timedelta(days=4)
        # 应用 where 方法，其中 other 参数是 td
        result = dti.where(mask, td)
        # 期望的结果，创建一个新的 Index，包含指定的值和类型
        expected = Index([td, td] + tail, dtype=object)
        assert expected[0] is td
        tm.assert_index_equal(result, expected)

    # 测试函数：测试 where 方法中处理不匹配 NaT 的情况
    def test_where_mismatched_nat(self, tz_aware_fixture):
        # 获取时区信息
        tz = tz_aware_fixture
        # 创建一个日期范围，从 "2013-01-01" 开始，包含三个日期，具有指定的时区
        dti = date_range("2013-01-01", periods=3, tz=tz)
        # 条件数组，指示哪些日期保留
        cond = np.array([True, False, True])

        # 创建一个 timedelta64 类型的 NaT 对象 tdnat
        tdnat = np.timedelta64("NaT", "ns")
        # 期望的结果，创建一个新的 Index，包含指定的值和类型
        expected = Index([dti[0], tdnat, dti[2]], dtype=object)
        assert expected[1] is tdnat

        # 应用 where 方法，根据条件 cond 和 other 参数 tdnat，返回结果
        result = dti.where(cond, tdnat)
        # 断言结果与期望相等
        tm.assert_index_equal(result, expected)
    # 定义一个测试函数，用于测试时间序列对象的时区处理功能
    def test_where_tz(self):
        # 创建一个日期范围对象，从"20130101"开始，包含3个周期，使用"US/Eastern"时区
        i = date_range("20130101", periods=3, tz="US/Eastern")
        
        # 对日期范围对象执行 where 操作，保留非空值的部分，存储结果在 result 变量中
        result = i.where(notna(i))
        
        # 将预期结果设定为原始日期范围对象 i
        expected = i
        
        # 使用测试工具 tm.assert_index_equal 检查 result 和 expected 是否相等
        tm.assert_index_equal(result, expected)
        
        # 复制日期范围对象 i 到 i2
        i2 = i.copy()
        
        # 创建一个新的索引对象 i2，包含 pd.NaT 和 pd.NaT，以及从 i 的第三个元素开始的其余元素
        i2 = Index([pd.NaT, pd.NaT] + i[2:].tolist())
        
        # 对日期范围对象 i 执行 where 操作，保留非空值的部分，存储结果在 result 变量中
        result = i.where(notna(i2))
        
        # 将预期结果设定为新创建的索引对象 i2
        expected = i2
        
        # 使用测试工具 tm.assert_index_equal 检查 result 和 expected 是否相等
        tm.assert_index_equal(result, expected)
    # 定义一个测试类 TestTake
    class TestTake:
        # 使用 pytest 的 parametrize 装饰器，参数化 tzstr 参数为 ["US/Eastern", "dateutil/US/Eastern"]
        @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
        # 定义测试方法 test_dti_take_dont_lose_meta，接受参数 tzstr
        def test_dti_take_dont_lose_meta(self, tzstr):
            # 调用 date_range 函数生成一个时间范围对象 rng，起始日期为 "1/1/2000"，共计 20 个时间点，时区为 tzstr
            rng = date_range("1/1/2000", periods=20, tz=tzstr)

            # 调用 rng 对象的 take 方法，选取前 5 个时间点组成 result 对象
            result = rng.take(range(5))
            # 断言结果对象的时区与原始 rng 对象的时区相同
            assert result.tz == rng.tz
            # 断言结果对象的频率与原始 rng 对象的频率相同
            assert result.freq == rng.freq

        # 定义测试方法 test_take_nan_first_datetime
        def test_take_nan_first_datetime(self):
            # 创建一个 DatetimeIndex 对象 index，包含 NaT、"20130101"、"20130102" 三个时间点
            index = DatetimeIndex([pd.NaT, Timestamp("20130101"), Timestamp("20130102")])
            # 调用 index 对象的 take 方法，根据给定的索引选择时间点，生成 result 对象
            result = index.take([-1, 0, 1])
            # 创建一个预期的 DatetimeIndex 对象 expected，按照索引顺序选择 index 对象中的时间点
            expected = DatetimeIndex([index[-1], index[0], index[1]])
            # 使用 assert_index_equal 函数断言 result 与 expected 相等
            tm.assert_index_equal(result, expected)

        # 使用 pytest 的 parametrize 装饰器，参数化 tz 参数为 [None, "Asia/Tokyo"]
        @pytest.mark.parametrize("tz", [None, "Asia/Tokyo"])
        # 定义测试方法 test_take，接受参数 tz
        def test_take(self, tz):
            # 创建一个时间范围对象 idx，从 "2011-01-01" 到 "2011-01-31"，每日频率，名称为 "idx"，时区为 tz
            idx = date_range("2011-01-01", "2011-01-31", freq="D", name="idx", tz=tz)

            # 调用 idx 对象的 take 方法，选取索引为 [0] 的时间点，生成 result 对象
            result = idx.take([0])
            # 断言 result 对象等于指定的 Timestamp 对象 "2011-01-01"，带有 idx 对象的时区
            assert result == Timestamp("2011-01-01", tz=idx.tz)

            # 再次调用 idx 对象的 take 方法，选取索引为 [0, 1, 2] 的时间点，生成 result 对象
            result = idx.take([0, 1, 2])
            # 创建预期的时间范围对象 expected，包含从 "2011-01-01" 到 "2011-01-03" 的时间点，每日频率，带有 idx 对象的时区，名称为 "idx"
            expected = date_range(
                "2011-01-01", "2011-01-03", freq="D", tz=idx.tz, name="idx"
            )
            # 使用 assert_index_equal 函数断言 result 与 expected 相等
            tm.assert_index_equal(result, expected)
            # 断言 result 对象的频率与 expected 相同
            assert result.freq == expected.freq

            # 类似地，依次测试其他选取索引的情况，分别断言结果与预期相等

        # 定义测试方法 test_take_invalid_kwargs
        def test_take_invalid_kwargs(self):
            # 创建一个时间范围对象 idx，从 "2011-01-01" 到 "2011-01-31"，每日频率，名称为 "idx"
            idx = date_range("2011-01-01", "2011-01-31", freq="D", name="idx")
            # 定义一个索引列表 indices
            indices = [1, 6, 5, 9, 10, 13, 15, 3]

            # 测试在调用 take 方法时传入未预期的关键字参数 'foo'，应抛出 TypeError 异常，异常信息为指定的 msg
            msg = r"take\(\) got an unexpected keyword argument 'foo'"
            with pytest.raises(TypeError, match=msg):
                idx.take(indices, foo=2)

            # 测试在调用 take 方法时传入不支持的 'out' 参数，应抛出 ValueError 异常，异常信息为指定的 msg
            msg = "the 'out' parameter is not supported"
            with pytest.raises(ValueError, match=msg):
                idx.take(indices, out=indices)

            # 测试在调用 take 方法时传入不支持的 'mode' 参数，应抛出 ValueError 异常，异常信息为指定的 msg
            msg = "the 'mode' parameter is not supported"
            with pytest.raises(ValueError, match=msg):
                idx.take(indices, mode="clip")

        # TODO: This method came from test_datetime; de-dup with version above
        # 使用 pytest 的 parametrize 装饰器，参数化 tz 参数为 [None, "US/Eastern", "Asia/Tokyo"]
    # 定义一个测试方法，用于测试在特定时区下的时间序列操作
    def test_take2(self, tz):
        # 创建一个包含几个日期时间对象的列表
        dates = [
            datetime(2010, 1, 1, 14),
            datetime(2010, 1, 1, 15),
            datetime(2010, 1, 1, 17),
            datetime(2010, 1, 1, 21),
        ]

        # 生成一个日期时间索引对象，起始时间为 "2010-01-01 09:00"，结束时间为 "2010-02-01 09:00"，频率为每小时一次
        idx = date_range(
            start="2010-01-01 09:00",
            end="2010-02-01 09:00",
            freq="h",
            tz=tz,
            name="idx",
        )

        # 创建一个预期的日期时间索引对象，与之前生成的 idx 对象具有相同的日期时间
        expected = DatetimeIndex(dates, freq=None, name="idx", dtype=idx.dtype)

        # 使用 take 方法获取索引为 [5, 6, 8, 12] 的时间戳，返回一个新的 DatetimeIndex 对象
        taken1 = idx.take([5, 6, 8, 12])

        # 使用索引运算符直接获取索引为 [5, 6, 8, 12] 的时间戳，同样返回一个新的 DatetimeIndex 对象
        taken2 = idx[[5, 6, 8, 12]]

        # 对每一个 taken 对象进行以下断言操作
        for taken in [taken1, taken2]:
            # 断言 taken 与预期的日期时间索引对象 expected 相等
            tm.assert_index_equal(taken, expected)
            # 断言 taken 是 DatetimeIndex 类型的对象
            assert isinstance(taken, DatetimeIndex)
            # 断言 taken 的频率为 None
            assert taken.freq is None
            # 断言 taken 的时区与 expected 相同
            assert taken.tz == expected.tz
            # 断言 taken 的名称与 expected 相同
            assert taken.name == expected.name

    # 定义一个测试方法，用于测试 take 方法的填充值功能
    def test_take_fill_value(self):
        # 创建一个具有日期时间索引的对象 idx，指定名称为 "xxx"
        idx = DatetimeIndex(["2011-01-01", "2011-02-01", "2011-03-01"], name="xxx")
        
        # 使用 take 方法获取指定索引 [1, 0, -1] 的日期时间戳，返回结果应与 expected 相等
        result = idx.take(np.array([1, 0, -1]))
        expected = DatetimeIndex(["2011-02-01", "2011-01-01", "2011-03-01"], name="xxx")
        tm.assert_index_equal(result, expected)

        # 使用 take 方法获取指定索引 [1, 0, -1] 的日期时间戳，设置 fill_value=True，填充值为 NaT
        result = idx.take(np.array([1, 0, -1]), fill_value=True)
        expected = DatetimeIndex(["2011-02-01", "2011-01-01", "NaT"], name="xxx")
        tm.assert_index_equal(result, expected)

        # 使用 take 方法获取指定索引 [1, 0, -1] 的日期时间戳，设置 allow_fill=False，不允许填充值
        result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        expected = DatetimeIndex(["2011-02-01", "2011-01-01", "2011-03-01"], name="xxx")
        tm.assert_index_equal(result, expected)

        # 测试在 fill_value=True 且 fill_value 不为 None 时，如果索引数组中存在小于 -1 的索引，是否会抛出 ValueError 异常
        msg = (
            "When allow_fill=True and fill_value is not None, "
            "all indices must be >= -1"
        )
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -5]), fill_value=True)

        # 测试在索引数组中存在超出范围的索引时，是否会抛出 IndexError 异常
        msg = "out of bounds"
        with pytest.raises(IndexError, match=msg):
            idx.take(np.array([1, -5]))
    # 定义测试函数 test_take_fill_value_with_timezone，用于测试时间索引的 take 方法
    def test_take_fill_value_with_timezone(self):
        # 创建一个带时区的日期时间索引 idx，包含三个日期
        idx = DatetimeIndex(
            ["2011-01-01", "2011-02-01", "2011-03-01"], name="xxx", tz="US/Eastern"
        )
        # 使用 take 方法按照给定的索引数组 [1, 0, -1] 获取对应的日期时间索引
        result = idx.take(np.array([1, 0, -1]))
        # 期望的结果是根据给定索引数组重新排列后的日期时间索引
        expected = DatetimeIndex(
            ["2011-02-01", "2011-01-01", "2011-03-01"], name="xxx", tz="US/Eastern"
        )
        # 断言结果与期望相等
        tm.assert_index_equal(result, expected)

        # 使用 take 方法并指定 fill_value=True，处理索引数组 [1, 0, -1]，在越界时填充 NaT
        result = idx.take(np.array([1, 0, -1]), fill_value=True)
        # 期望的结果是在越界时填充 NaT 的日期时间索引
        expected = DatetimeIndex(
            ["2011-02-01", "2011-01-01", "NaT"], name="xxx", tz="US/Eastern"
        )
        # 断言结果与期望相等
        tm.assert_index_equal(result, expected)

        # 使用 take 方法并指定 allow_fill=False 和 fill_value=True，不允许填充，处理索引数组 [1, 0, -1]
        result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        # 期望的结果是根据给定索引数组重新排列后的日期时间索引，不允许填充
        expected = DatetimeIndex(
            ["2011-02-01", "2011-01-01", "2011-03-01"], name="xxx", tz="US/Eastern"
        )
        # 断言结果与期望相等
        tm.assert_index_equal(result, expected)

        # 定义错误消息字符串，用于下面的异常断言
        msg = (
            "When allow_fill=True and fill_value is not None, "
            "all indices must be >= -1"
        )
        # 使用 pytest 来断言当 allow_fill=True 且 fill_value 不为空时，索引数组中所有索引必须 >= -1
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -5]), fill_value=True)

        # 定义错误消息字符串，用于下面的异常断言
        msg = "out of bounds"
        # 使用 pytest 来断言当索引数组中的索引超出范围时会抛出 IndexError 异常
        with pytest.raises(IndexError, match=msg):
            idx.take(np.array([1, -5]))
class TestGetLoc:
    # 测试获取位置键与单位不匹配的情况
    def test_get_loc_key_unit_mismatch(self):
        # 创建一个日期范围索引，包括三个周期
        idx = date_range("2000-01-01", periods=3)
        # 将第二个索引项按毫秒单位转换为时间戳
        key = idx[1].as_unit("ms")
        # 获取 key 在索引中的位置
        loc = idx.get_loc(key)
        # 断言位置应为 1
        assert loc == 1
        # 断言 key 在索引中存在
        assert key in idx

    # 测试获取位置键与单位不匹配且无法转换的情况
    def test_get_loc_key_unit_mismatch_not_castable(self):
        # 创建一个日期范围，将数据类型转换为以秒为单位的日期时间
        dta = date_range("2000-01-01", periods=3)._data.astype("M8[s]")
        dti = DatetimeIndex(dta)
        # 创建一个时间戳，单位为纳秒，并加上 1 纳秒
        key = dta[0].as_unit("ns") + pd.Timedelta(1)

        # 使用 pytest 引发 KeyError，匹配指定的错误消息
        with pytest.raises(
            KeyError, match=r"Timestamp\('2000-01-01 00:00:00.000000001'\)"
        ):
            dti.get_loc(key)

        # 断言 key 不在索引中
        assert key not in dti

    # 测试使用时间对象进行索引
    def test_get_loc_time_obj(self):
        # 创建一个小时频率的日期范围索引
        idx = date_range("2000-01-01", periods=24, freq="h")

        # 获取时间对象 12 点在索引中的位置
        result = idx.get_loc(time(12))
        expected = np.array([12])
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

        # 获取时间对象 12 点 30 分在索引中的位置
        result = idx.get_loc(time(12, 30))
        expected = np.array([])
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    # 使用参数化测试，测试时间对象索引的第二种情况
    @pytest.mark.parametrize("offset", [-10, 10])
    def test_get_loc_time_obj2(self, monkeypatch, offset):
        # GH#8667
        size_cutoff = 50
        n = size_cutoff + offset
        key = time(15, 11, 30)
        start = key.hour * 3600 + key.minute * 60 + key.second

        # 使用 monkeypatch 设置库中的大小截止值
        with monkeypatch.context():
            monkeypatch.setattr(libindex, "_SIZE_CUTOFF", size_cutoff)
            # 创建一个秒频率的日期范围索引
            idx = date_range("2014-11-26", periods=n, freq="s")
            # 创建一个随机数列的时间序列
            ts = pd.Series(np.random.default_rng(2).standard_normal(n), index=idx)
            locs = np.arange(start, n, 24 * 3600, dtype=np.intp)

            # 获取时间对象 key 在索引中的位置
            result = ts.index.get_loc(key)
            tm.assert_numpy_array_equal(result, locs)
            tm.assert_series_equal(ts[key], ts.iloc[locs])

            # 创建左右两个副本，并断言它们的序列相等
            left, right = ts.copy(), ts.copy()
            left[key] *= -10
            right.iloc[locs] *= -10
            tm.assert_series_equal(left, right)

    # 测试获取位置自然时间对象的情况
    def test_get_loc_time_nat(self):
        # GH#35114
        # 创建一个包含 NaT 的日期时间索引
        dti = DatetimeIndex([pd.NaT])
        tic = time(minute=12, second=43, microsecond=145224)

        # 获取时间对象 tic 在索引中的位置
        loc = dti.get_loc(tic)
        expected = np.array([], dtype=np.intp)
        tm.assert_numpy_array_equal(loc, expected)

    # 测试获取位置 NaT 对象的情况
    def test_get_loc_nat(self):
        # GH#20464
        # 创建一个日期时间索引，包含 "NaT"
        index = DatetimeIndex(["1/3/2000", "NaT"])

        # 断言获取 NaT 在索引中的位置为 1
        assert index.get_loc(pd.NaT) == 1

        # 断言获取 None 在索引中的位置为 1
        assert index.get_loc(None) == 1

        # 断言获取 NaN 在索引中的位置为 1
        assert index.get_loc(np.nan) == 1

        # 断言获取 NA 在索引中的位置为 1
        assert index.get_loc(pd.NA) == 1

        # 使用 pytest 引发 KeyError，匹配 "NaT" 的错误消息
        with pytest.raises(KeyError, match="NaT"):
            index.get_loc(np.timedelta64("NaT"))

    # 参数化测试，测试 timedelta 对象的情况
    @pytest.mark.parametrize("key", [pd.Timedelta(0), pd.Timedelta(1), timedelta(0)])
    # 定义一个测试方法，用于测试在给定无效键时获取时间差的行为
    def test_get_loc_timedelta_invalid_key(self, key):
        # 创建一个日期范围对象，从1970年1月1日开始，持续10个周期
        dti = date_range("1970-01-01", periods=10)
        # 定义一个错误消息，用于匹配抛出的类型错误异常
        msg = "Cannot index DatetimeIndex with [Tt]imedelta"
        # 使用 pytest 的断言，期望抛出 TypeError 异常，并且异常消息匹配预期的错误消息
        with pytest.raises(TypeError, match=msg):
            # 调用 DatetimeIndex 的 get_loc 方法，尝试使用给定的键来获取位置
            dti.get_loc(key)

    # 定义一个测试方法，用于测试在合理的键错误时获取位置的行为
    def test_get_loc_reasonable_key_error(self):
        # 创建一个日期时间索引对象，包含单个日期 "1/3/2000"
        index = DatetimeIndex(["1/3/2000"])
        # 使用 pytest 的断言，期望抛出 KeyError 异常，并且异常消息中包含字符串 "2000"
        with pytest.raises(KeyError, match="2000"):
            # 调用 DatetimeIndex 的 get_loc 方法，尝试使用给定的键来获取位置
            index.get_loc("1/1/2000")

    # 定义一个测试方法，用于测试在使用年份字符串作为键时获取位置的行为
    def test_get_loc_year_str(self):
        # 创建一个日期范围对象，从 "1/1/2000" 到 "1/1/2010"
        rng = date_range("1/1/2000", "1/1/2010")

        # 调用日期范围对象的 get_loc 方法，尝试使用年份字符串 "2009" 来获取位置
        result = rng.get_loc("2009")
        # 定义预期的结果，使用切片来表示从索引 3288 到 3652 的范围
        expected = slice(3288, 3653)
        # 使用断言来验证实际结果与预期结果是否相符
        assert result == expected
class TestContains:
    # 测试在具有重复日期的DatetimeIndex中，日期是否存在
    def test_dti_contains_with_duplicates(self):
        # 创建一个特定的日期时间对象
        d = datetime(2011, 12, 5, 20, 30)
        # 创建包含重复日期的DatetimeIndex对象
        ix = DatetimeIndex([d, d])
        # 断言特定日期是否在DatetimeIndex中
        assert d in ix

    # 参数化测试，验证DatetimeIndex中存在非唯一日期的情况
    @pytest.mark.parametrize(
        "vals",
        [
            [0, 1, 0],
            [0, 0, -1],
            [0, -1, -1],
            ["2015", "2015", "2016"],
            ["2015", "2015", "2014"],
        ],
    )
    def test_contains_nonunique(self, vals):
        # GH#9512，关于GitHub上的特定问题编号
        # 创建DatetimeIndex对象，包含给定的日期列表vals
        idx = DatetimeIndex(vals)
        # 断言第一个日期是否在DatetimeIndex中
        assert idx[0] in idx


class TestGetIndexer:
    # 测试根据日期对象获取索引器
    def test_get_indexer_date_objs(self):
        # 创建一个日期范围对象rng，从"1/1/2000"开始，包含20个时间点
        rng = date_range("1/1/2000", periods=20)
        # 调用map函数将日期转换为日期对象，然后获取其索引器
        result = rng.get_indexer(rng.map(lambda x: x.date()))
        # 获取日期范围对象rng自身的索引器
        expected = rng.get_indexer(rng)
        # 使用测试工具函数tm.assert_numpy_array_equal进行结果断言
        tm.assert_numpy_array_equal(result, expected)

    # 测试获取索引器的不同方法和选项
    def test_get_indexer(self):
        # 创建一个日期范围对象idx，从"2000-01-01"开始，包含3个日期
        idx = date_range("2000-01-01", periods=3)
        # 创建一个预期的整数数组exp，用于比较索引器的预期输出
        exp = np.array([0, 1, 2], dtype=np.intp)
        # 断言获取idx自身的索引器与预期的整数数组exp相等
        tm.assert_numpy_array_equal(idx.get_indexer(idx), exp)

        # 创建目标日期对象target，根据idx[0]和一些时间增量生成
        target = idx[0] + pd.to_timedelta(["-1 hour", "12 hours", "1 day 1 hour"])
        # 断言使用"pad"方法获取target的索引器与预期的整数数组相等
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "pad"), np.array([-1, 0, 1], dtype=np.intp)
        )
        # 断言使用"backfill"方法获取target的索引器与预期的整数数组相等
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "backfill"), np.array([0, 1, 2], dtype=np.intp)
        )
        # 断言使用"nearest"方法获取target的索引器与预期的整数数组相等
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "nearest"), np.array([0, 1, 1], dtype=np.intp)
        )
        # 断言使用"nearest"方法和指定的时间容差获取target的索引器与预期的整数数组相等
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "nearest", tolerance=pd.Timedelta("1 hour")),
            np.array([0, -1, 1], dtype=np.intp),
        )
        # 定义一个时间容差列表tol_raw
        tol_raw = [
            pd.Timedelta("1 hour"),
            pd.Timedelta("1 hour"),
            pd.Timedelta("1 hour").to_timedelta64(),
        ]
        # 断言使用"nearest"方法和指定的时间容差列表tol_raw获取target的索引器与预期的整数数组相等
        tm.assert_numpy_array_equal(
            idx.get_indexer(
                target, "nearest", tolerance=[np.timedelta64(x) for x in tol_raw]
            ),
            np.array([0, -1, 1], dtype=np.intp),
        )
        # 定义一个无效的时间容差列表tol_bad
        tol_bad = [
            pd.Timedelta("2 hour").to_timedelta64(),
            pd.Timedelta("1 hour").to_timedelta64(),
            "foo",
        ]
        # 断言当使用无效的时间容差列表tol_bad时，会抛出预期的ValueError异常
        msg = "Could not convert 'foo' to NumPy timedelta"
        with pytest.raises(ValueError, match=msg):
            idx.get_indexer(target, "nearest", tolerance=tol_bad)
        # 断言当使用无效的method参数时，会抛出预期的ValueError异常
        with pytest.raises(ValueError, match="abbreviation w/o a number"):
            idx.get_indexer(idx[[0]], method="nearest", tolerance="foo")

    # 参数化测试，验证不同类型的日期目标列表
    @pytest.mark.parametrize(
        "target",
        [
            [date(2020, 1, 1), Timestamp("2020-01-02")],
            [Timestamp("2020-01-01"), date(2020, 1, 2)],
        ],
    )
    # 测试函数：测试在混合数据类型情况下获取索引器的行为
    def test_get_indexer_mixed_dtypes(self, target):
        # 引用的GitHub问题链接，描述了与此测试相关的问题背景
        # https://github.com/pandas-dev/pandas/issues/33741
        # 创建包含两个时间戳的DatetimeIndex对象
        values = DatetimeIndex([Timestamp("2020-01-01"), Timestamp("2020-01-02")])
        # 获取目标索引器结果
        result = values.get_indexer(target)
        # 预期的索引器结果数组
        expected = np.array([0, 1], dtype=np.intp)
        # 使用测试工具函数验证实际结果与预期结果是否一致
        tm.assert_numpy_array_equal(result, expected)

    # 参数化测试函数：测试越界日期情况下获取索引器的行为
    @pytest.mark.parametrize(
        "target, positions",
        [
            # 第一个参数化情况：一个日期和一个时间戳
            ([date(9999, 1, 1), Timestamp("2020-01-01")], [-1, 0]),
            # 第二个参数化情况：一个时间戳和一个日期
            ([Timestamp("2020-01-01"), date(9999, 1, 1)], [0, -1]),
            # 第三个参数化情况：两个日期
            ([date(9999, 1, 1), date(9999, 1, 1)], [-1, -1]),
        ],
    )
    # 测试函数：测试越界日期情况下获取索引器的行为
    def test_get_indexer_out_of_bounds_date(self, target, positions):
        # 创建包含两个时间戳的DatetimeIndex对象
        values = DatetimeIndex([Timestamp("2020-01-01"), Timestamp("2020-01-02")])
        # 获取目标索引器结果
        result = values.get_indexer(target)
        # 根据参数化的positions创建预期的索引器结果数组
        expected = np.array(positions, dtype=np.intp)
        # 使用测试工具函数验证实际结果与预期结果是否一致
        tm.assert_numpy_array_equal(result, expected)

    # 测试函数：测试在需要单调性的情况下进行填充获取索引器的行为
    def test_get_indexer_pad_requires_monotonicity(self):
        # 创建日期范围对象rng，从"1/1/2000"到"3/1/2000"，按工作日频率
        rng = date_range("1/1/2000", "3/1/2000", freq="B")
        # 创建rng的一个非单调递增或递减的子集rng2
        rng2 = rng[[1, 0, 2]]
        # 预期引发的异常消息
        msg = "index must be monotonic increasing or decreasing"
        # 使用pytest的上下文管理器检查是否引发了预期的ValueError异常，且异常消息匹配预期消息
        with pytest.raises(ValueError, match=msg):
            # 对rng2使用get_indexer方法，指定填充方法为"pad"
            rng2.get_indexer(rng, method="pad")
class TestMaybeCastSliceBound:
    def test_maybe_cast_slice_bounds_empty(self):
        # GH#14354
        # 创建一个空时间序列，频率为每小时一次，没有时间段，截止时间为2015年
        empty_idx = date_range(freq="1h", periods=0, end="2015")

        # 将右边界可能转换为时间戳，结果应为"2015-01-02 23:59:59.999999999"
        right = empty_idx._maybe_cast_slice_bound("2015-01-02", "right")
        exp = Timestamp("2015-01-02 23:59:59.999999999")
        assert right == exp

        # 将左边界可能转换为时间戳，结果应为"2015-01-02 00:00:00"
        left = empty_idx._maybe_cast_slice_bound("2015-01-02", "left")
        exp = Timestamp("2015-01-02 00:00:00")
        assert left == exp

    def test_maybe_cast_slice_duplicate_monotonic(self):
        # https://github.com/pandas-dev/pandas/issues/16515
        # 创建一个具有重复日期的时间索引
        idx = DatetimeIndex(["2017", "2017"])
        # 将左边界可能转换为时间戳，结果应为"2017-01-01"
        result = idx._maybe_cast_slice_bound("2017-01-01", "left")
        expected = Timestamp("2017-01-01")
        assert result == expected


class TestGetSliceBounds:
    @pytest.mark.parametrize("box", [date, datetime, Timestamp])
    @pytest.mark.parametrize("side, expected", [("left", 4), ("right", 5)])
    def test_get_slice_bounds_datetime_within(
        self, box, side, expected, tz_aware_fixture
    ):
        # GH 35690
        # 获取一个时间范围内的工作日索引，并进行时区本地化
        tz = tz_aware_fixture
        index = bdate_range("2000-01-03", "2000-02-11").tz_localize(tz)
        key = box(year=2000, month=1, day=7)

        if tz is not None:
            with pytest.raises(TypeError, match="Cannot compare tz-naive"):
                # GH#36148 we require tzawareness-compat as of 2.0
                # 如果未指定时区，应该引发类型错误异常
                index.get_slice_bound(key, side=side)
        else:
            # 获取时间索引的切片边界，结果应为预期的索引位置
            result = index.get_slice_bound(key, side=side)
            assert result == expected

    @pytest.mark.parametrize("box", [datetime, Timestamp])
    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("year, expected", [(1999, 0), (2020, 30)])
    def test_get_slice_bounds_datetime_outside(
        self, box, side, year, expected, tz_aware_fixture
    ):
        # GH 35690
        # 获取一个时间范围内的工作日索引，并进行时区本地化
        tz = tz_aware_fixture
        index = bdate_range("2000-01-03", "2000-02-11").tz_localize(tz)
        key = box(year=year, month=1, day=7)

        if tz is not None:
            with pytest.raises(TypeError, match="Cannot compare tz-naive"):
                # GH#36148 we require tzawareness-compat as of 2.0
                # 如果未指定时区，应该引发类型错误异常
                index.get_slice_bound(key, side=side)
        else:
            # 获取时间索引的切片边界，结果应为预期的索引位置
            result = index.get_slice_bound(key, side=side)
            assert result == expected

    @pytest.mark.parametrize("box", [datetime, Timestamp])
    # 定义一个测试方法，用于测试 slice_locs 方法在处理带有时区信息的 DatetimeIndex 时的行为
    def test_slice_datetime_locs(self, box, tz_aware_fixture):
        # GH 34077：GitHub 上的 issue 编号，标识这段代码相关的问题
        tz = tz_aware_fixture
        # 创建一个带有时区信息的 DatetimeIndex 对象
        index = DatetimeIndex(["2010-01-01", "2010-01-03"]).tz_localize(tz)
        # 创建一个日期作为 slice_locs 方法的键值，以及一个作为结束值的日期
        key = box(2010, 1, 1)

        # 如果时区信息不为 None，则执行以下代码块
        if tz is not None:
            # 使用 pytest 模块断言，期望会抛出 TypeError 异常，并且异常信息包含 "Cannot compare tz-naive"
            with pytest.raises(TypeError, match="Cannot compare tz-naive"):
                # GH#36148: GitHub 上的另一个 issue 编号，指出这个断言的需要
                # 调用 DatetimeIndex 的 slice_locs 方法，尝试对比两个带时区信息的日期
                index.slice_locs(key, box(2010, 1, 2))
        # 如果时区信息为 None，则执行以下代码块
        else:
            # 调用 DatetimeIndex 的 slice_locs 方法，获取 key 到 box(2010, 1, 2) 之间的切片位置
            result = index.slice_locs(key, box(2010, 1, 2))
            # 预期的切片位置
            expected = (0, 1)
            # 使用断言验证实际结果与预期结果是否一致
            assert result == expected
# 定义一个测试类 TestIndexerBetweenTime，用于测试时间索引器的功能
class TestIndexerBetweenTime:
    
    # 定义测试方法 test_indexer_between_time，测试时间范围内索引器的行为
    def test_indexer_between_time(self):
        # GH#11818: 引用 GitHub issue 编号 11818
        # 创建一个日期范围对象 rng，从 "1/1/2000" 到 "1/5/2000"，频率为每5分钟
        rng = date_range("1/1/2000", "1/5/2000", freq="5min")
        
        # 定义错误消息，测试在特定情况下是否会抛出 ValueError 异常
        msg = r"Cannot convert arg \[datetime\.datetime\(2010, 1, 2, 1, 0\)\] to a time"
        
        # 使用 pytest 检查是否抛出指定异常和消息
        with pytest.raises(ValueError, match=msg):
            # 调用 rng 对象的 indexer_between_time 方法，期望在时间点 datetime(2010, 1, 2, 1) 到 datetime(2010, 1, 2, 5) 之间抛出异常
            rng.indexer_between_time(datetime(2010, 1, 2, 1), datetime(2010, 1, 2, 5))

    # 使用 pytest 的参数化装饰器，多次运行相同测试方法，每次使用不同的 unit 参数
    @pytest.mark.parametrize("unit", ["us", "ms", "s"])
    def test_indexer_between_time_non_nano(self, unit):
        # 对于简单的情况，非纳秒级的 indexer_between_time 应该与纳秒级的结果匹配

        # 创建日期范围对象 rng，从 "1/1/2000" 到 "1/5/2000"，频率为每5分钟
        rng = date_range("1/1/2000", "1/5/2000", freq="5min")
        
        # 获取 rng 对象内部的 numpy ndarray 数据
        arr_nano = rng._data._ndarray
        
        # 将 arr_nano 转换为指定单位 unit 的 numpy datetime 数组 arr
        arr = arr_nano.astype(f"M8[{unit}]")
        
        # 使用 arr 的类型创建新的数据对象 dta
        dta = type(rng._data)._simple_new(arr, dtype=arr.dtype)
        
        # 使用 dta 创建新的 DatetimeIndex 对象 dti
        dti = DatetimeIndex(dta)
        
        # 断言 dti 对象的数据类型与 arr 的数据类型相同
        assert dti.dtype == arr.dtype
        
        # 定义起始时间 tic 和结束时间 toc
        tic = time(1, 25)
        toc = time(2, 29)
        
        # 调用 dti 对象的 indexer_between_time 方法，返回时间范围内的索引数组 result
        result = dti.indexer_between_time(tic, toc)
        
        # 调用 rng 对象的 indexer_between_time 方法，期望返回相同的索引数组 expected
        expected = rng.indexer_between_time(tic, toc)
        
        # 使用 pytest 的 assert 辅助方法，比较两个 numpy 数组 result 和 expected 是否相等
        tm.assert_numpy_array_equal(result, expected)

        # 考虑包含非零微秒参数的情况
        tic = time(1, 25, 0, 45678)
        toc = time(2, 29, 0, 1234)
        
        # 再次调用 dti 对象的 indexer_between_time 方法，返回时间范围内的索引数组 result
        result = dti.indexer_between_time(tic, toc)
        
        # 再次调用 rng 对象的 indexer_between_time 方法，期望返回相同的索引数组 expected
        expected = rng.indexer_between_time(tic, toc)
        
        # 使用 pytest 的 assert 辅助方法，比较两个 numpy 数组 result 和 expected 是否相等
        tm.assert_numpy_array_equal(result, expected)
```