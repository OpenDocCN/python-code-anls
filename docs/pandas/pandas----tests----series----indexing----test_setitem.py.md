# `D:\src\scipysrc\pandas\pandas\tests\series\indexing\test_setitem.py`

```
    # 从 datetime 模块导入 date 和 datetime 类
    from datetime import (
        date,
        datetime,
    )
    # 从 decimal 模块导入 Decimal 类
    from decimal import Decimal

    # 导入 numpy 库并用 np 别名表示
    import numpy as np
    # 导入 pytest 库
    import pytest

    # 从 pandas.compat.numpy 模块导入 np_version_gte1p24 函数
    from pandas.compat.numpy import np_version_gte1p24
    # 从 pandas.errors 模块导入 IndexingError 异常类
    from pandas.errors import IndexingError

    # 从 pandas.core.dtypes.common 模块导入 is_list_like 函数
    from pandas.core.dtypes.common import is_list_like

    # 从 pandas 模块导入多个类和函数，用逗号分隔
    from pandas import (
        NA,
        Categorical,
        DataFrame,
        DatetimeIndex,
        Index,
        Interval,
        IntervalIndex,
        MultiIndex,
        NaT,
        Period,
        Series,
        Timedelta,
        Timestamp,
        array,
        concat,
        date_range,
        interval_range,
        period_range,
        timedelta_range,
    )
    # 导入 pandas._testing 模块并用 tm 别名表示
    import pandas._testing as tm

    # 从 pandas.tseries.offsets 模块导入 BDay 类
    from pandas.tseries.offsets import BDay


    class TestSetitemDT64Values:
        # 定义测试方法 test_setitem_none_nan
        def test_setitem_none_nan(self):
            # 创建一个 Series 对象，包含从 "1/1/2000" 开始的日期序列，共 10 个元素
            series = Series(date_range("1/1/2000", periods=10))
            # 将第 3 个位置的元素设置为 None
            series[3] = None
            # 断言第 3 个位置的元素是否为 NaT（pandas 中表示缺失的时间戳）
            assert series[3] is NaT

            # 将第 3 到第 5 个位置的元素设置为 None
            series[3:5] = None
            # 断言第 4 个位置的元素是否为 NaT
            assert series[4] is NaT

            # 将第 5 个位置的元素设置为 np.nan（NaN，pandas 中表示缺失的数值）
            series[5] = np.nan
            # 断言第 5 个位置的元素是否为 NaT
            assert series[5] is NaT

            # 将第 5 到第 7 个位置的元素设置为 np.nan
            series[5:7] = np.nan
            # 断言第 6 个位置的元素是否为 NaT
            assert series[6] is NaT

        # 定义测试方法 test_setitem_multiindex_empty_slice
        def test_setitem_multiindex_empty_slice(self):
            # 创建一个 MultiIndex 对象，包含多个元组作为索引
            idx = MultiIndex.from_tuples([("a", 1), ("b", 2)])
            # 创建一个 Series 对象，包含值为 [1, 2]，使用上述 MultiIndex 作为索引
            result = Series([1, 2], index=idx)
            # 复制 result 对象作为预期结果
            expected = result.copy()
            # 在空的切片上设置值为 0
            result.loc[[]] = 0
            # 使用 pandas 测试工具函数验证 result 与 expected 是否相等
            tm.assert_series_equal(result, expected)

        # 定义测试方法 test_setitem_with_string_index
        def test_setitem_with_string_index(self):
            # 创建一个 Series 对象，包含值为 [1, 2, 3]，并指定字符串索引
            ser = Series([1, 2, 3], index=["Date", "b", "other"], dtype=object)
            # 将索引为 "Date" 的元素设置为当前日期
            ser["Date"] = date.today()
            # 断言索引为 "Date" 的元素是否为当天日期
            assert ser.Date == date.today()
            # 再次断言索引为 "Date" 的元素是否为当天日期
            assert ser["Date"] == date.today()

        # 定义测试方法 test_setitem_tuple_with_datetimetz_values
        def test_setitem_tuple_with_datetimetz_values(self):
            # 创建一个带有时区信息的日期序列 arr
            arr = date_range("2017", periods=4, tz="US/Eastern")
            # 创建一个包含元组索引的 Series 对象，使用 arr 作为值，index 作为索引
            index = [(0, 1), (0, 2), (0, 3), (0, 4)]
            result = Series(arr, index=index)
            # 复制 result 对象作为预期结果
            expected = result.copy()
            # 将索引为 (0, 1) 的元素设置为 np.nan
            result[(0, 1)] = np.nan
            # 将预期结果的第一行元素设置为 np.nan
            expected.iloc[0] = np.nan
            # 使用 pandas 测试工具函数验证 result 与 expected 是否相等
            tm.assert_series_equal(result, expected)

        # 标记参数化测试的方法，参数为不同的时区字符串
        @pytest.mark.parametrize("tz", ["US/Eastern", "UTC", "Asia/Tokyo"])
    # 定义一个测试方法，用于测试带有时区信息的Series对象中的setitem操作
    def test_setitem_with_tz(self, tz, indexer_sli):
        # 创建一个原始Series对象，包含从"2016-01-01"开始，每小时频率，共3个时间点的时间序列，带有指定时区
        orig = Series(date_range("2016-01-01", freq="h", periods=3, tz=tz))
        # 断言原始Series对象的数据类型为带有时区信息的datetime64类型
        assert orig.dtype == f"datetime64[ns, {tz}]"

        # 期望的Series对象，包含特定时间戳和时区信息，与原始Series对象的数据类型一致
        exp = Series(
            [
                Timestamp("2016-01-01 00:00", tz=tz),
                Timestamp("2011-01-01 00:00", tz=tz),
                Timestamp("2016-01-01 02:00", tz=tz),
            ],
            dtype=orig.dtype,
        )

        # scalar设置操作
        ser = orig.copy()
        # 对索引为1的位置设置一个带有时区信息的时间戳
        indexer_sli(ser)[1] = Timestamp("2011-01-01", tz=tz)
        # 断言设置后的Series对象与期望的结果相等
        tm.assert_series_equal(ser, exp)

        # vector设置操作
        vals = Series(
            [Timestamp("2011-01-01", tz=tz), Timestamp("2012-01-01", tz=tz)],
            index=[1, 2],
            dtype=orig.dtype,
        )
        # 断言vals对象的数据类型为带有时区信息的datetime64类型
        assert vals.dtype == f"datetime64[ns, {tz}]"

        # 更新期望的Series对象，以包含新的时间戳和时区信息
        exp = Series(
            [
                Timestamp("2016-01-01 00:00", tz=tz),
                Timestamp("2011-01-01 00:00", tz=tz),
                Timestamp("2012-01-01 00:00", tz=tz),
            ],
            dtype=orig.dtype,
        )

        # 对原始Series对象进行复制
        ser = orig.copy()
        # 对索引为[1, 2]的位置设置新的时间戳和时区信息
        indexer_sli(ser)[[1, 2]] = vals
        # 断言设置后的Series对象与更新后的期望结果相等
        tm.assert_series_equal(ser, exp)

    # 定义一个测试方法，用于测试带有夏令时边界的时区信息的Series对象中的setitem操作
    def test_setitem_with_tz_dst(self, indexer_sli):
        # 定义时区为"US/Eastern"
        tz = "US/Eastern"
        # 创建一个原始Series对象，从"2016-11-06"开始，每小时频率，共3个时间点的时间序列，带有指定时区
        orig = Series(date_range("2016-11-06", freq="h", periods=3, tz=tz))
        # 断言原始Series对象的数据类型为带有时区信息的datetime64类型
        assert orig.dtype == f"datetime64[ns, {tz}]"

        # 期望的Series对象，包含特定时间戳、夏令时边界和时区信息，与原始Series对象的数据类型一致
        exp = Series(
            [
                Timestamp("2016-11-06 00:00-04:00", tz=tz),
                Timestamp("2011-01-01 00:00-05:00", tz=tz),
                Timestamp("2016-11-06 01:00-05:00", tz=tz),
            ],
            dtype=orig.dtype,
        )

        # scalar设置操作
        ser = orig.copy()
        # 对索引为1的位置设置一个带有时区信息的时间戳
        indexer_sli(ser)[1] = Timestamp("2011-01-01", tz=tz)
        # 断言设置后的Series对象与期望的结果相等
        tm.assert_series_equal(ser, exp)

        # vector设置操作
        vals = Series(
            [Timestamp("2011-01-01", tz=tz), Timestamp("2012-01-01", tz=tz)],
            index=[1, 2],
            dtype=orig.dtype,
        )
        # 断言vals对象的数据类型为带有时区信息的datetime64类型
        assert vals.dtype == f"datetime64[ns, {tz}]"

        # 更新期望的Series对象，以包含新的时间戳和时区信息
        exp = Series(
            [
                Timestamp("2016-11-06 00:00", tz=tz),
                Timestamp("2011-01-01 00:00", tz=tz),
                Timestamp("2012-01-01 00:00", tz=tz),
            ],
            dtype=orig.dtype,
        )

        # 对原始Series对象进行复制
        ser = orig.copy()
        # 对索引为[1, 2]的位置设置新的时间戳和时区信息
        indexer_sli(ser)[[1, 2]] = vals
        # 断言设置后的Series对象与更新后的期望结果相等
        tm.assert_series_equal(ser, exp)
    def test_object_series_setitem_dt64array_exact_match(self):
        # 定义测试函数，验证在设置日期时间数组时不被 numpy 转换为整数
        # 参考：https://github.com/numpy/numpy/issues/12550

        # 创建一个 Series 对象，包含一个空值，数据类型为 object
        ser = Series({"X": np.nan}, dtype=object)

        # 创建一个索引器列表，包含一个布尔值 True
        indexer = [True]

        # 创建一个 numpy 数组，包含一个日期时间值，数据类型为 "M8[ns]"
        value = np.array([4], dtype="M8[ns]")

        # 使用 iloc 方法根据索引器设置 ser 对象的值为上述日期时间数组
        ser.iloc[indexer] = value

        # 创建一个期望的 Series 对象，包含设置的日期时间值，索引为 ["X"]，数据类型为 object
        expected = Series([value[0]], index=["X"], dtype=object)

        # 断言：验证 expected 中的所有值是否都是 np.datetime64 类型
        assert all(isinstance(x, np.datetime64) for x in expected.values)

        # 使用 assert_series_equal 方法验证 ser 和 expected 是否相等
        tm.assert_series_equal(ser, expected)
class TestSetitemScalarIndexer:
    def test_setitem_negative_out_of_bounds(self):
        # 从版本 3.0 开始，整数键被视为标签，因此这里会进行扩展操作
        ser = Series(["a"] * 10, index=["a"] * 10)
        # 在索引位置 -11 处设置值为 "foo"
        ser[-11] = "foo"
        # 期望的结果是在索引位置为 -11 处插入 "foo"，并扩展 Series
        exp = Series(["a"] * 10 + ["foo"], index=["a"] * 10 + [-11])
        tm.assert_series_equal(ser, exp)

    @pytest.mark.parametrize("indexer", [tm.loc, tm.at])
    @pytest.mark.parametrize("ser_index", [0, 1])
    def test_setitem_series_object_dtype(self, indexer, ser_index):
        # GH#38303
        # 创建一个对象类型的 Series，元素为 [0, 0]
        ser = Series([0, 0], dtype="object")
        # 使用 indexer 获取索引器对象
        idxr = indexer(ser)
        # 在索引位置 0 处设置值为包含 [42] 的 Series，并指定索引为 [ser_index]
        idxr[0] = Series([42], index=[ser_index])
        # 期望的结果是一个对象类型的 Series，包含 [Series([42], index=[ser_index]), 0]
        expected = Series([Series([42], index=[ser_index]), 0], dtype="object")
        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize("index, exp_value", [(0, 42), (1, np.nan)])
    def test_setitem_series(self, index, exp_value):
        # GH#38303
        # 创建一个 Series，元素为 [0, 0]
        ser = Series([0, 0])
        # 使用 loc 方法在索引位置 0 处设置值为包含 [42] 的 Series，并指定索引为 [index]
        ser.loc[0] = Series([42], index=[index])
        # 期望的结果是一个 Series，元素为 [exp_value, 0]
        expected = Series([exp_value, 0])
        tm.assert_series_equal(ser, expected)


class TestSetitemSlices:
    def test_setitem_slice_float_raises(self, datetime_series):
        # 如果在 DatetimeIndex 上使用浮点类型的索引器进行切片操作，会抛出 TypeError
        msg = (
            "cannot do slice indexing on DatetimeIndex with these indexers "
            r"\[{key}\] of type float"
        )
        with pytest.raises(TypeError, match=msg.format(key=r"4\.0")):
            datetime_series[4.0:10.0] = 0

        with pytest.raises(TypeError, match=msg.format(key=r"4\.5")):
            datetime_series[4.5:10.0] = 0

    def test_setitem_slice(self):
        # 创建一个 Series，包含从 0 到 9 的整数，使用相同的索引
        ser = Series(range(10), index=list(range(10)))
        # 将索引从 -12 开始的所有元素设置为 0
        ser[-12:] = 0
        assert (ser == 0).all()

        # 将索引从 -12 开始的所有元素设置为 5
        ser[:-12] = 5
        assert (ser == 0).all()

    def test_setitem_slice_integers(self):
        # 创建一个 Series，包含从 2 到 16 的偶数，使用相应的索引
        ser = Series(
            np.random.default_rng(2).standard_normal(8),
            index=[2, 4, 6, 8, 10, 12, 14, 16],
        )

        # 将索引从开头到 4 的元素设置为 0
        ser[:4] = 0
        assert (ser[:4] == 0).all()
        # 确保索引从 4 开始的元素没有被设置为 0
        assert not (ser[4:] == 0).any()

    def test_setitem_slicestep(self):
        # 在编写测试时捕获到此错误
        series = Series(
            np.arange(20, dtype=np.float64), index=np.arange(20, dtype=np.int64)
        )

        # 将索引以步长为 2 的元素设置为 0
        series[::2] = 0
        assert (series[::2] == 0).all()

    def test_setitem_multiindex_slice(self, indexer_sli):
        # GH 8856
        # 创建一个多级索引，包含两个级别的组合
        mi = MultiIndex.from_product(([0, 1], list("abcde")))
        # 创建一个 Series，使用该多级索引，元素从 0 到 9
        result = Series(np.arange(10, dtype=np.int64), mi)
        # 使用 indexer_sli 获取索引器，并将索引以步长为 4 的元素设置为 100
        indexer_sli(result)[::4] = 100
        # 期望的结果是元素为 [100, 1, 2, 3, 100, 5, 6, 7, 100, 9] 的 Series
        expected = Series([100, 1, 2, 3, 100, 5, 6, 7, 100, 9], mi)
        tm.assert_series_equal(result, expected)


class TestSetitemBooleanMask:
    # 这里是 TestSetitemBooleanMask 类，后续的方法需要继续编写测试用例
    # 测试设置使用布尔掩码的赋值操作
    def test_setitem_mask_cast(self):
        # GH#2746
        # 需要进行类型提升
        ser = Series([1, 2], index=[1, 2], dtype="int64")
        # 使用布尔掩码选择部分数据并赋值新的 Series 对象
        ser[[True, False]] = Series([0], index=[1], dtype="int64")
        expected = Series([0, 2], index=[1, 2], dtype="int64")

        tm.assert_series_equal(ser, expected)

    # 测试使用布尔掩码时的赋值操作和类型提升
    def test_setitem_mask_align_and_promote(self):
        # GH#8387: 测试改变类型不会破坏对齐
        ts = Series(
            np.random.default_rng(2).standard_normal(100), index=np.arange(100, 0, -1)
        ).round(5)
        mask = ts > 0
        left = ts.copy()
        # 使用布尔掩码和类型转换操作
        right = ts[mask].copy().map(str)
        with tm.assert_produces_warning(
            FutureWarning, match="item of incompatible dtype"
        ):
            left[mask] = right
        expected = ts.map(lambda t: str(t) if t > 0 else t)
        tm.assert_series_equal(left, expected)

    # 测试使用布尔掩码时的赋值操作和字符串类型提升
    def test_setitem_mask_promote_strs(self):
        ser = Series([0, 1, 2, 0])
        mask = ser > 0
        ser2 = ser[mask].map(str)
        with tm.assert_produces_warning(
            FutureWarning, match="item of incompatible dtype"
        ):
            ser[mask] = ser2

        expected = Series([0, "1", "2", 0])
        tm.assert_series_equal(ser, expected)

    # 测试使用布尔掩码时的赋值操作和类型提升
    def test_setitem_mask_promote(self):
        ser = Series([0, "foo", "bar", 0])
        mask = Series([False, True, True, False])
        ser2 = ser[mask]
        ser[mask] = ser2

        expected = Series([0, "foo", "bar", 0])
        tm.assert_series_equal(ser, expected)

    # 测试使用布尔掩码进行赋值操作
    def test_setitem_boolean(self, string_series):
        mask = string_series > string_series.median()

        # 类似索引的 Series
        result = string_series.copy()
        result[mask] = string_series * 2
        expected = string_series * 2
        tm.assert_series_equal(result[mask], expected[mask])

        # 需要对齐
        result = string_series.copy()
        result[mask] = (string_series * 2)[0:5]
        expected = (string_series * 2)[0:5].reindex_like(string_series)
        expected[-mask] = string_series[mask]
        tm.assert_series_equal(result[mask], expected[mask])

    # 测试使用布尔掩码进行赋值操作的边界情况
    def test_setitem_boolean_corner(self, datetime_series):
        ts = datetime_series
        mask_shifted = ts.shift(1, freq=BDay()) > ts.median()

        msg = (
            r"Unalignable boolean Series provided as indexer \(index of "
            r"the boolean Series and of the indexed object do not match"
        )
        with pytest.raises(IndexingError, match=msg):
            ts[mask_shifted] = 1

        with pytest.raises(IndexingError, match=msg):
            ts.loc[mask_shifted] = 1

    # 测试使用布尔掩码进行赋值操作时的不同顺序
    def test_setitem_boolean_different_order(self, string_series):
        ordered = string_series.sort_values()

        copy = string_series.copy()
        copy[ordered > 0] = 0

        expected = string_series.copy()
        expected[expected > 0] = 0

        tm.assert_series_equal(copy, expected)
    @pytest.mark.parametrize("func", [list, np.array, Series])
    # 使用 pytest 的参数化功能，分别测试 list、np.array、Series 这三种类型的函数
    def test_setitem_boolean_python_list(self, func):
        # 测试用例名称: test_setitem_boolean_python_list
        # GH19406

        # 创建一个包含 None, "b", None 的 Series 对象
        ser = Series([None, "b", None])

        # 使用传入的 func 函数创建一个布尔掩码 mask
        mask = func([True, False, True])

        # 根据 mask 设定 ser 对象中对应位置的值为 ["a", "c"]
        ser[mask] = ["a", "c"]

        # 创建期望的 Series 对象，包含 ["a", "b", "c"]
        expected = Series(["a", "b", "c"])

        # 断言 ser 与期望值 expected 是否相等
        tm.assert_series_equal(ser, expected)

    def test_setitem_boolean_nullable_int_types(self, any_numeric_ea_dtype):
        # 测试用例名称: test_setitem_boolean_nullable_int_types
        # GH: 26468

        # 创建一个包含数值的 Series 对象，数据类型为 any_numeric_ea_dtype
        ser = Series([5, 6, 7, 8], dtype=any_numeric_ea_dtype)

        # 将 ser 中大于 6 的位置的值设定为另一个 Series 对象的值
        ser[ser > 6] = Series(range(4), dtype=any_numeric_ea_dtype)

        # 创建期望的 Series 对象
        expected = Series([5, 6, 2, 3], dtype=any_numeric_ea_dtype)

        # 断言 ser 与期望值 expected 是否相等
        tm.assert_series_equal(ser, expected)

        # 重新设置 ser 对象
        ser = Series([5, 6, 7, 8], dtype=any_numeric_ea_dtype)

        # 将使用 loc 方法进行索引的方式设定 ser 中大于 6 的位置的值
        ser.loc[ser > 6] = Series(range(4), dtype=any_numeric_ea_dtype)

        # 断言 ser 与期望值 expected 是否相等
        tm.assert_series_equal(ser, expected)

        # 重新设置 ser 对象
        ser = Series([5, 6, 7, 8], dtype=any_numeric_ea_dtype)

        # 创建另一个 Series 对象 loc_ser，包含 range(4) 的值
        loc_ser = Series(range(4), dtype=any_numeric_ea_dtype)

        # 使用 loc 方法设定 ser 中大于 6 的位置的值
        ser.loc[ser > 6] = loc_ser.loc[loc_ser > 1]

        # 断言 ser 与期望值 expected 是否相等
        tm.assert_series_equal(ser, expected)

    def test_setitem_with_bool_mask_and_values_matching_n_trues_in_length(self):
        # 测试用例名称: test_setitem_with_bool_mask_and_values_matching_n_trues_in_length
        # GH#30567

        # 创建一个包含 None 的 Series 对象，长度为 10
        ser = Series([None] * 10)

        # 创建一个布尔掩码 mask
        mask = [False] * 3 + [True] * 5 + [False] * 2

        # 使用 mask 设定 ser 对象中对应位置的值为 range(5)
        ser[mask] = range(5)

        # 将结果保存在 result 中
        result = ser

        # 创建期望的 Series 对象
        expected = Series([None] * 3 + list(range(5)) + [None] * 2, dtype=object)

        # 断言 result 与期望值 expected 是否相等
        tm.assert_series_equal(result, expected)

    def test_setitem_nan_with_bool(self):
        # 测试用例名称: test_setitem_nan_with_bool
        # GH 13034

        # 创建一个包含 True, False, True 的 Series 对象
        result = Series([True, False, True])

        # 使用 assert_produces_warning 断言产生 FutureWarning，并匹配 "item of incompatible dtype"
        with tm.assert_produces_warning(
            FutureWarning, match="item of incompatible dtype"
        ):
            # 将 result 的第一个元素设为 np.nan
            result[0] = np.nan

        # 创建期望的 Series 对象
        expected = Series([np.nan, False, True], dtype=object)

        # 断言 result 与期望值 expected 是否相等
        tm.assert_series_equal(result, expected)

    def test_setitem_mask_smallint_upcast(self):
        # 测试用例名称: test_setitem_mask_smallint_upcast

        # 创建一个包含 [1, 2, 3] 的 Series 对象，数据类型为 "int8"
        orig = Series([1, 2, 3], dtype="int8")

        # 创建一个包含 [999, 1000, 1001] 的 np.array 对象，数据类型为 np.int64
        alt = np.array([999, 1000, 1001], dtype=np.int64)

        # 创建一个布尔掩码 mask
        mask = np.array([True, False, True])

        # 复制 orig 对象到 ser 对象
        ser = orig.copy()

        # 使用 assert_produces_warning 断言产生 FutureWarning，并匹配 "item of incompatible dtype"
        with tm.assert_produces_warning(
            FutureWarning, match="item of incompatible dtype"
        ):
            # 使用 mask 设定 ser 对象中对应位置的值为 Series(alt)
            ser[mask] = Series(alt)

        # 创建期望的 Series 对象
        expected = Series([999, 2, 1001])

        # 断言 ser 与期望值 expected 是否相等
        tm.assert_series_equal(ser, expected)

        # 复制 orig 对象到 ser2 对象
        ser2 = orig.copy()

        # 使用 assert_produces_warning 断言产生 FutureWarning，并匹配 "item of incompatible dtype"
        with tm.assert_produces_warning(
            FutureWarning, match="item of incompatible dtype"
        ):
            # 使用 mask 设定 ser2 对象中对应位置的值为 Series(alt)，并在原地修改
            ser2.mask(mask, alt, inplace=True)

        # 断言 ser2 与期望值 expected 是否相等
        tm.assert_series_equal(ser2, expected)

        # 复制 orig 对象到 ser3 对象
        ser3 = orig.copy()

        # 使用 where 方法根据 mask 条件设置 ser3 对象的值为 Series(alt)
        res = ser3.where(~mask, Series(alt))

        # 断言 res 与期望值 expected 是否相等
        tm.assert_series_equal(res, expected)
    def test_setitem_mask_smallint_no_upcast(self):
        # 定义一个测试函数，用于测试在不需要向上转型的情况下设置掩码项
        # 类似于 test_setitem_mask_smallint_upcast，但是我们不能容纳 'alt'，
        # 而可以容纳 alt[mask] 而无需转型

        # 创建原始 Series，数据类型为 uint8
        orig = Series([1, 2, 3], dtype="uint8")
        # 创建备选 Series，数据类型为 int64
        alt = Series([245, 1000, 246], dtype=np.int64)

        # 创建布尔掩码数组
        mask = np.array([True, False, True])

        # 复制原始 Series
        ser = orig.copy()
        # 使用 alt 来设置 ser 的掩码项
        ser[mask] = alt
        # 期望的结果 Series，数据类型为 uint8
        expected = Series([245, 2, 246], dtype="uint8")
        # 检查 ser 是否与期望结果相等
        tm.assert_series_equal(ser, expected)

        # 复制原始 Series
        ser2 = orig.copy()
        # 使用 alt 替换掩码项，直接修改 ser2 本身
        ser2.mask(mask, alt, inplace=True)
        # 检查 ser2 是否与期望结果相等
        tm.assert_series_equal(ser2, expected)

        # TODO: ser.where(~mask, alt) unnecessarily upcasts to int64
        # 创建原始 Series 的副本
        ser3 = orig.copy()
        # 使用 alt 替换掩码项为 False 的部分
        res = ser3.where(~mask, alt)
        # 检查 res 是否与期望结果相等，不检查数据类型是否相同
        tm.assert_series_equal(res, expected, check_dtype=False)
class TestSetitemViewCopySemantics:
    def test_setitem_invalidates_datetime_index_freq(self):
        # GH#24096 altering a datetime64tz Series inplace invalidates the
        #  `freq` attribute on the underlying DatetimeIndex

        # 创建一个带时区的日期范围
        dti = date_range("20130101", periods=3, tz="US/Eastern")
        # 获取第二个时间戳
        ts = dti[1]
        # 使用日期范围创建一个 Series 对象
        ser = Series(dti)
        # 断言新创建的 Series 不是原始日期范围对象的视图
        assert ser._values is not dti
        # 断言 Series 对象底层的 ndarray 的 base 是日期范围对象的 ndarray 的 base
        assert ser._values._ndarray.base is dti._data._ndarray.base
        # 断言日期范围的频率为 "D"
        assert dti.freq == "D"
        # 在 Series 的第二个位置设置 NaT（Not a Time），这里是原地修改
        ser.iloc[1] = NaT
        # 断言 Series 对象的频率现在是 None
        assert ser._values.freq is None

        # 检查日期范围对象是否未被原地修改
        assert ser._values is not dti
        assert ser._values._ndarray.base is not dti._data._ndarray.base
        assert dti[1] == ts
        assert dti.freq == "D"

    def test_dt64tz_setitem_does_not_mutate_dti(self):
        # GH#21907, GH#24096
        # 创建一个带时区的日期范围
        dti = date_range("2016-01-01", periods=10, tz="US/Pacific")
        # 获取第一个时间戳
        ts = dti[0]
        # 使用日期范围创建一个 Series 对象
        ser = Series(dti)
        # 断言新创建的 Series 不是原始日期范围对象的视图
        assert ser._values is not dti
        # 断言 Series 对象底层的 ndarray 的 base 是日期范围对象的 ndarray 的 base
        assert ser._values._ndarray.base is dti._data._ndarray.base
        # 断言 Series 对象管理的 blocks[0] 的值的 ndarray 的 base 是日期范围对象的 ndarray 的 base
        assert ser._mgr.blocks[0].values._ndarray.base is dti._data._ndarray.base

        # 断言 Series 对象管理的 blocks[0] 的值不是日期范围对象
        assert ser._mgr.blocks[0].values is not dti

        # 在每隔三个位置设置 NaT，这里是原地修改
        ser[::3] = NaT
        # 断言 Series 的第一个位置现在是 NaT
        assert ser[0] is NaT
        # 断言日期范围的第一个位置仍然等于 ts
        assert dti[0] == ts


class TestSetitemCallable:
    def test_setitem_callable_key(self):
        # GH#12533
        # 创建一个 Series 对象，指定索引
        ser = Series([1, 2, 3, 4], index=list("ABCD"))
        # 使用 lambda 函数来设置索引为 "A" 的值为 -1
        ser[lambda x: "A"] = -1

        # 创建一个预期的 Series 对象
        expected = Series([-1, 2, 3, 4], index=list("ABCD"))
        # 断言两个 Series 对象是否相等
        tm.assert_series_equal(ser, expected)

    def test_setitem_callable_other(self):
        # GH#13299
        # 创建一个 lambda 函数来增加值
        inc = lambda x: x + 1

        # 创建一个对象类型的 Series 对象，避免设置时的上溢
        ser = Series([1, 2, -1, 4], dtype=object)
        # 使用 lambda 函数来设置小于 0 的值增加 1
        ser[ser < 0] = inc

        # 创建一个预期的 Series 对象
        expected = Series([1, 2, inc, 4])
        # 断言两个 Series 对象是否相等
        tm.assert_series_equal(ser, expected)


class TestSetitemWithExpansion:
    def test_setitem_empty_series(self):
        # GH#10193, GH#51363 changed in 3.0 to not do inference in Index.insert
        # 创建一个时间戳
        key = Timestamp("2012-01-01")
        # 创建一个空的对象类型的 Series 对象
        series = Series(dtype=object)
        # 设置给定键的值为 47
        series[key] = 47
        # 创建一个预期的 Series 对象
        expected = Series(47, Index([key], dtype=object))
        # 断言两个 Series 对象是否相等
        tm.assert_series_equal(series, expected)

    def test_setitem_empty_series_datetimeindex_preserves_freq(self):
        # GH#33573 our index should retain its freq
        # 创建一个空的 DatetimeIndex 对象，指定频率为 "D"，数据类型为 "M8[ns]"
        dti = DatetimeIndex([], freq="D", dtype="M8[ns]")
        # 创建一个空的对象类型的 Series 对象，使用上述 DatetimeIndex 作为索引
        series = Series([], index=dti, dtype=object)
        # 创建一个时间戳
        key = Timestamp("2012-01-01")
        # 设置给定键的值为 47
        series[key] = 47
        # 创建一个预期的 Series 对象
        expected = Series(47, DatetimeIndex([key], freq="D").as_unit("ns"))
        # 断言两个 Series 对象是否相等
        tm.assert_series_equal(series, expected)
        # 断言 Series 对象的索引频率与预期的索引频率相同
        assert series.index.freq == expected.index.freq
    def test_setitem_empty_series_timestamp_preserves_dtype(self):
        # 定义一个 Timestamp 对象，表示时间戳 1412526600000000000
        timestamp = Timestamp(1412526600000000000)
        # 创建一个 Series 对象，包含一个 timestamp 元素，索引为 "timestamp"，数据类型为 object
        series = Series([timestamp], index=["timestamp"], dtype=object)
        # 保存预期的 series["timestamp"] 值
        expected = series["timestamp"]

        # 创建一个空的 Series 对象，数据类型为 object
        series = Series([], dtype=object)
        # 向空 Series 对象中添加元素，键为 "anything"，值为 300.0
        series["anything"] = 300.0
        # 向空 Series 对象中添加元素，键为 "timestamp"，值为之前定义的 timestamp 对象
        series["timestamp"] = timestamp
        # 获取 series["timestamp"] 的值
        result = series["timestamp"]
        # 断言结果与预期相同
        assert result == expected

    @pytest.mark.parametrize(
        "td",
        [
            Timedelta("9 days"),  # 创建一个 Timedelta 对象，表示 9 天
            Timedelta("9 days").to_timedelta64(),  # 将 Timedelta 转换为 Timedelta64 类型
            Timedelta("9 days").to_pytimedelta(),  # 将 Timedelta 转换为 Python 的 timedelta 对象
        ],
    )
    def test_append_timedelta_does_not_cast(self, td, using_infer_string, request):
        # GH#22717 插入 Timedelta 对象不应该转换为 int64 类型
        if using_infer_string and not isinstance(td, Timedelta):
            # TODO: GH#56010 待解决的问题，推断类型为字符串时的处理
            request.applymarker(pytest.mark.xfail(reason="inferred as string"))

        # 创建一个预期的 Series 对象，包含两个元素：字符串 "x" 和前面定义的 td 对象
        expected = Series(["x", td], index=[0, "td"], dtype=object)

        # 创建一个包含字符串 "x" 的 Series 对象
        ser = Series(["x"])
        # 向 Series 对象中添加一个键为 "td"，值为 td 的元素
        ser["td"] = td
        # 断言 ser 与 expected 的内容相同
        tm.assert_series_equal(ser, expected)
        # 断言 ser["td"] 的类型是 Timedelta
        assert isinstance(ser["td"], Timedelta)

        # 创建一个包含字符串 "x" 的 Series 对象
        ser = Series(["x"])
        # 使用 loc 方法向 Series 对象中添加一个键为 "td"，值为 Timedelta("9 days") 的元素
        ser.loc["td"] = Timedelta("9 days")
        # 断言 ser 与 expected 的内容相同
        tm.assert_series_equal(ser, expected)
        # 断言 ser["td"] 的类型是 Timedelta
        assert isinstance(ser["td"], Timedelta)

    def test_setitem_with_expansion_type_promotion(self):
        # GH#12599 设置元素时支持类型提升
        ser = Series(dtype=object)
        # 设置 ser 的 "a" 键为 Timestamp("2016-01-01") 对象
        ser["a"] = Timestamp("2016-01-01")
        # 设置 ser 的 "b" 键为 3.0
        ser["b"] = 3.0
        # 设置 ser 的 "c" 键为字符串 "foo"
        ser["c"] = "foo"
        # 创建一个预期的 Series 对象，包含三个元素，分别是 Timestamp 对象、浮点数 3.0 和字符串 "foo"，索引为 ["a", "b", "c"]
        expected = Series([Timestamp("2016-01-01"), 3.0, "foo"], index=["a", "b", "c"])
        # 断言 ser 与 expected 的内容相同
        tm.assert_series_equal(ser, expected)

    def test_setitem_not_contained(self, string_series):
        # 设置一个不包含的键
        ser = string_series.copy()
        # 断言 "foobar" 不在 ser 的索引中
        assert "foobar" not in ser.index
        # 向 ser 中添加一个键为 "foobar"，值为 1 的元素
        ser["foobar"] = 1

        # 创建一个 Series 对象，包含一个元素 [1]，索引为 ["foobar"]，名称为 "series"
        app = Series([1], index=["foobar"], name="series")
        # 将 string_series 与 app 连接起来，得到预期的 Series 对象
        expected = concat([string_series, app])
        # 断言 ser 与 expected 的内容相同
        tm.assert_series_equal(ser, expected)

    def test_setitem_keep_precision(self, any_numeric_ea_dtype):
        # GH#32346 设置元素时保持精度
        ser = Series([1, 2], dtype=any_numeric_ea_dtype)
        # 将 ser 的第 2 个元素设置为 10
        ser[2] = 10
        # 创建一个预期的 Series 对象，包含三个元素 [1, 2, 10]，数据类型为 any_numeric_ea_dtype
        expected = Series([1, 2, 10], dtype=any_numeric_ea_dtype)
        # 断言 ser 与 expected 的内容相同
        tm.assert_series_equal(ser, expected)
    @pytest.mark.parametrize(
        "na, target_na, dtype, target_dtype, indexer, warn",
        [
            # 参数化测试数据，每个元组包含不同的输入值和期望输出
            (NA, NA, "Int64", "Int64", 1, None),  # 输入NA，期望NA，整数类型
            (NA, NA, "Int64", "Int64", 2, None),  # 输入NA，期望NA，整数类型
            (NA, np.nan, "int64", "float64", 1, None),  # 输入NA，期望NaN，整数到浮点数类型转换
            (NA, np.nan, "int64", "float64", 2, None),  # 输入NA，期望NaN，整数到浮点数类型转换
            (NaT, NaT, "int64", "object", 1, FutureWarning),  # 输入NaT，期望NaT，整数到对象类型转换，显示FutureWarning
            (NaT, NaT, "int64", "object", 2, None),  # 输入NaT，期望NaT，整数到对象类型转换
            (np.nan, NA, "Int64", "Int64", 1, None),  # 输入NaN，期望NA，整数类型
            (np.nan, NA, "Int64", "Int64", 2, None),  # 输入NaN，期望NA，整数类型
            (np.nan, NA, "Float64", "Float64", 1, None),  # 输入NaN，期望NA，浮点数类型
            (np.nan, NA, "Float64", "Float64", 2, None),  # 输入NaN，期望NA，浮点数类型
            (np.nan, np.nan, "int64", "float64", 1, None),  # 输入NaN，期望NaN，整数到浮点数类型转换
            (np.nan, np.nan, "int64", "float64", 2, None),  # 输入NaN，期望NaN，整数到浮点数类型转换
        ],
    )
    def test_setitem_enlarge_with_na(
        self, na, target_na, dtype, target_dtype, indexer, warn
    ):
        # GH#32346
        # 创建一个Series对象，初始化值为[1, 2]，指定数据类型为dtype
        ser = Series([1, 2], dtype=dtype)
        # 使用tm.assert_produces_warning检查是否会产生警告，期望的警告类型为warn，匹配字符串"incompatible dtype"
        with tm.assert_produces_warning(warn, match="incompatible dtype"):
            # 设置ser对象的索引为indexer的位置为na
            ser[indexer] = na
        # 根据indexer的值确定期望的值列表
        expected_values = [1, target_na] if indexer == 1 else [1, 2, target_na]
        # 创建一个期望的Series对象，值为expected_values，数据类型为target_dtype
        expected = Series(expected_values, dtype=target_dtype)
        # 使用tm.assert_series_equal检查ser和expected是否相等
        tm.assert_series_equal(ser, expected)

    def test_setitem_enlargement_object_none(self, nulls_fixture, using_infer_string):
        # GH#48665
        # 创建一个Series对象，初始值为["a", "b"]
        ser = Series(["a", "b"])
        # 设置ser的索引为3的位置为nulls_fixture
        ser[3] = nulls_fixture
        # 确定期望的数据类型
        dtype = (
            "string[pyarrow_numpy]"  # 如果使用infer_string并且nulls_fixture不是Decimal类型，则数据类型为"string[pyarrow_numpy]"
            if using_infer_string and not isinstance(nulls_fixture, Decimal)
            else object  # 否则数据类型为object
        )
        # 创建一个期望的Series对象，值为["a", "b", nulls_fixture]，索引为[0, 1, 3]，数据类型为dtype
        expected = Series(["a", "b", nulls_fixture], index=[0, 1, 3], dtype=dtype)
        # 使用tm.assert_series_equal检查ser和expected是否相等
        tm.assert_series_equal(ser, expected)
        # 如果using_infer_string为True，则检查ser的索引为3的位置是否为np.nan
        if using_infer_string:
            ser[3] is np.nan
        else:
            # 否则检查ser的索引为3的位置是否为nulls_fixture
            assert ser[3] is nulls_fixture
# 测试用例：向只读后备数据中的标量设置项
def test_setitem_scalar_into_readonly_backing_data():
    # GH#14359: 测试无法修改只读缓冲区的情况

    # 创建一个包含5个元素的全零数组
    array = np.zeros(5)
    # 将数组设为不可写，即使其内容为只读
    array.flags.writeable = False  # make the array immutable
    # 从不可写的数组创建一个 Series 对象
    series = Series(array, copy=False)

    # 遍历 series 的索引
    for n in series.index:
        msg = "assignment destination is read-only"
        # 确保在尝试修改时引发 ValueError 异常，并且异常信息匹配预期消息
        with pytest.raises(ValueError, match=msg):
            series[n] = 1

        # 断言原始数组的相应位置仍然为 0
        assert array[n] == 0


# 测试用例：向只读后备数据中的切片设置项
def test_setitem_slice_into_readonly_backing_data():
    # GH#14359: 测试无法修改只读缓冲区的情况

    # 创建一个包含5个元素的全零数组
    array = np.zeros(5)
    # 将数组设为不可写，即使其内容为只读
    array.flags.writeable = False  # make the array immutable
    # 从不可写的数组创建一个 Series 对象
    series = Series(array, copy=False)

    # 确保在尝试修改切片时引发 ValueError 异常，并且异常信息匹配预期消息
    msg = "assignment destination is read-only"
    with pytest.raises(ValueError, match=msg):
        series[1:3] = 1

    # 断言原始数组的所有元素仍然为 0
    assert not array.any()


# 测试用例：向分类数据中的设置项进行赋值操作
def test_setitem_categorical_assigning_ops():
    # 创建一个包含两个重复元素的分类 Series 对象
    orig = Series(Categorical(["b", "b"], categories=["a", "b"]))
    ser = orig.copy()

    # 将整个 Series 赋值为 "a"
    ser[:] = "a"
    exp = Series(Categorical(["a", "a"], categories=["a", "b"]))
    # 确保操作后的结果与期望结果相等
    tm.assert_series_equal(ser, exp)

    # 将索引为1的元素赋值为 "a"
    ser = orig.copy()
    ser[1] = "a"
    exp = Series(Categorical(["b", "a"], categories=["a", "b"]))
    tm.assert_series_equal(ser, exp)

    # 将索引大于0的元素赋值为 "a"
    ser = orig.copy()
    ser[ser.index > 0] = "a"
    tm.assert_series_equal(ser, exp)

    # 使用布尔数组选择器，将指定位置的元素赋值为 "a"
    ser = orig.copy()
    ser[[False, True]] = "a"
    tm.assert_series_equal(ser, exp)

    # 修改 Series 对象的索引，然后将 "y" 处的元素赋值为 "a"
    ser = orig.copy()
    ser.index = ["x", "y"]
    ser["y"] = "a"
    exp = Series(Categorical(["b", "a"], categories=["a", "b"]), index=["x", "y"])
    tm.assert_series_equal(ser, exp)


# 测试用例：向分类数据中设置 np.nan
def test_setitem_nan_into_categorical():
    # 确保可以将某些位置设置为 np.nan
    ser = Series(Categorical([1, 2, 3]))
    exp = Series(Categorical([1, np.nan, 3], categories=[1, 2, 3]))
    ser[1] = np.nan
    tm.assert_series_equal(ser, exp)


# 测试类：测试在布尔 Series 中将非布尔值赋值给布尔值
class TestSetitemCasting:
    @pytest.mark.parametrize("unique", [True, False])
    @pytest.mark.parametrize("val", [3, 3.0, "3"], ids=type)
    def test_setitem_non_bool_into_bool(self, val, indexer_sli, unique):
        # 不应将类似于 3 的值强制转换为布尔值
        ser = Series([True, False])
        if not unique:
            ser.index = [1, 1]

        # 确保在设置非布尔值时会产生 FutureWarning 警告，并且警告消息匹配预期
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            indexer_sli(ser)[1] = val
        # 断言被设置后的值的类型与设置前的值类型相同
        assert type(ser.iloc[1]) == type(val)

        # 根据情况设置预期的 Series 对象，并确保与实际结果相等
        expected = Series([True, val], dtype=object, index=ser.index)
        if not unique and indexer_sli is not tm.iloc:
            expected = Series([val, val], dtype=object, index=[1, 1])
        tm.assert_series_equal(ser, expected)
    # 定义一个测试方法，用于测试将布尔数组设置到包含布尔值的 pandas Series 中的情况
    def test_setitem_boolean_array_into_npbool(self):
        # GH#45462：参考 GitHub issue #45462
        # 创建一个包含 True, False, True 的 Series 对象
        ser = Series([True, False, True])
        # 获取该 Series 对象的内部值数组
        values = ser._values
        # 创建一个包含 True, False, None 的 numpy 数组
        arr = array([True, False, None])

        # 将 arr 的前两个元素设置到 ser 的前两个位置，由于没有缺失值（None），可以直接就地设置
        ser[:2] = arr[:2]  # no NAs -> can set inplace
        # 断言设置后的 ser 的内部值数组仍然是之前的 values，即没有重新分配内存
        assert ser._values is values

        # 使用 assert_produces_warning 上下文管理器断言在设置 ser 的第二个位置及之后的元素时会产生 FutureWarning 警告
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            # 将 arr 的第二个位置及之后的元素设置到 ser 中，由于存在 NA（None），会强制转换为布尔类型
            ser[1:] = arr[1:]  # has an NA -> cast to boolean dtype
        # 创建一个期望的 Series 对象，内容与 arr 相同
        expected = Series(arr)
        # 断言设置后的 ser 与期望的结果 expected 相等
        tm.assert_series_equal(ser, expected)
    """
    检查几种方法是否等效于 `obj[key] = val` 的情况

    我们假设：
        - obj.index 是默认的索引 Index(range(len(obj)))
        - setitem 操作不会扩展 obj
    """

    @pytest.fixture
    def is_inplace(self, obj, expected):
        """
        判断设置操作是否应该是原地进行的。
        """
        return expected.dtype == obj.dtype

    def check_indexer(self, obj, key, expected, val, indexer, is_inplace):
        """
        检查指定索引方式的设置操作是否正确

        Parameters:
        - obj: 待操作的对象
        - key: 索引键值
        - expected: 预期的操作结果
        - val: 要设置的值
        - indexer: 索引器函数
        - is_inplace: 是否期望为原地操作
        """
        orig = obj
        obj = obj.copy()  # 复制对象，以避免修改原始对象
        arr = obj._values  # 获取对象的内部值数组

        # 使用给定的索引器设置键值对应的值
        indexer(obj)[key] = val
        # 检查设置后对象是否与预期结果相等
        tm.assert_series_equal(obj, expected)

        # 检查是否符合原地操作的预期
        self._check_inplace(is_inplace, orig, arr, obj)

    def _check_inplace(self, is_inplace, orig, arr, obj):
        """
        检查是否符合原地操作的预期

        Parameters:
        - is_inplace: 是否期望为原地操作
        - orig: 原始对象
        - arr: 操作前的内部值数组
        - obj: 操作后的对象
        """
        if is_inplace is None:
            # 暂时不检查是否原地操作
            pass
        elif is_inplace:
            if arr.dtype.kind in ["m", "M"]:
                # 对于日期时间类型，可能不是相同的 DTA/TDA，但底层数据应相同
                assert arr._ndarray is obj._values._ndarray
            else:
                # 对于其他类型，应该是同一个值数组对象
                assert obj._values is arr
        else:
            # 否则原始数组应该保持不变
            tm.assert_equal(arr, orig._values)
    # 测试整数索引键的情况，检查索引键是否为整数，如果不是则跳过测试
    def test_int_key(self, obj, key, expected, warn, val, indexer_sli, is_inplace):
        if not isinstance(key, int):
            pytest.skip("Not relevant for int key")

        # 断言应产生警告，并匹配指定的警告信息，检查索引器的行为
        with tm.assert_produces_warning(warn, match="incompatible dtype"):
            self.check_indexer(obj, key, expected, val, indexer_sli, is_inplace)

        # 如果索引器是 tm.loc，则进行特定处理
        if indexer_sli is tm.loc:
            with tm.assert_produces_warning(warn, match="incompatible dtype"):
                self.check_indexer(obj, key, expected, val, tm.at, is_inplace)
        # 如果索引器是 tm.iloc，则进行特定处理
        elif indexer_sli is tm.iloc:
            with tm.assert_produces_warning(warn, match="incompatible dtype"):
                self.check_indexer(obj, key, expected, val, tm.iat, is_inplace)

        # 创建一个范围对象 rng，包含从 key 到 key + 1 的范围
        rng = range(key, key + 1)
        # 断言应产生警告，并匹配指定的警告信息，检查索引器的行为
        with tm.assert_produces_warning(warn, match="incompatible dtype"):
            self.check_indexer(obj, rng, expected, val, indexer_sli, is_inplace)

        # 如果索引器不是 tm.loc，则创建一个切片对象 slc
        if indexer_sli is not tm.loc:
            # 注意：这里没有使用 .loc，因为它处理切片边界的方式不同
            slc = slice(key, key + 1)
            # 断言应产生警告，并匹配指定的警告信息，检查索引器的行为
            with tm.assert_produces_warning(warn, match="incompatible dtype"):
                self.check_indexer(obj, slc, expected, val, indexer_sli, is_inplace)

        # 创建一个包含 key 的列表 ilkey
        ilkey = [key]
        # 断言应产生警告，并匹配指定的警告信息，检查索引器的行为
        with tm.assert_produces_warning(warn, match="incompatible dtype"):
            self.check_indexer(obj, ilkey, expected, val, indexer_sli, is_inplace)

        # 创建一个 numpy 数组 indkey，其值与 ilkey 相同
        indkey = np.array(ilkey)
        # 断言应产生警告，并匹配指定的警告信息，检查索引器的行为
        with tm.assert_produces_warning(warn, match="incompatible dtype"):
            self.check_indexer(obj, indkey, expected, val, indexer_sli, is_inplace)

        # 创建一个生成器 genkey，其内容为 [key] 的迭代器
        genkey = (x for x in [key])
        # 断言应产生警告，并匹配指定的警告信息，检查索引器的行为
        with tm.assert_produces_warning(warn, match="incompatible dtype"):
            self.check_indexer(obj, genkey, expected, val, indexer_sli, is_inplace)
    # 定义测试函数，测试在给定对象上使用布尔掩码设置元素
    def test_mask_key(self, obj, key, expected, warn, val, indexer_sli):
        # 创建一个与对象形状相同的布尔数组，全部为 False
        mask = np.zeros(obj.shape, dtype=bool)
        # 将索引 `key` 处的元素设为 True
        mask[key] = True

        # 复制对象副本，以确保不改变原始对象
        obj = obj.copy()

        # 如果值 `val` 是类列表且长度小于掩码中 True 值的数量
        if is_list_like(val) and len(val) < mask.sum():
            msg = "boolean index did not match indexed array along dimension"
            # 预期引发 IndexError 异常，并匹配指定消息
            with pytest.raises(IndexError, match=msg):
                indexer_sli(obj)[mask] = val
            return

        # 断言设置值时产生警告，警告信息匹配 "incompatible dtype"
        with tm.assert_produces_warning(warn, match="incompatible dtype"):
            indexer_sli(obj)[mask] = val
        # 断言最终结果与期望值 `expected` 相等
        tm.assert_series_equal(obj, expected)

    # 定义测试函数，测试 Series 对象的 where 方法
    def test_series_where(self, obj, key, expected, warn, val, is_inplace):
        # 创建一个与对象形状相同的布尔数组，全部为 False
        mask = np.zeros(obj.shape, dtype=bool)
        # 将索引 `key` 处的元素设为 True
        mask[key] = True

        # 如果值 `val` 是类列表且长度小于对象长度
        if is_list_like(val) and len(val) < len(obj):
            # 抛出 ValueError 异常，消息匹配 "operands could not be broadcast together with shapes"
            msg = "operands could not be broadcast together with shapes"
            with pytest.raises(ValueError, match=msg):
                # 在条件 `~mask` 下使用 obj.where 方法
                obj.where(~mask, val)
            return

        # 保存原始对象引用
        orig = obj
        # 复制对象副本，以确保不改变原始对象
        obj = obj.copy()
        # 获取对象内部值数组
        arr = obj._values

        # 使用 where 方法生成结果 `res`
        res = obj.where(~mask, val)

        # 处理特殊情况：如果 `val` 是 NA 并且结果数组类型是 object，则将期望值填充为 NA
        if val is NA and res.dtype == object:
            expected = expected.fillna(NA)
        # 处理特殊情况：如果 `val` 是 None 并且结果数组类型是 object
        elif val is None and res.dtype == object:
            assert expected.dtype == object
            expected = expected.copy()
            expected[expected.isna()] = None
        # 断言 where 方法的结果与期望值 `expected` 相等
        tm.assert_series_equal(res, expected)

        # 检查是否符合就地修改的预期，以及对象修改后的一致性
        self._check_inplace(is_inplace, orig, arr, obj)

    # 定义测试函数，测试 Index 对象的 where 方法
    def test_index_where(self, obj, key, expected, warn, val, using_infer_string):
        # 创建一个与对象形状相同的布尔数组，全部为 False
        mask = np.zeros(obj.shape, dtype=bool)
        # 将索引 `key` 处的元素设为 True
        mask[key] = True

        # 如果使用推断字符串并且对象的 dtype 是 object 类型
        if using_infer_string and obj.dtype == object:
            # 预期引发 TypeError 异常，消息匹配 "Scalar must"
            with pytest.raises(TypeError, match="Scalar must"):
                # 在条件 `~mask` 下使用 Index 对象的 where 方法
                Index(obj).where(~mask, val)
        else:
            # 使用 where 方法生成结果 `res`
            res = Index(obj).where(~mask, val)
            # 创建期望的 Index 对象
            expected_idx = Index(expected, dtype=expected.dtype)
            # 断言 where 方法的结果 `res` 与期望的 Index 对象 `expected_idx` 相等
            tm.assert_index_equal(res, expected_idx)

    # 定义测试函数，测试 Index 对象的 putmask 方法
    def test_index_putmask(self, obj, key, expected, warn, val, using_infer_string):
        # 创建一个与对象形状相同的布尔数组，全部为 False
        mask = np.zeros(obj.shape, dtype=bool)
        # 将索引 `key` 处的元素设为 True
        mask[key] = True

        # 如果使用推断字符串并且对象的 dtype 是 object 类型
        if using_infer_string and obj.dtype == object:
            # 预期引发 TypeError 异常，消息匹配 "Scalar must"
            with pytest.raises(TypeError, match="Scalar must"):
                # 在条件 `mask` 下使用 Index 对象的 putmask 方法
                Index(obj).putmask(mask, val)
        else:
            # 使用 putmask 方法生成结果 `res`
            res = Index(obj).putmask(mask, val)
            # 创建期望的 Index 对象，并指定其 dtype
            expected_idx = Index(expected, dtype=expected.dtype)
            # 断言 putmask 方法的结果 `res` 与期望的 Index 对象 `expected_idx` 相等
            tm.assert_index_equal(res, expected_idx)
@pytest.mark.parametrize(
    "obj,expected,key,warn",
    [
        pytest.param(
            # GH#45568 setting a valid NA value into IntervalDtype[int] should
            #  cast to IntervalDtype[float]
            Series(interval_range(1, 5)),
            Series(
                [Interval(1, 2), np.nan, Interval(3, 4), Interval(4, 5)],
                dtype="interval[float64]",
            ),
            1,
            FutureWarning,
            id="interval_int_na_value",
        ),
        pytest.param(
            # these induce dtype changes
            Series([2, 3, 4, 5, 6, 7, 8, 9, 10]),
            Series([np.nan, 3, np.nan, 5, np.nan, 7, np.nan, 9, np.nan]),
            slice(None, None, 2),
            None,
            id="int_series_slice_key_step",
        ),
        pytest.param(
            Series([True, True, False, False]),
            Series([np.nan, True, np.nan, False], dtype=object),
            slice(None, None, 2),
            FutureWarning,
            id="bool_series_slice_key_step",
        ),
        pytest.param(
            # these induce dtype changes
            Series(np.arange(10)),
            Series([np.nan, np.nan, np.nan, np.nan, np.nan, 5, 6, 7, 8, 9]),
            slice(None, 5),
            None,
            id="int_series_slice_key",
        ),
        pytest.param(
            # changes dtype GH#4463
            Series([1, 2, 3]),
            Series([np.nan, 2, 3]),
            0,
            None,
            id="int_series_int_key",
        ),
        pytest.param(
            # changes dtype GH#4463
            Series([False]),
            Series([np.nan], dtype=object),
            # TODO: maybe go to float64 since we are changing the _whole_ Series?
            0,
            FutureWarning,
            id="bool_series_int_key_change_all",
        ),
        pytest.param(
            # changes dtype GH#4463
            Series([False, True]),
            Series([np.nan, True], dtype=object),
            0,
            FutureWarning,
            id="bool_series_int_key",
        ),
    ],
)
class TestSetitemCastingEquivalents(SetitemCastingEquivalents):
    @pytest.fixture(params=[np.nan, np.float64("NaN"), None, NA])
    def val(self, request):
        """
        NA values that should generally be valid_na for *all* dtypes.

        Include both python float NaN and np.float64; only np.float64 has a
        `dtype` attribute.
        """
        return request.param


class TestSetitemTimedelta64IntoNumeric(SetitemCastingEquivalents):
    # timedelta64 should not be treated as integers when setting into
    #  numeric Series

    @pytest.fixture
    def val(self):
        td = np.timedelta64(4, "ns")
        return td
        # TODO: could also try np.full((1,), td)

    @pytest.fixture(params=[complex, int, float])
    def dtype(self, request):
        return request.param

    @pytest.fixture
    # 定义一个方法，根据给定的数据类型创建一个包含0到4的数组，并将其转换为指定类型的序列返回
    def obj(self, dtype):
        arr = np.arange(5).astype(dtype)  # 创建一个从0到4的数组，并将其转换为指定的数据类型
        ser = Series(arr)  # 使用数组创建一个 pandas Series 对象
        return ser  # 返回创建的 Series 对象
    
    # 使用 pytest 装饰器定义一个测试固件，根据给定的数据类型创建一个序列，并对序列进行特定操作后返回
    @pytest.fixture
    def expected(self, dtype):
        arr = np.arange(5).astype(dtype)  # 创建一个从0到4的数组，并将其转换为指定的数据类型
        ser = Series(arr)  # 使用数组创建一个 pandas Series 对象
        ser = ser.astype(object)  # 将序列的数据类型转换为 object 类型
        ser.iloc[0] = np.timedelta64(4, "ns")  # 修改序列的第一个元素为一个时间差对象
        return ser  # 返回修改后的 Series 对象作为测试固件的结果
    
    # 使用 pytest 装饰器定义一个测试固件，返回一个整数 0，作为测试的关键参数之一
    @pytest.fixture
    def key(self):
        return 0  # 返回整数 0
    
    # 使用 pytest 装饰器定义一个测试固件，返回一个 FutureWarning 对象，用于测试警告相关功能
    @pytest.fixture
    def warn(self):
        return FutureWarning  # 返回 FutureWarning 类型的警告对象
class TestSetitemDT64IntoInt(SetitemCastingEquivalents):
    # GH#39619 dont cast dt64 to int when doing this setitem

    @pytest.fixture(params=["M8[ns]", "m8[ns]"])
    def dtype(self, request):
        # 返回测试参数中的日期时间类型
        return request.param

    @pytest.fixture
    def scalar(self, dtype):
        # 创建一个指定日期时间的标量值
        val = np.datetime64("2021-01-18 13:25:00", "ns")
        if dtype == "m8[ns]":
            # 如果是'm8[ns]'类型，则返回零值
            val = val - val
        return val

    @pytest.fixture
    def expected(self, scalar):
        # 创建一个包含标量值和整数的Series对象，并验证第一个元素的类型与标量相同
        expected = Series([scalar, scalar, 3], dtype=object)
        assert isinstance(expected[0], type(scalar))
        return expected

    @pytest.fixture
    def obj(self):
        # 创建一个包含整数的Series对象
        return Series([1, 2, 3])

    @pytest.fixture
    def key(self):
        # 返回一个切片对象，从开头到倒数第二个元素
        return slice(None, -1)

    @pytest.fixture(params=[None, list, np.array])
    def val(self, scalar, request):
        # 根据请求的类型参数返回不同的值，如果是None则返回标量，否则返回包含两个标量的列表或数组
        box = request.param
        if box is None:
            return scalar
        return box([scalar, scalar])

    @pytest.fixture
    def warn(self):
        # 返回一个未来的警告类对象
        return FutureWarning


class TestSetitemNAPeriodDtype(SetitemCastingEquivalents):
    # Setting compatible NA values into Series with PeriodDtype

    @pytest.fixture
    def expected(self, key):
        # 创建一个包含日期周期范围的Series对象，将指定键的值设置为NaT
        exp = Series(period_range("2000-01-01", periods=10, freq="D"))
        exp._values.view("i8")[key] = NaT._value
        assert exp[key] is NaT or all(x is NaT for x in exp[key])
        return exp

    @pytest.fixture
    def obj(self):
        # 创建一个包含日期周期范围的Series对象
        return Series(period_range("2000-01-01", periods=10, freq="D"))

    @pytest.fixture(params=[3, slice(3, 5)])
    def key(self, request):
        # 返回测试参数中的键，可以是整数或切片对象
        return request.param

    @pytest.fixture(params=[None, np.nan])
    def val(self, request):
        # 根据请求的参数返回None或NaN
        return request.param

    @pytest.fixture
    def warn(self):
        # 返回None，表示没有警告
        return None


class TestSetitemNADatetimeLikeDtype(SetitemCastingEquivalents):
    # some nat-like values should be cast to datetime64/timedelta64 when
    #  inserting into a datetime64/timedelta64 series.  Others should coerce
    #  to object and retain their dtypes.
    # GH#18586 for td64 and boolean mask case

    @pytest.fixture(
        params=["m8[ns]", "M8[ns]", "datetime64[ns, UTC]", "datetime64[ns, US/Central]"]
    )
    def dtype(self, request):
        # 返回日期时间类型的参数，用于测试
        return request.param

    @pytest.fixture
    def obj(self, dtype):
        # 创建一个Series对象，其索引是指定类型的日期时间索引
        i8vals = date_range("2016-01-01", periods=3).asi8
        idx = Index(i8vals, dtype=dtype)
        assert idx.dtype == dtype
        return Series(idx)

    @pytest.fixture(
        params=[
            None,
            np.nan,
            NaT,
            np.timedelta64("NaT", "ns"),
            np.datetime64("NaT", "ns"),
        ]
    )
    def val(self, request):
        # 返回测试参数中的不同日期时间相关的值，如None、NaN、NaT等
        return request.param

    @pytest.fixture
    def warn(self):
        # 返回None，表示没有警告
        return None
    # 判断给定值是否适合原地操作（inplace）：
    # - 如果值为 NaT（datetime64 中的 "Not a Time"），则转换为对象
    # - 如果值为 timedelta64 中的 NaT，则转换为对象
    # - 如果值为带时区信息的 timedelta64 且不是 NaT，则保持为对象
    def is_inplace(self, val, obj):
        return val is NaT or val is None or val is np.nan or obj.dtype == val.dtype

    # 生成预期的结果对象，根据是否原地操作（inplace）选择数据类型：
    # - 如果是原地操作，则保持与原对象相同的数据类型
    # - 否则，使用对象数据类型
    @pytest.fixture
    def expected(self, obj, val, is_inplace):
        dtype = obj.dtype if is_inplace else object
        expected = Series([val] + list(obj[1:]), dtype=dtype)
        return expected

    # 返回测试用例中的固定键值
    @pytest.fixture
    def key(self):
        return 0

    # 返回警告信息，根据是否原地操作（inplace）决定是否返回未来警告
    @pytest.fixture
    def warn(self, is_inplace):
        return None if is_inplace else FutureWarning
class TestSetitemMismatchedTZCastsToObject(SetitemCastingEquivalents):
    # GH#24024
    # 测试用例类，用于测试时区不匹配时的类型转换到对象的情况

    @pytest.fixture
    def obj(self):
        # 返回一个包含两个时区为'US/Central'的日期时间序列的Series对象作为测试对象
        return Series(date_range("2000", periods=2, tz="US/Central"))

    @pytest.fixture
    def val(self):
        # 返回一个带有'US/Eastern'时区的Timestamp对象作为测试值
        return Timestamp("2000", tz="US/Eastern")

    @pytest.fixture
    def key(self):
        # 返回索引键为0，作为测试用例的键
        return 0

    @pytest.fixture
    def expected(self, obj, val):
        # 返回一个预期的Series对象，包含两个日期时间值，并将val转换为'US/Central'时区
        # 在2.0版本之前，会将val强制转换为对象类型，现在会转换为目标时区
        expected = Series(
            [
                val.tz_convert("US/Central"),
                Timestamp("2000-01-02 00:00:00-06:00", tz="US/Central"),
            ],
            dtype=obj.dtype,
        )
        return expected

    @pytest.fixture
    def warn(self):
        # 返回None，表示没有警告信息
        return None


@pytest.mark.parametrize(
    "obj,expected,warn",
    [
        # 对于数值系列，应该强制转换为NaN
        (Series([1, 2, 3]), Series([np.nan, 2, 3]), None),
        (Series([1.0, 2.0, 3.0]), Series([np.nan, 2.0, 3.0]), None),
        # 对于日期时间系列，应该强制转换为NaT
        (
            Series([datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)]),
            Series([NaT, datetime(2000, 1, 2), datetime(2000, 1, 3)]),
            None,
        ),
        # 对于对象系列，应该保留None值
        (Series(["foo", "bar", "baz"]), Series([None, "bar", "baz"]), None),
    ],
)
class TestSeriesNoneCoercion(SetitemCastingEquivalents):
    # 测试类，用于测试Series的None值强制转换行为

    @pytest.fixture
    def key(self):
        # 返回索引键为0，作为测试用例的键
        return 0

    @pytest.fixture
    def val(self):
        # 返回None作为测试值
        return None


class TestSetitemFloatIntervalWithIntIntervalValues(SetitemCastingEquivalents):
    # GH#44201 Cast to shared IntervalDtype rather than object
    # 测试类，用于测试将浮点间隔值转换为共享的IntervalDtype类型

    def test_setitem_example(self):
        # 示例测试方法，用于明确此测试类的目标
        idx = IntervalIndex.from_breaks(range(4))
        obj = Series(idx)
        val = Interval(0.5, 1.5)

        with tm.assert_produces_warning(
            FutureWarning, match="Setting an item of incompatible dtype"
        ):
            obj[0] = val
        assert obj.dtype == "Interval[float64, right]"

    @pytest.fixture
    def obj(self):
        # 返回一个包含间隔索引的Series对象作为测试对象
        idx = IntervalIndex.from_breaks(range(4))
        return Series(idx)

    @pytest.fixture
    def val(self):
        # 返回一个间隔对象作为测试值
        return Interval(0.5, 1.5)

    @pytest.fixture
    def key(self):
        # 返回索引键为0，作为测试用例的键
        return 0

    @pytest.fixture
    def expected(self, obj, val):
        # 返回一个预期的Series对象，包含val和obj其余部分的间隔值
        data = [val] + list(obj[1:])
        idx = IntervalIndex(data, dtype="Interval[float64]")
        return Series(idx)

    @pytest.fixture
    def warn(self):
        # 返回FutureWarning，表示测试中会产生此警告
        return FutureWarning


class TestSetitemRangeIntoIntegerSeries(SetitemCastingEquivalents):
    # GH#44261 Setting a range with sufficiently-small integers into
    #  small-itemsize integer dtypes should not need to upcast
    # 测试类，用于测试在小型整数dtype中设置范围时不需要进行向上转换的情况

    @pytest.fixture
    # 定义一个方法 obj，接受一个任意整数类型的 NumPy 数据类型作为参数
    def obj(self, any_int_numpy_dtype):
        # 使用给定的任意整数类型创建一个 NumPy 数据类型对象
        dtype = np.dtype(any_int_numpy_dtype)
        # 创建一个 Series 对象，其值为 [0, 1, 2, 3, 4]，数据类型为上面创建的 NumPy 数据类型
        ser = Series(range(5), dtype=dtype)
        # 返回创建的 Series 对象
        return ser

    # 定义一个 pytest 的 fixture，名为 val，返回一个范围为 [2, 3] 的迭代器
    @pytest.fixture
    def val(self):
        return range(2, 4)

    # 定义一个 pytest 的 fixture，名为 key，返回一个切片对象，范围为 [0, 2)
    @pytest.fixture
    def key(self):
        return slice(0, 2)

    # 定义一个 pytest 的 fixture，名为 expected，接受一个任意整数类型的 NumPy 数据类型作为参数
    @pytest.fixture
    def expected(self, any_int_numpy_dtype):
        # 使用给定的任意整数类型创建一个 NumPy 数据类型对象
        dtype = np.dtype(any_int_numpy_dtype)
        # 创建一个 Series 对象，其值为 [2, 3, 2, 3, 4]，数据类型为上面创建的 NumPy 数据类型
        exp = Series([2, 3, 2, 3, 4], dtype=dtype)
        # 返回创建的 Series 对象
        return exp

    # 定义一个 pytest 的 fixture，名为 warn，返回 None
    @pytest.fixture
    def warn(self):
        return None
@pytest.mark.parametrize(
    "val, warn",
    [
        # 测试参数化：传入一个包含两个浮点数数组和一个警告类或None的元组列表
        (np.array([2.0, 3.0]), None),
        (np.array([2.5, 3.5]), FutureWarning),
        (
            np.array([2**65, 2**65 + 1], dtype=np.float64),
            FutureWarning,
        ),  # 所有都是整数，但无法强制转换
    ],
)
class TestSetitemFloatNDarrayIntoIntegerSeries(SetitemCastingEquivalents):
    @pytest.fixture
    def obj(self):
        # 创建一个包含5个元素的整数Series对象
        return Series(range(5), dtype=np.int64)

    @pytest.fixture
    def key(self):
        # 创建一个切片对象，范围从0到2
        return slice(0, 2)

    @pytest.fixture
    def expected(self, val):
        if val[0] == 2:
            # 注意：此条件基于当前硬编码的"val"情况
            dtype = np.int64
        else:
            dtype = np.float64
        # 创建一个期望的Series对象，根据dtype不同设置不同的数据类型
        res_values = np.array(range(5), dtype=dtype)
        res_values[:2] = val
        return Series(res_values)


@pytest.mark.parametrize("val", [512, np.int16(512)])
class TestSetitemIntoIntegerSeriesNeedsUpcast(SetitemCastingEquivalents):
    @pytest.fixture
    def obj(self):
        # 创建一个包含[1, 2, 3]的int8类型Series对象
        return Series([1, 2, 3], dtype=np.int8)

    @pytest.fixture
    def key(self):
        # 设置key为1
        return 1

    @pytest.fixture
    def expected(self):
        # 创建一个期望的Series对象，将索引为1的元素替换为512，并强制转换为int16类型
        return Series([1, 512, 3], dtype=np.int16)

    @pytest.fixture
    def warn(self):
        # 返回FutureWarning警告
        return FutureWarning


@pytest.mark.parametrize("val", [2**33 + 1.0, 2**33 + 1.1, 2**62])
class TestSmallIntegerSetitemUpcast(SetitemCastingEquivalents):
    # https://github.com/pandas-dev/pandas/issues/39584#issuecomment-941212124
    @pytest.fixture
    def obj(self):
        # 创建一个包含[1, 2, 3]的i4类型Series对象
        return Series([1, 2, 3], dtype="i4")

    @pytest.fixture
    def key(self):
        # 设置key为0
        return 0

    @pytest.fixture
    def expected(self, val):
        if val % 1 != 0:
            # 如果val不是整数，则设置dtype为f8
            dtype = "f8"
        else:
            # 否则设置dtype为i8
            dtype = "i8"
        # 创建一个期望的Series对象，将索引为0的元素替换为val，并设置数据类型为dtype
        return Series([val, 2, 3], dtype=dtype)

    @pytest.fixture
    def warn(self):
        # 返回FutureWarning警告
        return FutureWarning


class CoercionTest(SetitemCastingEquivalents):
    # 从tests.indexing.test_coercion中迁移的测试

    @pytest.fixture
    def key(self):
        # 设置key为1
        return 1

    @pytest.fixture
    def expected(self, obj, key, val, exp_dtype):
        # 创建一个列表vals，内容与obj相同，将索引为key的元素替换为val
        vals = list(obj)
        vals[key] = val
        # 返回一个期望的Series对象，数据类型为exp_dtype
        return Series(vals, dtype=exp_dtype)


@pytest.mark.parametrize(
    "val,exp_dtype,warn",
    [(np.int32(1), np.int8, None), (np.int16(2**9), np.int16, FutureWarning)],
)
class TestCoercionInt8(CoercionTest):
    # 之前在tests.indexing.test_coercion中的test_setitem_series_int8测试

    @pytest.fixture
    def obj(self):
        # 创建一个包含[1, 2, 3, 4]的int8类型Series对象
        return Series([1, 2, 3, 4], dtype=np.int8)


@pytest.mark.parametrize("val", [1, 1.1, 1 + 1j, True])
@pytest.mark.parametrize("exp_dtype", [object])
class TestCoercionObject(CoercionTest):
    # 之前在tests.indexing.test_coercion中的test_setitem_series_object测试

    @pytest.fixture
    def obj(self):
        # 创建一个包含["a", "b", "c", "d"]的object类型Series对象
        return Series(["a", "b", "c", "d"], dtype=object)

    @pytest.fixture
    def warn(self):
        # 返回None，表示不会有警告
        return None
    [
        # 元组1: 包含复数值1，数据类型为np.complex128，未指定警告
        (1, np.complex128, None),
        # 元组2: 包含复数值1.1，数据类型为np.complex128，未指定警告
        (1.1, np.complex128, None),
        # 元组3: 包含复数值1 + 1j，数据类型为np.complex128，未指定警告
        (1 + 1j, np.complex128, None),
        # 元组4: 包含布尔值True，数据类型为object，将会触发FutureWarning警告
        (True, object, FutureWarning),
    ],
# 定义一个测试类 TestCoercionComplex，继承自 CoercionTest
class TestCoercionComplex(CoercionTest):
    # 之前在 tests.indexing.test_coercion 中的 test_setitem_series_complex128
    @pytest.fixture
    # 定义一个 pytest fixture，返回一个包含复数数据的 Series 对象
    def obj(self):
        return Series([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j])


# 使用 pytest.mark.parametrize 装饰器为 TestCoercionBool 类定义参数化测试
@pytest.mark.parametrize(
    "val,exp_dtype,warn",
    [
        (1, object, FutureWarning),
        ("3", object, FutureWarning),
        (3, object, FutureWarning),
        (1.1, object, FutureWarning),
        (1 + 1j, object, FutureWarning),
        (True, bool, None),
    ],
)
# 定义一个测试类 TestCoercionBool，继承自 CoercionTest
class TestCoercionBool(CoercionTest):
    # 之前在 tests.indexing.test_coercion 中的 test_setitem_series_bool
    @pytest.fixture
    # 定义一个 pytest fixture，返回一个包含布尔值数据的 Series 对象
    def obj(self):
        return Series([True, False, True, False], dtype=bool)


# 使用 pytest.mark.parametrize 装饰器为 TestCoercionInt64 类定义参数化测试
@pytest.mark.parametrize(
    "val,exp_dtype,warn",
    [
        (1, np.int64, None),
        (1.1, np.float64, FutureWarning),
        (1 + 1j, np.complex128, FutureWarning),
        (True, object, FutureWarning),
    ],
)
# 定义一个测试类 TestCoercionInt64，继承自 CoercionTest
class TestCoercionInt64(CoercionTest):
    # 之前在 tests.indexing.test_coercion 中的 test_setitem_series_int64
    @pytest.fixture
    # 定义一个 pytest fixture，返回一个包含整数数据的 Series 对象
    def obj(self):
        return Series([1, 2, 3, 4])


# 使用 pytest.mark.parametrize 装饰器为 TestCoercionFloat64 类定义参数化测试
@pytest.mark.parametrize(
    "val,exp_dtype,warn",
    [
        (1, np.float64, None),
        (1.1, np.float64, None),
        (1 + 1j, np.complex128, FutureWarning),
        (True, object, FutureWarning),
    ],
)
# 定义一个测试类 TestCoercionFloat64，继承自 CoercionTest
class TestCoercionFloat64(CoercionTest):
    # 之前在 tests.indexing.test_coercion 中的 test_setitem_series_float64
    @pytest.fixture
    # 定义一个 pytest fixture，返回一个包含双精度浮点数数据的 Series 对象
    def obj(self):
        return Series([1.1, 2.2, 3.3, 4.4])


# 使用 pytest.mark.parametrize 装饰器为 TestCoercionFloat32 类定义参数化测试
@pytest.mark.parametrize(
    "val,exp_dtype,warn",
    [
        (1, np.float32, None),
        pytest.param(
            1.1,
            np.float32,
            None,
            marks=pytest.mark.xfail(
                (
                    not np_version_gte1p24
                    or (np_version_gte1p24 and np._get_promotion_state() != "weak")
                ),
                reason="np.float32(1.1) ends up as 1.100000023841858, so "
                "np_can_hold_element raises and we cast to float64",
            ),
        ),
        (1 + 1j, np.complex128, FutureWarning),
        (True, object, FutureWarning),
        (np.uint8(2), np.float32, None),
        (np.uint32(2), np.float32, None),
        # float32 无法精确表示 np.iinfo(np.uint32).max
        # (最接近的是 4294967300.0，差距为 5.0)，因此需要转换为 float64
        (np.uint32(np.iinfo(np.uint32).max), np.float64, FutureWarning),
        (np.uint64(2), np.float32, None),
        (np.int64(2), np.float32, None),
    ],
)
# 定义一个测试类 TestCoercionFloat32，继承自 CoercionTest
class TestCoercionFloat32(CoercionTest):
    @pytest.fixture
    # 定义一个 pytest fixture，返回一个包含单精度浮点数数据的 Series 对象
    def obj(self):
        return Series([1.1, 2.2, 3.3, 4.4], dtype=np.float32)
    def test_slice_key(self, obj, key, expected, warn, val, indexer_sli, is_inplace):
        # 调用父类方法 test_slice_key，传递参数 obj, key, expected, warn, val, indexer_sli, is_inplace
        super().test_slice_key(obj, key, expected, warn, val, indexer_sli, is_inplace)

        # 如果 val 是 float 类型，则抛出断言错误
        if isinstance(val, float):
            # xfail 在这个测试中会 xpass，因为 test_slice_key 进行了短路处理
            raise AssertionError("xfail not relevant for this test.")
# 使用 pytest.mark.parametrize 装饰器，为测试类参数化，exp_dtype 表示期望的数据类型
@pytest.mark.parametrize(
    "exp_dtype",
    [
        "M8[ms]",          # 第一种期望的数据类型，表示毫秒精度的日期时间
        "M8[ms, UTC]",     # 第二种期望的数据类型，表示带有时区的毫秒精度日期时间
        "m8[ms]",          # 第三种期望的数据类型，表示以秒为单位的日期时间
    ],
)
# 定义一个测试类 TestCoercionDatetime64HigherReso，继承自 CoercionTest
class TestCoercionDatetime64HigherReso(CoercionTest):

    # pytest fixture，为 obj 方法提供依赖，exp_dtype 表示期望的数据类型
    @pytest.fixture
    def obj(self, exp_dtype):
        # 创建一个日期范围索引，从 "2011-01-01" 开始，频率为每天一次，周期为 4，单位为秒
        idx = date_range("2011-01-01", freq="D", periods=4, unit="s")
        if exp_dtype == "m8[ms]":  # 如果期望的数据类型是秒精度的日期时间
            idx = idx - Timestamp("1970-01-01")  # 将索引转换为从 1970 年开始的时间戳
            assert idx.dtype == "m8[s]"  # 断言索引的数据类型为秒精度的日期时间
        elif exp_dtype == "M8[ms, UTC]":  # 如果期望的数据类型是带时区的毫秒精度日期时间
            idx = idx.tz_localize("UTC")  # 将索引本地化为 UTC 时区
        return Series(idx)  # 返回由索引创建的 Series 对象

    # pytest fixture，为 val 方法提供依赖，exp_dtype 表示期望的数据类型
    @pytest.fixture
    def val(self, exp_dtype):
        ts = Timestamp("2011-01-02 03:04:05.678").as_unit("ms")  # 创建时间戳，并转换为毫秒单位
        if exp_dtype == "m8[ms]":  # 如果期望的数据类型是秒精度的日期时间
            return ts - Timestamp("1970-01-01")  # 返回时间戳减去 1970 年的时间戳
        elif exp_dtype == "M8[ms, UTC]":  # 如果期望的数据类型是带时区的毫秒精度日期时间
            return ts.tz_localize("UTC")  # 将时间戳本地化为 UTC 时区
        return ts  # 返回转换后的时间戳

    # pytest fixture，为 warn 方法提供依赖，返回 FutureWarning 类型对象
    @pytest.fixture
    def warn(self):
        return FutureWarning


# 使用 pytest.mark.parametrize 装饰器，为测试类参数化，val 表示值，exp_dtype 表示期望的数据类型，warn 表示警告类型
@pytest.mark.parametrize(
    "val,exp_dtype,warn",
    [
        (Timestamp("2012-01-01"), "datetime64[ns]", None),  # 第一组参数，时间戳和期望的数据类型
        (1, object, FutureWarning),  # 第二组参数，整数和期望的数据类型以及 FutureWarning 警告
        ("x", object, FutureWarning),  # 第三组参数，字符串和期望的数据类型以及 FutureWarning 警告
    ],
)
# 定义一个测试类 TestCoercionDatetime64，继承自 CoercionTest
class TestCoercionDatetime64(CoercionTest):

    # pytest fixture，为 obj 方法提供依赖，返回一个 Series 对象，包含从 "2011-01-01" 开始的 4 天日期范围
    @pytest.fixture
    def obj(self):
        return Series(date_range("2011-01-01", freq="D", periods=4))

    # pytest fixture，为 warn 方法提供依赖，返回 None，表示没有警告
    @pytest.fixture
    def warn(self):
        return None


# 使用 pytest.mark.parametrize 装饰器，为测试类参数化，val 表示值，exp_dtype 表示期望的数据类型，warn 表示警告类型
@pytest.mark.parametrize(
    "val,exp_dtype,warn",
    [
        (Timestamp("2012-01-01", tz="US/Eastern"), "datetime64[ns, US/Eastern]", None),  # 第一组参数，带时区的时间戳和期望的数据类型
        (Timestamp("2012-01-01", tz="US/Pacific"), "datetime64[ns, US/Eastern]", None),  # 第二组参数，带时区的时间戳和期望的数据类型
        (Timestamp("2012-01-01"), object, FutureWarning),  # 第三组参数，时间戳和期望的数据类型以及 FutureWarning 警告
        (1, object, FutureWarning),  # 第四组参数，整数和期望的数据类型以及 FutureWarning 警告
    ],
)
# 定义一个测试类 TestCoercionDatetime64TZ，继承自 CoercionTest
class TestCoercionDatetime64TZ(CoercionTest):

    # pytest fixture，为 obj 方法提供依赖，返回一个 Series 对象，包含从 "2011-01-01" 开始的 4 天日期范围，并使用 "US/Eastern" 时区
    @pytest.fixture
    def obj(self):
        tz = "US/Eastern"
        return Series(date_range("2011-01-01", freq="D", periods=4, tz=tz))

    # pytest fixture，为 warn 方法提供依赖，返回 None，表示没有警告
    @pytest.fixture
    def warn(self):
        return None


# 使用 pytest.mark.parametrize 装饰器，为测试类参数化，val 表示值，exp_dtype 表示期望的数据类型，warn 表示警告类型
@pytest.mark.parametrize(
    "val,exp_dtype,warn",
    [
        (Timedelta("12 day"), "timedelta64[ns]", None),  # 第一组参数，时间差和期望的数据类型
        (1, object, FutureWarning),  # 第二组参数，整数和期望的数据类型以及 FutureWarning 警告
        ("x", object, FutureWarning),  # 第三组参数，字符串和期望的数据类型以及 FutureWarning 警告
    ],
)
# 定义一个测试类 TestCoercionTimedelta64，继承自 CoercionTest
class TestCoercionTimedelta64(CoercionTest):

    # pytest fixture，为 obj 方法提供依赖，返回一个 Series 对象，包含时间差范围从 "1 day" 开始的 4 天
    @pytest.fixture
    def obj(self):
        return Series(timedelta_range("1 day", periods=4))

    # pytest fixture，为 warn 方法提供依赖，返回 None，表示没有警告
    @pytest.fixture
    def warn(self):
        return None


# 使用 pytest.mark.parametrize 装饰器，为测试类参数化，val 表示值，exp_dtype 表示期望的数据类型
@pytest.mark.parametrize(
    "val", ["foo", Period("2016", freq="Y"), Interval(1, 2, closed="both")]
)
# 使用 pytest.mark.parametrize 装饰器，为测试类参数化，exp_dtype 表示期望的数据类型为 object
@pytest.mark.parametrize("exp_dtype", [object])
# 定义一个测试类 TestPeriodIntervalCoercion，继承自 CoercionTest
class TestPeriodIntervalCoercion(CoercionTest):

    # GH#45768
    # pytest fixture，为 obj 方法提供依赖，返回一个参数化的 Series 对象列表
    @pytest.fixture(
        params=[
            period_range("2016-01-01", periods=3, freq="D"),  # 创建一个日期范围，从 "2016-01-01" 开始，周期为 3 天
            interval_range(1, 5),  # 创建一个区间范围，从 1 到 5，包括两
    # 定义一个方法 'obj'，接受一个参数 'request'，并返回一个 Series 对象
    def obj(self, request):
        return Series(request.param)

    # 使用 pytest 装饰器定义一个测试夹具 'warn'，返回 FutureWarning 类型的对象
    @pytest.fixture
    def warn(self):
        return FutureWarning
def test_20643():
    # 根据 GH#45121 关闭测试
    orig = Series([0, 1, 2], index=["a", "b", "c"])

    expected = Series([0, 2.7, 2], index=["a", "b", "c"])

    # 使用原始数据的副本
    ser = orig.copy()
    # 检查是否产生 FutureWarning 警告，匹配字符串 "incompatible dtype"
    with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
        # 使用 at 方法设置指定位置的值
        ser.at["b"] = 2.7
    # 断言 ser 和 expected 是否相等
    tm.assert_series_equal(ser, expected)

    ser = orig.copy()
    with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
        # 使用 loc 方法设置指定标签位置的值
        ser.loc["b"] = 2.7
    tm.assert_series_equal(ser, expected)

    ser = orig.copy()
    with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
        # 直接通过标签设置指定位置的值
        ser["b"] = 2.7
    tm.assert_series_equal(ser, expected)

    ser = orig.copy()
    with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
        # 使用 iat 方法设置指定整数位置的值
        ser.iat[1] = 2.7
    tm.assert_series_equal(ser, expected)

    ser = orig.copy()
    with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
        # 使用 iloc 方法设置指定整数位置的值
        ser.iloc[1] = 2.7
    tm.assert_series_equal(ser, expected)

    # 将原始 Series 转换为 DataFrame
    orig_df = orig.to_frame("A")
    expected_df = expected.to_frame("A")

    df = orig_df.copy()
    with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
        # 使用 at 方法设置 DataFrame 中指定标签的值
        df.at["b", "A"] = 2.7
    tm.assert_frame_equal(df, expected_df)

    df = orig_df.copy()
    with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
        # 使用 loc 方法设置 DataFrame 中指定标签的值
        df.loc["b", "A"] = 2.7
    tm.assert_frame_equal(df, expected_df)

    df = orig_df.copy()
    with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
        # 使用 iloc 方法设置 DataFrame 中指定整数位置的值
        df.iloc[1, 0] = 2.7
    tm.assert_frame_equal(df, expected_df)

    df = orig_df.copy()
    with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
        # 使用 iat 方法设置 DataFrame 中指定整数位置的值
        df.iat[1, 0] = 2.7
    tm.assert_frame_equal(df, expected_df)


def test_20643_comment():
    # 根据 https://github.com/pandas-dev/pandas/issues/20643#issuecomment-431244590
    # 在 GH#45121 之前修复
    orig = Series([0, 1, 2], index=["a", "b", "c"])
    expected = Series([np.nan, 1, 2], index=["a", "b", "c"])

    ser = orig.copy()
    # 使用 iat 方法设置指定整数位置的值为 None
    ser.iat[0] = None
    tm.assert_series_equal(ser, expected)

    ser = orig.copy()
    # 使用 iloc 方法设置指定整数位置的值为 None
    ser.iloc[0] = None
    tm.assert_series_equal(ser, expected)


def test_15413():
    # 通过 GH#45121 修复
    ser = Series([1, 2, 3])

    with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
        # 将 Series 中等于 2 的元素加上 0.5
        ser[ser == 2] += 0.5
    expected = Series([1, 2.5, 3])
    tm.assert_series_equal(ser, expected)

    ser = Series([1, 2, 3])
    with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
        # 将 Series 中索引为 1 的元素加上 0.5
        ser[1] += 0.5
    tm.assert_series_equal(ser, expected)

    ser = Series([1, 2, 3])
    with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
        # 使用 loc 方法将 Series 中索引为 1 的元素加上 0.5
        ser.loc[1] += 0.5
    tm.assert_series_equal(ser, expected)

    ser = Series([1, 2, 3])
    with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
        # 使用 iloc 方法将 Series 中整数位置为 1 的元素加上 0.5
        ser.iloc[1] += 0.5
    # 调用测试框架中的函数，验证序列 `ser` 是否与期望的序列 `expected` 相等
    tm.assert_series_equal(ser, expected)
    
    # 创建一个包含整数 1, 2, 3 的序列 `ser`
    ser = Series([1, 2, 3])
    
    # 使用 `assert_produces_warning` 上下文管理器，检测是否会产生 `FutureWarning` 警告，并且该警告消息包含字符串 "incompatible dtype"
    with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
        # 修改序列 `ser` 中索引为 1 的元素，使其增加 0.5
        ser.iat[1] += 0.5
    
    # 再次调用测试框架中的函数，验证修改后的序列 `ser` 是否与期望的序列 `expected` 相等
    tm.assert_series_equal(ser, expected)
    
    # 创建一个包含整数 1, 2, 3 的序列 `ser`
    ser = Series([1, 2, 3])
    
    # 使用 `assert_produces_warning` 上下文管理器，检测是否会产生 `FutureWarning` 警告，并且该警告消息包含字符串 "incompatible dtype"
    with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
        # 修改序列 `ser` 中索引为 1 的元素，使其增加 0.5
        ser.at[1] += 0.5
    
    # 再次调用测试框架中的函数，验证修改后的序列 `ser` 是否与期望的序列 `expected` 相等
    tm.assert_series_equal(ser, expected)
# GH#45121 修复了问题编号为 32878 的Bug
def test_32878_int_itemsize():
    # 创建一个包含5个元素的 numpy 整数数组，数据类型为 'i4'
    arr = np.arange(5).astype("i4")
    # 从数组创建一个 Pandas Series 对象
    ser = Series(arr)
    # 获取 np.int64 类型的最大值，作为待赋值的值
    val = np.int64(np.iinfo(np.int64).max)
    # 断言赋值操作会产生 FutureWarning 警告，警告信息匹配 "incompatible dtype"
    with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
        ser[0] = val
    # 创建一个期望的 Series 对象，包含预期的值和数据类型为 np.int64
    expected = Series([val, 1, 2, 3, 4], dtype=np.int64)
    # 断言 ser 和 expected 是否相等
    tm.assert_series_equal(ser, expected)


# GH#32878 修复了复杂类型的问题，之前会将 val 强制转换为 inf+0.000000e+00j
def test_32878_complex_itemsize():
    # 创建一个包含5个元素的 numpy 复数数组，数据类型为 'c8'
    arr = np.arange(5).astype("c8")
    # 从数组创建一个 Pandas Series 对象
    ser = Series(arr)
    # 获取 np.float64 类型的最大值
    val = np.finfo(np.float64).max
    # 将 val 转换为复数类型 'c16'
    val = val.astype("c16")

    # 断言赋值操作会产生 FutureWarning 警告，警告信息匹配 "incompatible dtype"
    with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
        ser[0] = val
    # 断言 ser[0] 和 val 是否相等
    assert ser[0] == val
    # 创建一个期望的 Series 对象，包含预期的值和数据类型为 'c16'
    expected = Series([val, 1, 2, 3, 4], dtype="c16")
    # 断言 ser 和 expected 是否相等
    tm.assert_series_equal(ser, expected)


# GH#37692 修复了问题编号为 37692 的Bug
def test_37692(indexer_al):
    # 创建一个包含3个元素的 Pandas Series 对象，指定了索引
    ser = Series([1, 2, 3], index=["a", "b", "c"])
    # 断言赋值操作会产生 FutureWarning 警告，警告信息匹配 "incompatible dtype"
    with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
        # 使用 indexer_al 函数设置索引为 'b' 的值为 "test"
        indexer_al(ser)["b"] = "test"
    # 创建一个期望的 Series 对象，包含预期的值和数据类型为 object
    expected = Series([1, "test", 3], index=["a", "b", "c"], dtype=object)
    # 断言 ser 和 expected 是否相等
    tm.assert_series_equal(ser, expected)


# GH#21513 修复了问题编号为 21513 的Bug
def test_setitem_bool_int_float_consistency(indexer_sli):
    # bool-with-int 和 bool-with-float 都会强制转换为 object 类型
    # int-with-float 和 float-with-int 只要能无损设置则不会强制转换
    for dtype in [np.float64, np.int64]:
        # 创建一个包含3个元素的 Pandas Series 对象，指定了索引和数据类型
        ser = Series(0, index=range(3), dtype=dtype)
        # 断言赋值操作会产生 FutureWarning 警告，警告信息匹配 "incompatible dtype"
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            # 使用 indexer_sli 函数设置索引为 0 的值为 True
            indexer_sli(ser)[0] = True
        # 断言 ser 的数据类型是否为 object
        assert ser.dtype == object

        # 创建一个包含3个元素的 Pandas Series 对象，指定了索引和数据类型为 bool
        ser = Series(0, index=range(3), dtype=bool)
        # 断言赋值操作会产生 FutureWarning 警告，警告信息匹配 "incompatible dtype"
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            # 设置索引为 0 的值为 dtype(1)
            ser[0] = dtype(1)
        # 断言 ser 的数据类型是否为 object
        assert ser.dtype == object

    # 1.0 可以无损保存，因此不会强制转换
    ser = Series(0, index=range(3), dtype=np.int64)
    # 设置索引为 0 的值为 np.float64(1.0)
    indexer_sli(ser)[0] = np.float64(1.0)
    # 断言 ser 的数据类型是否为 np.int64
    assert ser.dtype == np.int64

    # 1 可以无损保存，因此不会强制转换
    ser = Series(0, index=range(3), dtype=np.float64)
    # 设置索引为 0 的值为 np.int64(1)
    indexer_sli(ser)[0] = np.int64(1)


# GH#45070 处理在 __setitem__ 中出现 KeyError 的情况，如果回退时仍然会出现 ValueError
def test_setitem_positional_with_casting():
    # 创建一个包含3个元素的 Pandas Series 对象，指定了索引
    ser = Series([1, 2, 3], index=["a", "b", "c"])

    # 设置索引为 0 的值为 "X"
    ser[0] = "X"
    # 创建一个期望的 Series 对象，包含预期的值和数据类型为 object
    expected = Series([1, 2, 3, "X"], index=["a", "b", "c", 0], dtype=object)
    # 断言 ser 和 expected 是否相等
    tm.assert_series_equal(ser, expected)


# 处理将 float 设置为 int 时会发生强制类型转换的情况
def test_setitem_positional_float_into_int_coerces():
    # 创建一个包含3个元素的 Pandas Series 对象，指定了索引
    ser = Series([1, 2, 3], index=["a", "b", "c"])

    # 将索引为 0 的值设置为 1.5，预期会将 float 转换为 int 类型
    ser[0] = 1.5
    # 创建预期的 Series 对象，包含给定的值和索引
    expected = Series([1, 2, 3, 1.5], index=["a", "b", "c", 0])
    # 使用测试框架中的函数比较两个 Series 对象是否相等
    tm.assert_series_equal(ser, expected)
def test_setitem_int_not_positional():
    # GH#42215 deprecated falling back to positional on __setitem__ with an
    #  int not contained in the index; enforced in 2.0
    
    # 创建一个 Series 对象，指定数据和非数值型索引
    ser = Series([1, 2, 3, 4], index=[1.1, 2.1, 3.0, 4.1])
    # 确认在新的版本中不会回退到位置索引
    assert not ser.index._should_fallback_to_positional

    # 索引 3.0 存在于索引中，所以在新版本中的行为不会改变
    ser[3] = 10
    expected = Series([1, 2, 10, 4], index=ser.index)
    tm.assert_series_equal(ser, expected)

    # 在旧版本中，`ser[5] = 5` 会抛出 IndexError
    ser[5] = 5
    expected = Series([1, 2, 10, 4, 5], index=[1.1, 2.1, 3.0, 4.1, 5.0])
    tm.assert_series_equal(ser, expected)

    # 创建一个 IntervalIndex，并生成对应的 Series
    ii = IntervalIndex.from_breaks(range(10))[::2]
    ser2 = Series(range(len(ii)), index=ii)
    # 扩展索引并创建预期结果
    exp_index = ii.astype(object).append(Index([4]))
    expected2 = Series([0, 1, 2, 3, 4, 9], index=exp_index)
    # 在旧版本中，`ser2[4] = 9` 会将 4 解释为位置索引
    ser2[4] = 9
    tm.assert_series_equal(ser2, expected2)

    # 创建一个 MultiIndex，并生成对应的 Series
    mi = MultiIndex.from_product([ser.index, ["A", "B"]])
    ser3 = Series(range(len(mi)), index=mi)
    expected3 = ser3.copy()
    expected3.loc[4] = 99
    # 在旧版本中，`ser3[4] = 99` 会将 4 解释为位置索引
    ser3[4] = 99
    tm.assert_series_equal(ser3, expected3)


def test_setitem_with_bool_indexer():
    # GH#42530

    # 创建一个 DataFrame，并使用 pop 函数移除列 'b'，复制结果
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.pop("b").copy()
    # 使用布尔索引器设置部分值为 9
    result[[True, False, False]] = 9
    expected = Series(data=[9, 5, 6], name="b")
    tm.assert_series_equal(result, expected)

    # 使用布尔索引器设置 DataFrame 中的值为 10
    df.loc[[True, False, False], "a"] = 10
    expected = DataFrame({"a": [10, 2, 3]})
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize("size", range(2, 6))
@pytest.mark.parametrize(
    "mask", [[True, False, False, False, False], [True, False], [False]]
)
@pytest.mark.parametrize(
    "item", [2.0, np.nan, np.finfo(float).max, np.finfo(float).min]
)
@pytest.mark.parametrize("box", [np.array, list, tuple])
def test_setitem_bool_indexer_dont_broadcast_length1_values(size, mask, item, box):
    # GH#44265
    # 参见 tests.series.indexing.test_where.test_broadcast

    # 根据 mask 创建一个选择器
    selection = np.resize(mask, size)

    # 生成数据并创建 Series
    data = np.arange(size, dtype=float)
    ser = Series(data)

    if selection.sum() != 1:
        msg = (
            "cannot set using a list-like indexer with a different "
            "length than the value"
        )
        # 当选择器长度与值长度不同时，应当抛出 ValueError
        with pytest.raises(ValueError, match=msg):
            # GH#44265
            ser[selection] = box([item])
    else:
        # 在这种特殊情况下，设置操作等同于未包装的 item
        ser[selection] = box([item])

        expected = Series(np.arange(size, dtype=float))
        expected[selection] = item
        tm.assert_series_equal(ser, expected)


def test_setitem_empty_mask_dont_upcast_dt64():
    # 创建一个日期范围对象，并生成对应的 Series
    dti = date_range("2016-01-01", periods=3)
    ser = Series(dti)
    # 复制原始 Series 对象，创建一个新的对象 orig，保留原始数据
    orig = ser.copy()
    # 创建一个长度为 3 的布尔类型的数组 mask，所有元素初始化为 False
    mask = np.zeros(3, dtype=bool)

    # 使用 mask 数组选择 ser 中的元素，并将其设置为字符串 "foo"
    ser[mask] = "foo"
    # 断言操作未改变 ser 的数据类型，即不进行类型提升（no-op -> 不升级）
    assert ser.dtype == dti.dtype  # no-op -> dont upcast
    # 断言 ser 和 orig 两个 Series 对象相等，验证操作未影响数据内容
    tm.assert_series_equal(ser, orig)

    # 使用 mask 数组作为过滤条件，将 ser 中的元素根据条件设置为 "foo"，操作为原地修改
    ser.mask(mask, "foo", inplace=True)
    # 再次断言操作未改变 ser 的数据类型
    assert ser.dtype == dti.dtype  # no-op -> dont upcast
    # 再次验证 ser 和 orig 两个 Series 对象保持相等，确保操作未影响数据内容
    tm.assert_series_equal(ser, orig)
```