# `D:\src\scipysrc\pandas\pandas\tests\tslibs\test_array_to_datetime.py`

```
from datetime import (
    date,                # 导入 date 类，处理日期
    datetime,            # 导入 datetime 类，处理日期和时间
    timedelta,           # 导入 timedelta 类，处理时间间隔
    timezone,            # 导入 timezone 类，处理时区
)

from dateutil.tz.tz import tzoffset   # 导入 tzoffset 类，处理时区偏移量
import numpy as np                     # 导入 NumPy 库，用于数值计算
import pytest                          # 导入 Pytest 测试框架

from pandas._libs import (             # 导入 Pandas 私有库中的模块
    NaT,                               # 导入 NaT（Not a Time）对象，表示缺失时间
    iNaT,                              # 导入 iNaT 对象，表示缺失的整数时间
    tslib,                             # 导入 tslib 模块，时间序列基础库
)
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit  # 导入 Pandas 中的时间类型

from pandas import Timestamp           # 导入 Pandas 中的 Timestamp 类，处理时间戳
import pandas._testing as tm           # 导入 Pandas 测试工具模块

creso_infer = NpyDatetimeUnit.NPY_FR_GENERIC.value  # 获取 NPY_FR_GENERIC 的值，用于时间推断


class TestArrayToDatetimeResolutionInference:
    # TODO: tests that include tzs, ints

    def test_infer_all_nat(self):
        arr = np.array([NaT, np.nan], dtype=object)  # 创建包含 NaT 和 NaN 的 NumPy 数组
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)  # 调用 tslib.array_to_datetime 方法推断时间
        assert tz is None  # 断言时区为 None
        assert result.dtype == "M8[s]"  # 断言结果数据类型为 "M8[s]"（秒级）

    def test_infer_homogeoneous_datetimes(self):
        dt = datetime(2023, 10, 27, 18, 3, 5, 678000)  # 创建 datetime 对象
        arr = np.array([dt, dt, dt], dtype=object)   # 创建包含 datetime 对象的 NumPy 数组
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)  # 调用 tslib.array_to_datetime 方法推断时间
        assert tz is None  # 断言时区为 None
        expected = np.array([dt, dt, dt], dtype="M8[us]")  # 创建预期的 NumPy 数组，精度为微秒
        tm.assert_numpy_array_equal(result, expected)  # 使用测试工具比较 result 和 expected

    def test_infer_homogeoneous_date_objects(self):
        dt = datetime(2023, 10, 27, 18, 3, 5, 678000)  # 创建 datetime 对象
        dt2 = dt.date()  # 获取 datetime 对象的日期部分
        arr = np.array([None, dt2, dt2, dt2], dtype=object)  # 创建包含 datetime.date 对象的 NumPy 数组
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)  # 调用 tslib.array_to_datetime 方法推断时间
        assert tz is None  # 断言时区为 None
        expected = np.array([np.datetime64("NaT"), dt2, dt2, dt2], dtype="M8[s]")  # 创建预期的 NumPy 数组，精度为秒
        tm.assert_numpy_array_equal(result, expected)  # 使用测试工具比较 result 和 expected

    def test_infer_homogeoneous_dt64(self):
        dt = datetime(2023, 10, 27, 18, 3, 5, 678000)  # 创建 datetime 对象
        dt64 = np.datetime64(dt, "ms")  # 创建 np.datetime64 对象，精度为毫秒
        arr = np.array([None, dt64, dt64, dt64], dtype=object)  # 创建包含 np.datetime64 对象的 NumPy 数组
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)  # 调用 tslib.array_to_datetime 方法推断时间
        assert tz is None  # 断言时区为 None
        expected = np.array([np.datetime64("NaT"), dt64, dt64, dt64], dtype="M8[ms]")  # 创建预期的 NumPy 数组，精度为毫秒
        tm.assert_numpy_array_equal(result, expected)  # 使用测试工具比较 result 和 expected

    def test_infer_homogeoneous_timestamps(self):
        dt = datetime(2023, 10, 27, 18, 3, 5, 678000)  # 创建 datetime 对象
        ts = Timestamp(dt).as_unit("ns")  # 创建 Timestamp 对象，精度为纳秒
        arr = np.array([None, ts, ts, ts], dtype=object)  # 创建包含 Timestamp 对象的 NumPy 数组
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)  # 调用 tslib.array_to_datetime 方法推断时间
        assert tz is None  # 断言时区为 None
        expected = np.array([np.datetime64("NaT")] + [ts.asm8] * 3, dtype="M8[ns]")  # 创建预期的 NumPy 数组，精度为纳秒
        tm.assert_numpy_array_equal(result, expected)  # 使用测试工具比较 result 和 expected

    def test_infer_homogeoneous_datetimes_strings(self):
        item = "2023-10-27 18:03:05.678000"  # 创建表示时间的字符串
        arr = np.array([None, item, item, item], dtype=object)  # 创建包含时间字符串的 NumPy 数组
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)  # 调用 tslib.array_to_datetime 方法推断时间
        assert tz is None  # 断言时区为 None
        expected = np.array([np.datetime64("NaT"), item, item, item], dtype="M8[us]")  # 创建预期的 NumPy 数组，精度为微秒
        tm.assert_numpy_array_equal(result, expected)  # 使用测试工具比较 result 和 expected
    # 定义一个测试方法，用于测试处理异构数据类型的推断功能
    def test_infer_heterogeneous(self):
        # 定义一个包含日期时间字符串的变量
        dtstr = "2023-10-27 18:03:05.678000"

        # 创建一个包含不同格式的日期时间字符串及空值的 NumPy 数组，数据类型为 object
        arr = np.array([dtstr, dtstr[:-3], dtstr[:-7], None], dtype=object)
        
        # 使用自定义函数 array_to_datetime 处理数组 arr，推断其日期时间，返回结果和时区信息
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
        
        # 断言时区 tz 应为 None
        assert tz is None
        
        # 期望的结果是一个 NumPy 数组，其日期时间数据类型为 "M8[us]"
        expected = np.array(arr, dtype="M8[us]")
        
        # 使用测试工具方法 assert_numpy_array_equal 检查 result 和 expected 数组是否相等
        tm.assert_numpy_array_equal(result, expected)

        # 对数组 arr 的逆序进行相同的日期时间推断处理
        result, tz = tslib.array_to_datetime(arr[::-1], creso=creso_infer)
        
        # 再次断言时区 tz 应为 None
        assert tz is None
        
        # 使用测试工具方法 assert_numpy_array_equal 检查 result 和 expected 数组的逆序是否相等
        tm.assert_numpy_array_equal(result, expected[::-1])

    # 使用 pytest 的参数化装饰器，定义一个测试方法，用于测试处理 NaN、NaT 等不同类型数据的推断功能
    @pytest.mark.parametrize(
        "item", [float("nan"), NaT.value, float(NaT.value), "NaT", ""]
    )
    def test_infer_with_nat_int_float_str(self, item):
        # 定义一个日期时间对象 dt
        dt = datetime(2023, 11, 15, 15, 5, 6)
        
        # 创建一个包含日期时间对象 dt 和参数化中的 item 的 NumPy 数组，数据类型为 object
        arr = np.array([dt, item], dtype=object)
        
        # 使用自定义函数 array_to_datetime 处理数组 arr，推断其日期时间，返回结果和时区信息
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
        
        # 断言时区 tz 应为 None
        assert tz is None
        
        # 期望的结果是一个 NumPy 数组，其中包含 dt 和特定类型的 NaT，数据类型为 "M8[us]"
        expected = np.array([dt, np.datetime64("NaT")], dtype="M8[us]")
        
        # 使用测试工具方法 assert_numpy_array_equal 检查 result 和 expected 数组是否相等
        tm.assert_numpy_array_equal(result, expected)

        # 对数组 arr 的逆序进行相同的日期时间推断处理
        result2, tz2 = tslib.array_to_datetime(arr[::-1], creso=creso_infer)
        
        # 再次断言时区 tz2 应为 None
        assert tz2 is None
        
        # 使用测试工具方法 assert_numpy_array_equal 检查 result2 和 expected 数组的逆序是否相等
        tm.assert_numpy_array_equal(result2, expected[::-1])
class TestArrayToDatetimeWithTZResolutionInference:
    def test_array_to_datetime_with_tz_resolution(self):
        # 创建自定义时区偏移为+3600秒的时区对象
        tz = tzoffset("custom", 3600)
        # 创建包含日期时间字符串和NaT值的NumPy对象数组
        vals = np.array(["2016-01-01 02:03:04.567", NaT], dtype=object)
        # 将数组转换为带有时区信息的日期时间数组，不进行转换，不强制本地化，推断时区分辨率
        res = tslib.array_to_datetime_with_tz(vals, tz, False, False, creso_infer)
        # 断言结果数组的数据类型为'M8[ms]'
        assert res.dtype == "M8[ms]"

        # 创建包含datetime对象和NaT值的NumPy对象数组
        vals2 = np.array([datetime(2016, 1, 1, 2, 3, 4), NaT], dtype=object)
        # 将数组转换为带有时区信息的日期时间数组，不进行转换，不强制本地化，推断时区分辨率
        res2 = tslib.array_to_datetime_with_tz(vals2, tz, False, False, creso_infer)
        # 断言结果数组的数据类型为'M8[us]'
        assert res2.dtype == "M8[us]"

        # 创建包含NaT值和具体日期时间的NumPy对象数组
        vals3 = np.array([NaT, np.datetime64(12345, "s")], dtype=object)
        # 将数组转换为带有时区信息的日期时间数组，不进行转换，不强制本地化，推断时区分辨率
        res3 = tslib.array_to_datetime_with_tz(vals3, tz, False, False, creso_infer)
        # 断言结果数组的数据类型为'M8[s]'
        assert res3.dtype == "M8[s]"

    def test_array_to_datetime_with_tz_resolution_all_nat(self):
        # 创建自定义时区偏移为+3600秒的时区对象
        tz = tzoffset("custom", 3600)
        # 创建包含"NaT"字符串的NumPy对象数组
        vals = np.array(["NaT"], dtype=object)
        # 将数组转换为带有时区信息的日期时间数组，不进行转换，不强制本地化，推断时区分辨率
        res = tslib.array_to_datetime_with_tz(vals, tz, False, False, creso_infer)
        # 断言结果数组的数据类型为'M8[s]'
        assert res.dtype == "M8[s]"

        # 创建包含两个NaT值的NumPy对象数组
        vals2 = np.array([NaT, NaT], dtype=object)
        # 将数组转换为带有时区信息的日期时间数组，不进行转换，不强制本地化，推断时区分辨率
        res2 = tslib.array_to_datetime_with_tz(vals2, tz, False, False, creso_infer)
        # 断言结果数组的数据类型为'M8[s]'
        assert res2.dtype == "M8[s]"


@pytest.mark.parametrize(
    "data,expected",
    [
        (
            ["01-01-2013", "01-02-2013"],
            [
                "2013-01-01T00:00:00.000000000",
                "2013-01-02T00:00:00.000000000",
            ],
        ),
        (
            ["Mon Sep 16 2013", "Tue Sep 17 2013"],
            [
                "2013-09-16T00:00:00.000000000",
                "2013-09-17T00:00:00.000000000",
            ],
        ),
    ],
)
def test_parsing_valid_dates(data, expected):
    # 创建包含日期字符串的NumPy对象数组
    arr = np.array(data, dtype=object)
    # 调用函数将数组转换为日期时间数组，返回结果和状态信息
    result, _ = tslib.array_to_datetime(arr)

    # 创建预期的日期时间NumPy对象数组，指定数据类型为'M8[s]'
    expected = np.array(expected, dtype="M8[s]")
    # 使用测试工具断言结果数组等于预期数组
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "dt_string, expected_tz",
    [
        ["01-01-2013 08:00:00+08:00", 480],
        ["2013-01-01T08:00:00.000000000+0800", 480],
        ["2012-12-31T16:00:00.000000000-0800", -480],
        ["12-31-2012 23:00:00-01:00", -60],
    ],
)
def test_parsing_timezone_offsets(dt_string, expected_tz):
    # 所有具有偏移量的日期时间字符串等同于在添加时区偏移后的相同日期时间
    arr = np.array(["01-01-2013 00:00:00"], dtype=object)
    # 调用函数将数组转换为日期时间数组，返回结果和状态信息
    expected, _ = tslib.array_to_datetime(arr)
    if "000000000" in dt_string:
        # 如果日期时间字符串包含纳秒精度，将预期结果转换为'M8[ns]'数据类型
        expected = expected.astype("M8[ns]")

    # 创建包含单个日期时间字符串的NumPy对象数组
    arr = np.array([dt_string], dtype=object)
    # 调用函数将数组转换为日期时间数组和时区信息，返回结果和时区
    result, result_tz = tslib.array_to_datetime(arr)

    # 使用测试工具断言结果数组等于预期数组
    tm.assert_numpy_array_equal(result, expected)
    # 断言结果的时区等于预期的时区对象，该时区偏移为分钟形式
    assert result_tz == timezone(timedelta(minutes=expected_tz))


def test_parsing_non_iso_timezone_offset():
    # 创建包含特定日期时间字符串的NumPy对象数组
    dt_string = "01-01-2013T00:00:00.000000000+0000"
    arr = np.array([dt_string], dtype=object)
    # 使用 pytest 中的 assert_produces_warning 上下文管理器，忽略特定类型的警告
    with tm.assert_produces_warning(None):
        # 调用 tslib 模块的 array_to_datetime 函数，将 arr 转换为 datetime 对象
        # 返回结果包括 result 和 result_tz
        result, result_tz = tslib.array_to_datetime(arr)
    
    # 期望的结果是一个包含特定日期时间的 numpy 数组
    expected = np.array([np.datetime64("2013-01-01 00:00:00.000000000")])
    
    # 使用 pytest 的 assert_numpy_array_equal 函数验证 result 与 expected 数组相等
    tm.assert_numpy_array_equal(result, expected)
    
    # 使用标准的 Python assert 语句，确保 result_tz 是 timezone.utc
    assert result_tz is timezone.utc
# 测试解析不同时区偏移的功能
def test_parsing_different_timezone_offsets():
    # 使用 'see gh-17697' 作为测试的引用信息
    data = ["2015-11-18 15:30:00+05:30", "2015-11-18 15:30:00+06:30"]
    data = np.array(data, dtype=object)

    # 在转换数据为日期时间时，如果检测到混合的时区，则抛出 ValueError 异常
    msg = "Mixed timezones detected. Pass utc=True in to_datetime"
    with pytest.raises(ValueError, match=msg):
        tslib.array_to_datetime(data)


# 参数化测试函数，测试对于超出纳秒边界的日期时间对象的强制转换
@pytest.mark.parametrize(
    "invalid_date,exp_unit",
    [
        (date(1000, 1, 1), "s"),
        (datetime(1000, 1, 1), "us"),
        ("1000-01-01", "s"),
        ("Jan 1, 1000", "s"),
        (np.datetime64("1000-01-01"), "s"),
    ],
)
@pytest.mark.parametrize("errors", ["coerce", "raise"])
def test_coerce_outside_ns_bounds(invalid_date, exp_unit, errors):
    arr = np.array([invalid_date], dtype="object")

    # 将数组转换为日期时间对象，根据 errors 参数处理无效日期
    result, _ = tslib.array_to_datetime(arr, errors=errors)

    # 检查输出的日期时间对象的分辨率单位是否与预期一致
    out_reso = np.datetime_data(result.dtype)[0]
    assert out_reso == exp_unit

    # 创建 Timestamp 对象，并检查其时间单位是否符合预期
    ts = Timestamp(invalid_date)
    assert ts.unit == exp_unit

    # 创建预期结果的数组，并使用 tm.assert_numpy_array_equal 进行比较
    expected = np.array([ts._value], dtype=f"M8[{exp_unit}]")
    tm.assert_numpy_array_equal(result, expected)


# 测试在超出纳秒边界的情况下，只有一个有效日期的强制转换行为
def test_coerce_outside_ns_bounds_one_valid():
    arr = np.array(["1/1/1000", "1/1/2000"], dtype=object)

    # 使用 'coerce' 错误处理模式将数组转换为日期时间对象
    result, _ = tslib.array_to_datetime(arr, errors="coerce")

    # 创建预期结果的数组，并使用 tm.assert_numpy_array_equal 进行比较
    expected = ["1000-01-01T00:00:00.000000000", "2000-01-01T00:00:00.000000000"]
    expected = np.array(expected, dtype="M8[s]")
    tm.assert_numpy_array_equal(result, expected)


# 测试对于无效日期时间的强制转换行为
def test_coerce_of_invalid_datetimes():
    arr = np.array(["01-01-2013", "not_a_date", "1"], dtype=object)

    # 使用 'coerce' 错误处理模式将数组转换为日期时间对象
    result, _ = tslib.array_to_datetime(arr, errors="coerce")

    # 预期结果数组，包含一个有效日期和两个无效日期（转换为 iNaT）
    expected = ["2013-01-01T00:00:00.000000000", iNaT, iNaT]
    tm.assert_numpy_array_equal(result, np.array(expected, dtype="M8[s]"))


# 测试接近边界的日期时间对象转换，应该抛出 OutOfBoundsDatetime 异常
def test_to_datetime_barely_out_of_bounds():
    # 使用 'see gh-19382, gh-19529' 作为测试的引用信息
    #
    # 创建一个接近边界但超出纳秒边界的日期时间数组，并检查是否抛出预期的异常信息
    arr = np.array(["2262-04-11 23:47:16.854775808"], dtype=object)
    msg = "^Out of bounds nanosecond timestamp: 2262-04-11 23:47:16, at position 0$"

    with pytest.raises(tslib.OutOfBoundsDatetime, match=msg):
        tslib.array_to_datetime(arr)


# 参数化测试函数，测试接近边界但仍在纳秒边界内的日期时间对象转换
@pytest.mark.parametrize(
    "timestamp",
    [
        # 接近边界但缩放到纳秒会溢出，添加纳秒后会在边界内的时间戳
        "1677-09-21T00:12:43.145224193",
        "1677-09-21T00:12:43.145224999",
        # 总是有效的时间戳
        "1677-09-21T00:12:43.145225000",
    ],
)
def test_to_datetime_barely_inside_bounds(timestamp):
    # 使用 'see gh-57150' 作为测试的引用信息
    #
    # 创建一个接近边界但仍在纳秒边界内的日期时间数组，并进行转换
    result, _ = tslib.array_to_datetime(np.array([timestamp], dtype=object))
    # 使用测试工具 `tm` 的方法来断言 `result` 是否与给定的 Numpy 数组相等，该数组包含一个时间戳 `timestamp`，
    # 数据类型为日期时间类型 "M8[ns]"
    tm.assert_numpy_array_equal(result, np.array([timestamp], dtype="M8[ns]"))
class SubDatetime(datetime):
    pass

# 创建一个名为 SubDatetime 的类，该类继承自 datetime 类


@pytest.mark.parametrize("klass", [SubDatetime, datetime, Timestamp])

# 使用 pytest 的 parametrize 装饰器，定义了一个参数化测试，参数名为 klass，参数值为 SubDatetime 类、datetime 类和 Timestamp 类的列表


def test_datetime_subclass(klass):

# 定义名为 test_datetime_subclass 的测试函数，接受一个 klass 参数，用于测试不同的 datetime 子类


# GH 25851
# 确保子类化的 datetime 在 array_to_datetime 函数中能够正常工作

arr = np.array([klass(2000, 1, 1)], dtype=object)

# 创建一个包含一个 klass 类型对象的 numpy 数组 arr，对象的值为 klass 类的一个实例，表示日期为 2000 年 1 月 1 日


result, _ = tslib.array_to_datetime(arr)

# 调用 tslib 模块中的 array_to_datetime 函数，将数组 arr 转换为 datetime 数组，将结果存储在 result 变量中


expected = np.array(["2000-01-01T00:00:00.000000"], dtype="M8[us]")

# 创建一个预期的 numpy 数组 expected，包含一个字符串，表示预期的日期时间为 "2000-01-01T00:00:00.000000"，数据类型为微秒精度的 datetime


tm.assert_numpy_array_equal(result, expected)

# 使用 tm 模块中的 assert_numpy_array_equal 函数，断言 result 和 expected 两个 numpy 数组相等，用于验证测试的正确性
```