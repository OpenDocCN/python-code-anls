# `D:\src\scipysrc\pandas\pandas\tests\tslibs\test_conversion.py`

```
# 从 datetime 模块导入 datetime 和 timezone 类
from datetime import (
    datetime,
    timezone,
)

# 导入 numpy 库，并使用别名 np
import numpy as np

# 导入 pytest 库
import pytest

# 从 pandas._libs.tslibs 模块导入多个类和函数
from pandas._libs.tslibs import (
    OutOfBoundsTimedelta,
    astype_overflowsafe,
    conversion,
    iNaT,
    timezones,
    tz_convert_from_utc,
    tzconversion,
)

# 从 pandas 模块导入 Timestamp 和 date_range 函数
from pandas import (
    Timestamp,
    date_range,
)

# 导入 pandas._testing 模块，并使用别名 tm
import pandas._testing as tm


# 定义内部函数 _compare_utc_to_local，用于比较 UTC 时间与本地时间的转换
def _compare_utc_to_local(tz_didx):
    def f(x):
        return tzconversion.tz_convert_from_utc_single(x, tz_didx.tz)

    # 使用 tz_convert_from_utc 函数将时间从 UTC 转换为本地时间
    result = tz_convert_from_utc(tz_didx.asi8, tz_didx.tz)
    # 使用 numpy 的 vectorize 函数将 f 应用于所有输入，得到期望的结果
    expected = np.vectorize(f)(tz_didx.asi8)

    # 使用 pandas 测试工具中的 assert_numpy_array_equal 函数比较两个数组是否相等
    tm.assert_numpy_array_equal(result, expected)


# 定义内部函数 _compare_local_to_utc，用于比较本地时间与 UTC 时间的转换
def _compare_local_to_utc(tz_didx, naive_didx):
    # 检查 tz_localize 函数的向量化和逐点行为是否相同
    err1 = err2 = None
    try:
        # 将本地时间转换为 UTC 时间
        result = tzconversion.tz_localize_to_utc(naive_didx.asi8, tz_didx.tz)
        err1 = None
    except Exception as err:
        err1 = err

    try:
        # 使用 map 函数将本地时间标记为 tz_didx.tz 时区
        expected = naive_didx.map(lambda x: x.tz_localize(tz_didx.tz)).asi8
    except Exception as err:
        err2 = err

    # 检查错误类型是否相同，如果 err1 不为 None，则 err2 必须为 None
    if err1 is not None:
        assert type(err1) == type(err2)
    else:
        assert err2 is None
        # 使用 pandas 测试工具中的 assert_numpy_array_equal 函数比较两个数组是否相等
        tm.assert_numpy_array_equal(result, expected)


# 定义测试函数 test_tz_localize_to_utc_copies，用于测试时区转换到 UTC 时间的拷贝行为
def test_tz_localize_to_utc_copies():
    # GH#46460
    # 创建一个包含五个整数的 numpy 数组
    arr = np.arange(5, dtype="i8")
    # 将数组从 UTC 时间转换为本地时间
    result = tz_convert_from_utc(arr, tz=timezone.utc)
    # 使用 pandas 测试工具中的 assert_numpy_array_equal 函数比较两个数组是否相等
    tm.assert_numpy_array_equal(result, arr)
    # 检查 arr 和 result 是否共享内存
    assert not np.shares_memory(arr, result)

    # 将数组从 UTC 时间转换为无时区时间
    result = tz_convert_from_utc(arr, tz=None)
    # 使用 pandas 测试工具中的 assert_numpy_array_equal 函数比较两个数组是否相等
    tm.assert_numpy_array_equal(result, arr)
    # 检查 arr 和 result 是否共享内存
    assert not np.shares_memory(arr, result)


# 定义测试函数 test_tz_convert_single_matches_tz_convert_hourly，用于测试单个时区转换与每小时时区转换的匹配性
def test_tz_convert_single_matches_tz_convert_hourly(tz_aware_fixture):
    tz = tz_aware_fixture
    # 创建一个包含每小时时区信息的日期范围
    tz_didx = date_range("2014-03-01", "2015-01-10", freq="h", tz=tz)
    # 创建一个包含每小时日期范围的无时区日期范围
    naive_didx = date_range("2014-03-01", "2015-01-10", freq="h")

    # 比较 UTC 时间到本地时间的转换
    _compare_utc_to_local(tz_didx)
    # 比较本地时间到 UTC 时间的转换
    _compare_local_to_utc(tz_didx, naive_didx)


# 定义测试函数 test_tz_convert_single_matches_tz_convert，用于测试单个时区转换与每日和每年时区转换的匹配性
@pytest.mark.parametrize("freq", ["D", "YE"])
def test_tz_convert_single_matches_tz_convert(tz_aware_fixture, freq):
    tz = tz_aware_fixture
    # 创建一个指定频率的时区日期范围
    tz_didx = date_range("2018-01-01", "2020-01-01", freq=freq, tz=tz)
    # 创建一个指定频率的无时区日期范围
    naive_didx = date_range("2018-01-01", "2020-01-01", freq=freq)

    # 比较 UTC 时间到本地时间的转换
    _compare_utc_to_local(tz_didx)
    # 比较本地时间到 UTC 时间的转换
    _compare_local_to_utc(tz_didx, naive_didx)


# 定义测试函数 test_tz_convert_corner，用于测试极端情况下的时区转换
@pytest.mark.parametrize(
    "arr",
    [
        pytest.param([], id="empty"),
        pytest.param([iNaT], id="all_nat"),
    ],
)
def test_tz_convert_corner(arr):
    # 创建一个包含 iNaT 的 numpy 数组
    arr = np.array([iNaT], dtype=np.int64)
    # 将数组从 UTC 时间转换为亚洲/东京时区的本地时间
    result = tz_convert_from_utc(arr, timezones.maybe_get_tz("Asia/Tokyo"))
    # 使用 pandas 测试工具中的 assert_numpy_array_equal 函数比较两个数组是否相等
    tm.assert_numpy_array_equal(result, arr)


# 定义测试函数 test_tz_convert_readonly，用于测试只读数组的时区转换行为
def test_tz_convert_readonly():
    # GH#35530
    # 创建一个包含单个整数的只读 numpy 数组
    arr = np.array([0], dtype=np.int64)
    # 设置数组为只读状态
    arr.setflags(write=False)
    # 将数组从 UTC 时间转换为本地时间
    result = tz_convert_from_utc(arr, timezone.utc)
    # 使用 pandas 测试工具中的 assert_numpy_array_equal 函数比较两个数组是否相等
    tm.assert_numpy_array_equal(result, arr)


# 定义测试函数 test_length_zero_copy，用于测试长度为零的数组拷贝行为
@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("dtype", ["M8[ns]", "M8[s]"])
def test_length_zero_copy(dtype, copy):
    # 创建一个空的 NumPy 数组 `arr`，指定数据类型为给定的 `dtype`
    arr = np.array([], dtype=dtype)
    # 调用函数 `astype_overflowsafe`，将数组 `arr` 转换为指定的数据类型 `np.dtype("M8[ns]")`
    result = astype_overflowsafe(arr, copy=copy, dtype=np.dtype("M8[ns]"))
    # 如果设置了 `copy` 参数为 True，则确保结果 `result` 和原数组 `arr` 不共享内存
    if copy:
        assert not np.shares_memory(result, arr)
    # 如果 `copy` 参数为 False，并且转换后的数组 `result` 的数据类型与 `arr` 相同，则断言 `result` 就是 `arr`
    elif arr.dtype == result.dtype:
        assert result is arr
    # 否则，确保转换后的数组 `result` 和原数组 `arr` 不共享内存
    else:
        assert not np.shares_memory(result, arr)
# 定义一个测试函数，用于确保转换为大端序的 datetime64[ns] 类型安全
def test_ensure_datetime64ns_bigendian():
    # 创建一个 numpy 数组，包含一个以毫秒为单位的 datetime64 元素，且数据类型为 '>M8[ms]'
    arr = np.array([np.datetime64(1, "ms")], dtype=">M8[ms]")
    # 调用 astype_overflowsafe 函数，将数组元素转换为 dtype='M8[ns]' 类型，确保安全性
    result = astype_overflowsafe(arr, dtype=np.dtype("M8[ns]"))

    # 预期结果是一个包含一个以纳秒为单位的 datetime64 元素的 numpy 数组
    expected = np.array([np.datetime64(1, "ms")], dtype="M8[ns]")
    # 使用 assert_numpy_array_equal 断言检查 result 是否等于 expected
    tm.assert_numpy_array_equal(result, expected)


# 定义一个测试函数，用于确保 timedelta64[ns] 类型的转换不会溢出
def test_ensure_timedelta64ns_overflows():
    # 创建一个包含十个元素的 numpy 数组，元素类型为 'm8[Y]'，表示年的 timedelta64 类型，每个元素乘以 100
    arr = np.arange(10).astype("m8[Y]") * 100
    # 定义异常消息，用于检查是否会发生 timedelta64[ns] 类型的溢出
    msg = r"Cannot convert 300 years to timedelta64\[ns\] without overflow"
    # 使用 pytest.raises 检查调用 astype_overflowsafe 函数时是否会抛出 OutOfBoundsTimedelta 异常，并匹配异常消息
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        astype_overflowsafe(arr, dtype=np.dtype("m8[ns]"))


# 创建一个 datetime 的子类，用于测试子类化后的 datetime 对象是否能正常使用 localize_pydatetime 函数
class SubDatetime(datetime):
    pass


# 使用 pytest.mark.parametrize 注册一个参数化测试函数，用于测试不同类型的日期时间对象在使用 localize_pydatetime 函数时的行为
@pytest.mark.parametrize(
    "dt, expected",
    [
        # 参数化测试用例：测试 Timestamp 对象
        pytest.param(
            Timestamp("2000-01-01"),
            Timestamp("2000-01-01", tz=timezone.utc),
            id="timestamp",
        ),
        # 参数化测试用例：测试标准 datetime 对象
        pytest.param(
            datetime(2000, 1, 1),
            datetime(2000, 1, 1, tzinfo=timezone.utc),
            id="datetime",
        ),
        # 参数化测试用例：测试 SubDatetime 类的对象
        pytest.param(
            SubDatetime(2000, 1, 1),
            SubDatetime(2000, 1, 1, tzinfo=timezone.utc),
            id="subclassed_datetime",
        ),
    ],
)
# 定义参数化测试函数，用于测试 localize_pydatetime 函数是否能正确处理不同类型的日期时间对象
def test_localize_pydatetime_dt_types(dt, expected):
    # GH 25851
    # 测试案例：确保子类化的 datetime 对象在使用 localize_pydatetime 函数时能正常工作
    result = conversion.localize_pydatetime(dt, timezone.utc)
    # 使用 assert 断言检查结果是否与预期值 expected 相等
    assert result == expected
```