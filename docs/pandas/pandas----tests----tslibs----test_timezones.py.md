# `D:\src\scipysrc\pandas\pandas\tests\tslibs\test_timezones.py`

```
from datetime import (
    datetime,         # 导入 datetime 模块中的 datetime 类
    timedelta,        # 导入 datetime 模块中的 timedelta 类
    timezone,         # 导入 datetime 模块中的 timezone 类
)

import dateutil.tz     # 导入 dateutil 包中的 tz 模块
import pytest           # 导入 pytest 测试框架

from pandas._libs.tslibs import (
    conversion,       # 从 pandas._libs.tslibs 中导入 conversion 模块
    timezones,        # 从 pandas._libs.tslibs 中导入 timezones 模块
)
from pandas.compat import is_platform_windows  # 从 pandas.compat 中导入 is_platform_windows 函数

from pandas import Timestamp  # 导入 pandas 中的 Timestamp 类


def test_is_utc(utc_fixture):
    tz = timezones.maybe_get_tz(utc_fixture)  # 获取时区对象
    assert timezones.is_utc(tz)  # 断言时区对象是否为 UTC


def test_cache_keys_are_distinct_for_pytz_vs_dateutil():
    pytz = pytest.importorskip("pytz")  # 导入 pytz 包，如果不存在则跳过测试
    for tz_name in pytz.common_timezones:  # 遍历 pytz 包中的常见时区列表
        tz_p = timezones.maybe_get_tz(tz_name)  # 获取 pytz 时区对象
        tz_d = timezones.maybe_get_tz("dateutil/" + tz_name)  # 获取 dateutil 时区对象

    if tz_d is None:  # 如果 dateutil 时区对象不存在
        pytest.skip(tz_name + ": dateutil does not know about this one")  # 跳过测试

    if not (tz_name == "UTC" and is_platform_windows()):  # 如果不是在 Windows 平台并且时区名称不是 UTC
        # 它们在 Windows 上都变成 tzwin("UTC")
        assert timezones._p_tz_cache_key(tz_p) != timezones._p_tz_cache_key(tz_d)  # 断言两个时区对象的缓存键不相等


def test_tzlocal_repr():
    # see gh-13583
    ts = Timestamp("2011-01-01", tz=dateutil.tz.tzlocal())  # 创建一个带有本地时区的 Timestamp 对象
    assert ts.tz == dateutil.tz.tzlocal()  # 断言 Timestamp 对象的时区是本地时区
    assert "tz='tzlocal()')" in repr(ts)  # 断言在 Timestamp 对象的字符串表示中包含 "tz='tzlocal()')" 字符串


def test_tzlocal_maybe_get_tz():
    # see gh-13583
    tz = timezones.maybe_get_tz("tzlocal()")  # 获取 "tzlocal()" 对应的时区对象
    assert tz == dateutil.tz.tzlocal()  # 断言获取的时区对象是本地时区


def test_tzlocal_offset():
    # see gh-13583
    #
    # Get offset using normal datetime for test.
    ts = Timestamp("2011-01-01", tz=dateutil.tz.tzlocal())  # 创建一个带有本地时区的 Timestamp 对象

    offset = dateutil.tz.tzlocal().utcoffset(datetime(2011, 1, 1))  # 获取指定日期时间的本地时区偏移量
    offset = offset.total_seconds()  # 将偏移量转换为秒数

    assert ts._value + offset == Timestamp("2011-01-01")._value  # 断言带偏移量的 Timestamp 对象与未带偏移量的 Timestamp 对象相等


def test_tzlocal_is_not_utc():
    # even if the machine running the test is localized to UTC
    tz = dateutil.tz.tzlocal()  # 获取本地时区对象
    assert not timezones.is_utc(tz)  # 断言本地时区对象不是 UTC

    assert not timezones.tz_compare(tz, dateutil.tz.tzutc())  # 断言本地时区对象与 UTC 时区对象不相同


def test_tz_compare_utc(utc_fixture, utc_fixture2):
    tz = timezones.maybe_get_tz(utc_fixture)  # 获取第一个 UTC 时区对象
    tz2 = timezones.maybe_get_tz(utc_fixture2)  # 获取第二个 UTC 时区对象
    assert timezones.tz_compare(tz, tz2)  # 断言两个 UTC 时区对象相同


@pytest.fixture(
    params=[
        ("pytz/US/Eastern", lambda tz, x: tz.localize(x)),  # 参数化的 pytest fixture，使用 pytz 的本地化方法
        (dateutil.tz.gettz("US/Eastern"), lambda tz, x: x.replace(tzinfo=tz)),  # 参数化的 pytest fixture，使用 dateutil 的替换时区信息方法
    ]
)
def infer_setup(request):
    eastern, localize = request.param  # 获取参数化 fixture 的两个参数 eastern 和 localize

    if isinstance(eastern, str) and eastern.startswith("pytz/"):
        pytz = pytest.importorskip("pytz")  # 如果 eastern 是以 "pytz/" 开头的字符串，则导入 pytz 包
        eastern = pytz.timezone(eastern.removeprefix("pytz/"))  # 移除前缀后获取 pytz 的时区对象

    start_naive = datetime(2001, 1, 1)  # 创建一个 naive datetime 对象
    end_naive = datetime(2009, 1, 1)    # 创建另一个 naive datetime 对象

    start = localize(eastern, start_naive)  # 使用 localize 方法将 start_naive 本地化
    end = localize(eastern, end_naive)      # 使用 localize 方法将 end_naive 本地化

    return eastern, localize, start, end, start_naive, end_naive  # 返回 fixture 的参数化结果


def test_infer_tz_compat(infer_setup):
    eastern, _, start, end, start_naive, end_naive = infer_setup  # 解包 infer_setup fixture 的参数

    assert (
        timezones.infer_tzinfo(start, end)  # 推断 start 和 end 的时区信息
        is conversion.localize_pydatetime(start_naive, eastern).tzinfo  # 断言推断的时区信息与使用 pytz 的 localize 方法的结果相同
    )
    assert (
        timezones.infer_tzinfo(start, None)  # 推断 start 的时区信息
        is conversion.localize_pydatetime(start_naive, eastern).tzinfo  # 断言推断的时区信息与使用 pytz 的 localize 方法的结果相同
    )
    # 使用断言确保以下表达式为真：
    # 调用 timezones.infer_tzinfo 函数，以 None 和 end 作为参数，返回的时区信息
    # 应该等于调用 conversion.localize_pydatetime 函数，以 end_naive 和 eastern 作为参数，
    # 然后获取其 tzinfo 属性（即时区信息）后的结果。
    assert (
        timezones.infer_tzinfo(None, end)
        is conversion.localize_pydatetime(end_naive, eastern).tzinfo
    )
# 定义测试函数，用于验证时区推断函数 infer_tz_utc_localize 的正确性
def test_infer_tz_utc_localize(infer_setup):
    # 从 infer_setup 中获取起始和结束时间的 naive 版本
    _, _, start, end, start_naive, end_naive = infer_setup
    # 创建一个 UTC 时区对象
    utc = timezone.utc

    # 将起始时间和结束时间转换为 UTC 时区
    start = start_naive.astimezone(utc)
    end = end_naive.astimezone(utc)

    # 断言推断出的时区为 UTC
    assert timezones.infer_tzinfo(start, end) is utc


# 使用参数化装饰器定义测试函数，验证时区推断函数 infer_tz_mismatch 在时区不匹配时的行为
@pytest.mark.parametrize("ordered", [True, False])
def test_infer_tz_mismatch(infer_setup, ordered):
    # 从 infer_setup 中获取东部时区和其他变量
    eastern, _, _, _, start_naive, end_naive = infer_setup
    # 错误消息内容
    msg = "Inputs must both have the same timezone"

    # 创建一个 UTC 时区对象
    utc = timezone.utc
    # 将起始时间转换为 UTC 时区
    start = start_naive.astimezone(utc)
    # 将结束时间根据东部时区进行本地化
    end = conversion.localize_pydatetime(end_naive, eastern)

    # 根据 ordered 参数决定参数顺序
    args = (start, end) if ordered else (end, start)

    # 使用 pytest 断言引发 Assertion 错误，匹配错误消息
    with pytest.raises(AssertionError, match=msg):
        timezones.infer_tzinfo(*args)


# 定义测试函数，验证 maybe_get_tz 函数在传入无效类型时的异常处理
def test_maybe_get_tz_invalid_types():
    # 使用 pytest 断言引发 TypeError，匹配错误消息
    with pytest.raises(TypeError, match="<class 'float'>"):
        timezones.maybe_get_tz(44.0)

    # 使用 pytest 断言引发 TypeError，匹配错误消息
    with pytest.raises(TypeError, match="<class 'module'>"):
        timezones.maybe_get_tz(pytest)

    # 错误消息内容
    msg = "<class 'pandas._libs.tslibs.timestamps.Timestamp'>"
    # 使用 pytest 断言引发 TypeError，匹配错误消息
    with pytest.raises(TypeError, match=msg):
        timezones.maybe_get_tz(Timestamp("2021-01-01", tz="UTC"))


# 定义测试函数，验证 maybe_get_tz 函数在传入时区偏移字符串时的正确解析
def test_maybe_get_tz_offset_only():
    # 用例描述：见 gh-36004

    # 获取 UTC 时区对象
    tz = timezones.maybe_get_tz(timezone.utc)
    # 断言获取的时区对象与预期的 UTC 时区一致
    assert tz == timezone(timedelta(hours=0, minutes=0))

    # 测试不带 UTC+- 前缀的偏移字符串
    tz = timezones.maybe_get_tz("+01:15")
    assert tz == timezone(timedelta(hours=1, minutes=15))

    tz = timezones.maybe_get_tz("-01:15")
    assert tz == timezone(-timedelta(hours=1, minutes=15))

    # 测试带 UTC+- 前缀的偏移字符串
    tz = timezones.maybe_get_tz("UTC+02:45")
    assert tz == timezone(timedelta(hours=2, minutes=45))

    tz = timezones.maybe_get_tz("UTC-02:45")
    assert tz == timezone(-timedelta(hours=2, minutes=45))
```