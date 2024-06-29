# `D:\src\scipysrc\pandas\pandas\tests\tslibs\test_parsing.py`

```
"""
Tests for Timestamp parsing, aimed at pandas/_libs/tslibs/parsing.pyx
"""

# 导入所需的模块和库
from datetime import datetime  # 导入 datetime 模块中的 datetime 类
import re  # 导入正则表达式模块

from dateutil.parser import parse as du_parse  # 导入 dateutil 库中的 parse 函数并重命名为 du_parse
from hypothesis import given  # 导入 hypothesis 库中的 given 函数
import numpy as np  # 导入 numpy 库并重命名为 np
import pytest  # 导入 pytest 测试框架

from pandas._libs.tslibs import (  # 从 pandas._libs.tslibs 中导入以下模块：
    parsing,  # parsing 模块
    strptime,  # strptime 模块
)
from pandas._libs.tslibs.parsing import parse_datetime_string_with_reso  # 从 parsing 模块导入 parse_datetime_string_with_reso 函数
from pandas.compat import (  # 从 pandas.compat 中导入以下模块：
    ISMUSL,  # ISMUSL 常量
    WASM,  # WASM 常量
    is_platform_windows,  # is_platform_windows 函数
)
import pandas.util._test_decorators as td  # 导入 pandas.util._test_decorators 模块并重命名为 td

# Usually we wouldn't want this import in this test file (which is targeted at
#  tslibs.parsing), but it is convenient to test the Timestamp constructor at
#  the same time as the other parsing functions.
from pandas import Timestamp  # 导入 pandas 库中的 Timestamp 类
import pandas._testing as tm  # 导入 pandas._testing 模块
from pandas._testing._hypothesis import DATETIME_NO_TZ  # 导入 DATETIME_NO_TZ 常量

# 使用 pytest 标记测试用例，如果 WASM 为 True，则跳过测试
@pytest.mark.skipif(WASM, reason="tzset is not available on WASM")
# 使用 pytest 标记测试用例，如果运行平台为 Windows 或者 ISMUSL 为 True，则跳过测试
@pytest.mark.skipif(
    is_platform_windows() or ISMUSL,
    reason="TZ setting incorrect on Windows and MUSL Linux",
)
def test_parsing_tzlocal_deprecated():
    # GH#50791
    msg = (
        r"Parsing 'EST' as tzlocal \(dependent on system timezone\) "
        r"is no longer supported\. "
        "Pass the 'tz' keyword or call tz_localize after construction instead"
    )
    dtstr = "Jan 15 2004 03:00 EST"

    # 在 'US/Eastern' 时区下执行以下代码块
    with tm.set_timezone("US/Eastern"):
        # 使用 pytest 检查 parse_datetime_string_with_reso 函数是否会引发 ValueError，并验证错误消息是否匹配
        with pytest.raises(ValueError, match=msg):
            parse_datetime_string_with_reso(dtstr)

        # 使用 pytest 检查 parsing.py_parse_datetime_string 函数是否会引发 ValueError，并验证错误消息是否匹配
        with pytest.raises(ValueError, match=msg):
            parsing.py_parse_datetime_string(dtstr)

        # 使用 pytest 检查 Timestamp 构造函数是否会引发 ValueError，并验证错误消息是否匹配
        with pytest.raises(ValueError, match=msg):
            Timestamp(dtstr)


def test_parse_datetime_string_with_reso():
    # 测试 parse_datetime_string_with_reso 函数是否能正确解析 "4Q1984" 和 "4q1984"
    (parsed, reso) = parse_datetime_string_with_reso("4Q1984")
    (parsed_lower, reso_lower) = parse_datetime_string_with_reso("4q1984")

    # 断言两次调用的解析结果和分辨率相同
    assert reso == reso_lower
    assert parsed == parsed_lower


def test_parse_datetime_string_with_reso_nanosecond_reso():
    # GH#46811
    # 测试 parse_datetime_string_with_reso 函数是否能正确解析带有纳秒分辨率的日期字符串
    parsed, reso = parse_datetime_string_with_reso("2022-04-20 09:19:19.123456789")
    assert reso == "nanosecond"  # 断言分辨率为 "nanosecond"


def test_parse_datetime_string_with_reso_invalid_type():
    # 测试 parse_datetime_string_with_reso 函数在接收到无效输入时是否会引发 TypeError
    msg = "Argument 'date_string' has incorrect type (expected str, got tuple)"
    with pytest.raises(TypeError, match=re.escape(msg)):
        parse_datetime_string_with_reso((4, 5))


# 使用 pytest 参数化装饰器，对 test_parse_time_quarter_with_dash 函数进行多组参数化测试
@pytest.mark.parametrize(
    "dashed,normal", [("1988-Q2", "1988Q2"), ("2Q-1988", "2Q1988")]
)
def test_parse_time_quarter_with_dash(dashed, normal):
    # see gh-9688
    # 测试 parse_datetime_string_with_reso 函数对带有连字符和不带连字符的季度日期字符串的解析
    (parsed_dash, reso_dash) = parse_datetime_string_with_reso(dashed)
    (parsed, reso) = parse_datetime_string_with_reso(normal)

    # 断言两种格式的解析结果和分辨率相同
    assert parsed_dash == parsed
    assert reso_dash == reso


# 使用 pytest 参数化装饰器，对 test_parse_time_quarter_with_dash_error 函数进行多组参数化测试
@pytest.mark.parametrize("dashed", ["-2Q1992", "2-Q1992", "4-4Q1992"])
def test_parse_time_quarter_with_dash_error(dashed):
    # 测试 parse_datetime_string_with_reso 函数在接收到不支持的季度日期字符串时是否会产生预期的错误消息
    msg = f"Unknown datetime string format, unable to parse: {dashed}"
    # 使用 pytest 框架中的 `raises` 上下文管理器，期望捕获 `parsing.DateParseError` 异常，并验证异常消息与 `msg` 匹配。
    with pytest.raises(parsing.DateParseError, match=msg):
        # 调用 `parse_datetime_string_with_reso` 函数，尝试解析 `dashed` 变量所代表的日期时间字符串。
        parse_datetime_string_with_reso(dashed)
@pytest.mark.parametrize(
    "date_string,expected",
    [
        ("123.1234", False),  # 不是有效的日期格式，期望返回 False
        ("-50000", False),    # 不是有效的日期格式，期望返回 False
        ("999", False),       # 不是有效的日期格式，期望返回 False
        ("m", False),         # 不是有效的日期格式，期望返回 False
        ("T", False),         # 不是有效的日期格式，期望返回 False
        ("Mon Sep 16, 2013", True),  # 是有效的日期格式，期望返回 True
        ("2012-01-01", True),        # 是有效的日期格式，期望返回 True
        ("01/01/2012", True),        # 是有效的日期格式，期望返回 True
        ("01012012", True),          # 是有效的日期格式，期望返回 True
        ("0101", True),              # 是有效的日期格式，期望返回 True
        ("1-1", True),               # 是有效的日期格式，期望返回 True
    ],
)
def test_does_not_convert_mixed_integer(date_string, expected):
    assert parsing._does_string_look_like_datetime(date_string) is expected


@pytest.mark.parametrize(
    "date_str,kwargs,msg",
    [
        (
            "2013Q5",
            {},
            (
                "Incorrect quarterly string is given, "
                "quarter must be between 1 and 4: 2013Q5"
            ),
        ),
        # see gh-5418
        (
            "2013Q1",
            {"freq": "INVLD-L-DEC-SAT"},
            (
                "Unable to retrieve month information "
                "from given freq: INVLD-L-DEC-SAT"
            ),
        ),
    ],
)
def test_parsers_quarterly_with_freq_error(date_str, kwargs, msg):
    with pytest.raises(parsing.DateParseError, match=msg):
        parsing.parse_datetime_string_with_reso(date_str, **kwargs)


@pytest.mark.parametrize(
    "date_str,freq,expected",
    [
        ("2013Q2", None, datetime(2013, 4, 1)),   # 解析季度格式，预期结果为指定日期
        ("2013Q2", "Y-APR", datetime(2012, 8, 1)),  # 解析季度格式，根据频率返回指定日期
        ("2013-Q2", "Y-DEC", datetime(2013, 4, 1)),  # 解析季度格式，根据频率返回指定日期
    ],
)
def test_parsers_quarterly_with_freq(date_str, freq, expected):
    result, _ = parsing.parse_datetime_string_with_reso(date_str, freq=freq)
    assert result == expected


@pytest.mark.parametrize(
    "date_str", ["2Q 2005", "2Q-200Y", "2Q-200", "22Q2005", "2Q200.", "6Q-20"]
)
def test_parsers_quarter_invalid(date_str):
    if date_str == "6Q-20":
        msg = (
            "Incorrect quarterly string is given, quarter "
            f"must be between 1 and 4: {date_str}"
        )
    else:
        msg = f"Unknown datetime string format, unable to parse: {date_str}"

    with pytest.raises(ValueError, match=msg):
        parsing.parse_datetime_string_with_reso(date_str)


@pytest.mark.parametrize(
    "date_str,expected",
    [("201101", datetime(2011, 1, 1, 0, 0)), ("200005", datetime(2000, 5, 1, 0, 0))],
)
def test_parsers_month_freq(date_str, expected):
    result, _ = parsing.parse_datetime_string_with_reso(date_str, freq="ME")
    assert result == expected


@td.skip_if_not_us_locale
@pytest.mark.parametrize(
    "string,fmt",
    # 定义一个包含日期时间字符串和它们对应格式的元组列表
    [
        ("20111230", "%Y%m%d"),  # 格式化日期，例如 2011 年 12 月 30 日
        ("201112300000", "%Y%m%d%H%M"),  # 格式化日期和时间，例如 2011 年 12 月 30 日 00:00
        ("20111230000000", "%Y%m%d%H%M%S"),  # 格式化日期和详细时间，例如 2011 年 12 月 30 日 00:00:00
        ("20111230T00", "%Y%m%dT%H"),  # 格式化带 'T' 的日期和小时，例如 2011 年 12 月 30 日 00 时
        ("20111230T0000", "%Y%m%dT%H%M"),  # 格式化带 'T' 的日期和小时分钟，例如 2011 年 12 月 30 日 00 时 00 分
        ("20111230T000000", "%Y%m%dT%H%M%S"),  # 格式化带 'T' 的日期和完整时间，例如 2011 年 12 月 30 日 00 时 00 分 00 秒
        ("2011-12-30", "%Y-%m-%d"),  # 格式化带破折号的日期，例如 2011 年 12 月 30 日
        ("2011", "%Y"),  # 格式化年份，例如 2011 年
        ("2011-01", "%Y-%m"),  # 格式化带破折号的年份和月份，例如 2011 年 01 月
        ("30-12-2011", "%d-%m-%Y"),  # 格式化日-月-年，例如 30 日 12 月 2011 年
        ("2011-12-30 00:00:00", "%Y-%m-%d %H:%M:%S"),  # 格式化日期和时间，例如 2011 年 12 月 30 日 00 时 00 分 00 秒
        ("2011-12-30T00:00:00", "%Y-%m-%dT%H:%M:%S"),  # 格式化带 'T' 的日期和时间，例如 2011 年 12 月 30 日 00 时 00 分 00 秒
        ("2011-12-30T00:00:00UTC", "%Y-%m-%dT%H:%M:%S%Z"),  # 格式化带 'T' 的日期时间和时区，例如 2011 年 12 月 30 日 00 时 00 分 00 秒 UTC
        ("2011-12-30T00:00:00Z", "%Y-%m-%dT%H:%M:%S%z"),  # 格式化带 'T' 的日期时间和时区，例如 2011 年 12 月 30 日 00 时 00 分 00 秒 Z 时区
        ("2011-12-30T00:00:00+9", "%Y-%m-%dT%H:%M:%S%z"),  # 格式化带 'T' 的日期时间和时区，例如 2011 年 12 月 30 日 00 时 00 分 00 秒 +9 时区
        ("2011-12-30T00:00:00+09", "%Y-%m-%dT%H:%M:%S%z"),  # 格式化带 'T' 的日期时间和时区，例如 2011 年 12 月 30 日 00 时 00 分 00 秒 +09 时区
        ("2011-12-30T00:00:00+090", None),  # 无效的日期格式
        ("2011-12-30T00:00:00+0900", "%Y-%m-%dT%H:%M:%S%z"),  # 格式化带 'T' 的日期时间和时区，例如 2011 年 12 月 30 日 00 时 00 分 00 秒 +0900 时区
        ("2011-12-30T00:00:00-0900", "%Y-%m-%dT%H:%M:%S%z"),  # 格式化带 'T' 的日期时间和时区，例如 2011 年 12 月 30 日 00 时 00 分 00 秒 -0900 时区
        ("2011-12-30T00:00:00+09:00", "%Y-%m-%dT%H:%M:%S%z"),  # 格式化带 'T' 的日期时间和时区，例如 2011 年 12 月 30 日 00 时 00 分 00 秒 +09:00 时区
        ("2011-12-30T00:00:00+09:000", None),  # 无效的日期格式
        ("2011-12-30T00:00:00+9:0", "%Y-%m-%dT%H:%M:%S%z"),  # 格式化带 'T' 的日期时间和时区，例如 2011 年 12 月 30 日 00 时 00 分 00 秒 +9:0 时区
        ("2011-12-30T00:00:00+09:", None),  # 无效的日期格式
        ("2011-12-30T00:00:00.000000UTC", "%Y-%m-%dT%H:%M:%S.%f%Z"),  # 格式化带 'T' 的日期时间、微秒和时区，例如 2011 年 12 月 30 日 00 时 00 分 00 秒 000000 微秒 UTC
        ("2011-12-30T00:00:00.000000Z", "%Y-%m-%dT%H:%M:%S.%f%z"),  # 格式化带 'T' 的日期时间、微秒和时区，例如 2011 年 12 月 30 日 00 时 00 分 00 秒 000000 微秒 Z 时区
        ("2011-12-30T00:00:00.000000+9", "%Y-%m-%dT%H:%M:%S.%f%z"),  # 格式化带 'T' 的日期时间、微秒和时区，例如 2011 年 12 月 30 日 00 时 00 分 00 秒 000000 微秒 +9 时区
        ("2011-12-30T00:00:00.000000+09", "%Y-%m-%dT%H:%M:%S.%f%z"),  # 格式化带 'T' 的日期时间、微秒和时区，例如 2011 年 12 月 30 日 00 时 00 分 00 秒 000000 微秒 +09 时区
        ("2011-12-30T00:00:00.000000+090", None),  # 无效的日期格式
        ("2011-12-30T00:00:00.000000+0900", "%Y-%m-%dT%H:%M:%S.%f%z"),  # 格式化带 'T' 的日期时间、微秒和时区，例如 2011 年 12 月 30 日 00 时 00 分 00 秒 000000 微秒 +0900 时区
        ("2011-12-30T00:00:00.000000-0900", "%Y-%m-%dT%H:%M:%S.%f%z"),  # 格式化带 'T' 的日期时间、微秒和时区，例如 2011 年 12 月 30 日 00 时 00 分 00 秒 000000 微秒 -0900 时区
        ("2011-12-30T00:00:00.000000+09:00", "%Y-%m-%dT%H:%M:%S.%f%z"),  # 格式化带 'T' 的日期时间、微秒和时区，例如 2011 年 12 月 30 日 00 时 00 分 00 秒 000000 微秒 +09:00 时区
        ("2011-12-30T00:00:00.000000+09:000", None),  # 无效的日期格式
        ("2011-12
def test_guess_datetime_format_with_parseable_formats(string, fmt):
    # 使用 pytest 的装饰器 mark.parametrize 提供多组输入参数来执行测试
    with tm.maybe_produces_warning(
        UserWarning, fmt is not None and re.search(r"%d.*%m", fmt)
    ):
        # 调用 guess_datetime_format 函数来推测日期时间格式
        result = parsing.guess_datetime_format(string)
    # 使用 assert 来验证结果是否符合预期格式
    assert result == fmt


@pytest.mark.parametrize("dayfirst,expected", [(True, "%d/%m/%Y"), (False, "%m/%d/%Y")])
def test_guess_datetime_format_with_dayfirst(dayfirst, expected):
    # 使用 pytest 的装饰器 mark.parametrize 提供多组输入参数来执行测试
    ambiguous_string = "01/01/2011"
    # 调用 guess_datetime_format 函数来推测日期时间格式，传入额外的 dayfirst 参数
    result = parsing.guess_datetime_format(ambiguous_string, dayfirst=dayfirst)
    # 使用 assert 来验证结果是否符合预期格式
    assert result == expected


@td.skip_if_not_us_locale
@pytest.mark.parametrize(
    "string,fmt",
    [
        ("30/Dec/2011", "%d/%b/%Y"),
        ("30/December/2011", "%d/%B/%Y"),
        ("30/Dec/2011 00:00:00", "%d/%b/%Y %H:%M:%S"),
    ],
)
def test_guess_datetime_format_with_locale_specific_formats(string, fmt):
    # 使用 pytest 的装饰器 mark.parametrize 提供多组输入参数来执行测试
    # 调用 guess_datetime_format 函数来推测日期时间格式
    result = parsing.guess_datetime_format(string)
    # 使用 assert 来验证结果是否符合预期格式
    assert result == fmt


@pytest.mark.parametrize(
    "invalid_dt",
    [
        "01/2013",
        "12:00:00",
        "1/1/1/1",
        "this_is_not_a_datetime",
        "51a",
        "13/2019",
        "202001",  # YYYYMM isn't ISO8601
        "2020/01",  # YYYY/MM isn't ISO8601 either
        "87156549591102612381000001219H5",
    ],
)
def test_guess_datetime_format_invalid_inputs(invalid_dt):
    # 使用 pytest 的装饰器 mark.parametrize 提供多组输入参数来执行测试
    # 使用 assert 来验证无效输入是否返回 None
    assert parsing.guess_datetime_format(invalid_dt) is None


@pytest.mark.parametrize("invalid_type_dt", [9, datetime(2011, 1, 1)])
def test_guess_datetime_format_wrong_type_inputs(invalid_type_dt):
    # 使用 pytest 的装饰器 mark.parametrize 提供多组输入参数来执行测试
    # 使用 assert 来验证错误类型的输入是否引发 TypeError 异常
    with pytest.raises(
        TypeError,
        match=r"^Argument 'dt_str' has incorrect type \(expected str, got .*\)$",
    ):
        # 调用 guess_datetime_format 函数来推测日期时间格式
        parsing.guess_datetime_format(invalid_type_dt)


@pytest.mark.parametrize(
    "string,fmt,dayfirst,warning",
    [
        ("2011-1-1", "%Y-%m-%d", False, None),
        ("2011-1-1", "%Y-%d-%m", True, None),
        ("1/1/2011", "%m/%d/%Y", False, None),
        ("1/1/2011", "%d/%m/%Y", True, None),
        ("30-1-2011", "%d-%m-%Y", False, UserWarning),
        ("30-1-2011", "%d-%m-%Y", True, None),
        ("2011-1-1 0:0:0", "%Y-%m-%d %H:%M:%S", False, None),
        ("2011-1-1 0:0:0", "%Y-%d-%m %H:%M:%S", True, None),
        ("2011-1-3T00:00:0", "%Y-%m-%dT%H:%M:%S", False, None),
        ("2011-1-3T00:00:0", "%Y-%d-%mT%H:%M:%S", True, None),
        ("2011-1-1 00:00:00", "%Y-%m-%d %H:%M:%S", False, None),
        ("2011-1-1 00:00:00", "%Y-%d-%m %H:%M:%S", True, None),
    ],
)
def test_guess_datetime_format_no_padding(string, fmt, dayfirst, warning):
    # 使用 pytest 的装饰器 mark.parametrize 提供多组输入参数来执行测试
    # 详细测试不同的日期时间格式推测结果，包括可能产生的警告
    # 构造警告信息，指出日期解析使用了默认格式（dayfirst=False），可能导致警告
    msg = (
        rf"Parsing dates in {fmt} format when dayfirst=False \(the default\) "
        "was specified. "
        "Pass `dayfirst=True` or specify a format to silence this warning."
    )
    # 使用 pytest 中的 assert_produces_warning 上下文管理器，验证是否产生特定警告，并匹配预期的警告消息
    with tm.assert_produces_warning(warning, match=msg):
        # 调用日期解析函数，尝试猜测日期时间格式
        result = parsing.guess_datetime_format(string, dayfirst=dayfirst)
    # 断言解析结果与预期格式相符
    assert result == fmt
def test_try_parse_dates():
    # 创建一个包含日期字符串的 NumPy 数组
    arr = np.array(["5/1/2000", "6/1/2000", "7/1/2000"], dtype=object)
    # 使用自定义解析器尝试解析日期，并返回结果数组
    result = parsing.try_parse_dates(arr, parser=lambda x: du_parse(x, dayfirst=True))

    # 创建预期的日期数组，使用 dayfirst=True 解析每个日期字符串
    expected = np.array([du_parse(d, dayfirst=True) for d in arr])
    # 断言两个 NumPy 数组是否相等
    tm.assert_numpy_array_equal(result, expected)


def test_parse_datetime_string_with_reso_check_instance_type_raise_exception():
    # issue 20684 的测试案例
    msg = "Argument 'date_string' has incorrect type (expected str, got tuple)"
    # 使用 pytest 检查是否引发 TypeError 异常，并验证异常消息
    with pytest.raises(TypeError, match=re.escape(msg)):
        parse_datetime_string_with_reso((1, 2, 3))

    # 解析给定的日期字符串，验证返回值与预期元组是否相等
    result = parse_datetime_string_with_reso("2019")
    expected = (datetime(2019, 1, 1), "year")
    assert result == expected


@pytest.mark.parametrize(
    "fmt,expected",
    [
        ("%Y %m %d %H:%M:%S", True),
        ("%Y/%m/%d %H:%M:%S", True),
        (r"%Y\%m\%d %H:%M:%S", True),
        ("%Y-%m-%d %H:%M:%S", True),
        ("%Y.%m.%d %H:%M:%S", True),
        ("%Y%m%d %H:%M:%S", True),
        ("%Y-%m-%dT%H:%M:%S", True),
        ("%Y-%m-%dT%H:%M:%S%z", True),
        ("%Y-%m-%dT%H:%M:%S%Z", False),
        ("%Y-%m-%dT%H:%M:%S.%f", True),
        ("%Y-%m-%dT%H:%M:%S.%f%z", True),
        ("%Y-%m-%dT%H:%M:%S.%f%Z", False),
        ("%Y%m%d", True),
        ("%Y%m", False),
        ("%Y", True),
        ("%Y-%m-%d", True),
        ("%Y-%m", True),
    ],
)
def test_is_iso_format(fmt, expected):
    # 测试 strptime 模块中的 _test_format_is_iso 函数是否正确判断 ISO 格式
    result = strptime._test_format_is_iso(fmt)
    # 断言判断结果与预期是否相等
    assert result == expected


@pytest.mark.parametrize(
    "input",
    [
        "2018-01-01T00:00:00.123456789",
        "2018-01-01T00:00:00.123456",
        "2018-01-01T00:00:00.123",
    ],
)
def test_guess_datetime_format_f(input):
    # https://github.com/pandas-dev/pandas/issues/49043 的测试案例
    # 使用 parsing 模块中的 guess_datetime_format 函数猜测日期时间格式
    result = parsing.guess_datetime_format(input)
    # 预期的日期时间格式字符串
    expected = "%Y-%m-%dT%H:%M:%S.%f"
    # 断言猜测结果与预期是否相等
    assert result == expected


def _helper_hypothesis_delimited_date(call, date_string, **kwargs):
    msg, result = None, None
    try:
        # 调用指定的日期解析函数，尝试解析日期字符串
        result = call(date_string, **kwargs)
    except ValueError as err:
        # 如果解析出错，捕获异常并记录错误消息
        msg = str(err)
    return msg, result


@given(DATETIME_NO_TZ)
@pytest.mark.parametrize("delimiter", list(" -./"))
@pytest.mark.parametrize("dayfirst", [True, False])
@pytest.mark.parametrize(
    "date_format",
    ["%d %m %Y", "%m %d %Y", "%m %Y", "%Y %m %d", "%y %m %d", "%Y%m%d", "%y%m%d"],
)
def test_hypothesis_delimited_date(
    request, date_format, dayfirst, delimiter, test_datetime
):
    if date_format == "%m %Y" and delimiter == ".":
        # 如果日期格式是 "%m %Y" 并且分隔符是 "."，标记测试为失败
        request.applymarker(
            pytest.mark.xfail(
                reason="parse_datetime_string cannot reliably tell whether "
                "e.g. %m.%Y is a float or a date"
            )
        )
    # 根据指定的日期格式和分隔符生成日期字符串
    date_string = test_datetime.strftime(date_format.replace(" ", delimiter))

    # 使用 _helper_hypothesis_delimited_date 函数，测试解析日期字符串是否成功
    except_out_dateutil, result = _helper_hypothesis_delimited_date(
        parsing.py_parse_datetime_string, date_string, dayfirst=dayfirst
    )
    # 调用辅助函数 _helper_hypothesis_delimited_date 进行日期解析，获取返回的两个值
    except_in_dateutil, expected = _helper_hypothesis_delimited_date(
        du_parse,            # 使用的日期解析函数
        date_string,         # 待解析的日期字符串
        default=datetime(1, 1, 1),  # 默认日期值
        dayfirst=dayfirst,   # 是否优先解析日期的天部分
        yearfirst=False,     # 是否优先解析日期的年部分
    )

    # 断言检查日期解析前后的值是否相同
    assert except_out_dateutil == except_in_dateutil
    # 断言检查解析结果是否与预期的结果相同
    assert result == expected
```