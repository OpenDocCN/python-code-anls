# `D:\src\scipysrc\pandas\pandas\tests\tools\test_to_datetime.py`

```
# 导入标准库和第三方库
"""test to_datetime"""

import calendar  # 导入日历模块
from collections import deque  # 导入双端队列数据结构
from datetime import (  # 从 datetime 模块导入多个类和函数
    date,
    datetime,
    timedelta,
    timezone,
)
from decimal import Decimal  # 导入 Decimal 类，用于精确浮点数运算
import locale  # 导入 locale 模块，用于处理特定地域的数据格式化
import zoneinfo  # 导入 zoneinfo 模块，用于处理时区信息

from dateutil.parser import parse  # 从 dateutil 库中导入 parse 函数，用于解析日期时间字符串
import numpy as np  # 导入 NumPy 库，并使用 np 别名
import pytest  # 导入 pytest 库，用于编写和运行测试

from pandas._libs import tslib  # 导入 pandas 私有库中的 tslib 模块
from pandas._libs.tslibs import (  # 导入 pandas 私有库中的日期时间处理模块
    iNaT,  # 导入 iNaT 对象，表示缺失值
    parsing,  # 导入 parsing 模块，用于解析日期时间字符串
)
from pandas.compat import WASM  # 导入 pandas 兼容性库中的 WASM 对象
from pandas.errors import (  # 导入 pandas 错误模块中的异常类
    OutOfBoundsDatetime,  # 超出日期时间范围异常
    OutOfBoundsTimedelta,  # 超出时间间隔范围异常
)
import pandas.util._test_decorators as td  # 导入 pandas 测试装饰器模块，使用 td 别名

from pandas.core.dtypes.common import is_datetime64_ns_dtype  # 导入 pandas 核心模块中的函数

import pandas as pd  # 导入 pandas 库，并使用 pd 别名
from pandas import (  # 从 pandas 中导入多个重要类和函数
    DataFrame,
    DatetimeIndex,
    Index,
    NaT,
    Series,
    Timestamp,
    date_range,
    isna,
    to_datetime,
)
import pandas._testing as tm  # 导入 pandas 测试模块，使用 tm 别名
from pandas.core.arrays import DatetimeArray  # 导入 pandas 核心数组模块中的日期时间数组类
from pandas.core.tools import datetimes as tools  # 导入 pandas 核心工具中的日期时间模块，使用 tools 别名
from pandas.core.tools.datetimes import start_caching_at  # 导入 pandas 核心工具中的日期时间缓存函数

PARSING_ERR_MSG = (  # 设置解析错误消息的原始字符串
    r"You might want to try:\n"
    r"    - passing `format` if your strings have a consistent format;\n"
    r"    - passing `format=\'ISO8601\'` if your strings are all ISO8601 "
    r"but not necessarily in exactly the same format;\n"
    r"    - passing `format=\'mixed\'`, and the format will be inferred "
    r"for each element individually. You might want to use `dayfirst` "
    r"alongside this."
)


class TestTimeConversionFormats:  # 定义测试类 TestTimeConversionFormats
    def test_to_datetime_readonly(self, writable):  # 定义测试只读情况下的 to_datetime 方法
        # GH#34857
        arr = np.array([], dtype=object)  # 创建空的 NumPy 数组对象 arr，数据类型为 object
        arr.setflags(write=writable)  # 设置数组 arr 的写入标志
        result = to_datetime(arr)  # 调用 to_datetime 函数处理数组 arr
        expected = to_datetime([])  # 生成预期的空时间戳对象列表
        tm.assert_index_equal(result, expected)  # 使用测试工具断言结果与预期相等

    @pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，传入多组参数
        "format, expected",  # 定义参数化测试的参数名称
        [  # 参数化测试的参数列表
            [
                "%d/%m/%Y",
                [Timestamp("20000101"), Timestamp("20000201"), Timestamp("20000301")],
            ],
            [
                "%m/%d/%Y",
                [Timestamp("20000101"), Timestamp("20000102"), Timestamp("20000103")],
            ],
        ],
    )
    def test_to_datetime_format(self, cache, index_or_series, format, expected):
        values = index_or_series(["1/1/2000", "1/2/2000", "1/3/2000"])  # 创建索引或系列对象 values，包含日期时间字符串
        result = to_datetime(values, format=format, cache=cache)  # 调用 to_datetime 函数处理 values
        expected = index_or_series(expected)  # 生成预期的时间戳对象列表
        tm.assert_equal(result, expected)  # 使用测试工具断言结果与预期相等

    @pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，传入多组参数
        "arg, expected, format",  # 定义参数化测试的参数名称
        [  # 参数化测试的参数列表
            ["1/1/2000", "20000101", "%d/%m/%Y"],
            ["1/1/2000", "20000101", "%m/%d/%Y"],
            ["1/2/2000", "20000201", "%d/%m/%Y"],
            ["1/2/2000", "20000102", "%m/%d/%Y"],
            ["1/3/2000", "20000301", "%d/%m/%Y"],
            ["1/3/2000", "20000103", "%m/%d/%Y"],
        ],
    )
    def test_to_datetime_format_scalar(self, cache, arg, expected, format):
        result = to_datetime(arg, format=format, cache=cache)  # 调用 to_datetime 函数处理单个日期时间字符串
        expected = Timestamp(expected)  # 生成预期的时间戳对象
        assert result == expected  # 断言结果与预期相等
    # 测试函数：将整数序列转换为日期时间格式的 Series，使用指定的格式字符串 "%Y%m%d"
    def test_to_datetime_format_YYYYMMDD(self, cache):
        # 创建一个包含日期整数的 Series
        ser = Series([19801222, 19801222] + [19810105] * 5)
        # 创建期望的结果 Series，将每个整数转换为 Timestamp 对象
        expected = Series([Timestamp(x) for x in ser.apply(str)])

        # 调用待测试的函数 to_datetime，传入整数序列和指定的日期格式，返回结果
        result = to_datetime(ser, format="%Y%m%d", cache=cache)
        # 使用 Pandas 提供的测试工具，比较返回的结果和期望的结果 Series 是否相等
        tm.assert_series_equal(result, expected)

        # 将整数序列先转换为字符串序列，再调用 to_datetime 函数进行转换
        result = to_datetime(ser.apply(str), format="%Y%m%d", cache=cache)
        # 使用 Pandas 提供的测试工具，比较返回的结果和期望的结果 Series 是否相等
        tm.assert_series_equal(result, expected)

    # 测试函数：将整数序列转换为日期时间格式的 Series，使用指定的格式字符串 "%Y%m%d"，并处理 NaT
    def test_to_datetime_format_YYYYMMDD_with_nat(self, cache):
        # 将整数序列转换为浮点数序列，以便处理 NaT
        ser = Series([19801222, 19801222] + [19810105] * 5, dtype="float")
        # 创建期望的结果 Series，将每个整数转换为 Timestamp 对象，并设置其中一个值为 NaT
        expected = Series(
            [Timestamp("19801222"), Timestamp("19801222")]
            + [Timestamp("19810105")] * 5,
            dtype="M8[s]",
        )
        expected[2] = np.nan
        ser[2] = np.nan

        # 调用待测试的函数 to_datetime，传入序列和指定的日期格式，返回结果
        result = to_datetime(ser, format="%Y%m%d", cache=cache)
        # 使用 Pandas 提供的测试工具，比较返回的结果和期望的结果 Series 是否相等
        tm.assert_series_equal(result, expected)

        # 将序列转换为字符串序列，并处理其中一个值为字符串 "nat" 的情况，验证是否抛出预期的异常
        ser2 = ser.apply(str)
        ser2[2] = "nat"
        with pytest.raises(
            ValueError,
            match=(
                'unconverted data remains when parsing with format "%Y%m%d": ".0", '
                "at position 0"
            ),
        ):
            # https://github.com/pandas-dev/pandas/issues/50051
            # 调用待测试的函数 to_datetime，传入序列和指定的日期格式，验证是否抛出异常
            to_datetime(ser2, format="%Y%m%d", cache=cache)

    # 测试函数：将整数序列转换为日期时间格式的 Series，使用指定的格式字符串 "%Y%m"，并处理 NaT
    def test_to_datetime_format_YYYYMM_with_nat(self, cache):
        # 将整数序列转换为浮点数序列，以便处理 NaT
        ser = Series([198012, 198012] + [198101] * 5, dtype="float")
        # 创建期望的结果 Series，将每个整数转换为 Timestamp 对象，并设置其中一个值为 NaT
        expected = Series(
            [Timestamp("19801201"), Timestamp("19801201")]
            + [Timestamp("19810101")] * 5,
            dtype="M8[s]",
        )
        expected[2] = np.nan
        ser[2] = np.nan

        # 调用待测试的函数 to_datetime，传入序列和指定的日期格式，返回结果
        result = to_datetime(ser, format="%Y%m", cache=cache)
        # 使用 Pandas 提供的测试工具，比较返回的结果和期望的结果 Series 是否相等
        tm.assert_series_equal(result, expected)

    # 测试函数：将整数序列转换为日期时间格式的 Series，使用指定的格式字符串 "%Y%m%d"，并处理超出范围的日期
    def test_to_datetime_format_YYYYMMDD_oob_for_ns(self, cache):
        # 创建包含日期整数的 Series
        ser = Series([20121231, 20141231, 99991231])
        # 调用待测试的函数 to_datetime，传入序列和指定的日期格式，设置 errors 参数为 "raise"，返回结果
        result = to_datetime(ser, format="%Y%m%d", errors="raise", cache=cache)
        # 创建期望的结果 Series，将每个整数转换为指定格式的日期字符串
        expected = Series(
            np.array(["2012-12-31", "2014-12-31", "9999-12-31"], dtype="M8[s]"),
            dtype="M8[s]",
        )
        # 使用 Pandas 提供的测试工具，比较返回的结果和期望的结果 Series 是否相等
        tm.assert_series_equal(result, expected)

    # 测试函数：将整数序列转换为日期时间格式的 Series，使用指定的格式字符串 "%Y%m%d"，并处理超出范围的日期和类型转换
    def test_to_datetime_format_YYYYMMDD_coercion(self, cache):
        # 创建包含日期整数的 Series
        ser = Series([20121231, 20141231, 999999999999999999999999999991231])
        # 调用待测试的函数 to_datetime，传入序列和指定的日期格式，设置 errors 参数为 "coerce"，返回结果
        result = to_datetime(ser, format="%Y%m%d", errors="coerce", cache=cache)
        # 创建期望的结果 Series，将每个整数转换为指定格式的日期字符串，处理类型转换
        expected = Series(["20121231", "20141231", "NaT"], dtype="M8[s]")
        # 使用 Pandas 提供的测试工具，比较返回的结果和期望的结果 Series 是否相等
        tm.assert_series_equal(result, expected)
    @pytest.mark.parametrize(
        "input_s",
        [
            # 定义输入参数 input_s 的多组测试数据
            # 字符串类型的日期格式，最后一个元素为 None
            ["19801222", "20010112", None],
            ["19801222", "20010112", np.nan],
            ["19801222", "20010112", NaT],
            ["19801222", "20010112", "NaT"],
            # 整数类型的日期格式，最后一个元素为 None
            [19801222, 20010112, None],
            [19801222, 20010112, np.nan],
            [19801222, 20010112, NaT],
            [19801222, 20010112, "NaT"],
        ],
    )
    def test_to_datetime_format_YYYYMMDD_with_none(self, input_s):
        # GH 30011
        # 对于函数 to_datetime，测试指定格式 '%Y%m%d' 下的日期转换行为
        # 期望结果是一个 Series，包含指定日期和 NaT（Not a Time）的时间戳
        expected = Series([Timestamp("19801222"), Timestamp("20010112"), NaT])
        result = Series(to_datetime(input_s, format="%Y%m%d"))
        # 使用 pytest 框架的 assert_series_equal 函数验证结果与期望值是否相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "input_s, expected",
        [
            # 测试在字符串日期值之前存在 NaN 和无效日期值的情况
            [
                ["19801222", np.nan, "20010012", "10019999"],
                [Timestamp("19801222"), np.nan, np.nan, np.nan],
            ],
            # 测试在字符串日期值之后存在 NaN 和无效日期值的情况
            [
                ["19801222", "20010012", "10019999", np.nan],
                [Timestamp("19801222"), np.nan, np.nan, np.nan],
            ],
            # 测试在整数日期值之前存在 NaN 和无效日期值的情况
            [
                [20190813, np.nan, 20010012, 20019999],
                [Timestamp("20190813"), np.nan, np.nan, np.nan],
            ],
            # 测试在整数日期值之后存在 NaN 和无效日期值的情况
            [
                [20190813, 20010012, np.nan, 20019999],
                [Timestamp("20190813"), np.nan, np.nan, np.nan],
            ],
        ],
    )
    def test_to_datetime_format_YYYYMMDD_overflow(self, input_s, expected):
        # GH 25512
        # 对于函数 to_datetime，测试指定格式 '%Y%m%d' 下的日期转换行为，并设置错误处理方式为 'coerce'
        # 输入数据转为 Series 类型
        input_s = Series(input_s)
        # 执行日期转换，使用 'coerce' 错误处理方式
        result = to_datetime(input_s, format="%Y%m%d", errors="coerce")
        # 期望结果转为 Series 类型
        expected = Series(expected)
        # 使用 pytest 框架的 assert_series_equal 函数验证结果与期望值是否相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "data, format, expected",
        [
            # 测试包含 pd.NA 值的情况，指定日期时间格式为 '%Y%m%d%H%M%S'
            ([pd.NA], "%Y%m%d%H%M%S", ["NaT"]),
            # 测试包含 pd.NA 值的情况，日期时间格式为 None
            ([pd.NA], None, ["NaT"]),
            # 测试包含 pd.NA 值和有效日期时间值的情况，指定日期时间格式为 '%Y%m%d%H%M%S'
            (
                [pd.NA, "20210202202020"],
                "%Y%m%d%H%M%S",
                ["NaT", "2021-02-02 20:20:20"],
            ),
            # 测试包含有效日期时间值和 pd.NA 值的情况，指定日期时间格式为 '%y%m%d'
            (["201010", pd.NA], "%y%m%d", ["2020-10-10", "NaT"]),
            # 测试包含有效日期时间值和 pd.NA 值的情况，指定日期时间格式为 '%d%m%y'
            (["201010", pd.NA], "%d%m%y", ["2010-10-20", "NaT"]),
            # 测试包含 None、np.nan 和 pd.NA 值的情况，日期时间格式为 None
            ([None, np.nan, pd.NA], None, ["NaT", "NaT", "NaT"]),
            # 测试包含 None、np.nan 和 pd.NA 值的情况，指定日期时间格式为 '%Y%m%d'
            ([None, np.nan, pd.NA], "%Y%m%d", ["NaT", "NaT", "NaT"]),
        ],
    )
    def test_to_datetime_with_NA(self, data, format, expected):
        # GH#42957
        # 对于函数 to_datetime，测试包含 NA 值的日期时间格式转换行为
        # 执行日期时间转换
        result = to_datetime(data, format=format)
        # 期望结果转为 DatetimeIndex 类型
        expected = DatetimeIndex(expected)
        # 使用 pytest 框架的 assert_index_equal 函数验证结果与期望值是否相等
        tm.assert_index_equal(result, expected)
    # 测试特殊情况：包含 pd.NA 的日期转换为 DatetimeIndex
    def test_to_datetime_with_NA_with_warning(self):
        # GH#42957 表示 GitHub 上的 issue 编号
        result = to_datetime(["201010", pd.NA])
        expected = DatetimeIndex(["2010-10-20", "NaT"])  # 预期的日期时间索引
        tm.assert_index_equal(result, expected)  # 断言结果与预期相等

    # 测试不同格式下的日期时间转换
    def test_to_datetime_format_integer(self, cache):
        # GH 10178 表示 GitHub 上的 issue 编号
        ser = Series([2000, 2001, 2002])
        expected = Series([Timestamp(x) for x in ser.apply(str)])  # 期望的日期时间序列

        # 测试以 "%Y" 格式转换
        result = to_datetime(ser, format="%Y", cache=cache)
        tm.assert_series_equal(result, expected)  # 断言结果序列与期望序列相等

        ser = Series([200001, 200105, 200206])
        expected = Series([Timestamp(x[:4] + "-" + x[4:]) for x in ser.apply(str)])  # 期望的日期时间序列

        # 测试以 "%Y%m" 格式转换
        result = to_datetime(ser, format="%Y%m", cache=cache)
        tm.assert_series_equal(result, expected)  # 断言结果序列与期望序列相等

    # 测试微秒精度下的日期时间转换
    def test_to_datetime_format_microsecond(self, cache):
        month_abbr = calendar.month_abbr[4]
        val = f"01-{month_abbr}-2011 00:00:01.978"  # 待转换的日期时间字符串

        format = "%d-%b-%Y %H:%M:%S.%f"  # 格式化字符串
        result = to_datetime(val, format=format, cache=cache)
        exp = datetime.strptime(val, format)
        assert result == exp  # 断言结果与预期相等

    # 使用参数化测试不同格式的日期时间转换
    @pytest.mark.parametrize(
        "value, format, dt",
        [
            ["01/10/2010 15:20", "%m/%d/%Y %H:%M", Timestamp("2010-01-10 15:20")],
            ["01/10/2010 05:43", "%m/%d/%Y %I:%M", Timestamp("2010-01-10 05:43")],
            ["01/10/2010 13:56:01", "%m/%d/%Y %H:%M:%S", Timestamp("2010-01-10 13:56:01")],
            # 以下3个测试因受本地化影响可能失败
            pytest.param(
                "01/10/2010 08:14 PM",
                "%m/%d/%Y %I:%M %p",
                Timestamp("2010-01-10 20:14"),
                marks=pytest.mark.xfail(
                    locale.getlocale()[0] in ("zh_CN", "it_IT"),
                    reason="fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8",
                    strict=False,
                ),
            ),
            pytest.param(
                "01/10/2010 07:40 AM",
                "%m/%d/%Y %I:%M %p",
                Timestamp("2010-01-10 07:40"),
                marks=pytest.mark.xfail(
                    locale.getlocale()[0] in ("zh_CN", "it_IT"),
                    reason="fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8",
                    strict=False,
                ),
            ),
            pytest.param(
                "01/10/2010 09:12:56 AM",
                "%m/%d/%Y %I:%M:%S %p",
                Timestamp("2010-01-10 09:12:56"),
                marks=pytest.mark.xfail(
                    locale.getlocale()[0] in ("zh_CN", "it_IT"),
                    reason="fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8",
                    strict=False,
                ),
            ),
        ],
    )
    def test_to_datetime_format_time(self, cache, value, format, dt):
        assert to_datetime(value, format=format, cache=cache) == dt  # 断言结果与期望相等
    @td.skip_if_not_us_locale
    def test_to_datetime_with_non_exact(self, cache):
        # 使用装饰器跳过非美国区域的测试
        # 测试用例针对 GH 10834 编号问题进行验证
        # 针对 exact 关键字参数的测试
        ser = Series(
            ["19MAY11", "foobar19MAY11", "19MAY11:00:00:00", "19MAY11 00:00:00Z"]
        )
        # 调用 to_datetime 函数，exact 参数设置为 False，缓存设置为传入的 cache 参数
        result = to_datetime(ser, format="%d%b%y", exact=False, cache=cache)
        # 使用正则表达式从 ser 中提取日期部分，作为 to_datetime 的期望输出
        expected = to_datetime(
            ser.str.extract(r"(\d+\w+\d+)", expand=False), format="%d%b%y", cache=cache
        )
        # 使用 pytest 框架的 assert_series_equal 方法比较 result 和 expected
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "format, expected",
        [
            ("%Y-%m-%d", Timestamp(2000, 1, 3)),
            ("%Y-%d-%m", Timestamp(2000, 3, 1)),
            ("%Y-%m-%d %H", Timestamp(2000, 1, 3, 12)),
            ("%Y-%d-%m %H", Timestamp(2000, 3, 1, 12)),
            ("%Y-%m-%d %H:%M", Timestamp(2000, 1, 3, 12, 34)),
            ("%Y-%d-%m %H:%M", Timestamp(2000, 3, 1, 12, 34)),
            ("%Y-%m-%d %H:%M:%S", Timestamp(2000, 1, 3, 12, 34, 56)),
            ("%Y-%d-%m %H:%M:%S", Timestamp(2000, 3, 1, 12, 34, 56)),
            ("%Y-%m-%d %H:%M:%S.%f", Timestamp(2000, 1, 3, 12, 34, 56, 123456)),
            ("%Y-%d-%m %H:%M:%S.%f", Timestamp(2000, 3, 1, 12, 34, 56, 123456)),
            (
                "%Y-%m-%d %H:%M:%S.%f%z",
                Timestamp(2000, 1, 3, 12, 34, 56, 123456, tz="UTC+01:00"),
            ),
            (
                "%Y-%d-%m %H:%M:%S.%f%z",
                Timestamp(2000, 3, 1, 12, 34, 56, 123456, tz="UTC+01:00"),
            ),
        ],
    )
    def test_non_exact_doesnt_parse_whole_string(self, cache, format, expected):
        # 根据参数 format 指定的格式解析日期时间字符串
        # 验证 exact 参数为 False 时的行为，与传入的 expected 进行比较
        result = to_datetime(
            "2000-01-03 12:34:56.123456+01:00", format=format, exact=False
        )
        # 使用 assert 语句比较 result 和 expected 是否相等
        assert result == expected

    @pytest.mark.parametrize(
        "arg",
        [
            "2012-01-01 09:00:00.000000001",
            "2012-01-01 09:00:00.000001",
            "2012-01-01 09:00:00.001",
            "2012-01-01 09:00:00.001000",
            "2012-01-01 09:00:00.001000000",
        ],
    )
    def test_parse_nanoseconds_with_formula(self, cache, arg):
        # GH8989 编号问题
        # 当提供了格式时，截断纳秒部分
        expected = to_datetime(arg, cache=cache)
        # 使用指定的格式解析日期时间字符串，与提供的 expected 进行比较
        result = to_datetime(arg, format="%Y-%m-%d %H:%M:%S.%f", cache=cache)
        # 使用 assert 语句比较 result 和 expected 是否相等
        assert result == expected

    @pytest.mark.parametrize(
        "value,fmt,expected",
        [
            ["2009324", "%Y%W%w", "2009-08-13"],
            ["2013020", "%Y%U%w", "2013-01-13"],
        ],
    )
    def test_to_datetime_format_weeks(self, value, fmt, expected, cache):
        # 使用指定的格式 fmt 解析 value 为日期时间对象，与 expected 进行比较
        assert to_datetime(value, format=fmt, cache=cache) == Timestamp(expected)
    @pytest.mark.parametrize(
        "fmt,dates,expected_dates",
        [  # 参数化测试参数：日期时间格式、输入日期字符串、预期的日期时间对象列表
            [
                "%Y-%m-%d %H:%M:%S %Z",
                ["2010-01-01 12:00:00 UTC"] * 2,  # 使用UTC时区的日期字符串
                [Timestamp("2010-01-01 12:00:00", tz="UTC")] * 2,  # 预期的UTC时区的时间戳对象列表
            ],
            [
                "%Y-%m-%d %H:%M:%S%z",
                ["2010-01-01 12:00:00+0100"] * 2,  # 使用偏移+0100的日期时间字符串
                [
                    Timestamp(
                        "2010-01-01 12:00:00", tzinfo=timezone(timedelta(minutes=60))
                    )
                ]
                * 2,  # 预期的带有偏移的时间戳对象列表
            ],
            [
                "%Y-%m-%d %H:%M:%S %z",
                ["2010-01-01 12:00:00 +0100"] * 2,  # 使用带空格的偏移+0100的日期时间字符串
                [
                    Timestamp(
                        "2010-01-01 12:00:00", tzinfo=timezone(timedelta(minutes=60))
                    )
                ]
                * 2,  # 预期的带有偏移的时间戳对象列表
            ],
            [
                "%Y-%m-%d %H:%M:%S %z",
                ["2010-01-01 12:00:00 Z", "2010-01-01 12:00:00 Z"],  # 使用Z时区的日期时间字符串
                [
                    Timestamp(
                        "2010-01-01 12:00:00", tzinfo=timezone(timedelta(minutes=0))
                    ),
                    Timestamp(
                        "2010-01-01 12:00:00", tzinfo=timezone(timedelta(minutes=0))
                    ),
                ],  # 预期的无偏移的时间戳对象列表
            ],
        ],
    )
    def test_to_datetime_parse_tzname_or_tzoffset(self, fmt, dates, expected_dates):
        # GH 13486, GH 50887, GH 57275
        # 执行 to_datetime 函数进行日期时间解析
        result = to_datetime(dates, format=fmt)
        # 创建预期的索引对象
        expected = Index(expected_dates)
        # 使用测试工具验证结果是否符合预期
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "fmt,dates,expected_dates",
        [  # 参数化测试参数：日期时间格式、输入日期字符串、预期的日期时间对象列表
            [
                "%Y-%m-%d %H:%M:%S %Z",
                [
                    "2010-01-01 12:00:00 UTC",
                    "2010-01-01 12:00:00 GMT",
                    "2010-01-01 12:00:00 US/Pacific",
                ],  # 使用不同时区的日期时间字符串
                [
                    Timestamp("2010-01-01 12:00:00", tz="UTC"),
                    Timestamp("2010-01-01 12:00:00", tz="GMT"),
                    Timestamp("2010-01-01 12:00:00", tz="US/Pacific"),
                ],  # 预期的不同时区的时间戳对象列表
            ],
            [
                "%Y-%m-%d %H:%M:%S %z",
                ["2010-01-01 12:00:00 +0100", "2010-01-01 12:00:00 -0100"],  # 使用不同偏移的日期时间字符串
                [
                    Timestamp(
                        "2010-01-01 12:00:00", tzinfo=timezone(timedelta(minutes=60))
                    ),
                    Timestamp(
                        "2010-01-01 12:00:00", tzinfo=timezone(timedelta(minutes=-60))
                    ),
                ],  # 预期的带有不同偏移的时间戳对象列表
            ],
        ],
    )
    def test_to_datetime_parse_tzname_or_tzoffset_utc_false_removed(
        self, fmt, dates, expected_dates
    ):
        # GH#13486, GH#50887, GH#57275
        # 函数未传入 utc=True 抛出 ValueError 异常的测试
        msg = "Mixed timezones detected. Pass utc=True in to_datetime"
        with pytest.raises(ValueError, match=msg):
            to_datetime(dates, format=fmt)
    # 测试函数，验证将具有不同时区信息的日期转换为 UTC 时间
    def test_to_datetime_parse_tzname_or_tzoffset_different_tz_to_utc(self):
        # 定义包含不同时区信息的日期列表
        dates = [
            "2010-01-01 12:00:00 +0100",
            "2010-01-01 12:00:00 -0100",
            "2010-01-01 12:00:00 +0300",
            "2010-01-01 12:00:00 +0400",
        ]
        # 预期的转换为 UTC 后的日期列表
        expected_dates = [
            "2010-01-01 11:00:00+00:00",
            "2010-01-01 13:00:00+00:00",
            "2010-01-01 09:00:00+00:00",
            "2010-01-01 08:00:00+00:00",
        ]
        # 日期格式字符串
        fmt = "%Y-%m-%d %H:%M:%S %z"

        # 执行日期转换函数，并指定返回 UTC 时间
        result = to_datetime(dates, format=fmt, utc=True)
        # 创建预期的日期时间索引对象
        expected = DatetimeIndex(expected_dates)
        # 验证结果是否符合预期
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "offset", ["+0", "-1foo", "UTCbar", ":10", "+01:000:01", ""]
    )
    # 测试函数，验证处理格式错误的时区信息时是否会抛出 ValueError 异常
    def test_to_datetime_parse_timezone_malformed(self, offset):
        # 日期时间格式字符串
        fmt = "%Y-%m-%d %H:%M:%S %z"
        # 构造带有错误格式时区信息的日期字符串
        date = "2010-01-01 12:00:00 " + offset

        # 构造异常匹配的消息模板
        msg = "|".join(
            [
                r'^time data ".*" doesn\'t match format ".*", at position 0. '
                f"{PARSING_ERR_MSG}$",
                r'^unconverted data remains when parsing with format ".*": ".*", '
                f"at position 0. {PARSING_ERR_MSG}$",
            ]
        )
        # 使用 pytest 断言是否抛出指定异常类型，并匹配异常消息
        with pytest.raises(ValueError, match=msg):
            to_datetime([date], format=fmt)

    # 测试函数，验证解析带有时区名称的日期时间字符串
    def test_to_datetime_parse_timezone_keeps_name(self):
        # 日期时间格式字符串
        fmt = "%Y-%m-%d %H:%M:%S %z"
        # 创建带有时区名称的索引对象
        arg = Index(["2010-01-01 12:00:00 Z"], name="foo")
        # 执行日期时间转换函数
        result = to_datetime(arg, format=fmt)
        # 创建预期的日期时间索引对象，保留 UTC 时区名称
        expected = DatetimeIndex(["2010-01-01 12:00:00"], tz="UTC", name="foo")
        # 验证结果是否符合预期
        tm.assert_index_equal(result, expected)
    # 定义一个测试类 TestToDatetime，用于测试 to_datetime 函数的各种情况
    class TestToDatetime:
        # 使用 pytest.mark.filterwarnings 忽略警告 "Could not infer format"
        @pytest.mark.filterwarnings("ignore:Could not infer format")
        # 定义测试函数 test_to_datetime_overflow，测试 datetime 转换溢出的情况
        def test_to_datetime_overflow(self):
            # 准备测试参数
            arg = "08335394550"
            msg = 'Parsing "08335394550" to datetime overflows, at position 0'
            # 使用 pytest.raises 检查是否引发 OutOfBoundsDatetime 异常，并匹配指定消息
            with pytest.raises(OutOfBoundsDatetime, match=msg):
                to_datetime(arg)

            # 同上，但传入一个列表参数
            with pytest.raises(OutOfBoundsDatetime, match=msg):
                to_datetime([arg])

            # 测试错误处理为 "coerce" 时的返回值
            res = to_datetime(arg, errors="coerce")
            assert res is NaT  # 断言结果为 NaT (Not a Time)

            # 同上，传入列表参数时的处理
            res = to_datetime([arg], errors="coerce")
            exp = Index([NaT], dtype="M8[s]")  # 期望的结果是一个 NaT 的索引
            tm.assert_index_equal(res, exp)  # 断言索引相等

        # 定义测试函数 test_to_datetime_mixed_datetime_and_string，测试混合日期时间和字符串的情况
        def test_to_datetime_mixed_datetime_and_string(self):
            # 准备日期时间对象 d1 和 d2
            d1 = datetime(2020, 1, 1, 17, tzinfo=timezone(-timedelta(hours=1)))
            d2 = datetime(2020, 1, 1, 18, tzinfo=timezone(-timedelta(hours=1)))
            # 调用 to_datetime 函数，传入包含混合格式的日期时间字符串和 datetime 对象的列表
            res = to_datetime(["2020-01-01 17:00 -0100", d2])
            # 准备预期结果，将列表中的 datetime 对象转换为正确的时区
            expected = to_datetime([d1, d2]).tz_convert(timezone(timedelta(minutes=-60)))
            tm.assert_index_equal(res, expected)  # 断言索引相等

        # 定义测试函数 test_to_datetime_mixed_string_and_numeric，测试混合字符串和数值的情况
        def test_to_datetime_mixed_string_and_numeric(self):
            # 准备字符串和数值的列表
            vals = ["2016-01-01", 0]
            # 准备预期的 DatetimeIndex 结果，将字符串和数值分别转换为 Timestamp 对象
            expected = DatetimeIndex([Timestamp(x) for x in vals])
            # 分别调用 to_datetime 函数，传入不同的参数格式，获取结果
            result = to_datetime(vals, format="mixed")
            result2 = to_datetime(vals[::-1], format="mixed")[::-1]
            result3 = DatetimeIndex(vals)
            result4 = DatetimeIndex(vals[::-1])[::-1]

            # 使用 tm.assert_index_equal 断言各个结果与预期结果相等
            tm.assert_index_equal(result, expected)
            tm.assert_index_equal(result2, expected)
            tm.assert_index_equal(result3, expected)
            tm.assert_index_equal(result4, expected)

        # 使用 pytest.mark.parametrize 标记的测试函数，测试混合日期和字符串的情况
        @pytest.mark.parametrize(
            "format", ["%Y-%m-%d", "%Y-%d-%m"], ids=["ISO8601", "non-ISO8601"]
        )
        def test_to_datetime_mixed_date_and_string(self, format):
            # 准备日期对象 d1
            d1 = date(2020, 1, 2)
            # 调用 to_datetime 函数，传入包含混合格式的日期字符串和日期对象的列表，指定格式为 parametrize 的 format 参数
            res = to_datetime(["2020-01-01", d1], format=format)
            # 准备预期的 DatetimeIndex 结果，指定 dtype 为 "M8[s]"
            expected = DatetimeIndex(["2020-01-01", "2020-01-02"], dtype="M8[s]")
            tm.assert_index_equal(res, expected)  # 断言索引相等

        # 使用 pytest.mark.parametrize 标记的测试函数，测试不同格式的日期时间字符串
        @pytest.mark.parametrize(
            "fmt",
            ["%Y-%d-%m %H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z"],
            ids=["non-ISO8601 format", "ISO8601 format"],
        )
    @pytest.mark.parametrize(
        "utc, args, expected",
        [  # 参数化测试用例：utc、args、expected参数组成的列表
            pytest.param(
                True,  # 第一个参数：utc为True
                ["2000-01-01 01:00:00-08:00", "2000-01-01 02:00:00-08:00"],  # args为包含两个时区信息的日期时间字符串列表
                DatetimeIndex(  # 期望的结果是一个DatetimeIndex对象
                    ["2000-01-01 09:00:00+00:00", "2000-01-01 10:00:00+00:00"],  # 期望的DatetimeIndex包含的日期时间字符串
                    dtype="datetime64[us, UTC]",  # 指定结果的数据类型为带有微秒精度和UTC时区的datetime64
                ),
                id="all tz-aware, with utc",  # 测试用例的ID
            ),
            pytest.param(
                False,  # 第一个参数：utc为False
                ["2000-01-01 01:00:00+00:00", "2000-01-01 02:00:00+00:00"],  # args为包含两个UTC时间的日期时间字符串列表
                DatetimeIndex(  # 期望的结果是一个DatetimeIndex对象
                    ["2000-01-01 01:00:00+00:00", "2000-01-01 02:00:00+00:00"],  # 期望的DatetimeIndex包含的日期时间字符串
                ).as_unit("us"),  # 将结果转换为微秒单位
                id="all tz-aware, without utc",  # 测试用例的ID
            ),
            pytest.param(
                True,  # 第一个参数：utc为True
                ["2000-01-01 01:00:00-08:00", "2000-01-01 02:00:00+00:00"],  # args为包含混合时区的日期时间字符串列表
                DatetimeIndex(  # 期望的结果是一个DatetimeIndex对象
                    ["2000-01-01 09:00:00+00:00", "2000-01-01 02:00:00+00:00"],  # 期望的DatetimeIndex包含的日期时间字符串
                    dtype="datetime64[us, UTC]",  # 指定结果的数据类型为带有微秒精度和UTC时区的datetime64
                ),
                id="all tz-aware, mixed offsets, with utc",  # 测试用例的ID
            ),
            pytest.param(
                True,  # 第一个参数：utc为True
                ["2000-01-01 01:00:00", "2000-01-01 02:00:00+00:00"],  # args为包含一个时区感知字符串和一个naive字符串的列表
                DatetimeIndex(  # 期望的结果是一个DatetimeIndex对象
                    ["2000-01-01 01:00:00+00:00", "2000-01-01 02:00:00+00:00"],  # 期望的DatetimeIndex包含的日期时间字符串
                    dtype="datetime64[us, UTC]",  # 指定结果的数据类型为带有微秒精度和UTC时区的datetime64
                ),
                id="tz-aware string, naive pydatetime, with utc",  # 测试用例的ID
            ),
        ],
    )
    @pytest.mark.parametrize(
        "constructor",
        [Timestamp, lambda x: Timestamp(x).to_pydatetime()],  # 参数化测试构造函数列表
    )
    def test_to_datetime_mixed_datetime_and_string_with_format(
        self, fmt, utc, args, expected, constructor
    ):
        # https://github.com/pandas-dev/pandas/issues/49298
        # https://github.com/pandas-dev/pandas/issues/50254
        # 注意：ISO8601格式会走一个快速路径，所以我们需要检查ISO8601格式和非ISO8601格式
        ts1 = constructor(args[0])  # 使用构造函数处理第一个参数，生成时间戳对象ts1
        ts2 = args[1]  # 第二个参数直接赋值给ts2
        result = to_datetime([ts1, ts2], format=fmt, utc=utc)  # 调用to_datetime函数，传入参数和格式
        if constructor is Timestamp:  # 如果构造函数是Timestamp
            expected = expected.as_unit("s")  # 将期望结果转换为秒单位
        tm.assert_index_equal(result, expected)  # 使用assert_index_equal断言result和期望结果相等

    @pytest.mark.parametrize(
        "fmt",
        ["%Y-%d-%m %H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z"],  # 参数化测试日期时间格式列表
        ids=["non-ISO8601 format", "ISO8601 format"],  # 标记不同格式的ID
    )
    @pytest.mark.parametrize(
        "constructor",
        [Timestamp, lambda x: Timestamp(x).to_pydatetime()],  # 参数化测试构造函数列表
    )
    def test_to_datetime_mixed_dt_and_str_with_format_mixed_offsets_utc_false_removed(
        self, fmt, constructor
    ):
    ):
        # 在下面的测试函数中，处理了包含不同时区混合的情况
        # https://github.com/pandas-dev/pandas/issues/49298
        # https://github.com/pandas-dev/pandas/issues/50254
        # GH#57275
        # 注意：ISO8601格式的时间有一个快速路径，因此我们需要同时检查ISO8601格式和非ISO8601格式
        args = ["2000-01-01 01:00:00", "2000-01-01 02:00:00+00:00"]
        ts1 = constructor(args[0])
        ts2 = args[1]
        msg = "Mixed timezones detected. Pass utc=True in to_datetime"

        # 使用 pytest 的断言来检查是否抛出 ValueError 异常，并匹配特定的错误消息
        with pytest.raises(ValueError, match=msg):
            to_datetime([ts1, ts2], format=fmt, utc=False)

    @pytest.mark.parametrize(
        "fmt, expected",
        [
            pytest.param(
                "%Y-%m-%d %H:%M:%S%z",
                [
                    Timestamp("2000-01-01 09:00:00+0100", tz="UTC+01:00"),
                    Timestamp("2000-01-02 02:00:00+0200", tz="UTC+02:00"),
                    NaT,
                ],
                id="ISO8601, non-UTC",
            ),
            pytest.param(
                "%Y-%d-%m %H:%M:%S%z",
                [
                    Timestamp("2000-01-01 09:00:00+0100", tz="UTC+01:00"),
                    Timestamp("2000-02-01 02:00:00+0200", tz="UTC+02:00"),
                    NaT,
                ],
                id="non-ISO8601, non-UTC",
            ),
        ],
    )
    def test_to_datetime_mixed_offsets_with_none_tz_utc_false_removed(
        self, fmt, expected
    ):
        # 在下面的测试函数中，测试处理不同偏移量混合情况下，移除了非 UTC 时区并且时区为 None 的情况
        # https://github.com/pandas-dev/pandas/issues/50071
        # GH#57275
        msg = "Mixed timezones detected. Pass utc=True in to_datetime"

        # 使用 pytest 的断言来检查是否抛出 ValueError 异常，并匹配特定的错误消息
        with pytest.raises(ValueError, match=msg):
            to_datetime(
                ["2000-01-01 09:00:00+01:00", "2000-01-02 02:00:00+02:00", None],
                format=fmt,
                utc=False,
            )

    @pytest.mark.parametrize(
        "fmt, expected",
        [
            pytest.param(
                "%Y-%m-%d %H:%M:%S%z",
                DatetimeIndex(
                    ["2000-01-01 08:00:00+00:00", "2000-01-02 00:00:00+00:00", "NaT"],
                    dtype="datetime64[s, UTC]",
                ),
                id="ISO8601, UTC",
            ),
            pytest.param(
                "%Y-%d-%m %H:%M:%S%z",
                DatetimeIndex(
                    ["2000-01-01 08:00:00+00:00", "2000-02-01 00:00:00+00:00", "NaT"],
                    dtype="datetime64[s, UTC]",
                ),
                id="non-ISO8601, UTC",
            ),
        ],
    )
    def test_to_datetime_mixed_offsets_with_none(self, fmt, expected):
        # 在下面的测试函数中，测试处理不同偏移量混合情况下时区为 None 的情况
        # https://github.com/pandas-dev/pandas/issues/50071
        result = to_datetime(
            ["2000-01-01 09:00:00+01:00", "2000-01-02 02:00:00+02:00", None],
            format=fmt,
            utc=True,
        )
        # 使用 pytest 的断言来比较结果是否符合预期
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "fmt",
        ["%Y-%d-%m %H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z"],
        ids=["non-ISO8601 format", "ISO8601 format"],
    )
    @pytest.mark.parametrize(
        "args",
        [
            pytest.param(
                ["2000-01-01 01:00:00-08:00", "2000-01-01 02:00:00-07:00"],
                id="all tz-aware, mixed timezones, without utc",
            ),
        ],
    )
    @pytest.mark.parametrize(
        "constructor",
        [Timestamp, lambda x: Timestamp(x).to_pydatetime()],
    )


    # 使用 pytest 的 parametrize 装饰器为测试方法参数化，参数是一个列表
    def test_to_datetime_mixed_datetime_and_string_with_format_raises(
        self, fmt, args, constructor
    ):
        # https://github.com/pandas-dev/pandas/issues/49298
        # 注释：此测试用例用来验证当使用混合的日期时间和字符串格式时，是否会抛出指定异常
        # note: ISO8601 formats go down a fastpath, so we need to check both
        # a ISO8601 format and a non-ISO8601 one
        ts1 = constructor(args[0])  # 创建时间戳对象 ts1
        ts2 = constructor(args[1])  # 创建时间戳对象 ts2
        with pytest.raises(
            ValueError, match="cannot be converted to datetime64 unless utc=True"
        ):
            # 检查在不使用 utc=True 的情况下，是否能将时间戳列表转换为 datetime64
            to_datetime([ts1, ts2], format=fmt, utc=False)

    def test_to_datetime_np_str(self):
        # GH#32264
        # GH#48969
        value = np.str_("2019-02-04 10:18:46.297000+0000")

        ser = Series([value])  # 创建包含字符串 value 的 Series 对象

        exp = Timestamp("2019-02-04 10:18:46.297000", tz="UTC")  # 创建预期的 Timestamp 对象 exp

        assert to_datetime(value) == exp  # 验证 to_datetime 函数能否正确处理字符串 value
        assert to_datetime(ser.iloc[0]) == exp  # 验证 to_datetime 函数能否正确处理 Series 对象中的字符串

        res = to_datetime([value])  # 将字符串 value 转换为时间戳列表
        expected = Index([exp])  # 创建预期的索引对象 expected
        tm.assert_index_equal(res, expected)  # 验证转换结果与预期索引对象是否相等

        res = to_datetime(ser)  # 将 Series 对象转换为时间戳对象
        expected = Series(expected)  # 创建预期的 Series 对象
        tm.assert_series_equal(res, expected)  # 验证转换结果与预期 Series 对象是否相等

    @pytest.mark.parametrize(
        "s, _format, dt",
        [
            ["2015-1-1", "%G-%V-%u", datetime(2014, 12, 29, 0, 0)],
            ["2015-1-4", "%G-%V-%u", datetime(2015, 1, 1, 0, 0)],
            ["2015-1-7", "%G-%V-%u", datetime(2015, 1, 4, 0, 0)],
        ],
    )


    # 使用 pytest 的 parametrize 装饰器为测试方法参数化，参数是一个列表
    def test_to_datetime_iso_week_year_format(self, s, _format, dt):
        # See GH#16607
        # 注释：此测试用例用来验证 ISO week 年格式的日期字符串能否正确转换为指定的 datetime 对象
        assert to_datetime(s, format=_format) == dt  # 验证 to_datetime 函数的转换结果与预期 datetime 对象是否相等

    )
    @pytest.mark.parametrize("errors", ["raise", "coerce"])


    # 使用 pytest 的 parametrize 装饰器为测试方法参数化，参数是一个列表
    def test_error_iso_week_year(self, msg, s, _format, errors):
        # See GH#16607, GH#50308
        # 注释：此测试用例用来验证在给定错误的 ISO week 年格式时，是否会抛出指定的异常
        # This test checks for errors thrown when giving the wrong format
        # However, as discussed on PR#25541, overriding the locale
        # causes a different error to be thrown due to the format being
        # locale specific, but the test data is in english.
        # Therefore, the tests only run when locale is not overwritten,
        # as a sort of solution to this problem.
        if locale.getlocale() != ("zh_CN", "UTF-8") and locale.getlocale() != (
            "it_IT",
            "UTF-8",
        ):
            with pytest.raises(ValueError, match=msg):
                # 验证在给定的错误格式和错误处理方式时，to_datetime 函数是否会抛出指定的异常
                to_datetime(s, format=_format, errors=errors)

    @pytest.mark.parametrize("tz", [None, "US/Central"])


    # 使用 pytest 的 parametrize 装饰器为测试方法参数化，参数是一个列表
    def test_to_datetime_dtarr(self, tz):
        # DatetimeArray
        dti = date_range("1965-04-03", periods=19, freq="2W", tz=tz)  # 创建一个带有时区的日期时间索引对象
        arr = dti._data  # 获取日期时间索引对象的内部数据

        result = to_datetime(arr)  # 将日期时间数组转换为时间戳数组
        assert result is arr  # 验证转换结果与原始数据对象是否相同
    # 在 Windows 上不工作，因为 tzpath 没有设置正确
    @td.skip_if_windows
    @pytest.mark.parametrize("utc", [True, False])
    @pytest.mark.parametrize("tz", [None, "US/Central"])
    def test_to_datetime_arrow(self, tz, utc, index_or_series):
        # 导入 pyarrow 库，如果不存在则跳过测试
        pa = pytest.importorskip("pyarrow")

        # 创建一个时间范围从 "1965-04-03" 开始，以 2 周为频率，包含 19 个时间点的 DatetimeIndex 或 Series
        dti = date_range("1965-04-03", periods=19, freq="2W", tz=tz)
        dti = index_or_series(dti)

        # 将 DatetimeIndex 或 Series 转换为 pyarrow 的 ArrowDtype 类型
        dti_arrow = dti.astype(pd.ArrowDtype(pa.timestamp(unit="ns", tz=tz)))

        # 调用 to_datetime 函数进行转换
        result = to_datetime(dti_arrow, utc=utc)
        # 期望的转换结果，根据 utc 参数调整时区
        expected = to_datetime(dti, utc=utc).astype(
            pd.ArrowDtype(pa.timestamp(unit="ns", tz=tz if not utc else "UTC"))
        )
        if not utc and index_or_series is not Series:
            # 对于非 UTC 转换且不是 Series 的情况，检查返回的对象是否和原始对象相同
            # 因为 to_datetime 会返回一个新的对象
            assert result is dti_arrow
        if index_or_series is Series:
            # 如果是 Series，比较结果是否相等
            tm.assert_series_equal(result, expected)
        else:
            # 如果是 Index，比较结果是否相等
            tm.assert_index_equal(result, expected)

    def test_to_datetime_pydatetime(self):
        # 测试将 Python 的 datetime 对象转换为 pandas 的 Timestamp 对象
        actual = to_datetime(datetime(2008, 1, 15))
        assert actual == datetime(2008, 1, 15)

    def test_to_datetime_YYYYMMDD(self):
        # 测试将字符串形式的日期 "20080115" 转换为 pandas 的 Timestamp 对象
        actual = to_datetime("20080115")
        assert actual == datetime(2008, 1, 15)

    @td.skip_if_windows  # `tm.set_timezone` 在 Windows 上不可用
    @pytest.mark.skipif(WASM, reason="tzset 在 WASM 上不可用")
    def test_to_datetime_now(self):
        # 参考 GitHub issue #18666
        with tm.set_timezone("US/Eastern"):
            # 参考 GitHub issue #18705
            now = Timestamp("now")
            pdnow = to_datetime("now")
            pdnow2 = to_datetime(["now"])[0]

            # 这些值应该非常接近，给定 10 秒的余地
            assert abs(pdnow._value - now._value) < 1e10
            assert abs(pdnow2._value - now._value) < 1e10

            # 检查转换后的对象是否没有时区信息
            assert pdnow.tzinfo is None
            assert pdnow2.tzinfo is None

    @td.skip_if_windows  # `tm.set_timezone` 在 Windows 上不可用
    @pytest.mark.skipif(WASM, reason="tzset 在 WASM 上不可用")
    @pytest.mark.parametrize("tz", ["Pacific/Auckland", "US/Samoa"])
    def test_to_datetime_today(self, tz):
        # 用于测试将"today"转换为不同时间格式
        # 设置时区，以便在特定时区下执行以下操作
        with tm.set_timezone(tz):
            # 使用NumPy将"today"转换为纳秒精度的时间戳，并转换为整数
            nptoday = np.datetime64("today").astype("datetime64[us]").astype(np.int64)
            # 使用pandas将"today"转换为时间戳对象
            pdtoday = to_datetime("today")
            # 使用pandas将"today"转换为时间戳对象的数组的第一个元素
            pdtoday2 = to_datetime(["today"])[0]

            # 使用pandas创建当前时间戳对象
            tstoday = Timestamp("today")
            # 使用pandas创建当前时间的时间戳对象
            tstoday2 = Timestamp.today()

            # 断言以下这些时间戳对象应该相等，精度误差在10秒以内
            assert abs(pdtoday.normalize()._value - nptoday) < 1e10
            assert abs(pdtoday2.normalize()._value - nptoday) < 1e10
            assert abs(pdtoday._value - tstoday._value) < 1e10
            assert abs(pdtoday._value - tstoday2._value) < 1e10

            # 断言pdtoday和pdtoday2的时区信息应为None
            assert pdtoday.tzinfo is None
            assert pdtoday2.tzinfo is None

    @pytest.mark.parametrize("arg", ["now", "today"])
    def test_to_datetime_today_now_unicode_bytes(self, arg):
        # 调用to_datetime函数测试处理Unicode字符串和字节串的"now"和"today"
        to_datetime([arg])

    @pytest.mark.filterwarnings("ignore:Timestamp.utcnow is deprecated:FutureWarning")
    @pytest.mark.skipif(WASM, reason="tzset is not available on WASM")
    @pytest.mark.parametrize(
        "format, expected_ds",
        [
            ("%Y-%m-%d %H:%M:%S%z", "2020-01-03"),
            ("%Y-%d-%m %H:%M:%S%z", "2020-03-01"),
            (None, "2020-01-03"),
        ],
    )
    @pytest.mark.parametrize(
        "string, attribute",
        [
            ("now", "utcnow"),
            ("today", "today"),
        ],
    )
    def test_to_datetime_now_with_format(self, format, expected_ds, string, attribute):
        # https://github.com/pandas-dev/pandas/issues/50359
        # 使用指定格式和UTC时区将日期时间字符串转换为时间戳对象
        result = to_datetime(["2020-01-03 00:00:00Z", string], format=format, utc=True)
        # 生成预期的DatetimeIndex对象，用于断言结果是否与预期相符
        expected = DatetimeIndex(
            [expected_ds, getattr(Timestamp, attribute)()], dtype="datetime64[s, UTC]"
        )
        assert (expected - result).max().total_seconds() < 1

    @pytest.mark.parametrize(
        "dt", [np.datetime64("2000-01-01"), np.datetime64("2000-01-02")]
    )
    def test_to_datetime_dt64s(self, cache, dt):
        # 断言使用to_datetime函数将np.datetime64对象转换为Timestamp对象是否正确
        assert to_datetime(dt, cache=cache) == Timestamp(dt)

    @pytest.mark.parametrize(
        "arg, format",
        [
            ("2001-01-01", "%Y-%m-%d"),
            ("01-01-2001", "%d-%m-%Y"),
        ],
    )
    def test_to_datetime_dt64s_and_str(self, arg, format):
        # https://github.com/pandas-dev/pandas/issues/50036
        # 调用to_datetime函数测试处理日期时间字符串和np.datetime64对象
        result = to_datetime([arg, np.datetime64("2020-01-01")], format=format)
        # 生成预期的DatetimeIndex对象，用于断言结果是否与预期相符
        expected = DatetimeIndex(["2001-01-01", "2020-01-01"])
        tm.assert_index_equal(result, expected)
    @pytest.mark.parametrize(
        "dt", [np.datetime64("1000-01-01"), np.datetime64("5000-01-02")]
    )
    @pytest.mark.parametrize("errors", ["raise", "coerce"])
    # 定义测试函数，测试处理超出纳秒边界的 np.datetime64 对象
    def test_to_datetime_dt64s_out_of_ns_bounds(self, cache, dt, errors):
        # GH#50369 我们转换为最接近支持的分辨率，即 "s"
        ts = to_datetime(dt, errors=errors, cache=cache)
        # 断言返回的对象是 Timestamp 类的实例
        assert isinstance(ts, Timestamp)
        # 断言返回的 Timestamp 对象的单位是 "s"
        assert ts.unit == "s"
        # 断言返回的 Timestamp 对象的原始 np.datetime64 数据等于输入的 dt
        assert ts.asm8 == dt

        ts = Timestamp(dt)
        # 再次断言 Timestamp 对象的单位是 "s"
        assert ts.unit == "s"
        # 再次断言 Timestamp 对象的原始 np.datetime64 数据等于输入的 dt
        assert ts.asm8 == dt

    @pytest.mark.skip_ubsan
    # 测试处理超出整型边界的 np.datetime64 对象
    def test_to_datetime_dt64d_out_of_bounds(self, cache):
        # 创建一个超出整型边界的 np.datetime64 对象
        dt64 = np.datetime64(np.iinfo(np.int64).max, "D")

        # 预期的异常消息
        msg = "Out of bounds second timestamp: 25252734927768524-07-27"
        # 使用 pytest 检查是否会引发 OutOfBoundsDatetime 异常并匹配预期的消息
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp(dt64)
        # 使用 pytest 检查 to_datetime() 函数是否会引发 OutOfBoundsDatetime 异常并匹配预期的消息
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(dt64, errors="raise", cache=cache)

        # 断言当 errors="coerce" 时，处理超出边界的 np.datetime64 对象返回 NaT
        assert to_datetime(dt64, errors="coerce", cache=cache) is NaT

    @pytest.mark.parametrize("unit", ["s", "D"])
    # 测试处理 np.datetime64 对象数组
    def test_to_datetime_array_of_dt64s(self, cache, unit):
        # https://github.com/pandas-dev/pandas/issues/31491
        # 需要至少 50 个元素以确保缓存被使用
        dts = [
            np.datetime64("2000-01-01", unit),
            np.datetime64("2000-01-02", unit),
        ] * 30
        # 假设所有日期时间都在有效范围内，to_datetime() 返回与 Timestamp() 解析相等的数组
        result = to_datetime(dts, cache=cache)
        expected = DatetimeIndex([Timestamp(x).asm8 for x in dts], dtype="M8[s]")

        # 使用 pandas 测试工具检查结果与预期是否相等
        tm.assert_index_equal(result, expected)

        # 创建一个包含超出边界的 np.datetime64 对象的列表
        dts_with_oob = dts + [np.datetime64("9999-01-01")]

        # 截至 GH#51978，这种情况下我们不会引发异常
        to_datetime(dts_with_oob, errors="raise")

        # 断言处理超出边界的 np.datetime64 对象时，使用 errors="coerce" 返回预期的 DatetimeIndex
        result = to_datetime(dts_with_oob, errors="coerce", cache=cache)
        expected = DatetimeIndex(np.array(dts_with_oob, dtype="M8[s]"))
        tm.assert_index_equal(result, expected)

    def test_to_datetime_tz(self, cache):
        # xref 8260
        # uniform 返回一个带有时区信息的 DatetimeIndex
        arr = [
            Timestamp("2013-01-01 13:00:00-0800", tz="US/Pacific"),
            Timestamp("2013-01-02 14:00:00-0800", tz="US/Pacific"),
        ]
        # 测试处理带有时区信息的 Timestamp 对象数组
        result = to_datetime(arr, cache=cache)
        expected = DatetimeIndex(
            ["2013-01-01 13:00:00", "2013-01-02 14:00:00"], tz="US/Pacific"
        ).as_unit("s")
        # 使用 pandas 测试工具检查结果与预期是否相等
        tm.assert_index_equal(result, expected)
    def test_to_datetime_tz_mixed(self, cache):
        # mixed tzs will raise if errors='raise'
        # 如果 errors='raise'，混合时区会引发异常
        # https://github.com/pandas-dev/pandas/issues/50585
        # 参考 GitHub issue #50585
        arr = [
            Timestamp("2013-01-01 13:00:00", tz="US/Pacific"),
            Timestamp("2013-01-02 14:00:00", tz="US/Eastern"),
        ]
        msg = (
            "Tz-aware datetime.datetime cannot be "
            "converted to datetime64 unless utc=True"
        )
        # 设置断言，期望捕获 ValueError 异常，并匹配特定错误消息
        with pytest.raises(ValueError, match=msg):
            to_datetime(arr, cache=cache)

        result = to_datetime(arr, cache=cache, errors="coerce")
        # 期望得到的结果
        expected = DatetimeIndex(
            ["2013-01-01 13:00:00-08:00", "NaT"], dtype="datetime64[s, US/Pacific]"
        )
        # 断言结果与期望结果相等
        tm.assert_index_equal(result, expected)

    def test_to_datetime_different_offsets_removed(self, cache):
        # inspired by asv timeseries.ToDatetimeNONISO8601 benchmark
        # 受 asv timeseries.ToDatetimeNONISO8601 基准测试启发
        # see GH-26097 for more
        # 更多信息请参见 GitHub issue #26097
        # GH#57275
        ts_string_1 = "March 1, 2018 12:00:00+0400"
        ts_string_2 = "March 1, 2018 12:00:00+0500"
        arr = [ts_string_1] * 5 + [ts_string_2] * 5
        msg = "Mixed timezones detected. Pass utc=True in to_datetime"
        # 设置断言，期望捕获 ValueError 异常，并匹配特定错误消息
        with pytest.raises(ValueError, match=msg):
            to_datetime(arr, cache=cache)

    def test_to_datetime_tz_pytz(self, cache):
        # see gh-8260
        # 参见 GitHub issue #8260
        pytz = pytest.importorskip("pytz")
        us_eastern = pytz.timezone("US/Eastern")
        arr = np.array(
            [
                us_eastern.localize(
                    datetime(year=2000, month=1, day=1, hour=3, minute=0)
                ),
                us_eastern.localize(
                    datetime(year=2000, month=6, day=1, hour=3, minute=0)
                ),
            ],
            dtype=object,
        )
        result = to_datetime(arr, utc=True, cache=cache)
        expected = DatetimeIndex(
            ["2000-01-01 08:00:00+00:00", "2000-06-01 07:00:00+00:00"],
            dtype="datetime64[us, UTC]",
            freq=None,
        )
        # 断言结果与期望结果相等
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "init_constructor, end_constructor",
        [
            (Index, DatetimeIndex),
            (list, DatetimeIndex),
            (np.array, DatetimeIndex),
            (Series, Series),
        ],
    )
    def test_to_datetime_utc_true(self, cache, init_constructor, end_constructor):
        # See gh-11934 & gh-6415
        # 参见 GitHub issue #11934 和 #6415
        data = ["20100102 121314", "20100102 121315"]
        expected_data = [
            Timestamp("2010-01-02 12:13:14", tz="utc"),
            Timestamp("2010-01-02 12:13:15", tz="utc"),
        ]

        result = to_datetime(
            init_constructor(data), format="%Y%m%d %H%M%S", utc=True, cache=cache
        )
        expected = end_constructor(expected_data)
        # 断言结果与期望结果相等
        tm.assert_equal(result, expected)
    @pytest.mark.parametrize(
        "scalar, expected",
        [
            ["20100102 121314", Timestamp("2010-01-02 12:13:14", tz="utc")],
            ["20100102 121315", Timestamp("2010-01-02 12:13:15", tz="utc")],
        ],
    )
    # 参数化测试，针对不同的输入参数进行多次测试
    def test_to_datetime_utc_true_scalar(self, cache, scalar, expected):
        # 测试单个标量输入情况
        result = to_datetime(scalar, format="%Y%m%d %H%M%S", utc=True, cache=cache)
        # 断言结果是否与期望值相等
        assert result == expected

    def test_to_datetime_utc_true_with_series_single_value(self, cache):
        # GH 15760 UTC=True with Series
        ts = 1.5e18
        # 使用带有 Series 的输入进行 UTC=True 测试
        result = to_datetime(Series([ts]), utc=True, cache=cache)
        expected = Series([Timestamp(ts, tz="utc")])
        # 断言 Series 结果是否与期望值相等
        tm.assert_series_equal(result, expected)

    def test_to_datetime_utc_true_with_series_tzaware_string(self, cache):
        ts = "2013-01-01 00:00:00-01:00"
        expected_ts = "2013-01-01 01:00:00"
        data = Series([ts] * 3)
        # 使用带有 Series 的时区感知字符串输入进行 UTC=True 测试
        result = to_datetime(data, utc=True, cache=cache)
        expected = Series([Timestamp(expected_ts, tz="utc")] * 3)
        # 断言 Series 结果是否与期望值相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "date, dtype",
        [
            ("2013-01-01 01:00:00", "datetime64[ns]"),
            ("2013-01-01 01:00:00", "datetime64[ns, UTC]"),
        ],
    )
    # 参数化测试，针对不同的输入参数进行多次测试
    def test_to_datetime_utc_true_with_series_datetime_ns(self, cache, date, dtype):
        expected = Series(
            [Timestamp("2013-01-01 01:00:00", tz="UTC")], dtype="M8[ns, UTC]"
        )
        # 使用带有 Series 的 datetime64 输入进行 UTC=True 测试
        result = to_datetime(Series([date], dtype=dtype), utc=True, cache=cache)
        # 断言 Series 结果是否与期望值相等
        tm.assert_series_equal(result, expected)
    # 定义测试函数 test_to_datetime_tz_psycopg2，使用 pytest 参数 request 和 cache
    def test_to_datetime_tz_psycopg2(self, request, cache):
        # 导入并验证是否存在 psycopg2.tz 模块，否则跳过测试
        psycopg2_tz = pytest.importorskip("psycopg2.tz")

        # 创建两个固定偏移时间戳对象 tz1 和 tz2
        tz1 = psycopg2_tz.FixedOffsetTimezone(offset=-300, name=None)
        tz2 = psycopg2_tz.FixedOffsetTimezone(offset=-240, name=None)

        # 创建包含两个日期时间对象的 NumPy 数组 arr，分别带有 tz1 和 tz2 偏移信息
        arr = np.array(
            [
                datetime(2000, 1, 1, 3, 0, tzinfo=tz1),
                datetime(2000, 6, 1, 3, 0, tzinfo=tz2),
            ],
            dtype=object,
        )

        # 调用 to_datetime 函数，将 arr 转换为日期时间索引对象 result
        result = to_datetime(arr, errors="coerce", utc=True, cache=cache)

        # 创建预期的日期时间索引对象 expected
        expected = DatetimeIndex(
            ["2000-01-01 08:00:00+00:00", "2000-06-01 07:00:00+00:00"],
            dtype="datetime64[us, UTC]",
            freq=None,
        )

        # 断言 result 与 expected 相等
        tm.assert_index_equal(result, expected)

        # 创建只包含一个日期时间对象的日期时间索引 i，带有特定的时区信息
        i = DatetimeIndex(
            ["2000-01-01 08:00:00"],
            tz=psycopg2_tz.FixedOffsetTimezone(offset=-300, name=None),
        ).as_unit("us")

        # 断言 i 不是 datetime64 纳秒类型
        assert not is_datetime64_ns_dtype(i)

        # 调用 to_datetime 函数，将 i 转换为日期时间索引对象 result
        result = to_datetime(i, errors="coerce", cache=cache)

        # 断言 result 与 i 相等
        tm.assert_index_equal(result, i)

        # 再次调用 to_datetime 函数，将 i 转换为日期时间索引对象 result，并强制使用 UTC
        result = to_datetime(i, errors="coerce", utc=True, cache=cache)

        # 创建预期的日期时间索引对象 expected
        expected = DatetimeIndex(["2000-01-01 13:00:00"], dtype="datetime64[us, UTC]")

        # 断言 result 与 expected 相等
        tm.assert_index_equal(result, expected)
    # 定义一个测试函数，用于测试当输入格式不合法时是否能正确抛出异常
    def test_invalid_format_raises(self, errors):
        # 使用 pytest 的 raises 方法验证是否会抛出 ValueError 异常，并检查异常消息中是否包含特定字符串
        with pytest.raises(
            ValueError, match="':' is a bad directive in format 'H%:M%:S%"
        ):
            # 调用 to_datetime 函数，尝试将时间字符串转换为日期时间对象
            to_datetime(["00:00:00"], format="H%:M%:S%", errors=errors)

    # 使用 pytest 的参数化标记，定义一个测试函数，用于测试当输入的时间字符串不合法时是否能正确处理
    @pytest.mark.parametrize("value", ["a", "00:01:99"])
    @pytest.mark.parametrize("format", [None, "%H:%M:%S"])
    def test_datetime_invalid_scalar(self, value, format):
        # GH24763
        # 调用 to_datetime 函数，尝试将不合法的时间字符串转换为日期时间对象，期望返回 NaT（Not a Time）
        res = to_datetime(value, errors="coerce", format=format)
        assert res is NaT

        # 构造异常消息的正则表达式模式，验证是否会抛出 ValueError 异常，并检查异常消息是否符合预期
        msg = "|".join(
            [
                r'^time data "a" doesn\'t match format "%H:%M:%S", at position 0. '
                f"{PARSING_ERR_MSG}$",
                r'^Given date string "a" not likely a datetime, at position 0$',
                r'^unconverted data remains when parsing with format "%H:%M:%S": "9", '
                f"at position 0. {PARSING_ERR_MSG}$",
                r"^second must be in 0..59: 00:01:99, at position 0$",
            ]
        )
        # 使用 pytest 的 raises 方法验证是否会抛出 ValueError 异常，并检查异常消息是否符合预期
        with pytest.raises(ValueError, match=msg):
            to_datetime(value, errors="raise", format=format)

    # 使用 pytest 的参数化标记，定义一个测试函数，用于测试当输入的时间字符串超出范围时是否能正确处理
    @pytest.mark.parametrize("value", ["3000/12/11 00:00:00"])
    @pytest.mark.parametrize("format", [None, "%H:%M:%S"])
    def test_datetime_outofbounds_scalar(self, value, format):
        # GH24763
        # 调用 to_datetime 函数，尝试将超出范围的时间字符串转换为日期时间对象，期望返回 NaT
        res = to_datetime(value, errors="coerce", format=format)
        if format is None:
            # 如果 format 为 None，则预期返回一个 Timestamp 对象，且与输入值相等
            assert isinstance(res, Timestamp)
            assert res == Timestamp(value)
        else:
            # 如果 format 不为 None，则预期返回 NaT
            assert res is NaT

        if format is not None:
            # 如果 format 不为 None，构造异常消息的正则表达式模式，验证是否会抛出 ValueError 异常，并检查异常消息是否符合预期
            msg = r'^time data ".*" doesn\'t match format ".*", at position 0.'
            with pytest.raises(ValueError, match=msg):
                to_datetime(value, errors="raise", format=format)
        else:
            # 如果 format 为 None，预期返回一个 Timestamp 对象，且与输入值相等
            res = to_datetime(value, errors="raise", format=format)
            assert isinstance(res, Timestamp)
            assert res == Timestamp(value)

    # 使用 pytest 的参数化标记，定义一个测试函数，用于测试当输入的时间字符串列表中存在不合法值时是否能正确处理
    @pytest.mark.parametrize(
        ("values"), [(["a"]), (["00:01:99"]), (["a", "b", "99:00:00"])]
    )
    @pytest.mark.parametrize("format", [(None), ("%H:%M:%S")])
    # 定义一个测试方法，用于测试处理无效索引日期时间的情况
    def test_datetime_invalid_index(self, values, format):
        # GH24763：GitHub issue编号，标识这段代码解决的问题
        # 在测试中包含逻辑不是最佳实践，但这个测试难以参数化
        if format is None and len(values) > 1:
            # 如果日期格式为None且值的数量大于1，则发出UserWarning警告
            warn = UserWarning
        else:
            warn = None

        # 使用tm.assert_produces_warning上下文管理器来检查警告消息是否符合预期
        with tm.assert_produces_warning(
            warn, match="Could not infer format", raise_on_extra_warnings=False
        ):
            # 调用to_datetime函数尝试将日期时间字符串转换为DatetimeIndex对象
            res = to_datetime(values, errors="coerce", format=format)
        # 使用tm.assert_index_equal函数断言返回的DatetimeIndex对象是否符合预期
        tm.assert_index_equal(res, DatetimeIndex([NaT] * len(values)))

        # 构建消息正则表达式，用于检查值错误时引发的特定异常
        msg = "|".join(
            [
                r'^Given date string "a" not likely a datetime, at position 0$',
                r'^time data "a" doesn\'t match format "%H:%M:%S", at position 0. '
                f"{PARSING_ERR_MSG}$",
                r'^unconverted data remains when parsing with format "%H:%M:%S": "9", '
                f"at position 0. {PARSING_ERR_MSG}$",
                r"^second must be in 0..59: 00:01:99, at position 0$",
            ]
        )
        # 使用pytest.raises断言特定的值错误异常是否被引发，并检查异常消息是否匹配预期
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(
                warn, match="Could not infer format", raise_on_extra_warnings=False
            ):
                # 调用to_datetime函数尝试将日期时间字符串转换为DatetimeIndex对象，期望引发异常
                to_datetime(values, errors="raise", format=format)

    # 使用pytest的参数化标记定义一个参数化测试方法，测试to_datetime函数的缓存行为
    @pytest.mark.parametrize("utc", [True, None])
    @pytest.mark.parametrize("format", ["%Y%m%d %H:%M:%S", None])
    @pytest.mark.parametrize("constructor", [list, tuple, np.array, Index, deque])
    def test_to_datetime_cache(self, utc, format, constructor):
        # 定义一个日期时间字符串作为测试数据
        date = "20130101 00:00:00"
        # 创建一个包含大量相同日期时间字符串的数据结构
        test_dates = [date] * 10**5
        # 使用构造器创建数据结构，如列表、元组、numpy数组、索引对象或双端队列
        data = constructor(test_dates)

        # 调用to_datetime函数，测试带缓存和不带缓存的结果是否相等
        result = to_datetime(data, utc=utc, format=format, cache=True)
        expected = to_datetime(data, utc=utc, format=format, cache=False)

        # 使用tm.assert_index_equal断言两个DatetimeIndex对象是否相等
        tm.assert_index_equal(result, expected)

    # 测试从deque对象创建日期时间的行为
    def test_to_datetime_from_deque(self):
        # GH 29403：GitHub issue编号，标识这段代码解决的问题
        # 使用deque创建包含多个Timestamp对象的数据
        result = to_datetime(deque([Timestamp("2010-06-02 09:30:00")] * 51))
        # 创建一个期望的DatetimeIndex对象，包含多个相同的Timestamp对象
        expected = to_datetime([Timestamp("2010-06-02 09:30:00")] * 51)
        # 使用tm.assert_index_equal断言两个DatetimeIndex对象是否相等
        tm.assert_index_equal(result, expected)

    # 使用pytest的参数化标记定义一个参数化测试方法，测试Series对象在to_datetime函数中的缓存行为
    @pytest.mark.parametrize("utc", [True, None])
    @pytest.mark.parametrize("format", ["%Y%m%d %H:%M:%S", None])
    def test_to_datetime_cache_series(self, utc, format):
        # 定义一个日期时间字符串作为测试数据
        date = "20130101 00:00:00"
        # 创建一个包含大量相同日期时间字符串的Series对象
        test_dates = [date] * 10**5
        data = Series(test_dates)
        # 调用to_datetime函数，测试带缓存和不带缓存的结果是否相等
        result = to_datetime(data, utc=utc, format=format, cache=True)
        expected = to_datetime(data, utc=utc, format=format, cache=False)
        # 使用tm.assert_series_equal断言两个Series对象是否相等
        tm.assert_series_equal(result, expected)

    # 测试将标量日期时间字符串转换为Timestamp对象，并使用缓存
    def test_to_datetime_cache_scalar(self):
        # 定义一个日期时间字符串作为测试数据
        date = "20130101 00:00:00"
        # 调用to_datetime函数，测试将标量日期时间字符串转换为Timestamp对象的结果
        result = to_datetime(date, cache=True)
        expected = Timestamp("20130101 00:00:00")
        # 使用assert语句断言结果是否等于预期值
        assert result == expected
    # 使用 pytest 的 @pytest.mark.parametrize 装饰器定义参数化测试用例
    @pytest.mark.parametrize(
        # 参数化测试用例的参数：
        "datetimelikes,expected_values,exp_unit",
        (
            # 第一个测试用例参数和预期结果：
            (
                (None, np.nan) + (NaT,) * start_caching_at,  # 输入时间对象和缓存开始时间
                (NaT,) * (start_caching_at + 2),  # 预期输出时间对象和缓存开始时间
                "s",  # 时间单位为秒
            ),
            # 第二个测试用例参数和预期结果：
            (
                (None, Timestamp("2012-07-26")) + (NaT,) * start_caching_at,  # 输入时间对象和缓存开始时间
                (NaT, Timestamp("2012-07-26")) + (NaT,) * start_caching_at,  # 预期输出时间对象和缓存开始时间
                "s",  # 时间单位为秒
            ),
            # 第三个测试用例参数和预期结果：
            (
                (None,)  # 输入时间对象
                + (NaT,) * start_caching_at  # 缓存开始时间
                + ("2012 July 26", Timestamp("2012-07-26")),  # 其他时间对象
                (NaT,) * (start_caching_at + 1)  # 预期输出时间对象
                + (Timestamp("2012-07-26"), Timestamp("2012-07-26")),  # 预期输出时间对象
                "s",  # 时间单位为秒
            ),
        ),
    )
    # 定义测试方法：测试将对象转换为日期时间类型并进行缓存
    def test_convert_object_to_datetime_with_cache(
        self, datetimelikes, expected_values, exp_unit
    ):
        # GH#39882
        # 创建 Series 对象，其中包含日期时间对象，数据类型为 'object'
        ser = Series(
            datetimelikes,
            dtype="object",
        )
        # 使用 to_datetime 函数将 Series 转换为日期时间类型的结果
        result_series = to_datetime(ser, errors="coerce")
        # 创建预期的 Series 对象，数据类型为指定的日期时间单位
        expected_series = Series(
            expected_values,
            dtype=f"datetime64[{exp_unit}]",
        )
        # 使用 pytest 的 tm.assert_series_equal 函数比较结果 Series 和预期 Series
        tm.assert_series_equal(result_series, expected_series)

    # 使用 pytest 的 @pytest.mark.parametrize 装饰器定义参数化测试用例
    @pytest.mark.parametrize(
        # 参数化测试用例的参数：
        "input",
        [
            # 不同的输入 Series 对象，包含 NaT 和 None 值，数据类型为 'object'
            Series([NaT] * 20 + [None] * 20, dtype="object"),
            Series([NaT] * 60 + [None] * 60, dtype="object"),
            Series([None] * 20),
            Series([None] * 60),
            Series([""] * 20),
            Series([""] * 60),
            Series([pd.NA] * 20),
            Series([pd.NA] * 60),
            Series([np.nan] * 20),
            Series([np.nan] * 60),
        ],
    )
    # 定义测试方法：测试将空值类转换为 NaT
    def test_to_datetime_converts_null_like_to_nat(self, cache, input):
        # GH35888
        # 创建预期的 Series 对象，其中所有值均为 NaT，数据类型为 'M8[s]'
        expected = Series([NaT] * len(input), dtype="M8[s]")
        # 使用 to_datetime 函数将输入 Series 转换为日期时间类型的结果
        result = to_datetime(input, cache=cache)
        # 使用 pytest 的 tm.assert_series_equal 函数比较结果 Series 和预期 Series
        tm.assert_series_equal(result, expected)

    # 使用 pytest 的 @pytest.mark.parametrize 装饰器定义参数化测试用例
    @pytest.mark.parametrize(
        # 参数化测试用例的参数：
        "date, format",
        [
            # 不支持的日期格式和格式化字符串
            ("2017-20", "%Y-%W"),
            ("20 Sunday", "%W %A"),
            ("20 Sun", "%W %a"),
            ("2017-21", "%Y-%U"),
            ("20 Sunday", "%U %A"),
            ("20 Sun", "%U %a"),
        ],
    )
    # 定义测试方法：测试使用无年份和日期的周信息
    def test_week_without_day_and_calendar_year(self, date, format):
        # GH16774
        # 指定错误消息
        msg = "Cannot use '%W' or '%U' without day and year"
        # 使用 pytest 的 pytest.raises 检查是否引发 ValueError 异常，并验证异常消息
        with pytest.raises(ValueError, match=msg):
            to_datetime(date, format=format)

    # 定义测试方法：测试在特定条件下引发 ValueError 异常
    def test_to_datetime_coerce(self):
        # GH#26122, GH#57275
        # 包含混合时区的时间字符串列表
        ts_strings = [
            "March 1, 2018 12:00:00+0400",
            "March 1, 2018 12:00:00+0500",
            "20100240",
        ]
        # 指定错误消息
        msg = "Mixed timezones detected. Pass utc=True in to_datetime"
        # 使用 pytest 的 pytest.raises 检查是否引发 ValueError 异常，并验证异常消息
        with pytest.raises(ValueError, match=msg):
            to_datetime(ts_strings, errors="coerce")

    # 使用 pytest 的 @pytest.mark.parametrize 装饰器定义参数化测试用例
    @pytest.mark.parametrize(
        # 参数化测试用例的参数：
        "string_arg, format",
        [("March 1, 2018", "%B %d, %Y"), ("2018-03-01", "%Y-%m-%d")],
    )
    @pytest.mark.parametrize(
        "outofbounds",
        [  # 参数化测试，用于多次运行同一个测试函数，每次使用不同的参数
            datetime(9999, 1, 1),  # 使用 datetime 类创建一个日期对象
            date(9999, 1, 1),  # 使用 date 类创建一个日期对象
            np.datetime64("9999-01-01"),  # 使用 numpy 的 datetime64 类型创建日期对象
            "January 1, 9999",  # 字符串形式的日期
            "9999-01-01",  # ISO 格式的字符串日期
        ],
    )
    def test_to_datetime_coerce_oob(self, string_arg, format, outofbounds):
        # 处理日期超出范围的情况，详细描述在 GitHub 问题页面上
        ts_strings = [string_arg, outofbounds]  # 创建日期字符串列表
        result = to_datetime(ts_strings, errors="coerce", format=format)  # 调用 to_datetime 函数进行日期转换
        if isinstance(outofbounds, str) and (
            format.startswith("%B") ^ outofbounds.startswith("J")
        ):
            # 如果日期字符串与给定格式不匹配，则抛出异常，我们进行强制转换
            expected = DatetimeIndex([datetime(2018, 3, 1), NaT], dtype="M8[s]")  # 创建日期索引对象
        elif isinstance(outofbounds, datetime):
            expected = DatetimeIndex(
                [datetime(2018, 3, 1), outofbounds], dtype="M8[us]"
            )  # 创建日期索引对象，包括指定的日期对象
        else:
            expected = DatetimeIndex([datetime(2018, 3, 1), outofbounds], dtype="M8[s]")  # 创建日期索引对象
        tm.assert_index_equal(result, expected)  # 断言结果与预期相等

    def test_to_datetime_malformed_no_raise(self):
        # GH 28299
        # GH 48633
        ts_strings = ["200622-12-31", "111111-24-11"]  # 包含格式错误的日期字符串列表
        with tm.assert_produces_warning(
            UserWarning, match="Could not infer format", raise_on_extra_warnings=False
        ):
            result = to_datetime(ts_strings, errors="coerce")  # 使用 to_datetime 转换日期，忽略错误
        # TODO: should Index get "s" by default here?
        exp = Index([NaT, NaT], dtype="M8[s]")  # 创建预期的日期索引对象
        tm.assert_index_equal(result, exp)  # 断言结果与预期相等

    def test_to_datetime_malformed_raise(self):
        # GH 48633
        ts_strings = ["200622-12-31", "111111-24-11"]  # 包含格式错误的日期字符串列表
        msg = (
            'Parsed string "200622-12-31" gives an invalid tzoffset, which must '
            r"be between -timedelta\(hours=24\) and timedelta\(hours=24\), "
            "at position 0"
        )  # 异常消息的详细描述
        with pytest.raises(
            ValueError,
            match=msg,
        ):
            with tm.assert_produces_warning(
                UserWarning, match="Could not infer format"
            ):
                to_datetime(
                    ts_strings,
                    errors="raise",
                )  # 使用 to_datetime 转换日期，当出现错误时抛出异常

    def test_iso_8601_strings_with_same_offset(self):
        # GH 17697, 11736
        ts_str = "2015-11-18 15:30:00+05:30"  # ISO 8601 格式的日期字符串
        result = to_datetime(ts_str)  # 转换日期字符串为 Timestamp 对象
        expected = Timestamp(ts_str)  # 创建预期的 Timestamp 对象
        assert result == expected  # 断言结果与预期相等

        expected = DatetimeIndex([Timestamp(ts_str)] * 2)  # 创建包含相同 Timestamp 的日期索引对象
        result = to_datetime([ts_str] * 2)  # 批量转换日期字符串为日期索引对象
        tm.assert_index_equal(result, expected)  # 断言结果与预期相等

        result = DatetimeIndex([ts_str] * 2)  # 直接创建日期索引对象
        tm.assert_index_equal(result, expected)  # 断言结果与预期相等
    def test_iso_8601_strings_with_different_offsets_removed(self):
        # 根据 GitHub 问题编号处理不同偏移量的 ISO 8601 时间字符串
        ts_strings = ["2015-11-18 15:30:00+05:30", "2015-11-18 16:30:00+06:30", NaT]
        msg = "Mixed timezones detected. Pass utc=True in to_datetime"
        # 使用 pytest 检查是否抛出 ValueError 异常，并匹配指定的错误消息
        with pytest.raises(ValueError, match=msg):
            to_datetime(ts_strings)

    def test_iso_8601_strings_with_different_offsets_utc(self):
        # 处理带有不同偏移量的 ISO 8601 时间字符串，并设置 utc=True
        ts_strings = ["2015-11-18 15:30:00+05:30", "2015-11-18 16:30:00+06:30", NaT]
        result = to_datetime(ts_strings, utc=True)
        # 创建预期的 DatetimeIndex 对象，以 UTC 时区，并以秒为单位
        expected = DatetimeIndex(
            [Timestamp(2015, 11, 18, 10), Timestamp(2015, 11, 18, 10), NaT], tz="UTC"
        ).as_unit("s")
        # 使用 assert_index_equal 检查结果与预期是否相等
        tm.assert_index_equal(result, expected)

    def test_mixed_offsets_with_native_datetime_utc_false_raises(self):
        # 处理包含本地日期时间与混合偏移量的情况，预期抛出 ValueError
        vals = [
            "nan",
            Timestamp("1990-01-01"),
            "2015-03-14T16:15:14.123-08:00",
            "2019-03-04T21:56:32.620-07:00",
            None,
            "today",
            "now",
        ]
        ser = Series(vals)
        # 使用 assert 确保所有序列中的元素与原始值相等
        assert all(ser[i] is vals[i] for i in range(len(vals)))  # GH#40111

        msg = "Mixed timezones detected. Pass utc=True in to_datetime"
        # 使用 pytest 检查是否抛出 ValueError 异常，并匹配指定的错误消息
        with pytest.raises(ValueError, match=msg):
            to_datetime(ser)

    def test_non_iso_strings_with_tz_offset(self):
        # 处理非 ISO 格式的带时区偏移的字符串转换
        result = to_datetime(["March 1, 2018 12:00:00+0400"] * 2)
        # 创建预期的 DatetimeIndex 对象，带有指定的时区偏移
        expected = DatetimeIndex(
            [datetime(2018, 3, 1, 12, tzinfo=timezone(timedelta(minutes=240)))] * 2
        ).as_unit("s")
        # 使用 assert_index_equal 检查结果与预期是否相等
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "ts, expected",
        [
            (Timestamp("2018-01-01"), Timestamp("2018-01-01", tz="UTC")),
            (
                Timestamp("2018-01-01", tz="US/Pacific"),
                Timestamp("2018-01-01 08:00", tz="UTC"),
            ),
        ],
    )
    def test_timestamp_utc_true(self, ts, expected):
        # 处理带有 UTC=True 参数的时间戳转换，根据 GitHub 问题编号 24415
        result = to_datetime(ts, utc=True)
        # 使用 assert 检查结果与预期是否相等
        assert result == expected

    @pytest.mark.parametrize("dt_str", ["00010101", "13000101", "30000101", "99990101"])
    def test_to_datetime_with_format_out_of_bounds(self, dt_str):
        # 处理超出范围的日期时间格式转换，根据 GitHub 问题编号 9107
        res = to_datetime(dt_str, format="%Y%m%d")
        dtobj = datetime.strptime(dt_str, "%Y%m%d")
        expected = Timestamp(dtobj).as_unit("s")
        # 使用 assert 检查结果与预期是否相等，并检查单位是否相等
        assert res == expected
        assert res.unit == expected.unit

    def test_to_datetime_utc(self):
        # 处理带有 UTC=True 参数的时间数组转换
        arr = np.array([parse("2012-06-13T01:39:00Z")], dtype=object)
        result = to_datetime(arr, utc=True)
        # 使用 assert 检查结果的时区是否为 UTC
        assert result.tz is timezone.utc
    # 在 pandas 测试框架中的固定偏移时区类导入
    from pandas.tests.indexes.datetimes.test_timezones import FixedOffset

    # 创建一个固定偏移时区对象，偏移量为 -420 分钟（即 -07:00）
    fixed_off = FixedOffset(-420, "-07:00")

    # 创建包含固定偏移时区日期时间的列表
    dates = [
        datetime(2000, 1, 1, tzinfo=fixed_off),
        datetime(2000, 1, 2, tzinfo=fixed_off),
        datetime(2000, 1, 3, tzinfo=fixed_off),
    ]

    # 调用 to_datetime 函数，将日期时间列表转换为 pandas 的 DateTimeIndex
    result = to_datetime(dates)

    # 断言结果的时区应与固定偏移时区对象相同
    assert result.tz == fixed_off

@pytest.mark.parametrize(
    "date",
    [
        # 测试参数化：不同混合时区日期时间字符串的情况
        ["2020-10-26 00:00:00+06:00", "2020-10-26 00:00:00+01:00"],
        # 测试参数化：包含不同时区的 Timestamp 对象的情况
        ["2020-10-26 00:00:00+06:00", Timestamp("2018-01-01", tz="US/Pacific")],
        # 测试参数化：包含不同时区的 datetime 对象的情况
        [
            "2020-10-26 00:00:00+06:00",
            datetime(2020, 1, 1, 18).astimezone(
                zoneinfo.ZoneInfo("Australia/Melbourne")
            ),
        ],
    ],
)
def test_to_datetime_mixed_offsets_with_utc_false_removed(self, date):
    # GH#50887, GH#57275
    # 函数功能注释：检测混合时区的日期时间输入，验证是否抛出 ValueError
    msg = "Mixed timezones detected. Pass utc=True in to_datetime"

    # 使用 pytest 的断言检查是否抛出指定消息的 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        to_datetime(date, utc=False)
class TestToDatetimeUnit:
    @pytest.mark.parametrize("unit", ["Y", "M"])
    @pytest.mark.parametrize("item", [150, float(150)])
    def test_to_datetime_month_or_year_unit_int(self, cache, unit, item, request):
        # GH#50870 Note we have separate tests that pd.Timestamp gets these right
        # 使用给定的单位和项目创建一个时间戳对象
        ts = Timestamp(item, unit=unit)
        # 创建预期的日期时间索引对象
        expected = DatetimeIndex([ts], dtype="M8[ns]")

        # 测试 to_datetime 函数，确保返回的结果与预期的日期时间索引相等
        result = to_datetime([item], unit=unit, cache=cache)
        tm.assert_index_equal(result, expected)

        # 测试处理对象数组的情况
        result = to_datetime(np.array([item], dtype=object), unit=unit, cache=cache)
        tm.assert_index_equal(result, expected)

        # 测试处理普通数组的情况
        result = to_datetime(np.array([item]), unit=unit, cache=cache)
        tm.assert_index_equal(result, expected)

        # 测试包含 NaN 的情况
        result = to_datetime(np.array([item, np.nan]), unit=unit, cache=cache)
        # 确保结果的第二个值是 NaN
        assert result.isna()[1]
        # 确保只比较结果的第一个值与预期的日期时间索引
        tm.assert_index_equal(result[:1], expected)

    @pytest.mark.parametrize("unit", ["Y", "M"])
    def test_to_datetime_month_or_year_unit_non_round_float(self, cache, unit):
        # GH#50301
        # 测试对非整数浮点数使用 to_datetime 函数抛出 ValueError 异常的情况
        msg = f"Conversion of non-round float with unit={unit} is ambiguous"
        with pytest.raises(ValueError, match=msg):
            to_datetime([1.5], unit=unit, errors="raise")
        with pytest.raises(ValueError, match=msg):
            to_datetime(np.array([1.5]), unit=unit, errors="raise")

        # 测试给定日期字符串不是日期时间格式的情况
        msg = r"Given date string \"1.5\" not likely a datetime, at position 0"
        with pytest.raises(ValueError, match=msg):
            to_datetime(["1.5"], unit=unit, errors="raise")

        # 测试使用 errors="coerce" 参数处理的情况
        res = to_datetime([1.5], unit=unit, errors="coerce")
        expected = Index([NaT], dtype="M8[ns]")
        tm.assert_index_equal(res, expected)

        # 在 3.0 版本中，字符串 "1.5" 被解析为 NaT，因为它不是有效的日期时间格式
        res = to_datetime(["1.5"], unit=unit, errors="coerce")
        expected = to_datetime([NaT]).as_unit("ns")
        tm.assert_index_equal(res, expected)

        # 测试处理整数浮点数的情况
        res = to_datetime([1.0], unit=unit)
        expected = to_datetime([1], unit=unit)
        tm.assert_index_equal(res, expected)

    def test_unit(self, cache):
        # GH 11758
        # 测试在指定 format 和 unit 同时存在时抛出 ValueError 异常的情况
        msg = "cannot specify both format and unit"
        with pytest.raises(ValueError, match=msg):
            to_datetime([1], unit="D", format="%Y%m%d", cache=cache)
    def test_unit_array_mixed_nans(self, cache):
        # 创建一个包含多种数据类型和NaN的列表作为输入值
        values = [11111111111111111, 1, 1.0, iNaT, NaT, np.nan, "NaT", ""]

        # 调用to_datetime函数进行日期时间转换，指定单位为'D'，错误处理为'coerce'
        result = to_datetime(values, unit="D", errors="coerce", cache=cache)
        # 期望的结果是一个DatetimeIndex对象，指定了预期的日期时间格式
        expected = DatetimeIndex(
            ["NaT", "1970-01-02", "1970-01-02", "NaT", "NaT", "NaT", "NaT", "NaT"],
            dtype="M8[ns]",
        )
        # 使用assert_index_equal函数验证转换结果是否与期望相符
        tm.assert_index_equal(result, expected)

        # 设置错误消息，用于测试异常情况
        msg = "cannot convert input 11111111111111111 with the unit 'D'"
        # 使用pytest的raises函数检查是否会抛出OutOfBoundsDatetime异常，并验证异常消息
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(values, unit="D", errors="raise", cache=cache)

    def test_unit_array_mixed_nans_large_int(self, cache):
        # 创建一个包含大整数、NaN和其他值的列表作为输入值
        values = [1420043460000000000000000, iNaT, NaT, np.nan, "NaT"]

        # 调用to_datetime函数进行日期时间转换，指定单位为's'，错误处理为'coerce'
        result = to_datetime(values, errors="coerce", unit="s", cache=cache)
        # 期望的结果是一个DatetimeIndex对象，指定了预期的日期时间格式
        expected = DatetimeIndex(["NaT", "NaT", "NaT", "NaT", "NaT"], dtype="M8[ns]")
        # 使用assert_index_equal函数验证转换结果是否与期望相符
        tm.assert_index_equal(result, expected)

        # 设置错误消息，用于测试异常情况
        msg = "cannot convert input 1420043460000000000000000 with the unit 's'"
        # 使用pytest的raises函数检查是否会抛出OutOfBoundsDatetime异常，并验证异常消息
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(values, errors="raise", unit="s", cache=cache)

    def test_to_datetime_invalid_str_not_out_of_bounds_valuerror(self, cache):
        # 当输入值为字符串时，期望抛出ValueError异常而不是OutOfBoundsDatetime异常
        msg = "Unknown datetime string format, unable to parse: foo, at position 0"
        # 使用pytest的raises函数检查是否会抛出ValueError异常，并验证异常消息
        with pytest.raises(ValueError, match=msg):
            to_datetime("foo", errors="raise", unit="s", cache=cache)

    @pytest.mark.parametrize("error", ["raise", "coerce"])
    def test_unit_consistency(self, cache, error):
        # 测试日期时间转换的一致性
        expected = Timestamp("1970-05-09 14:25:11")
        # 调用to_datetime函数进行日期时间转换，指定单位为's'，错误处理方式由参数error决定
        result = to_datetime(11111111, unit="s", errors=error, cache=cache)
        # 使用assert语句验证转换结果是否与期望相符，并检查结果类型是否为Timestamp
        assert result == expected
        assert isinstance(result, Timestamp)

    @pytest.mark.parametrize("errors", ["raise", "coerce"])
    @pytest.mark.parametrize("dtype", ["float64", "int64"])
    def test_unit_with_numeric(self, cache, errors, dtype):
        # 测试从浮点数/整数到日期时间的转换
        # 创建一个包含浮点数/整数的numpy数组作为输入值
        arr = np.array([1.434692e18, 1.432766e18]).astype(dtype)
        # 调用to_datetime函数进行日期时间转换，指定错误处理方式和数据类型
        result = to_datetime(arr, errors=errors, cache=cache)
        # 期望的结果是一个DatetimeIndex对象，指定了预期的日期时间格式
        expected = DatetimeIndex(
            ["2015-06-19 05:33:20", "2015-05-27 22:33:20"], dtype="M8[ns]"
        )
        # 使用assert_index_equal函数验证转换结果是否与期望相符
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "exp, arr, warning",
        [
            [
                ["NaT", "2015-06-19 05:33:20", "2015-05-27 22:33:20"],
                ["foo", 1.434692e18, 1.432766e18],
                UserWarning,
            ],
            [
                ["2015-06-19 05:33:20", "2015-05-27 22:33:20", "NaT", "NaT"],
                [1.434692e18, 1.432766e18, "foo", "NaT"],
                None,
            ],
        ],
    )
    def test_unit_with_numeric_coerce(self, cache, exp, arr, warning):
        # 对于包含数值和字符串的输入数组 arr，确保进行类型强制转换
        # 生成预期的 DatetimeIndex 对象，数据类型为 "M8[ns]"
        expected = DatetimeIndex(exp, dtype="M8[ns]")
        # 使用警告断言确保警告被触发，匹配 "Could not infer format"
        with tm.assert_produces_warning(warning, match="Could not infer format"):
            # 调用 to_datetime 函数，将 arr 转换为日期时间对象，错误处理方式为 "coerce"
            # 使用缓存 cache 加速转换过程
            result = to_datetime(arr, errors="coerce", cache=cache)
        # 断言结果与预期 DatetimeIndex 对象相等
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "arr",
        [
            [Timestamp("20130101"), 1.434692e18, 1.432766e18],
            [1.434692e18, 1.432766e18, Timestamp("20130101")],
        ],
    )
    def test_unit_mixed(self, cache, arr):
        # GH#50453 在 2.0 版本之前，处理混合数值和日期时间时使用 errors="coerce"
        # 数值条目将被转换为 NaT，但从未清楚原因是什么。
        # 混合整数/日期时间输入数组 arr
        expected = Index([Timestamp(x) for x in arr], dtype="M8[ns]")
        # 将 arr 转换为日期时间对象，错误处理方式为 "coerce"，使用缓存 cache 加速转换过程
        result = to_datetime(arr, errors="coerce", cache=cache)
        # 断言结果与预期 Index 对象相等
        tm.assert_index_equal(result, expected)

        # GH#49037 在 2.0 版本之前，此处会引发异常，但对于 Series 一直可以正常工作，
        # 从未清楚为什么会禁止这种情况。
        # 将 arr 转换为日期时间对象，错误处理方式为 "raise"，使用缓存 cache 加速转换过程
        result = to_datetime(arr, errors="raise", cache=cache)
        # 断言结果与预期 Index 对象相等
        tm.assert_index_equal(result, expected)

        # 直接将 arr 转换为 DatetimeIndex 对象
        result = DatetimeIndex(arr)
        # 断言结果与预期 Index 对象相等
        tm.assert_index_equal(result, expected)

    def test_unit_rounding(self, cache):
        # GH 14156 & GH 20445: 参数会导致浮点数误差，但不会造成过早舍入。
        # 给定的值，以秒为单位转换为日期时间对象，使用缓存 cache 加速转换过程
        value = 1434743731.8770001
        result = to_datetime(value, unit="s", cache=cache)
        # 预期的 Timestamp 对象
        expected = Timestamp("2015-06-19 19:55:31.877000093")
        # 断言结果与预期 Timestamp 对象相等
        assert result == expected

        # 使用 Timestamp 类直接创建对象，以秒为单位转换为日期时间对象
        alt = Timestamp(value, unit="s")
        # 断言直接创建的对象与使用 to_datetime 转换的结果相等
        assert alt == result

    @pytest.mark.parametrize("dtype", [int, float])
    def test_to_datetime_unit(self, dtype):
        # 给定一个 epoch 时间
        epoch = 1370745748
        # 创建一个 Series 对象，包含从 epoch 开始的一系列时间戳，数据类型为 dtype
        ser = Series([epoch + t for t in range(20)]).astype(dtype)
        # 将 Series 对象转换为日期时间对象，单位为秒
        result = to_datetime(ser, unit="s")
        # 预期的 Series 对象，包含从 epoch 开始的一系列日期时间对象
        expected = Series(
            [
                Timestamp("2013-06-09 02:42:28") + timedelta(seconds=t)
                for t in range(20)
            ],
            dtype="M8[ns]",
        )
        # 断言结果与预期 Series 对象相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("null", [iNaT, np.nan])
    def test_to_datetime_unit_with_nulls(self, null):
        # 给定一个 epoch 时间
        epoch = 1370745748
        # 创建一个 Series 对象，包含从 epoch 开始的一系列时间戳和一个 null 值
        ser = Series([epoch + t for t in range(20)] + [null])
        # 将 Series 对象转换为日期时间对象，单位为秒
        result = to_datetime(ser, unit="s")
        # 预期的 Series 对象，包含从 epoch 开始的一系列日期时间对象和一个 NaT
        expected = Series(
            [Timestamp("2013-06-09 02:42:28") + timedelta(seconds=t) for t in range(20)]
            + [NaT],
            dtype="M8[ns]",
        )
        # 断言结果与预期 Series 对象相等
        tm.assert_series_equal(result, expected)
    # 测试函数，用于验证 to_datetime 函数在处理小数秒时的行为
    def test_to_datetime_unit_fractional_seconds(self):
        # 设定一个 epoch 时间作为起点
        epoch = 1370745748
        # 创建一个包含 epoch 时间加上一系列小数秒和 NaT 值的 Series 对象，并转换为浮点型
        ser = Series([epoch + t for t in np.arange(0, 2, 0.25)] + [iNaT]).astype(float)
        # 调用 to_datetime 函数，将 Series 转换为时间戳，单位为秒
        result = to_datetime(ser, unit="s")
        # 创建一个期望的时间戳 Series，包含一系列与 epoch 时间加上小数秒对应的时间戳和 NaT
        expected = Series(
            [
                Timestamp("2013-06-09 02:42:28") + timedelta(seconds=t)
                for t in np.arange(0, 2, 0.25)
            ]
            + [NaT],
            dtype="M8[ns]",
        )
        # 对结果进行毫秒级别的四舍五入处理
        result = result.round("ms")
        # 使用测试框架中的 assert 函数比较 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

    # 测试函数，验证 to_datetime 函数在处理特殊值（如 NaT 和 NaN）时的行为
    def test_to_datetime_unit_na_values(self):
        # 调用 to_datetime 函数，将包含数值、字符串 "NaT"、NaT 和 NaN 的列表转换为日期时间索引，单位为天
        result = to_datetime([1, 2, "NaT", NaT, np.nan], unit="D")
        # 创建一个期望的日期时间索引，包含数值 1 和 2 的日期时间索引，以及三个 "NaT" 字符串
        expected = DatetimeIndex(
            [Timestamp("1970-01-02"), Timestamp("1970-01-03")] + ["NaT"] * 3,
            dtype="M8[ns]",
        )
        # 使用测试框架中的 assert 函数比较 result 和 expected 是否相等
        tm.assert_index_equal(result, expected)

    # 使用 pytest.mark.parametrize 装饰器标记的测试函数，验证 to_datetime 函数在处理无效输入时的行为
    @pytest.mark.parametrize("bad_val", ["foo", 111111111])
    def test_to_datetime_unit_invalid(self, bad_val):
        if bad_val == "foo":
            # 如果输入值为 "foo"，设置错误消息为相应的格式
            msg = (
                "Unknown datetime string format, unable to parse: "
                f"{bad_val}, at position 2"
            )
        else:
            # 如果输入值为其他无效数字，设置错误消息为相应的格式
            msg = "cannot convert input 111111111 with the unit 'D', at position 2"
        # 使用 pytest.raises 检查是否抛出 ValueError 异常，并匹配特定的错误消息
        with pytest.raises(ValueError, match=msg):
            # 调用 to_datetime 函数，传入包含数值 1、2 和 bad_val 的列表，并设定单位为天
            to_datetime([1, 2, bad_val], unit="D")

    # 使用 pytest.mark.parametrize 装饰器标记的测试函数，验证 to_datetime 函数在处理 coerce 错误模式时的行为
    @pytest.mark.parametrize("bad_val", ["foo", 111111111])
    def test_to_timestamp_unit_coerce(self, bad_val):
        # 设定预期的日期时间索引，包含数值 1 和 2 的日期时间索引，以及一个 "NaT" 字符串
        expected = DatetimeIndex(
            [Timestamp("1970-01-02"), Timestamp("1970-01-03")] + ["NaT"] * 1,
            dtype="M8[ns]",
        )
        # 调用 to_datetime 函数，传入包含数值 1、2 和 bad_val 的列表，并设定单位为天，错误处理模式为 'coerce'
        result = to_datetime([1, 2, bad_val], unit="D", errors="coerce")
        # 使用测试框架中的 assert 函数比较 result 和 expected 是否相等
        tm.assert_index_equal(result, expected)
    def test_float_to_datetime_raise_near_bounds(self):
        # 测试用例标识：GH50183
        msg = "cannot convert input with unit 'D'"
        # 计算一天的纳秒数
        oneday_in_ns = 1e9 * 60 * 60 * 24
        # 计算 tsmax_in_days，即 2**63 纳秒转换为天数
        tsmax_in_days = 2**63 / oneday_in_ns  # 2**63 ns, in days
        # 构造一个测试成功的数据序列，分别在边界内部
        should_succeed = Series(
            [0, tsmax_in_days - 0.005, -tsmax_in_days + 0.005], dtype=float
        )
        # 期望的结果，将成功的数据序列转换为纳秒后再转换为 np.int64 类型
        expected = (should_succeed * oneday_in_ns).astype(np.int64)
        # 对于两种错误处理模式分别进行测试
        for error_mode in ["raise", "coerce"]:
            # 调用 to_datetime 函数进行转换
            result1 = to_datetime(should_succeed, unit="D", errors=error_mode)
            # 强制转换为 np.float64 类型，以便进行相对容差检查
            # （整数类型的数据不会进行精确检查）
            tm.assert_almost_equal(
                result1.astype(np.int64).astype(np.float64),
                expected.astype(np.float64),
                rtol=1e-10,
            )
        # 构造两个测试失败的数据序列，超出边界
        should_fail1 = Series([0, tsmax_in_days + 0.005], dtype=float)
        should_fail2 = Series([0, -tsmax_in_days - 0.005], dtype=float)
        # 使用 pytest 检查是否引发 OutOfBoundsDatetime 异常，并匹配特定消息
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(should_fail1, unit="D", errors="raise")
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(should_fail2, unit="D", errors="raise")
# 定义一个测试类 TestToDatetimeDataFrame
class TestToDatetimeDataFrame:
    
    # 使用 pytest 的 fixture 装饰器，为测试方法提供数据框架 df
    @pytest.fixture
    def df(self):
        # 返回一个 DataFrame 对象，包含多列年、月、日等时间相关数据
        return DataFrame(
            {
                "year": [2015, 2016],
                "month": [2, 3],
                "day": [4, 5],
                "hour": [6, 7],
                "minute": [58, 59],
                "second": [10, 11],
                "ms": [1, 1],
                "us": [2, 2],
                "ns": [3, 3],
            }
        )

    # 测试方法：测试 to_datetime 函数对数据框架 df 的处理结果
    def test_dataframe(self, df, cache):
        # 调用 to_datetime 函数，传入部分时间列，并使用缓存参数 cache
        result = to_datetime(
            {"year": df["year"], "month": df["month"], "day": df["day"]}, cache=cache
        )
        # 预期结果为一个 Series 对象，包含特定的时间戳格式
        expected = Series(
            [Timestamp("20150204 00:00:00"), Timestamp("20160305 00:0:00")]
        )
        # 使用测试框架的 assert_series_equal 函数比较结果和预期结果
        tm.assert_series_equal(result, expected)

        # 将部分时间列转换为字典形式，再调用 to_datetime 函数
        result = to_datetime(df[["year", "month", "day"]].to_dict(), cache=cache)
        # 再次比较结果和预期结果的 Series 对象
        tm.assert_series_equal(result, expected)

    # 测试方法：测试 to_datetime 函数处理具有可构建属性的字典形式的数据框架
    def test_dataframe_dict_with_constructable(self, df, cache):
        # 从 df 中提取部分时间列并转换为字典
        df2 = df[["year", "month", "day"]].to_dict()
        # 修改字典中的一个月份值
        df2["month"] = 2
        # 调用 to_datetime 函数处理修改后的 df2 字典
        result = to_datetime(df2, cache=cache)
        # 预期结果为一个 Series 对象，包含特定的时间戳格式
        expected2 = Series(
            [Timestamp("20150204 00:00:00"), Timestamp("20160205 00:0:00")]
        )
        # 比较处理结果和预期结果的 Series 对象
        tm.assert_series_equal(result, expected2)

    # 使用 pytest 的参数化装饰器，为 unit 参数提供两组字典
    @pytest.mark.parametrize(
        "unit",
        [
            {
                "year": "years",
                "month": "months",
                "day": "days",
                "hour": "hours",
                "minute": "minutes",
                "second": "seconds",
            },
            {
                "year": "year",
                "month": "month",
                "day": "day",
                "hour": "hour",
                "minute": "minute",
                "second": "second",
            },
        ],
    )
    # 测试方法：测试 to_datetime 函数使用字段别名处理列子集
    def test_dataframe_field_aliases_column_subset(self, df, cache, unit):
        # 根据 unit 字典提供的字段映射，重命名 df 的列，并调用 to_datetime 函数
        result = to_datetime(df[list(unit.keys())].rename(columns=unit), cache=cache)
        # 预期结果为一个 Series 对象，包含特定的时间戳格式和数据类型
        expected = Series(
            [Timestamp("20150204 06:58:10"), Timestamp("20160305 07:59:11")],
            dtype="M8[ns]",
        )
        # 比较处理结果和预期结果的 Series 对象
        tm.assert_series_equal(result, expected)

    # 测试方法：测试 to_datetime 函数使用字段别名处理整个数据框架
    def test_dataframe_field_aliases(self, df, cache):
        # 定义一个包含时间字段别名的字典 d
        d = {
            "year": "year",
            "month": "month",
            "day": "day",
            "hour": "hour",
            "minute": "minute",
            "second": "second",
            "ms": "ms",
            "us": "us",
            "ns": "ns",
        }
        # 根据字段别名字典 d，重命名 df 的列，并调用 to_datetime 函数
        result = to_datetime(df.rename(columns=d), cache=cache)
        # 预期结果为一个 Series 对象，包含特定的时间戳格式
        expected = Series(
            [
                Timestamp("20150204 06:58:10.001002003"),
                Timestamp("20160305 07:59:11.001002003"),
            ]
        )
        # 比较处理结果和预期结果的 Series 对象
        tm.assert_series_equal(result, expected)
    # 测试将数据框中的字符串列转换为日期时间类型
    def test_dataframe_str_dtype(self, df, cache):
        # 将数据框的所有列转换为字符串，再转换为日期时间类型，使用缓存加速转换过程
        result = to_datetime(df.astype(str), cache=cache)
        # 预期的日期时间序列
        expected = Series(
            [
                Timestamp("20150204 06:58:10.001002003"),
                Timestamp("20160305 07:59:11.001002003"),
            ]
        )
        # 断言转换结果与预期结果相等
        tm.assert_series_equal(result, expected)

    # 测试在转换日期时间时使用 coerce 参数处理异常数据
    def test_dataframe_coerce(self, cache):
        # 创建包含异常数据的数据框
        df2 = DataFrame({"year": [2015, 2016], "month": [2, 20], "day": [4, 5]})

        # 预期的异常消息正则表达式
        msg = (
            r'^cannot assemble the datetimes: time data ".+" doesn\'t '
            r'match format "%Y%m%d", at position 1\.'
        )
        # 使用 pytest 检查是否抛出预期的 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            to_datetime(df2, cache=cache)

        # 使用 coerce 参数处理异常数据
        result = to_datetime(df2, errors="coerce", cache=cache)
        # 预期的日期时间序列，处理后的异常数据返回 NaT (Not a Time)
        expected = Series([Timestamp("20150204 00:00:00"), NaT])
        # 断言转换结果与预期结果相等
        tm.assert_series_equal(result, expected)

    # 测试在数据框中包含额外列时是否抛出异常
    def test_dataframe_extra_keys_raises(self, df, cache):
        # 预期的异常消息正则表达式，指示有额外的列传递给日期时间转换函数
        msg = r"extra keys have been passed to the datetime assemblage: \[foo\]"
        # 复制数据框并添加额外的列
        df2 = df.copy()
        df2["foo"] = 1
        # 使用 pytest 检查是否抛出预期的 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            to_datetime(df2, cache=cache)

    # 使用参数化测试来验证缺少必要列时是否会抛出异常
    @pytest.mark.parametrize(
        "cols",
        [
            ["year"],
            ["year", "month"],
            ["year", "month", "second"],
            ["month", "day"],
            ["year", "day", "second"],
        ],
    )
    def test_dataframe_missing_keys_raises(self, df, cache, cols):
        # 预期的异常消息正则表达式，指示缺少必要的列
        msg = (
            r"to assemble mappings requires at least that \[year, month, "
            r"day\] be specified: \[.+\] is missing"
        )
        # 使用 pytest 检查是否抛出预期的 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            to_datetime(df[cols], cache=cache)

    # 测试数据框中是否抛出重复列名异常
    def test_dataframe_duplicate_columns_raises(self, cache):
        # 预期的异常消息，指示数据框中有重复的列名
        msg = "cannot assemble with duplicate keys"
        # 创建包含重复列名的数据框
        df2 = DataFrame({"year": [2015, 2016], "month": [2, 20], "day": [4, 5]})
        df2.columns = ["year", "year", "day"]
        # 使用 pytest 检查是否抛出预期的 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            to_datetime(df2, cache=cache)

        # 创建包含重复列名的数据框
        df2 = DataFrame(
            {"year": [2015, 2016], "month": [2, 20], "day": [4, 5], "hour": [4, 5]}
        )
        df2.columns = ["year", "month", "day", "day"]
        # 使用 pytest 检查是否抛出预期的 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            to_datetime(df2, cache=cache)

    # 测试处理 int16 类型数据框的日期时间转换
    def test_dataframe_int16(self, cache):
        # 创建包含 int16 类型数据的数据框
        df = DataFrame({"year": [2015, 2016], "month": [2, 3], "day": [4, 5]})

        # 将 int16 类型数据转换为日期时间类型
        result = to_datetime(df.astype("int16"), cache=cache)
        # 预期的日期时间序列
        expected = Series(
            [Timestamp("20150204 00:00:00"), Timestamp("20160305 00:00:00")]
        )
        # 断言转换结果与预期结果相等
        tm.assert_series_equal(result, expected)
    # 测试用例：测试包含混合数据类型的 DataFrame 转换为 datetime
    def test_dataframe_mixed(self, cache):
        # 创建包含混合数据类型的 DataFrame
        df = DataFrame({"year": [2015, 2016], "month": [2, 3], "day": [4, 5]})
        # 将 'month' 和 'day' 列转换为 int8 类型
        df["month"] = df["month"].astype("int8")
        df["day"] = df["day"].astype("int8")
        # 调用 to_datetime 函数进行转换
        result = to_datetime(df, cache=cache)
        # 创建预期结果 Series 对象，包含预期的时间戳
        expected = Series(
            [Timestamp("20150204 00:00:00"), Timestamp("20160305 00:00:00")]
        )
        # 使用测试工具函数检查结果是否符合预期
        tm.assert_series_equal(result, expected)

    # 测试用例：测试包含浮点数的 DataFrame 转换为 datetime
    def test_dataframe_float(self, cache):
        # 创建包含浮点数的 DataFrame
        df = DataFrame({"year": [2000, 2001], "month": [1.5, 1], "day": [1, 1]})
        # 准备用于匹配错误消息的正则表达式
        msg = (
            r"^cannot assemble the datetimes: unconverted data remains when parsing "
            r'with format ".*": "1", at position 0.'
        )
        # 使用 pytest 检查调用 to_datetime 函数时是否引发 ValueError 异常，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            to_datetime(df, cache=cache)

    # 测试用例：测试设置 utc=True 时的 DataFrame 转换为 datetime
    def test_dataframe_utc_true(self):
        # 创建包含日期数据的 DataFrame
        df = DataFrame({"year": [2015, 2016], "month": [2, 3], "day": [4, 5]})
        # 调用 to_datetime 函数进行转换，并设置 utc=True
        result = to_datetime(df, utc=True)
        # 创建预期结果 Series 对象，使用 numpy 数组指定日期时间的 dtype，并设置时区为 UTC
        expected = Series(
            np.array(["2015-02-04", "2016-03-05"], dtype="datetime64[s]")
        ).dt.tz_localize("UTC")
        # 使用测试工具函数检查结果是否符合预期
        tm.assert_series_equal(result, expected)
class TestToDatetimeMisc:
    # 定义测试方法，验证边界情况下的日期时间转换
    def test_to_datetime_barely_out_of_bounds(self):
        # GH#19529
        # GH#19382 接近边界的情况下，去除纳秒会导致在范围内的日期时间
        arr = np.array(["2262-04-11 23:47:16.854775808"], dtype=object)

        # 定义期望的错误信息模式，验证是否会引发 OutOfBoundsDatetime 异常
        msg = "^Out of bounds nanosecond timestamp: .*, at position 0"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(arr)

    @pytest.mark.parametrize(
        "arg, exp_str",
        [
            ["2012-01-01 00:00:00", "2012-01-01 00:00:00"],
            ["20121001", "2012-10-01"],  # 不符合 ISO 8601 标准的日期格式
        ],
    )
    # 定义测试方法，验证 ISO 8601 格式的日期时间转换
    def test_to_datetime_iso8601(self, cache, arg, exp_str):
        # 调用 to_datetime 函数进行转换
        result = to_datetime([arg], cache=cache)
        # 设置期望的 Timestamp 对象
        exp = Timestamp(exp_str)
        # 断言转换结果与期望结果是否一致
        assert result[0] == exp

    @pytest.mark.parametrize(
        "input, format",
        [
            ("2012", "%Y-%m"),
            ("2012-01", "%Y-%m-%d"),
            ("2012-01-01", "%Y-%m-%d %H"),
            ("2012-01-01 10", "%Y-%m-%d %H:%M"),
            ("2012-01-01 10:00", "%Y-%m-%d %H:%M:%S"),
            ("2012-01-01 10:00:00", "%Y-%m-%d %H:%M:%S.%f"),
            ("2012-01-01 10:00:00.123", "%Y-%m-%d %H:%M:%S.%f%z"),
            (0, "%Y-%m-%d"),
        ],
    )
    @pytest.mark.parametrize("exact", [True, False])
    # 定义测试方法，验证 ISO 8601 格式的日期时间转换失败情况
    def test_to_datetime_iso8601_fails(self, input, format, exact):
        # https://github.com/pandas-dev/pandas/issues/12649
        # 当 `format` 长于日期字符串时，无论 `exact` 的设置如何都会失败
        with pytest.raises(
            ValueError,
            match=(
                rf"time data \"{input}\" doesn't match format "
                rf"\"{format}\", at position 0"
            ),
        ):
            to_datetime(input, format=format, exact=exact)

    @pytest.mark.parametrize(
        "input, format",
        [
            ("2012-01-01", "%Y-%m"),
            ("2012-01-01 10", "%Y-%m-%d"),
            ("2012-01-01 10:00", "%Y-%m-%d %H"),
            ("2012-01-01 10:00:00", "%Y-%m-%d %H:%M"),
            (0, "%Y-%m-%d"),
        ],
    )
    # 定义测试方法，验证 ISO 8601 格式的日期时间转换精确匹配失败情况
    def test_to_datetime_iso8601_exact_fails(self, input, format):
        # https://github.com/pandas-dev/pandas/issues/12649
        # 当 `format` 短于日期字符串时，只有在 `exact=True` 时才会失败
        msg = "|".join(
            [
                '^unconverted data remains when parsing with format ".*": ".*"'
                f", at position 0. {PARSING_ERR_MSG}$",
                f'^time data ".*" doesn\'t match format ".*", at position 0. '
                f"{PARSING_ERR_MSG}$",
            ]
        )
        with pytest.raises(
            ValueError,
            match=(msg),
        ):
            to_datetime(input, format=format)
    @pytest.mark.parametrize(
        "input, format",
        [  # 参数化测试数据：输入日期字符串和对应的格式化字符串
            ("2012-01-01", "%Y-%m"),  # 测试用例1：年月格式
            ("2012-01-01 00", "%Y-%m-%d"),  # 测试用例2：年月日格式
            ("2012-01-01 00:00", "%Y-%m-%d %H"),  # 测试用例3：年月日小时格式
            ("2012-01-01 00:00:00", "%Y-%m-%d %H:%M"),  # 测试用例4：年月日小时分钟格式
        ],
    )
    def test_to_datetime_iso8601_non_exact(self, input, format):
        # 对非精确的 ISO 8601 格式日期进行测试
        expected = Timestamp(2012, 1, 1)  # 预期的时间戳对象
        result = to_datetime(input, format=format, exact=False)  # 调用函数，转换日期字符串为时间戳
        assert result == expected  # 断言结果与预期相同

    @pytest.mark.parametrize(
        "input, format",
        [  # 参数化测试数据：输入日期字符串和对应的格式化字符串
            ("2020-01", "%Y/%m"),  # 测试用例1：年月格式
            ("2020-01-01", "%Y/%m/%d"),  # 测试用例2：年月日格式
            ("2020-01-01 00", "%Y/%m/%dT%H"),  # 测试用例3：年月日T小时格式
            ("2020-01-01T00", "%Y/%m/%d %H"),  # 测试用例4：年月日小时格式
            ("2020-01-01 00:00", "%Y/%m/%dT%H:%M"),  # 测试用例5：年月日T小时分钟格式
            ("2020-01-01T00:00", "%Y/%m/%d %H:%M"),  # 测试用例6：年月日小时分钟格式
            ("2020-01-01 00:00:00", "%Y/%m/%dT%H:%M:%S"),  # 测试用例7：年月日T小时分钟秒格式
            ("2020-01-01T00:00:00", "%Y/%m/%d %H:%M:%S"),  # 测试用例8：年月日小时分钟秒格式
        ],
    )
    def test_to_datetime_iso8601_separator(self, input, format):
        # 测试 ISO 8601 格式日期带分隔符的情况
        with pytest.raises(
            ValueError,
            match=(
                rf"time data \"{input}\" doesn\'t match format "  # 引发值错误，匹配错误的时间数据和格式
                rf"\"{format}\", at position 0"
            ),
        ):
            to_datetime(input, format=format)  # 调用函数，尝试转换日期字符串为时间戳

    @pytest.mark.parametrize(
        "input, format",
        [  # 参数化测试数据：输入日期字符串和对应的格式化字符串
            ("2020-01", "%Y-%m"),  # 测试用例1：年月格式
            ("2020-01-01", "%Y-%m-%d"),  # 测试用例2：年月日格式
            ("2020-01-01 00", "%Y-%m-%d %H"),  # 测试用例3：年月日小时格式
            ("2020-01-01T00", "%Y-%m-%dT%H"),  # 测试用例4：年月日T小时格式
            ("2020-01-01 00:00", "%Y-%m-%d %H:%M"),  # 测试用例5：年月日小时分钟格式
            ("2020-01-01T00:00", "%Y-%m-%dT%H:%M"),  # 测试用例6：年月日T小时分钟格式
            ("2020-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"),  # 测试用例7：年月日小时分钟秒格式
            ("2020-01-01T00:00:00", "%Y-%m-%dT%H:%M:%S"),  # 测试用例8：年月日T小时分钟秒格式
            ("2020-01-01T00:00:00.000", "%Y-%m-%dT%H:%M:%S.%f"),  # 测试用例9：年月日T小时分钟秒毫秒格式
            ("2020-01-01T00:00:00.000000", "%Y-%m-%dT%H:%M:%S.%f"),  # 测试用例10：年月日T小时分钟秒微秒格式
            ("2020-01-01T00:00:00.000000000", "%Y-%m-%dT%H:%M:%S.%f"),  # 测试用例11：年月日T小时分钟秒纳秒格式
        ],
    )
    def test_to_datetime_iso8601_valid(self, input, format):
        # 测试 ISO 8601 格式日期的有效性
        expected = Timestamp(2020, 1, 1)  # 预期的时间戳对象
        result = to_datetime(input, format=format)  # 调用函数，转换日期字符串为时间戳
        assert result == expected  # 断言结果与预期相同
    def test_to_datetime_iso8601_non_padded(self, input, format):
        # 测试非填充 ISO8601 格式日期时间转换
        expected = Timestamp(2020, 1, 1)
        result = to_datetime(input, format=format)
        assert result == expected

    @pytest.mark.parametrize(
        "input, format",
        [
            ("2020-01-01T00:00:00.000000000+00:00", "%Y-%m-%dT%H:%M:%S.%f%z"),
            ("2020-01-01T00:00:00+00:00", "%Y-%m-%dT%H:%M:%S%z"),
            ("2020-01-01T00:00:00Z", "%Y-%m-%dT%H:%M:%S%z"),
        ],
    )
    def test_to_datetime_iso8601_with_timezone_valid(self, input, format):
        # 测试带时区信息的 ISO8601 格式日期时间转换
        expected = Timestamp(2020, 1, 1, tzinfo=timezone.utc)
        result = to_datetime(input, format=format)
        assert result == expected

    def test_to_datetime_default(self, cache):
        # 测试默认情况下的日期时间转换
        rs = to_datetime("2001", cache=cache)
        xp = datetime(2001, 1, 1)
        assert rs == xp

    @pytest.mark.xfail(reason="fails to enforce dayfirst=True, which would raise")
    def test_to_datetime_respects_dayfirst(self, cache):
        # 测试日期时间转换是否尊重 dayfirst=True 的情况
        # dayfirst 参数目前存在问题
        msg = "Invalid date specified"
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(UserWarning, match="Provide format"):
                to_datetime("01-13-2012", dayfirst=True, cache=cache)

    def test_to_datetime_on_datetime64_series(self, cache):
        # 测试在 datetime64 类型的 Series 上的日期时间转换
        # #2699
        ser = Series(date_range("1/1/2000", periods=10))

        result = to_datetime(ser, cache=cache)
        assert result[0] == ser[0]

    def test_to_datetime_with_space_in_series(self, cache):
        # 测试 Series 中包含空格的日期时间转换
        # GH 6428
        ser = Series(["10/18/2006", "10/18/2008", " "])
        msg = (
            r'^time data " " doesn\'t match format "%m/%d/%Y", '
            rf"at position 2. {PARSING_ERR_MSG}$"
        )
        with pytest.raises(ValueError, match=msg):
            to_datetime(ser, errors="raise", cache=cache)
        result_coerce = to_datetime(ser, errors="coerce", cache=cache)
        expected_coerce = Series(
            [datetime(2006, 10, 18), datetime(2008, 10, 18), NaT]
        ).dt.as_unit("s")
        tm.assert_series_equal(result_coerce, expected_coerce)

    @td.skip_if_not_us_locale
    def test_to_datetime_with_apply(self, cache):
        # 测试在 apply 函数中使用日期时间转换
        # 只有在 US/None 本地化环境下进行测试
        # GH 5195
        # 使用格式化字符串和 coerce 参数，单个项转换为日期时间
        td = Series(["May 04", "Jun 02", "Dec 11"], index=[1, 2, 3])
        expected = to_datetime(td, format="%b %y", cache=cache)
        result = td.apply(to_datetime, format="%b %y", cache=cache)
        tm.assert_series_equal(result, expected)
    def test_to_datetime_timezone_name(self):
        # 测试函数：test_to_datetime_timezone_name
        # 问题跟踪链接：https://github.com/pandas-dev/pandas/issues/49748
        # 调用 to_datetime 函数，将字符串转换为 Timestamp 对象，指定格式为 "%Y-%m-%d %H:%M:%S%Z"
        result = to_datetime("2020-01-01 00:00:00UTC", format="%Y-%m-%d %H:%M:%S%Z")
        # 期望结果：创建一个带有 UTC 时区的本地化 Timestamp 对象
        expected = Timestamp(2020, 1, 1).tz_localize("UTC")
        # 断言结果与期望值相等
        assert result == expected

    @td.skip_if_not_us_locale
    @pytest.mark.parametrize("errors", ["raise", "coerce"])
    def test_to_datetime_with_apply_with_empty_str(self, cache, errors):
        # 测试函数：test_to_datetime_with_apply_with_empty_str
        # 此处仅适用于 US/None 区域设置的测试
        # GitHub 问题跟踪：GH 5195, GH50251
        # 使用指定格式和 coerce 参数时，to_datetime 处理空字符串时会失败
        td = Series(["May 04", "Jun 02", ""], index=[1, 2, 3])
        # 期望结果：调用 to_datetime 处理 Series 对象 td，使用指定的格式和错误处理方式
        expected = to_datetime(td, format="%b %y", errors=errors, cache=cache)

        # 使用 apply 方法调用 to_datetime 处理每个元素
        result = td.apply(
            lambda x: to_datetime(x, format="%b %y", errors="coerce", cache=cache)
        )
        # 断言结果与期望值相等
        tm.assert_series_equal(result, expected)

    def test_to_datetime_empty_stt(self, cache):
        # 测试函数：test_to_datetime_empty_stt
        # 处理空字符串的情况
        result = to_datetime("", cache=cache)
        # 断言结果为 NaT（Not a Time）
        assert result is NaT

    def test_to_datetime_empty_str_list(self, cache):
        # 测试函数：test_to_datetime_empty_str_list
        # 处理空字符串列表的情况
        result = to_datetime(["", ""], cache=cache)
        # 断言所有结果都为缺失值（NaN）
        assert isna(result).all()

    def test_to_datetime_zero(self, cache):
        # 测试函数：test_to_datetime_zero
        # 处理整数的情况
        result = Timestamp(0)
        # 调用 to_datetime 处理整数，期望结果为 Timestamp 对象
        expected = to_datetime(0, cache=cache)
        # 断言结果与期望值相等
        assert result == expected

    def test_to_datetime_strings(self, cache):
        # 测试函数：test_to_datetime_strings
        # 处理字符串的情况
        # GitHub 问题跟踪：GH 3888 (字符串处理)
        # 期望结果：调用 to_datetime 处理字符串 "2012"，返回的第一个元素是 Timestamp 对象
        expected = to_datetime(["2012"], cache=cache)[0]
        # 调用 to_datetime 处理字符串 "2012"，返回 Timestamp 对象
        result = to_datetime("2012", cache=cache)
        # 断言结果与期望值相等
        assert result == expected

    def test_to_datetime_strings_variation(self, cache):
        # 测试函数：test_to_datetime_strings_variation
        array = ["2012", "20120101", "20120101 12:01:01"]
        # 使用列表推导调用 to_datetime 处理每个日期字符串
        expected = [to_datetime(dt_str, cache=cache) for dt_str in array]
        # 创建 Timestamp 对象列表
        result = [Timestamp(date_str) for date_str in array]
        # 检查两个列表的元素近似相等
        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize("result", [Timestamp("2012"), to_datetime("2012")])
    def test_to_datetime_strings_vs_constructor(self, result):
        # 测试函数：test_to_datetime_strings_vs_constructor
        # 比较 to_datetime 函数与 Timestamp 构造函数的结果
        expected = Timestamp(2012, 1, 1)
        # 断言结果与期望值相等
        assert result == expected

    def test_to_datetime_unprocessable_input(self, cache):
        # 测试函数：test_to_datetime_unprocessable_input
        # GitHub 问题跟踪：GH 4928, GH 21864
        # 使用 raise 错误处理方式调用 to_datetime 处理不可处理的输入
        msg = '^Given date string "1" not likely a datetime, at position 1$'
        # 断言抛出 ValueError 异常，并且异常消息符合指定的正则表达式模式
        with pytest.raises(ValueError, match=msg):
            to_datetime([1, "1"], errors="raise", cache=cache)

    def test_to_datetime_other_datetime64_units(self):
        # 测试函数：test_to_datetime_other_datetime64_units
        # 处理其他的 datetime64 单位的情况
        # 表示日期：5/25/2012
        scalar = np.int64(1337904000000000).view("M8[us]")
        as_obj = scalar.astype("O")

        index = DatetimeIndex([scalar])
        # 断言索引的第一个元素等于 scalar 的对象表示形式
        assert index[0] == scalar.astype("O")

        value = Timestamp(scalar)
        # 断言 Timestamp 对象等于 as_obj
        assert value == as_obj

    def test_to_datetime_list_of_integers(self):
        # 测试函数：test_to_datetime_list_of_integers
        rng = date_range("1/1/2000", periods=20)
        rng = DatetimeIndex(rng.values)

        ints = list(rng.asi8)

        # 调用 DatetimeIndex 构造函数处理整数列表 ints
        result = DatetimeIndex(ints)

        # 断言两个 DatetimeIndex 对象相等
        tm.assert_index_equal(rng, result)
    # 定义测试函数，测试日期时间转换函数的溢出情况
    def test_to_datetime_overflow(self):
        # 标记此测试用例解决的问题编号 GH-17637
        # 在这里我们演示了Timedelta范围的溢出
        msg = "Cannot cast 139999 days 00:00:00 to unit='ns' without overflow"
        # 使用pytest的断言来检查是否引发了OutOfBoundsTimedelta异常，并匹配特定的错误消息
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            # 调用日期范围生成函数，尝试创建一个超出范围的时间跨度
            date_range(start="1/1/1700", freq="B", periods=100000)

    # 定义测试函数，测试字符串日期时间转换函数处理无效操作的情况
    def test_string_invalid_operation(self, cache):
        # 创建包含无效日期时间字符串的NumPy数组
        invalid = np.array(["87156549591102612381000001219H5"], dtype=object)
        # GH 编号 #51084

        # 使用pytest的断言来检查是否引发了值错误异常，并匹配特定的错误消息
        with pytest.raises(ValueError, match="Unknown datetime string format"):
            # 调用字符串日期时间转换函数，尝试转换包含无效字符串的数组，期望引发异常
            to_datetime(invalid, errors="raise", cache=cache)

    # 定义测试函数，测试字符串日期时间转换函数处理缺失值和NaT的转换情况
    def test_string_na_nat_conversion(self, cache):
        # GH 编号 #999, #858

        # 创建包含日期时间字符串和NaN值的NumPy数组
        strings = np.array(["1/1/2000", "1/2/2000", np.nan, "1/4/2000"], dtype=object)

        # 创建预期的结果数组，用于存储转换后的日期时间对象
        expected = np.empty(4, dtype="M8[s]")
        # 遍历字符串数组，根据内容转换为对应的日期时间对象或NaT
        for i, val in enumerate(strings):
            if isna(val):
                expected[i] = iNaT
            else:
                expected[i] = parse(val)

        # 调用tslib中的日期时间数组转换函数，并获取结果的第一个元素
        result = tslib.array_to_datetime(strings)[0]
        # 使用测试工具（tm）来断言结果与预期结果的近似相等性
        tm.assert_almost_equal(result, expected)

        # 调用字符串日期时间转换函数，并检查返回结果是否为DatetimeIndex对象
        result2 = to_datetime(strings, cache=cache)
        assert isinstance(result2, DatetimeIndex)
        # 使用测试工具（tm）来断言转换后的结果数组与预期结果的数值部分相等
        tm.assert_numpy_array_equal(result, result2.values)

    # 定义测试函数，测试字符串日期时间转换函数处理不规范字符串的转换情况
    def test_string_na_nat_conversion_malformed(self, cache):
        # 创建包含不规范日期时间字符串和NaN值的NumPy数组
        malformed = np.array(["1/100/2000", np.nan], dtype=object)

        # GH 编号 10636，默认的错误处理方式现在是 'raise'
        msg = r"Unknown datetime string format"
        # 使用pytest的断言来检查是否引发了值错误异常，并匹配特定的错误消息
        with pytest.raises(ValueError, match=msg):
            to_datetime(malformed, errors="raise", cache=cache)

        with pytest.raises(ValueError, match=msg):
            to_datetime(malformed, errors="raise", cache=cache)

    # 定义测试函数，测试字符串日期时间转换函数处理带有名称的Series对象的转换情况
    def test_string_na_nat_conversion_with_name(self, cache):
        # 创建索引为字符串的Series对象，包含日期时间字符串和NaN值
        idx = ["a", "b", "c", "d", "e"]
        series = Series(
            ["1/1/2000", np.nan, "1/3/2000", np.nan, "1/5/2000"], index=idx, name="foo"
        )
        # 创建转换后的日期时间Series对象，与上述Series对象对应
        dseries = Series(
            [
                to_datetime("1/1/2000", cache=cache),
                np.nan,
                to_datetime("1/3/2000", cache=cache),
                np.nan,
                to_datetime("1/5/2000", cache=cache),
            ],
            index=idx,
            name="foo",
        )

        # 调用字符串日期时间转换函数，并断言结果Series与预期结果的近似相等性（忽略名称检查）
        result = to_datetime(series, cache=cache)
        tm.assert_series_equal(result, dseries, check_names=False)
        assert result.name == "foo"

        # 再次调用字符串日期时间转换函数，并断言结果Series与预期结果的近似相等性（忽略名称检查）
        dresult = to_datetime(dseries, cache=cache)
        tm.assert_series_equal(dresult, dseries, check_names=False)
        assert dresult.name == "foo"

    # 使用pytest的参数化装饰器，定义日期时间单位的多组参数化测试
    @pytest.mark.parametrize(
        "unit",
        ["h", "m", "s", "ms", "us", "ns"],
    )
    # 定义一个测试函数，用于测试根据不同的时间单位构造日期时间索引
    def test_dti_constructor_numpy_timeunits(self, cache, unit):
        # GH 9114
        # 使用给定的时间单位创建 NumPy 的日期时间 dtype 对象
        dtype = np.dtype(f"M8[{unit}]")
        # 使用缓存创建基础的日期时间索引对象，包括特定的日期时间和缺失值 "NaT"
        base = to_datetime(["2000-01-01T00:00", "2000-01-02T00:00", "NaT"], cache=cache)

        # 将基础日期时间索引的值转换为指定的 dtype
        values = base.values.astype(dtype)

        # 如果时间单位在 ["h", "m"] 中，则转换为最接近支持的单位 "s"
        if unit in ["h", "m"]:
            unit = "s"
        
        # 期望的 dtype 是根据调整后的单位重新创建的
        exp_dtype = np.dtype(f"M8[{unit}]")
        expected = DatetimeIndex(base.astype(exp_dtype))
        
        # 断言期望的 dtype 与实际的 dtype 相符
        assert expected.dtype == exp_dtype

        # 使用测试工具方法验证创建的日期时间索引与期望的索引相等
        tm.assert_index_equal(DatetimeIndex(values), expected)
        tm.assert_index_equal(to_datetime(values, cache=cache), expected)

    # 定义一个测试函数，用于测试日期时间索引的 dayfirst 参数
    def test_dayfirst(self, cache):
        # GH 5917
        # 提供一个日期字符串数组
        arr = ["10/02/2014", "11/02/2014", "12/02/2014"]
        # 创建一个预期的日期时间索引，强制转换为秒级单位
        expected = DatetimeIndex(
            [datetime(2014, 2, 10), datetime(2014, 2, 11), datetime(2014, 2, 12)]
        ).as_unit("s")

        # 分别测试不同的输入类型和 dayfirst 参数设置
        idx1 = DatetimeIndex(arr, dayfirst=True)
        idx2 = DatetimeIndex(np.array(arr), dayfirst=True)
        idx3 = to_datetime(arr, dayfirst=True, cache=cache)
        idx4 = to_datetime(np.array(arr), dayfirst=True, cache=cache)
        idx5 = DatetimeIndex(Index(arr), dayfirst=True)
        idx6 = DatetimeIndex(Series(arr), dayfirst=True)

        # 使用测试工具方法验证每个索引是否与预期的索引相等
        tm.assert_index_equal(expected, idx1)
        tm.assert_index_equal(expected, idx2)
        tm.assert_index_equal(expected, idx3)
        tm.assert_index_equal(expected, idx4)
        tm.assert_index_equal(expected, idx5)
        tm.assert_index_equal(expected, idx6)

    # 定义一个测试函数，用于测试 dayfirst 参数在警告处理方面的有效性
    def test_dayfirst_warnings_valid_input(self):
        # GH 12585
        # 准备警告信息的正则表达式模式
        warning_msg = (
            "Parsing dates in .* format when dayfirst=.* was specified. "
            "Pass `dayfirst=.*` or specify a format to silence this warning."
        )

        # CASE 1: valid input
        # 提供一个有效的日期字符串数组
        arr = ["31/12/2014", "10/03/2011"]
        # 创建一个预期的日期时间索引对象，指定 dtype 为 datetime64[s]，频率为 None
        expected = DatetimeIndex(
            ["2014-12-31", "2011-03-10"], dtype="datetime64[s]", freq=None
        )

        # A. dayfirst 参数设置正确，不应有警告
        res1 = to_datetime(arr, dayfirst=True)
        tm.assert_index_equal(expected, res1)

        # B. dayfirst 参数设置错误，应该有警告
        with tm.assert_produces_warning(UserWarning, match=warning_msg):
            res2 = to_datetime(arr, dayfirst=False)
        tm.assert_index_equal(expected, res2)

    # 定义一个测试函数，用于测试 dayfirst 参数在处理无效输入时是否引发异常
    def test_dayfirst_warnings_invalid_input(self):
        # CASE 2: invalid input
        # 无法一致地处理具有单一格式的日期字符串，预计会引发 ValueError 异常

        # 第一个日期格式为 DD/MM/YYYY，第二个日期格式为 MM/DD/YYYY
        arr = ["31/12/2014", "03/30/2011"]

        # 预期会引发 ValueError 异常，给出详细的异常信息匹配模式
        with pytest.raises(
            ValueError,
            match=(
                r'^time data "03/30/2011" doesn\'t match format '
                rf'"%d/%m/%Y", at position 1. {PARSING_ERR_MSG}$'
            ),
        ):
            to_datetime(arr, dayfirst=True)

    # 使用 pytest 的参数化标记，定义多个测试用例类别
    @pytest.mark.parametrize("klass", [DatetimeIndex, DatetimeArray._from_sequence])
    # 定义一个测试方法 `test_to_datetime_dta_tz`，用于测试将日期时间数据转换为指定时区的功能
    def test_to_datetime_dta_tz(self, klass):
        # GH#27733
        # 创建一个日期范围，从 "2015-04-05" 开始，连续三天的日期，并将其命名为 "foo"
        dti = date_range("2015-04-05", periods=3).rename("foo")
        # 期望的结果是将日期时间索引设置为 UTC 时区
        expected = dti.tz_localize("UTC")

        # 使用给定的类 `klass` 创建一个对象 `obj`，并将 `dti` 作为其参数
        obj = klass(dti)
        # 将期望的日期时间索引也使用 `klass` 创建为 `expected` 对象
        expected = klass(expected)

        # 调用 `to_datetime` 函数，将 `obj` 转换为 UTC 时区的日期时间对象
        result = to_datetime(obj, utc=True)
        # 使用测试框架的 `tm.assert_equal` 方法来比较 `result` 和 `expected` 是否相等
        tm.assert_equal(result, expected)
# 定义一个测试类 TestGuessDatetimeFormat，用于测试日期时间格式猜测功能
class TestGuessDatetimeFormat:
    
    # 使用 pytest 的参数化装饰器，指定多个测试参数
    @pytest.mark.parametrize(
        "test_list",
        [
            [
                "2011-12-30 00:00:00.000000",
                "2011-12-30 00:00:00.000000",
                "2011-12-30 00:00:00.000000",
            ],
            [np.nan, np.nan, "2011-12-30 00:00:00.000000"],
            ["", "2011-12-30 00:00:00.000000"],
            ["NaT", "2011-12-30 00:00:00.000000"],
            ["2011-12-30 00:00:00.000000", "random_string"],
            ["now", "2011-12-30 00:00:00.000000"],
            ["today", "2011-12-30 00:00:00.000000"],
        ],
    )
    # 定义测试方法 test_guess_datetime_format_for_array，参数为 test_list
    def test_guess_datetime_format_for_array(self, test_list):
        # 预期的日期时间格式
        expected_format = "%Y-%m-%d %H:%M:%S.%f"
        # 将 test_list 转换为 NumPy 数组，数据类型为 object
        test_array = np.array(test_list, dtype=object)
        # 断言调用 tools 模块的 _guess_datetime_format_for_array 方法返回预期的日期时间格式
        assert tools._guess_datetime_format_for_array(test_array) == expected_format

    # 使用自定义装饰器 td.skip_if_not_us_locale 标记的测试方法
    @td.skip_if_not_us_locale
    # 定义测试方法 test_guess_datetime_format_for_array_all_nans
    def test_guess_datetime_format_for_array_all_nans(self):
        # 创建包含 NaN 的 NumPy 数组，数据类型为对象
        format_for_string_of_nans = tools._guess_datetime_format_for_array(
            np.array([np.nan, np.nan, np.nan], dtype="O")
        )
        # 断言 format_for_string_of_nans 应为 None
        assert format_for_string_of_nans is None


# 定义测试类 TestToDatetimeInferFormat
class TestToDatetimeInferFormat:
    
    # 使用 pytest 的参数化装饰器，指定多个测试参数
    @pytest.mark.parametrize(
        "test_format", ["%m-%d-%Y", "%m/%d/%Y %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f"]
    )
    # 定义测试方法 test_to_datetime_infer_datetime_format_consistent_format，参数为 test_format
    def test_to_datetime_infer_datetime_format_consistent_format(
        self, cache, test_format
    ):
        # 创建时间序列 ser，包含 50 个小时频率的日期范围
        ser = Series(date_range("20000101", periods=50, freq="h"))
        # 将 ser 中的每个元素按照指定的 test_format 格式化为字符串
        s_as_dt_strings = ser.apply(lambda x: x.strftime(test_format))
        # 使用 to_datetime 方法将 s_as_dt_strings 转换为日期时间对象，指定格式为 test_format，启用缓存
        with_format = to_datetime(s_as_dt_strings, format=test_format, cache=cache)
        # 使用 to_datetime 方法将 s_as_dt_strings 转换为日期时间对象，自动推断格式，启用缓存
        without_format = to_datetime(s_as_dt_strings, cache=cache)
        # 断言 with_format 和 without_format 应当相等
        tm.assert_series_equal(with_format, without_format)

    # 定义测试方法 test_to_datetime_inconsistent_format，参数为 cache
    def test_to_datetime_inconsistent_format(self, cache):
        # 创建字符串数组 data
        data = ["01/01/2011 00:00:00", "01-02-2011 00:00:00", "2011-01-03T00:00:00"]
        # 创建 Series 对象 ser，数据类型为对象
        ser = Series(np.array(data))
        # 定义异常消息的正则表达式模式
        msg = (
            r'^time data "01-02-2011 00:00:00" doesn\'t match format '
            rf'"%m/%d/%Y %H:%M:%S", at position 1. {PARSING_ERR_MSG}$'
        )
        # 使用 pytest 的 raises 断言，验证调用 to_datetime 方法时会抛出 ValueError 异常，并匹配预期的异常消息
        with pytest.raises(ValueError, match=msg):
            to_datetime(ser, cache=cache)

    # 定义测试方法 test_to_datetime_consistent_format，参数为 cache
    def test_to_datetime_consistent_format(self, cache):
        # 创建字符串数组 data
        data = ["Jan/01/2011", "Feb/01/2011", "Mar/01/2011"]
        # 创建 Series 对象 ser，数据类型为对象
        ser = Series(np.array(data))
        # 调用 to_datetime 方法将 ser 转换为日期时间对象，启用缓存
        result = to_datetime(ser, cache=cache)
        # 创建预期的 Series 对象 expected，数据类型为 datetime64[s]
        expected = Series(
            ["2011-01-01", "2011-02-01", "2011-03-01"], dtype="datetime64[s]"
        )
        # 使用 tm.assert_series_equal 断言 result 和 expected 应当相等
        tm.assert_series_equal(result, expected)
    # 使用 pytest 框架测试 to_datetime 函数处理包含 NaN 值的 Series 对象的情况
    def test_to_datetime_series_with_nans(self, cache):
        # 创建包含日期时间字符串和 NaN 值的 Series 对象
        ser = Series(
            np.array(
                ["01/01/2011 00:00:00", np.nan, "01/03/2011 00:00:00", np.nan],
                dtype=object,
            )
        )
        # 调用 to_datetime 函数，转换日期时间，传入缓存参数
        result = to_datetime(ser, cache=cache)
        # 创建期望的 Series 对象，包含转换后的日期时间和 NaT（Not a Time）值
        expected = Series(["2011-01-01", NaT, "2011-01-03", NaT], dtype="datetime64[s]")
        # 使用 pytest 的 assert_series_equal 函数比较结果和期望的 Series 对象
        tm.assert_series_equal(result, expected)

    # 使用 pytest 框架测试处理以 NaN 值开头的 Series 对象的 to_datetime 函数
    def test_to_datetime_series_start_with_nans(self, cache):
        # 创建以 NaN 值开头的 Series 对象，包含日期时间字符串和 NaN 值
        ser = Series(
            np.array(
                [
                    np.nan,
                    np.nan,
                    "01/01/2011 00:00:00",
                    "01/02/2011 00:00:00",
                    "01/03/2011 00:00:00",
                ],
                dtype=object,
            )
        )
        # 调用 to_datetime 函数，转换日期时间，传入缓存参数
        result = to_datetime(ser, cache=cache)
        # 创建期望的 Series 对象，包含转换后的日期时间和 NaT 值
        expected = Series(
            [NaT, NaT, "2011-01-01", "2011-01-02", "2011-01-03"], dtype="datetime64[s]"
        )
        # 使用 pytest 的 assert_series_equal 函数比较结果和期望的 Series 对象
        tm.assert_series_equal(result, expected)

    # 使用 pytest 框架测试处理不同时区名称的 to_datetime 函数
    @pytest.mark.parametrize(
        "tz_name, offset",
        [("UTC", 0), ("UTC-3", 180), ("UTC+3", -180)],
    )
    def test_infer_datetime_format_tz_name(self, tz_name, offset):
        # GH 33133
        # 创建包含带有时区名称的日期时间字符串的 Series 对象
        ser = Series([f"2019-02-02 08:07:13 {tz_name}"])
        # 调用 to_datetime 函数，转换日期时间，不使用缓存参数
        result = to_datetime(ser)
        # 根据偏移量创建时区对象
        tz = timezone(timedelta(minutes=offset))
        # 创建期望的 Series 对象，包含本地化时区后的 Timestamp 对象
        expected = Series([Timestamp("2019-02-02 08:07:13").tz_localize(tz)])
        # 将期望的 Series 对象转换为秒单位
        expected = expected.dt.as_unit("s")
        # 使用 pytest 的 assert_series_equal 函数比较结果和期望的 Series 对象
        tm.assert_series_equal(result, expected)

    # 使用 pytest 框架测试处理带有零时区（Z）的日期时间字符串的 to_datetime 函数
    @pytest.mark.parametrize(
        "ts,zero_tz",
        [
            ("2019-02-02 08:07:13", "Z"),
            ("2019-02-02 08:07:13", ""),
            ("2019-02-02 08:07:13.012345", "Z"),
            ("2019-02-02 08:07:13.012345", ""),
        ],
    )
    def test_infer_datetime_format_zero_tz(self, ts, zero_tz):
        # GH 41047
        # 创建带有零时区标识的日期时间字符串的 Series 对象
        ser = Series([ts + zero_tz])
        # 调用 to_datetime 函数，转换日期时间，不使用缓存参数
        result = to_datetime(ser)
        # 根据零时区标识选择时区对象
        tz = timezone.utc if zero_tz == "Z" else None
        # 创建期望的 Series 对象，包含带有所选时区的 Timestamp 对象
        expected = Series([Timestamp(ts, tz=tz)])
        # 使用 pytest 的 assert_series_equal 函数比较结果和期望的 Series 对象
        tm.assert_series_equal(result, expected)

    # 使用 pytest 框架测试处理 ISO 8601 格式日期的 to_datetime 函数
    @pytest.mark.parametrize("format", [None, "%Y-%m-%d"])
    def test_to_datetime_iso8601_noleading_0s(self, cache, format):
        # GH 11871
        # 创建包含 ISO 8601 格式日期字符串的 Series 对象
        ser = Series(["2014-1-1", "2014-2-2", "2015-3-3"])
        # 创建期望的 Series 对象，包含转换后的日期时间
        expected = Series(
            [
                Timestamp("2014-01-01"),
                Timestamp("2014-02-02"),
                Timestamp("2015-03-03"),
            ]
        )
        # 调用 to_datetime 函数，转换日期时间，传入缓存和格式参数
        result = to_datetime(ser, format=format, cache=cache)
        # 使用 pytest 的 assert_series_equal 函数比较结果和期望的 Series 对象
        tm.assert_series_equal(result, expected)
class TestDaysInMonth:
    # tests for issue #10154

    @pytest.mark.parametrize(
        "arg, format",
        [
            ["2015-02-29", None],   # 参数化测试，检查日期转换函数对不合法日期的处理
            ["2015-02-29", "%Y-%m-%d"],   # 参数化测试，检查日期转换函数对不合法日期的处理
            ["2015-02-32", "%Y-%m-%d"],   # 参数化测试，检查日期转换函数对不合法日期的处理
            ["2015-04-31", "%Y-%m-%d"],   # 参数化测试，检查日期转换函数对不合法日期的处理
        ],
    )
    def test_day_not_in_month_coerce(self, cache, arg, format):
        # 断言检查，使用 'coerce' 错误处理模式转换日期
        assert isna(to_datetime(arg, errors="coerce", format=format, cache=cache))

    def test_day_not_in_month_raise(self, cache):
        # 使用 'raise' 错误处理模式，断言检查是否抛出值错误异常
        msg = "day is out of range for month: 2015-02-29, at position 0"
        with pytest.raises(ValueError, match=msg):
            to_datetime("2015-02-29", errors="raise", cache=cache)

    @pytest.mark.parametrize(
        "arg, format, msg",
        [
            (
                "2015-02-29",
                "%Y-%m-%d",
                f"^day is out of range for month, at position 0. {PARSING_ERR_MSG}$",
            ),
            (
                "2015-29-02",
                "%Y-%d-%m",
                f"^day is out of range for month, at position 0. {PARSING_ERR_MSG}$",
            ),
            (
                "2015-02-32",
                "%Y-%m-%d",
                '^unconverted data remains when parsing with format "%Y-%m-%d": "2", '
                f"at position 0. {PARSING_ERR_MSG}$",
            ),
            (
                "2015-32-02",
                "%Y-%d-%m",
                '^time data "2015-32-02" doesn\'t match format "%Y-%d-%m", '
                f"at position 0. {PARSING_ERR_MSG}$",
            ),
            (
                "2015-04-31",
                "%Y-%m-%d",
                f"^day is out of range for month, at position 0. {PARSING_ERR_MSG}$",
            ),
            (
                "2015-31-04",
                "%Y-%d-%m",
                f"^day is out of range for month, at position 0. {PARSING_ERR_MSG}$",
            ),
        ],
    )
    def test_day_not_in_month_raise_value(self, cache, arg, format, msg):
        # 参数化测试，检查日期转换函数对不合法日期的处理，验证是否抛出特定的值错误异常
        # 参考 GitHub issue #50462
        with pytest.raises(ValueError, match=msg):
            to_datetime(arg, errors="raise", format=format, cache=cache)
    # 使用 pytest.mark.parametrize 装饰器为单元测试参数化
    @pytest.mark.parametrize(
        # 定义参数化测试的参数：日期字符串和期望的日期时间对象
        "date_str, expected",
        [
            # 不同格式的日期字符串及其对应的期望日期时间对象
            ("2011-01-01", datetime(2011, 1, 1)),
            ("2Q2005", datetime(2005, 4, 1)),
            ("2Q05", datetime(2005, 4, 1)),
            ("2005Q1", datetime(2005, 1, 1)),
            ("05Q1", datetime(2005, 1, 1)),
            ("2011Q3", datetime(2011, 7, 1)),
            ("11Q3", datetime(2011, 7, 1)),
            ("3Q2011", datetime(2011, 7, 1)),
            ("3Q11", datetime(2011, 7, 1)),
            ("2000Q4", datetime(2000, 10, 1)),
            ("00Q4", datetime(2000, 10, 1)),
            ("4Q2000", datetime(2000, 10, 1)),
            ("4Q00", datetime(2000, 10, 1)),
            ("2000q4", datetime(2000, 10, 1)),
            ("2000-Q4", datetime(2000, 10, 1)),
            ("00-Q4", datetime(2000, 10, 1)),
            ("4Q-2000", datetime(2000, 10, 1)),
            ("4Q-00", datetime(2000, 10, 1)),
            ("00q4", datetime(2000, 10, 1)),
            ("2005", datetime(2005, 1, 1)),
            ("2005-11", datetime(2005, 11, 1)),
            ("2005 11", datetime(2005, 11, 1)),
            ("11-2005", datetime(2005, 11, 1)),
            ("11 2005", datetime(2005, 11, 1)),
            ("200511", datetime(2020, 5, 11)),
            ("20051109", datetime(2005, 11, 9)),
            ("20051109 10:15", datetime(2005, 11, 9, 10, 15)),
            ("20051109 08H", datetime(2005, 11, 9, 8, 0)),
            ("2005-11-09 10:15", datetime(2005, 11, 9, 10, 15)),
            ("2005-11-09 08H", datetime(2005, 11, 9, 8, 0)),
            ("2005/11/09 10:15", datetime(2005, 11, 9, 10, 15)),
            ("2005/11/09 10:15:32", datetime(2005, 11, 9, 10, 15, 32)),
            ("2005/11/09 10:15:32 AM", datetime(2005, 11, 9, 10, 15, 32)),
            ("2005/11/09 10:15:32 PM", datetime(2005, 11, 9, 22, 15, 32)),
            ("2005/11/09 08H", datetime(2005, 11, 9, 8, 0)),
            ("Thu Sep 25 10:36:28 2003", datetime(2003, 9, 25, 10, 36, 28)),
            ("Thu Sep 25 2003", datetime(2003, 9, 25)),
            ("Sep 25 2003", datetime(2003, 9, 25)),
            ("January 1 2014", datetime(2014, 1, 1)),
            ("2014-06", datetime(2014, 6, 1)),
            ("06-2014", datetime(2014, 6, 1)),
            ("2014-6", datetime(2014, 6, 1)),
            ("6-2014", datetime(2014, 6, 1)),
            ("20010101 12", datetime(2001, 1, 1, 12)),
            ("20010101 1234", datetime(2001, 1, 1, 12, 34)),
            ("20010101 123456", datetime(2001, 1, 1, 12, 34, 56)),
        ],
    )
    @pytest.mark.parametrize(
        "date_str, dayfirst, yearfirst, expected",
        [
            ("10-11-12", False, False, datetime(2012, 10, 11)),
            ("10-11-12", True, False, datetime(2012, 11, 10)),
            ("10-11-12", False, True, datetime(2010, 11, 12)),
            ("10-11-12", True, True, datetime(2010, 12, 11)),
            ("20/12/21", False, False, datetime(2021, 12, 20)),
            ("20/12/21", True, False, datetime(2021, 12, 20)),
            ("20/12/21", False, True, datetime(2020, 12, 21)),
            ("20/12/21", True, True, datetime(2020, 12, 21)),
        ],
    )


        # 使用 pytest 的 parametrize 标记，定义了多组参数化测试数据
        @pytest.mark.parametrize(
            "date_str, dayfirst, yearfirst, expected",
            [
                ("10-11-12", False, False, datetime(2012, 10, 11)),
                ("10-11-12", True, False, datetime(2012, 11, 10)),
                ("10-11-12", False, True, datetime(2010, 11, 12)),
                ("10-11-12", True, True, datetime(2010, 12, 11)),
                ("20/12/21", False, False, datetime(2021, 12, 20)),
                ("20/12/21", True, False, datetime(2021, 12, 20)),
                ("20/12/21", False, True, datetime(2020, 12, 21)),
                ("20/12/21", True, True, datetime(2020, 12, 21)),
            ],
        )
        # 这段代码定义了一个使用 pytest 的参数化测试，用于测试日期时间解析的不同情况
    # 定义测试方法，测试日期解析器的不同参数组合
    def test_parsers_dayfirst_yearfirst(
        self, cache, date_str, dayfirst, yearfirst, expected
    ):
        # 调试信息：以下是一系列日期字符串和对应的预期结果

        # OK
        # 2.5.1 10-11-12   [dayfirst=0, yearfirst=0] -> 2012-10-11 00:00:00
        # 2.5.2 10-11-12   [dayfirst=0, yearfirst=1] -> 2012-10-11 00:00:00
        # 2.5.3 10-11-12   [dayfirst=0, yearfirst=0] -> 2012-10-11 00:00:00

        # OK
        # 2.5.1 10-11-12   [dayfirst=0, yearfirst=1] -> 2010-11-12 00:00:00
        # 2.5.2 10-11-12   [dayfirst=0, yearfirst=1] -> 2010-11-12 00:00:00
        # 2.5.3 10-11-12   [dayfirst=0, yearfirst=1] -> 2010-11-12 00:00:00

        # bug fix in 2.5.2
        # 2.5.1 10-11-12   [dayfirst=1, yearfirst=1] -> 2010-11-12 00:00:00
        # 2.5.2 10-11-12   [dayfirst=1, yearfirst=1] -> 2010-12-11 00:00:00
        # 2.5.3 10-11-12   [dayfirst=1, yearfirst=1] -> 2010-12-11 00:00:00

        # OK
        # 2.5.1 10-11-12   [dayfirst=1, yearfirst=0] -> 2012-11-10 00:00:00
        # 2.5.2 10-11-12   [dayfirst=1, yearfirst=0] -> 2012-11-10 00:00:00
        # 2.5.3 10-11-12   [dayfirst=1, yearfirst=0] -> 2012-11-10 00:00:00

        # OK
        # 2.5.1 20/12/21   [dayfirst=0, yearfirst=0] -> 2021-12-20 00:00:00
        # 2.5.2 20/12/21   [dayfirst=0, yearfirst=0] -> 2021-12-20 00:00:00
        # 2.5.3 20/12/21   [dayfirst=0, yearfirst=0] -> 2021-12-20 00:00:00

        # OK
        # 2.5.1 20/12/21   [dayfirst=0, yearfirst=1] -> 2020-12-21 00:00:00
        # 2.5.2 20/12/21   [dayfirst=0, yearfirst=1] -> 2020-12-21 00:00:00
        # 2.5.3 20/12/21   [dayfirst=0, yearfirst=1] -> 2020-12-21 00:00:00

        # revert of bug in 2.5.2
        # 2.5.1 20/12/21   [dayfirst=1, yearfirst=1] -> 2020-12-21 00:00:00
        # 2.5.2 20/12/21   [dayfirst=1, yearfirst=1] -> month must be in 1..12
        # 2.5.3 20/12/21   [dayfirst=1, yearfirst=1] -> 2020-12-21 00:00:00

        # OK
        # 2.5.1 20/12/21   [dayfirst=1, yearfirst=0] -> 2021-12-20 00:00:00
        # 2.5.2 20/12/21   [dayfirst=1, yearfirst=0] -> 2021-12-20 00:00:00
        # 2.5.3 20/12/21   [dayfirst=1, yearfirst=0] -> 2021-12-20 00:00:00

        # str : dayfirst, yearfirst, expected

        # 使用 dateutil 库解析日期字符串，并断言解析结果与预期结果相等
        dateutil_result = parse(date_str, dayfirst=dayfirst, yearfirst=yearfirst)
        assert dateutil_result == expected

        # 调用 parsing 模块的方法解析日期字符串，获取解析结果和解析的精度信息
        result1, _ = parsing.parse_datetime_string_with_reso(
            date_str, dayfirst=dayfirst, yearfirst=yearfirst
        )

        # 如果不支持 dayfirst 和 yearfirst 参数，则使用 Timestamp 解析日期字符串，并断言结果与预期相等
        if not dayfirst and not yearfirst:
            result2 = Timestamp(date_str)
            assert result2 == expected

        # 使用 to_datetime 方法解析日期字符串，并传入缓存参数，断言结果与预期相等
        result3 = to_datetime(
            date_str, dayfirst=dayfirst, yearfirst=yearfirst, cache=cache
        )

        # 使用 DatetimeIndex 方法创建包含解析日期字符串的日期时间索引，并取得第一个元素，断言其与预期相等
        result4 = DatetimeIndex([date_str], dayfirst=dayfirst, yearfirst=yearfirst)[0]

        # 断言各种解析方法的结果与预期结果相等
        assert result1 == expected
        assert result3 == expected
        assert result4 == expected
    @pytest.mark.parametrize(
        "date_str, exp_def",
        [["10:15", datetime(1, 1, 1, 10, 15)], ["9:05", datetime(1, 1, 1, 9, 5)]],
    )
    # 使用 pytest 的 parametrize 装饰器为 test_parsers_timestring 方法提供多组参数化测试数据
    def test_parsers_timestring(self, date_str, exp_def):
        # 将 date_str 解析为 datetime 对象，与 dateutil 的解析结果进行比较
        exp_now = parse(date_str)

        # 调用 parsing 模块的 parse_datetime_string_with_reso 方法解析日期时间字符串
        result1, _ = parsing.parse_datetime_string_with_reso(date_str)
        # 调用 to_datetime 函数将时间字符串转换为 datetime 对象
        result2 = to_datetime(date_str)
        # 调用 to_datetime 函数处理包含时间字符串的列表，转换为 datetime 对象
        result3 = to_datetime([date_str])
        # 使用 Timestamp 类将时间字符串转换为 Timestamp 对象
        result4 = Timestamp(date_str)
        # 使用 DatetimeIndex 类将时间字符串转换为 DatetimeIndex 对象，并取第一个元素
        result5 = DatetimeIndex([date_str])[0]
        
        # 断言各个转换后的结果与预期的 datetime 对象相等
        assert result1 == exp_def
        assert result2 == exp_now
        assert result3 == exp_now
        assert result4 == exp_now
        assert result5 == exp_now

    @pytest.mark.parametrize(
        "dt_string, tz, dt_string_repr",
        [
            (
                "2013-01-01 05:45+0545",
                timezone(timedelta(minutes=345)),
                "Timestamp('2013-01-01 05:45:00+0545', tz='UTC+05:45')",
            ),
            (
                "2013-01-01 05:30+0530",
                timezone(timedelta(minutes=330)),
                "Timestamp('2013-01-01 05:30:00+0530', tz='UTC+05:30')",
            ),
        ],
    )
    # 使用 pytest 的 parametrize 装饰器为 test_parsers_timezone_minute_offsets_roundtrip 方法提供多组参数化测试数据
    def test_parsers_timezone_minute_offsets_roundtrip(
        self, cache, dt_string, tz, dt_string_repr
    ):
        # GH11708
        # 将基准时间字符串转换为 datetime 对象，并设定时区为 UTC，再转换为指定时区 tz
        base = to_datetime("2013-01-01 00:00:00", cache=cache)
        base = base.tz_localize("UTC").tz_convert(tz)
        
        # 使用 to_datetime 函数将 dt_string 转换为 datetime 对象
        dt_time = to_datetime(dt_string, cache=cache)
        
        # 断言基准时间与转换后的时间对象相等
        assert base == dt_time
        # 断言转换后的时间对象的 repr 字符串与预期的 dt_string_repr 相等
        assert dt_string_repr == repr(dt_time)
@pytest.fixture(params=["D", "s", "ms", "us", "ns"])
def units(request):
    """
    定义一个参数化的 fixture，用于测试不同的时间单位。

    * D - 天
    * s - 秒
    * ms - 毫秒
    * us - 微秒
    * ns - 纳秒
    """
    return request.param


@pytest.fixture
def julian_dates():
    """
    创建一个 Julian 日期范围的 fixture。

    返回包含从 '2014-1-1' 开始的 10 个日期的 Julian 日期数组。
    """
    return date_range("2014-1-1", periods=10).to_julian_date().values


class TestOrigin:
    def test_origin_and_unit(self):
        """
        测试 to_datetime 函数的 origin 和 unit 参数的功能。

        验证在指定不同的 origin 值和单位下，to_datetime 函数返回的时间戳是否符合预期。
        """
        # GH#42624
        ts = to_datetime(1, unit="s", origin=1)
        expected = Timestamp("1970-01-01 00:00:02")
        assert ts == expected

        ts = to_datetime(1, unit="s", origin=1_000_000_000)
        expected = Timestamp("2001-09-09 01:46:41")
        assert ts == expected

    def test_julian(self, julian_dates):
        """
        测试使用 Julian 日起源的 to_datetime 函数。

        验证传入 Julian 日期和 origin='julian' 参数时，to_datetime 函数返回的 Series 是否符合预期。
        """
        # gh-11276, gh-11745
        # for origin as julian

        result = Series(to_datetime(julian_dates, unit="D", origin="julian"))
        expected = Series(
            to_datetime(julian_dates - Timestamp(0).to_julian_date(), unit="D")
        )
        tm.assert_series_equal(result, expected)

    def test_unix(self):
        """
        测试使用 Unix 起源的 to_datetime 函数。

        验证传入整数日期和 origin='unix' 参数时，to_datetime 函数返回的 Series 是否符合预期。
        """
        result = Series(to_datetime([0, 1, 2], unit="D", origin="unix"))
        expected = Series(
            [Timestamp("1970-01-01"), Timestamp("1970-01-02"), Timestamp("1970-01-03")],
            dtype="M8[ns]",
        )
        tm.assert_series_equal(result, expected)

    def test_julian_round_trip(self):
        """
        测试 Julian 日历日期与 Julian 日期之间的转换。

        验证通过 to_datetime 函数将 Julian 日历日期转换为 Julian 日期，并进行逆操作，结果是否符合预期。
        """
        result = to_datetime(2456658, origin="julian", unit="D")
        assert result.to_julian_date() == 2456658

        # out-of-bounds
        msg = "1 is Out of Bounds for origin='julian'"
        with pytest.raises(ValueError, match=msg):
            to_datetime(1, origin="julian", unit="D")

    def test_invalid_unit(self, units, julian_dates):
        """
        测试当 origin='julian' 时，不同单位是否会引发错误。

        验证当单位不为 'D' 时，to_datetime 函数是否会抛出 ValueError 异常。
        """
        if units != "D":
            msg = "unit must be 'D' for origin='julian'"
            with pytest.raises(ValueError, match=msg):
                to_datetime(julian_dates, unit=units, origin="julian")

    @pytest.mark.parametrize("unit", ["ns", "D"])
    def test_invalid_origin(self, unit):
        """
        测试当指定了无效的起源时，是否会抛出错误。

        验证当 origin 参数不是数值时，to_datetime 函数是否会抛出 ValueError 异常。
        """
        msg = "it must be numeric with a unit specified"
        with pytest.raises(ValueError, match=msg):
            to_datetime("2005-01-01", origin="1960-01-01", unit=unit)

    @pytest.mark.parametrize(
        "epochs",
        [
            Timestamp(1960, 1, 1),
            datetime(1960, 1, 1),
            "1960-01-01",
            np.datetime64("1960-01-01"),
        ],
    )
    def test_epoch(self, units, epochs):
        """
        测试使用不同起源进行日期转换的功能。

        验证通过指定不同的起源和单位，to_datetime 函数是否能正确将整数数组转换为日期时间 Series。
        """
        epoch_1960 = Timestamp(1960, 1, 1)
        units_from_epochs = np.arange(5, dtype=np.int64)
        expected = Series(
            [pd.Timedelta(x, unit=units) + epoch_1960 for x in units_from_epochs]
        )

        result = Series(to_datetime(units_from_epochs, unit=units, origin=epochs))
        tm.assert_series_equal(result, expected)
    @pytest.mark.parametrize(
        "origin, exc",
        [
            ("random_string", ValueError),  # 参数化测试：测试传入随机字符串时是否抛出 ValueError 异常
            ("epoch", ValueError),  # 参数化测试：测试传入 epoch 字符串时是否抛出 ValueError 异常
            ("13-24-1990", ValueError),  # 参数化测试：测试传入格式不正确的日期字符串时是否抛出 ValueError 异常
            (datetime(1, 1, 1), OutOfBoundsDatetime),  # 参数化测试：测试传入 datetime 对象时是否抛出 OutOfBoundsDatetime 异常
        ],
    )
    def test_invalid_origins(self, origin, exc, units):
        # 构造匹配的异常消息
        msg = "|".join(
            [
                f"origin {origin} is Out of Bounds",  # 异常消息中包含 origin 值的信息
                f"origin {origin} cannot be converted to a Timestamp",  # 异常消息中包含 origin 值的信息
                "Cannot cast .* to unit='ns' without overflow",  # 异常消息中的固定文本
            ]
        )
        # 测试是否抛出期望的异常，并匹配特定的消息
        with pytest.raises(exc, match=msg):
            to_datetime(list(range(5)), unit=units, origin=origin)

    def test_invalid_origins_tzinfo(self):
        # GH16842 测试：验证传入带有时区信息的 datetime 对象时是否抛出 ValueError 异常，并匹配特定消息
        with pytest.raises(ValueError, match="must be tz-naive"):
            to_datetime(1, unit="D", origin=datetime(2000, 1, 1, tzinfo=timezone.utc))

    def test_incorrect_value_exception(self):
        # GH47495 测试：验证传入无法解析的日期字符串列表时是否抛出 ValueError 异常，并匹配特定消息
        msg = (
            "Unknown datetime string format, unable to parse: yesterday, at position 1"
        )
        with pytest.raises(ValueError, match=msg):
            to_datetime(["today", "yesterday"])

    @pytest.mark.parametrize(
        "format, warning",
        [
            (None, UserWarning),  # 参数化测试：测试传入 None 格式时是否不抛出警告
            ("%Y-%m-%d %H:%M:%S", None),  # 参数化测试：测试传入正确格式字符串时是否不抛出警告
            ("%Y-%d-%m %H:%M:%S", None),  # 参数化测试：测试传入格式不正确的字符串时是否不抛出警告
        ],
    )
    def test_to_datetime_out_of_bounds_with_format_arg(self, format, warning):
        # see gh-23830 测试：根据 GitHub issue 号 23830，验证传入指定格式字符串时是否抛出 ValueError 异常
        if format is None:
            # 测试：验证传入指定日期字符串和格式时的返回结果是否符合预期
            res = to_datetime("2417-10-10 00:00:00.00", format=format)
            assert isinstance(res, Timestamp)
            assert res.year == 2417
            assert res.month == 10
            assert res.day == 10
        else:
            # 测试：验证传入指定日期字符串和格式时是否抛出 ValueError 异常，并匹配特定消息
            msg = "unconverted data remains when parsing with format.*, at position 0"
            with pytest.raises(ValueError, match=msg):
                to_datetime("2417-10-10 00:00:00.00", format=format)

    @pytest.mark.parametrize(
        "arg, origin, expected_str",
        [
            [200 * 365, "unix", "2169-11-13 00:00:00"],  # 参数化测试：验证传入整数天数和起始日期时的返回结果是否符合预期
            [200 * 365, "1870-01-01", "2069-11-13 00:00:00"],  # 参数化测试：验证传入整数天数和起始日期时的返回结果是否符合预期
            [300 * 365, "1870-01-01", "2169-10-20 00:00:00"],  # 参数化测试：验证传入整数天数和起始日期时的返回结果是否符合预期
        ],
    )
    def test_processing_order(self, arg, origin, expected_str):
        # 确保在构造日期之前处理超出范围的情况
        result = to_datetime(arg, unit="D", origin=origin)
        expected = Timestamp(expected_str)
        assert result == expected

        result = to_datetime(200 * 365, unit="D", origin="1870-01-01")
        expected = Timestamp("2069-11-13 00:00:00")
        assert result == expected

        result = to_datetime(300 * 365, unit="D", origin="1870-01-01")
        expected = Timestamp("2169-10-20 00:00:00")
        assert result == expected
    `
    # 使用 pytest 的参数化装饰器 @pytest.mark.parametrize，定义多组参数化测试数据
    @pytest.mark.parametrize(
        "offset,utc,exp",
        [
            ["Z", True, "2019-01-01T00:00:00.000Z"],      # 参数化测试数据集：UTC时区，预期结果为指定日期时间字符串
            ["Z", None, "2019-01-01T00:00:00.000Z"],      # 参数化测试数据集：无时区，预期结果为指定日期时间字符串
            ["-01:00", True, "2019-01-01T01:00:00.000Z"], # 参数化测试数据集：带时区偏移，预期结果为指定日期时间字符串
            ["-01:00", None, "2019-01-01T00:00:00.000-01:00"], # 参数化测试数据集：带时区偏移，预期结果为指定日期时间字符串
        ],
    )
    # 定义测试方法 test_arg_tz_ns_unit，用于测试特定的日期时间转换功能
    def test_arg_tz_ns_unit(self, offset, utc, exp):
        # GH 25546，提供测试的 issue 或功能背景说明
        arg = "2019-01-01T00:00:00.000" + offset  # 根据偏移量 offset 构建测试用的日期时间字符串
        result = to_datetime([arg], unit="ns", utc=utc)  # 执行日期时间转换操作，返回结果
        expected = to_datetime([exp]).as_unit("ns")  # 将预期结果转换为单位为纳秒的日期时间对象
        tm.assert_index_equal(result, expected)  # 使用 pandas.testing.assert_index_equal 断言实际结果与预期结果相等
class TestShouldCache:
    # 定义测试类 TestShouldCache，用于测试 should_cache 函数的行为

    @pytest.mark.parametrize(
        "listlike,do_caching",
        [
            ([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], False),
            ([1, 1, 1, 1, 4, 5, 6, 7, 8, 9], True),
        ],
    )
    def test_should_cache(self, listlike, do_caching):
        # 参数化测试函数 test_should_cache，验证 should_cache 函数的结果是否符合预期
        assert (
            tools.should_cache(listlike, check_count=len(listlike), unique_share=0.7)
            == do_caching
        )

    @pytest.mark.parametrize(
        "unique_share,check_count, err_message",
        [
            (0.5, 11, r"check_count must be in next bounds: \[0; len\(arg\)\]"),
            (10, 2, r"unique_share must be in next bounds: \(0; 1\)"),
        ],
    )
    def test_should_cache_errors(self, unique_share, check_count, err_message):
        # 参数化测试函数 test_should_cache_errors，验证对应异常情况下 should_cache 函数的异常处理是否正确
        arg = [5] * 10

        with pytest.raises(AssertionError, match=err_message):
            tools.should_cache(arg, unique_share, check_count)

    @pytest.mark.parametrize(
        "listlike",
        [
            (deque([Timestamp("2010-06-02 09:30:00")] * 51)),
            ([Timestamp("2010-06-02 09:30:00")] * 51),
            (tuple([Timestamp("2010-06-02 09:30:00")] * 51)),
        ],
    )
    def test_no_slicing_errors_in_should_cache(self, listlike):
        # 参数化测试函数 test_no_slicing_errors_in_should_cache，验证 should_cache 函数对于不支持切片的情况是否能正确处理
        # GH#29403
        assert tools.should_cache(listlike) is True


def test_nullable_integer_to_datetime():
    # 测试函数 test_nullable_integer_to_datetime，验证对应的日期时间转换函数行为是否符合预期
    # Test for #30050
    ser = Series([1, 2, None, 2**61, None])
    ser = ser.astype("Int64")
    ser_copy = ser.copy()

    res = to_datetime(ser, unit="ns")

    expected = Series(
        [
            np.datetime64("1970-01-01 00:00:00.000000001"),
            np.datetime64("1970-01-01 00:00:00.000000002"),
            np.datetime64("NaT"),
            np.datetime64("2043-01-25 23:56:49.213693952"),
            np.datetime64("NaT"),
        ]
    )
    tm.assert_series_equal(res, expected)
    # Check that ser isn't mutated
    tm.assert_series_equal(ser, ser_copy)


@pytest.mark.parametrize("klass", [np.array, list])
def test_na_to_datetime(nulls_fixture, klass):
    # 参数化测试函数 test_na_to_datetime，测试对空值处理的日期时间转换行为是否正确
    if isinstance(nulls_fixture, Decimal):
        with pytest.raises(TypeError, match="not convertible to datetime"):
            to_datetime(klass([nulls_fixture]))

    else:
        result = to_datetime(klass([nulls_fixture]))

        assert result[0] is NaT


@pytest.mark.parametrize("errors", ["raise", "coerce"])
@pytest.mark.parametrize(
    "args, format",
    [
        (["03/24/2016", "03/25/2016", ""], "%m/%d/%Y"),
        (["2016-03-24", "2016-03-25", ""], "%Y-%m-%d"),
    ],
    ids=["non-ISO8601", "ISO8601"],
)
def test_empty_string_datetime(errors, args, format):
    # 参数化测试函数 test_empty_string_datetime，验证对空字符串的日期时间转换行为是否符合预期
    # GH13044, GH50251
    td = Series(args)

    # coerce empty string to pd.NaT
    result = to_datetime(td, format=format, errors=errors)
    expected = Series(["2016-03-24", "2016-03-25", NaT], dtype="datetime64[s]")
    tm.assert_series_equal(expected, result)


def test_empty_string_datetime_coerce__unit():
    # 测试函数 test_empty_string_datetime_coerce__unit，验证空字符串转换为 NaT 的行为是否符合预期
    # GH13044
    # coerce empty string to pd.NaT
    # 将输入的列表转换为日期时间索引对象，单位为秒，遇到错误时处理策略为"coerce"
    result = to_datetime([1, ""], unit="s", errors="coerce")
    # 预期的日期时间索引对象，包含指定日期时间和一个NaT（Not a Time）表示的缺失值，数据类型为"datetime64[ns]"
    expected = DatetimeIndex(["1970-01-01 00:00:01", "NaT"], dtype="datetime64[ns]")
    # 使用测试框架的方法验证预期输出和实际输出是否相等
    tm.assert_index_equal(expected, result)
    
    # 验证当设置错误处理策略为'raise'时，即使遇到错误也不会引发异常
    result = to_datetime([1, ""], unit="s", errors="raise")
    # 再次验证预期的日期时间索引对象
    tm.assert_index_equal(expected, result)
# 定义测试函数，测试将缓存应用于时间转换函数 to_datetime，确保结果具有单调递增的索引
def test_to_datetime_monotonic_increasing_index(cache):
    # GH28238
    # 设置开始缓存的时间点
    cstart = start_caching_at
    # 创建一个日期范围，从1980年开始，包含 cstart 个时间点，每年一次，年初
    times = date_range(Timestamp("1980"), periods=cstart, freq="YS")
    # 将日期时间索引转换为 DataFrame 格式，不包含索引，列名为 "DT"，并从中随机抽样 cstart 个时间点
    times = times.to_frame(index=False, name="DT").sample(n=cstart, random_state=1)
    # 将索引转换为浮点数，并按千分之一缩放，用于后续时间转换
    times.index = times.index.to_series().astype(float) / 1000
    # 对 times DataFrame 的第一列应用 to_datetime 函数，使用缓存
    result = to_datetime(times.iloc[:, 0], cache=cache)
    # 期望的结果是 times 的第一列
    expected = times.iloc[:, 0]
    # 断言 result 与 expected 相等
    tm.assert_series_equal(result, expected)


# 使用参数化测试，测试带有缓存和 coerce 错误处理的 to_datetime 函数
@pytest.mark.parametrize(
    "series_length",
    [40, start_caching_at, (start_caching_at + 1), (start_caching_at + 5)],
)
def test_to_datetime_cache_coerce_50_lines_outofbounds(series_length):
    # GH#45319
    # 创建一个包含多个日期时间的 Series，包括一个超出边界的日期
    ser = Series(
        [datetime.fromisoformat("1446-04-12 00:00:00+00:00")]
        + ([datetime.fromisoformat("1991-10-20 00:00:00+00:00")] * series_length),
        dtype=object,
    )
    # 使用 errors="coerce" 和 utc=True 参数，将 ser 转换为日期时间格式
    result1 = to_datetime(ser, errors="coerce", utc=True)

    # 期望的结果是将 ser 中的每个元素转换为 Timestamp 类型
    expected1 = Series([Timestamp(x) for x in ser])
    # 断言 expected1 的数据类型是 "M8[us, UTC]"
    assert expected1.dtype == "M8[us, UTC]"
    # 断言 result1 与 expected1 相等
    tm.assert_series_equal(result1, expected1)

    # 再次使用 to_datetime，使用 errors="raise" 和 utc=True 参数
    result3 = to_datetime(ser, errors="raise", utc=True)
    # 断言 result3 与 expected1 相等
    tm.assert_series_equal(result3, expected1)


# 测试带有格式化字符串解析纳秒的 to_datetime 函数
def test_to_datetime_format_f_parse_nanos():
    # GH 48767
    # 定义一个包含纳秒的时间戳和其对应格式的字符串
    timestamp = "15/02/2020 02:03:04.123456789"
    timestamp_format = "%d/%m/%Y %H:%M:%S.%f"
    # 使用指定的格式将 timestamp 转换为时间戳格式
    result = to_datetime(timestamp, format=timestamp_format)
    # 期望的结果是特定日期时间的 Timestamp 对象，包括纳秒部分
    expected = Timestamp(
        year=2020,
        month=2,
        day=15,
        hour=2,
        minute=3,
        second=4,
        microsecond=123456,
        nanosecond=789,
    )
    # 断言 result 与 expected 相等
    assert result == expected


# 测试混合使用 ISO8601 格式的 to_datetime 函数
def test_to_datetime_mixed_iso8601():
    # https://github.com/pandas-dev/pandas/issues/50411
    # 将包含多个日期字符串的列表转换为 DatetimeIndex，使用 ISO8601 格式
    result = to_datetime(["2020-01-01", "2020-01-01 05:00:00"], format="ISO8601")
    # 期望的结果是包含指定日期时间的 DatetimeIndex
    expected = DatetimeIndex(["2020-01-01 00:00:00", "2020-01-01 05:00:00"])
    # 断言 result 与 expected 相等
    tm.assert_index_equal(result, expected)


# 测试混合使用其他格式的 to_datetime 函数
def test_to_datetime_mixed_other():
    # https://github.com/pandas-dev/pandas/issues/50411
    # 将包含多个日期字符串的列表转换为 DatetimeIndex，使用混合格式
    result = to_datetime(["01/11/2000", "12 January 2000"], format="mixed")
    # 期望的结果是包含指定日期时间的 DatetimeIndex
    expected = DatetimeIndex(["2000-01-11", "2000-01-12"])
    # 断言 result 与 expected 相等
    tm.assert_index_equal(result, expected)


# 使用参数化测试，测试混合使用 ISO8601 或其他格式时的 to_datetime 函数，确保 'exact' 参数不适用于 'mixed' 或 'ISO8601' 格式
@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("format", ["ISO8601", "mixed"])
def test_to_datetime_mixed_or_iso_exact(exact, format):
    msg = "Cannot use 'exact' when 'format' is 'mixed' or 'ISO8601'"
    # 断言当 'exact' 参数与 'mixed' 或 'ISO8601' 格式一起使用时会引发 ValueError
    with pytest.raises(ValueError, match=msg):
        to_datetime(["2020-01-01"], exact=exact, format=format)


# 测试混合使用的 to_datetime 函数，当不是必须的 ISO8601 格式时，应该引发 ValueError
def test_to_datetime_mixed_not_necessarily_iso8601_raise():
    # https://github.com/pandas-dev/pandas/issues/50411
    # 断言当尝试将非 ISO8601 格式的日期转换时，会引发 ValueError，并指出具体位置
    with pytest.raises(
        ValueError, match="Time data 01-01-2000 is not ISO8601 format, at position 1"
    ):
        to_datetime(["2020-01-01", "01-01-2000"], format="ISO8601")


# 测试混合使用的 to_datetime 函数，当不是必须的 ISO8601 格式时，应该进行强制转换
def test_to_datetime_mixed_not_necessarily_iso8601_coerce():
    # https://github.com/pandas-dev/pandas/issues/50411
    # 调用 to_datetime 函数将日期字符串列表转换为日期时间对象
    result = to_datetime(
        ["2020-01-01", "01-01-2000"], format="ISO8601", errors="coerce"
    )
    # 使用 assert_index_equal 函数验证 result 是否与预期的日期时间索引一致
    tm.assert_index_equal(result, DatetimeIndex(["2020-01-01 00:00:00", NaT]))
def test_unknown_tz_raises():
    # 引发异常的测试用例，测试处理未知时区的情况
    dtstr = "2014 Jan 9 05:15 FAKE"
    msg = '.*un-recognized timezone "FAKE".'
    # 使用 pytest 检查是否引发 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        Timestamp(dtstr)

    with pytest.raises(ValueError, match=msg):
        to_datetime(dtstr)
    with pytest.raises(ValueError, match=msg):
        to_datetime([dtstr])


def test_from_numeric_arrow_dtype(any_numeric_ea_dtype):
    # 测试处理特定的 pyarrow 数据类型转换为 datetime 的功能
    # 先检查是否能导入 pyarrow 模块
    pytest.importorskip("pyarrow")
    ser = Series([1, 2], dtype=f"{any_numeric_ea_dtype.lower()}[pyarrow]")
    result = to_datetime(ser)
    expected = Series([1, 2], dtype="datetime64[ns]")
    # 使用 pandas 的 assert_series_equal 检查结果是否符合预期
    tm.assert_series_equal(result, expected)


def test_to_datetime_with_empty_str_utc_false_format_mixed():
    # 测试处理包含空字符串的 to_datetime 转换，同时混合不同的时间格式
    vals = ["2020-01-01 00:00+00:00", ""]
    result = to_datetime(vals, format="mixed")
    expected = Index([Timestamp("2020-01-01 00:00+00:00"), "NaT"], dtype="M8[s, UTC]")
    # 使用 pandas 的 assert_index_equal 检查索引结果是否符合预期
    tm.assert_index_equal(result, expected)

    # 检查另外两种情况是否与预期结果相同
    alt = to_datetime(vals)
    tm.assert_index_equal(alt, expected)
    alt2 = DatetimeIndex(vals)
    tm.assert_index_equal(alt2, expected)


def test_to_datetime_with_empty_str_utc_false_offsets_and_format_mixed():
    # 测试处理包含空字符串和混合不同格式的时区偏移情况
    # 预期会引发 ValueError 异常，提示检测到混合时区，建议传入 utc=True
    msg = "Mixed timezones detected. Pass utc=True in to_datetime"

    with pytest.raises(ValueError, match=msg):
        to_datetime(
            ["2020-01-01 00:00+00:00", "2020-01-01 00:00+02:00", ""], format="mixed"
        )


def test_to_datetime_mixed_tzs_mixed_types():
    # 测试处理混合不同时区和类型的情况
    ts = Timestamp("2016-01-02 03:04:05", tz="US/Pacific")
    dtstr = "2023-10-30 15:06+01"
    arr = [ts, dtstr]

    msg = (
        "Mixed timezones detected. Pass utc=True in to_datetime or tz='UTC' "
        "in DatetimeIndex to convert to a common timezone"
    )
    # 使用 pytest 检查是否引发 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        to_datetime(arr)
    with pytest.raises(ValueError, match=msg):
        to_datetime(arr, format="mixed")
    with pytest.raises(ValueError, match=msg):
        DatetimeIndex(arr)


def test_to_datetime_mixed_types_matching_tzs():
    # 测试处理包含混合类型但匹配时区的情况
    dtstr = "2023-11-01 09:22:03-07:00"
    ts = Timestamp(dtstr)
    arr = [ts, dtstr]
    # 执行多种方式的 to_datetime 转换，然后使用 assert_index_equal 检查结果是否符合预期
    res1 = to_datetime(arr)
    res2 = to_datetime(arr[::-1])[::-1]
    res3 = to_datetime(arr, format="mixed")
    res4 = DatetimeIndex(arr)

    expected = DatetimeIndex([ts, ts])
    tm.assert_index_equal(res1, expected)
    tm.assert_index_equal(res2, expected)
    tm.assert_index_equal(res3, expected)
    tm.assert_index_equal(res4, expected)


dtstr = "2020-01-01 00:00+00:00"
ts = Timestamp(dtstr)


@pytest.mark.filterwarnings("ignore:Could not infer format:UserWarning")
@pytest.mark.parametrize(
    "aware_val",
    [dtstr, Timestamp(dtstr)],
    ids=lambda x: type(x).__name__,
)
@pytest.mark.parametrize(
    "naive_val",
    [dtstr[:-6], ts.tz_localize(None), ts.date(), ts.asm8, ts.value, float(ts.value)],
    # 创建一个列表，包含以下元素：
    # 1. dtstr[:-6]: 使用切片操作去除日期字符串的最后六个字符
    # 2. ts.tz_localize(None): 去除时区信息的时间戳对象
    # 3. ts.date(): 获取时间戳的日期部分
    # 4. ts.asm8: 获取时间戳的 8 字节二进制表示
    # 5. ts.value: 获取时间戳的数值表示
    # 6. float(ts.value): 将时间戳数值表示转换为浮点数

    ids=lambda x: type(x).__name__,
    # 创建一个名为 ids 的关键字参数，其值是一个 lambda 函数，该函数接受参数 x，
    # 返回 x 对象的类型名称（类名），用于标识对象类型。
@pytest.mark.parametrize("naive_first", [True, False])
def test_to_datetime_mixed_awareness_mixed_types(aware_val, naive_val, naive_first):
    # 根据GH编号列出问题，这些问题涉及混合时区处理和错误提示
    # 空字符串解析为NaT（Not a Time）
    vals = [aware_val, naive_val, ""]

    vec = vals
    if naive_first:
        # 遗憾的是，行为依赖于顺序，因此我们以两种方式进行测试
        vec = [naive_val, aware_val, ""]

    # 判断是否同时为字符串类型，这些路径在_array_to_datetime_object中已经弃用并发出警告
    both_strs = isinstance(aware_val, str) and isinstance(naive_val, str)
    has_numeric = isinstance(naive_val, (int, float))
    both_datetime = isinstance(naive_val, datetime) and isinstance(aware_val, datetime)

    mixed_msg = (
        "Mixed timezones detected. Pass utc=True in to_datetime or tz='UTC' "
        "in DatetimeIndex to convert to a common timezone"
    )

    # 找到第一个非空值
    first_non_null = next(x for x in vec if x != "")
    # 如果第一个非空值不是字符串，则不通过_array_strptime进行日期时间格式猜测
    if not isinstance(first_non_null, str):
        # 这种情况会通过array_strptime，其行为不同
        msg = mixed_msg
        if naive_first and isinstance(aware_val, Timestamp):
            if isinstance(naive_val, Timestamp):
                msg = "Tz-aware datetime.datetime cannot be converted to datetime64"
            with pytest.raises(ValueError, match=msg):
                to_datetime(vec)
        else:
            if not naive_first and both_datetime:
                msg = "Cannot mix tz-aware with tz-naive values"
            with pytest.raises(ValueError, match=msg):
                to_datetime(vec)

        # 在utc=True的情况下不会发出警告或错误
        to_datetime(vec, utc=True)

    elif has_numeric and vec.index(aware_val) < vec.index(naive_val):
        msg = "time data .* doesn't match format"
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec)
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec, utc=True)

    elif both_strs and vec.index(aware_val) < vec.index(naive_val):
        msg = r"time data \"2020-01-01 00:00\" doesn't match format"
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec)
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec, utc=True)

    elif both_strs and vec.index(naive_val) < vec.index(aware_val):
        msg = "unconverted data remains when parsing with format"
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec)
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec, utc=True)

    else:
        msg = mixed_msg
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec)

        # 在utc=True的情况下不会发出警告或错误
        to_datetime(vec, utc=True)
    # 如果 both_strs 为真，执行以下操作
    if both_strs:
        # 将混合消息赋给变量 msg
        msg = mixed_msg
        # 使用 pytest 检查是否会引发 ValueError，并验证消息是否匹配
        with pytest.raises(ValueError, match=msg):
            # 调用 to_datetime 函数尝试转换 vec 到 datetime，使用 "mixed" 格式
            to_datetime(vec, format="mixed")
        # 使用 pytest 检查是否会引发 ValueError，并验证消息是否匹配
        with pytest.raises(ValueError, match=msg):
            # 创建 DatetimeIndex，验证消息是否匹配
            DatetimeIndex(vec)
    # 否则，如果 both_strs 不为真，执行以下操作
    else:
        # 将混合消息赋给变量 msg
        msg = mixed_msg
        # 如果 naive_first 为真且 aware_val 是 Timestamp 类型
        if naive_first and isinstance(aware_val, Timestamp):
            # 如果 naive_val 也是 Timestamp 类型，更新消息为特定内容
            if isinstance(naive_val, Timestamp):
                msg = "Tz-aware datetime.datetime cannot be converted to datetime64"
            # 使用 pytest 检查是否会引发 ValueError，并验证消息是否匹配
            with pytest.raises(ValueError, match=msg):
                # 调用 to_datetime 函数尝试转换 vec 到 datetime，使用 "mixed" 格式
                to_datetime(vec, format="mixed")
            # 使用 pytest 检查是否会引发 ValueError，并验证消息是否匹配
            with pytest.raises(ValueError, match=msg):
                # 创建 DatetimeIndex，验证消息是否匹配
                DatetimeIndex(vec)
        # 否则，如果 naive_first 不为真或者 both_datetime 为真，执行以下操作
        else:
            # 如果不是 naive_first 并且 both_datetime 为真，更新消息为特定内容
            if not naive_first and both_datetime:
                msg = "Cannot mix tz-aware with tz-naive values"
            # 使用 pytest 检查是否会引发 ValueError，并验证消息是否匹配
            with pytest.raises(ValueError, match=msg):
                # 调用 to_datetime 函数尝试转换 vec 到 datetime，使用 "mixed" 格式
                to_datetime(vec, format="mixed")
            # 使用 pytest 检查是否会引发 ValueError，并验证消息是否匹配
            with pytest.raises(ValueError, match=msg):
                # 创建 DatetimeIndex，验证消息是否匹配
                DatetimeIndex(vec)
```