# `D:\src\scipysrc\pandas\pandas\tests\tslibs\test_to_offset.py`

```
# 导入正则表达式模块
import re

# 导入 pytest 模块，用于编写和运行测试用例
import pytest

# 从 pandas._libs.tslibs 中导入时间差（Timedelta）、偏移量（offsets）、to_offset 函数
from pandas._libs.tslibs import (
    Timedelta,
    offsets,
    to_offset,
)

# 导入 pandas 测试工具模块
import pandas._testing as tm


# 使用 pytest.mark.parametrize 装饰器定义参数化测试用例，对 to_offset 函数进行测试
@pytest.mark.parametrize(
    "freq_input,expected",
    [
        (to_offset("10us"), offsets.Micro(10)),  # 测试微秒偏移量
        (offsets.Hour(), offsets.Hour()),  # 测试小时偏移量
        ("2h30min", offsets.Minute(150)),  # 测试小时和分钟偏移量
        ("2h 30min", offsets.Minute(150)),  # 测试带空格的小时和分钟偏移量
        ("2h30min15s", offsets.Second(150 * 60 + 15)),  # 测试小时、分钟和秒的偏移量
        ("2h 60min", offsets.Hour(3)),  # 测试超过60分钟的小时偏移量
        ("2h 20.5min", offsets.Second(8430)),  # 测试小时和带小数的分钟偏移量
        ("1.5min", offsets.Second(90)),  # 测试带小数的分钟偏移量
        ("0.5s", offsets.Milli(500)),  # 测试秒和毫秒偏移量
        ("15ms500us", offsets.Micro(15500)),  # 测试毫秒和微秒偏移量
        ("10s75ms", offsets.Milli(10075)),  # 测试秒和毫秒偏移量
        ("1s0.25ms", offsets.Micro(1000250)),  # 测试秒和带小数的毫秒偏移量
        ("2800ns", offsets.Nano(2800)),  # 测试纳秒偏移量
        ("2SME", offsets.SemiMonthEnd(2)),  # 测试半月末偏移量
        ("2SME-16", offsets.SemiMonthEnd(2, day_of_month=16)),  # 测试半月末偏移量并指定日期
        ("2SMS-14", offsets.SemiMonthBegin(2, day_of_month=14)),  # 测试半月初偏移量并指定日期
        ("2SMS-15", offsets.SemiMonthBegin(2)),  # 测试半月初偏移量
    ],
)
def test_to_offset(freq_input, expected):
    # 调用 to_offset 函数并验证返回结果是否符合预期
    result = to_offset(freq_input)
    assert result == expected


# 使用 pytest.mark.parametrize 装饰器定义参数化测试用例，对 to_offset 函数处理负数输入的情况进行测试
@pytest.mark.parametrize(
    "freqstr,expected",
    [("-1s", -1), ("-2SME", -2), ("-1SMS", -1), ("-5min10s", -310)],
)
def test_to_offset_negative(freqstr, expected):
    # 调用 to_offset 函数并验证返回结果的 n 属性是否符合预期
    result = to_offset(freqstr)
    assert result.n == expected


# 使用 pytest.mark.filterwarnings 装饰器忽略特定警告，再使用 pytest.mark.parametrize 装饰器定义参数化测试用例，对 to_offset 函数处理无效输入的情况进行测试
@pytest.mark.filterwarnings("ignore:.*'m' is deprecated.*:FutureWarning")
@pytest.mark.parametrize(
    "freqstr",
    [
        "2h20m",
        "us1",
        "-us",
        "3us1",
        "-2-3us",
        "-2D:3h",
        "1.5.0s",
        "2SMS-15-15",
        "2SMS-15D",
        "100foo",
        # 无效的前导符号
        "+-1d",
        "-+1h",
        "+1",
        "-7",
        "+d",
        "-m",
        # 无效的缩写锚点
        "SME-0",
        "SME-28",
        "SME-29",
        "SME-FOO",
        "BSM",
        "SME--1",
        "SMS-1",
        "SMS-28",
        "SMS-30",
        "SMS-BAR",
        "SMS-BYR",
        "BSMS",
        "SMS--2",
    ],
)
def test_to_offset_invalid(freqstr):
    # 查看 GitHub 问题编号 13930

    # 转义字符串，因为一些输入包含正则表达式特殊字符
    msg = re.escape(f"Invalid frequency: {freqstr}")
    # 调用 to_offset 函数，并断言会引发 ValueError 异常，异常消息匹配预期
    with pytest.raises(ValueError, match=msg):
        to_offset(freqstr)


# 对 to_offset 函数处理无效输入时不评估的情况进行测试
def test_to_offset_no_evaluate():
    msg = str(("", ""))
    # 调用 to_offset 函数，并断言会引发 TypeError 异常，异常消息匹配预期
    with pytest.raises(TypeError, match=msg):
        to_offset(("", ""))


# 对 to_offset 函数处理元组作为输入时的情况进行测试
def test_to_offset_tuple_unsupported():
    # 调用 to_offset 函数，并断言会引发 TypeError 异常，异常消息匹配预期
    with pytest.raises(TypeError, match="pass as a string instead"):
        to_offset((5, "T"))


# 使用 pytest.mark.parametrize 装饰器定义参数化测试用例，对 to_offset 函数处理带有空白字符的输入情况进行测试
@pytest.mark.parametrize(
    "freqstr,expected",
    [
        ("2D 3h", offsets.Hour(51)),  # 测试天和小时偏移量
        ("2 D3 h", offsets.Hour(51)),  # 测试带有空白的天和小时偏移量
        ("2 D 3 h", offsets.Hour(51)),  # 测试带有空白的天和小时偏移量
        ("  2 D 3 h  ", offsets.Hour(51)),  # 测试带有多余空白的天和小时偏移量
        ("   h    ", offsets.Hour()),  # 测试只有小时偏移量
        (" 3  h    ", offsets.Hour(3)),  # 测试带有多余空白的小时偏移量
    ],
)
def test_to_offset_whitespace(freqstr, expected):
    # 调用 to_offset 函数并验证返回结果是否符合预期
    result = to_offset(freqstr)
    assert result == expected
@pytest.mark.parametrize(
    "freqstr,expected", [("00h 00min 01s", 1), ("-00h 03min 14s", -194)]
)
def test_to_offset_leading_zero(freqstr, expected):
    # 调用 to_offset 函数，将频率字符串转换为偏移量对象
    result = to_offset(freqstr)
    # 断言结果的数值部分与预期值相等
    assert result.n == expected


@pytest.mark.parametrize("freqstr,expected", [("+1d", 1), ("+2h30min", 150)])
def test_to_offset_leading_plus(freqstr, expected):
    # 调用 to_offset 函数，将频率字符串转换为偏移量对象
    result = to_offset(freqstr)
    # 断言结果的数值部分与预期值相等
    assert result.n == expected


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"days": 1, "seconds": 1}, offsets.Second(86401)),
        ({"days": -1, "seconds": 1}, offsets.Second(-86399)),
        ({"hours": 1, "minutes": 10}, offsets.Minute(70)),
        ({"hours": 1, "minutes": -10}, offsets.Minute(50)),
        ({"weeks": 1}, offsets.Day(7)),
        ({"hours": 1}, offsets.Hour(1)),
        ({"hours": 1}, to_offset("60min")),  # 调用 to_offset 函数，将字符串转换为偏移量对象
        ({"microseconds": 1}, offsets.Micro(1)),
        ({"microseconds": 0}, offsets.Nano(0)),
    ],
)
def test_to_offset_pd_timedelta(kwargs, expected):
    # see gh-9064
    # 根据传入的关键字参数创建 Timedelta 对象
    td = Timedelta(**kwargs)
    # 调用 to_offset 函数，将 Timedelta 对象转换为偏移量对象
    result = to_offset(td)
    # 断言转换后的结果与预期的偏移量对象相等
    assert result == expected


@pytest.mark.parametrize(
    "shortcut,expected",
    [
        ("W", offsets.Week(weekday=6)),
        ("W-SUN", offsets.Week(weekday=6)),
        ("QE", offsets.QuarterEnd(startingMonth=12)),
        ("QE-DEC", offsets.QuarterEnd(startingMonth=12)),
        ("QE-MAY", offsets.QuarterEnd(startingMonth=5)),
        ("SME", offsets.SemiMonthEnd(day_of_month=15)),
        ("SME-15", offsets.SemiMonthEnd(day_of_month=15)),
        ("SME-1", offsets.SemiMonthEnd(day_of_month=1)),
        ("SME-27", offsets.SemiMonthEnd(day_of_month=27)),
        ("SMS-2", offsets.SemiMonthBegin(day_of_month=2)),
        ("SMS-27", offsets.SemiMonthBegin(day_of_month=27)),
    ],
)
def test_anchored_shortcuts(shortcut, expected):
    # 调用 to_offset 函数，将简写字符串转换为对应的偏移量对象
    result = to_offset(shortcut)
    # 断言转换后的结果与预期的偏移量对象相等
    assert result == expected


def test_to_offset_lowercase_frequency_w_deprecated():
    # GH#54939
    # 设置警告消息内容
    msg = "'w' is deprecated and will be removed in a future version"

    # 测试确保在使用 'w' 作为频率字符串时会产生 FutureWarning 警告
    with tm.assert_produces_warning(FutureWarning, match=msg):
        to_offset("2w")


@pytest.mark.parametrize(
    "freq_depr",
    [
        "2ye-mar",
        "2ys",
        "2qe",
        "2qs-feb",
        "2bqs",
        "2sms",
        "1sme",
        "2bms",
        "2cbme",
        "2me",
    ],
)
def test_to_offset_lowercase_frequency_raises(freq_depr):
    # 设置错误消息，显示所测试的频率字符串是无效的
    msg = f"Invalid frequency: {freq_depr}"

    # 测试确保当使用无效的频率字符串时会引发 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        to_offset(freq_depr)


@pytest.mark.parametrize(
    "freq_depr",
    [
        "2H",
        "2BH",
        "2MIN",
        "2S",
        "2Us",
        "2NS",
    ],
)
def test_to_offset_uppercase_frequency_deprecated(freq_depr):
    # GH#54939
    # 设置警告消息内容
    depr_msg = f"'{freq_depr[1:]}' is deprecated and will be removed in a " \
               f"future version, please use '{freq_depr.lower()[1:]}' instead."

    # 测试确保在使用大写字母频率字符串时会产生 FutureWarning 警告
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        to_offset(freq_depr)
```