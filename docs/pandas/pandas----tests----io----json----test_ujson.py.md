# `D:\src\scipysrc\pandas\pandas\tests\io\json\test_ujson.py`

```
import calendar  # 导入日历模块，提供日期相关的功能
import datetime  # 导入日期时间模块，提供处理日期时间的类和函数
import decimal  # 导入高精度十进制算术模块
import json  # 导入 JSON 编解码模块
import locale  # 导入与特定地域文化相关的数据格式化模块
import math  # 导入数学函数库
import re  # 导入正则表达式模块
import time  # 导入时间处理模块

import dateutil  # 导入日期和时间处理的工具扩展模块
import numpy as np  # 导入数值计算扩展库 NumPy
import pytest  # 导入 pytest 测试框架

import pandas._libs.json as ujson  # 导入 Pandas 内部的 JSON 加速库 ujson
from pandas.compat import IS64  # 从 Pandas 兼容模块中导入 IS64

from pandas import (  # 从 Pandas 库中导入以下对象
    DataFrame,  # 数据框架对象
    DatetimeIndex,  # 日期时间索引对象
    Index,  # 索引对象
    NaT,  # 表示缺失日期时间值的特殊对象
    PeriodIndex,  # 周期索引对象
    Series,  # 系列对象
    Timedelta,  # 表示时间差的对象
    Timestamp,  # 时间戳对象
    date_range,  # 创建日期范围的函数
)
import pandas._testing as tm  # 导入 Pandas 测试工具模块 tm


def _clean_dict(d):
    """
    Sanitize dictionary for JSON by converting all keys to strings.

    Parameters
    ----------
    d : dict
        The dictionary to convert.

    Returns
    -------
    cleaned_dict : dict
        Dictionary with all keys converted to strings.
    """
    return {str(k): v for k, v in d.items()}  # 返回将字典 d 中所有键转换为字符串的新字典


@pytest.fixture(
    params=[None, "split", "records", "values", "index"]  # 参数化装饰器，传入不同的值
)
def orient(request):
    """
    Fixture providing various orientations for testing.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Pytest request object to parameterize the fixture.

    Returns
    -------
    param : str or None
        Orientation mode to be tested.
    """
    return request.param  # 返回参数化的值作为测试的方向


class TestUltraJSONTests:
    @pytest.mark.skipif(not IS64, reason="not compliant on 32-bit, xref #15865")
    def test_encode_decimal(self):
        """
        Test encoding of decimal numbers to JSON and decoding back.

        Uses ujson library for encoding/decoding.

        Assertions are made to ensure correctness of encoding and decoding.

        This test also includes multiple cases for encoding with different
        precision settings.

        Notes
        -----
        This test is skipped on 32-bit platforms due to known issues.

        Returns
        -------
        None
        """
        sut = decimal.Decimal("1337.1337")  # 创建一个 Decimal 对象
        encoded = ujson.ujson_dumps(sut, double_precision=15)  # 使用 ujson 库将 Decimal 对象编码为 JSON 字符串
        decoded = ujson.ujson_loads(encoded)  # 使用 ujson 库将 JSON 字符串解码为 Python 对象
        assert decoded == 1337.1337  # 断言解码后的值与原始值相等

        sut = decimal.Decimal("0.95")  # 创建另一个 Decimal 对象
        encoded = ujson.ujson_dumps(sut, double_precision=1)  # 使用 ujson 库将 Decimal 对象编码为 JSON 字符串
        assert encoded == "1.0"  # 断言编码后的字符串值符合预期

        decoded = ujson.ujson_loads(encoded)  # 使用 ujson 库将 JSON 字符串解码为 Python 对象
        assert decoded == 1.0  # 断言解码后的值与预期的浮点数相等

        # 后续的代码块重复以上步骤，测试不同精度下的编码和解码

    @pytest.mark.parametrize("ensure_ascii", [True, False])
    def test_encode_string_conversion(self, ensure_ascii):
        # 测试字符串编码转换函数，参数 ensure_ascii 指定是否确保 ASCII 编码

        string_input = "A string \\ / \b \f \n \r \t </script> &"
        # 原始输入字符串，包含转义字符和特殊字符

        not_html_encoded = '"A string \\\\ \\/ \\b \\f \\n \\r \\t <\\/script> &"'
        # 非 HTML 编码的预期输出字符串

        html_encoded = (
            '"A string \\\\ \\/ \\b \\f \\n \\r \\t \\u003c\\/script\\u003e \\u0026"'
        )
        # HTML 编码的预期输出字符串，包含 Unicode 转义序列

        def helper(expected_output, **encode_kwargs):
            # 辅助函数，用于执行测试并验证输出是否符合预期
            output = ujson.ujson_dumps(
                string_input, ensure_ascii=ensure_ascii, **encode_kwargs
            )
            # 使用 ujson 库将输入字符串转换为 JSON 格式的字符串

            assert output == expected_output
            # 断言：转换后的输出与预期输出相等

            assert string_input == json.loads(output)
            # 断言：反序列化 JSON 输出后得到的字符串与原始输入相等

            assert string_input == ujson.ujson_loads(output)
            # 断言：使用 ujson 库解析 JSON 输出后得到的字符串与原始输入相等

        # 默认情况下，假定 encode_html_chars=False。
        helper(not_html_encoded)

        # 确保明确指定 encode_html_chars=False 时正常工作。
        helper(not_html_encoded, encode_html_chars=False)

        # 确保明确指定 encode_html_chars=True 时进行 HTML 编码。
        helper(html_encoded, encode_html_chars=True)
    # 测试双重转换编码的方法
    def test_encode_double_conversion(self, double_input):
        # 使用 ujson 库将输入的双精度浮点数转换为 JSON 格式的字符串
        output = ujson.ujson_dumps(double_input)
        # 断言：通过 round 函数保留五位小数后，ujson 转换的结果应与原始 JSON 加载的结果相等
        assert round(double_input, 5) == round(json.loads(output), 5)
        # 断言：同样保留五位小数后，ujson 转换的结果再加载应与原始输入相等
        assert round(double_input, 5) == round(ujson.ujson_loads(output), 5)

    # 测试包含十进制数的编码方法
    def test_encode_with_decimal(self):
        # 定义十进制输入为 1.0
        decimal_input = 1.0
        # 使用 ujson 库将十进制数转换为 JSON 格式的字符串
        output = ujson.ujson_dumps(decimal_input)

        # 断言：输出字符串应为 "1.0"
        assert output == "1.0"

    # 测试嵌套数组的编码方法
    def test_encode_array_of_nested_arrays(self):
        # 定义嵌套数组输入为 [[[[]]]] 的 20 个副本
        nested_input = [[[[]]]] * 20
        # 使用 ujson 库将嵌套数组转换为 JSON 格式的字符串
        output = ujson.ujson_dumps(nested_input)

        # 断言：ujson 转换的结果再加载应与原始输入相等
        assert nested_input == json.loads(output)
        # 断言：同样，ujson 转换的结果再加载应与原始输入相等
        assert nested_input == ujson.ujson_loads(output)

    # 测试双精度浮点数数组的编码方法
    def test_encode_array_of_doubles(self):
        # 定义双精度浮点数数组输入
        doubles_input = [31337.31337, 31337.31337, 31337.31337, 31337.31337] * 10
        # 使用 ujson 库将数组转换为 JSON 格式的字符串
        output = ujson.ujson_dumps(doubles_input)

        # 断言：ujson 转换的结果再加载应与原始输入相等
        assert doubles_input == json.loads(output)
        # 断言：同样，ujson 转换的结果再加载应与原始输入相等
        assert doubles_input == ujson.ujson_loads(output)

    # 测试双精度浮点数精度的编码方法
    def test_double_precision(self):
        # 定义双精度浮点数输入为 30.012345678901234
        double_input = 30.012345678901234
        # 使用 ujson 库将双精度浮点数转换为 JSON 格式的字符串，设置精度为 15
        output = ujson.ujson_dumps(double_input, double_precision=15)

        # 断言：ujson 转换的结果再加载应与原始输入相等
        assert double_input == json.loads(output)
        # 断言：同样，ujson 转换的结果再加载应与原始输入相等
        assert double_input == ujson.ujson_loads(output)

        # 遍历不同精度下的测试
        for double_precision in (3, 9):
            # 使用不同精度进行 ujson 转换
            output = ujson.ujson_dumps(double_input, double_precision=double_precision)
            # 对输入值进行指定精度的四舍五入
            rounded_input = round(double_input, double_precision)

            # 断言：ujson 转换的结果再加载应与指定精度下的四舍五入值相等
            assert rounded_input == json.loads(output)
            # 断言：同样，ujson 转换的结果再加载应与指定精度下的四舍五入值相等
            assert rounded_input == ujson.ujson_loads(output)

    # 使用 pytest 参数化标记来测试无效双精度浮点数精度的处理方法
    @pytest.mark.parametrize(
        "invalid_val",
        [
            20,
            -1,
            "9",
            None,
        ],
    )
    def test_invalid_double_precision(self, invalid_val):
        # 定义双精度浮点数输入
        double_input = 30.12345678901234567890
        # 预期抛出的异常类型取决于无效值的类型
        expected_exception = ValueError if isinstance(invalid_val, int) else TypeError
        # 预期的异常消息，匹配正则表达式，用于检测不同错误情况
        msg = (
            r"Invalid value '.*' for option 'double_precision', max is '15'|"
            r"an integer is required \(got type |"
            r"object cannot be interpreted as an integer"
        )
        # 使用 pytest 的 raises 断言来验证是否抛出预期的异常类型和消息
        with pytest.raises(expected_exception, match=msg):
            ujson.ujson_dumps(double_input, double_precision=invalid_val)

    # 测试字符串转换编码方法
    def test_encode_string_conversion2(self):
        # 定义字符串输入，包含特殊字符
        string_input = "A string \\ / \b \f \n \r \t"
        # 使用 ujson 库将字符串转换为 JSON 格式的字符串
        output = ujson.ujson_dumps(string_input)

        # 断言：ujson 转换的结果再加载应与原始输入相等
        assert string_input == json.loads(output)
        # 断言：同样，ujson 转换的结果再加载应与原始输入相等
        assert string_input == ujson.ujson_loads(output)
        # 断言：ujson 转换的结果应符合 JSON 字符串的格式
        assert output == '"A string \\\\ \\/ \\b \\f \\n \\r \\t"'

    # 使用 pytest 参数化标记来测试 Unicode 编码的方法
    @pytest.mark.parametrize(
        "unicode_input",
        ["Räksmörgås اسامة بن محمد بن عوض بن لادن", "\xe6\x97\xa5\xd1\x88"],
    )
    def test_encode_unicode_conversion(self, unicode_input):
        # 使用 ujson 库将 Unicode 输入转换为 JSON 格式的字符串
        enc = ujson.ujson_dumps(unicode_input)
        # 使用 ujson 库将 JSON 格式的字符串转换回 Unicode 格式
        dec = ujson.ujson_loads(enc)

        # 断言：ujson 转换的结果应与 json.dumps 的结果相等
        assert enc == json.dumps(unicode_input)
        # 断言：ujson 转换再加载的结果应与原始输入相等
        assert dec == json.loads(enc)
    # 测试处理转义字符的编码功能
    def test_encode_control_escaping(self):
        # 定义包含转义字符的输入字符串
        escaped_input = "\x19"
        # 使用 ujson 库将输入字符串转换为 JSON 格式的字符串
        enc = ujson.ujson_dumps(escaped_input)
        # 使用 ujson 库将 JSON 格式的字符串转换回 Python 对象
        dec = ujson.ujson_loads(enc)

        # 断言：转换前后的字符串应相等
        assert escaped_input == dec
        # 断言：ujson 和标准库 json 应生成相同的 JSON 字符串
        assert enc == json.dumps(escaped_input)

    # 测试处理 Unicode 代理对的编码功能
    def test_encode_unicode_surrogate_pair(self):
        # 定义包含 Unicode 代理对的输入字符串
        surrogate_input = "\xf0\x90\x8d\x86"
        # 使用 ujson 库将输入字符串转换为 JSON 格式的字符串
        enc = ujson.ujson_dumps(surrogate_input)
        # 使用 ujson 库将 JSON 格式的字符串转换回 Python 对象
        dec = ujson.ujson_loads(enc)

        # 断言：ujson 应生成与标准库 json 相同的 JSON 字符串
        assert enc == json.dumps(surrogate_input)
        # 断言：ujson 应正确将 JSON 字符串转换为 Python 对象
        assert dec == json.loads(enc)

    # 测试处理包含 4 字节 UTF-8 字符的编码功能
    def test_encode_unicode_4bytes_utf8(self):
        # 定义包含 4 字节 UTF-8 字符的输入字符串
        four_bytes_input = "\xf0\x91\x80\xb0TRAILINGNORMAL"
        # 使用 ujson 库将输入字符串转换为 JSON 格式的字符串
        enc = ujson.ujson_dumps(four_bytes_input)
        # 使用 ujson 库将 JSON 格式的字符串转换回 Python 对象
        dec = ujson.ujson_loads(enc)

        # 断言：ujson 应生成与标准库 json 相同的 JSON 字符串
        assert enc == json.dumps(four_bytes_input)
        # 断言：ujson 应正确将 JSON 字符串转换为 Python 对象
        assert dec == json.loads(enc)

    # 测试处理包含最高范围 4 字节 UTF-8 字符的编码功能
    def test_encode_unicode_4bytes_utf8highest(self):
        # 定义包含最高范围 4 字节 UTF-8 字符的输入字符串
        four_bytes_input = "\xf3\xbf\xbf\xbfTRAILINGNORMAL"
        # 使用 ujson 库将输入字符串转换为 JSON 格式的字符串
        enc = ujson.ujson_dumps(four_bytes_input)
        # 使用 ujson 库将 JSON 格式的字符串转换回 Python 对象
        dec = ujson.ujson_loads(enc)

        # 断言：ujson 应生成与标准库 json 相同的 JSON 字符串
        assert enc == json.dumps(four_bytes_input)
        # 断言：ujson 应正确将 JSON 字符串转换为 Python 对象
        assert dec == json.loads(enc)

    # 测试处理 Unicode 编码错误的情况
    def test_encode_unicode_error(self):
        # 定义包含 Unicode 错误的输入字符串
        string = "'\udac0'"
        # 定义 Unicode 编码错误的异常消息
        msg = (
            r"'utf-8' codec can't encode character '\\udac0' "
            r"in position 1: surrogates not allowed"
        )
        # 使用 pytest 检查 ujson 库是否能正确抛出 Unicode 编码错误
        with pytest.raises(UnicodeEncodeError, match=msg):
            ujson.ujson_dumps([string])

    # 测试处理嵌套数组的编码功能
    def test_encode_array_in_array(self):
        # 定义嵌套数组作为输入
        arr_in_arr_input = [[[[]]]]
        # 使用 ujson 库将输入对象转换为 JSON 格式的字符串
        output = ujson.ujson_dumps(arr_in_arr_input)

        # 断言：ujson 应生成与标准库 json 相同的 JSON 字符串
        assert arr_in_arr_input == json.loads(output)
        # 断言：ujson 应生成与标准库 json 相同的 JSON 字符串
        assert output == json.dumps(arr_in_arr_input)
        # 断言：ujson 应正确将 JSON 字符串转换为 Python 对象
        assert arr_in_arr_input == ujson.ujson_loads(output)

    # 使用参数化测试处理数值类型的编码功能
    @pytest.mark.parametrize(
        "num_input",
        [
            31337,
            -31337,  # 负数。
            -9223372036854775808,  # 较大的负数。
        ],
    )
    def test_encode_num_conversion(self, num_input):
        # 使用 ujson 库将数值转换为 JSON 格式的字符串
        output = ujson.ujson_dumps(num_input)
        # 断言：ujson 应正确将 JSON 字符串转换为数值
        assert num_input == json.loads(output)
        # 断言：ujson 应生成与标准库 json 相同的 JSON 字符串
        assert output == json.dumps(num_input)
        # 断言：ujson 应正确将 JSON 字符串转换为数值
        assert num_input == ujson.ujson_loads(output)

    # 测试处理列表类型的编码功能
    def test_encode_list_conversion(self):
        # 定义列表作为输入
        list_input = [1, 2, 3, 4]
        # 使用 ujson 库将输入对象转换为 JSON 格式的字符串
        output = ujson.ujson_dumps(list_input)

        # 断言：ujson 应正确将 JSON 字符串转换为列表
        assert list_input == json.loads(output)
        # 断言：ujson 应正确将 JSON 字符串转换为列表
        assert list_input == ujson.ujson_loads(output)

    # 测试处理字典类型的编码功能
    def test_encode_dict_conversion(self):
        # 定义字典作为输入
        dict_input = {"k1": 1, "k2": 2, "k3": 3, "k4": 4}
        # 使用 ujson 库将输入对象转换为 JSON 格式的字符串
        output = ujson.ujson_dumps(dict_input)

        # 断言：ujson 应正确将 JSON 字符串转换为字典
        assert dict_input == json.loads(output)
        # 断言：ujson 应正确将 JSON 字符串转换为字典
        assert dict_input == ujson.ujson_loads(output)

    # 使用参数化测试处理内置值的编码功能
    @pytest.mark.parametrize("builtin_value", [None, True, False])
    def test_encode_builtin_values_conversion(self, builtin_value):
        # 使用 ujson 库将内置值转换为 JSON 格式的字符串
        output = ujson.ujson_dumps(builtin_value)
        # 断言：ujson 应正确将 JSON 字符串转换为相应的内置值
        assert builtin_value == json.loads(output)
        # 断言：ujson 应生成与标准库 json 相同的 JSON 字符串
        assert output == json.dumps(builtin_value)
        # 断言：ujson 应正确将 JSON 字符串转换为相应的内置值
        assert builtin_value == ujson.ujson_loads(output)
    # 测试日期时间转换编码功能
    def test_encode_datetime_conversion(self):
        # 获取当前时间戳并转换为 datetime 对象
        datetime_input = datetime.datetime.fromtimestamp(time.time())
        # 使用 ujson 序列化 datetime 对象为 JSON 字符串，单位为秒
        output = ujson.ujson_dumps(datetime_input, date_unit="s")
        # 计算 datetime 对象的 UTC 时间戳
        expected = calendar.timegm(datetime_input.utctimetuple())

        # 断言 JSON 输出的整数值等于预期的时间戳
        assert int(expected) == json.loads(output)
        # 断言从 JSON 字符串加载的整数值等于预期的时间戳
        assert int(expected) == ujson.ujson_loads(output)

    # 测试日期转换编码功能
    def test_encode_date_conversion(self):
        # 获取当前时间戳并转换为 date 对象
        date_input = datetime.date.fromtimestamp(time.time())
        # 使用 ujson 序列化 date 对象为 JSON 字符串，单位为秒
        output = ujson.ujson_dumps(date_input, date_unit="s")

        # 构建 date 对象对应的时间元组
        tup = (date_input.year, date_input.month, date_input.day, 0, 0, 0)
        # 计算时间元组的 UTC 时间戳
        expected = calendar.timegm(tup)

        # 断言 JSON 输出的整数值等于预期的时间戳
        assert int(expected) == json.loads(output)
        # 断言从 JSON 字符串加载的整数值等于预期的时间戳
        assert int(expected) == ujson.ujson_loads(output)

    # 使用参数化测试多个时间转换编码功能
    @pytest.mark.parametrize(
        "test",
        [datetime.time(), datetime.time(1, 2, 3), datetime.time(10, 12, 15, 343243)],
    )
    def test_encode_time_conversion_basic(self, test):
        # 使用 ujson 序列化 time 对象为 JSON 字符串
        output = ujson.ujson_dumps(test)
        # 生成 ISO 格式的时间字符串作为预期输出
        expected = f'"{test.isoformat()}"'
        # 断言预期输出等于实际输出
        assert expected == output

    # 测试时区感知时间转换编码功能（使用 pytz 库）
    def test_encode_time_conversion_pytz(self):
        # 导入 pytz 库，如果未安装则跳过测试
        pytz = pytest.importorskip("pytz")
        # 创建具有时区信息的 time 对象
        test = datetime.time(10, 12, 15, 343243, pytz.utc)
        # 使用 ujson 序列化 time 对象为 JSON 字符串
        output = ujson.ujson_dumps(test)
        # 生成 ISO 格式的时间字符串作为预期输出
        expected = f'"{test.isoformat()}"'
        # 断言预期输出等于实际输出
        assert expected == output

    # 测试时区感知时间转换编码功能（使用 dateutil 库）
    def test_encode_time_conversion_dateutil(self):
        # 创建具有时区信息的 time 对象
        test = datetime.time(10, 12, 15, 343243, dateutil.tz.tzutc())
        # 使用 ujson 序列化 time 对象为 JSON 字符串
        output = ujson.ujson_dumps(test)
        # 生成 ISO 格式的时间字符串作为预期输出
        expected = f'"{test.isoformat()}"'
        # 断言预期输出等于实际输出
        assert expected == output

    # 使用参数化测试多个特殊日期时间值转换为 null 编码功能
    @pytest.mark.parametrize(
        "decoded_input", [NaT, np.datetime64("NaT"), np.nan, np.inf, -np.inf]
    )
    def test_encode_as_null(self, decoded_input):
        # 断言 ujson 序列化特殊值返回字符串 "null"
        assert ujson.ujson_dumps(decoded_input) == "null", "Expected null"

    # 测试 datetime 对象的不同单位编码功能
    def test_datetime_units(self):
        # 创建特定的 datetime 对象
        val = datetime.datetime(2013, 8, 17, 21, 17, 12, 215504)
        # 将 Timestamp 对象转换为纳秒单位的时间戳
        stamp = Timestamp(val).as_unit("ns")

        # 从 JSON 字符串加载并反序列化时间戳，单位为秒
        roundtrip = ujson.ujson_loads(ujson.ujson_dumps(val, date_unit="s"))
        # 断言反序列化后的时间戳等于预期的时间戳
        assert roundtrip == stamp._value // 10**9

        # 从 JSON 字符串加载并反序列化时间戳，单位为毫秒
        roundtrip = ujson.ujson_loads(ujson.ujson_dumps(val, date_unit="ms"))
        # 断言反序列化后的时间戳等于预期的时间戳
        assert roundtrip == stamp._value // 10**6

        # 从 JSON 字符串加载并反序列化时间戳，单位为微秒
        roundtrip = ujson.ujson_loads(ujson.ujson_dumps(val, date_unit="us"))
        # 断言反序列化后的时间戳等于预期的时间戳
        assert roundtrip == stamp._value // 10**3

        # 从 JSON 字符串加载并反序列化时间戳，单位为纳秒（默认单位）
        roundtrip = ujson.ujson_loads(ujson.ujson_dumps(val, date_unit="ns"))
        # 断言反序列化后的时间戳等于预期的时间戳
        assert roundtrip == stamp._value

        # 测试错误的单位参数时是否引发 ValueError 异常
        msg = "Invalid value 'foo' for option 'date_unit'"
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_dumps(val, date_unit="foo")
    # 定义一个测试方法，用于测试将字符串编码为 UTF-8 格式
    def test_encode_to_utf8(self):
        # 创建一个未编码的字符串，包含非ASCII字符
        unencoded = "\xe6\x97\xa5\xd1\x88"

        # 使用 ujson 库将未编码的字符串编码为 JSON 格式的字符串，确保不转义非ASCII字符
        enc = ujson.ujson_dumps(unencoded, ensure_ascii=False)
        # 使用 ujson 库将编码后的 JSON 格式字符串解码回 Python 对象
        dec = ujson.ujson_loads(enc)

        # 断言编码前后的结果应相同
        assert enc == json.dumps(unencoded, ensure_ascii=False)
        # 断言解码后的结果应与原始对象相同
        assert dec == json.loads(enc)

    # 定义一个测试方法，用于测试从 Unicode 字符串解码
    def test_decode_from_unicode(self):
        # 定义一个包含 Unicode 编码的 JSON 字符串
        unicode_input = '{"obj": 31337}'

        # 使用 ujson 库解析 Unicode 输入的 JSON 字符串为 Python 对象
        dec1 = ujson.ujson_loads(unicode_input)
        # 使用 ujson 库解析 Unicode 输入的 JSON 字符串为 Python 对象（使用 str 转换确保解析）
        dec2 = ujson.ujson_loads(str(unicode_input))

        # 断言两次解析结果应相同
        assert dec1 == dec2

    # 定义一个测试方法，用于测试编码时的递归深度限制
    def test_encode_recursion_max(self):
        # 设置最大递归深度为 8

        # 定义两个简单的类 O1 和 O2，构造一个递归结构
        class O2:
            member = 0

        class O1:
            member = 0

        decoded_input = O1()
        decoded_input.member = O2()
        decoded_input.member.member = decoded_input

        # 使用 ujson 库尝试对递归结构进行编码，应引发 OverflowError
        with pytest.raises(OverflowError, match="Maximum recursion level reached"):
            ujson.ujson_dumps(decoded_input)

    # 定义一个测试方法，用于测试解析无效 JSON 字符串
    def test_decode_jibberish(self):
        # 定义一个无效的字符串
        jibberish = "fdsa sda v9sa fdsa"
        # 预期解析时会抛出 ValueError 异常，异常信息应包含特定消息
        msg = "Unexpected character found when decoding 'false'"
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_loads(jibberish)

    # 使用 pytest 的参数化装饰器定义多组测试数据，测试解析损坏的 JSON 字符串
    @pytest.mark.parametrize(
        "broken_json",
        [
            "[",  # 损坏的数组开始符号
            "{",  # 损坏的对象开始符号
            "]",  # 损坏的数组结束符号
            "}",  # 损坏的对象结束符号
        ],
    )
    def test_decode_broken_json(self, broken_json):
        # 预期解析损坏的 JSON 字符串时会抛出 ValueError 异常，异常信息应包含特定消息
        msg = "Expected object or value"
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_loads(broken_json)

    # 使用 pytest 的参数化装饰器定义多组测试数据，测试解析深度过大的 JSON 字符串
    @pytest.mark.parametrize("too_big_char", ["[", "{"])
    def test_decode_depth_too_big(self, too_big_char):
        # 预期解析深度超出限制的 JSON 字符串时会抛出 ValueError 异常，异常信息应包含特定消息
        with pytest.raises(ValueError, match="Reached object decoding depth limit"):
            ujson.ujson_loads(too_big_char * (1024 * 1024))

    # 使用 pytest 的参数化装饰器定义多组测试数据，测试解析损坏的字符串
    @pytest.mark.parametrize(
        "bad_string",
        [
            '"TESTING',  # 未终止的字符串
            '"TESTING\\"',  # 未终止的转义
            "tru",  # 损坏的 True
            "fa",  # 损坏的 False
            "n",  # 损坏的 None
        ],
    )
    def test_decode_bad_string(self, bad_string):
        # 预期解析损坏的字符串时会抛出 ValueError 异常，异常信息应包含特定消息
        msg = (
            "Unexpected character found when decoding|"
            "Unmatched ''\"' when when decoding 'string'"
        )
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_loads(bad_string)

    # 使用 pytest 的参数化装饰器定义多组测试数据，测试解析损坏的 JSON 字符串
    @pytest.mark.parametrize(
        "broken_json, err_msg",
        [
            (
                '{{1337:""}}',
                "Key name of object must be 'string' when decoding 'object'",
            ),
            ('{{"key":"}', "Unmatched ''\"' when when decoding 'string'"),
            ("[[[true", "Unexpected character found when decoding array value (2)"),
        ],
    )
    def test_decode_broken_json_leak(self, broken_json, err_msg):
        # 循环多次，测试解析损坏的 JSON 字符串时是否会出现内存泄漏
        for _ in range(1000):
            # 预期解析损坏的 JSON 字符串时会抛出 ValueError 异常，异常信息应与预期的 err_msg 匹配
            with pytest.raises(ValueError, match=re.escape(err_msg)):
                ujson.ujson_loads(broken_json)
    @pytest.mark.parametrize(
        "invalid_dict",
        [
            "{{{{31337}}}}",  # Representing an invalid JSON object with no key.
            '{{{{"key":}}}}',  # Representing an invalid JSON object with no value.
            '{{{{"key"}}}}',  # Representing an invalid JSON object with no colon or value.
        ],
    )
    def test_decode_invalid_dict(self, invalid_dict):
        msg = (
            "Key name of object must be 'string' when decoding 'object'|"  # Error message for missing key.
            "No ':' found when decoding object value|"  # Error message for missing colon.
            "Expected object or value"  # Error message for expecting object or value.
        )
        # Asserting that ujson raises a ValueError with a message matching 'msg' when loading 'invalid_dict'.
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_loads(invalid_dict)

    @pytest.mark.parametrize("numeric_int_as_str", ["31337", "-31337"])
    def test_decode_numeric_int(self, numeric_int_as_str):
        # Asserting that converting 'numeric_int_as_str' to an integer equals loading it with ujson.
        assert int(numeric_int_as_str) == ujson.ujson_loads(numeric_int_as_str)

    def test_encode_null_character(self):
        wrapped_input = "31337 \x00 1337"
        output = ujson.ujson_dumps(wrapped_input)

        # Asserting that encoding and decoding null characters works correctly with ujson and json.
        assert wrapped_input == json.loads(output)
        assert output == json.dumps(wrapped_input)
        assert wrapped_input == ujson.ujson_loads(output)

        alone_input = "\x00"
        output = ujson.ujson_dumps(alone_input)

        # Asserting that encoding and decoding a standalone null character works correctly with ujson and json.
        assert alone_input == json.loads(output)
        assert output == json.dumps(alone_input)
        assert alone_input == ujson.ujson_loads(output)
        assert '"  \\u0000\\r\\n "' == ujson.ujson_dumps("  \u0000\r\n ")

    def test_decode_null_character(self):
        wrapped_input = '"31337 \\u0000 31337"'
        # Asserting that ujson correctly loads JSON with embedded null characters.
        assert ujson.ujson_loads(wrapped_input) == json.loads(wrapped_input)

    def test_encode_list_long_conversion(self):
        long_input = [
            9223372036854775807,
            9223372036854775807,
            9223372036854775807,
            9223372036854775807,
            9223372036854775807,
            9223372036854775807,
        ]
        output = ujson.ujson_dumps(long_input)

        # Asserting that encoding and decoding a list of long integers works correctly with ujson and json.
        assert long_input == json.loads(output)
        assert long_input == ujson.ujson_loads(output)

    @pytest.mark.parametrize("long_input", [9223372036854775807, 18446744073709551615])
    def test_encode_long_conversion(self, long_input):
        output = ujson.ujson_dumps(long_input)

        # Asserting that encoding and decoding a single long integer works correctly with ujson and json.
        assert long_input == json.loads(output)
        assert output == json.dumps(long_input)
        assert long_input == ujson.ujson_loads(output)

    @pytest.mark.parametrize("bigNum", [2**64, -(2**63) - 1])
    def test_dumps_ints_larger_than_maxsize(self, bigNum):
        encoding = ujson.ujson_dumps(bigNum)
        # Asserting that dumping and loading integers larger than the system max size works with ujson.
        assert str(bigNum) == encoding

        with pytest.raises(
            ValueError,
            match="Value is too big|Value is too small",
        ):
            # Asserting that ujson raises a ValueError when loading integers larger than the system max size.
            assert ujson.ujson_loads(encoding) == bigNum

    @pytest.mark.parametrize(
        "int_exp", ["1337E40", "1.337E40", "1337E+9", "1.337e+40", "1.337E-4"]
    )
    def test_decode_numeric_int_exp(self, int_exp):
        # Asserting that ujson correctly loads JSON numeric exponent notation.
        assert ujson.ujson_loads(int_exp) == json.loads(int_exp)
    # 当传入非字符串或字节对象而是None时，测试确保ujson.ujson_loads()引发TypeError异常，异常消息应为"a bytes-like object is required, not 'NoneType'"
    def test_loads_non_str_bytes_raises(self):
        msg = "a bytes-like object is required, not 'NoneType'"
        with pytest.raises(TypeError, match=msg):
            ujson.ujson_loads(None)

    # 使用pytest的参数化装饰器，为val提供多个测试值，验证在32位有符号整数的范围内（2**31 <= x < 2**32）解码数字的正确性
    @pytest.mark.parametrize("val", [3590016419, 2**31, 2**32, (2**32) - 1])
    def test_decode_number_with_32bit_sign_bit(self, val):
        # 测试确保在32位有符号整数的范围内（2**31 <= x < 2**32），能够正确解码数字。
        doc = f'{{"id": {val}}}'
        assert ujson.ujson_loads(doc)["id"] == val

    # 测试确保ujson.ujson_dumps()在处理大量字符转义时不会引发异常。
    def test_encode_big_escape(self):
        # 确保没有异常被引发。
        for _ in range(10):
            base = "\u00e5".encode()
            escape_input = base * 1024 * 1024 * 2
            ujson.ujson_dumps(escape_input)

    # 测试确保ujson.ujson_loads()在处理大量字符转义时不会引发异常。
    def test_decode_big_escape(self):
        # 确保没有异常被引发。
        for _ in range(10):
            base = "\u00e5".encode()
            quote = b'"'

            escape_input = quote + (base * 1024 * 1024 * 2) + quote
            ujson.ujson_loads(escape_input)

    # 测试确保将自定义对象转换为字典格式的正确性。
    def test_to_dict(self):
        d = {"key": 31337}

        # 定义一个带有toDict方法的类，返回预定义的字典d
        class DictTest:
            def toDict(self):
                return d

        # 创建类的实例
        o = DictTest()
        # 使用ujson.ujson_dumps()将对象o转换为JSON格式的字符串
        output = ujson.ujson_dumps(o)

        # 使用ujson.ujson_loads()将JSON格式的字符串解析为Python对象
        dec = ujson.ujson_loads(output)
        # 断言解析后的对象与预定义的字典d相等
        assert dec == d
    def test_default_handler(self):
        class _TestObject:
            def __init__(self, val) -> None:
                self.val = val

            @property
            def recursive_attr(self):
                return _TestObject("recursive_attr")

            def __str__(self) -> str:
                return str(self.val)

        msg = "Maximum recursion level reached"
        # 使用 pytest 断言捕获 OverflowError 异常，检查是否包含特定消息
        with pytest.raises(OverflowError, match=msg):
            ujson.ujson_dumps(_TestObject("foo"))
        # 使用 ujson 序列化 _TestObject 实例，验证输出与预期字符串相符
        assert '"foo"' == ujson.ujson_dumps(_TestObject("foo"), default_handler=str)

        # 定义自定义处理函数 my_handler，返回固定字符串 "foobar"
        def my_handler(_):
            return "foobar"

        # 使用 ujson 序列化 _TestObject 实例，应用自定义处理函数 my_handler
        assert '"foobar"' == ujson.ujson_dumps(
            _TestObject("foo"), default_handler=my_handler
        )

        # 定义会抛出 TypeError 异常的处理函数 my_handler_raises
        def my_handler_raises(_):
            raise TypeError("I raise for anything")

        # 使用 pytest 断言捕获 TypeError 异常，检查是否包含特定消息
        with pytest.raises(TypeError, match="I raise for anything"):
            ujson.ujson_dumps(_TestObject("foo"), default_handler=my_handler_raises)

        # 定义返回整数的处理函数 my_int_handler
        def my_int_handler(_):
            return 42

        # 使用 ujson 序列化 _TestObject 实例，应用返回整数的处理函数 my_int_handler
        assert (
            ujson.ujson_loads(
                ujson.ujson_dumps(_TestObject("foo"), default_handler=my_int_handler)
            )
            == 42
        )

        # 定义返回固定日期时间的处理函数 my_obj_handler
        def my_obj_handler(_):
            return datetime.datetime(2013, 2, 3)

        # 使用 ujson 序列化 datetime 对象，验证输出与预期日期时间相符
        assert ujson.ujson_loads(
            ujson.ujson_dumps(datetime.datetime(2013, 2, 3))
        ) == ujson.ujson_loads(
            ujson.ujson_dumps(_TestObject("foo"), default_handler=my_obj_handler)
        )

        # 创建 _TestObject 实例列表 obj_list
        obj_list = [_TestObject("foo"), _TestObject("bar")]
        # 使用 json 库序列化 _TestObject 实例列表，与使用 ujson 序列化后再反序列化的结果比较
        assert json.loads(json.dumps(obj_list, default=str)) == ujson.ujson_loads(
            ujson.ujson_dumps(obj_list, default_handler=str)
        )

    def test_encode_object(self):
        class _TestObject:
            def __init__(self, a, b, _c, d) -> None:
                self.a = a
                self.b = b
                self._c = _c
                self.d = d

            def e(self):
                return 5

        # 创建 _TestObject 实例 test_object
        test_object = _TestObject(a=1, b=2, _c=3, d=4)
        # 使用 ujson 序列化 _TestObject 实例，验证输出与预期字典相符
        assert ujson.ujson_loads(ujson.ujson_dumps(test_object)) == {
            "a": 1,
            "b": 2,
            "d": 4,
        }

    def test_ujson__name__(self):
        # 检查 ujson 模块的 __name__ 属性是否为 "pandas._libs.json"
        assert ujson.__name__ == "pandas._libs.json"
class TestNumpyJSONTests:
    # 使用 pytest 的参数化标记，参数为 bool_input 分别为 True 和 False
    @pytest.mark.parametrize("bool_input", [True, False])
    def test_bool(self, bool_input):
        # 将 bool_input 转换为布尔值
        b = bool(bool_input)
        # 断言将布尔值转换为 JSON 字符串再转回布尔值后应该与原始布尔值相等
        assert ujson.ujson_loads(ujson.ujson_dumps(b)) == b

    # 测试布尔数组的序列化和反序列化
    def test_bool_array(self):
        # 创建一个布尔类型的 NumPy 数组
        bool_array = np.array(
            [True, False, True, True, False, True, False, False], dtype=bool
        )
        # 使用 ujson 将数组转换为 JSON 字符串再转回数组
        output = np.array(ujson.ujson_loads(ujson.ujson_dumps(bool_array)), dtype=bool)
        # 断言反序列化后的数组与原始数组内容相等
        tm.assert_numpy_array_equal(bool_array, output)

    # 测试整数的序列化和反序列化
    def test_int(self, any_int_numpy_dtype):
        # 根据参数传入的 NumPy 数据类型创建一个整数实例
        klass = np.dtype(any_int_numpy_dtype).type
        num = klass(1)
        # 断言将整数转换为 JSON 字符串再转回整数后应该与原始整数相等
        assert klass(ujson.ujson_loads(ujson.ujson_dumps(num))) == num

    # 测试整数数组的序列化和反序列化
    def test_int_array(self, any_int_numpy_dtype):
        # 创建一个整数类型的 NumPy 数组
        arr = np.arange(100, dtype=int)
        # 将数组转换为指定的 NumPy 数据类型
        arr_input = arr.astype(any_int_numpy_dtype)

        # 使用 ujson 将数组转换为 JSON 字符串再转回数组
        arr_output = np.array(
            ujson.ujson_loads(ujson.ujson_dumps(arr_input)), dtype=any_int_numpy_dtype
        )
        # 断言反序列化后的数组与原始数组内容相等
        tm.assert_numpy_array_equal(arr_input, arr_output)

    # 测试最大整数值的序列化和反序列化
    def test_int_max(self, any_int_numpy_dtype):
        # 如果测试的整数类型是 64 位且当前平台不支持，则跳过测试
        if any_int_numpy_dtype in ("int64", "uint64") and not IS64:
            pytest.skip("Cannot test 64-bit integer on 32-bit platform")

        klass = np.dtype(any_int_numpy_dtype).type

        # 如果是 uint64 类型则设置为 int64 的最大值，因为它被编码为有符号数
        if any_int_numpy_dtype == "uint64":
            num = np.iinfo("int64").max
        else:
            num = np.iinfo(any_int_numpy_dtype).max

        # 断言反序列化后的整数与原始整数相等
        assert klass(ujson.ujson_loads(ujson.ujson_dumps(num))) == num

    # 测试浮点数的序列化和反序列化
    def test_float(self, float_numpy_dtype):
        # 根据参数传入的 NumPy 浮点数类型创建一个浮点数实例
        klass = np.dtype(float_numpy_dtype).type
        num = klass(256.2013)
        # 断言将浮点数转换为 JSON 字符串再转回浮点数后应该与原始浮点数相等
        assert klass(ujson.ujson_loads(ujson.ujson_dumps(num))) == num

    # 测试浮点数数组的序列化和反序列化
    def test_float_array(self, float_numpy_dtype):
        # 创建一个浮点数类型的 NumPy 数组
        arr = np.arange(12.5, 185.72, 1.7322, dtype=float)
        # 将数组转换为指定的 NumPy 浮点数类型
        float_input = arr.astype(float_numpy_dtype)

        # 使用 ujson 将数组转换为 JSON 字符串再转回数组
        float_output = np.array(
            ujson.ujson_loads(ujson.ujson_dumps(float_input, double_precision=15)),
            dtype=float_numpy_dtype,
        )
        # 断言反序列化后的数组与原始数组内容相等
        tm.assert_almost_equal(float_input, float_output)

    # 测试浮点数最大值的序列化和反序列化
    def test_float_max(self, float_numpy_dtype):
        # 根据参数传入的 NumPy 浮点数类型创建一个浮点数实例，使用其最大值的一部分
        klass = np.dtype(float_numpy_dtype).type
        num = klass(np.finfo(float_numpy_dtype).max / 10)

        # 断言反序列化后的浮点数与原始浮点数相等
        tm.assert_almost_equal(
            klass(ujson.ujson_loads(ujson.ujson_dumps(num, double_precision=15))), num
        )

    # 测试基本数组的序列化和反序列化
    def test_array_basic(self):
        # 创建一个多维的 NumPy 数组
        arr = np.arange(96)
        arr = arr.reshape((2, 2, 2, 2, 3, 2))

        # 使用 ujson 将数组转换为 JSON 字符串再转回数组
        tm.assert_numpy_array_equal(
            np.array(ujson.ujson_loads(ujson.ujson_dumps(arr))), arr
        )

    # 使用 pytest 的参数化标记，参数为 shape 分别为 (10, 10), (5, 5, 4), (100, 1)
    @pytest.mark.parametrize("shape", [(10, 10), (5, 5, 4), (100, 1)])
    def test_array_reshaped(self, shape):
        # 创建一个一维数组并根据参数 shape 转换为多维数组
        arr = np.arange(100)
        arr = arr.reshape(shape)

        # 使用 ujson 将数组转换为 JSON 字符串再转回数组
        tm.assert_numpy_array_equal(
            np.array(ujson.ujson_loads(ujson.ujson_dumps(arr))), arr
        )
    # 定义一个测试函数，用于测试数组转换和序列化
    def test_array_list(self):
        # 创建一个包含不同类型元素的列表
        arr_list = [
            "a",         # 字符串
            [],          # 空列表
            {},          # 空字典
            {},          # 另一个空字典
            [],          # 另一个空列表
            42,          # 整数
            97.8,        # 浮点数
            ["a", "b"],  # 包含字符串的列表
            {"key": "val"},  # 包含键值对的字典
        ]
        # 将列表转换为 NumPy 数组，数据类型为 object
        arr = np.array(arr_list, dtype=object)
        # 使用 ujson 序列化然后反序列化数组
        result = np.array(ujson.ujson_loads(ujson.ujson_dumps(arr)), dtype=object)
        # 断言反序列化后的数组与原数组相等
        tm.assert_numpy_array_equal(result, arr)

    # 定义测试函数，用于测试浮点数数组的序列化和反序列化
    def test_array_float(self):
        # 指定数组的数据类型为 np.float32
        dtype = np.float32

        # 创建一个浮点数数组，从 100.202 到 200.202，步长为 1
        arr = np.arange(100.202, 200.202, 1, dtype=dtype)
        # 将数组重新形状为 5x5x4 的三维数组
        arr = arr.reshape((5, 5, 4))

        # 使用 ujson 序列化然后反序列化数组
        arr_out = np.array(ujson.ujson_loads(ujson.ujson_dumps(arr)), dtype=dtype)
        # 断言反序列化后的数组与原数组几乎相等
        tm.assert_almost_equal(arr, arr_out)

    # 定义测试函数，用于测试0维数组的序列化
    def test_0d_array(self):
        # gh-18878
        # 准备异常消息的正则表达式模式
        msg = re.escape(
            "array(1) (numpy-scalar) is not JSON serializable at the moment"
        )
        # 断言当序列化0维数组时抛出特定类型错误，并匹配特定消息
        with pytest.raises(TypeError, match=msg):
            ujson.ujson_dumps(np.array(1))

    # 定义测试函数，用于测试长双精度浮点数数组的序列化
    def test_array_long_double(self):
        # 准备异常消息的正则表达式模式
        msg = re.compile(
            "1234.5.* \\(numpy-scalar\\) is not JSON serializable at the moment"
        )
        # 断言当序列化长双精度浮点数时抛出特定类型错误，并匹配特定消息
        with pytest.raises(TypeError, match=msg):
            ujson.ujson_dumps(np.longdouble(1234.5))
# 定义一个测试类 TestPandasJSONTests，用于测试与 Pandas 和 JSON 相关的功能
class TestPandasJSONTests:
    # 定义测试方法 test_dataframe，测试 DataFrame 的相关功能
    def test_dataframe(self, orient):
        # 定义数据类型为 np.int64
        dtype = np.int64
        
        # 创建一个 DataFrame 对象 df，包含两行三列的数据
        df = DataFrame(
            [[1, 2, 3], [4, 5, 6]],
            index=["a", "b"],
            columns=["x", "y", "z"],
            dtype=dtype,
        )
        
        # 根据 orient 参数设置编码参数，如果 orient 为 None，则设为空字典
        encode_kwargs = {} if orient is None else {"orient": orient}
        
        # 断言 df 中所有列的数据类型为 dtype
        assert (df.dtypes == dtype).all()
        
        # 使用 ujson 序列化和反序列化 df，得到 output
        output = ujson.ujson_loads(ujson.ujson_dumps(df, **encode_kwargs))
        
        # 再次断言 df 中所有列的数据类型为 dtype
        assert (df.dtypes == dtype).all()
        
        # 如果 orient 为 "split"，则对 output 进行清理后转换为 DataFrame
        if orient == "split":
            dec = _clean_dict(output)
            output = DataFrame(**dec)
        else:
            # 否则直接将 output 转换为 DataFrame
            output = DataFrame(output)
        
        # 根据 orient 的不同值进行 DataFrame 的修正
        if orient == "values":
            df.columns = [0, 1, 2]
            df.index = [0, 1]
        elif orient == "records":
            df.index = [0, 1]
        elif orient == "index":
            df = df.transpose()
        
        # 最后断言 df 中所有列的数据类型为 dtype，并比较 output 和 df 是否相等
        assert (df.dtypes == dtype).all()
        tm.assert_frame_equal(output, df)

    # 定义测试方法 test_dataframe_nested，测试包含嵌套 DataFrame 的功能
    def test_dataframe_nested(self, orient):
        # 创建一个 DataFrame 对象 df，包含两行三列的数据
        df = DataFrame(
            [[1, 2, 3], [4, 5, 6]], index=["a", "b"], columns=["x", "y", "z"]
        )
        
        # 创建一个嵌套字典 nested，包含两个键值对，值为相同的 df 对象及其副本
        nested = {"df1": df, "df2": df.copy()}
        
        # 根据 orient 参数设置编码参数，如果 orient 为 None，则设为空字典
        kwargs = {} if orient is None else {"orient": orient}
        
        # 使用 ujson 序列化和反序列化 nested，得到期望的字典 exp
        exp = {
            "df1": ujson.ujson_loads(ujson.ujson_dumps(df, **kwargs)),
            "df2": ujson.ujson_loads(ujson.ujson_dumps(df, **kwargs)),
        }
        
        # 断言序列化后的 nested 和期望的 exp 是否相等
        assert ujson.ujson_loads(ujson.ujson_dumps(nested, **kwargs)) == exp

    # 定义测试方法 test_series，测试 Series 的相关功能
    def test_series(self, orient):
        # 定义数据类型为 np.int64
        dtype = np.int64
        
        # 创建一个 Series 对象 s，包含六个元素
        s = Series(
            [10, 20, 30, 40, 50, 60],
            name="series",
            index=[6, 7, 8, 9, 10, 15],
            dtype=dtype,
        ).sort_values()
        
        # 断言 s 的数据类型为 dtype
        assert s.dtype == dtype
        
        # 根据 orient 参数设置编码参数，如果 orient 为 None，则设为空字典
        encode_kwargs = {} if orient is None else {"orient": orient}
        
        # 使用 ujson 序列化和反序列化 s，得到 output
        output = ujson.ujson_loads(ujson.ujson_dumps(s, **encode_kwargs))
        
        # 再次断言 s 的数据类型为 dtype
        assert s.dtype == dtype
        
        # 如果 orient 为 "split"，则对 output 进行清理后转换为 Series
        if orient == "split":
            dec = _clean_dict(output)
            output = Series(**dec)
        else:
            # 否则直接将 output 转换为 Series
            output = Series(output)
        
        # 根据 orient 的不同值进行 Series 的修正
        if orient in (None, "index"):
            s.name = None
            output = output.sort_values()
            s.index = ["6", "7", "8", "9", "10", "15"]
        elif orient in ("records", "values"):
            s.name = None
            s.index = [0, 1, 2, 3, 4, 5]
        
        # 最后断言 s 的数据类型为 dtype，并比较 output 和 s 是否相等
        assert s.dtype == dtype
        tm.assert_series_equal(output, s)
    # 测试嵌套的 Series 对象
    def test_series_nested(self, orient):
        # 创建一个 Series 对象，包含指定数据和索引，然后按值排序
        s = Series(
            [10, 20, 30, 40, 50, 60], name="series", index=[6, 7, 8, 9, 10, 15]
        ).sort_values()
        # 创建一个包含两个键值对的字典，每个值是 Series 对象及其副本
        nested = {"s1": s, "s2": s.copy()}
        # 如果 orient 不为 None，则创建一个包含 orient 参数的关键字参数字典；否则为空字典
        kwargs = {} if orient is None else {"orient": orient}

        # 创建一个期望结果的字典，使用 ujson 序列化和反序列化 s 对象
        exp = {
            "s1": ujson.ujson_loads(ujson.ujson_dumps(s, **kwargs)),
            "s2": ujson.ujson_loads(ujson.ujson_dumps(s, **kwargs)),
        }
        # 断言嵌套字典经过 ujson 序列化和反序列化后与期望结果相等
        assert ujson.ujson_loads(ujson.ujson_dumps(nested, **kwargs)) == exp

    # 测试 Index 对象
    def test_index(self):
        # 创建一个 Index 对象
        i = Index([23, 45, 18, 98, 43, 11], name="index")

        # 列索引化
        # 使用 ujson 序列化和反序列化 i，然后创建一个新的 Index 对象
        output = Index(ujson.ujson_loads(ujson.ujson_dumps(i)), name="index")
        # 断言原始 Index 对象与输出 Index 对象相等
        tm.assert_index_equal(i, output)

        # 使用 orient="split" 参数来序列化 i，然后清理返回的字典，并创建一个新的 Index 对象
        dec = _clean_dict(ujson.ujson_loads(ujson.ujson_dumps(i, orient="split")))
        output = Index(**dec)
        # 断言原始 Index 对象与输出 Index 对象相等
        tm.assert_index_equal(i, output)
        # 断言原始 Index 对象的名称与输出 Index 对象的名称相等
        assert i.name == output.name

        # 使用 orient="values" 参数来序列化 i，然后创建一个新的 Index 对象
        output = Index(
            ujson.ujson_loads(ujson.ujson_dumps(i, orient="values")), name="index"
        )
        # 断言原始 Index 对象与输出 Index 对象相等
        tm.assert_index_equal(i, output)

        # 使用 orient="records" 参数来序列化 i，然后创建一个新的 Index 对象
        output = Index(
            ujson.ujson_loads(ujson.ujson_dumps(i, orient="records")), name="index"
        )
        # 断言原始 Index 对象与输出 Index 对象相等
        tm.assert_index_equal(i, output)

        # 使用 orient="index" 参数来序列化 i，然后创建一个新的 Index 对象
        output = Index(
            ujson.ujson_loads(ujson.ujson_dumps(i, orient="index")), name="index"
        )
        # 断言原始 Index 对象与输出 Index 对象相等
        tm.assert_index_equal(i, output)

    # 测试 DatetimeIndex 对象
    def test_datetime_index(self):
        date_unit = "ns"

        # 创建一个 DatetimeIndex 对象
        rng = DatetimeIndex(list(date_range("1/1/2000", periods=20)), freq=None)
        # 使用 ujson 序列化和反序列化 rng，使用指定的日期单位
        encoded = ujson.ujson_dumps(rng, date_unit=date_unit)

        # 使用 ujson 反序列化 encoded 数据，创建一个新的 DatetimeIndex 对象
        decoded = DatetimeIndex(np.array(ujson.ujson_loads(encoded)))
        # 断言原始 DatetimeIndex 对象与解码后的 DatetimeIndex 对象相等
        tm.assert_index_equal(rng, decoded)

        # 创建一个 Series 对象，使用随机数填充，索引为 rng
        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
        # 使用 ujson 序列化和反序列化 ts，使用指定的日期单位
        decoded = Series(ujson.ujson_loads(ujson.ujson_dumps(ts, date_unit=date_unit)))

        # 获取解码后 Series 对象的索引值，并转换为 DatetimeIndex
        idx_values = decoded.index.values.astype(np.int64)
        decoded.index = DatetimeIndex(idx_values)
        # 断言原始 Series 对象与解码后的 Series 对象相等
        tm.assert_series_equal(ts, decoded)

    # 测试 ujson.ujson_loads 方法对无效数组的解码
    @pytest.mark.parametrize(
        "invalid_arr",
        [
            "[31337,]",  # 末尾有逗号
            "[,31337]",  # 开头有逗号
            "[]]",       # 括号不匹配
            "[,]",       # 只有逗号
        ],
    )
    def test_decode_invalid_array(self, invalid_arr):
        # 预期的错误消息
        msg = (
            "Expected object or value|Trailing data|"
            "Unexpected character found when decoding array value"
        )
        # 使用 pytest 断言，验证 ujson.ujson_loads 方法对无效数组的解码是否会抛出 ValueError，并且错误消息匹配预期
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_loads(invalid_arr)

    # 测试 ujson.ujson_loads 方法对有效数组的解码
    @pytest.mark.parametrize("arr", [[], [31337]])
    def test_decode_array(self, arr):
        # 断言经过 ujson 序列化和反序列化后，数组 arr 与其本身相等
        assert arr == ujson.ujson_loads(str(arr))

    # 测试极端数值的处理
    @pytest.mark.parametrize("extreme_num", [9223372036854775807, -9223372036854775808])
    # 测试极端数字是否能够正确解码
    def test_decode_extreme_numbers(self, extreme_num):
        assert extreme_num == ujson.ujson_loads(str(extreme_num))

    # 使用参数化测试，验证过大或过小的数字是否会触发 ValueError 异常
    @pytest.mark.parametrize("too_extreme_num", [f"{2**64}", f"{-2**63-1}"])
    def test_decode_too_extreme_numbers(self, too_extreme_num):
        with pytest.raises(
            ValueError,
            match="Value is too big|Value is too small",
        ):
            ujson.ujson_loads(too_extreme_num)

    # 测试带有尾部空白字符的 JSON 字符串是否能够正确解码为空字典
    def test_decode_with_trailing_whitespaces(self):
        assert {} == ujson.ujson_loads("{}\n\t ")

    # 使用参数化测试，验证带有非空白尾部字符的 JSON 字符串是否会触发 ValueError 异常
    def test_decode_with_trailing_non_whitespaces(self):
        with pytest.raises(ValueError, match="Trailing data"):
            ujson.ujson_loads("{}\n\t a")

    # 使用参数化测试，验证解码包含大整数的数组时是否会触发 ValueError 异常
    @pytest.mark.parametrize("value", [f"{2**64}", f"{-2**63-1}"])
    def test_decode_array_with_big_int(self, value):
        with pytest.raises(
            ValueError,
            match="Value is too big|Value is too small",
        ):
            ujson.ujson_loads(value)

    # 使用参数化测试，验证浮点数的 JSON 字符串是否能正确解码并且保持精度
    @pytest.mark.parametrize(
        "float_number",
        [
            1.1234567893,
            1.234567893,
            1.34567893,
            1.4567893,
            1.567893,
            1.67893,
            1.7893,
            1.893,
            1.3,
        ],
    )
    @pytest.mark.parametrize("sign", [-1, 1])
    def test_decode_floating_point(self, sign, float_number):
        float_number *= sign
        # 使用 pytest 的 assert_almost_equal 函数验证解码后的浮点数与原始数值的精度
        tm.assert_almost_equal(
            float_number, ujson.ujson_loads(str(float_number)), rtol=1e-15
        )

    # 测试编码大集合时是否能够正常执行，不会引发异常
    def test_encode_big_set(self):
        s = set(range(100000))
        ujson.ujson_dumps(s)  # 确保不会引发异常

    # 测试编码空集合时是否能够得到预期的 JSON 字符串表示
    def test_encode_empty_set(self):
        assert "[]" == ujson.ujson_dumps(set())

    # 测试编码和解码集合是否能够保持元素的一致性
    def test_encode_set(self):
        s = {1, 2, 3, 4, 5, 6, 7, 8, 9}
        enc = ujson.ujson_dumps(s)
        dec = ujson.ujson_loads(enc)

        for v in dec:
            assert v in s

    # 使用参数化测试，测试编码 Timedelta 对象是否能够得到 ISO 格式的 JSON 字符串
    @pytest.mark.parametrize(
        "td",
        [
            Timedelta(days=366),
            Timedelta(days=-1),
            Timedelta(hours=13, minutes=5, seconds=5),
            Timedelta(hours=13, minutes=20, seconds=30),
            Timedelta(days=-1, nanoseconds=5),
            Timedelta(nanoseconds=1),
            Timedelta(microseconds=1, nanoseconds=1),
            Timedelta(milliseconds=1, microseconds=1, nanoseconds=1),
            Timedelta(milliseconds=999, microseconds=999, nanoseconds=999),
        ],
    )
    def test_encode_timedelta_iso(self, td):
        # GH 28256: 验证 Timedelta 对象被正确编码为 ISO 格式的 JSON 字符串
        result = ujson.ujson_dumps(td, iso_dates=True)
        expected = f'"{td.isoformat()}"'

        assert result == expected

    # 测试编码 PeriodIndex 对象时是否能够得到预期的空 JSON 字符串表示
    def test_encode_periodindex(self):
        # GH 46683: 验证 PeriodIndex 对象能够被正确编码为空 JSON 字符串
        p = PeriodIndex(["2022-04-06", "2022-04-07"], freq="D")
        df = DataFrame(index=p)
        assert df.to_json() == "{}"
```