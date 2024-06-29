# `D:\src\scipysrc\pandas\pandas\tests\io\parser\test_quoting.py`

```
"""
Tests that quoting specifications are properly handled
during parsing for all of the parsers defined in parsers.py
"""

# 导入所需的库和模块
import csv
from io import StringIO

import pytest  # 导入 pytest 测试框架

from pandas.compat import PY311  # 导入兼容性模块
from pandas.errors import ParserError  # 导入解析错误模块

from pandas import DataFrame  # 导入 DataFrame 类
import pandas._testing as tm  # 导入 pandas 测试工具模块

# 忽略特定警告信息
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

# 定义装饰器标记，标记为 xfail_pyarrow 的测试会跳过
xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")

# 定义装饰器标记，标记为 skip_pyarrow 的测试会跳过
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


# 参数化测试函数，测试不合法的 quotechar 参数
@pytest.mark.parametrize(
    "kwargs,msg",
    [
        ({"quotechar": "foo"}, '"quotechar" must be a(n)? 1-character string'),
        (
            {"quotechar": None, "quoting": csv.QUOTE_MINIMAL},
            "quotechar must be set if quoting enabled",
        ),
        ({"quotechar": 2}, '"quotechar" must be string( or None)?, not int'),
    ],
)
@skip_pyarrow  # 标记为 xfail_pyarrow 的测试会跳过，因为会抛出 ParserError: CSV parse error: Empty CSV file or block
def test_bad_quote_char(all_parsers, kwargs, msg):
    data = "1,2,3"
    parser = all_parsers

    # 断言会抛出 TypeError 异常，异常信息符合 msg
    with pytest.raises(TypeError, match=msg):
        parser.read_csv(StringIO(data), **kwargs)


# 参数化测试函数，测试不合法的 quoting 参数
@pytest.mark.parametrize(
    "quoting,msg",
    [
        ("foo", '"quoting" must be an integer|Argument'),
        (10, 'bad "quoting" value'),  # quoting must be in the range [0, 3]
    ],
)
@xfail_pyarrow  # 标记为 xfail_pyarrow 的测试会跳过，因为会抛出 ValueError: The 'quoting' option is not supported
def test_bad_quoting(all_parsers, quoting, msg):
    data = "1,2,3"
    parser = all_parsers

    # 断言会抛出 TypeError 异常，异常信息符合 msg
    with pytest.raises(TypeError, match=msg):
        parser.read_csv(StringIO(data), quoting=quoting)


# 测试 quotechar 基础功能
def test_quote_char_basic(all_parsers):
    parser = all_parsers
    data = 'a,b,c\n1,2,"cat"'
    expected = DataFrame([[1, 2, "cat"]], columns=["a", "b", "c"])

    # 使用指定的 quotechar 参数解析 CSV 数据，并断言结果与预期一致
    result = parser.read_csv(StringIO(data), quotechar='"')
    tm.assert_frame_equal(result, expected)


# 参数化测试函数，测试各种 quotechar 参数
@pytest.mark.parametrize("quote_char", ["~", "*", "%", "$", "@", "P"])
def test_quote_char_various(all_parsers, quote_char):
    parser = all_parsers
    expected = DataFrame([[1, 2, "cat"]], columns=["a", "b", "c"])

    data = 'a,b,c\n1,2,"cat"'
    new_data = data.replace('"', quote_char)

    # 使用不同的 quotechar 参数解析修改后的 CSV 数据，并断言结果与预期一致
    result = parser.read_csv(StringIO(new_data), quotechar=quote_char)
    tm.assert_frame_equal(result, expected)


# 参数化测试函数，测试空或 None 的 quotechar 参数
@xfail_pyarrow  # 标记为 xfail_pyarrow 的测试会跳过，因为会抛出 ValueError: The 'quoting' option is not supported
@pytest.mark.parametrize("quoting", [csv.QUOTE_MINIMAL, csv.QUOTE_NONE])
@pytest.mark.parametrize("quote_char", ["", None])
def test_null_quote_char(all_parsers, quoting, quote_char):
    kwargs = {"quotechar": quote_char, "quoting": quoting}
    data = "a,b,c\n1,2,3"
    parser = all_parsers

    # 断言会抛出 TypeError 异常
    with pytest.raises(TypeError):
        parser.read_csv(StringIO(data), **kwargs)
    # 如果 quoting 参数不等于 csv.QUOTE_NONE，则进行异常检查
    # 根据条件设置异常消息
    msg = (
        '"quotechar" must be a 1-character string'
        if PY311 and all_parsers.engine == "python" and quote_char == ""
        else "quotechar must be set if quoting enabled"
    )
    
    # 使用 pytest 来断言是否抛出 TypeError 异常，并匹配预期的错误消息
    with pytest.raises(TypeError, match=msg):
        # 调用 parser 对象的 read_csv 方法读取 CSV 数据，并传入其他关键字参数
        parser.read_csv(StringIO(data), **kwargs)
    
    # 如果 quoting 参数为 csv.QUOTE_NONE，且不是 Python 3.11+ 且解析引擎不是 Python
    elif not (PY311 and all_parsers.engine == "python"):
        # 创建期望的 DataFrame 对象，用于预期的比较
        expected = DataFrame([[1, 2, 3]], columns=["a", "b", "c"])
        
        # 使用 parser 对象的 read_csv 方法读取 CSV 数据，并传入其他关键字参数
        result = parser.read_csv(StringIO(data), **kwargs)
        
        # 使用 pandas 的 tm 模块来断言 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize(
    "kwargs,exp_data",
    [
        ({}, [[1, 2, "foo"]]),  # 测试默认设置。
        # QUOTE_MINIMAL 只影响 CSV 写入，对读取没有影响。
        ({"quotechar": '"', "quoting": csv.QUOTE_MINIMAL}, [[1, 2, "foo"]]),
        # QUOTE_MINIMAL 只影响 CSV 写入，对读取没有影响。
        ({"quotechar": '"', "quoting": csv.QUOTE_ALL}, [[1, 2, "foo"]]),
        # QUOTE_NONE 告诉读取器不处理引号字符，直接读取。
        ({"quotechar": '"', "quoting": csv.QUOTE_NONE}, [[1, 2, '"foo"']]),
        # QUOTE_NONNUMERIC 告诉读取器将所有非引号字段转换为浮点数。
        ({"quotechar": '"', "quoting": csv.QUOTE_NONNUMERIC}, [[1.0, 2.0, "foo"]]),
    ],
)
@xfail_pyarrow  # 预期会出现 ValueError: 'quoting' 选项不被支持
def test_quoting_various(all_parsers, kwargs, exp_data):
    data = '1,2,"foo"'
    parser = all_parsers
    columns = ["a", "b", "c"]

    result = parser.read_csv(StringIO(data), names=columns, **kwargs)
    expected = DataFrame(exp_data, columns=columns)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "doublequote,exp_data", [(True, [[3, '4 " 5']]), (False, [[3, '4 " 5"']])]
)
def test_double_quote(all_parsers, doublequote, exp_data, request):
    parser = all_parsers
    data = 'a,b\n3,"4 "" 5"'

    if parser.engine == "pyarrow" and not doublequote:
        mark = pytest.mark.xfail(reason="结果不匹配")
        request.applymarker(mark)

    result = parser.read_csv(StringIO(data), quotechar='"', doublequote=doublequote)
    expected = DataFrame(exp_data, columns=["a", "b"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("quotechar", ['"', "\u0001"])
def test_quotechar_unicode(all_parsers, quotechar):
    # 参见 gh-14477
    data = "a\n1"
    parser = all_parsers
    expected = DataFrame({"a": [1]})

    result = parser.read_csv(StringIO(data), quotechar=quotechar)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("balanced", [True, False])
def test_unbalanced_quoting(all_parsers, balanced, request):
    # 参见 gh-22789。
    parser = all_parsers
    data = 'a,b,c\n1,2,"3'

    if parser.engine == "pyarrow" and not balanced:
        mark = pytest.mark.xfail(reason="结果不匹配")
        request.applymarker(mark)

    if balanced:
        # 调整引号的平衡，并且可以正确读取。
        expected = DataFrame([[1, 2, 3]], columns=["a", "b", "c"])
        result = parser.read_csv(StringIO(data + '"'))
        tm.assert_frame_equal(result, expected)
    else:
        msg = (
            "EOF inside string starting at row 1"
            if parser.engine == "c"
            else "unexpected end of data"
        )

        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data))
```