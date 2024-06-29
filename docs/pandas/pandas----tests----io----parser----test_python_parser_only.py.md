# `D:\src\scipysrc\pandas\pandas\tests\io\parser\test_python_parser_only.py`

```
"""
Tests that apply specifically to the Python parser. Unless specifically
stated as a Python-specific issue, the goal is to eventually move as many of
these tests out of this module as soon as the C parser can accept further
arguments when parsing.
"""

# 导入必要的模块和库
from __future__ import annotations  # 允许在类型注解中使用类型本身，仅 Python 3.7 以上支持

import csv  # 导入 CSV 模块
from io import (  # 从 io 模块导入以下对象
    BytesIO,  # 用于处理二进制数据的内存缓冲区
    StringIO,  # 用于处理文本数据的内存缓冲区
    TextIOWrapper,  # 用于将字节流包装成文本流的对象
)
from typing import TYPE_CHECKING  # 导入类型检查相关模块

import numpy as np  # 导入 NumPy 库，并重命名为 np
import pytest  # 导入 pytest 测试框架

from pandas.errors import (  # 从 pandas.errors 模块导入以下对象
    ParserError,  # 解析错误异常类
    ParserWarning,  # 解析警告异常类
)

from pandas import (  # 从 pandas 库导入以下对象
    DataFrame,  # 数据帧类
    Index,  # 索引类
    MultiIndex,  # 多重索引类
)
import pandas._testing as tm  # 导入 pandas 测试模块，并重命名为 tm

if TYPE_CHECKING:
    from collections.abc import Iterator  # 如果是类型检查阶段，则从 collections.abc 模块导入 Iterator 类


def test_default_separator(python_parser_only):
    # 测试默认分隔符处理
    #
    # 在 Python 中，csv.Sniffer 将 "o" 视为分隔符。
    data = "aob\n1o2\n3o4"
    parser = python_parser_only  # 使用 python_parser_only 作为解析器对象

    expected = DataFrame({"a": [1, 3], "b": [2, 4]})
    # 读取 CSV 数据，分隔符为 None
    result = parser.read_csv(StringIO(data), sep=None)
    # 断言结果与预期相同
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("skipfooter", ["foo", 1.5, True])
def test_invalid_skipfooter_non_int(python_parser_only, skipfooter):
    # 测试非整数 skipfooter 参数的无效情况
    #
    # 见 gh-15925 (comment)
    data = "a\n1\n2"
    parser = python_parser_only  # 使用 python_parser_only 作为解析器对象
    msg = "skipfooter must be an integer"  # 错误消息

    with pytest.raises(ValueError, match=msg):
        # 断言抛出 ValueError 异常，并匹配指定的错误消息
        parser.read_csv(StringIO(data), skipfooter=skipfooter)


def test_invalid_skipfooter_negative(python_parser_only):
    # 测试负数 skipfooter 参数的无效情况
    #
    # 见 gh-15925 (comment)
    data = "a\n1\n2"
    parser = python_parser_only  # 使用 python_parser_only 作为解析器对象
    msg = "skipfooter cannot be negative"  # 错误消息

    with pytest.raises(ValueError, match=msg):
        # 断言抛出 ValueError 异常，并匹配指定的错误消息
        parser.read_csv(StringIO(data), skipfooter=-1)


@pytest.mark.parametrize("kwargs", [{"sep": None}, {"delimiter": "|"}])
def test_sniff_delimiter(python_parser_only, kwargs):
    # 测试识别分隔符功能
    data = """index|A|B|C
foo|1|2|3
bar|4|5|6
baz|7|8|9
"""
    parser = python_parser_only  # 使用 python_parser_only 作为解析器对象

    # 读取 CSV 数据，传入额外的参数 kwargs
    result = parser.read_csv(StringIO(data), index_col=0, **kwargs)
    expected = DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        columns=["A", "B", "C"],
        index=Index(["foo", "bar", "baz"], name="index"),
    )
    # 断言结果与预期相同
    tm.assert_frame_equal(result, expected)


def test_sniff_delimiter_comment(python_parser_only):
    # 测试识别分隔符并忽略注释功能
    data = """# comment line
index|A|B|C
# comment line
foo|1|2|3 # ignore | this
bar|4|5|6
baz|7|8|9
"""
    parser = python_parser_only  # 使用 python_parser_only 作为解析器对象

    # 读取 CSV 数据，分隔符为 None，忽略以 "#" 开头的注释行
    result = parser.read_csv(StringIO(data), index_col=0, sep=None, comment="#")
    expected = DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        columns=["A", "B", "C"],
        index=Index(["foo", "bar", "baz"], name="index"),
    )
    # 断言结果与预期相同
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("encoding", [None, "utf-8"])
def test_sniff_delimiter_encoding(python_parser_only, encoding):
    # 测试识别分隔符并指定编码功能
    parser = python_parser_only  # 使用 python_parser_only 作为解析器对象
    data = """ignore this
ignore this too
index|A|B|C
foo|1|2|3
bar|4|5|6
baz|7|8|9
"""
    # 如果提供了编码方式，则将数据编码为指定编码格式
    if encoding is not None:
        data = data.encode(encoding)
        # 将编码后的数据封装成字节流对象
        data = BytesIO(data)
        # 使用指定编码格式创建文本I/O包装器对象
        data = TextIOWrapper(data, encoding=encoding)
    else:
        # 如果未提供编码方式，则将数据封装成字符串I/O对象
        data = StringIO(data)

    # 使用解析器读取CSV数据，设定首列为索引，分隔符为任意字符，跳过前两行，使用给定编码格式解析
    result = parser.read_csv(data, index_col=0, sep=None, skiprows=2, encoding=encoding)
    
    # 期望的数据框架，包含特定的数据内容、列名和索引
    expected = DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        columns=["A", "B", "C"],
        index=Index(["foo", "bar", "baz"], name="index"),
    )
    
    # 使用测试模块中的方法比较结果和期望的数据框架是否相等
    tm.assert_frame_equal(result, expected)
# 单行测试函数，用于测试 Python 解析器只能处理的 CSV 文件
def test_single_line(python_parser_only):
    # 见问题 gh-6607：探测分隔符
    parser = python_parser_only
    # 读取包含数据 "1,2" 的 CSV 文件，指定列名和无标题行，分隔符为 None
    result = parser.read_csv(StringIO("1,2"), names=["a", "b"], header=None, sep=None)

    # 期望的 DataFrame 结果
    expected = DataFrame({"a": [1], "b": [2]})
    # 断言结果与期望一致
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("kwargs", [{"skipfooter": 2}, {"nrows": 3}])
def test_skipfooter(python_parser_only, kwargs):
    # 见问题 gh-6607
    # 包含需要跳过的尾行的数据
    data = """A,B,C
1,2,3
4,5,6
7,8,9
want to skip this
also also skip this
"""
    parser = python_parser_only
    # 读取包含指定参数的 CSV 数据
    result = parser.read_csv(StringIO(data), **kwargs)

    # 期望的 DataFrame 结果
    expected = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["A", "B", "C"])
    # 断言结果与期望一致
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "compression,klass", [("gzip", "GzipFile"), ("bz2", "BZ2File")]
)
def test_decompression_regex_sep(python_parser_only, csv1, compression, klass):
    # 见问题 gh-6607
    parser = python_parser_only

    with open(csv1, "rb") as f:
        data = f.read()

    # 替换数据中的逗号为双冒号
    data = data.replace(b",", b"::")
    # 读取原始 CSV 文件作为期望结果
    expected = parser.read_csv(csv1)

    # 导入指定的压缩模块
    module = pytest.importorskip(compression)
    klass = getattr(module, klass)

    with tm.ensure_clean() as path:
        with klass(path, mode="wb") as tmp:
            tmp.write(data)

        # 使用双冒号作为分隔符读取压缩后的数据
        result = parser.read_csv(path, sep="::", compression=compression)
        # 断言结果与期望一致
        tm.assert_frame_equal(result, expected)


def test_read_csv_buglet_4x_multi_index(python_parser_only):
    # 见问题 gh-6607
    # 包含多层索引的 CSV 数据
    data = """                      A       B       C       D        E
one two three   four
a   b   10.0032 5    -0.5109 -2.3358 -0.4645  0.05076  0.3640
a   q   20      4     0.4473  1.4152  0.2834  1.00661  0.1744
x   q   30      3    -0.6662 -0.5243 -0.3580  0.89145  2.5838"""
    parser = python_parser_only

    # 期望的 DataFrame 结果
    expected = DataFrame(
        [
            [-0.5109, -2.3358, -0.4645, 0.05076, 0.3640],
            [0.4473, 1.4152, 0.2834, 1.00661, 0.1744],
            [-0.6662, -0.5243, -0.3580, 0.89145, 2.5838],
        ],
        columns=["A", "B", "C", "D", "E"],
        index=MultiIndex.from_tuples(
            [("a", "b", 10.0032, 5), ("a", "q", 20, 4), ("x", "q", 30, 3)],
            names=["one", "two", "three", "four"],
        ),
    )
    # 使用正则表达式分隔符读取 CSV 数据
    result = parser.read_csv(StringIO(data), sep=r"\s+")
    # 断言结果与期望一致
    tm.assert_frame_equal(result, expected)


def test_read_csv_buglet_4x_multi_index2(python_parser_only):
    # 见问题 gh-6893
    # 包含特定格式的 CSV 数据
    data = "      A B C\na b c\n1 3 7 0 3 6\n3 1 4 1 5 9"
    parser = python_parser_only

    # 期望的 DataFrame 结果
    expected = DataFrame.from_records(
        [(1, 3, 7, 0, 3, 6), (3, 1, 4, 1, 5, 9)],
        columns=list("abcABC"),
        index=list("abc"),
    )
    # 使用正则表达式分隔符读取 CSV 数据
    result = parser.read_csv(StringIO(data), sep=r"\s+")
    # 断言结果与期望一致
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("add_footer", [True, False])
def test_skipfooter_with_decimal(python_parser_only, add_footer):
    # 见问题 gh-6971
    data = "1#2\n3#4"
    parser = python_parser_only
    # 创建一个预期的 DataFrame，包含一列名为 "a" 的浮点数值
    expected = DataFrame({"a": [1.2, 3.4]})
    
    # 如果 add_footer 为 True，则表示数据末尾可能有一个额外的行（footer），
    # 使用 skipfooter 参数来跳过末尾的行，将其不纳入解析的数据中
    if add_footer:
        # 设置解析 CSV 数据时的参数字典，跳过末尾的一行
        kwargs = {"skipfooter": 1}
        # 在数据末尾追加一个示意的“Footer”行
        data += "\nFooter"
    else:
        # 如果没有额外的 footer 行，则不需要额外的参数
        kwargs = {}
    
    # 使用 parser 对象解析内存中的 CSV 数据，指定数据的列名为 "a"，小数点符号为 "#"
    result = parser.read_csv(StringIO(data), names=["a"], decimal="#", **kwargs)
    
    # 使用断言检查解析后的结果是否与预期的 DataFrame 相等
    tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize(
    "sep", ["::", "#####", "!!!", "123", "#1!c5", "%!c!d", "@@#4:2", "_!pd#_"]
)
@pytest.mark.parametrize(
    "encoding", ["utf-16", "utf-16-be", "utf-16-le", "utf-32", "cp037"]
)
# 定义测试函数，测试不同的分隔符和编码方式
def test_encoding_non_utf8_multichar_sep(python_parser_only, sep, encoding):
    # 创建预期的 DataFrame 对象，包含列 'a' 和 'b'，每列包含一个值
    expected = DataFrame({"a": [1], "b": [2]})
    # 获取用于解析的解析器对象
    parser = python_parser_only

    # 创建数据字符串，使用给定的分隔符将两个数字串联起来
    data = "1" + sep + "2"
    # 将数据字符串编码成指定编码方式的字节流
    encoded_data = data.encode(encoding)

    # 使用解析器对象读取编码后的字节流，指定分隔符和列名，返回结果
    result = parser.read_csv(
        BytesIO(encoded_data), sep=sep, names=["a", "b"], encoding=encoding
    )
    # 断言结果与预期的 DataFrame 相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("quoting", [csv.QUOTE_MINIMAL, csv.QUOTE_NONE])
# 定义测试函数，测试带引号和不带引号的情况下的多字符分隔符
def test_multi_char_sep_quotes(python_parser_only, quoting):
    # 设置关键字参数，指定分隔符为 ",," 
    kwargs = {"sep": ",,"}
    # 获取用于解析的解析器对象
    parser = python_parser_only

    # 创建包含多行数据的字符串
    data = 'a,,b\n1,,a\n2,,"2,,b"'

    # 根据引号处理方式进行断言
    if quoting == csv.QUOTE_NONE:
        # 如果不带引号，预期会抛出 ParserError 异常，且异常信息包含指定的消息
        msg = "Expected 2 fields in line 3, saw 3"
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), quoting=quoting, **kwargs)
    else:
        # 如果带引号，预期会抛出 ParserError 异常，且异常信息包含指定的消息
        msg = "ignored when a multi-char delimiter is used"
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), quoting=quoting, **kwargs)


# 定义测试函数，测试不带分隔符的情况下的处理
def test_none_delimiter(python_parser_only):
    # 获取用于解析的解析器对象
    parser = python_parser_only
    # 创建包含多行数据的字符串
    data = "a,b,c\n0,1,2\n3,4,5,6\n7,8,9"
    # 创建预期的 DataFrame 对象，包含列 'a', 'b', 'c'，每列包含两个值
    expected = DataFrame({"a": [0, 7], "b": [1, 8], "c": [2, 9]})

    # 预期第三行数据因为格式不正确会被跳过，并发出警告
    with tm.assert_produces_warning(
        ParserWarning, match="Skipping line 3", check_stacklevel=False
    ):
        # 使用解析器对象读取数据，指定头部行数、不指定分隔符、错误行处理方式，并返回结果
        result = parser.read_csv(
            StringIO(data), header=0, sep=None, on_bad_lines="warn"
        )
    # 断言结果与预期的 DataFrame 相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("data", ['a\n1\n"b"a', 'a,b,c\ncat,foo,bar\ndog,foo,"baz'])
@pytest.mark.parametrize("skipfooter", [0, 1])
# 定义测试函数，测试跳过数据尾部行出现的问题
def test_skipfooter_bad_row(python_parser_only, data, skipfooter):
    # 获取用于解析的解析器对象
    parser = python_parser_only
    if skipfooter:
        # 如果跳过尾部行数大于 0，预期会抛出 ParserError 异常，且异常信息包含指定的消息
        msg = "parsing errors in the skipped footer rows"
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), skipfooter=skipfooter)
    else:
        # 如果跳过尾部行数为 0，预期会抛出 ParserError 异常，且异常信息包含指定的消息
        msg = "unexpected end of data|expected after"
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), skipfooter=skipfooter)


# 定义测试函数，测试 Python 引擎处理文件时的情况
def test_python_engine_file_no_next(python_parser_only):
    # 获取用于解析的解析器对象
    parser = python_parser_only
    # 定义一个名为 NoNextBuffer 的类，用于模拟一个无下一个缓冲区的对象
    class NoNextBuffer:
        # 初始化方法，接受一个 csv 数据作为参数，并将其保存在实例变量 self.data 中
        def __init__(self, csv_data) -> None:
            self.data = csv_data

        # 返回一个迭代器，用于迭代实例中保存的 csv 数据
        def __iter__(self) -> Iterator:
            return self.data.__iter__()

        # 返回实例中保存的 csv 数据，模拟读取操作
        def read(self):
            return self.data

        # 返回实例中保存的 csv 数据的一行，模拟读取一行操作
        def readline(self):
            return self.data

    # 调用 parser 对象的 read_csv 方法，传入一个 NoNextBuffer 类的实例，该实例包含字符串 "a\n1"
    parser.read_csv(NoNextBuffer("a\n1"))
@pytest.mark.parametrize("bad_line_func", [lambda x: ["2", "3"], lambda x: x[:2]])
# 使用pytest的参数化装饰器，为test_on_bad_lines_callable函数提供两个不同的bad_line_func作为参数
def test_on_bad_lines_callable(python_parser_only, bad_line_func):
    # GH 5686
    # 设置解析器为python_parser_only，准备测试数据
    parser = python_parser_only
    data = """a,b
1,2
2,3,4,5,6
3,4
"""
    # 创建包含错误数据的StringIO对象
    bad_sio = StringIO(data)
    # 调用解析器的read_csv方法，使用指定的bad_line_func处理错误行
    result = parser.read_csv(bad_sio, on_bad_lines=bad_line_func)
    # 创建期望的DataFrame对象
    expected = DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]})
    # 断言结果与期望是否一致
    tm.assert_frame_equal(result, expected)


def test_on_bad_lines_callable_write_to_external_list(python_parser_only):
    # GH 5686
    # 设置解析器为python_parser_only，准备测试数据
    parser = python_parser_only
    data = """a,b
1,2
2,3,4,5,6
3,4
"""
    # 创建包含错误数据的StringIO对象
    bad_sio = StringIO(data)
    lst = []

    def bad_line_func(bad_line: list[str]) -> list[str]:
        # 定义bad_line_func函数，将错误行追加到lst列表中，返回固定的错误处理结果
        lst.append(bad_line)
        return ["2", "3"]

    # 调用解析器的read_csv方法，使用定义的bad_line_func处理错误行
    result = parser.read_csv(bad_sio, on_bad_lines=bad_line_func)
    # 创建期望的DataFrame对象
    expected = DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]})
    # 断言结果与期望是否一致
    assert lst == [["2", "3", "4", "5", "6"]]
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("bad_line_func", [lambda x: ["foo", "bar"], lambda x: x[:2]])
@pytest.mark.parametrize("sep", [",", "111"])
# 使用pytest的参数化装饰器，为test_on_bad_lines_callable_iterator_true函数提供多个参数组合
def test_on_bad_lines_callable_iterator_true(python_parser_only, bad_line_func, sep):
    # GH 5686
    # iterator=True和iterator=False有不同的代码路径
    # 设置解析器为python_parser_only，准备测试数据
    parser = python_parser_only
    data = f"""
0{sep}1
hi{sep}there
foo{sep}bar{sep}baz
good{sep}bye
"""
    # 创建包含错误数据的StringIO对象
    bad_sio = StringIO(data)
    # 调用解析器的read_csv方法，使用指定的bad_line_func处理错误行
    result_iter = parser.read_csv(
        bad_sio, on_bad_lines=bad_line_func, chunksize=1, iterator=True, sep=sep
    )
    # 创建期望的DataFrame对象列表
    expecteds = [
        {"0": "hi", "1": "there"},
        {"0": "foo", "1": "bar"},
        {"0": "good", "1": "bye"},
    ]
    # 遍历结果迭代器，并逐一比较结果与期望是否一致
    for i, (result, expected) in enumerate(zip(result_iter, expecteds)):
        expected = DataFrame(expected, index=range(i, i + 1))
        tm.assert_frame_equal(result, expected)


def test_on_bad_lines_callable_dont_swallow_errors(python_parser_only):
    # GH 5686
    # 设置解析器为python_parser_only，准备测试数据
    parser = python_parser_only
    data = """a,b
1,2
2,3,4,5,6
3,4
"""
    # 创建包含错误数据的StringIO对象
    bad_sio = StringIO(data)
    msg = "This function is buggy."

    def bad_line_func(bad_line):
        # 定义bad_line_func函数，抛出包含特定消息的ValueError异常
        raise ValueError(msg)

    # 使用pytest的断言，验证是否抛出特定消息的ValueError异常
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(bad_sio, on_bad_lines=bad_line_func)


def test_on_bad_lines_callable_not_expected_length(python_parser_only):
    # GH 5686
    # 设置解析器为python_parser_only，准备测试数据
    parser = python_parser_only
    data = """a,b
1,2
2,3,4,5,6
3,4
"""
    # 创建包含错误数据的StringIO对象
    bad_sio = StringIO(data)
    # 调用解析器的read_csv_check_warnings方法，使用lambda函数作为on_bad_lines处理不符合预期长度的行
    result = parser.read_csv_check_warnings(
        ParserWarning, "Length of header or names", bad_sio, on_bad_lines=lambda x: x
    )
    # 创建期望的DataFrame对象
    expected = DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]})
    # 断言结果与期望是否一致
    tm.assert_frame_equal(result, expected)


def test_on_bad_lines_callable_returns_none(python_parser_only):
    # GH 5686
    # 设置解析器为python_parser_only，准备测试数据
    parser = python_parser_only
    data = """a,b
1,2
2,3,4,5,6
3,4
"""
    # 创建包含错误数据的StringIO对象
    bad_sio = StringIO(data)
    # 调用解析器的read_csv方法，使用lambda函数返回None处理错误行
    result = parser.read_csv(bad_sio, on_bad_lines=lambda x: None)
    # 创建期望的DataFrame对象
    expected = DataFrame({"a": [1, 3], "b": [2, 4]})
    # 使用测试工具比较数据框架 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
# 测试处理在非预期行上使用指定索引列推断功能
def test_on_bad_lines_index_col_inferred(python_parser_only):
    # 从参数中获取仅使用Python解析器的实例
    parser = python_parser_only
    # 准备包含错误数据的CSV格式字符串
    data = """a,b
1,2,3
4,5,6
"""
    # 创建包含错误数据的字符串IO对象
    bad_sio = StringIO(data)

    # 使用自定义处理函数来读取CSV数据，将无效行替换为指定值
    result = parser.read_csv(bad_sio, on_bad_lines=lambda x: ["99", "99"])
    # 创建预期结果DataFrame，包含指定的列和索引
    expected = DataFrame({"a": [2, 5], "b": [3, 6]}, index=[1, 4])
    # 断言读取结果与预期结果是否相等
    tm.assert_frame_equal(result, expected)


# 测试当header参数为None且index_col参数为False时的CSV读取行为
def test_index_col_false_and_header_none(python_parser_only):
    # 从参数中获取仅使用Python解析器的实例
    parser = python_parser_only
    # 准备包含特定内容的CSV格式字符串
    data = """
0.5,0.03
0.1,0.2,0.3,2
"""
    # 使用指定配置读取CSV数据，并检查警告
    result = parser.read_csv_check_warnings(
        ParserWarning,
        "Length of header",
        StringIO(data),
        sep=",",
        header=None,
        index_col=False,
    )
    # 创建预期结果DataFrame，包含指定的列
    expected = DataFrame({0: [0.5, 0.1], 1: [0.03, 0.2]})
    # 断言读取结果与预期结果是否相等
    tm.assert_frame_equal(result, expected)


# 测试在不同行上不推断多级索引名称时的CSV读取行为
def test_header_int_do_not_infer_multiindex_names_on_different_line(python_parser_only):
    # 从参数中获取仅使用Python解析器的实例
    parser = python_parser_only
    # 准备包含特定内容的字符串IO对象
    data = StringIO("a\na,b\nc,d,e\nf,g,h")
    # 使用指定配置读取CSV数据，并检查警告
    result = parser.read_csv_check_warnings(
        ParserWarning, "Length of header", data, engine="python", index_col=False
    )
    # 创建预期结果DataFrame，包含指定的列
    expected = DataFrame({"a": ["a", "c", "f"]})
    # 断言读取结果与预期结果是否相等
    tm.assert_frame_equal(result, expected)


# 参数化测试，用于测试非数值列中的千位分隔符处理
@pytest.mark.parametrize(
    "dtype", [{"a": object}, {"a": str, "b": np.int64, "c": np.int64}]
)
def test_no_thousand_convert_with_dot_for_non_numeric_cols(python_parser_only, dtype):
    # 从参数中获取仅使用Python解析器的实例
    parser = python_parser_only
    # 准备包含特定内容的CSV格式字符串
    data = """\
a;b;c
0000.7995;16.000;0
3.03.001.00514;0;4.000
4923.600.041;23.000;131"""
    # 使用指定配置读取CSV数据，包含自定义数据类型和千位分隔符
    result = parser.read_csv(
        StringIO(data),
        sep=";",
        dtype=dtype,
        thousands=".",
    )
    # 创建预期结果DataFrame，包含处理后的数值列
    expected = DataFrame(
        {
            "a": ["0000.7995", "3.03.001.00514", "4923.600.041"],
            "b": [16000, 0, 23000],
            "c": [0, 4000, 131],
        }
    )
    # 断言读取结果与预期结果是否相等
    tm.assert_frame_equal(result, expected)


# 参数化测试，用于测试非数值列中的千位分隔符处理
@pytest.mark.parametrize(
    "dtype,expected",
    [
        (
            {"a": str, "b": np.float64, "c": np.int64},
            {
                "b": [16000.1, 0, 23000],
                "c": [0, 4001, 131],
            },
        ),
        (
            str,
            {
                "b": ["16,000.1", "0", "23,000"],
                "c": ["0", "4,001", "131"],
            },
        ),
    ],
)
def test_no_thousand_convert_for_non_numeric_cols(python_parser_only, dtype, expected):
    # 从参数中获取仅使用Python解析器的实例
    parser = python_parser_only
    # 准备包含特定内容的CSV格式字符串
    data = """a;b;c
0000,7995;16,000.1;0
3,03,001,00514;0;4,001
4923,600,041;23,000;131
"""
    # 使用指定配置读取CSV数据，包含自定义数据类型和千位分隔符
    result = parser.read_csv(
        StringIO(data),
        sep=";",
        dtype=dtype,
        thousands=",",
    )
    # 创建预期结果DataFrame，包含处理后的数据列
    expected = DataFrame(expected)
    expected.insert(0, "a", ["0000,7995", "3,03,001,00514", "4923,600,041"])
    # 断言读取结果与预期结果是否相等
    tm.assert_frame_equal(result, expected)
```