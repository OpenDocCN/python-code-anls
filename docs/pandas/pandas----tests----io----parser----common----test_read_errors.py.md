# `D:\src\scipysrc\pandas\pandas\tests\io\parser\common\test_read_errors.py`

```
"""
Tests that work on the Python, C and PyArrow engines but do not have a
specific classification into the other test modules.
"""

# 导入需要的库和模块
import codecs
import csv
from io import StringIO
import os
from pathlib import Path

import numpy as np
import pytest

# 导入 pandas 相关模块和类
from pandas.compat import PY311
from pandas.errors import (
    EmptyDataError,
    ParserError,
    ParserWarning,
)

from pandas import DataFrame
import pandas._testing as tm

# 定义 pytest 的标记
xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


# 测试空的小数标记
def test_empty_decimal_marker(all_parsers):
    data = """A|B|C
1|2,334|5
10|13|10.
"""
    # Parsers support only length-1 decimals
    msg = "Only length-1 decimal markers supported"
    parser = all_parsers

    if parser.engine == "pyarrow":
        msg = (
            "only single character unicode strings can be "
            "converted to Py_UCS4, got length 0"
        )

    # 使用 pytest 断言检查是否抛出 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), decimal="")


# 测试错误的流异常
def test_bad_stream_exception(all_parsers, csv_dir_path):
    # see gh-13652
    #
    # This test validates that both the Python engine and C engine will
    # raise UnicodeDecodeError instead of C engine raising ParserError
    # and swallowing the exception that caused read to fail.
    path = os.path.join(csv_dir_path, "sauron.SHIFT_JIS.csv")
    codec = codecs.lookup("utf-8")
    utf8 = codecs.lookup("utf-8")
    parser = all_parsers
    msg = "'utf-8' codec can't decode byte"

    # 流必须是二进制 UTF8 格式
    with (
        open(path, "rb") as handle,
        codecs.StreamRecoder(
            handle, utf8.encode, utf8.decode, codec.streamreader, codec.streamwriter
        ) as stream,
    ):
        # 使用 pytest 断言检查是否抛出 UnicodeDecodeError 异常，并匹配预期的错误消息
        with pytest.raises(UnicodeDecodeError, match=msg):
            parser.read_csv(stream)


# 测试格式错误
def test_malformed(all_parsers):
    # see gh-6607
    parser = all_parsers
    data = """ignore
A,B,C
1,2,3 # comment
1,2,3,4,5
2,3,4
"""
    msg = "Expected 3 fields in line 4, saw 5"
    err = ParserError
    if parser.engine == "pyarrow":
        msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        err = ValueError
    # 使用 pytest 断言检查是否抛出特定类型的异常，并匹配预期的错误消息
    with pytest.raises(err, match=msg):
        parser.read_csv(StringIO(data), header=1, comment="#")


# 测试格式错误的数据块
@pytest.mark.parametrize("nrows", [5, 3, None])
def test_malformed_chunks(all_parsers, nrows):
    data = """ignore
A,B,C
skip
1,2,3
3,5,10 # comment
1,2,3,4,5
2,3,4
"""
    parser = all_parsers

    if parser.engine == "pyarrow":
        msg = "The 'iterator' option is not supported with the 'pyarrow' engine"
        # 使用 pytest 断言检查是否抛出 ValueError 异常，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data),
                header=1,
                comment="#",
                iterator=True,
                chunksize=1,
                skiprows=[2],
            )
        return

    msg = "Expected 3 fields in line 6, saw 5"
    # 使用 parser 对象读取 CSV 格式的数据，数据来源是内存中的字符串 data
    with parser.read_csv(
        StringIO(data), header=1, comment="#", iterator=True, chunksize=1, skiprows=[2]
    ) as reader:
        # 使用 pytest 进行断言，验证在读取数据时是否会抛出 ParserError 异常，并且异常消息需匹配变量 msg
        with pytest.raises(ParserError, match=msg):
            # 使用 reader 对象读取指定行数的数据
            reader.read(nrows)
@xfail_pyarrow  # 标记此测试预期不会触发异常
def test_catch_too_many_names(all_parsers):
    # 准备测试数据，包含四行三列的 CSV 数据
    data = """\
1,2,3
4,,6
7,8,9
10,11,12\n"""
    # 从参数中获取所有的解析器
    parser = all_parsers
    # 根据解析器类型确定异常消息内容
    msg = (
        "Too many columns specified: expected 4 and found 3"
        if parser.engine == "c"
        else "Number of passed names did not match "
        "number of header fields in the file"
    )

    # 使用 pytest 断言捕获 ValueError 异常，并匹配预期的消息内容
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), header=0, names=["a", "b", "c", "d"])


@skip_pyarrow  # 跳过在 PyArrow 下的测试，因为会出现 CSV 解析错误：空的 CSV 文件或块
@pytest.mark.parametrize("nrows", [0, 1, 2, 3, 4, 5])
def test_raise_on_no_columns(all_parsers, nrows):
    # 从参数中获取所有的解析器
    parser = all_parsers
    # 准备包含指定行数空行的数据
    data = "\n" * nrows

    # 准备异常消息内容
    msg = "No columns to parse from file"
    # 使用 pytest 断言捕获 EmptyDataError 异常，并匹配预期的消息内容
    with pytest.raises(EmptyDataError, match=msg):
        parser.read_csv(StringIO(data))


def test_unexpected_keyword_parameter_exception(all_parsers):
    # GH-34976
    # 从参数中获取所有的解析器
    parser = all_parsers

    # 准备格式化的异常消息模板
    msg = "{}\\(\\) got an unexpected keyword argument 'foo'"
    # 使用 pytest 断言捕获 TypeError 异常，并匹配格式化后的消息内容
    with pytest.raises(TypeError, match=msg.format("read_csv")):
        parser.read_csv("foo.csv", foo=1)
    with pytest.raises(TypeError, match=msg.format("read_table")):
        parser.read_table("foo.tsv", foo=1)


def test_suppress_error_output(all_parsers):
    # see gh-15925
    # 从参数中获取所有的解析器
    parser = all_parsers
    # 准备包含有误行为数据的 CSV 数据
    data = "a\n1\n1,2,3\n4\n5,6,7"
    # 准备预期的 DataFrame 结果
    expected = DataFrame({"a": [1, 4]})

    # 使用指定的参数忽略错误行，读取 CSV 数据
    result = parser.read_csv(StringIO(data), on_bad_lines="skip")
    # 使用 pandas 测试工具函数验证结果与预期 DataFrame 相等
    tm.assert_frame_equal(result, expected)


def test_error_bad_lines(all_parsers):
    # see gh-15925
    # 从参数中获取所有的解析器
    parser = all_parsers
    # 准备包含有误行为数据的 CSV 数据
    data = "a\n1\n1,2,3\n4\n5,6,7"

    # 准备预期的异常消息
    msg = "Expected 1 fields in line 3, saw 3"

    # 如果解析器引擎是 pyarrow，则跳过测试，并附上原因链接
    if parser.engine == "pyarrow":
        pytest.skip(reason="https://github.com/apache/arrow/issues/38676")

    # 使用 pytest 断言捕获 ParserError 异常，并匹配预期的消息内容
    with pytest.raises(ParserError, match=msg):
        parser.read_csv(StringIO(data), on_bad_lines="error")


def test_warn_bad_lines(all_parsers):
    # see gh-15925
    # 从参数中获取所有的解析器
    parser = all_parsers
    # 准备包含有误行为数据的 CSV 数据
    data = "a\n1\n1,2,3\n4\n5,6,7"
    # 准备预期的 DataFrame 结果
    expected = DataFrame({"a": [1, 4]})
    # 准备匹配的警告消息
    match_msg = "Skipping line"

    # 如果解析器引擎是 pyarrow，则使用不同的警告消息
    expected_warning = ParserWarning
    if parser.engine == "pyarrow":
        match_msg = "Expected 1 columns, but found 3: 1,2,3"

    # 使用 pandas 测试工具函数验证警告消息是否出现，同时验证结果与预期 DataFrame 相等
    with tm.assert_produces_warning(
        expected_warning, match=match_msg, check_stacklevel=False
    ):
        result = parser.read_csv(StringIO(data), on_bad_lines="warn")
    tm.assert_frame_equal(result, expected)


def test_read_csv_wrong_num_columns(all_parsers):
    # Too few columns.
    # 准备包含不同列数的 CSV 数据
    data = """A,B,C,D,E,F
1,2,3,4,5,6
6,7,8,9,10,11,12
11,12,13,14,15,16
"""
    # 从参数中获取所有的解析器
    parser = all_parsers
    # 准备预期的异常消息
    msg = "Expected 6 fields in line 3, saw 7"

    # 如果解析器引擎是 pyarrow，则跳过测试，并附上原因链接
    if parser.engine == "pyarrow":
        pytest.skip(reason="https://github.com/apache/arrow/issues/38676")

    # 使用 pytest 断言捕获 ParserError 异常，并匹配预期的消息内容
    with pytest.raises(ParserError, match=msg):
        parser.read_csv(StringIO(data))
# 测试处理包含空字节字符的输入情况
def test_null_byte_char(request, all_parsers):
    # 见问题 gh-2741

    # 设置包含空字节字符的测试数据
    data = "\x00,foo"

    # 设置列名
    names = ["a", "b"]

    # 获取所有解析器
    parser = all_parsers

    # 根据解析器引擎和 Python 版本选择是否标记测试为预期失败
    if parser.engine == "c" or (parser.engine == "python" and PY311):
        if parser.engine == "python" and PY311:
            # 在 Python 3.11 中，空字节字符被读作空字符而不是空字符
            request.applymarker(
                pytest.mark.xfail(
                    reason="In Python 3.11, this is read as an empty character not null"
                )
            )
        # 预期的数据框架结果
        expected = DataFrame([[np.nan, "foo"]], columns=names)
        # 使用解析器读取 CSV 数据
        out = parser.read_csv(StringIO(data), names=names)
        # 断言输出与预期结果一致
        tm.assert_frame_equal(out, expected)
    else:
        if parser.engine == "pyarrow":
            # CSV 解析错误：空 CSV 文件或块："无法推断列数"
            pytest.skip(reason="https://github.com/apache/arrow/issues/38676")
        else:
            # 提示检测到空字节
            msg = "NULL byte detected"
        # 使用 pytest 断言预期引发解析器错误，并匹配特定消息
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), names=names)


# 测试打开文件时的异常情况处理
@pytest.mark.filterwarnings("always::ResourceWarning")
def test_open_file(all_parsers):
    # 问题 GH 39024

    # 获取所有解析器
    parser = all_parsers

    # 初始化错误消息和错误类型
    msg = "Could not determine delimiter"
    err = csv.Error

    # 根据解析器引擎选择不同的错误消息和错误类型
    if parser.engine == "c":
        msg = "object of type 'NoneType' has no len"
        err = TypeError
    elif parser.engine == "pyarrow":
        msg = "'utf-8' codec can't decode byte 0xe4"
        err = ValueError

    # 使用 tm.ensure_clean() 确保路径干净，并创建测试文件
    with tm.ensure_clean() as path:
        file = Path(path)
        file.write_bytes(b"\xe4\na\n1")

        # 使用 tm.assert_produces_warning(None) 断言不会触发 ResourceWarning
        with tm.assert_produces_warning(None):
            # 使用 pytest 断言预期引发特定类型的错误，并匹配特定消息
            with pytest.raises(err, match=msg):
                parser.read_csv(file, sep=None, encoding_errors="replace")


# 测试处理非法行输入时的异常情况处理
def test_invalid_on_bad_line(all_parsers):
    # 获取所有解析器
    parser = all_parsers

    # 设置测试数据
    data = "a\n1\n1,2,3\n4\n5,6,7"

    # 使用 pytest 断言预期引发特定类型的错误，并匹配特定消息
    with pytest.raises(ValueError, match="Argument abc is invalid for on_bad_lines"):
        parser.read_csv(StringIO(data), on_bad_lines="abc")


# 测试处理错误表头格式时的异常情况处理
def test_bad_header_uniform_error(all_parsers):
    # 获取所有解析器
    parser = all_parsers

    # 设置测试数据
    data = "+++123456789...\ncol1,col2,col3,col4\n1,2,3,4\n"

    # 设置预期的错误消息
    msg = "Expected 2 fields in line 2, saw 4"

    # 根据解析器引擎选择不同的错误消息
    if parser.engine == "c":
        msg = (
            "Could not construct index. Requested to use 1 "
            "number of columns, but 3 left to parse."
        )
    elif parser.engine == "pyarrow":
        # "CSV 解析错误：预期 1 列，实际得到 4 列：col1,col2,col3,col4"
        pytest.skip(reason="https://github.com/apache/arrow/issues/38676")

    # 使用 pytest 断言预期引发解析器错误，并匹配特定消息
    with pytest.raises(ParserError, match=msg):
        parser.read_csv(StringIO(data), index_col=0, on_bad_lines="error")


# 测试警告正确格式化的情况
def test_on_bad_lines_warn_correct_formatting(all_parsers):
    # 见问题 gh-15925

    # 获取所有解析器
    parser = all_parsers

    # 设置测试数据
    data = """1,2
a,b
a,b,c
a,b,d
a,b
"""

    # 设置预期的数据框架结果
    expected = DataFrame({"1": "a", "2": ["b"] * 2})

    # 匹配的警告消息
    match_msg = "Skipping line"

    # 预期的警告类型
    expected_warning = ParserWarning
    # 如果解析器的引擎是 "pyarrow"，则定义匹配消息为 "Expected 2 columns, but found 3: a,b,c"
    if parser.engine == "pyarrow":
        match_msg = "Expected 2 columns, but found 3: a,b,c"
    
    # 使用 assert_produces_warning 上下文管理器来捕获特定的警告消息
    with tm.assert_produces_warning(
        expected_warning, match=match_msg, check_stacklevel=False
    ):
        # 使用解析器解析给定的 CSV 数据（作为字符串流），在遇到不良行时发出警告
        result = parser.read_csv(StringIO(data), on_bad_lines="warn")
    
    # 检查解析后的结果是否与期望的数据帧相等，若不相等则会引发异常
    tm.assert_frame_equal(result, expected)
```