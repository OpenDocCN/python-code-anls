# `D:\src\scipysrc\pandas\pandas\tests\io\parser\test_encoding.py`

```
"""
Tests encoding functionality during parsing
for all of the parsers defined in parsers.py
"""

# 导入必要的库
from io import (
    BytesIO,          # 用于处理二进制数据的内存缓冲区
    TextIOWrapper,    # 用于将字节流转换为文本文件对象
)
import os              # 系统操作相关的库
import tempfile        # 用于创建临时文件和目录的库
import uuid            # 用于生成唯一标识符的库

import numpy as np     # 用于科学计算的库
import pytest          # 用于编写和运行测试的库

from pandas import (   # 数据分析和处理的库
    DataFrame,         # 用于表示二维数据的对象
    read_csv,          # 用于读取CSV文件的函数
)
import pandas._testing as tm  # pandas 测试工具集

# 忽略特定警告
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


def test_bytes_io_input(all_parsers):
    encoding = "cp1255"  # 指定编码格式
    parser = all_parsers  # 使用所有定义的解析器

    # 创建包含希伯来语和数字的字节流
    data = BytesIO("שלום:1234\n562:123".encode(encoding))
    # 使用解析器读取CSV格式的数据
    result = parser.read_csv(data, sep=":", encoding=encoding)

    # 预期的DataFrame对象
    expected = DataFrame([[562, 123]], columns=["שלום", "1234"])
    # 断言实际结果与预期结果相同
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # 跳过因空白CSV文件或块而导致的CSV解析错误
def test_read_csv_unicode(all_parsers):
    parser = all_parsers  # 使用所有定义的解析器
    # 创建包含Unicode编码的字节流
    data = BytesIO("\u0141aski, Jan;1".encode())

    # 使用解析器读取CSV格式的数据，指定分隔符和编码格式
    result = parser.read_csv(data, sep=";", encoding="utf-8", header=None)
    # 预期的DataFrame对象
    expected = DataFrame([["\u0141aski, Jan", 1]])
    # 断言实际结果与预期结果相同
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
@pytest.mark.parametrize("sep", [",", "\t"])  # 参数化测试分隔符
@pytest.mark.parametrize("encoding", ["utf-16", "utf-16le", "utf-16be"])  # 参数化测试编码格式
def test_utf16_bom_skiprows(all_parsers, sep, encoding):
    # see gh-2298
    parser = all_parsers  # 使用所有定义的解析器
    data = """skip this
skip this too
A,B,C
1,2,3
4,5,6""".replace(",", sep)  # 替换逗号为指定分隔符

    path = f"__{uuid.uuid4()}__.csv"  # 创建唯一路径名
    kwargs = {"sep": sep, "skiprows": 2}  # 定义关键字参数
    utf8 = "utf-8"  # 指定UTF-8编码格式

    with tm.ensure_clean(path) as path:
        bytes_data = data.encode(encoding)  # 使用指定编码格式编码数据

        with open(path, "wb") as f:
            f.write(bytes_data)  # 写入二进制数据到文件

        with TextIOWrapper(BytesIO(data.encode(utf8)), encoding=utf8) as bytes_buffer:
            # 使用解析器读取CSV格式的数据，指定编码格式和其他参数
            result = parser.read_csv(path, encoding=encoding, **kwargs)
            expected = parser.read_csv(bytes_buffer, encoding=utf8, **kwargs)  # 使用文本字节流作为预期结果
        tm.assert_frame_equal(result, expected)  # 断言实际结果与预期结果相同


def test_utf16_example(all_parsers, csv_dir_path):
    path = os.path.join(csv_dir_path, "utf16_ex.txt")  # CSV文件路径
    parser = all_parsers  # 使用所有定义的解析器

    # 使用解析器读取CSV格式的数据，指定编码格式和分隔符
    result = parser.read_csv(path, encoding="utf-16", sep="\t")
    # 断言结果长度为50
    assert len(result) == 50


def test_unicode_encoding(all_parsers, csv_dir_path):
    path = os.path.join(csv_dir_path, "unicode_series.csv")  # CSV文件路径
    parser = all_parsers  # 使用所有定义的解析器

    # 使用解析器读取CSV格式的数据，指定头部为None和编码格式
    result = parser.read_csv(path, header=None, encoding="latin-1")
    result = result.set_index(0)  # 设置DataFrame对象的索引
    got = result[1][1632]  # 获取特定位置的值

    expected = "\xc1 k\xf6ldum klaka (Cold Fever) (1994)"  # 预期的特定字符串
    assert got == expected  # 断言获取的值与预期值相同


@pytest.mark.parametrize(
    "data,kwargs,expected",
    [
        # Basic test
        ("a\n1", {}, [1]),
        # 在没有引号的情况下，数据中包含换行符的基本测试，预期结果为 [1]
        
        # "Regular" quoting
        ('"a"\n1', {"quotechar": '"'}, [1]),
        # 在使用双引号引用时的测试，预期结果为 [1]
        
        # Test in a data row instead of header
        ("b\n1", {"names": ["a"]}, ["b", "1"]),
        # 在数据行而不是标题行中进行测试，使用指定的列名 ["a"]，预期结果为 ["b", "1"]
        
        # Test in empty data row with skipping
        ("\n1", {"names": ["a"], "skip_blank_lines": True}, [1]),
        # 在空数据行中进行测试，并跳过空行，使用指定的列名 ["a"]，预期结果为 [1]
        
        # Test in empty data row without skipping
        (
            "\n1",
            {"names": ["a"], "skip_blank_lines": False},
            [np.nan, 1],
        ),
        # 在空数据行中进行测试，不跳过空行，使用指定的列名 ["a"]，预期结果为 [NaN, 1]
    ],
# 定义一个测试函数，用于测试带有 UTF-8 BOM 的 CSV 数据解析
def test_utf8_bom(all_parsers, data, kwargs, expected):
    # 见问题 gh-4793
    # 从传入的所有解析器中选择一个
    parser = all_parsers
    # UTF-8 BOM 字符
    bom = "\ufeff"
    # UTF-8 编码
    utf8 = "utf-8"

    # 定义一个函数，用于在数据前加入 BOM 并转换为字节流
    def _encode_data_with_bom(_data):
        bom_data = (bom + _data).encode(utf8)
        return BytesIO(bom_data)

    # 如果使用的是 pyarrow 引擎，并且数据为 "\n1"，并且 skip_blank_lines 参数为 True
    if (
        parser.engine == "pyarrow"
        and data == "\n1"
        and kwargs.get("skip_blank_lines", True)
    ):
        # 跳过测试，原因是空的 CSV 文件或块：无法推断列数
        pytest.skip(reason="https://github.com/apache/arrow/issues/38676")

    # 调用解析器读取带 BOM 的数据，并指定 UTF-8 编码以及其他参数
    result = parser.read_csv(_encode_data_with_bom(data), encoding=utf8, **kwargs)
    # 期望的结果数据框架
    expected = DataFrame({"a": expected})
    # 断言结果与期望相等
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，用于测试读取 CSV 文件时的编码别名
def test_read_csv_utf_aliases(all_parsers, utf_value, encoding_fmt):
    # 见问题 gh-13549
    # 期望的结果数据框架
    expected = DataFrame({"mb_num": [4.8], "multibyte": ["test"]})
    # 从传入的所有解析器中选择一个
    parser = all_parsers

    # 根据指定的 UTF 编码值格式化编码字符串
    encoding = encoding_fmt.format(utf_value)
    # 创建包含 CSV 数据的字节流
    data = "mb_num,multibyte\n4.8,test".encode(encoding)

    # 使用解析器读取 CSV 数据，并指定编码
    result = parser.read_csv(BytesIO(data), encoding=encoding)
    # 断言结果与期望相等
    tm.assert_frame_equal(result, expected)


# 使用参数化测试不同的文件路径和编码方式来测试二进制模式下文件缓冲区的读取
@pytest.mark.parametrize(
    "file_path,encoding",
    [
        (("io", "data", "csv", "test1.csv"), "utf-8"),
        (("io", "parser", "data", "unicode_series.csv"), "latin-1"),
        (("io", "parser", "data", "sauron.SHIFT_JIS.csv"), "shiftjis"),
    ],
)
def test_binary_mode_file_buffers(all_parsers, file_path, encoding, datapath):
    # 见问题 gh-23779: Python CSV 引擎在打开二进制文件时不应报错
    # 见问题 gh-31575: Python CSV 引擎在打开原始二进制文件时不应报错
    # 从传入的所有解析器中选择一个
    parser = all_parsers

    # 获取文件路径
    fpath = datapath(*file_path)
    # 使用解析器读取 CSV 文件，并指定编码
    expected = parser.read_csv(fpath, encoding=encoding)

    # 使用 'open' 函数以指定编码打开文件，并使用解析器读取 CSV 数据
    with open(fpath, encoding=encoding) as fa:
        result = parser.read_csv(fa)
        # 断言文件未关闭
        assert not fa.closed
    # 断言结果与期望相等
    tm.assert_frame_equal(expected, result)

    # 使用 'open' 函数以二进制模式打开文件，并指定编码，使用解析器读取 CSV 数据
    with open(fpath, mode="rb") as fb:
        result = parser.read_csv(fb, encoding=encoding)
        # 断言文件未关闭
        assert not fb.closed
    # 断言结果与期望相等
    tm.assert_frame_equal(expected, result)

    # 使用 'open' 函数以无缓冲的二进制模式打开文件，并指定编码，使用解析器读取 CSV 数据
    with open(fpath, mode="rb", buffering=0) as fb:
        result = parser.read_csv(fb, encoding=encoding)
        # 断言文件未关闭
        assert not fb.closed
    # 断言结果与期望相等
    tm.assert_frame_equal(expected, result)


# 使用参数化测试不同的条件来测试编码临时文件的读取
@pytest.mark.parametrize("pass_encoding", [True, False])
def test_encoding_temp_file(
    all_parsers, utf_value, encoding_fmt, pass_encoding, temp_file
):
    # 见问题 gh-24130
    # 从传入的所有解析器中选择一个
    parser = all_parsers
    # 根据 UTF 编码值格式化编码字符串
    encoding = encoding_fmt.format(utf_value)

    # 如果使用的是 pyarrow 引擎，并且 pass_encoding 为 True，并且 UTF 值为 16 或 32
    if parser.engine == "pyarrow" and pass_encoding is True and utf_value in [16, 32]:
        # 跳过测试，原因是这些情况会冻结
        pytest.skip("These cases freeze")

    # 期望的结果数据框架
    expected = DataFrame({"foo": ["bar"]})

    # 使用临时文件对象以写入和读取模式打开文件，并指定编码
    with temp_file.open(mode="w+", encoding=encoding) as f:
        # 写入数据到临时文件
        f.write("foo\nbar")
        # 将文件指针移到文件开头
        f.seek(0)

        # 使用解析器读取临时文件中的数据，并指定编码（如果 pass_encoding 为 True）
        result = parser.read_csv(f, encoding=encoding if pass_encoding else None)
        # 断言结果与期望相等
        tm.assert_frame_equal(result, expected)


# 定义一个测试函数，用于测试读取命名的编码临时文件
def test_encoding_named_temp_file(all_parsers):
    # 见问题 gh-31819
    # 将 all_parsers 赋值给 parser 变量
    parser = all_parsers
    # 设置 encoding 变量为 "shift-jis"，表示使用 Shift-JIS 编码
    encoding = "shift-jis"

    # 设置 title 变量为 "てすと"，表示标题文本为日语
    title = "てすと"
    # 设置 data 变量为 "こむ"，表示数据文本为日语
    data = "こむ"

    # 创建预期的 DataFrame 对象，包含一个标题为 title 的列，数据为 [data]
    expected = DataFrame({title: [data]})

    # 使用 tempfile 创建一个临时文件对象 f，用于存储临时数据
    with tempfile.NamedTemporaryFile() as f:
        # 向临时文件中写入标题和数据，并使用指定的编码进行编码
        f.write(f"{title}\n{data}".encode(encoding))

        # 将文件指针移动到文件开头
        f.seek(0)

        # 使用 parser 对象读取 CSV 文件 f，指定编码为 encoding
        result = parser.read_csv(f, encoding=encoding)
        # 使用测试框架检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
        # 断言临时文件 f 没有被关闭
        assert not f.closed
@pytest.mark.parametrize(
    "encoding", ["utf-8", "utf-16", "utf-16-be", "utf-16-le", "utf-32"]
)
def test_parse_encoded_special_characters(encoding):
    # GH16218 Verify parsing of data with encoded special characters
    # Data contains a Unicode 'FULLWIDTH COLON' (U+FF1A) at position (0,"a")
    data = "a\tb\n：foo\t0\nbar\t1\nbaz\t2"  # noqa: RUF001
    # 将数据编码成指定编码的字节流
    encoded_data = BytesIO(data.encode(encoding))
    # 调用 read_csv 函数解析编码后的数据，并设置分隔符和编码方式
    result = read_csv(encoded_data, delimiter="\t", encoding=encoding)

    # 创建期望结果的 DataFrame，包含特殊字符的测试数据
    expected = DataFrame(
        data=[["：foo", 0], ["bar", 1], ["baz", 2]],  # noqa: RUF001
        columns=["a", "b"],
    )
    # 使用 assert_frame_equal 函数比较实际结果和期望结果
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("encoding", ["utf-8", None, "utf-16", "cp1255", "latin-1"])
def test_encoding_memory_map(all_parsers, encoding):
    # GH40986
    # 获取所有解析器
    parser = all_parsers
    # 创建期望的 DataFrame，包含预期的数据和列
    expected = DataFrame(
        {
            "name": ["Raphael", "Donatello", "Miguel Angel", "Leonardo"],
            "mask": ["red", "purple", "orange", "blue"],
            "weapon": ["sai", "bo staff", "nunchunk", "katana"],
        }
    )
    # 使用 tm.ensure_clean() 创建一个临时文件，将 DataFrame 写入该文件
    with tm.ensure_clean() as file:
        expected.to_csv(file, index=False, encoding=encoding)

        # 如果解析器的引擎是 "pyarrow"，则验证 memory_map=True 选项是否受支持
        if parser.engine == "pyarrow":
            msg = "The 'memory_map' option is not supported with the 'pyarrow' engine"
            # 使用 pytest.raises 检查是否抛出预期的 ValueError 异常
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(file, encoding=encoding, memory_map=True)
            return

        # 使用 memory_map=True 选项读取 CSV 文件，并获取数据帧 df
        df = parser.read_csv(file, encoding=encoding, memory_map=True)
    # 使用 assert_frame_equal 函数比较读取结果和期望结果
    tm.assert_frame_equal(df, expected)


def test_chunk_splits_multibyte_char(all_parsers):
    """
    Chunk splits a multibyte character with memory_map=True

    GH 43540
    """
    # 获取所有解析器
    parser = all_parsers
    # 定义一个包含 2048 行数据的 DataFrame，每行数据为长度为 127 的字符串 "a"
    df = DataFrame(data=["a" * 127] * 2048)

    # 在最后一行末尾添加一个两字节的 UTF-8 编码字符 "ą"
    # UTF-8 编码的 "ą" 对应字节流为 b'\xc4\x85'
    df.iloc[2047] = "a" * 127 + "ą"
    # 使用 tm.ensure_clean("bug-gh43540.csv") 创建一个临时文件，并将 DataFrame 写入该文件
    with tm.ensure_clean("bug-gh43540.csv") as fname:
        df.to_csv(fname, index=False, header=False, encoding="utf-8")

        # 如果解析器的引擎是 "pyarrow"，则验证 memory_map=True 选项是否受支持
        if parser.engine == "pyarrow":
            msg = "The 'memory_map' option is not supported with the 'pyarrow' engine"
            # 使用 pytest.raises 检查是否抛出预期的 ValueError 异常
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(fname, header=None, memory_map=True)
            return

        # 使用 memory_map=True 选项读取 CSV 文件，并获取数据帧 dfr
        dfr = parser.read_csv(fname, header=None, memory_map=True)
    # 使用 assert_frame_equal 函数比较读取结果和原始 DataFrame df
    tm.assert_frame_equal(dfr, df)


def test_readcsv_memmap_utf8(all_parsers):
    """
    GH 43787

    Test correct handling of UTF-8 chars when memory_map=True and encoding is UTF-8
    """
    lines = []
    line_length = 128
    start_char = " "
    end_char = "\U00010080"
    # This for loop creates a list of 128-char strings
    # consisting of consecutive Unicode chars
    # 使用起始字符的 ASCII 值到结束字符的 ASCII 值（不包括结束字符）的范围，以指定的行长度生成多行字符串
    for lnum in range(ord(start_char), ord(end_char), line_length):
        # 构建一行字符串，包含从 lnum 到 lnum + 0x80（128）的字符，以及换行符
        line = "".join([chr(c) for c in range(lnum, lnum + 0x80)]) + "\n"
        try:
            # 尝试将生成的行字符串编码为 UTF-8
            line.encode("utf-8")
        except UnicodeEncodeError:
            # 如果遇到 Unicode 编码错误，则跳过当前行
            continue
        # 将合法的行字符串添加到 lines 列表中
        lines.append(line)
    # 将 all_parsers 赋值给 parser 变量
    parser = all_parsers
    # 使用 lines 创建 DataFrame 对象 df
    df = DataFrame(lines)
    # 使用上下文管理器 tm.ensure_clean("utf8test.csv") 打开一个文件，并将文件名作为 fname
    with tm.ensure_clean("utf8test.csv") as fname:
        # 将 DataFrame df 写入到 CSV 文件 fname 中，不包括索引和标题行，使用 UTF-8 编码
        df.to_csv(fname, index=False, header=False, encoding="utf-8")

        # 如果 parser 的引擎为 "pyarrow"
        if parser.engine == "pyarrow":
            # 设置错误消息
            msg = "The 'memory_map' option is not supported with the 'pyarrow' engine"
            # 使用 pytest 的断言检查，预期会引发 ValueError，并且错误消息匹配 msg
            with pytest.raises(ValueError, match=msg):
                # 使用 parser.read_csv 读取 CSV 文件 fname，不包括头部行，启用内存映射，使用 UTF-8 编码
                parser.read_csv(fname, header=None, memory_map=True, encoding="utf-8")
            # 函数返回，不继续执行后续代码
            return

        # 使用 parser.read_csv 读取 CSV 文件 fname，不包括头部行，启用内存映射，使用 UTF-8 编码
        dfr = parser.read_csv(fname, header=None, memory_map=True, encoding="utf-8")
    # 使用 tm.assert_frame_equal 断言 df 和 dfr 的内容相等
    tm.assert_frame_equal(df, dfr)
# 使用 pytest 的装饰器标记，声明该测试函数使用了一个名为 pyarrow_xfail 的fixture
# 这表示测试依赖 pyarrow 库，并且会处理其预期的失败情况
@pytest.mark.usefixtures("pyarrow_xfail")
# 参数化测试，mode 参数会分别取 "w+b" 和 "w+t" 两个值进行测试
@pytest.mark.parametrize("mode", ["w+b", "w+t"])
# 定义一个名为 test_not_readable 的测试函数，用于测试不可读情况
def test_not_readable(all_parsers, mode):
    # GH43439，标识相关 GitHub issue 或问题编号
    # 从 all_parsers 中选择一个解析器进行测试
    parser = all_parsers
    # 初始化内容变量为 b"abcd"
    content = b"abcd"
    # 如果 mode 包含 "t"，则将 content 赋值为 "abcd"，即字符串类型
    if "t" in mode:
        content = "abcd"
    # 使用 SpooledTemporaryFile 创建临时文件对象 handle
    # mode 参数指定文件打开模式，encoding 指定编码格式为 utf-8
    with tempfile.SpooledTemporaryFile(mode=mode, encoding="utf-8") as handle:
        # 向 handle 写入内容
        handle.write(content)
        # 将文件指针移动到文件开头
        handle.seek(0)
        # 使用 parser 对象的 read_csv 方法读取 handle 中的内容，并赋值给 df
        df = parser.read_csv(handle)
    # 创建一个期望的 DataFrame，列名为 ["abcd"]，不含任何行数据
    expected = DataFrame([], columns=["abcd"])
    # 使用 pytest 的测试工具 tm，断言 df 和 expected 的内容是否相等
    tm.assert_frame_equal(df, expected)
```