# `D:\src\scipysrc\pandas\pandas\tests\io\parser\test_textreader.py`

```
"""
Tests the TextReader class in parsers.pyx, which
is integral to the C engine in parsers.py
"""

# 导入所需的模块和库
from io import (
    BytesIO,  # 导入BytesIO类，用于处理二进制数据
    StringIO,  # 导入StringIO类，用于处理字符串数据
)

import numpy as np  # 导入NumPy库，用于数据处理
import pytest  # 导入Pytest库，用于编写和运行测试

import pandas._libs.parsers as parser  # 导入解析器模块
from pandas._libs.parsers import TextReader  # 导入TextReader类，用于文本数据读取
from pandas.errors import ParserWarning  # 导入解析器警告类

from pandas import DataFrame  # 导入DataFrame类，用于数据表格操作
import pandas._testing as tm  # 导入测试模块

from pandas.io.parsers import (  # 导入数据解析相关模块
    TextFileReader,  # 导入文本文件读取类
    read_csv,  # 导入CSV文件读取函数
)
from pandas.io.parsers.c_parser_wrapper import ensure_dtype_objs  # 导入数据类型对象确保函数


class TestTextReader:
    @pytest.fixture
    def csv_path(self, datapath):
        return datapath("io", "data", "csv", "test1.csv")  # 返回CSV文件路径的测试数据

    def test_file_handle(self, csv_path):
        with open(csv_path, "rb") as f:  # 打开CSV文件作为二进制读取
            reader = TextReader(f)  # 创建TextReader对象以处理文件内容
            reader.read()  # 读取文件内容

    def test_file_handle_mmap(self, csv_path):
        # this was never using memory_map=True
        with open(csv_path, "rb") as f:  # 打开CSV文件作为二进制读取
            reader = TextReader(f, header=None)  # 创建TextReader对象，指定无头部信息
            reader.read()  # 读取文件内容

    def test_StringIO(self, csv_path):
        with open(csv_path, "rb") as f:  # 打开CSV文件作为二进制读取
            text = f.read()  # 读取文件内容到text变量
        src = BytesIO(text)  # 将二进制数据封装到BytesIO对象中
        reader = TextReader(src, header=None)  # 创建TextReader对象，指定无头部信息
        reader.read()  # 读取文件内容

    def test_encoding_mismatch_warning(self, csv_path):
        # GH-57954
        with open(csv_path, encoding="UTF-8") as f:  # 打开CSV文件，指定UTF-8编码
            msg = "latin1 is different from the encoding"
            with pytest.raises(ValueError, match=msg):  # 使用pytest断言检查异常消息
                read_csv(f, encoding="latin1")  # 读取CSV文件，指定Latin-1编码

    def test_string_factorize(self):
        # should this be optional?
        data = "a\nb\na\nb\na"  # 定义字符串数据
        reader = TextReader(StringIO(data), header=None)  # 创建TextReader对象处理字符串数据，无头部信息
        result = reader.read()  # 读取数据
        assert len(set(map(id, result[0]))) == 2  # 断言检查结果第一列的唯一值个数为2

    def test_skipinitialspace(self):
        data = "a,   b\na,   b\na,   b\na,   b"  # 定义包含空格的字符串数据

        reader = TextReader(StringIO(data), skipinitialspace=True, header=None)  # 创建TextReader对象，跳过起始空格，无头部信息
        result = reader.read()  # 读取数据

        tm.assert_numpy_array_equal(
            result[0], np.array(["a", "a", "a", "a"], dtype=np.object_)
        )  # 使用测试模块断言检查结果第一列是否符合预期
        tm.assert_numpy_array_equal(
            result[1], np.array(["b", "b", "b", "b"], dtype=np.object_)
        )  # 使用测试模块断言检查结果第二列是否符合预期

    def test_parse_booleans(self):
        data = "True\nFalse\nTrue\nTrue"  # 定义包含布尔值的字符串数据

        reader = TextReader(StringIO(data), header=None)  # 创建TextReader对象处理字符串数据，无头部信息
        result = reader.read()  # 读取数据

        assert result[0].dtype == np.bool_  # 断言检查结果第一列的数据类型是否为布尔类型

    def test_delimit_whitespace(self):
        data = 'a  b\na\t\t "b"\n"a"\t \t b'  # 定义包含不同分隔符的字符串数据

        reader = TextReader(StringIO(data), delim_whitespace=True, header=None)  # 创建TextReader对象，使用空白作为分隔符，无头部信息
        result = reader.read()  # 读取数据

        tm.assert_numpy_array_equal(
            result[0], np.array(["a", "a", "a"], dtype=np.object_)
        )  # 使用测试模块断言检查结果第一列是否符合预期
        tm.assert_numpy_array_equal(
            result[1], np.array(["b", "b", "b"], dtype=np.object_)
        )  # 使用测试模块断言检查结果第二列是否符合预期
    def test_embedded_newline(self):
        # 创建包含换行符的字符串数据
        data = 'a\n"hello\nthere"\nthis'

        # 使用 StringIO 将数据转换为文本读取器对象
        reader = TextReader(StringIO(data), header=None)
        # 调用 read 方法读取数据
        result = reader.read()

        # 预期的 Numpy 数组结果，包含特定的字符串元素
        expected = np.array(["a", "hello\nthere", "this"], dtype=np.object_)
        # 使用 assert_numpy_array_equal 检查结果是否与预期相同
        tm.assert_numpy_array_equal(result[0], expected)

    def test_euro_decimal(self):
        # 创建包含欧洲风格小数的字符串数据
        data = "12345,67\n345,678"

        # 使用指定的分隔符和小数点符号创建文本读取器对象
        reader = TextReader(StringIO(data), delimiter=":", decimal=",", header=None)
        # 调用 read 方法读取数据
        result = reader.read()

        # 预期的 Numpy 数组结果，包含特定的浮点数元素
        expected = np.array([12345.67, 345.678])
        # 使用 assert_almost_equal 检查结果是否几乎等于预期
        tm.assert_almost_equal(result[0], expected)

    def test_integer_thousands(self):
        # 创建包含千位分隔符的整数字符串数据
        data = "123,456\n12,500"

        # 使用指定的分隔符和千位分隔符创建文本读取器对象
        reader = TextReader(StringIO(data), delimiter=":", thousands=",", header=None)
        # 调用 read 方法读取数据
        result = reader.read()

        # 预期的 Numpy 数组结果，包含特定的整数元素
        expected = np.array([123456, 12500], dtype=np.int64)
        # 使用 assert_almost_equal 检查结果是否几乎等于预期
        tm.assert_almost_equal(result[0], expected)

    def test_integer_thousands_alt(self):
        # 创建包含替代格式的千位分隔符的整数字符串数据
        data = "123.456\n12.500"

        # 使用指定的分隔符和千位分隔符创建另一种文本读取器对象
        reader = TextFileReader(
            StringIO(data), delimiter=":", thousands=".", header=None
        )
        # 调用 read 方法读取数据
        result = reader.read()

        # 预期的 DataFrame 结果，包含特定的整数元素
        expected = DataFrame([123456, 12500])
        # 使用 assert_frame_equal 检查结果是否与预期的 DataFrame 相同
        tm.assert_frame_equal(result, expected)

    def test_skip_bad_lines(self):
        # 创建包含多余数据行的字符串数据，用于测试跳过错误行功能
        # 参见 issue #2430 了解更多详情
        data = "a:b:c\nd:e:f\ng:h:i\nj:k:l:m\nl:m:n\no:p:q:r"

        # 使用指定的分隔符创建文本读取器对象
        reader = TextReader(StringIO(data), delimiter=":", header=None)
        # 设置预期的错误信息正则表达式
        msg = r"Error tokenizing data\. C error: Expected 3 fields in line 4, saw 4"
        # 使用 pytest 检查是否引发了 ParserError，并匹配预期的错误消息
        with pytest.raises(parser.ParserError, match=msg):
            reader.read()

        # 创建另一个文本读取器对象，设置在错误行数为2时跳过
        reader = TextReader(
            StringIO(data),
            delimiter=":",
            header=None,
            on_bad_lines=2,  # Skip
        )
        # 调用 read 方法读取数据
        result = reader.read()
        # 预期的字典结果，包含特定的 Numpy 数组元素
        expected = {
            0: np.array(["a", "d", "g", "l"], dtype=object),
            1: np.array(["b", "e", "h", "m"], dtype=object),
            2: np.array(["c", "f", "i", "n"], dtype=object),
        }
        # 使用 assert_array_dicts_equal 检查结果是否与预期的字典相同
        assert_array_dicts_equal(result, expected)

        # 使用 tm.assert_produces_warning 检查是否产生了 ParserWarning 警告
        with tm.assert_produces_warning(ParserWarning, match="Skipping line"):
            reader = TextReader(
                StringIO(data),
                delimiter=":",
                header=None,
                on_bad_lines=1,  # Warn
            )
            reader.read()

    def test_header_not_enough_lines(self):
        # 创建包含标题行不足情况的字符串数据
        data = "skip this\nskip this\na,b,c\n1,2,3\n4,5,6"

        # 使用指定的分隔符和标题行数创建文本读取器对象
        reader = TextReader(StringIO(data), delimiter=",", header=2)
        # 获取读取器对象的标题行信息
        header = reader.header
        # 预期的标题行结果，包含特定的字符串列表
        expected = [["a", "b", "c"]]
        # 使用 assert 检查读取的标题行是否与预期相同
        assert header == expected

        # 调用 read 方法读取数据记录
        recs = reader.read()
        # 预期的字典结果，包含特定的 Numpy 数组元素
        expected = {
            0: np.array([1, 4], dtype=np.int64),
            1: np.array([2, 5], dtype=np.int64),
            2: np.array([3, 6], dtype=np.int64),
        }
        # 使用 assert_array_dicts_equal 检查结果是否与预期的字典相同
        assert_array_dicts_equal(recs, expected)
    # 定义一个测试方法，测试处理转义字符的功能
    def test_escapechar(self):
        # 定义测试数据，包含转义字符的字符串
        data = '\\"hello world"\n\\"hello world"\n\\"hello world"'

        # 创建一个文本读取器对象，使用指定的参数：数据源是内存中的字符串流，分隔符为逗号，无表头，转义字符为反斜杠
        reader = TextReader(StringIO(data), delimiter=",", header=None, escapechar="\\")
        
        # 调用文本读取器的读取方法
        result = reader.read()
        
        # 预期的读取结果，一个字典，键为0，值为包含三个相同字符串的 NumPy 数组
        expected = {0: np.array(['"hello world"'] * 3, dtype=object)}
        
        # 断言读取结果与预期结果相等
        assert_array_dicts_equal(result, expected)

    def test_eof_has_eol(self):
        # 处理文件末尾有换行符的情况，尚未实现具体测试内容
        pass

    def test_na_substitution(self):
        # 测试缺失值替换的功能，尚未实现具体测试内容
        pass

    def test_numpy_string_dtype(self):
        # 定义一个测试方法，测试处理 NumPy 字符串数据类型的功能
        data = """\
    @pytest.mark.parametrize(
        "text, kwargs",  # 定义参数化测试的参数：text 是测试用例的数据，kwargs 是额外的关键字参数
        [  # 参数化测试用例列表开始
            ("a,b,c\r1,2,3\r4,5,6\r7,8,9\r10,11,12", {"delimiter": ","}),  # 第一组参数化测试用例：逗号分隔的数据
            (  # 第二组参数化测试用例：空白分隔的数据
                "a  b  c\r1  2  3\r4  5  6\r7  8  9\r10  11  12",
                {"delim_whitespace": True},
            ),
            ("a,b,c\r1,2,3\r4,5,6\r,88,9\r10,11,12", {"delimiter": ","}),  # 第三组参数化测试用例：包含空行的数据
            (  # 第四组参数化测试用例：复杂的数据，包含空白行和不完整行
                (
                    "A,B,C,D,E,F,G,H,I,J,K,L,M,N,O\r"
                    "AAAAA,BBBBB,0,0,0,0,0,0,0,0,0,0,0,0,0\r"
                    ",BBBBB,0,0,0,0,0,0,0,0,0,0,0,0,0"
                ),
                {"delimiter": ","},
            ),
            ("A  B  C\r  2  3\r4  5  6", {"delim_whitespace": True}),  # 第五组参数化测试用例：带空白的数据
            ("A B C\r2 3\r4 5 6", {"delim_whitespace": True}),  # 第六组参数化测试用例：更简单的带空白的数据
        ],  # 参数化测试用例列表结束
    )
    # 定义测试方法，用于测试处理 CR 分隔符的情况
    def test_cr_delimited(self, text, kwargs):
        # 将文本中的 "\r" 替换为 "\r\n"，以处理 CR 分隔符
        nice_text = text.replace("\r", "\r\n")
        # 使用 TextReader 从字符串流中读取文本，并传入额外的参数
        result = TextReader(StringIO(text), **kwargs).read()
        # 使用处理过的文本创建 TextReader 对象，并读取文本
        expected = TextReader(StringIO(nice_text), **kwargs).read()
        # 断言两个返回的结果是否相等
        assert_array_dicts_equal(result, expected)

    # 定义测试空字段末尾的情况
    def test_empty_field_eof(self):
        # 定义包含空字段末尾的测试数据
        data = "a,b,c\n1,2,3\n4,,"

        # 使用 TextReader 读取包含指定分隔符的字符串流，并返回结果
        result = TextReader(StringIO(data), delimiter=",").read()

        # 期望的结果字典，包含各列对应的 numpy 数组
        expected = {
            0: np.array([1, 4], dtype=np.int64),
            1: np.array(["2", ""], dtype=object),
            2: np.array(["3", ""], dtype=object),
        }
        # 断言两个返回的结果是否相等
        assert_array_dicts_equal(result, expected)

    # 使用参数化测试进行多次重复的空字段末尾内存访问漏洞测试
    @pytest.mark.parametrize("repeat", range(10))
    def test_empty_field_eof_mem_access_bug(self, repeat):
        # GH5664

        # 创建 DataFrame a
        a = DataFrame([["b"], [np.nan]], columns=["a"], index=["a", "c"])
        # 创建 DataFrame b
        b = DataFrame([[1, 1, 1, 0], [1, 1, 1, 0]], columns=list("abcd"), index=[1, 1])
        # 创建 DataFrame c
        c = DataFrame(
            [
                [1, 2, 3, 4],
                [6, np.nan, np.nan, np.nan],
                [8, 9, 10, 11],
                [13, 14, np.nan, np.nan],
            ],
            columns=list("abcd"),
            index=[0, 5, 7, 12],
        )

        # 使用 read_csv 从指定的字符串流读取数据，并进行 DataFrame 比较
        df = read_csv(StringIO("a,b\nc\n"), skiprows=0, names=["a"], engine="c")
        tm.assert_frame_equal(df, a)

        # 使用 read_csv 从指定的字符串流读取数据，并进行 DataFrame 比较
        df = read_csv(
            StringIO("1,1,1,1,0\n" * 2 + "\n" * 2), names=list("abcd"), engine="c"
        )
        tm.assert_frame_equal(df, b)

        # 使用 read_csv 从指定的字符串流读取数据，并进行 DataFrame 比较
        df = read_csv(
            StringIO("0,1,2,3,4\n5,6\n7,8,9,10,11\n12,13,14"),
            names=list("abcd"),
            engine="c",
        )
        tm.assert_frame_equal(df, c)

    # 测试空的 CSV 输入情况
    def test_empty_csv_input(self):
        # GH14867

        # 使用 read_csv 从空的字符串流读取数据，使用 chunksize 和其他参数
        with read_csv(
            StringIO(), chunksize=20, header=None, names=["a", "b", "c"]
        ) as df:
            # 断言返回的对象是 TextFileReader 类型
            assert isinstance(df, TextFileReader)
# 定义一个函数，用于断言两个字典中对应键的值（数组形式）相等
def assert_array_dicts_equal(left, right):
    # 遍历左侧字典的键值对
    for k, v in left.items():
        # 使用测试工具库中的函数比较左侧键对应的值（转换为NumPy数组）与右侧键对应的值（也转换为NumPy数组）是否相等
        tm.assert_numpy_array_equal(np.asarray(v), np.asarray(right[k]))
```