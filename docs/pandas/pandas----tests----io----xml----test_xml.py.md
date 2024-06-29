# `D:\src\scipysrc\pandas\pandas\tests\io\xml\test_xml.py`

```
from __future__ import annotations
# 导入未来支持的 annotations 特性，用于类型提示的更精确定义

from io import (
    BytesIO,
    StringIO,
)
# 导入字节流和字符串流的模块，用于内存中操作数据流

from lzma import LZMAError
# 导入 LZMAError，用于处理 LZMA 解压缩错误

import os
# 导入操作系统相关的模块，用于文件路径操作等

from tarfile import ReadError
# 导入 ReadError，用于处理 tar 文件读取错误

from urllib.error import HTTPError
# 导入 HTTPError，用于处理 HTTP 请求错误

from xml.etree.ElementTree import ParseError
# 导入 ParseError，用于处理 XML 解析错误

from zipfile import BadZipFile
# 导入 BadZipFile，用于处理 ZIP 文件错误

import numpy as np
# 导入 NumPy 库，用于科学计算

import pytest
# 导入 pytest，用于编写和运行测试用例

from pandas.compat import WASM
# 导入 WASM，兼容性库

from pandas.compat._optional import import_optional_dependency
# 导入 import_optional_dependency，用于导入可选依赖

from pandas.errors import (
    EmptyDataError,
    ParserError,
)
# 导入异常类 EmptyDataError 和 ParserError，用于处理数据解析时的异常情况

import pandas.util._test_decorators as td
# 导入测试装饰器，用于测试代码

import pandas as pd
# 导入 pandas 库

from pandas import (
    NA,
    DataFrame,
    Series,
)
# 导入 pandas 中常用的类和对象：NA、DataFrame、Series

import pandas._testing as tm
# 导入 pandas 测试模块

from pandas.core.arrays import (
    ArrowStringArray,
    StringArray,
)
# 导入 ArrowStringArray 和 StringArray，用于处理字符串数组

from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
# 导入 ArrowStringArrayNumpySemantics，处理 Arrow 字符串数组的 NumPy 语义

from pandas.io.common import get_handle
# 导入 get_handle，用于获取文件句柄

from pandas.io.xml import read_xml
# 导入 read_xml，用于读取 XML 文件

# CHECK LIST

# [x] - ValueError: "Values for parser can only be lxml or etree."
# etree
# [X] - ImportError: "lxml not found, please install or use the etree parser."
# [X] - TypeError: "expected str, bytes or os.PathLike object, not NoneType"
# [X] - ValueError: "Either element or attributes can be parsed not both."
# [X] - ValueError: "xpath does not return any nodes..."
# [X] - SyntaxError: "You have used an incorrect or unsupported XPath"
# [X] - ValueError: "names does not match length of child elements in xpath."
# [X] - TypeError: "...is not a valid type for names"
# [X] - ValueError: "To use stylesheet, you need lxml installed..."
# []  - URLError: (GENERAL ERROR WITH HTTPError AS SUBCLASS)
# [X] - HTTPError: "HTTP Error 404: Not Found"
# []  - OSError: (GENERAL ERROR WITH FileNotFoundError AS SUBCLASS)
# [X] - FileNotFoundError: "No such file or directory"
# []  - ParseError    (FAILSAFE CATCH ALL FOR VERY COMPLEX XML)
# [X] - UnicodeDecodeError: "'utf-8' codec can't decode byte 0xe9..."
# [X] - UnicodeError: "UTF-16 stream does not start with BOM"
# [X] - BadZipFile: "File is not a zip file"
# [X] - OSError: "Invalid data stream"
# [X] - LZMAError: "Input format not supported by decoder"
# [X] - ValueError: "Unrecognized compression type"
# [X] - PermissionError: "Forbidden"

# lxml
# [X] - ValueError: "Either element or attributes can be parsed not both."
# [X] - AttributeError: "__enter__"
# [X] - XSLTApplyError: "Cannot resolve URI"
# [X] - XSLTParseError: "document is not a stylesheet"
# [X] - ValueError: "xpath does not return any nodes."
# [X] - XPathEvalError: "Invalid expression"
# []  - XPathSyntaxError: (OLD VERSION IN lxml FOR XPATH ERRORS)
# [X] - TypeError: "empty namespace prefix is not supported in XPath"
# [X] - ValueError: "names does not match length of child elements in xpath."
# [X] - TypeError: "...is not a valid type for names"
# [X] - LookupError: "unknown encoding"
# []  - URLError: (USUALLY DUE TO NETWORKING)
# [X  - HTTPError: "HTTP Error 404: Not Found"
# [X] - OSError: "failed to load external entity"
# [X] - XMLSyntaxError: "Start tag expected, '<' not found"
# 创建一个包含几何形状信息的DataFrame，包括形状名称、角度、和边数
geom_df = DataFrame(
    {
        "shape": ["square", "circle", "triangle"],
        "degrees": [360, 360, 180],
        "sides": [4, np.nan, 3],
    }
)

# 默认命名空间的 XML 数据字符串
xml_default_nmsp = """\
<?xml version='1.0' encoding='utf-8'?>
<data xmlns="http://example.com">
  <row>
    <shape>square</shape>
    <degrees>360</degrees>
    <sides>4</sides>
  </row>
  <row>
    <shape>circle</shape>
    <degrees>360</degrees>
    <sides/>
  </row>
  <row>
    <shape>triangle</shape>
    <degrees>180</degrees>
    <sides>3</sides>
  </row>
</data>"""

# 带有前缀命名空间的 XML 数据字符串
xml_prefix_nmsp = """\
<?xml version='1.0' encoding='utf-8'?>
<doc:data xmlns:doc="http://example.com">
  <doc:row>
    <doc:shape>square</doc:shape>
    <doc:degrees>360</doc:degrees>
    <doc:sides>4.0</doc:sides>
  </doc:row>
  <doc:row>
    <doc:shape>circle</doc:shape>
    <doc:degrees>360</doc:degrees>
    <doc:sides/>
  </doc:row>
  <doc:row>
    <doc:shape>triangle</doc:shape>
    <doc:degrees>180</doc:degrees>
    <doc:sides>3.0</doc:sides>
  </doc:row>
</doc:data>"""

# 用于测试的 KML 数据的空 DataFrame
df_kml = DataFrame(
    {
        # Empty DataFrame for KML data
    }
)


# 测试函数：测试使用字面量 XML 数据时是否引发异常
def test_literal_xml_raises():
    # GH 53809
    # 检查是否导入了 pytest，如果没有则跳过测试
    pytest.importorskip("lxml")
    # 匹配错误信息中可能出现的两种情况
    msg = "|".join([r".*No such file or directory", r".*Invalid argument"])

    # 使用 pytest 检查是否会引发 FileNotFoundError 或 OSError 异常，并匹配特定消息
    with pytest.raises((FileNotFoundError, OSError), match=msg):
        read_xml(xml_default_nmsp)


# 为测试设置模式的 pytest fixture
@pytest.fixture(params=["rb", "r"])
def mode(request):
    return request.param


# 为解析器设置 pytest fixture，支持 lxml 和 etree
@pytest.fixture(params=[pytest.param("lxml", marks=td.skip_if_no("lxml")), "etree"])
def parser(request):
    return request.param


# 使用 iterparse 方式读取 XML 数据的函数
def read_xml_iterparse(data, **kwargs):
    # 确保使用临时文件路径进行清理
    with tm.ensure_clean() as path:
        # 将数据写入临时文件
        with open(path, "w", encoding="utf-8") as f:
            f.write(data)
        # 调用 read_xml 函数处理临时文件
        return read_xml(path, **kwargs)


# 使用 iterparse 方式读取压缩 XML 数据的函数
def read_xml_iterparse_comp(comp_path, compression_only, **kwargs):
    # 使用 get_handle 获取处理器句柄，以只读方式打开压缩文件
    with get_handle(comp_path, "r", compression=compression_only) as handles:
        # 确保使用临时文件路径进行清理
        with tm.ensure_clean() as path:
            # 将处理后的数据写入临时文件
            with open(path, "w", encoding="utf-8") as f:
                f.write(handles.handle.read())
            # 调用 read_xml 函数处理临时文件
            return read_xml(path, **kwargs)


# 文件/URL 相关的测试函数

# 测试文件解析器的一致性
def test_parser_consistency_file(xml_books):
    # 检查是否导入了 lxml，如果没有则跳过测试
    pytest.importorskip("lxml")
    # 使用 lxml 解析器读取 XML 数据，生成 DataFrame
    df_file_lxml = read_xml(xml_books, parser="lxml")
    # 使用 etree 解析器读取 XML 数据，生成 DataFrame
    df_file_etree = read_xml(xml_books, parser="etree")

    # 使用 lxml 解析器进行迭代解析 XML 数据，指定迭代方式
    df_iter_lxml = read_xml(
        xml_books,
        parser="lxml",
        iterparse={"book": ["category", "title", "year", "author", "price"]},
    )
    # 调用 read_xml 函数，解析 xml_books 文件，使用 etree 解析器，指定 iterparse 参数以提取指定的 XML 元素
    df_iter_etree = read_xml(
        xml_books,
        parser="etree",
        iterparse={"book": ["category", "title", "year", "author", "price"]},
    )

    # 使用测试框架 tm 进行 DataFrame 对象 df_file_lxml 和 df_file_etree 的内容比较
    tm.assert_frame_equal(df_file_lxml, df_file_etree)
    
    # 使用测试框架 tm 进行 DataFrame 对象 df_file_lxml 和 df_iter_lxml 的内容比较
    tm.assert_frame_equal(df_file_lxml, df_iter_lxml)
    
    # 使用测试框架 tm 进行 DataFrame 对象 df_iter_lxml 和 df_iter_etree 的内容比较
    tm.assert_frame_equal(df_iter_lxml, df_iter_etree)
# 标记此函数为网络相关测试
# 标记此函数为单CPU测试
@pytest.mark.network
@pytest.mark.single_cpu
def test_parser_consistency_url(parser, httpserver):
    # 使用 HTTP 服务器提供的 XML 内容进行测试
    httpserver.serve_content(content=xml_default_nmsp)

    # 使用 read_xml 函数以字符串输入方式解析 XML 数据，返回数据帧 df_xpath
    df_xpath = read_xml(StringIO(xml_default_nmsp), parser=parser)
    
    # 使用 read_xml 函数以字节流输入方式解析 XML 数据，设置迭代解析参数，返回数据帧 df_iter
    df_iter = read_xml(
        BytesIO(xml_default_nmsp.encode()),
        parser=parser,
        iterparse={"row": ["shape", "degrees", "sides"]},
    )

    # 使用测试工具比较两个数据帧是否相等
    tm.assert_frame_equal(df_xpath, df_iter)


# 测试文件式输入的 XML 解析
def test_file_like(xml_books, parser, mode):
    # 打开 XML 文件或流对象，根据模式选择编码方式
    with open(xml_books, mode, encoding="utf-8" if mode == "r" else None) as f:
        # 使用 read_xml 函数解析 XML 数据，返回数据帧 df_file
        df_file = read_xml(f, parser=parser)

    # 创建预期的数据帧 df_expected，用于与 df_file 进行比较
    df_expected = DataFrame(
        {
            "category": ["cooking", "children", "web"],
            "title": ["Everyday Italian", "Harry Potter", "Learning XML"],
            "author": ["Giada De Laurentiis", "J K. Rowling", "Erik T. Ray"],
            "year": [2005, 2005, 2003],
            "price": [30.00, 29.99, 39.95],
        }
    )

    # 使用测试工具比较两个数据帧是否相等
    tm.assert_frame_equal(df_file, df_expected)


# 测试文件 IO 的 XML 解析
def test_file_io(xml_books, parser, mode):
    # 打开 XML 文件或流对象，根据模式选择编码方式
    with open(xml_books, mode, encoding="utf-8" if mode == "r" else None) as f:
        # 读取 XML 数据到 xml_obj 变量
        xml_obj = f.read()

    # 使用 read_xml 函数解析 XML 数据，返回数据帧 df_io
    df_io = read_xml(
        (BytesIO(xml_obj) if isinstance(xml_obj, bytes) else StringIO(xml_obj)),
        parser=parser,
    )

    # 创建预期的数据帧 df_expected，用于与 df_io 进行比较
    df_expected = DataFrame(
        {
            "category": ["cooking", "children", "web"],
            "title": ["Everyday Italian", "Harry Potter", "Learning XML"],
            "author": ["Giada De Laurentiis", "J K. Rowling", "Erik T. Ray"],
            "year": [2005, 2005, 2003],
            "price": [30.00, 29.99, 39.95],
        }
    )

    # 使用测试工具比较两个数据帧是否相等
    tm.assert_frame_equal(df_io, df_expected)


# 测试带缓冲读取器的字符串 XML 解析
def test_file_buffered_reader_string(xml_books, parser, mode):
    # 打开 XML 文件或流对象，根据模式选择编码方式
    with open(xml_books, mode, encoding="utf-8" if mode == "r" else None) as f:
        # 读取 XML 数据到 xml_obj 变量
        xml_obj = f.read()

    # 根据模式处理 xml_obj，若为字节流则解码为字符串流，否则直接使用字符串流
    if mode == "rb":
        xml_obj = StringIO(xml_obj.decode())
    elif mode == "r":
        xml_obj = StringIO(xml_obj)

    # 使用 read_xml 函数解析 XML 数据，返回数据帧 df_str
    df_str = read_xml(xml_obj, parser=parser)

    # 创建预期的数据帧 df_expected，用于与 df_str 进行比较
    df_expected = DataFrame(
        {
            "category": ["cooking", "children", "web"],
            "title": ["Everyday Italian", "Harry Potter", "Learning XML"],
            "author": ["Giada De Laurentiis", "J K. Rowling", "Erik T. Ray"],
            "year": [2005, 2005, 2003],
            "price": [30.00, 29.99, 39.95],
        }
    )

    # 使用测试工具比较两个数据帧是否相等
    tm.assert_frame_equal(df_str, df_expected)


# 测试带缓冲读取器且无 XML 声明的 XML 解析
def test_file_buffered_reader_no_xml_declaration(xml_books, parser, mode):
    # 打开 XML 文件或流对象，根据模式选择编码方式
    with open(xml_books, mode, encoding="utf-8" if mode == "r" else None) as f:
        # 跳过第一行，读取其余 XML 数据到 xml_obj 变量
        next(f)
        xml_obj = f.read()

    # 根据模式处理 xml_obj，若为字节流则解码为字符串流，否则直接使用字符串流
    if mode == "rb":
        xml_obj = StringIO(xml_obj.decode())
    elif mode == "r":
        xml_obj = StringIO(xml_obj)

    # 使用 read_xml 函数解析 XML 数据，返回数据帧 df_str
    df_str = read_xml(xml_obj, parser=parser)
    # 创建一个预期的数据框，包含几列：category、title、author、year、price
    df_expected = DataFrame(
        {
            "category": ["cooking", "children", "web"],  # 类别列包括烹饪、儿童、网络
            "title": ["Everyday Italian", "Harry Potter", "Learning XML"],  # 标题列包括每日意大利菜、哈利波特、学习 XML
            "author": ["Giada De Laurentiis", "J K. Rowling", "Erik T. Ray"],  # 作者列包括吉娅达·德·洛伦蒂斯、J.K.罗琳、埃里克·雷
            "year": [2005, 2005, 2003],  # 出版年份列包括2005、2005、2003
            "price": [30.00, 29.99, 39.95],  # 价格列包括30.00、29.99、39.95
        }
    )

    # 使用测试工具（assert_frame_equal）比较数据框 df_str 和预期的数据框 df_expected 是否相等
    tm.assert_frame_equal(df_str, df_expected)
# 定义测试函数，用于测试处理包含中文字符集的 XML 字符串的功能
def test_string_charset(parser):
    # 定义包含中文字符的 XML 字符串
    txt = "<中文標籤><row><c1>1</c1><c2>2</c2></row></中文標籤>"
    
    # 调用 read_xml 函数解析字符串，并返回数据框 df_str
    df_str = read_xml(StringIO(txt), parser=parser)
    
    # 创建预期的数据框 df_expected，包含列"c1"和"c2"，每列一个值
    df_expected = DataFrame({"c1": 1, "c2": 2}, index=[0])
    
    # 使用 pandas 的 assert_frame_equal 函数比较 df_str 和 df_expected，确认它们相等
    tm.assert_frame_equal(df_str, df_expected)


# 定义测试函数，用于测试处理包含中文字符集的 XML 文件的功能
def test_file_charset(xml_doc_ch_utf, parser):
    # 调用 read_xml 函数解析 XML 文件 xml_doc_ch_utf，并返回数据框 df_file
    df_file = read_xml(xml_doc_ch_utf, parser=parser)
    
    # 创建预期的数据框 df_expected，包含列"問"、"答"和"a"，每列包含多行文本
    df_expected = DataFrame(
        {
            "問": [
                "問  若箇是邪而言破邪 何者是正而道(Sorry, this is Big5 only)申正",
                "問 既破有得申無得 亦應但破性執申假名以不",
                "問 既破性申假 亦應但破有申無 若有無兩洗 亦應性假雙破耶",
            ],
            "答": [
                "".join(
                    [
                        "答  邪既無量 正亦多途  大略為言不出二種 謂",
                        "有得與無得 有得是邪須破 無得是正須申\n\t\t故",
                    ]
                ),
                None,
                "答  不例  有無皆是性 所以須雙破 既分性假異 故有破不破",
            ],
            "a": [
                None,
                "答 性執是有得 假名是無得  今破有得申無得 即是破性執申假名也",
                None,
            ],
        }
    )
    
    # 使用 pandas 的 assert_frame_equal 函数比较 df_file 和 df_expected，确认它们相等
    tm.assert_frame_equal(df_file, df_expected)


# 定义测试函数，用于测试处理包含文件句柄的 XML 数据的功能
def test_file_handle_close(xml_books, parser):
    # 打开 XML 文件 xml_books，并将其内容读取到 BytesIO 对象中，再传递给 read_xml 函数解析
    with open(xml_books, "rb") as f:
        read_xml(BytesIO(f.read()), parser=parser)
        
        # 断言文件句柄 f 没有关闭
        assert not f.closed


# 定义测试函数，用于测试处理空字符串的情况（使用 lxml 解析器）
@pytest.mark.parametrize("val", ["", b""])
def test_empty_string_lxml(val):
    # 导入 pytest 的 lxml.etree 模块
    lxml_etree = pytest.importorskip("lxml.etree")
    
    # 设置错误消息 msg，用于匹配异常信息
    msg = "|".join(
        [
            "Document is empty",
            r"None \(line 0\)",  # 在 Mac 上使用 lxml 4.91 时看到的消息
        ]
    )
    
    # 根据输入值类型，创建相应的数据流对象 data
    if isinstance(val, str):
        data = StringIO(val)
    else:
        data = BytesIO(val)
    
    # 使用 pytest.raises 检查是否抛出预期的 lxml_etree.XMLSyntaxError 异常，匹配错误消息 msg
    with pytest.raises(lxml_etree.XMLSyntaxError, match=msg):
        read_xml(data, parser="lxml")


# 定义测试函数，用于测试处理空字符串的情况（使用 etree 解析器）
@pytest.mark.parametrize("val", ["", b""])
def test_empty_string_etree(val):
    # 根据输入值类型，创建相应的数据流对象 data
    if isinstance(val, str):
        data = StringIO(val)
    else:
        data = BytesIO(val)
    
    # 使用 pytest.raises 检查是否抛出预期的 ParseError 异常，匹配错误消息 "no element found"
    with pytest.raises(ParseError, match="no element found"):
        read_xml(data, parser="etree")


# 定义测试函数，用于测试处理错误的文件路径的情况
@pytest.mark.skipif(WASM, reason="limited file system access on WASM")
def test_wrong_file_path(parser):
    # 定义一个不存在的文件路径 filename
    filename = os.path.join("does", "not", "exist", "books.xml")
    
    # 使用 pytest.raises 检查是否抛出预期的 FileNotFoundError 异常，匹配错误消息 "No such file or directory"
    with pytest.raises(
        FileNotFoundError, match=r"\[Errno 2\] No such file or directory"
    ):
        read_xml(filename, parser=parser)


# 定义测试函数，用于测试处理 URL 访问的情况
@pytest.mark.network
@pytest.mark.single_cpu
def test_url(httpserver, xml_file):
    # 导入 lxml 库，跳过如果 lxml 未安装
    pytest.importorskip("lxml")
    
    # 打开 XML 文件 xml_file，并将其内容加载到 httpserver 中以进行模拟服务器内容
    with open(xml_file, encoding="utf-8") as f:
        httpserver.serve_content(content=f.read())
        
        # 调用 read_xml 函数从 httpserver 的 URL 中读取 XML 数据，并选取路径 ".//book[count(*)=4]"
        df_url = read_xml(httpserver.url, xpath=".//book[count(*)=4]")
    
    # 创建预期的数据框 df_expected，包含列"category"、"title"、"author"、"year"和"price"，每列多个值
    df_expected = DataFrame(
        {
            "category": ["cooking", "children", "web"],
            "title": ["Everyday Italian", "Harry Potter", "Learning XML"],
            "author": ["Giada De Laurentiis", "J K. Rowling", "Erik T. Ray"],
            "year": [2005, 2005, 2003],
            "price": [30.00, 29.99, 39.95],
        }
    )
    # 使用测试框架中的断言函数来比较两个数据帧 df_url 和 df_expected 是否相等
    tm.assert_frame_equal(df_url, df_expected)
# 为测试函数标记网络相关，用于pytest的标记
@pytest.mark.network
# 为测试函数标记单CPU执行，用于pytest的标记
@pytest.mark.single_cpu
# 定义测试函数，测试处理错误的URL情况
def test_wrong_url(parser, httpserver):
    # 设置HTTP服务器响应内容为"NOT FOUND"并返回404状态码
    httpserver.serve_content("NOT FOUND", code=404)
    # 使用pytest检查是否抛出HTTPError异常，并匹配特定错误信息
    with pytest.raises(HTTPError, match=("HTTP Error 404: NOT FOUND")):
        # 调用read_xml函数，尝试读取不存在的URL，预期抛出404错误
        read_xml(httpserver.url, xpath=".//book[count(*)=4]", parser=parser)


# CONTENT


# 定义测试函数，测试处理XML中的空白符情况
def test_whitespace(parser):
    # 定义包含XML数据的字符串
    xml = """
      <data>
        <row sides=" 4 ">
          <shape>
              square
          </shape>
          <degrees>&#009;360&#009;</degrees>
        </row>
        <row sides=" 0 ">
          <shape>
              circle
          </shape>
          <degrees>&#009;360&#009;</degrees>
        </row>
        <row sides=" 3 ">
          <shape>
              triangle
          </shape>
          <degrees>&#009;180&#009;</degrees>
        </row>
      </data>"""

    # 使用read_xml函数读取XML字符串，返回DataFrame对象，用于XPath的处理
    df_xpath = read_xml(StringIO(xml), parser=parser, dtype="string")

    # 使用read_xml_iterparse函数迭代解析XML字符串，返回DataFrame对象
    df_iter = read_xml_iterparse(
        xml,
        parser=parser,
        iterparse={"row": ["sides", "shape", "degrees"]},
        dtype="string",
    )

    # 预期的DataFrame对象，包含XML中的数据和空白符
    df_expected = DataFrame(
        {
            "sides": [" 4 ", " 0 ", " 3 "],
            "shape": [
                "\n              square\n          ",
                "\n              circle\n          ",
                "\n              triangle\n          ",
            ],
            "degrees": ["\t360\t", "\t360\t", "\t180\t"],
        },
        dtype="string",
    )

    # 使用pandas的测试工具tm，比较预期DataFrame和实际结果DataFrame，确认数据一致性
    tm.assert_frame_equal(df_xpath, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


# XPATH


# 定义测试函数，测试使用错误的XPath表达式情况（使用lxml解析器）
def test_empty_xpath_lxml(xml_books):
    # 确保导入lxml模块成功，否则跳过测试
    pytest.importorskip("lxml")
    # 使用pytest检查是否抛出值错误，匹配特定错误信息，说明XPath未返回任何节点
    with pytest.raises(ValueError, match=("xpath does not return any nodes")):
        # 调用read_xml函数，尝试使用不存在的XPath查询
        read_xml(xml_books, xpath=".//python", parser="lxml")


# 定义测试函数，测试使用错误的XPath表达式情况（使用etree解析器）
def test_bad_xpath_etree(xml_books):
    # 使用pytest检查是否抛出语法错误，匹配特定错误信息，说明XPath表达式错误
    with pytest.raises(
        SyntaxError, match=("You have used an incorrect or unsupported XPath")
    ):
        # 调用read_xml函数，尝试使用错误的XPath表达式查询
        read_xml(xml_books, xpath=".//[book]", parser="etree")


# 定义测试函数，测试使用错误的XPath表达式情况（使用lxml解析器）
def test_bad_xpath_lxml(xml_books):
    # 确保导入lxml模块的etree子模块成功，否则跳过测试
    lxml_etree = pytest.importorskip("lxml.etree")

    # 使用pytest检查是否抛出XPath解析错误，匹配特定错误信息，说明XPath表达式无效
    with pytest.raises(lxml_etree.XPathEvalError, match=("Invalid expression")):
        # 调用read_xml函数，尝试使用无效的XPath表达式查询
        read_xml(xml_books, xpath=".//[book]", parser="lxml")


# NAMESPACE


# 定义测试函数，测试使用默认命名空间的XPath查询情况
def test_default_namespace(parser):
    # 使用read_xml函数读取包含默认命名空间的XML数据，返回DataFrame对象
    df_nmsp = read_xml(
        StringIO(xml_default_nmsp),
        xpath=".//ns:row",
        namespaces={"ns": "http://example.com"},
        parser=parser,
    )

    # 使用read_xml_iterparse函数迭代解析包含默认命名空间的XML数据，返回DataFrame对象
    df_iter = read_xml_iterparse(
        xml_default_nmsp,
        parser=parser,
        iterparse={"row": ["shape", "degrees", "sides"]},
    )

    # 预期的DataFrame对象，包含通过默认命名空间解析的数据
    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": [360, 360, 180],
            "sides": [4.0, float("nan"), 3.0],
        }
    )

    # 使用pandas的测试工具tm，比较预期DataFrame和实际结果DataFrame，确认数据一致性
    tm.assert_frame_equal(df_nmsp, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


# 定义测试函数，测试使用带前缀的命名空间的XPath查询情况
def test_prefix_namespace(parser):
    # 使用指定的函数read_xml从XML数据中读取数据框架df_nmsp。
    df_nmsp = read_xml(
        StringIO(xml_prefix_nmsp),  # 将XML前缀数据转换为文本输入流
        xpath=".//doc:row",  # 使用XPath选择所有名为doc:row的元素
        namespaces={"doc": "http://example.com"},  # 设置命名空间，使得XPath中的doc前缀映射到指定的命名空间URI
        parser=parser,  # 使用给定的XML解析器解析XML数据
    )
    
    # 使用read_xml_iterparse函数从XML数据中迭代解析数据框架df_iter。
    df_iter = read_xml_iterparse(
        xml_prefix_nmsp,  # XML数据的前缀部分
        parser=parser,  # 使用指定的XML解析器解析XML数据
        iterparse={"row": ["shape", "degrees", "sides"]}  # 配置迭代解析器，指定要提取的元素及其属性
    )
    
    # 创建预期的数据框架df_expected，包含形状、度数和边数的数据。
    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],  # 形状数据列
            "degrees": [360, 360, 180],  # 度数数据列
            "sides": [4.0, float("nan"), 3.0],  # 边数数据列，包括NaN值
        }
    )
    
    # 使用测试工具tm.assert_frame_equal比较df_nmsp和df_expected，确保它们相等。
    tm.assert_frame_equal(df_nmsp, df_expected)
    
    # 使用测试工具tm.assert_frame_equal比较df_iter和df_expected，确保它们相等。
    tm.assert_frame_equal(df_iter, df_expected)
# 测试默认命名空间一致性
def test_consistency_default_namespace():
    # 导入 pytest 并跳过没有 lxml 的情况
    pytest.importorskip("lxml")
    # 使用 lxml 解析器读取默认命名空间中的 XML 数据
    df_lxml = read_xml(
        StringIO(xml_default_nmsp),
        xpath=".//ns:row",  # 使用 XPath 定位到命名空间为 ns 的行
        namespaces={"ns": "http://example.com"},  # 定义命名空间 ns
        parser="lxml",
    )

    # 使用 etree 解析器读取默认命名空间中的 XML 数据
    df_etree = read_xml(
        StringIO(xml_default_nmsp),
        xpath=".//doc:row",  # 使用 XPath 定位到命名空间为 doc 的行
        namespaces={"doc": "http://example.com"},  # 定义命名空间 doc
        parser="etree",
    )

    # 断言两个 DataFrame 对象的内容是否相等
    tm.assert_frame_equal(df_lxml, df_etree)


# 测试带前缀命名空间一致性
def test_consistency_prefix_namespace():
    # 导入 pytest 并跳过没有 lxml 的情况
    pytest.importorskip("lxml")
    # 使用 lxml 解析器读取带有前缀命名空间的 XML 数据
    df_lxml = read_xml(
        StringIO(xml_prefix_nmsp),
        xpath=".//doc:row",  # 使用 XPath 定位到命名空间为 doc 的行
        namespaces={"doc": "http://example.com"},  # 定义命名空间 doc
        parser="lxml",
    )

    # 使用 etree 解析器读取带有前缀命名空间的 XML 数据
    df_etree = read_xml(
        StringIO(xml_prefix_nmsp),
        xpath=".//doc:row",  # 使用 XPath 定位到命名空间为 doc 的行
        namespaces={"doc": "http://example.com"},  # 定义命名空间 doc
        parser="etree",
    )

    # 断言两个 DataFrame 对象的内容是否相等
    tm.assert_frame_equal(df_lxml, df_etree)


# 测试默认命名空间下缺少前缀的情况
def test_missing_prefix_with_default_namespace(xml_books, parser):
    # 使用 pytest 断言在 XPath 表达式中缺少前缀时会抛出 ValueError 异常
    with pytest.raises(ValueError, match=("xpath does not return any nodes")):
        read_xml(xml_books, xpath=".//Placemark", parser=parser)


# 测试在 etree 解析器下缺少前缀定义的情况
def test_missing_prefix_definition_etree(kml_cta_rail_lines):
    # 使用 pytest 断言在 XPath 表达式中使用未声明的命名空间前缀时会抛出 SyntaxError 异常
    with pytest.raises(SyntaxError, match=("you used an undeclared namespace prefix")):
        read_xml(kml_cta_rail_lines, xpath=".//kml:Placemark", parser="etree")


# 测试在 lxml 解析器下缺少前缀定义的情况
def test_missing_prefix_definition_lxml(kml_cta_rail_lines):
    # 导入 lxml.etree 模块，跳过没有 lxml 的情况
    lxml_etree = pytest.importorskip("lxml.etree")
    # 使用 pytest 断言在 XPath 表达式中使用未定义的命名空间前缀时会抛出 lxml.etree.XPathEvalError 异常
    with pytest.raises(lxml_etree.XPathEvalError, match=("Undefined namespace prefix")):
        read_xml(kml_cta_rail_lines, xpath=".//kml:Placemark", parser="lxml")


# 测试空命名空间前缀的情况
@pytest.mark.parametrize("key", ["", None])
def test_none_namespace_prefix(key):
    # 导入 pytest 并跳过没有 lxml 的情况
    pytest.importorskip("lxml")
    # 使用 pytest 断言在 XPath 表达式中使用空的命名空间前缀会抛出 TypeError 异常
    with pytest.raises(
        TypeError, match=("empty namespace prefix is not supported in XPath")
    ):
        read_xml(
            StringIO(xml_default_nmsp),
            xpath=".//kml:Placemark",  # 使用 XPath 定位到命名空间为 kml 的 Placemark 元素
            namespaces={key: "http://www.opengis.net/kml/2.2"},  # 定义空命名空间前缀和其对应的命名空间
            parser="lxml",
        )


# 测试文件中的元素和属性
def test_file_elems_and_attrs(xml_books, parser):
    # 使用指定解析器解析 XML 文件并获取 DataFrame 对象
    df_file = read_xml(xml_books, parser=parser)
    # 使用指定解析器和迭代器设置解析 XML 文件并获取 DataFrame 对象
    df_iter = read_xml(
        xml_books,
        parser=parser,
        iterparse={"book": ["category", "title", "author", "year", "price"]},
    )
    # 期望的 DataFrame 对象
    df_expected = DataFrame(
        {
            "category": ["cooking", "children", "web"],
            "title": ["Everyday Italian", "Harry Potter", "Learning XML"],
            "author": ["Giada De Laurentiis", "J K. Rowling", "Erik T. Ray"],
            "year": [2005, 2005, 2003],
            "price": [30.00, 29.99, 39.95],
        }
    )

    # 断言解析后的 DataFrame 对象与期望的 DataFrame 对象内容是否相等
    tm.assert_frame_equal(df_file, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


# 测试文件中仅有属性的情况
def test_file_only_attrs(xml_books, parser):
    # 使用指定解析器解析 XML 文件并仅获取属性组成的 DataFrame 对象
    df_file = read_xml(xml_books, attrs_only=True, parser=parser)
    # 使用指定解析器和迭代器设置解析 XML 文件并仅获取属性组成的 DataFrame 对象
    df_iter = read_xml(xml_books, parser=parser, iterparse={"book": ["category"]})
    # 创建一个预期的 DataFrame 对象，包含一个名为 "category" 的列，列中包含三个字符串元素 "cooking", "children", "web"
    df_expected = DataFrame({"category": ["cooking", "children", "web"]})
    
    # 使用 pandas.testing 库中的 assert_frame_equal 函数比较两个 DataFrame 对象 df_file 和 df_expected 是否相等
    tm.assert_frame_equal(df_file, df_expected)
    
    # 使用 pandas.testing 库中的 assert_frame_equal 函数比较两个 DataFrame 对象 df_iter 和 df_expected 是否相等
    tm.assert_frame_equal(df_iter, df_expected)
# 定义一个函数，用于测试仅包含元素的 XML 文件解析结果是否符合预期
def test_file_only_elems(xml_books, parser):
    # 调用 read_xml 函数，仅解析元素而不解析属性，返回数据框 df_file
    df_file = read_xml(xml_books, elems_only=True, parser=parser)
    
    # 调用 read_xml 函数，使用迭代解析模式，指定要解析的元素及其子元素列表，返回数据框 df_iter
    df_iter = read_xml(
        xml_books,
        parser=parser,
        iterparse={"book": ["title", "author", "year", "price"]},
    )
    
    # 创建预期的数据框 df_expected，包含书籍的标题、作者、年份和价格信息
    df_expected = DataFrame(
        {
            "title": ["Everyday Italian", "Harry Potter", "Learning XML"],
            "author": ["Giada De Laurentiis", "J K. Rowling", "Erik T. Ray"],
            "year": [2005, 2005, 2003],
            "price": [30.00, 29.99, 39.95],
        }
    )

    # 使用 pytest 中的 assert_frame_equal 函数比较 df_file 和 df_expected，确保它们相等
    tm.assert_frame_equal(df_file, df_expected)
    
    # 使用 pytest 中的 assert_frame_equal 函数比较 df_iter 和 df_expected，确保它们相等
    tm.assert_frame_equal(df_iter, df_expected)


# 定义一个函数，测试仅包含元素和属性的 XML 文件解析时是否会引发 ValueError 异常
def test_elem_and_attrs_only(kml_cta_rail_lines, parser):
    # 使用 pytest 的 raises 函数检查是否会引发 ValueError 异常，并验证异常消息是否匹配
    with pytest.raises(
        ValueError,
        match=("Either element or attributes can be parsed not both"),
    ):
        read_xml(kml_cta_rail_lines, elems_only=True, attrs_only=True, parser=parser)


# 定义一个函数，测试空属性解析时是否会引发 ValueError 异常
def test_empty_attrs_only(parser):
    # 定义一个 XML 字符串 xml，包含多个 <row> 元素，每个元素都有一个 shape 属性
    xml = """
      <data>
        <row>
          <shape sides="4">square</shape>
          <degrees>360</degrees>
        </row>
        <row>
          <shape sides="0">circle</shape>
          <degrees>360</degrees>
        </row>
        <row>
          <shape sides="3">triangle</shape>
          <degrees>180</degrees>
        </row>
      </data>"""

    # 使用 pytest 的 raises 函数检查是否会引发 ValueError 异常，并验证异常消息是否匹配
    with pytest.raises(
        ValueError,
        match=("xpath does not return any nodes or attributes"),
    ):
        read_xml(StringIO(xml), xpath="./row", attrs_only=True, parser=parser)


# 定义一个函数，测试空元素解析时是否会引发 ValueError 异常
def test_empty_elems_only(parser):
    # 定义一个 XML 字符串 xml，包含多个 <row> 元素，每个元素都有多个属性
    xml = """
      <data>
        <row sides="4" shape="square" degrees="360"/>
        <row sides="0" shape="circle" degrees="360"/>
        <row sides="3" shape="triangle" degrees="180"/>
      </data>"""

    # 使用 pytest 的 raises 函数检查是否会引发 ValueError 异常，并验证异常消息是否匹配
    with pytest.raises(
        ValueError,
        match=("xpath does not return any nodes or attributes"),
    ):
        read_xml(StringIO(xml), xpath="./row", elems_only=True, parser=parser)


# 定义一个函数，测试处理属性为中心的 XML 文件
def test_attribute_centric_xml():
    # 导入 pytest 的 importorskip 函数，如果导入 lxml 失败，则跳过此测试
    pytest.importorskip("lxml")
    
    # 定义一个 XML 字符串 xml，包含多个火车站的信息，每个站点用 <station> 元素表示，每个元素有 Name 和 coords 属性
    xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<TrainSchedule>
      <Stations>
         <station Name="Manhattan" coords="31,460,195,498"/>
         <station Name="Laraway Road" coords="63,409,194,455"/>
         <station Name="179th St (Orland Park)" coords="0,364,110,395"/>
         <station Name="153rd St (Orland Park)" coords="7,333,113,362"/>
         <station Name="143rd St (Orland Park)" coords="17,297,115,330"/>
         <station Name="Palos Park" coords="128,281,239,303"/>
         <station Name="Palos Heights" coords="148,257,283,279"/>
         <station Name="Worth" coords="170,230,248,255"/>
         <station Name="Chicago Ridge" coords="70,187,208,214"/>
         <station Name="Oak Lawn" coords="166,159,266,185"/>
         <station Name="Ashburn" coords="197,133,336,157"/>
         <station Name="Wrightwood" coords="219,106,340,133"/>
         <station Name="Chicago Union Sta" coords="220,0,360,43"/>
      </Stations>
</TrainSchedule>"""

# 这里保留空白
    # 使用自定义函数read_xml从XML字符串中读取数据，使用lxml解析器，默认提取所有'station'节点数据
    df_lxml = read_xml(StringIO(xml), xpath=".//station")
    
    # 使用自定义函数read_xml从XML字符串中读取数据，使用etree解析器，提取所有'station'节点数据
    df_etree = read_xml(StringIO(xml), xpath=".//station", parser="etree")
    
    # 使用自定义函数read_xml_iterparse从XML文件中迭代解析数据，使用lxml解析器，
    # 指定iterparse参数以仅提取'station'节点的'Name'和'coords'字段数据
    df_iter_lx = read_xml_iterparse(xml, iterparse={"station": ["Name", "coords"]})
    
    # 使用自定义函数read_xml_iterparse从XML文件中迭代解析数据，使用etree解析器，
    # 指定iterparse参数以仅提取'station'节点的'Name'和'coords'字段数据
    df_iter_et = read_xml_iterparse(
        xml, parser="etree", iterparse={"station": ["Name", "coords"]}
    )
    
    # 使用pandas.testing模块中的assert_frame_equal函数比较df_lxml和df_etree两个数据框是否相等
    tm.assert_frame_equal(df_lxml, df_etree)
    
    # 使用pandas.testing模块中的assert_frame_equal函数比较df_iter_lx和df_iter_et两个数据框是否相等
    tm.assert_frame_equal(df_iter_lx, df_iter_et)
# NAMES


def test_names_option_output(xml_books, parser):
    # 从 XML 文件中读取数据并返回一个 DataFrame，指定列名为 ["Col1", "Col2", "Col3", "Col4", "Col5"]，使用给定的解析器
    df_file = read_xml(
        xml_books, names=["Col1", "Col2", "Col3", "Col4", "Col5"], parser=parser
    )
    # 从 XML 文件中迭代解析数据并返回一个 DataFrame，使用给定的解析器和迭代器配置
    df_iter = read_xml(
        xml_books,
        parser=parser,
        names=["Col1", "Col2", "Col3", "Col4", "Col5"],
        iterparse={"book": ["category", "title", "author", "year", "price"]},
    )

    # 预期的 DataFrame 结果，用于比较测试
    df_expected = DataFrame(
        {
            "Col1": ["cooking", "children", "web"],
            "Col2": ["Everyday Italian", "Harry Potter", "Learning XML"],
            "Col3": ["Giada De Laurentiis", "J K. Rowling", "Erik T. Ray"],
            "Col4": [2005, 2005, 2003],
            "Col5": [30.00, 29.99, 39.95],
        }
    )

    # 比较两个 DataFrame 是否相等，用于测试断言
    tm.assert_frame_equal(df_file, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_repeat_names(parser):
    # 从 XML 字符串中读取数据，使用 XPath 表达式指定路径 ".//shape"，返回一个 DataFrame，指定列名为 ["type_dim", "shape", "type_edge"]，使用给定的解析器
    df_xpath = read_xml(
        StringIO(xml),
        xpath=".//shape",
        parser=parser,
        names=["type_dim", "shape", "type_edge"],
    )

    # 从 XML 字符串中迭代解析数据并返回一个 DataFrame，使用给定的解析器和迭代器配置，指定列名为 ["type_dim", "shape", "type_edge"]
    df_iter = read_xml_iterparse(
        xml,
        parser=parser,
        iterparse={"shape": ["type", "name", "type"]},
        names=["type_dim", "shape", "type_edge"],
    )

    # 预期的 DataFrame 结果，用于比较测试
    df_expected = DataFrame(
        {
            "type_dim": ["2D", "3D"],
            "shape": ["circle", "sphere"],
            "type_edge": ["curved", "curved"],
        }
    )

    # 比较两个 DataFrame 是否相等，用于测试断言
    tm.assert_frame_equal(df_xpath, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_repeat_values_new_names(parser):
    # 从 XML 字符串中读取数据，使用 XPath 表达式指定路径 ".//shape"，返回一个 DataFrame，指定列名为 ["name", "group"]，使用给定的解析器
    df_xpath = read_xml(
        StringIO(xml), xpath=".//shape", parser=parser, names=["name", "group"]
    )

    # 从 XML 字符串中迭代解析数据并返回一个 DataFrame，使用给定的解析器和迭代器配置，指定列名为 ["name", "group"]
    df_iter = read_xml_iterparse(
        xml,
        parser=parser,
        iterparse={"shape": ["name", "family"]},
        names=["name", "group"],
    )

    # 预期的 DataFrame 结果，用于比较测试
    df_expected = DataFrame(
        {
            "name": ["rectangle", "square", "ellipse", "circle"],
            "group": ["rectangle", "rectangle", "ellipse", "ellipse"],
        }
    )

    # 比较两个 DataFrame 是否相等，用于测试断言
    tm.assert_frame_equal(df_xpath, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_repeat_elements(parser):
    xml = """\
<shapes>
  <shape>
    <value item="name">circle</value>
    <value item="family">ellipse</value>
    <value item="degrees">360</value>
    <value item="sides">0</value>
  </shape>
  <shape>
    <value item="name">triangle</value>
    <value item="family">polygon</value>
    <value item="degrees">180</value>
"""

    # 在此处应该继续添加代码块和注释，但由于示例截止，无法提供后续内容。
  <value item="sides">3</value>
  # 定义一个 XML 元素 <value>，属性 item="sides"，其文本内容为 "3"

</shape>
  # 关闭上一个 <shape> 元素

<shape>
  # 开始一个新的 <shape> 元素

  <value item="name">square</value>
  # 定义一个 XML 元素 <value>，属性 item="name"，其文本内容为 "square"

  <value item="family">polygon</value>
  # 定义一个 XML 元素 <value>，属性 item="family"，其文本内容为 "polygon"

  <value item="degrees">360</value>
  # 定义一个 XML 元素 <value>，属性 item="degrees"，其文本内容为 "360"

  <value item="sides">4</value>
  # 定义一个 XML 元素 <value>，属性 item="sides"，其文本内容为 "4"

</shape>
  # 关闭当前 <shape> 元素
"""
xml 变量包含一个 XML 字符串，表示包含形状数据的文档。

df_xpath = read_xml(
    StringIO(xml),
    xpath=".//shape",
    parser=parser,
    names=["name", "family", "degrees", "sides"],
)
# 使用 read_xml 函数从 XML 字符串中读取数据到 DataFrame，通过 xpath 参数选择所有的 shape 元素，
# 使用指定的解析器 parser，并且指定列名为 ["name", "family", "degrees", "sides"]。

df_iter = read_xml_iterparse(
    xml,
    parser=parser,
    iterparse={"shape": ["value", "value", "value", "value"]},
    names=["name", "family", "degrees", "sides"],
)
# 使用 read_xml_iterparse 函数从 XML 字符串中迭代解析数据到 DataFrame，
# 使用指定的解析器 parser，通过 iterparse 参数选择 shape 元素及其子元素，
# 并指定列名为 ["name", "family", "degrees", "sides"]。

df_expected = DataFrame(
    {
        "name": ["circle", "triangle", "square"],
        "family": ["ellipse", "polygon", "polygon"],
        "degrees": [360, 180, 360],
        "sides": [0, 3, 4],
    }
)
# 创建一个预期的 DataFrame，包含预期的形状数据，用于后续的断言比较。

tm.assert_frame_equal(df_xpath, df_expected)
# 使用 pytest 中的 tm.assert_frame_equal 函数断言 df_xpath 和 df_expected 是否相等。

tm.assert_frame_equal(df_iter, df_expected)
# 使用 pytest 中的 tm.assert_frame_equal 函数断言 df_iter 和 df_expected 是否相等。
"""
    # 使用 pytest 的断言来验证 read_xml 函数在给定输入时是否会抛出 TypeError 异常，并且异常信息中包含字符串 "encoding None"
    with pytest.raises(TypeError, match="encoding None"):
        # 调用 read_xml 函数，传入 StringIO 对象作为数据源，指定解析器为 "lxml"，但未指定编码（encoding=None）
        read_xml(StringIO(data), parser="lxml", encoding=None)
# 测试函数：使用 etree 解析器读取不带编码的 XML 数据，转换为 DataFrame 对象
def test_none_encoding_etree():
    # GH#45133
    # 定义包含 XML 数据的字符串
    data = """<data>
  <row>
    <a>c</a>
  </row>
</data>
"""
    # 调用 read_xml 函数解析 XML 数据，返回结果存储在 result 中
    result = read_xml(StringIO(data), parser="etree", encoding=None)
    # 预期结果是一个 DataFrame 对象，包含列名为 "a"，数据为 ["c"]
    expected = DataFrame({"a": ["c"]})
    # 使用 pytest 模块的 assert_frame_equal 函数比较 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# PARSER

# 使用 lxml 解析器未安装时的测试函数装饰器
@td.skip_if_installed("lxml")
def test_default_parser_no_lxml(xml_books):
    # 使用 pytest 模块的 raises 函数检查是否抛出 ImportError 异常，
    # 匹配错误信息要求安装 lxml 或使用 etree 解析器
    with pytest.raises(
        ImportError, match=("lxml not found, please install or use the etree parser.")
    ):
        # 调用 read_xml 函数尝试读取 XML 数据
        read_xml(xml_books)


# 测试错误的解析器类型
def test_wrong_parser(xml_books):
    # 使用 pytest 模块的 raises 函数检查是否抛出 ValueError 异常，
    # 匹配错误信息要求解析器类型只能是 lxml 或 etree
    with pytest.raises(
        ValueError, match=("Values for parser can only be lxml or etree.")
    ):
        # 调用 read_xml 函数并指定错误的解析器类型 "bs4"
        read_xml(xml_books, parser="bs4")


# STYLESHEET

# 测试使用文件形式的样式表进行 XML 数据解析
def test_stylesheet_file(kml_cta_rail_lines, xsl_flatten_doc):
    # 导入 lxml 模块，如果未安装则跳过测试
    pytest.importorskip("lxml")
    # 使用 read_xml 函数解析 XML 数据，并应用样式表 xsl_flatten_doc
    df_style = read_xml(
        kml_cta_rail_lines,
        xpath=".//k:Placemark",
        namespaces={"k": "http://www.opengis.net/kml/2.2"},
        stylesheet=xsl_flatten_doc,
    )

    # 使用 pytest 模块的 assert_frame_equal 函数比较 df_kml 和 df_style 是否相等
    tm.assert_frame_equal(df_kml, df_style)

    # 使用 iterparse 参数解析 XML 数据，并进行 DataFrame 比较
    df_iter = read_xml(
        kml_cta_rail_lines,
        iterparse={
            "Placemark": [
                "id",
                "name",
                "styleUrl",
                "extrude",
                "altitudeMode",
                "coordinates",
            ]
        },
    )
    tm.assert_frame_equal(df_kml, df_iter)


# 测试使用类文件对象形式的样式表进行 XML 数据解析
def test_stylesheet_file_like(kml_cta_rail_lines, xsl_flatten_doc, mode):
    pytest.importorskip("lxml")
    with open(xsl_flatten_doc, mode, encoding="utf-8" if mode == "r" else None) as f:
        # 使用 read_xml 函数解析 XML 数据，并应用样式表 f
        df_style = read_xml(
            kml_cta_rail_lines,
            xpath=".//k:Placemark",
            namespaces={"k": "http://www.opengis.net/kml/2.2"},
            stylesheet=f,
        )

    # 使用 pytest 模块的 assert_frame_equal 函数比较 df_kml 和 df_style 是否相等
    tm.assert_frame_equal(df_kml, df_style)


# 测试使用 IO 流对象形式的样式表进行 XML 数据解析
def test_stylesheet_io(kml_cta_rail_lines, xsl_flatten_doc, mode):
    # 注意：默认情况下不检查未类型化函数的主体内容，可以考虑使用 --check-untyped-defs
    pytest.importorskip("lxml")
    # 定义 xsl_obj 变量的类型注释
    xsl_obj: BytesIO | StringIO  # type: ignore[annotation-unchecked]

    with open(xsl_flatten_doc, mode, encoding="utf-8" if mode == "r" else None) as f:
        if mode == "rb":
            xsl_obj = BytesIO(f.read())  # 将二进制数据读入 BytesIO 对象
        else:
            xsl_obj = StringIO(f.read())  # 将文本数据读入 StringIO 对象

    # 使用 read_xml 函数解析 XML 数据，并应用样式表 xsl_obj
    df_style = read_xml(
        kml_cta_rail_lines,
        xpath=".//k:Placemark",
        namespaces={"k": "http://www.opengis.net/kml/2.2"},
        stylesheet=xsl_obj,
    )

    # 使用 pytest 模块的 assert_frame_equal 函数比较 df_kml 和 df_style 是否相等
    tm.assert_frame_equal(df_kml, df_style)


# 测试使用缓冲读取器形式的样式表进行 XML 数据解析
def test_stylesheet_buffered_reader(kml_cta_rail_lines, xsl_flatten_doc, mode):
    pytest.importorskip("lxml")
    with open(xsl_flatten_doc, mode, encoding="utf-8" if mode == "r" else None) as f:
        # 使用 read_xml 函数解析 XML 数据，并应用样式表 f
        df_style = read_xml(
            kml_cta_rail_lines,
            xpath=".//k:Placemark",
            namespaces={"k": "http://www.opengis.net/kml/2.2"},
            stylesheet=f,
        )

    # 使用 pytest 模块的 assert_frame_equal 函数比较 df_kml 和 df_style 是否相等
    tm.assert_frame_equal(df_kml, df_style)


# 测试样式表字符集
def test_style_charset():
    # 留待后续添加具体测试内容
    # 导入 pytest 库并检查是否存在 lxml 库，如果不存在则跳过测试
    pytest.importorskip("lxml")
    # 定义一个包含中文标签的 XML 字符串
    xml = "<中文標籤><row><c1>1</c1><c2>2</c2></row></中文標籤>"

    # 定义一个 XSL 字符串，用于后续的 XML 转换操作
    xsl = """\
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
 <xsl:output omit-xml-declaration="yes" indent="yes"/>
 <xsl:strip-space elements="*"/>

 <xsl:template match="node()|@*">
     <!-- 匹配任何节点和属性，复制当前节点并应用模板 -->
     <xsl:copy>
       <xsl:apply-templates select="node()|@*"/>
     </xsl:copy>
 </xsl:template>

 <xsl:template match="中文標籤">
     <!-- 匹配名称为中文標籤的元素，将其替换为<根>元素 -->
     <根>
       <xsl:apply-templates />
     </根>
 </xsl:template>

</xsl:stylesheet>"""

    df_orig = read_xml(StringIO(xml))
    df_style = read_xml(StringIO(xml), stylesheet=StringIO(xsl))

    tm.assert_frame_equal(df_orig, df_style)
    <xsl:strip-space elements="*"/>
        # 设置 XSLT 样式表，移除所有元素节点间的空白
    
    <xsl:template match="@*|node()">
        # 定义 XSLT 模板匹配所有属性节点和元素节点
        <xsl:copy>
            # 复制当前节点
            <xsl:copy-of select="document('non_existent.xml')/*"/>
            # 复制 non_existent.xml 文档中的所有子节点到当前节点
        </xsl:copy>
    </xsl:template>
        # 结束当前模板
# 测试用例：测试读取 XML 时使用错误的 XSL 样式表情况下是否能正确抛出异常
def test_wrong_stylesheet(kml_cta_rail_lines, xml_data_path):
    # 导入 pytest，如果 pytest 未安装则跳过此测试
    pytest.importorskip("lxml.etree")
    
    # 获取 XSL 文件路径
    xsl = xml_data_path / "flatten_doesnt_exist.xsl"
    
    # 使用 pytest 断言检查是否抛出预期的 FileNotFoundError 异常
    with pytest.raises(
        FileNotFoundError, match=r"\[Errno 2\] No such file or directory"
    ):
        # 调用 read_xml 函数，传入 XML 数据和 XSL 文件路径
        read_xml(kml_cta_rail_lines, stylesheet=xsl)


# 测试用例：测试使用文件对象时是否能正确关闭文件
def test_stylesheet_file_close(kml_cta_rail_lines, xsl_flatten_doc, mode):
    # 注意：默认情况下，不检查无类型函数的主体
    # 可考虑使用 --check-untyped-defs 参数
    
    # 导入 pytest，如果 lxml 未安装则跳过此测试
    pytest.importorskip("lxml")
    
    xsl_obj: BytesIO | StringIO  # type: ignore[annotation-unchecked]
    
    # 打开 XSL 文件，根据模式选择使用 BytesIO 或 StringIO 进行包装
    with open(xsl_flatten_doc, mode, encoding="utf-8" if mode == "r" else None) as f:
        if mode == "rb":
            xsl_obj = BytesIO(f.read())
        else:
            xsl_obj = StringIO(f.read())
        
        # 调用 read_xml 函数，传入 XML 数据和 XSL 对象
        read_xml(kml_cta_rail_lines, stylesheet=xsl_obj)
        
        # 断言文件未关闭
        assert not f.closed


# 测试用例：测试在缺少 lxml 情况下使用样式表是否能正确抛出异常
def test_stylesheet_with_etree(kml_cta_rail_lines, xsl_flatten_doc):
    # 导入 pytest，如果 lxml 未安装则跳过此测试
    pytest.importorskip("lxml")
    
    with pytest.raises(
        ValueError, match=("To use stylesheet, you need lxml installed")
    ):
        # 调用 read_xml 函数，传入 XML 数据、解析器和 XSL 文件路径
        read_xml(kml_cta_rail_lines, parser="etree", stylesheet=xsl_flatten_doc)


# 测试用例：测试使用空的样式表时是否能正确抛出 XML 语法错误异常
@pytest.mark.parametrize("val", [StringIO(""), BytesIO(b"")])
def test_empty_stylesheet(val, kml_cta_rail_lines):
    lxml_etree = pytest.importorskip("lxml.etree")
    
    with pytest.raises(lxml_etree.XMLSyntaxError):
        # 调用 read_xml 函数，传入 XML 数据和空的样式表对象
        read_xml(kml_cta_rail_lines, stylesheet=val)


# 测试用例：测试文件对象在特定条件下是否能正确抛出 TypeError 异常
def test_file_like_iterparse(xml_books, parser, mode):
    with open(xml_books, mode, encoding="utf-8" if mode == "r" else None) as f:
        if mode == "r" and parser == "lxml":
            with pytest.raises(
                TypeError, match=("reading file objects must return bytes objects")
            ):
                # 调用 read_xml 函数，传入 XML 文件对象、解析器和 iterparse 参数
                read_xml(
                    f,
                    parser=parser,
                    iterparse={
                        "book": ["category", "title", "year", "author", "price"]
                    },
                )
            return None
        else:
            # 调用 read_xml 函数，传入 XML 文件对象、解析器和 iterparse 参数
            df_filelike = read_xml(
                f,
                parser=parser,
                iterparse={"book": ["category", "title", "year", "author", "price"]},
            )
    
    # 预期的 DataFrame 结果
    df_expected = DataFrame(
        {
            "category": ["cooking", "children", "web"],
            "title": ["Everyday Italian", "Harry Potter", "Learning XML"],
            "author": ["Giada De Laurentiis", "J K. Rowling", "Erik T. Ray"],
            "year": [2005, 2005, 2003],
            "price": [30.00, 29.99, 39.95],
        }
    )
    
    # 使用 assert_frame_equal 检查实际和预期的 DataFrame 是否相等
    tm.assert_frame_equal(df_filelike, df_expected)


# 测试用例：测试文件 IO 迭代解析是否能正常工作
def test_file_io_iterparse(xml_books, parser, mode):
    funcIO = StringIO if mode == "r" else BytesIO
    # 使用指定的模式和编码方式打开 XML 文件
    with open(
        xml_books,
        mode,
        encoding="utf-8" if mode == "r" else None,
    ) as f:
        # 将文件内容读取为字节流对象
        with funcIO(f.read()) as b:
            # 如果模式是读取且解析器是 lxml
            if mode == "r" and parser == "lxml":
                # 断言读取文件对象必须返回字节对象，否则抛出 TypeError 异常
                with pytest.raises(
                    TypeError, match=("reading file objects must return bytes objects")
                ):
                    # 调用 read_xml 函数读取 XML 数据
                    read_xml(
                        b,
                        parser=parser,
                        iterparse={
                            "book": ["category", "title", "year", "author", "price"]
                        },
                    )
                # 返回空值
                return None
            else:
                # 使用 read_xml 函数读取 XML 数据并存储在 df_fileio 中
                df_fileio = read_xml(
                    b,
                    parser=parser,
                    iterparse={
                        "book": ["category", "title", "year", "author", "price"]
                    },
                )

    # 创建期望的 DataFrame，包含预期的数据
    df_expected = DataFrame(
        {
            "category": ["cooking", "children", "web"],
            "title": ["Everyday Italian", "Harry Potter", "Learning XML"],
            "author": ["Giada De Laurentiis", "J K. Rowling", "Erik T. Ray"],
            "year": [2005, 2005, 2003],
            "price": [30.00, 29.99, 39.95],
        }
    )

    # 使用测试工具比较 df_fileio 和 df_expected 是否相等
    tm.assert_frame_equal(df_fileio, df_expected)
@pytest.mark.network
@pytest.mark.single_cpu
# 定义一个测试函数，用于测试处理 URL 路径错误的情况
def test_url_path_error(parser, httpserver, xml_file):
    with open(xml_file, encoding="utf-8") as f:
        # 启动 HTTP 服务器并提供 XML 内容
        httpserver.serve_content(content=f.read())
        # 使用 pytest 检查是否会抛出 ParserError 异常，并验证异常消息
        with pytest.raises(
            ParserError, match=("iterparse is designed for large XML files")
        ):
            # 调用 read_xml 函数，传入 HTTP 服务器的 URL、解析器和 iterparse 参数
            read_xml(
                httpserver.url,
                parser=parser,
                iterparse={"row": ["shape", "degrees", "sides", "date"]},
            )


# 定义一个测试函数，用于测试处理压缩文件错误的情况
def test_compression_error(parser, compression_only):
    with tm.ensure_clean(filename="geom_xml.zip") as path:
        # 将 DataFrame 保存为 XML 文件，并设置压缩方式
        geom_df.to_xml(path, parser=parser, compression=compression_only)

        # 使用 pytest 检查是否会抛出 ParserError 异常，并验证异常消息
        with pytest.raises(
            ParserError, match=("iterparse is designed for large XML files")
        ):
            # 调用 read_xml 函数，传入压缩后的 XML 文件路径、解析器和 iterparse 参数
            read_xml(
                path,
                parser=parser,
                iterparse={"row": ["shape", "degrees", "sides", "date"]},
                compression=compression_only,
            )


# 定义一个测试函数，用于测试处理字典类型错误的情况
def test_wrong_dict_type(xml_books, parser):
    with pytest.raises(TypeError, match="list is not a valid type for iterparse"):
        # 调用 read_xml 函数，传入 XML 文件路径、解析器和错误类型的 iterparse 参数
        read_xml(
            xml_books,
            parser=parser,
            iterparse=["category", "title", "year", "author", "price"],
        )


# 定义一个测试函数，用于测试处理字典值类型错误的情况
def test_wrong_dict_value(xml_books, parser):
    with pytest.raises(
        TypeError, match="<class 'str'> is not a valid type for value in iterparse"
    ):
        # 调用 read_xml 函数，传入 XML 文件路径、解析器和错误值类型的 iterparse 参数
        read_xml(xml_books, parser=parser, iterparse={"book": "category"})


# 定义一个测试函数，用于测试处理坏 XML 文件的情况
def test_bad_xml(parser):
    bad_xml = """\
<?xml version='1.0' encoding='utf-8'?>
  <row>
    <shape>square</shape>
    <degrees>00360</degrees>
    <sides>4.0</sides>
    <date>2020-01-01</date>
   </row>
  <row>
    <shape>circle</shape>
    <degrees>00360</degrees>
    <sides/>
    <date>2021-01-01</date>
  </row>
  <row>
    <shape>triangle</shape>
    <degrees>00180</degrees>
    <sides>3.0</sides>
    <date>2022-01-01</date>
  </row>
"""
    with tm.ensure_clean(filename="bad.xml") as path:
        with open(path, "w", encoding="utf-8") as f:
            # 将错误的 XML 写入文件
            f.write(bad_xml)

        # 使用 pytest 检查是否会抛出 SyntaxError 异常，并验证异常消息
        with pytest.raises(
            SyntaxError,
            match=(
                "Extra content at the end of the document|"
                "junk after document element"
            ),
        ):
            # 调用 read_xml 函数，传入错误 XML 文件路径、解析器、日期解析和 iterparse 参数
            read_xml(
                path,
                parser=parser,
                parse_dates=["date"],
                iterparse={"row": ["shape", "degrees", "sides", "date"]},
            )


# 定义一个测试函数，用于测试解析包含注释的 XML 内容的情况
def test_comment(parser):
    xml = """\
<!-- comment before root -->
<shapes>
  <!-- comment within root -->
  <shape>
    <name>circle</name>
    <type>2D</type>
  </shape>
  <shape>
    <name>sphere</name>
    <type>3D</type>
    <!-- comment within child -->
  </shape>
  <!-- comment within root -->
</shapes>
<!-- comment after root -->"""

    # 调用 read_xml 函数，传入包含注释的 XML 字符串和解析器，使用 XPath 查询形状元素
    df_xpath = read_xml(StringIO(xml), xpath=".//shape", parser=parser)
    # 使用 read_xml_iterparse 函数从 XML 文件中迭代解析数据，并指定解析器和迭代器配置
    df_iter = read_xml_iterparse(
        xml, parser=parser, iterparse={"shape": ["name", "type"]}
    )

    # 创建一个期望的 DataFrame，包含预期的数据列 "name" 和 "type"
    df_expected = DataFrame(
        {
            "name": ["circle", "sphere"],
            "type": ["2D", "3D"],
        }
    )

    # 使用测试工具 tm.assert_frame_equal 比较 df_xpath 和 df_expected 的内容是否相等
    tm.assert_frame_equal(df_xpath, df_expected)

    # 使用测试工具 tm.assert_frame_equal 比较 df_iter 和 df_expected 的内容是否相等
    tm.assert_frame_equal(df_iter, df_expected)
def test_dtd(parser):
    xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE non-profits [
    <!ELEMENT shapes (shape*) >
    <!ELEMENT shape ( name, type )>
    <!ELEMENT name (#PCDATA)>
]>
<shapes>
  <shape>
    <name>circle</name>
    <type>2D</type>
  </shape>
  <shape>
    <name>sphere</name>
    <type>3D</type>
  </shape>
</shapes>"""

    # 使用 read_xml 函数解析 XML，从中提取所有的 <shape> 元素作为 DataFrame
    df_xpath = read_xml(StringIO(xml), xpath=".//shape", parser=parser)

    # 使用 read_xml_iterparse 函数迭代解析 XML，指定解析 <shape> 元素及其子元素作为 DataFrame
    df_iter = read_xml_iterparse(
        xml, parser=parser, iterparse={"shape": ["name", "type"]}
    )

    # 创建期望的 DataFrame，包含两列：name 和 type
    df_expected = DataFrame(
        {
            "name": ["circle", "sphere"],
            "type": ["2D", "3D"],
        }
    )

    # 断言两个 DataFrame 是否相等
    tm.assert_frame_equal(df_xpath, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_processing_instruction(parser):
    xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="style.xsl"?>
<?display table-view?>
<?sort alpha-ascending?>
<?textinfo whitespace is allowed ?>
<?elementnames <shape>, <name>, <type> ?>
<shapes>
  <shape>
    <name>circle</name>
    <type>2D</type>
  </shape>
  <shape>
    <name>sphere</name>
    <type>3D</type>
  </shape>
</shapes>"""

    # 使用 read_xml 函数解析 XML，从中提取所有的 <shape> 元素作为 DataFrame
    df_xpath = read_xml(StringIO(xml), xpath=".//shape", parser=parser)

    # 使用 read_xml_iterparse 函数迭代解析 XML，指定解析 <shape> 元素及其子元素作为 DataFrame
    df_iter = read_xml_iterparse(
        xml, parser=parser, iterparse={"shape": ["name", "type"]}
    )

    # 创建期望的 DataFrame，包含两列：name 和 type
    df_expected = DataFrame(
        {
            "name": ["circle", "sphere"],
            "type": ["2D", "3D"],
        }
    )

    # 断言两个 DataFrame 是否相等
    tm.assert_frame_equal(df_xpath, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_no_result(xml_books, parser):
    # 使用 read_xml 函数尝试解析 XML，期望抛出 ParserError 异常，指定匹配异常信息
    with pytest.raises(
        ParserError, match="No result from selected items in iterparse."
    ):
        read_xml(
            xml_books,
            parser=parser,
            iterparse={"node": ["attr1", "elem1", "elem2", "elem3"]},
        )


def test_empty_data(xml_books, parser):
    # 使用 read_xml 函数尝试解析 XML，期望抛出 EmptyDataError 异常，指定匹配异常信息
    with pytest.raises(EmptyDataError, match="No columns to parse from file"):
        read_xml(
            xml_books,
            parser=parser,
            iterparse={"book": ["attr1", "elem1", "elem2", "elem3"]},
        )


def test_online_stylesheet():
    # 导入 pytest 的 importorskip 函数，如果 lxml 不存在则跳过测试
    pytest.importorskip("lxml")
    xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<catalog>
  <cd>
    <title>Empire Burlesque</title>
    <artist>Bob Dylan</artist>
    <country>USA</country>
    <company>Columbia</company>
    <price>10.90</price>
    <year>1985</year>
  </cd>
  <cd>
    <title>Hide your heart</title>
    <artist>Bonnie Tyler</artist>
    <country>UK</country>
    <company>CBS Records</company>
    <price>9.90</price>
    <year>1988</year>
  </cd>
  <cd>
    <title>Greatest Hits</title>
    <artist>Dolly Parton</artist>
    <country>USA</country>
    <company>RCA</company>
    <price>9.90</price>
    <year>1982</year>
  </cd>
  <cd>
    <title>Still got the blues</title>
    <artist>Gary Moore</artist>
    <country>UK</country>
    <company>Virgin records</company>
  <cd>
    <title>Private Dancer</title>  <!-- CD的标题 -->
    <artist>Tina Turner</artist>   <!-- 艺术家名称 -->
    <country>UK</country>         <!-- 发行国家 -->
    <company>Capitol</company>     <!-- 发行公司 -->
    <price>8.90</price>           <!-- CD的价格 -->
    <year>1983</year>             <!-- 发行年份 -->
  </cd>
    # XML片段，描述了多个CD唱片的信息
    <title>Midt om natten</title>  # CD的标题为Midt om natten
    <artist>Kim Larsen</artist>   # 艺术家是Kim Larsen
    <country>EU</country>         # 发行国家为EU
    <company>Medley</company>     # 唱片公司是Medley
    <price>7.80</price>           # 售价为7.80
    <year>1983</year>             # 发行年份为1983
      </cd>
      <cd>
    <title>Pavarotti Gala Concert</title>  # CD的标题为Pavarotti Gala Concert
    <artist>Luciano Pavarotti</artist>     # 艺术家是Luciano Pavarotti
    <country>UK</country>                 # 发行国家为UK
    <company>DECCA</company>              # 唱片公司是DECCA
    <price>9.90</price>                   # 售价为9.90
    <year>1991</year>                     # 发行年份为1991
      </cd>
      <cd>
    <title>The dock of the bay</title>    # CD的标题为The dock of the bay
    <artist>Otis Redding</artist>         # 艺术家是Otis Redding
    <country>USA</country>                # 发行国家为USA
    <COMPANY>Stax Records</COMPANY>       # 唱片公司是Stax Records（注意大小写）
    <PRICE>7.90</PRICE>                   # 售价为7.90（注意大小写）
    <YEAR>1968</YEAR>                     # 发行年份为1968（注意大小写）
      </cd>
      <cd>
    <title>Picture book</title>           # CD的标题为Picture book
    <artist>Simply Red</artist>           # 艺术家是Simply Red
    <country>EU</country>                 # 发行国家为EU
    <company>Elektra</company>            # 唱片公司是Elektra
    <price>7.20</price>                   # 售价为7.20
    <year>1985</year>                     # 发行年份为1985
      </cd>
      <cd>
    <title>Red</title>                    # CD的标题为Red
    <artist>The Communards</artist>       # 艺术家是The Communards
    <country>UK</country>                 # 发行国家为UK
    <company>London</company>             # 唱片公司是London
    <price>7.80</price>                   # 售价为7.80
    <year>1987</year>                     # 发行年份为1987
      </cd>
      <cd>
    <title>Unchain my heart</title>       # CD的标题为Unchain my heart
    <artist>Joe Cocker</artist>           # 艺术家是Joe Cocker
    <country>USA</country>                # 发行国家为USA
    <company>EMI</company>                # 唱片公司是EMI
    <price>8.20</price>                   # 售价为8.20
    <year>1987</year>                     # 发行年份为1987
    </cd>
# XML 样式表定义，用于将 XML 转换为 HTML 表格
xsl = """\
<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
<xsl:template match="/">
<html>
<body>
  <h2>My CD Collection</h2>
  <table border="1">
    <tr bgcolor="#9acd32">
      <th style="text-align:left">Title</th>
      <th style="text-align:left">Artist</th>
    </tr>
    <xsl:for-each select="catalog/cd">
    <tr>
      <td><xsl:value-of select="title"/></td>  <!-- 输出每个 CD 的标题 -->
      <td><xsl:value-of select="artist"/></td> <!-- 输出每个 CD 的艺术家 -->
    </tr>
    </xsl:for-each>
  </table>
</body>
</html>
</xsl:template>
</xsl:stylesheet>
"""

# 使用指定的 XML 和 XSL 样式表生成 DataFrame
df_xsl = read_xml(
    StringIO(xml),  # 从字符串中读取 XML 数据
    xpath=".//tr[td and position() <= 6]",  # 使用 XPath 选择前6个包含 td 的 tr 元素
    names=["title", "artist"],  # 指定列名为 "title" 和 "artist"
    stylesheet=StringIO(xsl),  # 使用 StringIO 对象作为样式表
)

# 期望的 DataFrame 结果
df_expected = DataFrame(
    {
        "title": {
            0: "Empire Burlesque",
            1: "Hide your heart",
            2: "Greatest Hits",
            3: "Still got the blues",
            4: "Eros",
        },
        "artist": {
            0: "Bob Dylan",
            1: "Bonnie Tyler",
            2: "Dolly Parton",
            3: "Gary Moore",
            4: "Eros Ramazzotti",
        },
    }
)

# 使用测试工具函数比较生成的 DataFrame 和预期的 DataFrame
tm.assert_frame_equal(df_expected, df_xsl)


# COMPRESSION


# 测试压缩读取功能
def test_compression_read(parser, compression_only):
    with tm.ensure_clean() as comp_path:
        geom_df.to_xml(
            comp_path, index=False, parser=parser, compression=compression_only
        )  # 将 DataFrame geom_df 写入到指定路径，使用指定的解析器和压缩算法

        df_xpath = read_xml(comp_path, parser=parser, compression=compression_only)  # 从压缩文件中读取 XML 数据为 DataFrame

        df_iter = read_xml_iterparse_comp(
            comp_path,
            compression_only,
            parser=parser,
            iterparse={"row": ["shape", "degrees", "sides"]},  # 使用 iterparse 参数
            compression=compression_only,
        )  # 使用迭代解析从压缩文件中读取 XML 数据为 DataFrame

    # 使用测试工具函数比较生成的 DataFrame 和原始的 DataFrame geom_df
    tm.assert_frame_equal(df_xpath, geom_df)
    tm.assert_frame_equal(df_iter, geom_df)


# 测试错误的压缩算法
def test_wrong_compression(parser, compression, compression_only):
    actual_compression = compression
    attempted_compression = compression_only

    if actual_compression == attempted_compression:
        pytest.skip(f"{actual_compression} == {attempted_compression}")  # 如果实际压缩算法与尝试的压缩算法相同，则跳过测试

    # 定义可能的错误和异常消息
    errors = {
        "bz2": (OSError, "Invalid data stream"),
        "gzip": (OSError, "Not a gzipped file"),
        "zip": (BadZipFile, "File is not a zip file"),
        "tar": (ReadError, "file could not be opened successfully"),
    }

    # 尝试导入可选依赖项，以及对应的错误处理
    zstd = import_optional_dependency("zstandard", errors="ignore")
    if zstd is not None:
        errors["zstd"] = (zstd.ZstdError, "Unknown frame descriptor")
    lzma = import_optional_dependency("lzma", errors="ignore")
    if lzma is not None:
        errors["xz"] = (LZMAError, "Input format not supported by decoder")

    # 获取当前压缩算法的错误类和错误消息
    error_cls, error_str = errors[attempted_compression]
    # 使用 tm.ensure_clean() 创建一个临时文件或目录，并将路径赋值给变量 path
    with tm.ensure_clean() as path:
        # 将 geom_df 转换为 XML 格式，并写入到指定路径的文件中，使用给定的解析器和压缩方式
        geom_df.to_xml(path, parser=parser, compression=actual_compression)
    
        # 使用 pytest.raises() 断言捕获特定类型的异常，并验证异常信息匹配给定的字符串
        with pytest.raises(error_cls, match=error_str):
            # 调用 read_xml 函数，读取指定路径的 XML 文件，使用指定的解析器和压缩方式
            read_xml(path, parser=parser, compression=attempted_compression)
# 测试不支持的压缩类型异常情况，使用 pytest 的断言检查是否引发 ValueError，并匹配异常信息
def test_unsuported_compression(parser):
    with pytest.raises(ValueError, match="Unrecognized compression type"):
        # 使用 tm.ensure_clean() 确保环境干净，获取临时文件路径
        with tm.ensure_clean() as path:
            # 调用 read_xml 函数读取 XML 数据，传入指定的解析器和未识别的压缩类型
            read_xml(path, parser=parser, compression="7z")


# 存储选项

# 标记为网络测试用例
# 标记为单 CPU 环境测试用例
@pytest.mark.network
@pytest.mark.single_cpu
def test_s3_parser_consistency(s3_public_bucket_with_data, s3so):
    # 导入 s3fs 库，如果不存在则跳过测试
    pytest.importorskip("s3fs")
    # 导入 lxml 库，如果不存在则跳过测试
    pytest.importorskip("lxml")
    # 构建 S3 文件路径
    s3 = f"s3://{s3_public_bucket_with_data.name}/books.xml"

    # 使用 lxml 解析器读取 XML 数据
    df_lxml = read_xml(s3, parser="lxml", storage_options=s3so)

    # 使用 etree 解析器读取 XML 数据
    df_etree = read_xml(s3, parser="etree", storage_options=s3so)

    # 使用 tm.assert_frame_equal 检查两个 DataFrame 是否相等
    tm.assert_frame_equal(df_lxml, df_etree)


# 测试读取 XML 数据时处理可空数据类型
def test_read_xml_nullable_dtypes(
    parser, string_storage, dtype_backend, using_infer_string
):
    # GH#50500
    # 定义 XML 数据字符串
    data = """<?xml version='1.0' encoding='utf-8'?>
<data xmlns="http://example.com">
<row>
  <a>x</a>
  <b>1</b>
  <c>4.0</c>
  <d>x</d>
  <e>2</e>
  <f>4.0</f>
  <g></g>
  <h>True</h>
  <i>False</i>
</row>
<row>
  <a>y</a>
  <b>2</b>
  <c>5.0</c>
  <d></d>
  <e></e>
  <f></f>
  <g></g>
  <h>False</h>
  <i></i>
</row>
</data>"""

    # 根据条件选择不同的数据存储方式和处理方式
    if using_infer_string:
        # 导入 pyarrow 库，如果不存在则跳过测试
        pa = pytest.importorskip("pyarrow")
        # 创建 ArrowStringArrayNumpySemantics 对象
        string_array = ArrowStringArrayNumpySemantics(pa.array(["x", "y"]))
        string_array_na = ArrowStringArrayNumpySemantics(pa.array(["x", None]))

    elif string_storage == "python":
        # 使用 Python 对象存储字符串数组
        string_array = StringArray(np.array(["x", "y"], dtype=np.object_))
        string_array_na = StringArray(np.array(["x", NA], dtype=np.object_))

    elif dtype_backend == "pyarrow":
        # 导入 pyarrow 库，如果不存在则跳过测试
        pa = pytest.importorskip("pyarrow")
        # 导入 ArrowExtensionArray 类
        from pandas.arrays import ArrowExtensionArray
        # 创建 ArrowExtensionArray 对象
        string_array = ArrowExtensionArray(pa.array(["x", "y"]))
        string_array_na = ArrowExtensionArray(pa.array(["x", None]))

    else:
        # 导入 pyarrow 库，如果不存在则跳过测试
        pa = pytest.importorskip("pyarrow")
        # 创建 ArrowStringArray 对象
        string_array = ArrowStringArray(pa.array(["x", "y"]))
        string_array_na = ArrowStringArray(pa.array(["x", None]))

    # 设置 pandas 的字符串存储模式
    with pd.option_context("mode.string_storage", string_storage):
        # 调用 read_xml 函数读取 XML 数据，指定解析器和数据类型处理方式
        result = read_xml(StringIO(data), parser=parser, dtype_backend=dtype_backend)

    # 期望的结果 DataFrame 对象
    expected = DataFrame(
        {
            "a": string_array,
            "b": Series([1, 2], dtype="Int64"),
            "c": Series([4.0, 5.0], dtype="Float64"),
            "d": string_array_na,
            "e": Series([2, NA], dtype="Int64"),
            "f": Series([4.0, NA], dtype="Float64"),
            "g": Series([NA, NA], dtype="Int64"),
            "h": Series([True, False], dtype="boolean"),
            "i": Series([False, NA], dtype="boolean"),
        }
    )
    # 如果数据类型后端是 "pyarrow"
    if dtype_backend == "pyarrow":
        # 导入 pytest 并检查是否可用，否则跳过测试
        pa = pytest.importorskip("pyarrow")
        # 从 pandas.arrays 中导入 ArrowExtensionArray
        from pandas.arrays import ArrowExtensionArray

        # 创建预期的 DataFrame
        expected = DataFrame(
            {
                # 对于每一列，在 ArrowExtensionArray 中创建由 pyarrow 数组转换而来的扩展数组
                col: ArrowExtensionArray(pa.array(expected[col], from_pandas=True))
                for col in expected.columns
            }
        )
        # 向预期的 DataFrame 添加一列 "g"，其中的值为 ArrowExtensionArray 中的空值数组
        expected["g"] = ArrowExtensionArray(pa.array([None, None]))

    # 使用 pandas.testing 模块中的 assert_frame_equal 函数比较 result 和 expected DataFrame 是否相等
    tm.assert_frame_equal(result, expected)
# 定义一个测试函数，用于测试传入无效 dtype_backend 参数时是否会引发 ValueError 异常
def test_invalid_dtype_backend():
    # 定义错误消息，指出 numpy 是无效的 dtype_backend 参数，只允许 'numpy_nullable' 和 'pyarrow'
    msg = (
        "dtype_backend numpy is invalid, only 'numpy_nullable' and "
        "'pyarrow' are allowed."
    )
    # 使用 pytest 的 raises 方法验证 read_xml 函数在使用 'numpy' 作为 dtype_backend 参数时是否会抛出 ValueError 异常，并且异常消息需要匹配设定的 msg
    with pytest.raises(ValueError, match=msg):
        read_xml("test", dtype_backend="numpy")
```