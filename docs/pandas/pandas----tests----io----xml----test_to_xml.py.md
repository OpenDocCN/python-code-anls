# `D:\src\scipysrc\pandas\pandas\tests\io\xml\test_to_xml.py`

```
# 导入所需模块和库
from __future__ import annotations  # 从未来版本导入类型注解支持

from io import (
    BytesIO,  # 导入字节流的IO支持
    StringIO,  # 导入字符串的IO支持
)
import os  # 导入操作系统相关功能

import numpy as np  # 导入数值计算库numpy
import pytest  # 导入用于单元测试的pytest框架

import pandas.util._test_decorators as td  # 导入pandas测试装饰器
from pandas import (
    NA,  # 导入pandas的NA标识
    DataFrame,  # 导入pandas的数据结构DataFrame
    Index,  # 导入pandas的索引类型Index
)
import pandas._testing as tm  # 导入pandas测试工具

from pandas.io.common import get_handle  # 导入pandas IO的通用函数get_handle
from pandas.io.xml import read_xml  # 导入pandas处理XML的函数read_xml

# CHECKLIST

# [x] - ValueError: "Values for parser can only be lxml or etree."

# etree
# [x] - ImportError: "lxml not found, please install or use the etree parser."
# [X] - TypeError: "...is not a valid type for attr_cols"
# [X] - TypeError: "...is not a valid type for elem_cols"
# [X] - LookupError: "unknown encoding"
# [X] - KeyError: "...is not included in namespaces"
# [X] - KeyError: "no valid column"
# [X] - ValueError: "To use stylesheet, you need lxml installed..."
# []  - OSError: (NEED PERMISSOIN ISSUE, DISK FULL, ETC.)
# [X] - FileNotFoundError: "No such file or directory"
# [X] - PermissionError: "Forbidden"

# lxml
# [X] - TypeError: "...is not a valid type for attr_cols"
# [X] - TypeError: "...is not a valid type for elem_cols"
# [X] - LookupError: "unknown encoding"
# []  - OSError: (NEED PERMISSOIN ISSUE, DISK FULL, ETC.)
# [X] - FileNotFoundError: "No such file or directory"
# [X] - KeyError: "...is not included in namespaces"
# [X] - KeyError: "no valid column"
# [X] - ValueError: "stylesheet is not a url, file, or xml string."
# []  - LookupError: (NEED WRONG ENCODING FOR FILE OUTPUT)
# []  - URLError: (USUALLY DUE TO NETWORKING)
# []  - HTTPError: (NEED AN ONLINE STYLESHEET)
# [X] - OSError: "failed to load external entity"
# [X] - XMLSyntaxError: "Opening and ending tag mismatch"
# [X] - XSLTApplyError: "Cannot resolve URI"
# [X] - XSLTParseError: "failed to compile"
# [X] - PermissionError: "Forbidden"


@pytest.fixture
def geom_df():
    # 创建一个包含几何形状数据的DataFrame对象
    return DataFrame(
        {
            "shape": ["square", "circle", "triangle"],  # 形状名称列
            "degrees": [360, 360, 180],  # 每种形状的角度列
            "sides": [4, np.nan, 3],  # 每种形状的边数列，可能包含NaN
        }
    )


@pytest.fixture
def planet_df():
    # 此处应该继续填写fixture函数的内容
    # 创建一个 DataFrame 对象，包含行星的信息：名称、类型、位置和质量
    return DataFrame(
        {
            # 行星的名称列表
            "planet": [
                "Mercury",   # 水星
                "Venus",     # 金星
                "Earth",     # 地球
                "Mars",      # 火星
                "Jupiter",   # 木星
                "Saturn",    # 土星
                "Uranus",    # 天王星
                "Neptune",   # 海王星
            ],
            # 行星的类型列表
            "type": [
                "terrestrial",   # 类地行星
                "terrestrial",   # 类地行星
                "terrestrial",   # 类地行星
                "terrestrial",   # 类地行星
                "gas giant",     # 气态巨行星
                "gas giant",     # 气态巨行星
                "ice giant",     # 冰巨行星
                "ice giant",     # 冰巨行星
            ],
            # 行星所处位置列表
            "location": [
                "inner",   # 内行星
                "inner",   # 内行星
                "inner",   # 内行星
                "inner",   # 内行星
                "outer",   # 外行星
                "outer",   # 外行星
                "outer",   # 外行星
                "outer",   # 外行星
            ],
            # 行星的质量列表
            "mass": [
                0.330114,    # 水星质量
                4.86747,     # 金星质量
                5.97237,     # 地球质量
                0.641712,    # 火星质量
                1898.187,    # 木星质量
                568.3174,    # 土星质量
                86.8127,     # 天王星质量
                102.4126,    # 海王星质量
            ],
        }
    )
@pytest.fixture
def from_file_expected():
    return """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row>
    <index>0</index>
    <category>cooking</category>
    <title>Everyday Italian</title>
    <author>Giada De Laurentiis</author>
    <year>2005</year>
    <price>30.0</price>
  </row>
  <row>
    <index>1</index>
    <category>children</category>
    <title>Harry Potter</title>
    <author>J K. Rowling</author>
    <year>2005</year>
    <price>29.99</price>
  </row>
  <row>
    <index>2</index>
    <category>web</category>
    <title>Learning XML</title>
    <author>Erik T. Ray</author>
    <year>2003</year>
    <price>39.95</price>
  </row>
</data>"""


def equalize_decl(doc):
    # 如果传入的文档不为None，则替换其中的XML声明部分，以适应不同的XML解析器要求
    if doc is not None:
        doc = doc.replace(
            '<?xml version="1.0" encoding="utf-8"?',
            "<?xml version='1.0' encoding='utf-8'?",
        )
    return doc


@pytest.fixture(params=["rb", "r"])
def mode(request):
    # 提供两种参数化的fixture，分别返回"rb"和"r"
    return request.param


@pytest.fixture(params=[pytest.param("lxml", marks=td.skip_if_no("lxml")), "etree"])
def parser(request):
    # 提供两种参数化的fixture，一个带有条件标记，要求有lxml库，另一个直接返回"etree"
    return request.param


# FILE OUTPUT


def test_file_output_str_read(xml_books, parser, from_file_expected):
    # 使用给定的XML数据文件和解析器参数读取数据，并将其转换为DataFrame
    df_file = read_xml(xml_books, parser=parser)

    # 使用tm模块确保在测试过程中生成的文件可以被安全地删除，文件名为"test.xml"
    with tm.ensure_clean("test.xml") as path:
        # 将DataFrame对象写入XML文件中
        df_file.to_xml(path, parser=parser)
        
        # 打开刚写入的XML文件，读取其内容为二进制数据，然后解码为UTF-8字符串并去除首尾空白字符
        with open(path, "rb") as f:
            output = f.read().decode("utf-8").strip()

        # 调用equalize_decl函数，将输出的XML内容中的XML声明部分统一格式化，以便比较
        output = equalize_decl(output)

        # 断言写入文件的内容与预期的XML字符串相同
        assert output == from_file_expected


def test_file_output_bytes_read(xml_books, parser, from_file_expected):
    # 使用给定的XML数据文件和解析器参数读取数据，并将其转换为DataFrame
    df_file = read_xml(xml_books, parser=parser)

    # 使用tm模块确保在测试过程中生成的文件可以被安全地删除，文件名为"test.xml"
    with tm.ensure_clean("test.xml") as path:
        # 将DataFrame对象写入XML文件中
        df_file.to_xml(path, parser=parser)
        
        # 打开刚写入的XML文件，读取其内容为二进制数据，然后解码为UTF-8字符串并去除首尾空白字符
        with open(path, "rb") as f:
            output = f.read().decode("utf-8").strip()

        # 调用equalize_decl函数，将输出的XML内容中的XML声明部分统一格式化，以便比较
        output = equalize_decl(output)

        # 断言写入文件的内容与预期的XML字符串相同
        assert output == from_file_expected


def test_str_output(xml_books, parser, from_file_expected):
    # 使用给定的XML数据文件和解析器参数读取数据，并将其转换为DataFrame
    df_file = read_xml(xml_books, parser=parser)

    # 将DataFrame对象转换为XML字符串，不经过文件写入操作
    output = df_file.to_xml(parser=parser)
    
    # 调用equalize_decl函数，将输出的XML内容中的XML声明部分统一格式化，以便比较
    output = equalize_decl(output)

    # 断言转换得到的XML字符串与预期的XML字符串相同
    assert output == from_file_expected


def test_wrong_file_path(parser, geom_df):
    # 指定一个不存在的文件路径
    path = "/my/fake/path/output.xml"

    # 断言尝试将数据写入不存在的目录会引发OSError，并且错误消息中包含特定路径信息
    with pytest.raises(
        OSError,
        match=(r"Cannot save file into a non-existent directory: .*path"),
    ):
        geom_df.to_xml(path, parser=parser)


# INDEX


def test_index_false(xml_books, parser):
    expected = """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row>
    <category>cooking</category>
    <title>Everyday Italian</title>
    <author>Giada De Laurentiis</author>
    <year>2005</year>
    <price>30.0</price>
  </row>
  <row>
    <category>children</category>
    <title>Harry Potter</title>
    <author>J K. Rowling</author>
    <year>2005</year>
    <price>29.99</price>
  </row>
  <row>
    <category>web</category>
    <title>Learning XML</title>
    <author>Erik T. Ray</author>
    <year>2003</year>  # 描述一个XML数据中的年份字段为2003
    <price>39.95</price>  # 描述一个XML数据中的价格字段为39.95
  </row>  # XML数据的一行结束标记
# 以 XML 格式声明的字符串，包含 XML 版本和编码信息
na_expected = """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row>
    <index>0</index>
    <shape>square</shape>
    <degrees>360</degrees>
    <sides>4.0</sides>
  </row>
  <row>
    <index>1</index>
    <shape>circle</shape>
    <degrees>360</degrees>
    <sides/>
  </row>
  <row>
    <index>2</index>
    <shape>triangle</shape>
    <degrees>180</degrees>
    <sides>3.0</sides>
  </row>
</data>"""


def test_na_elem_output(parser, geom_df):
    # 调用 DataFrame 的方法将其转换为 XML 格式的字符串
    output = geom_df.to_xml(parser=parser)
    # 规范化 XML 声明，确保格式一致性
    output = equalize_decl(output)

    # 断言转换后的 XML 字符串与预期的 XML 字符串相等
    assert output == na_expected


def test_na_empty_str_elem_option(parser, geom_df):
    # 调用 DataFrame 的方法将其转换为 XML 格式的字符串，空值用空字符串表示
    output = geom_df.to_xml(na_rep="", parser=parser)
    # 规范化 XML 声明，确保格式一致性
    output = equalize_decl(output)
    # 使用断言来验证变量 output 是否等于变量 na_expected，用于测试代码逻辑的正确性
    assert output == na_expected
# 定义测试函数，测试在指定条件下是否能正确生成包含空元素的 XML 输出
def test_na_empty_elem_option(parser, geom_df):
    # 预期的 XML 输出，包含了空元素的数据
    expected = """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row>
    <index>0</index>
    <shape>square</shape>
    <degrees>360</degrees>
    <sides>4.0</sides>
  </row>
  <row>
    <index>1</index>
    <shape>circle</shape>
    <degrees>360</degrees>
    <sides>0.0</sides>
  </row>
  <row>
    <index>2</index>
    <shape>triangle</shape>
    <degrees>180</degrees>
    <sides>3.0</sides>
  </row>
</data>"""

    # 调用 geom_df.to_xml 方法生成 XML 输出
    output = geom_df.to_xml(na_rep="0.0", parser=parser)
    # 调用 equalize_decl 函数，确保 XML 声明的一致性
    output = equalize_decl(output)

    # 断言生成的 XML 输出与预期输出一致
    assert output == expected


# ATTR_COLS


# 测试将指定列作为属性输出到 XML 的功能
def test_attrs_cols_nan_output(parser, geom_df):
    # 预期的 XML 输出，将指定列作为属性输出
    expected = """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row index="0" shape="square" degrees="360" sides="4.0"/>
  <row index="1" shape="circle" degrees="360"/>
  <row index="2" shape="triangle" degrees="180" sides="3.0"/>
</data>"""

    # 调用 geom_df.to_xml 方法生成 XML 输出，指定要作为属性的列名
    output = geom_df.to_xml(attr_cols=["shape", "degrees", "sides"], parser=parser)
    # 调用 equalize_decl 函数，确保 XML 声明的一致性
    output = equalize_decl(output)

    # 断言生成的 XML 输出与预期输出一致
    assert output == expected


# 测试将属性列名使用指定的前缀和命名空间输出到 XML 的功能
def test_attrs_cols_prefix(parser, geom_df):
    # 预期的 XML 输出，使用指定的前缀和命名空间将列名作为属性输出
    expected = """\
<?xml version='1.0' encoding='utf-8'?>
<doc:data xmlns:doc="http://example.xom">
  <doc:row doc:index="0" doc:shape="square" doc:degrees="360" doc:sides="4.0"/>
  <doc:row doc:index="1" doc:shape="circle" doc:degrees="360"/>
  <doc:row doc:index="2" doc:shape="triangle" doc:degrees="180" doc:sides="3.0"/>
</doc:data>"""

    # 调用 geom_df.to_xml 方法生成 XML 输出，指定要作为属性的列名、命名空间和前缀
    output = geom_df.to_xml(
        attr_cols=["index", "shape", "degrees", "sides"],
        namespaces={"doc": "http://example.xom"},
        prefix="doc",
        parser=parser,
    )
    # 调用 equalize_decl 函数，确保 XML 声明的一致性
    output = equalize_decl(output)

    # 断言生成的 XML 输出与预期输出一致
    assert output == expected


# 测试在不存在的列名作为属性时是否能引发 KeyError 异常
def test_attrs_unknown_column(parser, geom_df):
    with pytest.raises(KeyError, match=("no valid column")):
        geom_df.to_xml(attr_cols=["shape", "degree", "sides"], parser=parser)


# 测试在指定错误类型的列名时是否能引发 TypeError 异常
def test_attrs_wrong_type(parser, geom_df):
    with pytest.raises(TypeError, match=("is not a valid type for attr_cols")):
        geom_df.to_xml(attr_cols='"shape", "degree", "sides"', parser=parser)


# ELEM_COLS


# 测试将指定列作为元素输出到 XML 的功能
def test_elems_cols_nan_output(parser, geom_df):
    # 预期的 XML 输出，将指定列作为元素输出
    elems_cols_expected = """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row>
    <degrees>360</degrees>
    <sides>4.0</sides>
    <shape>square</shape>
  </row>
  <row>
    <degrees>360</degrees>
    <sides/>
    <shape>circle</shape>
  </row>
  <row>
    <degrees>180</degrees>
    <sides>3.0</sides>
    <shape>triangle</shape>
  </row>
</data>"""

    # 调用 geom_df.to_xml 方法生成 XML 输出，指定要作为元素的列名
    output = geom_df.to_xml(
        index=False, elem_cols=["degrees", "sides", "shape"], parser=parser
    )
    # 调用 equalize_decl 函数，确保 XML 声明的一致性
    output = equalize_decl(output)

    # 断言生成的 XML 输出与预期输出一致
    assert output == elems_cols_expected


# 测试在不存在的列名作为元素时是否能引发 KeyError 异常
def test_elems_unknown_column(parser, geom_df):
    with pytest.raises(KeyError, match=("no valid column")):
        geom_df.to_xml(elem_cols=["shape", "degree", "sides"], parser=parser)


# 测试在指定错误类型的列名时是否能引发 TypeError 异常
def test_elems_wrong_type(parser, geom_df):
    # 使用 pytest 来测试异常情况，预期捕获 TypeError 异常，并匹配特定错误信息
    with pytest.raises(TypeError, match=("is not a valid type for elem_cols")):
        # 调用 geom_df 对象的 to_xml 方法，传递了错误的 elem_cols 参数
        # elem_cols 参数被错误地传递为一个字符串，应为一个有效的类型（可能是一个列表或元组）
        geom_df.to_xml(elem_cols='"shape", "degree", "sides"', parser=parser)
# 定义测试函数，用于验证生成 XML 的元素和属性列的处理
def test_elems_and_attrs_cols(parser, geom_df):
    # 预期的 XML 字符串，包含了各种形状的几何数据
    elems_cols_expected = """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row shape="square">
    <degrees>360</degrees>
    <sides>4.0</sides>
  </row>
  <row shape="circle">
    <degrees>360</degrees>
    <sides/>
  </row>
  <row shape="triangle">
    <degrees>180</degrees>
    <sides>3.0</sides>
  </row>
</data>"""

    # 调用 geom_df 的 to_xml 方法，生成 XML 字符串
    output = geom_df.to_xml(
        index=False,
        elem_cols=["degrees", "sides"],  # 指定作为 XML 元素的列
        attr_cols=["shape"],             # 指定作为 XML 属性的列
        parser=parser,
    )
    # 调用 equalize_decl 函数，用于规范化 XML 声明
    output = equalize_decl(output)

    # 断言生成的 XML 字符串与预期的字符串相等
    assert output == elems_cols_expected


# HIERARCHICAL COLUMNS


# 定义测试函数，用于验证处理层次化列的 XML 生成
def test_hierarchical_columns(parser, planet_df):
    # 预期的 XML 字符串，包含了各种行星的质量统计信息
    expected = """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row>
    <location>inner</location>
    <type>terrestrial</type>
    <count_mass>4</count_mass>
    <sum_mass>11.81</sum_mass>
    <mean_mass>2.95</mean_mass>
  </row>
  <row>
    <location>outer</location>
    <type>gas giant</type>
    <count_mass>2</count_mass>
    <sum_mass>2466.5</sum_mass>
    <mean_mass>1233.25</mean_mass>
  </row>
  <row>
    <location>outer</location>
    <type>ice giant</type>
    <count_mass>2</count_mass>
    <sum_mass>189.23</sum_mass>
    <mean_mass>94.61</mean_mass>
  </row>
  <row>
    <location>All</location>
    <type/>
    <count_mass>8</count_mass>
    <sum_mass>2667.54</sum_mass>
    <mean_mass>333.44</mean_mass>
  </row>
</data>"""

    # 对 planet_df 进行透视操作，计算质量的统计信息
    pvt = planet_df.pivot_table(
        index=["location", "type"],
        values="mass",
        aggfunc=["count", "sum", "mean"],
        margins=True,
    ).round(2)

    # 调用透视表的 to_xml 方法，生成 XML 字符串
    output = pvt.to_xml(parser=parser)
    # 调用 equalize_decl 函数，用于规范化 XML 声明
    output = equalize_decl(output)

    # 断言生成的 XML 字符串与预期的字符串相等
    assert output == expected


# 定义测试函数，用于验证处理层次化列及其属性的 XML 生成
def test_hierarchical_attrs_columns(parser, planet_df):
    # 预期的 XML 字符串，包含了各种行星的质量统计信息，使用属性表示列名
    expected = """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row location="inner" type="terrestrial" count_mass="4" \
sum_mass="11.81" mean_mass="2.95"/>
  <row location="outer" type="gas giant" count_mass="2" \
sum_mass="2466.5" mean_mass="1233.25"/>
  <row location="outer" type="ice giant" count_mass="2" \
sum_mass="189.23" mean_mass="94.61"/>
  <row location="All" type="" count_mass="8" \
sum_mass="2667.54" mean_mass="333.44"/>
</data>"""

    # 对 planet_df 进行透视操作，计算质量的统计信息
    pvt = planet_df.pivot_table(
        index=["location", "type"],
        values="mass",
        aggfunc=["count", "sum", "mean"],
        margins=True,
    ).round(2)

    # 获取透视表的列名，并将其作为属性列传递给 to_xml 方法
    output = pvt.to_xml(attr_cols=list(pvt.reset_index().columns.values), parser=parser)
    # 调用 equalize_decl 函数，用于规范化 XML 声明
    output = equalize_decl(output)

    # 断言生成的 XML 字符串与预期的字符串相等
    assert output == expected


# MULTIINDEX


# 定义测试函数，用于验证处理多级索引的 XML 生成
def test_multi_index(parser, planet_df):
    # 预期的 XML 字符串，包含了各种行星的质量统计信息
    expected = """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row>
    <location>inner</location>
    <type>terrestrial</type>
    <count>4</count>
    <sum>11.81</sum>
    <mean>2.95</mean>
  </row>
  <row>
    <location>outer</location>
    <type>gas giant</type>
    <count>2</count>
    <sum>2466.5</sum>
    <mean>1233.25</mean>
  </row>
  <row>
    <location>outer</location>
    <type>ice giant</type>  # XML元素：描述对象类型为"ice giant"
    <count>2</count>        # XML元素：指定数量为2个
    <sum>189.23</sum>       # XML元素：总和为189.23
    <mean>94.61</mean>      # XML元素：平均值为94.61
  </row>                    # XML元素：行结束标签
# 定义测试函数，用于测试将DataFrame分组聚合后转换为XML格式的输出
def test_default_namespace(parser, geom_df):
    # 期望的XML输出结果，带有默认命名空间声明
    expected = """\
<?xml version='1.0' encoding='utf-8'?>
<data xmlns="http://example.com">
  <row>
    <index>0</index>
    <shape>square</shape>
    <degrees>360</degrees>
    <sides>4.0</sides>
  </row>
  <row>
    <index>1</index>
    <shape>circle</shape>
    <degrees>360</degrees>
    <sides/>
  </row>
  <row>
    <index>2</index>
    <shape>triangle</shape>
    <degrees>180</degrees>
    <sides>3.0</sides>
  </row>
</data>"""

    # 使用DataFrame的to_xml方法将数据转换为XML格式，指定默认命名空间
    output = geom_df.to_xml(namespaces={"": "http://example.com"}, parser=parser)
    # 调用equalize_decl函数，确保XML声明一致性
    output = equalize_decl(output)

    # 断言生成的XML输出与期望的XML结果一致
    assert output == expected


def test_unused_namespaces(parser, geom_df):
    # 期望的XML输出结果，带有未使用的命名空间声明
    expected = """\
<?xml version='1.0' encoding='utf-8'?>
<data xmlns:oth="http://other.org" xmlns:ex="http://example.com">
  <row>
    <index>0</index>
    <shape>square</shape>
    <degrees>360</degrees>
    <sides>4.0</sides>
  </row>
  <row>
    <index>1</index>
    <shape>circle</shape>
    <degrees>360</degrees>
    <sides/>
  </row>
  <row>
    <index>2</index>
    <shape>triangle</shape>
    <degrees>180</degrees>
    <sides>3.0</sides>
  </row>
</data>"""

    # 使用DataFrame的to_xml方法将数据转换为XML格式，指定多个命名空间
    output = geom_df.to_xml(
        namespaces={"oth": "http://other.org", "ex": "http://example.com"},
        parser=parser,
    )
    # 调用equalize_decl函数，确保XML声明一致性
    output = equalize_decl(output)

    # 断言生成的XML输出与期望的XML结果一致
    assert output == expected


def test_namespace_prefix(parser, geom_df):
    # 期望的XML输出结果，带有命名空间前缀
    expected = """\
<?xml version='1.0' encoding='utf-8'?>
<doc:data xmlns:doc="http://example.com">
  <doc:row>
    <doc:index>0</doc:index>
    <doc:shape>square</doc:shape>
    <doc:degrees>360</doc:degrees>
    <doc:sides>4.0</doc:sides>
  </doc:row>
  <doc:row>
    <doc:index>1</doc:index>
    <doc:shape>circle</doc:shape>
    <doc:degrees>360</doc:degrees>
    <doc:sides/>
  </doc:row>
  <doc:row>
    <doc:index>2</doc:index>
    <doc:shape>triangle</doc:shape>
    <doc:degrees>180</doc:degrees>
    <doc:sides>3.0</doc:sides>
  </doc:row>
</doc:data>"""

    # 使用DataFrame的to_xml方法将数据转换为XML格式，指定命名空间前缀
    output = geom_df.to_xml(namespaces={"doc": "http://example.com"}, parser=parser)

    # 注意：由于此处只有部分代码提供，缺少equalize_decl函数的实现，无法完成最后的处理步骤

    # 未能完整展示代码块，无法输出
    # 使用 geom_df 数据框架生成 XML 格式的输出
    output = geom_df.to_xml(
        namespaces={"doc": "http://example.com"}, prefix="doc", parser=parser
    )
    # 调用 equalize_decl 函数处理输出，确保 XML 声明的一致性
    output = equalize_decl(output)

    # 断言输出是否与预期结果 expected 相等，用于测试结果的正确性
    assert output == expected
# 检验当命名空间中缺少 "doc" 前缀时是否会引发 KeyError 异常
def test_missing_prefix_in_nmsp(parser, geom_df):
    with pytest.raises(KeyError, match=("doc is not included in namespaces")):
        geom_df.to_xml(
            namespaces={"": "http://example.com"}, prefix="doc", parser=parser
        )


# 测试同时指定了默认命名空间和 "doc" 前缀时的预期 XML 输出
def test_namespace_prefix_and_default(parser, geom_df):
    expected = """\
<?xml version='1.0' encoding='utf-8'?>
<doc:data xmlns:doc="http://other.org" xmlns="http://example.com">
  <doc:row>
    <doc:index>0</doc:index>
    <doc:shape>square</doc:shape>
    <doc:degrees>360</doc:degrees>
    <doc:sides>4.0</doc:sides>
  </doc:row>
  <doc:row>
    <doc:index>1</doc:index>
    <doc:shape>circle</doc:shape>
    <doc:degrees>360</doc:degrees>
    <doc:sides/>
  </doc:row>
  <doc:row>
    <doc:index>2</doc:index>
    <doc:shape>triangle</doc:shape>
    <doc:degrees>180</doc:degrees>
    <doc:sides>3.0</doc:sides>
  </doc:row>
</doc:data>"""

    output = geom_df.to_xml(
        namespaces={"": "http://example.com", "doc": "http://other.org"},
        prefix="doc",
        parser=parser,
    )
    output = equalize_decl(output)

    assert output == expected


# 定义预期的 XML 输出，包含指定的编码声明
encoding_expected = """\
<?xml version='1.0' encoding='ISO-8859-1'?>
<data>
  <row>
    <index>0</index>
    <rank>1</rank>
    <malename>José</malename>
    <femalename>Sofía</femalename>
  </row>
  <row>
    <index>1</index>
    <rank>2</rank>
    <malename>Luis</malename>
    <femalename>Valentina</femalename>
  </row>
  <row>
    <index>2</index>
    <rank>3</rank>
    <malename>Carlos</malename>
    <femalename>Isabella</femalename>
  </row>
  <row>
    <index>3</index>
    <rank>4</rank>
    <malename>Juan</malename>
    <femalename>Camila</femalename>
  </row>
  <row>
    <index>4</index>
    <rank>5</rank>
    <malename>Jorge</malename>
    <femalename>Valeria</femalename>
  </row>
</data>"""


# 测试指定编码选项为字符串时的 XML 输出
def test_encoding_option_str(xml_baby_names, parser):
    df_file = read_xml(xml_baby_names, parser=parser, encoding="ISO-8859-1").head(5)

    output = df_file.to_xml(encoding="ISO-8859-1", parser=parser)

    if output is not None:
        # lxml 和 etree 在 XML 声明中的引号和大小写上有所不同
        output = output.replace(
            '<?xml version="1.0" encoding="ISO-8859-1"?',
            "<?xml version='1.0' encoding='ISO-8859-1'?",
        )

    assert output == encoding_expected


# 测试正确指定编码时的文件输出
def test_correct_encoding_file(xml_baby_names):
    pytest.importorskip("lxml")
    df_file = read_xml(xml_baby_names, encoding="ISO-8859-1", parser="lxml")

    with tm.ensure_clean("test.xml") as path:
        df_file.to_xml(path, index=False, encoding="ISO-8859-1", parser="lxml")


# 使用参数化测试不正确的编码选项时的行为
@pytest.mark.parametrize("encoding", ["UTF-8", "UTF-16", "ISO-8859-1"])
def test_wrong_encoding_option_lxml(xml_baby_names, parser, encoding):
    pytest.importorskip("lxml")
    df_file = read_xml(xml_baby_names, encoding="ISO-8859-1", parser="lxml")
    # 使用 tm.ensure_clean 上下文管理器创建一个临时文件 "test.xml"，确保操作完成后文件会被清理
    with tm.ensure_clean("test.xml") as path:
        # 将 DataFrame df_file 转换为 XML 格式，并写入指定的路径
        # index=False 表示不包含行索引，encoding 指定编码方式，parser 指定解析器
        df_file.to_xml(path, index=False, encoding=encoding, parser=parser)
def test_misspelled_encoding(parser, geom_df):
    # 测试处理未正确拼写的编码名称时是否引发 LookupError 异常
    with pytest.raises(LookupError, match=("unknown encoding")):
        geom_df.to_xml(encoding="uft-8", parser=parser)


# PRETTY PRINT


def test_xml_declaration_pretty_print(geom_df):
    # 在使用 lxml 模块的情况下，测试 geom_df 对象转换为 XML 格式字符串，并验证格式化后的输出是否符合预期
    pytest.importorskip("lxml")
    expected = """\
<data>
  <row>
    <index>0</index>
    <shape>square</shape>
    <degrees>360</degrees>
    <sides>4.0</sides>
  </row>
  <row>
    <index>1</index>
    <shape>circle</shape>
    <degrees>360</degrees>
    <sides/>
  </row>
  <row>
    <index>2</index>
    <shape>triangle</shape>
    <degrees>180</degrees>
    <sides>3.0</sides>
  </row>
</data>"""

    # 调用 geom_df.to_xml 方法，设置 xml_declaration=False 参数，生成 XML 格式字符串并与预期输出进行比较
    output = geom_df.to_xml(xml_declaration=False)

    assert output == expected


def test_no_pretty_print_with_decl(parser, geom_df):
    # 测试在设置 pretty_print=False 的情况下，geom_df 对象转换为 XML 格式字符串的输出是否符合预期
    expected = (
        "<?xml version='1.0' encoding='utf-8'?>\n"
        "<data><row><index>0</index><shape>square</shape>"
        "<degrees>360</degrees><sides>4.0</sides></row><row>"
        "<index>1</index><shape>circle</shape><degrees>360"
        "</degrees><sides/></row><row><index>2</index><shape>"
        "triangle</shape><degrees>180</degrees><sides>3.0</sides>"
        "</row></data>"
    )

    # 调用 geom_df.to_xml 方法，设置 pretty_print=False 参数，生成 XML 格式字符串并与预期输出进行比较
    output = geom_df.to_xml(pretty_print=False, parser=parser)
    output = equalize_decl(output)

    # etree 在关闭标签时会添加空格，所以需要将输出中的 " />" 替换为 "/>"
    if output is not None:
        output = output.replace(" />", "/>")

    assert output == expected


def test_no_pretty_print_no_decl(parser, geom_df):
    # 测试在设置 pretty_print=False 和 xml_declaration=False 的情况下，geom_df 对象转换为 XML 格式字符串的输出是否符合预期
    expected = (
        "<data><row><index>0</index><shape>square</shape>"
        "<degrees>360</degrees><sides>4.0</sides></row><row>"
        "<index>1</index><shape>circle</shape><degrees>360"
        "</degrees><sides/></row><row><index>2</index><shape>"
        "triangle</shape><degrees>180</degrees><sides>3.0</sides>"
        "</row></data>"
    )

    # 调用 geom_df.to_xml 方法，设置 pretty_print=False 和 xml_declaration=False 参数，生成 XML 格式字符串并与预期输出进行比较
    output = geom_df.to_xml(xml_declaration=False, pretty_print=False, parser=parser)

    # etree 在关闭标签时会添加空格，所以需要将输出中的 " />" 替换为 "/>"
    if output is not None:
        output = output.replace(" />", "/>")

    assert output == expected


# PARSER


@td.skip_if_installed("lxml")
def test_default_parser_no_lxml(geom_df):
    # 测试在没有安装 lxml 模块时，调用 geom_df.to_xml 方法是否会引发 ImportError 异常
    with pytest.raises(
        ImportError, match=("lxml not found, please install or use the etree parser.")
    ):
        geom_df.to_xml()


def test_unknown_parser(geom_df):
    # 测试在传递未知的解析器名称时，调用 geom_df.to_xml 方法是否会引发 ValueError 异常
    with pytest.raises(
        ValueError, match=("Values for parser can only be lxml or etree.")
    ):
        geom_df.to_xml(parser="bs4")


# STYLESHEET

xsl_expected = """\
<?xml version="1.0" encoding="utf-8"?>
<data>
  <row>
    <field field="index">0</field>
    <field field="shape">square</field>
    <field field="degrees">360</field>
    <field field="sides">4.0</field>
  </row>
  <row>
    <field field="index">1</field>
    <field field="shape">circle</field>
    <field field="degrees">360</field>
    <field field="sides"/>
  </row>
  <row>
    <field field="index">2</field>
    <field field="shape">triangle</field>
    # XML中的一个字段，指定字段名为"degrees"，字段值为"180"
    <field field="degrees">180</field>
    # XML中的一个字段，指定字段名为"sides"，字段值为"3.0"
    <field field="sides">3.0</field>
  </row>
def test_stylesheet_file_like(xsl_row_field_output, mode, geom_df):
    # 导入必要的 pytest 模块，如果未安装 pytest 则跳过测试
    pytest.importorskip("lxml")
    # 打开指定路径的 XSL 文件，根据指定的模式和编码方式读取文件内容
    with open(
        xsl_row_field_output, mode, encoding="utf-8" if mode == "r" else None
    ) as f:
        # 使用 geom_df 对象生成 XML 输出，并使用打开的 XSL 文件作为样式表
        assert geom_df.to_xml(stylesheet=f) == xsl_expected


def test_stylesheet_io(xsl_row_field_output, mode, geom_df):
    # 注意：默认情况下，不检查未声明类型的函数体，可以考虑使用 --check-untyped-defs 选项
    pytest.importorskip("lxml")
    # 声明一个变量 xsl_obj，类型可以是 BytesIO 或 StringIO
    xsl_obj: BytesIO | StringIO  # type: ignore[annotation-unchecked]

    # 打开指定路径的 XSL 文件，根据指定的模式和编码方式读取文件内容
    with open(
        xsl_row_field_output, mode, encoding="utf-8" if mode == "r" else None
    ) as f:
        # 如果模式是二进制读取，则将文件内容读取到 BytesIO 对象中
        if mode == "rb":
            xsl_obj = BytesIO(f.read())
        else:
            # 否则将文件内容读取到 StringIO 对象中
            xsl_obj = StringIO(f.read())

    # 使用 geom_df 对象生成 XML 输出，并使用 xsl_obj 对象作为样式表
    output = geom_df.to_xml(stylesheet=xsl_obj)

    # 断言输出结果与预期的 xsl_expected 相等
    assert output == xsl_expected


def test_stylesheet_buffered_reader(xsl_row_field_output, mode, geom_df):
    pytest.importorskip("lxml")
    # 打开指定路径的 XSL 文件，根据指定的模式和编码方式读取文件内容
    with open(
        xsl_row_field_output, mode, encoding="utf-8" if mode == "r" else None
    ) as f:
        # 使用 geom_df 对象生成 XML 输出，并使用打开的 XSL 文件作为样式表
        output = geom_df.to_xml(stylesheet=f)

    # 断言输出结果与预期的 xsl_expected 相等
    assert output == xsl_expected


def test_stylesheet_wrong_path(geom_df):
    pytest.importorskip("lxml.etree")

    # 构建一个不存在的 XSL 文件路径
    xsl = os.path.join("does", "not", "exist", "row_field_output.xslt")

    # 使用 pytest 的断言来验证文件不存在的异常是否被抛出
    with pytest.raises(
        FileNotFoundError, match=r"\[Errno 2\] No such file or director"
    ):
        geom_df.to_xml(stylesheet=xsl)


@pytest.mark.parametrize("val", [StringIO(""), BytesIO(b"")])
def test_empty_string_stylesheet(val, geom_df):
    lxml_etree = pytest.importorskip("lxml.etree")

    # 准备用于测试空字符串样式表的异常消息
    msg = "|".join(
        [
            "Document is empty",
            "Start tag expected, '<' not found",
            # Seen on Mac with lxml 4.9.1
            r"None \(line 0\)",
        ]
    )

    # 使用 pytest 的断言来验证 XML 语法错误异常是否被抛出
    with pytest.raises(lxml_etree.XMLSyntaxError, match=msg):
        geom_df.to_xml(stylesheet=val)


def test_incorrect_xsl_syntax(geom_df):
    lxml_etree = pytest.importorskip("lxml.etree")

    # 准备一个含有语法错误的 XSL 样式表
    xsl = """\
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="xml" encoding="utf-8" indent="yes" >
    <xsl:strip-space elements="*"/>

    <xsl:template match="@*|node()">
        <xsl:copy>
            <xsl:apply-templates select="@*|node()"/>
        </xsl:copy>
    </xsl:template>

    <xsl:template match="row/*">
        <field>
            <xsl:attribute name="field">
                <xsl:value-of select="name()"/>
            </xsl:attribute>
            <xsl:value-of select="text()"/>
        </field>
    </xsl:template>
</xsl:stylesheet>"""

    # 使用 pytest 的断言来验证 XML 语法错误异常是否被抛出
    with pytest.raises(
        lxml_etree.XMLSyntaxError, match="Opening and ending tag mismatch"
    ):
        geom_df.to_xml(stylesheet=StringIO(xsl))


def test_incorrect_xsl_eval(geom_df):
    lxml_etree = pytest.importorskip("lxml.etree")

    # 准备一个含有评估错误的 XSL 样式表
    xsl = """\
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="xml" encoding="utf-8" indent="yes" />
    <xsl:strip-space elements="*"/>
    
    <xsl:template match="@*|node(*)">
        <!-- 匹配所有属性节点和元素节点 -->
        <xsl:copy>
            <!-- 复制当前节点 -->
            <xsl:apply-templates select="@*|node()"/>
            <!-- 应用模板到当前节点的所有属性和子节点 -->
        </xsl:copy>
    </xsl:template>
    
    <xsl:template match="row/*">
        <!-- 匹配名为 row 的元素的所有子元素 -->
        <field>
            <!-- 创建一个名为 field 的新元素 -->
            <xsl:attribute name="field">
                <!-- 在 field 元素上添加名为 field 的属性 -->
                <xsl:value-of select="name()"/>
                <!-- 设置属性值为当前子元素的名称 -->
            </xsl:attribute>
            <!-- 输出当前子元素的文本内容 -->
            <xsl:value-of select="text()"/>
        </field>
    </xsl:template>
def test_style_to_string(geom_df):
    pytest.importorskip("lxml")
    xsl = """\
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="text" indent="yes" />
    <xsl:strip-space elements="*"/>

    <xsl:param name="delim"><xsl:text>               </xsl:text></xsl:param>
    <xsl:template match="/data">
        <xsl:text>      shape  degrees  sides&#xa;</xsl:text>
        <xsl:apply-templates select="row"/>
    </xsl:template>

  
# 将DataFrame转换为XML字符串，并根据指定的XSLT样式表进行样式化输出

out_csv = geom_df.to_csv(lineterminator="\n")

# 如果输出的CSV不为空，则去除首尾空白字符
if out_csv is not None:
    out_csv = out_csv.strip()

# 使用指定的XSLT样式表将DataFrame转换为XML字符串
out_xml = geom_df.to_xml(stylesheet=StringIO(xsl))

# 断言转换后的CSV字符串与XML字符串相等
assert out_csv == out_xml
    # XSL 模板匹配行元素 "row"
    <xsl:template match="row">
        # 使用 xsl:value-of 输出拼接的文本内容，包括 index, shape, degrees, sides 等字段
        <xsl:value-of select="concat(index, ' ',
                                     substring($delim, 1, string-length('triangle')
                                               - string-length(shape) + 1),
                                     shape,
                                     substring($delim, 1, string-length(name(degrees))
                                               - string-length(degrees) + 2),
                                     degrees,
                                     substring($delim, 1, string-length(name(sides))
                                               - string-length(sides) + 2),
                                     sides)"/>
        # 输出换行符
        <xsl:text>&#xa;</xsl:text>
    </xsl:template>
geom_xml = """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row>
    <index>0</index>
    <shape>square</shape>
    <degrees>360</degrees>
    <sides>4.0</sides>
  </row>
  <row>
    <index>1</index>
    <shape>circle</shape>
    <degrees>360</degrees>
    <sides/>
  </row>
  <row>
    <index>2</index>
    <shape>triangle</shape>
    <degrees>180</degrees>
    <sides>3.0</sides>
  </row>
</data>"""



# 定义一个包含几何形状信息的 XML 字符串
geom_xml = """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row>
    <index>0</index>
    <shape>square</shape>
    <degrees>360</degrees>
    <sides>4.0</sides>
  </row>
  <row>
    <index>1</index>
    <shape>circle</shape>
    <degrees>360</degrees>
    <sides/>
  </row>
  <row>
    <index>2</index>
    <shape>triangle</shape>
    <degrees>180</degrees>
    <sides>3.0</sides>
  </row>
</data>"""



def test_compression_output(parser, compression_only, geom_df):
    with tm.ensure_clean() as path:
        # 将 DataFrame 转换为 XML 格式并写入文件
        geom_df.to_xml(path, parser=parser, compression=compression_only)

        # 从生成的文件中读取数据
        with get_handle(
            path,
            "r",
            compression=compression_only,
        ) as handle_obj:
            # 读取文件内容
            output = handle_obj.handle.read()

    # 标准化 XML 声明以及末尾的空白符
    output = equalize_decl(output)

    # 断言生成的 XML 输出与预期的 XML 字符串相匹配
    assert geom_xml == output.strip()



def test_filename_and_suffix_comp(
    parser, compression_only, geom_df, compression_to_extension
):
    # 构造压缩后的文件名，后缀为根据压缩类型映射的扩展名
    compfile = "xml." + compression_to_extension[compression_only]
    # 使用 tm.ensure_clean 函数确保在处理 compfile 文件时是干净的，返回处理后的路径
    with tm.ensure_clean(filename=compfile) as path:
        # 将 geom_df 数据框以 XML 格式写入到指定路径的文件中，使用指定的解析器和压缩方式
        geom_df.to_xml(path, parser=parser, compression=compression_only)

        # 使用 get_handle 函数获取指定路径文件的句柄对象，以只读方式打开，并指定压缩方式
        with get_handle(
            path,
            "r",
            compression=compression_only,
        ) as handle_obj:
            # 读取句柄对象中的内容赋给 output
            output = handle_obj.handle.read()

    # 对 output 进行声明均衡化处理
    output = equalize_decl(output)

    # 断言 geom_xml 等于去除首尾空白字符后的 output
    assert geom_xml == output.strip()
# GH#43903
def test_ea_dtypes(any_numeric_ea_dtype, parser):
    # 定义预期的 XML 字符串
    expected = """<?xml version='1.0' encoding='utf-8'?>
<data>
  <row>
    <index>0</index>
    <a/>
  </row>
</data>"""
    # 创建一个 DataFrame，其中只有一个列 'a'，并将其转换为指定的数据类型 any_numeric_ea_dtype
    df = DataFrame({"a": [NA]}).astype(any_numeric_ea_dtype)
    # 将 DataFrame 转换为 XML 格式的字符串，使用指定的解析器 parser
    result = df.to_xml(parser=parser)
    # 断言结果的 XML 声明部分被规范化后与预期值相等
    assert equalize_decl(result).strip() == expected


def test_unsuported_compression(parser, geom_df):
    # 使用 pytest 断言，检测是否会抛出 ValueError，且错误信息包含"Unrecognized compression type"
    with pytest.raises(ValueError, match="Unrecognized compression type"):
        # 确保在测试过程中 path 路径是干净的
        with tm.ensure_clean() as path:
            # 将 geom_df 写入到 XML 文件中，指定使用的解析器 parser，并使用不支持的压缩类型"7z"
            geom_df.to_xml(path, parser=parser, compression="7z")


# STORAGE OPTIONS


@pytest.mark.single_cpu
def test_s3_permission_output(parser, s3_public_bucket, geom_df):
    # 导入 s3fs 库，如果导入失败则跳过测试
    s3fs = pytest.importorskip("s3fs")
    # 导入 lxml 库，如果导入失败则跳过测试
    pytest.importorskip("lxml")

    # 检测是否会引发 PermissionError 或 FileNotFoundError 异常
    with tm.external_error_raised((PermissionError, FileNotFoundError)):
        # 创建 S3 文件系统对象，匿名方式连接
        fs = s3fs.S3FileSystem(anon=True)
        # 列出 S3 存储桶中的文件列表
        fs.ls(s3_public_bucket.name)

        # 将 geom_df 数据框写入到指定 S3 存储桶的 XML 文件中，使用 ZIP 压缩，指定解析器 parser
        geom_df.to_xml(
            f"s3://{s3_public_bucket.name}/geom.xml", compression="zip", parser=parser
        )
```