# `D:\src\scipysrc\pandas\pandas\tests\io\xml\test_xml_dtypes.py`

```
# 导入必要的模块和函数

from __future__ import annotations  # 使用未来版本的类型注解

from io import StringIO  # 导入StringIO模块，用于内存中操作字符串的I/O

import pytest  # 导入pytest模块，用于单元测试

from pandas.errors import ParserWarning  # 导入ParserWarning，用于处理解析器警告
import pandas.util._test_decorators as td  # 导入测试装饰器模块

from pandas import (  # 导入pandas库中的多个函数和类
    DataFrame,  # 数据帧类
    DatetimeIndex,  # 日期时间索引类
    Series,  # 系列类
    to_datetime,  # 转换为日期时间对象的函数
)
import pandas._testing as tm  # 导入测试辅助工具模块

from pandas.io.xml import read_xml  # 导入读取XML文件的函数

# 定义pytest的fixture，用于参数化测试的解析器选择
@pytest.fixture(params=[pytest.param("lxml", marks=td.skip_if_no("lxml")), "etree"])
def parser(request):
    return request.param

# 定义pytest的fixture，用于参数化测试的解析器选择和迭代解析设置
@pytest.fixture(
    params=[None, {"book": ["category", "title", "author", "year", "price"]}]
)
def iterparse(request):
    return request.param

# 定义一个函数，读取XML并进行迭代解析处理
def read_xml_iterparse(data, **kwargs):
    with tm.ensure_clean() as path:  # 使用测试辅助工具确保路径干净
        with open(path, "w", encoding="utf-8") as f:  # 打开并写入数据到指定路径的文件中
            f.write(data)
        return read_xml(path, **kwargs)  # 调用read_xml函数读取并解析XML文件

# 定义XML数据字符串，包含形状、角度和边数信息
xml_types = """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row>
    <shape>square</shape>
    <degrees>00360</degrees>
    <sides>4.0</sides>
   </row>
  <row>
    <shape>circle</shape>
    <degrees>00360</degrees>
    <sides/>
  </row>
  <row>
    <shape>triangle</shape>
    <degrees>00180</degrees>
    <sides>3.0</sides>
  </row>
</data>"""

# 定义XML数据字符串，包含形状、角度、边数和日期信息
xml_dates = """<?xml version='1.0' encoding='utf-8'?>
<data>
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
</data>"""

# 测试函数：测试以单个字符串指定的数据类型为字符串的情况
def test_dtype_single_str(parser):
    df_result = read_xml(StringIO(xml_types), dtype={"degrees": "str"}, parser=parser)  # 读取XML数据并将“degrees”列的数据类型指定为字符串
    df_iter = read_xml_iterparse(
        xml_types,
        parser=parser,
        dtype={"degrees": "str"},
        iterparse={"row": ["shape", "degrees", "sides"]},  # 迭代解析只保留指定列
    )

    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": ["00360", "00360", "00180"],
            "sides": [4.0, float("nan"), 3.0],  # 使用浮点数表示边数，未定义的值使用NaN
        }
    )

    tm.assert_frame_equal(df_result, df_expected)  # 断言DataFrame的结果符合预期
    tm.assert_frame_equal(df_iter, df_expected)  # 断言迭代解析的DataFrame结果符合预期

# 测试函数：测试所有数据类型都为字符串的情况
def test_dtypes_all_str(parser):
    df_result = read_xml(StringIO(xml_dates), dtype="string", parser=parser)  # 读取XML数据并将所有列的数据类型指定为字符串
    df_iter = read_xml_iterparse(
        xml_dates,
        parser=parser,
        dtype="string",
        iterparse={"row": ["shape", "degrees", "sides", "date"]},  # 迭代解析保留所有列
    )

    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": ["00360", "00360", "00180"],
            "sides": ["4.0", None, "3.0"],  # 使用字符串表示边数，未定义的值使用None
            "date": ["2020-01-01", "2021-01-01", "2022-01-01"],  # 日期以字符串形式表示
        },
        dtype="string",
    )

    tm.assert_frame_equal(df_result, df_expected)  # 断言DataFrame的结果符合预期
    tm.assert_frame_equal(df_iter, df_expected)  # 断言迭代解析的DataFrame结果符合预期

# 测试函数：测试包含名称的数据类型情况
def test_dtypes_with_names(parser):
    pass  # 这个测试函数暂时未实现，保留了一个占位符
    # 调用 read_xml 函数读取 XML 数据，返回结果 DataFrame
    df_result = read_xml(
        StringIO(xml_dates),  # 使用 StringIO 将 XML 数据封装成文本流
        names=["Col1", "Col2", "Col3", "Col4"],  # 指定生成的 DataFrame 的列名
        dtype={"Col2": "string", "Col3": "Int64", "Col4": "datetime64[ns]"},  # 指定每列的数据类型
        parser=parser,  # 传递解析器参数给 read_xml 函数
    )
    
    # 调用 read_xml_iterparse 函数使用迭代解析方式读取 XML 数据，返回结果 DataFrame
    df_iter = read_xml_iterparse(
        xml_dates,  # XML 数据字符串
        parser=parser,  # 解析器参数
        names=["Col1", "Col2", "Col3", "Col4"],  # 列名
        dtype={"Col2": "string", "Col3": "Int64", "Col4": "datetime64[ns]"},  # 数据类型
        iterparse={"row": ["shape", "degrees", "sides", "date"]},  # 迭代解析参数
    )
    
    # 创建预期的 DataFrame，包含预期结果的列和数据
    df_expected = DataFrame(
        {
            "Col1": ["square", "circle", "triangle"],
            "Col2": Series(["00360", "00360", "00180"]).astype("string"),
            "Col3": Series([4.0, float("nan"), 3.0]).astype("Int64"),
            "Col4": DatetimeIndex(
                ["2020-01-01", "2021-01-01", "2022-01-01"], dtype="M8[ns]"
            ),
        }
    )
    
    # 使用 tm.assert_frame_equal 检查 df_result 和 df_expected 是否相等
    tm.assert_frame_equal(df_result, df_expected)
    
    # 使用 tm.assert_frame_equal 检查 df_iter 和 df_expected 是否相等
    tm.assert_frame_equal(df_iter, df_expected)
# 测试函数，用于测试 read_xml 和 read_xml_iterparse 函数对于可空整数类型的处理
def test_dtype_nullable_int(parser):
    # 使用 read_xml 函数读取 XML 数据，指定 sides 列为 Int64 类型，并传入解析器 parser
    df_result = read_xml(StringIO(xml_types), dtype={"sides": "Int64"}, parser=parser)
    
    # 使用 read_xml_iterparse 函数逐行解析 XML 数据，指定 sides 列为 Int64 类型，并传入解析器 parser
    df_iter = read_xml_iterparse(
        xml_types,
        parser=parser,
        dtype={"sides": "Int64"},
        iterparse={"row": ["shape", "degrees", "sides"]},
    )

    # 期望的 DataFrame 结果，包括 shape、degrees 和 sides 列
    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": [360, 360, 180],
            "sides": Series([4.0, float("nan"), 3.0]).astype("Int64"),
        }
    )

    # 使用 assert_frame_equal 函数比较 df_result 和 df_expected，确认它们相等
    tm.assert_frame_equal(df_result, df_expected)
    # 使用 assert_frame_equal 函数比较 df_iter 和 df_expected，确认它们相等
    tm.assert_frame_equal(df_iter, df_expected)


# 测试函数，用于测试 read_xml 和 read_xml_iterparse 函数对于浮点数类型的处理
def test_dtype_float(parser):
    # 使用 read_xml 函数读取 XML 数据，指定 degrees 列为 float 类型，并传入解析器 parser
    df_result = read_xml(StringIO(xml_types), dtype={"degrees": "float"}, parser=parser)
    
    # 使用 read_xml_iterparse 函数逐行解析 XML 数据，指定 degrees 列为 float 类型，并传入解析器 parser
    df_iter = read_xml_iterparse(
        xml_types,
        parser=parser,
        dtype={"degrees": "float"},
        iterparse={"row": ["shape", "degrees", "sides"]},
    )

    # 期望的 DataFrame 结果，包括 shape、degrees 和 sides 列
    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": Series([360, 360, 180]).astype("float"),
            "sides": [4.0, float("nan"), 3.0],
        }
    )

    # 使用 assert_frame_equal 函数比较 df_result 和 df_expected，确认它们相等
    tm.assert_frame_equal(df_result, df_expected)
    # 使用 assert_frame_equal 函数比较 df_iter 和 df_expected，确认它们相等
    tm.assert_frame_equal(df_iter, df_expected)


# 测试函数，用于测试 read_xml 函数对于错误 dtype 设置的处理
def test_wrong_dtype(xml_books, parser, iterparse):
    # 使用 pytest.raises 确认 read_xml 函数对于设定错误的 dtype 抛出 ValueError 异常，
    # 并匹配特定错误消息
    with pytest.raises(
        ValueError, match=('Unable to parse string "Everyday Italian" at position 0')
    ):
        # 调用 read_xml 函数，传入错误的 dtype 设置和 parser
        read_xml(
            xml_books, dtype={"title": "Int64"}, parser=parser, iterparse=iterparse
        )


# 测试函数，用于测试 read_xml 和 read_xml_iterparse 函数同时使用 converters 和 dtype 的处理
def test_both_dtype_converters(parser):
    # 期望的 DataFrame 结果，包括 shape、degrees 和 sides 列，degrees 列被转换为字符串
    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": ["00360", "00360", "00180"],
            "sides": [4.0, float("nan"), 3.0],
        }
    )

    # 使用 assert_produces_warning 函数确认在同时使用 converters 和 dtype 时会产生 ParserWarning 警告
    with tm.assert_produces_warning(ParserWarning, match="Both a converter and dtype"):
        # 调用 read_xml 函数，传入 converters 和 dtype 设置，并使用 parser 解析
        df_result = read_xml(
            StringIO(xml_types),
            dtype={"degrees": "str"},
            converters={"degrees": str},
            parser=parser,
        )
        
        # 调用 read_xml_iterparse 函数，传入 converters 和 dtype 设置，并使用 parser 解析
        df_iter = read_xml_iterparse(
            xml_types,
            dtype={"degrees": "str"},
            converters={"degrees": str},
            parser=parser,
            iterparse={"row": ["shape", "degrees", "sides"]},
        )

        # 使用 assert_frame_equal 函数比较 df_result 和 df_expected，确认它们相等
        tm.assert_frame_equal(df_result, df_expected)
        # 使用 assert_frame_equal 函数比较 df_iter 和 df_expected，确认它们相等
        tm.assert_frame_equal(df_iter, df_expected)


# CONVERTERS


# 测试函数，用于测试 read_xml 和 read_xml_iterparse 函数对于 degrees 列转换为字符串的处理
def test_converters_str(parser):
    # 使用 read_xml 函数读取 XML 数据，指定 degrees 列使用 converters 转换为字符串，并传入解析器 parser
    df_result = read_xml(
        StringIO(xml_types), converters={"degrees": str}, parser=parser
    )
    
    # 使用 read_xml_iterparse 函数逐行解析 XML 数据，指定 degrees 列使用 converters 转换为字符串，并传入解析器 parser
    df_iter = read_xml_iterparse(
        xml_types,
        parser=parser,
        converters={"degrees": str},
        iterparse={"row": ["shape", "degrees", "sides"]},
    )

    # 期望的 DataFrame 结果，包括 shape、degrees 和 sides 列，degrees 列被转换为字符串
    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": ["00360", "00360", "00180"],
            "sides": [4.0, float("nan"), 3.0],
        }
    )

    # 使用 assert_frame_equal 函数比较 df_result 和 df_expected，确认它们相等
    tm.assert_frame_equal(df_result, df_expected)
    # 使用测试工具比较两个数据帧是否相等，并断言它们相等
    tm.assert_frame_equal(df_iter, df_expected)
# 测试日期转换器函数
def test_converters_date(parser):
    # 创建 lambda 函数用于将数据转换为 datetime 对象
    convert_to_datetime = lambda x: to_datetime(x)
    # 读取 XML 数据并使用指定的转换器转换日期列，返回数据框 df_result
    df_result = read_xml(
        StringIO(xml_dates), converters={"date": convert_to_datetime}, parser=parser
    )
    # 使用迭代解析读取 XML 数据，同时指定转换器和解析器选项，返回数据框 df_iter
    df_iter = read_xml_iterparse(
        xml_dates,
        parser=parser,
        converters={"date": convert_to_datetime},
        iterparse={"row": ["shape", "degrees", "sides", "date"]},
    )

    # 期望的数据框 df_expected，包含预期结果的列与数据
    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": [360, 360, 180],
            "sides": [4.0, float("nan"), 3.0],
            "date": to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
        }
    )

    # 断言两个数据框是否相等，用于测试结果的一致性
    tm.assert_frame_equal(df_result, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


# 测试错误的转换器类型
def test_wrong_converters_type(xml_books, parser, iterparse):
    # 使用 pytest 断言检查是否抛出了 TypeError 异常，异常消息匹配指定内容
    with pytest.raises(TypeError, match=("Type converters must be a dict or subclass")):
        # 读取 XML 数据时传入错误类型的转换器，应当抛出异常
        read_xml(
            xml_books, converters={"year", str}, parser=parser, iterparse=iterparse
        )


# 测试可调用函数转换器的异常情况
def test_callable_func_converters(xml_books, parser, iterparse):
    # 使用 pytest 断言检查是否抛出了 TypeError 异常，异常消息匹配指定内容
    with pytest.raises(TypeError, match=("'float' object is not callable")):
        # 读取 XML 数据时传入非可调用对象作为转换器，应当抛出异常
        read_xml(
            xml_books, converters={"year": float()}, parser=parser, iterparse=iterparse
        )


# 测试字符串作为转换器的异常情况
def test_callable_str_converters(xml_books, parser, iterparse):
    # 使用 pytest 断言检查是否抛出了 TypeError 异常，异常消息匹配指定内容
    with pytest.raises(TypeError, match=("'str' object is not callable")):
        # 读取 XML 数据时传入字符串作为转换器，应当抛出异常
        read_xml(
            xml_books, converters={"year": "float"}, parser=parser, iterparse=iterparse
        )


# 测试解析日期选项为 True 的情况
def test_parse_dates_true(parser):
    # 读取 XML 数据时将 parse_dates 设置为 True，实现日期解析
    df_result = read_xml(StringIO(xml_dates), parse_dates=True, parser=parser)
    # 使用迭代解析读取 XML 数据，同时指定 parse_dates 为 True
    df_iter = read_xml_iterparse(
        xml_dates,
        parser=parser,
        parse_dates=True,
        iterparse={"row": ["shape", "degrees", "sides", "date"]},
    )

    # 期望的数据框 df_expected，包含预期结果的列与数据
    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": [360, 360, 180],
            "sides": [4.0, float("nan"), 3.0],
            "date": to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
        }
    )

    # 断言两个数据框是否相等，用于测试结果的一致性
    tm.assert_frame_equal(df_result, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


# 测试按列名解析日期
def test_parse_dates_column_name(parser):
    # 读取 XML 数据时根据列名解析日期列，并返回数据框 df_result
    df_result = read_xml(StringIO(xml_dates), parse_dates=["date"], parser=parser)
    # 使用迭代解析读取 XML 数据，同时指定按列名解析日期
    df_iter = read_xml_iterparse(
        xml_dates,
        parser=parser,
        parse_dates=["date"],
        iterparse={"row": ["shape", "degrees", "sides", "date"]},
    )

    # 期望的数据框 df_expected，包含预期结果的列与数据
    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": [360, 360, 180],
            "sides": [4.0, float("nan"), 3.0],
            "date": to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
        }
    )

    # 断言两个数据框是否相等，用于测试结果的一致性
    tm.assert_frame_equal(df_result, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


# 测试按列索引解析日期
def test_parse_dates_column_index(parser):
    # 读取 XML 数据时根据列索引解析日期列，并返回数据框 df_result
    df_result = read_xml(StringIO(xml_dates), parse_dates=[3], parser=parser)
    # 使用迭代解析读取 XML 数据，同时指定按列索引解析日期
    df_iter = read_xml_iterparse(
        xml_dates,
        parser=parser,
        parse_dates=[3],
        iterparse={"row": ["shape", "degrees", "sides", "date"]},
    )

    # 期望的数据框 df_expected，包含预期结果的列与数据
    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": [360, 360, 180],
            "sides": [4.0, float("nan"), 3.0],
            "date": to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
        }
    )

    # 断言两个数据框是否相等，用于测试结果的一致性
    tm.assert_frame_equal(df_result, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)
    # 调用 read_xml 函数，解析给定的 XML 字符串，将结果保存在 df_result 中
    df_result = read_xml(StringIO(xml_dates), parse_dates=True, parser=parser)
    
    # 调用 read_xml_iterparse 函数，以迭代解析方式读取 XML 数据，将结果保存在 df_iter 中
    # parser 参数指定解析器，parse_dates=True 表示解析日期，iterparse={"row": ["shape", "degrees", "sides", "date"]} 指定了迭代解析的配置
    df_iter = read_xml_iterparse(
        xml_dates,
        parser=parser,
        parse_dates=True,
        iterparse={"row": ["shape", "degrees", "sides", "date"]},
    )
    
    # 创建 DataFrame 对象 df_expected，包含预期的数据内容，列名和数据类型
    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": [360, 360, 180],
            "sides": [4.0, float("nan"), 3.0],
            "date": ["2020-01-01", "2021-01-01", "2022-01-01"],
        }
    )
    
    # 使用 pandas.testing 模块的 assert_frame_equal 函数比较 df_result 和 df_expected 是否相等
    tm.assert_frame_equal(df_result, df_expected)
    
    # 使用 pandas.testing 模块的 assert_frame_equal 函数比较 df_iter 和 df_expected 是否相等
    tm.assert_frame_equal(df_iter, df_expected)
# 定义一个测试函数，用于测试日期首次解析函数
def test_day_first_parse_dates(parser):
    # XML 数据字符串，包含多个数据行，每行有形状、角度、边数和日期等信息
    xml = """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row>
    <shape>square</shape>
    <degrees>00360</degrees>
    <sides>4.0</sides>
    <date>31/12/2020</date>
   </row>
  <row>
    <shape>circle</shape>
    <degrees>00360</degrees>
    <sides/>
    <date>31/12/2021</date>
  </row>
  <row>
    <shape>triangle</shape>
    <degrees>00180</degrees>
    <sides>3.0</sides>
    <date>31/12/2022</date>
  </row>
</data>"""

    # 期望的数据帧，包含形状、角度、边数和日期，日期已经转换为 datetime 格式
    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": [360, 360, 180],
            "sides": [4.0, float("nan"), 3.0],
            "date": to_datetime(["2020-12-31", "2021-12-31", "2022-12-31"]),
        }
    )

    # 使用上下文管理器确保在解析日期格式时产生警告
    with tm.assert_produces_warning(
        UserWarning, match="Parsing dates in %d/%m/%Y format"
    ):
        # 调用 read_xml 函数解析 XML 数据，将日期列作为日期解析
        df_result = read_xml(StringIO(xml), parse_dates=["date"], parser=parser)
        
        # 调用 read_xml_iterparse 函数以迭代方式解析 XML 数据，同样将日期列作为日期解析
        df_iter = read_xml_iterparse(
            xml,
            parse_dates=["date"],
            parser=parser,
            iterparse={"row": ["shape", "degrees", "sides", "date"]},
        )

        # 断言两个数据帧是否相等，即预期结果与实际结果比较
        tm.assert_frame_equal(df_result, df_expected)
        tm.assert_frame_equal(df_iter, df_expected)


# 定义一个测试函数，用于测试解析日期类型错误的情况
def test_wrong_parse_dates_type(xml_books, parser, iterparse):
    # 使用 pytest 检测是否会引发类型错误异常，匹配异常消息为 "Only booleans and lists are accepted"
    with pytest.raises(TypeError, match="Only booleans and lists are accepted"):
        # 调用 read_xml 函数并传入错误的日期解析类型，应当引发类型错误异常
        read_xml(xml_books, parse_dates={"date"}, parser=parser, iterparse=iterparse)
```