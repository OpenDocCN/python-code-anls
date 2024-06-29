# `D:\src\scipysrc\pandas\pandas\tests\io\parser\test_dialect.py`

```
"""
Tests that dialects are properly handled during parsing
for all of the parsers defined in parsers.py
"""

import csv  # 导入csv模块，用于处理CSV文件
from io import StringIO  # 导入StringIO类，用于在内存中读写字符串

import pytest  # 导入pytest模块，用于编写和运行测试用例

from pandas.errors import ParserWarning  # 导入ParserWarning，处理解析器警告

from pandas import DataFrame  # 导入DataFrame类，用于操作和处理数据框
import pandas._testing as tm  # 导入测试相关模块

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)  # 使用pytest的mark.filterwarnings忽略特定警告信息


@pytest.fixture  # 定义测试用例的fixture
def custom_dialect():
    dialect_name = "weird"  # 自定义方言名称为"weird"
    dialect_kwargs = {  # 定义方言的各种参数
        "doublequote": False,
        "escapechar": "~",
        "delimiter": ":",
        "skipinitialspace": False,
        "quotechar": "`",
        "quoting": 3,
    }
    return dialect_name, dialect_kwargs  # 返回方言名称和参数字典作为fixture


def test_dialect(all_parsers):  # 定义测试方言的函数
    parser = all_parsers  # 使用all_parsers fixture获取解析器
    data = """\
label1,label2,label3
index1,"a,c,e
index2,b,d,f
"""  # 测试数据，包含逗号和引号混合的情况

    dia = csv.excel()  # 创建一个默认的CSV方言对象
    dia.quoting = csv.QUOTE_NONE  # 设置方言对象的引用样式为不引用

    if parser.engine == "pyarrow":  # 如果使用的是pyarrow引擎
        msg = "The 'dialect' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):  # 断言引发特定的 ValueError 异常
            parser.read_csv(StringIO(data), dialect=dia)  # 尝试使用指定方言解析CSV数据
        return  # 结束函数

    df = parser.read_csv(StringIO(data), dialect=dia)  # 使用指定方言解析CSV数据，并返回数据帧

    data = """\
label1,label2,label3
index1,a,c,e
index2,b,d,f
"""
    exp = parser.read_csv(StringIO(data))  # 使用默认方言解析期望的CSV数据
    exp.replace("a", '"a', inplace=True)  # 替换数据帧中的特定值
    tm.assert_frame_equal(df, exp)  # 断言两个数据帧是否相等


def test_dialect_str(all_parsers):  # 定义测试自定义方言字符串的函数
    dialect_name = "mydialect"  # 自定义方言名称为"mydialect"
    parser = all_parsers  # 使用all_parsers fixture获取解析器
    data = """\
fruit:vegetable
apple:broccoli
pear:tomato
"""  # 测试数据，使用自定义分隔符":"分隔

    exp = DataFrame({"fruit": ["apple", "pear"], "vegetable": ["broccoli", "tomato"]})  # 期望的数据帧

    with tm.with_csv_dialect(dialect_name, delimiter=":"):  # 使用指定的CSV方言进行上下文管理
        if parser.engine == "pyarrow":  # 如果使用的是pyarrow引擎
            msg = "The 'dialect' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):  # 断言引发特定的 ValueError 异常
                parser.read_csv(StringIO(data), dialect=dialect_name)  # 尝试使用自定义方言解析CSV数据
            return  # 结束函数

        df = parser.read_csv(StringIO(data), dialect=dialect_name)  # 使用自定义方言解析CSV数据，并返回数据帧
        tm.assert_frame_equal(df, exp)  # 断言两个数据帧是否相等


def test_invalid_dialect(all_parsers):  # 定义测试无效方言的函数
    class InvalidDialect:  # 定义一个无效的方言类
        pass

    data = "a\n1"  # 测试数据
    parser = all_parsers  # 使用all_parsers fixture获取解析器
    msg = "Invalid dialect"  # 错误消息

    with pytest.raises(ValueError, match=msg):  # 断言引发特定的 ValueError 异常
        parser.read_csv(StringIO(data), dialect=InvalidDialect)  # 尝试使用无效方言解析CSV数据


@pytest.mark.parametrize(  # 参数化测试用例
    "arg",
    [None, "doublequote", "escapechar", "skipinitialspace", "quotechar", "quoting"],
)
@pytest.mark.parametrize("value", ["dialect", "default", "other"])
def test_dialect_conflict_except_delimiter(all_parsers, custom_dialect, arg, value):
    # see gh-23761.
    dialect_name, dialect_kwargs = custom_dialect  # 获取自定义方言名称和参数
    parser = all_parsers  # 使用all_parsers fixture获取解析器

    expected = DataFrame({"a": [1], "b": [2]})  # 期望的数据帧
    data = "a:b\n1:2"  # 测试数据，使用自定义分隔符":"

    warning_klass = None  # 警告类设为None
    kwds = {}  # 关键字参数设为空字典

    # arg=None tests when we pass in the dialect without any other arguments.
    # 如果参数arg不为None，则执行以下逻辑
    if arg is not None:
        # 如果value等于"dialect"，表示没有冲突，不发出警告
        if value == "dialect":  # No conflict --> no warning.
            # 将kwds字典中的arg设为dialect_kwargs字典中arg对应的值
            kwds[arg] = dialect_kwargs[arg]
        # 如果value等于"default"，表示使用默认值，不发出警告
        elif value == "default":  # Default --> no warning.
            # 从pandas.io.parsers.base_parser模块导入parser_defaults，并将其arg对应的值赋给kwds字典中的arg
            from pandas.io.parsers.base_parser import parser_defaults
            kwds[arg] = parser_defaults[arg]
        else:
            # 如果value不为"default"且与dialect有冲突，发出警告
            # 设置警告类为ParserWarning
            warning_klass = ParserWarning
            # 将kwds字典中的arg设为"blah"
            kwds[arg] = "blah"

    # 使用指定的CSV方言名和dialect_kwargs设置tm.with_csv_dialect上下文管理器
    with tm.with_csv_dialect(dialect_name, **dialect_kwargs):
        # 如果parser的引擎为"pyarrow"
        if parser.engine == "pyarrow":
            # 提示信息
            msg = "The 'dialect' option is not supported with the 'pyarrow' engine"
            # 使用pytest.raises检测是否抛出指定的ValueError异常，并匹配提示信息
            with pytest.raises(ValueError, match=msg):
                # 调用parser.read_csv_check_warnings函数，传入相关参数
                parser.read_csv_check_warnings(
                    # 不会有警告因为我们会抛出异常
                    None,
                    "Conflicting values for",
                    StringIO(data),
                    dialect=dialect_name,
                    **kwds,
                )
            # 函数返回，结束执行
            return

        # 使用parser.read_csv_check_warnings函数读取CSV并检查警告
        result = parser.read_csv_check_warnings(
            warning_klass,
            "Conflicting values for",
            StringIO(data),
            dialect=dialect_name,
            **kwds,
        )
        # 使用tm.assert_frame_equal检查结果DataFrame是否与预期DataFrame相等
        tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize(
    "kwargs,warning_klass",
    [
        ({"sep": ","}, None),  # 当 sep 参数为默认值时，设置 sep_override=True
        ({"sep": "."}, ParserWarning),  # 当 sep 参数不是默认值时，设置 sep_override=False
        ({"delimiter": ":"}, None),  # 没有冲突的情况
        ({"delimiter": None}, None),  # 使用默认参数时，设置 sep_override=True
        ({"delimiter": ","}, ParserWarning),  # 存在冲突的情况
        ({"delimiter": "."}, ParserWarning),  # 存在冲突的情况
    ],
    ids=[
        "sep-override-true",  # 测试 sep 参数为默认值时的情况
        "sep-override-false",  # 测试 sep 参数不为默认值时的情况
        "delimiter-no-conflict",  # 测试没有冲突的 delimiter 参数的情况
        "delimiter-default-arg",  # 测试使用默认参数时的 delimiter 参数的情况
        "delimiter-conflict",  # 测试存在冲突的 delimiter 参数的情况
        "delimiter-conflict2",  # 测试存在冲突的 delimiter 参数的情况（第二个冲突案例）
    ],
)
def test_dialect_conflict_delimiter(all_parsers, custom_dialect, kwargs, warning_klass):
    # 查看 issue gh-23761.
    dialect_name, dialect_kwargs = custom_dialect
    parser = all_parsers

    expected = DataFrame({"a": [1], "b": [2]})
    data = "a:b\n1:2"

    with tm.with_csv_dialect(dialect_name, **dialect_kwargs):
        if parser.engine == "pyarrow":
            msg = "The 'dialect' option is not supported with the 'pyarrow' engine"
            # 检查在 'pyarrow' 引擎下是否会引发 ValueError 异常，匹配错误信息 msg
            with pytest.raises(ValueError, match=msg):
                parser.read_csv_check_warnings(
                    None,  # 无警告因为我们会引发异常
                    "Conflicting values for 'delimiter'",  # 异常消息中包含的文本
                    StringIO(data),
                    dialect=dialect_name,
                    **kwargs,
                )
            return
        # 执行读取 CSV 并检查警告
        result = parser.read_csv_check_warnings(
            warning_klass,
            "Conflicting values for 'delimiter'",  # 异常消息中包含的文本
            StringIO(data),
            dialect=dialect_name,
            **kwargs,
        )
        # 断言结果 DataFrame 与预期相等
        tm.assert_frame_equal(result, expected)
```