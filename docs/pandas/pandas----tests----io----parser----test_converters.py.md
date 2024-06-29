# `D:\src\scipysrc\pandas\pandas\tests\io\parser\test_converters.py`

```
"""
Tests column conversion functionality during parsing
for all of the parsers defined in parsers.py
"""

# 从io模块导入StringIO，用于创建内存中的文本流
from io import StringIO

# 导入parse函数，用于日期解析；导入numpy库并重命名为np；导入pytest库进行测试
from dateutil.parser import parse
import numpy as np
import pytest

# 导入pandas库，并从中导入DataFrame和Index类；导入pandas._testing模块并重命名为tm
import pandas as pd
from pandas import (
    DataFrame,
    Index,
)
import pandas._testing as tm


# 测试函数：验证converters选项必须是字典类型
def test_converters_type_must_be_dict(all_parsers):
    parser = all_parsers
    # 测试数据
    data = """index,A,B,C,D
foo,2,3,4,5
"""
    # 对于使用pyarrow引擎的情况，验证converters选项不支持，并抛出相应的错误信息
    if parser.engine == "pyarrow":
        msg = "The 'converters' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), converters=0)
        return
    # 对于其他情况，验证converters选项必须是TypeError类型，并抛出相应的错误信息
    with pytest.raises(TypeError, match="Type converters.+"):
        parser.read_csv(StringIO(data), converters=0)


# 参数化测试函数：验证converters功能是否正确
@pytest.mark.parametrize("column", [3, "D"])
@pytest.mark.parametrize(
    "converter",
    [parse, lambda x: int(x.split("/")[2])],  # 生成整数
)
def test_converters(all_parsers, column, converter):
    parser = all_parsers
    # 测试数据
    data = """A,B,C,D
a,1,2,01/01/2009
b,3,4,01/02/2009
c,4,5,01/03/2009
"""
    # 对于使用pyarrow引擎的情况，验证converters选项不支持，并抛出相应的错误信息
    if parser.engine == "pyarrow":
        msg = "The 'converters' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), converters={column: converter})
        return

    # 使用指定的converter转换数据，并验证转换后的结果与预期是否一致
    result = parser.read_csv(StringIO(data), converters={column: converter})

    expected = parser.read_csv(StringIO(data))
    expected["D"] = expected["D"].map(converter)

    tm.assert_frame_equal(result, expected)


# 测试函数：验证不进行隐式转换时的converters选项行为
def test_converters_no_implicit_conv(all_parsers):
    # see gh-2184
    parser = all_parsers
    # 测试数据
    data = """000102,1.2,A\n001245,2,B"""

    # 创建一个lambda函数作为converter，去除列0的空格
    converters = {0: lambda x: x.strip()}

    # 对于使用pyarrow引擎的情况，验证converters选项不支持，并抛出相应的错误信息
    if parser.engine == "pyarrow":
        msg = "The 'converters' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), header=None, converters=converters)
        return

    # 使用指定的converter转换数据，并验证转换后的结果与预期是否一致
    result = parser.read_csv(StringIO(data), header=None, converters=converters)

    # 预期结果：第0列不会被转换为数值类型，保持为对象类型
    expected = DataFrame([["000102", 1.2, "A"], ["001245", 2, "B"]])
    tm.assert_frame_equal(result, expected)


# 测试函数：验证特定的欧洲小数格式在converters选项下的处理
def test_converters_euro_decimal_format(all_parsers):
    # see gh-583
    converters = {}
    parser = all_parsers

    # 测试数据
    data = """Id;Number1;Number2;Text1;Text2;Number3
1;1521,1541;187101,9543;ABC;poi;4,7387
2;121,12;14897,76;DEF;uyt;0,3773
3;878,158;108013,434;GHI;rez;2,7356"""

    # 定义三个列的converter函数，用于将逗号替换为点号，并转换为浮点数
    converters["Number1"] = converters["Number2"] = converters["Number3"] = (
        lambda x: float(x.replace(",", "."))
    )

    # 对于使用pyarrow引擎的情况，验证converters选项不支持，并抛出相应的错误信息
    if parser.engine == "pyarrow":
        msg = "The 'converters' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), sep=";", converters=converters)
        return
    # 使用 parser 对象的 read_csv 方法读取 CSV 数据，并指定分隔符为 ';'，同时使用 converters 处理数据转换
    result = parser.read_csv(StringIO(data), sep=";", converters=converters)
    
    # 创建一个预期的 DataFrame 对象，包含三行数据，每行数据有六列
    expected = DataFrame(
        [
            [1, 1521.1541, 187101.9543, "ABC", "poi", 4.7387],
            [2, 121.12, 14897.76, "DEF", "uyt", 0.3773],
            [3, 878.158, 108013.434, "GHI", "rez", 2.7356],
        ],
        columns=["Id", "Number1", "Number2", "Text1", "Text2", "Number3"],  # 指定 DataFrame 的列名
    )
    
    # 使用 pandas.testing 模块中的 assert_frame_equal 函数比较 result 和 expected 两个 DataFrame 对象
    tm.assert_frame_equal(result, expected)
# 测试函数，用于测试处理带有 NaN 值的特殊情况
def test_converters_corner_with_nans(all_parsers):
    # 使用传入的所有解析器对象
    parser = all_parsers
    # 定义包含 NaN 值的测试数据
    data = """id,score,days
1,2,12
2,2-5,
3,,14+
4,6-12,2"""

    # 示例转换函数：处理 days 字段的转换
    def convert_days(x):
        # 去除字符串两端的空白字符
        x = x.strip()

        # 如果字符串为空，则返回 NaN
        if not x:
            return np.nan

        # 判断是否以 "+" 结尾
        is_plus = x.endswith("+")

        # 如果以 "+" 结尾，则将其转换为整数后加一
        if is_plus:
            x = int(x[:-1]) + 1
        else:
            x = int(x)

        return x

    # 另一个示例转换函数：处理 days 字段的转换，与 convert_days 函数类似
    def convert_days_sentinel(x):
        x = x.strip()

        if not x:
            return np.nan

        is_plus = x.endswith("+")

        if is_plus:
            x = int(x[:-1]) + 1
        else:
            x = int(x)

        return x

    # 示例转换函数：处理 score 字段的转换
    def convert_score(x):
        x = x.strip()

        if not x:
            return np.nan

        # 如果字符串中包含 "-"，则计算其平均值
        if x.find("-") > 0:
            val_min, val_max = map(int, x.split("-"))
            val = 0.5 * (val_min + val_max)
        else:
            val = float(x)

        return val

    # 存储测试结果的列表
    results = []

    # 遍历两个处理 days 的转换函数
    for day_converter in [convert_days, convert_days_sentinel]:
        # 如果解析器使用 pyarrow 引擎，则抛出 ValueError 异常
        if parser.engine == "pyarrow":
            msg = "The 'converters' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(
                    StringIO(data),
                    converters={"score": convert_score, "days": day_converter},
                    na_values=["", None],
                )
            continue

        # 使用解析器读取 CSV 数据，应用指定的转换函数
        result = parser.read_csv(
            StringIO(data),
            converters={"score": convert_score, "days": day_converter},
            na_values=["", None],
        )
        # 断言结果中的第二行 days 字段为 NaN
        assert pd.isna(result["days"][1])
        # 将结果添加到结果列表中
        results.append(result)

    # 如果解析器不是 pyarrow 引擎，则比较两个结果是否相等
    if parser.engine != "pyarrow":
        tm.assert_frame_equal(results[0], results[1])


# 使用参数化测试来验证转换器在处理索引列时的行为
@pytest.mark.parametrize("conv_f", [lambda x: x, str])
def test_converter_index_col_bug(all_parsers, conv_f):
    # 参考 GitHub 问题编号 1835 和 40589
    parser = all_parsers
    # 定义包含分号分隔符的测试数据
    data = "A;B\n1;2\n3;4"

    # 如果解析器使用 pyarrow 引擎，则抛出 ValueError 异常
    if parser.engine == "pyarrow":
        msg = "The 'converters' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data), sep=";", index_col="A", converters={"A": conv_f}
            )
        return

    # 使用解析器读取 CSV 数据，指定分隔符和转换函数
    rs = parser.read_csv(
        StringIO(data), sep=";", index_col="A", converters={"A": conv_f}
    )

    # 期望的结果 DataFrame
    xp = DataFrame({"B": [2, 4]}, index=Index(["1", "3"], name="A", dtype="object"))
    # 比较实际结果和期望结果是否相等
    tm.assert_frame_equal(rs, xp)


# 测试转换器处理对象类型列的情况
def test_converter_identity_object(all_parsers):
    # 参考 GitHub 问题编号 40589
    parser = all_parsers
    # 定义包含 A 和 B 列的测试数据
    data = "A,B\n1,2\n3,4"

    # 如果解析器使用 pyarrow 引擎，则抛出 ValueError 异常
    if parser.engine == "pyarrow":
        msg = "The 'converters' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), converters={"A": lambda x: x})
        return

    # 使用解析器读取 CSV 数据，指定转换函数用于 A 列
    rs = parser.read_csv(StringIO(data), converters={"A": lambda x: x})

    # 期望的结果 DataFrame
    xp = DataFrame({"A": ["1", "3"], "B": [2, 4]})
    # 使用测试工具包中的 assert_frame_equal 函数比较 rs 和 xp 两个数据框架是否相等
    tm.assert_frame_equal(rs, xp)
`
# 定义测试函数，接受所有解析器参数
def test_converter_multi_index(all_parsers):
    # GH 42446，指定当前使用的解析器
    parser = all_parsers
    # 定义测试数据字符串
    data = "A,B,B\nX,Y,Z\n1,2,3"

    # 检查解析器引擎是否为 'pyarrow'
    if parser.engine == "pyarrow":
        # 定义错误信息，'pyarrow' 引擎不支持 'converters' 选项
        msg = "The 'converters' option is not supported with the 'pyarrow' engine"
        # 使用 pytest 的断言机制，期望抛出 ValueError 异常，并匹配错误信息
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data),
                header=list(range(2)),
                converters={
                    ("A", "X"): np.int32,   # 将 ('A', 'X') 列转换为 np.int32 类型
                    ("B", "Y"): np.int32,   # 将 ('B', 'Y') 列转换为 np.int32 类型
                    ("B", "Z"): np.float32,  # 将 ('B', 'Z') 列转换为 np.float32 类型
                },
            )
        # 如果解析器是 'pyarrow'，则直接返回
        return

    # 否则，使用指定的解析器读取 CSV 数据，应用转换器
    result = parser.read_csv(
        StringIO(data),
        header=list(range(2)),
        converters={
            ("A", "X"): np.int32,   # 将 ('A', 'X') 列转换为 np.int32 类型
            ("B", "Y"): np.int32,   # 将 ('B', 'Y') 列转换为 np.int32 类型
            ("B", "Z"): np.float32,  # 将 ('B', 'Z') 列转换为 np.float32 类型
        },
    )

    # 定义期望的结果数据框架
    expected = DataFrame(
        {
            ("A", "X"): np.int32([1]),  # ('A', 'X') 列的期望数据
            ("B", "Y"): np.int32([2]),  # ('B', 'Y') 列的期望数据
            ("B", "Z"): np.float32([3]), # ('B', 'Z') 列的期望数据
        }
    )

    # 使用测试工具，断言读取结果与期望结果相等
    tm.assert_frame_equal(result, expected)
```