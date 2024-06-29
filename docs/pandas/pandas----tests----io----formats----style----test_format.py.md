# `D:\src\scipysrc\pandas\pandas\tests\io\formats\style\test_format.py`

```
# 导入必要的库：numpy 和 pytest
import numpy as np
import pytest

# 从 pandas 中导入需要的对象和函数
from pandas import (
    NA,                 # 缺失值常量
    DataFrame,          # 数据帧类
    IndexSlice,         # 切片索引类
    MultiIndex,         # 多重索引类
    NaT,                # 不可用时间常量
    Timestamp,          # 时间戳类
    option_context,     # 上下文选项函数
)

# 导入 jinja2，如果导入失败则跳过测试
pytest.importorskip("jinja2")
from pandas.io.formats.style import Styler         # 样式化器类
from pandas.io.formats.style_render import _str_escape  # 字符串转义函数


@pytest.fixture
def df():
    # 创建一个简单的数据帧对象
    return DataFrame(
        data=[[0, -0.609], [1, -1.228]],
        columns=["A", "B"],
        index=["x", "y"],
    )


@pytest.fixture
def styler(df):
    # 使用 Styler 类创建数据帧的样式化对象
    return Styler(df, uuid_len=0)


@pytest.fixture
def df_multi():
    # 创建一个包含多重索引的数据帧对象
    return (
        DataFrame(
            data=np.arange(16).reshape(4, 4),
            columns=MultiIndex.from_product([["A", "B"], ["a", "b"]]),
            index=MultiIndex.from_product([["X", "Y"], ["x", "y"]]),
        )
        .rename_axis(["0_0", "0_1"], axis=0)
        .rename_axis(["1_0", "1_1"], axis=1)
    )


@pytest.fixture
def styler_multi(df_multi):
    # 使用 Styler 类创建多重索引数据帧的样式化对象
    return Styler(df_multi, uuid_len=0)


def test_display_format(styler):
    # 测试样式化对象的显示格式
    ctx = styler.format("{:0.1f}")._translate(True, True)
    # 断言每个单元格的 display_value 属性包含在结果中
    assert all(["display_value" in c for c in row] for row in ctx["body"])
    # 断言每个数值的 display_value 属性长度不超过 3
    assert all([len(c["display_value"]) <= 3 for c in row[1:]] for row in ctx["body"])
    # 断言第一行第二列的 display_value 属性去除负号后长度不超过 3
    assert len(ctx["body"][0][1]["display_value"].lstrip("-")) <= 3


@pytest.mark.parametrize("index", [True, False])
@pytest.mark.parametrize("columns", [True, False])
def test_display_format_index(styler, index, columns):
    # 测试带索引的显示格式
    exp_index = ["x", "y"]
    if index:
        styler.format_index(lambda v: v.upper(), axis=0)  # 测试可调用函数
        exp_index = ["X", "Y"]

    exp_columns = ["A", "B"]
    if columns:
        styler.format_index("*{}*", axis=1)  # 测试字符串格式
        exp_columns = ["*A*", "*B*"]

    ctx = styler._translate(True, True)

    # 验证索引的显示值
    for r, row in enumerate(ctx["body"]):
        assert row[0]["display_value"] == exp_index[r]

    # 验证列的显示值
    for c, col in enumerate(ctx["head"][1:]):
        assert col["display_value"] == exp_columns[c]


def test_format_dict(styler):
    # 测试样式化对象应用字典格式化
    ctx = styler.format({"A": "{:0.1f}", "B": "{0:.2%}"})._translate(True, True)
    assert ctx["body"][0][1]["display_value"] == "0.0"
    assert ctx["body"][0][2]["display_value"] == "-60.90%"


def test_format_index_dict(styler):
    # 测试样式化对象应用索引字典格式化
    ctx = styler.format_index({0: lambda v: v.upper()})._translate(True, True)
    for i, val in enumerate(["X", "Y"]):
        assert ctx["body"][i][0]["display_value"] == val


def test_format_string(styler):
    # 测试样式化对象应用字符串格式化
    ctx = styler.format("{:.2f}")._translate(True, True)
    assert ctx["body"][0][1]["display_value"] == "0.00"
    assert ctx["body"][0][2]["display_value"] == "-0.61"
    assert ctx["body"][1][1]["display_value"] == "1.00"
    assert ctx["body"][1][2]["display_value"] == "-1.23"


def test_format_callable(styler):
    # 测试样式化对象应用可调用格式化
    ctx = styler.format(lambda v: "neg" if v < 0 else "pos")._translate(True, True)
    assert ctx["body"][0][1]["display_value"] == "pos"
    assert ctx["body"][0][2]["display_value"] == "neg"
    assert ctx["body"][1][1]["display_value"] == "pos"
    # 断言：确保 ctx 字典中的 "body" 键存在，并且它是一个列表，列表的第二个元素是一个字典，
    # 这个字典中的第三个元素的 "display_value" 键的值等于 "neg"
    assert ctx["body"][1][2]["display_value"] == "neg"
def test_format_with_na_rep():
    # 创建一个包含空值和数字的DataFrame，列名为"A"和"B"
    df = DataFrame([[None, None], [1.1, 1.2]], columns=["A", "B"])

    # 对DataFrame应用样式，并将缺失值表示为"-"
    ctx = df.style.format(None, na_rep="-")._translate(True, True)
    # 断言第一行第二列的显示值为"-"
    assert ctx["body"][0][1]["display_value"] == "-"
    # 断言第一行第三列的显示值为"-"
    assert ctx["body"][0][2]["display_value"] == "-"

    # 对DataFrame应用格式化为百分比的样式，并将缺失值表示为"-"
    ctx = df.style.format("{:.2%}", na_rep="-")._translate(True, True)
    # 断言第一行第二列的显示值为"-"
    assert ctx["body"][0][1]["display_value"] == "-"
    # 断言第一行第三列的显示值为"-"
    assert ctx["body"][0][2]["display_value"] == "-"
    # 断言第二行第二列的显示值为"110.00%"
    assert ctx["body"][1][1]["display_value"] == "110.00%"
    # 断言第二行第三列的显示值为"120.00%"
    assert ctx["body"][1][2]["display_value"] == "120.00%"

    # 对DataFrame应用格式化为百分比的样式，仅对"B"列生效，并将缺失值表示为"-"
    ctx = df.style.format("{:.2%}", na_rep="-", subset=["B"])._translate(True, True)
    # 断言第一行第三列的显示值为"-"
    assert ctx["body"][0][2]["display_value"] == "-"
    # 断言第二行第三列的显示值为"120.00%"
    assert ctx["body"][1][2]["display_value"] == "120.00%"


def test_format_index_with_na_rep():
    # 创建一个包含非数值缺失值的DataFrame，列名包括"A"、None、np.nan、NaT和NA
    df = DataFrame([[1, 2, 3, 4, 5]], columns=["A", None, np.nan, NaT, NA])
    # 对DataFrame的索引应用样式，并将缺失值表示为"--"
    ctx = df.style.format_index(None, na_rep="--", axis=1)._translate(True, True)
    # 断言头部的第一行第二列的显示值为"A"
    assert ctx["head"][0][1]["display_value"] == "A"
    # 断言头部的第一行的第三至第五列的显示值为"--"
    for i in [2, 3, 4, 5]:
        assert ctx["head"][0][i]["display_value"] == "--"


def test_format_non_numeric_na():
    # 创建一个包含对象和日期时间列的DataFrame，其中包括空值、NaT和具体日期时间
    df = DataFrame(
        {
            "object": [None, np.nan, "foo"],
            "datetime": [None, NaT, Timestamp("20120101")],
        }
    )
    # 对DataFrame应用样式，并将缺失值表示为"-"
    ctx = df.style.format(None, na_rep="-")._translate(True, True)
    # 断言第一行第二列的显示值为"-"
    assert ctx["body"][0][1]["display_value"] == "-"
    # 断言第一行第三列的显示值为"-"
    assert ctx["body"][0][2]["display_value"] == "-"
    # 断言第二行第二列的显示值为"-"
    assert ctx["body"][1][1]["display_value"] == "-"
    # 断言第二行第三列的显示值为"-"
    assert ctx["body"][1][2]["display_value"] == "-"


@pytest.mark.parametrize(
    "func, attr, kwargs",
    [
        ("format", "_display_funcs", {}),
        ("format_index", "_display_funcs_index", {"axis": 0}),
        ("format_index", "_display_funcs_columns", {"axis": 1}),
    ],
)
def test_format_clear(styler, func, attr, kwargs):
    # 断言默认情况下样式对象中不包含指定的格式化函数
    assert (0, 0) not in getattr(styler, attr)  # using default
    # 调用指定的格式化函数，并断言格式化函数被添加到样式对象中
    getattr(styler, func)("{:.2f}", **kwargs)
    assert (0, 0) in getattr(styler, attr)  # formatter is specified
    # 再次调用格式化函数，断言格式化函数已经从样式对象中移除
    getattr(styler, func)(**kwargs)
    assert (0, 0) not in getattr(styler, attr)  # formatter cleared to default


@pytest.mark.parametrize(
    "escape, exp",
    [
        ("html", "&lt;&gt;&amp;&#34;%$#_{}~^\\~ ^ \\ "),
        (
            "latex",
            '<>\\&"\\%\\$\\#\\_\\{\\}\\textasciitilde \\textasciicircum '
            "\\textbackslash \\textasciitilde \\space \\textasciicircum \\space "
            "\\textbackslash \\space ",
        ),
    ],
)
def test_format_escape_html(escape, exp):
    # 定义包含特殊字符的字符串，并创建包含该字符串的DataFrame
    chars = '<>&"%$#_{}~^\\~ ^ \\ '
    df = DataFrame([[chars]])

    # 对DataFrame应用样式，并指定格式化字符串为"&{0}&"，不进行转义
    s = Styler(df, uuid_len=0).format("&{0}&", escape=None)
    expected = f'<td id="T__row0_col0" class="data row0 col0" >&{chars}&</td>'
    # 断言生成的HTML包含预期的不转义字符
    assert expected in s.to_html()

    # 对DataFrame应用样式，并指定格式化字符串为"&{0}&"，进行指定的转义
    s = Styler(df, uuid_len=0).format("&{0}&", escape=escape)
    # 生成期望的 HTML 表格单元格内容，包含特定的 id 和 class
    expected = f'<td id="T__row0_col0" class="data row0 col0" >&{exp}&</td>'
    # 断言期望的 HTML 单元格内容是否在生成的表格字符串中
    assert expected in s.to_html()

    # 测试 format_index() 方法
    # 创建一个 DataFrame，并使用 Styler 对象对其进行格式化
    styler = Styler(DataFrame(columns=[chars]), uuid_len=0)
    # 使用指定的格式化字符串来格式化索引，不进行字符转义
    styler.format_index("&{0}&", escape=None, axis=1)
    # 断言格式化后的表头索引值是否与期望的字符匹配
    assert styler._translate(True, True)["head"][0][1]["display_value"] == f"&{chars}&"
    # 再次使用 format_index() 方法，这次进行字符转义
    styler.format_index("&{0}&", escape=escape, axis=1)
    # 断言格式化后的表头索引值是否与期望的表达式字符匹配
    assert styler._translate(True, True)["head"][0][1]["display_value"] == f"&{exp}&"
@pytest.mark.parametrize(
    "chars, expected",
    [  # 参数化测试，提供不同的输入字符和预期输出
        (
            r"$ \$&%#_{}~^\ $ &%#_{}~^\ $",  # 输入字符串示例
            "".join(
                [
                    r"$ \$&%#_{}~^\ $ ",  # 预期输出的一部分
                    r"\&\%\#\_\{\}\textasciitilde \textasciicircum ",  # 预期输出的一部分
                    r"\textbackslash \space \$",  # 预期输出的一部分
                ]
            ),
        ),
        (
            r"\( &%#_{}~^\ \) &%#_{}~^\ \(",  # 输入字符串示例
            "".join(
                [
                    r"\( &%#_{}~^\ \) ",  # 预期输出的一部分
                    r"\&\%\#\_\{\}\textasciitilde \textasciicircum ",  # 预期输出的一部分
                    r"\textbackslash \space \textbackslash (",  # 预期输出的一部分
                ]
            ),
        ),
        (
            r"$\&%#_{}^\$",  # 输入字符串示例
            r"\$\textbackslash \&\%\#\_\{\}\textasciicircum \textbackslash \$",  # 预期输出
        ),
        (
            r"$ \frac{1}{2} $ \( \frac{1}{2} \)",  # 输入字符串示例
            "".join(
                [
                    r"$ \frac{1}{2} $",  # 预期输出的一部分
                    r" \textbackslash ( \textbackslash frac\{1\}\{2\} \textbackslash )",  # 预期输出的一部分
                ]
            ),
        ),
    ],
)
def test_format_escape_latex_math(chars, expected):
    # GH 51903
    # latex-math escape works for each DataFrame cell separately. If we have
    # a combination of dollar signs and brackets, the dollar sign would apply.
    df = DataFrame([[chars]])  # 创建一个包含输入字符的DataFrame
    s = df.style.format("{0}", escape="latex-math")  # 应用样式并转义为 LaTeX 数学符号
    assert s._translate(True, True)["body"][0][1]["display_value"] == expected  # 断言样式化后的值与预期输出相符


def test_format_escape_na_rep():
    # tests the na_rep is not escaped
    df = DataFrame([['<>&"', None]])  # 创建一个包含特殊字符和空值的DataFrame
    s = Styler(df, uuid_len=0).format("X&{0}>X", escape="html", na_rep="&")  # 应用样式并使用 HTML 转义，设置 na_rep
    ex = '<td id="T__row0_col0" class="data row0 col0" >X&&lt;&gt;&amp;&#34;>X</td>'  # 预期输出字符串
    expected2 = '<td id="T__row0_col1" class="data row0 col1" >&</td>'  # 预期输出字符串
    assert ex in s.to_html()  # 断言预期字符串在转换为 HTML 后的输出中出现
    assert expected2 in s.to_html()  # 断言预期字符串在转换为 HTML 后的输出中出现

    # also test for format_index()
    df = DataFrame(columns=['<>&"', None])  # 创建一个带有特殊字符列名和空值的DataFrame
    styler = Styler(df, uuid_len=0)
    styler.format_index("X&{0}>X", escape="html", na_rep="&", axis=1)  # 应用索引样式并使用 HTML 转义，设置 na_rep
    ctx = styler._translate(True, True)  # 翻译 Styler 对象以获取上下文信息
    assert ctx["head"][0][1]["display_value"] == "X&&lt;&gt;&amp;&#34;>X"  # 断言头部样式化后的值与预期输出相符
    assert ctx["head"][0][2]["display_value"] == "&"  # 断言头部样式化后的值与预期输出相符


def test_format_escape_floats(styler):
    # test given formatter for number format is not impacted by escape
    s = styler.format("{:.1f}", escape="html")  # 应用样式并使用 HTML 转义，设置数字格式
    for expected in [">0.0<", ">1.0<", ">-1.2<", ">-0.6<"]:  # 预期输出字符串列表
        assert expected in s.to_html()  # 断言预期字符串在转换为 HTML 后的输出中出现
    # tests precision of floats is not impacted by escape
    s = styler.format(precision=1, escape="html")  # 应用样式并使用 HTML 转义，设置数字精度
    for expected in [">0<", ">1<", ">-1.2<", ">-0.6<"]:  # 预期输出字符串列表
        assert expected in s.to_html()  # 断言预期字符串在转换为 HTML 后的输出中出现


@pytest.mark.parametrize("formatter", [5, True, [2.0]])
@pytest.mark.parametrize("func", ["format", "format_index"])
def test_format_raises(styler, formatter, func):
    with pytest.raises(TypeError, match="expected str or callable"):  # 断言异常类型为 TypeError
        getattr(styler, func)(formatter)  # 调用 Styler 对象的指定方法并传入非预期参数类型

@pytest.mark.parametrize(
    # 定义一个包含测试数据的列表，每个元组包含一个精度和一个预期结果列表
    "precision, expected",
    [
        # 精度为1时的测试数据，预期结果是一个包含四个字符串的列表
        (1, ["1.0", "2.0", "3.2", "4.6"]),
        # 精度为2时的测试数据，预期结果是一个包含四个字符串的列表
        (2, ["1.00", "2.01", "3.21", "4.57"]),
        # 精度为3时的测试数据，预期结果是一个包含四个字符串的列表
        (3, ["1.000", "2.009", "3.212", "4.566"]),
    ],
# 定义一个测试函数，用于测试在指定精度下格式化样式化数据框的功能
def test_format_with_precision(precision, expected):
    # 创建一个数据框，包含一个浮点数列表，并指定列名为浮点数
    df = DataFrame([[1.0, 2.0090, 3.2121, 4.566]], columns=[1.0, 2.0090, 3.2121, 4.566])
    # 创建数据框的样式化对象
    styler = Styler(df)
    # 对样式化对象应用指定精度的格式化
    styler.format(precision=precision)
    # 对样式化对象的索引应用指定精度的格式化
    styler.format_index(precision=precision, axis=1)

    # 将样式化对象转换为上下文对象
    ctx = styler._translate(True, True)
    # 遍历期望结果列表，并逐一检查样式化后的显示值是否与期望一致
    for col, exp in enumerate(expected):
        assert ctx["body"][0][col + 1]["display_value"] == exp  # format test
        assert ctx["head"][0][col + 1]["display_value"] == exp  # format_index test


# 使用 pytest 提供的参数化功能进行多组测试
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize(
    "level, expected",
    [
        (0, ["X", "X", "_", "_"]),  # level int
        ("zero", ["X", "X", "_", "_"]),  # level name
        (1, ["_", "_", "X", "X"]),  # other level int
        ("one", ["_", "_", "X", "X"]),  # other level name
        ([0, 1], ["X", "X", "X", "X"]),  # both levels
        ([0, "zero"], ["X", "X", "_", "_"]),  # level int and name simultaneous
        ([0, "one"], ["X", "X", "X", "X"]),  # both levels as int and name
        (["one", "zero"], ["X", "X", "X", "X"]),  # both level names, reversed
    ],
)
# 定义测试函数，用于测试在不同轴上的索引级别格式化功能
def test_format_index_level(axis, level, expected):
    # 创建多重索引
    midx = MultiIndex.from_arrays([["_", "_"], ["_", "_"]], names=["zero", "one"])
    # 创建数据框
    df = DataFrame([[1, 2], [3, 4]])
    # 根据轴向决定是对索引还是对列进行多重索引的应用
    if axis == 0:
        df.index = midx
    else:
        df.columns = midx

    # 创建数据框的样式化对象，并应用索引级别的格式化
    styler = df.style.format_index(lambda v: "X", level=level, axis=axis)
    # 将样式化对象转换为上下文对象
    ctx = styler._translate(True, True)

    # 根据轴向决定比较索引或者列的显示值
    if axis == 0:  # compare index
        result = [ctx["body"][s][0]["display_value"] for s in range(2)]
        result += [ctx["body"][s][1]["display_value"] for s in range(2)]
    else:  # compare columns
        result = [ctx["head"][0][s + 1]["display_value"] for s in range(2)]
        result += [ctx["head"][1][s + 1]["display_value"] for s in range(2)]

    # 断言比较结果与期望结果是否一致
    assert expected == result


# 定义测试函数，用于测试在指定子集上的格式化功能
def test_format_subset():
    # 创建数据框
    df = DataFrame([[0.1234, 0.1234], [1.1234, 1.1234]], columns=["a", "b"])
    # 将数据框的样式化对象转换为上下文对象，应用不同的格式化方式和子集
    ctx = df.style.format(
        {"a": "{:0.1f}", "b": "{0:.2%}"}, subset=IndexSlice[0, :]
    )._translate(True, True)
    expected = "0.1"
    raw_11 = "1.123400"
    # 检查格式化后的显示值是否符合预期
    assert ctx["body"][0][1]["display_value"] == expected
    assert ctx["body"][1][1]["display_value"] == raw_11
    assert ctx["body"][0][2]["display_value"] == "12.34%"

    # 将数据框的样式化对象转换为上下文对象，只应用一种格式化方式和子集
    ctx = df.style.format("{:0.1f}", subset=IndexSlice[0, :])._translate(True, True)
    # 检查格式化后的显示值是否符合预期
    assert ctx["body"][0][1]["display_value"] == expected
    assert ctx["body"][1][1]["display_value"] == raw_11

    # 将数据框的样式化对象转换为上下文对象，只应用一种格式化方式和子集
    ctx = df.style.format("{:0.1f}", subset=IndexSlice["a"])._translate(True, True)
    # 检查格式化后的显示值是否符合预期
    assert ctx["body"][0][1]["display_value"] == expected
    assert ctx["body"][0][2]["display_value"] == "0.123400"

    # 将数据框的样式化对象转换为上下文对象，只应用一种格式化方式和子集
    ctx = df.style.format("{:0.1f}", subset=IndexSlice[0, "a"])._translate(True, True)
    # 检查格式化后的显示值是否符合预期
    assert ctx["body"][0][1]["display_value"] == expected
    assert ctx["body"][1][1]["display_value"] == raw_11
    # 使用 Pandas DataFrame 的样式对象 df.style 对数据进行格式化，保留一位小数
    ctx = df.style.format("{:0.1f}", subset=IndexSlice[[0, 1], ["a"]])._translate(
        True, True
    )
    # 断言：验证第一行第二列的显示值等于期望值 expected
    assert ctx["body"][0][1]["display_value"] == expected
    # 断言：验证第二行第二列的显示值等于 "1.1"
    assert ctx["body"][1][1]["display_value"] == "1.1"
    # 断言：验证第一行第三列的显示值等于 "0.123400"
    assert ctx["body"][0][2]["display_value"] == "0.123400"
    # 断言：验证第二行第三列的显示值等于 raw_11 变量的值
    assert ctx["body"][1][2]["display_value"] == raw_11
@pytest.mark.parametrize("formatter", [None, "{:,.1f}"])
@pytest.mark.parametrize("decimal", [".", "*"])
@pytest.mark.parametrize("precision", [None, 2])
@pytest.mark.parametrize("func, col", [("format", 1), ("format_index", 0)])
def test_format_thousands(formatter, decimal, precision, func, col):
    # 创建一个 DataFrame 对象，包含一个浮点数值，使用样式对象进行样式化
    styler = DataFrame([[1000000.123456789]], index=[1000000.123456789]).style
    # 调用指定的格式化函数（如 format 或 format_index），设置千位分隔符、格式化字符串等参数，并翻译样式
    result = getattr(styler, func)(  # testing float
        thousands="_", formatter=formatter, decimal=decimal, precision=precision
    )._translate(True, True)
    # 断言结果中指定列的显示值包含千位分隔符后的字符串
    assert "1_000_000" in result["body"][0][col]["display_value"]

    # 创建另一个 DataFrame 对象，包含一个整数值，使用样式对象进行样式化
    styler = DataFrame([[1000000]], index=[1000000]).style
    # 再次调用相同的格式化函数，测试整数值的样式化效果
    result = getattr(styler, func)(  # testing int
        thousands="_", formatter=formatter, decimal=decimal, precision=precision
    )._translate(True, True)
    # 断言结果中指定列的显示值包含千位分隔符后的字符串
    assert "1_000_000" in result["body"][0][col]["display_value"]

    # 创建另一个 DataFrame 对象，包含一个复数值，使用样式对象进行样式化
    styler = DataFrame([[1 + 1000000.123456789j]], index=[1 + 1000000.123456789j]).style
    # 再次调用相同的格式化函数，测试复数值的样式化效果
    result = getattr(styler, func)(  # testing complex
        thousands="_", formatter=formatter, decimal=decimal, precision=precision
    )._translate(True, True)
    # 断言结果中指定列的显示值包含千位分隔符后的字符串
    assert "1_000_000" in result["body"][0][col]["display_value"]


@pytest.mark.parametrize("formatter", [None, "{:,.4f}"])
@pytest.mark.parametrize("thousands", [None, ",", "*"])
@pytest.mark.parametrize("precision", [None, 4])
@pytest.mark.parametrize("func, col", [("format", 1), ("format_index", 0)])
def test_format_decimal(formatter, thousands, precision, func, col):
    # 创建一个 DataFrame 对象，包含一个浮点数值，使用样式对象进行样式化
    styler = DataFrame([[1000000.123456789]], index=[1000000.123456789]).style
    # 调用指定的格式化函数（如 format 或 format_index），设置小数点、千位分隔符、格式化字符串等参数，并翻译样式
    result = getattr(styler, func)(  # testing float
        decimal="_", formatter=formatter, thousands=thousands, precision=precision
    )._translate(True, True)
    # 断言结果中指定列的显示值包含指定小数点后的字符串
    assert "000_123" in result["body"][0][col]["display_value"]

    # 创建另一个 DataFrame 对象，包含一个复数值，使用样式对象进行样式化
    styler = DataFrame([[1 + 1000000.123456789j]], index=[1 + 1000000.123456789j]).style
    # 再次调用相同的格式化函数，测试复数值的样式化效果
    result = getattr(styler, func)(  # testing complex
        decimal="_", formatter=formatter, thousands=thousands, precision=precision
    )._translate(True, True)
    # 断言结果中指定列的显示值包含指定小数点后的字符串
    assert "000_123" in result["body"][0][col]["display_value"]


def test_str_escape_error():
    # 测试 _str_escape 函数是否正确抛出异常，检查传入的 escape 参数是否有效
    msg = "`escape` only permitted in {'html', 'latex', 'latex-math'}, got "
    with pytest.raises(ValueError, match=msg):
        _str_escape("text", "bad_escape")

    with pytest.raises(ValueError, match=msg):
        _str_escape("text", [])

    _str_escape(2.00, "bad_escape")  # OK since dtype is float


def test_long_int_formatting():
    # 创建一个包含长整数的 DataFrame 对象
    df = DataFrame(data=[[1234567890123456789]], columns=["test"])
    styler = df.style
    # 对 DataFrame 应用样式，并进行样式翻译
    ctx = styler._translate(True, True)
    # 断言结果中指定列的显示值为长整数的字符串形式
    assert ctx["body"][0][1]["display_value"] == "1234567890123456789"

    # 对 DataFrame 应用包含千位分隔符的样式，并进行样式翻译
    styler = df.style.format(thousands="_")
    ctx = styler._translate(True, True)
    # 断言结果中指定列的显示值包含千位分隔符后的长整数字符串形式
    assert ctx["body"][0][1]["display_value"] == "1_234_567_890_123_456_789"


def test_format_options():
    # 待实现的测试函数，用于测试格式化选项
    pass
    # 创建一个包含整数、浮点数和字符串的DataFrame对象
    df = DataFrame({"int": [2000, 1], "float": [1.009, None], "str": ["&<", "&~"]})
    # 获取DataFrame的样式对象，并进行翻译以生成样式上下文
    ctx = df.style._translate(True, True)

    # 测试选项: na_rep
    # 断言检查样式上下文中指定位置的显示值是否为 "nan"
    assert ctx["body"][1][2]["display_value"] == "nan"
    # 使用选项上下文设置 "styler.format.na_rep" 为 "MISSING"，重新生成样式上下文
    with option_context("styler.format.na_rep", "MISSING"):
        ctx_with_op = df.style._translate(True, True)
        # 断言检查更新后的样式上下文中指定位置的显示值是否为 "MISSING"
        assert ctx_with_op["body"][1][2]["display_value"] == "MISSING"

    # 测试选项: decimal 和 precision
    # 断言检查样式上下文中指定位置的显示值是否为 "1.009000"
    assert ctx["body"][0][2]["display_value"] == "1.009000"
    # 使用选项上下文设置 "styler.format.decimal" 为 "_"，重新生成样式上下文
    with option_context("styler.format.decimal", "_"):
        ctx_with_op = df.style._translate(True, True)
        # 断言检查更新后的样式上下文中指定位置的显示值是否为 "1_009000"
        assert ctx_with_op["body"][0][2]["display_value"] == "1_009000"
    # 使用选项上下文设置 "styler.format.precision" 为 2，重新生成样式上下文
    with option_context("styler.format.precision", 2):
        ctx_with_op = df.style._translate(True, True)
        # 断言检查更新后的样式上下文中指定位置的显示值是否为 "1.01"
        assert ctx_with_op["body"][0][2]["display_value"] == "1.01"

    # 测试选项: thousands
    # 断言检查样式上下文中指定位置的显示值是否为 "2000"
    assert ctx["body"][0][1]["display_value"] == "2000"
    # 使用选项上下文设置 "styler.format.thousands" 为 "_"，重新生成样式上下文
    with option_context("styler.format.thousands", "_"):
        ctx_with_op = df.style._translate(True, True)
        # 断言检查更新后的样式上下文中指定位置的显示值是否为 "2_000"
        assert ctx_with_op["body"][0][1]["display_value"] == "2_000"

    # 测试选项: escape
    # 断言检查样式上下文中指定位置的显示值是否为 "&<"
    assert ctx["body"][0][3]["display_value"] == "&<"
    # 断言检查样式上下文中指定位置的显示值是否为 "&~"
    assert ctx["body"][1][3]["display_value"] == "&~"
    # 使用选项上下文设置 "styler.format.escape" 为 "html"，重新生成样式上下文
    with option_context("styler.format.escape", "html"):
        ctx_with_op = df.style._translate(True, True)
        # 断言检查更新后的样式上下文中指定位置的显示值是否为 "&amp;&lt;"
        assert ctx_with_op["body"][0][3]["display_value"] == "&amp;&lt;"
    # 使用选项上下文设置 "styler.format.escape" 为 "latex"，重新生成样式上下文
    with option_context("styler.format.escape", "latex"):
        ctx_with_op = df.style._translate(True, True)
        # 断言检查更新后的样式上下文中指定位置的显示值是否为 "\\&\\textasciitilde "
        assert ctx_with_op["body"][1][3]["display_value"] == "\\&\\textasciitilde "
    # 使用选项上下文设置 "styler.format.escape" 为 "latex-math"，重新生成样式上下文
    with option_context("styler.format.escape", "latex-math"):
        ctx_with_op = df.style._translate(True, True)
        # 断言检查更新后的样式上下文中指定位置的显示值是否为 "\\&\\textasciitilde "
        assert ctx_with_op["body"][1][3]["display_value"] == "\\&\\textasciitilde "

    # 测试选项: formatter
    # 使用选项上下文设置 "styler.format.formatter" 对象格式化 "int" 列为 "{:,.2f}"，重新生成样式上下文
    with option_context("styler.format.formatter", {"int": "{:,.2f}"}):
        ctx_with_op = df.style._translate(True, True)
        # 断言检查更新后的样式上下文中指定位置的显示值是否为 "2,000.00"
        assert ctx_with_op["body"][0][1]["display_value"] == "2,000.00"
# 定义一个测试函数，用于测试 Styler 类在 precision=0 设置下的功能
def test_precision_zero(df):
    # 创建 Styler 对象，设置精度为 0
    styler = Styler(df, precision=0)
    # 转换 Styler 对象，获取上下文信息
    ctx = styler._translate(True, True)
    # 断言表格的第一行第三列的显示值为 "-1"
    assert ctx["body"][0][2]["display_value"] == "-1"
    # 断言表格的第二行第三列的显示值为 "-1"
    assert ctx["body"][1][2]["display_value"] == "-1"


# 使用 pytest 的参数化装饰器，测试不同的格式化选项验证函数
@pytest.mark.parametrize(
    "formatter, exp",
    [
        (lambda x: f"{x:.3f}", "9.000"),  # 使用 lambda 函数格式化为小数点后三位
        ("{:.2f}", "9.00"),  # 使用字符串格式化为小数点后两位
        ({0: "{:.1f}"}, "9.0"),  # 使用字典指定索引的格式化方式为小数点后一位
        (None, "9"),  # 使用 None 表示不进行格式化
    ],
)
def test_formatter_options_validator(formatter, exp):
    # 创建包含值为 9 的 DataFrame 对象
    df = DataFrame([[9]])
    # 在 styler.format.formatter 上下文中应用 formatter
    with option_context("styler.format.formatter", formatter):
        # 断言 exp 在 DataFrame 样式转换为 LaTeX 后的字符串中
        assert f" {exp} " in df.style.to_latex()


# 测试 formatter 选项引发异常情况
def test_formatter_options_raises():
    # 错误信息
    msg = "Value must be an instance of"
    # 断言在设置 styler.format.formatter 为 ["bad", "type"] 时引发 ValueError 异常，并匹配 msg 的错误信息
    with pytest.raises(ValueError, match=msg):
        with option_context("styler.format.formatter", ["bad", "type"]):
            # 创建空的 DataFrame 对象，并将其样式转换为 LaTeX 格式
            DataFrame().style.to_latex()


# 测试单级多索引的功能
def test_1level_multiindex():
    # GH 43383
    # 创建一个单级多索引对象
    midx = MultiIndex.from_product([[1, 2]], names=[""])
    # 创建一个值全部为 -1 的 DataFrame 对象，行索引为 midx，列索引为 [0, 1]
    df = DataFrame(-1, index=midx, columns=[0, 1])
    # 转换 DataFrame 样式，获取上下文信息
    ctx = df.style._translate(True, True)
    # 断言第一行第一列的显示值为 "1"
    assert ctx["body"][0][0]["display_value"] == "1"
    # 断言第一行第一列的可见性为 True
    assert ctx["body"][0][0]["is_visible"] is True
    # 断言第二行第一列的显示值为 "2"
    assert ctx["body"][1][0]["display_value"] == "2"
    # 断言第二行第一列的可见性为 True
    assert ctx["body"][1][0]["is_visible"] is True


# 测试布尔值格式化的功能
def test_boolean_format():
    # gh 46384: 布尔值在显示时不折叠为整数表示
    # 创建一个包含 True 和 False 的 DataFrame 对象
    df = DataFrame([[True, False]])
    # 转换 DataFrame 样式，获取上下文信息
    ctx = df.style._translate(True, True)
    # 断言第一行第二列的显示值为 True
    assert ctx["body"][0][1]["display_value"] is True
    # 断言第一行第三列的显示值为 False
    assert ctx["body"][0][2]["display_value"] is False


# 使用 pytest 的参数化装饰器，测试 relabel_index 函数的异常情况
@pytest.mark.parametrize(
    "hide, labels",
    [
        (False, [1, 2]),  # 不隐藏，标签长度为 2
        (True, [1, 2, 3, 4]),  # 隐藏，标签长度为 4
    ],
)
def test_relabel_raise_length(styler_multi, hide, labels):
    if hide:
        # 如果 hide 为 True，则隐藏 styler_multi 的部分索引
        styler_multi.hide(axis=0, subset=[("X", "x"), ("Y", "y")])
    # 断言在设置 labels 长度与隐藏状态不符时，调用 relabel_index 函数引发 ValueError 异常
    with pytest.raises(ValueError, match="``labels`` must be of length equal"):
        styler_multi.relabel_index(labels=labels)


# 测试 relabel_index 函数的功能
def test_relabel_index(styler_multi):
    # 设置新的标签
    labels = [(1, 2), (3, 4)]
    # 隐藏 styler_multi 的部分索引
    styler_multi.hide(axis=0, subset=[("X", "x"), ("Y", "y")])
    # 使用新的标签重命名索引
    styler_multi.relabel_index(labels=labels)
    # 转换 styler_multi 样式，获取上下文信息
    ctx = styler_multi._translate(True, True)
    # 断言 body 的第一行第一列的值为 {"value": "X", "display_value": 1} 的子集
    assert {"value": "X", "display_value": 1}.items() <= ctx["body"][0][0].items()
    # 断言 body 的第一行第二列的值为 {"value": "y", "display_value": 2} 的子集
    assert {"value": "y", "display_value": 2}.items() <= ctx["body"][0][1].items()
    # 断言 body 的第二行第一列的值为 {"value": "Y", "display_value": 3} 的子集
    assert {"value": "Y", "display_value": 3}.items() <= ctx["body"][1][0].items()
    # 断言 body 的第二行第二列的值为 {"value": "x", "display_value": 4} 的子集
    assert {"value": "x", "display_value": 4}.items() <= ctx["body"][1][1].items()


# 测试 relabel_columns 函数的功能
def test_relabel_columns(styler_multi):
    # 设置新的标签
    labels = [(1, 2), (3, 4)]
    # 隐藏 styler_multi 的部分列索引
    styler_multi.hide(axis=1, subset=[("A", "a"), ("B", "b")])
    # 使用新的标签重命名列索引
    styler_multi.relabel_index(axis=1, labels=labels)
    # 转换 styler_multi 样式，获取上下文信息
    ctx = styler_multi._translate(True, True)
    # 断言 head 的第一行第四列的值为 {"value": "A", "display_value": 1} 的子集
    assert {"value": "A", "display_value": 1}.items() <= ctx["head"][0][3].items()
    # 断言 head 的第一行第五列的值为 {"value": "B", "display_value": 3} 的子集
    assert {"value": "B", "display_value": 3}.items() <= ctx["head"][0][4].items()
    # 断言 head 的第二行第四列的值为 {"value": "b", "display_value": 2} 的子集
    assert {"value": "b", "display_value": 2}.items() <= ctx["head"][1][3].items()
    # 断言语句，验证左边字典的所有键值对是否都存在于右边字典中
    assert {"value": "a", "display_value": 4}.items() <= ctx["head"][1][4].items()
# 定义测试函数，用于测试重新标记索引的往返过程
def test_relabel_roundtrip(styler):
    # 使用styler对象重新标记索引
    styler.relabel_index(["{}", "{}"])
    # 获取翻译后的上下文
    ctx = styler._translate(True, True)
    # 断言首个数据行的内容符合预期
    assert {"value": "x", "display_value": "x"}.items() <= ctx["body"][0][0].items()
    # 断言第二个数据行的内容符合预期
    assert {"value": "y", "display_value": "y"}.items() <= ctx["body"][1][0].items()


# 使用pytest的参数化标记，测试格式化索引名称的多个情况
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize(
    "level, expected",
    [
        (0, ["X", "one"]),  # 索引级别为整数
        ("zero", ["X", "one"]),  # 索引级别为名称
        (1, ["zero", "X"]),  # 另一个索引级别为整数
        ("one", ["zero", "X"]),  # 另一个索引级别为名称
        ([0, 1], ["X", "X"]),  # 同时指定两个索引级别
        ([0, "zero"], ["X", "one"]),  # 同时指定整数和名称的索引级别
        ([0, "one"], ["X", "X"]),  # 同时指定整数和名称的两个索引级别
        (["one", "zero"], ["X", "X"]),  # 同时指定名称索引级别，但顺序相反
    ],
)
def test_format_index_names_level(axis, level, expected):
    # 创建一个包含隐藏名称的MultiIndex对象
    midx = MultiIndex.from_arrays([["_", "_"], ["_", "_"]], names=["zero", "one"])
    df = DataFrame([[1, 2], [3, 4]])
    # 根据axis参数选择操作索引或列
    if axis == 0:
        df.index = midx
    else:
        df.columns = midx

    # 使用styler对象格式化索引名称
    styler = df.style.format_index_names(lambda v: "X", level=level, axis=axis)
    # 获取翻译后的上下文
    ctx = styler._translate(True, True)

    if axis == 0:  # 比较索引
        result = [ctx["head"][1][s]["display_value"] for s in range(2)]
    else:  # 比较列
        result = [ctx["head"][s][0]["display_value"] for s in range(2)]
    # 断言结果符合预期
    assert expected == result


# 使用pytest的参数化标记，测试清除索引名称格式化的情况
@pytest.mark.parametrize(
    "attr, kwargs",
    [
        ("_display_funcs_index_names", {"axis": 0}),
        ("_display_funcs_column_names", {"axis": 1}),
    ],
)
def test_format_index_names_clear(styler, attr, kwargs):
    # 断言默认情况下不包含指定的格式化函数
    assert 0 not in getattr(styler, attr)  # 使用默认情况
    # 调用格式化索引名称函数
    styler.format_index_names("{:.2f}", **kwargs)
    # 断言现在包含了指定的格式化函数
    assert 0 in getattr(styler, attr)  # 格式化函数已指定
    # 再次调用格式化索引名称函数，清除格式化函数
    styler.format_index_names(**kwargs)
    # 断言现在不包含指定的格式化函数，回到默认状态
    assert 0 not in getattr(styler, attr)  # 格式化函数已清除至默认状态


# 使用pytest的参数化标记，测试可调用对象格式化索引名称的情况
@pytest.mark.parametrize("axis", [0, 1])
def test_format_index_names_callable(styler_multi, axis):
    # 获取翻译后的上下文
    ctx = styler_multi.format_index_names(
        lambda v: v.replace("_", "A"), axis=axis
    )._translate(True, True)
    # 检查翻译后的内容符合预期
    result = [
        ctx["head"][2][0]["display_value"],
        ctx["head"][2][1]["display_value"],
        ctx["head"][0][1]["display_value"],
        ctx["head"][1][1]["display_value"],
    ]
    # 根据axis参数选择预期的结果
    if axis == 0:
        expected = ["0A0", "0A1", "1_0", "1_1"]
    else:
        expected = ["0_0", "0_1", "1A0", "1A1"]
    # 断言结果符合预期
    assert result == expected


# 测试使用字典格式化索引名称的情况
def test_format_index_names_dict(styler_multi):
    # 获取翻译后的上下文
    ctx = (
        styler_multi.format_index_names({"0_0": "{:<<5}"})
        .format_index_names({"1_1": "{:>>4}"}, axis=1)
        ._translate(True, True)
    )
    # 断言特定索引位置的显示值符合预期
    assert ctx["head"][2][0]["display_value"] == "0_0<<"
    assert ctx["head"][1][1]["display_value"] == ">1_1"


# 测试处理隐藏级别索引名称的情况
def test_format_index_names_with_hidden_levels(styler_multi):
    # 获取翻译后的上下文
    ctx = styler_multi._translate(True, True)
    # 获取上下文中头部的完整高度
    full_head_height = len(ctx["head"])
    # 获取上下文中头部的完整宽度
    full_head_width = len(ctx["head"][0])
    # 断言确保头部的完整高度为3
    assert full_head_height == 3
    # 断言确保头部的完整宽度为6
    assert full_head_width == 6
    
    # 对上下文进行操作：隐藏第一轴第一级和第二轴第一级的内容
    # 同时格式化第二轴索引名称为右对齐，格式化第一轴索引名称为左对齐
    # 最后启用翻译功能
    ctx = (
        styler_multi.hide(axis=0, level=1)
        .hide(axis=1, level=1)
        .format_index_names("{:>>4}", axis=1)
        .format_index_names("{:!<5}")
        ._translate(True, True)
    )
    # 断言确保处理后的头部高度减少了1
    assert len(ctx["head"]) == full_head_height - 1
    # 断言确保处理后的头部宽度减少了1
    assert len(ctx["head"][0]) == full_head_width - 1
    # 断言头部第一行第一列的显示值为">1_0"
    assert ctx["head"][0][0]["display_value"] == ">1_0"
    # 断言头部第二行第一列的显示值为"0_0!!"
    assert ctx["head"][1][0]["display_value"] == "0_0!!"
```