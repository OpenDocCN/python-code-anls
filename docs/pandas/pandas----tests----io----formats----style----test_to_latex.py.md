# `D:\src\scipysrc\pandas\pandas\tests\io\formats\style\test_to_latex.py`

```
from textwrap import dedent  # 导入 textwrap 模块中的 dedent 函数，用于缩减多行字符串的缩进

import numpy as np  # 导入 NumPy 库，使用 np 别名
import pytest  # 导入 pytest 测试框架

from pandas import (  # 从 pandas 库中导入以下对象：
    DataFrame,  # DataFrame 类
    MultiIndex,  # MultiIndex 类
    Series,  # Series 类
    option_context,  # option_context 函数
)

pytest.importorskip("jinja2")  # 导入 jinja2 库，如果不存在则跳过测试

from pandas.io.formats.style import Styler  # 从 pandas 库中导入 Styler 类
from pandas.io.formats.style_render import (  # 从 pandas 库中导入以下函数：
    _parse_latex_cell_styles,  # _parse_latex_cell_styles 函数
    _parse_latex_css_conversion,  # _parse_latex_css_conversion 函数
    _parse_latex_header_span,  # _parse_latex_header_span 函数
    _parse_latex_table_styles,  # _parse_latex_table_styles 函数
    _parse_latex_table_wrapping,  # _parse_latex_table_wrapping 函数
)


@pytest.fixture  # 声明一个 pytest 的 fixture
def df():  # 定义名为 df 的 fixture 函数
    return DataFrame(  # 返回一个 DataFrame 对象
        {"A": [0, 1], "B": [-0.61, -1.22], "C": Series(["ab", "cd"], dtype=object)}  # DataFrame 的初始化数据
    )


@pytest.fixture  # 声明一个 pytest 的 fixture
def df_ext():  # 定义名为 df_ext 的 fixture 函数
    return DataFrame(  # 返回一个 DataFrame 对象
        {"A": [0, 1, 2], "B": [-0.61, -1.22, -2.22], "C": ["ab", "cd", "de"]}  # DataFrame 的初始化数据
    )


@pytest.fixture  # 声明一个 pytest 的 fixture
def styler(df):  # 定义名为 styler 的 fixture 函数，依赖于 df fixture
    return Styler(df, uuid_len=0, precision=2)  # 返回一个 Styler 对象，使用给定的参数


def test_minimal_latex_tabular(styler):  # 定义一个测试函数，测试最小化的 LaTeX 表格生成
    expected = dedent(  # 使用 dedent 函数缩减多行字符串的缩进
        """\
        \\begin{tabular}{lrrl}
         & A & B & C \\\\
        0 & 0 & -0.61 & ab \\\\
        1 & 1 & -1.22 & cd \\\\
        \\end{tabular}
        """
    )
    assert styler.to_latex() == expected  # 断言生成的 LaTeX 代码与预期的字符串相同


def test_tabular_hrules(styler):  # 定义一个测试函数，测试包含水平线的 LaTeX 表格生成
    expected = dedent(  # 使用 dedent 函数缩减多行字符串的缩进
        """\
        \\begin{tabular}{lrrl}
        \\toprule
         & A & B & C \\\\
        \\midrule
        0 & 0 & -0.61 & ab \\\\
        1 & 1 & -1.22 & cd \\\\
        \\bottomrule
        \\end{tabular}
        """
    )
    assert styler.to_latex(hrules=True) == expected  # 断言生成的 LaTeX 代码与预期的字符串相同


def test_tabular_custom_hrules(styler):  # 定义一个测试函数，测试自定义水平线的 LaTeX 表格生成
    styler.set_table_styles(  # 设置 Styler 对象的表格样式
        [
            {"selector": "toprule", "props": ":hline"},  # 设置表格顶部的样式为水平线
            {"selector": "bottomrule", "props": ":otherline"},  # 设置表格底部的样式为其他类型的线
        ]
    )  # 没有设置中间水平线
    expected = dedent(  # 使用 dedent 函数缩减多行字符串的缩进
        """\
        \\begin{tabular}{lrrl}
        \\hline
         & A & B & C \\\\
        0 & 0 & -0.61 & ab \\\\
        1 & 1 & -1.22 & cd \\\\
        \\otherline
        \\end{tabular}
        """
    )
    assert styler.to_latex() == expected  # 断言生成的 LaTeX 代码与预期的字符串相同


def test_column_format(styler):  # 定义一个测试函数，测试列格式设置的 LaTeX 表格生成
    # 默认设置已经在 `test_latex_minimal_tabular` 中测试过了
    styler.set_table_styles([{"selector": "column_format", "props": ":cccc"}])  # 设置列格式样式

    assert "\\begin{tabular}{rrrr}" in styler.to_latex(column_format="rrrr")  # 断言包含指定列格式的 LaTeX 代码
    styler.set_table_styles([{"selector": "column_format", "props": ":r|r|cc"}])  # 设置不同的列格式样式
    assert "\\begin{tabular}{r|r|cc}" in styler.to_latex()  # 断言包含指定列格式的 LaTeX 代码


def test_siunitx_cols(styler):  # 定义一个测试函数，测试使用 siunitx 宏包的 LaTeX 表格生成
    expected = dedent(  # 使用 dedent 函数缩减多行字符串的缩进
        """\
        \\begin{tabular}{lSSl}
        {} & {A} & {B} & {C} \\\\
        0 & 0 & -0.61 & ab \\\\
        1 & 1 & -1.22 & cd \\\\
        \\end{tabular}
        """
    )
    assert styler.to_latex(siunitx=True) == expected  # 断言生成的 LaTeX 代码与预期的字符串相同


def test_position(styler):  # 定义一个测试函数，测试 LaTeX 表格的位置设置
    assert "\\begin{table}[h!]" in styler.to_latex(position="h!")  # 断言包含指定位置设置的 LaTeX 代码
    assert "\\end{table}" in styler.to_latex(position="h!")  # 断言包含指定位置设置的 LaTeX 代码
    styler.set_table_styles([{"selector": "position", "props": ":b!"}])  # 设置表格位置样式
    assert "\\begin{table}[b!]" in styler.to_latex()  # 断言生成的 LaTeX 代码与预期的字符串相同
    assert "\\end{table}" in styler.to_latex()  # 断言生成的 LaTeX 代码与预期的字符串相同


@pytest.mark.parametrize("env", [None, "longtable"])  # 使用 pytest 的参数化装饰器进行参数化测试
def test_label(styler, env):
    # 断言生成的 LaTeX 中包含指定的标签
    assert "\n\\label{text}" in styler.to_latex(label="text", environment=env)
    # 设置表格样式，包含选择器和属性
    styler.set_table_styles([{"selector": "label", "props": ":{more §text}"}])
    # 断言生成的 LaTeX 中包含设置的新样式
    assert "\n\\label{more :text}" in styler.to_latex(environment=env)


def test_position_float_raises(styler):
    # 错误消息定义
    msg = "`position_float` should be one of 'raggedright', 'raggedleft', 'centering',"
    # 使用 pytest 来断言抛出特定错误消息的异常
    with pytest.raises(ValueError, match=msg):
        styler.to_latex(position_float="bad_string")

    # 错误消息定义
    msg = "`position_float` cannot be used in 'longtable' `environment`"
    # 使用 pytest 来断言抛出特定错误消息的异常
    with pytest.raises(ValueError, match=msg):
        styler.to_latex(position_float="centering", environment="longtable")


@pytest.mark.parametrize("label", [(None, ""), ("text", "\\label{text}")])
@pytest.mark.parametrize("position", [(None, ""), ("h!", "{table}[h!]")])
@pytest.mark.parametrize("caption", [(None, ""), ("text", "\\caption{text}")])
@pytest.mark.parametrize("column_format", [(None, ""), ("rcrl", "{tabular}{rcrl}")])
@pytest.mark.parametrize("position_float", [(None, ""), ("centering", "\\centering")])
def test_kwargs_combinations(
    styler, label, position, caption, column_format, position_float
):
    # 测试多种参数组合生成的 LaTeX 是否包含预期的内容
    result = styler.to_latex(
        label=label[0],
        position=position[0],
        caption=caption[0],
        column_format=column_format[0],
        position_float=position_float[0],
    )
    assert label[1] in result
    assert position[1] in result
    assert caption[1] in result
    assert column_format[1] in result
    assert position_float[1] in result


def test_custom_table_styles(styler):
    # 设置自定义表格样式，并断言生成的 LaTeX 中包含预期的表格内容
    styler.set_table_styles(
        [
            {"selector": "mycommand", "props": ":{myoptions}"},
            {"selector": "mycommand2", "props": ":{myoptions2}"},
        ]
    )
    expected = dedent(
        """\
        \\begin{table}
        \\mycommand{myoptions}
        \\mycommand2{myoptions2}
        """
    )
    assert expected in styler.to_latex()


def test_cell_styling(styler):
    # 设置单元格样式，并断言生成的 LaTeX 中包含预期的表格内容
    styler.highlight_max(props="itshape:;Huge:--wrap;")
    expected = dedent(
        """\
        \\begin{tabular}{lrrl}
         & A & B & C \\\\
        0 & 0 & \\itshape {\\Huge -0.61} & ab \\\\
        1 & \\itshape {\\Huge 1} & -1.22 & \\itshape {\\Huge cd} \\\\
        \\end{tabular}
        """
    )
    assert expected == styler.to_latex()


def test_multiindex_columns(df):
    # 设置多重索引列，并断言生成的 LaTeX 中包含预期的表格内容
    cidx = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "c")])
    df.columns = cidx
    expected = dedent(
        """\
        \\begin{tabular}{lrrl}
         & \\multicolumn{2}{r}{A} & B \\\\
         & a & b & c \\\\
        0 & 0 & -0.61 & ab \\\\
        1 & 1 & -1.22 & cd \\\\
        \\end{tabular}
        """
    )
    s = df.style.format(precision=2)
    assert expected == s.to_latex()
    # 期望的字符串，包含了一个 LaTeX 格式的表格定义
    expected = dedent(
        """\
        \\begin{tabular}{lrrl}
         & A & A & B \\\\
         & a & b & c \\\\
        0 & 0 & -0.61 & ab \\\\
        1 & 1 & -1.22 & cd \\\\
        \\end{tabular}
        """
    )
    # 将 DataFrame 的样式格式化为 LaTeX，设置数字精度为两位
    s = df.style.format(precision=2)
    # 断言期望的 LaTeX 表格字符串与格式化后的样式字符串相等
    assert expected == s.to_latex(sparse_columns=False)
def test_multicol_naive(df, multicol_align, siunitx, header):
    # 创建包含多级索引的行索引对象
    ridx = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("A", "c")])
    # 将创建的多级索引对象设置为DataFrame的列索引
    df.columns = ridx
    # 根据siunitx标志选择合适的格式字符串，设置level1变量
    level1 = " & a & b & c" if not siunitx else "{} & {a} & {b} & {c}"
    # 根据siunitx标志选择合适的列格式字符串，设置col_format变量
    col_format = "lrrl" if not siunitx else "lSSl"



def test_multiindex_row(df_ext):
    # 创建包含多级元组的行索引对象
    ridx = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "c")])
    # 将创建的多级索引对象设置为DataFrame的行索引
    df_ext.index = ridx
    # 期望的Latex表格字符串
    expected = dedent(
        """\
        \\begin{tabular}{llrrl}
         &  & A & B & C \\\\
        \\multirow[c]{2}{*}{A} & a & 0 & -0.61 & ab \\\\
         & b & 1 & -1.22 & cd \\\\
        B & c & 2 & -2.22 & de \\\\
        \\end{tabular}
        """
    )
    # 将DataFrame样式化为Latex格式
    styler = df_ext.style.format(precision=2)
    # 转换成Latex字符串并与期望值进行比较
    result = styler.to_latex()
    # 断言结果与期望值相等
    assert expected == result

    # 非稀疏索引的期望Latex表格字符串
    expected = dedent(
        """\
        \\begin{tabular}{llrrl}
         &  & A & B & C \\\\
        A & a & 0 & -0.61 & ab \\\\
        A & b & 1 & -1.22 & cd \\\\
        B & c & 2 & -2.22 & de \\\\
        \\end{tabular}
        """
    )
    # 将DataFrame样式化为Latex格式（非稀疏索引）
    result = styler.to_latex(sparse_index=False)
    # 断言结果与期望值相等
    assert expected == result



def test_multirow_naive(df_ext):
    # 创建包含多级元组的行索引对象
    ridx = MultiIndex.from_tuples([("X", "x"), ("X", "y"), ("Y", "z")])
    # 将创建的多级索引对象设置为DataFrame的行索引
    df_ext.index = ridx
    # 期望的Latex表格字符串
    expected = dedent(
        """\
        \\begin{tabular}{llrrl}
         &  & A & B & C \\\\
        X & x & 0 & -0.61 & ab \\\\
         & y & 1 & -1.22 & cd \\\\
        Y & z & 2 & -2.22 & de \\\\
        \\end{tabular}
        """
    )
    # 将DataFrame样式化为Latex格式，使用naive多行对齐
    styler = df_ext.style.format(precision=2)
    # 转换成Latex字符串并与期望值进行比较
    result = styler.to_latex(multirow_align="naive")
    # 断言结果与期望值相等
    assert expected == result



def test_multiindex_row_and_col(df_ext):
    # 创建包含多级元组的行索引对象和列索引对象
    cidx = MultiIndex.from_tuples([("Z", "a"), ("Z", "b"), ("Y", "c")])
    ridx = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "c")])
    # 将创建的多级索引对象分别设置为DataFrame的行索引和列索引
    df_ext.index, df_ext.columns = ridx, cidx
    # 期望的Latex表格字符串
    expected = dedent(
        """\
        \\begin{tabular}{llrrl}
         &  & \\multicolumn{2}{l}{Z} & Y \\\\
         &  & a & b & c \\\\
        \\multirow[b]{2}{*}{A} & a & 0 & -0.61 & ab \\\\
         & b & 1 & -1.22 & cd \\\\
        B & c & 2 & -2.22 & de \\\\
        \\end{tabular}
        """
    )
    # 将DataFrame样式化为Latex格式，使用b对齐的多行和l对齐的多列
    styler = df_ext.style.format(precision=2)
    # 转换成Latex字符串并与期望值进行比较
    result = styler.to_latex(multirow_align="b", multicol_align="l")
    # 断言结果与期望值相等
    assert result == expected

    # 非稀疏索引和列的期望Latex表格字符串
    expected = dedent(
        """\
        \\begin{tabular}{llrrl}
         &  & Z & Z & Y \\\\
         &  & a & b & c \\\\
        A & a & 0 & -0.61 & ab \\\\
        A & b & 1 & -1.22 & cd \\\\
        B & c & 2 & -2.22 & de \\\\
        \\end{tabular}
        """
    )
    # 将DataFrame样式化为Latex格式（非稀疏索引和列）
    result = styler.to_latex(sparse_index=False, sparse_columns=False)
    # 断言结果与期望值相等
    assert result == expected



@pytest.mark.parametrize(
    "multicol_align, siunitx, header",
    [
        ("naive-l", False, " & A & &"),
        ("naive-r", False, " & & & A"),
        ("naive-l", True, "{} & {A} & {} & {}"),
        ("naive-r", True, "{} & {} & {} & {A}"),
    ],
)
def test_multicol_naive(df, multicol_align, siunitx, header):
    # 创建包含多级元组的行索引对象
    ridx = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("A", "c")])
    # 将创建的多级索引对象设置为DataFrame的列索引
    df.columns = ridx
    # 根据siunitx和multicol_align参数设置header字符串
    level1 = " & a & b & c" if not siunitx else "{} & {a} & {b} & {c}"
    # 根据siunitx参数设置列格式字符串
    col_format = "lrrl" if not siunitx else "lSSl"
    # 生成预期的 LaTeX 格式字符串，包括表格环境、表头和数据行
    expected = dedent(
        f"""\
        \\begin{{tabular}}{{{col_format}}}
        {header} \\\\
        {level1} \\\\
        0 & 0 & -0.61 & ab \\\\
        1 & 1 & -1.22 & cd \\\\
        \\end{{tabular}}
        """
    )
    # 使用 Pandas DataFrame 的 styler 格式化数据，并设置小数点精度为 2
    styler = df.style.format(precision=2)
    # 将格式化后的 DataFrame 转换为 LaTeX 格式的字符串
    result = styler.to_latex(multicol_align=multicol_align, siunitx=siunitx)
    # 断言预期的 LaTeX 字符串与实际生成的 LaTeX 字符串相等
    assert expected == result
# 定义一个测试函数，用于测试 DataFrame 的多重选项设置
def test_multi_options(df_ext):
    # 创建多级行索引对象
    cidx = MultiIndex.from_tuples([("Z", "a"), ("Z", "b"), ("Y", "c")])
    # 创建多级列索引对象
    ridx = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "c")])
    # 将 DataFrame 的行索引和列索引设置为刚创建的索引对象
    df_ext.index, df_ext.columns = ridx, cidx
    # 通过设置精度为2来格式化 DataFrame，并获取样式对象
    styler = df_ext.style.format(precision=2)

    # 期望的 LaTeX 输出内容，使用 dedent 去除前导空格和缩进
    expected = dedent(
        """\
         &  & \\multicolumn{2}{r}{Z} & Y \\\\
         &  & a & b & c \\\\
        \\multirow[c]{2}{*}{A} & a & 0 & -0.61 & ab \\\\
        """
    )
    # 将 DataFrame 样式转换为 LaTeX，并断言期望的内容在结果中
    result = styler.to_latex()
    assert expected in result

    # 使用上下文管理器设置 styler.latex.multicol_align 为 "l"，断言对应的内容在 LaTeX 输出中
    with option_context("styler.latex.multicol_align", "l"):
        assert " &  & \\multicolumn{2}{l}{Z} & Y \\\\" in styler.to_latex()

    # 使用上下文管理器设置 styler.latex.multirow_align 为 "b"，断言对应的内容在 LaTeX 输出中
    with option_context("styler.latex.multirow_align", "b"):
        assert "\\multirow[b]{2}{*}{A} & a & 0 & -0.61 & ab \\\\" in styler.to_latex()


# 定义一个测试函数，用于测试隐藏多级索引的列
def test_multiindex_columns_hidden():
    # 创建一个包含数据的 DataFrame
    df = DataFrame([[1, 2, 3, 4]])
    # 设置 DataFrame 的列索引为多级索引对象
    df.columns = MultiIndex.from_tuples([("A", 1), ("A", 2), ("A", 3), ("B", 1)])
    # 获取 DataFrame 的样式对象
    s = df.style
    # 断言 LaTeX 输出中包含 "{tabular}{lrrrr}"
    assert "{tabular}{lrrrr}" in s.to_latex()
    # 清空表格样式并重置位置命令
    s.set_table_styles([])
    # 隐藏指定的列（这里隐藏 ("A", 2)）
    s.hide([("A", 2)], axis="columns")
    # 断言 LaTeX 输出中包含 "{tabular}{lrrr}"
    assert "{tabular}{lrrr}" in s.to_latex()


# 使用参数化测试装饰器定义测试函数，用于测试 DataFrame 样式的稀疏选项
@pytest.mark.parametrize(
    "option, value",
    [
        ("styler.sparse.index", True),
        ("styler.sparse.index", False),
        ("styler.sparse.columns", True),
        ("styler.sparse.columns", False),
    ],
)
def test_sparse_options(df_ext, option, value):
    # 创建多级行索引对象和多级列索引对象
    cidx = MultiIndex.from_tuples([("Z", "a"), ("Z", "b"), ("Y", "c")])
    ridx = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "c")])
    # 将 DataFrame 的行索引和列索引设置为刚创建的索引对象
    df_ext.index, df_ext.columns = ridx, cidx
    # 获取 DataFrame 的样式对象
    styler = df_ext.style

    # 获取未应用选项设置时的 LaTeX 输出
    latex1 = styler.to_latex()
    # 使用上下文管理器设置指定的稀疏选项，并获取相应的 LaTeX 输出
    with option_context(option, value):
        latex2 = styler.to_latex()
    # 断言应用选项前后的 LaTeX 输出是否符合预期
    assert (latex1 == latex2) is value


# 定义一个测试函数，用于测试隐藏行索引
def test_hidden_index(styler):
    # 隐藏 DataFrame 样式中的行索引
    styler.hide(axis="index")
    # 期望的 LaTeX 输出内容，使用 dedent 去除前导空格和缩进
    expected = dedent(
        """\
        \\begin{tabular}{rrl}
        A & B & C \\\\
        0 & -0.61 & ab \\\\
        1 & -1.22 & cd \\\\
        \\end{tabular}
        """
    )
    # 断言 DataFrame 样式转换为 LaTeX 输出是否与期望内容相符
    assert styler.to_latex() == expected


# 使用参数化测试装饰器定义测试函数，用于综合测试 DataFrame 样式的多个特性
@pytest.mark.parametrize("environment", ["table", "figure*", None])
def test_comprehensive(df_ext, environment):
    # 创建多级行索引对象和多级列索引对象
    cidx = MultiIndex.from_tuples([("Z", "a"), ("Z", "b"), ("Y", "c")])
    ridx = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "c")])
    # 将 DataFrame 的行索引和列索引设置为刚创建的索引对象
    df_ext.index, df_ext.columns = ridx, cidx
    # 获取 DataFrame 的样式对象
    stlr = df_ext.style
    # 设置表格标题为 "mycap"
    stlr.set_caption("mycap")
    stlr.set_table_styles(
        [
            {"selector": "label", "props": ":{fig§item}"},  # 设置表格样式，指定选择器为"label"，属性为"{fig§item}"
            {"selector": "position", "props": ":h!"},  # 设置选择器为"position"，属性为":h!"
            {"selector": "position_float", "props": ":centering"},  # 设置选择器为"position_float"，属性为":centering"
            {"selector": "column_format", "props": ":rlrlr"},  # 设置选择器为"column_format"，属性为":rlrlr"
            {"selector": "toprule", "props": ":toprule"},  # 设置选择器为"toprule"，属性为":toprule"
            {"selector": "midrule", "props": ":midrule"},  # 设置选择器为"midrule"，属性为":midrule"
            {"selector": "bottomrule", "props": ":bottomrule"},  # 设置选择器为"bottomrule"，属性为":bottomrule"
            {"selector": "rowcolors", "props": ":{3}{pink}{}"},  # 设置选择器为"rowcolors"，属性为":{3}{pink}{}"，自定义命令
        ]
    )
    stlr.highlight_max(axis=0, props="textbf:--rwrap;cellcolor:[rgb]{1,1,0.6}--rwrap")  # 对表格进行最大值高亮显示，指定轴为0，属性为"textbf:--rwrap;cellcolor:[rgb]{1,1,0.6}--rwrap"
    stlr.highlight_max(axis=None, props="Huge:--wrap;", subset=[("Z", "a"), ("Z", "b")])  # 对表格进行最大值高亮显示，无指定轴，属性为"Huge:--wrap;"，子集为[("Z", "a"), ("Z", "b")]

    expected = (
        """\
# 定义一个 LaTeX 表格环境
def test_environment_option(styler):
    # 使用指定的环境设置样式，并检查生成的 LaTeX 是否包含对应的环境声明
    with option_context("styler.latex.environment", "bar-env"):
        assert "\\begin{bar-env}" in styler.to_latex()
        assert "\\begin{foo-env}" in styler.to_latex(environment="foo-env")


# 测试解析 LaTeX 表格样式的函数
def test_parse_latex_table_styles(styler):
    # 设置表格样式列表，包含不同选择器和属性，验证解析函数返回的结果是否正确
    styler.set_table_styles(
        [
            {"selector": "foo", "props": [("attr", "value")]},
            {"selector": "bar", "props": [("attr", "overwritten")]},
            {"selector": "bar", "props": [("attr", "baz"), ("attr2", "ignored")]},
            {"selector": "label", "props": [("", "{fig§item}")]},
        ]
    )
    # 验证特定选择器的样式解析结果是否符合预期
    assert _parse_latex_table_styles(styler.table_styles, "bar") == "baz"

    # 验证样式中的特定字符替换是否正确进行（将 '§' 替换为 ':'）
    assert _parse_latex_table_styles(styler.table_styles, "label") == "{fig:item}"


# 测试基本的 LaTeX 单元格样式解析函数
def test_parse_latex_cell_styles_basic():
    # 定义单元格样式列表，验证解析后的 LaTeX 表达式是否符合预期
    cell_style = [("itshape", "--rwrap"), ("cellcolor", "[rgb]{0,1,1}--rwrap")]
    expected = "\\itshape{\\cellcolor[rgb]{0,1,1}{text}}"
    assert _parse_latex_cell_styles(cell_style, "text") == expected


# 使用参数化测试来验证 LaTeX 单元格样式解析函数中不同选项的处理情况
@pytest.mark.parametrize(
    "wrap_arg, expected",
    [
        ("", "\\command<options> <display_value>"),
        ("--wrap", "{\\command<options> <display_value>}"),
        ("--nowrap", "\\command<options> <display_value>"),
        ("--lwrap", "{\\command<options>} <display_value>"),
        ("--dwrap", "{\\command<options>}{<display_value>}"),
        ("--rwrap", "\\command<options>{<display_value>}"),
    ],
)
def test_parse_latex_cell_styles_braces(wrap_arg, expected):
    # 定义单元格样式列表，验证使用不同选项后的解析结果是否符合预期
    cell_style = [("<command>", f"<options>{wrap_arg}")]
    assert _parse_latex_cell_styles(cell_style, "<display_value>") == expected


# 测试解析 LaTeX 表头跨度的函数
def test_parse_latex_header_span():
    # 定义包含不同属性的单元格对象，验证解析后的 LaTeX 表达式是否符合预期
    cell = {"attributes": 'colspan="3"', "display_value": "text", "cellstyle": []}
    expected = "\\multicolumn{3}{Y}{text}"
    assert _parse_latex_header_span(cell, "X", "Y") == expected

    cell = {"attributes": 'rowspan="5"', "display_value": "text", "cellstyle": []}
    expected = "\\multirow[X]{5}{*}{text}"
    assert _parse_latex_header_span(cell, "X", "Y") == expected

    cell = {"display_value": "text", "cellstyle": []}
    # 验证对于无跨度属性的单元格，解析函数的返回是否只包含文本值
    assert _parse_latex_header_span(cell, "X", "Y") == "text"
    # 定义一个包含特定键值对的字典，表示一个单元格的显示值和样式
    cell = {"display_value": "text", "cellstyle": [("bfseries", "--rwrap")]}
    
    # 断言语句，调用函数 _parse_latex_header_span，预期其返回值与 "\\bfseries{text}" 相等
    assert _parse_latex_header_span(cell, "X", "Y") == "\\bfseries{text}"
# 设置表格样式，包括顶部规则、底部规则、中部规则和列格式选择器
styler.set_table_styles(
    [
        {"selector": "toprule", "props": ":value"},
        {"selector": "bottomrule", "props": ":value"},
        {"selector": "midrule", "props": ":value"},
        {"selector": "column_format", "props": ":value"},
    ]
)
# 断言调用_parse_latex_table_wrapping函数返回False
assert _parse_latex_table_wrapping(styler.table_styles, styler.caption) is False
# 断言调用_parse_latex_table_wrapping函数返回True，使用不同的标题参数
assert _parse_latex_table_wrapping(styler.table_styles, "some caption") is True

# 设置表格样式，包括非忽略选择器和其属性，不覆盖原有样式
styler.set_table_styles(
    [
        {"selector": "not-ignored", "props": ":value"},
    ],
    overwrite=False,
)
# 断言调用_parse_latex_table_wrapping函数返回True，使用None作为标题参数
assert _parse_latex_table_wrapping(styler.table_styles, None) is True
    # 生成预期的字符串，使用了多行字符串的格式化，包含了变量 {exp} 和 {inner_env}
    expected = dedent(
        f"""\
        0 & 0 & \\{exp} -0.61 & ab \\\\
        1 & \\{exp} 1 & -1.22 & \\{exp} cd \\\\
        \\end{{{inner_env}}}
    """
    )
    # 使用断言检查预期的字符串是否包含在结果中
    assert expected in result
def test_parse_latex_css_conversion_option():
    # 准备测试用例：定义一个简单的 CSS 列表
    css = [("command", "option--latex--wrap")]
    # 期望的结果是将 "option--latex--wrap" 转换为 "option--wrap"
    expected = [("command", "option--wrap")]
    # 调用函数进行转换
    result = _parse_latex_css_conversion(css)
    # 断言结果与期望是否一致
    assert result == expected


def test_styler_object_after_render(styler):
    # GH 42320: 测试在渲染之后，styler 对象的状态保持不变
    # 在渲染前创建 styler 的深拷贝
    pre_render = styler._copy(deepcopy=True)
    # 将 styler 转换为 LaTeX 格式，设置多个参数
    styler.to_latex(
        column_format="rllr",
        position="h",
        position_float="centering",
        hrules=True,
        label="my lab",
        caption="my cap",
    )
    # 断言渲染前后 styler 的 table_styles 属性和 caption 属性不变
    assert pre_render.table_styles == styler.table_styles
    assert pre_render.caption == styler.caption


def test_longtable_comprehensive(styler):
    # 测试长表格的全面性功能
    result = styler.to_latex(
        environment="longtable", hrules=True, label="fig:A", caption=("full", "short")
    )
    # 期望的 LaTeX 格式长表格内容
    expected = dedent(
        """\
        \\begin{longtable}{lrrl}
        \\caption[short]{full} \\label{fig:A} \\\\
        \\toprule
         & A & B & C \\\\
        \\midrule
        \\endfirsthead
        \\caption[]{full} \\\\
        \\toprule
         & A & B & C \\\\
        \\midrule
        \\endhead
        \\midrule
        \\multicolumn{4}{r}{Continued on next page} \\\\
        \\midrule
        \\endfoot
        \\bottomrule
        \\endlastfoot
        0 & 0 & -0.61 & ab \\\\
        1 & 1 & -1.22 & cd \\\\
        \\end{longtable}
    """
    )
    # 断言生成的 LaTeX 与期望的内容一致
    assert result == expected


def test_longtable_minimal(styler):
    # 测试长表格的最小化功能
    result = styler.to_latex(environment="longtable")
    # 期望的最小化 LaTeX 格式长表格内容
    expected = dedent(
        """\
        \\begin{longtable}{lrrl}
         & A & B & C \\\\
        \\endfirsthead
         & A & B & C \\\\
        \\endhead
        \\multicolumn{4}{r}{Continued on next page} \\\\
        \\endfoot
        \\endlastfoot
        0 & 0 & -0.61 & ab \\\\
        1 & 1 & -1.22 & cd \\\\
        \\end{longtable}
    """
    )
    # 断言生成的 LaTeX 与期望的内容一致
    assert result == expected


@pytest.mark.parametrize(
    "sparse, exp, siunitx",
    [
        (True, "{} & \\multicolumn{2}{r}{A} & {B}", True),
        (False, "{} & {A} & {A} & {B}", True),
        (True, " & \\multicolumn{2}{r}{A} & B", False),
        (False, " & A & A & B", False),
    ],
)
def test_longtable_multiindex_columns(df, sparse, exp, siunitx):
    # 参数化测试：测试多级索引列的长表格生成
    # 创建多级索引列
    cidx = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "c")])
    df.columns = cidx
    # 根据参数化设置生成期望的 LaTeX 内容
    with_si = "{} & {a} & {b} & {c} \\\\"
    without_si = " & a & b & c \\\\"
    expected = dedent(
        f"""\
        \\begin{{longtable}}{{l{"SS" if siunitx else "rr"}l}}
        {exp} \\\\
        {with_si if siunitx else without_si}
        \\endfirsthead
        {exp} \\\\
        {with_si if siunitx else without_si}
        \\endhead
        """
    )
    # 生成 DataFrame 的样式，并生成 LaTeX 输出
    result = df.style.to_latex(
        environment="longtable", sparse_columns=sparse, siunitx=siunitx
    )
    # 断言生成的 LaTeX 包含期望的内容
    assert expected in result


@pytest.mark.parametrize(
    "caption, cap_exp",
    [
        ("full", ("{full}", "")),
        (("full", "short"), ("{full}", "[short]")),
    ],
)
@pytest.mark.parametrize("label, lab_exp", [(None, ""), ("tab:A", " \\label{tab:A}")])
def test_longtable_caption_label(styler, caption, cap_exp, label, lab_exp):
    # 构造带有标签和标题的长表格的预期输出
    cap_exp1 = f"\\caption{cap_exp[1]}{cap_exp[0]}"
    cap_exp2 = f"\\caption[]{cap_exp[0]}"

    # 构造预期的长表格格式字符串
    expected = dedent(
        f"""\
        {cap_exp1}{lab_exp} \\\\
         & A & B & C \\\\
        \\endfirsthead
        {cap_exp2} \\\\
        """
    )
    # 断言预期字符串是否在转换后的 LaTeX 输出中
    assert expected in styler.to_latex(
        environment="longtable", caption=caption, label=label
    )


@pytest.mark.parametrize("index", [True, False])
@pytest.mark.parametrize(
    "columns, siunitx",
    [
        (True, True),
        (True, False),
        (False, False),
    ],
)
def test_apply_map_header_render_mi(df_ext, index, columns, siunitx):
    # 准备测试数据的行和列索引
    cidx = MultiIndex.from_tuples([("Z", "a"), ("Z", "b"), ("Y", "c")])
    ridx = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "c")])
    df_ext.index, df_ext.columns = ridx, cidx
    styler = df_ext.style

    # 定义用于映射样式的函数
    func = lambda v: "bfseries: --rwrap" if "A" in v or "Z" in v or "c" in v else None

    # 根据测试参数条件对样式进行索引和列的映射
    if index:
        styler.map_index(func, axis="index")
    if columns:
        styler.map_index(func, axis="columns")

    # 将样式转换为 LaTeX 格式
    result = styler.to_latex(siunitx=siunitx)

    # 断言预期的索引部分是否在结果中（依据 index 参数）
    expected_index = dedent(
        """\
    \\multirow[c]{2}{*}{\\bfseries{A}} & a & 0 & -0.610000 & ab \\\\
    \\bfseries{} & b & 1 & -1.220000 & cd \\\\
    B & \\bfseries{c} & 2 & -2.220000 & de \\\\
    """
    )
    assert (expected_index in result) is index

    # 构造预期的列部分字符串
    exp_cols_si = dedent(
        """\
    {} & {} & \\multicolumn{2}{r}{\\bfseries{Z}} & {Y} \\\\
    {} & {} & {a} & {b} & {\\bfseries{c}} \\\\
    """
    )
    exp_cols_no_si = """\
 &  & \\multicolumn{2}{r}{\\bfseries{Z}} & Y \\\\
 &  & a & b & \\bfseries{c} \\\\
"""
    # 断言预期的列部分是否在结果中（依据 columns 和 siunitx 参数）
    assert ((exp_cols_si if siunitx else exp_cols_no_si) in result) is columns


def test_repr_option(styler):
    # 断言样式对象的 HTML 表示中包含特定标签
    assert "<style" in styler._repr_html_()[:6]
    # 断言样式对象的 LaTeX 表示为空
    assert styler._repr_latex_() is None
    # 使用 option_context 测试设置 LaTeX 渲染模式下的样式表示
    with option_context("styler.render.repr", "latex"):
        # 断言 LaTeX 渲染模式下的样式对象是否包含表格开始标记
        assert "\\begin{tabular}" in styler._repr_latex_()[:15]
        # 断言 HTML 表示为空
        assert styler._repr_html_() is None


@pytest.mark.parametrize("option", ["hrules"])
def test_bool_options(styler, option):
    # 使用 option_context 测试设置样式的布尔选项
    with option_context(f"styler.latex.{option}", False):
        latex_false = styler.to_latex()
    with option_context(f"styler.latex.{option}", True):
        latex_true = styler.to_latex()
    # 断言关闭和开启选项下生成的 LaTeX 输出不同
    assert latex_false != latex_true  # options are reactive under to_latex(*no_args)


def test_siunitx_basic_headers(styler):
    # 断言使用 siunitx 选项时生成的 LaTeX 输出中包含特定格式的表头
    assert "{} & {A} & {B} & {C} \\\\" in styler.to_latex(siunitx=True)
    # 断言默认情况下的 LaTeX 输出中包含未格式化的表头
    assert " & A & B & C \\\\" in styler.to_latex()  # default siunitx=False


@pytest.mark.parametrize("axis", ["index", "columns"])
def test_css_convert_apply_index(styler, axis):
    # 测试应用 CSS 样式到索引或列标签
    styler.map_index(lambda x: "font-weight: bold;", axis=axis)
    # 对于 styler 对象的指定轴（行或列），迭代每个标签
    for label in getattr(styler, axis):
        # 断言确保带有 "\\bfseries" 样式的标签在导出为 LaTeX 格式时存在
        assert f"\\bfseries {label}" in styler.to_latex(convert_css=True)
# 测试隐藏 LaTeX 渲染中的索引
def test_hide_index_latex(styler):
    # GH 43637
    # 在 Styler 中隐藏第一行索引
    styler.hide([0], axis=0)
    # 将 Styler 转换为 LaTeX 格式的字符串
    result = styler.to_latex()
    # 预期的 LaTeX 字符串，使用 dedent 进行格式化
    expected = dedent(
        """\
        \\begin{tabular}{lrrl}
         & A & B & C \\\\
        1 & 1 & -1.22 & cd \\\\
        \\end{tabular}
        """
    )
    # 断言预期的结果与实际结果相同
    assert expected == result


# 测试隐藏多级索引列在 LaTeX 渲染中的对齐
def test_latex_hiding_index_columns_multiindex_alignment():
    # gh 43644
    # 创建多级索引对象 midx 和 cidx
    midx = MultiIndex.from_product(
        [["i0", "j0"], ["i1"], ["i2", "j2"]], names=["i-0", "i-1", "i-2"]
    )
    cidx = MultiIndex.from_product(
        [["c0"], ["c1", "d1"], ["c2", "d2"]], names=["c-0", "c-1", "c-2"]
    )
    # 创建 DataFrame 对象 df
    df = DataFrame(np.arange(16).reshape(4, 4), index=midx, columns=cidx)
    # 创建 Styler 对象 styler，设置 uuid_len 为 0
    styler = Styler(df, uuid_len=0)
    # 隐藏 axis=0 上 level=1 的索引
    styler.hide(level=1, axis=0)
    # 隐藏 axis=1 上 level=0 的索引
    styler.hide(level=0, axis=1)
    # 隐藏 axis=0 上 [("i0", "i1", "i2")] 的索引
    styler.hide([("i0", "i1", "i2")], axis=0)
    # 隐藏 axis=1 上 [("c0", "c1", "c2")] 的索引
    styler.hide([("c0", "c1", "c2")], axis=1)
    # 对 Styler 应用自定义函数，如果值为 5 设置颜色为红色
    styler.map(lambda x: "color:{red};" if x == 5 else "")
    # 对 Styler 的索引应用自定义函数，如果包含 'j' 设置颜色为蓝色
    styler.map_index(lambda x: "color:{blue};" if "j" in x else "")
    # 将 Styler 转换为 LaTeX 格式的字符串
    result = styler.to_latex()
    # 预期的 LaTeX 字符串，使用 dedent 进行格式化
    expected = dedent(
        """\
        \\begin{tabular}{llrrr}
         & c-1 & c1 & \\multicolumn{2}{r}{d1} \\\\
         & c-2 & d2 & c2 & d2 \\\\
        i-0 & i-2 &  &  &  \\\\
        i0 & \\color{blue} j2 & \\color{red} 5 & 6 & 7 \\\\
        \\multirow[c]{2}{*}{\\color{blue} j0} & i2 & 9 & 10 & 11 \\\\
        \\color{blue}  & \\color{blue} j2 & 13 & 14 & 15 \\\\
        \\end{tabular}
        """
    )
    # 断言预期的结果与实际结果相同
    assert result == expected


# 测试渲染链接
def test_rendered_links():
    # note the majority of testing is done in test_html.py: test_rendered_links
    # these test only the alternative latex format is functional
    # 创建包含链接文本的 DataFrame 对象 df
    df = DataFrame(["text www.domain.com text"])
    # 将 DataFrame 的样式设置为渲染为 LaTeX 的超链接格式
    result = df.style.format(hyperlinks="latex").to_latex()
    # 断言结果中包含特定的 LaTeX 格式化链接
    assert r"text \href{www.domain.com}{www.domain.com} text" in result


# 测试应用隐藏索引级别
def test_apply_index_hidden_levels():
    # gh 45156
    # 创建包含单元素的 DataFrame 对象，设置行列的多级索引名称
    styler = DataFrame(
        [[1]],
        index=MultiIndex.from_tuples([(0, 1)], names=["l0", "l1"]),
        columns=MultiIndex.from_tuples([(0, 1)], names=["c0", "c1"]),
    ).style
    # 隐藏 axis=1 上 level=1 的索引
    styler.hide(level=1)
    # 对 axis=1 上 level=0 的索引应用自定义函数，设置颜色为红色
    styler.map_index(lambda v: "color: red;", level=0, axis=1)
    # 将 Styler 转换为 LaTeX 格式的字符串，包含 CSS 转换
    result = styler.to_latex(convert_css=True)
    # 预期的 LaTeX 字符串，使用 dedent 进行格式化
    expected = dedent(
        """\
        \\begin{tabular}{lr}
        c0 & \\color{red} 0 \\\\
        c1 & 1 \\\\
        l0 &  \\\\
        0 & 1 \\\\
        \\end{tabular}
        """
    )
    # 断言预期的结果与实际结果相同
    assert result == expected


# 参数化测试 clines 的验证
@pytest.mark.parametrize("clines", ["bad", "index", "skip-last", "all", "data"])
def test_clines_validation(clines, styler):
    # 准备错误消息，说明 clines 值的错误
    msg = f"`clines` value of {clines} is invalid."
    # 断言调用 to_latex 方法时抛出 ValueError 异常，且异常信息匹配预期错误消息
    with pytest.raises(ValueError, match=msg):
        styler.to_latex(clines=clines)


# 参数化测试 clines 和 exp 的组合
@pytest.mark.parametrize(
    "clines, exp",
    [
        ("all;index", "\n\\cline{1-1}"),
        ("all;data", "\n\\cline{1-2}"),
        ("skip-last;index", ""),
        ("skip-last;data", ""),
        (None, ""),
    ],
)
# 参数化测试 LaTeX 输出环境
@pytest.mark.parametrize("env", ["table", "longtable"])
# 定义一个测试函数，用于测试指定条件下的 DataFrame 样式输出
def test_clines_index(clines, exp, env):
    # 创建一个包含数据的 DataFrame
    df = DataFrame([[1], [2], [3], [4]])
    # 将 DataFrame 样式转换为 LaTeX 格式的字符串，其中 clines 和 environment 参数控制样式
    result = df.style.to_latex(clines=clines, environment=env)
    # 期望的 LaTeX 字符串，包含了预期的表格样式和内容
    expected = f"""\
0 & 1 \\\\{exp}
1 & 2 \\\\{exp}
2 & 3 \\\\{exp}
3 & 4 \\\\{exp}
"""
    # 断言预期的字符串在实际结果中
    assert expected in result


# 使用 pytest 的参数化装饰器，定义多组参数来测试不同条件下的 DataFrame 样式输出
@pytest.mark.parametrize(
    "clines, expected",
    [
        (
            None,
            dedent(
                """\
            \\multirow[c]{2}{*}{A} & X & 1 \\\\
             & Y & 2 \\\\
            \\multirow[c]{2}{*}{B} & X & 3 \\\\
             & Y & 4 \\\\
            """
            ),
        ),
        (
            "skip-last;index",
            dedent(
                """\
            \\multirow[c]{2}{*}{A} & X & 1 \\\\
             & Y & 2 \\\\
            \\cline{1-2}
            \\multirow[c]{2}{*}{B} & X & 3 \\\\
             & Y & 4 \\\\
            \\cline{1-2}
            """
            ),
        ),
        (
            "skip-last;data",
            dedent(
                """\
            \\multirow[c]{2}{*}{A} & X & 1 \\\\
             & Y & 2 \\\\
            \\cline{1-3}
            \\multirow[c]{2}{*}{B} & X & 3 \\\\
             & Y & 4 \\\\
            \\cline{1-3}
            """
            ),
        ),
        (
            "all;index",
            dedent(
                """\
            \\multirow[c]{2}{*}{A} & X & 1 \\\\
            \\cline{2-2}
             & Y & 2 \\\\
            \\cline{1-2} \\cline{2-2}
            \\multirow[c]{2}{*}{B} & X & 3 \\\\
            \\cline{2-2}
             & Y & 4 \\\\
            \\cline{1-2} \\cline{2-2}
            """
            ),
        ),
        (
            "all;data",
            dedent(
                """\
            \\multirow[c]{2}{*}{A} & X & 1 \\\\
            \\cline{2-3}
             & Y & 2 \\\\
            \\cline{1-3} \\cline{2-3}
            \\multirow[c]{2}{*}{B} & X & 3 \\\\
            \\cline{2-3}
             & Y & 4 \\\\
            \\cline{1-3} \\cline{2-3}
            """
            ),
        ),
    ],
)
# 参数化测试函数，每组参数都测试一次 test_clines_index 函数
@pytest.mark.parametrize("env", ["table"])
def test_clines_multiindex(clines, expected, env):
    # 创建一个具有多级索引的 DataFrame
    midx = MultiIndex.from_product([["A", "-", "B"], [0], ["X", "Y"]])
    df = DataFrame([[1], [2], [99], [99], [3], [4]], index=midx)
    # 获取 DataFrame 样式对象
    styler = df.style
    # 隐藏特定的行和多级索引层级
    styler.hide([("-", 0, "X"), ("-", 0, "Y")])
    styler.hide(level=1)
    # 将 DataFrame 样式转换为 LaTeX 格式的字符串，其中 clines 和 environment 参数控制样式
    result = styler.to_latex(clines=clines, environment=env)
    # 断言预期的字符串在实际结果中
    assert expected in result


# 定义一个测试函数，测试指定条件下的 DataFrame 样式输出
def test_col_format_len(styler):
    # 测试长表格环境下的 DataFrame 样式输出
    result = styler.to_latex(environment="longtable", column_format="lrr{10cm}")
    # 期望的 LaTeX 字符串，包含了预期的表格样式和内容
    expected = r"\multicolumn{4}{r}{Continued on next page} \\"
    # 断言预期的字符串在实际结果中
    assert expected in result


# 定义一个测试函数，测试 DataFrame 样式的拼接输出
def test_concat(styler):
    # 将数据帧样式与其数据的聚合样式进行拼接，并将其转换为 LaTeX 格式的字符串
    result = styler.concat(styler.data.agg(["sum"]).style).to_latex()
    # 期望的 LaTeX 字符串，包含了预期的表格样式和内容
    expected = dedent(
        """\
    \\begin{tabular}{lrrl}
     & A & B & C \\\\
    0 & 0 & -0.61 & ab \\\\
    1 & 1 & -1.22 & cd \\\\
    """
    )
    # 断言预期的字符串在实际结果中
    assert expected in result
    sum & 1 & -1.830000 & abcd \\\\
    \\end{tabular}
    """
    )
    # 断言检查结果是否与期望值相等
    assert result == expected
def test_concat_recursion():
    # 测试隐藏行递归和应用样式

    # 创建第一个样式对象，使用DataFrame创建，隐藏第二行，突出显示最小值为红色
    styler1 = DataFrame([[1], [9]]).style.hide([1]).highlight_min(color="red")

    # 创建第二个样式对象，使用DataFrame创建，隐藏第一行，突出显示最小值为绿色
    styler2 = DataFrame([[9], [2]]).style.hide([0]).highlight_min(color="green")

    # 创建第三个样式对象，使用DataFrame创建，隐藏第二行，突出显示最小值为蓝色
    styler3 = DataFrame([[3], [9]]).style.hide([1]).highlight_min(color="blue")

    # 连接三个样式对象，生成Latex表格，并转换CSS
    result = styler1.concat(styler2.concat(styler3)).to_latex(convert_css=True)

    # 期望的Latex输出结果，使用dedent函数格式化多行字符串
    expected = dedent(
        """\
        \\begin{tabular}{lr}
         & 0 \\\\
        0 & {\\cellcolor{red}} 1 \\\\
        1 & {\\cellcolor{green}} 2 \\\\
        0 & {\\cellcolor{blue}} 3 \\\\
        \\end{tabular}
        """
    )

    # 断言结果与期望一致
    assert result == expected


def test_concat_chain():
    # 测试隐藏行递归和应用样式

    # 创建第一个样式对象，使用DataFrame创建，隐藏第二行，突出显示最小值为红色
    styler1 = DataFrame([[1], [9]]).style.hide([1]).highlight_min(color="red")

    # 创建第二个样式对象，使用DataFrame创建，隐藏第一行，突出显示最小值为绿色
    styler2 = DataFrame([[9], [2]]).style.hide([0]).highlight_min(color="green")

    # 创建第三个样式对象，使用DataFrame创建，隐藏第二行，突出显示最小值为蓝色
    styler3 = DataFrame([[3], [9]]).style.hide([1]).highlight_min(color="blue")

    # 连接三个样式对象，生成Latex表格，并转换CSS
    result = styler1.concat(styler2).concat(styler3).to_latex(convert_css=True)

    # 期望的Latex输出结果，使用dedent函数格式化多行字符串
    expected = dedent(
        """\
        \\begin{tabular}{lr}
         & 0 \\\\
        0 & {\\cellcolor{red}} 1 \\\\
        1 & {\\cellcolor{green}} 2 \\\\
        0 & {\\cellcolor{blue}} 3 \\\\
        \\end{tabular}
        """
    )

    # 断言结果与期望一致
    assert result == expected


@pytest.mark.parametrize(
    "columns, expected",
    [
        (
            None,
            dedent(
                """\
                \\begin{tabular}{l}
                \\end{tabular}
                """
            ),
        ),
        (
            ["a", "b", "c"],
            dedent(
                """\
                \\begin{tabular}{llll}
                 & a & b & c \\\\
                \\end{tabular}
                """
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "clines", [None, "all;data", "all;index", "skip-last;data", "skip-last;index"]
)
def test_empty_clines(columns, expected: str, clines: str):
    # GH 47203
    # 使用指定的列名创建DataFrame
    df = DataFrame(columns=columns)

    # 将DataFrame应用样式并生成Latex表格，根据参数选择是否包含空白行
    result = df.style.to_latex(clines=clines)

    # 断言结果与期望一致
    assert result == expected
```