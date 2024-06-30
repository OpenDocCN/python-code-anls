# `D:\src\scipysrc\sympy\sympy\printing\tests\test_tableform.py`

```
from sympy.core.singleton import S
from sympy.printing.tableform import TableForm  # 导入TableForm类，用于创建表格形式的字符串
from sympy.printing.latex import latex  # 导入latex函数，用于生成Latex格式的字符串
from sympy.abc import x  # 导入符号变量x
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数sqrt
from sympy.functions.elementary.trigonometric import sin  # 导入正弦函数sin
from sympy.testing.pytest import raises  # 导入raises函数，用于测试异常

from textwrap import dedent  # 导入dedent函数，用于去除多余的空格和缩进


def test_TableForm():
    s = str(TableForm([["a", "b"], ["c", "d"], ["e", 0]],
        headings="automatic"))  # 生成包含自动标题的表格字符串s
    assert s == (
        '  | 1 2\n'
        '-------\n'
        '1 | a b\n'
        '2 | c d\n'
        '3 | e  '
    )

    s = str(TableForm([["a", "b"], ["c", "d"], ["e", 0]],
        headings="automatic", wipe_zeros=False))  # 生成包含自动标题且保留零的表格字符串s
    assert s == dedent('''\
          | 1 2
        -------
        1 | a b
        2 | c d
        3 | e 0''')

    s = str(TableForm([[x**2, "b"], ["c", x**2], ["e", "f"]],
            headings=("automatic", None)))  # 生成包含自动和无标题的表格字符串s
    assert s == (
        '1 | x**2 b   \n'
        '2 | c    x**2\n'
        '3 | e    f   '
    )

    s = str(TableForm([["a", "b"], ["c", "d"], ["e", "f"]],
            headings=(None, "automatic")))  # 生成包含无标题和自动标题的表格字符串s
    assert s == dedent('''\
        1 2
        ---
        a b
        c d
        e f''')

    s = str(TableForm([[5, 7], [4, 2], [10, 3]],
            headings=[["Group A", "Group B", "Group C"], ["y1", "y2"]]))  # 生成包含指定标题的表格字符串s
    assert s == (
        '        | y1 y2\n'
        '---------------\n'
        'Group A | 5  7 \n'
        'Group B | 4  2 \n'
        'Group C | 10 3 '
    )

    raises(
        ValueError,
        lambda:
        TableForm(
            [[5, 7], [4, 2], [10, 3]],
            headings=[["Group A", "Group B", "Group C"], ["y1", "y2"]],
            alignments="middle")  # 测试引发值错误异常，因为不支持的对齐选项
    )

    s = str(TableForm([[5, 7], [4, 2], [10, 3]],
            headings=[["Group A", "Group B", "Group C"], ["y1", "y2"]],
            alignments="right"))  # 生成包含右对齐的表格字符串s
    assert s == dedent('''\
                | y1 y2
        ---------------
        Group A |  5  7
        Group B |  4  2
        Group C | 10  3''')

    # other alignment permutations
    d = [[1, 100], [100, 1]]
    s = TableForm(d, headings=(('xxx', 'x'), None), alignments='l')  # 生成指定左对齐的表格字符串s
    assert str(s) == (
        'xxx | 1   100\n'
        '  x | 100 1  '
    )

    s = TableForm(d, headings=(('xxx', 'x'), None), alignments='lr')  # 生成指定左右对齐的表格字符串s
    assert str(s) == dedent('''\
    xxx | 1   100
      x | 100   1''')

    s = TableForm(d, headings=(('xxx', 'x'), None), alignments='clr')  # 生成指定居中和左右对齐的表格字符串s
    assert str(s) == dedent('''\
    xxx | 1   100
     x  | 100   1''')

    s = TableForm(d, headings=(('xxx', 'x'), None))  # 生成默认对齐的表格字符串s
    assert str(s) == (
        'xxx | 1   100\n'
        '  x | 100 1  '
    )

    raises(ValueError, lambda: TableForm(d, alignments='clr'))  # 测试引发值错误异常，因为不支持的对齐选项

    # pad
    s = str(TableForm([[None, "-", 2], [1]], pad='?'))  # 生成指定填充字符的表格字符串s
    assert s == dedent('''\
        ? - 2
        1 ? ?''')


def test_TableForm_latex():
    s = latex(TableForm([[0, x**3], ["c", S.One/4], [sqrt(x), sin(x**2)]],
            wipe_zeros=True, headings=("automatic", "automatic")))  # 生成Latex格式的表格字符串s
    assert s == (
        '\\begin{tabular}{r l l}\n'  # 检查字符串 s 是否与给定的 LaTeX 格式表格字符串匹配
        ' & 1 & 2 \\\\\n'            # 第一行表头
        '\\hline\n'                  # 表头与表格内容的分隔线
        '1 &   & $x^{3}$ \\\\\n'     # 第一行数据
        '2 & $c$ & $\\frac{1}{4}$ \\\\\n'  # 第二行数据
        '3 & $\\sqrt{x}$ & $\\sin{\\left(x^{2} \\right)}$ \\\\\n'  # 第三行数据
        '\\end{tabular}'
    )
    s = latex(TableForm([[0, x**3], ["c", S.One/4], [sqrt(x), sin(x**2)]],
            wipe_zeros=True, headings=("automatic", "automatic"), alignments='l'))
    assert s == (
        '\\begin{tabular}{r l l}\n'  # 检查生成的 LaTeX 表格字符串 s 是否符合预期格式
        ' & 1 & 2 \\\\\n'            # 第一行表头
        '\\hline\n'                  # 表头与表格内容的分隔线
        '1 &   & $x^{3}$ \\\\\n'     # 第一行数据
        '2 & $c$ & $\\frac{1}{4}$ \\\\\n'  # 第二行数据
        '3 & $\\sqrt{x}$ & $\\sin{\\left(x^{2} \\right)}$ \\\\\n'  # 第三行数据
        '\\end{tabular}'
    )
    s = latex(TableForm([[0, x**3], ["c", S.One/4], [sqrt(x), sin(x**2)]],
            wipe_zeros=True, headings=("automatic", "automatic"), alignments='l'*3))
    assert s == (
        '\\begin{tabular}{l l l}\n'  # 检查生成的 LaTeX 表格字符串 s 是否符合预期格式
        ' & 1 & 2 \\\\\n'            # 第一行表头
        '\\hline\n'                  # 表头与表格内容的分隔线
        '1 &   & $x^{3}$ \\\\\n'     # 第一行数据
        '2 & $c$ & $\\frac{1}{4}$ \\\\\n'  # 第二行数据
        '3 & $\\sqrt{x}$ & $\\sin{\\left(x^{2} \\right)}$ \\\\\n'  # 第三行数据
        '\\end{tabular}'
    )
    s = latex(TableForm([["a", x**3], ["c", S.One/4], [sqrt(x), sin(x**2)]],
            headings=("automatic", "automatic")))
    assert s == (
        '\\begin{tabular}{r l l}\n'  # 检查生成的 LaTeX 表格字符串 s 是否符合预期格式
        ' & 1 & 2 \\\\\n'            # 第一行表头
        '\\hline\n'                  # 表头与表格内容的分隔线
        '1 & $a$ & $x^{3}$ \\\\\n'   # 第一行数据
        '2 & $c$ & $\\frac{1}{4}$ \\\\\n'  # 第二行数据
        '3 & $\\sqrt{x}$ & $\\sin{\\left(x^{2} \\right)}$ \\\\\n'  # 第三行数据
        '\\end{tabular}'
    )
    s = latex(TableForm([["a", x**3], ["c", S.One/4], [sqrt(x), sin(x**2)]],
            formats=['(%s)', None], headings=("automatic", "automatic")))
    assert s == (
        '\\begin{tabular}{r l l}\n'  # 检查生成的 LaTeX 表格字符串 s 是否符合预期格式
        ' & 1 & 2 \\\\\n'            # 第一行表头
        '\\hline\n'                  # 表头与表格内容的分隔线
        '1 & (a) & $x^{3}$ \\\\\n'   # 第一行数据，带格式化
        '2 & (c) & $\\frac{1}{4}$ \\\\\n'  # 第二行数据，带格式化
        '3 & (sqrt(x)) & $\\sin{\\left(x^{2} \\right)}$ \\\\\n'  # 第三行数据，带格式化
        '\\end{tabular}'
    )

    def neg_in_paren(x, i, j):
        if i % 2:
            return ('(%s)' if x < 0 else '%s') % x  # 根据条件返回带括号或不带括号的字符串表示
        else:
            pass  # 未指定情况下使用默认打印格式
    s = latex(TableForm([[-1, 2], [-3, 4]],
            formats=[neg_in_paren]*2, headings=("automatic", "automatic")))
    assert s == (
        '\\begin{tabular}{r l l}\n'  # 检查生成的 LaTeX 表格字符串 s 是否符合预期格式
        ' & 1 & 2 \\\\\n'            # 第一行表头
        '\\hline\n'                  # 表头与表格内容的分隔线
        '1 & -1 & 2 \\\\\n'          # 第一行数据，负数带括号
        '2 & (-3) & 4 \\\\\n'        # 第二行数据，负数带括号
        '\\end{tabular}'
    )
    s = latex(TableForm([["a", x**3], ["c", S.One/4], [sqrt(x), sin(x**2)]]))
    assert s == (
        '\\begin{tabular}{l l}\n'    # 检查生成的 LaTeX 表格字符串 s 是否符合预期格式
        '$a$ & $x^{3}$ \\\\\n'       # 第一行数据
        '$c$ & $\\frac{1}{4}$ \\\\\n'  # 第二行数据
        '$\\sqrt{x}$ & $\\sin{\\left(x^{2} \\right)}$ \\\\\n'  # 第三行数据
        '\\end{tabular}'
    )
```