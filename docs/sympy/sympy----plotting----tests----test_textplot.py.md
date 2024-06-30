# `D:\src\scipysrc\sympy\sympy\plotting\tests\test_textplot.py`

```
# 从 sympy.core.singleton 模块导入 S 单例对象
from sympy.core.singleton import S
# 从 sympy.core.symbol 模块导入 Symbol 符号对象
from sympy.core.symbol import Symbol
# 从 sympy.functions.elementary.exponential 模块导入 log 对数函数
from sympy.functions.elementary.exponential import log
# 从 sympy.functions.elementary.miscellaneous 模块导入 sqrt 平方根函数
from sympy.functions.elementary.miscellaneous import sqrt
# 从 sympy.functions.elementary.trigonometric 模块导入 sin 正弦函数
from sympy.functions.elementary.trigonometric import sin
# 从 sympy.plotting.textplot 模块导入 textplot_str 文本绘图函数
from sympy.plotting.textplot import textplot_str
# 从 sympy.utilities.exceptions 模块导入 ignore_warnings 函数

# 定义一个测试函数，用于验证坐标轴的对齐情况
def test_axes_alignment():
    # 创建一个符号变量 x
    x = Symbol('x')
    # 定义包含文本绘图数据的列表
    lines = [
        '      1 |                                                     ..',
        '        |                                                  ...  ',
        '        |                                                ..     ',
        '        |                                             ...       ',
        '        |                                          ...          ',
        '        |                                        ..             ',
        '        |                                     ...               ',
        '        |                                  ...                  ',
        '        |                                ..                     ',
        '        |                             ...                       ',
        '      0 |--------------------------...--------------------------',
        '        |                       ...                             ',
        '        |                     ..                                ',
        '        |                  ...                                  ',
        '        |               ...                                     ',
        '        |             ..                                        ',
        '        |          ...                                          ',
        '        |       ...                                             ',
        '        |     ..                                                ',
        '        |  ...                                                  ',
        '     -1 |_______________________________________________________',
        '         -1                         0                          1'
    ]
    # 使用 textplot_str 函数生成的文本绘图数据与预定义的 lines 列表进行断言比较
    assert lines == list(textplot_str(x, -1, 1))
    # 定义一个多行字符串列表，表示文本图形的行，每行包含一条数值线
    lines = [
        '      1 |                                                     ..',  # 第1行：包含数据点
        '        |                                                 ....  ',  # 第2行：包含更多数据点
        '        |                                              ...      ',  # 第3行：更少的数据点
        '        |                                           ...         ',  # 第4行：更少的数据点
        '        |                                       ....            ',  # 第5行：更多的数据点
        '        |                                    ...                ',  # 第6行：更少的数据点
        '        |                                 ...                   ',  # 第7行：更少的数据点
        '        |                             ....                      ',  # 第8行：更多的数据点
        '      0 |--------------------------...--------------------------',  # 第9行：基准线（零线）
        '        |                      ....                             ',  # 第10行：更多的数据点
        '        |                   ...                                 ',  # 第11行：更少的数据点
        '        |                ...                                    ',  # 第12行：更少的数据点
        '        |            ....                                       ',  # 第13行：更多的数据点
        '        |         ...                                           ',  # 第14行：更少的数据点
        '        |      ...                                              ',  # 第15行：更少的数据点
        '        |  ....                                                 ',  # 第16行：更多的数据点
        '     -1 |_______________________________________________________',  # 第17行：负一基准线
        '         -1                         0                          1'   # 底部刻度标记
    ]
    
    # 使用断言验证生成的图形行列表是否与 textplot_str 函数的输出一致
    assert lines == list(textplot_str(x, -1, 1, H=17))
# 定义一个测试函数，用于测试绘制函数在区间 [0, 1] 内的图形是否正确
def test_singularity():
    # 创建一个符号变量 x，用于表示代数表达式中的未知数
    x = Symbol('x')
    # 定义一个多行字符串列表，用于存储预期的文本图形的行
    lines = [
        '     54 | .                                                     ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '   27.5 |--.----------------------------------------------------',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |   .                                                   ',
        '        |    \\                                                  ',
        '        |     \\                                                 ',
        '        |      ..                                               ',
        '        |        ...                                            ',
        '        |           .............                               ',
        '      1 |_______________________________________________________',
        '         0                          0.5                        1'
    ]
    # 使用断言检查计算得到的文本图形行列表是否与预期的一致
    assert lines == list(textplot_str(1/x, 0, 1))
    # 定义一个列表 `lines`，包含了一系列字符串，表示某种图形的线条
    lines = [
        '      0 |                                                 ......',
        '        |                                         ........      ',
        '        |                                 ........              ',
        '        |                           ......                      ',
        '        |                      .....                            ',
        '        |                  ....                                 ',
        '        |               ...                                     ',
        '        |             ..                                        ',
        '        |          ...                                          ',
        '        |         /                                             ',
        '     -2 |-------..----------------------------------------------',
        '        |      /                                                ',
        '        |     /                                                 ',
        '        |    /                                                  ',
        '        |   .                                                   ',
        '        |                                                       ',
        '        |  .                                                    ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '     -4 |_______________________________________________________',
        '         0                          0.5                        1'
    ]
    # 使用 `ignore_warnings` 上下文管理器，忽略运行时的除零警告
    with ignore_warnings(RuntimeWarning):
        # 断言 `lines` 列表等于调用 `textplot_str` 函数返回的列表，该函数绘制一个对数函数在指定区间内的文本表示
        assert lines == list(textplot_str(log(x), 0, 1))
# 定义名为 test_sinc 的测试函数
def test_sinc():
    # 使用 SymPy 的 Symbol 函数创建一个符号变量 x
    x = Symbol('x')
    # 定义一个包含字符串形式的图表数据的列表
    lines = [
        '      1 |                          . .                          ',
        '        |                         .   .                         ',
        '        |                                                       ',
        '        |                        .     .                        ',
        '        |                                                       ',
        '        |                       .       .                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                      .         .                      ',
        '        |                                                       ',
        '    0.4 |-------------------------------------------------------',
        '        |                     .           .                     ',
        '        |                                                       ',
        '        |                    .             .                    ',
        '        |                                                       ',
        '        |    .....                                     .....    ',
        '        |  ..     \\         .               .         /     ..  ',
        '        | /        \\                                 /        \\ ',
        '        |/          \\      .                 .      /          \\',
        '        |            \\    /                   \\    /            ',
        '   -0.2 |_______________________________________________________',
        '         -10                        0                          10'
    ]
    # 使用 ignore_warnings 上下文管理器忽略 RuntimeWarning
    with ignore_warnings(RuntimeWarning):
        # 断言 lines 应该等于调用 textplot_str 函数得到的列表
        assert lines == list(textplot_str(sin(x)/x, -10, 10))


# 定义名为 test_imaginary 的测试函数，但尚未实现其内容
def test_imaginary():
    x = Symbol('x')
    lines = [
        '      1 |                                                     ..',
        '        |                                                   ..  ',
        '        |                                                ...    ',
        '        |                                              ..       ',
        '        |                                            ..         ',
        '        |                                          ..           ',
        '        |                                        ..             ',
        '        |                                      ..               ',
        '        |                                    ..                 ',
        '        |                                   /                   ',
        '    0.5 |----------------------------------/--------------------',
        '        |                                ..                     ',
        '        |                               /                       ',
        '        |                              .                        ',
        '        |                                                       ',
        '        |                             .                         ',
        '        |                            .                          ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '      0 |_______________________________________________________',
        '         -1                         0                          1'
    ]

这段代码定义了一个名为 `lines` 的列表，其中包含了一组字符串，这些字符串似乎代表了一个数学函数的图形表示，每个字符串对应一个特定的垂直位置，用点 `.` 表示。


    # RuntimeWarning: invalid value encountered in sqrt
    with ignore_warnings(RuntimeWarning):

通过 `ignore_warnings(RuntimeWarning)` 上下文管理器，忽略运行时警告，特别是在进行数学运算时可能会遇到的 `RuntimeWarning` 警告，如在求平方根时可能出现的非法值警告。


        assert list(textplot_str(sqrt(x), -1, 1)) == lines

进行断言检查，确保调用 `textplot_str` 函数并传入 `sqrt(x)` 的结果，以及区间 `[-1, 1]`，生成的文本图形与预定义的 `lines` 列表完全一致。
    # 定义一个包含文本绘图数据的列表，用于测试
    lines = [
        '      1 |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '      0 |-------------------------------------------------------',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '     -1 |_______________________________________________________',
        '         -1                         0                          1'
    ]
    # 断言对于给定的实数域中的虚数单位，生成的文本绘图字符串列表应与预期的 lines 相匹配
    assert list(textplot_str(S.ImaginaryUnit, -1, 1)) == lines
```