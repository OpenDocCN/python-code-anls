# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_mathtext.py`

```
# 从未来导入注释，允许在函数签名中使用类型提示的字符串表示法
from __future__ import annotations

# 导入必要的库
import io  # 用于处理字节流
from pathlib import Path  # 提供了处理路径的类
import platform  # 获取平台信息
import re  # 正则表达式模块，用于字符串匹配和处理
import shlex  # 用于解析命令行参数
from xml.etree import ElementTree as ET  # XML 解析库
from typing import Any  # 引入类型提示支持

import numpy as np  # 数组和数值计算
from packaging.version import parse as parse_version  # 解析版本号字符串的工具
import pyparsing  # 强大的解析器生成器
import pytest  # 测试框架

# 导入 matplotlib 库并从中选择性地导入特定的模块和函数
import matplotlib as mpl
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib.pyplot as plt  # 用于绘图
from matplotlib import mathtext, _mathtext  # 数学文本渲染相关的模块和私有模块

# 解析 pyparsing 版本信息
pyparsing_version = parse_version(pyparsing.__version__)

# 数学表达式测试集合，包括一系列 LaTeX 格式的数学表达式和一些注释
math_tests = [
    r'$a+b+\dot s+\dot{s}+\ldots$',  # 示例数学表达式
    r'$x\hspace{-0.2}\doteq\hspace{-0.2}y$',  # 示例数学表达式
    r'\$100.00 $\alpha \_$',  # 示例数学表达式
    r'$\frac{\$100.00}{y}$',  # 示例数学表达式
    r'$x   y$',  # 示例数学表达式
    r'$x+y\ x=y\ x<y\ x:y\ x,y\ x@y$',  # 示例数学表达式
    r'$100\%y\ x*y\ x/y x\$y$',  # 示例数学表达式
    r'$x\leftarrow y\ x\forall y\ x-y$',  # 示例数学表达式
    r'$x \sf x \bf x {\cal X} \rm x$',  # 示例数学表达式
    r'$x\ x\,x\;x\quad x\qquad x\!x\hspace{ 0.5 }y$',  # 示例数学表达式
    r'$\{ \rm braces \}$',  # 示例数学表达式
    r'$\left[\left\lfloor\frac{5}{\frac{\left(3\right)}{4}} y\right)\right]$',  # 示例数学表达式
    r'$\left(x\right)$',  # 示例数学表达式
    r'$\sin(x)$',  # 示例数学表达式
    r'$x_2$',  # 示例数学表达式
    r'$x^2$',  # 示例数学表达式
    r'$x^2_y$',  # 示例数学表达式
    r'$x_y^2$',  # 示例数学表达式
    (r'$\sum _{\genfrac{}{}{0}{}{0\leq i\leq m}{0<j<n}}f\left(i,j\right)'
     r'\mathcal{R}\prod_{i=\alpha_{i+1}}^\infty a_i \sin(2 \pi f x_i)'
     r"\sqrt[2]{\prod^\frac{x}{2\pi^2}_\infty}$"),  # 示例数学表达式
    r'$x = \frac{x+\frac{5}{2}}{\frac{y+3}{8}}$',  # 示例数学表达式
    r'$dz/dt = \gamma x^2 + {\rm sin}(2\pi y+\phi)$',  # 示例数学表达式
    r'Foo: $\alpha_{i+1}^j = {\rm sin}(2\pi f_j t_i) e^{-5 t_i/\tau}$',  # 示例数学表达式
    None,  # 占位符，没有数学表达式
    r'Variable $i$ is good',  # 示例数学表达式
    r'$\Delta_i^j$',  # 示例数学表达式
    r'$\Delta^j_{i+1}$',  # 示例数学表达式
    r'$\ddot{o}\acute{e}\grave{e}\hat{O}\breve{\imath}\tilde{n}\vec{q}$',  # 示例数学表达式
    r"$\arccos((x^i))$",  # 示例数学表达式
    r"$\gamma = \frac{x=\frac{6}{8}}{y} \delta$",  # 示例数学表达式
    r'$\limsup_{x\to\infty}$',  # 示例数学表达式
    None,  # 占位符，没有数学表达式
    r"$f'\quad f'''(x)\quad ''/\mathrm{yr}$",  # 示例数学表达式
    r'$\frac{x_2888}{y}$',  # 示例数学表达式
    r"$\sqrt[3]{\frac{X_2}{Y}}=5$",  # 示例数学表达式
    None,  # 占位符，没有数学表达式
    r"$\sqrt[3]{x}=5$",  # 示例数学表达式
    r'$\frac{X}{\frac{X}{Y}}$',  # 示例数学表达式
    r"$W^{3\beta}_{\delta_1 \rho_1 \sigma_2} = U^{3\beta}_{\delta_1 \rho_1} + \frac{1}{8 \pi 2} \int^{\alpha_2}_{\alpha_2} d \alpha^\prime_2 \left[\frac{ U^{2\beta}_{\delta_1 \rho_1} - \alpha^\prime_2U^{1\beta}_{\rho_1 \sigma_2} }{U^{0\beta}_{\rho_1 \sigma_2}}\right]$',  # 示例数学表达式
    r'$\mathcal{H} = \int d \tau \left(\epsilon E^2 + \mu H^2\right)$',  # 示例数学表达式
    r'$\widehat{abc}\widetilde{def}$',  # 示例数学表达式
    '$\\Gamma \\Delta \\Theta \\Lambda \\Xi \\Pi \\Sigma \\Upsilon \\Phi \\Psi \\Omega$',  # 示例数学表达式
    '$\\alpha \\beta \\gamma \\delta \\epsilon \\zeta \\eta \\theta \\iota \\lambda \\mu \\nu \\xi \\pi \\kappa \\rho \\sigma \\tau \\upsilon \\phi \\chi \\psi$',  # 示例数学表达式
    r'${x}^{2}{y}^{2}$',  # 示例数学表达式
    r'${}_{2}F_{3}$',  # 示例数学表达式
    r'$\frac{x+{y}^{2}}{k+1}$',  # 示例数学表达式
    r'$x+{y}^{\frac{2}{k+1}}$',  # 示例数学表达式
    r'$\frac{a}{b/2}$',  # 示例数学表达式
    r'${a}_{0}+\frac{1}{{a}_{1}+\frac{1}{{a}_{2}+\frac{1}{{a}_{3}+\frac{1}{{a}_{4}}}}}$',  # 示例数学表达式
]
    # 数学公式字符串，用于Matplotlib和TeX中的数学表达式
    r'${a}_{0}+\frac{1}{{a}_{1}+\frac{1}{{a}_{2}+\frac{1}{{a}_{3}+\frac{1}{{a}_{4}}}}}$',
    # 表示组合数，用于描述从n中选择k/2个项目的方法数
    r'$\binom{n}{k/2}$',
    # 多项式表达式，包括指数和分数项
    r'$\binom{p}{2}{x}^{2}{y}^{p-2}-\frac{1}{1-x}\frac{1}{1-{x}^{2}}$',
    # 指数表达式
    r'${x}^{2y}$',
    # 三重求和表达式
    r'$\sum _{i=1}^{p}\sum _{j=1}^{q}\sum _{k=1}^{r}{a}_{ij}{b}_{jk}{c}_{ki}$',
    # 嵌套根号表达式
    r'$\sqrt{1+\sqrt{1+\sqrt{1+\sqrt{1+\sqrt{1+\sqrt{1+\sqrt{1+x}}}}}}}$',
    # 偏微分方程表达式
    r'$\left(\frac{{\partial }^{2}}{\partial {x}^{2}}+\frac{{\partial }^{2}}{\partial {y}^{2}}\right){|\varphi \left(x+iy\right)|}^{2}=0$',
    # 指数塔表达式
    r'${2}^{{2}^{{2}^{x}}}$',
    # 定积分表达式
    r'${\int }_{1}^{x}\frac{\mathrm{dt}}{t}$',
    # 双重积分表达式
    r'$\int {\int }_{D}\mathrm{dx} \mathrm{dy}$',
    # 数学表达式不受支持的情况
    r'${y}_{{x}^{2}}$',
    # 数学表达式不受支持的情况
    r'${x}_{92}^{31415}+\pi $',
    # 数学表达式不受支持的情况
    r'${x}_{{y}_{b}^{a}}^{{z}_{c}^{d}}$',
    # 数学表达式不受支持的情况
    r'${y}_{3}^{\prime \prime \prime }$',
    # 数学表达式不受支持的情况
    r"$\left( \xi \left( 1 - \xi \right) \right)$",
    # 数学表达式不受支持的情况
    r"$\left(2 \, a=b\right)$",
    # 数学表达式不受支持的情况
    r"$? ! &$",
    # None占位符
    None,
    # None占位符
    None,
    # 多个数学表达式的绝对值和范数
    r"$\left\Vert \frac{a}{b} \right\Vert \left\vert \frac{a}{b} \right\vert \left\| \frac{a}{b}\right\| \left| \frac{a}{b} \right| \Vert a \Vert \vert b \vert \| a \| | b |$",
    # 特殊符号和字母的组合
    r'$\mathring{A}  \AA$',
    # 渲染数学公式：$M \, M \thinspace M \/ M \> M \: M \; M \ M \enspace M \quad M \qquad M \! M$
    r'$M \, M \thinspace M \/ M \> M \: M \; M \ M \enspace M \quad M \qquad M \! M$',

    # 渲染数学符号：$\Cap$ $\Cup$ $\leftharpoonup$ $\barwedge$ $\rightharpoonup$
    r'$\Cap$ $\Cup$ $\leftharpoonup$ $\barwedge$ $\rightharpoonup$',

    # 渲染数学符号和空间调整：$\hspace{-0.2}\dotplus\hspace{-0.2}$ $\hspace{-0.2}\doteq\hspace{-0.2}$ $\hspace{-0.2}\doteqdot\hspace{-0.2}$ $\ddots$
    r'$\hspace{-0.2}\dotplus\hspace{-0.2}$ $\hspace{-0.2}\doteq\hspace{-0.2}$ $\hspace{-0.2}\doteqdot\hspace{-0.2}$ $\ddots$',

    # 数学表达式和文字混合：$xyz^kx_kx^py^{p-2} d_i^jb_jc_kd x^j_i E^0 E^0_u$
    r'$xyz^kx_kx^py^{p-2} d_i^jb_jc_kd x^j_i E^0 E^0_u$',  # github issue #4873

    # 数学表达式和文字混合，使用大括号分组：${xyz}^k{x}_{k}{x}^{p}{y}^{p-2} {d}_{i}^{j}{b}_{j}{c}_{k}{d} {x}^{j}_{i}{E}^{0}{E}^0_u$
    r'${xyz}^k{x}_{k}{x}^{p}{y}^{p-2} {d}_{i}^{j}{b}_{j}{c}_{k}{d} {x}^{j}_{i}{E}^{0}{E}^0_u$',

    # 数学表达式包含多个积分符号和限定符号：${\int}_x^x x\oint_x^x x\int_{X}^{X}x\int_x x \int^x x \int_{x} x\int^{x}{\int}_{x} x{\int}^{x}_{x}x$
    r'${\int}_x^x x\oint_x^x x\int_{X}^{X}x\int_x x \int^x x \int_{x} x\int^{x}{\int}_{x} x{\int}^{x}_{x}x$',

    # 包含上标和下标的文字：testing$^{123}$
    r'testing$^{123}$',

    # 空值，None
    None,

    # 包含数学表达式和符号的混合：$6-2$; $-2$; $ -2$; ${-2}$; ${  -2}$; $20^{+3}_{-2}$
    r'$6-2$; $-2$; $ -2$; ${-2}$; ${  -2}$; $20^{+3}_{-2}$',

    # 包含上划线和分数的混合：$\overline{\omega}^x \frac{1}{2}_0^x$
    r'$\overline{\omega}^x \frac{1}{2}_0^x$',  # github issue #5444

    # 数字千位分隔符：$,$ $.$ $1{,}234{, }567{ , }890$ and $1,234,567,890$
    r'$,$ $.$ $1{,}234{, }567{ , }890$ and $1,234,567,890$',  # github issue 5799

    # 包含括号和上下标的数学表达式：$\left(X\right)_{a}^{b}$
    r'$\left(X\right)_{a}^{b}$',  # github issue 7615

    # 包含错误使用的数学表达式：$\dfrac{\$100.00}{y}$
    r'$\dfrac{\$100.00}{y}$',  # github issue #1888

    # 包含赋值表达式：$a=-b-c$
    r'$a=-b-c$'  # github issue #28180
# 'svgastext' tests switch svg output to embed text as text (rather than as
# paths).
svgastext_math_tests = [
    r'$-$-',  # List of test strings for SVG output with embedded text
]

# 'lightweight' tests test only a single fontset (dejavusans, which is the
# default) and only png outputs, in order to minimize the size of baseline
# images.
lightweight_math_tests = [
    r'$\sqrt[ab]{123}$',  # Test for square root with custom notation
    r'$x \overset{f}{\rightarrow} \overset{f}{x} \underset{xx}{ff} \overset{xx}{ff} \underset{f}{x} \underset{f}{\leftarrow} x$',  # Test for various math notations
    r'$\sum x\quad\sum^nx\quad\sum_nx\quad\sum_n^nx\quad\prod x\quad\prod^nx\quad\prod_nx\quad\prod_n^nx$',  # Test for summation and product notations
    r'$1.$ $2.$ $19680801.$ $a.$ $b.$ $mpl.$',  # Test for alphanumeric symbols
    r'$\text{text}_{\text{sub}}^{\text{sup}} + \text{\$foo\$} + \frac{\text{num}}{\mathbf{\text{den}}}\text{with space, curly brackets \{\}, and dash -}$',  # Test for text with subscripts, superscripts, and special symbols
    r'$\boldsymbol{abcde} \boldsymbol{+} \boldsymbol{\Gamma + \Omega} \boldsymbol{01234} \boldsymbol{\alpha * \beta}$',  # Test for bold symbols
    r'$\left\lbrace\frac{\left\lbrack A^b_c\right\rbrace}{\left\leftbrace D^e_f \right\rbrack}\right\rightbrace\ \left\leftparen\max_{x} \left\lgroup \frac{A}{B}\right\rgroup \right\rightparen$',  # Test for braces and brackets with nested expressions
    r'$\left( a\middle. b \right)$ $\left( \frac{a}{b} \middle\vert x_i \in P^S \right)$ $\left[ 1 - \middle| a\middle| + \left( x  - \left\lfloor \dfrac{a}{b}\right\rfloor \right)  \right]$',  # Test for parentheses and vertical lines with mathematical expressions
    r'$\sum_{\substack{k = 1\\ k \neq \lfloor n/2\rfloor}}^{n}P(i,j) \sum_{\substack{i \neq 0\\ -1 \leq i \leq 3\\ 1 \leq j \leq 5}} F^i(x,y) \sum_{\substack{\left \lfloor \frac{n}{2} \right\rfloor}} F(n)$',  # Test for summation with substacked conditions
]

digits = "0123456789"
uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
lowercase = "abcdefghijklmnopqrstuvwxyz"
uppergreek = ("\\Gamma \\Delta \\Theta \\Lambda \\Xi \\Pi \\Sigma \\Upsilon \\Phi \\Psi "
              "\\Omega")
lowergreek = ("\\alpha \\beta \\gamma \\delta \\epsilon \\zeta \\eta \\theta \\iota "
              "\\lambda \\mu \\nu \\xi \\pi \\kappa \\rho \\sigma \\tau \\upsilon "
              "\\phi \\chi \\psi")

all = [digits, uppercase, lowercase, uppergreek, lowergreek]

# Use stubs to reserve space if tests are removed
# stub should be of the form (None, N) where N is the number of strings that
# used to be tested
# Add new tests at the end.
font_test_specs: list[tuple[None | list[str], Any]] = [
    ([], all),  # Test all default font sets
    (['mathrm'], all),  # Test with 'mathrm' font variant
    (['mathbf'], all),  # Test with 'mathbf' font variant
    (['mathit'], all),  # Test with 'mathit' font variant
    (['mathtt'], [digits, uppercase, lowercase]),  # Test with 'mathtt' font variant
    (None, 3),  # Placeholder for stub
    (None, 3),  # Placeholder for stub
    (None, 3),  # Placeholder for stub
    (['mathbb'], [digits, uppercase, lowercase,
                  r'\Gamma \Pi \Sigma \gamma \pi']),  # Test with 'mathbb' font variant
    (['mathrm', 'mathbb'], [digits, uppercase, lowercase,
                            r'\Gamma \Pi \Sigma \gamma \pi']),  # Test with 'mathrm' and 'mathbb' font variants
    (['mathbf', 'mathbb'], [digits, uppercase, lowercase,
                            r'\Gamma \Pi \Sigma \gamma \pi']),  # Test with 'mathbf' and 'mathbb' font variants
    (['mathcal'], [uppercase]),  # Test with 'mathcal' font variant
    (['mathfrak'], [uppercase, lowercase]),  # Test with 'mathfrak' font variant
    (['mathbf', 'mathfrak'], [uppercase, lowercase]),  # Test with 'mathbf' and 'mathfrak' font variants
    (['mathscr'], [uppercase, lowercase]),  # Test with 'mathscr' font variant
]
    # 定义一个包含多个元组的列表，每个元组包含两部分：
    #   - 第一部分是一个字符串列表，表示数学字体的样式，如'mathsf', 'mathrm', 'mathbf'等
    #   - 第二部分是一个包含数字、大写字母和小写字母的列表，表示该字体样式可用于这些字符集合
    ([
        'mathsf'], [digits, uppercase, lowercase]),
        (['mathrm', 'mathsf'], [digits, uppercase, lowercase]),
        (['mathbf', 'mathsf'], [digits, uppercase, lowercase]),
        (['mathbfit'], all),
        ]
# 初始化一个空列表 `font_tests`，用于存储测试字体相关的数据，每个元素可以是 None 或字符串
font_tests: list[None | str] = []

# 遍历 `font_test_specs` 列表中的每个元组 (fonts, chars)
for fonts, chars in font_test_specs:
    # 如果 fonts 为 None，则将 None 添加到 font_tests 中 chars 次
    if fonts is None:
        font_tests.extend([None] * chars)
    else:
        # 否则，构建一个字符串模板 `wrapper`，用于将测试字符格式化为特定字体的格式
        wrapper = ''.join([
            ' '.join(fonts),                # 将字体名称用空格分隔并连接成字符串
            ' $',                           # 添加起始标记 $
            *(r'\%s{' % font for font in fonts),  # 对每个字体生成格式化字符串 \font{
            '%s',                           # 插入字符集
            *('}' for font in fonts),       # 添加闭合标记 }
            '$',                            # 添加结束标记 $
        ])
        # 遍历 chars 中的每个字符集合，将格式化后的字符串添加到 font_tests 中
        for set in chars:
            font_tests.append(wrapper % set)


# 下面是各种测试函数和参数化装饰器，负责渲染数学表达式并生成对应的图片进行比较
# 每个函数都对应不同的测试用例和字体集合，使用 Matplotlib 进行绘图和比较
# 使用 image_comparison 装饰器定义一个测试函数，用于比较图像（PNG 格式），基准图像为空，容差根据平台决定
@image_comparison(baseline_images=None, extensions=['png'],
                  tol=0.011 if platform.machine() in ('ppc64le', 's390x') else 0)
def test_mathfont_rendering(baseline_images, fontset, index, text):
    # 设置 mathtext 的字体集
    mpl.rcParams['mathtext.fontset'] = fontset
    # 创建一个 5.25x0.75 大小的图像对象
    fig = plt.figure(figsize=(5.25, 0.75))
    # 在图像中心添加文本，文本内容由参数 text 提供
    fig.text(0.5, 0.5, text,
             horizontalalignment='center', verticalalignment='center')


# 使用 check_figures_equal 装饰器定义一个测试函数，用于检查两个图像是否相等（PNG 格式）
@check_figures_equal(extensions=["png"])
def test_short_long_accents(fig_test, fig_ref):
    # 获取 mathtext 解析器中的重音映射表
    acc_map = _mathtext.Parser._accent_map
    # 找出所有长度为 1 的短重音符号
    short_accs = [s for s in acc_map if len(s) == 1]
    corresponding_long_accs = []
    # 对于每个短重音符号，找到相应的长重音符号
    for s in short_accs:
        l, = [l for l in acc_map if len(l) > 1 and acc_map[l] == acc_map[s]]
        corresponding_long_accs.append(l)
    # 在测试图像中添加短重音符号组合的数学表达式
    fig_test.text(0, .5, "$" + "".join(rf"\{s}a" for s in short_accs) + "$")
    # 在参考图像中添加相应的长重音符号组合的数学表达式
    fig_ref.text(
        0, .5, "$" + "".join(fr"\{l} a" for l in corresponding_long_accs) + "$")


# 定义一个测试函数，用于验证字体信息的获取
def test_fontinfo():
    # 查找特定字体（DejaVu Sans）的文件路径
    fontpath = mpl.font_manager.findfont("DejaVu Sans")
    # 使用 FT2Font 类加载字体文件
    font = mpl.ft2font.FT2Font(fontpath)
    # 获取字体文件中 "head" 表的数据
    table = font.get_sfnt_table("head")
    # 断言 "head" 表数据不为空
    assert table is not None
    # 断言 "head" 表的版本号为 (1, 0)


# 为了了解关于这个 xfail 的更多背景信息，请参见 gh-26152
# 使用 pytest.mark.xfail 标记测试为预期失败的情况，如果 pyparsing 版本为 (3, 1, 0)，则出现错误消息不正确
@pytest.mark.xfail(pyparsing_version.release == (3, 1, 0),
                   reason="Error messages are incorrect for this version")
# 使用 pytest.mark.parametrize 标记测试参数化，以便多次运行不同的输入组合
@pytest.mark.parametrize(
    'math, msg',
    [
        # 第一个元组：描述和预期
        (r'$\hspace{}$', r'Expected \hspace{space}'),
        # 第二个元组：描述和预期
        (r'$\hspace{foo}$', r'Expected \hspace{space}'),
        # 第三个元组：描述和预期
        (r'$\sinx$', r'Unknown symbol: \sinx'),
        # 第四个元组：描述和预期
        (r'$\dotx$', r'Unknown symbol: \dotx'),
        # 第五个元组：描述和预期
        (r'$\frac$', r'Expected \frac{num}{den}'),
        # 第六个元组：描述和预期
        (r'$\frac{}{}$', r'Expected \frac{num}{den}'),
        # 第七个元组：描述和预期
        (r'$\binom$', r'Expected \binom{num}{den}'),
        # 第八个元组：描述和预期
        (r'$\binom{}{}$', r'Expected \binom{num}{den}'),
        # 第九个元组：描述和预期
        (r'$\genfrac$', r'Expected \genfrac{ldelim}{rdelim}{rulesize}{style}{num}{den}'),
        # 第十个元组：描述和预期
        (r'$\genfrac{}{}{}{}{}{}$', r'Expected \genfrac{ldelim}{rdelim}{rulesize}{style}{num}{den}'),
        # 第十一个元组：描述和预期
        (r'$\sqrt$', r'Expected \sqrt{value}'),
        # 第十二个元组：描述和预期
        (r'$\sqrt f$', r'Expected \sqrt{value}'),
        # 第十三个元组：描述和预期
        (r'$\overline$', r'Expected \overline{body}'),
        # 第十四个元组：描述和预期
        (r'$\overline{}$', r'Expected \overline{body}'),
        # 第十五个元组：描述和预期
        (r'$\leftF$', r'Expected a delimiter'),
        # 第十六个元组：描述和预期
        (r'$\rightF$', r'Unknown symbol: \rightF'),
        # 第十七个元组：描述和预期
        (r'$\left(\right$', r'Expected a delimiter'),
        # 第十八个元组：描述和预期
        (r'$\left($', re.compile(r'Expected ("|\'\\)\\right["\']')),
        # 第十九个元组：描述和预期
        (r'$\dfrac$', r'Expected \dfrac{num}{den}'),
        # 第二十个元组：描述和预期
        (r'$\dfrac{}{}$', r'Expected \dfrac{num}{den}'),
        # 第二十一个元组：描述和预期
        (r'$\overset$', r'Expected \overset{annotation}{body}'),
        # 第二十二个元组：描述和预期
        (r'$\underset$', r'Expected \underset{annotation}{body}'),
        # 第二十三个元组：描述和预期
        (r'$\foo$', r'Unknown symbol: \foo'),
        # 第二十四个元组：描述和预期
        (r'$a^2^2$', r'Double superscript'),
        # 第二十五个元组：描述和预期
        (r'$a_2_2$', r'Double subscript'),
        # 第二十六个元组：描述和预期
        (r'$a^2_a^2$', r'Double superscript'),
        # 第二十七个元组：描述和预期
        (r'$a = {b$', r"Expected '}'"),
    ],
    
    # ids 列表的注释
    ids=[
        # 第一个 ID：描述
        'hspace without value',
        # 第二个 ID：描述
        'hspace with invalid value',
        # 第三个 ID：描述
        'function without space',
        # 第四个 ID：描述
        'accent without space',
        # 第五个 ID：描述
        'frac without parameters',
        # 第六个 ID：描述
        'frac with empty parameters',
        # 第七个 ID：描述
        'binom without parameters',
        # 第八个 ID：描述
        'binom with empty parameters',
        # 第九个 ID：描述
        'genfrac without parameters',
        # 第十个 ID：描述
        'genfrac with empty parameters',
        # 第十一个 ID：描述
        'sqrt without parameters',
        # 第十二个 ID：描述
        'sqrt with invalid value',
        # 第十三个 ID：描述
        'overline without parameters',
        # 第十四个 ID：描述
        'overline with empty parameter',
        # 第十五个 ID：描述
        'left with invalid delimiter',
        # 第十六个 ID：描述
        'right with invalid delimiter',
        # 第十七个 ID：描述
        'unclosed parentheses with sizing',
        # 第十八个 ID：描述
        'unclosed parentheses without sizing',
        # 第十九个 ID：描述
        'dfrac without parameters',
        # 第二十个 ID：描述
        'dfrac with empty parameters',
        # 第二十一个 ID：描述
        'overset without parameters',
        # 第二十二个 ID：描述
        'underset without parameters',
        # 第二十三个 ID：描述
        'unknown symbol',
        # 第二十四个 ID：描述
        'double superscript',
        # 第二十五个 ID：描述
        'double subscript',
        # 第二十六个 ID：描述
        'super on sub without braces',
        # 第二十七个 ID：描述
        'unclosed group',
    ]
@check_figures_equal(extensions=["png"])
def test_spaces(fig_test, fig_ref):
    # 在测试和参考图中添加文本，展示不同的空格命令的效果
    fig_test.text(.5, .5, r"$1\,2\>3\ 4$")
    fig_ref.text(.5, .5, r"$1\/2\:3~4$")


@check_figures_equal(extensions=["png"])
def test_operator_space(fig_test, fig_ref):
    # 在测试图中添加各种运算符与数字的组合，测试空格的排版效果
    fig_test.text(0.1, 0.1, r"$\log 6$")
    fig_test.text(0.1, 0.2, r"$\log(6)$")
    fig_test.text(0.1, 0.3, r"$\arcsin 6$")
    fig_test.text(0.1, 0.4, r"$\arcsin|6|$")
    fig_test.text(0.1, 0.5, r"$\operatorname{op} 6$")  # GitHub issue #553
    fig_test.text(0.1, 0.6, r"$\operatorname{op}[6]$")
    fig_test.text(0.1, 0.7, r"$\cos^2$")
    fig_test.text(0.1, 0.8, r"$\log_2$")
    fig_test.text(0.1, 0.9, r"$\sin^2 \cos$")  # GitHub issue #17852

    # 在参考图中添加相应的对比文本，使用正体字来显示数学操作符
    fig_ref.text(0.1, 0.1, r"$\mathrm{log\,}6$")
    fig_ref.text(0.1, 0.2, r"$\mathrm{log}(6)$")
    fig_ref.text(0.1, 0.3, r"$\mathrm{arcsin\,}6$")
    fig_ref.text(0.1, 0.4, r"$\mathrm{arcsin}|6|$")
    fig_ref.text(0.1, 0.5, r"$\mathrm{op\,}6$")
    fig_ref.text(0.1, 0.6, r"$\mathrm{op}[6]$")
    fig_ref.text(0.1, 0.7, r"$\mathrm{cos}^2$")
    fig_ref.text(0.1, 0.8, r"$\mathrm{log}_2$")
    fig_ref.text(0.1, 0.9, r"$\mathrm{sin}^2 \mathrm{\,cos}$")


@check_figures_equal(extensions=["png"])
def test_inverted_delimiters(fig_test, fig_ref):
    # 在测试和参考图中添加文本，展示倒置定界符的效果
    fig_test.text(.5, .5, r"$\left)\right($", math_fontfamily="dejavusans")
    fig_ref.text(.5, .5, r"$)($", math_fontfamily="dejavusans")


@check_figures_equal(extensions=["png"])
def test_genfrac_displaystyle(fig_test, fig_ref):
    # 在测试和参考图中添加文本，展示分式的效果
    fig_test.text(0.1, 0.1, r"$\dfrac{2x}{3y}$")

    # 计算下划线的粗细，用于设置参考图中的分式
    thickness = _mathtext.TruetypeFonts.get_underline_thickness(
        None, None, fontsize=mpl.rcParams["font.size"],
        dpi=mpl.rcParams["savefig.dpi"])
    fig_ref.text(0.1, 0.1, r"$\genfrac{}{}{%f}{0}{2x}{3y}$" % thickness)


def test_mathtext_fallback_valid():
    # 测试各种有效的回退字体设置
    for fallback in ['cm', 'stix', 'stixsans', 'None']:
        mpl.rcParams['mathtext.fallback'] = fallback


def test_mathtext_fallback_invalid():
    # 测试各种无效的回退字体设置，应该引发 ValueError 异常
    for fallback in ['abc', '']:
        with pytest.raises(ValueError, match="not a valid fallback font name"):
            mpl.rcParams['mathtext.fallback'] = fallback


@pytest.mark.parametrize(
    "fallback,fontlist",
    [("cm", ['DejaVu Sans', 'mpltest', 'STIXGeneral', 'cmr10', 'STIXGeneral']),
     ("stix", ['DejaVu Sans', 'mpltest', 'STIXGeneral'])])
def test_mathtext_fallback(fallback, fontlist):
    # 使用参数化测试各种回退字体设置，检查字体列表是否符合预期
    # 将自定义字体文件路径添加到 Matplotlib 的字体管理器中
    mpl.font_manager.fontManager.addfont(
        str(Path(__file__).resolve().parent / 'mpltest.ttf'))

    # 设置 SVG 渲染时不使用嵌入字体
    mpl.rcParams["svg.fonttype"] = 'none'

    # 设置数学文本使用自定义字体集
    mpl.rcParams['mathtext.fontset'] = 'custom'

    # 设置数学文本中普通字体使用指定的自定义字体
    mpl.rcParams['mathtext.rm'] = 'mpltest'

    # 设置数学文本中斜体字体使用指定的自定义字体的斜体版本
    mpl.rcParams['mathtext.it'] = 'mpltest:italic'

    # 设置数学文本中粗体字体使用指定的自定义字体的粗体版本
    mpl.rcParams['mathtext.bf'] = 'mpltest:bold'

    # 设置数学文本中粗斜体字体使用指定的自定义字体的粗斜体版本
    mpl.rcParams['mathtext.bfit'] = 'mpltest:italic:bold'

    # 设置数学文本的回退策略为指定的回退字体列表
    mpl.rcParams['mathtext.fallback'] = fallback

    # 定义一个包含特定数学符号的测试字符串
    test_str = r'a$A\AA\breve\gimel$'

    # 创建一个字节流缓冲区
    buff = io.BytesIO()

    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()

    # 向图形中添加文本，指定使用测试字符串、字体大小、水平对齐方式
    fig.text(.5, .5, test_str, fontsize=40, ha='center')

    # 将图形保存为 SVG 格式到缓冲区
    fig.savefig(buff, format="svg")

    # 从 SVG 字符串中解析出所有具有样式属性的 tspan 元素
    tspans = (ET.fromstring(buff.getvalue())
              .findall(".//{http://www.w3.org/2000/svg}tspan[@style]"))

    # 获取每个 tspan 元素的 style 属性中的最后一个属性，作为近似的字体属性
    char_fonts = [shlex.split(tspan.attrib["style"])[-1] for tspan in tspans]

    # 断言解析出的字符字体列表与预期的字体列表相等
    assert char_fonts == fontlist

    # 从 Matplotlib 的字体管理器中移除最后添加的字体
    mpl.font_manager.fontManager.ttflist.pop()
# 测试将数学公式转换为图像文件，并保存到临时路径
def test_math_to_image(tmp_path):
    # 调用函数将数学公式 $x^2$ 转换为图像，并保存到指定路径下的 example.png 文件
    mathtext.math_to_image('$x^2$', tmp_path / 'example.png')
    # 调用函数将数学公式 $x^2$ 转换为图像，并保存到字节流中
    mathtext.math_to_image('$x^2$', io.BytesIO())
    # 调用函数将数学公式 $x^2$ 转换为图像，并保存到字节流中，使用 'Maroon' 颜色
    mathtext.math_to_image('$x^2$', io.BytesIO(), color='Maroon')


# 使用图像比较装饰器对比测试不同数学字体下的图像输出
@image_comparison(baseline_images=['math_fontfamily_image.png'],
                  savefig_kwarg={'dpi': 40})
def test_math_fontfamily():
    # 创建一个大小为 (10, 3) 的图形对象
    fig = plt.figure(figsize=(10, 3))
    # 在图形中添加文本，使用 'dejavusans' 字体集合，显示给定的数学公式文本
    fig.text(0.2, 0.7, r"$This\ text\ should\ have\ one\ font$",
             size=24, math_fontfamily='dejavusans')
    # 在图形中添加文本，使用 'stix' 字体集合，显示给定的数学公式文本
    fig.text(0.2, 0.3, r"$This\ text\ should\ have\ another$",
             size=24, math_fontfamily='stix')


# 测试默认数学字体集合是否按预期设置
def test_default_math_fontfamily():
    # 设置 matplotlib 全局参数，指定数学文本使用 'cm' 字体集合
    mpl.rcParams['mathtext.fontset'] = 'cm'
    # 定义测试字符串，包含数学公式和文本混合内容
    test_str = r'abc$abc\alpha$'
    # 创建一个图形对象和对应的坐标轴
    fig, ax = plt.subplots()

    # 在图形中添加文本，使用 'Arial' 字体，验证其数学字体集合是否为 'cm'
    text1 = fig.text(0.1, 0.1, test_str, font='Arial')
    prop1 = text1.get_fontproperties()
    assert prop1.get_math_fontfamily() == 'cm'

    # 在图形中添加文本，使用 'Arial' 字体，验证其数学字体集合是否为 'cm'
    text2 = fig.text(0.2, 0.2, test_str, fontproperties='Arial')
    prop2 = text2.get_fontproperties()
    assert prop2.get_math_fontfamily() == 'cm'

    # 绘制图形但不渲染，用于测试设置是否生效
    fig.draw_without_rendering()


# 测试数学字体集合参数的顺序对比
def test_argument_order():
    # 设置 matplotlib 全局参数，指定数学文本使用 'cm' 字体集合
    mpl.rcParams['mathtext.fontset'] = 'cm'
    # 定义测试字符串，包含数学公式和文本混合内容
    test_str = r'abc$abc\alpha$'
    # 创建一个图形对象和对应的坐标轴
    fig, ax = plt.subplots()

    # 在图形中添加文本，使用 'Arial' 字体，指定数学字体集合为 'dejavusans'
    text1 = fig.text(0.1, 0.1, test_str,
                     math_fontfamily='dejavusans', font='Arial')
    prop1 = text1.get_fontproperties()
    assert prop1.get_math_fontfamily() == 'dejavusans'

    # 在图形中添加文本，使用 'Arial' 字体，指定数学字体集合为 'dejavusans'
    text2 = fig.text(0.2, 0.2, test_str,
                     math_fontfamily='dejavusans', fontproperties='Arial')
    prop2 = text2.get_fontproperties()
    assert prop2.get_math_fontfamily() == 'dejavusans'

    # 在图形中添加文本，使用 'Arial' 字体，指定数学字体集合为 'dejavusans'
    text3 = fig.text(0.3, 0.3, test_str,
                     font='Arial', math_fontfamily='dejavusans')
    prop3 = text3.get_fontproperties()
    assert prop3.get_math_fontfamily() == 'dejavusans'

    # 在图形中添加文本，使用 'Arial' 字体，指定数学字体集合为 'dejavusans'
    text4 = fig.text(0.4, 0.4, test_str,
                     fontproperties='Arial', math_fontfamily='dejavusans')
    prop4 = text4.get_fontproperties()
    assert prop4.get_math_fontfamily() == 'dejavusans'

    # 绘制图形但不渲染，用于测试设置是否生效
    fig.draw_without_rendering()


# 测试数学文本中的 cmr10 字体是否正确显示减号
def test_mathtext_cmr10_minus_sign():
    # 设置 matplotlib 字体族为 'cmr10'，启用数学文本渲染
    mpl.rcParams['font.family'] = 'cmr10'
    mpl.rcParams['axes.formatter.use_mathtext'] = True
    # 创建一个图形对象和对应的坐标轴
    fig, ax = plt.subplots()
    # 绘制一条包含减号的曲线，以验证是否会出现字形缺失警告
    ax.plot(range(-1, 1), range(-1, 1))
    # 绘制图形以确保没有警告出现
    fig.canvas.draw()


# 测试数学文本中的运算符和特殊符号显示是否正常
def test_mathtext_operators():
    # 定义包含多个数学运算符和特殊符号的数学公式字符串
    test_str = r'''
    \increment \smallin \notsmallowns
    \smallowns \QED \rightangle
    \smallintclockwise \smallvarointclockwise
    \smallointctrcclockwise
    \ratio \minuscolon \dotsminusdots
    \sinewave \simneqq \nlesssim
    \ngtrsim \nlessgtr \ngtrless
    \cupleftarrow \oequal \rightassert
    \rightModels \hermitmatrix \barvee
    \measuredrightangle \varlrtriangle
    \equalparallel \npreccurlyeq \nsucccurlyeq
    \nsqsubseteq \nsqsupseteq \sqsubsetneq
    '''
    # 创建一个包含数学符号字符串的列表
    test_str = r'''
    \sqsupsetneq  \disin \varisins
    \isins \isindot \varisinobar
    \isinobar \isinvb \isinE
    \nisd \varnis \nis
    \varniobar \niobar \bagmember
    \triangle'''.split()

    # 创建一个新的图形对象
    fig = plt.figure()
    
    # 对于列表中的每个数学符号字符串进行迭代
    for x, i in enumerate(test_str):
        # 在图形中添加文本，位置为(0.5, (x + 0.5)/len(test_str)，内容为数学符号字符串
        fig.text(0.5, (x + 0.5)/len(test_str), r'${%s}$' % i)

    # 绘制图形但不进行渲染
    fig.draw_without_rendering()
# 声明一个装饰器函数，用于检查两个图形是否相等，限定文件扩展名为"png"
@check_figures_equal(extensions=["png"])
# 定义一个测试函数，测试粗体符号的显示
def test_boldsymbol(fig_test, fig_ref):
    # 在测试图形中添加文本，显示粗体符号`\boldsymbol{\mathrm{abc0123\alpha}}`
    fig_test.text(0.1, 0.2, r"$\boldsymbol{\mathrm{abc0123\alpha}}$")
    # 在参考图形中添加文本，显示非粗体符号`\mathrm{abc0123\alpha}`
    fig_ref.text(0.1, 0.2, r"$\mathrm{abc0123\alpha}$")
```