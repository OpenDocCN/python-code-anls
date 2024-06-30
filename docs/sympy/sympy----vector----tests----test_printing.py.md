# `D:\src\scipysrc\sympy\sympy\vector\tests\test_printing.py`

```
# -*- coding: utf-8 -*-
# 导入 sympy 库中需要的类和函数
from sympy.core.function import Function
from sympy.integrals.integrals import Integral
from sympy.printing.latex import latex
from sympy.printing.pretty import pretty as xpretty
from sympy.vector import CoordSys3D, Del, Vector, express
from sympy.abc import a, b, c
from sympy.testing.pytest import XFAIL

# 定义 ASCII 美化输出函数
def pretty(expr):
    """ASCII pretty-printing"""
    return xpretty(expr, use_unicode=False, wrap_line=False)

# 定义 Unicode 美化输出函数
def upretty(expr):
    """Unicode pretty-printing"""
    return xpretty(expr, use_unicode=True, wrap_line=False)

# 初始化坐标系和向量表达式
# 用于测试的基本和冗长的向量/二阶张量表达式
N = CoordSys3D('N')
C = N.orient_new_axis('C', a, N.k)  # 定义新的坐标系 C，绕 N.k 轴旋转 a 弧度
v = []
d = []

# 将向量表达式添加到列表 v 中
v.append(Vector.zero)  # 添加零向量
v.append(N.i)  # 添加单位向量 i_N
v.append(-N.i)  # 添加反向单位向量 -i_N
v.append(N.i + N.j)  # 添加 i_N + j_N
v.append(a*N.i)  # 添加 a*i_N
v.append(a*N.i - b*N.j)  # 添加 a*i_N - b*j_N
v.append((a**2 + N.x)*N.i + N.k)  # 添加 (a**2 + x_N)*i_N + k_N
v.append((a**2 + b)*N.i + 3*(C.y - c)*N.k)  # 添加 (a**2 + b)*i_N + 3*(y_C - c)*k_N

# 添加复杂的向量表达式，使用函数 f 和积分 Integral
v.append(N.j - (Integral(f(b)) - C.x**2)*N.k)

# Unicode 美化输出结果，此处 upretty_v_8 表示第 8 个向量的 Unicode 美化输出形式
upretty_v_8 = """\
      ⎛   2   ⌠        ⎞    \n\
j_N + ⎜x_C  - ⎮ f(b) db⎟ k_N\n\
      ⎝       ⌡        ⎠    \
"""

# ASCII 美化输出结果，此处 pretty_v_8 表示第 8 个向量的 ASCII 美化输出形式
pretty_v_8 = """\
j_N + /         /       \\\n\
      |   2    |        |\n\
      |x_C  -  | f(b) db|\n\
      |        |        |\n\
      \\       /         / \
"""

# 添加更多向量表达式到列表 v 中
v.append(N.i + C.k)  # 添加 i_N + k_C
v.append(express(N.i, C))  # 使用坐标系 C 对 i_N 进行表达式展开
v.append((a**2 + b)*N.i + (Integral(f(b)))*N.k)  # 添加 (a**2 + b)*i_N + Integral(f(b))*k_N

# Unicode 美化输出结果，此处 upretty_v_11 表示第 11 个向量的 Unicode 美化输出形式
upretty_v_11 = """\
⎛ 2    ⎞        ⎛⌠        ⎞    \n\
⎝a  + b⎠ i_N  + ⎜⎮ f(b) db⎟ k_N\n\
                ⎝⌡        ⎠    \
"""

# ASCII 美化输出结果，此处 pretty_v_11 表示第 11 个向量的 ASCII 美化输出形式
pretty_v_11 = """\
/ 2    \\ + /  /       \\\n\
\\a  + b/ i_N| |        |\n\
           | | f(b) db|\n\
           | |        |\n\
           \\/         / \
"""

# 遍历向量列表 v，计算每个向量与 N.k 的点积并添加到列表 d 中
for x in v:
    d.append(x | N.k)

# 定义复杂的标量表达式 s
s = 3*N.x**2*C.y

# Unicode 美化输出结果，此处 upretty_s 表示标量 s 的 Unicode 美化输出形式
upretty_s = """\
         2\n\
3⋅y_C⋅x_N \
"""

# ASCII 美化输出结果，此处 pretty_s 表示标量 s 的 ASCII 美化输出形式
pretty_s = """\
         2\n\
3*y_C*x_N \
"""

# Unicode 美化输出结果，此处 upretty_d_7 表示第 7 个点积向量的 Unicode 美化输出形式
upretty_d_7 = """\
⎛ 2    ⎞                                     \n\
⎝a  + b⎠ (i_N|k_N)  + (3⋅y_C - 3⋅c) (k_N|k_N)\
"""

# ASCII 美化输出结果，此处 pretty_d_7 表示第 7 个点积向量的 ASCII 美化输出形式
pretty_d_7 = """\
/ 2    \\ (i_N|k_N) + (3*y_C - 3*c) (k_N|k_N)\n\
\\a  + b/                                    \
"""

# 定义测试函数 test_str_printing，用于验证字符串输出是否符合预期
def test_str_printing():
    assert str(v[0]) == '0'
    assert str(v[1]) == 'N.i'
    assert str(v[2]) == '(-1)*N.i'
    assert str(v[3]) == 'N.i + N.j'
    assert str(v[8]) == 'N.j + (C.x**2 - Integral(f(b), b))*N.k'
    assert str(v[9]) == 'C.k + N.i'
    assert str(s) == '3*C.y*N.x**2'
    assert str(d[0]) == '0'
    assert str(d[1]) == '(N.i|N.k)'
    assert str(d[4]) == 'a*(N.i|N.k)'
    assert str(d[5]) == 'a*(N.i|N.k) + (-b)*(N.j|N.k)'
    assert str(d[8]) == ('(N.j|N.k) + (C.x**2 - ' +
                         'Integral(f(b), b))*(N.k|N.k)')

# 标记整个测试函数为预期失败状态
@XFAIL
# 测试函数，用于验证 pretty 函数的 ASCII 输出是否正确
def test_pretty_printing_ascii():
    assert pretty(v[0]) == '0'  # 检查索引为 0 的向量 v 的 ASCII 输出
    assert pretty(v[1]) == 'i_N'  # 检查索引为 1 的向量 v 的 ASCII 输出
    assert pretty(v[5]) == '(a) i_N + (-b) j_N'  # 检查索引为 5 的向量 v 的 ASCII 输出
    assert pretty(v[8]) == pretty_v_8  # 检查索引为 8 的向量 v 的 ASCII 输出是否等于预定义的 pretty_v_8
    assert pretty(v[2]) == '(-1) i_N'  # 检查索引为 2 的向量 v 的 ASCII 输出
    assert pretty(v[11]) == pretty_v_11  # 检查索引为 11 的向量 v 的 ASCII 输出是否等于预定义的 pretty_v_11
    assert pretty(s) == pretty_s  # 检查对象 s 的 ASCII 输出是否等于预定义的 pretty_s
    assert pretty(d[0]) == '(0|0)'  # 检查索引为 0 的双分量 d 的 ASCII 输出
    assert pretty(d[5]) == '(a) (i_N|k_N) + (-b) (j_N|k_N)'  # 检查索引为 5 的双分量 d 的 ASCII 输出
    assert pretty(d[7]) == pretty_d_7  # 检查索引为 7 的双分量 d 的 ASCII 输出是否等于预定义的 pretty_d_7
    assert pretty(d[10]) == '(cos(a)) (i_C|k_N) + (-sin(a)) (j_C|k_N)'  # 检查索引为 10 的双分量 d 的 ASCII 输出

# 测试函数，用于验证 upretty 函数的 Unicode 输出是否正确
def test_pretty_print_unicode_v():
    assert upretty(v[0]) == '0'  # 检查索引为 0 的向量 v 的 Unicode 输出
    assert upretty(v[1]) == 'i_N'  # 检查索引为 1 的向量 v 的 Unicode 输出
    assert upretty(v[5]) == '(a) i_N + (-b) j_N'  # 检查索引为 5 的向量 v 的 Unicode 输出
    # 确保打印在其他对象中也能正常工作
    assert upretty(v[5].args) == '((a) i_N, (-b) j_N)'  # 检查索引为 5 的向量 v 的参数对象的 Unicode 输出
    assert upretty(v[8]) == upretty_v_8  # 检查索引为 8 的向量 v 的 Unicode 输出是否等于预定义的 upretty_v_8
    assert upretty(v[2]) == '(-1) i_N'  # 检查索引为 2 的向量 v 的 Unicode 输出
    assert upretty(v[11]) == upretty_v_11  # 检查索引为 11 的向量 v 的 Unicode 输出是否等于预定义的 upretty_v_11
    assert upretty(s) == upretty_s  # 检查对象 s 的 Unicode 输出是否等于预定义的 upretty_s
    assert upretty(d[0]) == '(0|0)'  # 检查索引为 0 的双分量 d 的 Unicode 输出
    assert upretty(d[5]) == '(a) (i_N|k_N) + (-b) (j_N|k_N)'  # 检查索引为 5 的双分量 d 的 Unicode 输出
    assert upretty(d[7]) == upretty_d_7  # 检查索引为 7 的双分量 d 的 Unicode 输出是否等于预定义的 upretty_d_7
    assert upretty(d[10]) == '(cos(a)) (i_C|k_N) + (-sin(a)) (j_C|k_N)'  # 检查索引为 10 的双分量 d 的 Unicode 输出

# 测试函数，用于验证 latex 函数的 LaTeX 输出是否正确
def test_latex_printing():
    assert latex(v[0]) == '\\mathbf{\\hat{0}}'  # 检查索引为 0 的向量 v 的 LaTeX 输出
    assert latex(v[1]) == '\\mathbf{\\hat{i}_{N}}'  # 检查索引为 1 的向量 v 的 LaTeX 输出
    assert latex(v[2]) == '- \\mathbf{\\hat{i}_{N}}'  # 检查索引为 2 的向量 v 的 LaTeX 输出
    assert latex(v[5]) == ('\\left(a\\right)\\mathbf{\\hat{i}_{N}} + ' +
                           '\\left(- b\\right)\\mathbf{\\hat{j}_{N}}')  # 检查索引为 5 的向量 v 的 LaTeX 输出
    assert latex(v[6]) == ('\\left(\\mathbf{{x}_{N}} + a^{2}\\right)\\mathbf{\\hat{i}_' +
                          '{N}} + \\mathbf{\\hat{k}_{N}}')  # 检查索引为 6 的向量 v 的 LaTeX 输出
    assert latex(v[8]) == ('\\mathbf{\\hat{j}_{N}} + \\left(\\mathbf{{x}_' +
                           '{C}}^{2} - \\int f{\\left(b \\right)}\\,' +
                           ' db\\right)\\mathbf{\\hat{k}_{N}}')  # 检查索引为 8 的向量 v 的 LaTeX 输出
    assert latex(s) == '3 \\mathbf{{y}_{C}} \\mathbf{{x}_{N}}^{2}'  # 检查对象 s 的 LaTeX 输出
    assert latex(d[0]) == '(\\mathbf{\\hat{0}}|\\mathbf{\\hat{0}})'  # 检查索引为 0 的双分量 d 的 LaTeX 输出
    assert latex(d[4]) == ('\\left(a\\right)\\left(\\mathbf{\\hat{i}_{N}}{\\middle|}' +
                           '\\mathbf{\\hat{k}_{N}}\\right)')  # 检查索引为 4 的双分量 d 的 LaTeX 输出
    assert latex(d[9]) == ('\\left(\\mathbf{\\hat{k}_{C}}{\\middle|}' +
                           '\\mathbf{\\hat{k}_{N}}\\right) + \\left(' +
                           '\\mathbf{\\hat{i}_{N}}{\\middle|}\\mathbf{' +
                           '\\hat{k}_{N}}\\right)')  # 检查索引为 9 的双分量 d 的 LaTeX 输出
    assert latex(d[11]) == ('\\left(a^{2} + b\\right)\\left(\\mathbf{\\hat{i}_{N}}' +
                            '{\\middle|}\\mathbf{\\hat{k}_{N}}\\right) + ' +
                            '\\left(\\int f{\\left(b \\right)}\\, db\\right)\\left(' +
                            '\\mathbf{\\hat{k}_{N}}{\\middle|}\\mathbf{' +
                            '\\hat{k}_{N}}\\right)')  # 检查索引为 11 的双分量 d 的 LaTeX 输出

# 测试函数，用于验证 issue_23058 的特定问题
def test_issue_23058():
    from sympy import symbols, sin, cos, pi, UnevaluatedExpr

    delop = Del()  # 实例化 Del 对象
    CC_   = CoordSys3D("C")  # 实例化三维坐标系对象 CC_
    y     = CC_.y  # 从 CC_ 获取 y 分量
    xhat  = CC_.i  # 从 CC_ 获取 i 分量

    t = symbols("t")  # 创建符号变量 t
    ten = symbols("10", positive=True)  # 创建正数符号变量 ten
    eps, mu = 4*pi*ten**(-11), ten**(-5)
    # 定义电介质常数 epsilon 和磁导率 mu，分别赋值为 4π×10^(-11) 和 10^(-5)

    Bx = 2 * ten**(-4) * cos(ten**5 * t) * sin(ten**(-3) * y)
    # 计算磁场 Bx 的 x 分量，公式为 2×10^(-4)×cos(10^5×t)×sin(10^(-3)×y)

    vecB = Bx * xhat
    # 构建磁场向量 vecB，其 x 分量为 Bx，y 和 z 分量为 0

    vecE = (1/eps) * Integral(delop.cross(vecB/mu).doit(), t)
    # 计算电场向量 vecE，公式为 (1/epsilon) × ∫(∇ × (vecB / mu)) dt
    # 其中 ∇ × 表示向量场的旋度，.doit() 表示执行积分计算

    vecE = vecE.doit()
    # 计算并更新电场向量 vecE 的值，得到最终结果

    vecB_str = """\
# 定义一个字符串，表示电磁场向量B的矢量形式，包含数学表达式和单位矢量
vecB_str = """\
⎛    -4    ⎛    5⎞    ⎛      -3⎞⎞     \n\
⎝2⋅10  ⋅cos⎝t⋅10 ⎠⋅sin⎝y_C⋅10  ⎠⎠ i_C \
"""

# 定义一个字符串，表示电磁场向量E的矢量形式，包含数学表达式和单位矢量
vecE_str = """\
⎛   4    ⎛  5  ⎞    ⎛y_C⎞ ⎞    \n\
⎜-10 ⋅sin⎝10 ⋅t⎠⋅cos⎜───⎟ ⎟ k_C\n\
⎜                   ⎜  3⎟ ⎟    \n\
⎜                   ⎝10 ⎠ ⎟    \n\
⎜─────────────────────────⎟    \n\
⎝           2⋅π           ⎠    \
"""

# 断言，验证电磁场向量B的Unicode字符串表示与预期字符串vecB_str相等
assert upretty(vecB) == vecB_str

# 断言，验证电磁场向量E的Unicode字符串表示与预期字符串vecE_str相等
assert upretty(vecE) == vecE_str

# 定义常数10，并用其计算ε和μ的值
ten = UnevaluatedExpr(10)
eps, mu = 4*pi*ten**(-11), ten**(-5)

# 计算电磁场向量Bx的数学表达式，其中包含时间t和空间坐标y的函数
Bx = 2 * ten**(-4) * cos(ten**5 * t) * sin(ten**(-3) * y)

# 计算完整的电磁场向量B，由Bx乘以x单位矢量构成
vecB = Bx * xhat

# 定义字符串，表示电磁场向量B的矢量形式，包含数学表达式和单位矢量
vecB_str = """\
⎛    -4    ⎛    5⎞    ⎛      -3⎞⎞     \n\
⎝2⋅10  ⋅cos⎝t⋅10 ⎠⋅sin⎝y_C⋅10  ⎠⎠ i_C \
"""

# 断言，验证电磁场向量B的Unicode字符串表示与预期字符串vecB_str相等
assert upretty(vecB) == vecB_str
```