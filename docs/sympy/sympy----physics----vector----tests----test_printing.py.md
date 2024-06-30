# `D:\src\scipysrc\sympy\sympy\physics\vector\tests\test_printing.py`

```
# -*- coding: utf-8 -*-

# 导入 SymPy 中所需的函数和符号
from sympy.core.function import Function
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (asin, cos, sin)
from sympy.physics.vector import ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.vector.printing import (VectorLatexPrinter, vpprint,
                                           vsprint, vsstrrepr, vlatex)

# 定义符号变量
a, b, c = symbols('a, b, c')
alpha, omega, beta = dynamicsymbols('alpha, omega, beta')

# 创建惯性参考系 A 和固定参考系 N
A = ReferenceFrame('A')
N = ReferenceFrame('N')

# 定义向量 v, w, ww, o，分别为线性组合的结果
v = a ** 2 * N.x + b * N.y + c * sin(alpha) * N.z
w = alpha * N.x + sin(omega) * N.y + alpha * beta * N.z
ww = alpha * N.x + asin(omega) * N.y - alpha.diff() * beta * N.z
o = a/b * N.x + (c+b)/a * N.y + c**2/b * N.z

# 定义叉乘结果 y, x, xx, xx2
y = a ** 2 * (N.x | N.y) + b * (N.y | N.y) + c * sin(alpha) * (N.z | N.y)
x = alpha * (N.x | N.x) + sin(omega) * (N.y | N.z) + alpha * beta * (N.z | N.x)
xx = N.x | (-N.y - N.z)
xx2 = N.x | (N.y + N.z)

# 定义一个函数，使用 ASCII 码进行向量美化打印
def ascii_vpretty(expr):
    return vpprint(expr, use_unicode=False, wrap_line=False)

# 定义一个函数，使用 Unicode 进行向量美化打印
def unicode_vpretty(expr):
    return vpprint(expr, use_unicode=True, wrap_line=False)

# 测试 LaTeX 打印功能
def test_latex_printer():
    r = Function('r')('t')
    assert VectorLatexPrinter().doprint(r ** 2) == "r^{2}"
    r2 = Function('r^2')('t')
    assert VectorLatexPrinter().doprint(r2.diff()) == r'\dot{r^{2}}'
    ra = Function('r__a')('t')
    assert VectorLatexPrinter().doprint(ra.diff().diff()) == r'\ddot{r^{a}}'

# 测试向量的美化打印
def test_vector_pretty_print():

    # TODO : The unit vectors should print with subscripts but they just
    # print as `n_x` instead of making `x` a subscript with unicode.

    # TODO : The pretty print division does not print correctly here:
    # w = alpha * N.x + sin(omega) * N.y + alpha / beta * N.z

    # 期望的 ASCII 美化打印结果
    expected = """\
 2                               \n\
a  n_x + b n_y + c*sin(alpha) n_z\
"""
    # 期望的 Unicode 美化打印结果
    uexpected = """\
 2                           \n\
a  n_x + b n_y + c⋅sin(α) n_z\
"""

    # 断言 ASCII 和 Unicode 打印结果
    assert ascii_vpretty(v) == expected
    assert unicode_vpretty(v) == uexpected

    # 期望的 ASCII 和 Unicode 美化打印结果
    expected = 'alpha n_x + sin(omega) n_y + alpha*beta n_z'
    uexpected = 'α n_x + sin(ω) n_y + α⋅β n_z'

    # 断言 ASCII 和 Unicode 打印结果
    assert ascii_vpretty(w) == expected
    assert unicode_vpretty(w) == uexpected

    # 期望的 ASCII 和 Unicode 美化打印结果
    expected = """\
                     2    \n\
a       b + c       c     \n\
- n_x + ----- n_y + -- n_z\n\
b         a         b     \
"""
    uexpected = """\
                     2    \n\
a       b + c       c     \n\
─ n_x + ───── n_y + ── n_z\n\
b         a         b     \
"""

    # 断言 ASCII 和 Unicode 打印结果
    assert ascii_vpretty(o) == expected
    assert unicode_vpretty(o) == uexpected

    # https://github.com/sympy/sympy/issues/26731
    assert ascii_vpretty(-A.x) == '-a_x'
    assert unicode_vpretty(-A.x) == '-a_x'

# 测试向量的 LaTeX 打印
def test_vector_latex():

    # 定义符号变量
    a, b, c, d, omega = symbols('a, b, c, d, omega')

    # 定义向量 v
    v = (a ** 2 + b / c) * A.x + sqrt(d) * A.y + cos(omega) * A.z
    # 确认向量表达式 vlatex(v) 的输出是否符合预期
    assert vlatex(v) == (r'(a^{2} + \frac{b}{c})\mathbf{\hat{a}_x} + '
                         r'\sqrt{d}\mathbf{\hat{a}_y} + '
                         r'\cos{\left(\omega \right)}'
                         r'\mathbf{\hat{a}_z}')
    
    # 定义动力学符号 theta, omega, alpha, q
    theta, omega, alpha, q = dynamicsymbols('theta, omega, alpha, q')
    
    # 构建向量表达式 v
    v = theta * A.x + omega * omega * A.y + (q * alpha) * A.z
    
    # 确认向量表达式 vlatex(v) 的输出是否符合预期
    assert vlatex(v) == (r'\theta\mathbf{\hat{a}_x} + '
                         r'\omega^{2}\mathbf{\hat{a}_y} + '
                         r'\alpha q\mathbf{\hat{a}_z}')
    
    # 定义动力学符号 phi1, phi2, phi3 和符号 theta1, theta2, theta3
    phi1, phi2, phi3 = dynamicsymbols('phi1, phi2, phi3')
    theta1, theta2, theta3 = symbols('theta1, theta2, theta3')
    
    # 构建向量表达式 v
    v = (sin(theta1) * A.x +
         cos(phi1) * cos(phi2) * A.y +
         cos(theta1 + phi3) * A.z)
    
    # 确认向量表达式 vlatex(v) 的输出是否符合预期
    assert vlatex(v) == (r'\sin{\left(\theta_{1} \right)}'
                         r'\mathbf{\hat{a}_x} + \cos{'
                         r'\left(\phi_{1} \right)} \cos{'
                         r'\left(\phi_{2} \right)}\mathbf{\hat{a}_y} + '
                         r'\cos{\left(\theta_{1} + '
                         r'\phi_{3} \right)}\mathbf{\hat{a}_z}')
    
    # 创建惯性参考系 N
    N = ReferenceFrame('N')
    
    # 定义符号 a, b, c, d, omega
    a, b, c, d, omega = symbols('a, b, c, d, omega')
    
    # 构建向量表达式 v
    v = (a ** 2 + b / c) * N.x + sqrt(d) * N.y + cos(omega) * N.z
    
    # 预期输出字符串
    expected = (r'(a^{2} + \frac{b}{c})\mathbf{\hat{n}_x} + '
                r'\sqrt{d}\mathbf{\hat{n}_y} + '
                r'\cos{\left(\omega \right)}'
                r'\mathbf{\hat{n}_z}')
    
    # 确认向量表达式 vlatex(v) 的输出是否符合预期
    assert vlatex(v) == expected
    
    # 尝试使用自定义单位向量
    N = ReferenceFrame('N', latexs=(r'\hat{i}', r'\hat{j}', r'\hat{k}'))
    
    # 重新构建向量表达式 v
    v = (a ** 2 + b / c) * N.x + sqrt(d) * N.y + cos(omega) * N.z
    
    # 更新预期输出字符串
    expected = (r'(a^{2} + \frac{b}{c})\hat{i} + '
                r'\sqrt{d}\hat{j} + '
                r'\cos{\left(\omega \right)}\hat{k}')
    
    # 确认向量表达式 vlatex(v) 的输出是否符合预期
    assert vlatex(v) == expected
    
    # 预期输出字符串
    expected = r'\alpha\mathbf{\hat{n}_x} + \operatorname{asin}{\left(\omega ' \
        r'\right)}\mathbf{\hat{n}_y} -  \beta \dot{\alpha}\mathbf{\hat{n}_z}'
    
    # 确认向量表达式 vlatex(ww) 的输出是否符合预期
    assert vlatex(ww) == expected
    
    # 预期输出字符串
    expected = r'- \mathbf{\hat{n}_x}\otimes \mathbf{\hat{n}_y} - ' \
        r'\mathbf{\hat{n}_x}\otimes \mathbf{\hat{n}_z}'
    
    # 确认向量表达式 vlatex(xx) 的输出是否符合预期
    assert vlatex(xx) == expected
    
    # 预期输出字符串
    expected = r'\mathbf{\hat{n}_x}\otimes \mathbf{\hat{n}_y} + ' \
        r'\mathbf{\hat{n}_x}\otimes \mathbf{\hat{n}_z}'
    
    # 确认向量表达式 vlatex(xx2) 的输出是否符合预期
    assert vlatex(xx2) == expected
def test_vector_latex_arguments():
    assert vlatex(N.x * 3.0, full_prec=False) == r'3.0\mathbf{\hat{n}_x}'
    assert vlatex(N.x * 3.0, full_prec=True) == r'3.00000000000000\mathbf{\hat{n}_x}'


def test_vector_latex_with_functions():
    # 创建一个参考坐标系对象 N
    N = ReferenceFrame('N')
    
    # 定义动力学符号 omega 和 alpha
    omega, alpha = dynamicsymbols('omega, alpha')
    
    # 创建向量 v = \dot{\omega} * \mathbf{\hat{n}_x}
    v = omega.diff() * N.x
    assert vlatex(v) == r'\dot{\omega}\mathbf{\hat{n}_x}'
    
    # 创建向量 v = \dot{\omega}^{\alpha} * \mathbf{\hat{n}_x}
    v = omega.diff() ** alpha * N.x
    assert vlatex(v) == (r'\dot{\omega}^{\alpha}'
                          r'\mathbf{\hat{n}_x}')


def test_dyadic_pretty_print():
    expected = """\
 2
a  n_x|n_y + b n_y|n_y + c*sin(alpha) n_z|n_y\
"""
    uexpected = """\
 2
a  n_x⊗n_y + b n_y⊗n_y + c⋅sin(α) n_z⊗n_y\
"""
    # 验证 ASCII 格式和 Unicode 格式的输出是否与预期一致
    assert ascii_vpretty(y) == expected
    assert unicode_vpretty(y) == uexpected
    
    expected = 'alpha n_x|n_x + sin(omega) n_y|n_z + alpha*beta n_z|n_x'
    uexpected = 'α n_x⊗n_x + sin(ω) n_y⊗n_z + α⋅β n_z⊗n_x'
    assert ascii_vpretty(x) == expected
    assert unicode_vpretty(x) == uexpected
    
    # 验证零矢量的 ASCII 和 Unicode 格式输出
    assert ascii_vpretty(Dyadic([])) == '0'
    assert unicode_vpretty(Dyadic([])) == '0'
    
    assert ascii_vpretty(xx) == '- n_x|n_y - n_x|n_z'
    assert unicode_vpretty(xx) == '- n_x⊗n_y - n_x⊗n_z'
    
    assert ascii_vpretty(xx2) == 'n_x|n_y + n_x|n_z'
    assert unicode_vpretty(xx2) == 'n_x⊗n_y + n_x⊗n_z'


def test_dyadic_latex():
    expected = (r'a^{2}\mathbf{\hat{n}_x}\otimes \mathbf{\hat{n}_y} + '
                r'b\mathbf{\hat{n}_y}\otimes \mathbf{\hat{n}_y} + '
                r'c \sin{\left(\alpha \right)}'
                r'\mathbf{\hat{n}_z}\otimes \mathbf{\hat{n}_y}')
    
    # 验证 LaTeX 格式的输出是否与预期一致
    assert vlatex(y) == expected
    
    expected = (r'\alpha\mathbf{\hat{n}_x}\otimes \mathbf{\hat{n}_x} + '
                r'\sin{\left(\omega \right)}\mathbf{\hat{n}_y}'
                r'\otimes \mathbf{\hat{n}_z} + '
                r'\alpha \beta\mathbf{\hat{n}_z}\otimes \mathbf{\hat{n}_x}')
    
    assert vlatex(x) == expected
    
    # 验证零矢量的 LaTeX 格式输出
    assert vlatex(Dyadic([])) == '0'


def test_dyadic_str():
    # 验证字符串格式的输出是否与预期一致
    assert vsprint(Dyadic([])) == '0'
    assert vsprint(y) == 'a**2*(N.x|N.y) + b*(N.y|N.y) + c*sin(alpha)*(N.z|N.y)'
    assert vsprint(x) == 'alpha*(N.x|N.x) + sin(omega)*(N.y|N.z) + alpha*beta*(N.z|N.x)'
    assert vsprint(ww) == "alpha*N.x + asin(omega)*N.y - beta*alpha'*N.z"
    assert vsprint(xx) == '- (N.x|N.y) - (N.x|N.z)'
    assert vsprint(xx2) == '(N.x|N.y) + (N.x|N.z)'


def test_vlatex(): # vlatex is broken #12078
    from sympy.physics.vector import vlatex
    
    x = symbols('x')
    J = symbols('J')
    
    f = Function('f')
    g = Function('g')
    h = Function('h')
    
    # 预期的 LaTeX 格式输出
    expected = r'J \left(\frac{d}{d x} g{\left(x \right)} - \frac{d}{d x} h{\left(x \right)}\right)'
    
    expr = J*f(x).diff(x).subs(f(x), g(x)-h(x))
    
    # 验证 vlatex 函数输出是否与预期一致
    assert vlatex(expr) == expected


def test_issue_13354():
    """
    Test for proper pretty printing of physics vectors with ADD
    instances in arguments.

    Test is exactly the one suggested in the original bug report by
    @moorepants.
    """
    # 这个函数用于测试带有 ADD 实例的物理向量的正确漂亮打印
    pass
    # 导入必要的符号和参考帧模块，创建符号变量 a, b, c
    a, b, c = symbols('a, b, c')
    
    # 创建参考帧 'A'，并定义向量 v 为 a*A.x + b*A.y + c*A.z
    A = ReferenceFrame('A')
    v = a * A.x + b * A.y + c * A.z
    
    # 定义向量 w 为 b*A.x + c*A.y + a*A.z
    w = b * A.x + c * A.y + a * A.z
    
    # 计算向量 z 为向量 w 和向量 v 的和
    z = w + v
    
    # 定义预期结果字符串，表示为 '(a + b) a_x + (b + c) a_y + (a + c) a_z'
    expected = """(a + b) a_x + (b + c) a_y + (a + c) a_z"""
    
    # 使用断言验证 ascii_vpretty(z) 函数生成的字符串与预期结果是否一致
    assert ascii_vpretty(z) == expected
def test_vector_derivative_printing():
    # First order
    v = omega.diff() * N.x
    # 断言 unicode_vpretty 输出符合预期
    assert unicode_vpretty(v) == 'ω̇ n_x'
    # 断言 ascii_vpretty 输出符合预期
    assert ascii_vpretty(v) == "omega'(t) n_x"

    # Second order
    v = omega.diff().diff() * N.x
    # 断言 vlatex 输出符合预期
    assert vlatex(v) == r'\ddot{\omega}\mathbf{\hat{n}_x}'
    # 断言 unicode_vpretty 输出符合预期
    assert unicode_vpretty(v) == 'ω̈ n_x'
    # 断言 ascii_vpretty 输出符合预期
    assert ascii_vpretty(v) == "omega''(t) n_x"

    # Third order
    v = omega.diff().diff().diff() * N.x
    # 断言 vlatex 输出符合预期
    assert vlatex(v) == r'\dddot{\omega}\mathbf{\hat{n}_x}'
    # 断言 unicode_vpretty 输出符合预期
    assert unicode_vpretty(v) == 'ω⃛ n_x'
    # 断言 ascii_vpretty 输出符合预期
    assert ascii_vpretty(v) == "omega'''(t) n_x"

    # Fourth order
    v = omega.diff().diff().diff().diff() * N.x
    # 断言 vlatex 输出符合预期
    assert vlatex(v) == r'\ddddot{\omega}\mathbf{\hat{n}_x}'
    # 断言 unicode_vpretty 输出符合预期
    assert unicode_vpretty(v) == 'ω⃜ n_x'
    # 断言 ascii_vpretty 输出符合预期
    assert ascii_vpretty(v) == "omega''''(t) n_x"

    # Fifth order
    v = omega.diff().diff().diff().diff().diff() * N.x
    # 断言 vlatex 输出符合预期
    assert vlatex(v) == r'\frac{d^{5}}{d t^{5}} \omega\mathbf{\hat{n}_x}'
    # 预期的多行输出字符串
    expected = '''\
 5            \n\
d             \n\
---(omega) n_x\n\
  5           \n\
dt            \
'''
    # 断言 unicode_vpretty 输出符合预期
    uexpected = '''\
 5        \n\
d         \n\
───(ω) n_x\n\
  5       \n\
dt        \
'''
    assert unicode_vpretty(v) == uexpected
    # 断言 ascii_vpretty 输出符合预期
    assert ascii_vpretty(v) == expected


def test_vector_str_printing():
    # 断言 vsprint 输出符合预期
    assert vsprint(w) == 'alpha*N.x + sin(omega)*N.y + alpha*beta*N.z'
    # 断言 vsprint 输出符合预期
    assert vsprint(omega.diff() * N.x) == "omega'*N.x"
    # 断言 vsstrrepr 输出符合预期
    assert vsstrrepr(w) == 'alpha*N.x + sin(omega)*N.y + alpha*beta*N.z'


def test_vector_str_arguments():
    # 断言 vsprint 输出符合预期，使用 full_prec=False
    assert vsprint(N.x * 3.0, full_prec=False) == '3.0*N.x'
    # 断言 vsprint 输出符合预期，使用 full_prec=True
    assert vsprint(N.x * 3.0, full_prec=True) == '3.00000000000000*N.x'


def test_issue_14041():
    import sympy.physics.mechanics as me

    A_frame = me.ReferenceFrame('A')
    thetad, phid = me.dynamicsymbols('theta, phi', 1)
    L = symbols('L')

    # 断言 vlatex 输出符合预期
    assert vlatex(L*(phid + thetad)**2*A_frame.x) == \
        r"L \left(\dot{\phi} + \dot{\theta}\right)^{2}\mathbf{\hat{a}_x}"
    # 断言 vlatex 输出符合预期
    assert vlatex((phid + thetad)**2*A_frame.x) == \
        r"\left(\dot{\phi} + \dot{\theta}\right)^{2}\mathbf{\hat{a}_x}"
    # 断言 vlatex 输出符合预期
    assert vlatex((phid*thetad)**a*A_frame.x) == \
        r"\left(\dot{\phi} \dot{\theta}\right)^{a}\mathbf{\hat{a}_x}"
```