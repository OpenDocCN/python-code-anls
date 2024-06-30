# `D:\src\scipysrc\sympy\sympy\printing\pretty\tests\test_pretty.py`

```
# -*- coding: utf-8 -*-

# 导入 SymPy 库中的具体类和函数

from sympy.concrete.products import Product  # 导入 Product 类
from sympy.concrete.summations import Sum  # 导入 Sum 类
from sympy.core.add import Add  # 导入 Add 类
from sympy.core.basic import Basic  # 导入 Basic 类
from sympy.core.containers import (Dict, Tuple)  # 导入 Dict 和 Tuple 类
from sympy.core.function import (Derivative, Function, Lambda, Subs)  # 导入 Derivative、Function、Lambda 和 Subs 类
from sympy.core.mul import Mul  # 导入 Mul 类
from sympy.core import (EulerGamma, GoldenRatio, Catalan)  # 导入 EulerGamma、GoldenRatio 和 Catalan 常数
from sympy.core.numbers import (I, Rational, oo, pi)  # 导入 I、Rational、oo 和 pi 常数
from sympy.core.power import Pow  # 导入 Pow 类
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)  # 导入 Eq、Ge、Gt、Le、Lt 和 Ne 类
from sympy.core.singleton import S  # 导入 S 单例
from sympy.core.symbol import (Symbol, symbols)  # 导入 Symbol 和 symbols 函数
from sympy.functions.elementary.complexes import conjugate  # 导入 conjugate 函数
from sympy.functions.elementary.exponential import LambertW  # 导入 LambertW 函数
from sympy.functions.special.bessel import (airyai, airyaiprime, airybi, airybiprime)  # 导入 Bessel 函数
from sympy.functions.special.delta_functions import Heaviside  # 导入 Heaviside 函数
from sympy.functions.special.error_functions import (fresnelc, fresnels)  # 导入 Fresnel 函数
from sympy.functions.special.singularity_functions import SingularityFunction  # 导入 SingularityFunction 类
from sympy.functions.special.zeta_functions import dirichlet_eta  # 导入 dirichlet_eta 函数
from sympy.geometry.line import (Ray, Segment)  # 导入 Ray 和 Segment 类
from sympy.integrals.integrals import Integral  # 导入 Integral 类
from sympy.logic.boolalg import (And, Equivalent, ITE, Implies, Nand, Nor, Not, Or, Xor)  # 导入逻辑运算类和函数
from sympy.matrices.dense import (Matrix, diag)  # 导入 Matrix 和 diag 函数
from sympy.matrices.expressions.slice import MatrixSlice  # 导入 MatrixSlice 类
from sympy.matrices.expressions.trace import Trace  # 导入 Trace 类
from sympy.polys.domains.finitefield import FF  # 导入有限域类 FF
from sympy.polys.domains.integerring import ZZ  # 导入整数环类 ZZ
from sympy.polys.domains.rationalfield import QQ  # 导入有理数域类 QQ
from sympy.polys.domains.realfield import RR  # 导入实数域类 RR
from sympy.polys.orderings import (grlex, ilex)  # 导入多项式排序类 grlex 和 ilex
from sympy.polys.polytools import groebner  # 导入 groebner 函数
from sympy.polys.rootoftools import (RootSum, rootof)  # 导入 RootSum 和 rootof 类
from sympy.series.formal import fps  # 导入 formal power series 类
from sympy.series.fourier import fourier_series  # 导入 Fourier 级数类
from sympy.series.limits import Limit  # 导入 Limit 类
from sympy.series.order import O  # 导入 O 类
from sympy.series.sequences import (SeqAdd, SeqFormula, SeqMul, SeqPer)  # 导入序列操作类
from sympy.sets.contains import Contains  # 导入 Contains 类
from sympy.sets.fancysets import Range  # 导入 Range 类
from sympy.sets.sets import (Complement, FiniteSet, Intersection, Interval, Union)  # 导入集合操作类
from sympy.codegen.ast import (Assignment, AddAugmentedAssignment,
    SubAugmentedAssignment, MulAugmentedAssignment, DivAugmentedAssignment, ModAugmentedAssignment)  # 导入代码生成抽象语法树类
from sympy.core.expr import UnevaluatedExpr  # 导入 UnevaluatedExpr 类
from sympy.physics.quantum.trace import Tr  # 导入 Tr 类

from sympy.functions import (Abs, Chi, Ci, Ei, KroneckerDelta,
    Piecewise, Shi, Si, atan2, beta, binomial, catalan, ceiling, cos,
    euler, exp, expint, factorial, factorial2, floor, gamma, hyper, log,
    meijerg, sin, sqrt, subfactorial, tan, uppergamma, lerchphi, polylog,
    elliptic_k, elliptic_f, elliptic_e, elliptic_pi, DiracDelta, bell,
    bernoulli, fibonacci, tribonacci, lucas, stieltjes, mathieuc, mathieus,
    mathieusprime, mathieucprime)  # 导入各类数学函数
from sympy.matrices import (Adjoint, Inverse, MatrixSymbol, Transpose,
                            KroneckerProduct, BlockMatrix, OneMatrix, ZeroMatrix)
# 导入矩阵相关的类和函数

from sympy.matrices.expressions import hadamard_power
# 导入矩阵表达式中的哈达玛德幂函数

from sympy.physics import mechanics
# 导入物理力学模块

from sympy.physics.control.lti import (TransferFunction, Feedback, TransferFunctionMatrix,
    Series, Parallel, MIMOSeries, MIMOParallel, MIMOFeedback, StateSpace)
# 导入控制理论中的线性时不变系统相关类和函数

from sympy.physics.units import joule, degree
# 导入物理单位 joule 和 degree

from sympy.printing.pretty import pprint, pretty as xpretty
# 导入 pretty 模块中的 pprint 和 xpretty 函数

from sympy.printing.pretty.pretty_symbology import center_accent, is_combining, center
# 导入 pretty 模块中的符号相关函数

from sympy.sets.conditionset import ConditionSet
# 导入条件集合模块中的 ConditionSet 类

from sympy.sets import ImageSet, ProductSet
# 导入集合模块中的 ImageSet 和 ProductSet 类

from sympy.sets.setexpr import SetExpr
# 导入集合表达式模块中的 SetExpr 类

from sympy.stats.crv_types import Normal
# 导入概率统计中的正态分布类 Normal

from sympy.stats.symbolic_probability import (Covariance, Expectation,
                                              Probability, Variance)
# 导入符号概率统计中的协方差、期望、概率和方差相关类

from sympy.tensor.array import (ImmutableDenseNDimArray, ImmutableSparseNDimArray,
                                MutableDenseNDimArray, MutableSparseNDimArray, tensorproduct)
# 导入张量数组相关类和函数

from sympy.tensor.functions import TensorProduct
# 导入张量函数中的 TensorProduct 类

from sympy.tensor.tensor import (TensorIndexType, tensor_indices, TensorHead,
                                 TensorElement, tensor_heads)
# 导入张量相关类和函数

from sympy.testing.pytest import raises, _both_exp_pow, warns_deprecated_sympy
# 导入测试相关函数和变量

from sympy.vector import CoordSys3D, Gradient, Curl, Divergence, Dot, Cross, Laplacian
# 导入向量分析相关类

import sympy as sym
# 导入 sympy 库并将其命名为 sym

class lowergamma(sym.lowergamma):
    pass   # 通过具有相同名称的子类测试符号继承的符号

a, b, c, d, x, y, z, k, n, s, p = symbols('a,b,c,d,x,y,z,k,n,s,p')
# 定义符号变量 a, b, c, d, x, y, z, k, n, s, p

f = Function("f")
# 创建一个以 "f" 为名称的函数符号

th = Symbol('theta')
ph = Symbol('phi')
# 创建表示角度的符号 th 和 ph
# 计算表达式 (1 + sqrt(5))**Rational(1,3)，返回结果
(1 + sqrt(5))**Rational(1,3)

# 计算表达式 2**(1/x)，返回结果
2**(1/x)

# 计算表达式 sqrt(2+pi)，返回结果
sqrt(2+pi)

# 计算复杂的数学表达式 (2+(1+x**2)/(2+x))**Rational(1,4)+(1+x**Rational(1,1000))/sqrt(3+x**2)，返回结果
(2+(1+x**2)/(2+x))**Rational(1,4)+(1+x**Rational(1,1000))/sqrt(3+x**2)


# DERIVATIVES:

# 求对数函数 log(x) 对 x 的导数，不进行求值
Derivative(log(x), x, evaluate=False)

# 求对数函数 log(x) 对 x 的导数，不进行求值，然后加上 x
Derivative(log(x), x, evaluate=False) + x

# 求 log(x) + x**2 对 x 和 y 的偏导数，不进行求值
Derivative(log(x) + x**2, x, y, evaluate=False)

# 求 2*x*y 对 y 和 x 的偏导数，不进行求值，然后加上 x**2
Derivative(2*x*y, y, x, evaluate=False) + x**2

# 对 beta(alpha) 求 alpha 的导数
beta(alpha).diff(alpha)


# INTEGRALS:

# 求 log(x) 对 x 的积分
Integral(log(x), x)

# 求 x**2 对 x 的积分
Integral(x**2, x)

# 求 (sin(x))**2 / (tan(x))**2 的积分
Integral((sin(x))**2 / (tan(x))**2)

# 求 x**(2**x) 对 x 的积分
Integral(x**(2**x), x)

# 求 x**2 对 x 从 1 到 2 的定积分
Integral(x**2, (x,1,2))

# 求 x**2 对 x 从 1/2 到 10 的定积分
Integral(x**2, (x,Rational(1,2),10))

# 求 x**2*y**2 对 x 和 y 的二重积分
Integral(x**2*y**2, x,y)

# 求 x**2 对 x 从负无穷到 1 的不定积分
Integral(x**2, (x, None, 1))

# 求 x**2 对 x 从 1 到正无穷的不定积分
Integral(x**2, (x, 1, None))

# 求 sin(th)/cos(ph) 对 th 从 0 到 pi，对 ph 从 0 到 2*pi 的二重积分
Integral(sin(th)/cos(ph), (th,0,pi), (ph, 0, 2*pi))


# MATRICES:

# 创建一个矩阵，第一行是 [x**2+1, 1]，第二行是 [y, x+y]
Matrix([[x**2+1, 1], [y, x+y]])

# 创建一个矩阵，第一行是 [x/y, y, th]，第二行是 [0, exp(I*k*ph), 1]
Matrix([[x/y, y, th], [0, exp(I*k*ph), 1]])


# PIECEWISE:

# 创建一个分段函数 Piecewise((x,x<1),(x**2,True))


# ITE:

# 创建一个 ITE 表达式 ITE(x, y, z)


# SEQUENCES (TUPLES, LISTS, DICTIONARIES):

# 创建一个空元组 ()
()

# 创建一个空列表 []
[]

# 创建一个空字典 {}
{}

# 创建一个只包含一个元素 1/x 的元组 (1/x,)
(1/x,)

# 创建一个列表，包含 x**2, 1/x, x, y, sin(th)**2/cos(ph)**2
[x**2, 1/x, x, y, sin(th)**2/cos(ph)**2]

# 创建一个元组，包含 x**2, 1/x, x, y, sin(th)**2/cos(ph)**2
(x**2, 1/x, x, y, sin(th)**2/cos(ph)**2)

# 创建一个字典 {x: sin(x)}
{x: sin(x)}

# 创建一个字典 {1/x: 1/y, x: sin(x)**2}
{1/x: 1/y, x: sin(x)**2}

# 创建一个只包含 x**2 的列表 [x**2]
[x**2]

# 创建一个只包含 x**2 的元组 (x**2,)
(x**2,)

# 创建一个字典 {x**2: 1}
{x**2: 1}


# LIMITS:

# 计算极限 Limit(x, x, oo)
Limit(x, x, oo)

# 计算极限 Limit(x**2, x, 0)
Limit(x**2, x, 0)

# 计算极限 Limit(1/x, x, 0)
Limit(1/x, x, 0)

# 计算极限 Limit(sin(x)/x, x, 0)
Limit(sin(x)/x, x, 0)


# UNITS:

# 1焦耳 = 1千克*米^2/秒
joule => kg*m**2/s


# SUBS:

# 将 f(x) 中的 x 替换为 ph**2
Subs(f(x), x, ph**2)

# 将 f(x) 的导数关于 x 替换为 x=0
Subs(f(x).diff(x), x, 0)

# 将 f(x) 的导数关于 x 和 y 替换为 x=0, y=1/2
Subs(f(x).diff(x)/y, (x, y), (0, Rational(1, 2)))


# ORDER:

# 创建一个 O 大O符号 O(1)
O(1)

# 创建一个 O 大O符号 O(1/x)
O(1/x)

# 创建一个 O 大O符号 O(x**2 + y**2)
O(x**2 + y**2)
    # 确保符号的漂亮打印格式正确
    assert upretty( Symbol('beta_1_2') ) == 'β₁ ₂'
    # 确保符号的漂亮打印格式正确，包含上标
    assert upretty( Symbol('beta^1^2') ) == 'β¹ ²'
    # 确保符号的漂亮打印格式正确，包含下标和上标的组合
    assert upretty( Symbol('beta_1^2') ) == 'β²₁'
    # 确保符号的漂亮打印格式正确，包含多位数的下标和上标
    assert upretty( Symbol('beta_10_20') ) == 'β₁₀ ₂₀'
    # 确保符号的漂亮打印格式正确，包含带有上标和下标的组合
    assert upretty( Symbol('beta_ax_gamma^i') ) == 'βⁱₐₓ ᵧ'
    # 确保符号的漂亮打印格式正确，包含上标和多个下标的组合
    assert upretty( Symbol("F^1^2_3_4") ) == 'F¹ ²₃ ₄'
    # 确保符号的漂亮打印格式正确，包含下标和多个上标的组合
    assert upretty( Symbol("F_1_2^3^4") ) == 'F³ ⁴₁ ₂'
    # 确保符号的漂亮打印格式正确，包含多个下标和上标的组合
    assert upretty( Symbol("F_1_2_3_4") ) == 'F₁ ₂₃ ₄'
    # 确保符号的漂亮打印格式正确，包含多个上标和下标的组合
    assert upretty( Symbol("F^1^2^3^4") ) == 'F¹ ²³ ⁴'
def test_upretty_subs_missing_in_24():
    # 检查 upretty 函数对于特定希腊字母符号的输出是否符合预期
    assert upretty( Symbol('F_beta') ) == 'Fᵦ'
    assert upretty( Symbol('F_gamma') ) == 'Fᵧ'
    assert upretty( Symbol('F_rho') ) == 'Fᵨ'
    assert upretty( Symbol('F_phi') ) == 'Fᵩ'
    assert upretty( Symbol('F_chi') ) == 'Fᵪ'

    # 检查 upretty 函数对于特定下标符号的输出是否符合预期
    assert upretty( Symbol('F_a') ) == 'Fₐ'
    assert upretty( Symbol('F_e') ) == 'Fₑ'
    assert upretty( Symbol('F_i') ) == 'Fᵢ'
    assert upretty( Symbol('F_o') ) == 'Fₒ'
    assert upretty( Symbol('F_u') ) == 'Fᵤ'
    assert upretty( Symbol('F_r') ) == 'Fᵣ'
    assert upretty( Symbol('F_v') ) == 'Fᵥ'
    assert upretty( Symbol('F_x') ) == 'Fₓ'


def test_missing_in_2X_issue_9047():
    # 检查 upretty 函数对于特定下标符号的输出是否符合预期（第二批测试）
    assert upretty( Symbol('F_h') ) == 'Fₕ'
    assert upretty( Symbol('F_k') ) == 'Fₖ'
    assert upretty( Symbol('F_l') ) == 'Fₗ'
    assert upretty( Symbol('F_m') ) == 'Fₘ'
    assert upretty( Symbol('F_n') ) == 'Fₙ'
    assert upretty( Symbol('F_p') ) == 'Fₚ'
    assert upretty( Symbol('F_s') ) == 'Fₛ'
    assert upretty( Symbol('F_t') ) == 'Fₜ'


def test_upretty_modifiers():
    # 测试 upretty 函数对于不同符号修饰的输出是否符合预期
    # Accent 修饰
    assert upretty( Symbol('Fmathring') ) == 'F̊'
    assert upretty( Symbol('Fddddot') ) == 'F⃜'
    assert upretty( Symbol('Fdddot') ) == 'F⃛'
    assert upretty( Symbol('Fddot') ) == 'F̈'
    assert upretty( Symbol('Fdot') ) == 'Ḟ'
    assert upretty( Symbol('Fcheck') ) == 'F̌'
    assert upretty( Symbol('Fbreve') ) == 'F̆'
    assert upretty( Symbol('Facute') ) == 'F́'
    assert upretty( Symbol('Fgrave') ) == 'F̀'
    assert upretty( Symbol('Ftilde') ) == 'F̃'
    assert upretty( Symbol('Fhat') ) == 'F̂'
    assert upretty( Symbol('Fbar') ) == 'F̅'
    assert upretty( Symbol('Fvec') ) == 'F⃗'
    assert upretty( Symbol('Fprime') ) == 'F′'
    assert upretty( Symbol('Fprm') ) == 'F′'
    # 测试无实际定义的面字体，确保修饰符被正确去除
    assert upretty( Symbol('Fbold') ) == 'Fbold'
    assert upretty( Symbol('Fbm') ) == 'Fbm'
    assert upretty( Symbol('Fcal') ) == 'Fcal'
    assert upretty( Symbol('Fscr') ) == 'Fscr'
    assert upretty( Symbol('Ffrak') ) == 'Ffrak'
    # Bracket 修饰
    assert upretty( Symbol('Fnorm') ) == '‖F‖'
    assert upretty( Symbol('Favg') ) == '⟨F⟩'
    assert upretty( Symbol('Fabs') ) == '|F|'
    assert upretty( Symbol('Fmag') ) == '|F|'
    # 组合修饰
    assert upretty( Symbol('xvecdot') ) == 'x⃗̇'
    assert upretty( Symbol('xDotVec') ) == 'ẋ⃗'
    assert upretty( Symbol('xHATNorm') ) == '‖x̂‖'
    assert upretty( Symbol('xMathring_yCheckPRM__zbreveAbs') ) == 'x̊_y̌′__|z̆|'
    assert upretty( Symbol('alphadothat_nVECDOT__tTildePrime') ) == 'α̇̂_n⃗̇__t̃′'
    assert upretty( Symbol('x_dot') ) == 'x_dot'
    assert upretty( Symbol('x__dot') ) == 'x__dot'


def test_pretty_Cycle():
    from sympy.combinatorics.permutations import Cycle
    # 检查 pretty 函数对于 Cycle 对象的输出是否符合预期
    assert pretty(Cycle(1, 2)) == '(1 2)'
    assert pretty(Cycle(2)) == '(2)'
    assert pretty(Cycle(1, 3)(4, 5)) == '(1 3)(4 5)'
    assert pretty(Cycle()) == '()'
# 定义一个测试函数，用于测试 Permutation 类的 pretty 输出函数
def test_pretty_Permutation():
    # 导入 Permutation 类
    from sympy.combinatorics.permutations import Permutation
    # 创建一个置换对象 p1，表示置换 (1 2)(3 4)
    p1 = Permutation(1, 2)(3, 4)
    # 测试 xpretty 函数输出置换 p1 的循环表示，使用 Unicode 格式
    assert xpretty(p1, perm_cyclic=True, use_unicode=True) == "(1 2)(3 4)"
    # 测试 xpretty 函数输出置换 p1 的循环表示，不使用 Unicode 格式
    assert xpretty(p1, perm_cyclic=True, use_unicode=False) == "(1 2)(3 4)"
    # 测试 xpretty 函数输出置换 p1 的表格表示，不使用循环格式和 Unicode
    assert xpretty(p1, perm_cyclic=False, use_unicode=True) == \
    '⎛0 1 2 3 4⎞\n'\
    '⎝0 2 1 4 3⎠'
    # 测试 xpretty 函数输出置换 p1 的表格表示，不使用循环格式和 Unicode
    assert xpretty(p1, perm_cyclic=False, use_unicode=False) == \
    "/0 1 2 3 4\\\n"\
    "\\0 2 1 4 3/"

    # 开始测试 deprecated 符号的警告
    with warns_deprecated_sympy():
        # 保存旧的 print_cyclic 属性
        old_print_cyclic = Permutation.print_cyclic
        # 设置 Permutation 类的 print_cyclic 属性为 False
        Permutation.print_cyclic = False
        # 测试 xpretty 函数输出置换 p1 的表格表示，使用 Unicode 格式
        assert xpretty(p1, use_unicode=True) == \
        '⎛0 1 2 3 4⎞\n'\
        '⎝0 2 1 4 3⎠'
        # 测试 xpretty 函数输出置换 p1 的表格表示，不使用 Unicode 格式
        assert xpretty(p1, use_unicode=False) == \
        "/0 1 2 3 4\\\n"\
        "\\0 2 1 4 3/"
        # 恢复旧的 print_cyclic 属性
        Permutation.print_cyclic = old_print_cyclic


# 定义一个测试函数，测试数学表达式的 pretty 和 upretty 函数
def test_pretty_basic():
    # 测试数学表达式 -1/2 的 ASCII 格式输出
    assert pretty( -Rational(1)/2 ) == '-1/2'
    # 测试数学表达式 -13/22 的 ASCII 格式输出
    assert pretty( -Rational(13)/22 ) == \
"""\
-13 \n\
----\n\
 22 \
"""
    # 测试无穷大符号的 ASCII 格式输出
    expr = oo
    ascii_str = \
"""\
oo\
"""
    # 测试无穷大符号的 Unicode 格式输出
    ucode_str = \
"""\
∞\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    # 测试表达式 x**2 的 ASCII 和 Unicode 格式输出
    expr = (x**2)
    ascii_str = \
"""\
 2\n\
x \
"""
    ucode_str = \
"""\
 2\n\
x \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    # 测试表达式 1/x 的 ASCII 和 Unicode 格式输出
    expr = 1/x
    ascii_str = \
"""\
1\n\
-\n\
x\
"""
    ucode_str = \
"""\
1\n\
─\n\
x\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    # 测试表达式 x**-1.0 的 ASCII 和 Unicode 格式输出
    expr = x**-1.0
    ascii_str = \
"""\
 -1.0\n\
x    \
"""
    ucode_str = \
"""\
 -1.0\n\
x    \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    # 测试表达式 Pow(S(2), -1.0, evaluate=False) 的 ASCII 和 Unicode 格式输出
    expr = Pow(S(2), -1.0, evaluate=False)
    ascii_str = \
"""\
 -1.0\n\
2    \
"""
    ucode_str = \
"""\
 -1.0\n\
2    \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    # 测试表达式 y*x**-2 的 ASCII 和 Unicode 格式输出
    expr = y*x**-2
    ascii_str = \
"""\
y \n\
--\n\
 2\n\
x \
"""
    ucode_str = \
"""\
y \n\
──\n\
 2\n\
x \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    # 测试表达式 x**Rational(1, 3) 的 ASCII 和 Unicode 格式输出
    expr = x**Rational(1, 3)
    ascii_str = \
"""\
 1/3\n\
x   \
"""
    ucode_str = \
"""\
 1/3\n\
x   \
"""
    assert xpretty(expr, use_unicode=False, wrap_line=False,\
    root_notation = False) == ascii_str
    assert xpretty(expr, use_unicode=True, wrap_line=False,\
    root_notation = False) == ucode_str

    # 测试表达式 x**Rational(-5, 2) 的 ASCII 和 Unicode 格式输出
    expr = x**Rational(-5, 2)
    ascii_str = \
"""\
 1  \n\
----\n\
 5/2\n\
x   \
"""
    ucode_str = \
"""\
 1  \n\
────\n\
 5/2\n\
x   \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    # 测试表达式 (-2)**x 的 ASCII 和 Unicode 格式输出
    expr = (-2)**x
    ascii_str = \
"""\
    x\n\
(-2) \
"""
    ucode_str = \
"""\
    x\n\
(-2) \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    # 测试表达式 Pow(3, 1, evaluate=False) 的 ASCII 和 Unicode 格式输出
    expr = Pow(3, 1, evaluate=False)
    ascii_str = \
"""\
 1\n\
3 \
"""
    ucode_str = \
"""\
 1\n\
3 \
"""
    # 断言表达式在漂亮打印后与 ASCII 字符串相等
    assert pretty(expr) == ascii_str
    # 断言表达式在漂亮 Unicode 打印后与 Unicode 字符串相等
    assert upretty(expr) == ucode_str
    
    # 定义一个新的数学表达式，这里是 x 的二次方加 x 加 1
    expr = (x**2 + x + 1)
    # 继续定义一个 ASCII 字符串，以反斜杠结尾，表示表达式的 ASCII 格式
    ascii_str_1 = \
"""\
         2\n\
1 + x + x \
"""
# ASCII字符串表示式1
ascii_str_2 = \
"""\
 2        \n\
x  + x + 1\
"""
# ASCII字符串表示式2
ascii_str_3 = \
"""\
 2        \n\
x  + 1 + x\
"""
# ASCII字符串表示式3
ucode_str_1 = \
"""\
         2\n\
1 + x + x \
"""
# Unicode字符串表示式1
ucode_str_2 = \
"""\
 2        \n\
x  + x + 1\
"""
# Unicode字符串表示式2
ucode_str_3 = \
"""\
 2        \n\
x  + 1 + x\
"""
# Unicode字符串表示式3

# 断言测试pretty函数输出与ASCII字符串表示式匹配
assert pretty(expr) in [ascii_str_1, ascii_str_2, ascii_str_3]
# 断言测试upretty函数输出与Unicode字符串表示式匹配
assert upretty(expr) in [ucode_str_1, ucode_str_2, ucode_str_3]

expr = 1 - x
ascii_str_1 = \
"""\
1 - x\
"""
# ASCII字符串表示式1
ascii_str_2 = \
"""\
-x + 1\
"""
# ASCII字符串表示式2
ucode_str_1 = \
"""\
1 - x\
"""
# Unicode字符串表示式1
ucode_str_2 = \
"""\
-x + 1\
"""
# Unicode字符串表示式2

# 断言测试pretty函数输出与ASCII字符串表示式匹配
assert pretty(expr) in [ascii_str_1, ascii_str_2]
# 断言测试upretty函数输出与Unicode字符串表示式匹配
assert upretty(expr) in [ucode_str_1, ucode_str_2]

expr = 1 - 2*x
ascii_str_1 = \
"""\
1 - 2*x\
"""
# ASCII字符串表示式1
ascii_str_2 = \
"""\
-2*x + 1\
"""
# ASCII字符串表示式2
ucode_str_1 = \
"""\
1 - 2⋅x\
"""
# Unicode字符串表示式1
ucode_str_2 = \
"""\
-2⋅x + 1\
"""
# Unicode字符串表示式2

# 断言测试pretty函数输出与ASCII字符串表示式匹配
assert pretty(expr) in [ascii_str_1, ascii_str_2]
# 断言测试upretty函数输出与Unicode字符串表示式匹配
assert upretty(expr) in [ucode_str_1, ucode_str_2]

expr = x/y
ascii_str = \
"""\
x\n\
-\n\
y\
"""
# ASCII字符串表示式
ucode_str = \
"""\
x\n\
─\n\
y\
"""
# Unicode字符串表示式

# 断言测试pretty函数输出与ASCII字符串表示式匹配
assert pretty(expr) == ascii_str
# 断言测试upretty函数输出与Unicode字符串表示式匹配
assert upretty(expr) == ucode_str

expr = -x/y
ascii_str = \
"""\
-x \n\
---\n\
 y \
"""
# ASCII字符串表示式
ucode_str = \
"""\
-x \n\
───\n\
 y \
"""
# Unicode字符串表示式

# 断言测试pretty函数输出与ASCII字符串表示式匹配
assert pretty(expr) == ascii_str
# 断言测试upretty函数输出与Unicode字符串表示式匹配
assert upretty(expr) == ucode_str

expr = (x + 2)/y
ascii_str_1 = \
"""\
2 + x\n\
-----\n\
  y  \
"""
# ASCII字符串表示式1
ascii_str_2 = \
"""\
x + 2\n\
-----\n\
  y  \
"""
# ASCII字符串表示式2
ucode_str_1 = \
"""\
2 + x\n\
─────\n\
  y  \
"""
# Unicode字符串表示式1
ucode_str_2 = \
"""\
x + 2\n\
─────\n\
  y  \
"""
# Unicode字符串表示式2

# 断言测试pretty函数输出与ASCII字符串表示式匹配
assert pretty(expr) in [ascii_str_1, ascii_str_2]
# 断言测试upretty函数输出与Unicode字符串表示式匹配
assert upretty(expr) in [ucode_str_1, ucode_str_2]

expr = (1 + x)*y
ascii_str_1 = \
"""\
y*(1 + x)\
"""
# ASCII字符串表示式1
ascii_str_2 = \
"""\
(1 + x)*y\
"""
# ASCII字符串表示式2
ascii_str_3 = \
"""\
y*(x + 1)\
"""
# ASCII字符串表示式3
ucode_str_1 = \
"""\
y⋅(1 + x)\
"""
# Unicode字符串表示式1
ucode_str_2 = \
"""\
(1 + x)⋅y\
"""
# Unicode字符串表示式2
ucode_str_3 = \
"""\
y⋅(x + 1)\
"""
# Unicode字符串表示式3

# 断言测试pretty函数输出与ASCII字符串表示式匹配
assert pretty(expr) in [ascii_str_1, ascii_str_2, ascii_str_3]
# 断言测试upretty函数输出与Unicode字符串表示式匹配
assert upretty(expr) in [ucode_str_1, ucode_str_2, ucode_str_3]

# 测试负号位置是否正确
expr = -5*x/(x + 10)
ascii_str_1 = \
"""\
-5*x  \n\
------\n\
10 + x\
"""
# ASCII字符串表示式1
ascii_str_2 = \
"""\
-5*x  \n\
------\n\
x + 10\
"""
# ASCII字符串表示式2
ucode_str_1 = \
"""\
-5⋅x  \n\
──────\n\
10 + x\
"""
# Unicode字符串表示式1
ucode_str_2 = \
"""\
-5⋅x  \n\
──────\n\
x + 10\
"""
# Unicode字符串表示式2

# 断言测试pretty函数输出与ASCII字符串表示式匹配
assert pretty(expr) in [ascii_str_1, ascii_str_2]
# 断言测试upretty函数输出与Unicode字符串表示式匹配
assert upretty(expr) in [ucode_str_1, ucode_str_2]

expr = -S.Half - 3*x
ascii_str = \
"""\
-3*x - 1/2\
"""
# ASCII字符串表示式
ucode_str = \
"""\
-3⋅x - 1/2\
"""
# Unicode字符串表示式

# 断言测试pretty函数输出与ASCII字符串表示式匹配
assert pretty(expr) == ascii_str
# 断言测试upretty函数输出与Unicode字符串表示式匹配
assert upretty(expr) == ucode_str

expr = S.Half - 3*x
ascii_str = \
"""\
1/2 - 3*x\
"""
# ASCII字符串表示式
ucode_str = \
"""\
1/2 - 3⋅x\
"""
# Unicode字符串表示式

# 断言测试pretty函数输出与ASCII字符串表示式匹配
assert pretty(expr) == ascii_str
# 断言测试upretty函数输出与Unicode字符串表示式匹配
assert upretty(expr) == ucode_str
"""
    expr = S.Half - 3*x/2
    ascii_str = \
"""\
1   3*x\n\
- - ---\n\
2    2 \
"""
    ucode_str = \
"""\
1   3⋅x\n\
─ - ───\n\
2    2 \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
"""

"""
    expr = -x/y
    ascii_str =\
"""\
-x \n\
---\n\
 y \
"""
    ucode_str =\
"""\
-x \n\
───\n\
 y \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = -x*z/y
    ascii_str =\
"""\
-x*z \n\
-----\n\
  y  \
"""
    ucode_str =\
"""\
-x⋅z \n\
─────\n\
  y  \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = x**2/y
    ascii_str =\
"""\
 2\n\
x \n\
--\n\
y \
"""
    ucode_str =\
"""\
 2\n\
x \n\
──\n\
y \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = -x**2/y
    ascii_str =\
"""\
  2 \n\
-x  \n\
----\n\
 y  \
"""
    ucode_str =\
"""\
  2 \n\
-x  \n\
────\n\
 y  \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = -x/(y*z)
    ascii_str =\
"""\
-x \n\
---\n\
y*z\
"""
    ucode_str =\
"""\
-x \n\
───\n\
y⋅z\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = -a/y**2
    ascii_str =\
"""\
-a \n\
---\n\
 2 \n\
y  \
"""
    ucode_str =\
"""\
-a \n\
───\n\
 2 \n\
y  \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = y**(-a/b)
    ascii_str =\
"""\
 -a \n\
 ---\n\
  b \n\
y   \
"""
    ucode_str =\
"""\
 -a \n\
 ───\n\
  b \n\
y   \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = -1/y**2
    ascii_str =\
"""\
-1 \n\
---\n\
 2 \n\
y  \
"""
    ucode_str =\
"""\
-1 \n\
───\n\
 2 \n\
y  \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = -10/b**2
    ascii_str =\
"""\
-10 \n\
----\n\
  2 \n\
 b  \
"""
    ucode_str =\
"""\
-10 \n\
────\n\
  2 \n\
 b  \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Rational(-200, 37)
    ascii_str =\
"""\
-200 \n\
-----\n\
 37  \
"""
    ucode_str =\
"""\
-200 \n\
─────\n\
 37  \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    # 创建一个乘法表达式 Mul，包含参数 1, 1, 2，且不进行求值
    expr = Mul(1, 1, 2, evaluate=False)
    # 断言 pretty 函数处理后的结果为 "1*1*2"
    assert pretty(expr) == "1*1*2"
    # 断言 upretty 函数处理后的结果为 "1⋅1⋅2"
    assert upretty(expr) == "1⋅1⋅2"

    # 创建一个加法表达式 Add，包含参数 0, 0, 1，且不进行求值
    expr = Add(0, 0, 1, evaluate=False)
    # 断言 pretty 函数处理后的结果为 "0 + 0 + 1"
    assert pretty(expr) == "0 + 0 + 1"
    # 断言 upretty 函数处理后的结果为 "0 + 0 + 1"
    assert upretty(expr) == "0 + 0 + 1"

    # 创建一个乘法表达式 Mul，包含参数 1, -1，且不进行求值
    expr = Mul(1, -1, evaluate=False)
    # 断言 pretty 函数处理后的结果为 "1*-1"
    assert pretty(expr) == "1*-1"
    # 断言 upretty 函数处理后的结果为 "1⋅-1"
    assert upretty(expr) == "1⋅-1"

    # 创建一个乘法表达式 Mul，包含参数 1.0, x，且不进行求值
    expr = Mul(1.0, x, evaluate=False)
    # 断言 pretty 函数处理后的结果为 "1.0*x"
    assert pretty(expr) == "1.0*x"
    # 断言 upretty 函数处理后的结果为 "1.0⋅x"
    assert upretty(expr) == "1.0⋅x"

    # 创建一个乘法表达式 Mul，包含参数 1, 1, 2, 3, x，且不进行求值
    expr = Mul(1, 1, 2, 3, x, evaluate=False)
    # 断言 pretty 函数处理后的结果为 "1*1*2*3*x"
    assert pretty(expr) == "1*1*2*3*x"
    # 断言 upretty 函数处理后的结果为 "1⋅1⋅2⋅3⋅x"
    assert upretty(expr) == "1⋅1⋅2⋅3⋅x"

    # 创建一个乘法表达式 Mul，包含参数 -1, 1，且不进行求值
    expr = Mul(-1, 1, evaluate=False)
    # 断言 pretty 函数处理后的结果为 "-1*1"
    assert pretty(expr) == "-1*1"
    # 断言 upretty 函数处理后的结果为 "-1⋅1"
    assert upretty(expr) == "-1⋅1"

    # 创建一个乘法表达式 Mul，包含参数 4, 3, 2, 1, 0, y, x，且不进行求值
    expr = Mul(4, 3, 2, 1, 0, y, x, evaluate=False)
    # 断言 pretty 函数处理后的结果为 "4*3*2*1*0*y*x"
    assert pretty(expr) == "4*3*2*1*0*y*x"
    # 断言 upretty 函数处理后的结果为 "4⋅3⋅2⋅1⋅0⋅y⋅x"
    assert upretty(expr) == "4⋅3⋅2⋅1⋅0⋅y⋅x"

    # 创建一个乘法表达式 Mul，包含参数 4, 3, 2, 1+z, 0, y, x，且不进行求值
    expr = Mul(4, 3, 2, 1+z, 0, y, x, evaluate=False)
    # 断言 pretty 函数处理后的结果为 "4*3*2*(z + 1)*0*y*x"
    assert pretty(expr) == "4*3*2*(z + 1)*0*y*x"
    # 断言 upretty 函数处理后的结果为 "4⋅3⋅2⋅(z + 1)⋅0⋅y⋅x"
    assert upretty(expr) == "4⋅3⋅2⋅(z + 1)⋅0⋅y⋅x"

    # 创建一个乘法表达式 Mul，包含参数 Rational(2, 3), Rational(5, 7)，且不进行求值
    expr = Mul(Rational(2, 3), Rational(5, 7), evaluate=False)
    # 断言 pretty 函数处理后的结果为 "2/3*5/7"
    assert pretty(expr) == "2/3*5/7"
    # 断言 upretty 函数处理后的结果为 "2/3⋅5/7"
    assert upretty(expr) == "2/3⋅5/7"

    # 创建一个乘法表达式 Mul，包含参数 x + y, Rational(1, 2)，且不进行求值
    expr = Mul(x + y, Rational(1, 2), evaluate=False)
    # 断言 pretty 函数处理后的结果为 "(x + y)*1/2"
    assert pretty(expr) == "(x + y)*1/2"
    # 断言 upretty 函数处理后的结果为 "(x + y)⋅1/2"
    assert upretty(expr) == "(x + y)⋅1/2"

    # 创建一个乘法表达式 Mul，包含参数 Rational(1, 2), x + y，且不进行求值
    expr = Mul(Rational(1, 2), x + y, evaluate=False)
    # 断言 pretty 函数处理后的结果为 "x + y\n-----\n  2  "
    assert pretty(expr) == "x + y\n-----\n  2  "
    # 断言 upretty 函数处理后的结果为 "x + y\n─────\n  2  "
    assert upretty(expr) == "x + y\n─────\n  2  "

    # 创建一个乘法表达式 Mul，包含参数 S.One, x + y，且不进行求值
    expr = Mul(S.One, x + y, evaluate=False)
    # 断言 pretty 函数处理后的结果为 "1*(x + y)"
    assert pretty(expr) == "1*(x + y)"
    # 断言 upretty 函数处理后的结果为 "1⋅(x + y)"
    assert upretty(expr) == "1⋅(x + y)"

    # 创建一个乘法表达式 Mul，包含参数 x - y, S.One，且不进行求值
    expr = Mul(x - y, S.One, evaluate=False)
    # 断言 pretty 函数处理后的结果为 "(x - y)*1"
    assert pretty(expr) == "(x - y)*1"
    # 断言 upretty 函数处理后的结果为 "(x - y)⋅1"
    assert upretty(expr) == "(x - y)⋅1"

    # 创建一个乘法表达式 Mul，包含参数 Rational(1, 2), x - y, S.One, x + y，且不进行求值
    expr = Mul(Rational(1, 2), x - y, S.One, x + y, evaluate=False)
    # 断言 pretty 函数处理后的结果为 "1/2*(x - y)*1*(x + y)"
    assert pretty(expr) == "1/2*(x - y)*1*(x + y)"
    # 断言 upretty 函数处理后的结果为 "1/2⋅(x - y)⋅1⋅(x + y)"
    assert upretty(expr) == "1/2⋅(x - y)⋅1⋅(x + y)"

    # 创建一个乘法表达式 Mul，包含参数 x + y, Rational(3, 4), S.One, y - z，且不进行求值
    expr = Mul(x + y, Rational(3, 4), S.One, y - z, evaluate=False)
    # 断言 pretty 函数处理后的结果为 "(x + y)*3/4*1*(y - z)"
def test_issue_5524():
    # 断言测试 pretty 函数对表达式的美化输出
    assert pretty(-(-x + 5)*(-x - 2*sqrt(2) + 5) - (-y + 5)*(-y + 5)) == \
"""\
         2           /         ___    \\\n\
- (5 - y)  + (x - 5)*\\-x - 2*\\/ 2  + 5/\
"""

    # 断言测试 upretty 函数对表达式的 Unicode 美化输出
    assert upretty(-(-x + 5)*(-x - 2*sqrt(2) + 5) - (-y + 5)*(-y + 5)) == \
"""\
         2                          \n\
- (5 - y)  + (x - 5)⋅(-x - 2⋅√2 + 5)\
"""


def test_pretty_ordering():
    # 断言测试 pretty 函数在不同排序模式下的输出
    assert pretty(x**2 + x + 1, order='lex') == \
"""\
 2        \n\
x  + x + 1\
"""
    assert pretty(x**2 + x + 1, order='rev-lex') == \
"""\
         2\n\
1 + x + x \
"""
    assert pretty(1 - x, order='lex') == '-x + 1'
    assert pretty(1 - x, order='rev-lex') == '1 - x'

    assert pretty(1 - 2*x, order='lex') == '-2*x + 1'
    assert pretty(1 - 2*x, order='rev-lex') == '1 - 2*x'

    # 测试 pretty 函数对复杂表达式的输出
    f = 2*x**4 + y**2 - x**2 + y**3
    assert pretty(f, order=None) == \
"""\
   4    2    3    2\n\
2*x  - x  + y  + y \
"""
    assert pretty(f, order='lex') == \
"""\
   4    2    3    2\n\
2*x  - x  + y  + y \
"""
    assert pretty(f, order='rev-lex') == \
"""\
 2    3    2      4\n\
y  + y  - x  + 2*x \
"""

    # 测试 pretty 函数对级数展开表达式的 ASCII 输出
    expr = x - x**3/6 + x**5/120 + O(x**6)
    ascii_str = \
"""\
     3    5         \n\
    x    x      / 6\\\n\
x - -- + --- + O\\x /\n\
    6    120        \
"""
    assert pretty(expr, order=None) == ascii_str

    # 测试 upretty 函数对级数展开表达式的 Unicode 输出
    ucode_str = \
"""\
     3    5         \n\
    x    x      ⎛ 6⎞\n\
x - ── + ─── + O⎝x ⎠\n\
    6    120        \
"""
    assert upretty(expr, order=None) == ucode_str


def test_EulerGamma():
    # 断言测试 pretty 函数对 EulerGamma 符号的输出
    assert pretty(EulerGamma) == str(EulerGamma) == "EulerGamma"
    assert upretty(EulerGamma) == "γ"


def test_GoldenRatio():
    # 断言测试 pretty 函数对 GoldenRatio 符号的输出
    assert pretty(GoldenRatio) == str(GoldenRatio) == "GoldenRatio"
    assert upretty(GoldenRatio) == "φ"


def test_Catalan():
    # 断言测试 pretty 函数对 Catalan 符号的输出
    assert pretty(Catalan) == upretty(Catalan) == "G"


def test_pretty_relational():
    # 断言测试 pretty 函数对关系运算符的输出
    expr = Eq(x, y)
    ascii_str = \
"""\
x = y\
"""
    ucode_str = \
"""\
x = y\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Lt(x, y)
    ascii_str = \
"""\
x < y\
"""
    ucode_str = \
"""\
x < y\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Gt(x, y)
    ascii_str = \
"""\
x > y\
"""
    ucode_str = \
"""\
x > y\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Le(x, y)
    ascii_str = \
"""\
x <= y\
"""
    ucode_str = \
"""\
x ≤ y\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Ge(x, y)
    ascii_str = \
"""\
x >= y\
"""
    ucode_str = \
"""\
x ≥ y\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Ne(x/(y + 1), y**2)
    ascii_str_1 = \
"""\
  x       2\n\
----- != y \n\
1 + y      \
"""
    ascii_str_2 = \
# 定义一个 ASCII 字符串表达式
ucode_str_1 = \
"""\
  x      2\n\
───── ≠ y \n\
1 + y     \
"""

# 定义另一个 ASCII 字符串表达式
ucode_str_2 = \
"""\
  x      2\n\
───── ≠ y \n\
y + 1     \
"""

# 断言表达式在美化后应等于上述两个 ASCII 字符串表达式之一
assert pretty(expr) in [ascii_str_1, ascii_str_2]
assert upretty(expr) in [ucode_str_1, ucode_str_2]


def test_Assignment():
    # 创建一个赋值表达式对象
    expr = Assignment(x, y)
    # ASCII 格式的字符串表示赋值表达式
    ascii_str = \
"""\
x := y\
"""
    # Unicode 格式的字符串表示赋值表达式
    ucode_str = \
"""\
x := y\
"""
    # 断言美化后的表达式应该等于定义的 ASCII 和 Unicode 字符串
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_AugmentedAssignment():
    # 创建一个加法增强赋值表达式对象
    expr = AddAugmentedAssignment(x, y)
    # ASCII 格式的字符串表示加法增强赋值表达式
    ascii_str = \
"""\
x += y\
"""
    # Unicode 格式的字符串表示加法增强赋值表达式
    ucode_str = \
"""\
x += y\
"""
    # 断言美化后的表达式应该等于定义的 ASCII 和 Unicode 字符串
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    # 创建一个减法增强赋值表达式对象
    expr = SubAugmentedAssignment(x, y)
    # ASCII 格式的字符串表示减法增强赋值表达式
    ascii_str = \
"""\
x -= y\
"""
    # Unicode 格式的字符串表示减法增强赋值表达式
    ucode_str = \
"""\
x -= y\
"""
    # 断言美化后的表达式应该等于定义的 ASCII 和 Unicode 字符串
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    # 创建一个乘法增强赋值表达式对象
    expr = MulAugmentedAssignment(x, y)
    # ASCII 格式的字符串表示乘法增强赋值表达式
    ascii_str = \
"""\
x *= y\
"""
    # Unicode 格式的字符串表示乘法增强赋值表达式
    ucode_str = \
"""\
x *= y\
"""
    # 断言美化后的表达式应该等于定义的 ASCII 和 Unicode 字符串
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    # 创建一个除法增强赋值表达式对象
    expr = DivAugmentedAssignment(x, y)
    # ASCII 格式的字符串表示除法增强赋值表达式
    ascii_str = \
"""\
x /= y\
"""
    # Unicode 格式的字符串表示除法增强赋值表达式
    ucode_str = \
"""\
x /= y\
"""
    # 断言美化后的表达式应该等于定义的 ASCII 和 Unicode 字符串
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    # 创建一个取模增强赋值表达式对象
    expr = ModAugmentedAssignment(x, y)
    # ASCII 格式的字符串表示取模增强赋值表达式
    ascii_str = \
"""\
x %= y\
"""
    # Unicode 格式的字符串表示取模增强赋值表达式
    ucode_str = \
"""\
x %= y\
"""
    # 断言美化后的表达式应该等于定义的 ASCII 和 Unicode 字符串
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_pretty_rational():
    # 创建一个有理数表达式对象
    expr = y*x**-2
    # ASCII 格式的字符串表示有理数表达式
    ascii_str = \
"""\
y \n\
--\n\
 2\n\
x \
"""
    # Unicode 格式的字符串表示有理数表达式
    ucode_str = \
"""\
y \n\
──\n\
 2\n\
x \
"""
    # 断言美化后的表达式应该等于定义的 ASCII 和 Unicode 字符串
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    # 创建一个有理数表达式对象
    expr = y**Rational(3, 2) * x**Rational(-5, 2)
    # ASCII 格式的字符串表示有理数表达式
    ascii_str = \
"""\
 3/2\n\
y   \n\
----\n\
 5/2\n\
x   \
"""
    # Unicode 格式的字符串表示有理数表达式
    ucode_str = \
"""\
 3/2\n\
y   \n\
────\n\
 5/2\n\
x   \
"""
    # 断言美化后的表达式应该等于定义的 ASCII 和 Unicode 字符串
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    # 创建一个三角函数表达式对象
    expr = sin(x)**3/tan(x)**2
    # ASCII 格式的字符串表示三角函数表达式
    ascii_str = \
"""\
   3   \n\
sin (x)\n\
-------\n\
   2   \n\
tan (x)\
"""
    # Unicode 格式的字符串表示三角函数表达式
    ucode_str = \
"""\
   3   \n\
sin (x)\n\
───────\n\
   2   \n\
tan (x)\
"""
    # 断言美化后的表达式应该等于定义的 ASCII 和 Unicode 字符串
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


@_both_exp_pow
def test_pretty_functions():
    """测试绝对值、共轭、指数、函数括号和阶乘。"""
    # 创建一个函数表达式对象
    expr = (2*x + exp(x))
    # ASCII 格式的字符串表示函数表达式，有两种可能性
    ascii_str_1 = \
"""\
       x\n\
2*x + e \
"""
    ascii_str_2 = \
"""\
 x      \n\
e  + 2*x\
"""
    # Unicode 格式的字符串表示函数表达式，有三种可能性
    ucode_str_1 = \
"""\
       x\n\
2⋅x + ℯ \
"""
    ucode_str_2 = \
"""\
 x     \n\
ℯ + 2⋅x\
"""
    ucode_str_3 = \
"""\
 x      \n\
ℯ  + 2⋅x\
"""
    # 断言美化后的表达式应该等于定义的 ASCII 和 Unicode 字符串中的一种
    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    assert upretty(expr) in [ucode_str_1, ucode_str_2, ucode_str_3]

    # 创建一个绝对值函数表达式对象
    expr = Abs(x)
    # ASCII 格式的字符串表示绝对值函数表达式
    ascii_str = \
"""\
|x|\
"""
    # Unicode 格式的字符串表示绝对值函数表达式
    ucode_str = \
"""\
│x│\
"""
    # 断言美化后的表达式应该等于定义的 ASCII 和 Unicode 字符串
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    # 创建一个绝对值函数表达式对象
    expr = Abs(x/(x**2 + 1))
    # ASCII 格式的字符串表示绝对值函数表达式
    ascii_str_1 = \
"""\
|  x   |\n\
|------|\n\
|     2|\n\
|1 + x |\
"""
    # 这行代码是一个多行字符串的第二行，继续之前的字符串
"""
生成一个 ASCII 艺术风格的数学表达式，使用反斜杠来转义特殊字符

ucode_str_1 = \
"""\
│  x   │\n\
│──────│\n\
│     2│\n\
│1 + x │\
"""
生成一个 Unicode 艺术风格的数学表达式，使用竖线和盒子来绘制

ucode_str_2 = \
"""\
│  x   │\n\
│──────│\n\
│ 2    │\n\
│x  + 1│\
"""
生成另一个 Unicode 艺术风格的数学表达式，使用竖线和盒子来绘制

assert pretty(expr) in [ascii_str_1, ascii_str_2]
assert upretty(expr) in [ucode_str_1, ucode_str_2]
验证函数 pretty 和 upretty 生成的数学表达式是否符合预期的 ASCII 和 Unicode 格式

expr = Abs(1 / (y - Abs(x)))
计算绝对值表达式，分子为 1，分母为 y 减去 x 的绝对值

ascii_str = \
"""\
    1    \n\
---------\n\
|y - |x||\
"""
生成一个 ASCII 艺术风格的绝对值表达式

ucode_str = \
"""\
    1    \n\
─────────\n\
│y - │x││\
"""
生成一个 Unicode 艺术风格的绝对值表达式

assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str
验证函数 pretty 和 upretty 生成的绝对值表达式是否符合预期的 ASCII 和 Unicode 格式

n = Symbol('n', integer=True)
定义一个整数符号 n

expr = factorial(n)
计算 n 的阶乘

ascii_str = \
"""\
n!\
"""
生成一个 ASCII 艺术风格的阶乘表达式

ucode_str = \
"""\
n!\
"""
生成一个 Unicode 艺术风格的阶乘表达式

assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str
验证函数 pretty 和 upretty 生成的阶乘表达式是否符合预期的 ASCII 和 Unicode 格式

expr = factorial(2*n)
计算 2*n 的阶乘

ascii_str = \
"""\
(2*n)!\
"""
生成一个 ASCII 艺术风格的阶乘表达式

ucode_str = \
"""\
(2⋅n)!\
"""
生成一个 Unicode 艺术风格的阶乘表达式

assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str
验证函数 pretty 和 upretty 生成的阶乘表达式是否符合预期的 ASCII 和 Unicode 格式

expr = factorial(factorial(factorial(n)))
计算 n 的阶乘的阶乘的阶乘

ascii_str = \
"""\
((n!)!)!\
"""
生成一个 ASCII 艺术风格的阶乘表达式

ucode_str = \
"""\
((n!)!)!\
"""
生成一个 Unicode 艺术风格的阶乘表达式

assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str
验证函数 pretty 和 upretty 生成的阶乘表达式是否符合预期的 ASCII 和 Unicode 格式

expr = factorial(n + 1)
计算 (n + 1) 的阶乘

ascii_str_1 = \
"""\
(1 + n)!\
"""
ascii_str_2 = \
"""\
(n + 1)!\
"""
ucode_str_1 = \
"""\
(1 + n)!\
"""
ucode_str_2 = \
"""\
(n + 1)!\
"""
验证函数 pretty 和 upretty 生成的阶乘表达式是否符合预期的 ASCII 和 Unicode 格式

expr = subfactorial(n)
计算 n 的子阶乘

ascii_str = \
"""\
!n\
"""
生成一个 ASCII 艺术风格的子阶乘表达式

ucode_str = \
"""\
!n\
"""
生成一个 Unicode 艺术风格的子阶乘表达式

assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str
验证函数 pretty 和 upretty 生成的子阶乘表达式是否符合预期的 ASCII 和 Unicode 格式

expr = subfactorial(2*n)
计算 2*n 的子阶乘

ascii_str = \
"""\
!(2*n)\
"""
生成一个 ASCII 艺术风格的子阶乘表达式

ucode_str = \
"""\
!(2⋅n)\
"""
生成一个 Unicode 艺术风格的子阶乘表达式

assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str
验证函数 pretty 和 upretty 生成的子阶乘表达式是否符合预期的 ASCII 和 Unicode 格式

n = Symbol('n', integer=True)
定义一个整数符号 n

expr = factorial2(n)
计算 n 的双阶乘

ascii_str = \
"""\
n!!\
"""
生成一个 ASCII 艺术风格的双阶乘表达式

ucode_str = \
"""\
n!!\
"""
生成一个 Unicode 艺术风格的双阶乘表达式

assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str
验证函数 pretty 和 upretty 生成的双阶乘表达式是否符合预期的 ASCII 和 Unicode 格式

expr = factorial2(2*n)
计算 2*n 的双阶乘

ascii_str = \
"""\
(2*n)!!\
"""
生成一个 ASCII 艺术风格的双阶乘表达式

ucode_str = \
"""\
(2⋅n)!!\
"""
生成一个 Unicode 艺术风格的双阶乘表达式

assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str
验证函数 pretty 和 upretty 生成的双阶乘表达式是否符合预期的 ASCII 和 Unicode 格式

expr = factorial2(factorial2(factorial2(n)))
计算 n 的双阶乘的双阶乘的双阶乘

ascii_str = \
"""\
((n!!)!!)!!\
"""
生成一个 ASCII 艺术风格的双阶乘表达式

ucode_str = \
"""\
((n!!)!!)!!\
"""
生成一个 Unicode 艺术风格的双阶乘表达式

assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str
验证函数 pretty 和 upretty 生成的双阶乘表达式是否符合预期的 ASCII 和 Unicode 格式

expr = factorial2(n + 1)
计算 (n + 1) 的双阶乘

ascii_str_1 = \
"""\
(1 + n)!!\
"""
ascii_str_2 = \
"""\
(n + 1)!!\
"""
ucode_str_1 = \
"""\
(1 + n)!!\
"""
ucode_str_2 = \
"""\
(n + 1)!!\
"""
验证函数 pretty 和 upretty 生成的双阶乘表达式是否符合预期的 ASCII 和 Unicode 格式

expr = 2*binomial(n, k)
计算二项式系数 C(n, k) 的两倍

ascii_str = \
"""\
  /n\\\n\
2*| |\n\
  \\k/\
"""
生成一个 ASCII 艺术风格的二项式系数表达式

ucode_str = \
"""\
  ⎛n⎞\n\
2⋅⎜ ⎟\n\
  ⎝k⎠\
"""
生成一个 Unicode 艺术风格的二项式系数表达式

assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str
验证函数 pretty 和 upretty 生成的二项式系数表达式是否符合预期的 ASCII 和 Unicode 格式

expr = 2*binomial(2*n, k)
计算二项式系数 C(2*n, k) 的两倍

ascii_str = \
"""\
  /2*n\\\n\
2*|   |\n\
  \\ k /\
"""
生成一个 ASCII 艺术风格的二项式系
    # 断言表达式经过 pretty 函数处理后应该等于 ascii_str
    assert pretty(expr) == ascii_str
    # 断言表达式经过 upretty 函数处理后应该等于 ucode_str
    assert upretty(expr) == ucode_str

    # 设置表达式为 2 乘以二项式函数 binomial(n**2, k)
    expr = 2*binomial(n**2, k)
    # 设置 ascii_str 变量为反斜杠结尾的多行字符串
    ascii_str = \
    """\
      / 2\\\n\
      |n |\n\
    2*|  |\n\
      \\k /\
    """
    # ASCII字符串表示的数学表达式，包含多行字符艺术字
    ucode_str = \
    """\
      ⎛ 2⎞\n\
      ⎜n ⎟\n\
    2⋅⎜  ⎟\n\
      ⎝k ⎠\
    """
    # Unicode字符串表示的数学表达式，使用数学符号代替ASCII字符艺术字

    # 断言测试pretty函数返回的ASCII表示是否等于ascii_str
    assert pretty(expr) == ascii_str
    # 断言测试upretty函数返回的Unicode表示是否等于ucode_str
    assert upretty(expr) == ucode_str

    # 计算Catalan数C(n)
    expr = catalan(n)
    # ASCII表示的Catalan数数学表达式
    ascii_str = \
    """\
    C \n\
     n\
    """
    # Unicode表示的Catalan数数学表达式
    ucode_str = \
    """\
    C \n\
     n\
    """
    # 断言测试pretty函数返回的ASCII表示是否等于ascii_str
    assert pretty(expr) == ascii_str
    # 断言测试upretty函数返回的Unicode表示是否等于ucode_str
    assert upretty(expr) == ucode_str

    # 计算Bell数B(n)
    expr = bell(n)
    # ASCII表示的Bell数数学表达式
    ascii_str = \
    """\
    B \n\
     n\
    """
    # Unicode表示的Bell数数学表达式
    ucode_str = \
    """\
    B \n\
     n\
    """
    # 断言测试pretty函数返回的ASCII表示是否等于ascii_str
    assert pretty(expr) == ascii_str
    # 断言测试upretty函数返回的Unicode表示是否等于ucode_str

    # 计算Bernoulli数B(n)
    expr = bernoulli(n)
    # ASCII表示的Bernoulli数数学表达式
    ascii_str = \
    """\
    B \n\
     n\
    """
    # Unicode表示的Bernoulli数数学表达式
    ucode_str = \
    """\
    B \n\
     n\
    """
    # 断言测试pretty函数返回的ASCII表示是否等于ascii_str
    assert pretty(expr) == ascii_str
    # 断言测试upretty函数返回的Unicode表示是否等于ucode_str

    # 计算Bernoulli多项式B_n(x)
    expr = bernoulli(n, x)
    # ASCII表示的Bernoulli多项式数学表达式
    ascii_str = \
    """\
    B (x)\n\
     n   \
    """
    # Unicode表示的Bernoulli多项式数学表达式
    ucode_str = \
    """\
    B (x)\n\
     n   \
    """
    # 断言测试pretty函数返回的ASCII表示是否等于ascii_str
    assert pretty(expr) == ascii_str
    # 断言测试upretty函数返回的Unicode表示是否等于ucode_str

    # 计算Fibonacci数F(n)
    expr = fibonacci(n)
    # ASCII表示的Fibonacci数数学表达式
    ascii_str = \
    """\
    F \n\
     n\
    """
    # Unicode表示的Fibonacci数数学表达式
    ucode_str = \
    """\
    F \n\
     n\
    """
    # 断言测试pretty函数返回的ASCII表示是否等于ascii_str
    assert pretty(expr) == ascii_str
    # 断言测试upretty函数返回的Unicode表示是否等于ucode_str

    # 计算Lucas数L(n)
    expr = lucas(n)
    # ASCII表示的Lucas数数学表达式
    ascii_str = \
    """\
    L \n\
     n\
    """
    # Unicode表示的Lucas数数学表达式
    ucode_str = \
    """\
    L \n\
     n\
    """
    # 断言测试pretty函数返回的ASCII表示是否等于ascii_str
    assert pretty(expr) == ascii_str
    # 断言测试upretty函数返回的Unicode表示是否等于ucode_str

    # 计算Tribonacci数T(n)
    expr = tribonacci(n)
    # ASCII表示的Tribonacci数数学表达式
    ascii_str = \
    """\
    T \n\
     n\
    """
    # Unicode表示的Tribonacci数数学表达式
    ucode_str = \
    """\
    T \n\
     n\
    """
    # 断言测试pretty函数返回的ASCII表示是否等于ascii_str
    assert pretty(expr) == ascii_str
    # 断言测试upretty函数返回的Unicode表示是否等于ucode_str

    # 计算Stieltjes常数γ(n)
    expr = stieltjes(n)
    # ASCII表示的Stieltjes常数数学表达式
    ascii_str = \
    """\
    stieltjes \n\
             n\
    """
    # Unicode表示的Stieltjes常数数学表达式
    ucode_str = \
    """\
    γ \n\
     n\
    """
    # 断言测试pretty函数返回的ASCII表示是否等于ascii_str
    assert pretty(expr) == ascii_str
    # 断言测试upretty函数返回的Unicode表示是否等于ucode_str

    # 计算Stieltjes常数γ(n, x)
    expr = stieltjes(n, x)
    # ASCII表示的Stieltjes常数数学表达式
    ascii_str = \
    """\
    stieltjes (x)\n\
             n   \
    """
    # Unicode表示的Stieltjes常数数学表达式
    ucode_str = \
    """\
    γ (x)\n\
     n   \
    """
    # 断言测试pretty函数返回的ASCII表示是否等于ascii_str
    assert pretty(expr) == ascii_str
    # 断言测试upretty函数返回的Unicode表示是否等于ucode_str

    # 计算Mathieu特殊函数C(x, y, z)
    expr = mathieuc(x, y, z)
    # ASCII表示的Mathieu特殊函数数学表达式
    ascii_str = 'C(x, y, z)'
    # Unicode表示的Mathieu特殊函数数学表达式
    ucode_str = 'C(x, y, z)'
    # 断言测试pretty函数返回的ASCII表示是否等于ascii_str
    assert pretty(expr) == ascii_str
    # 断言测试upretty函数返回的Unicode表示是否等于ucode_str

    # 计算Mathieu特殊函数S(x, y, z)
    expr = mathieus(x, y, z)
    # ASCII表示的Mathieu特殊函数数学表达式
    ascii_str = 'S(x, y, z)'
    # Unicode表示的Mathieu特殊函数数学表达式
    ucode_str = 'S(x, y, z)'
    # 断言测试pretty函数返回的ASCII表示是否等于ascii_str
    assert pretty(expr) == ascii_str
    # 断言测试upretty函数返回的Unicode表示是否等于ucode_str

    # 计算Mathieu特殊函数C'(x, y, z)
    expr = mathieucprime(x, y, z)
    # ASCII表示的Mathieu特殊函数导数数学表达式
    ascii_str = "C'(x, y, z)"
    # Unicode表示的Mathieu特殊函数导数数学表达式
    ucode_str = "C'(x, y, z)"
    # 断言测试pretty函数返回的ASCII表示是否等于ascii_str
    assert pretty(expr) == ascii_str
    # 断言测试upretty函数返回的Unicode表示是否等于ucode_str

    # 计算Mathieu特殊函数S'(x, y, z)
    expr = mathieusprime(x, y, z)
    # ASCII表示的Mathieu特殊函数导数数学表达式
    ascii_str = "S'(x, y, z)"
    # Unicode表示的Mathieu特殊函数导数数学表达式
    ucode_str = "S'(x, y, z)"
    # 断言测试pretty函数返回的ASCII表示是否等于ascii_str
    assert pretty(expr) == ascii_str
    # 断言测试upretty函数返回的Unicode表示是否等于ucode_str

    # 计算x的共轭复数
    expr = conjugate(x)
    # ASCII表示的共轭复数数学表达式
    ascii
    ascii_str_1 = \
"""\
 /  x     \\\n\
f|-----, y|\n\
 \\1 + y   /\
"""

这段代码定义了一个 ASCII 字符串 `ascii_str_1`，表示一个数学表达式的 ASCII 形式，具体表达式为 x / (1 + y)，格式化成带有斜线和分数线的形式。


    ascii_str_2 = \
"""\
 /  x     \\\n\
f|-----, y|\n\
 \\y + 1   /\
"""

这段代码定义了另一个 ASCII 字符串 `ascii_str_2`，表示同一个数学表达式 x / (y + 1)，也是带有斜线和分数线的形式。


    ucode_str_1 = \
"""\
 ⎛  x     ⎞\n\
f⎜─────, y⎟\n\
 ⎝1 + y   ⎠\
"""

这段代码定义了一个 Unicode 字符串 `ucode_str_1`，表示同样的数学表达式 x / (1 + y)，以 Unicode 格式输出，使用了括号和分数线。


    ucode_str_2 = \
"""\
 ⎛  x     ⎞\n\
f⎜─────, y⎟\n\
 ⎝y + 1   ⎠\
"""

这段代码定义了另一个 Unicode 字符串 `ucode_str_2`，表示数学表达式 x / (y + 1) 的 Unicode 格式，同样使用了括号和分数线。


    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    assert upretty(expr) in [ucode_str_1, ucode_str_2]

这段代码断言了函数 `pretty(expr)` 和 `upretty(expr)` 的输出应该分别在 `ascii_str_1` 和 `ascii_str_2` 组成的列表中，以及 `ucode_str_1` 和 `ucode_str_2` 组成的列表中。这用来验证程序输出的 ASCII 和 Unicode 格式的正确性。
    expr = ceiling(1 / (y - ceiling(x)))
    # 定义一个表达式，对 1 / (y - ceiling(x)) 向上取整
    ascii_str = \
"""\
       /      1       \\\n\
ceiling|--------------|\n\
       \\y - ceiling(x)/\
"""
    # 对应的 ASCII 字符串表示，展示了向上取整函数的表达式
    ucode_str = \
"""\
⎡   1   ⎤\n\
⎢───────⎥\n\
⎢y - ⌈x⌉⎥\
"""
    # 对应的 Unicode 字符串表示，展示了向上取整函数的表达式
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = euler(n)
    # 计算欧拉数 e^n 的表达式
    ascii_str = \
"""\
E \n\
 n\
"""
    ucode_str = \
"""\
E \n\
 n\
"""
    # 对应的 ASCII 和 Unicode 字符串表示
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = euler(1/(1 + 1/(1 + 1/n)))
    # 计算欧拉数 e^(1/(1 + 1/(1 + 1/n))) 的表达式
    ascii_str = \
"""\
E         \n\
     1    \n\
 ---------\n\
       1  \n\
 1 + -----\n\
         1\n\
     1 + -\n\
         n\
"""
    ucode_str = \
"""\
E         \n\
     1    \n\
 ─────────\n\
       1  \n\
 1 + ─────\n\
         1\n\
     1 + ─\n\
         n\
"""
    # 对应的 ASCII 和 Unicode 字符串表示
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = euler(n, x)
    # 计算欧拉数 E_n(x) 的表达式
    ascii_str = \
"""\
E (x)\n\
 n   \
"""
    ucode_str = \
"""\
E (x)\n\
 n   \
"""
    # 对应的 ASCII 和 Unicode 字符串表示
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = euler(n, x/2)
    # 计算欧拉数 E_n(x/2) 的表达式
    ascii_str = \
"""\
  /x\\\n\
E |-|\n\
 n\\2/\
"""
    ucode_str = \
"""\
  ⎛x⎞\n\
E ⎜─⎟\n\
 n⎝2⎠\
"""
    # 对应的 ASCII 和 Unicode 字符串表示
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    /      2        1000___    \n\
   /      x  + 1      \/ x  + 1\n\
    # 定义一个多行的 ASCII 艺术风格字符串，用于测试和比较 pretty 函数的输出
    ascii_str = \
"""\
4 /   2 + ------  + -----------\n\
\\/        x + 2        ________\n\
                      /  2     \n\
                    \\/  x  + 3 \
"""
    # 定义一个多行的 Unicode 艺术风格字符串，用于测试和比较 upretty 函数的输出
    ucode_str = \
"""\
     ____________              \n\
    ╱      2        1000___    \n\
   ╱      x  + 1      ╲╱ x  + 1\n\
4 ╱   2 + ──────  + ───────────\n\
╲╱        x + 2        ________\n\
                      ╱  2     \n\
                    ╲╱  x  + 3 \
"""

    # 使用 assert 断言来验证 pretty 函数输出是否符合预期的 ASCII 艺术风格字符串
    assert pretty(expr) == ascii_str
    # 使用 assert 断言来验证 upretty 函数输出是否符合预期的 Unicode 艺术风格字符串
    assert upretty(expr) == ucode_str
"""\
     2\n\
x ↦ x \
"""
# 定义表达式 expr，表示 Lambda 函数 x ↦ x**2 的平方
expr = Lambda(x, x**2)**2
# ASCII 字符串表示的 Lambda 函数
ascii_str = \
"""\
         2
/      2\\ \n\
\\x -> x / \
"""
# Unicode 字符串表示的 Lambda 函数
ucode_str = \
"""\
        2
⎛     2⎞ \n\
⎝x ↦ x ⎠ \
"""
# 断言表达式的 ASCII 表示和给定的 ascii_str 相等
assert pretty(expr) == ascii_str
# 断言表达式的 Unicode 表示和给定的 ucode_str 相等
assert upretty(expr) == ucode_str

# 定义表达式 expr，表示 Lambda 函数 (x, y) ↦ x
expr = Lambda((x, y), x)
# ASCII 字符串表示的 Lambda 函数
ascii_str = "(x, y) -> x"
# Unicode 字符串表示的 Lambda 函数
ucode_str = "(x, y) ↦ x"
# 断言表达式的 ASCII 表示和给定的 ascii_str 相等
assert pretty(expr) == ascii_str
# 断言表达式的 Unicode 表示和给定的 ucode_str 相等
assert upretty(expr) == ucode_str

# 定义表达式 expr，表示 Lambda 函数 (x, y) ↦ x**2
expr = Lambda((x, y), x**2)
# ASCII 字符串表示的 Lambda 函数
ascii_str = \
"""\
           2\n\
(x, y) -> x \
"""
# Unicode 字符串表示的 Lambda 函数
ucode_str = \
"""\
          2\n\
(x, y) ↦ x \
"""
# 断言表达式的 ASCII 表示和给定的 ascii_str 相等
assert pretty(expr) == ascii_str
# 断言表达式的 Unicode 表示和给定的 ucode_str 相等
assert upretty(expr) == ucode_str

# 定义表达式 expr，表示 Lambda 函数 ((x, y),) ↦ x**2
expr = Lambda(((x, y),), x**2)
# ASCII 字符串表示的 Lambda 函数
ascii_str = \
"""\
              2\n\
((x, y),) -> x \
"""
# Unicode 字符串表示的 Lambda 函数
ucode_str = \
"""\
             2\n\
((x, y),) ↦ x \
"""
# 断言表达式的 ASCII 表示和给定的 ascii_str 相等
assert pretty(expr) == ascii_str
# 断言表达式的 Unicode 表示和给定的 ucode_str 相等
assert upretty(expr) == ucode_str
# 定义一个测试函数，用于验证并比较多个传递函数的输出结果是否符合预期
def test_pretty_Parallel():
    # 定义传递函数 tf1，使用变量 x 和 y
    tf1 = TransferFunction(x + y, x - 2*y, y)
    # 定义传递函数 tf2，使用变量 x 和 y
    tf2 = TransferFunction(x - y, x + y, y)
    # 定义传递函数 tf3，使用变量 x 和 y
    tf3 = TransferFunction(x**2 + y, y - x, y)
    # 定义传递函数 tf4，使用变量 x 和 y
    tf4 = TransferFunction(y**2 - x, x**3 + x, y)

    # 定义传递函数矩阵 tfm1，包含多个传递函数
    tfm1 = TransferFunctionMatrix([[tf1, tf2], [tf3, -tf4], [-tf2, -tf1]])
    # 定义传递函数矩阵 tfm2，包含多个传递函数
    tfm2 = TransferFunctionMatrix([[-tf2, -tf1], [tf4, -tf3], [tf1, tf2]])
    # 定义传递函数矩阵 tfm3，包含多个传递函数
    tfm3 = TransferFunctionMatrix([[-tf1, tf2], [-tf3, tf4], [tf2, tf1]])
    # 定义传递函数矩阵 tfm4，包含多个传递函数
    tfm4 = TransferFunctionMatrix([[-tf1, -tf2], [-tf3, -tf4]])

    # 预期结果字符串 expected1，包含传递函数运算结果的可读格式
    expected1 = \
"""\
 x + y    x - y\n\
─────── + ─────\n\
x - 2⋅y   x + y\
"""
    # 预期结果字符串 expected2，包含传递函数运算结果的可读格式
    expected2 = \
"""\
-x + y   -x - y \n\
────── + ───────
x + y    x - 2⋅y\
"""
    # 预期结果字符串 expected3，包含传递函数运算结果的可读格式
    expected3 = \
"""\
 2                                  \n\
x  + y    x + y    ⎛-x - y ⎞ ⎛x - y⎞
────── + ─────── + ⎜───────⎟⋅⎜─────⎟
-x + y   x - 2⋅y   ⎝x - 2⋅y⎠ ⎝x + y⎠\
"""

    # 预期结果字符串 expected4，包含传递函数运算结果的可读格式
    expected4 = \
"""\
                            ⎛ 2    ⎞\n\
⎛ x + y ⎞ ⎛x - y⎞   ⎛x - y⎞ ⎜x  + y⎟\n\
⎜───────⎟⋅⎜─────⎟ + ⎜─────⎟⋅⎜──────⎟\n\
⎝x - 2⋅y⎠ ⎝x + y⎠   ⎝x + y⎠ ⎝-x + y⎠\
"""
    # 预期结果字符串 expected5，包含传递函数运算结果的可读格式
    expected5 = \
"""\
⎡ x + y   -x + y ⎤    ⎡ x - y    x + y ⎤    ⎡ x + y    x - y ⎤ \n\
⎢───────  ────── ⎥    ⎢ ─────   ───────⎥    ⎢───────   ───── ⎥ \n\
⎢x - 2⋅y  x + y  ⎥    ⎢ x + y   x - 2⋅y⎥    ⎢x - 2⋅y   x + y ⎥ \n\
⎢                ⎥    ⎢                ⎥    ⎢                ⎥ \n\
⎢ 2            2 ⎥    ⎢     2    2     ⎥    ⎢ 2            2 ⎥ \n\
⎢x  + y   x - y  ⎥    ⎢x - y    x  + y ⎥    ⎢x  + y   x - y  ⎥ \n\
⎢──────   ────── ⎥  + ⎢──────   ────── ⎥  + ⎢──────   ────── ⎥ \n\
⎢-x + y    3     ⎥    ⎢ 3       -x + y ⎥    ⎢-x + y    3     ⎥ \n\
⎢         x  + x ⎥    ⎢x  + x          ⎥    ⎢         x  + x ⎥ \n\
⎢                ⎥    ⎢                ⎥    ⎢                ⎥ \n\
expected6 = \
"""\
⎡ x - y    x + y ⎤                        ⎡-x + y   -x - y  ⎤ \n\
⎢ ─────   ───────⎥                        ⎢──────   ─────── ⎥ \n\
⎢ x + y   x - 2⋅y⎥  ⎡-x - y    -x + y⎤    ⎢x + y    x - 2⋅y ⎥ \n\
⎢                ⎥  ⎢───────   ──────⎥    ⎢                 ⎥ \n\
⎢     2    2     ⎥  ⎢x - 2⋅y   x + y ⎥    ⎢      2     2    ⎥ \n\
⎢x - y    x  + y ⎥  ⎢                ⎥    ⎢-x + y   - x  - y⎥ \n\
⎢──────   ────── ⎥ ⋅⎢   2           2⎥  + ⎢───────  ────────⎥ \n\
⎢ 3       -x + y ⎥  ⎢- x  - y  x - y ⎥    ⎢ 3        -x + y ⎥ \n\
⎢x  + x          ⎥  ⎢────────  ──────⎥    ⎢x  + x           ⎥ \n\
⎢                ⎥  ⎢ -x + y    3    ⎥    ⎢                 ⎥ \n\
⎢-x - y   -x + y ⎥  ⎣          x  + x⎦τ   ⎢ x + y    x - y  ⎥ \n\
⎢───────  ────── ⎥                        ⎢───────   ─────  ⎥ \n\
⎣x - 2⋅y  x + y  ⎦τ                       ⎣x - 2⋅y   x + y  ⎦τ\
"""
# 生成预期输出的Unicode字符串，包含数学表达式和排版

assert upretty(Parallel(tf1, tf2)) == expected1
# 断言：调用upretty函数对tf1和tf2并联后的输出，应该等于expected1的Unicode字符串

assert upretty(Parallel(-tf2, -tf1)) == expected2
# 断言：调用upretty函数对-tf2和-tf1并联后的输出，应该等于expected2的Unicode字符串

assert upretty(Parallel(tf3, tf1, Series(-tf1, tf2))) == expected3
# 断言：调用upretty函数对tf3、tf1和-tf1并tf2串联后并联的输出，应该等于expected3的Unicode字符串

assert upretty(Parallel(Series(tf1, tf2), Series(tf2, tf3))) == expected4
# 断言：调用upretty函数对tf1和tf2串联，以及tf2和tf3串联后并联的输出，应该等于expected4的Unicode字符串

assert upretty(MIMOParallel(-tfm3, -tfm2, tfm1)) == expected5
# 断言：调用upretty函数对-tfm3、-tfm2和tfm1进行MIMO并联后的输出，应该等于expected5的Unicode字符串

assert upretty(MIMOParallel(MIMOSeries(tfm4, -tfm2), tfm2)) == expected6
# 断言：调用upretty函数对MIMOSeries(tfm4, -tfm2)和tfm2进行MIMO并联后的输出，应该等于expected6的Unicode字符串
"""\
      ⎛ x + y ⎞ ⎛x - y⎞      \n\
      ⎜───────⎟⋅⎜─────⎟      \n\
      ⎝x - 2⋅y⎠ ⎝x + y⎠      \n\
─────────────────────────────\n\
1   ⎛ x + y ⎞ ⎛x - y⎞ ⎛1 - x⎞\n\
─ + ⎜───────⎟⋅⎜─────⎟⋅⎜─────⎟\n\
1   ⎝x - 2⋅y⎠ ⎝x + y⎠ ⎝x - y⎠\
"""
expected6 = \
"""\
           ⎛ 2          ⎞                   \n\
           ⎜y  - 2⋅y + 1⎟ ⎛1 - x⎞           \n\
           ⎜────────────⎟⋅⎜─────⎟           \n\
           ⎝   y + 5    ⎠ ⎝x - y⎠           \n\
────────────────────────────────────────────\n\
    ⎛ 2          ⎞                          \n\
1   ⎜y  - 2⋅y + 1⎟ ⎛1 - x⎞ ⎛x - y⎞ ⎛ x + y ⎞\n\
─ + ⎜────────────⎟⋅⎜─────⎟⋅⎜─────⎟⋅⎜───────⎟\n\
1   ⎝   y + 5    ⎠ ⎝x - y⎠ ⎝x + y⎠ ⎝x - 2⋅y⎠\
"""
expected7 = \
"""\
    ⎛       3⎞    \n\
    ⎜x - 2⋅y ⎟    \n\
    ⎜────────⎟    \n\
    ⎝ x + y  ⎠    \n\
──────────────────\n\
    ⎛       3⎞    \n\
1   ⎜x - 2⋅y ⎟ ⎛2⎞\n\
─ + ⎜────────⎟⋅⎜─⎟\n\
1   ⎝ x + y  ⎠ ⎝2⎠\
"""
expected8 = \
"""\
  ⎛1 - x⎞  \n\
  ⎜─────⎟  \n\
  ⎝x - y⎠  \n\
───────────\n\
1   ⎛1 - x⎞\n\
─ + ⎜─────⎟\n\
1   ⎝x - y⎠\
"""
expected9 = \
"""\
      ⎛ x + y ⎞ ⎛x - y⎞      \n\
      ⎜───────⎟⋅⎜─────⎟      \n\
      ⎝x - 2⋅y⎠ ⎝x + y⎠      \n\
─────────────────────────────\n\
1   ⎛ x + y ⎞ ⎛x - y⎞ ⎛1 - x⎞\n\
─ - ⎜───────⎟⋅⎜─────⎟⋅⎜─────⎟\n\
1   ⎝x - 2⋅y⎠ ⎝x + y⎠ ⎝x - y⎠\
"""
expected10 = \
"""\
  ⎛1 - x⎞  \n\
  ⎜─────⎟  \n\
  ⎝x - y⎠  \n\
───────────\n\
1   ⎛1 - x⎞\n\
─ - ⎜─────⎟\n\
1   ⎝x - y⎠\
"""

assert upretty(Feedback(tf, tf1)) == expected1
assert upretty(Feedback(tf, tf2*tf1*tf3)) == expected2
assert upretty(Feedback(tf1, tf2*tf3*tf5)) == expected3
assert upretty(Feedback(tf1*tf2, tf)) == expected4
assert upretty(Feedback(tf1*tf2, tf5)) == expected5
assert upretty(Feedback(tf3*tf5, tf2*tf1)) == expected6
assert upretty(Feedback(tf4, tf6)) == expected7
assert upretty(Feedback(tf5, tf)) == expected8

assert upretty(Feedback(tf1*tf2, tf5, 1)) == expected9
assert upretty(Feedback(tf5, tf, 1)) == expected10
# 定义测试函数，用于测试函数 upretty() 对 TransferFunctionMatrix 的输出结果是否正确
def test_pretty_TransferFunctionMatrix():
    # 创建多个 TransferFunction 实例
    tf1 = TransferFunction(x + y, x - 2*y, y)
    tf2 = TransferFunction(x - y, x + y, y)
    tf3 = TransferFunction(y**2 - 2*y + 1, y + 5, y)
    tf4 = TransferFunction(y, x**2 + x + 1, y)
    tf5 = TransferFunction(1 - x, x - y, y)
    tf6 = TransferFunction(2, 2, y)

    # 预期输出结果的字符串表示，使用三重引号包围
    expected1 = \
"""\
⎡ x + y ⎤ \n\
⎢───────⎥ \n\
⎢x - 2⋅y⎥ \n\
⎢       ⎥ \n\
⎢ x - y ⎥ \n\
⎢ ───── ⎥ \n\
⎣ x + y ⎦τ\
"""
    expected2 = \
"""\
⎡    x + y     ⎤ \n\
⎢   ───────    ⎥ \n\
⎢   x - 2⋅y    ⎥ \n\
⎢              ⎥ \n\
⎢    x - y     ⎥ \n\
⎢    ─────     ⎥ \n\
⎢    x + y     ⎥ \n\
⎢              ⎥ \n\
⎢   2          ⎥ \n\
⎢- y  + 2⋅y - 1⎥ \n\
⎢──────────────⎥ \n\
⎣    y + 5     ⎦τ\
"""
    expected3 = \
"""\
⎡   x + y        x - y   ⎤ \n\
⎢  ───────       ─────   ⎥ \n\
⎢  x - 2⋅y       x + y   ⎥ \n\
⎢                        ⎥ \n\
⎢ 2                      ⎥ \n\
⎢y  - 2⋅y + 1      y     ⎥ \n\
⎢────────────  ──────────⎥ \n\
⎢   y + 5       2        ⎥ \n\
⎢              x  + x + 1⎥ \n\
⎢                        ⎥ \n\
⎢   1 - x          2     ⎥ \n\
⎢   ─────          ─     ⎥ \n\
⎣   x - y          2     ⎦τ\
"""
    expected4 = \
"""\
⎡    x - y        x + y       y     ⎤ \n\
⎢    ─────       ───────  ──────────⎥ \n\
⎢    x + y       x - 2⋅y   2        ⎥ \n\
⎢                         x  + x + 1⎥ \n\
⎢                                   ⎥ \n\
⎢   2                               ⎥ \n\
⎢- y  + 2⋅y - 1   x - 1      -2     ⎥ \n\
⎢──────────────   ─────      ───    ⎥ \n\
⎣    y + 5        x - y       2     ⎦τ\
"""
    expected5 = \
"""\
⎡ x + y  x - y   x + y       y     ⎤ \n\
⎢───────⋅─────  ───────  ──────────⎥ \n\
⎢x - 2⋅y x + y  x - 2⋅y   2        ⎥ \n\
⎢                        x  + x + 1⎥ \n\
⎢                                  ⎥ \n\
⎢  1 - x   2     x + y      -2     ⎥ \n\
⎢  ───── + ─    ───────     ───    ⎥ \n\
⎣  x - y   2    x - 2⋅y      2     ⎦τ\
"""

    # 使用 assert 语句来验证 upretty 函数对 TransferFunctionMatrix 输出结果的正确性，并添加注释说明
    assert upretty(TransferFunctionMatrix([[tf1], [tf2]])) == expected1
    assert upretty(TransferFunctionMatrix([[tf1], [tf2], [-tf3]])) == expected2
    # 使用 assert 断言检查 TransferFunctionMatrix 对象转换为漂亮的字符串形式是否与预期的字符串相等
    assert upretty(TransferFunctionMatrix([[tf1, tf2], [tf3, tf4], [tf5, tf6]])) == expected3
    # 使用 assert 断言检查 TransferFunctionMatrix 对象转换为漂亮的字符串形式是否与预期的字符串相等
    assert upretty(TransferFunctionMatrix([[tf2, tf1, tf4], [-tf3, -tf5, -tf6]])) == expected4
    # 使用 assert 断言检查 TransferFunctionMatrix 对象转换为漂亮的字符串形式是否与预期的字符串相等
    assert upretty(TransferFunctionMatrix([[Series(tf2, tf1), tf1, tf4], [Parallel(tf6, tf5), tf1, -tf6]])) == \
        expected5
def test_pretty_StateSpace():
    # 创建 StateSpace 对象 ss1，使用给定的矩阵 [a], [b], [c], [d]
    ss1 = StateSpace(Matrix([a]), Matrix([b]), Matrix([c]), Matrix([d]))
    
    # 定义矩阵 A, B, C, D
    A = Matrix([[0, 1], [1, 0]])
    B = Matrix([1, 0])
    C = Matrix([[0, 1]])
    D = Matrix([0])
    
    # 创建 StateSpace 对象 ss2，使用定义的矩阵 A, B, C, D
    ss2 = StateSpace(A, B, C, D)
    
    # 创建 StateSpace 对象 ss3，使用指定的四个矩阵
    ss3 = StateSpace(Matrix([[-1.5, -2], [1, 0]]),
                    Matrix([[0.5, 0], [0, 1]]),
                    Matrix([[0, 1], [0, 2]]),
                    Matrix([[2, 2], [1, 1]]))

    # 定义期望的 Unicode 字符串表示 ss1 的结果
    expected1 = \
"""\
⎡[a]  [b]⎤\n\
⎢        ⎥\n\
⎣[c]  [d]⎦\
"""

    # 定义期望的 Unicode 字符串表示 ss2 的结果
    expected2 = \
"""\
⎡⎡0  1⎤  ⎡1⎤⎤\n\
⎢⎢    ⎥  ⎢ ⎥⎥\n\
⎢⎣1  0⎦  ⎣0⎦⎥\n\
⎢           ⎥\n\
⎣[0  1]  [0]⎦\
"""

    # 定义期望的 Unicode 字符串表示 ss3 的结果
    expected3 = \
"""\
⎡⎡-1.5  -2⎤  ⎡0.5  0⎤⎤\n\
⎢⎢        ⎥  ⎢      ⎥⎥\n\
⎢⎣ 1    0 ⎦  ⎣ 0   1⎦⎥\n\
⎢                    ⎥\n\
⎢  ⎡0  1⎤     ⎡2  2⎤ ⎥\n\
⎢  ⎢    ⎥     ⎢    ⎥ ⎥\n\
⎣  ⎣0  2⎦     ⎣1  1⎦ ⎦\
"""

    # 断言 ss1 的 Unicode 字符串表示与期望值相等
    assert upretty(ss1) == expected1
    
    # 断言 ss2 的 Unicode 字符串表示与期望值相等
    assert upretty(ss2) == expected2
    
    # 断言 ss3 的 Unicode 字符串表示与期望值相等
    assert upretty(ss3) == expected3


def test_pretty_order():
    # 创建 O(1) 的表达式 expr
    expr = O(1)
    
    # 定义 ASCII 表示的期望字符串
    ascii_str = \
"""\
O(1)\
"""
    
    # 定义 Unicode 表示的期望字符串
    ucode_str = \
"""\
O(1)\
"""
    
    # 断言 expr 的 ASCII 表示与期望值相等
    assert pretty(expr) == ascii_str
    
    # 断言 expr 的 Unicode 表示与期望值相等
    assert upretty(expr) == ucode_str

    # 创建 O(1/x) 的表达式 expr
    expr = O(1/x)
    
    # 定义 ASCII 表示的期望字符串
    ascii_str = \
"""\
 /1\\\n\
O|-|\n\
 \\x/\
"""
    
    # 定义 Unicode 表示的期望字符串
    ucode_str = \
"""\
 ⎛1⎞\n\
O⎜─⎟\n\
 ⎝x⎠\
"""
    
    # 断言 expr 的 ASCII 表示与期望值相等
    assert pretty(expr) == ascii_str
    
    # 断言 expr 的 Unicode 表示与期望值相等
    assert upretty(expr) == ucode_str

    # 创建 O(x**2 + y**2) 的表达式 expr
    expr = O(x**2 + y**2)
    
    # 定义 ASCII 表示的期望字符串
    ascii_str = \
"""\
 / 2    2                  \\\n\
O\\x  + y ; (x, y) -> (0, 0)/\
"""
    
    # 定义 Unicode 表示的期望字符串
    ucode_str = \
"""\
 ⎛ 2    2                 ⎞\n\
O⎝x  + y ; (x, y) → (0, 0)⎠\
"""
    
    # 断言 expr 的 ASCII 表示与期望值相等
    assert pretty(expr) == ascii_str
    
    # 断言 expr 的 Unicode 表示与期望值相等
    assert upretty(expr) == ucode_str

    # 创建带有限制的 O(1, (x, oo)) 的表达式 expr
    expr = O(1, (x, oo))
    
    # 定义 ASCII 表示的期望字符串
    ascii_str = \
"""\
O(1; x -> oo)\
"""
    
    # 定义 Unicode 表示的期望字符串
    ucode_str = \
"""\
O(1; x → ∞)\
"""
    
    # 断言 expr 的 ASCII 表示与期望值相等
    assert pretty(expr) == ascii_str
    
    # 断言 expr 的 Unicode 表示与期望值相等
    assert upretty(expr) == ucode_str

    # 创建带有限制的 O(1/x, (x, oo)) 的表达式 expr
    expr = O(1/x, (x, oo))
    
    # 定义 ASCII 表示的期望字符串
    ascii_str = \
"""\
 /1         \\\n\
O|-; x -> oo|\n\
 \\x         /\
"""
    
    # 定义 Unicode 表示的期望字符串
    ucode_str = \
"""\
 ⎛1       ⎞\n\
O⎜─; x → ∞⎟\n\
 ⎝x       ⎠\
"""
    
    # 断言 expr 的 ASCII 表示与期望值相等
    assert pretty(expr) == ascii_str
    
    # 断言 expr 的 Unicode 表示与期望值相等
    assert upretty(expr) == ucode_str

    # 创建带有多个限制的 O(x**2 + y**2, (x, oo), (y, oo)) 的表达式 expr
    expr = O(x**2 + y**2, (x, oo), (y, oo))
    
    # 定义 ASCII 表示的期望字符串
    ascii_str = \
"""\
 / 2    2                    \\\n\
O\\x  + y ; (x, y) -> (oo, oo)/\
"""
    
    # 定义 Unicode 表示的期望字符串
    ucode_str = \
"""\
 ⎛ 2    2                 ⎞\n\
O⎝x  + y ; (x, y) → (∞, ∞)⎠\
"""
    
    # 断言 expr 的 ASCII 表示与期望值相等
    assert pretty(expr) == ascii_str
    
    # 断言 expr 的 Unicode 表示与期望值相等
    assert upretty(expr) == ucode_str


def test_pretty_derivatives():
    # 创建对数函数的导数表达式，不进行求值
    expr = Derivative(log(x), x, evaluate=False)
    
    # 定义 ASCII 表示的期望字符串
    ascii_str = \
"""\
d         \n\
--(log(x))\n\
dx        \
"""
    
    # 定义 Unicode 表示的期望字符串
    ucode_str = \
"""\
d         \n\
──(log(x))\n\
dx        \
"""
    
    # 断言 expr 的 ASCII 表示与期望值相等
    assert pretty(expr) == ascii_str
    
    # 断言 expr 的 Unicode 表
    # 确保 pretty 函数处理后的结果在指定的 ASCII 字符串列表中
    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    # 确保 upretty 函数处理后的结果在指定的 Unicode 字符串列表中
    assert upretty(expr) in [ucode_str_1, ucode_str_2]

    # 定义一个表达式，计算其关于变量 x 的偏导数
    expr = Derivative(log(x + y) + x, x)
    # 将 ASCII 字符串的第一个片段赋给 ascii_str_1，使用反斜杠表示延续
    ascii_str_1 = \
    # Define an expression representing a mixed partial derivative with ASCII representation
    expr = Derivative(2*x*y, x, x, y)
    # Define the corresponding ASCII string representation for the expression
    ascii_str = \
"""\
   3         \n\
  d          \n\
------(2*x*y)\n\
     2       \n\
dy dx        \
"""
    # Define the corresponding Unicode (pretty) string representation for the expression
    ucode_str = \
"""\
   3         \n\
  ∂          \n\
─────(2⋅x⋅y)\n\
     2       \n\
∂y ∂x        \
"""
    # Perform assertion to validate ASCII representation of the expression
    assert pretty(expr) == ascii_str
    # Perform assertion to validate Unicode (pretty) representation of the expression
    assert upretty(expr) == ucode_str



    # Define an expression representing a partial derivative with respect to x repeated 17 times
    expr = Derivative(2*x*y, x, 17)
    # Define the corresponding ASCII string representation for the expression
    ascii_str = \
"""\
 17        \n\
d          \n\
----(2*x*y)\n\
  17       \n\
dx         \
"""
    # Define the corresponding Unicode (pretty) string representation for the expression
    ucode_str = \
"""\
 17        \n\
∂          \n\
────(2⋅x⋅y)\n\
  17       \n\
∂x         \
"""
    # Perform assertion to validate ASCII representation of the expression
    assert pretty(expr) == ascii_str
    # Perform assertion to validate Unicode (pretty) representation of the expression
    assert upretty(expr) == ucode_str



    # Define an expression representing a derivative with respect to x and y, added with x^2
    expr = Derivative(2*x*y, y, x) + x**2
    # Define one of the ASCII string representations for the expression
    ascii_str_1 = \
"""\
   2             \n\
  d             2\n\
-----(2*x*y) + x \n\
dx dy            \
"""
    # Define another ASCII string representation for the expression
    ascii_str_2 = \
"""\
        2        \n\
 2     d         \n\
x  + -----(2*x*y)\n\
     dx dy       \
"""
    # Define a third ASCII string representation for the expression
    ascii_str_3 = \
"""\
       2         \n\
 2    d          \n\
x  + -----(2*x*y)\n\
     dx dy       \
"""
    # Define the corresponding Unicode (pretty) string representations for the expression
    ucode_str_1 = \
"""\
   2             \n\
  ∂             2\n\
─────(2⋅x⋅y) + x \n\
∂x ∂y            \
"""
    ucode_str_2 = \
"""\
        2        \n\
 2     ∂         \n\
x  + ─────(2⋅x⋅y)\n\
     ∂x ∂y       \
"""
    ucode_str_3 = \
"""\
       2         \n\
 2    ∂          \n\
x  + ─────(2⋅x⋅y)\n\
     ∂x ∂y       \
"""
    # Perform assertions to validate ASCII representation of the expression against multiple strings
    assert pretty(expr) in [ascii_str_1, ascii_str_2, ascii_str_3]
    # Perform assertions to validate Unicode (pretty) representation of the expression against multiple strings
    assert upretty(expr) in [ucode_str_1, ucode_str_2, ucode_str_3]


These code blocks comprehensively annotate each segment of the provided mathematical expressions and their representations in ASCII and Unicode formats, ensuring clarity on their respective functionalities and assertions.
# 断言：验证 pretty 函数对表达式的 ASCII 表示是否与 ascii_str 相等
assert pretty(expr) == ascii_str
# 断言：验证 upretty 函数对表达式的 Unicode 表示是否与 ucode_str 相等
assert upretty(expr) == ucode_str

# 定义希腊字母符号
alpha = Symbol('alpha')
beta = Function('beta')
# 计算 beta(alpha) 对 alpha 的导数
expr = beta(alpha).diff(alpha)

# ASCII 表示
ascii_str = \
"""\
  d                \n\
------(beta(alpha))\n\
dalpha             \
"""
# Unicode 表示
ucode_str = \
"""\
d       \n\
──(β(α))\n\
dα      \
"""
# 断言：验证 pretty 函数对表达式的 ASCII 表示是否与 ascii_str 相等
assert pretty(expr) == ascii_str
# 断言：验证 upretty 函数对表达式的 Unicode 表示是否与 ucode_str 相等
assert upretty(expr) == ucode_str

# 构造 f(x) 对 x 的 n 阶导数表达式
expr = Derivative(f(x), (x, n))

# ASCII 表示
ascii_str = \
"""\
 n       \n\
d        \n\
---(f(x))\n\
  n      \n\
dx       \
"""
# Unicode 表示
ucode_str = \
"""\
 n       \n\
d        \n\
───(f(x))\n\
  n      \n\
dx       \
"""
# 断言：验证 pretty 函数对表达式的 ASCII 表示是否与 ascii_str 相等
assert pretty(expr) == ascii_str
# 断言：验证 upretty 函数对表达式的 Unicode 表示是否与 ucode_str 相等
assert upretty(expr) == ucode_str
"""\
  /  /           \n\
 |  |            \n\
 |  |  2  2      \n\
 |  | x *y  dx dy\n\
 |  |            \n\
/  /             \
"""
# 定义ASCII艺术字符串，表示一个数学表达式的轮廓

ucode_str = \
"""\
⌠ ⌠            \n\
⎮ ⎮  2  2      \n\
⎮ ⎮ x ⋅y  dx dy\n\
⌡ ⌡            \
"""
# 定义Unicode艺术字符串，表示与ASCII艺术字符串等效的数学表达式的轮廓

assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str

expr = Integral(sin(th)/cos(ph), (th, 0, pi), (ph, 0, 2*pi))
ascii_str = \
"""\
 2*pi pi                           \n\
   /   /                           \n\
  |   |                            \n\
  |   |  sin(theta)                \n\
  |   |  ---------- d(theta) d(phi)\n\
  |   |   cos(phi)                 \n\
  |   |                            \n\
 /   /                             \n\
0    0                             \
"""
# 定义ASCII字符串，表示一个积分表达式的数学形式

ucode_str = \
"""\
2⋅π π             \n\
 ⌠  ⌠             \n\
 ⎮  ⎮ sin(θ)      \n\
 ⎮  ⎮ ────── dθ dφ\n\
 ⎮  ⎮ cos(φ)      \n\
 ⌡  ⌡             \n\
 0  0             \
"""
# 定义Unicode字符串，表示与ASCII字符串等效的积分表达式的数学形式

assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str


def test_pretty_matrix():
    # Empty Matrix
    expr = Matrix()
    ascii_str = "[]"
    unicode_str = "[]"
    assert pretty(expr) == ascii_str
    assert upretty(expr) == unicode_str

    expr = Matrix(2, 0, lambda i, j: 0)
    assert pretty(expr) == ascii_str
    assert upretty(expr) == unicode_str

    expr = Matrix(0, 2, lambda i, j: 0)
    assert pretty(expr) == ascii_str
    assert upretty(expr) == unicode_str

    expr = Matrix([[x**2 + 1, 1], [y, x + y]])
    ascii_str_1 = \
"""\
[     2       ]
[1 + x     1  ]
[             ]
[  y     x + y]\
"""
ascii_str_2 = \
"""\
[ 2           ]
[x  + 1    1  ]
[             ]
[  y     x + y]\
"""
# 定义ASCII字符串，表示不同形状的矩阵表达式的数学形式

ucode_str_1 = \
"""\
⎡     2       ⎤
⎢1 + x     1  ⎥
⎢             ⎥
⎣  y     x + y⎦\
"""
ucode_str_2 = \
"""\
⎡ 2           ⎤
⎢x  + 1    1  ⎥
⎢             ⎥
⎣  y     x + y⎦\
"""
# 定义Unicode字符串，表示与ASCII字符串等效的矩阵表达式的数学形式

assert pretty(expr) in [ascii_str_1, ascii_str_2]
assert upretty(expr) in [ucode_str_1, ucode_str_2]

expr = Matrix([[x/y, y, th], [0, exp(I*k*ph), 1]])
ascii_str = \
"""\
[x                 ]
[-     y      theta]
[y                 ]
[                  ]
[    I*k*phi       ]
[0  e           1  ]\
"""
# 定义ASCII字符串，表示一个包含特定数学元素的矩阵表达式的数学形式

ucode_str = \
"""\
⎡x           ⎤
⎢─    y     θ⎥
⎢y           ⎥
⎢            ⎥
⎢    ⅈ⋅k⋅φ   ⎥
⎣0  ℯ       1⎦\
"""
# 定义Unicode字符串，表示与ASCII字符串等效的矩阵表达式的数学形式

assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str

unicode_str = \
"""\
⎡v̇_msc_00     0         0    ⎤
⎢                            ⎥
⎢   0      v̇_msc_01     0    ⎥
⎢                            ⎥
⎣   0         0      v̇_msc_02⎦\
"""
# 定义Unicode字符串，表示一个特定形状的矩阵表达式的数学形式

expr = diag(*MatrixSymbol('vdot_msc',1,3))
assert upretty(expr) == unicode_str
# 断言，确保生成的Unicode字符串与预期的一致


def test_pretty_ndim_arrays():
    x, y, z, w = symbols("x y z w")
    # 遍历四种不同类型的数组类：ImmutableDenseNDimArray、ImmutableSparseNDimArray、MutableDenseNDimArray、MutableSparseNDimArray
    for ArrayType in (ImmutableDenseNDimArray, ImmutableSparseNDimArray, MutableDenseNDimArray, MutableSparseNDimArray):
        # 创建一个基本的数组 M，使用当前循环的数组类型 ArrayType，并传入单一变量 x
        M = ArrayType(x)

        # 断言：使用 pretty 函数将 M 转换为可读字符串后应为 "x"
        assert pretty(M) == "x"
        # 断言：使用 upretty 函数将 M 转换为 Unicode 可读字符串后应为 "x"
        assert upretty(M) == "x"

        # 创建一个二维数组 M，使用当前循环的数组类型 ArrayType，并传入包含数值的列表
        M = ArrayType([[1/x, y], [z, w]])
        # 创建一个一维数组 M1，使用当前循环的数组类型 ArrayType，并传入包含数值的列表
        M1 = ArrayType([1/x, y, z])

        # 计算 M1 和 M 的张量积，结果存储在 M2 中
        M2 = tensorproduct(M1, M)
        # 计算 M 和 M 的张量积，结果存储在 M3 中
        M3 = tensorproduct(M, M)

        # 定义一个多行字符串 ascii_str，后续会继续添加内容
        ascii_str = \
"""
[1   ]\n\
[-  y]\n\
[x   ]\n\
[    ]\n\
[z  w]\
"""
# 定义一个名为ascii_str的字符串，包含一个ASCII格式的矩阵表示
ascii_str = \
"""\
[1   ]\n\
[-  y]\n\
[x   ]\n\
[    ]\n\
[z  w]\
"""

"""
⎡1   ⎤\n\
⎢─  y⎥\n\
⎢x   ⎥\n\
⎢    ⎥\n\
⎣z  w⎦\
"""
# 定义一个名为ucode_str的字符串，包含一个Unicode格式的矩阵表示
ucode_str = \
"""\
⎡1   ⎤\n\
⎢─  y⎥\n\
⎢x   ⎥\n\
⎢    ⎥\n\
⎣z  w⎦\
"""

# 断言函数pretty(M)的输出结果与ascii_str相等
assert pretty(M) == ascii_str
# 断言函数upretty(M)的输出结果与ucode_str相等
assert upretty(M) == ucode_str

"""
[1      ]\n\
[-  y  z]\n\
[x      ]\
"""
# 定义一个新的ASCII格式的字符串ascii_str，表示另一个矩阵
ascii_str = \
"""\
[1      ]\n\
[-  y  z]\n\
[x      ]\
"""

"""
⎡1      ⎤\n\
⎢─  y  z⎥\n\
⎣x      ⎦\
"""
# 定义一个新的Unicode格式的字符串ucode_str，表示另一个矩阵
ucode_str = \
"""\
⎡1      ⎤\n\
⎢─  y  z⎥\n\
⎣x      ⎦\
"""

# 断言函数pretty(M1)的输出结果与ascii_str相等
assert pretty(M1) == ascii_str
# 断言函数upretty(M1)的输出结果与ucode_str相等
assert upretty(M1) == ucode_str

"""
[[1   y]                       ]\n\
[[--  -]              [z      ]]\n\
[[ 2  x]  [ y    2 ]  [-   y*z]]\n\
[[x    ]  [ -   y  ]  [x      ]]\n\
[[     ]  [ x      ]  [       ]]\n\
[[z   w]  [        ]  [ 2     ]]\n\
[[-   -]  [y*z  w*y]  [z   w*z]]\n\
[[x   x]                       ]\
"""
# 定义一个新的ASCII格式的字符串ascii_str，表示一个更复杂的矩阵
ascii_str = \
"""\
[[1   y]                       ]\n\
[[--  -]              [z      ]]\n\
[[ 2  x]  [ y    2 ]  [-   y*z]]\n\
[[x    ]  [ -   y  ]  [x      ]]\n\
[[     ]  [ x      ]  [       ]]\n\
[[z   w]  [        ]  [ 2     ]]\n\
[[-   -]  [y*z  w*y]  [z   w*z]]\n\
[[x   x]                       ]\
"""

"""
⎡⎡1   y⎤                       ⎤\n\
⎢⎢──  ─⎥              ⎡z      ⎤⎥\n\
⎢⎢ 2  x⎥  ⎡ y    2 ⎤  ⎢─   y⋅z⎥⎥\n\
⎢⎢x    ⎥  ⎢ ─   y  ⎥  ⎢x      ⎥⎥\n\
⎢⎢     ⎥  ⎢ x      ⎥  ⎢       ⎥⎥\n\
⎢⎢z   w⎥  ⎢        ⎥  ⎢ 2     ⎥⎥\n\
⎢⎢─   ─⎥  ⎣y⋅z  w⋅y⎦  ⎣z   w⋅z⎦⎥\n\
⎣⎣x   x⎦                       ⎦\
"""
# 定义一个新的Unicode格式的字符串ucode_str，表示相同的复杂矩阵
ucode_str = \
"""\
⎡⎡1   y⎤                       ⎤\n\
⎢⎢──  ─⎥              ⎡z      ⎤⎥\n\
⎢⎢ 2  x⎥  ⎡ y    2 ⎤  ⎢─   y⋅z⎥⎥\n\
⎢⎢x    ⎥  ⎢ ─   y  ⎥  ⎢x      ⎥⎥\n\
⎢⎢     ⎥  ⎢ x      ⎥  ⎢       ⎥⎥\n\
⎢⎢z   w⎥  ⎢        ⎥  ⎢ 2     ⎥⎥\n\
⎢⎢─   ─⎥  ⎣y⋅z  w⋅y⎦  ⎣z   w⋅z⎦⎥\n\
⎣⎣x   x⎦                       ⎦\
"""

# 断言函数pretty(M2)的输出结果与ascii_str相等
assert pretty(M2) == ascii_str
# 断言函数upretty(M2)的输出结果与ucode_str相等
assert upretty(M2) == ucode_str

"""
[ [1   y]             ]\n\
[ [--  -]             ]\n\
[ [ 2  x]   [ y    2 ]]\n\
[ [x    ]   [ -   y  ]]\n\
[ [     ]   [ x      ]]\n\
[ [z   w]   [        ]]\n\
[ [-   -]   [y*z  w*y]]\n\
[ [x   x]             ]\n\
[                     ]\n\
[[z      ]  [ w      ]]\n\
[[-   y*z]  [ -   w*y]]\n\
[[x      ]  [ x      ]]\n\
[[       ]  [        ]]\n\
[[ 2     ]  [      2 ]]\n\
[[z   w*z]  [w*z  w  ]]\
"""
# 定义一个新的ASCII格式的字符串ascii_str，表示一个更大的矩阵
ascii_str = \
"""\
[ [1   y]             ]\n\
[ [--  -]             ]\n\
[ [ 2  x]   [ y    2 ]]\n\
[ [x    ]   [ -   y  ]]\n\
[ [     ]   [ x      ]]\n\
[ [z   w]   [        ]]\n\
[ [-   -]   [y*z  w*y]]\n\
[ [x   x]             ]\n\
[                     ]\n\
[[z      ]  [ w      ]]\n\
[[-   y*z]  [ -   w*y]]\n\
[[x      ]  [ x      ]]\n\
[[       ]  [        ]]\n\
[[ 2     ]  [      2 ]]\n\
[[z   w*z]  [w*z  w  ]]\
"""

"""
⎡ ⎡1   y⎤             ⎤\n\
⎢ ⎢──  ─⎥             ⎥\n\
⎢ ⎢ 2  x⎥   ⎡ y    2 ⎤⎥\n\
⎢ ⎢x    ⎥   ⎢ ─   y  ⎥⎥\n\
⎢ ⎢     ⎥   ⎢ x      ⎥⎥\n\
⎢ ⎢z   w⎥   ⎢        ⎥⎥\n\
⎢ ⎢─   ─⎥   ⎣y⋅z  w⋅y⎦⎥\n
# 定义测试函数 test_tensor_TensorProduct，用于测试 TensorProduct 的符号表示
def test_tensor_TensorProduct():
    # 创建符号矩阵 A 和 B，每个矩阵大小为 3x3
    A = MatrixSymbol("A", 3, 3)
    B = MatrixSymbol("B", 3, 3)
    # 断言 TensorProduct(A, B) 的 Unicode 表示为 "A⊗B"
    assert upretty(TensorProduct(A, B)) == "A⊗B"
    # 断言 TensorProduct(A, B, A) 的 Unicode 表示为 "A⊗B⊗A"
    assert upretty(TensorProduct(A, B, A)) == "A⊗B⊗A"


# 定义测试函数 test_diffgeom_print_WedgeProduct，用于测试 WedgeProduct 的符号表示
def test_diffgeom_print_WedgeProduct():
    # 导入符号计算库中的 R2 和 WedgeProduct 类
    from sympy.diffgeom.rn import R2
    from sympy.diffgeom import WedgeProduct
    # 创建二维空间 R2 的微分形式 dx 和 dy 的 WedgeProduct 对象 wp
    wp = WedgeProduct(R2.dx, R2.dy)
    # 断言 upretty(wp) 的 Unicode 表示为 "ⅆx∧ⅆy"
    assert upretty(wp) == "ⅆx∧ⅆy"
    # 断言 pretty(wp) 的表示为 "d x/\d y"
    assert pretty(wp) == r"d x/\d y"


# 定义测试函数 test_Adjoint，用于测试 Adjoint 的符号表示
def test_Adjoint():
    # 创建符号矩阵 X 和 Y，每个矩阵大小为 2x2
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    # 断言 pretty(Adjoint(X)) 的表示为 " +\nX "
    assert pretty(Adjoint(X)) == " +\nX "
    # 断言 pretty(Adjoint(X + Y)) 的表示为 "       +\n(X + Y) "
    assert pretty(Adjoint(X + Y)) == "       +\n(X + Y) "
    # 断言 pretty(Adjoint(X) + Adjoint(Y)) 的表示为 " +    +\nX  + Y "
    assert pretty(Adjoint(X) + Adjoint(Y)) == " +    +\nX  + Y "
    # 断言 pretty(Adjoint(X*Y)) 的表示为 "     +\n(X*Y) "
    assert pretty(Adjoint(X*Y)) == "     +\n(X*Y) "
    # 断言 pretty(Adjoint(Y)*Adjoint(X)) 的表示为 " +  +\nY *X "
    assert pretty(Adjoint(Y)*Adjoint(X)) == " +  +\nY *X "
    # 断言 pretty(Adjoint(X**2)) 的表示为 "    +\n/ 2\\ \n\\X / "
    assert pretty(Adjoint(X**2)) == "    +\n/ 2\\ \n\\X / "
    # 断言 pretty(Adjoint(X)**2) 的表示为 "    2\n/ +\\ \n\\X / "
    assert pretty(Adjoint(X)**2) == "    2\n/ +\\ \n\\X / "
    # 断言 pretty(Adjoint(Inverse(X))) 的表示为 "     +\n/ -1\\ \n\\X  / "
    assert pretty(Adjoint(Inverse(X))) == "     +\n/ -1\\ \n\\X  / "
    # 断言 pretty(Inverse(Adjoint(X))) 的表示为 "    -1\n/ +\\  \n\\X /  "
    assert pretty(Inverse(Adjoint(X))) == "    -1\n/ +\\  \n\\X /  "
    # 断言 pretty(Adjoint(Transpose(X))) 的表示为 "    +\n/ T\\ \n\\X / "
    assert pretty(Adjoint(Transpose(X))) == "    +\n/ T\\ \n\\X / "
    # 断言 pretty(Transpose(Adjoint(X))) 的表示为 "    T\n/ +\\ \n\\X / "
    assert pretty(Transpose(Adjoint(X))) == "    T\n/ +\\ \n\\X / "
    # 断言 upretty(Adjoint(X)) 的 Unicode 表示为 "†\nX "
    assert upretty(Adjoint(X)) == "†\nX "
    # 断言 upretty(Adjoint(X + Y)) 的 Unicode 表示为 "       †\n(X + Y) "
    assert upretty(Adjoint(X + Y)) == "       †\n(X + Y) "
    # 断言 upretty(Adjoint(X) + Adjoint(Y)) 的 Unicode 表示为 " †    †\nX  + Y "
    assert upretty(Adjoint(X) + Adjoint(Y)) == " †    †\nX  + Y "
    # 断言 upretty(Adjoint(X*Y)) 的 Unicode 表示为 "     †\n(X⋅Y) "
    assert upretty(Adjoint(X*Y)) == "     †\n(X⋅Y) "
    # 断言 upretty(Adjoint(Y)*Adjoint(X)) 的 Unicode 表示为 " †  †\nY ⋅X "
    assert upretty(Adjoint(Y)*Adjoint(X)) == " †  †\nY ⋅X "
    # 断言 upretty(Adjoint(X**2)) 的 Unicode 表示为 "    †\n⎛ 2⎞ \n⎝X ⎠ "
    assert upretty(Adjoint(X**2)) == "    †\n⎛ 2⎞ \n⎝X ⎠ "
    # 断言 upretty(Adjoint(X)**2) 的 Unicode 表示为 "    2\n⎛ †⎞ \n⎝X ⎠ "
    assert upretty(Adjoint(X)**2) == "    2\n⎛ †⎞ \n⎝X ⎠ "
    # 创建矩阵 m，内容为 ((1, 2), (3, 4))
    m = Matrix(((1, 2), (3, 4)))
    # 断言 upretty(Adjoint(m)) 的 Unicode 表示为
    # '      †\n⎡1  2⎤ \n⎢    ⎥ \n⎣3  4⎦ '
    assert upretty(Adjoint(m)) == \
        '      †\n⎡1  2⎤ \n⎢    ⎥ \n⎣3  4⎦ '
    # 断言 upretty(Adjoint(m+X)) 的 Unicode 表示为
    # '            †\n⎛⎡1  2⎤    ⎞ \n⎜⎢    ⎥ + X⎟ \n⎝⎣3  4⎦    ⎠ '
    assert upretty(Adjoint(m+X)) == \
        '            †\n⎛⎡1  2⎤    ⎞ \n⎜⎢    ⎥ + X⎟ \n⎝⎣3  4⎦    ⎠ '
    # 创建块矩阵 BlockMatrix(((OneMatrix(2, 2), X), (m, ZeroMatrix(2, 2))))
    # 断言 upretty(Adjoint(BlockMatrix(...))) 的 Unicode 表示为
    # '           †\n⎡  𝟙     X⎤ \n⎢         ⎥ \n⎢⎡1  2⎤   ⎥ \n⎢⎢    ⎥  𝟘⎥ \n⎣⎣3  4⎦   ⎦ '
    assert upretty(Adjoint(BlockMatrix(((OneMatrix(2, 2), X), (m, ZeroMatrix(2, 2)))))) == \
        '           †\n⎡  𝟙     X⎤ \n⎢         ⎥ \n⎢⎡1  2⎤   ⎥ \n⎢⎢    ⎥  𝟘⎥ \n⎣⎣3  4⎦   ⎦ '


# 定义测试函数 test
    # 断言：验证 Transpose(X) 的平方的美化输出是否为指定的字符串
    assert pretty(Transpose(X)**2) == "    2\n/ T\\ \n\\X / "
    
    # 断言：验证 Transpose(X) 的逆矩阵的美化输出是否为指定的字符串
    assert pretty(Transpose(Inverse(X))) == "     T\n/ -1\\ \n\\X  / "
    
    # 断言：验证 Inverse(Transpose(X)) 的美化输出是否为指定的字符串
    assert pretty(Inverse(Transpose(X))) == "    -1\n/ T\\  \n\\X /  "
    
    # 断言：验证 Transpose(X) 的 Unicode 美化输出是否为指定的字符串
    assert upretty(Transpose(X)) == " T\nX "
    
    # 断言：验证 Transpose(X + Y) 的 Unicode 美化输出是否为指定的字符串
    assert upretty(Transpose(X + Y)) == "       T\n(X + Y) "
    
    # 断言：验证 Transpose(X) + Transpose(Y) 的 Unicode 美化输出是否为指定的字符串
    assert upretty(Transpose(X) + Transpose(Y)) == " T    T\nX  + Y "
    
    # 断言：验证 Transpose(X*Y) 的 Unicode 美化输出是否为指定的字符串
    assert upretty(Transpose(X*Y)) == "     T\n(X⋅Y) "
    
    # 断言：验证 Transpose(Y)*Transpose(X) 的 Unicode 美化输出是否为指定的字符串
    assert upretty(Transpose(Y)*Transpose(X)) == " T  T\nY ⋅X "
    
    # 断言：验证 Transpose(X**2) 的 Unicode 美化输出是否为指定的字符串
    assert upretty(Transpose(X**2)) == \
        "    T\n⎛ 2⎞ \n⎝X ⎠ "
    
    # 断言：验证 Transpose(X)**2 的 Unicode 美化输出是否为指定的字符串
    assert upretty(Transpose(X)**2) == \
        "    2\n⎛ T⎞ \n⎝X ⎠ "
    
    # 断言：验证 Transpose(Inverse(X)) 的 Unicode 美化输出是否为指定的字符串
    assert upretty(Transpose(Inverse(X))) == \
        "     T\n⎛ -1⎞ \n⎝X  ⎠ "
    
    # 创建矩阵 m
    m = Matrix(((1, 2), (3, 4)))
    
    # 断言：验证 Transpose(m) 的 Unicode 美化输出是否为指定的字符串
    assert upretty(Transpose(m)) == \
        '      T\n'\
        '⎡1  2⎤ \n'\
        '⎢    ⎥ \n'\
        '⎣3  4⎦ '
    
    # 断言：验证 Transpose(m+X) 的 Unicode 美化输出是否为指定的字符串
    assert upretty(Transpose(m+X)) == \
        '            T\n'\
        '⎛⎡1  2⎤    ⎞ \n'\
        '⎜⎢    ⎥ + X⎟ \n'\
        '⎝⎣3  4⎦    ⎠ '
    
    # 断言：验证 Transpose(BlockMatrix(((OneMatrix(2, 2), X), (m, ZeroMatrix(2, 2))))) 的 Unicode 美化输出是否为指定的字符串
    assert upretty(Transpose(BlockMatrix(((OneMatrix(2, 2), X),
                                         (m, ZeroMatrix(2, 2)))))) == \
        '           T\n'\
        '⎡  𝟙     X⎤ \n'\
        '⎢         ⎥ \n'\
        '⎢⎡1  2⎤   ⎥ \n'\
        '⎢⎢    ⎥  𝟘⎥ \n'\
        '⎣⎣3  4⎦   ⎦ '
def test_MatrixSlice():
    # 定义整数符号变量 n
    n = Symbol('n', integer=True)
    # 定义符号变量 x, y, z, w, t
    x, y, z, w, t, = symbols('x y z w t')
    # 定义 n x n 的符号矩阵 X
    X = MatrixSymbol('X', n, n)
    # 定义 10 x 10 的符号矩阵 Y
    Y = MatrixSymbol('Y', 10, 10)
    # 定义 10 x 10 的符号矩阵 Z
    Z = MatrixSymbol('Z', 10, 10)

    # 创建矩阵切片表达式 expr，选择全部元素
    expr = MatrixSlice(X, (None, None, None), (None, None, None))
    assert pretty(expr) == upretty(expr) == 'X[:, :]'
    # 创建矩阵切片表达式 expr，选择特定行和列的子集
    expr = X[x:x + 1, y:y + 1]
    assert pretty(expr) == upretty(expr) == 'X[x:x + 1, y:y + 1]'
    # 创建矩阵切片表达式 expr，选择特定行和列的子集，并增加步长
    expr = X[x:x + 1:2, y:y + 1:2]
    assert pretty(expr) == upretty(expr) == 'X[x:x + 1:2, y:y + 1:2]'
    # 创建矩阵切片表达式 expr，选择前 x 行和后 y 列的子集
    expr = X[:x, y:]
    assert pretty(expr) == upretty(expr) == 'X[:x, y:]'
    # 创建矩阵切片表达式 expr，选择前 x 行和前 y 列的子集
    expr = X[:x, y:]
    assert pretty(expr) == upretty(expr) == 'X[:x, y:]'
    # 创建矩阵切片表达式 expr，选择从第 x 行到末尾和前 y 列的子集
    expr = X[x:, :y]
    assert pretty(expr) == upretty(expr) == 'X[x:, :y]'
    # 创建矩阵切片表达式 expr，选择从第 x 行到第 y 行和从第 z 列到第 w 列的子集
    expr = X[x:y, z:w]
    assert pretty(expr) == upretty(expr) == 'X[x:y, z:w]'
    # 创建矩阵切片表达式 expr，选择从第 x 行到第 y 行和从第 w 列到第 t 列，步长为 t 列到第 x 列的子集
    expr = X[x:y:t, w:t:x]
    assert pretty(expr) == upretty(expr) == 'X[x:y:t, w:t:x]'
    # 创建矩阵切片表达式 expr，选择从第 x 行开始，步长为 y 行，和从第 t 行开始，步长为 w 的列的子集
    expr = X[x::y, t::w]
    assert pretty(expr) == upretty(expr) == 'X[x::y, t::w]'
    # 创建矩阵切片表达式 expr，选择前 x 行，步长为 y 行，和前 t 列，步长为 w 的列的子集
    expr = X[:x:y, :t:w]
    assert pretty(expr) == upretty(expr) == 'X[:x:y, :t:w]'
    # 创建矩阵切片表达式 expr，选择步长为 x 的行和步长为 y 的列的子集
    expr = X[::x, ::y]
    assert pretty(expr) == upretty(expr) == 'X[::x, ::y]'
    # 创建矩阵切片表达式 expr，选择从第 0 行到末尾和从第 0 列到末尾的子集
    expr = MatrixSlice(X, (0, None, None), (0, None, None))
    assert pretty(expr) == upretty(expr) == 'X[:, :]'
    # 创建矩阵切片表达式 expr，选择从第 0 行到第 n 行和从第 0 列到第 n 列的子集
    expr = MatrixSlice(X, (None, n, None), (None, n, None))
    assert pretty(expr) == upretty(expr) == 'X[:, :]'
    # 创建矩阵切片表达式 expr，选择从第 0 行到第 n 行和从第 0 列到第 n 列的子集
    expr = MatrixSlice(X, (0, n, None), (0, n, None))
    assert pretty(expr) == upretty(expr) == 'X[:, :]'
    # 创建矩阵切片表达式 expr，选择从第 0 行到第 n 行，步长为 2，和从第 0 列到第 n 列，步长为 2 的子集
    expr = MatrixSlice(X, (0, n, 2), (0, n, 2))
    assert pretty(expr) == upretty(expr) == 'X[::2, ::2]'
    # 创建矩阵切片表达式 expr，选择从第 1 行到第 2 行，步长为 3，和从第 4 列到第 5 列，步长为 6 的子集
    expr = X[1:2:3, 4:5:6]
    assert pretty(expr) == upretty(expr) == 'X[1:2:3, 4:5:6]'
    # 创建矩阵切片表达式 expr，选择从第 1 行到第 3 行，步长为 5，和从第 4 列到第 6 列，步长为 8 的子集
    expr = X[1:3:5, 4:6:8]
    assert pretty(expr) == upretty(expr) == 'X[1:3:5, 4:6:8]'
    # 创建矩阵切片表达式 expr，选择从第 1 行到第 10 行，步长为 2，和从第 1 列到第 9 列，步长为 2 的子集
    expr = X[1:10:2]
    assert pretty(expr) == upretty(expr) == 'X[1:10:2, :]'
    # 创建矩阵切片表达式 expr，选择从第 0 行到第 5 行，和从第 1 列到第 9 列，步长为 2 的子集
    expr = Y[:5, 1:9:2]
    assert pretty(expr) == upretty(expr) == 'Y[:5, 1:9:2]'
    # 创建矩阵切片表达式 expr，选择从第 0 行到第 5 行，和从第 1 列到第 10 列，步长为 2 的子集
    expr = Y[:5, 1:10:2]
    assert pretty(expr) == upretty(expr) == 'Y[:5, 1::2]'
    # 创建矩阵切片表达式 expr，选择第 5 行，和从第 0 列到第 5 列，步长为 2 的子集
    expr = Y[5, :5:2]
    assert pretty(expr) == upretty(expr) == 'Y[5:6, :5:2]'
    # 创建矩阵切片表达式 expr，选择从第 0 行到第 1 行，和从第 0 列到第 1 列的子集
    expr = X[0:1, 0:1]
    assert pretty(expr) == upretty(expr) == 'X[:1, :1]'
    # 创建矩阵切片表达式 expr，选择从第 0 行到第 1 行，步长为 2，和从第 0 列到第 1 列，步长为 2 的子集
    expr = X[0:1:2, 0:1:2]
    assert pretty(expr) == upretty(expr) == 'X[:1:2, :1:2]'
    # 创建矩阵表达式 (Y + Z) 的矩阵切片 expr，选择从第
    # 使用 assert 断言来验证表达式 pretty(expr) == upretty(expr) == '(Y + Z)[2:, 2:]'
    assert pretty(expr) == upretty(expr) == '(Y + Z)[2:, 2:]'
def test_MatrixExpressions():
    # 定义符号变量 n
    n = Symbol('n', integer=True)
    # 定义一个 n × n 的矩阵符号 X
    X = MatrixSymbol('X', n, n)

    # 断言矩阵 X 的美化字符串与 Unicode 美化字符串都为 "X"
    assert pretty(X) == upretty(X) == "X"

    # 对矩阵 X.T * X 应用 sin 函数，生成新的表达式
    expr = (X.T * X).applyfunc(sin)

    # ASCII 格式的字符串表示
    ascii_str = """\
              / T  \\\n\
(d -> sin(d)).\\X *X/\
"""
    # Unicode 格式的字符串表示
    ucode_str = """\
             ⎛ T  ⎞\n\
(d ↦ sin(d))˳⎝X ⋅X⎠\
"""
    # 断言美化后的表达式与预期的 ASCII 和 Unicode 字符串相等
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    # 定义一个 lambda 函数 lamda(x) = 1/x
    lamda = Lambda(x, 1/x)
    # 对 n*X 应用 lamda 函数，生成新的表达式
    expr = (n * X).applyfunc(lamda)
    # ASCII 格式的字符串表示
    ascii_str = """\
/     1\\      \n\
|x -> -|.(n*X)\n\
\\     x/      \
"""
    # Unicode 格式的字符串表示
    ucode_str = """\
⎛    1⎞      \n\
⎜x ↦ ─⎟˳(n⋅X)\n\
⎝    x⎠      \
"""
    # 断言美化后的表达式与预期的 ASCII 和 Unicode 字符串相等
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_pretty_dotproduct():
    # 导入 DotProduct 类
    from sympy.matrices.expressions.dotproduct import DotProduct
    # 定义符号变量 n
    n = symbols("n", integer=True)
    # 定义两个 n × 1 的矩阵符号 A 和 B
    A = MatrixSymbol('A', n, 1)
    B = MatrixSymbol('B', n, 1)
    # 定义一个 1 × 3 的矩阵 C 和 D
    C = Matrix(1, 3, [1, 2, 3])
    D = Matrix(1, 3, [1, 3, 4])

    # 断言 DotProduct(A, B) 的美化字符串
    assert pretty(DotProduct(A, B)) == "A*B"
    # 断言 DotProduct(C, D) 的美化字符串
    assert pretty(DotProduct(C, D)) == "[1  2  3]*[1  3  4]"
    # 断言 DotProduct(A, B) 的 Unicode 美化字符串
    assert upretty(DotProduct(A, B)) == "A⋅B"
    # 断言 DotProduct(C, D) 的 Unicode 美化字符串
    assert upretty(DotProduct(C, D)) == "[1  2  3]⋅[1  3  4]"


def test_pretty_Determinant():
    # 导入所需的类和函数
    from sympy.matrices import Determinant, Inverse, BlockMatrix, OneMatrix, ZeroMatrix
    # 定义一个 2 × 2 的矩阵 m
    m = Matrix(((1, 2), (3, 4)))

    # 断言矩阵 m 的行列式的 Unicode 美化字符串
    assert upretty(Determinant(m)) == '│1  2│\n│    │\n│3  4│'
    # 断言矩阵 m 的逆矩阵的行列式的 Unicode 美化字符串
    assert upretty(Determinant(Inverse(m))) == \
        '│      -1│\n'\
        '│⎡1  2⎤  │\n'\
        '│⎢    ⎥  │\n'\
        '│⎣3  4⎦  │'
    # 定义一个 2 × 2 的矩阵符号 X
    X = MatrixSymbol('X', 2, 2)
    # 断言矩阵符号 X 的行列式的 Unicode 美化字符串
    assert upretty(Determinant(X)) == '│X│'
    # 断言 BlockMatrix 的行列式的 Unicode 美化字符串
    assert upretty(Determinant(X + m)) == \
        '│⎡1  2⎤    │\n'\
        '│⎢    ⎥ + X│\n'\
        '│⎣3  4⎦    │'

    # 定义符号变量 n
    n = Symbol('n', integer=True)
    # 定义一个 n × n 的矩阵符号 X
    X = MatrixSymbol('X', n, n)

    # 断言矩阵 X 的美化字符串与 Unicode 美化字符串都为 "X"
    assert pretty(X) == upretty(X) == "X"

    # 对矩阵 X.T * X 应用 sin 函数，生成新的表达式
    expr = (X.T * X).applyfunc(sin)

    # ASCII 格式的字符串表示
    ascii_str = """\
              / T  \\\n\
(d -> sin(d)).\\X *X/\
"""
    # Unicode 格式的字符串表示
    ucode_str = """\
             ⎛ T  ⎞\n\
(d ↦ sin(d))˳⎝X ⋅X⎠\
"""
    # 断言美化后的表达式与预期的 ASCII 和 Unicode 字符串相等
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    # 定义一个 lambda 函数 lamda(x) = 1/x
    lamda = Lambda(x, 1/x)
    # 对 n*X 应用 lamda 函数，生成新的表达式
    expr = (n * X).applyfunc(lamda)
    # ASCII 格式的字符串表示
    ascii_str = """\
/     1\\      \n\
|x -> -|.(n*X)\n\
\\     x/      \
"""
    # Unicode 格式的字符串表示
    ucode_str = """\
⎛    1⎞      \n\
⎜x ↦ ─⎟˳(n⋅X)\n\
⎝    x⎠      \
"""
    # 断言美化后的表达式与预期的 ASCII 和 Unicode 字符串相等
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_pretty_dotproduct():
    # 导入 DotProduct 类
    from sympy.matrices.expressions.dotproduct import DotProduct
    # 定义符号变量 n
    n = symbols("n", integer=True)
    # 定义两个 n × 1 的矩阵符号 A 和 B
    A = MatrixSymbol('A', n, 1)
    B = MatrixSymbol('B', n, 1)
    # 定义一个 1 × 3 的矩阵 C 和 D
    C = Matrix(1, 3, [1, 2, 3])
    D = Matrix(1, 3, [1, 3, 4])

    # 断言 DotProduct(A, B) 的美化字符串
    assert pretty(DotProduct(A, B)) == "A*B"
    # 断言 DotProduct(C, D) 的美化字符串
    assert pretty(DotProduct(C, D)) == "[1  2  3]*[1  3  4]"
    # 断言 DotProduct(A, B) 的 Unicode 美化字符串
    assert upretty(DotProduct(A, B)) == "A⋅B"
    # 断言 DotProduct(C, D) 的 Unicode 美化字符串
    assert upretty(DotProduct(C, D)) == "[1  2  3]⋅[1  3  4]"


def test_pretty_Determinant():
    # 导入所需的类和函数
    from sympy.matrices import Determinant, Inverse, BlockMatrix, OneMatrix, ZeroMatrix
    # 定义一个 2 × 2 的矩阵 m
    m = Matrix(((1, 2), (3, 4)))

    # 断言矩阵 m 的行列式的 Unicode 美化字符串
    assert upretty(Determinant(m)) == '│1  2│\n│    │\n│3  4│'
    # 断言矩阵 m 的逆矩阵的行列式的 Unicode 美化字符串
    assert upretty(Determinant(Inverse(m))) == \
        '│      -1│\n'\
        '│⎡1  2⎤  │\n'\
        '│⎢    ⎥  │\n'\
        '│⎣3  4⎦  │'
    # 定义一个 2 × 2 的矩阵符号 X
    X = MatrixSymbol('X', 2, 2)
    # 断言矩阵符号 X 的行列式的 Unicode 美化字符串
    assert upretty(Determinant(X)) == '│X│'
    # 断言 BlockMatrix 的行列式的 Unicode 美化字符串
    assert upretty(Determinant(X + m)) == \
        '│⎡1  2⎤    │\n'\
        '│⎢    ⎥ + X│\n'\
        '│⎣3  4⎦    │'
"""\
                      //x            \\    \n\
                      ||-   for x < 2|    \n\
                      ||y            |    \n\
    //x  for x > 0\\   ||             |    \n\
x + |<            | + |< 2           | + 1\n\
    \\\\y  otherwise/   ||y   for x > 2|    \n\
                      ||             |    \n\
                      ||1   otherwise|    \n\
                      \\\\             /    \
"""
# ASCII 表示的数学表达式字符串，包含条件分段定义的表达式
ascii_str = \
"""\
                      ⎛⎧x            ⎞    \n\
                      ⎜⎪─   for x < 2⎟    \n\
                      ⎜⎪y            ⎟    \n\
    ⎛⎧x  for x > 0⎞   ⎜⎪             ⎟    \n\
x + ⎜⎨            ⎟ + ⎜⎨ 2           ⎟ + 1\n\
    ⎝⎩y  otherwise⎠   ⎜⎪y   for x > 2⎟    \n\
                      ⎜⎪             ⎟    \n\
                      ⎜⎪1   otherwise⎟    \n\
                      ⎝⎩             ⎠    \
"""
# Unicode 表示的数学表达式字符串，包含条件分段定义的表达式
ucode_str = \
"""\
                      ⎛⎧x            ⎞    \n\
                      ⎜⎪─   for x < 2⎟    \n\
                      ⎜⎪y            ⎟    \n\
    ⎛⎧x  for x > 0⎞   ⎜⎪             ⎟    \n\
x + ⎜⎨            ⎟ + ⎜⎨ 2           ⎟ + 1\n\
    ⎝⎩y  otherwise⎠   ⎜⎪y   for x > 2⎟    \n\
                      ⎜⎪             ⎟    \n\
                      ⎜⎪1   otherwise⎟    \n\
                      ⎝⎩             ⎠    \
"""
# 确保使用 ASCII 表示和 Unicode 表示的函数输出结果正确
assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str

# 构建数学表达式，包括条件分段的定义，用于测试输出是否与 ASCII 和 Unicode 表示相符
expr = x - Piecewise((x, x > 0), (y, True)) + Piecewise((x/y, x < 2),
(y**2, x > 2), (1, True)) + 1
ascii_str = \
"""\
                      //x            \\    \n\
                      ||-   for x < 2|    \n\
                      ||y            |    \n\
    //x  for x > 0\\   ||             |    \n\
x - |<            | + |< 2           | + 1\n\
    \\\\y  otherwise/   ||y   for x > 2|    \n\
                      ||             |    \n\
                      ||1   otherwise|    \n\
                      \\\\             /    \
"""
# ASCII 表示的数学表达式字符串，包含条件分段定义的表达式
ascii_str = \
"""\
                      //x            \\    \n\
                      ||-   for x < 2|    \n\
                      ||y            |    \n\
    //x  for x > 0\\   ||             |    \n\
x - |<            | + |< 2           | + 1\n\
    \\\\y  otherwise/   ||y   for x > 2|    \n\
                      ||             |    \n\
                      ||1   otherwise|    \n\
                      \\\\             /    \
"""
# Unicode 表示的数学表达式字符串，包含条件分段定义的表达式
ucode_str = \
"""\
                      ⎛⎧x            ⎞    \n\
                      ⎜⎪─   for x < 2⎟    \n\
                      ⎜⎪y            ⎟    \n\
    ⎛⎧x  for x > 0⎞   ⎜⎪             ⎟    \n\
x - ⎜⎨            ⎟ + ⎜⎨ 2           ⎟ + 1\n\
    ⎝⎩y  otherwise⎠   ⎜⎪y   for x > 2⎟    \n\
                      ⎜⎪             ⎟    \n\
                      ⎜⎪1   otherwise⎟    \n\
                      ⎝⎩             ⎠    \
"""
# 确保使用 ASCII 表示和 Unicode 表示的函数输出结果正确
assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str

# 构建数学表达式，包括条件分段的定义，用于测试输出是否与 ASCII 和 Unicode 表示相符
expr = x*Piecewise((x, x > 0), (y, True))
ascii_str = \
"""\
  //x  for x > 0\\\n\
x*|<            |\n\
  \\\\y  otherwise/\
"""
# ASCII 表示的数学表达式字符串，包含条件分段定义的表达式
ascii_str = \
"""\
  //x  for x > 0\\\n\
x*|<            |\n\
  \\\\y  otherwise/\
"""
# Unicode 表示的数学表达式字符串，包含条件分段定义的表达式
ucode_str = \
"""\
  ⎛⎧x  for x > 0⎞\n\
x⋅⎜⎨            ⎟\n\
  ⎝⎩y  otherwise⎠\
"""
# 确保使用 ASCII 表示和 Unicode 表示的函数输出结果正确
assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str

# 构建数学表达式，包括条件分段的定义，用于测试输出是否与 ASCII 和 Unicode 表示相符
expr = Piecewise((x, x > 0), (y, True))*Piecewise((x/y, x < 2), (y**2, x >
2), (1, True))
ascii_str = \
"""\
                //x            \\\n\
                ||-   for x < 2|\n\
                ||y            |\n\
//x  for x > 0\\ ||             |\n\
|<            |*|< 2           |\n\
\\\\y  otherwise/ ||y   for x > 2|\n\
                ||             |\n\
                ||1   otherwise|\n\
                \\\\             /\
"""
# ASCII 表示的数学表达式字符串，包含条件分段定义的表达式
ascii_str = \
"""\
                //x            \\\n\
                ||-   for x < 2|\n\
                ||y            |\n\
//x  for x > 0\\ ||             |\n\
|<            |*|< 2           |\n\
\\\\y  otherwise/ ||y   for x > 2|\n\
                ||             |\n\
                ||1   otherwise|\n\
                \\\\             /\
"""
# Unicode 表示的数学表达式字符串，包含条件分段定义的表达式
ucode_str = \
"""\
                ⎛⎧x            ⎞\n\
                ⎜⎪─   for x < 2⎟\n\
                ⎜⎪y            ⎟\n\
⎛⎧x  for x > 0⎞ ⎜⎪             ⎟\n\
⎜⎨            ⎟⋅⎜⎨ 2           ⎟\n\
⎝⎩y  otherwise⎠ ⎜⎪y   for x > 2⎟\n\
                ⎜⎪             ⎟\n\
                ⎜⎪1   otherwise⎟\n\
                ⎝⎩             ⎠\
"""
# 确保使用 ASCII 表示和 Unicode 表示的函数输出结果正确
assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str
# 函数定义，计算给定表达式的 ASCII 形式和 Unicode 形式的字符串表示
def test_pretty():
    # 测试 Piecewise 表达式的 ASCII 和 Unicode 形式
    expr = -Piecewise((x, x > 0), (y, True))*Piecewise((x/y, x < 2), (y**2, x > 2), (1, True))
    ascii_str = \
"""\
                 //x            \\\n\
                 ||-   for x < 2|\n\
                 ||y            |\n\
 //x  for x > 0\\ ||             |\n\
-|<            |*|< 2           |\n\
 \\\\y  otherwise/ ||y   for x > 2|\n\
                 ||             |\n\
                 ||1   otherwise|\n\
                 \\\\             /\
"""
    ucode_str = \
"""\
                 ⎛⎧x            ⎞\n\
                 ⎜⎪─   for x < 2⎟\n\
                 ⎜⎪y            ⎟\n\
 ⎛⎧x  for x > 0⎞ ⎜⎪             ⎟\n\
-⎜⎨            ⎟⋅⎜⎨ 2           ⎟\n\
 ⎝⎩y  otherwise⎠ ⎜⎪y   for x > 2⎟\n\
                 ⎜⎪             ⎟\n\
                 ⎜⎪1   otherwise⎟\n\
                 ⎝⎩             ⎠\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    # 测试另一个 Piecewise 表达式的 ASCII 和 Unicode 形式
    expr = Piecewise((0, Abs(1/y) < 1), (1, Abs(y) < 1), (y*meijerg(((2, 1), ()), ((), (1, 0)), 1/y), True))
    ascii_str = \
"""\
/                                 1     \n\
|            0               for --- < 1\n\
|                                |y|    \n\
|                                       \n\
<            1               for |y| < 1\n\
|                                       \n\
|   __0, 2 /1, 2       | 1\\             \n\
|y*/__     |           | -|   otherwise \n\
\\  \\_|2, 2 \\      0, 1 | y/             \
"""
    ucode_str = \
"""\
⎧                                 1     \n\
⎪            0               for ─── < 1\n\
⎪                                │y│    \n\
⎪                                       \n\
⎨            1               for │y│ < 1\n\
⎪                                       \n\
⎪  ╭─╮0, 2 ⎛1, 2       │ 1⎞             \n\
⎪y⋅│╶┐     ⎜           │ ─⎟   otherwise \n\
⎩  ╰─╯2, 2 ⎝      0, 1 │ y⎠             \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    # XXX: 这里使用 evaluate=False 是因为 Piecewise._eval_power 展开了幂运算。
    # 测试 Pow 表达式的 ASCII 和 Unicode 形式
    expr = Pow(Piecewise((x, x > 0), (y, True)), 2, evaluate=False)
    ascii_str = \
"""\
               2\n\
//x  for x > 0\\ \n\
|<            | \n\
\\\\y  otherwise/ \
"""
    ucode_str = \
"""\
               2\n\
⎛⎧x  for x > 0⎞ \n\
⎜⎨            ⎟ \n\
⎝⎩y  otherwise⎠ \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


# 测试 ITE 表达式的 ASCII 和 Unicode 形式
def test_pretty_ITE():
    expr = ITE(x, y, z)
    assert pretty(expr) == (
        '/y    for x  \n'
        '<            \n'
        '\\z  otherwise'
        )
    assert upretty(expr) == """\
⎧y    for x  \n\
⎨            \n\
⎩z  otherwise\
"""


# 测试空序列的 ASCII 和 Unicode 形式
def test_pretty_seq():
    expr = ()
    ascii_str = \
"""\
()\
"""
    ucode_str = \
"""\
()\
"""
    # 断言表达式在美化后与 ASCII 字符串相等
    assert pretty(expr) == ascii_str
    
    # 断言表达式在 Unicode 美化后与 Unicode 字符串相等
    assert upretty(expr) == ucode_str
    
    # 初始化一个空列表 expr
    expr = []
    
    # 继续下一行的赋值操作，将后续的字符串赋给 ascii_str
    ascii_str = \
# 定义一个 ASCII 字符串表达式和 Unicode 字符串表达式，进行断言比较
ascii_str = \
"""\
{}\
"""
ucode_str = \
"""\
{}\
"""

# 断言 pretty 函数对 expr 的输出与 ascii_str 相同
assert pretty(expr) == ascii_str
# 断言 upretty 函数对 expr 的输出与 ucode_str 相同
assert upretty(expr) == ucode_str

# 初始化两个空的字典表达式，进行断言比较
expr = {}
expr_2 = {}
# 断言 pretty 函数对 expr 的输出与 ascii_str 相同
assert pretty(expr) == ascii_str
# 断言 pretty 函数对 expr_2 的输出与 ascii_str 相同
assert pretty(expr_2) == ascii_str
# 断言 upretty 函数对 expr 的输出与 ucode_str 相同
assert upretty(expr) == ucode_str
# 断言 upretty 函数对 expr_2 的输出与 ucode_str 相同
assert upretty(expr_2) == ucode_str

# 初始化一个包含元组的表达式，进行 ASCII 和 Unicode 字符串的断言比较
expr = (1/x,)
ascii_str = \
"""\
 1  \n\
(-,)\n\
 x  \
"""
ucode_str = \
"""\
⎛1 ⎞\n\
⎜─,⎟\n\
⎝x ⎠\
"""
# 断言 pretty 函数对 expr 的输出与 ascii_str 相同
assert pretty(expr) == ascii_str
# 断言 upretty 函数对 expr 的输出与 ucode_str 相同
assert upretty(expr) == ucode_str

# 初始化一个包含多个表达式的列表，进行 ASCII 和 Unicode 字符串的断言比较
expr = [x**2, 1/x, x, y, sin(th)**2/cos(ph)**2]
ascii_str = \
"""\
                 2        \n\
  2  1        sin (theta) \n\
[x , -, x, y, -----------]\n\
     x            2       \n\
               cos (phi)  \
"""
ucode_str = \
"""\
⎡                2   ⎤\n\
⎢ 2  1        sin (θ)⎥\n\
⎢x , ─, x, y, ───────⎥\n\
⎢    x           2   ⎥\n\
⎣             cos (φ)⎦\
"""
# 断言 pretty 函数对 expr 的输出与 ascii_str 相同
assert pretty(expr) == ascii_str
# 断言 upretty 函数对 expr 的输出与 ucode_str 相同
assert upretty(expr) == ucode_str

# 初始化一个包含多个表达式的元组，进行 ASCII 和 Unicode 字符串的断言比较
expr = (x**2, 1/x, x, y, sin(th)**2/cos(ph)**2)
ascii_str = \
"""\
                 2        \n\
  2  1        sin (theta) \n\
(x , -, x, y, -----------)\n\
     x            2       \n\
               cos (phi)  \
"""
ucode_str = \
"""\
⎛                2   ⎞\n\
⎜ 2  1        sin (θ)⎟\n\
⎜x , ─, x, y, ───────⎟\n\
⎜    x           2   ⎟\n\
⎝             cos (φ)⎠\
"""
# 断言 pretty 函数对 expr 的输出与 ascii_str 相同
assert pretty(expr) == ascii_str
# 断言 upretty 函数对 expr 的输出与 ucode_str 相同
assert upretty(expr) == ucode_str

# 初始化一个使用 Tuple 包装的表达式，进行 ASCII 和 Unicode 字符串的断言比较
expr = Tuple(x**2, 1/x, x, y, sin(th)**2/cos(ph)**2)
ascii_str = \
"""\
                 2        \n\
  2  1        sin (theta) \n\
(x , -, x, y, -----------)\n\
     x            2       \n\
               cos (phi)  \
"""
ucode_str = \
"""\
⎛                2   ⎞\n\
⎜ 2  1        sin (θ)⎟\n\
⎜x , ─, x, y, ───────⎟\n\
⎜    x           2   ⎟\n\
⎝             cos (φ)⎠\
"""
# 断言 pretty 函数对 expr 的输出与 ascii_str 相同
assert pretty(expr) == ascii_str
# 断言 upretty 函数对 expr 的输出与 ucode_str 相同
assert upretty(expr) == ucode_str

# 初始化一个包含字典表达式，进行 ASCII 和 Unicode 字符串的断言比较
expr = {x: sin(x)}
expr_2 = Dict({x: sin(x)})
ascii_str = \
"""\
{x: sin(x)}\
"""
ucode_str = \
"""\
{x: sin(x)}\
"""
# 断言 pretty 函数对 expr 的输出与 ascii_str 相同
assert pretty(expr) == ascii_str
# 断言 pretty 函数对 expr_2 的输出与 ascii_str 相同
assert pretty(expr_2) == ascii_str
# 断言 upretty 函数对 expr 的输出与 ucode_str 相同
assert upretty(expr) == ucode_str
# 断言 upretty 函数对 expr_2 的输出与 ucode_str 相同
assert upretty(expr_2) == ucode_str

# 初始化一个包含复杂字典表达式，进行 ASCII 和 Unicode 字符串的断言比较
expr = {1/x: 1/y, x: sin(x)**2}
expr_2 = Dict({1/x: 1/y, x: sin(x)**2})
ascii_str = \
"""\
 1  1        2    \n\
{-: -, x: sin (x)}\n\
 x  y             \
"""
ucode_str = \
"""\
⎧1  1        2   ⎫\n\
⎨─: ─, x: sin (x)⎬\n\
⎩x  y            ⎭\
"""
# 断言 pretty 函数对 expr 的输出与 ascii_str 相同
assert pretty(expr) == ascii_str
# 断言 pretty 函数对 expr_2 的输出与 ascii_str 相同
assert pretty(expr_2) == ascii_str
# 断言 upretty 函数对 expr 的输出与 ucode_str 相同
assert upretty(expr) == ucode_str
# 断言 upretty 函数对 expr_2 的输出与 ucode_str 相同
assert upretty(expr_2) == ucode_str

# 以下是一个修复以前高度偶数序列输出 bug 的示例
expr = [x**2]
ascii_str = \
"""\
  2 \n\
[x ]\
"""
ucode_str = \
"""\
⎡ 2⎤\n\
⎣x ⎦\
"""
# 断言 pretty 函数对 expr 的输出与 ascii_str 相同
assert pretty(expr) == ascii_str
# 断言 upretty 函数对 expr 的输出与 ucode_str 相同
assert upretty(expr) == ucode_str

expr = (x**2,)
ascii_str = \
"""\
  2  \n\
(x ,)\
"""
ucode_str = \
"""\
⎛ 2 ⎞\n\
⎝x ,⎠\
"""
# 断言 pretty 函数对 expr 的输出与 ascii_str 相同
assert pretty(expr) == ascii_str
# 断言 upretty 函数对 expr 的输出与 ucode_str 相同
assert upretty(expr) == ucode_str
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Tuple(x**2)
    ascii_str = \
"""\
  2  \n\
(x ,)\
"""
    ucode_str = \
"""\
⎛ 2 ⎞\n\
⎝x ,⎠\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
"""
# 检查 pretty() 和 upretty() 函数是否正确格式化表达式

expr = Tuple(x**2)
# 创建一个包含 x^2 的元组表达式

ascii_str = \
"""\
  2  \n\
(x ,)\
"""
# 期望的 ASCII 格式字符串

ucode_str = \
"""\
⎛ 2 ⎞\n\
⎝x ,⎠\
"""
# 期望的 Unicode 格式字符串

assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str


expr = {x**2: 1}
# 创建一个字典表达式，键为 x^2，值为 1

expr_2 = Dict({x**2: 1})
# 使用 Dict() 函数创建一个字典表达式，键为 x^2，值为 1

ascii_str = \
"""\
  2    \n\
{x : 1}\
"""
# 期望的 ASCII 格式字符串

ucode_str = \
"""\
⎧ 2   ⎫\n\
⎨x : 1⎬\n\
⎩     ⎭\
"""
# 期望的 Unicode 格式字符串

assert pretty(expr) == ascii_str
assert pretty(expr_2) == ascii_str
assert upretty(expr) == ucode_str
assert upretty(expr_2) == ucode_str
"""


def test_any_object_in_sequence():
    # Cf. issue 5306
    b1 = Basic()
    b2 = Basic(Basic())

    expr = [b2, b1]
    assert pretty(expr) == "[Basic(Basic()), Basic()]"
    assert upretty(expr) == "[Basic(Basic()), Basic()]"

    expr = {b2, b1}
    assert pretty(expr) == "{Basic(), Basic(Basic())}"
    assert upretty(expr) == "{Basic(), Basic(Basic())}"

    expr = {b2: b1, b1: b2}
    expr2 = Dict({b2: b1, b1: b2})
    assert pretty(expr) == "{Basic(): Basic(Basic()), Basic(Basic()): Basic()}"
    assert pretty(
        expr2) == "{Basic(): Basic(Basic()), Basic(Basic()): Basic()}"
    assert upretty(
        expr) == "{Basic(): Basic(Basic()), Basic(Basic()): Basic()}"
    assert upretty(
        expr2) == "{Basic(): Basic(Basic()), Basic(Basic()): Basic()}"


def test_print_builtin_set():
    assert pretty(set()) == 'set()'
    assert upretty(set()) == 'set()'

    assert pretty(frozenset()) == 'frozenset()'
    assert upretty(frozenset()) == 'frozenset()'

    s1 = {1/x, x}
    s2 = frozenset(s1)

    assert pretty(s1) == \
"""\
 1    \n\
{-, x}
 x    \
"""
    assert upretty(s1) == \
"""\
⎧1   ⎫
⎨─, x⎬
⎩x   ⎭\
"""

    assert pretty(s2) == \
"""\
           1     \n\
frozenset({-, x})
           x     \
"""
    assert upretty(s2) == \
"""\
         ⎛⎧1   ⎫⎞
frozenset⎜⎨─, x⎬⎟
         ⎝⎩x   ⎭⎠\
"""


def test_pretty_sets():
    s = FiniteSet
    assert pretty(s(*[x*y, x**2])) == \
"""\
  2      \n\
{x , x*y}\
"""
    assert pretty(s(*range(1, 6))) == "{1, 2, 3, 4, 5}"
    assert pretty(s(*range(1, 13))) == "{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}"

    assert pretty({x*y, x**2}) == \
"""\
  2      \n\
{x , x*y}\
"""
    assert pretty(set(range(1, 6))) == "{1, 2, 3, 4, 5}"
    assert pretty(set(range(1, 13))) == \
        "{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}"

    assert pretty(frozenset([x*y, x**2])) == \
"""\
            2       \n\
frozenset({x , x*y})\
"""
    assert pretty(frozenset(range(1, 6))) == "frozenset({1, 2, 3, 4, 5})"
    assert pretty(frozenset(range(1, 13))) == \
        "frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})"

    assert pretty(Range(0, 3, 1)) == '{0, 1, 2}'

    ascii_str = '{0, 1, ..., 29}'
    ucode_str = '{0, 1, …, 29}'
    assert pretty(Range(0, 30, 1)) == ascii_str
    assert upretty(Range(0, 30, 1)) == ucode_str

    ascii_str = '{30, 29, ..., 2}'
    ucode_str = '{30, 29, …, 2}'
    assert pretty(Range(30, 1, -1)) == ascii_str
    # 断言：验证 Range(30, 1, -1) 的美观输出是否等于 ucode_str
    assert upretty(Range(30, 1, -1)) == ucode_str
    
    # 设定 ASCII 字符串表示的范围
    ascii_str = '{0, 2, ...}'
    # 设定 Unicode 字符串表示的范围
    ucode_str = '{0, 2, …}'
    # 断言：验证 Range(0, oo, 2) 的美观输出是否等于 ASCII 字符串表示的范围
    assert pretty(Range(0, oo, 2)) == ascii_str
    # 断言：验证 Range(0, oo, 2) 的 Unicode 美观输出是否等于 Unicode 字符串表示的范围
    assert upretty(Range(0, oo, 2)) == ucode_str
    
    # 设定 ASCII 字符串表示的范围
    ascii_str = '{..., 2, 0}'
    # 设定 Unicode 字符串表示的范围
    ucode_str = '{…, 2, 0}'
    # 断言：验证 Range(oo, -2, -2) 的美观输出是否等于 ASCII 字符串表示的范围
    assert pretty(Range(oo, -2, -2)) == ascii_str
    # 断言：验证 Range(oo, -2, -2) 的 Unicode 美观输出是否等于 Unicode 字符串表示的范围
    assert upretty(Range(oo, -2, -2)) == ucode_str
    
    # 设定 ASCII 字符串表示的范围
    ascii_str = '{-2, -3, ...}'
    # 设定 Unicode 字符串表示的范围
    ucode_str = '{-2, -3, …}'
    # 断言：验证 Range(-2, -oo, -1) 的美观输出是否等于 ASCII 字符串表示的范围
    assert pretty(Range(-2, -oo, -1)) == ascii_str
    # 断言：验证 Range(-2, -oo, -1) 的 Unicode 美观输出是否等于 Unicode 字符串表示的范围
    assert upretty(Range(-2, -oo, -1)) == ucode_str
# 定义测试函数 test_pretty_SetExpr
def test_pretty_SetExpr():
    # 创建一个区间对象 Interval(1, 3)
    iv = Interval(1, 3)
    # 使用 Interval 对象创建 SetExpr 对象 se
    se = SetExpr(iv)
    # ASCII 表示的预期字符串
    ascii_str = "SetExpr([1, 3])"
    # Unicode 表示的预期字符串
    ucode_str = "SetExpr([1, 3])"
    # 断言 pretty(se) 返回的结果与 ascii_str 相等
    assert pretty(se) == ascii_str
    # 断言 upretty(se) 返回的结果与 ucode_str 相等
    assert upretty(se) == ucode_str


# 定义测试函数 test_pretty_ImageSet
def test_pretty_ImageSet():
    # 创建一个 ImageSet 对象 imgset，使用 Lambda 函数定义映射关系，两个集合作为参数
    imgset = ImageSet(Lambda((x, y), x + y), {1, 2, 3}, {3, 4})
    # ASCII 表示的预期字符串
    ascii_str = '{x + y | x in {1, 2, 3}, y in {3, 4}}'
    # Unicode 表示的预期字符串
    ucode_str = '{x + y │ x ∊ {1, 2, 3}, y ∊ {3, 4}}'
    # 断言 pretty(imgset) 返回的结果与 ascii_str 相等
    assert pretty(imgset) == ascii_str
    # 断言 upretty(imgset) 返回的结果与 ucode_str 相等
    assert upretty(imgset) == ucode_str

    # 创建另一个 ImageSet 对象 imgset，Lambda 函数定义映射关系，ProductSet 作为参数
    imgset = ImageSet(Lambda(((x, y),), x + y), ProductSet({1, 2, 3}, {3, 4}))
    # ASCII 表示的预期字符串
    ascii_str = '{x + y | (x, y) in {1, 2, 3} x {3, 4}}'
    # Unicode 表示的预期字符串
    ucode_str = '{x + y │ (x, y) ∊ {1, 2, 3} × {3, 4}}'
    # 断言 pretty(imgset) 返回的结果与 ascii_str 相等
    assert pretty(imgset) == ascii_str
    # 断言 upretty(imgset) 返回的结果与 ucode_str 相等
    assert upretty(imgset) == ucode_str

    # 创建另一个 ImageSet 对象 imgset，Lambda 函数定义映射关系，使用 S.Naturals 作为参数
    imgset = ImageSet(Lambda(x, x**2), S.Naturals)
    # ASCII 表示的预期字符串（包含多行内容）
    ascii_str = '''\
  2                 \n\
{x  | x in Naturals}'''
    # Unicode 表示的预期字符串（包含多行内容）
    ucode_str = '''\
⎧ 2 │      ⎫\n\
⎨x  │ x ∊ ℕ⎬\n\
⎩   │      ⎭'''
    # 断言 pretty(imgset) 返回的结果与 ascii_str 相等
    assert pretty(imgset) == ascii_str
    # 断言 upretty(imgset) 返回的结果与 ucode_str 相等
    assert upretty(imgset) == ucode_str

    # 创建另一个 ImageSet 对象 imgset，Lambda 函数定义映射关系，使用 S.Naturals 作为参数
    imgset = ImageSet(Lambda(x, 1/x**2), S.Naturals)
    # ASCII 表示的预期字符串（包含多行内容）
    ascii_str = '''\
 1                  \n\
{-- | x in Naturals}
  2                 \n\
 x                  '''
    # Unicode 表示的预期字符串（包含多行内容）
    ucode_str = '''\
⎧1  │      ⎫\n\
⎪── │ x ∊ ℕ⎪\n\
⎨ 2 │      ⎬\n\
⎪x  │      ⎪\n\
⎩   │      ⎭'''
    # 断言 pretty(imgset) 返回的结果与 ascii_str 相等
    assert pretty(imgset) == ascii_str
    # 断言 upretty(imgset) 返回的结果与 ucode_str 相等
    assert upretty(imgset) == ucode_str

    # 创建另一个 ImageSet 对象 imgset，Lambda 函数定义映射关系，两个 S.Naturals 作为参数
    imgset = ImageSet(Lambda((x, y), 1/(x + y)**2), S.Naturals, S.Naturals)
    # ASCII 表示的预期字符串（包含多行内容）
    ascii_str = '''\
    1                                    \n\
{-------- | x in Naturals, y in Naturals}
        2                                \n\
 (x + y)                                 '''
    # Unicode 表示的预期字符串（包含多行内容）
    ucode_str = '''\
⎧   1     │             ⎫
⎪──────── │ x ∊ ℕ, y ∊ ℕ⎪
⎨       2 │             ⎬
⎪(x + y)  │             ⎪
⎩         │             ⎭'''
    # 断言 pretty(imgset) 返回的结果与 ascii_str 相等
    assert pretty(imgset) == ascii_str
    # 断言 upretty(imgset) 返回的结果与 ucode_str 相等
    assert upretty(imgset) == ucode_str

    # issue 23449 centering issue
    # 断言 upretty(Symbol("ihat") / (Symbol("i") + 1)) 返回的结果与预期的 Unicode 字符串相等
    assert upretty([Symbol("ihat") / (Symbol("i") + 1)]) == '''\
⎡  î  ⎤
⎢─────⎥
⎣i + 1⎦\
'''
    # 断言 upretty(Matrix([Symbol("ihat"), Symbol("i") + 1])) 返回的结果与预期的 Unicode 字符串相等
    assert upretty(Matrix([Symbol("ihat"), Symbol("i") + 1])) == '''\
⎡  î  ⎤
⎢     ⎥
⎣i + 1⎦\
'''


# 定义测试函数 test_pretty_ConditionSet
def test_pretty_ConditionSet():
    # ASCII 表示的预期字符串
    ascii_str = '{x | x in (-oo, oo) and sin(x) = 0}'
    # Unicode 表示的预期字符串
    ucode_str = '{x │ x ∊ ℝ ∧ (sin(x) = 0)}'
    # 断言 pretty(ConditionSet(x, Eq(sin(x), 0), S.Reals)) 返回的结果与 ascii_str 相等
    assert pretty(ConditionSet(x, Eq(sin(x), 0), S.Reals)) == ascii_str
    # 断言 upretty(ConditionSet(x, Eq(sin(x), 0), S.Reals)) 返回的结果与 ucode_str 相等
    assert upretty(ConditionSet(x, Eq(sin(x), 0), S.Reals)) == ucode_str

    # 断言 pretty(ConditionSet(x, Contains(x, S.Reals, evaluate=False), FiniteSet(1))) 返回的结果为 '{1}'
    assert pretty(ConditionSet(x, Contains(x, S.Reals, evaluate=False), FiniteSet(1))) == '{1}'
    # 断言 upretty(ConditionSet(x, Contains(x, S.Reals, evaluate=False), FiniteSet(1))) 返回的结果为 '{1}'
    assert upretty(ConditionSet(x, Contains(x, S.Reals, evaluate=False), FiniteSet(1))) == '{1}'

    # 断言 pretty(ConditionSet(x, And(x > 1, x < -1), FiniteSet(1, 2, 3))) 返回的结果为 "EmptySet"
    assert pretty(ConditionSet(x, And(x > 1, x < -1), FiniteSet(1, 2, 3))) == "EmptySet"
    # 断言 upretty(ConditionSet(x, And(x > 1, x < -1), FiniteSet(1, 2, 3))) 返回的结果为 "∅"
    assert upretty(ConditionSet(x, And(x > 1, x < -1), FiniteSet(1, 2, 3))) == "∅"

    # 断言 pretty(ConditionSet(x, Or(x
    # 使用断言检查 ConditionSet 的字符串表示是否符合预期结果 '{2}'
    assert upretty(ConditionSet(x, Or(x > 1, x < -1), FiniteSet(1, 2))) == '{2}'
    
    # 创建一个 ConditionSet 对象 condset，表示 x 满足 1/x**2 > 0 的条件
    condset = ConditionSet(x, 1/x**2 > 0)
    
    # 定义一个多行的 ASCII 字符串，包含一个数字 '1' 和换行符 '\n'
    ascii_str = '''\
     1      \n\
# 定义一个条件集合，用于描述 x 的取值范围
condset = ConditionSet(x, 1/x**2 > 0, S.Reals)
# ASCII 格式的字符串，表示条件集合的数学表达式
ascii_str = '''\
                        1      \n\
{x | x in (-oo, oo) and -- > 0}
                         2     \n\
                        x      '''
# Unicode 格式的字符串，表示条件集合的数学表达式
ucode_str = '''\
⎧  │         ⎛1     ⎞⎫
⎪x │ x ∊ ℝ ∧ ⎜── > 0⎟⎪
⎨  │         ⎜ 2    ⎟⎬
⎪  │         ⎝x     ⎠⎪
⎩  │                 ⎭'''
# 确保 pretty 函数能正确转换条件集合为 ASCII 格式字符串
assert pretty(condset) == ascii_str
# 确保 upretty 函数能正确转换条件集合为 Unicode 格式字符串
assert upretty(condset) == ucode_str


def test_pretty_ComplexRegion():
    from sympy.sets.fancysets import ComplexRegion
    # 创建一个复数区域对象，表示平面上的矩形区域
    cregion = ComplexRegion(Interval(3, 5)*Interval(4, 6))
    # ASCII 格式的字符串，描述复数区域的数学表达式
    ascii_str = '{x + y*I | x, y in [3, 5] x [4, 6]}'
    # Unicode 格式的字符串，描述复数区域的数学表达式
    ucode_str = '{x + y⋅ⅈ │ x, y ∊ [3, 5] × [4, 6]}'
    # 确保 pretty 函数能正确转换复数区域为 ASCII 格式字符串
    assert pretty(cregion) == ascii_str
    # 确保 upretty 函数能正确转换复数区域为 Unicode 格式字符串
    assert upretty(cregion) == ucode_str

    # 创建一个极坐标系下的复数区域对象
    cregion = ComplexRegion(Interval(0, 1)*Interval(0, 2*pi), polar=True)
    # ASCII 格式的字符串，描述极坐标系下复数区域的数学表达式
    ascii_str = '{r*(I*sin(theta) + cos(theta)) | r, theta in [0, 1] x [0, 2*pi)}'
    # Unicode 格式的字符串，描述极坐标系下复数区域的数学表达式
    ucode_str = '{r⋅(ⅈ⋅sin(θ) + cos(θ)) │ r, θ ∊ [0, 1] × [0, 2⋅π)}'
    # 确保 pretty 函数能正确转换复数区域为 ASCII 格式字符串
    assert pretty(cregion) == ascii_str
    # 确保 upretty 函数能正确转换复数区域为 Unicode 格式字符串
    assert upretty(cregion) == ucode_str

    # 创建一个复数区域对象，表示在条件 [3, 1/a**2] x [4, 6] 下的矩形区域
    cregion = ComplexRegion(Interval(3, 1/a**2)*Interval(4, 6))
    # ASCII 格式的字符串，描述复数区域的数学表达式
    ascii_str = '''\
                       1            \n\
{x + y*I | x, y in [3, --] x [4, 6]}
                        2           \n\
                       a            '''
    # Unicode 格式的字符串，描述复数区域的数学表达式
    ucode_str = '''\
⎧        │        ⎡   1 ⎤         ⎫
⎪x + y⋅ⅈ │ x, y ∊ ⎢3, ──⎥ × [4, 6]⎪
⎨        │        ⎢    2⎥         ⎬
⎪        │        ⎣   a ⎦         ⎪
⎩        │                        ⎭'''
    # 确保 pretty 函数能正确转换复数区域为 ASCII 格式字符串
    assert pretty(cregion) == ascii_str
    # 确保 upretty 函数能正确转换复数区域为 Unicode 格式字符串
    assert upretty(cregion) == ucode_str

    # 创建一个极坐标系下的复数区域对象，在条件 [0, 1/a**2] x [0, 2*pi] 下
    cregion = ComplexRegion(Interval(0, 1/a**2)*Interval(0, 2*pi), polar=True)
    # ASCII 格式的字符串，描述极坐标系下复数区域的数学表达式
    ascii_str = '''\
                                                 1               \n\
{r*(I*sin(theta) + cos(theta)) | r, theta in [0, --] x [0, 2*pi)}
                                                  2              \n\
                                                 a               '''
    # Unicode 格式的字符串，描述极坐标系下复数区域的数学表达式
    ucode_str = '''\
⎧                      │        ⎡   1 ⎤           ⎫
⎪r⋅(ⅈ⋅sin(θ) + cos(θ)) │ r, θ ∊ ⎢0, ──⎥ × [0, 2⋅π)⎪
⎨                      │        ⎢    2⎥           ⎬
⎪                      │        ⎣   a ⎦           ⎪
⎩                      │                          ⎭'''
    # 确保 pretty 函数能正确转换复数区域为 ASCII 格式字符串
    assert pretty(cregion) == ascii_str
    # 确保 upretty 函数能正确转换复数区域为 Unicode 格式字符串
    assert upretty(cregion) == ucode_str


def test_pretty_Union_issue_10414():
    a, b = Interval(2, 3), Interval(4, 7)
    # Unicode 格式的字符串，表示两个区间的并集
    ucode_str = '[2, 3] ∪ [4, 7]'
    # ASCII 格式的字符串，表示两个区间的并集
    ascii_str = '[2, 3] U [4, 7]'
    # 确保 upretty 函数能正确转换并集为 Unicode 格式字符串
    assert upretty(Union(a, b)) == ucode_str
    # 确保 pretty 函数能正确转换并集为 ASCII 格式字符串
    assert pretty(Union(a, b)) == ascii_str


def test_pretty_Intersection_issue_10414():
    x, y, z, w = symbols('x, y, z, w')
    a, b = Interval(x, y), Interval(z, w)
    # Unicode 格式的字符串，表示两个区间的交集
    ucode_str = '[x, y] ∩ [z, w]'
    # ASCII 格式的字符串，表示两个区间的交集
    ascii_str = '[x, y] n [z, w]'
    # 断言：验证 Intersection(a, b) 的美化输出是否等于 Unicode 字符串 ucode_str
    assert upretty(Intersection(a, b)) == ucode_str
    
    # 断言：验证 Intersection(a, b) 的美化输出是否等于 ASCII 字符串 ascii_str
    assert pretty(Intersection(a, b)) == ascii_str
def test_ProductSet_exponent():
    # 定义 Unicode 字符串，用于对比测试结果
    ucode_str = '      1\n[0, 1] '
    # 断言 upretty 函数对区间 [0, 1] 进行指数运算后的结果与预期的 Unicode 字符串相等
    assert upretty(Interval(0, 1)**1) == ucode_str
    ucode_str = '      2\n[0, 1] '
    # 断言 upretty 函数对区间 [0, 1] 进行平方运算后的结果与预期的 Unicode 字符串相等
    assert upretty(Interval(0, 1)**2) == ucode_str


def test_ProductSet_parenthesis():
    # 定义 Unicode 字符串，用于对比测试结果
    ucode_str = '([4, 7] × {1, 2}) ∪ ([2, 3] × [4, 7])'

    # 定义区间 [2, 3] 和 [4, 7]
    a, b = Interval(2, 3), Interval(4, 7)
    # 断言 upretty 函数对集合并运算结果与预期的 Unicode 字符串相等
    assert upretty(Union(a*b, b*FiniteSet(1, 2))) == ucode_str


def test_ProductSet_prod_char_issue_10413():
    # 定义 ASCII 字符串和对应的 Unicode 字符串，用于对比测试结果
    ascii_str = '[2, 3] x [4, 7]'
    ucode_str = '[2, 3] × [4, 7]'

    # 定义区间 [2, 3] 和 [4, 7]
    a, b = Interval(2, 3), Interval(4, 7)
    # 断言 pretty 函数对区间乘积的 ASCII 和 upretty 函数对应的 Unicode 结果分别与预期字符串相等
    assert pretty(a*b) == ascii_str
    assert upretty(a*b) == ucode_str


def test_pretty_sequences():
    # 定义不同的序列对象
    s1 = SeqFormula(a**2, (0, oo))
    s2 = SeqPer((1, 2))

    # 定义 ASCII 和 Unicode 字符串，用于对比测试结果
    ascii_str = '[0, 1, 4, 9, ...]'
    ucode_str = '[0, 1, 4, 9, …]'

    # 断言 pretty 函数和 upretty 函数对序列 s1 的结果分别与预期的 ASCII 和 Unicode 字符串相等
    assert pretty(s1) == ascii_str
    assert upretty(s1) == ucode_str

    ascii_str = '[1, 2, 1, 2, ...]'
    ucode_str = '[1, 2, 1, 2, …]'
    # 断言 pretty 函数和 upretty 函数对序列 s2 的结果分别与预期的 ASCII 和 Unicode 字符串相等
    assert pretty(s2) == ascii_str
    assert upretty(s2) == ucode_str

    # 继续定义其他序列对象和相应的测试用例，按照类似的方式进行断言和注释
    # 省略其他测试用例的注释，依次类推
    # 使用 assert 断言来验证函数 pretty 对 s8 的处理结果是否与 ascii_str 相等
    assert pretty(s8) == ascii_str
    # 使用 assert 断言来验证函数 upretty 对 s8 的处理结果是否与 ucode_str 相等
    assert upretty(s8) == ucode_str
def test_pretty_FourierSeries():
    # 使用 FourierSeries 函数计算 x 的傅里叶级数 f
    f = fourier_series(x, (x, -pi, pi))

    # ASCII 格式的字符串，用于比较 pretty 函数输出
    ascii_str = \
"""\
                      2*sin(3*x)      \n\
2*sin(x) - sin(2*x) + ---------- + ...\n\
                          3           \
"""

    # Unicode 格式的字符串，用于比较 upretty 函数输出
    ucode_str = \
"""\
                      2⋅sin(3⋅x)    \n\
2⋅sin(x) - sin(2⋅x) + ────────── + …\n\
                          3         \
"""

    # 断言 pretty 函数对 f 的输出与 ascii_str 相符
    assert pretty(f) == ascii_str
    # 断言 upretty 函数对 f 的输出与 ucode_str 相符
    assert upretty(f) == ucode_str


def test_pretty_FormalPowerSeries():
    # 使用 FormalPowerSeries 函数计算 log(1 + x) 的形式幂级数 f
    f = fps(log(1 + x))

    # ASCII 格式的字符串，用于比较 pretty 函数输出
    ascii_str = \
"""\
 oo              \n\
____             \n\
\\   `            \n\
 \\         -k  k \n\
  \\   -(-1)  *x  \n\
  /   -----------\n\
 /         k     \n\
/___,            \n\
k = 1            \
"""

    # Unicode 格式的字符串，用于比较 upretty 函数输出
    ucode_str = \
"""\
 ∞               \n\
____             \n\
╲                \n\
 ╲         -k  k \n\
  ╲   -(-1)  ⋅x  \n\
  ╱   ───────────\n\
 ╱         k     \n\
╱                \n\
‾‾‾‾             \n\
k = 1            \
"""

    # 断言 pretty 函数对 f 的输出与 ascii_str 相符
    assert pretty(f) == ascii_str
    # 断言 upretty 函数对 f 的输出与 ucode_str 相符
    assert upretty(f) == ucode_str


def test_pretty_limits():
    # 创建极限表达式 Limit(x, x, oo)
    expr = Limit(x, x, oo)
    # ASCII 格式的字符串，用于比较 pretty 函数输出
    ascii_str = \
"""\
 lim x\n\
x->oo \
"""
    # Unicode 格式的字符串，用于比较 upretty 函数输出
    ucode_str = \
"""\
lim x\n\
x─→∞ \
"""
    # 断言 pretty 函数对 expr 的输出与 ascii_str 相符
    assert pretty(expr) == ascii_str
    # 断言 upretty 函数对 expr 的输出与 ucode_str 相符
    assert upretty(expr) == ucode_str

    # 创建极限表达式 Limit(x**2, x, 0)
    expr = Limit(x**2, x, 0)
    # ASCII 格式的字符串，用于比较 pretty 函数输出
    ascii_str = \
"""\
      2\n\
 lim x \n\
x->0+  \
"""
    # Unicode 格式的字符串，用于比较 upretty 函数输出
    ucode_str = \
"""\
      2\n\
 lim x \n\
x─→0⁺  \
"""
    # 断言 pretty 函数对 expr 的输出与 ascii_str 相符
    assert pretty(expr) == ascii_str
    # 断言 upretty 函数对 expr 的输出与 ucode_str 相符
    assert upretty(expr) == ucode_str

    # 创建极限表达式 Limit(1/x, x, 0)
    expr = Limit(1/x, x, 0)
    # ASCII 格式的字符串，用于比较 pretty 函数输出
    ascii_str = \
"""\
     1\n\
 lim -\n\
x->0+x\
"""
    # Unicode 格式的字符串，用于比较 upretty 函数输出
    ucode_str = \
"""\
     1\n\
 lim ─\n\
x─→0⁺x\
"""
    # 断言 pretty 函数对 expr 的输出与 ascii_str 相符
    assert pretty(expr) == ascii_str
    # 断言 upretty 函数对 expr 的输出与 ucode_str 相符
    assert upretty(expr) == ucode_str

    # 创建极限表达式 Limit(sin(x)/x, x, 0)
    expr = Limit(sin(x)/x, x, 0)
    # ASCII 格式的字符串，用于比较 pretty 函数输出
    ascii_str = \
"""\
     /sin(x)\\\n\
 lim |------|\n\
x->0+\\  x   /\
"""
    # Unicode 格式的字符串，用于比较 upretty 函数输出
    ucode_str = \
"""\
     ⎛sin(x)⎞\n\
 lim ⎜──────⎟\n\
x─→0⁺⎝  x   ⎠\
"""
    # 断言 pretty 函数对 expr 的输出与 ascii_str 相符
    assert pretty(expr) == ascii_str
    # 断言 upretty 函数对 expr 的输出与 ucode_str 相符
    assert upretty(expr) == ucode_str

    # 创建极限表达式 Limit(sin(x)/x, x, 0, "-")
    expr = Limit(sin(x)/x, x, 0, "-")
    # ASCII 格式的字符串，用于比较 pretty 函数输出
    ascii_str = \
"""\
     /sin(x)\\\n\
 lim |------|\n\
x->0-\\  x   /\
"""
    # Unicode 格式的字符串，用于比较 upretty 函数输出
    ucode_str = \
"""\
     ⎛sin(x)⎞\n\
 lim ⎜──────⎟\n\
x─→0⁻⎝  x   ⎠\
"""
    # 断言 pretty 函数对 expr 的输出与 ascii_str 相符
    assert pretty(expr) == ascii_str
    # 断言 upretty 函数对 expr 的输出与 ucode_str 相符
    assert upretty(expr) == ucode_str

    # 创建极限表达式 Limit(x + sin(x), x, 0)
    expr = Limit(x + sin(x), x, 0)
    # ASCII 格式的字符串，用于比较 pretty 函数输出
    ascii_str = \
"""\
 lim (x + sin(x))\n\
x->0+            \
"""
    # Unicode 格式的字符串，用于比较 upretty 函数输出
    ucode_str = \
"""\
 lim (x + sin(x))\n\
x─→0⁺            \
"""
    # 断言 pretty 函数对 expr 的输出与 ascii_str 相符
    assert pretty(expr) == ascii_str
    # 断言 upretty 函数对 expr 的输出与 ucode_str 相符
    assert upretty(expr) == ucode_str

    # 创建极限表达式 Limit(x, x, 0)**2
    expr = Limit(x, x, 0)**2
    # ASCII 格式的字符串，用于比较 pretty 函数输出
    ascii_str = \
"""\
        2\n\
/ lim x\\ \n\
\\x->0+ / \
"""
    # Unicode 格式的字符串，用于比较 upretty 函数输出
    ucode_str = \
"""\
        2\n\
⎛ lim x⎞ \n\
⎝x─→0⁺ ⎠ \
"""
    # 断言 pretty 函数对 expr 的输出与 ascii_str 相符
    assert pretty(expr) == ascii_str
    # 断言 upretty 函数对 expr 的输出与 ucode_str 相符
    assert upretty(expr) == ucode_str

    # 创建极限表达式 Limit(x*Limit(y/2,y,0), x, 0)
    expr = Limit(x*Limit(y/2,y,0), x, 0)
    # ASCII 格式的字符串，用于比较 pretty 函数输出
    ascii_str = \
"""\
     /       /y\\\\\n\
 lim |x* lim |-||\n\
x->0+\\  y->0+\\2//\
"""
    # Unicode 格式的字符串，用于比较 upretty 函数输出
    ucode_str = \
"""\
     ⎛       ⎛y
    # 计算表达式 2 * Limit(x * Limit(y/2, y, 0), x, 0)
    expr = 2 * Limit(x * Limit(y/2, y, 0), x, 0)
    # 定义一个跨行的字符串变量
    ascii_str = \
"""
定义了一个多行的字符串，包含ASCII艺术风格的数学表达式。
这个字符串展示了一个数学极限的表示方式。
"""
ascii_str = \
"""\
       /       /y\\\\\n\
2* lim |x* lim |-||\n\
  x->0+\\  y->0+\\2//\
"""

"""
定义了一个多行的字符串，包含Unicode艺术风格的数学表达式。
这个字符串展示了与上面相同的数学极限的表示方式，但使用了Unicode字符。
"""
ucode_str = \
"""\
       ⎛       ⎛y⎞⎞\n\
2⋅ lim ⎜x⋅ lim ⎜─⎟⎟\n\
  x─→0⁺⎝  y─→0⁺⎝2⎠⎠\
"""

"""
使用assert语句检查pretty函数和upretty函数对给定表达式的输出是否与ascii_str和ucode_str匹配。
"""
assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str

"""
定义了一个Limit对象的表达式，表示sin(x)在x趋向于0时的极限。
"""
expr = Limit(sin(x), x, 0, dir='+-')

"""
定义了一个ASCII艺术风格的字符串，展示了上述Limit对象的表示方式。
"""
ascii_str = \
"""\
lim sin(x)\n\
x->0      \
"""

"""
定义了一个Unicode艺术风格的字符串，展示了上述Limit对象的表示方式。
"""
ucode_str = \
"""\
lim sin(x)\n\
x─→0      \
"""

"""
使用assert语句检查pretty函数和upretty函数对给定表达式的输出是否与ascii_str和ucode_str匹配。
"""
assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str



def test_pretty_ComplexRootOf():
    """
    测试函数，用于测试对ComplexRootOf对象的美化输出。
    """
    expr = rootof(x**5 + 11*x - 2, 0)

    """
    定义了一个ASCII艺术风格的字符串，展示了ComplexRootOf对象的表示方式。
    """
    ascii_str = \
"""\
       / 5              \\\n\
CRootOf\\x  + 11*x - 2, 0/\
"""

    """
    定义了一个Unicode艺术风格的字符串，展示了ComplexRootOf对象的表示方式。
    """
    ucode_str = \
"""\
       ⎛ 5              ⎞\n\
CRootOf⎝x  + 11⋅x - 2, 0⎠\
"""

    """
    使用assert语句检查pretty函数和upretty函数对给定表达式的输出是否与ascii_str和ucode_str匹配。
    """
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_pretty_RootSum():
    """
    测试函数，用于测试对RootSum对象的美化输出。
    """
    expr = RootSum(x**5 + 11*x - 2, auto=False)

    """
    定义了一个ASCII艺术风格的字符串，展示了RootSum对象的表示方式。
    """
    ascii_str = \
"""\
       / 5           \\\n\
RootSum\\x  + 11*x - 2/\
"""

    """
    定义了一个Unicode艺术风格的字符串，展示了RootSum对象的表示方式。
    """
    ucode_str = \
"""\
       ⎛ 5           ⎞\n\
RootSum⎝x  + 11⋅x - 2⎠\
"""

    """
    使用assert语句检查pretty函数和upretty函数对给定表达式的输出是否与ascii_str和ucode_str匹配。
    """
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = RootSum(x**5 + 11*x - 2, Lambda(z, exp(z)))

    """
    定义了一个ASCII艺术风格的字符串，展示了带Lambda函数的RootSum对象的表示方式。
    """
    ascii_str = \
"""\
       / 5                   z\\\n\
RootSum\\x  + 11*x - 2, z -> e /\
"""

    """
    定义了一个Unicode艺术风格的字符串，展示了带Lambda函数的RootSum对象的表示方式。
    """
    ucode_str = \
"""\
       ⎛ 5                  z⎞\n\
RootSum⎝x  + 11⋅x - 2, z ↦ ℯ ⎠\
"""

    """
    使用assert语句检查pretty函数和upretty函数对给定表达式的输出是否与ascii_str和ucode_str匹配。
    """
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str



def test_GroebnerBasis():
    """
    测试函数，用于测试对GroebnerBasis对象的美化输出。
    """
    expr = groebner([], x, y)

    """
    定义了一个ASCII艺术风格的字符串，展示了GroebnerBasis对象的表示方式。
    """
    ascii_str = \
"""\
GroebnerBasis([], x, y, domain=ZZ, order=lex)\
"""

    """
    定义了一个Unicode艺术风格的字符串，展示了GroebnerBasis对象的表示方式。
    """
    ucode_str = \
"""\
GroebnerBasis([], x, y, domain=ℤ, order=lex)\
"""

    """
    使用assert语句检查pretty函数和upretty函数对给定表达式的输出是否与ascii_str和ucode_str匹配。
    """
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    F = [x**2 - 3*y - x + 1, y**2 - 2*x + y - 1]
    expr = groebner(F, x, y, order='grlex')

    """
    定义了一个ASCII艺术风格的字符串，展示了带多项式列表的GroebnerBasis对象的表示方式。
    """
    ascii_str = \
"""\
             /[ 2                 2              ]                              \\\n\
GroebnerBasis\\[x  - x - 3*y + 1, y  - 2*x + y - 1], x, y, domain=ZZ, order=grlex/\
"""

    """
    定义了一个Unicode艺术风格的字符串，展示了带多项式列表的GroebnerBasis对象的表示方式。
    """
    ucode_str = \
"""\
             ⎛⎡ 2                 2              ⎤                             ⎞\n\
GroebnerBasis⎝⎣x  - x - 3⋅y + 1, y  - 2⋅x + y - 1⎦, x, y, domain=ℤ, order=grlex⎠\
"""

    """
    使用assert语句检查pretty函数和upretty函数对给定表达式的输出是否与ascii_str和ucode_str匹配。
    """
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = expr.fglm('lex')

    """
    定义了一个ASCII艺术风格的字符串，展示了调用fglm方法后的GroebnerBasis对象的表示方式。
    """
    ascii_str = \
"""\
             /[       2           4      3      2           ]                            \\\n\
GroebnerBasis\\[2*x - y  - y + 1, y  + 2*y  - 3*y  - 16*y + 7], x, y, domain=ZZ, order=lex/\
"""

    """
    定义了一个Unicode艺术风格的字符串，展示了调用fglm方法后的GroebnerBasis对象的表示方式。
    """
    ucode_str = \
"""\
             ⎛⎡       2           4      3      2           ⎤                           ⎞\n\
GroebnerBasis⎝⎣2⋅x - y  - y + 1, y  + 2⋅y  - 3⋅y  - 16⋅y + 7⎦, x, y, domain
    # 断言表达式的美观输出是否等于 "Not(x)"
    assert pretty(expr) == "Not(x)"
    # 断言表达式的Unicode美观输出是否等于 "¬x"
    assert upretty(expr) == "¬x"

    # 创建一个 AND 逻辑运算表达式 (x ∧ y)
    expr = And(x, y)

    # 断言表达式的美观输出是否等于 "And(x, y)"
    assert pretty(expr) == "And(x, y)"
    # 断言表达式的Unicode美观输出是否等于 "x ∧ y"
    assert upretty(expr) == "x ∧ y"

    # 创建一个 OR 逻辑运算表达式 (Or(a, b, c, d, e, f))
    expr = Or(x, y)

    # 断言表达式的美观输出是否等于 "Or(x, y)"
    assert pretty(expr) == "Or(x, y)"
    # 断言表达式的Unicode美观输出是否等于 "x ∨ y"
    assert upretty(expr) == "x ∨ y"

    # 使用 symbols 函数创建符号列表 syms = [a, b, c, d, e, f]
    syms = symbols('a:f')
    # 创建一个 AND 逻辑运算表达式 (And(a, b, c, d, e, f))
    expr = And(*syms)

    # 断言表达式的美观输出是否等于 "And(a, b, c, d, e, f)"
    assert pretty(expr) == "And(a, b, c, d, e, f)"
    # 断言表达式的Unicode美观输出是否等于 "a ∧ b ∧ c ∧ d ∧ e ∧ f"
    assert upretty(expr) == "a ∧ b ∧ c ∧ d ∧ e ∧ f"

    # 创建一个 OR 逻辑运算表达式 (Or(a, b, c, d, e, f))
    expr = Or(*syms)

    # 断言表达式的美观输出是否等于 "Or(a, b, c, d, e, f)"
    assert pretty(expr) == "Or(a, b, c, d, e, f)"
    # 断言表达式的Unicode美观输出是否等于 "a ∨ b ∨ c ∨ d ∨ e ∨ f"
    assert upretty(expr) == "a ∨ b ∨ c ∨ d ∨ e ∨ f"

    # 创建一个 XOR 逻辑运算表达式 (Xor(x, y))
    expr = Xor(x, y, evaluate=False)

    # 断言表达式的美观输出是否等于 "Xor(x, y)"
    assert pretty(expr) == "Xor(x, y)"
    # 断言表达式的Unicode美观输出是否等于 "x ⊻ y"
    assert upretty(expr) == "x ⊻ y"

    # 创建一个 NAND 逻辑运算表达式 (Nand(x, y))
    expr = Nand(x, y, evaluate=False)

    # 断言表达式的美观输出是否等于 "Nand(x, y)"
    assert pretty(expr) == "Nand(x, y)"
    # 断言表达式的Unicode美观输出是否等于 "x ⊼ y"
    assert upretty(expr) == "x ⊼ y"

    # 创建一个 NOR 逻辑运算表达式 (Nor(x, y))
    expr = Nor(x, y, evaluate=False)

    # 断言表达式的美观输出是否等于 "Nor(x, y)"
    assert pretty(expr) == "Nor(x, y)"
    # 断言表达式的Unicode美观输出是否等于 "x ⊽ y"
    assert upretty(expr) == "x ⊽ y"

    # 创建一个 IMPLIES 逻辑运算表达式 (Implies(x, y))
    expr = Implies(x, y, evaluate=False)

    # 断言表达式的美观输出是否等于 "Implies(x, y)"
    assert pretty(expr) == "Implies(x, y)"
    # 断言表达式的Unicode美观输出是否等于 "x → y"
    assert upretty(expr) == "x → y"

    # 创建一个 IMPLIES 逻辑运算表达式 (Implies(y, x))
    expr = Implies(y, x, evaluate=False)

    # 断言表达式的美观输出是否等于 "Implies(y, x)"
    assert pretty(expr) == "Implies(y, x)"
    # 断言表达式的Unicode美观输出是否等于 "y → x"
    assert upretty(expr) == "y → x"

    # 创建一个 EQUIVALENT 逻辑运算表达式 (Equivalent(x, y))
    expr = Equivalent(x, y, evaluate=False)

    # 断言表达式的美观输出是否等于 "Equivalent(x, y)"
    assert pretty(expr) == "Equivalent(x, y)"
    # 断言表达式的Unicode美观输出是否等于 "x ⇔ y"
    assert upretty(expr) == "x ⇔ y"

    # 创建一个 EQUIVALENT 逻辑运算表达式 (Equivalent(y, x))
    expr = Equivalent(y, x, evaluate=False)

    # 断言表达式的美观输出是否等于 "Equivalent(x, y)"
    assert pretty(expr) == "Equivalent(x, y)"
    # 断言表达式的Unicode美观输出是否等于 "x ⇔ y"
    assert upretty(expr) == "x ⇔ y"
# 定义一个测试函数，用于测试 SymPy 打印器中的 pretty 和 upretty 函数
def test_pretty_Domain():
    # 创建一个 FF 类的实例，参数为 23，表示有限域 GF(23)
    expr = FF(23)

    # 断言使用 pretty 函数打印 expr 应该输出 "GF(23)"
    assert pretty(expr) == "GF(23)"
    # 断言使用 upretty 函数打印 expr 应该输出 "ℤ₂₃"
    assert upretty(expr) == "ℤ₂₃"

    # 将 expr 更新为 ZZ，表示整数环
    expr = ZZ

    # 断言使用 pretty 函数打印 expr 应该输出 "ZZ"
    assert pretty(expr) == "ZZ"
    # 断言使用 upretty 函数打印 expr 应该输出 "ℤ"
    assert upretty(expr) == "ℤ"

    # 将 expr 更新为 QQ，表示有理数域
    expr = QQ

    # 断言使用 pretty 函数打印 expr 应该输出 "QQ"
    assert pretty(expr) == "QQ"
    # 断言使用 upretty 函数打印 expr 应该输出 "ℚ"
    assert upretty(expr) == "ℚ"

    # 将 expr 更新为 RR，表示实数域
    expr = RR

    # 断言使用 pretty 函数打印 expr 应该输出 "RR"
    assert pretty(expr) == "RR"
    # 断言使用 upretty 函数打印 expr 应该输出 "ℝ"
    assert upretty(expr) == "ℝ"

    # 将 expr 更新为 QQ[x]，表示有理多项式环
    expr = QQ[x]

    # 断言使用 pretty 函数打印 expr 应该输出 "QQ[x]"
    assert pretty(expr) == "QQ[x]"
    # 断言使用 upretty 函数打印 expr 应该输出 "ℚ[x]"
    assert upretty(expr) == "ℚ[x]"

    # 将 expr 更新为 QQ[x, y]，表示有理多项式环
    expr = QQ[x, y]

    # 断言使用 pretty 函数打印 expr 应该输出 "QQ[x, y]"
    assert pretty(expr) == "QQ[x, y]"
    # 断言使用 upretty 函数打印 expr 应该输出 "ℚ[x, y]"
    assert upretty(expr) == "ℚ[x, y]"

    # 将 expr 更新为 ZZ.frac_field(x)，表示 x 上的有理函数域
    expr = ZZ.frac_field(x)

    # 断言使用 pretty 函数打印 expr 应该输出 "ZZ(x)"
    assert pretty(expr) == "ZZ(x)"
    # 断言使用 upretty 函数打印 expr 应该输出 "ℤ(x)"
    assert upretty(expr) == "ℤ(x)"

    # 将 expr 更新为 ZZ.frac_field(x, y)，表示 x, y 上的有理函数域
    expr = ZZ.frac_field(x, y)

    # 断言使用 pretty 函数打印 expr 应该输出 "ZZ(x, y)"
    assert pretty(expr) == "ZZ(x, y)"
    # 断言使用 upretty 函数打印 expr 应该输出 "ℤ(x, y)"
    assert upretty(expr) == "ℤ(x, y)"

    # 将 expr 更新为 QQ.poly_ring(x, y, order=grlex)，表示 x, y 上的有理多项式环，按 grlex 排序
    expr = QQ.poly_ring(x, y, order=grlex)

    # 断言使用 pretty 函数打印 expr 应该输出 "QQ[x, y, order=grlex]"
    assert pretty(expr) == "QQ[x, y, order=grlex]"
    # 断言使用 upretty 函数打印 expr 应该输出 "ℚ[x, y, order=grlex]"
    assert upretty(expr) == "ℚ[x, y, order=grlex)"

    # 将 expr 更新为 QQ.poly_ring(x, y, order=ilex)，表示 x, y 上的有理多项式环，按 ilex 排序
    expr = QQ.poly_ring(x, y, order=ilex)

    # 断言使用 pretty 函数打印 expr 应该输出 "QQ[x, y, order=ilex]"
    assert pretty(expr) == "QQ[x, y, order=ilex]"
    # 断言使用 upretty 函数打印 expr 应该输出 "ℚ[x, y, order=ilex]"
    assert upretty(expr) == "ℚ[x, y, order=ilex]"


# 测试 xpretty 函数的不同参数设置
def test_pretty_prec():
    # 断言使用 xpretty 函数打印 S("0.3")，完整精度模式，不换行应该输出 "0.300000000000000"
    assert xpretty(S("0.3"), full_prec=True, wrap_line=False) == "0.300000000000000"
    # 断言使用 xpretty 函数打印 S("0.3")，自动精度模式，不换行应该输出 "0.300000000000000"
    assert xpretty(S("0.3"), full_prec="auto", wrap_line=False) == "0.300000000000000"
    # 断言使用 xpretty 函数打印 S("0.3")，非精确模式，不换行应该输出 "0.3"
    assert xpretty(S("0.3"), full_prec=False, wrap_line=False) == "0.3"
    # 断言使用 xpretty 函数打印 S("0.3")*x，完整精度模式，不使用 Unicode，不换行应该在两种格式中的一种
    assert xpretty(S("0.3")*x, full_prec=True, use_unicode=False, wrap_line=False) in [
        "0.300000000000000*x",
        "x*0.300000000000000"
    ]
    # 断言使用 xpretty 函数打印 S("0.3")*x，自动精度模式，不使用 Unicode，不换行应该在两种格式中的一种
    assert xpretty(S("0.3")*x, full_prec="auto", use_unicode=False, wrap_line=False) in [
        "0.3*x",
        "x*0.3"
    ]
    # 断言使用 xpretty 函数打印 S("0.3")*x，非精确模式，不使用 Unicode，不换行应该在两种格式中的一种
    assert xpretty(S("0.3")*x, full_prec=False, use_unicode=False, wrap_line=False) in [
        "0.3*x",
        "x*0.3"
    ]


# 测试 pprint 函数的输出
def test_pprint():
    import sys
    from io import StringIO
    fd = StringIO()
    sso = sys.stdout
    sys.stdout = fd
    try:
        # 调用 pprint 打印 pi，不使用 Unicode，不换行
        pprint(pi, use_unicode=False, wrap_line=False)
    finally:
        sys.stdout = sso
    # 断言 fd.getvalue() 应该输出 'pi\n'
    assert fd.getvalue() == 'pi\n'


# 测试 pretty 函数对类的处理
def test_pretty_class():
    """Test that the printer dispatcher correctly handles classes."""
    class C:
        pass   # C has no .__class__ and this was causing problems

    class D:
        pass

    # 断言 pretty(C) 应该等于 str(C)
    assert pretty( C ) == str( C )
    # 断言 pretty(D) 应该等于 str(D)
    assert pretty( D ) == str( D )


# 测试 xpretty 函数的换行设置
def test_pretty_no_wrap_line():
    huge_expr = 0
    for i in range(20):
        huge_expr += i*sin(i + x)
    # 断言 xpretty(huge_expr) 应该包含换行符 '\n'
    assert xpretty(huge_expr            ).find('\n') != -1
    # 断言 xpretty(huge_expr, wrap_line=False) 不应该包含换行符 '\n'
    assert xpretty(huge_expr, wrap_line=False).find('\n') == -1


# 测试 pretty 函数对设置的处理
def test_settings():
    # 断言调用 pretty(S(4), method="garbage") 应该抛出 TypeError 异常
    raises(TypeError, lambda: pretty(S(4), method="garbage"))


# 测试 pretty 函数对求和表达式的打印
def test_pretty_sum():
    from sympy.abc import x, a, b, k
    # 计算无穷级数 \(\sum_{k=\infty}^{n} k^k\) 的表达式
    expr = Sum(k**k, (k, oo, n))
    # 延续 ASCII 字符串，继续下一行的赋值
    ascii_str = \
    """\
      n      \n\
     ___     \n\
     \\  `    \n\
      \\     k\n\
      /    k \n\
     /__,    \n\
    k = oo   \
    """

这段代码定义了一个多行字符串 `ascii_str`，其中包含了一个ASCII艺术风格的图形，描述了一个字符图案和文本内容。


    ucode_str = \
    """\
      n     \n\
     ___    \n\
     ╲      \n\
      ╲    k\n\
      ╱   k \n\
     ╱      \n\
     ‾‾‾    \n\
    k = ∞   \
    """

这段代码定义了另一个多行字符串 `ucode_str`，其中包含了一个Unicode艺术风格的图形，描述了一个字符图案和文本内容。


    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

这里进行了两个断言，验证了函数 `pretty(expr)` 和 `upretty(expr)` 的输出分别等于 `ascii_str` 和 `ucode_str`。


    expr = Sum(k**(Integral(x**n, (x, -oo, oo))), (k, 0, n**n))
    ascii_str = \
    """\
       n              \n\
      n               \n\
    ______            \n\
    \\     `           \n\
     \\        oo      \n\
      \\        /      \n\
       \\      |       \n\
        \\     |   n   \n\
         )    |  x  dx\n\
        /     |       \n\
       /     /        \n\
      /      -oo      \n\
     /      k         \n\
    /_____,           \n\
    k = 0            \
    """

这段代码定义了一个表达式 `expr`，它使用了 `Sum` 函数计算一个和，包括一个幂次和积分。`ascii_str` 定义了对应的ASCII艺术风格的字符串表示。


    ucode_str = \
    """\
       n            \n\
      n             \n\
    ______          \n\
    ╲               \n\
     ╲              \n\
      ╲     ∞       \n\
       ╲    ⌠       \n\
        ╲   ⎮   n   \n\
        ╱   ⎮  x  dx\n\
       ╱    ⌡       \n\
      ╱     -∞      \n\
     ╱     k        \n\
    ╱               \n\
    ‾‾‾‾‾‾          \n\
    k = 0           \
    """

这段代码定义了另一个字符串 `ucode_str`，它包含了一个Unicode艺术风格的字符串表示，描述了同样的表达式和符号。


    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

这里进行了两个断言，验证了函数 `pretty(expr)` 和 `upretty(expr)` 的输出分别等于 `ascii_str` 和 `ucode_str`。


    expr = Sum(k**(
        Integral(x**n, (x, -oo, oo))), (k, 0, Integral(x**x, (x, -oo, oo))))
    ascii_str = \
    """\
     oo                 \n\
      /                 \n\
     |                  \n\
     |   x              \n\
     |  x  dx           \n\
     |                  \n\
    /                   \n\
    -oo                 \n\
     ______             \n\
     \\     `            \n\
      \\         oo      \n\
       \\         /      \n\
        \\       |       \n\
         \\      |   n   \n\
          )     |  x  dx\n\
         /      |       \n\
        /      /        \n\
       /       -oo      \n\
      /       k         \n\
     /_____,            \n\
     k = 0             \
    """

这段代码定义了另一个表达式 `expr`，它使用了 `Sum` 函数计算一个和，包括一个幂次和两个积分。`ascii_str` 定义了对应的ASCII艺术风格的字符串表示。


    ucode_str = \
    """\
    ∞                 \n\
    ⌠                 \n\
    ⎮   x             \n\
    ⎮  x  dx          \n\
    ⌡                 \n\
    -∞                \n\
     ______           \n\
     ╲                \n\
      ╲               \n\
       ╲      ∞       \n\
        ╲     ⌠       \n\
         ╲    ⎮   n   \n\
         ╱    ⎮  x  dx\n\
        ╱     ⌡       \n\
       ╱      -∞      \n\
      ╱      k        \n\
     ╱                \n\
     ‾‾‾‾‾‾           \n\
     k = 0            \
    """

这段代码定义了另一个字符串 `ucode_str`，它包含了一个Unicode艺术风格的字符串表示，描述了同样的表达式和符号。


    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

这里进行了两个断言，验证了函数 `pretty(expr)` 和 `upretty(expr)` 的输出分别等于 `ascii_str` 和 `ucode_str`。
"""
这部分代码定义了两个多行字符串，分别是ascii_str和ucode_str，用来存储ASCII码和Unicode码的数学表达式。

ucode_str存储了一个用Unicode字符表示的数学表达式，包含了积分、求和符号等数学符号。

ascii_str存储了一个用ASCII字符表示的数学表达式，同样包含了积分、求和符号等数学符号。

这里使用了多行字符串的格式，在Python中，用三重引号包围的字符串可以跨越多行，并保留其原始格式。
"""
assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str

expr = Sum(k**(
    Integral(x**n, (x, -oo, oo))), (k, 0, x + n + x**2 + n**2 + (x/n) + (1/x)))
ascii_str = \
"""\
 2        2       1   x           \n\
n  + n + x  + x + - + -           \n\
                  x   n           \n\
        ______                    \n\
        \\     `                   \n\
         \\                oo      \n\
          \\                /      \n\
           \\              |       \n\
            \\             |   n   \n\
             )            |  x  dx\n\
            /             |       \n\
           /             /        \n\
          /              -oo      \n\
         /              k         \n\
        /_____,                   \n\
         k = 0                    \
"""
ucode_str = \
"""\
 2        2       1   x          \n\
n  + n + x  + x + ─ + ─          \n\
                  x   n          \n\
        ______                   \n\
        ╲                         \n\
         ╲                        \n\
          ╲                ∞      \n\
           ╲               ⌠      \n\
            ╲              ⎮   n  \n\
            ╱              ⎮  x dx\n\
           ╱               ⌡      \n\
          ╱                -∞     \n\
         ╱                k       \n\
        ╱                         \n\
        ‾‾‾‾‾‾                    \n\
     2        2       1   x       \n\
k = n  + n + x  + x + ─ + ─       \n\
                  x   n           \
"""

"""
这部分代码进行了表达式的断言比较，验证pretty函数和upretty函数对expr表达式的输出是否符合ascii_str和ucode_str的预期结果。

assert语句用于断言表达式的真实性，如果表达式为假，则抛出异常。

这里的expr表达式涉及到求和、积分等数学运算，ascii_str和ucode_str包含了预期的ASCII和Unicode表达式格式。
"""
# 定义表达式，求和 Sum(x + n + x**n, (x, 0, oo))，表达式中 oo 表示无穷大
expr = Sum(x + n + x**n, (x, 0, oo))
# 生成 ASCII 格式的字符串表示，显示无穷大符号和求和符号
ascii_str = \
"""\
n  + n + x  + x + ─ + ─          \n\
                  x   n          \n\
        ______                   \n\
        ╲                        \n\
         ╲                       \n\
          ╲              ∞       \n\
           ╲             ⌠       \n\
            ╲            ⎮   n   \n\
            ╱            ⎮  x  dx\n\
           ╱             ⌡       \n\
          ╱              -∞      \n\
         ╱              k        \n\
        ╱                        \n\
        ‾‾‾‾‾‾                   \n\
         k = 0                   \
"""
# 生成 Unicode 格式的字符串表示，显示无穷大符号和求和符号
ucode_str = \
"""\
  ∞    \n\
 __    \n\
 ╲     \n\
  ╲    \n\
  ╱   x\n\
 ╱     \n\
 ‾‾‾   \n\
x = 0  \
"""

# 断言 ASCII 格式和 Unicode 格式的字符串表示与预期相符
assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str

# 定义表达式，求和 Sum(x, (x, 0, oo))，表达式中 oo 表示无穷大
expr = Sum(x, (x, 0, oo))
# 生成 ASCII 格式和 Unicode 格式的字符串表示，显示无穷大符号和求和符号
ascii_str = \
"""\
 oo    \n\
 __    \n\
 \\ `   \n\
  )   x\n\
 /_,   \n\
x = 0  \
"""
ucode_str = \
"""\
  ∞    \n\
 ___   \n\
 ╲     \n\
  ╲    \n\
  ╱   x\n\
 ╱     \n\
 ‾‾‾   \n\
x = 0  \
"""

# 断言 ASCII 格式和 Unicode 格式的字符串表示与预期相符
assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str

# 定义表达式，求和 Sum(x**2, (x, 0, oo))，表达式中 oo 表示无穷大
expr = Sum(x**2, (x, 0, oo))
# 生成 ASCII 格式和 Unicode 格式的字符串表示，显示无穷大符号和求和符号
ascii_str = \
"""\
 oo     \n\
___     \n\
\\  `    \n\
 \\     2\n\
 /    x \n\
/__,    \n\
x = 0   \
"""
ucode_str = \
"""\
  ∞     \n\
 ___    \n\
 ╲      \n\
  ╲    2\n\
  ╱   x \n\
 ╱      \n\
 ‾‾‾    \n\
x = 0   \
"""

# 断言 ASCII 格式和 Unicode 格式的字符串表示与预期相符
assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str

# 定义表达式，求和 Sum(x/2, (x, 0, oo))，表达式中 oo 表示无穷大
expr = Sum(x/2, (x, 0, oo))
# 生成 ASCII 格式和 Unicode 格式的字符串表示，显示无穷大符号和求和符号
ascii_str = \
"""\
 oo    \n\
___    \n\
\\  `   \n\
 \\    x\n\
  )   -\n\
 /    2\n\
/__,   \n\
x = 0  \
"""
ucode_str = \
"""\
 ∞     \n\
____   \n\
╲      \n\
 ╲     \n\
  ╲   x\n\
  ╱   ─\n\
 ╱    2\n\
╱      \n\
‾‾‾‾   \n\
x = 0  \
"""

# 断言 ASCII 格式和 Unicode 格式的字符串表示与预期相符
assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str

# 定义表达式，求和 Sum(x**3/2, (x, 0, oo))，表达式中 oo 表示无穷大
expr = Sum(x**3/2, (x, 0, oo))
# 生成 ASCII 格式和 Unicode 格式的字符串表示，显示无穷大符号和求和符号
ascii_str = \
"""\
 oo     \n\
____    \n\
\\   `   \n\
 \\     3\n\
  \\   x \n\
  /   --\n\
 /    2 \n\
/___,   \n\
x = 0   \
"""
ucode_str = \
"""\
 ∞      \n\
____    \n\
╲       \n\
 ╲     3\n\
  ╲   x \n\
  ╱   ──\n\
 ╱    2 \n\
╱       \n\
‾‾‾‾    \n\
x = 0   \
"""

# 断言 ASCII 格式和 Unicode 格式的字符串表示与预期相符
assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str

# 定义表达式，求和 Sum((x**3*y**(x/2))**n, (x, 0, oo))，表达式中 oo 表示无穷大
expr = Sum((x**3*y**(x/2))**n, (x, 0, oo))
# 生成 ASCII 格式和 Unicode 格式的字符串表示，显示无穷大符号和求和符号
ascii_str = \
"""\
 oo           \n\
____          \n\
\\   `         \n\
 \\           n\n\
  \\   /    x\\ \n\
   )  |    -| \n\
  /   | 3  2| \n\
 /    \\x *y / \n\
/___,         \n\
x = 0         \
"""
ucode_str = \
"""\
  ∞           \n\
_____         \n\
╲             \n\
 ╲            \n\
  ╲          n\n\
   ╲  ⎛    x⎞ \n\
   ╱  ⎜    ─⎟ \n\
  ╱   ⎜ 3  2⎟ \n\
 ╱    ⎝x ⋅y ⎠ \n\
╱             \n\
‾‾‾‾‾         \n\
x = 0         \
"""

# 断言 ASCII 格式和 Unicode 格式的字符串表示与预期相符
assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str

# 定义表达式，求和 Sum(1/x**2, (x, 0, oo))，表达式中 oo 表示无穷大
expr = Sum(1/x**2, (x, 0, oo))
# 生成 ASCII 格式的字符串表示，显示无穷大符号和求和符号
ascii_str = \
"""\
 oo     \n\
____    \n\
\\   `   \n\
 \\    1 \n\
  \\   --\n\
  /    2\n\
 /    x \n\
/___,   \n\
x = 0   \
"""
# 生成 Unicode 格式的字符串表示，显示无穷大符号和求和符号
ucode_str = \
"""\
 ∞      \n\
____    \n\
╲       \n\
 ╲    1 \n\
  ╲   ──\n\
  ╱    2\n\
 ╱    x \n\
╱       \n\
‾‾‾‾    \n\
x = 0   \
"""

# 断言 ASCII 格式的字符串表示与预期相符
assert pretty(expr) == ascii_str
    # 使用 assert 断言来验证表达式的可读字符串表示是否等于指定的 Unicode 代码字符串
    assert upretty(expr) == ucode_str

    # 创建一个求和表达式，其中 y 的幂次为 a/b，x 的范围是从 0 到无穷大
    expr = Sum(1/y**(a/b), (x, 0, oo))
    
    # 将多行字符串赋给 ascii_str 变量，这是一个续行的示例
    ascii_str = \
# 定义一个多行字符串，包含ASCII艺术风格的文本图形
"""\
 oo       \n\
____      \n\
\\   `     \n\
 \\     -a \n\
  \\    ---\n\
  /     b \n\
 /    y   \n\
/___,     \n\
x = 0     \
"""

# 定义一个Unicode格式的多行字符串，包含Unicode艺术风格的文本图形
ucode_str = \
"""\
 ∞        \n\
____      \n\
╲         \n\
 ╲     -a \n\
  ╲    ───\n\
  ╱     b \n\
 ╱    y   \n\
╱         \n\
‾‾‾‾      \n\
x = 0     \
"""

# 断言表达式`pretty(expr)`的输出等于ASCII格式的字符串`ascii_str`
assert pretty(expr) == ascii_str
# 断言表达式`upretty(expr)`的输出等于Unicode格式的字符串`ucode_str`
assert upretty(expr) == ucode_str

# 定义表达式`expr`，计算一个求和的数学表达式
expr = Sum(1/y**(a/b), (x, 0, oo), (y, 1, 2))
# 定义ASCII格式的多行字符串，包含与表达式`expr`相关的ASCII艺术风格的数学表达式
ascii_str = \
"""\
  2     oo     \n\
____  ____     \n\
\\   ` \\   `    \n\
 \\     \\     -a\n\
  \\     \\    --\n\
  /     /    b \n\
 /     /    y  \n\
/___, /___,    \n\
y = 1 x = 0    \
"""
# 定义Unicode格式的多行字符串，包含与表达式`expr`相关的Unicode艺术风格的数学表达式
ucode_str = \
"""\
  2     ∞      \n\
____  ____     \n\
╲     ╲        \n\
 ╲     ╲     -a\n\
  ╲     ╲    ──\n\
  ╱     ╱    b \n\
 ╱     ╱    y  \n\
╱     ╱        \n\
‾‾‾‾  ‾‾‾‾     \n\
y = 1 x = 0    \
"""

# 定义表达式`expr`，计算另一个求和的数学表达式
expr = Sum(1/(1 + 1/(
    1 + 1/k)) + 1, (k, 111, 1 + 1/n), (k, 1/(1 + m), oo)) + 1/(1 + 1/k)
# 定义ASCII格式的多行字符串，包含与表达式`expr`相关的ASCII艺术风格的数学表达式
ascii_str = \
"""\
              1                          \n\
          1 + -                          \n\
   oo         n                          \n\
 _____    _____                          \n\
 \\    `   \\    `                         \n\
  \\        \\      /        1    \\        \n\
   \\        \\     |1 + ---------|        \n\
    \\        \\    |          1  |     1  \n\
     )        )   |    1 + -----| + -----\n\
    /        /    |            1|       1\n\
   /        /     |        1 + -|   1 + -\n\
  /        /      \\            k/       k\n\
 /____,   /____,                         \n\
      1   k = 111                        \n\
k = -----                                \n\
    m + 1                                \
"""
# 定义Unicode格式的多行字符串，包含与表达式`expr`相关的Unicode艺术风格的数学表达式
ucode_str = \
"""\
              1                          \n\
          1 + ─                          \n\
   ∞          n                          \n\
 ______   ______                         \n\
 ╲        ╲                              \n\
  ╲        ╲                             \n\
   ╲        ╲     ⎛        1    ⎞        \n\
    ╲        ╲    ⎜1 + ─────────⎟        \n\
     ╲        ╲   ⎜          1  ⎟     1  \n\
     ╱        ╱   ⎜    1 + ─────⎟ + ─────\n\
    ╱        ╱    ⎜            1⎟       1\n\
   ╱        ╱     ⎜        1 + ─⎟   1 + ─\n\
  ╱        ╱      ⎝            k⎠       k\n\
 ╱        ╱                              \n\
 ‾‾‾‾‾‾   ‾‾‾‾‾‾                         \n\
      1   k = 111                        \n\
k = ─────                                \n\
    m + 1                                \
"""

# 断言表达式`pretty(expr)`的输出等于ASCII格式的字符串`ascii_str`
assert pretty(expr) == ascii_str
# 断言表达式`upretty(expr)`的输出等于Unicode格式的字符串`ucode_str`
assert upretty(expr) == ucode_str
# 测试 sympy.physics.units 中的符号单位导入
from sympy.physics.units import kg, m, s
# 断言表达式 expr 的 Unicode 格式化输出是否为 "joule"
assert upretty(expr) == "joule"
# 断言表达式 expr 的 ASCII 格式化输出是否为 "joule"
assert pretty(expr) == "joule"
# 断言 expr 转换为 kg*m**2/s**2 后的 Unicode 格式化输出是否与 unicode_str1 相符
assert upretty(expr.convert_to(kg*m**2/s**2)) == unicode_str1
# 断言 expr 转换为 kg*m**2/s**2 后的 ASCII 格式化输出是否与 ascii_str1 相符
assert pretty(expr.convert_to(kg*m**2/s**2)) == ascii_str1
# 断言 3*kg*x*m**2*y/s**2 的 Unicode 格式化输出是否与 unicode_str2 相符
assert upretty(3*kg*x*m**2*y/s**2) == unicode_str2
# 断言 3*kg*x*m**2*y/s**2 的 ASCII 格式化输出是否与 ascii_str2 相符
assert pretty(3*kg*x*m**2*y/s**2) == ascii_str2


def test_pretty_Subs():
    # 创建一个函数 f(x)
    f = Function('f')
    # 创建 Subs 对象，表示 f(x) 在 x=ph**2 处的表达式
    expr = Subs(f(x), x, ph**2)
    # 断言 expr 的 ASCII 格式化输出是否与 ascii_str 相符
    assert pretty(expr) == ascii_str
    # 断言 expr 的 Unicode 格式化输出是否与 unicode_str 相符
    assert upretty(expr) == unicode_str

    # 创建 Subs 对象，表示 f(x) 对 x 求导数，然后在 x=0 处的表达式
    expr = Subs(f(x).diff(x), x, 0)
    # 断言 expr 的 ASCII 格式化输出是否与 ascii_str 相符
    assert pretty(expr) == ascii_str
    # 断言 expr 的 Unicode 格式化输出是否与 unicode_str 相符
    assert upretty(expr) == unicode_str

    # 创建 Subs 对象，表示 f(x) 对 x 求导数再除以 y，然后在 x=0, y=1/2 处的表达式
    expr = Subs(f(x).diff(x)/y, (x, y), (0, Rational(1, 2)))
    # 断言 expr 的 ASCII 格式化输出是否与 ascii_str 相符
    assert pretty(expr) == ascii_str
    # 断言 expr 的 Unicode 格式化输出是否与 unicode_str 相符
    assert upretty(expr) == unicode_str


def test_gammas():
    # 断言 lowergamma(x, y) 的 Unicode 格式化输出是否为 "γ(x, y)"
    assert upretty(lowergamma(x, y)) == "γ(x, y)"
    # 断言 uppergamma(x, y) 的 Unicode 格式化输出是否为 "Γ(x, y)"
    assert upretty(uppergamma(x, y)) == "Γ(x, y)"
    # 断言 gamma(x) 的 Unicode 格式化输出是否为 'Γ(x)'
    assert xpretty(gamma(x), use_unicode=True) == 'Γ(x)'
    # 断言 gamma 的 Unicode 格式化输出是否为 'Γ'
    assert xpretty(gamma, use_unicode=True) == 'Γ'
    # 断言 gamma(x) 的 Unicode 格式化输出是否为 'γ(x)'
    assert xpretty(symbols('gamma', cls=Function)(x), use_unicode=True) == 'γ(x)'
    # 断言 gamma 的 Unicode 格式化输出是否为 'γ'
    assert xpretty(symbols('gamma', cls=Function), use_unicode=True) == 'γ'


def test_beta():
    # 断言 beta(x,y) 的 Unicode 格式化输出是否为 'Β(x, y)'
    assert xpretty(beta(x,y), use_unicode=True) == 'Β(x, y)'
    # 断言 beta(x,y) 的 ASCII 格式化输出是否为 'B(x, y)'
    assert xpretty(beta(x,y), use_unicode=False) == 'B(x, y)'
    # 断言 beta 的 Unicode 格式化输出是否为 'Β'
    assert xpretty(beta, use_unicode=True) == 'Β'
    # 断言 beta 的 ASCII 格式化输出是否为 'B'
    assert xpretty(beta, use_unicode=False) == 'B'
    # 创建函数 beta(x)
    mybeta = Function('beta')
    # 断言 mybeta(x) 的 Unicode 格式化输出是否为 'β(x)'
    assert xpretty(mybeta(x), use_unicode=True) == 'β(x)'
    # 断言 mybeta(x, y, z) 的 ASCII 格式化输出是否为 'beta(x, y, z)'
    assert xpretty(mybeta(x, y, z), use_unicode=False) == 'beta(x, y, z)'
    # 断言 mybeta 的 Unicode 格式化输出是否为 'β'
    assert xpretty(mybeta, use_unicode=True) == 'β'


# 测试符号函数子类化
# 确保符号函数子类 mygamma 的 Unicode 格式化输出为 "mygamma"
assert xpretty(mygamma, use_unicode=True) == r"mygamma"
# 确保符号函数子类 mygamma(x) 的 Unicode 格式化输出为 "mygamma(x)"
assert xpretty(mygamma(x), use_unicode=True) == r"mygamma(x)"


def test_SingularityFunction():
    # 断言 SingularityFunction(x, 0, n) 的 Unicode 格式化输出是否与预期相符
    assert xpretty(SingularityFunction(x, 0, n), use_unicode=True) == (
"""\
   n\n\
<x> \
""")
    # 断言 SingularityFunction(x, 1, n) 的 Unicode 格式化输出是否与预期相符
    assert xpretty(SingularityFunction(x, 1, n), use_unicode=True) == (
"""\
       n\n\
<x - 1> \
""")
    # 使用 xpretty 函数处理 SingularityFunction 对象，并验证其输出是否等于...
    assert xpretty(SingularityFunction(x, -1, n), use_unicode=True) == (
"""\
       n\n\
<x + 1> \
""")
# 断言：验证使用 SingularityFunction(x, a, n) 生成的字符串是否与预期的 Unicode 形式匹配
assert xpretty(SingularityFunction(x, a, n), use_unicode=True) == (
"""\
        n\n\
<-a + x> \
""")
# 断言：验证使用 SingularityFunction(x, y, n) 生成的字符串是否与预期的 Unicode 形式匹配
assert xpretty(SingularityFunction(x, y, n), use_unicode=True) == (
"""\
       n\n\
<x - y> \
""")
# 断言：验证使用 SingularityFunction(x, 0, n) 生成的字符串是否与预期的 ASCII 形式匹配
assert xpretty(SingularityFunction(x, 0, n), use_unicode=False) == (
"""\
   n\n\
<x> \
""")
# 断言：验证使用 SingularityFunction(x, 1, n) 生成的字符串是否与预期的 ASCII 形式匹配
assert xpretty(SingularityFunction(x, 1, n), use_unicode=False) == (
"""\
       n\n\
<x - 1> \
""")
# 断言：验证使用 SingularityFunction(x, -1, n) 生成的字符串是否与预期的 ASCII 形式匹配
assert xpretty(SingularityFunction(x, -1, n), use_unicode=False) == (
"""\
       n\n\
<x + 1> \
""")
# 断言：验证使用 SingularityFunction(x, a, n) 生成的字符串是否与预期的 ASCII 形式匹配
assert xpretty(SingularityFunction(x, a, n), use_unicode=False) == (
"""\
        n\n\
<-a + x> \
""")
# 断言：验证使用 SingularityFunction(x, y, n) 生成的字符串是否与预期的 ASCII 形式匹配
assert xpretty(SingularityFunction(x, y, n), use_unicode=False) == (
"""\
       n\n\
<x - y> \
""")

def test_deltas():
    # 断言：验证 DiracDelta(x) 的美观打印结果是否为 'δ(x)'
    assert xpretty(DiracDelta(x), use_unicode=True) == 'δ(x)'
    # 断言：验证 DiracDelta(x, 1) 的美观打印结果是否为预期的 Unicode 形式
    assert xpretty(DiracDelta(x, 1), use_unicode=True) == \
"""\
 (1)    \n\
δ    (x)\
"""
    # 断言：验证 x*DiracDelta(x, 1) 的美观打印结果是否为预期的 Unicode 形式
    assert xpretty(x*DiracDelta(x, 1), use_unicode=True) == \
"""\
   (1)    \n\
x⋅δ    (x)\
"""

def test_hyper():
    expr = hyper((), (), z)
    ucode_str = \
"""\
 ┌─  ⎛  │  ⎞\n\
 ├─  ⎜  │ z⎟\n\
0╵ 0 ⎝  │  ⎠\
"""
    ascii_str = \
"""\
  _         \n\
 |_  /  |  \\\n\
 |   |  | z|\n\
0  0 \\  |  /\
"""
    # 断言：验证 hyper((), (), z) 的美观打印结果是否与预期的 ASCII 形式匹配
    assert pretty(expr) == ascii_str
    # 断言：验证 hyper((), (), z) 的美观打印结果是否与预期的 Unicode 形式匹配
    assert upretty(expr) == ucode_str

    expr = hyper((), (1,), x)
    ucode_str = \
"""\
 ┌─  ⎛  │  ⎞\n\
 ├─  ⎜  │ x⎟\n\
0╵ 1 ⎝1 │  ⎠\
"""
    ascii_str = \
"""\
  _         \n\
 |_  /  |  \\\n\
 |   |  | x|\n\
0  1 \\1 |  /\
"""
    # 断言：验证 hyper((), (1,), x) 的美观打印结果是否与预期的 ASCII 形式匹配
    assert pretty(expr) == ascii_str
    # 断言：验证 hyper((), (1,), x) 的美观打印结果是否与预期的 Unicode 形式匹配
    assert upretty(expr) == ucode_str

    expr = hyper([2], [1], x)
    ucode_str = \
"""\
 ┌─  ⎛2 │  ⎞\n\
 ├─  ⎜  │ x⎟\n\
1╵ 1 ⎝1 │  ⎠\
"""
    ascii_str = \
"""\
  _         \n\
 |_  /2 |  \\\n\
 |   |  | x|\n\
1  1 \\1 |  /\
"""
    # 断言：验证 hyper([2], [1], x) 的美观打印结果是否与预期的 ASCII 形式匹配
    assert pretty(expr) == ascii_str
    # 断言：验证 hyper([2], [1], x) 的美观打印结果是否与预期的 Unicode 形式匹配
    assert upretty(expr) == ucode_str

    expr = hyper((pi/3, -2*k), (3, 4, 5, -3), x)
    ucode_str = \
"""\
     ⎛  π         │  ⎞\n\
 ┌─  ⎜  ─, -2⋅k   │  ⎟\n\
 ├─  ⎜  3         │ x⎟\n\
2╵ 4 ⎜            │  ⎟\n\
     ⎝-3, 3, 4, 5 │  ⎠\
"""
    ascii_str = \
"""\
                      \n\
  _  / pi         |  \\\n\
 |_  | --, -2*k   |  |\n\
 |   | 3          | x|\n\
2  4 |            |  |\n\
     \\-3, 3, 4, 5 |  /\
"""

    # 断言：验证 hyper((pi/3, -2*k), (3, 4, 5, -3), x) 的美观打印结果是否与预期的 ASCII 形式匹配
    assert pretty(expr) == ascii_str
    # 断言：验证 hyper((pi/3, -2*k), (3, 4, 5, -3), x) 的美观打印结果是否与预期的 Unicode 形式匹配
    assert upretty(expr) == ucode_str

    expr = hyper((pi, S('2/3'), -2*k), (3, 4, 5, -3), x**2)
    ucode_str = \
"""\
 ┌─  ⎛2/3, π, -2⋅k │  2⎞\n\
 ├─  ⎜             │ x ⎟\n\
3╵ 4 ⎝-3, 3, 4, 5  │   ⎠\
"""
    ascii_str = \
"""\
  _                      \n\
 |_  /2/3, pi, -2*k |  2\\
 |   |              | x |
3  4 \\ -3, 3, 4, 5  |   /"""

    # 断言：验证 hyper((pi, S('2/3'), -2*k), (3, 4, 5, -3), x**2) 的美观打印结果是否与预期的 ASCII 形式匹配
    assert pretty(expr) == ascii_str
    # 断言：验证 hyper((pi, S('2/3'), -2*k), (3, 4, 5, -3), x**2) 的美观打印结果是否与预期的 Unicode 形式匹配
    assert upretty(expr) == ucode_str

    expr = hyper([1, 2], [3, 4], 1/(1/(1/(1/x + 1) + 1) + 1))
    ucode_str = \
"""\
     ⎛     │       1      ⎞\n\
     ⎜     │ ─────────────⎟\n\
     ⎜     │         1    ⎟\n\
 ┌─  ⎜1, 2 │ 1 + ─────────⎟\n\
 ├─  ⎜     │           1  ⎟\n\
# 以下是对符号表达式进行 ASCII 艺术化处理的断言测试

# 表示符号表达式为 ASCII 字符串，用于测试 pretty 函数
ascii_str = \
"""\
                           \n\
     /     |       1      \\\n\
     |     | -------------|\n\
  _  |     |         1    |\n\
 |_  |1, 2 | 1 + ---------|\n\
 |   |     |           1  |\n\
2  2 |3, 4 |     1 + -----|\n\
     |     |             1|\n\
     |     |         1 + -|\n\
     \\     |             x/\
"""

# 表示符号表达式为 Unicode 字符串，用于测试 upretty 函数
ucode_str = \
"""\
╭─╮2, 3 ⎛π, π, x     1    │  ⎞\n\
│╶┐     ⎜                 │ z⎟\n\
╰─╯4, 5 ⎝ 0, 1    1, 2, 3 │  ⎠\
"""

# 断言：将符号表达式 expr 用 pretty 函数处理后应该与 ASCII 字符串相等
assert pretty(expr) == ascii_str
# 断言：将符号表达式 expr 用 upretty 函数处理后应该与 Unicode 字符串相等
assert upretty(expr) == ucode_str
# 定义一个测试函数，测试非交换性符号的输出
def test_noncommutative():
    # 创建三个非交换符号 A, B, C
    A, B, C = symbols('A,B,C', commutative=False)

    # 创建表达式 A*B*C**-1
    expr = A*B*C**-1
    # ASCII 表示的字符串
    ascii_str = \
"""\
     -1\n\
A*B*C  \
"""
    # Unicode 表示的字符串
    ucode_str = \
"""\
     -1\n\
A⋅B⋅C  \
"""
    # 断言表达式的 ASCII 表示和 Unicode 表示与预期的字符串相同
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    # 创建表达式 C**-1*A*B
    expr = C**-1*A*B
    # ASCII 表示的字符串
    ascii_str = \
"""\
 -1    \n\
C  *A*B\
"""
    # Unicode 表示的字符串
    ucode_str = \
"""\
 -1    \n\
C  ⋅A⋅B\
"""
    # 断言表达式的 ASCII 表示和 Unicode 表示与预期的字符串相同
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    # 创建表达式 A*C**-1*B
    expr = A*C**-1*B
    # ASCII 表示的字符串
    ascii_str = \
"""\
   -1  \n\
A*C  *B\
"""
    # Unicode 表示的字符串
    ucode_str = \
"""\
   -1  \n\
A⋅C  ⋅B\
"""
    # 断言表达式的 ASCII 表示和 Unicode 表示与预期的字符串相同
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    # 创建表达式 A*C**-1*B/x
    expr = A*C**-1*B/x
    # ASCII 表示的字符串
    ascii_str = \
"""\
   -1  \n\
A*C  *B\n\
-------\n\
   x   \
"""
    # Unicode 表示的字符串
    ucode_str = \
"""\
   -1  \n\
A⋅C  ⋅B\n\
───────\n\
   x   \
"""
    # 断言表达式的 ASCII 表示和 Unicode 表示与预期的字符串相同
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    # 断言确保 pretty 函数能正确地返回给定函数对象的字符串表示形式
    assert pretty(Shi(x)) == 'Shi(x)'
    # 断言确保 pretty 函数能正确地返回给定函数对象的字符串表示形式
    assert pretty(Si(x)) == 'Si(x)'
    # 断言确保 pretty 函数能正确地返回给定函数对象的字符串表示形式
    assert pretty(Ci(x)) == 'Ci(x)'
    # 断言确保 pretty 函数能正确地返回给定函数对象的字符串表示形式
    assert pretty(Chi(x)) == 'Chi(x)'
    # 断言确保 upretty 函数能正确地返回给定函数对象的 Unicode 字符串表示形式
    assert upretty(Shi(x)) == 'Shi(x)'
    # 断言确保 upretty 函数能正确地返回给定函数对象的 Unicode 字符串表示形式
    assert upretty(Si(x)) == 'Si(x)'
    # 断言确保 upretty 函数能正确地返回给定函数对象的 Unicode 字符串表示形式
    assert upretty(Ci(x)) == 'Ci(x)'
    # 断言确保 upretty 函数能正确地返回给定函数对象的 Unicode 字符串表示形式
    assert upretty(Chi(x)) == 'Chi(x)'
def test_elliptic_functions():
    # 定义 ASCII 字符串表达式
    ascii_str = \
"""\
 /  1  \\\n\
K|-----|\n\
 \\z + 1/\
"""
    # 定义 Unicode 字符串表达式
    ucode_str = \
"""\
 ⎛  1  ⎞\n\
K⎜─────⎟\n\
 ⎝z + 1⎠\
"""
    # 计算椭圆函数 K(1/(z + 1)) 的表达式
    expr = elliptic_k(1/(z + 1))
    # 断言 pretty 函数输出的 ASCII 表示与 ascii_str 相等
    assert pretty(expr) == ascii_str
    # 断言 upretty 函数输出的 Unicode 表示与 ucode_str 相等
    assert upretty(expr) == ucode_str

    # 以下依次类似处理 elliptic_f, elliptic_e, elliptic_pi 函数的测试
    ascii_str = \
"""\
 / |  1  \\\n\
F|1|-----|\n\
 \\ |z + 1/\
"""
    ucode_str = \
"""\
 ⎛ │  1  ⎞\n\
F⎜1│─────⎟\n\
 ⎝ │z + 1⎠\
"""
    expr = elliptic_f(1, 1/(1 + z))
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    ascii_str = \
"""\
 /  1  \\\n\
E|-----|\n\
 \\z + 1/\
"""
    ucode_str = \
"""\
 ⎛  1  ⎞\n\
E⎜─────⎟\n\
 ⎝z + 1⎠\
"""
    expr = elliptic_e(1/(z + 1))
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    ascii_str = \
"""\
 / |  1  \\\n\
E|1|-----|\n\
 \\ |z + 1/\
"""
    ucode_str = \
"""\
 ⎛ │  1  ⎞\n\
E⎜1│─────⎟\n\
 ⎝ │z + 1⎠\
"""
    expr = elliptic_e(1, 1/(1 + z))
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    ascii_str = \
"""\
  / |4\\\n\
Pi|3|-|\n\
  \\ |x/\
"""
    ucode_str = \
"""\
 ⎛ │4⎞\n\
Π⎜3│─⎟\n\
 ⎝ │x⎠\
"""
    expr = elliptic_pi(3, 4/x)
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    ascii_str = \
"""\
  /   4| \\\n\
Pi|3; -|6|\n\
  \\   x| /\
"""
    ucode_str = \
"""\
 ⎛   4│ ⎞\n\
Π⎜3; ─│6⎟\n\
 ⎝   x│ ⎠\
"""
    expr = elliptic_pi(3, 4/x, 6)
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
"""
x = 0
"""

    assert pretty(Product(x**2, (x, 1, 2))**2) == \
"""\
           2
/  2      \\ \n\
|______   | \n\
| |  |   2| \n\
| |  |  x | \n\
| |  |    | \n\
\\x = 1    / \
"""
    assert upretty(Product(x**2, (x, 1, 2))**2) == \
"""\
           2
⎛  2      ⎞ \n\
⎜─┬──┬─   ⎟ \n\
⎜ │  │   2⎟ \n\
⎜ │  │  x ⎟ \n\
⎜ │  │    ⎟ \n\
⎝x = 1    ⎠ \
"""

    f = Function('f')
    assert pretty(Derivative(f(x), x)**2) == \
"""\
          2
/d       \\ \n\
|--(f(x))| \n\
\\dx      / \
"""
    assert upretty(Derivative(f(x), x)**2) == \
"""\
          2
⎛d       ⎞ \n\
⎜──(f(x))⎟ \n\
⎝dx      ⎠ \
"""


def test_issue_6739():
    ascii_str = \
"""\
  1  \n\
-----\n\
  ___\n\
\\/ x \
"""
    ucode_str = \
"""\
1 \n\
──\n\
√x\
"""
    assert pretty(1/sqrt(x)) == ascii_str
    assert upretty(1/sqrt(x)) == ucode_str


def test_complicated_symbol_unchanged():
    for symb_name in ["dexpr2_d1tau", "dexpr2^d1tau"]:
        assert pretty(Symbol(symb_name)) == symb_name


def test_categories():
    from sympy.categories import (Object, IdentityMorphism,
        NamedMorphism, Category, Diagram, DiagramGrid)

    A1 = Object("A1")
    A2 = Object("A2")
    A3 = Object("A3")

    f1 = NamedMorphism(A1, A2, "f1")
    f2 = NamedMorphism(A2, A3, "f2")
    id_A1 = IdentityMorphism(A1)

    K1 = Category("K1")

    assert pretty(A1) == "A1"
    assert upretty(A1) == "A₁"

    assert pretty(f1) == "f1:A1-->A2"
    assert upretty(f1) == "f₁:A₁——▶A₂"
    assert pretty(id_A1) == "id:A1-->A1"
    assert upretty(id_A1) == "id:A₁——▶A₁"

    assert pretty(f2*f1) == "f2*f1:A1-->A3"
    assert upretty(f2*f1) == "f₂∘f₁:A₁——▶A₃"

    assert pretty(K1) == "K1"
    assert upretty(K1) == "K₁"

    # Test how diagrams are printed.
    d = Diagram()
    assert pretty(d) == "EmptySet"
    assert upretty(d) == "∅"

    d = Diagram({f1: "unique", f2: S.EmptySet})
    assert pretty(d) == "{f2*f1:A1-->A3: EmptySet, id:A1-->A1: " \
        "EmptySet, id:A2-->A2: EmptySet, id:A3-->A3: " \
        "EmptySet, f1:A1-->A2: {unique}, f2:A2-->A3: EmptySet}"

    assert upretty(d) == "{f₂∘f₁:A₁——▶A₃: ∅, id:A₁——▶A₁: ∅, " \
        "id:A₂——▶A₂: ∅, id:A₃——▶A₃: ∅, f₁:A₁——▶A₂: {unique}, f₂:A₂——▶A₃: ∅}"

    d = Diagram({f1: "unique", f2: S.EmptySet}, {f2 * f1: "unique"})
    assert pretty(d) == "{f2*f1:A1-->A3: EmptySet, id:A1-->A1: " \
        "EmptySet, id:A2-->A2: EmptySet, id:A3-->A3: " \
        "EmptySet, f1:A1-->A2: {unique}, f2:A2-->A3: EmptySet}" \
        " ==> {f2*f1:A1-->A3: {unique}}"
    assert upretty(d) == "{f₂∘f₁:A₁——▶A₃: ∅, id:A₁——▶A₁: ∅, id:A₂——▶A₂: " \
        "∅, id:A₃——▶A₃: ∅, f₁:A₁——▶A₂: {unique}, f₂:A₂——▶A₃: ∅}"

    grid = DiagramGrid(d)
    assert pretty(grid) == "A1  A2\n      \nA3    "
    assert upretty(grid) == "A₁  A₂\n      \nA₃    "


def test_PrettyModules():
    R = QQ.old_poly_ring(x, y)
    F = R.free_module(2)
    M = F.submodule([x, y], [1, x**2])

    ucode_str = \
"""\
       2\n\
ℚ[x, y] \
"""
    # 将ASCII码字符串定义为一个多行字符串
    ascii_str = \
    """
    2\n\
QQ[x, y] \
"""
# 定义字符串常量，用于存储数学表达式的 Unicode 编码表示
# 这里是一个多行字符串，表示一个数学表达式

assert upretty(F) == ucode_str
assert pretty(F) == ascii_str
# 断言：验证符号表达式 F 的 Unicode 编码和 ASCII 编码是否与预期的 ucode_str 和 ascii_str 相同

ucode_str = \
"""\
╱        ⎡    2⎤╲\n\
╲[x, y], ⎣1, x ⎦╱\
"""
ascii_str = \
"""\
              2  \n\
<[x, y], [1, x ]>\
"""
# 定义变量 ucode_str 和 ascii_str，分别存储符号表达式 F 的 Unicode 编码和 ASCII 编码的预期字符串

assert upretty(M) == ucode_str
assert pretty(M) == ascii_str
# 断言：验证符号表达式 M 的 Unicode 编码和 ASCII 编码是否与预期的 ucode_str 和 ascii_str 相同

I = R.ideal(x**2, y)
# 创建理想 I，由环 R 中的多项式 x^2 和 y 组成

ucode_str = \
"""\
╱ 2   ╲\n\
╲x , y╱\
"""
ascii_str = \
"""\
  2    \n\
<x , y>\
"""
# 定义变量 ucode_str 和 ascii_str，分别存储理想 I 的 Unicode 编码和 ASCII 编码的预期字符串

assert upretty(I) == ucode_str
assert pretty(I) == ascii_str
# 断言：验证理想 I 的 Unicode 编码和 ASCII 编码是否与预期的 ucode_str 和 ascii_str 相同

Q = F / M
# 创建商环 Q，由符号表达式 F 和 M 的理想组成

ucode_str = \
"""\
           2     \n\
    ℚ[x, y]      \n\
─────────────────\n\
╱        ⎡    2⎤╲\n\
╲[x, y], ⎣1, x ⎦╱\
"""
ascii_str = \
"""\
            2    \n\
    QQ[x, y]     \n\
-----------------\n\
              2  \n\
<[x, y], [1, x ]>\
"""
# 定义变量 ucode_str 和 ascii_str，分别存储商环 Q 的 Unicode 编码和 ASCII 编码的预期字符串

assert upretty(Q) == ucode_str
assert pretty(Q) == ascii_str
# 断言：验证商环 Q 的 Unicode 编码和 ASCII 编码是否与预期的 ucode_str 和 ascii_str 相同

ucode_str = \
"""\
╱⎡    3⎤                                                ╲\n\
│⎢   x ⎥   ╱        ⎡    2⎤╲           ╱        ⎡    2⎤╲│\n\
│⎢1, ──⎥ + ╲[x, y], ⎣1, x ⎦╱, [2, y] + ╲[x, y], ⎣1, x ⎦╱│\n\
╲⎣   2 ⎦                                                ╱\
"""
ascii_str = \
"""\
      3                                                  \n\
     x                   2                           2   \n\
<[1, --] + <[x, y], [1, x ]>, [2, y] + <[x, y], [1, x ]>>\n\
     2                                                   \
"""
# 定义变量 ucode_str 和 ascii_str，分别存储复杂数学表达式的 Unicode 编码和 ASCII 编码的预期字符串

def test_QuotientRing():
    R = QQ.old_poly_ring(x)/[x**2 + 1]
    # 创建商环 R，由有理数域 QQ 中的 x 的旧多项式环除以多项式 x^2 + 1 得到

    ucode_str = \
"""\
  ℚ[x]  \n\
────────\n\
╱ 2    ╲\n\
╲x  + 1╱\
"""
    ascii_str = \
"""\
 QQ[x]  \n\
--------\n\
  2     \n\
<x  + 1>\
"""
    # 定义变量 ucode_str 和 ascii_str，分别存储商环 R 的 Unicode 编码和 ASCII 编码的预期字符串

    assert upretty(R) == ucode_str
    assert pretty(R) == ascii_str
    # 断言：验证商环 R 的 Unicode 编码和 ASCII 编码是否与预期的 ucode_str 和 ascii_str 相同

    ucode_str = \
"""\
    ╱ 2    ╲\n\
1 + ╲x  + 1╱\
"""
    ascii_str = \
"""\
      2     \n\
1 + <x  + 1>\
"""
    # 定义变量 ucode_str 和 ascii_str，分别存储 R.one（环 R 的单位元素）的 Unicode 编码和 ASCII 编码的预期字符串

    assert upretty(R.one) == ucode_str
    assert pretty(R.one) == ascii_str
    # 断言：验证环 R 的单位元素的 Unicode 编码和 ASCII 编码是否与预期的 ucode_str 和 ascii_str 相同

def test_Homomorphism():
    from sympy.polys.agca import homomorphism

    R = QQ.old_poly_ring(x)
    # 创建多项式环 R，由有理数域 QQ 中的 x 组成

    expr = homomorphism(R.free_module(1), R.free_module(1), [0])
    # 创建同态映射 expr，将自由模数为 1 的 R 到自由模数为 1 的 R，映射值为 [0]

    ucode_str = \
"""\
          1         1\n\
[0] : ℚ[x]  ──> ℚ[x] \
"""
    ascii_str = \
"""\
           1          1\n\
[0] : QQ[x]  --> QQ[x] \
"""
    # 定义变量 ucode_str 和 ascii_str，分别存储同态映射 expr 的 Unicode 编码和 ASCII 编码的预期字符串

    assert upretty(expr) == ucode_str
    assert pretty(expr) == ascii_str
    # 断言：验证同态映射 expr 的 Unicode 编码和 ASCII 编码是否与预期的 ucode_str 和 ascii_str 相同

    expr = homomorphism(R.free_module(2), R.free_module(2), [0, 0])
    # 创建同态映射 expr，将自由模数为 2 的 R 到自由模数为 2 的 R，映射值为 [0, 0]

    ucode_str = \
"""\
⎡0  0⎤       2         2\n\
⎢    ⎥ : ℚ[x]  ──> ℚ[x] \n\
⎣0  0⎦                  \
"""
    ascii_str = \
"""\
[0  0]        2          2\n\
[    ] : QQ[x]  --> QQ[x] \n\
[0  0]                    \
"""
    # 定义变量 ucode_str 和 ascii_str，分别存储同态映射 expr 的 Unicode 编码和 ASCII 编码的预期字符串

    assert upretty(expr) == ucode_str
    assert pretty(expr) == ascii_str
    # 断言：验证同态映射 expr 的 Unicode 编码和 ASCII 编码是否与预期的 ucode_str 和 ascii_str 相同

    expr = homomorphism(R.free_module(1), R.free_module(1) / [[x]], [0])
    # 创建同态映射 expr，将自由模数为 1 的 R 到 R 中由多项式 x 生成的理想的商模数为 1 的 R，映射值为 [0]

    ucode_str = \
"""\
                    1\n\
          1     ℚ[x] \n\
[0] : ℚ[x]  ──> ─────\n\
                <[x]>\
"""
    ascii_str = \
"""\
                      1\n\
           1     QQ[x] \n\
[0] : QQ[x]  --> ------\n\
                 <[x]>
    # 断言表达式 `upretty(expr)` 的结果应该等于 `ucode_str`
    assert upretty(expr) == ucode_str
    # 断言表达式 `pretty(expr)` 的结果应该等于 `ascii_str`
    assert pretty(expr) == ascii_str
# 定义测试函数 test_Tr，用于测试符号 A 和 B 的非交换乘积的迹
def test_Tr():
    # 创建符号 A 和 B，指定它们为非交换的
    A, B = symbols('A B', commutative=False)
    # 计算 A*B 的迹
    t = Tr(A*B)
    # 断言迹的美观输出等于 'Tr(A*B)'
    assert pretty(t) == r'Tr(A*B)'
    # 断言迹的 Unicode 美观输出等于 'Tr(A⋅B)'
    assert upretty(t) == 'Tr(A⋅B)'


# 定义测试函数 test_pretty_Add，用于测试加法表达式的美观输出
def test_pretty_Add():
    # 构建一个加法表达式，其中包含未评估的 Mul(-2, x - 2) 和常数 5
    eq = Mul(-2, x - 2, evaluate=False) + 5
    # 断言加法表达式的美观输出等于 '5 - 2*(x - 2)'
    assert pretty(eq) == '5 - 2*(x - 2)'


# 定义测试函数 test_issue_7179，用于测试逻辑运算符的 Unicode 美观输出
def test_issue_7179():
    # 断言逻辑非运算符 Not(Equivalent(x, y)) 的 Unicode 美观输出等于 'x ⇎ y'
    assert upretty(Not(Equivalent(x, y))) == 'x ⇎ y'
    # 断言逻辑非运算符 Not(Implies(x, y)) 的 Unicode 美观输出等于 'x ↛ y'
    assert upretty(Not(Implies(x, y))) == 'x ↛ y'


# 定义测试函数 test_issue_7180，用于测试等价运算符的 Unicode 美观输出
def test_issue_7180():
    # 断言等价运算符 Equivalent(x, y) 的 Unicode 美观输出等于 'x ⇔ y'
    assert upretty(Equivalent(x, y)) == 'x ⇔ y'


# 定义测试函数 test_pretty_Complement，用于测试集合补运算的美观输出
def test_pretty_Complement():
    # 断言实数集 S.Reals 减去自然数集 S.Naturals 的美观输出等于 '(-oo, oo) \\ Naturals'
    assert pretty(S.Reals - S.Naturals) == '(-oo, oo) \\ Naturals'
    # 断言实数集 S.Reals 减去自然数集 S.Naturals 的 Unicode 美观输出等于 'ℝ \\ ℕ'
    assert upretty(S.Reals - S.Naturals) == 'ℝ \\ ℕ'
    # 断言实数集 S.Reals 减去自然数零集 S.Naturals0 的美观输出等于 '(-oo, oo) \\ Naturals0'
    assert pretty(S.Reals - S.Naturals0) == '(-oo, oo) \\ Naturals0'
    # 断言实数集 S.Reals 减去自然数零集 S.Naturals0 的 Unicode 美观输出等于 'ℝ \\ ℕ₀'
    assert upretty(S.Reals - S.Naturals0) == 'ℝ \\ ℕ₀'


# 定义测试函数 test_pretty_SymmetricDifference，用于测试对称差运算的 Unicode 美观输出
def test_pretty_SymmetricDifference():
    # 导入 SymmetricDifference 类并测试其运算
    from sympy.sets.sets import SymmetricDifference
    # 断言区间 [2, 3] 和 [3, 5] 的对称差运算的 Unicode 美观输出等于 '[2, 3] ∆ [3, 5]'
    assert upretty(SymmetricDifference(Interval(2,3), Interval(3,5), evaluate=False)) == '[2, 3] ∆ [3, 5]'
    # 使用 raises 检查未实现的 SymmetricDifference 运算
    with raises(NotImplementedError):
        pretty(SymmetricDifference(Interval(2,3), Interval(3,5), evaluate=False))


# 定义测试函数 test_pretty_Contains，用于测试包含运算符的美观输出
def test_pretty_Contains():
    # 断言包含运算符 Contains(x, S.Integers) 的美观输出等于 'Contains(x, Integers)'
    assert pretty(Contains(x, S.Integers)) == 'Contains(x, Integers)'
    # 断言包含运算符 Contains(x, S.Integers) 的 Unicode 美观输出等于 'x ∈ ℤ'
    assert upretty(Contains(x, S.Integers)) == 'x ∈ ℤ'


# 定义测试函数 test_issue_8292，用于测试复杂表达式的美观输出
def test_issue_8292():
    # 导入 sympify 函数，并使用其创建一个复杂的符号表达式 e
    from sympy.core import sympify
    e = sympify('((x+x**4)/(x-1))-(2*(x-1)**4/(x-1)**4)', evaluate=False)
    # 定义 e 的 Unicode 美观输出
    ucode_str = \
"""\
           4    4    \n\
  2⋅(x - 1)    x  + x\n\
- ────────── + ──────\n\
          4    x - 1 \n\
   (x - 1)           \
"""
    # 定义 e 的 ASCII 美观输出
    ascii_str = \
"""\
           4    4    \n\
  2*(x - 1)    x  + x\n\
- ---------- + ------\n\
          4    x - 1 \n\
   (x - 1)           \
"""
    # 断言 e 的 ASCII 美观输出等于 ascii_str
    assert pretty(e) == ascii_str
    # 断言 e 的 Unicode 美观输出等于 ucode_str
    assert upretty(e) == ucode_str


# 定义测试函数 test_issue_4335，用于测试微分表达式的美观输出
def test_issue_4335():
    # 定义符号函数 y，并创建微分表达式 -y(x).diff(x)
    y = Function('y')
    expr = -y(x).diff(x)
    # 定义微分表达式的 Unicode 美观输出
    ucode_str = \
"""\
 d       \n\
-──(y(x))\n\
 dx      \
"""
    # 定义微分表达式的 ASCII 美观输出
    ascii_str = \
"""\
  d       \n\
- --(y(x))\n\
  dx      \
"""
    # 断言微分表达式的 ASCII 美观输出等于 ascii_str
    assert pretty(expr) == ascii_str
    # 断言微分表达式的 Unicode 美观输出等于 ucode_str
    assert upretty(expr) == ucode_str


# 定义测试函数 test_issue_8344，用于测试数学表达式的 Unicode 美观输出
def test_issue_8344():
    # 导入 sympify 函数，并使用其创建一个数学表达式 e
    from sympy.core import sympify
    e = sympify('2*x*y**2/1**2 + 1', evaluate=False)
    # 定义 e 的 Unicode 美观输出
    ucode_str = \
"""\
     2    \n\
2⋅x⋅y     \n\
────── + 1\n\
   2      \n\
  1       \
"""
    # 断言 e 的 Unicode 美观输出等于 ucode_str
    assert upretty(e) == ucode_str


# 定义测试函数 test_issue_6324，用于测试幂次表达式的 Unicode 美观输出
def test_issue_6324():
    # 使用 Pow 类创建幂次表达式 x = Pow(2, 3, evaluate=False) 和 y = Pow(10, -2, evaluate=False)
    x = Pow(2, 3, evaluate=False)
    y = Pow(10, -2, evaluate=False)
    # 创建乘法表达式 e = Mul(x, y, evaluate=False)
    e = Mul(x, y, evaluate=False)
    # 定义 e 的 Unicode 美观输出
    ucode_str = \
"""\
 3 \n\
2  \
    # 定义变量 e，表示为 lambda*x*Integral(phi(t)*pi*sin(pi*t), (t, 0, 1)) + lambda*x**2*Integral(phi(t)*2*pi*sin(2*pi*t), (t, 0, 1))
    e = lamda*x*Integral(phi(t)*pi*sin(pi*t), (t, 0, 1)) + lamda*x**2*Integral(phi(t)*2*pi*sin(2*pi*t), (t, 0, 1))
    # 定义变量 ucode_str，暂未赋值，待下一行继续赋值
    ucode_str = \
"""\
     1                              1                   \n\
   2 ⌠                              ⌠                   \n\
λ⋅x ⋅⎮ 2⋅π⋅φ(t)⋅sin(2⋅π⋅t) dt + λ⋅x⋅⎮ π⋅φ(t)⋅sin(π⋅t) dt\n\
     ⌡                              ⌡                   \n\
     0                              0                   \
"""
# 定义一个多行字符串，包含数学表达式，表示某个复杂的数学公式


def test_issue_9877():
    ucode_str1 = '(2, 3) ∪ ([1, 2] \\ {x})'
    a, b, c = Interval(2, 3, True, True), Interval(1, 2), FiniteSet(x)
    assert upretty(Union(a, Complement(b, c))) == ucode_str1

    ucode_str2 = '{x} ∩ {y} ∩ ({z} \\ [1, 2])'
    d, e, f, g = FiniteSet(x), FiniteSet(y), FiniteSet(z), Interval(1, 2)
    assert upretty(Intersection(d, e, Complement(f, g))) == ucode_str2


def test_issue_13651():
    expr1 = c + Mul(-1, a + b, evaluate=False)
    assert pretty(expr1) == 'c - (a + b)'
    expr2 = c + Mul(-1, a - b + d, evaluate=False)
    assert pretty(expr2) == 'c - (a - b + d)'


def test_pretty_primenu():
    from sympy.functions.combinatorial.numbers import primenu

    ascii_str1 = "nu(n)"
    ucode_str1 = "ν(n)"

    n = symbols('n', integer=True)
    assert pretty(primenu(n)) == ascii_str1
    assert upretty(primenu(n)) == ucode_str1


def test_pretty_primeomega():
    from sympy.functions.combinatorial.numbers import primeomega

    ascii_str1 = "Omega(n)"
    ucode_str1 = "Ω(n)"

    n = symbols('n', integer=True)
    assert pretty(primeomega(n)) == ascii_str1
    assert upretty(primeomega(n)) == ucode_str1


def test_pretty_Mod():
    from sympy.core import Mod

    ascii_str1 = "x mod 7"
    ucode_str1 = "x mod 7"

    ascii_str2 = "(x + 1) mod 7"
    ucode_str2 = "(x + 1) mod 7"

    ascii_str3 = "2*x mod 7"
    ucode_str3 = "2⋅x mod 7"

    ascii_str4 = "(x mod 7) + 1"
    ucode_str4 = "(x mod 7) + 1"

    ascii_str5 = "2*(x mod 7)"
    ucode_str5 = "2⋅(x mod 7)"

    x = symbols('x', integer=True)
    assert pretty(Mod(x, 7)) == ascii_str1
    assert upretty(Mod(x, 7)) == ucode_str1
    assert pretty(Mod(x + 1, 7)) == ascii_str2
    assert upretty(Mod(x + 1, 7)) == ucode_str2
    assert pretty(Mod(2 * x, 7)) == ascii_str3
    assert upretty(Mod(2 * x, 7)) == ucode_str3
    assert pretty(Mod(x, 7) + 1) == ascii_str4
    assert upretty(Mod(x, 7) + 1) == ucode_str4
    assert pretty(2 * Mod(x, 7)) == ascii_str5
    assert upretty(2 * Mod(x, 7)) == ucode_str5


def test_issue_11801():
    assert pretty(Symbol("")) == ""
    assert upretty(Symbol("")) == ""


def test_pretty_UnevaluatedExpr():
    x = symbols('x')
    he = UnevaluatedExpr(1/x)

    ucode_str = \
"""\
1\n\
─\n\
x\
"""
    assert upretty(he) == ucode_str

    ucode_str = \
"""\
   2\n\
⎛1⎞ \n\
⎜─⎟ \n\
⎝x⎠ \
"""
    assert upretty(he**2) == ucode_str

    ucode_str = \
"""\
    1\n\
1 + ─\n\
    x\
"""
    assert upretty(he + 1) == ucode_str

    ucode_str = \
('''\
  1\n\
x⋅─\n\
  x\
''')
    assert upretty(x*he) == ucode_str


def test_issue_10472():
    M = (Matrix([[0, 0], [0, 0]]), Matrix([0, 0]))
    ucode_str = \
"""\
⎛⎡0  0⎤  ⎡0⎤⎞
⎜⎢    ⎥, ⎢ ⎥⎟
⎝⎣0  0⎦  ⎣0⎦⎠\
"""
# 断言检查矩阵 M 的 Unicode 格式化输出是否与指定的 Unicode 代码字符串相匹配
assert upretty(M) == ucode_str


def test_MatrixElement_printing():
    # 对问题 #11821 的测试案例
    A = MatrixSymbol("A", 1, 3)
    B = MatrixSymbol("B", 1, 3)
    C = MatrixSymbol("C", 1, 3)

    ascii_str1 = "A_00"
    ucode_str1 = "A₀₀"
    # 断言检查矩阵元素 A[0, 0] 的 ASCII 格式化输出是否与指定的 ASCII 字符串相匹配
    assert pretty(A[0, 0])  == ascii_str1
    # 断言检查矩阵元素 A[0, 0] 的 Unicode 格式化输出是否与指定的 Unicode 字符串相匹配
    assert upretty(A[0, 0]) == ucode_str1

    ascii_str1 = "3*A_00"
    ucode_str1 = "3⋅A₀₀"
    # 断言检查表达式 3*A[0, 0] 的 ASCII 格式化输出是否与指定的 ASCII 字符串相匹配
    assert pretty(3*A[0, 0])  == ascii_str1
    # 断言检查表达式 3*A[0, 0] 的 Unicode 格式化输出是否与指定的 Unicode 字符串相匹配
    assert upretty(3*A[0, 0]) == ucode_str1

    ascii_str1 = "(-B + A)[0, 0]"
    ucode_str1 = "(-B + A)[0, 0]"
    # 使用 A - B 替换 C[0, 0] 中的 C，然后断言其 ASCII 和 Unicode 格式化输出是否与指定的字符串相匹配
    F = C[0, 0].subs(C, A - B)
    assert pretty(F)  == ascii_str1
    assert upretty(F) == ucode_str1


def test_issue_12675():
    x, y, t, j = symbols('x y t j')
    e = CoordSys3D('e')

    ucode_str = \
"""\
⎛   t⎞    \n\
⎜⎛x⎞ ⎟ j_e\n\
⎜⎜─⎟ ⎟    \n\
⎝⎝y⎠ ⎠    \
"""
    # 断言检查表达式 ((x/y)**t)*e.j 的 Unicode 格式化输出是否与指定的 Unicode 字符串相匹配
    assert upretty((x/y)**t*e.j) == ucode_str
    ucode_str = \
"""\
⎛1⎞    \n\
⎜─⎟ j_e\n\
⎝y⎠    \
"""
    # 断言检查表达式 (1/y)*e.j 的 Unicode 格式化输出是否与指定的 Unicode 字符串相匹配
    assert upretty((1/y)*e.j) == ucode_str


def test_MatrixSymbol_printing():
    # 对问题 #14237 的测试案例
    A = MatrixSymbol("A", 3, 3)
    B = MatrixSymbol("B", 3, 3)
    C = MatrixSymbol("C", 3, 3)
    # 断言检查矩阵表达式 -A*B*C 的 ASCII 格式化输出是否与指定的 ASCII 字符串相匹配
    assert pretty(-A*B*C) == "-A*B*C"
    # 断言检查矩阵表达式 A - B 的 ASCII 格式化输出是否与指定的 ASCII 字符串相匹配
    assert pretty(A - B) == "-B + A"
    # 断言检查矩阵表达式 A*B*C - A*B - B*C 的 ASCII 格式化输出是否与指定的 ASCII 字符串相匹配
    assert pretty(A*B*C - A*B - B*C) == "-A*B -B*C + A*B*C"

    # 对问题 #14814 的测试案例
    x = MatrixSymbol('x', n, n)
    y = MatrixSymbol('y*', n, n)
    # 断言检查矩阵表达式 -a*x + -2*y*y 的 ASCII 格式化输出是否与指定的 ASCII 字符串相匹配
    assert pretty(-a*x + -2*y*y) == ascii_str


def test_degree_printing():
    expr1 = 90*degree
    # 断言检查角度表达式 90*degree 的 ASCII 格式化输出是否为 '90°'
    assert pretty(expr1) == '90°'
    expr2 = x*degree
    # 断言检查角度表达式 x*degree 的 ASCII 格式化输出是否为 'x°'
    assert pretty(expr2) == 'x°'
    expr3 = cos(x*degree + 90*degree)
    # 断言检查角度表达式 cos(x*degree + 90*degree) 的 ASCII 格式化输出是否为 'cos(x° + 90°)'
    assert pretty(expr3) == 'cos(x° + 90°)'


def test_vector_expr_pretty_printing():
    A = CoordSys3D('A')

    # 断言检查向量叉乘的 Unicode 格式化输出是否与指定的 Unicode 字符串相匹配
    assert upretty(Cross(A.i, A.x*A.i+3*A.y*A.j)) == "(i_A)×((x_A) i_A + (3⋅y_A) j_A)"
    # 断言检查标量乘以向量叉乘的 Unicode 格式化输出是否与指定的 Unicode 字符串相匹配
    assert upretty(x*Cross(A.i, A.j)) == 'x⋅(i_A)×(j_A)'

    # 断言检查向量的旋度的 Unicode 格式化输出是否与指定的 Unicode 字符串相匹配
    assert upretty(Curl(A.x*A.i + 3*A.y*A.j)) == "∇×((x_A) i_A + (3⋅y_A) j_A)"
    # 断言检查向量的散度的 Unicode 格式化输出是否与指定的 Unicode 字符串相匹配
    assert upretty(Divergence(A.x*A.i + 3*A.y*A.j)) == "∇⋅((x_A) i_A + (3⋅y_A) j_A)"

    # 断言检查向量的点乘的 Unicode 格式化输出是否与指定的 Unicode 字符串相匹配
    assert upretty(Dot(A.i, A.x*A.i+3*A.y*A.j)) == "(i_A)⋅((x_A) i_A + (3⋅y_A) j_A)"

    # 断言检查向量的梯度的 Unicode 格式化输出是否与指定的 Unicode 字符串相匹配
    assert upretty(Gradient(A.x+3*A.y)) == "∇(x_A + 3⋅y_A)"
    # 断言检查向量的拉普拉斯算子的 Unicode 格式化输出是否与指定的 Unicode 字符串相匹配
    assert upretty(Laplacian(A.x+3*A.y)) == "∆(x_A + 3⋅y_A)"
    # TODO: add support for ASCII pretty.


def test_pretty_print_tensor_expr():
    L = TensorIndexType("L")
    i, j, k = tensor_indices("i j k", L)
    i0 = tensor_indices("i_0", L)
    A, B, C, D = tensor_heads("A B C D", [L])
    H = TensorHead("H", [L, L])

    expr = -i
    # 断言检查张量表达式 -i 的 ASCII 格式化输出是否与指定的 ASCII 字符串相匹配
    ascii_str = \
"""\
-i\
"""
    ucode_str = \
"""\
-i\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = A(i)
"""
expr = A(-i)
ascii_str = \
"""\
  \n\
A \n\
 i\
"""
ucode_str = \
"""\
  \n\
A \n\
 i\
"""
assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str

expr = -3*A(-i)
ascii_str = \
"""\
     \n\
-3*A \n\
    i\
"""
ucode_str = \
"""\
     \n\
-3⋅A \n\
    i\
"""
assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str

expr = H(i, -j)
ascii_str = \
"""\
 i \n\
H  \n\
  j\
"""
ucode_str = \
"""\
 i \n\
H  \n\
  j\
"""
assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str

expr = H(i, -i)
ascii_str = \
"""\
 L_0   \n\
H      \n\
    L_0\
"""
ucode_str = \
"""\
 L₀  \n\
H    \n\
   L₀\
"""
assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str

expr = H(i, -j)*A(j)*B(k)
ascii_str = \
"""\
 i     L_0  k\n\
H    *A   *B \n\
  L_0        \
"""
ucode_str = \
"""\
 i    L₀  k\n\
H   ⋅A  ⋅B \n\
  L₀       \
"""
assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str

expr = (1+x)*A(i)
ascii_str = \
"""\
         i\n\
(x + 1)*A \n\
          \
"""
ucode_str = \
"""\
         i\n\
(x + 1)⋅A \n\
          \
"""
assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str

expr = A(i) + 3*B(i)
ascii_str = \
"""\
   i    i\n\
3*B  + A \n\
         \
"""
ucode_str = \
"""\
   i    i\n\
3⋅B  + A \n\
         \
"""
assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str



"""
def test_pretty_print_tensor_partial_deriv():
    from sympy.tensor.toperators import PartialDerivative

    L = TensorIndexType("L")
    i, j, k = tensor_indices("i j k", L)

    A, B, C, D = tensor_heads("A B C D", [L])

    H = TensorHead("H", [L])

    expr = PartialDerivative(A(i), A(j))
    ascii_str = \
"""\
 d / i\\\n\
---|A |\n\
  j\\  /\n\
dA     \n\
       \
"""
ucode_str = \
"""\
 ∂ ⎛ i⎞\n\
───⎜A ⎟\n\
  j⎝  ⎠\n\
∂A     \n\
       \
"""
assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str

expr = A(i)*PartialDerivative(H(k, -i), A(j))
ascii_str = \
"""\
 L_0  d / k   \\\n\
A   *---|H    |\n\
       j\\  L_0/\n\
     dA        \n\
               \
"""
ucode_str = \
"""\
 L₀  ∂ ⎛ k  ⎞\n\
A  ⋅───⎜H   ⎟\n\
      j⎝  L₀⎠\n\
    ∂A       \n\
             \
"""
assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str

expr = A(i)*PartialDerivative(B(k)*C(-i) + 3*H(k, -i), A(j))
ascii_str = \
"""\
 L_0  d /   k       k     \\\n\
A   *---|3*H     + B *C   |\n\
       j\\    L_0       L_0/\n\
     dA                    \n\
                           \
"""
ucode_str = \
"""\
 L₀  ∂ ⎛   k      k    ⎞\n\
A  ⋅───⎜3⋅H    + B ⋅C  ⎟\n\
      j⎝    L₀       L₀⎠\n\
    ∂A                  \n\
                        \
"""
assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str
    # 定义一个表达式，包括 A(i) 和 B(i) 的和，乘以 C(j) 对 D(j) 的偏导数
    expr = (A(i) + B(i)) * PartialDerivative(C(j), D(j))
    
    # 继续拼接 ASCII 字符串
    ascii_str = \
"""
这段代码包含了多个测试用例，用于验证不同数学表达式的美观打印。

每个测试用例包括一个数学表达式和其预期的 ASCII 和 Unicode 打印结果的断言。

以下是每个测试用例的具体注释：
"""

# 第一个测试用例
expr = (A(i) + B(i))*PartialDerivative(C(-i), D(j))
ascii_str = \
"""\
/ L_0    L_0\\  d /    \\\n\
|A    + B   |*---|C   |\n\
\\           /   j\\ L_0/\n\
              dD       \n\
                       \
"""
ucode_str = \
"""\
⎛ L₀    L₀⎞  ∂ ⎛   ⎞\n\
⎜A   + B  ⎟⋅───⎜C  ⎟\n\
⎝         ⎠   j⎝ L₀⎠\n\
            ∂D      \n\
                    \
"""
assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str

# 第二个测试用例
expr = PartialDerivative(B(-i) + A(-i), A(-j), A(-n))
ucode_str = """\
   2            \n\
  ∂    ⎛       ⎞\n\
───────⎜A  + B ⎟\n\
       ⎝ i    i⎠\n\
∂A  ∂A          \n\
  n   j         \
"""
assert upretty(expr) == ucode_str

# 第三个测试用例
expr = PartialDerivative(3*A(-i), A(-j), A(-n))
ucode_str = """\
   2         \n\
  ∂    ⎛    ⎞\n\
───────⎜3⋅A ⎟\n\
       ⎝   i⎠\n\
∂A  ∂A       \n\
  n   j      \
"""
assert upretty(expr) == ucode_str

# 第四个测试用例
expr = TensorElement(H(i, j), {i:1})
ascii_str = \
"""\
 i=1,j\n\
H     \n\
      \
"""
ucode_str = ascii_str
assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str

# 第五个测试用例
expr = TensorElement(H(i, j), {i: 1, j: 1})
ascii_str = \
"""\
 i=1,j=1\n\
H       \n\
        \
"""
ucode_str = ascii_str
assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str

# 第六个测试用例
expr = TensorElement(H(i, j), {j: 1})
ascii_str = \
"""\
 i,j=1\n\
H     \n\
      \
"""
ucode_str = ascii_str

# 第七个测试用例
expr = TensorElement(H(-i, j), {-i: 1})
ascii_str = \
"""\
    j\n\
H    \n\
 i=1 \
"""
ucode_str = ascii_str
assert pretty(expr) == ascii_str
assert upretty(expr) == ucode_str
    # 使用 pretty 函数对 lerchphi(a, 1, 2) 的结果进行格式化美化显示
    pretty(lerchphi(a, 1, 2))
    # 将 'Φ(a, 1, 2)' 赋给变量 uresult，表示 lerchphi(a, 1, 2) 的 Unicode 显示形式
    uresult = 'Φ(a, 1, 2)'
    # 将 'lerchphi(a, 1, 2)' 赋给变量 aresult，表示 lerchphi(a, 1, 2) 的字符串表示形式
    aresult = 'lerchphi(a, 1, 2)'
    # 断言 pretty(lerchphi(a, 1, 2)) 的结果应该等于 aresult，用于测试 pretty 函数的输出是否符合预期的字符串形式
    assert pretty(lerchphi(a, 1, 2)) == aresult
    # 断言 upretty(lerchphi(a, 1, 2)) 的结果应该等于 uresult，用于测试 upretty 函数的输出是否符合预期的 Unicode 形式
    assert upretty(lerchphi(a, 1, 2)) == uresult
def test_issue_15583():

    # 创建一个参考坐标系对象 N
    N = mechanics.ReferenceFrame('N')
    # 预期结果字符串
    result = '(n_x, n_y, n_z)'
    # 使用 pretty 函数美化 N 的 x、y、z 分量，并返回结果字符串
    e = pretty((N.x, N.y, N.z))
    # 断言预期结果和美化结果相等
    assert e == result


def test_matrixSymbolBold():
    # Issue 15871

    # 定义一个函数 boldpretty，用于将表达式 expr 以粗体形式美化
    def boldpretty(expr):
        return xpretty(expr, use_unicode=True, wrap_line=False, mat_symbol_style="bold")

    # 导入 trace 函数
    from sympy.matrices.expressions.trace import trace
    # 定义一个 2x2 的矩阵符号 A
    A = MatrixSymbol("A", 2, 2)
    # 断言 boldpretty 对 trace(A) 的结果是 'tr(𝐀)'
    assert boldpretty(trace(A)) == 'tr(𝐀)'

    # 重新定义 A、B、C 为 3x3 的矩阵符号
    A = MatrixSymbol("A", 3, 3)
    B = MatrixSymbol("B", 3, 3)
    C = MatrixSymbol("C", 3, 3)

    # 断言 boldpretty 对 -A 的结果是 '-𝐀'
    assert boldpretty(-A) == '-𝐀'
    # 断言 boldpretty 对 A - A*B - B 的结果是 '-𝐁 -𝐀⋅𝐁 + 𝐀'
    assert boldpretty(A - A*B - B) == '-𝐁 -𝐀⋅𝐁 + 𝐀'
    # 断言 boldpretty 对 -A*B - A*B*C - B 的结果是 '-𝐁 -𝐀⋅𝐁 -𝐀⋅𝐁⋅𝐂'
    assert boldpretty(-A*B - A*B*C - B) == '-𝐁 -𝐀⋅𝐁 -𝐀⋅𝐁⋅𝐂'

    # 定义一个名为 Addot 的 3x1 矩阵符号
    A = MatrixSymbol("Addot", 3, 3)
    # 断言 boldpretty 对 A 的结果是 '𝐀̈'
    assert boldpretty(A) == '𝐀̈'
    # 定义一个名为 omega 的 3x1 矩阵符号
    omega = MatrixSymbol("omega", 3, 3)
    # 断言 boldpretty 对 omega 的结果是 'ω'
    assert boldpretty(omega) == 'ω'
    # 定义一个名为 omeganorm 的 3x1 矩阵符号
    omega = MatrixSymbol("omeganorm", 3, 3)
    # 断言 boldpretty 对 omeganorm 的结果是 '‖ω‖'
    assert boldpretty(omega) == '‖ω‖'

    # 定义一个符号 alpha
    a = Symbol('alpha')
    # 定义符号 b
    b = Symbol('b')
    # 定义一个名为 c 的 3x1 矩阵符号
    c = MatrixSymbol("c", 3, 1)
    # 定义一个名为 d 的 3x1 矩阵符号
    d = MatrixSymbol("d", 3, 1)

    # 断言 boldpretty 对 a*B*c+b*d 的结果是 'b⋅𝐝 + α⋅𝐁⋅𝐜'
    assert boldpretty(a*B*c+b*d) == 'b⋅𝐝 + α⋅𝐁⋅𝐜'

    # 重新定义 d 为名为 delta 的 3x1 矩阵符号
    d = MatrixSymbol("delta", 3, 1)
    # 重新定义 B 为名为 Beta 的 3x3 矩阵符号
    B = MatrixSymbol("Beta", 3, 3)

    # 断言 boldpretty 对 a*B*c+b*d 的结果是 'b⋅δ + α⋅Β⋅𝐜'
    assert boldpretty(a*B*c+b*d) == 'b⋅δ + α⋅Β⋅𝐜'

    # 定义一个名为 A_2 的 3x3 矩阵符号
    A = MatrixSymbol("A_2", 3, 3)
    # 断言 boldpretty 对 A 的结果是 '𝐀₂'


def test_center_accent():
    # 断言 center_accent 函数对 'a' 加 '\N{COMBINING TILDE}' 的结果是 'ã'
    assert center_accent('a', '\N{COMBINING TILDE}') == 'ã'
    # 断言 center_accent 函数对 'aa' 加 '\N{COMBINING TILDE}' 的结果是 'aã'
    assert center_accent('aa', '\N{COMBINING TILDE}') == 'aã'
    # 断言 center_accent 函数对 'aaa' 加 '\N{COMBINING TILDE}' 的结果是 'aãa'
    assert center_accent('aaa', '\N{COMBINING TILDE}') == 'aãa'
    # 断言 center_accent 函数对 'aaaa' 加 '\N{COMBINING TILDE}' 的结果是 'aaãa'
    assert center_accent('aaaa', '\N{COMBINING TILDE}') == 'aaãa'
    # 断言 center_accent 函数对 'aaaaa' 加 '\N{COMBINING TILDE}' 的结果是 'aaãaa'
    assert center_accent('aaaaa', '\N{COMBINING TILDE}') == 'aaãaa'
    # 断言 center_accent 函数对 'abcdefg' 加 '\N{COMBINING FOUR DOTS ABOVE}' 的结果是 'abcd⃜efg'


def test_imaginary_unit():
    # 导入 pretty 函数，因为它在之前被重新定义过
    from sympy.printing.pretty import pretty
    # 断言 pretty 对 1 + I 使用 ASCII 模式时的结果是 '1 + I'
    assert pretty(1 + I, use_unicode=False) == '1 + I'
    # 断言 pretty 对 1 + I 使用 Unicode 模式时的结果是 '1 + ⅈ'
    assert pretty(1 + I, use_unicode=True) == '1 + ⅈ'
    # 断言 pretty 对 1 + I 使用 ASCII 模式，并指定虚数单位为 'j' 时的结果是 '1 + I'
    assert pretty(1 + I, use_unicode=False, imaginary_unit='j') == '1 + I'
    # 断言 pretty 对 1 + I 使用 Unicode 模式，并指定虚数单位为 'j' 时的结果是 '1 + ⅉ'

    # 断言调用 Lambda: pretty(I, imaginary_unit=I) 会抛出 TypeError 异常
    raises(TypeError, lambda: pretty(I, imaginary_unit=I))
    # 断言调用 Lambda: pretty(I, imaginary_unit="kkk") 会抛出 ValueError 异常
    raises(ValueError, lambda: pretty(I, imaginary_unit="kkk"))


def test_str_special_matrices():
    # 导入 Identity、ZeroMatrix、OneMatrix 类
    from sympy.matrices import Identity, ZeroMatrix, OneMatrix
    # 断言 pretty 对 Identity(4) 的结果是 'I'
    assert pretty(Identity(4)) == 'I'
    # 断言 upretty 对 Identity(4) 的结果是 '𝕀'
    assert upretty(Identity(4)) == '𝕀'
    # 断言 pretty 对 ZeroMatrix(2, 2) 的结果是 '0'
    assert pretty(ZeroMatrix(2, 2)) == '0'
    # 断言 upretty 对 ZeroMatrix(2, 2) 的结果是 '𝟘'
    assert upretty(ZeroMatrix(2, 2)) == '𝟘'
    # 断言 pretty 对 OneMatrix(2, 2) 的结果是 '1'
    assert pretty(OneMatrix(2, 2)) == '1'
    # 断言 upretty 对 OneMatrix(2, 2) 的结果是 '𝟙'


def test_pretty_misc_functions():
    # 断言 pretty 对 LambertW(x) 的结果是 'W(x)'
    assert pretty(LambertW(x)) == 'W(x)'
    # 断言
    # 断言语句，验证 airyaiprime(x) 的美化输出是否为 "Ai'(x)"
    assert upretty(airyaiprime(x)) == "Ai'(x)"
    
    # 断言语句，验证 airybiprime(x) 的美化输出是否为 "Bi'(x)"
    assert pretty(airybiprime(x)) == "Bi'(x)"
    
    # 断言语句，验证 airybiprime(x) 的上标美化输出是否为 "Bi'(x)"
    assert upretty(airybiprime(x)) == "Bi'(x)"
    
    # 断言语句，验证 fresnelc(x) 的美化输出是否为 'C(x)'
    assert pretty(fresnelc(x)) == 'C(x)'
    
    # 断言语句，验证 fresnelc(x) 的上标美化输出是否为 'C(x)'
    assert upretty(fresnelc(x)) == 'C(x)'
    
    # 断言语句，验证 fresnels(x) 的美化输出是否为 'S(x)'
    assert pretty(fresnels(x)) == 'S(x)'
    
    # 断言语句，验证 fresnels(x) 的上标美化输出是否为 'S(x)'
    assert upretty(fresnels(x)) == 'S(x)'
    
    # 断言语句，验证 Heaviside(x) 的美化输出是否为 'Heaviside(x)'
    assert pretty(Heaviside(x)) == 'Heaviside(x)'
    
    # 断言语句，验证 Heaviside(x) 的上标美化输出是否为 'θ(x)'
    assert upretty(Heaviside(x)) == 'θ(x)'
    
    # 断言语句，验证 Heaviside(x, y) 的美化输出是否为 'Heaviside(x, y)'
    assert pretty(Heaviside(x, y)) == 'Heaviside(x, y)'
    
    # 断言语句，验证 Heaviside(x, y) 的上标美化输出是否为 'θ(x, y)'
    assert upretty(Heaviside(x, y)) == 'θ(x, y)'
    
    # 断言语句，验证 dirichlet_eta(x) 的美化输出是否为 'dirichlet_eta(x)'
    assert pretty(dirichlet_eta(x)) == 'dirichlet_eta(x)'
    
    # 断言语句，验证 dirichlet_eta(x) 的上标美化输出是否为 'η(x)'
    assert upretty(dirichlet_eta(x)) == 'η(x)'
def test_hadamard_power():
    # 声明符号变量 m, n, p，均为整数
    m, n, p = symbols('m, n, p', integer=True)
    # 声明矩阵符号 A 和 B，大小为 m x n
    A = MatrixSymbol('A', m, n)
    B = MatrixSymbol('B', m, n)

    # 测试打印输出:
    expr = hadamard_power(A, n)
    # ASCII 字符串表示
    ascii_str = \
"""\
 .n\n\
A  \
"""
    # Unicode 字符串表示
    ucode_str = \
"""\
 ∘n\n\
A  \
"""
    # 断言打印函数的输出与预期的 ASCII 和 Unicode 字符串相等
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = hadamard_power(A, 1+n)
    # ASCII 字符串表示
    ascii_str = \
"""\
 .(n + 1)\n\
A        \
"""
    # Unicode 字符串表示
    ucode_str = \
"""\
 ∘(n + 1)\n\
A        \
"""
    # 断言打印函数的输出与预期的 ASCII 和 Unicode 字符串相等
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = hadamard_power(A*B.T, 1+n)
    # ASCII 字符串表示
    ascii_str = \
"""\
      .(n + 1)\n\
/   T\\        \n\
\\A*B /        \
"""
    # Unicode 字符串表示
    ucode_str = \
"""\
      ∘(n + 1)\n\
⎛   T⎞        \n\
⎝A⋅B ⎠        \
"""
    # 断言打印函数的输出与预期的 ASCII 和 Unicode 字符串相等
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_issue_17258():
    n = Symbol('n', integer=True)
    # 断言对于整数符号 n 的求和表达式的打印输出是否正确
    assert pretty(Sum(n, (n, -oo, 1))) == \
    '   1     \n'\
    '  __     \n'\
    '  \\ `    \n'\
    '   )    n\n'\
    '  /_,    \n'\
    'n = -oo  '

    # 断言对于整数符号 n 的求和表达式的 Unicode 打印输出是否正确
    assert upretty(Sum(n, (n, -oo, 1))) == \
"""\
  1     \n\
 ___    \n\
 ╲      \n\
  ╲     \n\
  ╱    n\n\
 ╱      \n\
 ‾‾‾    \n\
n = -∞  \
"""


def test_is_combining():
    line = "v̇_m"
    # 断言检查每个字符是否为组合字符
    assert [is_combining(sym) for sym in line] == \
        [False, True, False, False]


def test_issue_17616():
    # 断言对于给定的表达式 pi**(1/exp(1)) 的打印输出是否正确
    assert pretty(pi**(1/exp(1))) == \
   '  / -1\\\n'\
   '  \\e  /\n'\
   'pi     '

    # 断言对于给定的表达式 pi**(1/exp(1)) 的 Unicode 打印输出是否正确
    assert upretty(pi**(1/exp(1))) == \
   ' ⎛ -1⎞\n'\
   ' ⎝ℯ  ⎠\n'\
   'π     '

    # 断言对于给定的表达式 pi**(1/pi) 的打印输出是否正确
    assert pretty(pi**(1/pi)) == \
    '  1 \n'\
    '  --\n'\
    '  pi\n'\
    'pi  '

    # 断言对于给定的表达式 pi**(1/pi) 的 Unicode 打印输出是否正确
    assert upretty(pi**(1/pi)) == \
    ' 1\n'\
    ' ─\n'\
    ' π\n'\
    'π '

    # 断言对于给定的表达式 pi**(1/EulerGamma) 的打印输出是否正确
    assert pretty(pi**(1/EulerGamma)) == \
    '      1     \n'\
    '  ----------\n'\
    '  EulerGamma\n'\
    'pi          '

    # 断言对于给定的表达式 pi**(1/EulerGamma) 的 Unicode 打印输出是否正确
    assert upretty(pi**(1/EulerGamma)) == \
    ' 1\n'\
    ' ─\n'\
    ' γ\n'\
    'π '

    z = Symbol("x_17")
    # 断言对于给定的表达式 7**(1/z) 的 Unicode 打印输出是否正确
    assert upretty(7**(1/z)) == \
    'x₁₇___\n'\
    ' ╲╱ 7 '

    # 断言对于给定的表达式 7**(1/z) 的打印输出是否正确
    assert pretty(7**(1/z)) == \
    'x_17___\n'\
    '  \\/ 7 '


def test_issue_17857():
    # 断言对于给定的表达式 Range(-oo, oo) 的打印输出是否正确
    assert pretty(Range(-oo, oo)) == '{..., -1, 0, 1, ...}'
    # 断言对于给定的表达式 Range(oo, -oo, -1) 的打印输出是否正确
    assert pretty(Range(oo, -oo, -1)) == '{..., 1, 0, -1, ...}'


def test_issue_18272():
    x = Symbol('x')
    n = Symbol('n')

    # 断言对于给定的条件集合的 Unicode 打印输出是否正确
    assert upretty(ConditionSet(x, Eq(-x + exp(x), 0), S.Complexes)) == \
    '⎧  │         ⎛      x    ⎞⎫\n'\
    '⎨x │ x ∊ ℂ ∧ ⎝-x + ℯ  = 0⎠⎬\n'\
    '⎩  │                      ⎭'

    # 断言对于给定的条件集合的 Unicode 打印输出是否正确
    assert upretty(ConditionSet(x, Contains(n/2, Interval(0, oo)), FiniteSet(-n/2, n/2))) == \
    '⎧  │     ⎧-n   n⎫   ⎛n         ⎞⎫\n'\
    '⎨x │ x ∊ ⎨───, ─⎬ ∧ ⎜─ ∈ [0, ∞)⎟⎬\n'\
    '⎩  │     ⎩ 2   2⎭   ⎝2         ⎠⎭'

    # 断言对于给定的条件集合的 Unicode 打印输出是否正确
    assert upretty(ConditionSet(x, Eq(Piecewise((1, x >= 3), (x/2 - 1/2, x >= 2), (1/2, x >= 1),
                (x/2, True)) - 1/2, 0), Interval(0, 3))) == \
    '⎧  │              ⎛⎛⎧   1     for x ≥ 3⎞          ⎞⎫\n'\
    '⎪  │              ⎜⎜⎪                  ⎟          ⎟⎪\n'
    # 定义一个包含多行字符串的文本块，表示一个数学表达式的 ASCII 图形化展示
        '⎪  │              ⎜⎜⎪x                 ⎟          ⎟⎪\n'\
        '⎪  │              ⎜⎜⎪─ - 0.5  for x ≥ 2⎟          ⎟⎪\n'\
        '⎪  │              ⎜⎜⎪2                 ⎟          ⎟⎪\n'\
        '⎨x │ x ∊ [0, 3] ∧ ⎜⎜⎨                  ⎟ - 0.5 = 0⎟⎬\n'\
        '⎪  │              ⎜⎜⎪  0.5    for x ≥ 1⎟          ⎟⎪\n'\
        '⎪  │              ⎜⎜⎪                  ⎟          ⎟⎪\n'\
        '⎪  │              ⎜⎜⎪   x              ⎟          ⎟⎪\n'\
        '⎪  │              ⎜⎜⎪   ─     otherwise⎟          ⎟⎪\n'\
        '⎩  │              ⎝⎝⎩   2              ⎠          ⎠⎭'
# 测试函数，用于验证 sympy.core.symbol 模块中的 Str 类的功能
def test_Str():
    # 导入 Str 类
    from sympy.core.symbol import Str
    # 断言 Str('x') 的漂亮打印结果为 'x'
    assert pretty(Str('x')) == 'x'


# 测试符号概率相关功能
def test_symbolic_probability():
    # 定义符号变量 mu 和 sigma
    mu = symbols("mu")
    sigma = symbols("sigma", positive=True)
    # 创建正态分布变量 X
    X = Normal("X", mu, sigma)
    # 断言 E[X] 的漂亮打印结果为 'E[X]'
    assert pretty(Expectation(X)) == r'E[X]'
    # 断言 Var(X) 的漂亮打印结果为 'Var(X)'
    assert pretty(Variance(X)) == r'Var(X)'
    # 断言 P(X > 0) 的漂亮打印结果为 'P(X > 0)'
    assert pretty(Probability(X > 0)) == r'P(X > 0)'
    # 创建另一个正态分布变量 Y
    Y = Normal("Y", mu, sigma)
    # 断言 Cov(X, Y) 的漂亮打印结果为 'Cov(X, Y)'
    assert pretty(Covariance(X, Y)) == 'Cov(X, Y)'


# 测试与 issue 21758 相关的功能
def test_issue_21758():
    # 导入 piecewise_fold 和 FourierSeries 类
    from sympy.functions.elementary.piecewise import piecewise_fold
    from sympy.series.fourier import FourierSeries
    # 定义符号变量和 FourierSeries 对象 fo
    x = Symbol('x')
    k, n = symbols('k n')
    fo = FourierSeries(x, (x, -pi, pi), (0, SeqFormula(0, (k, 1, oo)), SeqFormula(
        Piecewise((-2*pi*cos(n*pi)/n + 2*sin(n*pi)/n**2, (n > -oo) & (n < oo) & Ne(n, 0)),
                  (0, True))*sin(n*x)/pi, (n, 1, oo))))
    # 断言使用 upretty 和 piecewise_fold 处理 fo 后的打印结果
    assert upretty(piecewise_fold(fo)) == \
        '⎧                      2⋅sin(3⋅x)                                \n'\
        '⎪2⋅sin(x) - sin(2⋅x) + ────────── + …  for n > -∞ ∧ n < ∞ ∧ n ≠ 0\n'\
        '⎨                          3                                     \n'\
        '⎪                                                                \n'\
        '⎩                 0                            otherwise         '
    # 断言 FourierSeries 对象的漂亮打印结果为 '0'
    assert pretty(FourierSeries(x, (x, -pi, pi), (0, SeqFormula(0, (k, 1, oo)),
                                                 SeqFormula(0, (n, 1, oo))))) == '0'


# 测试差分几何相关功能
def test_diffgeom():
    # 导入差分几何相关类
    from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseScalarField
    # 定义实数符号变量 x, y
    x,y = symbols('x y', real=True)
    # 创建流形对象 m
    m = Manifold('M', 2)
    # 断言流形对象 m 的漂亮打印结果为 'M'
    assert pretty(m) == 'M'
    # 在流形对象 m 上创建 patch 对象 p
    p = Patch('P', m)
    # 断言 patch 对象 p 的漂亮打印结果为 "P"
    assert pretty(p) == "P"
    # 在 patch 对象 p 上创建坐标系 rect
    rect = CoordSystem('rect', p, [x, y])
    # 断言坐标系 rect 的漂亮打印结果为 "rect"
    assert pretty(rect) == "rect"
    # 在坐标系 rect 上创建基本标量场对象 b
    b = BaseScalarField(rect, 0)
    # 断言基本标量场对象 b 的漂亮打印结果为 "x"
    assert pretty(b) == "x"


# 测试不推荐使用的 prettyForm 相关功能
def test_deprecated_prettyForm():
    # 使用 warns_deprecated_sympy 上下文管理器
    with warns_deprecated_sympy():
        # 导入 pretty_symbology 模块的 xstr 函数，并断言其返回结果为 '1'
        from sympy.printing.pretty.pretty_symbology import xstr
        assert xstr(1) == '1'

    with warns_deprecated_sympy():
        # 导入 stringpict 模块的 prettyForm 类，并创建对象 p
        from sympy.printing.pretty.stringpict import prettyForm
        p = prettyForm('s', unicode='s')

    with warns_deprecated_sympy():
        # 断言对象 p 的 unicode 和 s 属性都为 's'
        assert p.unicode == p.s == 's'


# 测试 center 函数的功能
def test_center():
    # 断言 center('1', 2) 的结果为 '1 '
    assert center('1', 2) == '1 '
    # 断言 center('1', 3) 的结果为 ' 1 '
    assert center('1', 3) == ' 1 '
    # 断言 center('1', 3, '-') 的结果为 '-1-'
    assert center('1', 3, '-') == '-1-'
    # 断言 center('1', 5, '-') 的结果为 '--1--'
    assert center('1', 5, '-') == '--1--'
```