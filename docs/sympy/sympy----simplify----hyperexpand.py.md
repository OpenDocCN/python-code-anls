# `D:\src\scipysrc\sympy\sympy\simplify\hyperexpand.py`

```
"""
Expand Hypergeometric (and Meijer G) functions into named
special functions.

The algorithm for doing this uses a collection of lookup tables of
hypergeometric functions, and various of their properties, to expand
many hypergeometric functions in terms of special functions.

It is based on the following paper:
      Kelly B. Roach.  Meijer G Function Representations.
      In: Proceedings of the 1997 International Symposium on Symbolic and
      Algebraic Computation, pages 205-211, New York, 1997. ACM.

It is described in great(er) detail in the Sphinx documentation.
"""

# SUMMARY OF EXTENSIONS FOR MEIJER G FUNCTIONS
#
# o z**rho G(ap, bq; z) = G(ap + rho, bq + rho; z)
#
# o denote z*d/dz by D
#
# o It is helpful to keep in mind that ap and bq play essentially symmetric
#   roles: G(1/z) has slightly altered parameters, with ap and bq interchanged.
#
# o There are four shift operators:
#   A_J = b_J - D,     J = 1, ..., n
#   B_J = 1 - a_j + D, J = 1, ..., m
#   C_J = -b_J + D,    J = m+1, ..., q
#   D_J = a_J - 1 - D, J = n+1, ..., p
#
#   A_J, C_J increment b_J
#   B_J, D_J decrement a_J
#
# o The corresponding four inverse-shift operators are defined if there
#   is no cancellation. Thus e.g. an index a_J (upper or lower) can be
#   incremented if a_J != b_i for i = 1, ..., q.
#
# o Order reduction: if b_j - a_i is a non-negative integer, where
#   j <= m and i > n, the corresponding quotient of gamma functions reduces
#   to a polynomial. Hence the G function can be expressed using a G-function
#   of lower order.
#   Similarly if j > m and i <= n.
#
#   Secondly, there are paired index theorems [Adamchik, The evaluation of
#   integrals of Bessel functions via G-function identities]. Suppose there
#   are three parameters a, b, c, where a is an a_i, i <= n, b is a b_j,
#   j <= m and c is a denominator parameter (i.e. a_i, i > n or b_j, j > m).
#   Suppose further all three differ by integers.
#   Then the order can be reduced.
#   TODO work this out in detail.
#
# o An index quadruple is called suitable if its order cannot be reduced.
#   If there exists a sequence of shift operators transforming one index
#   quadruple into another, we say one is reachable from the other.
#
# o Deciding if one index quadruple is reachable from another is tricky. For
#   this reason, we use hand-built routines to match and instantiate formulas.
#

from collections import defaultdict       # 导入 defaultdict 类来创建默认字典
from itertools import product             # 导入 product 函数用于迭代器的笛卡尔积操作
from functools import reduce              # 导入 reduce 函数用于归约操作
from math import prod                    # 导入 prod 函数用于计算可迭代对象的乘积

from sympy import SYMPY_DEBUG            # 导入 SYMPY_DEBUG 用于 Sympy 的调试功能
from sympy.core import (S, Dummy, symbols, sympify, Tuple, expand, I, pi, Mul,
    EulerGamma, oo, zoo, expand_func, Add, nan, Expr, Rational)  # 导入多个 SymPy 核心类和函数
from sympy.core.mod import Mod           # 导入 Mod 类用于模运算
from sympy.core.sorting import default_sort_key  # 导入 default_sort_key 用于排序操作
# 导入 sympy 库中的各种特殊函数和类
from sympy.functions import (exp, sqrt, root, log, lowergamma, cos,
        besseli, gamma, uppergamma, expint, erf, sin, besselj, Ei, Ci, Si, Shi,
        sinh, cosh, Chi, fresnels, fresnelc, polar_lift, exp_polar, floor, ceiling,
        rf, factorial, lerchphi, Piecewise, re, elliptic_k, elliptic_e)
from sympy.functions.elementary.complexes import polarify, unpolarify
from sympy.functions.special.hyper import (hyper, HyperRep_atanh,
        HyperRep_power1, HyperRep_power2, HyperRep_log1, HyperRep_asin1,
        HyperRep_asin2, HyperRep_sqrts1, HyperRep_sqrts2, HyperRep_log2,
        HyperRep_cosasin, HyperRep_sinasin, meijerg)
from sympy.matrices import Matrix, eye, zeros
from sympy.polys import apart, poly, Poly
from sympy.series import residue
from sympy.simplify.powsimp import powdenest
from sympy.utilities.iterables import sift

# 定义一个函数 "_mod1"，用于计算取模运算 Mod(x, 1) 或者返回 x
def _mod1(x):
    # 如果 x 是一个数值，返回 Mod(x, 1)
    if x.is_Number:
        return Mod(x, 1)
    # 将 x 分解为常数项 c 和余项 x，然后返回 Mod(c, 1) + x
    c, x = x.as_coeff_Add()
    return Mod(c, 1) + x


# 定义一个函数 "add_formulae"，用于添加特定的数学公式到 formulae 列表中
def add_formulae(formulae):
    """ Create our knowledge base. """
    # 定义符号变量 a, b, c, z
    a, b, c, z = symbols('a b c, z', cls=Dummy)

    # 定义一个内部函数 "add"，用于添加超几何函数的公式到 formulae 列表
    def add(ap, bq, res):
        func = Hyper_Function(ap, bq)
        formulae.append(Formula(func, z, res, (a, b, c)))

    # 定义另一个内部函数 "addb"，用于添加超几何函数的公式到 formulae 列表，
    # 并且包括额外的矩阵参数 B, C, M
    def addb(ap, bq, B, C, M):
        func = Hyper_Function(ap, bq)
        formulae.append(Formula(func, z, None, (a, b, c), B, C, M))

    # Luke, Y. L. (1969), The Special Functions and Their Approximations,
    # Volume 1, section 6.2

    # 添加 0F0 超几何函数的公式
    add((), (), exp(z))

    # 添加 1F0 超几何函数的公式
    add((a, ), (), HyperRep_power1(-a, z))

    # 添加 2F1 超几何函数的公式
    addb((a, a - S.Half), (2*a, ),
         Matrix([HyperRep_power2(a, z),
                 HyperRep_power2(a + S.Half, z)/2]),
         Matrix([[1, 0]]),
         Matrix([[(a - S.Half)*z/(1 - z), (S.Half - a)*z/(1 - z)],
                 [a/(1 - z), a*(z - 2)/(1 - z)]]))
    addb((1, 1), (2, ),
         Matrix([HyperRep_log1(z), 1]), Matrix([[-1/z, 0]]),
         Matrix([[0, z/(z - 1)], [0, 0]]))
    addb((S.Half, 1), (S('3/2'), ),
         Matrix([HyperRep_atanh(z), 1]),
         Matrix([[1, 0]]),
         Matrix([[Rational(-1, 2), 1/(1 - z)/2], [0, 0]]))
    addb((S.Half, S.Half), (S('3/2'), ),
         Matrix([HyperRep_asin1(z), HyperRep_power1(Rational(-1, 2), z)]),
         Matrix([[1, 0]]),
         Matrix([[Rational(-1, 2), S.Half], [0, z/(1 - z)/2]]))
    # 添加基本双曲函数的贝塞尔积分和反双曲函数
    addb((a, S.Half + a), (S.Half, ),
         Matrix([HyperRep_sqrts1(-a, z), -HyperRep_sqrts2(-a - S.Half, z)]),
         Matrix([[1, 0]]),
         Matrix([[0, -a],
                 [z*(-2*a - 1)/2/(1 - z), S.Half - z*(-2*a - 1)/(1 - z)]]))

    # 根据 Prudnikov 等人的书中的公式，计算特殊函数的积分和级数
    addb([a, -a], [S.Half],
         Matrix([HyperRep_cosasin(a, z), HyperRep_sinasin(a, z)]),
         Matrix([[1, 0]]),
         Matrix([[0, -a], [a*z/(1 - z), 1/(1 - z)/2]]))

    # 计算完全椭圆积分 K(z) 和 E(z)，这两个函数是 2F1 函数
    addb([S.Half, S.Half], [S.One],
         Matrix([elliptic_k(z), elliptic_e(z)]),
         Matrix([[2/pi, 0]]),
         Matrix([[Rational(-1, 2), -1/(2*z-2)],
                 [Rational(-1, 2), S.Half]]))

    # 计算 3F2 函数
    addb([Rational(-1, 2), 1, 1], [S.Half, 2],
         Matrix([z*HyperRep_atanh(z), HyperRep_log1(z), 1]),
         Matrix([[Rational(-2, 3), -S.One/(3*z), Rational(2, 3)]]),
         Matrix([[S.Half, 0, z/(1 - z)/2],
                 [0, 0, z/(z - 1)],
                 [0, 0, 0]]))

    # 计算另一种形式的 3F2 函数
    addb([Rational(-1, 2), 1, 1], [2, 2],
         Matrix([HyperRep_power1(S.Half, z), HyperRep_log2(z), 1]),
         Matrix([[Rational(4, 9) - 16/(9*z), 4/(3*z), 16/(9*z)]]),
         Matrix([[z/2/(z - 1), 0, 0], [1/(2*(z - 1)), 0, S.Half], [0, 0, 0]]))

    # 计算 1F1 函数
    addb([1], [b], Matrix([z**(1 - b) * exp(z) * lowergamma(b - 1, z), 1]),
         Matrix([[b - 1, 0]]),
         Matrix([[1 - b + z, 1], [0, 0]]))

    # 计算另一种形式的 1F1 函数
    addb([a], [2*a],
         Matrix([z**(S.Half - a)*exp(z/2)*besseli(a - S.Half, z/2)
                 * gamma(a + S.Half)/4**(S.Half - a),
                 z**(S.Half - a)*exp(z/2)*besseli(a + S.Half, z/2)
                 * gamma(a + S.Half)/4**(S.Half - a)]),
         Matrix([[1, 0]]),
         Matrix([[z/2, z/2], [z/2, (z/2 - 2*a)]]))

    # 定义一个复数 mz，用来计算特定形式的积分
    mz = polar_lift(-1)*z
    addb([a], [a + 1],
         Matrix([mz**(-a)*a*lowergamma(a, mz), a*exp(z)]),
         Matrix([[1, 0]]),
         Matrix([[-a, 1], [0, z]]))

    # 计算特定函数表达式
    add([Rational(-1, 2)], [S.Half], exp(z) - sqrt(pi*z)*(-I)*erf(I*sqrt(z)))

    # 添加以获得 Fresnel 函数的 Laplace 变换的良好结果
    # 参考 https://functions.wolfram.com/07.22.03.6437.01
    # 基本规则
    # add([1], [Rational(3, 4), Rational(5, 4)],
    #     sqrt(pi) * (cos(2*sqrt(polar_lift(-1)*z))*fresnelc(2*root(polar_lift(-1)*z,4)/sqrt(pi)) +
    #                sin(2*sqrt(polar_lift(-1)*z))*fresnels(2*root(polar_lift(-1)*z,4)/sqrt(pi)))
    #    / (2*root(polar_lift(-1)*z,4)))
    # Manually tuned rule
    addb([1], [Rational(3, 4), Rational(5, 4)],
         Matrix([ sqrt(pi)*(I*sinh(2*sqrt(z))*fresnels(2*root(z, 4)*exp(I*pi/4)/sqrt(pi))
                            + cosh(2*sqrt(z))*fresnelc(2*root(z, 4)*exp(I*pi/4)/sqrt(pi)))
                  * exp(-I*pi/4)/(2*root(z, 4)),
                  sqrt(pi)*root(z, 4)*(sinh(2*sqrt(z))*fresnelc(2*root(z, 4)*exp(I*pi/4)/sqrt(pi))
                                      + I*cosh(2*sqrt(z))*fresnels(2*root(z, 4)*exp(I*pi/4)/sqrt(pi)))
                  *exp(-I*pi/4)/2,
                  1 ]),
         Matrix([[1, 0, 0]]),
         Matrix([[Rational(-1, 4),              1, Rational(1, 4)],
                 [              z, Rational(1, 4),              0],
                 [              0,              0,              0]]))

    # 2F2
    addb([S.Half, a], [Rational(3, 2), a + 1],
         Matrix([a/(2*a - 1)*(-I)*sqrt(pi/z)*erf(I*sqrt(z)),
                 a/(2*a - 1)*(polar_lift(-1)*z)**(-a)*
                 lowergamma(a, polar_lift(-1)*z),
                 a/(2*a - 1)*exp(z)]),
         Matrix([[1, -1, 0]]),
         Matrix([[Rational(-1, 2), 0, 1], [0, -a, 1], [0, 0, z]]))
    # We make a "basis" of four functions instead of three, and give EulerGamma
    # an extra slot (it could just be a coefficient to 1). The advantage is
    # that this way Polys will not see multivariate polynomials (it treats
    # EulerGamma as an indeterminate), which is *way* faster.
    addb([1, 1], [2, 2],
         Matrix([Ei(z) - log(z), exp(z), 1, EulerGamma]),
         Matrix([[1/z, 0, 0, -1/z]]),
         Matrix([[0, 1, -1, 0], [0, z, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))

    # 0F1
    add((), (S.Half, ), cosh(2*sqrt(z)))
    addb([], [b],
         Matrix([gamma(b)*z**((1 - b)/2)*besseli(b - 1, 2*sqrt(z)),
                 gamma(b)*z**(1 - b/2)*besseli(b, 2*sqrt(z))]),
         Matrix([[1, 0]]), Matrix([[0, 1], [z, (1 - b)]]))

    # 0F3
    x = 4*z**Rational(1, 4)

    def fp(a, z):
        return besseli(a, x) + besselj(a, x)

    def fm(a, z):
        return besseli(a, x) - besselj(a, x)

    # TODO branching
    addb([], [S.Half, a, a + S.Half],
         Matrix([fp(2*a - 1, z), fm(2*a, z)*z**Rational(1, 4),
                 fm(2*a - 1, z)*sqrt(z), fp(2*a, z)*z**Rational(3, 4)])
         * 2**(-2*a)*gamma(2*a)*z**((1 - 2*a)/4),
         Matrix([[1, 0, 0, 0]]),
         Matrix([[0, 1, 0, 0],
                 [0, S.Half - a, 1, 0],
                 [0, 0, S.Half, 1],
                 [z, 0, 0, 1 - a]]))
    x = 2*(4*z)**Rational(1, 4)*exp_polar(I*pi/4)
    # 添加B函数计算，传入参数分别为：列表[a]，列表[a - S.Half, 2*a]，以下为函数内部矩阵内容
         (2*sqrt(polar_lift(-1)*z))**(1 - 2*a)*gamma(2*a)**2 *
         Matrix([besselj(2*a - 1, x)*besseli(2*a - 1, x),
                 x*(besseli(2*a, x)*besselj(2*a - 1, x)
                    - besseli(2*a - 1, x)*besselj(2*a, x)),
                 x**2*besseli(2*a, x)*besselj(2*a, x),
                 x**3*(besseli(2*a, x)*besselj(2*a - 1, x)
                       + besseli(2*a - 1, x)*besselj(2*a, x))]),
         # 用于 Jacobi 矩阵的单位矩阵
         Matrix([[1, 0, 0, 0]]),
         # 用于 Gegenbauer 矩阵的参数矩阵
         Matrix([[0, Rational(1, 4), 0, 0],
                 [0, (1 - 2*a)/2, Rational(-1, 2), 0],
                 [0, 0, 1 - 2*a, Rational(1, 4)],
                 [-32*z, 0, 0, 1 - a]]))

    # 添加1F2函数计算，传入参数分别为：列表[a]，列表[a - S.Half, 2*a]，以下为函数内部矩阵内容
    addb([a], [a - S.Half, 2*a],
         Matrix([z**(S.Half - a)*besseli(a - S.Half, sqrt(z))**2,
                 z**(1 - a)*besseli(a - S.Half, sqrt(z))
                 *besseli(a - Rational(3, 2), sqrt(z)),
                 z**(Rational(3, 2) - a)*besseli(a - Rational(3, 2), sqrt(z))**2]),
         # 用于 Jacobi 矩阵的参数矩阵
         Matrix([[-gamma(a + S.Half)**2/4**(S.Half - a),
                 2*gamma(a - S.Half)*gamma(a + S.Half)/4**(1 - a),
                 0]]),
         # 用于 Gegenbauer 矩阵的参数矩阵
         Matrix([[1 - 2*a, 1, 0], [z/2, S.Half - a, S.Half], [0, z, 0]]))

    # 添加B函数计算，传入参数分别为：列表[S.Half]，列表[b, 2 - b]，以下为函数内部矩阵内容
    addb([S.Half], [b, 2 - b],
         # 以下为函数内部矩阵内容
         pi*(1 - b)/sin(pi*b)*
         Matrix([besseli(1 - b, sqrt(z))*besseli(b - 1, sqrt(z)),
                 sqrt(z)*(besseli(-b, sqrt(z))*besseli(b - 1, sqrt(z))
                          + besseli(1 - b, sqrt(z))*besseli(b, sqrt(z))),
                 besseli(-b, sqrt(z))*besseli(b, sqrt(z))]),
         # 用于 Jacobi 矩阵的单位矩阵
         Matrix([[1, 0, 0]]),
         # 用于 Gegenbauer 矩阵的参数矩阵
         Matrix([[b - 1, S.Half, 0],
                 [z, 0, z],
                 [0, S.Half, -b]]))

    # 添加B函数计算，传入参数分别为：列表[S.Half]，列表[Rational(3, 2), Rational(3, 2)]，以下为函数内部矩阵内容
    addb([S.Half], [Rational(3, 2), Rational(3, 2)],
         Matrix([Shi(2*sqrt(z))/2/sqrt(z), sinh(2*sqrt(z))/2/sqrt(z),
                 cosh(2*sqrt(z))]),
         # 用于 Jacobi 矩阵的单位矩阵
         Matrix([[1, 0, 0]]),
         # 用于 Gegenbauer 矩阵的参数矩阵
         Matrix([[Rational(-1, 2), S.Half, 0], [0, Rational(-1, 2), S.Half], [0, 2*z, 0]]))

    # 添加 FresnelS 函数计算，传入参数分别为：列表[Rational(3, 4)]，列表[Rational(3, 2), Rational(7, 4)]，以下为函数内部矩阵内容
    addb([Rational(3, 4)], [Rational(3, 2), Rational(7, 4)],
         Matrix(
             [ fresnels(
                 exp(
                     pi*I/4)*root(
                         z, 4)*2/sqrt(
                             pi) ) / (
                                 pi * (exp(pi*I/4)*root(z, 4)*2/sqrt(pi))**3 ),
               sinh(2*sqrt(z))/sqrt(z),
               cosh(2*sqrt(z)) ]),
         # 用于 Jacobi 矩阵的参数矩阵
         Matrix([[6, 0, 0]]),
         # 用于 Gegenbauer 矩阵的参数矩阵
         Matrix([[Rational(-3, 4),  Rational(1, 16), 0],
                 [ 0,      Rational(-1, 2),  1],
                 [ 0,       z,       0]]))
    # 2F3 模型函数，根据给定的参数进行计算
    # 当前的 Formula.find_instantiations 很慢，因为它会创建大约 3000 个实例化
    # 但是总体上来说还算可以接受
    addb([Rational(1, 4)], [S.Half, Rational(5, 4)],
         Matrix(
             [ sqrt(pi)*exp(-I*pi/4)*fresnelc(2*root(z, 4)*exp(I*pi/4)/sqrt(pi))/(2*root(z, 4)),
               cosh(2*sqrt(z)),
               sinh(2*sqrt(z))*sqrt(z) ]),
         Matrix([[1, 0, 0]]),
         Matrix([[Rational(-1, 4),  Rational(1, 4), 0     ],
                 [ 0,       0,      1     ],
                 [ 0,       z,      S.Half]]))

    # 2F3 模型的计算
    addb([a, a + S.Half], [2*a, b, 2*a - b + 1],
         gamma(b)*gamma(2*a - b + 1) * (sqrt(z)/2)**(1 - 2*a) *
         Matrix([besseli(b - 1, sqrt(z))*besseli(2*a - b, sqrt(z)),
                 sqrt(z)*besseli(b, sqrt(z))*besseli(2*a - b, sqrt(z)),
                 sqrt(z)*besseli(b - 1, sqrt(z))*besseli(2*a - b + 1, sqrt(z)),
                 besseli(b, sqrt(z))*besseli(2*a - b + 1, sqrt(z))]),
         Matrix([[1, 0, 0, 0]]),
         Matrix([[0, S.Half, S.Half, 0],
                 [z/2, 1 - b, 0, z/2],
                 [z/2, 0, b - 2*a, z/2],
                 [0, S.Half, S.Half, -2*a]]))

    # 3F3 模型的计算，参考 Wolfram 函数库的规则 https://functions.wolfram.com/07.31.03.0134.01
    # 最初加入的原因是为了处理 integrate(erf(a*z)/z**2, z) 和类似问题的求解
    addb([1, 1, a], [2, 2, a+1],
         Matrix([a*(log(-z) + expint(1, -z) + EulerGamma)/(z*(a**2 - 2*a + 1)),
                 a*(-z)**(-a)*(gamma(a) - uppergamma(a, -z))/(a - 1)**2,
                 a*exp(z)/(a**2 - 2*a + 1),
                 a/(z*(a**2 - 2*a + 1))]),
         Matrix([[1-a, 1, -1/z, 1]]),
         Matrix([[-1,0,-1/z,1],
                 [0,-a,1,0],
                 [0,0,z,0],
                 [0,0,0,-1]]))
# 定义一个函数，用于添加 Meijerg 函数的公式到给定的列表中
def add_meijerg_formulae(formulae):
    # 创建符号变量 a, b, c, z
    a, b, c, z = list(map(Dummy, 'abcz'))
    # 创建符号变量 rho
    rho = Dummy('rho')

    # 定义内部函数 add，用于向 formulae 列表中添加 MeijerFormula 对象
    def add(an, ap, bm, bq, B, C, M, matcher):
        formulae.append(MeijerFormula(an, ap, bm, bq, z, [a, b, c, rho],
                                      B, C, M, matcher))

    # 定义内部函数 detect_uppergamma，用于检测上半伽马函数的特定条件
    def detect_uppergamma(func):
        x = func.an[0]
        y, z = func.bm
        swapped = False
        # 如果条件满足，交换 y 和 z 的值
        if not _mod1((x - y).simplify()):
            swapped = True
            (y, z) = (z, y)
        # 检查 x - z 是否小于等于 0，并返回相应的字典和 G_Function 对象
        if _mod1((x - z).simplify()) or x - z > 0:
            return None
        l = [y, x]
        if swapped:
            l = [x, y]
        return {rho: y, a: x - y}, G_Function([x], [], l, [])

    # 向 formulae 列表中添加一个 Meijerg 函数公式
    add([a + rho], [], [rho, a + rho], [],
        Matrix([gamma(1 - a)*z**rho*exp(z)*uppergamma(a, z),
                gamma(1 - a)*z**(a + rho)]),
        Matrix([[1, 0]]),
        Matrix([[rho + z, -1], [0, a + rho]]),
        detect_uppergamma)

    # 定义内部函数 detect_3113，用于检测特定条件下的 3113 型 Meijerg 函数
    def detect_3113(func):
        """https://functions.wolfram.com/07.34.03.0984.01"""
        x = func.an[0]
        u, v, w = func.bm
        if _mod1((u - v).simplify()) == 0:
            if _mod1((v - w).simplify()) == 0:
                return
            sig = (S.Half, S.Half, S.Zero)
            x1, x2, y = u, v, w
        else:
            if _mod1((x - u).simplify()) == 0:
                sig = (S.Half, S.Zero, S.Half)
                x1, y, x2 = u, v, w
            else:
                sig = (S.Zero, S.Half, S.Half)
                y, x1, x2 = u, v, w

        # 检查特定条件，返回相应的字典和 G_Function 对象
        if (_mod1((x - x1).simplify()) != 0 or
            _mod1((x - x2).simplify()) != 0 or
            _mod1((x - y).simplify()) != S.Half or
                x - x1 > 0 or x - x2 > 0):
            return

        return {a: x}, G_Function([x], [], [x - S.Half + t for t in sig], [])

    # 计算 sin(2*sqrt(z)), cos(2*sqrt(z)), Si(2*sqrt(z)) - pi/2 和 Ci(2*sqrt(z))
    s = sin(2*sqrt(z))
    c_ = cos(2*sqrt(z))
    S_ = Si(2*sqrt(z)) - pi/2
    C = Ci(2*sqrt(z))
    
    # 向 formulae 列表中添加一个特定形式的 Meijerg 函数公式
    add([a], [], [a, a, a - S.Half], [],
        Matrix([sqrt(pi)*z**(a - S.Half)*(c_*S_ - s*C),
                sqrt(pi)*z**a*(s*S_ + c_*C),
                sqrt(pi)*z**a]),
        Matrix([[-2, 0, 0]]),
        Matrix([[a - S.Half, -1, 0], [z, a, S.Half], [0, 0, a]]),
        detect_3113)
    @property
    def args(self):
        """
        Return a tuple containing self.ap and self.bq.

        This property returns a tuple of two attributes, self.ap and self.bq.
        """
        return (self.ap, self.bq)

    @property
    def sizes(self):
        """
        Return a tuple containing the lengths of self.ap and self.bq.

        This property returns a tuple where the first element is the length of
        self.ap and the second element is the length of self.bq.
        """
        return (len(self.ap), len(self.bq))

    @property
    def gamma(self):
        """
        Calculate the count of negative integer upper parameters in self.ap.

        This property counts how many elements in self.ap are negative integers.
        """
        return sum(bool(x.is_integer and x.is_negative) for x in self.ap)

    def _hashable_content(self):
        """
        Return a tuple of hashable content.

        This method returns a tuple containing the hashable content of the
        object, including self.ap and self.bq.
        """
        return super()._hashable_content() + (self.ap, self.bq)

    def __call__(self, arg):
        """
        Call method to compute hypergeometric function.

        This method computes the hypergeometric function using self.ap,
        self.bq, and the argument 'arg'.
        """
        return hyper(self.ap, self.bq, arg)

    def build_invariants(self):
        """
        Compute and return the invariant vector for the hypergeometric function.

        This method computes an invariant vector which includes gamma, and
        parameter distributions mod 1 for self.ap and self.bq.
        """
        # Bucketize self.ap and self.bq based on mod 1 equivalence
        abuckets, bbuckets = sift(self.ap, _mod1), sift(self.bq, _mod1)

        def tr(bucket):
            """
            Transform a bucket into a sorted tuple of (mod, count) pairs.

            This function sorts the bucket and returns a tuple where each
            element is a tuple containing a modulus and the count of values
            in the bucket that match that modulus.
            """
            bucket = list(bucket.items())
            if not any(isinstance(x[0], Mod) for x in bucket):
                bucket.sort(key=lambda x: default_sort_key(x[0]))
            bucket = tuple([(mod, len(values)) for mod, values in bucket if
                    values])
            return bucket

        # Return the invariant vector tuple
        return (self.gamma, tr(abuckets), tr(bbuckets))
    # 评估从当前函数对象到给定函数对象 `func` 的步骤估计，若不可能到达则返回 -1
    def difficulty(self, func):
        """ Estimate how many steps it takes to reach ``func`` from self.
            Return -1 if impossible. """
        # 检查当前函数对象和目标函数对象 `func` 的 gamma 值是否相同
        if self.gamma != func.gamma:
            return -1
        # 对参数进行筛选，生成相应的桶
        oabuckets, obbuckets, abuckets, bbuckets = [sift(params, _mod1) for
                params in (self.ap, self.bq, func.ap, func.bq)]

        # 初始化步数差异为 0
        diff = 0
        # 遍历 a 桶和 b 桶的对应关系
        for bucket, obucket in [(abuckets, oabuckets), (bbuckets, obbuckets)]:
            # 合并并迭代所有模数
            for mod in set(list(bucket.keys()) + list(obucket.keys())):
                # 如果当前模数在其中一个桶中不存在或者两个桶中的列表长度不同，则返回 -1
                if (mod not in bucket) or (mod not in obucket) \
                        or len(bucket[mod]) != len(obucket[mod]):
                    return -1
                # 将桶中的列表元素分别排序
                l1 = list(bucket[mod])
                l2 = list(obucket[mod])
                l1.sort()
                l2.sort()
                # 逐对比较排序后的列表元素，计算它们之间的差值的绝对值并累加到步数差异中
                for i, j in zip(l1, l2):
                    diff += abs(i - j)

        # 返回计算出的步数差异
        return diff

    # 判断当前函数对象是否是一个合适的起始点
    def _is_suitable_origin(self):
        """
        Decide if ``self`` is a suitable origin.

        Explanation
        ===========

        A function is a suitable origin iff:
        * none of the ai equals bj + n, with n a non-negative integer
        * none of the ai is zero
        * none of the bj is a non-positive integer

        Note that this gives meaningful results only when none of the indices
        are symbolic.

        """
        # 检查是否存在 ai 等于 bj + n（n 为非负整数）的情况，若存在则不是合适的起始点
        for a in self.ap:
            for b in self.bq:
                if (a - b).is_integer and (a - b).is_negative is False:
                    return False
        # 检查是否存在 ai 等于 0 的情况，若存在则不是合适的起始点
        for a in self.ap:
            if a == 0:
                return False
        # 检查是否存在 bj 是非正整数的情况，若存在则不是合适的起始点
        for b in self.bq:
            if b.is_integer and b.is_nonpositive:
                return False
        # 若以上条件均不满足，则当前函数对象是一个合适的起始点
        return True
class G_Function(Expr):
    """ A Meijer G-function. """

    def __new__(cls, an, ap, bm, bq):
        # 创建新的实例，调用父类的构造方法
        obj = super().__new__(cls)
        # 将输入的参数转换为元组，并逐个扩展（expand）
        obj.an = Tuple(*list(map(expand, an)))
        obj.ap = Tuple(*list(map(expand, ap)))
        obj.bm = Tuple(*list(map(expand, bm)))
        obj.bq = Tuple(*list(map(expand, bq)))
        return obj

    @property
    def args(self):
        # 返回 G 函数的参数元组
        return (self.an, self.ap, self.bm, self.bq)

    def _hashable_content(self):
        # 返回用于哈希的内容，包括父类的哈希内容和参数元组
        return super()._hashable_content() + self.args

    def __call__(self, z):
        # 调用 meijerg 函数计算 G 函数在 z 处的值
        return meijerg(self.an, self.ap, self.bm, self.bq, z)

    def compute_buckets(self):
        """
        Compute buckets for the fours sets of parameters.

        Explanation
        ===========

        We guarantee that any two equal Mod objects returned are actually the
        same, and that the buckets are sorted by real part (an and bq
        descendending, bm and ap ascending).

        Examples
        ========

        >>> from sympy.simplify.hyperexpand import G_Function
        >>> from sympy.abc import y
        >>> from sympy import S

        >>> a, b = [1, 3, 2, S(3)/2], [1 + y, y, 2, y + 3]
        >>> G_Function(a, b, [2], [y]).compute_buckets()
        ({0: [3, 2, 1], 1/2: [3/2]},
        {0: [2], y: [y, y + 1, y + 3]}, {0: [2]}, {y: [y]})

        """
        # 创建四个空的默认字典
        dicts = pan, pap, pbm, pbq = [defaultdict(list) for i in range(4)]
        # 将参数列表中的每个集合（an, ap, bm, bq）映射到相应的字典中
        for dic, lis in zip(dicts, (self.an, self.ap, self.bm, self.bq)):
            for x in lis:
                dic[_mod1(x)].append(x)

        # 对每个字典中的列表进行排序，按照特定规则排序
        for dic, flip in zip(dicts, (True, False, False, True)):
            for m, items in dic.items():
                x0 = items[0]
                items.sort(key=lambda x: x - x0, reverse=flip)
                dic[m] = items

        # 返回排序后的字典组成的元组
        return tuple([dict(w) for w in dicts])

    @property
    def signature(self):
        # 返回一个元组，包含参数元组的长度（用于标识 G 函数的签名）
        return (len(self.an), len(self.ap), len(self.bm), len(self.bq))


# Dummy variable.
_x = Dummy('x')

class Formula:
    """
    This class represents hypergeometric formulae.

    Explanation
    ===========

    Its data members are:
    - z, the argument
    - closed_form, the closed form expression
    - symbols, the free symbols (parameters) in the formula
    - func, the function
    - B, C, M (see _compute_basis)

    Examples
    ========

    >>> from sympy.abc import a, b, z
    >>> from sympy.simplify.hyperexpand import Formula, Hyper_Function
    >>> func = Hyper_Function((a/2, a/3 + b, (1+a)/2), (a, b, (a+b)/7))
    >>> f = Formula(func, z, None, [a, b])

    """
    def _compute_basis(self, closed_form):
        """
        根据给定的闭式表达式计算函数集合 B=(f1, ..., fn)，一个 nxn 矩阵 M
        和一个 1xn 矩阵 C，使得:
           closed_form = C B
           z d/dz B = M B.
        """
        # 计算函数集合 B 的因子 afactors 和 bfactors
        afactors = [_x + a for a in self.func.ap]
        bfactors = [_x + b - 1 for b in self.func.bq]
        # 构造表达式 expr
        expr = _x * Mul(*bfactors) - self.z * Mul(*afactors)
        # 将表达式转化为多项式对象 poly
        poly = Poly(expr, _x)

        # 计算多项式的次数 n
        n = poly.degree() - 1
        # 初始化列表 b，并将给定的闭式表达式 closed_form 添加到 b 中
        b = [closed_form]
        # 使用 z 对闭式表达式进行求导，迭代计算出 b 中的其余元素
        for _ in range(n):
            b.append(self.z * b[-1].diff(self.z))

        # 将列表 b 转化为一个 Matrix 对象，作为函数集合 B
        self.B = Matrix(b)
        # 初始化矩阵 C，其中第一行为 [1, 0, ..., 0]
        self.C = Matrix([[1] + [0] * n])

        # 构造单位矩阵 m
        m = eye(n)
        # 在 m 的左侧插入一个 n 行 1 列的零矩阵，形成一个 n+1 行 n 列的矩阵
        m = m.col_insert(0, zeros(n, 1))
        # 从多项式 poly 中获取系数列表 l，并将其反转
        l = poly.all_coeffs()[1:]
        l.reverse()
        # 构造矩阵 M，其最后一行为 -l/poly.all_coeffs()[0]
        self.M = m.row_insert(n, -Matrix([l]) / poly.all_coeffs()[0])

    def __init__(self, func, z, res, symbols, B=None, C=None, M=None):
        """
        初始化函数对象，包括设置符号参数，计算函数基 B、矩阵 C 和矩阵 M（可选）。

        参数：
        func: 函数对象
        z: 符号变量 z
        res: 结果表达式
        symbols: 符号列表
        B: 函数基 B（可选）
        C: 矩阵 C（可选）
        M: 矩阵 M（可选）
        """
        # 将 z、res 和 symbols 转化为符号对象
        z = sympify(z)
        res = sympify(res)
        symbols = [x for x in sympify(symbols) if func.has(x)]

        # 设置对象的属性值
        self.z = z
        self.symbols = symbols
        self.B = B
        self.C = C
        self.M = M
        self.func = func

        # 如果 res 不为空，则调用 _compute_basis 方法计算函数基 B、矩阵 C 和矩阵 M
        if res is not None:
            self._compute_basis(res)

    @property
    def closed_form(self):
        """
        计算并返回闭式表达式，使用矩阵 C 和函数基 B 的组合求和。
        """
        return reduce(lambda s, m: s + m[0] * m[1], zip(self.C, self.B), S.Zero)
    def find_instantiations(self, func):
        """
        找到与 `func` 匹配的自由符号的替代项。

        返回替代字典的列表。注意，返回的实例化可能不匹配或无效！

        """
        from sympy.solvers import solve
        # 获取 func 的 ap 和 bq 属性
        ap = func.ap
        bq = func.bq
        # 检查参数数量是否匹配
        if len(ap) != len(self.func.ap) or len(bq) != len(self.func.bq):
            raise TypeError('Cannot instantiate other number of parameters')
        # 存储符号的值
        symbol_values = []
        # 遍历当前对象的符号
        for a in self.symbols:
            # 根据符号所属的参数类型添加对应的值
            if a in self.func.ap.args:
                symbol_values.append(ap)
            elif a in self.func.bq.args:
                symbol_values.append(bq)
            else:
                # 如果符号不属于 ap 或 bq，则引发错误
                raise ValueError("At least one of the parameters of the "
                        "formula must be equal to %s" % (a,))
        # 基本替换字典，用于生成所有可能的组合
        base_repl = [dict(list(zip(self.symbols, values)))
                for values in product(*symbol_values)]
        # 根据参数 ap 和 bq 进行分桶
        abuckets, bbuckets = [sift(params, _mod1) for params in [ap, bq]]
        # 计算每个桶的长度
        a_inv, b_inv = [{a: len(vals) for a, vals in bucket.items()}
                for bucket in [abuckets, bbuckets]]
        # 初始化关键值列表
        critical_values = [[0] for _ in self.symbols]
        # 存储最终结果的列表
        result = []
        # 创建虚拟变量
        _n = Dummy()
        # 遍历所有基本替换字典
        for repl in base_repl:
            # 获取符号在 ap 和 bq 中的分桶
            symb_a, symb_b = [sift(params, lambda x: _mod1(x.xreplace(repl)))
                for params in [self.func.ap, self.func.bq]]
            # 比较两个分桶中的元素
            for bucket, obucket in [(abuckets, symb_a), (bbuckets, symb_b)]:
                # 遍历所有模块并比较
                for mod in set(list(bucket.keys()) + list(obucket.keys())):
                    # 如果当前模块不在任一分桶中或长度不匹配，则中断
                    if (mod not in bucket) or (mod not in obucket) \
                            or len(bucket[mod]) != len(obucket[mod]):
                        break
                    # 遍历所有符号和关键值
                    for a, vals in zip(self.symbols, critical_values):
                        # 如果替换后的符号包含自由符号，则继续下一轮循环
                        if repl[a].free_symbols:
                            continue
                        # 从 obucket 中找到所有包含当前符号的表达式
                        exprs = [expr for expr in obucket[mod] if expr.has(a)]
                        # 复制替换字典并进行更新
                        repl0 = repl.copy()
                        repl0[a] += _n
                        # 遍历 obucket 中的表达式并求解
                        for expr in exprs:
                            for target in bucket[mod]:
                                # 求解方程 expr - target = 0，找到 _n 的值
                                n0, = solve(expr.xreplace(repl0) - target, _n)
                                # 如果求解出的值包含自由符号，则引发错误
                                if n0.free_symbols:
                                    raise ValueError("Value should not be true")
                                # 将求解出的值添加到关键值列表中
                                vals.append(n0)
            else:
                # 生成所有可能的值组合
                values = []
                for a, vals in zip(self.symbols, critical_values):
                    a0 = repl[a]
                    min_ = floor(min(vals))
                    max_ = ceiling(max(vals))
                    values.append([a0 + n for n in range(min_, max_ + 1)])
                # 将当前组合添加到结果列表中
                result.extend(dict(list(zip(self.symbols, l))) for l in product(*values))
        # 返回最终的结果列表
        return result
# 定义一个 FormulaCollection 类，用于存储和查找公式的集合
class FormulaCollection:
    """ A collection of formulae to use as origins. """

    # 初始化方法，设置空字典和空列表来存储符号和具体公式，然后添加一些预定义的公式
    def __init__(self):
        """ Doing this globally at module init time is a pain ... """
        self.symbolic_formulae = {}  # 存储符号公式的字典
        self.concrete_formulae = {}  # 存储具体公式的字典
        self.formulae = []  # 存储所有公式的列表

        add_formulae(self.formulae)  # 调用函数将公式添加到 formulae 列表中

        # 将 formulae 列表中的公式按照它们的 sizes 属性分类存储到 symbolic_formulae 或 concrete_formulae 字典中
        for f in self.formulae:
            sizes = f.func.sizes
            if len(f.symbols) > 0:
                self.symbolic_formulae.setdefault(sizes, []).append(f)
            else:
                inv = f.func.build_invariants()
                self.concrete_formulae.setdefault(sizes, {})[inv] = f

    # 查找给定 func 的起源公式
    def lookup_origin(self, func):
        """
        Given the suitable target ``func``, try to find an origin in our
        knowledge base.

        Examples
        ========

        >>> from sympy.simplify.hyperexpand import (FormulaCollection,
        ...     Hyper_Function)
        >>> f = FormulaCollection()
        >>> f.lookup_origin(Hyper_Function((), ())).closed_form
        exp(_z)
        >>> f.lookup_origin(Hyper_Function([1], ())).closed_form
        HyperRep_power1(-1, _z)

        >>> from sympy import S
        >>> i = Hyper_Function([S('1/4'), S('3/4 + 4')], [S.Half])
        >>> f.lookup_origin(i).closed_form
        HyperRep_sqrts1(-1/4, _z)
        """
        inv = func.build_invariants()  # 获取 func 的不变量
        sizes = func.sizes  # 获取 func 的大小

        # 如果在 concrete_formulae 中找到符合条件的公式，返回该公式
        if sizes in self.concrete_formulae and inv in self.concrete_formulae[sizes]:
            return self.concrete_formulae[sizes][inv]

        # 如果没有具体公式，尝试实例化
        if sizes not in self.symbolic_formulae:
            return None  # 没有找到适合的公式

        possible = []
        # 遍历 symbolic_formulae 中符合 sizes 的公式，找到可以实例化的替换集合
        for f in self.symbolic_formulae[sizes]:
            repls = f.find_instantiations(func)
            for repl in repls:
                func2 = f.func.xreplace(repl)
                if not func2._is_suitable_origin():
                    continue
                diff = func2.difficulty(func)
                if diff == -1:
                    continue
                possible.append((diff, repl, f, func2))

        # 按难度排序可能的起源公式
        possible.sort(key=lambda x: x[0])
        for _, repl, f, func2 in possible:
            # 构造一个新的 Formula 对象 f2，用于返回一个合适的起源公式
            f2 = Formula(func2, f.z, None, [], f.B.subs(repl),
                    f.C.subs(repl), f.M.subs(repl))
            # 检查公式中是否包含无效数值
            if not any(e.has(S.NaN, oo, -oo, zoo) for e in [f2.B, f2.M, f2.C]):
                return f2

        return None  # 没有找到适合的起源公式


# MeijerFormula 类，表示一个 Meijer G-函数公式
class MeijerFormula:
    """
    This class represents a Meijer G-function formula.

    Its data members are:
    - z, the argument
    - symbols, the free symbols (parameters) in the formula
    - func, the function
    - B, C, M (c/f ordinary Formula)
    """
    def __init__(self, an, ap, bm, bq, z, symbols, B, C, M, matcher):
        # 将输入的参数进行扩展并存储为元组类型
        an, ap, bm, bq = [Tuple(*list(map(expand, w))) for w in [an, ap, bm, bq]]
        # 使用扩展后的参数初始化 G_Function 对象
        self.func = G_Function(an, ap, bm, bq)
        # 存储参数 z
        self.z = z
        # 存储 symbols
        self.symbols = symbols
        # 存储匹配器对象
        self._matcher = matcher
        # 存储参数 B
        self.B = B
        # 存储参数 C
        self.C = C
        # 存储参数 M
        self.M = M

    @property
    def closed_form(self):
        # 计算并返回表达式的闭合形式
        return reduce(lambda s,m: s+m[0]*m[1], zip(self.C, self.B), S.Zero)

    def try_instantiate(self, func):
        """
        尝试将当前公式实例化为（几乎）与给定 func 匹配的形式。
        使用在初始化时传递的 _matcher 进行匹配。
        """
        # 如果给定函数的签名与当前函数对象的签名不匹配，则返回 None
        if func.signature != self.func.signature:
            return None
        # 使用匹配器对象尝试匹配 func
        res = self._matcher(func)
        # 如果匹配成功，则返回替换后的参数及新的函数对象
        if res is not None:
            subs, newfunc = res
            return MeijerFormula(newfunc.an, newfunc.ap, newfunc.bm, newfunc.bq,
                                 self.z, [],
                                 # 替换参数 B 中的变量
                                 self.B.subs(subs),
                                 # 替换参数 C 中的变量
                                 self.C.subs(subs),
                                 # 替换参数 M 中的变量
                                 self.M.subs(subs),
                                 None)
class MeijerFormulaCollection:
    """
    This class holds a collection of meijer g formulae.
    """

    def __init__(self):
        # 初始化一个空列表来存储 Meijer G 函数的公式
        formulae = []
        # 调用外部函数 add_meijerg_formulae 将公式添加到 formulae 列表中
        add_meijerg_formulae(formulae)
        # 使用 defaultdict 来组织 formulae 列表中的公式，按照签名分组
        self.formulae = defaultdict(list)
        for formula in formulae:
            self.formulae[formula.func.signature].append(formula)
        # 将 defaultdict 转换为普通的字典
        self.formulae = dict(self.formulae)

    def lookup_origin(self, func):
        """ Try to find a formula that matches func. """
        # 如果 func 的签名不在 formulae 字典中，则返回 None
        if func.signature not in self.formulae:
            return None
        # 遍历符合签名的公式列表，尝试实例化 func，返回匹配的结果
        for formula in self.formulae[func.signature]:
            res = formula.try_instantiate(func)
            if res is not None:
                return res


class Operator:
    """
    Base class for operators to be applied to our functions.

    Explanation
    ===========

    These operators are differential operators. They are by convention
    expressed in the variable D = z*d/dz (although this base class does
    not actually care).
    Note that when the operator is applied to an object, we typically do
    *not* blindly differentiate but instead use a different representation
    of the z*d/dz operator (see make_derivative_operator).

    To subclass from this, define a __init__ method that initializes a
    self._poly variable. This variable stores a polynomial. By convention
    the generator is z*d/dz, and acts to the right of all coefficients.

    Thus this poly
        x**2 + 2*z*x + 1
    represents the differential operator
        (z*d/dz)**2 + 2*z**2*d/dz.

    This class is used only in the implementation of the hypergeometric
    function expansion algorithm.
    """

    def apply(self, obj, op):
        """
        Apply ``self`` to the object ``obj``, where the generator is ``op``.

        Examples
        ========

        >>> from sympy.simplify.hyperexpand import Operator
        >>> from sympy.polys.polytools import Poly
        >>> from sympy.abc import x, y, z
        >>> op = Operator()
        >>> op._poly = Poly(x**2 + z*x + y, x)
        >>> op.apply(z**7, lambda f: f.diff(z))
        y*z**7 + 7*z**7 + 42*z**5
        """
        # 获取多项式的所有系数并反转顺序
        coeffs = self._poly.all_coeffs()
        coeffs.reverse()
        # 初始化一个列表，包含原始对象 obj
        diffs = [obj]
        # 对于多项式的每一个系数（除了第一个），依次应用 op 操作到 diffs 列表的最后一个元素
        for c in coeffs[1:]:
            diffs.append(op(diffs[-1]))
        # 初始化结果为多项式的第一个系数乘以 diffs 的第一个元素
        r = coeffs[0]*diffs[0]
        # 对于多项式的每一个系数和 diffs 的每一个元素，依次相乘并累加到结果 r 中
        for c, d in zip(coeffs[1:], diffs[1:]):
            r += c*d
        return r


class MultOperator(Operator):
    """ Simply multiply by a "constant" """

    def __init__(self, p):
        # 将常数 p 初始化为多项式的系数
        self._poly = Poly(p, _x)


class ShiftA(Operator):
    """ Increment an upper index. """

    def __init__(self, ai):
        # 将 ai 转换为符号表达式
        ai = sympify(ai)
        # 如果 ai 等于 0，则引发 ValueError 异常
        if ai == 0:
            raise ValueError('Cannot increment zero upper index.')
        # 将 _poly 初始化为多项式 (_x/ai + 1)
        self._poly = Poly(_x/ai + 1, _x)

    def __str__(self):
        # 返回字符串描述，表示增加上标
        return '<Increment upper %s.>' % (1/self._poly.all_coeffs()[0])


class ShiftB(Operator):
    """ Decrement a lower index. """
    # 初始化方法，接受一个参数 bi
    def __init__(self, bi):
        # 使用 sympify 函数将参数 bi 转换成符号表达式
        bi = sympify(bi)
        # 检查 bi 是否等于 1，如果是则抛出值错误异常
        if bi == 1:
            raise ValueError('Cannot decrement unit lower index.')
        # 创建一个多项式对象 _poly，计算公式为 _x/(bi - 1) + 1
        self._poly = Poly(_x/(bi - 1) + 1, _x)
    
    # 字符串表示方法，返回对象的字符串表示
    def __str__(self):
        # 返回类的字符串表示，其中包含一个计算结果，1/self._poly.all_coeffs()[0] + 1
        return '<Decrement lower %s.>' % (1/self._poly.all_coeffs()[0] + 1)
class UnShiftA(Operator):
    """ Decrement an upper index. """

    def __init__(self, ap, bq, i, z):
        """ Note: i counts from zero! """
        # 将输入参数转换为符号表达式
        ap, bq, i = list(map(sympify, [ap, bq, i]))

        # 初始化对象属性
        self._ap = ap
        self._bq = bq
        self._i = i

        # 将 ap 和 bq 转换为列表
        ap = list(ap)
        bq = list(bq)

        # 弹出索引为 i 的元素，并减一
        ai = ap.pop(i) - 1

        # 如果减一后的值为零，则抛出异常
        if ai == 0:
            raise ValueError('Cannot decrement unit upper index.')

        # 构建多项式 m
        m = Poly(z*ai, _x)
        for a in ap:
            m *= Poly(_x + a, _x)

        # 创建虚拟符号 A
        A = Dummy('A')
        n = D = Poly(ai*A - ai, A)
        for b in bq:
            n *= D + (b - 1).as_poly(A)

        # 计算 n 的常数项
        b0 = -n.nth(0)
        # 如果常数项为零，则抛出异常
        if b0 == 0:
            raise ValueError('Cannot decrement upper index: '
                             'cancels with lower')

        # 对 n 进行变换，再创建多项式对象
        n = Poly(Poly(n.all_coeffs()[:-1], A).as_expr().subs(A, _x/ai + 1), _x)

        # 计算并设置对象属性 _poly
        self._poly = Poly((n - m)/b0, _x)

    def __str__(self):
        return '<Decrement upper index #%s of %s, %s.>' % (self._i,
                                                        self._ap, self._bq)


class UnShiftB(Operator):
    """ Increment a lower index. """

    def __init__(self, ap, bq, i, z):
        """ Note: i counts from zero! """
        # 将输入参数转换为符号表达式
        ap, bq, i = list(map(sympify, [ap, bq, i]))

        # 初始化对象属性
        self._ap = ap
        self._bq = bq
        self._i = i

        # 将 ap 和 bq 转换为列表
        ap = list(ap)
        bq = list(bq)

        # 弹出索引为 i 的元素，并加一
        bi = bq.pop(i) + 1

        # 如果加一后的值为零，则抛出异常
        if bi == 0:
            raise ValueError('Cannot increment -1 lower index.')

        # 构建多项式 m
        m = Poly(_x*(bi - 1), _x)
        for b in bq:
            m *= Poly(_x + b - 1, _x)

        # 创建虚拟符号 B
        B = Dummy('B')
        D = Poly((bi - 1)*B - bi + 1, B)
        n = Poly(z, B)
        for a in ap:
            n *= (D + a.as_poly(B))

        # 计算 n 的常数项
        b0 = n.nth(0)
        # 如果常数项为零，则抛出异常
        if b0 == 0:
            raise ValueError('Cannot increment index: cancels with upper')

        # 对 n 进行变换，再创建多项式对象
        n = Poly(Poly(n.all_coeffs()[:-1], B).as_expr().subs(
            B, _x/(bi - 1) + 1), _x)

        # 计算并设置对象属性 _poly
        self._poly = Poly((m - n)/b0, _x)

    def __str__(self):
        return '<Increment lower index #%s of %s, %s.>' % (self._i,
                                                        self._ap, self._bq)


class MeijerShiftA(Operator):
    """ Increment an upper b index. """

    def __init__(self, bi):
        # 将输入参数转换为符号表达式
        bi = sympify(bi)
        # 创建多项式对象 _poly
        self._poly = Poly(bi - _x, _x)

    def __str__(self):
        return '<Increment upper b=%s.>' % (self._poly.all_coeffs()[1])


class MeijerShiftB(Operator):
    """ Decrement an upper a index. """

    def __init__(self, bi):
        # 将输入参数转换为符号表达式
        bi = sympify(bi)
        # 创建多项式对象 _poly
        self._poly = Poly(1 - bi + _x, _x)

    def __str__(self):
        return '<Decrement upper a=%s.>' % (1 - self._poly.all_coeffs()[1])


class MeijerShiftC(Operator):
    """ Increment a lower b index. """

    def __init__(self, bi):
        # 将输入参数转换为符号表达式
        bi = sympify(bi)
        # 创建多项式对象 _poly
        self._poly = Poly(-bi + _x, _x)

    def __str__(self):
        return '<Increment lower b=%s.>' % (-self._poly.all_coeffs()[1])


class MeijerShiftD(Operator):
    """ Decrement a lower a index. """

    # 初始化方法，接受参数 bi，并将其转换为 sympy 的表达式
    def __init__(self, bi):
        bi = sympify(bi)
        # 创建一个多项式对象 _poly，其表达式为 bi - 1 - _x，其中 _x 是一个符号变量
        self._poly = Poly(bi - 1 - _x, _x)

    # 返回对象的字符串表示形式
    def __str__(self):
        # 返回形如 '<Decrement lower a=<a+1>.>' 的字符串，其中 a 的值来自 _poly 的第二个系数加 1
        return '<Decrement lower a=%s.>' % (self._poly.all_coeffs()[1] + 1)
class MeijerUnShiftA(Operator):
    """ Decrement an upper b index. """

    def __init__(self, an, ap, bm, bq, i, z):
        """ Note: i counts from zero! """
        # 将输入的符号表达式列表化
        an, ap, bm, bq, i = list(map(sympify, [an, ap, bm, bq, i]))

        # 设置对象的内部变量
        self._an = an
        self._ap = ap
        self._bm = bm
        self._bq = bq
        self._i = i

        # 将列表转为可修改的列表
        an = list(an)
        ap = list(ap)
        bm = list(bm)
        bq = list(bq)
        # 弹出 bm 中索引为 i 的元素并将其减一
        bi = bm.pop(i) - 1

        # 构造多项式 m
        m = Poly(1, _x) * prod(Poly(b - _x, _x) for b in bm) * prod(Poly(_x - b, _x) for b in bq)

        # 创建符号变量 A
        A = Dummy('A')
        # 创建多项式 D
        D = Poly(bi - A, A)
        # 构造多项式 n
        n = Poly(z, A) * prod((D + 1 - a) for a in an) * prod((-D + a - 1) for a in ap)

        # 获取 n 的常数项
        b0 = n.nth(0)
        # 如果常数项为 0，则抛出 ValueError
        if b0 == 0:
            raise ValueError('Cannot decrement upper b index (cancels)')

        # 将 n 的表达式化简并替换 A 为 bi - _x
        n = Poly(Poly(n.all_coeffs()[:-1], A).as_expr().subs(A, bi - _x), _x)

        # 计算并设置对象的 _poly 属性
        self._poly = Poly((m - n)/b0, _x)

    def __str__(self):
        # 返回对象的字符串表示形式
        return '<Decrement upper b index #%s of %s, %s, %s, %s.>' % (self._i,
                                      self._an, self._ap, self._bm, self._bq)


class MeijerUnShiftB(Operator):
    """ Increment an upper a index. """

    def __init__(self, an, ap, bm, bq, i, z):
        """ Note: i counts from zero! """
        # 将输入的符号表达式列表化
        an, ap, bm, bq, i = list(map(sympify, [an, ap, bm, bq, i]))

        # 设置对象的内部变量
        self._an = an
        self._ap = ap
        self._bm = bm
        self._bq = bq
        self._i = i

        # 将列表转为可修改的列表
        an = list(an)
        ap = list(ap)
        bm = list(bm)
        bq = list(bq)
        # 弹出 an 中索引为 i 的元素并将其加一
        ai = an.pop(i) + 1

        # 构造多项式 m
        m = Poly(z, _x)
        for a in an:
            m *= Poly(1 - a + _x, _x)
        for a in ap:
            m *= Poly(a - 1 - _x, _x)

        # 创建符号变量 B
        B = Dummy('B')
        # 创建多项式 D
        D = Poly(B + ai - 1, B)
        # 构造多项式 n
        n = Poly(1, B)
        for b in bm:
            n *= (-D + b)
        for b in bq:
            n *= (D - b)

        # 获取 n 的常数项
        b0 = n.nth(0)
        # 如果常数项为 0，则抛出 ValueError
        if b0 == 0:
            raise ValueError('Cannot increment upper a index (cancels)')

        # 将 n 的表达式化简并替换 B 为 1 - ai + _x
        n = Poly(Poly(n.all_coeffs()[:-1], B).as_expr().subs(
            B, 1 - ai + _x), _x)

        # 计算并设置对象的 _poly 属性
        self._poly = Poly((m - n)/b0, _x)

    def __str__(self):
        # 返回对象的字符串表示形式
        return '<Increment upper a index #%s of %s, %s, %s, %s.>' % (self._i,
                                      self._an, self._ap, self._bm, self._bq)


class MeijerUnShiftC(Operator):
    """ Decrement a lower b index. """
    # XXX this is "essentially" the same as MeijerUnShiftA. This "essentially"
    #     can be made rigorous using the functional equation G(1/z) = G'(z),
    #     where G' denotes a G function of slightly altered parameters.
    #     However, sorting out the details seems harder than just coding it
    #     again.
    def __init__(self, an, ap, bm, bq, i, z):
        """ Note: i counts from zero! """
        # 将输入的参数转换为符号表达式
        an, ap, bm, bq, i = list(map(sympify, [an, ap, bm, bq, i]))

        # 设置对象的属性
        self._an = an
        self._ap = ap
        self._bm = bm
        self._bq = bq
        self._i = i

        # 将 bq 列表中第 i 个元素取出并减 1
        bi = bq.pop(i) - 1

        # 初始化多项式 m 为常数 1
        m = Poly(1, _x)
        # 将 bm 列表中的每个元素与 (x - _x) 的多项式相乘，构建 m
        for b in bm:
            m *= Poly(b - _x, _x)
        # 将 bq 列表中的每个元素与 (_x - b) 的多项式相乘，扩展 m
        for b in bq:
            m *= Poly(_x - b, _x)

        # 创建虚拟变量 C 和多项式 D = bi + C
        C = Dummy('C')
        D = Poly(bi + C, C)

        # 创建多项式 n = z
        n = Poly(z, C)
        # 将 an 列表中的每个元素与 (D + 1 - a) 相乘，扩展 n
        for a in an:
            n *= (D + 1 - a)
        # 将 ap 列表中的每个元素与 (-D + a - 1) 相乘，扩展 n
        for a in ap:
            n *= (-D + a - 1)

        # 获取 n 中的常数项
        b0 = n.nth(0)
        # 如果常数项为零，则抛出值错误异常
        if b0 == 0:
            raise ValueError('Cannot decrement lower b index (cancels)')

        # 将 n 的表达式部分除去最高次幂的系数，用 _x - bi 替换 C，得到新的多项式 n
        n = Poly(Poly(n.all_coeffs()[:-1], C).as_expr().subs(C, _x - bi), _x)

        # 计算并设置对象的属性 _poly，为 (m - n) / b0 的多项式
        self._poly = Poly((m - n) / b0, _x)

    def __str__(self):
        # 返回对象的字符串表示形式，包括对象的一些属性信息
        return '<Decrement lower b index #%s of %s, %s, %s, %s.>' % (self._i,
                                      self._an, self._ap, self._bm, self._bq)
class MeijerUnShiftD(Operator):
    """ Increment a lower a index. """
    # XXX This is essentially the same as MeijerUnShiftA.
    #     See comment at MeijerUnShiftC.

    def __init__(self, an, ap, bm, bq, i, z):
        """ Note: i counts from zero! """
        # 将输入的参数映射为sympy的符号表达式
        an, ap, bm, bq, i = list(map(sympify, [an, ap, bm, bq, i]))

        # 设置对象的内部变量
        self._an = an
        self._ap = ap
        self._bm = bm
        self._bq = bq
        self._i = i

        # 将ap中的第i个元素加一，得到ai
        ai = ap.pop(i) + 1

        # 创建一个多项式m，并初始化为z关于_x的多项式
        m = Poly(z, _x)
        for a in an:
            m *= Poly(1 - a + _x, _x)
        for a in ap:
            m *= Poly(a - 1 - _x, _x)

        # 创建一个虚拟变量B，表示移位操作符`D_I`
        B = Dummy('B')
        D = Poly(ai - 1 - B, B)
        n = Poly(1, B)
        for b in bm:
            n *= (-D + b)
        for b in bq:
            n *= (D - b)

        # 检查n的常数项是否为零，如果是则抛出异常
        b0 = n.nth(0)
        if b0 == 0:
            raise ValueError('Cannot increment lower a index (cancels)')

        # 将n中除去常数项的部分表示为关于B的表达式，再用ai - 1 - _x替换B，得到多项式n关于_x的表达式
        n = Poly(Poly(n.all_coeffs()[:-1], B).as_expr().subs(
            B, ai - 1 - _x), _x)

        # 计算最终的多项式_poly并存储在对象中
        self._poly = Poly((m - n)/b0, _x)

    def __str__(self):
        return '<Increment lower a index #%s of %s, %s, %s, %s.>' % (self._i,
                                      self._an, self._ap, self._bm, self._bq)


class ReduceOrder(Operator):
    """ Reduce Order by cancelling an upper and a lower index. """

    def __new__(cls, ai, bj):
        """ For convenience if reduction is not possible, return None. """
        # 将输入的参数映射为sympy的符号表达式
        ai = sympify(ai)
        bj = sympify(bj)
        n = ai - bj

        # 如果n不是整数或者小于零，则返回None
        if not n.is_Integer or n < 0:
            return None
        # 如果bj是整数且非正数，则返回None
        if bj.is_integer and bj.is_nonpositive:
            return None

        expr = Operator.__new__(cls)

        # 计算约简后的多项式p
        p = S.One
        for k in range(n):
            p *= (_x + bj + k)/(bj + k)

        # 将计算得到的多项式p存储在对象中
        expr._poly = Poly(p, _x)
        expr._a = ai
        expr._b = bj

        return expr

    @classmethod
    def _meijer(cls, b, a, sign):
        """ Cancel b + sign*s and a + sign*s
            This is for meijer G functions. """
        # 将输入的参数映射为sympy的符号表达式
        b = sympify(b)
        a = sympify(a)
        n = b - a

        # 如果n为负数或者不是整数，则返回None
        if n.is_negative or not n.is_Integer:
            return None

        expr = Operator.__new__(cls)

        # 计算约简后的多项式p
        p = S.One
        for k in range(n):
            p *= (sign*_x + a + k)

        # 将计算得到的多项式p存储在对象中
        expr._poly = Poly(p, _x)
        if sign == -1:
            expr._a = b
            expr._b = a
        else:
            expr._b = Add(1, a - 1, evaluate=False)
            expr._a = Add(1, b - 1, evaluate=False)

        return expr

    @classmethod
    def meijer_minus(cls, b, a):
        return cls._meijer(b, a, -1)

    @classmethod
    def meijer_plus(cls, a, b):
        return cls._meijer(1 - a, 1 - b, 1)

    def __str__(self):
        return '<Reduce order by cancelling upper %s with lower %s.>' % \
            (self._a, self._b)
    """ Order reduction algorithm used in Hypergeometric and Meijer G """
    # 将输入的 ap 和 bq 转换为列表（如果它们不已经是列表的话）
    ap = list(ap)
    bq = list(bq)

    # 使用给定的 key 函数对 ap 和 bq 列表进行排序
    ap.sort(key=key)
    bq.sort(key=key)

    # 初始化一个空列表 nap 来存储未匹配的 ap 元素
    nap = []
    # 初始化一个空列表 operators 来存储匹配到的操作符
    operators = []

    # 遍历 ap 列表中的每个元素 a
    for a in ap:
        # 初始化操作符为 None
        op = None
        # 遍历 bq 列表中的每个元素（我们将在此处修改 bq）
        for i in range(len(bq)):
            # 使用 gen 函数尝试生成操作符 op
            op = gen(a, bq[i])
            # 如果成功生成了操作符 op
            if op is not None:
                # 从 bq 中移除第 i 个元素
                bq.pop(i)
                # 退出内部循环
                break
        # 如果未成功生成操作符 op
        if op is None:
            # 将 a 添加到 nap 列表中，表示未匹配成功
            nap.append(a)
        else:
            # 将成功生成的操作符 op 添加到 operators 列表中
            operators.append(op)

    # 返回三个结果：未匹配的 ap 元素列表 nap，剩余的未处理的 bq 列表和生成的操作符列表 operators
    return nap, bq, operators
# 定义函数 reduce_order，用于降低超几何函数的阶数
def reduce_order(func):
    """
    Given the hypergeometric function ``func``, find a sequence of operators to
    reduces order as much as possible.
    
    Explanation
    ===========
    
    Return (newfunc, [operators]), where applying the operators to the
    hypergeometric function newfunc yields func.
    
    Examples
    ========
    
    >>> from sympy.simplify.hyperexpand import reduce_order, Hyper_Function
    >>> reduce_order(Hyper_Function((1, 2), (3, 4)))
    (Hyper_Function((1, 2), (3, 4)), [])
    >>> reduce_order(Hyper_Function((1,), (1,)))
    (Hyper_Function((), ()), [<Reduce order by cancelling upper 1 with lower 1.>])
    >>> reduce_order(Hyper_Function((2, 4), (3, 3)))
    (Hyper_Function((2,), (3,)), [<Reduce order by cancelling
    upper 4 with lower 3.>])
    """
    # 调用 _reduce_order 函数，计算新的超几何函数参数和操作符序列
    nap, nbq, operators = _reduce_order(func.ap, func.bq, ReduceOrder, default_sort_key)
    
    # 返回降阶后的新超几何函数和操作符序列
    return Hyper_Function(Tuple(*nap), Tuple(*nbq)), operators


# 定义函数 reduce_order_meijer，用于降低 Meijer G 函数的阶数
def reduce_order_meijer(func):
    """
    Given the Meijer G function parameters, ``func``, find a sequence of
    operators that reduces order as much as possible.
    
    Return newfunc, [operators].
    
    Examples
    ========
    
    >>> from sympy.simplify.hyperexpand import (reduce_order_meijer,
    ...                                         G_Function)
    >>> reduce_order_meijer(G_Function([3, 4], [5, 6], [3, 4], [1, 2]))[0]
    G_Function((4, 3), (5, 6), (3, 4), (2, 1))
    >>> reduce_order_meijer(G_Function([3, 4], [5, 6], [3, 4], [1, 8]))[0]
    G_Function((3,), (5, 6), (3, 4), (1,))
    >>> reduce_order_meijer(G_Function([3, 4], [5, 6], [7, 5], [1, 5]))[0]
    G_Function((3,), (), (), (1,))
    >>> reduce_order_meijer(G_Function([3, 4], [5, 6], [7, 5], [5, 3]))[0]
    G_Function((), (), (), ())
    """
    # 调用 _reduce_order 函数，分别处理参数 an 和 bq，以及 bm 和 ap
    nan, nbq, ops1 = _reduce_order(func.an, func.bq, ReduceOrder.meijer_plus,
                                   lambda x: default_sort_key(-x))
    nbm, nap, ops2 = _reduce_order(func.bm, func.ap, ReduceOrder.meijer_minus,
                                   default_sort_key)
    
    # 返回降阶后的新 Meijer G 函数和操作符序列
    return G_Function(nan, nap, nbm, nbq), ops1 + ops2


# 定义函数 make_derivative_operator，创建一个导数算子函数
def make_derivative_operator(M, z):
    """ Create a derivative operator, to be passed to Operator.apply. """
    # 内部函数 doit，接受一个参数 C，计算导数算子的作用结果
    def doit(C):
        r = z*C.diff(z) + C*M
        r = r.applyfunc(make_simp(z))
        return r
    return doit


# 定义函数 apply_operators，将一系列操作符应用于对象 obj
def apply_operators(obj, ops, op):
    """
    Apply the list of operators ``ops`` to object ``obj``, substituting
    ``op`` for the generator.
    """
    # 初始化结果为原始对象 obj
    res = obj
    # 逆序遍历操作符列表 ops，并依次应用到 res 上
    for o in reversed(ops):
        res = o.apply(res, op)
    # 返回应用操作符后的结果
    return res


# 定义函数 devise_plan，设计一个计划用于将目标超几何函数转换为原始超几何函数
def devise_plan(target, origin, z):
    """
    Devise a plan (consisting of shift and un-shift operators) to be applied
    to the hypergeometric function ``target`` to yield ``origin``.
    Returns a list of operators.
    
    Examples
    ========
    
    >>> from sympy.simplify.hyperexpand import devise_plan, Hyper_Function
    >>> from sympy.abc import z
    
    Nothing to do:
    """
    abuckets, bbuckets, nabuckets, nbbuckets = [sift(params, _mod1) for
            params in (target.ap, target.bq, origin.ap, origin.bq)]


    # 使用 sift 函数分别处理目标和原点的参数，返回四个不同的桶列表
    abuckets, bbuckets, nabuckets, nbbuckets = [sift(params, _mod1) for
            params in (target.ap, target.bq, origin.ap, origin.bq)]



    if len(list(abuckets.keys())) != len(list(nabuckets.keys())) or \
            len(list(bbuckets.keys())) != len(list(nbbuckets.keys())):
        raise ValueError('%s not reachable from %s' % (target, origin))


    # 检查目标和原点的桶数量是否相同，如果不同则抛出 ValueError 异常
    if len(list(abuckets.keys())) != len(list(nabuckets.keys())) or \
            len(list(bbuckets.keys())) != len(list(nbbuckets.keys())):
        raise ValueError('%s not reachable from %s' % (target, origin))



    ops = []


    # 初始化操作列表
    ops = []



    def do_shifts(fro, to, inc, dec):
        ops = []
        for i in range(len(fro)):
            if to[i] - fro[i] > 0:
                sh = inc
                ch = 1
            else:
                sh = dec
                ch = -1

            while to[i] != fro[i]:
                ops += [sh(fro, i)]
                fro[i] += ch

        return ops


    # 定义一个函数 do_shifts，用于将 fro 变量的值转移到 to 变量的值
    def do_shifts(fro, to, inc, dec):
        ops = []  # 初始化操作列表
        for i in range(len(fro)):
            if to[i] - fro[i] > 0:
                sh = inc  # 如果 to[i] 大于 fro[i]，则使用 inc 函数
                ch = 1
            else:
                sh = dec  # 否则使用 dec 函数
                ch = -1

            while to[i] != fro[i]:
                ops += [sh(fro, i)]  # 将 inc 或 dec 函数的结果添加到 ops 列表中
                fro[i] += ch  # 更新 fro[i] 的值，逐步将其调整到 to[i] 的值

        return ops  # 返回操作列表 ops



    def do_shifts_a(nal, nbk, al, aother, bother):
        """ Shift us from (nal, nbk) to (al, nbk). """
        return do_shifts(nal, al, lambda p, i: ShiftA(p[i]),
                         lambda p, i: UnShiftA(p + aother, nbk + bother, i, z))


    # 定义一个函数 do_shifts_a，用于从 (nal, nbk) 移动到 (al, nbk)
    def do_shifts_a(nal, nbk, al, aother, bother):
        """ Shift us from (nal, nbk) to (al, nbk). """
        # 调用 do_shifts 函数，传入 nal 和 al，使用 ShiftA 和 UnShiftA 函数进行操作
        return do_shifts(nal, al, lambda p, i: ShiftA(p[i]),
                         lambda p, i: UnShiftA(p + aother, nbk + bother, i, z))



    def do_shifts_b(nal, nbk, bk, aother, bother):
        """ Shift us from (nal, nbk) to (nal, bk). """
        return do_shifts(nbk, bk,
                         lambda p, i: UnShiftB(nal + aother, p + bother, i, z),
                         lambda p, i: ShiftB(p[i]))


    # 定义一个函数 do_shifts_b，用于从 (nal, nbk) 移动到 (nal, bk)
    def do_shifts_b(nal, nbk, bk, aother, bother):
        """ Shift us from (nal, nbk) to (nal, bk). """
        # 调用 do_shifts 函数，传入 nbk 和 bk，使用 UnShiftB 和 ShiftB 函数进行操作
        return do_shifts(nbk, bk,
                         lambda p, i: UnShiftB(nal + aother, p + bother, i, z),
                         lambda p, i: ShiftB(p[i]))
    # 对两个字典的键进行排序后遍历，使用默认排序键排序
    for r in sorted(list(abuckets.keys()) + list(bbuckets.keys()), key=default_sort_key):
        # 初始化空元组用于存放数据
        al = ()
        nal = ()
        bk = ()
        nbk = ()
        # 如果键 r 存在于 abuckets 字典中，则分配相应的值给 al 和 nal
        if r in abuckets:
            al = abuckets[r]
            nal = nabuckets[r]
        # 如果键 r 存在于 bbuckets 字典中，则分配相应的值给 bk 和 nbk
        if r in bbuckets:
            bk = bbuckets[r]
            nbk = nbbuckets[r]
        # 如果两个列表的长度不相等，则抛出 ValueError 异常
        if len(al) != len(nal) or len(bk) != len(nbk):
            raise ValueError('%s not reachable from %s' % (target, origin))

        # 对四个列表进行排序，使用默认排序键
        al, nal, bk, nbk = [sorted(w, key=default_sort_key)
            for w in [al, nal, bk, nbk]]

        # 定义函数 others，从字典中选出除 key 外的所有值并扁平化到列表中
        def others(dic, key):
            l = []
            for k in dic:
                if k != key:
                    l.extend(dic[k])
            return l
        # 获取除了当前 r 键以外的所有值列表
        aother = others(nabuckets, r)
        bother = others(nbbuckets, r)

        # 如果 al 列表长度为 0，则没有复杂情况，直接将 bk 列表数据移动到 nbk 列表
        if len(al) == 0:
            ops += do_shifts_b([], nbk, bk, aother, bother)
        # 如果 bk 列表长度为 0，则没有复杂情况，直接将 al 列表数据移动到 nal 列表
        elif len(bk) == 0:
            ops += do_shifts_a(nal, [], al, aother, bother)
        else:
            namax = nal[-1]
            amax = al[-1]

            # 如果 nbk 列表的第一个元素减去 namax 小于等于 0，或者 bk 列表的第一个元素减去 amax 小于等于 0，则抛出 ValueError 异常
            if nbk[0] - namax <= 0 or bk[0] - amax <= 0:
                raise ValueError('Non-suitable parameters.')

            # 如果 namax 减去 amax 大于 0，则向下移动数据，先处理 al 到 nal，再处理 bk 到 nbk
            if namax - amax > 0:
                ops += do_shifts_a(nal, nbk, al, aother, bother)
                ops += do_shifts_b(al, nbk, bk, aother, bother)
            else:
                # 如果 namax 减去 amax 小于等于 0，则向上移动数据，先处理 bk 到 nbk，再处理 al 到 nal
                ops += do_shifts_b(nal, nbk, bk, aother, bother)
                ops += do_shifts_a(nal, bk, al, aother, bother)

        # 更新字典中键 r 对应的值为 al 和 bk
        nabuckets[r] = al
        nbbuckets[r] = bk

    # 将操作列表反转
    ops.reverse()
    # 返回操作列表
    return ops
# 尝试识别从 k > 0 开始的超几何和
def try_shifted_sum(func, z):
    # 将 func.ap 中的元素按照是否为 S.Zero 进行分组，存储在 abuckets 中；类似地，将 func.bq 中的元素按照是否为 S.Zero 进行分组，存储在 bbuckets 中
    abuckets, bbuckets = sift(func.ap, _mod1), sift(func.bq, _mod1)
    # 如果 abuckets[S.Zero] 中不恰好有一个元素，则返回 None
    if len(abuckets[S.Zero]) != 1:
        return None
    # 取 abuckets[S.Zero] 中的第一个元素作为 r
    r = abuckets[S.Zero][0]
    # 如果 r <= 0，则返回 None
    if r <= 0:
        return None
    # 如果 bbuckets 中不包含 S.Zero，则返回 None
    if S.Zero not in bbuckets:
        return None
    # 将 bbuckets[S.Zero] 转换为列表 l，并按升序排序
    l = list(bbuckets[S.Zero])
    l.sort()
    # 取排序后的列表 l 中的第一个元素作为 k
    k = l[0]
    # 如果 k <= 0，则返回 None
    if k <= 0:
        return None

    # 复制 func.ap 列表到 nap，从 nap 中移除 r
    nap = list(func.ap)
    nap.remove(r)
    # 复制 func.bq 列表到 nbq，从 nbq 中移除 k
    nbq = list(func.bq)
    nbq.remove(k)
    # 将 k 减去 1
    k -= 1
    # 对 nap 中的每个元素减去 k
    nap = [x - k for x in nap]
    # 对 nbq 中的每个元素减去 k
    nbq = [x - k for x in nbq]

    # 创建一个空列表 ops
    ops = []
    # 对于范围在 r-1 内的每个 n，向 ops 添加一个 ShiftA(n + 1) 的操作
    for n in range(r - 1):
        ops.append(ShiftA(n + 1))
    # 将 ops 列表反转
    ops.reverse()

    # 计算阶乘 k 的倒数除以 z 的 k 次方，将结果赋给 fac
    fac = factorial(k)/z**k
    # 将 fac 乘以 nbq 中每个元素与 k 相关的一些函数值的乘积
    fac *= Mul(*[rf(b, k) for b in nbq])
    # 将 fac 除以 nap 中每个元素与 k 相关的一些函数值的乘积
    fac /= Mul(*[rf(a, k) for a in nap])

    # 将 fac 作为一个 MultOperator 添加到 ops 的末尾
    ops += [MultOperator(fac)]

    # 初始化 p 为 0
    p = 0
    # 对于范围在 k 内的每个 n
    for n in range(k):
        # 计算 z 的 n 次方除以 n 的阶乘，将结果赋给 m
        m = z**n/factorial(n)
        # 将 m 乘以 nap 中每个元素与 n 相关的一些函数值的乘积
        m *= Mul(*[rf(a, n) for a in nap])
        # 将 m 除以 nbq 中每个元素与 n 相关的一些函数值的乘积
        m /= Mul(*[rf(b, n) for b in nbq])
        # 将 m 加到 p 上
        p += m

    # 返回 Hyper_Function(nap, nbq)、ops 和 -p 的元组
    return Hyper_Function(nap, nbq), ops, -p


# 尝试识别多项式情况。如果不是这种情况，则返回 None。
# 要求 order 已完全简化。
def try_polynomial(func, z):
    # 将 func.ap 中的元素按照是否为 S.Zero 进行分组，存储在 abuckets 中；类似地，将 func.bq 中的元素按照是否为 S.Zero 进行分组，存储在 bbuckets 中
    abuckets, bbuckets = sift(func.ap, _mod1), sift(func.bq, _mod1)
    # 将 abuckets[S.Zero] 中的元素按升序排序
    a0 = abuckets[S.Zero]
    a0.sort()
    # 将 bbuckets[S.Zero] 中的元素按升序排序
    b0 = bbuckets[S.Zero]
    b0.sort()
    # 从 a0 中选择所有小于等于 0 的元素，存储在 al0 中
    al0 = [x for x in a0 if x <= 0]
    # 从 b0 中选择所有小于等于 0 的元素，存储在 bl0 中
    bl0 = [x for x in b0 if x <= 0]

    # 如果 bl0 存在且所有 al0 中的元素都小于 bl0 中的最后一个元素，则返回无穷大 oo
    if bl0 and all(a < bl0[-1] for a in al0):
        return oo
    # 如果 al0 为空，则返回 None
    if not al0:
        return None

    # 取 al0 中的最后一个元素作为 a
    a = al0[-1]
    # 初始化 fac 为 1，初始化 res 为 1
    fac = 1
    res = S.One
    # 对于 -a 范围内的每个 n
    for n in Tuple(*list(range(-a))):
        # 将 fac 乘以 z，除以 n + 1，乘以 func.ap 中每个元素与 n 相关的一些函数值的乘积，除以 func.bq 中每个元素与 n 相关的一些函数值的乘积
        fac *= z
        fac /= n + 1
        fac *= Mul(*[a + n for a in func.ap])
        fac /= Mul(*[b + n for b in func.bq])
        # 将 fac 加到 res 上
        res += fac
    # 返回 res
    return res


# 尝试找到 Hyper_Function "func" 的表达式，用 Lerch 传递来表示。
# 如果找不到这样的表达式，则返回 None。
def try_lerchphi(func):
    # 将 func.ap 中的元素按照是否为 S.Zero 进行分组，存储在 abuckets 中；类似地，将 func.bq 中的元素按照是否为 S.Zero 进行分组，存储在 bbuckets 中
    abuckets, bbuckets = sift(func.ap, _mod1), sift(func.bq, _mod1)

    # 创建空字典 paired
    paired = {}
    # 对于 abuckets 中的每个键值对
    for key, value in abuckets.items():
        # 如果 key 不等于 0 且不在 bbuckets 中，则返回 None
        if key != 0 and key not in bbuckets:
            return None
        # 否则，取 bbuckets 中对应 key 的值，并将键值对存储在 paired 中
        bvalue = bbuckets[key]
        paired[key] = (list(value), list(bvalue))
        # 从 bbuckets 中移除 key 对应的元素
        bbuckets.pop(key, None)
    # 如果 bbuckets 不为空字典，则返回 None
    if bbuckets != {}:
        return None
    # 如果 abuckets 中不包含 S.Zero，则返回 None
    if S.Zero not in abuckets:
        return None
    # 从 paired[S.Zero] 中取出 aints 和 bints
    aints, bints = paired[S.Zero]
    # 将 1 添加到 bints 的末尾，以考虑额外的 n! 项在分母中
    paired[S.Zero] = (aints, bints + [1])

    # 创建一个虚拟变量 t
    t = Dummy('t')
    # 初始化 numer 和 denom 为 1
    numer = S.One
    denom = S.One
    # 遍历 paired 字典中的每对键值对 (key, (avalue, bvalue))
    for key, (avalue, bvalue) in paired.items():
        # 检查 avlaue 和 bvalue 的长度是否相同，如果不同返回 None
        if len(avalue) != len(bvalue):
            return None
        
        # 对于每一对 (a, b) 分别来自 avalue 和 bvalue
        for a, b in zip(avalue, bvalue):
            # 如果 a - b 是正数，则设置 k 为 a - b，并更新 numer 和 denom
            if (a - b).is_positive:
                k = a - b
                numer *= rf(b + t, k)
                denom *= rf(b, k)
            else:
                # 如果 a - b 不是正数，则设置 k 为 b - a，并更新 numer 和 denom
                k = b - a
                numer *= rf(a, k)
                denom *= rf(a + t, k)

    # 现在进行偏分解分解
    # 我们收集两种结构：monomials 列表和 terms 字典
    # monomials 包含 (a, b) 对，代表 a*t**b，其中 b 是非负整数
    # terms 是一个字典，terms[a] = [(b, c)] 表示一个形如 b/(t-a)**c 的项
    part = apart(numer/denom, t)
    args = Add.make_args(part)
    monomials = []
    terms = {}
    for arg in args:
        numer, denom = arg.as_numer_denom()
        # 如果 denom 中不包含 t，则将 numer 分解为多项式 p
        if not denom.has(t):
            p = Poly(numer, t)
            # 如果 p 不是单项式，则抛出 TypeError 异常
            if not p.is_monomial:
                raise TypeError("p should be monomial")
            ((b, ), a) = p.LT()
            monomials += [(a/denom, b)]
            continue
        
        # 如果 numer 中包含 t，则需要进行偏分解
        if numer.has(t):
            raise NotImplementedError('Need partial fraction decomposition'
                                      ' with linear denominators')
        
        # 将 denom 分解为 indep 和 dep，其中 dep 是 t 的系数
        indep, [dep] = denom.as_coeff_mul(t)
        n = 1
        if dep.is_Pow:
            n = dep.exp
            dep = dep.base
        # 如果 dep 是 t，则 a = 0
        if dep == t:
            a == 0
        # 如果 dep 是 Add 类型，则将其分解为 a 和 tmp
        elif dep.is_Add:
            a, tmp = dep.as_independent(t)
            b = 1
            if tmp != t:
                b, _ = tmp.as_independent(t)
            # 如果 dep 不等于 b*t + a，则抛出异常
            if dep != b*t + a:
                raise NotImplementedError('unrecognised form %s' % dep)
            a /= b
            indep *= b**n
        else:
            raise NotImplementedError('unrecognised form of partial fraction')
        # 将 (numer/indep, n) 添加到 terms[a] 中
        terms.setdefault(a, []).append((numer/indep, n))

    # 现在有了这些信息，组装我们的公式
    # 所有的 monomials 产生有理函数，并放入一个基本元素
    # terms[a] 通过微分相关联。如果最大指数是 n，则我们需要对于 k = 1, 2, ..., n，
    # 使用 lerchphi(z, k, a)
    # deriv 将一个基础映射到其导数，表达为其他基础元素的 C(z)-线性组合。
    deriv = {}
    coeffs = {}
    z = Dummy('z')
    monomials.sort(key=lambda x: x[1])
    mon = {0: 1/(1 - z)}
    # 如果 monomials 非空，则根据 monomials[-1][1] 构建 mon 字典
    if monomials:
        for k in range(monomials[-1][1]):
            mon[k + 1] = z*mon[k].diff(z)
    # 对于每个 (a, n) 在 monomials 中，将 a*mon[n] 添加到 coeffs[S.One] 中
    for a, n in monomials:
        coeffs.setdefault(S.One, []).append(a*mon[n])
    # 遍历字典 terms，terms.items() 返回 (a, l)，a 是键，l 是值
    for a, l in terms.items():
        # 遍历 l 中的元素，每个元素是一个元组 (c, k)
        for c, k in l:
            # 将 lerchphi(z, k, a) 作为键，将 c 添加到 coeffs 字典的值列表中
            coeffs.setdefault(lerchphi(z, k, a), []).append(c)
        # 对 l 中的元素按照第二个元素 k 进行排序
        l.sort(key=lambda x: x[1])
        # 遍历从 2 到 l 中最后一个元素的第二个元素 l[-1][1]（包含）的范围
        for k in range(2, l[-1][1] + 1):
            # 构建 deriv 字典，键为 lerchphi(z, k, a)，值为包含两个元组的列表
            deriv[lerchphi(z, k, a)] = [(-a, lerchphi(z, k, a)),
                                        (1, lerchphi(z, k - 1, a))]
        # 构建 deriv 字典中的特殊情况，当 k 为 1 时
        deriv[lerchphi(z, 1, a)] = [(-a, lerchphi(z, 1, a)),
                                    (1/(1 - z), S.One)]
    
    # 初始化空字典 trans
    trans = {}
    # 枚举 deriv.keys()，并用序号 n 作为键值存入 trans 字典
    for n, b in enumerate([S.One] + list(deriv.keys())):
        trans[b] = n
    
    # 初始化 basis 列表，包含所有 trans.items() 按照值排序后的函数 b 的扩展
    basis = [expand_func(b) for (b, _) in sorted(trans.items(),
                                                 key=lambda x: x[1])]
    
    # 构建矩阵 B，包含 basis 列表中的元素
    B = Matrix(basis)
    
    # 构建矩阵 C，初始化为与 B 同样大小的零矩阵
    C = Matrix([[0]*len(B)])
    
    # 遍历 coeffs.items()，将每个 coeffs 字典中的项 c 添加到 C 中相应的位置
    for b, c in coeffs.items():
        C[trans[b]] = Add(*c)
    
    # 初始化 M 为与 B 相同大小的零矩阵
    M = zeros(len(B))
    
    # 遍历 deriv.items()，构建 M 矩阵，将 c 放入 M 的相应位置
    for b, l in deriv.items():
        for c, b2 in l:
            M[trans[b], trans[b2]] = c
    
    # 返回一个 Formula 对象，包含 func, z, B, C, M，其他参数为空列表或 None
    return Formula(func, z, None, [], B, C, M)
# 创建一个函数，用于构建表示超几何函数 `func` 的公式对象
def build_hypergeometric_formula(func):
    """
    Create a formula object representing the hypergeometric function ``func``.
    """

    # 我们知道 `ap` 中没有负整数，否则会触发“检测多项式”。
    # 但是 `ap` 可能为空。在这种情况下，我们可以使用不同的基础。
    # 我不知道哪种基础在所有情况下都有效。
    z = Dummy('z')

    # 如果 `func.ap` 不为空
    if func.ap:
        # 计算 `afactors` 和 `bfactors`
        afactors = [_x + a for a in func.ap]
        bfactors = [_x + b - 1 for b in func.bq]

        # 构建表达式
        expr = _x * Mul(*bfactors) - z * Mul(*afactors)
        poly = Poly(expr, _x)
        n = poly.degree()
        basis = []
        M = zeros(n)

        # 构建基础函数的矩阵
        for k in range(n):
            a = func.ap[0] + k
            basis += [hyper([a] + list(func.ap[1:]), func.bq, z)]
            if k < n - 1:
                M[k, k] = -a
                M[k, k + 1] = a

        B = Matrix(basis)
        C = Matrix([[1] + [0]*(n - 1)])

        # 计算导数
        derivs = [eye(n)]
        for k in range(n):
            derivs.append(M * derivs[k])

        # 计算系数
        l = poly.all_coeffs()
        l.reverse()
        res = [0] * n
        for k, c in enumerate(l):
            for r, d in enumerate(C * derivs[k]):
                res[r] += c * d

        # 计算 `M` 矩阵的最后一行
        for k, c in enumerate(res):
            M[n - 1, k] = -c / derivs[n - 1][0, n - 1] / poly.all_coeffs()[0]

        # 返回构建的公式对象
        return Formula(func, z, None, [], B, C, M)

    else:
        # 如果 `func.ap` 为空，则 `bq` 中没有非正整数
        basis = []
        bq = list(func.bq[:])

        # 构建基础函数的矩阵
        for i in range(len(bq)):
            basis += [hyper([], bq, z)]
            bq[i] += 1

        basis += [hyper([], bq, z)]
        B = Matrix(basis)
        n = len(B)
        C = Matrix([[1] + [0]*(n - 1)])
        M = zeros(n)
        M[0, n - 1] = z / Mul(*func.bq)

        # 填充 `M` 矩阵的其余部分
        for k in range(1, n):
            M[k, k - 1] = func.bq[k - 1]
            M[k, k] = -func.bq[k - 1]

        # 返回构建的公式对象
        return Formula(func, z, None, [], B, C, M)


def hyperexpand_special(ap, bq, z):
    """
    Try to find a closed-form expression for hyper(ap, bq, z), where ``z``
    is supposed to be a "special" value, e.g. 1.

    This function tries various of the classical summation formulae
    (Gauss, Saalschuetz, etc).
    """

    # 这段代码非常特定。有许多与这个问题相关的聪明算法
    # （尤其是 Zeliberger 的算法）。
    # 目前我们只希望几个简单的情况能够工作。
    p, q = len(ap), len(bq)
    z_ = z
    z = unpolarify(z)

    # 如果 `z` 等于 0，返回 SymPy 中的单位元素 S.One
    if z == 0:
        return S.One

    from sympy.simplify.simplify import simplify
    # 如果 p 等于 2 并且 q 等于 1，则执行以下代码块
    if p == 2 and q == 1:
        # 2F1 超几何函数的情况
        a, b, c = ap + bq
        # 如果 z 等于 1，则执行以下代码块
        if z == 1:
            # 使用高斯公式计算超几何函数的值
            return gamma(c - a - b)*gamma(c)/gamma(c - a)/gamma(c - b)
        # 如果 z 等于 -1 并且简化后的表达式 b - a + c 等于 1，则执行以下代码块
        if z == -1 and simplify(b - a + c) == 1:
            # 切换到库默公式计算超几何函数的值
            b, a = a, b
        # 如果 z 等于 -1 并且简化后的表达式 a - b + c 等于 1，则执行以下代码块
        if z == -1 and simplify(a - b + c) == 1:
            # 库默公式计算超几何函数的值
            if b.is_integer and b.is_negative:
                # 当 b 是整数且为负数时，使用特定公式计算
                return 2*cos(pi*b/2)*gamma(-b)*gamma(b - a + 1) \
                    /gamma(-b/2)/gamma(b/2 - a + 1)
            else:
                # 否则，使用一般的库默公式计算
                return gamma(b/2 + 1)*gamma(b - a + 1) \
                    /gamma(b + 1)/gamma(b/2 - a + 1)
    # 如果上述条件不满足，返回超几何函数的一般计算结果
    # TODO 大量的其他公式
    #      调查存在的算法
    return hyper(ap, bq, z_)
_collection = None
# 定义全局变量 _collection，用于存储 FormulaCollection 的实例对象

def _hyperexpand(func, z, ops0=[], z0=Dummy('z0'), premult=1, prem=0,
                 rewrite='default'):
    """
    尝试找到超几何函数 func 的表达式。

    Explanation
    ===========

    结果用虚拟变量 z0 表示。然后乘以 premult。然后应用 ops0。
    premult 必须是形如 a*z**prem 的形式，其中 a 与 z 无关。
    """

    if z.is_zero:
        return S.One
    # 如果 z 为零，则返回 1

    from sympy.simplify.simplify import simplify

    z = polarify(z, subs=False)
    # 极化 z，禁用替换
    if rewrite == 'default':
        rewrite = 'nonrepsmall'

    def carryout_plan(f, ops):
        C = apply_operators(f.C.subs(f.z, z0), ops,
                            make_derivative_operator(f.M.subs(f.z, z0), z0))
        C = apply_operators(C, ops0,
                            make_derivative_operator(f.M.subs(f.z, z0)
                                         + prem*eye(f.M.shape[0]), z0))

        if premult == 1:
            C = C.applyfunc(make_simp(z0))
        r = reduce(lambda s,m: s+m[0]*m[1], zip(C, f.B.subs(f.z, z0)), S.Zero)*premult
        res = r.subs(z0, z)
        if rewrite:
            res = res.rewrite(rewrite)
        return res
        # 执行计划，生成 C，然后对其应用操作 ops0 和 premult

    # TODO
    # 以下是可能的：
    # *) PFD Duplication (见 Kelly Roach 的论文)
    # *) 以类似精神，try_lerchphi() 可以被广泛泛化。

    global _collection
    if _collection is None:
        _collection = FormulaCollection()
        # 如果 _collection 为空，则创建 FormulaCollection 的实例对象

    debug('Trying to expand hypergeometric function ', func)
    # 调试信息，尝试展开超几何函数 func

    # 首先尽量减少阶数。
    func, ops = reduce_order(func)
    if ops:
        debug('  Reduced order to ', func)
    else:
        debug('  Could not reduce order.')
    # 减少 func 的阶数，返回减少后的函数和操作 ops

    # 现在尝试多项式情况
    res = try_polynomial(func, z0)
    if res is not None:
        debug('  Recognised polynomial.')
        p = apply_operators(res, ops, lambda f: z0*f.diff(z0))
        p = apply_operators(p*premult, ops0, lambda f: z0*f.diff(z0))
        return unpolarify(simplify(p).subs(z0, z))
        # 如果是多项式，返回简化后的结果

    # 尝试识别偏移求和
    p = S.Zero
    res = try_shifted_sum(func, z0)
    if res is not None:
        func, nops, p = res
        debug('  Recognised shifted sum, reduced order to ', func)
        ops += nops
    # 尝试识别偏移求和，并更新操作 ops

    # 应用多项式计划
    p = apply_operators(p, ops, lambda f: z0*f.diff(z0))
    p = apply_operators(p*premult, ops0, lambda f: z0*f.diff(z0))
    p = simplify(p).subs(z0, z)
    # 应用操作 ops 和 ops0 到 p，并简化结果

    # 尝试早期特殊扩展。
    if unpolarify(z) in [1, -1] and (len(func.ap), len(func.bq)) == (2, 1):
        f = build_hypergeometric_formula(func)
        r = carryout_plan(f, ops).replace(hyper, hyperexpand_special)
        if not r.has(hyper):
            return r + p
        # 如果满足条件，构建超几何公式并执行计划

    # 尝试在我们的 collection 中找到公式
    formula = _collection.lookup_origin(func)
    # 查找 func 在 collection 中的原始公式

    # 现在尝试 lerch phi 公式
    if formula is None:
        formula = try_lerchphi(func)
        # 如果在 collection 中未找到，则尝试 lerch phi 公式
    # 如果给定的 formula 参数为 None，则输出调试信息并使用超几何函数构建公式
    if formula is None:
        debug('  Could not find an origin. ',
              'Will return answer in terms of '
              'simpler hypergeometric functions.')
        formula = build_hypergeometric_formula(func)

    # 输出调试信息，指示找到了一个起点，并显示闭式和函数信息
    debug('  Found an origin: ', formula.closed_form, ' ', formula.func)

    # 需要找到将 formula 转换为 func 的操作符
    ops += devise_plan(func, formula.func, z0)

    # 执行操作计划，并将结果与 p 相加
    r = carryout_plan(formula, ops) + p

    # 对结果进行幂次简化，使用极坐标，并替换特殊的超几何函数
    return powdenest(r, polar=True).replace(hyper, hyperexpand_special)
    # 将G函数“fro”转换为G函数“to”的操作符计划
    """
    It is assumed that ``fro`` and ``to`` have the same signatures, and that in fact
    any corresponding pair of parameters differs by integers, and a direct path
    is possible. I.e. if there are parameters a1 b1 c1  and a2 b2 c2 it is
    assumed that a1 can be shifted to a2, etc. The only thing this routine
    determines is the order of shifts to apply, nothing clever will be tried.
    It is also assumed that ``fro`` is suitable.
    """
    # 确定函数签名相同，各对应参数可通过整数差直接转换的假设

    # 导入所需的模块和符号
    >>> from sympy.simplify.hyperexpand import (devise_plan_meijer,
    ...                                         G_Function)
    >>> from sympy.abc import z

    # 空计划示例
    >>> devise_plan_meijer(G_Function([1], [2], [3], [4]),
    ...                    G_Function([1], [2], [3], [4]), z)
    []

    # 非常简单的计划示例
    >>> devise_plan_meijer(G_Function([0], [], [], []),
    ...                    G_Function([1], [], [], []), z)
    [<Increment upper a index #0 of [0], [], [], [].>]
    >>> devise_plan_meijer(G_Function([0], [], [], []),
    ...                    G_Function([-1], [], [], []), z)
    [<Decrement upper a=0.>]
    >>> devise_plan_meijer(G_Function([], [1], [], []),
    ...                    G_Function([], [2], [], []), z)
    [<Increment lower a index #0 of [], [1], [], [].>]

    # 更复杂的计划示例
    >>> devise_plan_meijer(G_Function([0], [], [], []),
    ...                    G_Function([2], [], [], []), z)
    [<Increment upper a index #0 of [1], [], [], [].>,
    <Increment upper a index #0 of [0], [], [], [].>]
    >>> devise_plan_meijer(G_Function([0], [], [0], []),
    ...                    G_Function([-1], [], [1], []), z)
    [<Increment upper b=0.>, <Decrement upper a=0.>]

    # 计划的顺序很重要
    >>> devise_plan_meijer(G_Function([0], [], [0], []),
    ...                    G_Function([1], [], [1], []), z)
    [<Increment upper a index #0 of [0], [], [1], [].>, <Increment upper b=0.>]
    """
    # 目前我们使用以下简单的启发式方法：逆转移（若可能），否则进行移位。如果无法取得进展，则放弃。

    def try_shift(f, t, shifter, diff, counter):
        """尝试应用shifter以使“f”中的某些元素靠近其在“to”中的对应元素。
           diff为+/- 1，决定shifter的效果。counter是阻止移动的元素列表。

           如果改变成功，返回一个操作符，否则返回None。
        """
        for idx, (a, b) in enumerate(zip(f, t)):
            if (
                (a - b).is_integer and (b - a)/diff > 0 and
                    all(a != x for x in counter)):
                sh = shifter(idx)
                f[idx] += diff
                return sh

    # 将fro的各部分列表化
    fan = list(fro.an)
    fap = list(fro.ap)
    fbm = list(fro.bm)
    fbq = list(fro.bq)
    # 初始化一个空操作列表
    ops = []
    # 设置一个标志，表示是否进行了变化
    change = True
    # 循环直到没有变化为止
    while change:
        # 将变化标志设为 False，若有操作成功则会设置为 True 继续循环
        change = False
        # 尝试执行 MeijerUnShiftB 操作
        op = try_shift(fan, to.an,
                       lambda i: MeijerUnShiftB(fan, fap, fbm, fbq, i, z),
                       1, fbm + fbq)
        # 如果操作成功，则将操作添加到 ops 列表中，并将 change 标志设为 True 继续循环
        if op is not None:
            ops += [op]
            change = True
            continue
        # 尝试执行 MeijerUnShiftD 操作
        op = try_shift(fap, to.ap,
                       lambda i: MeijerUnShiftD(fan, fap, fbm, fbq, i, z),
                       1, fbm + fbq)
        if op is not None:
            ops += [op]
            change = True
            continue
        # 尝试执行 MeijerUnShiftA 操作
        op = try_shift(fbm, to.bm,
                       lambda i: MeijerUnShiftA(fan, fap, fbm, fbq, i, z),
                       -1, fan + fap)
        if op is not None:
            ops += [op]
            change = True
            continue
        # 尝试执行 MeijerUnShiftC 操作
        op = try_shift(fbq, to.bq,
                       lambda i: MeijerUnShiftC(fan, fap, fbm, fbq, i, z),
                       -1, fan + fap)
        if op is not None:
            ops += [op]
            change = True
            continue
        # 尝试执行 MeijerShiftB 操作
        op = try_shift(fan, to.an, lambda i: MeijerShiftB(fan[i]), -1, [])
        if op is not None:
            ops += [op]
            change = True
            continue
        # 尝试执行 MeijerShiftD 操作
        op = try_shift(fap, to.ap, lambda i: MeijerShiftD(fap[i]), -1, [])
        if op is not None:
            ops += [op]
            change = True
            continue
        # 尝试执行 MeijerShiftA 操作
        op = try_shift(fbm, to.bm, lambda i: MeijerShiftA(fbm[i]), 1, [])
        if op is not None:
            ops += [op]
            change = True
            continue
        # 尝试执行 MeijerShiftC 操作
        op = try_shift(fbq, to.bq, lambda i: MeijerShiftC(fbq[i]), 1, [])
        if op is not None:
            ops += [op]
            change = True
            continue
    # 检查是否成功转换所有的索引
    if fan != list(to.an) or fap != list(to.ap) or fbm != list(to.bm) or \
            fbq != list(to.bq):
        # 如果未成功转换，则抛出未实现错误
        raise NotImplementedError('Could not devise plan.')
    # 将操作列表反转，以返回正确的执行顺序
    ops.reverse()
    # 返回操作列表
    return ops
# 全局变量，用于存储 Meijer 函数的展开集合，默认为 None
_meijercollection = None


# 尝试展开指定 Meijer G 函数的表达式
def _meijergexpand(func, z0, allow_hyper=False, rewrite='default',
                   place=None):
    """
    Try to find an expression for the Meijer G function specified
    by the G_Function ``func``. If ``allow_hyper`` is True, then returning
    an expression in terms of hypergeometric functions is allowed.

    Currently this just does Slater's theorem.
    If expansions exist both at zero and at infinity, ``place``
    can be set to ``0`` or ``zoo`` for the preferred choice.
    """
    global _meijercollection
    # 如果 _meijercollection 为 None，则初始化为 MeijerFormulaCollection 的实例
    if _meijercollection is None:
        _meijercollection = MeijerFormulaCollection()
    
    # 如果 rewrite 参数为 'default'，则将其设为 None
    if rewrite == 'default':
        rewrite = None

    # 备份原始的 func
    func0 = func
    # 输出调试信息，尝试展开对应的 Meijer G 函数
    debug('Try to expand Meijer G function corresponding to ', func)

    # 我们将对解析延伸进行操作 - 最好使用一个新的符号
    z = Dummy('z')

    # 对 Meijer G 函数进行降阶操作，返回降阶后的函数和操作符
    func, ops = reduce_order_meijer(func)
    if ops:
        debug('  Reduced order to ', func)
    else:
        debug('  Could not reduce order.')

    # 尝试寻找直接的公式表达
    f = _meijercollection.lookup_origin(func)
    if f is not None:
        debug('  Found a Meijer G formula: ', f.func)
        # 设计执行 Meijer G 函数展开的操作计划
        ops += devise_plan_meijer(f.func, func, z)

        # 执行操作计划
        C = apply_operators(f.C.subs(f.z, z), ops,
                            make_derivative_operator(f.M.subs(f.z, z), z))

        # 对结果应用简化操作
        C = C.applyfunc(make_simp(z))
        r = C * f.B.subs(f.z, z)
        # 在 z=z0 处求值，并进行幂次简化
        r = r[0].subs(z, z0)
        return powdenest(r, polar=True)

    debug("  Could not find a direct formula. Trying Slater's theorem.")

    # TODO 可能的操作：
    # *) Paired Index Theorems
    # *) PFD Duplication
    #    (See Kelly Roach's paper for details on either.)
    #
    # TODO 同样，我们倾向于创建可以简化的 gamma 函数组合。

    # 定义用于测试 Slater 定理适用性的函数
    def can_do(pbm, pap):
        """ Test if slater applies. """
        for i in pbm:
            if len(pbm[i]) > 1:
                l = 0
                if i in pap:
                    l = len(pap[i])
                if l + 1 < len(pbm[i]):
                    return False
        return True

    # 定义一个符号 t
    t = Dummy('t')
    # 使用 Slater 定理尝试展开
    slater1, cond1 = do_slater(func.an, func.bm, func.ap, func.bq, z, z0)

    # 定义一个函数，用于对列表 l 进行转换
    def tr(l):
        return [1 - x for x in l]

    # 对 ops 中的每一个操作，进行变量替换
    for op in ops:
        op._poly = Poly(op._poly.subs({z: 1/t, _x: -_x}), _x)
    # 使用 Slater 定理尝试展开
    slater2, cond2 = do_slater(tr(func.bm), tr(func.an), tr(func.bq), tr(func.ap),
                               t, 1/z0)

    # 对结果进行幂次简化
    slater1 = powdenest(slater1.subs(z, z0), polar=True)
    slater2 = powdenest(slater2.subs(t, 1/z0), polar=True)
    if not isinstance(cond2, bool):
        cond2 = cond2.subs(t, 1/z)

    # 计算 func(z) 的值
    m = func(z)
    # 如果 m.delta 大于 0 或者以下条件成立：
    # - m.delta 等于 0 并且 m.ap 的长度等于 m.bq 的长度
    # - re(m.nu) < -1 并且 polar_lift(z0) 等于 polar_lift(1)
    # 则执行以下操作：
    # - 条件 delta > 0 意味着收敛区域是连通的，我们可以在整个收敛区域内进行解析延拓。
    # - 条件 delta==0, p==q, re(nu) < -1 意味着 G 在正实轴上连续，因此在 z=1 处的值是一致的。
    if m.delta > 0 or \
        (m.delta == 0 and len(m.ap) == len(m.bq) and
            (re(m.nu) < -1) is not False and polar_lift(z0) == polar_lift(1)):
        
        # 如果 cond1 不是 False，则将其设为 True
        if cond1 is not False:
            cond1 = True
        # 如果 cond2 不是 False，则将其设为 True
        if cond2 is not False:
            cond2 = True

    # 如果 cond1 是 True，则调用 slater1.rewrite(rewrite or 'nonrep')，否则调用 slater1.rewrite(rewrite or 'nonrepsmall')
    if cond1 is True:
        slater1 = slater1.rewrite(rewrite or 'nonrep')
    else:
        slater1 = slater1.rewrite(rewrite or 'nonrepsmall')
    
    # 如果 cond2 是 True，则调用 slater2.rewrite(rewrite or 'nonrep')，否则调用 slater2.rewrite(rewrite or 'nonrepsmall')
    if cond2 is True:
        slater2 = slater2.rewrite(rewrite or 'nonrep')
    else:
        slater2 = slater2.rewrite(rewrite or 'nonrepsmall')

    # 如果 cond1 和 cond2 都不是 False，则执行以下操作：
    # - 如果 place 等于 0，则将 cond2 设为 False
    # - 如果 place 等于 zoo，则将 cond1 设为 False
    if cond1 is not False and cond2 is not False:
        if place == 0:
            cond2 = False
        if place == zoo:
            cond1 = False

    # 如果 cond1 不是 bool 类型，则将其替换为 cond1.subs(z, z0)
    if not isinstance(cond1, bool):
        cond1 = cond1.subs(z, z0)
    
    # 如果 cond2 不是 bool 类型，则将其替换为 cond2.subs(z, z0)
    if not isinstance(cond2, bool):
        cond2 = cond2.subs(z, z0)

    # 定义一个函数 weight，根据 cond 的不同取值计算权重 c0
    def weight(expr, cond):
        if cond is True:
            c0 = 0
        elif cond is False:
            c0 = 1
        else:
            c0 = 2
        # 如果 expr 包含无穷大、无穷小、负无穷大或非数值，则设置 c0 为 3
        if expr.has(oo, zoo, -oo, nan):
            # XXX 这实际上不应发生，但考虑下述情况
            # S('meijerg(((0, -1/2, 0, -1/2, 1/2), ()), ((0,),
            #   (-1/2, -1/2, -1/2, -1)), exp_polar(I*pi))/4')
            c0 = 3
        return (c0, expr.count(hyper), expr.count_ops())

    # 计算 slater1 和 slater2 的权重 w1 和 w2
    w1 = weight(slater1, cond1)
    w2 = weight(slater2, cond2)

    # 如果 w1 和 w2 的最小值小于等于 (0, 1, oo)，则返回权重较小的 slater 函数
    if min(w1, w2) <= (0, 1, oo):
        if w1 < w2:
            return slater1
        else:
            return slater2

    # 如果 w1 和 w2 的第一个元素的最大值小于等于 1，并且第二个元素的最大值小于等于 1，则返回 Piecewise 对象
    if max(w1[0], w2[0]) <= 1 and max(w1[1], w2[1]) <= 1:
        return Piecewise((slater1, cond1), (slater2, cond2), (func0(z0), True))

    # 如果找不到不含超几何函数的表达式，则返回 Piecewise 对象 r
    r = Piecewise((slater1, cond1), (slater2, cond2), (func0(z0), True))
    
    # 如果 r 包含超几何函数并且不允许使用超几何函数，则输出调试信息
    if r.has(hyper) and not allow_hyper:
        debug('  Could express using hypergeometric functions, '
              'but not allowed.')
    
    # 如果 r 不包含超几何函数或允许使用超几何函数，则返回 r
    if not r.has(hyper) or allow_hyper:
        return r

    # 否则返回 func0(z0)
    return func0(z0)
# 定义函数 hyperexpand，用于展开超几何函数。
# 如果 allow_hyper 为 True，则允许部分简化，可能包含超几何函数。
# 如果 G-函数在零点和无穷处都有展开式，则可以通过 place 参数指定首选选择。
def hyperexpand(f, allow_hyper=False, rewrite='default', place=None):
    """
    Expand hypergeometric functions. If allow_hyper is True, allow partial
    simplification (that is a result different from input,
    but still containing hypergeometric functions).

    If a G-function has expansions both at zero and at infinity,
    ``place`` can be set to ``0`` or ``zoo`` to indicate the
    preferred choice.

    Examples
    ========

    >>> from sympy.simplify.hyperexpand import hyperexpand
    >>> from sympy.functions import hyper
    >>> from sympy.abc import z
    >>> hyperexpand(hyper([], [], z))
    exp(z)

    Non-hyperegeometric parts of the expression and hypergeometric expressions
    that are not recognised are left unchanged:

    >>> hyperexpand(1 + hyper([1, 1, 1], [], z))
    hyper((1, 1, 1), (), z) + 1
    """
    # 将输入的 f 转换为 SymPy 表达式
    f = sympify(f)

    # 定义内部函数 do_replace，用于替换超几何函数的表达式
    def do_replace(ap, bq, z):
        # 调用 _hyperexpand 函数来展开超几何函数 Hyper_Function(ap, bq)
        r = _hyperexpand(Hyper_Function(ap, bq), z, rewrite=rewrite)
        # 如果展开结果为 None，则返回未展开的超几何函数
        if r is None:
            return hyper(ap, bq, z)
        else:
            return r

    # 定义内部函数 do_meijer，用于替换 Meijer G 函数的表达式
    def do_meijer(ap, bq, z):
        # 调用 _meijergexpand 函数来展开 Meijer G 函数 G_Function(ap, bq)
        r = _meijergexpand(G_Function(ap[0], ap[1], bq[0], bq[1]), z,
                   allow_hyper, rewrite=rewrite, place=place)
        # 如果展开结果不包含 nan, zoo, oo, -oo，则返回展开后的结果
        if not r.has(nan, zoo, oo, -oo):
            return r
    
    # 替换 f 中的超几何函数调用为展开后的结果，替换 meijerg 函数调用为展开后的结果
    return f.replace(hyper, do_replace).replace(meijerg, do_meijer)
```