# `D:\src\scipysrc\sympy\sympy\integrals\meijerint.py`

```
"""
Integrate functions by rewriting them as Meijer G-functions.

There are three user-visible functions that can be used by other parts of the
sympy library to solve various integration problems:

- meijerint_indefinite
- meijerint_definite
- meijerint_inversion

They can be used to compute, respectively, indefinite integrals, definite
integrals over intervals of the real line, and inverse laplace-type integrals
(from c-I*oo to c+I*oo). See the respective docstrings for details.

The main references for this are:

[L] Luke, Y. L. (1969), The Special Functions and Their Approximations,
    Volume 1

[R] Kelly B. Roach.  Meijer G Function Representations.
    In: Proceedings of the 1997 International Symposium on Symbolic and
    Algebraic Computation, pages 205-211, New York, 1997. ACM.

[P] A. P. Prudnikov, Yu. A. Brychkov and O. I. Marichev (1990).
    Integrals and Series: More Special Functions, Vol. 3,.
    Gordon and Breach Science Publisher
"""

# 导入所需模块和函数
from __future__ import annotations  # 允许函数的类型提示返回自身类型
import itertools  # 提供用于迭代的工具函数

from sympy import SYMPY_DEBUG  # 导入 sympy 库的调试标志
from sympy.core import S, Expr  # 导入 sympy 核心模块中的 S 和 Expr 类
from sympy.core.add import Add  # 导入 sympy 核心模块中的 Add 类
from sympy.core.basic import Basic  # 导入 sympy 核心模块中的 Basic 类
from sympy.core.cache import cacheit  # 导入 sympy 核心模块中的 cacheit 函数
from sympy.core.containers import Tuple  # 导入 sympy 核心模块中的 Tuple 类
from sympy.core.exprtools import factor_terms  # 导入 sympy 核心模块中的 factor_terms 函数
from sympy.core.function import (expand, expand_mul, expand_power_base,  # 导入 sympy 核心模块中的函数
                                 expand_trig, Function)
from sympy.core.mul import Mul  # 导入 sympy 核心模块中的 Mul 类
from sympy.core.intfunc import ilcm  # 导入 sympy 核心模块中的 ilcm 函数
from sympy.core.numbers import Rational, pi  # 导入 sympy 核心模块中的 Rational 和 pi 类
from sympy.core.relational import Eq, Ne, _canonical_coeff  # 导入 sympy 核心模块中的类和函数
from sympy.core.sorting import default_sort_key, ordered  # 导入 sympy 核心模块中的函数
from sympy.core.symbol import Dummy, symbols, Wild, Symbol  # 导入 sympy 核心模块中的类和函数
from sympy.core.sympify import sympify  # 导入 sympy 核心模块中的 sympify 函数
from sympy.functions.combinatorial.factorials import factorial  # 导入 sympy 组合函数模块中的 factorial 函数
from sympy.functions.elementary.complexes import (re, im, arg, Abs, sign,  # 导入 sympy 复数函数模块中的函数
        unpolarify, polarify, polar_lift, principal_branch, unbranched_argument,
        periodic_argument)
from sympy.functions.elementary.exponential import exp, exp_polar, log  # 导入 sympy 指数函数模块中的函数
from sympy.functions.elementary.integers import ceiling  # 导入 sympy 整数函数模块中的 ceiling 函数
from sympy.functions.elementary.hyperbolic import (cosh, sinh,  # 导入 sympy 双曲函数模块中的函数
        _rewrite_hyperbolics_as_exp, HyperbolicFunction)
from sympy.functions.elementary.miscellaneous import sqrt  # 导入 sympy 杂项函数模块中的 sqrt 函数
from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold  # 导入 sympy 分段函数模块中的函数
from sympy.functions.elementary.trigonometric import (cos, sin, sinc,  # 导入 sympy 三角函数模块中的函数
        TrigonometricFunction)
from sympy.functions.special.bessel import besselj, bessely, besseli, besselk  # 导入 sympy 贝塞尔函数模块中的函数
from sympy.functions.special.delta_functions import DiracDelta, Heaviside  # 导入 sympy delta 函数模块中的函数
from sympy.functions.special.elliptic_integrals import elliptic_k, elliptic_e  # 导入 sympy 椭圆积分函数模块中的函数
from sympy.functions.special.error_functions import (erf, erfc, erfi, Ei,  # 导入 sympy 错误函数模块中的函数
        expint, Si, Ci, Shi, Chi, fresnels, fresnelc)
from sympy.functions.special.gamma_functions import gamma  # 导入 sympy 伽玛函数模块中的 gamma 函数
from sympy.functions.special.hyper import hyper, meijerg  # 导入 sympy 超函模块中的函数
from sympy.functions.special.singularity_functions import SingularityFunction  # 导入 sympy 奇异函数模块中的 SingularityFunction 类
# 从integrals模块导入Integral类
from .integrals import Integral
# 从sympy.logic.boolalg模块导入逻辑运算符And, Or, BooleanAtom, Not, BooleanFunction
from sympy.logic.boolalg import And, Or, BooleanAtom, Not, BooleanFunction
# 从sympy.polys模块导入cancel和factor函数
from sympy.polys import cancel, factor
# 从sympy.utilities.iterables模块导入multiset_partitions函数
from sympy.utilities.iterables import multiset_partitions
# 从sympy.utilities.misc模块导入debug函数，并将其重命名为_debug
from sympy.utilities.misc import debug as _debug
# 从sympy.utilities.misc模块导入debugf函数，并将其重命名为_debugf

# 在顶部保留此变量以便于引用
z = Dummy('z')

def _has(res, *f):
    # 如果res具有f中的所有函数，则返回True；对于Piecewise函数，只有当*所有*分段都具有f时才返回True
    res = piecewise_fold(res)
    if getattr(res, 'is_Piecewise', False):
        return all(_has(i, *f) for i in res.args)
    return res.has(*f)

def _create_lookup_table(table):
    """ 向函数到MeijerG查找表中添加公式。 """
    def wild(n):
        return Wild(n, exclude=[z])
    p, q, a, b, c = list(map(wild, 'pqabc'))
    n = Wild('n', properties=[lambda x: x.is_Integer and x > 0])
    t = p*z**q

    def add(formula, an, ap, bm, bq, arg=t, fac=S.One, cond=True, hint=True):
        table.setdefault(_mytype(formula, z), []).append((formula,
                                     [(fac, meijerg(an, ap, bm, bq, arg))], cond, hint))

    def addi(formula, inst, cond, hint=True):
        table.setdefault(
            _mytype(formula, z), []).append((formula, inst, cond, hint))

    def constant(a):
        return [(a, meijerg([1], [], [], [0], z)),
                (a, meijerg([], [1], [0], [], z))]
    table[()] = [(a, constant(a), True, True)]

    # [P], Section 8.
    class IsNonPositiveInteger(Function):

        @classmethod
        def eval(cls, arg):
            arg = unpolarify(arg)
            if arg.is_Integer is True:
                return arg <= 0

    # Section 8.4.2
    # TODO this needs more polar_lift (c/f entry for exp)
    # 添加Heaviside函数的MeijerG转换公式到查找表中
    add(Heaviside(t - b)*(t - b)**(a - 1), [a], [], [], [0], t/b,
        gamma(a)*b**(a - 1), And(b > 0))
    add(Heaviside(b - t)*(b - t)**(a - 1), [], [a], [0], [], t/b,
        gamma(a)*b**(a - 1), And(b > 0))
    add(Heaviside(z - (b/p)**(1/q))*(t - b)**(a - 1), [a], [], [], [0], t/b,
        gamma(a)*b**(a - 1), And(b > 0))
    add(Heaviside((b/p)**(1/q) - z)*(b - t)**(a - 1), [], [a], [0], [], t/b,
        gamma(a)*b**(a - 1), And(b > 0))
    add((b + t)**(-a), [1 - a], [], [0], [], t/b, b**(-a)/gamma(a),
        hint=Not(IsNonPositiveInteger(a)))
    add(Abs(b - t)**(-a), [1 - a], [(1 - a)/2], [0], [(1 - a)/2], t/b,
        2*sin(pi*a/2)*gamma(1 - a)*Abs(b)**(-a), re(a) < 1)
    add((t**a - b**a)/(t - b), [0, a], [], [0, a], [], t/b,
        b**(a - 1)*sin(a*pi)/pi)

    # 12
    def A1(r, sign, nu):
        return pi**Rational(-1, 2)*(-sign*nu/2)**(1 - 2*r)

    def tmpadd(r, sgn):
        # XXX the a**2 is bad for matching
        add((sqrt(a**2 + t) + sgn*a)**b/(a**2 + t)**r,
            [(1 + b)/2, 1 - 2*r + b/2], [],
            [(b - sgn*b)/2], [(b + sgn*b)/2], t/a**2,
            a**(b - 2*r)*A1(r, sgn, b))
    tmpadd(0, 1)
    tmpadd(0, -1)
    tmpadd(S.Half, 1)
    tmpadd(S.Half, -1)

    # 13
    def tmpadd(r, sgn):
        # 定义一个临时函数 tmpadd，接受两个参数 r 和 sgn
        add((sqrt(a + p*z**q) + sgn*sqrt(p)*z**(q/2))**b/(a + p*z**q)**r,
            [1 - r + sgn*b/2], [1 - r - sgn*b/2], [0, S.Half], [],
            p*z**q/a, a**(b/2 - r)*A1(r, sgn, b))
        # 在特定的参数下调用 add 函数

    tmpadd(0, 1)
    # 调用 tmpadd 函数，参数为 0 和 1
    tmpadd(0, -1)
    # 调用 tmpadd 函数，参数为 0 和 -1
    tmpadd(S.Half, 1)
    # 调用 tmpadd 函数，参数为 S.Half 和 1
    tmpadd(S.Half, -1)
    # 调用 tmpadd 函数，参数为 S.Half 和 -1
    # (those after look obscure)

    # Section 8.4.3
    add(exp(polar_lift(-1)*t), [], [], [0], [])
    # 调用 add 函数，计算指数函数 exp(polar_lift(-1)*t)，传入参数和空列表

    # TODO can do sin^n, sinh^n by expansion ... where?
    # 8.4.4 (hyperbolic functions)
    add(sinh(t), [], [1], [S.Half], [1, 0], t**2/4, pi**Rational(3, 2))
    # 调用 add 函数，计算双曲正弦函数 sinh(t)，传入参数和具体值

    add(cosh(t), [], [S.Half], [0], [S.Half, S.Half], t**2/4, pi**Rational(3, 2))
    # 调用 add 函数，计算双曲余弦函数 cosh(t)，传入参数和具体值

    # Section 8.4.5
    # TODO can do t + a. but can also do by expansion... (XXX not really)
    add(sin(t), [], [], [S.Half], [0], t**2/4, sqrt(pi))
    # 调用 add 函数，计算正弦函数 sin(t)，传入参数和具体值

    add(cos(t), [], [], [0], [S.Half], t**2/4, sqrt(pi))
    # 调用 add 函数，计算余弦函数 cos(t)，传入参数和具体值

    # Section 8.4.6 (sinc function)
    add(sinc(t), [], [], [0], [Rational(-1, 2)], t**2/4, sqrt(pi)/2)
    # 调用 add 函数，计算 sinc 函数，传入参数和具体值

    # Section 8.5.5
    def make_log1(subs):
        # 定义一个函数 make_log1，接受 subs 作为参数
        N = subs[n]
        # 从 subs 中获取 N
        return [(S.NegativeOne**N*factorial(N),
                 meijerg([], [1]*(N + 1), [0]*(N + 1), [], t))]
        # 返回一个列表，包含一个元组，其中包含根据给定参数计算得到的值

    def make_log2(subs):
        # 定义一个函数 make_log2，接受 subs 作为参数
        N = subs[n]
        # 从 subs 中获取 N
        return [(factorial(N),
                 meijerg([1]*(N + 1), [], [], [0]*(N + 1), t))]
        # 返回一个列表，包含一个元组，其中包含根据给定参数计算得到的值

    # TODO these only hold for positive p, and can be made more general
    #      but who uses log(x)*Heaviside(a-x) anyway ...
    # TODO also it would be nice to derive them recursively ...
    addi(log(t)**n*Heaviside(1 - t), make_log1, True)
    # 调用 addi 函数，计算 log(t)**n * Heaviside(1 - t)，使用 make_log1 函数生成项

    addi(log(t)**n*Heaviside(t - 1), make_log2, True)
    # 调用 addi 函数，计算 log(t)**n * Heaviside(t - 1)，使用 make_log2 函数生成项

    def make_log3(subs):
        # 定义一个函数 make_log3，接受 subs 作为参数
        return make_log1(subs) + make_log2(subs)
        # 返回 make_log1 和 make_log2 函数生成的结果的合并

    addi(log(t)**n, make_log3, True)
    # 调用 addi 函数，计算 log(t)**n，使用 make_log3 函数生成项

    addi(log(t + a),
         constant(log(a)) + [(S.One, meijerg([1, 1], [], [1], [0], t/a))],
         True)
    # 调用 addi 函数，计算 log(t + a)，传入常数和具体值作为参数

    addi(log(Abs(t - a)), constant(log(Abs(a))) +
         [(pi, meijerg([1, 1], [S.Half], [1], [0, S.Half], t/a))],
         True)
    # 调用 addi 函数，计算 log(Abs(t - a))，传入常数和具体值作为参数

    # Sections 8.4.9-10
    # TODO

    # Section 8.4.11
    addi(Ei(t),
         constant(-S.ImaginaryUnit*pi) + [(S.NegativeOne, meijerg([], [1], [0, 0], [],
                  t*polar_lift(-1)))],
         True)
    # 调用 addi 函数，计算 Ei(t)，传入常数和具体值作为参数

    # Section 8.4.12
    add(Si(t), [1], [], [S.Half], [0, 0], t**2/4, sqrt(pi)/2)
    # 调用 add 函数，计算 Sine integral Si(t)，传入参数和具体值

    add(Ci(t), [], [1], [0, 0], [S.Half], t**2/4, -sqrt(pi)/2)
    # 调用 add 函数，计算 Cosine integral Ci(t)，传入参数和具体值

    # Section 8.4.13
    add(Shi(t), [S.Half], [], [0], [Rational(-1, 2), Rational(-1, 2)], polar_lift(-1)*t**2/4,
        t*sqrt(pi)/4)
    # 调用 add 函数，计算 Sinh integral Shi(t)，传入参数和具体值

    add(Chi(t), [], [S.Half, 1], [0, 0], [S.Half, S.Half], t**2/4, -
        pi**S('3/2')/2)
    # 调用 add 函数，计算 Cosh integral Chi(t)，传入参数和具体值

    # generalized exponential integral
    add(expint(a, t), [], [a], [a - 1, 0], [], t)
    # 调用 add 函数，计算 generalized exponential integral expint(a, t)，传入参数和具体值

    # Section 8.4.14
    add(erf(t), [1], [], [S.Half], [0], t**2, 1/sqrt(pi))
    # 调用 add 函数，计算 Error function erf(t)，传入参数和具体值

    add(erfc(t), [], [1], [0, S.Half], [], t**2, 1/sqrt(pi))
    # 调用 add 函数，计算 Complementary error function erfc(t)，传入参数和具体值
    # This formula adds the value of the erfi(t) function with specific coefficients and arguments
    # The erfi function is the imaginary error function
    add(erfi(t), [S.Half], [], [0], [Rational(-1, 2)], -t**2, t/sqrt(pi))

    # These lines add values for Fresnel integrals S(t) and C(t) with specific coefficients and arguments
    add(fresnels(t), [1], [], [Rational(3, 4)], [0, Rational(1, 4)], pi**2*t**4/16, S.Half)
    add(fresnelc(t), [1], [], [Rational(1, 4)], [0, Rational(3, 4)], pi**2*t**4/16, S.Half)

    ##### bessel-type functions #####

    # Section 8.4.19: Add the Bessel function of the first kind J_a(t) with specific coefficients and arguments
    add(besselj(a, t), [], [], [a/2], [-a/2], t**2/4)

    # Section 8.4.20: Add the Bessel function of the second kind Y_a(t) with specific coefficients and arguments
    add(bessely(a, t), [], [-(a + 1)/2], [a/2, -a/2], [-(a + 1)/2], t**2/4)

    # Section 8.4.21 ?
    # Section 8.4.22: Add the modified Bessel function of the first kind I_a(t) with specific coefficients and arguments
    add(besseli(a, t), [], [(1 + a)/2], [a/2], [-a/2, (1 + a)/2], t**2/4, pi)

    # Section 8.4.23: Add the modified Bessel function of the second kind K_a(t) with specific coefficients and arguments
    add(besselk(a, t), [], [], [a/2, -a/2], [], t**2/4, S.Half)

    # Complete elliptic integrals K(z) and E(z)
    # Add the complete elliptic integral of the first kind K(t)
    add(elliptic_k(t), [S.Half, S.Half], [], [0], [0], -t, S.Half)
    # Add the complete elliptic integral of the second kind E(t)
    add(elliptic_e(t), [S.Half, 3*S.Half], [], [0], [0], -t, Rational(-1, 2)/2)
####################################################################
# First some helper functions.
####################################################################

# 导入时间测量工具 timethis 函数
from sympy.utilities.timeutils import timethis
# 设置 timeit 为 timethis 函数，用于测量 meijerg 函数的执行时间
timeit = timethis('meijerg')


def _mytype(f: Basic, x: Symbol) -> tuple[type[Basic], ...]:
    """ 
    Create a hashable entity describing the type of f.

    Args:
        f (Basic): A SymPy expression to analyze.
        x (Symbol): The symbol with respect to which to analyze the expression.

    Returns:
        tuple[type[Basic], ...]: A tuple describing the types found in the expression.

    Notes:
        Returns an empty tuple if x is not a free symbol in f, otherwise it includes
        types of f and its arguments sorted by class_key.

    """
    def key(x: type[Basic]) -> tuple[int, int, str]:
        return x.class_key()

    if x not in f.free_symbols:
        return ()
    elif f.is_Function:
        return type(f),
    return tuple(sorted((t for a in f.args for t in _mytype(a, x)), key=key))


class _CoeffExpValueError(ValueError):
    """
    Exception raised by _get_coeff_exp, for internal use only.
    """
    pass


def _get_coeff_exp(expr, x):
    """
    When expr is known to be of the form c*x**b, with c and/or b possibly 1,
    return c, b.

    Examples
    ========

    >>> from sympy.abc import x, a, b
    >>> from sympy.integrals.meijerint import _get_coeff_exp
    >>> _get_coeff_exp(a*x**b, x)
    (a, b)
    >>> _get_coeff_exp(x, x)
    (1, 1)
    >>> _get_coeff_exp(2*x, x)
    (2, 1)
    >>> _get_coeff_exp(x**3, x)
    (1, 3)
    """
    from sympy.simplify import powsimp
    (c, m) = expand_power_base(powsimp(expr)).as_coeff_mul(x)
    if not m:
        return c, S.Zero
    [m] = m
    if m.is_Pow:
        if m.base != x:
            raise _CoeffExpValueError('expr not of form a*x**b')
        return c, m.exp
    elif m == x:
        return c, S.One
    else:
        raise _CoeffExpValueError('expr not of form a*x**b: %s' % expr)


def _exponents(expr, x):
    """
    Find the exponents of ``x`` (not including zero) in ``expr``.

    Examples
    ========

    >>> from sympy.integrals.meijerint import _exponents
    >>> from sympy.abc import x, y
    >>> from sympy import sin
    >>> _exponents(x, x)
    {1}
    >>> _exponents(x**2, x)
    {2}
    >>> _exponents(x**2 + x, x)
    {1, 2}
    >>> _exponents(x**3*sin(x + x**y) + 1/x, x)
    {-1, 1, 3, y}
    """
    def _exponents_(expr, x, res):
        if expr == x:
            res.update([1])
            return
        if expr.is_Pow and expr.base == x:
            res.update([expr.exp])
            return
        for argument in expr.args:
            _exponents_(argument, x, res)
    res = set()
    _exponents_(expr, x, res)
    return res


def _functions(expr, x):
    """ 
    Find the types of functions in expr, to estimate the complexity.

    Args:
        expr (Basic): A SymPy expression to analyze.
        x (Symbol): The symbol with respect to which to analyze the expression.

    Returns:
        set: A set of function types found in the expression.

    """
    return {e.func for e in expr.atoms(Function) if x in e.free_symbols}


def _find_splitting_points(expr, x):
    """
    Find numbers a such that a linear substitution x -> x + a would
    (hopefully) simplify expr.

    Examples
    ========

    >>> from sympy.integrals.meijerint import _find_splitting_points as fsp
    >>> from sympy import sin
    >>> from sympy.abc import x
    >>> fsp(x, x)
    {0}
    >>> fsp((x-1)**3, x)
    {1}
    >>> fsp(sin(x+3)*x, x)
    {-3, 0}
    """
    pass  # Placeholder for the function body, actual implementation is missing
    # 使用 SymPy 的 Wild 类创建两个通配符对象 p 和 q，用于匹配表达式中的符号，排除 x 变量
    p, q = [Wild(n, exclude=[x]) for n in 'pq']

    # 定义一个递归函数 compute_innermost，用于计算表达式中最内层的 p*x + q 形式的子表达式
    def compute_innermost(expr, res):
        # 如果 expr 不是 SymPy 的表达式对象，则直接返回
        if not isinstance(expr, Expr):
            return
        
        # 尝试将 expr 与 p*x + q 的模式进行匹配
        m = expr.match(p*x + q)
        
        # 如果成功匹配并且 m[p] 不为零，则将 -m[q]/m[p] 的结果添加到结果集合 res 中
        if m and m[p] != 0:
            res.add(-m[q]/m[p])
            return
        
        # 如果 expr 是原子表达式，则返回
        if expr.is_Atom:
            return
        
        # 递归处理 expr 的每个子表达式
        for argument in expr.args:
            compute_innermost(argument, res)
    
    # 创建一个空的集合 innermost，用于存储计算结果
    innermost = set()
    
    # 调用 compute_innermost 函数计算表达式 expr 中最内层的 p*x + q 形式的子表达式
    compute_innermost(expr, innermost)
    
    # 返回计算得到的最内层子表达式集合
    return innermost
def _split_mul(f, x):
    """
    将表达式 ``f`` 分解为 fac、po、g，其中 fac 是常数因子，
    po = x**s （这里 s 是独立于 s 的一些指数），而 g 是“剩余部分”。

    Examples
    ========

    >>> from sympy.integrals.meijerint import _split_mul
    >>> from sympy import sin
    >>> from sympy.abc import s, x
    >>> _split_mul((3*x)**s*sin(x**2)*x, x)
    (3**s, x*x**s, sin(x**2))
    """
    fac = S.One  # 初始化常数因子为 1
    po = S.One   # 初始化幂部分为 1
    g = S.One    # 初始化剩余部分为 1
    f = expand_power_base(f)  # 展开 f 中的幂表达式

    args = Mul.make_args(f)  # 将 f 分解为乘积项
    for a in args:
        if a == x:
            po *= x  # 如果当前项是 x，更新 po
        elif x not in a.free_symbols:
            fac *= a  # 如果当前项不含 x 的符号，则更新 fac
        else:
            if a.is_Pow and x not in a.exp.free_symbols:
                # 如果当前项是幂且指数不含 x 的符号
                c, t = a.base.as_coeff_mul(x)
                if t != (x,):
                    c, t = expand_mul(a.base).as_coeff_mul(x)
                if t == (x,):
                    po *= x**a.exp
                    # 更新 fac，应用非极化操作
                    fac *= unpolarify(polarify(c**a.exp, subs=False))
                    continue
            g *= a  # 否则更新剩余部分 g

    return fac, po, g
    # 返回一个复数 C 和一个 Meijer G 函数的计算结果
    return C, meijerg(
        # 将 g.an 扩展到 n 维度
        inflate(g.an, n),
        # 将 g.aother 扩展到 n 维度
        inflate(g.aother, n),
        # 将 g.bm 扩展到 n 维度
        inflate(g.bm, n),
        # 将 g.bother 扩展到 n 维度
        inflate(g.bother, n),
        # 计算 g.argument 的 n 次方乘以 n 的 n*v 次方
        g.argument**n * n**(n*v)
    )
# 将函数 g 的 G 函数转换为其逆参数的版本 (即 G(1/x) -> G'(x))
def _flip_g(g):
    """ Turn the G function into one of inverse argument
        (i.e. G(1/x) -> G'(x)) """
    # 参见 [L]，第 5.2 节
    
    # 定义一个函数 tr，将列表中的每个元素替换为 1 减去原始值
    def tr(l):
        return [1 - a for a in l]
    
    # 使用 tr 函数对 g 的一些属性进行转换，得到新的参数用于调用 meijerg 函数
    return meijerg(tr(g.bm), tr(g.bother), tr(g.an), tr(g.aother), 1/g.argument)


# 将函数 g 的 G 函数扩展为另一个函数 H，该函数具有相同的定义方式，但积分因子是 d/Gamma(a*s)（按照常规轮廓约定）
def _inflate_fox_h(g, a):
    r"""
    Let d denote the integrand in the definition of the G function ``g``.
    Consider the function H which is defined in the same way, but with
    integrand d/Gamma(a*s) (contour conventions as usual).

    If ``a`` is rational, the function H can be written as C*G, for a constant C
    and a G-function G.

    This function returns C, G.
    """
    # 如果 a 小于 0，则对 g 进行反转并传入 -a
    if a < 0:
        return _inflate_fox_h(_flip_g(g), -a)
    
    # 使用 S 类创建有理数 p 和 q
    p = S(a.p)
    q = S(a.q)
    
    # 使用 s->qs 的替换，即通过 q 扩展 g。这会留下一个额外的 Gamma(p*s) 因子，我们使用高斯乘法定理处理它
    D, g = _inflate_g(g, q)
    z = g.argument
    
    # 根据公式进行调整
    D /= (2*pi)**((1 - p)/2)*p**Rational(-1, 2)
    z /= p**p
    bs = [(n + 1)/p for n in range(p)]
    
    # 返回结果
    return D, meijerg(g.an, g.aother, g.bm, list(g.bother) + bs, z)


# 缓存用于表示虚数的符号
_dummies: dict[tuple[str, str], Dummy]  = {}


# 返回一个虚数。如果请求相同的 token+name 超过一次，并且它不在 expr 中，则返回相同的虚数。
# 这是为了提高缓存效率。
def _dummy(name, token, expr, **kwargs):
    """
    Return a dummy. This will return the same dummy if the same token+name is
    requested more than once, and it is not already in expr.
    This is for being cache-friendly.
    """
    d = _dummy_(name, token, **kwargs)
    if d in expr.free_symbols:
        return Dummy(name, **kwargs)
    return d


# 返回一个与 name 和 token 关联的虚数。与全局声明具有相同效果。
def _dummy_(name, token, **kwargs):
    """
    Return a dummy associated to name and token. Same effect as declaring
    it globally.
    """
    global _dummies
    if not (name, token) in _dummies:
        _dummies[(name, token)] = Dummy(name, **kwargs)
    return _dummies[(name, token)]


# 检查函数 f(x)，当使用正实数上的 G 函数表示时，是否实际上几乎处处与 G 函数相符
def _is_analytic(f, x):
    """ Check if f(x), when expressed using G functions on the positive reals,
        will in fact agree with the G functions almost everywhere """
    # 检查表达式中是否存在 Heaviside、Abs 函数的符号
    return not any(x in expr.free_symbols for expr in f.atoms(Heaviside, Abs))


# 对条件 cond 进行简单的逻辑简化
def _condsimp(cond, first=True):
    """
    Do naive simplifications on ``cond``.

    Explanation
    ===========

    Note that this routine is completely ad-hoc, simplification rules being
    added as need arises rather than following any logical pattern.

    Examples
    ========

    >>> from sympy.integrals.meijerint import _condsimp as simp
    >>> from sympy import Or, Eq
    >>> from sympy.abc import x, y
    >>> simp(Or(x < y, Eq(x, y)))
    x <= y
    """
    if first:
        # 使用 _canonical_coeff 函数替换 cond 中的关系运算符
        cond = cond.replace(lambda _: _.is_Relational, _canonical_coeff)
        first = False
    if not isinstance(cond, BooleanFunction):
        return cond
    p, q, r = symbols('p q r', cls=Wild)
    # transforms tests use 0, 4, 5 and 11-14
    # meijer tests use 0, 2, 11, 14
    # joint_rv uses 6, 7
    rules = [
        (Or(p < q, Eq(p, q)), p <= q),  # Rule 0: If p is less than q or p equals q, then p is less than or equal to q.
        # The next two obviously are instances of a general pattern, but it is
        # easier to spell out the few cases we care about.
        (And(Abs(arg(p)) <= pi, Abs(arg(p) - 2*pi) <= pi),
         Eq(arg(p) - pi, 0)),  # Rule 1: If absolute value of arg(p) is less than or equal to pi and also arg(p) - 2*pi is less than or equal to pi, then arg(p) - pi equals 0.
        (And(Abs(2*arg(p) + pi) <= pi, Abs(2*arg(p) - pi) <= pi),
         Eq(arg(p), 0)),  # Rule 2: If absolute value of 2*arg(p) + pi is less than or equal to pi and also 2*arg(p) - pi is less than or equal to pi, then arg(p) equals 0.
        (And(Abs(2*arg(p) + pi) < pi, Abs(2*arg(p) - pi) <= pi),
         S.false),  # Rule 3: If absolute value of 2*arg(p) + pi is less than pi and absolute value of 2*arg(p) - pi is less than or equal to pi, then it is false.
        (And(Abs(arg(p) - pi/2) <= pi/2, Abs(arg(p) + pi/2) <= pi/2),
         Eq(arg(p), 0)),  # Rule 4: If absolute value of arg(p) - pi/2 is less than or equal to pi/2 and absolute value of arg(p) + pi/2 is less than or equal to pi/2, then arg(p) equals 0.
        (And(Abs(arg(p) - pi/2) <= pi/2, Abs(arg(p) + pi/2) < pi/2),
         S.false),  # Rule 5: If absolute value of arg(p) - pi/2 is less than or equal to pi/2 and absolute value of arg(p) + pi/2 is less than pi/2, then it is false.
        (And(Abs(arg(p**2/2 + 1)) < pi, Ne(Abs(arg(p**2/2 + 1)), pi)),
         S.true),  # Rule 6: If absolute value of arg(p**2/2 + 1) is less than pi and it is not equal to pi, then it is true.
        (Or(Abs(arg(p**2/2 + 1)) < pi, Ne(1/(p**2/2 + 1), 0)),
         S.true),  # Rule 7: If absolute value of arg(p**2/2 + 1) is less than pi or 1/(p**2/2 + 1) is not equal to 0, then it is true.
        (And(Abs(unbranched_argument(p)) <= pi,
           Abs(unbranched_argument(exp_polar(-2*pi*S.ImaginaryUnit)*p)) <= pi),
         Eq(unbranched_argument(exp_polar(-S.ImaginaryUnit*pi)*p), 0)),  # Rule 8: If absolute value of unbranched_argument(p) is less than or equal to pi and absolute value of unbranched_argument(exp_polar(-2*pi*S.ImaginaryUnit)*p) is less than or equal to pi, then unbranched_argument(exp_polar(-S.ImaginaryUnit*pi)*p) equals 0.
        (And(Abs(unbranched_argument(p)) <= pi/2,
           Abs(unbranched_argument(exp_polar(-pi*S.ImaginaryUnit)*p)) <= pi/2),
         Eq(unbranched_argument(exp_polar(-S.ImaginaryUnit*pi/2)*p), 0)),  # Rule 9: If absolute value of unbranched_argument(p) is less than or equal to pi/2 and absolute value of unbranched_argument(exp_polar(-pi*S.ImaginaryUnit)*p) is less than or equal to pi/2, then unbranched_argument(exp_polar(-S.ImaginaryUnit*pi/2)*p) equals 0.
        (Or(p <= q, And(p < q, r)), p <= q),  # Rule 10: If p is less than or equal to q or if p is less than q and r is true, then p is less than or equal to q.
        (Ne(p**2, 1) & (p**2 > 1), p**2 > 1),  # Rule 11: If p**2 is not equal to 1 and p**2 is greater than 1, then p**2 is greater than 1.
        (Ne(1/p, 1) & (cos(Abs(arg(p)))*Abs(p) > 1), Abs(p) > 1),  # Rule 12: If 1/p is not equal to 1 and cosine of absolute value of arg(p) times absolute value of p is greater than 1, then absolute value of p is greater than 1.
        (Ne(p, 2) & (cos(Abs(arg(p)))*Abs(p) > 2), Abs(p) > 2),  # Rule 13: If p is not equal to 2 and cosine of absolute value of arg(p) times absolute value of p is greater than 2, then absolute value of p is greater than 2.
        ((Abs(arg(p)) < pi/2) & (cos(Abs(arg(p)))*sqrt(Abs(p**2)) > 1), p**2 > 1),  # Rule 14: If absolute value of arg(p) is less than pi/2 and cosine of absolute value of arg(p) times square root of absolute value of p**2 is greater than 1, then p**2 is greater than 1.
    ]
    cond = cond.func(*[_condsimp(_, first) for _ in cond.args])
    change = True
    # 当存在改变时循环执行以下代码块
    while change:
        # 设置标志变量为 False，用于控制循环
        change = False
        # 遍历规则列表中的规则，irule 是索引，(fro, to) 是元组解包得到的规则
        for irule, (fro, to) in enumerate(rules):
            # 如果 fro 函数不等于 cond 函数，则继续下一次循环
            if fro.func != cond.func:
                continue
            # 遍历 cond 的参数列表，n 是索引，arg1 是参数
            for n, arg1 in enumerate(cond.args):
                # 如果 r 在 fro 的第一个参数的自由符号中
                if r in fro.args[0].free_symbols:
                    # 使用 arg1 匹配 fro 的第二个参数，设置 num 为 1
                    m = arg1.match(fro.args[1])
                    num = 1
                else:
                    # 否则设置 num 为 0，并使用 arg1 匹配 fro 的第一个参数
                    num = 0
                    m = arg1.match(fro.args[0])
                # 如果匹配不成功，则继续下一次循环
                if not m:
                    continue
                # 生成 otherargs 列表，剔除匹配后的参数
                otherargs = [x.subs(m) for x in fro.args[:num] + fro.args[num + 1:]]
                # 初始化 otherlist 为包含 n 的列表
                otherlist = [n]
                # 遍历 otherargs
                for arg2 in otherargs:
                    # 遍历 cond 的参数列表，k 是索引，arg3 是参数
                    for k, arg3 in enumerate(cond.args):
                        # 如果 k 在 otherlist 中，则继续下一次循环
                        if k in otherlist:
                            continue
                        # 如果 arg2 等于 arg3，则将 k 添加到 otherlist 中，并跳出循环
                        if arg2 == arg3:
                            otherlist += [k]
                            break
                        # 如果 arg3 是 And 类型，并且 arg2.args[1] 等于 r，同时 arg2 是 And 类型，
                        # 并且 arg2.args[0] 在 arg3.args 中，则将 k 添加到 otherlist 中，并跳出循环
                        if isinstance(arg3, And) and arg2.args[1] == r and \
                                isinstance(arg2, And) and arg2.args[0] in arg3.args:
                            otherlist += [k]
                            break
                        # 如果 arg3 是 And 类型，并且 arg2.args[0] 等于 r，同时 arg2 是 And 类型，
                        # 并且 arg2.args[1] 在 arg3.args 中，则将 k 添加到 otherlist 中，并跳出循环
                        if isinstance(arg3, And) and arg2.args[0] == r and \
                                isinstance(arg2, And) and arg2.args[1] in arg3.args:
                            otherlist += [k]
                            break
                # 如果 otherlist 的长度不等于 otherargs 的长度加一，则继续下一次循环
                if len(otherlist) != len(otherargs) + 1:
                    continue
                # 生成 newargs 列表，包含未包含在 otherlist 中的参数，并添加 to.subs(m)
                newargs = [arg_ for (k, arg_) in enumerate(cond.args)
                           if k not in otherlist] + [to.subs(m)]
                # 如果 SYMPY_DEBUG 为真，且 irule 不在指定的索引范围内，则打印使用的新规则信息
                if SYMPY_DEBUG:
                    if irule not in (0, 2, 4, 5, 6, 7, 11, 12, 13, 14):
                        print('used new rule:', irule)
                # 使用 newargs 创建一个新的 cond 函数对象
                cond = cond.func(*newargs)
                # 设置 change 为 True，表示发生了改变
                change = True
                # 退出当前 for 循环
                break

    # 最后的调整
    def rel_touchup(rel):
        # 如果关系操作符不是 '=='，或者 rhs 不等于 0，则直接返回 rel
        if rel.rel_op != '==' or rel.rhs != 0:
            return rel

        # 处理形如 Eq(*, 0) 的情况
        LHS = rel.lhs
        # 尝试匹配 LHS 是 arg(p)**q 的模式
        m = LHS.match(arg(p)**q)
        if not m:
            # 如果匹配不成功，则尝试匹配 unbranched_argument(polar_lift(p)**q) 的模式
            m = LHS.match(unbranched_argument(polar_lift(p)**q))
        if not m:
            # 如果仍然匹配不成功，且 LHS 是 periodic_argument 类型，并且 LHS 的第一个参数不是极坐标，
            # LHS 的第二个参数是正无穷，则返回 (LHS.args[0] > 0)
            if isinstance(LHS, periodic_argument) and not LHS.args[0].is_polar \
                    and LHS.args[1] is S.Infinity:
                return (LHS.args[0] > 0)
            # 否则返回 rel
            return rel
        # 如果匹配成功，则返回 (m[p] > 0)
        return (m[p] > 0)
    # 使用 rel_touchup 函数替换 cond 中所有满足 _.is_Relational 的表达式
    cond = cond.replace(lambda _: _.is_Relational, rel_touchup)
    # 如果 SYMPY_DEBUG 为真，则打印 _condsimp: cond
    if SYMPY_DEBUG:
        print('_condsimp: ', cond)
    # 返回最终的 cond 结果
    return cond
def _eval_cond(cond):
    """ Re-evaluate the conditions. """
    # 如果条件是布尔类型直接返回
    if isinstance(cond, bool):
        return cond
    # 否则对条件进行简化处理
    return _condsimp(cond.doit())

####################################################################
# Now the "backbone" functions to do actual integration.
####################################################################


def _my_principal_branch(expr, period, full_pb=False):
    """ Bring expr nearer to its principal branch by removing superfluous
        factors.
        This function does *not* guarantee to yield the principal branch,
        to avoid introducing opaque principal_branch() objects,
        unless full_pb=True. """
    # 计算表达式 expr 的主分支，如果不要求完整的主分支，去除不必要的因子
    res = principal_branch(expr, period)
    if not full_pb:
        # 如果不要求完整的主分支，将结果中的主分支对象替换为相应的表达式
        res = res.replace(principal_branch, lambda x, y: x)
    return res


def _rewrite_saxena_1(fac, po, g, x):
    """
    Rewrite the integral fac*po*g dx, from zero to infinity, as
    integral fac*G, where G has argument a*x. Note po=x**s.
    Return fac, G.
    """
    # 获取 po 的系数和指数
    _, s = _get_coeff_exp(po, x)
    # 获取 g.argument 的系数和指数
    a, b = _get_coeff_exp(g.argument, x)
    # 获取 g 的周期
    period = g.get_period()
    # 计算 a 的主分支
    a = _my_principal_branch(a, period)

    # We substitute t = x**b.
    # 计算常数 C
    C = fac/(Abs(b)*a**((s + 1)/b - 1))
    # 吸收 (at)**((1 + s)/b - 1) 的一个因子

    def tr(l):
        return [a + (1 + s)/b - 1 for a in l]
    # 转换 g.an 的参数并使用变换后的参数计算 meijerg 函数
    return C, meijerg(tr(g.an), tr(g.aother), tr(g.bm), tr(g.bother),
                      a*x)


def _check_antecedents_1(g, x, helper=False):
    r"""
    Return a condition under which the mellin transform of g exists.
    Any power of x has already been absorbed into the G function,
    so this is just $\int_0^\infty g\, dx$.

    See [L, section 5.6.1]. (Note that s=1.)

    If ``helper`` is True, only check if the MT exists at infinity, i.e. if
    $\int_1^\infty g\, dx$ exists.
    """
    # 如果更新这些条件，请同时更新文档
    # 获取 g.argument 的系数和指数
    delta = g.delta
    eta, _ = _get_coeff_exp(g.argument, x)
    m, n, p, q = S([len(g.bm), len(g.an), len(g.ap), len(g.bq)])

    if p > q:
        def tr(l):
            return [1 - x for x in l]
        return _check_antecedents_1(meijerg(tr(g.bm), tr(g.bother),
                                            tr(g.an), tr(g.aother), x/eta),
                                    x)

    # 创建用于条件检查的临时列表
    tmp = [-re(b) < 1 for b in g.bm] + [1 < 1 - re(a) for a in g.an]
    cond_3 = And(*tmp)

    tmp += [-re(b) < 1 for b in g.bother]
    tmp += [1 < 1 - re(a) for a in g.aother]
    cond_3_star = And(*tmp)

    cond_4 = (-re(g.nu) + (q + 1 - p)/2 > q - p)

    def debug(*msg):
        _debug(*msg)

    def debugf(string, arg):
        _debugf(string, arg)

    debug('Checking antecedents for 1 function:')
    debugf('  delta=%s, eta=%s, m=%s, n=%s, p=%s, q=%s',
           (delta, eta, m, n, p, q))
    debugf('  ap = %s, %s', (list(g.an), list(g.aother)))
    debugf('  bq = %s, %s', (list(g.bm), list(g.bother)))
    debugf('  cond_3=%s, cond_3*=%s, cond_4=%s', (cond_3, cond_3_star, cond_4))
    # 初始化一个空列表 conds，用于存储所有的条件
    conds = []

    # case 1
    # 初始化一个空列表 case1，用于存储 case 1 的条件
    case1 = []
    # 定义临时条件 tmp1
    tmp1 = [1 <= n, p < q, 1 <= m]
    # 定义临时条件 tmp2
    tmp2 = [1 <= p, 1 <= m, Eq(q, p + 1), Not(And(Eq(n, 0), Eq(m, p + 1)))]
    # 定义临时条件 tmp3
    tmp3 = [1 <= p, Eq(q, p)]
    # 循环生成 tmp3 的附加条件
    for k in range(ceiling(delta/2) + 1):
        tmp3 += [Ne(Abs(unbranched_argument(eta)), (delta - 2*k)*pi)]
    # 定义临时条件 tmp
    tmp = [delta > 0, Abs(unbranched_argument(eta)) < delta*pi]
    # 定义额外的条件 extra
    extra = [Ne(eta, 0), cond_3]
    # 如果 helper 为真，则清空额外的条件
    if helper:
        extra = []
    # 将 tmp1, tmp2, tmp3 的条件组合起来，并加上 tmp 和 extra，加入 case1 列表
    for t in [tmp1, tmp2, tmp3]:
        case1 += [And(*(t + tmp + extra))]
    # 将 case1 加入 conds 列表
    conds += case1
    # 输出调试信息，显示 case 1 的条件
    debug('  case 1:', case1)

    # case 2
    # 设置额外的条件 extra
    extra = [cond_3]
    # 如果 helper 为真，则清空额外的条件
    if helper:
        extra = []
    # 定义 case 2 的条件
    case2 = [And(Eq(n, 0), p + 1 <= m, m <= q,
                 Abs(unbranched_argument(eta)) < delta*pi, *extra)]
    # 将 case 2 的条件加入 conds 列表
    conds += case2
    # 输出调试信息，显示 case 2 的条件
    debug('  case 2:', case2)

    # case 3
    # 设置额外的条件 extra
    extra = [cond_3, cond_4]
    # 如果 helper 为真，则清空额外的条件
    if helper:
        extra = []
    # 定义 case 3 的条件
    case3 = [And(p < q, 1 <= m, delta > 0, Eq(Abs(unbranched_argument(eta)), delta*pi),
                 *extra)]
    # 加入另一个 case 3 的条件
    case3 += [And(p <= q - 2, Eq(delta, 0), Eq(Abs(unbranched_argument(eta)), 0), *extra)]
    # 将 case 3 的条件加入 conds 列表
    conds += case3
    # 输出调试信息，显示 case 3 的条件
    debug('  case 3:', case3)

    # TODO altered cases 4-7
    # TODO 根据情况修改第 4 到第 7 个 case

    # extra case from wofram functions site:
    # (reproduced verbatim from Prudnikov, section 2.24.2)
    # https://functions.wolfram.com/HypergeometricFunctions/MeijerG/21/02/01/
    # 定义额外的特殊 case 条件
    case_extra = []
    case_extra += [Eq(p, q), Eq(delta, 0), Eq(unbranched_argument(eta), 0), Ne(eta, 0)]
    # 如果 helper 为假，则添加 cond_3 到额外条件
    if not helper:
        case_extra += [cond_3]
    # 计算序列 s
    s = []
    for a, b in zip(g.ap, g.bq):
        s += [b - a]
    # 添加额外的条件到 case_extra
    case_extra += [re(Add(*s)) < 0]
    # 将 case_extra 组合成一个整体条件
    case_extra = And(*case_extra)
    # 将 case_extra 加入 conds 列表
    conds += [case_extra]
    # 输出调试信息，显示额外的 case 的条件
    debug('  extra case:', [case_extra])

    # 定义第二个额外的 case 条件
    case_extra_2 = [And(delta > 0, Abs(unbranched_argument(eta)) < delta*pi)]
    # 如果 helper 为假，则添加 cond_3 到额外条件
    if not helper:
        case_extra_2 += [cond_3]
    # 将 case_extra_2 组合成一个整体条件
    case_extra_2 = And(*case_extra_2)
    # 将 case_extra_2 加入 conds 列表
    conds += [case_extra_2]
    # 输出调试信息，显示第二个额外的 case 的条件
    debug('  second extra case:', [case_extra_2])

    # TODO This leaves only one case from the three listed by Prudnikov.
    #      Investigate if these indeed cover everything; if so, remove the rest.
    # TODO 确保上述 case 能够覆盖所有情况，如果确实如此，则移除其余 case。

    # 返回所有条件的逻辑或结果
    return Or(*conds)
def _check_antecedents(g1, g2, x):
    """
    Check antecedents for the functions g1 and g2 at the point x.

    Explanation
    ===========

    This function verifies certain conditions for the given Meijer G functions
    g1 and g2 with respect to the argument x.

    Parameters
    ==========
    g1 : Meijer G function
        The first Meijer G function to check.
    g2 : Meijer G function
        The second Meijer G function to check.
    x : symbol
        The point at which to check the conditions.

    Returns
    =======
    None

    Notes
    =====
    This function checks if the exponents associated with g1 and g2 are
    non-negative and rational, and if so, it may proceed with further operations
    on these functions.

    Examples
    ========
    
    >>> from sympy.integrals.meijerint import _check_antecedents
    >>> from sympy.abc import s, t, m
    >>> from sympy import meijerg
    >>> g1 = meijerg([], [], [0], [], s*t)
    >>> g2 = meijerg([], [], [m/2], [-m/2], t**2/4)
    >>> _check_antecedents(g1, g2, t)
    """
    # Extract the coefficients and exponents associated with g1 and g2
    _, b1 = _get_coeff_exp(g1.argument, x)
    _, b2 = _get_coeff_exp(g2.argument, x)

    # Ensure the exponents are non-negative by flipping the sign if necessary
    if (b1 < 0) == True:
        b1 = -b1
        g1 = _flip_g(g1)
    if (b2 < 0) == True:
        b2 = -b2
        g2 = _flip_g(g2)

    # Check if exponents are rational; if not, return None
    if not b1.is_Rational or not b2.is_Rational:
        return
    ```python`
    """
    Return a condition under which the integral theorem applies.
    """
    
    # Yes, this is madness.
    # XXX TODO this is a testing *nightmare*
    # NOTE if you update these conditions, please update the documentation as well
    
    # The following conditions are found in
    # [P], Section 2.24.1
    #
    # They are also reproduced (verbatim!) at
    # https://functions.wolfram.com/HypergeometricFunctions/MeijerG/21/02/03/
    #
    # Note: k=l=r=alpha=1
    
    # Extract coefficients and exponents from the arguments of g1 and g2
    sigma, _ = _get_coeff_exp(g1.argument, x)
    omega, _ = _get_coeff_exp(g2.argument, x)
    
    # Determine lengths of various parameter lists
    s, t, u, v = S([len(g1.bm), len(g1.an), len(g1.ap), len(g1.bq)])
    m, n, p, q = S([len(g2.bm), len(g2.an), len(g2.ap), len(g2.bq)])
    
    # Calculate intermediate variables based on extracted parameters
    bstar = s + t - (u + v)/2
    cstar = m + n - (p + q)/2
    rho = g1.nu + (u - v)/2 + 1
    mu = g2.nu + (p - q)/2 + 1
    phi = q - p - (v - u)
    eta = 1 - (v - u) - mu - rho
    
    # Compute complex ratios involving pi and absolute values of arguments
    psi = (pi*(q - m - n) + Abs(unbranched_argument(omega)))/(q - p)
    theta = (pi*(v - s - t) + Abs(unbranched_argument(sigma)))/(v - u)
    
    # Debugging output for verification purposes
    _debug('Checking antecedents:')
    _debugf('  sigma=%s, s=%s, t=%s, u=%s, v=%s, b*=%s, rho=%s',
            (sigma, s, t, u, v, bstar, rho))
    _debugf('  omega=%s, m=%s, n=%s, p=%s, q=%s, c*=%s, mu=%s,',
            (omega, m, n, p, q, cstar, mu))
    _debugf('  phi=%s, eta=%s, psi=%s, theta=%s', (phi, eta, psi, theta))
    
    # Define conditions using helper functions and logical operations
    def _c1():
        for g in [g1, g2]:
            for i, j in itertools.product(g.an, g.bm):
                diff = i - j
                if diff.is_integer and diff.is_positive:
                    return False
        return True
    c1 = _c1()
    c2 = And(*[re(1 + i + j) > 0 for i in g1.bm for j in g2.bm])
    c3 = And(*[re(1 + i + j) < 1 + 1 for i in g1.an for j in g2.an])
    c4 = And(*[(p - q)*re(1 + i - 1) - re(mu) > Rational(-3, 2) for i in g1.an])
    c5 = And(*[(p - q)*re(1 + i) - re(mu) > Rational(-3, 2) for i in g1.bm])
    c6 = And(*[(u - v)*re(1 + i - 1) - re(rho) > Rational(-3, 2) for i in g2.an])
    c7 = And(*[(u - v)*re(1 + i) - re(rho) > Rational(-3, 2) for i in g2.bm])
    c8 = (Abs(phi) + 2*re((rho - 1)*(q - p) + (v - u)*(q - p) + (mu -
          1)*(v - u)) > 0)
    c9 = (Abs(phi) - 2*re((rho - 1)*(q - p) + (v - u)*(q - p) + (mu -
          1)*(v - u)) > 0)
    c10 = (Abs(unbranched_argument(sigma)) < bstar*pi)
    c11 = Eq(Abs(unbranched_argument(sigma)), bstar*pi)
    c12 = (Abs(unbranched_argument(omega)) < cstar*pi)
    c13 = Eq(Abs(unbranched_argument(omega)), cstar*pi)
    
    # Explanation of non-implemented condition based on external sources
    # and observed discrepancies with existing implementations
    # The following condition is *not* implemented as stated on the wolfram
    # function site. In the book of Prudnikov there is an additional part
    # (the And involving re()). However, I only have this book in russian, and
    # I don't read any russian. The following condition is what other people
    # have told me it means.
    # Worryingly, it is different from the condition implemented in REDUCE.
    # The REDUCE implementation:
    # https://reduce-algebra.svn.sourceforge.net/svnroot/reduce-algebra/trunk/packages/defint/definta.red
    # (search for tst14)
    # The Wolfram alpha version:
    # https://functions.wolfram.com/HypergeometricFunctions/MeijerG/21/02/03/03/0014/
    # 计算 z0，这里使用了复数单位的虚部进行计算
    z0 = exp(-(bstar + cstar)*pi*S.ImaginaryUnit)
    # 计算 zos，为 z0 乘以 omega 除以 sigma，未极化
    zos = unpolarify(z0*omega/sigma)
    # 计算 zso，为 z0 乘以 sigma 除以 omega，未极化
    zso = unpolarify(z0*sigma/omega)
    # 如果 zos 等于 1/zso，则执行以下条件
    if zos == 1/zso:
        # 设置 c14 的条件，包括 phi 等于 0，并且 bstar + cstar 小于等于 1，
        # 以及以下任一条件满足：
        # 1. zos 不等于 1 并且 re(mu + rho + v - u) 小于 1
        # 2. re(mu + rho + q - p) 小于 1
        c14 = And(Eq(phi, 0), bstar + cstar <= 1,
                  Or(Ne(zos, 1), re(mu + rho + v - u) < 1,
                     re(mu + rho + q - p) < 1))
    else:
        # 定义内部函数 _cond(z)，返回值为 True 如果 abs(arg(1-z)) < pi，
        # 避免 arg(0) 的情况
        def _cond(z):
            '''Returns True if abs(arg(1-z)) < pi, avoiding arg(0).

            Explanation
            ===========

            If ``z`` is 1 then arg is NaN. This raises a
            TypeError on `NaN < pi`. Previously this gave `False` so
            this behavior has been hardcoded here but someone should
            check if this NaN is more serious! This NaN is triggered by
            test_meijerint() in test_meijerint.py:
            `meijerint_definite(exp(x), x, 0, I)`
            '''
            return z != 1 and Abs(arg(1 - z)) < pi

        # 设置 c14 的条件，包括 phi 等于 0，并且 bstar - 1 + cstar 小于等于 0，
        # 以及以下任一条件满足：
        # 1. zos 不等于 1 并且 _cond(zos) 成立
        # 2. re(mu + rho + v - u) 小于 1 并且 zos 等于 1
        c14 = And(Eq(phi, 0), bstar - 1 + cstar <= 0,
                  Or(And(Ne(zos, 1), _cond(zos)),
                     And(re(mu + rho + v - u) < 1, Eq(zos, 1))))

        # 设置 c14_alt 的条件，包括 phi 等于 0，并且 cstar - 1 + bstar 小于等于 0，
        # 以及以下任一条件满足：
        # 1. zso 不等于 1 并且 _cond(zso) 成立
        # 2. re(mu + rho + q - p) 小于 1 并且 zso 等于 1
        c14_alt = And(Eq(phi, 0), cstar - 1 + bstar <= 0,
                      Or(And(Ne(zso, 1), _cond(zso)),
                         And(re(mu + rho + q - p) < 1, Eq(zso, 1))))

        # 合并 c14 和 c14_alt 条件，因为在我们的情况下 r=k=l=1，
        # c14_alt 等同于用 (g1, g2) = (g2, g1) 调用它。以下条件列出了所有情况
        # （即我们无需手动反转参数），并尝试所有对称情况。
        c14 = Or(c14, c14_alt)

    '''
    When `c15` is NaN (e.g. from `psi` being NaN as happens during
    'test_issue_4992' and/or `theta` is NaN as in 'test_issue_6253',
    both in `test_integrals.py`) the comparison to 0 formerly gave False
    whereas now an error is raised. To keep the old behavior, the value
    of NaN is replaced with False but perhaps a closer look at this condition
    should be made: XXX how should conditions leading to c15=NaN be handled?
    '''
    try:
        # 计算 lambda_c 的值，可能会引发 TypeError，例如如果 lambda_c 是 NaN
        lambda_c = (q - p)*Abs(omega)**(1/(q - p))*cos(psi) \
            + (v - u)*Abs(sigma)**(1/(v - u))*cos(theta)
        
        # 检查 lambda_c 是否大于 0，并赋值给 c15
        if _eval_cond(lambda_c > 0) != False:
            c15 = (lambda_c > 0)
        else:
            # 定义 lambda_s0 函数，并根据条件生成 lambda_s 的 Piecewise 表达式
            def lambda_s0(c1, c2):
                return c1*(q - p)*Abs(omega)**(1/(q - p))*sin(psi) \
                    + c2*(v - u)*Abs(sigma)**(1/(v - u))*sin(theta)
            
            lambda_s = Piecewise(
                ((lambda_s0(+1, +1)*lambda_s0(-1, -1)),
                 And(Eq(unbranched_argument(sigma), 0), Eq(unbranched_argument(omega), 0))),
                (lambda_s0(sign(unbranched_argument(omega)), +1)*lambda_s0(sign(unbranched_argument(omega)), -1),
                 And(Eq(unbranched_argument(sigma), 0), Ne(unbranched_argument(omega), 0))),
                (lambda_s0(+1, sign(unbranched_argument(sigma)))*lambda_s0(-1, sign(unbranched_argument(sigma))),
                 And(Ne(unbranched_argument(sigma), 0), Eq(unbranched_argument(omega), 0))),
                (lambda_s0(sign(unbranched_argument(omega)), sign(unbranched_argument(sigma))), True))
            
            # 构建临时列表 tmp，根据条件生成 c15 的值
            tmp = [lambda_c > 0,
                   And(Eq(lambda_c, 0), Ne(lambda_s, 0), re(eta) > -1),
                   And(Eq(lambda_c, 0), Eq(lambda_s, 0), re(eta) > 0)]
            c15 = Or(*tmp)
    except TypeError:
        # 如果出现 TypeError 异常，将 c15 设为 False
        c15 = False
    
    # 遍历条件列表，输出每个条件的调试信息
    for cond, i in [(c1, 1), (c2, 2), (c3, 3), (c4, 4), (c5, 5), (c6, 6),
                    (c7, 7), (c8, 8), (c9, 9), (c10, 10), (c11, 11),
                    (c12, 12), (c13, 13), (c14, 14), (c15, 15)]:
        _debugf('  c%s: %s', (i, cond))

    # 构建并返回条件列表 conds，用于最终的逻辑表达式 Or(*conds)
    conds = []

    # 定义辅助函数 pr，输出当前条件的调试信息
    def pr(count):
        _debugf('  case %s: %s', (count, conds[-1]))
    
    # 添加不同的条件到 conds 列表，并分别输出调试信息
    conds += [And(m*n*s*t != 0, bstar.is_positive is True, cstar.is_positive is True, c1, c2, c3, c10,
                  c12)]  # 1
    pr(1)
    conds += [And(Eq(u, v), Eq(bstar, 0), cstar.is_positive is True, sigma.is_positive is True, re(rho) < 1,
                  c1, c2, c3, c12)]  # 2
    pr(2)
    conds += [And(Eq(p, q), Eq(cstar, 0), bstar.is_positive is True, omega.is_positive is True, re(mu) < 1,
                  c1, c2, c3, c10)]  # 3
    pr(3)
    conds += [And(Eq(p, q), Eq(u, v), Eq(bstar, 0), Eq(cstar, 0),
                  sigma.is_positive is True, omega.is_positive is True, re(mu) < 1, re(rho) < 1,
                  Ne(sigma, omega), c1, c2, c3)]  # 4
    pr(4)
    conds += [And(Eq(p, q), Eq(u, v), Eq(bstar, 0), Eq(cstar, 0),
                  sigma.is_positive is True, omega.is_positive is True, re(mu + rho) < 1,
                  Ne(omega, sigma), c1, c2, c3)]  # 5
    pr(5)
    conds += [And(p > q, s.is_positive is True, bstar.is_positive is True, cstar >= 0,
                  c1, c2, c3, c5, c10, c13)]  # 6
    pr(6)
    conds += [And(p < q, t.is_positive is True, bstar.is_positive is True, cstar >= 0,
                  c1, c2, c3, c4, c10, c13)]  # 7
    pr(7)
    # 将一组条件添加到 conds 列表中，每个条件是一个 And 对象，包含多个逻辑表达式
    conds += [And(u > v, m.is_positive is True, cstar.is_positive is True, bstar >= 0,
                  c1, c2, c3, c7, c11, c12)]  # 8
    # 打印第 8 个条件的相关信息
    pr(8)
    conds += [And(u < v, n.is_positive is True, cstar.is_positive is True, bstar >= 0,
                  c1, c2, c3, c6, c11, c12)]  # 9
    # 打印第 9 个条件的相关信息
    pr(9)
    conds += [And(p > q, Eq(u, v), Eq(bstar, 0), cstar >= 0, sigma.is_positive is True,
                  re(rho) < 1, c1, c2, c3, c5, c13)]  # 10
    # 打印第 10 个条件的相关信息
    pr(10)
    conds += [And(p < q, Eq(u, v), Eq(bstar, 0), cstar >= 0, sigma.is_positive is True,
                  re(rho) < 1, c1, c2, c3, c4, c13)]  # 11
    # 打印第 11 个条件的相关信息
    pr(11)
    conds += [And(Eq(p, q), u > v, bstar >= 0, Eq(cstar, 0), omega.is_positive is True,
                  re(mu) < 1, c1, c2, c3, c7, c11)]  # 12
    # 打印第 12 个条件的相关信息
    pr(12)
    conds += [And(Eq(p, q), u < v, bstar >= 0, Eq(cstar, 0), omega.is_positive is True,
                  re(mu) < 1, c1, c2, c3, c6, c11)]  # 13
    # 打印第 13 个条件的相关信息
    pr(13)
    conds += [And(p < q, u > v, bstar >= 0, cstar >= 0,
                  c1, c2, c3, c4, c7, c11, c13)]  # 14
    # 打印第 14 个条件的相关信息
    pr(14)
    conds += [And(p > q, u < v, bstar >= 0, cstar >= 0,
                  c1, c2, c3, c5, c6, c11, c13)]  # 15
    # 打印第 15 个条件的相关信息
    pr(15)
    conds += [And(p > q, u > v, bstar >= 0, cstar >= 0,
                  c1, c2, c3, c5, c7, c8, c11, c13, c14)]  # 16
    # 打印第 16 个条件的相关信息
    pr(16)
    conds += [And(p < q, u < v, bstar >= 0, cstar >= 0,
                  c1, c2, c3, c4, c6, c9, c11, c13, c14)]  # 17
    # 打印第 17 个条件的相关信息
    pr(17)
    conds += [And(Eq(t, 0), s.is_positive is True, bstar.is_positive is True, phi.is_positive is True, c1, c2, c10)]  # 18
    # 打印第 18 个条件的相关信息
    pr(18)
    conds += [And(Eq(s, 0), t.is_positive is True, bstar.is_positive is True, phi.is_negative is True, c1, c3, c10)]  # 19
    # 打印第 19 个条件的相关信息
    pr(19)
    conds += [And(Eq(n, 0), m.is_positive is True, cstar.is_positive is True, phi.is_negative is True, c1, c2, c12)]  # 20
    # 打印第 20 个条件的相关信息
    pr(20)
    conds += [And(Eq(m, 0), n.is_positive is True, cstar.is_positive is True, phi.is_positive is True, c1, c3, c12)]  # 21
    # 打印第 21 个条件的相关信息
    pr(21)
    conds += [And(Eq(s*t, 0), bstar.is_positive is True, cstar.is_positive is True,
                  c1, c2, c3, c10, c12)]  # 22
    # 打印第 22 个条件的相关信息
    pr(22)
    conds += [And(Eq(m*n, 0), bstar.is_positive is True, cstar.is_positive is True,
                  c1, c2, c3, c10, c12)]  # 23
    # 打印第 23 个条件的相关信息
    pr(23)

    # 下面的情况来自 [Luke1969]。据我所知，它 *不* 在 Prudnikov 的范围内。
    # 假设 G1 和 G2 是两个 G 函数。假设从 0 到 a > 0 存在积分（这是容易的部分），
    # G1 在无穷远处是指数衰减的，并且 G2 的 Mellin 变换存在。
    # 那么积分存在。
    # 检查 G1 和 G2 的前提条件是否满足
    mt1_exists = _check_antecedents_1(g1, x, helper=True)
    mt2_exists = _check_antecedents_1(g2, x, helper=True)
    conds += [And(mt2_exists, Eq(t, 0), u < s, bstar.is_positive is True, c10, c1, c2, c3)]
    # 打印 'E1' 相关信息
    pr('E1')
    conds += [And(mt2_exists, Eq(s, 0), v < t, bstar.is_positive is True, c10, c1, c2, c3)]
    # 打印 'E2' 相关信息
    pr('E2')
    conds += [And(mt1_exists, Eq(n, 0), p < m, cstar.is_positive is True, c12, c1, c2, c3)]
    # 添加条件：mt1_exists 为真，n 等于 0，p 小于 m，cstar.is_positive 为真，依次为 c12, c1, c2, c3
    pr('E3')
    # 打印信息 'E3'
    conds += [And(mt1_exists, Eq(m, 0), q < n, cstar.is_positive is True, c12, c1, c2, c3)]
    # 添加条件：mt1_exists 为真，m 等于 0，q 小于 n，cstar.is_positive 为真，依次为 c12, c1, c2, c3
    pr('E4')
    # 打印信息 'E4'

    # 如果条件满足，则直接返回结果
    r = Or(*conds)
    # 将 conds 中所有条件用 Or 连接起来
    if _eval_cond(r) != False:
        return r

    conds += [And(m + n > p, Eq(t, 0), Eq(phi, 0), s.is_positive is True, bstar.is_positive is True, cstar.is_negative is True,
                  Abs(unbranched_argument(omega)) < (m + n - p + 1)*pi,
                  c1, c2, c10, c14, c15)]  # 24
    # 添加条件：m + n 大于 p，t 等于 0，phi 等于 0，s.is_positive 为真，bstar.is_positive 为真，cstar.is_negative 为真
    # 同时满足 Abs(unbranched_argument(omega)) 小于 (m + n - p + 1)*pi，依次为 c1, c2, c10, c14, c15
    pr(24)
    # 打印信息 '24'
    conds += [And(m + n > q, Eq(s, 0), Eq(phi, 0), t.is_positive is True, bstar.is_positive is True, cstar.is_negative is True,
                  Abs(unbranched_argument(omega)) < (m + n - q + 1)*pi,
                  c1, c3, c10, c14, c15)]  # 25
    # 添加条件：m + n 大于 q，s 等于 0，phi 等于 0，t.is_positive 为真，bstar.is_positive 为真，cstar.is_negative 为真
    # 同时满足 Abs(unbranched_argument(omega)) 小于 (m + n - q + 1)*pi，依次为 c1, c3, c10, c14, c15
    pr(25)
    # 打印信息 '25'
    conds += [And(Eq(p, q - 1), Eq(t, 0), Eq(phi, 0), s.is_positive is True, bstar.is_positive is True,
                  cstar >= 0, cstar*pi < Abs(unbranched_argument(omega)),
                  c1, c2, c10, c14, c15)]  # 26
    # 添加条件：p 等于 q - 1，t 等于 0，phi 等于 0，s.is_positive 为真，bstar.is_positive 为真，cstar 大于等于 0
    # 同时满足 cstar*pi 小于 Abs(unbranched_argument(omega))，依次为 c1, c2, c10, c14, c15
    pr(26)
    # 打印信息 '26'
    conds += [And(Eq(p, q + 1), Eq(s, 0), Eq(phi, 0), t.is_positive is True, bstar.is_positive is True,
                  cstar >= 0, cstar*pi < Abs(unbranched_argument(omega)),
                  c1, c3, c10, c14, c15)]  # 27
    # 添加条件：p 等于 q + 1，s 等于 0，phi 等于 0，t.is_positive 为真，bstar.is_positive 为真，cstar 大于等于 0
    # 同时满足 cstar*pi 小于 Abs(unbranched_argument(omega))，依次为 c1, c3, c10, c14, c15
    pr(27)
    # 打印信息 '27'
    conds += [And(p < q - 1, Eq(t, 0), Eq(phi, 0), s.is_positive is True, bstar.is_positive is True,
                  cstar >= 0, cstar*pi < Abs(unbranched_argument(omega)),
                  Abs(unbranched_argument(omega)) < (m + n - p + 1)*pi,
                  c1, c2, c10, c14, c15)]  # 28
    # 添加条件：p 小于 q - 1，t 等于 0，phi 等于 0，s.is_positive 为真，bstar.is_positive 为真，cstar 大于等于 0
    # 同时满足 Abs(unbranched_argument(omega)) 小于 (m + n - p + 1)*pi，依次为 c1, c2, c10, c14, c15
    pr(28)
    # 打印信息 '28'
    conds += [And(
        p > q + 1, Eq(s, 0), Eq(phi, 0), t.is_positive is True, bstar.is_positive is True, cstar >= 0,
                  cstar*pi < Abs(unbranched_argument(omega)),
                  Abs(unbranched_argument(omega)) < (m + n - q + 1)*pi,
                  c1, c3, c10, c14, c15)]  # 29
    # 添加条件：p 大于 q + 1，s 等于 0，phi 等于 0，t.is_positive 为真，bstar.is_positive 为真，cstar 大于等于 0
    # 同时满足 cstar*pi 小于 Abs(unbranched_argument(omega))，依次为 c1, c3, c10, c14, c15
    pr(29)
    # 打印信息 '29'
    conds += [And(Eq(n, 0), Eq(phi, 0), s + t > 0, m.is_positive is True, cstar.is_positive is True, bstar.is_negative is True,
                  Abs(unbranched_argument(sigma)) < (s + t - u + 1)*pi,
                  c1, c2, c12, c14, c15)]  # 30
    # 添加条件：n 等于 0，phi 等于 0，s + t 大于 0，m.is_positive 为真，cstar.is_positive 为真，bstar.is_negative 为真
    # 同时满足 Abs(unbranched_argument(sigma)) 小于 (s + t - u + 1)*pi，依次为 c1, c2, c12, c14, c15
    pr(30)
    # 打印信息 '30'
    conds += [And(Eq(m, 0), Eq(phi, 0), s + t > v, n.is_positive is True, cstar.is_positive is True, bstar.is_negative is True,
                  Abs(unbranched_argument(sigma)) < (s + t - v + 1)*pi,
                  c1, c3, c12, c14, c15)]  # 31
    # 添加条件：m 等于 0，phi 等于 0，s + t 大于 v，n.is_positive 为真，cstar.is_positive 为真，bstar.is_negative 为真
    # 同时满足 Abs(unbranched_argument(sigma)) 小于 (s + t - v + 1)*pi，依次为 c1, c3, c12, c14, c15
    pr(31)
    # 打印信息 '31'
    conds += [And(Eq(n, 0), Eq(phi, 0), Eq(u, v - 1), m.is_positive is True, cstar.is_positive is True,
                  bstar >= 0, bstar*pi < Abs(unbranched_argument(sigma)),
                  Abs(unbranched_argument(sigma)) < (bstar + 1)*pi,
                  c1, c2, c12, c14, c15)]  # 32
    # 添加条件：n 等于 0，phi 等于 0
    conds += [And(Eq(m, 0), Eq(phi, 0), Eq(u, v + 1), n.is_positive is True, cstar.is_positive is True,
                  bstar >= 0, bstar*pi < Abs(unbranched_argument(sigma)),
                  Abs(unbranched_argument(sigma)) < (bstar + 1)*pi,
                  c1, c3, c12, c14, c15)]  # 33
    # 添加条件 33：要求 m 等于 0，phi 等于 0，u 等于 v + 1，同时 n 和 cstar 是正数，bstar 大于等于 0，
    # bstar 乘以 pi 小于 sigma 的幅角的绝对值，sigma 的幅角的绝对值小于 (bstar + 1) 乘以 pi。
    pr(33)  # 打印编号 33

    conds += [And(
        Eq(n, 0), Eq(phi, 0), u < v - 1, m.is_positive is True, cstar.is_positive is True, bstar >= 0,
        bstar*pi < Abs(unbranched_argument(sigma)),
        Abs(unbranched_argument(sigma)) < (s + t - u + 1)*pi,
        c1, c2, c12, c14, c15)]  # 34
    # 添加条件 34：要求 n 等于 0，phi 等于 0，u 小于 v - 1，同时 m 和 cstar 是正数，bstar 大于等于 0，
    # bstar 乘以 pi 小于 sigma 的幅角的绝对值，sigma 的幅角的绝对值小于 (s + t - u + 1) 乘以 pi。
    pr(34)  # 打印编号 34

    conds += [And(
        Eq(m, 0), Eq(phi, 0), u > v + 1, n.is_positive is True, cstar.is_positive is True, bstar >= 0,
        bstar*pi < Abs(unbranched_argument(sigma)),
        Abs(unbranched_argument(sigma)) < (s + t - v + 1)*pi,
        c1, c3, c12, c14, c15)]  # 35
    # 添加条件 35：要求 m 等于 0，phi 等于 0，u 大于 v + 1，同时 n 和 cstar 是正数，bstar 大于等于 0，
    # bstar 乘以 pi 小于 sigma 的幅角的绝对值，sigma 的幅角的绝对值小于 (s + t - v + 1) 乘以 pi。
    pr(35)  # 打印编号 35

    return Or(*conds)
    # 返回 conds 中所有条件的逻辑或结果

    # NOTE An alternative, but as far as I can tell weaker, set of conditions
    #      can be found in [L, section 5.6.2].
    # 注意：我认为还有一组条件更弱，可以参考 [L, section 5.6.2]。
def _check_antecedents_inversion(g, x):
    """ Check antecedents for the Laplace inversion integral. """

    # 输出调试信息，检查反演的前提条件
    _debug('Checking antecedents for inversion:')
    
    # 提取 g 的参数
    z = g.argument
    
    # 获取 g.argument 在 x 上的系数和指数
    _, e = _get_coeff_exp(z, x)
    
    # 如果指数 e 小于 0，则翻转 G 函数
    if e < 0:
        _debug('  Flipping G.')
        # 我们希望当 |x| -> oo 时参数变大
        return _check_antecedents_inversion(_flip_g(g), x)

    # 定义一个函数，生成收敛性声明条件
    def statement_half(a, b, c, z, plus):
        coeff, exponent = _get_coeff_exp(z, x)
        a *= exponent
        b *= coeff**c
        c *= exponent
        
        # 计算两个复数 w 的值
        wp = b*exp(S.ImaginaryUnit*re(c)*pi/2)
        wm = b*exp(-S.ImaginaryUnit*re(c)*pi/2)
        
        # 根据条件生成可能的条件列表
        conds = []
        if plus:
            w = wp
        else:
            w = wm
        conds += [And(Or(Eq(b, 0), re(c) <= 0), re(a) <= -1)]
        conds += [And(Ne(b, 0), Eq(im(c), 0), re(c) > 0, re(w) < 0)]
        conds += [And(Ne(b, 0), Eq(im(c), 0), re(c) > 0, re(w) <= 0,
                      re(a) <= -1)]
        
        # 返回条件的逻辑或
        return Or(*conds)

    # 定义一个函数，为 z**a * exp(b*z**c) 提供收敛性声明
    def statement(a, b, c, z):
        """ Provide a convergence statement for z**a * exp(b*z**c). """
        return And(statement_half(a, b, c, z, True),
                   statement_half(a, b, c, z, False))

    # 从[L]，5.7-10 节获取符号
    m, n, p, q = S([len(g.bm), len(g.an), len(g.ap), len(g.bq)])
    tau = m + n - p
    nu = q - m - n
    rho = (tau - nu)/2
    sigma = q - p
    
    # 根据 sigma 的不同值确定 epsilon
    if sigma == 1:
        epsilon = S.Half
    elif sigma > 1:
        epsilon = 1
    else:
        epsilon = S.NaN
        
    # 计算 theta 和 delta
    theta = ((1 - sigma)/2 + Add(*g.bq) - Add(*g.ap))/sigma
    delta = g.delta
    # 调试输出，显示变量 m, n, p, q, tau, nu, rho, sigma 的当前值
    _debugf('  m=%s, n=%s, p=%s, q=%s, tau=%s, nu=%s, rho=%s, sigma=%s',
            (m, n, p, q, tau, nu, rho, sigma))
    
    # 调试输出，显示变量 epsilon, theta, delta 的当前值
    _debugf('  epsilon=%s, theta=%s, delta=%s', (epsilon, theta, delta))

    # 首先检查计算是否有效。
    if not (g.delta >= e/2 or (p >= 1 and p >= q)):
        # 如果条件不满足，输出调试信息并返回 False
        _debug('  Computation not valid for these parameters.')
        return False

    # 现在检查是否存在倒转积分。

    # 测试 "条件 A"
    for a, b in itertools.product(g.an, g.bm):
        if (a - b).is_integer and a > b:
            # 如果条件不满足，输出调试信息并返回 False
            _debug('  Not a valid G function.')
            return False

    # 有两种情况。如果 p >= q，可以直接使用 Slater 展开式
    # 如 [L], 5.2 (11)。特别要注意，即使某些参数相差整数，该公式也适用！
    # （因为 G 函数在其参数上是连续的）
    # 当 p < q 时，需要使用 [L], 5.10 的定理。

    if p >= q:
        # 使用渐近 Slater 展开式，输出调试信息并返回条件表达式
        _debug('  Using asymptotic Slater expansion.')
        return And(*[statement(a - 1, 0, 0, z) for a in g.an])

    def E(z):
        # 定义 E 函数，返回条件表达式
        return And(*[statement(a - 1, 0, 0, z) for a in g.an])

    def H(z):
        # 定义 H 函数，返回条件表达式
        return statement(theta, -sigma, 1/sigma, z)

    def Hp(z):
        # 定义 Hp 函数，返回条件表达式
        return statement_half(theta, -sigma, 1/sigma, z, True)

    def Hm(z):
        # 定义 Hm 函数，返回条件表达式
        return statement_half(theta, -sigma, 1/sigma, z, False)

    # [L], 章节 5.10
    conds = []
    # 定理 1 -- 根据前面的测试，p < q
    conds += [And(1 <= n, 1 <= m, rho*pi - delta >= pi/2, delta > 0,
                  E(z*exp(S.ImaginaryUnit*pi*(nu + 1))))]

    # 定理 2，陈述 (2) 和 (3)
    conds += [And(p + 1 <= m, m + 1 <= q, delta > 0, delta < pi/2, n == 0,
                  (m - p + 1)*pi - delta >= pi/2,
                  Hp(z*exp(S.ImaginaryUnit*pi*(q - m))),
                  Hm(z*exp(-S.ImaginaryUnit*pi*(q - m))))]

    # 定理 2，陈述 (5) -- 根据前面的测试，p < q
    conds += [And(m == q, n == 0, delta > 0,
                  (sigma + epsilon)*pi - delta >= pi/2, H(z))]

    # 定理 3，陈述 (6) 和 (7)
    conds += [And(Or(And(p <= q - 2, 1 <= tau, tau <= sigma/2),
                     And(p + 1 <= m + n, m + n <= (p + q)/2)),
                  delta > 0, delta < pi/2, (tau + 1)*pi - delta >= pi/2,
                  Hp(z*exp(S.ImaginaryUnit*pi*nu)),
                  Hm(z*exp(-S.ImaginaryUnit*pi*nu)))]

    # 定理 4，陈述 (10) 和 (11) -- 根据前面的测试，p < q
    conds += [And(1 <= m, rho > 0, delta > 0, delta + rho*pi < pi/2,
                  (tau + epsilon)*pi - delta >= pi/2,
                  Hp(z*exp(S.ImaginaryUnit*pi*nu)),
                  Hm(z*exp(-S.ImaginaryUnit*pi*nu)))]

    # Trivial case
    conds += [m == 0]

    # TODO
    # 定理 5 非常一般化
    # 定理 6 包含 q=p+1 的特殊情况

    # 返回 Or 条件表达式
    return Or(*conds)
def _int_inversion(g, x, t):
    """
    Compute the Laplace inversion integral, assuming the formula applies.
    """
    # 获取 g.argument 和 x 的系数和指数
    b, a = _get_coeff_exp(g.argument, x)
    # 调用 meijerg 函数，并通过 _inflate_fox_h 函数处理返回结果
    C, g = _inflate_fox_h(meijerg(g.an, g.aother, g.bm, g.bother, b/t**a), -a)
    # 返回计算结果
    return C/t*g


####################################################################
# Finally, the real meat.
####################################################################

# 初始化 _lookup_table 为 None
_lookup_table = None


@cacheit
@timeit
def _rewrite_single(f, x, recursive=True):
    """
    Try to rewrite f as a sum of single G functions of the form
    C*x**s*G(a*x**b), where b is a rational number and C is independent of x.
    We guarantee that result.argument.as_coeff_mul(x) returns (a, (x**b,))
    or (a, ()).
    Returns a list of tuples (C, s, G) and a condition cond.
    Returns None on failure.
    """
    # 导入所需的模块和函数
    from .transforms import (mellin_transform, inverse_mellin_transform,
        IntegralTransformError, MellinTransformStripError)

    # 声明全局变量 _lookup_table
    global _lookup_table
    # 如果 _lookup_table 为 None，则创建并填充 _lookup_table
    if not _lookup_table:
        _lookup_table = {}
        _create_lookup_table(_lookup_table)

    # 如果 f 是 meijerg 类型的对象
    if isinstance(f, meijerg):
        # 提取 f.argument 在 x 中的系数和单项式
        coeff, m = factor(f.argument, x).as_coeff_mul(x)
        # 如果单项式数量大于1，则返回 None
        if len(m) > 1:
            return None
        m = m[0]
        # 如果单项式是幂函数且基数不是 x 或指数不是有理数，则返回 None
        if m.is_Pow:
            if m.base != x or not m.exp.is_Rational:
                return None
        elif m != x:
            return None
        # 返回重写后的 meijerg 对象的列表和 True
        return [(1, 0, meijerg(f.an, f.aother, f.bm, f.bother, coeff*m))], True

    # 备份原始的 f
    f_ = f
    # 在 f 中替换所有 x 为 z
    f = f.subs(x, z)
    # 获取 f 的类型
    t = _mytype(f, z)
    # 如果 t 在查找表中
    if t in _lookup_table:
        # 获取与 t 对应的列表 l
        l = _lookup_table[t]
        # 遍历 l 中的每个元组 (formula, terms, cond, hint)
        for formula, terms, cond, hint in l:
            # 在 formula 中查找匹配项，返回替换字典 subs
            subs = f.match(formula, old=True)
            if subs:
                subs_ = {}
                # 构建新的替换字典 subs_，将匹配项中的表达式去极化并仅保留指数
                for fro, to in subs.items():
                    subs_[fro] = unpolarify(polarify(to, lift=True),
                                            exponents_only=True)
                subs = subs_
                # 如果 hint 不是布尔值，则对其进行替换
                if not isinstance(hint, bool):
                    hint = hint.subs(subs)
                # 如果 hint 为 False，则跳过当前循环
                if hint == False:
                    continue
                # 如果 cond 不是布尔值或 BooleanAtom，则对其进行替换并去极化
                if not isinstance(cond, (bool, BooleanAtom)):
                    cond = unpolarify(cond.subs(subs))
                # 如果 _eval_cond(cond) 返回 False，则跳过当前循环
                if _eval_cond(cond) == False:
                    continue
                # 如果 terms 不是列表，则根据替换字典 subs 计算 terms
                if not isinstance(terms, list):
                    terms = terms(subs)
                # 初始化结果列表 res
                res = []
                # 遍历 terms 中的每个因子 fac 和函数 g
                for fac, g in terms:
                    # 对 fac 进行替换并去极化，只保留指数
                    r1 = _get_coeff_exp(unpolarify(fac.subs(subs).subs(z, x),
                                                   exponents_only=True), x)
                    try:
                        # 对 g 进行替换并去极化其参数
                        g = g.subs(subs).subs(z, x)
                    except ValueError:
                        continue
                    # 注意：这些替换可能导致出现无穷大（oo）、复无穷大（zoo）和其他不合理的情况。
                    #      尽管这些情况理论上不应该出现，为了安全起见仍然进行检查。
                    if Tuple(*(r1 + (g,))).has(S.Infinity, S.ComplexInfinity, S.NegativeInfinity):
                        continue
                    # 对 g 进行梅耶尔函数化简
                    g = meijerg(g.an, g.aother, g.bm, g.bother,
                                unpolarify(g.argument, exponents_only=True))
                    # 将 (r1, g) 加入结果列表 res
                    res.append(r1 + (g,))
                # 如果 res 不为空，则返回 res 和 cond
                if res:
                    return res, cond

    # 尝试递归梅林变换
    if not recursive:
        return None
    # 输出调试信息
    _debug('Trying recursive Mellin transform method.')

    # 定义我的梅林变换函数 my_imt
    def my_imt(F, s, x, strip):
        """ 调用 simplify() 太慢且通常不太有用，因为大多数时候它只是以需要撤销的方式因式化。但有时它可以去除表面上的极点。 """
        try:
            # 尝试进行反梅林变换，as_meijerg=True 表示返回梅耶尔函数形式，needeval=True 表示需要进行求值
            return inverse_mellin_transform(F, s, x, strip,
                                            as_meijerg=True, needeval=True)
        except MellinTransformStripError:
            from sympy.simplify import simplify
            # 如果出现 MellinTransformStripError，则使用 simplify(cancel(expand(F))) 后再次尝试反梅林变换
            return inverse_mellin_transform(
                simplify(cancel(expand(F))), s, x, strip,
                as_meijerg=True, needeval=True)
    
    # 将 f_ 赋值给 f
    f = f_
    # 为 s 创建一个虚拟符号，用于单独重写
    s = _dummy('s', 'rewrite-single', f)
    # 为了避免无限递归，强制使用两个 g 函数的情况
    # 定义自定义积分器函数 my_integrator，接受函数 f 和变量 x 作为参数
    def my_integrator(f, x):
        # 调用 _meijerint_definite_4 函数进行积分计算，只考虑双精度计算
        r = _meijerint_definite_4(f, x, only_double=True)
        # 如果计算结果不为 None
        if r is not None:
            # 导入 sympy 库中的 hyperexpand 函数
            from sympy.simplify import hyperexpand
            # 将结果 r 解构为 res 和 cond
            res, cond = r
            # 对 res 进行超级展开，并取消极化
            res = _my_unpolarify(hyperexpand(res, rewrite='nonrepsmall'))
            # 返回条件判断的结果
            return Piecewise((res, cond),
                             (Integral(f, (x, S.Zero, S.Infinity)), True))
        # 如果计算结果为 None，则返回原始积分表达式
        return Integral(f, (x, S.Zero, S.Infinity))

    # 尝试进行梅林变换（Mellin transform）
    try:
        # 调用 mellin_transform 函数，计算变换后的 F、strip 和一个无关的 _
        F, strip, _ = mellin_transform(f, x, s, integrator=my_integrator,
                                       simplify=False, needeval=True)
        # 调用 my_imt 函数，计算 g
        g = my_imt(F, s, x, strip)
    # 如果出现积分变换错误
    except IntegralTransformError:
        # 将 g 设为 None
        g = None
    
    # 如果 g 是 None
    if g is None:
        # 尝试通过解析延拓找到表达式
        # （如果在表达式中已经有这个虚拟变量，那么再添加一个就没有意义）
        a = _dummy_('a', 'rewrite-single')
        # 如果 a 不在 f 的自由符号中，并且 f 是解析的关于 x 的函数
        if a not in f.free_symbols and _is_analytic(f, x):
            try:
                # 将 f 中的 x 替换为 a*x，并进行梅林变换
                F, strip, _ = mellin_transform(f.subs(x, a*x), x, s,
                                               integrator=my_integrator,
                                               needeval=True, simplify=False)
                # 计算变换后的 g，并将 a 替换为 1
                g = my_imt(F, s, x, strip).subs(a, 1)
            except IntegralTransformError:
                # 将 g 设为 None
                g = None
    
    # 如果 g 是 None 或者包含无限大、NaN 或者复无穷大
    if g is None or g.has(S.Infinity, S.NaN, S.ComplexInfinity):
        # 打印调试信息：递归梅林变换失败
        _debug('Recursive Mellin transform failed.')
        # 返回 None
        return None
    
    # 将 g 分解为其组成部分
    args = Add.make_args(g)
    res = []
    # 遍历 g 的每个部分 f
    for f in args:
        # 将 f 分解为系数 c 和乘积 m
        c, m = f.as_coeff_mul(x)
        # 如果乘积 m 的长度大于 1
        if len(m) > 1:
            # 抛出未实现的错误：意外的形式...
            raise NotImplementedError('Unexpected form...')
        # 取乘积 m 的唯一元素 g
        g = m[0]
        # 获取 g 的系数和指数
        a, b = _get_coeff_exp(g.argument, x)
        # 将结果添加到 res 中，使用 meijerg 函数进行计算
        res += [(c, 0, meijerg(g.an, g.aother, g.bm, g.bother,
                               unpolarify(polarify(
                                   a, lift=True), exponents_only=True)
                               *x**b))]
    
    # 打印调试信息：递归梅林变换成功
    _debug('Recursive Mellin transform worked:', g)
    # 返回结果列表 res 和 True
    return res, True
# 尝试将函数 f 重写为单个 G 函数的和或乘积形式，使用参数 a*x**b
# 返回 fac, po, g，其中 f = fac*po*g，fac 不依赖于 x，po = x**s
# g 是 _rewrite_single 的结果
def _rewrite1(f, x, recursive=True):
    fac, po, g = _split_mul(f, x)  # 分离 f 为 fac*po*g 的形式，其中 fac 和 po 是关于 x 的表达式，g 是剩余部分
    g = _rewrite_single(g, x, recursive)  # 尝试重写 g 为单个 G 函数形式
    if g:
        return fac, po, g[0], g[1]  # 返回重写后的表达式的分解形式


# 尝试将函数 f 重写为两个 G 函数的乘积形式，使用参数 a*x**b
# 返回 fac, po, g1, g2，其中 f = fac*po*g1*g2，fac 是独立于 x 的，po = x**s
# g1 和 g2 是 _rewrite_single 的结果
# 如果失败则返回 None
def _rewrite2(f, x):
    fac, po, g = _split_mul(f, x)  # 分离 f 为 fac*po*g 的形式，其中 fac 和 po 是关于 x 的表达式，g 是剩余部分
    if any(_rewrite_single(expr, x, False) is None for expr in _mul_args(g)):
        return None  # 如果 g 中的任何表达式不能被重写为单个 G 函数，则返回 None
    l = _mul_as_two_parts(g)  # 将 g 分解为两部分
    if not l:
        return None  # 如果无法成功分解，则返回 None
    # 按指定顺序对 l 进行排序
    l = list(ordered(l, [
        lambda p: max(len(_exponents(p[0], x)), len(_exponents(p[1], x))),
        lambda p: max(len(_functions(p[0], x)), len(_functions(p[1], x))),
        lambda p: max(len(_find_splitting_points(p[0], x)),
                      len(_find_splitting_points(p[1], x)))]))

    # 遍历所有可能的递归标志和两部分表达式对 (fac1, fac2)
    for recursive, (fac1, fac2) in itertools.product((False, True), l):
        g1 = _rewrite_single(fac1, x, recursive)  # 尝试将 fac1 重写为单个 G 函数形式
        g2 = _rewrite_single(fac2, x, recursive)  # 尝试将 fac2 重写为单个 G 函数形式
        if g1 and g2:
            cond = And(g1[1], g2[1])  # 构建 g1 和 g2 的条件约束
            if cond != False:  # 如果条件不是假，则返回重写后的表达式的分解形式
                return fac, po, g1[0], g2[0], cond


# 计算函数 f 的不定积分，重写为一个 G 函数形式
# 返回重写后的结果
def meijerint_indefinite(f, x):
    f = sympify(f)  # 将 f 转换为 SymPy 表达式
    results = []
    # 对于 f 中找到的所有分裂点 a，按默认排序键进行排序
    for a in sorted(_find_splitting_points(f, x) | {S.Zero}, key=default_sort_key):
        res = _meijerint_indefinite_1(f.subs(x, x + a), x)  # 尝试计算替换后的 f 的不定积分
        if not res:
            continue
        res = res.subs(x, x - a)  # 对结果进行替换 x -> x - a
        if _has(res, hyper, meijerg):  # 如果结果包含超函數
            results.append(res)  # 将结果添加到列表中
        else:
            return res  # 直接返回结果
    if f.has(HyperbolicFunction):  # 如果 f 中包含双曲函数
        _debug('Try rewriting hyperbolics in terms of exp.')
        rv = meijerint_indefinite(
            _rewrite_hyperbolics_as_exp(f), x)  # 尝试将双曲函数重写为指数形式
        if rv:
            if not isinstance(rv, list):
                from sympy.simplify.radsimp import collect
                return collect(factor_terms(rv), rv.atoms(exp))  # 收集指数形式的结果
            results.extend(rv)  # 将结果添加到列表中
    if results:
        return next(ordered(results))  # 返回排序后的结果列表中的第一个


# 辅助函数，不尝试进行任何替换，直接计算函数 f 的不定积分
def _meijerint_indefinite_1(f, x):
    _debug('Trying to compute the indefinite integral of', f, 'wrt', x)  # 输出调试信息
    from sympy.simplify import hyperexpand, powdenest

    gs = _rewrite1(f, x)  # 尝试将 f 重写为单个 G 函数形式
    if gs is None:
        # 如果 gs 为 None，则说明调用我们的代码将会执行 expand() 并重试
        return None

    fac, po, gl, cond = gs
    _debug(' could rewrite:', gs)
    # 初始化结果为零
    res = S.Zero
    # 遍历 gl 中的每个元组 (C, s, g)
    for C, s, g in gl:
        # 从 g.argument 中获取系数和指数，存入 a 和 b
        a, b = _get_coeff_exp(g.argument, x)
        # 从 po 中获取系数，存入 c，然后加上 s
        _, c = _get_coeff_exp(po, x)
        c += s

        # 进行替换 t = a*x**b，得到被积函数 fac*t**rho*g
        fac_ = fac * C * x**(1 + c) / b
        rho = (c + 1) / b

        # 使用 t**rho*G(params, t) = G(params + rho, t) 进行积分变换
        # [L, 第150页, 方程式 (4)]
        # 并且积分 G(params, t) dt = G(1, params+1, 0, t)
        #   （或者一个类似的表达式，1 和 0 交换... 选择一个产生定义良好函数的表达式）
        # [R, 第5节]
        # （注意，这个虚拟变量会立即消失，所以我们可以安全地传递 S.One 给 ``expr``。）
        t = _dummy('t', 'meijerint-indefinite', S.One)

        def tr(p):
            return [a + rho for a in p]
        # 如果 tr(g.bm) 中任意元素是整数且小于等于 0，则执行特定的 meijerg 调用
        if any(b.is_integer and (b <= 0) == True for b in tr(g.bm)):
            r = -meijerg(
                list(g.an), list(g.aother) + [1 - rho], list(g.bm) + [-rho], list(g.bother), t)
        else:
            r = meijerg(
                list(g.an) + [1 - rho], list(g.aother), list(g.bm), list(g.bother) + [-rho], t)
        
        # 大多数情况下期望的反导数在 x = 0 的附近是定义良好的。
        if b.is_extended_nonnegative and not f.subs(x, 0).has(S.NaN, S.ComplexInfinity):
            place = 0  # 假设我们可以在零处展开
        else:
            place = None
        # 使用 hyperexpand 展开积分结果 r.subs(t, a*x**b)，并在指定的位置展开
        r = hyperexpand(r.subs(t, a*x**b), place=place)

        # 现在进行回代
        # 注意：我们确实希望 x 的幂次合并。
        res += powdenest(fac_ * r, polar=True)

    # 定义内部函数 _clean，用于处理结果
    def _clean(res):
        """This multiplies out superfluous powers of x we created, and chops off
        constants:
        
            >> _clean(x*(exp(x)/x - 1/x) + 3)
            exp(x)
        
        cancel is used before mul_expand since it is possible for an
        expression to have an additive constant that does not become isolated
        with simple expansion. Such a situation was identified in issue 6369:
        
        Examples
        ========
        
        >>> from sympy import sqrt, cancel
        >>> from sympy.abc import x
        >>> a = sqrt(2*x + 1)
        >>> bad = (3*x*a**5 + 2*x - a**5 + 1)/a**2
        >>> bad.expand().as_independent(x)[0]
        0
        >>> cancel(bad).expand().as_independent(x)[0]
        1
        """
        # 使用 cancel 和 expand_mul 处理 res，deep=False 表示不深度展开
        res = expand_mul(cancel(res), deep=False)
        # 返回去除常数项后的结果
        return Add._from_args(res.as_coeff_add(x)[1])

    # 对结果 res 进行 piecewise 折叠处理
    res = piecewise_fold(res, evaluate=None)
    if res.is_Piecewise:
        newargs = []
        # 如果 res 是 Piecewise 对象，则处理其 args
        for e, c in res.args:
            # 对每个 e 执行 _clean 和 _my_unpolarify 处理
            e = _my_unpolarify(_clean(e))
            newargs += [(e, c)]
        # 重新构建 Piecewise 对象
        res = Piecewise(*newargs, evaluate=False)
    else:
        # 否则，对整体结果执行 _clean 和 _my_unpolarify 处理
        res = _my_unpolarify(_clean(res))
    # 返回一个Piecewise对象，根据条件选择返回res或Integral(f, x)
    return Piecewise((res, _my_unpolarify(cond)), (Integral(f, x), True))
@timeit
def meijerint_definite(f, x, a, b):
    """
    Integrate ``f`` over the interval [``a``, ``b``], by rewriting it as a product
    of two G functions, or as a single G function.

    Return res, cond, where cond are convergence conditions.

    Examples
    ========

    >>> from sympy.integrals.meijerint import meijerint_definite
    >>> from sympy import exp, oo
    >>> from sympy.abc import x
    >>> meijerint_definite(exp(-x**2), x, -oo, oo)
    (sqrt(pi), True)

    This function is implemented as a succession of functions
    meijerint_definite, _meijerint_definite_2, _meijerint_definite_3,
    _meijerint_definite_4. Each function in the list calls the next one
    (presumably) several times. This means that calling meijerint_definite
    can be very costly.
    """
    # This consists of three steps:
    # 1) Change the integration limits to 0, oo
    # 2) Rewrite in terms of G functions
    # 3) Evaluate the integral
    #
    # There are usually several ways of doing this, and we want to try all.
    # This function does (1), calls _meijerint_definite_2 for step (2).
    
    # Log debugging information about the integration process
    _debugf('Integrating %s wrt %s from %s to %s.', (f, x, a, b))
    
    # Convert the integrand function f into a sympy expression
    f = sympify(f)
    
    # Check if the integrand contains DiracDelta terms, if so, return None
    if f.has(DiracDelta):
        _debug('Integrand has DiracDelta terms - giving up.')
        return None
    
    # Check if the integrand contains SingularityFunction terms, if so, return None
    if f.has(SingularityFunction):
        _debug('Integrand has Singularity Function terms - giving up.')
        return None
    
    # Store the original function and integration limits
    f_, x_, a_, b_ = f, x, a, b
    
    # Substitute the integration variable x with a dummy variable d to avoid conflicts
    d = Dummy('x')
    f = f.subs(x, d)
    x = d
    
    # If the integration limits are the same, return zero with True convergence condition
    if a == b:
        return (S.Zero, True)
    
    # Initialize a list to store results
    results = []
    
    # Case: integrating from -oo to +oo with finite upper limit b
    if a is S.NegativeInfinity and b is not S.Infinity:
        return meijerint_definite(f.subs(x, -x), x, -b, -a)
    
    # Case: integrating from -oo to +oo
    elif a is S.NegativeInfinity:
        # Find sensible points to split the integral
        _debug('  Integrating -oo to +oo.')
        innermost = _find_splitting_points(f, x)
        _debug('  Sensible splitting points:', innermost)
        
        # Iterate over the splitting points in descending order and also try at zero
        for c in sorted(innermost, key=default_sort_key, reverse=True) + [S.Zero]:
            _debug('  Trying to split at', c)
            
            # Skip non-real splitting points
            if not c.is_extended_real:
                _debug('  Non-real splitting point.')
                continue
            
            # Compute the first integral after splitting
            res1 = _meijerint_definite_2(f.subs(x, x + c), x)
            if res1 is None:
                _debug('  But could not compute first integral.')
                continue
            
            # Compute the second integral after splitting
            res2 = _meijerint_definite_2(f.subs(x, c - x), x)
            if res2 is None:
                _debug('  But could not compute second integral.')
                continue
            
            # Unpack results and conditions
            res1, cond1 = res1
            res2, cond2 = res2
            
            # Simplify and combine conditions
            cond = _condsimp(And(cond1, cond2))
            
            # If combined condition is false, skip
            if cond == False:
                _debug('  But combined condition is always false.')
                continue
            
            # Sum up the results of the split integrals
            res = res1 + res2
            return res, cond
    elif a is S.Infinity:
        # 如果 a 是无穷大，则计算 Meijer G 函数的定积分
        res = meijerint_definite(f, x, b, S.Infinity)
        # 返回结果为负数，并保留原结果的符号
        return -res[0], res[1]

    elif (a, b) == (S.Zero, S.Infinity):
        # 如果 a 是 0，b 是无穷大的常见情况，首先尝试直接计算
        res = _meijerint_definite_2(f, x)
        if res:
            if _has(res[0], meijerg):
                # 如果结果包含 Meijer G 函数，则将结果添加到结果列表
                results.append(res)
            else:
                # 否则直接返回结果
                return res

    else:
        if b is S.Infinity:
            # 对于 b 是无穷大的情况，找到分裂点进行尝试
            for split in _find_splitting_points(f, x):
                # 如果 a - split 大于等于 0，则尝试处理 x -> x + split
                if (a - split >= 0) == True:
                    _debugf('Trying x -> x + %s', split)
                    # 计算变换后的函数的定积分
                    res = _meijerint_definite_2(f.subs(x, x + split)
                                                *Heaviside(x + split - a), x)
                    if res:
                        if _has(res[0], meijerg):
                            # 如果结果包含 Meijer G 函数，则将结果添加到结果列表
                            results.append(res)
                        else:
                            # 否则直接返回结果
                            return res

        # 更新函数和限制条件
        f = f.subs(x, x + a)
        b = b - a
        a = 0
        # 如果新的 b 不是无穷大，则进行变换
        if b is not S.Infinity:
            phi = exp(S.ImaginaryUnit*arg(b))
            b = Abs(b)
            f = f.subs(x, phi*x)
            f *= Heaviside(b - x)*phi
            b = S.Infinity

        _debug('Changed limits to', a, b)
        _debug('Changed function to', f)
        # 计算变换后的函数的定积分
        res = _meijerint_definite_2(f, x)
        if res:
            if _has(res[0], meijerg):
                # 如果结果包含 Meijer G 函数，则将结果添加到结果列表
                results.append(res)
            else:
                # 否则直接返回结果
                return res

    # 如果函数包含双曲函数，则尝试用指数函数来重写
    if f_.has(HyperbolicFunction):
        _debug('Try rewriting hyperbolics in terms of exp.')
        # 尝试用指数函数重写双曲函数，然后计算定积分
        rv = meijerint_definite(
            _rewrite_hyperbolics_as_exp(f_), x_, a_, b_)
        if rv:
            if not isinstance(rv, list):
                from sympy.simplify.radsimp import collect
                # 对结果进行简化和收集
                rv = (collect(factor_terms(rv[0]), rv[0].atoms(exp)),) + rv[1:]
                return rv
            # 否则将结果列表扩展到当前结果列表中
            results.extend(rv)

    # 如果有结果，则返回排序后的第一个结果
    if results:
        return next(ordered(results))
def _guess_expansion(f, x):
    """ 尝试为被积函数 f(x) 猜测合理的重写形式。"""
    # 初始化结果列表，包含原始积分函数
    res = [(f, 'original integrand')]

    # 获取原始积分函数
    orig = res[-1][0]
    # 使用集合记录已经处理过的函数，起始包含原始函数
    saw = {orig}

    # 对原始函数进行乘法展开，并检查是否已经处理过
    expanded = expand_mul(orig)
    if expanded not in saw:
        res += [(expanded, 'expand_mul')]
        saw.add(expanded)

    # 对原始函数进行通常展开，并检查是否已经处理过
    expanded = expand(orig)
    if expanded not in saw:
        res += [(expanded, 'expand')]
        saw.add(expanded)

    # 如果原始函数包含三角函数或双曲函数，对其进行特定展开和乘法展开，并检查是否已处理过
    if orig.has(TrigonometricFunction, HyperbolicFunction):
        expanded = expand_mul(expand_trig(orig))
        if expanded not in saw:
            res += [(expanded, 'expand_trig, expand_mul')]
            saw.add(expanded)

    # 如果原始函数包含余弦或正弦函数，使用 sincos_to_sum 进行简化，并检查是否已处理过
    if orig.has(cos, sin):
        from sympy.simplify.fu import sincos_to_sum
        reduced = sincos_to_sum(orig)
        if reduced not in saw:
            res += [(reduced, 'trig power reduction')]
            saw.add(reduced)

    return res


def _meijerint_definite_2(f, x):
    """
    尝试从0到无穷大积分 f dx。

    该函数计算函数 f 的多种简化形式（例如通过调用 expand_mul(), trigexpand() - 见 _guess_expansion），
    并依次调用 _meijerint_definite_3。如果 _meijerint_definite_3 对任何简化后的函数成功，则返回其结果。
    """
    # 使用一个正数的虚拟变量 - 我们从0到oo进行积分
    dummy = _dummy('x', 'meijerint-definite2', f, positive=True)
    f = f.subs(x, dummy)
    x = dummy

    # 如果 f 为零函数，直接返回结果为零和 True
    if f == 0:
        return S.Zero, True

    # 对于通过 _guess_expansion 猜测的每个函数 g，尝试调用 _meijerint_definite_3
    for g, explanation in _guess_expansion(f, x):
        _debug('Trying', explanation)
        res = _meijerint_definite_3(g, x)
        if res:
            return res


def _meijerint_definite_3(f, x):
    """
    尝试从0到无穷大积分 f dx。

    该函数调用 _meijerint_definite_4 尝试计算积分。如果失败，则尝试使用线性性质。
    """
    # 尝试使用 _meijerint_definite_4 计算积分
    res = _meijerint_definite_4(f, x)
    if res and res[1] != False:
        return res
    # 如果 f 是 Add 类型（即表达式为加法），则尝试对每个项分别计算积分
    if f.is_Add:
        _debug('Expanding and evaluating all terms.')
        ress = [_meijerint_definite_4(g, x) for g in f.args]
        if all(r is not None for r in ress):
            conds = []
            res = S.Zero
            for r, c in ress:
                res += r
                conds += [c]
            c = And(*conds)
            if c != False:
                return res, c


def _my_unpolarify(f):
    """
    对 f 进行非极坐标化处理。
    """
    return _eval_cond(unpolarify(f))


@timeit
def _meijerint_definite_4(f, x, only_double=False):
    """
    尝试从0到无穷大积分 f dx。

    Explanation
    ===========

    该函数尝试应用文献中找到的积分定理，即尝试将 f 重写为一个或两个 G 函数的乘积。
    """
    The parameter ``only_double`` is used internally in the recursive algorithm
    to disable trying to rewrite f as a single G-function.
    """
    from sympy.simplify import hyperexpand
    # This function does (2) and (3)
    _debug('Integrating', f)
    # Try single G function.
    if not only_double:
        gs = _rewrite1(f, x, recursive=False)
        if gs is not None:
            fac, po, g, cond = gs
            _debug('Could rewrite as single G function:', fac, po, g)
            res = S.Zero
            for C, s, f in g:
                if C == 0:
                    continue
                C, f = _rewrite_saxena_1(fac*C, po*x**s, f, x)
                res += C*_int0oo_1(f, x)
                cond = And(cond, _check_antecedents_1(f, x))
                if cond == False:
                    break
            cond = _my_unpolarify(cond)
            if cond == False:
                _debug('But cond is always False.')
            else:
                _debug('Result before branch substitutions is:', res)
                return _my_unpolarify(hyperexpand(res)), cond

    # Try two G functions.
    gs = _rewrite2(f, x)
    if gs is not None:
        for full_pb in [False, True]:
            fac, po, g1, g2, cond = gs
            _debug('Could rewrite as two G functions:', fac, po, g1, g2)
            res = S.Zero
            for C1, s1, f1 in g1:
                for C2, s2, f2 in g2:
                    r = _rewrite_saxena(fac*C1*C2, po*x**(s1 + s2),
                                        f1, f2, x, full_pb)
                    if r is None:
                        _debug('Non-rational exponents.')
                        return
                    C, f1_, f2_ = r
                    _debug('Saxena subst for yielded:', C, f1_, f2_)
                    cond = And(cond, _check_antecedents(f1_, f2_, x))
                    if cond == False:
                        break
                    res += C*_int0oo(f1_, f2_, x)
                else:
                    continue
                break
            cond = _my_unpolarify(cond)
            if cond == False:
                _debugf('But cond is always False (full_pb=%s).', full_pb)
            else:
                _debugf('Result before branch substitutions is: %s', (res, ))
                if only_double:
                    return res, cond
                return _my_unpolarify(hyperexpand(res)), cond
# 定义一个函数，计算逆拉普拉斯变换 $\int_{c+i\infty}^{c-i\infty} f(x) e^{tx}\, dx$，
# 其中 $c$ 是大于函数 `f` 所有奇点实部的实数。

def meijerint_inversion(f, x, t):
    # 保存函数 `f` 的原始版本
    f_ = f
    # 保存参数 `t` 的原始值
    t_ = t
    # 创建一个虚数域的虚数变量 `t`，以避免 sqrt(t**2) = abs(t) 等情况
    t = Dummy('t', polar=True)
    # 将函数 `f` 中的 `t_` 替换为新定义的虚数变量 `t`
    f = f.subs(t_, t)
    
    # 调试信息：输出正在进行拉普拉斯反变换的表达式 `f`
    _debug('Laplace-inverting', f)
    
    # 检查函数 `f` 是否解析（全复平面上解析）
    if not _is_analytic(f, x):
        # 调试信息：表达式不是解析的情况下返回 None
        _debug('But expression is not analytic.')
        return None
    
    # 处理指数项，这些项对应于平移；我们将它们过滤掉，稍后会对结果进行平移。
    # 如果 `f` 是乘积，则将其分解为单独的因子；如果是指数函数则单独处理；其他情况直接保留。
    shift = S.Zero
    if f.is_Mul:
        args = list(f.args)
    elif isinstance(f, exp):
        args = [f]
    else:
        args = None
    
    if args:
        newargs = []
        exponentials = []
        while args:
            arg = args.pop()
            if isinstance(arg, exp):
                arg2 = expand(arg)
                if arg2.is_Mul:
                    args += arg2.args
                    continue
                try:
                    a, b = _get_coeff_exp(arg.args[0], x)
                except _CoeffExpValueError:
                    b = 0
                if b == 1:
                    exponentials.append(a)
                else:
                    newargs.append(arg)
            elif arg.is_Pow:
                arg2 = expand(arg)
                if arg2.is_Mul:
                    args += arg2.args
                    continue
                if x not in arg.base.free_symbols:
                    try:
                        a, b = _get_coeff_exp(arg.exp, x)
                    except _CoeffExpValueError:
                        b = 0
                    if b == 1:
                        exponentials.append(a*log(arg.base))
                newargs.append(arg)
            else:
                newargs.append(arg)
        # 计算所有指数项的总和
        shift = Add(*exponentials)
        # 将剩余的非指数项重新组合成新的表达式 `f`
        f = Mul(*newargs)
    
    # 如果 `f` 不包含变量 `x`，则表明其由常数和指数项组成
    if x not in f.free_symbols:
        # 调试信息：输出常数和指数平移的表达式 `f` 和 `shift`
        _debug('Expression consists of constant and exp shift:', f, shift)
        # 判断指数项是否为非实数，如果是，则无法是拉普拉斯变换
        cond = Eq(im(shift), 0)
        if cond == False:
            # 调试信息：指数项非实数，无法是拉普拉斯变换
            _debug('but shift is nonreal, cannot be a Laplace transform')
            return None
        # 计算结果为 `f` 乘以 `DiracDelta(t + shift)`
        res = f * DiracDelta(t + shift)
        # 调试信息：输出结果是一个 delta 函数，可能是条件性的
        _debug('Result is a delta function, possibly conditional:', res, cond)
        # 返回一个分段函数，表示条件性的结果
        return Piecewise((res.subs(t, t_), cond))
    
    # 对 `f` 进行重写处理，返回重写后的表达式
    gs = _rewrite1(f, x)
    # 如果 gs 不为 None，则进行以下操作
    if gs is not None:
        # 将 gs 解包为 fac, po, g, cond 四个变量
        fac, po, g, cond = gs
        # 打印调试信息，指出可以重写为单个 G 函数的情况，输出 fac, po, g 的值
        _debug('Could rewrite as single G function:', fac, po, g)
        # 初始化 res 为零
        res = S.Zero
        # 遍历 g 中的每个三元组 (C, s, f)
        for C, s, f in g:
            # 调用 _rewrite_inversion 函数，对 fac*C, po*x**s, f, x 进行重写操作
            C, f = _rewrite_inversion(fac*C, po*x**s, f, x)
            # 将 C*_int_inversion(f, x, t) 加到 res 中
            res += C*_int_inversion(f, x, t)
            # 更新 cond，与 _check_antecedents_inversion(f, x) 的结果取交集
            cond = And(cond, _check_antecedents_inversion(f, x))
            # 如果 cond 为 False，则退出循环
            if cond == False:
                break
        # 对 cond 进行去极化处理
        cond = _my_unpolarify(cond)
        # 如果 cond 为 False，则打印调试信息
        if cond == False:
            _debug('But cond is always False.')
        else:
            # 否则，打印调试信息，显示分支替换前的结果 res
            _debug('Result before branch substitution:', res)
            # 导入 hyperexpand 函数，对 res 进行超级展开
            from sympy.simplify import hyperexpand
            res = _my_unpolarify(hyperexpand(res))
            # 如果 res 中不包含 Heaviside 函数，则乘以 Heaviside(t)
            if not res.has(Heaviside):
                res *= Heaviside(t)
            # 将 t 替换为 t + shift
            res = res.subs(t, t + shift)
            # 如果 cond 不是布尔值，则也将 t 替换为 t + shift
            if not isinstance(cond, bool):
                cond = cond.subs(t, t + shift)
            # 导入 InverseLaplaceTransform 函数，返回一个 Piecewise 对象
            from .transforms import InverseLaplaceTransform
            return Piecewise((res.subs(t, t_), cond),
                             (InverseLaplaceTransform(f_.subs(t, t_), x, t_, None), True))
```