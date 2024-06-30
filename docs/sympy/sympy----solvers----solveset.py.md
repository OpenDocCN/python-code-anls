# `D:\src\scipysrc\sympy\sympy\solvers\solveset.py`

```
"""
This module contains functions to:

    - solve a single equation for a single variable, in any domain either real or complex.

    - solve a single transcendental equation for a single variable in any domain either real or complex.
      (currently supports solving in real domain only)

    - solve a system of linear equations with N variables and M equations.

    - solve a system of Non Linear Equations with N variables and M equations
"""
# 导入 sympy 库中所需的模块和函数

from sympy.core.sympify import sympify  # 导入 sympify 函数，用于将输入转换为 sympy 的表达式
from sympy.core import (S, Pow, Dummy, pi, Expr, Wild, Mul,  # 导入核心类和符号
                        Add, Basic)
from sympy.core.containers import Tuple  # 导入 Tuple 容器，用于处理元组
from sympy.core.function import (Lambda, expand_complex, AppliedUndef,  # 导入函数和方法，包括处理复数的函数
                                expand_log, _mexpand, expand_trig, nfloat)
from sympy.core.mod import Mod  # 导入 Mod 类，用于处理模运算
from sympy.core.numbers import I, Number, Rational, oo  # 导入数值类和常量
from sympy.core.intfunc import integer_log  # 导入整数函数，例如 integer_log
from sympy.core.relational import Eq, Ne, Relational  # 导入关系运算符
from sympy.core.sorting import default_sort_key, ordered  # 导入排序相关函数
from sympy.core.symbol import Symbol, _uniquely_named_symbol  # 导入符号相关类和方法
from sympy.core.sympify import _sympify  # 导入 _sympify 函数，用于内部类型转换
from sympy.core.traversal import preorder_traversal  # 导入前序遍历方法
from sympy.external.gmpy import gcd as number_gcd, lcm as number_lcm  # 导入外部依赖函数 gcd 和 lcm
from sympy.polys.matrices.linsolve import _linear_eq_to_dict  # 导入线性方程组求解相关函数
from sympy.polys.polyroots import UnsolvableFactorError  # 导入多项式根求解相关异常类
from sympy.simplify.simplify import simplify, fraction, trigsimp, nsimplify  # 导入简化函数和方法
from sympy.simplify import powdenest, logcombine  # 导入幂简化和对数合并函数
from sympy.functions import (log, tan, cot, sin, cos, sec, csc, exp,  # 导入数学函数
                             acos, asin, atan, acot, acsc, asec,
                             piecewise_fold, Piecewise)
from sympy.functions.combinatorial.numbers import totient  # 导入数论相关函数
from sympy.functions.elementary.complexes import Abs, arg, re, im  # 导入复数函数
from sympy.functions.elementary.hyperbolic import (HyperbolicFunction,  # 导入双曲函数
                            sinh, cosh, tanh, coth, sech, csch,
                            asinh, acosh, atanh, acoth, asech, acsch)
from sympy.functions.elementary.miscellaneous import real_root  # 导入实根函数
from sympy.functions.elementary.trigonometric import TrigonometricFunction  # 导入三角函数
from sympy.logic.boolalg import And, BooleanTrue  # 导入布尔逻辑运算类和对象
from sympy.sets import (FiniteSet, imageset, Interval, Intersection,  # 导入集合操作相关类和方法
                        Union, ConditionSet, ImageSet, Complement, Contains)
from sympy.sets.sets import Set, ProductSet  # 导入集合类和笛卡尔积类
from sympy.matrices import zeros, Matrix, MatrixBase  # 导入矩阵和矩阵基类
from sympy.ntheory.factor_ import divisors  # 导入因子计算函数
from sympy.ntheory.residue_ntheory import discrete_log, nthroot_mod  # 导入离散对数和模 n 次方根函数
from sympy.polys import (roots, Poly, degree, together, PolynomialError,  # 导入多项式操作相关类和函数
                         RootOf, factor, lcm, gcd)
from sympy.polys.polyerrors import CoercionFailed  # 导入多项式异常类
from sympy.polys.polytools import invert, groebner, poly  # 导入多项式工具类和方法
from sympy.polys.solvers import (sympy_eqs_to_ring, solve_lin_sys,  # 导入多项式求解器相关方法
    PolyNonlinearError)
from sympy.polys.matrices.linsolve import _linsolve  # 导入线性方程组求解器
from sympy.solvers.solvers import (checksol, denoms, unrad,  # 导入通用求解器相关方法

# 代码省略部分，因为后续导入项很多
    _simple_dens, recast_to_symbols)


    # 此行代码看起来像是函数或方法调用，但缺少前面的函数名或对象，不完整的代码片段
    # 无法确定其作用或含义，需要更多上下文才能正确注释
# 导入从 sympy.solvers.polysys 模块中的 solve_poly_system 函数
# 解决多项式系统的方程
from sympy.solvers.polysys import solve_poly_system

# 从 sympy.utilities 模块导入 filldedent 函数
# 用于修复多行字符串缩进的函数
from sympy.utilities import filldedent

# 从 sympy.utilities.iterables 模块导入 numbered_symbols, has_dups,
# is_sequence, iterable 函数
# 分别用于生成编号的符号、检查重复项、检查是否为序列、检查是否可迭代
from sympy.utilities.iterables import (numbered_symbols, has_dups,
                                       is_sequence, iterable)

# 从 sympy.calculus.util 模块导入 periodicity, continuous_domain, function_range 函数
# 分别用于计算函数的周期性、连续定义域、函数值域
from sympy.calculus.util import periodicity, continuous_domain, function_range

# 导入 types 模块中的 GeneratorType
# 用于表示生成器类型的对象
from types import GeneratorType


# 自定义异常类 NonlinearError，继承自 ValueError
# 当遇到非线性方程时抛出异常
class NonlinearError(ValueError):
    """Raised when unexpectedly encountering nonlinear equations"""
    pass


# 定义函数 _masked，接受函数 f 和多个 atoms 参数
# 用于将函数 f 中的 atoms 参数替换为虚拟符号，并返回替换后的函数和替换列表
def _masked(f, *atoms):
    """Return ``f``, with all objects given by ``atoms`` replaced with
    Dummy symbols, ``d``, and the list of replacements, ``(d, e)``,
    where ``e`` is an object of type given by ``atoms`` in which
    any other instances of atoms have been recursively replaced with
    Dummy symbols, too. The tuples are ordered so that if they are
    applied in sequence, the origin ``f`` will be restored.

    Examples
    ========

    >>> from sympy import cos
    >>> from sympy.abc import x
    >>> from sympy.solvers.solveset import _masked

    >>> f = cos(cos(x) + 1)
    >>> f, reps = _masked(cos(1 + cos(x)), cos)
    >>> f
    _a1
    >>> reps
    [(_a1, cos(_a0 + 1)), (_a0, cos(x))]
    >>> for d, e in reps:
    ...     f = f.xreplace({d: e})
    >>> f
    cos(cos(x) + 1)
    """
    # 使用 numbered_symbols 生成虚拟符号序列，以替换 atoms 参数
    sym = numbered_symbols('a', cls=Dummy, real=True)
    mask = []
    # 遍历 f 中所有 atoms 参数的对象
    for a in ordered(f.atoms(*atoms)):
        # 递归替换 atoms 中的对象为虚拟符号，并记录替换对
        for i in mask:
            a = a.replace(*i)
        mask.append((a, next(sym)))
    # 将 f 中的 atoms 参数对象替换为虚拟符号
    for i, (o, n) in enumerate(mask):
        f = f.replace(o, n)
        mask[i] = (n, o)
    # 反转替换对列表，以便按顺序还原原始 f
    mask = list(reversed(mask))
    return f, mask


# 定义函数 _invert，用于将复值方程 f(x) = y 转换为一组简化方程
# 返回简化函数 g(x) 和方程集合 set_h
def _invert(f_x, y, x, domain=S.Complexes):
    r"""
    Reduce the complex valued equation $f(x) = y$ to a set of equations

    $$\left\{g(x) = h_1(y),\  g(x) = h_2(y),\ \dots,\  g(x) = h_n(y) \right\}$$

    where $g(x)$ is a simpler function than $f(x)$.  The return value is a tuple
    $(g(x), \mathrm{set}_h)$, where $g(x)$ is a function of $x$ and $\mathrm{set}_h$ is
    the set of function $\left\{h_1(y), h_2(y), \dots, h_n(y)\right\}$.
    Here, $y$ is not necessarily a symbol.

    $\mathrm{set}_h$ contains the functions, along with the information
    about the domain in which they are valid, through set
    operations. For instance, if :math:`y = |x| - n` is inverted
    in the real domain, then $\mathrm{set}_h$ is not simply
    $\{-n, n\}$ as the nature of `n` is unknown; rather, it is:

    $$ \left(\left[0, \infty\right) \cap \left\{n\right\}\right) \cup
                       \left(\left(-\infty, 0\right] \cap \left\{- n\right\}\right)$$

    By default, the complex domain is used which means that inverting even
    seemingly simple functions like $\exp(x)$ will give very different
    results from those obtained in the real domain.
    (In the case of $\exp(x)$, the inversion via $\log$ is multi-valued
    in the complex domain, having infinitely many branches.)

    If you are working with real values only (or you are not sure which
    """
    # 未完待续，由于代码截断，请继续编辑以完成注释
    # 将输入的 x 转换为符号表达式
    x = sympify(x)
    # 如果 x 不是符号表达式，则抛出数值错误
    if not x.is_Symbol:
        raise ValueError("x must be a symbol")
    # 将输入的 f_x 转换为符号表达式
    f_x = sympify(f_x)
    # 如果 x 不是 f_x 中的自由符号，则抛出数值错误
    if x not in f_x.free_symbols:
        raise ValueError("Inverse of constant function doesn't exist")
    # 将输入的 y 转换为符号表达式
    y = sympify(y)
    # 如果 y 中包含 x 的符号，则抛出数值错误
    if x in y.free_symbols:
        raise ValueError("y should be independent of x ")
    
    # 如果 domain 是 S.Reals 的子集，则使用 _invert_real 函数求解反函数
    if domain.is_subset(S.Reals):
        x1, s = _invert_real(f_x, FiniteSet(y), x)
    else:
        # 否则，使用 _invert_complex 函数求解复数域中的反函数
        x1, s = _invert_complex(f_x, FiniteSet(y), x)
    
    # 如果求解后的 x1 与输入的 x 不同，则返回求解结果 x1 和 s
    if x1 != x:
        return x1, s
    
    # 如果 domain 是 S.Complexes，则返回 x1 和 s，避免不必要地与 S.Complexes 交集
    if domain is S.Complexes:
        return x1, s
    
    # 如果 s 是有限集合，则返回 x1 和 s 与 domain 的交集
    if isinstance(s, FiniteSet):
        return x1, s.intersect(domain)
    
    # 对于复杂的解集（如三角函数的反函数），已经包含了通用的有效条件，
    # 不需要与 S.Reals 的交集，但仍需考虑其子集的情况
    if domain is S.Reals:
        return x1, s
    else:
        return x1, s.intersect(domain)
# 将函数 `_invert` 赋值给变量 `invert_complex`
invert_complex = _invert

# 定义函数 `invert_real`，用于反转实值函数。与 `invert_complex` 类似，但在反转之前将域设置为 `S.Reals`
def invert_real(f_x, y, x):
    """
    Inverts a real-valued function. Same as :func:`invert_complex`, but sets
    the domain to ``S.Reals`` before inverting.
    """
    return _invert(f_x, y, x, S.Reals)

# 定义辅助函数 `_invert_real` 用于 `_invert` 的辅助操作
def _invert_real(f, g_ys, symbol):
    """Helper function for _invert."""

    # 如果 `f` 等于 `symbol` 或者 `g_ys` 是空集，直接返回 `(symbol, g_ys)`
    if f == symbol or g_ys is S.EmptySet:
        return (symbol, g_ys)

    # 创建一个实数虚拟变量 `n`
    n = Dummy('n', real=True)

    # 如果 `f` 是指数函数或者是形如 `exp(x)` 的幂函数
    if isinstance(f, exp) or (f.is_Pow and f.base == S.Exp1):
        # 递归调用 `_invert_real` 处理指数函数 `f.exp`，并应用 Lambda 表达式对 `g_ys` 进行映射
        return _invert_real(f.exp,
                            imageset(Lambda(n, log(n)), g_ys),
                            symbol)

    # 如果 `f` 有 `inverse` 方法且其逆存在，并且不是三角函数或双曲函数
    if hasattr(f, 'inverse') and f.inverse() is not None and not isinstance(f, (
            TrigonometricFunction,
            HyperbolicFunction,
            )):
        # 如果 `f` 的参数个数大于1，抛出错误
        if len(f.args) > 1:
            raise ValueError("Only functions with one argument are supported.")
        # 递归调用 `_invert_real` 处理 `f.args[0]`，并应用 Lambda 表达式对 `g_ys` 进行映射
        return _invert_real(f.args[0],
                            imageset(Lambda(n, f.inverse()(n)), g_ys),
                            symbol)

    # 如果 `f` 是绝对值函数
    if isinstance(f, Abs):
        # 调用 `_invert_abs` 处理 `f.args[0]`，并返回结果
        return _invert_abs(f.args[0], g_ys, symbol)

    # 如果 `f` 是加法表达式
    if f.is_Add:
        # 将 `f` 拆分为独立项 `g` 和 `h` 关于 `symbol` 的部分
        g, h = f.as_independent(symbol)
        # 如果 `g` 不为零，递归调用 `_invert_real` 处理 `h`，并应用 Lambda 表达式对 `g_ys` 进行映射
        if g is not S.Zero:
            return _invert_real(h, imageset(Lambda(n, n - g), g_ys), symbol)

    # 如果 `f` 是乘法表达式
    if f.is_Mul:
        # 将 `f` 拆分为独立项 `g` 和 `h` 关于 `symbol` 的部分
        g, h = f.as_independent(symbol)
        # 如果 `g` 不为一，递归调用 `_invert_real` 处理 `h`，并应用 Lambda 表达式对 `g_ys` 进行映射
        if g is not S.One:
            return _invert_real(h, imageset(Lambda(n, n/g), g_ys), symbol)
    # 检查表达式是否是幂运算
    if f.is_Pow:
        # 将幂运算拆解为底数和指数
        base, expo = f.args
        # 检查底数和指数是否含有给定符号
        base_has_sym = base.has(symbol)
        expo_has_sym = expo.has(symbol)

        # 如果指数不含给定符号
        if not expo_has_sym:

            # 如果指数是有理数
            if expo.is_rational:
                # 分离有理数指数的分子和分母
                num, den = expo.as_numer_denom()

                # 如果分母是偶数且分子是奇数且分母非零
                if den % 2 == 0 and num % 2 == 1 and not den.is_zero:
                    # 计算实根函数并应用于 g_ys 中的正数部分
                    root = Lambda(n, real_root(n, expo))
                    g_ys_pos = g_ys & Interval(0, oo)
                    res = imageset(root, g_ys_pos)
                    # 对底数进行逆变换
                    _inv, _set = _invert_real(base, res, symbol)
                    return (_inv, _set)

                # 如果分母是奇数
                if den % 2 == 1:
                    root = Lambda(n, real_root(n, expo))
                    res = imageset(root, g_ys)
                    # 如果分子是偶数，对结果应用负数变换
                    if num % 2 == 0:
                        neg_res = imageset(Lambda(n, -n), res)
                        return _invert_real(base, res + neg_res, symbol)
                    # 如果分子是奇数，直接对结果进行逆变换
                    elif num % 2 == 1:
                        return _invert_real(base, res, symbol)

            # 如果指数是无理数
            elif expo.is_irrational:
                root = Lambda(n, real_root(n, expo))
                g_ys_pos = g_ys & Interval(0, oo)
                res = imageset(root, g_ys_pos)
                # 对底数进行逆变换
                return _invert_real(base, res, symbol)

            else:
                # 无法确定指数类型，例如浮点数或无法确定有理数的奇偶性
                pass  # 使用默认返回值

        # 如果底数不含给定符号
        if not base_has_sym:
            rhs = g_ys.args[0]
            # 如果底数是正数，对指数应用逆变换
            if base.is_positive:
                return _invert_real(expo,
                    imageset(Lambda(n, log(n, base, evaluate=False)), g_ys), symbol)
            # 如果底数是负数，计算指数对数与 g_ys 的交集
            elif base.is_negative:
                s, b = integer_log(rhs, base)
                if b:
                    return _invert_real(expo, FiniteSet(s), symbol)
                else:
                    return (expo, S.EmptySet)
            # 如果底数是零
            elif base.is_zero:
                one = Eq(rhs, 1)
                # 特殊情况：0**x - 1
                if one == S.true:
                    return _invert_real(expo, FiniteSet(0), symbol)
                # 否则返回空集
                elif one == S.false:
                    return (expo, S.EmptySet)
    # 检查变量 f 是否属于 TrigonometricFunction 或 HyperbolicFunction 类型的实例
    if isinstance(f, (TrigonometricFunction, HyperbolicFunction)):
         # 如果是，则调用 _invert_trig_hyp_real 函数进行反三角和反双曲函数的逆运算
         # 返回结果包括反运算后的值和 g_ys，以及符号 symbol
         return _invert_trig_hyp_real(f, g_ys, symbol)

    # 如果 f 不是 TrigonometricFunction 或 HyperbolicFunction 的实例，则返回 f 和 g_ys
    return (f, g_ys)
# 缓存首次使用后的反三角和反双曲函数的字典。
_trig_inverses = None
_hyp_inverses = None

# 辅助函数，用于求解实数范围内的反三角和反双曲函数。
def _invert_trig_hyp_real(f, g_ys, symbol):
    """
    Helper function for inverting trigonometric and hyperbolic functions.

    This helper only handles inversion over the reals.

    For trigonometric functions, only finite `g_ys` sets are implemented.

    For hyperbolic functions, the set `g_ys` is checked against the domain of the
    respective inverse functions. Infinite `g_ys` sets are also supported.
    """
    if isinstance(f, HyperbolicFunction):
        n = Dummy('n', real=True)
        
        if isinstance(f, sinh):
            # 如果 f 是 sinh 函数，则反函数 asinh 在实数域上定义。
            return _invert_real(f.args[0], imageset(n, asinh(n), g_ys), symbol)

        if isinstance(f, cosh):
            # 如果 f 是 cosh 函数，则 g_ys 的定义域必须在 [1, ∞) 上。
            g_ys_dom = g_ys.intersect(Interval(1, oo))
            if isinstance(g_ys_dom, Intersection):
                # 如果无法正确解析定义域检查，则根据 g_ys 的类型决定处理方式。
                if isinstance(g_ys, FiniteSet):
                    # 如果 g_ys 是有限集，则允许调用 _invert_real() 添加与 S.Reals
                    # （或者一个子集 domain）的交集，以确保只返回有效（实数）的解。
                    # 这避免在返回的集合中添加“过多”的交集或条件集。
                    g_ys_dom = g_ys
                else:
                    return (f, g_ys)
            # 返回一个包含正负 acosh 映射结果的并集，用于求解反函数。
            return _invert_real(f.args[0], Union(
                imageset(n, acosh(n), g_ys_dom),
                imageset(n, -acosh(n), g_ys_dom)), symbol)

        if isinstance(f, sech):
            # 如果 f 是 sech 函数，则 g_ys 的定义域必须在 (0, 1] 上。
            g_ys_dom = g_ys.intersect(Interval.Lopen(0, 1))
            if isinstance(g_ys_dom, Intersection):
                # 如果无法正确解析定义域检查，则根据 g_ys 的类型决定处理方式。
                if isinstance(g_ys, FiniteSet):
                    g_ys_dom = g_ys
                else:
                    return (f, g_ys)
            # 返回一个包含正负 asech 映射结果的并集，用于求解反函数。
            return _invert_real(f.args[0], Union(
                imageset(n, asech(n), g_ys_dom),
                imageset(n, -asech(n), g_ys_dom)), symbol)

        if isinstance(f, tanh):
            # 如果 f 是 tanh 函数，则 g_ys 的定义域必须在 (-1, 1) 上。
            g_ys_dom = g_ys.intersect(Interval.open(-1, 1))
            if isinstance(g_ys_dom, Intersection):
                # 如果无法正确解析定义域检查，则根据 g_ys 的类型决定处理方式。
                if isinstance(g_ys, FiniteSet):
                    g_ys_dom = g_ys
                else:
                    return (f, g_ys)
            # 返回 atanh 映射结果，用于求解反函数。
            return _invert_real(f.args[0],
                imageset(n, atanh(n), g_ys_dom), symbol)

        if isinstance(f, coth):
            # 如果 f 是 coth 函数，则 g_ys 的定义域必须在实数集减去 (-1, 1) 上。
            g_ys_dom = g_ys - Interval(-1, 1)
            if isinstance(g_ys_dom, Complement):
                # 如果无法正确解析定义域检查，则根据 g_ys 的类型决定处理方式。
                if isinstance(g_ys, FiniteSet):
                    g_ys_dom = g_ys
                else:
                    return (f, g_ys)
            # 返回 acoth 映射结果，用于求解反函数。
            return _invert_real(f.args[0],
                imageset(n, acoth(n), g_ys_dom), symbol)

        if isinstance(f, csch):
            # 如果 f 是 csch 函数，则 g_ys 的定义域必须不包括 0。
            g_ys_dom = g_ys - FiniteSet(0)
            if isinstance(g_ys_dom, Complement):
                # 如果无法正确解析定义域检查，则根据 g_ys 的类型决定处理方式。
                if isinstance(g_ys, FiniteSet):
                    g_ys_dom = g_ys
                else:
                    return (f, g_ys)
            # 返回 acsch 映射结果，用于求解反函数。
            return _invert_real(f.args[0],
                imageset(n, acsch(n), g_ys_dom), symbol)
    # 如果 f 是三角函数对象而 g_ys 是有限集对象，则执行以下操作
    elif isinstance(f, TrigonometricFunction) and isinstance(g_ys, FiniteSet):
        
        # 定义内部函数 _get_trig_inverses，用于获取三角函数的逆函数及相关信息
        def _get_trig_inverses(func):
            global _trig_inverses
            # 如果 _trig_inverses 尚未初始化，则初始化它为一个字典，包含不同三角函数的逆函数及相关信息
            if _trig_inverses is None:
                _trig_inverses = {
                    sin : ((asin, lambda y: pi-asin(y)), 2*pi, Interval(-1, 1)),
                    cos : ((acos, lambda y: -acos(y)), 2*pi, Interval(-1, 1)),
                    tan : ((atan,), pi, S.Reals),
                    cot : ((acot,), pi, S.Reals),
                    sec : ((asec, lambda y: -asec(y)), 2*pi,
                        Union(Interval(-oo, -1), Interval(1, oo))),
                    csc : ((acsc, lambda y: pi-acsc(y)), 2*pi,
                        Union(Interval(-oo, -1), Interval(1, oo)))}
            # 返回指定函数 func 对应的逆函数及其周期和定义域信息
            return _trig_inverses[func]

        # 调用 _get_trig_inverses 函数获取三角函数 f 的逆函数、周期和定义域信息
        invs, period, rng = _get_trig_inverses(f.func)
        
        # 创建一个整数变量 Dummy('n', integer=True)
        n = Dummy('n', integer=True)
        
        # 定义内部函数 create_return_set，用于创建返回的 ConditionSet 对象
        def create_return_set(g):
            # 返回 ConditionSet，它将作为最终元组 (x, set) 的一部分
            # 构建包含各个逆函数映射的并集
            invsimg = Union(*[
                imageset(n, period*n + inv(g), S.Integers) for inv in invs])
            
            # 对三角函数 f 的逆函数和 g 的逆函数应用反函数，得到 inv_f 和 inv_g_ys
            inv_f, inv_g_ys = _invert_real(f.args[0], invsimg, symbol)
            
            # 如果 inv_f 等于 symbol，则表示反函数成功
            if inv_f == symbol:
                # 检查 g 是否在定义域 rng 内，生成对应的条件
                conds = rng.contains(g)
                # 返回符号 symbol 的 ConditionSet 对象
                return ConditionSet(symbol, conds, inv_g_ys)
            else:
                # 如果反函数失败，返回一个包含等式 Eq(f, g) 的 ConditionSet 对象
                return ConditionSet(symbol, Eq(f, g), S.Reals)

        # 使用列表推导创建返回值的并集，每个 g 均调用 create_return_set 函数
        retset = Union(*[create_return_set(g) for g in g_ys])
        # 返回符号 symbol 和 retset 的元组
        return (symbol, retset)

    else:
        # 如果不满足上述条件，则直接返回 f 和 g_ys 的元组
        return (f, g_ys)
# 定义一个辅助函数，用于反转三角和双曲函数在复数域上的结果

# 如果 f 是三角函数并且 g_ys 是有限集合
if isinstance(f, TrigonometricFunction) and isinstance(g_ys, FiniteSet):
    # 定义不同三角函数的反函数
    def inv(trig):
        # 如果 trig 是 sin 或 csc 函数
        if isinstance(trig, (sin, csc)):
            # 对应的反函数是 asin 或 acsc
            F = asin if isinstance(trig, sin) else acsc
            return (
                lambda a: 2*n*pi + F(a),        # 第一个反函数形式
                lambda a: 2*n*pi + pi - F(a))   # 第二个反函数形式
        # 如果 trig 是 cos 或 sec 函数
        if isinstance(trig, (cos, sec)):
            # 对应的反函数是 acos 或 asec
            F = acos if isinstance(trig, cos) else asec
            return (
                lambda a: 2*n*pi + F(a),        # 第一个反函数形式
                lambda a: 2*n*pi - F(a))        # 第二个反函数形式
        # 如果 trig 是 tan 或 cot 函数
        if isinstance(trig, (tan, cot)):
            # 对应的反函数是其逆函数
            return (lambda a: n*pi + trig.inverse()(a),)   # 反函数形式

    # 定义一个整数符号 n
    n = Dummy('n', integer=True)
    # 初始化反函数集合为空集
    invs = S.EmptySet
    # 对于每个三角函数的反函数 L
    for L in inv(f):
        # 对每个 g_ys 中的 g，将 L(g) 的整数图像加入到反函数集合中
        invs += Union(*[imageset(Lambda(n, L(g)), S.Integers) for g in g_ys])

    # 调用 _invert_complex 函数，将 f 的第一个参数和反函数集合 invs 以及 symbol 作为参数
    return _invert_complex(f.args[0], invs, symbol)
    elif isinstance(f, HyperbolicFunction) and isinstance(g_ys, FiniteSet):
        # 如果 f 是双曲函数类型并且 g_ys 是有限集合类型，则执行以下操作：
        
        # 定义一个内部函数，用于获取 f 的反函数信息
        def _get_hyp_inverses(func):
            global _hyp_inverses
            # 如果 _hyp_inverses 为 None，则初始化双曲反函数字典
            if _hyp_inverses is None:
                _hyp_inverses = {
                    sinh : ((asinh, lambda y: I*pi-asinh(y)), 2*I*pi, ()),
                    cosh : ((acosh, lambda y: -acosh(y)), 2*I*pi, ()),
                    tanh : ((atanh,), I*pi, (-1, 1)),
                    coth : ((acoth,), I*pi, (-1, 1)),
                    sech : ((asech, lambda y: -asech(y)), 2*I*pi, (0, )),
                    csch : ((acsch, lambda y: I*pi-acsch(y)), 2*I*pi, (0, ))
                }
            # 返回对应 func 的反函数信息
            return _hyp_inverses[func]

        # invs: 主要反函数的可迭代集合，例如 (acosh, -acosh).
        # excl: 需要检查的奇点的可迭代集合。
        invs, period, excl = _get_hyp_inverses(f.func)
        
        # 创建一个内部函数，用于生成返回的 ConditionSet
        def create_return_set(g):
            # 返回将成为最终 (x, set) 元组的 ConditionSet
            # invsimg: 根据主要反函数生成的并集
            invsimg = Union(*[
                imageset(n, period*n + inv(g), S.Integers) for inv in invs])
            # inv_f, inv_g_ys: 尝试对 f 的参数进行反转，生成 inv_g_ys
            inv_f, inv_g_ys = _invert_complex(f.args[0], invsimg, symbol)
            # 如果 inv_f 等于 symbol 表示反转成功
            if inv_f == symbol:
                # conds: 对于 g 不等于 excl 中的任何元素的条件
                conds = And(*[Ne(g, e) for e in excl])
                return ConditionSet(symbol, conds, inv_g_ys)
            else:
                # 如果反转失败，则返回一个包含等式 Eq(f, g) 的 ConditionSet
                return ConditionSet(symbol, Eq(f, g), S.Complexes)

        # retset: g_ys 中每个 g 对应的 ConditionSet 的并集
        retset = Union(*[create_return_set(g) for g in g_ys])
        # 返回结果元组 (symbol, retset)
        return (symbol, retset)

    else:
        # 如果条件不满足，则返回 (f, g_ys)
        return (f, g_ys)
# `_invert_complex` 是 `_invert` 的辅助函数，用于处理复杂的表达式反转操作。

def _invert_complex(f, g_ys, symbol):
    """Helper function for _invert."""

    # 如果 f 等于 symbol 或者 g_ys 是空集，直接返回 (symbol, g_ys)
    if f == symbol or g_ys is S.EmptySet:
        return (symbol, g_ys)

    # 创建一个虚拟变量 n
    n = Dummy('n')

    # 如果 f 是加法表达式
    if f.is_Add:
        # 将 f 分解为 g + h
        g, h = f.as_independent(symbol)
        # 如果 g 不为零，递归处理 h 部分
        if g is not S.Zero:
            return _invert_complex(h, imageset(Lambda(n, n - g), g_ys), symbol)

    # 如果 f 是乘法表达式
    if f.is_Mul:
        # 将 f 分解为 g * h
        g, h = f.as_independent(symbol)

        # 如果 g 不为 1
        if g is not S.One:
            # 如果 g 是无穷大、负无穷大或复无穷大，返回 (h, 空集)
            if g in {S.NegativeInfinity, S.ComplexInfinity, S.Infinity}:
                return (h, S.EmptySet)
            # 递归处理 h 部分
            return _invert_complex(h, imageset(Lambda(n, n/g), g_ys), symbol)

    # 如果 f 是幂函数
    if f.is_Pow:
        base, expo = f.args
        # 特殊情况：g**r = 0，其中 g_ys 是 0 的有限集合
        if expo.is_Rational and g_ys == FiniteSet(0):
            if expo.is_positive:
                return _invert_complex(base, g_ys, symbol)

    # 如果 f 具有 'inverse' 属性且可以反转
    if hasattr(f, 'inverse') and f.inverse() is not None and \
       not isinstance(f, TrigonometricFunction) and \
       not isinstance(f, HyperbolicFunction) and \
       not isinstance(f, exp):
        # 只支持只有一个参数的函数
        if len(f.args) > 1:
            raise ValueError("Only functions with one argument are supported.")
        # 递归处理 f.args[0]，并应用反函数到 g_ys
        return _invert_complex(f.args[0],
                               imageset(Lambda(n, f.inverse()(n)), g_ys), symbol)

    # 如果 f 是指数函数或者 (f 是幂函数且基数是 S.Exp1)
    if isinstance(f, exp) or (f.is_Pow and f.base == S.Exp1):
        # 如果 g_ys 是 ImageSet 类型
        if isinstance(g_ys, ImageSet):
            # 可以解决形如 `(d*exp(exp(...(exp(a*x + b))...) + c)` 的格式
            g_ys_expr = g_ys.lamda.expr
            g_ys_vars = g_ys.lamda.variables
            k = Dummy('k{}'.format(len(g_ys_vars)))
            g_ys_vars_1 = (k,) + g_ys_vars
            # 计算指数函数的反函数结果
            exp_invs = Union(*[imageset(Lambda((g_ys_vars_1,), (I*(2*k*pi + arg(g_ys_expr))
                                         + log(Abs(g_ys_expr)))), S.Integers**(len(g_ys_vars_1)))])
            return _invert_complex(f.exp, exp_invs, symbol)

        # 如果 g_ys 是 FiniteSet 类型
        elif isinstance(g_ys, FiniteSet):
            # 计算指数函数的反函数结果
            exp_invs = Union(*[imageset(Lambda(n, I*(2*n*pi + arg(g_y)) +
                                               log(Abs(g_y))), S.Integers)
                               for g_y in g_ys if g_y != 0])
            return _invert_complex(f.exp, exp_invs, symbol)

    # 如果 f 是三角函数或双曲函数
    if isinstance(f, (TrigonometricFunction, HyperbolicFunction)):
         return _invert_trig_hyp_complex(f, g_ys, symbol)

    # 默认情况下返回 (f, g_ys)
    return (f, g_ys)
    """
    如果 g_ys 不是有限集，返回空集。否则返回包含解的 ConditionSet，其中包含所有必需的条件。

    """
    # 检查 g_ys 是否为有限集
    if not g_ys.is_FiniteSet:
        # 如果 g_ys 不是有限集，尝试在非负数区间内求解
        pos = Intersection(g_ys, Interval(0, S.Infinity))
        # 对函数 f 在非负数区间 pos 上求解
        parg = _invert_real(f, pos, symbol)
        # 对函数 -f 在非负数区间 pos 上求解
        narg = _invert_real(-f, pos, symbol)
        # 如果正负解的第一个元素不相等，则抛出未实现的错误
        if parg[0] != narg[0]:
            raise NotImplementedError
        # 返回正负解的并集
        return parg[0], Union(narg[1], parg[1])

    # 检查条件：所有条件必须为真。如果有任何未知条件，则返回这些条件，这些条件必须满足
    unknown = []
    for a in g_ys.args:
        # 对于数字，检查是否为非负数；对于其他对象，检查是否为正数
        ok = a.is_nonnegative if a.is_Number else a.is_positive
        # 如果条件未知，则加入未知列表
        if ok is None:
            unknown.append(a)
        # 如果条件为假，则返回空集
        elif not ok:
            return symbol, S.EmptySet
    # 如果存在未知条件，创建包含这些条件的 And 表达式
    if unknown:
        conditions = And(*[Contains(i, Interval(0, oo))
                           for i in unknown])
    else:
        conditions = True
    # 创建虚拟符号 n，用于后续条件的表达
    n = Dummy('n', real=True)
    # 不同于前面的方法：在正负 g_ys 的图像集上求解函数 f
    g_x, values = _invert_real(f, Union(
        imageset(Lambda(n, n), g_ys),
        imageset(Lambda(n, -n), g_ys)), symbol)
    # 返回求解结果以及包含条件和值的 ConditionSet
    return g_x, ConditionSet(g_x, conditions, values)
# 定义一个函数，用于检查给定表达式 `f` 在符号 `symbol` 替换为点 `p` 后是否为有限的
def domain_check(f, symbol, p):
    """Returns False if point p is infinite or any subexpression of f
    is infinite or becomes so after replacing symbol with p. If none of
    these conditions is met then True will be returned.

    Examples
    ========

    >>> from sympy import Mul, oo
    >>> from sympy.abc import x
    >>> from sympy.solvers.solveset import domain_check
    >>> g = 1/(1 + (1/(x + 1))**2)
    >>> domain_check(g, x, -1)
    False
    >>> domain_check(x**2, x, 0)
    True
    >>> domain_check(1/x, x, oo)
    False

    * The function relies on the assumption that the original form
      of the equation has not been changed by automatic simplification.

    >>> domain_check(x/x, x, 0) # x/x is automatically simplified to 1
    True

    * To deal with automatic evaluations use evaluate=False:

    >>> domain_check(Mul(x, 1/x, evaluate=False), x, 0)
    False
    """
    # 将输入的表达式 `f` 和点 `p` 转换为 SymPy 的表达式
    f, p = sympify(f), sympify(p)
    # 如果点 `p` 是无穷大，则返回 False
    if p.is_infinite:
        return False
    # 调用辅助函数 `_domain_check` 进行进一步检查
    return _domain_check(f, symbol, p)


def _domain_check(f, symbol, p):
    # helper for domain check
    # 如果 `f` 是原子并且是有限的，则返回 True
    if f.is_Atom and f.is_finite:
        return True
    # 如果将 `symbol` 替换为 `p` 后，`f` 变为无穷大，则返回 False
    elif f.subs(symbol, p).is_infinite:
        return False
    # 如果 `f` 是 Piecewise 类型的表达式
    elif isinstance(f, Piecewise):
        # 逐个检查 Piecewise 中的每个分支
        for expr, cond in f.args:
            # 将条件 `cond` 中的 `symbol` 替换为 `p`
            condsubs = cond.subs(symbol, p)
            # 如果条件为 False，则继续下一个分支
            if condsubs is S.false:
                continue
            # 如果条件为 True，则递归检查相应的表达式 `expr`
            elif condsubs is S.true:
                return _domain_check(expr, symbol, p)
            else:
                # 对于 Piecewise，我们不能确定哪个分支成立，因此暂且认为通过检查
                return True
    else:
        # TODO : We should not blindly recurse through all args of arbitrary expressions like this
        # 对于其他类型的表达式，递归检查所有的子表达式 `g`
        return all(_domain_check(g, symbol, p)
                   for g in f.args)


def _is_finite_with_finite_vars(f, domain=S.Complexes):
    """
    Return True if the given expression is finite. For symbols that
    do not assign a value for `complex` and/or `real`, the domain will
    be used to assign a value; symbols that do not assign a value
    for `finite` will be made finite. All other assumptions are
    left unmodified.
    """
    # 定义函数 assumptions，用于处理符号 s 的假设条件
    def assumptions(s):
        # 从符号 s 的原始假设中获取字典 A
        A = s.assumptions0
        # 设置 'finite' 属性，默认为 True
        A.setdefault('finite', A.get('finite', True))
        # 如果域 domain 是实数集 S.Reals 的子集
        if domain.is_subset(S.Reals):
            # 如果设置了 'real' 属性，则同时设定 'complex' 为 True
            A.setdefault('real', True)
        else:
            # 如果不是实数集的子集，则只设置 'complex' 为 True
            # 不改变 'real'，因为复数不意味着实数
            A.setdefault('complex', True)
        # 返回处理后的假设字典 A
        return A

    # 创建替换字典 reps，将每个自由符号 s 替换为对应的虚拟符号
    reps = {s: Dummy(**assumptions(s)) for s in f.free_symbols}
    # 使用替换字典 reps 替换函数 f 中的符号，并检查替换后结果是否有限
    return f.xreplace(reps).is_finite
# 判断方程是否属于给定函数类别的方程
def _is_function_class_equation(func_class, f, symbol):
    """ Tests whether the equation is an equation of the given function class.

    The given equation belongs to the given function class if it is
    comprised of functions of the function class which are multiplied by
    or added to expressions independent of the symbol. In addition, the
    arguments of all such functions must be linear in the symbol as well.

    Examples
    ========

    >>> from sympy.solvers.solveset import _is_function_class_equation
    >>> from sympy import tan, sin, tanh, sinh, exp
    >>> from sympy.abc import x
    >>> from sympy.functions.elementary.trigonometric import TrigonometricFunction
    >>> from sympy.functions.elementary.hyperbolic import HyperbolicFunction
    >>> _is_function_class_equation(TrigonometricFunction, exp(x) + tan(x), x)
    False
    >>> _is_function_class_equation(TrigonometricFunction, tan(x) + sin(x), x)
    True
    >>> _is_function_class_equation(TrigonometricFunction, tan(x**2), x)
    False
    >>> _is_function_class_equation(TrigonometricFunction, tan(x + 2), x)
    True
    >>> _is_function_class_equation(HyperbolicFunction, tanh(x) + sinh(x), x)
    True
    """
    # 如果方程是乘法或加法形式，递归检查每个参数是否属于给定的函数类别
    if f.is_Mul or f.is_Add:
        return all(_is_function_class_equation(func_class, arg, symbol)
                   for arg in f.args)

    # 如果是幂函数，检查指数部分是否不包含符号变量，否则返回 False
    if f.is_Pow:
        if not f.exp.has(symbol):
            return _is_function_class_equation(func_class, f.base, symbol)
        else:
            return False

    # 如果函数不包含符号变量，且是给定函数类别的实例，检查其参数是否是符号变量的线性多项式
    if not f.has(symbol):
        if isinstance(f, func_class):
            try:
                g = Poly(f.args[0], symbol)
                return g.degree() <= 1
            except PolynomialError:
                return False
        else:
            return False
    else:
        return False


def _solve_as_rational(f, symbol, domain):
    """ solve rational functions"""
    # 将有理函数整理并扩展，然后分离出分子和分母
    f = together(_mexpand(f, recursive=True), deep=True)
    g, h = fraction(f)
    # 如果分母不含符号变量，尝试将分子视为多项式解
    if not h.has(symbol):
        try:
            return _solve_as_poly(g, symbol, domain)
        except NotImplementedError:
            # 如果多项式的系数在某个环上，而找不到根，返回条件集合
            return ConditionSet(symbol, Eq(f, 0), domain)
        except CoercionFailed:
            # 包含无穷大、无穷小或非数字
            return S.EmptySet
    else:
        # 分母含符号变量时，分别解出有效解和无效解，返回有效解集合减去无效解集合
        valid_solns = _solveset(g, symbol, domain)
        invalid_solns = _solveset(h, symbol, domain)
        return valid_solns - invalid_solns


class _SolveTrig1Error(Exception):
    """Raised when _solve_trig1 heuristics do not apply"""

def _solve_trig(f, symbol, domain):
    """Function to call other helpers to solve trigonometric equations """
    # 如果 f 由单个三角函数组成（可能多次出现），应该通过直接反演或在适当的变量变换后反演来解决
    #
    # _solve_trig is currently only called by _solveset for trig/hyperbolic
    # functions of an argument linear in x. Inverting a symbolic argument should
    # include a guard against division by zero in order to have a result that is
    # consistent with similar processing done by _solve_trig1.
    # (Ideally _invert should add these conditions by itself.)
    # 初始化 trig_expr 和 count 变量
    trig_expr, count = None, 0
    # 使用前序遍历获取表达式树中的所有节点
    for expr in preorder_traversal(f):
        # 检查是否是三角函数或双曲函数，并且表达式包含 symbol 符号
        if isinstance(expr, (TrigonometricFunction,
                            HyperbolicFunction)) and expr.has(symbol):
            # 如果 trig_expr 还未赋值，则将其赋值为当前表达式，并且计数为 1
            if not trig_expr:
                trig_expr, count = expr, 1
            # 如果当前表达式与 trig_expr 相同，则计数增加
            elif expr == trig_expr:
                count += 1
            # 如果遇到不同类型的表达式，则重置 trig_expr 和 count，并且退出循环
            else:
                trig_expr, count = False, 0
                break
    # 如果 count 为 1，则直接求解反函数
    if count == 1:
        # 直接求解反函数，并返回条件集合
        x, sol = _invert(f, 0, symbol, domain)
        # 如果反函数求解后的变量 x 与 symbol 相同
        if x == symbol:
            # 初始化条件为 True
            cond = True
            # 如果 trig_expr 中除了 symbol 还有其他自由符号
            if trig_expr.free_symbols - {symbol}:
                # 将 trig_expr 的第一个参数作为独立变量 a，并且将余下部分作为 h
                a, h = trig_expr.args[0].as_independent(symbol, as_Add=True)
                # 将 h 作为独立变量 m
                m, h = h.as_independent(symbol, as_Add=False)
                # 将 m 化为分子和分母
                num, den = m.as_numer_denom()
                # 设置条件为分子和分母均不为零
                cond = Ne(num, 0) & Ne(den, 0)
            # 返回带有条件的符号解集合
            return ConditionSet(symbol, cond, sol)
        else:
            # 返回带有等式条件的符号解集合
            return ConditionSet(symbol, Eq(f, 0), domain)
    # 如果 count 大于 1，则通过变量替换求解
    elif count:
        # 创建一个虚拟变量 y
        y = Dummy('y')
        # 在 f 中用 y 替换 trig_expr，并求解新的表达式的解集
        f_cov = f.subs(trig_expr, y)
        sol_cov = solveset(f_cov, y, domain)
        # 如果 sol_cov 是有限集合，则返回集合中每个解的 _solve_trig 的结果的并集
        if isinstance(sol_cov, FiniteSet):
            return Union(
                *[_solve_trig(trig_expr-s, symbol, domain) for s in sol_cov])

    # 初始化 sol 为 None
    sol = None
    try:
        # 多个三角函数或双曲函数，通过重写为指数求解
        sol = _solve_trig1(f, symbol, domain)
    except _SolveTrig1Error:
        try:
            # 多个三角函数或双曲函数，通过重写为 tan(x/2) 求解
            sol = _solve_trig2(f, symbol, domain)
        except ValueError:
            # 抛出未实现的错误信息
            raise NotImplementedError(filldedent('''
                Solution to this kind of trigonometric equations
                is yet to be implemented'''))
    # 返回解集合
    return sol
def _solve_trig1(f, symbol, domain):
    """Primary solver for trigonometric and hyperbolic equations

    Returns either the solution set as a ConditionSet (auto-evaluated to a
    union of ImageSets if no variables besides 'symbol' are involved) or
    raises _SolveTrig1Error if f == 0 cannot be solved.

    Notes
    =====
    Algorithm:
    1. Do a change of variable x -> mu*x in arguments to trigonometric and
    hyperbolic functions, in order to reduce them to small integers. (This
    step is crucial to keep the degrees of the polynomials of step 4 low.)
    2. Rewrite trigonometric/hyperbolic functions as exponentials.
    3. Proceed to a 2nd change of variable, replacing exp(I*x) or exp(x) by y.
    4. Solve the resulting rational equation.
    5. Use invert_complex or invert_real to return to the original variable.
    6. If the coefficients of 'symbol' were symbolic in nature, add the
    necessary consistency conditions in a ConditionSet.

    """
    # Prepare change of variable based on function type
    x = Dummy('x')
    if _is_function_class_equation(HyperbolicFunction, f, symbol):
        # For hyperbolic functions, use exponential form
        cov = exp(x)
        inverter = invert_real if domain.is_subset(S.Reals) else invert_complex
    else:
        # For trigonometric functions, use exponential with imaginary unit
        cov = exp(I*x)
        inverter = invert_complex

    # Simplify the equation involving trigonometric and hyperbolic functions
    f = trigsimp(f)
    f_original = f
    # Find all trigonometric and hyperbolic functions in the equation
    trig_functions = f.atoms(TrigonometricFunction, HyperbolicFunction)
    trig_arguments = [e.args[0] for e in trig_functions]

    # Check if the equation simplification made it independent of 'symbol'
    if not any(a.has(symbol) for a in trig_arguments):
        return solveset(f_original, symbol, domain)

    denominators = []
    numerators = []
    # Extract coefficients of 'symbol' from trigonometric arguments
    for ar in trig_arguments:
        try:
            poly_ar = Poly(ar, symbol)
        except PolynomialError:
            raise _SolveTrig1Error("trig argument is not a polynomial")
        if poly_ar.degree() > 1:  # Ensure the degree of 'symbol' is <= 1
            raise _SolveTrig1Error("degree of variable must not exceed one")
        if poly_ar.degree() == 0:  # If degree is 0, skip
            continue
        c = poly_ar.all_coeffs()[0]   # Get the coefficient of 'symbol'
        numerators.append(fraction(c)[0])
        denominators.append(fraction(c)[1])

    # Determine the multiplier 'mu' for the change of variable
    mu = lcm(denominators)/gcd(numerators)
    # Substitute the variable with mu*x and perform further simplifications
    f = f.subs(symbol, mu*x)
    f = f.rewrite(exp)
    f = together(f)
    g, h = fraction(f)
    y = Dummy('y')
    g, h = g.expand(), h.expand()
    g, h = g.subs(cov, y), h.subs(cov, y)

    # Ensure the change of variable is valid
    if g.has(x) or h.has(x):
        raise _SolveTrig1Error("change of variable not possible")

    # Solve the simplified equations involving 'y'
    solns = solveset_complex(g, y) - solveset_complex(h, y)
    # Check if the solution is a ConditionSet (which should not occur)
    if isinstance(solns, ConditionSet):
        raise _SolveTrig1Error("polynomial has ConditionSet solution")
    # 如果 solns 是 FiniteSet 类型
    if isinstance(solns, FiniteSet):
        # 如果 solns 中包含 RootOf 对象，则抛出异常
        if any(isinstance(s, RootOf) for s in solns):
            raise _SolveTrig1Error("polynomial results in RootOf object")
        
        # 恢复变量的改变
        cov = cov.subs(x, symbol/mu)
        
        # 对每个 solns 中的元素应用 inverter 函数，并取返回结果的并集
        result = Union(*[inverter(cov, s, symbol)[1] for s in solns])
        
        # 如果 mu 中包含 Symbol 类型的符号
        if mu.has(Symbol):
            # 获取 mu 中的符号集合
            syms = (mu).atoms(Symbol)
            # 将 mu 表示为分子和分母
            munum, muden = fraction(mu)
            # 提取分子中独立于 syms 的部分作为条件的分子
            condnum = munum.as_independent(*syms, as_Add=False)[1]
            # 提取分母中独立于 syms 的部分作为条件的分母
            condden = muden.as_independent(*syms, as_Add=False)[1]
            # 构造条件：cond 表示分子和分母都不为零
            cond = And(Ne(condnum, 0), Ne(condden, 0))
        else:
            # 如果 mu 中不包含 Symbol 类型的符号，则条件为 True
            cond = True
        
        # 返回作为 ConditionSet 的符号、条件和结果的集合
        if domain is S.Complexes:
            # 如果 domain 是复数集合，则直接返回 ConditionSet
            return ConditionSet(symbol, cond, result)
        else:
            # 否则，返回 ConditionSet 的结果与 domain 的交集
            return ConditionSet(symbol, cond, Intersection(result, domain))
    
    # 如果 solns 是空集 S.EmptySet，则返回空集
    elif solns is S.EmptySet:
        return S.EmptySet
    
    # 如果 solns 既不是 FiniteSet 也不是空集，抛出异常
    else:
        raise _SolveTrig1Error("polynomial solutions must form FiniteSet")
# 解决三角函数方程的辅助函数，当第一个辅助函数失败时调用
def _solve_trig2(f, symbol, domain):
    # 简化三角函数表达式
    f = trigsimp(f)
    f_original = f
    # 提取表达式中的三角函数
    trig_functions = f.atoms(sin, cos, tan, sec, cot, csc)
    # 提取三角函数的参数
    trig_arguments = [e.args[0] for e in trig_functions]
    denominators = []  # 初始化分母列表
    numerators = []     # 初始化分子列表

    # todo: 如果使用 tanh 而不是 tan，类似地可以扩展此求解器以处理双曲函数
    if not trig_functions:
        # 如果没有三角函数，则返回条件集合
        return ConditionSet(symbol, Eq(f_original, 0), domain)

    # todo: 下面的预处理（提取分子、分母、gcd、lcm、mu 等）应该更新为 _solve_trig1 中的增强版本（见＃19507）
    for ar in trig_arguments:
        try:
            poly_ar = Poly(ar, symbol)
        except PolynomialError:
            raise ValueError("give up, we cannot solve if this is not a polynomial in x")
        if poly_ar.degree() > 1:  # 如果多项式的次数大于1，抛出错误
            raise ValueError("degree of variable inside polynomial should not exceed one")
        if poly_ar.degree() == 0:  # 如果多项式的次数为0，则跳过
            continue
        c = poly_ar.all_coeffs()[0]   # 获取 'symbol' 的系数
        try:
            numerators.append(Rational(c).p)  # 提取分子
            denominators.append(Rational(c).q)  # 提取分母
        except TypeError:
            return ConditionSet(symbol, Eq(f_original, 0), domain)

    x = Dummy('x')

    # 计算 mu 的值
    mu = Rational(2)*number_lcm(*denominators)/number_gcd(*numerators)
    # 替换符号 'symbol' 为 mu*x
    f = f.subs(symbol, mu*x)
    # 将表达式重写为关于 tan 的形式
    f = f.rewrite(tan)
    # 展开三角函数
    f = expand_trig(f)
    # 合并表达式
    f = together(f)

    # 分离分子和分母
    g, h = fraction(f)
    y = Dummy('y')
    g, h = g.expand(), h.expand()
    g, h = g.subs(tan(x), y), h.subs(tan(x), y)

    # 如果 g 或 h 中包含 x，则返回条件集合
    if g.has(x) or h.has(x):
        return ConditionSet(symbol, Eq(f_original, 0), domain)
    # 解方程 g(y) = 0 和 h(y) = 0
    solns = solveset(g, y, S.Reals) - solveset(h, y, S.Reals)

    if isinstance(solns, FiniteSet):
        # 如果解是有限集合，则计算结果
        result = Union(*[invert_real(tan(symbol/mu), s, symbol)[1]
                         for s in solns])
        dsol = invert_real(tan(symbol/mu), oo, symbol)[1]
        if degree(h) > degree(g):  # 如果分母的次数大于分子，则可能有额外的解在 Lim(denom-->oo)
            result = Union(result, dsol)
        return Intersection(result, domain)
    elif solns is S.EmptySet:
        # 如果解为空集，则返回空集
        return S.EmptySet
    else:
        # 否则返回条件集合
        return ConditionSet(symbol, Eq(f_original, 0), S.Reals)


# 使用多项式技术解方程，如果方程本身是多项式或者通过变量变换可以成为多项式
def _solve_as_poly(f, symbol, domain=S.Complexes):
    result = None
    # 如果多项式 f 是关于 symbol 的多项式
    if f.is_polynomial(symbol):
        # 求解多项式 f 在 symbol 变量上的根，包括三次、四次和五次方程
        solns = roots(f, symbol, cubics=True, quartics=True,
                      quintics=True, domain='EX')
        # 计算根的个数总和
        num_roots = sum(solns.values())
        # 如果多项式的阶数小于等于根的个数，则将根作为有限集合的结果
        if degree(f, symbol) <= num_roots:
            result = FiniteSet(*solns.keys())
        else:
            # 将多项式 f 转化为 Poly 对象
            poly = Poly(f, symbol)
            # 获取多项式的所有根
            solns = poly.all_roots()
            # 如果多项式的阶数小于等于根的个数，则将根作为有限集合的结果
            if poly.degree() <= len(solns):
                result = FiniteSet(*solns)
            else:
                # 否则将满足条件 Eq(f, 0) 的 symbol 构成的条件集合作为结果
                result = ConditionSet(symbol, Eq(f, 0), domain)
    else:
        # 如果 f 不是关于 symbol 的多项式，则转化为 Poly 对象
        poly = Poly(f)
        # 如果转化失败，则将满足条件 Eq(f, 0) 的 symbol 构成的条件集合作为结果
        if poly is None:
            result = ConditionSet(symbol, Eq(f, 0), domain)
        # 获取所有包含 symbol 的生成器
        gens = [g for g in poly.gens if g.has(symbol)]

        # 如果存在仅含有一个生成器的情况
        if len(gens) == 1:
            # 重新定义 poly 为以该生成器为变量的 Poly 对象
            poly = Poly(poly, gens[0])
            # 获取 poly 的生成器和阶数
            gen = poly.gen
            deg = poly.degree()
            # 将 poly 转化为表达式，并标记为复合多项式
            poly = Poly(poly.as_expr(), poly.gen, composite=True)
            # 求解复合多项式的所有根，并将结果作为有限集合的解
            poly_solns = FiniteSet(*roots(poly, cubics=True, quartics=True,
                                          quintics=True).keys())

            # 如果有限集合的解少于多项式的阶数，则将满足条件 Eq(f, 0) 的 symbol 构成的条件集合作为结果
            if len(poly_solns) < deg:
                result = ConditionSet(symbol, Eq(f, 0), domain)

            # 如果生成器不等于 symbol，则执行如下操作
            if gen != symbol:
                # 定义 y 为虚拟变量
                y = Dummy('y')
                # 根据 domain 的子集，选择实数或复数求逆操作
                inverter = invert_real if domain.is_subset(S.Reals) else invert_complex
                # 获取左右两边的等式
                lhs, rhs_s = inverter(gen, y, symbol)
                # 如果左侧等于 symbol，则执行以下操作
                if lhs == symbol:
                    # 将 rhs_s 中的 y 替换为 poly_solns 中的每个 s，并取并集作为结果
                    result = Union(*[rhs_s.subs(y, s) for s in poly_solns])
                    # 如果结果是有限集合，并且 gen 是 Pow 类型且 gen.base 是 Rational 类型，则对结果进行对数扩展
                    if isinstance(result, FiniteSet) and isinstance(gen, Pow
                            ) and gen.base.is_Rational:
                        result = FiniteSet(*[expand_log(i) for i in result])
                else:
                    # 否则将满足条件 Eq(f, 0) 的 symbol 构成的条件集合作为结果
                    result = ConditionSet(symbol, Eq(f, 0), domain)
        else:
            # 如果没有仅含一个生成器的情况，则将满足条件 Eq(f, 0) 的 symbol 构成的条件集合作为结果
            result = ConditionSet(symbol, Eq(f, 0), domain)

    # 如果结果不为空
    if result is not None:
        # 如果结果是有限集合
        if isinstance(result, FiniteSet):
            # 对解进行简化，如将 -sqrt(-I) 简化为 sqrt(2)/2 - sqrt(2)*I/2
            # 对于包含符号或未定义函数的解不进行展开，因为展开会使得解更复杂
            # 例如，expand_complex(a) 返回 re(a) + I*im(a)
            if all(s.atoms(Symbol, AppliedUndef) == set() and not isinstance(s, RootOf)
                   for s in result):
                # 定义虚拟变量 s
                s = Dummy('s')
                # 将每个解 s 进行复数扩展，并将结果作为 imageset 的输出
                result = imageset(Lambda(s, expand_complex(s)), result)
        # 如果结果是有限集合，并且 domain 不是 S.Complexes，则避免与 S.Complexes 的无谓交集
        if isinstance(result, FiniteSet) and domain != S.Complexes:
            # 实际条件应在其他地方处理，这里只需要交集结果
            result = result.intersection(domain)
        # 返回结果
        return result
    else:
        # 如果结果为空，则将满足条件 Eq(f, 0) 的 symbol 构成的条件集合作为结果
        return ConditionSet(symbol, Eq(f, 0), domain)
# 解决带有根式的方程的辅助函数
def _solve_radical(f, unradf, symbol, solveset_solver):
    """ Helper function to solve equations with radicals """
    # 解压缩 unradf，获得方程和附加信息
    res = unradf
    eq, cov = res if res else (f, [])
    # 如果没有附加信息
    if not cov:
        # 计算解集，去除分母中的解集
        result = solveset_solver(eq, symbol) - \
            Union(*[solveset_solver(g, symbol) for g in denoms(f, symbol)])
    else:
        y, yeq = cov
        # 如果 y - I 无解
        if not solveset_solver(y - I, y):
            # 创建一个实数变量 yreal
            yreal = Dummy('yreal', real=True)
            # 替换 yeq 和 eq 中的 y 为 yreal
            yeq = yeq.xreplace({y: yreal})
            eq = eq.xreplace({y: yreal})
            y = yreal
        # 求解 yeq 关于 symbol 的解集
        g_y_s = solveset_solver(yeq, symbol)
        # 求解 eq 关于 y 的解集
        f_y_sols = solveset_solver(eq, y)
        # 将 f_y_sols 中每个解 g_y 转换为 imageset(Lambda(y, g_y))
        result = Union(*[imageset(Lambda(y, g_y), f_y_sols)
                         for g_y in g_y_s])

    # 检查解集 solutions 中的 FiniteSet
    def check_finiteset(solutions):
        f_set = []  # 存储 FiniteSet 的解
        c_set = []  # 存储 ConditionSet 的解
        for s in solutions:
            # 检查是否为 f 的解
            if checksol(f, symbol, s):
                f_set.append(s)
            else:
                c_set.append(s)
        # 返回 FiniteSet(*f_set) 加上 ConditionSet(symbol, Eq(f, 0), FiniteSet(*c_set))
        return FiniteSet(*f_set) + ConditionSet(symbol, Eq(f, 0), FiniteSet(*c_set))

    # 检查不同类型的解集 solutions
    def check_set(solutions):
        # 如果 solutions 是空集，则直接返回
        if solutions is S.EmptySet:
            return solutions
        # 如果 solutions 是 ConditionSet，则返回它自身
        elif isinstance(solutions, ConditionSet):
            # XXX: 或许应该检查基本集？
            return solutions
        # 如果 solutions 是 FiniteSet，则调用 check_finiteset 进行处理
        elif isinstance(solutions, FiniteSet):
            return check_finiteset(solutions)
        # 如果 solutions 是 Complement，则递归地应用 check_set 到 A 和 B
        elif isinstance(solutions, Complement):
            A, B = solutions.args
            return Complement(check_set(A), B)
        # 如果 solutions 是 Union，则递归地应用 check_set 到每个子集
        elif isinstance(solutions, Union):
            return Union(*[check_set(s) for s in solutions.args])
        else:
            # XXX: 这里应该检查更多的情况。目前测试套件中涉及的情况已经列出。
            return solutions

    # 对 result 进行最终的检查，得到最终的解集
    solution_set = check_set(result)

    return solution_set


# 解决涉及绝对值函数的方程的辅助函数
def _solve_abs(f, symbol, domain):
    """ Helper function to solve equation involving absolute value function """
    # 如果 domain 不是实数集，则抛出 ValueError 异常
    if not domain.is_subset(S.Reals):
        raise ValueError(filldedent('''
            Absolute values cannot be inverted in the
            complex domain.'''))
    # 匹配模式 p*Abs(q) + r
    p, q, r = Wild('p'), Wild('q'), Wild('r')
    pattern_match = f.match(p*Abs(q) + r) or {}
    # 获取匹配后的 p, q, r 值，如果没有则使用默认值 S.Zero
    f_p, f_q, f_r = [pattern_match.get(i, S.Zero) for i in (p, q, r)]
    # 如果 f_p 和 f_q 都不是零多项式
    if not (f_p.is_zero or f_q.is_zero):
        # 计算 f_q 的连续定义域
        domain = continuous_domain(f_q, symbol, domain)
        
        # 导入不等式求解模块中的 solve_univariate_inequality 函数
        from .inequalities import solve_univariate_inequality
        
        # 解 f_q >= 0 的一元不等式，得到正条件
        q_pos_cond = solve_univariate_inequality(f_q >= 0, symbol,
                                                 relational=False, domain=domain, continuous=True)
        
        # 得到负条件为正条件的补集
        q_neg_cond = q_pos_cond.complement(domain)

        # 解方程 f_p*f_q + f_r = 0，并与正条件相交，得到正解集
        sols_q_pos = solveset_real(f_p*f_q + f_r,
                                   symbol).intersect(q_pos_cond)
        
        # 解方程 f_p*(-f_q) + f_r = 0，并与负条件相交，得到负解集
        sols_q_neg = solveset_real(f_p*(-f_q) + f_r,
                                   symbol).intersect(q_neg_cond)
        
        # 返回正解集和负解集的并集
        return Union(sols_q_pos, sols_q_neg)
    else:
        # 如果 f_p 或 f_q 是零多项式，则返回方程 f = 0 的条件集合
        return ConditionSet(symbol, Eq(f, 0), domain)
# Helper function to solve equations using the principle of "Decomposition and Rewriting".
# This function takes a function `f`, a symbol `symbol`, and a domain `domain`.
def solve_decomposition(f, symbol, domain):
    """
    Function to solve equations via the principle of "Decomposition
    and Rewriting".

    Examples
    ========
    >>> from sympy import exp, sin, Symbol, pprint, S
    >>> from sympy.solvers.solveset import solve_decomposition as sd
    >>> x = Symbol('x')
    >>> f1 = exp(2*x) - 3*exp(x) + 2
    >>> sd(f1, x, S.Reals)
    {0, log(2)}
    >>> f2 = sin(x)**2 + 2*sin(x) + 1
    >>> pprint(sd(f2, x, S.Reals), use_unicode=False)
              3*pi
    {2*n*pi + ---- | n in Integers}
               2
    >>> f3 = sin(x + 2)
    >>> pprint(sd(f3, x, S.Reals), use_unicode=False)
    {2*n*pi - 2 | n in Integers} U {2*n*pi - 2 + pi | n in Integers}

    """
    # Importing the necessary function from sympy.solvers.decompogen
    from sympy.solvers.decompogen import decompogen

    # Decompose the given function `f` into simpler functions using `symbol` as the variable
    g_s = decompogen(f, symbol)

    # Initialize `y_s` as the set containing 0, which is the target for solving f = 0
    y_s = FiniteSet(0)

    # Iterate over each decomposed function `g` in `g_s`
    for g in g_s:
        # Calculate the range of function `g` over the domain `domain`
        frange = function_range(g, symbol, domain)
        # Intersect `frange` with `y_s`
        y_s = Intersection(frange, y_s)

        # Initialize `result` as an empty set
        result = S.EmptySet

        # Depending on the type of `y_s`, find solutions for `g = y_s` or `g = 0`
        if isinstance(y_s, FiniteSet):
            for y in y_s:
                solutions = solveset(Eq(g, y), symbol, domain)
                if not isinstance(solutions, ConditionSet):
                    result += solutions
        else:
            if isinstance(y_s, ImageSet):
                iter_iset = (y_s,)
            elif isinstance(y_s, Union):
                iter_iset = y_s.args
            elif y_s is S.EmptySet:
                # If `y_s` is empty, no solutions exist in the given domain
                return S.EmptySet

            # Iterate over each set in `iter_iset`
            for iset in iter_iset:
                # Solve `iset.lambda.expr = g` for `symbol` in `domain`
                new_solutions = solveset(Eq(iset.lamda.expr, g), symbol, domain)
                dummy_var = tuple(iset.lamda.expr.free_symbols)[0]
                (base_set,) = iset.base_sets

                # Depending on the type of `new_solutions`, handle accordingly
                if isinstance(new_solutions, FiniteSet):
                    new_exprs = new_solutions
                elif isinstance(new_solutions, Intersection):
                    if isinstance(new_solutions.args[1], FiniteSet):
                        new_exprs = new_solutions.args[1]

                # Add new expressions to `result`
                for new_expr in new_exprs:
                    result += ImageSet(Lambda(dummy_var, new_expr), base_set)

        # If `result` is empty, return a ConditionSet indicating no solutions for `f = 0`
        if result is S.EmptySet:
            return ConditionSet(symbol, Eq(f, 0), domain)

        # Update `y_s` with `result`
        y_s = result

    # Return the final set of solutions
    return y_s


def _solveset(f, symbol, domain, _check=False):
    """Helper for solveset to return a result from an expression
    that has already been sympify'ed and is known to contain the
    given symbol."""
    # _check controls whether the answer is checked or not
    from sympy.simplify.simplify import signsimp

    if isinstance(f, BooleanTrue):
        return domain
    # 保存原始表达式以备后用
    orig_f = f
    # 如果 f 是乘法表达式
    if f.is_Mul:
        # 将 f 分解为系数和剩余部分，相对于给定符号的独立部分
        coeff, f = f.as_independent(symbol, as_Add=False)
        # 如果系数是无穷大或者无穷小，则重新组合原始表达式
        if coeff in {S.ComplexInfinity, S.NegativeInfinity, S.Infinity}:
            f = together(orig_f)
    # 如果 f 是加法表达式
    elif f.is_Add:
        # 将 f 分解为符号相关和剩余部分
        a, h = f.as_independent(symbol)
        # 从剩余部分中分解出符号相关和一个独立的数值部分
        m, h = h.as_independent(symbol, as_Add=False)
        # 如果独立的数值部分不是无穷大、零或无穷小
        if m not in {S.ComplexInfinity, S.Zero, S.Infinity, S.NegativeInfinity}:
            # 重新定义 f 为 a/m + h，其中 a 是原始符号相关部分
            f = a/m + h  # XXX condition `m != 0` should be added to soln

    # 设置解方程的函数
    solver = lambda f, x, domain=domain: _solveset(f, x, domain)
    # 设置反转方程的函数
    inverter = lambda f, rhs, symbol: _invert(f, rhs, symbol, domain)

    # 初始化结果为空集
    result = S.EmptySet

    # 如果 f 展开后为零，返回定义域
    if f.expand().is_zero:
        return domain
    # 如果 f 不包含给定符号，返回空集
    elif not f.has(symbol):
        return S.EmptySet
    # 如果 f 是乘法表达式且所有乘法项在有限变量定义域内
    elif f.is_Mul and all(_is_finite_with_finite_vars(m, domain) for m in f.args):
        # 对于有限输入，如果 f(x)*g(x)==0 的解与 Union(f(x)==0, g(x)==0) 不同
        # 一般情况下，g(x) 在 f(x)==0 的值处可以无限增长。只有在 f 和 g 都在有限输入下是有限的情况下才使用这个技巧。
        result = Union(*[solver(m, symbol) for m in f.args])
    # 如果 f 是三角函数或双曲函数的方程
    elif (_is_function_class_equation(TrigonometricFunction, f, symbol) or \
            _is_function_class_equation(HyperbolicFunction, f, symbol)):
        result = _solve_trig(f, symbol, domain)
    # 如果 f 是一个复数的参数
    elif isinstance(f, arg):
        a = f.args[0]
        # 结果是实部大于零且虚部的解集的交集
        result = Intersection(_solveset(re(a) > 0, symbol, domain),
                              _solveset(im(a), symbol, domain))
    # 如果 f 是分段函数
    elif f.is_Piecewise:
        # 获取表达式-集合对，并解决每个对中的表达式
        expr_set_pairs = f.as_expr_set_pairs(domain)
        for (expr, in_set) in expr_set_pairs:
            if in_set.is_Relational:
                in_set = in_set.as_set()
            # 将每个表达式的解集合并到结果中
            solns = solver(expr, symbol, in_set)
            result += solns
    # 如果 f 是方程
    elif isinstance(f, Eq):
        # 求解方程 f.lhs + (-f.rhs) == 0 的解集
        result = solver(Add(f.lhs, -f.rhs, evaluate=False), symbol, domain)

    # 如果 f 是关系式
    elif f.is_Relational:
        # 导入不等式求解函数，并尝试解决不等式
        from .inequalities import solve_univariate_inequality
        try:
            result = solve_univariate_inequality(
                f, symbol, domain=domain, relational=False)
        except NotImplementedError:
            # 如果无法解决，返回符号和关系条件的集合
            result = ConditionSet(symbol, f, domain)
        return result
    # 如果 f 是模运算
    elif _is_modular(f, symbol):
        # 解决模运算
        result = _solve_modular(f, symbol, domain)

    # 如果结果是条件集合
    if isinstance(result, ConditionSet):
        if isinstance(f, Expr):
            num, den = f.as_numer_denom()
            # 如果分母中包含符号
            if den.has(symbol):
                # 解决分子方程
                _result = _solveset(num, symbol, domain)
                # 如果结果不是条件集合，计算奇点
                if not isinstance(_result, ConditionSet):
                    singularities = _solveset(den, symbol, domain)
                    result = _result - singularities
    #`
    # 检查 _check 是否为 True
    if _check:
        # 如果 result 是 ConditionSet 类型，表示条件集，直接返回 result，不进行处理
        if isinstance(result, ConditionSet):
            # it wasn't solved or has enumerated all conditions
            # -- leave it alone
            return result

        # 除去除了包含符号的核心部分，供测试使用
        # 如果 orig_f 是 Expr 类型，提取其与符号无关的部分
        if isinstance(orig_f, Expr):
            # 将 orig_f 转换为不含符号的表达式，返回与符号独立的部分
            fx = orig_f.as_independent(symbol, as_Add=True)[1]
            # 再次将其转换为不含符号的表达式，返回与符号独立的部分
            fx = fx.as_independent(symbol, as_Add=False)[1]
        else:
            fx = orig_f

        # 如果 result 是 FiniteSet 类型，进行结果有效性检查
        if isinstance(result, FiniteSet):
            # 检查 result 中的所有元素是否为无效解，使用域检查函数 domain_check
            result = FiniteSet(*[s for s in result
                      if isinstance(s, RootOf)  # 检查元素是否为 RootOf 类型
                      or domain_check(fx, symbol, s)])  # 检查元素是否在域中有效

    #
def _is_modular(f, symbol):
    """
    Helper function to check if the equation is a modular equation of the form:
    A - Mod(B, C) = 0

    A -> This can or cannot be a function of symbol.
    B -> This is surely a function of symbol.
    C -> It is an integer.

    Parameters
    ==========

    f : Expr
        The equation to be checked.

    symbol : Symbol
        The concerned variable for which the equation is to be checked.

    Examples
    ========

    >>> from sympy import symbols, exp, Mod
    >>> from sympy.solvers.solveset import _is_modular as check
    >>> x, y = symbols('x y')
    >>> check(Mod(x, 3) - 1, x)
    True
    >>> check(Mod(x, 3) - 1, y)
    False
    >>> check(Mod(x, 3)**2 - 5, x)
    False
    >>> check(Mod(x, 3)**2 - y, x)
    False
    >>> check(exp(Mod(x, 3)) - 1, x)
    False
    >>> check(Mod(3, y) - 1, y)
    False
    """

    # Check if the equation f contains the Mod function
    if not f.has(Mod):
        return False

    # Extract all instances of Mod from f
    modterms = list(f.atoms(Mod))

    # Check conditions for a modular equation
    return (len(modterms) == 1 and  # only one Mod should be present
            modterms[0].args[0].has(symbol) and  # B -> function of symbol
            modterms[0].args[1].is_integer and  # C -> to be an integer
            any(isinstance(term, Mod)
                for term in list(_term_factors(f)))  # free from other funcs
            )


def _invert_modular(modterm, rhs, n, symbol):
    """
    Helper function to invert a modular equation of the form:
    Mod(a, m) - rhs = 0

    Generally it is inverted as (a, ImageSet(Lambda(n, m*n + rhs), S.Integers)).
    More simplified form will be returned if possible.

    If it is not invertible then (modterm, rhs) is returned.

    The following cases arise while inverting equation Mod(a, m) - rhs = 0:

    1. If a is symbol then m*n + rhs is the required solution.

    2. If a is an instance of Add then we try to find two symbol independent
       parts of a and the symbol independent part gets transferred to the other
       side and again _invert_modular is called on the symbol dependent part.

    3. If a is an instance of Mul then same as we did in Add, we separate out the
       symbol dependent and symbol independent parts and transfer the symbol
       independent part to the rhs with the help of invert and again _invert_modular
       is called on the symbol dependent part.
    """
    a, m = modterm.args
    # 将 modterm 分解为 a 和 m，a 是表达式的一部分，m 是模数

    if rhs.is_integer is False:
        # 如果 rhs 不是整数，则返回符号和空集
        return symbol, S.EmptySet

    if rhs.is_real is False or any(term.is_real is False
            for term in list(_term_factors(a))):
        # 如果 rhs 不是实数，或者 a 中的任何项不是实数，则返回原始的 modterm 和 rhs
        return modterm, rhs

    if abs(rhs) >= abs(m):
        # 如果 rhs 的绝对值大于等于 m，则返回符号和空集
        return symbol, S.EmptySet

    if a == symbol:
        # 如果 a 等于 symbol，则返回符号和形如 m*n + rhs 的整数图像集
        return symbol, ImageSet(Lambda(n, m*n + rhs), S.Integers)

    if a.is_Add:
        # 如果 a 是加法
        g, h = a.as_independent(symbol)
        # 将 a 拆分成 g 和 h，其中 g 是独立于 symbol 的部分
        if g is not S.Zero:
            x_indep_term = rhs - Mod(g, m)
            # 计算独立于 symbol 的术语 x_indep_term
            return _invert_modular(Mod(h, m), Mod(x_indep_term, m), n, symbol)

    if a.is_Mul:
        # 如果 a 是乘法
        g, h = a.as_independent(symbol)
        # 将 a 拆分成 g 和 h，其中 g 是独立于 symbol 的部分
        if g is not S.One:
            x_indep_term = rhs*invert(g, m)
            # 计算独立于 symbol 的术语 x_indep_term
            return _invert_modular(Mod(h, m), Mod(x_indep_term, m), n, symbol)
    # 检查是否表达式 a 是幂运算表达式
    if a.is_Pow:
        # 将 a 表达式拆解为底数 base 和指数 expo
        base, expo = a.args
        # 如果指数 expo 中包含 symbol，但底数 base 中不包含 symbol
        if expo.has(symbol) and not base.has(symbol):
            # 对于特定条件下的简单情况，直接返回结果 modterm, rhs
            if not m.is_Integer and rhs.is_Integer and a.base.is_Integer:
                return modterm, rhs

            # 计算 mdiv，通过除以 m.p 和 rhs.p 的最大公约数使得 m 和 rhs 变得互质
            mdiv = m.p // number_gcd(m.p, rhs.p)
            try:
                # 计算离散对数 remainder = log_{a.base}(mdiv) mod rhs.p
                remainder = discrete_log(mdiv, rhs.p, a.base.p)
            except ValueError:  # 如果离散对数不存在，则返回 modterm, rhs
                return modterm, rhs

            # 计算指数 expo 的最小周期 period，并用其构造解的集合
            period = totient(m)
            for p in divisors(period):
                # 寻找比 totient(m) 更小的周期 p，使得 a.base^p ≡ 1 (mod m / number_gcd(m.p, a.base.p))
                if pow(a.base, p, m // number_gcd(m.p, a.base.p)) == 1:
                    period = p
                    break

            # 返回解集合，形式为 ImageSet(Lambda(n, period*n + remainder), S.Naturals0)
            return expo, ImageSet(Lambda(n, period*n + remainder), S.Naturals0)
        
        # 如果底数 base 中包含 symbol，但指数 expo 中不包含 symbol
        elif base.has(symbol) and not expo.has(symbol):
            try:
                # 计算满足条件的余根列表 remainder_list = nthroot_mod(rhs, expo, m, all_roots=True)
                remainder_list = nthroot_mod(rhs, expo, m, all_roots=True)
                # 如果 remainder_list 为空，则返回 symbol, S.EmptySet
                if remainder_list == []:
                    return symbol, S.EmptySet
            except (ValueError, NotImplementedError):
                return modterm, rhs
            
            # 初始化解集合 g_n 为空集合 S.EmptySet
            g_n = S.EmptySet
            # 遍历余根列表 remainder_list，并构造解的集合 g_n
            for rem in remainder_list:
                g_n += ImageSet(Lambda(n, m*n + rem), S.Integers)
            # 返回解集合 base, g_n
            return base, g_n

    # 如果以上条件均不满足，则返回默认结果 modterm, rhs
    return modterm, rhs
# 定义解决形如 A - Mod(B, C) = 0 的模方程的辅助函数，其中 A 可以或不可以是符号的函数，B 确定是符号的函数，而 C 是整数。
def _solve_modular(f, symbol, domain):
    r"""
    # 解决模方程 f = 0 的辅助函数，其中 f 是要解的模方程，形式为 Mod(A, C) - B = 0。
    # 当 A 不是符号的函数时，_solve_modular 目前只能解决这种情况。

    Parameters
    ==========

    f : Expr
        要解决的模方程，形式为 f = 0

    symbol : Symbol
        要解的方程中的变量。

    domain : Set
        方程解的集合。必须是整数的子集。

    Returns
    =======

    满足给定模方程的整数解集合。
    如果方程无解，则返回一个 ConditionSet。

    Examples
    ========

    >>> from sympy.solvers.solveset import _solve_modular as solve_modulo
    >>> from sympy import S, Symbol, sin, Intersection, Interval, Mod
    >>> x = Symbol('x')
    >>> solve_modulo(Mod(5*x - 8, 7) - 3, x, S.Integers)
    ImageSet(Lambda(_n, 7*_n + 5), Integers)
    >>> solve_modulo(Mod(5*x - 8, 7) - 3, x, S.Reals)  # domain should be subset of integers.
    ConditionSet(x, Eq(Mod(5*x + 6, 7) - 3, 0), Reals)
    >>> solve_modulo(-7 + Mod(x, 5), x, S.Integers)
    EmptySet
    >>> solve_modulo(Mod(12**x, 21) - 18, x, S.Integers)
    ImageSet(Lambda(_n, 6*_n + 2), Naturals0)
    >>> solve_modulo(Mod(sin(x), 7) - 3, x, S.Integers) # not solvable
    ConditionSet(x, Eq(Mod(sin(x), 7) - 3, 0), Integers)
    >>> solve_modulo(3 - Mod(x, 5), x, Intersection(S.Integers, Interval(0, 100)))
    Intersection(ImageSet(Lambda(_n, 5*_n + 3), Integers), Range(0, 101, 1))
    """
    # 从 f 中提取 modterm 和 g_y
    unsolved_result = ConditionSet(symbol, Eq(f, 0), domain)
    modterm = list(f.atoms(Mod))[0]  # 提取模运算的项
    rhs = -S.One*(f.subs(modterm, S.Zero))  # 计算右侧的值

    if f.as_coefficients_dict()[modterm].is_negative:
        # 检查主方程中 modterm 的系数是否为负数
        rhs *= -S.One

    if not domain.is_subset(S.Integers):
        return unsolved_result  # 如果域不是整数的子集，则返回未解决的结果

    if rhs.has(symbol):
        # 如果 rhs 中包含 symbol，则无法解决
        return unsolved_result

    n = Dummy('n', integer=True)
    f_x, g_n = _invert_modular(modterm, rhs, n, symbol)  # 使用 _invert_modular 函数反转模方程

    if f_x == modterm and g_n == rhs:
        return unsolved_result  # 如果反转后的结果与原始 modterm 和 rhs 相同，则返回未解决的结果

    if f_x == symbol:
        if domain is not S.Integers:
            return domain.intersect(g_n)  # 如果 domain 不是整数，则返回 domain 与 g_n 的交集
        return g_n  # 返回 g_n
    # 检查 g_n 是否为 ImageSet 类型的实例
    if isinstance(g_n, ImageSet):
        # 获取 g_n 对象中的 lambda 表达式
        lamda_expr = g_n.lamda.expr
        # 获取 g_n 对象中 lambda 表达式的变量
        lamda_vars = g_n.lamda.variables
        # 获取 g_n 对象中的基础集合
        base_sets = g_n.base_sets
        # 解方程 f_x - lamda_expr = 0，返回整数解集合
        sol_set = _solveset(f_x - lamda_expr, symbol, S.Integers)
        
        # 如果解集合为有限集（FiniteSet 类型）
        if isinstance(sol_set, FiniteSet):
            # 初始化空集 tmp_sol
            tmp_sol = S.EmptySet
            # 遍历解集合中的每个解 sol
            for sol in sol_set:
                # 将 ImageSet(Lambda(lamda_vars, sol), *base_sets) 加入 tmp_sol
                tmp_sol += ImageSet(Lambda(lamda_vars, sol), *base_sets)
            # 更新 sol_set 为合并后的 tmp_sol
            sol_set = tmp_sol
        else:
            # 如果解集合为无限集，将其作为 ImageSet(Lambda(lamda_vars, sol_set), *base_sets)
            sol_set = ImageSet(Lambda(lamda_vars, sol_set), *base_sets)
        
        # 返回 domain 和 sol_set 的交集
        return domain.intersect(sol_set)

    # 如果 g_n 不是 ImageSet 类型的实例，返回未解决的结果 unsolved_result
    return unsolved_result
def _term_factors(f):
    """
    Iterator to get the factors of all terms present
    in the given equation.

    Parameters
    ==========
    f : Expr
        Equation that needs to be addressed

    Returns
    =======
    Factors of all terms present in the equation.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.solvers.solveset import _term_factors
    >>> x = symbols('x')
    >>> list(_term_factors(-2 - x**2 + x*(x + 1)))
    [-2, -1, x**2, x, x + 1]
    """
    # 使用 Add.make_args 方法将表达式拆分为加法操作数，并迭代处理每个加法操作数
    for add_arg in Add.make_args(f):
        # 使用 Mul.make_args 方法将每个加法操作数拆分为乘法操作数，并迭代返回每个乘法操作数
        yield from Mul.make_args(add_arg)


def _solve_exponential(lhs, rhs, symbol, domain):
    r"""
    Helper function for solving (supported) exponential equations.

    Exponential equations are the sum of (currently) at most
    two terms with one or both of them having a power with a
    symbol-dependent exponent.

    For example

    .. math:: 5^{2x + 3} - 5^{3x - 1}

    .. math:: 4^{5 - 9x} - e^{2 - x}

    Parameters
    ==========

    lhs, rhs : Expr
        The exponential equation to be solved, `lhs = rhs`

    symbol : Symbol
        The variable in which the equation is solved

    domain : Set
        A set over which the equation is solved.

    Returns
    =======

    A set of solutions satisfying the given equation.
    A ``ConditionSet`` if the equation is unsolvable or
    if the assumptions are not properly defined, in that case
    a different style of ``ConditionSet`` is returned having the
    solution(s) of the equation with the desired assumptions.

    Examples
    ========

    >>> from sympy.solvers.solveset import _solve_exponential as solve_expo
    >>> from sympy import symbols, S
    >>> x = symbols('x', real=True)
    >>> a, b = symbols('a b')
    >>> solve_expo(2**x + 3**x - 5**x, 0, x, S.Reals)  # not solvable
    ConditionSet(x, Eq(2**x + 3**x - 5**x, 0), Reals)
    >>> solve_expo(a**x - b**x, 0, x, S.Reals)  # solvable but incorrect assumptions
    ConditionSet(x, (a > 0) & (b > 0), {0})
    >>> solve_expo(3**(2*x) - 2**(x + 3), 0, x, S.Reals)
    {-3*log(2)/(-2*log(3) + log(2))}
    >>> solve_expo(2**x - 4**x, 0, x, S.Reals)
    {0}

    * Proof of correctness of the method

    The logarithm function is the inverse of the exponential function.
    The defining relation between exponentiation and logarithm is:

    .. math:: {\log_b x} = y \enspace if \enspace b^y = x

    Therefore if we are given an equation with exponent terms, we can
    convert every term to its corresponding logarithmic form. This is
    achieved by taking logarithms and expanding the equation using
    logarithmic identities so that it can easily be handled by ``solveset``.

    For example:

    .. math:: 3^{2x} = 2^{x + 3}

    Taking log both sides will reduce the equation to

    .. math:: (2x)\log(3) = (x + 3)\log(2)

    This form can be easily handed by ``solveset``.
    """
    # 创建一个 ConditionSet 对象，用于未解决的结果，假设方程无解或假设定义不正确
    unsolved_result = ConditionSet(symbol, Eq(lhs - rhs, 0), domain)
    # 对 lhs 进行幂展开处理
    newlhs = powdenest(lhs)
    # 如果 lhs 不等于 newlhs，则尝试因式分解新的表达式
    if lhs != newlhs:
        # it may also be advantageous to factor the new expr
        neweq = factor(newlhs - rhs)
        # 如果新表达式不等于原始的 lhs - rhs，则使用 _solveset 再次尝试求解
        if neweq != (lhs - rhs):
            return _solveset(neweq, symbol, domain)  # try again with _solveset

    # 如果 lhs 不是 Add 类型或者不是包含两个参数的 Add 类型，则返回未解决的结果
    if not (isinstance(lhs, Add) and len(lhs.args) == 2):
        # solving for the sum of more than two powers is possible
        # but not yet implemented
        return unsolved_result

    # 如果 rhs 不等于 0，则返回未解决的结果
    if rhs != 0:
        return unsolved_result

    # 将 lhs 的参数列表排序后，分别赋值给 a 和 b
    a, b = list(ordered(lhs.args))
    # 将 a 和 b 分别作为独立项和符号的独立部分获取出来
    a_term = a.as_independent(symbol)[1]
    b_term = b.as_independent(symbol)[1]

    # 将 a_term 和 b_term 分别作为基数和指数获取出来
    a_base, a_exp = a_term.as_base_exp()
    b_base, b_exp = b_term.as_base_exp()

    # 如果 domain 是实数集的子集，则设置条件为：
    if domain.is_subset(S.Reals):
        conditions = And(
            a_base > 0,
            b_base > 0,
            Eq(im(a_exp), 0),
            Eq(im(b_exp), 0))
    else:
        # 否则设置条件为基数不等于 0
        conditions = And(
            Ne(a_base, 0),
            Ne(b_base, 0))

    # 分别对 L = expand_log(log(a), force=True) 和 R = expand_log(log(-b), force=True) 进行计算
    L, R = (expand_log(log(i), force=True) for i in (a, -b))
    # 使用 _solveset 求解 L - R = 0 的解集
    solutions = _solveset(L - R, symbol, domain)

    # 返回符号、条件和解集构成的 ConditionSet 对象
    return ConditionSet(symbol, conditions, solutions)
# 检查给定表达式是否包含指定变量的指数形式，返回布尔值
def _is_exponential(f, symbol):
    # 初始化返回值为 False
    rv = False
    # 遍历每个表达式项的因子
    for expr_arg in _term_factors(f):
        # 如果表达式项不包含指定变量，则跳过
        if symbol not in expr_arg.free_symbols:
            continue
        # 如果表达式项是幂运算且指数部分包含指定变量，或者是指数函数
        if (isinstance(expr_arg, Pow) and
           symbol in expr_arg.base.free_symbols or
           isinstance(expr_arg, exp)):
            rv = True  # 指定变量在指数中
        else:
            return False  # 非指数方式依赖于指定变量
    return rv


# 解决可以简化为单个对数实例的对数方程的辅助函数
def _solve_logarithm(lhs, rhs, symbol, domain):
    # 对数方程是包含 `\log` 项的方程，可以使用各种对数恒等式简化为单个 `\log` 项或常数
    lhs, rhs : Expr
        The logarithmic equation to be solved, `lhs = rhs`
    symbol : Symbol
    """
    对给定的方程进行求解，处理带有多个对数的情况。

    .. math:: \log(x - 3) + \log(x + 3) = 0

    使用对数的恒等式将其转化为简化形式：

    使用恒等式：

    .. math:: \log(a) + \log(b) = \log(ab)

    方程变为：

    .. math:: \log((x - 3)(x + 3))

    这个方程只包含一个对数，可以通过重新写成指数形式来解决。

    """
    # 对左侧的对数表达式进行合并和简化
    new_lhs = logcombine(lhs, force=True)
    # 构造新的等式，左侧减去右侧
    new_f = new_lhs - rhs

    # 调用求解函数来解决这个新的等式，求解变量为 symbol，在指定的 domain 范围内
    return _solveset(new_f, symbol, domain)
# 判断方程是否处于对数形式 `a\log(f(x)) + b\log(g(x)) + ... + c`
# 如果是则返回 `True`，否则返回 `False`
def _is_logarithmic(f, symbol):
    rv = False  # 默认返回值为 False
    # 遍历方程的每个项
    for term in Add.make_args(f):
        saw_log = False  # 是否在当前项中发现了对数项的标志
        # 检查当前项中的每个因子
        for term_arg in Mul.make_args(term):
            if symbol not in term_arg.free_symbols:
                continue  # 如果当前因子不依赖于 symbol，则继续下一个因子的检查
            if isinstance(term_arg, log):
                if saw_log:
                    return False  # 如果在当前项中已经发现了多个对数函数，则返回 False
                saw_log = True  # 标记当前项中发现了一个对数函数
            else:
                return False  # 如果当前因子依赖于 symbol 但不是对数函数，则返回 False
        if saw_log:
            rv = True  # 如果当前项包含对数函数，则设置返回值为 True
    return rv  # 返回最终结果


# 如果函数返回 `False`，则 Lambert 求解器 (`_solve_lambert`) 不会被调用
def _is_lambert(f, symbol):
    # 快速检查 Lambert 求解器可能能够处理的情况
    # 1. 包含超过两个操作数和涉及任何 `Pow`、`exp`、`HyperbolicFunction`、`TrigonometricFunction`、`log` 项的方程
    # 2. 对于 `Pow`、`exp`，指数应包含 `symbol`，对于 `HyperbolicFunction`、`TrigonometricFunction`、`log`，应包含 `symbol`
    # 3. 对于 `HyperbolicFunction`、`TrigonometricFunction`，方程中的三角函数数量应少于符号数量
    #    （例如 `A*cos(x) + B*sin(x) - c` 不是 Lambert 类型）
    """
    Some forms of lambert equations are:
        1. X**X = C
        2. X*(B*log(X) + D)**A = C
        3. A*log(B*X + A) + d*X = C
        4. (B*X + A)*exp(d*X + g) = C
        5. g*exp(B*X + h) - B*X = C
        6. A*D**(E*X + g) - B*X = C
        7. A*cos(X) + B*sin(X) - D*X = C
        8. A*cosh(X) + B*sinh(X) - D*X = C

    Where X is any variable,
          A, B, C, D, E are any constants,
          g, h are linear functions or log terms.
    """
    # 返回 `False` 则表示 Lambert 求解器 (`_solve_lambert`) 不会被调用
    return True  # 默认情况下，允许 Lambert 求解器进行处理
    """
    检查给定的方程是否符合 Lambert 函数的特征，返回布尔值表示是否符合条件。

    Parameters
    ==========
    f : sympy.Expr
        给定的数学表达式，用来检查是否符合 Lambert 函数的特征。
    symbol : sympy.Symbol
        表示数学符号的符号变量，用于检查表达式中是否包含这些符号。

    Returns
    =======
    bool
        如果方程符合 Lambert 函数的特征，则返回 True；否则返回 False。

    See Also
    ========
    _solve_lambert
    """

    # 将表达式展开后得到的各项因子的列表
    term_factors = list(_term_factors(f.expand()))

    # 方程中包含的符号的总数
    no_of_symbols = len([arg for arg in term_factors if arg.has(symbol)])

    # 方程中包含的双曲函数和三角函数项的总数
    no_of_trig = len([arg for arg in term_factors \
        if arg.has(HyperbolicFunction, TrigonometricFunction)])

    if f.is_Add and no_of_symbols >= 2:
        # 如果是加法表达式且方程中至少有两个符号
        # `log`, `HyperbolicFunction`, `TrigonometricFunction` 应该含有符号
        # 并且三角函数的数量应小于符号的数量
        lambert_funcs = (log, HyperbolicFunction, TrigonometricFunction)
        if any(isinstance(arg, lambert_funcs) \
               for arg in term_factors if arg.has(symbol)):
            if no_of_trig < no_of_symbols:
                return True
        # 或者 `Pow`, `exp` 应该含有指数符号
        elif any(isinstance(arg, (Pow, exp)) \
                 for arg in term_factors if (arg.as_base_exp()[1]).has(symbol)):
            return True

    # 如果以上条件都不满足，则返回 False
    return False
# 定义一个函数来解决超越方程，这是 ``solveset`` 的一个辅助函数，应该在内部使用。
# ``_transolve`` 目前支持以下类型的方程：

def _transolve(f, symbol, domain):
    r"""
    # 解决超越方程的函数。它是 ``solveset`` 的一个辅助函数，应该在内部使用。
    # ``_transolve`` 目前支持以下类型的方程：
    
    - 指数方程
    - 对数方程

    Parameters
    ==========

    f : Any transcendental equation that needs to be solved.
        This needs to be an expression, which is assumed
        to be equal to ``0``.
    # 需要解决的任何超越方程，应该是一个表达式，假定等于 ``0``。

    symbol : The variable for which the equation is solved.
        This needs to be of class ``Symbol``.
    # 要解方程的变量。这应该是一个 ``Symbol`` 类。

    domain : A set over which the equation is solved.
        This needs to be of class ``Set``.
    # 方程解决的域。这应该是一个 ``Set`` 类。

    Returns
    =======

    Set
        A set of values for ``symbol`` for which ``f`` is equal to
        zero. An ``EmptySet`` is returned if ``f`` does not have solutions
        in respective domain. A ``ConditionSet`` is returned as unsolved
        object if algorithms to evaluate complete solution are not
        yet implemented.
    # 返回值：
    # - ``symbol`` 的值集合，使得 ``f`` 等于零。如果 ``f`` 在相应的域内没有解，则返回 ``EmptySet``。
    # - 如果尚未实现算法来评估完整解，则返回 ``ConditionSet`` 作为未解决的对象。

    How to use ``_transolve``
    =========================

    ``_transolve`` should not be used as an independent function, because
    it assumes that the equation (``f``) and the ``symbol`` comes from
    ``solveset`` and might have undergone a few modification(s).
    To use ``_transolve`` as an independent function the equation (``f``)
    and the ``symbol`` should be passed as they would have been by
    ``solveset``.
    # 如何使用 ``_transolve``：
    # ``_transolve`` 不应作为独立函数使用，因为它假设方程（``f``）和变量（``symbol``）来自于 ``solveset`` 并且可能已经经历了一些修改。
    # 要将 ``_transolve`` 作为独立函数使用，应该像它们在 ``solveset`` 中被传递一样传递方程（``f``）和变量（``symbol``）。

    Examples
    ========

    # 示例：
    
    >>> from sympy.solvers.solveset import _transolve as transolve
    >>> from sympy.solvers.solvers import _tsolve as tsolve
    >>> from sympy import symbols, S, pprint
    >>> x = symbols('x', real=True) # assumption added
    >>> transolve(5**(x - 3) - 3**(2*x + 1), x, S.Reals)
    {-(log(3) + 3*log(5))/(-log(5) + 2*log(3))}
    # 如何工作：
    
    How ``_transolve`` works
    ========================

    # 如何工作：

    ``_transolve`` uses two types of helper functions to solve equations
    of a particular class:

    # ``_transolve`` 使用两种类型的辅助函数来解决特定类别的方程：

    Identifying helpers: To determine whether a given equation
    belongs to a certain class of equation or not. Returns either
    ``True`` or ``False``.

    # 辨认助手：确定给定方程是否属于某个方程类别。返回 ``True`` 或 ``False``。

    Solving helpers: Once an equation is identified, a corresponding
    helper either solves the equation or returns a form of the equation
    that ``solveset`` might better be able to handle.

    # 解决助手：一旦识别出一个方程，相应的助手要么解决这个方程，要么返回一个 ``solveset`` 更容易处理的形式。

    * Philosophy behind the module

    # 模块背后的理念：

    The purpose of ``_transolve`` is to take equations which are not
    already polynomial in their generator(s) and to either recast them
    as such through a valid transformation or to solve them outright.
    A pair of helper functions for each class of supported
    transcendental functions are employed for this purpose. One
    identifies the transcendental form of an equation and the other
    either solves it or recasts it into a tractable form that can be
    solved by  ``solveset``.
    # ``_transolve`` 的目的是接受不是已经在其生成器中是多项式的方程，并通过有效的转换将其重塑为这样的形式或直接解决它们。
    # 为此目的使用了每个支持的超越函数类别的一对辅助函数。一个用于识别方程的超越形式，另一个要么解决它，要么将其重塑为 ``solveset`` 可以处理的可处理形式。

    For example, an equation in the form `ab^{f(x)} - cd^{g(x)} = 0`
    can be transformed to

    # 例如，形如 `ab^{f(x)} - cd^{g(x)} = 0` 的方程可以转换为
    `\log(a) + f(x)\log(b) - \log(c) - g(x)\log(d) = 0`
    # 这是一个数学方程，描述了一个对数方程的形式，其中包含函数 f(x) 和 g(x)。

    (under certain assumptions) and this can be solved with ``solveset``
    # 在特定假设下，可以使用 ``solveset`` 来解决这个方程。

    if `f(x)` and `g(x)` are in polynomial form.
    # 假设 f(x) 和 g(x) 是多项式形式。

    How ``_transolve`` is better than ``_tsolve``
    =============================================

    1) Better output
    # 1) 更好的输出

    ``_transolve`` provides expressions in a more simplified form.
    # ``_transolve`` 提供的表达式更简化。

    Consider a simple exponential equation
    # 考虑一个简单的指数方程

    >>> f = 3**(2*x) - 2**(x + 3)
    >>> pprint(transolve(f, x, S.Reals), use_unicode=False)
    # 使用 _transolve 解方程 f，求解实数域下的 x
        -3*log(2)
    {------------------}
     -2*log(3) + log(2)
    >>> pprint(tsolve(f, x), use_unicode=False)
         /   3     \
         | --------|
         | log(2/9)|
    [-log\2         /]
    # 使用 tsolve 解方程 f，给出的结果较为复杂

    2) Extensible
    # 2) 可扩展性

    The API of ``_transolve`` is designed such that it is easily
    # ``_transolve`` 的 API 被设计成易于扩展

    extensible, i.e. the code that solves a given class of
    # 即解决特定类别方程的代码

    equations is encapsulated in a helper and not mixed in with
    # 被封装在一个辅助函数中，而不是与主函数混合在一起

    the code of ``_transolve`` itself.

    3) Modular
    # 3) 模块化

    ``_transolve`` is designed to be modular i.e, for every class of
    # ``_transolve`` 被设计成模块化，即对于每一类方程

    equation a separate helper for identification and solving is
    # 都有独立的辅助函数用于识别和解决

    implemented. This makes it easy to change or modify any of the
    # 这使得可以轻松更改或修改任何

    method implemented directly in the helpers without interfering
    # 直接在辅助函数中实现的方法，而不影响

    with the actual structure of the API.

    4) Faster Computation
    # 4) 计算速度更快

    Solving equation via ``_transolve`` is much faster as compared to
    # 通过 ``_transolve`` 解方程比 ``_tsolve`` 快得多

    ``_tsolve``. In ``solve``, attempts are made computing every possibility
    # 在 ``solve`` 中，会尝试计算每一个可能的解

    to get the solutions. This series of attempts makes solving a bit
    # 这一系列尝试使得解方程变得稍慢

    slow. In ``_transolve``, computation begins only after a particular
    # 在 ``_transolve`` 中，计算只在特定类型的方程被识别后开始

    type of equation is identified.

    How to add new class of equations
    =================================

    Adding a new class of equation solver is a three-step procedure:
    # 添加一个新类别的方程解法是一个三步骤的过程：

    - Identify the type of the equations
    # - 确定方程的类型

      Determine the type of the class of equations to which they belong:
      # 确定它们属于哪种类型的方程类别：

      it could be of ``Add``, ``Pow``, etc. types. Separate internal functions
      # 可能是 ``Add``、``Pow`` 等类型。使用不同的内部函数

      are used for each type. Write identification and solving helpers
      # 为每一种类型编写识别和解决方案的辅助函数

      and use them from within the routine for the given type of equation
      # 并在给定类型的方程的主函数中使用它们

      (after adding it, if necessary). Something like:

      .. code-block:: python

        def add_type(lhs, rhs, x):
            ....
            if _is_exponential(lhs, x):
                new_eq = _solve_exponential(lhs, rhs, x)
        ....
        rhs, lhs = eq.as_independent(x)
        if lhs.is_Add:
            result = add_type(lhs, rhs, x)

    - Define the identification helper.
    # - 定义识别辅助函数

    - Define the solving helper.
    # - 定义解决辅助函数

    Apart from this, a few other things needs to be taken care while
    # 除此之外，添加方程解算器时还需注意一些其他事项

    adding an equation solver:

    - Naming conventions:
    # - 命名约定：

      Name of the identification helper should be as
      # 识别辅助函数的命名应该是

      ``_is_class`` where class will be the name or abbreviation
      # ``_is_class``，其中 class 是方程类别的名称或缩写

      of the class of equation. The solving helper will be named as
      # 解决辅助函数应该命名为

      ``_solve_class``.
      # ``_solve_class``。

      For example: for exponential equations it becomes
      # 例如：指数方程的命名应该是

      ``_is_exponential`` and ``_solve_expo``.
      # ``_is_exponential`` 和 ``_solve_expo``。
    def add_type(lhs, rhs, symbol, domain):
        """
        Helper for ``_transolve`` to handle equations of
        ``Add`` type, i.e. equations taking the form as
        ``a*f(x) + b*g(x) + .... = c``.
        For example: 4**x + 8**x = 0
        """
        # 创建一个条件集合，表示方程 lhs - rhs = 0 的解集
        result = ConditionSet(symbol, Eq(lhs - rhs, 0), domain)

        # 检查是否为指数类型的方程
        if _is_exponential(lhs, symbol):
            # 如果是指数类型方程，则调用指数求解函数
            result = _solve_exponential(lhs, rhs, symbol, domain)
        # 检查是否为对数类型的方程
        elif _is_logarithmic(lhs, symbol):
            # 如果是对数类型方程，则调用对数求解函数
            result = _solve_logarithm(lhs, rhs, symbol, domain)

        return result

    # 创建一个条件集合，表示方程 f = 0 的解集
    result = ConditionSet(symbol, Eq(f, 0), domain)

    # 调用 invert_complex 处理复杂的求解问题，根据指定的领域处理不同情况
    lhs, rhs_s = invert_complex(f, 0, symbol, domain)

    # 如果 rhs_s 是一个有限集
    if isinstance(rhs_s, FiniteSet):
        # 断言这个集合只有一个元素
        assert (len(rhs_s.args)) == 1
        rhs = rhs_s.args[0]

        # 如果 lhs 是加法类型的表达式
        if lhs.is_Add:
            # 调用 add_type 处理加法类型的方程
            result = add_type(lhs, rhs, symbol, domain)
    else:
        # 否则，直接将 rhs_s 赋给 result
        result = rhs_s

    return result
# 定义解方程或不等式的函数，返回解集合

def solveset(f, symbol=None, domain=S.Complexes):
    r"""Solves a given inequality or equation with set as output

    Parameters
    ==========

    f : Expr or a relational.
        The target equation or inequality  目标方程或不等式
    symbol : Symbol
        The variable for which the equation is solved  要解的变量
    domain : Set
        The domain over which the equation is solved  解方程的定义域

    Returns
    =======

    Set
        A set of values for `symbol` for which `f` is True or is equal to
        zero. An :class:`~.EmptySet` is returned if `f` is False or nonzero.
        A :class:`~.ConditionSet` is returned as unsolved object if algorithms
        to evaluate complete solution are not yet implemented.  返回`symbol`的值集合，
        其中`f`为真或等于零。如果`f`为假或非零，则返回`EmptySet`。
        如果算法尚未实现完整的解法，则返回`ConditionSet`作为未解决的对象。

    ``solveset`` claims to be complete in the solution set that it returns.  ``solveset``声称返回的解集是完整的。

    Raises
    ======

    NotImplementedError
        The algorithms to solve inequalities in complex domain  are
        not yet implemented.  尚未实现在复数域内解不等式的算法。
    ValueError
        The input is not valid.  输入无效。
    RuntimeError
        It is a bug, please report to the github issue tracker.  这是一个错误，请报告给GitHub问题跟踪器。

    Notes
    =====

    Python interprets 0 and 1 as False and True, respectively, but
    in this function they refer to solutions of an expression. So 0 and 1
    return the domain and EmptySet, respectively, while True and False
    return the opposite (as they are assumed to be solutions of relational
    expressions).  Python分别将0和1解释为False和True，但在这个函数中它们指的是表达式的解。
    因此，0和1分别返回域和EmptySet，而True和False返回相反的结果（因为它们被假设为关系表达式的解）。

    See Also
    ========

    solveset_real: solver for real domain  实数域求解器
    solveset_complex: solver for complex domain  复数域求解器

    Examples
    ========

    >>> from sympy import exp, sin, Symbol, pprint, S, Eq
    >>> from sympy.solvers.solveset import solveset, solveset_real

    * The default domain is complex. Not specifying a domain will lead
      to the solving of the equation in the complex domain (and this
      is not affected by the assumptions on the symbol):  默认域是复数域。不指定域将导致在复数域中求解方程（这不受对符号的假设影响）：

    >>> x = Symbol('x')
    >>> pprint(solveset(exp(x) - 1, x), use_unicode=False)
    {2*n*I*pi | n in Integers}

    >>> x = Symbol('x', real=True)
    >>> pprint(solveset(exp(x) - 1, x), use_unicode=False)
    {2*n*I*pi | n in Integers}

    * If you want to use ``solveset`` to solve the equation in the
      real domain, provide a real domain. (Using ``solveset_real``
      does this automatically.)  如果要在实数域中使用``solveset``求解方程，
      提供一个实数域。（使用``solveset_real``会自动完成此操作。）

    >>> R = S.Reals
    >>> x = Symbol('x')
    >>> solveset(exp(x) - 1, x, R)
    {0}
    >>> solveset_real(exp(x) - 1, x)
    {0}

    The solution is unaffected by assumptions on the symbol:  对符号的假设不影响解的结果：

    >>> p = Symbol('p', positive=True)
    >>> pprint(solveset(p**2 - 4))
    {-2, 2}

    When a :class:`~.ConditionSet` is returned, symbols with assumptions that
    would alter the set are replaced with more generic symbols:  返回``ConditionSet``时，
    具有可能改变集合的假设的符号被替换为更通用的符号：

    >>> i = Symbol('i', imaginary=True)
    >>> solveset(Eq(i**2 + i*sin(i), 1), i, domain=S.Reals)
    ConditionSet(_R, Eq(_R**2 + _R*sin(_R) - 1, 0), Reals)

    * Inequalities can be solved over the real domain only. Use of a complex
      domain leads to a NotImplementedError.  只能在实数域中解不等式。使用复数域会导致``NotImplementedError``。

    >>> solveset(exp(x) > 1, x, R)
    # 创建一个开放的区间，从0到正无穷大
    Interval.open(0, oo)
    
    """
    # 将输入的函数f和符号symbol转换为Sympy表达式
    f = sympify(f)
    symbol = sympify(symbol)
    
    # 如果f为真，则返回给定的定义域
    if f is S.true:
        return domain
    
    # 如果f为假，则返回空集
    if f is S.false:
        return S.EmptySet
    
    # 如果f不是Sympy表达式、关系表达式或数值，则引发值错误异常
    if not isinstance(f, (Expr, Relational, Number)):
        raise ValueError("%s is not a valid SymPy expression" % f)
    
    # 如果symbol不是Sympy表达式或关系表达式，并且不为None，则引发值错误异常
    if not isinstance(symbol, (Expr, Relational)) and symbol is not None:
        raise ValueError("%s is not a valid SymPy symbol" % (symbol,))
    
    # 如果domain不是有效的集合类型，则引发值错误异常
    if not isinstance(domain, Set):
        raise ValueError("%s is not a valid domain" %(domain))
    
    # 获取f中的自由符号集合
    free_symbols = f.free_symbols
    
    # 如果f包含Piecewise函数，则将其进行展开处理
    if f.has(Piecewise):
        f = piecewise_fold(f)
    
    # 如果symbol为None且没有自由符号，则尝试判断f与0的关系
    if symbol is None and not free_symbols:
        b = Eq(f, 0)
        if b is S.true:
            return domain
        elif b is S.false:
            return S.EmptySet
        else:
            raise NotImplementedError(filldedent('''
                relationship between value and 0 is unknown: %s''' % b))
    
    # 如果symbol为None且只有一个自由符号，则将其指定为symbol；如果有多个自由符号，则引发值错误异常
    if symbol is None:
        if len(free_symbols) == 1:
            symbol = free_symbols.pop()
        elif free_symbols:
            raise ValueError(filldedent('''
                The independent variable must be specified for a
                multivariate equation.'''))
    # 如果symbol不是Symbol类型，则将f和symbol重新转换为符号，然后调用solveset求解
    elif not isinstance(symbol, Symbol):
        f, s, swap = recast_to_symbols([f], [symbol])
        # 如果解集是ConditionSet类型，则需要使用xreplace
        return solveset(f[0], s[0], domain).xreplace(swap)
    
    # 对于在实数域内的symbol，检查其原始假设是否为实数；在复数域内的symbol，检查其原始假设是否为复数
    newsym = None
    if domain.is_subset(S.Reals):
        if symbol._assumptions_orig != {'real': True}:
            newsym = Dummy('R', real=True)
    elif domain.is_subset(S.Complexes):
        if symbol._assumptions_orig != {'complex': True}:
            newsym = Dummy('C', complex=True)
    
    # 如果newsym不为None，则使用newsym替换symbol求解，并尝试用原始symbol替换newsym以获得最终解
    if newsym is not None:
        rv = solveset(f.xreplace({symbol: newsym}), newsym, domain)
        try:
            _rv = rv.xreplace({newsym: symbol})
        except TypeError:
            _rv = rv
        if rv.dummy_eq(_rv):
            rv = _rv
        return rv
    
    # 对于包含Abs函数的情况，通过_absorbing函数处理Abs函数的分段表达式，确保结果的准确性
    f, mask = _masked(f, Abs)
    f = f.rewrite(Piecewise)  # 将不是Abs函数的部分转换为Piecewise形式
    for d, e in mask:
        # 将Abs函数中的部分也转换为Piecewise形式
        e = e.func(e.args[0].rewrite(Piecewise))
        f = f.xreplace({d: e})
    f = piecewise_fold(f)  # 将f进行Piecewise函数的折叠处理，确保结果的简洁性和正确性
    
    # 调用_solveset函数求解最终结果并返回
    return _solveset(f, symbol, domain, _check=True)
# 解决实数域下的方程，使用 sympy 的 solveset 函数
def solveset_real(f, symbol):
    return solveset(f, symbol, S.Reals)


# 解决复数域下的方程，使用 sympy 的 solveset 函数
def solveset_complex(f, symbol):
    return solveset(f, symbol, S.Complexes)


# 多变量求解器的基本实现
# 用于内部使用（未准备对外公开）
def _solveset_multi(eqs, syms, domains):
    rep = {}
    # 将实数域的变量替换为具有 real=True 的符号
    for sym, dom in zip(syms, domains):
        if dom is S.Reals:
            rep[sym] = Symbol(sym.name, real=True)
    # 替换方程中的符号为新定义的实数域符号
    eqs = [eq.subs(rep) for eq in eqs]
    syms = [sym.subs(rep) for sym in syms]

    syms = tuple(syms)

    if len(eqs) == 0:
        return ProductSet(*domains)

    if len(syms) == 1:
        sym = syms[0]
        domain = domains[0]
        # 对每个方程求解，得到解集合的交集
        solsets = [solveset(eq, sym, domain) for eq in eqs]
        solset = Intersection(*solsets)
        # 创建映射集合，按要求处理
        return ImageSet(Lambda((sym,), (sym,)), solset).doit()

    # 根据自由符号数量对方程进行排序
    eqs = sorted(eqs, key=lambda eq: len(eq.free_symbols & set(syms)))

    for n, eq in enumerate(eqs):
        sols = []
        all_handled = True
        for sym in syms:
            if sym not in eq.free_symbols:
                continue
            # 解决每个符号对应的方程
            sol = solveset(eq, sym, domains[syms.index(sym)])

            if isinstance(sol, FiniteSet):
                i = syms.index(sym)
                symsp = syms[:i] + syms[i+1:]
                domainsp = domains[:i] + domains[i+1:]
                eqsp = eqs[:n] + eqs[n+1:]
                # 对于每个解，继续递归求解其余方程
                for s in sol:
                    eqsp_sub = [eq.subs(sym, s) for eq in eqsp]
                    sol_others = _solveset_multi(eqsp_sub, symsp, domainsp)
                    fun = Lambda((symsp,), symsp[:i] + (s,) + symsp[i:])
                    sols.append(ImageSet(fun, sol_others).doit())
            else:
                all_handled = False
        if all_handled:
            return Union(*sols)


# 使用 solveset 函数解方程，并根据解集的类型返回相应的输出
def solvify(f, symbol, domain):
    """Solves an equation using solveset and returns the solution in accordance
    with the `solve` output API.

    Returns
    =======

    We classify the output based on the type of solution returned by `solveset`.

    Solution    |    Output
    ----------------------------------------
    FiniteSet   | list

    ImageSet,   | list (if `f` is periodic)
    Union       |

    Union       | list (with FiniteSet)

    EmptySet    | empty list

    Others      | None


    Raises
    ======

    NotImplementedError
        A ConditionSet is the input.

    Examples
    ========

    >>> from sympy.solvers.solveset import solvify
    >>> from sympy.abc import x
    >>> from sympy import S, tan, sin, exp
    >>> solvify(x**2 - 9, x, S.Reals)
    [-3, 3]
    >>> solvify(sin(x) - 1, x, S.Reals)
    [pi/2]
    >>> solvify(tan(x), x, S.Reals)
    [0]
    >>> solvify(exp(x) - 1, x, S.Complexes)

    >>> solvify(exp(x) - 1, x, S.Reals)
    [0]

    """
    # 使用 solveset 函数解方程
    solution_set = solveset(f, symbol, domain)
    result = None
    # 根据解集的类型进行分类和处理
    if solution_set is S.EmptySet:
        result = []
    elif isinstance(solution_set, ConditionSet):
        # 如果 solution_set 是 ConditionSet 类型，则抛出未实现错误，无法解决这个方程。
        raise NotImplementedError('solveset is unable to solve this equation.')

    elif isinstance(solution_set, FiniteSet):
        # 如果 solution_set 是 FiniteSet 类型，则将其转换为列表形式。
        result = list(solution_set)

    else:
        # 计算函数 f 关于 symbol 的周期性
        period = periodicity(f, symbol)
        # 如果存在周期性
        if period is not None:
            # 初始化一个空集合 solutions
            solutions = S.EmptySet
            # 初始化一个空元组 iter_solutions
            iter_solutions = ()
            # 如果 solution_set 是 ImageSet 类型，则将其作为唯一的元素放入 iter_solutions 元组中
            if isinstance(solution_set, ImageSet):
                iter_solutions = (solution_set,)
            # 如果 solution_set 是 Union 类型，并且所有的子项都是 ImageSet 类型，则将其作为 iter_solutions 元组
            elif isinstance(solution_set, Union):
                if all(isinstance(i, ImageSet) for i in solution_set.args):
                    iter_solutions = solution_set.args

            # 遍历 iter_solutions 中的每个 solution，并将其与区间 [0, period) 的交集添加到 solutions 中
            for solution in iter_solutions:
                solutions += solution.intersect(Interval(0, period, False, True))

            # 如果 solutions 是 FiniteSet 类型，则将其转换为列表形式
            if isinstance(solutions, FiniteSet):
                result = list(solutions)

        else:
            # 将 solution_set 与 domain 的交集赋值给 solution
            solution = solution_set.intersect(domain)
            # 如果 solution 是 Union 类型
            if isinstance(solution, Union):
                # 如果其中至少有一个子项是 FiniteSet 类型，则将所有 FiniteSet 类型的子项扁平化为列表形式
                if any(isinstance(i, FiniteSet) for i in solution.args):
                    result = [sol for soln in solution.args \
                     for sol in soln.args if isinstance(soln,FiniteSet)]
                else:
                    # 否则返回 None
                    return None

            # 如果 solution 是 FiniteSet 类型，则将其直接添加到 result 列表中
            elif isinstance(solution, FiniteSet):
                result += solution

    # 返回最终的结果列表 result
    return result
# 定义函数 linear_coeffs，用于提取方程中与给定符号对应的系数和常数项
def linear_coeffs(eq, *syms, dict=False):
    """Return a list whose elements are the coefficients of the
    corresponding symbols in the sum of terms in  ``eq``.
    The additive constant is returned as the last element of the
    list.

    Raises
    ======

    NonlinearError
        The equation contains a nonlinear term
    ValueError
        duplicate or unordered symbols are passed

    Parameters
    ==========

    dict - (default False) when True, return coefficients as a
        dictionary with coefficients keyed to syms that were present;
        key 1 gives the constant term

    Examples
    ========

    >>> from sympy.solvers.solveset import linear_coeffs
    >>> from sympy.abc import x, y, z
    >>> linear_coeffs(3*x + 2*y - 1, x, y)
    [3, 2, -1]

    It is not necessary to expand the expression:

        >>> linear_coeffs(x + y*(z*(x*3 + 2) + 3), x)
        [3*y*z + 1, y*(2*z + 3)]

    When nonlinear is detected, an error will be raised:

        * even if they would cancel after expansion (so the
        situation does not pass silently past the caller's
        attention)

        >>> eq = 1/x*(x - 1) + 1/x
        >>> linear_coeffs(eq.expand(), x)
        [0, 1]
        >>> linear_coeffs(eq, x)
        Traceback (most recent call last):
        ...
        NonlinearError:
        nonlinear in given generators

        * when there are cross terms

        >>> linear_coeffs(x*(y + 1), x, y)
        Traceback (most recent call last):
        ...
        NonlinearError:
        symbol-dependent cross-terms encountered

        * when there are terms that contain an expression
        dependent on the symbols that is not linear

        >>> linear_coeffs(x**2, x)
        Traceback (most recent call last):
        ...
        NonlinearError:
        nonlinear in given generators
    """
    # 将方程符号化处理
    eq = _sympify(eq)
    # 检查符号数目及其唯一性
    if len(syms) == 1 and iterable(syms[0]) and not isinstance(syms[0], Basic):
        raise ValueError('expecting unpacked symbols, *syms')
    symset = set(syms)
    if len(symset) != len(syms):
        raise ValueError('duplicate symbols given')
    try:
        # 转换为线性方程组的字典形式
        d, c = _linear_eq_to_dict([eq], symset)
        d = d[0]
        c = c[0]
    except PolyNonlinearError as err:
        # 如果出现非线性项，抛出 NonlinearError 异常
        raise NonlinearError(str(err))
    # 如果需要返回字典形式的系数
    if dict:
        if c:
            d[S.One] = c
        return d
    # 否则返回列表形式，最后一个元素是常数项
    rv = [S.Zero]*(len(syms) + 1)
    rv[-1] = c
    for i, k in enumerate(syms):
        if k not in d:
            continue
        rv[i] = d[k]
    return rv


# 定义函数 linear_eq_to_matrix，将给定的方程组转换为矩阵形式
def linear_eq_to_matrix(equations, *symbols):
    r"""
    Converts a given System of Equations into Matrix form.
    Here `equations` must be a linear system of equations in
    `symbols`. Element ``M[i, j]`` corresponds to the coefficient
    # 检查符号集合是否为空，如果为空则抛出值错误异常
    if not symbols:
        raise ValueError(filldedent('''
            Symbols must be given, for which coefficients
            are to be found.
            '''))

    # 检查第一个符号是否为集合，如果是则抛出类型错误异常
    if isinstance(symbols[0], set):
        raise TypeError(
            "Unordered 'set' type is not supported as input for symbols.")

    # 如果symbols的第一个元素是可迭代对象，则将symbols重新赋值为它的第一个元素
    if hasattr(symbols[0], '__iter__'):
        symbols = symbols[0]

    # 检查symbols中是否存在重复的符号，如果有则抛出值错误异常
    if has_dups(symbols):
        raise ValueError('Symbols must be unique')

    # 将equations转换为Sympy表达式的列表形式
    equations = sympify(equations)

    # 如果equations是MatrixBase类型，则转换为列表
    if isinstance(equations, MatrixBase):
        equations = list(equations)
    # 如果equations是Expr或Eq类型，则转换为包含该单个表达式的列表
    elif isinstance(equations, (Expr, Eq)):
        equations = [equations]
    # 如果equations不是序列类型，则抛出值错误异常
    elif not is_sequence(equations):
        raise ValueError(filldedent('''
            Equation(s) must be given as a sequence, Expr,
            Eq or Matrix.
            '''))

    # 尝试构建方程的字典形式
    try:
        # 调用函数_linear_eq_to_dict将方程列表和符号列表转换为字典形式
        eq, c = _linear_eq_to_dict(equations, symbols)
    # 处理多项式非线性错误，将其转换为非线性错误并抛出
    except PolyNonlinearError as err:
        raise NonlinearError(str(err))
    
    # 准备输出矩阵
    # 获取方程数和未知数数目
    n, m = shape = len(eq), len(symbols)
    
    # 创建未知数到索引的映射字典
    ix = dict(zip(symbols, range(m)))
    
    # 初始化一个 n x m 的零矩阵 A
    A = zeros(*shape)
    
    # 遍历方程列表 eq
    for row, d in enumerate(eq):
        # 遍历方程中的每个未知数及其系数
        for k in d:
            # 获取未知数在 ix 中的列索引
            col = ix[k]
            # 将方程 d[k] 的系数存入矩阵 A 的相应位置
            A[row, col] = d[k]
    
    # 创建一个 n x 1 的列向量 b，其元素为列表 c 中各元素的负值
    b = Matrix(n, 1, [-i for i in c])
    
    # 返回矩阵 A 和向量 b 作为结果
    return A, b
# 定义一个函数 linsolve，用于解决 $N$ 个线性方程组成的系统，含有 $M$ 个变量；支持欠定和超定系统。
def linsolve(system, *symbols):
    r"""
    解决 $N$ 个线性方程组成的系统，含有 $M$ 个变量；支持欠定和超定系统。
    可能的解数量为零、一或无穷多。
    零解会引发 ValueError，而无穷多解会使用给定的符号来表示参数化。
    对于唯一解，返回一个 :class:`~.FiniteSet` 包含有序元组的集合。

    所有标准输入格式都被支持：
    对于给定的方程集，下面是相应的输入类型：

    .. math:: 3x + 2y -   z = 1
    .. math:: 2x - 2y + 4z = -2
    .. math:: 2x -   y + 2z = 0

    * 增广矩阵形式，`system` 如下所示:

    $$ \text{system} = \left[\begin{array}{cccc}
        3 &  2 & -1 &  1\\
        2 & -2 &  4 & -2\\
        2 & -1 &  2 &  0
        \end{array}\right] $$

    ::

        system = Matrix([[3, 2, -1, 1], [2, -2, 4, -2], [2, -1, 2, 0]])

    * 方程列表形式

    ::

        system  =  [3x + 2y - z - 1, 2x - 2y + 4z + 2, 2x - y + 2z]

    * 矩阵形式的输入 $A$ 和 $b$ （来自 $Ax = b$）如下给出:

    $$ A = \left[\begin{array}{ccc}
        3 &  2 & -1 \\
        2 & -2 &  4 \\
        2 & -1 &  2
        \end{array}\right] \ \  b = \left[\begin{array}{c}
        1 \\ -2 \\ 0
        \end{array}\right] $$

    ::

        A = Matrix([[3, 2, -1], [2, -2, 4], [2, -1, 2]])
        b = Matrix([[1], [-2], [0]])
        system = (A, b)

    符号可以始终传递，但实际上只在以下情况下需要：
    1) 传递一个方程组，并且
    2) 以欠定矩阵的形式传递系统，且希望控制结果中自由变量的名称。
    如果未用于情况1，则会引发错误；但是如果未提供用于情况2的符号，则会内部生成符号。
    对于情况2提供符号时，应至少有与矩阵A列数相同数量的符号。

    这里使用的算法是高斯-约当消元法，消元后得到行阶梯形矩阵。

    返回
    =======

    包含未知数的有序元组值的 :class:`~.FiniteSet` 集合，该集合是系统 `system` 的解。
    （将元组包装在 FiniteSet 中用于在整个 solveset 过程中保持一致的输出格式。）

    如果线性系统不一致，返回 EmptySet。

    引发
    ======

    ValueError
        输入无效。
        未提供符号。

    示例
    ========

    >>> from sympy import Matrix, linsolve, symbols
    >>> x, y, z = symbols("x, y, z")
    >>> A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    >>> b = Matrix([3, 6, 9])
    >>> A
    Matrix([
    [1, 2,  3],
    [4, 5,  6],
    [7, 8, 10]])
    >>> b
    Matrix([
    [3],
    [6],
    [9]])
    >>> linsolve((A, b), [x, y, z])
    {(-1, 2, 0)}
    """
    * Parametric Solution: In case the system is underdetermined, the
      function will return a parametric solution in terms of the given
      symbols. Those that are free will be returned unchanged. e.g. in
      the system below, `z` is returned as the solution for variable z;
      it can take on any value.

    >>> A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> b = Matrix([3, 6, 9])
    >>> linsolve((A, b), x, y, z)
    {(z - 1, 2 - 2*z, z)}

    If no symbols are given, internally generated symbols will be used.
    The ``tau0`` in the third position indicates (as before) that the third
    variable -- whatever it is named -- can take on any value:

    >>> linsolve((A, b))
    {(tau0 - 1, 2 - 2*tau0, tau0)}

    * List of equations as input

    >>> Eqns = [3*x + 2*y - z - 1, 2*x - 2*y + 4*z + 2, - x + y/2 - z]
    >>> linsolve(Eqns, x, y, z)
    {(1, -2, -2)}

    * Augmented matrix as input

    >>> aug = Matrix([[2, 1, 3, 1], [2, 6, 8, 3], [6, 8, 18, 5]])
    >>> aug
    Matrix([
    [2, 1,  3, 1],
    [2, 6,  8, 3],
    [6, 8, 18, 5]])
    >>> linsolve(aug, x, y, z)
    {(3/10, 2/5, 0)}

    * Solve for symbolic coefficients

    >>> a, b, c, d, e, f = symbols('a, b, c, d, e, f')
    >>> eqns = [a*x + b*y - c, d*x + e*y - f]
    >>> linsolve(eqns, x, y)
    {((-b*f + c*e)/(a*e - b*d), (a*f - c*d)/(a*e - b*d))}

    * A degenerate system returns solution as set of given
      symbols.

    >>> system = Matrix(([0, 0, 0], [0, 0, 0], [0, 0, 0]))
    >>> linsolve(system, x, y)
    {(x, y)}

    * For an empty system linsolve returns empty set

    >>> linsolve([], x)
    EmptySet

    * An error is raised if any nonlinearity is detected, even
      if it could be removed with expansion

    >>> linsolve([x*(1/x - 1)], x)
    Traceback (most recent call last):
    ...
    NonlinearError: nonlinear term: 1/x

    >>> linsolve([x*(y + 1)], x, y)
    Traceback (most recent call last):
    ...
    NonlinearError: nonlinear cross-term: x*(y + 1)

    >>> linsolve([x**2 - 1], x)
    Traceback (most recent call last):
    ...
    NonlinearError: nonlinear term: x**2
    """
    if not system:
        return S.EmptySet

    # If second argument is an iterable
    if symbols and hasattr(symbols[0], '__iter__'):
        symbols = symbols[0]
    sym_gen = isinstance(symbols, GeneratorType)
    dup_msg = 'duplicate symbols given'

    b = None  # if we don't get b the input was bad
    # unpack system
    # 检查 `system` 是否可迭代
    if hasattr(system, '__iter__'):
    
        # 情况 1：(A, b)
        if len(system) == 2 and isinstance(system[0], MatrixBase):
            # 如果 `system` 是长度为 2 的元组，并且第一个元素是 MatrixBase 类型
            A, b = system  # 将第一个元素赋给 A，第二个元素赋给 b

        # 情况 2：(eq1, eq2, ...)
        if not isinstance(system[0], MatrixBase):
            # 如果第一个元素不是 MatrixBase 类型
            if sym_gen or not symbols:
                # 如果 sym_gen 为真值或者 symbols 为空
                raise ValueError(filldedent('''
                    When passing a system of equations, the explicit
                    symbols for which a solution is being sought must
                    be given as a sequence, too.
                '''))
            if len(set(symbols)) != len(symbols):
                # 如果 symbols 中有重复的符号
                raise ValueError(dup_msg)

            # 转换方程式为 sympy 对象列表
            eqs = system
            eqs = [sympify(eq) for eq in eqs]
            try:
                # 使用 _linsolve 函数求解线性方程组
                sol = _linsolve(eqs, symbols)
            except PolyNonlinearError as exc:
                # 抛出 NonlinearError 异常，例如 cos(x) 包含了生成器集合的元素
                raise NonlinearError(str(exc))

            if sol is None:
                return S.EmptySet

            # 将解转换为有限集合
            sol = FiniteSet(Tuple(*(sol.get(sym, sym) for sym in symbols)))
            return sol

    elif isinstance(system, MatrixBase) and not (
            symbols and not isinstance(symbols, GeneratorType) and
            isinstance(symbols[0], MatrixBase)):
        # 情况 3：A 增广矩阵 b
        A, b = system[:, :-1], system[:, -1:]

    if b is None:
        # 如果 b 为空，抛出 ValueError 异常
        raise ValueError("Invalid arguments")
    if sym_gen:
        # 如果 sym_gen 为真值
        symbols = [next(symbols) for i in range(A.cols)]
        symset = set(symbols)
        if any(symset & (A.free_symbols | b.free_symbols)):
            # 如果符号集合与 A 和 b 的自由符号集合有重合
            raise ValueError(filldedent('''
                At least one of the symbols provided
                already appears in the system to be solved.
                One way to avoid this is to use Dummy symbols in
                the generator, e.g. numbered_symbols('%s', cls=Dummy)
            ''' % symbols[0].name.rstrip('1234567890')))
        elif len(symset) != len(symbols):
            # 如果符号集合中的符号数量与 symbols 中的数量不匹配
            raise ValueError(dup_msg)

    if not symbols:
        # 如果 symbols 为空
        symbols = [Dummy() for _ in range(A.cols)]
        # 生成一个唯一命名的符号名称
        name = _uniquely_named_symbol('tau', (A, b),
            compare=lambda i: str(i).rstrip('1234567890')).name
        gen  = numbered_symbols(name)
    else:
        gen = None

    # 这仅仅是 solve_lin_sys 函数的一个包装
    eqs = []
    rows = A.tolist()
    for rowi, bi in zip(rows, b):
        # 对于每一行 A[i], b[i]，构建一个方程式
        terms = [elem * sym for elem, sym in zip(rowi, symbols) if elem]
        terms.append(-bi)
        eqs.append(Add(*terms))

    # 将方程式转换为 sympy 的表达式环
    eqs, ring = sympy_eqs_to_ring(eqs, symbols)
    # 使用 solve_lin_sys 函数求解线性系统
    sol = solve_lin_sys(eqs, ring, _raw=False)
    if sol is None:
        return S.EmptySet
    #sol = {sym:val for sym, val in sol.items() if sym != val}
    # 使用给定的符号集合构建一个元组，并将它转换为有限集（FiniteSet）
    sol = FiniteSet(Tuple(*(sol.get(sym, sym) for sym in symbols)))

    # 如果提供了生成器对象 gen
    if gen is not None:
        # 获取解集 sol 中的自由符号集合
        solsym = sol.free_symbols
        # 创建一个字典 rep，将符号集合中的符号映射到生成器 gen 生成的值
        rep = {sym: next(gen) for sym in symbols if sym in solsym}
        # 使用 rep 字典替换解集 sol 中的符号，得到替换后的解集
        sol = sol.subs(rep)

    # 返回最终的解集 sol
    return sol
##############################################################################
# ------------------------------nonlinsolve ---------------------------------#
##############################################################################


def _return_conditionset(eqs, symbols):
    # 将方程列表转换为等式表达式 Eq(lhs, 0)，生成一个生成器对象
    eqs = (Eq(lhs, 0) for lhs in eqs)
    # 创建条件集合 ConditionSet 对象，表示方程组的解集合
    condition_set = ConditionSet(
        Tuple(*symbols), And(*eqs), S.Complexes**len(symbols))
    return condition_set


def substitution(system, symbols, result=[{}], known_symbols=[],
                 exclude=[], all_symbols=None):
    r"""
    使用替换方法解决 `system` 中的方程组。在 :func:`~.nonlinsolve` 中调用。
    当方程是非多项式方程时使用。

    Parameters
    ==========

    system : list of equations
        目标方程组
    symbols : list of symbols to be solved.
        要解的变量列表
    known_symbols : list of solved symbols
        已知变量的值
    result : An empty list or list of dict
        如果没有已知符号值，则为空列表，否则为符号作为键，对应值的字典。
    exclude : Set of expression.
        方程组的分母表达式，最终解不应满足这些表达式。
    all_symbols : known_symbols + symbols(unsolved).

    Returns
    =======

    一个有序元组的有限集，包含 `all_symbols` 的值，使得 `system` 有解。
    元组中的值顺序与参数 `all_symbols` 中的符号顺序相同。如果 `all_symbols` 为 None，
    则与参数 `symbols` 中的符号顺序相同。

    请注意，一般的有限集是无序的，这里返回的解决方案不仅仅是一个解的有限集，而是一个
    有序元组的有限集，即 FiniteSet 的第一个 & 唯一参数是一个有序元组的解决方案，它是有序的，
    因此返回的解决方案是有序的。

    另请注意，解决方案也可以作为有序元组返回，FiniteSet 只是一个包装器 `{}`，用于在整个
    solveset 过程中保持一致的输出格式。

    Raises
    ======

    ValueError
        输入无效。
        未给出符号。
    AttributeError
        输入的符号不是 :class:`~.Symbol` 类型。

    Examples
    ========

    >>> from sympy import symbols, substitution
    >>> x, y = symbols('x, y', real=True)
    >>> substitution([x + y], [x], [{y: 1}], [y], set([]), [x, y])
    {(-1, 1)}

    * 当你想要一个不满足 $x + 1 = 0$ 的解

    >>> substitution([x + y], [x], [{y: 1}], [y], set([x + 1]), [y, x])
    EmptySet
    >>> substitution([x + y], [x], [{y: 1}], [y], set([x - 1]), [y, x])
    {(1, -1)}
    # 对给定的符号替换系统进行求解，并返回解集合
    >>> substitution([x + y - 1, y - x**2 + 5], [x, y])
    {(-3, 4), (2, -1)}

    # 返回实数和复数解

    >>> x, y, z = symbols('x, y, z')
    >>> from sympy import exp, sin
    >>> substitution([exp(x) - sin(y), y**2 - 4], [x, y])
    {(ImageSet(Lambda(_n, I*(2*_n*pi + pi) + log(sin(2))), Integers), -2),
     (ImageSet(Lambda(_n, 2*_n*I*pi + log(sin(2))), Integers), 2)}

    # 返回包含复数解的解集合

    >>> eqs = [z**2 + exp(2*x) - sin(y), -3 + exp(-y)]
    >>> substitution(eqs, [y, z])
    {(-log(3), -sqrt(-exp(2*x) - sin(log(3)))),
     (-log(3), sqrt(-exp(2*x) - sin(log(3)))),
     (ImageSet(Lambda(_n, 2*_n*I*pi - log(3)), Integers),
      ImageSet(Lambda(_n, -sqrt(-exp(2*x) + sin(2*_n*I*pi - log(3)))), Integers)),
     (ImageSet(Lambda(_n, 2*_n*I*pi - log(3)), Integers),
      ImageSet(Lambda(_n, sqrt(-exp(2*x) + sin(2*_n*I*pi - log(3)))), Integers))}

    """

    # 如果未提供系统，则返回空集
    if not system:
        return S.EmptySet

    # 将每个方程转化为等价的形式：左侧表达式减去右侧表达式
    for i, e in enumerate(system):
        if isinstance(e, Eq):
            system[i] = e.lhs - e.rhs

    # 如果未提供符号列表，则引发值错误异常
    if not symbols:
        msg = ('Symbols must be given, for which solution of the '
               'system is to be found.')
        raise ValueError(filldedent(msg))

    # 如果符号不是序列类型，则引发类型错误异常
    if not is_sequence(symbols):
        msg = ('symbols should be given as a sequence, e.g. a list.'
               'Not type %s: %s')
        raise TypeError(filldedent(msg % (type(symbols), symbols)))

    # 如果符号列表的第一个元素不是符号类型，则引发值错误异常
    if not getattr(symbols[0], 'is_Symbol', False):
        msg = ('Iterable of symbols must be given as '
               'second argument, not type %s: %s')
        raise ValueError(filldedent(msg % (type(symbols[0]), symbols[0])))

    # 如果未指定所有符号，则默认与符号相同
    if all_symbols is None:
        all_symbols = symbols

    # 保存原始结果到old_result
    old_result = result

    # 存储特定符号的补集和交集
    complements = {}
    intersections = {}

    # 当total_solveset_call等于total_conditionset时，表示solveset未能解析所有方程
    total_conditionset = -1
    total_solveset_call = -1

    def _unsolved_syms(eq, sort=False):
        """返回方程中未解出的符号集合。"""
        free = eq.free_symbols
        unsolved = (free - set(known_symbols)) & set(all_symbols)
        if sort:
            unsolved = list(unsolved)
            unsolved.sort(key=default_sort_key)
        return unsolved

    # 按照方程中潜在符号数量的顺序对方程进行排序
    eqs_in_better_order = list(
        ordered(system, lambda _: len(_unsolved_syms(_))))
    # 定义一个函数，将解集、交集字典和补集字典作为参数
    def add_intersection_complement(result, intersection_dict, complement_dict):
        # 如果 solveset 返回了某个符号的交集/补集，将其加入最终解集中
        final_result = []
        
        # 遍历结果集合中的每一个结果
        for res in result:
            # 复制当前结果，以便修改
            res_copy = res
            
            # 遍历当前结果中的每对键和值
            for key_res, value_res in res.items():
                # 初始化交集和补集为 None
                intersect_set, complement_set = None, None
                
                # 在交集字典中查找当前键对应的交集集合
                for key_sym, value_sym in intersection_dict.items():
                    if key_sym == key_res:
                        intersect_set = value_sym
                
                # 在补集字典中查找当前键对应的补集集合
                for key_sym, value_sym in complement_dict.items():
                    if key_sym == key_res:
                        complement_set = value_sym
                
                # 如果找到了交集或补集
                if intersect_set or complement_set:
                    # 将当前值转换为有限集
                    new_value = FiniteSet(value_res)
                    
                    # 如果存在交集且不是 S.Complexes，则计算交集
                    if intersect_set and intersect_set != S.Complexes:
                        new_value = Intersection(new_value, intersect_set)
                    
                    # 如果存在补集，则计算补集
                    if complement_set:
                        new_value = Complement(new_value, complement_set)
                    
                    # 如果新值是空集，则将当前结果标记为 None 并跳出循环
                    if new_value is S.EmptySet:
                        res_copy = None
                        break
                    
                    # 如果新值是有限集且只包含一个元素，则将其转换为普通集合
                    elif new_value.is_FiniteSet and len(new_value) == 1:
                        res_copy[key_res] = set(new_value).pop()
                    
                    # 否则直接将新值赋给当前键对应的值
                    else:
                        res_copy[key_res] = new_value
            
            # 如果当前结果不为 None，则将其加入最终结果集合
            if res_copy is not None:
                final_result.append(res_copy)
        
        # 返回最终的结果集合
        return final_result
    def _extract_main_soln(sym, sol, soln_imageset):
        """Separate the Complements, Intersections, ImageSet lambda expr and
        its base_set. This function returns the unmasked sol from different classes
        of sets and also returns the appended ImageSet elements in a
        soln_imageset dict: `{unmasked element: ImageSet}`.
        """
        # 如果解(sol)是ConditionSet类型，则提取其基础集合(base_set)
        if isinstance(sol, ConditionSet):
            sol = sol.base_set

        # 如果解(sol)是Complement类型，则提取其补集并记录
        if isinstance(sol, Complement):
            complements[sym] = sol.args[1]  # 记录补集
            sol = sol.args[0]  # 更新解为补集的主集合

        # 如果解(sol)是Union类型，则处理其成员
        if isinstance(sol, Union):
            sol_args = sol.args
            sol = S.EmptySet
            # 按顺序处理并附加FiniteSet元素
            for sol_arg2 in sol_args:
                if isinstance(sol_arg2, FiniteSet):
                    sol += sol_arg2  # 附加FiniteSet
                else:
                    sol += FiniteSet(sol_arg2)  # 直接附加其他类型如ImageSet

        # 如果解(sol)是Intersection类型，则处理其成员
        if isinstance(sol, Intersection):
            # Interval/Set总是位于args的第一个位置
            if sol.args[0] not in (S.Reals, S.Complexes):
                intersections[sym] = sol.args[0]  # 记录Intersection中的集合
            sol = sol.args[1]  # 更新解为Intersection中的第二个元素

        # 如果解(sol)是ImageSet类型，则处理其成员
        if isinstance(sol, ImageSet):
            soln_imagest = sol
            expr2 = sol.lamda.expr
            sol = FiniteSet(expr2)  # 更新解为ImageSet中的表达式
            soln_imageset[expr2] = soln_imagest  # 记录ImageSet到soln_imageset字典中

        # 如果解(sol)不是FiniteSet类型，则转换为FiniteSet
        if not isinstance(sol, FiniteSet):
            sol = FiniteSet(sol)

        return sol, soln_imageset
    # 检查排除条件，根据图像集合确认是否排除解
    def _check_exclude(rnew, imgset_yes):
        rnew_ = rnew  # 复制 rnew 到 rnew_
        if imgset_yes:
            # 替换所有虚拟变量（图像集合 lambda 变量）为零，再进行 `checksol`。
            # 考虑到 `checksol` 的基本解。
            rnew_copy = rnew.copy()  # 复制 rnew 到 rnew_copy
            dummy_n = imgset_yes[0]  # 获取第一个图像集合的虚拟变量
            for key_res, value_res in rnew_copy.items():
                rnew_copy[key_res] = value_res.subs(dummy_n, 0)  # 将虚拟变量替换为零
            rnew_ = rnew_copy  # 更新 rnew_
        
        # 如果满足 `exclude` 列表中任意表达式的条件，则 satisfy_exclude 为真。
        try:
            # 类似于 `Mod(-log(3), 2*I*pi)` 目前无法简化，因此 `checksol` 返回 `TypeError`。
            # 当此问题修复后，应删除此 try 块。Mod(-log(3), 2*I*pi) == -log(3)
            satisfy_exclude = any(
                checksol(d, rnew_) for d in exclude)
        except TypeError:
            satisfy_exclude = None  # 处理 `TypeError` 异常时，设为 None
        return satisfy_exclude

    # 恢复图像集合的原始状态，将 rnew 中与 original_imageset 键相交的项替换为原始图像
    def _restore_imgset(rnew, original_imageset, newresult):
        restore_sym = set(rnew.keys()) & \
            set(original_imageset.keys())  # 找到 rnew 和 original_imageset 公共的键集合
        for key_sym in restore_sym:
            img = original_imageset[key_sym]  # 获取原始图像
            rnew[key_sym] = img  # 替换 rnew 中的对应键为原始图像
        if rnew not in newresult:
            newresult.append(rnew)  # 如果 rnew 不在 newresult 中，则添加到 newresult 中

    # 将等式添加到结果列表中，如果不满足条件则删除解决方案
    def _append_eq(eq, result, res, delete_soln, n=None):
        u = Dummy('u')  # 创建虚拟变量 u
        if n:
            eq = eq.subs(n, 0)  # 如果 n 存在，则将 eq 中的 n 替换为零
        satisfy = eq if eq in (True, False) else checksol(u, u, eq, minimal=True)  # 检查等式是否满足
        if satisfy is False:
            delete_soln = True  # 如果不满足，则设定 delete_soln 为 True
            res = {}  # 清空 res
        else:
            result.append(res)  # 将 res 添加到 result 中
        return result, res, delete_soln
    def _append_new_soln(rnew, sym, sol, imgset_yes, soln_imageset,
                         original_imageset, newresult, eq=None):
        """If `rnew` (A dict <symbol: soln>) contains valid soln
        append it to `newresult` list.
        `imgset_yes` is (base, dummy_var) if there was imageset in previously
         calculated result(otherwise empty tuple). `original_imageset` is dict
         of imageset expr and imageset from this result.
        `soln_imageset` dict of imageset expr and imageset of new soln.
        """
        # 检查是否需要排除当前解
        satisfy_exclude = _check_exclude(rnew, imgset_yes)
        
        # 标记是否删除当前解
        delete_soln = False
        
        # 如果不需要排除当前解
        if not satisfy_exclude:
            local_n = None
            
            # 如果是图像集
            if imgset_yes:
                local_n = imgset_yes[0]
                base = imgset_yes[1]
                
                # 如果符号 `sym` 和解 `sol` 均不为空
                if sym and sol:
                    # 获取解中的虚拟变量列表
                    dummy_list = list(sol.atoms(Dummy))
                    
                    # 使用先前的图像集中的虚拟变量 `local_n`，替换解中的虚拟变量
                    local_n_list = [local_n for i in range(0, len(dummy_list))]
                    dummy_zip = zip(dummy_list, local_n_list)
                    
                    # 创建一个 Lambda 函数，用于替换解中的虚拟变量
                    lam = Lambda(local_n, sol.subs(dummy_zip))
                    
                    # 将新的图像集解添加到 rnew 中
                    rnew[sym] = ImageSet(lam, base)
                
                # 如果有等式 `eq` 给定
                if eq is not None:
                    newresult, rnew, delete_soln = _append_eq(eq, newresult, rnew, delete_soln, local_n)
            
            # 如果不是图像集，但有等式 `eq` 给定
            elif eq is not None:
                newresult, rnew, delete_soln = _append_eq(eq, newresult, rnew, delete_soln)
            
            # 如果解 `sol` 在 `soln_imageset` 字典中
            elif sol in soln_imageset.keys():
                # 将预先计算好的图像集解添加到 rnew 中
                rnew[sym] = soln_imageset[sol]
                
                # 恢复原始的图像集
                _restore_imgset(rnew, original_imageset, newresult)
            
            # 否则，将 rnew 直接添加到 newresult 中
            else:
                newresult.append(rnew)
        
        # 如果需要排除当前解
        elif satisfy_exclude:
            delete_soln = True
            rnew = {}  # 清空 rnew
        
        # 最终恢复原始的图像集
        _restore_imgset(rnew, original_imageset, newresult)
        
        # 返回更新后的 newresult 和删除标记
        return newresult, delete_soln
    def _new_order_result(result, eq):
        # 将结果分为第一优先级和第二优先级。首先应使用使等式值为零的结果 `res`，
        # 然后使用其他结果（第二优先级）。如果不这样做，可能会错过某些解。
        first_priority = []
        second_priority = []
        for res in result:
            # 检查结果中是否有任何不是 ImageSet 类型的值
            if not any(isinstance(val, ImageSet) for val in res.values()):
                # 如果通过代入使等式 `eq` 的结果为零，则添加到第一优先级
                if eq.subs(res) == 0:
                    first_priority.append(res)
                else:
                    # 否则添加到第二优先级
                    second_priority.append(res)
        if first_priority or second_priority:
            # 如果存在第一优先级或第二优先级的结果，则返回它们的组合
            return first_priority + second_priority
        # 否则返回原始结果
        return result

    # 使用 solveset_real 解决方程组，返回新的结果、solve_call1 和 cnd_call1
    new_result_real, solve_call1, cnd_call1 = _solve_using_known_values(
        old_result, solveset_real)
    # 使用 solveset_complex 解决方程组，返回新的结果、solve_call2 和 cnd_call2
    new_result_complex, solve_call2, cnd_call2 = _solve_using_known_values(
        old_result, solveset_complex)

    # 如果 total_solveset_call 等于 total_conditionset
    # 则 solveset 未能解决所有方程。
    # 在这种情况下，返回一个 ConditionSet。
    total_conditionset += (cnd_call1 + cnd_call2)
    total_solveset_call += (solve_call1 + solve_call2)

    if total_conditionset == total_solveset_call and total_solveset_call != -1:
        # 如果 total_conditionset 等于 total_solveset_call 且不为 -1，则返回条件集合
        return _return_conditionset(eqs_in_better_order, all_symbols)

    # 不保留重复的解决方案
    filtered_complex = []
    for i in list(new_result_complex):
        for j in list(new_result_real):
            if i.keys() != j.keys():
                continue
            # 检查是否所有对应项都满足 dummy_eq 条件，排除全部为整数的情况
            if all(a.dummy_eq(b) for a, b in zip(i.values(), j.values()) \
                if not (isinstance(a, int) and isinstance(b, int))):
                break
        else:
            filtered_complex.append(i)
    # 将实数和复数解合并为整体结果
    result = new_result_real + filtered_complex

    result_all_variables = []
    result_infinite = []
    for res in result:
        if not res:
            # 表示 {None : None}
            continue
        # 如果长度 < len(all_symbols) 表示有无限解
        # 某些或所有的解取决于一个符号。
        # 例如：{x: y+2} 然后最终解 {x: y+2, y: y}
        if len(res) < len(all_symbols):
            solved_symbols = res.keys()
            unsolved = list(filter(
                lambda x: x not in solved_symbols, all_symbols))
            for unsolved_sym in unsolved:
                res[unsolved_sym] = unsolved_sym
            result_infinite.append(res)
        if res not in result_all_variables:
            result_all_variables.append(res)

    if result_infinite:
        # 存在通用解
        # 例如: [{x: -1, y : 1}, {x : -y, y: y}] 然后
        # 返回 [{x : -y, y : y}]
        result_all_variables = result_infinite
    if intersections or complements:
        # 添加交集和补集
        result_all_variables = add_intersection_complement(
            result_all_variables, intersections, complements)

    # 转换为有序元组
    result = S.EmptySet
    # 遍历 result_all_variables 中的每个元素 r
    for r in result_all_variables:
        # 对于每个 r，创建一个临时列表 temp，其中包含 r 中每个符号对应的值
        temp = [r[symb] for symb in all_symbols]
        # 将 temp 转换为元组，并将其添加到 result 的有限集中
        result += FiniteSet(tuple(temp))
    # 返回包含所有结果的有限集 result
    return result
# Solve a system of equations symbolically using Sympy's solveset
def _solveset_work(system, symbols):
    soln = solveset(system[0], symbols[0])
    # If the solution is a FiniteSet, convert each element to a tuple
    if isinstance(soln, FiniteSet):
        _soln = FiniteSet(*[(s,) for s in soln])
        return _soln
    else:
        # If not a FiniteSet, wrap the solution in a tuple inside a FiniteSet
        return FiniteSet(tuple(FiniteSet(soln)))


# Handle a positive dimensional system of polynomial equations using a Groebner basis
def _handle_positive_dimensional(polys, symbols, denominators):
    from sympy.polys.polytools import groebner
    # Sort symbols for consistent ordering in Groebner basis calculation
    _symbols = list(symbols)
    _symbols.sort(key=default_sort_key)
    # Compute the Groebner basis of polys with respect to _symbols
    basis = groebner(polys, _symbols, polys=True)
    new_system = []
    # Convert each polynomial equation in the basis to an expression
    for poly_eq in basis:
        new_system.append(poly_eq.as_expr())
    result = [{}]
    # Substitute symbols in new_system using substitution function
    result = substitution(
        new_system, symbols, result, [],
        denominators)
    return result


# Handle a zero dimensional system of polynomial equations using Sympy's solve_poly_system
def _handle_zero_dimensional(polys, symbols, system):
    # Solve the system of polynomial equations polys with respect to symbols
    result = solve_poly_system(polys, *symbols)
    # Filter out non-solutions by checking against the original system equations
    result_update = S.EmptySet
    for res in result:
        dict_sym_value = dict(list(zip(symbols, res)))
        # Check if res satisfies all equations in system
        if all(checksol(eq, dict_sym_value) for eq in system):
            result_update += FiniteSet(res)
    return result_update


# Separate polynomial and non-polynomial equations from the system
def _separate_poly_nonpoly(system, symbols):
    polys = []
    polys_expr = []
    nonpolys = []
    # Store denominators involving symbols
    denominators = set()
    # Store expressions with radicals that were processed using unrad
    unrad_changed = []
    poly = None
    for eq in system:
        # Identify and store denominators involving symbols
        denominators.update(_simple_dens(eq, symbols))
        # Convert equality to expression if it's an equation
        if isinstance(eq, Eq):
            eq = eq.lhs - eq.rhs
        # Try to remove radicals using unrad function
        without_radicals = unrad(simplify(eq), *symbols)
        if without_radicals:
            unrad_changed.append(eq)
            eq_unrad, cov = without_radicals
            if not cov:
                eq = eq_unrad
        # If eq is an expression, convert it to polynomial and expression forms
        if isinstance(eq, Expr):
            eq = eq.as_numer_denom()[0]
            poly = eq.as_poly(*symbols, extension=True)
        elif simplify(eq).is_number:
            continue
        # Classify equations as polynomials or non-polynomials
        if poly is not None:
            polys.append(poly)
            polys_expr.append(poly.as_expr())
        else:
            nonpolys.append(eq)
    return polys, polys_expr, nonpolys, denominators, unrad_changed


# Handle system of polynomial equations by attempting direct solution or returning a new system
# for further processing using Groebner basis
def _handle_poly(polys, symbols):
    # _handle_poly(polys, symbols) -> (poly_sol, poly_eqs)
    #
    # We will return possible solution information to nonlinsolve as well as a
    # new system of polynomial equations to be solved if we cannot solve
    # everything directly here. The new system of polynomial equations will be
    # a lex-order Groebner basis for the original system. The lex basis
    # 定义特殊情况下的解集表示形式，用于非线性方程组的解和替换操作
    no_information = [{}]   # 没有解决任何方程的情况
    no_solutions = []       # 系统不一致，没有解决方案

    # 如果不需要进一步尝试解这些方程，则返回空方程列表
    no_equations = []

    # 检查是否存在非精确的多项式（即包含非精确的浮点数系数）
    inexact = any(not p.domain.is_Exact for p in polys)
    if inexact:
        # 如果有非精确的浮点数系数，将其转换为有理数以便计算 Groebner 基
        polys = [poly(nsimplify(p, rational=True)) for p in polys]

    # 计算 Groebner 基，使用 grevlex 排序
    basis = groebner(polys, symbols, order='grevlex', polys=False)

    #
    # 无解情况？
    #
    if 1 in basis:
        # 没有解
        poly_sol = no_solutions
        poly_eqs = no_equations

    #
    # 有限解集（零维情况）
    #
    elif basis.is_zero_dimensional:
        # 将 Groebner 基转换为 lex 排序
        basis = basis.fglm('lex')

        # 如果存在非精确的浮点系数，将其转换回浮点数
        if inexact:
            basis = [nfloat(p) for p in basis]

        # 尝试使用 solve_poly_system 解决零维情况的方程组
        try:
            result = solve_poly_system(basis, *symbols, strict=True)
        except UnsolvableFactorError:
            # 解失败，无法完全使用根式解决。返回 lex 排序的基础方程集合以便处理
            poly_sol = no_information
            poly_eqs = list(basis)
        else:
            # 解成功，返回解集及空的待解方程列表
            poly_sol = [dict(zip(symbols, res)) for res in result]
            poly_eqs = no_equations

    #
    # 无限解族（正维情况）
    #
    # 否则情况下，当无法使用fglm方法将grevlex基底转换为lex，同时solve_poly_system无法解决方程组时，
    # 我们希望返回一个lex基底，但由于无法使用fglm，因此在这里直接计算lex基底。
    # 重新计算基底所需的时间通常远少于通过替换解决新系统所需的时间。

    # 设置poly_sol为no_information，表示未找到多项式解
    poly_sol = no_information
    # 使用'lex'顺序计算Groebner基底，返回一个多项式列表
    poly_eqs = list(groebner(polys, symbols, order='lex', polys=False))

    # 如果设置了inexact标志，将每个多项式转换为浮点数
    if inexact:
        poly_eqs = [nfloat(p) for p in poly_eqs]

# 返回poly_sol和poly_eqs作为函数的结果
return poly_sol, poly_eqs
# 定义非线性方程组求解函数
def nonlinsolve(system, *symbols):
    r"""
    Solve system of $N$ nonlinear equations with $M$ variables, which means both
    under and overdetermined systems are supported. Positive dimensional
    system is also supported (A system with infinitely many solutions is said
    to be positive-dimensional). In a positive dimensional system the solution will
    be dependent on at least one symbol. Returns both real solution
    and complex solution (if they exist).

    Parameters
    ==========

    system : list of equations
        The target system of equations
    symbols : list of Symbols
        symbols should be given as a sequence eg. list

    Returns
    =======

    A :class:`~.FiniteSet` of ordered tuple of values of `symbols` for which the `system`
    has solution. Order of values in the tuple is same as symbols present in
    the parameter `symbols`.

    Please note that general :class:`~.FiniteSet` is unordered, the solution
    returned here is not simply a :class:`~.FiniteSet` of solutions, rather it
    is a :class:`~.FiniteSet` of ordered tuple, i.e. the first and only
    argument to :class:`~.FiniteSet` is a tuple of solutions, which is
    ordered, and, hence ,the returned solution is ordered.

    Also note that solution could also have been returned as an ordered tuple,
    FiniteSet is just a wrapper ``{}`` around the tuple. It has no other
    significance except for the fact it is just used to maintain a consistent
    output format throughout the solveset.

    For the given set of equations, the respective input types
    are given below:

    .. math:: xy - 1 = 0
    .. math:: 4x^2 + y^2 - 5 = 0

    ::

       system  = [x*y - 1, 4*x**2 + y**2 - 5]
       symbols = [x, y]

    Raises
    ======

    ValueError
        The input is not valid.
        The symbols are not given.
    AttributeError
        The input symbols are not `Symbol` type.

    Examples
    ========

    >>> from sympy import symbols, nonlinsolve
    >>> x, y, z = symbols('x, y, z', real=True)
    >>> nonlinsolve([x*y - 1, 4*x**2 + y**2 - 5], [x, y])
    {(-1, -1), (-1/2, -2), (1/2, 2), (1, 1)}

    1. Positive dimensional system and complements:

    >>> from sympy import pprint
    >>> from sympy.polys.polytools import is_zero_dimensional
    >>> a, b, c, d = symbols('a, b, c, d', extended_real=True)
    >>> eq1 =  a + b + c + d
    >>> eq2 = a*b + b*c + c*d + d*a
    >>> eq3 = a*b*c + b*c*d + c*d*a + d*a*b
    >>> eq4 = a*b*c*d - 1
    >>> system = [eq1, eq2, eq3, eq4]
    >>> is_zero_dimensional(system)
    False
    >>> pprint(nonlinsolve(system, [a, b, c, d]), use_unicode=False)
      -1       1               1      -1
    {(---, -d, -, {d} \ {0}), (-, -d, ---, {d} \ {0})}
       d       d               d       d
    >>> nonlinsolve([(x+y)**2 - 4, x + y - 2], [x, y])
    {(2 - y, y)}

    2. If some of the equations are non-polynomial then `nonlinsolve`
    will call the ``substitution`` function and return real and complex solutions,
    """
    # 如果系统是非线性方程组且零维，则使用非线性求解器求解，并返回解集（包括实数和复数解，如果存在）。
    >>> from sympy import exp, sin
    >>> nonlinsolve([exp(x) - sin(y), y**2 - 4], [x, y])
    {(ImageSet(Lambda(_n, I*(2*_n*pi + pi) + log(sin(2))), Integers), -2),
     (ImageSet(Lambda(_n, 2*_n*I*pi + log(sin(2))), Integers), 2)}
    
    # 如果系统是非线性多项式且零维，则使用非线性求解器求解，并返回解集（包括实数和复数解，如果存在），使用 solve_poly_system 函数。
    >>> from sympy import sqrt
    >>> nonlinsolve([x**2 - 2*y**2 -2, x*y - 2], [x, y])
    {(-2, -1), (2, 1), (-sqrt(2)*I, sqrt(2)*I), (sqrt(2)*I, -sqrt(2)*I)}
    
    # nonlinsolve 也可以解决一些线性系统（零维或正维），因为它使用 groebner 函数获取 Groebner 基础，然后使用 substitution 函数将其作为新的系统基础。但不推荐使用 nonlinsolve 解决线性系统，因为 linsolve 在一般线性系统上表现更好。
    >>> nonlinsolve([x + 2*y -z - 3, x - y - 4*z + 9, y + z - 4], [x, y, z])
    {(3*z - 5, 4 - z, z)}
    
    # 对只有多项式方程且只有实数解的系统使用 solve_poly_system 进行求解。
    >>> e1 = sqrt(x**2 + y**2) - 10
    >>> e2 = sqrt(y**2 + (-x + 10)**2) - 3
    >>> nonlinsolve((e1, e2), (x, y))
    {(191/20, -3*sqrt(391)/20), (191/20, 3*sqrt(391)/20)}
    >>> nonlinsolve([x**2 + 2/y - 2, x + y - 3], [x, y])
    {(1, 2), (1 - sqrt(5), 2 + sqrt(5)), (1 + sqrt(5), 2 - sqrt(5))}
    >>> nonlinsolve([x**2 + 2/y - 2, x + y - 3], [y, x])
    {(2, 1), (2 - sqrt(5), 1 + sqrt(5)), (2 + sqrt(5), 1 - sqrt(5))}
    
    # 建议使用符号代替三角函数或函数类。例如，用符号替换 $\sin(x)$，用符号替换 $f(x)$ 等。从 nonlinsolve 获取解后，可以使用 solveset 获取 $x$ 的值。
    # 与旧求解器 `_solve_system` 相比，nonlinsolve 的优势：
    # =============================================================
    # 1. 正维系统求解器：nonlinsolve 能够返回正维系统的解。它找到正维系统的 Groebner 基础（称为基础），然后可以开始解方程（优先使用基础中变量最少的方程），使用 solveset 并将已解出的解值代入基础的其他方程中，以获得最小变量的解。这里重要的是如何替换已知值及其在哪些方程中替换。
    # 2. 实数和复数解：nonlinsolve 返回实数和复数解。如果系统中所有方程都是多项式方程，则使用 solve_poly_system 返回实数和复数解。如果系统中不全是多项式方程，则使用 substitution 方法处理这些多项式和非多项式方程。
    """
    # 如果系统为空，则返回空集
    if not system:
        return S.EmptySet

    # 如果符号集合为空，则抛出值错误
    if not symbols:
        msg = ('Symbols must be given, for which solution of the '
               'system is to be found.')
        raise ValueError(filldedent(msg))

    # 如果symbols的第一个元素可迭代，则将symbols设置为该可迭代对象
    if hasattr(symbols[0], '__iter__'):
        symbols = symbols[0]

    # 如果symbols不是可迭代对象或为空，则抛出索引错误
    if not is_sequence(symbols) or not symbols:
        msg = ('Symbols must be given, for which solution of the '
               'system is to be found.')
        raise IndexError(filldedent(msg))

    # 将symbols中的每个元素转换为_sympify对象
    symbols = list(map(_sympify, symbols))

    # 重新组织系统和symbols，可能会进行符号替换
    system, symbols, swap = recast_to_symbols(system, symbols)

    # 如果存在符号替换，则通过nonlinsolve求解系统
    if swap:
        soln = nonlinsolve(system, symbols)
        # 返回经过符号替换后的解集
        return FiniteSet(*[tuple(i.xreplace(swap) for i in s) for s in soln])

    # 如果系统中只有一个方程且symbols中只有一个符号，则调用_solveset_work函数
    if len(system) == 1 and len(symbols) == 1:
        return _solveset_work(system, symbols)

    # nonlinsolve的主要代码从这里开始

    # 将系统分成多项式和非多项式部分
    polys, polys_expr, nonpolys, denominators, unrad_changed = \
        _separate_poly_nonpoly(system, symbols)

    # 初始化多项式方程和解字典
    poly_eqs = []
    poly_sol = [{}]

    # 如果存在多项式部分，则处理多项式方程
    if polys:
        poly_sol, poly_eqs = _handle_poly(polys, symbols)
        # 如果有解并且多项式方程的所有符号都已通过unrad_changed解决，则验证解
        if poly_sol and poly_sol[0]:
            poly_syms = set().union(*(eq.free_symbols for eq in polys))
            unrad_syms = set().union(*(eq.free_symbols for eq in unrad_changed))
            if unrad_syms == poly_syms and unrad_changed:
                poly_sol = [sol for sol in poly_sol if checksol(unrad_changed, sol)]

    # 将未解决的多项式和非多项式方程合并
    remaining = poly_eqs + nonpolys

    # 将解字典转换为元组的函数
    to_tuple = lambda sol: tuple(sol[s] for s in symbols)

    # 如果没有剩余方程需要解决，则直接从solve_poly_system返回解集
    if not remaining:
        return FiniteSet(*map(to_tuple, poly_sol))
    else:
        # 处理以下情况：
        #
        #  1. 如果 solve_poly_system 失败，则使用 Groebner 基础。
        #  2. 在正维度情况下使用 Groebner 基础。
        #  3. 处理任何非多项式方程。
        #
        # 如果 solve_poly_system 成功，则将这些解作为初步结果传入。
        subs_res = substitution(remaining, symbols, result=poly_sol, exclude=denominators)

        # 如果 substitution 返回的不是 FiniteSet 类型，则直接返回 subs_res
        if not isinstance(subs_res, FiniteSet):
            return subs_res

        # 检查由替换产生的解。当前仅对那些具有非 Set 变量值的解进行检查。
        if unrad_changed:
            # 将解转换为符号到解的字典形式
            result = [dict(zip(symbols, sol)) for sol in subs_res.args]
            # 选择出符合条件的解，条件是至少有一个变量是集合类型或者 checksol(unrad_changed, sol) 不等于 False
            correct_sols = [sol for sol in result if any(isinstance(v, Set) for v in sol)
                            or checksol(unrad_changed, sol) != False]
            # 将正确的解转换为 FiniteSet 并返回
            return FiniteSet(*map(to_tuple, correct_sols))
        else:
            # 如果没有进行非平方根处理，则直接返回 subs_res
            return subs_res
```