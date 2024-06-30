# `D:\src\scipysrc\sympy\sympy\assumptions\refine.py`

```
# 导入未来的注释标记，允许在类型注释中使用类名
from __future__ import annotations
# 导入 Callable 类型用于类型注释
from typing import Callable

# 导入 SymPy 核心模块中的各种符号、表达式类
from sympy.core import S, Add, Expr, Basic, Mul, Pow, Rational
# 导入逻辑运算模块中的模糊逻辑运算函数
from sympy.core.logic import fuzzy_not
# 导入布尔代数模块中的布尔类型
from sympy.logic.boolalg import Boolean

# 导入 SymPy 假设模块中的 ask 和 Q 函数，忽略类型检查
from sympy.assumptions import ask, Q  # type: ignore

# 定义一个函数 refine，接受一个表达式 expr 和假设 assumptions，默认为 True
def refine(expr, assumptions=True):
    """
    Simplify an expression using assumptions.

    Explanation
    ===========

    Unlike :func:`~.simplify()` which performs structural simplification
    without any assumption, this function transforms the expression into
    the form which is only valid under certain assumptions. Note that
    ``simplify()`` is generally not done in refining process.

    Refining boolean expression involves reducing it to ``S.true`` or
    ``S.false``. Unlike :func:`~.ask()`, the expression will not be reduced
    if the truth value cannot be determined.

    Examples
    ========

    >>> from sympy import refine, sqrt, Q
    >>> from sympy.abc import x
    >>> refine(sqrt(x**2), Q.real(x))
    Abs(x)
    >>> refine(sqrt(x**2), Q.positive(x))
    x

    >>> refine(Q.real(x), Q.positive(x))
    True
    >>> refine(Q.positive(x), Q.real(x))
    Q.positive(x)

    See Also
    ========

    sympy.simplify.simplify.simplify : Structural simplification without assumptions.
    sympy.assumptions.ask.ask : Query for boolean expressions using assumptions.
    """
    # 如果 expr 不是 Basic 类型，直接返回它自己
    if not isinstance(expr, Basic):
        return expr

    # 如果 expr 不是原子表达式，对其参数逐个递归调用 refine 函数
    if not expr.is_Atom:
        args = [refine(arg, assumptions) for arg in expr.args]
        # TODO: this will probably not work with Integral or Polynomial
        expr = expr.func(*args)

    # 如果 expr 有 _eval_refine 方法，则调用它
    if hasattr(expr, '_eval_refine'):
        ref_expr = expr._eval_refine(assumptions)
        if ref_expr is not None:
            return ref_expr

    # 获取表达式的类名
    name = expr.__class__.__name__
    # 从 handlers_dict 中获取处理函数，若不存在则返回 expr 自身
    handler = handlers_dict.get(name, None)
    if handler is None:
        return expr

    # 调用 handler 处理表达式，根据返回的新表达式判断是否进行进一步 refine
    new_expr = handler(expr, assumptions)
    if (new_expr is None) or (expr == new_expr):
        return expr

    # 如果 new_expr 不是 Expr 类型，则直接返回它
    if not isinstance(new_expr, Expr):
        return new_expr

    # 递归调用 refine 函数，继续进行简化
    return refine(new_expr, assumptions)


# 定义处理绝对值表达式的 refine_abs 函数
def refine_abs(expr, assumptions):
    """
    Handler for the absolute value.

    Examples
    ========

    >>> from sympy import Q, Abs
    >>> from sympy.assumptions.refine import refine_abs
    >>> from sympy.abc import x
    >>> refine_abs(Abs(x), Q.real(x))
    >>> refine_abs(Abs(x), Q.positive(x))
    x
    >>> refine_abs(Abs(x), Q.negative(x))
    -x

    """
    # 导入绝对值函数
    from sympy.functions.elementary.complexes import Abs
    # 获取绝对值的参数
    arg = expr.args[0]
    # 如果参数 arg 是实数并且不是负数，则返回 arg
    if ask(Q.real(arg), assumptions) and \
            fuzzy_not(ask(Q.negative(arg), assumptions)):
        # if it's nonnegative
        return arg
    # 如果参数 arg 是负数，则返回 -arg
    if ask(Q.negative(arg), assumptions):
        return -arg
    # 如果 arg 是 Mul 类型，则暂时不处理
    # 检查参数 arg 是否是 Mul 类的实例
    if isinstance(arg, Mul):
        # 对 arg 中每个元素进行细化处理，使用 assumptions 参数
        r = [refine(abs(a), assumptions) for a in arg.args]
        
        # 初始化两个空列表，用于分别存放非 Abs 类型和 Abs 类型的元素
        non_abs = []
        in_abs = []
        
        # 遍历 r 列表中的每个元素 i
        for i in r:
            # 如果 i 是 Abs 类型的实例，则将其第一个参数添加到 in_abs 列表中
            if isinstance(i, Abs):
                in_abs.append(i.args[0])
            # 否则将 i 添加到 non_abs 列表中
            else:
                non_abs.append(i)
        
        # 返回结果，其中 Mul(*non_abs) 构造一个 Mul 类型对象，Abs(Mul(*in_abs)) 构造一个 Abs 类型对象
        return Mul(*non_abs) * Abs(Mul(*in_abs))
# 定义函数 refine_Pow，用于处理 Pow 类型的表达式
def refine_Pow(expr, assumptions):
    """
    Handler for instances of Pow.

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.refine import refine_Pow
    >>> from sympy.abc import x,y,z
    >>> refine_Pow((-1)**x, Q.real(x))
    >>> refine_Pow((-1)**x, Q.even(x))
    1
    >>> refine_Pow((-1)**x, Q.odd(x))
    -1

    For powers of -1, even parts of the exponent can be simplified:

    >>> refine_Pow((-1)**(x+y), Q.even(x))
    (-1)**y
    >>> refine_Pow((-1)**(x+y+z), Q.odd(x) & Q.odd(z))
    (-1)**y
    >>> refine_Pow((-1)**(x+y+2), Q.odd(x))
    (-1)**(y + 1)
    >>> refine_Pow((-1)**(x+3), True)
    (-1)**(x + 1)

    """

    # 导入所需的函数库和模块
    from sympy.functions.elementary.complexes import Abs
    from sympy.functions import sign

    # 如果表达式的底数是绝对值函数 Abs
    if isinstance(expr.base, Abs):
        # 如果表达式的底数的第一个参数是实数，并且指数是偶数，根据假设简化表达式
        if ask(Q.real(expr.base.args[0]), assumptions) and \
                ask(Q.even(expr.exp), assumptions):
            return expr.base.args[0] ** expr.exp
    # 检查是否应用了 Q.real 条件到 expr.base，使用给定的假设
    if ask(Q.real(expr.base), assumptions):
        # 如果 expr.base 是一个数值
        if expr.base.is_number:
            # 如果 expr.exp 是偶数
            if ask(Q.even(expr.exp), assumptions):
                # 返回绝对值的 expr.base 的 expr.exp 次方
                return abs(expr.base) ** expr.exp
            # 如果 expr.exp 是奇数
            if ask(Q.odd(expr.exp), assumptions):
                # 返回符号乘以 expr.base 的绝对值的 expr.exp 次方
                return sign(expr.base) * abs(expr.base) ** expr.exp
        # 如果 expr.exp 是 Rational 类型
        if isinstance(expr.exp, Rational):
            # 如果 expr.base 是 Pow 类型
            if isinstance(expr.base, Pow):
                # 返回绝对值的 expr.base.base 的 (expr.base.exp * expr.exp) 次方
                return abs(expr.base.base) ** (expr.base.exp * expr.exp)

        # 如果 expr.base 是 S.NegativeOne
        if expr.base is S.NegativeOne:
            # 如果 expr.exp 是一个 Add 对象
            if expr.exp.is_Add:

                old = expr

                # 对于 (-1) 的幂，可以移除以下部分:
                #  - 偶数项
                #  - 奇数项的成对项
                #  - 单个奇数项 + 1
                #  - 数值常数 N 可以用 mod(N, 2) 替换

                # 获取 expr.exp 的系数和项
                coeff, terms = expr.exp.as_coeff_add()
                terms = set(terms)
                even_terms = set()
                odd_terms = set()
                initial_number_of_terms = len(terms)

                # 遍历所有项，根据假设检查其是否为偶数或奇数
                for t in terms:
                    if ask(Q.even(t), assumptions):
                        even_terms.add(t)
                    elif ask(Q.odd(t), assumptions):
                        odd_terms.add(t)

                # 从 terms 中移除偶数项和奇数项
                terms -= even_terms
                if len(odd_terms) % 2:
                    terms -= odd_terms
                    new_coeff = (coeff + S.One) % 2
                else:
                    terms -= odd_terms
                    new_coeff = coeff % 2

                # 如果新系数不等于旧系数或项的数量减少
                if new_coeff != coeff or len(terms) < initial_number_of_terms:
                    terms.add(new_coeff)
                    # 更新 expr 为 expr.base 的 Add(*terms) 次方
                    expr = expr.base**(Add(*terms))

                # 处理 (-1)**((-1)**n/2 + m/2) 的情况
                e2 = 2*expr.exp
                if ask(Q.even(e2), assumptions):
                    if e2.could_extract_minus_sign():
                        e2 *= expr.base
                if e2.is_Add:
                    i, p = e2.as_two_terms()
                    if p.is_Pow and p.base is S.NegativeOne:
                        if ask(Q.integer(p.exp), assumptions):
                            i = (i + 1)/2
                            if ask(Q.even(i), assumptions):
                                return expr.base**p.exp
                            elif ask(Q.odd(i), assumptions):
                                return expr.base**(p.exp + 1)
                            else:
                                return expr.base**(p.exp + i)

                # 如果旧表达式和当前表达式不同，则返回更新后的表达式
                if old != expr:
                    return expr
# 处理 atan2 函数的情况
def refine_atan2(expr, assumptions):
    # 导入 atan 函数
    from sympy.functions.elementary.trigonometric import atan
    # 提取表达式中的 y 和 x
    y, x = expr.args
    # 如果 y 是实数且 x 是正数，返回 atan(y / x)
    if ask(Q.real(y) & Q.positive(x), assumptions):
        return atan(y / x)
    # 如果 y 是负数且 x 是负数，返回 atan(y / x) - pi
    elif ask(Q.negative(y) & Q.negative(x), assumptions):
        return atan(y / x) - S.Pi
    # 如果 y 是正数且 x 是负数，返回 atan(y / x) + pi
    elif ask(Q.positive(y) & Q.negative(x), assumptions):
        return atan(y / x) + S.Pi
    # 如果 y 是零且 x 是负数，返回 pi
    elif ask(Q.zero(y) & Q.negative(x), assumptions):
        return S.Pi
    # 如果 y 是正数且 x 是零，返回 pi/2
    elif ask(Q.positive(y) & Q.zero(x), assumptions):
        return S.Pi/2
    # 如果 y 是负数且 x 是零，返回 -pi/2
    elif ask(Q.negative(y) & Q.zero(x), assumptions):
        return -S.Pi/2
    # 如果 y 和 x 都是零，返回 NaN
    elif ask(Q.zero(y) & Q.zero(x), assumptions):
        return S.NaN
    else:
        # 如果没有匹配的情况，返回原始表达式
        return expr


# 处理实部的情况
def refine_re(expr, assumptions):
    # 提取表达式的参数
    arg = expr.args[0]
    # 如果参数是实数，返回参数本身
    if ask(Q.real(arg), assumptions):
        return arg
    # 如果参数是虚数，返回 0
    if ask(Q.imaginary(arg), assumptions):
        return S.Zero
    # 其他情况下，调用 _refine_reim 函数处理
    return _refine_reim(expr, assumptions)


# 处理虚部的情况
def refine_im(expr, assumptions):
    # 提取表达式的参数
    arg = expr.args[0]
    # 如果参数是实数，返回 0
    if ask(Q.real(arg), assumptions):
        return S.Zero
    # 如果参数是虚数，返回 -I * arg
    if ask(Q.imaginary(arg), assumptions):
        return -S.ImaginaryUnit * arg
    # 其他情况下，调用 _refine_reim 函数处理
    return _refine_reim(expr, assumptions)


# 处理复数的参数的情况
def refine_arg(expr, assumptions):
    # 提取表达式的参数
    rg = expr.args[0]
    # 如果参数是正数，返回 0
    if ask(Q.positive(rg), assumptions):
        return S.Zero
    # 如果参数是负数，返回 pi
    if ask(Q.negative(rg), assumptions):
        return S.Pi
    # 其他情况下，返回 None
    return None
def _refine_reim(expr, assumptions):
    # 辅助函数，用于处理 refine_re 和 refine_im
    # 将表达式展开为复数形式
    expanded = expr.expand(complex=True)
    # 如果展开后与原表达式不同，则尝试精炼展开后的表达式
    if expanded != expr:
        refined = refine(expanded, assumptions)
        # 如果精炼后的表达式与展开后的表达式不同，则返回精炼后的表达式
        if refined != expanded:
            return refined
    # 最好保持表达式不变
    return None


def refine_sign(expr, assumptions):
    """
    处理符号函数的处理器。

    Examples
    ========

    >>> from sympy.assumptions.refine import refine_sign
    >>> from sympy import Symbol, Q, sign, im
    >>> x = Symbol('x', real=True)
    >>> expr = sign(x)
    >>> refine_sign(expr, Q.positive(x) & Q.nonzero(x))
    1
    >>> refine_sign(expr, Q.negative(x) & Q.nonzero(x))
    -1
    >>> refine_sign(expr, Q.zero(x))
    0
    >>> y = Symbol('y', imaginary=True)
    >>> expr = sign(y)
    >>> refine_sign(expr, Q.positive(im(y)))
    I
    >>> refine_sign(expr, Q.negative(im(y)))
    -I
    """
    arg = expr.args[0]
    # 如果参数是零，返回零
    if ask(Q.zero(arg), assumptions):
        return S.Zero
    # 如果参数是实数
    if ask(Q.real(arg)):
        # 如果参数是正数，返回1
        if ask(Q.positive(arg), assumptions):
            return S.One
        # 如果参数是负数，返回-1
        if ask(Q.negative(arg), assumptions):
            return S.NegativeOne
    # 如果参数是虚数
    if ask(Q.imaginary(arg)):
        arg_re, arg_im = arg.as_real_imag()
        # 如果虚部是正数，返回虚数单位
        if ask(Q.positive(arg_im), assumptions):
            return S.ImaginaryUnit
        # 如果虚部是负数，返回负的虚数单位
        if ask(Q.negative(arg_im), assumptions):
            return -S.ImaginaryUnit
    # 默认情况下返回原始表达式
    return expr


def refine_matrixelement(expr, assumptions):
    """
    处理矩阵元素的处理器。

    Examples
    ========

    >>> from sympy.assumptions.refine import refine_matrixelement
    >>> from sympy import MatrixSymbol, Q
    >>> X = MatrixSymbol('X', 3, 3)
    >>> refine_matrixelement(X[0, 1], Q.symmetric(X))
    X[0, 1]
    >>> refine_matrixelement(X[1, 0], Q.symmetric(X))
    X[0, 1]
    """
    from sympy.matrices.expressions.matexpr import MatrixElement
    matrix, i, j = expr.args
    # 如果矩阵是对称的
    if ask(Q.symmetric(matrix), assumptions):
        # 如果元素的行列索引差可以提取负号，则返回原始表达式
        if (i - j).could_extract_minus_sign():
            return expr
        # 否则返回对应的对称元素
        return MatrixElement(matrix, j, i)

handlers_dict: dict[str, Callable[[Expr, Boolean], Expr]] = {
    'Abs': refine_abs,
    'Pow': refine_Pow,
    'atan2': refine_atan2,
    're': refine_re,
    'im': refine_im,
    'arg': refine_arg,
    'sign': refine_sign,
    'MatrixElement': refine_matrixelement
}
```