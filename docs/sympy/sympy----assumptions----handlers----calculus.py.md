# `D:\src\scipysrc\sympy\sympy\assumptions\handlers\calculus.py`

```
"""
This module contains query handlers responsible for calculus queries:
infinitesimal, finite, etc.
"""

# 导入必要的模块和类
from sympy.assumptions import Q, ask
from sympy.core import Add, Mul, Pow, Symbol
from sympy.core.numbers import (NegativeInfinity, GoldenRatio,
    Infinity, Exp1, ComplexInfinity, ImaginaryUnit, NaN, Number, Pi, E,
    TribonacciConstant)
from sympy.functions import cos, exp, log, sign, sin
from sympy.logic.boolalg import conjuncts

# 导入自定义的谓词类
from ..predicates.calculus import (FinitePredicate, InfinitePredicate,
    PositiveInfinitePredicate, NegativeInfinitePredicate)

# 定义 Symbol 类型的有限谓词处理函数
@FinitePredicate.register(Symbol)
def _(expr, assumptions):
    """
    Handles Symbol.
    """
    # 如果符号已知是否有限，返回其有限性
    if expr.is_finite is not None:
        return expr.is_finite
    # 如果在假设中有限性已知，则返回 True
    if Q.finite(expr) in conjuncts(assumptions):
        return True
    # 否则返回 None，表示有限性未知
    return None

# 定义 Add 类型的有限谓词处理函数
@FinitePredicate.register(Add)
def _(expr, assumptions):
    """
    Return True if expr is bounded, False if not and None if unknown.

    Truth Table:

    +-------+-----+-----------+-----------+
    |       |     |           |           |
    |       |  B  |     U     |     ?     |
    |       |     |           |           |
    +-------+-----+---+---+---+---+---+---+
    |       |     |   |   |   |   |   |   |
    |       |     |'+'|'-'|'x'|'+'|'-'|'x'|
    |       |     |   |   |   |   |   |   |
    +-------+-----+---+---+---+---+---+---+
    |       |     |           |           |
    |   B   |  B  |     U     |     ?     |
    |       |     |           |           |
    +---+---+-----+---+---+---+---+---+---+
    |   |   |     |   |   |   |   |   |   |
    |   |'+'|     | U | ? | ? | U | ? | ? |
    |   |   |     |   |   |   |   |   |   |
    |   +---+-----+---+---+---+---+---+---+
    |   |   |     |   |   |   |   |   |   |
    | U |'-'|     | ? | U | ? | ? | U | ? |
    |   |   |     |   |   |   |   |   |   |
    |   +---+-----+---+---+---+---+---+---+
    |   |   |     |           |           |
    |   |'x'|     |     ?     |     ?     |
    |   |   |     |           |           |
    +---+---+-----+---+---+---+---+---+---+
    |       |     |           |           |
    |   ?   |     |           |     ?     |
    |       |     |           |           |
    +-------+-----+-----------+---+---+---+

        * 'B' = Bounded

        * 'U' = Unbounded

        * '?' = unknown boundedness

        * '+' = positive sign

        * '-' = negative sign

        * 'x' = sign unknown

        * All Bounded -> True

        * 1 Unbounded and the rest Bounded -> False

        * >1 Unbounded, all with same known sign -> False

        * Any Unknown and unknown sign -> None

        * Else -> None

    When the signs are not the same you can have an undefined
    result as in oo - oo, hence 'bounded' is also undefined.
    """
    sign = -1  # 符号表示未知或无穷大
    result = True
    # 遍历表达式的每一个参数
    for arg in expr.args:
        # 查询参数是否有界，并根据假设返回结果
        _bounded = ask(Q.finite(arg), assumptions)
        # 如果参数已经被证明有界，则跳过此参数的处理
        if _bounded:
            continue
        
        # 查询参数是否为扩展正数，并根据假设返回结果
        s = ask(Q.extended_positive(arg), assumptions)
        
        # 如果已经有过多个符号，或者当前参数的符号与之前不一致，
        # 或者当前参数符号为None且之前的符号和有界性任一为None，则返回None
        if sign != -1 and s != sign or \
                s is None and None in (_bounded, sign):
            return None
        else:
            # 更新当前已知的符号
            sign = s
        
        # 如果结果不为False，则保持不变
        if result is not False:
            result = _bounded
    
    # 返回最终结果
    return result
# 注册一个函数来处理 Mul 类型表达式，实现其有限性谓词
@FinitePredicate.register(Mul)
def _(expr, assumptions):
    """
    Return True if expr is bounded, False if not and None if unknown.

    Truth Table:

    +---+---+---+--------+
    |   |   |   |        |
    |   | B | U |   ?    |
    |   |   |   |        |
    +---+---+---+---+----+
    |   |   |   |   |    |
    |   |   |   | s | /s |
    |   |   |   |   |    |
    +---+---+---+---+----+
    |   |   |   |        |
    | B | B | U |   ?    |
    |   |   |   |        |
    +---+---+---+---+----+
    |   |   |   |   |    |
    | U |   | U | U | ?  |
    |   |   |   |   |    |
    +---+---+---+---+----+
    |   |   |   |        |
    | ? |   |   |   ?    |
    |   |   |   |        |
    +---+---+---+---+----+

        * B = Bounded

        * U = Unbounded

        * ? = unknown boundedness

        * s = signed (hence nonzero)

        * /s = not signed
    """
    result = True
    for arg in expr.args:
        # 查询每个参数是否有限，根据谓词的假设
        _bounded = ask(Q.finite(arg), assumptions)
        if _bounded:
            continue
        elif _bounded is None:
            # 如果有一个参数的有限性未知，根据其他条件判断整体有限性
            if result is None:
                return None
            if ask(Q.extended_nonzero(arg), assumptions) is None:
                return None
            if result is not False:
                result = None
        else:
            result = False
    return result

# 注册一个函数来处理 Pow 类型表达式，实现其有限性谓词
@FinitePredicate.register(Pow)
def _(expr, assumptions):
    """
    * Unbounded ** NonZero -> Unbounded

    * Bounded ** Bounded -> Bounded

    * Abs()<=1 ** Positive -> Bounded

    * Abs()>=1 ** Negative -> Bounded

    * Otherwise unknown
    """
    if expr.base == E:
        # 特殊情况，底数为自然常数 e，判断指数的有限性
        return ask(Q.finite(expr.exp), assumptions)

    base_bounded = ask(Q.finite(expr.base), assumptions)
    exp_bounded = ask(Q.finite(expr.exp), assumptions)
    if base_bounded is None and exp_bounded is None:  # 常见情况
        return None
    if base_bounded is False and ask(Q.extended_nonzero(expr.exp), assumptions):
        return False
    if base_bounded and exp_bounded:
        return True
    if (abs(expr.base) <= 1) == True and ask(Q.extended_positive(expr.exp), assumptions):
        return True
    if (abs(expr.base) >= 1) == True and ask(Q.extended_negative(expr.exp), assumptions):
        return True
    if (abs(expr.base) >= 1) == True and exp_bounded is False:
        return False
    return None

# 注册一个函数来处理 exp 类型表达式，实现其有限性谓词
@FinitePredicate.register(exp)
def _(expr, assumptions):
    return ask(Q.finite(expr.exp), assumptions)

# 注册一个函数来处理 log 类型表达式，实现其有限性谓词
@FinitePredicate.register(log)
def _(expr, assumptions):
    # 如果参数是无穷大，返回 False
    if ask(Q.infinite(expr.args[0]), assumptions):
        return False
    return ask(~Q.zero(expr.args[0]), assumptions)

# 注册一个函数来处理 cos, sin, Number, Pi, Exp1, GoldenRatio,
# TribonacciConstant, ImaginaryUnit, sign 类型表达式，实现其有限性谓词
@FinitePredicate.register_many(cos, sin, Number, Pi, Exp1, GoldenRatio,
    TribonacciConstant, ImaginaryUnit, sign)
def _(expr, assumptions):
    return True

# 注册一个函数来处理 ComplexInfinity, Infinity, NegativeInfinity 类型表达式，实现其有限性谓词
@FinitePredicate.register_many(ComplexInfinity, Infinity, NegativeInfinity)
def _(expr, assumptions):
    # 这些表达式都是无限的，返回 False
    # 返回布尔值 False，表示函数执行未成功或条件未满足
    return False
# 使用 FinitePredicate 注册 NaN，返回 None
@FinitePredicate.register(NaN)
def _(expr, assumptions):
    return None


# 使用 InfinitePredicate 注册 ComplexInfinity, Infinity, NegativeInfinity，返回 True
@InfinitePredicate.register_many(ComplexInfinity, Infinity, NegativeInfinity)
def _(expr, assumptions):
    return True


# 使用 PositiveInfinitePredicate 注册 Infinity，返回 True
@PositiveInfinitePredicate.register(Infinity)
def _(expr, assumptions):
    return True


# 使用 PositiveInfinitePredicate 注册 NegativeInfinity, ComplexInfinity，返回 False
@PositiveInfinitePredicate.register_many(NegativeInfinity, ComplexInfinity)
def _(expr, assumptions):
    return False


# 使用 NegativeInfinitePredicate 注册 NegativeInfinity，返回 True
@NegativeInfinitePredicate.register(NegativeInfinity)
def _(expr, assumptions):
    return True


# 使用 NegativeInfinitePredicate 注册 Infinity, ComplexInfinity，返回 False
@NegativeInfinitePredicate.register_many(Infinity, ComplexInfinity)
def _(expr, assumptions):
    return False
```