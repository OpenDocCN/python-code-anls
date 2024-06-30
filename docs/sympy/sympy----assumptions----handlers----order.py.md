# `D:\src\scipysrc\sympy\sympy\assumptions\handlers\order.py`

```
"""
Handlers related to order relations: positive, negative, etc.
"""

# 导入符号计算库中需要使用的模块和函数
from sympy.assumptions import Q, ask
from sympy.core import Add, Basic, Expr, Mul, Pow
from sympy.core.logic import fuzzy_not, fuzzy_and, fuzzy_or
from sympy.core.numbers import E, ImaginaryUnit, NaN, I, pi
from sympy.functions import Abs, acos, acot, asin, atan, exp, factorial, log
from sympy.matrices import Determinant, Trace
from sympy.matrices.expressions.matexpr import MatrixElement

# 导入自定义异常类
from sympy.multipledispatch import MDNotImplementedError

# 导入订单相关的预测功能
from ..predicates.order import (NegativePredicate, NonNegativePredicate,
    NonZeroPredicate, ZeroPredicate, NonPositivePredicate, PositivePredicate,
    ExtendedNegativePredicate, ExtendedNonNegativePredicate,
    ExtendedNonPositivePredicate, ExtendedNonZeroPredicate,
    ExtendedPositivePredicate,)


# NegativePredicate

# 定义处理 Basic 类型的数值负数预测函数
def _NegativePredicate_number(expr, assumptions):
    r, i = expr.as_real_imag()
    # 如果虚部可以被符号化地证明为零，则仅评估实部；否则评估虚部，
    # 如果虚部实际上评估为零，则在实部和零之间进行比较。
    if not i:
        r = r.evalf(2)
        if r._prec != 1:
            return r < 0
    else:
        i = i.evalf(2)
        if i._prec != 1:
            if i != 0:
                return False
            r = r.evalf(2)
            if r._prec != 1:
                return r < 0

# 使用 Multiple Dispatch 注册 Basic 类型的数值负数预测函数
@NegativePredicate.register(Basic)
def _(expr, assumptions):
    if expr.is_number:
        return _NegativePredicate_number(expr, assumptions)

# 使用 Multiple Dispatch 注册 Expr 类型的负数预测函数
@NegativePredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_negative
    if ret is None:
        raise MDNotImplementedError
    return ret

# 使用 Multiple Dispatch 注册 Add 类型的负数预测函数
@NegativePredicate.register(Add)
def _(expr, assumptions):
    """
    Positive + Positive -> Positive,
    Negative + Negative -> Negative
    """
    if expr.is_number:
        return _NegativePredicate_number(expr, assumptions)

    r = ask(Q.real(expr), assumptions)
    if r is not True:
        return r

    nonpos = 0
    for arg in expr.args:
        if ask(Q.negative(arg), assumptions) is not True:
            if ask(Q.positive(arg), assumptions) is False:
                nonpos += 1
            else:
                break
    else:
        if nonpos < len(expr.args):
            return True

# 使用 Multiple Dispatch 注册 Mul 类型的负数预测函数
@NegativePredicate.register(Mul)
def _(expr, assumptions):
    if expr.is_number:
        return _NegativePredicate_number(expr, assumptions)
    result = None
    for arg in expr.args:
        if result is None:
            result = False
        if ask(Q.negative(arg), assumptions):
            result = not result
        elif ask(Q.positive(arg), assumptions):
            pass
        else:
            return
    return result

# 使用 Multiple Dispatch 注册 Pow 类型的负数预测函数
@NegativePredicate.register(Pow)
def _(expr, assumptions):
    """
    Real ** Even -> NonNegative
    Real ** Odd  -> same_as_base
    """
    NonNegative ** Positive -> NonNegative
    """
    # 如果表达式的基数为自然常数E，指数为任意表达式：
    if expr.base == E:
        # 指数函数始终为正数：
        if ask(Q.real(expr.exp), assumptions):
            return False
        return

    # 如果表达式是一个数值：
    if expr.is_number:
        # 调用专门处理数值的负数断言函数
        return _NegativePredicate_number(expr, assumptions)

    # 如果基数为实数：
    if ask(Q.real(expr.base), assumptions):
        # 如果基数为正数：
        if ask(Q.positive(expr.base), assumptions):
            # 如果指数为实数：
            if ask(Q.real(expr.exp), assumptions):
                return False
        # 如果指数为偶数：
        if ask(Q.even(expr.exp), assumptions):
            return False
        # 如果指数为奇数：
        if ask(Q.odd(expr.exp), assumptions):
            # 返回基数为负数的断言结果
            return ask(Q.negative(expr.base), assumptions)
@NegativePredicate.register_many(Abs, ImaginaryUnit)
def _(expr, assumptions):
    # 注册 Abs 和 ImaginaryUnit 类型的表达式为 NegativePredicate
    return False

@NegativePredicate.register(exp)
def _(expr, assumptions):
    # 如果表达式 exp 是实数，返回 False
    if ask(Q.real(expr.exp), assumptions):
        return False
    # 否则抛出 MDNotImplementedError 异常
    raise MDNotImplementedError


# NonNegativePredicate

@NonNegativePredicate.register(Basic)
def _(expr, assumptions):
    # 如果表达式是数值类型
    if expr.is_number:
        # 判断是否为非负数的模糊否定值
        notnegative = fuzzy_not(_NegativePredicate_number(expr, assumptions))
        # 如果是非负数，返回表达式是否为实数
        if notnegative:
            return ask(Q.real(expr), assumptions)
        # 否则返回 notnegative 的值
        else:
            return notnegative

@NonNegativePredicate.register(Expr)
def _(expr, assumptions):
    # 获取表达式是否为非负数的信息
    ret = expr.is_nonnegative
    # 如果信息为 None，抛出 MDNotImplementedError 异常
    if ret is None:
        raise MDNotImplementedError
    # 否则返回 ret
    return ret


# NonZeroPredicate

@NonZeroPredicate.register(Expr)
def _(expr, assumptions):
    # 获取表达式是否为非零的信息
    ret = expr.is_nonzero
    # 如果信息为 None，抛出 MDNotImplementedError 异常
    if ret is None:
        raise MDNotImplementedError
    # 否则返回 ret
    return ret

@NonZeroPredicate.register(Basic)
def _(expr, assumptions):
    # 如果表达式不是实数，返回 False
    if ask(Q.real(expr)) is False:
        return False
    # 如果表达式是数值类型
    if expr.is_number:
        # 对表达式进行评估
        i = expr.evalf(2)
        # 定义非零函数
        def nonz(i):
            if i._prec != 1:
                return i != 0
        # 返回评估结果中的非零值的模糊或
        return fuzzy_or(nonz(i) for i in i.as_real_imag())

@NonZeroPredicate.register(Add)
def _(expr, assumptions):
    # 如果所有参数都是正数或者都是负数，返回 True
    if all(ask(Q.positive(x), assumptions) for x in expr.args) \
            or all(ask(Q.negative(x), assumptions) for x in expr.args):
        return True

@NonZeroPredicate.register(Mul)
def _(expr, assumptions):
    # 遍历乘法表达式的每个参数
    for arg in expr.args:
        # 获取参数是否为非零的信息
        result = ask(Q.nonzero(arg), assumptions)
        # 如果有一个参数为非零，继续循环；否则返回结果
        if result:
            continue
        return result
    return True

@NonZeroPredicate.register(Pow)
def _(expr, assumptions):
    # 返回指数是否为非零的信息
    return ask(Q.nonzero(expr.base), assumptions)

@NonZeroPredicate.register(Abs)
def _(expr, assumptions):
    # 返回绝对值是否为非零的信息
    return ask(Q.nonzero(expr.args[0]), assumptions)

@NonZeroPredicate.register(NaN)
def _(expr, assumptions):
    # NaN 总是返回 None
    return None


# ZeroPredicate

@ZeroPredicate.register(Expr)
def _(expr, assumptions):
    # 获取表达式是否为零的信息
    ret = expr.is_zero
    # 如果信息为 None，抛出 MDNotImplementedError 异常
    if ret is None:
        raise MDNotImplementedError
    # 否则返回 ret
    return ret

@ZeroPredicate.register(Basic)
def _(expr, assumptions):
    # 返回表达式是否为非零和是否为实数的模糊与
    return fuzzy_and([fuzzy_not(ask(Q.nonzero(expr), assumptions)),
        ask(Q.real(expr), assumptions)])

@ZeroPredicate.register(Mul)
def _(expr, assumptions):
    # TODO: This should be deducible from the nonzero handler
    # 返回乘法表达式中的参数是否为零的模糊或
    return fuzzy_or(ask(Q.zero(arg), assumptions) for arg in expr.args)


# NonPositivePredicate

@NonPositivePredicate.register(Expr)
def _(expr, assumptions):
    # 获取表达式是否为非正数的信息
    ret = expr.is_nonpositive
    # 如果信息为 None，抛出 MDNotImplementedError 异常
    if ret is None:
        raise MDNotImplementedError
    # 否则返回 ret
    return ret

@NonPositivePredicate.register(Basic)
def _(expr, assumptions):
    # 如果表达式是数值类型
    if expr.is_number:
        # 判断是否为正数的模糊否定值
        notpositive = fuzzy_not(_PositivePredicate_number(expr, assumptions))
        # 如果是非正数，返回表达式是否为实数
        if notpositive:
            return ask(Q.real(expr), assumptions)
        # 否则返回 notpositive 的值
        else:
            return notpositive
# PositivePredicate

# 定义一个私有函数，用于处理表达式为实数加虚数的情况
def _PositivePredicate_number(expr, assumptions):
    r, i = expr.as_real_imag()
    
    # 如果虚部为零，则只评估实部；否则评估虚部，如果虚部为零则比较实部和零的大小关系
    if not i:
        r = r.evalf(2)
        if r._prec != 1:
            return r > 0
    else:
        i = i.evalf(2)
        if i._prec != 1:
            if i != 0:
                return False
            r = r.evalf(2)
            if r._prec != 1:
                return r > 0

# 将 Expr 类型的表达式注册到 PositivePredicate，返回其是否为正数
@PositivePredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_positive
    if ret is None:
        raise MDNotImplementedError
    return ret

# 将 Basic 类型的表达式注册到 PositivePredicate，如果是数字则调用 _PositivePredicate_number 处理
@PositivePredicate.register(Basic)
def _(expr, assumptions):
    if expr.is_number:
        return _PositivePredicate_number(expr, assumptions)

# 将 Mul 类型的表达式注册到 PositivePredicate
@PositivePredicate.register(Mul)
def _(expr, assumptions):
    if expr.is_number:
        return _PositivePredicate_number(expr, assumptions)
    
    result = True
    for arg in expr.args:
        if ask(Q.positive(arg), assumptions):
            continue
        elif ask(Q.negative(arg), assumptions):
            result = result ^ True
        else:
            return
    return result

# 将 Add 类型的表达式注册到 PositivePredicate
@PositivePredicate.register(Add)
def _(expr, assumptions):
    if expr.is_number:
        return _PositivePredicate_number(expr, assumptions)

    r = ask(Q.real(expr), assumptions)
    if r is not True:
        return r

    nonneg = 0
    for arg in expr.args:
        if ask(Q.positive(arg), assumptions) is not True:
            if ask(Q.negative(arg), assumptions) is False:
                nonneg += 1
            else:
                break
    else:
        if nonneg < len(expr.args):
            return True

# 将 Pow 类型的表达式注册到 PositivePredicate
@PositivePredicate.register(Pow)
def _(expr, assumptions):
    # 特殊情况：如果底数为 E
    if expr.base == E:
        if ask(Q.real(expr.exp), assumptions):
            return True
        if ask(Q.imaginary(expr.exp), assumptions):
            return ask(Q.even(expr.exp/(I*pi)), assumptions)
        return

    # 如果是数字，则调用 _PositivePredicate_number 处理
    if expr.is_number:
        return _PositivePredicate_number(expr, assumptions)
    
    # 如果底数为正数，并且指数为实数，则返回 True
    if ask(Q.positive(expr.base), assumptions):
        if ask(Q.real(expr.exp), assumptions):
            return True
    
    # 如果底数为负数，并且指数为偶数，则返回 True；如果指数为奇数，则返回 False
    if ask(Q.negative(expr.base), assumptions):
        if ask(Q.even(expr.exp), assumptions):
            return True
        if ask(Q.odd(expr.exp), assumptions):
            return False

# 将 exp 函数注册到 PositivePredicate
@PositivePredicate.register(exp)
def _(expr, assumptions):
    # 如果指数为实数，则返回 True
    if ask(Q.real(expr.exp), assumptions):
        return True
    # 如果指数为虚数，则检查是否为偶数的倍数
    if ask(Q.imaginary(expr.exp), assumptions):
        return ask(Q.even(expr.exp/(I*pi)), assumptions)

# 将 log 函数注册到 PositivePredicate
@PositivePredicate.register(log)
def _(expr, assumptions):
    # 检查参数是否为实数
    r = ask(Q.real(expr.args[0]), assumptions)
    if r is not True:
        return r
    # 检查参数是否大于 1
    if ask(Q.positive(expr.args[0] - 1), assumptions):
        return True
    # 如果 ask 函数返回 Q.negative(expr.args[0] - 1) 的结果为真，则执行以下操作
    if ask(Q.negative(expr.args[0] - 1), assumptions):
        # 返回 False
        return False
# 使用 PositivePredicate 的 register 方法注册 factorial 函数的断言
@PositivePredicate.register(factorial)
def _(expr, assumptions):
    # 从表达式的参数中获取 x
    x = expr.args[0]
    # 如果 x 是整数并且是正数，则返回 True
    if ask(Q.integer(x) & Q.positive(x), assumptions):
        return True

# 使用 PositivePredicate 的 register 方法注册 ImaginaryUnit 类的断言
@PositivePredicate.register(ImaginaryUnit)
def _(expr, assumptions):
    # 对于 ImaginaryUnit 类的表达式，总是返回 False
    return False

# 使用 PositivePredicate 的 register 方法注册 Abs 函数的断言
@PositivePredicate.register(Abs)
def _(expr, assumptions):
    # 检查 Abs 函数的参数是否非零，并根据给定的假设返回结果
    return ask(Q.nonzero(expr), assumptions)

# 使用 PositivePredicate 的 register 方法注册 Trace 类的断言
@PositivePredicate.register(Trace)
def _(expr, assumptions):
    # 如果 Trace 类的参数是正定的，则返回 True
    if ask(Q.positive_definite(expr.arg), assumptions):
        return True

# 使用 PositivePredicate 的 register 方法注册 Determinant 类的断言
@PositivePredicate.register(Determinant)
def _(expr, assumptions):
    # 如果 Determinant 类的参数是正定的，则返回 True
    if ask(Q.positive_definite(expr.arg), assumptions):
        return True

# 使用 PositivePredicate 的 register 方法注册 MatrixElement 类的断言
@PositivePredicate.register(MatrixElement)
def _(expr, assumptions):
    # 如果 MatrixElement 对象的 i 和 j 相等，并且其父对象是正定的，则返回 True
    if (expr.i == expr.j
            and ask(Q.positive_definite(expr.parent), assumptions)):
        return True

# 使用 PositivePredicate 的 register 方法注册 atan 函数的断言
@PositivePredicate.register(atan)
def _(expr, assumptions):
    # 检查 atan 函数的参数是否为正数，并根据给定的假设返回结果
    return ask(Q.positive(expr.args[0]), assumptions)

# 使用 PositivePredicate 的 register 方法注册 asin 函数的断言
@PositivePredicate.register(asin)
def _(expr, assumptions):
    x = expr.args[0]
    # 如果 x 是正数且不超过 1，或者是负数且不小于 -1，则返回 True 或 False
    if ask(Q.positive(x) & Q.nonpositive(x - 1), assumptions):
        return True
    if ask(Q.negative(x) & Q.nonnegative(x + 1), assumptions):
        return False

# 使用 PositivePredicate 的 register 方法注册 acos 函数的断言
@PositivePredicate.register(acos)
def _(expr, assumptions):
    x = expr.args[0]
    # 如果 x - 1 非正并且 x + 1 非负，则返回 True
    if ask(Q.nonpositive(x - 1) & Q.nonnegative(x + 1), assumptions):
        return True

# 使用 PositivePredicate 的 register 方法注册 acot 函数的断言
@PositivePredicate.register(acot)
def _(expr, assumptions):
    # 检查 acot 函数的参数是否为实数，并根据给定的假设返回结果
    return ask(Q.real(expr.args[0]), assumptions)

# 使用 PositivePredicate 的 register 方法注册 NaN 类的断言
@PositivePredicate.register(NaN)
def _(expr, assumptions):
    # 对于 NaN 类的表达式，返回 None
    return None


# 使用 ExtendedNegativePredicate 的 register 方法注册 object 的断言
@ExtendedNegativePredicate.register(object)
def _(expr, assumptions):
    # 检查表达式是否为负数或负无穷，并根据给定的假设返回结果
    return ask(Q.negative(expr) | Q.negative_infinite(expr), assumptions)


# 使用 ExtendedPositivePredicate 的 register 方法注册 object 的断言
@ExtendedPositivePredicate.register(object)
def _(expr, assumptions):
    # 检查表达式是否为正数或正无穷，并根据给定的假设返回结果
    return ask(Q.positive(expr) | Q.positive_infinite(expr), assumptions)


# 使用 ExtendedNonZeroPredicate 的 register 方法注册 object 的断言
@ExtendedNonZeroPredicate.register(object)
def _(expr, assumptions):
    # 检查表达式是否为负无穷、负数、零、正数或正无穷，并根据给定的假设返回结果
    return ask(
        Q.negative_infinite(expr) | Q.negative(expr) | Q.positive(expr) | Q.positive_infinite(expr),
        assumptions)


# 使用 ExtendedNonPositivePredicate 的 register 方法注册 object 的断言
@ExtendedNonPositivePredicate.register(object)
def _(expr, assumptions):
    # 检查表达式是否为负无穷、负数或零，并根据给定的假设返回结果
    return ask(
        Q.negative_infinite(expr) | Q.negative(expr) | Q.zero(expr),
        assumptions)


# 使用 ExtendedNonNegativePredicate 的 register 方法注册 object 的断言
@ExtendedNonNegativePredicate.register(object)
def _(expr, assumptions):
    # 检查表达式是否为零、正数或正无穷，并根据给定的假设返回结果
    return ask(
        Q.zero(expr) | Q.positive(expr) | Q.positive_infinite(expr),
        assumptions)
```