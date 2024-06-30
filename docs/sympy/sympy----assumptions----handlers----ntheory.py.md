# `D:\src\scipysrc\sympy\sympy\assumptions\handlers\ntheory.py`

```
"""
Handlers for keys related to number theory: prime, even, odd, etc.
"""

# 导入必要的模块和函数
from sympy.assumptions import Q, ask
from sympy.core import Add, Basic, Expr, Float, Mul, Pow, S
from sympy.core.numbers import (ImaginaryUnit, Infinity, Integer, NaN,
    NegativeInfinity, NumberSymbol, Rational, int_valued)
from sympy.functions import Abs, im, re
from sympy.ntheory import isprime

# 导入自定义断言和谓词
from sympy.multipledispatch import MDNotImplementedError
from ..predicates.ntheory import (PrimePredicate, CompositePredicate,
    EvenPredicate, OddPredicate)


# PrimePredicate

# 定义处理表达式的函数，检查是否为整数
def _PrimePredicate_number(expr, assumptions):
    # 判断是否为精确值
    exact = not expr.atoms(Float)
    try:
        i = int(expr.round())
        if (expr - i).equals(0) is False:
            raise TypeError
    except TypeError:
        return False
    # 如果是精确值，则检查是否为质数
    if exact:
        return isprime(i)
    # 如果不是精确值，则不提供True或False，因为数字是近似值

# 注册针对Expr类型的PrimePredicate处理方法
@PrimePredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_prime
    if ret is None:
        raise MDNotImplementedError
    return ret

# 注册针对Basic类型的PrimePredicate处理方法
@PrimePredicate.register(Basic)
def _(expr, assumptions):
    if expr.is_number:
        return _PrimePredicate_number(expr, assumptions)

# 注册针对Mul类型的PrimePredicate处理方法
@PrimePredicate.register(Mul)
def _(expr, assumptions):
    if expr.is_number:
        return _PrimePredicate_number(expr, assumptions)
    # 检查Mul表达式中的每个参数是否为整数
    for arg in expr.args:
        if not ask(Q.integer(arg), assumptions):
            return None
    # 如果Mul表达式中的某个参数既是数字又是合数，则返回False
    for arg in expr.args:
        if arg.is_number and arg.is_composite:
            return False

# 注册针对Pow类型的PrimePredicate处理方法
@PrimePredicate.register(Pow)
def _(expr, assumptions):
    """
    Integer**Integer     -> !Prime
    """
    if expr.is_number:
        return _PrimePredicate_number(expr, assumptions)
    # 检查指数和底数是否都是整数
    if ask(Q.integer(expr.exp), assumptions) and \
            ask(Q.integer(expr.base), assumptions):
        return False

# 注册针对Integer类型的PrimePredicate处理方法
@PrimePredicate.register(Integer)
def _(expr, assumptions):
    return isprime(expr)

# 注册针对Rational, Infinity, NegativeInfinity, ImaginaryUnit等类型的PrimePredicate处理方法
@PrimePredicate.register_many(Rational, Infinity, NegativeInfinity, ImaginaryUnit)
def _(expr, assumptions):
    return False

# 注册针对Float类型的PrimePredicate处理方法
@PrimePredicate.register(Float)
def _(expr, assumptions):
    return _PrimePredicate_number(expr, assumptions)

# 注册针对NumberSymbol类型的PrimePredicate处理方法
@PrimePredicate.register(NumberSymbol)
def _(expr, assumptions):
    return _PrimePredicate_number(expr, assumptions)

# 注册针对NaN类型的PrimePredicate处理方法
@PrimePredicate.register(NaN)
def _(expr, assumptions):
    return None


# CompositePredicate

# 注册针对Expr类型的CompositePredicate处理方法
@CompositePredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_composite
    if ret is None:
        raise MDNotImplementedError
    return ret

# 注册针对Basic类型的CompositePredicate处理方法
@CompositePredicate.register(Basic)
def _(expr, assumptions):
    _positive = ask(Q.positive(expr), assumptions)
    # 如果 _positive 为真，则进入条件判断
    if _positive:
        # 调用 ask 函数，验证 expr 是否为整数，并基于 assumptions 给出答案
        _integer = ask(Q.integer(expr), assumptions)
        # 如果 _integer 为真，则进入条件判断
        if _integer:
            # 调用 ask 函数，验证 expr 是否为素数，并基于 assumptions 给出答案
            _prime = ask(Q.prime(expr), assumptions)
            # 如果 _prime 为 None，则直接返回，表示无法确定是否为素数
            if _prime is None:
                return
            # 如果 expr 等于 1，则直接返回 False，因为 1 是正整数但不是素数
            if expr.equals(1):
                return False
            # 返回 expr 是否不是素数的结果，即是否为合数
            return not _prime
        else:
            # 如果 _integer 不为真，则直接返回 _integer 的值（通常为 None 或 False）
            return _integer
    else:
        # 如果 _positive 不为真，则直接返回 _positive 的值（通常为 False）
        return _positive
# EvenPredicate

# 定义一个私有方法，用于判断表达式是否为偶数
def _EvenPredicate_number(expr, assumptions):
    # 如果表达式是浮点数或者Float类型的对象
    if isinstance(expr, (float, Float)):
        # 如果其被视为整数值
        if int_valued(expr):
            return None  # 返回空值
        return False  # 否则返回假
    try:
        i = int(expr.round())  # 尝试将表达式四舍五入为整数
    except TypeError:
        return False  # 如果无法转换为整数则返回假
    # 检查表达式减去其整数部分是否为零，判断其是否为整数
    if not (expr - i).equals(0):
        return False  # 不是整数则返回假
    return i % 2 == 0  # 返回表达式是否为偶数的判断结果

# 注册函数，处理Expr类型的表达式
@EvenPredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_even  # 调用表达式对象的is_even方法判断是否为偶数
    if ret is None:
        raise MDNotImplementedError  # 如果返回值为None，则抛出未实现错误
    return ret  # 返回判断结果

# 注册函数，处理Basic类型的表达式
@EvenPredicate.register(Basic)
def _(expr, assumptions):
    if expr.is_number:
        return _EvenPredicate_number(expr, assumptions)  # 如果是数字则调用私有方法判断是否为偶数

# 注册函数，处理Mul类型的表达式
@EvenPredicate.register(Mul)
def _(expr, assumptions):
    """
    Even * Integer    -> Even
    Even * Odd        -> Even
    Integer * Odd     -> ?
    Odd * Odd         -> Odd
    Even * Even       -> Even
    Integer * Integer -> Even if Integer + Integer = Odd
    otherwise         -> ?
    """
    if expr.is_number:
        return _EvenPredicate_number(expr, assumptions)  # 如果是数字则调用私有方法判断是否为偶数
    even, odd, irrational, acc = False, 0, False, 1  # 初始化变量
    for arg in expr.args:
        # 检查所有参数是否为整数并且至少有一个为偶数
        if ask(Q.integer(arg), assumptions):
            if ask(Q.even(arg), assumptions):
                even = True
            elif ask(Q.odd(arg), assumptions):
                odd += 1
            elif not even and acc != 1:
                if ask(Q.odd(acc + arg), assumptions):
                    even = True
        elif ask(Q.irrational(arg), assumptions):
            # 一个无理数使结果为False，两个使结果未定义
            if irrational:
                break
            irrational = True
        else:
            break
        acc = arg
    else:
        if irrational:
            return False
        if even:
            return True
        if odd == len(expr.args):
            return False

# 注册函数，处理Add类型的表达式
@EvenPredicate.register(Add)
def _(expr, assumptions):
    """
    Even + Odd  -> Odd
    Even + Even -> Even
    Odd  + Odd  -> Even
    """
    if expr.is_number:
        return _EvenPredicate_number(expr, assumptions)  # 如果是数字则调用私有方法判断是否为偶数
    _result = True  # 初始化结果为真
    for arg in expr.args:
        if ask(Q.even(arg), assumptions):
            pass
        elif ask(Q.odd(arg), assumptions):
            _result = not _result
        else:
            break
    else:
        return _result

# 注册函数，处理Pow类型的表达式
@EvenPredicate.register(Pow)
def _(expr, assumptions):
    if expr.is_number:
        return _EvenPredicate_number(expr, assumptions)  # 如果是数字则调用私有方法判断是否为偶数
    if ask(Q.integer(expr.exp), assumptions):
        if ask(Q.positive(expr.exp), assumptions):
            return ask(Q.even(expr.base), assumptions)
        elif ask(~Q.negative(expr.exp) & Q.odd(expr.base), assumptions):
            return False
        elif expr.base is S.NegativeOne:
            return False

# 注册函数，处理Integer类型的表达式
@EvenPredicate.register(Integer)
def _(expr, assumptions):
    return not bool(expr.p & 1)  # 返回整数是否为偶数的判断结果

# 注册多个函数，处理Rational、Infinity、NegativeInfinity、ImaginaryUnit等类型的表达式
@EvenPredicate.register_many(Rational, Infinity, NegativeInfinity, ImaginaryUnit)
# 定义一个函数 `_`，接受表达式和假设条件作为参数，总是返回 False
def _(expr, assumptions):
    return False

# 为 `EvenPredicate` 的 `NumberSymbol` 类型注册一个匿名函数
@EvenPredicate.register(NumberSymbol)
def _(expr, assumptions):
    # 调用 `_EvenPredicate_number` 函数处理表达式和假设条件，返回结果
    return _EvenPredicate_number(expr, assumptions)

# 为 `EvenPredicate` 的 `Abs` 类型注册一个匿名函数
@EvenPredicate.register(Abs)
def _(expr, assumptions):
    # 如果表达式参数是实数，则询问是否是偶数，返回结果
    if ask(Q.real(expr.args[0]), assumptions):
        return ask(Q.even(expr.args[0]), assumptions)

# 为 `EvenPredicate` 的 `re` 类型注册一个匿名函数
@EvenPredicate.register(re)
def _(expr, assumptions):
    # 如果表达式参数是实数，则询问是否是偶数，返回结果
    if ask(Q.real(expr.args[0]), assumptions):
        return ask(Q.even(expr.args[0]), assumptions)

# 为 `EvenPredicate` 的 `im` 类型注册一个匿名函数
@EvenPredicate.register(im)
def _(expr, assumptions):
    # 如果表达式参数是实数，则始终返回 True
    if ask(Q.real(expr.args[0]), assumptions):
        return True

# 为 `EvenPredicate` 的 `NaN` 类型注册一个匿名函数
@EvenPredicate.register(NaN)
def _(expr, assumptions):
    # 总是返回 None
    return None

# 定义一个函数 `_`，为 `OddPredicate` 的 `Expr` 类型注册匿名函数
@OddPredicate.register(Expr)
def _(expr, assumptions):
    # 检查表达式是否为奇数，如果无法确定则引发未实现错误
    ret = expr.is_odd
    if ret is None:
        raise MDNotImplementedError
    return ret

# 为 `OddPredicate` 的 `Basic` 类型注册一个匿名函数
@OddPredicate.register(Basic)
def _(expr, assumptions):
    # 检查表达式是否为整数，如果是，则进一步检查是否为奇数，返回相反值
    _integer = ask(Q.integer(expr), assumptions)
    if _integer:
        _even = ask(Q.even(expr), assumptions)
        if _even is None:
            return None
        return not _even
    return _integer
```