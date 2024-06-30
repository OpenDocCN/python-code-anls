# `D:\src\scipysrc\sympy\sympy\assumptions\handlers\sets.py`

```
"""
Handlers for predicates related to set membership: integer, rational, etc.
"""

from sympy.assumptions import Q, ask  # 导入符号逻辑相关的库函数 Q 和 ask
from sympy.core import Add, Basic, Expr, Mul, Pow, S  # 导入核心类和函数
from sympy.core.numbers import (AlgebraicNumber, ComplexInfinity, Exp1, Float,
    GoldenRatio, ImaginaryUnit, Infinity, Integer, NaN, NegativeInfinity,
    Number, NumberSymbol, Pi, pi, Rational, TribonacciConstant, E)  # 导入数学常数和特殊数值类
from sympy.core.logic import fuzzy_bool  # 导入模糊布尔逻辑函数
from sympy.functions import (Abs, acos, acot, asin, atan, cos, cot, exp, im,
    log, re, sin, tan)  # 导入数学函数
from sympy.core.numbers import I  # 导入虚数单位
from sympy.core.relational import Eq  # 导入关系运算符
from sympy.functions.elementary.complexes import conjugate  # 导入复数函数
from sympy.matrices import Determinant, MatrixBase, Trace  # 导入矩阵相关类
from sympy.matrices.expressions.matexpr import MatrixElement  # 导入矩阵元素类

from sympy.multipledispatch import MDNotImplementedError  # 导入多分派未实现错误

from .common import test_closed_group  # 导入测试闭合群的通用函数
from ..predicates.sets import (IntegerPredicate, RationalPredicate,
    IrrationalPredicate, RealPredicate, ExtendedRealPredicate,
    HermitianPredicate, ComplexPredicate, ImaginaryPredicate,
    AntihermitianPredicate, AlgebraicPredicate)  # 导入集合谓词类


# IntegerPredicate

def _IntegerPredicate_number(expr, assumptions):
    # 辅助函数：检查表达式是否为整数
    try:
        i = int(expr.round())  # 尝试将表达式四舍五入为整数
        if not (expr - i).equals(0):  # 如果表达式与其四舍五入后的整数值不相等
            raise TypeError  # 抛出类型错误
        return True  # 返回 True 表示表达式是整数
    except TypeError:
        return False  # 捕获类型错误时返回 False


@IntegerPredicate.register_many(int, Integer)  # 注册多个类型到 IntegerPredicate
def _(expr, assumptions):
    return True  # 表示所有 int 和 Integer 类型都被视为整数


@IntegerPredicate.register_many(Exp1, GoldenRatio, ImaginaryUnit, Infinity,
        NegativeInfinity, Pi, Rational, TribonacciConstant)
def _(expr, assumptions):
    return False  # 表示特定数学常数不是整数


@IntegerPredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_integer  # 调用表达式的 is_integer 方法
    if ret is None:
        raise MDNotImplementedError  # 如果 is_integer 返回 None，则抛出未实现错误
    return ret


@IntegerPredicate.register_many(Add, Pow)
def _(expr, assumptions):
    """
    * Integer + Integer       -> Integer
    * Integer + !Integer      -> !Integer
    * !Integer + !Integer -> ?
    """
    if expr.is_number:  # 如果表达式是数值
        return _IntegerPredicate_number(expr, assumptions)  # 调用辅助函数检查是否为整数
    return test_closed_group(expr, assumptions, Q.integer)  # 调用测试闭合群的通用函数


@IntegerPredicate.register(Mul)
def _(expr, assumptions):
    """
    * Integer*Integer      -> Integer
    * Integer*Irrational   -> !Integer
    * Odd/Even             -> !Integer
    * Integer*Rational     -> ?
    """
    if expr.is_number:  # 如果表达式是数值
        return _IntegerPredicate_number(expr, assumptions)  # 调用辅助函数检查是否为整数
    _output = True
    for arg in expr.args:  # 遍历表达式的每个参数
        if not ask(Q.integer(arg), assumptions):  # 如果参数不是整数
            if arg.is_Rational:  # 如果参数是有理数
                if arg.q == 2:
                    return ask(Q.even(2*expr), assumptions)  # 如果分母为 2，则检查是否为偶数
                if ~(arg.q & 1):
                    return None  # 如果分母为偶数，则返回 None
            elif ask(Q.irrational(arg), assumptions):  # 如果参数是无理数
                if _output:
                    _output = False  # 第一次遇到无理数后将 _output 置为 False
                else:
                    return  # 如果已经遇到过无理数，则返回
            else:
                return  # 如果不是整数也不是无理数，则返回
    # 返回函数的输出变量 _output
    return _output
# 注册 IntegerPredicate 的处理器函数，处理 Abs 表达式
@IntegerPredicate.register(Abs)
def _(expr, assumptions):
    return ask(Q.integer(expr.args[0]), assumptions)

# 注册多个处理器函数来处理 Determinant、MatrixElement 和 Trace 表达式
@IntegerPredicate.register_many(Determinant, MatrixElement, Trace)
def _(expr, assumptions):
    return ask(Q.integer_elements(expr.args[0]), assumptions)


# RationalPredicate

# 注册处理 Rational 类型表达式的处理器函数
@RationalPredicate.register(Rational)
def _(expr, assumptions):
    return True

# 注册处理 Float 类型表达式的处理器函数
@RationalPredicate.register(Float)
def _(expr, assumptions):
    return None

# 注册多个处理器函数来处理 Exp1、GoldenRatio、ImaginaryUnit 等表达式
@RationalPredicate.register_many(Exp1, GoldenRatio, ImaginaryUnit, Infinity,
    NegativeInfinity, Pi, TribonacciConstant)
def _(expr, assumptions):
    return False

# 注册处理 Expr 类型表达式的处理器函数
@RationalPredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_rational
    if ret is None:
        raise MDNotImplementedError
    return ret

# 注册多个处理器函数来处理 Add、Mul 类型表达式
@RationalPredicate.register_many(Add, Mul)
def _(expr, assumptions):
    """
    * Rational + Rational     -> Rational
    * Rational + !Rational    -> !Rational
    * !Rational + !Rational   -> ?
    """
    if expr.is_number:
        if expr.as_real_imag()[1]:
            return False
    return test_closed_group(expr, assumptions, Q.rational)

# 注册处理 Pow 类型表达式的处理器函数
@RationalPredicate.register(Pow)
def _(expr, assumptions):
    """
    * Rational ** Integer      -> Rational
    * Irrational ** Rational   -> Irrational
    * Rational ** Irrational   -> ?
    """
    if expr.base == E:
        x = expr.exp
        if ask(Q.rational(x), assumptions):
            return ask(~Q.nonzero(x), assumptions)
        return

    if ask(Q.integer(expr.exp), assumptions):
        return ask(Q.rational(expr.base), assumptions)
    elif ask(Q.rational(expr.exp), assumptions):
        if ask(Q.prime(expr.base), assumptions):
            return False

# 注册多个处理器函数来处理 asin、atan、cos、sin、tan 类型表达式
@RationalPredicate.register_many(asin, atan, cos, sin, tan)
def _(expr, assumptions):
    x = expr.args[0]
    if ask(Q.rational(x), assumptions):
        return ask(~Q.nonzero(x), assumptions)

# 注册处理 exp 类型表达式的处理器函数
@RationalPredicate.register(exp)
def _(expr, assumptions):
    x = expr.exp
    if ask(Q.rational(x), assumptions):
        return ask(~Q.nonzero(x), assumptions)

# 注册多个处理器函数来处理 acot、cot 类型表达式
@RationalPredicate.register_many(acot, cot)
def _(expr, assumptions):
    x = expr.args[0]
    if ask(Q.rational(x), assumptions):
        return False

# 注册多个处理器函数来处理 acos、log 类型表达式
@RationalPredicate.register_many(acos, log)
def _(expr, assumptions):
    x = expr.args[0]
    if ask(Q.rational(x), assumptions):
        return ask(~Q.nonzero(x - 1), assumptions)


# IrrationalPredicate

# 注册处理 Expr 类型表达式的处理器函数
@IrrationalPredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_irrational
    if ret is None:
        raise MDNotImplementedError
    return ret

# 注册处理 Basic 类型表达式的处理器函数
@IrrationalPredicate.register(Basic)
def _(expr, assumptions):
    _real = ask(Q.real(expr), assumptions)
    if _real:
        _rational = ask(Q.rational(expr), assumptions)
        if _rational is None:
            return None
        return not _rational
    else:
        return _real


# RealPredicate

# 定义处理 Real 类型表达式的处理函数
def _RealPredicate_number(expr, assumptions):
    # 让 as_real_imag() 先工作，因为表达式可能是
    # 从表达式中获取其实部和虚部，选择虚部，并保留两位小数进行数值评估
    i = expr.as_real_imag()[1].evalf(2)
    # 如果评估后的虚部精度不等于1，则返回非 i
    if i._prec != 1:
        return not i
    # 如果无法确定虚部是否为 0，则允许返回 None
    # 表示无法确定 i 是否为 0
# 注册多个函数为 RealPredicate 的处理器，处理 Abs, Exp1, Float, GoldenRatio, im, Pi, Rational, re, TribonacciConstant 类型的表达式
@RealPredicate.register_many(Abs, Exp1, Float, GoldenRatio, im, Pi, Rational, re, TribonacciConstant)
def _(expr, assumptions):
    # 总是返回 True，表示这些表达式被认为是实数
    return True

# 注册多个函数为 RealPredicate 的处理器，处理 ImaginaryUnit, Infinity, NegativeInfinity 类型的表达式
@RealPredicate.register_many(ImaginaryUnit, Infinity, NegativeInfinity)
def _(expr, assumptions):
    # 总是返回 False，表示这些表达式被认为不是实数
    return False

# 注册 Expr 类型的函数为 RealPredicate 的处理器
@RealPredicate.register(Expr)
def _(expr, assumptions):
    # 检查表达式是否为实数，如果返回 None 则抛出 MDNotImplementedError 异常
    ret = expr.is_real
    if ret is None:
        raise MDNotImplementedError
    return ret

# 注册 Add 类型的函数为 RealPredicate 的处理器
@RealPredicate.register(Add)
def _(expr, assumptions):
    """
    * Real + Real              -> Real
    * Real + (Complex & !Real) -> !Real
    """
    # 如果表达式是数字，调用 _RealPredicate_number 处理
    if expr.is_number:
        return _RealPredicate_number(expr, assumptions)
    # 否则调用 test_closed_group 函数来检查表达式的实数性质，使用 Q.real 的假设
    return test_closed_group(expr, assumptions, Q.real)

# 注册 Mul 类型的函数为 RealPredicate 的处理器
@RealPredicate.register(Mul)
def _(expr, assumptions):
    """
    * Real*Real               -> Real
    * Real*Imaginary          -> !Real
    * Imaginary*Imaginary     -> Real
    """
    # 如果表达式是数字，调用 _RealPredicate_number 处理
    if expr.is_number:
        return _RealPredicate_number(expr, assumptions)
    # 否则遍历表达式的参数，根据 Q.real 和 Q.imaginary 条件来确定结果
    result = True
    for arg in expr.args:
        if ask(Q.real(arg), assumptions):
            pass
        elif ask(Q.imaginary(arg), assumptions):
            result = result ^ True
        else:
            break
    else:
        return result

# 注册 Pow 类型的函数为 RealPredicate 的处理器
@RealPredicate.register(Pow)
def _(expr, assumptions):
    """
    * Real**Integer              -> Real
    * Positive**Real             -> Real
    * Real**(Integer/Even)       -> Real if base is nonnegative
    * Real**(Integer/Odd)        -> Real
    * Imaginary**(Integer/Even)  -> Real
    * Imaginary**(Integer/Odd)   -> not Real
    * Imaginary**Real            -> ? since Real could be 0 (giving real)
                                    or 1 (giving imaginary)
    * b**Imaginary               -> Real if log(b) is imaginary and b != 0
                                    and exponent != integer multiple of
                                    I*pi/log(b)
    * Real**Real                 -> ? e.g. sqrt(-1) is imaginary and
                                    sqrt(2) is not
    """
    # 如果表达式是数字，调用 _RealPredicate_number 处理
    if expr.is_number:
        return _RealPredicate_number(expr, assumptions)

    # 如果底数是自然对数 e
    if expr.base == E:
        return ask(
            Q.integer(expr.exp/I/pi) | Q.real(expr.exp), assumptions
        )

    # 如果底数是 exp 函数或者是形如 exp(base) 的形式
    if expr.base.func == exp or (expr.base.is_Pow and expr.base.base == E):
        if ask(Q.imaginary(expr.base.exp), assumptions):
            if ask(Q.imaginary(expr.exp), assumptions):
                return True
        # 如果 i = (exp 的参数)/(I*pi) 是整数或半整数倍的 I*pi，则 2*i 将是整数。
        # 此外，exp(i*I*pi) = (-1)**i 所以表达式的实数性质可以通过用 (-1)**i 替换 exp(i*I*pi) 来确定。
        i = expr.base.exp/I/pi
        if ask(Q.integer(2*i), assumptions):
            return ask(Q.real((S.NegativeOne**i)**expr.exp), assumptions)
        return
    # 如果询问表达式的基数是否虚部不为零，则进入条件判断
    if ask(Q.imaginary(expr.base), assumptions):
        # 如果询问表达式的指数是否为整数，则进入条件判断
        if ask(Q.integer(expr.exp), assumptions):
            # 判断指数是否为奇数
            odd = ask(Q.odd(expr.exp), assumptions)
            if odd is not None:
                # 返回奇数的否定值（偶数为True，奇数为False）
                return not odd
            # 如果无法确定奇偶性，则返回None
            return

    # 如果询问表达式的指数是否虚部不为零，则进入条件判断
    if ask(Q.imaginary(expr.exp), assumptions):
        # 如果询问对数基数是否虚部不为零，则进入条件判断
        imlog = ask(Q.imaginary(log(expr.base)), assumptions)
        if imlog is not None:
            # 返回对数基数是否为虚数的结果
            # I**i -> real, log(I) is imag;
            # (2*I)**i -> complex, log(2*I) is not imag
            return imlog

    # 如果询问表达式的基数是否为实数，则进入条件判断
    if ask(Q.real(expr.base), assumptions):
        # 如果询问表达式的指数是否为实数，则进入条件判断
        if ask(Q.real(expr.exp), assumptions):
            # 如果表达式的指数为有理数且分母为偶数，则进入条件判断
            if expr.exp.is_Rational and \
                    ask(Q.even(expr.exp.q), assumptions):
                # 返回基数是否为正数的结果
                return ask(Q.positive(expr.base), assumptions)
            # 如果表达式的指数为整数，则返回True
            elif ask(Q.integer(expr.exp), assumptions):
                return True
            # 如果基数为正数，则返回True
            elif ask(Q.positive(expr.base), assumptions):
                return True
            # 如果基数为负数，则返回False
            elif ask(Q.negative(expr.base), assumptions):
                return False
@RealPredicate.register_many(cos, sin)
def _(expr, assumptions):
    # 如果表达式的第一个参数是实数，返回True
    if ask(Q.real(expr.args[0]), assumptions):
            return True

@RealPredicate.register(exp)
def _(expr, assumptions):
    # 检查指数是否是整数除以π的结果或者是实数
    return ask(
        Q.integer(expr.exp/I/pi) | Q.real(expr.exp), assumptions
    )

@RealPredicate.register(log)
def _(expr, assumptions):
    # 检查参数是否为正数
    return ask(Q.positive(expr.args[0]), assumptions)

@RealPredicate.register_many(Determinant, MatrixElement, Trace)
def _(expr, assumptions):
    # 检查参数是否具有实元素
    return ask(Q.real_elements(expr.args[0]), assumptions)


# ExtendedRealPredicate

@ExtendedRealPredicate.register(object)
def _(expr, assumptions):
    # 检查对象是否为负无穷、负数、零、正数或正无穷
    return ask(Q.negative_infinite(expr)
               | Q.negative(expr)
               | Q.zero(expr)
               | Q.positive(expr)
               | Q.positive_infinite(expr),
            assumptions)

@ExtendedRealPredicate.register_many(Infinity, NegativeInfinity)
def _(expr, assumptions):
    # 对于无穷大和负无穷大，始终返回True
    return True

@ExtendedRealPredicate.register_many(Add, Mul, Pow) # type:ignore
def _(expr, assumptions):
    # 测试表达式是否属于扩展实数集合
    return test_closed_group(expr, assumptions, Q.extended_real)


# HermitianPredicate

@HermitianPredicate.register(object) # type:ignore
def _(expr, assumptions):
    # 如果表达式是矩阵基类的实例，则返回None；否则检查其是否为实数
    if isinstance(expr, MatrixBase):
        return None
    return ask(Q.real(expr), assumptions)

@HermitianPredicate.register(Add) # type:ignore
def _(expr, assumptions):
    """
    * Hermitian + Hermitian  -> Hermitian
    * Hermitian + !Hermitian -> !Hermitian
    """
    # 如果表达式是数字，则抛出未实现的错误；否则测试其是否属于埃尔米特群体
    if expr.is_number:
        raise MDNotImplementedError
    return test_closed_group(expr, assumptions, Q.hermitian)

@HermitianPredicate.register(Mul) # type:ignore
def _(expr, assumptions):
    """
    As long as there is at most only one noncommutative term:

    * Hermitian*Hermitian         -> Hermitian
    * Hermitian*Antihermitian     -> !Hermitian
    * Antihermitian*Antihermitian -> Hermitian
    """
    # 如果表达式是数字，则抛出未实现的错误；否则检查乘积的性质以确定其是否为埃尔米特
    if expr.is_number:
        raise MDNotImplementedError
    nccount = 0
    result = True
    for arg in expr.args:
        if ask(Q.antihermitian(arg), assumptions):
            result = result ^ True
        elif not ask(Q.hermitian(arg), assumptions):
            break
        if ask(~Q.commutative(arg), assumptions):
            nccount += 1
            if nccount > 1:
                break
    else:
        return result

@HermitianPredicate.register(Pow) # type:ignore
def _(expr, assumptions):
    """
    * Hermitian**Integer -> Hermitian
    """
    # 如果表达式是数字，则抛出未实现的错误；否则检查幂运算是否为埃尔米特
    if expr.is_number:
        raise MDNotImplementedError
    if expr.base == E:
        if ask(Q.hermitian(expr.exp), assumptions):
            return True
        raise MDNotImplementedError
    if ask(Q.hermitian(expr.base), assumptions):
        if ask(Q.integer(expr.exp), assumptions):
            return True
    raise MDNotImplementedError

@HermitianPredicate.register_many(cos, sin) # type:ignore
def _(expr, assumptions):
    # 检查三角函数的参数是否为埃尔米特
    if ask(Q.hermitian(expr.args[0]), assumptions):
        return True
    raise MDNotImplementedError
@HermitianPredicate.register(exp) # type:ignore
def _(expr, assumptions):
    # 如果表达式 expr.exp 是 Hermitean 的，返回 True
    if ask(Q.hermitian(expr.exp), assumptions):
        return True
    # 如果不是 Hermitean 的，抛出未实现错误
    raise MDNotImplementedError

@HermitianPredicate.register(MatrixBase) # type:ignore
def _(mat, assumptions):
    # 获取矩阵的行数和列数
    rows, cols = mat.shape
    ret_val = True
    # 遍历矩阵的上三角部分
    for i in range(rows):
        for j in range(i, cols):
            # 检查矩阵元素是否满足共轭对称性条件
            cond = fuzzy_bool(Eq(mat[i, j], conjugate(mat[j, i])))
            # 如果无法确定条件，ret_val 设为 None
            if cond is None:
                ret_val = None
            # 如果条件不成立，返回 False
            if cond == False:
                return False
    # 如果 ret_val 是 None，则抛出未实现错误
    if ret_val is None:
        raise MDNotImplementedError
    # 返回最终结果
    return ret_val


# ComplexPredicate

@ComplexPredicate.register_many(Abs, cos, exp, im, ImaginaryUnit, log, Number, # type:ignore
    NumberSymbol, re, sin)
def _(expr, assumptions):
    # 对于给定的表达式，始终返回 True
    return True

@ComplexPredicate.register_many(Infinity, NegativeInfinity) # type:ignore
def _(expr, assumptions):
    # 对于无穷大和负无穷大，始终返回 False
    return False

@ComplexPredicate.register(Expr) # type:ignore
def _(expr, assumptions):
    # 检查表达式是否为复数类型
    ret = expr.is_complex
    # 如果无法确定，抛出未实现错误
    if ret is None:
        raise MDNotImplementedError
    # 返回检查结果
    return ret

@ComplexPredicate.register_many(Add, Mul) # type:ignore
def _(expr, assumptions):
    # 检查加法和乘法运算是否封闭在复数集合中
    return test_closed_group(expr, assumptions, Q.complex)

@ComplexPredicate.register(Pow) # type:ignore
def _(expr, assumptions):
    # 对于指数运算，如果底数是自然常数 e，则返回 True
    if expr.base == E:
        return True
    # 否则检查是否封闭在复数集合中
    return test_closed_group(expr, assumptions, Q.complex)

@ComplexPredicate.register_many(Determinant, MatrixElement, Trace) # type:ignore
def _(expr, assumptions):
    # 检查矩阵行列式、矩阵元素、迹的元素是否属于复数集合
    return ask(Q.complex_elements(expr.args[0]), assumptions)

@ComplexPredicate.register(NaN) # type:ignore
def _(expr, assumptions):
    # 对于 NaN，返回 None
    return None


# ImaginaryPredicate

def _Imaginary_number(expr, assumptions):
    # 先尝试将表达式视为实部和虚部的形式来评估
    r = expr.as_real_imag()[0].evalf(2)
    # 如果评估结果的精度不为 1，则返回其相反值（即是否为虚数）
    if r._prec != 1:
        return not r
    # 否则，允许返回 None，表示无法确定是否为 0

@ImaginaryPredicate.register(ImaginaryUnit) # type:ignore
def _(expr, assumptions):
    # 对于虚数单位，始终返回 True
    return True

@ImaginaryPredicate.register(Expr) # type:ignore
def _(expr, assumptions):
    # 检查表达式是否为虚数
    ret = expr.is_imaginary
    # 如果无法确定，抛出未实现错误
    if ret is None:
        raise MDNotImplementedError
    # 返回检查结果
    return ret

@ImaginaryPredicate.register(Add) # type:ignore
def _(expr, assumptions):
    """
    * Imaginary + Imaginary -> Imaginary
    * Imaginary + Complex   -> ?
    * Imaginary + Real      -> !Imaginary
    """
    # 如果表达式是数字，尝试评估是否为虚数
    if expr.is_number:
        return _Imaginary_number(expr, assumptions)

    reals = 0
    # 遍历表达式的所有参数
    for arg in expr.args:
        # 如果参数被认为是虚数，继续
        if ask(Q.imaginary(arg), assumptions):
            pass
        # 如果参数被认为是实数，增加计数
        elif ask(Q.real(arg), assumptions):
            reals += 1
        # 如果有非虚数和非实数参数，跳出循环
        else:
            break
    else:
        # 如果没有实数参数，则返回 True
        if reals == 0:
            return True
        # 如果有一个或所有参数为实数，则返回 False
        if reals in (1, len(expr.args)):
            # 两个实数可能相加为 0，从而产生虚数
            return False

@ImaginaryPredicate.register(Mul) # type:ignore
# 定义一个函数 '_'，用于根据给定的表达式和假设判断其是否为虚数
def _(expr, assumptions):
    """
    * Real*Imaginary      -> Imaginary
    * Imaginary*Imaginary -> Real
    """
    # 如果表达式是一个数值，返回根据虚数判断的结果
    if expr.is_number:
        return _Imaginary_number(expr, assumptions)
    
    # 初始化结果为 False，实数计数为 0
    result = False
    reals = 0
    
    # 遍历表达式中的每个参数
    for arg in expr.args:
        # 如果参数被判定为虚数，则结果异或操作为 True
        if ask(Q.imaginary(arg), assumptions):
            result = result ^ True
        # 如果参数不是实数，则中断循环
        elif not ask(Q.real(arg), assumptions):
            break
        # 统计实数参数个数
        else:
            reals += 1
    else:
        # 如果所有参数都是实数，则根据异或结果返回最终结果
        if reals == len(expr.args):
            return False
        return result

@ImaginaryPredicate.register(Pow) # type:ignore
def _(expr, assumptions):
    """
    * Imaginary**Odd        -> Imaginary
    * Imaginary**Even       -> Real
    * b**Imaginary          -> !Imaginary if exponent is an integer
                               multiple of I*pi/log(b)
    * Imaginary**Real       -> ?
    * Positive**Real        -> Real
    * Negative**Integer     -> Real
    * Negative**(Integer/2) -> Imaginary
    * Negative**Real        -> not Imaginary if exponent is not Rational
    """
    # 如果表达式是一个数值，返回根据虚数判断的结果
    if expr.is_number:
        return _Imaginary_number(expr, assumptions)

    # 处理基数为自然对数 e 的情况
    if expr.base == E:
        a = expr.exp/I/pi
        return ask(Q.integer(2*a) & ~Q.integer(a), assumptions)

    # 处理基数为 e 的幂函数或者指数函数的情况
    if expr.base.func == exp or (expr.base.is_Pow and expr.base.base == E):
        if ask(Q.imaginary(expr.base.exp), assumptions):
            if ask(Q.imaginary(expr.exp), assumptions):
                return False
            i = expr.base.exp/I/pi
            if ask(Q.integer(2*i), assumptions):
                return ask(Q.imaginary((S.NegativeOne**i)**expr.exp), assumptions)

    # 处理基数为虚数的情况
    if ask(Q.imaginary(expr.base), assumptions):
        if ask(Q.integer(expr.exp), assumptions):
            odd = ask(Q.odd(expr.exp), assumptions)
            if odd is not None:
                return odd
            return

    # 处理指数为虚数的情况
    if ask(Q.imaginary(expr.exp), assumptions):
        imlog = ask(Q.imaginary(log(expr.base)), assumptions)
        if imlog is not None:
            # I**i -> real; (2*I)**i -> complex ==> not imaginary
            return False

    # 处理基数和指数都为实数的情况
    if ask(Q.real(expr.base) & Q.real(expr.exp), assumptions):
        if ask(Q.positive(expr.base), assumptions):
            return False
        else:
            rat = ask(Q.rational(expr.exp), assumptions)
            if not rat:
                return rat
            if ask(Q.integer(expr.exp), assumptions):
                return False
            else:
                half = ask(Q.integer(2*expr.exp), assumptions)
                if half:
                    return ask(Q.negative(expr.base), assumptions)
                return half

@ImaginaryPredicate.register(log) # type:ignore
def _(expr, assumptions):
    # 如果参数是实数，则判断它是否为正数，如果是则返回 False
    if ask(Q.real(expr.args[0]), assumptions):
        if ask(Q.positive(expr.args[0]), assumptions):
            return False
        return
    # 如果参数不是实数，则返回空
    # XXX it should be enough to do
    # return ask(Q.nonpositive(expr.args[0]), assumptions)
    # but ask(Q.nonpositive(exp(x)), Q.imaginary(x)) -> None;
    # 检查条件：如果表达式的第一个参数是指数函数 exp，或者是以自然常数 e 为底的幂运算
    if expr.args[0].func == exp or (expr.args[0].is_Pow and expr.args[0].base == E):
        # 进一步检查：如果指数部分是虚数单位 i 或者 -i，则返回 True
        if expr.args[0].exp in [I, -I]:
            return True
    # 检查表达式的第一个参数是否具有虚部，如果没有虚部则返回 False
    im = ask(Q.imaginary(expr.args[0]), assumptions)
    if im is False:
        return False
# 注册 ImaginaryPredicate 的匿名函数，用于表达式和假设
@ImaginaryPredicate.register(exp) # type:ignore
def _(expr, assumptions):
    # 计算表达式的 exp 属性除以 π*i 后的结果
    a = expr.exp/I/pi
    # 返回关于 2*a 是整数且 a 不是整数的询问结果
    return ask(Q.integer(2*a) & ~Q.integer(a), assumptions)

# 注册 ImaginaryPredicate 的匿名函数，用于 Number 和 NumberSymbol
@ImaginaryPredicate.register_many(Number, NumberSymbol) # type:ignore
def _(expr, assumptions):
    # 返回表达式是否不是实数的判断结果
    return not (expr.as_real_imag()[1] == 0)

# 注册 ImaginaryPredicate 的匿名函数，用于 NaN 类型
@ImaginaryPredicate.register(NaN) # type:ignore
def _(expr, assumptions):
    # 返回空值
    return None


# AntihermitianPredicate

# 注册 AntihermitianPredicate 的匿名函数，用于任意对象
@AntihermitianPredicate.register(object) # type:ignore
def _(expr, assumptions):
    # 如果表达式是矩阵基类，则返回空值
    if isinstance(expr, MatrixBase):
        return None
    # 如果表达式为零，则返回 True
    if ask(Q.zero(expr), assumptions):
        return True
    # 否则，返回表达式是否是虚数的询问结果
    return ask(Q.imaginary(expr), assumptions)

# 注册 AntihermitianPredicate 的匿名函数，用于 Add 类型的表达式
@AntihermitianPredicate.register(Add) # type:ignore
def _(expr, assumptions):
    """
    * Antihermitian + Antihermitian  -> Antihermitian
    * Antihermitian + !Antihermitian -> !Antihermitian
    """
    # 如果表达式是数值，则抛出未实现错误
    if expr.is_number:
        raise MDNotImplementedError
    # 测试表达式是否在闭合群 Q.antihermitian 中
    return test_closed_group(expr, assumptions, Q.antihermitian)

# 注册 AntihermitianPredicate 的匿名函数，用于 Mul 类型的表达式
@AntihermitianPredicate.register(Mul) # type:ignore
def _(expr, assumptions):
    """
    As long as there is at most only one noncommutative term:

    * Hermitian*Hermitian         -> !Antihermitian
    * Hermitian*Antihermitian     -> Antihermitian
    * Antihermitian*Antihermitian -> !Antihermitian
    """
    # 如果表达式是数值，则抛出未实现错误
    if expr.is_number:
        raise MDNotImplementedError
    nccount = 0
    result = False
    for arg in expr.args:
        # 如果参数符合反厄米特条件，则取反结果
        if ask(Q.antihermitian(arg), assumptions):
            result = result ^ True
        # 如果参数不符合厄米特条件，则中断循环
        elif not ask(Q.hermitian(arg), assumptions):
            break
        # 如果参数是非交换的，则计数加一
        if ask(~Q.commutative(arg), assumptions):
            nccount += 1
            if nccount > 1:
                break
    else:
        return result

# 注册 AntihermitianPredicate 的匿名函数，用于 Pow 类型的表达式
@AntihermitianPredicate.register(Pow) # type:ignore
def _(expr, assumptions):
    """
    * Hermitian**Integer  -> !Antihermitian
    * Antihermitian**Even -> !Antihermitian
    * Antihermitian**Odd  -> Antihermitian
    """
    # 如果表达式是数值，则抛出未实现错误
    if expr.is_number:
        raise MDNotImplementedError
    if ask(Q.hermitian(expr.base), assumptions):
        if ask(Q.integer(expr.exp), assumptions):
            return False
    elif ask(Q.antihermitian(expr.base), assumptions):
        if ask(Q.even(expr.exp), assumptions):
            return False
        elif ask(Q.odd(expr.exp), assumptions):
            return True
    # 如果以上条件都不满足，则抛出未实现错误
    raise MDNotImplementedError

# 注册 AntihermitianPredicate 的匿名函数，用于 MatrixBase 类型的表达式
@AntihermitianPredicate.register(MatrixBase) # type:ignore
def _(mat, assumptions):
    # 获取矩阵的行数和列数
    rows, cols = mat.shape
    ret_val = True
    for i in range(rows):
        for j in range(i, cols):
            # 模糊布尔运算，判断矩阵元素是否满足厄米特共轭关系
            cond = fuzzy_bool(Eq(mat[i, j], -conjugate(mat[j, i])))
            if cond is None:
                ret_val = None
            if cond == False:
                return False
    # 如果存在模糊布尔结果为空，则抛出未实现错误
    if ret_val is None:
        raise MDNotImplementedError
    return ret_val


# AlgebraicPredicate

# 注册 AlgebraicPredicate 的匿名函数，用于多个特定类型的表达式
@AlgebraicPredicate.register_many(AlgebraicNumber, Float, GoldenRatio, # type:ignore
    ImaginaryUnit, TribonacciConstant)
# 注册一个代数谓词，接受一个表达式和假设集合，并总是返回True
def _(expr, assumptions):
    return True

# 注册多个代数谓词，用于特定的数学表达式，总是返回False
@AlgebraicPredicate.register_many(ComplexInfinity, Exp1, Infinity, # type:ignore
    NegativeInfinity, Pi)
def _(expr, assumptions):
    return False

# 注册多个代数谓词，适用于Add和Mul类型的表达式，调用test_closed_group函数检查是否封闭在代数集合内
@AlgebraicPredicate.register_many(Add, Mul) # type:ignore
def _(expr, assumptions):
    return test_closed_group(expr, assumptions, Q.algebraic)

# 注册一个代数谓词，适用于Pow类型的表达式，根据底数和指数的特性进行代数性质的检查
@AlgebraicPredicate.register(Pow) # type:ignore
def _(expr, assumptions):
    if expr.base == E:
        if ask(Q.algebraic(expr.exp), assumptions):
            return ask(~Q.nonzero(expr.exp), assumptions)
        return
    return expr.exp.is_Rational and ask(Q.algebraic(expr.base), assumptions)

# 注册一个代数谓词，适用于有理数类型的表达式，检查分母是否为非零
@AlgebraicPredicate.register(Rational) # type:ignore
def _(expr, assumptions):
    return expr.q != 0

# 注册多个代数谓词，适用于asin、atan、cos、sin、tan类型的表达式，检查参数是否代数数且非零
@AlgebraicPredicate.register_many(asin, atan, cos, sin, tan) # type:ignore
def _(expr, assumptions):
    x = expr.args[0]
    if ask(Q.algebraic(x), assumptions):
        return ask(~Q.nonzero(x), assumptions)

# 注册一个代数谓词，适用于exp类型的表达式，检查指数是否代数且非零
@AlgebraicPredicate.register(exp) # type:ignore
def _(expr, assumptions):
    x = expr.exp
    if ask(Q.algebraic(x), assumptions):
        return ask(~Q.nonzero(x), assumptions)

# 注册多个代数谓词，适用于acot、cot类型的表达式，检查参数是否代数，总是返回False
@AlgebraicPredicate.register_many(acot, cot) # type:ignore
def _(expr, assumptions):
    x = expr.args[0]
    if ask(Q.algebraic(x), assumptions):
        return False

# 注册多个代数谓词，适用于acos、log类型的表达式，检查参数是否代数且不等于1
@AlgebraicPredicate.register_many(acos, log) # type:ignore
def _(expr, assumptions):
    x = expr.args[0]
    if ask(Q.algebraic(x), assumptions):
        return ask(~Q.nonzero(x - 1), assumptions)
```