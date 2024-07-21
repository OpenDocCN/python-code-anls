# `.\pytorch\torch\utils\_sympy\functions.py`

```
# mypy: allow-untyped-defs
# 导入所需的模块和库
import functools  # 导入 functools 模块
import math  # 导入 math 模块
import operator  # 导入 operator 模块
import sys  # 导入 sys 模块

import sympy  # 导入 sympy 库
from sympy import S  # 从 sympy 中导入 S 符号

from .numbers import int_oo  # 从当前包中导入 int_oo

__all__ = [  # 定义导出的模块列表
    "FloorDiv",  # Python 风格的 floor division 类
    "ModularIndexing",
    "CleanDiv",
    "CeilDiv",
    "IntTrueDiv",
    "FloatTrueDiv",
    "LShift",
    "RShift",
    "IsNonOverlappingAndDenseIndicator",
    "RoundToInt",
    "RoundDecimal",
    "ToFloat",
    "FloatPow",
    "PowByNatural",
    "Identity",
]


def _keep_float(f):
    # 保留浮点数功能的装饰器函数
    @functools.wraps(f)
    def inner(*args):
        r = f(*args)
        # 如果参数中包含 sympy.Float 类型且结果不是 sympy.Float 类型，则转换为浮点数类型
        if any(isinstance(a, sympy.Float) for a in args) and not isinstance(
            r, sympy.Float
        ):
            r = sympy.Float(float(r))
        return r

    return inner


def fuzzy_eq(x, y):
    # 模糊相等性判断函数，如果其中一个参数为 None 则返回 None，否则返回 x == y 的比较结果
    if None in (x, y):
        return None
    return x == y


# TODO: In Triton, // rounds to zero, but in Python, it is floor division.
# When we can prove both arguments are non-negative, we should just have a
# GenericFloorDiv (name pending) which can codegen efficiently in Python/C,
# and then PythonFloorDiv and CIntDiv which have the appropriate rounding
# semantics.
#
# Right now, FloorDiv de facto changes behavior if arguments are negative or
# not, this can potentially cause correctness issues.
class FloorDiv(sympy.Function):
    """
    Python-style floor division, round to -Inf.

    We maintain this so that:
    1. We can use divisibility guards to simplify FloorDiv(a, b) to a / b.
    2. Printing out the expression is nicer (compared to say, representing a//b as (a - a % b) / b)

    NB: This is Python-style floor division, round to -Inf
    """

    nargs = (2,)  # 函数参数数目为两个

    # 优先级，与乘法相同
    precedence = 50  # precedence of mul  # noqa: F811

    is_integer = True  # 是整数类型的运算

    @property
    def base(self):
        return self.args[0]  # 返回第一个参数

    @property
    def divisor(self):
        return self.args[1]  # 返回第二个参数

    def _sympystr(self, printer):
        # 返回用于打印的字符串表示形式
        base = printer.parenthesize(self.base, self.precedence)
        divisor = printer.parenthesize(self.divisor, self.precedence)
        return f"({base}//{divisor})"  # 返回 Python 风格的 floor division 表示形式

    # 自动求值
    @classmethod
    # 定义一个静态方法 `eval`，接受三个参数：cls, base 和 divisor
    def eval(cls, base, divisor):
        # 对特定测试进行评估，可能涉及维度约束的解决
        # 在不等式求解器触发断言时
        # assert base.is_integer, base
        # assert divisor.is_integer, divisor

        # 由于 SymPy 难以检查类型，我们没有提供与 Python 相同的错误消息。
        
        # 如果除数为零，则抛出 ZeroDivisionError 异常
        if divisor.is_zero:
            raise ZeroDivisionError("division by zero")
        
        # 如果 base 或 divisor 是无穷大或无穷小，则返回 NaN
        if base in (int_oo, -int_oo, sympy.oo, -sympy.oo) and divisor in (
            int_oo,
            -int_oo,
            sympy.oo,
            -sympy.oo,
        ):
            return sympy.nan
        
        # 如果 base 或 divisor 是 NaN，则返回 NaN
        if base is sympy.nan or divisor is sympy.nan:
            return sympy.nan

        # 如果 base 是零，则返回 sympy.S.Zero
        if base.is_zero:
            return sympy.S.Zero
        
        # 如果 base 是整数且 divisor 等于 1，则返回 base
        if base.is_integer and divisor == 1:
            return base
        
        # 如果 base 是整数且 divisor 等于 -1，则返回 base 乘以 -1
        if base.is_integer and divisor == -1:
            return sympy.Mul(base, -1)
        
        # 如果 base 和 divisor 都是 sympy.Number 类型，并且其中之一是无穷大或无穷小
        # 则计算浮点数除法结果并返回相应的符号
        if (
            isinstance(base, sympy.Number)
            and isinstance(divisor, sympy.Number)
            and (
                base in (int_oo, -int_oo, sympy.oo, -sympy.oo)
                or divisor in (int_oo, -int_oo, sympy.oo, -sympy.oo)
            )
        ):
            r = float(base) / float(divisor)
            if r == math.inf:
                return int_oo
            elif r == -math.inf:
                return -int_oo
            elif math.isnan(r):
                return sympy.nan
            else:
                return sympy.Integer(math.floor(r))
        
        # 如果 base 和 divisor 都是 sympy.Integer 类型，则返回整数除法结果
        if isinstance(base, sympy.Integer) and isinstance(divisor, sympy.Integer):
            return sympy.Integer(int(base) // int(divisor))
        
        # 如果 base 是 FloorDiv 类型，则返回重新计算后的 FloorDiv 对象
        if isinstance(base, FloorDiv):
            return FloorDiv(base.args[0], base.args[1] * divisor)

        # 将 base 拆分为加法项，并尝试对每个项进行整数除法，然后返回 FloorDiv 和除法的和
        # 仅在 floor 是一个恒等式时有效，即 x / b 是一个整数时。
        for term in sympy.Add.make_args(base):
            quotient = term / divisor
            if quotient.is_integer and isinstance(divisor, sympy.Integer):
                # 注意：即使 divisor 不是整数，这里也是正确的，但会创建带有动态形状问题的有理数表达式。
                return FloorDiv(base - term, divisor) + quotient

        # 尝试计算 base 和 divisor 的最大公约数，并返回简化后的 FloorDiv 对象
        # 如果计算过程中出现 PolynomialError，则忽略异常
        try:
            gcd = sympy.gcd(base, divisor)
            if gcd != 1:
                return FloorDiv(
                    sympy.simplify(base / gcd), sympy.simplify(divisor / gcd)
                )
        except sympy.PolynomialError:
            pass  # https://github.com/pytorch/pytorch/issues/108276
# 定义一个继承自 sympy.Function 的类 ModularIndexing，表示模数索引操作
class ModularIndexing(sympy.Function):
    """
    ModularIndexing(a, b, c) => (a // b) % c where % is the C modulus
    """

    # 设置 nargs 属性为 (3,)，表明此函数接受三个参数
    nargs = (3,)
    # 设置 is_integer 属性为 True，表明函数返回的结果是整数

    @classmethod
    def eval(cls, base, divisor, modulus):
        # 如果 base 为 0 或 modulus 为 1，则返回整数 0
        if base == 0 or modulus == 1:
            return sympy.Integer(0)

        # 检查 base、divisor、modulus 是否都为 sympy.Integer 类型
        if (
            isinstance(base, sympy.Integer)
            and isinstance(divisor, sympy.Integer)
            and isinstance(modulus, sympy.Integer)
        ):
            # 返回 (base // divisor) % modulus 的结果
            return (base // divisor) % modulus

        try:
            # 尝试计算 base 和 divisor 的最大公约数 gcd
            if divisor != 1:
                gcd = sympy.gcd(base, divisor)
                # 如果 gcd 不为 1，则进行模数索引操作
                if gcd != 1:
                    return ModularIndexing(
                        sympy.simplify(base / gcd),
                        sympy.simplify(divisor / gcd),
                        modulus,
                    )
        except sympy.PolynomialError:
            pass  # 捕获异常，链接提到了一个问题，暂时忽略

        # 如果 base 是 sympy.Add 类型
        if isinstance(base, sympy.Add):
            new_terms = []
            all_positive = True
            # 遍历 base 的每个 term
            for term in base.args:
                # 如果 term 和 modulus * divisor 的最大公约数不等于 modulus * divisor
                if sympy.gcd(term, modulus * divisor) != modulus * divisor:
                    # 如果 term 是负数或者第一个因子是负数的乘积
                    if (isinstance(term, sympy.Integer) and term < 0) or (
                        isinstance(term, sympy.Mul)
                        and isinstance(term.args[0], sympy.Integer)
                        and term.args[0] < 0
                    ):
                        # 对于负数 term，暂时无法使用 // 操作，返回 None
                        all_positive = False
                        break
                    else:
                        # 将非负数 term 添加到 new_terms 中
                        new_terms.append(term)

            # 如果 new_terms 的长度不等于 base.args 的长度，并且 all_positive 为 True
            if len(new_terms) != len(base.args) and all_positive:
                # 返回对 new_terms 求和后的模数索引操作结果
                return ModularIndexing(sum(new_terms), divisor, modulus)

        # 如果 base 是 FloorDiv 类型的实例
        if isinstance(base, FloorDiv):
            # 对 base 的第一个参数进行模数索引操作
            return ModularIndexing(base.args[0], base.args[1] * divisor, modulus)

    def _eval_is_nonnegative(self):
        # 获取函数的前两个参数
        p, q = self.args[:2]
        # 判断 p 和 q 是否具有非负属性，返回模糊相等的结果
        return fuzzy_eq(p.is_nonnegative, q.is_nonnegative)  # type: ignore[attr-defined]

    def _eval_is_positive(self):
        # 获取函数的前两个参数
        p, q = self.args[:2]
        # 判断 p 和 q 是否具有正数属性，返回模糊相等的结果
        return fuzzy_eq(p.is_positive, q.is_positive)  # type: ignore[attr-defined]


# 定义一个继承自 sympy.Function 的类 Where，表示三元运算符
class Where(sympy.Function):
    """
    Good ol' ternary operator
    """

    # 设置 nargs 属性为 (3,)，表明此函数接受三个参数
    nargs = (3,)

    def _eval_is_integer(self):
        # 判断第二个和第三个参数是否都为整数，返回布尔值或 None
        return True if self.args[1].is_integer and self.args[2].is_integer else None  # type: ignore[attr-defined]

    def _eval_is_nonnegative(self):
        # 判断第二个和第三个参数是否都为非负数，返回布尔值或 None
        return (
            True
            if self.args[1].is_nonnegative and self.args[2].is_nonnegative  # type: ignore[attr-defined]
            else None
        )
    # 定义一个方法 `_eval_is_positive`，用于评估当前对象是否为正数
    def _eval_is_positive(self):
        # 返回 True，如果第二个和第三个参数都被定义为正数；否则返回 None
        return True if self.args[1].is_positive and self.args[2].is_positive else None  # type: ignore[attr-defined]

    # 定义一个类方法 `eval`，用于条件求值
    @classmethod
    def eval(cls, c, p, q):
        # 如果条件 `c` 等于 sympy.true，则返回 `p`
        if c == sympy.true:
            return p
        # 如果条件 `c` 等于 sympy.false，则返回 `q`
        elif c == sympy.false:
            return q
# Python 风格的模运算：从右侧取符号
class PythonMod(sympy.Function):
    nargs = (2,)  # 定义此函数的参数数量为两个

    is_integer = True  # 声明此函数返回整数类型的结果

    @classmethod
    def eval(cls, p, q):
        # 对于给定的两个参数 p 和 q，评估模运算的结果
        # 触发于 sympy.solvers.inequalities.reduce_inequalities
        # assert p.is_integer, p
        # assert q.is_integer, q

        if q.is_zero:
            raise ZeroDivisionError("Modulo by zero")  # 如果 q 为零，则抛出除零错误

        # 三种情况：
        #   1. p == 0
        #   2. p 是 q 或者 -q
        #   3. p 是整数且 q == 1
        if p is S.Zero or p in (q, -q) or q == 1:
            return S.Zero  # 返回零

        # 如果它们都是字面值，则进行求值
        if q.is_Number and p.is_Number:
            return p % q  # 返回 p 除以 q 的余数

        # 如果 q == 2，则取决于 p 是奇数还是偶数
        if q.is_Number and q == 2:
            if p.is_even:
                return S.Zero  # 如果 p 是偶数，返回零
            if p.is_odd:
                return S.One  # 如果 p 是奇数，返回一

        # 如果 p 是 q 的倍数
        r = p / q
        if r.is_integer:
            return S.Zero  # 返回零

        # 如果 p < q 并且其比值为正，则：
        #   - floor(p / q) = 0
        #   - p % q = p - floor(p / q) * q = p
        less = p < q
        if less.is_Boolean and bool(less) and r.is_positive:
            return p  # 返回 p

        if sympy.Mod(p, q) == 0:
            return S.Zero  # 返回零

    # 注意：args[1] 用于 PythonMod
    def _eval_is_nonnegative(self):
        return True if self.args[1].is_positive else None  # 返回真，如果第二个参数是正数，否则返回空（类型提示：忽略属性定义）

    def _eval_is_nonpositive(self):
        return True if self.args[1].is_negative else None  # 返回真，如果第二个参数是负数，否则返回空（类型提示：忽略属性定义）

# 通用模运算：仅在非负参数上定义
class Mod(sympy.Function):
    nargs = (2,)  # 定义此函数的参数数量为两个

    is_integer = True  # 声明此函数返回整数类型的结果
    is_nonnegative = True  # 声明此函数返回非负结果

    @classmethod
    # 定义一个类方法 `eval`，用于计算模运算
    def eval(cls, p, q):
        # 该方法的源自：sympy/core/mod.py

        # 在以下情况下被触发：
        # python test/test_dynamic_shapes.py -k TestDimConstraints.test_dim_constraints_solve_full
        # 断言 p 是整数，如果不是则抛出异常
        # 断言 q 是整数，如果不是则抛出异常
        if q.is_zero:
            # 如果 q 为零，则抛出 ZeroDivisionError 异常
            raise ZeroDivisionError("Modulo by zero")

        # 三种情况：
        #   1. p == 0
        #   2. p 是 q 或者 -q
        #   3. p 是整数且 q == 1
        if p is S.Zero or p in (q, -q) or q == 1:
            # 返回零值 S.Zero
            return S.Zero

        # 如果 p 和 q 都是数字字面值，则计算 p 对 q 的模
        if q.is_Number and p.is_Number:
            # 断言 p 大于等于 0，如果不是则抛出异常
            assert p >= 0, p
            # 断言 q 大于等于 1，如果不是则抛出异常
            assert q >= 1, q
            # 返回 p 对 q 的模运算结果
            return p % q

        # 如果 q 是数字且等于 2，则根据 p 的奇偶性返回 S.Zero 或 S.One
        if q.is_Number and q == 2:
            # 如果 p 是偶数，返回 S.Zero
            if p.is_even:
                return S.Zero
            # 如果 p 是奇数，返回 S.One
            if p.is_odd:
                return S.One

        # 如果 p 是 q 的倍数，则返回 S.Zero
        r = p / q
        if r.is_integer:
            return S.Zero

        # 如果 p 小于 q 且其比率为正，则返回 p
        less = p < q
        if less.is_Boolean and bool(less) and r.is_positive:
            return p
class CleanDiv(FloorDiv):
    """
    Div where we can assume no rounding.
    This is to enable future optimizations.
    """
    pass



# 不进行任何向上或向下取整操作的除法，用于未来的优化
class CeilToInt(sympy.Function):
    is_integer = True

    @classmethod
    def eval(cls, number):
        # 如果参数为正无穷或负无穷，返回对应的整数无穷
        if number in (sympy.oo, int_oo):
            return int_oo
        if number in (-sympy.oo, -int_oo):
            return -int_oo
        if isinstance(number, sympy.Number):
            # 使用向上取整函数，将浮点数转换为整数
            return sympy.Integer(math.ceil(float(number)))



# 不进行任何向上或向下取整操作的除法，用于未来的优化
class FloorToInt(sympy.Function):
    is_integer = True

    @classmethod
    def eval(cls, number):
        # 如果参数为正无穷或负无穷，返回对应的整数无穷
        if number in (sympy.oo, int_oo):
            return int_oo
        if number in (-sympy.oo, int_oo):
            return -int_oo
        if isinstance(number, sympy.Number):
            # 使用向下取整函数，将浮点数转换为整数
            return sympy.Integer(math.floor(float(number)))



class CeilDiv(sympy.Function):
    """
    Div used in indexing that rounds up.
    """
    is_integer = True

    def __new__(cls, base, divisor):
        base = sympy.sympify(base)
        divisor = sympy.sympify(divisor)
        # 如果 base 和 divisor 的最大公约数等于 divisor，则返回 CleanDiv 对象
        if sympy.gcd(base, divisor) == divisor:
            return CleanDiv(base, divisor)
        else:
            # 否则返回 FloorDiv 对象，计算 base + (divisor - 1) 除以 divisor 的结果
            return FloorDiv(base + (divisor - 1), divisor)



class LShift(sympy.Function):
    is_integer = True

    @classmethod
    def eval(cls, base, shift):
        # 如果 shift 小于 0，则抛出 ValueError 异常
        if shift < 0:
            raise ValueError("negative shift count")
        # 返回左移运算的结果，base 左移 shift 位
        return base * 2**shift



class RShift(sympy.Function):
    is_integer = True

    @classmethod
    def eval(cls, base, shift):
        # 如果 shift 小于 0，则抛出 ValueError 异常
        if shift < 0:
            raise ValueError("negative shift count")
        # 返回右移运算的结果，base 右移 shift 位
        return base // 2**shift



def safe_pow(base, exp):
    sign = 1
    # 如果 base 小于 0，则将 base 变为其绝对值，并根据 exp 的奇偶性确定 sign 的值
    if base < 0:
        base = -base
        sign = 1 if exp % 2 == 0 else -1
    # 调用 _safe_pow 函数计算 base 的 exp 次方，并考虑符号
    return sign * _safe_pow(base, exp)



# 防止溢出的幂函数
def _safe_pow(base, exponent):
    if exponent < 0:
        raise ValueError("Exponent must be non-negative.")

    if exponent == 0:
        return 1

    half_exp = safe_pow(base, exponent // 2)
    # 如果 half_exp 为正无穷，则返回正无穷
    if half_exp is int_oo:
        return int_oo

    # TODO: microoptimization is to avoid overflowing into arbitrary precision
    # and detect overflow prior to doing operations

    # 计算 base 的 exponent 次方，避免溢出
    result = half_exp * half_exp
    if result > sys.maxsize:
        return int_oo

    if exponent % 2 == 1:
        result *= base
        if result > sys.maxsize:
            return int_oo

    return result



class PowByNatural(sympy.Function):
    is_integer = True

    @classmethod
    # 此处需要继续添加注释
    # 定义一个类方法 eval，用于计算数学表达式的值，支持 sympy 的整数和无穷大类型
    def eval(cls, base, exp):
        # 检查 base 和 exp 是否都是 sympy.Integer 类型的整数
        if isinstance(base, sympy.Integer) and isinstance(exp, sympy.Integer):
            # 调用 safe_pow 函数计算 base 的 exp 次幂，处理安全性和边界情况
            r = safe_pow(base, exp)
            # 如果计算结果是负无穷或正无穷，则直接返回该值
            if r in (-int_oo, int_oo):
                return r
            # 否则将结果转换为 sympy.Integer 类型并返回
            return sympy.Integer(r)
        
        # 如果 exp 是 sympy.Integer 类型的整数，则使用 sympy.Pow 计算 base 的 exp 次幂
        if isinstance(exp, sympy.Integer):
            # 返回 sympy.Pow 对象，用于表示 base 的 exp 次幂
            return sympy.Pow(base, exp)
        
        # 如果 exp 是无穷大（int_oo 或 sympy.oo）
        if exp in (int_oo, sympy.oo):
            # 如果 base 是非负数，则返回正无穷 int_oo
            if base.is_nonnegative:
                return int_oo
            # 如果 base 是负数，则返回 sympy.zoo，用于表示负无穷
            elif base.is_negative:
                return sympy.zoo  # this is apparently what (-2)**sympy.oo does
        
        # 注意：不要将 exp 转换为 sympy.Pow，否则会丢失 exp 是自然数的信息
# base is assumed to be nonnegative, thereby prevent complex numbers from
# occuring
class FloatPow(sympy.Function):
    is_integer = False  # 类属性：指示该函数不处理整数
    is_real = True      # 类属性：指示该函数处理实数

    @classmethod
    def eval(cls, base, exp):
        # NB: These test sympy.Number, not sympy.Float, because:
        #   - Sometimes we may have sympy.oo or int_oo, and that's not a Float
        #     (but coerces to math.Inf)
        #   - Sometimes Float(0.0) will unpredictably decay to Integer(0),
        #     but we should still accept it in floatey contexts
        if isinstance(base, sympy.Number) and isinstance(exp, sympy.Number):
            # 若 base 和 exp 均为 sympy.Number 类型，则计算它们的浮点数幂
            return sympy.Float(float(base) ** float(exp))
        # NB: do not do any nontrivial reasoning


# Overloaded to be compatible with regular Python.
# https://github.com/pytorch/pytorch/issues/90900
#
# In particular, sympy division is willing to simplify x/x == 1
# where 1 is an integer, but this must be a float if x was float.
class FloatTrueDiv(sympy.Function):
    is_integer = False  # 类属性：指示该函数不处理整数
    is_real = True      # 类属性：指示该函数处理实数

    @classmethod
    def eval(cls, base, divisor):
        # assert base.is_integer is not True, base
        # assert divisor.is_integer is not True, divisor

        if divisor.is_zero:
            raise ZeroDivisionError("division by zero")

        if isinstance(base, sympy.Number) and isinstance(divisor, sympy.Number):
            # 若 base 和 divisor 均为 sympy.Number 类型，则计算它们的浮点数除法
            return sympy.Float(float(base) / float(divisor))


# Overloaded to be compatible with regular Python.  We distinguish this from
# FloatTrueDiv, because the code generation has to be different for this case:
# Python has a fancy algorithm for integer true division that isn't just
# "promote both arguments to float and use float division", so you need to
# codegen it differently.  While technically you can work it out from the
# types of the input, this is often inconvenient to do in Inductor codegen,
# so just have a different operator
# NB: Right now, Inductor codegen doesn't implement this correctly lol
class IntTrueDiv(sympy.Function):
    is_integer = False  # 类属性：指示该函数不处理整数
    is_real = True      # 类属性：指示该函数处理实数

    @classmethod
    def eval(cls, base, divisor):
        if divisor.is_zero:
            raise ZeroDivisionError("division by zero")

        if (
            isinstance(base, sympy.Number)
            and isinstance(divisor, sympy.Number)
            and (
                base in (int_oo, -int_oo, sympy.oo, -sympy.oo)
                or divisor in (int_oo, -int_oo, sympy.oo, -sympy.oo)
            )
        ):
            # 若 base 或 divisor 为特定的数值（如无穷大），则计算它们的浮点数除法
            # 不必担心精度问题，结果为零或无穷大
            return sympy.Float(float(base) / float(divisor))
        if isinstance(base, sympy.Integer) and isinstance(divisor, sympy.Integer):
            # 若 base 和 divisor 均为整数类型，则计算它们的浮点数除法
            return sympy.Float(int(base) / int(divisor))


# TODO: As an indicator, this != 0 implies == 1 (and vice versa).
# Because we do not have the ability to guard on the stride permutation
# 在当前情况下，很难进一步推断此条件为真的含义，
# 尽管我们知道张量在某种布局下是连续的，但我们不知道具体是哪种布局
# （然而，例如，你可以推断将其重塑为一维张量是无障碍的。）
class IsNonOverlappingAndDenseIndicator(sympy.Function):
    # 声明此类的实例是整数
    is_integer = True

    # 类方法 eval 用于评估函数的值
    @classmethod
    def eval(cls, *args):
        # 断言参数数量为偶数
        assert len(args) % 2 == 0
        dim = len(args) // 2
        sizes = args[0:dim]  # 尺寸参数
        strides = args[dim:]  # 步幅参数

        # 导入 torch.__init__.py 中的 sym_node，避免导入循环
        from torch.fx.experimental.symbolic_shapes import (
            eval_is_non_overlapping_and_dense,
        )

        # 如果所有参数都是 sympy.Integer 类型
        if all(isinstance(a, sympy.Integer) for a in args):
            # 调用 eval_is_non_overlapping_and_dense 函数计算结果
            return eval_is_non_overlapping_and_dense(
                [int(a) for a in sizes], [int(a) for a in strides]
            )

        # 如果维度为 1
        if dim == 1:
            # 手动实现秩为一的短路
            if strides[0].is_Number and strides[0] == 1:
                return 1  # 返回 1

            if sizes[0].is_Number and sizes[0] < 2:
                return 1  # 返回 1

            # 0 的情况在上面的条件已经涵盖了

            # TODO: 无法访问大小无关性很糟糕：如果我们对大小类似但未支持的 SymInt 进行大小无关的测试，
            # 当我们有一个大小类似的 u0 步幅和一个大小类似的 u1 尺寸时，我们可以自信地返回零。
            # 或许对该函数进行高级的 ValueRanges 分析能帮助解决这个问题。

        # 如果所有步幅都是整数
        if all(isinstance(a, sympy.Integer) for a in strides):
            assert dim != 0
            # 当所有步幅都是整数时，可以对尺寸进行排序，最大步幅对应的尺寸是符号化的
            s_sizes, s_strides = zip(
                *sorted(zip(sizes, strides), key=operator.itemgetter(1))
            )
            # 在最大尺寸位置放入任意值，它将被忽略
            if all(isinstance(a, sympy.Integer) for a in s_sizes[:-1]):
                s_sizes = s_sizes[:-1] + (42,)
                # 可以重用常规的 eval，因为它对维度的排列是不变的
                return eval_is_non_overlapping_and_dense(
                    [int(a) for a in s_sizes], [int(a) for a in s_strides]
                )

        return None


# 注意：这与 Python 中的 math.trunc 不一致
class TruncToFloat(sympy.Function):
    # 声明此类的实例不是整数，是实数
    is_integer = False
    is_real = True

    # 类方法 eval 用于评估函数的值
    @classmethod
    def eval(cls, number):
        # 如果 number 是 sympy.Number 类型
        if isinstance(number, sympy.Number):
            # 注意：使用截断到整数是安全的，这就是 math.trunc 所做的，
            # 因为 Python 的整数是任意精度的，所以在这样做时不会失去精度
            return sympy.Float(math.trunc(float(number)))
# 定义一个自定义函数类 TruncToInt，继承自 sympy.Function
class TruncToInt(sympy.Function):
    # 表示该函数返回的结果是整数
    is_integer = True

    # 类方法 eval，用于计算函数的返回值
    @classmethod
    def eval(cls, number):
        # assert number.is_integer is not True, number
        # 如果 number 是正无穷或者 int_oo，则返回 int_oo
        if number in (sympy.oo, int_oo):
            return int_oo
        # 如果 number 是负无穷或者 -int_oo，则返回 -int_oo
        if number in (-sympy.oo, -int_oo):
            return -int_oo
        # 如果 number 是 sympy.Number 类型，则将其转换为 float，再取其整数部分，并返回对应的 sympy.Integer 类型
        if isinstance(number, sympy.Number):
            return sympy.Integer(math.trunc(float(number)))


# 定义一个自定义函数类 RoundToInt，继承自 sympy.Function
# 表示将浮点数四舍五入为整数的操作
class RoundToInt(sympy.Function):
    # 表示该函数返回的结果是整数
    is_integer = True

    # 类方法 eval，用于计算函数的返回值
    @classmethod
    def eval(cls, number):
        # assert number.is_integer is not True, number
        # 如果 number 是正无穷，则返回 int_oo
        if number is sympy.oo:
            return int_oo
        # 如果 number 是负无穷，则返回 -int_oo
        if number is -sympy.oo:
            return -int_oo
        # 如果 number 是 sympy.Number 类型，则将其转换为 float，然后使用 round 函数四舍五入到最接近的整数，并返回对应的 sympy.Integer 类型
        if isinstance(number, sympy.Number):
            return sympy.Integer(round(float(number), 0))


# 定义一个自定义函数类 RoundDecimal，继承自 sympy.Function
# 表示将浮点数四舍五入到指定小数位数的操作
class RoundDecimal(sympy.Function):
    # 表示该函数返回的结果是实数但不一定是整数
    is_integer = False
    is_real = True

    # 类方法 eval，用于计算函数的返回值
    @classmethod
    def eval(cls, number, ndigits):
        # assert number.is_integer is not True, number
        # 如果 number 和 ndigits 都是 sympy.Number 和 sympy.Integer 类型，则将 number 转换为 float，然后使用 round 函数四舍五入到指定的小数位数，并返回对应的 sympy.Float 类型
        if isinstance(number, sympy.Number) and isinstance(ndigits, sympy.Integer):
            return sympy.Float(round(float(number), int(ndigits)))


# 定义一个自定义函数类 ToFloat，继承自 sympy.Function
# 表示将整数或特殊值转换为浮点数的操作
class ToFloat(sympy.Function):
    # 表示该函数返回的结果是实数但不一定是整数
    is_integer = False
    is_real = True

    # 类方法 eval，用于计算函数的返回值
    @classmethod
    def eval(cls, number):
        # 如果 number 是正无穷或负无穷，则直接返回该值
        if number in [sympy.oo, -sympy.oo]:
            return number
        # 如果 number 是整数，则将其转换为 sympy.Float 类型
        if isinstance(number, sympy.Integer):
            return sympy.Float(int(number))
        # 如果 number 是 int_oo，则返回 sympy.oo；如果是 -int_oo，则返回 -sympy.oo
        if number is int_oo:
            return sympy.oo
        if number is -int_oo:
            return -sympy.oo


# 定义一个自定义函数类 Identity，继承自 sympy.Function
# 用于包装一个参数，阻止展开和其他优化
class Identity(sympy.Function):
    """
    阻止展开和其他优化
    """

    # 重写 __repr__ 方法，返回对象的字符串表示形式
    def __repr__(self):
        return f"Identity({self.args[0]})"

    # 返回参数的实数属性
    def _eval_is_real(self):
        return self.args[0].is_real

    # 返回参数的整数属性，忽略类型检查警告
    def _eval_is_integer(self):
        return self.args[0].is_integer  # type: ignore[attr-defined]
    # 定义一个自定义的 sympy 函数类 OpaqueUnaryFn，继承自 sympy.Function
    class OpaqueUnaryFn(sympy.Function):
        """
        Unlike the builtin sympy functions on real numbers like sympy.sqrt,
        these equivalents do not do any nontrivial reasoning besides
        constant propagation.  This helps avoid performing transformations
        that are valid for real numbers but are invalid for floating point;
        in particular, while we are willing to make optimizations that change
        numerics for Tensor compute, we are NOT willing to make optimziations
        that change numerics for size compute.
        """
        
        # 设置类变量 _torch_handler_name，其值为 name 参数的值
        _torch_handler_name = name

        # 定义类方法 eval，用于计算函数对参数 a 的值进行评估
        @classmethod
        def eval(cls, a):
            # 如果 a 是 sympy.Integer 或 sympy.Float 类型的实例
            if isinstance(a, (sympy.Integer, sympy.Float)):
                # Python 在进行计算之前会将 a 转换为 float64 类型，例如：
                # >>> math.sin(2**53+1)
                # -0.848925964814655
                # >>> math.sin(float(2**53+1))
                # -0.848925964814655
                try:
                    # 尝试使用 math 模块中的 name 函数对 float(a) 进行计算，并转换为 sympy.Float 类型返回
                    return sympy.Float(getattr(math, name)(float(a)))
                # 处理溢出错误，使用 sympy 的语义处理无穷大/溢出，可能会返回一些奇怪的对象
                except OverflowError:
                    # 直接调用 sympy 模块中的 name 函数对 a 进行计算并返回结果
                    return getattr(sympy, name)(a)
            # 如果 a 是无穷大、负无穷大、复数无穷大、或者 int_oo、-int_oo 等特定的对象
            elif a in [sympy.oo, -sympy.oo, sympy.zoo, -sympy.zoo, int_oo, -int_oo]:
                # 如果 a 是 int_oo，则将其替换为 sympy.oo
                if a is int_oo:
                    a = sympy.oo
                # 如果 a 是 -int_oo，则将其替换为 -sympy.oo
                if a is -int_oo:
                    a = -sympy.oo
                # 调用 sympy 模块中的 name 函数对 a 进行计算并返回结果
                return getattr(sympy, name)(a)
            # 如果 a 不满足以上条件，返回 None
            return None

    # 将 OpaqueUnaryFn 类的 __name__ 属性设置为 "OpaqueUnaryFn_" 加上 name 参数的值
    OpaqueUnaryFn.__name__ = "OpaqueUnaryFn_" + name

    # 返回定义好的 OpaqueUnaryFn 类
    return OpaqueUnaryFn
# 在 torch/fx/experimental/sym_node.py 中的 math_op_names 中保持同步
OpaqueUnaryFn_sqrt = make_opaque_unary_fn("sqrt")
# 创建一个操作符，代表求平方根的不透明一元函数
OpaqueUnaryFn_cos = make_opaque_unary_fn("cos")
# 创建一个操作符，代表余弦函数的不透明一元函数
OpaqueUnaryFn_cosh = make_opaque_unary_fn("cosh")
# 创建一个操作符，代表双曲余弦函数的不透明一元函数
OpaqueUnaryFn_sin = make_opaque_unary_fn("sin")
# 创建一个操作符，代表正弦函数的不透明一元函数
OpaqueUnaryFn_sinh = make_opaque_unary_fn("sinh")
# 创建一个操作符，代表双曲正弦函数的不透明一元函数
OpaqueUnaryFn_tan = make_opaque_unary_fn("tan")
# 创建一个操作符，代表正切函数的不透明一元函数
OpaqueUnaryFn_tanh = make_opaque_unary_fn("tanh")
# 创建一个操作符，代表双曲正切函数的不透明一元函数
OpaqueUnaryFn_asin = make_opaque_unary_fn("asin")
# 创建一个操作符，代表反正弦函数的不透明一元函数
OpaqueUnaryFn_acos = make_opaque_unary_fn("acos")
# 创建一个操作符，代表反余弦函数的不透明一元函数
OpaqueUnaryFn_atan = make_opaque_unary_fn("atan")
# 创建一个操作符，代表反正切函数的不透明一元函数
OpaqueUnaryFn_exp = make_opaque_unary_fn("exp")
# 创建一个操作符，代表指数函数的不透明一元函数
OpaqueUnaryFn_log = make_opaque_unary_fn("log")
# 创建一个操作符，代表对数函数的不透明一元函数
OpaqueUnaryFn_asinh = make_opaque_unary_fn("asinh")
# 创建一个操作符，代表反双曲正弦函数的不透明一元函数
```