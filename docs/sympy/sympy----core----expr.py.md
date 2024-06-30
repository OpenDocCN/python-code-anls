# `D:\src\scipysrc\sympy\sympy\core\expr.py`

```
# 引入未来版本的注释语法，使得类型提示中的类型可以作为表达式的一部分
from __future__ import annotations

# 引入类型检查模块
from typing import TYPE_CHECKING
# 引入集合类的抽象基类
from collections.abc import Iterable
# 引入 functools 模块中的 reduce 函数
from functools import reduce
# 引入正则表达式模块
import re

# 从当前包中导入 sympify 和 _sympify 函数
from .sympify import sympify, _sympify
# 从当前包中导入 Basic 和 Atom 类
from .basic import Basic, Atom
# 从当前包中导入单例对象 S
from .singleton import S
# 从当前包中导入 EvalfMixin、pure_complex 和 DEFAULT_MAXPREC 等内容
from .evalf import EvalfMixin, pure_complex, DEFAULT_MAXPREC
# 从当前包中导入装饰器函数
from .decorators import call_highest_priority, sympify_method_args, sympify_return
# 从当前包中导入缓存函数
from .cache import cacheit
# 从当前包中导入模数求逆函数
from .intfunc import mod_inverse
# 从当前包中导入默认排序键函数
from .sorting import default_sort_key
# 从当前包中导入数值类型的分类
from .kind import NumberKind
# 从 sympy.utilities.exceptions 模块导入 sympy_deprecation_warning 异常
from sympy.utilities.exceptions import sympy_deprecation_warning
# 从 sympy.utilities.misc 模块导入各种实用函数
from sympy.utilities.misc import as_int, func_name, filldedent
# 从 sympy.utilities.iterables 模块导入 has_variety 和 sift 函数
from sympy.utilities.iterables import has_variety, sift
# 从 mpmath.libmp 模块导入 mpf_log 和 prec_to_dps 函数
from mpmath.libmp import mpf_log, prec_to_dps
# 从 mpmath.libmp.libintmath 模块导入 giant_steps 函数
from mpmath.libmp.libintmath import giant_steps

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 从当前包中导入 Number 类型
    from .numbers import Number

# 从 collections 模块中导入 defaultdict 类
from collections import defaultdict


# 定义一个辅助函数 _corem，用于在 extract_additively 中提取内容
def _corem(eq, c):  # helper for extract_additively
    # 返回 co, diff，其中 co 是 eq 中 c 的系数构成的列表，diff 是不包含 c 的部分
    co = []
    non = []
    # 遍历 eq 中的每一项
    for i in Add.make_args(eq):
        # 获取当前项 i 关于变量 c 的系数
        ci = i.coeff(c)
        # 如果系数 ci 为 0，则将 i 加入 non 列表
        if not ci:
            non.append(i)
        else:
            # 否则将 ci 加入 co 列表
            co.append(ci)
    # 返回一个 Add 对象，co 列表中的项相加得到的结果作为 co，non 列表中的项相加得到的结果作为 non
    return Add(*co), Add(*non)


# 定义一个代数表达式类 Expr，继承自 Basic 和 EvalfMixin
@sympify_method_args
class Expr(Basic, EvalfMixin):
    """
    代数表达式的基类。

    Explanation
    ===========

    所有需要定义算术操作的类都应该继承自这个类，而不是 Basic 类（Basic 类应该
    仅用于参数存储和表达式操作，如模式匹配、替换等）。

    如果要重写表达式的比较操作：
    应该使用 _eval_is_ge 来定义不等式，或者使用 _eval_is_eq，进行多分派。
    _eval_is_ge 在 x >= y 时返回 True，在 x < y 时返回 False，在类型不可比较或比较不确定时返回 None。

    See Also
    ========

    sympy.core.basic.Basic
    """

    __slots__: tuple[str, ...] = ()

    # 标识此类是一个标量，即其自导数为 1
    is_scalar = True  # self derivative is 1

    @property
    def _diff_wrt(self):
        """
        Return True if one can differentiate with respect to this
        object, else False.

        Explanation
        ===========
        
        Subclasses such as Symbol, Function and Derivative return True
        to enable derivatives wrt them. The implementation in Derivative
        separates the Symbol and non-Symbol (_diff_wrt=True) variables and
        temporarily converts the non-Symbols into Symbols when performing
        the differentiation. By default, any object deriving from Expr
        will behave like a scalar with self.diff(self) == 1. If this is
        not desired then the object must also set `is_scalar = False` or
        else define an _eval_derivative routine.

        Note, see the docstring of Derivative for how this should work
        mathematically. In particular, note that expr.subs(yourclass, Symbol)
        should be well-defined on a structural level, or this will lead to
        inconsistent results.

        Examples
        ========

        >>> from sympy import Expr
        >>> e = Expr()
        >>> e._diff_wrt
        False
        >>> class MyScalar(Expr):
        ...     _diff_wrt = True
        ...
        >>> MyScalar().diff(MyScalar())
        1
        >>> class MySymbol(Expr):
        ...     _diff_wrt = True
        ...     is_scalar = False
        ...
        >>> MySymbol().diff(MySymbol())
        Derivative(MySymbol(), MySymbol())
        """
        return False

    @cacheit
    def sort_key(self, order=None):
        """
        Compute a sorting key for the expression to facilitate sorting.

        Parameters
        ==========
        order : optional
            The sorting order.

        Returns
        =======
        tuple
            A tuple consisting of:
            1. The class key of the expression.
            2. Sorted arguments based on the type of expression.
            3. Sorting key for the exponent.
            4. Coefficient of the expression.

        Notes
        =====
        
        This method generates a tuple that can be used to compare and sort
        instances of expressions based on specific sorting criteria. It handles
        different types of expressions such as powers, dummies, atoms, sums,
        and products, ensuring a consistent sorting order.

        Examples
        ========
        
        >>> from sympy import Symbol, exp
        >>> x = Symbol('x')
        >>> (2*x).sort_key()
        (1, ('x',), 1, 2)
        >>> (x**2).sort_key()
        (<class 'sympy.core.power.Pow'>, (1, ('x',), 1, 2))
        >>> (exp(x)).sort_key()
        (<class 'sympy.core.function.FunctionClass'>, (1, ('exp', ('x',)), 1, 1))
        """
        
        coeff, expr = self.as_coeff_Mul()

        if expr.is_Pow:
            if expr.base is S.Exp1:
                # If we remove this, many doctests will go crazy:
                # (keeps E**x sorted like the exp(x) function,
                #  part of exp(x) to E**x transition)
                expr, exp = Function("exp")(expr.exp), S.One
            else:
                expr, exp = expr.args
        else:
            exp = S.One

        if expr.is_Dummy:
            args = (expr.sort_key(),)
        elif expr.is_Atom:
            args = (str(expr),)
        else:
            if expr.is_Add:
                args = expr.as_ordered_terms(order=order)
            elif expr.is_Mul:
                args = expr.as_ordered_factors(order=order)
            else:
                args = expr.args

            args = tuple(
                [ default_sort_key(arg, order=order) for arg in args ])

        args = (len(args), tuple(args))
        exp = exp.sort_key(order=order)

        return expr.class_key(), args, exp, coeff
    def _hashable_content(self):
        """Return a tuple of information about self that can be used to
        compute the hash. If a class defines additional attributes,
        like ``name`` in Symbol, then this method should be updated
        accordingly to return such relevant attributes.
        Defining more than _hashable_content is necessary if __eq__ has
        been defined by a class. See note about this in Basic.__eq__."""
        # 返回一个包含 self 信息的元组，用于计算哈希值。如果类定义了额外的属性，
        # 比如 Symbol 类中的 ``name``，则应相应更新此方法以返回相关属性。
        # 如果类定义了 __eq__ 方法，那么需要定义更多方法以适应。参见 Basic.__eq__ 中的相关说明。
        return self._args

    # ***************
    # * Arithmetics *
    # ***************
    # Expr 及其子类使用 _op_priority 来确定传递给二元特殊方法（如 __mul__ 等）的对象，
    # 将由哪个对象处理操作。通常情况下，'call_highest_priority' 装饰器将选择具有最高 _op_priority 的对象来处理调用。
    # 希望定义自己的二元特殊方法的自定义子类应设置比默认值更高的 _op_priority 值。
    #
    # **注意**：
    # 这是一个临时修复，最终将用更好更强大的方法替换。参见问题 5510。
    _op_priority = 10.0

    @property
    def _add_handler(self):
        # 返回 Add 类作为加法处理器
        return Add

    @property
    def _mul_handler(self):
        # 返回 Mul 类作为乘法处理器
        return Mul

    def __pos__(self):
        # 正数运算符，返回自身
        return self

    def __neg__(self):
        # 负数运算符，使用 Mul 类的 __neg__ 方法创建一个 -1 在位置 0 的二元乘法表达式
        c = self.is_commutative
        return Mul._from_args((S.NegativeOne, self), c)

    def __abs__(self) -> Expr:
        # 绝对值运算符，返回 Abs 函数应用于自身的结果
        from sympy.functions.elementary.complexes import Abs
        return Abs(self)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__radd__')
    def __add__(self, other):
        # 加法运算符，返回 Add 类的实例，表示 self + other
        return Add(self, other)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__add__')
    def __radd__(self, other):
        # 反向加法运算符，返回 Add 类的实例，表示 other + self
        return Add(other, self)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        # 减法运算符，返回 Add 类的实例，表示 self - other
        return Add(self, -other)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        # 反向减法运算符，返回 Add 类的实例，表示 other - self
        return Add(other, -self)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        # 乘法运算符，返回 Mul 类的实例，表示 self * other
        return Mul(self, other)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        # 反向乘法运算符，返回 Mul 类的实例，表示 other * self
        return Mul(other, self)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__rpow__')
    def _pow(self, other):
        # 指数运算符，返回 Pow 类的实例，表示 self ** other
        return Pow(self, other)
    # 定义特殊方法 __pow__，处理指数运算
    def __pow__(self, other, mod=None) -> Expr:
        # 如果 mod 为 None，使用内部方法 _pow 处理
        if mod is None:
            return self._pow(other)
        try:
            # 将 self, other, mod 转换为整数形式
            _self, other, mod = as_int(self), as_int(other), as_int(mod)
            # 如果指数 other 非负，返回 pow(_self, other, mod) 的 sympify 结果
            if other >= 0:
                return _sympify(pow(_self, other, mod))
            # 如果指数 other 为负，返回 pow(_self, -other, mod) 的模反元素的 sympify 结果
            else:
                return _sympify(mod_inverse(pow(_self, -other, mod), mod))
        except ValueError:
            # 如果出现 ValueError，调用 self._pow(other) 并取模 mod 返回结果
            power = self._pow(other)
            try:
                return power % mod
            except TypeError:
                return NotImplemented

    # 定义特殊方法 __rpow__，处理反向指数运算
    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__pow__')
    def __rpow__(self, other):
        # 返回 Pow(other, self) 的结果
        return Pow(other, self)

    # 定义特殊方法 __truediv__，处理真除运算
    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__rtruediv__')
    def __truediv__(self, other):
        # 计算分母的倒数
        denom = Pow(other, S.NegativeOne)
        # 如果 self 是 1，直接返回分母 denom
        if self is S.One:
            return denom
        else:
            # 否则返回 self * denom 的结果
            return Mul(self, denom)

    # 定义特殊方法 __rtruediv__，处理反向真除运算
    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__truediv__')
    def __rtruediv__(self, other):
        # 计算 self 的倒数
        denom = Pow(self, S.NegativeOne)
        # 如果 other 是 1，直接返回 denom
        if other is S.One:
            return denom
        else:
            # 否则返回 other * denom 的结果
            return Mul(other, denom)

    # 定义特殊方法 __mod__，处理取模运算
    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__rmod__')
    def __mod__(self, other):
        # 返回 Mod(self, other) 的结果
        return Mod(self, other)

    # 定义特殊方法 __rmod__，处理反向取模运算
    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__mod__')
    def __rmod__(self, other):
        # 返回 Mod(other, self) 的结果
        return Mod(other, self)

    # 定义特殊方法 __floordiv__，处理整除运算
    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__rfloordiv__')
    def __floordiv__(self, other):
        # 导入整数处理模块 floor，返回 self / other 的地板除结果
        from sympy.functions.elementary.integers import floor
        return floor(self / other)

    # 定义特殊方法 __rfloordiv__，处理反向整除运算
    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__floordiv__')
    def __rfloordiv__(self, other):
        # 导入整数处理模块 floor，返回 other / self 的地板除结果
        from sympy.functions.elementary.integers import floor
        return floor(other / self)

    # 定义特殊方法 __divmod__，处理除和取模运算
    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__rdivmod__')
    def __divmod__(self, other):
        # 导入整数处理模块 floor，返回 self / other 和 Mod(self, other) 的结果
        from sympy.functions.elementary.integers import floor
        return floor(self / other), Mod(self, other)

    # 定义特殊方法 __rdivmod__，处理反向除和取模运算
    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__divmod__')
    def __rdivmod__(self, other):
        # 导入整数处理模块 floor，返回 other / self 和 Mod(other, self) 的结果
        from sympy.functions.elementary.integers import floor
        return floor(other / self), Mod(other, self)
    def __int__(self):
        # 如果对象不是数值类型，则抛出类型错误异常
        if not self.is_number:
            raise TypeError("Cannot convert symbols to int")
        # 对对象四舍五入到小数点后两位
        r = self.round(2)
        # 如果四舍五入后的结果不是数值类型，则抛出类型错误异常
        if not r.is_Number:
            raise TypeError("Cannot convert complex to int")
        # 如果四舍五入后的结果是特殊值（NaN, Infinity, NegativeInfinity），则抛出类型错误异常
        if r in (S.NaN, S.Infinity, S.NegativeInfinity):
            raise TypeError("Cannot convert %s to int" % r)
        # 将四舍五入后的结果转换为整数
        i = int(r)
        # 如果整数值为0，则直接返回0
        if not i:
            return i
        # 如果四舍五入后的结果为整数，则进行以下处理
        if int_valued(r):
            # 对非整数对象进行以下测试
            # 如果对象大于整数i，则返回i
            if (self > i) is S.true:
                return i
            # 如果对象小于整数i，则返回i-1
            if (self < i) is S.true:
                return i - 1
            # 检查对象是否与整数i相等
            ok = self.equals(i)
            # 如果无法准确计算对象与整数的相等性，则抛出类型错误异常
            if ok is None:
                raise TypeError('cannot compute int value accurately')
            # 如果对象与整数相等，则返回整数i
            if ok:
                return i
            # 如果对象与整数相差一，根据正负返回相应值
            return i - (1 if i > 0 else -1)
        # 返回整数值i
        return i

    def __float__(self):
        # 对对象进行数值评估
        result = self.evalf()
        # 如果评估后的结果是数值类型，则返回其浮点数表示
        if result.is_Number:
            return float(result)
        # 如果评估后的结果是复数类型，则抛出类型错误异常
        if result.is_number and result.as_real_imag()[1]:
            raise TypeError("Cannot convert complex to float")
        # 如果评估后的结果既非数值类型也非复数类型，则抛出类型错误异常
        raise TypeError("Cannot convert expression to float")

    def __complex__(self):
        # 对对象进行数值评估
        result = self.evalf()
        # 将评估后的实部和虚部转换为浮点数，构造复数对象并返回
        re, im = result.as_real_imag()
        return complex(float(re), float(im))

    @sympify_return([('other', 'Expr')], NotImplemented)
    def __ge__(self, other):
        # 导入关系模块中的GreaterThan类，返回当前对象是否大于等于另一对象的比较结果
        from .relational import GreaterThan
        return GreaterThan(self, other)

    @sympify_return([('other', 'Expr')], NotImplemented)
    def __le__(self, other):
        # 导入关系模块中的LessThan类，返回当前对象是否小于等于另一对象的比较结果
        from .relational import LessThan
        return LessThan(self, other)

    @sympify_return([('other', 'Expr')], NotImplemented)
    def __gt__(self, other):
        # 导入关系模块中的StrictGreaterThan类，返回当前对象是否严格大于另一对象的比较结果
        from .relational import StrictGreaterThan
        return StrictGreaterThan(self, other)

    @sympify_return([('other', 'Expr')], NotImplemented)
    def __lt__(self, other):
        # 导入关系模块中的StrictLessThan类，返回当前对象是否严格小于另一对象的比较结果
        from .relational import StrictLessThan
        return StrictLessThan(self, other)

    def __trunc__(self):
        # 如果对象不是数值类型，则抛出类型错误异常
        if not self.is_number:
            raise TypeError("Cannot truncate symbols and expressions")
        else:
            # 否则，返回对象的整数形式
            return Integer(self)

    def __format__(self, format_spec: str):
        # 如果对象是数值类型，则尝试匹配给定格式
        if self.is_number:
            # 使用正则表达式匹配浮点数格式
            mt = re.match(r'\+?\d*\.(\d+)f', format_spec)
            # 如果匹配成功
            if mt:
                # 获取精度值并对对象四舍五入
                prec = int(mt.group(1))
                rounded = self.round(prec)
                # 如果四舍五入后的结果是整数，则返回格式化后的整数
                if rounded.is_Integer:
                    return format(int(rounded), format_spec)
                # 如果四舍五入后的结果是浮点数，则返回格式化后的浮点数
                if rounded.is_Float:
                    return format(rounded, format_spec)
        # 如果对象不是数值类型或格式不匹配，则调用父类的格式化方法
        return super().__format__(format_spec)
    def _from_mpmath(x, prec):
        # 如果 x 对象具有 "_mpf_" 属性，表示是一个 mpmath 的浮点数对象
        if hasattr(x, "_mpf_"):
            # 使用 Float._new 方法将其转换为 SymPy 的浮点数对象，设置精度为 prec
            return Float._new(x._mpf_, prec)
        # 如果 x 对象具有 "_mpc_" 属性，表示是一个 mpmath 的复数对象
        elif hasattr(x, "_mpc_"):
            # 将复数对象分离成实部和虚部
            re, im = x._mpc_
            # 使用 Float._new 方法分别将实部和虚部转换为 SymPy 的浮点数对象，设置精度为 prec
            re = Float._new(re, prec)
            im = Float._new(im, prec) * S.ImaginaryUnit  # 将虚部乘以虚数单位
            # 返回实部和虚部的和作为 SymPy 的复数对象
            return re + im
        else:
            # 如果 x 不是 mpmath 的浮点数或复数对象，则抛出类型错误异常
            raise TypeError("expected mpmath number (mpf or mpc)")

    @property
    def is_number(self):
        """Returns True if ``self`` has no free symbols and no
        undefined functions (AppliedUndef, to be precise). It will be
        faster than ``if not self.free_symbols``, however, since
        ``is_number`` will fail as soon as it hits a free symbol
        or undefined function.

        Examples
        ========

        >>> from sympy import Function, Integral, cos, sin, pi
        >>> from sympy.abc import x
        >>> f = Function('f')

        >>> x.is_number
        False
        >>> f(1).is_number
        False
        >>> (2*x).is_number
        False
        >>> (2 + Integral(2, x)).is_number
        False
        >>> (2 + Integral(2, (x, 1, 2))).is_number
        True

        Not all numbers are Numbers in the SymPy sense:

        >>> pi.is_number, pi.is_Number
        (True, False)

        If something is a number it should evaluate to a number with
        real and imaginary parts that are Numbers; the result may not
        be comparable, however, since the real and/or imaginary part
        of the result may not have precision.

        >>> cos(1).is_number and cos(1).is_comparable
        True

        >>> z = cos(1)**2 + sin(1)**2 - 1
        >>> z.is_number
        True
        >>> z.is_comparable
        False

        See Also
        ========

        sympy.core.basic.Basic.is_comparable
        """
        # 返回一个布尔值，判断当前对象是否是一个数值类型（没有自由符号和未定义函数）
        return all(obj.is_number for obj in self.args)
    def _random(self, n=None, re_min=-1, im_min=-1, re_max=1, im_max=1):
        """Return self evaluated, if possible, replacing free symbols with
        random complex values, if necessary.

        Explanation
        ===========

        The random complex value for each free symbol is generated
        by the random_complex_number routine giving real and imaginary
        parts in the range given by the re_min, re_max, im_min, and im_max
        values. The returned value is evaluated to a precision of n
        (if given) else the maximum of 15 and the precision needed
        to get more than 1 digit of precision. If the expression
        could not be evaluated to a number, or could not be evaluated
        to more than 1 digit of precision, then None is returned.

        Examples
        ========

        >>> from sympy import sqrt
        >>> from sympy.abc import x, y
        >>> x._random()                         # doctest: +SKIP
        0.0392918155679172 + 0.916050214307199*I
        >>> x._random(2)                        # doctest: +SKIP
        -0.77 - 0.87*I
        >>> (x + y/2)._random(2)                # doctest: +SKIP
        -0.57 + 0.16*I
        >>> sqrt(2)._random(2)
        1.4

        See Also
        ========

        sympy.core.random.random_complex_number
        """

        # 获取自由符号集合
        free = self.free_symbols
        # 默认精度为1
        prec = 1

        # 如果存在自由符号，则生成相应的随机复数，并用其替换自由符号
        if free:
            from sympy.core.random import random_complex_number
            a, c, b, d = re_min, re_max, im_min, im_max
            # 创建用于替换的字典，将自由符号映射为随机复数
            reps = dict(list(zip(free, [random_complex_number(a, b, c, d, rational=True)
                           for zi in free])))
            try:
                # 尝试对表达式求值，并计算其绝对值
                nmag = abs(self.evalf(2, subs=reps))
            except (ValueError, TypeError):
                # 如果求值过程中出现错误，返回None
                # 例如，由于超出范围而导致的 evalf 问题
                return None
        else:
            # 如果没有自由符号，则直接进行求值，计算其绝对值
            reps = {}
            nmag = abs(self.evalf(2))

        # 如果 nmag 没有 '_prec' 属性，返回 None
        if not hasattr(nmag, '_prec'):
            # 例如，exp_polar(2*I*pi) 无法求值但 is_number 为True
            return None

        # 如果 nmag 的精度为1，尝试逐步增加精度直到达到最大默认精度，以获得有效结果
        if nmag._prec == 1:
            for prec in giant_steps(2, DEFAULT_MAXPREC):
                nmag = abs(self.evalf(prec, subs=reps))
                if nmag._prec != 1:
                    break

        # 如果 nmag 的精度不为1，则根据需要设定的精度 n 进行求值
        if nmag._prec != 1:
            if n is None:
                n = max(prec, 15)
            return self.evalf(n, subs=reps)

        # 如果未获得任何有效结果，返回 None
        return None
    # 定义一个方法用于评估对象是否是扩展的正数或负数
    def _eval_is_extended_positive_negative(self, positive):
        # 导入数字段相关的模块和异常类
        from sympy.polys.numberfields import minimal_polynomial
        from sympy.polys.polyerrors import NotAlgebraic
        
        # 如果对象是一个数字
        if self.is_number:
            # 尝试评估对象在精度为2时的值
            try:
                n2 = self._eval_evalf(2)
            # 捕获值错误异常
            # XXX: 这里不应该捕获这个异常
            # 捕获 ValueError: hypsum() failed to converge to the requested
            # 34 bits of accuracy
            except ValueError:
                return None
            
            # 如果 n2 为 None，则返回 None
            if n2 is None:
                return None
            
            # 如果 n2 的精度为 1（没有显著性），则返回 None
            if getattr(n2, '_prec', 1) == 1:
                return None
            
            # 如果 n2 是 NaN，则返回 None
            if n2 is S.NaN:
                return None

            # 对对象在精度为2时进行实际评估
            f = self.evalf(2)
            
            # 如果 f 是浮点数
            if f.is_Float:
                match = f, S.Zero
            else:
                # 否则尝试找到纯复数匹配项
                match = pure_complex(f)
            
            # 如果找不到匹配项，则返回 False
            if match is None:
                return False
            
            # 分别获取实部和虚部
            r, i = match
            
            # 如果实部和虚部都是数字
            if not (i.is_Number and r.is_Number):
                return False
            
            # 如果实部和虚部的精度都不是 1
            if r._prec != 1 and i._prec != 1:
                # 如果虚部为零且实部符号满足条件（正数或负数），则返回 True
                return bool(not i and ((r > 0) if positive else (r < 0)))
            
            # 如果实部精度是 1，且虚部要么为空要么精度也是 1，
            # 并且对象是代数的，并且不包含函数形式
            elif r._prec == 1 and (not i or i._prec == 1) and \
                    self._eval_is_algebraic() and not self.has(Function):
                try:
                    # 尝试计算对象的最小多项式，如果结果是符号则返回 False
                    if minimal_polynomial(self).is_Symbol:
                        return False
                except (NotAlgebraic, NotImplementedError):
                    pass

    # 定义一个方法用于评估对象是否是扩展的正数
    def _eval_is_extended_positive(self):
        return self._eval_is_extended_positive_negative(positive=True)

    # 定义一个方法用于评估对象是否是扩展的负数
    def _eval_is_extended_negative(self):
        return self._eval_is_extended_positive_negative(positive=False)
    def _eval_interval(self, x, a, b):
        """
        返回指定区间的函数求值。对于大多数函数来说，计算方式是：

        self.subs(x, b) - self.subs(x, a),

        如果在计算 subs 时返回 NaN，或者在 a 和 b 之间存在奇点，则可能使用 limit() 进行计算。

        如果 b 或者 a 是 None，则分别只计算 -self.subs(x, a) 或 self.subs(b, x)。

        """

        from sympy.calculus.accumulationbounds import AccumBounds  # 导入累积边界模块
        from sympy.functions.elementary.exponential import log  # 导入对数函数
        from sympy.series.limits import limit, Limit  # 导入极限计算相关函数
        from sympy.sets.sets import Interval  # 导入区间类
        from sympy.solvers.solveset import solveset  # 导入求解方程的函数

        if (a is None and b is None):
            raise ValueError('Both interval ends cannot be None.')  # 抛出值错误，区间两端不能都是 None

        def _eval_endpoint(left):
            c = a if left else b
            if c is None:
                return S.Zero  # 如果端点是 None，则返回零
            else:
                C = self.subs(x, c)  # 计算在端点处的函数值
                if C.has(S.NaN, S.Infinity, S.NegativeInfinity,
                         S.ComplexInfinity, AccumBounds):
                    if (a < b) != False:
                        C = limit(self, x, c, "+" if left else "-")  # 使用 limit 计算函数在端点处的极限
                    else:
                        C = limit(self, x, c, "-" if left else "+")

                    if isinstance(C, Limit):
                        raise NotImplementedError("Could not compute limit")  # 如果计算极限时出现问题，则抛出未实现错误
            return C

        if a == b:
            return S.Zero  # 如果 a 等于 b，则返回零

        A = _eval_endpoint(left=True)  # 计算左端点处的函数值
        if A is S.NaN:
            return A  # 如果左端点处的函数值是 NaN，则直接返回 NaN

        B = _eval_endpoint(left=False)  # 计算右端点处的函数值

        if (a and b) is None:
            return B - A  # 如果 a 和 b 都是 None，则返回右端点减去左端点的结果

        value = B - A  # 计算最终的区间函数值

        if a.is_comparable and b.is_comparable:
            if a < b:
                domain = Interval(a, b)
            else:
                domain = Interval(b, a)
            # 检查在区间内 self 的奇点
            # 如果 singularities 是 ConditionSet（不可迭代的），捕获异常并继续
            singularities = solveset(self.cancel().as_numer_denom()[1], x,
                domain=domain)
            for logterm in self.atoms(log):
                singularities = singularities | solveset(logterm.args[0], x,
                    domain=domain)
            try:
                for s in singularities:
                    if value is S.NaN:
                        # 无需继续添加，结果将保持为 NaN
                        break
                    if not s.is_comparable:
                        continue
                    if (a < s) == (s < b) == True:
                        value += -limit(self, x, s, "+") + limit(self, x, s, "-")  # 更新区间函数值，考虑奇点的影响
                    elif (b < s) == (s < a) == True:
                        value += limit(self, x, s, "+") - limit(self, x, s, "-")  # 更新区间函数值，考虑奇点的影响
            except TypeError:
                pass

        return value  # 返回最终计算得到的区间函数值
    # 子类方法，计算 self**other，处理 other 不为 NaN、0 或 1 的情况
    def _eval_power(self, other):
        return None

    # 返回复数的共轭，如果是实数返回自身，如果是虚数返回负数
    def _eval_conjugate(self):
        if self.is_extended_real:
            return self
        elif self.is_imaginary:
            return -self

    # 返回对象的复数共轭
    def conjugate(self):
        from sympy.functions.elementary.complexes import conjugate as c
        return c(self)

    # 返回对象在给定变量 x 和方向 cdir 下的方向导数
    def dir(self, x, cdir):
        if self.is_zero:
            return S.Zero
        from sympy.functions.elementary.exponential import log
        minexp = S.Zero
        arg = self
        while arg:
            minexp += S.One
            arg = arg.diff(x)
            coeff = arg.subs(x, 0)
            if coeff is S.NaN:
                coeff = arg.limit(x, 0)
            if coeff is S.ComplexInfinity:
                try:
                    coeff, _ = arg.leadterm(x)
                    if coeff.has(log(x)):
                        raise ValueError()
                except ValueError:
                    coeff = arg.limit(x, 0)
            if coeff != S.Zero:
                break
        return coeff*cdir**minexp

    # 返回对象的转置
    def _eval_transpose(self):
        from sympy.functions.elementary.complexes import conjugate
        if (self.is_complex or self.is_infinite):
            return self
        elif self.is_hermitian:
            return conjugate(self)
        elif self.is_antihermitian:
            return -conjugate(self)

    # 返回对象的转置
    def transpose(self):
        from sympy.functions.elementary.complexes import transpose
        return transpose(self)

    # 返回对象的伴随
    def _eval_adjoint(self):
        from sympy.functions.elementary.complexes import conjugate, transpose
        if self.is_hermitian:
            return self
        elif self.is_antihermitian:
            return -self
        obj = self._eval_conjugate()
        if obj is not None:
            return transpose(obj)
        obj = self._eval_transpose()
        if obj is not None:
            return conjugate(obj)

    # 返回对象的伴随
    def adjoint(self):
        from sympy.functions.elementary.complexes import adjoint
        return adjoint(self)

    @classmethod
    def _parse_order(cls, order):
        """解析和配置项的顺序。"""
        # 导入 monomial_key 函数
        from sympy.polys.orderings import monomial_key

        # 获取 order 对象的 startswith 方法，用于检查是否以 'rev-' 开头
        startswith = getattr(order, "startswith", None)
        if startswith is None:
            reverse = False  # 如果没有 startswith 方法，则默认不反转顺序
        else:
            reverse = startswith('rev-')  # 如果以 'rev-' 开头，则反转顺序
            if reverse:
                order = order[4:]  # 如果反转顺序，则从第五个字符开始截取 order

        # 使用给定的 order 构造 monomial_key 函数对象
        monom_key = monomial_key(order)

        # 定义一个递归函数 neg，用于对单项进行取反操作
        def neg(monom):
            return tuple([neg(m) if isinstance(m, tuple) else -m for m in monom])

        # 定义一个 key 函数，用于生成排序所需的键值
        def key(term):
            _, ((re, im), monom, ncpart) = term

            # 对 monomial 使用 monom_key 函数进行处理，并取反
            monom = neg(monom_key(monom))
            # 对 ncpart 中的每个元素按指定 order 排序
            ncpart = tuple([e.sort_key(order=order) for e in ncpart])
            # 构造 coeff 元组
            coeff = ((bool(im), im), (re, im))

            return monom, ncpart, coeff

        return key, reverse

    def as_ordered_factors(self, order=None):
        """返回有序因子列表（如果是 Mul），否则返回 [self]。"""
        return [self]

    def as_poly(self, *gens, **args):
        """将 ``self`` 转换为多项式或返回 ``None``。

        解释
        ===========

        >>> from sympy import sin
        >>> from sympy.abc import x, y

        >>> print((x**2 + x*y).as_poly())
        Poly(x**2 + x*y, x, y, domain='ZZ')

        >>> print((x**2 + x*y).as_poly(x, y))
        Poly(x**2 + x*y, x, y, domain='ZZ')

        >>> print((x**2 + sin(y)).as_poly(x, y))
        None

        """
        # 导入异常处理相关的模块和类
        from sympy.polys.polyerrors import PolynomialError, GeneratorsNeeded
        from sympy.polys.polytools import Poly

        try:
            # 尝试将 self 转换为 Poly 对象，使用给定的 gens 和 args
            poly = Poly(self, *gens, **args)

            # 如果转换后不是 Poly 对象，则返回 None
            if not poly.is_Poly:
                return None
            else:
                return poly  # 如果是 Poly 对象，则返回该对象
        except (PolynomialError, GeneratorsNeeded):
            # 捕获可能的异常情况，返回 None
            # 例如，exp(x).as_poly(x) 会触发 PolynomialError
            # S(2).as_poly() 会触发 GeneratorsNeeded
            return None
    def as_ordered_terms(self, order=None, data=False):
        """
        Transform an expression to an ordered list of terms.

        Examples
        ========

        >>> from sympy import sin, cos
        >>> from sympy.abc import x

        >>> (sin(x)**2*cos(x) + sin(x)**2 + 1).as_ordered_terms()
        [sin(x)**2*cos(x), sin(x)**2, 1]

        """

        from .numbers import Number, NumberSymbol  # 导入需要的模块

        if order is None and self.is_Add:
            # 检查特殊情况：Add(Number, Mul(Number, expr))，第一个数字为正，第二个数字为负
            key = lambda x:not isinstance(x, (Number, NumberSymbol))
            # 提取出所有的加法项，并按照指定的排序键排序
            add_args = sorted(Add.make_args(self), key=key)
            if (len(add_args) == 2
                and isinstance(add_args[0], (Number, NumberSymbol))
                and isinstance(add_args[1], Mul)):
                # 如果加法项有两个，并且第一个是数字或数字符号，第二个是乘法项
                mul_args = sorted(Mul.make_args(add_args[1]), key=key)
                if (len(mul_args) == 2
                    and isinstance(mul_args[0], Number)
                    and add_args[0].is_positive
                    and mul_args[0].is_negative):
                    return add_args  # 返回排序好的加法项列表

        # 解析排序方式和是否逆序
        key, reverse = self._parse_order(order)
        # 将表达式转换为项和生成器
        terms, gens = self.as_terms()

        # 如果没有任何项是 Order 类型的，则按照给定的键和顺序排序
        if not any(term.is_Order for term, _ in terms):
            ordered = sorted(terms, key=key, reverse=reverse)
        else:
            _terms, _order = [], []

            # 将项分为普通项和 Order 类型的项，分别排序
            for term, repr in terms:
                if not term.is_Order:
                    _terms.append((term, repr))
                else:
                    _order.append((term, repr))

            ordered = sorted(_terms, key=key, reverse=True) \
                + sorted(_order, key=key, reverse=True)

        # 如果需要返回数据，则返回排序好的项和生成器；否则只返回排序好的项
        if data:
            return ordered, gens
        else:
            return [term for term, _ in ordered]
    def as_terms(self):
        """
        Transform an expression to a list of terms.

        This method decomposes an expression into constituent terms,
        extracting coefficients, commutative parts, and non-commutative parts.
        It returns a structured list of tuples where each tuple represents a term
        along with its components: coefficient, commutative part dictionary,
        and non-commutative part tuple.
        """
        from .exprtools import decompose_power  # 导入表达式工具中的 decompose_power 函数

        gens, terms = set(), []  # 初始化集合 gens 和空列表 terms

        for term in Add.make_args(self):  # 遍历表达式中的每一个项
            coeff, _term = term.as_coeff_Mul()  # 提取每个项的系数和乘积部分

            coeff = complex(coeff)  # 将系数转换为复数类型
            cpart, ncpart = {}, []  # 初始化 commutative 和 non-commutative 部分的数据结构

            if _term is not S.One:  # 如果乘积部分不是 1
                for factor in Mul.make_args(_term):  # 遍历乘积部分的每一个因子
                    if factor.is_number:  # 如果因子是数字
                        try:
                            coeff *= complex(factor)  # 更新系数
                        except (TypeError, ValueError):
                            pass
                        else:
                            continue

                    if factor.is_commutative:  # 如果因子是可交换的
                        base, exp = decompose_power(factor)  # 分解因子的幂次

                        cpart[base] = exp  # 更新 commutative 部分的字典
                        gens.add(base)  # 将基数加入 gens 集合
                    else:
                        ncpart.append(factor)  # 将非交换因子添加到 non-commutative 部分的列表

            coeff = coeff.real, coeff.imag  # 取系数的实部和虚部
            ncpart = tuple(ncpart)  # 将 non-commutative 部分转换为元组

            terms.append((term, (coeff, cpart, ncpart)))  # 将处理后的项添加到 terms 列表中

        gens = sorted(gens, key=default_sort_key)  # 对 gens 集合中的元素进行排序

        k, indices = len(gens), {}  # 初始化 k 和空字典 indices

        for i, g in enumerate(gens):  # 枚举 gens 集合中的元素
            indices[g] = i  # 构建基数到索引的映射关系

        result = []  # 初始化结果列表

        for term, (coeff, cpart, ncpart) in terms:  # 遍历处理后的每一个项
            monom = [0]*k  # 创建长度为 k 的零向量 monom

            for base, exp in cpart.items():  # 遍历 commutative 部分的字典
                monom[indices[base]] = exp  # 根据 indices 映射更新 monom 向量

            result.append((term, (coeff, tuple(monom), ncpart)))  # 将构建好的项添加到结果列表中

        return result, gens  # 返回最终的结果列表和 gens 集合

    def removeO(self):
        """
        Removes the additive O(..) symbol if there is one.

        This method is intended to remove any 'Big O' notation symbol
        from the expression if present.
        """
        return self  # 目前仅返回自身，未实现具体功能

    def getO(self):
        """
        Returns the additive O(..) symbol if there is one, else None.

        This method checks if there is any 'Big O' notation symbol in the expression
        and returns it if found, otherwise returns None.
        """
        return None  # 目前仅返回 None，未实现具体功能
    # 返回表达式的阶数。
    def getn(self):
        """
        Returns the order of the expression.

        Explanation
        ===========

        The order is determined either from the O(...) term. If there
        is no O(...) term, it returns None.

        Examples
        ========

        >>> from sympy import O
        >>> from sympy.abc import x
        >>> (1 + x + O(x**2)).getn()
        2
        >>> (1 + x).getn()

        """
        # 获取表达式中的 O(...) 项
        o = self.getO()
        # 如果没有 O(...) 项，则返回 None
        if o is None:
            return None
        # 如果 o 是 Order 对象
        elif o.is_Order:
            # 获取 Order 对象的表达式部分
            o = o.expr
            # 处理特殊情况：如果表达式为常数 1，则阶数为 0
            if o is S.One:
                return S.Zero
            # 如果表达式是一个符号，则阶数为 1
            if o.is_Symbol:
                return S.One
            # 如果表达式是幂函数 Pow，则阶数为幂的指数部分
            if o.is_Pow:
                return o.args[1]
            # 如果表达式是乘积 Mul
            if o.is_Mul:  # x**n*log(x)**n or x**n/log(x)**n
                # 遍历乘积的每一项
                for oi in o.args:
                    # 如果乘积项是符号，则阶数为 1
                    if oi.is_Symbol:
                        return S.One
                    # 如果乘积项是幂函数 Pow
                    if oi.is_Pow:
                        from .symbol import Dummy, Symbol
                        # 导入符号相关模块
                        syms = oi.atoms(Symbol)
                        # 获取幂函数中的符号集合
                        if len(syms) == 1:
                            x = syms.pop()
                            # 将幂函数中的符号替换为临时符号
                            oi = oi.subs(x, Dummy('x', positive=True))
                            # 如果基数是符号且指数是有理数，则返回指数的绝对值
                            if oi.base.is_Symbol and oi.exp.is_Rational:
                                return abs(oi.exp)

        # 如果无法确定阶数，则引发 NotImplementedError 异常
        raise NotImplementedError('not sure of order of %s' % o)

    # 计算表达式中的操作数数量
    def count_ops(self, visual=None):
        # 导入计算操作数的函数，并返回其结果
        from .function import count_ops
        return count_ops(self, visual)
    def args_cnc(self, cset=False, warn=True, split_1=True):
        """
        Return [commutative factors, non-commutative factors] of self.

        Explanation
        ===========

        self is treated as a Mul and the ordering of the factors is maintained.
        If ``cset`` is True the commutative factors will be returned in a set.
        If there were repeated factors (as may happen with an unevaluated Mul)
        then an error will be raised unless it is explicitly suppressed by
        setting ``warn`` to False.

        Note: -1 is always separated from a Number unless split_1 is False.

        Examples
        ========

        >>> from sympy import symbols, oo
        >>> A, B = symbols('A B', commutative=0)
        >>> x, y = symbols('x y')
        >>> (-2*x*y).args_cnc()
        [[-1, 2, x, y], []]
        >>> (-2.5*x).args_cnc()
        [[-1, 2.5, x], []]
        >>> (-2*x*A*B*y).args_cnc()
        [[-1, 2, x, y], [A, B]]
        >>> (-2*x*A*B*y).args_cnc(split_1=False)
        [[-2, x, y], [A, B]]
        >>> (-2*x*y).args_cnc(cset=True)
        [{-1, 2, x, y}, []]

        The arg is always treated as a Mul:

        >>> (-2 + x + A).args_cnc()
        [[], [x - 2 + A]]
        >>> (-oo).args_cnc() # -oo is a singleton
        [[-1, oo], []]
        """

        # Check if self is a Mul expression; if not, treat self as a single-element Mul
        if self.is_Mul:
            args = list(self.args)
        else:
            args = [self]

        # Separate commutative and non-commutative factors
        for i, mi in enumerate(args):
            if not mi.is_commutative:
                c = args[:i]
                nc = args[i:]
                break
        else:
            c = args
            nc = []

        # Adjust the representation of -1 in the commutative factors list if split_1 is True
        if c and split_1 and (
            c[0].is_Number and
            c[0].is_extended_negative and
                c[0] is not S.NegativeOne):
            c[:1] = [S.NegativeOne, -c[0]]

        # Convert commutative factors to a set if cset is True, and check for repeated elements
        if cset:
            clen = len(c)
            c = set(c)
            if clen and warn and len(c) != clen:
                raise ValueError('repeated commutative arguments: %s' %
                                 [ci for ci in c if list(self.args).count(ci) > 1])

        return [c, nc]

    def as_expr(self, *gens):
        """
        Convert a polynomial to a SymPy expression.

        Examples
        ========

        >>> from sympy import sin
        >>> from sympy.abc import x, y

        >>> f = (x**2 + x*y).as_poly(x, y)
        >>> f.as_expr()
        x**2 + x*y

        >>> sin(x).as_expr()
        sin(x)

        """
        return self
    # 定义一个方法，用于从表达式中提取给定表达式的符号系数。
    def as_coefficient(self, expr):
        """
        Extracts symbolic coefficient at the given expression. In
        other words, this functions separates 'self' into the product
        of 'expr' and 'expr'-free coefficient. If such separation
        is not possible it will return None.

        Examples
        ========

        >>> from sympy import E, pi, sin, I, Poly
        >>> from sympy.abc import x

        >>> E.as_coefficient(E)
        1
        >>> (2*E).as_coefficient(E)
        2
        >>> (2*sin(E)*E).as_coefficient(E)

        Two terms have E in them so a sum is returned. (If one were
        desiring the coefficient of the term exactly matching E then
        the constant from the returned expression could be selected.
        Or, for greater precision, a method of Poly can be used to
        indicate the desired term from which the coefficient is
        desired.)

        >>> (2*E + x*E).as_coefficient(E)
        x + 2
        >>> _.args[0]  # just want the exact match
        2
        >>> p = Poly(2*E + x*E); p
        Poly(x*E + 2*E, x, E, domain='ZZ')
        >>> p.coeff_monomial(E)
        2
        >>> p.nth(0, 1)
        2

        Since the following cannot be written as a product containing
        E as a factor, None is returned. (If the coefficient ``2*x`` is
        desired then the ``coeff`` method should be used.)

        >>> (2*E*x + x).as_coefficient(E)
        >>> (2*E*x + x).coeff(E)
        2*x

        >>> (E*(x + 1) + x).as_coefficient(E)

        >>> (2*pi*I).as_coefficient(pi*I)
        2
        >>> (2*I).as_coefficient(pi*I)

        See Also
        ========

        coeff: return sum of terms have a given factor
        as_coeff_Add: separate the additive constant from an expression
        as_coeff_Mul: separate the multiplicative constant from an expression
        as_independent: separate x-dependent terms/factors from others
        sympy.polys.polytools.Poly.coeff_monomial: efficiently find the single coefficient of a monomial in Poly
        sympy.polys.polytools.Poly.nth: like coeff_monomial but powers of monomial terms are used
        """

        # 使用 extract_multiplicatively 方法从 self 中提取与 expr 相乘的结果
        r = self.extract_multiplicatively(expr)
        # 如果成功提取并且结果中不含 expr，则返回提取的结果
        if r and not r.has(expr):
            return r
    def as_real_imag(self, deep=True, **hints):
        """Performs complex expansion on 'self' and returns a tuple
           containing collected both real and imaginary parts. This
           method cannot be confused with re() and im() functions,
           which does not perform complex expansion at evaluation.

           However it is possible to expand both re() and im()
           functions and get exactly the same results as with
           a single call to this function.

           >>> from sympy import symbols, I

           >>> x, y = symbols('x,y', real=True)

           >>> (x + y*I).as_real_imag()
           (x, y)

           >>> from sympy.abc import z, w

           >>> (z + w*I).as_real_imag()
           (re(z) - im(w), re(w) + im(z))

        """
        # 如果 'ignore' 在 hints 中并且等于 self，则返回 None
        if hints.get('ignore') == self:
            return None
        else:
            # 导入 sympy 库中的复数部分函数 re 和 im
            from sympy.functions.elementary.complexes import im, re
            # 返回一个包含 self 实部和虚部的元组
            return (re(self), im(self))

    def as_powers_dict(self):
        """Return self as a dictionary of factors with each factor being
        treated as a power. The keys are the bases of the factors and the
        values, the corresponding exponents. The resulting dictionary should
        be used with caution if the expression is a Mul and contains non-
        commutative factors since the order that they appeared will be lost in
        the dictionary.

        See Also
        ========
        as_ordered_factors: An alternative for noncommutative applications,
                            returning an ordered list of factors.
        args_cnc: Similar to as_ordered_factors, but guarantees separation
                  of commutative and noncommutative factors.
        """
        # 创建一个默认字典对象 d，所有值的默认类型为 int
        d = defaultdict(int)
        # 更新字典 d，将 self 视为一个因子并将其作为一个基数和指数对存储在字典中
        d.update([self.as_base_exp()])
        # 返回构建好的字典 d
        return d
    def as_coefficients_dict(self, *syms):
        """
        Return a dictionary mapping terms to their Rational coefficient.
        Since the dictionary is a defaultdict, inquiries about terms which
        were not present will return a coefficient of 0.

        If symbols `syms` are provided, any multiplicative terms
        independent of them will be considered a coefficient and a
        regular dictionary of syms-dependent generators as keys and
        their corresponding coefficients as values will be returned.

        Examples
        ========

        >>> from sympy.abc import a, x, y
        >>> (3*x + a*x + 4).as_coefficients_dict()
        {1: 4, x: 3, a*x: 1}
        >>> _[a]
        0
        >>> (3*a*x).as_coefficients_dict()
        {a*x: 3}
        >>> (3*a*x).as_coefficients_dict(x)
        {x: 3*a}
        >>> (3*a*x).as_coefficients_dict(y)
        {1: 3*a*x}
        """
        # Create a defaultdict to store coefficients associated with terms
        d = defaultdict(list)
        
        # If no symbols provided, process all terms in the expression
        if not syms:
            # Break down the expression into additive components
            for ai in Add.make_args(self):
                # Separate coefficient and term
                c, m = ai.as_coeff_Mul()
                # Append coefficient to the list associated with term m
                d[m].append(c)
            # Consolidate lists into coefficients where applicable
            for k, v in d.items():
                if len(v) == 1:
                    d[k] = v[0]
                else:
                    d[k] = Add(*v)
        else:
            # Separate the expression into independent and dependent parts based on symbols
            ind, dep = self.as_independent(*syms, as_Add=True)
            # Process dependent part into multiplicative components
            for i in Add.make_args(dep):
                if i.is_Mul:
                    # Separate coefficient and symbols from the term
                    c, x = i.as_coeff_mul(*syms)
                    if c is S.One:
                        d[i].append(c)
                    else:
                        d[i._new_rawargs(*x)].append(c)
                elif i:
                    # Treat as a standalone term with coefficient 1
                    d[i].append(S.One)
            # Consolidate lists into coefficients where applicable
            d = {k: Add(*d[k]) for k in d}
            # Include independent part in the coefficient dictionary if nonzero
            if ind is not S.Zero:
                d.update({S.One: ind})
        
        # Convert to a regular dictionary and return
        di = defaultdict(int)
        di.update(d)
        return di

    def as_base_exp(self) -> tuple[Expr, Expr]:
        """
        Return the expression itself and the exponent 1, representing
        the expression in the form a -> b ** e.

        Examples
        ========

        >>> x.as_base_exp()
        (x, 1)
        """
        # Return the expression and the exponent 1
        return self, S.One
    def as_coeff_mul(self, *deps, **kwargs) -> tuple[Expr, tuple[Expr, ...]]:
        """Return the tuple (c, args) where self is written as a Mul, ``m``.

        c should be a Rational multiplied by any factors of the Mul that are
        independent of deps.

        args should be a tuple of all other factors of m; args is empty
        if self is a Number or if self is independent of deps (when given).

        This should be used when you do not know if self is a Mul or not but
        you want to treat self as a Mul or if you want to process the
        individual arguments of the tail of self as a Mul.

        - if you know self is a Mul and want only the head, use self.args[0];
        - if you do not want to process the arguments of the tail but need the
          tail then use self.as_two_terms() which gives the head and tail;
        - if you want to split self into an independent and dependent parts
          use ``self.as_independent(*deps)``

        >>> from sympy import S
        >>> from sympy.abc import x, y
        >>> (S(3)).as_coeff_mul()
        (3, ())
        >>> (3*x*y).as_coeff_mul()
        (3, (x, y))
        >>> (3*x*y).as_coeff_mul(x)
        (3*y, (x,))
        >>> (3*y).as_coeff_mul(x)
        (3*y, ())
        """
        # 如果给定了依赖项（deps），检查 self 是否包含这些依赖项，若不包含则返回 (self, ())
        if deps:
            if not self.has(*deps):
                return self, ()
        # 如果没有依赖项或者 self 包含了所有的依赖项，则返回 (1, (self,))
        return S.One, (self,)

    def as_coeff_add(self, *deps) -> tuple[Expr, tuple[Expr, ...]]:
        """Return the tuple (c, args) where self is written as an Add, ``a``.

        c should be a Rational added to any terms of the Add that are
        independent of deps.

        args should be a tuple of all other terms of ``a``; args is empty
        if self is a Number or if self is independent of deps (when given).

        This should be used when you do not know if self is an Add or not but
        you want to treat self as an Add or if you want to process the
        individual arguments of the tail of self as an Add.

        - if you know self is an Add and want only the head, use self.args[0];
        - if you do not want to process the arguments of the tail but need the
          tail then use self.as_two_terms() which gives the head and tail.
        - if you want to split self into an independent and dependent parts
          use ``self.as_independent(*deps)``

        >>> from sympy import S
        >>> from sympy.abc import x, y
        >>> (S(3)).as_coeff_add()
        (3, ())
        >>> (3 + x).as_coeff_add()
        (3, (x,))
        >>> (3 + x + y).as_coeff_add(x)
        (y + 3, (x,))
        >>> (3 + y).as_coeff_add(x)
        (y + 3, ())
        """
        # 如果给定了依赖项（deps），检查 self 是否自由于这些依赖项，若不自由则返回 (self, ())
        if deps:
            if not self.has_free(*deps):
                return self, ()
        # 如果没有依赖项或者 self 自由于所有的依赖项，则返回 (0, (self,))
        return S.Zero, (self,)
    def primitive(self):
        """
        从 self 的每个项中提取出一个正有理数，该方法不使用递归。这类似于 as_coeff_Mul() 方法，
        但是 primitive 总是提取一个正有理数（而不是负数或浮点数）。

        Examples
        ========

        >>> from sympy.abc import x
        >>> (3*(x + 1)**2).primitive()
        (3, (x + 1)**2)
        >>> a = (6*x + 2); a.primitive()
        (2, 3*x + 1)
        >>> b = (x/2 + 3); b.primitive()
        (1/2, x + 6)
        >>> (a*b).primitive() == (1, a*b)
        True
        """
        if not self:
            # 如果 self 是零，则返回 (1, 0)
            return S.One, S.Zero
        c, r = self.as_coeff_Mul(rational=True)
        if c.is_negative:
            # 如果 c 是负数，则取其相反数
            c, r = -c, -r
        # 返回 c（正有理数）和 r（剩余部分）
        return c, r

    def as_content_primitive(self, radical=False, clear=True):
        """
        该方法应该递归地从所有参数中移除一个有理数，并返回该有理数（内容）和新的 self（基本部分）。
        内容应该始终为正，并且 Mul(*foo.as_content_primitive()) == foo 应成立。
        基本部分不需要处于规范形式，并且应尽可能保留底层结构（即不应用于 self 的 expand_mul）。

        Examples
        ========

        >>> from sympy import sqrt
        >>> from sympy.abc import x, y, z

        >>> eq = 2 + 2*x + 2*y*(3 + 3*y)

        as_content_primitive 函数是递归的，并保留结构：

        >>> eq.as_content_primitive()
        (2, x + 3*y*(y + 1) + 1)

        整数幂将从基数中提取有理数：

        >>> ((2 + 6*x)**2).as_content_primitive()
        (4, (3*x + 1)**2)
        >>> ((2 + 6*x)**(2*y)).as_content_primitive()
        (1, (2*(3*x + 1))**(2*y))

        项在它们的 as_content_primitives 加入后可能合并：

        >>> ((5*(x*(1 + y)) + 2*x*(3 + 3*y))).as_content_primitive()
        (11, x*(y + 1))
        >>> ((3*(x*(1 + y)) + 2*x*(3 + 3*y))).as_content_primitive()
        (9, x*(y + 1))
        >>> ((3*(z*(1 + y)) + 2.0*x*(3 + 3*y))).as_content_primitive()
        (1, 6.0*x*(y + 1) + 3*z*(y + 1))
        >>> ((5*(x*(1 + y)) + 2*x*(3 + 3*y))**2).as_content_primitive()
        (121, x**2*(y + 1)**2)
        >>> ((x*(1 + y) + 0.4*x*(3 + 3*y))**2).as_content_primitive()
        (1, 4.84*x**2*(y + 1)**2)

        根式内容也可以从基本部分中提取出来：

        >>> (2*sqrt(2) + 4*sqrt(10)).as_content_primitive(radical=True)
        (2, sqrt(2)*(1 + 2*sqrt(5)))

        如果 clear=False（默认为 True），则如果可以将内容分布到留下一个或多个具有整数系数的项，
        则不会从 Add 中移除内容。

        >>> (x/2 + y).as_content_primitive()
        (1/2, x + 2*y)
        >>> (x/2 + y).as_content_primitive(clear=False)
        (1, x/2 + y)
        """
        return S.One, self
    # 定义一个方法，返回表达式的分子和分母
    def as_numer_denom(self):
        """Return the numerator and the denominator of an expression.

        expression -> a/b -> a, b

        This is just a stub that should be defined by
        an object's class methods to get anything else.

        See Also
        ========

        normal: return ``a/b`` instead of ``(a, b)``

        """
        # 返回自身表达式和分母为1的符号
        return self, S.One

    # 定义一个方法，将表达式返回为分数形式
    def normal(self):
        """Return the expression as a fraction.

        expression -> a/b

        See Also
        ========

        as_numer_denom: return ``(a, b)`` instead of ``a/b``

        """
        # 从.mul模块导入_unevaluated_Mul函数
        from .mul import _unevaluated_Mul
        # 调用as_numer_denom方法获取分子和分母
        n, d = self.as_numer_denom()
        # 如果分母为1，直接返回分子
        if d is S.One:
            return n
        # 如果分母是一个数值，返回n乘以1/d的乘法操作
        if d.is_Number:
            return _unevaluated_Mul(n, 1/d)
        # 否则返回n除以d的除法操作
        else:
            return n/d
    # 定义一个方法，用于从 self 中减去 c，使所有匹配的系数朝零方向移动，如果不能执行减法操作则返回 None
    def extract_additively(self, c):
        """
        Return self - c if it's possible to subtract c from self and
        make all matching coefficients move towards zero, else return None.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> e = 2*x + 3
        >>> e.extract_additively(x + 1)
        x + 2
        >>> e.extract_additively(3*x)
        >>> e.extract_additively(4)
        >>> (y*(x + 1)).extract_additively(x + 1)
        >>> ((x + 1)*(x + 2*y + 1) + 3).extract_additively(x + 1)
        (x + 1)*(x + 2*y) + 3

        See Also
        ========
        extract_multiplicatively
        coeff
        as_coefficient
        """

        # 将 c 转换为 sympy 表达式
        c = sympify(c)
        
        # 如果 self 是 NaN，则返回 None
        if self is S.NaN:
            return None
        
        # 如果 c 是零，则返回 self
        if c.is_zero:
            return self
        
        # 如果 c 和 self 相等，则返回零
        elif c == self:
            return S.Zero
        
        # 如果 self 是零，则返回 None
        elif self == S.Zero:
            return None
        
        # 如果 self 是一个数值
        if self.is_Number:
            # 如果 c 不是数值，则返回 None
            if not c.is_Number:
                return None
            co = self
            diff = co - c
            # 判断是否可以从 co 中减去 c，并且差值 diff 在正确的范围内
            if (co > 0 and diff >= 0 and diff < co or
                    co < 0 and diff <= 0 and diff > co):
                return diff
            return None
        
        # 如果 c 是一个数值
        if c.is_Number:
            # 将 self 分解为常数项和非常数项
            co, t = self.as_coeff_Add()
            # 尝试从 co 中减去 c
            xa = co.extract_additively(c)
            if xa is None:
                return None
            return xa + t
        
        # 如果 c 是一个加法表达式并且其第一个元素是数值
        if c.is_Add and c.args[0].is_Number:
            # 获取 self 关于 c 的系数
            co = self.coeff(c)
            # 计算整个项作为项因子的情况
            xa0 = (co.extract_additively(1) or 0)*c
            if xa0:
                diff = self - co*c
                return (xa0 + (diff.extract_additively(c) or diff)) or None
            # 逐项处理
            h, t = c.as_coeff_Add()
            sh, st = self.as_coeff_Add()
            xa = sh.extract_additively(h)
            if xa is None:
                return None
            xa2 = st.extract_additively(t)
            if xa2 is None:
                return None
            return xa + xa2
        
        # 整个项作为项因子的情况
        co, diff = _corem(self, c)
        xa0 = (co.extract_additively(1) or 0)*c
        if xa0:
            return (xa0 + (diff.extract_additively(c) or diff)) or None
        # 逐项处理
        coeffs = []
        for a in Add.make_args(c):
            ac, at = a.as_coeff_Mul()
            co = self.coeff(at)
            if not co:
                return None
            coc, cot = co.as_coeff_Add()
            xa = coc.extract_additively(ac)
            if xa is None:
                return None
            self -= co*at
            coeffs.append((cot + xa)*at)
        coeffs.append(self)
        return Add(*coeffs)
    # 返回一个集合，包含当前表达式节点中的自由符号（即未被限定范围的符号）
    def expr_free_symbols(self):
        """
        Like ``free_symbols``, but returns the free symbols only if
        they are contained in an expression node.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> (x + y).expr_free_symbols # doctest: +SKIP
        {x, y}

        If the expression is contained in a non-expression object, do not return
        the free symbols. Compare:

        >>> from sympy import Tuple
        >>> t = Tuple(x + y)
        >>> t.expr_free_symbols # doctest: +SKIP
        set()
        >>> t.free_symbols
        {x, y}
        """
        # 发出 SymPy 弃用警告，建议使用 free_symbols 来获取表达式的自由符号集合
        sympy_deprecation_warning("""
        The expr_free_symbols property is deprecated. Use free_symbols to get
        the free symbols of an expression.
        """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-expr-free-symbols")
        # 返回所有子节点中表达式的自由符号的集合
        return {j for i in self.args for j in i.expr_free_symbols}

    # 判断当前对象是否有 -1 作为首个因子，或者在求和中是否负号的数量多于正号的数量
    def could_extract_minus_sign(self):
        """Return True if self has -1 as a leading factor or has
        more literal negative signs than positive signs in a sum,
        otherwise False.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> e = x - y
        >>> {i.could_extract_minus_sign() for i in (e, -e)}
        {False, True}

        Though the ``y - x`` is considered like ``-(x - y)``, since it
        is in a product without a leading factor of -1, the result is
        false below:

        >>> (x*(y - x)).could_extract_minus_sign()
        False

        To put something in canonical form wrt to sign, use `signsimp`:

        >>> from sympy import signsimp
        >>> signsimp(x*(y - x))
        -x*(x - y)
        >>> _.could_extract_minus_sign()
        True
        """
        # 总是返回 False，表示当前对象不具备提取负号的条件
        return False
    def extract_branch_factor(self, allow_half=False):
        """
        Try to write self as ``exp_polar(2*pi*I*n)*z`` in a nice way.
        Return (z, n).

        >>> from sympy import exp_polar, I, pi
        >>> from sympy.abc import x, y
        >>> exp_polar(I*pi).extract_branch_factor()
        (exp_polar(I*pi), 0)
        >>> exp_polar(2*I*pi).extract_branch_factor()
        (1, 1)
        >>> exp_polar(-pi*I).extract_branch_factor()
        (exp_polar(I*pi), -1)
        >>> exp_polar(3*pi*I + x).extract_branch_factor()
        (exp_polar(x + I*pi), 1)
        >>> (y*exp_polar(-5*pi*I)*exp_polar(3*pi*I + 2*pi*x)).extract_branch_factor()
        (y*exp_polar(2*pi*x), -1)
        >>> exp_polar(-I*pi/2).extract_branch_factor()
        (exp_polar(-I*pi/2), 0)

        If allow_half is True, also extract exp_polar(I*pi):

        >>> exp_polar(I*pi).extract_branch_factor(allow_half=True)
        (1, 1/2)
        >>> exp_polar(2*I*pi).extract_branch_factor(allow_half=True)
        (1, 1)
        >>> exp_polar(3*I*pi).extract_branch_factor(allow_half=True)
        (1, 3/2)
        >>> exp_polar(-I*pi).extract_branch_factor(allow_half=True)
        (1, -1/2)
        """
        from sympy.functions.elementary.exponential import exp_polar
        from sympy.functions.elementary.integers import ceiling

        n = S.Zero  # 初始化整数部分 n 为零
        res = S.One  # 初始化结果 res 为一
        args = Mul.make_args(self)  # 将表达式 self 分解为乘法操作数列表
        exps = []  # 初始化指数列表

        # 遍历每个乘法操作数
        for arg in args:
            if isinstance(arg, exp_polar):
                exps += [arg.exp]  # 如果是指数函数，将其指数加入到列表中
            else:
                res *= arg  # 否则将其乘到结果中

        piimult = S.Zero  # 初始化 pi 的倍数为零
        extras = []  # 初始化额外项列表

        ipi = S.Pi * S.ImaginaryUnit  # 计算虚数单位乘以 π

        # 处理指数列表
        while exps:
            exp = exps.pop()
            if exp.is_Add:
                exps += exp.args
                continue
            if exp.is_Mul:
                coeff = exp.as_coefficient(ipi)
                if coeff is not None:
                    piimult += coeff
                    continue
            extras += [exp]

        # 处理 pi 的系数
        if piimult.is_number:
            coeff = piimult
            tail = ()
        else:
            coeff, tail = piimult.as_coeff_add(*piimult.free_symbols)

        # 向下取整到最接近的偶数倍数
        branchfact = ceiling(coeff / 2 - S.Half) * 2
        n += branchfact / 2
        c = coeff - branchfact

        # 如果允许半整数倍数
        if allow_half:
            nc = c.extract_additively(1)
            if nc is not None:
                n += S.Half
                c = nc

        # 计算新的指数表达式
        newexp = ipi * Add(*((c, ) + tail)) + Add(*extras)

        # 如果新的指数非零，则乘到结果中
        if newexp != 0:
            res *= exp_polar(newexp)

        return res, n
    def is_polynomial(self, *syms):
        r"""
        Return True if self is a polynomial in syms and False otherwise.

        This checks if self is an exact polynomial in syms.  This function
        returns False for expressions that are "polynomials" with symbolic
        exponents.  Thus, you should be able to apply polynomial algorithms to
        expressions for which this returns True, and Poly(expr, *syms) should
        work if and only if expr.is_polynomial(*syms) returns True. The
        polynomial does not have to be in expanded form.  If no symbols are
        given, all free symbols in the expression will be used.

        This is not part of the assumptions system.  You cannot do
        Symbol('z', polynomial=True).

        Examples
        ========

        >>> from sympy import Symbol, Function
        >>> x = Symbol('x')
        >>> ((x**2 + 1)**4).is_polynomial(x)
        True
        >>> ((x**2 + 1)**4).is_polynomial()
        True
        >>> (2**x + 1).is_polynomial(x)
        False
        >>> (2**x + 1).is_polynomial(2**x)
        True
        >>> f = Function('f')
        >>> (f(x) + 1).is_polynomial(x)
        False
        >>> (f(x) + 1).is_polynomial(f(x))
        True
        >>> (1/f(x) + 1).is_polynomial(f(x))
        False

        >>> n = Symbol('n', nonnegative=True, integer=True)
        >>> (x**n + 1).is_polynomial(x)
        False

        This function does not attempt any nontrivial simplifications that may
        result in an expression that does not appear to be a polynomial to
        become one.

        >>> from sympy import sqrt, factor, cancel
        >>> y = Symbol('y', positive=True)
        >>> a = sqrt(y**2 + 2*y + 1)
        >>> a.is_polynomial(y)
        False
        >>> factor(a)
        y + 1
        >>> factor(a).is_polynomial(y)
        True

        >>> b = (y**2 + 2*y + 1)/(y + 1)
        >>> b.is_polynomial(y)
        False
        >>> cancel(b)
        y + 1
        >>> cancel(b).is_polynomial(y)
        True

        See also .is_rational_function()

        """
        # 如果给定了符号参数，将它们转换为符号表达式的集合；否则使用表达式中的所有自由符号
        if syms:
            syms = set(map(sympify, syms))
        else:
            syms = self.free_symbols
            # 如果没有自由符号，则表达式本身即为多项式
            if not syms:
                return True
        
        # 调用实例方法 _eval_is_polynomial 来判断是否是多项式
        return self._eval_is_polynomial(syms)

    def _eval_is_polynomial(self, syms):
        # 如果表达式本身是符号集合中的一个符号，则是多项式
        if self in syms:
            return True
        # 如果表达式不包含任何符号，则是常数多项式
        if not self.has_free(*syms):
            # constant polynomial
            return True
        # 子类应该返回 True 或 False，表示是否是多项式
    # 检测当前表达式是否为有理函数，即两个多项式的比值，其中多项式的变量为给定的符号集合 syms。
    def is_rational_function(self, *syms):
        """
        Test whether function is a ratio of two polynomials in the given
        symbols, syms. When syms is not given, all free symbols will be used.
        The rational function does not have to be in expanded or in any kind of
        canonical form.

        This function returns False for expressions that are "rational
        functions" with symbolic exponents.  Thus, you should be able to call
        .as_numer_denom() and apply polynomial algorithms to the result for
        expressions for which this returns True.

        This is not part of the assumptions system.  You cannot do
        Symbol('z', rational_function=True).

        Examples
        ========

        >>> from sympy import Symbol, sin
        >>> from sympy.abc import x, y

        >>> (x/y).is_rational_function()
        True

        >>> (x**2).is_rational_function()
        True

        >>> (x/sin(y)).is_rational_function(y)
        False

        >>> n = Symbol('n', integer=True)
        >>> (x**n + 1).is_rational_function(x)
        False

        This function does not attempt any nontrivial simplifications that may
        result in an expression that does not appear to be a rational function
        to become one.

        >>> from sympy import sqrt, factor
        >>> y = Symbol('y', positive=True)
        >>> a = sqrt(y**2 + 2*y + 1)/y
        >>> a.is_rational_function(y)
        False
        >>> factor(a)
        (y + 1)/y
        >>> factor(a).is_rational_function(y)
        True

        See also is_algebraic_expr().

        """
        # 如果给定了符号集合 syms，则将其转换为符号对象的集合；否则使用表达式中的所有自由符号。
        if syms:
            syms = set(map(sympify, syms))
        else:
            syms = self.free_symbols
            # 如果表达式中没有自由符号，则检查当前表达式是否在 _illegal 中，若不在则返回 True。
            if not syms:
                return self not in _illegal

        # 调用 _eval_is_rational_function 方法检查当前表达式是否为有理函数。
        return self._eval_is_rational_function(syms)

    # 用于子类实现，判断当前表达式是否为有理函数，依据是表达式是否包含在给定的符号集合 syms 中或表达式是否不含有给定符号集合 syms 的任何自由变量。
    def _eval_is_rational_function(self, syms):
        if self in syms:
            return True
        if not self.has_xfree(syms):
            return True
        # 子类应返回 True 或 False
    def is_meromorphic(self, x, a):
        """
        This tests whether an expression is meromorphic as
        a function of the given symbol ``x`` at the point ``a``.

        This method is intended as a quick test that will return
        None if no decision can be made without simplification or
        more detailed analysis.

        Examples
        ========

        >>> from sympy import zoo, log, sin, sqrt
        >>> from sympy.abc import x

        >>> f = 1/x**2 + 1 - 2*x**3
        >>> f.is_meromorphic(x, 0)
        True
        >>> f.is_meromorphic(x, 1)
        True
        >>> f.is_meromorphic(x, zoo)
        True

        >>> g = x**log(3)
        >>> g.is_meromorphic(x, 0)
        False
        >>> g.is_meromorphic(x, 1)
        True
        >>> g.is_meromorphic(x, zoo)
        False

        >>> h = sin(1/x)*x**2
        >>> h.is_meromorphic(x, 0)
        False
        >>> h.is_meromorphic(x, 1)
        True
        >>> h.is_meromorphic(x, zoo)
        True

        Multivalued functions are considered meromorphic when their
        branches are meromorphic. Thus most functions are meromorphic
        everywhere except at essential singularities and branch points.
        In particular, they will be meromorphic also on branch cuts
        except at their endpoints.

        >>> log(x).is_meromorphic(x, -1)
        True
        >>> log(x).is_meromorphic(x, 0)
        False
        >>> sqrt(x).is_meromorphic(x, -1)
        True
        >>> sqrt(x).is_meromorphic(x, 0)
        False

        """
        # 检查参数 x 是否为符号类型，否则引发类型错误
        if not x.is_symbol:
            raise TypeError("{} should be of symbol type".format(x))
        # 将参数 a 转换为 SymPy 表达式
        a = sympify(a)

        # 调用内部方法进行实际的 meromorphic 测试
        return self._eval_is_meromorphic(x, a)

    def _eval_is_meromorphic(self, x, a):
        # 如果当前表达式是 x 自身，则认为是 meromorphic
        if self == x:
            return True
        # 如果当前表达式不含自由变量 x，则认为是 meromorphic
        if not self.has_free(x):
            return True
        # 子类应该返回 True 或 False
        # 这里并未具体实现，留给子类来决定是否 meromorphic
        # （在该类中，_eval_is_meromorphic 方法的实现应该由子类提供）
    def is_algebraic_expr(self, *syms):
        """
        This tests whether a given expression is algebraic or not, in the
        given symbols, syms. When syms is not given, all free symbols
        will be used. The rational function does not have to be in expanded
        or in any kind of canonical form.

        This function returns False for expressions that are "algebraic
        expressions" with symbolic exponents. This is a simple extension to the
        is_rational_function, including rational exponentiation.

        Examples
        ========

        >>> from sympy import Symbol, sqrt
        >>> x = Symbol('x', real=True)
        >>> sqrt(1 + x).is_rational_function()
        False
        >>> sqrt(1 + x).is_algebraic_expr()
        True

        This function does not attempt any nontrivial simplifications that may
        result in an expression that does not appear to be an algebraic
        expression to become one.

        >>> from sympy import exp, factor
        >>> a = sqrt(exp(x)**2 + 2*exp(x) + 1)/(exp(x) + 1)
        >>> a.is_algebraic_expr(x)
        False
        >>> factor(a).is_algebraic_expr()
        True

        See Also
        ========

        is_rational_function

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Algebraic_expression

        """
        # 如果给定了符号参数syms，则将其转换为符号集合
        if syms:
            syms = set(map(sympify, syms))
        else:
            # 否则使用自由符号集合，如果为空则直接返回True
            syms = self.free_symbols
            if not syms:
                return True

        # 调用内部方法_eval_is_algebraic_expr进行实际的代数表达式检测
        return self._eval_is_algebraic_expr(syms)

    def _eval_is_algebraic_expr(self, syms):
        # 如果自身是syms集合中的符号之一，则返回True
        if self in syms:
            return True
        # 如果表达式不含有syms中的任何自由符号，则返回True
        if not self.has_free(*syms):
            return True
        # 子类应该返回True或False
    # 对象方法，返回一个迭代器，用于生成级数的每一项
    def lseries(self, x=None, x0=0, dir='+', logx=None, cdir=0):
        """
        Wrapper for series yielding an iterator of the terms of the series.

        Note: an infinite series will yield an infinite iterator. The following,
        for exaxmple, will never terminate. It will just keep printing terms
        of the sin(x) series::

          for term in sin(x).lseries(x):
              print term

        The advantage of lseries() over nseries() is that many times you are
        just interested in the next term in the series (i.e. the first term for
        example), but you do not know how many you should ask for in nseries()
        using the "n" parameter.

        See also nseries().
        """
        # 调用 series 方法，返回级数的迭代器，不限制项数（n=None）
        return self.series(x, x0, n=None, dir=dir, logx=logx, cdir=cdir)

    # 对象方法，用于生成级数的每一项的默认实现，基于 nseries() 方法的自适应增加 "n" 的方式
    def _eval_lseries(self, x, logx=None, cdir=0):
        # 初始化项数为 0
        n = 0
        # 初始级数为通过 _eval_nseries 方法计算的结果，使用 n=0
        series = self._eval_nseries(x, n=n, logx=logx, cdir=cdir)

        # 当结果是 Order 类型时，增加 n 的值，重新计算级数
        while series.is_Order:
            n += 1
            series = self._eval_nseries(x, n=n, logx=logx, cdir=cdir)

        # 移除 Order 部分，生成第一项并返回
        e = series.removeO()
        yield e
        # 如果第一项是零，则直接返回
        if e is S.Zero:
            return

        # 开始生成剩余的项
        while 1:
            while 1:
                # 增加 n 的值，重新计算级数并移除 Order 部分
                n += 1
                series = self._eval_nseries(x, n=n, logx=logx, cdir=cdir).removeO()
                # 如果当前项不等于前一项 e，则结束内部循环
                if e != series:
                    break
                # 如果级数减去自身化简后为零，则直接返回
                if (series - self).cancel() is S.Zero:
                    return
            # 生成当前项与前一项的差值，并更新前一项 e
            yield series - e
            e = series
    def nseries(self, x=None, x0=0, n=6, dir='+', logx=None, cdir=0):
        """
        Wrapper to _eval_nseries if assumptions allow, else to series.

        If x is given, x0 is 0, dir='+', and self has x, then _eval_nseries is
        called. This calculates "n" terms in the innermost expressions and
        then builds up the final series just by "cross-multiplying" everything
        out.

        The optional ``logx`` parameter can be used to replace any log(x) in the
        returned series with a symbolic value to avoid evaluating log(x) at 0. A
        symbol to use in place of log(x) should be provided.

        Advantage -- it's fast, because we do not have to determine how many
        terms we need to calculate in advance.

        Disadvantage -- you may end up with less terms than you may have
        expected, but the O(x**n) term appended will always be correct and
        so the result, though perhaps shorter, will also be correct.

        If any of those assumptions is not met, this is treated like a
        wrapper to series which will try harder to return the correct
        number of terms.

        See also lseries().

        Examples
        ========

        >>> from sympy import sin, log, Symbol
        >>> from sympy.abc import x, y
        >>> sin(x).nseries(x, 0, 6)
        x - x**3/6 + x**5/120 + O(x**6)
        >>> log(x+1).nseries(x, 0, 5)
        x - x**2/2 + x**3/3 - x**4/4 + O(x**5)

        Handling of the ``logx`` parameter --- in the following example the
        expansion fails since ``sin`` does not have an asymptotic expansion
        at -oo (the limit of log(x) as x approaches 0):

        >>> e = sin(log(x))
        >>> e.nseries(x, 0, 6)
        Traceback (most recent call last):
        ...
        PoleError: ...
        ...
        >>> logx = Symbol('logx')
        >>> e.nseries(x, 0, 6, logx=logx)
        sin(logx)

        In the following example, the expansion works but only returns self
        unless the ``logx`` parameter is used:

        >>> e = x**y
        >>> e.nseries(x, 0, 2)
        x**y
        >>> e.nseries(x, 0, 2, logx=logx)
        exp(logx*y)

        """
        # 如果 x 存在且不在自由符号中，则返回自身
        if x and x not in self.free_symbols:
            return self
        # 如果 x 为 None 或者 x0 不为 0 或者 dir 不为 '+'，则调用 series 方法进行级数展开
        if x is None or x0 or dir != '+':  # {see XPOS above} or (x.is_positive == x.is_negative == None):
            return self.series(x, x0, n, dir, cdir=cdir)
        else:
            # 否则调用 _eval_nseries 方法进行渐近级数展开
            return self._eval_nseries(x, n=n, logx=logx, cdir=cdir)
    # 定义一个方法 _eval_nseries，用于在 x=0 处从正方向计算自身的级数项，直到 O(x**n)
    def _eval_nseries(self, x, n, logx, cdir):
        """
        Return terms of series for self up to O(x**n) at x=0
        from the positive direction.

        This is a method that should be overridden in subclasses. Users should
        never call this method directly (use .nseries() instead), so you do not
        have to write docstrings for _eval_nseries().
        """
        # 抛出 NotImplementedError 异常，提示子类需要实现此方法以提供在 x=0 处从正方向到 O(x**n) 的级数项
        raise NotImplementedError(filldedent("""
                     The _eval_nseries method should be added to
                     %s to give terms up to O(x**n) at x=0
                     from the positive direction so it is available when
                     nseries calls it.""" % self.func)
                     )

    # 定义一个计算极限的方法，求解 x->xlim 时的极限
    def limit(self, x, xlim, dir='+'):
        """ Compute limit x->xlim.
        """
        # 导入 limit 函数并调用，返回极限值
        from sympy.series.limits import limit
        return limit(self, x, xlim, dir)

    # 定义一个计算主导级数项的方法（已废弃）
    def compute_leading_term(self, x, logx=None):
        """Deprecated function to compute the leading term of a series.

        as_leading_term is only allowed for results of .series()
        This is a wrapper to compute a series first.
        """
        # 导入 SymPyDeprecationWarning 并发出警告，提示该函数已废弃
        from sympy.utilities.exceptions import SymPyDeprecationWarning

        SymPyDeprecationWarning(
            feature="compute_leading_term",
            useinstead="as_leading_term",
            issue=21843,
            deprecated_since_version="1.12"
        ).warn()

        # 如果表达式包含 Piecewise 函数，则对其进行处理
        from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold
        if self.has(Piecewise):
            expr = piecewise_fold(self)
        else:
            expr = self

        # 如果移除高阶项后结果为 0，则直接返回原表达式
        if self.removeO() == 0:
            return self

        # 导入 Dummy 符号和 log 函数
        from .symbol import Dummy
        from sympy.functions.elementary.exponential import log
        from sympy.series.order import Order

        # 备份 logx 到 _logx
        _logx = logx
        # 如果 logx 未指定，则创建一个新的 Dummy 符号
        logx = Dummy('logx') if logx is None else logx
        # 初始化结果为 Order(1)
        res = Order(1)
        incr = S.One
        # 循环求解直到结果不再是 Order 类型
        while res.is_Order:
            # 调用 _eval_nseries 方法计算当前表达式的级数项，并进行简化处理
            res = expr._eval_nseries(x, n=1+incr, logx=logx).cancel().powsimp().trigsimp()
            # 增加指数 incr 以便下一次迭代
            incr *= 2

        # 如果原始 logx 未指定，则将结果中的 logx 替换为 log(x)
        if _logx is None:
            res = res.subs(logx, log(x))

        # 返回结果的主导级数项
        return res.as_leading_term(x)

    # 使用 cacheit 装饰器修饰下一个方法（未提供完整代码）
    @cacheit
    def as_leading_term(self, *symbols, logx=None, cdir=0):
        """
        Returns the leading (nonzero) term of the series expansion of self.

        The _eval_as_leading_term routines are used to do this, and they must
        always return a non-zero value.

        Examples
        ========

        >>> from sympy.abc import x
        >>> (1 + x + x**2).as_leading_term(x)
        1
        >>> (1/x**2 + x + x**2).as_leading_term(x)
        x**(-2)

        """
        # 处理多个符号参数的情况
        if len(symbols) > 1:
            c = self
            for x in symbols:
                c = c.as_leading_term(x, logx=logx, cdir=cdir)
            return c
        # 处理没有符号参数的情况
        elif not symbols:
            return self
        # 将第一个符号参数转换成符号表达式
        x = sympify(symbols[0])
        # 如果不是符号表达式，则引发异常
        if not x.is_symbol:
            raise ValueError('expecting a Symbol but got %s' % x)
        # 如果符号不在自由符号集合中，则返回自身
        if x not in self.free_symbols:
            return self
        # 调用私有方法 _eval_as_leading_term 获取主导项
        obj = self._eval_as_leading_term(x, logx=logx, cdir=cdir)
        # 如果主导项不为 None，则使用 powsimp 进行简化
        if obj is not None:
            from sympy.simplify.powsimp import powsimp
            return powsimp(obj, deep=True, combine='exp')
        # 如果主导项为 None，则引发未实现异常
        raise NotImplementedError('as_leading_term(%s, %s)' % (self, x))

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        """
        Returns the leading term of the series expansion of self with respect to x.

        This method should be overridden in derived classes.

        Parameters
        ----------
        x : Symbol
            The symbol with respect to which the leading term is computed.
        logx : Symbol, optional
            The logarithm of x, default is None.
        cdir : int, optional
            The direction in which the leading term is computed, default is 0.

        Returns
        -------
        Expr
            The leading term of the series expansion.

        """
        return self

    def as_coeff_exponent(self, x) -> tuple[Expr, Expr]:
        """ ``c*x**e -> c,e `` where x can be any symbolic expression. """
        # 导入函数 collect 用于收集关于 x 的表达式
        from sympy.simplify.radsimp import collect
        s = collect(self, x)
        # 将表达式 s 拆分为系数和幂次部分
        c, p = s.as_coeff_mul(x)
        # 如果幂次部分只有一个元素
        if len(p) == 1:
            # 将其拆分为基数和指数
            b, e = p[0].as_base_exp()
            # 如果基数和 x 相同，则返回系数和指数
            if b == x:
                return c, e
        # 如果没有匹配的情况，则返回原始表达式和零
        return s, S.Zero

    def leadterm(self, x, logx=None, cdir=0):
        """
        Returns the leading term a*x**b as a tuple (a, b).

        Examples
        ========

        >>> from sympy.abc import x
        >>> (1+x+x**2).leadterm(x)
        (1, 0)
        >>> (1/x**2+x+x**2).leadterm(x)
        (1, -2)

        """
        # 导入 Dummy 和 log 函数
        from .symbol import Dummy
        from sympy.functions.elementary.exponential import log
        # 获取主导项
        l = self.as_leading_term(x, logx=logx, cdir=cdir)
        # 创建一个虚拟符号 logx
        d = Dummy('logx')
        # 如果主导项包含 log(x)，则替换为虚拟符号
        if l.has(log(x)):
            l = l.subs(log(x), d)
        # 获取系数和指数
        c, e = l.as_coeff_exponent(x)
        # 如果系数中包含 x 的符号，则引发异常
        if x in c.free_symbols:
            raise ValueError(filldedent("""
                cannot compute leadterm(%s, %s). The coefficient
                should have been free of %s but got %s""" % (self, x, x, c)))
        # 将虚拟符号 logx 替换回 log(x) 并返回系数和指数
        c = c.subs(d, log(x))
        return c, e

    def as_coeff_Mul(self, rational: bool = False) -> tuple['Number', Expr]:
        """Efficiently extract the coefficient of a product."""
        # 返回默认系数为 1，表达式为自身
        return S.One, self

    def as_coeff_Add(self, rational=False) -> tuple['Number', Expr]:
        """Efficiently extract the coefficient of a summation."""
        # 返回默认系数为 0，表达式为自身
        return S.Zero, self
    def fps(self, x=None, x0=0, dir=1, hyper=True, order=4, rational=True,
            full=False):
        """
        Compute formal power series of self.

        See the docstring of the :func:`fps` function in sympy.series.formal for
        more information.
        """
        # 导入 sympy.series.formal 模块中的 fps 函数，并使用其计算形式幂级数
        from sympy.series.formal import fps

        # 调用 fps 函数计算形式幂级数并返回结果
        return fps(self, x, x0, dir, hyper, order, rational, full)

    def fourier_series(self, limits=None):
        """Compute fourier sine/cosine series of self.

        See the docstring of the :func:`fourier_series` in sympy.series.fourier
        for more information.
        """
        # 导入 sympy.series.fourier 模块中的 fourier_series 函数，并使用其计算傅里叶正弦/余弦级数
        from sympy.series.fourier import fourier_series

        # 调用 fourier_series 函数计算傅里叶级数并返回结果
        return fourier_series(self, limits)

    ###################################################################################
    ##################### DERIVATIVE, INTEGRAL, FUNCTIONAL METHODS ####################
    ###################################################################################

    def diff(self, *symbols, **assumptions):
        assumptions.setdefault("evaluate", True)
        # 调用 _derivative_dispatch 函数处理 self 对象的微分，传入 symbols 和 assumptions 参数
        return _derivative_dispatch(self, *symbols, **assumptions)

    ###########################################################################
    ###################### EXPRESSION EXPANSION METHODS #######################
    ###########################################################################

    # Relevant subclasses should override _eval_expand_hint() methods.  See
    # the docstring of expand() for more info.

    def _eval_expand_complex(self, **hints):
        # 将表达式拆分为实部和虚部
        real, imag = self.as_real_imag(**hints)
        # 返回以复数形式扩展后的表达式
        return real + S.ImaginaryUnit*imag

    @staticmethod
    def _expand_hint(expr, hint, deep=True, **hints):
        """
        Helper for ``expand()``.  Recursively calls ``expr._eval_expand_hint()``.

        Returns ``(expr, hit)``, where expr is the (possibly) expanded
        ``expr`` and ``hit`` is ``True`` if ``expr`` was truly expanded and
        ``False`` otherwise.
        """
        hit = False
        # 如果 deep 为真且 expr 有 args 属性且不是原子表达式，则递归调用 _expand_hint 函数
        if deep and getattr(expr, 'args', ()) and not expr.is_Atom:
            sargs = []
            for arg in expr.args:
                arg, arghit = Expr._expand_hint(arg, hint, **hints)
                hit |= arghit
                sargs.append(arg)

            if hit:
                expr = expr.func(*sargs)

        # 如果 expr 具有 hint 方法，则调用该方法进行表达式扩展
        if hasattr(expr, hint):
            newexpr = getattr(expr, hint)(**hints)
            # 如果扩展后的表达式与原表达式不同，则返回新表达式和 True
            if newexpr != expr:
                return (newexpr, True)

        # 返回扩展后的表达式及扩展是否命中的标记
        return (expr, hit)

    @cacheit
    ###########################################################################
    ################### GLOBAL ACTION VERB WRAPPER METHODS ####################
    ###########################################################################
    def integrate(self, *args, **kwargs):
        """
        调用 sympy.integrals 模块中的 integrate 函数，对当前对象进行积分操作。
        """
        from sympy.integrals.integrals import integrate
        return integrate(self, *args, **kwargs)

    def nsimplify(self, constants=(), tolerance=None, full=False):
        """
        调用 sympy.simplify 模块中的 nsimplify 函数，对当前对象进行符号简化操作。
        """
        from sympy.simplify.simplify import nsimplify
        return nsimplify(self, constants, tolerance, full)

    def separate(self, deep=False, force=False):
        """
        调用当前模块下的 expand_power_base 函数，用于展开或分离幂函数的基数。
        """
        from .function import expand_power_base
        return expand_power_base(self, deep=deep, force=force)

    def collect(self, syms, func=None, evaluate=True, exact=False, distribute_order_term=True):
        """
        调用 sympy.simplify 模块中的 collect 函数，用于收集表达式中的指定符号。
        """
        from sympy.simplify.radsimp import collect
        return collect(self, syms, func, evaluate, exact, distribute_order_term)

    def together(self, *args, **kwargs):
        """
        调用 sympy.polys 模块中的 together 函数，用于将有理表达式合并为一个分数。
        """
        from sympy.polys.rationaltools import together
        return together(self, *args, **kwargs)

    def apart(self, x=None, **args):
        """
        调用 sympy.polys 模块中的 apart 函数，用于对有理函数进行部分分式分解。
        """
        from sympy.polys.partfrac import apart
        return apart(self, x, **args)

    def ratsimp(self):
        """
        调用 sympy.simplify 模块中的 ratsimp 函数，用于对有理表达式进行有理化简。
        """
        from sympy.simplify.ratsimp import ratsimp
        return ratsimp(self)

    def trigsimp(self, **args):
        """
        调用 sympy.simplify 模块中的 trigsimp 函数，用于三角函数表达式的简化。
        """
        from sympy.simplify.trigsimp import trigsimp
        return trigsimp(self, **args)

    def radsimp(self, **kwargs):
        """
        调用 sympy.simplify 模块中的 radsimp 函数，用于有理表达式的根式化简。
        """
        from sympy.simplify.radsimp import radsimp
        return radsimp(self, **kwargs)

    def powsimp(self, *args, **kwargs):
        """
        调用 sympy.simplify 模块中的 powsimp 函数，用于幂函数表达式的简化。
        """
        from sympy.simplify.powsimp import powsimp
        return powsimp(self, *args, **kwargs)

    def combsimp(self):
        """
        调用 sympy.simplify 模块中的 combsimp 函数，用于组合数学表达式的简化。
        """
        from sympy.simplify.combsimp import combsimp
        return combsimp(self)

    def gammasimp(self):
        """
        调用 sympy.simplify 模块中的 gammasimp 函数，用于伽玛函数表达式的简化。
        """
        from sympy.simplify.gammasimp import gammasimp
        return gammasimp(self)

    def factor(self, *gens, **args):
        """
        调用 sympy.polys.polytools 模块中的 factor 函数，用于对多项式进行因式分解。
        """
        from sympy.polys.polytools import factor
        return factor(self, *gens, **args)

    def cancel(self, *gens, **args):
        """
        调用 sympy.polys 模块中的 cancel 函数，用于对多项式进行约分。
        """
        from sympy.polys.polytools import cancel
        return cancel(self, *gens, **args)
    # 定义一个方法，用于计算 self 对 g 取模后的乘法逆元
    # self 和 g 可能是符号表达式
    def invert(self, g, *gens, **args):
        """Return the multiplicative inverse of ``self`` mod ``g``
        where ``self`` (and ``g``) may be symbolic expressions).

        See Also
        ========
        sympy.core.intfunc.mod_inverse, sympy.polys.polytools.invert
        """
        # 如果 self 和 g 都是数值且没有符号部分，则直接调用 mod_inverse 函数
        if self.is_number and getattr(g, 'is_number', True):
            return mod_inverse(self, g)
        # 否则从 sympy.polys.polytools 导入 invert 函数，调用它计算结果
        from sympy.polys.polytools import invert
        return invert(self, g, *gens, **args)

    # 定义一个特殊方法 __round__，将其设置为 round 函数的别名
    __round__ = round

    # 定义一个方法，用于计算 self 对变量 x 的导数矩阵的行
    def _eval_derivative_matrix_lines(self, x):
        from sympy.matrices.expressions.matexpr import _LeftRightArgs
        # 返回一个列表，列表中的元素是 _LeftRightArgs 对象，每个对象包含一个元素 1 和一个元素 1
        # higher 参数为调用 _eval_derivative 方法的结果
        return [_LeftRightArgs([S.One, S.One], higher=self._eval_derivative(x))]
class AtomicExpr(Atom, Expr):
    """
    A parent class for objects that are both atoms and Exprs.

    For example: Symbol, Number, Rational, Integer, ...
    But not: Add, Mul, Pow, ...
    """
    
    # 是否为数字（这里设为 False 表示不是数字）
    is_number = False
    # 是否为原子表达式（这里设为 True 表示是原子表达式）
    is_Atom = True

    # 使用 __slots__ 来优化内存使用，空元组表示不允许添加新属性
    __slots__ = ()

    # 求导数的方法，如果自身就是对应的符号 s，则返回 1；否则返回 0
    def _eval_derivative(self, s):
        if self == s:
            return S.One
        return S.Zero

    # 对象求 n 次导数的方法，根据不同类型的 s 返回不同的 Piecewise 对象
    def _eval_derivative_n_times(self, s, n):
        from .containers import Tuple
        from sympy.matrices.expressions.matexpr import MatrixExpr
        from sympy.matrices.matrixbase import MatrixBase
        
        # 如果 s 是 MatrixBase、Tuple、Iterable 或 MatrixExpr 类型，则调用父类方法
        if isinstance(s, (MatrixBase, Tuple, Iterable, MatrixExpr)):
            return super()._eval_derivative_n_times(s, n)
        
        from .relational import Eq
        from sympy.functions.elementary.piecewise import Piecewise
        
        # 如果 self 等于 s，则返回相应的 Piecewise 对象
        return Piecewise((self, Eq(n, 0)), (1, Eq(n, 1)), (0, True))

    # 判断对象是否为多项式
    def _eval_is_polynomial(self, syms):
        return True

    # 判断对象是否为有理函数
    def _eval_is_rational_function(self, syms):
        return self not in _illegal

    # 判断对象是否为亚纯函数
    def _eval_is_meromorphic(self, x, a):
        from sympy.calculus.accumulationbounds import AccumBounds
        
        # 当对象不是数字或者是有限的时候，并且不是 AccumBounds 类型，则返回 True
        return (not self.is_Number or self.is_finite) and not isinstance(self, AccumBounds)

    # 判断对象是否为代数表达式
    def _eval_is_algebraic_expr(self, syms):
        return True

    # 对对象进行 nseries 展开，这里直接返回自身
    def _eval_nseries(self, x, n, logx, cdir=0):
        return self

    # 返回表达式中的自由符号集合，这个属性已经被废弃
    @property
    def expr_free_symbols(self):
        sympy_deprecation_warning("""
        The expr_free_symbols property is deprecated. Use free_symbols to get
        the free symbols of an expression.
        """,
        deprecated_since_version="1.9",
        active_deprecations_target="deprecated-expr-free-symbols")
        
        return {self}


def _mag(x):
    r"""Return integer $i$ such that $0.1 \le x/10^i < 1$

    Examples
    ========

    >>> from sympy.core.expr import _mag
    >>> from sympy import Float
    >>> _mag(Float(.1))
    0
    >>> _mag(Float(.01))
    -1
    >>> _mag(Float(1234))
    4
    """
    from math import log10, ceil, log
    xpos = abs(x.n())
    if not xpos:
        return S.Zero
    try:
        mag_first_dig = int(ceil(log10(xpos)))
    except (ValueError, OverflowError):
        mag_first_dig = int(ceil(Float(mpf_log(xpos._mpf_, 53))/log(10)))
    
    # 检查是否 off by 1
    if (xpos/10**mag_first_dig) >= 1:
        assert 1 <= (xpos/10**mag_first_dig) < 10
        mag_first_dig += 1
    
    return mag_first_dig


class UnevaluatedExpr(Expr):
    """
    Expression that is not evaluated unless released.

    Examples
    ========

    >>> from sympy import UnevaluatedExpr
    >>> from sympy.abc import x
    >>> x*(1/x)
    1
    >>> x*UnevaluatedExpr(1/x)
    x*1/x

    """

    # 创建一个未求值的表达式对象
    def __new__(cls, arg, **kwargs):
        # 将参数转换为符号表达式
        arg = _sympify(arg)
        # 调用父类构造方法创建对象
        obj = Expr.__new__(cls, arg, **kwargs)
        return obj
    # 定义一个方法 `doit`，接受关键字参数 `hints`
    def doit(self, **hints):
        # 如果 `hints` 中存在键 "deep" 且其值为 True
        if hints.get("deep", True):
            # 调用 `self.args` 中第一个元素的 `doit` 方法，并传递 `hints` 参数
            return self.args[0].doit(**hints)
        else:
            # 如果 `hints` 中不存在键 "deep" 或其值不为 True，则返回 `self.args` 中第一个元素
            return self.args[0]
# 定义一个函数 unchanged，用于检查应用函数 func 到参数 args 是否未改变
# 可以代替断言语句 `assert foo == foo`。

def unchanged(func, *args):
    """Return True if `func` applied to the `args` is unchanged.
    Can be used instead of `assert foo == foo`.

    Examples
    ========

    >>> from sympy import Piecewise, cos, pi
    >>> from sympy.core.expr import unchanged
    >>> from sympy.abc import x

    >>> unchanged(cos, 1)  # instead of assert cos(1) == cos(1)
    True

    >>> unchanged(cos, pi)
    False

    Comparison of args uses the builtin capabilities of the object's
    arguments to test for equality so args can be defined loosely. Here,
    the ExprCondPair arguments of Piecewise compare as equal to the
    tuples that can be used to create the Piecewise:

    >>> unchanged(Piecewise, (x, x > 1), (0, True))
    True
    """
    
    # 调用 func 函数，并将 args 作为参数传递给它
    f = func(*args)
    # 检查函数应用后的结果是否未改变，即函数的类型和参数是否与原来的一致
    return f.func == func and f.args == args


# 定义一个 ExprBuilder 类，用于构建表达式
class ExprBuilder:
    def __init__(self, op, args=None, validator=None, check=True):
        # 如果 op 不是可调用对象，则抛出 TypeError 异常
        if not hasattr(op, "__call__"):
            raise TypeError("op {} needs to be callable".format(op))
        self.op = op
        # 如果 args 为 None，则初始化为空列表，否则使用提供的 args
        if args is None:
            self.args = []
        else:
            self.args = args
        self.validator = validator
        # 如果 validator 不为 None 且 check 为 True，则进行验证
        if (validator is not None) and check:
            self.validate()

    @staticmethod
    def _build_args(args):
        # 构建参数列表，如果参数是 ExprBuilder 对象，则调用其 build 方法
        return [i.build() if isinstance(i, ExprBuilder) else i for i in args]

    def validate(self):
        # 如果 validator 为 None，则直接返回
        if self.validator is None:
            return
        # 构建参数列表，并调用 validator 函数进行验证
        args = self._build_args(self.args)
        self.validator(*args)

    def build(self, check=True):
        # 构建参数列表
        args = self._build_args(self.args)
        # 如果 validator 存在且 check 为 True，则进行验证
        if self.validator and check:
            self.validator(*args)
        # 应用操作符 op 到参数列表 args，并返回结果
        return self.op(*args)

    def append_argument(self, arg, check=True):
        # 向参数列表 args 中添加新的参数 arg
        self.args.append(arg)
        # 如果 validator 存在且 check 为 True，则验证参数列表 args
        if self.validator and check:
            self.validate(*self.args)

    def __getitem__(self, item):
        # 如果 item 为 0，则返回操作符 op，否则返回 args[item-1]
        if item == 0:
            return self.op
        else:
            return self.args[item-1]

    def __repr__(self):
        # 返回构建后的表达式的字符串表示
        return str(self.build())

    def search_element(self, elem):
        # 在参数列表 args 中搜索元素 elem
        for i, arg in enumerate(self.args):
            if isinstance(arg, ExprBuilder):
                # 如果参数是 ExprBuilder 对象，则递归调用其 search_index 方法
                ret = arg.search_index(elem)
                if ret is not None:
                    return (i,) + ret
            elif id(arg) == id(elem):
                # 如果参数与 elem 的 id 相同，则返回其索引
                return (i,)
        # 如果未找到 elem，则返回 None
        return None
```