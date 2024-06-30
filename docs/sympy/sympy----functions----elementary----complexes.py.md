# `D:\src\scipysrc\sympy\sympy\functions\elementary\complexes.py`

```
# 导入所需模块中的类型注释模块
from typing import Tuple as tTuple

# 导入 sympy 库中的具体模块和类
from sympy.core import S, Add, Mul, sympify, Symbol, Dummy, Basic
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.function import (Function, Derivative, ArgumentIndexError,
    AppliedUndef, expand_mul)
from sympy.core.logic import fuzzy_not, fuzzy_or
from sympy.core.numbers import pi, I, oo
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise

###############################################################################
######################### REAL and IMAGINARY PARTS ############################
###############################################################################

# 定义一个函数类 re，表示返回表达式的实部
class re(Function):
    """
    Returns real part of expression. This function performs only
    elementary analysis and so it will fail to decompose properly
    more complicated expressions. If completely simplified result
    is needed then use ``Basic.as_real_imag()`` or perform complex
    expansion on instance of this function.

    Examples
    ========

    >>> from sympy import re, im, I, E, symbols
    >>> x, y = symbols('x y', real=True)
    >>> re(2*E)
    2*E
    >>> re(2*I + 17)
    17
    >>> re(2*I)
    0
    >>> re(im(x) + x*I + 2)
    2
    >>> re(5 + I + 2)
    7

    Parameters
    ==========

    arg : Expr
        Real or complex expression.

    Returns
    =======

    expr : Expr
        Real part of expression.

    See Also
    ========

    im
    """

    # 函数的参数 args 是一个元组，包含表达式对象
    args: tTuple[Expr]

    # 表示该函数返回的是扩展实数部分
    is_extended_real = True
    # 隐式地在复数投影上工作
    unbranched = True  # implicitly works on the projection to C
    # 非全纯的，存在奇点
    _singularities = True  # non-holomorphic

    # 定义类方法，可以进一步扩展
    @classmethod
    def eval(cls, arg):
        # 如果参数是 NaN，则返回 NaN
        if arg is S.NaN:
            return S.NaN
        # 如果参数是 ComplexInfinity，则返回 NaN
        elif arg is S.ComplexInfinity:
            return S.NaN
        # 如果参数是扩展实数，则直接返回该参数
        elif arg.is_extended_real:
            return arg
        # 如果参数是纯虚数或者虚部乘以虚数单位后是扩展实数，则返回零
        elif arg.is_imaginary or (I*arg).is_extended_real:
            return S.Zero
        # 如果参数是矩阵，则返回其实部
        elif arg.is_Matrix:
            return arg.as_real_imag()[0]
        # 如果参数是函数并且是共轭函数，则返回其实部
        elif arg.is_Function and isinstance(arg, conjugate):
            return re(arg.args[0])
        else:
            # 进行以下分类：包含、反转和排除
            included, reverted, excluded = [], [], []
            args = Add.make_args(arg)
            for term in args:
                coeff = term.as_coefficient(I)

                if coeff is not None:
                    if not coeff.is_extended_real:
                        reverted.append(coeff)
                elif not term.has(I) and term.is_extended_real:
                    excluded.append(term)
                else:
                    # 尝试进行高级展开。如果不可能，则不再尝试 re(arg)
                    real_imag = term.as_real_imag(ignore=arg)
                    if real_imag:
                        excluded.append(real_imag[0])
                    else:
                        included.append(term)

            if len(args) != len(included):
                # 将包含、反转和排除后的结果进行组合
                a, b, c = (Add(*xs) for xs in [included, reverted, excluded])

                # 返回组合后的结果
                return cls(a) - im(b) + c

    def as_real_imag(self, deep=True, **hints):
        """
        返回具有零虚部的实数部分。

        """
        return (self, S.Zero)

    def _eval_derivative(self, x):
        # 如果自变量是扩展实数或者参数的实部是扩展实数，则返回参数关于自变量的导数的实部
        if x.is_extended_real or self.args[0].is_extended_real:
            return re(Derivative(self.args[0], x, evaluate=True))
        # 如果自变量是纯虚数或者参数的实部是纯虚数，则返回参数关于自变量的导数的负虚部乘以虚数单位
        if x.is_imaginary or self.args[0].is_imaginary:
            return -I * im(Derivative(self.args[0], x, evaluate=True))

    def _eval_rewrite_as_im(self, arg, **kwargs):
        # 返回参数减去其虚部乘以虚数单位
        return self.args[0] - I * im(self.args[0])

    def _eval_is_algebraic(self):
        # 返回参数是否是代数的
        return self.args[0].is_algebraic

    def _eval_is_zero(self):
        # 参数是否为零，如果是纯虚数则不为零
        # is_imaginary 意味着非零
        return fuzzy_or([self.args[0].is_imaginary, self.args[0].is_zero])

    def _eval_is_finite(self):
        # 如果参数是有限的，则返回 True
        if self.args[0].is_finite:
            return True

    def _eval_is_complex(self):
        # 如果参数是有限的，则返回 True
        if self.args[0].is_finite:
            return True
class im(Function):
    """
    Returns imaginary part of expression. This function performs only
    elementary analysis and so it will fail to decompose properly more
    complicated expressions. If completely simplified result is needed then
    use ``Basic.as_real_imag()`` or perform complex expansion on instance of
    this function.

    Examples
    ========

    >>> from sympy import re, im, E, I
    >>> from sympy.abc import x, y
    >>> im(2*E)
    0
    >>> im(2*I + 17)
    2
    >>> im(x*I)
    re(x)
    >>> im(re(x) + y)
    im(y)
    >>> im(2 + 3*I)
    3

    Parameters
    ==========

    arg : Expr
        Real or complex expression.

    Returns
    =======

    expr : Expr
        Imaginary part of expression.

    See Also
    ========

    re
    """

    args: tTuple[Expr]  # 声明元组类型参数args，包含表达式Expr类型

    is_extended_real = True  # 设置属性is_extended_real为True，表示扩展实数
    unbranched = True  # 设置属性unbranched为True，暗含在复数投影上工作
    _singularities = True  # 设置属性_singularities为True，表示非全纯的

    @classmethod
    def eval(cls, arg):
        # 对参数arg进行评估，返回其虚部的表达式
        if arg is S.NaN:
            return S.NaN
        elif arg is S.ComplexInfinity:
            return S.NaN
        elif arg.is_extended_real:
            return S.Zero
        elif arg.is_imaginary or (I*arg).is_extended_real:
            return -I * arg
        elif arg.is_Matrix:
            return arg.as_real_imag()[1]
        elif arg.is_Function and isinstance(arg, conjugate):
            return -im(arg.args[0])
        else:
            included, reverted, excluded = [], [], []
            args = Add.make_args(arg)
            for term in args:
                coeff = term.as_coefficient(I)

                if coeff is not None:
                    if not coeff.is_extended_real:
                        reverted.append(coeff)
                    else:
                        excluded.append(coeff)
                elif term.has(I) or not term.is_extended_real:
                    # 尝试进行高级展开。如果不可能，则不要再尝试im(arg)
                    real_imag = term.as_real_imag(ignore=arg)
                    if real_imag:
                        excluded.append(real_imag[1])
                    else:
                        included.append(term)

            if len(args) != len(included):
                a, b, c = (Add(*xs) for xs in [included, reverted, excluded])

                return cls(a) + re(b) + c

    def as_real_imag(self, deep=True, **hints):
        """
        Return the imaginary part with a zero real part.

        """
        return (self, S.Zero)

    def _eval_derivative(self, x):
        # 对象的导数评估，返回表达式的虚部的导数
        if x.is_extended_real or self.args[0].is_extended_real:
            return im(Derivative(self.args[0], x, evaluate=True))
        if x.is_imaginary or self.args[0].is_imaginary:
            return -I \
                * re(Derivative(self.args[0], x, evaluate=True))
    # 将当前对象重写为一个复数，使用实部和虚部的表达式
    def _eval_rewrite_as_re(self, arg, **kwargs):
        return -I*(self.args[0] - re(self.args[0]))
    
    # 判断当前对象的参数是否代数的
    def _eval_is_algebraic(self):
        return self.args[0].is_algebraic
    
    # 判断当前对象是否为零，基于其参数是否为扩展实数
    def _eval_is_zero(self):
        return self.args[0].is_extended_real
    
    # 判断当前对象是否有限，依据其参数是否为有限数值
    def _eval_is_finite(self):
        # 如果参数是有限数值，则返回True
        if self.args[0].is_finite:
            return True
    
    # 判断当前对象是否为复数，基于其参数是否为有限数值
    def _eval_is_complex(self):
        # 如果参数是有限数值，则返回True
        if self.args[0].is_finite:
            return True
# 定义一个复数函数 `sign`，继承自 `Function` 类
class sign(Function):
    """
    Returns the complex sign of an expression:

    Explanation
    ===========

    If the expression is real the sign will be:

        * $1$ if expression is positive
        * $0$ if expression is equal to zero
        * $-1$ if expression is negative

    If the expression is imaginary the sign will be:

        * $I$ if im(expression) is positive
        * $-I$ if im(expression) is negative

    Otherwise an unevaluated expression will be returned. When evaluated, the
    result (in general) will be ``cos(arg(expr)) + I*sin(arg(expr))``.

    Examples
    ========

    >>> from sympy import sign, I

    >>> sign(-1)
    -1
    >>> sign(0)
    0
    >>> sign(-3*I)
    -I
    >>> sign(1 + I)
    sign(1 + I)
    >>> _.evalf()
    0.707106781186548 + 0.707106781186548*I

    Parameters
    ==========

    arg : Expr
        Real or imaginary expression.

    Returns
    =======

    expr : Expr
        Complex sign of expression.

    See Also
    ========

    Abs, conjugate
    """

    # 标记该函数处理复数
    is_complex = True
    # 标记该函数具有奇点（特殊点）
    _singularities = True

    # 实现函数的具体操作，返回复数表达式的符号
    def doit(self, **hints):
        # 调用父类的 `doit` 方法
        s = super().doit()
        # 如果结果与输入相同且输入不为零，则返回输入除以其绝对值
        if s == self and self.args[0].is_zero is False:
            return self.args[0] / Abs(self.args[0])
        return s

    @classmethod
    # 定义一个类方法 eval，用于对给定的参数 arg 进行评估
    def eval(cls, arg):
        # 处理我们能够处理的情况
        # 检查参数是否为乘法表达式
        if arg.is_Mul:
            # 分解乘法表达式，得到系数 c 和乘积项 args
            c, args = arg.as_coeff_mul()
            # 初始化未知项列表
            unk = []
            # 初始化符号 s
            s = sign(c)
            # 遍历乘积项
            for a in args:
                # 如果当前项是扩展负数
                if a.is_extended_negative:
                    s = -s
                # 如果当前项是扩展正数
                elif a.is_extended_positive:
                    pass
                else:
                    # 如果当前项是虚数
                    if a.is_imaginary:
                        # 获取虚部
                        ai = im(a)
                        # 如果虚部是可比较的
                        if ai.is_comparable:
                            # 乘以虚数单位 I
                            s *= I
                            # 如果虚部是扩展负数
                            if ai.is_extended_negative:
                                # 由于 ai 可能不是数字，无法使用 sign(ai)，
                                # 所以手动调整符号 s
                                s = -s
                        else:
                            # 将未知项添加到列表中
                            unk.append(a)
                    else:
                        # 将未知项添加到列表中
                        unk.append(a)
            # 如果系数 c 是 1，并且未知项列表和乘积项列表长度相等，则返回 None
            if c is S.One and len(unk) == len(args):
                return None
            # 返回计算结果
            return s * cls(arg._new_rawargs(*unk))
        
        # 如果参数是 NaN
        if arg is S.NaN:
            return S.NaN
        
        # 如果参数是零
        if arg.is_zero:
            return S.Zero
        
        # 如果参数是扩展正数
        if arg.is_extended_positive:
            return S.One
        
        # 如果参数是扩展负数
        if arg.is_extended_negative:
            return S.NegativeOne
        
        # 如果参数是函数类型
        if arg.is_Function:
            # 如果参数是 sign 函数的实例
            if isinstance(arg, sign):
                return arg
        
        # 如果参数是虚数
        if arg.is_imaginary:
            # 如果参数是幂函数，并且指数是 1/2
            if arg.is_Pow and arg.exp is S.Half:
                # 我们捕获这种情况是因为非平凡的平方根参数不会被展开
                # 例如 sqrt(1-sqrt(2)) --x-->  to I*sqrt(sqrt(2) - 1)
                return I
            
            # 否则，构造 -I * arg，并检查其符号
            arg2 = -I * arg
            # 如果构造后的参数是扩展正数，返回虚数单位 I
            if arg2.is_extended_positive:
                return I
            # 如果构造后的参数是扩展负数，返回负虚数单位 -I
            if arg2.is_extended_negative:
                return -I

    # 定义一个私有方法 _eval_Abs，用于计算绝对值
    def _eval_Abs(self):
        # 如果参数的绝对值不是模糊的零
        if fuzzy_not(self.args[0].is_zero):
            # 返回 1
            return S.One

    # 定义一个私有方法 _eval_conjugate，用于计算共轭
    def _eval_conjugate(self):
        # 返回参数的共轭的符号
        return sign(conjugate(self.args[0]))

    # 定义一个私有方法 _eval_derivative，用于计算导数
    def _eval_derivative(self, x):
        # 如果参数是扩展实数
        if self.args[0].is_extended_real:
            # 导入 DiracDelta 函数
            from sympy.functions.special.delta_functions import DiracDelta
            # 返回导数的计算结果
            return 2 * Derivative(self.args[0], x, evaluate=True) \
                * DiracDelta(self.args[0])
        
        # 如果参数是虚数
        elif self.args[0].is_imaginary:
            # 导入 DiracDelta 函数
            from sympy.functions.special.delta_functions import DiracDelta
            # 返回导数的计算结果
            return 2 * Derivative(self.args[0], x, evaluate=True) \
                * DiracDelta(-I * self.args[0])

    # 定义一个私有方法 _eval_is_nonnegative，用于判断是否非负数
    def _eval_is_nonnegative(self):
        # 如果参数是非负数，返回 True
        if self.args[0].is_nonnegative:
            return True

    # 定义一个私有方法 _eval_is_nonpositive，用于判断是否非正数
    def _eval_is_nonpositive(self):
        # 如果参数是非正数，返回 True
        if self.args[0].is_nonpositive:
            return True

    # 定义一个私有方法 _eval_is_imaginary，用于判断是否虚数
    def _eval_is_imaginary(self):
        # 返回参数是否为虚数
        return self.args[0].is_imaginary

    # 定义一个私有方法 _eval_is_integer，用于判断是否整数
    def _eval_is_integer(self):
        # 返回参数是否为扩展实数
        return self.args[0].is_extended_real

    # 定义一个私有方法 _eval_is_zero，用于判断是否为零
    def _eval_is_zero(self):
        # 返回参数是否为零
        return self.args[0].is_zero
    # 定义一个方法来评估幂运算
    def _eval_power(self, other):
        # 如果第一个参数不是零，并且第二个参数是整数并且是偶数
        if (
            fuzzy_not(self.args[0].is_zero) and
            other.is_integer and
            other.is_even
        ):
            # 返回常数1
            return S.One

    # 定义一个方法来进行级数展开
    def _eval_nseries(self, x, n, logx, cdir=0):
        # 获取第一个参数
        arg0 = self.args[0]
        # 计算在x=0处的极限
        x0 = arg0.subs(x, 0)
        # 如果x0不等于0，返回在x0处的函数值
        if x0 != 0:
            return self.func(x0)
        # 如果cdir不等于0，确定在指定方向上的极限
        if cdir != 0:
            cdir = arg0.dir(x, cdir)
        # 如果方向小于0，则返回-1，否则返回1
        return -S.One if re(cdir) < 0 else S.One

    # 定义一个方法将表达式重写为Piecewise形式
    def _eval_rewrite_as_Piecewise(self, arg, **kwargs):
        # 如果参数是扩展实数
        if arg.is_extended_real:
            # 返回Piecewise对象，根据参数的不同值返回1，-1或0
            return Piecewise((1, arg > 0), (-1, arg < 0), (0, True))

    # 定义一个方法将表达式重写为Heaviside函数的形式
    def _eval_rewrite_as_Heaviside(self, arg, **kwargs):
        # 导入Heaviside函数
        from sympy.functions.special.delta_functions import Heaviside
        # 如果参数是扩展实数
        if arg.is_extended_real:
            # 返回Heaviside函数关于参数的表达式乘以2再减去1
            return Heaviside(arg) * 2 - 1

    # 定义一个方法将表达式重写为绝对值函数的形式
    def _eval_rewrite_as_Abs(self, arg, **kwargs):
        # 返回Piecewise对象，根据参数的不同值返回0或arg/Abs(arg)
        return Piecewise((0, Eq(arg, 0)), (arg / Abs(arg), True))

    # 定义一个方法简化表达式
    def _eval_simplify(self, **kwargs):
        # 对第一个参数进行因式分解后返回函数本身的形式
        return self.func(factor_terms(self.args[0]))  # XXX include doit?
class Abs(Function):
    """
    Return the absolute value of the argument.

    Explanation
    ===========

    This is an extension of the built-in function ``abs()`` to accept symbolic
    values.  If you pass a SymPy expression to the built-in ``abs()``, it will
    pass it automatically to ``Abs()``.

    Examples
    ========

    >>> from sympy import Abs, Symbol, S, I
    >>> Abs(-1)
    1
    >>> x = Symbol('x', real=True)
    >>> Abs(-x)
    Abs(x)
    >>> Abs(x**2)
    x**2
    >>> abs(-x) # The Python built-in
    Abs(x)
    >>> Abs(3*x + 2*I)
    sqrt(9*x**2 + 4)
    >>> Abs(8*I)
    8

    Note that the Python built-in will return either an Expr or int depending on
    the argument::

        >>> type(abs(-1))
        <... 'int'>
        >>> type(abs(S.NegativeOne))
        <class 'sympy.core.numbers.One'>

    Abs will always return a SymPy object.

    Parameters
    ==========

    arg : Expr
        Real or complex expression.

    Returns
    =======

    expr : Expr
        Absolute value returned can be an expression or integer depending on
        input arg.

    See Also
    ========

    sign, conjugate
    """

    args: tTuple[Expr]

    is_extended_real = True
    is_extended_negative = False
    is_extended_nonnegative = True
    unbranched = True
    _singularities = True  # non-holomorphic

    def fdiff(self, argindex=1):
        """
        Get the first derivative of the argument to Abs().

        """
        if argindex == 1:
            # Return the sign of the argument to Abs()
            return sign(self.args[0])
        else:
            # Raise an error if the argument index is invalid
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def _eval_is_real(self):
        # Check if the argument to Abs() is finite to determine if it's real
        if self.args[0].is_finite:
            return True

    def _eval_is_integer(self):
        # Check if the argument to Abs() is extended real and integer
        if self.args[0].is_extended_real:
            return self.args[0].is_integer

    def _eval_is_extended_nonzero(self):
        # Check if the argument to Abs() is nonzero
        return fuzzy_not(self._args[0].is_zero)

    def _eval_is_zero(self):
        # Check if the argument to Abs() is zero
        return self._args[0].is_zero

    def _eval_is_extended_positive(self):
        # Check if the argument to Abs() is nonzero (positive)
        return fuzzy_not(self._args[0].is_zero)

    def _eval_is_rational(self):
        # Check if the argument to Abs() is extended real and rational
        if self.args[0].is_extended_real:
            return self.args[0].is_rational

    def _eval_is_even(self):
        # Check if the argument to Abs() is extended real and even
        if self.args[0].is_extended_real:
            return self.args[0].is_even

    def _eval_is_odd(self):
        # Check if the argument to Abs() is extended real and odd
        if self.args[0].is_extended_real:
            return self.args[0].is_odd

    def _eval_is_algebraic(self):
        # Check if the argument to Abs() is algebraic
        return self.args[0].is_algebraic

    def _eval_power(self, exponent):
        # Evaluate the power of the argument to Abs() with integer exponent
        if self.args[0].is_extended_real and exponent.is_integer:
            if exponent.is_even:
                return self.args[0]**exponent
            elif exponent is not S.NegativeOne and exponent.is_Integer:
                return self.args[0]**(exponent - 1)*self
        return
    # 导入对数函数以及其他必要的模块或函数
    from sympy.functions.elementary.exponential import log
    
    # 从表达式的主导项获取方向，并做必要的对数替换
    direction = self.args[0].leadterm(x)[0]
    if direction.has(log(x)):
        direction = direction.subs(log(x), logx)
    
    # 对表达式进行 n 次级数展开，乘以方向的符号并展开结果
    s = self.args[0]._eval_nseries(x, n=n, logx=logx)
    return (sign(direction)*s).expand()

    # 如果表达式是实数或虚数，则计算其对 x 的导数乘以符号函数
    if self.args[0].is_extended_real or self.args[0].is_imaginary:
        return Derivative(self.args[0], x, evaluate=True) \
            * sign(conjugate(self.args[0]))
    
    # 否则，计算实部和虚部的导数，并根据表达式的绝对值返回重写的结果
    rv = (re(self.args[0]) * Derivative(re(self.args[0]), x,
        evaluate=True) + im(self.args[0]) * Derivative(im(self.args[0]),
            x, evaluate=True)) / Abs(self.args[0])
    return rv.rewrite(sign)

    # 将表达式重写为 Heaviside 函数的形式（仅对实数参数有效）
    # 注意：Heaviside 函数不支持复数参数
    from sympy.functions.special.delta_functions import Heaviside
    if arg.is_extended_real:
        return arg*(Heaviside(arg) - Heaviside(-arg))

    # 将表达式重写为 Piecewise 函数的形式，处理实数和虚数情况
    if arg.is_extended_real:
        return Piecewise((arg, arg >= 0), (-arg, True))
    elif arg.is_imaginary:
        return Piecewise((I*arg, I*arg >= 0), (-I*arg, True))

    # 将表达式重写为符号函数的形式
    return arg/sign(arg)

    # 将表达式重写为共轭函数的形式
    return sqrt(arg*conjugate(arg))
class arg(Function):
    r"""
    Returns the argument (in radians) of a complex number. The argument is
    evaluated in consistent convention with ``atan2`` where the branch-cut is
    taken along the negative real axis and ``arg(z)`` is in the interval
    $(-\pi,\pi]$. For a positive number, the argument is always 0; the
    argument of a negative number is $\pi$; and the argument of 0
    is undefined and returns ``nan``. So the ``arg`` function will never nest
    greater than 3 levels since at the 4th application, the result must be
    nan; for a real number, nan is returned on the 3rd application.

    Examples
    ========

    >>> from sympy import arg, I, sqrt, Dummy
    >>> from sympy.abc import x
    >>> arg(2.0)
    0
    >>> arg(I)
    pi/2
    >>> arg(sqrt(2) + I*sqrt(2))
    pi/4
    >>> arg(sqrt(3)/2 + I/2)
    pi/6
    >>> arg(4 + 3*I)
    atan(3/4)
    >>> arg(0.8 + 0.6*I)
    0.643501108793284
    >>> arg(arg(arg(arg(x))))
    nan
    >>> real = Dummy(real=True)
    >>> arg(arg(arg(real)))
    nan

    Parameters
    ==========

    arg : Expr
        Real or complex expression.

    Returns
    =======

    value : Expr
        Returns arc tangent of arg measured in radians.

    """

    is_extended_real = True  # 设置为扩展实数
    is_real = True  # 设置为实数
    is_finite = True  # 设置为有限的
    _singularities = True  # 非全纯的（非解析的）

    @classmethod
    def eval(cls, arg):
        a = arg  # 将参数保存到变量a中
        for i in range(3):
            if isinstance(a, cls):
                a = a.args[0]  # 如果a是当前类的实例，则取其第一个参数
            else:
                if i == 2 and a.is_extended_real:
                    return S.NaN  # 如果第三次迭代且a是扩展实数，则返回NaN
                break
        else:
            return S.NaN  # 超过3次嵌套返回NaN

        from sympy.functions.elementary.exponential import exp, exp_polar
        if isinstance(arg, exp_polar):
            return periodic_argument(arg, oo)  # 如果参数是极坐标指数形式，则返回周期参数
        elif isinstance(arg, exp):
            i_ = im(arg.args[0])
            if i_.is_comparable:
                i_ %= 2*S.Pi
                if i_ > S.Pi:
                    i_ -= 2*S.Pi
                return i_  # 返回对数函数的周期参数

        if not arg.is_Atom:
            c, arg_ = factor_terms(arg).as_coeff_Mul()
            if arg_.is_Mul:
                arg_ = Mul(*[a if (sign(a) not in (-1, 1)) else
                    sign(a) for a in arg_.args])
            arg_ = sign(c)*arg_
        else:
            arg_ = arg
        if any(i.is_extended_positive is None for i in arg_.atoms(AppliedUndef)):
            return  # 如果arg_中的某些参数非正，则返回空值
        from sympy.functions.elementary.trigonometric import atan2
        x, y = arg_.as_real_imag()
        rv = atan2(y, x)  # 计算arctan(y/x)
        if rv.is_number:
            return rv  # 如果结果是数值，则返回结果
        if arg_ != arg:
            return cls(arg_, evaluate=False)  # 如果arg_不等于arg，则返回arg_的类

    def _eval_derivative(self, t):
        x, y = self.args[0].as_real_imag()
        return (x * Derivative(y, t, evaluate=True) - y *
                    Derivative(x, t, evaluate=True)) / (x**2 + y**2)
        # 对参数t的实部和虚部进行微分运算，并返回结果
    # 定义一个方法 `_eval_rewrite_as_atan2`，用于将表达式重写为 atan2 函数的形式
    def _eval_rewrite_as_atan2(self, arg, **kwargs):
        # 导入 atan2 函数
        from sympy.functions.elementary.trigonometric import atan2
        # 获取表达式 self 的实部 x 和虚部 y
        x, y = self.args[0].as_real_imag()
        # 返回 atan2 函数的结果，以 y 和 x 作为参数
        return atan2(y, x)
class conjugate(Function):
    """
    Returns the *complex conjugate* [1]_ of an argument.
    In mathematics, the complex conjugate of a complex number
    is given by changing the sign of the imaginary part.

    Thus, the conjugate of the complex number
    :math:`a + ib` (where $a$ and $b$ are real numbers) is :math:`a - ib`

    Examples
    ========

    >>> from sympy import conjugate, I
    >>> conjugate(2)
    2
    >>> conjugate(I)
    -I
    >>> conjugate(3 + 2*I)
    3 - 2*I
    >>> conjugate(5 - I)
    5 + I

    Parameters
    ==========

    arg : Expr
        Real or complex expression.

    Returns
    =======

    arg : Expr
        Complex conjugate of arg as real, imaginary or mixed expression.

    See Also
    ========

    sign, Abs

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Complex_conjugation
    """
    _singularities = True  # 非全纯的特性

    @classmethod
    def eval(cls, arg):
        # 调用参数的_eval_conjugate()方法获得其共轭对象
        obj = arg._eval_conjugate()
        if obj is not None:
            return obj

    def inverse(self):
        # 返回conjugate函数自身，表示其逆运算是自身
        return conjugate

    def _eval_Abs(self):
        # 返回参数的绝对值，使用evaluate=True确保求值
        return Abs(self.args[0], evaluate=True)

    def _eval_adjoint(self):
        # 返回参数的伴随矩阵（共轭转置）
        return transpose(self.args[0])

    def _eval_conjugate(self):
        # 返回参数自身，因为共轭对象本身就是自身
        return self.args[0]

    def _eval_derivative(self, x):
        if x.is_real:
            # 如果参数是实数，则返回参数关于x的导数的共轭
            return conjugate(Derivative(self.args[0], x, evaluate=True))
        elif x.is_imaginary:
            # 如果参数是虚数，则返回参数关于x的导数的负共轭
            return -conjugate(Derivative(self.args[0], x, evaluate=True))

    def _eval_transpose(self):
        # 返回参数的转置操作
        return adjoint(self.args[0])

    def _eval_is_algebraic(self):
        # 判断参数是否是代数表达式
        return self.args[0].is_algebraic


class transpose(Function):
    """
    Linear map transposition.

    Examples
    ========

    >>> from sympy import transpose, Matrix, MatrixSymbol
    >>> A = MatrixSymbol('A', 25, 9)
    >>> transpose(A)
    A.T
    >>> B = MatrixSymbol('B', 9, 22)
    >>> transpose(B)
    B.T
    >>> transpose(A*B)
    B.T*A.T
    >>> M = Matrix([[4, 5], [2, 1], [90, 12]])
    >>> M
    Matrix([
    [ 4,  5],
    [ 2,  1],
    [90, 12]])
    >>> transpose(M)
    Matrix([
    [4, 2, 90],
    [5, 1, 12]])

    Parameters
    ==========

    arg : Matrix
         Matrix or matrix expression to take the transpose of.

    Returns
    =======

    value : Matrix
        Transpose of arg.

    """

    @classmethod
    def eval(cls, arg):
        # 调用参数的_eval_transpose()方法获得其转置
        obj = arg._eval_transpose()
        if obj is not None:
            return obj

    def _eval_adjoint(self):
        # 返回参数的共轭转置
        return conjugate(self.args[0])

    def _eval_conjugate(self):
        # 返回参数的伴随矩阵（共轭转置）
        return adjoint(self.args[0])

    def _eval_transpose(self):
        # 返回参数自身，表示其转置操作本身就是自身
        return self.args[0]


class adjoint(Function):
    """
    Conjugate transpose or Hermite conjugation.

    Examples
    ========

    >>> from sympy import adjoint, MatrixSymbol
    >>> A = MatrixSymbol('A', 10, 5)
    >>> adjoint(A)
    Adjoint(A)

    Parameters
    ==========
    # 定义一个类方法 `eval`，用于计算参数 `arg` 的伴随矩阵或转置共轭
    @classmethod
    def eval(cls, arg):
        # 调用参数 `arg` 对象的 `_eval_adjoint()` 方法，获取其伴随矩阵
        obj = arg._eval_adjoint()
        # 如果伴随矩阵存在，则返回它
        if obj is not None:
            return obj
        # 否则调用参数 `arg` 对象的 `_eval_transpose()` 方法，获取其转置矩阵
        obj = arg._eval_transpose()
        # 如果转置矩阵存在，则返回其共轭
        if obj is not None:
            return conjugate(obj)

    # 定义一个方法 `_eval_adjoint`，返回当前对象 `self` 的第一个参数
    def _eval_adjoint(self):
        return self.args[0]

    # 定义一个方法 `_eval_conjugate`，返回当前对象 `self` 的第一个参数的转置
    def _eval_conjugate(self):
        return transpose(self.args[0])

    # 定义一个方法 `_eval_transpose`，返回当前对象 `self` 的第一个参数的共轭
    def _eval_transpose(self):
        return conjugate(self.args[0])

    # 定义一个方法 `_latex`，用于生成当前对象 `self` 的 LaTeX 表示
    def _latex(self, printer, exp=None, *args):
        # 获取当前对象 `self` 的第一个参数的 LaTeX 表示
        arg = printer._print(self.args[0])
        # 构造伴随矩阵的 LaTeX 表示，包括可选的指数 `exp`
        tex = r'%s^{\dagger}' % arg
        if exp:
            tex = r'\left(%s\right)^{%s}' % (tex, exp)
        return tex

    # 定义一个方法 `_pretty`，用于生成当前对象 `self` 的美化打印表示
    def _pretty(self, printer, *args):
        # 导入所需的模块
        from sympy.printing.pretty.stringpict import prettyForm
        # 获取当前对象 `self` 的第一个参数的美化打印表示
        pform = printer._print(self.args[0], *args)
        # 根据打印机的使用Unicode标志 `_use_unicode`，为打印表示添加伴随符号
        if printer._use_unicode:
            pform = pform**prettyForm('\N{DAGGER}')
        else:
            pform = pform**prettyForm('+')
        return pform
# 定义一个函数类 `polar_lift`，用于将参数提升到对数的黎曼曲面上，使用标准分支。
class polar_lift(Function):
    """
    Lift argument to the Riemann surface of the logarithm, using the
    standard branch.
    
    Examples
    ========
    
    >>> from sympy import Symbol, polar_lift, I
    >>> p = Symbol('p', polar=True)
    >>> x = Symbol('x')
    >>> polar_lift(4)
    4*exp_polar(0)
    >>> polar_lift(-4)
    4*exp_polar(I*pi)
    >>> polar_lift(-I)
    exp_polar(-I*pi/2)
    >>> polar_lift(I + 2)
    polar_lift(2 + I)
    
    >>> polar_lift(4*x)
    4*polar_lift(x)
    >>> polar_lift(4*p)
    4*p
    
    Parameters
    ==========
    
    arg : Expr
        Real or complex expression.
    
    See Also
    ========
    
    sympy.functions.elementary.exponential.exp_polar
    periodic_argument
    """

    # 设置 `is_polar` 属性为 True，表示此类是极坐标数
    is_polar = True
    # 设置 `is_comparable` 属性为 False，表示无法使用 `evalf()` 进行数值化计算
    is_comparable = False  # Cannot be evalf'd.

    @classmethod
    def eval(cls, arg):
        # 导入 `arg` 的复数部分的函数
        from sympy.functions.elementary.complexes import arg as argument
        # 如果 `arg` 是一个数值
        if arg.is_number:
            # 计算其复数部分
            ar = argument(arg)
            # 一般来说，我们希望确认某些已知的情况，
            # 例如 `not ar.has(argument) and not ar.has(atan)`
            # 但现在我们会更加严格，只看它是否为已知值之一
            if ar in (0, pi/2, -pi/2, pi):
                # 导入指数函数 `exp_polar`
                from sympy.functions.elementary.exponential import exp_polar
                # 返回根据已知角度 `ar` 计算的极坐标数值
                return exp_polar(I*ar)*abs(arg)

        # 如果 `arg` 是一个乘法表达式
        if arg.is_Mul:
            args = arg.args
        else:
            args = [arg]
        included = []
        excluded = []
        positive = []
        # 遍历参数列表
        for arg in args:
            # 如果参数是极坐标数
            if arg.is_polar:
                included += [arg]
            # 如果参数是正数
            elif arg.is_positive:
                positive += [arg]
            # 否则
            else:
                excluded += [arg]
        # 如果被排除的参数不为空
        if len(excluded) < len(args):
            # 如果有被排除的参数
            if excluded:
                # 返回极坐标数乘积乘以被排除参数的极坐标提升
                return Mul(*(included + positive))*polar_lift(Mul(*excluded))
            # 如果只有包含的参数
            elif included:
                # 返回包含参数的乘积
                return Mul(*(included + positive))
            # 否则
            else:
                # 导入指数函数 `exp_polar`
                from sympy.functions.elementary.exponential import exp_polar
                # 返回正数的乘积乘以指数函数 `exp_polar` 的 0 值
                return Mul(*positive)*exp_polar(0)

    # 定义用于数值化计算的函数 `_eval_evalf`
    def _eval_evalf(self, prec):
        """ Careful! any evalf of polar numbers is flaky """
        # 返回参数列表第一个元素的数值化计算结果
        return self.args[0]._eval_evalf(prec)

    # 定义用于求绝对值的函数 `_eval_Abs`
    def _eval_Abs(self):
        # 返回参数的绝对值
        return Abs(self.args[0], evaluate=True)


# 定义一个函数类 `periodic_argument`
class periodic_argument(Function):
    r"""
    Represent the argument on a quotient of the Riemann surface of the
    logarithm. That is, given a period $P$, always return a value in
    $(-P/2, P/2]$, by using $\exp(PI) = 1$.

    Examples
    ========

    >>> from sympy import exp_polar, periodic_argument
    >>> from sympy import I, pi
    >>> periodic_argument(exp_polar(10*I*pi), 2*pi)
    0
    """
    @classmethod
    def _getunbranched(cls, ar):
        # 导入必要的指数函数和对数函数
        from sympy.functions.elementary.exponential import exp_polar, log
        
        # 如果 ar 是乘积表达式，则将其参数提取出来
        if ar.is_Mul:
            args = ar.args
        else:
            args = [ar]
        
        # 初始化无分支参数为 0
        unbranched = 0
        
        # 遍历 ar 的每个参数
        for a in args:
            # 如果参数不是极坐标形式，直接取其辐角
            if not a.is_polar:
                unbranched += arg(a)
            # 如果参数是 exp_polar 类型，则取其实部和虚部的辐角部分
            elif isinstance(a, exp_polar):
                unbranched += a.exp.as_real_imag()[1]
            # 如果参数是幂函数形式，则计算其辐角部分
            elif a.is_Pow:
                re, im = a.exp.as_real_imag()
                unbranched += re * unbranched_argument(a.base) + im * log(abs(a.base))
            # 如果参数是 polar_lift 类型，则取其参数的辐角
            elif isinstance(a, polar_lift):
                unbranched += arg(a.args[0])
            # 如果参数类型无法处理，则返回 None
            else:
                return None
        
        # 返回计算得到的无分支参数
        return unbranched

    @classmethod
    def eval(cls, ar, period):
        # 我们的策略是在对数的黎曼面上评估参数，然后进行减少。
        # 注意，这意味着对于 period != 2*pi 和非极坐标数，使用此函数可能不是一个好主意。
        
        # 如果 period 不是正数，返回 None
        if not period.is_extended_positive:
            return None
        
        # 如果 period 为无穷大，并且 ar 是 principal_branch 类型，则返回其参数的周期辐角
        if period == oo and isinstance(ar, principal_branch):
            return periodic_argument(*ar.args)
        
        # 如果 ar 是 polar_lift 类型，并且 period 大于等于 2*pi，则返回 ar 参数的周期辐角
        if isinstance(ar, polar_lift) and period >= 2*pi:
            return periodic_argument(ar.args[0], period)
        
        # 如果 ar 是乘积表达式，移除其中的正数因子，并计算其周期辐角
        if ar.is_Mul:
            newargs = [x for x in ar.args if not x.is_positive]
            if len(newargs) != len(ar.args):
                return periodic_argument(Mul(*newargs), period)
        
        # 获取 ar 的无分支参数
        unbranched = cls._getunbranched(ar)
        
        # 如果无法获取无分支参数，返回 None
        if unbranched is None:
            return None
        
        # 导入必要的三角函数
        from sympy.functions.elementary.trigonometric import atan, atan2
        
        # 如果无分支参数中包含周期辐角、atan2 或 atan 函数，返回 None
        if unbranched.has(periodic_argument, atan2, atan):
            return None
        
        # 如果 period 为无穷大，则直接返回计算得到的无分支参数
        if period == oo:
            return unbranched
        
        # 如果 period 不为无穷大，则进行周期化修正
        if period != oo:
            from sympy.functions.elementary.integers import ceiling
            from sympy import S
            n = ceiling(unbranched / period - S.Half) * period
            # 如果 n 中不包含 ceiling 函数，返回修正后的无分支参数
            if not n.has(ceiling):
                return unbranched - n
    # 定义一个方法 `_eval_evalf`，接受精度参数 `prec`
    def _eval_evalf(self, prec):
        # 将方法的参数解构为 z 和 period
        z, period = self.args
        
        # 如果 period 为无穷大 oo
        if period == oo:
            # 获取 z 的无分支值
            unbranched = periodic_argument._getunbranched(z)
            # 如果无分支值为 None，则返回当前对象自身
            if unbranched is None:
                return self
            # 递归调用无分支值的 `_eval_evalf` 方法，传递相同的精度参数
            return unbranched._eval_evalf(prec)
        
        # 计算 z 对于无穷大 period 的周期性参数
        ub = periodic_argument(z, oo)._eval_evalf(prec)
        
        # 导入 ceiling 函数，用于向上取整
        from sympy.functions.elementary.integers import ceiling
        
        # 计算 ub 减去向上取整后的表达式，并对结果应用精度评估
        return (ub - ceiling(ub/period - S.Half)*period)._eval_evalf(prec)
# 定义函数 unbranched_argument，返回参数 arg 的周期论点，周期为无穷大
def unbranched_argument(arg):
    # 调用 periodic_argument 函数，周期参数设为无穷大
    return periodic_argument(arg, oo)


# 定义类 principal_branch，表示将极坐标数降至对数黎曼面的主分支
class principal_branch(Function):
    """
    代表一个极坐标数在对数黎曼面的商上被降至主分支。

    Explanation
    ===========

    这是一个两个参数的函数。第一个参数是极坐标数 `z`，第二个是正实数或无穷大 `p`。
    结果是 ``z mod exp_polar(I*p)``。

    Examples
    ========

    >>> from sympy import exp_polar, principal_branch, oo, I, pi
    >>> from sympy.abc import z
    >>> principal_branch(z, oo)
    z
    >>> principal_branch(exp_polar(2*pi*I)*3, 2*pi)
    3*exp_polar(0)
    >>> principal_branch(exp_polar(2*pi*I)*3*z, 2*pi)
    3*principal_branch(z, 2*pi)

    Parameters
    ==========

    x : Expr
        一个极坐标数。

    period : Expr
        正实数或无穷大。

    See Also
    ========

    sympy.functions.elementary.exponential.exp_polar
    polar_lift : 将参数提升到对数黎曼面
    periodic_argument
    """

    # 类变量，表示这是一个极坐标数
    is_polar = True
    # 类变量，表示不能总是进行 evalf 计算
    is_comparable = False  # 不能总是进行 evalf 计算

    @classmethod
    # 定义一个方法 eval，接受两个参数：x 和 period
    def eval(self, x, period):
        # 从 sympy 库中导入 exp_polar 函数
        from sympy.functions.elementary.exponential import exp_polar
        
        # 如果 x 是 polar_lift 类型的实例，则返回其在给定周期下的主分支
        if isinstance(x, polar_lift):
            return principal_branch(x.args[0], period)
        
        # 如果 period 为无穷大，则直接返回 x
        if period == oo:
            return x
        
        # 计算 x 的上界 unbranched argument 和在给定周期下的分支 barg
        ub = periodic_argument(x, oo)
        barg = periodic_argument(x, period)
        
        # 如果 unbranched argument 和周期分支不同，并且它们都不包含 periodic_argument
        if ub != barg and not ub.has(periodic_argument) \
                and not barg.has(periodic_argument):
            
            # 对 x 进行 polar_lift 处理
            pl = polar_lift(x)
            
            # 定义一个替换函数 mr，用于处理 polar_lift 之外的表达式
            def mr(expr):
                if not isinstance(expr, Symbol):
                    return polar_lift(expr)
                return expr
            
            # 使用替换函数 mr 替换 polar_lift 中的表达式
            pl = pl.replace(polar_lift, mr)
            
            # 重新计算 unbranched argument
            ub = periodic_argument(pl, oo)
            
            # 如果 pl 中不包含 polar_lift
            if not pl.has(polar_lift):
                # 如果 unbranched argument 和周期分支不同
                if ub != barg:
                    res = exp_polar(I*(barg - ub)) * pl
                else:
                    res = pl
                
                # 如果 res 不是极坐标形式，并且不包含 exp_polar，则乘以 exp_polar(0)
                if not res.is_polar and not res.has(exp_polar):
                    res *= exp_polar(0)
                
                return res
        
        # 如果 x 不包含自由符号
        if not x.free_symbols:
            c, m = x, ()
        else:
            # 将 x 分解为系数和乘积形式
            c, m = x.as_coeff_mul(*x.free_symbols)
        
        others = []
        # 对乘积形式中的每个因子进行处理
        for y in m:
            if y.is_positive:
                c *= y
            else:
                others += [y]
        
        m = tuple(others)
        
        # 计算 c 的周期分支
        arg = periodic_argument(c, period)
        
        # 如果 arg 中包含 periodic_argument，则返回 None
        if arg.has(periodic_argument):
            return None
        
        # 如果 arg 是数值，并且 unbranched_argument(c) 不等于 arg
        # 或者 arg 为 0，而 m 不为空且 c 不为 1
        if arg.is_number and (unbranched_argument(c) != arg or
                              (arg == 0 and m != () and c != 1)):
            # 如果 arg 为 0，则返回 abs(c) 乘以 Mul(*m) 的主分支
            if arg == 0:
                return abs(c) * principal_branch(Mul(*m), period)
            
            # 否则返回 exp_polar(I*arg) 乘以 Mul(*m) 的主分支再乘以 abs(c)
            return principal_branch(exp_polar(I*arg) * Mul(*m), period) * abs(c)
        
        # 如果 arg 是数值，并且 abs(arg) 小于 period/2 或者 arg 等于 period/2，并且 m 为空
        if arg.is_number and ((abs(arg) < period/2) == True or arg == period/2) \
                and m == ():
            # 返回 exp_polar(arg*I) 乘以 abs(c)
            return exp_polar(arg * I) * abs(c)

    # 定义一个私有方法 _eval_evalf，接受一个参数 prec
    def _eval_evalf(self, prec):
        # 将 self 的两个参数解包为 z 和 period
        z, period = self.args
        
        # 计算 z 在给定周期下的周期分支，然后对其进行数值评估
        p = periodic_argument(z, period)._eval_evalf(prec)
        
        # 如果 p 的绝对值大于 pi 或者 p 等于 -pi，则返回 self，无法进行数值评估
        if abs(p) > pi or p == -pi:
            return self
        
        # 从 sympy 库中导入 exp 函数
        from sympy.functions.elementary.exponential import exp
        
        # 返回 abs(z) 乘以 exp(I*p) 的数值评估结果
        return (abs(z) * exp(I*p))._eval_evalf(prec)
# 定义一个私有函数 _polarify，用于处理 sympy 表达式的极坐标转换
def _polarify(eq, lift, pause=False):
    # 导入积分相关的模块
    from sympy.integrals.integrals import Integral
    # 如果表达式已经是极坐标形式，则直接返回
    if eq.is_polar:
        return eq
    # 如果表达式是数值且未暂停且 lift 为 False，则将其转换为极坐标形式
    if eq.is_number and not pause:
        return polar_lift(eq)
    # 如果表达式是符号，并且未暂停且 lift 为 True，则将其转换为极坐标形式
    if isinstance(eq, Symbol) and not pause and lift:
        return polar_lift(eq)
    # 如果表达式是原子表达式，则直接返回
    elif eq.is_Atom:
        return eq
    # 如果表达式是加法表达式，则递归处理每个子表达式并进行极坐标转换
    elif eq.is_Add:
        r = eq.func(*[_polarify(arg, lift, pause=True) for arg in eq.args])
        # 如果 lift 为 True，则对整个表达式应用极坐标 lift
        if lift:
            return polar_lift(r)
        return r
    # 如果表达式是幂函数且底数为 S.Exp1（即自然指数 e），则处理指数部分并保持底数不变
    elif eq.is_Pow and eq.base == S.Exp1:
        return eq.func(S.Exp1, _polarify(eq.exp, lift, pause=False))
    # 如果表达式是函数表达式，则递归处理每个参数并应用极坐标转换
    elif eq.is_Function:
        return eq.func(*[_polarify(arg, lift, pause=False) for arg in eq.args])
    # 如果表达式是积分对象，则处理积分函数和积分变量
    elif isinstance(eq, Integral):
        # 不对积分变量应用极坐标 lift
        func = _polarify(eq.function, lift, pause=pause)
        limits = []
        # 处理积分限制
        for limit in eq.args[1:]:
            var = _polarify(limit[0], lift=False, pause=pause)
            rest = _polarify(limit[1:], lift=lift, pause=pause)
            limits.append((var,) + rest)
        return Integral(*((func,) + tuple(limits)))
    # 其他情况，直接应用表达式的函数并递归处理每个参数
    else:
        return eq.func(*[_polarify(arg, lift, pause=pause)
                         if isinstance(arg, Expr) else arg for arg in eq.args])


# 定义一个公共函数 polarify，用于将表达式中的所有数值转换为极坐标等价物
def polarify(eq, subs=True, lift=False):
    """
    Turn all numbers in eq into their polar equivalents (under the standard
    choice of argument).

    Note that no attempt is made to guess a formal convention of adding
    polar numbers, expressions like $1 + x$ will generally not be altered.

    Note also that this function does not promote ``exp(x)`` to ``exp_polar(x)``.

    If ``subs`` is ``True``, all symbols which are not already polar will be
    substituted for polar dummies; in this case the function behaves much
    like :func:`~.posify`.

    If ``lift`` is ``True``, both addition statements and non-polar symbols are
    changed to their ``polar_lift()``ed versions.
    Note that ``lift=True`` implies ``subs=False``.

    Examples
    ========

    >>> from sympy import polarify, sin, I
    >>> from sympy.abc import x, y
    >>> expr = (-x)**y
    >>> expr.expand()
    (-x)**y
    >>> polarify(expr)
    ((_x*exp_polar(I*pi))**_y, {_x: x, _y: y})
    >>> polarify(expr)[0].expand()
    _x**_y*exp_polar(_y*I*pi)
    >>> polarify(x, lift=True)
    polar_lift(x)
    >>> polarify(x*(1+y), lift=True)
    polar_lift(x)*polar_lift(y + 1)

    Adds are treated carefully:

    >>> polarify(1 + sin((1 + I)*x))
    (sin(_x*polar_lift(1 + I)) + 1, {_x: x})
    """
    # 如果 lift 为 True，则设置 subs 为 False
    if lift:
        subs = False
    # 将输入的表达式转换为 sympy 的表达式对象，然后应用 _polarify 函数进行极坐标转换
    eq = _polarify(sympify(eq), lift)
    # 如果 subs 为 False，则直接返回转换后的表达式
    if not subs:
        return eq
    # 如果 subs 为 True，则为所有未转换为极坐标的符号创建极坐标虚拟变量
    reps = {s: Dummy(s.name, polar=True) for s in eq.free_symbols}
    eq = eq.subs(reps)
    # 返回转换后的表达式及转换映射
    return eq, {r: s for s, r in reps.items()}


# 定义一个私有函数 _unpolarify，用于将极坐标形式的表达式还原为普通表达式
def _unpolarify(eq, exponents_only, pause=False):
    # 如果表达式不是 Basic 类型或者是原子表达式，则直接返回
    if not isinstance(eq, Basic) or eq.is_Atom:
        return eq
    # 如果不暂停处理：
    if not pause:
        # 导入 sympy 库中的指数函数 exp 和极坐标指数函数 exp_polar
        from sympy.functions.elementary.exponential import exp, exp_polar
        # 如果 eq 是 exp_polar 类型的对象：
        if isinstance(eq, exp_polar):
            # 返回对 eq.exp 解除极坐标化的结果
            return exp(_unpolarify(eq.exp, exponents_only))
        # 如果 eq 是 principal_branch 类型的对象，并且第二个参数是 2*pi：
        if isinstance(eq, principal_branch) and eq.args[1] == 2*pi:
            # 返回对 eq.args[0] 解除极坐标化的结果
            return _unpolarify(eq.args[0], exponents_only)
        # 如果 eq 是加法、乘法、布尔运算或者关系运算中的等式或不等式，且包含特定条件：
        if (
            eq.is_Add or eq.is_Mul or eq.is_Boolean or
            eq.is_Relational and (
                eq.rel_op in ('==', '!=') and 0 in eq.args or
                eq.rel_op not in ('==', '!='))
        ):
            # 返回对 eq.args 中每个元素解除极坐标化的结果，重新构造相同类型的对象
            return eq.func(*[_unpolarify(x, exponents_only) for x in eq.args])
        # 如果 eq 是 polar_lift 类型的对象：
        if isinstance(eq, polar_lift):
            # 返回对 eq.args[0] 解除极坐标化的结果
            return _unpolarify(eq.args[0], exponents_only)

    # 如果 eq 是幂运算类型的对象：
    if eq.is_Pow:
        # 对指数部分进行解除极坐标化
        expo = _unpolarify(eq.exp, exponents_only)
        # 对底数部分进行解除极坐标化，特定条件下不进行解除
        base = _unpolarify(eq.base, exponents_only,
            not (expo.is_integer and not pause))
        # 返回解除极坐标化后的结果
        return base**expo

    # 如果 eq 是函数类型的对象，并且其函数被标记为 unbranched：
    if eq.is_Function and getattr(eq.func, 'unbranched', False):
        # 返回函数应用到解除极坐标化的参数后的结果
        return eq.func(*[_unpolarify(x, exponents_only, exponents_only)
            for x in eq.args])

    # 对于其他情况，返回函数应用到解除极坐标化的参数后的结果
    return eq.func(*[_unpolarify(x, exponents_only, True) for x in eq.args])
def unpolarify(eq, subs=None, exponents_only=False):
    """
    如果 `p` 表示从对数的黎曼曲面到复数线的投影，
    返回 `eq` 的简化版本 `eq'`，使得 `p(eq') = p(eq)`。
    最后应用替换 subs。（这是一种方便，因为在某种意义上，`unpolarify` 是 :func:`polarify` 的反操作。）

    Examples
    ========

    >>> from sympy import unpolarify, polar_lift, sin, I
    >>> unpolarify(polar_lift(I + 2))
    2 + I
    >>> unpolarify(sin(polar_lift(I + 7)))
    sin(7 + I)
    """

    # 如果 eq 是布尔值，直接返回
    if isinstance(eq, bool):
        return eq

    # 将 eq 转换为 SymPy 表达式
    eq = sympify(eq)

    # 如果指定了 subs 参数，则应用替换
    if subs is not None:
        return unpolarify(eq.subs(subs))

    # 初始化变量
    changed = True
    pause = False

    # 如果 exponents_only 为 True，则设置 pause 为 True
    if exponents_only:
        pause = True

    # 循环直到不再发生变化
    while changed:
        changed = False
        # 调用 _unpolarify 函数处理 eq
        res = _unpolarify(eq, exponents_only, pause)
        # 如果结果改变了，则更新 eq 并继续循环
        if res != eq:
            changed = True
            eq = res
        # 如果结果是布尔值，直接返回
        if isinstance(res, bool):
            return res

    # 最后，替换 Exp(0) 为 1 是始终正确的。
    # 同样，polar_lift(0) -> 0 也是正确的。
    from sympy.functions.elementary.exponential import exp_polar
    return res.subs({exp_polar(0): 1, polar_lift(0): 0})
```