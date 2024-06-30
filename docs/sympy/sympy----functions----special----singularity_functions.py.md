# `D:\src\scipysrc\sympy\sympy\functions\special\singularity_functions.py`

```
# 从 sympy 库中导入必要的模块和函数
from sympy.core import S, oo, diff                     # 导入 S, oo, diff 函数
from sympy.core.function import Function, ArgumentIndexError   # 导入 Function 类和 ArgumentIndexError 异常类
from sympy.core.logic import fuzzy_not                  # 导入 fuzzy_not 函数
from sympy.core.relational import Eq                    # 导入 Eq 函数
from sympy.functions.elementary.complexes import im    # 导入 im 函数
from sympy.functions.elementary.piecewise import Piecewise   # 导入 Piecewise 函数
from sympy.functions.special.delta_functions import Heaviside   # 导入 Heaviside 函数

###############################################################################
############################# SINGULARITY FUNCTION ############################
###############################################################################

# 定义 SingularityFunction 类，继承自 Function 类
class SingularityFunction(Function):
    r"""
    Singularity functions are a class of discontinuous functions.

    Explanation
    ===========

    Singularity functions take a variable, an offset, and an exponent as
    arguments. These functions are represented using Macaulay brackets as:

    SingularityFunction(x, a, n) := <x - a>^n

    The singularity function will automatically evaluate to
    ``Derivative(DiracDelta(x - a), x, -n - 1)`` if ``n < 0``
    and ``(x - a)**n*Heaviside(x - a, 1)`` if ``n >= 0``.

    Examples
    ========

    >>> from sympy import SingularityFunction, diff, Piecewise, DiracDelta, Heaviside, Symbol
    >>> from sympy.abc import x, a, n
    >>> SingularityFunction(x, a, n)
    SingularityFunction(x, a, n)
    >>> y = Symbol('y', positive=True)
    >>> n = Symbol('n', nonnegative=True)
    >>> SingularityFunction(y, -10, n)
    (y + 10)**n
    >>> y = Symbol('y', negative=True)
    >>> SingularityFunction(y, 10, n)
    0
    >>> SingularityFunction(x, 4, -1).subs(x, 4)
    oo
    >>> SingularityFunction(x, 10, -2).subs(x, 10)
    oo
    >>> SingularityFunction(4, 1, 5)
    243
    >>> diff(SingularityFunction(x, 1, 5) + SingularityFunction(x, 1, 4), x)
    4*SingularityFunction(x, 1, 3) + 5*SingularityFunction(x, 1, 4)
    >>> diff(SingularityFunction(x, 4, 0), x, 2)
    SingularityFunction(x, 4, -2)
    >>> SingularityFunction(x, 4, 5).rewrite(Piecewise)
    Piecewise(((x - 4)**5, x >= 4), (0, True))
    >>> expr = SingularityFunction(x, a, n)
    >>> y = Symbol('y', positive=True)
    >>> n = Symbol('n', nonnegative=True)
    >>> expr.subs({x: y, a: -10, n: n})
    (y + 10)**n

    The methods ``rewrite(DiracDelta)``, ``rewrite(Heaviside)``, and
    ``rewrite('HeavisideDiracDelta')`` returns the same output. One can use any
    of these methods according to their choice.

    >>> expr = SingularityFunction(x, 4, 5) + SingularityFunction(x, -3, -1) - SingularityFunction(x, 0, -2)
    >>> expr.rewrite(Heaviside)
    (x - 4)**5*Heaviside(x - 4, 1) + DiracDelta(x + 3) - DiracDelta(x, 1)
    >>> expr.rewrite(DiracDelta)
    (x - 4)**5*Heaviside(x - 4, 1) + DiracDelta(x + 3) - DiracDelta(x, 1)
    >>> expr.rewrite('HeavisideDiracDelta')
    (x - 4)**5*Heaviside(x - 4, 1) + DiracDelta(x + 3) - DiracDelta(x, 1)

    See Also
    ========

    DiracDelta, Heaviside

    References
    ==========
    """
    Defines a class representing a DiracDelta function.

    Explanation
    ===========

    This class represents a DiracDelta function and provides methods to compute
    its derivative (`fdiff` method). The `fdiff` method calculates the first
    derivative of the DiracDelta function with respect to one of its arguments.

    """

    # 初始化一个布尔型变量，表示函数是否是实数函数
    is_real = True

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of a DiracDelta Function.

        Explanation
        ===========

        This method computes the first derivative of the DiracDelta function
        with respect to its argument. It handles different cases based on the
        exponent `n` of the DiracDelta function:
        - If `n` is zero, negative one, negative two, or negative three, it
          returns the function with `n` reduced by one.
        - If `n` is positive, it returns the function multiplied by `n-1`.

        Parameters
        ----------
        argindex : int, optional
            Index of the argument with respect to which the derivative is taken.
            Default is 1.

        Returns
        -------
        expr
            The resulting expression after differentiation.

        Raises
        ------
        ArgumentIndexError
            If `argindex` is not 1, indicating an unsupported argument index.

        """

        # 根据参数索引判断要对哪个参数求导
        if argindex == 1:
            # 解析函数参数
            x, a, n = self.args
            # 根据 DiracDelta 函数的阶数 n 进行不同的求导处理
            if n in (S.Zero, S.NegativeOne, S(-2), S(-3)):
                return self.func(x, a, n-1)
            elif n.is_positive:
                return n*self.func(x, a, n-1)
        else:
            # 抛出参数索引错误异常，因为只支持对第一个参数求导
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, variable, offset, exponent):
        """
        Returns a simplified form or a value of Singularity Function depending
        on the argument passed by the object.

        Explanation
        ===========

        The ``eval()`` method is automatically called when the
        ``SingularityFunction`` class is about to be instantiated and it
        returns either some simplified instance or the unevaluated instance
        depending on the argument passed. In other words, ``eval()`` method is
        not needed to be called explicitly, it is being called and evaluated
        once the object is called.

        Examples
        ========

        >>> from sympy import SingularityFunction, Symbol, nan
        >>> from sympy.abc import x, a, n
        >>> SingularityFunction(x, a, n)
        SingularityFunction(x, a, n)
        >>> SingularityFunction(5, 3, 2)
        4
        >>> SingularityFunction(x, a, nan)
        nan
        >>> SingularityFunction(x, 3, 0).subs(x, 3)
        1
        >>> SingularityFunction(4, 1, 5)
        243
        >>> x = Symbol('x', positive = True)
        >>> a = Symbol('a', negative = True)
        >>> n = Symbol('n', nonnegative = True)
        >>> SingularityFunction(x, a, n)
        (-a + x)**n
        >>> x = Symbol('x', negative = True)
        >>> a = Symbol('a', positive = True)
        >>> SingularityFunction(x, a, n)
        0

        """

        x = variable  # Assign the variable `x`
        a = offset    # Assign the offset `a`
        n = exponent  # Assign the exponent `n`
        shift = (x - a)  # Calculate the shift (difference between x and a)

        # Check for conditions where Singularity Functions cannot be evaluated
        if fuzzy_not(im(shift).is_zero):
            raise ValueError("Singularity Functions are defined only for Real Numbers.")
        if fuzzy_not(im(n).is_zero):
            raise ValueError("Singularity Functions are not defined for imaginary exponents.")
        if shift is S.NaN or n is S.NaN:
            return S.NaN
        if (n + 4).is_negative:
            raise ValueError("Singularity Functions are not defined for exponents less than -4.")
        if shift.is_extended_negative:
            return S.Zero
        if n.is_nonnegative:
            if shift.is_zero:  # Return 0 if shift is zero
                return S.Zero**n
            if shift.is_extended_nonnegative:  # Return shift raised to the power of n if shift is nonnegative
                return shift**n
        if n in (S.NegativeOne, -2, -3, -4):
            if shift.is_negative or shift.is_extended_positive:
                return S.Zero  # Return 0 if shift is negative or positive
            if shift.is_zero:
                return oo  # Return infinity if shift is zero

    def _eval_rewrite_as_Piecewise(self, *args, **kwargs):
        '''
        Converts a Singularity Function expression into its Piecewise form.

        '''

        x, a, n = self.args  # Unpack variables x, a, n from self.args

        # Rewrite Singularity Function based on its conditions into Piecewise form
        if n in (S.NegativeOne, S(-2), S(-3), S(-4)):
            return Piecewise((oo, Eq(x - a, 0)), (0, True))  # Piecewise representation for negative exponents
        elif n.is_nonnegative:
            return Piecewise(((x - a)**n, x - a >= 0), (0, True))  # Piecewise representation for nonnegative exponents
    def _eval_rewrite_as_Heaviside(self, *args, **kwargs):
        '''
        使用 Heaviside 函数和 Dirac Delta 函数重写 Singularity Function 表达式。

        '''
        # 提取参数 x, a, n
        x, a, n = self.args

        # 如果 n 等于 -4，返回 Heaviside(x - a) 的 4 阶导数
        if n == -4:
            return diff(Heaviside(x - a), x.free_symbols.pop(), 4)
        # 如果 n 等于 -3，返回 Heaviside(x - a) 的 3 阶导数
        if n == -3:
            return diff(Heaviside(x - a), x.free_symbols.pop(), 3)
        # 如果 n 等于 -2，返回 Heaviside(x - a) 的 2 阶导数
        if n == -2:
            return diff(Heaviside(x - a), x.free_symbols.pop(), 2)
        # 如果 n 等于 -1，返回 Heaviside(x - a) 的 1 阶导数
        if n == -1:
            return diff(Heaviside(x - a), x.free_symbols.pop(), 1)
        # 如果 n 是非负数，返回 (x - a)**n * Heaviside(x - a, 1)
        if n.is_nonnegative:
            return (x - a)**n * Heaviside(x - a, 1)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        z, a, n = self.args
        # 计算 z - a 在 x=0 处的值
        shift = (z - a).subs(x, 0)
        # 如果 n 小于 0，返回零
        if n < 0:
            return S.Zero
        # 如果 n 是零且 shift 是零，根据 cdir 的值返回零或一
        elif n.is_zero and shift.is_zero:
            return S.Zero if cdir == -1 else S.One
        # 如果 shift 是正数，返回 shift 的 n 次方
        elif shift.is_positive:
            return shift**n
        # 其他情况返回零
        return S.Zero

    def _eval_nseries(self, x, n, logx=None, cdir=0):
        z, a, n = self.args
        # 计算 z - a 在 x=0 处的值
        shift = (z - a).subs(x, 0)
        # 如果 n 小于 0，返回零
        if n < 0:
            return S.Zero
        # 如果 n 是零且 shift 是零，根据 cdir 的值返回零或一
        elif n.is_zero and shift.is_zero:
            return S.Zero if cdir == -1 else S.One
        # 如果 shift 是正数，递归计算 (z - a)**n 的 nseries 展开
        elif shift.is_positive:
            return ((z - a)**n)._eval_nseries(x, n, logx=logx, cdir=cdir)
        # 其他情况返回零
        return S.Zero

    # 将 _eval_rewrite_as_DiracDelta 和 _eval_rewrite_as_HeavisideDiracDelta 都重写为 _eval_rewrite_as_Heaviside
    _eval_rewrite_as_DiracDelta = _eval_rewrite_as_Heaviside
    _eval_rewrite_as_HeavisideDiracDelta = _eval_rewrite_as_Heaviside
```