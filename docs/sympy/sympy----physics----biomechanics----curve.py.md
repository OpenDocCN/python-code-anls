# `D:\src\scipysrc\sympy\sympy\physics\biomechanics\curve.py`

```
# 导入必要的模块和类
from dataclasses import dataclass
from sympy.core.expr import UnevaluatedExpr
from sympy.core.function import ArgumentIndexError, Function
from sympy.core.numbers import Float, Integer
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import cosh, sinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.printing.precedence import PRECEDENCE

# 定义可以被公开访问的类名列表
__all__ = [
    'CharacteristicCurveCollection',
    'CharacteristicCurveFunction',
    'FiberForceLengthActiveDeGroote2016',
    'FiberForceLengthPassiveDeGroote2016',
    'FiberForceLengthPassiveInverseDeGroote2016',
    'FiberForceVelocityDeGroote2016',
    'FiberForceVelocityInverseDeGroote2016',
    'TendonForceLengthDeGroote2016',
    'TendonForceLengthInverseDeGroote2016',
]

# 定义特征曲线函数的基类
class CharacteristicCurveFunction(Function):
    """Base class for all musculotendon characteristic curve functions."""

    @classmethod
    def eval(cls):
        # 抛出错误，不允许直接实例化特征曲线基类
        msg = (
            f'Cannot directly instantiate {cls.__name__!r}, instances of '
            f'characteristic curves must be of a concrete subclass.'
        )
        raise TypeError(msg)

    def _print_code(self, printer):
        """Print code for the function defining the curve using a printer.

        Explanation
        ===========

        The order of operations may need to be controlled as constant folding
        the numeric terms within the equations of a musculotendon
        characteristic curve can sometimes results in a numerically-unstable
        expression.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print a string representation of the
            characteristic curve as valid code in the target language.
        """
        # 使用打印机对象打印特征曲线的代码表示
        return printer._print(printer.parenthesize(
            self.doit(deep=False, evaluate=False), PRECEDENCE['Atom'],
        ))

    # 下面的函数都是输出代码的特定方法，都调用了_print_code方法
    _ccode = _print_code
    _cupycode = _print_code
    _cxxcode = _print_code
    _fcode = _print_code
    _jaxcode = _print_code
    _lambdacode = _print_code
    _mpmathcode = _print_code
    _octave = _print_code
    _pythoncode = _print_code
    _numpycode = _print_code
    _scipycode = _print_code


class TendonForceLengthDeGroote2016(CharacteristicCurveFunction):
    r"""Tendon force-length curve based on De Groote et al., 2016 [1]_.

    Explanation
    ===========

    Gives the normalized tendon force produced as a function of normalized
    tendon length.

    The function is defined by the equation:

    $fl^T = c_0 \exp{c_3 \left( \tilde{l}^T - c_1 \right)} - c_2$

    with constant values of $c_0 = 0.2$, $c_1 = 0.995$, $c_2 = 0.25$, and
    $c_3 = 33.93669377311689$.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    @classmethod
    # 定义一个类方法，用于创建基于 De Groote 2016 肌腱力长特性的实例
    specific and required properties. For example, the function produces no
    force when the tendon is in an unstrained state. It also produces a force
    of 1 normalized unit when the tendon is under a 5% strain.

    Examples
    ========

    The preferred way to instantiate :class:`TendonForceLengthDeGroote2016` is using
    the :meth:`~.with_defaults` constructor because this will automatically
    populate the constants within the characteristic curve equation with the
    floating point values from the original publication. This constructor takes
    a single argument corresponding to normalized tendon length. We'll create a
    :class:`~.Symbol` called ``l_T_tilde`` to represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import TendonForceLengthDeGroote2016
    >>> l_T_tilde = Symbol('l_T_tilde')
    >>> fl_T = TendonForceLengthDeGroote2016.with_defaults(l_T_tilde)
    >>> fl_T
    TendonForceLengthDeGroote2016(l_T_tilde, 0.2, 0.995, 0.25,
    33.93669377311689)

    It's also possible to populate the four constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1, c2, c3 = symbols('c0 c1 c2 c3')
    >>> fl_T = TendonForceLengthDeGroote2016(l_T_tilde, c0, c1, c2, c3)
    >>> fl_T
    TendonForceLengthDeGroote2016(l_T_tilde, c0, c1, c2, c3)

    You don't just have to use symbols as the arguments, it's also possible to
    use expressions. Let's create a new pair of symbols, ``l_T`` and
    ``l_T_slack``, representing tendon length and tendon slack length
    respectively. We can then represent ``l_T_tilde`` as an expression, the
    ratio of these.

    >>> l_T, l_T_slack = symbols('l_T l_T_slack')
    >>> l_T_tilde = l_T/l_T_slack
    >>> fl_T = TendonForceLengthDeGroote2016.with_defaults(l_T_tilde)
    >>> fl_T
    TendonForceLengthDeGroote2016(l_T/l_T_slack, 0.2, 0.995, 0.25,
    33.93669377311689)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> fl_T.doit(evaluate=False)
    -0.25 + 0.2*exp(33.93669377311689*(l_T/l_T_slack - 0.995))

    The function can also be differentiated. We'll differentiate with respect
    to l_T using the ``diff`` method on an instance with the single positional
    argument ``l_T``.

    >>> fl_T.diff(l_T)
    6.787338754623378*exp(33.93669377311689*(l_T/l_T_slack - 0.995))/l_T_slack

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """
    def with_defaults(cls, l_T_tilde):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the tendon force-length function using the
        four constant values specified in the original publication.

        These have the values:

        $c_0 = 0.2$
        $c_1 = 0.995$
        $c_2 = 0.25$
        $c_3 = 33.93669377311689$

        Parameters
        ==========

        l_T_tilde : Any (sympifiable)
            Normalized tendon length.

        """
        # Define constant values as floating-point numbers
        c0 = Float('0.2')
        c1 = Float('0.995')
        c2 = Float('0.25')
        c3 = Float('33.93669377311689')
        # Return a new instance of the class with the specified constants
        return cls(l_T_tilde, c0, c1, c2, c3)

    @classmethod
    def eval(cls, l_T_tilde, c0, c1, c2, c3):
        """Evaluation of basic inputs.

        Parameters
        ==========

        l_T_tilde : Any (sympifiable)
            Normalized tendon length.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``0.2``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``0.995``.
        c2 : Any (sympifiable)
            The third constant in the characteristic equation. The published
            value is ``0.25``.
        c3 : Any (sympifiable)
            The fourth constant in the characteristic equation. The published
            value is ``33.93669377311689``.

        """
        # Placeholder method, typically used for evaluating expressions (not implemented here)
        pass

    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
        # Evaluate the expression numerically with given precision
        return self.doit(deep=False, evaluate=False)._eval_evalf(prec)

    def doit(self, deep=True, evaluate=True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``l_T_tilde`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
        # Extract arguments and constants
        l_T_tilde, *constants = self.args
        if deep:
            hints['evaluate'] = evaluate
            # Recursively evaluate l_T_tilde and constants if deep=True
            l_T_tilde = l_T_tilde.doit(deep=deep, **hints)
            c0, c1, c2, c3 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            # Use the constants as they are if deep=False
            c0, c1, c2, c3 = constants

        if evaluate:
            # Return evaluated expression using constants and l_T_tilde
            return c0 * exp(c3 * (l_T_tilde - c1)) - c2

        # Return unevaluated expression with constants and l_T_tilde
        return c0 * exp(c3 * UnevaluatedExpr(l_T_tilde - c1)) - c2
   `
    def fdiff(self, argindex=1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
        # 解包函数的参数，赋值给相应的变量
        l_T_tilde, c0, c1, c2, c3 = self.args
        # 如果指定的参数索引为 1，返回第一个参数对函数的导数表达式
        if argindex == 1:
            return c0*c3*exp(c3*UnevaluatedExpr(l_T_tilde - c1))
        # 如果指定的参数索引为 2，返回第二个参数对函数的导数表达式
        elif argindex == 2:
            return exp(c3*UnevaluatedExpr(l_T_tilde - c1))
        # 如果指定的参数索引为 3，返回第三个参数对函数的导数表达式
        elif argindex == 3:
            return -c0*c3*exp(c3*UnevaluatedExpr(l_T_tilde - c1))
        # 如果指定的参数索引为 4，返回常数 -1 作为第四个参数对函数的导数
        elif argindex == 4:
            return Integer(-1)
        # 如果指定的参数索引为 5，返回第五个参数对函数的导数表达式
        elif argindex == 5:
            return c0*(l_T_tilde - c1)*exp(c3*UnevaluatedExpr(l_T_tilde - c1))

        # 如果参数索引不在上述范围内，抛出 ArgumentIndexError 异常
        raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """Inverse function.

        Parameters
        ==========

        argindex : int
            Value to start indexing the arguments at. Default is ``1``.

        """
        # 返回反函数类 TendonForceLengthInverseDeGroote2016
        return TendonForceLengthInverseDeGroote2016

    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """
        # 获取函数定义中的第一个参数 l_T_tilde
        l_T_tilde = self.args[0]
        # 使用给定的 LaTeX 打印器打印 l_T_tilde 的 LaTeX 表示形式
        _l_T_tilde = printer._print(l_T_tilde)
        # 返回 LaTeX 表示的字符串，格式为 \operatorname{fl}^T \left( %s \right)
        return r'\operatorname{fl}^T \left( %s \right)' % _l_T_tilde
# 定义一个特征曲线函数类 TendonForceLengthInverseDeGroote2016，继承自 CharacteristicCurveFunction
class TendonForceLengthInverseDeGroote2016(CharacteristicCurveFunction):
    r"""Inverse tendon force-length curve based on De Groote et al., 2016 [1]_.

    Explanation
    ===========

    Gives the normalized tendon length that produces a specific normalized
    tendon force.

    The function is defined by the equation:

    ${fl^T}^{-1} = frac{\log{\frac{fl^T + c_2}{c_0}}}{c_3} + c_1$

    with constant values of $c_0 = 0.2$, $c_1 = 0.995$, $c_2 = 0.25$, and
    $c_3 = 33.93669377311689$. This function is the exact analytical inverse
    of the related tendon force-length curve ``TendonForceLengthDeGroote2016``.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces no
    force when the tendon is in an unstrained state. It also produces a force
    of 1 normalized unit when the tendon is under a 5% strain.

    Examples
    ========

    The preferred way to instantiate :class:`TendonForceLengthInverseDeGroote2016` is
    using the :meth:`~.with_defaults` constructor because this will automatically
    populate the constants within the characteristic curve equation with the
    floating point values from the original publication. This constructor takes
    a single argument corresponding to normalized tendon force-length, which is
    equal to the tendon force. We'll create a :class:`~.Symbol` called ``fl_T`` to
    represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import TendonForceLengthInverseDeGroote2016
    >>> fl_T = Symbol('fl_T')
    >>> l_T_tilde = TendonForceLengthInverseDeGroote2016.with_defaults(fl_T)
    >>> l_T_tilde
    TendonForceLengthInverseDeGroote2016(fl_T, 0.2, 0.995, 0.25,
    33.93669377311689)

    It's also possible to populate the four constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1, c2, c3 = symbols('c0 c1 c2 c3')
    >>> l_T_tilde = TendonForceLengthInverseDeGroote2016(fl_T, c0, c1, c2, c3)
    >>> l_T_tilde
    TendonForceLengthInverseDeGroote2016(fl_T, c0, c1, c2, c3)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> l_T_tilde.doit(evaluate=False)
    c1 + log((c2 + fl_T)/c0)/c3

    The function can also be differentiated. We'll differentiate with respect
    to l_T using the ``diff`` method on an instance with the single positional
    argument ``l_T``.

    >>> l_T_tilde.diff(fl_T)
    1/(c3*(c2 + fl_T))

    References
    ==========

    """
    """
    [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936
    """

    @classmethod
    def with_defaults(cls, fl_T):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the inverse tendon force-length function
        using the four constant values specified in the original publication.

        These have the values:

        $c_0 = 0.2$
        $c_1 = 0.995$
        $c_2 = 0.25$
        $c_3 = 33.93669377311689$

        Parameters
        ==========

        fl_T : Any (sympifiable)
            Normalized tendon force as a function of tendon length.

        """
        # Define constants c0, c1, c2, c3 with their respective values as Float objects
        c0 = Float('0.2')
        c1 = Float('0.995')
        c2 = Float('0.25')
        c3 = Float('33.93669377311689')
        # Return a new instance of the class using the specified constants
        return cls(fl_T, c0, c1, c2, c3)

    @classmethod
    def eval(cls, fl_T, c0, c1, c2, c3):
        """Evaluation of basic inputs.

        Parameters
        ==========

        fl_T : Any (sympifiable)
            Normalized tendon force as a function of tendon length.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``0.2``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``0.995``.
        c2 : Any (sympifiable)
            The third constant in the characteristic equation. The published
            value is ``0.25``.
        c3 : Any (sympifiable)
            The fourth constant in the characteristic equation. The published
            value is ``33.93669377311689``.

        """
        # Placeholder function; typically would perform some evaluation based on inputs

    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
        # Evaluate the expression numerically with specified precision
        return self.doit(deep=False, evaluate=False)._eval_evalf(prec)
    def doit(self, deep=True, evaluate=True, **hints):
        """
        Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether `doit` should recursively call itself. Default is `True`.
        evaluate : bool.
            Whether the SymPy expression should be evaluated during construction.
            If `False`, constant folding is avoided for better numerical stability
            in sensible ranges of `l_T_tilde`. Default is `True`.
        **kwargs : dict[str, Any]
            Additional keyword arguments passed recursively to `doit`.

        """
        fl_T, *constants = self.args
        if deep:
            hints['evaluate'] = evaluate
            # Recursively evaluate fl_T and constants
            fl_T = fl_T.doit(deep=deep, **hints)
            c0, c1, c2, c3 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            c0, c1, c2, c3 = constants

        if evaluate:
            # Return the evaluated expression
            return log((fl_T + c2)/c0)/c3 + c1

        # Return the unevaluated expression
        return log(UnevaluatedExpr((fl_T + c2)/c0))/c3 + c1

    def fdiff(self, argindex=1):
        """
        Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            Index of the argument with respect to which the derivative is taken.
            Argument indexes start at `1`. Default is `1`.

        """
        fl_T, c0, c1, c2, c3 = self.args
        if argindex == 1:
            return 1/(c3*(fl_T + c2))
        elif argindex == 2:
            return -1/(c0*c3)
        elif argindex == 3:
            return Integer(1)
        elif argindex == 4:
            return 1/(c3*(fl_T + c2))
        elif argindex == 5:
            return -log(UnevaluatedExpr((fl_T + c2)/c0))/c3**2

        # Raise an error if the argument index is invalid
        raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Inverse function.

        Parameters
        ==========

        argindex : int
            Starting index for the function arguments. Default is `1`.

        """
        # Return the class representing the inverse function
        return TendonForceLengthDeGroote2016

    def _latex(self, printer):
        """
        Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer used to output the LaTeX string representation.

        """
        fl_T = self.args[0]
        _fl_T = printer._print(fl_T)
        return r'\left( \operatorname{fl}^T \right)^{-1} \left( %s \right)' % _fl_T
class FiberForceLengthPassiveDeGroote2016(CharacteristicCurveFunction):
    r"""Passive muscle fiber force-length curve based on De Groote et al., 2016
    [1]_.

    Explanation
    ===========

    The function is defined by the equation:

    $fl^M_{pas} = \frac{\frac{\exp{c_1 \left(\tilde{l^M} - 1\right)}}{c_0} - 1}{\exp{c_1} - 1}$

    with constant values of $c_0 = 0.6$ and $c_1 = 4.0$.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces a
    passive fiber force very close to 0 for all normalized fiber lengths
    between 0 and 1.

    Examples
    ========

    The preferred way to instantiate :class:`FiberForceLengthPassiveDeGroote2016` is
    using the :meth:`~.with_defaults` constructor because this will automatically
    populate the constants within the characteristic curve equation with the
    floating point values from the original publication. This constructor takes
    a single argument corresponding to normalized muscle fiber length. We'll
    create a :class:`~.Symbol` called ``l_M_tilde`` to represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import FiberForceLengthPassiveDeGroote2016
    >>> l_M_tilde = Symbol('l_M_tilde')
    >>> fl_M = FiberForceLengthPassiveDeGroote2016.with_defaults(l_M_tilde)
    >>> fl_M
    FiberForceLengthPassiveDeGroote2016(l_M_tilde, 0.6, 4.0)

    It's also possible to populate the two constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1 = symbols('c0 c1')
    >>> fl_M = FiberForceLengthPassiveDeGroote2016(l_M_tilde, c0, c1)
    >>> fl_M
    FiberForceLengthPassiveDeGroote2016(l_M_tilde, c0, c1)

    You don't just have to use symbols as the arguments, it's also possible to
    use expressions. Let's create a new pair of symbols, ``l_M`` and
    ``l_M_opt``, representing muscle fiber length and optimal muscle fiber
    length respectively. We can then represent ``l_M_tilde`` as an expression,
    the ratio of these.

    >>> l_M, l_M_opt = symbols('l_M l_M_opt')
    >>> l_M_tilde = l_M/l_M_opt
    >>> fl_M = FiberForceLengthPassiveDeGroote2016.with_defaults(l_M_tilde)
    >>> fl_M
    FiberForceLengthPassiveDeGroote2016(l_M/l_M_opt, 0.6, 4.0)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> fl_M.doit(evaluate=False)
    0.0186573603637741*(-1 + exp(6.66666666666667*(l_M/l_M_opt - 1)))

    The function can also be differentiated. We'll differentiate with respect
    to l_M using the ``diff`` method on an instance with the single positional
    argument ``l_M``.

    >>> fl_M.diff(l_M)
    # 计算肌肉纤维 passivity 功能的长度相关项
    0.12438240242516*exp(6.66666666666667*(l_M/l_M_opt - 1))/l_M_opt

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    @classmethod
    def with_defaults(cls, l_M_tilde):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the muscle fiber passive force-length
        function using the four constant values specified in the original
        publication.

        These have the values:

        $c_0 = 0.6$
        $c_1 = 4.0$

        Parameters
        ==========

        l_M_tilde : Any (sympifiable)
            Normalized muscle fiber length.

        """
        # 设置常数 c0 和 c1 为推荐的默认值
        c0 = Float('0.6')
        c1 = Float('4.0')
        return cls(l_M_tilde, c0, c1)

    @classmethod
    def eval(cls, l_M_tilde, c0, c1):
        """Evaluation of basic inputs.

        Parameters
        ==========

        l_M_tilde : Any (sympifiable)
            Normalized muscle fiber length.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``0.6``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``4.0``.

        """
        # 这是一个占位函数，用于评估基本输入，实际实现可能需要添加具体功能
        pass

    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
        # 通过 evalf 数值化地评估表达式
        return self.doit(deep=False, evaluate=False)._eval_evalf(prec)

    def doit(self, deep=True, evaluate=True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``l_T_tilde`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
        # 获取参数 l_M_tilde 和常数列表 constants
        l_M_tilde, *constants = self.args
        if deep:
            # 根据 deep 参数选择是否递归调用 doit
            hints['evaluate'] = evaluate
            l_M_tilde = l_M_tilde.doit(deep=deep, **hints)
            # 对 constants 列表中的常数进行递归调用 doit
            c0, c1 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            c0, c1 = constants

        if evaluate:
            # 如果 evaluate 参数为 True，返回数值化的表达式结果
            return (exp((c1*(l_M_tilde - 1))/c0) - 1)/(exp(c1) - 1)

        # 如果 evaluate 参数为 False，返回未评估的表达式结果
        return (exp((c1*UnevaluatedExpr(l_M_tilde - 1))/c0) - 1)/(exp(c1) - 1)
    # 计算函数关于单个参数的导数。
    def fdiff(self, argindex=1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
        # 从参数中获取函数的参数 l_M_tilde, c0, c1
        l_M_tilde, c0, c1 = self.args
        # 如果 argindex 为 1，返回对第一个参数的导数
        if argindex == 1:
            return c1 * exp(c1 * UnevaluatedExpr(l_M_tilde - 1) / c0) / (c0 * (exp(c1) - 1))
        # 如果 argindex 为 2，返回对第二个参数的导数
        elif argindex == 2:
            return (
                -c1 * exp(c1 * UnevaluatedExpr(l_M_tilde - 1) / c0)
                * UnevaluatedExpr(l_M_tilde - 1) / (c0**2 * (exp(c1) - 1))
            )
        # 如果 argindex 为 3，返回对第三个参数的导数
        elif argindex == 3:
            return (
                -exp(c1) * (-1 + exp(c1 * UnevaluatedExpr(l_M_tilde - 1) / c0)) / (exp(c1) - 1)**2
                + exp(c1 * UnevaluatedExpr(l_M_tilde - 1) / c0) * (l_M_tilde - 1) / (c0 * (exp(c1) - 1))
            )

        # 如果 argindex 超出参数范围，抛出参数索引错误
        raise ArgumentIndexError(self, argindex)

    # 返回反函数
    def inverse(self, argindex=1):
        """Inverse function.

        Parameters
        ==========

        argindex : int
            Value to start indexing the arguments at. Default is ``1``.

        """
        # 返回 FiberForceLengthPassiveInverseDeGroote2016 函数作为反函数
        return FiberForceLengthPassiveInverseDeGroote2016

    # 返回函数曲线的 LaTeX 表示
    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """
        # 获取函数的参数 l_M_tilde
        l_M_tilde = self.args[0]
        # 将 l_M_tilde 的 LaTeX 表示输出为字符串
        _l_M_tilde = printer._print(l_M_tilde)
        # 返回函数曲线的 LaTeX 表示
        return r'\operatorname{fl}^M_{pas} \left( %s \right)' % _l_M_tilde
# 定义一个新的类 FiberForceLengthPassiveInverseDeGroote2016，继承自 CharacteristicCurveFunction 类
class FiberForceLengthPassiveInverseDeGroote2016(CharacteristicCurveFunction):
    # 类的文档字符串，描述了这个类的作用和基于的文献引用
    r"""Inverse passive muscle fiber force-length curve based on De Groote et
    al., 2016 [1]_.

    Explanation
    ===========
    
    Gives the normalized muscle fiber length that produces a specific normalized
    passive muscle fiber force.

    The function is defined by the equation:

    ${fl^M_{pas}}^{-1} = \frac{c_0 \log{\left(\exp{c_1} - 1\right)fl^M_pas + 1}}{c_1} + 1$

    with constant values of $c_0 = 0.6$ and $c_1 = 4.0$. This function is the
    exact analytical inverse of the related tendon force-length curve
    ``FiberForceLengthPassiveDeGroote2016``.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces a
    passive fiber force very close to 0 for all normalized fiber lengths
    between 0 and 1.

    Examples
    ========

    The preferred way to instantiate
    :class:`FiberForceLengthPassiveInverseDeGroote2016` is using the
    :meth:`~.with_defaults` constructor because this will automatically populate the
    constants within the characteristic curve equation with the floating point
    values from the original publication. This constructor takes a single
    argument corresponding to the normalized passive muscle fiber length-force
    component of the muscle fiber force. We'll create a :class:`~.Symbol` called
    ``fl_M_pas`` to represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import FiberForceLengthPassiveInverseDeGroote2016
    >>> fl_M_pas = Symbol('fl_M_pas')
    >>> l_M_tilde = FiberForceLengthPassiveInverseDeGroote2016.with_defaults(fl_M_pas)
    >>> l_M_tilde
    FiberForceLengthPassiveInverseDeGroote2016(fl_M_pas, 0.6, 4.0)

    It's also possible to populate the two constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1 = symbols('c0 c1')
    >>> l_M_tilde = FiberForceLengthPassiveInverseDeGroote2016(fl_M_pas, c0, c1)
    >>> l_M_tilde
    FiberForceLengthPassiveInverseDeGroote2016(fl_M_pas, c0, c1)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> l_M_tilde.doit(evaluate=False)
    c0*log(1 + fl_M_pas*(exp(c1) - 1))/c1 + 1

    The function can also be differentiated. We'll differentiate with respect
    to fl_M_pas using the ``diff`` method on an instance with the single positional
    argument ``fl_M_pas``.

    >>> l_M_tilde.diff(fl_M_pas)
    c0*(exp(c1) - 1)/(c1*(fl_M_pas*(exp(c1) - 1) + 1))

    References
    ==========

    """
    """
    [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    @classmethod
    def with_defaults(cls, fl_M_pas):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the inverse muscle fiber passive force-length
        function using the four constant values specified in the original
        publication.

        These have the values:

        $c_0 = 0.6$
        $c_1 = 4.0$

        Parameters
        ==========

        fl_M_pas : Any (sympifiable)
            Normalized passive muscle fiber force as a function of muscle fiber
            length.

        """
        # Define constant c0 with a floating-point value of '0.6'
        c0 = Float('0.6')
        # Define constant c1 with a floating-point value of '4.0'
        c1 = Float('4.0')
        # Return a new instance of the class with the provided constants
        return cls(fl_M_pas, c0, c1)

    @classmethod
    def eval(cls, fl_M_pas, c0, c1):
        """Evaluation of basic inputs.

        Parameters
        ==========

        fl_M_pas : Any (sympifiable)
            Normalized passive muscle fiber force.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``0.6``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``4.0``.

        """
        # Placeholder function, does nothing in this implementation
        pass

    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
        # Evaluate the SymPy expression numerically with precision 'prec'
        return self.doit(deep=False, evaluate=False)._eval_evalf(prec)

    def doit(self, deep=True, evaluate=True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``l_T_tilde`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
        # Extract the first argument as fl_M_pas and remaining as constants
        fl_M_pas, *constants = self.args
        if deep:
            # If deep evaluation is requested, recursively evaluate fl_M_pas and constants
            hints['evaluate'] = evaluate
            fl_M_pas = fl_M_pas.doit(deep=deep, **hints)
            c0, c1 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            # Otherwise, use constants directly
            c0, c1 = constants

        if evaluate:
            # If evaluation is requested, compute the expression numerically
            return c0*log(fl_M_pas*(exp(c1) - 1) + 1)/c1 + 1

        # If evaluation is not requested, return unevaluated expression
        return c0*log(UnevaluatedExpr(fl_M_pas*(exp(c1) - 1)) + 1)/c1 + 1
    def fdiff(self, argindex=1):
        """计算函数对单个参数的导数。

        Parameters
        ==========

        argindex : int
            要对其进行导数计算的函数参数的索引。参数索引从 ``1`` 开始。
            默认为 ``1``。

        """
        fl_M_pas, c0, c1 = self.args
        if argindex == 1:
            return c0*(exp(c1) - 1)/(c1*(fl_M_pas*(exp(c1) - 1) + 1))
        elif argindex == 2:
            return log(fl_M_pas*(exp(c1) - 1) + 1)/c1
        elif argindex == 3:
            return (
                c0*fl_M_pas*exp(c1)/(c1*(fl_M_pas*(exp(c1) - 1) + 1))
                - c0*log(fl_M_pas*(exp(c1) - 1) + 1)/c1**2
            )

        raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """返回反函数。

        Parameters
        ==========

        argindex : int
            开始索引参数的值。默认为 ``1``。

        """
        return FiberForceLengthPassiveDeGroote2016

    def _latex(self, printer):
        """打印函数曲线的 LaTeX 表示。

        Parameters
        ==========

        printer : Printer
            用于打印 LaTeX 字符串表示的打印机。

        """
        fl_M_pas = self.args[0]
        _fl_M_pas = printer._print(fl_M_pas)
        return r'\left( \operatorname{fl}^M_{pas} \right)^{-1} \left( %s \right)' % _fl_M_pas
class FiberForceLengthActiveDeGroote2016(CharacteristicCurveFunction):
    r"""Active muscle fiber force-length curve based on De Groote et al., 2016
    [1]_.

    Explanation
    ===========

    The function is defined by the equation:

    $fl_{\text{act}}^M = c_0 \exp\left(-\frac{1}{2}\left(\frac{\tilde{l}^M - c_1}{c_2 + c_3 \tilde{l}^M}\right)^2\right)
    + c_4 \exp\left(-\frac{1}{2}\left(\frac{\tilde{l}^M - c_5}{c_6 + c_7 \tilde{l}^M}\right)^2\right)
    + c_8 \exp\left(-\frac{1}{2}\left(\frac{\tilde{l}^M - c_9}{c_{10} + c_{11} \tilde{l}^M}\right)^2\right)$

    with constant values of $c0 = 0.814$, $c1 = 1.06$, $c2 = 0.162$,
    $c3 = 0.0633$, $c4 = 0.433$, $c5 = 0.717$, $c6 = -0.0299$, $c7 = 0.2$,
    $c8 = 0.1$, $c9 = 1.0$, $c10 = 0.354$, and $c11 = 0.0$.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces a
    active fiber force of 1 at a normalized fiber length of 1, and an active
    fiber force of 0 at normalized fiber lengths of 0 and 2.

    Examples
    ========

    The preferred way to instantiate :class:`FiberForceLengthActiveDeGroote2016` is
    using the :meth:`~.with_defaults` constructor because this will automatically
    populate the constants within the characteristic curve equation with the
    floating point values from the original publication. This constructor takes
    a single argument corresponding to normalized muscle fiber length. We'll
    create a :class:`~.Symbol` called ``l_M_tilde`` to represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import FiberForceLengthActiveDeGroote2016
    >>> l_M_tilde = Symbol('l_M_tilde')
    >>> fl_M = FiberForceLengthActiveDeGroote2016.with_defaults(l_M_tilde)
    >>> fl_M
    FiberForceLengthActiveDeGroote2016(l_M_tilde, 0.814, 1.06, 0.162, 0.0633,
    0.433, 0.717, -0.0299, 0.2, 0.1, 1.0, 0.354, 0.0)

    It's also possible to populate the two constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = symbols('c0:12')
    >>> fl_M = FiberForceLengthActiveDeGroote2016(l_M_tilde, c0, c1, c2, c3,
    ...     c4, c5, c6, c7, c8, c9, c10, c11)
    >>> fl_M
    FiberForceLengthActiveDeGroote2016(l_M_tilde, c0, c1, c2, c3, c4, c5, c6,
    c7, c8, c9, c10, c11)

    You don't just have to use symbols as the arguments, it's also possible to
    use expressions. Let's create a new pair of symbols, ``l_M`` and
    ``l_M_opt``, representing muscle fiber length and optimal muscle fiber
    length respectively. We can then represent ``l_M_tilde`` as an expression,
    the ratio of these.

    >>> l_M, l_M_opt = symbols('l_M l_M_opt')
    >>> l_M_tilde = l_M/l_M_opt
    >>> fl_M = FiberForceLengthActiveDeGroote2016.with_defaults(l_M_tilde)
    >>> fl_M
    FiberForceLengthActiveDeGroote2016(l_M/l_M_opt, 0.814, 1.06, 0.162, 0.0633,
    0.433, 0.717, -0.0299, 0.2, 0.1, 1.0, 0.354, 0.0)

# 调用函数 `FiberForceLengthActiveDeGroote2016`，传入多个参数，计算并返回肌纤维长度对应的活跃力长度。参数包括肌纤维长度与最优长度的比例，以及其他控制参数。


    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> fl_M.doit(evaluate=False)
    0.814*exp(-19.0519737844841*(l_M/l_M_opt
    - 1.06)**2/(0.390740740740741*l_M/l_M_opt + 1)**2)
    + 0.433*exp(-12.5*(l_M/l_M_opt - 0.717)**2/(l_M/l_M_opt - 0.1495)**2)
    + 0.1*exp(-3.98991349867535*(l_M/l_M_opt - 1.0)**2)

# 对于 `fl_M` 实例，调用 `doit` 方法来检查该函数代表的符号表达式。使用 `evaluate=False` 关键字参数以保持表达式的标准形式，不简化任何常数。


    The function can also be differentiated. We'll differentiate with respect
    to l_M using the ``diff`` method on an instance with the single positional
    argument ``l_M``.

    >>> fl_M.diff(l_M)
    ((-0.79798269973507*l_M/l_M_opt
    + 0.79798269973507)*exp(-3.98991349867535*(l_M/l_M_opt - 1.0)**2)
    + (10.825*(-l_M/l_M_opt + 0.717)/(l_M/l_M_opt - 0.1495)**2
    + 10.825*(l_M/l_M_opt - 0.717)**2/(l_M/l_M_opt
    - 0.1495)**3)*exp(-12.5*(l_M/l_M_opt - 0.717)**2/(l_M/l_M_opt - 0.1495)**2)
    + (31.0166133211401*(-l_M/l_M_opt + 1.06)/(0.390740740740741*l_M/l_M_opt
    + 1)**2 + 13.6174190361677*(0.943396226415094*l_M/l_M_opt
    - 1)**2/(0.390740740740741*l_M/l_M_opt
    + 1)**3)*exp(-21.4067977442463*(0.943396226415094*l_M/l_M_opt
    - 1)**2/(0.390740740740741*l_M/l_M_opt + 1)**2))/l_M_opt

# 可以对该函数进行微分。使用 `diff` 方法对 `fl_M` 实例相对于 `l_M` 进行微分，其中 `l_M` 是唯一的位置参数。


    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    @classmethod

# 提供引用文献，描述了解决肌肉冗余问题的直接拼接最优控制问题公式评估的方法。
    def with_defaults(cls, l_M_tilde):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the inverse muscle fiber act force-length
        function using the four constant values specified in the original
        publication.

        These have the values:

        $c0 = 0.814$
        $c1 = 1.06$
        $c2 = 0.162$
        $c3 = 0.0633$
        $c4 = 0.433$
        $c5 = 0.717$
        $c6 = -0.0299$
        $c7 = 0.2$
        $c8 = 0.1$
        $c9 = 1.0$
        $c10 = 0.354$
        $c11 = 0.0$

        Parameters
        ==========

        fl_M_act : Any (sympifiable)
            Normalized passive muscle fiber force as a function of muscle fiber
            length.

        """
        # Define constants with specific float values
        c0 = Float('0.814')
        c1 = Float('1.06')
        c2 = Float('0.162')
        c3 = Float('0.0633')
        c4 = Float('0.433')
        c5 = Float('0.717')
        c6 = Float('-0.0299')
        c7 = Float('0.2')
        c8 = Float('0.1')
        c9 = Float('1.0')
        c10 = Float('0.354')
        c11 = Float('0.0')
        
        # Return a new instance of the class with the specified constants
        return cls(l_M_tilde, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11)

    @classmethod
    # 定义一个类方法，用于评估基本输入参数。
    def eval(cls, l_M_tilde, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11):
        """Evaluation of basic inputs.

        Parameters
        ==========

        l_M_tilde : Any (sympifiable)
            Normalized muscle fiber length.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``0.814``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``1.06``.
        c2 : Any (sympifiable)
            The third constant in the characteristic equation. The published
            value is ``0.162``.
        c3 : Any (sympifiable)
            The fourth constant in the characteristic equation. The published
            value is ``0.0633``.
        c4 : Any (sympifiable)
            The fifth constant in the characteristic equation. The published
            value is ``0.433``.
        c5 : Any (sympifiable)
            The sixth constant in the characteristic equation. The published
            value is ``0.717``.
        c6 : Any (sympifiable)
            The seventh constant in the characteristic equation. The published
            value is ``-0.0299``.
        c7 : Any (sympifiable)
            The eighth constant in the characteristic equation. The published
            value is ``0.2``.
        c8 : Any (sympifiable)
            The ninth constant in the characteristic equation. The published
            value is ``0.1``.
        c9 : Any (sympifiable)
            The tenth constant in the characteristic equation. The published
            value is ``1.0``.
        c10 : Any (sympifiable)
            The eleventh constant in the characteristic equation. The published
            value is ``0.354``.
        c11 : Any (sympifiable)
            The twelfth constant in the characteristic equation. The published
            value is ``0.0``.

        """
        # 此处为占位符，目前函数体未实现任何具体功能，故直接 pass
        pass

    # 定义一个方法，用于在给定精度下数值求解表达式，使用 evalf 方法进行数值求解。
    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
        # 调用 doit 方法，深度优先，不进行求值，再调用 _eval_evalf 方法进行数值求解
        return self.doit(deep=False, evaluate=False)._eval_evalf(prec)
    def doit(self, deep=True, evaluate=True, **hints):
        """
        Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``l_M_tilde`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.
        """
        # 解包 self.args 获取 l_M_tilde 和 constants
        l_M_tilde, *constants = self.args

        # 如果 deep 参数为 True，则设置 hints 中的 evaluate 参数为当前 evaluate 值
        if deep:
            hints['evaluate'] = evaluate
            # 对 l_M_tilde 和 constants 中的每个常数进行 doit 操作
            l_M_tilde = l_M_tilde.doit(deep=deep, **hints)
            constants = [c.doit(deep=deep, **hints) for c in constants]

        # 将 constants 解包为 c0 到 c11
        c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = constants

        # 如果 evaluate 参数为 True，则返回计算后的表达式
        if evaluate:
            return (
                c0*exp(-(((l_M_tilde - c1)/(c2 + c3*l_M_tilde))**2)/2)
                + c4*exp(-(((l_M_tilde - c5)/(c6 + c7*l_M_tilde))**2)/2)
                + c8*exp(-(((l_M_tilde - c9)/(c10 + c11*l_M_tilde))**2)/2)
            )
        
        # 如果 evaluate 参数为 False，则返回未评估的表达式
        return (
            c0*exp(-((UnevaluatedExpr(l_M_tilde - c1)/(c2 + c3*l_M_tilde))**2)/2)
            + c4*exp(-((UnevaluatedExpr(l_M_tilde - c5)/(c6 + c7*l_M_tilde))**2)/2)
            + c8*exp(-((UnevaluatedExpr(l_M_tilde - c9)/(c10 + c11*l_M_tilde))**2)/2)
        )

    def _latex(self, printer):
        """
        Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.
        """
        # 获取 self.args 中的 l_M_tilde
        l_M_tilde = self.args[0]
        # 打印 l_M_tilde 的 LaTeX 表示，并返回格式化后的 LaTeX 字符串
        _l_M_tilde = printer._print(l_M_tilde)
        return r'\operatorname{fl}^M_{act} \left( %s \right)' % _l_M_tilde
class FiberForceVelocityDeGroote2016(CharacteristicCurveFunction):
    r"""Muscle fiber force-velocity curve based on De Groote et al., 2016 [1]_.

    Explanation
    ===========

    Gives the normalized muscle fiber force produced as a function of
    normalized tendon velocity.

    The function is defined by the equation:

    $fv^M = c_0 \log{\left(c_1 \tilde{v}_m + c_2\right) + \sqrt{\left(c_1 \tilde{v}_m + c_2\right)^2 + 1}} + c_3$

    with constant values of $c_0 = -0.318$, $c_1 = -8.149$, $c_2 = -0.374$, and
    $c_3 = 0.886$.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces a
    normalized muscle fiber force of 1 when the muscle fibers are contracting
    isometrically (they have an extension rate of 0).

    Examples
    ========

    The preferred way to instantiate :class:`FiberForceVelocityDeGroote2016` is using
    the :meth:`~.with_defaults` constructor because this will automatically populate
    the constants within the characteristic curve equation with the floating
    point values from the original publication. This constructor takes a single
    argument corresponding to normalized muscle fiber extension velocity. We'll
    create a :class:`~.Symbol` called ``v_M_tilde`` to represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import FiberForceVelocityDeGroote2016
    >>> v_M_tilde = Symbol('v_M_tilde')
    >>> fv_M = FiberForceVelocityDeGroote2016.with_defaults(v_M_tilde)
    >>> fv_M
    FiberForceVelocityDeGroote2016(v_M_tilde, -0.318, -8.149, -0.374, 0.886)

    It's also possible to populate the four constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1, c2, c3 = symbols('c0 c1 c2 c3')
    >>> fv_M = FiberForceVelocityDeGroote2016(v_M_tilde, c0, c1, c2, c3)
    >>> fv_M
    FiberForceVelocityDeGroote2016(v_M_tilde, c0, c1, c2, c3)

    You don't just have to use symbols as the arguments, it's also possible to
    use expressions. Let's create a new pair of symbols, ``v_M`` and
    ``v_M_max``, representing muscle fiber extension velocity and maximum
    muscle fiber extension velocity respectively. We can then represent
    ``v_M_tilde`` as an expression, the ratio of these.

    >>> v_M, v_M_max = symbols('v_M v_M_max')
    >>> v_M_tilde = v_M/v_M_max
    >>> fv_M = FiberForceVelocityDeGroote2016.with_defaults(v_M_tilde)
    >>> fv_M
    FiberForceVelocityDeGroote2016(v_M/v_M_max, -0.318, -8.149, -0.374, 0.886)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> fv_M.doit(evaluate=False)
    """
    # 继承自CharacteristicCurveFunction，表示De Groote et al., 2016年的肌肉纤维力速度曲线
    # 给出了随着标准化肌腱速度变化而产生的标准化肌肉纤维力的函数
    # 使用方程定义了该函数，其中的常数值分别为c0 = -0.318, c1 = -8.149, c2 = -0.374, c3 = 0.886
    # 这些常数值在原始文献中被精选以给予曲线特定的和所需的属性
    # 示例中展示了如何实例化和使用这个类，并说明了如何使用默认值或自定义常数值
    pass
    # Calculate the muscle fiber force-velocity relationship based on the given formula.
    # This equation involves a logarithm and square root operation.
    # It models the force-velocity behavior of muscle fibers.
    0.886 - 0.318*log(-8.149*v_M/v_M_max - 0.374 + sqrt(1 + (-8.149*v_M/v_M_max - 0.374)**2))

"""

    @classmethod
    def with_defaults(cls, v_M_tilde):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the muscle fiber force-velocity function
        using the four constant values specified in the original publication.

        These have the values:

        $c_0 = -0.318$
        $c_1 = -8.149$
        $c_2 = -0.374$
        $c_3 = 0.886$

        Parameters
        ==========

        v_M_tilde : Any (sympifiable)
            Normalized muscle fiber extension velocity.

        """
        c0 = Float('-0.318')
        c1 = Float('-8.149')
        c2 = Float('-0.374')
        c3 = Float('0.886')
        # Return a new instance of the class using the provided constants
        return cls(v_M_tilde, c0, c1, c2, c3)

    @classmethod
    def eval(cls, v_M_tilde, c0, c1, c2, c3):
        """Evaluation of basic inputs.

        Parameters
        ==========

        v_M_tilde : Any (sympifiable)
            Normalized muscle fiber extension velocity.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``-0.318``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``-8.149``.
        c2 : Any (sympifiable)
            The third constant in the characteristic equation. The published
            value is ``-0.374``.
        c3 : Any (sympifiable)
            The fourth constant in the characteristic equation. The published
            value is ``0.886``.

        """
        pass

    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
        # Evaluate the expression numerically with the given precision
        return self.doit(deep=False, evaluate=False)._eval_evalf(prec)
    def doit(self, deep=True, evaluate=True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable form for values of ``v_M_tilde`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
        # 提取第一个参数作为 `v_M_tilde`，其余参数作为 `constants`
        v_M_tilde, *constants = self.args
        
        # 如果 `deep` 参数为 True，则递归调用 `doit` 方法
        if deep:
            # 将 `evaluate` 参数传递给 `hints` 字典
            hints['evaluate'] = evaluate
            # 对 `v_M_tilde` 执行 `doit` 方法，递归调用，传递 `hints` 字典
            v_M_tilde = v_M_tilde.doit(deep=deep, **hints)
            # 对 `constants` 中的每个常数执行 `doit` 方法，递归调用，传递 `hints` 字典
            c0, c1, c2, c3 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            # 如果 `deep` 参数为 False，则直接使用 `constants` 中的值
            c0, c1, c2, c3 = constants
        
        # 如果 `evaluate` 参数为 True，则返回表达式的计算结果
        if evaluate:
            return c0*log(c1*v_M_tilde + c2 + sqrt((c1*v_M_tilde + c2)**2 + 1)) + c3
        
        # 如果 `evaluate` 参数为 False，则返回表达式的未计算状态，使用 `UnevaluatedExpr` 包裹
        return c0*log(c1*v_M_tilde + c2 + sqrt(UnevaluatedExpr(c1*v_M_tilde + c2)**2 + 1)) + c3

    def fdiff(self, argindex=1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
        # 从 `self.args` 中获取参数，`v_M_tilde` 是第一个参数，后面是常数参数 `c0, c1, c2, c3`
        v_M_tilde, c0, c1, c2, c3 = self.args
        
        # 根据 `argindex` 的值返回对应参数的偏导数
        if argindex == 1:
            return c0*c1/sqrt(UnevaluatedExpr(c1*v_M_tilde + c2)**2 + 1)
        elif argindex == 2:
            return log(
                c1*v_M_tilde + c2
                + sqrt(UnevaluatedExpr(c1*v_M_tilde + c2)**2 + 1)
            )
        elif argindex == 3:
            return c0*v_M_tilde/sqrt(UnevaluatedExpr(c1*v_M_tilde + c2)**2 + 1)
        elif argindex == 4:
            return c0/sqrt(UnevaluatedExpr(c1*v_M_tilde + c2)**2 + 1)
        elif argindex == 5:
            return Integer(1)
        
        # 如果 `argindex` 超出范围，则引发异常
        raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """Inverse function.

        Parameters
        ==========

        argindex : int
            Value to start indexing the arguments at. Default is ``1``.

        """
        # 返回反函数的类 `FiberForceVelocityInverseDeGroote2016`
        return FiberForceVelocityInverseDeGroote2016

    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """
        # 获取第一个参数 `v_M_tilde`
        v_M_tilde = self.args[0]
        # 使用打印机对象打印 `v_M_tilde` 的 LaTeX 表示
        _v_M_tilde = printer._print(v_M_tilde)
        # 返回带有 LaTeX 表示的函数字符串
        return r'\operatorname{fv}^M \left( %s \right)' % _v_M_tilde
# 定义一个类，继承自 CharacteristicCurveFunction，表示基于 De Groote 等人 2016 年的逆肌肉纤维力-速度曲线
class FiberForceVelocityInverseDeGroote2016(CharacteristicCurveFunction):
    r"""Inverse muscle fiber force-velocity curve based on De Groote et al.,
    2016 [1]_.

    Explanation
    ===========

    Gives the normalized muscle fiber velocity that produces a specific
    normalized muscle fiber force.

    The function is defined by the equation:

    ${fv^M}^{-1} = \frac{\sinh{\frac{fv^M - c_3}{c_0}} - c_2}{c_1}$

    with constant values of $c_0 = -0.318$, $c_1 = -8.149$, $c_2 = -0.374$, and
    $c_3 = 0.886$. This function is the exact analytical inverse of the related
    muscle fiber force-velocity curve ``FiberForceVelocityDeGroote2016``.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces a
    normalized muscle fiber force of 1 when the muscle fibers are contracting
    isometrically (they have an extension rate of 0).

    Examples
    ========

    The preferred way to instantiate :class:`FiberForceVelocityInverseDeGroote2016`
    is using the :meth:`~.with_defaults` constructor because this will automatically
    populate the constants within the characteristic curve equation with the
    floating point values from the original publication. This constructor takes
    a single argument corresponding to normalized muscle fiber force-velocity
    component of the muscle fiber force. We'll create a :class:`~.Symbol` called
    ``fv_M`` to represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import FiberForceVelocityInverseDeGroote2016
    >>> fv_M = Symbol('fv_M')
    >>> v_M_tilde = FiberForceVelocityInverseDeGroote2016.with_defaults(fv_M)
    >>> v_M_tilde
    FiberForceVelocityInverseDeGroote2016(fv_M, -0.318, -8.149, -0.374, 0.886)

    It's also possible to populate the four constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1, c2, c3 = symbols('c0 c1 c2 c3')
    >>> v_M_tilde = FiberForceVelocityInverseDeGroote2016(fv_M, c0, c1, c2, c3)
    >>> v_M_tilde
    FiberForceVelocityInverseDeGroote2016(fv_M, c0, c1, c2, c3)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> v_M_tilde.doit(evaluate=False)
    (-c2 + sinh((-c3 + fv_M)/c0))/c1

    The function can also be differentiated. We'll differentiate with respect
    to fv_M using the ``diff`` method on an instance with the single positional
    argument ``fv_M``.

    >>> v_M_tilde.diff(fv_M)
    cosh((-c3 + fv_M)/c0)/(c0*c1)

    References
    ==========

    """
    """
    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    @classmethod
    def with_defaults(cls, fv_M):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the inverse muscle fiber force-velocity
        function using the four constant values specified in the original
        publication.

        These have the values:

        $c_0 = -0.318$
        $c_1 = -8.149$
        $c_2 = -0.374$
        $c_3 = 0.886$

        Parameters
        ==========

        fv_M : Any (sympifiable)
            Normalized muscle fiber extension velocity.

        """
        # Define constants c0, c1, c2, c3 with specified values
        c0 = Float('-0.318')
        c1 = Float('-8.149')
        c2 = Float('-0.374')
        c3 = Float('0.886')
        # Return a new instance of the class using the specified constants
        return cls(fv_M, c0, c1, c2, c3)

    @classmethod
    def eval(cls, fv_M, c0, c1, c2, c3):
        """Evaluation of basic inputs.

        Parameters
        ==========

        fv_M : Any (sympifiable)
            Normalized muscle fiber force as a function of muscle fiber
            extension velocity.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``-0.318``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``-8.149``.
        c2 : Any (sympifiable)
            The third constant in the characteristic equation. The published
            value is ``-0.374``.
        c3 : Any (sympifiable)
            The fourth constant in the characteristic equation. The published
            value is ``0.886``.

        """
        # Placeholder function, intended for evaluating inputs but currently does nothing
        pass

    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
        # Evaluate the expression numerically with a given precision `prec`
        return self.doit(deep=False, evaluate=False)._eval_evalf(prec)
    def doit(self, deep=True, evaluate=True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether `doit` should be recursively called. Default is `True`.
        evaluate : bool
            Whether the SymPy expression should be evaluated as it is
            constructed. If `False`, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable form for values of `fv_M` that correspond to a sensible
            operating range for a musculotendon. Default is `True`.
        **hints : dict
            Additional keyword argument pairs to be recursively passed to
            `doit`.

        """
        # Extract `fv_M` and remaining constants from `self.args`
        fv_M, *constants = self.args
        if deep:
            # Update `hints` dictionary with `evaluate` flag
            hints['evaluate'] = evaluate
            # Recursively call `doit` on `fv_M` and constants
            fv_M = fv_M.doit(deep=deep, **hints)
            # Apply `doit` recursively on each constant
            c0, c1, c2, c3 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            # If not deep, assign constants directly
            c0, c1, c2, c3 = constants

        if evaluate:
            # Evaluate the expression with `fv_M`, `c0`, `c1`, `c2`, `c3`
            return (sinh((fv_M - c3)/c0) - c2)/c1
        else:
            # Return unevaluated expression with `UnevaluatedExpr`
            return (sinh(UnevaluatedExpr(fv_M - c3)/c0) - c2)/c1

    def fdiff(self, argindex=1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at `1`.
            Default is `1`.

        """
        # Extract `fv_M` and constants from `self.args`
        fv_M, c0, c1, c2, c3 = self.args
        if argindex == 1:
            # Compute first derivative
            return cosh((fv_M - c3)/c0)/(c0*c1)
        elif argindex == 2:
            # Compute derivative with respect to the second argument
            return (c3 - fv_M)*cosh((fv_M - c3)/c0)/(c0**2*c1)
        elif argindex == 3:
            # Compute derivative with respect to the third argument
            return (c2 - sinh((fv_M - c3)/c0))/c1**2
        elif argindex == 4:
            # Compute derivative with respect to the fourth argument
            return -1/c1
        elif argindex == 5:
            # Compute derivative with respect to the fifth argument
            return -cosh((fv_M - c3)/c0)/(c0*c1)

        # Raise error if `argindex` is out of range
        raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """Inverse function.

        Parameters
        ==========

        argindex : int
            Value to start indexing the arguments at. Default is `1`.

        """
        # Return the inverse function FiberForceVelocityDeGroote2016
        return FiberForceVelocityDeGroote2016

    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """
        # Extract `fv_M` from `self.args`
        fv_M = self.args[0]
        # Print LaTeX representation of `fv_M`
        _fv_M = printer._print(fv_M)
        # Return LaTeX string representation
        return r'\left( \operatorname{fv}^M \right)^{-1} \left( %s \right)' % _fv_M
# 使用 dataclass 装饰器创建一个不可变的数据容器类，用于组合相关的特征曲线函数
@dataclass(frozen=True)
class CharacteristicCurveCollection:
    """Simple data container to group together related characteristic curves."""

    # 定义属性，每个属性都是一个特征曲线函数对象
    tendon_force_length: CharacteristicCurveFunction
    tendon_force_length_inverse: CharacteristicCurveFunction
    fiber_force_length_passive: CharacteristicCurveFunction
    fiber_force_length_passive_inverse: CharacteristicCurveFunction
    fiber_force_length_active: CharacteristicCurveFunction
    fiber_force_velocity: CharacteristicCurveFunction
    fiber_force_velocity_inverse: CharacteristicCurveFunction

    # 定义迭代器方法，支持对该类的实例进行迭代
    def __iter__(self):
        """Iterator support for ``CharacteristicCurveCollection``."""
        yield self.tendon_force_length
        yield self.tendon_force_length_inverse
        yield self.fiber_force_length_passive
        yield self.fiber_force_length_passive_inverse
        yield self.fiber_force_length_active
        yield self.fiber_force_velocity
        yield self.fiber_force_velocity_inverse
```