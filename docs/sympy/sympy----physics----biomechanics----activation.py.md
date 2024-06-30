# `D:\src\scipysrc\sympy\sympy\physics\biomechanics\activation.py`

```
r"""Activation dynamics for musclotendon models.

Musculotendon models are able to produce active force when they are activated,
which is when a chemical process has taken place within the muscle fibers
causing them to voluntarily contract. Biologically this chemical process (the
diffusion of :math:`\textrm{Ca}^{2+}` ions) is not the input in the system,
electrical signals from the nervous system are. These are termed excitations.
Activation dynamics, which relates the normalized excitation level to the
normalized activation level, can be modeled by the models present in this
module.

"""

from abc import ABC, abstractmethod
from functools import cached_property

from sympy.core.symbol import Symbol
from sympy.core.numbers import Float, Integer, Rational
from sympy.functions.elementary.hyperbolic import tanh
from sympy.matrices.dense import MutableDenseMatrix as Matrix, zeros
from sympy.physics.biomechanics._mixin import _NamedMixin
from sympy.physics.mechanics import dynamicsymbols


__all__ = [
    'ActivationBase',
    'FirstOrderActivationDeGroote2016',
    'ZerothOrderActivation',
]


class ActivationBase(ABC, _NamedMixin):
    """Abstract base class for all activation dynamics classes to inherit from.

    Notes
    =====

    Instances of this class cannot be directly instantiated by users. However,
    it can be used to created custom activation dynamics types through
    subclassing.

    """

    def __init__(self, name):
        """Initializer for ``ActivationBase``."""
        self.name = str(name)

        # Symbols
        self._e = dynamicsymbols(f"e_{name}")  # Define dynamic symbol for excitation
        self._a = dynamicsymbols(f"a_{name}")  # Define dynamic symbol for activation

    @classmethod
    @abstractmethod
    def with_defaults(cls, name):
        """Alternate constructor that provides recommended defaults for
        constants."""
        pass

    @property
    def excitation(self):
        """Dynamic symbol representing excitation.

        Explanation
        ===========

        The alias ``e`` can also be used to access the same attribute.

        """
        return self._e

    @property
    def e(self):
        """Dynamic symbol representing excitation.

        Explanation
        ===========

        The alias ``excitation`` can also be used to access the same attribute.

        """
        return self._e

    @property
    def activation(self):
        """Dynamic symbol representing activation.

        Explanation
        ===========

        The alias ``a`` can also be used to access the same attribute.

        """
        return self._a

    @property
    def a(self):
        """Dynamic symbol representing activation.

        Explanation
        ===========

        The alias ``activation`` can also be used to access the same attribute.

        """
        return self._a

    @property
    @abstractmethod
    def order(self):
        """Order of the (differential) equation governing activation."""
        pass

    @property
    @abstractmethod
    def state_vars(self):
        """
        Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        The alias ``x`` can also be used to access the same attribute.
        """
        pass

    @property
    @abstractmethod
    def x(self):
        """
        Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        The alias ``state_vars`` can also be used to access the same attribute.
        """
        pass

    @property
    @abstractmethod
    def input_vars(self):
        """
        Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========

        The alias ``r`` can also be used to access the same attribute.
        """
        pass

    @property
    @abstractmethod
    def r(self):
        """
        Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========

        The alias ``input_vars`` can also be used to access the same attribute.
        """
        pass

    @property
    @abstractmethod
    def constants(self):
        """
        Ordered column matrix of non-time varying symbols present in ``M``
        and ``F``.

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        Explanation
        ===========

        The alias ``p`` can also be used to access the same attribute.
        """
        pass

    @property
    @abstractmethod
    def p(self):
        """
        Ordered column matrix of non-time varying symbols present in ``M``
        and ``F``.

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        Explanation
        ===========

        The alias ``constants`` can also be used to access the same attribute.
        """
        pass

    @property
    @abstractmethod
    def M(self):
        """
        Ordered square matrix of coefficients on the LHS of ``M x' = F``.

        Explanation
        ===========

        The square matrix that forms part of the LHS of the linear system of
        ordinary differential equations governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.
        """
        pass
    @property
    @abstractmethod
    def F(self):
        """Ordered column matrix of equations on the RHS of ``M x' = F``.
    
        Explanation
        ===========
    
        The column matrix that forms the RHS of the linear system of ordinary
        differential equations governing the activation dynamics:
    
        ``M(x, r, t, p) x' = F(x, r, t, p)``.
    
        """
        pass
    
    
    
    @abstractmethod
    def rhs(self):
        """
        
        Explanation
        ===========
        
        The solution to the linear system of ordinary differential equations
        governing the activation dynamics:
        
        ``M(x, r, t, p) x' = F(x, r, t, p)``.
        
        """
        pass
    
    
    
    def __eq__(self, other):
        """Equality check for activation dynamics."""
        if type(self) != type(other):
            return False
        if self.name != other.name:
            return False
        return True
    
    
    
    def __repr__(self):
        """Default representation of activation dynamics."""
        return f'{self.__class__.__name__}({self.name!r})'
class ZerothOrderActivation(ActivationBase):
    """Simple zeroth-order activation dynamics mapping excitation to
    activation.

    Explanation
    ===========

    Zeroth-order activation dynamics are useful in instances where you want to
    reduce the complexity of your musculotendon dynamics as they simply map
    excitation to activation. As a result, no additional state equations are
    introduced to your system. They also remove a potential source of delay
    between the input and dynamics of your system as no (ordinary) differential
    equations are involved.

    """

    def __init__(self, name):
        """Initializer for ``ZerothOrderActivation``.

        Parameters
        ==========

        name : str
            The name identifier associated with the instance. Must be a string
            of length at least 1.

        """
        super().__init__(name)

        # Zeroth-order activation dynamics has activation equal excitation so
        # overwrite the symbol for activation with the excitation symbol.
        self._a = self._e

    @classmethod
    def with_defaults(cls, name):
        """Alternate constructor that provides recommended defaults for
        constants.

        Explanation
        ===========

        As this concrete class doesn't implement any constants associated with
        its dynamics, this ``classmethod`` simply creates a standard instance
        of ``ZerothOrderActivation``. An implementation is provided to ensure
        a consistent interface between all ``ActivationBase`` concrete classes.

        """
        return cls(name)

    @property
    def order(self):
        """Order of the (differential) equation governing activation."""
        return 0

    @property
    def state_vars(self):
        """Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        As zeroth-order activation dynamics simply maps excitation to
        activation, this class has no associated state variables and so this
        property returns an empty column ``Matrix`` with shape (0, 1).

        The alias ``x`` can also be used to access the same attribute.

        """
        return zeros(0, 1)

    @property
    def x(self):
        """Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        As zeroth-order activation dynamics simply maps excitation to
        activation, this class has no associated state variables and so this
        property returns an empty column ``Matrix`` with shape (0, 1).

        The alias ``state_vars`` can also be used to access the same attribute.

        """
        return zeros(0, 1)

    @property
    def _e(self):
        """Symbol representing excitation."""
        return self.excitation
    def input_vars(self):
        """
        Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========
        Excitation is the only input in zeroth-order activation dynamics and so
        this property returns a column Matrix with one entry, 'e', and
        shape (1, 1).

        The alias 'r' can also be used to access the same attribute.
        """
        return Matrix([self._e])

    @property
    def r(self):
        """
        Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========
        Excitation is the only input in zeroth-order activation dynamics and so
        this property returns a column Matrix with one entry, 'e', and
        shape (1, 1).

        The alias 'input_vars' can also be used to access the same attribute.
        """
        return Matrix([self._e])

    @property
    def constants(self):
        """
        Ordered column matrix of non-time varying symbols present in 'M'
        and 'F'.

        Only symbolic constants are returned. If a numeric type (e.g. Float)
        has been used instead of Symbol for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        Explanation
        ===========
        As zeroth-order activation dynamics simply maps excitation to
        activation, this class has no associated constants and so this property
        return an empty column Matrix with shape (0, 1).

        The alias 'p' can also be used to access the same attribute.
        """
        return zeros(0, 1)

    @property
    def p(self):
        """
        Ordered column matrix of non-time varying symbols present in 'M'
        and 'F'.

        Only symbolic constants are returned. If a numeric type (e.g. Float)
        has been used instead of Symbol for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        Explanation
        ===========
        As zeroth-order activation dynamics simply maps excitation to
        activation, this class has no associated constants and so this property
        return an empty column Matrix with shape (0, 1).

        The alias 'constants' can also be used to access the same attribute.
        """
        return zeros(0, 1)
    def M(self):
        """
        Ordered square matrix of coefficients on the LHS of ``M x' = F``.

        Explanation
        ===========

        The square matrix that forms part of the LHS of the linear system of
        ordinary differential equations governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        As zeroth-order activation dynamics have no state variables, this
        linear system has dimension 0 and therefore ``M`` is an empty square
        ``Matrix`` with shape (0, 0).

        """
        # 返回一个空的 0x0 的方阵 Matrix，用于表示线性系统的左侧系数
        return Matrix([])

    @property
    def F(self):
        """
        Ordered column matrix of equations on the RHS of ``M x' = F``.

        Explanation
        ===========

        The column matrix that forms the RHS of the linear system of ordinary
        differential equations governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        As zeroth-order activation dynamics have no state variables, this
        linear system has dimension 0 and therefore ``F`` is an empty column
        ``Matrix`` with shape (0, 1).

        """
        # 返回一个空的 0x1 的列向量 Matrix，用于表示线性系统的右侧方程
        return zeros(0, 1)

    def rhs(self):
        """
        Ordered column matrix of equations for the solution of ``M x' = F``.

        Explanation
        ===========

        The solution to the linear system of ordinary differential equations
        governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        As zeroth-order activation dynamics have no state variables, this
        linear has dimension 0 and therefore this method returns an empty
        column ``Matrix`` with shape (0, 1).

        """
        # 返回一个空的 0x1 的列向量 Matrix，用于表示线性系统的解
        return zeros(0, 1)
# 定义一个基于 De Groote et al., 2016 的一阶激活动力学模型类 FirstOrderActivationDeGroote2016，继承自 ActivationBase
class FirstOrderActivationDeGroote2016(ActivationBase):
    r"""First-order activation dynamics based on De Groote et al., 2016 [1]_.

    Explanation
    ===========

    Gives the first-order activation dynamics equation for the rate of change
    of activation with respect to time as a function of excitation and
    activation.

    The function is defined by the equation:

    .. math::

        \frac{da}{dt} = \left(\frac{\frac{1}{2} + a0}{\tau_a \left(\frac{1}{2}
            + \frac{3a}{2}\right)} + \frac{\left(\frac{1}{2}
            + \frac{3a}{2}\right) \left(\frac{1}{2} - a0\right)}{\tau_d}\right)
            \left(e - a\right)

    where

    .. math::

        a0 = \frac{\tanh{\left(b \left(e - a\right) \right)}}{2}

    with constant values of :math:`tau_a = 0.015`, :math:`tau_d = 0.060`, and
    :math:`b = 10`.

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    # 初始化方法，用于设置激活时间常数，去激活时间常数和平滑率等参数
    def __init__(self,
        name,
        activation_time_constant=None,
        deactivation_time_constant=None,
        smoothing_rate=None,
    ):
        """Initializer for ``FirstOrderActivationDeGroote2016``.

        Parameters
        ==========
        activation time constant : Symbol | Number | None
            The value of the activation time constant governing the delay
            between excitation and activation when excitation exceeds
            activation.
        deactivation time constant : Symbol | Number | None
            The value of the deactivation time constant governing the delay
            between excitation and activation when activation exceeds
            excitation.
        smoothing_rate : Symbol | Number | None
            The slope of the hyperbolic tangent function used to smooth between
            the switching of the equations where excitation exceed activation
            and where activation exceed excitation. The recommended value to
            use is ``10``, but values between ``0.1`` and ``100`` can be used.

        """
        # 调用父类的初始化方法，设置实例的名称
        super().__init__(name)

        # 实例变量，用于存储传入的激活时间常数，去激活时间常数和平滑率参数
        self.activation_time_constant = activation_time_constant
        self.deactivation_time_constant = deactivation_time_constant
        self.smoothing_rate = smoothing_rate

    @classmethod
    def with_defaults(cls, name):
        r"""Alternate constructor that will use the published constants.

        Explanation
        ===========

        Returns an instance of ``FirstOrderActivationDeGroote2016`` using the
        three constant values specified in the original publication.

        These have the values:

        :math:`tau_a = 0.015`
        :math:`tau_d = 0.060`
        :math:`b = 10`

        """
        # 创建浮点数对象 tau_a，设定其值为 0.015
        tau_a = Float('0.015')
        # 创建浮点数对象 tau_d，设定其值为 0.060
        tau_d = Float('0.060')
        # 创建浮点数对象 b，设定其值为 10.0
        b = Float('10.0')
        # 使用给定的参数创建并返回 FirstOrderActivationDeGroote2016 实例
        return cls(name, tau_a, tau_d, b)

    @property
    def activation_time_constant(self):
        """Delay constant for activation.

        Explanation
        ===========

        The alias ``tau_a`` can also be used to access the same attribute.

        """
        # 返回对象的 activation_time_constant 属性值
        return self._tau_a

    @activation_time_constant.setter
    def activation_time_constant(self, tau_a):
        # 如果对象已有 _tau_a 属性，则抛出异常，因为该属性是不可变的
        if hasattr(self, '_tau_a'):
            msg = (
                f'Can\'t set attribute `activation_time_constant` to '
                f'{repr(tau_a)} as it is immutable and already has value '
                f'{self._tau_a}.'
            )
            raise AttributeError(msg)
        # 否则，设置对象的 _tau_a 属性为 Symbol 对象，如果 tau_a 为 None 则设为默认值
        self._tau_a = Symbol(f'tau_a_{self.name}') if tau_a is None else tau_a

    @property
    def tau_a(self):
        """Delay constant for activation.

        Explanation
        ===========

        The alias ``activation_time_constant`` can also be used to access the
        same attribute.

        """
        # 返回对象的 tau_a 属性值，即 _tau_a
        return self._tau_a

    @property
    def deactivation_time_constant(self):
        """Delay constant for deactivation.

        Explanation
        ===========

        The alias ``tau_d`` can also be used to access the same attribute.

        """
        # 返回对象的 deactivation_time_constant 属性值
        return self._tau_d

    @deactivation_time_constant.setter
    def deactivation_time_constant(self, tau_d):
        # 如果对象已有 _tau_d 属性，则抛出异常，因为该属性是不可变的
        if hasattr(self, '_tau_d'):
            msg = (
                f'Can\'t set attribute `deactivation_time_constant` to '
                f'{repr(tau_d)} as it is immutable and already has value '
                f'{self._tau_d}.'
            )
            raise AttributeError(msg)
        # 否则，设置对象的 _tau_d 属性为 Symbol 对象，如果 tau_d 为 None 则设为默认值
        self._tau_d = Symbol(f'tau_d_{self.name}') if tau_d is None else tau_d

    @property
    def tau_d(self):
        """Delay constant for deactivation.

        Explanation
        ===========

        The alias ``deactivation_time_constant`` can also be used to access the
        same attribute.

        """
        # 返回对象的 tau_d 属性值，即 _tau_d
        return self._tau_d

    @property
    def smoothing_rate(self):
        """Smoothing constant for the hyperbolic tangent term.

        Explanation
        ===========

        The alias ``b`` can also be used to access the same attribute.

        """
        # 返回对象的 smoothing_rate 属性值，即 _b
        return self._b

    @smoothing_rate.setter
    def smoothing_rate(self, b):
        # 检查对象是否具有属性 '_b'
        if hasattr(self, '_b'):
            # 如果已经存在属性 '_b'，则抛出属性错误，指示该属性不可变且已经有值
            msg = (
                f'Can\'t set attribute `smoothing_rate` to {b!r} as it is '
                f'immutable and already has value {self._b!r}.'
            )
            raise AttributeError(msg)
        # 如果不存在属性 '_b'，则设置 self._b 为新的 Symbol 对象或者给定的 b 值
        self._b = Symbol(f'b_{self.name}') if b is None else b

    @property
    def b(self):
        """Smoothing constant for the hyperbolic tangent term.

        Explanation
        ===========

        The alias ``smoothing_rate`` can also be used to access the same
        attribute.

        """
        # 返回属性 '_b'，作为平滑常数的属性，可以使用别名 'smoothing_rate' 访问相同的属性
        return self._b

    @property
    def order(self):
        """Order of the (differential) equation governing activation."""
        # 返回整数值 1，表示激活方程的阶数
        return 1

    @property
    def state_vars(self):
        """Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        The alias ``x`` can also be used to access the same attribute.

        """
        # 返回一个包含 self._a 的有序列向量，表示时间函数的状态变量
        return Matrix([self._a])

    @property
    def x(self):
        """Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        The alias ``state_vars`` can also be used to access the same attribute.

        """
        # 返回一个包含 self._a 的有序列向量，表示时间函数的状态变量，可以使用别名 'state_vars' 访问相同的属性
        return Matrix([self._a])

    @property
    def input_vars(self):
        """Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========

        The alias ``r`` can also be used to access the same attribute.

        """
        # 返回一个包含 self._e 的有序列向量，表示时间函数的输入变量
        return Matrix([self._e])

    @property
    def r(self):
        """Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========

        The alias ``input_vars`` can also be used to access the same attribute.

        """
        # 返回一个包含 self._e 的有序列向量，表示时间函数的输入变量，可以使用别名 'input_vars' 访问相同的属性
        return Matrix([self._e])

    @property
    def constants(self):
        """Ordered column matrix of non-time varying symbols present in ``M``
        and ``F``.

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        Explanation
        ===========

        The alias ``p`` can also be used to access the same attribute.

        """
        # 返回一个包含非时间变化符号的有序列向量，这些符号存在于 'M' 和 'F' 中
        constants = [self._tau_a, self._tau_d, self._b]
        symbolic_constants = [c for c in constants if not c.is_number]
        return Matrix(symbolic_constants) if symbolic_constants else zeros(0, 1)
    def p(self):
        """
        Ordered column matrix of non-time varying symbols present in ``M``
        and ``F``.

        Explanation
        ===========

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        The alias ``constants`` can also be used to access the same attribute.
        """
        constants = [self._tau_a, self._tau_d, self._b]
        # Filter out numeric constants to get symbolic constants
        symbolic_constants = [c for c in constants if not c.is_number]
        # Return a matrix of symbolic constants or an empty matrix if none exist
        return Matrix(symbolic_constants) if symbolic_constants else zeros(0, 1)

    @property
    def M(self):
        """
        Ordered square matrix of coefficients on the LHS of ``M x' = F``.

        Explanation
        ===========

        The square matrix that forms part of the LHS of the linear system of
        ordinary differential equations governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.
        """
        return Matrix([Integer(1)])

    @property
    def F(self):
        """
        Ordered column matrix of equations on the RHS of ``M x' = F``.

        Explanation
        ===========

        The column matrix that forms the RHS of the linear system of ordinary
        differential equations governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.
        """
        return Matrix([self._da_eqn])

    def rhs(self):
        """
        Ordered column matrix of equations for the solution of ``M x' = F``.

        Explanation
        ===========

        The solution to the linear system of ordinary differential equations
        governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.
        """
        return Matrix([self._da_eqn])

    @cached_property
    def _da_eqn(self):
        """
        Calculate activation dynamics equation.

        Explanation
        ===========

        Computes the equation governing the activation dynamics based on
        certain parameters and attributes.
        """
        HALF = Rational(1, 2)
        # Calculate components of the activation dynamics equation
        a0 = HALF * tanh(self._b * (self._e - self._a))
        a1 = (HALF + Rational(3, 2) * self._a)
        a2 = (HALF + a0) / (self._tau_a * a1)
        a3 = a1 * (HALF - a0) / self._tau_d
        activation_dynamics_equation = (a2 + a3) * (self._e - self._a)
        return activation_dynamics_equation

    def __eq__(self, other):
        """
        Equality check for ``FirstOrderActivationDeGroote2016``.

        Explanation
        ===========

        Compares two instances of ``FirstOrderActivationDeGroote2016`` for equality
        based on their attributes.
        """
        if type(self) != type(other):
            return False
        # Compare relevant attributes for equality
        self_attrs = (self.name, self.tau_a, self.tau_d, self.b)
        other_attrs = (other.name, other.tau_a, other.tau_d, other.b)
        return self_attrs == other_attrs
    def __repr__(self):
        """
        对象的字符串表示形式，返回一个描述对象的字符串。
        """
        return (
            f'{self.__class__.__name__}({self.name!r}, '  # 返回类名和对象名称的格式化字符串
            f'activation_time_constant={self.tau_a!r}, '  # 包含激活时间常数的格式化字符串
            f'deactivation_time_constant={self.tau_d!r}, '  # 包含去激活时间常数的格式化字符串
            f'smoothing_rate={self.b!r})'  # 包含平滑率的格式化字符串
        )
```