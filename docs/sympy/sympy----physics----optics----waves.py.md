# `D:\src\scipysrc\sympy\sympy\physics\optics\waves.py`

```
"""
This module has all the classes and functions related to waves in optics.

**Contains**

* TWave
"""

__all__ = ['TWave']

from sympy.core.basic import Basic                          # 导入基本符号运算类 Basic
from sympy.core.expr import Expr                            # 导入表达式类 Expr
from sympy.core.function import Derivative, Function        # 导入导数和函数类
from sympy.core.numbers import (Number, pi, I)              # 导入数值类，包括 pi 和虚数单位 I
from sympy.core.singleton import S                          # 导入单例类 S
from sympy.core.symbol import (Symbol, symbols)             # 导入符号和符号集合类
from sympy.core.sympify import _sympify, sympify           # 导入符号化函数
from sympy.functions.elementary.exponential import exp     # 导入指数函数 exp
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数 sqrt
from sympy.functions.elementary.trigonometric import (atan2, cos, sin)  # 导入三角函数 atan2, cos, sin
from sympy.physics.units import speed_of_light, meter, second  # 导入物理单位

c = speed_of_light.convert_to(meter/second)  # 计算光速并转换为米/秒单位


class TWave(Expr):
    """
    This is a simple transverse sine wave travelling in a one-dimensional space.
    Basic properties are required at the time of creation of the object,
    but they can be changed later with respective methods provided.

    Explanation
    ===========

    It is represented as :math:`A \times cos(k*x - \omega \times t + \phi )`,
    where :math:`A` is the amplitude, :math:`\omega` is the angular frequency,
    :math:`k` is the wavenumber (spatial frequency), :math:`x` is a spatial variable
    to represent the position on the dimension on which the wave propagates,
    and :math:`\phi` is the phase angle of the wave.


    Arguments
    =========

    amplitude : Sympifyable
        Amplitude of the wave.
    frequency : Sympifyable
        Frequency of the wave.
    phase : Sympifyable
        Phase angle of the wave.
    time_period : Sympifyable
        Time period of the wave.
    n : Sympifyable
        Refractive index of the medium.

    Raises
    =======

    ValueError : When neither frequency nor time period is provided
        or they are not consistent.
    TypeError : When anything other than TWave objects is added.


    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.optics import TWave
    >>> A1, phi1, A2, phi2, f = symbols('A1, phi1, A2, phi2, f')
    >>> w1 = TWave(A1, f, phi1)
    >>> w2 = TWave(A2, f, phi2)
    >>> w3 = w1 + w2  # Superposition of two waves
    >>> w3
    TWave(sqrt(A1**2 + 2*A1*A2*cos(phi1 - phi2) + A2**2), f,
        atan2(A1*sin(phi1) + A2*sin(phi2), A1*cos(phi1) + A2*cos(phi2)), 1/f, n)
    >>> w3.amplitude
    sqrt(A1**2 + 2*A1*A2*cos(phi1 - phi2) + A2**2)
    >>> w3.phase
    atan2(A1*sin(phi1) + A2*sin(phi2), A1*cos(phi1) + A2*cos(phi2))
    >>> w3.speed
    299792458*meter/(second*n)
    >>> w3.angular_velocity
    2*pi*f

    """
    def __new__(
            cls,
            amplitude,
            frequency=None,
            phase=S.Zero,
            time_period=None,
            n=Symbol('n')):
        # 如果指定了时间周期，将其转换为 SymPy 对象
        if time_period is not None:
            time_period = _sympify(time_period)
            # 根据时间周期计算频率
            _frequency = S.One/time_period
        # 如果指定了频率，将其转换为 SymPy 对象
        if frequency is not None:
            frequency = _sympify(frequency)
            # 根据频率计算时间周期
            _time_period = S.One/frequency
            # 检查频率和时间周期是否一致，若不一致则抛出异常
            if time_period is not None:
                if frequency != S.One/time_period:
                    raise ValueError("frequency and time_period should be consistent.")
        # 如果既没有指定频率也没有指定时间周期，抛出异常
        if frequency is None and time_period is None:
            raise ValueError("Either frequency or time period is needed.")
        # 如果未指定频率，则使用之前计算得到的频率
        if frequency is None:
            frequency = _frequency
        # 如果未指定时间周期，则使用之前计算得到的时间周期
        if time_period is None:
            time_period = _time_period

        # 将振幅、相位、序号转换为 SymPy 对象
        amplitude = _sympify(amplitude)
        phase = _sympify(phase)
        n = sympify(n)
        # 创建并返回 TWave 对象
        obj = Basic.__new__(cls, amplitude, frequency, phase, time_period, n)
        return obj

    @property
    def amplitude(self):
        """
        返回波的振幅。

        示例
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.amplitude
        A
        """
        return self.args[0]

    @property
    def frequency(self):
        """
        返回波的频率，
        单位为每秒钟的周期数。

        示例
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.frequency
        f
        """
        return self.args[1]

    @property
    def phase(self):
        """
        返回波的相位角，
        单位为弧度。

        示例
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.phase
        phi
        """
        return self.args[2]

    @property
    def time_period(self):
        """
        返回波的时间周期，
        单位为每个周期的秒数。

        示例
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.time_period
        1/f
        """
        return self.args[3]

    @property
    def n(self):
        """
        返回介质的折射率
        """
        return self.args[4]
    def wavelength(self):
        """
        Returns the wavelength (spatial period) of the wave,
        in meters per cycle.
        It depends on the medium of the wave.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.wavelength
        299792458*meter/(second*f*n)
        """
        # 计算并返回波的波长，单位为每个周期的米数
        return c/(self.frequency*self.n)


    @property
    def speed(self):
        """
        Returns the propagation speed of the wave,
        in meters per second.
        It is dependent on the propagation medium.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.speed
        299792458*meter/(second*n)
        """
        # 计算并返回波的传播速度，单位为米每秒
        return self.wavelength*self.frequency

    @property
    def angular_velocity(self):
        """
        Returns the angular velocity of the wave,
        in radians per second.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.angular_velocity
        2*pi*f
        """
        # 计算并返回波的角速度，单位为每秒弧度
        return 2*pi*self.frequency

    @property
    def wavenumber(self):
        """
        Returns the wavenumber of the wave,
        in radians per meter.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.wavenumber
        pi*second*f*n/(149896229*meter)
        """
        # 计算并返回波数，单位为每米弧度
        return 2*pi/self.wavelength

    def __str__(self):
        """String representation of a TWave."""
        # 返回 TWave 对象的字符串表示形式
        from sympy.printing import sstr
        return type(self).__name__ + sstr(self.args)

    __repr__ = __str__
    def __add__(self, other):
        """
        Addition of two waves will result in their superposition.
        The type of interference will depend on their phase angles.
        """
        # 检查参数是否为 TWave 类型
        if isinstance(other, TWave):
            # 检查两个波的频率和波长是否相同
            if self.frequency == other.frequency and self.wavelength == other.wavelength:
                # 计算叠加波的振幅和相位
                return TWave(sqrt(self.amplitude**2 + other.amplitude**2 + 2 *
                                  self.amplitude*other.amplitude*cos(
                                      self.phase - other.phase)),
                             self.frequency,
                             atan2(self.amplitude*sin(self.phase)
                             + other.amplitude*sin(other.phase),
                             self.amplitude*cos(self.phase)
                             + other.amplitude*cos(other.phase))
                             )
            else:
                # 抛出异常，不同频率的波无法进行干涉
                raise NotImplementedError("Interference of waves with different frequencies"
                    " has not been implemented.")
        else:
            # 抛出异常，不能将非 TWave 对象添加到波中
            raise TypeError(type(other).__name__ + " and TWave objects cannot be added.")

    def __mul__(self, other):
        """
        Multiplying a wave by a scalar rescales the amplitude of the wave.
        """
        # 将 other 转换为符号表达式
        other = sympify(other)
        # 检查 other 是否为数字类型
        if isinstance(other, Number):
            # 返回一个振幅按比例缩放的新的 TWave 对象
            return TWave(self.amplitude*other, *self.args[1:])
        else:
            # 抛出异常，不能将非数字对象与 TWave 对象相乘
            raise TypeError(type(other).__name__ + " and TWave objects cannot be multiplied.")

    def __sub__(self, other):
        # 调用 __add__ 方法，用 -1 倍的 other 替代实现减法
        return self.__add__(-1*other)

    def __neg__(self):
        # 调用 __mul__ 方法，用 -1 倍的 self 实现取负操作
        return self.__mul__(-1)

    def __radd__(self, other):
        # 调用 __add__ 方法，反向操作
        return self.__add__(other)

    def __rmul__(self, other):
        # 调用 __mul__ 方法，反向操作
        return self.__mul__(other)

    def __rsub__(self, other):
        # 调用 __neg__ 和 __radd__ 方法，反向操作
        return (-self).__radd__(other)

    def _eval_rewrite_as_sin(self, *args, **kwargs):
        # 返回正弦函数形式的波表达式
        return self.amplitude*sin(self.wavenumber*Symbol('x')
            - self.angular_velocity*Symbol('t') + self.phase + pi/2, evaluate=False)

    def _eval_rewrite_as_cos(self, *args, **kwargs):
        # 返回余弦函数形式的波表达式
        return self.amplitude*cos(self.wavenumber*Symbol('x')
            - self.angular_velocity*Symbol('t') + self.phase)

    def _eval_rewrite_as_pde(self, *args, **kwargs):
        # 返回偏微分方程的形式
        mu, epsilon, x, t = symbols('mu, epsilon, x, t')
        E = Function('E')
        return Derivative(E(x, t), x, 2) + mu*epsilon*Derivative(E(x, t), t, 2)

    def _eval_rewrite_as_exp(self, *args, **kwargs):
        # 返回指数函数形式的波表达式
        return self.amplitude*exp(I*(self.wavenumber*Symbol('x')
            - self.angular_velocity*Symbol('t') + self.phase))
```