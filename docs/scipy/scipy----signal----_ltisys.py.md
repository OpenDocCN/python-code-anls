# `D:\src\scipysrc\scipy\scipy\signal\_ltisys.py`

```
"""
ltisys -- a collection of classes and functions for modeling linear
time invariant systems.
"""
#
# Author: Travis Oliphant 2001
#
# Feb 2010: Warren Weckesser
#   Rewrote lsim2 and added impulse2.
# Apr 2011: Jeffrey Armstrong <jeff@approximatrix.com>
#   Added dlsim, dstep, dimpulse, cont2discrete
# Aug 2013: Juan Luis Cano
#   Rewrote abcd_normalize.
# Jan 2015: Irvin Probst irvin DOT probst AT ensta-bretagne DOT fr
#   Added pole placement
# Mar 2015: Clancy Rowley
#   Rewrote lsim
# May 2015: Felix Berkenkamp
#   Split lti class into subclasses
#   Merged discrete systems and added dlti

import warnings

# np.linalg.qr fails on some tests with LinAlgError: zgeqrf returns -7
# use scipy's qr until this is solved

from scipy.linalg import qr as s_qr
from scipy import linalg
from scipy.interpolate import make_interp_spline
from ._filter_design import (tf2zpk, zpk2tf, normalize, freqs, freqz, freqs_zpk,
                            freqz_zpk)
from ._lti_conversion import (tf2ss, abcd_normalize, ss2tf, zpk2ss, ss2zpk,
                             cont2discrete, _atleast_2d_or_none)

import numpy as np
from numpy import (real, atleast_1d, squeeze, asarray, zeros,
                   dot, transpose, ones, linspace)
import copy

__all__ = ['lti', 'dlti', 'TransferFunction', 'ZerosPolesGain', 'StateSpace',
           'lsim', 'impulse', 'step', 'bode',
           'freqresp', 'place_poles', 'dlsim', 'dstep', 'dimpulse',
           'dfreqresp', 'dbode']


class LinearTimeInvariant:
    def __new__(cls, *system, **kwargs):
        """Create a new object, don't allow direct instances."""
        if cls is LinearTimeInvariant:
            raise NotImplementedError('The LinearTimeInvariant class is not '
                                      'meant to be used directly, use `lti` '
                                      'or `dlti` instead.')
        return super().__new__(cls)

    def __init__(self):
        """
        Initialize the `lti` baseclass.

        The heavy lifting is done by the subclasses.
        """
        super().__init__()

        self.inputs = None  # 初始化输入为 None
        self.outputs = None  # 初始化输出为 None
        self._dt = None  # 初始化采样时间为 None

    @property
    def dt(self):
        """Return the sampling time of the system, `None` for `lti` systems."""
        return self._dt  # 返回系统的采样时间，对于 lti 系统返回 None

    @property
    def _dt_dict(self):
        if self.dt is None:
            return {}
        else:
            return {'dt': self.dt}  # 如果系统有采样时间，返回包含采样时间的字典

    @property
    def zeros(self):
        """Zeros of the system."""
        return self.to_zpk().zeros  # 返回系统的零点

    @property
    def poles(self):
        """Poles of the system."""
        return self.to_zpk().poles  # 返回系统的极点
    # 将对象转换为 `StateSpace` 系统，不进行复制操作
    def _as_ss(self):
        """Convert to `StateSpace` system, without copying.

        Returns
        -------
        sys: StateSpace
            The `StateSpace` system. If the class is already an instance of
            `StateSpace` then this instance is returned.
        """
        # 如果对象已经是 `StateSpace` 的实例，则直接返回自身
        if isinstance(self, StateSpace):
            return self
        else:
            # 否则调用对象的 `to_ss` 方法进行转换
            return self.to_ss()

    # 将对象转换为 `ZerosPolesGain` 系统，不进行复制操作
    def _as_zpk(self):
        """Convert to `ZerosPolesGain` system, without copying.

        Returns
        -------
        sys: ZerosPolesGain
            The `ZerosPolesGain` system. If the class is already an instance of
            `ZerosPolesGain` then this instance is returned.
        """
        # 如果对象已经是 `ZerosPolesGain` 的实例，则直接返回自身
        if isinstance(self, ZerosPolesGain):
            return self
        else:
            # 否则调用对象的 `to_zpk` 方法进行转换
            return self.to_zpk()

    # 将对象转换为 `TransferFunction` 系统，不进行复制操作
    def _as_tf(self):
        """Convert to `TransferFunction` system, without copying.

        Returns
        -------
        sys: ZerosPolesGain
            The `TransferFunction` system. If the class is already an instance of
            `TransferFunction` then this instance is returned.
        """
        # 如果对象已经是 `TransferFunction` 的实例，则直接返回自身
        if isinstance(self, TransferFunction):
            return self
        else:
            # 否则调用对象的 `to_tf` 方法进行转换
            return self.to_tf()
class lti(LinearTimeInvariant):
    r"""
    Continuous-time linear time invariant system base class.

    Parameters
    ----------
    *system : arguments
        The `lti` class can be instantiated with either 2, 3 or 4 arguments.
        The following gives the number of arguments and the corresponding
        continuous-time subclass that is created:

            * 2: `TransferFunction`:  (numerator, denominator)
            * 3: `ZerosPolesGain`: (zeros, poles, gain)
            * 4: `StateSpace`:  (A, B, C, D)

        Each argument can be an array or a sequence.

    See Also
    --------
    ZerosPolesGain, StateSpace, TransferFunction, dlti

    Notes
    -----
    `lti` instances do not exist directly. Instead, `lti` creates an instance
    of one of its subclasses: `StateSpace`, `TransferFunction` or
    `ZerosPolesGain`.

    If (numerator, denominator) is passed in for ``*system``, coefficients for
    both the numerator and denominator should be specified in descending
    exponent order (e.g., ``s^2 + 3s + 5`` would be represented as ``[1, 3,
    5]``).

    Changing the value of properties that are not directly part of the current
    system representation (such as the `zeros` of a `StateSpace` system) is
    very inefficient and may lead to numerical inaccuracies. It is better to
    convert to the specific system representation first. For example, call
    ``sys = sys.to_zpk()`` before accessing/changing the zeros, poles or gain.

    Examples
    --------
    >>> from scipy import signal

    >>> signal.lti(1, 2, 3, 4)
    StateSpaceContinuous(
    array([[1]]),
    array([[2]]),
    array([[3]]),
    array([[4]]),
    dt: None
    )

    Construct the transfer function
    :math:`H(s) = \frac{5(s - 1)(s - 2)}{(s - 3)(s - 4)}`:

    >>> signal.lti([1, 2], [3, 4], 5)
    ZerosPolesGainContinuous(
    array([1, 2]),
    array([3, 4]),
    5,
    dt: None
    )

    Construct the transfer function :math:`H(s) = \frac{3s + 4}{1s + 2}`:

    >>> signal.lti([3, 4], [1, 2])
    TransferFunctionContinuous(
    array([3., 4.]),
    array([1., 2.]),
    dt: None
    )

    """
    def __new__(cls, *system):
        """Create an instance of the appropriate subclass."""
        # 检查是否直接调用了 lti 类，根据传入参数的数量选择合适的子类实例化
        if cls is lti:
            N = len(system)
            if N == 2:
                # 如果有两个参数，创建 TransferFunctionContinuous 的实例
                return TransferFunctionContinuous.__new__(
                    TransferFunctionContinuous, *system)
            elif N == 3:
                # 如果有三个参数，创建 ZerosPolesGainContinuous 的实例
                return ZerosPolesGainContinuous.__new__(
                    ZerosPolesGainContinuous, *system)
            elif N == 4:
                # 如果有四个参数，创建 StateSpaceContinuous 的实例
                return StateSpaceContinuous.__new__(StateSpaceContinuous,
                                                    *system)
            else:
                # 如果参数数量不符合要求，抛出 ValueError 异常
                raise ValueError("`system` needs to be an instance of `lti` "
                                 "or have 2, 3 or 4 arguments.")
        # 如果 __new__ 是从子类调用的，让子类调用自己的函数
        return super().__new__(cls)
    def __init__(self, *system):
        """
        Initialize the `lti` baseclass.

        The heavy lifting is done by the subclasses.
        """
        # 调用父类的初始化方法，传递系统参数
        super().__init__(*system)

    def impulse(self, X0=None, T=None, N=None):
        """
        Return the impulse response of a continuous-time system.
        See `impulse` for details.
        """
        # 调用 `impulse` 函数，返回连续时间系统的冲激响应
        return impulse(self, X0=X0, T=T, N=N)

    def step(self, X0=None, T=None, N=None):
        """
        Return the step response of a continuous-time system.
        See `step` for details.
        """
        # 调用 `step` 函数，返回连续时间系统的阶跃响应
        return step(self, X0=X0, T=T, N=N)

    def output(self, U, T, X0=None):
        """
        Return the response of a continuous-time system to input `U`.
        See `lsim` for details.
        """
        # 调用 `lsim` 函数，返回连续时间系统对输入 `U` 的响应
        return lsim(self, U, T, X0=X0)

    def bode(self, w=None, n=100):
        """
        Calculate Bode magnitude and phase data of a continuous-time system.

        Returns a 3-tuple containing arrays of frequencies [rad/s], magnitude
        [dB] and phase [deg]. See `bode` for details.
        
        Examples
        --------
        >>> from scipy import signal
        >>> import matplotlib.pyplot as plt

        >>> sys = signal.TransferFunction([1], [1, 1])
        >>> w, mag, phase = sys.bode()

        >>> plt.figure()
        >>> plt.semilogx(w, mag)    # Bode magnitude plot
        >>> plt.figure()
        >>> plt.semilogx(w, phase)  # Bode phase plot
        >>> plt.show()

        """
        # 调用 `bode` 函数，计算连续时间系统的波德幅度和相位数据
        return bode(self, w=w, n=n)

    def freqresp(self, w=None, n=10000):
        """
        Calculate the frequency response of a continuous-time system.

        Returns a 2-tuple containing arrays of frequencies [rad/s] and
        complex magnitude.
        See `freqresp` for details.
        """
        # 调用 `freqresp` 函数，计算连续时间系统的频率响应
        return freqresp(self, w=w, n=n)

    def to_discrete(self, dt, method='zoh', alpha=None):
        """Return a discretized version of the current system.

        Parameters: See `cont2discrete` for details.

        Returns
        -------
        sys: instance of `dlti`
        """
        # 抛出未实现异常，提示当前系统类不支持 `to_discrete` 方法
        raise NotImplementedError('to_discrete is not implemented for this '
                                  'system class.')
class dlti(LinearTimeInvariant):
    r"""
    Discrete-time linear time invariant system base class.
    
    Parameters
    ----------
    *system: arguments
        The `dlti` class can be instantiated with either 2, 3 or 4 arguments.
        The following gives the number of arguments and the corresponding
        discrete-time subclass that is created:
        
            * 2: `TransferFunction`:  (numerator, denominator)
            * 3: `ZerosPolesGain`: (zeros, poles, gain)
            * 4: `StateSpace`:  (A, B, C, D)
        
        Each argument can be an array or a sequence.
    dt: float, optional
        Sampling time [s] of the discrete-time systems. Defaults to ``True``
        (unspecified sampling time). Must be specified as a keyword argument,
        for example, ``dt=0.1``.
        
    See Also
    --------
    ZerosPolesGain, StateSpace, TransferFunction, lti
    
    Notes
    -----
    `dlti` instances do not exist directly. Instead, `dlti` creates an instance
    of one of its subclasses: `StateSpace`, `TransferFunction` or
    `ZerosPolesGain`.
    
    Changing the value of properties that are not directly part of the current
    system representation (such as the `zeros` of a `StateSpace` system) is
    very inefficient and may lead to numerical inaccuracies.  It is better to
    convert to the specific system representation first. For example, call
    ``sys = sys.to_zpk()`` before accessing/changing the zeros, poles or gain.
    
    If (numerator, denominator) is passed in for ``*system``, coefficients for
    both the numerator and denominator should be specified in descending
    exponent order (e.g., ``z^2 + 3z + 5`` would be represented as ``[1, 3,
    5]``).
    
    .. versionadded:: 0.18.0
    
    Examples
    --------
    >>> from scipy import signal
    
    >>> signal.dlti(1, 2, 3, 4)
    StateSpaceDiscrete(
    array([[1]]),
    array([[2]]),
    array([[3]]),
    array([[4]]),
    dt: True
    )
    
    >>> signal.dlti(1, 2, 3, 4, dt=0.1)
    StateSpaceDiscrete(
    array([[1]]),
    array([[2]]),
    array([[3]]),
    array([[4]]),
    dt: 0.1
    )
    
    Construct the transfer function
    :math:`H(z) = \frac{5(z - 1)(z - 2)}{(z - 3)(z - 4)}` with a sampling time
    of 0.1 seconds:
    
    >>> signal.dlti([1, 2], [3, 4], 5, dt=0.1)
    ZerosPolesGainDiscrete(
    array([1, 2]),
    array([3, 4]),
    5,
    dt: 0.1
    )
    
    Construct the transfer function :math:`H(z) = \frac{3z + 4}{1z + 2}` with
    a sampling time of 0.1 seconds:
    
    >>> signal.dlti([3, 4], [1, 2], dt=0.1)
    TransferFunctionDiscrete(
    array([3., 4.]),
    array([1., 2.]),
    dt: 0.1
    )
    
    """
    def __new__(cls, *system, **kwargs):
        """
        Create an instance of the appropriate subclass.

        Depending on the number of arguments in `system`, instantiate
        the corresponding subclass of `dlti` (either TransferFunctionDiscrete,
        ZerosPolesGainDiscrete, or StateSpaceDiscrete).
        """
        if cls is dlti:
            # Determine the number of arguments passed to `system`
            N = len(system)
            if N == 2:
                # If there are 2 arguments, instantiate TransferFunctionDiscrete
                return TransferFunctionDiscrete.__new__(
                    TransferFunctionDiscrete, *system, **kwargs)
            elif N == 3:
                # If there are 3 arguments, instantiate ZerosPolesGainDiscrete
                return ZerosPolesGainDiscrete.__new__(ZerosPolesGainDiscrete,
                                                      *system, **kwargs)
            elif N == 4:
                # If there are 4 arguments, instantiate StateSpaceDiscrete
                return StateSpaceDiscrete.__new__(StateSpaceDiscrete, *system,
                                                  **kwargs)
            else:
                # Raise an error if `system` does not match expected cases
                raise ValueError("`system` needs to be an instance of `dlti` "
                                 "or have 2, 3 or 4 arguments.")
        # If __new__ was called from a subclass, let it call its own functions
        return super().__new__(cls)

    def __init__(self, *system, **kwargs):
        """
        Initialize the `lti` baseclass.

        Sets up the sampling time (`dt`) and initializes the superclass (`super()`).
        """
        dt = kwargs.pop('dt', True)
        super().__init__(*system, **kwargs)

        # Set the sampling time (`dt`) for the system
        self.dt = dt

    @property
    def dt(self):
        """
        Return the sampling time (`dt`) of the system.

        Getter method for accessing the sampling time attribute (`_dt`).
        """
        return self._dt

    @dt.setter
    def dt(self, dt):
        """
        Set the sampling time (`dt`) of the system.

        Setter method for setting the sampling time attribute (`_dt`).
        """
        self._dt = dt

    def impulse(self, x0=None, t=None, n=None):
        """
        Return the impulse response of the discrete-time `dlti` system.

        Calls the `dimpulse` function to compute and return the impulse response.
        """
        return dimpulse(self, x0=x0, t=t, n=n)

    def step(self, x0=None, t=None, n=None):
        """
        Return the step response of the discrete-time `dlti` system.

        Calls the `dstep` function to compute and return the step response.
        """
        return dstep(self, x0=x0, t=t, n=n)

    def output(self, u, t, x0=None):
        """
        Return the response of the discrete-time system to input `u`.

        Calls the `dlsim` function to compute and return the system's response to the input.
        """
        return dlsim(self, u, t, x0=x0)

    def bode(self, w=None, n=100):
        r"""
        Calculate Bode magnitude and phase data of a discrete-time system.

        Returns a 3-tuple containing arrays of frequencies [rad/s], magnitude
        [dB] and phase [deg].

        Uses the `dbode` function to calculate and return the Bode plot data.
        """
        return dbode(self, w=w, n=n)
    # 定义一个方法 `freqresp`，用于计算离散时间系统的频率响应。
    # 可选参数 `w` 表示角频率，`n` 表示采样点数，`whole` 表示是否计算整个频率范围。
    def freqresp(self, w=None, n=10000, whole=False):
        """
        Calculate the frequency response of a discrete-time system.

        Returns a 2-tuple containing arrays of frequencies [rad/s] and
        complex magnitude.
        See `dfreqresp` for details.

        """
        # 调用外部定义的 `dfreqresp` 函数来计算频率响应，并返回其结果。
        return dfreqresp(self, w=w, n=n, whole=whole)
class TransferFunction(LinearTimeInvariant):
    r"""Linear Time Invariant system class in transfer function form.

    Represents the system as the continuous-time transfer function
    :math:`H(s)=\sum_{i=0}^N b[N-i] s^i / \sum_{j=0}^M a[M-j] s^j` or the
    discrete-time transfer function
    :math:`H(z)=\sum_{i=0}^N b[N-i] z^i / \sum_{j=0}^M a[M-j] z^j`, where
    :math:`b` are elements of the numerator `num`, :math:`a` are elements of
    the denominator `den`, and ``N == len(b) - 1``, ``M == len(a) - 1``.
    `TransferFunction` systems inherit additional
    functionality from the `lti`, respectively the `dlti` classes, depending on
    which system representation is used.

    Parameters
    ----------
    *system: arguments
        The `TransferFunction` class can be instantiated with 1 or 2
        arguments. The following gives the number of input arguments and their
        interpretation:

            * 1: `lti` or `dlti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 2: array_like: (numerator, denominator)
    dt: float, optional
        Sampling time [s] of the discrete-time systems. Defaults to `None`
        (continuous-time). Must be specified as a keyword argument, for
        example, ``dt=0.1``.

    See Also
    --------
    ZerosPolesGain, StateSpace, lti, dlti
    tf2ss, tf2zpk, tf2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `TransferFunction` system representation (such as the `A`, `B`, `C`, `D`
    state-space matrices) is very inefficient and may lead to numerical
    inaccuracies.  It is better to convert to the specific system
    representation first. For example, call ``sys = sys.to_ss()`` before
    accessing/changing the A, B, C, D system matrices.

    If (numerator, denominator) is passed in for ``*system``, coefficients
    for both the numerator and denominator should be specified in descending
    exponent order (e.g. ``s^2 + 3s + 5`` or ``z^2 + 3z + 5`` would be
    represented as ``[1, 3, 5]``)

    Examples
    --------
    Construct the transfer function
    :math:`H(s) = \frac{s^2 + 3s + 3}{s^2 + 2s + 1}`:

    >>> from scipy import signal

    >>> num = [1, 3, 3]
    >>> den = [1, 2, 1]

    >>> signal.TransferFunction(num, den)
    TransferFunctionContinuous(
    array([1., 3., 3.]),
    array([1., 2., 1.]),
    dt: None
    )

    Construct the transfer function
    :math:`H(z) = \frac{z^2 + 3z + 3}{z^2 + 2z + 1}` with a sampling time of
    0.1 seconds:

    >>> signal.TransferFunction(num, den, dt=0.1)
    TransferFunctionDiscrete(
    array([1., 3., 3.]),
    array([1., 2., 1.]),
    dt: 0.1
    )
    """
    def __new__(cls, *system, **kwargs):
        """处理对象转换，如果输入是 lti 的实例。"""
        # 如果输入参数长度为1且第一个参数是 LinearTimeInvariant 的实例，则返回其对应的 TF 表示
        if len(system) == 1 and isinstance(system[0], LinearTimeInvariant):
            return system[0].to_tf()

        # 选择是否从 `lti` 或者 `dlti` 继承
        if cls is TransferFunction:
            # 如果没有指定离散时间参数，则返回连续时间的 TransferFunction
            if kwargs.get('dt') is None:
                return TransferFunctionContinuous.__new__(
                    TransferFunctionContinuous,
                    *system,
                    **kwargs)
            else:
                # 否则返回离散时间的 TransferFunction
                return TransferFunctionDiscrete.__new__(
                    TransferFunctionDiscrete,
                    *system,
                    **kwargs)

        # 没有特殊的转换需求，调用父类的 __new__
        return super().__new__(cls)

    def __init__(self, *system, **kwargs):
        """初始化状态空间 LTI 系统。"""
        # 如果第一个系统参数是 LinearTimeInvariant 的实例，则直接返回
        if isinstance(system[0], LinearTimeInvariant):
            return

        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 初始化分子和分母
        self._num = None
        self._den = None

        # 对输入的系统参数进行标准化并赋值给分子和分母
        self.num, self.den = normalize(*system)

    def __repr__(self):
        """返回系统传递函数的表示形式。"""
        return (
            f'{self.__class__.__name__}(\n'
            f'{repr(self.num)},\n'
            f'{repr(self.den)},\n'
            f'dt: {repr(self.dt)}\n)'
        )

    @property
    def num(self):
        """`TransferFunction` 系统的分子。"""
        return self._num

    @num.setter
    def num(self, num):
        # 确保分子至少是一维的数组
        self._num = atleast_1d(num)

        # 更新输出和输入的维度
        if len(self.num.shape) > 1:
            self.outputs, self.inputs = self.num.shape
        else:
            self.outputs = 1
            self.inputs = 1

    @property
    def den(self):
        """`TransferFunction` 系统的分母。"""
        return self._den

    @den.setter
    def den(self, den):
        # 确保分母至少是一维的数组
        self._den = atleast_1d(den)

    def _copy(self, system):
        """
        复制另一个 `TransferFunction` 对象的参数

        Parameters
        ----------
        system : `TransferFunction`
            要复制的 `StateSpace` 系统对象

        """
        # 复制分子和分母参数
        self.num = system.num
        self.den = system.den

    def to_tf(self):
        """
        返回当前 `TransferFunction` 系统的副本。

        Returns
        -------
        sys : `TransferFunction` 的实例
            当前系统的副本

        """
        return copy.deepcopy(self)
    def to_zpk(self):
        """
        Convert system representation to `ZerosPolesGain`.

        Returns
        -------
        sys : instance of `ZerosPolesGain`
            Zeros, poles, gain representation of the current system

        """
        # 使用 tf2zpk 函数将传递给它的分子和分母系数转换为零点、极点和增益的表示形式，并返回对应的对象
        return ZerosPolesGain(*tf2zpk(self.num, self.den),
                              **self._dt_dict)

    def to_ss(self):
        """
        Convert system representation to `StateSpace`.

        Returns
        -------
        sys : instance of `StateSpace`
            State space model of the current system

        """
        # 使用 tf2ss 函数将传递给它的分子和分母系数转换为状态空间模型的表示形式，并返回对应的对象
        return StateSpace(*tf2ss(self.num, self.den),
                          **self._dt_dict)

    @staticmethod
    def _z_to_zinv(num, den):
        """Change a transfer function from the variable `z` to `z**-1`.

        Parameters
        ----------
        num, den: 1d array_like
            Sequences representing the coefficients of the numerator and
            denominator polynomials, in order of descending degree of 'z'.
            That is, ``5z**2 + 3z + 2`` is presented as ``[5, 3, 2]``.

        Returns
        -------
        num, den: 1d array_like
            Sequences representing the coefficients of the numerator and
            denominator polynomials, in order of ascending degree of 'z**-1'.
            That is, ``5 + 3 z**-1 + 2 z**-2`` is presented as ``[5, 3, 2]``.
        """
        # 计算分子和分母多项式系数的逆变换，从 z 到 z**-1 表示
        diff = len(num) - len(den)
        if diff > 0:
            den = np.hstack((np.zeros(diff), den))
        elif diff < 0:
            num = np.hstack((np.zeros(-diff), num))
        return num, den

    @staticmethod
    def _zinv_to_z(num, den):
        """Change a transfer function from the variable `z` to `z**-1`.

        Parameters
        ----------
        num, den: 1d array_like
            Sequences representing the coefficients of the numerator and
            denominator polynomials, in order of ascending degree of 'z**-1'.
            That is, ``5 + 3 z**-1 + 2 z**-2`` is presented as ``[5, 3, 2]``.

        Returns
        -------
        num, den: 1d array_like
            Sequences representing the coefficients of the numerator and
            denominator polynomials, in order of descending degree of 'z'.
            That is, ``5z**2 + 3z + 2`` is presented as ``[5, 3, 2]``.
        """
        # 计算分子和分母多项式系数的逆变换，从 z**-1 到 z 表示
        diff = len(num) - len(den)
        if diff > 0:
            den = np.hstack((den, np.zeros(diff)))
        elif diff < 0:
            num = np.hstack((num, np.zeros(-diff)))
        return num, den
class TransferFunctionContinuous(TransferFunction, lti):
    r"""
    Continuous-time Linear Time Invariant system in transfer function form.

    Represents the system as the transfer function
    :math:`H(s)=\sum_{i=0}^N b[N-i] s^i / \sum_{j=0}^M a[M-j] s^j`, where
    :math:`b` are elements of the numerator `num`, :math:`a` are elements of
    the denominator `den`, and ``N == len(b) - 1``, ``M == len(a) - 1``.
    Continuous-time `TransferFunction` systems inherit additional
    functionality from the `lti` class.

    Parameters
    ----------
    *system: arguments
        The `TransferFunction` class can be instantiated with 1 or 2
        arguments. The following gives the number of input arguments and their
        interpretation:

            * 1: `lti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 2: array_like: (numerator, denominator)

    See Also
    --------
    ZerosPolesGain, StateSpace, lti
    tf2ss, tf2zpk, tf2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `TransferFunction` system representation (such as the `A`, `B`, `C`, `D`
    state-space matrices) is very inefficient and may lead to numerical
    inaccuracies.  It is better to convert to the specific system
    representation first. For example, call ``sys = sys.to_ss()`` before
    accessing/changing the A, B, C, D system matrices.

    If (numerator, denominator) is passed in for ``*system``, coefficients
    for both the numerator and denominator should be specified in descending
    exponent order (e.g. ``s^2 + 3s + 5`` would be represented as
    ``[1, 3, 5]``)

    Examples
    --------
    Construct the transfer function
    :math:`H(s) = \frac{s^2 + 3s + 3}{s^2 + 2s + 1}`:

    >>> from scipy import signal

    >>> num = [1, 3, 3]
    >>> den = [1, 2, 1]

    >>> signal.TransferFunction(num, den)
    TransferFunctionContinuous(
    array([ 1.,  3.,  3.]),
    array([ 1.,  2.,  1.]),
    dt: None
    )

    """

    def to_discrete(self, dt, method='zoh', alpha=None):
        """
        Returns the discretized `TransferFunction` system.

        Parameters: See `cont2discrete` for details.

        Returns
        -------
        sys: instance of `dlti` and `StateSpace`
        """
        # 转换为离散时间的传递函数系统，通过调用 cont2discrete 函数实现
        return TransferFunction(*cont2discrete((self.num, self.den),
                                               dt,
                                               method=method,
                                               alpha=alpha)[:-1],
                                dt=dt)


class TransferFunctionDiscrete(TransferFunction, dlti):
    r"""
    Discrete-time Linear Time Invariant system in transfer function form.

    Represents the system as the transfer function
    :math:`H(z)=\sum_{i=0}^N b[N-i] z^i / \sum_{j=0}^M a[M-j] z^j`, where
    :math:`b` are elements of the numerator `num`, :math:`a` are elements of
    the denominator `den`, and ``N == len(b) - 1``, ``M == len(a) - 1``.

    """
    Discrete-time `TransferFunction` systems inherit additional functionality
    from the `dlti` class.

    Parameters
    ----------
    *system: arguments
        The `TransferFunction` class can be instantiated with 1 or 2
        arguments. The following gives the number of input arguments and their
        interpretation:

            * 1: `dlti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 2: array_like: (numerator, denominator)
              
    dt: float, optional
        Sampling time [s] of the discrete-time systems. Defaults to `True`
        (unspecified sampling time). Must be specified as a keyword argument,
        for example, ``dt=0.1``.

    See Also
    --------
    ZerosPolesGain, StateSpace, dlti
    tf2ss, tf2zpk, tf2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `TransferFunction` system representation (such as the `A`, `B`, `C`, `D`
    state-space matrices) is very inefficient and may lead to numerical
    inaccuracies.

    If (numerator, denominator) is passed in for ``*system``, coefficients
    for both the numerator and denominator should be specified in descending
    exponent order (e.g., ``z^2 + 3z + 5`` would be represented as
    ``[1, 3, 5]``).

    Examples
    --------
    Construct the transfer function
    :math:`H(z) = \frac{z^2 + 3z + 3}{z^2 + 2z + 1}` with a sampling time of
    0.5 seconds:

    >>> from scipy import signal

    >>> num = [1, 3, 3]
    >>> den = [1, 2, 1]

    >>> signal.TransferFunction(num, den, dt=0.5)
    TransferFunctionDiscrete(
    array([ 1.,  3.,  3.]),
    array([ 1.,  2.,  1.]),
    dt: 0.5
    )

    """
    pass
class ZerosPolesGain(LinearTimeInvariant):
    r"""
    Linear Time Invariant system class in zeros, poles, gain form.

    Represents the system as the continuous- or discrete-time transfer function
    :math:`H(s)=k \prod_i (s - z[i]) / \prod_j (s - p[j])`, where :math:`k` is
    the `gain`, :math:`z` are the `zeros` and :math:`p` are the `poles`.
    `ZerosPolesGain` systems inherit additional functionality from the `lti`,
    respectively the `dlti` classes, depending on which system representation
    is used.

    Parameters
    ----------
    *system : arguments
        The `ZerosPolesGain` class can be instantiated with 1 or 3
        arguments. The following gives the number of input arguments and their
        interpretation:

            * 1: `lti` or `dlti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 3: array_like: (zeros, poles, gain)
    
    dt: float, optional
        Sampling time [s] of the discrete-time systems. Defaults to `None`
        (continuous-time). Must be specified as a keyword argument, for
        example, ``dt=0.1``.

    See Also
    --------
    TransferFunction, StateSpace, lti, dlti
    zpk2ss, zpk2tf, zpk2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `ZerosPolesGain` system representation (such as the `A`, `B`, `C`, `D`
    state-space matrices) is very inefficient and may lead to numerical
    inaccuracies. It is better to convert to the specific system
    representation first. For example, call ``sys = sys.to_ss()`` before
    accessing/changing the A, B, C, D system matrices.

    Examples
    --------
    Construct the transfer function
    :math:`H(s) = \frac{5(s - 1)(s - 2)}{(s - 3)(s - 4)}`:

    >>> from scipy import signal

    >>> signal.ZerosPolesGain([1, 2], [3, 4], 5)
    ZerosPolesGainContinuous(
    array([1, 2]),
    array([3, 4]),
    5,
    dt: None
    )

    Construct the transfer function
    :math:`H(z) = \frac{5(z - 1)(z - 2)}{(z - 3)(z - 4)}` with a sampling time
    of 0.1 seconds:

    >>> signal.ZerosPolesGain([1, 2], [3, 4], 5, dt=0.1)
    ZerosPolesGainDiscrete(
    array([1, 2]),
    array([3, 4]),
    5,
    dt: 0.1
    )
    """
    # 处理对象转换，如果输入是 `lti` 的实例
    def __new__(cls, *system, **kwargs):
        """Handle object conversion if input is an instance of `lti`"""
        if len(system) == 1 and isinstance(system[0], LinearTimeInvariant):
            # 如果系统参数仅包含一个 `lti` 实例，则返回其对应的零极点增益表达式
            return system[0].to_zpk()

        # 选择继承自 `lti` 还是 `dlti`
        if cls is ZerosPolesGain:
            if kwargs.get('dt') is None:
                # 如果没有离散时间参数，创建连续时间的零极点增益对象
                return ZerosPolesGainContinuous.__new__(
                    ZerosPolesGainContinuous,
                    *system,
                    **kwargs)
            else:
                # 如果有离散时间参数，创建离散时间的零极点增益对象
                return ZerosPolesGainDiscrete.__new__(
                    ZerosPolesGainDiscrete,
                    *system,
                    **kwargs
                    )

        # 没有特殊的转换需求，使用默认的对象创建方式
        return super().__new__(cls)

    def __init__(self, *system, **kwargs):
        """Initialize the zeros, poles, gain system."""
        # 如果第一个系统参数是 `lti` 实例，则直接返回，初始化在 `__new__` 中已处理
        if isinstance(system[0], LinearTimeInvariant):
            return

        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 初始化零点、极点和增益
        self._zeros = None
        self._poles = None
        self._gain = None

        # 设置零点、极点和增益
        self.zeros, self.poles, self.gain = system

    def __repr__(self):
        """Return representation of the `ZerosPolesGain` system."""
        # 返回 `ZerosPolesGain` 系统的字符串表示形式
        return (
            f'{self.__class__.__name__}(\n'
            f'{repr(self.zeros)},\n'
            f'{repr(self.poles)},\n'
            f'{repr(self.gain)},\n'
            f'dt: {repr(self.dt)}\n)'
        )

    @property
    def zeros(self):
        """Zeros of the `ZerosPolesGain` system."""
        # 返回 `ZerosPolesGain` 系统的零点
        return self._zeros

    @zeros.setter
    def zeros(self, zeros):
        # 设置 `ZerosPolesGain` 系统的零点，并确保至少是一维数组
        self._zeros = atleast_1d(zeros)

        # 更新输出和输入的维度信息
        if len(self.zeros.shape) > 1:
            self.outputs, self.inputs = self.zeros.shape
        else:
            self.outputs = 1
            self.inputs = 1

    @property
    def poles(self):
        """Poles of the `ZerosPolesGain` system."""
        # 返回 `ZerosPolesGain` 系统的极点
        return self._poles

    @poles.setter
    def poles(self, poles):
        # 设置 `ZerosPolesGain` 系统的极点
        self._poles = atleast_1d(poles)

    @property
    def gain(self):
        """Gain of the `ZerosPolesGain` system."""
        # 返回 `ZerosPolesGain` 系统的增益
        return self._gain

    @gain.setter
    def gain(self, gain):
        # 设置 `ZerosPolesGain` 系统的增益
        self._gain = gain

    def _copy(self, system):
        """
        Copy the parameters of another `ZerosPolesGain` system.

        Parameters
        ----------
        system : instance of `ZerosPolesGain`
            The zeros, poles gain system that is to be copied

        """
        # 复制另一个 `ZerosPolesGain` 系统的参数：极点、零点和增益
        self.poles = system.poles
        self.zeros = system.zeros
        self.gain = system.gain
    def to_tf(self):
        """
        Convert system representation to `TransferFunction`.

        Returns
        -------
        sys : instance of `TransferFunction`
            Transfer function of the current system

        """
        # 使用当前对象的零点、极点和增益，调用 zpk2tf 函数生成传递函数对象，并传入时域字典参数
        return TransferFunction(*zpk2tf(self.zeros, self.poles, self.gain),
                                **self._dt_dict)

    def to_zpk(self):
        """
        Return a copy of the current 'ZerosPolesGain' system.

        Returns
        -------
        sys : instance of `ZerosPolesGain`
            The current system (copy)

        """
        # 使用 copy.deepcopy 复制当前对象并返回
        return copy.deepcopy(self)

    def to_ss(self):
        """
        Convert system representation to `StateSpace`.

        Returns
        -------
        sys : instance of `StateSpace`
            State space model of the current system

        """
        # 使用当前对象的零点、极点和增益，调用 zpk2ss 函数生成状态空间模型对象，并传入时域字典参数
        return StateSpace(*zpk2ss(self.zeros, self.poles, self.gain),
                          **self._dt_dict)
class ZerosPolesGainContinuous(ZerosPolesGain, lti):
    r"""
    Continuous-time Linear Time Invariant system in zeros, poles, gain form.

    Represents the system as the continuous time transfer function
    :math:`H(s)=k \prod_i (s - z[i]) / \prod_j (s - p[j])`, where :math:`k` is
    the `gain`, :math:`z` are the `zeros` and :math:`p` are the `poles`.
    Continuous-time `ZerosPolesGain` systems inherit additional functionality
    from the `lti` class.

    Parameters
    ----------
    *system : arguments
        The `ZerosPolesGain` class can be instantiated with 1 or 3
        arguments. The following gives the number of input arguments and their
        interpretation:

            * 1: `lti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 3: array_like: (zeros, poles, gain)

    See Also
    --------
    TransferFunction, StateSpace, lti
    zpk2ss, zpk2tf, zpk2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `ZerosPolesGain` system representation (such as the `A`, `B`, `C`, `D`
    state-space matrices) is very inefficient and may lead to numerical
    inaccuracies.  It is better to convert to the specific system
    representation first. For example, call ``sys = sys.to_ss()`` before
    accessing/changing the A, B, C, D system matrices.

    Examples
    --------
    Construct the transfer function
    :math:`H(s)=\frac{5(s - 1)(s - 2)}{(s - 3)(s - 4)}`:

    >>> from scipy import signal

    >>> signal.ZerosPolesGain([1, 2], [3, 4], 5)
    ZerosPolesGainContinuous(
    array([1, 2]),
    array([3, 4]),
    5,
    dt: None
    )

    """

    def to_discrete(self, dt, method='zoh', alpha=None):
        """
        Returns the discretized `ZerosPolesGain` system.

        Parameters: See `cont2discrete` for details.

        Returns
        -------
        sys: instance of `dlti` and `ZerosPolesGain`
        """
        # 转换为离散时间 `ZerosPolesGain` 系统
        return ZerosPolesGain(
            *cont2discrete((self.zeros, self.poles, self.gain),
                           dt,
                           method=method,
                           alpha=alpha)[:-1],
            dt=dt)


class ZerosPolesGainDiscrete(ZerosPolesGain, dlti):
    r"""
    Discrete-time Linear Time Invariant system in zeros, poles, gain form.

    Represents the system as the discrete-time transfer function
    :math:`H(z)=k \prod_i (z - q[i]) / \prod_j (z - p[j])`, where :math:`k` is
    the `gain`, :math:`q` are the `zeros` and :math:`p` are the `poles`.
    Discrete-time `ZerosPolesGain` systems inherit additional functionality
    from the `dlti` class.

    Parameters
    ----------
    *system : arguments
        The `ZerosPolesGain` class can be instantiated with 1 or 3
        arguments. The following gives the number of input arguments and their
        interpretation:

            * 1: `dlti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 3: array_like: (zeros, poles, gain)
# 定义 StateSpace 类，表示线性时不变系统的状态空间形式
class StateSpace(LinearTimeInvariant):
    r"""
    线性时不变系统的状态空间形式。

    表示系统为连续时间的一阶微分方程 :math:`\dot{x} = A x + B u` 或离散时间的差分方程
    :math:`x[k+1] = A x[k] + B u[k]`。`StateSpace` 系统从 `lti` 或 `dlti` 类继承额外的功能，
    具体取决于所使用的系统表示。

    Parameters
    ----------
    *system: arguments
        `StateSpace` 类可以使用 1 个或 4 个参数进行实例化。
        下面给出了输入参数的数量及其解释：

            * 1 个参数：`lti` 或 `dlti` 系统：(`StateSpace`, `TransferFunction` 或 `ZerosPolesGain`)
            * 4 个参数：数组形式：(A, B, C, D)
    dt: float, optional
        离散时间系统的采样时间 [秒]。默认为 `None`（连续时间）。必须作为关键字参数指定，例如 `dt=0.1`。

    See Also
    --------
    TransferFunction, ZerosPolesGain, lti, dlti
    ss2zpk, ss2tf, zpk2sos

    Notes
    -----
    修改不属于 `StateSpace` 系统表示的属性的值（如 `zeros` 或 `poles`）非常低效，
    可能导致数值不准确。最好先转换为特定的系统表示。例如，在访问/修改零点、极点或增益之前，
    调用 ``sys = sys.to_zpk()``。

    Examples
    --------
    >>> from scipy import signal
    >>> import numpy as np
    >>> a = np.array([[0, 1], [0, 0]])
    >>> b = np.array([[0], [1]])
    >>> c = np.array([[1, 0]])
    >>> d = np.array([[0]])

    >>> sys = signal.StateSpace(a, b, c, d)
    >>> print(sys)
    StateSpaceContinuous(
    array([[0, 1],
           [0, 0]]),
    array([[0],
           [1]]),
    array([[1, 0]]),
    array([[0]]),
    dt: None
    )

    >>> sys.to_discrete(0.1)
    StateSpaceDiscrete(
    array([[1. , 0.1],
           [0. , 1. ]]),
    array([[0.005],
           [0.1  ]]),
    array([[1, 0]]),
    array([[0]]),
    dt: 0.1
    )

    >>> a = np.array([[1, 0.1], [0, 1]])
    >>> b = np.array([[0.005], [0.1]])

    >>> signal.StateSpace(a, b, c, d, dt=0.1)
    StateSpaceDiscrete(
    array([[1. , 0.1],
           [0. , 1. ]]),
    array([[0.005],
           [0.1  ]]),
    array([[1, 0]]),
    array([[0]]),
    dt: 0.1
    )

    """

    # 覆盖 NumPy 二元操作和通用函数
    __array_priority__ = 100.0
    __array_ufunc__ = None
    def __new__(cls, *system, **kwargs):
        """
        Create new StateSpace object and settle inheritance.
        """
        # Handle object conversion if input is an instance of `lti`
        if len(system) == 1 and isinstance(system[0], LinearTimeInvariant):
            # If `system` is an instance of `LinearTimeInvariant`, convert it to StateSpace
            return system[0].to_ss()

        # Choose whether to inherit from `lti` or from `dlti`
        if cls is StateSpace:
            if kwargs.get('dt') is None:
                # If `dt` keyword argument is not provided, create continuous StateSpace
                return StateSpaceContinuous.__new__(StateSpaceContinuous,
                                                    *system, **kwargs)
            else:
                # If `dt` keyword argument is provided, create discrete StateSpace
                return StateSpaceDiscrete.__new__(StateSpaceDiscrete,
                                                  *system, **kwargs)

        # No special conversion needed, call super to create instance
        return super().__new__(cls)

    def __init__(self, *system, **kwargs):
        """
        Initialize the state space lti/dlti system.
        """
        # Conversion of lti instances is handled in __new__
        if isinstance(system[0], LinearTimeInvariant):
            return  # If `system` is already an instance of `LinearTimeInvariant`, do nothing

        # Remove system arguments, not needed by parents anymore
        super().__init__(**kwargs)

        # Initialize attributes for state space matrices
        self._A = None
        self._B = None
        self._C = None
        self._D = None

        # Normalize and set the state space matrices A, B, C, D
        self.A, self.B, self.C, self.D = abcd_normalize(*system)

    def __repr__(self):
        """
        Return representation of the `StateSpace` system.
        """
        return (
            f'{self.__class__.__name__}(\n'
            f'{repr(self.A)},\n'
            f'{repr(self.B)},\n'
            f'{repr(self.C)},\n'
            f'{repr(self.D)},\n'
            f'dt: {repr(self.dt)}\n)'
        )

    def _check_binop_other(self, other):
        """
        Check if `other` is compatible for binary operations.
        """
        return isinstance(other, (StateSpace, np.ndarray, float, complex,
                                  np.number, int))
    def __mul__(self, other):
        """
        Post-multiply another system or a scalar

        Handles multiplication of systems in the sense of a frequency domain
        multiplication. That means, given two systems E1(s) and E2(s), their
        multiplication, H(s) = E1(s) * E2(s), means that applying H(s) to U(s)
        is equivalent to first applying E2(s), and then E1(s).

        Notes
        -----
        For SISO systems the order of system application does not matter.
        However, for MIMO systems, where the two systems are matrices, the
        order above ensures standard Matrix multiplication rules apply.
        """
        # Check if 'other' is compatible for binary operation
        if not self._check_binop_other(other):
            return NotImplemented

        # If 'other' is a StateSpace object, perform multiplication
        if isinstance(other, StateSpace):
            # Disallow mix of discrete and continuous systems.
            if type(other) is not type(self):
                return NotImplemented

            # Ensure both systems have the same sampling period
            if self.dt != other.dt:
                raise TypeError('Cannot multiply systems with different `dt`.')

            # Dimensions of the matrices involved
            n1 = self.A.shape[0]   # Number of states in self
            n2 = other.A.shape[0]  # Number of states in other

            # Construct the new StateSpace matrices after multiplication
            # State matrix
            a = np.vstack((np.hstack((self.A, np.dot(self.B, other.C))),
                           np.hstack((np.zeros((n2, n1)), other.A))))
            # Input matrix
            b = np.vstack((np.dot(self.B, other.D), other.B))
            # Output matrix
            c = np.hstack((self.C, np.dot(self.D, other.C)))
            # Feedthrough matrix
            d = np.dot(self.D, other.D)
        else:
            # If 'other' is a scalar or matrix, perform scaling in post-multiplication
            # State matrix remains unchanged
            a = self.A
            # Input matrix scaled by 'other'
            b = np.dot(self.B, other)
            # Output matrix remains unchanged
            c = self.C
            # Feedthrough matrix scaled by 'other'
            d = np.dot(self.D, other)

        # Determine common data type for matrices a, b, c, d
        common_dtype = np.result_type(a.dtype, b.dtype, c.dtype, d.dtype)
        # Return a new StateSpace object with matrices a, b, c, d and common_dtype
        return StateSpace(np.asarray(a, dtype=common_dtype),
                          np.asarray(b, dtype=common_dtype),
                          np.asarray(c, dtype=common_dtype),
                          np.asarray(d, dtype=common_dtype),
                          **self._dt_dict)
    def __rmul__(self, other):
        """
        Pre-multiply a scalar or matrix (but not StateSpace).
        """
        # 检查是否可以执行乘法操作
        if not self._check_binop_other(other) or isinstance(other, StateSpace):
            return NotImplemented

        # 只有输出部分会被缩放
        a = self.A
        b = self.B
        # 计算缩放后的输出向量
        c = np.dot(other, self.C)
        d = np.dot(other, self.D)

        # 确定返回对象的数据类型
        common_dtype = np.result_type(a.dtype, b.dtype, c.dtype, d.dtype)
        return StateSpace(np.asarray(a, dtype=common_dtype),
                          np.asarray(b, dtype=common_dtype),
                          np.asarray(c, dtype=common_dtype),
                          np.asarray(d, dtype=common_dtype),
                          **self._dt_dict)

    def __neg__(self):
        """
        Negate the system (equivalent to pre-multiplying by -1).
        """
        # 返回一个被取反的系统对象
        return StateSpace(self.A, self.B, -self.C, -self.D, **self._dt_dict)

    def __add__(self, other):
        """
        Adds two systems in the sense of frequency domain addition.
        """
        # 检查是否可以执行加法操作
        if not self._check_binop_other(other):
            return NotImplemented

        if isinstance(other, StateSpace):
            # 不允许混合离散和连续系统
            if type(other) is not type(self):
                raise TypeError(f'Cannot add {type(self)} and {type(other)}')

            if self.dt != other.dt:
                raise TypeError('Cannot add systems with different `dt`.')

            # 系统的互联
            a = linalg.block_diag(self.A, other.A)
            b = np.vstack((self.B, other.B))
            c = np.hstack((self.C, other.C))
            d = self.D + other.D
        else:
            other = np.atleast_2d(other)
            if self.D.shape == other.shape:
                # 标量或矩阵实际上是一个静态系统 (A=0, B=0, C=0)
                a = self.A
                b = self.B
                c = self.C
                d = self.D + other
            else:
                raise ValueError("Cannot add systems with incompatible "
                                 f"dimensions ({self.D.shape} and {other.shape})")

        # 确定返回对象的数据类型
        common_dtype = np.result_type(a.dtype, b.dtype, c.dtype, d.dtype)
        return StateSpace(np.asarray(a, dtype=common_dtype),
                          np.asarray(b, dtype=common_dtype),
                          np.asarray(c, dtype=common_dtype),
                          np.asarray(d, dtype=common_dtype),
                          **self._dt_dict)
    # 定义对象之间的减法操作，重载 `__sub__` 方法
    def __sub__(self, other):
        # 检查二元操作的有效性，如果无效则返回 NotImplemented
        if not self._check_binop_other(other):
            return NotImplemented
        
        # 返回当前对象与 `other` 的相反数的加法结果
        return self.__add__(-other)

    # 定义右加法操作的方法，重载 `__radd__` 方法
    def __radd__(self, other):
        # 检查二元操作的有效性，如果无效则返回 NotImplemented
        if not self._check_binop_other(other):
            return NotImplemented
        
        # 返回 `other` 与当前对象的加法结果
        return self.__add__(other)

    # 定义右减法操作的方法，重载 `__rsub__` 方法
    def __rsub__(self, other):
        # 检查二元操作的有效性，如果无效则返回 NotImplemented
        if not self._check_binop_other(other):
            return NotImplemented
        
        # 返回当前对象的相反数与 `other` 的加法结果
        return (-self).__add__(other)

    # 定义除法操作的方法，重载 `__truediv__` 方法
    def __truediv__(self, other):
        """
        Divide by a scalar
        """
        # 检查二元操作的有效性，或者 `other` 是 `StateSpace` 类型则返回 NotImplemented
        if not self._check_binop_other(other) or isinstance(other, StateSpace):
            return NotImplemented
        
        # 如果 `other` 是 numpy 数组且维度大于0，则抛出异常
        if isinstance(other, np.ndarray) and other.ndim > 0:
            raise ValueError("Cannot divide StateSpace by non-scalar numpy arrays")
        
        # 返回当前对象乘以 `1/other` 的乘法结果
        return self.__mul__(1/other)

    @property
    def A(self):
        """State matrix of the `StateSpace` system."""
        return self._A

    @A.setter
    def A(self, A):
        # 设置状态矩阵 `_A`，调用 `_atleast_2d_or_none` 函数处理参数 `A`
        self._A = _atleast_2d_or_none(A)

    @property
    def B(self):
        """Input matrix of the `StateSpace` system."""
        return self._B

    @B.setter
    def B(self, B):
        # 设置输入矩阵 `_B`，调用 `_atleast_2d_or_none` 函数处理参数 `B`
        self._B = _atleast_2d_or_none(B)
        # 设置输入数 `inputs` 为 `_B` 的最后一个维度的大小
        self.inputs = self.B.shape[-1]

    @property
    def C(self):
        """Output matrix of the `StateSpace` system."""
        return self._C

    @C.setter
    def C(self, C):
        # 设置输出矩阵 `_C`，调用 `_atleast_2d_or_none` 函数处理参数 `C`
        self._C = _atleast_2d_or_none(C)
        # 设置输出数 `outputs` 为 `_C` 的第一个维度的大小
        self.outputs = self.C.shape[0]

    @property
    def D(self):
        """Feedthrough matrix of the `StateSpace` system."""
        return self._D

    @D.setter
    def D(self, D):
        # 设置传递矩阵 `_D`，调用 `_atleast_2d_or_none` 函数处理参数 `D`
        self._D = _atleast_2d_or_none(D)

    def _copy(self, system):
        """
        Copy the parameters of another `StateSpace` system.

        Parameters
        ----------
        system : instance of `StateSpace`
            The state-space system that is to be copied

        """
        # 复制另一个 `StateSpace` 系统的参数到当前对象
        self.A = system.A
        self.B = system.B
        self.C = system.C
        self.D = system.D

    def to_tf(self, **kwargs):
        """
        Convert system representation to `TransferFunction`.

        Parameters
        ----------
        kwargs : dict, optional
            Additional keywords passed to `ss2zpk`

        Returns
        -------
        sys : instance of `TransferFunction`
            Transfer function of the current system

        """
        # 转换系统表示为 `TransferFunction` 对象，返回转换后的结果
        return TransferFunction(*ss2tf(self._A, self._B, self._C, self._D,
                                       **kwargs), **self._dt_dict)
    # 将系统表示转换为 `ZerosPolesGain` 形式
    def to_zpk(self, **kwargs):
        """
        Convert system representation to `ZerosPolesGain`.

        Parameters
        ----------
        kwargs : dict, optional
            Additional keywords passed to `ss2zpk`

        Returns
        -------
        sys : instance of `ZerosPolesGain`
            Zeros, poles, gain representation of the current system

        """
        # 调用 `ss2zpk` 函数将状态空间表示转换为零极点增益形式，并传递额外的关键字参数
        return ZerosPolesGain(*ss2zpk(self._A, self._B, self._C, self._D,
                                      **kwargs), **self._dt_dict)

    # 返回当前 `StateSpace` 系统的深拷贝
    def to_ss(self):
        """
        Return a copy of the current `StateSpace` system.

        Returns
        -------
        sys : instance of `StateSpace`
            The current system (copy)

        """
        # 使用 `copy.deepcopy` 创建当前 `StateSpace` 系统的深拷贝并返回
        return copy.deepcopy(self)
class StateSpaceContinuous(StateSpace, lti):
    r"""
    Continuous-time Linear Time Invariant system in state-space form.

    Represents the system as the continuous-time, first order differential
    equation :math:`\dot{x} = A x + B u`.
    Continuous-time `StateSpace` systems inherit additional functionality
    from the `lti` class.

    Parameters
    ----------
    *system: arguments
        The `StateSpace` class can be instantiated with 1 or 3 arguments.
        The following gives the number of input arguments and their
        interpretation:

            * 1: `lti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 4: array_like: (A, B, C, D)

    See Also
    --------
    TransferFunction, ZerosPolesGain, lti
    ss2zpk, ss2tf, zpk2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `StateSpace` system representation (such as `zeros` or `poles`) is very
    inefficient and may lead to numerical inaccuracies.  It is better to
    convert to the specific system representation first. For example, call
    ``sys = sys.to_zpk()`` before accessing/changing the zeros, poles or gain.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal

    >>> a = np.array([[0, 1], [0, 0]])
    >>> b = np.array([[0], [1]])
    >>> c = np.array([[1, 0]])
    >>> d = np.array([[0]])

    >>> sys = signal.StateSpace(a, b, c, d)
    >>> print(sys)
    StateSpaceContinuous(
    array([[0, 1],
           [0, 0]]),
    array([[0],
           [1]]),
    array([[1, 0]]),
    array([[0]]),
    dt: None
    )

    """

    def to_discrete(self, dt, method='zoh', alpha=None):
        """
        Returns the discretized `StateSpace` system.

        Parameters: See `cont2discrete` for details.

        Returns
        -------
        sys: instance of `dlti` and `StateSpace`
        """
        # 调用 cont2discrete 函数将连续时间系统转换为离散时间系统
        return StateSpace(*cont2discrete((self.A, self.B, self.C, self.D),
                                         dt,
                                         method=method,
                                         alpha=alpha)[:-1],
                          dt=dt)


class StateSpaceDiscrete(StateSpace, dlti):
    r"""
    Discrete-time Linear Time Invariant system in state-space form.

    Represents the system as the discrete-time difference equation
    :math:`x[k+1] = A x[k] + B u[k]`.
    `StateSpace` systems inherit additional functionality from the `dlti`
    class.

    Parameters
    ----------
    *system: arguments
        The `StateSpace` class can be instantiated with 1 or 3 arguments.
        The following gives the number of input arguments and their
        interpretation:

            * 1: `dlti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 4: array_like: (A, B, C, D)

    """
    # dt: float, optional
    #     离散时间系统的采样时间 [秒]。默认为 `True`（未指定采样时间）。必须作为关键字参数指定，例如 ``dt=0.1``。
    # 
    # See Also
    # --------
    # TransferFunction, ZerosPolesGain, dlti
    # ss2zpk, ss2tf, zpk2sos
    # 
    # Notes
    # -----
    # 更改不属于 `StateSpace` 系统表示的属性的值（例如 `zeros` 或 `poles`）非常低效，并可能导致数值不准确。
    # 最好先转换为特定的系统表示。例如，在访问/更改零点、极点或增益之前调用 ``sys = sys.to_zpk()``。
    # 
    # Examples
    # --------
    # >>> import numpy as np
    # >>> from scipy import signal
    # 
    # >>> a = np.array([[1, 0.1], [0, 1]])
    # >>> b = np.array([[0.005], [0.1]])
    # >>> c = np.array([[1, 0]])
    # >>> d = np.array([[0]])
    # 
    # >>> signal.StateSpace(a, b, c, d, dt=0.1)
    # StateSpaceDiscrete(
    # array([[ 1. ,  0.1],
    #        [ 0. ,  1. ]]),
    # array([[ 0.005],
    #        [ 0.1  ]]),
    # array([[1, 0]]),
    # array([[0]]),
    # dt: 0.1
    # )
    """
    pass
# 定义函数 lsim，用于模拟连续时间线性系统的输出
def lsim(system, U, T, X0=None, interp=True):
    """
    Simulate output of a continuous-time linear system.

    Parameters
    ----------
    system : an instance of the LTI class or a tuple describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:

        * 1: (instance of `lti`)
        * 2: (num, den)
        * 3: (zeros, poles, gain)
        * 4: (A, B, C, D)

    U : array_like
        An input array describing the input at each time `T`
        (interpolation is assumed between given times).  If there are
        multiple inputs, then each column of the rank-2 array
        represents an input.  If U = 0 or None, a zero input is used.
    T : array_like
        The time steps at which the input is defined and at which the
        output is desired.  Must be nonnegative, increasing, and equally spaced.
    X0 : array_like, optional
        The initial conditions on the state vector (zero by default).
    interp : bool, optional
        Whether to use linear (True, the default) or zero-order-hold (False)
        interpolation for the input array.

    Returns
    -------
    T : 1D ndarray
        Time values for the output.
    yout : 1D ndarray
        System response.
    xout : ndarray
        Time evolution of the state vector.

    Notes
    -----
    If (num, den) is passed in for ``system``, coefficients for both the
    numerator and denominator should be specified in descending exponent
    order (e.g. ``s^2 + 3s + 5`` would be represented as ``[1, 3, 5]``).

    Examples
    --------
    We'll use `lsim` to simulate an analog Bessel filter applied to
    a signal.

    >>> import numpy as np
    >>> from scipy.signal import bessel, lsim
    >>> import matplotlib.pyplot as plt

    Create a low-pass Bessel filter with a cutoff of 12 Hz.

    >>> b, a = bessel(N=5, Wn=2*np.pi*12, btype='lowpass', analog=True)

    Generate data to which the filter is applied.

    >>> t = np.linspace(0, 1.25, 500, endpoint=False)

    The input signal is the sum of three sinusoidal curves, with
    frequencies 4 Hz, 40 Hz, and 80 Hz.  The filter should mostly
    eliminate the 40 Hz and 80 Hz components, leaving just the 4 Hz signal.

    >>> u = (np.cos(2*np.pi*4*t) + 0.6*np.sin(2*np.pi*40*t) +
    ...      0.5*np.cos(2*np.pi*80*t))

    Simulate the filter with `lsim`.

    >>> tout, yout, xout = lsim((b, a), U=u, T=t)

    Plot the result.

    >>> plt.plot(t, u, 'r', alpha=0.5, linewidth=1, label='input')
    >>> plt.plot(tout, yout, 'k', linewidth=1.5, label='output')
    >>> plt.legend(loc='best', shadow=True, framealpha=1)
    >>> plt.grid(alpha=0.3)
    >>> plt.xlabel('t')
    >>> plt.show()

    In a second example, we simulate a double integrator ``y'' = u``, with
    a constant input ``u = 1``.  We'll use the state space representation
    of the integrator.

    >>> from scipy.signal import lti
    >>> A = np.array([[0.0, 1.0], [0.0, 0.0]])
    """
    >>> B = np.array([[0.0], [1.0]])
    >>> C = np.array([[1.0, 0.0]])
    >>> D = 0.0
    >>> system = lti(A, B, C, D)
    
    
    # 定义系统的输入矩阵 B，输出矩阵 C，以及传递函数的传输常数 D
    B = np.array([[0.0], [1.0]])
    C = np.array([[1.0, 0.0]])
    D = 0.0
    # 根据给定的状态空间表示 (A, B, C, D) 创建系统对象
    system = lti(A, B, C, D)
    
    t = np.linspace(0, 5, num=50)
    u = np.ones_like(t)
    
    
    # 定义时间向量 t，以及输入信号向量 u，用于模拟系统的响应
    t = np.linspace(0, 5, num=50)
    u = np.ones_like(t)
    
    # 执行模拟，并绘制系统的输出响应 y，预期绘制的曲线为 y = 0.5*t**2
    tout, y, x = lsim(system, u, t)
    plt.plot(t, y)
    plt.grid(alpha=0.3)
    plt.xlabel('t')
    plt.show()
    
    
    """
    如果 system 是 lti 类型的对象，则将其转换为状态空间表示 sys。
    如果 system 是 dlti 类型的对象，则抛出 AttributeError。
    如果 system 是其他类型的对象，则将其解析为 lti 对象，并转换为状态空间表示 sys。
    """
    if isinstance(system, lti):
        sys = system._as_ss()
    elif isinstance(system, dlti):
        raise AttributeError('lsim can only be used with continuous-time '
                             'systems.')
    else:
        sys = lti(*system)._as_ss()
    
    T = atleast_1d(T)
    if len(T.shape) != 1:
        raise ValueError("T must be a rank-1 array.")
    
    # 将系统的状态空间矩阵 A, B, C, D 转换为 numpy 数组
    A, B, C, D = map(np.asarray, (sys.A, sys.B, sys.C, sys.D))
    n_states = A.shape[0]
    n_inputs = B.shape[1]
    
    n_steps = T.size
    if X0 is None:
        X0 = zeros(n_states, sys.A.dtype)
    xout = np.empty((n_steps, n_states), sys.A.dtype)
    
    if T[0] == 0:
        xout[0] = X0
    elif T[0] > 0:
        # 向前推进到初始时间，输入为零
        xout[0] = dot(X0, linalg.expm(transpose(A) * T[0]))
    else:
        raise ValueError("Initial time must be nonnegative")
    
    no_input = (U is None or
                (isinstance(U, (int, float)) and U == 0.) or
                not np.any(U))
    
    if n_steps == 1:
        yout = squeeze(xout @ C.T)
        if not no_input:
            yout += squeeze(U @ D.T)
        return T, yout, squeeze(xout)
    
    dt = T[1] - T[0]
    if not np.allclose(np.diff(T), dt):
        raise ValueError("Time steps are not equally spaced.")
    
    if no_input:
        # 零输入情况：直接使用矩阵指数
        expAT_dt = linalg.expm(A.T * dt)
        for i in range(1, n_steps):
            xout[i] = xout[i-1] @ expAT_dt
        yout = squeeze(xout @ C.T)
        return T, yout, squeeze(xout)
    
    # 非零输入情况
    U = atleast_1d(U)
    if U.ndim == 1:
        U = U[:, np.newaxis]
    
    if U.shape[0] != n_steps:
        raise ValueError("U must have the same number of rows "
                         "as elements in T.")
    
    if U.shape[1] != n_inputs:
        raise ValueError("System does not define that many inputs.")
    if not interp:
        # 如果不使用插值，采用零阶保持法
        # 算法：对于从时间0到时间dt的积分，我们解决以下方程组
        #   xdot = A x + B u,  x(0) = x0
        #   udot = 0,          u(0) = u0.
        #
        # 解法如下：
        #   [ x(dt) ]       [ A*dt   B*dt ] [ x0 ]
        #   [ u(dt) ] = exp [  0     0    ] [ u0 ]
        M = np.vstack([np.hstack([A * dt, B * dt]),
                       np.zeros((n_inputs, n_states + n_inputs))])
        # 转置所有内容，因为状态和输入是行向量
        expMT = linalg.expm(M.T)
        Ad = expMT[:n_states, :n_states]
        Bd = expMT[n_states:, :n_states]
        for i in range(1, n_steps):
            xout[i] = xout[i-1] @ Ad + U[i-1] @ Bd
    else:
        # 如果使用插值，采用线性插值法
        # 算法：对于从时间0到时间dt的积分，输入u在u(0)=u0和u(dt)=u1之间进行线性插值，我们解决以下方程组
        #   xdot = A x + B u,        x(0) = x0
        #   udot = (u1 - u0) / dt,   u(0) = u0.
        #
        # 解法如下：
        #   [ x(dt) ]       [ A*dt  B*dt  0 ] [  x0   ]
        #   [ u(dt) ] = exp [  0     0    I ] [  u0   ]
        #   [u1 - u0]       [  0     0    0 ] [u1 - u0]
        M = np.vstack([np.hstack([A * dt, B * dt,
                                  np.zeros((n_states, n_inputs))]),
                       np.hstack([np.zeros((n_inputs, n_states + n_inputs)),
                                  np.identity(n_inputs)]),
                       np.zeros((n_inputs, n_states + 2 * n_inputs))])
        expMT = linalg.expm(M.T)
        Ad = expMT[:n_states, :n_states]
        Bd1 = expMT[n_states+n_inputs:, :n_states]
        Bd0 = expMT[n_states:n_states + n_inputs, :n_states] - Bd1
        for i in range(1, n_steps):
            xout[i] = xout[i-1] @ Ad + U[i-1] @ Bd0 + U[i] @ Bd1

    # 计算输出yout，其中C是输出矩阵，D是反馈矩阵
    yout = squeeze(xout @ C.T) + squeeze(U @ D.T)
    # 返回时间向量T，输出yout和状态向量xout
    return T, yout, squeeze(xout)
# 计算响应时间的合理样本集，用于连续系统的冲激和阶跃响应
def _default_response_times(A, n):
    """Compute a reasonable set of time samples for the response time.

    This function is used by `impulse` and `step` to compute the response time
    when the `T` argument to the function is None.

    Parameters
    ----------
    A : array_like
        The system matrix, which is square.
    n : int
        The number of time samples to generate.

    Returns
    -------
    t : ndarray
        The 1-D array of length `n` of time samples at which the response
        is to be computed.
    """
    # 计算系统矩阵的特征值
    vals = linalg.eigvals(A)
    # 取实部的绝对值的最小值
    r = min(abs(real(vals)))
    # 处理当最小值为零时的情况
    if r == 0.0:
        r = 1.0
    # 计算临界时间
    tc = 1.0 / r
    # 创建时间样本的线性空间
    t = linspace(0.0, 7 * tc, n)
    return t


def impulse(system, X0=None, T=None, N=None):
    """Impulse response of continuous-time system.

    Parameters
    ----------
    system : an instance of the LTI class or a tuple of array_like
        describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:

            * 1 (instance of `lti`)
            * 2 (num, den)
            * 3 (zeros, poles, gain)
            * 4 (A, B, C, D)

    X0 : array_like, optional
        Initial state-vector. Defaults to zero.
    T : array_like, optional
        Time points. Computed if not given.
    N : int, optional
        The number of time points to compute (if `T` is not given).

    Returns
    -------
    T : ndarray
        A 1-D array of time points.
    yout : ndarray
        A 1-D array containing the impulse response of the system (except for
        singularities at zero).

    Notes
    -----
    If (num, den) is passed in for ``system``, coefficients for both the
    numerator and denominator should be specified in descending exponent
    order (e.g. ``s^2 + 3s + 5`` would be represented as ``[1, 3, 5]``).

    Examples
    --------
    Compute the impulse response of a second order system with a repeated
    root: ``x''(t) + 2*x'(t) + x(t) = u(t)``

    >>> from scipy import signal
    >>> system = ([1.0], [1.0, 2.0, 1.0])
    >>> t, y = signal.impulse(system)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(t, y)

    """
    # 如果系统是 lti 类的实例，则转换为状态空间表示
    if isinstance(system, lti):
        sys = system._as_ss()
    # 如果系统是 dlti 类的实例，则抛出错误
    elif isinstance(system, dlti):
        raise AttributeError('impulse can only be used with continuous-time '
                             'systems.')
    # 如果系统是其他类型的数组或元组，则转换为 lti 类实例的状态空间表示
    else:
        sys = lti(*system)._as_ss()
    # 计算初始状态向量 X
    if X0 is None:
        X = squeeze(sys.B)
    else:
        X = squeeze(sys.B + X0)
    # 如果未指定时间点数目 N，则默认为 100
    if N is None:
        N = 100
    # 如果未指定时间点数组 T，则根据默认函数生成
    if T is None:
        T = _default_response_times(sys.A, N)
    else:
        T = asarray(T)

    # 使用 lsim 函数计算系统的冲激响应
    _, h, _ = lsim(sys, 0., T, X, interp=False)
    return T, h


def step(system, X0=None, T=None, N=None):
    """Step response of continuous-time system.

    Parameters
    ----------
    system : an instance of the LTI class or a tuple of array_like
        describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:

            * 1 (instance of `lti`)
            * 2 (num, den)
            * 3 (zeros, poles, gain)
            * 4 (A, B, C, D)

    X0 : array_like, optional
        Initial state-vector. Defaults to zero.
    T : array_like, optional
        Time points. Computed if not given.
    N : int, optional
        The number of time points to compute (if `T` is not given).

    Returns
    -------
    T : ndarray
        A 1-D array of time points.
    yout : ndarray
        A 1-D array containing the step response of the system.

    """
    """
    Calculate the step response of a linear time-invariant (LTI) system.

    Parameters
    ----------
    system : an instance of the LTI class or a tuple of array_like
        describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:

            * 1 (instance of `lti`)
              Single LTI system instance.
            * 2 (num, den)
              Numerator and denominator coefficients of the transfer function.
            * 3 (zeros, poles, gain)
              Zeros, poles, and gain of the system.
            * 4 (A, B, C, D)
              State-space representation matrices.

    X0 : array_like, optional
        Initial state-vector (default is zero).
    T : array_like, optional
        Time points (computed if not given).
    N : int, optional
        Number of time points to compute if `T` is not given.

    Returns
    -------
    T : 1D ndarray
        Output time points.
    yout : 1D ndarray
        Step response of system.


    Notes
    -----
    If (num, den) is passed in for ``system``, coefficients for both the
    numerator and denominator should be specified in descending exponent
    order (e.g. ``s^2 + 3s + 5`` would be represented as ``[1, 3, 5]``).

    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> lti = signal.lti([1.0], [1.0, 1.0])
    >>> t, y = signal.step(lti)
    >>> plt.plot(t, y)
    >>> plt.xlabel('Time [s]')
    >>> plt.ylabel('Amplitude')
    >>> plt.title('Step response for 1. Order Lowpass')
    >>> plt.grid()
    """

    # Determine the type of system provided and convert to state-space form
    if isinstance(system, lti):
        # If 'system' is already an instance of lti, convert it to state-space
        sys = system._as_ss()
    elif isinstance(system, dlti):
        # Raise an error if 'system' is an instance of dlti (discrete-time LTI)
        raise AttributeError('step can only be used with continuous-time '
                             'systems.')
    else:
        # Otherwise, assume 'system' is a tuple and convert it to lti and then to state-space
        sys = lti(*system)._as_ss()

    # Set default number of time points if not provided
    if N is None:
        N = 100

    # Compute time points if 'T' is not provided
    if T is None:
        T = _default_response_times(sys.A, N)
    else:
        T = asarray(T)

    # Create a unit input vector 'U' for simulation
    U = ones(T.shape, sys.A.dtype)

    # Simulate the system using the 'lsim' function to compute the step response
    vals = lsim(sys, U, T, X0=X0, interp=False)

    # Return the time points 'T' and the corresponding step response 'vals'
    return vals[0], vals[1]
# 计算连续时间系统的 Bode 幅度和相位数据

def bode(system, w=None, n=100):
    """
    Calculate Bode magnitude and phase data of a continuous-time system.

    Parameters
    ----------
    system : an instance of the LTI class or a tuple describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:

            * 1 (instance of `lti`)
            * 2 (num, den)
            * 3 (zeros, poles, gain)
            * 4 (A, B, C, D)

    w : array_like, optional
        Array of frequencies (in rad/s). Magnitude and phase data is calculated
        for every value in this array. If not given a reasonable set will be
        calculated.
    n : int, optional
        Number of frequency points to compute if `w` is not given. The `n`
        frequencies are logarithmically spaced in an interval chosen to
        include the influence of the poles and zeros of the system.

    Returns
    -------
    w : 1D ndarray
        Frequency array [rad/s]
    mag : 1D ndarray
        Magnitude array [dB]
    phase : 1D ndarray
        Phase array [deg]

    Notes
    -----
    If (num, den) is passed in for ``system``, coefficients for both the
    numerator and denominator should be specified in descending exponent
    order (e.g. ``s^2 + 3s + 5`` would be represented as ``[1, 3, 5]``).

    .. versionadded:: 0.11.0

    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> sys = signal.TransferFunction([1], [1, 1])
    >>> w, mag, phase = signal.bode(sys)

    >>> plt.figure()
    >>> plt.semilogx(w, mag)    # Bode magnitude plot
    >>> plt.figure()
    >>> plt.semilogx(w, phase)  # Bode phase plot
    >>> plt.show()

    """
    # 调用 freqresp 函数计算系统的频率响应
    w, y = freqresp(system, w=w, n=n)

    # 计算幅度响应，转换为分贝
    mag = 20.0 * np.log10(abs(y))
    # 计算相位响应，确保相位连续性，并转换为角度
    phase = np.unwrap(np.arctan2(y.imag, y.real)) * 180.0 / np.pi

    # 返回频率数组、幅度数组和相位数组
    return w, mag, phase


def freqresp(system, w=None, n=10000):
    r"""Calculate the frequency response of a continuous-time system.

    Parameters
    ----------
    system : an instance of the `lti` class or a tuple describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:

            * 1 (instance of `lti`)
            * 2 (num, den)
            * 3 (zeros, poles, gain)
            * 4 (A, B, C, D)

    w : array_like, optional
        Array of frequencies (in rad/s). Magnitude and phase data is
        calculated for every value in this array. If not given, a reasonable
        set will be calculated.
    n : int, optional
        Number of frequency points to compute if `w` is not given. The `n`
        frequencies are logarithmically spaced in an interval chosen to
        include the influence of the poles and zeros of the system.

    Returns
    -------
    w : 1D ndarray
        Frequency array [rad/s]
    H : 1D ndarray
        Array of complex magnitude values

    Notes
    -----
    If (num, den) is passed in for ``system``, coefficients for both the
    ...
    """
    # 此函数负责计算连续时间系统的频率响应，返回频率数组和复数幅度值数组
    numerator and denominator should be specified in descending exponent
    order (e.g. ``s^2 + 3s + 5`` would be represented as ``[1, 3, 5]``).

    Examples
    --------
    Generating the Nyquist plot of a transfer function

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    Construct the transfer function :math:`H(s) = \frac{5}{(s-1)^3}`:

    >>> s1 = signal.ZerosPolesGain([], [1, 1, 1], [5])

    >>> w, H = signal.freqresp(s1)

    >>> plt.figure()
    >>> plt.plot(H.real, H.imag, "b")
    >>> plt.plot(H.real, -H.imag, "r")
    >>> plt.show()
    """
# 如果系统是 lti 类型，则根据其类型进行处理
if isinstance(system, lti):
    # 如果系统是 TransferFunction 或者 ZerosPolesGain 类型，则直接使用
    if isinstance(system, (TransferFunction, ZerosPolesGain)):
        sys = system
    else:
        # 否则将其转换为 ZerosPolesGain 类型
        sys = system._as_zpk()
# 如果系统是 dlti 类型，则抛出错误，因为 freqresp 只能处理连续时间系统
elif isinstance(system, dlti):
    raise AttributeError('freqresp can only be used with continuous-time '
                         'systems.')
else:
    # 如果系统是一个通用的系统描述（numerator, denominator），则将其转换为 ZerosPolesGain 类型
    sys = lti(*system)._as_zpk()

# 检查系统是否为单输入单输出（SISO）系统，否则抛出错误
if sys.inputs != 1 or sys.outputs != 1:
    raise ValueError("freqresp() requires a SISO (single input, single "
                     "output) system.")

# 如果指定了频率数组 w，则使用该数组；否则使用参数 n
if w is not None:
    worN = w
else:
    worN = n

# 根据系统类型调用不同的频率响应计算函数
if isinstance(sys, TransferFunction):
    # 在 freqs() 调用中，sys.num.ravel() 用于处理 sys.num 是包含单行的二维数组的情况
    w, h = freqs(sys.num.ravel(), sys.den, worN=worN)

elif isinstance(sys, ZerosPolesGain):
    w, h = freqs_zpk(sys.zeros, sys.poles, sys.gain, worN=worN)

return w, h
# This class provides a convenient way to create objects with attributes initialized from keyword arguments.
# It allows accessing attributes like dictionary keys.
# see https://code.activestate.com/recipes/52308/
class Bunch:
    def __init__(self, **kwds):
        # Update the instance's __dict__ with provided keyword arguments
        self.__dict__.update(kwds)


def _valid_inputs(A, B, poles, method, rtol, maxiter):
    """
    Check the poles come in complex conjugate pairs
    Check shapes of A, B and poles are compatible.
    Check the method chosen is compatible with provided poles
    Return update method to use and ordered poles
    """
    # Convert poles to a NumPy array for consistency
    poles = np.asarray(poles)
    if poles.ndim > 1:
        raise ValueError("Poles must be a 1D array like.")

    # Ensure poles are ordered as complex conjugate pairs
    poles = _order_complex_poles(poles)

    # Check dimensions and properties of matrix A
    if A.ndim > 2:
        raise ValueError("A must be a 2D array/matrix.")
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    # Check dimensions of matrix B
    if B.ndim > 2:
        raise ValueError("B must be a 2D array/matrix")
    
    # Ensure the number of requested poles does not exceed the size of A
    if len(poles) > A.shape[0]:
        raise ValueError("maximum number of poles is %d but you asked for %d" %
                         (A.shape[0], len(poles)))
    
    # Ensure the number of provided poles matches the size of A
    if len(poles) < A.shape[0]:
        raise ValueError("number of poles is %d but you should provide %d" %
                         (len(poles), A.shape[0]))

    # Ensure no pole is repeated more than the rank of matrix B times
    r = np.linalg.matrix_rank(B)
    for p in poles:
        if np.sum(p == poles) > r:
            raise ValueError("at least one of the requested pole is repeated "
                             "more than rank(B) times")

    # Choose the appropriate update method based on the specified method keyword
    update_loop = _YT_loop
    if method not in ('KNV0','YT'):
        raise ValueError("The method keyword must be one of 'YT' or 'KNV0'")

    # If method is 'KNV0', further checks are needed
    if method == "KNV0":
        update_loop = _KNV0_loop
        # Ensure all poles are real for KNV0 method
        if not all(np.isreal(poles)):
            raise ValueError("Complex poles are not supported by KNV0")

    # Ensure maxiter is at least 1
    if maxiter < 1:
        raise ValueError("maxiter must be at least equal to 1")

    # Ensure rtol is within valid range (0 <= rtol <= 1)
    # Negative rtol can be used to force maxiter iterations, hence not checked
    if rtol > 1:
        raise ValueError("rtol can not be greater than 1")

    # Return the chosen update method and ordered poles
    return update_loop, poles


def _order_complex_poles(poles):
    """
    Check we have complex conjugates pairs and reorder P according to YT, ie
    real_poles, complex_i, conjugate complex_i, ....
    The lexicographic sort on the complex poles is added to help the user to
    compare sets of poles.
    """
    # Separate real and imaginary parts of poles and reorder them accordingly
    ordered_poles = np.sort(poles[np.isreal(poles)])
    im_poles = []
    for p in np.sort(poles[np.imag(poles) < 0]):
        if np.conj(p) in poles:
            im_poles.extend((p, np.conj(p)))

    # Concatenate ordered real poles and their conjugates
    ordered_poles = np.hstack((ordered_poles, im_poles))

    # Ensure all complex poles have their conjugates
    if poles.shape[0] != len(ordered_poles):
        raise ValueError("Complex poles must come with their conjugates")

    return ordered_poles


def _KNV0(B, ker_pole, transfer_matrix, j, poles):
    """
    Algorithm "KNV0" Kautsky et Al. Robust pole
    assignment in linear state feedback, Int journal of Control
    """
    # Implementation details of the KNV0 algorithm would go here
    1985, vol 41 p 1129->1155
    https://la.epfl.ch/files/content/sites/la/files/
        users/105941/public/KautskyNicholsDooren

    """
    # 从基础中移除第 j 列
    transfer_matrix_not_j = np.delete(transfer_matrix, j, axis=1)
    # 如果我们对这个矩阵进行 QR 分解，完整模式下 Q=Q0|Q1
    # 那么 Q1 将会是 Q0 的正交单列，这正是我们要找的！

    # 在合并 gh-4249 后，通过使用 QR 更新而不是完整的 QR 分解，可以显著提高速度

    # 要使用 numpy 的 qr 函数进行调试，请取消下面一行的注释
    # Q, R = np.linalg.qr(transfer_matrix_not_j, mode="complete")
    Q, R = s_qr(transfer_matrix_not_j, mode="full")

    # 计算 ker_pole[j] 的转置与自身的乘积
    mat_ker_pj = np.dot(ker_pole[j], ker_pole[j].T)
    # 计算 yj，yj 是 mat_ker_pj 与 Q 的最后一列的乘积
    yj = np.dot(mat_ker_pj, Q[:, -1])

    # 如果 Q[:, -1] 与 ker_pole[j] "几乎"正交，那么它在 ker_pole[j] 中的投影将会接近于 0。
    # 我们要找的向量应该在 ker_pole[j] 中，因此我们使用 transfer_matrix[:, j]
    if not np.allclose(yj, 0):
        # 计算 xj，将 yj 单位化得到 xj
        xj = yj / np.linalg.norm(yj)
        # 将 xj 赋值给 transfer_matrix 的第 j 列
        transfer_matrix[:, j] = xj

        # KNV 不支持复杂的极点，使用 YT 技术，下面两行似乎在大部分情况下工作正常，
        # 但并不完全可靠：
        # transfer_matrix[:, j]=real(xj)
        # transfer_matrix[:, j+1]=imag(xj)

        # 如果你想测试复杂极点的支持，请将以下内容添加到函数开头：
        #    if ~np.isreal(P[j]) and (j>=B.shape[0]-1 or P[j]!=np.conj(P[j+1])):
        #        return
        # 当 imag(xj) 接近于 0 时会出现问题，我不知道如何修复这个问题
def _YT_real(ker_pole, Q, transfer_matrix, i, j):
    """
    Applies algorithm from YT section 6.1 page 19 related to real pairs
    """
    # step 1 page 19: Extract the second to last column of Q as u and the last column as v
    u = Q[:, -2, np.newaxis]
    v = Q[:, -1, np.newaxis]

    # step 2 page 19: Compute matrix m using the formula involving ker_pole[i], ker_pole[j], u, and v
    m = np.dot(np.dot(ker_pole[i].T, np.dot(u, v.T) - np.dot(v, u.T)), ker_pole[j])

    # step 3 page 19: Perform singular value decomposition (SVD) of matrix m
    um, sm, vm = np.linalg.svd(m)
    # Extract the first two columns of um as mu1 and mu2, and the first two rows of vm as nu1 and nu2
    mu1, mu2 = um.T[:2, :, np.newaxis]
    nu1, nu2 = vm[:2, :, np.newaxis]

    # Step 4 page 20: Construct transfer_matrix_j_mo_transfer_matrix_j as a vertical stack of columns i and j of transfer_matrix
    transfer_matrix_j_mo_transfer_matrix_j = np.vstack((
            transfer_matrix[:, i, np.newaxis],
            transfer_matrix[:, j, np.newaxis]))

    # Check if the first and second singular values (sm[0] and sm[1]) are not approximately equal
    if not np.allclose(sm[0], sm[1]):
        # Compute ker_pole_imo_mu1 and ker_pole_i_nu1 using matrix multiplication
        ker_pole_imo_mu1 = np.dot(ker_pole[i], mu1)
        ker_pole_i_nu1 = np.dot(ker_pole[j], nu1)
        # Concatenate ker_pole_imo_mu1 and ker_pole_i_nu1 vertically to form ker_pole_mu_nu
        ker_pole_mu_nu = np.vstack((ker_pole_imo_mu1, ker_pole_i_nu1))
    else:
        # Construct ker_pole_ij by concatenating ker_pole[i] with zeros and zeros with ker_pole[j]
        ker_pole_ij = np.vstack((
                                np.hstack((ker_pole[i], np.zeros(ker_pole[i].shape))),
                                np.hstack((np.zeros(ker_pole[j].shape), ker_pole[j]))
                                ))
        # Concatenate mu1 with mu2 horizontally and nu1 with nu2 horizontally to form mu_nu_matrix
        mu_nu_matrix = np.vstack((np.hstack((mu1, mu2)), np.hstack((nu1, nu2))))
        # Compute ker_pole_mu_nu using matrix multiplication
        ker_pole_mu_nu = np.dot(ker_pole_ij, mu_nu_matrix)

    # Compute transfer_matrix_ij using matrix operations involving ker_pole_mu_nu and transfer_matrix_j_mo_transfer_matrix_j
    transfer_matrix_ij = np.dot(np.dot(ker_pole_mu_nu, ker_pole_mu_nu.T),
                                transfer_matrix_j_mo_transfer_matrix_j)

    # Check if transfer_matrix_ij is not approximately zero
    if not np.allclose(transfer_matrix_ij, 0):
        # Normalize transfer_matrix_ij and assign to transfer_matrix columns i and j
        transfer_matrix_ij = (np.sqrt(2) * transfer_matrix_ij /
                              np.linalg.norm(transfer_matrix_ij))
        transfer_matrix[:, i] = transfer_matrix_ij[:transfer_matrix[:, i].shape[0], 0]
        transfer_matrix[:, j] = transfer_matrix_ij[transfer_matrix[:, i].shape[0]:, 0]
    else:
        # If transfer_matrix_ij is approximately zero, assign ker_pole_mu_nu to transfer_matrix columns i and j
        transfer_matrix[:, i] = ker_pole_mu_nu[:transfer_matrix[:, i].shape[0], 0]
        transfer_matrix[:, j] = ker_pole_mu_nu[transfer_matrix[:, i].shape[0]:, 0]


def _YT_complex(ker_pole, Q, transfer_matrix, i, j):
    """
    Applies algorithm from YT section 6.2 page 20 related to complex pairs
    """
    # step 1 page 20: Construct complex vector u from real parts ur and imaginary parts ui of Q's second to last and last columns
    ur = np.sqrt(2) * Q[:, -2, np.newaxis]
    ui = np.sqrt(2) * Q[:, -1, np.newaxis]
    u = ur + 1j * ui

    # step 2 page 20
    # 从 ker_pole 列表中取出第 i 行
    ker_pole_ij = ker_pole[i]
    # 计算矩阵 m，这里是一系列矩阵乘积和转置的运算
    m = np.dot(np.dot(np.conj(ker_pole_ij.T), np.dot(u, np.conj(u).T) -
               np.dot(np.conj(u), u.T)), ker_pole_ij)

    # 在第 20 页的第 3 步
    # 计算矩阵 m 的特征值和特征向量
    e_val, e_vec = np.linalg.eig(m)
    # 根据特征值的模大小对特征值索引进行排序
    e_val_idx = np.argsort(np.abs(e_val))
    # 提取第一个和第二个最大的特征向量
    mu1 = e_vec[:, e_val_idx[-1], np.newaxis]
    mu2 = e_vec[:, e_val_idx[-2], np.newaxis]

    # 接下来是第 20 页第 6.2 节中公式的粗略 Python 翻译（第 4 步）

    # 注意 transfer_matrix_i 已经被分解为
    # transfer_matrix[i]=real(transfer_matrix_i) 和
    # transfer_matrix[j]=imag(transfer_matrix_i)
    # 构造复数转移矩阵
    transfer_matrix_j_mo_transfer_matrix_j = (
        transfer_matrix[:, i, np.newaxis] +
        1j * transfer_matrix[:, j, np.newaxis]
        )
    
    # 如果两个最大特征值的模大小不相等
    if not np.allclose(np.abs(e_val[e_val_idx[-1]]),
                              np.abs(e_val[e_val_idx[-2]])):
        # 计算 ker_pole_ij 与 mu1 的乘积
        ker_pole_mu = np.dot(ker_pole_ij, mu1)
    else:
        # 否则，构造特征向量矩阵 mu1_mu2_matrix，并计算 ker_pole_ij 与其乘积
        mu1_mu2_matrix = np.hstack((mu1, mu2))
        ker_pole_mu = np.dot(ker_pole_ij, mu1_mu2_matrix)
    
    # 计算 transfer_matrix_i_j
    transfer_matrix_i_j = np.dot(np.dot(ker_pole_mu, np.conj(ker_pole_mu.T)),
                              transfer_matrix_j_mo_transfer_matrix_j)

    # 如果 transfer_matrix_i_j 不接近于零
    if not np.allclose(transfer_matrix_i_j, 0):
        # 归一化 transfer_matrix_i_j
        transfer_matrix_i_j = (transfer_matrix_i_j /
            np.linalg.norm(transfer_matrix_i_j))
        # 将结果存入 transfer_matrix 中
        transfer_matrix[:, i] = np.real(transfer_matrix_i_j[:, 0])
        transfer_matrix[:, j] = np.imag(transfer_matrix_i_j[:, 0])
    else:
        # 如果接近零，与 YT_real 中的思路类似
        transfer_matrix[:, i] = np.real(ker_pole_mu[:, 0])
        transfer_matrix[:, j] = np.imag(ker_pole_mu[:, 0])
def _YT_loop(ker_pole, transfer_matrix, poles, B, maxiter, rtol):
    """
    Algorithm "YT" Tits, Yang. Globally Convergent
    Algorithms for Robust Pole Assignment by State Feedback
    https://hdl.handle.net/1903/5598
    The poles P have to be sorted accordingly to section 6.2 page 20

    """
    # 计算实部为非零的极点数目
    nb_real = poles[np.isreal(poles)].shape[0]
    # 计算实部为非零的极点数目的一半
    hnb = nb_real // 2

    # 根据论文中的建议，初始化更新顺序的列表
    if nb_real > 0:
        # 更新顺序的初始化，用于将最大的实部极点更新为最小的实部极点
        update_order = [[nb_real], [1]]
    else:
        update_order = [[],[]]

    # 计算实部为偶数索引的极点
    r_comp = np.arange(nb_real+1, len(poles)+1, 2)
    # 步骤 1.a
    r_p = np.arange(1, hnb+nb_real % 2)
    update_order[0].extend(2*r_p)
    update_order[1].extend(2*r_p+1)
    # 步骤 1.b
    update_order[0].extend(r_comp)
    update_order[1].extend(r_comp+1)
    # 步骤 1.c
    r_p = np.arange(1, hnb+1)
    update_order[0].extend(2*r_p-1)
    update_order[1].extend(2*r_p)
    # 步骤 1.d
    if hnb == 0 and np.isreal(poles[0]):
        update_order[0].append(1)
        update_order[1].append(1)
    update_order[0].extend(r_comp)
    update_order[1].extend(r_comp+1)
    # 步骤 2.a
    r_j = np.arange(2, hnb+nb_real % 2)
    for j in r_j:
        for i in range(1, hnb+1):
            update_order[0].append(i)
            update_order[1].append(i+j)
    # 步骤 2.b
    if hnb == 0 and np.isreal(poles[0]):
        update_order[0].append(1)
        update_order[1].append(1)
    update_order[0].extend(r_comp)
    update_order[1].extend(r_comp+1)
    # 步骤 2.c
    r_j = np.arange(2, hnb+nb_real % 2)
    for j in r_j:
        for i in range(hnb+1, nb_real+1):
            idx_1 = i+j
            if idx_1 > nb_real:
                idx_1 = i+j-nb_real
            update_order[0].append(i)
            update_order[1].append(idx_1)
    # 步骤 2.d
    if hnb == 0 and np.isreal(poles[0]):
        update_order[0].append(1)
        update_order[1].append(1)
    update_order[0].extend(r_comp)
    update_order[1].extend(r_comp+1)
    # 步骤 3.a
    for i in range(1, hnb+1):
        update_order[0].append(i)
        update_order[1].append(i+hnb)
    # 步骤 3.b
    if hnb == 0 and np.isreal(poles[0]):
        update_order[0].append(1)
        update_order[1].append(1)
    update_order[0].extend(r_comp)
    update_order[1].extend(r_comp+1)

    # 调整更新顺序为 numpy 数组并减去 1，以适应 Python 的索引从 0 开始
    update_order = np.array(update_order).T - 1
    # 初始化停止标志和尝试次数
    stop = False
    nb_try = 0
    # 在最大迭代次数内且未停止的情况下进行循环
    while nb_try < maxiter and not stop:
        # 计算传递矩阵的行列式的绝对值
        det_transfer_matrixb = np.abs(np.linalg.det(transfer_matrix))
        # 按照指定的更新顺序遍历
        for i, j in update_order:
            # 处理对角元素相等的情况
            if i == j:
                # 断言对角元素为0时，即i不等于0时抛出异常信息
                assert i == 0, "i!=0 for KNV call in YT"
                # 断言极点为实数时进行KNV操作
                assert np.isreal(poles[i]), "calling KNV on a complex pole"
                _KNV0(B, ker_pole, transfer_matrix, i, poles)
            else:
                # 删除传递矩阵中的第i列和第j列
                transfer_matrix_not_i_j = np.delete(transfer_matrix, (i, j),
                                                    axis=1)
                # 通过QR更新而不是完整的QR分解，实现了gh-4249的合并后速度显著提升
                # 解开下面一行的注释，使用numpy的QR分解进行调试
                # Q, _ = np.linalg.qr(transfer_matrix_not_i_j, mode="complete")
                Q, _ = s_qr(transfer_matrix_not_i_j, mode="full")

                # 如果极点i是实数
                if np.isreal(poles[i]):
                    # 断言极点j也是实数，否则抛出异常
                    assert np.isreal(poles[j]), "mixing real and complex in YT_real" + str(poles)
                    # 对于实数极点的情况，调用_YT_real函数
                    _YT_real(ker_pole, Q, transfer_matrix, i, j)
                else:
                    # 断言极点i不是实数，否则抛出异常
                    assert ~np.isreal(poles[i]), "mixing real and complex in YT_real" + str(poles)
                    # 对于复数极点的情况，调用_YT_complex函数
                    _YT_complex(ker_pole, Q, transfer_matrix, i, j)

        # 计算更新后传递矩阵的行列式的绝对值，并取当前和前一步的相对误差
        det_transfer_matrix = np.max((np.sqrt(np.spacing(1)),
                                  np.abs(np.linalg.det(transfer_matrix))))
        cur_rtol = np.abs(
            (det_transfer_matrix -
             det_transfer_matrixb) /
            det_transfer_matrix)
        # 如果相对误差小于设定的容差，并且行列式的绝对值大于最小浮点数的平方根，则停止迭代
        if cur_rtol < rtol and det_transfer_matrix > np.sqrt(np.spacing(1)):
            # YT书第21页的收敛测试
            stop = True
        # 增加迭代次数计数器
        nb_try += 1
    # 返回停止标志、当前相对误差和迭代次数
    return stop, cur_rtol, nb_try
# 循环遍历所有极点，并应用 KNV 方法 0 算法
def _KNV0_loop(ker_pole, transfer_matrix, poles, B, maxiter, rtol):
    """
    Loop over all poles one by one and apply KNV method 0 algorithm
    """
    # 这个方法的作用在于能够从 YT 中调用 _KNV0 而不需要循环遍历所有极点，
    # 否则将 _KNV0_loop 和 _KNV0 合并在一个函数中就足够了
    stop = False  # 停止标志，初始化为 False
    nb_try = 0  # 迭代计数器，初始化为 0
    while nb_try < maxiter and not stop:  # 循环直到达到最大迭代次数或者停止条件满足
        det_transfer_matrixb = np.abs(np.linalg.det(transfer_matrix))  # 计算传递矩阵的行列式的绝对值
        for j in range(B.shape[0]):  # 遍历输入矩阵 B 的行数
            _KNV0(B, ker_pole, transfer_matrix, j, poles)  # 调用 _KNV0 函数处理当前行的数据

        det_transfer_matrix = np.max((np.sqrt(np.spacing(1)),  # 计算传递矩阵的行列式的最大值
                                      np.abs(np.linalg.det(transfer_matrix))))
        cur_rtol = np.abs((det_transfer_matrix - det_transfer_matrixb) /
                          det_transfer_matrix)  # 计算当前相对误差
        if cur_rtol < rtol and det_transfer_matrix > np.sqrt(np.spacing(1)):
            # 如果当前相对误差小于设定的容忍度，并且行列式大于一个很小的数
            # （用于数值稳定性），则满足收敛条件
            stop = True  # 设置停止标志为 True

        nb_try += 1  # 迭代计数器加一
    return stop, cur_rtol, nb_try  # 返回停止标志、当前相对误差和迭代次数


def place_poles(A, B, poles, method="YT", rtol=1e-3, maxiter=30):
    """
    Compute K such that eigenvalues (A - dot(B, K))=poles.

    K is the gain matrix such as the plant described by the linear system
    ``AX+BU`` will have its closed-loop poles, i.e the eigenvalues ``A - B*K``,
    as close as possible to those asked for in poles.

    SISO, MISO and MIMO systems are supported.

    Parameters
    ----------
    A, B : ndarray
        State-space representation of linear system ``AX + BU``.
    poles : array_like
        Desired real poles and/or complex conjugates poles.
        Complex poles are only supported with ``method="YT"`` (default).
    method: {'YT', 'KNV0'}, optional
        Which method to choose to find the gain matrix K. One of:

            - 'YT': Yang Tits
            - 'KNV0': Kautsky, Nichols, Van Dooren update method 0

        See References and Notes for details on the algorithms.
    rtol: float, optional
        After each iteration the determinant of the eigenvectors of
        ``A - B*K`` is compared to its previous value, when the relative
        error between these two values becomes lower than `rtol` the algorithm
        stops.  Default is 1e-3.
    maxiter: int, optional
        Maximum number of iterations to compute the gain matrix.
        Default is 30.

    Returns
    -------
    full_state_feedback : Bunch object
        full_state_feedback is composed of:
            gain_matrix : 1-D ndarray
                # 闭环矩阵 K，使得 ``A-BK`` 的特征值尽可能接近请求的极点
            computed_poles : 1-D ndarray
                # ``A-BK`` 对应的极点，首先是递增排列的实数极点，然后是按词典顺序排列的复共轭极点
            requested_poles : 1-D ndarray
                # 算法要求放置的极点，按照与上述相同的顺序排列，它们可能与实际达到的极点不同
            X : 2-D ndarray
                # 传递矩阵 X，满足 ``X * diag(poles) = (A - B*K)*X`` （见注释）
            rtol : float
                # 在 ``det(X)`` 上达到的相对容差（见注释）。当可以解决系统 ``diag(poles) = (A - B*K)`` 时，`rtol` 为 NaN；当优化算法无法执行任何操作时，即 ``B.shape[1] == 1`` 时，`rtol` 为 0。
            nb_iter : int
                # 收敛前执行的迭代次数。当可以解决系统 ``diag(poles) = (A - B*K)`` 时，`nb_iter` 为 NaN；当优化算法无法执行任何操作时，即 ``B.shape[1] == 1`` 时，`nb_iter` 为 0。

    Notes
    -----
    # Tits and Yang (YT) 的论文是 Kautsky et al. (KNV) 论文的更新版本。KNV 依赖于对传递矩阵 X 的 rank-1 更新，使得 ``X * diag(poles) = (A - B*K)*X``；而 YT 使用 rank-2 更新。这通常会得到更加健壮的解决方案（参见 [2]_ 第 21-22 页）。此外，YT 算法支持复杂极点，而 KNV 则不支持其原始版本中。本实现仅实现了 KNV 提出的更新方法 0，因此称为 ``'KNV0'``。

    # 在 Matlab 的 ``place`` 函数中，扩展的 KNV 用于复杂极点，而 YT 则由 Slicot 在名为 ``robpole`` 的非自由许可下分发。不清楚和未记录的是，如何将 KNV0 扩展到复杂极点（Tits 和 Yang 在其论文第 14 页声称其方法不能用于扩展 KNV 到复杂极点），因此本实现仅在此实现中支持 YT 对其的支持。

    # 对于 MIMO 系统的极点配置问题，解决方案并不唯一，因此两种方法均从一个试探性的传递矩阵开始，以各种方式修改它以增加其行列式。已证明两种方法都能收敛到稳定的解决方案，但根据选择初始传递矩阵的方式，它们会收敛到不同的解决方案。因此，“'KNV0'”不一定会产生类似 Matlab 或任何其他算法实现的结果。
    Using the default method ``'YT'`` should be fine in most cases; ``'KNV0'``
    is only provided because it is needed by ``'YT'`` in some specific cases.
    Furthermore ``'YT'`` gives on average more robust results than ``'KNV0'``
    when ``abs(det(X))`` is used as a robustness indicator.

    [2]_ is available as a technical report on the following URL:
    https://hdl.handle.net/1903/5598

    References
    ----------
    .. [1] J. Kautsky, N.K. Nichols and P. van Dooren, "Robust pole assignment
           in linear state feedback", International Journal of Control, Vol. 41
           pp. 1129-1155, 1985.
    .. [2] A.L. Tits and Y. Yang, "Globally convergent algorithms for robust
           pole assignment by state feedback", IEEE Transactions on Automatic
           Control, Vol. 41, pp. 1432-1452, 1996.

    Examples
    --------
    A simple example demonstrating real pole placement using both KNV and YT
    algorithms.  This is example number 1 from section 4 of the reference KNV
    publication ([1]_):

    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> A = np.array([[ 1.380,  -0.2077,  6.715, -5.676  ],
    ...               [-0.5814, -4.290,   0,      0.6750 ],
    ...               [ 1.067,   4.273,  -6.654,  5.893  ],
    ...               [ 0.0480,  4.273,   1.343, -2.104  ]])
    >>> B = np.array([[ 0,      5.679 ],
    ...               [ 1.136,  1.136 ],
    ...               [ 0,      0,    ],
    ...               [-3.146,  0     ]])
    >>> P = np.array([-0.2, -0.5, -5.0566, -8.6659])

    Now compute K with KNV method 0, with the default YT method and with the YT
    method while forcing 100 iterations of the algorithm and print some results
    after each call.

    >>> fsf1 = signal.place_poles(A, B, P, method='KNV0')
    >>> fsf1.gain_matrix
    array([[ 0.20071427, -0.96665799,  0.24066128, -0.10279785],
           [ 0.50587268,  0.57779091,  0.51795763, -0.41991442]])

    >>> fsf2 = signal.place_poles(A, B, P)  # uses YT method
    >>> fsf2.computed_poles
    array([-8.6659, -5.0566, -0.5   , -0.2   ])

    >>> fsf3 = signal.place_poles(A, B, P, rtol=-1, maxiter=100)
    >>> fsf3.X
    array([[ 0.52072442+0.j, -0.08409372+0.j, -0.56847937+0.j,  0.74823657+0.j],
           [-0.04977751+0.j, -0.80872954+0.j,  0.13566234+0.j, -0.29322906+0.j],
           [-0.82266932+0.j, -0.19168026+0.j, -0.56348322+0.j, -0.43815060+0.j],
           [ 0.22267347+0.j,  0.54967577+0.j, -0.58387806+0.j, -0.40271926+0.j]])

    The absolute value of the determinant of X is a good indicator to check the
    robustness of the results, both ``'KNV0'`` and ``'YT'`` aim at maximizing
    it.  Below a comparison of the robustness of the results above:

    >>> abs(np.linalg.det(fsf1.X)) < abs(np.linalg.det(fsf2.X))
    True
    >>> abs(np.linalg.det(fsf2.X)) < abs(np.linalg.det(fsf3.X))
    True

    Now a simple example for complex poles:
    # 定义矩阵 A，表示系统的状态空间矩阵
    A = np.array([[ 0,  7/3.,  0,   0   ],
                  [ 0,   0,    0,  7/9. ],
                  [ 0,   0,    0,   0   ],
                  [ 0,   0,    0,   0   ]])
    
    # 定义矩阵 B，表示输入控制矩阵
    B = np.array([[ 0,  0 ],
                  [ 0,  0 ],
                  [ 1,  0 ],
                  [ 0,  1 ]])
    
    # 定义极点 P，这些极点决定了系统闭环特性
    P = np.array([-3, -1, -2-1j, -2+1j]) / 3.
    
    # 使用 signal 模块中的 place_poles 函数来将极点 P 放置在系统 A、B 中
    fsf = signal.place_poles(A, B, P, method='YT')
    
    # 准备绘制极点在复平面上的图形
    
    # 生成一组角度值，用于绘制单位圆
    t = np.linspace(0, 2*np.pi, 401)
    # 绘制单位圆，以黑色虚线表示
    plt.plot(np.cos(t), np.sin(t), 'k--')
    
    # 绘制期望的极点在复平面上的位置，以白色圆点表示
    plt.plot(fsf.requested_poles.real, fsf.requested_poles.imag,
             'wo', label='Desired')
    
    # 绘制实际放置的极点在复平面上的位置，以蓝色 × 号表示
    plt.plot(fsf.computed_poles.real, fsf.computed_poles.imag, 'bx',
             label='Placed')
    
    # 显示网格
    plt.grid()
    # 设置坐标轴比例相等
    plt.axis('image')
    # 设置坐标轴范围
    plt.axis([-1.1, 1.1, -1.1, 1.1])
    # 添加图例，放置在右上角
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, numpoints=1)
    
    """
    # 执行所有输入的检查工作，但它可能只会给代码增加噪音
    update_loop, poles = _valid_inputs(A, B, poles, method, rtol, maxiter)
    
    # 当前达到的相对容差的值
    cur_rtol = 0
    # 收敛前所需的迭代次数
    nb_iter = 0
    
    # 步骤 A: 对 B 进行 QR 分解，参考 KN 第 1132 页
    # 若要使用 numpy 的 qr 调试，请取消下面的一行注释
    # u, z = np.linalg.qr(B, mode="complete")
    u, z = s_qr(B, mode="full")
    
    # 计算矩阵 B 的秩
    rankB = np.linalg.matrix_rank(B)
    
    # 将 QR 分解得到的结果分解为两部分 u0 和 u1
    u0 = u[:, :rankB]
    u1 = u[:, rankB:]
    
    # 取 z 的前 rankB 行，表示 B 的 QR 分解的第二部分
    z = z[:rankB, :]
    
    # 如果我们可以使用单位矩阵作为 X，解决方案将是显而易见的
    """
    # 如果矩阵 B 是方阵且满秩，方程 A + BK = inv(X)*diag(P)*X 的解只有一个
    # 其中 X=eye(A.shape[0])，即单位矩阵
    # 即 K = inv(B) * (diag(P) - A)
    # 如果矩阵 B 的行数等于其秩（但不是方阵），则存在多个解，可以通过最小二乘法选择其中一个解
    # => 在两种情况下都使用 lstsq 函数。
    # 在这两种情况下，传递矩阵 X 将是单位矩阵 eye(A.shape[0])，且我几乎想不出更好的选择，因此无需优化。
    #
    # 对于复杂的极点，我们使用以下技巧：
    #
    # 矩阵 |a -b| 的特征值为 a+b 和 a-b
    #      |b  a|
    #
    # 矩阵 |a+bi  0| 的特征值为 a+bi 和 a-bi
    #      |0  a-bi|
    #
    # 例如，对于实数域中的第一个矩阵，得到的解对于复数域中的第二个矩阵是适用的。
    diag_poles = np.zeros(A.shape)
    idx = 0
    while idx < poles.shape[0]:
        p = poles[idx]
        diag_poles[idx, idx] = np.real(p)
        # 如果极点 p 是非实数，则处理复数部分
        if ~np.isreal(p):
            diag_poles[idx, idx+1] = -np.imag(p)
            diag_poles[idx+1, idx+1] = np.real(p)
            diag_poles[idx+1, idx] = np.imag(p)
            idx += 1  # 跳过下一个极点
        idx += 1

    # 使用最小二乘法求解控制增益矩阵 K，使得 B*K = diag_poles - A
    gain_matrix = np.linalg.lstsq(B, diag_poles - A, rcond=-1)[0]

    # 转移矩阵 X 是单位矩阵 eye(A.shape[0])
    transfer_matrix = np.eye(A.shape[0])

    # 当前相对容差和迭代次数初始化为 NaN
    cur_rtol = np.nan
    nb_iter = np.nan

    # 注意：Kautsky 的求解是针对 A+BK，但通常形式是 A-BK
    gain_matrix = -gain_matrix

    # 矩阵 K 仍然可能包含接近零虚部的复数部分，这里取其实部
    gain_matrix = np.real(gain_matrix)

    # 创建一个命名元组 Bunch() 来存储完整状态反馈的信息
    full_state_feedback = Bunch()

    # 存储计算得到的控制增益矩阵 K
    full_state_feedback.gain_matrix = gain_matrix

    # 存储计算得到的系统极点（按照特定顺序处理复数极点）
    full_state_feedback.computed_poles = _order_complex_poles(
        np.linalg.eig(A - np.dot(B, gain_matrix))[0]
    )

    # 存储输入的极点 poles
    full_state_feedback.requested_poles = poles

    # 存储传递矩阵 X
    full_state_feedback.X = transfer_matrix

    # 存储当前相对容差和迭代次数
    full_state_feedback.rtol = cur_rtol
    full_state_feedback.nb_iter = nb_iter

    # 返回完整状态反馈信息
    return full_state_feedback
    # 将系统参数转换为离散时间状态空间（dlti-StateSpace）
    if isinstance(system, lti):
        # 如果系统类型是 lti，抛出异常，dlsim 只能用于离散时间的 dlti 系统
        raise AttributeError('dlsim can only be used with discrete-time dlti systems.')
    elif not isinstance(system, dlti):
        # 如果系统不是 dlti 类型，则将其转换为 dlti 对象
        system = dlti(*system[:-1], dt=system[-1])

    # 确保输出结果与输入系统类型兼容
    is_ss_input = isinstance(system, StateSpace)
    # 将系统转换为状态空间表示
    system = system._as_ss()

    # 将输入数组 u 至少转换为一维数组
    u = np.atleast_1d(u)

    # 如果输入 u 是一维数组，则转换为二维数组的列向量
    if u.ndim == 1:
        u = np.atleast_2d(u).T

    # 计算输出数组的样本数和最后的停止时间
    if t is None:
        out_samples = len(u)
        # 根据输入数组长度确定停止时间
        stoptime = (out_samples - 1) * system.dt
    else:
        # 如果指定了时间步长 t，则以 t 中最后一个时间为准
        stoptime = t[-1]
        # 计算输出样本数，确保涵盖整个指定时间范围
        out_samples = int(np.floor(stoptime / system.dt)) + 1

    # 预先建立输出数组
    xout = np.zeros((out_samples, system.A.shape[0]))  # 状态向量的时间演化
    yout = np.zeros((out_samples, system.C.shape[0]))  # 系统响应输出
    tout = np.linspace(0.0, stoptime, num=out_samples)  # 输出的时间点序列

    # 检查初始条件并设置状态向量的初始值
    if x0 is None:
        # 如果未指定初始状态，则默认为零向量
        xout[0, :] = np.zeros((system.A.shape[1],))
    else:
        # 如果指定了初始状态，则使用给定的初始状态向量
        xout[0, :] = np.asarray(x0)

    # 对输入信号进行预插值，以匹配所需的时间步长
    if t is None:
        # 如果未指定时间步长 t，则输入信号不需要插值
        u_dt = u
    else:
        # 如果输入 u 是一维的，将其转换为二维列向量
        if len(u.shape) == 1:
            u = u[:, np.newaxis]

        # 使用线性样条插值函数 make_interp_spline 对输入信号 u 进行插值，生成 u_dt
        u_dt = make_interp_spline(t, u, k=1)(tout)

    # 模拟系统的输出
    for i in range(0, out_samples - 1):
        # 计算下一个时间步的状态向量 xout[i+1, :]
        xout[i+1, :] = (np.dot(system.A, xout[i, :]) +
                        np.dot(system.B, u_dt[i, :]))
        # 计算当前时间步的输出向量 yout[i, :]
        yout[i, :] = (np.dot(system.C, xout[i, :]) +
                      np.dot(system.D, u_dt[i, :]))

    # 最后一个时间点的输出计算
    yout[out_samples-1, :] = (np.dot(system.C, xout[out_samples-1, :]) +
                              np.dot(system.D, u_dt[out_samples-1, :]))

    # 根据是否为状态空间输入选择返回的结果
    if is_ss_input:
        # 返回时间向量 tout、输出向量 yout 和状态向量 xout
        return tout, yout, xout
    else:
        # 只返回时间向量 tout 和输出向量 yout
        return tout, yout
# 离散时间系统的冲激响应函数。
def dimpulse(system, x0=None, t=None, n=None):
    """
    Impulse response of discrete-time system.

    Parameters
    ----------
    system : tuple of array_like or instance of `dlti`
        A tuple describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:

            * 1: (instance of `dlti`)
              If only one element, `system` is already an instance of `dlti`.
            * 3: (num, den, dt)
              Coefficients of the transfer function along with the sampling period `dt`.
            * 4: (zeros, poles, gain, dt)
              Zeros, poles, gain of the system, along with the sampling period `dt`.
            * 5: (A, B, C, D, dt)
              State-space representation matrices A, B, C, D along with the sampling period `dt`.

    x0 : array_like, optional
        Initial state-vector.  Defaults to zero.
    t : array_like, optional
        Time points.  Computed if not given.
    n : int, optional
        The number of time points to compute (if `t` is not given).

    Returns
    -------
    tout : ndarray
        Time values for the output, as a 1-D array.
    yout : tuple of ndarray
        Impulse response of system.  Each element of the tuple represents
        the output of the system based on an impulse in each input.

    See Also
    --------
    impulse, dstep, dlsim, cont2discrete

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> butter = signal.dlti(*signal.butter(3, 0.5))
    >>> t, y = signal.dimpulse(butter, n=25)
    >>> plt.step(t, np.squeeze(y))
    >>> plt.grid()
    >>> plt.xlabel('n [samples]')
    >>> plt.ylabel('Amplitude')

    """
    # 如果 `system` 是 `dlti` 的实例，则转换为状态空间表示
    if isinstance(system, dlti):
        system = system._as_ss()
    # 如果 `system` 是 `lti` 的实例，则抛出错误，因为 `dimpulse` 只能用于离散时间的 `dlti` 系统
    elif isinstance(system, lti):
        raise AttributeError('dimpulse can only be used with discrete-time '
                             'dlti systems.')
    else:
        # 否则，根据提供的参数构建一个 `dlti` 对象，并转换为状态空间表示
        system = dlti(*system[:-1], dt=system[-1])._as_ss()

    # 如果未指定 `n`，默认为 100 个样本点
    if n is None:
        n = 100

    # 如果未指定时间 `t`，则根据 `n` 和系统的采样周期 `dt` 计算时间点
    if t is None:
        t = np.linspace(0, n * system.dt, n, endpoint=False)
    else:
        t = np.asarray(t)

    # 初始化输出变量
    yout = None

    # 对于每个输入，实现一个冲激响应
    for i in range(0, system.inputs):
        # 创建一个单位冲激信号
        u = np.zeros((t.shape[0], system.inputs))
        u[0, i] = 1.0

        # 计算系统的离散时间模拟响应
        one_output = dlsim(system, u, t=t, x0=x0)

        # 将每个输入的响应添加到输出变量中
        if yout is None:
            yout = (one_output[1],)
        else:
            yout = yout + (one_output[1],)

        # 获取时间数组
        tout = one_output[0]

    # 返回时间数组和响应结果
    return tout, yout
    # Convert system to dlti-StateSpace if it's an instance of dlti
    if isinstance(system, dlti):
        system = system._as_ss()
    # Raise an error if system is not an instance of lti
    elif isinstance(system, lti):
        raise AttributeError('dstep can only be used with discrete-time dlti systems.')
    # Convert system to dlti-StateSpace if it's not already in that form
    else:
        system = dlti(*system[:-1], dt=system[-1])._as_ss()
    
    # Default to 100 samples if the number of time points `n` is not specified
    if n is None:
        n = 100
    
    # If `t` (time points) is not specified, create a linearly spaced array of `n` points
    # based on the system's time step (`system.dt`)
    if t is None:
        t = np.linspace(0, n * system.dt, n, endpoint=False)
    else:
        t = np.asarray(t)
    
    # Initialize `yout` as None
    yout = None
    
    # Iterate over each input of the system
    for i in range(0, system.inputs):
        # Create a step input `u` for the i-th input
        u = np.zeros((t.shape[0], system.inputs))
        u[:, i] = np.ones((t.shape[0],))
    
        # Simulate the system response to the step input `u`
        one_output = dlsim(system, u, t=t, x0=x0)
    
        # Store the output (`one_output[1]`) in `yout`
        if yout is None:
            yout = (one_output[1],)
        else:
            yout = yout + (one_output[1],)
    
        # Store the time points (`one_output[0]`) in `tout`
        tout = one_output[0]
    
    # Return the time points `tout` and the system outputs `yout`
    return tout, yout
    """
    Calculate the frequency response of a discrete-time system.

    Parameters
    ----------
    system : an instance of the `dlti` class or a tuple describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:

            * 1 (instance of `dlti`)
            * 2 (numerator, denominator, dt)
            * 3 (zeros, poles, gain, dt)
            * 4 (A, B, C, D, dt)

    w : array_like, optional
        Array of frequencies (in radians/sample). Magnitude and phase data is
        calculated for every value in this array. If not given a reasonable
        set will be calculated.
    n : int, optional
        Number of frequency points to compute if `w` is not given. The `n`
        frequencies are logarithmically spaced in an interval chosen to
        include the influence of the poles and zeros of the system.
    whole : bool, optional
        Normally, if 'w' is not given, frequencies are computed from 0 to the
        Nyquist frequency, pi radians/sample (upper-half of unit-circle). If
        `whole` is True, compute frequencies from 0 to 2*pi radians/sample.

    Returns
    -------
    w : 1D ndarray
        Frequency array [radians/sample]
    H : 1D ndarray
        Array of complex magnitude values

    Notes
    -----
    If (num, den) is passed in for ``system``, coefficients for both the
    numerator and denominator should be specified in descending exponent
    order (e.g. ``z^2 + 3z + 5`` would be represented as ``[1, 3, 5]``).

    .. versionadded:: 0.18.0

    Examples
    --------
    Generating the Nyquist plot of a transfer function

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    Construct the transfer function
    :math:`H(z) = \\frac{1}{z^2 + 2z + 3}` with a sampling time of 0.05
    seconds:

    >>> sys = signal.TransferFunction([1], [1, 2, 3], dt=0.05)

    >>> w, H = signal.dfreqresp(sys)

    >>> plt.figure()
    >>> plt.plot(H.real, H.imag, "b")
    >>> plt.plot(H.real, -H.imag, "r")
    >>> plt.show()

    """
    if not isinstance(system, dlti):
        if isinstance(system, lti):
            raise AttributeError('dfreqresp can only be used with '
                                 'discrete-time systems.')

        # 将系统描述元组转换为 `dlti` 实例
        system = dlti(*system[:-1], dt=system[-1])

    if isinstance(system, StateSpace):
        # 如果系统是状态空间模型，则转换为传递函数模型
        system = system._as_tf()

    if not isinstance(system, (TransferFunction, ZerosPolesGain)):
        raise ValueError('Unknown system type')

    if system.inputs != 1 or system.outputs != 1:
        raise ValueError("dfreqresp requires a SISO (single input, single "
                         "output) system.")

    if w is not None:
        worN = w
    else:
        # 如果未提供频率数组 `w`，则使用默认的频率点数 `n`
        worN = n
    # 如果给定的系统是一个传递函数（TransferFunction）对象
    if isinstance(system, TransferFunction):
        # 将传递函数对象的分子和分母多项式从变量'z'转换为变量'z^-1'，以符合freqz函数的要求
        num, den = TransferFunction._z_to_zinv(system.num.ravel(), system.den)
        # 使用freqz函数计算系统的频率响应
        w, h = freqz(num, den, worN=worN, whole=whole)
    
    # 如果给定的系统是一个零极点增益（ZerosPolesGain）对象
    elif isinstance(system, ZerosPolesGain):
        # 使用freqz_zpk函数计算系统的频率响应，传入系统的零点、极点和增益
        w, h = freqz_zpk(system.zeros, system.poles, system.gain, worN=worN,
                         whole=whole)
    
    # 返回计算得到的频率和响应结果
    return w, h
def dbode(system, w=None, n=100):
    r"""
    Calculate Bode magnitude and phase data of a discrete-time system.

    Parameters
    ----------
    system : an instance of the LTI class or a tuple describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:

            * 1 (instance of `dlti`)
            * 2 (num, den, dt)
            * 3 (zeros, poles, gain, dt)
            * 4 (A, B, C, D, dt)

    w : array_like, optional
        Array of frequencies (in radians/sample). Magnitude and phase data is
        calculated for every value in this array. If not given a reasonable
        set will be calculated.
    n : int, optional
        Number of frequency points to compute if `w` is not given. The `n`
        frequencies are logarithmically spaced in an interval chosen to
        include the influence of the poles and zeros of the system.

    Returns
    -------
    w : 1D ndarray
        Frequency array [rad/time_unit]
    mag : 1D ndarray
        Magnitude array [dB]
    phase : 1D ndarray
        Phase array [deg]

    Notes
    -----
    If (num, den) is passed in for ``system``, coefficients for both the
    numerator and denominator should be specified in descending exponent
    order (e.g. ``z^2 + 3z + 5`` would be represented as ``[1, 3, 5]``).

    .. versionadded:: 0.18.0

    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    Construct the transfer function :math:`H(z) = \frac{1}{z^2 + 2z + 3}` with
    a sampling time of 0.05 seconds:

    >>> sys = signal.TransferFunction([1], [1, 2, 3], dt=0.05)

    Equivalent: sys.bode()

    >>> w, mag, phase = signal.dbode(sys)

    >>> plt.figure()
    >>> plt.semilogx(w, mag)    # Bode magnitude plot
    >>> plt.figure()
    >>> plt.semilogx(w, phase)  # Bode phase plot
    >>> plt.show()

    """
    # 调用 dfreqresp 函数计算频率响应
    w, y = dfreqresp(system, w=w, n=n)

    # 判断 system 是否为 dlti 类的实例，如果是，获取采样时间 dt
    if isinstance(system, dlti):
        dt = system.dt
    else:
        dt = system[-1]  # 否则，获取系统参数元组的最后一个元素作为采样时间 dt

    # 计算幅度响应的分贝值
    mag = 20.0 * np.log10(abs(y))
    # 计算相位响应的角度值
    phase = np.rad2deg(np.unwrap(np.angle(y)))

    # 返回频率数组除以采样时间 dt，以及幅度响应和相位响应数组
    return w / dt, mag, phase
```