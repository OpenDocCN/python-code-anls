# `D:\src\scipysrc\sympy\sympy\physics\control\control_plots.py`

```
# 从 sympy 库中导入需要的符号和函数
from sympy.core.numbers import I, pi
from sympy.functions.elementary.exponential import (exp, log)
from sympy.polys.partfrac import apart
from sympy.core.symbol import Dummy
from sympy.external import import_module
from sympy.functions import arg, Abs
from sympy.integrals.laplace import _fast_inverse_laplace
from sympy.physics.control.lti import SISOLinearTimeInvariant
from sympy.plotting.series import LineOver1DRangeSeries
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polytools import Poly
from sympy.printing.latex import latex

# 指定此模块中可以被导出的符号列表
__all__ = ['pole_zero_numerical_data', 'pole_zero_plot',
    'step_response_numerical_data', 'step_response_plot',
    'impulse_response_numerical_data', 'impulse_response_plot',
    'ramp_response_numerical_data', 'ramp_response_plot',
    'bode_magnitude_numerical_data', 'bode_phase_numerical_data',
    'bode_magnitude_plot', 'bode_phase_plot', 'bode_plot']

# 尝试导入 matplotlib.pyplot 模块，如果失败，则将 plt 设为 None
matplotlib = import_module(
        'matplotlib', import_kwargs={'fromlist': ['pyplot']},
        catch=(RuntimeError,))

# 如果导入成功，将 matplotlib.pyplot 赋值给 plt
if matplotlib:
    plt = matplotlib.pyplot


def _check_system(system):
    """
    Function to check whether the dynamical system passed for plots is
    compatible or not.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The dynamical system object to be checked.

    Raises
    ======

    NotImplementedError
        If the system is not of type SISOLinearTimeInvariant.

    ValueError
        If the system has more than one free symbol or contains unsupported
        expressions involving exponential functions.
    """
    if not isinstance(system, SISOLinearTimeInvariant):
        raise NotImplementedError("Only SISO LTI systems are currently supported.")
    sys = system.to_expr()
    len_free_symbols = len(sys.free_symbols)
    if len_free_symbols > 1:
        raise ValueError("Extra degree of freedom found. Make sure"
            " that there are no free symbols in the dynamical system other"
            " than the variable of Laplace transform.")
    if sys.has(exp):
        # Should test that exp is not part of a constant, in which case
        # no exception is required, compare exp(s) with s*exp(1)
        raise NotImplementedError("Time delay terms are not supported.")


def _poly_roots(poly):
    """
    Function to get the roots of a polynomial.

    Parameters
    ==========

    poly : Poly
        The polynomial object for which roots are to be computed.

    Returns
    =======

    list
        List of roots of the polynomial as Python float/complex numbers.
    """
    def _eval(l):
        return [float(i) if i.is_real else complex(i) for i in l]
    if poly.domain in (QQ, ZZ):
        return _eval(poly.all_roots())
    # XXX: Use all_roots() for irrational coefficients when possible
    # See https://github.com/sympy/sympy/issues/22943
    return _eval(poly.nroots())


def pole_zero_numerical_data(system):
    """
    Returns the numerical data of poles and zeros of the system.

    It is internally used by ``pole_zero_plot`` to get the data
    for plotting poles and zeros. Users can use this data to further
    analyse the dynamics of the system or plot using a different
    backend/plotting-module.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The system for which the pole-zero data is to be computed.

    Returns
    =======

    tuple
        (zeros, poles) where
        zeros : list
            Zeros of the system as a list of Python float/complex.
        poles : list
            Poles of the system as a list of Python float/complex.
    """
    # 实现在系统传递给 laplace 倒数变换时检查这个系统的问题
    _check_system(system)
    # 生成包含系统的值的表达式
    system_expression = system.to_expr()
    # 返回所有根的计算数值
    return (_poly_roots(system_expression.as_numer_denom()[0]), 
            _poly_roots(system_expression.as_numer_denom()[1]))
    # 检查系统是否符合要求，如果不符合则抛出 NotImplementedError 异常
    _check_system(system)
    # 计算系统的等效 TransferFunction 对象
    system = system.doit()  # Get the equivalent TransferFunction object.
    
    # 将系统的分子多项式转换为 Poly 对象
    num_poly = Poly(system.num, system.var)
    # 将系统的分母多项式转换为 Poly 对象
    den_poly = Poly(system.den, system.var)
    
    # 返回分子多项式和分母多项式的根，作为元组返回
    return _poly_roots(num_poly), _poly_roots(den_poly)
def pole_zero_plot(system, pole_color='blue', pole_markersize=10,
    zero_color='orange', zero_markersize=7, grid=True, show_axes=True,
    show=True, **kwargs):
    r"""
    Returns the Pole-Zero plot (also known as PZ Plot or PZ Map) of a system.

    A Pole-Zero plot is a graphical representation of a system's poles and
    zeros. It is plotted on a complex plane, with circular markers representing
    the system's zeros and 'x' shaped markers representing the system's poles.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type systems
        The system for which the pole-zero plot is to be computed.
    pole_color : str, tuple, optional
        The color of the pole points on the plot. Default color
        is blue. The color can be provided as a matplotlib color string,
        or a 3-tuple of floats each in the 0-1 range.
    pole_markersize : Number, optional
        The size of the markers used to mark the poles in the plot.
        Default pole markersize is 10.
    zero_color : str, tuple, optional
        The color of the zero points on the plot. Default color
        is orange. The color can be provided as a matplotlib color string,
        or a 3-tuple of floats each in the 0-1 range.
    zero_markersize : Number, optional
        The size of the markers used to mark the zeros in the plot.
        Default zero markersize is 7.
    grid : boolean, optional
        If ``True``, the plot will have a grid. Defaults to True.
    show_axes : boolean, optional
        If ``True``, the coordinate axes will be shown. Defaults to False.
    show : boolean, optional
        If ``True``, the plot will be displayed otherwise
        the equivalent matplotlib ``plot`` object will be returned.
        Defaults to True.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from sympy.physics.control.control_plots import pole_zero_plot
        >>> tf1 = TransferFunction(s**2 + 1, s**4 + 4*s**3 + 6*s**2 + 5*s + 2, s)
        >>> pole_zero_plot(tf1)   # doctest: +SKIP

    See Also
    ========

    pole_zero_numerical_data

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Pole%E2%80%93zero_plot

    """

    # Obtain numerical data of zeros and poles from the system
    zeros, poles = pole_zero_numerical_data(system)

    # Extract real and imaginary parts of zeros and poles for plotting
    zero_real = [i.real for i in zeros]
    zero_imag = [i.imag for i in zeros]

    pole_real = [i.real for i in poles]
    pole_imag = [i.imag for i in poles]

    # Plot poles with 'x' markers and zeros with 'o' markers
    plt.plot(pole_real, pole_imag, 'x', mfc='none',
        markersize=pole_markersize, color=pole_color)
    plt.plot(zero_real, zero_imag, 'o', markersize=zero_markersize,
        color=zero_color)

    # Labeling and title of the plot
    plt.xlabel('Real Axis')
    plt.ylabel('Imaginary Axis')
    plt.title(f'Poles and Zeros of ${latex(system)}$', pad=20)

    # Display grid if enabled
    if grid:
        plt.grid()
    # 如果设置了 show_axes 参数为 True，则在图上绘制水平和垂直的黑色参考线
    if show_axes:
        plt.axhline(0, color='black')  # 在图上绘制水平的黑色参考线
        plt.axvline(0, color='black')  # 在图上绘制垂直的黑色参考线
    
    # 如果设置了 show 参数为 True，则显示当前图形并返回
    if show:
        plt.show()  # 显示当前绘制的图形
        return  # 结束函数并返回

    # 如果没有设置 show 参数为 True，则仅返回当前的 plt 对象
    return plt  # 返回 matplotlib.pyplot 对象，允许后续操作或进一步绘图
def step_response_numerical_data(system, prec=8, lower_limit=0,
    upper_limit=10, **kwargs):
    """
    Returns the numerical values of the points in the step response plot
    of a SISO continuous-time system. By default, adaptive sampling
    is used. If the user wants to instead get a uniformly
    sampled response, then ``adaptive`` kwarg should be passed ``False``
    and ``n`` must be passed as additional kwargs.
    Refer to the parameters of class :class:`sympy.plotting.series.LineOver1DRangeSeries`
    for more details.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The system for which the unit step response data is to be computed.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0.
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    kwargs :
        Additional keyword arguments are passed to the underlying
        :class:`sympy.plotting.series.LineOver1DRangeSeries` class.

    Returns
    =======

    tuple : (x, y)
        x = Time-axis values of the points in the step response. NumPy array.
        y = Amplitude-axis values of the points in the step response. NumPy array.

    Raises
    ======

    ValueError
        When the lower_limit parameter is less than 0.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunction
    >>> from sympy.physics.control.control_plots import step_response_numerical_data
    >>> tf1 = TransferFunction(s, s**2 + 5*s + 8, s)
    >>> step_response_numerical_data(tf1)   # doctest: +SKIP
    ([0.0, 0.025413462339411542, 0.0484508722725343, ... , 9.670250533855183, 9.844291913708725, 10.0],
    [0.0, 0.023844582399907256, 0.042894276802320226, ..., 6.828770759094287e-12, 6.456457160755703e-12])

    See Also
    ========

    step_response_plot

    """
    # 检查 lower_limit 是否小于 0，如果是则抛出 ValueError 异常
    if lower_limit < 0:
        raise ValueError("Lower limit of time must be greater "
            "than or equal to zero.")
    
    # 检查并确保系统是 SISO 线性时不变系统
    _check_system(system)
    
    # 创建一个虚拟变量 _x
    _x = Dummy("x")
    
    # 将系统的传递函数表达式转换为分数形式，并进行部分分解
    expr = system.to_expr()/(system.var)
    expr = apart(expr, system.var, full=True)
    
    # 使用快速逆拉普拉斯变换计算表达式的数值，精度为 prec
    _y = _fast_inverse_laplace(expr, system.var, _x).evalf(prec)
    
    # 返回由 LineOver1DRangeSeries 类生成的数据点
    return LineOver1DRangeSeries(_y, (_x, lower_limit, upper_limit),
        **kwargs).get_points()
    """
    Plot the unit step response of a Single Input Single Output (SISO)
    Linear Time Invariant (LTI) system.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type
        The LTI SISO system for which the Step Response is to be computed.
    color : str, tuple, optional
        The color of the line. Default is Blue.
    show : boolean, optional
        If ``True``, the plot will be displayed otherwise
        the equivalent matplotlib ``plot`` object will be returned.
        Defaults to True.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0.
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    show_axes : boolean, optional
        If ``True``, the coordinate axes will be shown. Defaults to False.
    grid : boolean, optional
        If ``True``, the plot will have a grid. Defaults to True.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from sympy.physics.control.control_plots import step_response_plot
        >>> tf1 = TransferFunction(8*s**2 + 18*s + 32, s**3 + 6*s**2 + 14*s + 24, s)
        >>> step_response_plot(tf1)   # doctest: +SKIP

    See Also
    ========

    impulse_response_plot, ramp_response_plot

    References
    ==========

    .. [1] https://www.mathworks.com/help/control/ref/lti.step.html

    """
    # Compute the numerical data (time points and amplitudes) for the step response
    x, y = step_response_numerical_data(system, prec=prec,
        lower_limit=lower_limit, upper_limit=upper_limit, **kwargs)
    
    # Plot the step response using matplotlib
    plt.plot(x, y, color=color)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Unit Step Response of ${latex(system)}$', pad=20)

    # Add a grid to the plot if specified
    if grid:
        plt.grid()

    # Show coordinate axes if specified
    if show_axes:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')

    # Show the plot if specified, otherwise return the matplotlib plot object
    if show:
        plt.show()
        return

    return plt
# 定义函数，计算系统的单位冲激响应的数值数据点
def impulse_response_numerical_data(system, prec=8, lower_limit=0,
    upper_limit=10, **kwargs):
    """
    Returns the numerical values of the points in the impulse response plot
    of a SISO continuous-time system. By default, adaptive sampling
    is used. If the user wants to instead get an uniformly
    sampled response, then ``adaptive`` kwarg should be passed ``False``
    and ``n`` must be passed as additional kwargs.
    Refer to the parameters of class :class:`sympy.plotting.series.LineOver1DRangeSeries`
    for more details.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The system for which the impulse response data is to be computed.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0.
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    kwargs :
        Additional keyword arguments are passed to the underlying
        :class:`sympy.plotting.series.LineOver1DRangeSeries` class.

    Returns
    =======

    tuple : (x, y)
        x = Time-axis values of the points in the impulse response. NumPy array.
        y = Amplitude-axis values of the points in the impulse response. NumPy array.

    Raises
    ======

    NotImplementedError
        When a SISO LTI system is not passed.

        When time delay terms are present in the system.

    ValueError
        When more than one free symbol is present in the system.
        The only variable in the transfer function should be
        the variable of the Laplace transform.

        When ``lower_limit`` parameter is less than 0.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunction
    >>> from sympy.physics.control.control_plots import impulse_response_numerical_data
    >>> tf1 = TransferFunction(s, s**2 + 5*s + 8, s)
    >>> impulse_response_numerical_data(tf1)   # doctest: +SKIP
    ([0.0, 0.06616480200395854,... , 9.854500743565858, 10.0],
    [0.9999999799999999, 0.7042848373025861,...,7.170748906965121e-13, -5.1901263495547205e-12])

    See Also
    ========

    impulse_response_plot

    """
    # 检查时间下限是否大于等于0，否则引发数值错误
    if lower_limit < 0:
        raise ValueError("Lower limit of time must be greater "
            "than or equal to zero.")
    # 检查系统是否为单输入单输出（SISO）线性时不变（LTI）系统
    _check_system(system)
    # 创建虚拟变量 _x
    _x = Dummy("x")
    # 获取系统的表达式，并进行部分分解
    expr = system.to_expr()
    expr = apart(expr, system.var, full=True)
    # 对表达式进行快速逆拉普拉斯变换，并进行指定精度的数值计算
    _y = _fast_inverse_laplace(expr, system.var, _x).evalf(prec)
    # 返回使用给定参数绘制的响应曲线上的数据点
    return LineOver1DRangeSeries(_y, (_x, lower_limit, upper_limit),
        **kwargs).get_points()
    # 从给定的 SISO 线性时不变系统中计算脉冲响应数据
    x, y = impulse_response_numerical_data(system, prec=prec,
        lower_limit=lower_limit, upper_limit=upper_limit, **kwargs)
    # 使用 matplotlib 绘制脉冲响应的线条，并设置颜色
    plt.plot(x, y, color=color)
    # 设置 x 轴标签
    plt.xlabel('Time (s)')
    # 设置 y 轴标签
    plt.ylabel('Amplitude')
    # 设置图表标题，使用 LaTeX 格式显示系统表达式
    plt.title(f'Impulse Response of ${latex(system)}$', pad=20)

    # 如果需要显示网格，添加网格线
    if grid:
        plt.grid()
    # 如果需要显示坐标轴，添加水平和垂直的坐标轴线
    if show_axes:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    # 如果 show 参数为 True，则显示图表并返回 None
    if show:
        plt.show()
        return

    # 否则，返回 matplotlib 的 plot 对象
    return plt
# 定义一个函数，返回 SISO 连续时间系统的阶跃响应图中的数值点
def ramp_response_numerical_data(system, slope=1, prec=8,
    lower_limit=0, upper_limit=10, **kwargs):
    """
    Returns the numerical values of the points in the ramp response plot
    of a SISO continuous-time system. By default, adaptive sampling
    is used. If the user wants to instead get an uniformly
    sampled response, then ``adaptive`` kwarg should be passed ``False``
    and ``n`` must be passed as additional kwargs.
    Refer to the parameters of class :class:`sympy.plotting.series.LineOver1DRangeSeries`
    for more details.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The system for which the ramp response data is to be computed.
    slope : Number, optional
        The slope of the input ramp function. Defaults to 1.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0.
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    kwargs :
        Additional keyword arguments are passed to the underlying
        :class:`sympy.plotting.series.LineOver1DRangeSeries` class.

    Returns
    =======

    tuple : (x, y)
        x = Time-axis values of the points in the ramp response plot. NumPy array.
        y = Amplitude-axis values of the points in the ramp response plot. NumPy array.

    Raises
    ======

    NotImplementedError
        When a SISO LTI system is not passed.

        When time delay terms are present in the system.

    ValueError
        When more than one free symbol is present in the system.
        The only variable in the transfer function should be
        the variable of the Laplace transform.

        When ``lower_limit`` parameter is less than 0.

        When ``slope`` is negative.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunction
    >>> from sympy.physics.control.control_plots import ramp_response_numerical_data
    >>> tf1 = TransferFunction(s, s**2 + 5*s + 8, s)
    >>> ramp_response_numerical_data(tf1)   # doctest: +SKIP
    (([0.0, 0.12166980856813935,..., 9.861246379582118, 10.0],
    [1.4504508011325967e-09, 0.006046440489058766,..., 0.12499999999568202, 0.12499999999661349]))

    See Also
    ========

    ramp_response_plot

    """
    # 如果斜率小于0，则引发值错误
    if slope < 0:
        raise ValueError("Slope must be greater than or equal"
            " to zero.")
    # 如果时间下限小于0，则引发值错误
    if lower_limit < 0:
        raise ValueError("Lower limit of time must be greater "
            "than or equal to zero.")
    # 检查系统是否为SISO LTI系统，否则引发未实现错误
    _check_system(system)
    # 创建一个虚拟变量 _x
    _x = Dummy("x")
    # 计算表达式，得到响应的数学表达式
    expr = (slope*system.to_expr())/((system.var)**2)
    # 对表达式进行局部分式分解
    expr = apart(expr, system.var, full=True)
    # 对表达式进行快速逆拉普拉斯变换，以获得时间域的响应
    _y = _fast_inverse_laplace(expr, system.var, _x).evalf(prec)
    # 返回由 LineOver1DRangeSeries 类获取的响应点的元组
    return LineOver1DRangeSeries(_y, (_x, lower_limit, upper_limit),
        **kwargs).get_points()
# 定义一个函数，用于绘制连续时间系统的坡度响应图
def ramp_response_plot(system, slope=1, color='b', prec=8, lower_limit=0,
                       upper_limit=10, show_axes=False, grid=True, show=True, **kwargs):
    """
    Returns the ramp response of a continuous-time system.

    Ramp function is defined as the straight line
    passing through origin ($f(x) = mx$). The slope of
    the ramp function can be varied by the user and
    the default value is 1.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type
        The LTI SISO system for which the Ramp Response is to be computed.
    slope : Number, optional
        The slope of the input ramp function. Defaults to 1.
    color : str, tuple, optional
        The color of the line. Default is Blue.
    show : boolean, optional
        If ``True``, the plot will be displayed otherwise
        the equivalent matplotlib ``plot`` object will be returned.
        Defaults to True.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0.
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    show_axes : boolean, optional
        If ``True``, the coordinate axes will be shown. Defaults to False.
    grid : boolean, optional
        If ``True``, the plot will have a grid. Defaults to True.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from sympy.physics.control.control_plots import ramp_response_plot
        >>> tf1 = TransferFunction(s, (s+4)*(s+8), s)
        >>> ramp_response_plot(tf1, upper_limit=2)   # doctest: +SKIP

    See Also
    ========

    step_response_plot, impulse_response_plot

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Ramp_function

    """
    # 调用内部函数获取坡度响应的数值数据
    x, y = ramp_response_numerical_data(system, slope=slope, prec=prec,
                                        lower_limit=lower_limit, upper_limit=upper_limit, **kwargs)
    # 绘制坡度响应图
    plt.plot(x, y, color=color)
    # 设置 x 轴标签
    plt.xlabel('Time (s)')
    # 设置 y 轴标签
    plt.ylabel('Amplitude')
    # 设置图表标题，包括系统的 LaTeX 表示和坡度值
    plt.title(f'Ramp Response of ${latex(system)}$ [Slope = {slope}]', pad=20)

    # 如果需要网格线，则添加网格
    if grid:
        plt.grid()
    # 如果需要显示坐标轴，则添加坐标轴
    if show_axes:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    # 如果需要显示图表，则显示图表并返回
    if show:
        plt.show()
        return

    # 如果不需要显示图表，则返回 matplotlib 的 plot 对象
    return plt
    _check_system(system)
    # 调用内部函数 `_check_system` 检查输入的系统是否符合要求

    expr = system.to_expr()
    # 将输入的系统转换为表达式，以便后续操作

    freq_units = ('rad/sec', 'Hz')
    # 定义允许的频率单位，分别为弧度/秒和赫兹

    if freq_unit not in freq_units:
        # 如果输入的频率单位不在允许的列表中，抛出异常
        raise ValueError('Only "rad/sec" and "Hz" are accepted frequency units.')

    _w = Dummy("w", real=True)
    # 创建一个实数域的符号变量 `_w`

    if freq_unit == 'Hz':
        repl = I*_w*2*pi
        # 如果频率单位是赫兹，则使用频率变量 `_w`，并转换为弧度/秒单位
    else:
        repl = I*_w
        # 如果频率单位是弧度/秒，则直接使用频率变量 `_w`

    w_expr = expr.subs({system.var: repl})
    # 替换表达式中的系统变量为相应的频率表达式

    mag = 20*log(Abs(w_expr), 10)
    # 计算幅度响应，使用对数尺度（dB）

    x, y = LineOver1DRangeSeries(mag,
        (_w, 10**initial_exp, 10**final_exp), xscale='log', **kwargs).get_points()
    # 使用 `LineOver1DRangeSeries` 类生成幅频曲线的数据点，返回 x 和 y 值

    return x, y
    # 返回幅频曲线的 x 和 y 值
# 返回连续时间系统的波德幅度图。
# 参数与“bode_plot”相同。
def bode_magnitude_plot(system, initial_exp=-5, final_exp=5,
    color='b', show_axes=False, grid=True, show=True, freq_unit='rad/sec', **kwargs):
    # 获取系统的波德幅度数值数据
    x, y = bode_magnitude_numerical_data(system, initial_exp=initial_exp,
        final_exp=final_exp, freq_unit=freq_unit)
    # 绘制波德幅度图
    plt.plot(x, y, color=color, **kwargs)
    # 设置 x 轴为对数尺度
    plt.xscale('log')

    # 设置 x 轴标签，显示频率单位（对数尺度）
    plt.xlabel('Frequency (%s) [Log Scale]' % freq_unit)
    # 设置 y 轴标签
    plt.ylabel('Magnitude (dB)')
    # 设置图表标题，使用 LaTeX 格式显示系统表达式
    plt.title(f'Bode Plot (Magnitude) of ${latex(system)}$', pad=20)

    # 若需要绘制网格，则添加网格线
    if grid:
        plt.grid(True)
    # 若需要显示坐标轴，则绘制 x 轴和 y 轴的零线
    if show_axes:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    # 若需要显示图形，则展示图形并返回
    if show:
        plt.show()
        return

    # 若不需要显示图形，则返回 plt 对象以供进一步操作
    return plt
    [-2.5000000000291665e-05, -3.6180885085e-05, -5.08895483066e-05,...,-3.1415085799262523, -3.14155265358979])

    See Also
    ========

    bode_magnitude_plot, bode_phase_numerical_data

    """
    _check_system(system)  # 调用函数检查系统参数的有效性
    expr = system.to_expr()  # 将系统对象转换为表达式
    freq_units = ('rad/sec', 'Hz')  # 可接受的频率单位
    phase_units = ('rad', 'deg')  # 可接受的相位单位
    if freq_unit not in freq_units:  # 检查频率单位是否有效
        raise ValueError('Only "rad/sec" and "Hz" are accepted frequency units.')
    if phase_unit not in phase_units:  # 检查相位单位是否有效
        raise ValueError('Only "rad" and "deg" are accepted phase units.')

    _w = Dummy("w", real=True)  # 创建一个实数虚拟变量 w
    if freq_unit == 'Hz':  # 如果频率单位是 Hz
        repl = I*_w*2*pi  # 替换表达式中的频率单位为角频率
    else:  # 如果频率单位是 rad/sec
        repl = I*_w  # 替换表达式中的频率单位为角频率
    w_expr = expr.subs({system.var: repl})  # 将变量替换后的表达式赋值给 w_expr

    if phase_unit == 'deg':  # 如果相位单位是度
        phase = arg(w_expr)*180/pi  # 计算相位并转换为度
    else:  # 如果相位单位是弧度
        phase = arg(w_expr)  # 计算相位（弧度）

    x, y = LineOver1DRangeSeries(phase,
        (_w, 10**initial_exp, 10**final_exp), xscale='log', **kwargs).get_points()  # 生成相位对应的 x, y 数据点

    half = None  # 初始化 half 变量为 None
    if phase_unwrap:  # 如果需要进行相位展开
        if(phase_unit == 'rad'):  # 如果相位单位是弧度
            half = pi  # 设置 half 为 pi
        elif(phase_unit == 'deg'):  # 如果相位单位是度
            half = 180  # 设置 half 为 180
    if half:  # 如果 half 不为 None
        unit = 2*half  # 计算单位的值为 2*half
        for i in range(1, len(y)):
            diff = y[i] - y[i - 1]
            if diff > half:      # 如果相位跳变超过 half
                y[i] = (y[i] - unit)  # 调整 y[i] 的值
            elif diff < -half:   # 如果相位跳变小于 -half
                y[i] = (y[i] + unit)  # 调整 y[i] 的值

    return x, y  # 返回计算得到的 x, y 数据点
# 定义函数 `bode_phase_plot`，用于绘制连续时间系统的 Bode 相位图
def bode_phase_plot(system, initial_exp=-5, final_exp=5,
    color='b', show_axes=False, grid=True, show=True, freq_unit='rad/sec', phase_unit='rad', phase_unwrap=True, **kwargs):
    r"""
    Returns the Bode phase plot of a continuous-time system.

    See ``bode_plot`` for all the parameters.
    """
    # 调用函数 `bode_phase_numerical_data`，获取数值数据 x, y
    x, y = bode_phase_numerical_data(system, initial_exp=initial_exp,
        final_exp=final_exp, freq_unit=freq_unit, phase_unit=phase_unit, phase_unwrap=phase_unwrap)
    # 绘制 Bode 相位图
    plt.plot(x, y, color=color, **kwargs)
    # 设置 x 轴为对数刻度
    plt.xscale('log')

    # 设置 x 轴标签，包括频率单位和对数刻度
    plt.xlabel('Frequency (%s) [Log Scale]' % freq_unit)
    # 设置 y 轴标签，显示相位单位
    plt.ylabel('Phase (%s)' % phase_unit)
    # 设置图表标题，使用 LaTeX 格式显示系统表达式
    plt.title(f'Bode Plot (Phase) of ${latex(system)}$', pad=20)

    # 如果需要显示网格线
    if grid:
        plt.grid(True)
    # 如果需要显示坐标轴
    if show_axes:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    # 如果需要显示图表
    if show:
        plt.show()
        return

    # 返回 matplotlib 的 plot 对象
    return plt
    # 生成并获取系统的频率响应的振幅（幅频特性）图
    mag = bode_magnitude_plot(system, initial_exp=initial_exp, final_exp=final_exp,
        show=False, grid=grid, show_axes=show_axes,
        freq_unit=freq_unit, **kwargs)
    
    # 设置振幅图的标题为系统的 LaTeX 表示形式
    mag.title(f'Bode Plot of ${latex(system)}$', pad=20)
    
    # 清除振幅图的 x 轴标签
    mag.xlabel(None)
    
    # 创建一个新的子图（相位图），位置在第二行第一列
    plt.subplot(212)
    
    # 生成并获取系统的相位响应（相频特性）图
    bode_phase_plot(system, initial_exp=initial_exp, final_exp=final_exp,
        show=False, grid=grid, show_axes=show_axes, freq_unit=freq_unit, 
        phase_unit=phase_unit, phase_unwrap=phase_unwrap, **kwargs).title(None)

    # 如果设置了 show 参数为 True，则显示绘制的图形
    if show:
        plt.show()
        # 直接返回，函数结束
        return
    
    # 否则返回 matplotlib 的当前图形对象
    return plt
```