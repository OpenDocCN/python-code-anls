# `D:\src\scipysrc\scipy\scipy\integrate\_ivp\ivp.py`

```
# 导入inspect模块，用于检查和分析Python对象
import inspect
# 导入NumPy库，并使用别名np
import numpy as np
# 从当前包中导入bdf模块中的BDF类
from .bdf import BDF
# 从当前包中导入radau模块中的Radau类
from .radau import Radau
# 从当前包中导入rk模块中的RK23, RK45, DOP853类
from .rk import RK23, RK45, DOP853
# 从当前包中导入lsoda模块中的LSODA类
from .lsoda import LSODA
# 从SciPy库中导入OptimizeResult类
from scipy.optimize import OptimizeResult
# 从当前包中导入common模块中的EPS, OdeSolution变量或类
from .common import EPS, OdeSolution
# 从当前包中导入base模块中的OdeSolver类
from .base import OdeSolver

# 方法字典，将字符串映射到相应的ODE求解器类
METHODS = {'RK23': RK23,
           'RK45': RK45,
           'DOP853': DOP853,
           'Radau': Radau,
           'BDF': BDF,
           'LSODA': LSODA}

# 消息字典，将整数编码映射到相应的消息字符串
MESSAGES = {0: "The solver successfully reached the end of the integration interval.",
            1: "A termination event occurred."}

# 继承自OptimizeResult类的ODE结果类
class OdeResult(OptimizeResult):
    pass

# 准备事件函数，标准化事件函数并提取属性
def prepare_events(events):
    """Standardize event functions and extract attributes."""
    # 如果事件是可调用的，则将其转换为元组
    if callable(events):
        events = (events,)
    
    # 创建空数组以存储最大事件数和方向
    max_events = np.empty(len(events))
    direction = np.empty(len(events))
    for i, event in enumerate(events):
        # 获取事件的terminal属性，默认为None
        terminal = getattr(event, 'terminal', None)
        # 获取事件的direction属性，默认为0
        direction[i] = getattr(event, 'direction', 0)
        
        # 如果terminal属性为None或0，则最大事件数为无穷大
        message = ('The `terminal` attribute of each event '
                   'must be a boolean or positive integer.')
        if terminal is None or terminal == 0:
            max_events[i] = np.inf
        # 如果terminal是正整数，则最大事件数为该值
        elif int(terminal) == terminal and terminal > 0:
            max_events[i] = terminal
        # 否则，引发数值错误
        else:
            raise ValueError(message)
    
    # 返回标准化后的事件列表、最大事件数和方向数组
    return events, max_events, direction

# 解决与ODE事件对应的方程
def solve_event_equation(event, sol, t_old, t):
    """Solve an equation corresponding to an ODE event.

    The equation is ``event(t, y(t)) = 0``, here ``y(t)`` is known from an
    ODE solver using some sort of interpolation. It is solved by
    `scipy.optimize.brentq` with xtol=atol=4*EPS.

    Parameters
    ----------
    event : callable
        Function ``event(t, y)``.
    sol : callable
        Function ``sol(t)`` which evaluates an ODE solution between `t_old`
        and  `t`.
    t_old, t : float
        Previous and new values of time. They will be used as a bracketing
        interval.

    Returns
    -------
    root : float
        Found solution.
    """
    # 从SciPy库中导入brentq函数
    from scipy.optimize import brentq
    # 使用brentq函数解决方程event(t, sol(t)) = 0，使用4倍的EPS作为xtol和rtol
    return brentq(lambda t: event(t, sol(t)), t_old, t,
                  xtol=4 * EPS, rtol=4 * EPS)

# 处理事件的辅助函数
def handle_events(sol, events, active_events, event_count, max_events,
                  t_old, t):
    """Helper function to handle events.

    Parameters
    ----------
    sol : DenseOutput
        Function ``sol(t)`` which evaluates an ODE solution between `t_old`
        and  `t`.
    events : list of callables, length n_events
        Event functions with signatures ``event(t, y)``.
    active_events : ndarray
        Indices of events which occurred.
    event_count : ndarray
        Current number of occurrences for each event.
    max_events : ndarray, shape (n_events,)
        Number of occurrences allowed for each event before integration
        termination is issued.
    t_old, t : float
        Previous and new values of time.

    Returns
    -------
    # `root_indices : ndarray`
    #   事件在 `t_old` 和 `t` 之间及潜在终止前为零的索引数组
    # `roots : ndarray`
    #   发生事件的 t 值数组
    # `terminate : bool`
    #   指示是否发生了终止事件的布尔值
    roots = [solve_event_equation(events[event_index], sol, t_old, t)
             for event_index in active_events]
    
    # 将列表 `roots` 转换为 NumPy 数组
    roots = np.asarray(roots)
    
    # 如果有任何活动事件达到最大事件次数，则执行以下条件
    if np.any(event_count[active_events] >= max_events[active_events]):
        # 如果 t 大于 t_old，则根据 roots 的升序排序
        if t > t_old:
            order = np.argsort(roots)
        else:
            # 如果 t 不大于 t_old，则根据 roots 的降序排序
            order = np.argsort(-roots)
        # 按照排序顺序重新排列活动事件和 roots 数组
        active_events = active_events[order]
        roots = roots[order]
        # 找到第一个满足事件次数超过最大值的事件的索引，截取相关数组
        t = np.nonzero(event_count[active_events]
                       >= max_events[active_events])[0][0]
        active_events = active_events[:t + 1]
        roots = roots[:t + 1]
        # 设置终止标志为 True
        terminate = True
    else:
        # 如果没有事件达到最大次数，则设置终止标志为 False
        terminate = False
    
    # 返回活动事件数组、roots 数组和终止标志
    return active_events, roots, terminate
# 定义函数：找出在积分步骤期间发生的事件
def find_active_events(g, g_new, direction):
    """Find which event occurred during an integration step.

    Parameters
    ----------
    g, g_new : array_like, shape (n_events,)
        Values of event functions at a current and next points.
    direction : ndarray, shape (n_events,)
        Event "direction" according to the definition in `solve_ivp`.

    Returns
    -------
    active_events : ndarray
        Indices of events which occurred during the step.
    """
    # 将输入的 g 和 g_new 转换为 NumPy 数组
    g, g_new = np.asarray(g), np.asarray(g_new)
    # 找到 g <= 0 且 g_new >= 0 的条件
    up = (g <= 0) & (g_new >= 0)
    # 找到 g >= 0 且 g_new <= 0 的条件
    down = (g >= 0) & (g_new <= 0)
    # 找到 g 与 g_new 任一满足条件的情况
    either = up | down
    # 根据事件方向和条件创建掩码
    mask = (up & (direction > 0) |
            down & (direction < 0) |
            either & (direction == 0))

    # 返回满足条件的事件索引
    return np.nonzero(mask)[0]


# 定义函数：解决一个常微分方程组的初值问题
def solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False,
              events=None, vectorized=False, args=None, **options):
    """Solve an initial value problem for a system of ODEs.

    This function numerically integrates a system of ordinary differential
    equations given an initial value::

        dy / dt = f(t, y)
        y(t0) = y0

    Here t is a 1-D independent variable (time), y(t) is an
    N-D vector-valued function (state), and an N-D
    vector-valued function f(t, y) determines the differential equations.
    The goal is to find y(t) approximately satisfying the differential
    equations, given an initial value y(t0)=y0.

    Some of the solvers support integration in the complex domain, but note
    that for stiff ODE solvers, the right-hand side must be
    complex-differentiable (satisfy Cauchy-Riemann equations [11]_).
    To solve a problem in the complex domain, pass y0 with a complex data type.
    Another option always available is to rewrite your problem for real and
    imaginary parts separately.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system: the time derivative of the state ``y``
        at time ``t``. The calling signature is ``fun(t, y)``, where ``t`` is a
        scalar and ``y`` is an ndarray with ``len(y) = len(y0)``. Additional
        arguments need to be passed if ``args`` is used (see documentation of
        ``args`` argument). ``fun`` must return an array of the same shape as
        ``y``. See `vectorized` for more information.
    t_span : 2-member sequence
        Interval of integration (t0, tf). The solver starts with t=t0 and
        integrates until it reaches t=tf. Both t0 and tf must be floats
        or values interpretable by the float conversion function.
    y0 : array_like, shape (n,)
        Initial state. For problems in the complex domain, pass `y0` with a
        complex data type (even if the initial value is purely real).
    method : str, optional
        Integration method to use (default is 'RK45').
    t_eval : array_like or None, optional
        Times at which to store the computed solution (default is None).
    dense_output : bool, optional
        Whether to use a solver providing dense output (True) or not (False,
        default).
    events : callable, optional
        Event functions to detect occurrences during integration. These
        functions must have the signature ``event(t, y)`` and return a float.
    vectorized : bool, optional
        Whether `fun` is implemented in a vectorized fashion (True) or not
        (False, default).
    args : tuple, optional
        Additional arguments to pass to the user-defined functions.
    **options
        Additional keyword arguments passed to the solver. See Notes section
        below for more details.

    """
    # 省略函数内部实现的详细注释，因为示例中只展示了函数定义和参数说明
    pass
    # 定义一个参数 `method`，可以是字符串或者 `OdeSolver` 对象，用于选择积分方法
    method : string or `OdeSolver`, optional
        # 要使用的积分方法有：

            * 'RK45'（默认）：5(4)阶显式Runge-Kutta方法 [1]_。
              误差控制假设为4阶方法的精度，但是使用5阶精确的公式（进行局部外推）。
              密集输出使用四次插值多项式 [2]_。可在复数域中应用。
            * 'RK23'：3(2)阶显式Runge-Kutta方法 [3]_。
              误差控制假设为2阶方法的精度，但是使用3阶精确的公式（进行局部外推）。
              密集输出使用三次埃尔米特多项式。可在复数域中应用。
            * 'DOP853'：8阶显式Runge-Kutta方法 [13]_。
              Python实现的“DOP853”算法，最初是Fortran写的 [14]_。
              密集输出使用精确到7阶的7阶插值多项式。
              可在复数域中应用。
            * 'Radau'：5阶Radau IIA隐式Runge-Kutta方法 [4]_。
              误差控制采用三阶精确的嵌入式公式。
              密集输出使用满足配点条件的三次多项式。
            * 'BDF'：基于反向微分公式的隐式多步变阶（1到5阶）方法 [5]_。
              实现遵循文献 [6]_ 中描述的方法。
              使用准恒定步骤方案，通过NDF修改增强精度。
              可在复数域中应用。
            * 'LSODA'：Adams/BDF方法，具有自动刚性检测和切换 [7]_, [8]_。
              这是一个来自ODEPACK Fortran求解器的包装。

        # 显式Runge-Kutta方法（'RK23'，'RK45'，'DOP853'）适用于非刚性问题，
        # 隐式方法（'Radau'，'BDF'）适用于刚性问题 [9]_。
        # 在Runge-Kutta方法中，推荐使用'DOP853'以获得高精度解（低`rtol`和`atol`值）。

        # 如果不确定，首先尝试运行'RK45'。如果迭代次数异常多、发散或失败，
        # 则问题可能是刚性的，应使用'Radau'或'BDF'。
        # 'LSODA'也可以是一个良好的通用选择，但可能不那么方便，因为它包装了老的Fortran代码。

        # 您还可以传递从`OdeSolver`派生的任意类来实现求解器。
    t_eval : array_like or None, optional
        Times at which to store the computed solution, must be sorted and lie
        within `t_span`. If None (default), use points selected by the solver.
    dense_output : bool, optional
        Whether to compute a continuous solution. Default is False.
    events : callable, or list of callables, optional
        Events to track. If None (default), no events will be tracked.
        Each event occurs at the zeros of a continuous function of time and
        state. Each function must have the signature ``event(t, y)`` where
        additional argument have to be passed if ``args`` is used (see
        documentation of ``args`` argument). Each function must return a
        float. The solver will find an accurate value of `t` at which
        ``event(t, y(t)) = 0`` using a root-finding algorithm. By default,
        all zeros will be found. The solver looks for a sign change over
        each step, so if multiple zero crossings occur within one step,
        events may be missed. Additionally each `event` function might
        have the following attributes:

            terminal: bool or int, optional
                When boolean, whether to terminate integration if this event occurs.
                When integral, termination occurs after the specified the number of
                occurrences of this event.
                Implicitly False if not assigned.
            direction: float, optional
                Direction of a zero crossing. If `direction` is positive,
                `event` will only trigger when going from negative to positive,
                and vice versa if `direction` is negative. If 0, then either
                direction will trigger event. Implicitly 0 if not assigned.

        You can assign attributes like ``event.terminal = True`` to any
        function in Python.
    vectorized : bool, optional
        Whether `fun` can be called in a vectorized fashion. Default is False.

        If ``vectorized`` is False, `fun` will always be called with ``y`` of
        shape ``(n,)``, where ``n = len(y0)``.

        If ``vectorized`` is True, `fun` may be called with ``y`` of shape
        ``(n, k)``, where ``k`` is an integer. In this case, `fun` must behave
        such that ``fun(t, y)[:, i] == fun(t, y[:, i])`` (i.e. each column of
        the returned array is the time derivative of the state corresponding
        with a column of ``y``).

        Setting ``vectorized=True`` allows for faster finite difference
        approximation of the Jacobian by methods 'Radau' and 'BDF', but
        will result in slower execution for other methods and for 'Radau' and
        'BDF' in some circumstances (e.g. small ``len(y0)``).
    args : tuple, optional
        Additional arguments to pass to the user-defined functions. If provided,
        these arguments will be passed to all user-defined functions used in the solver.
        For example, if a function `fun` has the signature `fun(t, y, a, b, c)`, then `args`
        must be a tuple of length 3 containing values for `a`, `b`, and `c`.

    **options
        Options passed to the chosen solver. These options are specific to the solver used.
        Refer to the solver's documentation for details on available options.

    first_step : float or None, optional
        Initial step size for the solver. If `None`, the solver will choose an appropriate initial step size.

    max_step : float, optional
        Maximum allowed step size for the solver. By default, this is set to `np.inf`, meaning
        the solver is not constrained by a maximum step size.

    rtol, atol : float or array_like, optional
        Relative and absolute tolerances for the solver. The solver ensures that the local error
        estimates are kept below `atol + rtol * abs(y)`. Here, `rtol` controls relative accuracy
        (number of correct digits), while `atol` controls absolute accuracy (number of correct decimal places).
        It's important to set `atol` to a value smaller than `rtol * abs(y)` to ensure `rtol` dominates the error.
        Default values are `1e-3` for `rtol` and `1e-6` for `atol`. If `y` components have different scales,
        consider passing an array_like with shape (n,) for `atol` to set different tolerances for each component.

    jac : array_like, sparse_matrix, callable or None, optional
        Jacobian matrix of the right-hand side of the system with respect to `y`, required by certain solver methods
        such as 'Radau', 'BDF', and 'LSODA'. The Jacobian has shape (n, n) where (i, j) element is `d f_i / d y_j`.

        - If `jac` is an array_like or sparse_matrix, it's assumed constant and not supported by 'LSODA'.
        - If `jac` is callable, it depends on `t` and `y` and will be called as `jac(t, y)`. Additional arguments
          can be passed using `args` if specified.
        - If `None` (default), the Jacobian will be approximated using finite differences.

        Providing an explicit Jacobian function is recommended for accuracy, especially for stiff systems.
    # `jac_sparsity`参数：用于定义有限差分近似中雅可比矩阵的稀疏结构。
    # 如果`jac`不为`None`，则此参数将被忽略。
    # 如果雅可比矩阵每行只有少量非零元素，提供稀疏结构可以显著加速计算 [10]_。
    # 零条目意味着雅可比矩阵对应元素始终为零。
    # 如果为None（默认），则假定雅可比矩阵为稠密的。
    # 不支持'LSODA'方法，请使用`lband`和`uband`代替。

    # `lband`和`uband`参数：用于定义'LSODA'方法中雅可比矩阵的带宽。
    # 具体而言，``jac[i, j] != 0``仅当``i - lband <= j <= i + uband``时成立。
    # 默认为None。设置这些参数要求你的`jac`函数以打包格式返回雅可比矩阵：
    # 返回的数组必须有``n``列和``uband + lband + 1``行，其中雅可比矩阵的对角线写入。
    # 具体来说，``jac_packed[uband + i - j , j] = jac[i, j]``。
    # 同样的格式在`scipy.linalg.solve_banded`中使用（查看示例说明）。
    # 这些参数也可以与``jac=None``一起使用，以减少通过有限差分估算的雅可比元素数量。

    # `min_step`参数：'LSODA'方法允许的最小步长。
    # 默认情况下，`min_step`为零。

    # 返回值：
    # Bunch对象，具有以下字段定义：
    # t : ndarray，形状为 (n_points,)，时间点。
    # y : ndarray，形状为 (n, n_points)，在`t`时刻的解的值。
    # sol : `OdeSolution`或None，找到的解作为`OdeSolution`实例；如果`dense_output`设置为False，则为None。
    # t_events : list of ndarray或None，每种事件类型的事件数组列表。如果`events`为None，则为None。
    # y_events : list of ndarray或None，对于每个`t_events`值，对应的解的值。如果`events`为None，则为None。
    # nfev : int，右手边求值的次数。
    # njev : int，雅可比矩阵求值的次数。
    # nlu : int，LU分解的次数。
    # status : int，算法终止的原因：
    #     * -1: 积分步骤失败。
    #     *  0: 求解器成功到达`tspan`的末尾。
    #     *  1: 发生终止事件。
    # message : string，算法终止原因的人类可读描述。
    # success : bool，如果求解器到达区间末尾或发生终止事件，则为True（``status >= 0``）。
    """
    .. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
           formulae", Journal of Computational and Applied Mathematics, Vol. 6,
           No. 1, pp. 19-26, 1980.
    .. [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
           of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.
    .. [3] P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas",
           Appl. Math. Lett. Vol. 2, No. 4. pp. 321-325, 1989.
    .. [4] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II:
           Stiff and Differential-Algebraic Problems", Sec. IV.8.
    .. [5] `Backward Differentiation Formula
            <https://en.wikipedia.org/wiki/Backward_differentiation_formula>`_
            on Wikipedia.
    .. [6] L. F. Shampine, M. W. Reichelt, "THE MATLAB ODE SUITE", SIAM J. SCI.
           COMPUTE., Vol. 18, No. 1, pp. 1-22, January 1997.
    .. [7] A. C. Hindmarsh, "ODEPACK, A Systematized Collection of ODE
           Solvers," IMACS Transactions on Scientific Computation, Vol 1.,
           pp. 55-64, 1983.
    .. [8] L. Petzold, "Automatic selection of methods for solving stiff and
           nonstiff systems of ordinary differential equations", SIAM Journal
           on Scientific and Statistical Computing, Vol. 4, No. 1, pp. 136-148,
           1983.
    .. [9] `Stiff equation <https://en.wikipedia.org/wiki/Stiff_equation>`_ on
           Wikipedia.
    .. [10] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
            sparse Jacobian matrices", Journal of the Institute of Mathematics
            and its Applications, 13, pp. 117-120, 1974.
    .. [11] `Cauchy-Riemann equations
             <https://en.wikipedia.org/wiki/Cauchy-Riemann_equations>`_ on
             Wikipedia.
    .. [12] `Lotka-Volterra equations
            <https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations>`_
            on Wikipedia.
    .. [13] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
            Equations I: Nonstiff Problems", Sec. II.
    .. [14] `Page with original Fortran code of DOP853
            <http://www.unige.ch/~hairer/software.html>`_.

    Examples
    --------
    Basic exponential decay showing automatically chosen time points.

    >>> import numpy as np
    >>> from scipy.integrate import solve_ivp
    >>> def exponential_decay(t, y): return -0.5 * y
    >>> sol = solve_ivp(exponential_decay, [0, 10], [2, 4, 8])
    >>> print(sol.t)
    [ 0.          0.11487653  1.26364188  3.06061781  4.81611105  6.57445806
      8.33328988 10.        ]
    >>> print(sol.y)
    [[2.         1.88836035 1.06327177 0.43319312 0.18017253 0.07483045
      0.03107158 0.01350781]
     [4.         3.7767207  2.12654355 0.86638624 0.36034507 0.14966091
      0.06214316 0.02701561]
     [8.         7.5534414  4.25308709 1.73277247 0.72069014 0.29932181
      0.12428631 0.05403123]]

    Specifying points where the solution is desired.
    """
    >>> sol = solve_ivp(exponential_decay, [0, 10], [2, 4, 8],
    ...                 t_eval=[0, 1, 2, 4, 10])


    # 使用 solve_ivp 函数求解指数衰减方程 exponential_decay
    # 初始时刻 t=0 到 t=10 的时间范围
    # 初始条件为 [2, 4, 8]
    # 指定特定时间点 t_eval=[0, 1, 2, 4, 10] 以返回解的值
    sol = solve_ivp(exponential_decay, [0, 10], [2, 4, 8],
                    t_eval=[0, 1, 2, 4, 10])



    >>> print(sol.t)
    [ 0  1  2  4 10]


    # 打印求解结果中的时间点 sol.t
    print(sol.t)



    >>> print(sol.y)
    [[2.         1.21305369 0.73534021 0.27066736 0.01350938]
     [4.         2.42610739 1.47068043 0.54133472 0.02701876]
     [8.         4.85221478 2.94136085 1.08266944 0.05403753]]


    # 打印求解结果中的解向量 sol.y
    print(sol.y)



    Cannon fired upward with terminal event upon impact. The ``terminal`` and
    ``direction`` fields of an event are applied by monkey patching a function.
    Here ``y[0]`` is position and ``y[1]`` is velocity. The projectile starts
    at position 0 with velocity +10. Note that the integration never reaches
    t=100 because the event is terminal.


    # 描述了一个发射角度朝上的炮弹，遇到地面时终止事件。事件的 ``terminal`` 和
    # ``direction`` 字段通过对函数进行猴子补丁操作。这里的 ``y[0]`` 是位置，``y[1]`` 是速度。
    # 抛射物从位置 0 开始，速度为 +10。请注意，由于事件是终端的，积分从未达到 t=100。



    >>> def upward_cannon(t, y): return [y[1], -0.5]
    >>> def hit_ground(t, y): return y[0]
    >>> hit_ground.terminal = True
    >>> hit_ground.direction = -1
    >>> sol = solve_ivp(upward_cannon, [0, 100], [0, 10], events=hit_ground)


    # 定义了两个函数 upward_cannon 和 hit_ground，分别描述了炮弹上升的情况和地面撞击的情况
    # 设置 hit_ground 函数的终止属性为 True，方向属性为 -1
    # 使用 solve_ivp 函数求解 upward_cannon 函数的数值解，时间范围为 [0, 100]
    # 初始条件为 [0, 10]，并且使用 hit_ground 函数作为事件的终止条件
    sol = solve_ivp(upward_cannon, [0, 100], [0, 10], events=hit_ground)



    >>> print(sol.t_events)
    [array([40.])]


    # 打印事件发生的时间 sol.t_events
    print(sol.t_events)



    >>> print(sol.t)
    [0.00000000e+00 9.99900010e-05 1.09989001e-03 1.10988901e-02
     1.11088891e-01 1.11098890e+00 1.11099890e+01 4.00000000e+01]


    # 打印求解结果中的时间点 sol.t
    print(sol.t)



    >>> print(sol.sol(sol.t_events[1][0]))
    [100.   0.]


    # 打印在事件发生时刻 sol.t_events[1][0] 的解向量 sol.sol(sol.t_events[1][0])
    print(sol.sol(sol.t_events[1][0]))



    >>> print(sol.y_events)
    [array([[-5.68434189e-14, -1.00000000e+01]]),
     array([[1.00000000e+02, 1.77635684e-15]])]


    # 打印在事件发生时刻的解向量列表 sol.y_events
    print(sol.y_events)



    >>> def apex(t, y): return y[1]
    >>> sol = solve_ivp(upward_cannon, [0, 100], [0, 10],
    ...                 events=(hit_ground, apex), dense_output=True)


    # 定义函数 apex 描述炮弹运动的最高点
    # 使用 solve_ivp 函数求解 upward_cannon 函数的数值解，时间范围为 [0, 100]
    # 初始条件为 [0, 10]，并且设置 hit_ground 和 apex 函数作为事件的终止条件
    # 同时设置 dense_output=True 以返回稠密解
    sol = solve_ivp(upward_cannon, [0, 100], [0, 10],
                    events=(hit_ground, apex), dense_output=True)



    >>> print(sol.t_events)
    [array([40.]), array([20.])]


    # 打印事件发生的时间 sol.t_events
    print(sol.t_events)



    >>> print(sol.t)
    [0.00000000e+00 9.99900010e-05 1.09989001e-03 1.10988901e-02
     1.11088891e-01 1.11098890e+00 1.11099890e+01 4.00000000e+01]


    # 打印求解结果中的时间点 sol.t
    print(sol.t)



    >>> print(sol.sol(sol.t_events[1][0]))
    [100.   0.]


    # 打印在事件发生时刻 sol.t_events[1][0] 的解向量 sol.sol(sol.t_events[1][0])
    print(sol.sol(sol.t_events[1][0]))



    >>> print(sol.y_events)
    [array([[-5.68434189e-14, -1.00000000e+01]]),
     array([[1.00000000e+02, 1.77635684e-15]])]


    # 打印在事件发生时刻的解向量列表 sol.y_events
    print(sol.y_events)



    As an example of a system with additional parameters, we'll implement
    the Lotka-Volterra equations [12]_.


    # 作为具有额外参数的系统的示例，我们将实现 Lotka-Volterra 方程 [12]_



    >>> def lotkavolterra(t, z, a, b, c, d):
    ...     x, y = z
    ...     return [a*x - b*x*y, -c*y + d*x*y]
    ...


    # 定义 Lotka-Volterra 方程的函数 lotkavolterra，接受参数 t, z, a, b, c, d
    # 其中 z 是状态向量，a, b, c, d 是额外的参数
    def lotkavolterra(t, z, a, b, c, d):
        x, y = z
        return [a*x - b*x*y, -c*y + d*x*y]



    We pass in the parameter values a=1.5, b=1, c=3 and d=1 with the `args`
    argument.


    # 使用参数 a=1.5, b=1, c=3, d=1 通过 `args` 参数传递给函数



    >>> sol = solve_ivp(lotkavolterra, [0, 15], [10, 5], args=(1.5, 1, 3, 1),
    ...                 dense_output=True)
    """
    equation ``y' = Ay`` with complex matrix ``A``.

    >>> A = np.array([[-0.25 + 0.14j, 0, 0.33 + 0.44j],
    ...               [0.25 + 0.58j, -0.2 + 0.14j, 0],
    ...               [0, 0.2 + 0.4j, -0.1 + 0.97j]])

    Solving an IVP with ``A`` from above and ``y`` as 3x1 vector:

    >>> def deriv_vec(t, y):
    ...     return A @ y
    >>> result = solve_ivp(deriv_vec, [0, 25],
    ...                    np.array([10 + 0j, 20 + 0j, 30 + 0j]),
    ...                    t_eval=np.linspace(0, 25, 101))
    >>> print(result.y[:, 0])
    [10.+0.j 20.+0.j 30.+0.j]
    >>> print(result.y[:, -1])
    [18.46291039+45.25653651j 10.01569306+36.23293216j
     -4.98662741+80.07360388j]

    Solving an IVP with ``A`` from above with ``y`` as 3x3 matrix :

    >>> def deriv_mat(t, y):
    ...     return (A @ y.reshape(3, 3)).flatten()
    >>> y0 = np.array([[2 + 0j, 3 + 0j, 4 + 0j],
    ...                [5 + 0j, 6 + 0j, 7 + 0j],
    ...                [9 + 0j, 34 + 0j, 78 + 0j]])

    >>> result = solve_ivp(deriv_mat, [0, 25], y0.flatten(),
    ...                    t_eval=np.linspace(0, 25, 101))
    >>> print(result.y[:, 0].reshape(3, 3))
    [[ 2.+0.j  3.+0.j  4.+0.j]
     [ 5.+0.j  6.+0.j  7.+0.j]
     [ 9.+0.j 34.+0.j 78.+0.j]]
    >>> print(result.y[:, -1].reshape(3, 3))
    [[  5.67451179 +12.07938445j  17.2888073  +31.03278837j
        37.83405768 +63.25138759j]
     [  3.39949503 +11.82123994j  21.32530996 +44.88668871j
        53.17531184+103.80400411j]
     [ -2.26105874 +22.19277664j -15.1255713  +70.19616341j
       -38.34616845+153.29039931j]]

    """
    # 检查方法是否在有效方法列表中，或者是否是 OdeSolver 类的子类
    if method not in METHODS and not (
            inspect.isclass(method) and issubclass(method, OdeSolver)):
        raise ValueError(f"`method` must be one of {METHODS} or OdeSolver class.")

    # 将时间区间转换为浮点数
    t0, tf = map(float, t_span)

    # 检查是否提供了额外参数 args
    if args is not None:
        # 尝试将 args 转换为列表，以检查是否可迭代
        try:
            _ = [*(args)]
        except TypeError as exp:
            # 如果无法转换，则给出错误提示
            suggestion_tuple = (
                "Supplied 'args' cannot be unpacked. Please supply `args`"
                f" as a tuple (e.g. `args=({args},)`)"
            )
            raise TypeError(suggestion_tuple) from exp

        # 创建一个新的函数 fun，将原始的 fun 和 args 封装在其中
        def fun(t, x, fun=fun):
            return fun(t, x, *args)
        
        # 获取选项中的雅可比矩阵函数 jac
        jac = options.get('jac')
        # 如果 jac 是可调用的，则创建一个新的 lambda 函数，将原始的 jac 和 args 封装在其中
        if callable(jac):
            options['jac'] = lambda t, x: jac(t, x, *args)
    # 如果 t_eval 不是 None，则将其转换为 NumPy 数组
    if t_eval is not None:
        t_eval = np.asarray(t_eval)
        # 检查 t_eval 是否是一维数组，如果不是则抛出 ValueError 异常
        if t_eval.ndim != 1:
            raise ValueError("`t_eval` must be 1-dimensional.")

        # 检查 t_eval 中的值是否都在 t_span 的范围内
        if np.any(t_eval < min(t0, tf)) or np.any(t_eval > max(t0, tf)):
            raise ValueError("Values in `t_eval` are not within `t_span`.")

        # 计算 t_eval 中相邻值之间的差分
        d = np.diff(t_eval)
        # 如果 t0 < tf 并且存在非正差分，则抛出 ValueError 异常；或者如果 t0 > tf 并且存在非负差分，则同样抛出异常
        if tf > t0 and np.any(d <= 0) or tf < t0 and np.any(d >= 0):
            raise ValueError("Values in `t_eval` are not properly sorted.")

        # 如果 tf > t0，则将 t_eval_i 初始化为 0；否则，反转 t_eval 以便降序排列，并将 t_eval_i 设置为数组长度
        if tf > t0:
            t_eval_i = 0
        else:
            # 将 t_eval 反转以降序排列，为了使用 np.searchsorted
            t_eval = t_eval[::-1]
            # t_eval_i 作为切片的上界
            t_eval_i = t_eval.shape[0]

    # 如果 method 在 METHODS 中，则将其替换为对应的函数
    if method in METHODS:
        method = METHODS[method]

    # 使用选定的求解方法初始化 solver 对象，并传入必要的参数
    solver = method(fun, t0, y0, tf, vectorized=vectorized, **options)

    # 根据 t_eval 和 dense_output 的情况初始化 ts 和 ys 列表
    if t_eval is None:
        ts = [t0]
        ys = [y0]
    elif t_eval is not None and dense_output:
        ts = []
        ti = [t0]
        ys = []
    else:
        ts = []
        ys = []

    # 初始化 interpolants 列表用于存储插值函数
    interpolants = []

    # 如果 events 不是 None，则准备事件函数并初始化相关变量
    if events is not None:
        events, max_events, event_dir = prepare_events(events)
        # 初始化 event_count 数组，长度为 events 的个数，用于记录事件触发次数
        event_count = np.zeros(len(events))
        if args is not None:
            # 将用户定义的事件函数用 lambda 包装起来，以隐藏额外的参数
            # 原始的事件函数作为关键字参数传递给 lambda 函数，以保持原始函数在作用域内
            events = [lambda t, x, event=event: event(t, x, *args)
                      for event in events]
        # 计算初始时刻的事件函数值并存储在 g 列表中
        g = [event(t0, y0) for event in events]
        # 初始化 t_events 和 y_events 列表，用于存储事件发生时刻和对应的状态
        t_events = [[] for _ in range(len(events))]
        y_events = [[] for _ in range(len(events))]
    else:
        # 如果 events 是 None，则将 t_events 和 y_events 设置为 None
        t_events = None
        y_events = None

    # 初始化状态变量为 None
    status = None
    # 当状态为 None 时，继续求解微分方程直到状态确定
    while status is None:
        # 调用求解器的一步求解方法，并获取返回的消息
        message = solver.step()

        # 根据求解器的状态更新整体状态
        if solver.status == 'finished':
            status = 0  # 求解成功，状态设为 0
        elif solver.status == 'failed':
            status = -1  # 求解失败，状态设为 -1 并退出循环
            break

        # 保存当前的时间、状态和解
        t_old = solver.t_old
        t = solver.t
        y = solver.y

        # 如果需要稠密输出
        if dense_output:
            # 获取稠密输出的解，并添加到插值器列表中
            sol = solver.dense_output()
            interpolants.append(sol)
        else:
            sol = None

        # 如果定义了事件
        if events is not None:
            # 计算当前时间下所有事件的状态
            g_new = [event(t, y) for event in events]
            # 找到活跃事件
            active_events = find_active_events(g, g_new, event_dir)
            # 如果有活跃事件
            if active_events.size > 0:
                if sol is None:
                    sol = solver.dense_output()

                # 更新事件计数
                event_count[active_events] += 1
                # 处理事件
                root_indices, roots, terminate = handle_events(
                    sol, events, active_events, event_count, max_events,
                    t_old, t)

                # 将事件的时间和对应的解添加到事件列表中
                for e, te in zip(root_indices, roots):
                    t_events[e].append(te)
                    y_events[e].append(sol(te))

                # 如果需要终止求解
                if terminate:
                    status = 1  # 状态设为 1，表示终止
                    t = roots[-1]  # 更新时间
                    y = sol(t)  # 更新解

            # 更新事件状态
            g = g_new

        # 如果没有指定 t_eval，则将当前时间 t 和对应的解 y 添加到列表中
        if t_eval is None:
            ts.append(t)
            ys.append(y)
        else:
            # 如果指定了 t_eval
            if solver.direction > 0:
                t_eval_i_new = np.searchsorted(t_eval, t, side='right')
                t_eval_step = t_eval[t_eval_i:t_eval_i_new]
            else:
                t_eval_i_new = np.searchsorted(t_eval, t, side='left')
                # 需要进行两次切片操作，因为无法使用反向切片包含第 0 个元素。
                t_eval_step = t_eval[t_eval_i_new:t_eval_i][::-1]

            # 如果存在需要评估的时间步长
            if t_eval_step.size > 0:
                if sol is None:
                    sol = solver.dense_output()
                ts.append(t_eval_step)
                ys.append(sol(t_eval_step))
                t_eval_i = t_eval_i_new  # 更新 t_eval 索引

        # 如果指定了 t_eval 并且需要稠密输出
        if t_eval is not None and dense_output:
            ti.append(t)  # 将当前时间添加到 ti 列表中

    # 根据状态获取相应的消息
    message = MESSAGES.get(status, message)

    # 如果存在事件时间列表，则转换为 numpy 数组
    if t_events is not None:
        t_events = [np.asarray(te) for te in t_events]
        y_events = [np.asarray(ye) for ye in y_events]

    # 如果未指定 t_eval，则将 ts 和 ys 转换为 numpy 数组并堆叠
    if t_eval is None:
        ts = np.array(ts)
        ys = np.vstack(ys).T
    elif ts:  # 如果 ts 列表非空
        ts = np.hstack(ts)
        ys = np.hstack(ys)

    # 如果需要稠密输出
    if dense_output:
        if t_eval is None:
            # 根据方法选择是否使用备用段
            sol = OdeSolution(
                ts, interpolants, alt_segment=True if method in [BDF, LSODA] else False
            )
        else:
            # 根据方法选择是否使用备用段
            sol = OdeSolution(
                ti, interpolants, alt_segment=True if method in [BDF, LSODA] else False
            )
    else:
        sol = None
    # 返回ODE求解器的结果对象，包括时间点(t)、解向量(y)、解对象(sol)、事件发生时间(t_events)、事件发生处解向量(y_events)、
    # 求解器函数调用次数(nfev)、雅可比矩阵函数调用次数(njev)、线性代数函数调用次数(nlu)、
    # 求解器的状态(status)、消息(message)以及成功标志(success)，成功标志根据状态(status)是否大于等于0判断。
    return OdeResult(t=ts, y=ys, sol=sol, t_events=t_events, y_events=y_events,
                     nfev=solver.nfev, njev=solver.njev, nlu=solver.nlu,
                     status=status, message=message, success=status >= 0)
```