# `D:\src\scipysrc\scipy\scipy\integrate\_ivp\lsoda.py`

```
# 导入必要的库
import numpy as np
from scipy.integrate import ode
# 导入所需的本地模块和函数
from .common import validate_tol, validate_first_step, warn_extraneous
from .base import OdeSolver, DenseOutput

# 定义 LSODA 类，继承自 OdeSolver 类
class LSODA(OdeSolver):
    """Adams/BDF method with automatic stiffness detection and switching.

    This is a wrapper to the Fortran solver from ODEPACK [1]_. It switches
    automatically between the nonstiff Adams method and the stiff BDF method.
    The method was originally detailed in [2]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system: the time derivative of the state ``y``
        at time ``t``. The calling signature is ``fun(t, y)``, where ``t`` is a
        scalar and ``y`` is an ndarray with ``len(y) = len(y0)``. ``fun`` must
        return an array of the same shape as ``y``. See `vectorized` for more
        information.
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    first_step : float or None, optional
        Initial step size. Default is ``None`` which means that the algorithm
        should choose.
    min_step : float, optional
        Minimum allowed step size. Default is 0.0, i.e., the step size is not
        bounded and determined solely by the solver.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
        relative accuracy (number of correct digits), while `atol` controls
        absolute accuracy (number of correct decimal places). To achieve the
        desired `rtol`, set `atol` to be smaller than the smallest value that
        can be expected from ``rtol * abs(y)`` so that `rtol` dominates the
        allowable error. If `atol` is larger than ``rtol * abs(y)`` the
        number of correct digits is not guaranteed. Conversely, to achieve the
        desired `atol` set `rtol` such that ``rtol * abs(y)`` is always smaller
        than `atol`. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    """
    # LSODA 类的初始化方法
    def __init__(self, fun, t0, y0, t_bound, first_step=None, min_step=0.0,
                 max_step=np.inf, rtol=1e-3, atol=1e-6):
        # 调用父类 OdeSolver 的初始化方法
        super().__init__(fun, t0, y0, t_bound, first_step, min_step, max_step,
                         validate_tol(rtol, atol)[0])
        # 验证并设置首步长
        self.first_step = validate_first_step(first_step)
        # 设置相对误差和绝对误差
        self.rtol, self.atol = validate_tol(rtol, atol)

    # 警告多余参数的函数装饰器
    def warn_extraneous(self):
        return warn_extraneous()
    jac : None or callable, optional
        # 可选参数，表示系统右侧的雅可比矩阵对于 y 的导数。
        # 雅可比矩阵的形状为 (n, n)，其元素 (i, j) 等于 d f_i / d y_j。
        # 函数将被调用为 jac(t, y)。如果为 None（默认），将通过有限差分来近似雅可比矩阵。
        # 通常建议提供雅可比矩阵，而不是依赖有限差分的近似。
        
    lband, uband : int or None
        # 定义雅可比矩阵的带宽参数。
        # 即 jac[i, j] != 0 仅当 i - lband <= j <= i + uband。
        # 设置这些需要你的 jac 函数以紧凑格式返回雅可比矩阵：
        # 返回的数组必须有 n 列和 uband + lband + 1 行，在这些行中雅可比矩阵的对角线被写入。
        # 具体来说，jac_packed[uband + i - j, j] = jac[i, j]。
        # 相同的格式也用于 `scipy.linalg.solve_banded` 中。
        # 这些参数也可以与 jac=None 一起使用，以减少有限差分估计的雅可比元素数量。

    vectorized : bool, optional
        # 表示 `fun` 是否可以以向量化的方式调用。
        # False（默认）推荐用于此求解器。
        
        # 如果 vectorized=False，`fun` 将始终以形状为 (n,) 的 `y` 被调用，其中 `n = len(y0)`。
        
        # 如果 vectorized=True，`fun` 可能以形状为 (n, k) 的 `y` 被调用，其中 `k` 是整数。
        # 在这种情况下，`fun` 必须表现为 `fun(t, y)[:, i] == fun(t, y[:, i])`（即返回数组的每一列是对应 `y` 列的状态的时间导数）。
        
        # 设置 `vectorized=True` 允许通过 'Radau' 和 'BDF' 方法更快地进行雅可比矩阵的有限差分近似，
        # 但会导致该求解器的执行速度变慢。

    Attributes
    ----------
    n : int
        # 方程的数量。

    status : string
        # 求解器的当前状态：'running'，'finished' 或 'failed'。

    t_bound : float
        # 边界时间。

    direction : float
        # 积分的方向：+1 或 -1。

    t : float
        # 当前时间。

    y : ndarray
        # 当前状态。

    t_old : float
        # 上一个时间。如果尚未执行任何步骤，则为 None。

    nfev : int
        # 右侧函数的评估次数。

    njev : int
        # 雅可比矩阵的评估次数。

References
----------
.. [1] A. C. Hindmarsh, "ODEPACK, A Systematized Collection of ODE
       Solvers," IMACS Transactions on Scientific Computation, Vol 1.,
       pp. 55-64, 1983.
    """
    根据 Petzold 的论文 [2]，实现了一个基于 LSODA 方法的常微分方程求解器。

    初始化方法，设置求解器的各项参数和初始条件。
    """
    def __init__(self, fun, t0, y0, t_bound, first_step=None, min_step=0.0,
                 max_step=np.inf, rtol=1e-3, atol=1e-6, jac=None, lband=None,
                 uband=None, vectorized=False, **extraneous):
        warn_extraneous(extraneous)  # 警告处理额外的参数，可能是不必要的输入
        super().__init__(fun, t0, y0, t_bound, vectorized)  # 调用父类的初始化方法

        if first_step is None:
            first_step = 0  # 如果未提供首步长参数，则使用 LSODA 方法的默认值
        else:
            first_step = validate_first_step(first_step, t0, t_bound)  # 验证首步长的合法性

        first_step *= self.direction  # 考虑方向性（正向或反向）

        if max_step == np.inf:
            max_step = 0  # 如果最大步长设置为无穷大，则使用 LSODA 方法的默认值
        elif max_step <= 0:
            raise ValueError("`max_step` must be positive.")  # 最大步长必须为正数

        if min_step < 0:
            raise ValueError("`min_step` must be nonnegative.")  # 最小步长必须为非负数

        rtol, atol = validate_tol(rtol, atol, self.n)  # 验证相对误差和绝对误差的合法性

        solver = ode(self.fun, jac)  # 创建一个 ODE 求解器对象
        solver.set_integrator('lsoda', rtol=rtol, atol=atol, max_step=max_step,
                              min_step=min_step, first_step=first_step,
                              lband=lband, uband=uband)  # 配置 LSODA 求解器的参数
        solver.set_initial_value(y0, t0)  # 设置初始值

        # 将 t_bound 注入到 rwork 数组中，用于 itask=5 的情况
        solver._integrator.rwork[0] = self.t_bound
        solver._integrator.call_args[4] = solver._integrator.rwork

        self._lsoda_solver = solver  # 保存 LSODA 求解器对象的引用

    def _step_impl(self):
        solver = self._lsoda_solver
        integrator = solver._integrator

        # 根据 LSODA 的文档，itask=5 表示单步积分，并且不超过 t_bound
        itask = integrator.call_args[2]
        integrator.call_args[2] = 5  # 设置积分任务为 itask=5
        solver._y, solver.t = integrator.run(
            solver.f, solver.jac or (lambda: None), solver._y, solver.t,
            self.t_bound, solver.f_params, solver.jac_params)  # 运行积分过程
        integrator.call_args[2] = itask  # 恢复原始的积分任务设置

        if solver.successful():
            self.t = solver.t  # 更新当前时间
            self.y = solver._y  # 更新当前状态
            # 根据 LSODA Fortran 源码，njev 等于 nlu
            self.njev = integrator.iwork[12]  # 记录用于雅可比矩阵评估的函数调用次数
            self.nlu = integrator.iwork[12]  # 记录线性求解器调用的次数
            return True, None  # 返回成功标志和空的错误信息
        else:
            return False, 'Unexpected istate in LSODA.'  # 返回失败标志和错误信息
    # 获取 lsoda_solver 对象中的整数工作数组 iwork
    iwork = self._lsoda_solver._integrator.iwork
    # 获取 lsoda_solver 对象中的实数工作数组 rwork
    rwork = self._lsoda_solver._integrator.rwork

    # 我们需要生成 Nordsieck 历史数组 yh，直到最后一次成功迭代使用的阶数。
    # 步长大小并不重要，因为它将在 LsodaDenseOutput 中进行缩放。
    # 由于 ODEPACK 的 LSODA 实现会生成下一个迭代所需的 Nordsieck 历史状态，因此可能需要做一些额外工作。

    # iwork[13] 包含最后一次成功迭代使用的阶数，而 iwork[14] 包含下一次尝试的阶数。
    order = iwork[13]

    # rwork[11] 包含下一次尝试的步长大小，而 rwork[10] 包含最后一次成功迭代的步长大小。
    h = rwork[11]

    # rwork[20:20 + (iwork[14] + 1) * self.n] 包含 Nordsieck 数组的条目，
    # 这些条目是下一次迭代所需的状态。我们希望获取直到最后一个成功步长的阶数的条目，因此使用以下方法。
    yh = np.reshape(rwork[20:20 + (order + 1) * self.n],
                    (self.n, order + 1), order='F').copy()
    if iwork[14] < order:
        # 如果阶数被设置为减小，那么 yh 的最后一列在 ODEPACK 的 LSODA 实现中未被更新，
        # 因为这一列在下一次迭代中不会被使用。我们必须重新缩放这一列，使关联的步长与其他列保持一致。
        yh[:, -1] *= (h / rwork[10]) ** order

    # 返回一个 LsodaDenseOutput 对象，用于提供稠密输出
    return LsodaDenseOutput(self.t_old, self.t, h, order, yh)
class LsodaDenseOutput(DenseOutput):
    # 继承自 DenseOutput 类的 LsodaDenseOutput 类，用于提供稠密输出
    def __init__(self, t_old, t, h, order, yh):
        # 初始化方法，设置初始时间 t_old，当前时间 t，步长 h，阶数 order 和预测值 yh
        super().__init__(t_old, t)
        # 调用父类 DenseOutput 的初始化方法
        self.h = h
        # 设置步长 h
        self.yh = yh
        # 设置预测值 yh
        self.p = np.arange(order + 1)
        # 创建一个包含 0 到 order 的数组，用于计算多项式的次幂

    def _call_impl(self, t):
        # 私有方法 _call_impl，根据时间 t 计算插值多项式在 t 处的值
        if t.ndim == 0:
            # 如果 t 是标量
            x = ((t - self.t) / self.h) ** self.p
            # 计算 (t - self.t) / self.h 的 self.p 次幂
        else:
            # 如果 t 是数组
            x = ((t - self.t) / self.h) ** self.p[:, None]
            # 计算 (t - self.t) / self.h 的 self.p 次幂（每个多项式次幂作用在 t 的每个元素上）

        return np.dot(self.yh, x)
        # 返回预测值 yh 与 x 的点乘结果，即插值多项式在 t 处的值
```