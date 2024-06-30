# `D:\src\scipysrc\scipy\scipy\integrate\_ivp\rk.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
from .base import OdeSolver, DenseOutput  # 从当前包中导入 OdeSolver 和 DenseOutput 类
from .common import (validate_max_step, validate_tol, select_initial_step,
                     norm, warn_extraneous, validate_first_step)  # 从当前包中导入多个函数
from . import dop853_coefficients  # 从当前包中导入 dop853_coefficients 模块

SAFETY = 0.9  # 安全系数，用于根据误差的渐近行为计算步长

MIN_FACTOR = 0.2  # 最小允许的步长减小比例
MAX_FACTOR = 10  # 最大允许的步长增大比例


def rk_step(fun, t, y, f, h, A, B, C, K):
    """Perform a single Runge-Kutta step.

    This function computes a prediction of an explicit Runge-Kutta method and
    also estimates the error of a less accurate method.

    Notation for Butcher tableau is as in [1]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t : float
        Current time.
    y : ndarray, shape (n,)
        Current state.
    f : ndarray, shape (n,)
        Current value of the derivative, i.e., ``fun(x, y)``.
    h : float
        Step to use.
    A : ndarray, shape (n_stages, n_stages)
        Coefficients for combining previous RK stages to compute the next
        stage. For explicit methods the coefficients at and above the main
        diagonal are zeros.
    B : ndarray, shape (n_stages,)
        Coefficients for combining RK stages for computing the final
        prediction.
    C : ndarray, shape (n_stages,)
        Coefficients for incrementing time for consecutive RK stages.
        The value for the first stage is always zero.
    K : ndarray, shape (n_stages + 1, n)
        Storage array for putting RK stages here. Stages are stored in rows.
        The last row is a linear combination of the previous rows with
        coefficients

    Returns
    -------
    y_new : ndarray, shape (n,)
        Solution at t + h computed with a higher accuracy.
    f_new : ndarray, shape (n,)
        Derivative ``fun(t + h, y_new)``.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """
    K[0] = f  # 将当前导数值存储在 K 的第一行

    # 计算每个 RK 阶段的预测值
    for s, (a, c) in enumerate(zip(A[1:], C[1:]), start=1):
        dy = np.dot(K[:s].T, a[:s]) * h  # 计算对当前步长的预测偏差
        K[s] = fun(t + c * h, y + dy)  # 计算下一个 RK 阶段的预测值

    y_new = y + h * np.dot(K[:-1].T, B)  # 计算高精度解
    f_new = fun(t + h, y_new)  # 计算高精度解的导数

    K[-1] = f_new  # 将高精度解的导数存储在 K 的最后一行

    return y_new, f_new  # 返回高精度解和其导数


class RungeKutta(OdeSolver):
    """Base class for explicit Runge-Kutta methods."""
    C: np.ndarray = NotImplemented  # RK 方法的时间增量系数数组
    A: np.ndarray = NotImplemented  # RK 方法的阶段组合系数数组
    B: np.ndarray = NotImplemented  # RK 方法的预测系数数组
    E: np.ndarray = NotImplemented  # RK 方法的误差估计数组
    P: np.ndarray = NotImplemented  # RK 方法的阶数估计数组
    order: int = NotImplemented  # RK 方法的阶数
    error_estimator_order: int = NotImplemented  # RK 方法的误差估计阶数
    n_stages: int = NotImplemented  # RK 方法的阶段数
    def __init__(self, fun, t0, y0, t_bound, max_step=np.inf,
                 rtol=1e-3, atol=1e-6, vectorized=False,
                 first_step=None, **extraneous):
        warn_extraneous(extraneous)
        # 调用超类的初始化方法，传入函数、初始时间、初始状态、时间上限、向量化标志等参数
        super().__init__(fun, t0, y0, t_bound, vectorized,
                         support_complex=True)
        # 初始化旧状态为 None
        self.y_old = None
        # 验证和设置最大步长
        self.max_step = validate_max_step(max_step)
        # 验证和设置相对误差和绝对误差容忍度
        self.rtol, self.atol = validate_tol(rtol, atol, self.n)
        # 计算初始函数值
        self.f = self.fun(self.t, self.y)
        # 选择初始步长
        if first_step is None:
            self.h_abs = select_initial_step(
                self.fun, self.t, self.y, t_bound, max_step, self.f, self.direction,
                self.error_estimator_order, self.rtol, self.atol)
        else:
            self.h_abs = validate_first_step(first_step, t0, t_bound)
        # 初始化 K 矩阵
        self.K = np.empty((self.n_stages + 1, self.n), dtype=self.y.dtype)
        # 计算错误估计的指数
        self.error_exponent = -1 / (self.error_estimator_order + 1)
        # 初始化前一步的步长为 None
        self.h_previous = None

    def _estimate_error(self, K, h):
        # 估计误差
        return np.dot(K.T, self.E) * h

    def _estimate_error_norm(self, K, h, scale):
        # 估计误差的范数
        return norm(self._estimate_error(K, h) / scale)

    def _step_impl(self):
        t = self.t
        y = self.y

        max_step = self.max_step
        rtol = self.rtol
        atol = self.atol

        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)

        # 根据步长绝对值确定 h_abs
        if self.h_abs > max_step:
            h_abs = max_step
        elif self.h_abs < min_step:
            h_abs = min_step
        else:
            h_abs = self.h_abs

        step_accepted = False
        step_rejected = False

        # 进行步进过程，直至接受步长
        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            # 调整 t_new，以确保不超过时间边界
            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t
            h_abs = np.abs(h)

            # 执行 Runge-Kutta 步进，计算新状态 y_new 和新函数值 f_new
            y_new, f_new = rk_step(self.fun, t, y, self.f, h, self.A,
                                   self.B, self.C, self.K)
            # 计算误差的容许范围
            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
            error_norm = self._estimate_error_norm(self.K, h, scale)

            # 根据误差范数调整步长
            if error_norm < 1:
                if error_norm == 0:
                    factor = MAX_FACTOR
                else:
                    factor = min(MAX_FACTOR,
                                 SAFETY * error_norm ** self.error_exponent)

                if step_rejected:
                    factor = min(1, factor)

                h_abs *= factor

                step_accepted = True
            else:
                h_abs *= max(MIN_FACTOR,
                             SAFETY * error_norm ** self.error_exponent)
                step_rejected = True

        # 更新前一步信息和状态信息
        self.h_previous = h
        self.y_old = y

        self.t = t_new
        self.y = y_new

        self.h_abs = h_abs
        self.f = f_new

        return True, None
    # 实现稠密输出的内部方法，计算状态矩阵 K 的转置与 P 的乘积
    def _dense_output_impl(self):
        Q = self.K.T.dot(self.P)
        # 创建一个 RkDenseOutput 对象，使用旧时间 t_old、新时间 t、旧状态 y_old 和计算得到的 Q
        return RkDenseOutput(self.t_old, self.t, self.y_old, Q)
# 定义一个名为 RK23 的类，继承自 RungeKutta 类
class RK23(RungeKutta):
    """Explicit Runge-Kutta method of order 3(2).

    这是一个三阶(2阶)的显式 Runge-Kutta 方法。

    This uses the Bogacki-Shampine pair of formulas [1]_. The error is controlled
    assuming accuracy of the second-order method, but steps are taken using the
    third-order accurate formula (local extrapolation is done). A cubic Hermite
    polynomial is used for the dense output.

    使用 Bogacki-Shampine 的一对公式 [1]_。误差控制假设二阶方法的精度，但采用三阶精确公式进行步骤
    (进行局部外推)。用于稠密输出的是三次 Hermite 多项式。

    Can be applied in the complex domain.

    可以应用于复数域。

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

    直接 Runge-Kutta 方法的三阶(2阶)显式实现类。

    """
    order = 3
    # 指定 Runge-Kutta 方法的阶数为 3

    error_estimator_order = 2
    # 指定误差估计器的阶数为 2

    n_stages = 3
    # 指定 Runge-Kutta 方法的阶段数为 3

    C = np.array([0, 1/2, 3/4])
    # 设置时间步长系数向量 C，用于计算下一个步长的时间

    A = np.array([
        [0, 0, 0],
        [1/2, 0, 0],
        [0, 3/4, 0]
    ])
    # 设置 Runge-Kutta 方法的系数矩阵 A，定义了步长和时间步的权重

    B = np.array([2/9, 1/3, 4/9])
    # 设置权重向量 B，用于计算当前步长的状态

    E = np.array([5/72, -1/12, -1/9, 1/8])
    # 设置误差估计向量 E，用于计算步长的误差估计

    P = np.array([[1, -4 / 3, 5 / 9],
                  [0, 1, -2/3],
                  [0, 4/3, -8/9],
                  [0, -1, 1]])
    # 设置增强矩阵 P，用于加强步长和时间步之间的关系
# 定义一个继承自 RungeKutta 类的 RK45 类
class RK45(RungeKutta):
    """Explicit Runge-Kutta method of order 5(4).

    This uses the Dormand-Prince pair of formulas [1]_. The error is controlled
    assuming accuracy of the fourth-order method accuracy, but steps are taken
    using the fifth-order accurate formula (local extrapolation is done).
    A quartic interpolation polynomial is used for the dense output [2]_.

    Can be applied in the complex domain.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(t, y)``.
        Here ``t`` is a scalar, and there are two options for the ndarray ``y``:
        It can either have shape (n,); then ``fun`` must return array_like with
        shape (n,). Alternatively it can have shape (n, k); then ``fun``
        must return an array_like with shape (n, k), i.e., each column
        corresponds to a single column in ``y``. The choice between the two
        options is determined by `vectorized` argument (see below).
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
    vectorized : bool, optional
        Whether `fun` is implemented in a vectorized fashion. Default is False.

    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    """
    y : ndarray
        当前的状态向量。
    t_old : float
        上一个时间步的时间。如果尚未进行任何步骤，则为None。
    step_size : float
        最后一次成功步长的大小。如果尚未进行任何步骤，则为None。
    nfev : int
        系统右侧函数的评估次数。
    njev : int
        雅可比矩阵的评估次数。对于此求解器，始终为0，因为不使用雅可比矩阵。
    nlu : int
        LU分解的次数。对于此求解器，始终为0。

    References
    ----------
    .. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
           formulae", Journal of Computational and Applied Mathematics, Vol. 6,
           No. 1, pp. 19-26, 1980.
    .. [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
           of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.
    """
    order = 5
    error_estimator_order = 4
    n_stages = 6
    C = np.array([0, 1/5, 3/10, 4/5, 8/9, 1])
    A = np.array([
        [0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
    ])
    B = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    E = np.array([-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525,
                  1/40])
    # Corresponds to the optimum value of c_6 from [2]_.
    P = np.array([
        [1, -8048581381/2820520608, 8663915743/2820520608,
         -12715105075/11282082432],
        [0, 0, 0, 0],
        [0, 131558114200/32700410799, -68118460800/10900136933,
         87487479700/32700410799],
        [0, -1754552775/470086768, 14199869525/1410260304,
         -10690763975/1880347072],
        [0, 127303824393/49829197408, -318862633887/49829197408,
         701980252875 / 199316789632],
        [0, -282668133/205662961, 2019193451/616988883, -1453857185/822651844],
        [0, 40617522/29380423, -110615467/29380423, 69997945/29380423]])
class DOP853(RungeKutta):
    """Explicit Runge-Kutta method of order 8.

    This is a Python implementation of "DOP853" algorithm originally written
    in Fortran [1]_, [2]_. Note that this is not a literal translation, but
    the algorithmic core and coefficients are the same.

    Can be applied in the complex domain.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(t, y)``.
        Here, ``t`` is a scalar, and there are two options for the ndarray ``y``:
        It can either have shape (n,); then ``fun`` must return array_like with
        shape (n,). Alternatively it can have shape (n, k); then ``fun``
        must return an array_like with shape (n, k), i.e. each column
        corresponds to a single column in ``y``. The choice between the two
        options is determined by `vectorized` argument (see below).
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
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e. the step size is not
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
    vectorized : bool, optional
        Whether `fun` is implemented in a vectorized fashion. Default is False.

    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    # 步长大小，用于记录最后一次成功步长的大小。如果还没有进行过步长操作，则为None。
    step_size : float
    # 系统右手边函数的评估次数。
    nfev : int
    # 雅可比矩阵的评估次数。对于这个求解器，始终为0，因为它不使用雅可比矩阵。
    njev : int
    # LU分解的次数。对于这个求解器，始终为0。
    nlu : int

    References
    ----------
    # Hairer等人的参考文献，详细描述了DOP853算法的应用于非刚性问题。
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.
    # 原始Fortran代码页面，包含DOP853算法的实现。
    .. [2] `Page with original Fortran code of DOP853
            <http://www.unige.ch/~hairer/software.html>`_.
    """
    # DOP853算法的阶数
    n_stages = dop853_coefficients.N_STAGES
    # 方法的阶数
    order = 8
    # 误差估计器的阶数
    error_estimator_order = 7
    # 系数矩阵A
    A = dop853_coefficients.A[:n_stages, :n_stages]
    # 系数矩阵B
    B = dop853_coefficients.B
    # 系数向量C
    C = dop853_coefficients.C[:n_stages]
    # E3向量
    E3 = dop853_coefficients.E3
    # E5向量
    E5 = dop853_coefficients.E5
    # 系数D
    D = dop853_coefficients.D

    # 额外的系数矩阵A_EXTRA
    A_EXTRA = dop853_coefficients.A[n_stages + 1:]
    # 额外的系数向量C_EXTRA
    C_EXTRA = dop853_coefficients.C[n_stages + 1:]

    # 初始化函数，设置ODESolver基类的参数，包括系统函数fun、初始时间t0、初始状态y0、
    # 最终时间t_bound、最大步长max_step、相对误差容限rtol、绝对误差容限atol、
    # 是否向量化vectorized、第一步大小first_step等。
    def __init__(self, fun, t0, y0, t_bound, max_step=np.inf,
                 rtol=1e-3, atol=1e-6, vectorized=False,
                 first_step=None, **extraneous):
        # 调用父类ODESolver的初始化方法
        super().__init__(fun, t0, y0, t_bound, max_step, rtol, atol,
                         vectorized, first_step, **extraneous)
        # 初始化K_extended数组，扩展的K数组大小为N_STAGES_EXTENDED x self.n
        self.K_extended = np.empty((dop853_coefficients.N_STAGES_EXTENDED,
                                    self.n), dtype=self.y.dtype)
        # 初始化K数组，大小为self.n_stages + 1 x self.n
        self.K = self.K_extended[:self.n_stages + 1]

    # 用于估计误差的函数，参数包括K（当前步长的向量）、步长h。
    def _estimate_error(self, K, h):  # Left for testing purposes.
        # 计算误差估计值err5
        err5 = np.dot(K.T, self.E5)
        # 计算误差估计值err3
        err3 = np.dot(K.T, self.E3)
        # 计算修正因子的分母
        denom = np.hypot(np.abs(err5), 0.1 * np.abs(err3))
        # 初始化修正因子为全1数组
        correction_factor = np.ones_like(err5)
        # 根据条件修正修正因子
        mask = denom > 0
        correction_factor[mask] = np.abs(err5[mask]) / denom[mask]
        # 返回估计的误差值
        return h * err5 * correction_factor

    # 用于估计误差的函数，返回误差的范数，参数包括K（当前步长的向量）、步长h和缩放比例scale。
    def _estimate_error_norm(self, K, h, scale):
        # 计算误差估计值err5
        err5 = np.dot(K.T, self.E5) / scale
        # 计算误差估计值err3
        err3 = np.dot(K.T, self.E3) / scale
        # 计算误差估计值的范数平方
        err5_norm_2 = np.linalg.norm(err5)**2
        err3_norm_2 = np.linalg.norm(err3)**2
        # 如果两个范数都为0，则返回0.0
        if err5_norm_2 == 0 and err3_norm_2 == 0:
            return 0.0
        # 计算误差估计的分母
        denom = err5_norm_2 + 0.01 * err3_norm_2
        # 返回误差估计的范数值
        return np.abs(h) * err5_norm_2 / np.sqrt(denom * len(scale))
    # 计算稠密输出的实现函数 `_dense_output_impl`
    def _dense_output_impl(self):
        # 使用扩展后的 K 值
        K = self.K_extended
        # 前一步长 h 的值
        h = self.h_previous
        # 遍历额外的 A_EXTRA 和 C_EXTRA 元组，从 self.n_stages + 1 开始计数
        for s, (a, c) in enumerate(zip(self.A_EXTRA, self.C_EXTRA),
                                   start=self.n_stages + 1):
            # 计算 dy，是 K[:s].T 和 a[:s] 的乘积再乘以 h
            dy = np.dot(K[:s].T, a[:s]) * h
            # 计算下一个 K[s] 的值，使用 self.fun 函数
            K[s] = self.fun(self.t_old + c * h, self.y_old + dy)

        # 初始化一个空的 F 矩阵，形状为 (dop853_coefficients.INTERPOLATOR_POWER, self.n)
        F = np.empty((dop853_coefficients.INTERPOLATOR_POWER, self.n),
                     dtype=self.y_old.dtype)

        # 记录当前的 f_old
        f_old = K[0]
        # 计算 delta_y，即当前状态变量 y 与上一个状态变量 y_old 的差
        delta_y = self.y - self.y_old

        # 计算 F 的各行值
        F[0] = delta_y
        F[1] = h * f_old - delta_y
        F[2] = 2 * delta_y - h * (self.f + f_old)
        F[3:] = h * np.dot(self.D, K)

        # 返回一个 Dop853DenseOutput 对象，其中包含稠密输出所需的参数
        return Dop853DenseOutput(self.t_old, self.t, self.y_old, F)
# 定义一个继承自 DenseOutput 的 RkDenseOutput 类
class RkDenseOutput(DenseOutput):
    # 初始化方法，接受旧时间 t_old，新时间 t，旧状态 y_old 和 Q 矩阵
    def __init__(self, t_old, t, y_old, Q):
        # 调用父类 DenseOutput 的初始化方法，传入 t_old 和 t
        super().__init__(t_old, t)
        # 计算时间步长 h
        self.h = t - t_old
        # 设置矩阵 Q
        self.Q = Q
        # 确定阶数，从矩阵 Q 的列数减去 1
        self.order = Q.shape[1] - 1
        # 保存旧状态 y_old
        self.y_old = y_old

    # 实现 DenseOutput 中的 _call_impl 方法，根据时间 t 计算输出
    def _call_impl(self, t):
        # 计算归一化时间变量 x
        x = (t - self.t_old) / self.h
        # 如果时间 t 是标量
        if t.ndim == 0:
            # 复制 x，使其长度为阶数加 1，然后计算累积乘积
            p = np.tile(x, self.order + 1)
            p = np.cumprod(p)
        else:
            # 如果时间 t 是数组，复制 x 到一个形状为 (阶数加 1, 数组长度) 的数组，然后按列计算累积乘积
            p = np.tile(x, (self.order + 1, 1))
            p = np.cumprod(p, axis=0)
        # 计算输出 y，使用 Q 矩阵乘以 p，乘积再乘以时间步长 h
        y = self.h * np.dot(self.Q, p)
        # 如果输出 y 是二维数组，加上 y_old 的转置
        if y.ndim == 2:
            y += self.y_old[:, None]
        else:
            # 如果输出 y 是一维数组，加上 y_old
            y += self.y_old

        return y


# 定义一个继承自 DenseOutput 的 Dop853DenseOutput 类
class Dop853DenseOutput(DenseOutput):
    # 初始化方法，接受旧时间 t_old，新时间 t，旧状态 y_old 和 F 列表
    def __init__(self, t_old, t, y_old, F):
        # 调用父类 DenseOutput 的初始化方法，传入 t_old 和 t
        super().__init__(t_old, t)
        # 计算时间步长 h
        self.h = t - t_old
        # 设置 F 列表
        self.F = F
        # 保存旧状态 y_old
        self.y_old = y_old

    # 实现 DenseOutput 中的 _call_impl 方法，根据时间 t 计算输出
    def _call_impl(self, t):
        # 计算归一化时间变量 x
        x = (t - self.t_old) / self.h

        # 如果时间 t 是标量
        if t.ndim == 0:
            # 创建一个与 y_old 相同形状的零数组 y
            y = np.zeros_like(self.y_old)
        else:
            # 如果时间 t 是数组，扩展 x 的
```