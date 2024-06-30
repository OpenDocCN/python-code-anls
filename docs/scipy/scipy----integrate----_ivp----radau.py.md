# `D:\src\scipysrc\scipy\scipy\integrate\_ivp\radau.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
from scipy.linalg import lu_factor, lu_solve  # 导入 LU 分解和求解线性方程组的函数
from scipy.sparse import csc_matrix, issparse, eye  # 导入稀疏矩阵相关函数
from scipy.sparse.linalg import splu  # 导入稀疏矩阵的 LU 分解函数
from scipy.optimize._numdiff import group_columns  # 导入用于数值微分的函数
from .common import (validate_max_step, validate_tol, select_initial_step,
                     norm, num_jac, EPS, warn_extraneous,
                     validate_first_step)  # 从本地模块导入一些通用函数和常量
from .base import OdeSolver, DenseOutput  # 从本地模块导入 OdeSolver 和 DenseOutput 类

S6 = 6 ** 0.5  # 计算并赋值常数 S6 的平方根值

# Butcher tableau. A is not used directly, see below.
C = np.array([(4 - S6) / 10, (4 + S6) / 10, 1])  # 设置 Radau 方法的 Butcher 表 C 系数

E = np.array([-13 - 7 * S6, -13 + 7 * S6, -1]) / 3  # 设置 Radau 方法的 Butcher 表 E 系数

# Eigendecomposition of A is done: A = T L T**-1. There is 1 real eigenvalue
# and a complex conjugate pair. They are written below.
MU_REAL = 3 + 3 ** (2 / 3) - 3 ** (1 / 3)  # 计算实部特征值 MU_REAL

MU_COMPLEX = (3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3))
              - 0.5j * (3 ** (5 / 6) + 3 ** (7 / 6)))  # 计算复数特征值 MU_COMPLEX

# These are transformation matrices.
T = np.array([  # 定义变换矩阵 T
    [0.09443876248897524, -0.14125529502095421, 0.03002919410514742],
    [0.25021312296533332, 0.20412935229379994, -0.38294211275726192],
    [1, 1, 0]])

TI = np.array([  # 定义逆变换矩阵 TI
    [4.17871859155190428, 0.32768282076106237, 0.52337644549944951],
    [-4.17871859155190428, -0.32768282076106237, 0.47662355450055044],
    [0.50287263494578682, -2.57192694985560522, 0.59603920482822492]])

# These linear combinations are used in the algorithm.
TI_REAL = TI[0]  # 选择逆变换矩阵的实部部分
TI_COMPLEX = TI[1] + 1j * TI[2]  # 选择逆变换矩阵的复数部分

# Interpolator coefficients.
P = np.array([  # 定义插值器系数 P
    [13/3 + 7*S6/3, -23/3 - 22*S6/3, 10/3 + 5 * S6],
    [13/3 - 7*S6/3, -23/3 + 22*S6/3, 10/3 - 5 * S6],
    [1/3, -8/3, 10/3]])

NEWTON_MAXITER = 6  # 设置牛顿迭代的最大迭代次数
MIN_FACTOR = 0.2  # 最小允许步长缩小比例
MAX_FACTOR = 10  # 最大允许步长增大比例


def solve_collocation_system(fun, t, y, h, Z0, scale, tol,
                             LU_real, LU_complex, solve_lu):
    """Solve the collocation system.

    Parameters
    ----------
    fun : callable
        系统的右手边函数。
    t : float
        当前时间。
    y : ndarray, shape (n,)
        当前状态。
    h : float
        尝试的步长。
    Z0 : ndarray, shape (3, n)
        解的初始猜测。它决定在时间 t + h * C 处的新值 `y`，其中 `C` 是 Radau 方法的常数。
    scale : ndarray, shape (n)
        问题的容差标度，即 `rtol * abs(y) + atol`。
    tol : float
        解决系统的容差。该值与通过 `scale` 归一化的误差进行比较。
    LU_real, LU_complex
        系统雅可比矩阵的 LU 分解。
    solve_lu : callable
        给定 LU 分解，解线性系统的可调用函数。其签名为 `solve_lu(LU, b)`。

    Returns
    -------
    converged : bool
        迭代是否收敛。
    n_iter : int
        完成的迭代次数。
    Z : ndarray, shape (3, n)
        找到的解。

    """
    rate : float
        The rate of convergence.
    """
    # 获取输入向量 y 的长度
    n = y.shape[0]
    # 计算实部的 M 值
    M_real = MU_REAL / h
    # 计算复数部分的 M 值
    M_complex = MU_COMPLEX / h

    # 初始向量 W 是 TI 与 Z0 的乘积
    W = TI.dot(Z0)
    # 初始向量 Z 等于 Z0
    Z = Z0

    # 初始化 F 数组，形状为 (3, n)
    F = np.empty((3, n))
    # 计算 h 乘以常数 C 的结果
    ch = h * C

    # 用于存储上一步的 dW_norm 值
    dW_norm_old = None
    # 初始化 dW 数组，与 W 具有相同的形状
    dW = np.empty_like(W)
    # 迭代是否已收敛的标志
    converged = False
    # 初始收敛速率为 None
    rate = None
    # 开始牛顿迭代过程，最多进行 NEWTON_MAXITER 次
    for k in range(NEWTON_MAXITER):
        # 对三个方向进行迭代
        for i in range(3):
            # 计算函数值 F[i]，对应于 t + ch[i] 和 y + Z[i]
            F[i] = fun(t + ch[i], y + Z[i])

        # 如果 F 中存在非有限值，终止迭代
        if not np.all(np.isfinite(F)):
            break

        # 计算实部和复部分的 f 值
        f_real = F.T.dot(TI_REAL) - M_real * W[0]
        f_complex = F.T.dot(TI_COMPLEX) - M_complex * (W[1] + 1j * W[2])

        # 使用 LU 分解求解线性方程组，得到 dW_real 和 dW_complex
        dW_real = solve_lu(LU_real, f_real)
        dW_complex = solve_lu(LU_complex, f_complex)

        # 将 dW_real 和复部分的实部和虚部组成 dW 数组
        dW[0] = dW_real
        dW[1] = dW_complex.real
        dW[2] = dW_complex.imag

        # 计算当前步长的归一化范数
        dW_norm = norm(dW / scale)

        # 如果有前一步的归一化范数值，计算收敛速率 rate
        if dW_norm_old is not None:
            rate = dW_norm / dW_norm_old

        # 如果速率已知且满足收敛条件，则终止迭代
        if (rate is not None and (rate >= 1 or
                rate ** (NEWTON_MAXITER - k) / (1 - rate) * dW_norm > tol)):
            break

        # 更新 W 向量
        W += dW
        # 更新 Z 向量
        Z = T.dot(W)

        # 如果 dW_norm 为零或者速率已知且满足收敛条件，则迭代收敛
        if (dW_norm == 0 or
                rate is not None and rate / (1 - rate) * dW_norm < tol):
            converged = True
            break

        # 更新 dW_norm_old
        dW_norm_old = dW_norm

    # 返回迭代是否收敛，迭代次数加一，最终 Z 向量和收敛速率 rate
    return converged, k + 1, Z, rate
def predict_factor(h_abs, h_abs_old, error_norm, error_norm_old):
    """
    Predict by which factor to increase/decrease the step size.
    
    The algorithm is described in [1]_.

    Parameters
    ----------
    h_abs, h_abs_old : float
        Current and previous values of the step size, `h_abs_old` can be None
        (see Notes).
    error_norm, error_norm_old : float
        Current and previous values of the error norm, `error_norm_old` can
        be None (see Notes).

    Returns
    -------
    factor : float
        Predicted factor.

    Notes
    -----
    If `h_abs_old` and `error_norm_old` are both not None then a two-step
    algorithm is used, otherwise a one-step algorithm is used.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations II: Stiff and Differential-Algebraic Problems", Sec. IV.8.
    """
    # Determine the multiplier based on current and previous step sizes and error norms
    if error_norm_old is None or h_abs_old is None or error_norm == 0:
        # Use a multiplier of 1 if any of the necessary previous values are None or if error_norm is zero
        multiplier = 1
    else:
        # Compute the multiplier using the two-step algorithm described in [1]
        multiplier = h_abs / h_abs_old * (error_norm_old / error_norm) ** 0.25

    # Compute the factor with error norm considerations, ignoring divide-by-zero errors
    with np.errstate(divide='ignore'):
        factor = min(1, multiplier) * error_norm ** -0.25

    return factor


class Radau(OdeSolver):
    """
    Implicit Runge-Kutta method of Radau IIA family of order 5.

    The implementation follows [1]_. The error is controlled with a
    third-order accurate embedded formula. A cubic polynomial which satisfies
    the collocation conditions is used for the dense output.

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
    """
    pass
    # 相对误差和绝对误差容限：`rtol` 和 `atol`
    rtol, atol : float and array_like, optional
        # 相对误差和绝对误差容限。解算器保持局部误差估计小于 ``atol + rtol * abs(y)``。
        # 这里 `rtol` 控制相对精度（正确数字的数量），而 `atol` 控制绝对精度（正确小数点的数量）。
        # 为了达到期望的 `rtol`，需将 `atol` 设为小于预期 ``rtol * abs(y)`` 的最小值，以确保 `rtol` 主导允许的误差。
        # 如果 `atol` 大于 ``rtol * abs(y)``，则不能保证正确数字的数量。
        # 相反，为了达到期望的 `atol`，需设置 `rtol`，使得 ``rtol * abs(y)`` 始终小于 `atol`。
        # 如果 y 的各分量具有不同的尺度，可能通过传递形状为 (n,) 的 array_like 来为不同的分量设置不同的 `atol` 值是有益的。
        # 默认值为 `rtol` 为 1e-3，`atol` 为 1e-6。
    jac : {None, array_like, sparse_matrix, callable}, optional
        # 系统右侧的雅可比矩阵相对于 y 的矩阵，此方法所需。
        # 雅可比矩阵的形状为 (n, n)，其元素 (i, j) 等于 ``d f_i / d y_j``。
        # 有三种定义雅可比矩阵的方式：
        # * 如果是 array_like 或 sparse_matrix，则假定雅可比矩阵是常数。
        # * 如果是 callable，则假定雅可比矩阵依赖于 t 和 y；会按需调用为 ``jac(t, y)``。
        #   对于 'Radau' 和 'BDF' 方法，返回值可能是稀疏矩阵。
        # * 如果是 None（默认），雅可比矩阵将通过有限差分来近似。
        # 通常建议提供雅可比矩阵而不是依赖有限差分的近似。
    jac_sparsity : {None, array_like, sparse matrix}, optional
        # 定义雅可比矩阵的稀疏结构，用于有限差分近似。
        # 其形状必须为 (n, n)。如果 `jac` 不是 `None`，则忽略此参数。
        # 如果雅可比矩阵在每行中只有少量非零元素，则提供稀疏结构将大大加速计算 [2]_。
        # 零条目意味着雅可比矩阵中的对应元素始终为零。如果为 None（默认），则假定雅可比矩阵是密集的。
    vectorized : bool, optional
        Whether `fun` can be called in a vectorized fashion. Default is False.
        # 参数 `vectorized`：指示是否可以以向量化方式调用 `fun` 函数，默认为 False。

        If ``vectorized`` is False, `fun` will always be called with ``y`` of
        shape ``(n,)``, where ``n = len(y0)``.
        # 如果 `vectorized` 为 False，则 `fun` 函数将始终以形状为 ``(n,)`` 的 `y` 被调用，
        # 其中 `n = len(y0)`。

        If ``vectorized`` is True, `fun` may be called with ``y`` of shape
        ``(n, k)``, where ``k`` is an integer. In this case, `fun` must behave
        such that ``fun(t, y)[:, i] == fun(t, y[:, i])`` (i.e. each column of
        the returned array is the time derivative of the state corresponding
        with a column of ``y``).
        # 如果 `vectorized` 为 True，则 `fun` 函数可以以形状为 ``(n, k)`` 的 `y` 被调用，
        # 其中 `k` 是整数。在这种情况下，`fun` 必须表现出 `fun(t, y)[:, i] == fun(t, y[:, i])`
        # （即返回数组的每一列都是对应于 `y` 的列的状态的时间导数）。

        Setting ``vectorized=True`` allows for faster finite difference
        approximation of the Jacobian by this method, but may result in slower
        execution overall in some circumstances (e.g. small ``len(y0)``).
        # 将 `vectorized=True` 设置为可以通过该方法更快地近似计算雅可比矩阵的有限差分，
        # 但在某些情况下可能导致总体执行速度较慢（例如 `len(y0)` 较小的情况）。

    Attributes
    ----------
    n : int
        Number of equations.
        # 方程的数量。

    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
        # 求解器的当前状态：'running'（运行中）、'finished'（已完成）或 'failed'（失败）。

    t_bound : float
        Boundary time.
        # 边界时间。

    direction : float
        Integration direction: +1 or -1.
        # 积分方向：+1 或 -1。

    t : float
        Current time.
        # 当前时间。

    y : ndarray
        Current state.
        # 当前状态的数组。

    t_old : float
        Previous time. None if no steps were made yet.
        # 上一个时间。如果尚未进行步骤，则为 None。

    step_size : float
        Size of the last successful step. None if no steps were made yet.
        # 上一个成功步骤的大小。如果尚未进行步骤，则为 None。

    nfev : int
        Number of evaluations of the right-hand side.
        # 右侧函数的评估次数。

    njev : int
        Number of evaluations of the Jacobian.
        # 雅可比矩阵的评估次数。

    nlu : int
        Number of LU decompositions.
        # LU 分解的数量。

    References
    ----------
    .. [1] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II:
           Stiff and Differential-Algebraic Problems", Sec. IV.8.
        # 参考文献 [1]：E. Hairer, G. Wanner,《求解常微分方程 II：刚性和微分代数问题》，第 IV.8 节。

    .. [2] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
           sparse Jacobian matrices", Journal of the Institute of Mathematics
           and its Applications, 13, pp. 117-120, 1974.
        # 参考文献 [2]：A. Curtis, M. J. D. Powell, 和 J. Reid，《关于稀疏雅可比矩阵估计的论文》，
        # 数学研究所及其应用期刊，第 13 卷，117-120 页，1974 年。
    def __init__(self, fun, t0, y0, t_bound, max_step=np.inf,
                 rtol=1e-3, atol=1e-6, jac=None, jac_sparsity=None,
                 vectorized=False, first_step=None, **extraneous):
        # 检查并警告多余的参数
        warn_extraneous(extraneous)
        # 调用父类的初始化方法，传入函数、初始时间、初始状态、终止时间和向量化标志
        super().__init__(fun, t0, y0, t_bound, vectorized)
        # 初始化存储旧状态的变量
        self.y_old = None
        # 验证并设置最大步长
        self.max_step = validate_max_step(max_step)
        # 验证并设置相对误差和绝对误差容差
        self.rtol, self.atol = validate_tol(rtol, atol, self.n)
        # 计算初始函数值
        self.f = self.fun(self.t, self.y)

        # 选择初始步长，假设使用与误差控制相同的阶数
        if first_step is None:
            self.h_abs = select_initial_step(
                self.fun, self.t, self.y, t_bound, max_step, self.f, self.direction,
                3, self.rtol, self.atol)
        else:
            self.h_abs = validate_first_step(first_step, t0, t_bound)

        # 初始化其他状态变量
        self.h_abs_old = None
        self.error_norm_old = None

        # 计算牛顿法的容差
        self.newton_tol = max(10 * EPS / rtol, min(0.03, rtol ** 0.5))
        self.sol = None

        # 初始化雅可比矩阵相关的变量
        self.jac_factor = None
        # 验证并设置雅可比矩阵及其稀疏性
        self.jac, self.J = self._validate_jac(jac, jac_sparsity)

        # 根据雅可比矩阵的稀疏性定义不同的 LU 分解和解法
        if issparse(self.J):
            def lu(A):
                self.nlu += 1
                return splu(A)

            def solve_lu(LU, b):
                return LU.solve(b)

            I = eye(self.n, format='csc')
        else:
            def lu(A):
                self.nlu += 1
                return lu_factor(A, overwrite_a=True)

            def solve_lu(LU, b):
                return lu_solve(LU, b, overwrite_b=True)

            I = np.identity(self.n)

        # 设置 LU 分解、LU 解法和单位矩阵
        self.lu = lu
        self.solve_lu = solve_lu
        self.I = I

        # 初始化其他状态变量
        self.current_jac = True
        self.LU_real = None
        self.LU_complex = None
        self.Z = None
    # 定义一个方法用于验证雅可比矩阵，接受雅可比矩阵 `jac` 和稀疏矩阵 `sparsity` 作为参数
    def _validate_jac(self, jac, sparsity):
        # 保存当前对象的时间属性到局部变量 t0
        t0 = self.t
        # 保存当前对象的状态属性到局部变量 y0
        y0 = self.y

        # 如果参数 jac 为 None，则根据 sparsity 进行处理
        if jac is None:
            # 如果 sparsity 不为 None，则处理稀疏矩阵 sparsity
            if sparsity is not None:
                # 如果 sparsity 是稀疏矩阵，则将其转换为压缩稀疏列格式
                if issparse(sparsity):
                    sparsity = csc_matrix(sparsity)
                # 对 sparsity 进行列分组
                groups = group_columns(sparsity)
                # 将 sparsity 和分组信息作为元组保存到 sparsity 中
                sparsity = (sparsity, groups)

            # 定义一个内部函数 jac_wrapped，用于计算雅可比矩阵 J
            def jac_wrapped(t, y, f):
                # 增加雅可比矩阵计算次数的计数器
                self.njev += 1
                # 调用 num_jac 函数计算数值雅可比矩阵 J，同时更新内部状态
                J, self.jac_factor = num_jac(self.fun_vectorized, t, y, f,
                                             self.atol, self.jac_factor,
                                             sparsity)
                # 返回计算得到的雅可比矩阵 J
                return J
            # 使用当前对象的属性 t0, y0 和方法 self.f 计算初始时刻的雅可比矩阵 J
            J = jac_wrapped(t0, y0, self.f)

        # 如果 jac 是可调用对象，则直接调用 jac 计算雅可比矩阵 J
        elif callable(jac):
            # 调用参数 jac 计算雅可比矩阵 J
            J = jac(t0, y0)
            # 将雅可比计算次数的计数器设置为 1
            self.njev = 1
            # 如果 J 是稀疏矩阵，则将其转换为压缩稀疏列格式
            if issparse(J):
                J = csc_matrix(J)

                # 定义一个内部函数 jac_wrapped，用于计算雅可比矩阵 J
                def jac_wrapped(t, y, _=None):
                    # 增加雅可比矩阵计算次数的计数器
                    self.njev += 1
                    # 调用参数 jac 计算雅可比矩阵 J，并将结果转换为压缩稀疏列格式
                    return csc_matrix(jac(t, y), dtype=float)

            else:
                # 将 J 转换为浮点数数组
                J = np.asarray(J, dtype=float)

                # 定义一个内部函数 jac_wrapped，用于计算雅可比矩阵 J
                def jac_wrapped(t, y, _=None):
                    # 增加雅可比矩阵计算次数的计数器
                    self.njev += 1
                    # 调用参数 jac 计算雅可比矩阵 J，并将结果转换为浮点数数组
                    return np.asarray(jac(t, y), dtype=float)

            # 检查计算得到的雅可比矩阵 J 的形状是否为 (self.n, self.n)
            if J.shape != (self.n, self.n):
                # 如果形状不符合预期，抛出 ValueError 异常
                raise ValueError("`jac` is expected to have shape {}, but "
                                 "actually has {}."
                                 .format((self.n, self.n), J.shape))

        # 如果 jac 既不为 None 也不是可调用对象，则处理已给定的矩阵 jac
        else:
            # 如果 jac 是稀疏矩阵，则将其转换为压缩稀疏列格式
            if issparse(jac):
                J = csc_matrix(jac)
            else:
                # 将 jac 转换为浮点数数组
                J = np.asarray(jac, dtype=float)

            # 检查计算得到的雅可比矩阵 J 的形状是否为 (self.n, self.n)
            if J.shape != (self.n, self.n):
                # 如果形状不符合预期，抛出 ValueError 异常
                raise ValueError("`jac` is expected to have shape {}, but "
                                 "actually has {}."
                                 .format((self.n, self.n), J.shape))
            # 将 jac_wrapped 设置为 None
            jac_wrapped = None

        # 返回最终计算得到的雅可比矩阵函数和雅可比矩阵 J
        return jac_wrapped, J

    # 计算稠密输出的方法，使用当前对象的属性 Z 和 P 进行计算
    def _compute_dense_output(self):
        # 计算矩阵乘积 Q = Z^T * P
        Q = np.dot(self.Z.T, P)
        # 返回一个 RadauDenseOutput 对象，以记录计算结果
        return RadauDenseOutput(self.t_old, self.t, self.y_old, Q)

    # 返回当前解的稠密输出对象的方法
    def _dense_output_impl(self):
        # 直接返回当前对象的解对象 self.sol
        return self.sol
class RadauDenseOutput(DenseOutput):
    # RadauDenseOutput 类，继承自 DenseOutput 类
    def __init__(self, t_old, t, y_old, Q):
        # 初始化方法，接受 t_old, t, y_old, Q 四个参数
        super().__init__(t_old, t)
        # 调用父类 DenseOutput 的初始化方法
        self.h = t - t_old
        # 计算时间步长 h
        self.Q = Q
        # 存储 Q 矩阵
        self.order = Q.shape[1] - 1
        # 计算 Q 矩阵的阶数
        self.y_old = y_old
        # 存储之前的 y 值

    def _call_impl(self, t):
        # 内部调用方法，接受参数 t
        x = (t - self.t_old) / self.h
        # 计算时间比例因子 x
        if t.ndim == 0:
            # 如果 t 的维度为 0
            p = np.tile(x, self.order + 1)
            # 用 x 创建一个重复数组 p
            p = np.cumprod(p)
            # 对 p 中的元素进行累积乘积操作
        else:
            # 如果 t 的维度不为 0
            p = np.tile(x, (self.order + 1, 1))
            # 用 x 创建一个形状为 (self.order + 1, 1) 的重复数组 p
            p = np.cumprod(p, axis=0)
            # 对 p 中的元素进行累积乘积操作，沿着第一个轴（竖直方向）进行
        # 这里没有乘以 h，不是错误。
        # 这里是因为之后会用 Q 矩阵来乘以这些值，h 的影响已经在 Q 矩阵中考虑了
        y = np.dot(self.Q, p)
        # 计算 Q 矩阵与 p 的乘积，得到输出 y
        if y.ndim == 2:
            # 如果 y 的维度为 2
            y += self.y_old[:, None]
            # 将 y_old 扩展为列向量后加到 y 上
        else:
            # 如果 y 的维度不为 2
            y += self.y_old
            # 直接将 y_old 加到 y 上

        return y
        # 返回计算结果 y
```