# `D:\src\scipysrc\scipy\scipy\integrate\_odepack_py.py`

```
# 作者：Travis Oliphant

# 定义模块的公开接口，包括 odeint 函数和 ODEintWarning 类
__all__ = ['odeint', 'ODEintWarning']

# 导入 NumPy 库，并使用 np 作为别名
import numpy as np

# 导入 _odepack 模块，该模块应该是定义了底层的 ODE 求解器
from . import _odepack

# 导入 copy 函数，用于复制对象
from copy import copy

# 导入 warnings 模块，用于处理警告
import warnings

# 定义一个名为 ODEintWarning 的警告类，用于在 odeint 执行过程中抛出警告
class ODEintWarning(Warning):
    """Warning raised during the execution of `odeint`."""
    pass

# 定义一个字典 _msgs，用于存储不同整数代码对应的信息
_msgs = {2: "Integration successful.",
         1: "Nothing was done; the integration time was 0.",
         -1: "Excess work done on this call (perhaps wrong Dfun type).",
         -2: "Excess accuracy requested (tolerances too small).",
         -3: "Illegal input detected (internal error).",
         -4: "Repeated error test failures (internal error).",
         -5: "Repeated convergence failures (perhaps bad Jacobian or tolerances).",
         -6: "Error weight became zero during problem.",
         -7: "Internal workspace insufficient to finish (internal error).",
         -8: "Run terminated (internal error)."
         }

# 定义 odeint 函数，用于解决一组常微分方程
def odeint(func, y0, t, args=(), Dfun=None, col_deriv=0, full_output=0,
           ml=None, mu=None, rtol=None, atol=None, tcrit=None, h0=0.0,
           hmax=0.0, hmin=0.0, ixpr=0, mxstep=0, mxhnil=0, mxordn=12,
           mxords=5, printmessg=0, tfirst=False):
    """
    Integrate a system of ordinary differential equations.

    .. note:: For new code, use `scipy.integrate.solve_ivp` to solve a
              differential equation.

    Solve a system of ordinary differential equations using lsoda from the
    FORTRAN library odepack.

    Solves the initial value problem for stiff or non-stiff systems
    of first order ode-s::

        dy/dt = func(y, t, ...)  [or func(t, y, ...)]

    where y can be a vector.

    .. note:: By default, the required order of the first two arguments of
              `func` are in the opposite order of the arguments in the system
              definition function used by the `scipy.integrate.ode` class and
              the function `scipy.integrate.solve_ivp`. To use a function with
              the signature ``func(t, y, ...)``, the argument `tfirst` must be
              set to ``True``.

    Parameters
    ----------
    func : callable(y, t, ...) or callable(t, y, ...)
        Computes the derivative of y at t.
        If the signature is ``callable(t, y, ...)``, then the argument
        `tfirst` must be set ``True``.
        `func` must not modify the data in `y`, as it is a
        view of the data used internally by the ODE solver.
    y0 : array
        Initial condition on y (can be a vector).
    t : array
        A sequence of time points for which to solve for y. The initial
        value point should be the first element of this sequence.
        This sequence must be monotonically increasing or monotonically
        decreasing; repeated values are allowed.
    args : tuple, optional
        Extra arguments to pass to function.
    Dfun : callable(y, t, ...) or None, optional
        Gradient (Jacobian) of func.
    col_deriv : bool, optional
        True if Dfun defines derivatives down columns (faster), otherwise across rows.
    full_output : bool, optional
        True if to return a dictionary of optional outputs as the second output.
    ml, mu : int, optional
        If either of these are not None or non-negative, then the Jacobian is assumed to be banded.
        These give the number of lower and upper non-zero diagonals in the Jacobian.
        Setting ml = mu = 0 indicates a full (not banded) Jacobian.
    rtol, atol : float, optional
        Relative and absolute tolerances.
    tcrit : array_like, optional
        Vector of critical points (e.g., singularities) where integration care should be taken.
    h0, hmax, hmin : float, optional
        Initial and maximum step sizes and minimum step size allowed.
    ixpr, mxstep, mxhnil, mxordn, mxords : int, optional
        Other optional inputs.
    printmessg : bool, optional
        Whether to print warning messages.
    tfirst : bool, optional
        Whether the first two arguments of `func` are `t, y` (True) or `y, t` (False).

    Returns
    -------
    y : ndarray, shape (len(t), len(y0))
        Array containing the value of y for each desired time in t,
        with the initial value y0 in the first row.
    infodict : dict, only returned if full_output == True
        Dictionary containing additional output information:

            'hu'    : vector of step sizes successfully used for each time step.
            'tcur'  : vector with the value of t reached for each time step.
            'tolsf' : vector of tolerance scale factors, greater than 1.0, computed when a request for too much accuracy was detected.
            'tsw'   : value of t at the time of the last method switch, if switching methods.
            'nst'   : cumulative number of time steps.
            'nfe'   : cumulative number of function evaluations for the system.
            'nje'   : cumulative number of jacobian evaluations for the system.
            'nqu'   : cumulative number of iterations in the integration scheme.
            'imxer' : index of the component of largest magnitude in the vector of estimated local errors on the last successful step.
            'lenrw' : number of double precision floating point words in RWORK.
            'leniw' : number of integer words in IWORK.
            'mused' : method indicator for the last successful step: 1=Adams (non-stiff), 2=BDF (stiff).
    """
    pass  # 此处为占位符，odeint 函数的实际实现被省略
    # Dfun 是一个可调用对象，表示 func 的梯度（雅可比矩阵）
    # 如果签名是 `callable(t, y, ...)`，则必须将参数 `tfirst` 设置为 True
    # `Dfun` 不能修改 `y` 中的数据，因为它是 ODE 求解器内部使用的数据视图

    # col_deriv 是一个布尔值，可选参数
    # 如果为 True，表示 `Dfun` 定义了沿列的导数（更快），否则 `Dfun` 应定义跨行的导数

    # full_output 是一个布尔值，可选参数
    # 如果为 True，表示要返回作为第二个输出的可选输出字典

    # printmessg 是一个布尔值，可选参数
    # 是否打印收敛消息

    # tfirst 是一个布尔值，可选参数
    # 如果为 True，则 `func`（和 `Dfun` 如果给定）的前两个参数必须是 `t, y`，而不是默认的 `y, t`

    # 返回值
    # y : 数组，形状为 (len(t), len(y0))
    # 包含每个所需时间点 t 的 y 值的数组，初始值 `y0` 在第一行

    # infodict : 字典，仅在 full_output == True 时返回
    # 包含额外输出信息的字典

    # =======  ============================================================
    # key      meaning
    # =======  ============================================================
    # 'hu'     每个时间步成功使用的步长向量
    # 'tcur'   每个时间步达到的 t 值的向量（将至少与输入时间一样大）
    # 'tolsf'  容差比例因子的向量，大于 1.0，当检测到对精度要求过高时计算
    # 'tsw'    最后一次方法切换时的 t 值（给出每个时间步）
    # 'nst'    累积时间步数
    # 'nfe'    每个时间步的累积函数评估次数
    # 'nje'    每个时间步的累积雅可比矩阵评估次数
    # 'nqu'    每个成功步骤的方法阶数的向量
    # 'imxer'  在错误返回时加权局部误差向量（e / ewt）的最大分量索引，否则为 -1
    # 'lenrw'  所需双精度工作数组的长度
    # 'leniw'  所需整数工作数组的长度
    # 'mused'  每个成功时间步的方法指示符的向量：
    #          1: adams（非刚性），2: bdf（刚性）
    # =======  ============================================================
    ml, mu : int, optional
        如果其中任一参数不是 None 或非负，则假定雅可比矩阵是带状的。ml 和 mu 分别表示此带状矩阵中的下限和上限非零对角线数目。
        对于带状情况，Dfun 应返回一个矩阵，其行包含非零带（从最低对角线开始）。
        因此，从 Dfun 返回的矩阵 jac 应当具有形状 ``(ml + mu + 1, len(y0))`` 当 ``ml >= 0`` 或 ``mu >= 0`` 时。
        jac 中的数据必须存储，使得 ``jac[i - j + mu, j]`` 包含第 ``i`` 个方程关于第 ``j`` 个状态变量的导数。
        如果 `col_deriv` 为 True，则必须返回 `jac` 的转置。

    rtol, atol : float, optional
        输入参数 `rtol` 和 `atol` 决定求解器执行的误差控制。
        求解器将根据形式为 ``max-norm of (e / ewt) <= 1`` 的不等式控制估计的局部误差向量 `e`，其中 ewt 是计算为 ``ewt = rtol * abs(y) + atol`` 的正误差权重向量。
        rtol 和 atol 可以是与 y 相同长度的向量或标量。默认为 1.49012e-8。

    tcrit : ndarray, optional
        关键点向量，例如积分需谨慎处理的奇点。

    h0 : float, (0: 由求解器决定), optional
        尝试在第一步上使用的步长大小。

    hmax : float, (0: 由求解器决定), optional
        允许的最大绝对步长大小。

    hmin : float, (0: 由求解器决定), optional
        允许的最小绝对步长大小。

    ixpr : bool, optional
        是否在方法切换时生成额外的打印输出。

    mxstep : int, (0: 由求解器决定), optional
        每个 t 积分点允许的最大（内部定义的）步数。

    mxhnil : int, (0: 由求解器决定), optional
        打印的最大消息数。

    mxordn : int, (0: 由求解器决定), optional
        非刚性（Adams）方法允许的最大阶数。

    mxords : int, (0: 由求解器决定), optional
        刚性（BDF）方法允许的最大阶数。
    if ml is None:
        ml = -1  # 如果 ml 为 None，则将其设为 -1，用于函数调用内部改为零
    if mu is None:
        mu = -1  # 如果 mu 为 None，则将其设为 -1，用于函数调用内部改为零

    # 计算时间步长 dt
    dt = np.diff(t)
    # 检查时间数组 t 是否单调递增或单调递减，不允许重复值
    if not ((dt >= 0).all() or (dt <= 0).all()):
        raise ValueError("The values in t must be monotonically increasing "
                         "or monotonically decreasing; repeated values are "
                         "allowed.")

    # 复制时间数组 t 和初始条件数组 y0，避免修改原始数据
    t = copy(t)
    y0 = copy(y0)

    # 调用底层的 ODE 求解函数 _odepack.odeint，传入相应参数进行求解
    output = _odepack.odeint(func, y0, t, args, Dfun, col_deriv, ml, mu,
                             full_output, rtol, atol, tcrit, h0, hmax, hmin,
                             ixpr, mxstep, mxhnil, mxordn, mxords,
                             int(bool(tfirst)))
    
    # 如果返回值小于 0，发出警告信息，建议使用 full_output = 1 获取详细信息
    if output[-1] < 0:
        warning_msg = (f"{_msgs[output[-1]]} Run with full_output = 1 to "
                       f"get quantitative information.")
        warnings.warn(warning_msg, ODEintWarning, stacklevel=2)
    # 如果 printmessg 为真，输出特定的警告信息
    elif printmessg:
        warning_msg = _msgs[output[-1]]
        warnings.warn(warning_msg, ODEintWarning, stacklevel=2)

    # 如果需要完整的输出，将信息添加到 output[1]['message'] 中
    if full_output:
        output[1]['message'] = _msgs[output[-1]]

    # 移除最后一个元素，通常是状态信息，返回求解结果
    output = output[:-1]
    if len(output) == 1:
        return output[0]
    else:
        return output
```