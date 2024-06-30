# `D:\src\scipysrc\scipy\scipy\integrate\_ivp\base.py`

```
# 导入 NumPy 库，用于科学计算
import numpy as np

# 定义一个辅助函数，用于检查所有求解器共有的参数
def check_arguments(fun, y0, support_complex):
    """Helper function for checking arguments common to all solvers."""
    # 将 y0 转换为 NumPy 数组
    y0 = np.asarray(y0)
    
    # 检查 y0 是否为复数类型
    if np.issubdtype(y0.dtype, np.complexfloating):
        # 如果 y0 是复数类型但不支持复数求解，则抛出异常
        if not support_complex:
            raise ValueError("`y0` is complex, but the chosen solver does "
                             "not support integration in a complex domain.")
        # 设置数据类型为复数
        dtype = complex
    else:
        # 设置数据类型为浮点数
        dtype = float
    
    # 将 y0 转换为指定数据类型，避免拷贝
    y0 = y0.astype(dtype, copy=False)

    # 检查 y0 的维度是否为 1
    if y0.ndim != 1:
        raise ValueError("`y0` must be 1-dimensional.")

    # 检查 y0 中所有元素是否有限
    if not np.isfinite(y0).all():
        raise ValueError("All components of the initial state `y0` must be finite.")

    # 定义一个包装函数，将 fun 的输出转换为指定数据类型
    def fun_wrapped(t, y):
        return np.asarray(fun(t, y), dtype=dtype)

    # 返回包装后的函数及处理过的 y0
    return fun_wrapped, y0

# 定义一个 ODE 求解器的基类
class OdeSolver:
    """Base class for ODE solvers.
    # 为了实现一个新的求解器，需要按照以下准则操作：

        # 1. 构造函数必须接受基类中列出的参数以及任何特定于求解器的其他参数。
           # 构造函数必须接受任意的额外参数 **extraneous**，但要使用 `common.warn_extraneous` 函数警告这些参数是无关的。
           # 不要将这些参数传递给基类。

        # 2. 求解器必须实现一个私有方法 `_step_impl(self)`，它将求解器推进一步。它必须返回一个元组 ``(success, message)``
           # 其中 `success` 是一个布尔值，指示步骤是否成功，`message` 是一个字符串，包含步骤失败的描述或者为 None。

        # 3. 求解器必须实现一个私有方法 `_dense_output_impl(self)`，
           # 返回一个 `DenseOutput` 对象，覆盖最后成功的步骤。

        # 4. 求解器必须具有下面列出的属性。
           # 注意 `t_old` 和 `step_size` 会自动更新。

        # 5. 使用 `fun(self, t, y)` 方法来评估系统的右手边（rhs），这样函数评估的次数 `nfev` 将会自动跟踪。

        # 6. 为了方便起见，基类提供了 `fun_single(self, t, y)` 和 `fun_vectorized(self, t, y)` 方法，
           # 分别用于以非矢量化和矢量化的方式评估 rhs（无论构造函数中的 `fun` 如何实现）。这些调用不会增加 `nfev`。

        # 7. 如果求解器使用雅可比矩阵和LU分解，则应跟踪雅可比矩阵评估的次数 `njev` 和 LU 分解的次数 `nlu`。

        # 8. 根据惯例，用于计算雅可比矩阵有限差分近似的函数评估不应计入 `nfev` 中，
           # 因此在计算雅可比矩阵的有限差分近似时，请使用 `fun_single(self, t, y)` 或 `fun_vectorized(self, t, y)`。

    # 参数
    # ----------
    # fun : callable
        # 系统右手边：状态 ``y`` 在时间 ``t`` 的时间导数。调用签名是 ``fun(t, y)``，其中 ``t`` 是标量，``y`` 是形状为 (len(y0),) 的 ndarray。
        # ``fun`` 必须返回与 ``y`` 相同形状的数组。有关更多信息，请参见 `vectorized`。

    # t0 : float
        # 初始时间。

    # y0 : array_like, shape (n,)
        # 初始状态。

    # t_bound : float
        # 边界时间 —— 积分不会超过它。它还确定积分的方向。
    vectorized : bool
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
        will result in slower execution for other methods. It can also
        result in slower overall execution for 'Radau' and 'BDF' in some
        circumstances (e.g. small ``len(y0)``).
    support_complex : bool, optional
        Whether integration in a complex domain should be supported.
        Generally determined by a derived solver class capabilities.
        Default is False.

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
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number of the system's rhs evaluations.
    njev : int
        Number of the Jacobian evaluations.
    nlu : int
        Number of LU decompositions.
    """
    TOO_SMALL_STEP = "Required step size is less than spacing between numbers."

    # 初始化方法，设置求解器的初始状态和参数
    def __init__(self, fun, t0, y0, t_bound, vectorized,
                 support_complex=False):
        self.t_old = None
        self.t = t0
        # 检查参数并设置函数和初始状态
        self._fun, self.y = check_arguments(fun, y0, support_complex)
        self.t_bound = t_bound
        self.vectorized = vectorized

        # 根据 vectorized 参数选择相应的函数处理方式
        if vectorized:
            # 定义支持向量化处理的函数
            def fun_single(t, y):
                return self._fun(t, y[:, None]).ravel()
            fun_vectorized = self._fun
        else:
            fun_single = self._fun

            # 定义非向量化处理的函数
            def fun_vectorized(t, y):
                f = np.empty_like(y)
                for i, yi in enumerate(y.T):
                    f[:, i] = self._fun(t, yi)
                return f

        # 定义主要的函数调用接口
        def fun(t, y):
            self.nfev += 1
            return self.fun_single(t, y)

        # 将定义的函数绑定到类的实例上
        self.fun = fun
        self.fun_single = fun_single
        self.fun_vectorized = fun_vectorized

        # 设置时间步长方向
        self.direction = np.sign(t_bound - t0) if t_bound != t0 else 1
        self.n = self.y.size
        self.status = 'running'

        # 初始化计数器
        self.nfev = 0
        self.njev = 0
        self.nlu = 0

    @property
    # 如果上一步时间步长为 None，则返回 None
    def step_size(self):
        if self.t_old is None:
            return None
        else:
            # 返回当前时间步长与上一步时间步长的绝对差值
            return np.abs(self.t - self.t_old)

    def step(self):
        """执行一步积分运算。

        Returns
        -------
        message : string or None
            求解器的报告。如果 `self.status` 在步骤完成后为 'failed'，则通常是失败的原因，否则为 None。
        """
        if self.status != 'running':
            # 如果求解器状态不是 'running'，则抛出运行时错误
            raise RuntimeError("Attempt to step on a failed or finished "
                               "solver.")

        if self.n == 0 or self.t == self.t_bound:
            # 处理空求解器或者无积分的特殊情况
            self.t_old = self.t
            self.t = self.t_bound
            message = None
            self.status = 'finished'
        else:
            t = self.t
            # 调用内部的步骤实现方法，获取成功状态和消息
            success, message = self._step_impl()

            if not success:
                # 如果步骤执行不成功，则将求解器状态标记为 'failed'
                self.status = 'failed'
            else:
                self.t_old = t
                # 检查是否达到积分终点，根据方向和时间判断是否完成积分
                if self.direction * (self.t - self.t_bound) >= 0:
                    self.status = 'finished'

        return message

    def dense_output(self):
        """计算上一次成功步骤的局部插值。

        Returns
        -------
        sol : `DenseOutput`
            上一次成功步骤的局部插值。
        """
        if self.t_old is None:
            # 如果上一步时间步长为 None，则抛出运行时错误
            raise RuntimeError("Dense output is available after a successful "
                               "step was made.")

        if self.n == 0 or self.t == self.t_old:
            # 处理空求解器和无积分的特殊情况，返回常数插值对象
            return ConstantDenseOutput(self.t_old, self.t, self.y)
        else:
            # 调用内部的密集输出实现方法，获取局部插值对象
            return self._dense_output_impl()

    def _step_impl(self):
        # 抽象方法，子类需实现具体的步骤实现逻辑
        raise NotImplementedError

    def _dense_output_impl(self):
        # 抽象方法，子类需实现具体的密集输出实现逻辑
        raise NotImplementedError
# 定义一个基础类，用于插值ODE求解器生成的步长上的局部插值器。
#
# 它在 `t_min` 和 `t_max` 之间进行插值（见下面的属性）。
# 虽然允许在此间隔之外进行评估，但精度不能保证。
class DenseOutput:
    """Base class for local interpolant over step made by an ODE solver.

    It interpolates between `t_min` and `t_max` (see Attributes below).
    Evaluation outside this interval is not forbidden, but the accuracy is not
    guaranteed.

    Attributes
    ----------
    t_min, t_max : float
        Time range of the interpolation.
    """
    
    def __init__(self, t_old, t):
        # 设置旧时间 `t_old` 和当前时间 `t`
        self.t_old = t_old
        self.t = t
        # 计算 `t_min` 和 `t_max`，分别为 `t` 和 `t_old` 的最小和最大值
        self.t_min = min(t, t_old)
        self.t_max = max(t, t_old)

    def __call__(self, t):
        """Evaluate the interpolant.

        Parameters
        ----------
        t : float or array_like with shape (n_points,)
            Points to evaluate the solution at.

        Returns
        -------
        y : ndarray, shape (n,) or (n, n_points)
            Computed values. Shape depends on whether `t` was a scalar or a
            1-D array.
        """
        # 将参数 `t` 转换为 numpy 数组
        t = np.asarray(t)
        # 如果 `t` 的维度大于 1，抛出错误，要求 `t` 必须是 float 或者 1-D 数组
        if t.ndim > 1:
            raise ValueError("`t` must be a float or a 1-D array.")
        # 调用实际的插值实现函数
        return self._call_impl(t)

    def _call_impl(self, t):
        # 抽象方法，需要在子类中实现具体的插值逻辑
        raise NotImplementedError


class ConstantDenseOutput(DenseOutput):
    """Constant value interpolator.

    This class used for degenerate integration cases: equal integration limits
    or a system with 0 equations.
    """
    
    def __init__(self, t_old, t, value):
        # 调用父类的构造函数初始化 `t_old` 和 `t`
        super().__init__(t_old, t)
        # 设置常数值 `value`
        self.value = value

    def _call_impl(self, t):
        # 如果 `t` 是标量，返回常数值 `self.value`
        if t.ndim == 0:
            return self.value
        else:
            # 如果 `t` 是一维数组，创建一个与 `self.value` 形状相同的空数组 `ret`
            # 然后用 `self.value` 的每一行填充 `ret`
            ret = np.empty((self.value.shape[0], t.shape[0]))
            ret[:] = self.value[:, None]
            return ret
```