# `D:\src\scipysrc\scipy\scipy\optimize\_zeros_py.py`

```
# 引入警告模块，用于管理警告信息的显示
import warnings
# 引入命名元组，用于创建带有命名字段的数据结构
from collections import namedtuple
# 引入操作符模块，用于操作符相关的功能
import operator
# 从当前包中导入 _zeros 模块
from . import _zeros
# 从 _optimize 模块中导入 OptimizeResult 类
from ._optimize import OptimizeResult
# 导入 numpy 库，并使用 np 别名
import numpy as np

# 设置默认迭代次数
_iter = 100
# 设置默认的容差值
_xtol = 2e-12
# 设置默认的相对容差值，使用 numpy 中 float 类型的机器精度
_rtol = 4 * np.finfo(float).eps

# 定义模块公开的接口列表
__all__ = ['newton', 'bisect', 'ridder', 'brentq', 'brenth', 'toms748',
           'RootResults']

# 定义不同错误类型对应的整数码
_ECONVERGED = 0
_ESIGNERR = -1  # used in _chandrupatla
_ECONVERR = -2
_EVALUEERR = -3
_ECALLBACK = -4
_EINPROGRESS = 1

# 定义各种错误类型的字符串描述
CONVERGED = 'converged'
SIGNERR = 'sign error'
CONVERR = 'convergence error'
VALUEERR = 'value error'
INPROGRESS = 'No error'

# 构建错误码到错误描述的映射字典
flag_map = {_ECONVERGED: CONVERGED, _ESIGNERR: SIGNERR, _ECONVERR: CONVERR,
            _EVALUEERR: VALUEERR, _EINPROGRESS: INPROGRESS}

# 定义一个类，表示根的求解结果，继承自 OptimizeResult 类
class RootResults(OptimizeResult):
    """Represents the root finding result.

    Attributes
    ----------
    root : float
        Estimated root location.
    iterations : int
        Number of iterations needed to find the root.
    function_calls : int
        Number of times the function was called.
    converged : bool
        True if the routine converged.
    flag : str
        Description of the cause of termination.
    method : str
        Root finding method used.

    """

    def __init__(self, root, iterations, function_calls, flag, method):
        # 初始化根的估计值
        self.root = root
        # 初始化迭代次数
        self.iterations = iterations
        # 初始化函数调用次数
        self.function_calls = function_calls
        # 根据错误码初始化是否收敛的标志
        self.converged = flag == _ECONVERGED
        # 根据错误码选择相应的错误描述
        if flag in flag_map:
            self.flag = flag_map[flag]
        else:
            self.flag = flag
        # 初始化使用的求根方法
        self.method = method

# 定义一个函数，根据 full_output 参数决定是否返回完整的结果对象
def results_c(full_output, r, method):
    if full_output:
        # 如果需要完整输出，则解构元组 r，创建 RootResults 对象
        x, funcalls, iterations, flag = r
        results = RootResults(root=x,
                              iterations=iterations,
                              function_calls=funcalls,
                              flag=flag, method=method)
        return x, results
    else:
        # 如果不需要完整输出，则直接返回 r
        return r

# 定义一个内部函数，根据 full_output 参数决定是否返回完整的结果对象
def _results_select(full_output, r, method):
    """Select from a tuple of (root, funccalls, iterations, flag)"""
    x, funcalls, iterations, flag = r
    if full_output:
        # 如果需要完整输出，则创建 RootResults 对象
        results = RootResults(root=x,
                              iterations=iterations,
                              function_calls=funcalls,
                              flag=flag, method=method)
        return x, results
    # 如果不需要完整输出，则返回根的值 x
    return x

# 定义一个装饰器函数，用于处理函数计算中出现的 NaN 值
def _wrap_nan_raise(f):

    def f_raise(x, *args):
        # 调用原始函数 f 计算结果
        fx = f(x, *args)
        # 记录函数调用次数
        f_raise._function_calls += 1
        # 如果计算结果为 NaN，则抛出 ValueError 异常
        if np.isnan(fx):
            msg = (f'The function value at x={x} is NaN; '
                   'solver cannot continue.')
            err = ValueError(msg)
            err._x = x
            err._function_calls = f_raise._function_calls
            raise err
        return fx

    # 初始化函数调用次数
    f_raise._function_calls = 0
    return f_raise

# 定义求解函数 newton，包括函数、初始值、导数、参数等
def newton(func, x0, fprime=None, args=(), tol=1.48e-8, maxiter=50,
           fprime2=None, x1=None, rtol=0.0,
           full_output=False, disp=True):
    """
    Find a root of a real or complex function using the Newton-Raphson
    (or secant or Halley's) method.

    Find a root of the scalar-valued function `func` given a nearby scalar
    starting point `x0`.
    The Newton-Raphson method is used if the derivative `fprime` of `func`
    is provided, otherwise the secant method is used. If the second order
    derivative `fprime2` of `func` is also provided, then Halley's method is
    used.

    If `x0` is a sequence with more than one item, `newton` returns an array:
    the roots of the function from each (scalar) starting point in `x0`.
    In this case, `func` must be vectorized to return a sequence or array of
    the same shape as its first argument. If `fprime` (`fprime2`) is given,
    then its return must also have the same shape: each element is the first
    (second) derivative of `func` with respect to its only variable evaluated
    at each element of its first argument.

    `newton` is for finding roots of a scalar-valued functions of a single
    variable. For problems involving several variables, see `root`.

    Parameters
    ----------
    func : callable
        The function whose root is wanted. It must be a function of a
        single variable of the form ``f(x,a,b,c...)``, where ``a,b,c...``
        are extra arguments that can be passed in the `args` parameter.
    x0 : float, sequence, or ndarray
        An initial estimate of the root that should be somewhere near the
        actual root. If not scalar, then `func` must be vectorized and return
        a sequence or array of the same shape as its first argument.
    fprime : callable, optional
        The derivative of the function when available and convenient. If it
        is None (default), then the secant method is used.
    args : tuple, optional
        Extra arguments to be used in the function call.
    tol : float, optional
        The allowable error of the root's value. If `func` is complex-valued,
        a larger `tol` is recommended as both the real and imaginary parts
        of `x` contribute to ``|x - x0|``.
    maxiter : int, optional
        Maximum number of iterations.
    fprime2 : callable, optional
        The second order derivative of the function when available and
        convenient. If it is None (default), then the normal Newton-Raphson
        or the secant method is used. If it is not None, then Halley's method
        is used.
    x1 : float, optional
        Another estimate of the root that should be somewhere near the
        actual root. Used if `fprime` is not provided.
    rtol : float, optional
        Tolerance (relative) for termination.
    """
    # full_output 参数，用于控制是否返回详细输出
    full_output : bool, optional
        # 如果 `full_output` 是 False（默认），则返回根。
        如果为 True 并且 `x0` 是标量，则返回值为 ``(x, r)``，其中 ``x``
        是根，``r`` 是一个 `RootResults` 对象。
        如果为 True 并且 `x0` 是非标量，则返回值为 ``(x, converged,
        zero_der)``（详见返回部分的说明）。

    # disp 参数，控制是否在算法不收敛时引发 RuntimeError
    disp : bool, optional
        # 如果为 True，在算法不收敛时引发 RuntimeError，错误消息中包含迭代次数和当前函数值。
        否则，收敛状态记录在 `RootResults` 返回对象中。
        当 `x0` 不是标量时被忽略。
        *注意：这与显示无关，然而，`disp` 关键字为了向后兼容性而不能重命名。*

    # 返回值说明部分

    # root 参数，返回函数零点的估计位置，可以是 float、序列或者 ndarray
    Returns
    -------
    root : float, sequence, or ndarray
        # 估计的函数零点位置。

    # r 参数，可选的 `RootResults` 对象，当 `full_output=True` 且 `x0` 是标量时存在
    r : `RootResults`, optional
        # 当 `full_output=True` 且 `x0` 是标量时存在。包含有关收敛信息的对象。特别地，
        ``r.converged`` 为 True 表示方法收敛。

    # converged 参数，可选的布尔值 ndarray，当 `full_output=True` 且 `x0` 是非标量时存在
    converged : ndarray of bool, optional
        # 当 `full_output=True` 且 `x0` 是非标量时存在。对于向量函数，指示哪些元素成功收敛。

    # zero_der 参数，可选的布尔值 ndarray，当 `full_output=True` 且 `x0` 是非标量时存在
    zero_der : ndarray of bool, optional
        # 当 `full_output=True` 且 `x0` 是非标量时存在。对于向量函数，指示哪些元素导数为零。
    # 如果容差 tol 小于等于 0，则抛出值错误异常
    if tol <= 0:
        raise ValueError("tol too small (%g <= 0)" % tol)
    # 将 maxiter 转换为整数索引，确保其为正数
    maxiter = operator.index(maxiter)
    if maxiter < 1:
        # 如果 maxiter 小于 1，则抛出值错误异常
        raise ValueError("maxiter must be greater than 0")
    # 如果 x0 的大小大于 1，则调用 _array_newton 函数进行数组形式的牛顿法计算
    if np.size(x0) > 1:
        return _array_newton(func, x0, fprime, args, tol, maxiter, fprime2,
                             full_output)

    # 将 x0 转换为浮点数。使用 np.asarray 是因为我们希望 x0 是一个 numpy 对象，
    # 而不是 Python 对象。例如，np.complex(1+1j) > 0 是可能的，但 (1 + 1j) > 0 会引发 TypeError
    x0 = np.asarray(x0)[()] * 1.0
    # 将初始值 p0 设为 x0
    p0 = x0
    # 初始化函数调用次数为 0
    funcalls = 0
    # 如果给定 fprime 不为 None，则使用牛顿-拉弗森法求解
    if fprime is not None:
        # 选择使用牛顿法作为求解方法
        method = "newton"
        # 迭代求解
        for itr in range(maxiter):
            # 计算当前点的函数值 fval
            fval = func(p0, *args)
            funcalls += 1
            # 如果 fval 等于 0，则找到了一个根，终止迭代
            if fval == 0:
                return _results_select(
                    full_output, (p0, funcalls, itr, _ECONVERGED), method)
            # 计算当前点的导数值 fder
            fder = fprime(p0, *args)
            funcalls += 1
            # 如果 fder 等于 0，牛顿法失败，引发异常
            if fder == 0:
                msg = "Derivative was zero."
                if disp:
                    msg += (
                        " Failed to converge after %d iterations, value is %s."
                        % (itr + 1, p0))
                    raise RuntimeError(msg)
                # 发出运行时警告并返回结果
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                return _results_select(
                    full_output, (p0, funcalls, itr + 1, _ECONVERR), method)
            # 计算牛顿步长
            newton_step = fval / fder
            # 如果有 fprime2，则执行 Halley 方法
            if fprime2:
                fder2 = fprime2(p0, *args)
                funcalls += 1
                method = "halley"
                # 计算调整项 adj
                adj = newton_step * fder2 / fder / 2
                # 只有当 adj 绝对值小于 1 时才执行 Halley 步骤
                # 理由：如果 1-adj < 0，则 Halley 方法会使 x 向 Newton 方法相反的方向移动，
                # 但如果 x 足够接近根，这种情况不会发生。
                if np.abs(adj) < 1:
                    newton_step /= 1.0 - adj
            # 计算新的近似根 p
            p = p0 - newton_step
            # 如果 p 和 p0 的相对误差小于给定的容忍度 rtol 或绝对误差小于 atol，则认为找到了根
            if np.isclose(p, p0, rtol=rtol, atol=tol):
                return _results_select(
                    full_output, (p, funcalls, itr + 1, _ECONVERGED), method)
            # 更新 p0 为当前迭代的近似根 p
            p0 = p
    else:
        # 使用割线法求解
        method = "secant"
        # 如果给定了初始值 x1，则使用该值作为 p1
        if x1 is not None:
            # 检查 x1 和 x0 是否相等，如果相等则抛出数值错误异常
            if x1 == x0:
                raise ValueError("x1 and x0 must be different")
            p1 = x1
        else:
            # 否则使用默认的 eps 值为 1e-4 来计算 p1
            eps = 1e-4
            p1 = x0 * (1 + eps)
            p1 += (eps if p1 >= 0 else -eps)
        # 计算 p0 和 p1 对应的函数值 q0 和 q1
        q0 = func(p0, *args)
        funcalls += 1
        q1 = func(p1, *args)
        funcalls += 1
        # 如果 q1 的绝对值小于 q0，则交换 p0 和 p1，以保证 q0 是绝对值较小的函数值
        if abs(q1) < abs(q0):
            p0, p1, q0, q1 = p1, p0, q1, q0
        # 迭代求解过程，最多进行 maxiter 次迭代
        for itr in range(maxiter):
            # 如果 q1 等于 q0，则检查是否 p1 不等于 p0，如果不等则抛出运行时错误异常
            if q1 == q0:
                if p1 != p0:
                    msg = "Tolerance of %s reached." % (p1 - p0)
                    if disp:
                        msg += (
                            " Failed to converge after %d iterations, value is %s."
                            % (itr + 1, p1))
                        raise RuntimeError(msg)
                    warnings.warn(msg, RuntimeWarning, stacklevel=2)
                # 计算 p 的值作为迭代结束的结果
                p = (p1 + p0) / 2.0
                return _results_select(
                    full_output, (p, funcalls, itr + 1, _ECONVERR), method)
            else:
                # 根据割线法计算下一个近似根 p
                if abs(q1) > abs(q0):
                    p = (-q0 / q1 * p1 + p0) / (1 - q0 / q1)
                else:
                    p = (-q1 / q0 * p0 + p1) / (1 - q1 / q0)
            # 如果 p 与 p1 的相对误差小于给定的相对容差 rtol 或绝对误差小于给定的绝对容差 tol，则迭代结束
            if np.isclose(p, p1, rtol=rtol, atol=tol):
                return _results_select(
                    full_output, (p, funcalls, itr + 1, _ECONVERGED), method)
            # 更新 p0, q0, p1, q1 为下一次迭代的值
            p0, q0 = p1, q1
            p1 = p
            q1 = func(p1, *args)
            funcalls += 1

    # 如果迭代过程中未达到收敛条件且 disp=True，则抛出运行时错误异常
    if disp:
        msg = ("Failed to converge after %d iterations, value is %s."
               % (itr + 1, p))
        raise RuntimeError(msg)

    # 返回迭代结束时的结果，包括最终的根 p、函数调用次数 funcalls、迭代次数 itr + 1 以及方法名称 method
    return _results_select(full_output, (p, funcalls, itr + 1, _ECONVERR), method)
# 一个针对数组的Newton、Halley和弦截法的向量化实现函数。

# 不要直接使用这个方法。当 `np.size(x0) > 1` 为真时，`newton` 方法会调用这个函数。详细信息请参见 `newton` 的文档。

# 显式地复制 `x0`，因为 `p` 将会被原地修改，但用户的数组不应该被改变。
p = np.array(x0, copy=True)

# 创建一个与 `p` 相同形状的布尔数组，用于标记失败的位置。
failures = np.ones_like(p, dtype=bool)

# 创建一个与 `failures` 相同形状的数组，用于标记非零导数的位置。
nz_der = np.ones_like(failures)

if fprime is not None:
    # 如果提供了导数函数，则使用牛顿-拉夫逊法
    for iteration in range(maxiter):
        # 计算函数值 `fval`
        fval = np.asarray(func(p, *args))

        # 如果所有的 `fval` 都是 0，则表示所有的根都已找到，终止迭代
        if not fval.any():
            failures = fval.astype(bool)
            break

        # 计算导数值 `fder`
        fder = np.asarray(fprime(p, *args))

        # 标记非零导数位置
        nz_der = (fder != 0)

        # 如果所有的导数都是零，则终止迭代
        if not nz_der.any():
            break

        # 计算牛顿步长
        dp = fval[nz_der] / fder[nz_der]

        if fprime2 is not None:
            # 如果提供了二阶导数函数，则进行 Halley 方法修正
            fder2 = np.asarray(fprime2(p, *args))
            dp = dp / (1.0 - 0.5 * dp * fder2[nz_der] / fder[nz_der])

        # 只更新非零导数位置的值
        p = np.asarray(p, dtype=np.result_type(p, dp, np.float64))
        p[nz_der] -= dp

        # 标记尚未收敛的位置
        failures[nz_der] = np.abs(dp) >= tol

        # 如果没有任何失败的位置（不包括零导数的情况），则终止迭代
        if not failures[nz_der].any():
            break

else:
    # 如果未提供导数函数，则使用弦截法
    dx = np.finfo(float).eps**0.33
    p1 = p * (1 + dx) + np.where(p >= 0, dx, -dx)
    q0 = np.asarray(func(p, *args))
    q1 = np.asarray(func(p1, *args))
    active = np.ones_like(p, dtype=bool)

    for iteration in range(maxiter):
        nz_der = (q1 != q0)

        # 如果所有的导数都是零，则终止迭代
        if not nz_der.any():
            p = (p1 + p) / 2.0
            break

        # 计算弦截步长
        dp = (q1 * (p1 - p))[nz_der] / (q1 - q0)[nz_der]

        # 只更新非零导数位置的值
        p = np.asarray(p, dtype=np.result_type(p, p1, dp, np.float64))
        p[nz_der] = p1[nz_der] - dp

        # 处理零导数的位置
        active_zero_der = ~nz_der & active
        p[active_zero_der] = (p1 + p)[active_zero_der] / 2.0
        active &= nz_der

        # 标记尚未收敛的位置
        failures[nz_der] = np.abs(dp) >= tol

        # 如果没有任何失败的位置（不包括零导数的情况），则终止迭代
        if not failures[nz_der].any():
            break

        # 更新迭代的数据
        p1, p = p, p1
        q0 = q1
        q1 = np.asarray(func(p1, *args))

# 标记所有零导数且尚未收敛的位置
zero_der = ~nz_der & failures
    # 如果 zero_der 中有任何元素为 True，则进入条件判断
    if zero_der.any():
        # 当存在 Secant 方法的警告
        if fprime is None:
            # 找到所有非零的 dp (p1 不等于 p)
            nonzero_dp = (p1 != p)
            # 找到既是零导数又是非零 dp 的情况
            zero_der_nz_dp = (zero_der & nonzero_dp)
            if zero_der_nz_dp.any():
                # 计算 RMS（均方根误差）
                rms = np.sqrt(
                    sum((p1[zero_der_nz_dp] - p[zero_der_nz_dp]) ** 2)
                )
                # 发出 RuntimeWarning 警告，指示 RMS 达到某个值
                warnings.warn(f'RMS of {rms:g} reached', RuntimeWarning, stacklevel=3)
        # 当存在 Newton 或 Halley 方法的警告
        else:
            # 决定是 'all' 还是 'some' 导数为零
            all_or_some = 'all' if zero_der.all() else 'some'
            msg = f'{all_or_some:s} derivatives were zero'
            # 发出 RuntimeWarning 警告，指示某些导数为零
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
    # 如果 failures 中有任何元素为 True，则进入条件判断
    elif failures.any():
        # 决定是 'all' 还是 'some' 未收敛
        all_or_some = 'all' if failures.all() else 'some'
        msg = f'{all_or_some:s} failed to converge after {maxiter:d} iterations'
        if failures.all():
            # 如果全部失败，则抛出 RuntimeError 异常
            raise RuntimeError(msg)
        # 发出 RuntimeWarning 警告，指示部分未收敛
        warnings.warn(msg, RuntimeWarning, stacklevel=3)

    # 如果 full_output 为 True，则返回详细结果
    if full_output:
        # 创建命名元组 result，包含 root（根）、converged（是否收敛）、zero_der（零导数情况）
        result = namedtuple('result', ('root', 'converged', 'zero_der'))
        # 更新 p 为 result 类型，包括收敛状态和零导数情况
        p = result(p, ~failures, zero_der)

    # 返回计算结果 p
    return p
# 使用二分法在区间 [a, b] 内寻找函数 f 的根。

def bisect(f, a, b, args=(),
           xtol=_xtol, rtol=_rtol, maxiter=_iter,
           full_output=False, disp=True):
    """
    Find root of a function within an interval using bisection.

    Basic bisection routine to find a root of the function `f` between the
    arguments `a` and `b`. `f(a)` and `f(b)` cannot have the same signs.
    Slow but sure.

    Parameters
    ----------
    f : function
        Python function returning a number.  `f` must be continuous, and
        f(a) and f(b) must have opposite signs.
    a : scalar
        One end of the bracketing interval [a,b].
    b : scalar
        The other end of the bracketing interval [a,b].
    xtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter must be positive.
    rtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter cannot be smaller than its default value of
        ``4*np.finfo(float).eps``.
    maxiter : int, optional
        If convergence is not achieved in `maxiter` iterations, an error is
        raised. Must be >= 0.
    args : tuple, optional
        Containing extra arguments for the function `f`.
        `f` is called by ``apply(f, (x)+args)``.
    full_output : bool, optional
        If `full_output` is False, the root is returned. If `full_output` is
        True, the return value is ``(x, r)``, where x is the root, and r is
        a `RootResults` object.
    disp : bool, optional
        If True, raise RuntimeError if the algorithm didn't converge.
        Otherwise, the convergence status is recorded in a `RootResults`
        return object.

    Returns
    -------
    root : float
        Root of `f` between `a` and `b`.
    r : `RootResults` (present if ``full_output = True``)
        Object containing information about the convergence. In particular,
        ``r.converged`` is True if the routine converged.

    Examples
    --------

    >>> def f(x):
    ...     return (x**2 - 1)

    >>> from scipy import optimize

    >>> root = optimize.bisect(f, 0, 2)
    >>> root
    1.0

    >>> root = optimize.bisect(f, -2, 0)
    >>> root
    -1.0

    See Also
    --------
    brentq, brenth, bisect, newton
    fixed_point : scalar fixed-point finder
    fsolve : n-dimensional root-finding

    """
    # 如果 args 不是 tuple 类型，则将其转换为 tuple
    if not isinstance(args, tuple):
        args = (args,)
    # 将 maxiter 转换为整数类型
    maxiter = operator.index(maxiter)
    # 如果 xtol 小于等于 0，则抛出 ValueError 异常
    if xtol <= 0:
        raise ValueError("xtol too small (%g <= 0)" % xtol)
    # 如果 rtol 小于默认值 _rtol，则抛出 ValueError 异常
    if rtol < _rtol:
        raise ValueError(f"rtol too small ({rtol:g} < {_rtol:g})")
    # 对函数 f 进行包装，处理 NaN 值的情况
    f = _wrap_nan_raise(f)
    # 调用底层的二分法实现，返回结果对象 r
    r = _zeros._bisect(f, a, b, xtol, rtol, maxiter, args, full_output, disp)
    # 根据 full_output 参数返回相应的结果
    return results_c(full_output, r, "bisect")
# 使用 Ridder 方法在给定区间 [a, b] 内寻找函数 f 的根
def ridder(f, a, b, args=(),
           xtol=_xtol, rtol=_rtol, maxiter=_iter,
           full_output=False, disp=True):
    """
    Find a root of a function in an interval using Ridder's method.

    Parameters
    ----------
    f : function
        Python function returning a number. f must be continuous, and f(a) and
        f(b) must have opposite signs.
    a : scalar
        One end of the bracketing interval [a,b].
    b : scalar
        The other end of the bracketing interval [a,b].
    xtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter must be positive.
    rtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter cannot be smaller than its default value of
        ``4*np.finfo(float).eps``.
    maxiter : int, optional
        If convergence is not achieved in `maxiter` iterations, an error is
        raised. Must be >= 0.
    args : tuple, optional
        Containing extra arguments for the function `f`.
        `f` is called by ``apply(f, (x)+args)``.
    full_output : bool, optional
        If `full_output` is False, the root is returned. If `full_output` is
        True, the return value is ``(x, r)``, where `x` is the root, and `r` is
        a `RootResults` object.
    disp : bool, optional
        If True, raise RuntimeError if the algorithm didn't converge.
        Otherwise, the convergence status is recorded in any `RootResults`
        return object.

    Returns
    -------
    root : float
        Root of `f` between `a` and `b`.
    r : `RootResults` (present if ``full_output = True``)
        Object containing information about the convergence.
        In particular, ``r.converged`` is True if the routine converged.

    See Also
    --------
    brentq, brenth, bisect, newton : 1-D root-finding
    fixed_point : scalar fixed-point finder

    Notes
    -----
    Uses [Ridders1979]_ method to find a root of the function `f` between the
    arguments `a` and `b`. Ridders' method is faster than bisection, but not
    generally as fast as the Brent routines. [Ridders1979]_ provides the
    classic description and source of the algorithm. A description can also be
    found in any recent edition of Numerical Recipes.

    The routine used here diverges slightly from standard presentations in
    order to be a bit more careful of tolerance.

    References
    ----------
    .. [Ridders1979]
       Ridders, C. F. J. "A New Algorithm for Computing a
       Single Root of a Real Continuous Function."
       IEEE Trans. Circuits Systems 26, 979-980, 1979.

    Examples
    --------

    >>> def f(x):
    ...     return (x**2 - 1)

    >>> from scipy import optimize

    >>> root = optimize.ridder(f, 0, 2)
    >>> root
    1.0

    >>> root = optimize.ridder(f, -2, 0)
    >>> root
    """

    # 初始化迭代器次数和迭代过程中的参数
    # 设置初始的区间和中点
    x1 = a
    x2 = b
    # 获取函数 f 在区间端点处的值
    f1 = f(x1, *args)
    f2 = f(x2, *args)

    # 迭代过程中记录的变量
    x0 = None
    r = None
    # 确保区间的两个端点具有相反的符号
    if np.sign(f1) == np.sign(f2):
        raise ValueError("Root must be bracketed in [a, b].")

    # 主循环执行 Ridder 方法的迭代过程
    for i in range(maxiter):
        # 计算区间中点
        x3 = 0.5 * (x1 + x2)
        f3 = f(x3, *args)

        # 计算 Ridder 方法的更新公式
        dx = x3 - x1
        if f3 == 0:
            root = x3
            break

        # 选择 Ridder 方法的下一个点
        s = np.sqrt(f3**2 - f1*f2)
        if s == 0:
            raise ValueError("Error: no finite steps available")
        dx = (x3 - x1) * f3 / s
        x4 = x3 + dx
        f4 = f(x4, *args)

        # 更新区间和函数值
        if x3 < x4:
            x1 = x3
            x2 = x4
            f1 = f3
            f2 = f4
        else:
            x1 = x4
            x2 = x3
            f1 = f4
            f2 = f3

        # 检查是否已经满足收敛条件
        if full_output:
            r = RootResults(root, x1, x2, f1, f2, xtol, rtol, iter=i)
            if np.allclose(root, x0, atol=xtol, rtol=rtol):
                r.converged = True
                break
        else:
            if np.allclose(x1, x2, atol=xtol, rtol=rtol):
                break
        x0 = root

    else:
        if disp:
            raise RuntimeError("Failed to converge after {} iterations.".format(maxiter))

    if full_output:
        return root, r
    else:
        return root
    # 检查 args 是否为元组，如果不是，则转换为只包含 args 的元组
    if not isinstance(args, tuple):
        args = (args,)
    # 将 maxiter 转换为整数索引
    maxiter = operator.index(maxiter)
    # 如果 xtol 小于等于 0，则抛出 ValueError 异常
    if xtol <= 0:
        raise ValueError("xtol too small (%g <= 0)" % xtol)
    # 如果 rtol 小于预设的最小相对误差 _rtol，则抛出 ValueError 异常
    if rtol < _rtol:
        raise ValueError(f"rtol too small ({rtol:g} < {_rtol:g})")
    # 对函数 f 进行装饰，处理 NaN 值的情况
    f = _wrap_nan_raise(f)
    # 调用 _zeros._ridder 方法执行 Ridder 方法进行根查找
    r = _zeros._ridder(f, a, b, xtol, rtol, maxiter, args, full_output, disp)
    # 返回 Ridder 方法的结果
    return results_c(full_output, r, "ridder")
# 定义函数 brentq，使用 Brent 方法在区间 [a, b] 中查找函数 f 的根
def brentq(f, a, b, args=(),
           xtol=_xtol, rtol=_rtol, maxiter=_iter,
           full_output=False, disp=True):
    """
    Find a root of a function in a bracketing interval using Brent's method.

    Uses the classic Brent's method to find a root of the function `f` on
    the sign changing interval [a , b]. Generally considered the best of the
    rootfinding routines here. It is a safe version of the secant method that
    uses inverse quadratic extrapolation. Brent's method combines root
    bracketing, interval bisection, and inverse quadratic interpolation. It is
    sometimes known as the van Wijngaarden-Dekker-Brent method. Brent (1973)
    claims convergence is guaranteed for functions computable within [a,b].

    [Brent1973]_ provides the classic description of the algorithm. Another
    description can be found in a recent edition of Numerical Recipes, including
    [PressEtal1992]_. A third description is at
    http://mathworld.wolfram.com/BrentsMethod.html. It should be easy to
    understand the algorithm just by reading our code. Our code diverges a bit
    from standard presentations: we choose a different formula for the
    extrapolation step.

    Parameters
    ----------
    f : function
        Python function returning a number. The function :math:`f`
        must be continuous, and :math:`f(a)` and :math:`f(b)` must
        have opposite signs.
    a : scalar
        One end of the bracketing interval :math:`[a, b]`.
    b : scalar
        The other end of the bracketing interval :math:`[a, b]`.
    xtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter must be positive. For nice functions, Brent's
        method will often satisfy the above condition with ``xtol/2``
        and ``rtol/2``. [Brent1973]_
    rtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter cannot be smaller than its default value of
        ``4*np.finfo(float).eps``. For nice functions, Brent's
        method will often satisfy the above condition with ``xtol/2``
        and ``rtol/2``. [Brent1973]_
    maxiter : int, optional
        If convergence is not achieved in `maxiter` iterations, an error is
        raised. Must be >= 0.
    args : tuple, optional
        Containing extra arguments for the function `f`.
        `f` is called by ``apply(f, (x)+args)``.
    full_output : bool, optional
        If `full_output` is False, the root is returned. If `full_output` is
        True, the return value is ``(x, r)``, where `x` is the root, and `r` is
        a `RootResults` object.
    disp : bool, optional
        If True, raise RuntimeError if the algorithm didn't converge.
        Otherwise, the convergence status is recorded in any `RootResults`
        return object.

    Returns
    -------
    # 如果 `args` 不是元组，则将其转换为元组
    if not isinstance(args, tuple):
        args = (args,)
    # 将 `maxiter` 强制转换为整数类型
    maxiter = operator.index(maxiter)
    # 如果 `xtol` 小于等于 0，则抛出数值错误异常
    if xtol <= 0:
        raise ValueError("xtol too small (%g <= 0)" % xtol)
    # 如果 `rtol` 小于预定义的 `_rtol` 值，则抛出数值错误异常
    if rtol < _rtol:
        raise ValueError(f"rtol too small ({rtol:g} < {_rtol:g})")
    # 将函数 `f` 包装以处理 NaN 值的异常
    f = _wrap_nan_raise(f)
    # 调用 `_zeros._brentq` 函数执行 Brent 方法进行求解
    r = _zeros._brentq(f, a, b, xtol, rtol, maxiter, args, full_output, disp)
    # 将求解结果转换为 `RootResults` 类，并返回使用 `brentq` 方法求解的结果
    return results_c(full_output, r, "brentq")
# 使用 Brent 方法查找在给定区间 [a, b] 内的函数 f 的根
def brenth(f, a, b, args=(),
           xtol=_xtol, rtol=_rtol, maxiter=_iter,
           full_output=False, disp=True):
    """Find a root of a function in a bracketing interval using Brent's
    method with hyperbolic extrapolation.

    A variation on the classic Brent routine to find a root of the function f
    between the arguments a and b that uses hyperbolic extrapolation instead of
    inverse quadratic extrapolation. Bus & Dekker (1975) guarantee convergence
    for this method, claiming that the upper bound of function evaluations here
    is 4 or 5 times that of bisection.
    f(a) and f(b) cannot have the same signs. Generally, on a par with the
    brent routine, but not as heavily tested. It is a safe version of the
    secant method that uses hyperbolic extrapolation.
    The version here is by Chuck Harris, and implements Algorithm M of
    [BusAndDekker1975]_, where further details (convergence properties,
    additional remarks and such) can be found

    Parameters
    ----------
    f : function
        Python function returning a number. f must be continuous, and f(a) and
        f(b) must have opposite signs.
    a : scalar
        One end of the bracketing interval [a,b].
    b : scalar
        The other end of the bracketing interval [a,b].
    xtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter must be positive. As with `brentq`, for nice
        functions the method will often satisfy the above condition
        with ``xtol/2`` and ``rtol/2``.
    rtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter cannot be smaller than its default value of
        ``4*np.finfo(float).eps``. As with `brentq`, for nice functions
        the method will often satisfy the above condition with
        ``xtol/2`` and ``rtol/2``.
    maxiter : int, optional
        If convergence is not achieved in `maxiter` iterations, an error is
        raised. Must be >= 0.
    args : tuple, optional
        Containing extra arguments for the function `f`.
        `f` is called by ``apply(f, (x)+args)``.
    full_output : bool, optional
        If `full_output` is False, the root is returned. If `full_output` is
        True, the return value is ``(x, r)``, where `x` is the root, and `r` is
        a `RootResults` object.
    disp : bool, optional
        If True, raise RuntimeError if the algorithm didn't converge.
        Otherwise, the convergence status is recorded in any `RootResults`
        return object.

    Returns
    -------
    root : float
        Root of `f` between `a` and `b`.
    r : `RootResults` (present if ``full_output = True``)
        Object containing information about the convergence. In particular,
        ``r.converged`` is True if the routine converged.

    See Also
    --------
    """
    fmin, fmin_powell, fmin_cg, fmin_bfgs, fmin_ncg : multivariate local optimizers
    leastsq : nonlinear least squares minimizer
    fmin_l_bfgs_b, fmin_tnc, fmin_cobyla : constrained multivariate optimizers
    basinhopping, differential_evolution, brute : global optimizers
    fminbound, brent, golden, bracket : local scalar minimizers
    fsolve : N-D root-finding
    brentq, brenth, ridder, bisect, newton : 1-D root-finding
    fixed_point : scalar fixed-point finder

    References
    ----------
    .. [BusAndDekker1975]
       Bus, J. C. P., Dekker, T. J.,
       "Two Efficient Algorithms with Guaranteed Convergence for Finding a Zero
       of a Function", ACM Transactions on Mathematical Software, Vol. 1, Issue
       4, Dec. 1975, pp. 330-345. Section 3: "Algorithm M".
       :doi:`10.1145/355656.355659`

    Examples
    --------
    >>> def f(x):
    ...     return (x**2 - 1)

    >>> from scipy import optimize

    >>> root = optimize.brenth(f, -2, 0)
    >>> root
    -1.0

    >>> root = optimize.brenth(f, 0, 2)
    >>> root
    1.0

    """
    # 如果参数不是元组，则转换为元组
    if not isinstance(args, tuple):
        args = (args,)
    # 将 maxiter 强制转换为整数
    maxiter = operator.index(maxiter)
    # 如果 xtol 小于等于 0，则抛出 ValueError 异常
    if xtol <= 0:
        raise ValueError("xtol too small (%g <= 0)" % xtol)
    # 如果 rtol 小于默认的 _rtol，则抛出 ValueError 异常
    if rtol < _rtol:
        raise ValueError(f"rtol too small ({rtol:g} < {_rtol:g})")
    # 对函数 f 进行封装以处理 NaN 值
    f = _wrap_nan_raise(f)
    # 调用底层函数 _brenth 执行 Brent 方法求解
    r = _zeros._brenth(f, a, b, xtol, rtol, maxiter, args, full_output, disp)
    # 返回经结果处理后的最终输出
    return results_c(full_output, r, "brenth")
################################
# TOMS "Algorithm 748: Enclosing Zeros of Continuous Functions", by
#  Alefeld, G. E. and Potra, F. A. and Shi, Yixun,
#  See [1]

# 定义一个函数，检查一组浮点数是否满足条件，确保不是 None、不是 0、全部是有限数，并且彼此之间不非常接近
def _notclose(fs, rtol=_rtol, atol=_xtol):
    notclosefvals = (
            all(fs) and all(np.isfinite(fs)) and
            not any(any(np.isclose(_f, fs[i + 1:], rtol=rtol, atol=atol))
                    for i, _f in enumerate(fs[:-1])))
    return notclosefvals


# 定义一个函数，执行一步割线法（secant method），在数值上略加注意
def _secant(xvals, fvals):
    """Perform a secant step, taking a little care"""
    # 割线法有多种数学上等价的表达式
    # x2 = x0 - (x1 - x0)/(f1 - f0) * f0
    #    = x1 - (x1 - x0)/(f1 - f0) * f1
    #    = (-x1 * f0 + x0 * f1) / (f1 - f0)
    #    = (-f0 / f1 * x1 + x0) / (1 - f0 / f1)
    #    = (-f1 / f0 * x0 + x1) / (1 - f1 / f0)
    x0, x1 = xvals[:2]
    f0, f1 = fvals[:2]
    if f0 == f1:
        return np.nan
    if np.abs(f1) > np.abs(f0):
        x2 = (-f0 / f1 * x1 + x0) / (1 - f0 / f1)
    else:
        x2 = (-f1 / f0 * x0 + x1) / (1 - f1 / f0)
    return x2


# 定义一个函数，更新一个区间（bracket），给定新的点(c, fc)，并返回被丢弃的端点
def _update_bracket(ab, fab, c, fc):
    """Update a bracket given (c, fc), return the discarded endpoints."""
    fa, fb = fab
    # 确定要丢弃的端点的索引位置
    idx = (0 if np.sign(fa) * np.sign(fc) > 0 else 1)
    rx, rfx = ab[idx], fab[idx]
    # 更新端点和对应的函数值
    fab[idx] = fc
    ab[idx] = c
    return rx, rfx


# 计算给定xvals和fvals对的分裂差商（divided differences）矩阵
def _compute_divided_differences(xvals, fvals, N=None, full=True, forward=True):
    """Return a matrix of divided differences for the xvals, fvals pairs

    DD[i, j] = f[x_{i-j}, ..., x_i] for 0 <= j <= i

    If full is False, just return the main diagonal(or last row):
      f[a], f[a, b] and f[a, b, c].
    If forward is False, return f[c], f[b, c], f[a, b, c]."""
    if full:
        if forward:
            xvals = np.asarray(xvals)
        else:
            xvals = np.array(xvals)[::-1]
        M = len(xvals)
        N = M if N is None else min(N, M)
        DD = np.zeros([M, N])
        DD[:, 0] = fvals[:]
        for i in range(1, N):
            DD[i:, i] = (np.diff(DD[i - 1:, i - 1]) /
                         (xvals[i:] - xvals[:M - i]))
        return DD

    xvals = np.asarray(xvals)
    dd = np.array(fvals)
    row = np.array(fvals)
    idx2Use = (0 if forward else -1)
    dd[0] = fvals[idx2Use]
    for i in range(1, len(xvals)):
        denom = xvals[i:i + len(row) - 1] - xvals[:len(row) - 1]
        row = np.diff(row)[:] / denom
        dd[i] = row[idx2Use]
    return dd


# 计算通过指定位置点xvals和fvals的多项式p(x)在x处的值
def _interpolated_poly(xvals, fvals, x):
    """Compute p(x) for the polynomial passing through the specified locations.

    Use Neville's algorithm to compute p(x) where p is the minimal degree
    polynomial passing through the points xvals, fvals"""
    xvals = np.asarray(xvals)
    N = len(xvals)
    Q = np.zeros([N, N])
    D = np.zeros([N, N])
    Q[:, 0] = fvals[:]
    D[:, 0] = fvals[:]
    # 对 k 从 1 到 N-1 进行循环，执行以下操作：
    for k in range(1, N):
        # 计算 alpha，即 D[k:, k-1] 减去 Q[k-1:N-1, k-1] 的结果
        alpha = D[k:, k - 1] - Q[k - 1:N - 1, k - 1]
        # 计算 diffik，即 xvals 的从第 0 到 N-k-1 项减去从第 k 到 N-1 项的结果
        diffik = xvals[0:N - k] - xvals[k:N]
        # 计算 Q[k:, k]，使用公式 (xvals[k:] - x) / diffik * alpha
        Q[k:, k] = (xvals[k:] - x) / diffik * alpha
        # 计算 D[k:, k]，使用公式 (xvals[:N-k] - x) / diffik * alpha
        D[k:, k] = (xvals[:N - k] - x) / diffik * alpha
    # 返回 Q[-1, 1:] 的和加上 Q[-1, 0] 的结果
    # 预期随着 x 接近根，Q[-1, 1:] 相对于 Q[-1, 0] 将会很小
    return np.sum(Q[-1, 1:]) + Q[-1, 0]
# 定义一个函数 `_inverse_poly_zero`，执行逆立方插值，将函数值映射为 x 值
"""Inverse cubic interpolation f-values -> x-values

Given four points (fa, a), (fb, b), (fc, c), (fd, d) with
fa, fb, fc, fd all distinct, find poly IP(y) through the 4 points
and compute x=IP(0).
"""
return _interpolated_poly([fa, fb, fc, fd], [a, b, c, d], 0)


# 定义一个函数 `_newton_quadratic`，应用类似牛顿-拉弗森法的步骤，使用分裂差分近似 f'
"""Apply Newton-Raphson like steps, using divided differences to approximate f'

ab is a real interval [a, b] containing a root,
fab holds the real values of f(a), f(b)
d is a real number outside [ab, b]
k is the number of steps to apply
"""
a, b = ab
fa, fb = fab
_, B, A = _compute_divided_differences([a, b, d], [fa, fb, fd],
                                       forward=True, full=False)

# _P  is the quadratic polynomial through the 3 points
def _P(x):
    # Horner evaluation of fa + B * (x - a) + A * (x - a) * (x - b)
    return (A * (x - b) + B) * (x - a) + fa

if A == 0:
    r = a - fa / B
else:
    r = (a if np.sign(A) * np.sign(fa) > 0 else b)
    # Apply k Newton-Raphson steps to _P(x), starting from x=r
    for i in range(k):
        r1 = r - _P(r) / (B + A * (2 * r - a - b))
        if not (ab[0] < r1 < ab[1]):
            if (ab[0] < r < ab[1]):
                return r
            r = sum(ab) / 2.0
            break
        r = r1

return r


# 定义一个类 TOMS748Solver，解决 f(x, *args) == 0 问题，使用 Alefeld, Potro & Shi 的 Algorithm748
class TOMS748Solver:
    """Solve f(x, *args) == 0 using Algorithm748 of Alefeld, Potro & Shi.
    """
    _MU = 0.5
    _K_MIN = 1
    _K_MAX = 100  # A very high value for real usage. Expect 1, 2, maybe 3.

    def __init__(self):
        self.f = None
        self.args = None
        self.function_calls = 0
        self.iterations = 0
        self.k = 2
        # ab=[a,b] is a global interval containing a root
        self.ab = [np.nan, np.nan]
        # fab is function values at a, b
        self.fab = [np.nan, np.nan]
        self.d = None
        self.fd = None
        self.e = None
        self.fe = None
        self.disp = False
        self.xtol = _xtol
        self.rtol = _rtol
        self.maxiter = _iter

    def configure(self, xtol, rtol, maxiter, disp, k):
        self.disp = disp
        self.xtol = xtol
        self.rtol = rtol
        self.maxiter = maxiter
        # Silently replace a low value of k with 1
        self.k = max(k, self._K_MIN)
        # Noisily replace a high value of k with self._K_MAX
        if self.k > self._K_MAX:
            msg = "toms748: Overriding k: ->%d" % self._K_MAX
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
            self.k = self._K_MAX
    # 调用用户提供的函数并更新计数
    def _callf(self, x, error=True):
        """Call the user-supplied function, update book-keeping"""
        # 调用函数计算 f(x)
        fx = self.f(x, *self.args)
        # 增加函数调用次数
        self.function_calls += 1
        # 如果 fx 不是有限值且 error 参数为 True，则抛出异常
        if not np.isfinite(fx) and error:
            raise ValueError(f"Invalid function value: f({x:f}) -> {fx} ")
        return fx

    # 打包结果和统计信息为元组
    def get_result(self, x, flag=_ECONVERGED):
        r"""Package the result and statistics into a tuple."""
        return (x, self.function_calls, self.iterations, flag)

    # 更新区间
    def _update_bracket(self, c, fc):
        return _update_bracket(self.ab, self.fab, c, fc)

    # 准备进行迭代
    def start(self, f, a, b, args=()):
        r"""Prepare for the iterations."""
        self.function_calls = 0
        self.iterations = 0

        # 设置函数 f 和参数 args
        self.f = f
        self.args = args
        # 将区间 [a, b] 存入 self.ab
        self.ab[:] = [a, b]
        # 检查 a 的有效性
        if not np.isfinite(a) or np.imag(a) != 0:
            raise ValueError("Invalid x value: %s " % (a))
        # 检查 b 的有效性
        if not np.isfinite(b) or np.imag(b) != 0:
            raise ValueError("Invalid x value: %s " % (b))

        # 计算 f(a)
        fa = self._callf(a)
        # 检查 f(a) 的有效性
        if not np.isfinite(fa) or np.imag(fa) != 0:
            raise ValueError(f"Invalid function value: f({a:f}) -> {fa} ")
        # 如果 f(a) 为零，返回收敛状态和 a
        if fa == 0:
            return _ECONVERGED, a
        # 计算 f(b)
        fb = self._callf(b)
        # 检查 f(b) 的有效性
        if not np.isfinite(fb) or np.imag(fb) != 0:
            raise ValueError(f"Invalid function value: f({b:f}) -> {fb} ")
        # 如果 f(b) 为零，返回收敛状态和 b
        if fb == 0:
            return _ECONVERGED, b

        # 检查 f(a) 和 f(b) 的符号是否相同
        if np.sign(fb) * np.sign(fa) > 0:
            raise ValueError("f(a) and f(b) must have different signs, but "
                             f"f({a:e})={fa:e}, f({b:e})={fb:e} ")
        # 将 f(a) 和 f(b) 存入 self.fab
        self.fab[:] = [fa, fb]

        # 返回进行中状态和区间中点的估计值
        return _EINPROGRESS, sum(self.ab) / 2.0

    # 获取当前状态
    def get_status(self):
        """Determine the current status."""
        # 获取当前区间端点 a 和 b
        a, b = self.ab[:2]
        # 如果 a 和 b 接近，返回收敛状态和区间中点的估计值
        if np.isclose(a, b, rtol=self.rtol, atol=self.xtol):
            return _ECONVERGED, sum(self.ab) / 2.0
        # 如果迭代次数达到最大限制，返回收敛失败状态和区间中点的估计值
        if self.iterations >= self.maxiter:
            return _ECONVERR, sum(self.ab) / 2.0
        # 否则返回进行中状态和区间中点的估计值
        return _EINPROGRESS, sum(self.ab) / 2.0
    def solve(self, f, a, b, args=(),
              xtol=_xtol, rtol=_rtol, k=2, maxiter=_iter, disp=True):
        r"""Solve f(x) = 0 given an interval containing a root."""
        # 配置求解器的参数和设置
        self.configure(xtol=xtol, rtol=rtol, maxiter=maxiter, disp=disp, k=k)
        
        # 开始求解，获取初始状态和估计的根
        status, xn = self.start(f, a, b, args)
        
        # 如果已经收敛到解
        if status == _ECONVERGED:
            return self.get_result(xn)

        # 第一步使用切线法确定第三个点c
        c = _secant(self.ab, self.fab)
        
        # 如果c不在[a, b]范围内，则取a和b的中点作为c
        if not self.ab[0] < c < self.ab[1]:
            c = sum(self.ab) / 2.0
        
        # 计算c点处的函数值fc
        fc = self._callf(c)
        
        # 如果c点处的函数值为0，则直接返回c作为根
        if fc == 0:
            return self.get_result(c)

        # 更新bracket的边界和对应的函数值
        self.d, self.fd = self._update_bracket(c, fc)
        
        # 初始化e和fe为None
        self.e, self.fe = None, None
        
        # 迭代次数加一
        self.iterations += 1

        # 开始迭代直到收敛或者迭代失败
        while True:
            status, xn = self.iterate()
            if status == _ECONVERGED:
                return self.get_result(xn)
            if status == _ECONVERR:
                fmt = "Failed to converge after %d iterations, bracket is %s"
                # 如果设置了disp为True，则抛出运行时错误
                if disp:
                    msg = fmt % (self.iterations + 1, self.ab)
                    raise RuntimeError(msg)
                # 否则返回最后一个估计的根和ECONVERR状态
                return self.get_result(xn, _ECONVERR)
# 定义一个函数 toms748，使用 TOMS 算法 748 方法来寻找函数的根
def toms748(f, a, b, args=(), k=1,
            xtol=_xtol, rtol=_rtol, maxiter=_iter,
            full_output=False, disp=True):
    """
    Find a root using TOMS Algorithm 748 method.

    Implements the Algorithm 748 method of Alefeld, Potro and Shi to find a
    root of the function `f` on the interval ``[a , b]``, where ``f(a)`` and
    `f(b)` must have opposite signs.

    It uses a mixture of inverse cubic interpolation and
    "Newton-quadratic" steps. [APS1995].

    Parameters
    ----------
    f : function
        Python function returning a scalar. The function :math:`f`
        must be continuous, and :math:`f(a)` and :math:`f(b)`
        have opposite signs.
    a : scalar,
        lower boundary of the search interval
    b : scalar,
        upper boundary of the search interval
    args : tuple, optional
        containing extra arguments for the function `f`.
        `f` is called by ``f(x, *args)``.
    k : int, optional
        The number of Newton quadratic steps to perform each
        iteration. ``k>=1``.
    xtol : scalar, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter must be positive.
    rtol : scalar, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root.
    maxiter : int, optional
        If convergence is not achieved in `maxiter` iterations, an error is
        raised. Must be >= 0.
    full_output : bool, optional
        If `full_output` is False, the root is returned. If `full_output` is
        True, the return value is ``(x, r)``, where `x` is the root, and `r` is
        a `RootResults` object.
    disp : bool, optional
        If True, raise RuntimeError if the algorithm didn't converge.
        Otherwise, the convergence status is recorded in the `RootResults`
        return object.

    Returns
    -------
    root : float
        Approximate root of `f`
    r : `RootResults` (present if ``full_output = True``)
        Object containing information about the convergence. In particular,
        ``r.converged`` is True if the routine converged.

    See Also
    --------
    brentq, brenth, ridder, bisect, newton
    fsolve : find roots in N dimensions.

    Notes
    -----
    `f` must be continuous.
    Algorithm 748 with ``k=2`` is asymptotically the most efficient
    algorithm known for finding roots of a four times continuously
    differentiable function.
    In contrast with Brent's algorithm, which may only decrease the length of
    the enclosing bracket on the last step, Algorithm 748 decreases it each
    iteration with the same asymptotic efficiency as it finds the root.

    For easy statement of efficiency indices, assume that `f` has 4
    continuouous deriviatives.
    For ``k=1``, the convergence order is at least 2.7, and with about
    """
    """
    如果输入的容差 xtol 小于等于 0，则抛出值错误异常。
    """
    if xtol <= 0:
        raise ValueError("xtol too small (%g <= 0)" % xtol)
    
    """
    如果相对容差 rtol 小于全局定义的 _rtol 的四分之一，则抛出值错误异常。
    """
    if rtol < _rtol / 4:
        raise ValueError(f"rtol too small ({rtol:g} < {_rtol/4:g})")
    
    """
    将最大迭代次数 maxiter 转换为整数索引。如果 maxiter 小于 1，则抛出值错误异常。
    """
    maxiter = operator.index(maxiter)
    if maxiter < 1:
        raise ValueError("maxiter must be greater than 0")
    
    """
    如果参数 a 不是有限的，则抛出值错误异常。
    """
    if not np.isfinite(a):
        raise ValueError("a is not finite %s" % a)
    
    """
    如果参数 b 不是有限的，则抛出值错误异常。
    """
    if not np.isfinite(b):
        raise ValueError("b is not finite %s" % b)
    
    """
    如果 a 大于等于 b，则抛出值错误异常，说明 a 和 b 不构成一个有效的区间 [a, b]。
    """
    if a >= b:
        raise ValueError(f"a and b are not an interval [{a}, {b}]")
    
    """
    如果 k 不大于等于 1，则抛出值错误异常。
    """
    if not k >= 1:
        raise ValueError("k too small (%s < 1)" % k)
    
    """
    如果参数 args 不是元组，则将其转换为单元素的元组。
    """
    if not isinstance(args, tuple):
        args = (args,)
    
    """
    将函数 f 使用 _wrap_nan_raise 进行包装，以处理 NaN 值异常。
    """
    f = _wrap_nan_raise(f)
    
    """
    创建 TOMS748Solver 的实例 solver。
    """
    solver = TOMS748Solver()
    
    """
    使用 solver 对象解决函数 f 在区间 [a, b] 上的根，并返回结果。
    """
    result = solver.solve(f, a, b, args=args, k=k, xtol=xtol, rtol=rtol,
                          maxiter=maxiter, disp=disp)
    
    """
    将结果解析为 x（根）、function_calls（函数调用次数）、iterations（迭代次数）和 flag（收敛标志）。
    """
    x, function_calls, iterations, flag = result
    
    """
    根据 full_output 参数，选择要返回的结果，这里返回 (x, function_calls, iterations, flag)。
    """
    return _results_select(full_output, (x, function_calls, iterations, flag),
                           "toms748")
```