# `D:\src\scipysrc\scipy\scipy\optimize\_direct_py.py`

```
from __future__ import annotations
from typing import (  # noqa: UP035
    Any, Callable, Iterable, TYPE_CHECKING
)

import numpy as np
from scipy.optimize import OptimizeResult
from ._constraints import old_bound_to_new, Bounds
from ._direct import direct as _direct  # type: ignore

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ['direct']

# 错误信息列表，用于记录优化过程中可能出现的错误情况
ERROR_MESSAGES = (
    "Number of function evaluations done is larger than maxfun={}",
    "Number of iterations is larger than maxiter={}",
    "u[i] < l[i] for some i",
    "maxfun is too large",
    "Initialization failed",
    "There was an error in the creation of the sample points",
    "An error occurred while the function was sampled",
    "Maximum number of levels has been reached.",
    "Forced stop",
    "Invalid arguments",
    "Out of memory",
)

# 成功信息列表，用于记录优化过程中可能出现的成功情况
SUCCESS_MESSAGES = (
    ("The best function value found is within a relative error={} "
     "of the (known) global optimum f_min"),
    ("The volume of the hyperrectangle containing the lowest function value "
     "found is below vol_tol={}"),
    ("The side length measure of the hyperrectangle containing the lowest "
     "function value found is below len_tol={}"),
)

# 定义 direct 函数，用 DIRECT 算法寻找函数的全局最小值
def direct(
    func: Callable[[npt.ArrayLike, tuple[Any]], float],
    bounds: Iterable | Bounds,
    *,
    args: tuple = (),
    eps: float = 1e-4,
    maxfun: int | None = None,
    maxiter: int = 1000,
    locally_biased: bool = True,
    f_min: float = -np.inf,
    f_min_rtol: float = 1e-4,
    vol_tol: float = 1e-16,
    len_tol: float = 1e-6,
    callback: Callable[[npt.ArrayLike], None] | None = None
) -> OptimizeResult:
    """
    Finds the global minimum of a function using the
    DIRECT algorithm.

    Parameters
    ----------
    func : callable
        The objective function to be minimized.
        ``func(x, *args) -> float``
        where ``x`` is an 1-D array with shape (n,) and ``args`` is a tuple of
        the fixed parameters needed to completely specify the function.
    bounds : sequence or `Bounds`
        Bounds for variables. There are two ways to specify the bounds:

        1. Instance of `Bounds` class.
        2. ``(min, max)`` pairs for each element in ``x``.

    args : tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.
    eps : float, optional
        Minimal required difference of the objective function values
        between the current best hyperrectangle and the next potentially
        optimal hyperrectangle to be divided. In consequence, `eps` serves as a
        tradeoff between local and global search: the smaller, the more local
        the search becomes. Default is 1e-4.
    """
    # 函数体内部实现略过，不在这里进行具体的注释
    pass
    # 最大函数评估次数的上限，如果为 `None`，将自动设置为 `1000 * N`，其中 `N` 表示维度数量。
    # 如果问题非常高维且 `max_fun` 过高，会被限制以控制 DIRECT 算法的内存使用约为 1 GiB。
    # 默认为 `None`。
    maxfun : int or None, optional

    # 最大迭代次数。默认为 1000。
    maxiter : int, optional

    # 是否使用局部偏置变体 DIRECT_L 算法。如果为 `True`（默认），使用局部偏置变体；
    # 如果为 `False`，使用原始的无偏 DIRECT 算法。对于存在许多局部最小值的难题，建议设为 `False`。
    locally_biased : bool, optional

    # 全局最优解的函数值。仅在已知全局最优解时设置此值。默认为 `-np.inf`，因此此终止准则被禁用。
    f_min : float, optional

    # 当当前最佳最小值 `f` 与提供的全局最小值 `f_min` 的相对误差小于 `f_min_rtol` 时终止优化。
    # 仅当 `f_min` 被设置时才使用此参数。必须介于 0 和 1 之间。默认为 1e-4。
    f_min_rtol : float, optional

    # 当包含最低函数值的超矩形的体积小于完整搜索空间的 `vol_tol` 时终止优化。
    # 必须介于 0 和 1 之间。默认为 1e-16。
    vol_tol : float, optional

    # 如果 `locally_biased=True`，则当包含最低函数值的超矩形的标准化最大边长的一半小于 `len_tol` 时终止优化。
    # 如果 `locally_biased=False`，则当包含最低函数值的超矩形的标准化对角线的一半小于 `len_tol` 时终止优化。
    # 必须介于 0 和 1 之间。默认为 1e-6。
    len_tol : float, optional

    # 回调函数，其签名为 `callback(xk)`，其中 `xk` 表示迄今找到的最佳函数值。

    Returns
    -------
    res : OptimizeResult
        优化结果，表示为 `OptimizeResult` 对象。
        重要属性包括：`x` 解数组，`success` 表示优化器是否成功退出的布尔标志，
        `message` 描述终止原因的字符串。查看 `OptimizeResult` 获取其他属性的描述。

    Notes
    -----
    DIviding RECTangles (DIRECT) 是一种确定性全局优化算法，能够通过在搜索空间中采样潜在解来最小化黑盒函数，
    其变量受到下界和上界约束 [1]_。该算法从
    normalising the search space to an n-dimensional unit hypercube.
    It samples the function at the center of this hypercube and at 2n
    (n is the number of variables) more points, 2 in each coordinate
    direction. Using these function values, DIRECT then divides the
    domain into hyperrectangles, each having exactly one of the sampling
    points as its center. In each iteration, DIRECT chooses, using the `eps`
    parameter which defaults to 1e-4, some of the existing hyperrectangles
    to be further divided. This division process continues until either the
    maximum number of iterations or maximum function evaluations allowed
    are exceeded, or the hyperrectangle containing the minimal value found
    so far becomes small enough. If `f_min` is specified, the optimization
    will stop once this function value is reached within a relative tolerance.
    The locally biased variant of DIRECT (originally called DIRECT_L) [2]_ is
    used by default. It makes the search more locally biased and more
    efficient for cases with only a few local minima.

    A note about termination criteria: `vol_tol` refers to the volume of the
    hyperrectangle containing the lowest function value found so far. This
    volume decreases exponentially with increasing dimensionality of the
    problem. Therefore `vol_tol` should be decreased to avoid premature
    termination of the algorithm for higher dimensions. This does not hold
    for `len_tol`: it refers either to half of the maximal side length
    (for ``locally_biased=True``) or half of the diagonal of the
    hyperrectangle (for ``locally_biased=False``).

    This code is based on the DIRECT 2.0.4 Fortran code by Gablonsky et al. at
    https://ctk.math.ncsu.edu/SOFTWARE/DIRECTv204.tar.gz .
    This original version was initially converted via f2c and then cleaned up
    and reorganized by Steven G. Johnson, August 2007, for the NLopt project.
    The `direct` function wraps the C implementation.

    .. versionadded:: 1.9.0

    References
    ----------
    .. [1] Jones, D.R., Perttunen, C.D. & Stuckman, B.E. Lipschitzian
        optimization without the Lipschitz constant. J Optim Theory Appl
        79, 157-181 (1993).
    .. [2] Gablonsky, J., Kelley, C. A Locally-Biased form of the DIRECT
        Algorithm. Journal of Global Optimization 21, 27-37 (2001).

    Examples
    --------
    The following example is a 2-D problem with four local minima: minimizing
    the Styblinski-Tang function
    (https://en.wikipedia.org/wiki/Test_functions_for_optimization).

    >>> from scipy.optimize import direct, Bounds
    >>> def styblinski_tang(pos):
    ...     x, y = pos
    ...     return 0.5 * (x**4 - 16*x**2 + 5*x + y**4 - 16*y**2 + 5*y)
    >>> bounds = Bounds([-4., -4.], [4., 4.])
    >>> result = direct(styblinski_tang, bounds)
    >>> result.x, result.fun, result.nfev
    array([-2.90321597, -2.90321597]), -78.3323279095383, 2011


注释：
    The correct global minimum was found but with a huge number of function
    evaluations (2011). Loosening the termination tolerances `vol_tol` and
    `len_tol` can be used to stop DIRECT earlier.

    >>> result = direct(styblinski_tang, bounds, len_tol=1e-3)
    >>> result.x, result.fun, result.nfev
    array([-2.9044353, -2.9044353]), -78.33230330754142, 207

    """
    # 将边界转换为新的 Bounds 类（如果需要）
    if not isinstance(bounds, Bounds):
        if isinstance(bounds, list) or isinstance(bounds, tuple):
            lb, ub = old_bound_to_new(bounds)
            bounds = Bounds(lb, ub)
        else:
            message = ("bounds must be a sequence or "
                       "instance of Bounds class")
            raise ValueError(message)

    # 将边界转换为连续的 np.ndarray，并确保是 np.float64 类型
    lb = np.ascontiguousarray(bounds.lb, dtype=np.float64)
    ub = np.ascontiguousarray(bounds.ub, dtype=np.float64)

    # 验证边界是否合理
    # 检查下限是否小于上限
    if not np.all(lb < ub):
        raise ValueError('Bounds are not consistent min < max')
    # 检查是否存在无穷大
    if (np.any(np.isinf(lb)) or np.any(np.isinf(ub))):
        raise ValueError("Bounds must not be inf.")

    # 验证容差值
    if (vol_tol < 0 or vol_tol > 1):
        raise ValueError("vol_tol must be between 0 and 1.")
    if (len_tol < 0 or len_tol > 1):
        raise ValueError("len_tol must be between 0 and 1.")
    if (f_min_rtol < 0 or f_min_rtol > 1):
        raise ValueError("f_min_rtol must be between 0 and 1.")

    # 验证 maxfun 和 maxiter
    if maxfun is None:
        maxfun = 1000 * lb.shape[0]
    if not isinstance(maxfun, int):
        raise ValueError("maxfun must be of type int.")
    if maxfun < 0:
        raise ValueError("maxfun must be > 0.")
    if not isinstance(maxiter, int):
        raise ValueError("maxiter must be of type int.")
    if maxiter < 0:
        raise ValueError("maxiter must be > 0.")

    # 验证布尔类型参数
    if not isinstance(locally_biased, bool):
        raise ValueError("locally_biased must be True or False.")

    # 定义内部函数 _func_wrap，用于处理函数调用和返回值
    def _func_wrap(x, args=None):
        x = np.asarray(x)
        if args is None:
            f = func(x)
        else:
            f = func(x, *args)
        # 确保返回值为浮点数
        return np.asarray(f).item()

    # 调用 _direct 函数执行优化
    # TODO: fix disp argument
    x, fun, ret_code, nfev, nit = _direct(
        _func_wrap,
        np.asarray(lb), np.asarray(ub),
        args,
        False, eps, maxfun, maxiter,
        locally_biased,
        f_min, f_min_rtol,
        vol_tol, len_tol, callback
    )

    # 格式化消息内容，根据返回代码选择对应的消息
    format_val = (maxfun, maxiter, f_min_rtol, vol_tol, len_tol)
    if ret_code > 2:
        message = SUCCESS_MESSAGES[ret_code - 3].format(
                    format_val[ret_code - 1])
    elif 0 < ret_code <= 2:
        message = ERROR_MESSAGES[ret_code - 1].format(format_val[ret_code - 1])
    elif 0 > ret_code > -100:
        message = ERROR_MESSAGES[abs(ret_code) + 1]
    # 如果 ret_code 大于等于 3，则从 ERROR_MESSAGES 中选择相应的消息，否则选择默认的错误消息
    else:
        message = ERROR_MESSAGES[ret_code + 99]

    # 返回优化结果对象，包括优化的结果向量 x，目标函数值 fun，优化状态 ret_code，
    # 成功标志 success（ret_code 大于 2 时为 True），消息 message，
    # 函数评估次数 nfev，迭代次数 nit
    return OptimizeResult(x=np.asarray(x), fun=fun, status=ret_code,
                          success=ret_code > 2, message=message,
                          nfev=nfev, nit=nit)
```