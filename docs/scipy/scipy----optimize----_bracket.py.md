# `D:\src\scipysrc\scipy\scipy\optimize\_bracket.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import scipy._lib._elementwise_iterative_method as eim  # 导入 SciPy 库中的特定模块
from scipy._lib._util import _RichResult  # 导入 SciPy 库中的 _RichResult 类

_ELIMITS = -1  # 在 _bracket_root 函数中用于限制的常数
_ESTOPONESIDE = 2  # 在 _bracket_root 函数中用于单边停止的常数

def _bracket_root_iv(func, xl0, xr0, xmin, xmax, factor, args, maxiter):
    """
    Bracket the root of a monotonic scalar function of one variable
    using initial values and constraints.

    Parameters
    ----------
    func : callable
        The function for which the root is to be bracketed.
    xl0, xr0: float array_like
        Starting guess of bracket, which need not contain a root.
    xmin, xmax: float or None
        Minimum and maximum bounds for the root.
    factor : float or None
        Factor to expand the bracket interval if needed.
    args : tuple
        Additional arguments to pass to `func`.
    maxiter : int
        Maximum number of iterations for bracketing.

    Returns
    -------
    tuple
        A tuple containing the function, initial values, bounds,
        expansion factor, additional arguments, and maximum iterations.
    """
    if not callable(func):
        raise ValueError('`func` must be callable.')

    if not np.iterable(args):
        args = (args,)

    xl0 = np.asarray(xl0)[()]
    if not np.issubdtype(xl0.dtype, np.number) or np.iscomplex(xl0).any():
        raise ValueError('`xl0` must be numeric and real.')

    xr0 = xl0 + 1 if xr0 is None else xr0
    xmin = -np.inf if xmin is None else xmin
    xmax = np.inf if xmax is None else xmax
    factor = 2. if factor is None else factor
    xl0, xr0, xmin, xmax, factor = np.broadcast_arrays(xl0, xr0, xmin, xmax, factor)

    if not np.issubdtype(xr0.dtype, np.number) or np.iscomplex(xr0).any():
        raise ValueError('`xr0` must be numeric and real.')

    if not np.issubdtype(xmin.dtype, np.number) or np.iscomplex(xmin).any():
        raise ValueError('`xmin` must be numeric and real.')

    if not np.issubdtype(xmax.dtype, np.number) or np.iscomplex(xmax).any():
        raise ValueError('`xmax` must be numeric and real.')

    if not np.issubdtype(factor.dtype, np.number) or np.iscomplex(factor).any():
        raise ValueError('`factor` must be numeric and real.')
    if not np.all(factor > 1):
        raise ValueError('All elements of `factor` must be greater than 1.')

    maxiter = np.asarray(maxiter)
    message = '`maxiter` must be a non-negative integer.'
    if (not np.issubdtype(maxiter.dtype, np.number) or maxiter.shape != tuple()
            or np.iscomplex(maxiter)):
        raise ValueError(message)
    maxiter_int = int(maxiter[()])
    if not maxiter == maxiter_int or maxiter < 0:
        raise ValueError(message)

    return func, xl0, xr0, xmin, xmax, factor, args, maxiter
    xmin, xmax : float array_like, optional
        # xmin 和 xmax 是浮点数数组，可选参数，表示 bracket 的最小和最大允许端点（包括）。必须与 xl0 和 xr0 广播兼容。

    factor : float array_like, default: 2
        # factor 是浮点数数组，默认为 2，用于扩展 bracket。详细信息请参见备注。

    args : tuple, optional
        # args 是元组，可选参数，传递给 func 的额外位置参数。必须与 xl0、xr0、xmin 和 xmax 广播兼容。如果需要 bracket 包含的可调用函数需要不与这些数组广播兼容的参数，则使用 func 包装该可调用函数，使 func 仅接受 x 和广播兼容的数组。

    maxiter : int, optional
        # 算法执行的最大迭代次数。

    Returns
    -------
    res : _RichResult
        # 返回一个 `scipy._lib._util._RichResult` 的实例，具有以下属性。描述假设值为标量，但如果 func 返回一个数组，则输出将是相同形状的数组。

        xl, xr : float
            # 如果算法成功终止，则是 bracket 的下限和上限。

        fl, fr : float
            # bracket 下限和上限处的函数值。

        nfev : int
            # 找到 bracket 所需的函数评估次数。这与调用 func 的次数不同，因为函数可能在单个调用中的多个点上进行评估。

        nit : int
            # 执行的算法迭代次数。

        status : int
            # 表示算法退出状态的整数。

            - ``0`` : 算法生成了一个有效的 bracket。
            - ``-1`` : bracket 扩展到允许的限制，但未找到 bracket。
            - ``-2`` : 达到了最大迭代次数。
            - ``-3`` : 遇到了非有限值。
            - ``-4`` : 由 `callback` 终止迭代。
            - ``-5`` : 初始 bracket 不满足 `xmin <= xl0 < xr0 < xmax`。
            - ``1`` : 算法正常进行（仅在 `callback` 中）。
            - ``2`` : 在相反的搜索方向中找到了一个 bracket（仅在 `callback` 中）。

        success : bool
            # 当算法成功终止时为 ``True``（状态 ``0``）。

    Notes
    -----
    # 此函数泛化了 `scipy.stats` 中分散的算法。策略是迭代地扩展 bracket ``(l, r)`` 直到 ``func(l) < 0 < func(r)``。以下是 bracket 向左扩展的过程。

    - 如果未提供 `xmin`，则 `xl0` 和 `l` 之间的距离将逐步增加 `factor`。
    - 如果提供了 `xmin`，则 `xmin` 和 `l` 之间的距离将逐步减少 `factor`。注意，这也会增加 bracket 的大小。
    Growth of the bracket to the right is analogous.

    Growth of the bracket in one direction stops when the endpoint is no longer
    finite, the function value at the endpoint is no longer finite, or the
    endpoint reaches its limiting value (`xmin` or `xmax`). Iteration terminates
    when the bracket stops growing in both directions, the bracket surrounds
    the root, or a root is found (accidentally).

    If two brackets are found - that is, a bracket is found on both sides in
    the same iteration, the smaller of the two is returned.
    If roots of the function are found, both `l` and `r` are set to the
    leftmost root.
    """  # noqa: E501

    # Todo:
    # - find bracket with sign change in specified direction
    # - Add tolerance
    # - allow factor < 1?

    # Initialize callback function to None (not used/tested here)
    callback = None

    # Perform initial bracketing of the root using _bracket_root_iv function
    temp = _bracket_root_iv(func, xl0, xr0, xmin, xmax, factor, args, maxiter)
    func, xl0, xr0, xmin, xmax, factor, args, maxiter = temp

    # Initialize xs with the bracket endpoints
    xs = (xl0, xr0)

    # Initialize temporary variables using eim._initialize function
    temp = eim._initialize(func, xs, args)
    func, xs, fs, args, shape, dtype, xp = temp  # line split for PEP8

    # Unpack xs into xl0 and xr0
    xl0, xr0 = xs

    # Broadcast xmin and xmax to match shape and convert to specified dtype
    xmin = np.broadcast_to(xmin, shape).astype(dtype, copy=False).ravel()
    xmax = np.broadcast_to(xmax, shape).astype(dtype, copy=False).ravel()

    # Identify invalid brackets based on boundary conditions
    invalid_bracket = ~((xmin <= xl0) & (xl0 < xr0) & (xr0 <= xmax))

    # The approach treats left and right searches independently for bracketing
    # `x` is the "moving" end of the bracket
    x = np.concatenate(xs)
    f = np.concatenate(fs)
    invalid_bracket = np.concatenate((invalid_bracket, invalid_bracket))
    n = len(x) // 2

    # `x_last` is the previous location of the moving end of the bracket. If
    # the signs of `f` and `f_last` are different, `x` and `x_last` form a
    # bracket.
    x_last = np.concatenate((x[n:], x[:n]))
    f_last = np.concatenate((f[n:], f[:n]))

    # `x0` is the "fixed" end of the bracket.
    x0 = x_last

    # Limit array for each bracket
    limit = np.concatenate((xmin, xmax))

    # Broadcast factor to match shape and convert to specified dtype
    factor = np.broadcast_to(factor, shape).astype(dtype, copy=False).ravel()
    factor = np.concatenate((factor, factor))

    # Array of active indices for processing
    active = np.arange(2*n)

    # Broadcast arguments to match shape
    args = [np.concatenate((arg, arg)) for arg in args]

    # Expand shape tuple to include an extra dimension
    shape = shape + (2,)

    # `d` is for "distance".
    # For searches without a limit, the distance between the fixed end of the
    # bracket `x0` and the moving end `x` will grow by `factor` each iteration.
    # For searches with a limit, the distance between the `limit` and moving
    # end of the bracket `x` will shrink by `factor` each iteration.
    i = np.isinf(limit)  # identify indices where limit is infinite
    ni = ~i  # complement of i, indices where limit is finite
    d = np.zeros_like(x)  # initialize array d with zeros, same shape as x
    d[i] = x[i] - x0[i]  # calculate d for infinite limit brackets
    d[ni] = limit[ni] - x[ni]  # calculate d for finite limit brackets

    status = np.full_like(x, eim._EINPROGRESS, dtype=int)  # initialize status array with _EINPROGRESS
    status[invalid_bracket] = eim._EINPUTERR  # mark invalid brackets with _EINPUTERR
    nit, nfev = 0, 1  # initialize nit and nfev variables

    work = _RichResult(x=x, x0=x0, f=f, limit=limit, factor=factor,
                       active=active, d=d, x_last=x_last, f_last=f_last,
                       nit=nit, nfev=nfev, status=status, args=args,
                       xl=None, xr=None, fl=None, fr=None, n=n)
    res_work_pairs = [('status', 'status'), ('xl', 'xl'), ('xr', 'xr'),
                      ('nit', 'nit'), ('nfev', 'nfev'), ('fl', 'fl'),
                      ('fr', 'fr'), ('x', 'x'), ('f', 'f'),
                      ('x_last', 'x_last'), ('f_last', 'f_last')]

    def pre_func_eval(work):
        # Initialize moving end of bracket
        x = np.zeros_like(work.x)

        # Unlimited brackets grow by `factor` by increasing distance from fixed
        # end to moving end.
        i = np.isinf(work.limit)  # indices of unlimited brackets
        work.d[i] *= work.factor[i]  # increase distance for unlimited brackets
        x[i] = work.x0[i] + work.d[i]  # update position of moving end for unlimited brackets

        # Limited brackets grow by decreasing the distance from the limit to
        # the moving end.
        ni = ~i  # indices of limited brackets
        work.d[ni] /= work.factor[ni]  # decrease distance for limited brackets
        x[ni] = work.limit[ni] - work.d[ni]  # update position of moving end for limited brackets

        return x

    def post_func_eval(x, f, work):
        # Keep track of the previous location of the moving end so that we can
        # return a narrower bracket. (The alternative is to remember the
        # original fixed end, but then the bracket would be wider than needed.)
        work.x_last = work.x  # store current position of moving end
        work.f_last = work.f  # store current function value at moving end
        work.x = x  # update current position of moving end
        work.f = f  # update current function value at moving end
    # 检查终止条件的函数，根据不同条件设置工作状态并返回停止标志位
    def check_termination(work):
        # Condition 0: initial bracket is invalid
        # 初始括号无效时停止搜索
        stop = (work.status == eim._EINPUTERR)

        # Condition 1: a valid bracket (or the root itself) has been found
        # 找到有效的括号（或根本身），标记为收敛状态并停止搜索
        sf = np.sign(work.f)
        sf_last = np.sign(work.f_last)
        i = ((sf_last == -sf) | (sf_last == 0) | (sf == 0)) & ~stop
        work.status[i] = eim._ECONVERGED
        stop[i] = True

        # Condition 2: the other side's search found a valid bracket.
        # 另一侧的搜索找到了有效的括号。
        # 如果我们刚刚找到了右侧搜索的括号，我们可以停止左侧搜索，反之亦然。
        # 为此，我们需要设置另一侧搜索的状态；
        # 这有些棘手，因为 `work.status` 只包含*活动*元素，所以我们不立即知道需要设置的元素的索引 - 或者它是否仍在那里。
        # 为了方便起见，`work.active` 包含每个搜索的单位整数索引。
        # 索引 `k` (`k < n`) 和 `k + n` 分别对应左侧和右侧搜索。元素从 `work.active` 中删除，就像它们从 `work.status` 中删除一样，
        # 因此我们使用 `work.active` 来帮助找到 `work.status` 中的正确位置。
        also_stop = (work.active[i] + work.n) % (2*work.n)
        # 检查它们是否仍然活动。
        # 首先，我们需要找出如果它们确实存在，它们会出现在 `work.active` 中的位置。
        j = np.searchsorted(work.active, also_stop)
        # 如果位置超过 `work.active` 的长度，则它们不在那里。
        j = j[j < len(work.active)]
        # 检查它们是否仍在那里。
        j = j[also_stop == work.active[j]]
        # 现在将这些转换为布尔索引以在 `work.status` 中使用。
        i = np.zeros_like(stop)
        i[j] = True  # 布尔索引，指示哪些元素也可以停止
        i = i & ~stop
        work.status[i] = _ESTOPONESIDE
        stop[i] = True

        # Condition 3: moving end of bracket reaches limit
        # 括号的移动端点达到了限制
        i = (work.x == work.limit) & ~stop
        work.status[i] = _ELIMITS
        stop[i] = True

        # Condition 4: non-finite value encountered
        # 遇到非有限值
        i = ~(np.isfinite(work.x) & np.isfinite(work.f)) & ~stop
        work.status[i] = eim._EVALUEERR
        stop[i] = True

        # 返回停止标志位
        return stop

    # 后续终止检查函数，目前未实现任何操作
    def post_termination_check(work):
        pass

    # 调用循环函数 `_loop` 执行迭代优化过程，并返回最终结果
    return eim._loop(work, callback, shape, maxiter, func, args, dtype,
                     pre_func_eval, post_func_eval, check_termination,
                     post_termination_check, customize_result, res_work_pairs,
                     xp)
    # 确保传入的函数 `func` 是可调用的
    if not callable(func):
        raise ValueError('`func` must be callable.')

    # 如果 `args` 不可迭代，将其转换为元组
    if not np.iterable(args):
        args = (args,)

    # 将 `xm0` 转换为数组，并确保其为实数类型的数值
    xm0 = np.asarray(xm0)[()]
    if not np.issubdtype(xm0.dtype, np.number) or np.iscomplex(xm0).any():
        raise ValueError('`xm0` must be numeric and real.')

    # 设置默认的 `xmin` 和 `xmax` 值，确保它们为实数类型的数值
    xmin = -np.inf if xmin is None else xmin
    xmax = np.inf if xmax is None else xmax

    # 如果 `xl0` 未提供，使用 NaN 作为临时值以便广播使用，等待 `xmin` 的验证后计算默认值
    xl0_not_supplied = False
    if xl0 is None:
        xl0 = np.nan
        xl0_not_supplied = True

    # 如果 `xr0` 未提供，使用 NaN 作为临时值以便广播使用，等待 `xmax` 的验证后计算默认值
    xr0_not_supplied = False
    if xr0 is None:
        xr0 = np.nan
        xr0_not_supplied = True

    # 设置 `factor` 的默认值为 2.0，确保其为实数类型的数值
    factor = 2.0 if factor is None else factor

    # 将 `xl0`, `xm0`, `xr0`, `xmin`, `xmax`, `factor` 广播为相同形状的数组
    xl0, xm0, xr0, xmin, xmax, factor = np.broadcast_arrays(
        xl0, xm0, xr0, xmin, xmax, factor
    )

    # 确保 `xl0` 为实数类型的数值
    if not np.issubdtype(xl0.dtype, np.number) or np.iscomplex(xl0).any():
        raise ValueError('`xl0` must be numeric and real.')

    # 确保 `xr0` 为实数类型的数值
    if not np.issubdtype(xr0.dtype, np.number) or np.iscomplex(xr0).any():
        raise ValueError('`xr0` must be numeric and real.')

    # 确保 `xmin` 为实数类型的数值
    if not np.issubdtype(xmin.dtype, np.number) or np.iscomplex(xmin).any():
        raise ValueError('`xmin` must be numeric and real.')

    # 确保 `xmax` 为实数类型的数值
    if not np.issubdtype(xmax.dtype, np.number) or np.iscomplex(xmax).any():
        raise ValueError('`xmax` must be numeric and real.')

    # 确保 `factor` 为实数类型的数值，并且所有元素大于 1
    if not np.issubdtype(factor.dtype, np.number) or np.iscomplex(factor).any():
        raise ValueError('`factor` must be numeric and real.')
    if not np.all(factor > 1):
        raise ValueError('All elements of `factor` must be greater than 1.')

    # 如果 `xl0` 或 `xr0` 未提供，则根据 `xm0` 和 `xmin`, `xmax` 计算默认值
    if xl0_not_supplied:
        xl0 = xm0 - np.minimum((xm0 - xmin)/16, 0.5)
    if xr0_not_supplied:
        xr0 = xm0 + np.minimum((xmax - xm0)/16, 0.5)

    # 将 `maxiter` 转换为数组，并确保其为非负整数
    maxiter = np.asarray(maxiter)
    message = '`maxiter` must be a non-negative integer.'
    if (not np.issubdtype(maxiter.dtype, np.number) or maxiter.shape != tuple()
            or np.iscomplex(maxiter)):
        raise ValueError(message)
    maxiter_int = int(maxiter[()])
    if not maxiter == maxiter_int or maxiter < 0:
        raise ValueError(message)

    # 返回修正后的参数元组，用于计算最小值的函数调用
    return func, xm0, xl0, xr0, xmin, xmax, factor, args, maxiter
    # func: callable
    # 要进行区间寻找最小值的函数。
    # 签名必须是::
    #
    #     func(x: ndarray, *args) -> ndarray
    #
    # 其中 `x` 的每个元素是有限实数，`args` 是一个元组，可以包含与 `x` 广播兼容的任意数量的数组。
    # `func` 必须是逐元素函数：对于所有索引 `i`，`func(x)[i]` 必须等于 `func(x[i])`。
    xm0: float array_like
        # 中点初始猜测的起始点。
    xl0, xr0: float array_like, optional
        # 左右端点初始猜测的起始点。必须与 `xm0` 广播兼容。
    xmin, xmax : float array_like, optional
        # 区间的最小和最大允许端点，包括这些值。必须与 `xl0`, `xm0`, 和 `xr0` 广播兼容。
    factor : float array_like, optional
        # 控制在下坡方向上扩展区间端点的因子。在设置下坡方向限制为 `xmax` 或 `xmin` 的情况下，其行为不同。详见注释。
    args : tuple, optional
        # 传递给 `func` 的额外位置参数。必须与 `xl0`, `xm0`, `xr0`, `xmin`, 和 `xmax` 广播兼容。
        # 如果要进行区间寻找的可调用函数需要不与这些数组广播兼容的参数，请将该可调用函数包装在 `func` 中，使得 `func` 仅接受 `x` 和广播兼容的数组。
    maxiter : int, optional
        # 算法执行的最大迭代次数。函数评估的次数比迭代次数多三次。

    Returns
    -------
    res : _RichResult
        # `_RichResult` 的实例，具有以下属性。描述假设值为标量，但若 `func` 返回数组，则输出将为相同形状的数组。
        xl, xm, xr : float
            # 如果算法成功终止，则为找到的支撑点的左、中、右点。
        fl, fm, fr : float
            # 支撑点左、中、右点的函数值。
        nfev : int
            # 寻找支撑点所需的函数评估次数。
        nit : int
            # 执行算法的迭代次数。
        status : int
            # 表示算法退出状态的整数。

            - ``0`` : 算法找到有效支撑点。
            - ``-1`` : 支撑点扩展到允许的极限。假设单调性，这意味着极限端点是一个极小值点。
            - ``-2`` : 达到最大迭代次数。
            - ``-3`` : 遇到非有限值。
            - ``-4`` : ``None`` 通过。
            - ``-5`` : 初始支撑点不满足 `xmin <= xl0 < xm0 < xr0 <= xmax`。
        success : bool
            # 当算法成功终止时为 ``True``（状态 ``0``）。

    Notes
    -----
    # 与 `scipy.optimize.bracket` 类似，此函数旨在找到实数点 ``xl < xm < xr``，
    # 使得 ``f(xl) >= f(xm)`` 和 ``f(xr) >= f(xm)``，至少一个不等式严格成立。
    # 与 `scipy.optimize.bracket` 不同的是，此函数可以在数组输入上向量化操作，
    # 只要输入数组可广播。另外，用户可以指定所需支撑点的最小和最大端点。

    # 给定初始的三个点 ``xl = xl0``, ``xm = xm0``, ``xr = xr0``，
    # 算法检查这些点是否已经给出了有效的支撑点。如果没有，
    # 将选择一个新的端点 ``w`` 在“下坡”方向上，``xm`` 成为新的对立端点，
    # `xl` 或 `xr` 成为新的中间点，取决于哪个方向是向下的。从这里开始重复算法。

    # 根据下坡方向是否设置了边界 `xmin` 或 `xmax`，选择不同的方法来选择新的端点 `w`。
    # 不失一般性，假设向右是下坡方向，使得 ``f(xl) > f(xm) > f(xr)``。如果右侧没有边界，
    # 则 `w` 被选为 ``xr + factor * (xr - xm)``，其中 `factor` 由用户控制（默认为 2.0），
    # 以便步长按几何比例增加。如果存在边界，例如 `xmax`，则 `w` 被选为
    """
    `xmax - (xmax - xr)/factor`, with steps slowing to a stop at
    `xmax`. This cautious approach ensures that a minimum near but distinct from
    the boundary isn't missed while also detecting whether or not the `xmax` is
    a minimizer when `xmax` is reached after a finite number of steps.
    """  # noqa: E501

    # 设置回调函数为 None，这里只是为了不测试它
    callback = None  # works; I just don't want to test it

    # 调用 _bracket_minimum_iv 函数进行初始搜索区间的定位
    temp = _bracket_minimum_iv(func, xm0, xl0, xr0, xmin, xmax, factor, args, maxiter)
    func, xm0, xl0, xr0, xmin, xmax, factor, args, maxiter = temp

    # 初始化搜索点列表和函数值列表
    xs = (xl0, xm0, xr0)
    temp = eim._initialize(func, xs, args)
    func, xs, fs, args, shape, dtype, xp = temp

    # 更新搜索点和函数值
    xl0, xm0, xr0 = xs
    fl0, fm0, fr0 = fs

    # 将 xmin 和 xmax 扩展为与搜索点相同的形状，并转换为指定的数据类型
    xmin = np.broadcast_to(xmin, shape).astype(dtype, copy=False).ravel()
    xmax = np.broadcast_to(xmax, shape).astype(dtype, copy=False).ravel()

    # 检查是否存在无效的搜索区间，标记为 invalid_bracket
    invalid_bracket = ~((xmin <= xl0) & (xl0 < xm0) & (xm0 < xr0) & (xr0 <= xmax))

    # 复制 factor 以便后续修改，np.broadcast_to 返回只读视图
    factor = np.broadcast_to(factor, shape).astype(dtype, copy=True).ravel()

    # 简化逻辑：如果 f(xl0) < f(xr0)，则交换 xl0 和 xr0，确保在从 xl0 到 xr0 的方向上始终向下
    comp = fl0 < fr0
    xl0[comp], xr0[comp] = xr0[comp], xl0[comp]
    fl0[comp], fr0[comp] = fr0[comp], fl0[comp]

    # 根据方向限制，只需要在我们搜索的方向上进行限制
    limit = np.where(comp, xmin, xmax)

    # 判断是否存在无限制的情况
    unlimited = np.isinf(limit)
    limited = ~unlimited

    # 计算步长，对于有限制的情况，步长会除以 factor
    step = np.empty_like(xl0)
    step[unlimited] = (xr0[unlimited] - xm0[unlimited])
    step[limited] = (limit[limited] - xr0[limited])
    factor[limited] = 1 / factor[limited]

    # 初始化状态数组为 EINPROGRESS，对于无效的搜索区间标记为 EINPUTERR
    status = np.full_like(xl0, eim._EINPROGRESS, dtype=int)
    status[invalid_bracket] = eim._EINPUTERR

    # 初始化迭代次数和函数评估次数
    nit, nfev = 0, 3

    # 创建 _RichResult 对象，存储优化算法的状态和参数
    work = _RichResult(xl=xl0, xm=xm0, xr=xr0, xr0=xr0, fl=fl0, fm=fm0, fr=fr0,
                       step=step, limit=limit, limited=limited, factor=factor, nit=nit,
                       nfev=nfev, status=status, args=args)

    # 指定结果和工作变量之间的对应关系
    res_work_pairs = [('status', 'status'), ('xl', 'xl'), ('xm', 'xm'), ('xr', 'xr'),
                      ('nit', 'nit'), ('nfev', 'nfev'), ('fl', 'fl'), ('fm', 'fm'),
                      ('fr', 'fr')]
    # 执行预处理评估函数，根据传入的工作对象修改状态
    def pre_func_eval(work):
        # 将步长乘以因子，更新工作对象中的步长属性
        work.step *= work.factor
        # 创建一个与 xr 相同形状的空数组 x
        x = np.empty_like(work.xr)
        # 对于非限制区域，计算新的端点 x
        x[~work.limited] = work.xr0[~work.limited] + work.step[~work.limited]
        # 对于限制区域，计算新的端点 x，当新端点等于旧端点且足够接近限制时，使用限制作为新端点
        x[work.limited] = work.limit[work.limited] - work.step[work.limited]
        x[work.limited] = np.where(
            x[work.limited] == work.xr[work.limited],
            work.limit[work.limited],
            x[work.limited],
        )
        # 返回更新后的端点数组 x
        return x

    # 执行后处理评估函数，更新工作对象中的端点和函数值
    def post_func_eval(x, f, work):
        # 更新工作对象中的左、中、右端点和函数值
        work.xl, work.xm, work.xr = work.xm, work.xr, x
        work.fl, work.fm, work.fr = work.fm, work.fr, f

    # 检查终止条件，更新工作对象中的状态属性并返回终止标志数组
    def check_termination(work):
        # 条件 0：初始区间无效，设置终止标志为 True
        stop = (work.status == eim._EINPUTERR)

        # 条件 1：找到有效的区间
        i = (
            (work.fl >= work.fm) & (work.fr > work.fm)
            | (work.fl > work.fm) & (work.fr >= work.fm)
        ) & ~stop
        # 设置状态为收敛状态，并将终止标志设置为 True
        work.status[i] = eim._ECONVERGED
        stop[i] = True

        # 条件 2：移动区间端点达到限制
        i = (work.xr == work.limit) & ~stop
        # 设置状态为限制状态，并将终止标志设置为 True
        work.status[i] = _ELIMITS
        stop[i] = True

        # 条件 3：遇到非有限值
        i = ~(np.isfinite(work.xr) & np.isfinite(work.fr)) & ~stop
        # 设置状态为数值错误状态，并将终止标志设置为 True
        work.status[i] = eim._EVALUEERR
        stop[i] = True

        # 返回最终的终止标志数组
        return stop

    # 后终止条件检查函数，暂未实现具体功能
    def post_termination_check(work):
        pass

    # 定制结果函数，重新排列 xl 和 xr 的条目，以确保 xl 总是小于或等于 xr
    def customize_result(res, shape):
        # 如果由于 f(xl0) < f(xr0) 而交换了 xl 和 xr，则重新调整它们的顺序
        comp = res['xl'] > res['xr']
        res['xl'][comp], res['xr'][comp] = res['xr'][comp], res['xl'][comp]
        res['fl'][comp], res['fr'][comp] = res['fr'][comp], res['fl'][comp]
        # 返回形状参数 shape（此处返回结果的形状似乎存在错误，应该返回 res，可能是注释的错误）
        return shape

    # 调用优化循环函数，并传入一系列参数和回调函数
    return eim._loop(work, callback, shape,
                     maxiter, func, args, dtype,
                     pre_func_eval, post_func_eval,
                     check_termination, post_termination_check,
                     customize_result, res_work_pairs, xp)
```