# `D:\src\scipysrc\scipy\scipy\integrate\_tanhsinh.py`

```
# mypy: disable-error-code="attr-defined"
# 引入所需的库和模块
import numpy as np
from scipy import special
import scipy._lib._elementwise_iterative_method as eim
from scipy._lib._util import _RichResult

# todo:
#  figure out warning situation
#  address https://github.com/scipy/scipy/pull/18650#discussion_r1233032521
#  without `minweight`, we are also suppressing infinities within the interval.
#    Is that OK? If so, we can probably get rid of `status=3`.
#  Add heuristic to stop when improvement is too slow / antithrashing
#  support singularities? interval subdivision? this feature will be added
#    eventually, but do we adjust the interface now?
#  When doing log-integration, should the tolerances control the error of the
#    log-integral or the error of the integral?  The trouble is that `log`
#    inherently looses some precision so it may not be possible to refine
#    the integral further. Example: 7th moment of stats.f(15, 20)
#  respect function evaluation limit?
#  make public?

# 定义一个函数 `_tanhsinh`，使用 tanh-sinh 积分方法进行数值积分
def _tanhsinh(f, a, b, *, args=(), log=False, maxfun=None, maxlevel=None,
              minlevel=2, atol=None, rtol=None, preserve_shape=False,
              callback=None):
    """Evaluate a convergent integral numerically using tanh-sinh quadrature.

    In practice, tanh-sinh quadrature achieves quadratic convergence for
    many integrands: the number of accurate *digits* scales roughly linearly
    with the number of function evaluations [1]_.

    Either or both of the limits of integration may be infinite, and
    singularities at the endpoints are acceptable. Divergent integrals and
    integrands with non-finite derivatives or singularities within an interval
    are out of scope, but the latter may be evaluated be calling `_tanhsinh` on
    each sub-interval separately.

    Parameters
    ----------
    f : callable
        The function to be integrated. The signature must be::
            func(x: ndarray, *fargs) -> ndarray
         where each element of ``x`` is a finite real and ``fargs`` is a tuple,
         which may contain an arbitrary number of arrays that are broadcastable
         with `x`. ``func`` must be an elementwise-scalar function; see
         documentation of parameter `preserve_shape` for details.
         If ``func`` returns a value with complex dtype when evaluated at
         either endpoint, subsequent arguments ``x`` will have complex dtype
         (but zero imaginary part).
    a, b : array_like
        Real lower and upper limits of integration. Must be broadcastable.
        Elements may be infinite.
    args : tuple, optional
        Additional positional arguments to be passed to `func`. Must be arrays
        broadcastable with `a` and `b`. If the callable to be integrated
        requires arguments that are not broadcastable with `a` and `b`, wrap
        that callable with `f`. See Examples.
    log : bool, optional
        If True, perform logarithmic integration.
    maxfun : int or None, optional
        Maximum number of function evaluations.
    maxlevel : int or None, optional
        Maximum recursion depth.
    minlevel : int, optional
        Minimum recursion depth.
    atol, rtol : float or None, optional
        Absolute and relative tolerances.
    preserve_shape : bool, optional
        If True, `func` is applied elementwise, with results of the same shape
        as the inputs.
    callback : callable, optional
        Function called after each evaluation.

    Returns
    -------
    result : _RichResult
        A `_RichResult` object that encapsulates the result of the integration.
    """
    # 是否返回对数形式的积分结果，如果是，则积分函数返回对数形式的被积函数值，
    # 并且绝对误差和相对误差以对数形式表示。在这种情况下，结果对象将包含积分的对数和误差。
    # 这对于那些数值下溢或上溢可能导致不准确的被积函数很有用。
    # 当 `log=True` 时，被积函数（即 `f` 的指数函数）必须是实数，
    # 但可以是负数，此时被积函数的对数是一个具有虚部，其虚部为π的奇数倍的复数。
    log : bool, default: False
    
    # 算法的最大细化级别，默认为 10。
    # 在零级时，调用 `f` 一次，执行 16 次函数评估。
    # 在每个后续级别，再调用一次 `f`，大约加倍已执行的函数评估次数。
    # 因此，对于许多被积函数，每个后续级别将使结果中的有效数字加倍（直到浮点精度的极限）。
    # 算法将在完成 `maxlevel` 级别或满足另一个终止条件后终止，以先到者为准。
    maxlevel : int, default: 10
    
    # 迭代开始的级别，默认为 2。这不会改变总函数评估次数或函数评估的横坐标，
    # 它仅仅改变了 `f` 被调用的次数。
    # 如果 `minlevel=k`，则被积函数在级别 `0` 到 `k` 的所有横坐标上将在一次调用中进行评估。
    # 注意，如果 `minlevel` 超过 `maxlevel`，则忽略提供的 `minlevel`，并将 `minlevel` 设置为 `maxlevel`。
    minlevel : int, default: 2
    
    # 绝对终止容差（默认为 0）和相对终止容差（默认为 `eps**0.75`，
    # 其中 `eps` 是结果数据类型的精度），可选参数。
    # 误差估计如 [1]_ 第 5 节所述。虽然在理论上不严格或保守，
    # 但据说在实践中效果很好。如果 `log` 为 False，则必须是非负且有限的数；
    # 如果 `log` 为 True，则必须表示为非负且有限数的对数。
    atol, rtol : float, optional
    preserve_shape : bool, default: False
        # 是否保持形状的标志，默认为 False
        在以下内容中，“f 的参数”指数组 ``x`` 和 ``fargs`` 中的任何数组。令 ``shape`` 为 `a`、`b` 和 `args` 的广播形状（概念上与传递给 `f` 的 ``fargs`` 是不同的）。

        - 当 ``preserve_shape=False`` 时（默认），`f` 必须接受任何可广播形状的参数。

        - 当 ``preserve_shape=True`` 时，`f` 必须接受形状为 ``shape`` 或 ``shape + (n,)`` 的参数，其中 ``(n,)`` 是在函数计算时的自变量数。

        在任何情况下，对于 `x` 中的每个标量元素 ``xi``，`f` 返回的数组必须在相同的索引处包含标量 ``f(xi)``。因此，输出的形状始终与输入 ``x`` 的形状相同。

        参见示例。

    callback : callable, optional
        # 可选的用户提供的回调函数，在第一次迭代之前和每次迭代之后调用。
        被调用为 ``callback(res)``，其中 ``res`` 是一个类似于 `_differentiate` 返回的 ``_RichResult`` 对象（但包含当前迭代的所有变量值）。如果 `callback` 抛出 `StopIteration` 异常，算法将立即终止，`_tanhsinh` 将返回一个结果对象。

    Returns
    -------
    res : _RichResult
        # 返回一个 `scipy._lib._util._RichResult` 的实例，具有以下属性。（描述假设返回值为标量；然而，如果 `func` 返回数组，则输出将是相同形状的数组。）
        success : bool
            当算法成功终止时为 ``True``（状态 ``0``）。
        status : int
            表示算法退出状态的整数。
            ``0`` : 算法收敛到指定的容差。
            ``-1`` : （未使用）
            ``-2`` : 达到最大迭代次数。
            ``-3`` : 遇到非有限值。
            ``-4`` : 被 `callback` 终止迭代。
            ``1`` : 算法正常进行中（仅在 `callback` 中）。
        integral : float
            积分的估计值。
        error : float
            误差的估计值。仅在完成二级或更高级别时才可用；否则为 NaN。
        maxlevel : int
            使用的最大细化级别。
        nfev : int
            对 `func` 评估的点数。

    See Also
    --------
    quad, quadrature

    Notes
    -----
    # 实现了 [1]_ 中描述的算法，为了有限精度算术做了一些微小的调整，包括 [2]_ 和 [3]_ 中描述的一些内容。tanh-sinh 方案最初在 [4]_ 中引入。
    # Import necessary libraries and modules for numerical integration
    import numpy as np
    from scipy.integrate._tanhsinh import _tanhsinh
    
    # Define the function to be integrated, in this case the Gaussian function
    def f(x):
        return np.exp(-x**2)
    
    # Perform numerical integration using tanh-sinh quadrature over the infinite interval from -∞ to +∞
    res = _tanhsinh(f, -np.inf, np.inf)
    
    # Print the computed value of the integral of the Gaussian function over the infinite interval
    res.integral  # true value is np.sqrt(np.pi), 1.7724538509055159
    
    # Print the error estimate for the computed integral
    res.error  # actual error is 0
    
    # Demonstrate integration over a finite interval [-20, 20] where the Gaussian function is nonzero
    _tanhsinh(f, -20, 20).integral
    
    # Demonstrate integration over an interval from -∞ to 1000, where the Gaussian function is effectively zero outside [-20, 20]
    _tanhsinh(f, -np.inf, 1000).integral
    
    # Break the integral into parts at the singularity to avoid issues with unfavorable integration limits
    _tanhsinh(f, -np.inf, 0).integral + _tanhsinh(f, 0, 1000).integral
    
    # Example illustrating the use of log-integration for extremely large or small magnitudes
    res = _tanhsinh(f, 20, 30, rtol=1e-10)
    res.integral, res.error
    def log_f(x):
        return -x**2
    np.exp(res.integral), np.exp(res.error)
    
    # Demonstrate integration with broadcastable arrays and elementwise integration
    from scipy import stats
    dist = stats.gausshyper(13.8, 3.12, 2.51, 5.18)
    a, b = dist.support()
    # 使用 numpy.linspace 创建包含 100 个元素的等间距数组 x，范围从 a 到 b
    >>> x = np.linspace(a, b, 100)
    
    # 调用 _tanhsinh 函数计算 dist.pdf 在区间 [a, x] 上的数值积分结果，保存在 res 中
    >>> res = _tanhsinh(dist.pdf, a, x)
    
    # 计算 dist.cdf 在数组 x 上的累积分布函数值，保存在 ref 中
    >>> ref = dist.cdf(x)
    
    # 检查 res.integral 是否与 ref 数值接近，返回比较结果的布尔值
    >>> np.allclose(res.integral, ref)
    
    # 默认情况下，preserve_shape 参数为 False，因此可调用的函数 f 可以接受任何可以广播形状的数组作为参数。
    # 例如：
    
    # 创建一个空列表 shapes，用于存储不同调用中参数 x 和 c 的广播后的形状
    >>> shapes = []
    
    # 定义一个函数 f，它接受参数 x 和 c，计算 np.sin(c*x) 并记录参数 x 的广播形状到 shapes 中
    >>> def f(x, c):
    ...    shape = np.broadcast_shapes(x.shape, c.shape)
    ...    shapes.append(shape)
    ...    return np.sin(c*x)
    
    # 定义 c 的值为列表 [1, 10, 30, 100]
    >>> c = [1, 10, 30, 100]
    
    # 调用 _tanhsinh 函数，计算函数 f 在区间 [0, 1] 上，参数 c 为输入的数值积分结果，minlevel 设置为 1
    >>> res = _tanhsinh(f, 0, 1, args=(c,), minlevel=1)
    
    # 打印 shapes 列表，显示不同调用中参数 x 和 c 的广播形状
    >>> shapes
    [(4,), (4, 66), (3, 64), (2, 128), (1, 256)]
    
    # 通过增加 c 的值来理解 shapes 的变化，较高的 c 值对应较高频率的正弦波，使得被积函数更复杂，
    # 需要更多函数评估来达到目标精度：
    
    # 打印 res.nfev 数组，显示每次函数评估的次数
    >>> res.nfev
    array([ 67, 131, 259, 515])
    
    # 初始的 shape (4,) 对应于在单个横坐标和四个频率下评估被积函数；这用于输入验证和确定存储结果的数组的大小和 dtype。
    # 下一个 shape 对应于在初始横坐标网格和所有四个频率下评估被积函数。
    # 后续的函数调用会使函数在评估时加倍总横坐标数，但是频率较少因为相应的积分已经收敛到所需的容差。
    # 这节省了函数评估以提高性能，但要求函数接受任何形状的参数。
    
    # "向量值" 被积函数，例如为 scipy.integrate.quad_vec 编写的函数，不太可能满足这一要求。
    # 例如，考虑下面的函数：
    
    # 定义一个函数 f，返回一个包含多个数组的列表，不兼容 _tanhsinh 函数的写法；例如，输出的形状不同于输入的形状。
    >>> def f(x):
    ...    return [x, np.sin(10*x), np.cos(30*x), x*np.sin(100*x)**2]
    
    # 这样的被积函数不能直接与 _tanhsinh 使用；例如，输出的形状不同于输入的形状。
    # 这样的函数 *可以* 通过引入额外的参数进行转换以兼容，但这样会很不方便。
    # 在这种情况下，更简单的解决方案是使用 preserve_shape。
    
    # 创建一个空列表 shapes，用于存储不同调用中参数 x 的形状
    >>> shapes = []
    
    # 定义一个函数 f，记录参数 x 的形状到 shapes 中，并返回一个向量值积分函数的兼容形式
    >>> def f(x):
    ...     shapes.append(x.shape)
    ...     x0, x1, x2, x3 = x
    ...     return [x0, np.sin(10*x1), np.cos(30*x2), x3*np.sin(100*x3)]
    
    # 创建一个全为零的长度为 4 的 numpy 数组 a
    >>> a = np.zeros(4)
    
    # 调用 _tanhsinh 函数，计算函数 f 在区间 [a, 1] 上，保持参数形状设置为 True
    >>> res = _tanhsinh(f, a, 1, preserve_shape=True)
    
    # 打印 shapes 列表，显示不同调用中参数 x 的形状
    >>> shapes
    [(4,), (4, 66), (4, 64), (4, 128), (4, 256)]
    
    # 在这里，参数 a 和 b 的广播形状为 (4,)。当 preserve_shape=True 时，函数可以接受形状为 (4,) 或 (4, n) 的参数 x，这正是我们观察到的。
    (f, a, b, log, maxfun, maxlevel, minlevel,
     atol, rtol, args, preserve_shape, callback) = _tanhsinh_iv(
        f, a, b, log, maxfun, maxlevel, minlevel, atol,
        rtol, args, preserve_shape, callback)

# 解构 `_tanhsinh_iv` 函数返回的多个变量，用于初始化积分参数和设置。


    # Initialization
    # `eim._initialize` does several important jobs, including
    # ensuring that limits, each of the `args`, and the output of `f`
    # broadcast correctly and are of consistent types. To save a function
    # evaluation, I pass the midpoint of the integration interval. This comes
    # at a cost of some gymnastics to ensure that the midpoint has the right
    # shape and dtype. Did you know that 0d and >0d arrays follow different
    # type promotion rules?
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        # Compute the midpoint of the integration interval
        c = ((a.ravel() + b.ravel())/2).reshape(a.shape)
        inf_a, inf_b = np.isinf(a), np.isinf(b)
        # Adjust midpoint for infinite values in 'a' and 'b'
        c[inf_a] = b[inf_a] - 1  # takes care of infinite a
        c[inf_b] = a[inf_b] + 1  # takes care of infinite b
        c[inf_a & inf_b] = 0  # takes care of infinite a and b
        # Initialize using `eim._initialize` with the computed midpoint
        temp = eim._initialize(f, (c,), args, complex_ok=True,
                               preserve_shape=preserve_shape)
    f, xs, fs, args, shape, dtype, xp = temp
    a = np.broadcast_to(a, shape).astype(dtype).ravel()
    b = np.broadcast_to(b, shape).astype(dtype).ravel()

# 初始化过程，其中 `eim._initialize` 函数确保限制、参数和函数输出在广播时类型一致。
# 使用 `np.errstate` 确保在计算中忽略溢出、无效和除以零的错误。
# 计算积分区间的中点 `c`，处理无穷值情况。
# 通过 `eim._initialize` 初始化并获取积分需要的参数和类型信息。


    # Transform improper integrals
    a, b, a0, negative, abinf, ainf, binf = _transform_integrals(a, b)

# 转换不适当的积分边界，确保积分范围符合计算要求。


    # Define variables we'll need
    nit, nfev = 0, 1  # one function evaluation performed above
    zero = -np.inf if log else 0
    pi = dtype.type(np.pi)
    maxiter = maxlevel - minlevel + 1
    eps = np.finfo(dtype).eps
    if rtol is None:
        rtol = 0.75*np.log(eps) if log else eps**0.75

# 定义需要的变量：
# - `nit` 和 `nfev` 初始化为 0 和 1，表示已执行一次函数评估。
# - `zero` 根据 `log` 值决定是 `-np.inf` 还是 `0`。
# - `pi` 表示圆周率，类型根据 `dtype` 确定。
# - `maxiter` 表示迭代的最大级别。
# - `eps` 表示 `dtype` 类型的机器精度。
# - 如果 `rtol` 为 `None`，则根据 `log` 的值设置 `rtol` 的默认值。


    Sn = np.full(shape, zero, dtype=dtype).ravel()  # latest integral estimate
    Sn[np.isnan(a) | np.isnan(b) | np.isnan(fs[0])] = np.nan
    Sk = np.empty_like(Sn).reshape(-1, 1)[:, 0:0]  # all integral estimates
    aerr = np.full(shape, np.nan, dtype=dtype).ravel()  # absolute error
    status = np.full(shape, eim._EINPROGRESS, dtype=int).ravel()
    h0 = np.real(_get_base_step(dtype=dtype))  # base step

# 初始化积分估计和相关变量：
# - `Sn` 是最新的积分估计数组。
# - 根据条件设置 `Sn` 中的无效值为 `nan`。
# - `Sk` 是所有积分估计的空数组。
# - `aerr` 是绝对误差的数组，初始为 `nan`。
# - `status` 是积分状态的数组，初始为 `_EINPROGRESS`。
# - `h0` 是基础步长的实部值。


    # For term `d4` of error estimate ([1] Section 5), we need to keep the
    # most extreme abscissae and corresponding `fj`s, `wj`s in Euler-Maclaurin
    # sum. Here, we initialize these variables.
    xr0 = np.full(shape, -np.inf, dtype=dtype).ravel()
    fr0 = np.full(shape, np.nan, dtype=dtype).ravel()
    wr0 = np.zeros(shape, dtype=dtype).ravel()
    xl0 = np.full(shape, np.inf, dtype=dtype).ravel()
    fl0 = np.full(shape, np.nan, dtype=dtype).ravel()
    wl0 = np.zeros(shape, dtype=dtype).ravel()
    d4 = np.zeros(shape, dtype=dtype).ravel()

# 初始化用于误差估计中 `d4` 项所需的变量：
# - `xr0`, `fr0`, `wr0` 是保存极端点和对应函数值和权重的变量。
# - `xl0`, `fl0`, `wl0` 是另一组保存极端点和对应函数值和权重的变量。
# - `d4` 是误差估计的一部分，初始化为零。
    # 创建一个 `_RichResult` 对象，用于存储各种结果和参数
    work = _RichResult(
        Sn=Sn, Sk=Sk, aerr=aerr, h=h0, log=log, dtype=dtype, pi=pi, eps=eps,
        a=a.reshape(-1, 1), b=b.reshape(-1, 1),  # 设置积分的上下限
        n=minlevel, nit=nit, nfev=nfev, status=status,  # 迭代和评估计数
        xr0=xr0, fr0=fr0, wr0=wr0, xl0=xl0, fl0=fl0, wl0=wl0, d4=d4,  # 错误估计
        ainf=ainf, binf=binf, abinf=abinf, a0=a0.reshape(-1, 1))  # 变换参数

    # 常数标量如果不需要在 `work` 中传递到 `tanhsinh` 之外，不需要放入 `work` 中。
    # 例如：atol, rtol, h0, minlevel。

    # `work` 对象中的术语与结果之间的对应关系
    res_work_pairs = [('status', 'status'), ('integral', 'Sn'),
                      ('error', 'aerr'), ('nit', 'nit'), ('nfev', 'nfev')]

    def pre_func_eval(work):
        # 确定在评估 `f` 时使用的横坐标
        work.h = h0 / 2**work.n
        xjc, wj = _get_pairs(work.n, h0, dtype=work.dtype,
                             inclusive=(work.n == minlevel))
        work.xj, work.wj = _transform_to_limits(xjc, wj, work.a, work.b)

        # 对于无穷积分限的横坐标替换
        xj = work.xj.copy()
        xj[work.abinf] = xj[work.abinf] / (1 - xj[work.abinf]**2)
        xj[work.binf] = 1/xj[work.binf] - 1 + work.a0[work.binf]
        xj[work.ainf] *= -1
        return xj

    def post_func_eval(x, fj, work):
        # 根据无穷积分限的替换，加权积分被积函数
        if work.log:
            fj[work.abinf] += (np.log(1 + work.xj[work.abinf] ** 2)
                               - 2*np.log(1 - work.xj[work.abinf] ** 2))
            fj[work.binf] -= 2 * np.log(work.xj[work.binf])
        else:
            fj[work.abinf] *= ((1 + work.xj[work.abinf]**2) /
                               (1 - work.xj[work.abinf]**2)**2)
            fj[work.binf] *= work.xj[work.binf]**-2.

        # 使用欧拉-麦克劳林求和法估计积分
        fjwj, Sn = _euler_maclaurin_sum(fj, work)
        if work.Sk.shape[-1]:
            Snm1 = work.Sk[:, -1]
            Sn = (special.logsumexp([Snm1 - np.log(2), Sn], axis=0) if log
                  else Snm1 / 2 + Sn)

        work.fjwj = fjwj
        work.Sn = Sn
    def check_termination(work):
        """检查是否终止，根据收敛或遇到非有限值"""
    
        # 创建一个布尔数组，用于标记终止条件
        stop = np.zeros(work.Sn.shape, dtype=bool)
    
        # 如果是第一次迭代且积分限制相等，则提前终止
        if work.nit == 0:
            # 扁平化单维度数组
            i = (work.a == work.b).ravel()
            # 如果 log 为真，则设置为 -inf，否则为 0
            zero = -np.inf if log else 0
            work.Sn[i] = zero  # 设置 work.Sn 的部分值为 zero
            work.aerr[i] = zero  # 设置 work.aerr 的部分值为 zero
            work.status[i] = eim._ECONVERGED  # 设置对应状态为收敛状态
            stop[i] = True  # 标记为终止状态
        else:
            # 如果满足收敛条件，则终止
            work.rerr, work.aerr = _estimate_error(work)
            if log:
                i = ((work.rerr < rtol) | (work.rerr + np.real(work.Sn) < atol))
            else:
                i = ((work.rerr < rtol) | (work.rerr * abs(work.Sn) < atol))
            work.status[i] = eim._ECONVERGED  # 设置对应状态为收敛状态
            stop[i] = True  # 标记为终止状态
    
        # 如果积分估计变得无效，则终止
        if log:
            i = (np.isposinf(np.real(work.Sn)) | np.isnan(work.Sn)) & ~stop
        else:
            i = ~np.isfinite(work.Sn) & ~stop
        work.status[i] = eim._EVALUEERR  # 设置对应状态为值错误状态
        stop[i] = True  # 标记为终止状态
    
        return stop
    
    
    def post_termination_check(work):
        """终止后的检查"""
    
        work.n += 1  # 增加迭代次数
        work.Sk = np.concatenate((work.Sk, work.Sn[:, np.newaxis]), axis=-1)  # 将 work.Sn 添加到 work.Sk 中
        return
    
    
    def customize_result(res, shape):
        """定制结果"""
    
        # 如果积分限制为 b < a，则需要对最终结果进行取反
        if log and np.any(negative):
            pi = res['integral'].dtype.type(np.pi)
            j = np.complex64(1j)  # 最小复数类型
            res['integral'] = res['integral'] + negative * pi * j
        else:
            res['integral'][negative] *= -1
    
        # 对于该算法，报告最大级别比报告迭代次数更合适
        res['maxlevel'] = minlevel + res['nit'] - 1
        res['maxlevel'][res['nit'] == 0] = -1
        del res['nit']
        return shape
    
    
    # 初始阶段禁止所有警告，因为代码中有许多预期的警告情况
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        res = eim._loop(work, callback, shape, maxiter, f, args, dtype, pre_func_eval,
                        post_func_eval, check_termination, post_termination_check,
                        customize_result, res_work_pairs, xp, preserve_shape)
    return res
# 计算给定数据类型的基本步长，通常用于数值积分算法中
def _get_base_step(dtype=np.float64):
    # 计算基本步长时，需要保持一定距离远离浮点数极限
    fmin = 4*np.finfo(dtype).tiny  # 略微远离极限
    # 计算 tmax，即当 abscissa 补充值 xjc 下溢时的参数值
    tmax = np.arcsinh(np.log(2/fmin - 1) / np.pi)

    # 根据 tmax 选择基本步长 h0 作为积分算法的起始步长
    h0 = tmax / _N_BASE_STEPS
    return h0.astype(dtype)


_N_BASE_STEPS = 8


def _compute_pair(k, h0):
    # 计算第 k 级别的 abscissa-weight 对。参见文献 [1] 第 9 页。

    # 目前我们使用 64 位精度进行计算和存储。如果以后支持更高精度的数据类型，
    # 最好使用最高精度进行计算。或者一旦有与 Array API 兼容的任意精度数组，
    # 我们可以以所需的精度进行计算。

    # 每个级别 k 的 abscissa-weight 对使用 h = 2 **-k 来适应浮点运算，参考文献 [2]。
    h = h0 / 2**k
    max = _N_BASE_STEPS * 2**k

    # 对于第一次迭代之后的迭代，“....需要仅在每个级别的奇数索引 abscissas 处评估函数。”
    j = np.arange(max+1) if k == 0 else np.arange(1, max+1, 2)
    jh = j * h

    # 在这种情况下，权重 wj = u1/cosh(u2)^2，其中...
    pi_2 = np.pi / 2
    u1 = pi_2*np.cosh(jh)
    u2 = pi_2*np.sinh(jh)
    # 分母在这里会变大。不需要警告溢出和下溢。
    wj = u1 / np.cosh(u2)**2
    # 实际上我们存储 1-xj = 1/(...)。
    xjc = 1 / (np.exp(u2) * np.cosh(u2))  # xj 补充值 = np.tanh(u2)

    # 当 k == 0 时，零级 xj 对应于 xj = 0。为简化代码，函数将在该处评估两次；每次权重减半。
    wj[0] = wj[0] / 2 if k == 0 else wj[0]

    return xjc, wj  # 以完整精度存储
        

def _pair_cache(k, h0):
    # 缓存直到指定级别的 abscissa-weight 对。
    # 连续级别的 abscissae 和 weights 被串联。
    # `index` 记录与每个级别对应的索引:
    pass
    # 如果当前的h0不等于缓存中的_h0值，则进行以下操作
    if h0 != _pair_cache.h0:
        # 清空缓存中的_xjc和_wj数组
        _pair_cache.xjc = np.empty(0)
        _pair_cache.wj = np.empty(0)
        # 将indices数组初始化为包含一个元素0的列表
        _pair_cache.indices = [0]

    # 将_pair_cache.xjc添加到xjcs列表中
    xjcs = [_pair_cache.xjc]
    # 将_pair_cache.wj添加到wjs列表中
    wjs = [_pair_cache.wj]

    # 遍历_pair_cache.indices列表，从倒数第二个元素到k+1，每次迭代计算一对(xjc, wj)，并将其添加到xjcs和wjs列表中
    for i in range(len(_pair_cache.indices)-1, k + 1):
        xjc, wj = _compute_pair(i, h0)
        xjcs.append(xjc)
        wjs.append(wj)
        # 将新计算的xjc数组的长度添加到indices数组中作为新的索引位置
        _pair_cache.indices.append(_pair_cache.indices[-1] + len(xjc))

    # 将xjcs列表中的所有数组连接起来，赋值给_pair_cache.xjc
    _pair_cache.xjc = np.concatenate(xjcs)
    # 将wjs列表中的所有数组连接起来，赋值给_pair_cache.wj
    _pair_cache.wj = np.concatenate(wjs)
    # 更新缓存中的_h0值为当前的h0
    _pair_cache.h0 = h0
# Initialize cache variables for abscissa-weight pairs
_pair_cache.xjc = np.empty(0)  # Empty numpy array for cached abscissae
_pair_cache.wj = np.empty(0)   # Empty numpy array for cached weights
_pair_cache.indices = [0]      # Initial indices list with a single element 0
_pair_cache.h0 = None          # Initialize h0 cache variable as None


def _get_pairs(k, h0, inclusive=False, dtype=np.float64):
    """
    Retrieve the specified abscissa-weight pairs from the cache.
    
    Parameters:
    - k: int, level of pairs to retrieve
    - h0: value, auxiliary parameter determining the pairs retrieved
    - inclusive: bool, if True, return pairs up to and including level k
    - dtype: data type, type for returned arrays
    
    Returns:
    - xjc[start:end]: numpy array, abscissae within specified range
    - wj[start:end]: numpy array, weights corresponding to abscissae
    """
    # Check if cache needs updating based on level k and h0
    if len(_pair_cache.indices) <= k+2 or h0 != _pair_cache.h0:
        _pair_cache(k, h0)

    xjc = _pair_cache.xjc    # Retrieve cached abscissae
    wj = _pair_cache.wj      # Retrieve cached weights
    indices = _pair_cache.indices  # Retrieve indices for accessing pairs

    start = 0 if inclusive else indices[k]     # Start index for slicing
    end = indices[k+1]                         # End index for slicing

    return xjc[start:end].astype(dtype), wj[start:end].astype(dtype)


def _transform_to_limits(xjc, wj, a, b):
    """
    Transform integral according to user-specified limits.

    Parameters:
    - xjc: numpy array, abscissae to transform
    - wj: numpy array, weights corresponding to abscissae
    - a: float, lower limit of transformation
    - b: float, upper limit of transformation
    
    Returns:
    - xj: numpy array, transformed abscissae
    - wj: numpy array, adjusted weights after transformation
    """
    alpha = (b - a) / 2
    xj = np.concatenate((-alpha * xjc + b, alpha * xjc + a), axis=-1)
    wj = wj * alpha  # Adjust weights according to transformation
    wj = np.concatenate((wj, wj), axis=-1)  # Duplicate weights for both ends

    # Zero out weights for points outside specified limits due to precision issues
    invalid = (xj <= a) | (xj >= b)
    wj[invalid] = 0

    return xj, wj


def _euler_maclaurin_sum(fj, work):
    """
    Perform the Euler-Maclaurin Sum, following [1] Section 4.

    Parameters:
    - fj: numpy array, function values at specified abscissae
    - work: object, containing various work variables and arrays
    
    Returns:
    - None
    """
    xr0, fr0, wr0 = work.xr0, work.fr0, work.wr0
    xl0, fl0, wl0 = work.xl0, work.fl0, work.wl0

    xj, fj, wj = work.xj.T, fj.T, work.wj.T
    n_x, n_active = xj.shape  # Number of abscissae and active elements

    xr, xl = xj.reshape(2, n_x // 2, n_active).copy()  # Split abscissae into halves
    fr, fl = fj.reshape(2, n_x // 2, n_active)         # Split function values accordingly
    wr, wl = wj.reshape(2, n_x // 2, n_active)         # Split weights accordingly

    invalid_r = ~np.isfinite(fr) | (wr == 0)  # Invalid indices for right side
    invalid_l = ~np.isfinite(fl) | (wl == 0)  # Invalid indices for left side

    xr[invalid_r] = -np.inf  # Set invalid values to negative infinity for right side
    ir = np.argmax(xr, axis=0, keepdims=True)  # Index of maximum abscissa on right side
    xr_max = np.take_along_axis(xr, ir, axis=0)[0]  # Maximum abscissa value
    fr_max = np.take_along_axis(fr, ir, axis=0)[0]  # Corresponding function value
    wr_max = np.take_along_axis(wr, ir, axis=0)[0]  # Corresponding weight

    # Points outside limits should have zero weights
    invalid = (xj <= a) | (xj >= b)
    wj[invalid] = 0
    # 计算当前级别中的最大横坐标是否大于之前所有级别的最大横坐标
    j = xr_max > xr0
    # 更新记录当前级别中的最大横坐标、函数值和权重
    xr0[j] = xr_max[j]
    fr0[j] = fr_max[j]
    wr0[j] = wr_max[j]

    # 将无效的横坐标索引设置为无穷大
    xl[invalid_l] = np.inf
    # 找到当前级别中最小横坐标的整数索引
    il = np.argmin(xl, axis=0, keepdims=True)
    # 获取该索引处的横坐标、函数值和权重
    xl_min = np.take_along_axis(xl, il, axis=0)[0]
    fl_min = np.take_along_axis(fl, il, axis=0)[0]
    wl_min = np.take_along_axis(wl, il, axis=0)[0]
    # 找到当前级别中最小横坐标是否小于之前所有级别的最小横坐标
    j = xl_min < xl0
    # 更新记录当前级别中的最小横坐标、函数值和权重
    xl0[j] = xl_min[j]
    fl0[j] = fl_min[j]
    wl0[j] = wl_min[j]

    # 计算误差估计值 `d4`，即左侧或右侧项的较大值
    flwl0 = fl0 + np.log(wl0) if work.log else fl0 * wl0  # 左侧项
    frwr0 = fr0 + np.log(wr0) if work.log else fr0 * wr0  # 右侧项
    magnitude = np.real if work.log else np.abs
    work.d4 = np.maximum(magnitude(flwl0), magnitude(frwr0))

    # 处理由于接近奇点导致函数值数值无穷大的情况，采用替换策略
    fr0b = np.broadcast_to(fr0[np.newaxis, :], fr.shape)
    fl0b = np.broadcast_to(fl0[np.newaxis, :], fl.shape)
    fr[invalid_r] = fr0b[invalid_r]
    fl[invalid_l] = fl0b[invalid_l]

    # 当 wj 为零时，log 函数会发出警告
    fjwj = fj + np.log(work.wj) if work.log else fj * work.wj

    # 更新积分估计值
    Sn = (special.logsumexp(fjwj + np.log(work.h), axis=-1) if work.log
          else np.sum(fjwj, axis=-1) * work.h)

    # 更新工作空间中的记录值
    work.xr0, work.fr0, work.wr0 = xr0, fr0, wr0
    work.xl0, work.fl0, work.wl0 = xl0, fl0, wl0

    return fjwj, Sn
# 估算误差，根据 [1] 第 5 节进行计算
def _estimate_error(work):
    if work.n == 0 or work.nit == 0:
        # 如果工作的 n 为 0 或迭代次数为 0，根据文献建议误差应该为 NaN
        nan = np.full_like(work.Sn, np.nan)
        return nan, nan

    indices = _pair_cache.indices

    n_active = len(work.Sn)  # 活跃元素的数量
    axis_kwargs = dict(axis=-1, keepdims=True)

    # 如果 Sk 的最后一个维度长度为 0，表示需要一个起始值，此时尚未计算低级别的积分估计
    if work.Sk.shape[-1] == 0:
        h = 2 * work.h  # 此级别的步长
        n_x = indices[work.n]  # 达到此级别的横坐标数量
        # 从所有级别的右和左 fjwj 项中连接的最后一个轴上的项。仅获取直到此级别的项。
        fjwj_rl = work.fjwj.reshape(n_active, 2, -1)
        fjwj = fjwj_rl[:, :, :n_x].reshape(n_active, 2*n_x)
        # 计算此级别的 Euler-Maclaurin 和
        Snm1 = (special.logsumexp(fjwj, **axis_kwargs) + np.log(h) if work.log
                else np.sum(fjwj, **axis_kwargs) * h)
        work.Sk = np.concatenate((Snm1, work.Sk), axis=-1)

    if work.n == 1:
        # 当 n 为 1 时，返回全 NaN 数组作为误差
        nan = np.full_like(work.Sn, np.nan)
        return nan, nan

    # 如果 Sk 的最后一个维度长度小于 2，表示需要第二个级别的起始值
    if work.Sk.shape[-1] < 2:
        h = 4 * work.h  # 此级别的步长
        n_x = indices[work.n-1]  # 达到此级别的横坐标数量
        # 从所有级别的右和左 fjwj 项中连接的最后一个轴上的项。仅获取直到此级别的项。
        fjwj_rl = work.fjwj.reshape(len(work.Sn), 2, -1)
        fjwj = fjwj_rl[..., :n_x].reshape(n_active, 2*n_x)
        # 计算此级别的 Euler-Maclaurin 和
        Snm2 = (special.logsumexp(fjwj, **axis_kwargs) + np.log(h) if work.log
                else np.sum(fjwj, **axis_kwargs) * h)
        work.Sk = np.concatenate((Snm2, work.Sk), axis=-1)

    # 获取 Sk 中的倒数第二个和倒数第一个值
    Snm2 = work.Sk[..., -2]
    Snm1 = work.Sk[..., -1]

    e1 = work.eps  # 错误估计的阈值
    # 如果 `work.log` 为真，则进行以下操作
    if work.log:
        # 计算 `e1` 的自然对数
        log_e1 = np.log(e1)
        
        # 当前仅支持在对数尺度下的实积分。所有复数值都有虚部以 `pi*j` 的增量，
        # 这仅携带了原始积分的符号信息，因此在这里使用 `np.real` 相当于在实数尺度下的绝对值。
        # 使用 `np.real` 是为了处理实部，即绝对值在实数尺度下的应用。
        d1 = np.real(special.logsumexp([work.Sn, Snm1 + work.pi*1j], axis=0))
        d2 = np.real(special.logsumexp([work.Sn, Snm2 + work.pi*1j], axis=0))
        
        # 计算 `log_e1` 与 `work.fjwj` 的实部的最大值之和
        d3 = log_e1 + np.max(np.real(work.fjwj), axis=-1)
        
        # 从 `work` 对象中获取 `d4` 的值
        d4 = work.d4
        
        # 计算相对误差的估计值 `aerr`，取各项中的最大值
        aerr = np.max([d1 ** 2 / d2, 2 * d1, d3, d4], axis=0)
        
        # 计算相对误差 `rerr`，取 `log_e1` 和 `aerr` 与 `work.Sn` 的实部的差的最大值
        rerr = np.maximum(log_e1, aerr - np.real(work.Sn))
    
    # 如果 `work.log` 为假，则执行以下操作
    else:
        # 注意：这里不必要地计算每个值的对数的十进制对数。
        
        # 计算 `work.Sn` 与 `Snm1` 之间的绝对差
        d1 = np.abs(work.Sn - Snm1)
        
        # 计算 `work.Sn` 与 `Snm2` 之间的绝对差
        d2 = np.abs(work.Sn - Snm2)
        
        # 计算 `e1` 乘以 `work.fjwj` 的绝对值的最大值
        d3 = e1 * np.max(np.abs(work.fjwj), axis=-1)
        
        # 从 `work` 对象中获取 `d4` 的值
        d4 = work.d4
        
        # 计算相对误差的估计值 `aerr`，取各项中的最大值
        # 使用 `np.max` 和 `np.abs` 对 `d1` 和 `d2` 进行操作
        with np.errstate(divide='ignore'):
            aerr = np.max([d1**(np.log(d1)/np.log(d2)), d1**2, d3, d4], axis=0)
        
        # 计算相对误差 `rerr`，取 `e1` 与 `aerr` 除以 `work.Sn` 的绝对值的最大值
        rerr = np.maximum(e1, aerr/np.abs(work.Sn))
    
    # 返回相对误差 `rerr` 和误差估计值 `aerr`，并将其形状重塑为 `work.Sn` 的形状
    return rerr, aerr.reshape(work.Sn.shape)
# 将积分变换为有限形式 a < b
# 对于 b < a 的情况，交换限制并最终结果乘以 -1
# 对于右侧的无限限制，使用替换 x = 1/t - 1 + a
# 对于左侧的无限限制，我们将 x = -x，然后像上面一样处理
# 对于无限限制，我们替换 x = t / (1-t**2)
def _transform_integrals(a, b):
    # 判断是否需要对 a, b 进行反向处理
    negative = b < a
    a[negative], b[negative] = b[negative], a[negative]

    # 处理同时为无限的情况
    abinf = np.isinf(a) & np.isinf(b)
    a[abinf], b[abinf] = -1, 1

    # 处理左侧为无限的情况
    ainf = np.isinf(a)
    a[ainf], b[ainf] = -b[ainf], -a[ainf]

    # 处理右侧为无限的情况，并保存原始的 a
    binf = np.isinf(b)
    a0 = a.copy()
    a[binf], b[binf] = 0, 1

    # 返回处理后的 a, b 以及一些标志值
    return a, b, a0, negative, abinf, ainf, binf


# 输入验证和标准化
def _tanhsinh_iv(f, a, b, log, maxfun, maxlevel, minlevel,
                 atol, rtol, args, preserve_shape, callback):
    # 检查函数 f 是否可调用
    message = '`f` must be callable.'
    if not callable(f):
        raise ValueError(message)

    # 确保 a 和 b 都是实数
    message = 'All elements of `a` and `b` must be real numbers.'
    a, b = np.broadcast_arrays(a, b)
    if np.any(np.iscomplex(a)) or np.any(np.iscomplex(b)):
        raise ValueError(message)

    # 检查 log 是否为布尔类型
    message = '`log` must be True or False.'
    if log not in {True, False}:
        raise ValueError(message)
    log = bool(log)

    # 如果 atol 为 None，则根据 log 设置默认值
    if atol is None:
        atol = -np.inf if log else 0

    # 如果 rtol 为 None，则设置临时值为 0
    rtol_temp = rtol if rtol is not None else 0.

    # 将 atol 和 rtol 放入数组 params，并检查它们必须是实数
    params = np.asarray([atol, rtol_temp, 0.])
    message = "`atol` and `rtol` must be real numbers."
    if not np.issubdtype(params.dtype, np.floating):
        raise ValueError(message)

    # 如果 log 为 True，则确保 atol 和 rtol 不是正无穷
    if log:
        message = '`atol` and `rtol` may not be positive infinity.'
        if np.any(np.isposinf(params)):
            raise ValueError(message)
    else:
        # 如果 log 为 False，则确保 atol 和 rtol 非负且有限
        message = '`atol` and `rtol` must be non-negative and finite.'
        if np.any(params < 0) or np.any(np.isinf(params)):
            raise ValueError(message)
    atol = params[0]
    rtol = rtol if rtol is None else params[1]

    # 定义一个大整数 BIGINT
    BIGINT = float(2**62)

    # 如果 maxfun 和 maxlevel 都为 None，则将 maxlevel 设为 10
    if maxfun is None and maxlevel is None:
        maxlevel = 10

    # 如果 maxfun 为 None，则设为 BIGINT
    maxfun = BIGINT if maxfun is None else maxfun
    # 如果 maxlevel 为 None，则设为 BIGINT
    maxlevel = BIGINT if maxlevel is None else maxlevel

    # 确保 maxfun、maxlevel 和 minlevel 都是整数且非负
    message = '`maxfun`, `maxlevel`, and `minlevel` must be integers.'
    params = np.asarray([maxfun, maxlevel, minlevel])
    if not (np.issubdtype(params.dtype, np.number)
            and np.all(np.isreal(params))
            and np.all(params.astype(np.int64) == params)):
        raise ValueError(message)
    message = '`maxfun`, `maxlevel`, and `minlevel` must be non-negative.'
    if np.any(params < 0):
        raise ValueError(message)
    maxfun, maxlevel, minlevel = params.astype(np.int64)
    # 将 minlevel 设置为 min(minlevel, maxlevel)
    minlevel = min(minlevel, maxlevel)

    # 如果 args 不可迭代，则转为元组
    if not np.iterable(args):
        args = (args,)

    # 确保 preserve_shape 是布尔类型
    message = '`preserve_shape` must be True or False.'
    if preserve_shape not in {True, False}:
        raise ValueError(message)
    # 检查回调函数是否存在且可调用，如果回调函数存在但不可调用，抛出值错误异常
    if callback is not None and not callable(callback):
        raise ValueError('`callback` must be callable.')

    # 返回元组 (f, a, b, log, maxfun, maxlevel, minlevel,
    #          atol, rtol, args, preserve_shape, callback)
    return (f, a, b, log, maxfun, maxlevel, minlevel,
            atol, rtol, args, preserve_shape, callback)
def _logsumexp(x, axis=0):
    # 计算对数和指数函数的和，处理空数组情况
    x = np.asarray(x)  # 将输入转换为 NumPy 数组
    shape = list(x.shape)  # 获取数组的形状
    if shape[axis] == 0:  # 如果指定轴的长度为 0
        shape.pop(axis)  # 移除该轴
        return np.full(shape, fill_value=-np.inf, dtype=x.dtype)  # 返回填充了 -inf 的数组
    else:
        return special.logsumexp(x, axis=axis)  # 计算对数和指数函数的和


def _nsum_iv(f, a, b, step, args, log, maxterms, atol, rtol):
    # 输入验证和标准化

    message = '`f` must be callable.'
    if not callable(f):  # 检查 `f` 是否可调用
        raise ValueError(message)

    message = 'All elements of `a`, `b`, and `step` must be real numbers.'
    a, b, step = np.broadcast_arrays(a, b, step)  # 广播数组以保证统一形状
    dtype = np.result_type(a.dtype, b.dtype, step.dtype)  # 获取结果类型
    if not np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.complexfloating):
        raise ValueError(message)  # 如果类型不是实数或是复数，则引发错误

    valid_a = np.isfinite(a)  # 检查 `a` 是否有限
    valid_b = b >= a  # 检查 `b` 是否大于等于 `a`，NaN 会被视为 False
    valid_step = np.isfinite(step) & (step > 0)  # 检查 `step` 是否有限且大于 0
    valid_abstep = valid_a & valid_b & valid_step  # 组合有效性条件

    message = '`log` must be True or False.'
    if log not in {True, False}:  # 检查 `log` 是否为 True 或 False
        raise ValueError(message)

    if atol is None:
        atol = -np.inf if log else 0  # 如果 `atol` 为 None，则根据 `log` 设置默认值

    rtol_temp = rtol if rtol is not None else 0.  # 如果 `rtol` 为 None，则设置为 0

    params = np.asarray([atol, rtol_temp, 0.])  # 将参数转换为 NumPy 数组
    message = "`atol` and `rtol` must be real numbers."
    if not np.issubdtype(params.dtype, np.floating):  # 检查 `atol` 和 `rtol` 是否为实数
        raise ValueError(message)

    if log:
        message = '`atol`, `rtol` may not be positive infinity or NaN.'
        if np.any(np.isposinf(params) | np.isnan(params)):  # 检查 `atol` 和 `rtol` 是否为正无穷或 NaN
            raise ValueError(message)
    else:
        message = '`atol`, and `rtol` must be non-negative and finite.'
        if np.any((params < 0) | (~np.isfinite(params))):  # 检查 `atol` 和 `rtol` 是否非负有限
            raise ValueError(message)
    atol = params[0]  # 获取 `atol`
    rtol = rtol if rtol is None else params[1]  # 获取 `rtol`

    maxterms_int = int(maxterms)  # 将 `maxterms` 转换为整数
    if maxterms_int != maxterms or maxterms < 0:  # 检查 `maxterms` 是否为非负整数
        message = "`maxterms` must be a non-negative integer."
        raise ValueError(message)

    if not np.iterable(args):  # 检查 `args` 是否可迭代
        args = (args,)  # 如果不可迭代，转换为元组

    return f, a, b, step, valid_abstep, args, log, maxterms_int, atol, rtol


def _nsum(f, a, b, step=1, args=(), log=False, maxterms=int(2**20), atol=None,
          rtol=None):
    r"""Evaluate a convergent sum.

    For finite `b`, this evaluates::

        f(a + np.arange(n)*step).sum()

    where ``n = int((b - a) / step) + 1``. If `f` is smooth, positive, and
    monotone decreasing, `b` may be infinite, in which case the infinite sum
    is approximated using integration.

    Parameters
    ----------
    ```
    f : callable
        # 要对项求和的函数。其签名必须是::

        #     f(x: ndarray, *args) -> ndarray

        # 其中 `x` 的每个元素都是有限的实数，`args` 是一个元组，可以包含与 `x` 可广播的任意数量的数组。`f` 必须表示 `x` 的平滑、正值和单调递减函数；_nsum 不检查这些条件是否满足，如果违反可能会返回错误的结果。

    a, b : array_like
        # 被求和项的实数下限和上限。必须可以广播。
        # 每个 `a` 的元素必须是有限的，并且小于对应的 `b` 的元素，但是 `b` 的元素可以是无限的。

    step : array_like
        # 求和项之间的有限正实数步长。必须可以广播，与 `a` 和 `b` 广播。
    
    args : tuple, optional
        # 要传递给 `f` 的额外位置参数。必须与 `a`、`b` 和 `step` 可广播的数组相匹配。
        # 如果要求和的可调用函数需要与 `a`、`b` 和 `step` 不可广播的参数，用 `f` 包装它。参见示例。
    
    log : bool, default: False
        # 设置为 True 表示 `f` 返回项的对数，并且 `atol` 和 `rtol` 表示绝对和相对误差的对数。
        # 在这种情况下，结果对象将包含总和和误差的对数。对于可能因数值下溢或溢出导致不准确的求和项非常有用。
    
    maxterms : int, default: 2**32
        # 直接求和时要评估的最大项数。对于输入验证和积分评估可能执行额外的函数评估。
    
    atol, rtol : float, optional
        # 绝对终止容差（默认值：0）和相对终止容差（默认值为 ``eps**0.5``，其中 ``eps`` 是结果 dtype 的精度），分别是非负的有限数，如果 `log` 是 False，则必须如此；如果 `log` 是 True，则必须表示为非负和有限数的对数。
    res : _RichResult
        # 定义一个变量 res，类型为 `_RichResult`，包含以下属性描述
        An instance of `scipy._lib._util._RichResult` with the following
        attributes. (The descriptions are written as though the values will be
        scalars; however, if `func` returns an array, the outputs will be
        arrays of the same shape.)
        
        success : bool
            # 表示算法是否成功终止，值为 `True` 表示成功（状态为 `0`）。
        status : int
            # 表示算法的退出状态的整数表示。
            ``0`` : 算法收敛到指定的容差。
            ``-1`` : `a`、`b` 或 `step` 的元素无效。
            ``-2`` : 数值积分达到其迭代限制；总和可能发散。
            ``-3`` : 遇到非有限值。
        sum : float
            # 估计的总和值。
        error : float
            # 假设所有项均为非负数，估计的绝对误差。
        nfev : int
            # 调用 `func` 评估的点数。

    See Also
    --------
    tanhsinh

    Notes
    -----
    # 无限求和的方法与无限级数的收敛积分检验相关：
    # 假设 `step` 大小为 1 以简化表述，单调递减函数的和受下限

    .. math::

        \int_u^\infty f(x) dx \leq \sum_{k=u}^\infty f(k) \leq \int_u^\infty f(x) dx + f(u)

    # 让 :math:`a` 表示 `a`，:math:`n` 表示 `maxterms`，:math:`\epsilon_a`
    # 表示 `atol`，:math:`\epsilon_r` 表示 `rtol`。
    # 实现首先评估积分 :math:`S_l=\int_a^\infty f(x) dx` 作为无限和的下界。
    # 然后，寻找一个值 :math:`c > a`，使得 :math:`f(c) < \epsilon_a + S_l \epsilon_r`，
    # 如果存在；否则，令 :math:`c = a + n`。然后无限和被近似为

    .. math::

        \sum_{k=a}^{c-1} f(k) + \int_c^\infty f(x) dx + f(c)/2

    # 报告的误差是 :math:`f(c)/2` 加上数值积分的误差估计。
    # 上述方法适用于非单位 `step` 和对于直接求和太大的有限 `b`，即 ``b - a + 1 > maxterms``。

    References
    ----------
    [1] Wikipedia. "Integral test for convergence."
    https://en.wikipedia.org/wiki/Integral_test_for_convergence

    Examples
    --------
    # 计算倒数平方整数的无限和。
    >>> import numpy as np
    >>> from scipy.integrate._tanhsinh import _nsum
    >>> res = _nsum(lambda k: 1/k**2, 1, np.inf, maxterms=1e3)
    >>> ref = np.pi**2/6  # 真实值
    >>> res.error  # 估计误差
    4.990014980029223e-07
    >>> (res.sum - ref)/ref  # 真实误差
    -1.0101760641302586e-10
    >>> res.nfev  # callable 被评估的点数
    1142
    
    # 计算整数的倒数的幂和 ``p`` 的无限和。
    >>> from scipy import special  # 导入 scipy 库中的 special 模块，用于数学特殊函数的计算
    >>> p = np.arange(2, 10)  # 创建一个 numpy 数组 p，包含从2到9的整数
    >>> res = _nsum(lambda k, p: 1/k**p, 1, np.inf, maxterms=1e3, args=(p,))  # 调用 _nsum 函数计算无穷级数和，lambda 函数定义了级数的一般形式，使用 p 作为参数
    >>> ref = special.zeta(p, 1)  # 使用 scipy 的 zeta 函数计算 Riemann zeta 函数在给定参数 p 和 s=1 处的值
    >>> np.allclose(res.sum, ref)  # 检查 res 的和是否与 ref 接近，返回布尔值
    True
    
    """ # noqa: E501
    # Potential future work:
    # - more careful testing of when `b` is slightly less than `a` plus an
    #   integer multiple of step (needed before this is public)
    # - improve error estimate of `_direct` sum
    # - add other methods for convergence acceleration (Richardson, epsilon)
    # - support infinite lower limit?
    # - support negative monotone increasing functions?
    # - b < a / negative step?
    # - complex-valued function?
    # - check for violations of monotonicity?

    # Function-specific input validation / standardization
    tmp = _nsum_iv(f, a, b, step, args, log, maxterms, atol, rtol)
    f, a, b, step, valid_abstep, args, log, maxterms, atol, rtol = tmp
    # 调用 _nsum_iv 函数对输入参数进行验证和标准化，返回标准化后的各个参数

    # Additional elementwise algorithm input validation / standardization
    tmp = eim._initialize(f, (a,), args, complex_ok=False)
    f, xs, fs, args, shape, dtype, xp = tmp
    # 使用 eim 模块的 _initialize 函数对额外的算法输入参数进行验证和标准化，返回标准化后的各个参数

    # Finish preparing `a`, `b`, and `step` arrays
    a = xs[0]
    # 将 xs 中的第一个元素赋值给 a
    b = np.broadcast_to(b, shape).ravel().astype(dtype)
    # 使用广播将 b 扩展为与 shape 相同的形状，然后转换为指定的数据类型
    step = np.broadcast_to(step, shape).ravel().astype(dtype)
    # 使用广播将 step 扩展为与 shape 相同的形状，然后转换为指定的数据类型
    valid_abstep = np.broadcast_to(valid_abstep, shape).ravel()
    # 使用广播将 valid_abstep 扩展为与 shape 相同的形状
    nterms = np.floor((b - a) / step)
    # 计算 nterms，即由 (b - a) / step 向下取整得到的数组
    b = a + nterms*step
    # 更新 b，确保它们符合计算要求的值范围

    # Define constants
    eps = np.finfo(dtype).eps
    # 计算给定数据类型的机器精度
    zero = np.asarray(-np.inf if log else 0, dtype=dtype)[()]
    # 根据 log 参数定义 zero 常量，若 log 为真则为 -∞，否则为 0
    if rtol is None:
        rtol = 0.5*np.log(eps) if log else eps**0.5
    # 若 rtol 未定义，则根据 log 参数设置默认值
    constants = (dtype, log, eps, zero, rtol, atol, maxterms)
    # 将常量打包为元组

    # Prepare result arrays
    S = np.empty_like(a)
    # 创建与 a 相同形状的空数组 S
    E = np.empty_like(a)
    # 创建与 a 相同形状的空数组 E
    status = np.zeros(len(a), dtype=int)
    # 创建整型数组 status，长度与 a 相同，初始值为零
    nfev = np.ones(len(a), dtype=int)  # one function evaluation above
    # 创建整型数组 nfev，长度与 a 相同，初始值为一，表示已执行一次函数求值

    # Branch for direct sum evaluation / integral approximation / invalid input
    i1 = (nterms + 1 <= maxterms) & valid_abstep
    # 创建布尔数组 i1，判断是否可以直接求和
    i2 = (nterms + 1 > maxterms) & valid_abstep
    # 创建布尔数组 i2，判断是否需要间接求和
    i3 = ~valid_abstep
    # 创建布尔数组 i3，判断输入是否无效

    if np.any(i1):
        args_direct = [arg[i1] for arg in args]
        # 根据布尔数组 i1 选择参数 args 的子集 args_direct
        tmp = _direct(f, a[i1], b[i1], step[i1], args_direct, constants)
        # 调用 _direct 函数计算直接求和结果
        S[i1], E[i1] = tmp[:-1]
        # 将结果存储到 S 和 E 数组中
        nfev[i1] += tmp[-1]
        # 更新 nfev 中相应索引的值
        status[i1] = -3 * (~np.isfinite(S[i1]))
        # 根据直接求和结果更新 status 中相应索引的值

    if np.any(i2):
        args_indirect = [arg[i2] for arg in args]
        # 根据布尔数组 i2 选择参数 args 的子集 args_indirect
        tmp = _integral_bound(f, a[i2], b[i2], step[i2], args_indirect, constants)
        # 调用 _integral_bound 函数计算积分边界估计结果
        S[i2], E[i2], status[i2] = tmp[:-1]
        # 将结果存储到 S、E 和 status 数组中
        nfev[i2] += tmp[-1]
        # 更新 nfev 中相应索引的值

    if np.any(i3):
        S[i3], E[i3] = np.nan, np.nan
        # 对于无效输入，将 S 和 E 数组中相应索引位置设为 NaN
        status[i3] = -1
        # 更新 status 中相应索引的值为 -1，表示无效输入

    # Return results
    S, E = S.reshape(shape)[()], E.reshape(shape)[()]
    # 将 S 和 E 数组调整为原始形状
    status, nfev = status.reshape(shape)[()], nfev.reshape(shape)[()]
    # 将 status 和 nfev 数组调整为原始形状
    return _RichResult(sum=S, error=E, status=status, success=status == 0,
                       nfev=nfev)
    # 返回 _RichResult 对象，包含计算结果、错误估计、状态和求值次数
# 直接计算求和

# 在分布的上下文中使用时，`args`应包含分布参数。我们已经进行了广播以简化，但是当分布参数相同而求和限制不同时，
# 可以减少函数评估。大致如下：
# - 计算在 min(a) 和 max(b) 之间所有点的函数值，
# - 计算累积和，
# - 取与 b 和 a 对应的累积和元素的差值。
# 这部分留待将来优化。

def _direct(f, a, b, step, args, constants, inclusive=True):
    dtype, log, eps, zero, _, _, _ = constants

    # 为了允许在单个向量化调用中进行计算，找到需要评估函数的点的最大数量（在所有切片中）。
    # 注意：如果 `inclusive` 是 `True`，那么在求和中我们需要额外的 `1` 项。
    # 我认为在 Python 中使用 `True` 作为 `1` 不是很好的风格，因此在使用之前显式地将其转换为 `int`。
    inclusive_adjustment = int(inclusive)
    steps = np.round((b - a) / step) + inclusive_adjustment
    # 等价地，steps = np.round((b - a) / step) + inclusive
    max_steps = int(np.max(steps))

    # 在每个切片中，函数将在相同数量的点上评估，但是超出右求和限制 `b` 的过多点将被替换为 NaN，
    # 以（可能）减少这些不必要计算的时间。为了与其他逐元素算法保持一致，使用一个新的最后轴进行这些计算。
    a2, b2, step2 = a[:, np.newaxis], b[:, np.newaxis], step[:, np.newaxis]
    args2 = [arg[:, np.newaxis] for arg in args]
    ks = a2 + np.arange(max_steps, dtype=dtype) * step2
    i_nan = ks >= (b2 + inclusive_adjustment * step2 / 2)
    ks[i_nan] = np.nan
    fs = f(ks, *args2)

    # 在 NaN 处评估的函数值也为 NaN，在求和中这些 NaN 将被归零处理。
    # 在某些情况下，逐片循环可能比这样向量化更快。这是可以后续添加的优化。
    fs[i_nan] = zero
    nfev = max_steps - i_nan.sum(axis=-1)
    S = _logsumexp(fs, axis=-1) if log else np.sum(fs, axis=-1)
    # 粗略的、非保守的误差估计。参见 gh-19667 以获取改进的建议。
    E = np.real(S) + np.log(eps) if log else eps * abs(S)
    return S, E, nfev


# 用积分估计边界值的和

def _integral_bound(f, a, b, step, args, constants):
    dtype, log, _, _, rtol, atol, maxterms = constants
    log2 = np.log(2, dtype=dtype)

    # 得到和的下界，并计算有效的绝对容差
    lb = _tanhsinh(f, a, b, args=args, atol=atol, rtol=rtol, log=log)
    tol = np.broadcast_to(atol, lb.integral.shape)
    tol = _logsumexp((tol, rtol + lb.integral)) if log else tol + rtol * lb.integral
    i_skip = lb.status < 0  # 如果积分发散，则避免不必要的函数评估
    tol[i_skip] = np.nan
    status = lb.status
    # 在 `_direct` 中，我们需要为评估函数的点添加一个临时的新轴。
    # 在末尾附加轴，以保持与其他逐元素算法的一致性。
    a2 = a[..., np.newaxis]
    step2 = step[..., np.newaxis]
    args2 = [arg[..., np.newaxis] for arg in args]

    # 找到小于容差的项的位置（如果可能）
    log2maxterms = np.floor(np.log2(maxterms)) if maxterms else 0
    n_steps = np.concatenate([2**np.arange(0, log2maxterms), [maxterms]], dtype=dtype)
    nfev = len(n_steps)
    ks = a2 + n_steps * step2
    fks = f(ks, *args2)
    nt = np.minimum(np.sum(fks > tol[:, np.newaxis], axis=-1),  n_steps.shape[-1]-1)
    n_steps = n_steps[nt]

    # 直接评估到该项的总和
    k = a + n_steps * step
    left, left_error, left_nfev = _direct(f, a, k, step, args,
                                          constants, inclusive=False)
    i_skip |= np.isposinf(left)  # 如果总和不是有限的，则没有继续计算的意义
    status[np.isposinf(left)] = -3
    k[i_skip] = np.nan

    # 使用积分来估计剩余的总和
    # 可能的优化方案：如果没有小于容差的项，则无需计算更好的精度积分。
    # 类似于：
    # atol = np.maximum(atol, np.minimum(fk/2 - fb/2))
    # rtol = np.maximum(rtol, np.minimum((fk/2 - fb/2)/left))
    # 其中 `fk`/`fb` 是当前在下面计算的。
    right = _tanhsinh(f, k, b, args=args, atol=atol, rtol=rtol, log=log)

    # 计算从各部分得到的完整估计和误差
    fk = fks[np.arange(len(fks)), nt]
    fb = f(b, *args)
    nfev += 1
    if log:
        log_step = np.log(step)
        S_terms = (left, right.integral - log_step, fk - log2, fb - log2)
        S = _logsumexp(S_terms, axis=0)
        E_terms = (left_error, right.error - log_step, fk-log2, fb-log2+np.pi*1j)
        E = _logsumexp(E_terms, axis=0).real
    else:
        S = left + right.integral/step + fk/2 + fb/2
        E = left_error + right.error/step + fk/2 - fb/2
    status[~i_skip] = right.status[~i_skip]
    return S, E, status, left_nfev + right.nfev + nfev + lb.nfev
```