# `D:\src\scipysrc\scipy\scipy\optimize\_chandrupatla.py`

```
import math
import numpy as np
import scipy._lib._elementwise_iterative_method as eim
from scipy._lib._util import _RichResult
from scipy._lib._array_api import xp_clip, xp_minimum, xp_sign

# 导入所需的模块和类
# math: 提供数学函数
# numpy: 提供多维数组和相关操作
# scipy._lib._elementwise_iterative_method: 导入特定的迭代方法模块
# scipy._lib._util._RichResult: 导入丰富结果类
# scipy._lib._array_api: 导入数组操作相关接口

# TODO:
# - (maybe?) don't use fancy indexing assignment
# - figure out how to replace the new `try`/`except`s

def _chandrupatla(func, a, b, *, args=(), xatol=None, xrtol=None,
                  fatol=None, frtol=0, maxiter=None, callback=None):
    """Find the root of an elementwise function using Chandrupatla's algorithm.

    For each element of the output of `func`, `chandrupatla` seeks the scalar
    root that makes the element 0. This function allows for `a`, `b`, and the
    output of `func` to be of any broadcastable shapes.

    Parameters
    ----------
    func : callable
        The function whose root is desired. The signature must be::

            func(x: ndarray, *args) -> ndarray

         where each element of ``x`` is a finite real and ``args`` is a tuple,
         which may contain an arbitrary number of components of any type(s).
         ``func`` must be an elementwise function: each element ``func(x)[i]``
         must equal ``func(x[i])`` for all indices ``i``. `_chandrupatla`
         seeks an array ``x`` such that ``func(x)`` is an array of zeros.
    a, b : array_like
        The lower and upper bounds of the root of the function. Must be
        broadcastable with one another.
    args : tuple, optional
        Additional positional arguments to be passed to `func`.
    xatol, xrtol, fatol, frtol : float, optional
        Absolute and relative tolerances on the root and function value.
        See Notes for details.
    maxiter : int, optional
        The maximum number of iterations of the algorithm to perform.
        The default is the maximum possible number of bisections within
        the (normal) floating point numbers of the relevant dtype.
    callback : callable, optional
        An optional user-supplied function to be called before the first
        iteration and after each iteration.
        Called as ``callback(res)``, where ``res`` is a ``_RichResult``
        similar to that returned by `_chandrupatla` (but containing the current
        iterate's values of all variables). If `callback` raises a
        ``StopIteration``, the algorithm will terminate immediately and
        `_chandrupatla` will return a result.

    Returns
    -------
    res : _RichResult
        An instance of `scipy._lib._util._RichResult` with the following
        attributes. The descriptions are written as though the values will be
        scalars; however, if `func` returns an array, the outputs will be
        arrays of the same shape.

        x : float
            The root of the function, if the algorithm terminated successfully.
        nfev : int
            The number of times the function was called to find the root.
        nit : int
            The number of iterations of Chandrupatla's algorithm performed.
        status : int
            An integer representing the exit status of the algorithm.
            ``0`` : The algorithm converged to the specified tolerances.
            ``-1`` : The algorithm encountered an invalid bracket.
            ``-2`` : The maximum number of iterations was reached.
            ``-3`` : A non-finite value was encountered.
            ``-4`` : Iteration was terminated by `callback`.
            ``1`` : The algorithm is proceeding normally (in `callback` only).
        success : bool
            ``True`` when the algorithm terminated successfully (status ``0``).
        fun : float
            The value of `func` evaluated at `x`.
        xl, xr : float
            The lower and upper ends of the bracket.
        fl, fr : float
            The function value at the lower and upper ends of the bracket.

    Notes
    -----
    Implemented based on Chandrupatla's original paper [1]_.

    If ``xl`` and ``xr`` are the left and right ends of the bracket,
    ``xmin = xl if abs(func(xl)) <= abs(func(xr)) else xr``,
    and ``fmin0 = min(func(a), func(b))``, then the algorithm is considered to
    have converged when ``abs(xr - xl) < xatol + abs(xmin) * xrtol`` or
    ``fun(xmin) <= fatol + abs(fmin0) * frtol``. This is equivalent to the
    termination condition described in [1]_ with ``xrtol = 4e-10``,
    ``xatol = 1e-5``, and ``fatol = frtol = 0``. The default values are
    ``xatol = 4*tiny``, ``xrtol = 4*eps``, ``frtol = 0``, and ``fatol = tiny``,
    where ``eps`` and ``tiny`` are the precision and smallest normal number
    of the result ``dtype`` of function inputs and outputs.

    References
    ----------

    .. [1] Chandrupatla, Tirupathi R.
        "A new hybrid quadratic/bisection algorithm for finding the zero of a
        nonlinear function without using derivatives".
        Advances in Engineering Software, 28(3), 145-149.
        https://doi.org/10.1016/s0965-9978(96)00051-8

    See Also
    --------
    brentq, brenth, ridder, bisect, newton

    Examples
    --------
    >>> from scipy import optimize
    >>> def f(x, c):
    ...     return x**3 - 2*x - c
    >>> c = 5
    >>> res = optimize._chandrupatla._chandrupatla(f, 0, 3, args=(c,))
    >>> res.x
    2.0945514818937463

    >>> c = [3, 4, 5]
    >>> res = optimize._chandrupatla._chandrupatla(f, 0, 3, args=(c,))
    >>> res.x
    array([1.8932892 , 2.        , 2.09455148])
    res = _chandrupatla_iv(func, args, xatol, xrtol,
                           fatol, frtol, maxiter, callback)
    # 调用 _chandrupatla_iv 函数执行迭代算法，并获取返回结果
    func, args, xatol, xrtol, fatol, frtol, maxiter, callback = res
    # 将 _chandrupatla_iv 返回的结果解包并赋值给对应的变量

    # Initialization
    temp = eim._initialize(func, (a, b), args)
    # 调用 _initialize 函数进行初始化，获取返回的临时变量
    func, xs, fs, args, shape, dtype, xp = temp
    # 解包临时变量到各个变量中
    x1, x2 = xs
    f1, f2 = fs
    status = xp.full_like(x1, eim._EINPROGRESS, dtype=xp.int32)  # in progress
    # 使用 xp.full_like 创建一个与 x1 相同形状的数组，填充为 eim._EINPROGRESS，表示正在进行中
    nit, nfev = 0, 2  # two function evaluations performed above
    # 初始化 nit 和 nfev 变量，表示迭代次数和函数评估次数
    finfo = xp.finfo(dtype)
    # 获取 dtype 类型的信息，存储在 finfo 变量中
    xatol = 4*finfo.smallest_normal if xatol is None else xatol
    # 如果 xatol 为 None，则设置为 4 倍的 smallest_normal
    xrtol = 4*finfo.eps if xrtol is None else xrtol
    # 如果 xrtol 为 None，则设置为 4 倍的 eps
    fatol = finfo.smallest_normal if fatol is None else fatol
    # 如果 fatol 为 None，则设置为 smallest_normal
    frtol = frtol * xp_minimum(xp.abs(f1), xp.abs(f2))
    # 计算 frtol，乘以 xp.abs(f1) 和 xp.abs(f2) 的最小值
    maxiter = (math.log2(finfo.max) - math.log2(finfo.smallest_normal)
               if maxiter is None else maxiter)
    # 如果 maxiter 为 None，则设置为计算所得的值，否则使用 maxiter 自身的值
    work = _RichResult(x1=x1, f1=f1, x2=x2, f2=f2, x3=None, f3=None, t=0.5,
                       xatol=xatol, xrtol=xrtol, fatol=fatol, frtol=frtol,
                       nit=nit, nfev=nfev, status=status)
    # 创建 _RichResult 对象 work，存储各种参数和状态
    res_work_pairs = [('status', 'status'), ('x', 'xmin'), ('fun', 'fmin'),
                      ('nit', 'nit'), ('nfev', 'nfev'), ('xl', 'x1'),
                      ('fl', 'f1'), ('xr', 'x2'), ('fr', 'f2')]
    # 创建元组列表 res_work_pairs，用于存储结果的名称和对应的变量名

    def pre_func_eval(work):
        # [1] Figure 1 (first box)
        # 定义 pre_func_eval 函数，用于计算预测点 x
        x = work.x1 + work.t * (work.x2 - work.x1)
        return x

    def post_func_eval(x, f, work):
        # [1] Figure 1 (first diamond and boxes)
        # Note: y/n are reversed in figure; compare to BASIC in appendix
        # 定义 post_func_eval 函数，用于处理函数评估后的结果，更新工作状态
        work.x3, work.f3 = (xp.asarray(work.x2, copy=True),
                            xp.asarray(work.f2, copy=True))
        j = xp.sign(f) == xp.sign(work.f1)
        nj = ~j
        work.x3[j], work.f3[j] = work.x1[j], work.f1[j]
        work.x2[nj], work.f2[nj] = work.x1[nj], work.f1[nj]
        work.x1, work.f1 = x, f
    def check_termination(work):
        # [1] Figure 1 (second diamond)
        # 检查所有终止条件并记录状态。

        # 根据 [1] Section 4 (first two sentences) 进行判断
        i = xp.abs(work.f1) < xp.abs(work.f2)
        # 根据条件选择最小函数值及其对应的参数值
        work.xmin = xp.where(i, work.x1, work.x2)
        work.fmin = xp.where(i, work.f1, work.f2)
        # 初始化终止条件数组
        stop = xp.zeros_like(work.x1, dtype=xp.bool)  # termination condition met

        # 如果函数值容差已满足，则报告成功收敛，不考虑其他条件。
        # 注意，`frtol` 已重新定义为 `frtol = frtol * minimum(f1, f2)`，其中 `f1` 和 `f2` 是在区间两端评估的函数值。
        i = xp.abs(work.fmin) <= work.fatol + work.frtol
        work.status[i] = eim._ECONVERGED
        stop[i] = True

        # 如果区间不再有效，则报告失败（除非已满足函数容差，如上所检测）。
        i = (xp_sign(work.f1) == xp_sign(work.f2)) & ~stop
        NaN = xp.asarray(xp.nan, dtype=work.xmin.dtype)
        work.xmin[i], work.fmin[i], work.status[i] = NaN, NaN, eim._ESIGNERR
        stop[i] = True

        # 如果参数非有限或任一函数值为 NaN，则报告失败。
        x_nonfinite = ~(xp.isfinite(work.x1) & xp.isfinite(work.x2))
        f_nan = xp.isnan(work.f1) & xp.isnan(work.f2)
        i = (x_nonfinite | f_nan) & ~stop
        work.xmin[i], work.fmin[i], work.status[i] = NaN, NaN, eim._EVALUEERR
        stop[i] = True

        # 这是二分法中使用的收敛标准。Chandrupatla 的准则与此等效，只是 `xrtol` 上乘了 4 倍。
        work.dx = xp.abs(work.x2 - work.x1)
        work.tol = xp.abs(work.xmin) * work.xrtol + work.xatol
        i = work.dx < work.tol
        work.status[i] = eim._ECONVERGED
        stop[i] = True

        return stop

    def post_termination_check(work):
        # [1] Figure 1 (third diamond and boxes / Equation 1)
        # 计算 xi1, phi1, alpha
        xi1 = (work.x1 - work.x2) / (work.x3 - work.x2)
        phi1 = (work.f1 - work.f2) / (work.f3 - work.f2)
        alpha = (work.x3 - work.x1) / (work.x2 - work.x1)
        # 根据 Figure 1 中的条件进行判断
        j = ((1 - xp.sqrt(1 - xi1)) < phi1) & (phi1 < xp.sqrt(xi1))

        f1j, f2j, f3j, alphaj = work.f1[j], work.f2[j], work.f3[j], alpha[j]
        t = xp.full_like(alpha, 0.5)
        t[j] = (f1j / (f1j - f2j) * f3j / (f3j - f2j)
                - alphaj * f1j / (f3j - f1j) * f2j / (f2j - f3j))

        # [1] Figure 1 (last box; see also BASIC in appendix with comment
        # "Adjust T Away from the Interval Boundary")
        # 调整 t 以避开区间边界
        tl = 0.5 * work.tol / work.dx
        work.t = xp_clip(t, tl, 1 - tl)
    # 定义函数 customize_result，定制化处理优化结果
    def customize_result(res, shape):
        # 从结果字典中获取 xl, xr, fl, fr 四个关键字段
        xl, xr, fl, fr = res['xl'], res['xr'], res['fl'], res['fr']
        # 根据条件 i，选择更新 res 字典中的 xl 和 xr 字段
        i = res['xl'] < res['xr']
        res['xl'] = xp.where(i, xl, xr)
        res['xr'] = xp.where(i, xr, xl)
        # 根据条件 i，选择更新 res 字典中的 fl 和 fr 字段
        res['fl'] = xp.where(i, fl, fr)
        res['fr'] = xp.where(i, fr, fl)
        # 返回形状 shape，但函数并未使用该返回值
        return shape

    # 调用外部传入的 eim._loop 函数，进行优化工作的迭代计算
    return eim._loop(work, callback, shape, maxiter, func, args, dtype,
                     pre_func_eval, post_func_eval, check_termination,
                     post_termination_check, customize_result, res_work_pairs,
                     xp=xp)
# 输入验证函数 `_chandrupatla_iv`，用于验证输入参数的合法性和完整性

if not callable(func):
    # 如果 `func` 不可调用，则抛出值错误异常
    raise ValueError('`func` must be callable.')

if not np.iterable(args):
    # 如果 `args` 不可迭代，则将其转换为包含单个元素的元组
    args = (args,)

# 确保容差值为浮点数，不是数组；可以使用 NumPy 来处理
tols = np.asarray([xatol if xatol is not None else 1,
                   xrtol if xrtol is not None else 1,
                   fatol if fatol is not None else 1,
                   frtol if frtol is not None else 1])
if (not np.issubdtype(tols.dtype, np.number) or np.any(tols < 0)
        or np.any(np.isnan(tols)) or tols.shape != (4,)):
    # 确保容差值是非负标量
    raise ValueError('Tolerances must be non-negative scalars.')

if maxiter is not None:
    maxiter_int = int(maxiter)
    if maxiter != maxiter_int or maxiter < 0:
        # 如果 `maxiter` 不是非负整数，则抛出值错误异常
        raise ValueError('`maxiter` must be a non-negative integer.')

if callback is not None and not callable(callback):
    # 如果 `callback` 不可调用，则抛出值错误异常
    raise ValueError('`callback` must be callable.')

# 返回验证后的参数 `func, args, xatol, xrtol, fatol, frtol, maxiter, callback`
return func, args, xatol, xrtol, fatol, frtol, maxiter, callback
    # 最大迭代次数，可选参数
    maxiter : int, optional
        # 算法执行的最大迭代次数。
    # 回调函数，可选参数
    callback : callable, optional
        # 可选的用户提供的函数，在第一次迭代之前和每次迭代之后调用。
        # 被调用为 `callback(res)`，其中 `res` 是类似于 `_chandrupatla_minimize` 返回的 `_RichResult` 的对象
        # 如果 `callback` 抛出 `StopIteration` 异常，算法将立即终止并返回结果。
        
    Returns
    -------
    res : _RichResult
        # 返回 `scipy._lib._util._RichResult` 的实例，具有以下属性。
        # 描述假设返回值为标量，但如果 `func` 返回数组，则输出将是相同形状的数组。
        
        success : bool
            # 当算法成功终止时为 `True` (状态为 `0`)。
        status : int
            # 表示算法退出状态的整数。
            # `0` : 算法收敛到指定的容差。
            # `-1` : 算法遇到无效的区间。
            # `-2` : 达到最大迭代次数。
            # `-3` : 遇到非有限值。
            # `-4` : 被 `callback` 终止迭代。
            # `1` : 算法正常进行中 (`callback` 中使用)。
        x : float
            # 如果算法成功终止，则为函数的最小值点。
        fun : float
            # 在 `x` 处评估 `func` 的值。
        nfev : int
            # 评估 `func` 的次数。
        nit : int
            # 执行的算法迭代次数。
        xl, xm, xr : float
            # 最终的三点区间。
        fl, fm, fr : float
            # 区间点的函数值。
    
    Notes
    -----
    # 基于 Chandrupatla 的原始论文 [1]_ 实现。
    
    # 如果 `x1 < x2 < x3` 是区间的点，且 `f1 > f2 <= f3` 是这些点上 `func` 的值，
    # 那么算法收敛于当 `x3 - x1 <= abs(x2)*xrtol + xatol` 或 `(f1 - 2*f2 + f3)/2 <= abs(f2)*frtol + fatol` 时。
    # 注意，第一个条件与 [1]_ 中描述的终止条件不同。
    # `xrtol` 的默认值是适当 dtype 的平方根精度，而 `xatol = fatol = frtol` 是适当 dtype 的最小正常数。
    
    References
    ----------
    .. [1] Chandrupatla, Tirupathi R. (1998).
        "An efficient quadratic fit-sectioning algorithm for minimization
        without derivatives".
        Computer Methods in Applied Mechanics and Engineering, 152 (1-2),
        211-217. https://doi.org/10.1016/S0045-7825(97)00190-4

    See Also
    --------
    golden, brent, bounded

    Examples
    --------
    >>> from scipy.optimize._chandrupatla import _chandrupatla_minimize
    >>> def f(x, args=1):
    ...     return (x - args)**2
    >>> res = _chandrupatla_minimize(f, -5, 0, 5)
    >>> res.x
    1.0
    >>> c = [1, 1.5, 2]
    >>> res = _chandrupatla_minimize(f, -5, 0, 5, args=(c,))
    >>> res.x
    array([1. , 1.5, 2. ])
    """
    # 调用 _chandrupatla_iv 函数执行最小化过程，并获取返回结果元组
    res = _chandrupatla_iv(func, args, xatol, xrtol,
                           fatol, frtol, maxiter, callback)
    func, args, xatol, xrtol, fatol, frtol, maxiter, callback = res

    # Initialization 初始化部分
    xs = (x1, x2, x3)
    # 初始化 EIM 对象并获取返回结果元组
    temp = eim._initialize(func, xs, args)
    func, xs, fs, args, shape, dtype, xp = temp  # line split for PEP8
    x1, x2, x3 = xs
    f1, f2, f3 = fs
    phi = dtype.type(0.5 + 0.5*5**0.5)  # golden ratio 黄金比率
    status = np.full_like(x1, eim._EINPROGRESS, dtype=int)  # in progress 表示进行中
    nit, nfev = 0, 3  # three function evaluations performed above 上面执行了三次函数评估
    fatol = np.finfo(dtype).tiny if fatol is None else fatol  # 如果 fatol 为 None，则使用 dtype 的最小正数值
    frtol = np.finfo(dtype).tiny if frtol is None else frtol  # 如果 frtol 为 None，则使用 dtype 的最小正数值
    xatol = np.finfo(dtype).tiny if xatol is None else xatol  # 如果 xatol 为 None，则使用 dtype 的最小正数值
    xrtol = np.sqrt(np.finfo(dtype).eps) if xrtol is None else xrtol  # 如果 xrtol 为 None，则使用 dtype 的最小正数值的平方根

    # Ensure that x1 < x2 < x3 initially. 确保初始时 x1 < x2 < x3
    xs, fs = np.vstack((x1, x2, x3)), np.vstack((f1, f2, f3))
    i = np.argsort(xs, axis=0)
    x1, x2, x3 = np.take_along_axis(xs, i, axis=0)
    f1, f2, f3 = np.take_along_axis(fs, i, axis=0)
    q0 = x3.copy()  # "At the start, q0 is set at x3..." ([1] after (7)) 起始时，将 q0 设置为 x3 的拷贝

    # Create a _RichResult object to store and manage the workspace 创建 _RichResult 对象来存储和管理工作空间
    work = _RichResult(x1=x1, f1=f1, x2=x2, f2=f2, x3=x3, f3=f3, phi=phi,
                       xatol=xatol, xrtol=xrtol, fatol=fatol, frtol=frtol,
                       nit=nit, nfev=nfev, status=status, q0=q0, args=args)
    # Define pairs of results and corresponding attributes 定义结果和相应属性的对
    res_work_pairs = [('status', 'status'),
                      ('x', 'x2'), ('fun', 'f2'),
                      ('nit', 'nit'), ('nfev', 'nfev'),
                      ('xl', 'x1'), ('xm', 'x2'), ('xr', 'x3'),
                      ('fl', 'f1'), ('fm', 'f2'), ('fr', 'f3')]
    # 定义一个函数用于预测函数的最小值位置
    def pre_func_eval(work):
        # `_check_termination` 首先调用 -> `x3 - x2 > x2 - x1`
        # 但是我们先计算一些将要重复使用的项
        x21 = work.x2 - work.x1  # 计算 x2 - x1
        x32 = work.x3 - work.x2  # 计算 x3 - x2

        # [1] 第三节中，使用上一节开发的关系计算二次最小点 Q1
        A = x21 * (work.f3 - work.f2)
        B = x32 * (work.f1 - work.f2)
        C = A / (A + B)
        # q1 = C * (work.x1 + work.x2) / 2 + (1 - C) * (work.x2 + work.x3) / 2
        q1 = 0.5 * (C*(work.x1 - work.x3) + work.x2 + work.x3)  # 更快的计算方式
        # 这是一个数组，因此乘以 0.5 不会改变其数据类型

        # "如果 Q1 和 Q0 足够接近... Q1 被接受，如果它距离内部点 x2 足够远"
        i = abs(q1 - work.q0) < 0.5 * abs(x21)  # [1] (7)
        xi = q1[i]
        # 后来，在第 (9) 步之后，"如果点 Q1 在 +/- xtol 邻域内，新点被选择在距离 x2 较大间隔 tol 处。"
        # 参见 "Accept Ql adjust if close to X2" 后的 QBASIC 代码。
        j = abs(q1[i] - work.x2[i]) <= work.xtol[i]
        xi[j] = work.x2[i][j] + np.sign(x32[i][j]) * work.xtol[i][j]

        # "如果条件 (7) 不满足，进行大间隔的黄金分割以引入新点。"
        # (为了简单起见，我们计算所有点，但我们只改变满足条件的元素。)
        x = work.x2 + (2 - work.phi) * x32
        x[i] = xi

        # "我们将 Q0 定义为前一次迭代中 Q1 的值。"
        work.q0 = q1
        return x
    def post_func_eval(x, f, work):
        # 根据新点更新三点区间的标准逻辑。在 QBASIC 代码中，参见 "IF SGN(X-X2) = SGN(X3-X2) THEN..."。
        # 这里涉及大量数据复制，可能受益于代码优化或在 Pythran 中实现。
        
        # 根据 x 和 work.x2 的符号比较判断是否在同一方向上
        i = np.sign(x - work.x2) == np.sign(work.x3 - work.x2)
        
        # 按照 i 的条件分别提取 x, x1, x2, x3 和对应的 f, f1, f2, f3
        xi, x1i, x2i, x3i = x[i], work.x1[i], work.x2[i], work.x3[i],
        fi, f1i, f2i, f3i = f[i], work.f1[i], work.f2[i], work.f3[i]
        
        # 根据条件 j 更新 x3 和 f3 或者 x1, f1, x2, f2
        j = fi > f2i
        x3i[j], f3i[j] = xi[j], fi[j]
        j = ~j
        x1i[j], f1i[j], x2i[j], f2i[j] = x2i[j], f2i[j], xi[j], fi[j]

        # 对于非 i 的情况，同样进行相似的更新
        ni = ~i
        xni, x1ni, x2ni, x3ni = x[ni], work.x1[ni], work.x2[ni], work.x3[ni],
        fni, f1ni, f2ni, f3ni = f[ni], work.f1[ni], work.f2[ni], work.f3[ni]
        j = fni > f2ni
        x1ni[j], f1ni[j] = xni[j], fni[j]
        j = ~j
        x3ni[j], f3ni[j], x2ni[j], f2ni[j] = x2ni[j], f2ni[j], xni[j], fni[j]

        # 更新 work 对象中的 x1, x2, x3 和对应的 f1, f2, f3
        work.x1[i], work.x2[i], work.x3[i] = x1i, x2i, x3i
        work.f1[i], work.f2[i], work.f3[i] = f1i, f2i, f3i
        work.x1[ni], work.x2[ni], work.x3[ni] = x1ni, x2ni, x3ni,
        work.f1[ni], work.f2[ni], work.f3[ni] = f1ni, f2ni, f3ni
    # 检查终止条件并记录状态
    def check_termination(work):
        # 初始化一个布尔数组，用于标记终止条件是否满足
        stop = np.zeros_like(work.x1, dtype=bool)  # termination condition met

        # 若 bracket 是无效的，则停止并且不返回最小化器/最小值
        i = ((work.f2 > work.f1) | (work.f2 > work.f3))
        # 将无效的 bracket 的 x2 和 f2 设为 NaN，并标记停止和错误状态
        work.x2[i], work.f2[i] = np.nan, np.nan
        stop[i], work.status[i] = True, eim._ESIGNERR

        # 非有限值的情况下停止并且不返回最小化器/最小值
        finite = np.isfinite(work.x1 + work.x2 + work.x3 + work.f1 + work.f2 + work.f3)
        i = ~(finite | stop)
        # 将非有限值的 x2 和 f2 设为 NaN，并标记停止和错误状态
        work.x2[i], work.f2[i] = np.nan, np.nan
        stop[i], work.status[i] = True, eim._EVALUEERR

        # [1] 第三部分："如果需要，将点1和点3互换，使得(x2, x3)成为更大的区间。"
        # 注意：我曾使用 np.choose；这种方式更快。这里也可以保存例如 `work.x3 - work.x2` 以便复用，
        # 但我尝试过并没有注意到速度提升，所以保持简单。
        i = abs(work.x3 - work.x2) < abs(work.x2 - work.x1)
        # 交换符合条件的 x1, x3 和 f1, f3
        temp = work.x1[i]
        work.x1[i] = work.x3[i]
        work.x3[i] = temp
        temp = work.f1[i]
        work.f1[i] = work.f3[i]
        work.f3[i] = temp

        # [1] 第三部分 (212页底部)："我们设置了一个公差值 xtol..."
        # 计算基于区间的收敛性
        work.xtol = abs(work.x2) * work.xrtol + work.xatol  # [1] (8)
        # 根据区间的收敛性，达到收敛条件时...
        # 注意：允许相等性以防 `xtol=0`
        i = abs(work.x3 - work.x2) <= 2 * work.xtol  # [1] (9)

        # "我们使用...定义 ftol"
        ftol = abs(work.f2) * work.frtol + work.fatol  # [1] (10)
        # 根据函数值的收敛性，达到收敛条件时...
        # 注意 1：原地修改以包含函数值的容差。
        # 注意 2：文本中没有 2 的因子；参见 QBASIC DO 循环的开始
        i |= (work.f1 - 2 * work.f2 + work.f3) <= 2 * ftol  # [1] (11)
        i &= ~stop
        stop[i], work.status[i] = True, eim._ECONVERGED

        return stop

    def post_termination_check(work):
        pass

    def customize_result(res, shape):
        xl, xr, fl, fr = res['xl'], res['xr'], res['fl'], res['fr']
        i = res['xl'] < res['xr']
        # 根据条件 i 对结果进行定制化处理
        res['xl'] = np.choose(i, (xr, xl))
        res['xr'] = np.choose(i, (xl, xr))
        res['fl'] = np.choose(i, (fr, fl))
        res['fr'] = np.choose(i, (fl, fr))
        return shape

    return eim._loop(work, callback, shape, maxiter, func, args, dtype,
                     pre_func_eval, post_func_eval, check_termination,
                     post_termination_check, customize_result, res_work_pairs,
                     xp=xp)
```