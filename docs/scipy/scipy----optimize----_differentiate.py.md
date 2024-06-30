# `D:\src\scipysrc\scipy\scipy\optimize\_differentiate.py`

```
# mypython
    # Disable mypy error code "attr-defined" for this module
    # 导入 NumPy 库，简称为 np
    import numpy as np
    # 导入 scipy 库中的特定模块 _elementwise_iterative_method
    import scipy._lib._elementwise_iterative_method as eim
    # 从 scipy 库中导入 _RichResult 类
    from scipy._lib._util import _RichResult
    # 从 scipy 库中导入 array_namespace 函数
    from scipy._lib._array_api import array_namespace

    # 定义全局变量 _EERRORINCREASE，用于 _differentiate 函数
    _EERRORINCREASE = -1  # used in _differentiate

    def _differentiate_iv(func, x, args, atol, rtol, maxiter, order, initial_step,
                          step_factor, step_direction, preserve_shape, callback):
        # 输入验证函数 `_differentiate`

        # 检查 func 是否为可调用对象
        if not callable(func):
            raise ValueError('`func` must be callable.')

        # 如果 args 不可迭代，则转换为元组
        if not np.iterable(args):
            args = (args,)

        # 确认容差参数为浮点数，不是数组；可以使用 NumPy 进行转换
        message = 'Tolerances and step parameters must be non-negative scalars.'
        tols = np.asarray([atol if atol is not None else 1,
                           rtol if rtol is not None else 1,
                           initial_step, step_factor])
        if (not np.issubdtype(tols.dtype, np.number) or np.any(tols < 0)
                or np.any(np.isnan(tols)) or tols.shape != (4,)):
            raise ValueError(message)
        initial_step = float(tols[2])
        step_factor = float(tols[3])

        # 将 maxiter 转换为整数，并验证其为正整数
        maxiter_int = int(maxiter)
        if maxiter != maxiter_int or maxiter <= 0:
            raise ValueError('`maxiter` must be a positive integer.')

        # 将 order 转换为整数，并验证其为正整数
        order_int = int(order)
        if order_int != order or order <= 0:
            raise ValueError('`order` must be a positive integer.')

        # 使用 array_namespace 处理 x，确保 x 和 step_direction 可广播
        xp_temp = array_namespace(x)
        x, step_direction = xp_temp.broadcast_arrays(x, xp_temp.asarray(step_direction))

        # 检查 preserve_shape 参数是否为 True 或 False
        message = '`preserve_shape` must be True or False.'
        if preserve_shape not in {True, False}:
            raise ValueError(message)

        # 如果 callback 存在，则验证其为可调用对象
        if callback is not None and not callable(callback):
            raise ValueError('`callback` must be callable.')

        # 返回参数元组
        return (func, x, args, atol, rtol, maxiter_int, order_int, initial_step,
                step_factor, step_direction, preserve_shape, callback)

    def _differentiate(func, x, *, args=(), atol=None, rtol=None, maxiter=10,
                       order=8, initial_step=0.5, step_factor=2.0,
                       step_direction=0, preserve_shape=False, callback=None):
        """Evaluate the derivative of an elementwise scalar function numerically.

        Parameters
        ----------
        func : callable
            The function whose derivative is desired. The signature must be::

                func(x: ndarray, *fargs) -> ndarray

             where each element of ``x`` is a finite real number and ``fargs`` is a tuple,
             which may contain an arbitrary number of arrays that are broadcastable
             with `x`. ``func`` must be an elementwise function: each element
             ``func(x)[i]`` must equal ``func(x[i])`` for all indices ``i``.
        x : array_like
            Abscissae at which to evaluate the derivative.
        """
    args : tuple, optional
        # `args`是传递给 `func` 的额外位置参数的元组。必须是可以与 `x` 广播的数组。
        # 如果要求被微分的可调用函数需要与 `x` 不可广播的参数，则使用 `func` 进行包装。参见示例。
    atol, rtol : float, optional
        # 停止条件的绝对和相对容差：当 ``res.error < atol + rtol * abs(res.df)`` 时，迭代将停止。
        # 默认的 `atol` 是适当数据类型的最小正常数，`rtol` 是适当数据类型精度的平方根。
    order : int, default: 8
        # 要使用的有限差分公式的正整数阶数。奇数将被四舍五入为下一个偶数。
    initial_step : float, default: 0.5
        # 有限差分导数近似的（绝对）初始步长大小。
    step_factor : float, default: 2.0
        # 每次迭代中步长大小被 *减小* 的因子；即迭代 1 中的步长是 ``initial_step/step_factor``。
        # 如果 ``step_factor < 1``，则后续步长将大于初始步长；这在不希望步长小于某个阈值时可能很有用（例如，由于减法取消误差）。
    maxiter : int, default: 10
        # 算法执行的最大迭代次数。见备注。
    step_direction : array_like
        # 表示有限差分步骤方向的数组（用于 `x` 接近函数域边界时）。必须与 `x` 和所有 `args` 广播。
        # 其中 0（默认）使用中心差分；负数（例如 -1）使用非正步长；正数（例如 1）所有步骤都是非负的。
    preserve_shape : bool, default: False
        # 在以下情况中，“func 的参数” 指的是数组 ``x`` 和 ``fargs`` 中的任何数组。设 ``shape`` 是 `x` 和所有 `args` 的广播形状
        # （概念上不同于传递给 `f` 的 `fargs`）。

        # - 当 ``preserve_shape=False``（默认）时，`f` 必须接受任何广播形状的参数。

        # - 当 ``preserve_shape=True`` 时，`f` 必须接受形状为 ``shape`` 或 ``shape + (n,)`` 的参数，其中 ``(n,)`` 是函数评估时的横坐标数。

        # 在任何情况下，对于 `x` 中的每个标量元素 ``xi``，`f` 返回的数组必须包含相同索引处的标量 ``f(xi)``。
        # 因此，输出的形状始终是输入 ``x`` 的形状。

        # 参见示例。
    # callback : callable, optional
    #     可选的用户提供的回调函数，在第一次迭代之前和每次迭代之后调用。
    #     被调用为 `callback(res)`，其中 `res` 是类似于 `_differentiate` 返回的 `_RichResult` 对象
    #     如果 `callback` 抛出 `StopIteration` 异常，算法将立即终止，`_differentiate` 将返回结果。

    # Returns
    # -------
    # res : _RichResult
    #     返回一个 `scipy._lib._util._RichResult` 的实例，具有以下属性（描述假设返回值是标量；如果 `func` 返回数组，则输出将是相同形状的数组）。

    #     success : bool
    #         当算法成功终止时为 `True`（状态为 `0`）。
    #     status : int
    #         表示算法退出状态的整数。
    #         `0` : 算法收敛到指定的容差。
    #         `-1` : 误差估计增加，因此迭代被终止。
    #         `-2` : 达到最大迭代次数。
    #         `-3` : 遇到非有限值。
    #         `-4` : 被 `callback` 终止迭代。
    #         `1` : 算法正常进行（仅在 `callback` 中）。
    #     df : float
    #         如果算法成功终止，则 `func` 在 `x` 处的导数。
    #     error : float
    #         误差的估计值：当前导数估计与前一个迭代中的估计之间的差的大小。
    #     nit : int
    #         执行的迭代次数。
    #     nfev : int
    #         对 `func` 评估的点数。
    #     x : float
    #         评估 `func` 的导数的值（在与 `args` 和 `step_direction` 广播后）。

    # Notes
    # -----
    # 实现灵感来源于 jacobi [1]_, numdifftools [2]_ 和 DERIVEST [3]_，但实现更直接地遵循了泰勒级数的理论（可以说有些天真）。
    # 在第一次迭代中，使用阶为 `order` 的有限差分公式估计导数，最大步长为 `initial_step`。
    # 每个后续迭代，最大步长减少 `step_factor`，再次估计导数，直到达到终止条件。
    # 误差估计是当前导数近似值与前一个迭代的估计之间的差的大小。

    # 有限差分公式的插值模板设计为“嵌套”：在第一次迭代中，在 `func` 中评估 `order + 1` 个点之后，仅在两个新点处评估 `func`。
    # Show the convergence of the approximation as the step size is reduced.
    # Each iteration, the step size is reduced by `step_factor`, so for
    # sufficiently small initial step, each iteration reduces the error by a
    # factor of ``1/step_factor**order`` until finite precision arithmetic
    # inhibits further improvement.
    iter = list(range(1, 12))  # maximum iterations
    hfac = 2  # step size reduction per iteration
    hdir = [-1, 0, 1]  # compare left-, central-, and right- steps
    order = 4  # order of differentiation formula
    x = 1
    ref = df(x)
    errors = []  # true error
    for i in iter:
        # Perform numerical differentiation using `_differentiate` function
        # with specified parameters to estimate the derivative.
        res = _differentiate(f, x, maxiter=i, step_factor=hfac,
                             step_direction=hdir, order=order,
                             atol=0, rtol=0)  # prevent early termination
        # Calculate absolute error between the estimated derivative and the true derivative
        errors.append(abs(res.df - ref))
    errors = np.array(errors)
    >>> plt.semilogy(iter, errors[:, 0], label='left differences')
    >>> plt.semilogy(iter, errors[:, 1], label='central differences')
    >>> plt.semilogy(iter, errors[:, 2], label='right differences')
    >>> plt.xlabel('iteration')
    >>> plt.ylabel('error')
    >>> plt.legend()
    >>> plt.show()
    
    
    # 绘制三条误差随迭代次数变化的半对数图（Y轴取对数）
    # 分别绘制左偏差、中心差分和右偏差的误差随迭代次数的变化曲线
    >>> (errors[1, 1] / errors[0, 1], 1 / hfac**order)
    (0.06215223140159822, 0.0625)
    
    The implementation is vectorized over `x`, `step_direction`, and `args`.
    The function is evaluated once before the first iteration to perform input
    validation and standardization, and once per iteration thereafter.
    
    >>> def f(x, p):
    ...     print('here')
    ...     f.nit += 1
    ...     return x**p
    >>> f.nit = 0
    >>> def df(x, p):
    ...     return p*x**(p-1)
    >>> x = np.arange(1, 5)
    >>> p = np.arange(1, 6).reshape((-1, 1))
    >>> hdir = np.arange(-1, 2).reshape((-1, 1, 1))
    >>> res = _differentiate(f, x, args=(p,), step_direction=hdir, maxiter=1)
    >>> np.allclose(res.df, df(x, p))
    True
    >>> res.df.shape
    (3, 5, 4)
    >>> f.nit
    2
    
    By default, `preserve_shape` is False, and therefore the callable
    `f` may be called with arrays of any broadcastable shapes.
    For example:
    
    >>> shapes = []
    >>> def f(x, c):
    ...    shape = np.broadcast_shapes(x.shape, c.shape)
    ...    shapes.append(shape)
    ...    return np.sin(c*x)
    >>>
    >>> c = [1, 5, 10, 20]
    >>> res = _differentiate(f, 0, args=(c,))
    >>> shapes
    [(4,), (4, 8), (4, 2), (3, 2), (2, 2), (1, 2)]
    
    To understand where these shapes are coming from - and to better
    understand how `_differentiate` computes accurate results - note that
    higher values of ``c`` correspond with higher frequency sinusoids.
    The higher frequency sinusoids make the function's derivative change
    faster, so more function evaluations are required to achieve the target
    accuracy:
    
    >>> res.nfev
    array([11, 13, 15, 17])
    
    The initial ``shape``, ``(4,)``, corresponds with evaluating the
    function at a single abscissa and all four frequencies; this is used
    for input validation and to determine the size and dtype of the arrays
    that store results. The next shape corresponds with evaluating the
    function at an initial grid of abscissae and all four frequencies.
    Successive calls to the function evaluate the function at two more
    abscissae, increasing the effective order of the approximation by two.
    However, in later function evaluations, the function is evaluated at
    fewer frequencies because the corresponding derivative has already
    converged to the required tolerance. This saves function evaluations to
    improve performance, but it requires the function to accept arguments of
    any shape.
    
    "Vector-valued" functions are unlikely to satisfy this requirement.
    For example, consider
    
    >>> def f(x):
    ...    return [x, np.sin(3*x), x+np.sin(10*x), np.sin(20*x)*(x-1)**2]
    
    
    # 计算给定函数 `f` 在点 `x` 处的导数，返回结果的属性包括 `df` 和 `nfev`
    # 这里的 `_differentiate` 函数将 `f` 在 `x` 处用 `p` 参数进行微分，`step_direction` 控制微分方向，`maxiter` 限制最大迭代次数为1
    # `np.allclose(res.df, df(x, p))` 验证 `_differentiate` 返回的导数结果 `res.df` 是否与预期的导数 `df(x, p)` 接近
    # `res.df.shape` 返回导数结果的形状信息
    # `f.nit` 输出函数 `f` 被调用的次数，此处应为2次
    """
    # TODO (followup):
    #  - investigate behavior at saddle points
    #  - array initial_step / step_factor?
    #  - multivariate functions?

    Perform numerical differentiation using an iterative method with initial values and settings.

    res = _differentiate_iv(func, x, args, atol, rtol, maxiter, order, initial_step,
                            step_factor, step_direction, preserve_shape, callback)
    Obtain results from an iterative numerical differentiation function `_differentiate_iv`.

    (func, x, args, atol, rtol, maxiter, order,
     h0, fac, hdir, preserve_shape, callback) = res
    Unpack the results tuple into individual variables for further use.

    # Initialization
    # Initialize various parameters and prepare for the differentiation process.
    # `_initialize` function sets up initial conditions and validates inputs.
    temp = eim._initialize(func, (x,), args, preserve_shape=preserve_shape)
    func, xs, fs, args, shape, dtype, xp = temp
    Extract function, initial points, function values, arguments, shape, data type,
    and execution provider from the temporary result.

    finfo = xp.finfo(dtype)
    Determine the floating-point characteristics for the given data type.

    atol = finfo.smallest_normal if atol is None else atol
    Set absolute tolerance to the smallest normal value if not provided explicitly.

    rtol = finfo.eps**0.5 if rtol is None else rtol
    Set relative tolerance based on machine epsilon if not provided explicitly.

    x, f = xs[0], fs[0]
    Extract initial points and function values from their respective containers.

    df = xp.full_like(f, xp.nan)
    Initialize an array `df` filled with NaNs, representing the differentiated values.

    # Ideally we'd broadcast the shape of `hdir` in `_elementwise_algo_init`, but
    # it's simpler to do it here than to generalize `_elementwise_algo_init` further.
    # `hdir` and `x` are already broadcasted in `_differentiate_iv`, so we know
    # that `hdir` can be broadcasted to the final shape.
    hdir = xp.astype(xp.sign(hdir), dtype)
    Convert `hdir` to the sign of its values, ensuring consistent type.

    hdir = xp.broadcast_to(hdir, shape)
    Broadcast `hdir` to match the shape of other relevant arrays.

    hdir = xp.reshape(hdir, (-1,))
    Reshape `hdir` into a 1-dimensional array for further processing.

    status = xp.full_like(x, eim._EINPROGRESS, dtype=xp.int32)  # in progress
    Initialize `status` array filled with `_EINPROGRESS` values, indicating ongoing computation.

    nit, nfev = 0, 1
    Initialize iteration count `nit` and function evaluation count `nfev`.

    # Boolean indices of left, central, right, and (all) one-sided steps
    il = hdir < 0
    ic = hdir == 0
    ir = hdir > 0
    io = il | ir
    Determine boolean indices for left, central, right, and all one-sided steps.

    # Most of these attributes are reasonably obvious, but:
    # - `fs` holds all the function values of all active `x`. The zeroth
    #   axis corresponds with active points `x`, the first axis corresponds
    #   with the different steps (in the order described in
    #   `_differentiate_weights`).
    """
    # `terms`（可能需要更好的命名）是 `order` 的一半，而且始终是偶数。
    work = _RichResult(x=x, df=df, fs=f[:, xp.newaxis], error=xp.nan, h=h0,
                       df_last=xp.nan, error_last=xp.nan, h0=h0, fac=fac,
                       atol=atol, rtol=rtol, nit=nit, nfev=nfev,
                       status=status, dtype=dtype, terms=(order+1)//2,
                       hdir=hdir, il=il, ic=ic, ir=ir, io=io)
    # 这里是`work`对象中的术语与最终结果之间的对应关系。在这种情况下，映射是显而易见的。
    # 注意，`success`会自动添加到结果中。
    res_work_pairs = [('status', 'status'), ('df', 'df'), ('error', 'error'),
                      ('nit', 'nit'), ('nfev', 'nfev'), ('x', 'x')]

    def pre_func_eval(work):
        """确定需要评估函数的横坐标。

        请参阅 `_differentiate_weights` 获取横坐标的模式描述（即样板）。

        在第一次迭代中，`work.fs` 中只有一个存储的函数值 `f(x)`，因此我们需要在 `order` 个新点上评估。
        在后续迭代中，我们在两个新点上评估。注意，`work.x` 在与所有 `args` 广播后总是扁平化为一个1D数组，因此我们在函数的一个调用中增加一个新轴并评估所有点。

        改进建议：
        - 考虑测量实际采取的步长，因为使用浮点运算 `(x + h) - x` 不完全等于 `h`。
        - 如果 `x` 太大无法解析步长，则自动调整步长。
        - 如果没有中心差分步骤或单侧步骤，则可能可以节省一些工作。
        """
        n = work.terms  # `order` 的一半
        h = work.h  # 步长
        c = work.fac  # 步长缩减因子
        d = c**0.5  # 步长缩减因子的平方根（单侧样板）
        # 注意 - 在分配 `x_eval` 之前无需关注 dtype

        if work.nit == 0:
            hc = h / c**xp.arange(n, dtype=work.dtype)
            hc = xp.concat((-xp.flip(hc), hc))
        else:
            hc = xp.asarray([-h, h]) / c**(n-1)

        if work.nit == 0:
            hr = h / d**xp.arange(2*n, dtype=work.dtype)
        else:
            hr = xp.asarray([h, h/d]) / c**(n-1)

        n_new = 2*n if work.nit == 0 else 2  # 新横坐标的数量
        x_eval = xp.zeros((work.hdir.shape[0], n_new), dtype=work.dtype)
        il, ic, ir = work.il, work.ic, work.ir
        x_eval[ir] = work.x[ir][:, xp.newaxis] + hr
        x_eval[ic] = work.x[ic][:, xp.newaxis] + hc
        x_eval[il] = work.x[il][:, xp.newaxis] - hr
        return x_eval
    # 定义一个函数，用于检查优化算法的终止条件
    def check_termination(work):
        # 创建一个与work.df相同形状的布尔类型的数组stop，用于标记终止条件
        stop = xp.astype(xp.zeros_like(work.df), xp.bool)
    
        # 检查是否满足收敛条件，根据工作对象的误差、公差和相对误差判断
        i = work.error < work.atol + work.rtol * abs(work.df)
        work.status[i] = eim._ECONVERGED  # 标记为收敛
        stop[i] = True  # 停止优化
    
        # 如果迭代次数大于0，检查是否出现非有限值或者增加误差的情况
        i = ~((xp.isfinite(work.x) & xp.isfinite(work.df)) | stop)
        work.df[i], work.status[i] = xp.nan, eim._EVALUEERR  # 将无效的导数设置为NaN，并标记为值错误
        stop[i] = True  # 停止优化
    
        # 根据启发式策略，检查是否出现误差增加的情况，避免步长过小引起的浮点数运算取消问题
        i = (work.error > work.error_last * 10) & ~stop
        work.status[i] = _EERRORINCREASE  # 标记为误差增加
        stop[i] = True  # 停止优化
    
        # 返回布尔数组stop，指示是否应该终止优化
        return stop
    
    # 定义一个空函数post_termination_check，用于终止后检查（未实现具体功能）
    def post_termination_check(work):
        return
    
    # 定义一个函数customize_result，将结果定制为指定的形状shape，但是该实现直接返回shape，未做实际处理
    def customize_result(res, shape):
        return shape
    
    # 返回调用eim._loop函数的结果，传递了多个参数用于执行迭代优化
    return eim._loop(work, callback, shape, maxiter, func, args, dtype,
                     pre_func_eval, post_func_eval, check_termination,
                     post_termination_check, customize_result, res_work_pairs,
                     xp, preserve_shape)
def _differentiate_weights(work, n, xp):
    # This function computes weights for finite difference formulas to estimate derivatives.
    # It discusses the theory behind finite differences, specifically centered differences,
    # which approximate derivatives with second-order accuracy.
    
    # By default, eighth-order formulas are used, with a stencil of points symmetrically
    # spaced around `x`. This stencil allows for iterative refinement by reducing step
    # size `h` with a factor `c`, reusing function evaluations for efficiency.
    
    # The function leaves open the possibility for future enhancements, such as
    # Richardson extrapolation or one-sided differences, by storing all function values
    # in `work.fs`. This approach aims to improve numerical accuracy and stability.
    
    # Parameters:
    # - work: Object containing function evaluations and intermediate data.
    # - n: Number of points in the stencil.
    # - xp: List of points in the stencil around which derivatives are estimated.
    
    # The resulting weights `wi` allow the computation of the derivative approximation
    # using the formula: f'(x) ~ (w1*f(x) + w2*f(x+h) + w3*f(x-h))/h, with an error of O(h**2).
    
    # Returns:
    # None
    
    # Note: The detailed explanation of the theory and approach to computing weights
    # is provided here for clarity and future development reference.
    pass
    # 如果用户以双精度指定 `fac`，但是 `x` 和 `args` 是单精度，`fac` 将被转换为单精度。
    # 这里我们始终使用双精度进行中间计算，以避免权重中的额外误差。
    fac = float(work.fac)

    # 注意，如果用户回到浮点精度并使用单精度的 `x` 和 `args`，那么 `fac` 不一定等于
    # 之前缓存的 `_differentiate_weights.fac`（较低精度）。这将需要重新计算权重。
    # 这个问题可以解决，但现在时间已晚，并且影响较小。
    if fac != _differentiate_weights.fac:
        # 如果 `fac` 更新了，清空之前缓存的中心和右侧差分权重
        _differentiate_weights.central = []
        _differentiate_weights.right = []
        _differentiate_weights.fac = fac

    # 如果中心差分权重的长度不等于 `2*n + 1`，则重新计算权重
    if len(_differentiate_weights.central) != 2*n + 1:
        # 中心差分权重。考虑重构此部分代码以使其更加紧凑。
        # 注意：在此处使用 NumPy 是可以的；我们最终会将其转换为 xp 类型。
        i = np.arange(-n, n + 1)
        p = np.abs(i) - 1.  # 中心点的幂为 `p` - 1，但符号 `s` 为 0
        s = np.sign(i)

        h = s / fac ** p
        A = np.vander(h, increasing=True).T
        b = np.zeros(2*n + 1)
        b[1] = 1
        weights = np.linalg.solve(A, b)

        # 强制施加恒等式以提高准确性
        weights[n] = 0
        for i in range(n):
            weights[-i-1] = -weights[i]

        # 缓存这些权重。除非步长因子发生变化，否则我们只需计算一次。
        _differentiate_weights.central = weights

        # 单侧差分权重。左侧单侧权重（带负步长）是右侧单侧权重的相反数，因此无需单独计算。
        i = np.arange(2*n + 1)
        p = i - 1.
        s = np.sign(i)

        h = s / np.sqrt(fac) ** p
        A = np.vander(h, increasing=True).T
        b = np.zeros(2 * n + 1)
        b[1] = 1
        weights = np.linalg.solve(A, b)

        # 缓存右侧单侧权重
        _differentiate_weights.right = weights

    # 返回计算好的权重，转换为指定的 `work.dtype` 类型
    return (xp.asarray(_differentiate_weights.central, dtype=work.dtype),
            xp.asarray(_differentiate_weights.right, dtype=work.dtype))
# 初始化空列表，用于存储中心差分权重
_differentiate_weights.central = []
# 初始化空列表，用于存储右侧差分权重
_differentiate_weights.right = []
# 初始化为None，用于存储因子
_differentiate_weights.fac = None

# 定义函数 _jacobian，用于数值计算函数的雅可比矩阵
def _jacobian(func, x, *, atol=None, rtol=None, maxiter=10,
              order=8, initial_step=0.5, step_factor=2.0):
    r"""Evaluate the Jacobian of a function numerically.

    Parameters
    ----------
    func : callable
        欲求雅可比矩阵的函数。其签名必须为::

            func(x: ndarray) -> ndarray

         其中每个元素 `x` 均为有限实数。如果待求导的函数接受额外参数，可将其包装
         (例如使用 `functools.partial` 或 `lambda` )，并将包装后的可调用对象传入 `_jacobian`。
         有关向量化和输入输出的维度，请参阅注释。
    x : array_like
        评估雅可比矩阵的点。必须至少有一个维度。有关维度和向量化的详细信息，请参阅注释。
    atol, rtol : float, optional
        停止条件的绝对和相对容差：当每个雅可比矩阵元素满足条件时，迭代将停止，
        即 ``res.error < atol + rtol * abs(res.df)``。默认的 `atol` 是适当数据类型的最小正数，
        默认的 `rtol` 是适当数据类型精度的平方根。
    order : int, default: 8
        所使用的有限差分公式的阶数（正整数）。奇数将被向上取整到下一个偶数。
    initial_step : float, default: 0.5
        有限差分导数近似的初始步长（绝对值）。
    step_factor : float, default: 2.0
        每次迭代步长被 *减少* 的因子；即第一次迭代的步长为 ``initial_step/step_factor``。
        如果 ``step_factor < 1``，则后续步长将大于初始步长；这可能在希望避免步长小于某个阈值时
        （例如由于减法取消误差）会有用。
    maxiter : int, default: 10
        算法执行的最大迭代次数。

    Returns
    -------
    res : _RichResult
        # 定义变量 res，表示一个 `_RichResult` 类的实例，具有以下属性：

        success : bool array
            # 布尔型数组，指示算法是否成功终止（状态为 `0`）。

        status : int array
            # 整数数组，表示算法的退出状态。
            # `0` : 算法收敛到指定的容差。
            # `-1` : 误差估计增加，因此终止迭代。
            # `-2` : 达到最大迭代次数。
            # `-3` : 遇到非有限值。
            # `-4` : 被 `callback` 终止迭代。
            # `1` : 算法正常进行中（仅在 `callback` 中）。

        df : float array
            # 在算法成功终止时，`func` 在 `x` 处的雅可比矩阵。

        error : float array
            # 误差估计：当前估计的导数与上一次迭代中的估计之间的差的大小。

        nit : int array
            # 执行的迭代次数。

        nfev : int array
            # 评估 `func` 的点数。

        x : float array
            # 评估 `func` 的导数的值。

    See Also
    --------
    _differentiate

    Notes
    -----
    # 假设我们希望评估函数 :math:`f: \mathbf{R^m} \rightarrow \mathbf{R^n}` 的雅可比矩阵，
    # 并分配给变量 `m` 和 `n` 分别为正整数值 :math:`m` 和 :math:`n`。
    # 如果我们希望在单个点评估雅可比矩阵，那么：

    - argument `x` must be an array of shape ``(m,)``
      # 参数 `x` 必须是形状为 ``(m,)`` 的数组。

    - argument `func` must be vectorized to accept an array of shape ``(m, p)``.
      The first axis represents the :math:`m` inputs of :math:`f`; the second
      is for evaluating the function at multiple points in a single call.
      # 参数 `func` 必须向量化，接受形状为 ``(m, p)`` 的数组。
      # 第一个轴表示 :math:`f` 的 :math:`m` 个输入；第二个轴用于在单个调用中评估多个点的函数。

    - argument `func` must return an array of shape ``(n, p)``. The first
      axis represents the :math:`n` outputs of :math:`f`; the second
      is for the result of evaluating the function at multiple points.
      # 参数 `func` 必须返回形状为 ``(n, p)`` 的数组。
      # 第一个轴表示 :math:`f` 的 :math:`n` 个输出；第二个轴表示在多个点评估函数的结果。

    - attribute ``df`` of the result object will be an array of shape ``(n, m)``,
      the Jacobian.
      # 结果对象的属性 `df` 将是形状为 ``(n, m)`` 的数组，即雅可比矩阵。

    # 此函数也是向量化的，即可以在单次调用中评估 `k` 个点的雅可比矩阵。
    # 在这种情况下，`x` 将是形状为 ``(m, k)`` 的数组，`func` 将接受形状为 ``(m, k, p)`` 的数组，
    # 并返回形状为 ``(n, k, p)`` 的数组，结果的 `df` 属性将具有形状 ``(n, m, k)``。

    References
    ----------
    .. [1] Jacobian matrix and determinant, *Wikipedia*,
           https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant

    Examples
    --------
    The Rosenbrock function maps from :math:`\mathbf{R}^m \rightarrow \mathbf{R}`;
    # Rosenbrock 函数映射从 :math:`\mathbf{R}^m` 到 :math:`\mathbf{R}`；
    x = np.asarray(x)  # 将输入参数 x 转换为 NumPy 数组
    int_dtype = np.issubdtype(x.dtype, np.integer)  # 检查 x 的数据类型是否为整数类型
    x0 = np.asarray(x, dtype=float) if int_dtype else x  # 如果 x 是整数类型，则将其转换为浮点类型数组，否则保持不变

    if x0.ndim < 1:  # 如果 x0 的维度小于 1，即 x0 不是至少一维的数组
        message = "Argument `x` must be at least 1-D."  # 抛出值错误，提示 x 必须至少是一维数组
        raise ValueError(message)

    m = x0.shape[0]  # 获取 x0 的第一维度大小，即向量的长度
    i = np.arange(m)  # 创建一个从 0 到 m-1 的整数数组 i，用于后续操作中的索引使用

    def wrapped(x):
        p = () if x.ndim == x0.ndim else (x.shape[-1],)  # 如果 x 的维度与 x0 的维度相同，p 为空元组；否则为 x 的最后一维的大小
        new_dims = (1,) if x.ndim == x0.ndim else (1, -1)  # 根据 x 的维度是否与 x0 相同，确定新维度元组
        new_shape = (m, m) + x0.shape[1:] + p  # 创建新的形状元组，包括 m*m，以及 x0 的剩余维度和 p 的维度
        xph = np.expand_dims(x0, new_dims)  # 将 x0 沿新维度进行扩展
        xph = np.broadcast_to(xph, new_shape).copy()  # 广播 xph 到新形状，并复制以确保独立副本
        xph[i, i] = x  # 将 x 的值复制到 xph 的对角线位置
        return func(xph)  # 返回传入函数 func 的处理结果

    res = _differentiate(wrapped, x, atol=atol, rtol=rtol,
                         maxiter=maxiter, order=order, initial_step=initial_step,
                         step_factor=step_factor, preserve_shape=True)
    del res.x  # 删除结果对象 res 中的属性 x，因为在这里对其进行广播是无意义的
    return res  # 返回最终的结果对象 res
```