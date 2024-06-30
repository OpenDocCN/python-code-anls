# `D:\src\scipysrc\scipy\scipy\_lib\_elementwise_iterative_method.py`

```
# `_elementwise_iterative_method.py` includes tools for writing functions that
# - are vectorized to work elementwise on arrays,
# - implement non-trivial, iterative algorithms with a callback interface, and
# - return rich objects with iteration count, termination status, etc.
#
# Examples include:
# `scipy.optimize._chandrupatla._chandrupatla` for scalar rootfinding,
# `scipy.optimize._chandrupatla._chandrupatla_minimize` for scalar minimization,
# `scipy.optimize._differentiate._differentiate` for numerical differentiation,
# `scipy.optimize._bracket._bracket_root` for finding rootfinding brackets,
# `scipy.optimize._bracket._bracket_minimize` for finding minimization brackets,
# `scipy.integrate._tanhsinh._tanhsinh` for numerical quadrature.

import math
import numpy as np
from ._util import _RichResult, _call_callback_maybe_halt
from ._array_api import array_namespace, size as xp_size

# Error codes for various conditions during iterative methods
_ESIGNERR = -1   # Sign error during iteration
_ECONVERR = -2   # Convergence error during iteration
_EVALUEERR = -3  # Error in value during iteration
_ECALLBACK = -4  # Error in callback function
_EINPUTERR = -5  # Error in input parameters
_ECONVERGED = 0  # Iterative method converged successfully
_EINPROGRESS = 1  # Iterative method is still in progress

def _initialize(func, xs, args, complex_ok=False, preserve_shape=None):
    """Initialize abscissa, function, and args arrays for elementwise function

    Parameters
    ----------
    func : callable
        An elementwise function with signature

            func(x: ndarray, *args) -> ndarray

        where each element of ``x`` is a finite real and ``args`` is a tuple,
        which may contain an arbitrary number of arrays that are broadcastable
        with ``x``.
    xs : tuple of arrays
        Finite real abscissa arrays. Must be broadcastable.
    args : tuple, optional
        Additional positional arguments to be passed to `func`.
    preserve_shape : bool, default:False
        When ``preserve_shape=False`` (default), `func` may be passed
        arguments of any shape; `_scalar_optimization_loop` is permitted
        to reshape and compress arguments at will. When
        ``preserve_shape=False``, arguments passed to `func` must have shape
        `shape` or ``shape + (n,)``, where ``n`` is any integer.

    Returns
    -------
    xs, fs, args : tuple of arrays
        Broadcasted, writeable, 1D abscissa and function value arrays (or
        NumPy floats, if appropriate). The dtypes of the `xs` and `fs` are
        `xfat`; the dtype of the `args` are unchanged.
    shape : tuple of ints
        Original shape of broadcasted arrays.
    xfat : NumPy dtype
        Result dtype of abscissae, function values, and args determined using
        `np.result_type`, except integer types are promoted to `np.float64`.

    Raises
    ------
    ValueError
        If the result dtype is not that of a real scalar

    Notes
    -----
    Useful for initializing the input of SciPy functions that accept
    an elementwise callable, abscissae, and arguments; e.g.
    `scipy.optimize._chandrupatla`.
    """
    nx = len(xs)
    xp = array_namespace(*xs)

    # Try to preserve `dtype`, but we need to ensure that the arguments are at
    # This comment seems to be incomplete in the original code snippet.
    # least floats before passing them into the function; integers can overflow
    # and cause failure.
    # There might be benefit to combining the `xs` into a single array and
    # calling `func` once on the combined array. For now, keep them separate.
    xas = xp.broadcast_arrays(*xs, *args)  # broadcast and rename
    # Determine the common data type for broadcasting
    xat = xp.result_type(*[xa.dtype for xa in xas])
    # If the common type is integral, convert it to float to avoid overflow issues
    xat = xp.asarray(1.).dtype if xp.isdtype(xat, "integral") else xat
    # Separate `xs` and `args` after broadcasting
    xs, args = xas[:nx], xas[nx:]
    # Convert elements in `xs` to arrays with specified dtype `xat`
    xs = [xp.asarray(x, dtype=xat) for x in xs]  # use copy=False when implemented
    # Apply the function `func` to each element in `xs` and store results in `fs`
    fs = [xp.asarray(func(x, *args)) for x in xs]
    # Get the shape of the first element in `xs`
    shape = xs[0].shape
    # Get the shape of the first element in `fs`
    fshape = fs[0].shape

    if preserve_shape:
        # bind original shape/func now to avoid late-binding gotcha
        # Define a nested function `func` that preserves shape and original func
        def func(x, *args, shape=shape, func=func,  **kwargs):
            i = (0,)*(len(fshape) - len(shape))
            return func(x[i], *args, **kwargs)
        # Broadcast the shapes of `fshape` and `shape` using NumPy
        shape = np.broadcast_shapes(fshape, shape)  # just shapes; use of NumPy OK
        # Broadcast elements in `xs` to the new `shape`
        xs = [xp.broadcast_to(x, shape) for x in xs]
        # Broadcast elements in `args` to the new `shape`
        args = [xp.broadcast_to(arg, shape) for arg in args]

    message = ("The shape of the array returned by `func` must be the same as "
               "the broadcasted shape of `x` and all other `args`.")
    if preserve_shape is not None:  # only in tanhsinh for now
        # Update the error message if `preserve_shape` is `False`
        message = f"When `preserve_shape=False`, {message.lower()}"
    # Check if all elements in `fs` have the same shape as `shape`
    shapes_equal = [f.shape == shape for f in fs]
    # Raise an error if shapes are not equal
    if not all(shapes_equal):  # use Python all to reduce overhead
        raise ValueError(message)

    # These algorithms tend to mix the dtypes of the abscissae and function
    # values, so figure out what the result will be and convert them all to
    # that type from the outset.
    # Determine the common dtype for `fs` and `xat`
    xfat = xp.result_type(*([f.dtype for f in fs] + [xat]))
    # Raise an error if complex numbers are not allowed and `xfat` is not real
    if not complex_ok and not xp.isdtype(xfat, "real floating"):
        raise ValueError("Abscissae and function output must be real numbers.")
    # Convert elements in `xs` and `fs` to `xfat` dtype
    xs = [xp.asarray(x, dtype=xfat, copy=True) for x in xs]
    fs = [xp.asarray(f, dtype=xfat, copy=True) for f in fs]

    # To ensure that we can do indexing, we'll work with at least 1d arrays,
    # but remember the appropriate shape of the output.
    # Reshape elements in `xs`, `fs`, and `args` to 1-dimensional arrays
    xs = [xp.reshape(x, (-1,)) for x in xs]
    fs = [xp.reshape(f, (-1,)) for f in fs]
    args = [xp.reshape(xp.asarray(arg, copy=True), (-1,)) for arg in args]
    # Return function `func`, reshaped `xs`, `fs`, `args`, original `shape`, `xfat`, and `xp`
    return func, xs, fs, args, shape, xfat, xp
# 定义一个名为 _loop 的函数，用于执行矢量化标量优化算法的主循环。

def _loop(work, callback, shape, maxiter, func, args, dtype, pre_func_eval,
          post_func_eval, check_termination, post_termination_check,
          customize_result, res_work_pairs, xp, preserve_shape=False):
    """Main loop of a vectorized scalar optimization algorithm
    
    Parameters
    ----------
    work : _RichResult
        存储在迭代之间需要保留的所有变量。必须包含属性 `nit`、`nfev` 和 `success`。
    callback : callable
        用户指定的回调函数。
    shape : tuple of ints
        所有输出数组的形状。
    maxiter : int
        算法的最大迭代次数。
    func : callable
        正在优化或解决的用户指定可调用对象。
    args : tuple
        要传递给 `func` 的额外位置参数。
    dtype : NumPy dtype
        所有横坐标和函数值的通用数据类型。
    pre_func_eval : callable
        一个函数，接受 `work` 并返回 `x`，表示要评估 `func` 的活动元素 `x`。
        可以在迭代开始时在 `work` 上执行任何算法步骤，但在评估 `func` 之前。
    post_func_eval : callable
        一个函数，接受 `x`、`func(x)` 和 `work`。
        可以在评估 `func` 但在终止检查之前的迭代中间修改 `work` 的属性。
    check_termination : callable
        一个函数，接受 `work` 并返回 `stop`，一个布尔数组，指示哪些活动元素已满足终止条件。
    post_termination_check : callable
        一个函数，接受 `work`。
        可以在终止检查后但在迭代结束前修改 `work` 的任何算法步骤。
    customize_result : callable
        一个函数，接受 `res` 和 `shape`，并返回 `shape`。
        可以根据偏好修改 `res`（原地修改），并在需要时修改 `shape`。
    res_work_pairs : list of (str, str)
        标识 `res` 的属性和 `work` 的属性之间的对应关系。
        即，当适当时，将 `work` 的活动元素的属性复制到 `res` 的适当索引处。
        顺序决定了在打印 _RichResult 属性时的顺序。
    xp : module
        用于执行数组操作的库，如 NumPy 或 Cupy。
    preserve_shape : bool, optional
        是否保持形状不变的标志，默认为 False。

    Returns
    -------
    res : _RichResult
        最终的结果对象

    Notes
    -----
    除了提供结构外，此框架为矢量化优化算法提供了几项重要服务。

    - 处理迭代计数、函数评估计数、用户指定的回调以及相关的终止条件等常见任务。
    """
    # 如果没有提供 xp 参数，则抛出未实现错误
    if xp is None:
        raise NotImplementedError("Must provide xp.")
    
    # 设置回调终止标志为 False
    cb_terminate = False
    
    # 初始化结果对象和活跃元素索引数组
    n_elements = math.prod(shape)
    active = xp.arange(n_elements)  # 正在进行中的元素索引
    res_dict = {i: xp.zeros(n_elements, dtype=dtype) for i, j in res_work_pairs}  # 结果字典初始化
    res_dict['success'] = xp.zeros(n_elements, dtype=xp.bool)  # 成功标志初始化
    res_dict['status'] = xp.full(n_elements, _EINPROGRESS, dtype=xp.int32)  # 状态初始化为进行中
    res_dict['nit'] = xp.zeros(n_elements, dtype=xp.int32)  # 迭代次数初始化
    res_dict['nfev'] = xp.zeros(n_elements, dtype=xp.int32)  # 函数评估次数初始化
    res = _RichResult(res_dict)  # 创建结果对象
    work.args = args  # 设置工作参数
    
    # 检查终止条件，并更新活跃索引
    active = _check_termination(work, res, res_work_pairs, active,
                                check_termination, preserve_shape, xp)
    
    # 如果有回调函数，则准备结果并检查是否需要中止
    if callback is not None:
        temp = _prepare_result(work, res, res_work_pairs, active, shape,
                               customize_result, preserve_shape, xp)
        if _call_callback_maybe_halt(callback, temp):
            cb_terminate = True
    
    # 主循环，直到达到最大迭代次数或活跃索引为空或回调终止标志被设置
    while work.nit < maxiter and xp_size(active) and not cb_terminate and n_elements:
        x = pre_func_eval(work)  # 预处理函数评估
    
        # 如果工作参数存在且第一个参数不是一维的，则调整其维度以匹配 x
        if work.args and work.args[0].ndim != x.ndim:
            args = []
            for arg in work.args:
                n_new_dims = x.ndim - arg.ndim
                new_shape = arg.shape + (1,) * n_new_dims
                args.append(xp.reshape(arg, new_shape))
            work.args = args
    
        x_shape = x.shape
        if preserve_shape:
            x = xp.reshape(x, (shape + (-1,)))
        f = func(x, *work.args)  # 调用目标函数
        f = xp.asarray(f, dtype=dtype)  # 将结果转换为指定类型
        if preserve_shape:
            x = xp.reshape(x, x_shape)
            f = xp.reshape(f, x_shape)
        work.nfev += 1 if x.ndim == 1 else x.shape[-1]  # 更新函数评估次数
    
        post_func_eval(x, f, work)  # 后处理函数评估
    
        work.nit += 1  # 更新迭代次数
        active = _check_termination(work, res, res_work_pairs, active,
                                    check_termination, preserve_shape, xp)  # 再次检查终止条件
    
        # 如果有回调函数，则准备结果并检查是否需要中止
        if callback is not None:
            temp = _prepare_result(work, res, res_work_pairs, active, shape,
                                   customize_result, preserve_shape, xp)
            if _call_callback_maybe_halt(callback, temp):
                cb_terminate = True
                break
        if xp_size(active) == 0:
            break  # 如果活跃索引为空，则结束主循环
    
        post_termination_check(work)  # 后续终止检查处理
    
    # 根据回调终止标志设置工作状态
    work.status[:] = _ECALLBACK if cb_terminate else _ECONVERR
    
    # 返回最终准备好的结果
    return _prepare_result(work, res, res_work_pairs, active, shape,
                           customize_result, preserve_shape, xp)
# 检查终止条件，更新 `res` 中的元素与 `work` 中对应的元素，并压缩 `work`。
def _check_termination(work, res, res_work_pairs, active, check_termination,
                       preserve_shape, xp):
    # 检查终止条件，返回一个布尔数组 `stop`
    stop = check_termination(work)

    # 如果任何元素满足终止条件
    if xp.any(stop):
        # 更新结果对象的活动元素，使用已满足终止条件的活动元素
        _update_active(work, res, res_work_pairs, active, stop, preserve_shape, xp)

        # 如果需要保持形状
        if preserve_shape:
            # 根据活动索引更新 `stop`
            stop = stop[active]

        # 筛选未满足终止条件的活动索引
        proceed = ~stop
        active = active[proceed]

        # 如果不需要保持形状
        if not preserve_shape:
            # 压缩数组以避免不必要的计算
            for key, val in work.items():
                # 需要找到比这些 try/except 更好的方法
                # 需要以某种方式将可压缩的数值参数与非数值参数区分开来
                if key == 'args':
                    continue
                try:
                    work[key] = val[proceed]
                except (IndexError, TypeError, KeyError):  # 不可压缩的数组
                    work[key] = val
            # 更新参数列表
            work.args = [arg[proceed] for arg in work.args]

    # 返回更新后的活动索引
    return active


# 更新结果对象 `res` 的活动索引
def _update_active(work, res, res_work_pairs, active, mask, preserve_shape, xp):
    # 通过 `res_work_pairs` 中的映射关系，更新 `res` 的活动索引
    update_dict = {key1: work[key2] for key1, key2 in res_work_pairs}
    # 将 `work` 的状态信息转化为 `success` 字段
    update_dict['success'] = work.status == 0

    # 如果提供了掩码 `mask`
    if mask is not None:
        # 如果需要保持形状
        if preserve_shape:
            # 创建一个全零数组作为活动掩码
            active_mask = xp.zeros_like(mask)
            active_mask[active] = 1
            # 与给定掩码 `mask` 相与，生成最终的活动掩码
            active_mask = active_mask & mask
            # 对于每个键值对，根据掩码更新 `res`
            for key, val in update_dict.items():
                try:
                    res[key][active_mask] = val[active_mask]
                except (IndexError, TypeError, KeyError):
                    res[key][active_mask] = val
        else:
            # 根据活动索引和掩码筛选出最终的活动索引
            active_mask = active[mask]
            # 对于每个键值对，根据掩码更新 `res`
            for key, val in update_dict.items():
                try:
                    res[key][active_mask] = val[mask]
                except (IndexError, TypeError, KeyError):
                    res[key][active_mask] = val
    else:
        # 如果未提供掩码，直接根据活动索引更新 `res`
        for key, val in update_dict.items():
            if preserve_shape:
                try:
                    val = val[active]
                except (IndexError, TypeError, KeyError):
                    pass
            res[key][active] = val


# 准备结果对象 `res`，通过创建副本，复制最新的
def _prepare_result(work, res, res_work_pairs, active, shape, customize_result,
                    preserve_shape, xp):
    # 此函数尚未完成，待续...
    # 复制结果对象以确保不修改原始数据
    res = res.copy()
    # 根据给定的工作数据和结果数据对列表更新结果对象
    _update_active(work, res, res_work_pairs, active, None, preserve_shape, xp)

    # 根据自定义函数调整结果的形状
    shape = customize_result(res, shape)

    # 遍历结果对象中的每个键值对
    for key, val in res.items():
        # 如果使用的数组库不是 NumPy，并且值不是数值类型，可能会导致问题
        temp = xp.reshape(val, shape)
        # 将调整形状后的值重新赋给结果对象的对应键
        res[key] = temp[()] if temp.ndim == 0 else temp

    # 设置结果对象中的特殊键 '_order_keys'，按特定顺序排列
    res['_order_keys'] = ['success'] + [i for i, j in res_work_pairs]
    # 创建并返回一个包含调整后结果的 _RichResult 对象
    return _RichResult(**res)
```