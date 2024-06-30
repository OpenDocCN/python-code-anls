# `D:\src\scipysrc\scipy\scipy\optimize\_numdiff.py`

```
"""Routines for numerical differentiation."""
import functools
import numpy as np
from numpy.linalg import norm

from scipy.sparse.linalg import LinearOperator
from ..sparse import issparse, csc_matrix, csr_matrix, coo_matrix, find
from ._group_columns import group_dense, group_sparse
from scipy._lib._array_api import atleast_nd, array_namespace


def _adjust_scheme_to_bounds(x0, h, num_steps, scheme, lb, ub):
    """Adjust final difference scheme to the presence of bounds.

    Parameters
    ----------
    x0 : ndarray, shape (n,)
        Point at which we wish to estimate derivative.
    h : ndarray, shape (n,)
        Desired absolute finite difference steps.
    num_steps : int
        Number of `h` steps in one direction required to implement finite
        difference scheme. For example, 2 means that we need to evaluate
        f(x0 + 2 * h) or f(x0 - 2 * h)
    scheme : {'1-sided', '2-sided'}
        Whether steps in one or both directions are required. In other
        words '1-sided' applies to forward and backward schemes, '2-sided'
        applies to center schemes.
    lb : ndarray, shape (n,)
        Lower bounds on independent variables.
    ub : ndarray, shape (n,)
        Upper bounds on independent variables.

    Returns
    -------
    h_adjusted : ndarray, shape (n,)
        Adjusted absolute step sizes. Step size decreases only if a sign flip
        or switching to one-sided scheme doesn't allow to take a full step.
    use_one_sided : ndarray of bool, shape (n,)
        Whether to switch to one-sided scheme. Informative only for
        ``scheme='2-sided'``.
    """
    # Determine whether to use one-sided scheme based on `scheme`
    if scheme == '1-sided':
        use_one_sided = np.ones_like(h, dtype=bool)  # Default to one-sided
    elif scheme == '2-sided':
        h = np.abs(h)
        use_one_sided = np.zeros_like(h, dtype=bool)  # Default to two-sided
    else:
        raise ValueError("`scheme` must be '1-sided' or '2-sided'.")

    # If no bounds are provided, return original step sizes and scheme decision
    if np.all((lb == -np.inf) & (ub == np.inf)):
        return h, use_one_sided

    # Calculate the total step size in each direction
    h_total = h * num_steps
    h_adjusted = h.copy()

    # Calculate distances to lower and upper bounds
    lower_dist = x0 - lb
    upper_dist = ub - x0

    # Adjust step sizes based on the chosen scheme
    if scheme == '1-sided':
        # Forward step adjustments
        x = x0 + h_total
        violated = (x < lb) | (x > ub)
        fitting = np.abs(h_total) <= np.maximum(lower_dist, upper_dist)
        h_adjusted[violated & fitting] *= -1

        # Adjustments for upper and lower bounds
        forward = (upper_dist >= lower_dist) & ~fitting
        h_adjusted[forward] = upper_dist[forward] / num_steps
        backward = (upper_dist < lower_dist) & ~fitting
        h_adjusted[backward] = -lower_dist[backward] / num_steps
    # 如果方案为'2-sided'，执行以下操作
    elif scheme == '2-sided':
        # 计算是否为中央区域，即下限距离和上限距离均大于等于总高度
        central = (lower_dist >= h_total) & (upper_dist >= h_total)

        # 计算前向区域，即上限距离大于等于下限距离且不在中央区域内
        forward = (upper_dist >= lower_dist) & ~central
        # 调整前向区域的高度调整量，限制为0.5倍上限距离除以步数的最小值
        h_adjusted[forward] = np.minimum(
            h[forward], 0.5 * upper_dist[forward] / num_steps)
        # 标记前向区域为单侧使用
        use_one_sided[forward] = True

        # 计算后向区域，即上限距离小于下限距离且不在中央区域内
        backward = (upper_dist < lower_dist) & ~central
        # 调整后向区域的高度调整量，限制为负的0.5倍下限距离除以步数的最小值
        h_adjusted[backward] = -np.minimum(
            h[backward], 0.5 * lower_dist[backward] / num_steps)
        # 标记后向区域为单侧使用
        use_one_sided[backward] = True

        # 计算最小距离，为上限距离和下限距离除以步数的最小值
        min_dist = np.minimum(upper_dist, lower_dist) / num_steps
        # 调整中央区域的高度调整量，条件是不在中央区域内且绝对值的高度调整量小于等于最小距离
        adjusted_central = (~central & (np.abs(h_adjusted) <= min_dist))
        h_adjusted[adjusted_central] = min_dist[adjusted_central]
        # 标记调整中央区域为双侧使用
        use_one_sided[adjusted_central] = False

    # 返回调整后的高度数组和单侧使用标记数组
    return h_adjusted, use_one_sided
# 使用 functools.lru_cache 装饰器，实现结果缓存，提高函数性能
@functools.lru_cache
def _eps_for_method(x0_dtype, f0_dtype, method):
    """
    计算用于给定数据类型和数值微分步骤方法的相对 EPS 步长。

    对于较大的浮点类型，使用逐渐减小的步长。

    Parameters
    ----------
    f0_dtype: np.dtype
        函数评估的数据类型

    x0_dtype: np.dtype
        参数向量的数据类型

    method: {'2-point', '3-point', 'cs'}
        数值微分的方法选择

    Returns
    -------
    EPS: float
        相对步长大小。可以是 np.float16, np.float32, np.float64 中的一种

    Notes
    -----
    默认的相对步长将是 np.float64。但是，如果 x0 或 f0 是较小的浮点类型（np.float16, np.float32），
    则选择最小的浮点类型。
    """
    # 默认的 EPS 值为 np.float64 的机器精度
    EPS = np.finfo(np.float64).eps

    # 判断 x0 是否为浮点数类型
    x0_is_fp = False
    if np.issubdtype(x0_dtype, np.inexact):
        # 如果是浮点类型，则覆盖默认的 EPS 值
        EPS = np.finfo(x0_dtype).eps
        x0_itemsize = np.dtype(x0_dtype).itemsize
        x0_is_fp = True

    # 判断 f0 是否为浮点数类型
    if np.issubdtype(f0_dtype, np.inexact):
        f0_itemsize = np.dtype(f0_dtype).itemsize
        # 在 x0 是浮点数类型的情况下，选择 x0 和 f0 之间较小的 itemsize
        if x0_is_fp and f0_itemsize < x0_itemsize:
            EPS = np.finfo(f0_dtype).eps

    # 根据 method 的选择返回不同的 EPS 值
    if method in ["2-point", "cs"]:
        return EPS**0.5
    elif method in ["3-point"]:
        return EPS**(1/3)
    else:
        raise RuntimeError("Unknown step method, should be one of "
                           "{'2-point', '3-point', 'cs'}")


def _compute_absolute_step(rel_step, x0, f0, method):
    """
    根据相对步长计算绝对步长，用于有限差分计算。

    Parameters
    ----------
    rel_step: None or array-like
        有限差分计算的相对步长
    x0 : np.ndarray
        参数向量
    f0 : np.ndarray or scalar
        函数值或标量
    method : {'2-point', '3-point', 'cs'}
        数值微分方法

    Returns
    -------
    h : float
        绝对步长大小

    Notes
    -----
    `h` 总是 np.float64。但是，如果 `x0` 或 `f0` 是较小的浮点类型（如 np.float32），
    则绝对步长将从最小的浮点大小计算而来。
    """
    # 替代 np.sign(x0)，因为我们需要在 x0 == 0 时 sign_x0 为 1。
    sign_x0 = (x0 >= 0).astype(float) * 2 - 1

    # 获取相对步长 rstep
    rstep = _eps_for_method(x0.dtype, f0.dtype, method)

    # 如果 rel_step 为 None，则计算绝对步长
    if rel_step is None:
        abs_step = rstep * sign_x0 * np.maximum(1.0, np.abs(x0))
    else:
        # 用户请求了特定的相对步长。
        # 不要乘以 max(1, abs(x0)，因为如果 x0 < 1，则不使用他们请求的步长。
        abs_step = rel_step * sign_x0 * np.abs(x0)

        # 但是我们不希望 abs_step 为 0，如果 rel_step 是 0，或者 x0 是 0，这种情况可能发生。
        # 相反，用一个实际的步长来替代
        dx = ((x0 + abs_step) - x0)
        abs_step = np.where(dx == 0,
                            rstep * sign_x0 * np.maximum(1.0, np.abs(x0)),
                            abs_step)

    # 返回最终确定的绝对步长
    return abs_step
def _prepare_bounds(bounds, x0):
    """
    准备新样式的边界，从指定 x0 中的下限和上限的二元组。如果一个值没有边界，则预期的下/上界将是 -np.inf/np.inf。

    Examples
    --------
    >>> _prepare_bounds([(0, 1, 2), (1, 2, np.inf)], [0.5, 1.5, 2.5])
    (array([0., 1., 2.]), array([ 1.,  2., inf]))
    """
    # 将边界转换为浮点数的 NumPy 数组
    lb, ub = (np.asarray(b, dtype=float) for b in bounds)
    # 如果 lb 是标量，将其扩展为与 x0 相同形状的数组
    if lb.ndim == 0:
        lb = np.resize(lb, x0.shape)
    # 如果 ub 是标量，将其扩展为与 x0 相同形状的数组
    if ub.ndim == 0:
        ub = np.resize(ub, x0.shape)
    # 返回处理后的下限和上限数组
    return lb, ub


def group_columns(A, order=0):
    """为稀疏有限差分组合列的二维矩阵。

    如果在每一行中至少有一个列为零，则两列属于同一组。使用贪婪的顺序算法构建组。

    Parameters
    ----------
    A : array_like or sparse matrix, shape (m, n)
        要分组列的矩阵。
    order : int, iterable of int with shape (n,) or None
        定义列枚举顺序的置换数组。
        如果是 int 或 None，则使用随机置换，并使用 `order` 作为随机种子。默认为 0，即使用随机置换但保证可重复性。

    Returns
    -------
    groups : ndarray of int, shape (n,)
        包含从 0 到 n_groups-1 的值，其中 n_groups 是找到的组数。每个值 ``groups[i]`` 是一个分配给第 i 列的组的索引。仅当 n_groups 显著小于 n 时，此过程才有帮助。

    References
    ----------
    .. [1] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
           sparse Jacobian matrices", Journal of the Institute of Mathematics
           and its Applications, 13 (1974), pp. 117-120.
    """
    # 如果 A 是稀疏矩阵，则将其转换为 CSC 格式的稀疏矩阵
    if issparse(A):
        A = csc_matrix(A)
    else:
        # 至少将 A 转换为二维数组并将非零元素转换为整数类型
        A = np.atleast_2d(A)
        A = (A != 0).astype(np.int32)

    # 确保 A 是二维矩阵
    if A.ndim != 2:
        raise ValueError("`A` must be 2-dimensional.")

    m, n = A.shape

    # 如果 order 是 None 或标量，则使用随机种子创建随机排列
    if order is None or np.isscalar(order):
        rng = np.random.RandomState(order)
        order = rng.permutation(n)
    else:
        # 将 order 转换为 NumPy 数组，并检查其形状是否正确
        order = np.asarray(order)
        if order.shape != (n,):
            raise ValueError("`order` has incorrect shape.")

    # 按照 order 对 A 的列进行重新排序
    A = A[:, order]

    # 根据 A 的类型，确定是使用稀疏组合算法还是密集组合算法
    if issparse(A):
        groups = group_sparse(m, n, A.indices, A.indptr)
    else:
        groups = group_dense(m, n, A)

    # 将分组结果按照 order 还原到原始顺序
    groups[order] = groups.copy()

    # 返回分组结果
    return groups


def approx_derivative(fun, x0, method='3-point', rel_step=None, abs_step=None,
                      f0=None, bounds=(-np.inf, np.inf), sparsity=None,
                      as_linear_operator=False, args=(), kwargs={}):
    """计算向量值函数的有限差分近似的导数。

    如果一个函数从 R^n 映射到 R^m，则它的导数形成 m-by-n 的矩阵。

    Parameters
    ----------
    fun : callable
        要计算其导数的函数。
    x0 : ndarray, shape (n,)
        函数的输入向量，其中 n 是函数的参数数量。
    method : {'3-point'}, optional
        计算导数的方法，默认为 '3-point'。
    rel_step : None or ndarray, shape (n,), optional
        相对步长，用于 '3-point' 方法，默认为 None。
    abs_step : None or ndarray, shape (n,), optional
        绝对步长，用于 '3-point' 方法，默认为 None。
    f0 : ndarray, shape (m,), optional
        在 x0 处计算的函数值，用于 '3-point' 方法，默认为 None。
    bounds : tuple or list, optional
        变量的下限和上限，默认为 (-np.inf, np.inf)。
    sparsity : None, optional
        可选的稀疏模式，默认为 None。
    as_linear_operator : bool, optional
        如果为 True，则返回的导数作为线性操作符，默认为 False。
    args : tuple, optional
        传递给 fun 的额外参数，默认为空元组。
    kwargs : dict, optional
        传递给 fun 的额外关键字参数，默认为空字典。
    
    Returns
    -------
    derivatives : ndarray
        函数在 x0 处计算的导数数组。

    """
    # 这里是函数的实现，具体内容省略
    # 定义函数 `approx_derivative`，用于估算函数在给定点处的偏导数
    def approx_derivative(fun, x0, method='3-point', rel_step=None,
                          abs_step=None, f0=None, bounds=None):
        # 根据 `fun` 的定义，估算函数在 `x0` 处的偏导数
        """
        Estimate the Jacobian matrix of a vector-valued function using
        finite differences.

        This function computes an approximation of the Jacobian matrix,
        where an element (i, j) is a partial derivative of f[i] with respect
        to x[j].

        Parameters
        ----------
        fun : callable
            Function of which to estimate the derivatives. The argument x
            passed to this function is ndarray of shape (n,) (never a scalar
            even if n=1). It must return 1-D array_like of shape (m,) or a scalar.
        x0 : array_like of shape (n,) or float
            Point at which to estimate the derivatives. Float will be converted
            to a 1-D array.
        method : {'3-point', '2-point', 'cs'}, optional
            Finite difference method to use:
                - '2-point' - use the first order accuracy forward or backward
                              difference.
                - '3-point' - use central difference in interior points and the
                              second order accuracy forward or backward difference
                              near the boundary.
                - 'cs' - use a complex-step finite difference scheme. This assumes
                         that the user function is real-valued and can be
                         analytically continued to the complex plane. Otherwise,
                         produces bogus results.
        rel_step : None or array_like, optional
            Relative step size to use. If None (default) the absolute step size is
            computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``, with
            `rel_step` being selected automatically, see Notes. Otherwise
            ``h = rel_step * sign(x0) * abs(x0)``. For ``method='3-point'`` the
            sign of `h` is ignored. The calculated step size is possibly adjusted
            to fit into the bounds.
        abs_step : array_like, optional
            Absolute step size to use, possibly adjusted to fit into the bounds.
            For ``method='3-point'`` the sign of `abs_step` is ignored. By default
            relative steps are used, only if ``abs_step is not None`` are absolute
            steps used.
        f0 : None or array_like, optional
            If not None it is assumed to be equal to ``fun(x0)``, in this case
            the ``fun(x0)`` is not called. Default is None.
        bounds : tuple of array_like, optional
            Lower and upper bounds on independent variables. Defaults to no bounds.
            Each bound must match the size of `x0` or be a scalar, in the latter
            case the bound will be the same for all variables. Use it to limit the
            range of function evaluation. Bounds checking is not implemented
            when `as_linear_operator` is True.
        """
    # `sparsity`参数定义了Jacobian矩阵的稀疏结构。可以为None、类数组、稀疏矩阵或二元组。
    # 如果Jacobian矩阵在每行中只有少量非零元素，则可以通过单次函数评估估计其多列[3]_。
    # 要执行这种经济计算，需要两个要素：
    #
    # * `structure`：形状为(m, n)的数组或稀疏矩阵。零元素意味着Jacobian对应元素恒等于零。
    # * `groups`：形状为(n,)的数组。给定稀疏结构的列分组，使用`group_columns`函数获取。
    #
    # 单个数组或稀疏矩阵被解释为稀疏结构，并且在函数内部计算分组。元组被解释为(structure, groups)。
    # 如果为None（默认），将使用标准的密集差分。
    as_linear_operator : bool, optional
        # 当为True时，函数返回一个`scipy.sparse.linalg.LinearOperator`。
        # 否则，根据`sparsity`的定义返回一个密集数组或稀疏矩阵。线性操作符提供了一种高效计算`J.dot(p)`的方式，
        # 其中`p`是形状为(n,)的任意向量，但不允许直接访问矩阵的单个元素。默认情况下，`as_linear_operator`为False。
    args, kwargs : tuple and dict, optional
        # 额外传递给`fun`的参数。默认都为空。
        # 调用签名为`fun(x, *args, **kwargs)`。
    """
    If any of the absolute or relative steps produces an indistinguishable
    difference from the original `x0`, ``(x0 + dx) - x0 == 0``, then a
    automatic step size is substituted for that particular entry.
    """

    # 检查所选的数值微分方法是否有效，必须是'2-point', '3-point', 'cs'中的一个，否则抛出数值错误
    if method not in ['2-point', '3-point', 'cs']:
        raise ValueError("Unknown method '%s'. " % method)

    # 使用数组命名空间函数将x0转换为合适的数组表示
    xp = array_namespace(x0)
    # 将x0至少提升为1维，并根据需要使用xp（数组库）进行扩展
    _x = atleast_nd(x0, ndim=1, xp=xp)
    # 如果_x的数据类型是浮点数，确定数据类型为xp.float64
    _dtype = xp.float64
    if xp.isdtype(_x.dtype, "real floating"):
        _dtype = _x.dtype

    # 将x0强制转换为指定的浮点数类型
    x0 = xp.astype(_x, _dtype)

    # 如果x0的维度大于1，则引发值错误异常
    if x0.ndim > 1:
        raise ValueError("`x0` must have at most 1 dimension.")

    # 准备边界，将边界参数与x0匹配
    lb, ub = _prepare_bounds(bounds, x0)

    # 如果lb或ub的形状与x0的形状不一致，则引发值错误异常
    if lb.shape != x0.shape or ub.shape != x0.shape:
        raise ValueError("Inconsistent shapes between bounds and `x0`.")
    # 如果要求返回线性操作符，并且下界和上界不全为无穷大，则抛出值错误异常
    if as_linear_operator and not (np.all(np.isinf(lb))
                                   and np.all(np.isinf(ub))):
        raise ValueError("Bounds not supported when "
                         "`as_linear_operator` is True.")

    # 定义一个包装函数，用于确保用户函数使用与初始点 x0 相同的浮点类型
    def fun_wrapped(x):
        # 如果 x 的数据类型为实数浮点数，将 x 转换为与 x0 相同的数据类型
        if xp.isdtype(x.dtype, "real floating"):
            x = xp.astype(x, x0.dtype)

        # 调用用户定义的函数 fun，至少转换为一维数组
        f = np.atleast_1d(fun(x, *args, **kwargs))
        # 如果返回值的维度大于1，则抛出运行时错误异常
        if f.ndim > 1:
            raise RuntimeError("`fun` return value has "
                               "more than 1 dimension.")
        return f

    # 如果没有提供初始函数值 f0，则计算初始点 x0 处的函数值
    if f0 is None:
        f0 = fun_wrapped(x0)
    else:
        # 将提供的初始函数值 f0 至少转换为一维数组
        f0 = np.atleast_1d(f0)
        # 如果 f0 的维度大于1，则抛出值错误异常
        if f0.ndim > 1:
            raise ValueError("`f0` passed has more than 1 dimension.")

    # 如果初始点 x0 超出了边界约束 lb 和 ub，则抛出值错误异常
    if np.any((x0 < lb) | (x0 > ub)):
        raise ValueError("`x0` violates bound constraints.")

    # 如果要求返回线性操作符，则根据指定方法计算线性操作符的差分
    if as_linear_operator:
        # 如果未指定相对步长 rel_step，则根据指定方法确定一个默认的相对步长
        if rel_step is None:
            rel_step = _eps_for_method(x0.dtype, f0.dtype, method)

        # 返回使用线性操作符差分的结果
        return _linear_operator_difference(fun_wrapped, x0,
                                           f0, rel_step, method)
    else:
        # 默认情况下使用相对步长 rel_step
        if abs_step is None:
            # 计算绝对步长 h，考虑到方法和边界情况的影响
            h = _compute_absolute_step(rel_step, x0, f0, method)
        else:
            # 用户指定了绝对步长
            sign_x0 = (x0 >= 0).astype(float) * 2 - 1
            h = abs_step

            # 不能有零步长。如果 x0 很大或很小，则使用相对步长
            dx = ((x0 + h) - x0)
            h = np.where(dx == 0,
                         _eps_for_method(x0.dtype, f0.dtype, method) *
                         sign_x0 * np.maximum(1.0, np.abs(x0)),
                         h)

        # 根据指定方法调整步长以适应不同的数值差分方案
        if method == '2-point':
            h, use_one_sided = _adjust_scheme_to_bounds(
                x0, h, 1, '1-sided', lb, ub)
        elif method == '3-point':
            h, use_one_sided = _adjust_scheme_to_bounds(
                x0, h, 1, '2-sided', lb, ub)
        elif method == 'cs':
            use_one_sided = False

        # 如果稀疏性未指定，则返回密集数值差分的结果
        if sparsity is None:
            return _dense_difference(fun_wrapped, x0, f0, h,
                                     use_one_sided, method)
        else:
            # 如果稀疏性不是稀疏矩阵且长度为2，则解析结构和组
            if not issparse(sparsity) and len(sparsity) == 2:
                structure, groups = sparsity
            else:
                structure = sparsity
                groups = group_columns(sparsity)

            # 将结构转换为稀疏矩阵形式，确保组是至少一维的数组
            if issparse(structure):
                structure = csc_matrix(structure)
            else:
                structure = np.atleast_2d(structure)

            groups = np.atleast_1d(groups)
            # 返回稀疏数值差分的结果
            return _sparse_difference(fun_wrapped, x0, f0, h,
                                      use_one_sided, structure,
                                      groups, method)
# 定义一个函数，计算线性算子的差分
def _linear_operator_difference(fun, x0, f0, h, method):
    # 获取向量 f0 的长度
    m = f0.size
    # 获取向量 x0 的长度
    n = x0.size

    # 根据 method 参数选择不同的差分方法
    if method == '2-point':
        # 定义一个向量乘以矩阵的函数 matvec(p)
        def matvec(p):
            # 如果向量 p 全为零，则返回全零向量
            if np.array_equal(p, np.zeros_like(p)):
                return np.zeros(m)
            # 计算步长 dx
            dx = h / norm(p)
            # 计算新的点 x
            x = x0 + dx * p
            # 计算函数在新点 x 处的值与原始值 f0 的差分 df
            df = fun(x) - f0
            # 返回差分值除以步长 dx
            return df / dx

    elif method == '3-point':
        # 定义一个向量乘以矩阵的函数 matvec(p)
        def matvec(p):
            # 如果向量 p 全为零，则返回全零向量
            if np.array_equal(p, np.zeros_like(p)):
                return np.zeros(m)
            # 计算步长 dx
            dx = 2 * h / norm(p)
            # 计算两个新的点 x1 和 x2
            x1 = x0 - (dx / 2) * p
            x2 = x0 + (dx / 2) * p
            # 计算函数在新点 x1 和 x2 处的值
            f1 = fun(x1)
            f2 = fun(x2)
            # 计算两点之间的差分 df，并除以步长 dx
            df = f2 - f1
            return df / dx

    elif method == 'cs':
        # 定义一个向量乘以矩阵的函数 matvec(p)
        def matvec(p):
            # 如果向量 p 全为零，则返回全零向量
            if np.array_equal(p, np.zeros_like(p)):
                return np.zeros(m)
            # 计算复数步长 dx，并乘以虚数单位 1.j
            dx = h / norm(p)
            x = x0 + dx * p * 1.j
            # 计算函数在新点 x 处的虚部作为差分值 df，并除以步长 dx
            f1 = fun(x)
            df = f1.imag
            return df / dx

    else:
        # 如果 method 参数不在预期的选项中，抛出运行时错误
        raise RuntimeError("Never be here.")

    # 返回一个线性算子 LinearOperator，使用 matvec 函数定义
    return LinearOperator((m, n), matvec)


# 定义一个函数，计算密集算子的差分
def _dense_difference(fun, x0, f0, h, use_one_sided, method):
    # 获取向量 f0 的长度
    m = f0.size
    # 获取向量 x0 的长度
    n = x0.size
    # 创建一个空的矩阵用于存放 Jacobian 的转置
    J_transposed = np.empty((n, m))
    # 创建副本 x1, x2, xc 以便在迭代中修改
    x1 = x0.copy()
    x2 = x0.copy()
    xc = x0.astype(complex, copy=True)

    # 遍历步长 h 中的每一个元素
    for i in range(h.size):
        # 根据 method 参数选择不同的差分方法
        if method == '2-point':
            # 修改 x1[i] 以计算差分 dx
            x1[i] += h[i]
            dx = x1[i] - x0[i]  # 重新计算确切可表示的 dx
            # 计算函数在新点 x1 处的值与原始值 f0 的差分 df
            df = fun(x1) - f0
        elif method == '3-point' and use_one_sided[i]:
            # 修改 x1[i] 和 x2[i] 以计算差分 dx
            x1[i] += h[i]
            x2[i] += 2 * h[i]
            dx = x2[i] - x0[i]
            # 计算函数在新点 x1 和 x2 处的值
            f1 = fun(x1)
            f2 = fun(x2)
            # 计算三点差分 df，并除以 dx
            df = -3.0 * f0 + 4 * f1 - f2
        elif method == '3-point' and not use_one_sided[i]:
            # 修改 x1[i] 和 x2[i] 以计算差分 dx
            x1[i] -= h[i]
            x2[i] += h[i]
            dx = x2[i] - x1[i]
            # 计算函数在新点 x1 和 x2 处的值
            f1 = fun(x1)
            f2 = fun(x2)
            # 计算三点差分 df，并除以 dx
            df = f2 - f1
        elif method == 'cs':
            # 修改 xc[i] 以计算复数步长 dx
            xc[i] += h[i] * 1.j
            # 计算函数在新点 xc 处的虚部作为差分 df，并设置 dx
            f1 = fun(xc)
            df = f1.imag
            dx = h[i]
        else:
            # 如果 method 参数不在预期的选项中，抛出运行时错误
            raise RuntimeError("Never be here.")

        # 将计算得到的 df/dx 存储在 J_transposed 的第 i 列
        J_transposed[i] = df / dx
        # 恢复 x1[i], x2[i], xc[i] 的原始值 x0[i]
        x1[i] = x2[i] = xc[i] = x0[i]

    # 如果 m 等于 1，则将 J_transposed 摊平为一维数组
    if m == 1:
        J_transposed = np.ravel(J_transposed)

    # 返回 Jacobian 的转置矩阵
    return J_transposed.T


# 定义一个函数，计算稀疏算子的差分
def _sparse_difference(fun, x0, f0, h, use_one_sided,
                       structure, groups, method):
    # 获取向量 f0 的长度
    m = f0.size
    # 获取向量 x0 的长度
    n = x0.size
    # 创建空列表用于存放稀疏矩阵的行索引、列索引和分数
    row_indices = []
    col_indices = []
    fractions = []

    # 计算分组数目
    n_groups = np.max(groups) + 1
    for group in range(n_groups):
        # 对于每个组，同时扰动属于同一组的变量。
        e = np.equal(group, groups)
        # 根据组的掩码，生成扰动向量。
        h_vec = h * e
        if method == '2-point':
            # 使用两点差分方法
            x = x0 + h_vec
            # 计算变量的增量
            dx = x - x0
            # 计算函数在扰动后的差值
            df = fun(x) - f0
            # 将结果写入对应于扰动变量的列中。
            cols, = np.nonzero(e)
            # 在结构数组中找到所选列中所有非零元素的行索引和列索引。
            i, j, _ = find(structure[:, cols])
            # 恢复完整数组中的列索引。
            j = cols[j]
        elif method == '3-point':
            # 使用三点差分方法
            # 分别处理单侧和双侧方案。
            x1 = x0.copy()
            x2 = x0.copy()

            mask_1 = use_one_sided & e
            x1[mask_1] += h_vec[mask_1]
            x2[mask_1] += 2 * h_vec[mask_1]

            mask_2 = ~use_one_sided & e
            x1[mask_2] -= h_vec[mask_2]
            x2[mask_2] += h_vec[mask_2]

            # 计算增量向量
            dx = np.zeros(n)
            dx[mask_1] = x2[mask_1] - x0[mask_1]
            dx[mask_2] = x2[mask_2] - x1[mask_2]

            # 计算在两个点上函数的值
            f1 = fun(x1)
            f2 = fun(x2)

            cols, = np.nonzero(e)
            i, j, _ = find(structure[:, cols])
            j = cols[j]

            mask = use_one_sided[j]
            df = np.empty(m)

            # 计算导数的分数形式
            rows = i[mask]
            df[rows] = -3 * f0[rows] + 4 * f1[rows] - f2[rows]

            rows = i[~mask]
            df[rows] = f2[rows] - f1[rows]
        elif method == 'cs':
            # 使用复数步长差分方法
            f1 = fun(x0 + h_vec*1.j)
            # 提取虚部作为导数
            df = f1.imag
            dx = h_vec
            cols, = np.nonzero(e)
            i, j, _ = find(structure[:, cols])
            j = cols[j]
        else:
            # 抛出数值错误，应该不会到达这里。
            raise ValueError("Never be here.")

        # 计算分数的数组，存储行索引、列索引和分数。
        row_indices.append(i)
        col_indices.append(j)
        fractions.append(df[i] / dx[j])

    # 合并所有行索引、列索引和分数为一个数组，并构建稀疏矩阵。
    row_indices = np.hstack(row_indices)
    col_indices = np.hstack(col_indices)
    fractions = np.hstack(fractions)
    J = coo_matrix((fractions, (row_indices, col_indices)), shape=(m, n))
    return csr_matrix(J)
# 检查一个计算导数（雅可比或梯度）的函数是否正确，通过与有限差分逼近进行比较。

def check_derivative(fun, jac, x0, bounds=(-np.inf, np.inf), args=(),
                     kwargs={}):
    """Check correctness of a function computing derivatives (Jacobian or
    gradient) by comparison with a finite difference approximation.
    
    Parameters
    ----------
    fun : callable
        要估计导数的函数。传递给此函数的参数 x 是形状为 (n,) 的 ndarray（即使 n=1 也不是标量）。
        它必须返回形状为 (m,) 的一维数组或标量。
    jac : callable
        计算 `fun` 的雅可比矩阵的函数。其参数 x 的工作方式与 `fun` 相同。
        返回值必须是具有适当形状的 array_like 或稀疏矩阵。
    x0 : array_like of shape (n,) or float
        用于估计导数的点。如果是 float，会转换为一维数组。
    bounds : 2-tuple of array_like, optional
        自变量的下限和上限。默认为无限制。每个边界必须与 `x0` 的大小匹配或为标量，
        在后一种情况下，所有变量的边界将相同。用于限制函数评估的范围。
    args, kwargs : tuple and dict, optional
        传递给 `fun` 和 `jac` 的额外参数。默认为空。调用签名为 ``fun(x, *args, **kwargs)``
        和 `jac` 的调用方式相同。
        
    Returns
    -------
    accuracy : float
        所有相对误差绝对值大于 1 的元素的最大值，和所有绝对值小于等于 1 的元素的绝对误差。
        如果 `accuracy` 在 1e-6 或更低的数量级，则你的 `jac` 实现可能是正确的。

    See Also
    --------
    approx_derivative : 计算导数的有限差分逼近。

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize._numdiff import check_derivative
    >>>
    >>>
    >>> def f(x, c1, c2):
    ...     return np.array([x[0] * np.sin(c1 * x[1]),
    ...                      x[0] * np.cos(c2 * x[1])])
    ...
    >>> def jac(x, c1, c2):
    ...     return np.array([
    ...         [np.sin(c1 * x[1]),  c1 * x[0] * np.cos(c1 * x[1])],
    ...         [np.cos(c2 * x[1]), -c2 * x[0] * np.sin(c2 * x[1])]
    ...     ])
    ...
    >>>
    >>> x0 = np.array([1.0, 0.5 * np.pi])
    >>> check_derivative(f, jac, x0, args=(1, 2))
    2.4492935982947064e-16
    """
    
    # 使用给定参数调用 `jac` 函数，得到要测试的雅可比矩阵 `J_to_test`
    J_to_test = jac(x0, *args, **kwargs)
    
    # 如果 `J_to_test` 是稀疏矩阵
    if issparse(J_to_test):
        # 使用稀疏矩阵 `J_to_test` 进行有限差分逼近 `approx_derivative`
        J_diff = approx_derivative(fun, x0, bounds=bounds, sparsity=J_to_test,
                                   args=args, kwargs=kwargs)
        
        # 将 `J_to_test` 转换为 CSR 稀疏矩阵
        J_to_test = csr_matrix(J_to_test)
        
        # 计算绝对误差 `abs_err`
        abs_err = J_to_test - J_diff
        
        # 找到非零元素的位置和绝对误差数据
        i, j, abs_err_data = find(abs_err)
        
        # 从 `J_diff` 中提取数据
        J_diff_data = np.asarray(J_diff[i, j]).ravel()
        
        # 返回最大的相对误差绝对值除以最大的绝对误差数据绝对值和 1 的最大值
        return np.max(np.abs(abs_err_data) /
                      np.maximum(1, np.abs(J_diff_data)))
    else:
        # 使用数值方法计算函数在给定点的近似导数
        J_diff = approx_derivative(fun, x0, bounds=bounds,
                                   args=args, kwargs=kwargs)
        # 计算测试得到的雅可比矩阵与数值近似导数之间的绝对误差
        abs_err = np.abs(J_to_test - J_diff)
        # 计算相对误差，避免除以零的情况
        return np.max(abs_err / np.maximum(1, np.abs(J_diff)))
```