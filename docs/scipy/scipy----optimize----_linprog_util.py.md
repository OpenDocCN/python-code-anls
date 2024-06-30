# `D:\src\scipysrc\scipy\scipy\optimize\_linprog_util.py`

```
"""
Method agnostic utility functions for linear programming
"""

# 导入所需库和模块
import numpy as np  # 导入 NumPy 库，用于数值计算
import scipy.sparse as sps  # 导入 SciPy 库中的稀疏矩阵模块
from warnings import warn  # 导入警告模块中的 warn 函数
from ._optimize import OptimizeWarning  # 导入自定义优化警告类
from scipy.optimize._remove_redundancy import (  # 导入用于去除冗余的函数
    _remove_redundancy_svd, _remove_redundancy_pivot_sparse,
    _remove_redundancy_pivot_dense, _remove_redundancy_id
)
from collections import namedtuple  # 导入命名元组模块

# 定义线性规划问题的命名元组
_LPProblem = namedtuple('_LPProblem',
                        'c A_ub b_ub A_eq b_eq bounds x0 integrality')
_LPProblem.__new__.__defaults__ = (None,) * 7  # 设置默认参数，只要 c 是必需的
_LPProblem.__doc__ = \
    """ Represents a linear-programming problem.

    Attributes
    ----------
    c : 1D array
        The coefficients of the linear objective function to be minimized.
    A_ub : 2D array, optional
        The inequality constraint matrix. Each row of ``A_ub`` specifies the
        coefficients of a linear inequality constraint on ``x``.
    b_ub : 1D array, optional
        The inequality constraint vector. Each element represents an
        upper bound on the corresponding value of ``A_ub @ x``.
    A_eq : 2D array, optional
        The equality constraint matrix. Each row of ``A_eq`` specifies the
        coefficients of a linear equality constraint on ``x``.
    b_eq : 1D array, optional
        The equality constraint vector. Each element of ``A_eq @ x`` must equal
        the corresponding element of ``b_eq``.
    bounds : various valid formats, optional
        The bounds of ``x``, as ``min`` and ``max`` pairs.
        If bounds are specified for all N variables separately, valid formats
        are:
        * a 2D array (N x 2);
        * a sequence of N sequences, each with 2 values.
        If all variables have the same bounds, the bounds can be specified as
        a 1-D or 2-D array or sequence with 2 scalar values.
        If all variables have a lower bound of 0 and no upper bound, the bounds
        parameter can be omitted (or given as None).
        Absent lower and/or upper bounds can be specified as -numpy.inf (no
        lower bound), numpy.inf (no upper bound) or None (both).
    x0 : 1D array, optional
        Guess values of the decision variables, which will be refined by
        the optimization algorithm. This argument is currently used only by the
        'revised simplex' method, and can only be used if `x0` represents a
        basic feasible solution.
"""
    integrality : 1-D array or int, optional
        # 表示每个决策变量的整数性约束类型。

        ``0`` : 连续变量；没有整数约束。

        ``1`` : 整数变量；决策变量必须是整数，在给定的`bounds`范围内。

        ``2`` : 半连续变量；决策变量必须在给定的`bounds`范围内或者取值为 ``0``。

        ``3`` : 半整数变量；决策变量必须是整数在给定的`bounds`范围内或者取值为 ``0``。

        默认情况下，所有变量都是连续的。

        对于混合整数约束，请提供一个形状为 `c.shape` 的数组。
        如果需要从较短的输入推断每个决策变量的约束，请使用 `np.broadcast_to` 将参数广播到 `c.shape` 的形状。

        此参数目前仅在 ``'highs'`` 方法中使用，否则将被忽略。

    Notes
    -----
    # 这个命名元组支持两种初始化方式：
    >>> lp1 = _LPProblem(c=[-1, 4], A_ub=[[-3, 1], [1, 2]], b_ub=[6, 4])
    >>> lp2 = _LPProblem([-1, 4], [[-3, 1], [1, 2]], [6, 4])

    注意，这里只有 ``c`` 是必需的参数，而其他所有参数 ``A_ub``, ``b_ub``, ``A_eq``, ``b_eq``, ``bounds``, ``x0`` 都是可选的，默认值为 None。
    例如，可以设置 ``A_eq`` 和 ``b_eq`` 而不设置 ``A_ub`` 或 ``b_ub``：
    >>> lp3 = _LPProblem(c=[-1, 4], A_eq=[[2, 1]], b_eq=[10])
    """
# 检查稀疏输入的有效性，用于线性规划中的稀疏预处理
def _check_sparse_inputs(options, meth, A_ub, A_eq):
    """
    Check the provided ``A_ub`` and ``A_eq`` matrices conform to the specified
    optional sparsity variables.

    Parameters
    ----------
    A_ub : 2-D array, optional
        2-D array such that ``A_ub @ x`` gives the values of the upper-bound
        inequality constraints at ``x``.
    A_eq : 2-D array, optional
        2-D array such that ``A_eq @ x`` gives the values of the equality
        constraints at ``x``.
    options : dict
        A dictionary of solver options. All methods accept the following
        generic options:

            maxiter : int
                Maximum number of iterations to perform.
            disp : bool
                Set to True to print convergence messages.

        For method-specific options, see :func:`show_options('linprog')`.
    method : str, optional
        The algorithm used to solve the standard form problem.

    Returns
    -------
    A_ub : 2-D array, optional
        2-D array such that ``A_ub @ x`` gives the values of the upper-bound
        inequality constraints at ``x``.
    A_eq : 2-D array, optional
        2-D array such that ``A_eq @ x`` gives the values of the equality
        constraints at ``x``.
    options : dict
        A dictionary of solver options. All methods accept the following
        generic options:

            maxiter : int
                Maximum number of iterations to perform.
            disp : bool
                Set to True to print convergence messages.

        For method-specific options, see :func:`show_options('linprog')`.
    """
    # 这是一个未记录的用于单元测试稀疏预处理的选项
    _sparse_presolve = options.pop('_sparse_presolve', False)
    # 如果 `_sparse_presolve` 为 True 且 A_eq 不为 None，则将 A_eq 转换为稀疏 COO 格式
    if _sparse_presolve and A_eq is not None:
        A_eq = sps.coo_matrix(A_eq)
    # 如果 `_sparse_presolve` 为 True 且 A_ub 不为 None，则将 A_ub 转换为稀疏 COO 格式
    if _sparse_presolve and A_ub is not None:
        A_ub = sps.coo_matrix(A_ub)

    # 检查 A_eq 或 A_ub 是否为稀疏矩阵
    sparse_constraint = sps.issparse(A_eq) or sps.issparse(A_ub)

    # 首选方法集合和密集方法集合的定义
    preferred_methods = {"highs", "highs-ds", "highs-ipm"}
    dense_methods = {"simplex", "revised simplex"}
    # 如果方法 meth 在密集方法集合中且有稀疏约束，则抛出 ValueError 异常
    if meth in dense_methods and sparse_constraint:
        raise ValueError(f"Method '{meth}' does not support sparse "
                         "constraint matrices. Please consider using one of "
                         f"{preferred_methods}.")

    # 获取选项字典中的 `sparse` 键对应的值，默认为 False
    sparse = options.get('sparse', False)
    # 如果 `sparse` 为 False 且发现稀疏约束，则将 `sparse` 设置为 True，并发出警告
    if not sparse and sparse_constraint and meth == 'interior-point':
        options['sparse'] = True
        warn("Sparse constraint matrix detected; setting 'sparse':True.",
             OptimizeWarning, stacklevel=4)
    return options, A_ub, A_eq


def _format_A_constraints(A, n_x, sparse_lhs=False):
    """Format the left hand side of the constraints to a 2-D array

    Parameters
    ----------
    A : 2-D array
        2-D array such that ``A @ x`` gives the values of the upper-bound
        (in)equality constraints at ``x``.
    n_x : int
        The number of variables in the linear programming problem.
    """
    sparse_lhs : bool
        是否有 `A_ub` 或 `A_eq` 是稀疏的。如果是，则返回一个 coo_matrix 而不是一个 numpy 数组。

    Returns
    -------
    np.ndarray or sparse.coo_matrix
        返回一个二维数组，使得 ``A @ x`` 在 ``x`` 处给出上界约束的值。

    """
    # 如果 sparse_lhs 为 True，则返回一个稀疏的 coo_matrix
    if sparse_lhs:
        return sps.coo_matrix(
            (0, n_x) if A is None else A, dtype=float, copy=True
        )
    # 如果 A 为 None，则返回一个形状为 (0, n_x) 的全零 numpy 数组
    elif A is None:
        return np.zeros((0, n_x), dtype=float)
    # 否则返回一个复制的 numpy 数组 A
    else:
        return np.array(A, dtype=float, copy=True)
# 格式化约束的上限到一个一维数组

def _format_b_constraints(b):
    """Format the upper bounds of the constraints to a 1-D array
    
    Parameters
    ----------
    b : 1-D array
        1-D array of values representing the upper-bound of each (in)equality
        constraint (row) in ``A``.
        
    Returns
    -------
    1-D np.array
        1-D array of values representing the upper-bound of each (in)equality
        constraint (row) in ``A``.
    """
    # 如果 b 是 None，则返回一个空的浮点数数组
    if b is None:
        return np.array([], dtype=float)
    # 将 b 转换为浮点数类型的数组，并去除多余的维度（squeeze）
    b = np.array(b, dtype=float, copy=True).squeeze()
    # 如果 b 的大小不为 1，则直接返回 b；否则将 b 重塑为一维数组并返回
    return b if b.size != 1 else b.reshape(-1)


# 清理线性规划问题的输入数据格式

def _clean_inputs(lp):
    """
    Given user inputs for a linear programming problem, return the
    objective vector, upper bound constraints, equality constraints,
    and simple bounds in a preferred format.

    Parameters
    ----------
    lp : A `scipy.optimize._linprog_util._LPProblem` consisting of the following fields:

        c : 1D array
            The coefficients of the linear objective function to be minimized.
        A_ub : 2D array, optional
            The inequality constraint matrix. Each row of ``A_ub`` specifies the
            coefficients of a linear inequality constraint on ``x``.
        b_ub : 1D array, optional
            The inequality constraint vector. Each element represents an
            upper bound on the corresponding value of ``A_ub @ x``.
        A_eq : 2D array, optional
            The equality constraint matrix. Each row of ``A_eq`` specifies the
            coefficients of a linear equality constraint on ``x``.
        b_eq : 1D array, optional
            The equality constraint vector. Each element of ``A_eq @ x`` must equal
            the corresponding element of ``b_eq``.
        bounds : various valid formats, optional
            The bounds of ``x``, as ``min`` and ``max`` pairs.
            If bounds are specified for all N variables separately, valid formats are:
            * a 2D array (2 x N or N x 2);
            * a sequence of N sequences, each with 2 values.
            If all variables have the same bounds, a single pair of values can
            be specified. Valid formats are:
            * a sequence with 2 scalar values;
            * a sequence with a single element containing 2 scalar values.
            If all variables have a lower bound of 0 and no upper bound, the bounds
            parameter can be omitted (or given as None).
        x0 : 1D array, optional
            Guess values of the decision variables, which will be refined by
            the optimization algorithm. This argument is currently used only by the
            'revised simplex' method, and can only be used if `x0` represents a
            basic feasible solution.

    Returns
    -------
    ```
    """
    # 返回问题的线性目标函数系数、上界约束、等式约束和简单边界，以首选格式返回
    return {
        'c': np.asarray(lp.c),
        'A_ub': np.asarray(lp.A_ub) if lp.A_ub is not None else None,
        'b_ub': _format_b_constraints(lp.b_ub) if lp.b_ub is not None else None,
        'A_eq': np.asarray(lp.A_eq) if lp.A_eq is not None else None,
        'b_eq': np.asarray(lp.b_eq) if lp.b_eq is not None else None,
        'bounds': np.asarray(lp.bounds) if lp.bounds is not None else None,
        'x0': np.asarray(lp.x0) if lp.x0 is not None else None
    }
    c, A_ub, b_ub, A_eq, b_eq, bounds, x0, integrality = lp
    # 将 lp 解构为各个优化问题参数：目标函数系数 c，不等式约束矩阵 A_ub，不等式约束向量 b_ub，
    # 等式约束矩阵 A_eq，等式约束向量 b_eq，变量边界 bounds，初始猜测值 x0，整数约束标志 integrality。

    if c is None:
        raise TypeError
    # 如果目标函数系数 c 为空，抛出 TypeError 异常。

    try:
        c = np.array(c, dtype=np.float64, copy=True).squeeze()
    except ValueError as e:
        raise TypeError(
            "Invalid input for linprog: c must be a 1-D array of numerical "
            "coefficients") from e
    else:
        # 如果 c 是单个值，将其转换为 1-D 数组。
        if c.size == 1:
            c = c.reshape(-1)

        n_x = len(c)
        # 检查 c 的维度和元素性质
        if n_x == 0 or len(c.shape) != 1:
            raise ValueError(
                "Invalid input for linprog: c must be a 1-D array and must "
                "not have more than one non-singleton dimension")
        if not np.isfinite(c).all():
            raise ValueError(
                "Invalid input for linprog: c must not contain values "
                "inf, nan, or None")

    sparse_lhs = sps.issparse(A_eq) or sps.issparse(A_ub)
    # 检查 A_eq 或 A_ub 是否为稀疏矩阵

    try:
        A_ub = _format_A_constraints(A_ub, n_x, sparse_lhs=sparse_lhs)
    except ValueError as e:
        raise TypeError(
            "Invalid input for linprog: A_ub must be a 2-D array "
            "of numerical values") from e
    # 尝试将 A_ub 格式化为正确的约束形式，如果失败则抛出 TypeError 异常。
    # 否则，处理不等式约束的情况
    else:
        # 获取不等式约束矩阵的行数
        n_ub = A_ub.shape[0]
        # 检查A_ub的维度是否为二维且列数与决策变量数量相等，否则抛出数值错误
        if len(A_ub.shape) != 2 or A_ub.shape[1] != n_x:
            raise ValueError(
                "Invalid input for linprog: A_ub must have exactly two "
                "dimensions, and the number of columns in A_ub must be "
                "equal to the size of c")
        # 检查A_ub是否为稀疏矩阵且其数据是否全为有限值，或者A_ub不为稀疏矩阵且其数据是否全为有限值，否则抛出数值错误
        if (sps.issparse(A_ub) and not np.isfinite(A_ub.data).all()
                or not sps.issparse(A_ub) and not np.isfinite(A_ub).all()):
            raise ValueError(
                "Invalid input for linprog: A_ub must not contain values "
                "inf, nan, or None")

    try:
        # 格式化不等式约束的右侧向量b_ub
        b_ub = _format_b_constraints(b_ub)
    except ValueError as e:
        # 如果格式化过程中出错，则抛出类型错误，并提供详细错误信息
        raise TypeError(
            "Invalid input for linprog: b_ub must be a 1-D array of "
            "numerical values, each representing the upper bound of an "
            "inequality constraint (row) in A_ub") from e
    else:
        # 检查b_ub的形状是否为(n_ub,)，即一维数组，并与不等式约束矩阵A_ub的行数相等，否则抛出数值错误
        if b_ub.shape != (n_ub,):
            raise ValueError(
                "Invalid input for linprog: b_ub must be a 1-D array; b_ub "
                "must not have more than one non-singleton dimension and "
                "the number of rows in A_ub must equal the number of values "
                "in b_ub")
        # 检查b_ub是否不包含无穷大、NaN或None值，否则抛出数值错误
        if not np.isfinite(b_ub).all():
            raise ValueError(
                "Invalid input for linprog: b_ub must not contain values "
                "inf, nan, or None")

    try:
        # 格式化等式约束矩阵A_eq，确保其符合要求
        A_eq = _format_A_constraints(A_eq, n_x, sparse_lhs=sparse_lhs)
    except ValueError as e:
        # 如果格式化过程中出错，则抛出类型错误，并提供详细错误信息
        raise TypeError(
            "Invalid input for linprog: A_eq must be a 2-D array "
            "of numerical values") from e
    else:
        # 获取等式约束矩阵A_eq的行数
        n_eq = A_eq.shape[0]
        # 检查A_eq的维度是否为二维且列数与决策变量数量相等，否则抛出数值错误
        if len(A_eq.shape) != 2 or A_eq.shape[1] != n_x:
            raise ValueError(
                "Invalid input for linprog: A_eq must have exactly two "
                "dimensions, and the number of columns in A_eq must be "
                "equal to the size of c")

        # 检查A_eq是否为稀疏矩阵且其数据是否全为有限值，或者A_eq不为稀疏矩阵且其数据是否全为有限值，否则抛出数值错误
        if (sps.issparse(A_eq) and not np.isfinite(A_eq.data).all()
                or not sps.issparse(A_eq) and not np.isfinite(A_eq).all()):
            raise ValueError(
                "Invalid input for linprog: A_eq must not contain values "
                "inf, nan, or None")

    try:
        # 格式化等式约束的右侧向量b_eq，确保其符合要求
        b_eq = _format_b_constraints(b_eq)
    except ValueError as e:
        # 如果格式化过程中出错，则抛出类型错误，并提供详细错误信息
        raise TypeError(
            "Invalid input for linprog: b_eq must be a dense, 1-D array of "
            "numerical values, each representing the right hand side of an "
            "equality constraint (row) in A_eq") from e
    # 否则，检查 b_eq 的形状是否为 (n_eq,)
    if b_eq.shape != (n_eq,):
        # 如果不是，抛出 ValueError 异常，说明 b_eq 必须是一维数组；
        # A_eq 的行数必须等于 b_eq 中的值数目
        raise ValueError(
            "Invalid input for linprog: b_eq must be a 1-D array; b_eq "
            "must not have more than one non-singleton dimension and "
            "the number of rows in A_eq must equal the number of values "
            "in b_eq")

    # 检查 b_eq 中是否包含无限大、NaN 或 None 的值
    if not np.isfinite(b_eq).all():
        # 如果有，抛出 ValueError 异常，说明 b_eq 不能包含无限大、NaN 或 None 的值
        raise ValueError(
            "Invalid input for linprog: b_eq must not contain values "
            "inf, nan, or None")

    # x0 提供了一个可选的求解器的起始解。如果 x0 是 None，则跳过检查，初始解将自动生成。
    if x0 is not None:
        try:
            # 将 x0 转换为浮点数类型的一维数组
            x0 = np.array(x0, dtype=float, copy=True).squeeze()
        except ValueError as e:
            # 如果转换失败，抛出 TypeError 异常，说明 x0 必须是数值系数的一维数组
            raise TypeError(
                "Invalid input for linprog: x0 must be a 1-D array of "
                "numerical coefficients") from e
        # 如果 x0 是标量，则将其重塑为一维数组
        if x0.ndim == 0:
            x0 = x0.reshape(-1)
        # 检查 x0 的长度是否为零或者是否为一维数组
        if len(x0) == 0 or x0.ndim != 1:
            # 如果不是，抛出 ValueError 异常，说明 x0 应该是一维数组，且不能有多于一个非单例维度
            raise ValueError(
                "Invalid input for linprog: x0 should be a 1-D array; it "
                "must not have more than one non-singleton dimension")
        # 检查 x0 和 c 的元素数是否相同
        if not x0.size == c.size:
            # 如果不相同，抛出 ValueError 异常，说明 x0 和 c 应该包含相同数量的元素
            raise ValueError(
                "Invalid input for linprog: x0 and c should contain the "
                "same number of elements")
        # 检查 x0 中是否包含无限大、NaN 或 None 的值
        if not np.isfinite(x0).all():
            # 如果有，抛出 ValueError 异常，说明 x0 不能包含无限大、NaN 或 None 的值
            raise ValueError(
                "Invalid input for linprog: x0 must not contain values "
                "inf, nan, or None")

    # Bounds 可以是以下格式之一：
    # (1) 一个二维数组或序列，形状为 N x 2
    # (2) 包含两个标量的一维或二维序列或数组
    # (3) None（或空序列或数组）
    # 未指定的边界可以用 None 或 (-)np.inf 表示。
    # 所有格式都将转换为 N x 2 的 np.array，未指定边界的地方用 (-)np.inf 表示。

    # 准备一个干净的边界数组
    bounds_clean = np.zeros((n_x, 2), dtype=float)

    # 将 bounds 转换为 numpy 数组。
    # 如果 bounds 是 None 或空序列或数组，或者是空的二维数组，统一将其表示为 (0, np.inf)
    try:
        bounds_conv = np.atleast_2d(np.array(bounds, dtype=float))
    except ValueError as e:
        # 如果转换失败，抛出 ValueError 异常，说明无法解释 bounds，检查值和维度
        raise ValueError(
            "Invalid input for linprog: unable to interpret bounds, "
            "check values and dimensions: " + e.args[0]) from e
    except TypeError as e:
        # 如果转换失败，抛出 TypeError 异常，说明无法解释 bounds，检查值和维度
        raise TypeError(
            "Invalid input for linprog: unable to interpret bounds, "
            "check values and dimensions: " + e.args[0]) from e

    # 检查边界选项
    bsh = bounds_conv.shape
    # 如果边界数组 bsh 的长度大于2，则抛出 ValueError 异常，说明不处理多维边界输入
    if len(bsh) > 2:
        raise ValueError(
            "Invalid input for linprog: provide a 2-D array for bounds, "
            f"not a {len(bsh):d}-D array.")
    # 如果 bsh 是一个二维数组，并且每个维度都等于 (n_x, 2)
    elif np.all(bsh == (n_x, 2)):
        # 直接使用已转换的边界 bounds_conv
        bounds_clean = bounds_conv
    # 如果 bsh 是一个二维数组，并且等于 (2, 1) 或者 (1, 2)
    elif (np.all(bsh == (2, 1)) or np.all(bsh == (1, 2))):
        # 将边界数组展平成一维数组
        bounds_flat = bounds_conv.flatten()
        # 将展平后的数组的第一个元素赋给 bounds_clean 的第一列
        bounds_clean[:, 0] = bounds_flat[0]
        # 将展平后的数组的第二个元素赋给 bounds_clean 的第二列
        bounds_clean[:, 1] = bounds_flat[1]
    # 如果 bsh 是一个二维数组，并且每个维度分别为 (2, n_x)
    elif np.all(bsh == (2, n_x)):
        # 抛出 ValueError 异常，提示不接受 2 x N 的数组作为边界输入
        raise ValueError(
            f"Invalid input for linprog: provide a {n_x:d} x 2 array for bounds, "
            f"not a 2 x {n_x:d} array.")
    else:
        # 其他情况下，抛出 ValueError 异常，说明无法解释具有此维度元组的边界
        raise ValueError(
            "Invalid input for linprog: unable to interpret bounds with this "
            f"dimension tuple: {bsh}.")

    # 上述过程可能会创建包含 NaN 的值，需要将这些 NaN 转换为特定的边界值
    # 将第一列中的 NaN 转换为 -np.inf，第二列中的 NaN 转换为 np.inf
    i_none = np.isnan(bounds_clean[:, 0])
    bounds_clean[i_none, 0] = -np.inf
    i_none = np.isnan(bounds_clean[:, 1])
    bounds_clean[i_none, 1] = np.inf

    # 返回一个 _LPProblem 对象，使用传入的参数初始化
    return _LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds_clean, x0, integrality)
# 给定线性规划问题的预处理函数，用于识别明显的不可行性、冗余性和无界性，尽可能收紧变量边界，并消除固定变量。
def _presolve(lp, rr, rr_method, tol=1e-9):
    """
    Given inputs for a linear programming problem in preferred format,
    presolve the problem: identify trivial infeasibilities, redundancies,
    and unboundedness, tighten bounds where possible, and eliminate fixed
    variables.

    Parameters
    ----------
    lp : A `scipy.optimize._linprog_util._LPProblem` consisting of the following fields:

        c : 1D array
            The coefficients of the linear objective function to be minimized.
        A_ub : 2D array, optional
            The inequality constraint matrix. Each row of ``A_ub`` specifies the
            coefficients of a linear inequality constraint on ``x``.
        b_ub : 1D array, optional
            The inequality constraint vector. Each element represents an
            upper bound on the corresponding value of ``A_ub @ x``.
        A_eq : 2D array, optional
            The equality constraint matrix. Each row of ``A_eq`` specifies the
            coefficients of a linear equality constraint on ``x``.
        b_eq : 1D array, optional
            The equality constraint vector. Each element of ``A_eq @ x`` must equal
            the corresponding element of ``b_eq``.
        bounds : 2D array
            The bounds of ``x``, as ``min`` and ``max`` pairs, one for each of the N
            elements of ``x``. The N x 2 array contains lower bounds in the first
            column and upper bounds in the 2nd. Unbounded variables have lower
            bound -np.inf and/or upper bound np.inf.
        x0 : 1D array, optional
            Guess values of the decision variables, which will be refined by
            the optimization algorithm. This argument is currently used only by the
            'revised simplex' method, and can only be used if `x0` represents a
            basic feasible solution.

    rr : bool
        If ``True`` attempts to eliminate any redundant rows in ``A_eq``.
        Set False if ``A_eq`` is known to be of full row rank, or if you are
        looking for a potential speedup (at the expense of reliability).
    rr_method : string
        Method used to identify and remove redundant rows from the
        equality constraint matrix after presolve.
    tol : float
        The tolerance which determines when a solution is "close enough" to
        zero in Phase 1 to be considered a basic feasible solution or close
        enough to positive to serve as an optimal solution.

    Returns
    -------
    """
    # 从注释 [5] 的 Andersen 和 Andersen 的思想中得到灵感
    # 然而，与参考文献不同的是，这些操作在将问题转换为标准形式之前执行
    # 这样做有几个优点：
    # * 人工变量尚未添加，因此矩阵较小
    # * 界限尚未转换为约束条件（最好在预处理后执行，因为预处理可能会调整简单界限）
    # * 存在许多可以改进的地方，即：
    # * 实现来自[5]的其余检查
    # * 循环预处理直到不再进行额外更改
    # * 在冗余移除[2]中实现额外的效率改进

    c, A_ub, b_ub, A_eq, b_eq, bounds, x0, _ = lp

    revstack = []               # 记录从问题中消除的变量
    # 如果消除变量，则成本函数中的常数项可能会被添加
    c0 = 0
    complete = False            # 如果检测到问题不可行/无界，则complete为True
    x = np.zeros(c.shape)       # 如果在预处理中完成，这是解向量

    status = 0                  # 除非另有确定，否则所有均正常
    message = ""

    # 复制下界和上界以防止反馈
    lb = bounds[:, 0].copy()
    ub = bounds[:, 1].copy()

    m_eq, n = A_eq.shape
    m_ub, n = A_ub.shape

    if (rr_method is not None
            and rr_method.lower() not in {"svd", "pivot", "id"}):
        message = ("'" + str(rr_method) + "' is not a valid option "
                   "for redundancy removal. Valid options are 'SVD', "
                   "'pivot', and 'ID'.")
        raise ValueError(message)

    if sps.issparse(A_eq):
        A_eq = A_eq.tocsr()
        A_ub = A_ub.tocsr()

        def where(A):
            return A.nonzero()

        vstack = sps.vstack
    else:
        where = np.where
        vstack = np.vstack

    # 上界大于下界
    if np.any(ub < lb) or np.any(lb == np.inf) or np.any(ub == -np.inf):
        status = 2
        message = ("The problem is (trivially) infeasible since one "
                   "or more upper bounds are smaller than the corresponding "
                   "lower bounds, a lower bound is np.inf or an upper bound "
                   "is -np.inf.")
        complete = True
        return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0),
                c0, x, revstack, complete, status, message)

    # 等式约束中的零行
    zero_row = np.array(np.sum(A_eq != 0, axis=1) == 0).flatten()
    # 检查是否存在等式约束中的零行
    if np.any(zero_row):
        # 如果存在零行，并且存在不为零的等式约束条件
        if np.any(
            np.logical_and(
                zero_row,
                np.abs(b_eq) > tol)):  # test_zero_row_1
            # 设置问题状态为2，表示问题（显然）无法解决，因为等式约束矩阵中有一行全为零，
            # 而对应的约束值不为零
            status = 2
            # 设置消息，说明问题无法解决的原因
            message = ("The problem is (trivially) infeasible due to a row "
                       "of zeros in the equality constraint matrix with a "
                       "nonzero corresponding constraint value.")
            # 设置完成标志为True，表示问题已经处理完毕
            complete = True
            # 返回一个新的_LPProblem对象和相关变量，以及状态和消息
            return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0),
                    c0, x, revstack, complete, status, message)
        else:  # test_zero_row_2
            # 如果等式约束条件为零，可以完全删除这个方程
            A_eq = A_eq[np.logical_not(zero_row), :]
            b_eq = b_eq[np.logical_not(zero_row)]

    # 检查是否存在不等式约束中的零行
    zero_row = np.array(np.sum(A_ub != 0, axis=1) == 0).flatten()
    if np.any(zero_row):
        # 如果存在零行，并且不等式约束条件小于-tol
        if np.any(np.logical_and(zero_row, b_ub < -tol)):  # test_zero_row_1
            # 设置问题状态为2，表示问题（显然）无法解决，因为不等式约束矩阵中有一行全为零，
            # 而对应的约束值小于零
            status = 2
            # 设置消息，说明问题无法解决的原因
            message = ("The problem is (trivially) infeasible due to a row "
                       "of zeros in the equality constraint matrix with a "
                       "nonzero corresponding  constraint value.")
            # 设置完成标志为True，表示问题已经处理完毕
            complete = True
            # 返回一个新的_LPProblem对象和相关变量，以及状态和消息
            return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0),
                    c0, x, revstack, complete, status, message)
        else:  # test_zero_row_2
            # 如果不等式约束条件为非负数，可以完全删除这个约束
            A_ub = A_ub[np.logical_not(zero_row), :]
            b_ub = b_ub[np.logical_not(zero_row)]

    # 将等式约束和不等式约束垂直堆叠成新的矩阵A
    A = vstack((A_eq, A_ub))
    # 如果 A 的行数大于 0，则进行以下操作
    if A.shape[0] > 0:
        # 找出 A 中全为零的列，并转换为布尔数组
        zero_col = np.array(np.sum(A != 0, axis=0) == 0).flatten()
        # 当目标系数 c 小于 0 时，将对应的 x 设置为上界 ub
        x[np.logical_and(zero_col, c < 0)] = ub[
            np.logical_and(zero_col, c < 0)]
        # 当目标系数 c 大于 0 时，将对应的 x 设置为下界 lb
        x[np.logical_and(zero_col, c > 0)] = lb[
            np.logical_and(zero_col, c > 0)]
        # 如果存在无穷大的 x，则问题状态为 3，返回相关信息并终止
        if np.any(np.isinf(x)):  # 如果有无界的变量没有界限
            status = 3
            message = ("If feasible, the problem is (trivially) unbounded "
                       "due  to a zero column in the constraint matrices. If "
                       "you wish to check whether the problem is infeasible, "
                       "turn presolve off.")
            complete = True
            return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0),
                    c0, x, revstack, complete, status, message)
        # 当目标系数 c 小于 0 时，调整下界 lb 为上界 ub
        lb[np.logical_and(zero_col, c < 0)] = ub[
            np.logical_and(zero_col, c < 0)]
        # 当目标系数 c 大于 0 时，调整上界 ub 为下界 lb
        ub[np.logical_and(zero_col, c > 0)] = lb[
            np.logical_and(zero_col, c > 0)]

    # 在等式约束中存在只有一个非零元素的行（singleton row）
    # 这会固定一个变量并移除对应的约束
    singleton_row = np.array(np.sum(A_eq != 0, axis=1) == 1).flatten()
    rows = np.where(singleton_row)[0]
    cols = np.where(A_eq[rows, :])[1]
    if len(rows) > 0:
        for row, col in zip(rows, cols):
            val = b_eq[row] / A_eq[row, col]
            if not lb[col] - tol <= val <= ub[col] + tol:
                # 如果固定的值不在界限内，则问题状态为 2，返回相关信息并终止
                status = 2
                message = ("The problem is (trivially) infeasible because a "
                           "singleton row in the equality constraints is "
                           "inconsistent with the bounds.")
                complete = True
                return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0),
                        c0, x, revstack, complete, status, message)
            else:
                # 设置该变量的上下界为固定的值，变量将在后续移除
                lb[col] = val
                ub[col] = val
        # 移除已处理的 singleton row
        A_eq = A_eq[np.logical_not(singleton_row), :]
        b_eq = b_eq[np.logical_not(singleton_row)]

    # 在不等式约束中存在只有一个非零元素的行（singleton row）
    # 这表示一个简单的边界条件，对应的约束可以被移除
    singleton_row = np.array(np.sum(A_ub != 0, axis=1) == 1).flatten()
    cols = np.where(A_ub[singleton_row, :])[1]
    rows = np.where(singleton_row)[0]
    # 如果存在约束条件
    if len(rows) > 0:
        # 遍历行和对应的列
        for row, col in zip(rows, cols):
            # 计算变量的值
            val = b_ub[row] / A_ub[row, col]
            # 如果是上界
            if A_ub[row, col] > 0:
                # 如果计算得到的值小于下界减去容差，则问题无解
                if val < lb[col] - tol:
                    complete = True
                # 否则如果计算得到的值小于上界，则更新上界
                elif val < ub[col]:
                    ub[col] = val
            else:
                # 如果是下界
                # 如果计算得到的值大于上界加上容差，则问题无解
                if val > ub[col] + tol:
                    complete = True
                # 否则如果计算得到的值大于下界，则更新下界
                elif val > lb[col]:
                    lb[col] = val
            # 如果问题无解，则设置状态和消息并返回
            if complete:
                status = 2
                message = ("The problem is (trivially) infeasible because a "
                           "singleton row in the upper bound constraints is "
                           "inconsistent with the bounds.")
                return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0),
                        c0, x, revstack, complete, status, message)
        # 删除单例行后更新约束条件
        A_ub = A_ub[np.logical_not(singleton_row), :]
        b_ub = b_ub[np.logical_not(singleton_row)]

    # 如果所有变量的上下界相同，则可以移除变量
    i_f = np.abs(lb - ub) < tol   # "fixed"变量的索引
    i_nf = np.logical_not(i_f)    # "not fixed"变量的索引

    # 检查边界相等但不可行的情况
    if np.all(i_f):  # 如果边界定义了解，检查其一致性
        residual = b_eq - A_eq.dot(lb)
        slack = b_ub - A_ub.dot(lb)
        # 如果存在松弛变量小于零或者不是所有约束都很接近零
        if ((A_ub.size > 0 and np.any(slack < 0)) or
                (A_eq.size > 0 and not np.allclose(residual, 0))):
            status = 2
            message = ("The problem is (trivially) infeasible because the "
                       "bounds fix all variables to values inconsistent with "
                       "the constraints")
            complete = True
            return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0),
                    c0, x, revstack, complete, status, message)

    # 复制上下界
    ub_mod = ub
    lb_mod = lb
    # 如果存在任何非零的布尔索引 i_f
    if np.any(i_f):
        # 计算目标函数常数项的增量
        c0 += c[i_f].dot(lb[i_f])
        # 调整等式约束右侧向量
        b_eq = b_eq - A_eq[:, i_f].dot(lb[i_f])
        # 调整不等式约束右侧向量
        b_ub = b_ub - A_ub[:, i_f].dot(lb[i_f])
        # 更新目标系数向量为仅包含非零索引部分
        c = c[i_nf]
        # 保留用户给定的初始猜测值 x0，与预处理后的解 x 分开
        x_undo = lb[i_f]  # 注意 x[i_f] 全部是零
        x = x[i_nf]
        # 如果存在用户给定的初始猜测值 x0，则也要更新为仅包含非零索引部分
        if x0 is not None:
            x0 = x0[i_nf]
        # 更新等式约束矩阵 A_eq 为仅包含非零索引部分的子集
        A_eq = A_eq[:, i_nf]
        # 更新不等式约束矩阵 A_ub 为仅包含非零索引部分的子集
        A_ub = A_ub[:, i_nf]
        # 更新修改后的下界
        lb_mod = lb[i_nf]
        # 更新修改后的上界
        ub_mod = ub[i_nf]

        # 定义一个函数 rev，用于恢复 x_mod，将 x_undo 插入到 x_mod 中指定的位置
        def rev(x_mod):
            # 将 x_undo 插入到 x_mod 的指定位置以恢复 x 的原始状态
            i = np.flatnonzero(i_f)
            N = len(i)  # 需要恢复的变量数目
            index_offset = np.arange(N)
            insert_indices = i - index_offset
            x_rev = np.insert(x_mod.astype(float), insert_indices, x_undo)
            return x_rev

        # 将 rev 函数添加到 revstack 中，以便后续使用
        revstack.append(rev)

    # 如果没有约束条件，说明问题是平凡的
    if A_eq.size == 0 and A_ub.size == 0:
        # 清空等式约束右侧向量
        b_eq = np.array([])
        # 清空不等式约束右侧向量
        b_ub = np.array([])
        # 如果目标系数向量 c 也是空的
        if c.size == 0:
            status = 0
            message = ("The solution was determined in presolve as there are "
                       "no non-trivial constraints.")
        # 如果存在非零目标系数并且边界不是有限的情况
        elif (np.any(np.logical_and(c < 0, ub_mod == np.inf)) or
              np.any(np.logical_and(c > 0, lb_mod == -np.inf))):
            status = 3
            message = ("The problem is (trivially) unbounded "
                       "because there are no non-trivial constraints and "
                       "a) at least one decision variable is unbounded "
                       "above and its corresponding cost is negative, or "
                       "b) at least one decision variable is unbounded below "
                       "and its corresponding cost is positive. ")
        else:
            status = 0
            message = ("The solution was determined in presolve as there are "
                       "no non-trivial constraints.")
        # 表示已经完成预处理
        complete = True
        # 对于目标系数为负的情况，将 x 设置为上界
        x[c < 0] = ub_mod[c < 0]
        # 对于目标系数为正的情况，将 x 设置为下界
        x[c > 0] = lb_mod[c > 0]
        # 对于目标系数为零的情况，将 x 设置为有限的边界或零
        x_zero_c = ub_mod[c == 0]
        x_zero_c[np.isinf(x_zero_c)] = ub_mod[c == 0][np.isinf(x_zero_c)]
        x_zero_c[np.isinf(x_zero_c)] = 0
        x[c == 0] = x_zero_c
        # 如果这不是预处理的最后一步，应将边界转换回数组并在此返回
    # 将修改后的 lb_mod 和 ub_mod 转换回 N x 2 的边界数组
    bounds = np.hstack((lb_mod[:, np.newaxis], ub_mod[:, np.newaxis]))

    # 从等式约束中移除冗余（线性相关）的行
    n_rows_A = A_eq.shape[0]
    redundancy_warning = ("A_eq does not appear to be of full row rank. To "
                          "improve performance, check the problem formulation "
                          "for redundant equality constraints.")

    # 如果 A_eq 是稀疏矩阵
    if (sps.issparse(A_eq)):
        if rr and A_eq.size > 0:  # TODO: 是否可以进行快速的稀疏矩阵秩检查？
            # 使用稀疏矩阵的主元消去法移除冗余行
            rr_res = _remove_redundancy_pivot_sparse(A_eq, b_eq)
            A_eq, b_eq, status, message = rr_res
            # 如果移除了行，则发出警告
            if A_eq.shape[0] < n_rows_A:
                warn(redundancy_warning, OptimizeWarning, stacklevel=1)
            # 如果状态不为 0，标记问题为完整
            if status != 0:
                complete = True
        # 返回一个元组，包含 LP 问题的实例及其它相关信息
        return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0),
                c0, x, revstack, complete, status, message)

    # 下面的代码尝试根据性能判断使用哪种冗余移除算法，这是一个预测
    small_nullspace = 5
    if rr and A_eq.size > 0:
        try:  # TODO: 使用第一个奇异值分解结果来优化 _remove_redundancy_svd
            # 使用 numpy 的矩阵秩计算 A_eq 的秩
            rank = np.linalg.matrix_rank(A_eq)
        # 如果出现异常，fallback 到 _remove_redundancy_pivot_dense
        except Exception:
            rank = 0
    # 如果存在冗余性警告（rr为True），且约束矩阵A_eq非空且行秩小于行数
    if rr and A_eq.size > 0 and rank < A_eq.shape[0]:
        # 发出冗余性警告
        warn(redundancy_warning, OptimizeWarning, stacklevel=3)
        # 计算行空间的维数
        dim_row_nullspace = A_eq.shape[0] - rank
        # 如果未指定冗余性移除方法
        if rr_method is None:
            # 如果行空间维数小于等于指定的小空间阈值small_nullspace
            if dim_row_nullspace <= small_nullspace:
                # 使用奇异值分解方法移除冗余性
                rr_res = _remove_redundancy_svd(A_eq, b_eq)
                A_eq, b_eq, status, message = rr_res
            # 如果行空间维数大于小空间阈值或者状态码为4
            if dim_row_nullspace > small_nullspace or status == 4:
                # 使用稠密矩阵主元法移除冗余性
                rr_res = _remove_redundancy_pivot_dense(A_eq, b_eq)
                A_eq, b_eq, status, message = rr_res
        else:
            # 如果指定了冗余性移除方法
            rr_method = rr_method.lower()
            # 根据指定的方法类型进行冗余性移除
            if rr_method == "svd":
                rr_res = _remove_redundancy_svd(A_eq, b_eq)
                A_eq, b_eq, status, message = rr_res
            elif rr_method == "pivot":
                rr_res = _remove_redundancy_pivot_dense(A_eq, b_eq)
                A_eq, b_eq, status, message = rr_res
            elif rr_method == "id":
                rr_res = _remove_redundancy_id(A_eq, b_eq, rank)
                A_eq, b_eq, status, message = rr_res
            else:  # 如果不应该到达这里；选项有效性在上面已检查
                pass
        # 如果处理后的约束矩阵行数小于初始秩
        if A_eq.shape[0] < rank:
            # 提示由于数值问题无法自动移除冗余的等式约束
            message = ("Due to numerical issues, redundant equality "
                       "constraints could not be removed automatically. "
                       "Try providing your constraint matrices as sparse "
                       "matrices to activate sparse presolve, try turning "
                       "off redundancy removal, or try turning off presolve "
                       "altogether.")
            # 状态码设为4表示问题
            status = 4
        # 如果状态码不为0，表示问题不完全解决
        if status != 0:
            # 标记问题已完全处理
            complete = True
    # 返回线性规划问题对象，包括目标系数向量c，上界约束矩阵A_ub，上界向量b_ub，
    # 等式约束矩阵A_eq，等式约束向量b_eq，变量边界bounds，初始解x0
    # 以及一些额外的信息：c0，x，revstack，是否完全解决的标记complete，状态码status，信息message
    return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0),
            c0, x, revstack, complete, status, message)
def _parse_linprog(lp, options, meth):
    """
    Parse the provided linear programming problem

    ``_parse_linprog`` employs two main steps ``_check_sparse_inputs`` and
    ``_clean_inputs``. ``_check_sparse_inputs`` checks for sparsity in the
    provided constraints (``A_ub`` and ``A_eq) and if these match the provided
    sparsity optional values.

    ``_clean inputs`` checks of the provided inputs. If no violations are
    identified the objective vector, upper bound constraints, equality
    constraints, and simple bounds are returned in the expected format.

    Parameters
    ----------
    lp : A `scipy.optimize._linprog_util._LPProblem` consisting of the following fields:

        c : 1D array
            The coefficients of the linear objective function to be minimized.
        A_ub : 2D array, optional
            The inequality constraint matrix. Each row of ``A_ub`` specifies the
            coefficients of a linear inequality constraint on ``x``.
        b_ub : 1D array, optional
            The inequality constraint vector. Each element represents an
            upper bound on the corresponding value of ``A_ub @ x``.
        A_eq : 2D array, optional
            The equality constraint matrix. Each row of ``A_eq`` specifies the
            coefficients of a linear equality constraint on ``x``.
        b_eq : 1D array, optional
            The equality constraint vector. Each element of ``A_eq @ x`` must equal
            the corresponding element of ``b_eq``.
        bounds : various valid formats, optional
            The bounds of ``x``, as ``min`` and ``max`` pairs.
            If bounds are specified for all N variables separately, valid formats are:
            * a 2D array (2 x N or N x 2);
            * a sequence of N sequences, each with 2 values.
            If all variables have the same bounds, a single pair of values can
            be specified. Valid formats are:
            * a sequence with 2 scalar values;
            * a sequence with a single element containing 2 scalar values.
            If all variables have a lower bound of 0 and no upper bound, the bounds
            parameter can be omitted (or given as None).
        x0 : 1D array, optional
            Guess values of the decision variables, which will be refined by
            the optimization algorithm. This argument is currently used only by the
            'revised simplex' method, and can only be used if `x0` represents a
            basic feasible solution.

    options : dict
        A dictionary of solver options. All methods accept the following
        generic options:

            maxiter : int
                Maximum number of iterations to perform.
            disp : bool
                Set to True to print convergence messages.

        For method-specific options, see :func:`show_options('linprog')`.

    Returns
    -------
    """

    # 在此函数中，首先调用_check_sparse_inputs和_clean_inputs函数，解析线性规划问题并进行检查和清理
    # 返回解析后的线性规划问题的相关数据
    # 如果 options 参数为 None，则设置为空字典
    if options is None:
        options = {}

    # 复制 options 字典，以便进行进一步的处理
    solver_options = {k: v for k, v in options.items()}
    
    # 调用 _check_sparse_inputs 函数，对 solver_options、lp.A_ub 和 lp.A_eq 进行检查和处理
    solver_options, A_ub, A_eq = _check_sparse_inputs(solver_options, meth,
                                                      lp.A_ub, lp.A_eq)
    
    # 调用 _clean_inputs 函数，处理 lp 中的 A_ub 和 A_eq，并返回更新后的 lp 对象
    lp = _clean_inputs(lp._replace(A_ub=A_ub, A_eq=A_eq))
    
    # 返回处理后的 lp 对象和 solver_options 字典
    return lp, solver_options
# 定义函数_get_Abc，用于将线性规划问题转换为标准形式，添加松弛变量并进行必要的变量替换
def _get_Abc(lp, c0):
    """
    给定形式为线性规划问题的lp：

    最小化::

        c @ x

    满足以下条件::

        A_ub @ x <= b_ub
        A_eq @ x == b_eq
        lb <= x <= ub

    其中 ``lb = 0`` 而 ``ub = None``，除非在 ``bounds`` 中设置。

    将问题转换为标准形式：

    最小化::

        c @ x

    满足以下条件::

        A @ x == b
        x >= 0

    通过添加松弛变量和必要的变量替换。

    参数
    ----------
    lp : `scipy.optimize._linprog_util._LPProblem` 类型，包含以下字段：

        c : 1维数组
            要最小化的线性目标函数的系数。
        A_ub : 2维数组，可选
            不等式约束矩阵。每行 ``A_ub`` 指定 ``x`` 的线性不等式约束的系数。
        b_ub : 1维数组，可选
            不等式约束向量。每个元素代表对应的 ``A_ub @ x`` 的上界。
        A_eq : 2维数组，可选
            等式约束矩阵。每行 ``A_eq`` 指定 ``x`` 的线性等式约束的系数。
        b_eq : 1维数组，可选
            等式约束向量。 ``A_eq @ x`` 的每个元素必须等于 ``b_eq`` 的对应元素。
        bounds : 2维数组
            ``x`` 的边界，第一列是下界，第二列是上界。边界可能会被预处理过程进一步缩紧。
        x0 : 1维数组，可选
            决策变量的初始猜测值，将由优化算法进一步优化。目前只有 'revised simplex' 方法使用，且只有在 `x0` 表示基本可行解时可用。

    c0 : 浮点数
        由于固定（已消除）变量而导致的目标函数中的常数项。

    返回
    -------
    A : 2维数组
        使得 ``A`` @ ``x`` 在 ``x`` 处给出等式约束的值的二维数组。
    b : 1维数组
        表示标准形式问题中每个等式约束（行）的右手边的值的一维数组。
    c : 1维数组
        要最小化的线性目标函数的系数（标准形式问题）。
    c0 : 浮点数
        由于固定（已消除）变量而导致的目标函数中的常数项（标准形式问题）。
    x0 : 1维数组
        独立变量的起始值，将由优化算法进一步优化。

    参考文献
    ----------
    .. [9] Bertsimas, Dimitris, and J. Tsitsiklis. "Introduction to linear
           programming." Athena Scientific 1 (1997): 997.

    """
    # 解构参数 lp
    c, A_ub, b_ub, A_eq, b_eq, bounds, x0, integrality = lp
    # 如果 A_eq 是稀疏矩阵，则将 sparse 标记设为 True，并将 A_eq 和 A_ub 转换为 CSR 格式稀疏矩阵
    if sps.issparse(A_eq):
        sparse = True
        A_eq = sps.csr_matrix(A_eq)
        A_ub = sps.csr_matrix(A_ub)

        # 定义用于水平拼接和垂直拼接稀疏块的函数
        def hstack(blocks):
            return sps.hstack(blocks, format="csr")

        def vstack(blocks):
            return sps.vstack(blocks, format="csr")

        # 定义零矩阵和单位矩阵生成函数
        zeros = sps.csr_matrix
        eye = sps.eye
    else:
        # 如果 A_eq 不是稀疏矩阵，则将 sparse 标记设为 False，并使用 NumPy 的函数进行拼接
        sparse = False
        hstack = np.hstack
        vstack = np.vstack
        zeros = np.zeros
        eye = np.eye

    # 由于 lbs 和 ubs 可能会被修改，这些修改会影响 bounds，因此进行复制
    bounds = np.array(bounds, copy=True)

    # 将问题修改为所有变量仅具有非负界限
    lbs = bounds[:, 0]  # 取出界限数组中的下界
    ubs = bounds[:, 1]  # 取出界限数组中的上界
    m_ub, n_ub = A_ub.shape  # 获取 A_ub 矩阵的行数和列数

    # 检查是否存在无下界或无上界的情况，并创建相应的布尔掩码
    lb_none = np.equal(lbs, -np.inf)
    ub_none = np.equal(ubs, np.inf)
    lb_some = np.logical_not(lb_none)
    ub_some = np.logical_not(ub_none)

    # 处理无下界但有上界的变量：替换 xi = -xi'，同时调整相关的 c 向量和初始解 x0
    l_nolb_someub = np.logical_and(lb_none, ub_some)
    i_nolb = np.nonzero(l_nolb_someub)[0]
    lbs[l_nolb_someub], ubs[l_nolb_someub] = (
        -ubs[l_nolb_someub], -lbs[l_nolb_someub])
    lb_none = np.equal(lbs, -np.inf)
    ub_none = np.equal(ubs, np.inf)
    lb_some = np.logical_not(lb_none)
    ub_some = np.logical_not(ub_none)
    c[i_nolb] *= -1
    if x0 is not None:
        x0[i_nolb] *= -1
    if len(i_nolb) > 0:
        if A_ub.shape[0] > 0:  # 有时需要针对稀疏数组进行这样的操作，这很奇怪
            A_ub[:, i_nolb] *= -1
        if A_eq.shape[0] > 0:
            A_eq[:, i_nolb] *= -1

    # 处理有上界的变量：添加不等式约束
    i_newub, = ub_some.nonzero()
    ub_newub = ubs[ub_some]
    n_bounds = len(i_newub)
    if n_bounds > 0:
        shape = (n_bounds, A_ub.shape[1])
        if sparse:
            idxs = (np.arange(n_bounds), i_newub)
            A_ub = vstack((A_ub, sps.csr_matrix((np.ones(n_bounds), idxs),
                                                shape=shape)))
        else:
            A_ub = vstack((A_ub, np.zeros(shape)))
            A_ub[np.arange(m_ub, A_ub.shape[0]), i_newub] = 1
        b_ub = np.concatenate((b_ub, np.zeros(n_bounds)))
        b_ub[m_ub:] = ub_newub

    # 合并 A_ub 和 A_eq 形成新的约束矩阵 A1，以及合并 b_ub 和 b_eq 形成新的约束向量 b
    A1 = vstack((A_ub, A_eq))
    b = np.concatenate((b_ub, b_eq))

    # 扩展目标函数系数向量 c 和初始解 x0，以处理无界变量的情况
    c = np.concatenate((c, np.zeros((A_ub.shape[0],))))
    if x0 is not None:
        x0 = np.concatenate((x0, np.zeros((A_ub.shape[0],))))

    # 处理无界变量：将其拆分为正负部分，并更新相关的约束矩阵 A1 和目标函数系数向量 c
    l_free = np.logical_and(lb_none, ub_none)
    i_free = np.nonzero(l_free)[0]
    n_free = len(i_free)
    c = np.concatenate((c, np.zeros(n_free)))
    if x0 is not None:
        x0 = np.concatenate((x0, np.zeros(n_free)))
    A1 = hstack((A1[:, :n_ub], -A1[:, i_free]))
    c[n_ub:n_ub+n_free] = -c[i_free]
    # 如果给定了初始点 x0
    if x0 is not None:
        # 找到自由变量中小于零的索引
        i_free_neg = x0[i_free] < 0
        # 负值的自由变量设为它们的相反数
        x0[np.arange(n_ub, A1.shape[1])[i_free_neg]] = -x0[i_free[i_free_neg]]
        # 正值的自由变量设为零
        x0[i_free[i_free_neg]] = 0

    # 添加松弛变量
    A2 = vstack([eye(A_ub.shape[0]), zeros((A_eq.shape[0], A_ub.shape[0]))])

    # 合并 A1 和 A2 形成新的约束矩阵 A
    A = hstack([A1, A2])

    # 处理下界：替换变量 xi = xi' + lb
    # 现在目标函数中有一个常数项
    i_shift = np.nonzero(lb_some)[0]
    lb_shift = lbs[lb_some].astype(float)
    c0 += np.sum(lb_shift * c[i_shift])
    # 如果稀疏模式开启
    if sparse:
        # 重新整形 b 为列向量
        b = b.reshape(-1, 1)
        # 将 A 转换为压缩列稀疏矩阵
        A = A.tocsc()
        # 调整 b，减去 A 的某些列乘以 lb_shift 的乘积
        b -= (A[:, i_shift] * sps.diags(lb_shift)).sum(axis=1)
        # 展平 b
        b = b.ravel()
    else:
        # 调整 b，减去 A 的某些列乘以 lb_shift 的乘积
        b -= (A[:, i_shift] * lb_shift).sum(axis=1)
    # 如果给定了初始点 x0
    if x0 is not None:
        # 调整 x0，减去 lb_shift
        x0[i_shift] -= lb_shift

    # 返回结果：约束矩阵 A，约束向量 b，目标函数系数向量 c，目标函数常数项 c0，初始点 x0
    return A, b, c, c0, x0
def _round_to_power_of_two(x):
    """
    Round elements of the array to the nearest power of two.
    """
    # 返回数组元素最接近的二的幂次方值
    return 2**np.around(np.log2(x))


def _autoscale(A, b, c, x0):
    """
    Scales the problem according to equilibration from [12].
    Also normalizes the right hand side vector by its maximum element.
    """
    m, n = A.shape

    # 初始化行和列的缩放因子
    C = 1
    R = 1

    if A.size > 0:
        # 计算行的最大绝对值，并进行处理
        R = np.max(np.abs(A), axis=1)
        if sps.issparse(A):
            R = R.toarray().flatten()
        R[R == 0] = 1
        R = 1/_round_to_power_of_two(R)
        # 对 A 和 b 进行缩放
        A = sps.diags(R)*A if sps.issparse(A) else A*R.reshape(m, 1)
        b = b*R

        # 计算列的最大绝对值，并进行处理
        C = np.max(np.abs(A), axis=0)
        if sps.issparse(A):
            C = C.toarray().flatten()
        C[C == 0] = 1
        C = 1/_round_to_power_of_two(C)
        # 对 A 和 c 进行缩放
        A = A*sps.diags(C) if sps.issparse(A) else A*C
        c = c*C

    # 计算 b 的最大绝对值并进行缩放
    b_scale = np.max(np.abs(b)) if b.size > 0 else 1
    if b_scale == 0:
        b_scale = 1.
    b = b/b_scale

    # 如果有初始解 x0，进行进一步缩放
    if x0 is not None:
        x0 = x0/b_scale*(1/C)
    return A, b, c, x0, C, b_scale


def _unscale(x, C, b_scale):
    """
    Converts solution to _autoscale problem -> solution to original problem.
    """
    try:
        n = len(C)
        # 如果 C 是稀疏矩阵或者标量，此处会失败，这是可以接受的
        # 这仅在原始单纯形法中需要（从不稀疏）
    except TypeError:
        n = len(x)

    # 将经过缩放的解 x 转换回原始问题的解
    return x[:n]*b_scale*C


def _display_summary(message, status, fun, iteration):
    """
    Print the termination summary of the linear program

    Parameters
    ----------
    message : str
            A string descriptor of the exit status of the optimization.
    status : int
        An integer representing the exit status of the optimization::

                0 : Optimization terminated successfully
                1 : Iteration limit reached
                2 : Problem appears to be infeasible
                3 : Problem appears to be unbounded
                4 : Serious numerical difficulties encountered

    fun : float
        Value of the objective function.
    iteration : iteration
        The number of iterations performed.
    """
    # 打印线性规划的终止摘要信息
    print(message)
    if status in (0, 1):
        print(f"         Current function value: {fun: <12.6f}")
    print(f"         Iterations: {iteration:d}")


def _postsolve(x, postsolve_args, complete=False):
    """
    Given solution x to presolved, standard form linear program x, add
    fixed variables back into the problem and undo the variable substitutions
    to get solution to original linear program. Also, calculate the objective
    function value, slack in original upper bound constraints, and residuals
    in original equality constraints.

    Parameters
    ----------
    x : 1-D array
        Solution vector to the standard-form problem.
    postsolve_args : tuple
        Data needed by _postsolve to convert the solution to the standard-form
        problem into the solution to the original problem, including:
    """
    # lp 是一个线性规划问题对象，包含以下字段：

    # c: 1D 数组
    #     线性目标函数的系数，用于最小化。
    # A_ub: 2D 数组，可选
    #     不等式约束矩阵。每行指定变量 `x` 的线性不等式约束的系数。
    # b_ub: 1D 数组，可选
    #     不等式约束向量。每个元素表示对应的 `A_ub @ x` 的上界限制。
    # A_eq: 2D 数组，可选
    #     等式约束矩阵。每行指定变量 `x` 的线性等式约束的系数。
    # b_eq: 1D 数组，可选
    #     等式约束向量。要求 `A_eq @ x` 的每个元素必须等于 `b_eq` 中对应的元素。
    # bounds: 2D 数组
    #     变量 `x` 的边界。第一列是下界，第二列是上界。这些边界可能会在预处理过程中被进一步紧缩。
    # x0: 1D 数组，可选
    #     决策变量的初始猜测值，将由优化算法进一步优化。目前只有“修正单纯形法”使用此参数，
    #     并且仅当 `x0` 表示基本可行解时才能使用。

    # revstack: 函数列表
    #     列表中的函数用于反转 `_presolve()` 的操作。
    #     函数签名为 x_org = f(x_mod)，其中 x_mod 是预处理步骤的结果，x_org 是该步骤开始时的值。
    # complete: 布尔值
    #     指示解是否在预处理中确定（如果是，则为 True）。

    # 返回：
    # x: 1D 数组
    #     原始线性规划问题的解向量。
    # fun: 浮点数
    #     原始问题的最优目标值。
    # slack: 1D 数组
    #     上界约束的松弛变量（非负），即 `b_ub - A_ub @ x`。
    # con: 1D 数组
    #     等式约束的残差（理论上为零），即 `b - A_eq @ x`。

    """
    # 注意，所有的输入都是原始未修改过的版本
    # 没有移除任何行或列

    # 从 postsolve_args 中解包参数
    c, A_ub, b_ub, A_eq, b_eq, bounds, x0, integrality = postsolve_args[0]
    revstack, C, b_scale = postsolve_args[1:]

    # 对 x 进行反缩放操作，使用 C 和 b_scale
    x = _unscale(x, C, b_scale)

    # 撤销 _get_Abc() 的变量替换
    # 如果 "complete" 为真，表示问题在预处理中已解决；这里不需要做任何操作
    n_x = bounds.shape[0]
    # 如果未完成且边界不为None，则执行以下操作（边界可能从不为None）
    if not complete and bounds is not None:
        # 初始化未限定的变量数目为0
        n_unbounded = 0
        # 遍历边界列表中的每个元素和索引
        for i, bi in enumerate(bounds):
            # 提取下界和上界
            lbi = bi[0]
            ubi = bi[1]
            # 如果下界为负无穷并且上界为正无穷
            if lbi == -np.inf and ubi == np.inf:
                # 未限定变量数目加1
                n_unbounded += 1
                # 调整x[i]，以消除已被移除的变量影响
                x[i] = x[i] - x[n_x + n_unbounded - 1]
            else:
                # 如果下界为负无穷
                if lbi == -np.inf:
                    # 调整x[i]，将上界ubi加入
                    x[i] = ubi - x[i]
                else:
                    # 否则，将下界lbi加入x[i]
                    x[i] += lbi
    
    # 移除所有剩余的人工变量
    x = x[:n_x]
    
    # 如果问题中有被移除的变量，将它们重新添加到解向量中
    # 将revstack中的函数以反向顺序应用于解向量x
    for rev in reversed(revstack):
        x = rev(x)
    
    # 计算目标函数值
    fun = x.dot(c)
    
    # 计算原始上界约束的松弛量（slack）
    slack = b_ub - A_ub.dot(x)
    
    # 计算原始等式约束的残差（con）
    con = b_eq - A_eq.dot(x)
    
    # 返回计算得到的解向量x，目标函数值fun，上界约束松弛量slack，和等式约束残差con
    return x, fun, slack, con
    # Somewhat arbitrary: 设置一个相对随意的容差阈值，用于数值比较
    tol = np.sqrt(tol) * 10

    # 如果解向量 x 为 None，则表示问题在求解过程中出现不可行或无界限情况
    if x is None:
        # 当状态为 0 时，表示使用 HiGHS Simplex Primal 求解器，未提供解决方案
        if status == 0:
            status = 4  # 将状态更新为 4，表示严重数值困难
            message = ("The solver did not provide a solution nor did it "
                       "report a failure. Please submit a bug report.")
        return status, message

    # 检查解向量 x, 目标函数值 fun, 松弛变量 slack 和约束残差 con 是否包含 NaN 值
    contains_nans = (
        np.isnan(x).any()
        or np.isnan(fun)
        or np.isnan(slack).any()
        or np.isnan(con).any()
    )

    # 如果包含 NaN 值，则解向量 x 不可行
    if contains_nans:
        is_feasible = False
    else:
        # 如果 integrality 为 None，则将其设为 0
        if integrality is None:
            integrality = 0

        # 检查解向量 x 是否在其边界范围内，并根据 integrality 的值进一步限制 x 的取值
        valid_bounds = (x >= bounds[:, 0] - tol) & (x <= bounds[:, 1] + tol)
        valid_bounds |= (integrality > 1) & np.isclose(x, 0, atol=tol)

        # 检查边界是否有效
        invalid_bounds = not np.all(valid_bounds)

        # 检查松弛变量是否有效
        invalid_slack = status != 3 and (slack < -tol).any()

        # 检查约束残差是否有效
        invalid_con = status != 3 and (np.abs(con) > tol).any()

        # 判断解向量 x 是否可行
        is_feasible = not (invalid_bounds or invalid_slack or invalid_con)
    # 如果求解器状态为0且不可行解为True，则更新状态为4，并设置错误信息
    if status == 0 and not is_feasible:
        status = 4
        message = ("The solution does not satisfy the constraints within the "
                   "required tolerance of " + f"{tol:.2E}" + ", yet "
                   "no errors were raised and there is no certificate of "
                   "infeasibility or unboundedness. Check whether "
                   "the slack and constraint residuals are acceptable; "
                   "if not, consider enabling presolve, adjusting the "
                   "tolerance option(s), and/or using a different method. "
                   "Please consider submitting a bug report.")
    
    # 如果求解器状态为2且可行解为True，则设置状态为4，并设置警告信息
    elif status == 2 and is_feasible:
        # 出现在单纯形法(phase one)完成后得到接近基本可行解的情况。后处理可以将解变为基本解，但这个解不是最优解
        status = 4
        message = ("The solution is feasible, but the solver did not report "
                   "that the solution was optimal. Please try a different "
                   "method.")

    # 返回更新后的状态和相应的消息
    return status, message
```