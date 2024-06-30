# `D:\src\scipysrc\scipy\scipy\optimize\_linprog_simplex.py`

```
"""
Simplex method for linear programming

The *simplex* method uses a traditional, full-tableau implementation of
Dantzig's simplex algorithm [1]_, [2]_ (*not* the Nelder-Mead simplex).
This algorithm is included for backwards compatibility and educational
purposes.

    .. versionadded:: 0.15.0

Warnings
--------

The simplex method may encounter numerical difficulties when pivot
values are close to the specified tolerance. If encountered try
remove any redundant constraints, change the pivot strategy to Bland's
rule or increase the tolerance value.

Alternatively, more robust methods maybe be used. See
:ref:`'interior-point' <optimize.linprog-interior-point>` and
:ref:`'revised simplex' <optimize.linprog-revised_simplex>`.

References
----------
.. [1] Dantzig, George B., Linear programming and extensions. Rand
       Corporation Research Study Princeton Univ. Press, Princeton, NJ,
       1963
.. [2] Hillier, S.H. and Lieberman, G.J. (1995), "Introduction to
       Mathematical Programming", McGraw-Hill, Chapter 4.
"""

import numpy as np
from warnings import warn
from ._optimize import OptimizeResult, OptimizeWarning, _check_unknown_options
from ._linprog_util import _postsolve


def _pivot_col(T, tol=1e-9, bland=False):
    """
    Given a linear programming simplex tableau, determine the column
    of the variable to enter the basis.

    Parameters
    ----------
    T : 2-D array
        A 2-D array representing the simplex tableau, T, corresponding to the
        linear programming problem. It should have the form:

        [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],
         [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],
         .
         .
         .
         [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],
         [c[0],   c[1], ...,   c[n_total],    0]]

        for a Phase 2 problem, or the form:

        [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],
         [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],
         .
         .
         .
         [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],
         [c[0],   c[1], ...,   c[n_total],   0],
         [c'[0],  c'[1], ...,  c'[n_total],  0]]

         for a Phase 1 problem (a problem in which a basic feasible solution is
         sought prior to maximizing the actual objective. ``T`` is modified in
         place by ``_solve_simplex``.
    tol : float
        Elements in the objective row larger than -tol will not be considered
        for pivoting. Nominally this value is zero, but numerical issues
        cause a tolerance about zero to be necessary.
    bland : bool
        If True, use Bland's rule for selection of the column (select the
        first column with a negative coefficient in the objective row,
        regardless of magnitude).

    Returns
    -------
    status: bool
        True if a suitable pivot column was found, otherwise False.
        A return of False indicates that the linear programming simplex
        algorithm is complete.
    """
    # 创建一个掩码数组，标记最后一行除了最后一个元素外小于等于 -tol 的元素位置
    ma = np.ma.masked_where(T[-1, :-1] >= -tol, T[-1, :-1], copy=False)
    # 如果掩码数组中没有被标记的元素，返回 False 和 NaN
    if ma.count() == 0:
        return False, np.nan
    # 如果使用 Bland's rule，返回 True 和第一个非掩码位置的索引
    if bland:
        # ma.mask 有时是零维的，确保至少是一维
        return True, np.nonzero(np.logical_not(np.atleast_1d(ma.mask)))[0][0]
    # 否则，返回 True 和最小值所在位置的索引
    return True, np.ma.nonzero(ma == ma.min())[0][0]
# 在线性规划单纯形表中找到进行主元操作的行
def _pivot_row(T, basis, pivcol, phase, tol=1e-9, bland=False):
    """
    Given a linear programming simplex tableau, determine the row for the
    pivot operation.

    Parameters
    ----------
    T : 2-D array
        A 2-D array representing the simplex tableau, T, corresponding to the
        linear programming problem. It should have the form:

        [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],
         [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],
         .
         .
         .
         [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],
         [c[0],   c[1], ...,   c[n_total],    0]]

        for a Phase 2 problem, or the form:

        [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],
         [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],
         .
         .
         .
         [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],
         [c[0],   c[1], ...,   c[n_total],   0],
         [c'[0],  c'[1], ...,  c'[n_total],  0]]

         for a Phase 1 problem (a Problem in which a basic feasible solution is
         sought prior to maximizing the actual objective. ``T`` is modified in
         place by ``_solve_simplex``.
    basis : array
        A list of the current basic variables.
    pivcol : int
        The index of the pivot column.
    phase : int
        The phase of the simplex algorithm (1 or 2).
    tol : float
        Elements in the pivot column smaller than tol will not be considered
        for pivoting. Nominally this value is zero, but numerical issues
        cause a tolerance about zero to be necessary.
    bland : bool
        If True, use Bland's rule for selection of the row (if more than one
        row can be used, choose the one with the lowest variable index).

    Returns
    -------
    status: bool
        True if a suitable pivot row was found, otherwise False. A return
        of False indicates that the linear programming problem is unbounded.
    row: int
        The index of the row of the pivot element. If status is False, row
        will be returned as nan.
    """
    # 根据单纯形算法的阶段确定 k 的值
    if phase == 1:
        k = 2
    else:
        k = 1
    # 创建一个掩码数组，标记小于等于 tol 的主元素
    ma = np.ma.masked_where(T[:-k, pivcol] <= tol, T[:-k, pivcol], copy=False)
    # 如果所有标记的元素都小于等于 tol，则线性规划问题无界
    if ma.count() == 0:
        return False, np.nan
    # 创建另一个掩码数组，用于计算主元素对应的约束
    mb = np.ma.masked_where(T[:-k, pivcol] <= tol, T[:-k, -1], copy=False)
    # 计算最小比值 q，即主元所对应的约束值除以主元素值
    q = mb / ma
    # 找到所有最小比值的行索引
    min_rows = np.ma.nonzero(q == q.min())[0]
    # 如果使用 Bland 规则，则选择最小的基变量索引行
    if bland:
        return True, min_rows[np.argmin(np.take(basis, min_rows))]
    # 否则选择第一个最小比值的行索引
    return True, min_rows[0]


def _apply_pivot(T, basis, pivrow, pivcol, tol=1e-9):
    """
    Pivot the simplex tableau inplace on the element given by (pivrow, pivol).
    The entering variable corresponds to the column given by pivcol forcing
    the variable basis[pivrow] to leave the basis.

    Parameters
    ----------
    T : 2-D which tableau, simplex programming, representing A
    basis : array variables. index piv follows ```python
    array for operation tolerances column; treated.
    basis: tableau, Phase 2. function ``
   ```
    # 将基变量数组中第 pivrow 行的元素更新为 pivcol，即更新基变量对应的列索引
    basis[pivrow] = pivcol

    # 获取选定的主元素值，即当前主元素的值
    pivval = T[pivrow, pivcol]

    # 将主元素所在行除以主元素的值，实现主元素归一化
    T[pivrow] = T[pivrow] / pivval

    # 遍历表格的每一行，对于非主元素行，通过线性组合消除主元素列的其他元素
    for irow in range(T.shape[0]):
        if irow != pivrow:
            T[irow] = T[irow] - T[pivrow] * T[irow, pivcol]

    # 检查选定的主元素是否接近于零，如果接近，则发出警告
    if np.isclose(pivval, tol, atol=0, rtol=1e4):
        # 构造警告消息，说明主元素操作产生的主元素值接近于指定的容差tol
        message = (
            f"The pivot operation produces a pivot value of:{pivval: .1e}, "
            "which is only slightly greater than the specified "
            f"tolerance{tol: .1e}. This may lead to issues regarding the "
            "numerical stability of the simplex method. "
            "Removing redundant constraints, changing the pivot strategy "
            "via Bland's rule or increasing the tolerance may "
            "help reduce the issue.")
        # 发出警告消息
        warn(message, OptimizeWarning, stacklevel=5)
# 使用 Simplex 方法解决标准形式的线性规划问题。线性规划问题的标准形式如下：

# 最小化:
#     c @ x

# 约束条件:
#     A @ x == b
#     x >= 0

def _solve_simplex(T, n, basis, callback, postsolve_args,
                   maxiter=1000, tol=1e-9, phase=2, bland=False, nit0=0,
                   ):
    """
    Solve a linear programming problem in "standard form" using the Simplex
    Method. Linear Programming is intended to solve the following problem form:

    Minimize::

        c @ x

    Subject to::

        A @ x == b
            x >= 0

    Parameters
    ----------
    T : 2-D array
        表示 Simplex 表的二维数组 T，对应于线性规划问题。对于 Phase 2 问题，它应该具有如下形式：

        [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],
         [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],
         .
         .
         .
         [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],
         [c[0],   c[1], ...,   c[n_total],    0]]

        对于 Phase 1 问题（在最大化实际目标之前寻找基本可行解的问题），它应该具有如下形式：

        [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],
         [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],
         .
         .
         .
         [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],
         [c[0],   c[1], ...,   c[n_total],   0],
         [c'[0],  c'[1], ...,  c'[n_total],  0]]

        在 place 修改的 Simplex 表由 _solve_simplex 修改。

    n : int
        问题中真实变量的数量。

    basis : 1-D array
        基本变量的索引数组，使得 basis[i] 包含第 i 行对应的基本变量的列。
        _solve_simplex 会在 place 修改 basis。

    callback : function
        在每个 Simplex 方法迭代步骤之后调用的回调函数。

    postsolve_args : tuple
        传递给 postsolve 函数的参数元组。

    maxiter : int, optional
        Simplex 方法的最大迭代次数，默认为 1000。

    tol : float, optional
        公差值，默认为 1e-9。

    phase : int, optional
        Simplex 方法的阶段（Phase），默认为 2。

    bland : bool, optional
        指示是否使用 Bland 规则的布尔值，默认为 False。

    nit0 : int, optional
        初始迭代次数，默认为 0。
    """
    # 函数主体部分的代码在这里开始...
    # 回调函数，可选参数
    callback : callable, optional
        # 如果提供了回调函数，将在每次迭代算法中调用
        # 回调函数必须接受一个 `scipy.optimize.OptimizeResult` 对象，包含以下字段：
        x : 1-D array
            # 当前解向量
        fun : float
            # 目标函数当前值
        success : bool
            # 只有在阶段成功完成时为 True。对于大多数迭代来说，这个值将为 False。
        slack : 1-D array
            # 松弛变量的值。每个松弛变量对应一个不等式约束。如果松弛变量为零，则表示对应的约束是活跃的。
        con : 1-D array
            # 等式约束的（名义上为零的）残差，即 `b - A_eq @ x`
        phase : int
            # 正在执行的优化阶段。在第一阶段中，寻找一个基本可行解，并且 T 有一行额外代表备选目标函数。
        status : int
            # 表示优化退出状态的整数：
            # 0: 优化成功终止
            # 1: 达到迭代限制
            # 2: 问题似乎是不可行的
            # 3: 问题似乎是无界的
            # 4: 遇到严重的数值困难
        nit : int
            # 执行的迭代次数
        message : str
            # 优化退出状态的字符串描述
    postsolve_args : tuple
        # 由 `_postsolve` 使用的数据，将解转换为标准形式问题到原始问题的解
    maxiter : int
        # 在中止优化之前执行的最大迭代次数
    tol : float
        # 决定何时解被视为“足够接近”零，以便在第一阶段被认为是基本可行解或足够接近正以作为最优解
    phase : int
        # 正在执行的优化阶段。在第一阶段中，寻找一个基本可行解，并且 T 有一行额外代表备选目标函数。
    bland : bool
        # 如果为 True，则使用 Bland's 规则选择主元素。在由于循环导致收敛失败的问题中，使用 Bland's 规则可以在牺牲较优路径的情况下提供收敛性。
    nit0 : int
        # 在两阶段问题中用来保持准确的迭代总数的初始迭代号码

    Returns
    -------
    nit : int
        # 迭代次数。在两阶段问题中用来保持准确的迭代总数。
    """
    status : int
        An integer representing the exit status of the optimization::

         0 : Optimization terminated successfully
         1 : Iteration limit reached
         2 : Problem appears to be infeasible
         3 : Problem appears to be unbounded
         4 : Serious numerical difficulties encountered
    """

    # 将初始迭代次数设为给定的初始值 nit0
    nit = nit0
    # 设置初始状态为成功
    status = 0
    # 初始化消息为空字符串
    message = ''
    # 设定是否完成优化为 False
    complete = False

    # 根据所处的阶段设定 m 的值
    if phase == 1:
        m = T.shape[1] - 2
    elif phase == 2:
        m = T.shape[1] - 1
    else:
        # 如果 phase 不是 1 或 2，则抛出数值错误
        raise ValueError("Argument 'phase' to _solve_simplex must be 1 or 2")

    # 在阶段 2 进行以下操作
    if phase == 2:
        # 检查是否仍然有人工变量在基中
        # 如果是，则检查该行的系数和对应的非人工变量列是否存在非零系数
        # 如果存在，则在这一项进行主元操作（pivot）；如果不存在，则开始阶段 2
        # 对基中的所有人工变量执行此操作
        # 参考文献: "An Introduction to Linear Programming and Game Theory"
        # by Paul R. Thie, Gerard E. Keough, 3rd Ed,
        # Chapter 3.7 Redundant Systems (pag 102)
        for pivrow in [row for row in range(basis.size)
                       if basis[row] > T.shape[1] - 2]:
            non_zero_row = [col for col in range(T.shape[1] - 1)
                            if abs(T[pivrow, col]) > tol]
            if len(non_zero_row) > 0:
                pivcol = non_zero_row[0]
                # 应用主元操作（pivot）
                _apply_pivot(T, basis, pivrow, pivcol, tol)
                nit += 1

    # 如果基的长度为 m 的长度为 0，则创建一个空的解向量
    if len(basis[:m]) == 0:
        solution = np.empty(T.shape[1] - 1, dtype=np.float64)
    else:
        # 否则创建一个足够大的解向量以容纳所有可能的变量
        solution = np.empty(max(T.shape[1] - 1, max(basis[:m]) + 1),
                            dtype=np.float64)
    # 当未完成时执行循环，直到找到最优解或达到最大迭代次数
    while not complete:
        # 查找主元列
        pivcol_found, pivcol = _pivot_col(T, tol, bland)
        # 如果未找到主元列，则设置状态为未找到解，完成标志设为True
        if not pivcol_found:
            pivcol = np.nan
            pivrow = np.nan
            status = 0
            complete = True
        else:
            # 找到主元列后，查找主元行
            pivrow_found, pivrow = _pivot_row(T, basis, pivcol, phase, tol, bland)
            # 如果未找到主元行，则设置状态为无限制解，完成标志设为True
            if not pivrow_found:
                status = 3
                complete = True

        # 如果有回调函数，则执行后处理和回调操作
        if callback is not None:
            # 初始化解向量为零
            solution[:] = 0
            # 将基向量对应的解赋值给solution向量
            solution[basis[:n]] = T[:n, -1]
            # 提取出解向量x，并执行后处理操作
            x = solution[:m]
            x, fun, slack, con = _postsolve(
                x, postsolve_args
            )
            # 构造优化结果对象
            res = OptimizeResult({
                'x': x,
                'fun': fun,
                'slack': slack,
                'con': con,
                'status': status,
                'message': message,
                'nit': nit,
                'success': status == 0 and complete,
                'phase': phase,
                'complete': complete,
                })
            # 调用回调函数，传递优化结果对象res
            callback(res)

        # 如果仍未完成，则检查是否超过最大迭代次数
        if not complete:
            if nit >= maxiter:
                # 超过最大迭代次数，设置状态为迭代次数超限，完成标志设为True
                status = 1
                complete = True
            else:
                # 应用主元法进行变换，并增加迭代次数计数器
                _apply_pivot(T, basis, pivrow, pivcol, tol)
                nit += 1

    # 返回迭代次数和最终状态
    return nit, status
# 使用两阶段单纯形法最小化线性目标函数，受线性等式和非负约束条件限制。
def _linprog_simplex(c, c0, A, b, callback, postsolve_args,
                     maxiter=1000, tol=1e-9, disp=False, bland=False,
                     **unknown_options):
    """
    Minimize a linear objective function subject to linear equality and
    non-negativity constraints using the two phase simplex method.
    Linear programming is intended to solve problems of the following form:

    Minimize::

        c @ x

    Subject to::

        A @ x == b
            x >= 0

    User-facing documentation is in _linprog_doc.py.

    Parameters
    ----------
    c : 1-D array
        Coefficients of the linear objective function to be minimized.
    c0 : float
        Constant term in objective function due to fixed (and eliminated)
        variables. (Purely for display.)
    A : 2-D array
        2-D array such that ``A @ x``, gives the values of the equality
        constraints at ``x``.
    b : 1-D array
        1-D array of values representing the right hand side of each equality
        constraint (row) in ``A``.
    callback : callable, optional
        If a callback function is provided, it will be called within each
        iteration of the algorithm. The callback function must accept a single
        `scipy.optimize.OptimizeResult` consisting of the following fields:

            x : 1-D array
                Current solution vector
            fun : float
                Current value of the objective function
            success : bool
                True when an algorithm has completed successfully.
            slack : 1-D array
                The values of the slack variables. Each slack variable
                corresponds to an inequality constraint. If the slack is zero,
                the corresponding constraint is active.
            con : 1-D array
                The (nominally zero) residuals of the equality constraints,
                that is, ``b - A_eq @ x``
            phase : int
                The phase of the algorithm being executed.
            status : int
                An integer representing the status of the optimization::

                     0 : Algorithm proceeding nominally
                     1 : Iteration limit reached
                     2 : Problem appears to be infeasible
                     3 : Problem appears to be unbounded
                     4 : Serious numerical difficulties encountered
            nit : int
                The number of iterations performed.
            message : str
                A string descriptor of the exit status of the optimization.
    postsolve_args : tuple
        Data needed by _postsolve to convert the solution to the standard-form
        problem into the solution to the original problem.

    Options
    -------
    maxiter : int
       The maximum number of iterations to perform.
    tol : float
        Termination tolerance to be used for various termination criteria.
    disp : bool
        If True, print exit status message to sys.stdout
    bland : bool
        If True, use Bland's rule for selection of entering and leaving variables.
    **unknown_options : dict
        Additional options passed directly to the underlying linear programming solver.
    """
    # tol : float
    #     容差值，用于确定在第一阶段中解接近零以被视为基本可行解，或者接近正数以被视为最优解。
    # bland : bool
    #     如果为 True，则使用 Bland 的反循环规则 [3]_ 来选择主元，以防止循环。如果为 False，则选择可能更快导致收敛解的主元。
    #     后一种方法在极少数情况下可能会发生循环（不收敛）。
    # unknown_options : dict
    #     此特定求解器未使用的可选参数。如果 `unknown_options` 不为空，则发出警告列出所有未使用的选项。

    Returns
    -------
    x : 1-D array
        解向量。
    status : int
        表示优化退出状态的整数::

         0 : 优化成功终止
         1 : 达到迭代限制
         2 : 问题看起来是不可行的
         3 : 问题看起来是无界的
         4 : 遇到严重的数值困难

    message : str
        优化退出状态的字符串描述。
    iteration : int
        解决问题所花费的迭代次数。

    References
    ----------
    .. [1] Dantzig, George B., Linear programming and extensions. Rand
           Corporation Research Study Princeton Univ. Press, Princeton, NJ,
           1963
    .. [2] Hillier, S.H. and Lieberman, G.J. (1995), "Introduction to
           Mathematical Programming", McGraw-Hill, Chapter 4.
    .. [3] Bland, Robert G. New finite pivoting rules for the simplex method.
           Mathematics of Operations Research (2), 1977: pp. 103-107.


    Notes
    -----
    顶层 ``linprog`` 模块和特定方法的求解器之间的预期问题形式有所不同。
    特定方法的求解器期望标准形式的问题：

    最小化::

        c @ x

    使得::

        A @ x == b
            x >= 0

    而顶层 ``linprog`` 模块期望形式为：

    最小化::

        c @ x

    使得::

        A_ub @ x <= b_ub
        A_eq @ x == b_eq
         lb <= x <= ub

    其中 ``lb = 0`` 而 ``ub = None``，除非在 ``bounds`` 中设置。

    原始问题包含等式、上界和变量约束，而方法特定的求解器要求等式约束和变量非负性。

    ``linprog`` 模块通过将简单边界转换为上界约束，为不等式约束引入非负松弛变量，并表达无界变量为两个非负变量的差异，将原始问题转换为标准形式。
    """
    # 检查未知选项
    _check_unknown_options(unknown_options)

    # 初始化优化退出状态为成功（0）
    status = 0
    # 定义一个消息字典，用于存储优化过程中的状态信息
    messages = {0: "Optimization terminated successfully.",
                1: "Iteration limit reached.",
                2: "Optimization failed. Unable to find a feasible"
                   " starting point.",
                3: "Optimization failed. The problem appears to be unbounded.",
                4: "Optimization failed. Singular matrix encountered."}

    # 获取矩阵 A 的形状，n 为行数，m 为列数
    n, m = A.shape

    # 检查所有约束条件，确保 b >= 0
    is_negative_constraint = np.less(b, 0)
    A[is_negative_constraint] *= -1
    b[is_negative_constraint] *= -1

    # 创建人工变量并设置其为基本变量
    av = np.arange(n) + m
    basis = av.copy()

    # 构建第一阶段单纯形法表格，添加人工变量，堆叠约束、目标行和伪目标行
    row_constraints = np.hstack((A, np.eye(n), b[:, np.newaxis]))
    row_objective = np.hstack((c, np.zeros(n), c0))
    row_pseudo_objective = -row_constraints.sum(axis=0)
    row_pseudo_objective[av] = 0
    T = np.vstack((row_constraints, row_objective, row_pseudo_objective))

    # 执行第一阶段单纯形法求解
    nit1, status = _solve_simplex(T, n, basis, callback=callback,
                                  postsolve_args=postsolve_args,
                                  maxiter=maxiter, tol=tol, phase=1,
                                  bland=bland
                                  )

    # 如果伪目标值接近零，移除表格中的最后一行并进行第二阶段
    nit2 = nit1
    if abs(T[-1, -1]) < tol:
        # 移除表格中的伪目标行
        T = T[:-1, :]
        # 从表格中移除人工变量的列
        T = np.delete(T, av, 1)
    else:
        # 第一阶段单纯形法未能找到可行起始点
        status = 2
        # 更新消息字典中的状态信息，说明失败原因
        messages[status] = (
            "Phase 1 of the simplex method failed to find a feasible "
            "solution. The pseudo-objective function evaluates to {0:.1e} "
            "which exceeds the required tolerance of {1} for a solution to be "
            "considered 'close enough' to zero to be a basic solution. "
            "Consider increasing the tolerance to be greater than {0:.1e}. "
            "If this tolerance is unacceptably  large the problem may be "
            "infeasible.".format(abs(T[-1, -1]), tol)
        )

    # 如果第一阶段成功，执行第二阶段单纯形法求解
    if status == 0:
        nit2, status = _solve_simplex(T, n, basis, callback=callback,
                                      postsolve_args=postsolve_args,
                                      maxiter=maxiter, tol=tol, phase=2,
                                      bland=bland, nit0=nit1
                                      )

    # 构造解向量并返回最优解 x、状态码 status、状态消息和迭代次数 nit2
    solution = np.zeros(n + m)
    solution[basis[:n]] = T[:n, -1]
    x = solution[:m]

    return x, status, messages[status], int(nit2)
```