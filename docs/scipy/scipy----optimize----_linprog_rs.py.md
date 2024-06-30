# `D:\src\scipysrc\scipy\scipy\optimize\_linprog_rs.py`

```
"""
Revised simplex method for linear programming

The *revised simplex* method uses the method described in [1]_, except
that a factorization [2]_ of the basis matrix, rather than its inverse,
is efficiently maintained and used to solve the linear systems at each
iteration of the algorithm.

.. versionadded:: 1.3.0

References
----------
.. [1] Bertsimas, Dimitris, and J. Tsitsiklis. "Introduction to linear
           programming." Athena Scientific 1 (1997): 997.
.. [2] Bartels, Richard H. "A stabilization of the simplex method."
            Journal in  Numerische Mathematik 16.5 (1971): 414-434.

"""
# Author: Matt Haberland

import numpy as np
from numpy.linalg import LinAlgError

from scipy.linalg import solve
from ._optimize import _check_unknown_options
from ._bglu_dense import LU
from ._bglu_dense import BGLU as BGLU
from ._linprog_util import _postsolve
from ._optimize import OptimizeResult


def _phase_one(A, b, x0, callback, postsolve_args, maxiter, tol, disp,
               maxupdate, mast, pivot):
    """
    The purpose of phase one is to find an initial basic feasible solution
    (BFS) to the original problem.

    Generates an auxiliary problem with a trivial BFS and an objective that
    minimizes infeasibility of the original problem. Solves the auxiliary
    problem using the main simplex routine (phase two). This either yields
    a BFS to the original problem or determines that the original problem is
    infeasible. If feasible, phase one detects redundant rows in the original
    constraint matrix and removes them, then chooses additional indices as
    necessary to complete a basis/BFS for the original problem.
    """

    m, n = A.shape
    status = 0

    # generate auxiliary problem to get initial BFS
    # 生成辅助问题以获得初始基本可行解（BFS）
    A, b, c, basis, x, status = _generate_auxiliary_problem(A, b, x0, tol)

    if status == 6:
        residual = c.dot(x)
        iter_k = 0
        return x, basis, A, b, residual, status, iter_k

    # solve auxiliary problem
    # 解决辅助问题
    phase_one_n = n
    iter_k = 0
    x, basis, status, iter_k = _phase_two(c, A, x, basis, callback,
                                          postsolve_args,
                                          maxiter, tol, disp,
                                          maxupdate, mast, pivot,
                                          iter_k, phase_one_n)

    # check for infeasibility
    # 检查不可行性
    residual = c.dot(x)
    if status == 0 and residual > tol:
        status = 2

    # drive artificial variables out of basis
    # 驱使人工变量离开基
    # TODO: test redundant row removal better
    # TODO: make solve more efficient with BGLU? This could take a while.
    keep_rows = np.ones(m, dtype=bool)
    # 遍历基变量中大于等于给定阈值 n 的列
    for basis_column in basis[basis >= n]:
        # 根据当前的基变量集合 B，选取对应的 A 列子集
        B = A[:, basis]
        try:
            # 尝试求解基变量中的矩阵 B 在全体 A 中的线性组合，可能效率低下
            basis_finder = np.abs(solve(B, A))  # inefficient
            
            # 找到最相关的行，其列索引为 basis_column
            pertinent_row = np.argmax(basis_finder[:, basis_column])
            
            # 创建一个布尔数组标志哪些列是可选的
            eligible_columns = np.ones(n, dtype=bool)
            eligible_columns[basis[basis < n]] = 0
            
            # 提取可选列的索引
            eligible_column_indices = np.where(eligible_columns)[0]
            
            # 在 eligible_columns 中找到最相关的列索引，用于更新基变量
            index = np.argmax(basis_finder[:, :n][pertinent_row, eligible_columns])
            
            # 新的基变量列索引
            new_basis_column = eligible_column_indices[index]
            
            # 如果找到的新基变量的相关性小于给定的容差 tol，则标记为不保留
            if basis_finder[pertinent_row, new_basis_column] < tol:
                keep_rows[pertinent_row] = False
            else:
                # 否则更新基变量集合中与原始基变量列相同的列为新的基变量列
                basis[basis == basis_column] = new_basis_column
        except LinAlgError:
            # 如果求解过程中出现线性代数错误，则更新状态为 4
            status = 4

    # 形成原始问题的解
    A = A[keep_rows, :n]
    basis = basis[keep_rows]
    x = x[:n]
    m = A.shape[0]
    # 返回解向量 x，基变量集合 basis，剩余的约束条件矩阵 A，右侧向量 b，残差 residual，状态码 status，迭代次数 iter_k
    return x, basis, A, b, residual, status, iter_k
# 定义一个函数，用于获取更多的基本列，以解决辅助问题在基本变量中包含人工列的情况。
def _get_more_basis_columns(A, basis):
    """
    Called when the auxiliary problem terminates with artificial columns in
    the basis, which must be removed and replaced with non-artificial
    columns. Finds additional columns that do not make the matrix singular.
    """
    m, n = A.shape  # 获取矩阵 A 的行数 m 和列数 n

    # options for inclusion are those that aren't already in the basis
    # 创建一个长度为 m+n 的布尔类型数组 bl，基于 basis 中已有的索引设置为 True
    a = np.arange(m+n)
    bl = np.zeros(len(a), dtype=bool)
    bl[basis] = 1
    options = a[~bl]  # 选出不在 basis 中的索引，即可作为新的基本列的候选项
    options = options[options < n]  # 确保候选项在非人工列范围内

    # form basis matrix
    B = np.zeros((m, m))  # 创建一个 m 行 m 列的零矩阵 B
    B[:, 0:len(basis)] = A[:, basis]  # 将 A 中基本列的内容复制到 B 的前 len(basis) 列

    if (basis.size > 0 and
            np.linalg.matrix_rank(B[:, :len(basis)]) < len(basis)):
        raise Exception("Basis has dependent columns")  # 如果基本矩阵 B 的前 len(basis) 列线性相关，抛出异常

    rank = 0  # 初始化排名变量为 0，仅进入循环
    for i in range(n):  # 循环尝试 n 次，寻找新的基本列
        # permute the options, and take as many as needed
        new_basis = np.random.permutation(options)[:m-len(basis)]  # 随机排列候选项，并取出需要数量的新基本列
        B[:, len(basis):] = A[:, new_basis]  # 更新基本矩阵 B，加入新的基本列
        rank = np.linalg.matrix_rank(B)  # 计算更新后的基本矩阵 B 的秩
        if rank == m:  # 如果秩等于 m，说明找到了合适的基本列
            break

    return np.concatenate((basis, new_basis))  # 返回扩展后的基本列索引数组
    # 如果 x0 为 None，则选择所有约束条件作为非零约束条件索引
    # 否则，选择 r 中大于 tol 的索引作为非零约束条件索引
    if x0 is None:
        nonzero_constraints = np.arange(m)
    else:
        nonzero_constraints = np.where(r > tol)[0]

    # 选择绝对值大于 tol 的 x 的索引作为初始基础列索引
    basis = np.where(np.abs(x) > tol)[0]

    # 如果非零约束条件为空并且基础列数小于等于约束条件数，则已经是基础可行解（BFS）
    if len(nonzero_constraints) == 0 and len(basis) <= m:
        c = np.zeros(n)
        # 获取更多的基础列
        basis = _get_more_basis_columns(A, basis)
        return A, b, c, basis, x, status
    # 如果非零约束条件数大于 m 减去基础列数或者 x 中有负数，则无法得到平凡基础可行解
    elif (len(nonzero_constraints) > m - len(basis) or
          np.any(x < 0)):
        c = np.zeros(n)
        status = 6
        return A, b, c, basis, x, status

    # 选择适合作为初始基础的现有列
    cols, rows = _select_singleton_columns(A, r)

    # 找到需要用列单例来清零的行
    i_tofix = np.isin(rows, nonzero_constraints)
    # 这些列不能已经在基础中
    i_notinbasis = np.logical_not(np.isin(cols, basis))
    i_fix_without_aux = np.logical_and(i_tofix, i_notinbasis)
    rows = rows[i_fix_without_aux]
    cols = cols[i_fix_without_aux]

    # 只能使用辅助变量清零的行索引
    arows = nonzero_constraints[np.logical_not(
                                np.isin(nonzero_constraints, rows))]
    n_aux = len(arows)
    acols = n + np.arange(n_aux)          # 辅助列的索引

    # 不是猜测得到的基础列
    basis_ng = np.concatenate((cols, acols))
    # 需要清零的行
    basis_ng_rows = np.concatenate((rows, arows))

    # 添加辅助单例列
    A = np.hstack((A, np.zeros((m, n_aux))))
    A[arows, acols] = 1

    # 生成初始基础可行解
    x = np.concatenate((x, np.zeros(n_aux)))
    x[basis_ng] = r[basis_ng_rows] / A[basis_ng_rows, basis_ng]

    # 生成最小化非可行性的成本
    c = np.zeros(n_aux + n)
    c[acols] = 1

    # 基础列对应于猜测中的非零列，用于清零剩余约束的单例列，以及任何额外的列以得到完整的 m 列
    basis = np.concatenate((basis, basis_ng))
    # 添加需要的基础列
    basis = _get_more_basis_columns(A, basis)

    return A, b, c, basis, x, status
def _select_singleton_columns(A, b):
    """
    Finds singleton columns for which the singleton entry is of the same sign
    as the right-hand side; these columns are eligible for inclusion in an
    initial basis. Determines the rows in which the singleton entries are
    located. For each of these rows, returns the indices of the one singleton
    column and its corresponding row.
    """
    # 找到所有单列的索引以及相应的行索引
    column_indices = np.nonzero(np.sum(np.abs(A) != 0, axis=0) == 1)[0]
    columns = A[:, column_indices]          # 单列数组
    row_indices = np.zeros(len(column_indices), dtype=int)
    nonzero_rows, nonzero_columns = np.nonzero(columns)
    row_indices[nonzero_columns] = nonzero_rows   # 对应的行索引

    # 保留符号与右手边相同的单列，因为BFS的所有元素必须为非负数
    same_sign = A[row_indices, column_indices]*b[row_indices] >= 0
    column_indices = column_indices[same_sign][::-1]
    row_indices = row_indices[same_sign][::-1]
    # 反转顺序，以便以下步骤选择最右边的列作为初始基础，通常是松弛变量

    # 对于每一行，保留具有条目的最右边的单列
    unique_row_indices, first_columns = np.unique(row_indices,
                                                  return_index=True)
    return column_indices[first_columns], unique_row_indices


def _find_nonzero_rows(A, tol):
    """
    Returns logical array indicating the locations of rows with at least
    one nonzero element.
    """
    return np.any(np.abs(A) > tol, axis=1)


def _select_enter_pivot(c_hat, bl, a, rule="bland", tol=1e-12):
    """
    Selects a pivot to enter the basis. Currently Bland's rule - the smallest
    index that has a negative reduced cost - is the default.
    """
    if rule.lower() == "mrc":  # 最小化约简成本的索引
        return a[~bl][np.argmin(c_hat)]
    else:  # 最小负约简成本的索引
        return a[~bl][c_hat < -tol][0]


def _display_iter(phase, iteration, slack, con, fun):
    """
    Print indicators of optimization status to the console.
    """
    header = True if not iteration % 20 else False

    if header:
        print("Phase",
              "Iteration",
              "Minimum Slack      ",
              "Constraint Residual",
              "Objective          ")

    # :<X.Y 左对齐，在X位数中占Y位数
    fmt = '{0:<6}{1:<10}{2:<20.13}{3:<20.13}{4:<20.13}'
    try:
        slack = np.min(slack)
    except ValueError:
        slack = "NA"
    print(fmt.format(phase, iteration, slack, np.linalg.norm(con), fun))
# 定义一个函数，用于在单纯形法的第二阶段执行优化过程。
# 该函数实现从基本可行解开始，连续移动到具有更低约化成本的相邻基本可行解。
# 当不存在更低约化成本的基本可行解或问题确定为无界时终止。

def _phase_two(c, A, x, b, callback, postsolve_args, maxiter, tol, disp,
               maxupdate, mast, pivot, iteration=0, phase_one_n=None):
    """
    单纯形法的核心部分。从基本可行解开始，逐步移动到具有较低约化成本的相邻基本可行解。
    当不存在更低约化成本的基本可行解或问题确定为无界时终止。

    该实现基于LU分解的修订单纯形法。与维护表格或基础矩阵的逆不同，我们保留基础矩阵的分解，
    允许有效地解决线性系统，同时避免与求逆矩阵相关的稳定性问题。
    """
    
    m, n = A.shape                    # 获取矩阵 A 的行数 m 和列数 n
    status = 0                       # 初始化优化状态为 0
    a = np.arange(n)                # 列索引数组，表示矩阵 A 的所有列索引
    ab = np.arange(m)               # 列索引数组，表示基础矩阵的所有列索引 B
    
    if maxupdate:
        # 如果允许最大更新，则使用 BGLU 对象进行基础矩阵的因子分解；类似于 B = A[:, b]
        B = BGLU(A, b, maxupdate, mast)
    else:
        # 否则使用 LU 分解对基础矩阵 B 进行分解
        B = LU(A, b)
    for iteration in range(iteration, maxiter):
        # 循环执行迭代过程，从当前迭代次数开始到最大迭代次数之前

        if disp or callback is not None:
            # 如果需要显示或者存在回调函数，则执行显示和回调
            _display_and_callback(phase_one_n, x, postsolve_args, status,
                                  iteration, disp, callback)

        bl = np.zeros(len(a), dtype=bool)
        bl[b] = 1
        # 创建布尔数组 bl，长度与 a 相同，并将基变量索引 b 对应位置设为 True

        xb = x[b]       # basic variables
        cb = c[b]       # basic costs
        # 从变量 x 和成本 c 中提取基变量对应的值

        try:
            v = B.solve(cb, transposed=True)    # similar to v = solve(B.T, cb)
            # 使用基向量 B 解线性方程 B.T * v = cb，得到基向量的解 v
        except LinAlgError:
            status = 4
            break
        # 若出现线性代数错误，则将状态设置为 4 并中断迭代

        c_hat = c - v.dot(A)    # reduced cost
        c_hat = c_hat[~bl]
        # 计算减少的成本 c_hat，只考虑非基变量的部分

        if np.all(c_hat >= -tol):  # all reduced costs positive -> terminate
            break
        # 如果所有减少的成本都大于等于负容差 tol，则终止迭代

        j = _select_enter_pivot(c_hat, bl, a, rule=pivot, tol=tol)
        # 选择进入基变量集的变量 j，使用给定的进入规则和容差 tol

        u = B.solve(A[:, j])        # similar to u = solve(B, A[:, j])
        # 使用基向量 B 解线性方程 B * u = A[:, j]，得到 u

        i = u > tol                 # if none of the u are positive, unbounded
        if not np.any(i):
            status = 3
            break
        # 如果所有 u 都小于等于容差 tol，则问题无界，将状态设置为 3 并中断迭代

        th = xb[i]/u[i]
        l = np.argmin(th)           # implicitly selects smallest subscript
        th_star = th[l]             # step size
        # 计算步长 th_star，并找到最小的 th 对应的索引 l

        x[b] = x[b] - th_star*u     # take step
        x[j] = th_star
        # 更新变量 x 中的基变量和非基变量的值，执行一步迭代

        B.update(ab[i][l], j)       # modify basis
        b = B.b
        # 更新基向量 B 中的基索引集合 b

    else:
        # 如果循环正常结束（没有 break 语句），则表示进行了额外的一步迭代
        iteration += 1
        status = 1
        if disp or callback is not None:
            _display_and_callback(phase_one_n, x, postsolve_args, status,
                                  iteration, disp, callback)
        # 增加迭代次数，显示信息并调用回调函数

    return x, b, status, iteration
    # 返回最终的变量 x，基变量集合 b，状态 status，以及迭代次数 iteration
# 定义一个函数来解决线性规划问题，使用修订型单纯形算法的两阶段方法。

def _linprog_rs(c, c0, A, b, x0, callback, postsolve_args,
                maxiter=5000, tol=1e-12, disp=False,
                maxupdate=10, mast=False, pivot="mrc",
                **unknown_options):
    """
    Solve the following linear programming problem via a two-phase
    revised simplex algorithm.::

        minimize:     c @ x

        subject to:  A @ x == b
                     0 <= x < oo

    User-facing documentation is in _linprog_doc.py.

    Parameters
    ----------
    c : 1-D array
        Coefficients of the linear objective function to be minimized.
    c0 : float
        Constant term in objective function due to fixed (and eliminated)
        variables. (Currently unused.)
    A : 2-D array
        2-D array which, when matrix-multiplied by ``x``, gives the values of
        the equality constraints at ``x``.
    b : 1-D array
        1-D array of values representing the RHS of each equality constraint
        (row) in ``A_eq``.
    x0 : 1-D array, optional
        Starting values of the independent variables, which will be refined by
        the optimization algorithm. For the revised simplex method, these must
        correspond with a basic feasible solution.
    callback : callable, optional
        If a callback function is provided, it will be called within each
        iteration of the algorithm. The callback function must accept a single
        `scipy.optimize.OptimizeResult` consisting of the following fields:

            x : 1-D array
                Current solution vector.
            fun : float
                Current value of the objective function ``c @ x``.
            success : bool
                True only when an algorithm has completed successfully,
                so this is always False as the callback function is called
                only while the algorithm is still iterating.
            slack : 1-D array
                The values of the slack variables. Each slack variable
                corresponds to an inequality constraint. If the slack is zero,
                the corresponding constraint is active.
            con : 1-D array
                The (nominally zero) residuals of the equality constraints,
                that is, ``b - A_eq @ x``.
            phase : int
                The phase of the algorithm being executed.
            status : int
                For revised simplex, this is always 0 because if a different
                status is detected, the algorithm terminates.
            nit : int
                The number of iterations performed.
            message : str
                A string descriptor of the exit status of the optimization.
    postsolve_args : tuple
        Data needed by _postsolve to convert the solution to the standard-form
        problem into the solution to the original problem.

    Options
    -------
    maxiter : int, optional
        The maximum number of iterations to perform in either phase.
    tol : float, optional
        Tolerance for numerical errors.
    disp : bool, optional
        If True, print the process during optimization.
    maxupdate : int, optional
        Maximum number of updates performed in a single iteration.
    mast : bool, optional
        If True, use Markowitz's anti-cycling pivot selection strategy.
    pivot : str, optional
        Method to choose the pivot element in the simplex algorithm.
    **unknown_options : dict
        Other keyword arguments passed to the function.

    """
    # tol : float
    #     容差值，用于确定第一阶段解是否“足够接近”零，从而被视为基本可行解，
    #     或者足够接近正数以作为最优解。
    # disp : bool
    #     如果将优化状态的指示器打印到每次迭代的控制台，则设置为True。
    # maxupdate : int
    #     LU分解执行的最大更新次数。达到此更新次数后，基矩阵将从头开始进行因式分解。
    # mast : bool
    #     最小化摊销解决时间。如果启用，将测量使用基础因式分解解决线性系统的平均时间。
    #     通常，在初始因式分解后，每次连续求解后，平均求解时间会减少，因为因式分解比解操作（和更新）需要更多时间。
    #     然而，最终，更新后的因式分解变得足够复杂，平均求解时间开始增加。当检测到这种情况时，基础矩阵将重新进行因式分解。
    #     启用此选项以最大化速度，但可能会出现非确定性行为。如果maxupdate为0，则忽略此选项。
    # pivot : "mrc" or "bland"
    #     基轴选择规则：最小减少成本（默认）或 Bland's 规则。如果达到迭代限制并且存在循环怀疑，则选择 Bland's 规则。
    # unknown_options : dict
    #     此特定求解器未使用的可选参数。如果`unknown_options`非空，则发出警告，列出所有未使用的选项。
    #
    # Returns
    # -------
    # x : 1-D array
    #     解向量。
    # status : int
    #     表示优化退出状态的整数：
    #     0 : 优化成功终止
    #     1 : 达到迭代限制
    #     2 : 问题似乎不可行
    #     3 : 问题似乎无界
    #     4 : 遇到数值困难
    #     5 : 没有约束条件；打开预处理
    #     6 : 猜测的 x0 无法转换为基本可行解
    # message : str
    #     优化退出状态的字符串描述。
    # iteration : int
    #     解决问题所用的迭代次数。
    """
    检查未知选项，如果有未使用的选项则发出警告。
    """
    _check_unknown_options(unknown_options)
    # 定义一组优化算法运行时可能出现的消息，用于不同的优化终止情况的描述
    messages = [
        "Optimization terminated successfully.",  # 最优化成功终止
        "Iteration limit reached.",  # 达到迭代次数限制
        "The problem appears infeasible, as the phase one auxiliary "
        "problem terminated successfully with a residual of {0:.1e}, "
        "greater than the tolerance {1} required for the solution to "
        "be considered feasible. Consider increasing the tolerance to "
        "be greater than {0:.1e}. If this tolerance is unnaceptably "
        "large, the problem is likely infeasible.",  # 问题不可行，阶段一辅助问题成功终止，显示残差和容差的信息
        "The problem is unbounded, as the simplex algorithm found "
        "a basic feasible solution from which there is a direction "
        "with negative reduced cost in which all decision variables "
        "increase.",  # 问题无界，单纯形算法找到一个基础可行解，所有决策变量在一个负的降低成本方向上增加
        "Numerical difficulties encountered; consider trying "
        "method='interior-point'.",  # 遇到数值困难，考虑尝试内点法（interior-point）
        "Problems with no constraints are trivially solved; please "
        "turn presolve on.",  # 没有约束的问题可以轻松解决，建议打开预处理（presolve）
        "The guess x0 cannot be converted to a basic feasible "
        "solution. "  # 初始猜测 x0 无法转换为基础可行解
    ]
    
    if A.size == 0:  # 如果 A 的大小为 0，即无约束情况，返回一个全零向量及相应的状态码、消息和迭代次数
        return np.zeros(c.shape), 5, messages[5], 0
    
    # 运行阶段一优化过程，得到最优解 x、基、约束矩阵 A、右侧向量 b、残差、状态、迭代次数
    x, basis, A, b, residual, status, iteration = (
        _phase_one(A, b, x0, callback, postsolve_args,
                   maxiter, tol, disp, maxupdate, mast, pivot)
    )
    
    if status == 0:  # 如果阶段一优化成功完成
        # 运行阶段二优化过程，得到最优解 x、基、状态、迭代次数
        x, basis, status, iteration = _phase_two(c, A, x, basis, callback,
                                                 postsolve_args,
                                                 maxiter, tol, disp,
                                                 maxupdate, mast, pivot,
                                                 iteration)
        
    # 返回最优解 x、状态、使用状态码对应的消息格式化后的描述及迭代次数
    return x, status, messages[status].format(residual, tol), iteration
```