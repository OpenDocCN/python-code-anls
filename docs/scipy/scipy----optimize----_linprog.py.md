# `D:\src\scipysrc\scipy\scipy\optimize\_linprog.py`

```
"""
A top-level linear programming interface.

.. versionadded:: 0.15.0

Functions
---------
.. autosummary::
   :toctree: generated/

    linprog
    linprog_verbose_callback
    linprog_terse_callback

"""

import numpy as np  # 导入 NumPy 库

from ._optimize import OptimizeResult, OptimizeWarning  # 导入优化结果类和警告类
from warnings import warn  # 导入警告模块
from ._linprog_highs import _linprog_highs  # 导入 Highs 方法接口
from ._linprog_ip import _linprog_ip  # 导入 interior-point 方法接口
from ._linprog_simplex import _linprog_simplex  # 导入 simplex 方法接口
from ._linprog_rs import _linprog_rs  # 导入 revised simplex 方法接口
from ._linprog_doc import (_linprog_highs_doc, _linprog_ip_doc,  # 导入方法文档
                           _linprog_rs_doc, _linprog_simplex_doc,
                           _linprog_highs_ipm_doc, _linprog_highs_ds_doc)
from ._linprog_util import (  # 导入线性规划实用函数
    _parse_linprog, _presolve, _get_Abc, _LPProblem, _autoscale,
    _postsolve, _check_result, _display_summary)
from copy import deepcopy  # 导入深度拷贝函数

__all__ = ['linprog', 'linprog_verbose_callback', 'linprog_terse_callback']  # 公开的接口名称列表

__docformat__ = "restructuredtext en"  # 文档格式设置为 reStructuredText

LINPROG_METHODS = [  # 线性规划方法列表
    'simplex', 'revised simplex', 'interior-point', 'highs', 'highs-ds', 'highs-ipm'
]


def linprog_verbose_callback(res):
    """
    A sample callback function demonstrating the linprog callback interface.
    This callback produces detailed output to sys.stdout before each iteration
    and after the final iteration of the simplex algorithm.

    Parameters
    ----------
    res : A `scipy.optimize.OptimizeResult` consisting of the following fields:

        x : 1-D array
            The independent variable vector which optimizes the linear
            programming problem.
        fun : float
            Value of the objective function.
        success : bool
            True if the algorithm succeeded in finding an optimal solution.
        slack : 1-D array
            The values of the slack variables. Each slack variable corresponds
            to an inequality constraint. If the slack is zero, then the
            corresponding constraint is active.
        con : 1-D array
            The (nominally zero) residuals of the equality constraints, that is,
            ``b - A_eq @ x``
        phase : int
            The phase of the optimization being executed. In phase 1 a basic
            feasible solution is sought and the T has an additional row
            representing an alternate objective function.
        status : int
            An integer representing the exit status of the optimization::

                 0 : Optimization terminated successfully
                 1 : Iteration limit reached
                 2 : Problem appears to be infeasible
                 3 : Problem appears to be unbounded
                 4 : Serious numerical difficulties encountered

        nit : int
            The number of iterations performed.
        message : str
            A string descriptor of the exit status of the optimization.
    """
    x = res['x']  # 获取优化结果中的自变量向量 x
    fun = res['fun']  # 获取优化结果中的目标函数值 fun
    phase = res['phase']  # 获取优化结果中的优化阶段 phase
    status = res['status']  # 获取优化结果中的优化状态 status
    nit = res['nit']  # 获取优化结果中的迭代次数 nit
    # 从结果字典中获取 'message' 字段，用于后续输出
    message = res['message']
    # 从结果字典中获取 'complete' 字段，用于后续条件判断
    complete = res['complete']
    
    # 获取当前的 NumPy 打印选项并保存以备恢复
    saved_printoptions = np.get_printoptions()
    # 设置新的 NumPy 打印选项，设置行宽为500，设置浮点数格式化为小数点后四位
    np.set_printoptions(linewidth=500,
                        formatter={'float': lambda x: f"{x: 12.4f}"})
    
    # 根据不同的条件输出不同的信息
    if status:
        # 如果存在 status，表明简单法算法提前退出
        print('--------- Simplex Early Exit -------\n')
        print(f'The simplex method exited early with status {status:d}')
        print(message)
    elif complete:
        # 如果 complete 为 True，表明简单法算法已完成
        print('--------- Simplex Complete --------\n')
        print(f'Iterations required: {nit}')
    else:
        # 其他情况下输出当前迭代次数
        print(f'--------- Iteration {nit:d}  ---------\n')
    
    # 如果迭代次数大于0，输出当前的目标函数值和解向量
    if nit > 0:
        if phase == 1:
            # 如果处于第一阶段，输出当前的伪目标值
            print('Current Pseudo-Objective Value:')
        else:
            # 否则输出当前的目标函数值
            print('Current Objective Value:')
        print('f = ', fun)
        print()
        print('Current Solution Vector:')
        print('x = ', x)
        print()
    
    # 恢复之前保存的 NumPy 打印选项
    np.set_printoptions(**saved_printoptions)
def linprog_terse_callback(res):
    """
    A sample callback function demonstrating the linprog callback interface.
    This callback produces brief output to sys.stdout before each iteration
    and after the final iteration of the simplex algorithm.

    Parameters
    ----------
    res : A `scipy.optimize.OptimizeResult` consisting of the following fields:

        x : 1-D array
            The independent variable vector which optimizes the linear
            programming problem.
        fun : float
            Value of the objective function.
        success : bool
            True if the algorithm succeeded in finding an optimal solution.
        slack : 1-D array
            The values of the slack variables. Each slack variable corresponds
            to an inequality constraint. If the slack is zero, then the
            corresponding constraint is active.
        con : 1-D array
            The (nominally zero) residuals of the equality constraints, that is,
            ``b - A_eq @ x``.
        phase : int
            The phase of the optimization being executed. In phase 1 a basic
            feasible solution is sought and the T has an additional row
            representing an alternate objective function.
        status : int
            An integer representing the exit status of the optimization::

                 0 : Optimization terminated successfully
                 1 : Iteration limit reached
                 2 : Problem appears to be infeasible
                 3 : Problem appears to be unbounded
                 4 : Serious numerical difficulties encountered

        nit : int
            The number of iterations performed.
        message : str
            A string descriptor of the exit status of the optimization.
    """
    # 从 res 中提取迭代次数
    nit = res['nit']
    # 从 res 中提取优化后的变量向量
    x = res['x']

    # 如果是第一次迭代，打印表头
    if nit == 0:
        print("Iter:   X:")
    # 打印迭代次数和当前变量向量 x
    print(f"{nit: <5d}   ", end="")
    print(x)


def linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
            bounds=(0, None), method='highs', callback=None,
            options=None, x0=None, integrality=None):
    r"""
    Linear programming: minimize a linear objective function subject to linear
    equality and inequality constraints.

    Linear programming solves problems of the following form:

    .. math::

        \min_x \ & c^T x \\
        \mbox{such that} \ & A_{ub} x \leq b_{ub},\\
        & A_{eq} x = b_{eq},\\
        & l \leq x \leq u ,

    where :math:`x` is a vector of decision variables; :math:`c`,
    :math:`b_{ub}`, :math:`b_{eq}`, :math:`l`, and :math:`u` are vectors; and
    :math:`A_{ub}` and :math:`A_{eq}` are matrices.

    Alternatively, that's:

        - minimize ::

            c @ x

        - such that ::

            A_ub @ x <= b_ub
            A_eq @ x == b_eq
            lb <= x <= ub

    Note that by default ``lb = 0`` and ``ub = None``. Other bounds can be
    specified with ``bounds``.

    Parameters
    ----------
    c : 1-D array
        Coefficients of the linear objective function to be minimized.
    A_ub : 2-D array, optional
        2-D array such that ``A_ub @ x`` gives the values of the inequality
        constraints at ``x``.
    b_ub : 1-D array, optional
        1-D array of values representing the upper bounds of the inequality
        constraints (``A_ub @ x <= b_ub``).
    A_eq : 2-D array, optional
        2-D array such that ``A_eq @ x`` gives the values of the equality
        constraints at ``x``.
    b_eq : 1-D array, optional
        1-D array of values representing the RHS of the equality constraints
        (``A_eq @ x = b_eq``).
    bounds : sequence, optional
        Bounds for variables. ``(min, max)`` pairs for each element in ``x``,
        defining the lower and upper bounds for each decision variable. Use
        ``None`` to indicate that there is no bound. By default, bounds are
        ``(0, None)``, i.e., all decision variables are non-negative.
    method : str, optional
        Method to solve the problem. Default is 'highs'.
    callback : callable, optional
        If a callback function is provided, it will be called after each
        iteration of the simplex algorithm.
    options : dict, optional
        A dictionary of solver options.
    x0 : 1-D array, optional
        Initial guess. By default, the initial guess is ``None``, which means
        that the solver will use a default guess.
    integrality : array_like, optional
        A list of bools or integers indicating which variables must take
        integer values. If integers, it indicates the variable indices.

    Returns
    -------
    res : OptimizeResult
        A `scipy.optimize.OptimizeResult` consisting of the following fields:

            x : 1-D array
                The independent variable vector which optimizes the linear
                programming problem.
            fun : float
                Value of the objective function.
            success : bool
                True if the algorithm succeeded in finding an optimal solution.
            slack : 1-D array
                The values of the slack variables. Each slack variable corresponds
                to an inequality constraint. If the slack is zero, then the
                corresponding constraint is active.
            con : 1-D array
                The (nominally zero) residuals of the equality constraints, that is,
                ``b - A_eq @ x``.
            phase : int
                The phase of the optimization being executed. In phase 1 a basic
                feasible solution is sought and the T has an additional row
                representing an alternate objective function.
            status : int
                An integer representing the exit status of the optimization::

                     0 : Optimization terminated successfully
                     1 : Iteration limit reached
                     2 : Problem appears to be infeasible
                     3 : Problem appears to be unbounded
                     4 : Serious numerical difficulties encountered

            nit : int
                The number of iterations performed.
            message : str
                A string descriptor of the exit status of the optimization.
    """
    # 实现线性规划的算法，具体细节由具体的线性规划算法库实现
    pass
    # 线性目标函数的系数数组，用于定义要最小化的线性目标函数
    c : 1-D array
        The coefficients of the linear objective function to be minimized.
    
    # 不等式约束条件的系数矩阵。每一行指定了对 x 的线性不等式约束的系数
    A_ub : 2-D array, optional
        The inequality constraint matrix. Each row of ``A_ub`` specifies the
        coefficients of a linear inequality constraint on ``x``.
    
    # 不等式约束条件的上界向量。每个元素表示对应于 ``A_ub @ x`` 的上界
    b_ub : 1-D array, optional
        The inequality constraint vector. Each element represents an
        upper bound on the corresponding value of ``A_ub @ x``.
    
    # 等式约束条件的系数矩阵。每一行指定了对 x 的线性等式约束的系数
    A_eq : 2-D array, optional
        The equality constraint matrix. Each row of ``A_eq`` specifies the
        coefficients of a linear equality constraint on ``x``.
    
    # 等式约束条件的等式向量。 ``A_eq @ x`` 的每个元素必须等于 ``b_eq`` 的对应元素
    b_eq : 1-D array, optional
        The equality constraint vector. Each element of ``A_eq @ x`` must equal
        the corresponding element of ``b_eq``.
    
    # 决策变量的最小值和最大值 ``(min, max)`` 对。每个元素在 ``x`` 中定义其最小和最大值
    # 如果提供单个元组 ``(min, max)``，那么 ``min`` 和 ``max`` 将作为所有决策变量的边界
    # 使用 ``None`` 表示没有边界。例如，默认边界 ``(0, None)`` 表示所有决策变量都为非负数，
    # 元组 ``(None, None)`` 表示没有边界，即所有变量允许为任意实数
    bounds : sequence, optional
        A sequence of ``(min, max)`` pairs for each element in ``x``, defining
        the minimum and maximum values of that decision variable.
        If a single tuple ``(min, max)`` is provided, then ``min`` and ``max``
        will serve as bounds for all decision variables.
        Use ``None`` to indicate that there is no bound. For instance, the
        default bound ``(0, None)`` means that all decision variables are
        non-negative, and the pair ``(None, None)`` means no bounds at all,
        i.e. all variables are allowed to be any real.
    
    # 解决标准形式问题所使用的算法
    # 支持 :ref:`'highs' <optimize.linprog-highs>`（默认）,
    # :ref:`'highs-ds' <optimize.linprog-highs-ds>`,
    # :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`,
    # :ref:`'interior-point' <optimize.linprog-interior-point>`（遗留）,
    # :ref:`'revised simplex' <optimize.linprog-revised_simplex>`（遗留）,
    # 和 :ref:`'simplex' <optimize.linprog-simplex>`（遗留）等方法
    # 遗留方法已被弃用，并将在 SciPy 1.11.0 中移除
    method : str, optional
        The algorithm used to solve the standard form problem.
        :ref:`'highs' <optimize.linprog-highs>` (default),
        :ref:`'highs-ds' <optimize.linprog-highs-ds>`,
        :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`,
        :ref:`'interior-point' <optimize.linprog-interior-point>` (legacy),
        :ref:`'revised simplex' <optimize.linprog-revised_simplex>` (legacy),
        and
        :ref:`'simplex' <optimize.linprog-simplex>` (legacy) are supported.
        The legacy methods are deprecated and will be removed in SciPy 1.11.0.
    # 如果提供了回调函数，则在每次算法迭代时至少调用一次该函数
    callback : callable, optional
        # 回调函数必须接受一个 `scipy.optimize.OptimizeResult` 对象作为参数
        If a callback function is provided, it will be called at least once per
        iteration of the algorithm. The callback function must accept a single
        `scipy.optimize.OptimizeResult` consisting of the following fields:

        x : 1-D array
            # 当前的解向量
            The current solution vector.
        fun : float
            # 目标函数 `c @ x` 的当前值
            The current value of the objective function ``c @ x``.
        success : bool
            # 当算法成功完成时为 `True`
            ``True`` when the algorithm has completed successfully.
        slack : 1-D array
            # 松弛变量的值，通常是正数，``b_ub - A_ub @ x``
            The (nominally positive) values of the slack,
            ``b_ub - A_ub @ x``.
        con : 1-D array
            # 等式约束的残差，通常是零，``b_eq - A_eq @ x``
            The (nominally zero) residuals of the equality constraints,
            ``b_eq - A_eq @ x``.
        phase : int
            # 当前执行的算法阶段
            The phase of the algorithm being executed.
        status : int
            # 表示算法状态的整数

            ``0`` : 优化正常进行中.

            ``1`` : 达到迭代次数限制.

            ``2`` : 问题看起来不可行.

            ``3`` : 问题看起来无界.

            ``4`` : 遇到数值困难.

            nit : int
                # 当前迭代次数
                The current iteration number.
            message : str
                # 描述算法状态的字符串
                A string descriptor of the algorithm status.

        Callback functions are not currently supported by the HiGHS methods.
    options : dict, optional
        A dictionary of solver options. All methods accept the following
        options:

        maxiter : int
            Maximum number of iterations to perform.
            Default: see method-specific documentation.
        disp : bool
            Set to ``True`` to print convergence messages.
            Default: ``False``.
        presolve : bool
            Set to ``False`` to disable automatic presolve.
            Default: ``True``.

        All methods except the HiGHS solvers also accept:

        tol : float
            A tolerance which determines when a residual is "close enough" to
            zero to be considered exactly zero.
        autoscale : bool
            Set to ``True`` to automatically perform equilibration.
            Consider using this option if the numerical values in the
            constraints are separated by several orders of magnitude.
            Default: ``False``.
        rr : bool
            Set to ``False`` to disable automatic redundancy removal.
            Default: ``True``.
        rr_method : string
            Method used to identify and remove redundant rows from the
            equality constraint matrix after presolve. For problems with
            dense input, the available methods for redundancy removal are:

            "SVD":
                Repeatedly performs singular value decomposition on
                the matrix, detecting redundant rows based on nonzeros
                in the left singular vectors that correspond with
                zero singular values. May be fast when the matrix is
                nearly full rank.
            "pivot":
                Uses the algorithm presented in [5]_ to identify
                redundant rows.
            "ID":
                Uses a randomized interpolative decomposition.
                Identifies columns of the matrix transpose not used in
                a full-rank interpolative decomposition of the matrix.
            None:
                Uses "svd" if the matrix is nearly full rank, that is,
                the difference between the matrix rank and the number
                of rows is less than five. If not, uses "pivot". The
                behavior of this default is subject to change without
                prior notice.

            Default: None.
            For problems with sparse input, this option is ignored, and the
            pivot-based algorithm presented in [5]_ is used.

        For method-specific options, see
        :func:`show_options('linprog') <show_options>`.

    x0 : 1-D array, optional
        Guess values of the decision variables, which will be refined by
        the optimization algorithm. This argument is currently used only by the
        'revised simplex' method, and can only be used if `x0` represents a
        basic feasible solution.
    integrality : 1-D array or int, optional
        # 表示每个决策变量的整数约束类型。

        ``0`` : 连续变量；无整数约束。

        ``1`` : 整数变量；决策变量必须是整数，在 `bounds` 范围内。

        ``2`` : 半连续变量；决策变量必须在 `bounds` 范围内或取值为 ``0``。

        ``3`` : 半整数变量；决策变量必须是整数，在 `bounds` 范围内或取值为 ``0``。

        默认情况下，所有变量均为连续变量。

        对于混合整数约束，请提供一个形状为 `c.shape` 的数组。为了从较短的输入推断每个决策变量的约束，该参数将使用 `np.broadcast_to` 广播到 `c.shape`。

        当前只有 ``'highs'`` 方法使用此参数，其他情况下会被忽略。

    Returns
    -------
    res : OptimizeResult
        # 包含以下字段的 :class:`scipy.optimize.OptimizeResult`。注意，字段的返回类型可能取决于优化是否成功，因此建议在依赖其他字段之前检查 `OptimizeResult.status`。

        x : 1-D array
            最小化目标函数并满足约束条件时决策变量的值。
        fun : float
            目标函数 ``c @ x`` 的最优值。
        slack : 1-D array
            （名义上为正的）松弛变量的值， ``b_ub - A_ub @ x``。
        con : 1-D array
            等式约束的残差， ``b_eq - A_eq @ x``。
        success : bool
            当算法成功找到最优解时为 ``True`` 。
        status : int
            表示算法退出状态的整数。

            ``0`` : 优化成功终止。

            ``1`` : 达到迭代限制。

            ``2`` : 问题似乎不可行。

            ``3`` : 问题似乎无界。

            ``4`` : 遇到数值困难。

        nit : int
            在所有阶段执行的总迭代次数。
        message : str
            算法退出状态的字符串描述。

    See Also
    --------
    show_options : 解算器接受的额外选项。

    Notes
    -----
    本节描述了可以通过 `method` 参数选择的可用解算器。

    `'highs-ds'` 和 `'highs-ipm'` 是 HiGHS 单纯形和内点法解算器的接口 [13]_ 。

    `'highs'`（默认）在这两者之间自动选择。这些是SciPy中最快的线性规划解算器，特别适用于大型稀疏问题；哪一个更快取决于问题的性质。
    The other solvers (`'interior-point'`, `'revised simplex'`, and
    `'simplex'`) are legacy methods and will be removed in SciPy 1.11.0.

    Method *highs-ds* is a wrapper of the C++ high performance dual
    revised simplex implementation (HSOL) [13]_, [14]_. Method *highs-ipm*
    is a wrapper of a C++ implementation of an **i**\ nterior-\ **p**\ oint
    **m**\ ethod [13]_; it features a crossover routine, so it is as accurate
    as a simplex solver. Method *highs* chooses between the two automatically.
    For new code involving `linprog`, we recommend explicitly choosing one of
    these three method values.

    .. versionadded:: 1.6.0

    Method *interior-point* uses the primal-dual path following algorithm
    as outlined in [4]_. This algorithm supports sparse constraint matrices and
    is typically faster than the simplex methods, especially for large, sparse
    problems. Note, however, that the solution returned may be slightly less
    accurate than those of the simplex methods and will not, in general,
    correspond with a vertex of the polytope defined by the constraints.

    .. versionadded:: 1.0.0

    Method *revised simplex* uses the revised simplex method as described in
    [9]_, except that a factorization [11]_ of the basis matrix, rather than
    its inverse, is efficiently maintained and used to solve the linear systems
    at each iteration of the algorithm.

    .. versionadded:: 1.3.0

    Method *simplex* uses a traditional, full-tableau implementation of
    Dantzig's simplex algorithm [1]_, [2]_ (*not* the
    Nelder-Mead simplex). This algorithm is included for backwards
    compatibility and educational purposes.

    .. versionadded:: 0.15.0

    Before applying *interior-point*, *revised simplex*, or *simplex*,
    a presolve procedure based on [8]_ attempts
    to identify trivial infeasibilities, trivial unboundedness, and potential
    problem simplifications. Specifically, it checks for:

    - rows of zeros in ``A_eq`` or ``A_ub``, representing trivial constraints;
    - columns of zeros in ``A_eq`` `and` ``A_ub``, representing unconstrained
      variables;
    - column singletons in ``A_eq``, representing fixed variables; and
    - column singletons in ``A_ub``, representing simple bounds.

    If presolve reveals that the problem is unbounded (e.g. an unconstrained
    and unbounded variable has negative cost) or infeasible (e.g., a row of
    zeros in ``A_eq`` corresponds with a nonzero in ``b_eq``), the solver
    terminates with the appropriate status code. Note that presolve terminates
    as soon as any sign of unboundedness is detected; consequently, a problem
    may be reported as unbounded when in reality the problem is infeasible
    (but infeasibility has not been detected yet). Therefore, if it is
    important to know whether the problem is actually infeasible, solve the
    problem again with option ``presolve=False``.
    # 如果在单次预处理中未检测到不可行性或无界性，将可能的边界收紧，并移除问题中的固定变量。
    # 然后，移除``A_eq``矩阵中线性相关的行（除非它们表示不可行性），以避免主要求解过程中的数值困难。
    # 需要注意的是，几乎线性相关的行（在预定义的容差范围内）也可能被移除，这在极少数情况下可能会改变最优解。
    # 如果这是一个问题，请从问题表述中消除冗余，并选择``rr=False``或``presolve=False``选项运行。

    # 这里可以进行几个潜在的改进：应实施在[8]_中概述的额外预处理检查，
    # 预处理程序应多次运行（直到不能进一步简化为止），并且应在冗余移除程序中实现更多来自[5]_的效率改进。

    # 在预处理完成后，通过将（收紧的）简单边界转换为上限约束，为不等式约束引入非负松弛变量，
    # 并将无界变量表示为两个非负变量之差，将问题转换为标准形式。
    # 可选地，可以通过均衡化[12]_自动缩放问题。
    # 所选的算法解决标准形式问题，并且后处理程序将结果转换为原始问题的解。

    # 参考文献
    # ----------
    # .. [1] Dantzig, George B., Linear programming and extensions. Rand
    #        Corporation Research Study Princeton Univ. Press, Princeton, NJ,
    #        1963
    # .. [2] Hillier, S.H. and Lieberman, G.J. (1995), "Introduction to
    #        Mathematical Programming", McGraw-Hill, Chapter 4.
    # .. [3] Bland, Robert G. New finite pivoting rules for the simplex method.
    #        Mathematics of Operations Research (2), 1977: pp. 103-107.
    # .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
    #        optimizer for linear programming: an implementation of the
    #        homogeneous algorithm." High performance optimization. Springer US,
    #        2000. 197-232.
    # .. [5] Andersen, Erling D. "Finding all linearly dependent rows in
    #        large-scale linear programming." Optimization Methods and Software
    #        6.3 (1995): 219-227.
    # .. [6] Freund, Robert M. "Primal-Dual Interior-Point Methods for Linear
    #        Programming based on Newton's Method." Unpublished Course Notes,
    #        March 2004. Available 2/25/2017 at
    #        https://ocw.mit.edu/courses/sloan-school-of-management/15-084j-nonlinear-programming-spring-2004/lecture-notes/lec14_int_pt_mthd.pdf
    # 导入 scipy.optimize 中的线性规划函数 linprog
    >>> from scipy.optimize import linprog
    
    # 定义线性规划问题的目标函数系数向量 c
    >>> c = [-1, 4]
    
    # 定义线性规划问题的不等式约束系数矩阵 A
    >>> A = [[-3, 1], [1, 2]]
    
    # 定义线性规划问题的不等式约束右侧向量 b
    >>> b = [6, 4]
    
    # 定义变量 x0 的取值范围 (-∞, +∞)
    >>> x0_bounds = (None, None)
    
    # 定义变量 x1 的取值范围 (-3, +∞)
    >>> x1_bounds = (-3, None)
    
    # 调用 linprog 函数求解线性规划问题，将结果存储在 res 变量中
    >>> res = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds])
    
    # 打印输出最优解对应的目标函数值
    >>> res.fun
    -22.0
    
    # 打印输出最优解向量
    >>> res.x
    array([10., -3.])
    
    # 打印输出优化求解的状态信息
    >>> res.message
    'Optimization terminated successfully. (HiGHS Status 7: Optimal)'
    
    # 输出线性规划问题的不等式约束的残差（slack）和对偶变量（Lagrange 乘子）
    >>> res.ineqlin
    residual: [ 3.900e+01  0.000e+00]
    marginals: [-0.000e+00 -1.000e+00]
    """
    to decrease by ``eps`` if we add a small amount ``eps`` to the right hand
    side of the second inequality constraint:

    >>> eps = 0.05
    >>> b[1] += eps
    >>> linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds]).fun
    -22.05

    Also, because the residual on the first inequality constraint is 39, we
    can decrease the right hand side of the first constraint by 39 without
    affecting the optimal solution.

    >>> b = [6, 4]  # reset to original values
    >>> b[0] -= 39
    >>> linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds]).fun
    -22.0
    """

    # 将方法名称转换为小写
    meth = method.lower()
    # 支持的求解方法集合
    methods = {"highs", "highs-ds", "highs-ipm",
               "simplex", "revised simplex", "interior-point"}

    # 检查方法是否在支持的方法集合中，否则引发值错误
    if meth not in methods:
        raise ValueError(f"Unknown solver '{method}'")

    # 如果指定了初始点并且方法不是'revised simplex'，则发出警告
    if x0 is not None and meth != "revised simplex":
        warning_message = "x0 is used only when method is 'revised simplex'. "
        warn(warning_message, OptimizeWarning, stacklevel=2)

    # 如果存在整数约束并且方法不是'highs'，则忽略整数约束并发出警告；否则根据 c 的形状广播整数约束
    if np.any(integrality) and not meth == "highs":
        integrality = None
        warning_message = ("Only `method='highs'` supports integer "
                           "constraints. Ignoring `integrality`.")
        warn(warning_message, OptimizeWarning, stacklevel=2)
    elif np.any(integrality):
        integrality = np.broadcast_to(integrality, np.shape(c))
    else:
        integrality = None

    # 创建线性规划问题对象
    lp = _LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0, integrality)
    # 解析线性规划问题及求解器选项
    lp, solver_options = _parse_linprog(lp, options, meth)
    tol = solver_options.get('tol', 1e-9)

    # 如果方法以'highs'开头，使用 HiGHS 求解器求解
    if meth.startswith('highs'):
        if callback is not None:
            raise NotImplementedError("HiGHS solvers do not support the "
                                      "callback interface.")
        # 配置不同 HiGHS 求解器的参数
        highs_solvers = {'highs-ipm': 'ipm', 'highs-ds': 'simplex',
                         'highs': None}
        # 调用 HiGHS 求解器求解线性规划问题
        sol = _linprog_highs(lp, solver=highs_solvers[meth],
                             **solver_options)
        # 检查结果并返回优化结果对象
        sol['status'], sol['message'] = (
            _check_result(sol['x'], sol['fun'], sol['status'], sol['slack'],
                          sol['con'], lp.bounds, tol, sol['message'],
                          integrality))
        sol['success'] = sol['status'] == 0
        return OptimizeResult(sol)

    # 发出警告，指明使用过时的方法
    warn(f"`method='{meth}'` is deprecated and will be removed in SciPy "
         "1.11.0. Please use one of the HiGHS solvers (e.g. "
         "`method='highs'`) in new code.", DeprecationWarning, stacklevel=2)

    # 初始化迭代次数和解决状态
    iteration = 0
    complete = False  # will become True if solved in presolve
    undo = []

    # 保留原始数组以计算原始问题的松弛/残差
    lp_o = deepcopy(lp)

    # 解决简单问题，消除变量，收紧边界等
    rr_method = solver_options.pop('rr_method', None)  # 需要弹出这些参数
    rr = solver_options.pop('rr', True)  # 这些参数不会传递给方法
    c0 = 0  # 可能会在目标函数中得到一个常数项
    # 如果 solver_options 中包含 'presolve'，则执行预处理步骤
    if solver_options.pop('presolve', True):
        # 调用 _presolve 函数进行预处理，并返回预处理后的结果
        (lp, c0, x, undo, complete, status, message) = _presolve(lp, rr,
                                                                 rr_method,
                                                                 tol)

    C, b_scale = 1, 1  # 如果未使用 autoscale，用于简单的反缩放
    # 准备 postsolve 需要的参数元组
    postsolve_args = (lp_o._replace(bounds=lp.bounds), undo, C, b_scale)

    # 如果预处理不完整，重新获取线性规划问题的参数
    if not complete:
        # 调用 _get_Abc 函数获取线性规划问题的参数
        A, b, c, c0, x0 = _get_Abc(lp, c0)
        # 如果 solver_options 中包含 'autoscale'，则执行自动缩放
        if solver_options.pop('autoscale', False):
            # 调用 _autoscale 函数对问题进行自动缩放
            A, b, c, x0, C, b_scale = _autoscale(A, b, c, x0)
            # 更新 postsolve_args，包含缩放相关的参数
            postsolve_args = postsolve_args[:-2] + (C, b_scale)

        # 根据选择的求解方法（meth），调用对应的线性规划求解函数
        if meth == 'simplex':
            x, status, message, iteration = _linprog_simplex(
                c, c0=c0, A=A, b=b, callback=callback,
                postsolve_args=postsolve_args, **solver_options)
        elif meth == 'interior-point':
            x, status, message, iteration = _linprog_ip(
                c, c0=c0, A=A, b=b, callback=callback,
                postsolve_args=postsolve_args, **solver_options)
        elif meth == 'revised simplex':
            x, status, message, iteration = _linprog_rs(
                c, c0=c0, A=A, b=b, x0=x0, callback=callback,
                postsolve_args=postsolve_args, **solver_options)

    # 根据求解后的结果，进行后处理
    disp = solver_options.get('disp', False)

    # 调用 _postsolve 函数进行后处理，得到最终解和相关信息
    x, fun, slack, con = _postsolve(x, postsolve_args, complete)

    # 调用 _check_result 函数检查最终结果的有效性和精确度
    status, message = _check_result(x, fun, status, slack, con, lp_o.bounds,
                                    tol, message, integrality)

    # 如果设置了 disp 标志，则显示求解摘要信息
    if disp:
        _display_summary(message, status, fun, iteration)

    # 组装并返回优化结果的字典形式
    sol = {
        'x': x,
        'fun': fun,
        'slack': slack,
        'con': con,
        'status': status,
        'message': message,
        'nit': iteration,
        'success': status == 0}
    
    # 将结果封装成 OptimizeResult 对象并返回
    return OptimizeResult(sol)
```