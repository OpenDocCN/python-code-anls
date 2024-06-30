# `D:\src\scipysrc\scipy\scipy\optimize\_linprog_doc.py`

```
# 定义一个函数，使用高斯求解器最小化线性目标函数，同时满足线性不等式和等式约束
def _linprog_highs_doc(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                       bounds=None, method='highs', callback=None,
                       maxiter=None, disp=False, presolve=True,
                       time_limit=None,
                       dual_feasibility_tolerance=None,
                       primal_feasibility_tolerance=None,
                       ipm_optimality_tolerance=None,
                       simplex_dual_edge_weight_strategy=None,
                       mip_rel_gap=None,
                       **unknown_options):
    r"""
    Linear programming: minimize a linear objective function subject to linear
    equality and inequality constraints using one of the HiGHS solvers.

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

    minimize::

        c @ x

    such that::

        A_ub @ x <= b_ub
        A_eq @ x == b_eq
        lb <= x <= ub

    Note that by default ``lb = 0`` and ``ub = None`` unless specified with
    ``bounds``.

    Parameters
    ----------
    c : 1-D array
        The coefficients of the linear objective function to be minimized.
    A_ub : 2-D array, optional
        The inequality constraint matrix. Each row of ``A_ub`` specifies the
        coefficients of a linear inequality constraint on ``x``.
    b_ub : 1-D array, optional
        The inequality constraint vector. Each element represents an
        upper bound on the corresponding value of ``A_ub @ x``.
    A_eq : 2-D array, optional
        The equality constraint matrix. Each row of ``A_eq`` specifies the
        coefficients of a linear equality constraint on ``x``.
    b_eq : 1-D array, optional
        The equality constraint vector. Each element of ``A_eq @ x`` must equal
        the corresponding element of ``b_eq``.
    bounds : sequence, optional
        A sequence of ``(min, max)`` pairs for each element in ``x``, defining
        the minimum and maximum values of that decision variable. Use ``None``
        to indicate that there is no bound. By default, bounds are
        ``(0, None)`` (all decision variables are non-negative).
        If a single tuple ``(min, max)`` is provided, then ``min`` and
        ``max`` will serve as bounds for all decision variables.
    method : str
        这是针对 'highs' 方法的特定文档，它会自动选择在不同优化方法之间进行切换：
        :ref:`'highs-ds' <optimize.linprog-highs-ds>` 和
        :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`。
        :ref:`'interior-point' <optimize.linprog-interior-point>`（默认），
        :ref:`'revised simplex' <optimize.linprog-revised_simplex>` 和
        :ref:`'simplex' <optimize.linprog-simplex>`（传统方法）也是可用的。

    integrality : 1-D array or int, optional
        指示每个决策变量的整数约束类型。

        ``0``：连续变量；没有整数约束。

        ``1``：整数变量；决策变量必须是 `bounds` 范围内的整数。

        ``2``：半连续变量；决策变量必须在 `bounds` 范围内或者取值为 ``0``。

        ``3``：半整数变量；决策变量必须是 `bounds` 范围内的整数或者取值为 ``0``。

        默认情况下，所有变量都是连续的。

        对于混合整数约束，请提供一个形状为 `c.shape` 的数组。
        如果要从较短的输入推断每个决策变量的约束，则使用 `np.broadcast_to` 将参数广播为 `c.shape`。

        此参数目前仅由 ``'highs'`` 方法使用，否则会被忽略。

    Options
    -------
    maxiter : int
        在任一阶段执行的最大迭代次数。
        对于 :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`，这不包括交叉迭代次数。
        默认值是平台上一个 ``int`` 类型的最大可能值。

    disp : bool (default: ``False``)
        如果要在优化过程中向控制台打印优化状态指示器，则设置为 ``True``。

    presolve : bool (default: ``True``)
        在发送到主求解器之前，预处理试图识别无关紧要的不可行性，
        识别无关紧要的无界性，并简化问题。
        通常建议保持默认设置 ``True``；如果要禁用预处理，则设置为 ``False``。

    time_limit : float
        分配给解决问题的最大时间（以秒为单位）；
        默认值是平台上一个 ``double`` 类型的最大可能值。

    dual_feasibility_tolerance : double (default: 1e-07)
        对于 :ref:`'highs-ds' <optimize.linprog-highs-ds>` 的双重可行性容限。
        此值与 ``primal_feasibility_tolerance`` 的最小值一起用于
        :ref:`'highs-ipm' <optimize.linprog-highs-ipm>` 的可行性容限。
    primal_feasibility_tolerance : double (default: 1e-07)
        # 设置原始可行性容差，用于 'highs-ds' 优化器
        # 此值与 dual_feasibility_tolerance 取较小者，作为 'highs-ipm' 的可行性容差
        The minimum of this and ``dual_feasibility_tolerance``
        is used for the feasibility tolerance of
        :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`.

    ipm_optimality_tolerance : double (default: ``1e-08``)
        # 设置 'highs-ipm' 优化器的最优性容差
        # 最小允许值为 1e-12
        Optimality tolerance for
        :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`.

    simplex_dual_edge_weight_strategy : str (default: None)
        # 设置单纯形双重边缘权重策略
        # 默认为 None，自动选择以下策略之一
        ``'dantzig'`` uses Dantzig's original strategy of choosing the most
        negative reduced cost.

        ``'devex'`` uses the strategy described in [15]_.

        ``steepest`` uses the exact steepest edge strategy as described in
        [16]_.

        ``'steepest-devex'`` begins with the exact steepest edge strategy
        until the computation is too costly or inexact and then switches to
        the devex method.

        Currently, ``None`` always selects ``'steepest-devex'``, but this
        may change as new options become available.

    mip_rel_gap : double (default: None)
        # 设置 MIP 求解器的终止准则
        # 当原始目标值与对偶目标界之间的缩放值 <= mip_rel_gap 时终止求解
        Termination criterion for MIP solver: solver will terminate when the
        gap between the primal objective value and the dual objective bound,
        scaled by the primal objective value, is <= mip_rel_gap.

    unknown_options : dict
        # 未使用的可选参数
        # 如果 ``unknown_options`` 非空，则发出警告，列出所有未使用的选项
        Optional arguments not used by this particular solver. If
        ``unknown_options`` is non-empty, a warning is issued listing
        all unused options.
    # 这是一个空函数，用于占位，暂时没有具体的实现代码。
    """
    References
    ----------
    .. [13] Huangfu, Q., Galabova, I., Feldmeier, M., and Hall, J. A. J.
           "HiGHS - high performance software for linear optimization."
           https://highs.dev/
    .. [14] Huangfu, Q. and Hall, J. A. J. "Parallelizing the dual revised
           simplex method." Mathematical Programming Computation, 10 (1),
           119-142, 2018. DOI: 10.1007/s12532-017-0130-5
    .. [15] Harris, Paula MJ. "Pivot selection methods of the Devex LP code."
            Mathematical programming 5.1 (1973): 1-28.
    .. [16] Goldfarb, Donald, and John Ker Reid. "A practicable steepest-edge
            simplex algorithm." Mathematical Programming 12.1 (1977): 361-371.
    """
def _linprog_highs_ds_doc(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                          bounds=None, method='highs-ds', callback=None,
                          maxiter=None, disp=False, presolve=True,
                          time_limit=None,
                          dual_feasibility_tolerance=None,
                          primal_feasibility_tolerance=None,
                          simplex_dual_edge_weight_strategy=None,
                          **unknown_options):
    r"""
    Linear programming: minimize a linear objective function subject to linear
    equality and inequality constraints using the HiGHS dual simplex solver.

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

    minimize::

        c @ x

    such that::

        A_ub @ x <= b_ub
        A_eq @ x == b_eq
        lb <= x <= ub

    Note that by default ``lb = 0`` and ``ub = None`` unless specified with
    ``bounds``.

    Parameters
    ----------
    c : 1-D array
        The coefficients of the linear objective function to be minimized.
    A_ub : 2-D array, optional
        The inequality constraint matrix. Each row of ``A_ub`` specifies the
        coefficients of a linear inequality constraint on ``x``.
    b_ub : 1-D array, optional
        The inequality constraint vector. Each element represents an
        upper bound on the corresponding value of ``A_ub @ x``.
    A_eq : 2-D array, optional
        The equality constraint matrix. Each row of ``A_eq`` specifies the
        coefficients of a linear equality constraint on ``x``.
    b_eq : 1-D array, optional
        The equality constraint vector. Each element of ``A_eq @ x`` must equal
        the corresponding element of ``b_eq``.
    bounds : sequence, optional
        A sequence of ``(min, max)`` pairs for each element in ``x``, defining
        the minimum and maximum values of that decision variable. Use ``None``
        to indicate that there is no bound. By default, bounds are
        ``(0, None)`` (all decision variables are non-negative).
        If a single tuple ``(min, max)`` is provided, then ``min`` and
        ``max`` will serve as bounds for all decision variables.
    method : str
        Specifies the method for solving the linear programming problem. 'highs-ds'
        indicates the use of the HiGHS dual simplex solver.

    Options
    -------
    callback : function, optional
        If provided, this callback function will be called after each iteration
        of the simplex algorithm.
    maxiter : int, optional
        Maximum number of iterations to perform.
    disp : bool, optional
        Set to True to print convergence messages.
    presolve : bool, optional
        Set to False to disable presolve.
    time_limit : float, optional
        Maximum solving time in seconds.
    dual_feasibility_tolerance : float, optional
        Dual feasibility tolerance.
    primal_feasibility_tolerance : float, optional
        Primal feasibility tolerance.
    simplex_dual_edge_weight_strategy : str, optional
        Strategy for dual edge weights in the simplex algorithm.
    **unknown_options : dict, optional
        Additional options specific to the solver.

    """
    # 最大迭代次数，用于优化过程的最大迭代次数限制
    maxiter : int
        The maximum number of iterations to perform in either phase.
        Default is the largest possible value for an ``int`` on the platform.
    
    # 是否显示优化状态指示器的布尔值，默认为 ``False``
    disp : bool (default: ``False``)
        Set to ``True`` if indicators of optimization status are to be
        printed to the console during optimization.
    
    # 是否进行预处理的布尔值，默认为 ``True``
    presolve : bool (default: ``True``)
        Presolve attempts to identify trivial infeasibilities,
        identify trivial unboundedness, and simplify the problem before
        sending it to the main solver. It is generally recommended
        to keep the default setting ``True``; set to ``False`` if
        presolve is to be disabled.
    
    # 解决问题的最大时间限制，单位为秒，默认为平台上 ``double`` 类型的最大可能值
    time_limit : float
        The maximum time in seconds allotted to solve the problem;
        default is the largest possible value for a ``double`` on the
        platform.
    
    # 对于 'highs-ds' 方法的双重可行性容忍度，默认为 1e-07
    dual_feasibility_tolerance : double (default: 1e-07)
        Dual feasibility tolerance for
        :ref:`'highs-ds' <optimize.linprog-highs-ds>`.
    
    # 对于 'highs-ds' 方法的原始可行性容忍度，默认为 1e-07
    primal_feasibility_tolerance : double (default: 1e-07)
        Primal feasibility tolerance for
        :ref:`'highs-ds' <optimize.linprog-highs-ds>`.
    
    # 简单型双重边缘权重策略的选择
    simplex_dual_edge_weight_strategy : str (default: None)
        Strategy for simplex dual edge weights. The default, ``None``,
        automatically selects one of the following.
        
        ``'dantzig'`` uses Dantzig's original strategy of choosing the most
        negative reduced cost.
        
        ``'devex'`` uses the strategy described in [15]_.
        
        ``steepest`` uses the exact steepest edge strategy as described in
        [16]_.
        
        ``'steepest-devex'`` begins with the exact steepest edge strategy
        until the computation is too costly or inexact and then switches to
        the devex method.
        
        Currently, ``None`` always selects ``'steepest-devex'``, but this
        may change as new options become available.
    
    # 未知选项的字典，对于此特定求解器不使用的可选参数
    unknown_options : dict
        Optional arguments not used by this particular solver. If
        ``unknown_options`` is non-empty, a warning is issued listing
        all unused options.
    
    # 返回值说明
    Returns
    -------
    Notes
    -----
    
    # 'highs-ds' 方法是 C++ 高性能双重修正单纯形实现 (HSOL) 的包装
    Method :ref:`'highs-ds' <optimize.linprog-highs-ds>` is a wrapper
    of the C++ high performance dual revised simplex implementation (HSOL)
    [13]_, [14]_.
    
    # 'highs-ipm' 方法是 C++ 实现的内点法的包装
    Method :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`
    is a wrapper of a C++ implementation of an **i**\ nterior-\ **p**\ oint
    **m**\ ethod [13]_; it features a crossover routine, so it is as accurate
    as a simplex solver.
    
    # 'highs' 方法会自动在 'highs-ds' 和 'highs-ipm' 方法之间选择
    Method :ref:`'highs' <optimize.linprog-highs>` chooses
    between the two automatically. For new code involving `linprog`, we
    recommend explicitly choosing one of these three method values instead of
    :ref:`'interior-point' <optimize.linprog-interior-point>` (default),
    :ref:`'revised simplex' <optimize.linprog-revised_simplex>`, and
    :ref:`'simplex' <optimize.linprog-simplex>` (legacy).
    The result fields `ineqlin`, `eqlin`, `lower`, and `upper` all contain
    `marginals`, or partial derivatives of the objective function with respect
    to the right-hand side of each constraint. These partial derivatives are
    also referred to as "Lagrange multipliers", "dual values", and
    "shadow prices". The sign convention of `marginals` is opposite that
    of Lagrange multipliers produced by many nonlinear solvers.

    References
    ----------
    .. [13] Huangfu, Q., Galabova, I., Feldmeier, M., and Hall, J. A. J.
           "HiGHS - high performance software for linear optimization."
           https://highs.dev/
    .. [14] Huangfu, Q. and Hall, J. A. J. "Parallelizing the dual revised
           simplex method." Mathematical Programming Computation, 10 (1),
           119-142, 2018. DOI: 10.1007/s12532-017-0130-5
    .. [15] Harris, Paula MJ. "Pivot selection methods of the Devex LP code."
            Mathematical programming 5.1 (1973): 1-28.
    .. [16] Goldfarb, Donald, and John Ker Reid. "A practicable steepest-edge
            simplex algorithm." Mathematical Programming 12.1 (1977): 361-371.
    """
    这部分代码用于说明变量 `ineqlin`、`eqlin`、`lower` 和 `upper` 中的值，它们都包含了目标函数对于每个约束右侧的偏导数，也称为“拉格朗日乘子”、“对偶值”和“影子价格”。`marginals` 的符号约定与许多非线性求解器产生的拉格朗日乘子相反。
    包含了关于这些概念的文献引用，详细说明了相关概念及其应用背景。
    pass
def _linprog_highs_ipm_doc(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                           bounds=None, method='highs-ipm', callback=None,
                           maxiter=None, disp=False, presolve=True,
                           time_limit=None,
                           dual_feasibility_tolerance=None,
                           primal_feasibility_tolerance=None,
                           ipm_optimality_tolerance=None,
                           **unknown_options):
    r"""
    Linear programming: minimize a linear objective function subject to linear
    equality and inequality constraints using the HiGHS interior point solver.

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

    minimize::

        c @ x

    such that::

        A_ub @ x <= b_ub
        A_eq @ x == b_eq
        lb <= x <= ub

    Note that by default ``lb = 0`` and ``ub = None`` unless specified with
    ``bounds``.

    Parameters
    ----------
    c : 1-D array
        The coefficients of the linear objective function to be minimized.
    A_ub : 2-D array, optional
        The inequality constraint matrix. Each row of ``A_ub`` specifies the
        coefficients of a linear inequality constraint on ``x``.
    b_ub : 1-D array, optional
        The inequality constraint vector. Each element represents an
        upper bound on the corresponding value of ``A_ub @ x``.
    A_eq : 2-D array, optional
        The equality constraint matrix. Each row of ``A_eq`` specifies the
        coefficients of a linear equality constraint on ``x``.
    b_eq : 1-D array, optional
        The equality constraint vector. Each element of ``A_eq @ x`` must equal
        the corresponding element of ``b_eq``.
    bounds : sequence, optional
        A sequence of ``(min, max)`` pairs for each element in ``x``, defining
        the minimum and maximum values of that decision variable. Use ``None``
        to indicate that there is no bound. By default, bounds are
        ``(0, None)`` (all decision variables are non-negative).
        If a single tuple ``(min, max)`` is provided, then ``min`` and
        ``max`` will serve as bounds for all decision variables.
    method : str
        Specifies the optimization method. Default is 'highs-ipm', which uses
        the HiGHS interior point solver.

        Other methods available are:
        - 'highs-ds'
        - 'interior-point'
        - 'revised simplex'
        - 'simplex' (legacy)

    callback : callable, optional
        If specified, this function will be called after each iteration of the
        optimization process with the current state.
    maxiter : int, optional
        Maximum number of iterations to perform.
    disp : bool, optional
        Set to True to print convergence messages.
    presolve : bool, optional
        Whether to use presolve to simplify the problem before solving.
    time_limit : float, optional
        Maximum time allowed in seconds for the solver to run.
    dual_feasibility_tolerance : float, optional
        Tolerance for dual feasibility in the solver.
    primal_feasibility_tolerance : float, optional
        Tolerance for primal feasibility in the solver.
    ipm_optimality_tolerance : float, optional
        Tolerance for the optimality condition in the interior point solver.
    **unknown_options : dict, optional
        Additional options specific to the solver can be passed as keyword
        arguments.

    """
    Options
    -------
    maxiter : int
        The maximum number of iterations to perform in either phase.
        For :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`, this does not
        include the number of crossover iterations. Default is the largest
        possible value for an ``int`` on the platform.
    disp : bool (default: ``False``)
        Set to ``True`` if indicators of optimization status are to be
        printed to the console during optimization.
    presolve : bool (default: ``True``)
        Presolve attempts to identify trivial infeasibilities,
        identify trivial unboundedness, and simplify the problem before
        sending it to the main solver. It is generally recommended
        to keep the default setting ``True``; set to ``False`` if
        presolve is to be disabled.
    time_limit : float
        The maximum time in seconds allotted to solve the problem;
        default is the largest possible value for a ``double`` on the
        platform.
    dual_feasibility_tolerance : double (default: 1e-07)
        The minimum of this and ``primal_feasibility_tolerance``
        is used for the feasibility tolerance of
        :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`.
    primal_feasibility_tolerance : double (default: 1e-07)
        The minimum of this and ``dual_feasibility_tolerance``
        is used for the feasibility tolerance of
        :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`.
    ipm_optimality_tolerance : double (default: ``1e-08``)
        Optimality tolerance for
        :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`.
        Minimum allowable value is 1e-12.
    unknown_options : dict
        Optional arguments not used by this particular solver. If
        ``unknown_options`` is non-empty, a warning is issued listing
        all unused options.

    Returns
    -------
    Notes
    -----

    Method :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`
    is a wrapper of a C++ implementation of an **i**\ nterior-\ **p**\ oint
    **m**\ ethod [13]_; it features a crossover routine, so it is as accurate
    as a simplex solver.
    Method :ref:`'highs-ds' <optimize.linprog-highs-ds>` is a wrapper
    of the C++ high performance dual revised simplex implementation (HSOL)
    [13]_, [14]_. Method :ref:`'highs' <optimize.linprog-highs>` chooses
    between the two automatically. For new code involving `linprog`, we
    recommend explicitly choosing one of these three method values instead of
    :ref:`'interior-point' <optimize.linprog-interior-point>` (default),
    :ref:`'revised simplex' <optimize.linprog-revised_simplex>`, and
    :ref:`'simplex' <optimize.linprog-simplex>` (legacy).

    The result fields `ineqlin`, `eqlin`, `lower`, and `upper` all contain
    `marginals`, or partial derivatives of the objective function with respect
    to the right-hand side of each constraint. These partial derivatives are
    also referred to as "Lagrange multipliers", "dual values", and
    "shadow prices". The sign convention of `marginals` is opposite that
    of Lagrange multipliers produced by many nonlinear solvers.

    References
    ----------
    .. [13] Huangfu, Q., Galabova, I., Feldmeier, M., and Hall, J. A. J.
           "HiGHS - high performance software for linear optimization."
           https://highs.dev/
    .. [14] Huangfu, Q. and Hall, J. A. J. "Parallelizing the dual revised
           simplex method." Mathematical Programming Computation, 10 (1),
           119-142, 2018. DOI: 10.1007/s12532-017-0130-5
# 定义线性规划的内部文档函数，用于最小化线性目标函数，同时满足线性不等式和等式约束，使用内点法 [4]_ 方法
def _linprog_ip_doc(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                    bounds=None, method='interior-point', callback=None,
                    maxiter=1000, disp=False, presolve=True,
                    tol=1e-8, autoscale=False, rr=True,
                    alpha0=.99995, beta=0.1, sparse=False,
                    lstsq=False, sym_pos=True, cholesky=True, pc=True,
                    ip=False, permc_spec='MMD_AT_PLUS_A', **unknown_options):
    r"""
    线性规划：使用 [4]_ 的内点法最小化线性目标函数，同时满足线性等式和不等式约束。

    .. deprecated:: 1.9.0
        `method='interior-point'` 将在 SciPy 1.11.0 版本中移除。
        它将被 `method='highs'` 取代，后者更快速和更稳健。

    线性规划解决如下形式的问题：

    .. math::

        \min_x \ & c^T x \\
        \mbox{subject to} \ & A_{ub} x \leq b_{ub},\\
        & A_{eq} x = b_{eq},\\
        & l \leq x \leq u ,

    其中 :math:`x` 是决策变量向量；:math:`c`, :math:`b_{ub}`, :math:`b_{eq}`, :math:`l` 和 :math:`u` 是向量；
    :math:`A_{ub}` 和 :math:`A_{eq}` 是矩阵。

    或者说：

    最小化::

        c @ x

    满足条件::

        A_ub @ x <= b_ub
        A_eq @ x == b_eq
        lb <= x <= ub

    注意，默认情况下 `lb = 0`，`ub = None`，除非通过 `bounds` 指定。

    Parameters
    ----------
    c : 1-D array
        要最小化的线性目标函数的系数。
    A_ub : 2-D array, optional
        不等式约束矩阵。每行 `A_ub` 指定 `x` 的线性不等式约束的系数。
    b_ub : 1-D array, optional
        不等式约束向量。每个元素表示对应的 `A_ub @ x` 的上界。
    A_eq : 2-D array, optional
        等式约束矩阵。每行 `A_eq` 指定 `x` 的线性等式约束的系数。
    b_eq : 1-D array, optional
        等式约束向量。`A_eq @ x` 的每个元素必须等于 `b_eq` 的对应元素。
    bounds : sequence, optional
        每个决策变量 `x` 的 `(min, max)` 对的序列，定义该决策变量的最小和最大值。使用 `None` 表示无界。
        默认情况下，边界是 `(0, None)`（所有决策变量为非负数）。
        如果提供单个元组 `(min, max)`，则 `min` 和 `max` 将作为所有决策变量的边界。

    method : str, optional
        解决线性规划问题的方法。默认为 `'interior-point'`，即内点法。

    callback : callable, optional
        在迭代过程中每次迭代时调用的可调用对象，接受一个参数：一个表示当前状态的 `OptimizeResult` 对象。

    maxiter : int, optional
        允许的最大迭代次数。

    disp : bool, optional
        是否显示优化过程的详细信息。

    presolve : bool, optional
        是否进行预处理。

    tol : float, optional
        容忍度。

    autoscale : bool, optional
        是否自动缩放问题。

    rr : bool, optional
        是否应用矫正策略。

    alpha0 : float, optional
        用于计算目标函数的初始点的参数。

    beta : float, optional
        用于控制搜索方向的参数。

    sparse : bool, optional
        是否使用稀疏矩阵。

    lstsq : bool, optional
        是否使用最小二乘法。

    sym_pos : bool, optional
        是否对矩阵进行正定性检查。

    cholesky : bool, optional
        是否使用 Cholesky 分解。

    pc : bool, optional
        是否使用预处理。

    ip : bool, optional
        是否使用内点法。

    permc_spec : str, optional
        指定置换策略。

    **unknown_options : dict, optional
        未知选项。

    Returns
    -------
    None
    """
    pass
    # 方法名，指定线性规划求解方法为 'interior-point'，可选的方法包括 'highs', 'highs-ds', 'highs-ipm',
    # 'revised simplex', 和 'simplex'（已过时）
    method : str
        This is the method-specific documentation for 'interior-point'.
        :ref:`'highs' <optimize.linprog-highs>`,
        :ref:`'highs-ds' <optimize.linprog-highs-ds>`,
        :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`,
        :ref:`'revised simplex' <optimize.linprog-revised_simplex>`, and
        :ref:`'simplex' <optimize.linprog-simplex>` (legacy)
        are also available.
    
    # 可选参数，用于每次迭代执行的回调函数
    callback : callable, optional
    
    # 选项
    -------
    
    # 最大迭代次数，默认为 1000
    maxiter : int (default: 1000)
        The maximum number of iterations of the algorithm.
    
    # 是否在每次迭代时将优化状态指示输出到控制台，默认为 False
    disp : bool (default: False)
        Set to ``True`` if indicators of optimization status are to be printed
        to the console each iteration.
    
    # 是否启用预处理，默认为 True；预处理尝试识别平凡的不可行性、平凡的无界性，并在将问题发送给主求解器之前简化问题
    presolve : bool (default: True)
        Presolve attempts to identify trivial infeasibilities,
        identify trivial unboundedness, and simplify the problem before
        sending it to the main solver. It is generally recommended
        to keep the default setting ``True``; set to ``False`` if
        presolve is to be disabled.
    
    # 所有终止准则使用的终止容差，默认为 1e-8；参见 [4]_ 第 4.5 节
    tol : float (default: 1e-8)
        Termination tolerance to be used for all termination criteria;
        see [4]_ Section 4.5.
    
    # 设置为 True 时，自动执行均衡操作；如果约束中的数值相差几个数量级，则考虑使用此选项
    autoscale : bool (default: False)
        Set to ``True`` to automatically perform equilibration.
        Consider using this option if the numerical values in the
        constraints are separated by several orders of magnitude.
    
    # 设置为 False 以禁用自动冗余移除，默认为 True
    rr : bool (default: True)
        Set to ``False`` to disable automatic redundancy removal.
    
    # Mehrota 的预测-校正搜索方向的最大步长；参见 [4]_ 表 8.1 中的 :math:`\beta_{3}`
    alpha0 : float (default: 0.99995)
        The maximal step size for Mehrota's predictor-corrector search
        direction; see :math:`\beta_{3}` of [4]_ Table 8.1.
    
    # 当 Mehrota 的预测-校正未使用时，路径参数 :math:`\mu` 的期望减少量；参见 [6]_
    beta : float (default: 0.1)
        The desired reduction of the path parameter :math:`\mu` (see [6]_)
        when Mehrota's predictor-corrector is not in use (uncommon).
    
    # 如果问题经过预处理后要被视为稀疏问题，则设置为 True；如果 A_eq 或 A_ub 是稀疏矩阵之一，则此选项将自动设置为 True
    sparse : bool (default: False)
        Set to ``True`` if the problem is to be treated as sparse after
        presolve. If either ``A_eq`` or ``A_ub`` is a sparse matrix,
        this option will automatically be set ``True``, and the problem
        will be treated as sparse even during presolve. If your constraint
        matrices contain mostly zeros and the problem is not very small (less
        than about 100 constraints or variables), consider setting ``True``
        or providing ``A_eq`` and ``A_ub`` as sparse matrices.
    
    # 如果问题预计条件非常差，则设置为 True；除非遇到严重的数值困难，否则应该保持 False
    lstsq : bool (default: ``False``)
        Set to ``True`` if the problem is expected to be very poorly
        conditioned. This should always be left ``False`` unless severe
        numerical difficulties are encountered. Leave this at the default
        unless you receive a warning message suggesting otherwise.
    sym_pos : bool (default: True)
        # 是否假定问题会生成一个条件良好的对称正定法方程矩阵，默认为 True。
        # 除非接收到警告信息建议修改，一般不需要改变这个值。

    cholesky : bool (default: True)
        # 如果要通过明确的 Cholesky 分解解决法方程组，然后进行前向/后向替换，设置为 True。
        # 对于数值行为良好的问题，这通常更快。

    pc : bool (default: True)
        # 如果要使用 Mehrota 的预测-校正方法，保持为 True。
        # 几乎总是（如果不是总是）有利的。

    ip : bool (default: False)
        # 如果希望使用改进的初始点建议（来源于 [4] 章节 4.3），设置为 True。
        # 这取决于问题本身，有时可能是有利的。

    permc_spec : str (default: 'MMD_AT_PLUS_A')
        # （仅在 sparse=True，lstsq=False，sym_pos=True，且无 SuiteSparse 时有效）
        # 算法的每次迭代中需要对矩阵进行因子分解。
        # 此选项指定如何置换矩阵的列以保持稀疏性。可接受的值包括：
        # - "NATURAL": 自然顺序。
        # - "MMD_ATA": A^T A 结构的最小度数排序。
        # - "MMD_AT_PLUS_A": A^T + A 结构的最小度数排序。
        # - "COLAMD": 近似最小度数列排序。
        # 此选项可能影响内点算法的收敛性；测试不同的值以确定哪个对您的问题表现最好。
        # 更多信息请参考 scipy.sparse.linalg.splu。

    unknown_options : dict
        # 此求解器不使用的可选参数。如果 unknown_options 非空，则会发出警告列出所有未使用的选项。

    Returns
    -------
    res : OptimizeResult
        A :class:`scipy.optimize.OptimizeResult` consisting of the fields:

        x : 1-D array
            The values of the decision variables that minimizes the
            objective function while satisfying the constraints.
        fun : float
            The optimal value of the objective function ``c @ x``.
        slack : 1-D array
            The (nominally positive) values of the slack variables,
            ``b_ub - A_ub @ x``.
        con : 1-D array
            The (nominally zero) residuals of the equality constraints,
            ``b_eq - A_eq @ x``.
        success : bool
            ``True`` when the algorithm succeeds in finding an optimal
            solution.
        status : int
            An integer representing the exit status of the algorithm.

            ``0`` : Optimization terminated successfully.

            ``1`` : Iteration limit reached.

            ``2`` : Problem appears to be infeasible.

            ``3`` : Problem appears to be unbounded.

            ``4`` : Numerical difficulties encountered.

        message : str
            A string descriptor of the exit status of the algorithm.
        nit : int
            The total number of iterations performed in all phases.


    Notes
    -----
    This method implements the algorithm outlined in [4]_ with ideas from [8]_
    and a structure inspired by the simpler methods of [6]_.

    The primal-dual path following method begins with initial 'guesses' of
    the primal and dual variables of the standard form problem and iteratively
    attempts to solve the (nonlinear) Karush-Kuhn-Tucker conditions for the
    problem with a gradually reduced logarithmic barrier term added to the
    objective. This particular implementation uses a homogeneous self-dual
    formulation, which provides certificates of infeasibility or unboundedness
    where applicable.

    The default initial point for the primal and dual variables is that
    defined in [4]_ Section 4.4 Equation 8.22. Optionally (by setting initial
    point option ``ip=True``), an alternate (potentially improved) starting
    point can be calculated according to the additional recommendations of
    [4]_ Section 4.4.

    A search direction is calculated using the predictor-corrector method
    (single correction) proposed by Mehrota and detailed in [4]_ Section 4.1.
    (A potential improvement would be to implement the method of multiple
    corrections described in [4]_ Section 4.2.) In practice, this is
    accomplished by solving the normal equations, [4]_ Section 5.1 Equations
    8.31 and 8.32, derived from the Newton equations [4]_ Section 5 Equations
    8.25 (compare to [4]_ Section 4 Equations 8.6-8.8). The advantage of
    solving the normal equations rather than 8.25 directly is that the
    matrices involved are symmetric positive definite, so Cholesky
    decomposition can be used rather than the more expensive LU factorization.
    # 在默认选项下，求解器的选择取决于第三方软件的可用性以及问题的条件。
    
    # 对于密集问题，求解器按以下顺序尝试：
    
    # 1. 使用 ``scipy.linalg.cho_factor`` 进行因式分解
    
    # 2. 使用 ``scipy.linalg.solve`` 并设置选项 ``sym_pos=True``
    
    # 3. 使用 ``scipy.linalg.solve`` 并设置选项 ``sym_pos=False``
    
    # 4. 使用 ``scipy.linalg.lstsq`` 进行最小二乘求解
    
    # 对于稀疏问题：
    
    # 1. 如果安装了 scikit-sparse 和 SuiteSparse，使用 ``sksparse.cholmod.cholesky``
    
    # 2. 如果安装了 scikit-umfpack 和 SuiteSparse，使用 ``scipy.sparse.linalg.factorized``
    
    # 3. 使用 ``scipy.sparse.linalg.splu``，它使用 SciPy 提供的 SuperLU
    
    # 4. 使用 ``scipy.sparse.linalg.lsqr`` 进行最小二乘求解
    
    # 如果求解器由于任何原因失败，将按照指定顺序尝试更加健壮（但更慢）的求解器。
    # 由于尝试、失败和重新开始因式分解可能耗时较长，因此如果问题在数值上具有挑战性，
    # 可以设置选项来跳过失败的求解器。设置 ``cholesky=False`` 跳过到第二个求解器，
    # ``sym_pos=False`` 跳过到第三个求解器，``lstsq=True`` 跳过到第四个求解器，
    # 这对密集和稀疏问题都适用。
    
    # [4]_ 第 5.3 节和 [10]_ 第 4.1-4.2 节提出了改进处理稀疏问题中的稠密列的方法；
    # 后者还讨论了解放变量替换方法所带来的精度问题的缓解方法。
    
    # 计算搜索方向后，计算不激活非负约束条件的最大可能步长，并应用这个步长和 1 之间的较小值
    # （如 [4]_ 第 4.1 节中所述）。[4]_ 第 4.3 节建议了选择步长的改进方法。
    
    # 根据 [4]_ 第 4.5 节的终止条件测试新点。所有检查都使用可以通过 ``tol`` 选项设置的同一容差。
    # （一个潜在的改进是公开不同的容差，可以独立设置。）如果检测到最优性、无界性或不可行性，
    # 解决过程终止；否则重复。
    
    # ``linprog`` 模块顶层期望问题的形式如下：
    
    # 最小化::
    
    #     c @ x
    
    # Subject to::
    
    #     A_ub @ x <= b_ub
    #     A_eq @ x == b_eq
    #     lb <= x <= ub
    
    # 其中 ``lb = 0`` 而 ``ub = None``，除非在 ``bounds`` 中设置。问题会自动转换为以下标准形式：
    
    # 最小化::
    
    #     c @ x
    
    # Subject to::
    
    #     A @ x == b
    #         x >= 0
    
    # 也就是说，原始问题包含等式、上界约束和变量约束，而特定方法的求解器要求等式约束和变量非负性。
    # ``linprog`` 通过将简单边界转换为上界来将原始问题转换为标准形式。
    # bound constraints, introducing non-negative slack variables for inequality
    # constraints, and expressing unbounded variables as the difference between
    # two non-negative variables. The problem is converted back to the original
    # form before results are reported.

    # References
    # ----------
    # .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
    #        optimizer for linear programming: an implementation of the
    #        homogeneous algorithm." High performance optimization. Springer US,
    #        2000. 197-232.
    # .. [6] Freund, Robert M. "Primal-Dual Interior-Point Methods for Linear
    #        Programming based on Newton's Method." Unpublished Course Notes,
    #        March 2004. Available 2/25/2017 at
    #        https://ocw.mit.edu/courses/sloan-school-of-management/15-084j-nonlinear-programming-spring-2004/lecture-notes/lec14_int_pt_mthd.pdf
    # .. [8] Andersen, Erling D., and Knud D. Andersen. "Presolving in linear
    #        programming." Mathematical Programming 71.2 (1995): 221-245.
    # .. [9] Bertsimas, Dimitris, and J. Tsitsiklis. "Introduction to linear
    #        programming." Athena Scientific 1 (1997): 997.
    # .. [10] Andersen, Erling D., et al. Implementation of interior point
    #         methods for large scale linear programming. HEC/Universite de
    #         Geneve, 1996.
    """
    pass
    ```
# 定义一个线性规划函数，使用修正单纯形法最小化线性目标函数，同时满足线性不等式和等式约束
def _linprog_rs_doc(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                    bounds=None, method='interior-point', callback=None,
                    x0=None, maxiter=5000, disp=False, presolve=True,
                    tol=1e-12, autoscale=False, rr=True, maxupdate=10,
                    mast=False, pivot="mrc", **unknown_options):
    """
    线性规划：使用修正单纯形法最小化线性目标函数，同时满足线性不等式和等式约束。

    .. deprecated:: 1.9.0
        `method='revised simplex'` 在 SciPy 1.11.0 中将被移除。
        它被 `method='highs'` 替代，因为后者更快且更稳健。

    线性规划解决以下形式的问题：

    minimize::
    
        c @ x

    subject to::

        A_ub @ x <= b_ub
        A_eq @ x == b_eq
        lb <= x <= ub

    其中：x 是决策变量向量；c, b_ub, b_eq, l, u 是向量；A_ub 和 A_eq 是矩阵。

    Parameters
    ----------
    c : 1-D array
        要最小化的线性目标函数的系数。
    A_ub : 2-D array, optional
        不等式约束矩阵。每行指定对 x 的线性不等式约束的系数。
    b_ub : 1-D array, optional
        不等式约束向量。每个元素表示对应的 A_ub @ x 的上界。
    A_eq : 2-D array, optional
        等式约束矩阵。每行指定对 x 的线性等式约束的系数。
    b_eq : 1-D array, optional
        等式约束向量。A_eq @ x 的每个元素必须等于 b_eq 中对应的元素。
    bounds : sequence, optional
        每个决策变量 x 的 ``(min, max)`` 对序列，定义该决策变量的最小值和最大值。使用 ``None`` 表示没有边界。
        默认情况下，边界是 ``(0, None)``（所有决策变量均为非负）。
        如果提供单个元组 ``(min, max)``，则 ``min`` 和 ``max`` 将作为所有决策变量的边界。
    method : str, optional
        求解方法，默认为 'interior-point'。
    callback : callable, optional
        每次迭代的回调函数，接受当前解 x 和目标函数值 f(x) 作为参数。
    x0 : 1-D array, optional
        初始猜测解。
    maxiter : int, optional
        最大迭代次数，默认为 5000。
    disp : bool, optional
        是否显示求解过程中的信息，默认为 False。
    presolve : bool, optional
        是否预处理问题，默认为 True。
    tol : float, optional
        容许误差，默认为 1e-12。
    autoscale : bool, optional
        是否自动缩放问题，默认为 False。
    rr : bool, optional
        是否使用限制调整，默认为 True。
    maxupdate : int, optional
        每次更新的最大限制数，默认为 10。
    mast : bool, optional
        是否使用加强技术，默认为 False。
    pivot : str, optional
        管理变量的方式，默认为 "mrc"。
    **unknown_options :
        其它未知参数，将传递给求解器。

    Returns
    -------
    OptimizeResult
        最优解和相关信息的对象。
    """
    # 实现线性规划求解的具体逻辑
    pass
    method : str
        这是'revised simplex'方法的特定文档说明。
        还有其他可用的方法包括：':ref:`'highs' <optimize.linprog-highs>',
        ':ref:`'highs-ds' <optimize.linprog-highs-ds>',
        ':ref:`'highs-ipm' <optimize.linprog-highs-ipm>',
        ':ref:`'interior-point' <optimize.linprog-interior-point>` (默认)，
        和':ref:`'simplex' <optimize.linprog-simplex>` (旧版)。
    callback : callable, optional
        每次迭代执行的回调函数。
    x0 : 1-D array, optional
        决策变量的初始猜测值，将会被优化算法细化。只有'revised simplex'方法当前使用，并且只能在`x0`表示基本可行解时使用。

    Options
    -------
    maxiter : int (default: 5000)
       执行的最大迭代次数，无论是在哪个阶段。
    disp : bool (default: False)
        如果要在每次迭代时将优化状态指示器打印到控制台，则设置为``True``。
    presolve : bool (default: True)
        预处理尝试在将问题发送到主求解器之前识别显而易见的不可行性或无界性，并简化问题。
        通常建议保持默认设置``True``；如果要禁用预处理，则设置为``False``。
    tol : float (default: 1e-12)
        决定何时将解视为“足够接近”零以被视为基本可行解或足够接近正数以被视为最优解的容差。
    autoscale : bool (default: False)
        设置为``True``以自动进行均衡化。
        如果约束中的数值相差几个数量级，考虑使用此选项。
    rr : bool (default: True)
        设置为``False``以禁用自动冗余移除。
    maxupdate : int (default: 10)
        在LU分解上执行的最大更新次数。
        达到此更新次数后，基础矩阵将从头开始进行因式分解。
    mast : bool (default: False)
        最小化摊销求解时间。
        如果启用，将测量使用基础因式分解求解线性系统的平均时间。
        通常，初始因式分解比求解操作（和更新）需要更多时间。
        然而，随着时间的推移，更新的因式分解变得足够复杂，平均求解时间开始增加。
        检测到这种情况后，将从头重新因式分解基础矩阵。
        启用此选项以最大化速度，但可能存在非确定性行为。
        如果``maxupdate``为0，则忽略此选项。
    pivot : "mrc" or "bland" (default: "mrc")
        Pivot rule: Minimum Reduced Cost ("mrc") or Bland's rule ("bland").
        Choose Bland's rule if iteration limit is reached and cycling is
        suspected.
    unknown_options : dict
        Optional arguments not used by this particular solver. If
        `unknown_options` is non-empty a warning is issued listing all
        unused options.

    Returns
    -------
    res : OptimizeResult
        A :class:`scipy.optimize.OptimizeResult` consisting of the fields:

        x : 1-D array
            The values of the decision variables that minimizes the
            objective function while satisfying the constraints.
        fun : float
            The optimal value of the objective function ``c @ x``.
        slack : 1-D array
            The (nominally positive) values of the slack variables,
            ``b_ub - A_ub @ x``.
        con : 1-D array
            The (nominally zero) residuals of the equality constraints,
            ``b_eq - A_eq @ x``.
        success : bool
            ``True`` when the algorithm succeeds in finding an optimal
            solution.
        status : int
            An integer representing the exit status of the algorithm.

            ``0`` : Optimization terminated successfully.

            ``1`` : Iteration limit reached.

            ``2`` : Problem appears to be infeasible.

            ``3`` : Problem appears to be unbounded.

            ``4`` : Numerical difficulties encountered.

            ``5`` : Problem has no constraints; turn presolve on.

            ``6`` : Invalid guess provided.

        message : str
            A string descriptor of the exit status of the algorithm.
        nit : int
            The total number of iterations performed in all phases.


    Notes
    -----
    Method *revised simplex* uses the revised simplex method as described in
    [9]_, except that a factorization [11]_ of the basis matrix, rather than
    its inverse, is efficiently maintained and used to solve the linear systems
    at each iteration of the algorithm.

    References
    ----------
    .. [9] Bertsimas, Dimitris, and J. Tsitsiklis. "Introduction to linear
           programming." Athena Scientific 1 (1997): 997.
    .. [11] Bartels, Richard H. "A stabilization of the simplex method."
            Journal in  Numerische Mathematik 16.5 (1971): 414-434.
    """
    pass


注释：

    pivot : "mrc" or "bland" (default: "mrc")
        Pivot规则：最小化减少成本（"mrc"）或Bland规则（"bland"）。
        当达到迭代限制且存在循环时选择Bland规则。
    unknown_options : dict
        此求解器未使用的可选参数。如果 `unknown_options` 非空，则发出警告，列出所有未使用的选项。

    Returns
    -------
    res : OptimizeResult
        一个:class:`scipy.optimize.OptimizeResult`，包含以下字段：

        x : 1-D array
            使目标函数最小化并满足约束条件的决策变量的值。
        fun : float
            目标函数 ``c @ x`` 的最优值。
        slack : 1-D array
            松弛变量的（名义上的正）值， ``b_ub - A_ub @ x``。
        con : 1-D array
            等式约束的（名义上的零）残差， ``b_eq - A_eq @ x``。
        success : bool
            当算法成功找到最优解时为 ``True`` 。
        status : int
            表示算法退出状态的整数。

            ``0`` : 优化成功终止。

            ``1`` : 达到迭代限制。

            ``2`` : 问题似乎不可行。

            ``3`` : 问题似乎无界。

            ``4`` : 遇到数值困难。

            ``5`` : 问题没有约束；开启预处理。

            ``6`` : 提供的初始猜测无效。

        message : str
            算法退出状态的字符串描述。
        nit : int
            所有阶段中执行的总迭代次数。

    Notes
    -----
    *修订单纯形法* 方法使用[9]_中描述的修订单纯形法，不同之处在于在算法的每次迭代中，维护并使用基础矩阵的因子分解[11]_，而不是其逆。

    References
    ----------
    .. [9] Bertsimas, Dimitris, and J. Tsitsiklis. "Introduction to linear
           programming." Athena Scientific 1 (1997): 997.
    .. [11] Bartels, Richard H. "A stabilization of the simplex method."
            Journal in  Numerische Mathematik 16.5 (1971): 414-434.
# 定义线性规划函数，使用表格法的单纯形方法来最小化线性目标函数，同时满足线性等式和不等式约束。

def _linprog_simplex_doc(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                         bounds=None, method='interior-point', callback=None,
                         maxiter=5000, disp=False, presolve=True,
                         tol=1e-12, autoscale=False, rr=True, bland=False,
                         **unknown_options):
    r"""
    Linear programming: minimize a linear objective function subject to linear
    equality and inequality constraints using the tableau-based simplex method.

    .. deprecated:: 1.9.0
        `method='simplex'` will be removed in SciPy 1.11.0.
        It is replaced by `method='highs'` because the latter is
        faster and more robust.

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

    minimize::

        c @ x

    such that::

        A_ub @ x <= b_ub
        A_eq @ x == b_eq
        lb <= x <= ub

    Note that by default ``lb = 0`` and ``ub = None`` unless specified with
    ``bounds``.

    Parameters
    ----------
    c : 1-D array
        The coefficients of the linear objective function to be minimized.
    A_ub : 2-D array, optional
        The inequality constraint matrix. Each row of ``A_ub`` specifies the
        coefficients of a linear inequality constraint on ``x``.
    b_ub : 1-D array, optional
        The inequality constraint vector. Each element represents an
        upper bound on the corresponding value of ``A_ub @ x``.
    A_eq : 2-D array, optional
        The equality constraint matrix. Each row of ``A_eq`` specifies the
        coefficients of a linear equality constraint on ``x``.
    b_eq : 1-D array, optional
        The equality constraint vector. Each element of ``A_eq @ x`` must equal
        the corresponding element of ``b_eq``.
    bounds : sequence, optional
        A sequence of ``(min, max)`` pairs for each element in ``x``, defining
        the minimum and maximum values of that decision variable. Use ``None``
        to indicate that there is no bound. By default, bounds are
        ``(0, None)`` (all decision variables are non-negative).
        If a single tuple ``(min, max)`` is provided, then ``min`` and
        ``max`` will serve as bounds for all decision variables.
    method : str
        这是针对'simplex'方法的特定文档说明。
        还可以使用 :ref:`'highs' <optimize.linprog-highs>`,
        :ref:`'highs-ds' <optimize.linprog-highs-ds>`,
        :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`,
        :ref:`'interior-point' <optimize.linprog-interior-point>` (默认)，
        和 :ref:`'revised simplex' <optimize.linprog-revised_simplex>`。
    callback : callable, optional
        可选的回调函数，在每次迭代时执行。

    Options
    -------
    maxiter : int (default: 5000)
        执行每个阶段的最大迭代次数。
    disp : bool (default: False)
        如果要在每次迭代时将优化状态的指示输出到控制台，则设置为 ``True``。
    presolve : bool (default: True)
        预处理尝试在将问题发送到主求解器之前识别微不足道的不可行性，
        识别微不足道的无界性，并简化问题。通常建议保持默认设置 ``True``；
        如果要禁用预处理，请设置为 ``False``。
    tol : float (default: 1e-12)
        决定何时解在第1阶段“足够接近”零以被视为基本可行解或足够接近正以成为最优解的容差。
    autoscale : bool (default: False)
        设置为 ``True`` 以自动执行均衡处理。
        如果约束中的数值相差几个数量级，则考虑使用此选项。
    rr : bool (default: True)
        设置为 ``False`` 以禁用自动冗余移除。
    bland : bool
        如果为True，则使用Bland的反循环规则 [3]_ 选择枢轴以防止循环。
        如果为False，则选择枢轴应更快地导致收敛解决方案。后一种方法偶尔会导致循环（不收敛）。
    unknown_options : dict
        此特定求解器未使用的可选参数。如果 `unknown_options` 不为空，
        则发出警告并列出所有未使用的选项。

    Returns
    -------
    res : OptimizeResult
        一个:class:`scipy.optimize.OptimizeResult`对象，包含以下字段：

        x : 1-D array
            决策变量的取值，使得在满足约束条件的情况下最小化目标函数。
        fun : float
            目标函数 ``c @ x`` 的最优值。
        slack : 1-D array
            松弛变量的值（通常是正值），即 ``b_ub - A_ub @ x``。
        con : 1-D array
            等式约束的残差（通常为零），即 ``b_eq - A_eq @ x``。
        success : bool
            当算法成功找到最优解时为 ``True``。
        status : int
            表示算法退出状态的整数。

            ``0`` : 优化成功终止。
    
            ``1`` : 达到迭代限制。
    
            ``2`` : 问题似乎不可行。
    
            ``3`` : 问题似乎无界。
    
            ``4`` : 遇到数值困难。
    
        message : str
            算法退出状态的描述字符串。
        nit : int
            所有阶段中执行的总迭代次数。

    References
    ----------
    .. [1] Dantzig, George B., Linear programming and extensions. Rand
           Corporation Research Study Princeton Univ. Press, Princeton, NJ,
           1963
    .. [2] Hillier, S.H. and Lieberman, G.J. (1995), "Introduction to
           Mathematical Programming", McGraw-Hill, Chapter 4.
    .. [3] Bland, Robert G. New finite pivoting rules for the simplex method.
           Mathematics of Operations Research (2), 1977: pp. 103-107.
    ```
```