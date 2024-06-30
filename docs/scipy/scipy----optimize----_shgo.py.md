# `D:\src\scipysrc\scipy\scipy\optimize\_shgo.py`

```
"""shgo: The simplicial homology global optimisation algorithm."""
# 导入必要的模块和库
from collections import namedtuple  # 导入命名元组模块
import time  # 导入时间模块
import logging  # 导入日志记录模块
import warnings  # 导入警告模块
import sys  # 导入系统模块

import numpy as np  # 导入NumPy库

from scipy import spatial  # 导入SciPy中的空间模块
from scipy.optimize import OptimizeResult, minimize, Bounds  # 导入优化相关模块
from scipy.optimize._optimize import MemoizeJac  # 导入Jacobi矩阵缓存模块
from scipy.optimize._constraints import new_bounds_to_old  # 导入旧边界转换模块
from scipy.optimize._minimize import standardize_constraints  # 导入标准化约束模块
from scipy._lib._util import _FunctionWrapper  # 导入函数包装器模块

from scipy.optimize._shgo_lib._complex import Complex  # 导入复杂对象模块

__all__ = ['shgo']  # 模块中公开的接口列表


def shgo(
    func, bounds, args=(), constraints=None, n=100, iters=1, callback=None,
    minimizer_kwargs=None, options=None, sampling_method='simplicial', *,
    workers=1
):
    """
    Finds the global minimum of a function using SHG optimization.

    SHGO stands for "simplicial homology global optimization".

    Parameters
    ----------
    func : callable
        The objective function to be minimized.  Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a tuple of any additional fixed parameters needed to
        completely specify the function.
    bounds : sequence or `Bounds`
        Bounds for variables. There are two ways to specify the bounds:

        1. Instance of `Bounds` class.
        2. Sequence of ``(min, max)`` pairs for each element in `x`.

    args : tuple, optional
        Any additional fixed parameters needed to completely specify the
        objective function.
    constraints : {Constraint, dict} or List of {Constraint, dict}, optional
        Constraints definition. Only for COBYLA, COBYQA, SLSQP and trust-constr.
        See the tutorial [5]_ for further details on specifying constraints.

        .. note::

           Only COBYLA, COBYQA, SLSQP, and trust-constr local minimize methods
           currently support constraint arguments. If the ``constraints``
           sequence used in the local optimization problem is not defined in
           ``minimizer_kwargs`` and a constrained method is used then the
           global ``constraints`` will be used.
           (Defining a ``constraints`` sequence in ``minimizer_kwargs``
           means that ``constraints`` will not be added so if equality
           constraints and so forth need to be added then the inequality
           functions in ``constraints`` need to be added to
           ``minimizer_kwargs`` too).
           COBYLA only supports inequality constraints.

        .. versionchanged:: 1.11.0

           ``constraints`` accepts `NonlinearConstraint`, `LinearConstraint`.
    """
    # 实现SHGO算法，寻找函数的全局最小值

    # 返回函数的全局最小值
    pass
    # n : int, optional
    #    用于构建单纯复形的采样点数量。对于默认的“simplicial”采样方法，生成2**dim + 1个采样点，
    #    而不是默认的 n=100。对于所有其他指定的值 `n`，生成 `n` 个采样点。
    #    对于“sobol”，“halton”和其他任意的 `sampling_methods`，生成 `n=100` 或另一个指定的采样点数。
    iters : int, optional
        # 用于构建单纯复形的迭代次数。默认为 1。
    callback : callable, optional
        # 每次迭代后调用的回调函数，形式为 ``callback(xk)``，其中 ``xk`` 是当前的参数向量。
    minimizer_kwargs : dict, optional
        # 传递给最小化器 ``scipy.optimize.minimize`` 的额外关键字参数。
        # 一些重要的选项包括：
        #   * method : str
        #       最小化方法。如果未给出，则根据问题是否有约束或边界来选择 BFGS、L-BFGS-B、SLSQP 之一。
        #   * args : tuple
        #       传递给目标函数（``func``）及其导数（Jacobian、Hessian）的额外参数。
        #   * options : dict, optional
        #       默认情况下，容差被指定为 ``{ftol: 1e-12}``。
    sampling_method : str or function, optional
        # 当前内置的采样方法选项有 ``halton``、``sobol`` 和 ``simplicial``。
        # 默认的 ``simplicial`` 提供了理论上的全局最小值收敛保证。
        # ``halton`` 和 ``sobol`` 方法在采样点生成方面更快，但以失去保证收敛为代价。
        # 在大多数相对简单的问题中，收敛相对较快，更为适用。
        # 用户定义的采样函数必须接受两个参数 `n`（每次调用的维度为 `dim` 的采样点数量），
        # 并输出形状为 `n x dim` 的采样点数组。
    workers : int or map-like callable, optional
        # 并行采样和本地串行最小化运行。
        # 提供 -1 使用所有可用的 CPU 核心，或者提供一个整数使用那么多进程（使用 `multiprocessing.Pool <multiprocessing>`）。
        # 或者提供一个类似于映射的可调用对象，例如 `multiprocessing.Pool.map` 进行并行评估。
        # 此评估作为 ``workers(func, iterable)`` 进行。
        # 需要 `func` 是可 pickle 化的。
        # .. versionadded:: 1.11.0
    Returns
    -------
    res : OptimizeResult
        # 优化结果，表示为 `OptimizeResult` 对象。
        # 主要属性包括：
        # ``x``：全局最小值对应的解数组，
        # ``fun``：全局解处的函数输出值，
        # ``xl``：局部最小值解的有序列表，
        # ``funl``：对应局部解的函数输出值，
        # ``success``：布尔标志，指示优化器是否成功退出，
        # ``message``：描述终止原因的消息，
        # ``nfev``：包括采样调用在内的总目标函数评估次数，
        # ``nlfev``：来自所有局部搜索优化的总目标函数评估次数，
        # ``nit``：全局程序执行的迭代次数。

    Notes
    -----
    # 使用单纯同调全局优化（SHGO）算法进行全局优化 [1]_。
    # 适用于解决一般的非线性规划（NLP）和黑箱优化问题，以达到全局最优解（低维问题）。

    # 一般来说，优化问题的形式为::

    #     minimize f(x) subject to

    #     g_i(x) >= 0,  i = 1,...,m
    #     h_j(x)  = 0,  j = 1,...,p

    # 其中 x 是一个或多个变量的向量。``f(x)`` 是目标函数 ``R^n -> R``,
    # ``g_i(x)`` 是不等式约束，``h_j(x)`` 是等式约束。

    # 可选地，可以使用 `bounds` 参数指定每个元素的上下界。

    # 虽然 SHGO 的大部分理论优势只对 Lipschitz 平滑函数 ``f(x)`` 证明了有效，
    # 但是如果使用默认采样方法，该算法也被证明可以收敛于全局最优解，
    # 即使 ``f(x)`` 是非连续、非凸和非平滑的情况 [1]_。

    # 可以通过 ``minimizer_kwargs`` 参数指定局部搜索方法，
    # 该参数将传递给 ``scipy.optimize.minimize``。默认情况下使用 ``SLSQP`` 方法。
    # 一般建议在问题定义了不等式约束时使用 ``SLSQP``, ``COBYLA`` 或 ``COBYQA`` 局部最小化方法，
    # 因为其他方法不使用约束条件。

    # ``halton`` 和 ``sobol`` 方法点使用 `scipy.stats.qmc` 生成。可以使用任何其他 QMC 方法。

    References
    ----------
    # .. [1] Endres, SC, Sandrock, C, Focke, WW (2018) "A simplicial homology
    #        algorithm for lipschitz optimisation", Journal of Global
    #        Optimization.
    # .. [2] Joe, SW and Kuo, FY (2008) "Constructing Sobol' sequences with
    #        better  two-dimensional projections", SIAM J. Sci. Comput. 30,
    #        2635-2654.
    .. [3] Hock, W and Schittkowski, K (1981) "Test examples for nonlinear
           programming codes", Lecture Notes in Economics and Mathematical
           Systems, 187. Springer-Verlag, New York.
           http://www.ai7.uni-bayreuth.de/test_problem_coll.pdf
    .. [4] Wales, DJ (2015) "Perspective: Insight into reaction coordinates and
           dynamics from the potential energy landscape",
           Journal of Chemical Physics, 142(13), 2015.
    .. [5] https://docs.scipy.org/doc/scipy/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize
    这里是一些参考文献和链接，可能包含有关优化问题的背景和更多信息。

    Examples
    --------
    首先考虑最小化 Rosenbrock 函数 `rosen` 的问题：

    >>> from scipy.optimize import rosen, shgo
    导入 rosen 函数和 shgo 最优化方法

    >>> bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]
    定义变量的取值范围列表，用于约束问题的维度

    >>> result = shgo(rosen, bounds)
    使用 shgo 方法对 rosen 函数进行最优化求解

    >>> result.x, result.fun
    打印结果的最优解和最优函数值

    (array([1., 1., 1., 1., 1.]), 2.920392374190081e-18)
    输出最优解数组和对应的最优函数值

    Note that bounds determine the dimensionality of the objective
    function and is therefore a required input, however you can specify
    empty bounds using ``None`` or objects like ``np.inf`` which will be
    converted to large float numbers.

    >>> bounds = [(None, None), ]*4
    重新定义变量的取值范围列表，此处使用 None 表示不加约束

    >>> result = shgo(rosen, bounds)
    使用 shgo 方法对 rosen 函数进行最优化求解，此时无约束条件

    >>> result.x
    打印结果的最优解数组

    array([0.99999851, 0.99999704, 0.99999411, 0.9999882 ])
    输出最优解数组

    Next, we consider the Eggholder function, a problem with several local
    minima and one global minimum. We will demonstrate the use of arguments and
    the capabilities of `shgo`.
    (https://en.wikipedia.org/wiki/Test_functions_for_optimization)

    >>> import numpy as np
    导入 numpy 库

    >>> def eggholder(x):
    ...     return (-(x[1] + 47.0)
    ...             * np.sin(np.sqrt(abs(x[0]/2.0 + (x[1] + 47.0))))
    ...             - x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47.0))))
    ...             )
    定义 Eggholder 函数及其表达式

    >>> bounds = [(-512, 512), (-512, 512)]
    定义变量的取值范围列表，用于 Eggholder 函数的约束问题

    `shgo` has built-in low discrepancy sampling sequences. First, we will
    input 64 initial sampling points of the *Sobol'* sequence:

    >>> result = shgo(eggholder, bounds, n=64, sampling_method='sobol')
    使用 shgo 方法对 eggholder 函数进行最优化求解，使用 Sobol' 序列进行初始采样

    >>> result.x, result.fun
    打印结果的最优解和最优函数值

    (array([512.        , 404.23180824]), -959.6406627208397)
    输出最优解数组和对应的最优函数值

    `shgo` also has a return for any other local minima that was found, these
    can be called using:

    >>> result.xl
    输出找到的所有局部最优解的数组列表

    array([[ 512.        ,  404.23180824],
           [ 283.0759062 , -487.12565635],
           [-294.66820039, -462.01964031],
           [-105.87688911,  423.15323845],
           [-242.97926   ,  274.38030925],
           [-506.25823477,    6.3131022 ],
           [-408.71980731, -156.10116949],
           [ 150.23207937,  301.31376595],
           [  91.00920901, -391.283763  ],
           [ 202.89662724, -269.38043241],
           [ 361.66623976, -106.96493868],
           [-219.40612786, -244.06020508]])
    输出所有局部最优解的数组列表

    >>> result.funl
    array([-959.64066272, -718.16745962, -704.80659592, -565.99778097,
           -559.78685655, -557.36868733, -507.87385942, -493.9605115 ,
           -426.48799655, -421.15571437, -419.31194957, -410.98477763])


# 这是一个 numpy 数组，包含一组数值，可能是优化算法返回的目标函数值
These results are useful in applications where there are many global minima
and the values of other global minima are desired or where the local minima
can provide insight into the system (for example morphologies
in physical chemistry [4]_).

If we want to find a larger number of local minima, we can increase the
number of sampling points or the number of iterations. We'll increase the
number of sampling points to 64 and the number of iterations from the
default of 1 to 3. Using ``simplicial`` this would have given us
64 x 3 = 192 initial sampling points.

>>> result_2 = shgo(eggholder,
...                 bounds, n=64, iters=3, sampling_method='sobol')
>>> len(result.xl), len(result_2.xl)
(12, 23)

Note the difference between, e.g., ``n=192, iters=1`` and ``n=64,
iters=3``.
In the first case the promising points contained in the minimiser pool
are processed only once. In the latter case it is processed every 64
sampling points for a total of 3 times.

To demonstrate solving problems with non-linear constraints consider the
following example from Hock and Schittkowski problem 73 (cattle-feed)
[3]_::

    minimize: f = 24.55 * x_1 + 26.75 * x_2 + 39 * x_3 + 40.50 * x_4

    subject to: 2.3 * x_1 + 5.6 * x_2 + 11.1 * x_3 + 1.3 * x_4 - 5    >= 0,

                12 * x_1 + 11.9 * x_2 + 41.8 * x_3 + 52.1 * x_4 - 21
                    -1.645 * sqrt(0.28 * x_1**2 + 0.19 * x_2**2 +
                                  20.5 * x_3**2 + 0.62 * x_4**2)      >= 0,

                x_1 + x_2 + x_3 + x_4 - 1                             == 0,

                1 >= x_i >= 0 for all i

The approximate answer given in [3]_ is::

    f([0.6355216, -0.12e-11, 0.3127019, 0.05177655]) = 29.894378

>>> def f(x):  # (cattle-feed)
...     return 24.55*x[0] + 26.75*x[1] + 39*x[2] + 40.50*x[3]
...
>>> def g1(x):
...     return 2.3*x[0] + 5.6*x[1] + 11.1*x[2] + 1.3*x[3] - 5  # >=0
...
>>> def g2(x):
...     return (12*x[0] + 11.9*x[1] +41.8*x[2] + 52.1*x[3] - 21
...             - 1.645 * np.sqrt(0.28*x[0]**2 + 0.19*x[1]**2
...                             + 20.5*x[2]**2 + 0.62*x[3]**2)
...             ) # >=0
...
>>> def h1(x):
...     return x[0] + x[1] + x[2] + x[3] - 1  # == 0
...
>>> cons = ({'type': 'ineq', 'fun': g1},
...         {'type': 'ineq', 'fun': g2},
...         {'type': 'eq', 'fun': h1})
>>> bounds = [(0, 1.0),]*4
>>> res = shgo(f, bounds, n=150, constraints=cons)
    >>> res
     message: Optimization terminated successfully.
     success: True
         fun: 29.894378159142136
        funl: [ 2.989e+01]
           x: [ 6.355e-01  1.137e-13  3.127e-01  5.178e-02] # may vary
          xl: [[ 6.355e-01  1.137e-13  3.127e-01  5.178e-02]] # may vary
         nit: 1
        nfev: 142 # may vary
       nlfev: 35 # may vary
       nljev: 5
       nlhev: 0

    >>> g1(res.x), g2(res.x), h1(res.x)
    (-5.062616992290714e-14, -2.9594104944408173e-12, 0.0)

    """
    # 如果需要，将Bounds类转换为旧版Bounds
    if isinstance(bounds, Bounds):
        bounds = new_bounds_to_old(bounds.lb, bounds.ub, len(bounds.lb))

    # 初始化SHGO类
    # 使用上下文管理器确保任何并行化资源被释放
    with SHGO(func, bounds, args=args, constraints=constraints, n=n,
               iters=iters, callback=callback,
               minimizer_kwargs=minimizer_kwargs,
               options=options, sampling_method=sampling_method,
               workers=workers) as shc:
        # 运行算法，处理结果并测试成功
        shc.iterate_all()

    if not shc.break_routine:
        if shc.disp:
            logging.info("Successfully completed construction of complex.")

    # 测试迭代后的成功性
    if len(shc.LMC.xl_maps) == 0:
        # 如果采样未能找到池，则返回最低采样点并带有警告
        shc.find_lowest_vertex()
        shc.break_routine = True
        shc.fail_routine(mes="Failed to find a feasible minimizer point. "
                             f"Lowest sampling point = {shc.f_lowest}")
        shc.res.fun = shc.f_lowest
        shc.res.x = shc.x_lowest
        shc.res.nfev = shc.fn
        shc.res.tnev = shc.n_sampled
    else:
        # 测试最优解是否违反任何约束条件
        pass  # TODO

    # 确认程序成功运行
    if not shc.break_routine:
        shc.res.message = 'Optimization terminated successfully.'
        shc.res.success = True

    # 返回最终结果
    return shc.res
# 定义一个名为 SHGO 的类
class SHGO:

    # Initiation aids
    # 初始化选项
    def init_options(self, options):
        """
        Initiates the options.

        Can also be useful to change parameters after class initiation.

        Parameters
        ----------
        options : dict
            A dictionary containing various optimization options.

        Returns
        -------
        None
        """

        # Update 'options' dict passed to optimize.minimize
        # 更新传递给 optimize.minimize 的 'options' 字典
        self.minimizer_kwargs['options'].update(options)

        # Ensure that 'jac', 'hess', and 'hessp' are passed directly to
        # `minimize` as keywords, not as part of its 'options' dictionary.
        # 确保 'jac'、'hess' 和 'hessp' 直接作为关键字传递给 `minimize`
        # 而不是作为 'options' 字典的一部分。
        for opt in ['jac', 'hess', 'hessp']:
            if opt in self.minimizer_kwargs['options']:
                self.minimizer_kwargs[opt] = (
                    self.minimizer_kwargs['options'].pop(opt))

        # Default settings:
        # 默认设置:

        # 是否在每次迭代中进行最小化
        self.minimize_every_iter = options.get('minimize_every_iter', True)

        # Algorithm limits
        # 算法限制

        # 最大迭代次数
        self.maxiter = options.get('maxiter', None)

        # 在可行域中的最大函数评估次数
        self.maxfev = options.get('maxfev', None)

        # 总的采样评估次数（包括在非可行点搜索）
        self.maxev = options.get('maxev', None)

        # 允许的最大处理运行时间
        self.init = time.time()
        self.maxtime = options.get('maxtime', None)

        # 如果 'f_min' 在 options 中指定
        if 'f_min' in options:
            # 指定已知的最小目标函数值
            self.f_min_true = options['f_min']
            # 目标函数值容差
            self.f_tol = options.get('f_tol', 1e-4)
        else:
            self.f_min_true = None

        # 最小网格尺寸
        self.minhgrd = options.get('minhgrd', None)

        # Objective function knowledge
        # 目标函数的知识

        # 是否具有对称性
        self.symmetry = options.get('symmetry', False)
        if self.symmetry:
            self.symmetry = [0, ] * len(self.bounds)
        else:
            self.symmetry = None

        # Algorithm functionality
        # 算法功能

        # 是否只评估少数最佳候选者
        self.local_iter = options.get('local_iter', False)

        # 是否在无限约束条件下进行采样
        self.infty_cons_sampl = options.get('infty_constraints', True)

        # Feedback
        # 反馈

        # 是否显示优化过程信息
        self.disp = options.get('disp', False)

    # Enter method for context management
    # 上下文管理的进入方法
    def __enter__(self):
        return self

    # Exit method for context management
    # 上下文管理的退出方法
    def __exit__(self, *args):
        return self.HC.V._mapwrapper.__exit__(*args)

    # Iteration properties
    # 迭代属性

    # Main construction loop:
    # 主要的构造循环:
    def iterate_all(self):
        """
        Construct for `iters` iterations.

        If uniform sampling is used, every iteration adds 'n' sampling points.

        Iterations if a stopping criteria (e.g., sampling points or
        processing time) has been met.

        """
        # 如果设置了显示选项，记录日志，表示正在进行第一代的拆分
        if self.disp:
            logging.info('Splitting first generation')

        # 当全局停止条件未达成时循环执行以下操作
        while not self.stop_global:
            # 如果设置了打断标志，退出循环
            if self.break_routine:
                break
            # 调用 iterate 方法执行迭代计算
            self.iterate()
            # 执行停止条件的判断和处理
            self.stopping_criteria()

        # 构建最小化池
        # 只有在每次迭代不进行最小化时才需要最后一次迭代
        if not self.minimize_every_iter:
            if not self.break_routine:
                # 调用 find_minima 方法，找到最小值
                self.find_minima()

        # 设置迭代次数到结果对象中
        self.res.nit = self.iters_done  # + 1
        # 设置评估函数调用次数到对象属性中
        self.fn = self.HC.V.nfev

    def find_minima(self):
        """
        Construct the minimizer pool, map the minimizers to local minima
        and sort the results into a global return object.
        """
        # 如果设置了显示选项，记录日志，表示正在搜索最小化池
        if self.disp:
            logging.info('Searching for minimizer pool...')

        # 调用 minimizers 方法，构建最小化器池
        self.minimizers()

        # 如果找到了最小值
        if len(self.X_min) != 0:
            # 使用局部最小化方法最小化最小化器池
            # 注意，如果 Options['local_iter'] 是 int 类型，只会最小化指定数量的候选者
            self.minimise_pool(self.local_iter)
            # 对结果进行排序，并构建全局返回对象
            self.sort_result()

            # 记录最低值以便在失败时报告
            self.f_lowest = self.res.fun
            self.x_lowest = self.res.x
        else:
            # 如果未找到最小值，执行 find_lowest_vertex 方法
            self.find_lowest_vertex()

        # 如果设置了显示选项，记录日志，输出最小化池的内容
        if self.disp:
            logging.info(f"Minimiser pool = SHGO.X_min = {self.X_min}")

    def find_lowest_vertex(self):
        # 在单纯形复合体的顶点中找到最低目标函数值
        self.f_lowest = np.inf
        for x in self.HC.V.cache:
            if self.HC.V[x].f < self.f_lowest:
                # 如果设置了显示选项，记录日志，输出当前最低目标函数值
                if self.disp:
                    logging.info(f'self.HC.V[x].f = {self.HC.V[x].f}')
                self.f_lowest = self.HC.V[x].f
                self.x_lowest = self.HC.V[x].x_a
        for lmc in self.LMC.cache:
            if self.LMC[lmc].f_min < self.f_lowest:
                self.f_lowest = self.LMC[lmc].f_min
                self.x_lowest = self.LMC[lmc].x_l

        # 如果没有找到可行点，将最低目标函数值和相应点置为 None
        if self.f_lowest == np.inf:
            self.f_lowest = None
            self.x_lowest = None
    # 计算最小迭代次数
    mi = min(x for x in [self.iters, self.maxiter] if x is not None)
    # 如果指定了输出信息，记录迭代完成情况
    if self.disp:
        logging.info(f'Iterations done = {self.iters_done} / {mi}')
    # 如果指定了迭代次数，并且达到或超过设定值，停止全局优化
    if self.iters is not None:
        if self.iters_done >= (self.iters):
            self.stop_global = True

    # 如果指定了最大迭代次数，并且达到或超过设定值，停止全局优化
    if self.maxiter is not None:
        if self.iters_done >= (self.maxiter):
            self.stop_global = True
    return self.stop_global

# 有限函数评估次数
def finite_fev(self):
    # 如果指定了输出信息，记录函数评估完成情况
    if self.disp:
        logging.info(f'Function evaluations done = {self.fn} / {self.maxfev}')
    # 如果函数评估次数达到或超过设定值，停止全局优化
    if self.fn >= self.maxfev:
        self.stop_global = True
    return self.stop_global

# 有限评估次数（包括不可行采样点）
def finite_ev(self):
    # 如果指定了输出信息，记录采样评估完成情况
    if self.disp:
        logging.info(f'Sampling evaluations done = {self.n_sampled} '
                     f'/ {self.maxev}')
    # 如果采样评估次数达到或超过设定值，停止全局优化
    if self.n_sampled >= self.maxev:
        self.stop_global = True

# 有限时间（根据最大允许时间）
def finite_time(self):
    # 如果指定了输出信息，记录经过的时间
    if self.disp:
        logging.info(f'Time elapsed = {time.time() - self.init} '
                     f'/ {self.maxtime}')
    # 如果经过的时间超过设定的最大时间，停止全局优化
    if (time.time() - self.init) >= self.maxtime:
        self.stop_global = True

# 有限精度（根据已知的最小函数值）
def finite_precision(self):
    """
    Stop the algorithm if the final function value is known

    Specify in options (with ``self.f_min_true = options['f_min']``)
    and the tolerance with ``f_tol = options['f_tol']``
    """
    # 查找最低采样点的函数值
    self.find_lowest_vertex()
    # 如果指定了输出信息，记录最低函数值和指定的最小值
    if self.disp:
        logging.info(f'Lowest function evaluation = {self.f_lowest}')
        logging.info(f'Specified minimum = {self.f_min_true}')
    # 如果没有返回可行点，则保持停止全局优化状态
    if self.f_lowest is None:
        return self.stop_global

    # 根据指定的百分比误差停止算法
    if self.f_min_true == 0.0:
        if self.f_lowest <= self.f_tol:
            self.stop_global = True
    else:
        # 计算相对误差
        pe = (self.f_lowest - self.f_min_true) / abs(self.f_min_true)
        if self.f_lowest <= self.f_min_true:
            self.stop_global = True
            # 如果相对误差超过设定的两倍容差，发出警告
            if abs(pe) >= 2 * self.f_tol:
                warnings.warn(
                    f"A much lower value than expected f* = {self.f_min_true} "
                    f"was found f_lowest = {self.f_lowest}",
                    stacklevel=3
                )
        # 如果相对误差小于等于设定的容差，停止全局优化
        if pe <= self.f_tol:
            self.stop_global = True

    return self.stop_global
    def finite_homology_growth(self):
        """
        Stop the algorithm if homology group rank did not grow in iteration.
        """
        # 检查 LMC 的大小，如果为 0，则直接返回，不需要停止
        if self.LMC.size == 0:
            return  # pass on no reason to stop yet.
        # 计算当前迭代的同伦群秩增长量
        self.hgrd = self.LMC.size - self.hgr

        # 更新同伦群秩的值为当前 LMC 的大小
        self.hgr = self.LMC.size
        # 如果同伦群秩的增长量小于等于最小增长量阈值，则标记全局停止
        if self.hgrd <= self.minhgrd:
            self.stop_global = True
        # 如果开启了显示日志，则记录当前的同伦群增长值和最小增长值
        if self.disp:
            logging.info(f'Current homology growth = {self.hgrd} '
                         f' (minimum growth = {self.minhgrd})')
        # 返回全局停止标志
        return self.stop_global

    def stopping_criteria(self):
        """
        Various stopping criteria ran every iteration

        Returns
        -------
        stop : bool
        """
        # 如果设定了最大迭代次数，调用 finite_iterations 方法进行检查
        if self.maxiter is not None:
            self.finite_iterations()
        # 如果设定了迭代次数，调用 finite_iterations 方法进行检查
        if self.iters is not None:
            self.finite_iterations()
        # 如果设定了最大函数评估次数，调用 finite_fev 方法进行检查
        if self.maxfev is not None:
            self.finite_fev()
        # 如果设定了最大事件次数，调用 finite_ev 方法进行检查
        if self.maxev is not None:
            self.finite_ev()
        # 如果设定了最大时间，调用 finite_time 方法进行检查
        if self.maxtime is not None:
            self.finite_time()
        # 如果设定了最小函数值的精度，调用 finite_precision 方法进行检查
        if self.f_min_true is not None:
            self.finite_precision()
        # 如果设定了最小同伦群增长量，调用 finite_homology_growth 方法进行检查
        if self.minhgrd is not None:
            self.finite_homology_growth()
        # 返回全局停止标志
        return self.stop_global

    def iterate(self):
        # 执行复杂体迭代
        self.iterate_complex()

        # 如果设置每次迭代都进行最小化处理，并且没有中断例程，则找到最小值
        if self.minimize_every_iter:
            if not self.break_routine:
                self.find_minima()  # Process minimizer pool

        # 更新迭代次数
        self.iters_done += 1
    def iterate_hypercube(self):
        """
        Iterate a subdivision of the complex
        
        Note: called with ``self.iterate_complex()`` after class initiation
        """
        # 迭代复杂结构的一个子分区

        # 迭代复杂结构
        if self.disp:
            logging.info('Constructing and refining simplicial complex graph '
                         'structure')
        # 如果需要显示信息，记录构建和优化单纯形复杂图结构的日志信息

        if self.n is None:
            self.HC.refine_all()
            self.n_sampled = self.HC.V.size()  # 计算采样点数量
        else:
            self.HC.refine(self.n)
            self.n_sampled += self.n

        if self.disp:
            logging.info('Triangulation completed, evaluating all constraints '
                         'and objective function values.')
        # 如果需要显示信息，记录三角剖分完成，并评估所有约束和目标函数值的日志信息

        # 将极小值重新加入复杂结构
        if len(self.LMC.xl_maps) > 0:
            for xl in self.LMC.cache:
                v = self.HC.V[xl]
                v_near = v.star()
                for v in v.nn:
                    v_near = v_near.union(v.nn)
                # 重新连接顶点到复杂结构
                # if self.HC.connect_vertex_non_symm(tuple(self.LMC[xl].x_l),
                #                                   near=v_near):
                #    continue
                # else:
                    # 如果在 v_near 中找不到，就在所有顶点中搜索（非常昂贵的操作）
                #    self.HC.connect_vertex_non_symm(tuple(self.LMC[xl].x_l)
                #                                    )

        # 评估所有约束和函数
        self.HC.V.process_pools()
        if self.disp:
            logging.info('Evaluations completed.')
        # 如果需要显示信息，记录评估完成的日志信息

        # 通过三角剖分.py中的程序统计可行采样点数量
        self.fn = self.HC.V.nfev
        return
    # 定义一个方法用于迭代生成 Delaunay 三角化的点集合

    """
    Build a complex of Delaunay triangulated points

    Note: called with ``self.iterate_complex()`` after class initiation
    """
    # 增加当前点集的数量
    self.nc += self.n
    # 使用采样点填充表面，可能包括无穷远点
    self.sampled_surface(infty_cons_sampl=self.infty_cons_sampl)

    # 将采样点加入三角剖分中，构建 self.Tri
    # 如果开启了显示模式，则记录相关信息
    if self.disp:
        logging.info(f'self.n = {self.n}')
        logging.info(f'self.nc = {self.nc}')
        logging.info('Constructing and refining simplicial complex graph '
                     'structure from sampling points.')

    # 如果维度小于2，则进行一维三角剖分
    if self.dim < 2:
        # 对采样点进行排序
        self.Ind_sorted = np.argsort(self.C, axis=0)
        self.Ind_sorted = self.Ind_sorted.flatten()
        tris = []
        # 构建一维三角剖分
        for ind, ind_s in enumerate(self.Ind_sorted):
            if ind > 0:
                tris.append(self.Ind_sorted[ind - 1:ind + 1])

        tris = np.array(tris)
        # 存储一维三角剖分结果
        self.Tri = namedtuple('Tri', ['points', 'simplices'])(self.C, tris)
        self.points = {}
    else:
        # 如果点的数量大于维度加1，则进行 Delaunay 三角剖分
        if self.C.shape[0] > self.dim + 1:  # 确保可以构建一个单纯形
            self.delaunay_triangulation(n_prc=self.n_prc)
        self.n_prc = self.C.shape[0]

    # 如果开启了显示模式，则记录相关信息
    if self.disp:
        logging.info('Triangulation completed, evaluating all '
                     'constraints and objective function values.')

    # 如果存在 self.Tri 属性，则将其顶点和简单形传递给 HC.vf_to_vv 方法
    if hasattr(self, 'Tri'):
        self.HC.vf_to_vv(self.Tri.points, self.Tri.simplices)

    # 处理所有池
    # 评估所有约束和函数
    if self.disp:
        logging.info('Triangulation completed, evaluating all constraints '
                     'and objective function values.')

    # 评估所有约束和函数
    self.HC.V.process_pools()

    # 如果开启了显示模式，则记录相关信息
    if self.disp:
        logging.info('Evaluations completed.')

    # 将通过三角剖分.py例程计算得到的可行采样点数进行计数
    self.fn = self.HC.V.nfev
    self.n_sampled = self.nc  # 在三角剖分中计数的采样点数
    return
    # 返回所有最小化器的索引
    def minimizers(self):
        """
        Returns the indexes of all minimizers
        返回所有最小化器的索引
        """
        self.minimizer_pool = []
        # 注意：可以在这里实现并行化处理

        # 遍历顶点缓存中的每个顶点
        for x in self.HC.V.cache:
            in_LMC = False
            # 检查顶点是否在最小化器候选集合中
            if len(self.LMC.xl_maps) > 0:
                for xlmi in self.LMC.xl_maps:
                    if np.all(np.array(x) == np.array(xlmi)):
                        in_LMC = True

            # 如果顶点在最小化器候选集合中，则跳过处理
            if in_LMC:
                continue

            # 如果顶点是最小化器
            if self.HC.V[x].minimiser():
                # 如果设置了显示标志，则记录信息
                if self.disp:
                    logging.info('=' * 60)
                    logging.info(f'v.x = {self.HC.V[x].x_a} is minimizer')
                    logging.info(f'v.f = {self.HC.V[x].f} is minimizer')
                    logging.info('=' * 30)

                # 将最小化器添加到最小化器池中（如果尚未添加）
                if self.HC.V[x] not in self.minimizer_pool:
                    self.minimizer_pool.append(self.HC.V[x])

                # 如果设置了显示标志，则记录邻居信息
                if self.disp:
                    logging.info('Neighbors:')
                    logging.info('=' * 30)
                    for vn in self.HC.V[x].nn:
                        logging.info(f'x = {vn.x} || f = {vn.f}')

                    logging.info('=' * 60)

        # 初始化存储最小化器 f 值的列表和最小化器 x 值的列表
        self.minimizer_pool_F = []
        self.X_min = []

        # 初始化在顶点缓存中存储标准化元组的字典（超立方体采样中使用的缓存）
        self.X_min_cache = {}

        # 遍历最小化器池中的每个顶点
        for v in self.minimizer_pool:
            # 将顶点的 x_a 添加到 X_min 列表中
            self.X_min.append(v.x_a)
            # 将顶点的 f 添加到 minimizer_pool_F 列表中
            self.minimizer_pool_F.append(v.f)
            # 使用顶点的 x_a 作为键，顶点的 x 作为值，添加到 X_min_cache 字典中
            self.X_min_cache[tuple(v.x_a)] = v.x

        # 将最小化器 f 值列表转换为 NumPy 数组
        self.minimizer_pool_F = np.array(self.minimizer_pool_F)
        # 将最小化器 x 值列表转换为 NumPy 数组
        self.X_min = np.array(self.X_min)

        # 如果是全局模式，则执行排序最小化器池中的元素
        # TODO: 只有在全局模式下才执行此操作
        self.sort_min_pool()

        # 返回最小化器 x 值的列表 X_min
        return self.X_min

    # 本地最小化
    # 最小化器池处理
    def minimise_pool(self, force_iter=False):
        """
        This processing method can optionally minimise only the best candidate
        solutions in the minimiser pool

        Parameters
        ----------
        force_iter : int
                     Number of starting minimizers to process (can be specified
                     globally or locally)

        """
        # Find first local minimum
        # NOTE: Since we always minimize this value regardless it is a waste to
        # build the topograph first before minimizing
        # 调用 minimize 方法，对 minimizer_pool 中的第一个元素进行最小化处理，返回结果
        lres_f_min = self.minimize(self.X_min[0], ind=self.minimizer_pool[0])

        # Trim minimized point from current minimizer set
        # 从当前的 minimizer pool 中移除最小化的点
        self.trim_min_pool(0)

        while not self.stop_l_iter:
            # Global stopping criteria:
            # 全局停止条件检查
            self.stopping_criteria()

            # Note first iteration is outside loop:
            # 注意：第一次迭代是在循环外进行的
            if force_iter:
                force_iter -= 1
                if force_iter == 0:
                    self.stop_l_iter = True
                    break

            if np.shape(self.X_min)[0] == 0:
                self.stop_l_iter = True
                break

            # Construct topograph from current minimizer set
            # (NOTE: This is a very small topograph using only the minizer pool
            #        , it might be worth using some graph theory tools instead.
            # 从当前的 minimizer set 构建拓扑图
            # 注意：这是一个非常小的拓扑图，只使用 minimizer pool，可能值得使用一些图论工具来代替。
            self.g_topograph(lres_f_min.x, self.X_min)

            # Find local minimum at the miniser with the greatest Euclidean
            # distance from the current solution
            # 在与当前解最大欧氏距离的 miniser 处找到局部最小值
            ind_xmin_l = self.Z[:, -1]
            lres_f_min = self.minimize(self.Ss[-1, :], self.minimizer_pool[-1])

            # Trim minimised point from current minimizer set
            # 从当前的 minimizer pool 中移除最小化的点
            self.trim_min_pool(ind_xmin_l)

        # Reset controls
        # 重置控制变量
        self.stop_l_iter = False
        return

    def sort_min_pool(self):
        # Sort to find minimum func value in min_pool
        # 对 min_pool 中的最小函数值进行排序
        self.ind_f_min = np.argsort(self.minimizer_pool_F)
        self.minimizer_pool = np.array(self.minimizer_pool)[self.ind_f_min]
        self.minimizer_pool_F = np.array(self.minimizer_pool_F)[
            self.ind_f_min]
        return

    def trim_min_pool(self, trim_ind):
        # Trim minimised point from minimizer pool
        # 从 minimizer pool 中移除最小化的点
        self.X_min = np.delete(self.X_min, trim_ind, axis=0)
        self.minimizer_pool_F = np.delete(self.minimizer_pool_F, trim_ind)
        self.minimizer_pool = np.delete(self.minimizer_pool, trim_ind)
        return
    def g_topograph(self, x_min, X_min):
        """
        Returns the topographical vector stemming from the specified value
        ``x_min`` for the current feasible set ``X_min`` with True boolean
        values indicating positive entries and False values indicating
        negative entries.

        """
        # Convert x_min to a NumPy array
        x_min = np.array([x_min])
        # Calculate Euclidean distances between x_min and each point in X_min
        self.Y = spatial.distance.cdist(x_min, X_min, 'euclidean')
        # Find sorted indexes of spatial distances
        self.Z = np.argsort(self.Y, axis=-1)

        # Select the closest point in X_min based on sorted indexes
        self.Ss = X_min[self.Z][0]
        # Rearrange minimizer pool based on sorted indexes
        self.minimizer_pool = self.minimizer_pool[self.Z]
        # Select the first element from the rearranged minimizer pool
        self.minimizer_pool = self.minimizer_pool[0]
        # Return the selected closest point in X_min
        return self.Ss

    # Local bound functions
    def construct_lcb_simplicial(self, v_min):
        """
        Construct locally (approximately) convex bounds

        Parameters
        ----------
        v_min : Vertex object
                The minimizer vertex

        Returns
        -------
        cbounds : list of lists
            List of size dimension with length-2 list of bounds for each
            dimension.

        """
        # Initialize cbounds with the current bounds
        cbounds = [[x_b_i[0], x_b_i[1]] for x_b_i in self.bounds]
        
        # Loop over nearest neighbors of v_min
        for vn in v_min.nn:
            # Loop over coordinates of each nearest neighbor
            for i, x_i in enumerate(vn.x_a):
                # Update lower bound if x_i is lower than the current lower bound
                if (x_i < v_min.x_a[i]) and (x_i > cbounds[i][0]):
                    cbounds[i][0] = x_i

                # Update upper bound if x_i is higher than the current upper bound
                if (x_i > v_min.x_a[i]) and (x_i < cbounds[i][1]):
                    cbounds[i][1] = x_i

        # Log cbounds if disp is True
        if self.disp:
            logging.info(f'cbounds found for v_min.x_a = {v_min.x_a}')
            logging.info(f'cbounds = {cbounds}')

        # Return updated bounds
        return cbounds

    def construct_lcb_delaunay(self, v_min, ind=None):
        """
        Construct locally (approximately) convex bounds

        Parameters
        ----------
        v_min : Vertex object
                The minimizer vertex

        Returns
        -------
        cbounds : list of lists
            List of size dimension with length-2 list of bounds for each
            dimension.
        """
        # Initialize cbounds with the current bounds
        cbounds = [[x_b_i[0], x_b_i[1]] for x_b_i in self.bounds]

        # Return the initialized bounds
        return cbounds
    # 用于计算局部最小值的函数，使用指定的采样点作为起始值
    def minimize(self, x_min, ind=None):
        """
        This function is used to calculate the local minima using the specified
        sampling point as a starting value.

        Parameters
        ----------
        x_min : vector of floats
            Current starting point to minimize.

        Returns
        -------
        lres : OptimizeResult
            The local optimization result represented as a `OptimizeResult`
            object.
        """
        # 如果设置了显示选项，记录顶点最小化映射信息
        if self.disp:
            logging.info(f'Vertex minimiser maps = {self.LMC.v_maps}')

        # 如果顶点已经运行过，则返回之前计算的结果
        if self.LMC[x_min].lres is not None:
            logging.info(f'Found self.LMC[x_min].lres = '
                         f'{self.LMC[x_min].lres}')
            return self.LMC[x_min].lres

        # 如果设置了回调函数，记录最小化起始点的回调信息
        if self.callback is not None:
            logging.info(f'Callback for minimizer starting at {x_min}:')

        # 如果设置了显示选项，记录最小化的起始点
        if self.disp:
            logging.info(f'Starting minimization at {x_min}...')

        # 如果采样方法是'simplicial'，则处理边界条件
        if self.sampling_method == 'simplicial':
            x_min_t = tuple(x_min)
            # 在顶点缓存中找到规范化的元组
            x_min_t_norm = self.X_min_cache[tuple(x_min_t)]
            x_min_t_norm = tuple(x_min_t_norm)
            g_bounds = self.construct_lcb_simplicial(self.HC.V[x_min_t_norm])
            # 如果在最小化求解器参数中指定了边界条件，则更新最小化器参数
            if 'bounds' in self.min_solver_args:
                self.minimizer_kwargs['bounds'] = g_bounds
                logging.info(self.minimizer_kwargs['bounds'])

        # 如果采样方法不是'simplicial'，则使用其他方法处理边界条件
        else:
            g_bounds = self.construct_lcb_delaunay(x_min, ind=ind)
            # 如果在最小化求解器参数中指定了边界条件，则更新最小化器参数
            if 'bounds' in self.min_solver_args:
                self.minimizer_kwargs['bounds'] = g_bounds
                logging.info(self.minimizer_kwargs['bounds'])

        # 如果设置了显示选项，并且最小化求解器参数中包含边界条件，则记录边界信息
        if self.disp and 'bounds' in self.minimizer_kwargs:
            logging.info('bounds in kwarg:')
            logging.info(self.minimizer_kwargs['bounds'])

        # 使用scipy.optimize.minimize进行局部最小化
        lres = minimize(self.func, x_min, **self.minimizer_kwargs)

        # 如果设置了显示选项，记录局部最小化的结果
        if self.disp:
            logging.info(f'lres = {lres}')

        # 更新总体最小化结果的函数评估次数
        self.res.nlfev += lres.nfev
        if 'njev' in lres:
            self.res.nljev += lres.njev
        if 'nhev' in lres:
            self.res.nlhev += lres.nhev

        # 处理由于NumPy数组的特定情况而引起的异常
        try:  # Needed because of the brain dead 1x1 NumPy arrays
            lres.fun = lres.fun[0]
        except (IndexError, TypeError):
            lres.fun

        # 将最小化结果添加到最小化映射集合中
        self.LMC[x_min]
        self.LMC.add_res(x_min, lres, bounds=g_bounds)

        # 返回局部最小化的结果
        return lres

    # 局部最小化处理后的后处理
    def sort_result(self):
        """
        Sort results and build the global return object
        """
        # 从本地最小化缓存中排序结果
        results = self.LMC.sort_cache_result()
        # 将排序后的结果赋给全局返回对象的属性
        self.res.xl = results['xl']
        self.res.funl = results['funl']
        self.res.x = results['x']
        self.res.fun = results['fun']

        # 将局部函数评估添加到采样函数评估中
        # 计算可行顶点的数量并添加到局部函数评估中：
        self.res.nfev = self.fn + self.res.nlfev
        # 返回结果对象
        return self.res

    # 算法控制
    def fail_routine(self, mes=("Failed to converge")):
        # 设置中断例程标志为True
        self.break_routine = True
        # 设置成功标志为False
        self.res.success = False
        # 设置X_min为一个长度为1的空列表
        self.X_min = [None]
        # 设置结果消息
        self.res.message = mes

    def sampled_surface(self, infty_cons_sampl=False):
        """
        Sample the function surface.

        There are 2 modes, if ``infty_cons_sampl`` is True then the sampled
        points that are generated outside the feasible domain will be
        assigned an ``inf`` value in accordance with SHGO rules.
        This guarantees convergence and usually requires less objective
        function evaluations at the computational costs of more Delaunay
        triangulation points.

        If ``infty_cons_sampl`` is False, then the infeasible points are
        discarded and only a subspace of the sampled points are used. This
        comes at the cost of the loss of guaranteed convergence and usually
        requires more objective function evaluations.
        """
        # 生成采样点
        if self.disp:
            logging.info('Generating sampling points')
        # 调用采样方法生成点集
        self.sampling(self.nc, self.dim)
        # 如果存在局部最小化映射，将其加入C中
        if len(self.LMC.xl_maps) > 0:
            self.C = np.vstack((self.C, np.array(self.LMC.xl_maps)))
        # 如果不是无穷约束采样
        if not infty_cons_sampl:
            # 寻找可行点的子空间
            if self.g_cons is not None:
                self.sampling_subspace()

        # 对剩余的样本进行排序
        self.sorted_samples()

        # 计算目标函数的参考值
        self.n_sampled = self.nc

    def sampling_custom(self, n, dim):
        """
        Generates uniform sampling points in a hypercube and scales the points
        to the bound limits.
        """
        # 生成自定义的采样点
        # 在超立方体中生成均匀采样点，并缩放到边界限制内
        # 如果尚未进行采样
        if self.n_sampled == 0:
            self.C = self.sampling_function(n, dim)
        else:
            self.C = self.sampling_function(n, dim)
        # 按边界分布
        for i in range(len(self.bounds)):
            self.C[:, i] = (self.C[:, i] *
                            (self.bounds[i][1] - self.bounds[i][0])
                            + self.bounds[i][0])
        # 返回采样点集
        return self.C
    def sampling_subspace(self):
        """Find subspace of feasible points from g_func definition"""
        # 遍历约束函数列表，检查每个约束在当前采样点集合上的可行性
        for ind, g in enumerate(self.g_cons):
            # 构建布尔数组，标记每个采样点是否满足当前约束
            feasible = np.array(
                [np.all(g(x_C, *self.g_args[ind]) >= 0.0) for x_C in self.C],
                dtype=bool
            )
            # 根据可行性布尔数组筛选出满足约束的采样点集合
            self.C = self.C[feasible]

            if self.C.size == 0:
                # 若无可行采样点，则更新结果消息，提示增加采样量
                self.res.message = ('No sampling point found within the '
                                    + 'feasible set. Increasing sampling '
                                    + 'size.')
                # 若需要显示信息，则记录消息到日志
                if self.disp:
                    logging.info(self.res.message)

    def sorted_samples(self):  # Validated
        """Find indexes of the sorted sampling points"""
        # 对采样点集合进行排序，并记录排序后的索引及排序后的采样点集合
        self.Ind_sorted = np.argsort(self.C, axis=0)
        self.Xs = self.C[self.Ind_sorted]
        return self.Ind_sorted, self.Xs

    def delaunay_triangulation(self, n_prc=0):
        if hasattr(self, 'Tri') and self.qhull_incremental:
            # 如果存在 Tri 属性且支持增量构建，则添加新的采样点到三角剖分对象中
            self.Tri.add_points(self.C[n_prc:, :])
        else:
            try:
                # 尝试创建 Delaunay 三角剖分对象，处理可能的 QhullError 异常
                self.Tri = spatial.Delaunay(self.C,
                                            incremental=self.qhull_incremental,
                                            )
            except spatial.QhullError:
                if str(sys.exc_info()[1])[:6] == 'QH6239':
                    # 处理 Qhull 错误 QH6239，提示用户可能的性能下降
                    logging.warning('QH6239 Qhull precision error detected, '
                                    'this usually occurs when no bounds are '
                                    'specified, Qhull can only run with '
                                    'handling cocircular/cospherical points'
                                    ' and in this case incremental mode is '
                                    'switched off. The performance of shgo '
                                    'will be reduced in this mode.')
                    # 自动禁用增量模式并重新尝试创建 Delaunay 三角剖分对象
                    self.qhull_incremental = False
                    self.Tri = spatial.Delaunay(self.C,
                                                incremental=self.qhull_incremental)
                else:
                    # 如果不是 QH6239 错误，则重新引发异常
                    raise

        return self.Tri
class LMap:
    def __init__(self, v):
        self.v = v  # Initialize instance variable v
        self.x_l = None  # Initialize instance variable x_l
        self.lres = None  # Initialize instance variable lres
        self.f_min = None  # Initialize instance variable f_min
        self.lbounds = []  # Initialize instance variable lbounds


class LMapCache:
    def __init__(self):
        self.cache = {}  # Initialize dictionary to cache mappings

        # Lists for search queries
        self.v_maps = []  # Initialize list for v mappings
        self.xl_maps = []  # Initialize list for xl mappings
        self.xl_maps_set = set()  # Initialize set for xl mappings to avoid duplicates
        self.f_maps = []  # Initialize list for f mappings
        self.lbound_maps = []  # Initialize list for lbounds mappings
        self.size = 0  # Initialize size counter for cache entries

    def __getitem__(self, v):
        try:
            v = np.ndarray.tolist(v)  # Convert v to list if it's a numpy ndarray
        except TypeError:
            pass
        v = tuple(v)  # Convert v to tuple
        try:
            return self.cache[v]  # Return cached value if v exists in cache
        except KeyError:
            xval = LMap(v)  # Create new LMap object with v as key
            self.cache[v] = xval  # Cache the new LMap object
            return self.cache[v]  # Return the cached LMap object

    def add_res(self, v, lres, bounds=None):
        v = np.ndarray.tolist(v)  # Convert v to list if it's a numpy ndarray
        v = tuple(v)  # Convert v to tuple
        self.cache[v].x_l = lres.x  # Assign lres.x to x_l of cached LMap object
        self.cache[v].lres = lres  # Assign lres to lres of cached LMap object
        self.cache[v].f_min = lres.fun  # Assign lres.fun to f_min of cached LMap object
        self.cache[v].lbounds = bounds  # Assign bounds to lbounds of cached LMap object

        # Update cache size
        self.size += 1

        # Cache lists for search queries
        self.v_maps.append(v)  # Add v to v_maps list
        self.xl_maps.append(lres.x)  # Add lres.x to xl_maps list
        self.xl_maps_set.add(tuple(lres.x))  # Add tuple(lres.x) to xl_maps_set (set of xl mappings)
        self.f_maps.append(lres.fun)  # Add lres.fun to f_maps list
        self.lbound_maps.append(bounds)  # Add bounds to lbound_maps list

    def sort_cache_result(self):
        """
        Sort results and build the global return object
        """
        results = {}  # Initialize dictionary for results

        # Sort results and save
        self.xl_maps = np.array(self.xl_maps)  # Convert xl_maps to numpy array
        self.f_maps = np.array(self.f_maps)  # Convert f_maps to numpy array

        # Sorted indexes in Func_min
        ind_sorted = np.argsort(self.f_maps)  # Get indices that would sort f_maps

        # Save ordered list of minima
        results['xl'] = self.xl_maps[ind_sorted]  # Assign ordered xl values to results['xl']
        self.f_maps = np.array(self.f_maps)  # Convert f_maps to numpy array
        results['funl'] = self.f_maps[ind_sorted]  # Assign ordered fun values to results['funl']
        results['funl'] = results['funl'].T  # Transpose results['funl']

        # Find global of all minimizers
        results['x'] = self.xl_maps[ind_sorted[0]]  # Assign global minimum xl to results['x']
        results['fun'] = self.f_maps[ind_sorted[0]]  # Assign global minimum fun value to results['fun']

        self.xl_maps = np.ndarray.tolist(self.xl_maps)  # Convert xl_maps back to list
        self.f_maps = np.ndarray.tolist(self.f_maps)  # Convert f_maps back to list
        return results  # Return the sorted results dictionary
```