# `D:\src\scipysrc\scipy\scipy\optimize\_differentialevolution.py`

```
# 导入警告模块，用于可能的警告信息
import warnings

# 导入科学计算库NumPy
import numpy as np

# 导入全局优化相关模块和函数
from scipy.optimize import OptimizeResult, minimize

# 导入优化算法内部函数和状态信息函数
from scipy.optimize._optimize import _status_message, _wrap_callback

# 导入科学计算库中的实用工具函数和类
from scipy._lib._util import (check_random_state, MapWrapper, _FunctionWrapper,
                              rng_integers)

# 导入优化算法中的约束相关类
from scipy.optimize._constraints import (Bounds, new_bounds_to_old,
                                         NonlinearConstraint, LinearConstraint)

# 导入稀疏矩阵判断函数
from scipy.sparse import issparse

# 定义模块对外公开的函数和类列表
__all__ = ['differential_evolution']

# 定义机器精度
_MACHEPS = np.finfo(np.float64).eps


def differential_evolution(func, bounds, args=(), strategy='best1bin',
                           maxiter=1000, popsize=15, tol=0.01,
                           mutation=(0.5, 1), recombination=0.7, seed=None,
                           callback=None, disp=False, polish=True,
                           init='latinhypercube', atol=0, updating='immediate',
                           workers=1, constraints=(), x0=None, *,
                           integrality=None, vectorized=False):
    """
    使用差分进化算法寻找多元函数的全局最小值。

    差分进化算法 [1]_ 是一种随机算法。它不使用梯度方法来寻找最小值，
    可以搜索候选空间的大范围，但通常需要比传统的基于梯度的技术更多的函数评估次数。

    该算法由Storn和Price提出 [2]_ 。

    Parameters
    ----------
    func : callable
        要最小化的目标函数。必须是形如 ``f(x, *args)`` 的函数，其中 ``x`` 是一个
        1-D 数组形式的参数，``args`` 是一个元组，包含完全指定函数所需的任何额外固定参数。
        参数的数量N等于 ``len(x)``。
    bounds : sequence or `Bounds`
        变量的边界。有两种指定边界的方式：

            1. `Bounds` 类的实例。
            2. 对于 ``x`` 中的每个元素，定义 `func` 的优化参数的有限下限和上限边界的 ``(min, max)`` 对。

        边界的总数用于确定参数的数量N。如果有边界相等的参数，自由参数的总数是 ``N - N_equal``。
    args : tuple, optional
        完全指定目标函数所需的任何额外固定参数。
    # 设置差分进化算法的策略，可以是字符串或可调用对象
    strategy : {str, callable}, optional
        差分进化算法使用的策略，可选的字符串包括：
    
            - 'best1bin'
            - 'best1exp'
            - 'rand1bin'
            - 'rand1exp'
            - 'rand2bin'
            - 'rand2exp'
            - 'randtobest1bin'
            - 'randtobest1exp'
            - 'currenttobest1bin'
            - 'currenttobest1exp'
            - 'best2exp'
            - 'best2bin'
    
        默认为 'best1bin'。可通过提供一个可调用对象来定制差分进化策略，该对象必须具有以下形式：
        ``strategy(candidate: int, population: np.ndarray, rng=None)``
        其中 ``candidate`` 是要进化的种群的索引，``population`` 是形状为 ``(S, N)`` 的数组，
        包含所有种群成员（其中 S 是种群总大小），``rng`` 是求解器内使用的随机数生成器。
        ``candidate`` 的范围是 ``[0, S)``。
        ``strategy`` 必须返回一个形状为 ``(N,)`` 的试验向量。将该试验向量的适应度与 ``population[candidate]`` 的适应度进行比较。
    
        .. versionchanged:: 1.12.0
            通过可调用对象定制进化策略。
    
    # 最大迭代次数，整个种群进行进化的最大代数
    maxiter : int, optional
        整个种群进行进化的最大代数。如果没有精炼，最大函数评估次数为：
        ``(maxiter + 1) * popsize * (N - N_equal)``
    
    # 种群大小的乘数，种群有 ``popsize * (N - N_equal)`` 个个体
    popsize : int, optional
        用于设置总种群大小的乘数。种群有 ``popsize * (N - N_equal)`` 个个体。
        如果通过 `init` 关键字提供了初始种群，则此关键字将被覆盖。
        当使用 ``init='sobol'`` 时，种群大小计算为 ``popsize * (N - N_equal)`` 的下一个 2 的幂。
    
    # 收敛的相对容差，当满足以下条件时求解停止：
    # ``np.std(population_energies) <= atol + tol * np.abs(np.mean(population_energies))``
    # 其中 `atol` 和 `tol` 分别是绝对容差和相对容差。
    tol : float, optional
        收敛的相对容差，当满足以下条件时求解停止：
        ``np.std(population_energies) <= atol + tol * np.abs(np.mean(population_energies))``
        其中 `atol` 和 `tol` 分别是绝对容差和相对容差。
    mutation : float or tuple(float, float), optional
        # 控制变异操作的参数，也称为差分权重，通常表示为 F
        # 如果指定为 float 类型，应在 [0, 2] 的范围内
        # 如果指定为 tuple ``(min, max)``，则进行抖动操作
        # 抖动会在每一代随机改变变异常数，当前代的变异常数取自于 ``U[min, max)`` 区间
        # 抖动可以显著加快收敛速度
        # 增加变异常数会增加搜索半径，但会减慢收敛速度。
    recombination : float, optional
        # 重组常数，应在 [0, 1] 的范围内。在文献中也称为交叉概率，表示为 CR
        # 增加此值允许更多的变异体进入下一代，但会增加种群不稳定性的风险。
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        # 随机种子。如果 `seed` 是 None 或者 `np.random`，则使用 `numpy.random.RandomState` 单例。
        # 如果 `seed` 是一个整数，则使用一个新的 ``RandomState`` 实例，并使用 `seed` 进行种子初始化。
        # 如果 `seed` 已经是 ``Generator`` 或 ``RandomState`` 实例，则直接使用该实例。
        # 为了重复性的最小化结果，请指定 `seed`。
    disp : bool, optional
        # 是否在每次迭代时打印评估的 `func` 值。
    callback : callable, optional
        # 每次迭代后调用的可调用对象。具有以下签名：
        #     ``callback(intermediate_result: OptimizeResult)``
        # 其中 ``intermediate_result`` 是一个关键字参数，包含一个 `OptimizeResult` 对象，具有 ``x`` 和 ``fun`` 属性，
        # 分别表示迄今为止找到的最佳解和目标函数值。
        # 注意，参数的名称必须为 ``intermediate_result`` 才能将 `OptimizeResult` 传递给回调函数。
        # 回调函数还支持如下签名：
        #     ``callback(x, convergence: float=val)``
        # ``val`` 表示种群收敛的分数值。当 ``val`` 大于 ``1.0`` 时，函数停止。
        # 使用内省确定调用哪个签名。
        # 如果回调函数引发 ``StopIteration`` 或返回 ``True``，则全局最小化将停止；仍会执行任何精炼操作。
        #
        # .. versionchanged:: 1.12.0
        #     callback 现在接受 ``intermediate_result`` 关键字。
    # 是否进行最终的优化操作，默认为 True
    polish : bool, optional
        # 如果为 True，则使用 `scipy.optimize.minimize` 函数和 `L-BFGS-B` 方法对最佳种群成员进行优化
        # 如果问题是有约束的，则使用 `trust-constr` 方法代替
        # 对于包含许多约束的大问题，由于雅可比矩阵的计算，优化操作可能需要很长时间
    init : str or array-like, optional
        # 指定种群初始化类型的选择
        # 可以是以下之一：
        # - 'latinhypercube'
        # - 'sobol'
        # - 'halton'
        # - 'random'
        # 或者是形状为 ``(S, N)`` 的数组，其中 S 是总种群大小，N 是参数数量
        # `init` 在使用之前会被裁剪到 `bounds` 范围内
        # 默认值为 'latinhypercube'，拉丁超立方采样试图最大化参数空间的覆盖率
        # 'sobol' 和 'halton' 是更优的替代方案，可以更大程度地最大化参数空间
        # 'sobol' 将确保初始种群大小为 ``popsize * (N - N_equal)`` 的下一个 2 的幂
        # 'halton' 没有特定要求，但效率稍低
        # 详细信息请参见 `scipy.stats.qmc`
        # 'random' 使用随机初始化种群 - 缺点是可能会发生聚类，从而阻止整个参数空间的覆盖
        # 使用数组指定种群可以用来在已知解存在的位置紧密初始化一组初始猜测，从而缩短收敛时间
    atol : float, optional
        # 收敛的绝对容差，当 ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))`` 时停止求解
        # 其中 `atol` 和 `tol` 分别是绝对容差和相对容差
    updating : {'immediate', 'deferred'}, optional
        # 如果是 ``'immediate'``，最佳解向量在单个代的内部持续更新
        # 这可以加快收敛速度，因为试验向量可以利用最佳解的持续改进
        # 如果是 ``'deferred'``，最佳解向量每一代只更新一次
        # 只有 ``'deferred'`` 兼容并行化或向量化，`workers` 和 `vectorized` 关键字可以覆盖此选项
        # .. versionadded:: 1.2.0
    workers : int or map-like callable, optional
        如果 `workers` 是一个整数，则将种群分为 `workers` 个部分并并行评估
        （使用 `multiprocessing.Pool <multiprocessing>`）。
        提供 -1 来使用所有可用的 CPU 核心。
        或者提供一个类似映射的可调用对象，例如 `multiprocessing.Pool.map`，
        用于并行评估种群。这种评估将作为 ``workers(func, iterable)`` 进行。
        如果 ``workers != 1``，此选项将覆盖 `updating` 关键字为 ``updating='deferred'``。
        如果 ``workers != 1``，此选项将覆盖 `vectorized` 关键字。
        要求 `func` 是可被 pickle 序列化的。

        .. versionadded:: 1.2.0

    constraints : {NonLinearConstraint, LinearConstraint, Bounds}
        求解器的约束条件，除了 `bounds` 关键字应用的约束之外。
        使用 Lampinen [5]_ 的方法。

        .. versionadded:: 1.4.0

    x0 : None or array-like, optional
        提供最小化问题的初始猜测。一旦种群被初始化，该向量将替换第一个（最佳）成员。
        即使 `init` 给出了一个初始种群，这种替换也会发生。
        ``x0.shape == (N,)``。

        .. versionadded:: 1.7.0

    integrality : 1-D array, optional
        对于每个决策变量，一个布尔值，指示决策变量是否被限制为整数值。
        数组会被广播到 ``(N,)``。
        如果任何决策变量被约束为整数，它们在优化过程中将不会改变。
        只有在下限和上限之间的整数值会被使用。
        如果在边界之间没有整数值，则会引发 `ValueError`。

        .. versionadded:: 1.9.0

    vectorized : bool, optional
        如果 ``vectorized is True``，`func` 将会收到一个形状为 ``(N, S)`` 的 `x` 数组，
        并且预期返回一个形状为 ``(S,)`` 的数组，其中 `S` 是要计算的解向量数量。
        如果应用了约束条件，用于构建 `Constraint` 对象的每个函数应该接受形状为 ``(N, S)`` 的 `x` 数组，
        并返回形状为 ``(M, S)`` 的数组，其中 `M` 是约束组件的数量。
        此选项是 `workers` 提供的并行化的替代方案，可以通过减少多个函数调用的解释器开销来加快优化速度。
        如果 ``workers != 1``，则此关键字被忽略。
        如果 ``workers != 1``，此选项将覆盖 `updating` 关键字为 ``updating='deferred'``。
        关于何时使用 ``'vectorized'`` 和何时使用 ``'workers'``，请参见备注部分的进一步讨论。

        .. versionadded:: 1.9.0

    Returns
    -------
    # res : OptimizeResult
    #     优化结果，表示为 `OptimizeResult` 对象。
    #     重要的属性包括：``x`` 解数组，``success`` 布尔标志，指示优化器是否成功退出，
    #     ``message`` 描述终止原因，
    #     ``population`` 存在于种群中的解向量，
    #     ``population_energies`` 每个解在目标函数中的值。
    #     参见 `OptimizeResult` 获取其他属性的描述。如果使用了 `polish`，并且通过抛光获得了更低的最小值，
    #     则 `OptimizeResult` 还包含 ``jac`` 属性。
    #     如果最终的解决方案不满足应用的约束条件，则 ``success`` 将为 `False`。

    # Notes
    # -----
    # 差分进化是一种基于随机种群的方法，适用于全局优化问题。在每次种群迭代中，
    # 算法通过将每个候选解与其他候选解混合来创建一个试验候选解进行变异。
    # 有几种策略 [3]_ 用于创建试验候选解，适合不同的问题。'best1bin' 策略对许多系统来说是一个良好的起点。
    # 在这种策略中，从种群中随机选择两个成员。它们的差异用于变异到目前为止最好的成员（在 'best' 中的 'best1bin' 中的 'best'）, :math:`x_0`:

    # .. math::

    #     b' = x_0 + mutation * (x_{r_0} - x_{r_1})

    # 然后构造一个试验向量。从随机选择的第 i 个参数开始，试验顺序地从 ``b'`` 或原始候选解中填充（按模数）参数。
    # 使用二项分布（'best1bin' 中的 'bin'）来决定是使用 ``b'`` 还是原始候选解。生成一个范围为 [0, 1) 的随机数。
    # 如果这个数字小于 `recombination` 常数，则加载参数来自 ``b'``，否则加载参数来自原始候选解。
    # 最后一个参数始终从 ``b'`` 加载。构建试验候选解后，评估其适应度。
    # 如果试验候选解比原始候选解更好，则替换原始候选解。如果它还优于全局最佳候选解，则也替换该解。

    # 可用的其他策略在 Qiang 和 Mitchell（2014年）[3]_ 中有概述。
    # Differential evolution 算法中的不同变体，用于生成候选解向量 b'
    .. math::
            rand1* : b' = x_{r_0} + mutation*(x_{r_1} - x_{r_2})

            rand2* : b' = x_{r_0} + mutation*(x_{r_1} + x_{r_2}
                                                - x_{r_3} - x_{r_4})

            best1* : b' = x_0 + mutation*(x_{r_0} - x_{r_1})

            best2* : b' = x_0 + mutation*(x_{r_0} + x_{r_1}
                                            - x_{r_2} - x_{r_3})

            currenttobest1* : b' = x_i + mutation*(x_0 - x_i
                                                     + x_{r_0} - x_{r_1})

            randtobest1* : b' = x_{r_0} + mutation*(x_0 - x_{r_0}
                                                      + x_{r_1} - x_{r_2})

    # 在这些公式中，整数 :math:`r_0, r_1, r_2, r_3, r_4` 是从区间 [0, NP) 中随机选择的索引，
    # 其中 NP 是总体数量，索引 i 表示原始候选解向量的索引。用户可以通过向 strategy 参数提供可调用对象，
    # 完全定制生成试验候选解向量的方式。

    # 为了增加找到全局最小值的机会，建议增大 popsize（种群大小）、mutation（变异率）和 dithering（微调），
    # 同时减小 recombination（重组率）。这样做可以扩大搜索半径，但会减慢收敛速度。

    # 默认情况下，最佳解向量在单个迭代中持续更新（``updating='immediate'``）。这是对原始差分进化算法的修改 [4]，
    # 可以加快收敛速度，因为试验向量可以立即从改进的解中受益。若要使用原始的 Storn 和 Price 行为，
    # 每次迭代只更新一次最佳解向量，请设置 ``updating='deferred'``。

    # ``'deferred'`` 方法支持并行化和向量化（``'workers'`` 和 ``'vectorized'`` 关键字）。
    # 这些方法可以通过更有效地使用计算资源来提高最小化速度。``'workers'`` 可以将计算分布到多个处理器上。
    # 默认情况下使用 Python 的 multiprocessing 模块，但也可以使用其他方法，如在集群上使用消息传递接口（MPI） [6]_ [7]_。
    # 这些方法的开销（创建新进程等）可能很大，因此计算速度不一定随处理器数量的增加而线性提升。
    # 并行化特别适用于计算开销较大的目标函数。如果目标函数较为简单，则 ``'vectorized'`` 可以通过每次迭代只调用一次目标函数，
    # 而不是为所有种群成员多次调用来减少解释器开销。

    # .. versionadded:: 0.15.0

    # 参考文献
    # ----------
    # .. [1] Differential evolution, Wikipedia,
    #        http://en.wikipedia.org/wiki/Differential_evolution
    # 导入必要的库和函数
    >>> import numpy as np
    >>> from scipy.optimize import rosen, differential_evolution
    
    # 设定优化变量的范围
    >>> bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]
    
    # 使用 differential_evolution 函数对 Rosenbrock 函数进行优化
    >>> result = differential_evolution(rosen, bounds)
    
    # 打印优化结果的最优解和目标函数值
    >>> result.x, result.fun
    (array([1., 1., 1., 1., 1.]), 1.9216496320061384e-19)

    # 使用 deferred 更新策略和两个 worker 进行并行优化
    >>> result = differential_evolution(rosen, bounds, updating='deferred',
    ...                                 workers=2)
    
    # 打印并行优化的结果的最优解和目标函数值
    >>> result.x, result.fun
    (array([1., 1., 1., 1., 1.]), 1.9216496320061384e-19)

    # 定义线性约束条件，使得 x[0] + x[1] <= 1.9
    >>> lc = LinearConstraint([[1, 1]], -np.inf, 1.9)
    
    # 设定变量的上下限
    >>> bounds = Bounds([0., 0.], [2., 2.])
    
    # 使用 differential_evolution 函数进行带约束条件的优化
    >>> result = differential_evolution(rosen, bounds, constraints=lc,
    ...                                 seed=1)
    
    # 打印带约束条件优化的结果的最优解和目标函数值
    >>> result.x, result.fun
    (array([0.96632622, 0.93367155]), 0.0011352416852625719)

    # 定义 Ackley 函数
    >>> def ackley(x):
    ...     arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
    ...     arg2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi * x[1]))
    ...     return -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e
    
    # 设定 Ackley 函数的变量范围
    >>> bounds = [(-5, 5), (-5, 5)]
    
    # 使用 differential_evolution 函数对 Ackley 函数进行优化
    >>> result = differential_evolution(ackley, bounds, seed=1)
    
    # 打印 Ackley 函数优化结果的最优解和目标函数值
    >>> result.x, result.fun
    (array([0., 0.]), 4.440892098500626e-16)
    """# noqa: E501

    # 使用向量化的方式实现的Ackley函数，可以通过 'vectorized' 关键字来指定。
    # 注意到这种方式可以减少函数评估的次数。
    >>> result = differential_evolution(
    ...     ackley, bounds, vectorized=True, updating='deferred', seed=1
    ... )

    # 输出优化结果中的最优解和最优值
    >>> result.x, result.fun
    (array([0., 0.]), 4.440892098500626e-16)

    # 下面的自定义策略函数模仿了 'best1bin' 策略：

    # 定义一个自定义策略函数，接受候选解、种群和随机数生成器作为输入
    >>> def custom_strategy_fn(candidate, population, rng=None):
    ...     # 获取参数个数，假设 population 是一个二维数组
    ...     parameter_count = population.shape(-1)
    ...     # 设定变异率和重组率
    ...     mutation, recombination = 0.7, 0.9
    ...     # 复制候选解作为试验解
    ...     trial = np.copy(population[candidate])
    ...     # 随机选择填充点
    ...     fill_point = rng.choice(parameter_count)
    ...
    ...     # 创建种群索引数组，并打乱顺序
    ...     pool = np.arange(len(population))
    ...     rng.shuffle(pool)
    ...
    ...     # 选择两个唯一且不等于候选解的随机索引
    ...     idxs = []
    ...     while len(idxs) < 2 and len(pool) > 0:
    ...         idx = pool[0]
    ...         pool = pool[1:]
    ...         if idx != candidate:
    ...             idxs.append(idx)
    ...
    ...     r0, r1 = idxs[:2]
    ...
    ...     # 计算变异操作
    ...     bprime = (population[0] + mutation *
    ...               (population[r0] - population[r1]))
    ...
    ...     # 生成交叉点数组，并根据重组率进行设置
    ...     crossovers = rng.uniform(size=parameter_count)
    ...     crossovers = crossovers < recombination
    ...     crossovers[fill_point] = True
    ...     # 根据交叉点数组生成最终试验解
    ...     trial = np.where(crossovers, bprime, trial)
    ...     return trial

    # 使用上下文管理器来确保在结束时清理所有创建的 Pool 对象。
    with DifferentialEvolutionSolver(func, bounds, args=args,
                                     strategy=strategy,
                                     maxiter=maxiter,
                                     popsize=popsize, tol=tol,
                                     mutation=mutation,
                                     recombination=recombination,
                                     seed=seed, polish=polish,
                                     callback=callback,
                                     disp=disp, init=init, atol=atol,
                                     updating=updating,
                                     workers=workers,
                                     constraints=constraints,
                                     x0=x0,
                                     integrality=integrality,
                                     vectorized=vectorized) as solver:
        # 调用 solver.solve() 方法执行优化求解
        ret = solver.solve()

    # 返回求解结果
    return ret
class DifferentialEvolutionSolver:
    """This class implements the differential evolution solver

    Parameters
    ----------
    func : callable
        The objective function to be minimized. Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a tuple of any additional fixed parameters needed to
        completely specify the function. The number of parameters, N, is equal
        to ``len(x)``.
    bounds : sequence or `Bounds`
        Bounds for variables. There are two ways to specify the bounds:

            1. Instance of `Bounds` class.
            2. ``(min, max)`` pairs for each element in ``x``, defining the
               finite lower and upper bounds for the optimizing argument of
               `func`.

        The total number of bounds is used to determine the number of
        parameters, N. If there are parameters whose bounds are equal the total
        number of free parameters is ``N - N_equal``.
    args : tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.
    strategy : {str, callable}, optional
        The differential evolution strategy to use. Should be one of:

            - 'best1bin'
            - 'best1exp'
            - 'rand1bin'
            - 'rand1exp'
            - 'rand2bin'
            - 'rand2exp'
            - 'randtobest1bin'
            - 'randtobest1exp'
            - 'currenttobest1bin'
            - 'currenttobest1exp'
            - 'best2exp'
            - 'best2bin'

        The default is 'best1bin'. Strategies that may be
        implemented are outlined in 'Notes'.

        Alternatively the differential evolution strategy can be customized
        by providing a callable that constructs a trial vector. The callable
        must have the form
        ``strategy(candidate: int, population: np.ndarray, rng=None)``,
        where ``candidate`` is an integer specifying which entry of the
        population is being evolved, ``population`` is an array of shape
        ``(S, N)`` containing all the population members (where S is the
        total population size), and ``rng`` is the random number generator
        being used within the solver.
        ``candidate`` will be in the range ``[0, S)``.
        ``strategy`` must return a trial vector with shape ``(N,)``. The
        fitness of this trial vector is compared against the fitness of
        ``population[candidate]``.
    maxiter : int, optional
        The maximum number of generations over which the entire population is
        evolved. The maximum number of function evaluations (with no polishing)
        is: ``(maxiter + 1) * popsize * (N - N_equal)``
    """
    # 总体种群大小的倍数，用于设置总体大小。种群大小为 ``popsize * (N - N_equal)`` 个个体。
    # 如果通过 `init` 关键字提供了初始种群，则此关键字会被覆盖。
    # 当使用 ``init='sobol'`` 时，种群大小被计算为 ``popsize * (N - N_equal)`` 的下一个2的幂。
    popsize : int, optional
    
    # 收敛的相对容差，当满足以下条件时求解停止：
    # ``np.std(population_energies) <= atol + tol * np.abs(np.mean(population_energies))``
    # 其中 `atol` 和 `tol` 分别是绝对容差和相对容差。
    tol : float, optional
    
    # 变异常数。在文献中也称为差分权重，用符号 F 表示。
    # 如果指定为一个 float，应在 [0, 2] 的范围内。
    # 如果指定为一个元组 ``(min, max)``，则采用抖动。抖动会在每一代中随机改变变异常数。
    # 对于每一代，变异常数取自于 U[min, max)。抖动可以显著加快收敛速度。
    # 增加变异常数会增加搜索半径，但会降低收敛速度。
    mutation : float or tuple(float, float), optional
    
    # 重组常数，应在 [0, 1] 的范围内。在文献中也称为交叉概率，用符号 CR 表示。
    # 增加此值允许更多的变异体进入下一代，但会增加种群稳定性的风险。
    recombination : float, optional
    
    # 随机数种子。如果 `seed` 是 None（或 `np.random`），则使用 `numpy.random.RandomState` 单例。
    # 如果 `seed` 是一个整数，则使用一个新的 ``RandomState`` 实例，种子为 `seed`。
    # 如果 `seed` 已经是一个 ``Generator`` 或 ``RandomState`` 实例，则直接使用该实例。
    # 为了可重复的最小化过程，请指定 `seed`。
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
    
    # 是否在每次迭代时打印评估的 `func`。
    disp : bool, optional
    callback : callable, optional
        # 可选参数，回调函数，在每次迭代后被调用。具有以下两种签名之一：
        
        # 第一种签名：
        # ``callback(intermediate_result: OptimizeResult)``
        # 其中 ``intermediate_result`` 是一个关键字参数，包含一个 `OptimizeResult`，
        # 其属性包括 ``x`` 和 ``fun``，分别表示迄今找到的最佳解和目标函数值。
        
        # 第二种签名：
        # ``callback(x, convergence: float=val)``
        # ``val`` 表示群体收敛的分数值。当 ``val`` 大于 ``1.0`` 时，函数停止执行。
        
        # 根据参数的名称来确定调用哪种签名方式。
        
        # 如果回调函数抛出 ``StopIteration`` 或返回 ``True``，全局最小化将会停止；
        # 但仍会继续进行优化处理。
        
        # .. versionchanged:: 1.12.0
        #    回调函数接受 ``intermediate_result`` 关键字。

    polish : bool, optional
        # 可选参数，默认为 True。如果为 True，则在最后使用 `scipy.optimize.minimize` 
        # 和 `L-BFGS-B` 方法对最佳群体成员进行优化处理，这可能会略微改善最小化过程。
        # 如果研究的是有约束条件的问题，则改为使用 `trust-constr` 方法。
        # 对于有许多约束条件的大问题，由于需要计算雅可比矩阵，优化处理可能会花费较长时间。

    maxfun : int, optional
        # 可选参数，设置最大函数评估次数。然而，设置 `maxiter` 会更有意义。
    # 初始化种群的方式，可以是字符串或类似数组的数据类型，可选参数
    # 指定种群初始化的方式，应为以下之一：
    
    # - 'latinhypercube'
    # - 'sobol'
    # - 'halton'
    # - 'random'
    # - 数组，指定初始化种群的具体值。数组的形状应为 ``(S, N)``, 其中 S 是总种群大小，N 是参数数量。
    #   `init` 在使用前将被剪切到 `bounds` 内。
    
    # 默认值为 'latinhypercube'。拉丁超立方采样试图最大化覆盖可用参数空间。
    
    # 'sobol' 和 'halton' 是更好的选择，能够进一步最大化参数空间。
    # 'sobol' 将强制设定一个初始种群大小，该大小计算方式为在 `popsize * (N - N_equal)` 后的下一个2的幂。
    # 'halton' 没有此类要求，但效率略低。详见 `scipy.stats.qmc` 获取更多细节。
    
    # 'random' 会随机初始化种群 - 这可能导致聚类，从而阻止覆盖整个参数空间。
    # 使用数组指定种群，例如，在已知解存在的位置创建紧密群集的初始猜测，从而减少收敛时间。
    init : str or array-like, optional
    
    # 收敛的绝对容差，求解停止的条件为
    # ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
    # 其中 `atol` 和 `tol` 分别为绝对容差和相对容差。
    atol : float, optional
    
    # 更新策略，可以是 'immediate' 或 'deferred' 之一，可选参数
    # 如果为 ``'immediate'``，最佳解向量在单个代中持续更新 [4]_。这可能导致更快的收敛，
    # 因为试验向量可以利用最佳解的持续改进。
    # 如果为 ``'deferred'``，最佳解向量每代更新一次。仅 ``'deferred'`` 兼容并行化或向量化，
    # `workers` 和 `vectorized` 关键字可以覆盖此选项。
    updating : {'immediate', 'deferred'}, optional
    
    # 工作者数，可以是整数或映射型可调用对象，可选参数
    # 如果 `workers` 是整数，则种群被分成 `workers` 个部分并行评估
    # （使用 `multiprocessing.Pool <multiprocessing>`）。
    # 提供 `-1` 使用所有可用的核心数。
    # 或者提供一个映射型可调用对象，如 `multiprocessing.Pool.map` 用于并行评估种群。
    # 此评估作为 ``workers(func, iterable)`` 进行。
    # 如果 `workers != 1`，此选项将覆盖 `updating` 关键字为 `updating='deferred'`。
    # 需要 `func` 可以 pickle。
    
    # 参见 `multiprocessing` 获取更多详情。
    workers : int or map-like callable, optional
    # 约束条件：非线性约束、线性约束或边界约束的集合，这些约束不受 `bounds` 参数的影响。
    # 使用 Lampinen 的方法处理。
    constraints : {NonLinearConstraint, LinearConstraint, Bounds}
    
    # 初始猜测向量，用于最小化过程的起始点。一旦种群初始化完成，这个向量会替换第一个（最好的）个体。
    # 即使给定了初始种群，也会进行替换。
    # `x0.shape == (N,)`
    x0 : None or array-like, optional
    
    # 整数性约束：一个一维数组，指示每个决策变量是否限制为整数值。数组会广播成 `(N,)` 形状。
    # 如果任何决策变量被约束为整数，则它们在优化过程中不会改变。
    # 只使用在边界之间的整数值。
    # 如果边界之间没有整数值，则会引发 `ValueError` 异常。
    integrality : 1-D array, optional
    
    # 向量化选项：如果 `vectorized` 为 `True`，则 `func` 函数接收形状为 `(N, S)` 的 `x` 数组，
    # 并期望返回形状为 `(S,)` 的数组，其中 `S` 是要计算解向量的数量。
    # 如果应用了约束条件，则用于构建 `Constraint` 对象的每个函数都应接受形状为 `(N, S)` 的 `x` 数组，
    # 并返回形状为 `(M, S)` 的数组，其中 `M` 是约束组件的数量。
    # 这个选项是对 `workers` 提供的并行化的替代，可能有助于提高优化速度。
    # 如果 `workers != 1`，则忽略此关键字。
    # 此选项将覆盖 `updating` 关键字，设置为 `updating='deferred'`。
    vectorized : bool, optional

""" # noqa: E501

# 突变策略方法的分派（二项式或指数型）。
_binomial = {'best1bin': '_best1',
             'randtobest1bin': '_randtobest1',
             'currenttobest1bin': '_currenttobest1',
             'best2bin': '_best2',
             'rand2bin': '_rand2',
             'rand1bin': '_rand1'}

_exponential = {'best1exp': '_best1',
                'rand1exp': '_rand1',
                'randtobest1exp': '_randtobest1',
                'currenttobest1exp': '_currenttobest1',
                'best2exp': '_best2',
                'rand2exp': '_rand2'}

__init_error_msg = ("The population initialization method must be one of "
                    "'latinhypercube' or 'random', or an array of shape "
                    "(S, N) where N is the number of parameters and S>5")
    def init_population_lhs(self):
        """
        Initializes the population with Latin Hypercube Sampling.
        Latin Hypercube Sampling ensures that each parameter is uniformly
        sampled over its range.
        """
        rng = self.random_number_generator

        # Each parameter range needs to be sampled uniformly. The scaled
        # parameter range ([0, 1)) needs to be split into
        # `self.num_population_members` segments, each of which has the following
        # size:
        segsize = 1.0 / self.num_population_members

        # Within each segment we sample from a uniform random distribution.
        # We need to do this sampling for each parameter.
        samples = (segsize * rng.uniform(size=self.population_shape)
                   # Offset each segment to cover the entire parameter range [0, 1)
                   + np.linspace(0., 1., self.num_population_members,
                                 endpoint=False)[:, np.newaxis])

        # Create an array for population of candidate solutions.
        self.population = np.zeros_like(samples)

        # Initialize population of candidate solutions by permutation of the
        # random samples.
        for j in range(self.parameter_count):
            order = rng.permutation(range(self.num_population_members))
            self.population[:, j] = samples[order, j]

        # reset population energies
        self.population_energies = np.full(self.num_population_members,
                                           np.inf)

        # reset number of function evaluations counter
        self._nfev = 0

    def init_population_qmc(self, qmc_engine):
        """Initializes the population with a QMC method.

        QMC methods ensures that each parameter is uniformly
        sampled over its range.

        Parameters
        ----------
        qmc_engine : str
            The QMC method to use for initialization. Can be one of
            ``latinhypercube``, ``sobol`` or ``halton``.

        """
        from scipy.stats import qmc

        rng = self.random_number_generator

        # Create an array for population of candidate solutions.
        if qmc_engine == 'latinhypercube':
            sampler = qmc.LatinHypercube(d=self.parameter_count, seed=rng)
        elif qmc_engine == 'sobol':
            sampler = qmc.Sobol(d=self.parameter_count, seed=rng)
        elif qmc_engine == 'halton':
            sampler = qmc.Halton(d=self.parameter_count, seed=rng)
        else:
            raise ValueError(self.__init_error_msg)

        # Generate QMC samples based on the selected method
        self.population = sampler.random(n=self.num_population_members)

        # reset population energies
        self.population_energies = np.full(self.num_population_members,
                                           np.inf)

        # reset number of function evaluations counter
        self._nfev = 0
    def init_population_random(self):
        """
        Initializes the population at random. This type of initialization
        can possess clustering, Latin Hypercube sampling is generally better.
        """
        # 使用随机数生成器初始化种群
        rng = self.random_number_generator
        self.population = rng.uniform(size=self.population_shape)

        # 重置种群能量
        self.population_energies = np.full(self.num_population_members,
                                           np.inf)

        # 重置函数评估次数计数器
        self._nfev = 0

    def init_population_array(self, init):
        """
        Initializes the population with a user specified population.

        Parameters
        ----------
        init : np.ndarray
            Array specifying subset of the initial population. The array should
            have shape (S, N), where N is the number of parameters.
            The population is clipped to the lower and upper bounds.
        """
        # 确保使用浮点数数组
        popn = np.asarray(init, dtype=np.float64)

        if (np.size(popn, 0) < 5 or
                popn.shape[1] != self.parameter_count or
                len(popn.shape) != 2):
            # 如果提供的种群形状不符合要求，则引发值错误
            raise ValueError("The population supplied needs to have shape"
                             " (S, len(x)), where S > 4.")

        # 缩放值并裁剪到边界，并分配给种群
        self.population = np.clip(self._unscale_parameters(popn), 0, 1)

        # 计算种群成员数量
        self.num_population_members = np.size(self.population, 0)

        # 更新种群形状
        self.population_shape = (self.num_population_members,
                                 self.parameter_count)

        # 重置种群能量
        self.population_energies = np.full(self.num_population_members,
                                           np.inf)

        # 重置函数评估次数计数器
        self._nfev = 0

    @property
    def x(self):
        """
        The best solution from the solver
        """
        # 返回求解器得出的最佳解决方案
        return self._scale_parameters(self.population[0])

    @property
    def convergence(self):
        """
        The standard deviation of the population energies divided by their
        mean.
        """
        if np.any(np.isinf(self.population_energies)):
            return np.inf
        # 返回种群能量的标准差除以它们的均值
        return (np.std(self.population_energies) /
                (np.abs(np.mean(self.population_energies)) + _MACHEPS))

    def converged(self):
        """
        Return True if the solver has converged.
        """
        if np.any(np.isinf(self.population_energies)):
            return False

        # 如果种群能量的标准差小于等于收敛容差，则返回True
        return (np.std(self.population_energies) <=
                self.atol +
                self.tol * np.abs(np.mean(self.population_energies)))
    # 定义一个方法 `_result`，用于生成一个中间的 OptimizeResult 对象
    def _result(self, **kwds):
        # 从关键字参数 `kwds` 中获取优化迭代次数 `nit`，默认为 None
        nit = kwds.get('nit', None)
        # 从关键字参数 `kwds` 中获取优化信息 `message`，默认为 None
        message = kwds.get('message', None)
        # 从关键字参数 `kwds` 中获取警告标志 `warning_flag`，默认为 False
        warning_flag = kwds.get('warning_flag', False)
        # 创建 OptimizeResult 对象 `result`，包括以下属性：
        # - x: 最优解向量
        # - fun: 最优解对应的目标函数值
        # - nfev: 函数评估次数
        # - nit: 优化迭代次数
        # - message: 优化消息
        # - success: 优化是否成功的标志，根据 `warning_flag` 判断
        # - population: 经过缩放的种群参数
        # - population_energies: 种群中各个个体的能量值
        result = OptimizeResult(
            x=self.x,
            fun=self.population_energies[0],
            nfev=self._nfev,
            nit=nit,
            message=message,
            success=(warning_flag is not True),
            population=self._scale_parameters(self.population),
            population_energies=self.population_energies
        )
        
        # 如果存在包装的约束条件 `_wrapped_constraints`
        if self._wrapped_constraints:
            # 对每个约束条件 `c`，计算当前最优解 `result.x` 的违反程度
            result.constr = [c.violation(result.x)
                             for c in self._wrapped_constraints]
            # 计算所有约束条件的最大违反程度
            result.constr_violation = np.max(np.concatenate(result.constr))
            # 将最大违反程度赋值给 `result.maxcv`
            result.maxcv = result.constr_violation
            # 如果最大违反程度大于 0，则将优化结果的 `success` 标志设为 False
            if result.maxcv > 0:
                result.success = False

        # 返回生成的 OptimizeResult 对象 `result`
        return result
    def _promote_lowest_energy(self):
        # swaps 'best solution' into first population entry
        # 获取当前种群成员的索引
        idx = np.arange(self.num_population_members)
        # 获取可行解的索引
        feasible_solutions = idx[self.feasible]
        if feasible_solutions.size:
            # 找到最佳的可行解
            idx_t = np.argmin(self.population_energies[feasible_solutions])
            l = feasible_solutions[idx_t]
        else:
            # 如果没有可行解，选择违反约束最少的“最佳”非可行解
            l = np.argmin(np.sum(self.constraint_violation, axis=1))

        # 将最佳解与第一个种群成员进行交换
        self.population_energies[[0, l]] = self.population_energies[[l, 0]]
        self.population[[0, l], :] = self.population[[l, 0], :]
        self.feasible[[0, l]] = self.feasible[[l, 0]]
        self.constraint_violation[[0, l], :] = (
            self.constraint_violation[[l, 0], :]
        )
    # 计算给定解集合的所有约束条件的总违规量

    def _constraint_violation_fn(self, x):
        """
        计算所有约束条件的总违规量，针对一组解决方案。

        Parameters
        ----------
        x : ndarray
            解向量。形状为 (S, N) 或 (N,)，其中 S 是要研究的解的数量，N 是参数的数量。

        Returns
        -------
        cv : ndarray
            约束条件的总违规量。形状为 ``(S, M)``，其中 M 是约束组件的总数
            （不一定等于 len(self._wrapped_constraints)）。
        """
        # 计算解向量 x 中有多少个解
        S = np.size(x) // self.parameter_count
        # 初始化一个大小为 (S, self.total_constraints) 的全零数组
        _out = np.zeros((S, self.total_constraints))
        offset = 0
        for con in self._wrapped_constraints:
            # 将输入向量转置以传递给约束条件函数
            # 约束条件函数的输入/输出是 {(N, S), (N,)} --> (M, S)
            # _constraint_violation_fn 的输入是 (S, N) 或 (N,)，所以要进行转置
            # 输出从 (M, S) 转置为 (S, M) 以供进一步使用
            c = con.violation(x.T).T

            # 检查 c 的形状是否正确
            if c.shape[-1] != con.num_constr or (S > 1 and c.shape[0] != S):
                raise RuntimeError("Constraint 返回的数组形状不正确。如果 `vectorized` 为 False，"
                                   "则 Constraint 应返回形状为 (M,) 的数组。如果 `vectorized` 为 True，"
                                   "则 Constraint 必须返回形状为 (M, S) 的数组，其中 S 是解向量的数量，"
                                   "M 是给定 Constraint 对象中约束组件的数量。")

            # 将 c 重新形状为 (S, con.num_constr)，以便放入 _out 中
            c = np.reshape(c, (S, con.num_constr))
            # 将 c 放入 _out 的适当位置
            _out[:, offset:offset + con.num_constr] = c
            offset += con.num_constr

        return _out
    # 实现对种群的可行性计算，返回每个个体的可行性和约束违反情况
    def _calculate_population_feasibilities(self, population):
        """
        Calculate the feasibilities of a population.

        Parameters
        ----------
        population : ndarray
            An array of parameter vectors normalised to [0, 1] using lower
            and upper limits. Has shape ``(np.size(population, 0), N)``.

        Returns
        -------
        feasible, constraint_violation : ndarray, ndarray
            Boolean array of feasibility for each population member, and an
            array of the constraint violation for each population member.
            constraint_violation has shape ``(np.size(population, 0), M)``,
            where M is the number of constraints.
        """
        # 获取种群个体数量
        num_members = np.size(population, 0)
        
        # 如果没有约束条件，则直接返回所有个体均为可行的情况
        if not self._wrapped_constraints:
            # 无约束条件的快捷方式，所有个体都设为可行，且约束违反为零
            return np.ones(num_members, bool), np.zeros((num_members, 1))

        # 将参数向量进行缩放处理，使其在 [0, 1] 范围内
        # (S, N)
        parameters_pop = self._scale_parameters(population)

        # 如果采用向量化计算
        if self.vectorized:
            # 计算每个个体的约束违反情况
            # (S, M)
            constraint_violation = np.array(
                self._constraint_violation_fn(parameters_pop)
            )
        else:
            # 逐个计算每个个体的约束违反情况
            # (S, 1, M)
            constraint_violation = np.array([self._constraint_violation_fn(x)
                                             for x in parameters_pop])
            # 如果使用上述列表推导式，会生成形状为 (S, 1, M) 的数组，
            # 因为每次迭代生成的是形状为 (1, M) 的数组。与向量化版本返回的 (S, M) 形状有所不同，
            # 因此需要去除第一个轴以匹配预期的形状。
            constraint_violation = constraint_violation[:, 0]

        # 计算每个个体的可行性，如果约束违反总和大于0，则设为不可行
        feasible = ~(np.sum(constraint_violation, axis=1) > 0)

        return feasible, constraint_violation

    # 迭代器方法，使对象可以被迭代
    def __iter__(self):
        return self

    # 进入上下文管理器时调用的方法，返回对象本身
    def __enter__(self):
        return self

    # 退出上下文管理器时调用的方法，将退出调用传递给内部的 _mapwrapper 对象
    def __exit__(self, *args):
        return self._mapwrapper.__exit__(*args)
    def _accept_trial(self, energy_trial, feasible_trial, cv_trial,
                      energy_orig, feasible_orig, cv_orig):
        """
        Trial is accepted if:
        * it satisfies all constraints and provides a lower or equal objective
          function value, while both the compared solutions are feasible
        - or -
        * it is feasible while the original solution is infeasible,
        - or -
        * it is infeasible, but provides a lower or equal constraint violation
          for all constraint functions.

        This test corresponds to section III of Lampinen [1]_.

        Parameters
        ----------
        energy_trial : float
            Energy of the trial solution
        feasible_trial : float
            Feasibility of trial solution
        cv_trial : array-like
            Excess constraint violation for the trial solution
        energy_orig : float
            Energy of the original solution
        feasible_orig : float
            Feasibility of original solution
        cv_orig : array-like
            Excess constraint violation for the original solution

        Returns
        -------
        accepted : bool

        """
        # 根据 Lampinen [1] 第 III 节的规则进行试验解的接受判断
        if feasible_orig and feasible_trial:
            return energy_trial <= energy_orig
        elif feasible_trial and not feasible_orig:
            return True
        elif not feasible_trial and (cv_trial <= cv_orig).all():
            # 如果试验解不可行，但在所有约束函数上提供了更低或相等的违约
            # cv_trial < cv_orig 意味着试验解和原解都不可行
            return True

        return False

    def _scale_parameters(self, trial):
        """将0到1之间的数字缩放到参数范围内。"""
        # trial 可以是形状为 (N, ) 或 (L, N) 的数组，其中 L 是被缩放的解的数量
        scaled = self.__scale_arg1 + (trial - 0.5) * self.__scale_arg2
        if np.count_nonzero(self.integrality):
            i = np.broadcast_to(self.integrality, scaled.shape)
            scaled[i] = np.round(scaled[i])
        return scaled

    def _unscale_parameters(self, parameters):
        """将参数反向缩放到0到1之间的数字。"""
        return (parameters - self.__scale_arg1) * self.__recip_scale_arg2 + 0.5

    def _ensure_constraint(self, trial):
        """确保参数在限制范围内。"""
        mask = np.bitwise_or(trial > 1, trial < 0)
        if oob := np.count_nonzero(mask):
            trial[mask] = self.random_number_generator.uniform(size=oob)
    # 定义一个方法用于自定义变异操作，接受一个候选个体作为参数
    def _mutate_custom(self, candidate):
        # 获取随机数生成器
        rng = self.random_number_generator
        # 错误信息，指出策略必须有特定的签名和返回形状
        msg = (
            "strategy must have signature"
            " f(candidate: int, population: np.ndarray, rng=None) returning an"
            " array of shape (N,)"
        )
        # 缩放参数集合中的参数
        _population = self._scale_parameters(self.population)
        # 如果候选个体不是数组（单个条目）
        if not len(np.shape(candidate)):
            # 使用指定的策略生成一个试验向量
            trial = self.strategy(candidate, _population, rng=rng)
            # 检查试验向量的形状是否符合预期
            if trial.shape != (self.parameter_count,):
                raise RuntimeError(msg)
        else:
            # 获取候选个体数组的长度
            S = candidate.shape[0]
            # 为每个候选个体生成一个试验向量
            trial = np.array(
                [self.strategy(c, _population, rng=rng) for c in candidate],
                dtype=float
            )
            # 检查所有试验向量的形状是否符合预期
            if trial.shape != (S, self.parameter_count):
                raise RuntimeError(msg)
        # 对生成的试验向量进行反缩放
        return self._unscale_parameters(trial)

    # 定义一个方法用于生成多个候选个体的试验向量
    def _mutate_many(self, candidates):
        """Create trial vectors based on a mutation strategy."""
        # 获取随机数生成器
        rng = self.random_number_generator

        # 候选个体的数量
        S = len(candidates)
        # 如果策略是可调用的函数，则使用自定义变异方法
        if callable(self.strategy):
            return self._mutate_custom(candidates)

        # 复制当前种群中的候选个体作为试验向量
        trial = np.copy(self.population[candidates])
        # 为每个候选个体选择样本
        samples = np.array([self._select_samples(c, 5) for c in candidates])

        # 根据策略选择不同的变异函数
        if self.strategy in ['currenttobest1exp', 'currenttobest1bin']:
            bprime = self.mutation_func(candidates, samples)
        else:
            bprime = self.mutation_func(samples)

        # 随机生成填充点的索引
        fill_point = rng_integers(rng, self.parameter_count, size=S)
        # 生成随机交叉点
        crossovers = rng.uniform(size=(S, self.parameter_count))
        crossovers = crossovers < self.cross_over_probability

        # 如果策略属于二项式变异类型
        if self.strategy in self._binomial:
            # 对于二项式变异，最后一个始终来自bprime向量
            i = np.arange(S)
            crossovers[i, fill_point[i]] = True
            trial = np.where(crossovers, bprime, trial)
            return trial

        # 如果策略属于指数变异类型
        elif self.strategy in self._exponential:
            # 将第一个交叉点设置为True
            crossovers[..., 0] = True
            # 对于每个候选个体执行指数变异操作
            for j in range(S):
                i = 0
                init_fill = fill_point[j]
                while (i < self.parameter_count and crossovers[j, i]):
                    trial[j, init_fill] = bprime[j, init_fill]
                    init_fill = (init_fill + 1) % self.parameter_count
                    i += 1

            return trial
    def _mutate(self, candidate):
        """Create a trial vector based on a mutation strategy."""
        # 获取随机数生成器
        rng = self.random_number_generator

        # 如果策略是可调用的函数，则使用自定义的变异方法
        if callable(self.strategy):
            return self._mutate_custom(candidate)

        # 随机选择填充点
        fill_point = rng_integers(rng, self.parameter_count)
        # 选择用于变异的样本
        samples = self._select_samples(candidate, 5)

        # 复制当前候选解作为试验向量
        trial = np.copy(self.population[candidate])

        # 根据策略选择变异函数
        if self.strategy in ['currenttobest1exp', 'currenttobest1bin']:
            bprime = self.mutation_func(candidate, samples)
        else:
            bprime = self.mutation_func(samples)

        # 生成参数数量的交叉概率向量
        crossovers = rng.uniform(size=self.parameter_count)
        crossovers = crossovers < self.cross_over_probability

        # 根据策略选择变异方式
        if self.strategy in self._binomial:
            # 对于二项式策略，最后一个总是来自bprime向量
            # 如果使用循环填充点，必须将最后一个设为True
            crossovers[fill_point] = True
            # 根据交叉概率进行交叉操作
            trial = np.where(crossovers, bprime, trial)
            return trial

        elif self.strategy in self._exponential:
            i = 0
            crossovers[0] = True
            # 当crossovers[i]为True时，进行指数策略的交叉
            while i < self.parameter_count and crossovers[i]:
                trial[fill_point] = bprime[fill_point]
                fill_point = (fill_point + 1) % self.parameter_count
                i += 1

            return trial

    def _best1(self, samples):
        """best1bin, best1exp"""
        # samples.shape == (S, 5)
        # 或者
        # samples.shape == (5,)
        # 从样本中提取r0和r1
        r0, r1 = samples[..., :2].T
        # 返回基于最佳策略的变异向量
        return (self.population[0] + self.scale *
                (self.population[r0] - self.population[r1]))

    def _rand1(self, samples):
        """rand1bin, rand1exp"""
        # 从样本中提取r0, r1, r2
        r0, r1, r2 = samples[..., :3].T
        # 返回基于随机策略的变异向量
        return (self.population[r0] + self.scale *
                (self.population[r1] - self.population[r2]))

    def _randtobest1(self, samples):
        """randtobest1bin, randtobest1exp"""
        # 从样本中提取r0, r1, r2
        r0, r1, r2 = samples[..., :3].T
        # 复制r0的解向量作为bprime
        bprime = np.copy(self.population[r0])
        # 根据随机到最佳策略生成变异向量
        bprime += self.scale * (self.population[0] - bprime)
        bprime += self.scale * (self.population[r1] - self.population[r2])
        return bprime

    def _currenttobest1(self, candidate, samples):
        """currenttobest1bin, currenttobest1exp"""
        # 从样本中提取r0, r1
        r0, r1 = samples[..., :2].T
        # 根据当前到最佳策略生成变异向量
        bprime = (self.population[candidate] + self.scale *
                  (self.population[0] - self.population[candidate] +
                   self.population[r0] - self.population[r1]))
        return bprime
    # 定义函数_best2，用于执行best2bin和best2exp策略
    def _best2(self, samples):
        """best2bin, best2exp"""
        # 从samples中提取前四个元素，并转置得到r0, r1, r2, r3
        r0, r1, r2, r3 = samples[..., :4].T
        # 计算bprime值，应用best2策略公式
        bprime = (self.population[0] + self.scale *
                  (self.population[r0] + self.population[r1] -
                   self.population[r2] - self.population[r3]))

        return bprime

    # 定义函数_rand2，用于执行rand2bin和rand2exp策略
    def _rand2(self, samples):
        """rand2bin, rand2exp"""
        # 从samples中提取前五个元素，并转置得到r0, r1, r2, r3, r4
        r0, r1, r2, r3, r4 = samples[..., :5].T
        # 计算bprime值，应用rand2策略公式
        bprime = (self.population[r0] + self.scale *
                  (self.population[r1] + self.population[r2] -
                   self.population[r3] - self.population[r4]))

        return bprime

    # 定义函数_select_samples，用于选择随机样本索引
    def _select_samples(self, candidate, number_samples):
        """
        obtain random integers from range(self.num_population_members),
        without replacement. You can't have the original candidate either.
        """
        # 打乱随机种群索引
        self.random_number_generator.shuffle(self._random_population_index)
        # 从打乱后的索引中选取指定数量的样本索引，同时确保不包括原始候选者
        idxs = self._random_population_index[:number_samples + 1]
        return idxs[idxs != candidate][:number_samples]
class _ConstraintWrapper:
    """Object to wrap/evaluate user defined constraints.

    Very similar in practice to `PreparedConstraint`, except that no evaluation
    of jac/hess is performed (explicit or implicit).

    If created successfully, it will contain the attributes listed below.

    Parameters
    ----------
    constraint : {`NonlinearConstraint`, `LinearConstraint`, `Bounds`}
        Constraint to check and prepare.
    x0 : array_like
        Initial vector of independent variables, shape (N,)

    Attributes
    ----------
    fun : callable
        Function defining the constraint wrapped by one of the convenience
        classes.
    bounds : 2-tuple
        Contains lower and upper bounds for the constraints --- lb and ub.
        These are converted to ndarray and have a size equal to the number of
        the constraints.

    Notes
    -----
    _ConstraintWrapper.fun and _ConstraintWrapper.violation can get sent
    arrays of shape (N, S) or (N,), where S is the number of vectors of shape
    (N,) to consider constraints for.
    """

    def __init__(self, constraint, x0):
        self.constraint = constraint

        # Define the function `fun` based on the type of constraint
        if isinstance(constraint, NonlinearConstraint):
            def fun(x):
                x = np.asarray(x)
                return np.atleast_1d(constraint.fun(x))
        elif isinstance(constraint, LinearConstraint):
            def fun(x):
                if issparse(constraint.A):
                    A = constraint.A
                else:
                    A = np.atleast_2d(constraint.A)

                res = A.dot(x)
                # Handle different shapes of `x` and `res`
                if x.ndim == 1 and res.ndim == 2:
                    res = np.asarray(res)[:, 0]
                return res
        elif isinstance(constraint, Bounds):
            def fun(x):
                return np.asarray(x)
        else:
            raise ValueError("`constraint` of an unknown type is passed.")

        self.fun = fun

        # Convert lower and upper bounds to ndarray
        lb = np.asarray(constraint.lb, dtype=float)
        ub = np.asarray(constraint.ub, dtype=float)

        x0 = np.asarray(x0)

        # Determine the number of constraints and parameter count
        f0 = fun(x0)
        self.num_constr = m = f0.size
        self.parameter_count = x0.size

        # Ensure bounds have the correct dimensions
        if lb.ndim == 0:
            lb = np.resize(lb, m)
        if ub.ndim == 0:
            ub = np.resize(ub, m)

        self.bounds = (lb, ub)

    def __call__(self, x):
        return np.atleast_1d(self.fun(x))
    # 返回超出约束的量，计算约束超出的情况。
    def violation(self, x):
        """How much the constraint is exceeded by.

        Parameters
        ----------
        x : array-like
            Vector of independent variables, (N, S), where N is number of
            parameters and S is the number of solutions to be investigated.

        Returns
        -------
        excess : array-like
            How much the constraint is exceeded by, for each of the
            constraints specified by `_ConstraintWrapper.fun`.
            Has shape (M, S) where M is the number of constraint components.
        """
        # 计算约束函数在给定参数向量 x 下的值，期望 ev 的形状为 (num_constr, S) 或 (num_constr,)
        ev = self.fun(np.asarray(x))

        try:
            # 计算下界超出量，确保不低于零
            excess_lb = np.maximum(self.bounds[0] - ev.T, 0)
            # 计算上界超出量，确保不低于零
            excess_ub = np.maximum(ev.T - self.bounds[1], 0)
        except ValueError as e:
            # 捕获值错误异常，通常由约束函数返回的数组形状不正确引起
            raise RuntimeError("An array returned from a Constraint has"
                               " the wrong shape. If `vectorized is False`"
                               " the Constraint should return an array of"
                               " shape (M,). If `vectorized is True` then"
                               " the Constraint must return an array of"
                               " shape (M, S), where S is the number of"
                               " solution vectors and M is the number of"
                               " constraint components in a given"
                               " Constraint object.") from e

        # 汇总下界和上界超出量，并转置结果以匹配期望的形状 (M, S)
        v = (excess_lb + excess_ub).T
        return v
```