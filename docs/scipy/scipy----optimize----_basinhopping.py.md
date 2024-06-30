# `D:\src\scipysrc\scipy\scipy\optimize\_basinhopping.py`

```
"""
basinhopping: The basinhopping global optimization algorithm
"""
# 导入必要的库
import numpy as np  # 导入 NumPy 库，用于数值计算
import math  # 导入 math 库，提供基本的数学函数
import inspect  # 导入 inspect 库，用于检查对象的内部信息
import scipy.optimize  # 导入 SciPy 库的优化模块
from scipy._lib._util import check_random_state  # 导入 SciPy 内部的随机状态检查函数

# 定义公开的接口
__all__ = ['basinhopping']

# 定义参数元组
_params = (inspect.Parameter('res_new', kind=inspect.Parameter.KEYWORD_ONLY),
           inspect.Parameter('res_old', kind=inspect.Parameter.KEYWORD_ONLY))
# 创建新的接受测试签名
_new_accept_test_signature = inspect.Signature(parameters=_params)


class Storage:
    """
    Class used to store the lowest energy structure
    """
    def __init__(self, minres):
        self._add(minres)

    def _add(self, minres):
        # 初始化存储最低能量结构的对象
        self.minres = minres
        self.minres.x = np.copy(minres.x)

    def update(self, minres):
        # 更新存储的最低能量结构对象
        if minres.success and (minres.fun < self.minres.fun
                               or not self.minres.success):
            self._add(minres)
            return True
        else:
            return False

    def get_lowest(self):
        # 获取当前存储的最低能量结构对象
        return self.minres


class BasinHoppingRunner:
    """This class implements the core of the basinhopping algorithm.

    x0 : ndarray
        The starting coordinates.
    minimizer : callable
        The local minimizer, with signature ``result = minimizer(x)``.
        The return value is an `optimize.OptimizeResult` object.
    step_taking : callable
        This function displaces the coordinates randomly. Signature should
        be ``x_new = step_taking(x)``. Note that `x` may be modified in-place.
    accept_tests : list of callables
        Each test is passed the kwargs `f_new`, `x_new`, `f_old` and
        `x_old`. These tests will be used to judge whether or not to accept
        the step. The acceptable return values are True, False, or ``"force
        accept"``. If any of the tests return False then the step is rejected.
        If ``"force accept"``, then this will override any other tests in
        order to accept the step. This can be used, for example, to forcefully
        escape from a local minimum that ``basinhopping`` is trapped in.
    disp : bool, optional
        Display status messages.

    """
    # BasinHoppingRunner 类，实现了 basinhopping 算法的核心功能
    # 初始化函数，接受初始点 x0、最小化器 minimizer、步骤采取器 step_taking、接受测试 accept_tests 和显示标志 disp
    def __init__(self, x0, minimizer, step_taking, accept_tests, disp=False):
        # 复制初始点 x0 到实例变量 self.x
        self.x = np.copy(x0)
        # 将最小化器对象赋给实例变量 self.minimizer
        self.minimizer = minimizer
        # 将步骤采取器对象赋给实例变量 self.step_taking
        self.step_taking = step_taking
        # 将接受测试对象赋给实例变量 self.accept_tests
        self.accept_tests = accept_tests
        # 将显示标志赋给实例变量 self.disp
        self.disp = disp

        # 初始化步数计数器
        self.nstep = 0

        # 初始化返回结果对象
        self.res = scipy.optimize.OptimizeResult()
        # 初始化最小化失败次数为 0
        self.res.minimization_failures = 0

        # 进行初始最小化操作
        minres = minimizer(self.x)
        # 如果最小化失败，则增加最小化失败次数，并在显示标志开启时打印警告信息
        if not minres.success:
            self.res.minimization_failures += 1
            if self.disp:
                print("warning: basinhopping: local minimization failure")
        # 复制最小化结果中的最优解到 self.x 和最优解的能量到 self.energy
        self.x = np.copy(minres.x)
        self.energy = minres.fun
        # 将当前最优最小化结果赋给 incumbent_minres，表示迄今为止找到的最佳最小化结果
        self.incumbent_minres = minres
        # 在显示标志开启时打印当前步骤号和能量值
        if self.disp:
            print("basinhopping step %d: f %g" % (self.nstep, self.energy))

        # 初始化存储类对象，并以 minres 作为初始参数
        self.storage = Storage(minres)

        # 如果最小化结果对象 minres 具有属性 "nfev"，则将其赋给返回结果对象 self.res 的属性 nfev
        if hasattr(minres, "nfev"):
            self.res.nfev = minres.nfev
        # 如果最小化结果对象 minres 具有属性 "njev"，则将其赋给返回结果对象 self.res 的属性 njev
        if hasattr(minres, "njev"):
            self.res.njev = minres.njev
        # 如果最小化结果对象 minres 具有属性 "nhev"，则将其赋给返回结果对象 self.res 的属性 nhev
        if hasattr(minres, "nhev"):
            self.res.nhev = minres.nhev
    def _monte_carlo_step(self):
        """执行一次蒙特卡罗迭代

        随机扰动坐标，进行最小化，并决定是否接受新的坐标。
        """
        # 复制self.x以便在step_taking算法中进行原地修改
        x_after_step = np.copy(self.x)
        x_after_step = self.step_taking(x_after_step)

        # 进行局部最小化
        minres = self.minimizer(x_after_step)
        x_after_quench = minres.x
        energy_after_quench = minres.fun
        if not minres.success:
            self.res.minimization_failures += 1
            if self.disp:
                print("warning: basinhopping: local minimization failure")
        if hasattr(minres, "nfev"):
            self.res.nfev += minres.nfev
        if hasattr(minres, "njev"):
            self.res.njev += minres.njev
        if hasattr(minres, "nhev"):
            self.res.nhev += minres.nhev

        # 基于self.accept_tests决定是否接受此步骤。如果任何测试为False，则拒绝此步骤。
        # 如果任何测试返回特殊字符串'force accept'，则无条件接受此步骤。这可用于强制
        # 从局部最小值中逃脱，如果普通的盆地跳跃步骤不足够。
        accept = True
        for test in self.accept_tests:
            if inspect.signature(test) == _new_accept_test_signature:
                testres = test(res_new=minres, res_old=self.incumbent_minres)
            else:
                testres = test(f_new=energy_after_quench, x_new=x_after_quench,
                               f_old=self.energy, x_old=self.x)

            if testres == 'force accept':
                accept = True
                break
            elif testres is None:
                raise ValueError("accept_tests must return True, False, or "
                                 "'force accept'")
            elif not testres:
                accept = False

        # 将接受测试的结果报告给step_taking类。这是为了自适应步骤采取。
        if hasattr(self.step_taking, "report"):
            self.step_taking.report(accept, f_new=energy_after_quench,
                                    x_new=x_after_quench, f_old=self.energy,
                                    x_old=self.x)

        return accept, minres
    def one_cycle(self):
        """执行一次基于随机优化的算法循环
        
        每次调用该方法，算法步数加一。
        """
        self.nstep += 1  # 增加算法步数计数器

        new_global_min = False  # 初始化标志位，表示是否找到新的全局最小值

        # 执行一次蒙特卡洛步骤，返回接受标志和最小化结果
        accept, minres = self._monte_carlo_step()

        if accept:
            # 如果接受最小化结果，则更新能量和位置信息
            self.energy = minres.fun
            self.x = np.copy(minres.x)
            self.incumbent_minres = minres  # 记录当前找到的最佳最小化结果（即潜在的全局最小值）
            # 更新存储，检查是否发现新的全局最小值
            new_global_min = self.storage.update(minres)

        # 打印一些信息
        if self.disp:
            self.print_report(minres.fun, accept)
            if new_global_min:
                # 如果发现了新的全局最小值，则输出提示信息
                print("found new global minimum on step %d with function"
                      " value %g" % (self.nstep, self.energy))

        # 将一些变量保存为BasinHoppingRunner对象的属性
        self.xtrial = minres.x  # 保存试验点的位置
        self.energy_trial = minres.fun  # 保存试验点的能量
        self.accept = accept  # 保存接受标志

        return new_global_min  # 返回是否发现新的全局最小值的标志

    def print_report(self, energy_trial, accept):
        """打印状态更新报告
        
        打印当前基于随机优化的算法步骤的状态信息，包括能量和接受标志等。
        """
        minres = self.storage.get_lowest()  # 获取存储中的最低值结果
        print("basinhopping step %d: f %g trial_f %g accepted %d "
              " lowest_f %g" % (self.nstep, self.energy, energy_trial,
                                accept, minres.fun))
class AdaptiveStepsize:
    """
    Class to implement adaptive stepsize.

    This class wraps the step taking class and modifies the stepsize to
    ensure the true acceptance rate is as close as possible to the target.

    Parameters
    ----------
    takestep : callable
        The step taking routine.  Must contain modifiable attribute
        takestep.stepsize
    accept_rate : float, optional
        The target step acceptance rate
    interval : int, optional
        Interval for how often to update the stepsize
    factor : float, optional
        The step size is multiplied or divided by this factor upon each
        update.
    verbose : bool, optional
        Print information about each update

    """
    def __init__(self, takestep, accept_rate=0.5, interval=50, factor=0.9,
                 verbose=True):
        # Initialize AdaptiveStepsize object with given parameters
        self.takestep = takestep  # Set the step taking routine
        self.target_accept_rate = accept_rate  # Set target step acceptance rate
        self.interval = interval  # Set interval for stepsize update
        self.factor = factor  # Set factor by which step size is adjusted
        self.verbose = verbose  # Set verbosity for printing updates

        # Initialize step counting variables
        self.nstep = 0  # Number of steps taken
        self.nstep_tot = 0  # Total number of steps
        self.naccept = 0  # Number of accepted steps

    def __call__(self, x):
        # Call method to take a step with current state
        return self.take_step(x)

    def _adjust_step_size(self):
        # Adjust the step size based on current acceptance rate
        old_stepsize = self.takestep.stepsize  # Get current step size
        accept_rate = float(self.naccept) / self.nstep  # Calculate acceptance rate

        if accept_rate > self.target_accept_rate:
            # Reduce step size if acceptance rate is too high
            self.takestep.stepsize /= self.factor
        else:
            # Increase step size if acceptance rate is too low
            self.takestep.stepsize *= self.factor

        if self.verbose:
            # Print adjustment information if verbose mode is enabled
            print(f"adaptive stepsize: acceptance rate {accept_rate:f} target "
                  f"{self.target_accept_rate:f} new stepsize "
                  f"{self.takestep.stepsize:g} old stepsize {old_stepsize:g}")

    def take_step(self, x):
        # Take a step using the current step taking routine
        self.nstep += 1  # Increment step count
        self.nstep_tot += 1  # Increment total step count

        if self.nstep % self.interval == 0:
            # Perform step size adjustment at regular intervals
            self._adjust_step_size()

        return self.takestep(x)  # Return the result of the step taking routine

    def report(self, accept, **kwargs):
        # Method to report the result of the step
        if accept:
            self.naccept += 1  # Increment acceptance count


class RandomDisplacement:
    """Add a random displacement of maximum size `stepsize` to each coordinate.

    Calling this updates `x` in-place.

    Parameters
    ----------
    stepsize : float, optional
        Maximum stepsize in any dimension
    random_gen : {None, int, `numpy.random.Generator`,
                  `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    """
    # 初始化函数，用于实例化对象时设置参数
    def __init__(self, stepsize=0.5, random_gen=None):
        # 设置对象的步长属性，默认为0.5
        self.stepsize = stepsize
        # 使用给定的随机数生成器或默认的随机数生成器来初始化对象的随机数生成器属性
        self.random_gen = check_random_state(random_gen)
    
    # 调用函数，用于实现对象实例的可调用行为
    def __call__(self, x):
        # 对输入的数组 x 中的每个元素加上一个从均匀分布中随机抽取的值，这个随机值的范围为 [-self.stepsize, self.stepsize]
        x += self.random_gen.uniform(-self.stepsize, self.stepsize, np.shape(x))
        # 返回处理后的数组 x
        return x
class MinimizerWrapper:
    """
    Wrap a minimizer function as a minimizer class.
    """
    def __init__(self, minimizer, func=None, **kwargs):
        # Initialize the MinimizerWrapper with a specified minimizer function,
        # an optional objective function (func), and additional keyword arguments.
        self.minimizer = minimizer
        self.func = func
        self.kwargs = kwargs

    def __call__(self, x0):
        # If func is None, call the minimizer directly with x0 and kwargs.
        # Otherwise, call the minimizer with func, x0, and kwargs.
        if self.func is None:
            return self.minimizer(x0, **self.kwargs)
        else:
            return self.minimizer(self.func, x0, **self.kwargs)


class Metropolis:
    """
    Metropolis acceptance criterion for Markov Chain Monte Carlo (MCMC) simulations.

    Parameters
    ----------
    T : float
        The "temperature" parameter for the accept or reject criterion.
    random_gen : {None, int, `numpy.random.Generator`,
                  `numpy.random.RandomState`}, optional
        Random number generator used for acceptance test.

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
    """

    def __init__(self, T, random_gen=None):
        # Set the inverse temperature beta. Handle the special case where T = 0 to avoid division by zero.
        self.beta = 1.0 / T if T != 0 else float('inf')
        # Ensure random_gen is a valid random state generator.
        self.random_gen = check_random_state(random_gen)

    def accept_reject(self, res_new, res_old):
        """
        Acceptance-rejection criteria based on energy difference.

        Parameters
        ----------
        res_new : OptimizeResult
            Result of the current state in the optimization.
        res_old : OptimizeResult
            Result of the previous state in the optimization.

        Returns
        -------
        bool
            True if the new state is accepted, False otherwise.
        """
        with np.errstate(invalid='ignore'):
            # Compute the product of energy difference and inverse temperature beta,
            # handle potential invalid value warnings due to zero difference.
            prod = -(res_new.fun - res_old.fun) * self.beta
            # Calculate the acceptance probability.
            w = math.exp(min(0, prod))

        # Generate a uniform random number and apply acceptance conditions.
        rand = self.random_gen.uniform()
        return w >= rand and (res_new.success or not res_old.success)

    def __call__(self, *, res_new, res_old):
        """
        Perform Metropolis acceptance-rejection test on optimization results.

        Parameters
        ----------
        res_new : OptimizeResult
            Result of the current state in the optimization.
        res_old : OptimizeResult
            Result of the previous state in the optimization.

        Returns
        -------
        bool
            True if the new state is accepted, False otherwise.
        """
        return bool(self.accept_reject(res_new, res_old))


def basinhopping(func, x0, niter=100, T=1.0, stepsize=0.5,
                 minimizer_kwargs=None, take_step=None, accept_test=None,
                 callback=None, interval=50, disp=False, niter_success=None,
                 seed=None, *, target_accept_rate=0.5, stepwise_factor=0.9):
    """
    Perform basinhopping optimization using Metropolis-Hastings algorithm.

    Parameters
    ----------
    func : callable
        Objective function to minimize.
    x0 : array_like
        Initial guess.
    niter : int, optional
        Number of iterations.
    T : float, optional
        Temperature parameter for the Metropolis criterion.
    stepsize : float, optional
        Step size for the random step generation.
    minimizer_kwargs : dict, optional
        Extra keyword arguments to pass to the minimizer.
    take_step : callable, optional
        Custom step-taking function.
    accept_test : callable, optional
        Custom acceptance test function.
    callback : callable, optional
        Function to call after each iteration.
    interval : int, optional
        Number of iterations between printing progress messages.
    disp : bool, optional
        Whether to print progress messages.
    niter_success : int, optional
        Number of successful iterations before stopping.
    seed : int or `numpy.random.RandomState`, optional
        Seed for random number generation.
    target_accept_rate : float, optional
        Target acceptance rate for Metropolis criterion.
    stepwise_factor : float, optional
        Step size adjustment factor for Metropolis criterion.
    """
    """Find the global minimum of a function using the basin-hopping algorithm.

    Basin-hopping is a two-phase method that combines a global stepping
    algorithm with local minimization at each step. Designed to mimic
    the natural process of energy minimization of clusters of atoms, it works
    well for similar problems with "funnel-like, but rugged" energy landscapes
    [5]_.

    As the step-taking, step acceptance, and minimization methods are all
    customizable, this function can also be used to implement other two-phase
    methods.

    Parameters
    ----------
    func : callable ``f(x, *args)``
        Function to be optimized.  ``args`` can be passed as an optional item
        in the dict `minimizer_kwargs`
    x0 : array_like
        Initial guess.
    niter : integer, optional
        The number of basin-hopping iterations. There will be a total of
        ``niter + 1`` runs of the local minimizer.
    T : float, optional
        The "temperature" parameter for the acceptance or rejection criterion.
        Higher "temperatures" mean that larger jumps in function value will be
        accepted.  For best results `T` should be comparable to the
        separation (in function value) between local minima.
    stepsize : float, optional
        Maximum step size for use in the random displacement.
    minimizer_kwargs : dict, optional
        Extra keyword arguments to be passed to the local minimizer
        `scipy.optimize.minimize` Some important options could be:

            method : str
                The minimization method (e.g. ``"L-BFGS-B"``)
            args : tuple
                Extra arguments passed to the objective function (`func`) and
                its derivatives (Jacobian, Hessian).

    take_step : callable ``take_step(x)``, optional
        Replace the default step-taking routine with this routine. The default
        step-taking routine is a random displacement of the coordinates, but
        other step-taking algorithms may be better for some systems.
        `take_step` can optionally have the attribute ``take_step.stepsize``.
        If this attribute exists, then `basinhopping` will adjust
        ``take_step.stepsize`` in order to try to optimize the global minimum
        search.
    accept_test : callable, ``accept_test(f_new=f_new, x_new=x_new, f_old=fold, x_old=x_old)``, optional
        Define a test which will be used to judge whether to accept the
        step. This will be used in addition to the Metropolis test based on
        "temperature" `T`. The acceptable return values are True,
        False, or ``"force accept"``. If any of the tests return False
        then the step is rejected. If the latter, then this will override any
        other tests in order to accept the step. This can be used, for example,
        to forcefully escape from a local minimum that `basinhopping` is
        trapped in.
    """
    # callback: callable, ``callback(x, f, accept)``, optional
    # 回调函数，用于处理找到的每一个局部最小值。`x` 和 `f` 是试探最小值的坐标和函数值，`accept` 表示该最小值是否被接受。
    # 可以用来保存最低的 N 个最小值，或者通过返回 True 指定用户定义的停止条件来停止 `basinhopping` 过程。

    # interval: integer, optional
    # 更新 `stepsize` 的间隔。

    # disp: bool, optional
    # 设置为 True 时打印状态消息。

    # niter_success: integer, optional
    # 如果全局最小值候选在这么多次迭代中保持不变，则停止运行。

    # seed: {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
    # 如果 `seed` 是 None（或 `np.random`），则使用 `numpy.random.RandomState` 单例。
    # 如果 `seed` 是一个整数，则使用一个新的 ``RandomState`` 实例，并用 `seed` 进行初始化。
    # 如果 `seed` 已经是一个 ``Generator`` 或 ``RandomState`` 实例，则直接使用该实例。
    # 指定 `seed` 可以使得最小化过程具有可重复性。使用该种子生成的随机数仅影响默认的 Metropolis `accept_test` 和默认的 `take_step`。
    # 如果提供自定义的 `take_step` 和 `accept_test`，并且这些函数使用随机数生成，则这些函数负责它们的随机数生成器状态。

    # target_accept_rate: float, optional
    # 用于调整 `stepsize` 的目标接受率。如果当前接受率高于目标接受率，则增加 `stepsize`，否则减小。
    # 取值范围为 (0, 1)，默认为 0.5。
    # .. versionadded:: 1.8.0

    # stepwise_factor: float, optional
    # 每次更新时 `stepsize` 的乘法或除法因子。取值范围为 (0, 1)，默认为 0.9。
    # .. versionadded:: 1.8.0

    # Returns
    # -------
    # res : OptimizeResult
    #     表示优化结果的 `OptimizeResult` 对象。
    #     重要属性包括：``x`` 解数组，``fun`` 在解处的函数值，以及 ``message`` 描述终止原因。
    #     通过 `lowest_optimization_result` 属性可以访问所选最小化器在最低最小值处返回的 `OptimizeResult` 对象。
    #     详细的其他属性描述请参见 `OptimizeResult`。

    # See Also
    # --------
    # minimize :
    #     每个 `basinhopping` 步骤调用一次的局部最小化函数。
    #     `minimizer_kwargs` 会传递给此例程。

    # Notes
    # -----
    Basin-hopping is a stochastic algorithm which attempts to find the global
    minimum of a smooth scalar function of one or more variables [1]_ [2]_ [3]_
    [4]_. The algorithm in its current form was described by David Wales and
    Jonathan Doye [2]_ http://www-wales.ch.cam.ac.uk/.

    The algorithm is iterative with each cycle composed of the following
    features

    1) random perturbation of the coordinates

    2) local minimization

    3) accept or reject the new coordinates based on the minimized function
       value

    The acceptance test used here is the Metropolis criterion of standard Monte
    Carlo algorithms, although there are many other possibilities [3]_.

    This global minimization method has been shown to be extremely efficient
    for a wide variety of problems in physics and chemistry. It is
    particularly useful when the function has many minima separated by large
    barriers. See the `Cambridge Cluster Database
    <https://www-wales.ch.cam.ac.uk/CCD.html>`_ for databases of molecular
    systems that have been optimized primarily using basin-hopping. This
    database includes minimization problems exceeding 300 degrees of freedom.

    See the free software program `GMIN <https://www-wales.ch.cam.ac.uk/GMIN>`_
    for a Fortran implementation of basin-hopping. This implementation has many
    variations of the procedure described above, including more
    advanced step taking algorithms and alternate acceptance criterion.

    For stochastic global optimization there is no way to determine if the true
    global minimum has actually been found. Instead, as a consistency check,
    the algorithm can be run from a number of different random starting points
    to ensure the lowest minimum found in each example has converged to the
    global minimum. For this reason, `basinhopping` will by default simply
    run for the number of iterations `niter` and return the lowest minimum
    found. It is left to the user to ensure that this is in fact the global
    minimum.

    Choosing `stepsize`:  This is a crucial parameter in `basinhopping` and
    depends on the problem being solved. The step is chosen uniformly in the
    region from x0-stepsize to x0+stepsize, in each dimension. Ideally, it
    should be comparable to the typical separation (in argument values) between
    local minima of the function being optimized. `basinhopping` will, by
    default, adjust `stepsize` to find an optimal value, but this may take
    many iterations. You will get quicker results if you set a sensible
    initial value for ``stepsize``.

    Choosing `T`: The parameter `T` is the "temperature" used in the
    Metropolis criterion. Basinhopping steps are always accepted if
    ``func(xnew) < func(xold)``. Otherwise, they are accepted with
    probability::

        exp( -(func(xnew) - func(xold)) / T )

    So, for best results, `T` should to be comparable to the typical
    difference (in function values) between local minima. (The height of
    "walls" between local minima is irrelevant.)



    If `T` is 0, the algorithm becomes Monotonic Basin-Hopping, in which all
    steps that increase energy are rejected.



    .. versionadded:: 0.12.0



    References
    ----------
    .. [1] Wales, David J. 2003, Energy Landscapes, Cambridge University Press,
        Cambridge, UK.
    .. [2] Wales, D J, and Doye J P K, Global Optimization by Basin-Hopping and
        the Lowest Energy Structures of Lennard-Jones Clusters Containing up to
        110 Atoms.  Journal of Physical Chemistry A, 1997, 101, 5111.
    .. [3] Li, Z. and Scheraga, H. A., Monte Carlo-minimization approach to the
        multiple-minima problem in protein folding, Proc. Natl. Acad. Sci. USA,
        1987, 84, 6611.
    .. [4] Wales, D. J. and Scheraga, H. A., Global optimization of clusters,
        crystals, and biomolecules, Science, 1999, 285, 1368.
    .. [5] Olson, B., Hashmi, I., Molloy, K., and Shehu1, A., Basin Hopping as
        a General and Versatile Optimization Framework for the Characterization
        of Biological Macromolecules, Advances in Artificial Intelligence,
        Volume 2012 (2012), Article ID 674832, :doi:`10.1155/2012/674832`



    Examples
    --------
    The following example is a 1-D minimization problem, with many
    local minima superimposed on a parabola.



    >>> import numpy as np
    >>> from scipy.optimize import basinhopping
    >>> func = lambda x: np.cos(14.5 * x - 0.3) + (x + 0.2) * x
    >>> x0 = [1.]



    Basinhopping, internally, uses a local minimization algorithm. We will use
    the parameter `minimizer_kwargs` to tell basinhopping which algorithm to
    use and how to set up that minimizer. This parameter will be passed to
    `scipy.optimize.minimize`.



    >>> minimizer_kwargs = {"method": "BFGS"}
    >>> ret = basinhopping(func, x0, minimizer_kwargs=minimizer_kwargs,
    ...                    niter=200)



    >>> # the global minimum is:
    >>> ret.x, ret.fun
    -0.1951, -1.0009



    Next consider a 2-D minimization problem. Also, this time, we
    will use gradient information to significantly speed up the search.



    >>> def func2d(x):
    ...     f = np.cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] +
    ...                                                            0.2) * x[0]
    ...     df = np.zeros(2)
    ...     df[0] = -14.5 * np.sin(14.5 * x[0] - 0.3) + 2. * x[0] + 0.2
    ...     df[1] = 2. * x[1] + 0.2
    ...     return f, df



    We'll also use a different local minimization algorithm. Also, we must tell
    the minimizer that our function returns both energy and gradient (Jacobian).



    >>> minimizer_kwargs = {"method":"L-BFGS-B", "jac":True}
    >>> x0 = [1.0, 1.0]
    >>> ret = basinhopping(func2d, x0, minimizer_kwargs=minimizer_kwargs,
    ...                    niter=200)
    >>> print("global minimum: x = [%.4f, %.4f], f(x) = %.4f" % (ret.x[0],
    """ # numpy/numpydoc#87  # noqa: E501
    # 如果目标接受率小于等于0或大于等于1，则引发值错误
    if target_accept_rate <= 0. or target_accept_rate >= 1.:
        raise ValueError('target_accept_rate has to be in range (0, 1)')
    # 如果步骤因子小于等于0或大于等于1，则引发值错误
    if stepwise_factor <= 0. or stepwise_factor >= 1.:
        raise ValueError('stepwise_factor has to be in range (0, 1)')

    # 将x0转换为NumPy数组
    x0 = np.array(x0)

    # 设置np.random生成器
    rng = check_random_state(seed)

    # 设置最小化器
    # 如果未提供minimizer_kwargs，则设为空字典
    if minimizer_kwargs is None:
        minimizer_kwargs = dict()
    # 封装最小化器函数，并传入func和minimizer_kwargs
    wrapped_minimizer = MinimizerWrapper(scipy.optimize.minimize, func,
                                         **minimizer_kwargs)

    # 设置步进算法
    # 如果给定了 take_step 参数
    if take_step is not None:
        # 检查 take_step 是否可调用，如果不可调用则引发 TypeError 异常
        if not callable(take_step):
            raise TypeError("take_step must be callable")
        
        # 如果 take_step 具有 stepsize 属性，则使用 AdaptiveStepsize 封装 take_step
        # 控制其 stepsize
        if hasattr(take_step, "stepsize"):
            take_step_wrapped = AdaptiveStepsize(
                take_step, interval=interval,
                accept_rate=target_accept_rate,
                factor=stepwise_factor,
                verbose=disp)
        else:
            # 否则直接使用原始的 take_step
            take_step_wrapped = take_step
    else:
        # 如果未提供 take_step 参数，则使用默认的 RandomDisplacement
        displace = RandomDisplacement(stepsize=stepsize, random_gen=rng)
        take_step_wrapped = AdaptiveStepsize(displace, interval=interval,
                                             accept_rate=target_accept_rate,
                                             factor=stepwise_factor,
                                             verbose=disp)

    # 设置接受测试列表
    accept_tests = []
    if accept_test is not None:
        # 如果 accept_test 不可调用则引发 TypeError 异常
        if not callable(accept_test):
            raise TypeError("accept_test must be callable")
        # 将 accept_test 添加到 accept_tests 列表中
        accept_tests = [accept_test]

    # 使用默认的 Metropolis 算法
    metropolis = Metropolis(T, random_gen=rng)
    accept_tests.append(metropolis)

    # 如果 niter_success 为 None，则设置为 niter + 2
    if niter_success is None:
        niter_success = niter + 2

    # 创建 BasinHoppingRunner 对象
    bh = BasinHoppingRunner(x0, wrapped_minimizer, take_step_wrapped,
                            accept_tests, disp=disp)

    # 在构造 BasinHoppingRunner 期间会调用 wrapped_minimizer 一次，因此运行回调函数
    if callable(callback):
        callback(bh.storage.minres.x, bh.storage.minres.fun, True)

    # 开始主迭代循环
    count, i = 0, 0
    # 初始化消息列表
    message = ["requested number of basinhopping iterations completed"
               " successfully"]
    for i in range(niter):
        # 执行一次循环迭代
        new_global_min = bh.one_cycle()

        # 如果定义了回调函数，则调用回调函数
        if callable(callback):
            # 是否应传递 x 的副本？
            val = callback(bh.xtrial, bh.energy_trial, bh.accept)
            if val is not None:
                if val:
                    message = ["callback function requested stop early by"
                               "returning True"]
                    break

        # 计数器递增
        count += 1
        # 如果发现新的全局最小值，则重置计数器
        if new_global_min:
            count = 0
        # 如果计数器超过成功迭代次数 niter_success，则更新消息并跳出循环
        elif count > niter_success:
            message = ["success condition satisfied"]
            break

    # 准备返回对象
    res = bh.res
    # 设置最优结果
    res.lowest_optimization_result = bh.storage.get_lowest()
    # 复制最优解
    res.x = np.copy(res.lowest_optimization_result.x)
    # 设置最优函数值
    res.fun = res.lowest_optimization_result.fun
    # 设置消息
    res.message = message
    # 设置迭代次数
    res.nit = i + 1
    # 设置成功标志
    res.success = res.lowest_optimization_result.success
    # 返回结果对象
    return res
```