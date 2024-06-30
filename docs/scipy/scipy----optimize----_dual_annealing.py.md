# `D:\src\scipysrc\scipy\scipy\optimize\_dual_annealing.py`

```
# Dual Annealing implementation.
# Copyright (c) 2018 Sylvain Gubian <sylvain.gubian@pmi.com>,
# Yang Xiang <yang.xiang@pmi.com>
# Author: Sylvain Gubian, Yang Xiang, PMP S.A.

"""
A Dual Annealing global optimization algorithm
"""

# 导入必要的库
import numpy as np  # 导入numpy库，用于数值计算
from scipy.optimize import OptimizeResult  # 导入OptimizeResult类，优化结果的封装
from scipy.optimize import minimize, Bounds  # 导入minimize函数和Bounds类
from scipy.special import gammaln  # 导入gammaln函数，伽玛函数的对数
from scipy._lib._util import check_random_state  # 导入check_random_state函数，检查随机数状态
from scipy.optimize._constraints import new_bounds_to_old  # 导入new_bounds_to_old函数，转换新旧约束表示

__all__ = ['dual_annealing']  # 模块的公开接口，包含dual_annealing函数

class VisitingDistribution:
    """
    Class used to generate new coordinates based on the distorted
    Cauchy-Lorentz distribution. Depending on the steps within the strategy
    chain, the class implements the strategy for generating new location
    changes.

    Parameters
    ----------
    lb : array_like
        A 1-D NumPy ndarray containing lower bounds of the generated
        components. Neither NaN or inf are allowed.
    ub : array_like
        A 1-D NumPy ndarray containing upper bounds for the generated
        components. Neither NaN or inf are allowed.
    visiting_param : float
        Parameter for visiting distribution. Default value is 2.62.
        Higher values give the visiting distribution a heavier tail, this
        makes the algorithm jump to a more distant region.
        The value range is (1, 3]. Its value is fixed for the life of the
        object.
    rand_gen : {`~numpy.random.RandomState`, `~numpy.random.Generator`}
        A `~numpy.random.RandomState`, `~numpy.random.Generator` object
        for using the current state of the created random generator container.
    """

    TAIL_LIMIT = 1.e8  # 尾部限制常数
    MIN_VISIT_BOUND = 1.e-10  # 最小访问边界

    def __init__(self, lb, ub, visiting_param, rand_gen):
        """
        Initialize the VisitingDistribution object.

        Parameters
        ----------
        lb : array_like
            Lower bounds for the generated components.
        ub : array_like
            Upper bounds for the generated components.
        visiting_param : float
            Parameter controlling the visiting distribution's tail heaviness.
        rand_gen : {`~numpy.random.RandomState`, `~numpy.random.Generator`}
            Random number generator.

        Notes
        -----
        Initializes parameters and precomputes factors for the visiting distribution.
        """
        self._visiting_param = visiting_param  # 访问分布的参数
        self.rand_gen = rand_gen  # 随机数生成器对象
        self.lower = lb  # 生成组件的下界
        self.upper = ub  # 生成组件的上界
        self.bound_range = ub - lb  # 上界和下界之间的范围

        # Precompute factors based on visiting_param
        self._factor2 = np.exp((4.0 - self._visiting_param) * np.log(
            self._visiting_param - 1.0))  # 预计算因子2
        self._factor3 = np.exp((2.0 - self._visiting_param) * np.log(2.0)
                               / (self._visiting_param - 1.0))  # 预计算因子3
        self._factor4_p = np.sqrt(np.pi) * self._factor2 / (self._factor3 * (
            3.0 - self._visiting_param))  # 预计算因子4的正分量

        self._factor5 = 1.0 / (self._visiting_param - 1.0) - 0.5  # 预计算因子5
        self._d1 = 2.0 - self._factor5  # 预计算因子d1
        self._factor6 = np.pi * (1.0 - self._factor5) / np.sin(
            np.pi * (1.0 - self._factor5)) / np.exp(gammaln(self._d1))  # 预计算因子6
    def visiting(self, x, step, temperature):
        """ 根据策略链中的步骤生成新的坐标。
        如果步骤小于坐标维度，则同时改变所有坐标的值；
        否则，仅改变一个坐标的值。
        新值由 visit_fn 方法计算得出。
        """
        dim = x.size
        if step < dim:
            # 改变所有坐标的访问值
            visits = self.visit_fn(temperature, dim)
            upper_sample, lower_sample = self.rand_gen.uniform(size=2)
            visits[visits > self.TAIL_LIMIT] = self.TAIL_LIMIT * upper_sample
            visits[visits < -self.TAIL_LIMIT] = -self.TAIL_LIMIT * lower_sample
            x_visit = visits + x
            a = x_visit - self.lower
            b = np.fmod(a, self.bound_range) + self.bound_range
            x_visit = np.fmod(b, self.bound_range) + self.lower
            x_visit[np.fabs(
                x_visit - self.lower) < self.MIN_VISIT_BOUND] += 1.e-10
        else:
            # 根据策略链步骤仅改变一个坐标的值
            x_visit = np.copy(x)
            visit = self.visit_fn(temperature, 1)[0]
            if visit > self.TAIL_LIMIT:
                visit = self.TAIL_LIMIT * self.rand_gen.uniform()
            elif visit < -self.TAIL_LIMIT:
                visit = -self.TAIL_LIMIT * self.rand_gen.uniform()
            index = step - dim
            x_visit[index] = visit + x[index]
            a = x_visit[index] - self.lower[index]
            b = np.fmod(a, self.bound_range[index]) + self.bound_range[index]
            x_visit[index] = np.fmod(b, self.bound_range[
                index]) + self.lower[index]
            if np.fabs(x_visit[index] - self.lower[
                    index]) < self.MIN_VISIT_BOUND:
                x_visit[index] += self.MIN_VISIT_BOUND
        return x_visit

    def visit_fn(self, temperature, dim):
        """ 根据参考文献[2]中第405页的 Visita 公式计算访问值 """
        x, y = self.rand_gen.normal(size=(dim, 2)).T

        factor1 = np.exp(np.log(temperature) / (self._visiting_param - 1.0))
        factor4 = self._factor4_p * factor1

        # 计算 sigmax
        x *= np.exp(-(self._visiting_param - 1.0) * np.log(
            self._factor6 / factor4) / (3.0 - self._visiting_param))

        den = np.exp((self._visiting_param - 1.0) * np.log(np.fabs(y)) /
                     (3.0 - self._visiting_param))

        return x / den
# 用于记录能量状态的类。在任何时候，它知道当前使用的坐标和最近找到的最佳位置。
class EnergyState:
    # 生成有效起始点的最大尝试次数
    MAX_REINIT_COUNT = 1000

    # 初始化方法，设置初始属性
    def __init__(self, lower, upper, callback=None):
        self.ebest = None  # 最佳能量值
        self.current_energy = None  # 当前能量值
        self.current_location = None  # 当前位置
        self.xbest = None  # 最佳坐标
        self.lower = lower  # 初始随机分量的下界
        self.upper = upper  # 初始随机分量的上界
        self.callback = callback  # 回调函数，对所有找到的极小值进行调用
    # 重置算法状态，初始化当前搜索位置。如果未提供 x0，则在指定边界内生成随机位置。
    def reset(self, func_wrapper, rand_gen, x0=None):
        """
        Initialize current location is the search domain. If `x0` is not
        provided, a random location within the bounds is generated.
        """
        # 如果未提供初始位置 x0，则生成一个在指定边界内的随机位置
        if x0 is None:
            self.current_location = rand_gen.uniform(self.lower, self.upper,
                                                     size=len(self.lower))
        else:
            # 如果提供了初始位置 x0，则复制该位置作为当前位置
            self.current_location = np.copy(x0)
        # 初始化错误标志和重新初始化计数器
        init_error = True
        reinit_counter = 0
        # 进入初始化错误处理循环
        while init_error:
            # 计算当前位置的能量值（调用外部函数来计算）
            self.current_energy = func_wrapper.fun(self.current_location)
            # 检查能量值是否为 None
            if self.current_energy is None:
                raise ValueError('Objective function is returning None')
            # 检查能量值是否为无穷大或 NaN
            if (not np.isfinite(self.current_energy) or np.isnan(
                    self.current_energy)):
                # 如果出现无效能量值，并且重新初始化计数器超过预设值，则停止算法
                if reinit_counter >= EnergyState.MAX_REINIT_COUNT:
                    init_error = False
                    message = (
                        'Stopping algorithm because function '
                        'create NaN or (+/-) infinity values even with '
                        'trying new random parameters'
                    )
                    raise ValueError(message)
                # 否则，重新生成随机位置并增加重新初始化计数器
                self.current_location = rand_gen.uniform(self.lower,
                                                         self.upper,
                                                         size=self.lower.size)
                reinit_counter += 1
            else:
                # 如果能够成功初始化，设置初始化错误标志为 False
                init_error = False
            # 如果是第一次重置，则初始化 ebest 和 xbest
            if self.ebest is None and self.xbest is None:
                self.ebest = self.current_energy
                self.xbest = np.copy(self.current_location)
            # 否则，在重新退火重置的情况下保持它们不变

    # 更新当前最佳能量值和位置
    def update_best(self, e, x, context):
        self.ebest = e
        self.xbest = np.copy(x)
        # 如果设置了回调函数，则调用回调函数，并根据返回值决定是否提前停止算法
        if self.callback is not None:
            val = self.callback(x, e, context)
            if val is not None:
                if val:
                    return ('Callback function requested to stop early by '
                           'returning True')

    # 更新当前能量值和位置
    def update_current(self, e, x):
        self.current_energy = e
        self.current_location = np.copy(x)
# 实现了策略链中的马尔可夫链，用于位置接受和局部搜索决策的类

class StrategyChain:
    """
    Class that implements within a Markov chain the strategy for location
    acceptance and local search decision making.

    Parameters
    ----------
    acceptance_param : float
        Parameter for acceptance distribution. It is used to control the
        probability of acceptance. The lower the acceptance parameter, the
        smaller the probability of acceptance. Default value is -5.0 with
        a range (-1e4, -5].
    visit_dist : VisitingDistribution
        Instance of `VisitingDistribution` class.
    func_wrapper : ObjectiveFunWrapper
        Instance of `ObjectiveFunWrapper` class.
    minimizer_wrapper: LocalSearchWrapper
        Instance of `LocalSearchWrapper` class.
    rand_gen : {None, int, `numpy.random.Generator`,
                `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
    energy_state: EnergyState
        Instance of `EnergyState` class.

    """

    def __init__(self, acceptance_param, visit_dist, func_wrapper,
                 minimizer_wrapper, rand_gen, energy_state):
        # 初始化本地策略链的最小能量和位置
        self.emin = energy_state.current_energy
        self.xmin = np.array(energy_state.current_location)
        
        # 全局优化器状态
        self.energy_state = energy_state
        
        # 接受参数
        self.acceptance_param = acceptance_param
        
        # 访问分布实例
        self.visit_dist = visit_dist
        
        # 目标函数包装器
        self.func_wrapper = func_wrapper
        
        # 局部最小化器包装器
        self.minimizer_wrapper = minimizer_wrapper
        
        # 未改进的索引计数
        self.not_improved_idx = 0
        
        # 最大未改进索引计数
        self.not_improved_max_idx = 1000
        
        # 随机数生成器
        self._rand_gen = rand_gen
        
        # 温度步长
        self.temperature_step = 0
        
        # K 值
        self.K = 100 * len(energy_state.current_location)
    # 根据接受-拒绝准则确定是否接受新的状态，并更新相关状态
    def accept_reject(self, j, e, x_visit):
        # 生成一个均匀分布的随机数
        r = self._rand_gen.uniform()
        # 计算接受概率 pqv_temp
        pqv_temp = 1.0 - ((1.0 - self.acceptance_param) *
            (e - self.energy_state.current_energy) / self.temperature_step)
        # 如果接受概率小于等于0，则将 pqv 设为0
        if pqv_temp <= 0.:
            pqv = 0.
        else:
            # 计算实际的接受概率 pqv
            pqv = np.exp(np.log(pqv_temp) / (
                1. - self.acceptance_param))

        # 判断生成的随机数 r 是否小于等于接受概率 pqv，若是则接受新的状态
        if r <= pqv:
            # 我们接受新的位置并更新状态
            self.energy_state.update_current(e, x_visit)
            self.xmin = np.copy(self.energy_state.current_location)

        # 如果长时间没有改善
        if self.not_improved_idx >= self.not_improved_max_idx:
            # 如果是第一次迭代或者当前能量小于最小能量，更新最小能量和位置
            if j == 0 or self.energy_state.current_energy < self.emin:
                self.emin = self.energy_state.current_energy
                self.xmin = np.copy(self.energy_state.current_location)

    # 运行模拟退火算法的主循环
    def run(self, step, temperature):
        # 计算当前温度步长
        self.temperature_step = temperature / float(step + 1)
        # 增加未改善计数器
        self.not_improved_idx += 1
        # 对当前位置的每个维度进行两次循环
        for j in range(self.energy_state.current_location.size * 2):
            # 如果是第一次迭代，设置 energy_state_improved 为 True
            if j == 0:
                if step == 0:
                    self.energy_state_improved = True
                else:
                    self.energy_state_improved = False
            # 计算访问分布，并得到新的位置 x_visit
            x_visit = self.visit_dist.visiting(
                self.energy_state.current_location, j, temperature)
            # 调用目标函数计算当前位置的能量 e
            e = self.func_wrapper.fun(x_visit)
            # 如果能量 e 比当前能量小，则接受新位置
            if e < self.energy_state.current_energy:
                # 我们获得了更好的能量值
                self.energy_state.update_current(e, x_visit)
                # 如果能量 e 比最佳能量 ebest 小，则更新最佳位置和能量
                if e < self.energy_state.ebest:
                    val = self.energy_state.update_best(e, x_visit, 0)
                    if val is not None:
                        if val:
                            return val
                    # 标记状态改善，重置未改善计数器
                    self.energy_state_improved = True
                    self.not_improved_idx = 0
            else:
                # 如果没有改善，根据接受-拒绝准则决定是否接受新位置
                self.accept_reject(j, e, x_visit)
            # 如果函数评估次数达到最大限制，则返回达到最大函数调用数的消息
            if self.func_wrapper.nfev >= self.func_wrapper.maxfun:
                return ('Maximum number of function call reached '
                        'during annealing')
        # 结束 StrategyChain 循环
    def local_search(self):
        # Decision making for performing a local search
        # based on strategy chain results
        # If energy has been improved or no improvement since too long,
        # performing a local search with the best strategy chain location
        
        # 如果能量状态有所改善或者长时间没有改善，决定是否进行局部搜索
        if self.energy_state_improved:
            # Global energy has improved, let's see if LS improves further
            # 如果全局能量已经改善，查看局部搜索是否进一步改善
            e, x = self.minimizer_wrapper.local_search(self.energy_state.xbest,
                                                       self.energy_state.ebest)
            if e < self.energy_state.ebest:
                self.not_improved_idx = 0
                val = self.energy_state.update_best(e, x, 1)
                if val is not None:
                    if val:
                        return val
                self.energy_state.update_current(e, x)
            # Check if maximum number of function calls reached during LS
            # 检查是否在局部搜索中达到了最大函数调用次数
            if self.func_wrapper.nfev >= self.func_wrapper.maxfun:
                return ('Maximum number of function call reached '
                        'during local search')

        # Check probability of a need to perform a LS even if no improvement
        # 检查即使没有改善，也需要执行局部搜索的概率
        do_ls = False
        if self.K < 90 * len(self.energy_state.current_location):
            pls = np.exp(self.K * (
                self.energy_state.ebest - self.energy_state.current_energy) /
                self.temperature_step)
            if pls >= self._rand_gen.uniform():
                do_ls = True
        
        # Global energy not improved, let's see what LS gives
        # on the best strategy chain location
        # 如果全局能量没有改善，查看在最佳策略链位置进行局部搜索的结果
        if self.not_improved_idx >= self.not_improved_max_idx:
            do_ls = True
        if do_ls:
            e, x = self.minimizer_wrapper.local_search(self.xmin, self.emin)
            self.xmin = np.copy(x)
            self.emin = e
            self.not_improved_idx = 0
            self.not_improved_max_idx = self.energy_state.current_location.size
            if e < self.energy_state.ebest:
                val = self.energy_state.update_best(
                    self.emin, self.xmin, 2)
                if val is not None:
                    if val:
                        return val
                self.energy_state.update_current(e, x)
            # Check if maximum number of function calls reached during dual annealing
            # 检查是否在双退火过程中达到了最大函数调用次数
            if self.func_wrapper.nfev >= self.func_wrapper.maxfun:
                return ('Maximum number of function call reached '
                        'during dual annealing')
class ObjectiveFunWrapper:
    # 封装一个目标函数的包装器类

    def __init__(self, func, maxfun=1e7, *args):
        # 初始化函数，接受目标函数、最大函数评估次数和其他参数
        self.func = func
        self.args = args
        # 目标函数评估次数
        self.nfev = 0
        # 梯度函数评估次数（如果使用）
        self.ngev = 0
        # 目标函数的黑塞矩阵评估次数（如果使用）
        self.nhev = 0
        self.maxfun = maxfun

    def fun(self, x):
        # 函数方法，对目标函数进行评估，并增加评估次数计数
        self.nfev += 1
        return self.func(x, *self.args)


class LocalSearchWrapper:
    """
    Class used to wrap around the minimizer used for local search
    Default local minimizer is SciPy minimizer L-BFGS-B
    """
    # 用于封装本地搜索最小化器的类
    # 默认本地最小化器是 SciPy 的 L-BFGS-B 最小化器

    LS_MAXITER_RATIO = 6
    LS_MAXITER_MIN = 100
    LS_MAXITER_MAX = 1000

    def __init__(self, search_bounds, func_wrapper, *args, **kwargs):
        # 初始化函数，接受搜索边界、函数包装器和其他参数
        self.func_wrapper = func_wrapper
        self.kwargs = kwargs
        self.jac = self.kwargs.get('jac', None)
        self.hess = self.kwargs.get('hess', None)
        self.hessp = self.kwargs.get('hessp', None)
        self.kwargs.pop("args", None)
        self.minimizer = minimize
        bounds_list = list(zip(*search_bounds))
        self.lower = np.array(bounds_list[0])
        self.upper = np.array(bounds_list[1])

        # 如果未指定最小化器，则使用 SciPy minimize 方法 'L-BFGS-B'
        if not self.kwargs:
            n = len(self.lower)
            ls_max_iter = min(max(n * self.LS_MAXITER_RATIO,
                                  self.LS_MAXITER_MIN),
                              self.LS_MAXITER_MAX)
            self.kwargs['method'] = 'L-BFGS-B'
            self.kwargs['options'] = {
                'maxiter': ls_max_iter,
            }
            self.kwargs['bounds'] = list(zip(self.lower, self.upper))
        else:
            # 如果定义了梯度函数，使用包装后的版本
            if callable(self.jac):
                def wrapped_jac(x):
                    return self.jac(x, *args)
                self.kwargs['jac'] = wrapped_jac
            # 如果定义了黑塞矩阵函数，使用包装后的版本
            if callable(self.hess):
                def wrapped_hess(x):
                    return self.hess(x, *args)
                self.kwargs['hess'] = wrapped_hess
            # 如果定义了黑塞矩阵向量积函数，使用包装后的版本
            if callable(self.hessp):
                def wrapped_hessp(x, p):
                    return self.hessp(x, p, *args)
                self.kwargs['hessp'] = wrapped_hessp
    # 定义一个局部搜索函数，从给定的起始点 x 开始，能量值为 e
    def local_search(self, x, e):
        # 复制 x 的副本，以便后续比较
        x_tmp = np.copy(x)
        # 调用 self.minimizer 方法来最小化 self.func_wrapper.fun 函数，起始点为 x，传入额外参数 **self.kwargs
        mres = self.minimizer(self.func_wrapper.fun, x, **self.kwargs)
        # 如果 mres 中包含 'njev' 属性，将其值加到 self.func_wrapper.ngev 上
        if 'njev' in mres:
            self.func_wrapper.ngev += mres.njev
        # 如果 mres 中包含 'nhev' 属性，将其值加到 self.func_wrapper.nhev 上
        if 'nhev' in mres:
            self.func_wrapper.nhev += mres.nhev
        # 检查 mres.x 和 mres.fun 是否都是有限的值
        is_finite = np.all(np.isfinite(mres.x)) and np.isfinite(mres.fun)
        # 检查 mres.x 是否在 self.lower 和 self.upper 定义的边界之内
        in_bounds = np.all(mres.x >= self.lower) and np.all(mres.x <= self.upper)
        # 判断 mres 是否为有效解
        is_valid = is_finite and in_bounds

        # 如果 mres 是有效解并且其函数值比 e 小，则返回 mres 的函数值和解向量 mres.x
        if is_valid and mres.fun < e:
            return mres.fun, mres.x
        else:
            # 否则返回初始的能量值 e 和初始的解向量 x_tmp
            return e, x_tmp
# 使用 Dual Annealing 方法寻找目标函数的全局最小值
def dual_annealing(func, bounds, args=(), maxiter=1000,
                   minimizer_kwargs=None, initial_temp=5230.,
                   restart_temp_ratio=2.e-5, visit=2.62, accept=-5.0,
                   maxfun=1e7, seed=None, no_local_search=False,
                   callback=None, x0=None):
    """
    Find the global minimum of a function using Dual Annealing.

    Parameters
    ----------
    func : callable
        要最小化的目标函数。必须以形式 ``f(x, *args)`` 给出，其中 ``x`` 是一个一维数组的参数，
        而 ``args`` 是一个元组，包含了完全指定函数所需的任何额外固定参数。
    bounds : sequence or `Bounds`
        变量的边界。有两种指定边界的方式：
        
        1. `Bounds` 类的实例。
        2. 对于 `x` 中的每个元素，使用 ``(min, max)`` 对组成的序列。

    args : tuple, optional
        需要完全指定目标函数的任何额外固定参数。
    maxiter : int, optional
        全局搜索迭代的最大次数。默认值为 1000。
    minimizer_kwargs : dict, optional
        传递给局部最小化器 (`minimize`) 的关键字参数。
        一个重要的选项可以是最小化器使用的方法 ``method``。
        如果没有提供关键字参数，则局部最小化器默认为 'L-BFGS-B' 并使用已提供的边界。
        如果指定了 `minimizer_kwargs`，则字典必须包含控制局部最小化所需的所有参数。
        此字典中自动忽略 `args`，因为它会自动传递。`bounds` 不会自动传递给局部最小化器，
        因为该方法可能不支持这些边界。
    initial_temp : float, optional
        初始温度，使用较高的值有助于更广泛地搜索能量景观，使得 Dual Annealing 能够逃离
        其中被困的局部最小值。默认值为 5230。范围是 (0.01, 5.e4]。
    restart_temp_ratio : float, optional
        在退火过程中，温度逐渐降低，当达到 ``initial_temp * restart_temp_ratio`` 时，
        将触发重新退火过程。比率的默认值为 2e-5。范围是 (0, 1)。
    visit : float, optional
        访问分布的参数。默认值为 2.62。较高的值使访问分布具有更重的尾部，这使得算法
        能够跳跃到更远的区域。值范围是 (1, 3]。
    accept : float, optional
        接受分布的参数。用于控制接受的概率。接受参数越低，接受概率越小。
        默认值为 -5.0，范围为 (-1e4, -5]。
    """
    # 最大函数调用次数限制，用于控制优化算法的迭代次数
    maxfun : int, optional
        # 如果算法在局部搜索中，超过这个限制后，将会停止
        Soft limit for the number of objective function calls. If the
        algorithm is in the middle of a local search, this number will be
        exceeded, the algorithm will stop just after the local search is
        done. Default value is 1e7.
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        # 控制随机数种子以实现可重复的最小化过程
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
        Specify `seed` for repeatable minimizations. The random numbers
        generated with this seed only affect the visiting distribution function
        and new coordinates generation.
    no_local_search : bool, optional
        # 如果设置为True，则执行传统的广义模拟退火算法，不应用局部搜索策略
        If `no_local_search` is set to True, a traditional Generalized
        Simulated Annealing will be performed with no local search
        strategy applied.
    callback : callable, optional
        # 回调函数，用于在找到所有最小值时调用
        A callback function with signature ``callback(x, f, context)``,
        which will be called for all minima found.
        ``x`` and ``f`` are the coordinates and function value of the
        latest minimum found, and ``context`` has value in [0, 1, 2], with the
        following meaning:

            - 0: minimum detected in the annealing process.
            - 1: detection occurred in the local search process.
            - 2: detection done in the dual annealing process.

        If the callback implementation returns True, the algorithm will stop.
    x0 : ndarray, shape(n,), optional
        # 单个N维起始点的坐标
        Coordinates of a single N-D starting point.

    Returns
    -------
    res : OptimizeResult
        # 优化结果对象，包括解决方案数组``x``、函数在解决方案处的值``fun``以及描述终止原因的``message``
        The optimization result represented as a `OptimizeResult` object.
        Important attributes are: ``x`` the solution array, ``fun`` the value
        of the function at the solution, and ``message`` which describes the
        cause of the termination.
        See `OptimizeResult` for a description of other attributes.

    Notes
    -----
    # 此函数实现双退火优化算法。这种随机化方法源于[3]_，结合了CSA（经典模拟退火）和FSA（快速模拟退火）[1]_ [2]_的泛化，同时结合了在接受的位置上应用局部搜索的策略[4]_。
    # 该算法的另一种实现描述在[5]_中，其基准测试结果在[6]_中展示。此方法引入了一种高级方法来优化广义退火过程中找到的解。该算法使用畸变的柯西-洛伦兹访问分布，其形状由参数:math:`q_{v}`控制。
    This function implements the Dual Annealing optimization. This stochastic
    approach derived from [3]_ combines the generalization of CSA (Classical
    Simulated Annealing) and FSA (Fast Simulated Annealing) [1]_ [2]_ coupled
    to a strategy for applying a local search on accepted locations [4]_.
    An alternative implementation of this same algorithm is described in [5]_
    and benchmarks are presented in [6]_. This approach introduces an advanced
    method to refine the solution found by the generalized annealing
    process. This algorithm uses a distorted Cauchy-Lorentz visiting
    distribution, with its shape controlled by the parameter :math:`q_{v}`
    """
    .. math::

        g_{q_{v}}(\\Delta x(t)) \\propto \\frac{ \\
        \\left[T_{q_{v}}(t) \\right]^{-\\frac{D}{3-q_{v}}}}{ \\
        \\left[{1+(q_{v}-1)\\frac{(\\Delta x(t))^{2}} { \\
        \\left[T_{q_{v}}(t)\\right]^{\\frac{2}{3-q_{v}}}}}\\right]^{ \\
        \\frac{1}{q_{v}-1}+\\frac{D-1}{2}}}

    Where :math:`t` is the artificial time. This visiting distribution is used
    to generate a trial jump distance :math:`\\Delta x(t)` of variable
    :math:`x(t)` under artificial temperature :math:`T_{q_{v}}(t)`.

    From the starting point, after calling the visiting distribution
    function, the acceptance probability is computed as follows:

    .. math::

        p_{q_{a}} = \\min{\\{1,\\left[1-(1-q_{a}) \\beta \\Delta E \\right]^{ \\
        \\frac{1}{1-q_{a}}}\\}}

    Where :math:`q_{a}` is a acceptance parameter. For :math:`q_{a}<1`, zero
    acceptance probability is assigned to the cases where

    .. math::

        [1-(1-q_{a}) \\beta \\Delta E] < 0

    The artificial temperature :math:`T_{q_{v}}(t)` is decreased according to

    .. math::

        T_{q_{v}}(t) = T_{q_{v}}(1) \\frac{2^{q_{v}-1}-1}{\\left( \\
        1 + t\\right)^{q_{v}-1}-1}

    Where :math:`q_{v}` is the visiting parameter.

    This section defines mathematical formulations and descriptions related to
    the generalized simulated annealing process based on Tsallis statistics.
    These equations govern the behavior of the trial jumps, acceptance
    probabilities, and temperature adjustments during the optimization process.

    .. versionadded:: 1.2.0

    References
    ----------
    .. [1] Tsallis C. Possible generalization of Boltzmann-Gibbs
        statistics. Journal of Statistical Physics, 52, 479-487 (1998).
    .. [2] Tsallis C, Stariolo DA. Generalized Simulated Annealing.
        Physica A, 233, 395-406 (1996).
    .. [3] Xiang Y, Sun DY, Fan W, Gong XG. Generalized Simulated
        Annealing Algorithm and Its Application to the Thomson Model.
        Physics Letters A, 233, 216-220 (1997).
    .. [4] Xiang Y, Gong XG. Efficiency of Generalized Simulated
        Annealing. Physical Review E, 62, 4473 (2000).
    .. [5] Xiang Y, Gubian S, Suomela B, Hoeng J. Generalized
        Simulated Annealing for Efficient Global Optimization: the GenSA
        Package for R. The R Journal, Volume 5/1 (2013).
    .. [6] Mullen, K. Continuous Global Optimization in R. Journal of
        Statistical Software, 60(6), 1 - 45, (2014).
        :doi:`10.18637/jss.v060.i06`

    Examples
    --------
    The following example is a 10-D problem, with many local minima.
    The function involved is called Rastrigin
    (https://en.wikipedia.org/wiki/Rastrigin_function)

    >>> import numpy as np
    >>> from scipy.optimize import dual_annealing
    >>> func = lambda x: np.sum(x*x - 10*np.cos(2*np.pi*x)) + 10*np.size(x)
    >>> lw = [-5.12] * 10
    >>> up = [5.12] * 10
    >>> ret = dual_annealing(func, bounds=list(zip(lw, up)))
    >>> ret.x
    array([-4.26437714e-09, -3.91699361e-09, -1.86149218e-09, -3.97165720e-09,
           -6.29151648e-09, -6.53145322e-09, -3.93616815e-09, -6.55623025e-09,
           -6.05775280e-09, -5.00668935e-09]) # random
    >>> ret.fun
    0.000000

    """
    # 如果 bounds 是 Bounds 类的实例，则将其转换为旧版界限形式
    if isinstance(bounds, Bounds):
        bounds = new_bounds_to_old(bounds.lb, bounds.ub, len(bounds.lb))

    # 如果提供了初始点 x0 并且其长度与 bounds 不匹配，则抛出数值错误
    if x0 is not None and not len(x0) == len(bounds):
        raise ValueError('Bounds size does not match x0')

    # 将 bounds 转换为 lower 和 upper 边界数组
    lu = list(zip(*bounds))
    lower = np.array(lu[0])
    upper = np.array(lu[1])

    # 检查重新启动温度比率是否在合适范围内
    if restart_temp_ratio <= 0. or restart_temp_ratio >= 1.:
        raise ValueError('Restart temperature ratio has to be in range (0, 1)')

    # 检查边界是否有效，包括是否包含无穷大或NaN值
    if (np.any(np.isinf(lower)) or np.any(np.isinf(upper)) or np.any(
            np.isnan(lower)) or np.any(np.isnan(upper))):
        raise ValueError('Some bounds values are inf values or nan values')

    # 检查边界是否一致，即 lower 是否全部小于 upper
    if not np.all(lower < upper):
        raise ValueError('Bounds are not consistent min < max')

    # 检查 lower 和 upper 的长度是否相同
    if not len(lower) == len(upper):
        raise ValueError('Bounds do not have the same dimensions')

    # 创建目标函数的包装器对象
    func_wrapper = ObjectiveFunWrapper(func, maxfun, *args)

    # 如果 minimizer_kwargs 为 None，则将其置为空字典
    minimizer_kwargs = minimizer_kwargs or {}

    # 创建局部搜索算法的包装器对象
    minimizer_wrapper = LocalSearchWrapper(
        bounds, func_wrapper, *args, **minimizer_kwargs)

    # 如果提供了种子，则创建可重现运行的随机状态对象
    rand_state = check_random_state(seed)

    # 初始化能量状态对象
    energy_state = EnergyState(lower, upper, callback)
    energy_state.reset(func_wrapper, rand_state, x0)

    # 计算重新启动时的温度阈值
    temperature_restart = initial_temp * restart_temp_ratio

    # 创建访问分布的实例
    visit_dist = VisitingDistribution(lower, upper, visit, rand_state)

    # 创建策略链的实例
    strategy_chain = StrategyChain(accept, visit_dist, func_wrapper,
                                   minimizer_wrapper, rand_state, energy_state)

    # 初始化停止标志和迭代次数
    need_to_stop = False
    iteration = 0

    # 初始化消息列表
    message = []

    # 创建用于返回优化结果的 OptimizeResult 对象
    optimize_res = OptimizeResult()
    optimize_res.success = True
    optimize_res.status = 0

    # 计算 t1，用于搜索循环中的计算
    t1 = np.exp((visit - 1) * np.log(2.0)) - 1.0

    # 运行搜索循环
    # 当条件 need_to_stop 为 False 时循环执行以下代码块
    while not need_to_stop:
        # 在指定的最大迭代次数内循环
        for i in range(maxiter):
            # 计算当前迭代步数对应的温度
            s = float(i) + 2.0
            t2 = np.exp((visit - 1) * np.log(s)) - 1.0
            temperature = initial_temp * t1 / t2
            # 如果迭代次数达到最大限制，则记录消息并终止优化
            if iteration >= maxiter:
                message.append("Maximum number of iteration reached")
                need_to_stop = True
                break
            # 如果温度低于重新退火阈值，重设能量状态并中断当前迭代
            if temperature < temperature_restart:
                energy_state.reset(func_wrapper, rand_state)
                break
            # 执行策略链中的运行操作
            val = strategy_chain.run(i, temperature)
            # 如果策略链返回值不为 None，则记录消息并终止优化，标记为失败
            if val is not None:
                message.append(val)
                need_to_stop = True
                optimize_res.success = False
                break
            # 如果允许进行局部搜索，尝试在策略链末尾执行局部搜索
            if not no_local_search:
                val = strategy_chain.local_search()
                # 如果局部搜索返回值不为 None，则记录消息并终止优化，标记为失败
                if val is not None:
                    message.append(val)
                    need_to_stop = True
                    optimize_res.success = False
                    break
            # 迭代次数加一
            iteration += 1

    # 设置 OptimizeResult 结果对象的各个属性值
    optimize_res.x = energy_state.xbest
    optimize_res.fun = energy_state.ebest
    optimize_res.nit = iteration
    optimize_res.nfev = func_wrapper.nfev
    optimize_res.njev = func_wrapper.ngev
    optimize_res.nhev = func_wrapper.nhev
    optimize_res.message = message
    # 返回优化结果对象
    return optimize_res
```