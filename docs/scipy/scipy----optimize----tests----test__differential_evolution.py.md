# `D:\src\scipysrc\scipy\scipy\optimize\tests\test__differential_evolution.py`

```
"""
Unit tests for the differential global minimization algorithm.
"""
# 导入必要的库和模块
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import platform

# 导入 DifferentialEvolutionSolver 类和相关函数
from scipy.optimize._differentialevolution import (DifferentialEvolutionSolver,
                                                   _ConstraintWrapper)
from scipy.optimize import differential_evolution, OptimizeResult
from scipy.optimize._constraints import (Bounds, NonlinearConstraint,
                                         LinearConstraint)
from scipy.optimize import rosen, minimize
from scipy.sparse import csr_matrix
from scipy import stats

# 导入 NumPy 库及其测试模块
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_almost_equal,
                           assert_string_equal, assert_, suppress_warnings)
# 导入 pytest 断言函数和 raises 函数
from pytest import raises as assert_raises, warns
import pytest


class TestDifferentialEvolutionSolver:
    # 设置测试环境的初始化方法
    def setup_method(self):
        # 保存旧的错误处理设置，将无效操作设为抛出异常
        self.old_seterr = np.seterr(invalid='raise')
        # 定义变量 limits 和 bounds，用于测试
        self.limits = np.array([[0., 0.],
                                [2., 2.]])
        self.bounds = [(0., 2.), (0., 2.)]

        # 创建一个 DifferentialEvolutionSolver 的实例对象 dummy_solver，用于测试
        self.dummy_solver = DifferentialEvolutionSolver(self.quadratic,
                                                        [(0, 100)])

        # 创建另一个 DifferentialEvolutionSolver 的实例对象 dummy_solver2，用于测试变异策略
        self.dummy_solver2 = DifferentialEvolutionSolver(self.quadratic,
                                                         [(0, 1)],
                                                         popsize=7,
                                                         mutation=0.5)
        # 创建一个只包含 7 个成员的种群
        # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        population = np.atleast_2d(np.arange(0.1, 0.8, 0.1)).T
        self.dummy_solver2.population = population

    # 测试结束后的清理方法
    def teardown_method(self):
        # 恢复旧的错误处理设置
        np.seterr(**self.old_seterr)

    # 定义一个二次方程的测试函数
    def quadratic(self, x):
        return x[0]**2

    # 测试 _best1 方法的变异策略
    def test__mutate1(self):
        # 策略为 */1/*，例如 rand/1/bin, best/1/exp 等
        result = np.array([0.05])
        trial = self.dummy_solver2._best1(np.array([2, 3, 4, 5, 6]))
        assert_allclose(trial, result)

        result = np.array([0.25])
        trial = self.dummy_solver2._rand1(np.array([2, 3, 4, 5, 6]))
        assert_allclose(trial, result)

    # 测试 _best2 方法的变异策略
    def test__mutate2(self):
        # 策略为 */2/*，例如 rand/2/bin, best/2/exp 等
        # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        result = np.array([-0.1])
        trial = self.dummy_solver2._best2(np.array([2, 3, 4, 5, 6]))
        assert_allclose(trial, result)

        result = np.array([0.1])
        trial = self.dummy_solver2._rand2(np.array([2, 3, 4, 5, 6]))
        assert_allclose(trial, result)

    # 测试 _randtobest1 方法的变异策略
    def test__randtobest1(self):
        # 策略为 randtobest/1/*
        result = np.array([0.15])
        trial = self.dummy_solver2._randtobest1(np.array([2, 3, 4, 5, 6]))
        assert_allclose(trial, result)
    # 测试 _currenttobest1 方法，验证其在给定参数下的输出是否符合预期结果
    def test__currenttobest1(self):
        # 设置测试用例的期望输出结果
        result = np.array([0.1])
        # 调用 _currenttobest1 方法进行计算，传入参数并获取返回值
        trial = self.dummy_solver2._currenttobest1(
            1,
            np.array([2, 3, 4, 5, 6])
        )
        # 使用 assert_allclose 断言方法验证计算结果与期望结果的接近程度
        assert_allclose(trial, result)

    # 测试 DifferentialEvolutionSolver 是否能够使用 dithering 进行初始化
    def test_can_init_with_dithering(self):
        # 设置 dithering 参数作为变异率，并创建 DifferentialEvolutionSolver 实例
        mutation = (0.5, 1)
        solver = DifferentialEvolutionSolver(self.quadratic,
                                             self.bounds,
                                             mutation=mutation)

        # 使用 assert_equal 断言方法验证 solver 实例中的 dithering 参数是否与 mutation 一致
        assert_equal(solver.dither, list(mutation))

    # 测试 DifferentialEvolutionSolver 是否能够拒绝无效的变异率参数
    def test_invalid_mutation_values_arent_accepted(self):
        # 函数选择 rosen
        func = rosen
        # 测试无效变异率参数 (0.5, 3)，期望抛出 ValueError 异常
        mutation = (0.5, 3)
        assert_raises(ValueError,
                          DifferentialEvolutionSolver,
                          func,
                          self.bounds,
                          mutation=mutation)

        # 测试无效变异率参数 (-1, 1)，期望抛出 ValueError 异常
        mutation = (-1, 1)
        assert_raises(ValueError,
                          DifferentialEvolutionSolver,
                          func,
                          self.bounds,
                          mutation=mutation)

        # 测试无效变异率参数 (0.1, np.nan)，期望抛出 ValueError 异常
        mutation = (0.1, np.nan)
        assert_raises(ValueError,
                          DifferentialEvolutionSolver,
                          func,
                          self.bounds,
                          mutation=mutation)

        # 测试有效但单一值的变异率参数 0.5，验证 solver 实例中的 scale 是否为 0.5，dither 是否为 None
        mutation = 0.5
        solver = DifferentialEvolutionSolver(func,
                                             self.bounds,
                                             mutation=mutation)
        assert_equal(0.5, solver.scale)
        assert_equal(None, solver.dither)

    # 测试 func 函数是否符合 differential_evolution 要求的函数签名
    def test_invalid_functional(self):
        # 定义一个测试函数 func，返回一个包含两个值的数组
        def func(x):
            return np.array([np.sum(x ** 2), np.sum(x)])

        # 使用 assert_raises 和 match 参数验证调用 differential_evolution 函数时是否会抛出 RuntimeError 异常
        with assert_raises(
                RuntimeError,
                match=r"func\(x, \*args\) must return a scalar value"):
            differential_evolution(func, [(-2, 2), (-2, 2)])

    # 测试 _scale_parameters 方法是否能正确缩放参数
    def test__scale_parameters(self):
        # 设置测试用例的期望输出结果
        trial = np.array([0.3])
        # 使用 assert_equal 断言方法验证 _scale_parameters 方法的输出是否与期望结果相等
        assert_equal(30, self.dummy_solver._scale_parameters(trial))

        # 验证当参数限制反转时，_scale_parameters 方法是否依然有效
        self.dummy_solver.limits = np.array([[100], [0.]])
        assert_equal(30, self.dummy_solver._scale_parameters(trial))

    # 测试 _unscale_parameters 方法是否能正确反缩放参数
    def test__unscale_parameters(self):
        # 设置测试用例的期望输出结果
        trial = np.array([30])
        # 使用 assert_equal 断言方法验证 _unscale_parameters 方法的输出是否与期望结果相等
        assert_equal(0.3, self.dummy_solver._unscale_parameters(trial))

        # 验证当参数限制反转时，_unscale_parameters 方法是否依然有效
        self.dummy_solver.limits = np.array([[100], [0.]])
        assert_equal(0.3, self.dummy_solver._unscale_parameters(trial))
    def test_equal_bounds(self):
        # 设置 numpy 错误状态，当无效操作发生时抛出异常
        with np.errstate(invalid='raise'):
            # 创建 DifferentialEvolutionSolver 对象，传入目标函数和参数边界
            solver = DifferentialEvolutionSolver(
                self.quadratic,
                bounds=[(2.0, 2.0), (1.0, 3.0)]
            )
            # 调用 _unscale_parameters 方法，将参数反向缩放
            v = solver._unscale_parameters([2.0, 2.0])
            # 使用 assert_allclose 检查结果 v 是否接近于 0.5
            assert_allclose(v, 0.5)

        # 使用 differential_evolution 函数，传入目标函数和参数边界
        res = differential_evolution(self.quadratic, [(2.0, 2.0), (3.0, 3.0)])
        # 使用 assert_equal 检查结果 res.x 是否等于 [2.0, 3.0]
        assert_equal(res.x, [2.0, 3.0])

    def test__ensure_constraint(self):
        # 创建包含无效值的 numpy 数组
        trial = np.array([1.1, -100, 0.9, 2., 300., -0.00001])
        # 调用 _ensure_constraint 方法，确保数组中特定元素的约束条件
        self.dummy_solver._ensure_constraint(trial)

        # 使用 assert_equal 检查 trial[2] 是否等于 0.9
        assert_equal(trial[2], 0.9)
        # 使用 assert_ 检查数组 trial 的所有元素是否在 [0, 1] 范围内
        assert_(np.logical_and(trial >= 0, trial <= 1).all())

    def test_differential_evolution(self):
        # 创建 DifferentialEvolutionSolver 对象，指定目标函数和参数边界
        solver = DifferentialEvolutionSolver(
            self.quadratic, [(-2, 2)], maxiter=1, polish=False
        )
        # 调用 solve 方法求解优化问题
        result = solver.solve()
        # 使用 assert_equal 检查 result.fun 是否等于目标函数在 result.x 处的值
        assert_equal(result.fun, self.quadratic(result.x))

        # 创建另一个 DifferentialEvolutionSolver 对象，指定不同参数
        solver = DifferentialEvolutionSolver(
            self.quadratic, [(-2, 2)], maxiter=1, polish=True
        )
        # 再次调用 solve 方法求解优化问题
        result = solver.solve()
        # 使用 assert_equal 检查 result.fun 是否等于目标函数在 result.x 处的值
        assert_equal(result.fun, self.quadratic(result.x))

    def test_best_solution_retrieval(self):
        # 创建 DifferentialEvolutionSolver 对象，指定目标函数和参数边界
        solver = DifferentialEvolutionSolver(self.quadratic, [(-2, 2)])
        # 调用 solve 方法求解优化问题
        result = solver.solve()
        # 使用 assert_equal 检查 result.x 是否等于 solver.x
        assert_equal(result.x, solver.x)

    def test_callback_terminates(self):
        # 定义参数边界
        bounds = [(0, 2), (0, 2)]
        # 预期的消息内容
        expected_msg = 'callback function requested stop early'

        # 定义返回 True 的回调函数
        def callback_python_true(param, convergence=0.):
            return True

        # 使用 differential_evolution 函数，传入目标函数、参数边界和回调函数
        result = differential_evolution(
            rosen, bounds, callback=callback_python_true
        )
        # 使用 assert_string_equal 检查 result.message 是否等于 expected_msg
        assert_string_equal(result.message, expected_msg)

        # 定义抛出 StopIteration 异常的回调函数
        def callback_stop(intermediate_result):
            raise StopIteration

        # 使用 differential_evolution 函数，传入目标函数、参数边界和抛出异常的回调函数
        result = differential_evolution(rosen, bounds, callback=callback_stop)
        # 使用 assert 检查 result.success 是否为 False
        assert not result.success

        # 定义返回 [10] 的回调函数
        def callback_evaluates_true(param, convergence=0.):
            # DE 应该在 bool(self.callback) 为 True 时停止
            return [10]

        # 使用 differential_evolution 函数，传入目标函数、参数边界和回调函数
        result = differential_evolution(rosen, bounds, callback=callback_evaluates_true)
        # 使用 assert_string_equal 检查 result.message 是否等于 expected_msg
        assert_string_equal(result.message, expected_msg)
        # 使用 assert 检查 result.success 是否为 False
        assert not result.success

        # 定义返回空列表的回调函数
        def callback_evaluates_false(param, convergence=0.):
            return []

        # 使用 differential_evolution 函数，传入目标函数、参数边界和回调函数
        result = differential_evolution(rosen, bounds,
                                        callback=callback_evaluates_false)
        # 使用 assert 检查 result.success 是否为 True
        assert result.success
    def test_args_tuple_is_passed(self):
        # 测试参数元组是否正确传递给成本函数。
        bounds = [(-10, 10)]
        args = (1., 2., 3.)

        def quadratic(x, *args):
            if type(args) != tuple:
                raise ValueError('args should be a tuple')
            return args[0] + args[1] * x + args[2] * x**2.

        # 使用差分进化算法求解二次函数的最小值
        result = differential_evolution(quadratic,
                                        bounds,
                                        args=args,
                                        polish=True)
        assert_almost_equal(result.fun, 2 / 3.)

    def test_init_with_invalid_strategy(self):
        # 测试传递无效策略时是否引发 ValueError 异常
        func = rosen
        bounds = [(-3, 3)]
        assert_raises(ValueError,
                          differential_evolution,
                          func,
                          bounds,
                          strategy='abc')

    def test_bounds_checking(self):
        # 测试边界检查功能是否正常工作
        func = rosen
        bounds = [(-3)]
        assert_raises(ValueError,
                          differential_evolution,
                          func,
                          bounds)
        bounds = [(-3, 3), (3, 4, 5)]
        assert_raises(ValueError,
                          differential_evolution,
                          func,
                          bounds)

        # 测试是否可以使用新类型 Bounds 对象
        result = differential_evolution(rosen, Bounds([0, 0], [2, 2]))
        assert_almost_equal(result.x, (1., 1.))

    def test_select_samples(self):
        # select_samples 应该返回 5 个不同的随机数。
        limits = np.arange(12., dtype='float64').reshape(2, 6)
        bounds = list(zip(limits[0, :], limits[1, :]))
        solver = DifferentialEvolutionSolver(None, bounds, popsize=1)
        candidate = 0
        r1, r2, r3, r4, r5 = solver._select_samples(candidate, 5)
        assert_equal(
            len(np.unique(np.array([candidate, r1, r2, r3, r4, r5]))), 6)

    def test_maxiter_stops_solve(self):
        # 测试如果超过最大迭代次数，求解器是否会停止。
        solver = DifferentialEvolutionSolver(rosen, self.bounds, maxiter=1)
        result = solver.solve()
        assert_equal(result.success, False)
        assert_equal(result.message,
                        'Maximum number of iterations has been exceeded.')
    def test_maxfun_stops_solve(self):
        # 测试当初始过程中超过最大函数评估次数时求解器停止的情况

        # 创建一个 DifferentialEvolutionSolver 对象，使用 Rosenbrock 函数作为目标函数，
        # 给定的边界 self.bounds，设置最大函数评估次数为1，不进行后处理
        solver = DifferentialEvolutionSolver(rosen, self.bounds, maxfun=1,
                                             polish=False)
        # 执行求解器的求解方法
        result = solver.solve()

        # 断言求解结果的函数评估次数为2
        assert_equal(result.nfev, 2)
        # 断言求解是否成功为 False
        assert_equal(result.success, False)
        # 断言消息为“超过最大函数评估次数”
        assert_equal(result.message,
                     'Maximum number of function evaluations has '
                     'been exceeded.')

        # 测试在实际最小化过程中如果超过最大函数评估次数，则求解器停止的情况
        # 必须关闭后处理，因为即使达到 maxfun，后处理仍然会发生
        # 对于 popsize=5 和 len(bounds)=2 的情况下，初始过程中只有10个函数评估
        solver = DifferentialEvolutionSolver(rosen,
                                             self.bounds,
                                             popsize=5,
                                             polish=False,
                                             maxfun=40)
        result = solver.solve()

        # 断言求解结果的函数评估次数为41
        assert_equal(result.nfev, 41)
        # 断言求解是否成功为 False
        assert_equal(result.success, False)
        # 断言消息为“超过最大函数评估次数”
        assert_equal(result.message,
                     'Maximum number of function evaluations has '
                     'been exceeded.')

        # 现在针对 updating='deferred' 的情况重复上述步骤
        # 47 个函数评估不是种群大小的倍数，因此 maxfun 在种群评估的过程中途达到
        solver = DifferentialEvolutionSolver(rosen,
                                             self.bounds,
                                             popsize=5,
                                             polish=False,
                                             maxfun=47,
                                             updating='deferred')
        result = solver.solve()

        # 断言求解结果的函数评估次数为47
        assert_equal(result.nfev, 47)
        # 断言求解是否成功为 False
        assert_equal(result.success, False)
        # 断言消息为“已达到最大函数评估次数”
        assert_equal(result.message,
                     'Maximum number of function evaluations has '
                     'been reached.')
    def test_seed_gives_repeatability(self):
        # 测试种子参数在差分进化算法中确保结果的重复性
        result = differential_evolution(self.quadratic,
                                        [(-100, 100)],
                                        polish=False,
                                        seed=1,
                                        tol=0.5)
        result2 = differential_evolution(self.quadratic,
                                        [(-100, 100)],
                                        polish=False,
                                        seed=1,
                                        tol=0.5)
        # 断言两次运行的结果向量 x 相同
        assert_equal(result.x, result2.x)
        # 断言两次运行的评估次数 nfev 相同
        assert_equal(result.nfev, result2.nfev)

    def test_random_generator(self):
        # 检查是否可以使用 np.random.Generator（要求 numpy >= 1.17）
        # 获取一个 np.random.Generator 对象
        rng = np.random.default_rng()

        # 初始化方式列表
        inits = ['random', 'latinhypercube', 'sobol', 'halton']
        for init in inits:
            # 使用不同的初始化方式运行差分进化算法
            differential_evolution(self.quadratic,
                                   [(-100, 100)],
                                   polish=False,
                                   seed=rng,
                                   tol=0.5,
                                   init=init)

    def test_exp_runs(self):
        # 测试指数变异策略是否正常运行
        solver = DifferentialEvolutionSolver(rosen,
                                             self.bounds,
                                             strategy='best1exp',
                                             maxiter=1)

        # 执行求解过程
        solver.solve()

    def test_gh_4511_regression(self):
        # 这个测试修复了差分进化文档示例中的问题，使用了一个定制的 popsize 触发了一个偏差
        # 我们不关心在这个测试中解决优化问题，因此使用 maxiter=1 来减少测试时间
        bounds = [(-5, 5), (-5, 5)]
        # 使用 popsize=49 运行差分进化算法，修复问题
        differential_evolution(rosen, bounds, popsize=49, maxiter=1)

    def test_calculate_population_energies(self):
        # 如果 popsize 为 3，那么整个种群大小为 (6,)
        solver = DifferentialEvolutionSolver(rosen, self.bounds, popsize=3)
        # 计算种群的能量值
        solver._calculate_population_energies(solver.population)
        # 提升能量最低的个体
        solver._promote_lowest_energy()
        # 断言能量值最低的个体索引为 0
        assert_equal(np.argmin(solver.population_energies), 0)

        # 初始能量值计算应该需要 6 次 nfev
        assert_equal(solver._nfev, 6)
    # 测试 DifferentialEvolutionSolver 是否可迭代
    # 如果 popsize 为 3，则整体生成的大小为 (6,)
    solver = DifferentialEvolutionSolver(rosen, self.bounds, popsize=3,
                                         maxfun=12)
    # 获取迭代器的下一个元素，返回解 x 和目标函数值 fun
    x, fun = next(solver)
    # 断言 x 的第一维大小为 2
    assert_equal(np.size(x, 0), 2)

    # 初始计算能量需要 6 次函数评估（nfev），演化 6 个种群成员也需要 6 次函数评估
    assert_equal(solver._nfev, 12)

    # 下一代应停止，因为超过了 maxfun
    assert_raises(StopIteration, next, solver)

    # 检查迭代求解器能否进行适当的最小化
    solver = DifferentialEvolutionSolver(rosen, self.bounds)
    # 获取下一个解和目标函数值
    _, fun_prev = next(solver)
    for i, soln in enumerate(solver):
        # 获取当前解和目标函数值
        x_current, fun_current = soln
        # 断言前一个目标函数值大于等于当前目标函数值，确保是在最小化
        assert fun_prev >= fun_current
        # 更新前一个解和目标函数值
        _, fun_prev = x_current, fun_current
        # 如果达到 50 次迭代，则跳出循环
        if i == 50:
            break

# 测试收敛性
solver = DifferentialEvolutionSolver(rosen, self.bounds, tol=0.2,
                                     polish=False)
# 运行求解方法
solver.solve()
# 断言收敛性小于 0.2
assert_(solver.convergence < 0.2)

# 测试 maxiter 为 None 的情况（Github Issue #5731）
# 在版本 0.17 之前，maxiter 和 maxfun 的默认值为 None。
# 数值上的默认值现在分别是 1000 和 np.inf。然而，一些脚本仍然会为这两个参数提供 None，
# 这会在 solve 方法中引发 TypeError。
solver = DifferentialEvolutionSolver(rosen, self.bounds, maxiter=None,
                                     maxfun=None)
# 运行求解方法
solver.solve()
    def test_population_initiation(self):
        # 测试种群初始化的不同模式

        # 初始化必须是'latinhypercube'或'random'
        # 如果传入其他内容则引发 ValueError
        assert_raises(ValueError,
                      DifferentialEvolutionSolver,
                      *(rosen, self.bounds),
                      **{'init': 'rubbish'})

        solver = DifferentialEvolutionSolver(rosen, self.bounds)

        # 检查种群初始化：
        # 1) 将 _nfev 重置为 0
        # 2) 所有种群的能量值设为 np.inf
        solver.init_population_random()
        assert_equal(solver._nfev, 0)
        assert_(np.all(np.isinf(solver.population_energies)))

        solver.init_population_lhs()
        assert_equal(solver._nfev, 0)
        assert_(np.all(np.isinf(solver.population_energies)))

        solver.init_population_qmc(qmc_engine='halton')
        assert_equal(solver._nfev, 0)
        assert_(np.all(np.isinf(solver.population_energies)))

        solver = DifferentialEvolutionSolver(rosen, self.bounds, init='sobol')
        solver.init_population_qmc(qmc_engine='sobol')
        assert_equal(solver._nfev, 0)
        assert_(np.all(np.isinf(solver.population_energies)))

        # 可以用自定义数组进行初始化
        population = np.linspace(-1, 3, 10).reshape(5, 2)
        solver = DifferentialEvolutionSolver(rosen, self.bounds,
                                             init=population,
                                             strategy='best2bin',
                                             atol=0.01, seed=1, popsize=5)

        assert_equal(solver._nfev, 0)
        assert_(np.all(np.isinf(solver.population_energies)))
        assert_(solver.num_population_members == 5)
        assert_(solver.population_shape == (5, 2))

        # 检查种群是否正确初始化
        unscaled_population = np.clip(solver._unscale_parameters(population),
                                      0, 1)
        assert_almost_equal(solver.population[:5], unscaled_population)

        # 种群值必须被裁剪到定义的边界范围内
        assert_almost_equal(np.min(solver.population[:5]), 0)
        assert_almost_equal(np.max(solver.population[:5]), 1)

        # 如果数组形状不正确，则不能用数组进行初始化
        # 这个例子会有过多的参数
        population = np.linspace(-1, 3, 15).reshape(5, 3)
        assert_raises(ValueError,
                      DifferentialEvolutionSolver,
                      *(rosen, self.bounds),
                      **{'init': population})

        # 提供初始解决方案
        # 边界是 [(0, 2), (0, 2)]
        x0 = np.random.uniform(low=0.0, high=2.0, size=2)
        solver = DifferentialEvolutionSolver(
            rosen, self.bounds, x0=x0
        )
        # 参数被缩放到单位区间
        assert_allclose(solver.population[0], x0 / 2.0)
    def test_x0(self):
        # smoke test that checks that x0 is usable.
        # 进行简单的测试以确保 x0 可用。
        res = differential_evolution(rosen, self.bounds, x0=[0.2, 0.8])
        # 断言优化成功
        assert res.success

        # check what happens if some of the x0 lay outside the bounds
        # 检查如果一些 x0 超出边界会发生什么
        with assert_raises(ValueError):
            differential_evolution(rosen, self.bounds, x0=[0.2, 2.1])

    def test_infinite_objective_function(self):
        # Test that there are no problems if the objective function
        # returns inf on some runs
        # 测试如果目标函数在某些运行中返回 inf 是否会出现问题
        def sometimes_inf(x):
            if x[0] < .5:
                return np.inf
            return x[1]
        bounds = [(0, 1), (0, 1)]
        differential_evolution(sometimes_inf, bounds=bounds, disp=False)

    def test_deferred_updating(self):
        # check setting of deferred updating, with default workers
        # 检查使用延迟更新设置，默认使用工作进程
        bounds = [(0., 2.), (0., 2.)]
        solver = DifferentialEvolutionSolver(rosen, bounds, updating='deferred')
        assert_(solver._updating == 'deferred')
        assert_(solver._mapwrapper._mapfunc is map)
        res = solver.solve()
        assert res.success

        # check that deferred updating works with an exponential crossover
        # 检查延迟更新与指数交叉操作一起工作的情况
        res = differential_evolution(
            rosen, bounds, updating='deferred', strategy='best1exp'
        )
        assert res.success

    def test_immediate_updating(self):
        # check setting of immediate updating, with default workers
        # 检查使用即时更新设置，默认使用工作进程
        bounds = [(0., 2.), (0., 2.)]
        solver = DifferentialEvolutionSolver(rosen, bounds)
        assert_(solver._updating == 'immediate')

        # Safely forking from a multithreaded process is
        # problematic, and deprecated in Python 3.12, so
        # we use a slower but portable alternative
        # see gh-19848
        # 从多线程进程中安全分叉是有问题的，并且在 Python 3.12 中已弃用，因此我们使用更慢但可移植的替代方案
        # 参见 gh-19848
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(2) as p:
            # should raise a UserWarning because the updating='immediate'
            # is being overridden by the workers keyword
            # 应该引发 UserWarning，因为 workers 关键字覆盖了 updating='immediate'
            with warns(UserWarning):
                with DifferentialEvolutionSolver(rosen, bounds, workers=p.map) as s:
                    pass
            assert s._updating == 'deferred'

    @pytest.mark.fail_slow(10)
    def test_parallel(self):
        # smoke test for parallelization with deferred updating
        # 使用延迟更新进行并行化的简单测试
        bounds = [(0., 2.), (0., 2.)]
        # use threads instead of Process to speed things up for this simple example
        # 使用线程而不是进程来加快这个简单示例的速度
        with ThreadPool(2) as p, DifferentialEvolutionSolver(
            rosen, bounds, updating='deferred', workers=p.map, tol=0.1, popsize=3
        ) as solver:
            assert solver._mapwrapper.pool is not None
            assert solver._updating == 'deferred'
            solver.solve()

        with DifferentialEvolutionSolver(
            rosen, bounds, updating='deferred', workers=2, popsize=3, tol=0.1
        ) as solver:
            assert solver._mapwrapper.pool is not None
            assert solver._updating == 'deferred'
            solver.solve()
    # 测试函数，验证 DifferentialEvolutionSolver 类的收敛性检查功能
    def test_converged(self):
        # 创建 DifferentialEvolutionSolver 实例，传入 Rosenbrock 函数和变量范围
        solver = DifferentialEvolutionSolver(rosen, [(0, 2), (0, 2)])
        # 执行求解过程
        solver.solve()
        # 断言求解器已经收敛
        assert_(solver.converged())

    # 测试约束违反函数
    def test_constraint_violation_fn(self):
        # 定义约束函数 constr_f，返回约束值的列表
        def constr_f(x):
            return [x[0] + x[1]]

        # 定义第二个约束函数 constr_f2，返回约束值的数组
        def constr_f2(x):
            return np.array([x[0]**2 + x[1], x[0] - x[1]])

        # 创建 NonlinearConstraint 实例 nlc，约束函数为 constr_f，约束范围为负无穷到 1.9
        nlc = NonlinearConstraint(constr_f, -np.inf, 1.9)

        # 创建 DifferentialEvolutionSolver 实例 solver，传入 Rosenbrock 函数和变量范围，
        # 并指定约束为 nlc
        solver = DifferentialEvolutionSolver(rosen, [(0, 2), (0, 2)],
                                             constraints=(nlc,))

        # 测试 _constraint_violation_fn 方法，检查约束违反情况
        cv = solver._constraint_violation_fn(np.array([1.0, 1.0]))
        assert_almost_equal(cv, 0.1)

        # 创建第二个 NonlinearConstraint 实例 nlc2，约束函数为 constr_f2，约束范围为负无穷到 1.8
        nlc2 = NonlinearConstraint(constr_f2, -np.inf, 1.8)

        # 创建 DifferentialEvolutionSolver 实例 solver，传入 Rosenbrock 函数和变量范围，
        # 并指定约束为 nlc 和 nlc2
        solver = DifferentialEvolutionSolver(rosen, [(0, 2), (0, 2)],
                                             constraints=(nlc, nlc2))

        # 对于多个约束，约束违反应该被连接起来
        xs = [(1.2, 1), (2.0, 2.0), (0.5, 0.5)]
        vs = [(0.3, 0.64, 0.0), (2.1, 4.2, 0.0), (0, 0, 0)]

        # 遍历测试不同的输入，检查约束违反情况
        for x, v in zip(xs, vs):
            cv = solver._constraint_violation_fn(np.array(x))
            assert_allclose(cv, np.atleast_2d(v))

        # 使用向量化计算一系列解的约束违反情况
        assert_allclose(
            solver._constraint_violation_fn(np.array(xs)), np.array(vs)
        )

        # 下面的代码行用于 _calculate_population_feasibilities 方法。
        # 当 x.shape == (N,) 时，_constraint_violation_fn 返回一个 (1, M) 的数组。
        # 因此，这个列表推导应该生成一个 (S, 1, M) 的数组。
        constraint_violation = np.array([solver._constraint_violation_fn(x)
                                         for x in np.array(xs)])
        assert constraint_violation.shape == (3, 1, 3)

        # 如果约束函数没有返回正确的结果，我们需要合理的错误消息
        def constr_f3(x):
            # 返回 (S, M) 形状，而不是 (M, S)
            return constr_f2(x).T

        # 创建第三个 NonlinearConstraint 实例 nlc2，约束函数为 constr_f3，约束范围为负无穷到 1.8
        nlc2 = NonlinearConstraint(constr_f3, -np.inf, 1.8)

        # 创建 DifferentialEvolutionSolver 实例 solver，传入 Rosenbrock 函数和变量范围，
        # 并指定约束为 nlc 和 nlc2，并设置 vectorized=False
        solver = DifferentialEvolutionSolver(rosen, [(0, 2), (0, 2)],
                                             constraints=(nlc, nlc2),
                                             vectorized=False)
        # 将向量化设置为 True
        solver.vectorized = True

        # 使用 pytest 检查是否抛出预期的 RuntimeError 异常，匹配特定的错误消息
        with pytest.raises(
                RuntimeError, match="An array returned from a Constraint"
        ):
            solver._constraint_violation_fn(np.array(xs))
    # 定义一个测试函数，用于测试约束条件下种群的可行性
    def test_constraint_population_feasibilities(self):
        # 定义一个约束函数，返回一个列表，表示约束条件
        def constr_f(x):
            return [x[0] + x[1]]

        # 定义另一个约束函数，返回一个列表，表示多个约束条件
        def constr_f2(x):
            return [x[0]**2 + x[1], x[0] - x[1]]

        # 创建一个非线性约束对象，约束条件为 x[0] + x[1] <= 1.9
        nlc = NonlinearConstraint(constr_f, -np.inf, 1.9)

        # 创建一个差分进化求解器对象，优化目标函数为 rosen，变量范围为 [(0, 2), (0, 2)]
        # 设置约束条件为 nlc
        solver = DifferentialEvolutionSolver(rosen, [(0, 2), (0, 2)],
                                             constraints=(nlc,))

        # 测试种群的可行性是否正确
        # 输入种群数据 np.array([[0.5, 0.5], [1., 1.]])
        feas, cv = solver._calculate_population_feasibilities(
            np.array([[0.5, 0.5], [1., 1.]]))
        # 检查是否符合期望的可行性结果
        assert_equal(feas, [False, False])
        # 检查约束违反程度的近似值是否正确
        assert_almost_equal(cv, np.array([[0.1], [2.1]]))
        # 检查 cv 的形状是否为 (2, 1)
        assert cv.shape == (2, 1)

        # 创建另一个非线性约束对象，约束条件为 x[0]**2 + x[1] <= 1.8 和 x[0] - x[1] <= 0
        nlc2 = NonlinearConstraint(constr_f2, -np.inf, 1.8)

        # 对于 vectorize 参数为 False 和 True 分别进行测试
        for vectorize in [False, True]:
            # 创建差分进化求解器对象，优化目标函数为 rosen，变量范围为 [(0, 2), (0, 2)]
            # 设置约束条件为 nlc 和 nlc2，vectorized 参数为 vectorize，updating 参数为 'deferred'
            solver = DifferentialEvolutionSolver(rosen, [(0, 2), (0, 2)],
                                                 constraints=(nlc, nlc2),
                                                 vectorized=vectorize,
                                                 updating='deferred')

            # 测试种群的可行性是否正确
            # 输入种群数据 np.array([[0.5, 0.5], [0.6, 0.5]])
            feas, cv = solver._calculate_population_feasibilities(
                np.array([[0.5, 0.5], [0.6, 0.5]]))
            # 检查是否符合期望的可行性结果
            assert_equal(feas, [False, False])
            # 检查约束违反程度的近似值是否正确
            assert_almost_equal(cv, np.array([[0.1, 0.2, 0], [0.3, 0.64, 0]]))

            # 测试种群的可行性是否正确
            # 输入种群数据 np.array([[0.5, 0.5], [1., 1.]])
            feas, cv = solver._calculate_population_feasibilities(
                np.array([[0.5, 0.5], [1., 1.]]))
            # 检查是否符合期望的可行性结果
            assert_equal(feas, [False, False])
            # 检查约束违反程度的近似值是否正确
            assert_almost_equal(cv, np.array([[0.1, 0.2, 0], [2.1, 4.2, 0]]))
            # 检查 cv 的形状是否为 (2, 3)
            assert cv.shape == (2, 3)

            # 测试种群的可行性是否正确
            # 输入种群数据 np.array([[0.25, 0.25], [1., 1.]])
            feas, cv = solver._calculate_population_feasibilities(
                np.array([[0.25, 0.25], [1., 1.]]))
            # 检查是否符合期望的可行性结果
            assert_equal(feas, [True, False])
            # 检查约束违反程度的近似值是否正确
            assert_almost_equal(cv, np.array([[0.0, 0.0, 0.], [2.1, 4.2, 0]]))
            # 检查 cv 的形状是否为 (2, 3)

    # 定义一个测试函数，用于测试求解器在约束条件下的解是否符合预期
    def test_constraint_solve(self):
        # 定义一个约束函数，返回一个数组，表示约束条件
        def constr_f(x):
            return np.array([x[0] + x[1]])

        # 创建一个非线性约束对象，约束条件为 x[0] + x[1] <= 1.9
        nlc = NonlinearConstraint(constr_f, -np.inf, 1.9)

        # 创建一个差分进化求解器对象，优化目标函数为 rosen，变量范围为 [(0, 2), (0, 2)]
        # 设置约束条件为 nlc
        solver = DifferentialEvolutionSolver(rosen, [(0, 2), (0, 2)],
                                             constraints=(nlc,))

        # 使用 solver.solve() 求解，并期望出现 UserWarning
        with warns(UserWarning):
            res = solver.solve()

        # 检查求解结果是否满足约束条件 constr_f(x) <= 1.9
        assert constr_f(res.x) <= 1.9
        # 检查求解是否成功
        assert res.success

    # 标记该测试为 fail_slow，允许其运行时间较长（最多 10 秒）
    @pytest.mark.fail_slow(10)
    def test_impossible_constraint(self):
        # 定义一个约束函数，返回 x[0] + x[1] 的数组表示
        def constr_f(x):
            return np.array([x[0] + x[1]])

        # 创建一个非线性约束对象，约束条件为 x[0] + x[1] >= -1
        nlc = NonlinearConstraint(constr_f, -np.inf, -1)

        # 创建一个差分进化求解器对象，使用 Rosenbrock 函数作为目标函数，
        # 变量范围为 [(0, 2), (0, 2)]，设置了非线性约束条件
        solver = DifferentialEvolutionSolver(
            rosen, [(0, 2), (0, 2)], constraints=(nlc,), popsize=1, seed=1, maxiter=100
        )

        # 由于在找到的最不可行解上尝试了 'trust-constr' 精磨，因此发出 UserWarning 警告。
        with warns(UserWarning):
            # 调用求解器的 solve 方法
            res = solver.solve()

        # 断言：返回结果中最大约束违反度大于 0
        assert res.maxcv > 0
        # 断言：求解未成功
        assert not res.success

        # 测试当种群中没有可行解时，_promote_lowest_energy 方法是否有效。
        # 在这种情况下，应该提升最低约束违反的解。
        solver = DifferentialEvolutionSolver(
            rosen, [(0, 2), (0, 2)], constraints=(nlc,), polish=False)
        next(solver)
        # 断言：种群中不存在全部可行解
        assert not solver.feasible.all()
        # 断言：种群能量中并非全部都是有限的
        assert not np.isfinite(solver.population_energies).all()

        # 交换种群中两个条目的位置
        l = 20
        cv = solver.constraint_violation[0]

        solver.population_energies[[0, l]] = solver.population_energies[[l, 0]]
        solver.population[[0, l], :] = solver.population[[l, 0], :]
        solver.constraint_violation[[0, l], :] = (
            solver.constraint_violation[[l, 0], :])

        # 调用 _promote_lowest_energy 方法
        solver._promote_lowest_energy()
        # 断言：第一个约束违反度与原始约束违反度相同
        assert_equal(solver.constraint_violation[0], cv)

    def test_accept_trial(self):
        # _accept_trial(self, energy_trial, feasible_trial, cv_trial,
        #               energy_orig, feasible_orig, cv_orig)
        # 定义一个约束函数，返回 x[0] + x[1] 的列表表示
        def constr_f(x):
            return [x[0] + x[1]]
        
        # 创建一个差分进化求解器对象，使用 Rosenbrock 函数作为目标函数，
        # 变量范围为 [(0, 2), (0, 2)]，设置了非线性约束条件
        nlc = NonlinearConstraint(constr_f, -np.inf, 1.9)
        solver = DifferentialEvolutionSolver(rosen, [(0, 2), (0, 2)],
                                             constraints=(nlc,))
        # 获取 _accept_trial 方法的引用
        fn = solver._accept_trial
        
        # 断言：当两个解都是可行的时候，选择能量更低的解
        assert fn(0.1, True, np.array([0.]), 1.0, True, np.array([0.]))
        # 断言：当 trial 解可行而 original 解不可行时，返回 False
        assert not fn(1.0, True, np.array([0.0]), 0.1, True, np.array([0.0]))
        # 断言：当两个解都是可行的且能量相同时，返回 True
        assert fn(0.1, True, np.array([0.]), 0.1, True, np.array([0.]))

        # 断言：当 trial 解可行而 original 解不可行时，返回 True
        assert fn(9.9, True, np.array([0.]), 1.0, False, np.array([1.]))

        # 断言：当 trial 和 original 解都不可行时，cv_trial 必须 <= cv_original 才会返回 True
        assert fn(0.1, False, np.array([0.5, 0.5]),
                   1.0, False, np.array([1., 1.0]))
        assert fn(0.1, False, np.array([0.5, 0.5]),
                   1.0, False, np.array([1., 0.50]))
        assert not fn(1.0, False, np.array([0.5, 0.5]),
                       1.0, False, np.array([1.0, 0.4]))
    # 定义一个测试函数，用于测试约束包装器的功能
    def test_constraint_wrapper(self):
        # 定义下界数组
        lb = np.array([0, 20, 30])
        # 定义上界数组，其中包含正无穷值
        ub = np.array([0.5, np.inf, 70])
        # 定义初始点数组
        x0 = np.array([1, 2, 3])
        # 创建 Bounds 类型的约束包装器对象
        pc = _ConstraintWrapper(Bounds(lb, ub), x0)
        # 断言：初始点 x0 存在违规约束
        assert (pc.violation(x0) > 0).any()
        # 断言：特定点 [0.25, 21, 31] 没有违规约束
        assert (pc.violation([0.25, 21, 31]) == 0).all()

        # 检查矢量化的 Bounds 约束
        # 创建一个形状为 (5, 3) 的二维数组 xs
        xs = np.arange(1, 16).reshape(5, 3)
        violations = []
        # 遍历 xs 中的每一个点 x，并计算其违规约束
        for x in xs:
            violations.append(pc.violation(x))
        # 断言：矩阵 xs 转置后的违规约束与 violations 数组的转置结果非常接近
        np.testing.assert_allclose(pc.violation(xs.T), np.array(violations).T)

        # 修改初始点数组 x0
        x0 = np.array([1, 2, 3, 4])
        # 定义一个包含多个线性约束的系数矩阵 A
        A = np.array([[1, 2, 3, 4], [5, 0, 0, 6], [7, 0, 8, 0]])
        # 创建 LinearConstraint 类型的约束包装器对象
        pc = _ConstraintWrapper(LinearConstraint(A, -np.inf, 0), x0)
        # 断言：初始点 x0 存在违规约束
        assert (pc.violation(x0) > 0).any()
        # 断言：特定点 [-10, 2, -10, 4] 没有违规约束
        assert (pc.violation([-10, 2, -10, 4]) == 0).all()

        # 检查矢量化的 LinearConstraint，适用于 7 个参数向量，每个长度为 4，有 3 个约束
        xs = np.arange(1, 29).reshape(7, 4)
        violations = []
        # 遍历 xs 中的每一个点 x，并计算其违规约束
        for x in xs:
            violations.append(pc.violation(x))
        # 断言：矩阵 xs 转置后的违规约束与 violations 数组的转置结果非常接近
        np.testing.assert_allclose(pc.violation(xs.T), np.array(violations).T)

        # 创建 LinearConstraint 类型的约束包装器对象，系数矩阵 A 转换为稀疏矩阵
        pc = _ConstraintWrapper(LinearConstraint(csr_matrix(A), -np.inf, 0),
                                x0)
        # 断言：初始点 x0 存在违规约束
        assert (pc.violation(x0) > 0).any()
        # 断言：特定点 [-10, 2, -10, 4] 没有违规约束
        assert (pc.violation([-10, 2, -10, 4]) == 0).all()

        # 定义一个非线性约束函数 fun(x)，返回 A 矩阵与 x 的乘积
        def fun(x):
            return A.dot(x)

        # 创建 NonlinearConstraint 类型的约束包装器对象
        nonlinear = NonlinearConstraint(fun, -np.inf, 0)
        pc = _ConstraintWrapper(nonlinear, [-10, 2, -10, 4])
        # 断言：初始点 x0 存在违规约束
        assert (pc.violation(x0) > 0).any()
        # 断言：特定点 [-10, 2, -10, 4] 没有违规约束
        assert (pc.violation([-10, 2, -10, 4]) == 0).all()
    def test_constraint_wrapper_violation(self):
        def cons_f(x):
            # 定义约束函数cons_f，接受一个数组x，返回一个形状为(M, S)的numpy数组，
            # 其中N是参数数量，S是要检查的解向量数量，M是约束分量的数量
            return np.array([x[0] ** 2 + x[1],
                             x[0] ** 2 - x[1]])

        # 创建非线性约束对象nlc，使用cons_f作为约束函数，约束范围为[-1, -0.85]到[2, 2]
        nlc = NonlinearConstraint(cons_f, [-1, -0.8500], [2, 2])
        
        # 创建约束包装对象pc，使用nlc作为约束条件，初始点为[0.5, 1]
        pc = _ConstraintWrapper(nlc, [0.5, 1])
        
        # 断言pc.bounds[0]的大小为2
        assert np.size(pc.bounds[0]) == 2

        # 定义测试数据集xs和对应的期望结果集vs
        xs = [(0.5, 1), (0.5, 1.2), (1.2, 1.2), (0.1, -1.2), (0.1, 2.0)]
        vs = [(0, 0), (0, 0.1), (0.64, 0), (0.19, 0), (0.01, 1.14)]

        # 对于xs和vs中的每对数据，验证pc.violation(x)的输出是否接近v
        for x, v in zip(xs, vs):
            assert_allclose(pc.violation(x), v)

        # 验证pc.violation对输入数组xs进行向量化后的结果是否接近数组vs的转置
        assert_allclose(pc.violation(np.array(xs).T),
                        np.array(vs).T)
        
        # 验证pc.fun对输入数组xs进行向量化后的输出形状是否为(2, len(xs))
        assert pc.fun(np.array(xs).T).shape == (2, len(xs))
        
        # 验证pc.violation对输入数组xs进行向量化后的输出形状是否为(2, len(xs))
        assert pc.violation(np.array(xs).T).shape == (2, len(xs))
        
        # 断言pc.num_constr为2，即约束的数量
        assert pc.num_constr == 2
        
        # 断言pc.parameter_count为2，即参数的数量
        assert pc.parameter_count == 2

    def test_matrix_linear_constraint(self):
        # 修复gh20041：使用np.matrix构造LinearConstraint导致_ConstraintWrapper返回错误形状的约束违规
        with suppress_warnings() as sup:
            sup.filter(PendingDeprecationWarning)
            # 创建一个2x4的np.matrix对象matrix
            matrix = np.matrix([[1, 1, 1, 1.],
                                [2, 2, 2, 2.]])
        
        # 创建线性约束对象lc，使用matrix作为约束矩阵，约束范围为0到1
        lc = LinearConstraint(matrix, 0, 1)
        
        # 创建约束包装对象cw，使用lc作为约束条件，初始点为全1的数组x0
        x0 = np.ones(4)
        cw = _ConstraintWrapper(lc, x0)
        
        # 断言cw.violation(x0)的输出形状为(2,)
        assert cw.violation(x0).shape == (2,)

        # 创建一个4x5的数组xtrial，验证cw.violation对其进行向量化后的输出形状为(2, 5)
        xtrial = np.arange(4 * 5).reshape(4, 5)
        assert cw.violation(xtrial).shape == (2, 5)
    def test_L2(self):
        # Lampinen ([5]) test problem 2

        def f(x):
            x = np.hstack(([0], x))  # 将 x 前添加一个 0，使索引从 1 开始，以匹配参考文献
            fun = ((x[1]-10)**2 + 5*(x[2]-12)**2 + x[3]**4 + 3*(x[4]-11)**2 +
                   10*x[5]**6 + 7*x[6]**2 + x[7]**4 - 4*x[6]*x[7] - 10*x[6] -
                   8*x[7])
            return fun

        def c1(x):
            x = np.hstack(([0], x))  # 将 x 前添加一个 0，使索引从 1 开始，以匹配参考文献
            return [127 - 2*x[1]**2 - 3*x[2]**4 - x[3] - 4*x[4]**2 - 5*x[5],
                    196 - 23*x[1] - x[2]**2 - 6*x[6]**2 + 8*x[7],
                    282 - 7*x[1] - 3*x[2] - 10*x[3]**2 - x[4] + x[5],
                    -4*x[1]**2 - x[2]**2 + 3*x[1]*x[2] - 2*x[3]**2 -
                    5*x[6] + 11*x[7]]

        # 定义非线性约束条件 N
        N = NonlinearConstraint(c1, 0, np.inf)
        # 定义变量的上下界
        bounds = [(-10, 10)]*7
        constraints = (N)

        # 忽略警告输出
        with suppress_warnings() as sup:
            sup.filter(UserWarning)
            # 使用差分进化算法求解优化问题
            res = differential_evolution(f, bounds, strategy='best1bin',
                                         seed=1234, constraints=constraints)

        # 期望的最优函数值
        f_opt = 680.6300599487869
        # 期望的最优解向量
        x_opt = (2.330499, 1.951372, -0.4775414, 4.365726,
                 -0.6244870, 1.038131, 1.594227)

        # 断言最优解函数值接近期望值
        assert_allclose(f(x_opt), f_opt)
        # 断言优化结果的函数值接近期望值
        assert_allclose(res.fun, f_opt)
        # 断言优化结果的解向量接近期望解向量，允许的绝对误差为 1e-5
        assert_allclose(res.x, x_opt, atol=1e-5)
        # 断言优化成功
        assert res.success
        # 断言所有约束条件都满足
        assert_(np.all(np.array(c1(res.x)) >= 0))
        # 断言优化结果在定义的变量上下界内
        assert_(np.all(res.x >= np.array(bounds)[:, 0]))
        assert_(np.all(res.x <= np.array(bounds)[:, 1]))

    @pytest.mark.fail_slow(10)
    def test_L3(self):
        # Lampinen ([5]) test problem 3

        # 定义目标函数 f(x)
        def f(x):
            # 将 x 向量补零成为长度为 11 的向量，以匹配索引为 1 的参考
            x = np.hstack(([0], x))  # 1-indexed to match reference
            # 计算目标函数的表达式
            fun = (x[1]**2 + x[2]**2 + x[1]*x[2] - 14*x[1] - 16*x[2] +
                   (x[3]-10)**2 + 4*(x[4]-5)**2 + (x[5]-3)**2 + 2*(x[6]-1)**2 +
                   5*x[7]**2 + 7*(x[8]-11)**2 + 2*(x[9]-10)**2 +
                   (x[10] - 7)**2 + 45
                   )
            return fun  # 最大化该函数

        # 初始化 A 矩阵
        A = np.zeros((4, 11))
        # 填充 A 矩阵的部分元素
        A[1, [1, 2, 7, 8]] = -4, -5, 3, -9
        A[2, [1, 2, 7, 8]] = -10, 8, 17, -2
        A[3, [1, 2, 9, 10]] = 8, -2, -5, 2
        A = A[1:, 1:]  # 取 A 的子矩阵
        # 初始化 b 向量
        b = np.array([-105, 0, -12])

        # 定义约束条件 c1(x)
        def c1(x):
            # 将 x 向量补零成为长度为 11 的向量，以匹配索引为 1 的参考
            x = np.hstack(([0], x))  # 1-indexed to match reference
            # 返回约束条件的列表
            return [3*x[1] - 6*x[2] - 12*(x[9]-8)**2 + 7*x[10],
                    -3*(x[1]-2)**2 - 4*(x[2]-3)**2 - 2*x[3]**2 + 7*x[4] + 120,
                    -x[1]**2 - 2*(x[2]-2)**2 + 2*x[1]*x[2] - 14*x[5] + 6*x[6],
                    -5*x[1]**2 - 8*x[2] - (x[3]-6)**2 + 2*x[4] + 40,
                    -0.5*(x[1]-8)**2 - 2*(x[2]-4)**2 - 3*x[5]**2 + x[6] + 30]

        # 创建线性约束对象 L
        L = LinearConstraint(A, b, np.inf)
        # 创建非线性约束对象 N
        N = NonlinearConstraint(c1, 0, np.inf)
        # 设定变量的取值范围
        bounds = [(-10, 10)]*10
        # 将约束条件包装成元组
        constraints = (L, N)

        # 使用 suppress_warnings 上下文管理器，抑制 UserWarning
        with suppress_warnings() as sup:
            sup.filter(UserWarning)
            # 进行差分进化优化，寻找最优解
            res = differential_evolution(f, bounds, seed=1234,
                                         constraints=constraints, popsize=3)

        # 设定最优解 x_opt 和最优函数值 f_opt
        x_opt = (2.171996, 2.363683, 8.773926, 5.095984, 0.9906548,
                 1.430574, 1.321644, 9.828726, 8.280092, 8.375927)
        f_opt = 24.3062091

        # 使用 assert_allclose 进行精度比较，确保函数值接近最优值
        assert_allclose(f(x_opt), f_opt, atol=1e-5)
        # 使用 assert_allclose 进行精度比较，确保优化结果接近最优解
        assert_allclose(res.x, x_opt, atol=1e-6)
        # 使用 assert_allclose 进行精度比较，确保优化结果的函数值接近最优值
        assert_allclose(res.fun, f_opt, atol=1e-5)
        # 断言优化成功
        assert res.success
        # 断言线性约束条件满足
        assert_(np.all(A @ res.x >= b))
        # 断言非线性约束条件满足
        assert_(np.all(np.array(c1(res.x)) >= 0))
        # 断言变量在取值范围内
        assert_(np.all(res.x >= np.array(bounds)[:, 0]))
        assert_(np.all(res.x <= np.array(bounds)[:, 1]))

    @pytest.mark.fail_slow(10)
    def test_L4(self):
        # Lampinen ([5]) test problem 4
        # 定义函数 f(x)，返回 x 前三个元素的和
        def f(x):
            return np.sum(x[:3])

        # 创建一个 4x9 的全零数组 A
        A = np.zeros((4, 9))
        # 修改数组 A 的部分元素
        A[1, [4, 6]] = 0.0025, 0.0025
        A[2, [5, 7, 4]] = 0.0025, 0.0025, -0.0025
        A[3, [8, 5]] = 0.01, -0.01
        # 裁剪数组 A 的部分，去掉第一行和第一列
        A = A[1:, 1:]
        # 创建一个包含三个元素的数组 b
        b = np.array([1, 1, 1])

        # 定义函数 c1(x)，返回三个约束条件的结果
        def c1(x):
            # 在数组 x 前插入一个元素 0，使索引从 1 开始匹配参考
            x = np.hstack(([0], x))  # 1-indexed to match reference
            return [x[1]*x[6] - 833.33252*x[4] - 100*x[1] + 83333.333,
                    x[2]*x[7] - 1250*x[5] - x[2]*x[4] + 1250*x[4],
                    x[3]*x[8] - 1250000 - x[3]*x[5] + 2500*x[5]]

        # 创建线性约束对象 L
        L = LinearConstraint(A, -np.inf, 1)
        # 创建非线性约束对象 N
        N = NonlinearConstraint(c1, 0, np.inf)

        # 设定变量的取值范围 bounds
        bounds = [(100, 10000)] + [(1000, 10000)]*2 + [(10, 1000)]*5
        constraints = (L, N)

        # 使用 suppress_warnings 上下文管理器，过滤掉 UserWarning 警告
        with suppress_warnings() as sup:
            sup.filter(UserWarning)
            # 使用差分进化算法求解优化问题
            res = differential_evolution(
                f, bounds, strategy='best1bin', seed=1234,
                constraints=constraints, popsize=3, tol=0.05
            )

        # 期望的最优函数值
        f_opt = 7049.248

        # 期望的最优解 x_opt
        x_opt = [579.306692, 1359.97063, 5109.9707, 182.0177, 295.601172,
                217.9823, 286.416528, 395.601172]

        # 断言优化后的函数值接近于期望值 f_opt
        assert_allclose(f(x_opt), f_opt, atol=0.001)
        # 断言优化结果的函数值接近于期望值 f_opt
        assert_allclose(res.fun, f_opt, atol=0.001)

        # 如果是在 32 位 Windows 上，增加容错性，见 gh-11693
        if (platform.system() == 'Windows' and np.dtype(np.intp).itemsize < 8):
            assert_allclose(res.x, x_opt, rtol=2.4e-6, atol=0.0035)
        else:
            # 从 macOS + MKL 失败中确定的容错性，见 gh-12701
            assert_allclose(res.x, x_opt, rtol=5e-6, atol=0.0024)

        # 断言优化成功
        assert res.success
        # 断言线性约束满足
        assert_(np.all(A @ res.x <= b))
        # 断言非线性约束满足
        assert_(np.all(np.array(c1(res.x)) >= 0))
        # 断言变量在取值范围内
        assert_(np.all(res.x >= np.array(bounds)[:, 0]))
        assert_(np.all(res.x <= np.array(bounds)[:, 1]))

    @pytest.mark.fail_slow(10)
    def test_L5(self):
        # Lampinen ([5]) test problem 5
        
        # 定义目标函数 f(x)
        def f(x):
            x = np.hstack(([0], x))  # 1-indexed to match reference
            # 计算目标函数的值
            fun = (np.sin(2*np.pi*x[1])**3 * np.sin(2*np.pi*x[2]) /
                   (x[1]**3 * (x[1] + x[2])))
            return -fun  # 最大化目标函数值

        # 定义约束条件 c1(x)
        def c1(x):
            x = np.hstack(([0], x))  # 1-indexed to match reference
            # 返回约束条件的列表
            return [x[1]**2 - x[2] + 1,
                    1 - x[1] + (x[2] - 4)**2]

        # 创建非线性约束对象 N
        N = NonlinearConstraint(c1, -np.inf, 0)
        
        # 设置变量的取值范围
        bounds = [(0, 10)] * 2
        
        # 定义约束条件集合
        constraints = (N)

        # 使用差分进化算法求解优化问题
        res = differential_evolution(f, bounds, strategy='rand1bin', seed=1234,
                                     constraints=constraints)

        # 预期的最优解和最优值
        x_opt = (1.22797135, 4.24537337)
        f_opt = -0.095825
        
        # 断言目标函数在最优解处的值接近预期最优值
        assert_allclose(f(x_opt), f_opt, atol=2e-5)
        
        # 断言优化结果的最优值接近预期最优值
        assert_allclose(res.fun, f_opt, atol=1e-4)
        
        # 断言优化成功
        assert res.success
        
        # 断言优化结果满足约束条件
        assert_(np.all(np.array(c1(res.x)) <= 0))
        
        # 断言优化结果在变量取值范围内
        assert_(np.all(res.x >= np.array(bounds)[:, 0]))
        assert_(np.all(res.x <= np.array(bounds)[:, 1]))

    @pytest.mark.fail_slow(10)
    def test_L6(self):
        # Lampinen ([5]) test problem 6
        
        # 定义目标函数 f(x)
        def f(x):
            x = np.hstack(([0], x))  # 1-indexed to match reference
            # 计算目标函数的值
            fun = (x[1] - 10)**3 + (x[2] - 20)**3
            return fun

        # 定义约束条件 c1(x)
        def c1(x):
            x = np.hstack(([0], x))  # 1-indexed to match reference
            # 返回约束条件的列表
            return [(x[1] - 5)**2 + (x[2] - 5)**2 - 100,
                    -(x[1] - 6)**2 - (x[2] - 5)**2 + 82.81]

        # 创建非线性约束对象 N
        N = NonlinearConstraint(c1, 0, np.inf)
        
        # 设置变量的取值范围
        bounds = [(13, 100), (0, 100)]
        
        # 定义约束条件集合
        constraints = (N)
        
        # 使用差分进化算法求解优化问题
        res = differential_evolution(f, bounds, strategy='rand1bin', seed=1234,
                                     constraints=constraints, tol=1e-7)

        # 预期的最优解和最优值
        x_opt = (14.095, 0.84296)
        f_opt = -6961.814744

        # 断言目标函数在最优解处的值接近预期最优值
        assert_allclose(f(x_opt), f_opt, atol=1e-6)
        
        # 断言优化结果的最优值接近预期最优值
        assert_allclose(res.fun, f_opt, atol=0.001)
        
        # 断言优化结果的变量值接近预期最优解
        assert_allclose(res.x, x_opt, atol=1e-4)
        
        # 断言优化成功
        assert res.success
        
        # 断言优化结果满足约束条件
        assert_(np.all(np.array(c1(res.x)) >= 0))
        
        # 断言优化结果在变量取值范围内
        assert_(np.all(res.x >= np.array(bounds)[:, 0]))
        assert_(np.all(res.x <= np.array(bounds)[:, 1]))
    def test_L7(self):
        # Lampinen ([5]) test problem 7
        # 定义目标函数 f(x)，这里使用了 NumPy 的函数和数组操作
        def f(x):
            x = np.hstack(([0], x))  # 将 x 扩展成从1开始索引的数组以匹配参考文献
            # 定义目标函数表达式
            fun = (5.3578547*x[3]**2 + 0.8356891*x[1]*x[5] +
                   37.293239*x[1] - 40792.141)
            return fun

        # 定义约束函数 c1(x)，同样使用了 NumPy 的函数和数组操作
        def c1(x):
            x = np.hstack(([0], x))  # 将 x 扩展成从1开始索引的数组以匹配参考文献
            # 返回一个包含多个约束条件的列表
            return [
                    85.334407 + 0.0056858*x[2]*x[5] + 0.0006262*x[1]*x[4] -
                    0.0022053*x[3]*x[5],

                    80.51249 + 0.0071317*x[2]*x[5] + 0.0029955*x[1]*x[2] +
                    0.0021813*x[3]**2,

                    9.300961 + 0.0047026*x[3]*x[5] + 0.0012547*x[1]*x[3] +
                    0.0019085*x[3]*x[4]
                    ]

        # 创建一个非线性约束对象 N
        N = NonlinearConstraint(c1, [0, 90, 20], [92, 110, 25])

        # 定义变量的取值范围 bounds
        bounds = [(78, 102), (33, 45)] + [(27, 45)]*3
        # 定义约束条件 constraints
        constraints = (N)

        # 使用全局优化算法 differential_evolution 求解最优化问题
        res = differential_evolution(f, bounds, strategy='rand1bin', seed=1234,
                                     constraints=constraints)

        # 使用我们的最佳解而非 Lampinen/Koziel 的解。Koziel 的解不满足约束条件，Lampinen 的 f_opt 明显错误。
        x_opt = [78.00000686, 33.00000362, 29.99526064, 44.99999971,
                 36.77579979]

        # 目标函数的最优值
        f_opt = -30665.537578

        # 使用 assert_allclose 进行数值比较，验证最优解和最优值的正确性
        assert_allclose(f(x_opt), f_opt)
        assert_allclose(res.x, x_opt, atol=1e-3)
        assert_allclose(res.fun, f_opt, atol=1e-3)

        # 验证全局优化算法是否成功找到最优解
        assert res.success
        # 验证最优解是否满足约束条件
        assert_(np.all(np.array(c1(res.x)) >= np.array([0, 90, 20])))
        assert_(np.all(np.array(c1(res.x)) <= np.array([92, 110, 25])))
        # 验证最优解是否在定义的变量取值范围内
        assert_(np.all(res.x >= np.array(bounds)[:, 0]))
        assert_(np.all(res.x <= np.array(bounds)[:, 1]))

    @pytest.mark.xslow
    @pytest.mark.xfail(platform.machine() == 'ppc64le',
                       reason="fails on ppc64le")
    def test_L8(self):
        def f(x):
            x = np.hstack(([0], x))  # 将输入数组 x 前面插入一个 0，使索引从 1 开始
            fun = 3*x[1] + 0.000001*x[1]**3 + 2*x[2] + 0.000002/3*x[2]**3  # 定义函数 fun，依赖于 x 的元素
            return fun

        A = np.zeros((3, 5))  # 创建一个 3x5 的全零矩阵 A
        A[1, [4, 3]] = 1, -1  # 设置 A 的第 2 行中的第 5 和第 4 列为 1 和 -1
        A[2, [3, 4]] = 1, -1  # 设置 A 的第 3 行中的第 4 和第 5 列为 1 和 -1
        A = A[1:, 1:]  # 截取 A 的子矩阵，从第 2 行和第 2 列开始，变成 2x4 的矩阵
        b = np.array([-.55, -.55])  # 创建一个包含两个元素的数组 b，分别为 -0.55

        def c1(x):
            x = np.hstack(([0], x))  # 将输入数组 x 前面插入一个 0，使索引从 1 开始
            return [
                    1000*np.sin(-x[3]-0.25) + 1000*np.sin(-x[4]-0.25) +
                    894.8 - x[1],  # 第一个约束条件
                    1000*np.sin(x[3]-0.25) + 1000*np.sin(x[3]-x[4]-0.25) +
                    894.8 - x[2],  # 第二个约束条件
                    1000*np.sin(x[4]-0.25) + 1000*np.sin(x[4]-x[3]-0.25) +
                    1294.8  # 第三个约束条件
                    ]
        L = LinearConstraint(A, b, np.inf)  # 创建线性约束对象 L
        N = NonlinearConstraint(c1, np.full(3, -0.001), np.full(3, 0.001))  # 创建非线性约束对象 N

        bounds = [(0, 1200)]*2+[(-.55, .55)]*2  # 设置变量边界条件
        constraints = (L, N)  # 将约束条件组成一个元组

        with suppress_warnings() as sup:
            sup.filter(UserWarning)  # 过滤掉 UserWarning 类型的警告
            # 原来的 Lampinen 测试使用 rand1bin，但这需要大量 CPU 时间。更改策略为 best1bin 可显著加快速度
            res = differential_evolution(f, bounds, strategy='best1bin',
                                         seed=1234, constraints=constraints,
                                         maxiter=5000)  # 使用差分进化算法求解优化问题

        x_opt = (679.9453, 1026.067, 0.1188764, -0.3962336)  # 期望的最优解 x*
        f_opt = 5126.4981  # 期望的最优函数值 f(x*)

        assert_allclose(f(x_opt), f_opt, atol=1e-3)  # 断言最优解 x* 对应的函数值接近于期望值 f_opt
        assert_allclose(res.x[:2], x_opt[:2], atol=2e-3)  # 断言结果中的前两个变量值接近于 x* 的前两个变量值
        assert_allclose(res.x[2:], x_opt[2:], atol=2e-3)  # 断言结果中的后两个变量值接近于 x* 的后两个变量值
        assert_allclose(res.fun, f_opt, atol=2e-2)  # 断言优化结果的最优函数值接近于期望的最优函数值 f_opt
        assert res.success  # 断言优化过程成功
        assert_(np.all(A@res.x >= b))  # 断言线性约束条件被满足
        assert_(np.all(np.array(c1(res.x)) >= -0.001))  # 断言非线性约束条件被满足（下界）
        assert_(np.all(np.array(c1(res.x)) <= 0.001))  # 断言非线性约束条件被满足（上界）
        assert_(np.all(res.x >= np.array(bounds)[:, 0]))  # 断言变量在设定的下界内
        assert_(np.all(res.x <= np.array(bounds)[:, 1]))  # 断言变量在设定的上界内

    @pytest.mark.fail_slow(5)
    def test_L9(self):
        # Lampinen ([5]) test problem 9

        def f(x):
            x = np.hstack(([0], x))  # 将数组 x 前面添加一个 0，以匹配参考文献的索引方式
            return x[1]**2 + (x[2]-1)**2

        def c1(x):
            x = np.hstack(([0], x))  # 将数组 x 前面添加一个 0，以匹配参考文献的索引方式
            return [x[2] - x[1]**2]

        N = NonlinearConstraint(c1, [-.001], [0.001])  # 创建一个非线性约束条件 N

        bounds = [(-1, 1)]*2  # 定义变量的上下界条件
        constraints = (N)  # 将约束条件 N 放入 constraints 元组中
        res = differential_evolution(f, bounds, strategy='rand1bin', seed=1234,
                                     constraints=constraints)  # 使用差分进化算法求解优化问题

        x_opt = [np.sqrt(2)/2, 0.5]  # 最优解的预期值
        f_opt = 0.75  # 最优目标函数值的预期值

        assert_allclose(f(x_opt), f_opt)  # 断言最优解对应的目标函数值接近预期值
        assert_allclose(np.abs(res.x), x_opt, atol=1e-3)  # 断言优化结果的解向量接近预期最优解，允许一定的数值误差
        assert_allclose(res.fun, f_opt, atol=1e-3)  # 断言优化结果的目标函数值接近预期最优值，允许一定的数值误差
        assert res.success  # 断言优化成功
        assert_(np.all(np.array(c1(res.x)) >= -0.001))  # 断言优化结果满足约束条件 c1 的下界
        assert_(np.all(np.array(c1(res.x)) <= 0.001))  # 断言优化结果满足约束条件 c1 的上界
        assert_(np.all(res.x >= np.array(bounds)[:, 0]))  # 断言优化结果的解向量中的每个元素都大于等于对应的下界
        assert_(np.all(res.x <= np.array(bounds)[:, 1]))  # 断言优化结果的解向量中的每个元素都小于等于对应的上界

    @pytest.mark.fail_slow(10)
    def test_integrality(self):
        # test fitting discrete distribution to data
        rng = np.random.default_rng(6519843218105)  # 使用指定的随机数生成器种子创建随机数生成器对象
        dist = stats.nbinom  # 指定离散分布为负二项分布
        shapes = (5, 0.5)  # 分布的参数形状
        x = dist.rvs(*shapes, size=10000, random_state=rng)  # 从指定分布生成随机变量 x

        def func(p, *args):
            dist, x = args
            # negative log-likelihood function
            ll = -np.log(dist.pmf(x, *p)).sum(axis=-1)  # 计算负对数似然函数
            if np.isnan(ll):  # 处理当 x 超出分布支持范围时的情况
                ll = np.inf  # 设定无穷大作为似然函数值
            return ll

        integrality = [True, False]  # 积分性约束列表
        bounds = [(1, 18), (0, 0.95)]  # 参数 p 的边界条件

        res = differential_evolution(func, bounds, args=(dist, x),
                                     integrality=integrality, polish=False,
                                     seed=rng)  # 使用差分进化算法优化函数 func

        # tolerance has to be fairly relaxed for the second parameter
        # because we're fitting a distribution to random variates.
        assert res.x[0] == 5  # 断言优化结果中第一个参数接近预期值 5
        assert_allclose(res.x, shapes, rtol=0.025)  # 断言优化结果中所有参数接近预期形状参数，允许一定的相对误差

        # check that we can still use integrality constraints with polishing
        res2 = differential_evolution(func, bounds, args=(dist, x),
                                      integrality=integrality, polish=True,
                                      seed=rng)  # 使用差分进化算法优化函数 func，并进行多次局部优化

        def func2(p, *args):
            n, dist, x = args
            return func(np.array([n, p[0]]), dist, x)

        # compare the DE derived solution to an LBFGSB solution (that doesn't
        # have to find the integral values). Note we're setting x0 to be the
        # output from the first DE result, thereby making the polishing step
        # and this minimisation pretty much equivalent.
        LBFGSB = minimize(func2, res2.x[1], args=(5, dist, x),
                          bounds=[(0, 0.95)])  # 使用边界条件优化函数 func2

        assert_allclose(res2.x[1], LBFGSB.x)  # 断言差分进化和 LBFGSB 最小化得到的解向量接近
        assert res2.fun <= res.fun  # 断言经过局部优化后的差分进化结果不劣于未优化的结果
    # 定义测试方法，测试整数性限制
    def test_integrality_limits(self):
        # 定义简单的函数 f(x) = x
        def f(x):
            return x

        # 整数性列表，表示每个维度是否需要整数限制
        integrality = [True, False, True]
        # 边界列表，每个元素是一个元组，表示每个维度的取值范围
        bounds = [(0.2, 1.1), (0.9, 2.2), (3.3, 4.9)]

        # 创建一个差分进化求解器对象，不包含整数性约束
        solver = DifferentialEvolutionSolver(f, bounds=bounds, polish=False,
                                             integrality=False)
        # 断言：确保 solver 对象的 limits[0] 等于 [0.2, 0.9, 3.3]
        assert_allclose(solver.limits[0], [0.2, 0.9, 3.3])
        # 断言：确保 solver 对象的 limits[1] 等于 [1.1, 2.2, 4.9]

        # 创建一个差分进化求解器对象，包含整数性约束
        solver = DifferentialEvolutionSolver(f, bounds=bounds, polish=False,
                                             integrality=integrality)
        # 断言：确保 solver 对象的 limits[0] 等于 [0.5, 0.9, 3.5]
        assert_allclose(solver.limits[0], [0.5, 0.9, 3.5])
        # 断言：确保 solver 对象的 limits[1] 等于 [1.5, 2.2, 4.5]
        assert_allclose(solver.limits[1], [1.5, 2.2, 4.5])
        # 断言：确保 solver 对象的 integrality 属性等于 [True, False, True]
        assert_equal(solver.integrality, [True, False, True])
        # 断言：确保 solver 对象的 polish 属性为 False
        assert solver.polish is False

        # 修改边界，创建另一个差分进化求解器对象，包含整数性约束
        bounds = [(-1.2, -0.9), (0.9, 2.2), (-10.3, 4.1)]
        solver = DifferentialEvolutionSolver(f, bounds=bounds, polish=False,
                                             integrality=integrality)
        # 断言：确保 solver 对象的 limits[0] 等于 [-1.5, 0.9, -10.5]
        assert_allclose(solver.limits[0], [-1.5, 0.9, -10.5])
        # 断言：确保 solver 对象的 limits[1] 等于 [-0.5, 2.2, 4.5]

        # 一个下界为 -1.2 的值被转换为 np.nextafter(np.ceil(-1.2) - 0.5, np.inf)
        # 与上界类似的过程。检查转换是否有效
        # 断言：确保四舍五入后的 solver 对象的 limits[0] 等于 [-1.0, 1.0, -10.0]
        assert_allclose(np.round(solver.limits[0]), [-1.0, 1.0, -10.0])
        # 断言：确保四舍五入后的 solver 对象的 limits[1] 等于 [-1.0, 2.0, 4.0]
        assert_allclose(np.round(solver.limits[1]), [-1.0, 2.0, 4.0])

        # 修改边界，创建另一个差分进化求解器对象，包含整数性约束
        bounds = [(-10.2, -8.1), (0.9, 2.2), (-10.9, -9.9999)]
        solver = DifferentialEvolutionSolver(f, bounds=bounds, polish=False,
                                             integrality=integrality)
        # 断言：确保 solver 对象的 limits[0] 等于 [-10.5, 0.9, -10.5]
        assert_allclose(solver.limits[0], [-10.5, 0.9, -10.5])
        # 断言：确保 solver 对象的 limits[1] 等于 [-8.5, 2.2, -9.5]

        # 修改边界，创建另一个差分进化求解器对象，包含整数性约束
        bounds = [(-10.2, -10.1), (0.9, 2.2), (-10.9, -9.9999)]
        # 使用 pytest 检查是否会抛出 ValueError 异常，匹配指定的错误信息
        with pytest.raises(ValueError, match='One of the integrality'):
            DifferentialEvolutionSolver(f, bounds=bounds, polish=False,
                                        integrality=integrality)
    # 定义一个测试方法，用于测试向量化函数的行为
    def test_vectorized(self):
        
        # 定义一个普通的二次函数，计算输入数组的平方和
        def quadratic(x):
            return np.sum(x**2)

        # 定义一个向量化的二次函数，计算输入数组的每列的平方和
        def quadratic_vec(x):
            return np.sum(x**2, axis=0)

        # 使用 pytest 断言检查，当试图使用 vectorized=True 但未使用 'updating'='immediate' 时，应该引发 RuntimeError 异常
        with pytest.raises(RuntimeError, match='The vectorized function'):
            differential_evolution(quadratic, self.bounds,
                                   vectorized=True, updating='deferred')

        # 当使用 vectorized=True 时，警告检查更新关键字被覆盖
        with warns(UserWarning, match="differential_evolution: the 'vector"):
            differential_evolution(quadratic_vec, self.bounds,
                                   vectorized=True)

        # 当使用 vectorized=True 时，警告检查工作进程关键字被覆盖
        with warns(UserWarning, match="differential_evolution: the 'workers"):
            differential_evolution(quadratic_vec, self.bounds,
                                   vectorized=True, workers=map,
                                   updating='deferred')

        # 定义一个变量用于记录函数调用次数
        ncalls = [0]

        # 定义一个向量化的 Rosenbrock 函数，并在范围内进行优化，使用 'updating'='deferred'
        def rosen_vec(x):
            ncalls[0] += 1
            return rosen(x)

        bounds = [(0, 10), (0, 10)]

        # 分别使用非向量化和向量化的方式运行 differential_evolution，进行优化
        res1 = differential_evolution(rosen, bounds, updating='deferred',
                                      seed=1)
        res2 = differential_evolution(rosen_vec, bounds, vectorized=True,
                                      updating='deferred', seed=1)

        # 断言两次最小化运行的结果应该是功能上等效的
        assert_allclose(res1.x, res2.x)
        assert ncalls[0] == res2.nfev
        assert res1.nit == res2.nit

    # 定义一个测试向量化约束条件的方法
    def test_vectorized_constraints(self):
        
        # 定义一个简单的约束函数，返回一个由 x[0] + x[1] 组成的数组
        def constr_f(x):
            return np.array([x[0] + x[1]])

        # 定义另一个约束函数，返回一个由 x[0]**2 + x[1] 和 x[0] - x[1] 组成的数组
        def constr_f2(x):
            return np.array([x[0]**2 + x[1], x[0] - x[1]])

        # 创建两个非线性约束对象
        nlc1 = NonlinearConstraint(constr_f, -np.inf, 1.9)
        nlc2 = NonlinearConstraint(constr_f2, (0.9, 0.5), (2.0, 2.0))

        # 定义一个向量化的 Rosenbrock 函数，接受一个 (len(x0), S) 数组作为输入，返回一个 (S,) 数组
        def rosen_vec(x):
            v = 100 * (x[1:] - x[:-1]**2.0)**2.0
            v += (1 - x[:-1])**2.0
            return np.squeeze(v)

        bounds = [(0, 10), (0, 10)]

        # 分别使用非向量化和向量化的方式运行 differential_evolution，进行优化，同时应用约束条件
        res1 = differential_evolution(rosen, bounds, updating='deferred',
                                      seed=1, constraints=[nlc1, nlc2],
                                      polish=False)
        res2 = differential_evolution(rosen_vec, bounds, vectorized=True,
                                      updating='deferred', seed=1,
                                      constraints=[nlc1, nlc2],
                                      polish=False)

        # 断言两次最小化运行的结果应该是功能上等效的
        assert_allclose(res1.x, res2.x)
    # 定义一个测试方法，用于测试约束条件违反时的错误消息

        # 定义一个函数 func，接受一个参数 x，并返回 np.cos(x[0]) + np.sin(x[1]) 的计算结果
        def func(x):
            return np.cos(x[0]) + np.sin(x[1])

        # 创建两个非线性约束对象 c0 和 c1
        # c0: x[1] - (x[0]-1)**2 >= 0
        c0 = NonlinearConstraint(lambda x: x[1] - (x[0]-1)**2, 0, np.inf)
        # c1: x[1] + x[0]**2 <= 0
        c1 = NonlinearConstraint(lambda x: x[1] + x[0]**2, -np.inf, 0)

        # 使用差分进化算法求解 func 的最小值
        result = differential_evolution(func,
                                        bounds=[(-1, 2), (-1, 1)],
                                        constraints=[c0, c1],
                                        maxiter=10,
                                        polish=False,
                                        seed=864197532)
        # 断言求解是否成功，应为 False
        assert result.success is False

        # 断言错误消息中是否包含 "MAXCV = 0.4"
        # 错误消息中的具体数字可能因实现变更而变化，如果需要，文本可以缩减为 "MAXCV = 0."
        assert "MAXCV = 0.4" in result.message

    # 使用 pytest 的 mark 标记，指定这个测试可以允许一定的慢速失败，这是一个特定请求下的 fail-slow 异常，参见 gh-20806
    @pytest.mark.fail_slow(20)
    def test_strategy_fn(self):
        # 测试策略函数的功能，模拟一个内置策略
        parameter_count = 4
        popsize = 10
        bounds = [(0, 10.)] * parameter_count
        total_popsize = parameter_count * popsize
        mutation = 0.8
        recombination = 0.7

        calls = [0]
        def custom_strategy_fn(candidate, population, rng=None):
            # 自定义策略函数，用于差分进化算法中的演变策略
            calls[0] += 1
            trial = np.copy(population[candidate])
            fill_point = rng.choice(parameter_count)

            pool = np.arange(total_popsize)
            rng.shuffle(pool)
            idxs = pool[:2 + 1]
            idxs = idxs[idxs != candidate][:2]

            r0, r1 = idxs[:2]

            bprime = (population[0] + mutation *
                    (population[r0] - population[r1]))

            crossovers = rng.uniform(size=parameter_count)
            crossovers = crossovers < recombination
            crossovers[fill_point] = True
            trial = np.where(crossovers, bprime, trial)
            return trial

        solver = DifferentialEvolutionSolver(
            rosen,
            bounds,
            popsize=popsize,
            recombination=recombination,
            mutation=mutation,
            maxiter=2,
            strategy=custom_strategy_fn,
            seed=10,
            polish=False
        )
        assert solver.strategy is custom_strategy_fn
        solver.solve()
        assert calls[0] > 0

        # 检查使用 updating='deferred' 的自定义策略是否有效
        res = differential_evolution(
            rosen, bounds, strategy=custom_strategy_fn, updating='deferred'
        )
        assert res.success

        def custom_strategy_fn(candidate, population, rng=None):
            # 返回一个固定数组作为演化策略的自定义函数
            return np.array([1.0, 2.0])

        with pytest.raises(RuntimeError, match="strategy*"):
            # 检查重新定义的策略函数是否触发 RuntimeError
            differential_evolution(
                rosen,
                bounds,
                strategy=custom_strategy_fn
            )
```