# `D:\src\scipysrc\scipy\scipy\optimize\tests\test__dual_annealing.py`

```
# Dual annealing unit tests implementation.
# Copyright (c) 2018 Sylvain Gubian <sylvain.gubian@pmi.com>,
# Yang Xiang <yang.xiang@pmi.com>
# Author: Sylvain Gubian, PMP S.A.
"""
Unit tests for the dual annealing global optimizer
"""
# 导入所需的模块和函数
from scipy.optimize import dual_annealing, Bounds

# 导入 dual_annealing 模块中的一些类和函数
from scipy.optimize._dual_annealing import EnergyState
from scipy.optimize._dual_annealing import LocalSearchWrapper
from scipy.optimize._dual_annealing import ObjectiveFunWrapper
from scipy.optimize._dual_annealing import StrategyChain
from scipy.optimize._dual_annealing import VisitingDistribution
from scipy.optimize import rosen, rosen_der
import pytest
import numpy as np
# 导入测试所需的断言函数
from numpy.testing import assert_equal, assert_allclose, assert_array_less
from pytest import raises as assert_raises
# 导入随机数生成函数
from scipy._lib._util import check_random_state


class TestDualAnnealing:

    def setup_method(self):
        # 定义一个始终返回无穷大的测试函数
        self.weirdfunc = lambda x: np.inf
        # 用于测试函数的二维边界
        self.ld_bounds = [(-5.12, 5.12)] * 2
        # 用于测试函数的四维边界
        self.hd_bounds = self.ld_bounds * 4
        # 用于测试访问函数的生成值数量
        self.nbtestvalues = 5000
        # 高温参数
        self.high_temperature = 5230
        # 低温参数
        self.low_temperature = 0.1
        # 访问分布参数
        self.qv = 2.62
        # 随机数种子
        self.seed = 1234
        # 使用种子生成随机状态对象
        self.rs = check_random_state(self.seed)
        # 调用函数计数器
        self.nb_fun_call = 0
        # 目标函数梯度计数器
        self.ngev = 0

    def callback(self, x, f, context):
        # 用于测试回调机制的函数。如果 f <= 1.0，则返回 True 终止优化过程
        if f <= 1.0:
            return True

    def func(self, x, args=()):
        # 使用 Rastrigin 函数进行测试
        if args:
            shift = args
        else:
            shift = 0
        # Rastrigin 函数计算
        y = np.sum((x - shift) ** 2 - 10 * np.cos(2 * np.pi * (
            x - shift))) + 10 * np.size(x) + shift
        # 增加函数调用计数
        self.nb_fun_call += 1
        return y

    def rosen_der_wrapper(self, x, args=()):
        # 增加目标函数梯度计数
        self.ngev += 1
        # 调用 Rosenbrock 函数的梯度计算函数
        return rosen_der(x, *args)

    # FIXME: there are some discontinuities in behaviour as a function of `qv`,
    #        this needs investigating - see gh-12384
    @pytest.mark.parametrize('qv', [1.1, 1.41, 2, 2.62, 2.9])
    def test_visiting_stepping(self, qv):
        # 获取边界的上下限值列表
        lu = list(zip(*self.ld_bounds))
        lower = np.array(lu[0])
        upper = np.array(lu[1])
        dim = lower.size
        # 使用给定的上下限、qv 和随机状态生成访问分布对象
        vd = VisitingDistribution(lower, upper, qv, self.rs)
        values = np.zeros(dim)
        # 执行访问函数的测试，返回低温状态下的步进结果
        x_step_low = vd.visiting(values, 0, self.high_temperature)
        # 确保只有第一个分量被改变
        assert_equal(np.not_equal(x_step_low, 0), True)
        values = np.zeros(dim)
        # 执行访问函数的测试，返回高温状态下的步进结果
        x_step_high = vd.visiting(values, dim, self.high_temperature)
        # 确保除了 dim 处的分量之外，其他分量都被改变
        assert_equal(np.not_equal(x_step_high[0], 0), True)
    # 使用 pytest 的参数化装饰器，为该测试方法多次传递不同的参数
    @pytest.mark.parametrize('qv', [2.25, 2.62, 2.9])
    # 定义测试访问高温情况下访问分布的方法
    def test_visiting_dist_high_temperature(self, qv):
        # 解压缩低和高边界，并转换为 NumPy 数组
        lu = list(zip(*self.ld_bounds))
        lower = np.array(lu[0])
        upper = np.array(lu[1])
        # 创建访问分布对象 vd，使用给定的参数 qv 和随机状态对象 self.rs
        vd = VisitingDistribution(lower, upper, qv, self.rs)
        # 使用高温值来计算访问函数的值
        values = vd.visit_fn(self.high_temperature, self.nbtestvalues)

        # 验证：访问分布是扭曲的柯西-洛伦兹分布，没有定义一阶及更高阶矩（无均值定义，无方差定义）
        # 检查生成的值是否包含大尾部值
        assert_array_less(np.min(values), 1e-10)
        assert_array_less(1e+10, np.max(values))

    # 定义测试重置方法
    def test_reset(self):
        # 使用 ObjectiveFunWrapper 包装奇怪的函数 self.weirdfunc
        owf = ObjectiveFunWrapper(self.weirdfunc)
        # 解压缩低和高边界，并转换为 NumPy 数组
        lu = list(zip(*self.ld_bounds))
        lower = np.array(lu[0])
        upper = np.array(lu[1])
        # 创建能量状态对象 es
        es = EnergyState(lower, upper)
        # 验证：重置能量状态 es 时会引发 ValueError 异常，使用 owf 包装的函数，不使用随机状态检查
        assert_raises(ValueError, es.reset, owf, check_random_state(None))

    # 定义测试低维度优化方法
    def test_low_dim(self):
        # 使用双重退火算法进行优化，传递函数 self.func 和低维边界 self.ld_bounds，设置随机种子 self.seed
        ret = dual_annealing(self.func, self.ld_bounds, seed=self.seed)
        # 验证：返回的最小函数值近似为 0，绝对误差容忍度为 1e-12
        assert_allclose(ret.fun, 0., atol=1e-12)
        # 验证：优化成功，ret.success 为 True
        assert ret.success

    # 使用 pytest 的标记 fail_slow(10) 标记测试高维度优化方法
    @pytest.mark.fail_slow(10)
    def test_high_dim(self):
        # 使用双重退火算法进行优化，传递函数 self.func 和高维边界 self.hd_bounds，设置随机种子 self.seed
        ret = dual_annealing(self.func, self.hd_bounds, seed=self.seed)
        # 验证：返回的最小函数值近似为 0，绝对误差容忍度为 1e-12
        assert_allclose(ret.fun, 0., atol=1e-12)
        # 验证：优化成功，ret.success 为 True
        assert ret.success

    # 定义测试无局部搜索的低维度优化方法
    def test_low_dim_no_ls(self):
        # 使用双重退火算法进行优化，传递函数 self.func 和低维边界 self.ld_bounds，设置无局部搜索和随机种子 self.seed
        ret = dual_annealing(self.func, self.ld_bounds, no_local_search=True, seed=self.seed)
        # 验证：返回的最小函数值近似为 0，绝对误差容忍度为 1e-4
        assert_allclose(ret.fun, 0., atol=1e-4)

    # 使用 pytest 的标记 fail_slow(10) 标记测试无局部搜索的高维度优化方法
    @pytest.mark.fail_slow(10)
    def test_high_dim_no_ls(self):
        # 使用双重退火算法进行优化，传递函数 self.func 和高维边界 self.hd_bounds，设置无局部搜索和随机种子 self.seed
        ret = dual_annealing(self.func, self.hd_bounds, no_local_search=True, seed=self.seed)
        # 验证：返回的最小函数值近似为 0，绝对误差容忍度为 1e-4
        assert_allclose(ret.fun, 0., atol=1e-4)

    # 定义测试函数调用次数方法
    def test_nb_fun_call(self):
        # 使用双重退火算法进行优化，传递函数 self.func 和低维边界 self.ld_bounds，设置随机种子 self.seed
        ret = dual_annealing(self.func, self.ld_bounds, seed=self.seed)
        # 验证：函数调用次数与预期的 self.nb_fun_call 相等
        assert_equal(self.nb_fun_call, ret.nfev)

    # 定义测试无局部搜索的函数调用次数方法
    def test_nb_fun_call_no_ls(self):
        # 使用双重退火算法进行优化，传递函数 self.func 和低维边界 self.ld_bounds，设置无局部搜索和随机种子 self.seed
        ret = dual_annealing(self.func, self.ld_bounds, no_local_search=True, seed=self.seed)
        # 验证：函数调用次数与预期的 self.nb_fun_call 相等
        assert_equal(self.nb_fun_call, ret.nfev)

    # 定义测试最大重初始化次数方法
    def test_max_reinit(self):
        # 验证：在传递奇怪函数 self.weirdfunc 和低维边界 self.ld_bounds 时会引发 ValueError 异常
        assert_raises(ValueError, dual_annealing, self.weirdfunc, self.ld_bounds)

    # 使用 pytest 的标记 fail_slow(10) 标记测试复现性方法
    @pytest.mark.fail_slow(10)
    def test_reproduce(self):
        # 使用双重退火算法进行优化，传递函数 self.func 和低维边界 self.ld_bounds，设置随机种子 self.seed
        res1 = dual_annealing(self.func, self.ld_bounds, seed=self.seed)
        # 使用相同的设置再次运行双重退火算法，以验证结果是否可复现
        res2 = dual_annealing(self.func, self.ld_bounds, seed=self.seed)
        res3 = dual_annealing(self.func, self.ld_bounds, seed=self.seed)
        # 验证：如果结果可复现，则 x 组件应完全相同，当未使用种子时，这并非如此
        assert_equal(res1.x, res2.x)
        assert_equal(res1.x, res3.x)
    def test_rand_gen(self):
        # 检查 np.random.Generator 可以被使用（numpy >= 1.17）
        # 获取一个 np.random.Generator 对象
        rng = np.random.default_rng(1)

        # 使用 rng 种子运行 dual_annealing 函数，并记录结果
        res1 = dual_annealing(self.func, self.ld_bounds, seed=rng)

        # 再次设置相同的种子
        rng = np.random.default_rng(1)

        # 使用相同的 rng 种子再次运行 dual_annealing 函数，并记录结果
        res2 = dual_annealing(self.func, self.ld_bounds, seed=rng)

        # 断言：如果结果是可复现的，那么找到的 x 组件必须完全相同，这里用 assert_equal 进行验证
        assert_equal(res1.x, res2.x)

    def test_bounds_integrity(self):
        # 错误的边界范围
        wrong_bounds = [(-5.12, 5.12), (1, 0), (5.12, 5.12)]

        # 断言：使用这些错误的边界范围调用 dual_annealing 函数会抛出 ValueError 异常
        assert_raises(ValueError, dual_annealing, self.func, wrong_bounds)

    def test_bound_validity(self):
        # 无效的边界范围
        invalid_bounds = [(-5, 5), (-np.inf, 0), (-5, 5)]

        # 断言：使用这些无效的边界范围调用 dual_annealing 函数会抛出 ValueError 异常
        assert_raises(ValueError, dual_annealing, self.func, invalid_bounds)

        invalid_bounds = [(-5, 5), (0, np.inf), (-5, 5)]

        # 断言：使用这些无效的边界范围调用 dual_annealing 函数会抛出 ValueError 异常
        assert_raises(ValueError, dual_annealing, self.func, invalid_bounds)

        invalid_bounds = [(-5, 5), (0, np.nan), (-5, 5)]

        # 断言：使用这些无效的边界范围调用 dual_annealing 函数会抛出 ValueError 异常
        assert_raises(ValueError, dual_annealing, self.func, invalid_bounds)

    def test_deprecated_local_search_options_bounds(self):
        # 内部函数 func 定义，计算简单的数学表达式
        def func(x):
            return np.sum((x - 5) * (x - 1))

        # 创建 bounds 列表，用于定义变量的搜索范围
        bounds = list(zip([-6, -5], [6, 5]))

        # 使用 pytest 来检查警告信息，确保方法 "CG" 无法处理的警告会被触发
        with pytest.warns(RuntimeWarning, match=r"Method CG cannot handle "):
            dual_annealing(
                func,
                bounds=bounds,
                minimizer_kwargs={"method": "CG", "bounds": bounds})

    def test_minimizer_kwargs_bounds(self):
        # 内部函数 func 定义，计算简单的数学表达式
        def func(x):
            return np.sum((x - 5) * (x - 1))

        # 创建 bounds 列表，用于定义变量的搜索范围
        bounds = list(zip([-6, -5], [6, 5]))

        # 测试确保 bounds 参数可以正常传递（参见 gh-10831）
        dual_annealing(
            func,
            bounds=bounds,
            minimizer_kwargs={"method": "SLSQP", "bounds": bounds})

        # 使用 pytest 来检查警告信息，确保方法 "CG" 无法处理的警告会被触发
        with pytest.warns(RuntimeWarning, match=r"Method CG cannot handle "):
            dual_annealing(
                func,
                bounds=bounds,
                minimizer_kwargs={"method": "CG", "bounds": bounds})

    def test_max_fun_ls(self):
        # 使用给定的 func 函数和 ld_bounds 调用 dual_annealing 函数，设置最大函数评估次数为 100
        ret = dual_annealing(self.func, self.ld_bounds, maxfun=100,
                             seed=self.seed)

        # 计算局部搜索的最大迭代次数，用于后续断言
        ls_max_iter = min(max(
            len(self.ld_bounds) * LocalSearchWrapper.LS_MAXITER_RATIO,
            LocalSearchWrapper.LS_MAXITER_MIN),
            LocalSearchWrapper.LS_MAXITER_MAX)

        # 断言：返回结果的函数评估次数应该小于等于 100 加上局部搜索的最大迭代次数
        assert ret.nfev <= 100 + ls_max_iter
        # 断言：返回结果的成功标志应为假（即未成功）

    def test_max_fun_no_ls(self):
        # 使用给定的 func 函数和 ld_bounds 调用 dual_annealing 函数，设置不使用局部搜索并且最大函数评估次数为 500
        ret = dual_annealing(self.func, self.ld_bounds,
                             no_local_search=True, maxfun=500, seed=self.seed)

        # 断言：返回结果的函数评估次数应该小于等于 500
        assert ret.nfev <= 500
        # 断言：返回结果的成功标志应为假（即未成功）
    # 测试 dual_annealing 函数的 maxiter 参数是否正常工作
    def test_maxiter(self):
        ret = dual_annealing(self.func, self.ld_bounds, maxiter=700,
                             seed=self.seed)
        # 断言迭代次数是否不超过 700
        assert ret.nit <= 700

    # 测试确保参数正确传递给 dual_annealing 函数
    def test_fun_args_ls(self):
        ret = dual_annealing(self.func, self.ld_bounds,
                             args=((3.14159,)), seed=self.seed)
        # 断言优化结果的目标函数值接近 3.14159，允许误差为 1e-6
        assert_allclose(ret.fun, 3.14159, atol=1e-6)

    # 测试确保参数正确传递给纯模拟退火的 dual_annealing 函数
    def test_fun_args_no_ls(self):
        ret = dual_annealing(self.func, self.ld_bounds,
                             args=((3.14159, )), no_local_search=True,
                             seed=self.seed)
        # 断言优化结果的目标函数值接近 3.14159，允许误差为 1e-4
        assert_allclose(ret.fun, 3.14159, atol=1e-4)

    # 测试 callback 函数是否能够使算法在目标函数值 <= 1.0 时停止
    def test_callback_stop(self):
        ret = dual_annealing(self.func, self.ld_bounds,
                             callback=self.callback, seed=self.seed)
        # 断言优化结果的目标函数值不超过 1.0
        assert ret.fun <= 1.0
        # 断言优化过程中输出的信息包含 'stop early'
        assert 'stop early' in ret.message[0]
        # 断言优化未成功完成
        assert not ret.success

    # 使用参数化测试，测试多种方法在 dual_annealing 函数中的最小化效果
    @pytest.mark.parametrize('method, atol', [
        ('Nelder-Mead', 2e-5),
        ('COBYLA', 1e-5),
        ('COBYQA', 1e-8),
        ('Powell', 1e-8),
        ('CG', 1e-8),
        ('BFGS', 1e-8),
        ('TNC', 1e-8),
        ('SLSQP', 2e-7),
    ])
    def test_multi_ls_minimizer(self, method, atol):
        ret = dual_annealing(self.func, self.ld_bounds,
                             minimizer_kwargs=dict(method=method),
                             seed=self.seed)
        # 断言优化结果的目标函数值接近 0.，允许的误差在 atol 范围内
        assert_allclose(ret.fun, 0., atol=atol)

    # 测试 dual_annealing 函数在错误的 restart_temp_ratio 参数下是否会引发 ValueError
    def test_wrong_restart_temp(self):
        assert_raises(ValueError, dual_annealing, self.func,
                      self.ld_bounds, restart_temp_ratio=1)
        assert_raises(ValueError, dual_annealing, self.func,
                      self.ld_bounds, restart_temp_ratio=0)

    # 使用梯度信息测试 dual_annealing 函数的最小化效果
    def test_gradient_gnev(self):
        minimizer_opts = {
            'jac': self.rosen_der_wrapper,
        }
        ret = dual_annealing(rosen, self.ld_bounds,
                             minimizer_kwargs=minimizer_opts,
                             seed=self.seed)
        # 断言优化过程中的雅可比评估次数与预期相等
        assert ret.njev == self.ngev

    # 使用文档字符串中的例子进行测试，确保 dual_annealing 函数能够正确优化目标函数
    @pytest.mark.fail_slow(10)
    def test_from_docstring(self):
        def func(x):
            return np.sum(x * x - 10 * np.cos(2 * np.pi * x)) + 10 * np.size(x)
        lw = [-5.12] * 10
        up = [5.12] * 10
        ret = dual_annealing(func, bounds=list(zip(lw, up)), seed=1234)
        # 断言优化结果的目标函数最优解接近给定值，允许的误差在 4e-8 范围内
        assert_allclose(ret.x,
                        [-4.26437714e-09, -3.91699361e-09, -1.86149218e-09,
                         -3.97165720e-09, -6.29151648e-09, -6.53145322e-09,
                         -3.93616815e-09, -6.55623025e-09, -6.05775280e-09,
                         -5.00668935e-09], atol=4e-8)
        # 断言优化结果的目标函数值接近 0.，允许的误差在 5e-13 范围内
        assert_allclose(ret.fun, 0.000000, atol=5e-13)
    @pytest.mark.parametrize('new_e, temp_step, accepted, accept_rate', [
        (0, 100, 1000, 1.0097587941791923),
        (0, 2, 1000, 1.2599210498948732),
        (10, 100, 878, 0.8786035869128718),
        (10, 60, 695, 0.6812920690579612),
        (2, 100, 990, 0.9897404249173424),
    ])
    def test_accept_reject_probabilistic(
            self, new_e, temp_step, accepted, accept_rate):
        # Test accepts unconditionally with e < current_energy and
        # probabilistically with e > current_energy

        rs = check_random_state(123)  # 创建一个随机状态对象

        count_accepted = 0  # 初始化计数器
        iterations = 1000  # 设定迭代次数

        accept_param = -5  # 设置接受参数
        current_energy = 1  # 设置当前能量值
        for _ in range(iterations):
            energy_state = EnergyState(lower=None, upper=None)  # 创建能量状态对象
            # 使用当前能量值更新能量状态，位置参数为 [0]
            energy_state.update_current(current_energy, [0])

            chain = StrategyChain(
                accept_param, None, None, None, rs, energy_state)  # 创建策略链对象
            # 通常这个值在 run() 方法中设置
            chain.temperature_step = temp_step  # 设置温度步长

            # 检查更新是否被接受
            chain.accept_reject(j=1, e=new_e, x_visit=[2])
            if energy_state.current_energy == new_e:  # 如果当前能量等于新能量
                count_accepted += 1  # 计数器加一

        assert count_accepted == accepted  # 断言接受的次数与预期相符

        # 检查接受率
        pqv = 1 - (1 - accept_param) * (new_e - current_energy) / temp_step
        rate = 0 if pqv <= 0 else np.exp(np.log(pqv) / (1 - accept_param))
        assert_allclose(rate, accept_rate)  # 断言接受率与预期相符

    @pytest.mark.fail_slow(10)
    def test_bounds_class(self):
        # test that result does not depend on the bounds type
        def func(x):
            f = np.sum(x * x - 10 * np.cos(2 * np.pi * x)) + 10 * np.size(x)
            return f
        lw = [-5.12] * 5  # 定义下界数组
        up = [5.12] * 5   # 定义上界数组

        # 无界全局最小值是全零。大多数边界将迫使设计变量远离无界最小值，并在解决方案中激活。
        up[0] = -2.0  # 更新上界
        up[1] = -1.0  # 更新上界
        lw[3] = 1.0   # 更新下界
        lw[4] = 2.0   # 更新下界

        # 运行优化
        bounds = Bounds(lw, up)  # 创建边界对象
        ret_bounds_class = dual_annealing(func, bounds=bounds, seed=1234)  # 调用双退火算法优化

        bounds_old = list(zip(lw, up))  # 创建旧版边界列表
        ret_bounds_list = dual_annealing(func, bounds=bounds_old, seed=1234)  # 使用旧版边界运行双退火算法

        # 测试找到的最小值、函数评估次数和迭代次数是否匹配
        assert_allclose(ret_bounds_class.x, ret_bounds_list.x, atol=1e-8)  # 断言最小值数组匹配
        assert_allclose(ret_bounds_class.x, np.arange(-2, 3), atol=1e-7)  # 断言最小值数组接近预期范围
        assert_allclose(ret_bounds_list.fun, ret_bounds_class.fun, atol=1e-9)  # 断言函数值接近
        assert ret_bounds_list.nfev == ret_bounds_class.nfev  # 断言函数评估次数相等

    @pytest.mark.fail_slow(10)  # 标记为慢速失败测试
    def test_callable_jac_hess_with_args_gh11052(self):
        # 测试当 `jac` 是可调用的且使用了 `args` 时，dual_annealing 是否能正常运行。
        # 这个例子来自于 gh-11052。

        # 在关闭 gh20614 时扩展到 hess。
        
        # 使用种子生成一个随机数生成器 rng
        rng = np.random.default_rng(94253637693657847462)

        # 定义函数 f(x, power)，计算 x^power 的指数和
        def f(x, power):
            return np.sum(np.exp(x ** power))

        # 定义函数 jac(x, power)，计算 f 对 x 的雅可比矩阵
        def jac(x, power):
            return np.exp(x ** power) * power * x ** (power - 1)

        # 定义函数 hess(x, power)，计算 f 对 x 的黑塞矩阵
        def hess(x, power):
            # 使用 WolframAlpha 计算得出 d^2/dx^2 e^(x^p)
            return np.diag(
                power * np.exp(x ** power) * x ** (power - 2) *
                (power * x ** power + power - 1)
            )

        # 定义函数 hessp(x, p, power)，计算 hess(x, power) 与向量 p 的乘积
        def hessp(x, p, power):
            return hess(x, power) @ p

        # 使用 dual_annealing 运行优化，不包含 jac 和 hess
        res1 = dual_annealing(f, args=(2, ), bounds=[[0, 1], [0, 1]], seed=rng,
                              minimizer_kwargs=dict(method='L-BFGS-B'))

        # 使用 dual_annealing 运行优化，包含 jac 函数
        res2 = dual_annealing(f, args=(2, ), bounds=[[0, 1], [0, 1]], seed=rng,
                              minimizer_kwargs=dict(method='L-BFGS-B',
                                                    jac=jac))

        # 使用 dual_annealing 运行优化，包含 jac 和 hess 函数
        res3 = dual_annealing(f, args=(2, ), bounds=[[0, 1], [0, 1]], seed=rng,
                              minimizer_kwargs=dict(method='newton-cg',
                                                    jac=jac, hess=hess))

        # 使用 dual_annealing 运行优化，包含 jac 和 hessp 函数
        res4 = dual_annealing(f, args=(2, ), bounds=[[0, 1], [0, 1]], seed=rng,
                              minimizer_kwargs=dict(method='newton-cg',
                                                    jac=jac, hessp=hessp))

        # 检查 res1 和 res2 的最小化函数值是否在指定相对容差下接近
        assert_allclose(res1.fun, res2.fun, rtol=1e-6)

        # 检查 res3 和 res2 的最小化函数值是否在指定相对容差下接近
        assert_allclose(res3.fun, res2.fun, rtol=1e-6)

        # 检查 res4 和 res2 的最小化函数值是否在指定相对容差下接近
        assert_allclose(res4.fun, res2.fun, rtol=1e-6)
```