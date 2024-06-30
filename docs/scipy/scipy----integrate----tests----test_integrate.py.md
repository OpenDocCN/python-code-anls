# `D:\src\scipysrc\scipy\scipy\integrate\tests\test_integrate.py`

```
# Authors: Nils Wagner, Ed Schofield, Pauli Virtanen, John Travers
"""
Tests for numerical integration.
"""
# 导入必要的库
import numpy as np
# 从 numpy 库中导入多个函数和类
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
                   allclose)
# 从 numpy.testing 模块中导入多个断言函数
from numpy.testing import (
    assert_, assert_array_almost_equal,
    assert_allclose, assert_array_equal, assert_equal, assert_warns)
# 从 pytest 模块中导入 raises 函数并起别名为 assert_raises
from pytest import raises as assert_raises
# 从 scipy.integrate 模块中导入 ODE 求解相关的函数和类
from scipy.integrate import odeint, ode, complex_ode

#------------------------------------------------------------------------------
# Test ODE integrators
#------------------------------------------------------------------------------

# 定义一个测试类 TestOdeint，用于测试 odeint 函数
class TestOdeint:
    # Check integrate.odeint

    # 定义一个内部方法 _do_problem，用于执行具体的测试问题
    def _do_problem(self, problem):
        # 生成一个时间数组 t，步长为 0.05，范围从 0.0 到 problem.stop_t
        t = arange(0.0, problem.stop_t, 0.05)

        # 基本情况下的测试
        z, infodict = odeint(problem.f, problem.z0, t, full_output=True)
        assert_(problem.verify(z, t))

        # 使用 tfirst=True 的情况
        z, infodict = odeint(lambda t, y: problem.f(y, t), problem.z0, t,
                             full_output=True, tfirst=True)
        assert_(problem.verify(z, t))

        # 如果定义了 problem.jac，使用 Dfun 的情况
        if hasattr(problem, 'jac'):
            z, infodict = odeint(problem.f, problem.z0, t, Dfun=problem.jac,
                                 full_output=True)
            assert_(problem.verify(z, t))

            # 使用 Dfun 和 tfirst=True 的情况
            z, infodict = odeint(lambda t, y: problem.f(y, t), problem.z0, t,
                                 Dfun=lambda t, y: problem.jac(y, t),
                                 full_output=True, tfirst=True)
            assert_(problem.verify(z, t))

    # 定义一个测试方法 test_odeint，用于测试不同的问题类 PROBLEMS 中的问题
    def test_odeint(self):
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if problem.cmplx:
                continue
            self._do_problem(problem)

# 定义一个测试类 TestODEClass，用于测试 ODE 求解器的通用行为
class TestODEClass:

    ode_class = None   # Set in subclass.

    # 定义一个内部方法 _do_problem，用于执行具体的测试问题
    def _do_problem(self, problem, integrator, method='adams'):

        # ode 的回调函数参数顺序与 odeint 不同
        def f(t, z):
            return problem.f(z, t)
        
        # 如果定义了 problem.jac，定义雅可比矩阵函数 jac
        jac = None
        if hasattr(problem, 'jac'):
            def jac(t, z):
                return problem.jac(z, t)

        # 设置积分器的参数
        integrator_params = {}
        if problem.lband is not None or problem.uband is not None:
            integrator_params['uband'] = problem.uband
            integrator_params['lband'] = problem.lband

        # 创建 ODE 求解器对象 ig
        ig = self.ode_class(f, jac)
        ig.set_integrator(integrator,
                          atol=problem.atol/10,
                          rtol=problem.rtol/10,
                          method=method,
                          **integrator_params)

        # 设置初始条件
        ig.set_initial_value(problem.z0, t=0.0)
        # 进行积分计算，得到结果 z
        z = ig.integrate(problem.stop_t)

        # 断言结果 z 与 ig.y 相等
        assert_array_equal(z, ig.y)
        # 断言积分是否成功
        assert_(ig.successful(), (problem, method))
        # 断言返回码是否大于 0
        assert_(ig.get_return_code() > 0, (problem, method))
        # 验证问题是否通过问题的 verify 方法
        assert_(problem.verify(array([z]), problem.stop_t), (problem, method))
class TestOde(TestODEClass):
    # 定义一个测试类 TestOde，继承自 TestODEClass

    ode_class = ode
    # 设置类变量 ode_class 为 ode

    def test_vode(self):
        # 测试 vode 求解器
        for problem_cls in PROBLEMS:
            # 遍历所有的问题类别 PROBLEMS
            problem = problem_cls()
            # 创建一个具体的问题实例
            if problem.cmplx:
                # 如果问题复杂，则跳过此问题
                continue
            if not problem.stiff:
                # 如果问题不是刚性的
                self._do_problem(problem, 'vode', 'adams')
                # 调用 _do_problem 方法，使用 'vode' 求解器和 'adams' 方法求解问题
            self._do_problem(problem, 'vode', 'bdf')
            # 调用 _do_problem 方法，使用 'vode' 求解器和 'bdf' 方法求解问题

    def test_zvode(self):
        # 测试 zvode 求解器
        for problem_cls in PROBLEMS:
            # 遍历所有的问题类别 PROBLEMS
            problem = problem_cls()
            # 创建一个具体的问题实例
            if not problem.stiff:
                # 如果问题不是刚性的
                self._do_problem(problem, 'zvode', 'adams')
                # 调用 _do_problem 方法，使用 'zvode' 求解器和 'adams' 方法求解问题
            self._do_problem(problem, 'zvode', 'bdf')
            # 调用 _do_problem 方法，使用 'zvode' 求解器和 'bdf' 方法求解问题

    def test_lsoda(self):
        # 测试 lsoda 求解器
        for problem_cls in PROBLEMS:
            # 遍历所有的问题类别 PROBLEMS
            problem = problem_cls()
            # 创建一个具体的问题实例
            if problem.cmplx:
                # 如果问题复杂，则跳过此问题
                continue
            self._do_problem(problem, 'lsoda')
            # 调用 _do_problem 方法，使用 'lsoda' 求解器求解问题

    def test_dopri5(self):
        # 测试 dopri5 求解器
        for problem_cls in PROBLEMS:
            # 遍历所有的问题类别 PROBLEMS
            problem = problem_cls()
            # 创建一个具体的问题实例
            if problem.cmplx:
                # 如果问题复杂，则跳过此问题
                continue
            if problem.stiff:
                # 如果问题是刚性的，则跳过此问题
                continue
            if hasattr(problem, 'jac'):
                # 如果问题具有 'jac' 属性，则跳过此问题
                continue
            self._do_problem(problem, 'dopri5')
            # 调用 _do_problem 方法，使用 'dopri5' 求解器求解问题

    def test_dop853(self):
        # 测试 dop853 求解器
        for problem_cls in PROBLEMS:
            # 遍历所有的问题类别 PROBLEMS
            problem = problem_cls()
            # 创建一个具体的问题实例
            if problem.cmplx:
                # 如果问题复杂，则跳过此问题
                continue
            if problem.stiff:
                # 如果问题是刚性的，则跳过此问题
                continue
            if hasattr(problem, 'jac'):
                # 如果问题具有 'jac' 属性，则跳过此问题
                continue
            self._do_problem(problem, 'dop853')
            # 调用 _do_problem 方法，使用 'dop853' 求解器求解问题

    def test_concurrent_fail(self):
        # 并发失败测试
        for sol in ('vode', 'zvode', 'lsoda'):
            # 遍历求解器列表 ('vode', 'zvode', 'lsoda')
            def f(t, y):
                return 1.0
                # 定义一个函数 f，返回固定值 1.0

            r = ode(f).set_integrator(sol)
            # 创建一个 ODE 求解器实例 r，使用给定的求解器 sol
            r.set_initial_value(0, 0)
            # 设置求解器的初始值

            r2 = ode(f).set_integrator(sol)
            # 创建另一个 ODE 求解器实例 r2，使用相同的求解器 sol
            r2.set_initial_value(0, 0)
            # 设置求解器的初始值

            r.integrate(r.t + 0.1)
            # 对求解器 r 进行积分计算，增加时间步长 0.1
            r2.integrate(r2.t + 0.1)
            # 对求解器 r2 进行积分计算，增加时间步长 0.1

            assert_raises(RuntimeError, r.integrate, r.t + 0.1)
            # 使用 assert_raises 断言，预期在 r 再次积分时会抛出 RuntimeError 异常
    # 定义一个测试函数，用于测试并发情况下的ODE求解器
    def test_concurrent_ok(self):
        # 定义一个简单的微分方程函数
        def f(t, y):
            return 1.0

        # 第一组测试：对于每种ODE求解器进行测试
        for k in range(3):
            for sol in ('vode', 'zvode', 'lsoda', 'dopri5', 'dop853'):
                # 创建ODE对象，选择特定的求解器
                r = ode(f).set_integrator(sol)
                # 设置初始值
                r.set_initial_value(0, 0)

                # 创建另一个ODE对象，选择相同的求解器
                r2 = ode(f).set_integrator(sol)
                r2.set_initial_value(0, 0)

                # 进行第一次积分计算
                r.integrate(r.t + 0.1)
                r2.integrate(r2.t + 0.1)
                r2.integrate(r2.t + 0.1)

                # 断言第一个ODE对象的结果应接近0.1
                assert_allclose(r.y, 0.1)
                # 断言第二个ODE对象的结果应接近0.2
                assert_allclose(r2.y, 0.2)

            # 第二组测试：针对一些特定的ODE求解器再次进行测试
            for sol in ('dopri5', 'dop853'):
                # 创建ODE对象，选择特定的求解器
                r = ode(f).set_integrator(sol)
                r.set_initial_value(0, 0)

                # 创建另一个ODE对象，选择相同的求解器
                r2 = ode(f).set_integrator(sol)
                r2.set_initial_value(0, 0)

                # 进行积分计算
                r.integrate(r.t + 0.1)
                r.integrate(r.t + 0.1)
                r2.integrate(r2.t + 0.1)
                r.integrate(r.t + 0.1)
                r2.integrate(r2.t + 0.1)

                # 断言第一个ODE对象的结果应接近0.3
                assert_allclose(r.y, 0.3)
                # 断言第二个ODE对象的结果应接近0.2
                assert_allclose(r2.y, 0.2)
class TestComplexOde(TestODEClass):

    ode_class = complex_ode

    def test_vode(self):
        # Check the vode solver
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if not problem.stiff:
                # Call _do_problem method with 'vode' solver for non-stiff problems
                self._do_problem(problem, 'vode', 'adams')
            else:
                # Call _do_problem method with 'vode' solver for stiff problems
                self._do_problem(problem, 'vode', 'bdf')

    def test_lsoda(self):
        # Check the lsoda solver
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            # Call _do_problem method with 'lsoda' solver
            self._do_problem(problem, 'lsoda')

    def test_dopri5(self):
        # Check the dopri5 solver
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if problem.stiff:
                continue
            if hasattr(problem, 'jac'):
                continue
            # Call _do_problem method with 'dopri5' solver for non-stiff problems without 'jac'
            self._do_problem(problem, 'dopri5')

    def test_dop853(self):
        # Check the dop853 solver
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if problem.stiff:
                continue
            if hasattr(problem, 'jac'):
                continue
            # Call _do_problem method with 'dop853' solver for non-stiff problems without 'jac'
            self._do_problem(problem, 'dop853')


class TestSolout:
    # Check integrate.ode correctly handles solout for dopri5 and dop853
    def _run_solout_test(self, integrator):
        # Check correct usage of solout
        ts = []
        ys = []
        t0 = 0.0
        tend = 10.0
        y0 = [1.0, 2.0]

        def solout(t, y):
            # Callback function to store time points 't' and corresponding state 'y'
            ts.append(t)
            ys.append(y.copy())

        def rhs(t, y):
            # Right-hand side function defining the ODE system
            return [y[0] + y[1], -y[1]**2]

        ig = ode(rhs).set_integrator(integrator)
        ig.set_solout(solout)
        ig.set_initial_value(y0, t0)
        # Integrate the ODE system up to 'tend'
        ret = ig.integrate(tend)
        # Assert initial state 'y0' matches the first recorded state
        assert_array_equal(ys[0], y0)
        # Assert final integrated state matches 'ret'
        assert_array_equal(ys[-1], ret)
        # Assert initial time 't0' matches the first recorded time
        assert_equal(ts[0], t0)
        # Assert final time 'tend' matches the last recorded time
        assert_equal(ts[-1], tend)

    def test_solout(self):
        for integrator in ('dopri5', 'dop853'):
            # Run solout test for 'dopri5' and 'dop853' integrators
            self._run_solout_test(integrator)

    def _run_solout_after_initial_test(self, integrator):
        # Check if solout works even if it is set after the initial value.
        ts = []
        ys = []
        t0 = 0.0
        tend = 10.0
        y0 = [1.0, 2.0]

        def solout(t, y):
            # Callback function to store time points 't' and corresponding state 'y'
            ts.append(t)
            ys.append(y.copy())

        def rhs(t, y):
            # Right-hand side function defining the ODE system
            return [y[0] + y[1], -y[1]**2]

        ig = ode(rhs).set_integrator(integrator)
        ig.set_initial_value(y0, t0)
        ig.set_solout(solout)
        # Integrate the ODE system up to 'tend'
        ret = ig.integrate(tend)
        # Assert initial state 'y0' matches the first recorded state
        assert_array_equal(ys[0], y0)
        # Assert final integrated state matches 'ret'
        assert_array_equal(ys[-1], ret)
        # Assert initial time 't0' matches the first recorded time
        assert_equal(ts[0], t0)
        # Assert final time 'tend' matches the last recorded time
        assert_equal(ts[-1], tend)

    def test_solout_after_initial(self):
        for integrator in ('dopri5', 'dop853'):
            # Run solout test (after setting initial value) for 'dopri5' and 'dop853' integrators
            self._run_solout_after_initial_test(integrator)
    # 定义测试函数，用于检验通过 solout 方法正确停止积分器
    def _run_solout_break_test(self, integrator):
        # 初始化时间列表和状态列表
        ts = []
        ys = []
        # 初始时间和结束时间
        t0 = 0.0
        tend = 10.0
        # 初始状态
        y0 = [1.0, 2.0]

        # 定义 solout 方法，用于在积分过程中记录时间和状态，并在特定条件下停止积分
        def solout(t, y):
            ts.append(t)  # 记录当前时间点
            ys.append(y.copy())  # 记录当前状态，使用副本以避免引用问题
            if t > tend/2.0:
                return -1  # 当时间超过一半时，返回 -1 以停止积分

        # 定义右端项函数
        def rhs(t, y):
            return [y[0] + y[1], -y[1]**2]

        # 创建积分器对象，并设置积分器类型
        ig = ode(rhs).set_integrator(integrator)
        ig.set_solout(solout)  # 设置 solout 方法用于处理每个时间步的输出
        ig.set_initial_value(y0, t0)  # 设置初始时间和状态
        ret = ig.integrate(tend)  # 进行积分，返回最终状态
        # 断言，检查初始状态和最终状态是否一致
        assert_array_equal(ys[0], y0)
        assert_array_equal(ys[-1], ret)
        # 断言，检查时间列表中的第一个时间是否等于初始时间
        assert_equal(ts[0], t0)
        # 断言，检查最后一个记录的时间是否大于结束时间的一半
        assert_(ts[-1] > tend/2.0)
        # 断言，检查最后一个记录的时间是否小于结束时间
        assert_(ts[-1] < tend)

    # 定义测试函数，测试不同积分器类型下的 solout 方法停止功能
    def test_solout_break(self):
        # 遍历测试不同的积分器类型
        for integrator in ('dopri5', 'dop853'):
            self._run_solout_break_test(integrator)
class TestComplexSolout:
    # Check integrate.ode correctly handles solout for dopri5 and dop853

    def _run_solout_test(self, integrator):
        # Check correct usage of solout
        ts = []  # 用于存储时间步长
        ys = []  # 用于存储状态变量
        t0 = 0.0  # 初始时间
        tend = 20.0  # 最终时间
        y0 = [0.0]  # 初始状态向量

        def solout(t, y):
            ts.append(t)  # 记录时间步长
            ys.append(y.copy())  # 记录状态变量的拷贝

        def rhs(t, y):
            return [1.0/(t - 10.0 - 1j)]  # 定义右手边函数

        ig = complex_ode(rhs).set_integrator(integrator)  # 创建复数ODE对象
        ig.set_solout(solout)  # 设置solout函数
        ig.set_initial_value(y0, t0)  # 设置初始状态和时间
        ret = ig.integrate(tend)  # 进行积分计算
        assert_array_equal(ys[0], y0)  # 断言初始状态正确
        assert_array_equal(ys[-1], ret)  # 断言最终状态正确
        assert_equal(ts[0], t0)  # 断言起始时间正确
        assert_equal(ts[-1], tend)  # 断言终止时间正确

    def test_solout(self):
        for integrator in ('dopri5', 'dop853'):
            self._run_solout_test(integrator)

    def _run_solout_break_test(self, integrator):
        # Check correct usage of stopping via solout
        ts = []  # 用于存储时间步长
        ys = []  # 用于存储状态变量
        t0 = 0.0  # 初始时间
        tend = 20.0  # 最终时间
        y0 = [0.0]  # 初始状态向量

        def solout(t, y):
            ts.append(t)  # 记录时间步长
            ys.append(y.copy())  # 记录状态变量的拷贝
            if t > tend/2.0:
                return -1  # 当时间超过一半时停止积分

        def rhs(t, y):
            return [1.0/(t - 10.0 - 1j)]  # 定义右手边函数

        ig = complex_ode(rhs).set_integrator(integrator)  # 创建复数ODE对象
        ig.set_solout(solout)  # 设置solout函数
        ig.set_initial_value(y0, t0)  # 设置初始状态和时间
        ret = ig.integrate(tend)  # 进行积分计算
        assert_array_equal(ys[0], y0)  # 断言初始状态正确
        assert_array_equal(ys[-1], ret)  # 断言最终状态正确
        assert_equal(ts[0], t0)  # 断言起始时间正确
        assert_(ts[-1] > tend/2.0)  # 断言停止时间大于一半
        assert_(ts[-1] < tend)  # 断言停止时间小于最终时间

    def test_solout_break(self):
        for integrator in ('dopri5', 'dop853'):
            self._run_solout_break_test(integrator)


#------------------------------------------------------------------------------
# Test problems
#------------------------------------------------------------------------------

class ODE:
    """
    ODE problem
    """
    stiff = False
    cmplx = False
    stop_t = 1
    z0 = []

    lband = None
    uband = None

    atol = 1e-6
    rtol = 1e-5


class SimpleOscillator(ODE):
    r"""
    Free vibration of a simple oscillator::
        m \ddot{u} + k u = 0, u(0) = u_0 \dot{u}(0) \dot{u}_0
    Solution::
        u(t) = u_0*cos(sqrt(k/m)*t)+\dot{u}_0*sin(sqrt(k/m)*t)/sqrt(k/m)
    """
    stop_t = 1 + 0.09
    z0 = array([1.0, 0.1], float)

    k = 4.0
    m = 1.0

    def f(self, z, t):
        tmp = zeros((2, 2), float)
        tmp[0, 1] = 1.0
        tmp[1, 0] = -self.k / self.m
        return dot(tmp, z)

    def verify(self, zs, t):
        omega = sqrt(self.k / self.m)
        u = self.z0[0]*cos(omega*t) + self.z0[1]*sin(omega*t)/omega
        return allclose(u, zs[:, 0], atol=self.atol, rtol=self.rtol)


class ComplexExp(ODE):
    r"""The equation :lm:`\dot u = i u`"""
    stop_t = 1.23*pi
    z0 = exp([1j, 2j, 3j, 4j, 5j])
    cmplx = True

    def f(self, z, t):
        return 1j*z

    def jac(self, z, t):
        return 1j*eye(5)
    # 定义一个类方法 `verify`，用于验证复数值在指定条件下的近似性
    def verify(self, zs, t):
        # 计算复数值 u，使用初始复数 self.z0 和指数函数 exp(1j*t)
        u = self.z0 * exp(1j*t)
        # 使用 NumPy 函数 `allclose` 检查复数值 u 和 zs 是否在给定的容差范围内近似相等
        return allclose(u, zs, atol=self.atol, rtol=self.rtol)
class Pi(ODE):
    r"""Integrate 1/(t + 1j) from t=-10 to t=10"""
    stop_t = 20  # 设置积分的终止时间为20
    z0 = [0]  # 设置初始条件
    cmplx = True  # 标记为复数积分

    def f(self, z, t):
        return array([1./(t - 10 + 1j)])  # 定义微分方程的右手边函数

    def verify(self, zs, t):
        u = -2j * np.arctan(10)  # 计算验证结果的期望值
        return allclose(u, zs[-1, :], atol=self.atol, rtol=self.rtol)


class CoupledDecay(ODE):
    r"""
    3 coupled decays suited for banded treatment
    (banded mode makes it necessary when N>>3)
    """

    stiff = True  # 标记此ODE问题为刚性问题
    stop_t = 0.5  # 设置积分的终止时间为0.5
    z0 = [5.0, 7.0, 13.0]  # 设置初始条件向量
    lband = 1  # 设置下带宽
    uband = 0  # 设置上带宽

    lmbd = [0.17, 0.23, 0.29]  # 设置虚构的衰减常数

    def f(self, z, t):
        lmbd = self.lmbd
        return np.array([-lmbd[0]*z[0],
                         -lmbd[1]*z[1] + lmbd[0]*z[0],
                         -lmbd[2]*z[2] + lmbd[1]*z[1]])  # 定义微分方程的右手边函数

    def jac(self, z, t):
        # 定义雅可比矩阵函数，返回压缩存储格式的雅可比矩阵
        lmbd = self.lmbd
        j = np.zeros((self.lband + self.uband + 1, 3), order='F')

        def set_j(ri, ci, val):
            j[self.uband + ri - ci, ci] = val

        set_j(0, 0, -lmbd[0])
        set_j(1, 0, lmbd[0])
        set_j(1, 1, -lmbd[1])
        set_j(2, 1, lmbd[1])
        set_j(2, 2, -lmbd[2])
        return j

    def verify(self, zs, t):
        # 手工推导出的验证公式
        lmbd = np.array(self.lmbd)
        d10 = lmbd[1] - lmbd[0]
        d21 = lmbd[2] - lmbd[1]
        d20 = lmbd[2] - lmbd[0]
        e0 = np.exp(-lmbd[0] * t)
        e1 = np.exp(-lmbd[1] * t)
        e2 = np.exp(-lmbd[2] * t)
        u = np.vstack((
            self.z0[0] * e0,
            self.z0[1] * e1 + self.z0[0] * lmbd[0] / d10 * (e0 - e1),
            self.z0[2] * e2 + self.z0[1] * lmbd[1] / d21 * (e1 - e2) +
            lmbd[1] * lmbd[0] * self.z0[0] / d10 *
            (1 / d20 * (e0 - e2) - 1 / d21 * (e1 - e2)))).transpose()
        return allclose(u, zs, atol=self.atol, rtol=self.rtol)


PROBLEMS = [SimpleOscillator, ComplexExp, Pi, CoupledDecay]

#------------------------------------------------------------------------------

# 下面是一些独立的函数定义，与前面的类定义无直接关系

def f(t, x):
    dxdt = [x[1], -x[0]]  # 定义一个简单的一阶微分方程组
    return dxdt


def jac(t, x):
    j = array([[0.0, 1.0],
               [-1.0, 0.0]])  # 返回给定微分方程组的雅可比矩阵
    return j


def f1(t, x, omega):
    dxdt = [omega*x[1], -omega*x[0]]  # 定义一个带参数的一阶微分方程组
    return dxdt


def jac1(t, x, omega):
    j = array([[0.0, omega],
               [-omega, 0.0]])  # 返回给定带参数微分方程组的雅可比矩阵
    return j


def f2(t, x, omega1, omega2):
    dxdt = [omega1*x[1], -omega2*x[0]]  # 定义一个带多个参数的一阶微分方程组
    return dxdt


def jac2(t, x, omega1, omega2):
    j = array([[0.0, omega1],
               [-omega2, 0.0]])  # 返回给定带多个参数微分方程组的雅可比矩阵
    return j


def fv(t, x, omega):
    dxdt = [omega[0]*x[1], -omega[1]*x[0]]  # 定义一个使用向量作为参数的一阶微分方程组
    return dxdt
def jacv(t, x, omega):
    # 定义雅可比矩阵，根据给定的角速度 omega
    j = array([[0.0, omega[0]],
               [-omega[1], 0.0]])
    return j


class ODECheckParameterUse:
    """调用带有多种参数用法的 ODE 类求解器。"""

    # 在运行该类的测试之前，必须设置 solver_name。

    # 在子类中设置这些参数。
    solver_name = ''
    solver_uses_jac = False

    def _get_solver(self, f, jac):
        # 创建一个 ODE 求解器，传入函数 f 和雅可比矩阵 jac。
        solver = ode(f, jac)
        if self.solver_uses_jac:
            # 如果求解器需要雅可比矩阵，则设置积分器并指定公差和相对误差。
            solver.set_integrator(self.solver_name, atol=1e-9, rtol=1e-7,
                                  with_jacobian=self.solver_uses_jac)
        else:
            # 否则，只设置积分器并指定公差和相对误差。
            # XXX 如果求解器无法使用 with_jacobian 参数，应该始终接受此关键字参数，
            # 并且如果设置为 True，可能会引发异常。
            solver.set_integrator(self.solver_name, atol=1e-9, rtol=1e-7)
        return solver

    def _check_solver(self, solver):
        # 设置初始条件并积分直到 t = pi。
        ic = [1.0, 0.0]
        solver.set_initial_value(ic, 0.0)
        solver.integrate(pi)
        # 检查积分结果是否与预期值接近。
        assert_array_almost_equal(solver.y, [-1.0, 0.0])

    def test_no_params(self):
        # 测试不带参数的情况。
        solver = self._get_solver(f, jac)
        self._check_solver(solver)

    def test_one_scalar_param(self):
        # 测试只有一个标量参数的情况。
        solver = self._get_solver(f1, jac1)
        omega = 1.0
        solver.set_f_params(omega)
        if self.solver_uses_jac:
            solver.set_jac_params(omega)
        self._check_solver(solver)

    def test_two_scalar_params(self):
        # 测试有两个标量参数的情况。
        solver = self._get_solver(f2, jac2)
        omega1 = 1.0
        omega2 = 1.0
        solver.set_f_params(omega1, omega2)
        if self.solver_uses_jac:
            solver.set_jac_params(omega1, omega2)
        self._check_solver(solver)

    def test_vector_param(self):
        # 测试使用向量参数的情况。
        solver = self._get_solver(fv, jacv)
        omega = [1.0, 1.0]
        solver.set_f_params(omega)
        if self.solver_uses_jac:
            solver.set_jac_params(omega)
        self._check_solver(solver)

    def test_warns_on_failure(self):
        # 设置 nsteps 很小以确保失败时发出警告。
        solver = self._get_solver(f, jac)
        solver.set_integrator(self.solver_name, nsteps=1)
        ic = [1.0, 0.0]
        solver.set_initial_value(ic, 0.0)
        # 断言确保在积分过程中失败时会引发 UserWarning。
        assert_warns(UserWarning, solver.integrate, pi)


class TestDOPRI5CheckParameterUse(ODECheckParameterUse):
    solver_name = 'dopri5'
    solver_uses_jac = False


class TestDOP853CheckParameterUse(ODECheckParameterUse):
    solver_name = 'dop853'
    solver_uses_jac = False


class TestVODECheckParameterUse(ODECheckParameterUse):
    solver_name = 'vode'
    solver_uses_jac = True


class TestZVODECheckParameterUse(ODECheckParameterUse):
    solver_name = 'zvode'
    solver_uses_jac = True


class TestLSODACheckParameterUse(ODECheckParameterUse):
    solver_name = 'lsoda'
    solver_uses_jac = True


def test_odeint_trivial_time():
    # 测试当只给定一个时间点时 odeint 是否能成功运行。
    # 初始化初始条件 y0 为 1
    y0 = 1
    # 初始化时间点 t 为 [0]
    t = [0]
    # 使用 odeint 求解微分方程 dy/dt = -y，返回结果 y 和附加信息 info
    y, info = odeint(lambda y, t: -y, y0, t, full_output=True)
    # 使用 assert_array_equal 断言函数验证 y 是否等于 np.array([[y0]])，用于回归测试 gh-4282
    assert_array_equal(y, np.array([[y0]]))
# 定义一个用于测试 `odeint` 函数的函数，测试其使用 `Dfun`、`ml` 和 `mu` 选项。

def test_odeint_banded_jacobian():
    # 定义一个函数 `func`，接受参数 `y`, `t`, `c`，返回 `c` 与 `y` 的矩阵乘积
    def func(y, t, c):
        return c.dot(y)

    # 定义一个函数 `jac`，接受参数 `y`, `t`, `c`，返回常数矩阵 `c`
    def jac(y, t, c):
        return c

    # 定义一个函数 `jac_transpose`，接受参数 `y`, `t`, `c`，返回常数矩阵 `c` 的转置
    def jac_transpose(y, t, c):
        return c.T.copy(order='C')

    # 定义一个函数 `bjac_rows`，接受参数 `y`, `t`, `c`，返回带状雅可比矩阵
    def bjac_rows(y, t, c):
        jac = np.vstack((np.r_[0, np.diag(c, 1)],
                            np.diag(c),
                            np.r_[np.diag(c, -1), 0],
                            np.r_[np.diag(c, -2), 0, 0]))
        return jac

    # 定义一个函数 `bjac_cols`，接受参数 `y`, `t`, `c`，返回带状雅可比矩阵的转置
    def bjac_cols(y, t, c):
        return bjac_rows(y, t, c).T.copy(order='C')

    # 定义一个常数矩阵 `c`
    c = array([[-205, 0.01, 0.00, 0.0],
               [0.1, -2.50, 0.02, 0.0],
               [1e-3, 0.01, -2.0, 0.01],
               [0.00, 0.00, 0.1, -1.0]])

    # 初始化初始向量 `y0`
    y0 = np.ones(4)
    # 定义时间点向量 `t`
    t = np.array([0, 5, 10, 100])

    # 使用完整的雅可比矩阵 `jac` 进行求解
    sol1, info1 = odeint(func, y0, t, args=(c,), full_output=True,
                         atol=1e-13, rtol=1e-11, mxstep=10000,
                         Dfun=jac)

    # 使用转置后的完整雅可比矩阵 `jac_transpose`，并设置 `col_deriv=True` 进行求解
    sol2, info2 = odeint(func, y0, t, args=(c,), full_output=True,
                         atol=1e-13, rtol=1e-11, mxstep=10000,
                         Dfun=jac_transpose, col_deriv=True)

    # 使用带状雅可比矩阵 `bjac_rows` 进行求解
    sol3, info3 = odeint(func, y0, t, args=(c,), full_output=True,
                         atol=1e-13, rtol=1e-11, mxstep=10000,
                         Dfun=bjac_rows, ml=2, mu=1)

    # 使用转置后的带状雅可比矩阵 `bjac_cols`，并设置 `col_deriv=True` 进行求解
    sol4, info4 = odeint(func, y0, t, args=(c,), full_output=True,
                         atol=1e-13, rtol=1e-11, mxstep=10000,
                         Dfun=bjac_cols, ml=2, mu=1, col_deriv=True)

    # 断言 `sol1` 与 `sol2` 的近似相等，用于验证解的正确性
    assert_allclose(sol1, sol2, err_msg="sol1 != sol2")
    # 断言 `sol1` 与 `sol3` 的近似相等，用于验证解的正确性
    assert_allclose(sol1, sol3, atol=1e-12, err_msg="sol1 != sol3")
    # 断言 `sol3` 与 `sol4` 的近似相等，用于验证解的正确性
    assert_allclose(sol3, sol4, err_msg="sol3 != sol4")

    # 验证使用完整雅可比矩阵与使用带状雅可比矩阵时雅可比矩阵计算次数相同，用于回归测试
    assert_array_equal(info1['nje'], info2['nje'])
    assert_array_equal(info3['nje'], info4['nje'])

    # 测试 `tfirst` 参数的使用
    sol1ty, info1ty = odeint(lambda t, y, c: func(y, t, c), y0, t, args=(c,),
                             full_output=True, atol=1e-13, rtol=1e-11,
                             mxstep=10000,
                             Dfun=lambda t, y, c: jac(y, t, c), tfirst=True)
    # 由于浮点数计算的确定性，这些结果应该完全相等，使用小的相对容差进行验证
    assert_allclose(sol1, sol1ty, rtol=1e-12, err_msg="sol1 != sol1ty")


def test_odeint_errors():
    # 定义一个一维系统的微分方程，该系统的微分方程为 dx/dt = -100*x
    def sys1d(x, t):
        return -100*x
    
    # 定义一个会产生除以零错误的函数
    def bad1(x, t):
        return 1.0/0
    
    # 定义一个返回字符串的函数，通常情况下无法作为微分方程函数
    def bad2(x, t):
        return "foo"
    
    # 定义一个会产生除以零错误的雅可比矩阵函数
    def bad_jac1(x, t):
        return 1.0/0
    
    # 定义一个返回嵌套列表的雅可比矩阵函数，通常情况下无法作为雅可比矩阵
    def bad_jac2(x, t):
        return [["foo"]]
    
    # 定义一个二维系统的微分方程，该系统的微分方程为 dx0/dt = -100*x0, dx1/dt = -0.1*x1
    def sys2d(x, t):
        return [-100*x[0], -0.1*x[1]]
    
    # 定义一个会产生除以零错误的二维系统的雅可比矩阵函数
    def sys2d_bad_jac(x, t):
        return [[1.0/0, 0], [0, -0.1]]
    
    # 使用 assert_raises 断言来验证调用 odeint 函数时，会抛出 ZeroDivisionError 异常，因为 bad1 函数会除以零
    assert_raises(ZeroDivisionError, odeint, bad1, 1.0, [0, 1])
    
    # 使用 assert_raises 断言来验证调用 odeint 函数时，会抛出 ValueError 异常，因为 bad2 函数返回一个字符串而不是数值
    assert_raises(ValueError, odeint, bad2, 1.0, [0, 1])
    
    # 使用 assert_raises 断言来验证调用 odeint 函数时，会抛出 ZeroDivisionError 异常，因为 sys1d 函数的雅可比矩阵 bad_jac1 函数会除以零
    assert_raises(ZeroDivisionError, odeint, sys1d, 1.0, [0, 1], Dfun=bad_jac1)
    
    # 使用 assert_raises 断言来验证调用 odeint 函数时，会抛出 ValueError 异常，因为 sys1d 函数的雅可比矩阵 bad_jac2 函数返回一个嵌套列表而不是数值
    assert_raises(ValueError, odeint, sys1d, 1.0, [0, 1], Dfun=bad_jac2)
    
    # 使用 assert_raises 断言来验证调用 odeint 函数时，会抛出 ZeroDivisionError 异常，因为 sys2d 函数在 t=0 时的雅可比矩阵 sys2d_bad_jac 函数会除以零
    assert_raises(ZeroDivisionError, odeint, sys2d, [1.0, 1.0], [0, 1], Dfun=sys2d_bad_jac)
def test_odeint_bad_shapes():
    # Tests of some errors that can occur with odeint.

    # 定义一个简单的右手边函数，返回固定的列表 [1, -1]
    def badrhs(x, t):
        return [1, -1]

    # 定义一个简单的系统函数，返回固定值 -100*x
    def sys1(x, t):
        return -100*x

    # 定义一个错误的雅可比矩阵函数，返回一个不正确的二维数组 [[0, 0, 0]]
    def badjac(x, t):
        return [[0, 0, 0]]

    # y0 必须是最多一维的数组或列表，这里是一个错误的二维数组
    bad_y0 = [[0, 0], [0, 0]]
    # 断言会抛出 ValueError 异常，因为 y0 的维度不正确
    assert_raises(ValueError, odeint, sys1, bad_y0, [0, 1])

    # t 必须是最多一维的数组或列表，这里是一个错误的二维数组
    bad_t = [[0, 1], [2, 3]]
    # 断言会抛出 ValueError 异常，因为 t 的维度不正确
    assert_raises(ValueError, odeint, sys1, [10.0], bad_t)

    # y0 是标量 10，但 badrhs(x, t) 返回一个长度为 2 的列表 [1, -1]
    # 这里期望会抛出 RuntimeError
    assert_raises(RuntimeError, odeint, badrhs, 10, [0, 1])

    # badjac(x, t) 返回的数组形状不正确
    # 断言会抛出 RuntimeError
    assert_raises(RuntimeError, odeint, sys1, [10, 10], [0, 1], Dfun=badjac)


def test_repeated_t_values():
    """Regression test for gh-8217."""

    # 定义一个简单的微分方程函数 func(x, t) = -0.25*x
    def func(x, t):
        return -0.25*x

    # 创建一个长度为 10 的零数组作为时间点 t
    t = np.zeros(10)
    # 使用 odeint 求解微分方程 func，初始条件为 [1.]，时间点为 t
    sol = odeint(func, [1.], t)
    # 断言求解结果 sol 等于一个元素全为 1 的数组
    assert_array_equal(sol, np.ones((len(t), 1)))

    # 计算常数 tau = 4 * ln(2)
    tau = 4*np.log(2)
    # 创建一个时间点 t，其中包含重复值
    t = [0]*9 + [tau, 2*tau, 2*tau, 3*tau]
    # 使用 odeint 求解微分方程 func，初始条件为 [1, 2]，时间点为 t
    # 设置相对和绝对误差容限为 1e-12
    sol = odeint(func, [1, 2], t, rtol=1e-12, atol=1e-12)
    # 期望的解 sol，是一个与 t 长度相同的数组，每行是 func 的计算结果
    expected_sol = np.array([[1.0, 2.0]]*9 +
                            [[0.5, 1.0],
                             [0.25, 0.5],
                             [0.25, 0.5],
                             [0.125, 0.25]])
    # 断言求解结果 sol 等于期望的解 expected_sol
    assert_allclose(sol, expected_sol)

    # 边界情况：空的时间序列 t
    sol = odeint(func, [1.], [])
    # 断言求解结果 sol 是一个空的浮点64位数组
    assert_array_equal(sol, np.array([], dtype=np.float64).reshape((0, 1)))

    # 时间点 t 不是单调递增的情况
    # 断言会抛出 ValueError 异常，因为 t 不是单调递增的
    assert_raises(ValueError, odeint, func, [1.], [0, 1, 0.5, 0])
    # 断言会抛出 ValueError 异常，因为 t 不是单调递增的
    assert_raises(ValueError, odeint, func, [1, 2, 3], [0, -1, -2, 3])
```