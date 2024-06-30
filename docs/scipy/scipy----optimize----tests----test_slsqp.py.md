# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_slsqp.py`

```
"""
Unit test for SLSQP optimization.
"""
# 导入需要的测试工具和断言方法
from numpy.testing import (assert_, assert_array_almost_equal,
                           assert_allclose, assert_equal)
from pytest import raises as assert_raises
import pytest
import numpy as np

# 导入优化相关的函数和类
from scipy.optimize import fmin_slsqp, minimize, Bounds, NonlinearConstraint


class MyCallBack:
    """pass a custom callback function

    This makes sure it's being used.
    """
    def __init__(self):
        self.been_called = False
        self.ncalls = 0

    def __call__(self, x):
        self.been_called = True
        self.ncalls += 1


class TestSLSQP:
    """
    Test SLSQP algorithm using Example 14.4 from Numerical Methods for
    Engineers by Steven Chapra and Raymond Canale.
    This example maximizes the function f(x) = 2*x*y + 2*x - x**2 - 2*y**2,
    which has a maximum at x=2, y=1.
    """
    def setup_method(self):
        # 设置测试方法的选项，禁止显示优化过程信息
        self.opts = {'disp': False}

    def fun(self, d, sign=1.0):
        """
        Arguments:
        d     - A list of two elements, where d[0] represents x and d[1] represents y
                 in the following equation.
        sign - A multiplier for f. Since we want to optimize it, and the SciPy
               optimizers can only minimize functions, we need to multiply it by
               -1 to achieve the desired solution
        Returns:
        2*x*y + 2*x - x**2 - 2*y**2

        """
        x = d[0]
        y = d[1]
        # 计算函数值
        return sign*(2*x*y + 2*x - x**2 - 2*y**2)

    def jac(self, d, sign=1.0):
        """
        This is the derivative of fun, returning a NumPy array
        representing df/dx and df/dy.

        """
        x = d[0]
        y = d[1]
        # 计算梯度向量
        dfdx = sign*(-2*x + 2*y + 2)
        dfdy = sign*(2*x - 4*y)
        return np.array([dfdx, dfdy], float)

    def fun_and_jac(self, d, sign=1.0):
        # 返回函数值和梯度
        return self.fun(d, sign), self.jac(d, sign)

    def f_eqcon(self, x, sign=1.0):
        """ Equality constraint """
        # 返回等式约束
        return np.array([x[0] - x[1]])

    def fprime_eqcon(self, x, sign=1.0):
        """ Equality constraint, derivative """
        # 返回等式约束的雅可比矩阵
        return np.array([[1, -1]])

    def f_eqcon_scalar(self, x, sign=1.0):
        """ Scalar equality constraint """
        # 返回等式约束的标量形式
        return self.f_eqcon(x, sign)[0]

    def fprime_eqcon_scalar(self, x, sign=1.0):
        """ Scalar equality constraint, derivative """
        # 返回等式约束标量形式的雅可比向量
        return self.fprime_eqcon(x, sign)[0].tolist()

    def f_ieqcon(self, x, sign=1.0):
        """ Inequality constraint """
        # 返回不等式约束
        return np.array([x[0] - x[1] - 1.0])

    def fprime_ieqcon(self, x, sign=1.0):
        """ Inequality constraint, derivative """
        # 返回不等式约束的雅可比矩阵
        return np.array([[1, -1]])

    def f_ieqcon2(self, x):
        """ Vector inequality constraint """
        # 返回向量形式的不等式约束
        return np.asarray(x)

    def fprime_ieqcon2(self, x):
        """ Vector inequality constraint, derivative """
        # 返回向量形式不等式约束的单位矩阵
        return np.identity(x.shape[0])

    # minimize
    def test_minimize_unbounded_approximated(self):
        # 使用 SLSQP 方法进行最小化，无界问题，使用近似雅可比矩阵。
        # 定义可能的雅可比选项
        jacs = [None, False, '2-point', '3-point']
        # 遍历所有雅可比选项
        for jac in jacs:
            # 调用 minimize 函数进行最小化优化
            res = minimize(self.fun, [-1.0, 1.0], args=(-1.0, ),
                           jac=jac, method='SLSQP',
                           options=self.opts)
            # 断言优化成功
            assert_(res['success'], res['message'])
            # 断言优化结果接近预期值 [2, 1]
            assert_allclose(res.x, [2, 1])

    def test_minimize_unbounded_given(self):
        # 使用 SLSQP 方法进行最小化，无界问题，指定雅可比矩阵。
        # 调用 minimize 函数进行最小化优化
        res = minimize(self.fun, [-1.0, 1.0], args=(-1.0, ),
                       jac=self.jac, method='SLSQP', options=self.opts)
        # 断言优化成功
        assert_(res['success'], res['message'])
        # 断言优化结果接近预期值 [2, 1]
        assert_allclose(res.x, [2, 1])

    def test_minimize_bounded_approximated(self):
        # 使用 SLSQP 方法进行最小化，有界问题，使用近似雅可比矩阵。
        # 定义可能的雅可比选项
        jacs = [None, False, '2-point', '3-point']
        # 遍历所有雅可比选项
        for jac in jacs:
            # 调用 minimize 函数进行最小化优化
            with np.errstate(invalid='ignore'):
                res = minimize(self.fun, [-1.0, 1.0], args=(-1.0, ),
                               jac=jac,
                               bounds=((2.5, None), (None, 0.5)),
                               method='SLSQP', options=self.opts)
            # 断言优化成功
            assert_(res['success'], res['message'])
            # 断言优化结果接近预期值 [2.5, 0.5]
            assert_allclose(res.x, [2.5, 0.5])
            # 断言第一个优化结果大于等于 2.5
            assert_(2.5 <= res.x[0])
            # 断言第二个优化结果小于等于 0.5
            assert_(res.x[1] <= 0.5)

    def test_minimize_unbounded_combined(self):
        # 使用 SLSQP 方法进行最小化，无界问题，同时使用函数和雅可比矩阵。
        # 调用 minimize 函数进行最小化优化
        res = minimize(self.fun_and_jac, [-1.0, 1.0], args=(-1.0, ),
                       jac=True, method='SLSQP', options=self.opts)
        # 断言优化成功
        assert_(res['success'], res['message'])
        # 断言优化结果接近预期值 [2, 1]
        assert_allclose(res.x, [2, 1])

    def test_minimize_equality_approximated(self):
        # 使用 SLSQP 方法进行最小化，等式约束问题，使用近似雅可比矩阵。
        # 定义可能的雅可比选项
        jacs = [None, False, '2-point', '3-point']
        # 遍历所有雅可比选项
        for jac in jacs:
            # 调用 minimize 函数进行最小化优化
            res = minimize(self.fun, [-1.0, 1.0], args=(-1.0, ),
                           jac=jac,
                           constraints={'type': 'eq',
                                        'fun': self.f_eqcon,
                                        'args': (-1.0, )},
                           method='SLSQP', options=self.opts)
            # 断言优化成功
            assert_(res['success'], res['message'])
            # 断言优化结果接近预期值 [1, 1]
            assert_allclose(res.x, [1, 1])

    def test_minimize_equality_given(self):
        # 使用 SLSQP 方法进行最小化，等式约束问题，指定雅可比矩阵。
        # 调用 minimize 函数进行最小化优化
        res = minimize(self.fun, [-1.0, 1.0], jac=self.jac,
                       method='SLSQP', args=(-1.0,),
                       constraints={'type': 'eq', 'fun': self.f_eqcon,
                                    'args': (-1.0, )},
                       options=self.opts)
        # 断言优化成功
        assert_(res['success'], res['message'])
        # 断言优化结果接近预期值 [1, 1]
        assert_allclose(res.x, [1, 1])
    def test_minimize_equality_given2(self):
        # 使用 SLSQP 方法进行最小化：给定雅可比矩阵的等式约束。
        res = minimize(self.fun, [-1.0, 1.0], method='SLSQP',
                       jac=self.jac, args=(-1.0,),
                       constraints={'type': 'eq',
                                    'fun': self.f_eqcon,
                                    'args': (-1.0, ),
                                    'jac': self.fprime_eqcon},
                       options=self.opts)
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [1, 1])

    def test_minimize_equality_given_cons_scalar(self):
        # 使用 SLSQP 方法进行最小化：标量等式约束，给定雅可比矩阵的函数和约束。
        res = minimize(self.fun, [-1.0, 1.0], method='SLSQP',
                       jac=self.jac, args=(-1.0,),
                       constraints={'type': 'eq',
                                    'fun': self.f_eqcon_scalar,
                                    'args': (-1.0, ),
                                    'jac': self.fprime_eqcon_scalar},
                       options=self.opts)
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [1, 1])

    def test_minimize_inequality_given(self):
        # 使用 SLSQP 方法进行最小化：给定雅可比矩阵的不等式约束。
        res = minimize(self.fun, [-1.0, 1.0], method='SLSQP',
                       jac=self.jac, args=(-1.0, ),
                       constraints={'type': 'ineq',
                                    'fun': self.f_ieqcon,
                                    'args': (-1.0, )},
                       options=self.opts)
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [2, 1], atol=1e-3)

    def test_minimize_inequality_given_vector_constraints(self):
        # 使用 SLSQP 方法进行最小化：向量不等式约束，给定雅可比矩阵。
        res = minimize(self.fun, [-1.0, 1.0], jac=self.jac,
                       method='SLSQP', args=(-1.0,),
                       constraints={'type': 'ineq',
                                    'fun': self.f_ieqcon2,
                                    'jac': self.fprime_ieqcon2},
                       options=self.opts)
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [2, 1])
    # 定义一个测试函数，用于测试带有有界约束的最小化问题
    def test_minimize_bounded_constraint(self):
        # 定义约束条件函数 c(x)，确保 x[0] 和 x[1] 在 [0, 1] 范围内
        def c(x):
            assert 0 <= x[0] <= 1 and 0 <= x[1] <= 1, x
            return x[0] ** 0.5 + x[1]

        # 定义目标函数 f(x)，确保 x[0] 和 x[1] 在 [0, 1] 范围内
        def f(x):
            assert 0 <= x[0] <= 1 and 0 <= x[1] <= 1, x
            return -x[0] ** 2 + x[1] ** 2

        # 创建一个非线性约束列表，包含约束函数 c(x) 和约束范围 [0, 1.5]
        cns = [NonlinearConstraint(c, 0, 1.5)]
        # 初始化起始点 x0
        x0 = np.asarray([0.9, 0.5])
        # 定义边界条件，x[0] 和 x[1] 的范围分别为 [0, 1.0]
        bnd = Bounds([0., 0.], [1.0, 1.0])
        # 调用 minimize 函数进行最小化，使用 SLSQP 方法，传入目标函数、起始点、边界条件和约束条件
        minimize(f, x0, method='SLSQP', bounds=bnd, constraints=cns)

    # 定义一个测试函数，用于测试带有边界和等式约束的最小化问题
    def test_minimize_bound_equality_given2(self):
        # 调用 minimize 函数进行最小化，使用 SLSQP 方法，传入目标函数、初始点、雅可比矩阵、边界条件和约束条件
        res = minimize(self.fun, [-1.0, 1.0], method='SLSQP',
                       jac=self.jac, args=(-1.0, ),
                       bounds=[(-0.8, 1.), (-1, 0.8)],
                       constraints={'type': 'eq',
                                    'fun': self.f_eqcon,
                                    'args': (-1.0, ),
                                    'jac': self.fprime_eqcon},
                       options=self.opts)
        # 断言最小化结果的成功性和精度
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [0.8, 0.8], atol=1e-3)
        # 断言结果 x[0] 和 x[1] 在指定的边界范围内
        assert_(-0.8 <= res.x[0] <= 1)
        assert_(-1 <= res.x[1] <= 0.8)

    # 定义一个测试函数，用于测试未约束的最小化问题（使用 fmin_slsqp 函数）
    def test_unbounded_approximated(self):
        # 调用 fmin_slsqp 函数进行最小化，传入目标函数、初始点、参数、输出参数和控制选项
        res = fmin_slsqp(self.fun, [-1.0, 1.0], args=(-1.0, ),
                         iprint=0, full_output=1)
        # 解析结果变量
        x, fx, its, imode, smode = res
        # 断言模式为 0，即最优解找到
        assert_(imode == 0, imode)
        # 断言结果 x 与期望值接近
        assert_array_almost_equal(x, [2, 1])

    # 定义一个测试函数，用于测试未约束的最小化问题（使用 fmin_slsqp 函数）
    def test_unbounded_given(self):
        # 调用 fmin_slsqp 函数进行最小化，传入目标函数、初始点、参数、导数、输出参数和控制选项
        res = fmin_slsqp(self.fun, [-1.0, 1.0], args=(-1.0, ),
                         fprime=self.jac, iprint=0,
                         full_output=1)
        # 解析结果变量
        x, fx, its, imode, smode = res
        # 断言模式为 0，即最优解找到
        assert_(imode == 0, imode)
        # 断言结果 x 与期望值接近
        assert_array_almost_equal(x, [2, 1])

    # 定义一个测试函数，用于测试带有等式约束的最小化问题（使用 fmin_slsqp 函数）
    def test_equality_approximated(self):
        # 调用 fmin_slsqp 函数进行最小化，传入目标函数、初始点、参数、等式约束函数、输出参数和控制选项
        res = fmin_slsqp(self.fun, [-1.0, 1.0], args=(-1.0,),
                         eqcons=[self.f_eqcon],
                         iprint=0, full_output=1)
        # 解析结果变量
        x, fx, its, imode, smode = res
        # 断言模式为 0，即最优解找到
        assert_(imode == 0, imode)
        # 断言结果 x 与期望值接近
        assert_array_almost_equal(x, [1, 1])

    # 定义一个测试函数，用于测试带有等式约束的最小化问题（使用 fmin_slsqp 函数）
    def test_equality_given(self):
        # 调用 fmin_slsqp 函数进行最小化，传入目标函数、初始点、导数、参数、等式约束函数、输出参数和控制选项
        res = fmin_slsqp(self.fun, [-1.0, 1.0],
                         fprime=self.jac, args=(-1.0,),
                         eqcons=[self.f_eqcon], iprint=0,
                         full_output=1)
        # 解析结果变量
        x, fx, its, imode, smode = res
        # 断言模式为 0，即最优解找到
        assert_(imode == 0, imode)
        # 断言结果 x 与期望值接近
        assert_array_almost_equal(x, [1, 1])
    def test_equality_given2(self):
        # 使用 SLSQP 算法求解带有等式约束的优化问题，给定了目标函数和约束函数的雅可比矩阵。
        res = fmin_slsqp(self.fun, [-1.0, 1.0],
                         fprime=self.jac, args=(-1.0,),
                         f_eqcons = self.f_eqcon,
                         fprime_eqcons = self.fprime_eqcon,
                         iprint = 0,
                         full_output = 1)
        x, fx, its, imode, smode = res
        # 断言优化模式 imode 应为 0
        assert_(imode == 0, imode)
        # 断言最优解 x 应接近 [1, 1]
        assert_array_almost_equal(x, [1, 1])

    def test_inequality_given(self):
        # 使用 SLSQP 算法求解带有不等式约束的优化问题，给定了目标函数和约束函数的雅可比矩阵。
        res = fmin_slsqp(self.fun, [-1.0, 1.0],
                         fprime=self.jac, args=(-1.0, ),
                         ieqcons = [self.f_ieqcon],
                         iprint = 0, full_output = 1)
        x, fx, its, imode, smode = res
        # 断言优化模式 imode 应为 0
        assert_(imode == 0, imode)
        # 断言最优解 x 应接近 [2, 1]，精度为 3 位小数
        assert_array_almost_equal(x, [2, 1], decimal=3)

    def test_bound_equality_given2(self):
        # 使用 SLSQP 算法求解带有边界和等式约束的优化问题，给定了目标函数和约束函数的雅可比矩阵。
        res = fmin_slsqp(self.fun, [-1.0, 1.0],
                         fprime=self.jac, args=(-1.0, ),
                         bounds = [(-0.8, 1.), (-1, 0.8)],
                         f_eqcons = self.f_eqcon,
                         fprime_eqcons = self.fprime_eqcon,
                         iprint = 0, full_output = 1)
        x, fx, its, imode, smode = res
        # 断言优化模式 imode 应为 0
        assert_(imode == 0, imode)
        # 断言最优解 x 应接近 [0.8, 0.8]，精度为 3 位小数
        assert_array_almost_equal(x, [0.8, 0.8], decimal=3)
        # 断言 x[0] 在 -0.8 到 1 之间
        assert_(-0.8 <= x[0] <= 1)
        # 断言 x[1] 在 -1 到 0.8 之间
        assert_(-1 <= x[1] <= 0.8)

    def test_scalar_constraints(self):
        # 对于 gh-2182 的回归测试
        # 使用 SLSQP 算法解决只有标量约束的优化问题
        x = fmin_slsqp(lambda z: z**2, [3.],
                       ieqcons=[lambda z: z[0] - 1],
                       iprint=0)
        # 断言最优解 x 应为 [1.]
        assert_array_almost_equal(x, [1.])

        x = fmin_slsqp(lambda z: z**2, [3.],
                       f_ieqcons=lambda z: [z[0] - 1],
                       iprint=0)
        # 断言最优解 x 应为 [1.]
        assert_array_almost_equal(x, [1.])

    def test_integer_bounds(self):
        # 这个测试不应该引发异常
        # 使用 SLSQP 算法解决具有整数边界的优化问题
        fmin_slsqp(lambda z: z**2 - 1, [0], bounds=[[0, 1]], iprint=0)

    def test_array_bounds(self):
        # NumPy 在某些情况下会将 n 维的单元素数组视为标量。
        # `fmin_slsqp` 对 `bounds` 的处理仍然支持此行为。
        # 使用 SLSQP 算法解决具有数组边界的优化问题
        bounds = [(-np.inf, np.inf), (np.array([2]), np.array([3]))]
        x = fmin_slsqp(lambda z: np.sum(z**2 - 1), [2.5, 2.5], bounds=bounds,
                       iprint=0)
        # 断言最优解 x 应为 [0, 2]
        assert_array_almost_equal(x, [0, 2])

    def test_obj_must_return_scalar(self):
        # 对 Github Issue #5433 的回归测试
        # 如果目标函数没有返回标量，则抛出 ValueError
        with assert_raises(ValueError):
            fmin_slsqp(lambda x: [0, 1], [1, 2, 3])
    def test_obj_returns_scalar_in_list(self):
        # 测试 Github 问题 #5433 和 PR #6691
        # 目标函数应能返回包含标量的长度为1的 Python 列表
        fmin_slsqp(lambda x: [0], [1, 2, 3], iprint=0)

    def test_callback(self):
        # 最小化，method='SLSQP'：无界约束，近似雅可比。检查回调函数
        callback = MyCallBack()
        # 调用 minimize 函数，传入初始点 [-1.0, 1.0]，参数 (-1.0,)，使用 SLSQP 方法进行优化，使用自定义回调函数
        res = minimize(self.fun, [-1.0, 1.0], args=(-1.0, ),
                       method='SLSQP', callback=callback, options=self.opts)
        assert_(res['success'], res['message'])
        assert_(callback.been_called)
        assert_equal(callback.ncalls, res['nit'])

    def test_inconsistent_linearization(self):
        # SLSQP 必须能够解决此问题，即使在起始点处的线性化问题不可行。

        # 线性化约束为
        #
        #    2*x0[0]*x[0] >= 1
        #
        # 在 x0 = [0, 1] 处，第二个约束明显不可行。
        # 这将在 LSQ 子程序中调用 n2==1。
        x = [0, 1]
        def f1(x):
            return x[0] + x[1] - 2
        def f2(x):
            return x[0] ** 2 - 1
        # 使用 SLSQP 方法进行最小化
        sol = minimize(
            lambda x: x[0]**2 + x[1]**2,
            x,
            constraints=({'type':'eq','fun': f1},
                         {'type':'ineq','fun': f2}),
            bounds=((0,None), (0,None)),
            method='SLSQP')
        x = sol.x

        assert_allclose(f1(x), 0, atol=1e-8)
        assert_(f2(x) >= -1e-8)
        assert_(sol.success, sol)

    def test_regression_5743(self):
        # SLSQP 不应指示此问题成功，因为它是不可行的。
        x = [1, 2]
        # 使用 SLSQP 方法进行最小化
        sol = minimize(
            lambda x: x[0]**2 + x[1]**2,
            x,
            constraints=({'type':'eq','fun': lambda x: x[0]+x[1]-1},
                         {'type':'ineq','fun': lambda x: x[0]-2}),
            bounds=((0,None), (0,None)),
            method='SLSQP')
        assert_(not sol.success, sol)

    def test_gh_6676(self):
        def func(x):
            return (x[0] - 1)**2 + 2*(x[1] - 1)**2 + 0.5*(x[2] - 1)**2

        # 使用 SLSQP 方法进行最小化
        sol = minimize(func, [0, 0, 0], method='SLSQP')
        assert_(sol.jac.shape == (3,))

    def test_invalid_bounds(self):
        # 当下界大于上界时，抛出正确的错误。
        # 参见 Github 问题 6875。
        bounds_list = [
            ((1, 2), (2, 1)),
            ((2, 1), (1, 2)),
            ((2, 1), (2, 1)),
            ((np.inf, 0), (np.inf, 0)),
            ((1, -np.inf), (0, 1)),
        ]
        for bounds in bounds_list:
            with assert_raises(ValueError):
                minimize(self.fun, [-1.0, 1.0], bounds=bounds, method='SLSQP')
    def test_bounds_clipping(self):
        #
        # SLSQP returns bogus results for initial guess out of bounds, gh-6859
        #

        # 定义测试函数 f(x)，计算 (x[0] - 1)^2 的值
        def f(x):
            return (x[0] - 1)**2

        # 测试情况1: 初始猜测值超出界限 [(None, 0)]，期望结果为 0
        sol = minimize(f, [10], method='slsqp', bounds=[(None, 0)])
        assert_(sol.success)
        assert_allclose(sol.x, 0, atol=1e-10)

        # 测试情况2: 初始猜测值超出界限 [(2, None)]，期望结果为 2
        sol = minimize(f, [-10], method='slsqp', bounds=[(2, None)])
        assert_(sol.success)
        assert_allclose(sol.x, 2, atol=1e-10)

        # 测试情况3: 初始猜测值超出界限 [(None, 0)]，期望结果为 0
        sol = minimize(f, [-10], method='slsqp', bounds=[(None, 0)])
        assert_(sol.success)
        assert_allclose(sol.x, 0, atol=1e-10)

        # 测试情况4: 初始猜测值超出界限 [(2, None)]，期望结果为 2
        sol = minimize(f, [10], method='slsqp', bounds=[(2, None)])
        assert_(sol.success)
        assert_allclose(sol.x, 2, atol=1e-10)

        # 测试情况5: 初始猜测值在界限 [-1, 0] 内，期望结果为 0
        sol = minimize(f, [-0.5], method='slsqp', bounds=[(-1, 0)])
        assert_(sol.success)
        assert_allclose(sol.x, 0, atol=1e-10)

        # 测试情况6: 初始猜测值在界限 [-1, 0] 内，期望结果为 0
        sol = minimize(f, [10], method='slsqp', bounds=[(-1, 0)])
        assert_(sol.success)
        assert_allclose(sol.x, 0, atol=1e-10)

    def test_infeasible_initial(self):
        # Check SLSQP behavior with infeasible initial point

        # 定义测试函数 f(x)，计算 x^2 - 2*x + 1 的值
        def f(x):
            x, = x
            return x*x - 2*x + 1

        # 定义不同的约束条件
        cons_u = [{'type': 'ineq', 'fun': lambda x: 0 - x}]
        cons_l = [{'type': 'ineq', 'fun': lambda x: x - 2}]
        cons_ul = [{'type': 'ineq', 'fun': lambda x: 0 - x},
                   {'type': 'ineq', 'fun': lambda x: x + 1}]

        # 测试情况1: 使用约束条件 cons_u，期望结果为 0
        sol = minimize(f, [10], method='slsqp', constraints=cons_u)
        assert_(sol.success)
        assert_allclose(sol.x, 0, atol=1e-10)

        # 测试情况2: 使用约束条件 cons_l，期望结果为 2
        sol = minimize(f, [-10], method='slsqp', constraints=cons_l)
        assert_(sol.success)
        assert_allclose(sol.x, 2, atol=1e-10)

        # 测试情况3: 使用约束条件 cons_u，期望结果为 0
        sol = minimize(f, [-10], method='slsqp', constraints=cons_u)
        assert_(sol.success)
        assert_allclose(sol.x, 0, atol=1e-10)

        # 测试情况4: 使用约束条件 cons_l，期望结果为 2
        sol = minimize(f, [10], method='slsqp', constraints=cons_l)
        assert_(sol.success)
        assert_allclose(sol.x, 2, atol=1e-10)

        # 测试情况5: 使用约束条件 cons_ul，期望结果为 0
        sol = minimize(f, [-0.5], method='slsqp', constraints=cons_ul)
        assert_(sol.success)
        assert_allclose(sol.x, 0, atol=1e-10)

        # 测试情况6: 使用约束条件 cons_ul，期望结果为 0
        sol = minimize(f, [10], method='slsqp', constraints=cons_ul)
        assert_(sol.success)
        assert_allclose(sol.x, 0, atol=1e-10)

    def test_inconsistent_inequalities(self):
        # gh-7618

        # 定义成本函数 cost(x)，计算 -x[0] + 4*x[1] 的值
        def cost(x):
            return -1 * x[0] + 4 * x[1]

        # 定义不一致的约束条件函数 ineqcons1 和 ineqcons2
        def ineqcons1(x):
            return x[1] - x[0] - 1

        def ineqcons2(x):
            return x[0] - x[1]

        # 定义初始点 x0 和变量边界 bounds
        x0 = (1,5)
        bounds = ((-5, 5), (-5, 5))

        # 定义约束 cons，包含不一致的不等式约束条件
        cons = (dict(type='ineq', fun=ineqcons1), dict(type='ineq', fun=ineqcons2))

        # 尝试最小化成本函数 cost，使用 SLSQP 方法，带有变量边界和不一致的约束条件
        res = minimize(cost, x0, method='SLSQP', bounds=bounds, constraints=cons)

        # 断言结果：预期最小化过程失败，因为不一致的约束条件导致无解
        assert_(not res.success)
    def test_new_bounds_type(self):
        # 定义测试函数 f(x)，计算 x[0]^2 + x[1]^2
        def f(x):
            return x[0] ** 2 + x[1] ** 2
        # 定义变量 bounds，表示参数的上下界
        bounds = Bounds([1, 0], [np.inf, np.inf])
        # 使用 SLSQP 方法进行最小化优化，传入初始点 [0, 0] 和约束 bounds
        sol = minimize(f, [0, 0], method='slsqp', bounds=bounds)
        # 断言优化成功
        assert_(sol.success)
        # 断言优化结果 sol.x 接近 [1, 0]
        assert_allclose(sol.x, [1, 0])

    def test_nested_minimization(self):

        class NestedProblem:
            # 初始化函数
            def __init__(self):
                self.F_outer_count = 0

            # 外层目标函数 F_outer
            def F_outer(self, x):
                self.F_outer_count += 1
                # 如果外层调用次数超过 1000 次，抛出异常
                if self.F_outer_count > 1000:
                    raise Exception("Nested minimization failed to terminate.")
                # 内层优化调用 minimize，使用 SLSQP 方法
                inner_res = minimize(self.F_inner, (3, 4), method="SLSQP")
                # 断言内层优化成功
                assert_(inner_res.success)
                # 断言内层优化结果 inner_res.x 接近 [1, 1]
                assert_allclose(inner_res.x, [1, 1])
                # 返回外层目标函数值
                return x[0]**2 + x[1]**2 + x[2]**2

            # 内层目标函数 F_inner
            def F_inner(self, x):
                return (x[0] - 1)**2 + (x[1] - 1)**2

            # 解决方法，调用外层优化函数 minimize
            def solve(self):
                outer_res = minimize(self.F_outer, (5, 5, 5), method="SLSQP")
                # 断言外层优化成功
                assert_(outer_res.success)
                # 断言外层优化结果 outer_res.x 接近 [0, 0, 0]
                assert_allclose(outer_res.x, [0, 0, 0])

        # 创建 NestedProblem 对象
        problem = NestedProblem()
        # 解决问题
        problem.solve()

    def test_gh1758(self):
        # 定义测试函数 fun(x)，计算 sqrt(x[1])
        def fun(x):
            return np.sqrt(x[1])

        # 定义第一个等式约束函数 f_eqcon(x)
        def f_eqcon(x):
            """ Equality constraint """
            return x[1] - (2 * x[0]) ** 3

        # 定义第二个等式约束函数 f_eqcon2(x)
        def f_eqcon2(x):
            """ Equality constraint """
            return x[1] - (-x[0] + 1) ** 3

        # 设置约束条件 c1 和 c2
        c1 = {'type': 'eq', 'fun': f_eqcon}
        c2 = {'type': 'eq', 'fun': f_eqcon2}

        # 使用 SLSQP 方法进行最小化优化，传入初始点 [8, 0.25]，约束条件 c1 和 c2，以及变量边界
        res = minimize(fun, [8, 0.25], method='SLSQP',
                       constraints=[c1, c2], bounds=[(-0.5, 1), (0, 8)])

        # 断言优化目标函数值 res.fun 接近 0.5443310539518
        np.testing.assert_allclose(res.fun, 0.5443310539518)
        # 断言优化结果 res.x 接近 [0.33333333, 0.2962963]
        np.testing.assert_allclose(res.x, [0.33333333, 0.2962963])
        # 断言优化成功
        assert res.success

    def test_gh9640(self):
        # 设置随机种子
        np.random.seed(10)
        # 定义不等式约束 cons
        cons = ({'type': 'ineq', 'fun': lambda x: -x[0] - x[1] - 3},
                {'type': 'ineq', 'fun': lambda x: x[1] + x[2] - 2})
        # 定义变量边界 bnds
        bnds = ((-2, 2), (-2, 2), (-2, 2))

        # 定义目标函数 target(x)，始终返回 1
        def target(x):
            return 1
        # 设置初始点 x0
        x0 = [-1.8869783504471584, -0.640096352696244, -0.8174212253407696]
        # 使用 SLSQP 方法进行最小化优化，传入初始点 x0，约束条件 cons 和 bnds
        res = minimize(target, x0, method='SLSQP', bounds=bnds, constraints=cons,
                       options={'disp':False, 'maxiter':10000})

        # 断言优化失败，因为问题不可行
        assert not res.success
    # 定义测试函数，验证参数是否始终在指定的边界内
    def test_parameters_stay_within_bounds(self):
        # 设置随机种子，以便结果可重复
        np.random.seed(1)
        # 定义参数的上下界
        bounds = Bounds(np.array([0.1]), np.array([1.0]))
        # 计算输入参数的数量
        n_inputs = len(bounds.lb)
        # 生成初始参数 x0，确保在指定的边界内
        x0 = np.array(bounds.lb + (bounds.ub - bounds.lb) *
                      np.random.random(n_inputs))

        # 定义目标函数 f(x)，其中 x 必须在下界和上界之间
        def f(x):
            assert (x >= bounds.lb).all()  # 断言确保所有 x 的分量都不小于下界
            return np.linalg.norm(x)  # 返回 x 的 L2 范数作为目标函数值

        # 在使用 SLSQP 方法最小化 f(x) 时，监测警告并断言结果成功
        with pytest.warns(RuntimeWarning, match='x were outside bounds'):
            res = minimize(f, x0, method='SLSQP', bounds=bounds)
            assert res.success  # 断言优化成功
```