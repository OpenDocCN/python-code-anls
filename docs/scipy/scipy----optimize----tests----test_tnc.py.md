# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_tnc.py`

```
"""
Unit tests for TNC optimization routine from tnc.py
"""
# 导入 pytest 库，用于单元测试
import pytest
# 导入 numpy.testing 库的 assert_allclose 和 assert_equal 函数，用于断言测试结果
from numpy.testing import assert_allclose, assert_equal

# 导入 numpy 库，并从 math 模块中导入 pow 函数
import numpy as np
from math import pow

# 导入 scipy 库中的 optimize 模块
from scipy import optimize


# 定义测试类 TestTnc
class TestTnc:
    """TNC non-linear optimization.

    These tests are taken from Prof. K. Schittkowski's test examples
    for constrained non-linear programming.

    http://www.uni-bayreuth.de/departments/math/~kschittkowski/home.htm

    """
    # 设置每个测试方法的初始化方法
    def setup_method(self):
        # 设置 minimize 方法的选项
        self.opts = {'disp': False, 'maxfun': 200}

    # 定义目标函数 f1 和其对应的雅可比函数 g1
    def f1(self, x, a=100.0):
        return a * pow((x[1] - pow(x[0], 2)), 2) + pow(1.0 - x[0], 2)

    def g1(self, x, a=100.0):
        dif = [0, 0]
        dif[1] = 2 * a * (x[1] - pow(x[0], 2))
        dif[0] = -2.0 * (x[0] * (dif[1] - 1.0) + 1.0)
        return dif

    def fg1(self, x, a=100.0):
        return self.f1(x, a), self.g1(x, a)

    # 定义目标函数 f3 和其对应的雅可比函数 g3
    def f3(self, x):
        return x[1] + pow(x[1] - x[0], 2) * 1.0e-5

    def g3(self, x):
        dif = [0, 0]
        dif[0] = -2.0 * (x[1] - x[0]) * 1.0e-5
        dif[1] = 1.0 - dif[0]
        return dif

    def fg3(self, x):
        return self.f3(x), self.g3(x)

    # 定义目标函数 f4 和其对应的雅可比函数 g4
    def f4(self, x):
        return pow(x[0] + 1.0, 3) / 3.0 + x[1]

    def g4(self, x):
        dif = [0, 0]
        dif[0] = pow(x[0] + 1.0, 2)
        dif[1] = 1.0
        return dif

    def fg4(self, x):
        return self.f4(x), self.g4(x)

    # 定义目标函数 f5 和其对应的雅可比函数 g5
    def f5(self, x):
        return np.sin(x[0] + x[1]) + pow(x[0] - x[1], 2) - \
                1.5 * x[0] + 2.5 * x[1] + 1.0

    def g5(self, x):
        dif = [0, 0]
        v1 = np.cos(x[0] + x[1])
        v2 = 2.0*(x[0] - x[1])

        dif[0] = v1 + v2 - 1.5
        dif[1] = v1 - v2 + 2.5
        return dif

    def fg5(self, x):
        return self.f5(x), self.g5(x)

    # 定义目标函数 f38 和其对应的雅可比函数 g38
    def f38(self, x):
        return (100.0 * pow(x[1] - pow(x[0], 2), 2) +
                pow(1.0 - x[0], 2) + 90.0 * pow(x[3] - pow(x[2], 2), 2) +
                pow(1.0 - x[2], 2) + 10.1 * (pow(x[1] - 1.0, 2) +
                                             pow(x[3] - 1.0, 2)) +
                19.8 * (x[1] - 1.0) * (x[3] - 1.0)) * 1.0e-5

    def g38(self, x):
        dif = [0, 0, 0, 0]
        dif[0] = (-400.0 * x[0] * (x[1] - pow(x[0], 2)) -
                  2.0 * (1.0 - x[0])) * 1.0e-5
        dif[1] = (200.0 * (x[1] - pow(x[0], 2)) + 20.2 * (x[1] - 1.0) +
                  19.8 * (x[3] - 1.0)) * 1.0e-5
        dif[2] = (- 360.0 * x[2] * (x[3] - pow(x[2], 2)) -
                  2.0 * (1.0 - x[2])) * 1.0e-5
        dif[3] = (180.0 * (x[3] - pow(x[2], 2)) + 20.2 * (x[3] - 1.0) +
                  19.8 * (x[1] - 1.0)) * 1.0e-5
        return dif

    def fg38(self, x):
        return self.f38(x), self.g38(x)

    # 定义目标函数 f45，没有定义其雅可比函数，暂时未完成
    def f45(self, x):
        return 2.0 - x[0] * x[1] * x[2] * x[3] * x[4] / 120.0

# 注意：f45 函数没有定义其对应的雅可比函数，可能还未完成实现
    # 定义函数 g45，计算给定向量 x 的梯度向量
    def g45(self, x):
        # 初始化一个长度为 5 的列表，用于存储梯度的各分量
        dif = [0] * 5
        # 计算梯度向量的各分量，根据给定公式进行计算
        dif[0] = - x[1] * x[2] * x[3] * x[4] / 120.0
        dif[1] = - x[0] * x[2] * x[3] * x[4] / 120.0
        dif[2] = - x[0] * x[1] * x[3] * x[4] / 120.0
        dif[3] = - x[0] * x[1] * x[2] * x[4] / 120.0
        dif[4] = - x[0] * x[1] * x[2] * x[3] / 120.0
        # 返回计算得到的梯度向量
        return dif

    # 定义函数 fg45，返回函数 f45 和 g45 在输入 x 下的结果元组
    def fg45(self, x):
        return self.f45(x), self.g45(x)

    # tests
    # 使用 method='TNC' 进行最小化
    def test_minimize_tnc1(self):
        # 初始点和边界条件设置
        x0, bnds = [-2, 1], ([-np.inf, None], [-1.5, None])
        xopt = [1, 1]  # 期望的最优解
        iterx = []  # 用于测试回调函数

        # 调用 optimize.minimize 进行优化，使用 TNC 方法，并指定梯度函数和其他选项
        res = optimize.minimize(self.f1, x0, method='TNC', jac=self.g1,
                                bounds=bnds, options=self.opts,
                                callback=iterx.append)
        # 检查最优解的函数值是否与期望的最优解函数值非常接近
        assert_allclose(res.fun, self.f1(xopt), atol=1e-8)
        # 检查迭代次数是否与结果对象中记录的迭代次数相等
        assert_equal(len(iterx), res.nit)

    # 使用 method='TNC' 进行最小化的另一个测试
    def test_minimize_tnc1b(self):
        x0, bnds = np.array([-2, 1]), ([-np.inf, None], [-1.5, None])
        xopt = [1, 1]
        # 调用 optimize.minimize 进行优化，使用 TNC 方法，仅指定边界条件和选项，并获取最优解
        x = optimize.minimize(self.f1, x0, method='TNC',
                              bounds=bnds, options=self.opts).x
        # 检查最优解的函数值是否与期望的最优解函数值非常接近
        assert_allclose(self.f1(x), self.f1(xopt), atol=1e-4)

    # 使用 method='TNC' 进行最小化的另一个测试，这次使用了包含梯度的函数 fg1
    def test_minimize_tnc1c(self):
        x0, bnds = [-2, 1], ([-np.inf, None],[-1.5, None])
        xopt = [1, 1]
        # 调用 optimize.minimize 进行优化，使用 TNC 方法，指定梯度计算为 True，边界条件和选项，并获取最优解
        x = optimize.minimize(self.fg1, x0, method='TNC',
                              jac=True, bounds=bnds,
                              options=self.opts).x
        # 检查最优解的函数值是否与期望的最优解函数值非常接近
        assert_allclose(self.f1(x), self.f1(xopt), atol=1e-8)

    # 使用 method='TNC' 进行最小化的另一个测试
    def test_minimize_tnc2(self):
        x0, bnds = [-2, 1], ([-np.inf, None], [1.5, None])
        xopt = [-1.2210262419616387, 1.5]
        # 调用 optimize.minimize 进行优化，使用 TNC 方法，指定梯度函数和其他选项，并获取最优解
        x = optimize.minimize(self.f1, x0, method='TNC',
                              jac=self.g1, bounds=bnds,
                              options=self.opts).x
        # 检查最优解的函数值是否与期望的最优解函数值非常接近
        assert_allclose(self.f1(x), self.f1(xopt), atol=1e-8)

    # 使用 method='TNC' 进行最小化的另一个测试
    def test_minimize_tnc3(self):
        x0, bnds = [10, 1], ([-np.inf, None], [0.0, None])
        xopt = [0, 0]
        # 调用 optimize.minimize 进行优化，使用 TNC 方法，指定梯度函数和其他选项，并获取最优解
        x = optimize.minimize(self.f3, x0, method='TNC',
                              jac=self.g3, bounds=bnds,
                              options=self.opts).x
        # 检查最优解的函数值是否与期望的最优解函数值非常接近
        assert_allclose(self.f3(x), self.f3(xopt), atol=1e-8)

    # 使用 method='TNC' 进行最小化的另一个测试
    def test_minimize_tnc4(self):
        x0,bnds = [1.125, 0.125], [(1, None), (0, None)]
        xopt = [1, 0]
        # 调用 optimize.minimize 进行优化，使用 TNC 方法，指定梯度函数和其他选项，并获取最优解
        x = optimize.minimize(self.f4, x0, method='TNC',
                              jac=self.g4, bounds=bnds,
                              options=self.opts).x
        # 检查最优解的函数值是否与期望的最优解函数值非常接近
        assert_allclose(self.f4(x), self.f4(xopt), atol=1e-8)

    # 使用 method='TNC' 进行最小化的另一个测试
    def test_minimize_tnc5(self):
        x0, bnds = [0, 0], [(-1.5, 4),(-3, 3)]
        xopt = [-0.54719755119659763, -1.5471975511965976]
        # 调用 optimize.minimize 进行优化，使用 TNC 方法，指定梯度函数和其他选项，并获取最优解
        x = optimize.minimize(self.f5, x0, method='TNC',
                              jac=self.g5, bounds=bnds,
                              options=self.opts).x
        # 检查最优解的函数值是否与期望的最优解函数值非常接近
        assert_allclose(self.f5(x), self.f5(xopt), atol=1e-8)
    # 定义一个测试方法，用于测试 minimize 函数在特定条件下的表现
    def test_minimize_tnc38(self):
        # 初始点和边界条件的设置
        x0, bnds = np.array([-3, -1, -3, -1]), [(-10, 10)]*4
        # 期望的最优解
        xopt = [1]*4
        # 使用 TNC 方法最小化 self.f38 函数
        x = optimize.minimize(self.f38, x0, method='TNC',
                              jac=self.g38, bounds=bnds,
                              options=self.opts).x
        # 断言最小化后的函数值接近期望值
        assert_allclose(self.f38(x), self.f38(xopt), atol=1e-8)

    # 定义一个测试方法，用于测试 minimize 函数在特定条件下的表现
    def test_minimize_tnc45(self):
        # 初始点和边界条件的设置
        x0, bnds = [2] * 5, [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
        # 期望的最优解
        xopt = [1, 2, 3, 4, 5]
        # 使用 TNC 方法最小化 self.f45 函数
        x = optimize.minimize(self.f45, x0, method='TNC',
                              jac=self.g45, bounds=bnds,
                              options=self.opts).x
        # 断言最小化后的函数值接近期望值
        assert_allclose(self.f45(x), self.f45(xopt), atol=1e-8)

    # 定义一个测试方法，测试 fmin_tnc 函数在特定条件下的表现
    def test_tnc1(self):
        # 设置目标函数 fg1，初始点 x，和边界条件 bounds
        fg, x, bounds = self.fg1, [-2, 1], ([-np.inf, None], [-1.5, None])
        # 期望的最优解
        xopt = [1, 1]

        # 使用 fmin_tnc 函数进行最小化
        x, nf, rc = optimize.fmin_tnc(fg, x, bounds=bounds, args=(100.0, ),
                                      messages=optimize._tnc.MSG_NONE,
                                      maxfun=200)

        # 断言最小化后的函数值接近期望值
        assert_allclose(self.f1(x), self.f1(xopt), atol=1e-8,
                        err_msg="TNC failed with status: " +
                                optimize._tnc.RCSTRINGS[rc])

    # 定义一个测试方法，测试 fmin_tnc 函数在特定条件下的表现
    def test_tnc1b(self):
        # 设置初始点 x 和边界条件 bounds
        x, bounds = [-2, 1], ([-np.inf, None], [-1.5, None])
        # 期望的最优解
        xopt = [1, 1]

        # 使用 fmin_tnc 函数进行最小化，启用自动梯度
        x, nf, rc = optimize.fmin_tnc(self.f1, x, approx_grad=True,
                                      bounds=bounds,
                                      messages=optimize._tnc.MSG_NONE,
                                      maxfun=200)

        # 断言最小化后的函数值接近期望值
        assert_allclose(self.f1(x), self.f1(xopt), atol=1e-4,
                        err_msg="TNC failed with status: " +
                                optimize._tnc.RCSTRINGS[rc])

    # 定义一个测试方法，测试 fmin_tnc 函数在特定条件下的表现
    def test_tnc1c(self):
        # 设置初始点 x 和边界条件 bounds
        x, bounds = [-2, 1], ([-np.inf, None], [-1.5, None])
        # 期望的最优解
        xopt = [1, 1]

        # 使用 fmin_tnc 函数进行最小化，指定梯度函数 self.g1
        x, nf, rc = optimize.fmin_tnc(self.f1, x, fprime=self.g1,
                                      bounds=bounds,
                                      messages=optimize._tnc.MSG_NONE,
                                      maxfun=200)

        # 断言最小化后的函数值接近期望值
        assert_allclose(self.f1(x), self.f1(xopt), atol=1e-8,
                        err_msg="TNC failed with status: " +
                                optimize._tnc.RCSTRINGS[rc])

    # 定义一个测试方法，测试 fmin_tnc 函数在特定条件下的表现
    def test_tnc2(self):
        # 设置目标函数 fg1，初始点 x，和边界条件 bounds
        fg, x, bounds = self.fg1, [-2, 1], ([-np.inf, None], [1.5, None])
        # 期望的最优解
        xopt = [-1.2210262419616387, 1.5]

        # 使用 fmin_tnc 函数进行最小化
        x, nf, rc = optimize.fmin_tnc(fg, x, bounds=bounds,
                                      messages=optimize._tnc.MSG_NONE,
                                      maxfun=200)

        # 断言最小化后的函数值接近期望值
        assert_allclose(self.f1(x), self.f1(xopt), atol=1e-8,
                        err_msg="TNC failed with status: " +
                                optimize._tnc.RCSTRINGS[rc])
    # 定义测试函数 test_tnc3，用于测试 TNC 优化器的功能
    def test_tnc3(self):
        # 定义目标函数 fg，初始点 x，和变量范围 bounds
        fg, x, bounds = self.fg3, [10, 1], ([-np.inf, None], [0.0, None])
        # 设定期望的最优点 xopt 初始值
        xopt = [0, 0]

        # 使用 optimize.fmin_tnc 函数进行优化，返回最优解 x，函数调用次数 nf，和返回码 rc
        x, nf, rc = optimize.fmin_tnc(fg, x, bounds=bounds,
                                      messages=optimize._tnc.MSG_NONE,
                                      maxfun=200)

        # 断言优化结果与预期最优点的函数值接近，否则输出错误信息
        assert_allclose(self.f3(x), self.f3(xopt), atol=1e-8,
                        err_msg="TNC failed with status: " +
                                optimize._tnc.RCSTRINGS[rc])

    # 定义测试函数 test_tnc4，用于测试 TNC 优化器的功能
    def test_tnc4(self):
        # 定义目标函数 fg，初始点 x，和变量范围 bounds
        fg, x, bounds = self.fg4, [1.125, 0.125], [(1, None), (0, None)]
        # 设定期望的最优点 xopt 初始值
        xopt = [1, 0]

        # 使用 optimize.fmin_tnc 函数进行优化，返回最优解 x，函数调用次数 nf，和返回码 rc
        x, nf, rc = optimize.fmin_tnc(fg, x, bounds=bounds,
                                      messages=optimize._tnc.MSG_NONE,
                                      maxfun=200)

        # 断言优化结果与预期最优点的函数值接近，否则输出错误信息
        assert_allclose(self.f4(x), self.f4(xopt), atol=1e-8,
                        err_msg="TNC failed with status: " +
                                optimize._tnc.RCSTRINGS[rc])

    # 定义测试函数 test_tnc5，用于测试 TNC 优化器的功能
    def test_tnc5(self):
        # 定义目标函数 fg，初始点 x，和变量范围 bounds
        fg, x, bounds = self.fg5, [0, 0], [(-1.5, 4),(-3, 3)]
        # 设定期望的最优点 xopt 初始值
        xopt = [-0.54719755119659763, -1.5471975511965976]

        # 使用 optimize.fmin_tnc 函数进行优化，返回最优解 x，函数调用次数 nf，和返回码 rc
        x, nf, rc = optimize.fmin_tnc(fg, x, bounds=bounds,
                                      messages=optimize._tnc.MSG_NONE,
                                      maxfun=200)

        # 断言优化结果与预期最优点的函数值接近，否则输出错误信息
        assert_allclose(self.f5(x), self.f5(xopt), atol=1e-8,
                        err_msg="TNC failed with status: " +
                                optimize._tnc.RCSTRINGS[rc])

    # 定义测试函数 test_tnc38，用于测试 TNC 优化器的功能
    def test_tnc38(self):
        # 定义目标函数 fg，初始点 x，和变量范围 bounds
        fg, x, bounds = self.fg38, np.array([-3, -1, -3, -1]), [(-10, 10)]*4
        # 设定期望的最优点 xopt 初始值
        xopt = [1]*4

        # 使用 optimize.fmin_tnc 函数进行优化，返回最优解 x，函数调用次数 nf，和返回码 rc
        x, nf, rc = optimize.fmin_tnc(fg, x, bounds=bounds,
                                      messages=optimize._tnc.MSG_NONE,
                                      maxfun=200)

        # 断言优化结果与预期最优点的函数值接近，否则输出错误信息
        assert_allclose(self.f38(x), self.f38(xopt), atol=1e-8,
                        err_msg="TNC failed with status: " +
                                optimize._tnc.RCSTRINGS[rc])

    # 定义测试函数 test_tnc45，用于测试 TNC 优化器的功能
    def test_tnc45(self):
        # 定义目标函数 fg，初始点 x，和变量范围 bounds
        fg, x, bounds = self.fg45, [2] * 5, [(0, 1), (0, 2), (0, 3),
                                             (0, 4), (0, 5)]
        # 设定期望的最优点 xopt 初始值
        xopt = [1, 2, 3, 4, 5]

        # 使用 optimize.fmin_tnc 函数进行优化，返回最优解 x，函数调用次数 nf，和返回码 rc
        x, nf, rc = optimize.fmin_tnc(fg, x, bounds=bounds,
                                      messages=optimize._tnc.MSG_NONE,
                                      maxfun=200)

        # 断言优化结果与预期最优点的函数值接近，否则输出错误信息
        assert_allclose(self.f45(x), self.f45(xopt), atol=1e-8,
                        err_msg="TNC failed with status: " +
                                optimize._tnc.RCSTRINGS[rc])
    def test_raising_exceptions(self):
        # tnc was ported to cython from hand-crafted cpython code
        # check that Exception handling works.
        # 定义一个函数myfunc，抛出RuntimeError异常
        def myfunc(x):
            raise RuntimeError("myfunc")

        # 定义一个函数myfunc1，调用optimize.rosen函数
        def myfunc1(x):
            return optimize.rosen(x)

        # 定义一个回调函数callback，抛出ValueError异常
        def callback(x):
            raise ValueError("callback")

        # 使用pytest断言，检测是否捕获到RuntimeError异常
        with pytest.raises(RuntimeError):
            optimize.minimize(myfunc, [0, 1], method="TNC")

        # 使用pytest断言，检测是否捕获到ValueError异常
        with pytest.raises(ValueError):
            optimize.minimize(
                myfunc1, [0, 1], method="TNC", callback=callback
            )

    def test_callback_shouldnt_affect_minimization(self):
        # gh14879. The output of a TNC minimization was different depending
        # on whether a callback was used or not. The two should be equivalent.
        # The issue was that TNC was unscaling/scaling x, and this process was
        # altering x in the process. Now the callback uses an unscaled
        # temporary copy of x.
        # 定义一个空的回调函数callback
        def callback(x):
            pass

        # 定义函数fun为optimize.rosen
        fun = optimize.rosen
        # 定义变量bounds为[(0, 10)] * 4
        bounds = [(0, 10)] * 4
        # 定义初始点x0为[1, 2, 3, 4.]
        x0 = [1, 2, 3, 4.]
        # 调用optimize.minimize函数进行优化，不使用回调函数
        res = optimize.minimize(
            fun, x0, bounds=bounds, method="TNC", options={"maxfun": 1000}
        )
        # 调用optimize.minimize函数进行优化，并使用空回调函数callback
        res2 = optimize.minimize(
            fun, x0, bounds=bounds, method="TNC", options={"maxfun": 1000},
            callback=callback
        )
        # 使用assert_allclose断言res2.x与res.x应该接近
        assert_allclose(res2.x, res.x)
        # 使用assert_allclose断言res2.fun与res.fun应该接近
        assert_allclose(res2.fun, res.fun)
        # 使用assert_equal断言res2.nfev与res.nfev相等
        assert_equal(res2.nfev, res.nfev)
```