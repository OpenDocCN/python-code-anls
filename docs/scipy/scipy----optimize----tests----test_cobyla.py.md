# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_cobyla.py`

```
import math  # 导入 math 模块，用于数学运算

import numpy as np  # 导入 numpy 库，用于数值计算
from numpy.testing import assert_allclose, assert_, assert_array_equal  # 从 numpy.testing 中导入断言方法
import pytest  # 导入 pytest 模块，用于单元测试

from scipy.optimize import fmin_cobyla, minimize, Bounds  # 导入 scipy 中的优化函数和约束类


class TestCobyla:
    def setup_method(self):
        self.x0 = [4.95, 0.66]  # 初始化优化变量的起始点
        self.solution = [math.sqrt(25 - (2.0/3)**2), 2.0/3]  # 预期的优化结果
        self.opts = {'disp': False, 'rhobeg': 1, 'tol': 1e-5,
                     'maxiter': 100}  # 优化函数的选项参数

    def fun(self, x):
        return x[0]**2 + abs(x[1])**3  # 待优化的目标函数

    def con1(self, x):
        return x[0]**2 + x[1]**2 - 25  # 第一个约束条件

    def con2(self, x):
        return -self.con1(x)  # 第二个约束条件，与第一个相反

    @pytest.mark.xslow(True, reason='not slow, but noisy so only run rarely')
    def test_simple(self, capfd):
        # 使用 fmin_cobyla 进行简单的优化测试，显示优化过程
        x = fmin_cobyla(self.fun, self.x0, [self.con1, self.con2], rhobeg=1,
                        rhoend=1e-5, maxfun=100, disp=True)
        assert_allclose(x, self.solution, atol=1e-4)  # 断言优化结果与预期解的接近程度

    def test_minimize_simple(self):
        class Callback:
            def __init__(self):
                self.n_calls = 0
                self.last_x = None

            def __call__(self, x):
                self.n_calls += 1
                self.last_x = x

        callback = Callback()

        # 使用 method='COBYLA' 进行最小化优化
        cons = ({'type': 'ineq', 'fun': self.con1},
                {'type': 'ineq', 'fun': self.con2})
        sol = minimize(self.fun, self.x0, method='cobyla', constraints=cons,
                       callback=callback, options=self.opts)
        assert_allclose(sol.x, self.solution, atol=1e-4)  # 断言最小化优化的结果与预期解的接近程度
        assert_(sol.success, sol.message)  # 断言优化成功
        assert_(sol.maxcv < 1e-5, sol)  # 断言最大约束违反值小于给定阈值
        assert_(sol.nfev < 70, sol)  # 断言函数评估次数少于给定阈值
        assert_(sol.fun < self.fun(self.solution) + 1e-3, sol)  # 断言优化目标函数值小于预期解的函数值
        assert_(sol.nfev == callback.n_calls,
                "Callback is not called exactly once for every function eval.")  # 断言回调函数被恰好调用了每次函数评估次数相同的次数
        assert_array_equal(
            sol.x,
            callback.last_x,
            "Last design vector sent to the callback is not equal to returned value.",
        )  # 断言回调函数中最后一个设计向量与返回的优化结果相等

    def test_minimize_constraint_violation(self):
        np.random.seed(1234)
        pb = np.random.rand(10, 10)
        spread = np.random.rand(10)

        def p(w):
            return pb.dot(w)

        def f(w):
            return -(w * spread).sum()

        def c1(w):
            return 500 - abs(p(w)).sum()

        def c2(w):
            return 5 - abs(p(w).sum())

        def c3(w):
            return 5 - abs(p(w)).max()

        cons = ({'type': 'ineq', 'fun': c1},
                {'type': 'ineq', 'fun': c2},
                {'type': 'ineq', 'fun': c3})
        w0 = np.zeros((10,))
        sol = minimize(f, w0, method='cobyla', constraints=cons,
                       options={'catol': 1e-6})
        assert_(sol.maxcv > 1e-6)  # 断言最大约束违反值大于给定阈值
        assert_(not sol.success)  # 断言优化失败


def test_vector_constraints():
    # 测试 fmin_cobyla 和 minimize 能够接受同时返回数字和数组的约束组合
    # 定义一个函数 fun，计算给定点 (x[0], x[1]) 到固定点 (1, 2.5) 的距离平方
    def fun(x):
        return (x[0] - 1)**2 + (x[1] - 2.5)**2
    
    # 定义一个函数 fmin，返回 fun(x) - 1 的结果，即 fun 函数值减去 1
    def fmin(x):
        return fun(x) - 1
    
    # 定义一个函数 cons1，计算由线性约束组成的数组 a 与给定点 x 之间的关系
    def cons1(x):
        a = np.array([[1, -2, 2], [-1, -2, 6], [-1, 2, 2]])
        return np.array([a[i, 0] * x[0] + a[i, 1] * x[1] +
                         a[i, 2] for i in range(len(a))])
    
    # 定义一个函数 cons2，简单地返回输入的 x，用作边界条件 x > 0 的身份函数
    def cons2(x):
        return x
    
    # 初始化起始点 x0
    x0 = np.array([2, 0])
    
    # 将 fun, cons1, cons2 函数组成的列表保存到 cons_list 中
    cons_list = [fun, cons1, cons2]
    
    # 给定一个期望的最优解 xsol
    xsol = [1.4, 1.7]
    # 给定一个期望的最优目标函数值 fsol
    fsol = 0.8
    
    # 测试 fmin_cobyla 函数，期望得到与 xsol 接近的解
    sol = fmin_cobyla(fun, x0, cons_list, rhoend=1e-5)
    assert_allclose(sol, xsol, atol=1e-4)
    
    # 再次使用 fmin_cobyla 函数，期望得到目标函数值接近 1 的解
    sol = fmin_cobyla(fun, x0, fmin, rhoend=1e-5)
    assert_allclose(fun(sol), 1, atol=1e-4)
    
    # 测试 minimize 函数，使用 cons_list 中的约束类型进行最小化
    constraints = [{'type': 'ineq', 'fun': cons} for cons in cons_list]
    sol = minimize(fun, x0, constraints=constraints, tol=1e-5)
    assert_allclose(sol.x, xsol, atol=1e-4)
    assert_(sol.success, sol.message)
    assert_allclose(sol.fun, fsol, atol=1e-4)
    
    # 再次使用 minimize 函数，使用 fmin 作为约束函数，期望得到目标函数值接近 1 的解
    constraints = {'type': 'ineq', 'fun': fmin}
    sol = minimize(fun, x0, constraints=constraints, tol=1e-5)
    assert_allclose(sol.fun, 1, atol=1e-4)
# 定义一个测试类 TestBounds，用于测试 cobyla 方法对边界的支持（仅当通过 `minimize` 使用时）
# 无效边界的测试在 test_optimize.TestOptimizeSimple.test_minimize_invalid_bounds 中进行

class TestBounds:

    def test_basic(self):
        # 定义一个简单的二次函数 f(x)，计算输入向量 x 的平方和作为函数值
        def f(x):
            return np.sum(x**2)

        # 设置下界 lb 和上界 ub
        lb = [-1, None, 1, None, -0.5]
        ub = [-0.5, -0.5, None, None, -0.5]
        # 将 lb 和 ub 组合成 bounds 列表，其中每个元素是一个元组 (a, b)，对应每个维度的下界和上界
        bounds = [(a, b) for a, b in zip(lb, ub)]
        # 这些边界会在内部被转换为 Bounds 对象

        # 调用 minimize 函数，使用 'cobyla' 方法进行优化，传入初始点 x0 和边界 bounds
        res = minimize(f, x0=[1, 2, 3, 4, 5], method='cobyla', bounds=bounds)
        # 定义参考结果 ref
        ref = [-0.5, -0.5, 1, 0, -0.5]
        # 断言优化成功
        assert res.success
        # 使用 assert_allclose 函数断言优化得到的最优点 res.x 与参考结果 ref 的接近程度在给定的容差范围内
        assert_allclose(res.x, ref, atol=1e-3)

    def test_unbounded(self):
        # 定义一个简单的二次函数 f(x)，计算输入向量 x 的平方和作为函数值
        def f(x):
            return np.sum(x**2)

        # 创建一个无界的 Bounds 对象，指定每个维度的下界和上界为 [-∞, -∞] 和 [∞, ∞]
        bounds = Bounds([-np.inf, -np.inf], [np.inf, np.inf])
        # 调用 minimize 函数，使用 'cobyla' 方法进行优化，传入初始点 x0 和边界 bounds
        res = minimize(f, x0=[1, 2], method='cobyla', bounds=bounds)
        # 断言优化成功
        assert res.success
        # 使用 assert_allclose 函数断言优化得到的最优点 res.x 接近 0，容差为 1e-3
        assert_allclose(res.x, 0, atol=1e-3)

        # 创建一个有界的 Bounds 对象，指定第一个维度的下界为 1，第二个维度的下界为 -∞，其它维度为默认的 [∞, ∞]
        bounds = Bounds([1, -np.inf], [np.inf, np.inf])
        # 调用 minimize 函数，使用 'cobyla' 方法进行优化，传入初始点 x0 和边界 bounds
        res = minimize(f, x0=[1, 2], method='cobyla', bounds=bounds)
        # 断言优化成功
        assert res.success
        # 使用 assert_allclose 函数断言优化得到的最优点 res.x 接近 [1, 0]，容差为 1e-3
        assert_allclose(res.x, [1, 0], atol=1e-3)
```