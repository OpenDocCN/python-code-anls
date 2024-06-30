# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_cobyqa.py`

```
# 导入所需的库和模块
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from scipy.optimize import (
    Bounds,
    LinearConstraint,
    NonlinearConstraint,
    OptimizeResult,
    minimize,
)

# 定义测试类 TestCOBYQA，用于测试 COBYQA 方法
class TestCOBYQA:
    
    # 在每个测试方法运行前初始化测试所需的参数和选项
    def setup_method(self):
        self.x0 = [4.95, 0.66]  # 初始变量值
        self.options = {'maxfev': 100}  # 优化选项，最大函数评估次数为 100

    # 定义静态方法 fun，计算给定变量 x 下的目标函数值
    @staticmethod
    def fun(x, c=1.0):
        return x[0]**2 + c * abs(x[1])**3

    # 定义静态方法 con，表示优化问题中的约束条件
    @staticmethod
    def con(x):
        return x[0]**2 + x[1]**2 - 25.0  # 约束条件 x^2 + y^2 = 25

    # 测试最小化函数 minimize 的简单用法
    def test_minimize_simple(self):
        # 定义用于测试回调函数的类 Callback
        class Callback:
            def __init__(self):
                self.n_calls = 0

            def __call__(self, x):
                assert isinstance(x, np.ndarray)
                self.n_calls += 1

        # 定义用于测试回调函数的类 CallbackNewSyntax，使用新的回调语法
        class CallbackNewSyntax:
            def __init__(self):
                self.n_calls = 0

            def __call__(self, intermediate_result):
                assert isinstance(intermediate_result, OptimizeResult)
                self.n_calls += 1

        # 创建 Callback 的实例 callback 和 CallbackNewSyntax 的实例 callback_new_syntax
        callback = Callback()
        callback_new_syntax = CallbackNewSyntax()

        # 设置非线性约束条件为 self.con 定义的函数对象
        constraints = NonlinearConstraint(self.con, 0.0, 0.0)

        # 使用 COBYQA 方法进行优化，使用 Callback 进行回调
        sol = minimize(
            self.fun,
            self.x0,
            method='cobyqa',
            constraints=constraints,
            callback=callback,
            options=self.options,
        )

        # 使用 COBYQA 方法进行优化，使用 CallbackNewSyntax 进行回调
        sol_new = minimize(
            self.fun,
            self.x0,
            method='cobyqa',
            constraints=constraints,
            callback=callback_new_syntax,
            options=self.options,
        )

        # 预期的最优解
        solution = [np.sqrt(25.0 - 4.0 / 9.0), 2.0 / 3.0]

        # 对结果进行断言验证
        assert_allclose(sol.x, solution, atol=1e-4)  # 验证最优解的数值接近程度
        assert sol.success, sol.message  # 验证优化是否成功
        assert sol.maxcv < 1e-8, sol  # 验证最大约束违反量是否小于给定阈值
        assert sol.nfev <= 100, sol  # 验证函数评估次数是否符合预期
        assert sol.fun < self.fun(solution) + 1e-3, sol  # 验证目标函数值是否达到预期精度
        assert sol.nfev == callback.n_calls, \
            "Callback is not called exactly once for every function eval."  # 验证回调函数是否按预期调用

        # 验证新语法下的结果是否与旧语法下的结果一致
        assert_equal(sol.x, sol_new.x)  # 验证最优解是否相等
        assert sol_new.success, sol_new.message  # 验证新语法下的优化是否成功
        assert sol.fun == sol_new.fun  # 验证新语法下的目标函数值是否与旧语法相同
        assert sol.maxcv == sol_new.maxcv  # 验证新语法下的最大约束违反量是否与旧语法相同
        assert sol.nfev == sol_new.nfev  # 验证新语法下的函数评估次数是否与旧语法相同
        assert sol.nit == sol_new.nit  # 验证新语法下的迭代次数是否与旧语法相同
        assert sol_new.nfev == callback_new_syntax.n_calls, \
            "Callback is not called exactly once for every function eval."  # 验证新语法下回调函数是否按预期调用
    # 定义一个测试函数，用于测试在最小化过程中对边界进行约束的情况
    def test_minimize_bounds(self):
        
        # 定义一个内部函数，用于检查解是否在给定边界内
        def fun_check_bounds(x):
            # 断言：检查解 x 是否在边界 lb <= x <= ub 内
            assert np.all(bounds.lb <= x) and np.all(x <= bounds.ub)
            # 调用外部函数 self.fun，计算给定 x 的目标函数值
            return self.fun(x)

        # 创建一个边界对象 bounds，定义变量的下界和上界
        bounds = Bounds([4.5, 0.6], [5.0, 0.7])
        # 创建一个非线性约束对象 constraints，用于约束问题
        constraints = NonlinearConstraint(self.con, 0.0, 0.0)
        # 调用 minimize 函数进行最小化优化
        sol = minimize(
            fun_check_bounds,    # 待优化的目标函数
            self.x0,             # 初始猜测值
            method='cobyqa',     # 优化方法
            bounds=bounds,       # 变量边界
            constraints=constraints,  # 约束条件
            options=self.options,     # 优化选项
        )
        # 预期的解
        solution = [np.sqrt(25.0 - 4.0 / 9.0), 2.0 / 3.0]
        # 断言：验证最优解 sol.x 是否与预期解 solution 在给定的容差范围内相等
        assert_allclose(sol.x, solution, atol=1e-4)
        # 断言：验证优化是否成功
        assert sol.success, sol.message
        # 断言：验证最大约束违反值是否小于给定阈值
        assert sol.maxcv < 1e-8, sol
        # 断言：检查最优解 sol.x 是否在边界 bounds 内
        assert np.all(bounds.lb <= sol.x) and np.all(sol.x <= bounds.ub), sol
        # 断言：验证函数评估次数是否少于等于100次
        assert sol.nfev <= 100, sol
        # 断言：验证最优函数值是否小于给定的目标函数值加上容差值
        assert sol.fun < self.fun(solution) + 1e-3, sol

        # 创建一个新的边界对象 bounds，定义变量的下界和上界，以检验边界在解处为活跃的情况
        bounds = Bounds([5.0, 0.6], [5.5, 0.65])
        # 重新调用 minimize 函数进行优化
        sol = minimize(
            fun_check_bounds,    # 待优化的目标函数
            self.x0,             # 初始猜测值
            method='cobyqa',     # 优化方法
            bounds=bounds,       # 变量边界
            constraints=constraints,  # 约束条件
            options=self.options,     # 优化选项
        )
        # 断言：验证优化是否失败
        assert not sol.success, sol.message
        # 断言：验证最大约束违反值是否大于给定阈值
        assert sol.maxcv > 0.35, sol
        # 断言：检查最优解 sol.x 是否在边界 bounds 内
        assert np.all(bounds.lb <= sol.x) and np.all(sol.x <= bounds.ub), sol
        # 断言：验证函数评估次数是否少于等于100次
        assert sol.nfev <= 100, sol

    # 定义一个测试函数，用于测试在最小化过程中线性约束的情况
    def test_minimize_linear_constraints(self):
        # 创建一个线性约束对象 constraints，用于约束问题
        constraints = LinearConstraint([1.0, 1.0], 1.0, 1.0)
        # 调用 minimize 函数进行最小化优化
        sol = minimize(
            self.fun,            # 待优化的目标函数
            self.x0,             # 初始猜测值
            method='cobyqa',     # 优化方法
            constraints=constraints,  # 约束条件
            options=self.options,     # 优化选项
        )
        # 预期的解
        solution = [(4 - np.sqrt(7)) / 3, (np.sqrt(7) - 1) / 3]
        # 断言：验证最优解 sol.x 是否与预期解 solution 在给定的容差范围内相等
        assert_allclose(sol.x, solution, atol=1e-4)
        # 断言：验证优化是否成功
        assert sol.success, sol.message
        # 断言：验证最大约束违反值是否小于给定阈值
        assert sol.maxcv < 1e-8, sol
        # 断言：验证函数评估次数是否少于等于100次
        assert sol.nfev <= 100, sol
        # 断言：验证最优函数值是否小于给定的目标函数值加上容差值
        assert sol.fun < self.fun(solution) + 1e-3, sol

    # 定义一个测试函数，用于测试在最小化过程中传递额外参数的情况
    def test_minimize_args(self):
        # 创建一个非线性约束对象 constraints，用于约束问题
        constraints = NonlinearConstraint(self.con, 0.0, 0.0)
        # 调用 minimize 函数进行最小化优化
        sol = minimize(
            self.fun,            # 待优化的目标函数
            self.x0,             # 初始猜测值
            args=(2.0,),         # 额外的参数
            method='cobyqa',     # 优化方法
            constraints=constraints,  # 约束条件
            options=self.options,     # 优化选项
        )
        # 预期的解
        solution = [np.sqrt(25.0 - 4.0 / 36.0), 2.0 / 6.0]
        # 断言：验证最优解 sol.x 是否与预期解 solution 在给定的容差范围内相等
        assert_allclose(sol.x, solution, atol=1e-4)
        # 断言：验证优化是否成功
        assert sol.success, sol.message
        # 断言：验证最大约束违反值是否小于给定阈值
        assert sol.maxcv < 1e-8, sol
        # 断言：验证函数评估次数是否少于等于100次
        assert sol.nfev <= 100, sol
        # 断言：验证最优函数值是否小于给定的目标函数值加上容差值
        assert sol.fun < self.fun(solution, 2.0) + 1e-3, sol
    def test_minimize_array(self):
        # 定义将数组转换为指定维度形状的函数
        def fun_array(x, dim):
            # 调用 self.fun 计算得到的结果转换为 NumPy 数组
            f = np.array(self.fun(x))
            # 将结果重新整形为指定维度的形状
            return np.reshape(f, (1,) * dim)

        # 定义变量的范围限制条件
        bounds = Bounds([4.5, 0.6], [5.0, 0.7])
        # 定义非线性约束条件
        constraints = NonlinearConstraint(self.con, 0.0, 0.0)
        # 使用 cobyqa 方法最小化 self.fun 函数
        sol = minimize(
            self.fun,
            self.x0,
            method='cobyqa',
            bounds=bounds,
            constraints=constraints,
            options=self.options,
        )

        # 对于不同的维度进行循环测试
        for dim in [0, 1, 2]:
            # 使用 cobyqa 方法最小化 fun_array 函数
            sol_array = minimize(
                fun_array,
                self.x0,
                args=(dim,),
                method='cobyqa',
                bounds=bounds,
                constraints=constraints,
                options=self.options,
            )
            # 断言最小化结果 sol 的解与 sol_array 的解相等
            assert_equal(sol.x, sol_array.x)
            # 断言 sol_array 的成功标志为 True
            assert sol_array.success, sol_array.message
            # 断言 sol 的最优目标函数值与 sol_array 的相等
            assert sol.fun == sol_array.fun
            # 断言 sol 的最大约束违反值与 sol_array 的相等
            assert sol.maxcv == sol_array.maxcv
            # 断言 sol 的函数评估次数与 sol_array 的相等
            assert sol.nfev == sol_array.nfev
            # 断言 sol 的迭代次数与 sol_array 的相等
            assert sol.nit == sol_array.nit

        # 当 fun 返回多于一个元素的数组时，期望引发 TypeError 异常
        with pytest.raises(TypeError):
            minimize(
                lambda x: np.array([self.fun(x), self.fun(x)]),
                self.x0,
                method='cobyqa',
                bounds=bounds,
                constraints=constraints,
                options=self.options,
            )

    def test_minimize_maxfev(self):
        # 定义非线性约束条件
        constraints = NonlinearConstraint(self.con, 0.0, 0.0)
        # 定义最大函数评估次数选项
        options = {'maxfev': 2}
        # 使用 cobyqa 方法最小化 self.fun 函数
        sol = minimize(
            self.fun,
            self.x0,
            method='cobyqa',
            constraints=constraints,
            options=options,
        )
        # 断言最小化结果的成功标志为 False
        assert not sol.success, sol.message
        # 断言最小化结果的函数评估次数小于或等于 2
        assert sol.nfev <= 2, sol

    def test_minimize_maxiter(self):
        # 定义非线性约束条件
        constraints = NonlinearConstraint(self.con, 0.0, 0.0)
        # 定义最大迭代次数选项
        options = {'maxiter': 2}
        # 使用 cobyqa 方法最小化 self.fun 函数
        sol = minimize(
            self.fun,
            self.x0,
            method='cobyqa',
            constraints=constraints,
            options=options,
        )
        # 断言最小化结果的成功标志为 False
        assert not sol.success, sol.message
        # 断言最小化结果的迭代次数小于或等于 2
        assert sol.nit <= 2, sol

    def test_minimize_f_target(self):
        # 定义非线性约束条件
        constraints = NonlinearConstraint(self.con, 0.0, 0.0)
        # 使用 cobyqa 方法最小化 self.fun 函数，得到参考解
        sol_ref = minimize(
            self.fun,
            self.x0,
            method='cobyqa',
            constraints=constraints,
            options=self.options,
        )
        # 复制选项字典
        options = dict(self.options)
        # 设置 f_target 选项为参考解的最优目标函数值
        options['f_target'] = sol_ref.fun
        # 使用 cobyqa 方法最小化 self.fun 函数
        sol = minimize(
            self.fun,
            self.x0,
            method='cobyqa',
            constraints=constraints,
            options=options,
        )
        # 断言最小化结果的成功标志为 True
        assert sol.success, sol.message
        # 断言最小化结果的最大约束违反值小于 1e-8
        assert sol.maxcv < 1e-8, sol
        # 断言最小化结果的函数评估次数小于或等于参考解的函数评估次数
        assert sol.nfev <= sol_ref.nfev, sol
        # 断言最小化结果的最优目标函数值小于或等于参考解的最优目标函数值
        assert sol.fun <= sol_ref.fun, sol
```