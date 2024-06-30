# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_constraint_conversion.py`

```
"""
Unit test for constraint conversion
"""

import numpy as np
from numpy.testing import (assert_array_almost_equal,
                           assert_allclose, assert_warns, suppress_warnings)
import pytest
from scipy.optimize import (NonlinearConstraint, LinearConstraint,
                            OptimizeWarning, minimize, BFGS)
from .test_minimize_constrained import (Maratos, HyperbolicIneq, Rosenbrock,
                                        IneqRosenbrock, EqIneqRosenbrock,
                                        BoundedRosenbrock, Elec)

class TestOldToNew:
    x0 = (2, 0)
    bnds = ((0, None), (0, None))
    method = "trust-constr"

    def test_constraint_dictionary_1(self):
        # 定义目标函数
        def fun(x):
            return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2
        # 定义约束条件列表
        cons = ({'type': 'ineq', 'fun': lambda x: x[0] - 2 * x[1] + 2},
                {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
                {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})

        # 屏蔽特定警告
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "delta_grad == 0.0")
            # 最小化目标函数
            res = minimize(fun, self.x0, method=self.method,
                           bounds=self.bnds, constraints=cons)
        # 断言最优解接近期望值
        assert_allclose(res.x, [1.4, 1.7], rtol=1e-4)
        assert_allclose(res.fun, 0.8, rtol=1e-4)

    def test_constraint_dictionary_2(self):
        # 定义目标函数
        def fun(x):
            return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2
        # 定义等式约束条件字典
        cons = {'type': 'eq',
                'fun': lambda x, p1, p2: p1*x[0] - p2*x[1],
                'args': (1, 1.1),
                'jac': lambda x, p1, p2: np.array([[p1, -p2]])}
        # 屏蔽特定警告
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "delta_grad == 0.0")
            # 最小化目标函数
            res = minimize(fun, self.x0, method=self.method,
                           bounds=self.bnds, constraints=cons)
        # 断言最优解接近期望值
        assert_allclose(res.x, [1.7918552, 1.62895927])
        assert_allclose(res.fun, 1.3857466063348418)

    def test_constraint_dictionary_3(self):
        # 定义目标函数
        def fun(x):
            return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2
        # 定义约束条件列表
        cons = [{'type': 'ineq', 'fun': lambda x: x[0] - 2 * x[1] + 2},
                NonlinearConstraint(lambda x: x[0] - x[1], 0, 0)]

        # 屏蔽特定警告
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "delta_grad == 0.0")
            # 最小化目标函数
            res = minimize(fun, self.x0, method=self.method,
                           bounds=self.bnds, constraints=cons)
        # 断言最优解接近期望值
        assert_allclose(res.x, [1.75, 1.75], rtol=1e-4)
        assert_allclose(res.fun, 1.125, rtol=1e-4)

class TestNewToOld:
    @pytest.mark.fail_slow(2)
    # 定义一个测试函数，用于测试多个约束条件对象的优化行为
    def test_multiple_constraint_objects(self):
        # 定义一个简单的目标函数，计算给定参数向量的平方和
        def fun(x):
            return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2 + (x[2] - 0.75) ** 2
        
        # 初始参数向量
        x0 = [2, 0, 1]
        
        # 定义约束条件列表，这里仅包含不等式约束，可使用 cobyla 方法
        coni = []

        # 可选的优化方法列表
        methods = ["slsqp", "cobyla", "cobyqa", "trust-constr"]

        # 添加混合了旧式和新式约束的情况
        coni.append([{'type': 'ineq', 'fun': lambda x: x[0] - 2 * x[1] + 2},
                     NonlinearConstraint(lambda x: x[0] - x[1], -1, 1)])

        coni.append([LinearConstraint([1, -2, 0], -2, np.inf),
                     NonlinearConstraint(lambda x: x[0] - x[1], -1, 1)])

        coni.append([NonlinearConstraint(lambda x: x[0] - 2 * x[1] + 2, 0, np.inf),
                     NonlinearConstraint(lambda x: x[0] - x[1], -1, 1)])

        # 遍历约束条件列表
        for con in coni:
            # 用于存储各个方法下优化结果的字典
            funs = {}
            # 遍历优化方法列表
            for method in methods:
                # 忽略警告，特别是用户警告
                with suppress_warnings() as sup:
                    sup.filter(UserWarning)
                    # 调用 minimize 函数进行优化，传入目标函数、初始参数、约束条件和优化方法
                    result = minimize(fun, x0, method=method, constraints=con)
                    # 记录该方法下的优化结果值
                    funs[method] = result.fun
            
            # 断言各优化方法下的优化结果与 trust-constr 方法的结果非常接近
            assert_allclose(funs['slsqp'], funs['trust-constr'], rtol=1e-4)
            assert_allclose(funs['cobyla'], funs['trust-constr'], rtol=1e-4)
            assert_allclose(funs['cobyqa'], funs['trust-constr'], rtol=1e-4)

    # 为该测试函数标记一个 fail_slow 的 pytest 标签，设定阈值为 20
    @pytest.mark.fail_slow(20)
# 定义一个测试类 TestNewToOldSLSQP
class TestNewToOldSLSQP:
    # 设定优化方法为 'slsqp'
    method = 'slsqp'
    # 创建一个 Elec 对象，设置电子数为 2
    elec = Elec(n_electrons=2)
    # 设置 Elec 对象的优化结果向量
    elec.x_opt = np.array([-0.58438468, 0.58438466, 0.73597047,
                           -0.73597044, 0.34180668, -0.34180667])
    # 创建一个 BoundedRosenbrock 对象
    brock = BoundedRosenbrock()
    # 设置 BoundedRosenbrock 对象的优化结果向量
    brock.x_opt = [0, 0]
    # 创建一个问题列表，包含 Maratos, HyperbolicIneq, Rosenbrock, IneqRosenbrock,
    # EqIneqRosenbrock, elec, brock 这些问题对象
    list_of_problems = [Maratos(),
                        HyperbolicIneq(),
                        Rosenbrock(),
                        IneqRosenbrock(),
                        EqIneqRosenbrock(),
                        elec,
                        brock
                        ]

    # 定义测试函数 test_list_of_problems
    def test_list_of_problems(self):

        # 遍历问题列表中的每个问题对象
        for prob in self.list_of_problems:
            # 忽略特定的警告类型
            with suppress_warnings() as sup:
                sup.filter(UserWarning)
                # 调用 minimize 函数，对当前问题 prob 进行优化
                result = minimize(prob.fun, prob.x0,
                                  method=self.method,
                                  bounds=prob.bounds,
                                  constraints=prob.constr)

            # 断言优化结果的解 result.x 与问题的预期最优解 prob.x_opt 很接近（精度为 3 位小数）
            assert_array_almost_equal(result.x, prob.x_opt, decimal=3)

    # 定义测试函数 test_warn_mixed_constraints
    def test_warn_mixed_constraints(self):
        # 定义一个简单的目标函数 fun
        def fun(x):
            return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2 + (x[2] - 0.75) ** 2
        # 定义非线性约束条件 cons
        cons = NonlinearConstraint(lambda x: [x[0]**2 - x[1], x[1] - x[2]],
                                   [1.1, .8], [1.1, 1.4])
        # 定义变量边界 bnds
        bnds = ((0, None), (0, None), (0, None))
        # 忽略特定的警告类型
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "delta_grad == 0.0")
            # 断言调用 minimize 函数时会引发 OptimizeWarning 警告
            assert_warns(OptimizeWarning, minimize, fun, (2, 0, 1),
                         method=self.method, bounds=bnds, constraints=cons)
    def test_warn_ignored_options(self):
        # 测试函数：检查是否警告忽略了约束选项

        # 定义目标函数
        def fun(x):
            return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2 + (x[2] - 0.75) ** 2
        
        # 设置初始点
        x0 = (2, 0, 1)

        # 根据方法选择约束条件
        if self.method == "slsqp":
            bnds = ((0, None), (0, None), (0, None))
        else:
            bnds = None

        # 定义非线性约束条件：x[0] >= 2
        cons = NonlinearConstraint(lambda x: x[0], 2, np.inf)
        # 进行优化，检查是否有警告信息
        res = minimize(fun, x0, method=self.method,
                       bounds=bnds, constraints=cons)
        # 没有约束选项时不应有警告
        assert_allclose(res.fun, 1)

        # 定义线性约束条件：x[0] >= 2
        cons = LinearConstraint([1, 0, 0], 2, np.inf)
        # 进行优化，检查是否有警告信息
        res = minimize(fun, x0, method=self.method,
                       bounds=bnds, constraints=cons)
        # 没有约束选项时不应有警告
        assert_allclose(res.fun, 1)

        # 多个不同设置的非线性约束条件
        cons = []
        cons.append(NonlinearConstraint(lambda x: x[0]**2, 2, np.inf,
                                        keep_feasible=True))
        cons.append(NonlinearConstraint(lambda x: x[0]**2, 2, np.inf,
                                        hess=BFGS()))
        cons.append(NonlinearConstraint(lambda x: x[0]**2, 2, np.inf,
                                        finite_diff_jac_sparsity=42))
        cons.append(NonlinearConstraint(lambda x: x[0]**2, 2, np.inf,
                                        finite_diff_rel_step=42))
        cons.append(LinearConstraint([1, 0, 0], 2, np.inf,
                                     keep_feasible=True))
        
        # 遍历每个约束条件，检查是否有优化警告
        for con in cons:
            assert_warns(OptimizeWarning, minimize, fun, x0,
                         method=self.method, bounds=bnds, constraints=cons)
# 定义一个名为 TestNewToOldCobyla 的测试类，用于测试 'cobyla' 方法
class TestNewToOldCobyla:
    # 类属性，设定优化方法为 'cobyla'
    method = 'cobyla'

    # 包含两个 Elec 对象的列表，分别初始化时传入不同的电子数目
    list_of_problems = [
                        Elec(n_electrons=2),
                        Elec(n_electrons=4),
                        ]

    # 使用 pytest.mark.slow 标记为慢速测试的方法
    @pytest.mark.slow
    # 定义测试方法 test_list_of_problems
    def test_list_of_problems(self):

        # 遍历 self.list_of_problems 中的每个 prob
        for prob in self.list_of_problems:

            # 使用 suppress_warnings 上下文管理器，过滤掉 UserWarning
            with suppress_warnings() as sup:
                sup.filter(UserWarning)

                # 使用 'trust-constr' 方法进行优化，得到真实值 truth
                truth = minimize(prob.fun, prob.x0,
                                 method='trust-constr',
                                 bounds=prob.bounds,
                                 constraints=prob.constr)

                # 使用 self.method 方法进行优化，得到结果值 result
                result = minimize(prob.fun, prob.x0,
                                  method=self.method,
                                  bounds=prob.bounds,
                                  constraints=prob.constr)

            # 断言 result.fun 与 truth.fun 在相对容差 1e-3 范围内相等
            assert_allclose(result.fun, truth.fun, rtol=1e-3)
```