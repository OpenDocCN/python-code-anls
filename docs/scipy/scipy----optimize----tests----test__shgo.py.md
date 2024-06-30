# `D:\src\scipysrc\scipy\scipy\optimize\tests\test__shgo.py`

```
import logging  # 导入 logging 模块，用于记录日志
import sys  # 导入 sys 模块，用于访问系统相关的参数和功能

import numpy as np  # 导入 NumPy 库，用于数值计算
import time  # 导入 time 模块，用于时间相关操作
from multiprocessing import Pool  # 导入 Pool 类，用于实现并行计算
from numpy.testing import assert_allclose, IS_PYPY  # 导入 NumPy 测试相关函数和变量
import pytest  # 导入 pytest 模块，用于编写和运行测试
from pytest import raises as assert_raises, warns  # 导入 pytest 的 raises 和 warns 函数
from scipy.optimize import (shgo, Bounds, minimize_scalar, minimize, rosen,  # 导入 SciPy 优化模块相关函数
                            rosen_der, rosen_hess, NonlinearConstraint)
from scipy.optimize._constraints import new_constraint_to_old  # 导入 SciPy 优化模块中的约束转换函数
from scipy.optimize._shgo import SHGO  # 导入 SciPy 优化模块中的 SHGO 类


class StructTestFunction:
    def __init__(self, bounds, expected_x, expected_fun=None,
                 expected_xl=None, expected_funl=None):
        self.bounds = bounds  # 设置实例变量 bounds
        self.expected_x = expected_x  # 设置实例变量 expected_x
        self.expected_fun = expected_fun  # 设置实例变量 expected_fun
        self.expected_xl = expected_xl  # 设置实例变量 expected_xl
        self.expected_funl = expected_funl  # 设置实例变量 expected_funl


def wrap_constraints(g):
    cons = []  # 初始化约束列表 cons
    if g is not None:  # 如果 g 不为 None
        if not isinstance(g, (tuple, list)):  # 如果 g 不是 tuple 或 list
            g = (g,)  # 将 g 转换为单元素元组
        else:
            pass  # g 已经是 tuple 或 list，无需额外操作
        for g in g:  # 遍历 g 中的每个函数 g
            cons.append({'type': 'ineq',  # 将约束类型设为不等式约束
                         'fun': g})  # 设置约束函数为 g
        cons = tuple(cons)  # 将列表转换为元组
    else:
        cons = None  # 如果 g 为 None，则将 cons 设为 None
    return cons  # 返回约束条件 cons


class StructTest1(StructTestFunction):
    def f(self, x):
        return x[0] ** 2 + x[1] ** 2  # 定义函数 f，返回 x[0]^2 + x[1]^2

    @staticmethod
    def g(x):
        return -(np.sum(x, axis=0) - 6.0)  # 定义静态方法 g，返回 -(x[0] + x[1] - 6.0)

    cons = wrap_constraints(g)  # 调用 wrap_constraints 方法生成约束条件


test1_1 = StructTest1(bounds=[(-1, 6), (-1, 6)],  # 创建 StructTest1 的实例 test1_1
                      expected_x=[0, 0])  # 指定期望的最优解 expected_x


test1_2 = StructTest1(bounds=[(0, 1), (0, 1)],  # 创建 StructTest1 的实例 test1_2
                      expected_x=[0, 0])  # 指定期望的最优解 expected_x


test1_3 = StructTest1(bounds=[(None, None), (None, None)],  # 创建 StructTest1 的实例 test1_3
                      expected_x=[0, 0])  # 指定期望的最优解 expected_x


class StructTest2(StructTestFunction):
    """
    Scalar function with several minima to test all minimiser retrievals
    """

    def f(self, x):
        return (x - 30) * np.sin(x)  # 定义函数 f，返回 (x - 30) * sin(x)

    @staticmethod
    def g(x):
        return 58 - np.sum(x, axis=0)  # 定义静态方法 g，返回 58 - (x[0] + x[1])

    cons = wrap_constraints(g)  # 调用 wrap_constraints 方法生成约束条件
# 创建一个`
# 创建一个名为 test2_1 的 StructTest2 对象，用于测试结构化函数
test2_1 = StructTest2(bounds=[(0, 60)],
                      expected_x=[1.53567906],
                      expected_fun=-28.44677132,
                      # 重要：测试 funl 返回值的顺序是否正确
                      expected_xl=np.array([[1.53567906],
                                            [55.01782167],
                                            [7.80894889],
                                            [48.74797493],
                                            [14.07445705],
                                            [42.4913859],
                                            [20.31743841],
                                            [36.28607535],
                                            [26.43039605],
                                            [30.76371366]]),

                      expected_funl=np.array([-28.44677132, -24.99785984,
                                              -22.16855376, -18.72136195,
                                              -15.89423937, -12.45154942,
                                              -9.63133158, -6.20801301,
                                              -3.43727232, -0.46353338])
                      )

# 创建一个名为 test2_2 的 StructTest2 对象，用于测试结构化函数
test2_2 = StructTest2(bounds=[(0, 4.5)],
                      expected_x=[1.53567906],
                      expected_fun=[-28.44677132],
                      expected_xl=np.array([[1.53567906]]),
                      expected_funl=np.array([-28.44677132])
                      )


class StructTest3(StructTestFunction):
    """
    Hock and Schittkowski 18 problem (HS18). Hoch and Schittkowski (1981)
    http://www.ai7.uni-bayreuth.de/test_problem_coll.pdf
    Minimize: f = 0.01 * (x_1)**2 + (x_2)**2

    Subject to: x_1 * x_2 - 25.0 >= 0,
                (x_1)**2 + (x_2)**2 - 25.0 >= 0,
                2 <= x_1 <= 50,
                0 <= x_2 <= 50.

    Approx. Answer:
        f([(250)**0.5 , (2.5)**0.5]) = 5.0
    """

    # 重写 f 函数以定义问题的目标函数
    def f(self, x):
        return 0.01 * (x[0]) ** 2 + (x[1]) ** 2

    # 定义第一个约束函数 g1
    def g1(x):
        return x[0] * x[1] - 25.0

    # 定义第二个约束函数 g2
    def g2(x):
        return x[0] ** 2 + x[1] ** 2 - 25.0

    # 将 g1 和 g2 组合成一个元组 g
    # cons = wrap_constraints(g)

    # 定义约束函数 g，返回 g1 和 g2 的结果
    def g(x):
        return x[0] * x[1] - 25.0, x[0] ** 2 + x[1] ** 2 - 25.0

    # 创建一个 NonlinearConstraint 对象 __nlc，并将其包装为元组 cons
    # 这里检查 shgo 是否能接收新样式的约束
    __nlc = NonlinearConstraint(g, 0, np.inf)
    cons = (__nlc,)

# 创建一个名为 test3_1 的 StructTest3 对象，用于测试结构化函数
test3_1 = StructTest3(bounds=[(2, 50), (0, 50)],
                      expected_x=[250 ** 0.5, 2.5 ** 0.5],
                      expected_fun=5.0
                      )


class StructTest4(StructTestFunction):
    """
    Hock and Schittkowski 11 problem (HS11). Hoch and Schittkowski (1981)

    NOTE: Did not find in original reference to HS collection, refer to
          Henderson (2015) problem 7 instead. 02.03.2016
    """

# 创建一个名为 StructTest4 的空类，用于文档说明
    # 定义函数 f，计算给定向量 x 的复杂表达式的值
    def f(self, x):
        return ((x[0] - 10) ** 2 + 5 * (x[1] - 12) ** 2 + x[2] ** 4
                + 3 * (x[3] - 11) ** 2 + 10 * x[4] ** 6 + 7 * x[5] ** 2 + x[
                    6] ** 4
                - 4 * x[5] * x[6] - 10 * x[5] - 8 * x[6]
                )

    # 定义函数 g1，计算给定向量 x 满足的约束条件 g1 的值
    def g1(x):
        return -(2 * x[0] ** 2 + 3 * x[1] ** 4 + x[2] + 4 * x[3] ** 2
                 + 5 * x[4] - 127)

    # 定义函数 g2，计算给定向量 x 满足的约束条件 g2 的值
    def g2(x):
        return -(7 * x[0] + 3 * x[1] + 10 * x[2] ** 2 + x[3] - x[4] - 282.0)

    # 定义函数 g3，计算给定向量 x 满足的约束条件 g3 的值
    def g3(x):
        return -(23 * x[0] + x[1] ** 2 + 6 * x[5] ** 2 - 8 * x[6] - 196)

    # 定义函数 g4，计算给定向量 x 满足的约束条件 g4 的值
    def g4(x):
        return -(4 * x[0] ** 2 + x[1] ** 2 - 3 * x[0] * x[1] + 2 * x[2] ** 2
                 + 5 * x[5] - 11 * x[6])

    # 将约束函数 g1, g2, g3, g4 存储在元组 g 中
    g = (g1, g2, g3, g4)

    # 调用 wrap_constraints 函数，将约束函数 g 封装成约束对象并赋值给 cons 变量
    cons = wrap_constraints(g)
test4_1 = StructTest4(bounds=[(-10, 10), ] * 7,
                      expected_x=[2.330499, 1.951372, -0.4775414,
                                  4.365726, -0.6244870, 1.038131, 1.594227],
                      expected_fun=680.6300573
                      )

该代码定义了一个 `StructTest4` 类的实例 `test4_1`，设置了该实例的边界 (`bounds`)、预期输入 (`expected_x`) 和预期函数值 (`expected_fun`)。


class StructTest5(StructTestFunction):
    def f(self, x):
        return (
            -(x[1] + 47.0)*np.sin(np.sqrt(abs(x[0]/2.0 + (x[1] + 47.0))))
            - x[0]*np.sin(np.sqrt(abs(x[0] - (x[1] + 47.0))))
        )

    g = None
    cons = wrap_constraints(g)

定义了 `StructTest5` 类，继承自 `StructTestFunction`。其中 `f` 方法是类的一个函数成员，计算一个特定函数的值。`g` 被设置为 `None`，并作为参数传递给 `wrap_constraints` 函数。


test5_1 = StructTest5(bounds=[(-512, 512), (-512, 512)],
                      expected_fun=[-959.64066272085051],
                      expected_x=[512., 404.23180542])

创建了 `StructTest5` 类的一个实例 `test5_1`，设置了边界 (`bounds`)、预期函数值 (`expected_fun`) 和预期输入 (`expected_x`)。


class StructTestLJ(StructTestFunction):
    """
    LennardJones objective function. Used to test symmetry constraints
    settings.
    """

    def f(self, x, *args):
        print(f'x = {x}')
        self.N = args[0]
        k = int(self.N / 3)
        s = 0.0

        for i in range(k - 1):
            for j in range(i + 1, k):
                a = 3 * i
                b = 3 * j
                xd = x[a] - x[b]
                yd = x[a + 1] - x[b + 1]
                zd = x[a + 2] - x[b + 2]
                ed = xd * xd + yd * yd + zd * zd
                ud = ed * ed * ed
                if ed > 0.0:
                    s += (1.0 / ud - 2.0) / ud

        return s

    g = None
    cons = wrap_constraints(g)


N = 6
boundsLJ = list(zip([-4.0] * 6, [4.0] * 6))

testLJ = StructTestLJ(bounds=boundsLJ,
                      expected_fun=[-1.0],
                      expected_x=None,
                      # expected_x=[-2.71247337e-08,
                      #            -2.71247337e-08,
                      #            -2.50000222e+00,
                      #            -2.71247337e-08,
                      #            -2.71247337e-08,
                      #            -1.50000222e+00]
                      )

定义了 `StructTestLJ` 类，描述了一个 Lennard-Jones 目标函数。它的 `f` 方法计算函数值，而 `N` 是一个参数。`boundsLJ` 是该类实例化时设置的边界。`testLJ` 是 `StructTestLJ` 的一个实例，设置了边界 (`bounds`)、预期函数值 (`expected_fun`) 和预期输入 (`expected_x`)。


class StructTestS(StructTestFunction):
    def f(self, x):
        return ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2
                + (x[2] - 0.5) ** 2 + (x[3] - 0.5) ** 2)

    g = None
    cons = wrap_constraints(g)


test_s = StructTestS(bounds=[(0, 2.0), ] * 4,
                     expected_fun=0.0,
                     expected_x=np.ones(4) - 0.5
                     )

定义了 `StructTestS` 类，继承自 `StructTestFunction`，`f` 方法计算一个特定的函数。`test_s` 是 `StructTestS` 的一个实例，设置了边界 (`bounds`)、预期函数值 (`expected_fun`) 和预期输入 (`expected_x`)。


class StructTestTable(StructTestFunction):
    def f(self, x):
        if x[0] == 3.0 and x[1] == 3.0:
            return 50
        else:
            return 100

    g = None
    cons = wrap_constraints(g)


test_table = StructTestTable(bounds=[(-10, 10), (-10, 10)],
                             expected_fun=[50],
                             expected_x=[3.0, 3.0])

定义了 `StructTestTable` 类，继承自 `StructTestFunction`，其中 `f` 方法根据输入 `x` 的值返回不同的函数值。`test_table` 是 `StructTestTable` 的一个实例，设置了边界 (`bounds`)、预期函数值 (`expected_fun`) 和预期输入 (`expected_x`)。


class StructTestInfeasible(StructTestFunction):
    """
    Test function with no feasible domain.
    """

    def f(self, x, *args):
        return x[0] ** 2 + x[1] ** 2

    def g1(x):
        return x[0] + x[1] - 1

    def g2(x):
        return -(x[0] + x[1] - 1)

定义了 `StructTestInfeasible` 类，继承自 `StructTestFunction`，描述了一个测试函数，其 `f` 方法计算一个特定的函数。类中还包含了两个额外的函数 `g1` 和 `g2`，分别计算给定输入 `x` 的值。
    # 定义函数 g3，计算并返回一个数值
    def g3(x):
        return -x[0] + x[1] - 1

    # 定义函数 g4，计算并返回一个数值，等价于 g3 的负数
    def g4(x):
        return -(-x[0] + x[1] - 1)

    # 创建一个包含 g1、g2、g3 和 g4 函数的元组
    g = (g1, g2, g3, g4)

    # 调用 wrap_constraints 函数，将 g 元组作为参数传递，并返回结果
    cons = wrap_constraints(g)
# 创建一个StructTestInfeasible对象，用于测试不可行的结构。
test_infeasible = StructTestInfeasible(bounds=[(2, 50), (-1, 1)],
                                       expected_fun=None,
                                       expected_x=None
                                       )

# 使用pytest.mark.skip装饰器标记，表明这个测试不会被运行。
@pytest.mark.skip("Not a test")
def run_test(test, args=(), test_atol=1e-5, n=100, iters=None,
             callback=None, minimizer_kwargs=None, options=None,
             sampling_method='sobol', workers=1):
    # 使用shgo进行全局优化测试，并返回优化结果。
    res = shgo(test.f, test.bounds, args=args, constraints=test.cons,
               n=n, iters=iters, callback=callback,
               minimizer_kwargs=minimizer_kwargs, options=options,
               sampling_method=sampling_method, workers=workers)

    # 打印优化结果
    print(f'res = {res}')
    # 记录优化结果到日志中
    logging.info(f'res = {res}')

    # 如果测试期望的最优解不为None，则使用np.testing.assert_allclose进行比较。
    if test.expected_x is not None:
        np.testing.assert_allclose(res.x, test.expected_x,
                                   rtol=test_atol,
                                   atol=test_atol)

    # （可选的测试）如果测试期望的函数值不为None，则使用np.testing.assert_allclose进行比较。
    if test.expected_fun is not None:
        np.testing.assert_allclose(res.fun,
                                   test.expected_fun,
                                   atol=test_atol)

    # （可选的测试）如果测试期望的最优解的下限不为None，则使用np.testing.assert_allclose进行比较。
    if test.expected_xl is not None:
        np.testing.assert_allclose(res.xl,
                                   test.expected_xl,
                                   atol=test_atol)

    # （可选的测试）如果测试期望的函数值的下限不为None，则使用np.testing.assert_allclose进行比较。
    if test.expected_funl is not None:
        np.testing.assert_allclose(res.funl,
                                   test.expected_funl,
                                   atol=test_atol)
    return


# Base test functions:
class TestShgoSobolTestFunctions:
    """
    Global optimisation tests with Sobol sampling:
    """

    # Sobol algorithm
    def test_f1_1_sobol(self):
        """Multivariate test function 1:
        x[0]**2 + x[1]**2 with bounds=[(-1, 6), (-1, 6)]"""
        run_test(test1_1)

    def test_f1_2_sobol(self):
        """Multivariate test function 1:
         x[0]**2 + x[1]**2 with bounds=[(0, 1), (0, 1)]"""
        run_test(test1_2)

    def test_f1_3_sobol(self):
        """Multivariate test function 1:
        x[0]**2 + x[1]**2 with bounds=[(None, None),(None, None)]"""
        options = {'disp': True}
        run_test(test1_3, options=options)

    def test_f2_1_sobol(self):
        """Univariate test function on
        f(x) = (x - 30) * sin(x) with bounds=[(0, 60)]"""
        run_test(test2_1)

    def test_f2_2_sobol(self):
        """Univariate test function on
        f(x) = (x - 30) * sin(x) bounds=[(0, 4.5)]"""
        run_test(test2_2)

    def test_f3_sobol(self):
        """NLP: Hock and Schittkowski problem 18"""
        run_test(test3_1)

    @pytest.mark.slow
    def test_f4_sobol(self):
        """NLP: (High dimensional) Hock and Schittkowski 11 problem (HS11)"""
        options = {'infty_constraints': False}
        # run_test(test4_1, n=990, options=options)
        run_test(test4_1, n=990 * 2, options=options)
    # 定义测试函数 `test_f5_1_sobol`，用于测试 `test5_1` 函数，使用 Sobol 序列生成输入数据
    def test_f5_1_sobol(self):
        """NLP: Eggholder, multimodal"""
        # 调用 `run_test` 函数运行 `test5_1`，参数 `n=60` 表示生成 60 组数据进行测试
        run_test(test5_1, n=60)

    # 定义测试函数 `test_f5_2_sobol`，用于测试 `test5_1` 函数，使用 Sobol 序列生成输入数据并设置迭代次数
    def test_f5_2_sobol(self):
        """NLP: Eggholder, multimodal"""
        # 调用 `run_test` 函数运行 `test5_1`，参数 `n=60` 表示生成 60 组数据进行测试，`iters=5` 表示进行 5 次迭代
        run_test(test5_1, n=60, iters=5)

        # 以下代码被注释掉，不会执行
        # def test_t911(self):
        #    """1D tabletop function"""
        #    run_test(test11_1)
class TestShgoSimplicialTestFunctions:
    """
    Global optimisation tests with Simplicial sampling:
    """

    def test_f1_1_simplicial(self):
        """Multivariate test function 1:
        x[0]**2 + x[1]**2 with bounds=[(-1, 6), (-1, 6)]"""
        # 运行名为 test1_1 的测试函数，使用 simplicial 抽样方法，一次运行一次
        run_test(test1_1, n=1, sampling_method='simplicial')

    def test_f1_2_simplicial(self):
        """Multivariate test function 1:
        x[0]**2 + x[1]**2 with bounds=[(0, 1), (0, 1)]"""
        # 运行名为 test1_2 的测试函数，使用 simplicial 抽样方法，一次运行一次
        run_test(test1_2, n=1, sampling_method='simplicial')

    def test_f1_3_simplicial(self):
        """Multivariate test function 1: x[0]**2 + x[1]**2
        with bounds=[(None, None),(None, None)]"""
        # 运行名为 test1_3 的测试函数，使用 simplicial 抽样方法，每次运行 5 次
        run_test(test1_3, n=5, sampling_method='simplicial')

    def test_f2_1_simplicial(self):
        """Univariate test function on
        f(x) = (x - 30) * sin(x) with bounds=[(0, 60)]"""
        # 运行名为 test2_1 的测试函数，使用 simplicial 抽样方法，每次运行 200 次，迭代 7 次，关闭每次迭代最小化选项
        options = {'minimize_every_iter': False}
        run_test(test2_1, n=200, iters=7, options=options,
                 sampling_method='simplicial')

    def test_f2_2_simplicial(self):
        """Univariate test function on
        f(x) = (x - 30) * sin(x) bounds=[(0, 4.5)]"""
        # 运行名为 test2_2 的测试函数，使用 simplicial 抽样方法，一次运行一次
        run_test(test2_2, n=1, sampling_method='simplicial')

    def test_f3_simplicial(self):
        """NLP: Hock and Schittkowski problem 18"""
        # 运行名为 test3_1 的测试函数，使用 simplicial 抽样方法，一次运行一次
        run_test(test3_1, n=1, sampling_method='simplicial')

    @pytest.mark.slow
    def test_f4_simplicial(self):
        """NLP: (High dimensional) Hock and Schittkowski 11 problem (HS11)"""
        # 运行名为 test4_1 的测试函数，使用 simplicial 抽样方法，一次运行一次，并标记为慢速测试（slow）
        run_test(test4_1, n=1, sampling_method='simplicial')

    def test_lj_symmetry_old(self):
        """LJ: Symmetry-constrained test function"""
        # 运行 testLJ 测试函数，传递参数 args=(6,)，使用 simplicial 抽样方法，每次运行 300 次，迭代 1 次，设置对称性参数为 True，显示结果
        options = {'symmetry': True,
                   'disp': True}
        args = (6,)  # Number of atoms
        run_test(testLJ, args=args, n=300,
                 options=options, iters=1,
                 sampling_method='simplicial')

    def test_f5_1_lj_symmetry(self):
        """LJ: Symmetry constrained test function"""
        # 运行 testLJ 测试函数，传递参数 args=(6,)，使用 simplicial 抽样方法，每次运行 300 次，迭代 1 次，设置对称性参数为 [0, 0, 0, 0, 0, 0]，显示结果
        options = {'symmetry': [0, ] * 6,
                   'disp': True}
        args = (6,)  # No. of atoms

        run_test(testLJ, args=args, n=300,
                 options=options, iters=1,
                 sampling_method='simplicial')

    def test_f5_2_cons_symmetry(self):
        """Symmetry constrained test function"""
        # 运行名为 test1_1 的测试函数，使用 simplicial 抽样方法，每次运行 200 次，迭代 1 次，设置对称性参数为 [0, 0]，显示结果
        options = {'symmetry': [0, 0],
                   'disp': True}

        run_test(test1_1, n=200,
                 options=options, iters=1,
                 sampling_method='simplicial')

    @pytest.mark.fail_slow(10)
    def test_f5_3_cons_symmetry(self):
        """Assymmetrically constrained test function"""
        # 运行名为 test_s 的测试函数，使用 simplicial 抽样方法，每次运行 10000 次，迭代 1 次，设置对称性参数为 [0, 0, 0, 0, 3]，显示结果
        options = {'symmetry': [0, 0, 0, 3],
                   'disp': True}

        run_test(test_s, n=10000,
                 options=options,
                 iters=1,
                 sampling_method='simplicial')

    @pytest.mark.skip("Not a test")
    # 定义一个测试函数，用于测试在完全对称的问题上找到最小值，基于 gh10429
    def test_f0_min_variance(self):
        # 给定变量 x 的平均值
        avg = 0.5  # Given average value of x
        
        # 定义约束条件：x 的平均值等于给定的 avg
        cons = {'type': 'eq', 'fun': lambda x: np.mean(x) - avg}
        
        # 使用 SHGO 算法，最小化在给定约束条件下 x 的方差
        res = shgo(np.var, bounds=6 * [(0, 1)], constraints=cons)
        
        # 断言最小化过程成功完成
        assert res.success
        
        # 断言最小化后的方差值接近于零，允许的绝对误差为 1e-15
        assert_allclose(res.fun, 0, atol=1e-15)
        
        # 断言最小化后的 x 值接近于 0.5
        assert_allclose(res.x, 0.5)

    # 标记此测试函数为跳过，因为它不是一个测试用例
    @pytest.mark.skip("Not a test")
    def test_f0_min_variance_1D(self):
        # 定义一个在完全对称的一维问题上找到最小值的测试函数，基于 gh10538
        
        # 定义目标函数 fun(x)
        def fun(x):
            return x * (x - 1.0) * (x - 0.5)
        
        # 定义变量 x 的取值范围
        bounds = [(0, 1)]
        
        # 使用 SHGO 算法，最小化目标函数 fun(x) 在给定的变量范围内
        res = shgo(fun, bounds=bounds)
        
        # 使用 minimize_scalar 函数，计算目标函数在同一范围内的最小值
        ref = minimize_scalar(fun, bounds=bounds[0])
        
        # 断言最小化过程成功完成
        assert res.success
        
        # 断言 SHGO 算法得到的最小化值与 minimize_scalar 函数得到的最小化值接近
        assert_allclose(res.fun, ref.fun)
        
        # 断言 SHGO 算法得到的最小化变量与 minimize_scalar 函数得到的最小化变量接近，相对误差为 1e-6
        assert_allclose(res.x, ref.x, rtol=1e-6)
# Argument test functions
class TestShgoArguments:
    # Test case for iterative simplicial sampling on TestFunction 1 (multivariate)
    def test_1_1_simpl_iter(self):
        """Iterative simplicial sampling on TestFunction 1 (multivariate)"""
        # Run test using `run_test` function with specific parameters
        run_test(test1_2, n=None, iters=2, sampling_method='simplicial')

    # Test case for iterative simplicial sampling on TestFunction 2 (univariate)
    def test_1_2_simpl_iter(self):
        """Iterative simplicial on TestFunction 2 (univariate)"""
        # Define additional options dictionary
        options = {'minimize_every_iter': False}
        # Run test using `run_test` function with extended parameters
        run_test(test2_1, n=None, iters=9, options=options,
                 sampling_method='simplicial')

    # Test case for iterative Sobol sampling on TestFunction 1 (multivariate)
    def test_2_1_sobol_iter(self):
        """Iterative Sobol sampling on TestFunction 1 (multivariate)"""
        # Run test using `run_test` function with specified parameters
        run_test(test1_2, n=None, iters=1, sampling_method='sobol')

    # Test case for iterative Sobol sampling on TestFunction 2 (univariate)
    def test_2_2_sobol_iter(self):
        """Iterative Sobol sampling on TestFunction 2 (univariate)"""
        # Run optimization using `shgo` function for TestFunction 2
        res = shgo(test2_1.f, test2_1.bounds, constraints=test2_1.cons,
                   n=None, iters=1, sampling_method='sobol')
        # Assert the closeness of the optimization result with expected values
        np.testing.assert_allclose(res.x, test2_1.expected_x, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(res.fun, test2_1.expected_fun, atol=1e-5)

    # Test case for iterative sampling on TestFunction 1 and 2 (multivariate and univariate)
    def test_3_1_disp_simplicial(self):
        """Iterative sampling on TestFunction 1 and 2  (multi and univariate)"""
        # Define a callback function for local minimization
        def callback_func(x):
            print("Local minimization callback test")

        # Iterate over test functions and run `shgo` with specified options and callbacks
        for test in [test1_1, test2_1]:
            shgo(test.f, test.bounds, iters=1,
                 sampling_method='simplicial',
                 callback=callback_func, options={'disp': True})
            shgo(test.f, test.bounds, n=1, sampling_method='simplicial',
                 callback=callback_func, options={'disp': True})

    # Test case for iterative sampling on TestFunction 1 and 2 (multivariate and univariate)
    def test_3_2_disp_sobol(self):
        """Iterative sampling on TestFunction 1 and 2 (multi and univariate)"""
        # Define a callback function for local minimization
        def callback_func(x):
            print("Local minimization callback test")

        # Iterate over test functions and run `shgo` with specified options and callbacks
        for test in [test1_1, test2_1]:
            shgo(test.f, test.bounds, iters=1, sampling_method='sobol',
                 callback=callback_func, options={'disp': True})

            shgo(test.f, test.bounds, n=1, sampling_method='simplicial',
                 callback=callback_func, options={'disp': True})

    # Test case for handling `args` in `shgo`, previously causing failures
    def test_args_gh14589(self):
        """Using `args` used to cause `shgo` to fail; see #14589, #15986,
        #16506"""
        # Run `shgo` with a function involving multiple arguments and assert results
        res = shgo(func=lambda x, y, z: x * z + y, bounds=[(0, 3)], args=(1, 2))
        # Define a reference optimization using `shgo` with a simpler function
        ref = shgo(func=lambda x: 2 * x + 1, bounds=[(0, 3)])
        # Assert the closeness of the optimization results
        assert_allclose(res.fun, ref.fun)
        assert_allclose(res.x, ref.x)

    # Test case for testing known function minima stopping criteria
    @pytest.mark.slow
    def test_4_1_known_f_min(self):
        """Test known function minima stopping criteria"""
        # Specify options for the test, including a known function value and tolerance
        options = {'f_min': test4_1.expected_fun,
                   'f_tol': 1e-6,
                   'minimize_every_iter': True}
        # Run test using `run_test` function with extended parameters
        # TODO: Make default n higher for faster tests
        run_test(test4_1, n=None, test_atol=1e-5, options=options,
                 sampling_method='simplicial')
    def test_4_2_known_f_min(self):
        """Test Global mode limiting local evaluations"""
        options = {  # 指定已知函数值
            'f_min': test4_1.expected_fun,  # 设置最小函数值为预期函数值
            'f_tol': 1e-6,  # 设置函数值容差
            'minimize_every_iter': True,  # 每次迭代都进行局部最小化
            'local_iter': 1}  # 指定局部迭代次数为1

        run_test(test4_1, n=None, test_atol=1e-5, options=options,
                 sampling_method='simplicial')

    def test_4_4_known_f_min(self):
        """Test Global mode limiting local evaluations for 1D funcs"""
        options = {  # 指定已知函数值
            'f_min': test2_1.expected_fun,  # 设置最小函数值为预期函数值
            'f_tol': 1e-6,  # 设置函数值容差
            'minimize_every_iter': True,  # 每次迭代都进行局部最小化
            'local_iter': 1,  # 指定局部迭代次数为1
            'infty_constraints': False}  # 禁用无限约束条件

        res = shgo(test2_1.f, test2_1.bounds, constraints=test2_1.cons,
                   n=None, iters=None, options=options,
                   sampling_method='sobol')
        np.testing.assert_allclose(res.x, test2_1.expected_x, rtol=1e-5, atol=1e-5)

    def test_5_1_simplicial_argless(self):
        """Test Default simplicial sampling settings on TestFunction 1"""
        res = shgo(test1_1.f, test1_1.bounds, constraints=test1_1.cons)
        np.testing.assert_allclose(res.x, test1_1.expected_x, rtol=1e-5, atol=1e-5)

    def test_5_2_sobol_argless(self):
        """Test Default sobol sampling settings on TestFunction 1"""
        res = shgo(test1_1.f, test1_1.bounds, constraints=test1_1.cons,
                   sampling_method='sobol')
        np.testing.assert_allclose(res.x, test1_1.expected_x, rtol=1e-5, atol=1e-5)

    def test_6_1_simplicial_max_iter(self):
        """Test that maximum iteration option works on TestFunction 3"""
        options = {'max_iter': 2}  # 设置最大迭代次数为2
        res = shgo(test3_1.f, test3_1.bounds, constraints=test3_1.cons,
                   options=options, sampling_method='simplicial')
        np.testing.assert_allclose(res.x, test3_1.expected_x, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(res.fun, test3_1.expected_fun, atol=1e-5)

    def test_6_2_simplicial_min_iter(self):
        """Test that maximum iteration option works on TestFunction 3"""
        options = {'min_iter': 2}  # 设置最小迭代次数为2
        res = shgo(test3_1.f, test3_1.bounds, constraints=test3_1.cons,
                   options=options, sampling_method='simplicial')
        np.testing.assert_allclose(res.x, test3_1.expected_x, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(res.fun, test3_1.expected_fun, atol=1e-5)
    def test_7_1_minkwargs(self):
        """Test the minimizer_kwargs arguments for solvers with constraints"""
        # 遍历测试使用的求解器列表
        for solver in ['COBYLA', 'COBYQA', 'SLSQP']:
            # 注意：将全局约束传递给 SLSQP 的测试在其他单元测试中进行，通常会运行 test4_1
            # 设置最小化器关键字参数字典，包括求解方法和约束条件
            minimizer_kwargs = {'method': solver,
                                'constraints': test3_1.cons}
            # 运行测试，指定迭代次数、测试精度、最小化器关键字参数和采样方法
            run_test(test3_1, n=100, test_atol=1e-3,
                     minimizer_kwargs=minimizer_kwargs,
                     sampling_method='sobol')

    def test_7_2_minkwargs(self):
        """Test the minimizer_kwargs default inits"""
        # 设置默认的最小化器关键字参数字典，只包括 ftol 参数
        minimizer_kwargs = {'ftol': 1e-5}
        options = {'disp': True}  # 用于覆盖率分析目的
        # 使用 SHGO 方法运行测试，指定目标函数、边界、约束条件和最小化器关键字参数
        SHGO(test3_1.f, test3_1.bounds, constraints=test3_1.cons[0],
             minimizer_kwargs=minimizer_kwargs, options=options)

    def test_7_3_minkwargs(self):
        """Test minimizer_kwargs arguments for solvers without constraints"""
        # 遍历测试使用的求解器列表，这些求解器不带约束条件
        for solver in ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG',
                       'L-BFGS-B', 'TNC', 'dogleg', 'trust-ncg', 'trust-exact',
                       'trust-krylov']:
            # 定义目标函数的雅可比矩阵和黑塞矩阵
            def jac(x):
                return np.array([2 * x[0], 2 * x[1]]).T

            def hess(x):
                return np.array([[2, 0], [0, 2]])

            # 设置最小化器关键字参数字典，包括求解方法、雅可比矩阵和黑塞矩阵
            minimizer_kwargs = {'method': solver,
                                'jac': jac,
                                'hess': hess}
            # 记录求解器信息到日志
            logging.info(f"Solver = {solver}")
            logging.info("=" * 100)
            # 运行测试，指定迭代次数、测试精度、最小化器关键字参数和采样方法
            run_test(test1_1, n=100, test_atol=1e-3,
                     minimizer_kwargs=minimizer_kwargs,
                     sampling_method='sobol')

    def test_8_homology_group_diff(self):
        # 设置选项字典，包括最小化格点数和每次迭代都最小化
        options = {'minhgrd': 1,
                   'minimize_every_iter': True}
        # 运行测试，指定测试对象、迭代次数和选项字典，使用 simplicial 采样方法
        run_test(test1_1, n=None, iters=None, options=options,
                 sampling_method='simplicial')

    def test_9_cons_g(self):
        """Test single function constraint passing"""
        # 使用 SHGO 方法运行测试，指定目标函数、边界和单一约束条件
        SHGO(test3_1.f, test3_1.bounds, constraints=test3_1.cons[0])

    @pytest.mark.xfail(IS_PYPY and sys.platform == 'win32',
            reason="Failing and fix in PyPy not planned (see gh-18632)")
    def test_10_finite_time(self):
        """Test single function constraint passing"""
        # 设置选项字典，包括最大运行时间
        options = {'maxtime': 1e-15}

        # 定义一个函数，使其运行时间为 1e-14 秒
        def f(x):
            time.sleep(1e-14)
            return 0.0

        # 使用 SHGO 方法运行测试，指定目标函数、边界、迭代次数和选项字典
        res = shgo(f, test1_1.bounds, iters=5, options=options)
        # 断言只运行了 1 次迭代而不是 5 次
        assert res.nit == 1
    def test_11_f_min_0(self):
        """Test to cover the case where f_lowest == 0"""
        # 定义测试选项，设置 f_min 为 0.0，并显示详细信息
        options = {'f_min': 0.0,
                   'disp': True}
        # 运行 shgo 函数进行优化，使用 Sobol 抽样方法
        res = shgo(test1_2.f, test1_2.bounds, n=10, iters=None,
                   options=options, sampling_method='sobol')
        # 断言优化结果的第一个和第二个变量的最优解为 0
        np.testing.assert_equal(0, res.x[0])
        np.testing.assert_equal(0, res.x[1])

    # @nottest
    @pytest.mark.skip(reason="no way of currently testing this")
    def test_12_sobol_inf_cons(self):
        """Test to cover the case where f_lowest == 0"""
        # TODO: This test doesn't cover anything new, it is unknown what the
        # original test was intended for as it was never complete. Delete or
        # replace in the future.
        # 定义测试选项，设置最大运行时间为极小值，f_min 为 0.0
        options = {'maxtime': 1e-15,
                   'f_min': 0.0}
        # 运行 shgo 函数进行优化，使用 Sobol 抽样方法
        res = shgo(test1_2.f, test1_2.bounds, n=1, iters=None,
                   options=options, sampling_method='sobol')
        # 断言优化结果的最优函数值为 0.0
        np.testing.assert_equal(0.0, res.fun)

    def test_13_high_sobol(self):
        """Test init of high-dimensional sobol sequences"""

        def f(x):
            return 0

        # 定义参数范围，创建 SHGO 对象，使用 Sobol 抽样方法
        bounds = [(None, None), ] * 41
        SHGOc = SHGO(f, bounds, sampling_method='sobol')
        # 调用 SHGO 对象的抽样函数，生成 Sobol 序列点集
        SHGOc.sampling_function(2, 50)

    def test_14_local_iter(self):
        """Test limited local iterations for a pseudo-global mode"""
        # 定义选项，设置局部迭代次数为 4
        options = {'local_iter': 4}
        # 运行测试函数，使用指定选项和默认的 Sobol 抽样方法
        run_test(test5_1, n=60, options=options)

    def test_15_min_every_iter(self):
        """Test minimize every iter options and cover function cache"""
        # 定义选项，设置每次迭代都最小化函数
        options = {'minimize_every_iter': True}
        # 运行测试函数，设置迭代次数为 7，使用 Sobol 抽样方法
        run_test(test1_1, n=1, iters=7, options=options,
                 sampling_method='sobol')

    def test_16_disp_bounds_minimizer(self, capsys):
        """Test disp=True with minimizers that do not support bounds """
        # 定义选项，设置显示详细信息
        options = {'disp': True}
        # 定义最小化器参数，选择 Nelder-Mead 方法
        minimizer_kwargs = {'method': 'nelder-mead'}
        # 运行测试函数，使用简单形式的抽样方法，设置选项和最小化器参数
        run_test(test1_2, sampling_method='simplicial',
                 options=options, minimizer_kwargs=minimizer_kwargs)

    def test_17_custom_sampling(self):
        """Test the functionality to add custom sampling methods to shgo"""

        def sample(n, d):
            return np.random.uniform(size=(n, d))

        # 运行测试函数，设置抽样方法为自定义的 sample 函数，样本数为 30
        run_test(test1_1, n=30, sampling_method=sample)
    def test_20_constrained_args(self):
        """Test that constraints can be passed to arguments"""

        # 定义一个函数 eggholder，计算给定参数 x 的特定函数值
        def eggholder(x):
            return (
                -(x[1] + 47.0)*np.sin(np.sqrt(abs(x[0] / 2.0 + (x[1] + 47.0))))
                - x[0]*np.sin(np.sqrt(abs(x[0] - (x[1] + 47.0))))
            )

        # 定义一个线性函数 f，计算给定参数 x 的线性组合
        def f(x):  # (cattle-feed)
            return 24.55 * x[0] + 26.75 * x[1] + 39 * x[2] + 40.50 * x[3]

        # 定义变量 bounds，表示参数 x 的上下界
        bounds = [(0, 1.0), ] * 4

        # 定义一个函数 g1_modified，计算带有参数 i 的线性约束条件
        def g1_modified(x, i):
            return i * 2.3 * x[0] + i * 5.6 * x[1] + 11.1 * x[2] + 1.3 * x[3] - 5  # >=0

        # 定义一个函数 g2，计算参数 x 的非线性约束条件
        def g2(x):
            return (
                12*x[0] + 11.9*x[1] + 41.8*x[2] + 52.1*x[3] - 21
                - 1.645*np.sqrt(
                    0.28*x[0]**2 + 0.19*x[1]**2 + 20.5*x[2]**2 + 0.62*x[3]**2
                )
            )  # >=0

        # 定义一个函数 h1，计算参数 x 的等式约束条件
        def h1(x):
            return x[0] + x[1] + x[2] + x[3] - 1  # == 0

        # 定义约束条件 cons，包括一个带参数的不等式约束 g1_modified，一个不等式约束 g2 和一个等式约束 h1
        cons = ({'type': 'ineq', 'fun': g1_modified, "args": (0,)},
                {'type': 'ineq', 'fun': g2},
                {'type': 'eq', 'fun': h1})

        # 使用 SHGO 算法求解优化问题，包括给定的上下界 bounds 和约束条件 cons，使用默认的采样方法
        shgo(f, bounds, n=300, iters=1, constraints=cons)

        # 使用 SHGO 算法求解优化问题，包括给定的上下界 bounds、约束条件 cons 和采样方法 'sobol'
        shgo(f, bounds, n=300, iters=1, constraints=cons,
             sampling_method='sobol')
    def test_21_1_jac_true(self):
        """Test that shgo can handle objective functions that return the
        gradient alongside the objective value. Fixes gh-13547"""
        
        # 定义一个测试函数，返回目标函数的值和梯度
        def func(x):
            return np.sum(np.power(x, 2)), 2 * x
        
        # 使用 shgo 函数进行优化，指定参数和边界
        shgo(
            func,
            bounds=[[-1, 1], [1, 2]],
            n=100, iters=5,
            sampling_method="sobol",
            minimizer_kwargs={'method': 'SLSQP', 'jac': True}
        )

        # 重新定义函数，仅返回目标函数的值
        def func(x):
            return np.sum(x ** 2), 2 * x
        
        # 定义新的边界
        bounds = [[-1, 1], [1, 2], [-1, 1], [1, 2], [0, 3]]
        
        # 使用 shgo 函数进行优化，包括梯度信息，使用 'SLSQP' 方法
        res = shgo(func, bounds=bounds, sampling_method="sobol",
                   minimizer_kwargs={'method': 'SLSQP', 'jac': True})
        
        # 对结果进行参考优化，使用 minimize 函数，同时指定梯度
        ref = minimize(func, x0=[1, 1, 1, 1, 1], bounds=bounds,
                       jac=True)
        
        # 断言优化成功
        assert res.success
        # 断言优化结果的函数值与参考结果的函数值非常接近
        assert_allclose(res.fun, ref.fun)
        # 断言优化结果的变量值与参考结果的变量值非常接近，使用相对公差 1e-15
        assert_allclose(res.x, ref.x, atol=1e-15)

    @pytest.mark.parametrize('derivative', ['jac', 'hess', 'hessp'])
    def test_21_2_derivative_options(self, derivative):
        """shgo used to raise an error when passing `options` with 'jac'
        # see gh-12963. check that this is resolved
        """
        
        # 定义目标函数
        def objective(x):
            return 3 * x[0] * x[0] + 2 * x[0] + 5
        
        # 定义梯度函数
        def gradient(x):
            return 6 * x[0] + 2
        
        # 定义 Hessian 函数
        def hess(x):
            return 6
        
        # 定义 Hessian 向量乘法函数
        def hessp(x, p):
            return 6 * p
        
        # 将所有导数函数组成字典
        derivative_funcs = {'jac': gradient, 'hess': hess, 'hessp': hessp}
        
        # 根据传入的导数类型选择对应的导数函数，构建选项字典
        options = {derivative: derivative_funcs[derivative]}
        
        # 设置最小化器参数
        minimizer_kwargs = {'method': 'trust-constr'}
        
        # 设置边界
        bounds = [(-100, 100)]
        
        # 使用 shgo 函数进行优化，传入选项字典和最小化器参数
        res = shgo(objective, bounds, minimizer_kwargs=minimizer_kwargs,
                   options=options)
        
        # 使用 minimize 函数进行参考优化，传入初始点和边界，同时传入选项字典
        ref = minimize(objective, x0=[0], bounds=bounds, **minimizer_kwargs,
                       **options)
        
        # 断言优化成功
        assert res.success
        # 断言优化结果的函数值与参考结果的函数值非常接近
        np.testing.assert_allclose(res.fun, ref.fun)
        # 断言优化结果的变量值与参考结果的变量值非常接近
        np.testing.assert_allclose(res.x, ref.x)

    def test_21_3_hess_options_rosen(self):
        """Ensure the Hessian gets passed correctly to the local minimizer
        routine. Previous report gh-14533.
        """
        
        # 定义边界
        bounds = [(0, 1.6), (0, 1.6), (0, 1.4), (0, 1.4), (0, 1.4)]
        
        # 定义选项字典，包括 Jacobian 函数和 Hessian 函数
        options = {'jac': rosen_der, 'hess': rosen_hess}
        
        # 设置最小化器参数
        minimizer_kwargs = {'method': 'Newton-CG'}
        
        # 使用 shgo 函数进行优化，传入选项字典和最小化器参数
        res = shgo(rosen, bounds, minimizer_kwargs=minimizer_kwargs,
                   options=options)
        
        # 使用 minimize 函数进行参考优化，传入初始点和方法，同时传入选项字典
        ref = minimize(rosen, np.zeros(5), method='Newton-CG',
                       **options)
        
        # 断言优化成功
        assert res.success
        # 断言优化结果的函数值与参考结果的函数值非常接近，使用相对公差 1e-15
        assert_allclose(res.fun, ref.fun)
        # 断言优化结果的变量值与参考结果的变量值非常接近，使用相对公差 1e-15
        assert_allclose(res.x, ref.x, atol=1e-15)
    def test_21_arg_tuple_sobol(self):
        """定义一个测试函数，用于测试 `shgo` 在使用 Sobol 抽样时传递参数 `args` 是否会导致错误。
        # 参见 gh-12114。检查此问题是否已解决"""

        def fun(x, k):
            return x[0] ** k
        # 定义一个简单的函数 fun，接受一个向量 x 和参数 k，计算 x[0] 的 k 次方

        constraints = ({'type': 'ineq', 'fun': lambda x: x[0] - 1})
        # 定义一个约束条件，要求 x[0] 大于等于 1

        bounds = [(0, 10)]
        # 定义变量的边界条件，x[0] 的取值范围在 [0, 10]

        res = shgo(fun, bounds, args=(1,), constraints=constraints,
                   sampling_method='sobol')
        # 使用 shgo 函数进行全局优化，传入函数 fun、边界 bounds、参数 args=(1,)、约束条件 constraints 和抽样方法 'sobol'

        ref = minimize(fun, np.zeros(1), bounds=bounds, args=(1,),
                       constraints=constraints)
        # 使用 minimize 函数进行局部优化，传入函数 fun、初始点 np.zeros(1)、边界 bounds、参数 args=(1,) 和约束条件 constraints

        assert res.success
        # 断言全局优化结果成功

        assert_allclose(res.fun, ref.fun)
        # 断言全局优化结果的目标函数值与局部优化结果的目标函数值近似相等

        assert_allclose(res.x, ref.x)
        # 断言全局优化结果的最优点与局部优化结果的最优点近似相等
# Failure test functions
class TestShgoFailures:
    # 测试最大迭代次数不足时的失败情况
    def test_1_maxiter(self):
        """Test failure on insufficient iterations"""
        options = {'maxiter': 2}
        # 使用 SHGO 算法测试函数 test4_1.f 在指定边界 test4_1.bounds 上的最优化问题，
        # 使用 Sobol 方法进行采样，迭代次数为 2
        res = shgo(test4_1.f, test4_1.bounds, n=2, iters=None,
                   options=options, sampling_method='sobol')

        # 断言结果 res 的成功标志为 False
        np.testing.assert_equal(False, res.success)
        # 断言结果 res 的总评估次数为 4
        # np.testing.assert_equal(4, res.nfev)
        np.testing.assert_equal(4, res.tnev)

    # 测试未知采样方法时的失败情况
    def test_2_sampling(self):
        """Rejection of unknown sampling method"""
        # 断言使用 SHGO 算法调用 test1_1.f 在 test1_1.bounds 上时，
        # 使用了未知的采样方法 'not_Sobol'，会引发 ValueError 异常
        assert_raises(ValueError, shgo, test1_1.f, test1_1.bounds,
                      sampling_method='not_Sobol')

    # 测试在没有找到最小化器的情况下达到最大指定函数评估次数后算法停止的情况
    def test_3_1_no_min_pool_sobol(self):
        """Check that the routine stops when no minimiser is found
           after maximum specified function evaluations"""
        options = {'maxfev': 10,
                   # 'maxev': 10,
                   'disp': True}
        # 使用 SHGO 算法测试函数 test_table.f 在 test_table.bounds 上的最优化问题，
        # 使用 Sobol 方法进行采样，最大函数评估次数为 10
        res = shgo(test_table.f, test_table.bounds, n=3, options=options,
                   sampling_method='sobol')
        # 断言结果 res 的成功标志为 False
        np.testing.assert_equal(False, res.success)
        # 断言结果 res 的总函数评估次数为 12
        np.testing.assert_equal(12, res.nfev)

    # 测试在没有找到最小化器的情况下达到最大指定采样评估次数后算法停止的情况
    def test_3_2_no_min_pool_simplicial(self):
        """Check that the routine stops when no minimiser is found
           after maximum specified sampling evaluations"""
        options = {'maxev': 10,
                   'disp': True}
        # 使用 SHGO 算法测试函数 test_table.f 在 test_table.bounds 上的最优化问题，
        # 使用 simplicial 方法进行采样，最大采样评估次数为 10
        res = shgo(test_table.f, test_table.bounds, n=3, options=options,
                   sampling_method='simplicial')
        # 断言结果 res 的成功标志为 False
        np.testing.assert_equal(False, res.success)

    # 测试指定边界的错误情况（上界小于下界）
    def test_4_1_bound_err(self):
        """Specified bounds ub > lb"""
        bounds = [(6, 3), (3, 5)]
        # 断言使用 SHGO 算法调用 test1_1.f 在指定错误边界 bounds 上时，会引发 ValueError 异常
        assert_raises(ValueError, shgo, test1_1.f, bounds)

    # 测试指定边界格式错误的情况（应为 (lb, ub) 形式）
    def test_4_2_bound_err(self):
        """Specified bounds are of the form (lb, ub)"""
        bounds = [(3, 5, 5), (3, 5)]
        # 断言使用 SHGO 算法调用 test1_1.f 在指定错误边界 bounds 上时，会引发 ValueError 异常
        assert_raises(ValueError, shgo, test1_1.f, bounds)

    # 测试在无法解决问题的情况下，达到最大指定采样评估次数后算法停止的情况（使用 infty_constraints 选项）
    def test_5_1_1_infeasible_sobol(self):
        """Ensures the algorithm terminates on infeasible problems
           after maxev is exceeded. Use infty constraints option"""
        options = {'maxev': 100,
                   'disp': True}
        # 使用 SHGO 算法测试函数 test_infeasible.f 在 test_infeasible.bounds 上的最优化问题，
        # 使用 Sobol 方法进行采样，最大采样评估次数为 100，使用了 infty_constraints 选项
        res = shgo(test_infeasible.f, test_infeasible.bounds,
                   constraints=test_infeasible.cons, n=100, options=options,
                   sampling_method='sobol')

        # 断言结果 res 的成功标志为 False
        np.testing.assert_equal(False, res.success)

    # 测试在无法解决问题的情况下，达到最大指定采样评估次数后算法停止的情况（不使用 infty_constraints 选项）
    def test_5_1_2_infeasible_sobol(self):
        """Ensures the algorithm terminates on infeasible problems
           after maxev is exceeded. Do not use infty constraints option"""
        options = {'maxev': 100,
                   'disp': True,
                   'infty_constraints': False}
        # 使用 SHGO 算法测试函数 test_infeasible.f 在 test_infeasible.bounds 上的最优化问题，
        # 使用 Sobol 方法进行采样，最大采样评估次数为 100，不使用 infty_constraints 选项
        res = shgo(test_infeasible.f, test_infeasible.bounds,
                   constraints=test_infeasible.cons, n=100, options=options,
                   sampling_method='sobol')

        # 断言结果 res 的成功标志为 False
        np.testing.assert_equal(False, res.success)
    def test_5_2_infeasible_simplicial(self):
        """Ensures the algorithm terminates on infeasible problems
           after maxev is exceeded."""
        # 设置算法参数选项
        options = {'maxev': 1000,
                   'disp': False}

        # 运行 SHGO 算法以处理不可行问题，使用简单抽样方法
        res = shgo(test_infeasible.f, test_infeasible.bounds,
                   constraints=test_infeasible.cons, n=100, options=options,
                   sampling_method='simplicial')

        # 断言算法运行未成功
        np.testing.assert_equal(False, res.success)

    def test_6_1_lower_known_f_min(self):
        """Test Global mode limiting local evaluations with f* too high"""
        # 指定已知函数值，设置其他选项
        options = {  # Specify known function value
            'f_min': test2_1.expected_fun + 2.0,
            'f_tol': 1e-6,
            # Specify number of local iterations to perform+
            'minimize_every_iter': True,
            'local_iter': 1,
            'infty_constraints': False}
        args = (test2_1.f, test2_1.bounds)
        kwargs = {'constraints': test2_1.cons,
                  'n': None,
                  'iters': None,
                  'options': options,
                  'sampling_method': 'sobol'
                  }
        # 验证函数调用时引发 UserWarning
        warns(UserWarning, shgo, *args, **kwargs)

    def test(self):
        from scipy.optimize import rosen, shgo
        bounds = [(0, 2), (0, 2), (0, 2), (0, 2), (0, 2)]

        def fun(x):
            fun.nfev += 1
            return rosen(x)

        fun.nfev = 0

        # 使用 SHGO 算法优化 Rosenbrock 函数
        result = shgo(fun, bounds)
        # 输出最优解、最优函数值及函数评估次数
        print(result.x, result.fun, fun.nfev)  # 50
# 定义一个测试类 TestShgoReturns，用于测试 shgo 函数的返回值
class TestShgoReturns:
    
    # 测试简单采样方法下的 nfev 参数
    def test_1_nfev_simplicial(self):
        # 设定变量边界
        bounds = [(0, 2), (0, 2), (0, 2), (0, 2), (0, 2)]

        # 定义被优化的目标函数，并计数函数调用次数
        def fun(x):
            fun.nfev += 1
            return rosen(x)

        # 初始化函数调用计数器
        fun.nfev = 0

        # 使用 shgo 函数优化目标函数
        result = shgo(fun, bounds)
        
        # 断言函数调用次数与优化结果中的 nfev 参数相等
        np.testing.assert_equal(fun.nfev, result.nfev)

    # 测试 Sobol 采样方法下的 nfev 参数
    def test_1_nfev_sobol(self):
        # 设定变量边界
        bounds = [(0, 2), (0, 2), (0, 2), (0, 2), (0, 2)]

        # 定义被优化的目标函数，并计数函数调用次数
        def fun(x):
            fun.nfev += 1
            return rosen(x)

        # 初始化函数调用计数器
        fun.nfev = 0

        # 使用 shgo 函数优化目标函数，指定使用 Sobol 采样方法
        result = shgo(fun, bounds, sampling_method='sobol')
        
        # 断言函数调用次数与优化结果中的 nfev 参数相等
        np.testing.assert_equal(fun.nfev, result.nfev)


# 测试向量约束
def test_vector_constraint():
    # 定义一个简单的二次约束函数
    def quad(x):
        x = np.asarray(x)
        return [np.sum(x ** 2)]

    # 创建一个非线性约束对象
    nlc = NonlinearConstraint(quad, [2.2], [3])
    
    # 将新约束转换为旧约束形式
    oldc = new_constraint_to_old(nlc, np.array([1.0, 1.0]))

    # 使用 shgo 函数优化目标函数，应用 Sobol 采样方法和转换后的约束
    res = shgo(rosen, [(0, 10), (0, 10)], constraints=oldc, sampling_method='sobol')
    
    # 断言优化结果满足约束条件
    assert np.all(np.sum((res.x)**2) >= 2.2)
    assert np.all(np.sum((res.x) ** 2) <= 3.0)
    assert res.success


# 测试信任约束方法
@pytest.mark.filterwarnings("ignore:delta_grad")
def test_trust_constr():
    # 定义一个简单的二次约束函数
    def quad(x):
        x = np.asarray(x)
        return [np.sum(x ** 2)]

    # 创建一个非线性约束对象
    nlc = NonlinearConstraint(quad, [2.6], [3])
    
    # 设定优化参数字典
    minimizer_kwargs = {'method': 'trust-constr'}
    
    # 使用 shgo 函数优化目标函数，应用 Sobol 采样方法和信任约束方法
    res = shgo(
        rosen,
        [(0, 10), (0, 10)],
        constraints=nlc,
        sampling_method='sobol',
        minimizer_kwargs=minimizer_kwargs
    )
    
    # 断言优化结果满足约束条件
    assert np.all(np.sum((res.x)**2) >= 2.6)
    assert np.all(np.sum((res.x) ** 2) <= 3.0)
    assert res.success


# 测试等式约束
def test_equality_constraints():
    # 定义变量边界，约束概率值在 0 和 1 之间
    bounds = [(0.9, 4.0)] * 2  

    # 定义一个错误的约束函数
    def faulty(x):
        return x[0] + x[1]

    # 创建一个非线性约束对象
    nlc = NonlinearConstraint(faulty, 3.9, 3.9)
    
    # 使用 shgo 函数优化目标函数，应用约束条件
    res = shgo(rosen, bounds=bounds, constraints=nlc)
    
    # 断言优化结果满足约束条件
    assert_allclose(np.sum(res.x), 3.9)

    # 重新定义约束函数
    def faulty(x):
        return x[0] + x[1] - 3.9

    # 使用字典形式定义等式约束
    constraints = {'type': 'eq', 'fun': faulty}
    
    # 使用 shgo 函数优化目标函数，应用约束条件
    res = shgo(rosen, bounds=bounds, constraints=constraints)
    
    # 断言优化结果满足约束条件
    assert_allclose(np.sum(res.x), 3.9)

    # 重新设定变量边界，使得变量和为1
    bounds = [(0, 1.0)] * 4  

    # 重新定义约束函数
    def faulty(x):
        return x[0] + x[1] + x[2] + x[3] - 1

    # 使用字典形式定义等式约束
    constraints = {'type': 'eq', 'fun': faulty}
    
    # 使用 shgo 函数优化目标函数，应用 Sobol 采样方法和约束条件
    res = shgo(
        lambda x: - np.prod(x),
        bounds=bounds,
        constraints=constraints,
        sampling_method='sobol'
    )
    
    # 断言优化结果满足约束条件
    assert_allclose(np.sum(res.x), 1.0)


# 测试 gh16971 情况
def test_gh16971():
    # 定义一个简单的约束函数
    def cons(x):
        return np.sum(x**2) - 0

    # 创建一个约束字典
    c = {'fun': cons, 'type': 'ineq'}
    
    # 设定优化参数字典
    minimizer_kwargs = {
        'method': 'COBYLA',
        'options': {'rhobeg': 5, 'tol': 5e-1, 'catol': 0.05}
    }
    # 使用 SHGO 算法进行优化，目标函数为 rosen，变量范围为 (0, 10) × 2
    s = SHGO(
        rosen, [(0, 10)]*2, constraints=c, minimizer_kwargs=minimizer_kwargs
    )

    # 断言优化器的方法是 'cobyla'
    assert s.minimizer_kwargs['method'].lower() == 'cobyla'
    
    # 断言优化器的参数选项中 'catol' 的值为 0.05
    assert s.minimizer_kwargs['options']['catol'] == 0.05
```